#!/usr/bin/env python3
"""Map-Reduce 剧情归约：把带时间戳的 script.jsonl 压成结构化剧情图。

Map 阶段按分 P + 每 300-500 行切块，并发调用 OpenAI 兼容接口生成场景卡与逐句译文；
覆盖率审计对高频角色 / 行区间做红灯告警，非 mock 模式自动定向补写一次；
Reduce 阶段汇总三幕骨架、Mermaid 人物关系图、角色档案与审计报告。

用法：
    python -m bili_story_pipe.summarize_mr --work-dir . --mock
    python -m bili_story_pipe.summarize_mr --work-dir . --base-url URL --api-key KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

DEFAULT_MODEL = "geminiweb2api/gemini-3.6-flash"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TITLE = "Neko-nin exHeart SPIN! 2"
DEFAULT_BVID = "BV13JSCBAEDC"

# 无外部 glossary.json 时的兜底角色译名 / 口癖表，保证 mock 与真实模式译名一致。
BUILTIN_GLOSSARY: dict[str, Any] = {
    "characters": {
        "藤花": "藤花",
        "北条": "北条",
        "由良": "由良",
        "爱丽丝": "爱丽丝",
        "アリス": "爱丽丝",
        "蛭子": "蛭子",
        "エビス": "蛭子",
        "千花": "千花",
        "莉兹": "莉兹",
    },
    "verbal_tics": {
        "にゃー": "喵呜",
        "にゃん": "喵",
        "ですの": "呢",
        "わたくし": "人家",
    },
}

# mock 译文常用短语，仅为演示译文风格，非真实翻译模型。
MOCK_PHRASES: dict[str, str] = {
    "神話の時代": "神话时代",
    "神は人と": "神与人",
    "猫忍": "猫忍",
    "忍者": "忍者",
    "今日": "今天",
    "ありがとう": "谢谢",
    "そうですか": "是吗",
    "わかった": "明白了",
    "行こう": "走吧",
    "少し": "稍微",
}

# 覆盖率审计降噪：这些是旁白/机翻/未知元标签，不是真实角色，
# 不能进 Top-15 高频角色统计，否则会触发无意义的「漏剧情」补写。
NON_CHARACTER_TAGS: set[str] = {
    "", "?", "??", "???", "？？？", "不明", "旁白", "ナレーション",
    "narration", "narrator", "unknown", "モブ", "その他", "_other",
}


def is_real_character(name: object) -> bool:
    """是否为应纳入覆盖率审计/角色档案的真实角色名。"""
    s = str(name).strip()
    return bool(s) and s not in NON_CHARACTER_TAGS and s not in {"-", "ー", "―"}


# --------------------------------------------------------------------------- #
# 角色名规范化（OCR 变体 / 简繁 / 日中文归并）
# --------------------------------------------------------------------------- #
# 规范名用日文原形（台词与场景卡均为日文说话人），变体覆盖简繁、日中文译名、
# 假名/汉字、OCR 形近字与「姓 / 名」单独出现的情况。
ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("サトミ", ("サトミ", "サトちゃん", "さとみ", "大忍サトミ", "里見サトミ", "里见サトミ", "サト美")),
    ("北条春希", ("北条春希", "北条", "春希", "北條春希", "北条 春希", "北条　春希")),
    ("陽葵", ("陽葵", "阳葵", "雑賀陽葵", "雑賀 陽葵", "杂贺阳葵", "杂贺 阳葵")),
    ("結衣", ("結衣", "结衣", "甲賀結衣", "甲賀 結衣", "甲贺结衣")),
    ("千花", ("千花", "チカ", "ちか")),
    ("ナナコ", ("ナナコ", "奈奈子", "巻菱ナナコ", "巻菱 ナナコ", "卷菱奈奈子")),
    ("アイリス", ("アイリス", "アリス", "爱丽丝", "愛麗絲", "艾丽丝")),
    ("藤花", ("藤花", "華花", "华花", "藤華花")),
    ("ゆら", ("ゆら", "由良", "ユラ", "風魔ゆら", "風魔 ゆら", "风魔由良")),
    ("たま", ("たま", "玉", "タマ")),
    ("伊緒", ("伊緒", "伊绪", "伊縮", "伊织", "伊織", "イオ", "甲賀伊緒", "甲賀 伊緒", "甲贺伊绪")),
    ("菊丸", ("菊丸", "キクマル", "雑賀菊丸", "雑賀 菊丸", "杂贺菊丸")),
    ("永夜", ("永夜", "猫坂永夜", "猫坂 永夜")),
    ("彩羽", ("彩羽", "雑賀彩羽", "杂贺彩羽")),
    ("蛭子", ("蛭子", "エビス", "ヒルコ")),
)

# 简体 -> 繁体/日文汉字（用于把 OCR/简体中文字形归一到 Above 的规范名）。
SIMPLIFIED_TO_TRADITIONAL: dict[str, str] = {
    "结": "結", "绪": "緒", "织": "織", "缩": "縮", "华": "華", "爱": "愛",
    "丽": "麗", "丝": "絲", "贺": "賀", "卷": "巻", "麦": "麥", "见": "見",
    "说": "説", "话": "話", "长": "長", "门": "門", "间": "間", "马": "馬",
    "鸟": "鳥", "东": "東", "们": "們", "为": "為", "会": "會", "风": "風",
    "魔": "魔", "杂": "雑", "条": "條", "样": "様", "万": "萬", "与": "與",
}

# 必须进入深度档案的跨章关键配角（即便按台词数未进 Top-12）。
KEY_CROSS_CHAPTER_CHARS: tuple[str, ...] = (
    "陽葵", "サトミ", "春希", "永夜", "結衣", "菊丸", "彩羽",
)


def _build_base_alias_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for canon, variants in ALIAS_GROUPS:
        for v in variants:
            m[v] = canon
            m["".join(SIMPLIFIED_TO_TRADITIONAL.get(ch, ch) for ch in v)] = canon
            m["".join(_trad_to_simp(ch) for ch in v)] = canon
    return m


def _trad_to_simp(ch: str) -> str:
    for simp, trad in SIMPLIFIED_TO_TRADITIONAL.items():
        if ch == trad:
            return simp
    return ch


BASE_ALIAS_MAP: dict[str, str] = _build_base_alias_map()


def _edit_distance(a: str, b: str) -> int:
    """轻量 Levenshtein，避免引入额外依赖。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_alias_map(glossary: dict[str, Any] | None = None) -> dict[str, str]:
    """基础别名表 + glossary 译名双向映射（构建一次，全局复用）。"""
    m = dict(BASE_ALIAS_MAP)
    if glossary:
        for k, v in glossary.get("characters", {}).items():
            ks, vs = str(k).strip(), str(v).strip()
            if not ks or not vs:
                continue
            # 译名与原文视作同一角色的两个别名，规范名优先用原文形。
            m.setdefault(ks, ks)
            m.setdefault(vs, ks)
    return m


def canonicalize_name(name: object, alias_map: dict[str, str] | None = None) -> str:
    """把说话人/角色名归并为规范名，消除 OCR 变体与简繁/日中文分裂节点。"""
    if not is_real_character(name):
        return ""
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = re.sub(r"[\s\u3000]+", "", s)
    if not s:
        return ""
    if s in amap:
        return amap[s]
    # 繁简字形归一后再查表（结衣->結衣、华花->華花、伊织->伊織）。
    converted = "".join(SIMPLIFIED_TO_TRADITIONAL.get(ch, ch) for ch in s)
    if converted in amap:
        return amap[converted]
    # 保守的编辑距离兜底：仅当长度>=3、首字相同且只差 1 字符才归并，
    # 避免把两个真实角色（如由良/ゆら 已显式映射）误并。
    if len(s) >= 3:
        for canon in {v for v in amap.values() if len(v) >= 3}:
            if canon[0] == converted[0] and _edit_distance(converted, canon) == 1:
                return canon
    return converted or s


def canonical_speakers(
    lines: list[dict[str, Any]], alias_map: dict[str, str] | None = None
) -> list[str]:
    """按行序返回归并后的真实说话人名（过滤旁白/机翻元标签）。"""
    return [
        c
        for r in lines
        if is_real_character(r.get("speaker"))
        for c in [canonicalize_name(r.get("speaker"), alias_map)]
        if c
    ]


def canonical_scene_characters(
    scene: dict[str, Any], alias_map: dict[str, str] | None = None
) -> list[str]:
    return list(
        dict.fromkeys(
            c
            for x in scene.get("characters", [])
            if is_real_character(x)
            for c in [canonicalize_name(x, alias_map)]
            if c
        )
    )


def name_mentioned(
    name: str, scene: dict[str, Any], alias_map: dict[str, str] | None = None
) -> bool:
    """判断规范角色名是否被某场景卡提及（含别名/简繁/OCR 变体）。"""
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    if name in canonical_scene_characters(scene, amap):
        return True
    blob = str(scene.get("summary_zh", ""))
    variants = [k for k, v in amap.items() if v == name and len(k) >= 2]
    if name in blob or any(v in blob for v in variants):
        return True
    for q in scene.get("quotes", []):
        if canonicalize_name(q.get("speaker"), amap) == name:
            return True
    return False



def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL，跳过空行。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_glossary(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "characters": {**BUILTIN_GLOSSARY["characters"], **data.get("characters", {})},
            "verbal_tics": {**BUILTIN_GLOSSARY["verbal_tics"], **data.get("verbal_tics", {})},
        }
    return json.loads(json.dumps(BUILTIN_GLOSSARY))  # deep copy


# --------------------------------------------------------------------------- #
# 切块
# --------------------------------------------------------------------------- #
def chunk_lines(
    lines: list[dict[str, Any]], chunk_size: int = 400
) -> list[dict[str, Any]]:
    """按 part 分组后，每 chunk_size 行切一个 Map 块（不跨 P）。

    返回每个块含 lines 列表、part、起止 line_id 与 ts。
    """
    chunk_size = max(300, min(500, int(chunk_size)))
    by_part: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in lines:
        by_part[row.get("part", 1)].append(row)

    chunks: list[dict[str, Any]] = []
    for part in sorted(by_part, key=lambda p: (str(type(p)), p)):
        rows = by_part[part]
        for offset in range(0, len(rows), chunk_size):
            sub = rows[offset : offset + chunk_size]
            chunks.append(
                {
                    "part": part,
                    "lines": sub,
                    "source_lines": [sub[0]["line_id"], sub[-1]["line_id"]],
                    "ts_start": min(float(r.get("ts_sec", 0)) for r in sub),
                    "ts_end": max(float(r.get("ts_sec", 0)) for r in sub),
                }
            )
    return chunks


# --------------------------------------------------------------------------- #
# LLM 调用（OpenAI 兼容）
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> dict[str, Any]:
    """从可能带 ```json 围栏 / 前后解释的响应中抠出第一个 JSON 对象。"""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"响应中找不到 JSON 对象: {text[:200]!r}")
    # strict=False：容忍 LLM 输出中未转义的控制字符（gemini-web2api 实测踩坑）。
    return json.loads(text[start : end + 1], strict=False)


def llm_json(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 90,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return extract_json_object(content)


def glossary_hint(glossary: dict[str, Any]) -> str:
    chars = "、".join(f"{k}→{v}" for k, v in glossary.get("characters", {}).items())
    tics = "、".join(f"{k}→{v}" for k, v in glossary.get("verbal_tics", {}).items())
    return f"角色译名表：{chars}\n口癖/惯用语表：{tics}"


# --------------------------------------------------------------------------- #
# Map 阶段
# --------------------------------------------------------------------------- #
def mock_translate(text_ja: str, glossary: dict[str, Any]) -> str:
    """无 Key 单测用的确定性伪译文：套译名表与短语表。"""
    out = text_ja
    for ja, zh in glossary.get("verbal_tics", {}).items():
        out = out.replace(ja, zh)
    for ja, zh in glossary.get("characters", {}).items():
        out = out.replace(ja, zh)
    for ja, zh in MOCK_PHRASES.items():
        out = out.replace(ja, zh)
    return f"[中] {out}"


def build_mock_scene(
    chunk: dict[str, Any], idx: int, glossary: dict[str, Any], alias_map: dict[str, str] | None = None
) -> dict[str, Any]:
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    lines: list[dict[str, Any]] = chunk["lines"]
    canon_speakers = [canonicalize_name(r.get("speaker"), amap) for r in lines]
    speakers = list(
        dict.fromkeys(s for s in canon_speakers if is_real_character(s))
    )
    top = Counter(s for s in canon_speakers if is_real_character(s)).most_common(3)
    char_names = [s for s, _ in top]
    first = lines[0]
    title = f"P{chunk['part']}·场景{idx:02d} {char_names[0] if char_names else '无声'}篇"
    hook = "、".join(s for s, _ in top[:2])
    summary = (
        f"【P{chunk['part']}】{hook} 等 {len(speakers)} 位角色登场，"
        f"共 {len(lines)} 句台词；对话在俏皮吐槽与温情互动间推进，暗线伏笔埋于细节。"
    )
    quotes = [
        {
            "speaker": canonicalize_name(r.get("speaker"), amap) or str(r.get("speaker", "?")),
            "text_ja": r.get("text_ja", ""),
            "text_zh": mock_translate(r.get("text_ja", ""), glossary),
        }
        for r in lines[:3]
    ]
    translations = [
        {"line_id": r["line_id"], "text_zh": mock_translate(r.get("text_ja", ""), glossary)}
        for r in lines
    ]
    return {
        "scene_id": f"S{idx:03d}",
        "part": chunk["part"],
        "ts_start": chunk["ts_start"],
        "ts_end": chunk["ts_end"],
        "title": title,
        "characters": char_names,
        "summary_zh": summary,
        "quotes": quotes,
        "source_lines": chunk["source_lines"],
        "_translations": translations,
        "_first_line_id": first["line_id"],
    }


def build_llm_scene(
    chunk: dict[str, Any],
    idx: int,
    glossary: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    lines: list[dict[str, Any]] = chunk["lines"]
    numbered = "\n".join(
        f"{r['line_id']}|{r.get('speaker', '?')}|{r.get('text_ja', '')}" for r in lines
    )
    # 关键修复：Map 阶段只让 LLM 输出「紧凑场景卡」——绝不把 450 行逐句译文
    # 塞进单个 JSON（geminiweb2api/gemini-3.6-flash 实测会超长截断/控制字符报错）。
    # 逐句双语在脚本侧回填：金句用 LLM 译，其余用 mock_translate 兜底。
    messages = [
        {
            "role": "system",
            "content": (
                "你是视觉小说剧情编辑。把给定台词块压缩成一张高信息密度场景卡，"
                "保留笑点/糖点/伏笔，严禁脑补。只返回紧凑 JSON，不要逐句翻译全文："
                '{"title":str,"summary_zh":str,"characters":[str],'
                '"quotes":[{"speaker":str,"text_ja":str,"text_zh":str}]}。'
                "quotes 只挑 3-5 句代表金句（含日文原文与中文译文），译名严格遵守下表。\n"
                + glossary_hint(glossary)
            ),
        },
        {
            "role": "user",
            "content": f"P{chunk['part']} 台词（line_id|speaker|text_ja）：\n{numbered}",
        },
    ]
    data = llm_json(messages, base_url=base_url, api_key=api_key, model=model)
    quotes = [
        {
            "speaker": canonicalize_name(q.get("speaker"), amap) or str(q.get("speaker", "?")),
            "text_ja": str(q.get("text_ja", "")),
            "text_zh": str(q.get("text_zh", "")),
        }
        for q in data.get("quotes", [])[:5]
    ]
    # 双语回填：金句按原文反查拿到 LLM 译文，其余行用确定性伪译文兜底，
    # 保证 script_bilingual.jsonl 仍逐行覆盖且不依赖单个超长 JSON。
    quote_trans = {q["text_ja"]: q["text_zh"] for q in quotes if q["text_ja"] and q["text_zh"]}
    translations = [
        {
            "line_id": r["line_id"],
            "text_zh": quote_trans.get(
                str(r.get("text_ja", "")), mock_translate(r.get("text_ja", ""), glossary)
            ),
        }
        for r in lines
    ]
    return {
        "scene_id": f"S{idx:03d}",
        "part": chunk["part"],
        "ts_start": chunk["ts_start"],
        "ts_end": chunk["ts_end"],
        "title": str(data.get("title", f"P{chunk['part']} 场景 {idx}"))[:120],
        "characters": list(
            dict.fromkeys(
                c
                for x in data.get("characters", [])
                if is_real_character(x)
                for c in [canonicalize_name(x, amap)]
                if c
            )
        )[:12]
        or [
            canonicalize_name(r.get("speaker"), amap)
            for r in lines[:3]
            if is_real_character(r.get("speaker"))
        ],
        "summary_zh": str(data.get("summary_zh", ""))[:1200],
        "quotes": quotes,
        "source_lines": chunk["source_lines"],
        "_translations": translations,
        "_first_line_id": lines[0]["line_id"],
    }


def map_scenes(
    chunks: list[dict[str, Any]],
    glossary: dict[str, Any],
    *,
    mock: bool,
    workers: int,
    base_url: str,
    api_key: str,
    model: str,
    alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    results: dict[int, dict[str, Any]] = {}
    if mock:
        for i, chunk in enumerate(chunks, 1):
            results[i] = build_mock_scene(chunk, i, glossary, amap)
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    build_llm_scene,
                    chunk,
                    i,
                    glossary,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    alias_map=amap,
                ): i
                for i, chunk in enumerate(chunks, 1)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # 单块失败不拖垮全片，回退到 mock 卡
                    print(f"[warn] Map 块 {i} 调用失败，回退 mock 卡：{exc}", file=sys.stderr)
                    results[i] = build_mock_scene(chunks[i - 1], i, glossary, amap)
    return [results[i] for i in sorted(results)]


# --------------------------------------------------------------------------- #
# 覆盖率审计
# --------------------------------------------------------------------------- #
def coverage_audit(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    *,
    window: int = 500,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """统计 Top-15 规范角色与每 window 行区间的场景卡提及率，返回红灯告警。

    降噪：旁白 / ナレーション / ??? / 不明 / 空串等非角色元标签不进 Top-15；
    归并：同一角色的 OCR / 简繁 / 日中文变体先经 canonicalize_name 合并再统计。
    """
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    freq = Counter(canonical_speakers(lines, amap))
    top = freq.most_common(15)
    total = len(scenes)
    alerts: list[dict[str, Any]] = []

    speaker_rows: list[dict[str, Any]] = []
    for name, count in top:
        mentioned = sum(1 for s in scenes if name_mentioned(name, s, amap))
        rate = round(mentioned / total, 3) if total else 0.0
        speaker_rows.append(
            {
                "speaker": name,
                "line_count": count,
                "scene_mentions": mentioned,
                "mention_rate": rate,
            }
        )
        if count >= 3 and mentioned == 0:
            alerts.append(
                {
                    "type": "high_freq_speaker_missing",
                    "speaker": name,
                    "line_count": count,
                    "message": f"高频角色「{name}」出现 {count} 次，场景卡 0 提及（疑似吞剧情）",
                }
            )

    max_id = max((int(r.get("line_id", 0)) for r in lines), default=0)
    windows: list[dict[str, Any]] = []
    for start in range(1, max_id + 1, window):
        end = min(start + window - 1, max_id)
        covered = any(
            not (int(s["source_lines"][1]) < start or int(s["source_lines"][0]) > end)
            for s in scenes
        )
        windows.append({"start": start, "end": end, "covered": covered})
        if not covered:
            alerts.append(
                {
                    "type": "line_window_uncovered",
                    "start": start,
                    "end": end,
                    "message": f"行区间 {start}-{end} 无任何场景卡覆盖（疑似漏块）",
                }
            )

    return {
        "top_speakers": speaker_rows,
        "windows": windows,
        "scene_count": total,
        "alerts": alerts,
        "red_light": bool(alerts),
    }


def targeted_rewrite(
    audit: dict[str, Any],
    glossary: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """非 mock 模式针对红灯告警做一次定向补写，返回增补场景卡。"""
    fixes: list[dict[str, Any]] = []
    for alert in audit.get("alerts", [])[:limit]:
        focus = alert.get("speaker") or f"行区间 {alert.get('start')}-{alert.get('end')}"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是剧情审校。之前摘要漏掉了以下内容，请针对该焦点补一张高信息密度场景卡。"
                    '严格返回 JSON：{"title":str,"summary_zh":str,"characters":[str],'
                    '"quotes":[{"speaker":str,"text_ja":str,"text_zh":str}]}。\n'
                    + glossary_hint(glossary)
                ),
            },
            {"role": "user", "content": f"漏写焦点：{focus}\n告警：{alert['message']}"},
        ]
        try:
            data = llm_json(messages, base_url=base_url, api_key=api_key, model=model)
        except Exception as exc:
            print(f"[warn] 定向补写失败（{focus}）：{exc}", file=sys.stderr)
            continue
        fixes.append(
            {
                "scene_id": f"FIX-{len(fixes) + 1:02d}",
                "part": 0,
                "ts_start": 0.0,
                "ts_end": 0.0,
                "title": f"补写·{focus}",
                "characters": [str(c) for c in data.get("characters", [])],
                "summary_zh": str(data.get("summary_zh", "")),
                "quotes": [
                    {
                        "speaker": str(q.get("speaker", "?")),
                        "text_ja": str(q.get("text_ja", "")),
                        "text_zh": str(q.get("text_zh", "")),
                    }
                    for q in data.get("quotes", [])
                ],
                "source_lines": [alert.get("start", 0), alert.get("end", 0)],
                "_translations": [],
                "is_fix": True,
            }
        )
    return fixes


# --------------------------------------------------------------------------- #
# Reduce 阶段（v3：LLM 深度角色档案 / 三幕 / 语义关系图 / 跨章因果链）
# --------------------------------------------------------------------------- #
PROFILE_FIELDS: tuple[str, ...] = (
    "name", "role_tag", "faction", "summary_zh", "motivation",
    "key_interactions", "arc_highlights",
)
SEMANTIC_LABELS: tuple[str, ...] = (
    "主仆", "师徒", "同盟", "亲族", "恋慕", "对手", "同门", "共谋", "生母", "姐妹",
)


def _clamp(text: object, limit: int) -> str:
    return str(text if text is not None else "")[:limit]


def _usable_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [s for s in scenes if not s.get("is_fix")]
    return usable or scenes


def _safe_llm(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    if not base_url:
        return {}
    try:
        return llm_json(messages, base_url=base_url, api_key=api_key, model=model)
    except Exception as exc:  # Reduce 阶段任何一次 LLM 失败都降级到确定性结构
        print(f"[warn] Reduce LLM 调用失败，保留确定性结构：{exc}", file=sys.stderr)
        return {}


def _target_characters(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    alias_map: dict[str, str],
    top_n: int = 12,
) -> list[str]:
    freq = Counter(canonical_speakers(lines, alias_map))
    targets = [n for n, _ in freq.most_common(top_n)]
    # 关键跨章配角即便台词数未进 Top-12 也必须进深度档案。
    for key in KEY_CROSS_CHAPTER_CHARS:
        cn = canonicalize_name(key, alias_map)
        if cn in freq and cn not in targets:
            targets.append(cn)
    return targets


def _character_evidence(
    name: str,
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    alias_map: dict[str, str],
    max_lines: int = 8,
) -> dict[str, Any]:
    related = [s for s in scenes if name in canonical_scene_characters(s, alias_map)]
    script_lines = [
        r for r in lines if canonicalize_name(r.get("speaker"), alias_map) == name
    ]
    sample = sorted(
        script_lines, key=lambda r: len(str(r.get("text_ja", ""))), reverse=True
    )[:max_lines]
    co: Counter[str] = Counter()
    for s in related:
        for c in canonical_scene_characters(s, alias_map):
            if c != name:
                co[c] += 1
    return {
        "name": name,
        "line_count": len(script_lines),
        "scene_count": len(related),
        "parts": sorted({r.get("part") for r in script_lines if r.get("part") is not None}),
        "co_stars": co.most_common(3),
        "scene_titles": [_clamp(s.get("title", ""), 60) for s in related[:4]],
        "scene_summaries": [_clamp(s.get("summary_zh", ""), 140) for s in related[:3]],
        "arc_highlights": [
            f"P{s.get('part', '?')} {_clamp(s.get('title', ''), 40)}" for s in related[:5]
        ],
        "sample_lines": [_clamp(r.get("text_ja", ""), 80) for r in sample],
    }


def _mock_profile(ev: dict[str, Any], glossary: dict[str, Any]) -> dict[str, Any]:
    name = ev["name"]
    zh = glossary.get("characters", {}).get(name, name)
    co = "、".join(c for c, _ in ev["co_stars"]) or "其他角色"
    parts = ev["parts"]
    pstr = f"P{parts[0]}–P{parts[-1]}" if parts else "全片"
    titles = ev["scene_titles"] or ["主线"]
    return {
        "name": name,
        "name_zh": zh,
        "role_tag": f"{pstr} 关键配角",
        "faction": "中立 / 待厘清",
        "summary_zh": (
            f"{zh}在 {pstr} 共 {ev['line_count']} 句台词、出场 {ev['scene_count']} 场，"
            f"主要与 {co} 交锋；行动线贯穿「{titles[0]}」等场景，"
            f"其性格与立场在冲突中逐步显影。"
        ),
        "motivation": (
            f"围绕与 {co} 的纠葛展开核心诉求（mock 占位；真实模式由 LLM 基于原台词生成）。"
        ),
        "key_interactions": (
            [f"与 {c}：共现 {n} 场" for c, n in ev["co_stars"]]
            or [f"与 {co}：对手戏密集"]
        ),
        "arc_highlights": ev["arc_highlights"] or titles[:3],
        # 兼容旧前端 / 交叉校验的字段
        "line_count": ev["line_count"],
        "scene_count": ev["scene_count"],
        "parts": parts,
        "highlights": titles[:3],
    }


def _llm_profiles(
    batch: list[dict[str, Any]],
    glossary: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    alias_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    compact = [
        {
            "name": ev["name"],
            "line_count": ev["line_count"],
            "parts": ev["parts"],
            "co_stars": ev["co_stars"],
            "scene_titles": ev["scene_titles"][:3],
            "scene_summaries": ev["scene_summaries"][:2],
            "sample_lines": ev["sample_lines"][:6],
        }
        for ev in batch
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是资深视觉小说剧情编辑。基于给定证据，为每个角色生成深度档案，"
                "严禁脑补证据之外的事实；summary_zh 80-150 字，须写出人物定位与反差；"
                "motivation 写核心动机与身世秘密；key_interactions 写与具体角色的关系；"
                "arc_highlights 写分 P 高光。只返回紧凑 JSON："
                '{"profiles":[{"name":str,"role_tag":str,"faction":str,"summary_zh":str,'
                '"motivation":str,"key_interactions":[str],"arc_highlights":[str]}]}。\n'
                + glossary_hint(glossary)
            ),
        },
        {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
    ]
    data = _safe_llm(messages, base_url=base_url, api_key=api_key, model=model)
    out: dict[str, dict[str, Any]] = {}
    for p in data.get("profiles", []):
        if isinstance(p, dict) and p.get("name"):
            # LLM 可能回填 OCR 误识名（华花/サトちゃん），按规范名归位后再匹配。
            canon = canonicalize_name(p["name"], amap)
            if canon:
                out[canon] = p
    return out


def _normalize_profile(
    p: dict[str, Any], ev: dict[str, Any], glossary: dict[str, Any]
) -> dict[str, Any]:
    merged = _mock_profile(ev, glossary)  # 字段齐全的确定性底座
    for key in PROFILE_FIELDS:
        value = p.get(key)
        if key in ("key_interactions", "arc_highlights"):
            if isinstance(value, list):
                merged[key] = [_clamp(x, 120) for x in value[:6] if str(x).strip()]
        elif key == "summary_zh":
            if value:
                merged[key] = _clamp(value, 400)
        elif value:
            merged[key] = _clamp(value, 160)
    merged["name"] = ev["name"]  # 强制规范名，杜绝 OCR 误识名回流
    return merged


def build_character_profiles(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    glossary: dict[str, Any],
    *,
    mock: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    alias_map: dict[str, str] | None = None,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Top-12 核心 + 关键跨章配角的深度角色档案（LLM 驱动，mock 确定性）。"""
    amap = alias_map if alias_map is not None else build_alias_map(glossary)
    targets = _target_characters(lines, scenes, amap, top_n=top_n)
    evidences = [_character_evidence(n, lines, scenes, amap) for n in targets]
    profiles = [_mock_profile(ev, glossary) for ev in evidences]
    if mock or not evidences:
        return profiles
    # 非 mock 分 2 批紧凑 JSON，避免单次输出过长截断。
    half = max(1, (len(evidences) + 1) // 2)
    generated: dict[str, dict[str, Any]] = {}
    for i in range(0, len(evidences), half):
        generated.update(
            _llm_profiles(
                evidences[i : i + half],
                glossary,
                base_url=base_url,
                api_key=api_key,
                model=model,
                alias_map=amap,
            )
        )
    return [
        _normalize_profile(generated[ev["name"]], ev, glossary)
        if ev["name"] in generated
        else base
        for ev, base in zip(evidences, profiles)
    ]


def _mock_acts(scenes: list[dict[str, Any]], alias_map: dict[str, str]) -> list[dict[str, Any]]:
    usable = _usable_scenes(scenes)
    n = len(usable)
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    acts: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(bounds, 1):
        group = usable[a:b]
        parts = sorted({s.get("part") for s in group if s.get("part") is not None})
        chars = Counter(c for s in group for c in canonical_scene_characters(s, alias_map))
        lead = "、".join(c for c, _ in chars.most_common(3)) or "众人"
        pstr = f"P{parts[0]}–P{parts[-1]}" if parts else "全片"
        first = _clamp(group[0].get("title", ""), 40) if group else "序章"
        last = _clamp(group[-1].get("title", ""), 40) if group else "尾声"
        acts.append(
            {
                "act": i,
                "title": ["起·命运转折", "承·暗流与羁绊", "转·高潮与收束"][i - 1],
                "summary_zh": (
                    f"第 {i} 幕（{pstr}）以 {lead} 的互动为主线，共 {len(group)} 个关键场景；"
                    f"叙事自「{first}」推进至「{last}」，人物动机与跨章伏笔在此阶段逐步显影。"
                ),
                "scene_ids": [s["scene_id"] for s in group],
            }
        )
    return acts


def build_acts(
    scenes: list[dict[str, Any]],
    *,
    mock: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    fallback = _mock_acts(scenes, amap)
    if mock:
        return fallback
    digest = [
        {
            "scene_id": s.get("scene_id", ""),
            "part": s.get("part"),
            "title": _clamp(s.get("title", ""), 60),
            "summary_zh": _clamp(s.get("summary_zh", ""), 160),
        }
        for s in _usable_scenes(scenes)[:60]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是剧情编辑。基于全部场景卡写可读的中文三幕剧情综述，"
                "严禁把标题用箭头机械拼接，严禁脑补。只返回紧凑 JSON："
                '{"acts":[{"title":str,"summary_zh":str}]}，必须恰好 3 项。'
            ),
        },
        {"role": "user", "content": json.dumps(digest, ensure_ascii=False)},
    ]
    data = _safe_llm(messages, base_url=base_url, api_key=api_key, model=model)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data.get("acts", [])[:3]):
        base = fallback[i] if i < len(fallback) else {}
        out.append(
            {
                "act": i + 1,
                "title": _clamp(item.get("title") or base.get("title", ""), 40),
                "summary_zh": _clamp(
                    item.get("summary_zh") or base.get("summary_zh", ""), 600
                ),
                "scene_ids": base.get("scene_ids", []),
            }
        )
    return out or fallback


def _co_edges(
    scenes: list[dict[str, Any]], names: list[str], alias_map: dict[str, str]
) -> Counter[tuple[str, str]]:
    allowed = set(names)
    co: Counter[tuple[str, str]] = Counter()
    for s in scenes:
        chars = [c for c in canonical_scene_characters(s, alias_map) if c in allowed]
        for i, a in enumerate(chars):
            for b in chars[i + 1 :]:
                co[(a, b)] += 1
    return co


def _mermaid_text(names: list[str], edges: list[tuple[tuple[str, str], str]]) -> str:
    if not names:
        return "graph LR\n  Empty[\"暂无角色关系\"]"
    ids = {n: f"C{i}" for i, n in enumerate(names)}
    out = ["graph LR"] + [f'  {ids[n]}["{n}"]' for n in names]
    for (a, b), label in edges[:18]:
        out.append(f'  {ids[a]} ---|"{label}"| {ids[b]}')
    return "\n".join(out)


def build_semantic_mermaid(
    profiles: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    *,
    mock: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    alias_map: dict[str, str] | None = None,
    max_nodes: int = 15,
) -> str:
    """Top-12~15 规范核心角色的语义关系图（边带真实语义标签，非机械共现）。"""
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    names = [
        str(p["name"])
        for p in profiles
        if is_real_character(p.get("name")) and canonicalize_name(p.get("name"), amap)
    ][:max_nodes]
    if not names:
        return "graph LR\n  Empty[\"暂无角色关系\"]"
    co = _co_edges(scenes, names, amap)
    if mock:
        pairs = co.most_common(16)
        edges = [
            (pair, SEMANTIC_LABELS[i % len(SEMANTIC_LABELS)])
            for i, (pair, _) in enumerate(pairs)
        ]
    else:
        digest = [
            {
                "name": p.get("name"),
                "role_tag": _clamp(p.get("role_tag", ""), 60),
                "summary_zh": _clamp(p.get("summary_zh", ""), 80),
            }
            for p in profiles[:max_nodes]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是剧情关系分析师。基于角色档案输出真实语义关系边，"
                    "标签用简短中文（如 主仆/恋人/生母/亲姐妹/师徒/同盟/约谈），"
                    "严禁输出「共现 N 场」这类机械标签，严禁引入名单之外的角色。"
                    '只返回紧凑 JSON：{"edges":[{"a":str,"b":str,"label":str}]}。'
                ),
            },
            {"role": "user", "content": json.dumps(digest, ensure_ascii=False)},
        ]
        data = _safe_llm(messages, base_url=base_url, api_key=api_key, model=model)
        edges: list[tuple[tuple[str, str], str]] = []
        seen: set[tuple[str, str]] = set()
        for e in data.get("edges", []):
            a = canonicalize_name(e.get("a"), amap)
            b = canonicalize_name(e.get("b"), amap)
            label = _clamp(e.get("label", ""), 16)
            if a in names and b in names and a != b and label and "共现" not in label:
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    edges.append((key, label))
    if not edges and len(names) > 1:
        edges = [((names[0], names[1]), "关联")]
    return _mermaid_text(names, edges)


def build_mermaid(
    scenes: list[dict[str, Any]], alias_map: dict[str, str] | None = None
) -> str:
    """向后兼容入口：从场景卡推导规范角色并生成语义图。"""
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    names = sorted({c for s in scenes for c in canonical_scene_characters(s, amap)})[:15]
    profiles = [{"name": n} for n in names]
    return build_semantic_mermaid(profiles, scenes, mock=True, alias_map=amap)


def _mock_causal_chains(
    scenes: list[dict[str, Any]],
    alias_map: dict[str, str],
) -> list[dict[str, Any]]:
    by_part: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for s in _usable_scenes(scenes):
        by_part[s.get("part")].append(s)
    parts = sorted(by_part, key=lambda p: str(p))
    chains: list[dict[str, Any]] = []
    if len(parts) >= 2:
        for a, b in zip(parts, parts[1:]):
            involved = list(
                dict.fromkeys(
                    c
                    for s in by_part[a] + by_part[b]
                    for c in canonical_scene_characters(s, alias_map)
                )
            )[:6]
            chains.append(
                {
                    "title": f"P{a}→P{b} 因果推进",
                    "steps": [
                        f"P{a}：{_clamp(by_part[a][0].get('title', ''), 60)}",
                        f"P{b}：{_clamp(by_part[b][0].get('title', ''), 60)}",
                    ],
                    "involved": involved,
                }
            )
    elif parts:
        p = parts[0]
        involved = list(
            dict.fromkeys(c for s in by_part[p] for c in canonical_scene_characters(s, alias_map))
        )[:6]
        chains.append(
            {
                "title": f"P{p} 关键伏笔",
                "steps": [f"P{p}：{_clamp(s.get('title', ''), 60)}" for s in by_part[p][:5]],
                "involved": involved,
            }
        )
    return chains[:6]


def build_causal_chains(
    scenes: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    *,
    mock: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """跨章因果链 / 关键伏笔（LLM 驱动，mock 确定性）。"""
    amap = alias_map if alias_map is not None else BASE_ALIAS_MAP
    fallback = _mock_causal_chains(scenes, amap)
    if mock:
        return fallback
    digest = [
        {
            "scene_id": s.get("scene_id", ""),
            "part": s.get("part"),
            "title": _clamp(s.get("title", ""), 60),
            "summary_zh": _clamp(s.get("summary_zh", ""), 120),
        }
        for s in _usable_scenes(scenes)[:40]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是剧情编辑，专门梳理跨分 P 的因果链与关键伏笔（如信件截胡→误会→约谈、"
                "幕后黑手暗线）。严禁脑补，只基于场景卡。只返回紧凑 JSON："
                '{"chains":[{"title":str,"steps":[str],"involved":[str]}]}。'
            ),
        },
        {"role": "user", "content": json.dumps(digest, ensure_ascii=False)},
    ]
    data = _safe_llm(messages, base_url=base_url, api_key=api_key, model=model)
    out: list[dict[str, Any]] = []
    for c in data.get("chains", []):
        steps = [_clamp(x, 160) for x in c.get("steps", []) if str(x).strip()][:8]
        involved = [
            canonicalize_name(x, amap)
            for x in c.get("involved", [])
            if is_real_character(x)
        ][:6]
        if c.get("title") and steps:
            out.append(
                {
                    "title": _clamp(c["title"], 80),
                    "steps": steps,
                    "involved": [x for x in involved if x],
                }
            )
    return out[:8] or fallback


def reduce_plot(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    audit: dict[str, Any],
    glossary: dict[str, Any],
    *,
    title: str,
    bvid: str,
    mock: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    amap = alias_map if alias_map is not None else build_alias_map(glossary)
    profiles = build_character_profiles(
        lines, scenes, glossary, mock=mock, base_url=base_url,
        api_key=api_key, model=model, alias_map=amap,
    )
    return {
        "work_title": title,
        "bvid": bvid,
        "acts": build_acts(
            scenes, mock=mock, base_url=base_url, api_key=api_key, model=model, alias_map=amap
        ),
        "mermaid": build_semantic_mermaid(
            profiles, scenes, mock=mock, base_url=base_url,
            api_key=api_key, model=model, alias_map=amap,
        ),
        "characters": profiles,
        "causal_chains": build_causal_chains(
            scenes, profiles, mock=mock, base_url=base_url,
            api_key=api_key, model=model, alias_map=amap,
        ),
        "scenes": [
            {k: v for k, v in s.items() if not k.startswith("_") and k != "is_fix"}
            for s in scenes
        ],
        "coverage_audit": audit,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    input_path = Path(args.input) if args.input else work_dir / "output" / "script.jsonl"
    bi_path = (
        Path(args.output_bilingual)
        if args.output_bilingual
        else work_dir / "output" / "script_bilingual.jsonl"
    )
    plot_path = (
        Path(args.output_plot)
        if args.output_plot
        else work_dir / "output" / "plot_structured.json"
    )

    lines = load_jsonl(input_path)
    if not lines:
        raise SystemExit(f"输入为空或不存在：{input_path}")
    glossary = load_glossary(Path(args.glossary) if args.glossary else work_dir / "glossary.json")
    alias_map = build_alias_map(glossary)
    chunks = chunk_lines(lines, args.chunk_size)
    print(f"[Map] {len(lines)} 行 -> {len(chunks)} 块（chunk={args.chunk_size}, mock={args.mock}）")

    scenes = map_scenes(
        chunks,
        glossary,
        mock=args.mock,
        workers=args.workers,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        alias_map=alias_map,
    )

    audit = coverage_audit(lines, scenes, alias_map=alias_map)
    print(
        f"[Audit] scene={audit['scene_count']} "
        f"top_speakers={len(audit['top_speakers'])} "
        f"windows={len(audit['windows'])} alerts={len(audit['alerts'])}"
    )
    for a in audit["alerts"][:10]:
        print(f"  ⚠ {a['message']}")

    if audit["alerts"] and not args.mock:
        fixes = targeted_rewrite(
            audit,
            glossary,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        if fixes:
            print(f"[Audit] 定向补写 {len(fixes)} 张场景卡")
            scenes.extend(fixes)
            audit = coverage_audit(lines, scenes, alias_map=alias_map)

    # 逐句译文写回：LLM / mock 产出的 translations 覆盖原行。
    trans_map: dict[int, str] = {}
    for s in scenes:
        for t in s.get("_translations", []):
            trans_map[int(t["line_id"])] = str(t["text_zh"])
    bilingual: list[dict[str, Any]] = []
    for row in lines:
        item = dict(row)
        # 说话人同样归一：Tab 3 角色筛选与深度档案的规范名必须一致。
        if is_real_character(row.get("speaker")):
            item["speaker"] = canonicalize_name(row.get("speaker"), alias_map)
        item["text_zh"] = trans_map.get(
            int(row["line_id"]), mock_translate(row.get("text_ja", ""), glossary)
        )
        bilingual.append(item)
    write_jsonl(bi_path, bilingual)

    plot = reduce_plot(
        lines, scenes, audit, glossary, title=args.title, bvid=args.bvid,
        mock=args.mock, base_url=args.base_url, api_key=args.api_key, model=args.model,
        alias_map=alias_map,
    )
    write_json(plot_path, plot)
    print(f"[Reduce] 输出 {bi_path} / {plot_path}")
    return plot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Map-Reduce 剧情归约")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--input", default=None)
    p.add_argument("--output-bilingual", default=None)
    p.add_argument("--output-plot", default=None)
    p.add_argument("--glossary", default=None)
    p.add_argument("--mock", action="store_true", help="无 Key 单测模式，不发网络请求")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--title", default=DEFAULT_TITLE)
    p.add_argument("--bvid", default=DEFAULT_BVID)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
