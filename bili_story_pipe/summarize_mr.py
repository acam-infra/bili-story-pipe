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


def build_mock_scene(chunk: dict[str, Any], idx: int, glossary: dict[str, Any]) -> dict[str, Any]:
    lines: list[dict[str, Any]] = chunk["lines"]
    speakers = list(
        dict.fromkeys(
            str(r.get("speaker", "?")) for r in lines if is_real_character(r.get("speaker"))
        )
    )
    top = Counter(
        str(r.get("speaker", "?")) for r in lines if is_real_character(r.get("speaker"))
    ).most_common(3)
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
            "speaker": str(r.get("speaker", "?")),
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
) -> dict[str, Any]:
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
            "speaker": str(q.get("speaker", "?")),
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
        "characters": [str(c) for c in data.get("characters", [])][:12] or [
            str(r.get("speaker", "?")) for r in lines[:3] if is_real_character(r.get("speaker"))
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
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if mock:
        for i, chunk in enumerate(chunks, 1):
            results[i] = build_mock_scene(chunk, i, glossary)
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
                ): i
                for i, chunk in enumerate(chunks, 1)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # 单块失败不拖垮全片，回退到 mock 卡
                    print(f"[warn] Map 块 {i} 调用失败，回退 mock 卡：{exc}", file=sys.stderr)
                    results[i] = build_mock_scene(chunks[i - 1], i, glossary)
    return [results[i] for i in sorted(results)]


# --------------------------------------------------------------------------- #
# 覆盖率审计
# --------------------------------------------------------------------------- #
def coverage_audit(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    *,
    window: int = 500,
) -> dict[str, Any]:
    """统计 Top-15 角色与每 window 行区间的场景卡提及率，返回红灯告警。

    降噪：旁白 / ナレーション / ??? / 不明 / 空串等非角色元标签不进 Top-15，
    避免把旁白当成「漏剧情高频角色」触发无意义补写。
    """
    freq = Counter(
        str(r.get("speaker", "?"))
        for r in lines
        if is_real_character(r.get("speaker"))
    )
    top = freq.most_common(15)
    total = len(scenes)
    alerts: list[dict[str, Any]] = []

    speaker_rows: list[dict[str, Any]] = []
    for name, count in top:
        mentioned = sum(
            1
            for s in scenes
            if name in s.get("characters", [])
            or name in s.get("summary_zh", "")
            or any(str(q.get("speaker", "")) == name for q in s.get("quotes", []))
        )
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
# Reduce 阶段
# --------------------------------------------------------------------------- #
def _mermaid_id(name: str, index: int) -> str:
    return f"C{index}"


def build_mermaid(scenes: list[dict[str, Any]]) -> str:
    co: Counter[tuple[str, str]] = Counter()
    for s in scenes:
        chars = sorted({str(c) for c in s.get("characters", [])})
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                co[(chars[i], chars[j])] += 1
    names = sorted({c for pair in co for c in pair})
    ids = {name: _mermaid_id(name, i) for i, name in enumerate(names)}
    out = ["graph LR"]
    for name in names:
        out.append(f'  {ids[name]}["{name}"]')
    for (a, b), n in co.most_common(24):
        out.append(f'  {ids[a]} ---|"共现 {n} 场"| {ids[b]}')
    return "\n".join(out) if len(out) > 1 else "graph LR\n  Empty[\"暂无角色共现\"]"


def build_character_profiles(
    lines: list[dict[str, Any]], scenes: list[dict[str, Any]], glossary: dict[str, Any]
) -> list[dict[str, Any]]:
    freq = Counter(
        str(r.get("speaker", "?"))
        for r in lines
        if is_real_character(r.get("speaker"))
    )
    scene_count = Counter()
    for s in scenes:
        for c in set(str(x) for x in s.get("characters", [])):
            scene_count[c] += 1
    profiles: list[dict[str, Any]] = []
    for name, count in freq.most_common(8):
        zh = glossary.get("characters", {}).get(name, name)
        related = [s for s in scenes if name in s.get("characters", [])]
        highlights = [s.get("title", "") for s in related[:3]]
        profiles.append(
            {
                "name": name,
                "name_zh": zh,
                "line_count": count,
                "scene_count": scene_count.get(name, 0),
                "parts": sorted({s.get("part") for s in related}),
                "summary_zh": (
                    f"{zh}共 {count} 句台词、出场 {scene_count.get(name, 0)} 场，"
                    f"是推动本段剧情的关键角色之一。"
                ),
                "highlights": highlights,
            }
        )
    return profiles


def build_acts(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [s for s in scenes if not s.get("is_fix")]
    if not usable:
        usable = scenes
    n = len(usable)
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    acts: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(bounds, 1):
        group = usable[a:b]
        titles = " → ".join(s.get("title", "") for s in group[:8])
        acts.append(
            {
                "act": i,
                "title": ["起·命运转折", "承·暗流与羁绊", "转·高潮与收束"][i - 1],
                "summary_zh": (
                    f"第 {i} 幕涵盖 {len(group)} 张场景卡（P"
                    f"{group[0]['part'] if group else '-'}–P{group[-1]['part'] if group else '-'}）："
                    f"{titles or '暂无内容'}。"
                ),
                "scene_ids": [s["scene_id"] for s in group],
            }
        )
    return acts


def reduce_plot(
    lines: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    audit: dict[str, Any],
    glossary: dict[str, Any],
    *,
    title: str,
    bvid: str,
) -> dict[str, Any]:
    return {
        "work_title": title,
        "bvid": bvid,
        "acts": build_acts(scenes),
        "mermaid": build_mermaid(scenes),
        "characters": build_character_profiles(lines, scenes, glossary),
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
    )

    audit = coverage_audit(lines, scenes)
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
            audit = coverage_audit(lines, scenes)

    # 逐句译文写回：LLM / mock 产出的 translations 覆盖原行。
    trans_map: dict[int, str] = {}
    for s in scenes:
        for t in s.get("_translations", []):
            trans_map[int(t["line_id"])] = str(t["text_zh"])
    bilingual: list[dict[str, Any]] = []
    for row in lines:
        item = dict(row)
        item["text_zh"] = trans_map.get(
            int(row["line_id"]), mock_translate(row.get("text_ja", ""), glossary)
        )
        bilingual.append(item)
    write_jsonl(bi_path, bilingual)

    plot = reduce_plot(
        lines, scenes, audit, glossary, title=args.title, bvid=args.bvid
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
