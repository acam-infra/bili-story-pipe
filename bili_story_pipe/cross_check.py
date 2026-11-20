#!/usr/bin/env python3
"""联网三方交叉校验：VNDB Kana API（+ 可选外挂检索）对齐本地归约结论。

输出 cross_check.json，每条核验项带状态徽章：
  verified  ✅ 双源印证    /  uncovered ⚠️ 视频未覆盖   /  conflict ❌ 冲突告警

用法：
    python -m bili_story_pipe.cross_check --work-dir . --title "猫忍えくすはーとSPIN! 2"
    python -m bili_story_pipe.cross_check --work-dir . --search-cmd "grok-search --json"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

VNDB_API = "https://api.vndb.org/kana"
BADGES = {
    "verified": "✅ 双源印证",
    "uncovered": "⚠️ 视频未覆盖",
    "conflict": "❌ 冲突告警",
}

# 中文/日文译名 -> VNDB 罗马字别名，用于跨语言对齐（glossary 译名表同源）。
# v58009《猫忍えくすはーとSPIN! 2》实测暴露：VNDB 的 name 是罗马字、original
# 才是日文原名（如「巻菱 ナナコ」），OCR 台词写的是「ナナコ」——仅靠罗马字
# name 必然误报「视频未覆盖」。这里补齐译名/日文名 -> 罗马字的映射。
NAME_ALIASES: dict[str, list[str]] = {
    "藤花": ["touka", "fujihana"],
    "由良": ["fuuma yura", "yura"],
    "北条": ["houjou haruki", "haruki"],
    "爱丽丝": ["alice", "arisu", "alicia", "iris"],
    "蛭子": ["ebisu"],
    "千花": ["nekoe chika", "chika"],
    "莉兹": ["liz", "lis"],
    # v58009 实测主要角色（OCR 台词用日文原名/名，需在译名表里能反查）。
    # 用「完整罗马字姓名」作别名键，避免共用姓氏（如 巻菱 Nanako/Kikumaru、
    # 甲賀 Io/Yue）互相串台造成误命中。
    "ナナコ": ["makibishi nanako", "nanako"],
    "菊丸": ["makibishi kikumaru", "kikumaru"],
    "伊緒": ["kouga io", "io"],
    "アイリス": ["iris"],
    "ゆら": ["fuuma yura", "yura"],
    "たま": ["fuuma tama", "tama"],
    "春希": ["houjou haruki", "haruki"],
}
ADULT_TAG_HINTS = ("sex", "hentai", "nudity", "cum", "rape", "anal", "oral")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff]+", "", str(text).lower())


def character_match_terms(ch: dict[str, Any] | str) -> list[str]:
    """生成一个 VNDB 角色（或角色名）的全部可比对齐词。

    覆盖：日文原名 original（含拆出的姓/名）、罗马字 name（含拆词）、aliases、
    glossary 中文/日文译名。例：
      {"name": "Makibishi Nanako", "original": "巻菱 ナナコ"}
      -> ["Makibishi Nanako", "Makibishi", "Nanako", "巻菱 ナナコ", "巻菱", "ナナコ"]
    """
    if isinstance(ch, str):
        name, original, aliases = ch, "", []
    else:
        name = str(ch.get("name", "") or "")
        original = str(ch.get("original", "") or "")
        aliases = ch.get("aliases", []) or []

    terms: list[str] = []
    for source in [name, original] + [str(a) for a in aliases]:
        if not source:
            continue
        terms.append(source)
        terms.extend(t for t in re.split(r"\s+", source) if t)

    # glossary 反查：若某个词命中了译名表的罗马字别名，补上对应的中文/日文译名。
    lowered = " ".join(terms).lower()
    for zh, romaji_list in NAME_ALIASES.items():
        if any(a.lower() in lowered for a in romaji_list):
            terms.append(zh)
    return [t for t in dict.fromkeys(terms) if t]


def _pick_best(results: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    """在 VNDB 检索结果里挑最贴近输入标题的条目（含数字/关键字命中）。"""
    if not results:
        return None
    query = str(title).lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", query))
    q_digits = set(re.findall(r"\d+", query))

    def score(vn: dict[str, Any]) -> int:
        name = str(vn.get("title", "")).lower()
        s = 0
        if _norm(name) == _norm(title):
            s += 100
        for tok in q_tokens:
            if tok in name:
                s += 4
        for d in q_digits:
            if re.search(rf"(?<!\d){re.escape(d)}(?!\d)", name):
                s += 10
        if "SPIN!" in name.upper():
            s += 2
        return s

    return max(results, key=score)


def fetch_vndb(
    title: str,
    *,
    base_url: str = VNDB_API,
    timeout: int = 20,
    results: int = 8,
) -> dict[str, Any]:
    """真实联网检索 VNDB Kana API：vn 官方信息 + 该作角色列表。"""
    session = requests.Session()
    vn_payload = {
        "filters": ["search", "=", title],
        "fields": (
            "title, released, description, platforms, length_minutes, "
            "tags{name, rating}"
        ),
        "results": results,
    }
    resp = session.post(base_url.rstrip("/") + "/vn", json=vn_payload, timeout=timeout)
    resp.raise_for_status()
    candidates = resp.json().get("results", [])
    best = _pick_best(candidates, title)
    if not best:
        return {"found": False, "query": title, "candidates": []}

    characters: list[dict[str, Any]] = []
    try:
        ch_payload = {
            "filters": ["vn", "=", ["id", "=", best["id"]]],
            # 关键修复：显式请求 original 与 aliases。日文原名 original 才是
            # OCR 台词里出现的写法（「巻菱 ナナコ」「甲賀 伊緒」「アイリス」）。
            "fields": "id,name,original,aliases,gender,vns{role}",
            "results": 100,
        }
        ch_resp = session.post(
            base_url.rstrip("/") + "/character", json=ch_payload, timeout=timeout
        )
        ch_resp.raise_for_status()
        for ch in ch_resp.json().get("results", []):
            roles = [
                str(v.get("role", ""))
                for v in ch.get("vns", [])
                if str(v.get("id")) == str(best["id"])
            ]
            characters.append(
                {
                    "id": ch.get("id", ""),
                    "name": ch.get("name", ""),
                    "original": ch.get("original", ""),
                    "aliases": ch.get("aliases", []) or [],
                    "gender": ch.get("gender"),
                    "roles": roles,
                }
            )
    except Exception as exc:  # 角色端点失败不阻断主流程
        print(f"[warn] VNDB 角色端点失败：{exc}", file=sys.stderr)

    return {
        "found": True,
        "query": title,
        "id": best.get("id", ""),
        "title": best.get("title", ""),
        "released": best.get("released", ""),
        "description": best.get("description", ""),
        "length_minutes": best.get("length_minutes"),
        "platforms": best.get("platforms", []),
        "tags": [
            {"name": t.get("name", ""), "rating": t.get("rating")}
            for t in best.get("tags", [])
        ],
        "characters": characters,
        "candidate_count": len(candidates),
    }


def run_search_cmd(search_cmd: str, title: str, *, timeout: int = 30) -> str:
    """可选外挂检索钩子：把 title 作为最后一个参数附加到命令后执行。"""
    if not search_cmd or not search_cmd.strip():
        return ""
    try:
        argv = shlex.split(search_cmd) + [title]
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (proc.stdout or "").strip()
    except Exception as exc:
        print(f"[warn] 外挂检索失败：{exc}", file=sys.stderr)
        return ""


def build_external_baseline(
    vndb: dict[str, Any], search_text: str = ""
) -> dict[str, Any]:
    """把外部原始返回规整为可对齐的结构化基线。"""
    if not vndb.get("found"):
        return {
            "available": False,
            "query": vndb.get("query", ""),
            "candidates": vndb.get("candidates", []),
            "search_snippets": search_text[:2000],
        }
    desc = vndb.get("description", "")
    return {
        "available": True,
        "source": "VNDB Kana API" + (" + search-cmd" if search_text else ""),
        "id": vndb.get("id"),
        "title": vndb.get("title"),
        "released": vndb.get("released"),
        "length_minutes": vndb.get("length_minutes"),
        "platforms": vndb.get("platforms", []),
        "tags": vndb.get("tags", []),
        "characters": vndb.get("characters", []),
        "official_synopsis": re.sub(r"\[[^\]]*\]", "", desc).strip()[:3000],
        "search_snippets": search_text[:2000],
    }


def _matched_local_names(plot: dict[str, Any]) -> str:
    names = [str(c.get("name", "")) for c in plot.get("characters", [])]
    names += [str(c.get("name_zh", "")) for c in plot.get("characters", [])]
    for s in plot.get("scenes", []):
        names += [str(c) for c in s.get("characters", [])]
    return " | ".join(dict.fromkeys(n for n in names if n)).lower()


def _character_present(ch: dict[str, Any] | str, local_blob: str) -> bool:
    """判断 VNDB 角色是否出现在本地台词/场景卡里。

    对齐日文原名 original（拆姓/名）、罗马字 name、aliases 与 glossary 中文译名，
    彻底消除把 Nanako / Kikumaru / Io / Iris 误判为「视频未覆盖」的假告警。
    """
    blob = _norm(local_blob)
    if not blob:
        return False
    for term in character_match_terms(ch):
        nt = _norm(term)
        if nt and nt in blob:
            return True
    # 反向兜底：本地出现中文译名，且该角色的罗马字/日文名命中译名表别名。
    if isinstance(ch, dict):
        low_name = str(ch.get("name", "") or "").lower()
        low_orig = str(ch.get("original", "") or "")
        for zh, romaji_list in NAME_ALIASES.items():
            if _norm(zh) in blob and (
                any(a.lower() in low_name for a in romaji_list) or zh in low_orig
            ):
                return True
    return False


def _scene_lines_for(
    names: list[dict[str, Any] | str], plot: dict[str, Any]
) -> list[int]:
    for s in plot.get("scenes", []):
        chars = [str(c) for c in s.get("characters", [])]
        blob = " | ".join(chars).lower()
        if any(_character_present(n, blob) for n in names):
            return [int(s.get("source_lines", [0, 0])[0]), int(s.get("source_lines", [0, 0])[1])]
    return []


def compute_duration_minutes(
    script_lines: list[dict[str, Any]],
) -> tuple[float, dict[int, float]]:
    """按每个 part 取 max(ts_sec) 后跨全部 part 求和，返回 (总分钟, 每 part 分钟)。

    ts_sec 为「该 P 内经过秒数」，故单 P 时长 = max(ts_sec)，全片时长 = 各 P 之和。
    旧实现用全局 max-min（仅相当于单个 P 跨度），9P 实测被低估为 134 分钟；
    求和后应约 733 分钟，与 VNDB length_minutes=608 同量级。
    """
    per_part_max: dict[Any, float] = {}
    for row in script_lines:
        part = row.get("part", 1)
        try:
            ts = float(row.get("ts_sec", 0) or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if part not in per_part_max or ts > per_part_max[part]:
            per_part_max[part] = ts
    detail = {int(p): round(v / 60.0, 2) for p, v in per_part_max.items()}
    total = round(sum(per_part_max.values()) / 60.0, 2)
    return total, detail


def cross_compare(
    plot: dict[str, Any],
    baseline: dict[str, Any],
    *,
    script_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """三方对齐：外部基线 vs 本地归约结论 -> 带徽章的 Diff 项。"""
    items: list[dict[str, Any]] = []
    counter = 0

    def add(
        dimension: str,
        claim: str,
        external: str,
        local: str,
        status: str,
        note: str = "",
        source_lines: list[int] | None = None,
    ) -> None:
        nonlocal counter
        counter += 1
        items.append(
            {
                "id": f"CC-{counter:03d}",
                "dimension": dimension,
                "claim": claim,
                "external_evidence": external,
                "local_evidence": local,
                "status": status,
                "badge": BADGES[status],
                "note": note,
                "source_lines": source_lines or [],
            }
        )

    if not baseline.get("available"):
        add(
            "外部基线",
            "联网检索可用性",
            "VNDB / 外挂检索均未返回有效结果",
            "仅本地归约",
            "uncovered",
            "外部源不可达，无法做双源印证",
        )
        return _finalize(plot, baseline, items)

    local_blob = _matched_local_names(plot)
    local_title = str(plot.get("work_title", ""))

    # 1) 作品身份锚定
    ext_title = str(baseline.get("title", ""))
    local_nums = set(re.findall(r"\d+", local_title))
    ext_nums = set(re.findall(r"\d+", ext_title))
    title_ok = _norm(ext_title) == _norm(local_title) or bool(
        local_nums & ext_nums
    )
    if title_ok:
        add(
            "作品身份",
            f"目标作品：{local_title or ext_title}",
            f"VNDB {baseline.get('id')} {ext_title}（{baseline.get('released')}）",
            local_title or "本地未标注标题",
            "verified",
            "官方条目唯一锚定，后续比对基于此条目",
        )
    else:
        src = _scene_lines_for([], plot)
        add(
            "作品身份",
            f"目标作品：{local_title}",
            f"VNDB {ext_title}",
            local_title,
            "conflict",
            "标题主编号不一致，疑似对齐到错误条目，需人工复核",
            src,
        )

    # 2) 角色阵容覆盖
    ext_chars = baseline.get("characters", [])
    main_chars = [
        c
        for c in ext_chars
        if any(r in ("main", "primary") for r in c.get("roles", []))
    ][:8] or ext_chars[:8]
    missing: list[dict[str, Any]] = []
    for ch in main_chars:
        name = str(ch.get("name", "") or "")
        original = str(ch.get("original", "") or "")
        label = original or name
        if not name and not original:
            continue
        if _character_present(ch, local_blob):
            add(
                "角色阵容",
                f"角色 {label} 是否出场",
                f"VNDB 角色 {name}（原名 {original}，{', '.join(ch.get('roles', []))}）",
                "本地归约命中该角色日文原名 / 罗马字 / 别名 / 中文译名",
                "verified",
            )
        else:
            missing.append(ch)
    if missing:
        missing_labels = [
            str(c.get("original") or c.get("name") or "") for c in missing[:6]
        ]
        add(
            "角色阵容",
            f"外部角色未覆盖：{', '.join(missing_labels)}",
            f"VNDB 列出主要角色 {len(main_chars)} 位",
            "本地场景卡/角色档案未提及",
            "uncovered",
            "可能对应多路线 / 未录屏角色，主动提醒读者",
            _scene_lines_for(missing[:2], plot),
        )

    # 3) 时长覆盖（跨 P 累加：每 P 取 max(ts_sec)，再对全部 P 求和）
    ext_len = baseline.get("length_minutes")
    if ext_len:
        span_min, per_part_min = compute_duration_minutes(script_lines or [])
        ratio = round(span_min / float(ext_len), 3)
        detail = " ".join(f"P{p}:{v}min" for p, v in sorted(per_part_min.items()))
        if span_min >= 0.6 * float(ext_len):
            add(
                "时长覆盖",
                f"视频覆盖 {span_min:.0f}/{ext_len} 分钟",
                f"VNDB 标称时长 {ext_len} 分钟",
                f"9P 台词时长求和 {span_min:.0f} 分钟（{ratio:.0%}）{detail}",
                "verified",
            )
        else:
            add(
                "时长覆盖",
                f"视频仅覆盖 {span_min:.0f}/{ext_len} 分钟",
                f"VNDB 标称时长 {ext_len} 分钟",
                f"台词时长求和 {span_min:.0f} 分钟（{ratio:.0%}）{detail}",
                "uncovered",
                "存在未覆盖时长，可能为其他路线 / 前作内容 / 纯音乐段落",
            )

    # 4) 标签与成人内容提示
    tags = [str(t.get("name", "")) for t in baseline.get("tags", [])]
    adult = [t for t in tags if any(h in t.lower() for h in ADULT_TAG_HINTS)]
    if adult:
        add(
            "内容分级",
            f"外部标注成人向标签：{', '.join(adult[:4])}",
            f"VNDB tags: {', '.join(tags[:8])}",
            "本地归约未标注 H 场景",
            "uncovered",
            "当前视频可能为全年龄版，H / 多结局分支未覆盖",
        )
    elif tags:
        add(
            "内容分级",
            f"全年龄向标签：{', '.join(tags[:4])}",
            f"VNDB tags: {', '.join(tags[:8])}",
            "本地场景卡未提示成人分支",
            "verified",
        )

    # 5) 官方简介关键设定是否体现
    synopsis = str(baseline.get("official_synopsis", ""))
    keywords = [
        w
        for w in re.findall(r"[A-Za-z]{4,}", synopsis)
        if w.lower() not in {"after", "learns", "that", "with", "from", "this", "what", "about"}
    ][:8]
    local_text = " ".join(
        str(s.get("summary_zh", "")) for s in plot.get("scenes", [])
    ).lower()
    hits = [w for w in keywords if w.lower() in local_text]
    if keywords and not hits:
        add(
            "剧情要点",
            "官方简介关键设定未在本视频台词中体现",
            f"VNDB 简介关键词：{', '.join(keywords[:6])}",
            "本地场景摘要 0 命中",
            "uncovered",
            "可能涉及前作设定 / 未录屏分支，建议结合外挂检索补证",
        )
    elif hits:
        add(
            "剧情要点",
            f"官方设定命中：{', '.join(hits[:5])}",
            f"VNDB 简介关键词：{', '.join(keywords[:6])}",
            "本地场景摘要命中相关叙事元素",
            "verified",
        )

    # 6) 冲突规则（明确矛盾才报 ❌，附原文行号）
    for s in plot.get("scenes", []):
        text = str(s.get("summary_zh", ""))
        if "普通民宅" in text and ("ninja" in synopsis.lower() or "忍" in text):
            add(
                "据点设定",
                "本地摘要称「普通民宅」",
                "VNDB 官方简介含忍者/屋敷设定",
                text[:120],
                "conflict",
                "与官方设定矛盾，回溯原文行号复核",
                [int(s.get("source_lines", [0, 0])[0]), int(s.get("source_lines", [0, 0])[1])],
            )
            break

    return _finalize(plot, baseline, items)


def _finalize(
    plot: dict[str, Any], baseline: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = {k: sum(1 for i in items if i["status"] == k) for k in BADGES}
    return {
        "work_title": plot.get("work_title", ""),
        "bvid": plot.get("bvid", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "external_baseline": baseline,
        "items": items,
        "summary": {**counts, "total": len(items)},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    plot_path = (
        Path(args.plot) if args.plot else work_dir / "output" / "plot_structured.json"
    )
    out_path = (
        Path(args.output)
        if args.output
        else work_dir / "output" / "cross_check.json"
    )
    plot = json.loads(plot_path.read_text(encoding="utf-8"))
    # 时长统计必须从带 part 字段的原始 script.jsonl 读取（双语文件是同一批行 +
    # text_zh，亦可回退）。旧版只读 bilingual 且用全局 max-min，导致跨 P 时长被低估。
    script_lines: list[dict[str, Any]] = []
    src_path = work_dir / "output" / "script.jsonl"
    bi_path = work_dir / "output" / "script_bilingual.jsonl"
    dur_path = src_path if src_path.exists() else bi_path
    if dur_path.exists():
        with dur_path.open("r", encoding="utf-8") as fh:
            script_lines = [json.loads(l) for l in fh if l.strip()]

    if args.no_network:
        vndb = {"found": False, "query": args.title, "candidates": []}
    else:
        vndb = fetch_vndb(args.title, base_url=args.vndb_url, timeout=args.timeout)
    search_text = run_search_cmd(args.search_cmd, args.title) if args.search_cmd else ""
    baseline = build_external_baseline(vndb, search_text)

    if baseline.get("available"):
        print(
            f"[VNDB] {baseline['id']} {baseline['title']} "
            f"released={baseline['released']} length={baseline['length_minutes']}min "
            f"tags={len(baseline['tags'])} chars={len(baseline['characters'])}"
        )
    else:
        print("[VNDB] 外部基线不可用（--no-network 或检索失败）")

    result = cross_compare(plot, baseline, script_lines=script_lines)
    write_json(out_path, result)
    s = result["summary"]
    print(
        f"[CrossCheck] total={s['total']} verified={s['verified']} "
        f"uncovered={s['uncovered']} conflict={s['conflict']} -> {out_path}"
    )
    for item in result["items"]:
        print(f"  {item['badge']} [{item['dimension']}] {item['claim']}")
    return result


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VNDB 联网交叉校验")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--plot", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--title", default="猫忍えくすはーとSPIN! 2")
    p.add_argument("--vndb-url", default=os.environ.get("VNDB_API", VNDB_API))
    p.add_argument("--search-cmd", default="", help="外挂检索命令，title 作为末参附加")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--no-network", action="store_true", help="离线单测用，不发网络请求")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
