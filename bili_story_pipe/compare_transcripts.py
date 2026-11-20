#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_transcripts.py -- LAB-2129 新旧台词提取效果对比评测脚本.

Usage::

    python3 compare_transcripts.py \
        --new-jsonl <script.jsonl> \
        --old-dir <P1..P9.txt 目录> \
        --output <diff_report.md>

读取新版 ``script.jsonl``（LAB-2125/2129 契约）与旧版 ``P1.txt..P9.txt``
（LAB-2120 VLM 底稿），产出 Markdown 对比报告：分 P 行数/置信度/时间戳覆盖、
噪音率、核心角色频次，以及「新版捕获而旧版 fps=1/5 漏掉」的样本台词。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:  # 作为包导入时复用 extract_ocr 的过滤规则
    from .extract_ocr import is_overlay_noise
except ImportError:  # 直接 python3 compare_transcripts.py 运行时

    def is_overlay_noise(text: str) -> bool:  # type: ignore[misc]
        tokens = ("翻译结束", "翻译结果", "耗时：", "耗时:", "利时：", "利时:")
        return bool(text) and any(tok in str(text) for tok in tokens)


# 核心角色（含常见别名/跨译名写法），键为报告中展示的角色名。
CORE_CHARACTERS: dict[str, tuple[str, ...]] = {
    "藤花": ("藤花",),
    "菊丸": ("菊丸",),
    "ナナコ": ("ナナコ", "Nanako"),
    "チカ": ("チカ", "千花"),
    "伊緒": ("伊緒", "Io"),
    "北条/春希": ("北条", "春希"),
    "ゆら": ("ゆら",),
    "たま": ("たま",),
    "アイリス": ("アイリス", "Iris"),
    "蛭子": ("蛭子",),
}

# LAB-2120 实查底稿含 76 处 VLM 批注噪音（不明 / 枚目は… / 第一条）。
VLM_NOISE_TOKENS = ("枚目は", "第一条", "不明")
OVERLAY_LABEL = "机翻状态栏残留"
VLM_LABEL = "VLM 批注"
_PUNCT_RE = re.compile(r"[\s\u3000，。、！？!?,.\-—…「」『』（）()\[\]【】:：;；~〜]")


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def parse_new_jsonl(path: str | Path) -> list[dict]:
    """Read a contract-compliant ``script.jsonl`` into a list of dicts."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("text_ja", "").strip():
                out.append(row)
    return out


def parse_old_dir(old_dir: str | Path) -> dict[int, list[str]]:
    """Read ``P*.txt`` transcripts into ``{part: [raw_line, ...]}``."""
    root = Path(old_dir)
    result: dict[int, list[str]] = {}
    if not root.is_dir():
        return result
    for txt in root.glob("P*.txt"):
        m = re.search(r"P(\d+)", txt.name)
        if not m:
            continue
        part = int(m.group(1))
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        result[part] = lines
    return result


def clean_old_line(line: str) -> tuple[str, str]:
    """Split an old ``[角色] 台词`` / ``角色：台词`` line into (speaker, text)."""
    s = line.strip()
    m = re.match(r"^\[([^\]]+)\]\s*[:：]?\s*(.*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^([^：:\[\]]{1,12})[：:]\s*(.*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", s


def normalize_text(text: str) -> str:
    """Normalize away whitespace/punctuation for cross-version text matching."""
    return _PUNCT_RE.sub("", str(text)).lower()


def is_old_vlm_noise(line: str) -> bool:
    return any(tok in line for tok in VLM_NOISE_TOKENS)


def _mmss(ts_sec: float) -> str:
    ts = max(0, int(float(ts_sec)))
    return f"{ts // 60:02d}:{ts % 60:02d}"


def _pct(new: int, old: int) -> str:
    if old > 0:
        delta = (new - old) / old * 100.0
        return f"{delta:+.1f}%"
    if new > 0:
        return "新增"
    return "0.0%"


def _rate(noise: int, total: int) -> str:
    return f"{(noise / total * 100.0):.1f}%" if total else "0.0%"


def _char_count(text: str, aliases: tuple[str, ...]) -> int:
    return sum(str(text).count(alias) for alias in aliases)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(
    new_lines: list[dict],
    old_by_part: dict[int, list[str]],
    new_path: str = "",
    old_dir: str = "",
) -> str:
    """Build the full Markdown comparison report as a string."""
    new_by_part: dict[int, list[dict]] = {}
    for row in new_lines:
        try:
            part = int(row.get("part", 0))
        except (TypeError, ValueError):
            continue
        new_by_part.setdefault(part, []).append(row)

    lines: list[str] = []
    lines.append("# 新旧台词提取效果对比报告")
    lines.append("")
    lines.append(f"- 新版脚本：`{new_path or 'script.jsonl'}`")
    lines.append(f"- 旧版台词目录：`{old_dir or 'P1..P9.txt'}`")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ---- 1. 分 P 行数对比 ----
    lines.append("## 1. 分 P（P1–P9）新旧行数对比")
    lines.append("")
    lines.append("| 分 P | 新行数 | 旧行数 | 增幅 | 平均置信度 ↑ | 时间戳覆盖范围 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for part in range(1, 10):
        nrows = new_by_part.get(part, [])
        orows = old_by_part.get(part, [])
        confs = []
        for r in nrows:
            try:
                confs.append(float(r.get("conf", 1.0)))
            except (TypeError, ValueError):
                pass
        avg_conf = f"{sum(confs) / len(confs):.3f}" if confs else "—"
        tss = []
        for r in nrows:
            try:
                tss.append(float(r.get("ts_sec", 0.0)))
            except (TypeError, ValueError):
                pass
        coverage = f"{_mmss(min(tss))}–{_mmss(max(tss))}" if tss else "—"
        lines.append(
            f"| P{part} | {len(nrows)} | {len(orows)} | {_pct(len(nrows), len(orows))} | {avg_conf} | {coverage} |"
        )
    lines.append("")

    # ---- 2. 噪音率对比 ----
    old_vlm = sum(1 for rows in old_by_part.values() for ln in rows if is_old_vlm_noise(ln))
    old_overlay = sum(
        1
        for rows in old_by_part.values()
        for ln in rows
        if is_overlay_noise(clean_old_line(ln)[1])
    )
    old_total = sum(len(rows) for rows in old_by_part.values())
    new_vlm = sum(1 for row in new_lines if any(tok in str(row.get("text_ja", "")) for tok in VLM_NOISE_TOKENS))
    new_overlay = sum(1 for row in new_lines if is_overlay_noise(str(row.get("text_ja", ""))))
    new_total = len(new_lines)
    lines.append("## 2. 噪音率对比")
    lines.append("")
    lines.append("| 版本 | VLM 批注行 ↓ | 机翻状态栏残留行 ↓ | 总行数 | 噪音率 ↓ |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| 旧版（fps=1/5） | {old_vlm} | {old_overlay} | {old_total} | "
        f"{_rate(old_vlm + old_overlay, old_total)} |"
    )
    lines.append(
        f"| 新版（script.jsonl） | {new_vlm} | {new_overlay} | {new_total} | "
        f"{_rate(new_vlm + new_overlay, new_total)} |"
    )
    lines.append("")

    # ---- 3. 核心角色频次 ----
    new_text = "\n".join(f"{r.get('speaker', '')} {r.get('text_ja', '')}" for r in new_lines)
    old_text = "\n".join(ln for rows in old_by_part.values() for ln in rows)
    lines.append("## 3. 核心角色名出现频次对比")
    lines.append("")
    lines.append("| 角色 | 新版出现次数 ↑ | 旧版出现次数 ↑ | 变化 |")
    lines.append("|---|---:|---:|---:|")
    for name, aliases in CORE_CHARACTERS.items():
        n = _char_count(new_text, aliases)
        o = _char_count(old_text, aliases)
        delta = n - o
        lines.append(f"| {name} | {n} | {o} | {delta:+d} |")
    lines.append("")

    # ---- 4. 新版新捕获样本 ----
    old_norm: set[str] = set()
    for rows in old_by_part.values():
        for ln in rows:
            norm = normalize_text(clean_old_line(ln)[1])
            if norm:
                old_norm.add(norm)

    candidates = []
    for row in new_lines:
        norm = normalize_text(str(row.get("text_ja", "")))
        if not norm or norm in old_norm:
            continue
        candidates.append(row)
    # 短句优先，其次按分 P / 时间排序，便于人工复核「fps=1/5 漏掉的短句」。
    candidates.sort(key=lambda r: (len(str(r.get("text_ja", ""))), int(r.get("part", 0) or 0), float(r.get("ts_sec", 0) or 0)))

    lines.append("## 4. 新版新捕获样本（旧版 fps=1/5 漏掉的短句/关键台词）")
    lines.append("")
    seen: set[str] = set()
    picked = 0
    for row in candidates:
        norm = normalize_text(str(row.get("text_ja", "")))
        if norm in seen:
            continue
        seen.add(norm)
        picked += 1
        try:
            part = int(row.get("part", 0))
            ts = float(row.get("ts_sec", 0.0))
        except (TypeError, ValueError):
            continue
        speaker = str(row.get("speaker", "") or "").strip()
        prefix = f"{speaker}：" if speaker else ""
        lines.append(f"{picked}. `[P{part} {_mmss(ts)}]` {prefix}{row.get('text_ja', '')}")
        if picked >= 10:
            break
    if picked == 0:
        lines.append("（本样例中未发现旧版完全漏掉的新句）")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LAB-2129 新旧台词提取效果对比")
    p.add_argument("--new-jsonl", required=True, help="新版 script.jsonl 路径")
    p.add_argument("--old-dir", required=True, help="旧版 P1.txt..P9.txt 所在目录")
    p.add_argument("--output", required=True, help="输出 Markdown 报告路径")
    args = p.parse_args(argv)

    new_lines = parse_new_jsonl(args.new_jsonl)
    old_by_part = parse_old_dir(args.old_dir)
    report = build_report(new_lines, old_by_part, args.new_jsonl, args.old_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[compare_transcripts] new_lines={len(new_lines)} old_parts={len(old_by_part)} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
