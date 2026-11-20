#!/usr/bin/env python3
"""模块 B v2 实测缺陷修复单测。

覆盖 LAB-2130 三处精度问题的回归验证：
  1. VNDB Kana API 拉取日文原名 original + 跨语言角色对齐（消除 Nanako/Kikumaru/
     Io/Iris 误报「视频未覆盖」）；
  2. 跨 9P 时长按 part 取 max(ts_sec) 后求和（9P ~735 min > 600，标 verified）；
  3. summarize_mr 紧凑场景卡 JSON + strict=False 容错 + 旁白/ナレーション 审计降噪。

运行：pytest -v tests/test_analyze_v2.py
其中 test_vndb_original_names_and_matching 为真实公网单测（api.vndb.org）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bili_story_pipe import build_viewer, cross_check, summarize_mr  # noqa: E402

TITLE = "猫忍えくすはーとSPIN! 2"
BVID = "BV13JSCBAEDC"
V58009_LENGTH = 608

# 9 个分 P 的实测比例：单 P ≈ 4900 s ≈ 81.67 min，9P 求和 ≈ 735 min。
SPANS = {
    "ナナコ": ("Makibishi Nanako", "巻菱 ナナコ"),
    "菊丸": ("Makibishi Kikumaru", "巻菱 菊丸"),
    "伊緒": ("Kouga Io", "甲賀 伊緒"),
    "アイリス": ("Iris", "アイリス"),
    "藤花": ("Touka", "藤花"),
    "ゆら": ("Fuuma Yura", "風魔 ゆら"),
}


def p9_rows(lines_per_part: int = 4) -> list[dict]:
    """构造 9P 台词，每 P 最大 ts_sec = 4900s，用于跨 P 时长求和验证。"""
    rows: list[dict] = []
    lid = 1
    speakers = list(SPANS.keys())
    for part in range(1, 10):
        for j in range(lines_per_part):
            rows.append(
                {
                    "line_id": lid,
                    "part": part,
                    "ts_sec": round(j * 4900.0 / max(1, lines_per_part - 1), 1),
                    "speaker": speakers[(lid - 1) % len(speakers)],
                    "text_ja": f"P{part} の台詞 {j}。",
                    "conf": 0.96,
                    "needs_review": False,
                }
            )
            lid += 1
    return rows


def run_module(module: str, *args: object, timeout: int = 180) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def v2_pipeline(tmp_path_factory):
    """端到端跑 9P 样例：mock 归约 -> 真实 VNDB 校验 -> 离线 HTML。"""
    work = tmp_path_factory.mktemp("bili_v2")
    out = work / "output"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "script.jsonl").open("w", encoding="utf-8") as fh:
        for row in p9_rows():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    r1 = run_module("bili_story_pipe.summarize_mr", "--work-dir", work, "--mock")
    assert r1.returncode == 0, f"summarize_mr 失败:\n{r1.stdout}\n{r1.stderr}"
    r2 = run_module("bili_story_pipe.cross_check", "--work-dir", work, "--title", TITLE)
    assert r2.returncode == 0, f"cross_check 失败:\n{r2.stdout}\n{r2.stderr}"
    r3 = run_module(
        "bili_story_pipe.build_viewer", "--work-dir", work, "--output", "demo.html"
    )
    assert r3.returncode == 0, f"build_viewer 失败:\n{r3.stdout}\n{r3.stderr}"
    print("\n[summarize_mr]", r1.stdout.strip())
    print("[cross_check]", r2.stdout.strip())
    print("[build_viewer]", r3.stdout.strip())
    return work


# --------------------------------------------------------------------------- #
# 1. 真实 VNDB Kana API：日文原名 + 跨语言角色对齐（核心）
# --------------------------------------------------------------------------- #
def test_vndb_original_names_and_matching():
    vndb = None
    last_err = None
    for _ in range(3):  # 公网偶发抖动，重试
        try:
            vndb = cross_check.fetch_vndb(TITLE, timeout=30)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    assert vndb is not None, f"VNDB API 三次均失败：{last_err}"
    assert vndb["found"] is True
    assert vndb["id"] == "v58009"
    assert vndb["length_minutes"] == V58009_LENGTH
    assert "SPIN! 2" in vndb["title"]

    chars = vndb["characters"]
    assert chars, "角色端点未返回角色"
    # 关键：字段里必须有 original（日文原名），并含 aliases 字段
    assert all("original" in c for c in chars)
    assert all("aliases" in c for c in chars)
    originals = {c["original"] for c in chars}
    print("\n[VNDB 日文原名实测]", sorted(originals))

    matched: list[str] = []
    for given_ja, (romaji, full_ja) in SPANS.items():
        # 台词里只有「名」（ナナコ / 菊丸 / 伊緒 / アイリス / 藤花 / ゆら）
        ch = next((c for c in chars if full_ja == c.get("original")), None)
        assert ch is not None, f"VNDB 缺少角色原名 {full_ja}（romaji={romaji}）"
        assert full_ja in originals
        # 日文名（given）与全名（姓+名）都应命中
        blob = f"{given_ja}が登場した、{full_ja}と話す。"
        assert cross_check._character_present(ch, blob), (
            f"日文原名对齐失败：{full_ja} vs {blob}"
        )
        # 罗马字名也应命中
        assert cross_check._character_present(ch, romaji)
        # 证实「光靠罗马字 name 必然漏」——这是原缺陷的根因
        assert ch["name"].lower() not in blob.lower(), (
            f"罗马字意外出现在纯日文台词里，失去回归意义：{ch['name']}"
        )
        matched.append(f"{romaji}/{full_ja}✅")
    print("[角色对齐实测]", " ".join(matched))


# --------------------------------------------------------------------------- #
# 2. 跨 9P 时长求和 + verified 徽章
# --------------------------------------------------------------------------- #
def test_cross_p_duration_sum():
    rows = p9_rows()
    total, detail = cross_check.compute_duration_minutes(rows)
    assert total > 600, f"跨 P 时长求和异常：{total} min"
    assert len(detail) == 9
    assert all(v > 70 for v in detail.values()), detail

    # 旧实现（全局 max-min 只相当于单 P 跨度）被量化证伪
    all_ts = [r["ts_sec"] for r in rows]
    old_span = (max(all_ts) - min(all_ts)) / 60.0
    assert old_span < 100, old_span
    assert total > old_span * 5, (total, old_span)
    print(f"\n[时长修复实测] 旧跨度 {old_span:.1f}min -> 跨9P求和 {total:.1f}min")


def test_duration_marked_verified():
    baseline = {
        "available": True,
        "source": "VNDB Kana API (synthetic length)",
        "id": "v58009",
        "title": "Neko-nin exHeart SPIN! 2",
        "released": "2025-11-28",
        "length_minutes": V58009_LENGTH,
        "tags": [],
        "characters": [],
        "official_synopsis": "",
    }
    plot = {"work_title": TITLE, "bvid": BVID, "characters": [], "scenes": []}
    result = cross_check.cross_compare(plot, baseline, script_lines=p9_rows())
    items = [i for i in result["items"] if i["dimension"] == "时长覆盖"]
    assert items, "缺少时长覆盖核验项"
    assert items[0]["status"] == "verified", items[0]
    assert "9P" in items[0]["local_evidence"]
    print("[时长徽章实测]", items[0]["claim"], items[0]["badge"])


# --------------------------------------------------------------------------- #
# 3. summarize_mr：strict=False 容错 + 覆盖率审计降噪
# --------------------------------------------------------------------------- #
def test_extract_json_object_strict_false():
    # 原始换行控制字符直接出现在 JSON 字符串里（strict=True 会抛 JSONDecodeError）
    raw = '```json\n{"title": "场景\nA", "summary_zh": "第一行\n第二行"}\n```'
    parsed = summarize_mr.extract_json_object(raw)
    assert parsed["title"] == "场景\nA"
    assert parsed["summary_zh"] == "第一行\n第二行"


def test_compact_scene_card_contract():
    """mock 场景卡只含紧凑字段，不含 translations 大对象。"""
    chunk = {
        "part": 1,
        "lines": p9_rows(6)[:6],
        "source_lines": [1, 6],
        "ts_start": 0.0,
        "ts_end": 4900.0,
    }
    scene = summarize_mr.build_mock_scene(chunk, 1, summarize_mr.BUILTIN_GLOSSARY)
    assert {"title", "summary_zh", "characters", "quotes"} <= set(scene)
    assert "translations" not in scene, "Map 场景卡不应携带 translations 大对象"
    assert all({"speaker", "text_ja", "text_zh"} <= set(q) for q in scene["quotes"])


def test_coverage_audit_denoise():
    lines: list[dict] = []
    lid = 1
    for speaker, n in [("旁白", 20), ("ナレーション", 15), ("???", 10), ("不明", 8), ("藤花", 6)]:
        for _ in range(n):
            lines.append(
                {"line_id": lid, "part": 1, "ts_sec": lid * 10.0,
                 "speaker": speaker, "text_ja": "x"}
            )
            lid += 1
    scenes = [
        {"scene_id": "S1", "characters": ["藤花"], "summary_zh": "藤花登场",
         "quotes": [{"speaker": "藤花"}], "source_lines": [1, 59]}
    ]
    audit = summarize_mr.coverage_audit(lines, scenes)
    names = [r["speaker"] for r in audit["top_speakers"]]
    assert names == ["藤花"], f"元标签未排除：{names}"
    assert not any(a.get("type") == "high_freq_speaker_missing" for a in audit["alerts"])
    assert audit["red_light"] is False

    # 直接校验过滤判定
    for tag in ["旁白", "ナレーション", "???", "不明", "", "  "]:
        assert summarize_mr.is_real_character(tag) is False
    assert summarize_mr.is_real_character("藤花") is True


# --------------------------------------------------------------------------- #
# 4. 端到端（mock 归约 + 真实 VNDB + HTML），验证产物契约未被破坏
# --------------------------------------------------------------------------- #
def test_v2_pipeline_end_to_end(v2_pipeline):
    out = v2_pipeline / "output"

    bilingual = [json.loads(l) for l in (out / "script_bilingual.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(bilingual) == 36
    assert all(r.get("text_zh") for r in bilingual)

    cc = json.loads((out / "cross_check.json").read_text(encoding="utf-8"))
    assert cc["external_baseline"]["id"] == "v58009"
    dur = [i for i in cc["items"] if i["dimension"] == "时长覆盖"]
    assert dur and dur[0]["status"] == "verified", dur
    # VNDB 角色项不应再出现「Nanako/Kikumaru/Io/Iris 未覆盖」假告警
    missing = [i for i in cc["items"] if i["dimension"] == "角色阵容" and i["status"] == "uncovered"]
    print("\n[角色未覆盖项实测]", [i["claim"] for i in missing])

    plot = json.loads((out / "plot_structured.json").read_text(encoding="utf-8"))
    top = [r["speaker"] for r in plot["coverage_audit"]["top_speakers"]]
    assert "旁白" not in top and "ナレーション" not in top

    html = (v2_pipeline / "demo.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert '<script id="viewer-data" type="application/json">' in html
