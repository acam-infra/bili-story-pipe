#!/usr/bin/env python3
"""模块 B 完整单元测试：mock 归约 / 覆盖率红灯 / 真实 VNDB API / 离线 HTML 浏览器。

运行：pytest -v tests/test_analyze_viewer.py
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

SPEAKERS = ["藤花", "北条", "由良", "爱丽丝", "蛭子"]
SAMPLE_TEXT = [
    "神話の時代、神は人と共にあった。",
    "猫忍の奥義、ここで見せてあげるわ。",
    "北条「そうですか。わかった、行こう」",
    "由良が忍者屋敷の秘密を語り始める。",
    "爱丽丝は少し困った顔をしていた。",
    "蛭子の影が静かに近づいてくる。",
    "藤花「ありがとう、今日は本当に」",
    "千花が猫耳を揺らして飛び込んできた。",
    "にゃー、ここが私の居場所だから。",
    "物語は新しい章へと続いていく。",
]


def sample_rows(n: int = 60) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "line_id": i,
                "part": (i - 1) // 7 + 1,
                "ts_sec": round(i * 12.5, 1),
                "speaker": SPEAKERS[(i - 1) % len(SPEAKERS)],
                "text_ja": SAMPLE_TEXT[(i - 1) % len(SAMPLE_TEXT)],
                "conf": 0.96,
                "needs_review": False,
            }
        )
    return rows


def run_module(module: str, *args: object, timeout: int = 150) -> subprocess.CompletedProcess:
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
def pipeline(tmp_path_factory):
    """构造 60 条模拟台词并跑完整链路（mock 归约 + 真实 VNDB 校验 + HTML）。"""
    work = tmp_path_factory.mktemp("bili_pipeline")
    out = work / "output"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "script.jsonl").open("w", encoding="utf-8") as fh:
        for row in sample_rows(60):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    r1 = run_module("bili_story_pipe.summarize_mr", "--work-dir", work, "--mock")
    assert r1.returncode == 0, f"summarize_mr 失败:\n{r1.stdout}\n{r1.stderr}"
    r2 = run_module(
        "bili_story_pipe.cross_check",
        "--work-dir",
        work,
        "--title",
        "猫忍えくすはーとSPIN! 2",
    )
    assert r2.returncode == 0, f"cross_check 失败:\n{r2.stdout}\n{r2.stderr}"
    r3 = run_module(
        "bili_story_pipe.build_viewer",
        "--work-dir",
        work,
        "--output",
        "demo_viewer.html",
    )
    assert r3.returncode == 0, f"build_viewer 失败:\n{r3.stdout}\n{r3.stderr}"
    print("\n[summarize_mr]", r1.stdout.strip())
    print("[cross_check]", r2.stdout.strip())
    print("[build_viewer]", r3.stdout.strip())
    return work


# --------------------------------------------------------------------------- #
# 1. summarize_mr --mock
# --------------------------------------------------------------------------- #
def test_summarize_mock(pipeline):
    out = pipeline / "output"
    bi = out / "script_bilingual.jsonl"
    plot_p = out / "plot_structured.json"
    assert bi.exists() and plot_p.exists()

    bilingual = [json.loads(l) for l in bi.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(bilingual) == 60
    assert all(r.get("text_zh") for r in bilingual)
    assert all({"line_id", "part", "ts_sec", "speaker", "text_ja"} <= set(r) for r in bilingual)

    plot = json.loads(plot_p.read_text(encoding="utf-8"))
    assert {"work_title", "bvid", "acts", "mermaid", "characters", "scenes", "coverage_audit"} <= set(plot)
    assert len(plot["acts"]) == 3
    assert len(plot["scenes"]) >= 9  # 9 个分 P 各至少一张
    assert plot["mermaid"].startswith("graph LR")
    assert len(plot["characters"]) >= 5
    assert plot["bvid"] == "BV13JSCBAEDC"

    for s in plot["scenes"][:5]:
        assert {"scene_id", "part", "ts_start", "ts_end", "title", "characters",
                "summary_zh", "quotes", "source_lines"} <= set(s)
        assert s["quotes"] and all({"speaker", "text_ja", "text_zh"} <= set(q) for q in s["quotes"])
        assert len(s["source_lines"]) == 2


# --------------------------------------------------------------------------- #
# 2. 覆盖率红灯审计（故意构造漏角色场景）
# --------------------------------------------------------------------------- #
def test_coverage_audit_detects_red_light():
    lines = []
    lid = 1
    for part in (1, 2, 3):
        for _ in range(10):
            lines.append({"line_id": lid, "part": part, "speaker": "藤花" if lid % 2 else "北条",
                          "text_ja": "x", "ts_sec": lid * 10.0})
            lid += 1
    # 再补一段高行号台词，制造一个未被场景卡覆盖的 500 行滑窗
    for lid in range(501, 506):
        lines.append({"line_id": lid, "part": 9, "speaker": "北条",
                      "text_ja": "x", "ts_sec": lid * 10.0})
    # 场景卡故意只提北条，不提高频角色藤花；且只覆盖 1-10 行
    scenes = [
        {"scene_id": "S1", "characters": ["北条"], "summary_zh": "只有北条",
         "quotes": [{"speaker": "北条"}], "source_lines": [1, 10]}
    ]
    audit = summarize_mr.coverage_audit(lines, scenes)
    assert audit["red_light"] is True
    types = [a["type"] for a in audit["alerts"]]
    assert "high_freq_speaker_missing" in types
    assert "line_window_uncovered" in types
    missing = next(a for a in audit["alerts"] if a["type"] == "high_freq_speaker_missing")
    assert missing["speaker"] == "藤花"
    assert missing["line_count"] >= 3

    # 完整覆盖且高频角色均被提及时不应亮红灯
    good_scenes = [
        {"scene_id": "S1", "characters": ["藤花", "北条"], "summary_zh": "藤花与北条对话",
         "quotes": [{"speaker": "藤花"}], "source_lines": [1, 10]},
        {"scene_id": "S2", "characters": ["藤花", "北条"], "summary_zh": "二人再会",
         "quotes": [], "source_lines": [11, 30]},
        {"scene_id": "S3", "characters": ["藤花", "北条"], "summary_zh": "尾声",
         "quotes": [], "source_lines": [501, 505]},
    ]
    assert summarize_mr.coverage_audit(lines, good_scenes)["red_light"] is False


# --------------------------------------------------------------------------- #
# 3. 真实 VNDB API 查询（公网实测）
# --------------------------------------------------------------------------- #
def test_vndb_api_live():
    vndb = None
    last_err = None
    for _ in range(2):
        try:
            vndb = cross_check.fetch_vndb("猫忍えくすはーとSPIN! 2", timeout=25)
            break
        except Exception as exc:  # 公网偶发抖动，重试一次
            last_err = exc
    assert vndb is not None, f"VNDB API 两次均失败：{last_err}"
    assert vndb["found"] is True
    assert vndb["id"] == "v58009"
    assert "SPIN! 2" in vndb["title"]
    assert vndb["released"] == "2025-11-28"
    assert vndb["length_minutes"] == 608
    assert vndb["description"].strip()
    assert len(vndb["characters"]) > 0
    print(
        f"\n[VNDB 实测] id={vndb['id']} title={vndb['title']} "
        f"released={vndb['released']} length={vndb['length_minutes']}min "
        f"tags={[t['name'] for t in vndb['tags']]} "
        f"chars={len(vndb['characters'])} "
        f"top_chars={[c['name'] for c in vndb['characters'][:5]]}"
    )


# --------------------------------------------------------------------------- #
# 4. 三方交叉校验输出 & 徽章
# --------------------------------------------------------------------------- #
def test_cross_check_output(pipeline):
    data = json.loads((pipeline / "output" / "cross_check.json").read_text(encoding="utf-8"))
    assert {"work_title", "bvid", "external_baseline", "items", "summary"} <= set(data)
    assert data["external_baseline"]["id"] == "v58009"
    assert data["items"]
    for item in data["items"]:
        assert item["status"] in cross_check.BADGES
        assert item["badge"] == cross_check.BADGES[item["status"]]
        assert isinstance(item["source_lines"], list)
    assert data["summary"]["verified"] >= 1
    assert data["summary"]["uncovered"] >= 1
    assert data["summary"]["total"] == len(data["items"])


def test_conflict_badge_on_title_mismatch():
    """标题编号矛盾时应回 ❌ 冲突告警，并附原文行号。"""
    baseline = {
        "available": True,
        "source": "synthetic",
        "id": "v58009",
        "title": "Neko-nin exHeart SPIN! 2",
        "released": "2025-11-28",
        "length_minutes": 608,
        "tags": [],
        "characters": [],
        "official_synopsis": "",
    }
    plot = {
        "work_title": "Neko-nin exHeart SPIN! 3",
        "bvid": "BV13JSCBAEDC",
        "characters": [],
        "scenes": [{"source_lines": [12, 40], "summary_zh": "普通民宅", "characters": []}],
    }
    result = cross_check.cross_compare(plot, baseline, script_lines=[])
    conflicts = [i for i in result["items"] if i["status"] == "conflict"]
    assert conflicts
    assert any(i["dimension"] == "作品身份" for i in conflicts)
    assert all(isinstance(i["source_lines"], list) for i in conflicts)


# --------------------------------------------------------------------------- #
# 5. 离线单页 HTML 浏览器
# --------------------------------------------------------------------------- #
def test_build_viewer(pipeline):
    html_path = pipeline / "demo_viewer.html"
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "交互式剧情浏览器" in text
    assert text.count("data-tab=") >= 3
    assert "<style>" in text and "</style>" in text
    assert "localStorage" in text and "theme-toggle" in text
    assert "mermaid" in text.lower()
    assert "https://cdn.jsdelivr.net/npm/mermaid" in text
    assert '<script id="viewer-data" type="application/json">' in text
    assert "https://www.bilibili.com/video/BV13JSCBAEDC/?p=1&t=" in text
    assert "✅ 双源印证" in text and "⚠️ 视频未覆盖" in text

    # 数据内嵌完整：60 条台词、5 个角色筛选项、9 个分 P
    payload = text.split('<script id="viewer-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(payload)
    assert len(data["transcript"]) == 60
    assert len(data["scenes"]) >= 9
    assert data["bvid"] == "BV13JSCBAEDC"
    assert set(SPEAKERS) <= set(data["speakers"])
    assert set(data["parts"]) == set(range(1, 10))
    for t in data["transcript"]:
        assert t["url"].startswith(f"https://www.bilibili.com/video/{data['bvid']}/?p=")
        assert f"&t={int(t['ts_sec'])}" in t["url"]
        assert len(t["ts"]) == 5 and t["ts"][2] == ":"


def test_bilibili_url_format():
    url = build_viewer.bilibili_url("BV13JSCBAEDC", 6, 872.9)
    assert url == "https://www.bilibili.com/video/BV13JSCBAEDC/?p=6&t=872"
