#!/usr/bin/env python3
"""LAB-2135：Tab 1 v4 改造单元测试。

覆盖：
  1. 角色名规范化 / OCR 变体归并（canonicalize_name + alias_map）；
  2. 深度角色档案 schema（mock 确定性 + LLM 路径）；
  3. 语义关系图（Top-15 规范节点 + 真实语义边标签，非机械共现）；
  4. causal_chains 跨章因果链；
  5. v4 HTML：node --check 零报错 + 无阻塞外链 + DOM 执行验证。

运行：pytest -v tests/test_tab1_v4.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bili_story_pipe import build_viewer, summarize_mr  # noqa: E402

# 真实台词中同一角色的 12 组 OCR / 简繁 / 日中文变体
RAW_NAMES = [
    "サトちゃん", "北条", "阳葵", "結衣", "チカ", "ナナコ",
    "アイリス", "華花", "由良", "玉", "伊縮", "猫坂永夜",
]
CANON_NAMES = {
    "サトミ", "北条春希", "陽葵", "結衣", "千花", "ナナコ",
    "アイリス", "藤花", "ゆら", "たま", "伊緒", "永夜",
}


def run_module(module: str, *args: object, timeout: int = 150) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


@pytest.fixture(scope="module")
def v4_pipeline(tmp_path_factory):
    work = tmp_path_factory.mktemp("bili_v4")
    out = work / "output"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    lid = 1
    for part in range(1, 10):
        for name in RAW_NAMES:
            rows.append(
                {
                    "line_id": lid, "part": part, "ts_sec": round(lid * 12.5, 1),
                    "speaker": name, "text_ja": f"P{part} {name} の台詞 {lid}。",
                    "conf": 0.96, "needs_review": False,
                }
            )
            lid += 1
    (out / "script.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    r = run_module("bili_story_pipe.summarize_mr", "--work-dir", work, "--mock")
    assert r.returncode == 0, f"summarize_mr --mock 失败:\n{r.stdout}\n{r.stderr}"
    return work


# --------------------------------------------------------------------------- #
# 1. 角色名规范化
# --------------------------------------------------------------------------- #
def test_canonicalize_name_alias_merge():
    sm = summarize_mr
    pairs = {
        "サトちゃん": "サトミ", "サトミ": "サトミ", "大忍サトミ": "サトミ",
        "北条": "北条春希", "春希": "北条春希", "北条春希": "北条春希",
        "阳葵": "陽葵", "雑賀陽葵": "陽葵", "雑賀 陽葵": "陽葵",
        "结衣": "結衣", "結衣": "結衣", "甲賀結衣": "結衣",
        "チカ": "千花", "千花": "千花",
        "ナナコ": "ナナコ", "奈奈子": "ナナコ",
        "アイリス": "アイリス", "爱丽丝": "アイリス", "アリス": "アイリス",
        "藤花": "藤花", "華花": "藤花",
        "ゆら": "ゆら", "由良": "ゆら",
        "たま": "たま", "玉": "たま",
        "伊緒": "伊緒", "伊绪": "伊緒", "伊縮": "伊緒", "伊织": "伊緒", "伊織": "伊緒",
    }
    for raw, canon in pairs.items():
        assert sm.canonicalize_name(raw) == canon, f"{raw} 应归并为 {canon}"
    # 非角色元标签不应进归并
    for tag in ("", "???", "旁白", "ナレーション", "不明"):
        assert sm.canonicalize_name(tag) == ""
    assert sm.is_real_character("サトミ") is True


def test_canonical_scene_and_speaker_helpers():
    sm = summarize_mr
    scene = {
        "characters": ["サトちゃん", "北条", "陽葵", "旁白"],
        "summary_zh": "サトミと北条春希が話す",
        "quotes": [{"speaker": "华花"}],
    }
    assert sm.canonical_scene_characters(scene) == ["サトミ", "北条春希", "陽葵"]
    assert sm.name_mentioned("藤花", scene) is True   # 华花 -> 藤花，出现在 quotes
    assert sm.name_mentioned("サトミ", scene) is True


# --------------------------------------------------------------------------- #
# 2. 深度角色档案 / 三幕 / 语义图 / 因果链（mock 端到端）
# --------------------------------------------------------------------------- #
def test_deep_profile_schema_mock(v4_pipeline):
    plot = json.loads((v4_pipeline / "output" / "plot_structured.json").read_text(encoding="utf-8"))
    chars = plot["characters"]
    names = {c["name"] for c in chars}
    assert CANON_NAMES <= names, f"规范角色缺失: {CANON_NAMES - names}"
    # OCR / 简繁变体绝不回流到档案名（已本身是规范名的除外）
    variant_only = {raw for raw in RAW_NAMES if summarize_mr.canonicalize_name(raw) != raw}
    assert variant_only & names == set(), f"变体名回流: {variant_only & names}"

    for c in chars:
        assert set(summarize_mr.PROFILE_FIELDS) <= set(c)
        for key in ("role_tag", "faction", "summary_zh", "motivation"):
            assert isinstance(c[key], str) and c[key].strip(), (c["name"], key)
        for key in ("key_interactions", "arc_highlights"):
            assert isinstance(c[key], list) and c[key] and all(str(x).strip() for x in c[key])
        assert "是推动本段剧情的关键角色之一" not in c["summary_zh"]
        # 兼容字段仍在（cross_check 依赖 name_zh）
        assert {"name_zh", "line_count", "scene_count", "parts"} <= set(c)

    # 三幕：非箭头机械拼接
    acts = plot["acts"]
    assert len(acts) == 3
    for a in acts:
        assert a["summary_zh"].strip() and " → " not in a["summary_zh"]
        assert a["scene_ids"]


def test_semantic_mermaid_and_causal_chains(v4_pipeline):
    plot = json.loads((v4_pipeline / "output" / "plot_structured.json").read_text(encoding="utf-8"))
    mermaid = plot["mermaid"]
    assert mermaid.startswith("graph LR")
    lines = mermaid.splitlines()
    nodes = [l for l in lines if re.match(r'^\s+C\d+\["', l)]
    edges = [l for l in lines if "---|" in l]
    assert 1 <= len(nodes) <= 15, nodes
    assert edges, "语义图应有关系边"
    for l in edges:
        assert "共现" not in l, f"不应出现机械共现标签: {l}"
        assert re.search(r'---\|"[^"]+"\|', l), l
    node_names = {re.search(r'\["([^"]+)"\]', l).group(1) for l in nodes}
    assert node_names <= CANON_NAMES

    chains = plot["causal_chains"]
    assert chains
    for ch in chains:
        assert set(ch) == {"title", "steps", "involved"}
        assert isinstance(ch["title"], str) and ch["title"].strip()
        assert isinstance(ch["steps"], list) and ch["steps"]
        assert isinstance(ch["involved"], list)
        assert set(ch["involved"]) <= CANON_NAMES


def test_bilingual_speaker_canonicalized(v4_pipeline):
    rows = [
        json.loads(l) for l in (v4_pipeline / "output" / "script_bilingual.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert set(r["speaker"] for r in rows) == CANON_NAMES
    assert all(r.get("text_zh") for r in rows)


# --------------------------------------------------------------------------- #
# 3. LLM 路径：2 批紧凑 JSON 解析为规范 schema（monkeypatch，不发网络）
# --------------------------------------------------------------------------- #
def test_llm_profile_path_normalizes(monkeypatch):
    sm = summarize_mr
    lines = [
        {"line_id": i, "part": (i - 1) // 5 + 1, "ts_sec": i * 10.0,
         "speaker": "サトちゃん" if i % 2 else "华花", "text_ja": "台詞" + str(i)}
        for i in range(1, 21)
    ]
    scenes = [
        {"scene_id": f"S{i:02d}", "part": i, "characters": ["サトミ", "藤花"],
         "summary_zh": "サトミと藤花", "quotes": [], "source_lines": [i * 5 - 4, i * 5]}
        for i in range(1, 5)
    ]

    calls = {"n": 0}

    def fake_llm(messages, *, base_url, api_key, model):
        calls["n"] += 1
        # LLM 用误识名 "华花" 回填，规范化层必须把它改回 "藤花"
        return {"profiles": [
            {"name": n, "role_tag": "师父", "faction": "里见系",
             "summary_zh": "深度人物定位", "motivation": "寻找女儿",
             "key_interactions": ["与春希：师徒"], "arc_highlights": ["P9 决战"]}
            for n in ("サトミ", "华花")
        ]}

    monkeypatch.setattr(sm, "llm_json", fake_llm)
    profiles = sm.build_character_profiles(
        lines, scenes, sm.BUILTIN_GLOSSARY, mock=False, base_url="http://stub",
        api_key="k", model="stub",
    )
    assert calls["n"] == 2, "Top 角色应分 2 批调用 LLM"
    by_name = {p["name"]: p for p in profiles}
    assert "藤花" in by_name and "华花" not in by_name
    assert by_name["藤花"]["motivation"] == "寻找女儿"
    assert by_name["藤花"]["faction"] == "里见系"
    assert set(summarize_mr.PROFILE_FIELDS) <= set(by_name["藤花"])


# --------------------------------------------------------------------------- #
# 4. v4 HTML：node --check + 零阻塞外链 + DOM 执行
# --------------------------------------------------------------------------- #
NODE_HARNESS = r"""
const fs=require('fs'), vm=require('vm');
const expectedLabel=process.argv[process.argv.length-1]||'';
const files=process.argv.slice(2,-1);
function El(id){ return {id:id,innerHTML:'',textContent:'',style:{},value:'',checked:false,
  disabled:false,max:'',_attrs:{},addEventListener:function(){},
  getAttribute:function(k){return this._attrs[k]||null;},
  setAttribute:function(k,v){this._attrs[k]=v;},querySelectorAll:function(){return [];}}; }
var ids=['fatal-error','theme-btn','graph','mermaid-host','mermaid-btn','acts','chars',
  'causal','cross','scene-list','part-filter','trx-body','page-info','prev-page',
  'next-page','jump-page','go-page','search','speaker-filters'];
var els={}; ids.forEach(function(i){els[i]=El(i);});
var document={body:El('body'),head:El('head'),
  getElementById:function(id){return els[id]||null;},
  querySelectorAll:function(){return [];},createElement:function(){return El('new');}};
var win={localStorage:{getItem:function(){throw new Error('SecurityError');},
  setItem:function(){throw new Error('SecurityError');}},addEventListener:function(){}};
var ctx={document:document,window:win,setTimeout:function(){return 0;},clearTimeout:function(){}};
vm.createContext(ctx);
files.forEach(function(f){ vm.runInContext(fs.readFileSync(f,'utf8'),ctx,{filename:f}); });
ctx.window.__VW_TEST__.switchTab('trx');
ctx.window.__VW_TEST__.filterBySpeaker('サトミ');
function h(id){ return ctx.document.getElementById(id).innerHTML; }
console.log(JSON.stringify({
  fatal: ctx.document.getElementById('fatal-error').textContent,
  profile: h('chars').indexOf('filter-speaker')>=0 && h('chars').indexOf('核心动机')>=0 &&
           h('chars').indexOf('badge faction')>=0 && h('chars').indexOf('关键人际交互')>=0 &&
           h('chars').indexOf('分 P 高光')>=0,
  graphLabel: h('graph').indexOf('edge-label')>=0,
  semanticLabel: expectedLabel ? h('graph').indexOf(expectedLabel)>=0 : true,
  causal: h('causal').indexOf('chain-steps')>=0,
  trxRows: h('trx-body').split('<tr').length-1,
  state: ctx.window.__VW_TEST__.state()
}));
"""


def test_v4_html_node_check_and_dom(v4_pipeline, tmp_path):
    plot = v4_pipeline / "output" / "plot_structured.json"
    script = v4_pipeline / "output" / "script_bilingual.jsonl"
    html_path = tmp_path / "viewer_v4.html"
    r = run_module(
        "bili_story_pipe.build_viewer", "--plot", str(plot), "--script", str(script),
        "--output", str(html_path),
    )
    assert r.returncode == 0, f"build_viewer 失败:\n{r.stdout}\n{r.stderr}"
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "id=\"chars\"" in text and "id=\"causal\"" in text
    assert "filter-speaker" in text and "chain-steps" in text

    # 零同步外部 script / 字体 / CSS
    assert not re.search(r'<script[^>]+src\s*=\s*["\']https?://', text), "存在同步外部脚本"

    scripts = re.findall(r"<script[^>]*>(.*?)</script>", text, re.S)
    assert len(scripts) >= 2
    js_files = []
    for i, js in enumerate(scripts):
        p = tmp_path / f"chk_{i}.js"
        p.write_text(js, encoding="utf-8")
        js_files.append(str(p))
        chk = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
        assert chk.returncode == 0, f"node --check 失败({i}): {chk.stderr}"

    # 从内嵌数据取一个真实语义边标签，验证它被渲染进离线 SVG
    payload = text.split('<script id="viewer-data">', 1)[1].split("</script>", 1)[0].strip()
    data = json.loads(payload[len("window.__VWDATA__ = "):-1])
    assert data["v"] == 4 and "causal_chains" in data
    edge = re.search(r'---\|"([^"]+)"\|', data["mermaid"])
    label = edge.group(1) if edge else ""

    harness = tmp_path / "harness.js"
    harness.write_text(NODE_HARNESS, encoding="utf-8")
    out = subprocess.run(
        ["node", str(harness), *js_files, label], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, f"DOM 执行失败: {out.stderr}"
    result = json.loads(out.stdout.strip())
    assert result["fatal"] == "", f"页面初始化抛错: {result['fatal']}"
    assert result["profile"] is True, result
    assert result["graphLabel"] is True, result
    assert result["semanticLabel"] is True, f"语义边标签未渲染: {result}"
    assert result["causal"] is True, result
    assert result["trxRows"] > 0, result
