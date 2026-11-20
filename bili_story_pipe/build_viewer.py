#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_viewer.py v4 — compile bili-story-pipe outputs into ONE offline HTML viewer.

LAB-2135 upgrades (on top of all v3 fixes below):

4. Tab 1 deep data quality.
   - Profile cards render role_tag / faction badge / summary_zh / motivation /
     key_interactions / arc_highlights, plus a one-click "筛选该角色台词" that
     jumps to Tab 3 pre-filtered to that canonical speaker.
   - New "跨章因果与关键伏笔" section renders ``plot.causal_chains``.
   - The instant offline SVG now parses the semantic Mermaid graph (canonical
     Top-15 nodes + real relation labels) instead of drawing a star with no labels.

LAB-2132 fixes (all three fatal front-end bugs):

1. ZERO blocking external resources.
   - No synchronous ``<script src="http...">`` and no external font/CSS in the
     document. The page renders instantly (0 ms) with an inline, offline SVG
     character graph plus plain text.
   - Mermaid is OPTIONAL: injected via a dynamically created ``<script async>``
     either on user click ("尝试渲染 Mermaid 图") or after ``window.load``. A
     5-second silent timeout keeps the offline view if the CDN is unreachable.

2. localStorage safety + visible errors.
   - ``safeStorage = { get(k,d), set(k,v) }`` wraps every access in try/catch
     and falls back to an in-memory object (survives file://, Safari, iframes).
   - The whole main logic sits inside an outer try/catch; any exception paints a
     readable red banner at the top instead of a silent blank page.

3. Lazy + paginated 8,223-row transcript.
   - Tab 3 is rendered only on first activation / first search (no 8k DOM rows
     at page load). 100 rows per page with prev / next / jump / hit count.
   - Embedded JSON uses short keys and drops empty ``text_zh`` to shrink HTML.
   - ``<``, ``>``, ``&``, U+2028/2029 are escaped so no Japanese line can ever
     break out of the ``<script id="viewer-data">`` block.

CLI
---
    python3 build_viewer.py --plot plot_structured.json \
        --script script_bilingual.jsonl --cross cross_check.json \
        --output viewer.html [--bvid BV...] [--title "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Data preparation (pure, unit-testable)
# --------------------------------------------------------------------------- #

# short key -> meaning: i=line_id, p=part, t=ts_sec, s=speaker, j=text_ja, z=text_zh
TRANSCRIPT_KEYS = ("i", "p", "t", "s", "j", "z")


def _first(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key from ``d`` (tolerant of schema variants)."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def compress_transcript(lines: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shorten keys and drop empty translations to shrink embedded JSON."""
    out: List[Dict[str, Any]] = []
    for idx, ln in enumerate(lines or [], start=1):
        if not isinstance(ln, dict):
            continue
        row: Dict[str, Any] = {
            "i": _first(ln, "line_id", "i", "id", default=idx),
            "p": _first(ln, "part", "p", default=1),
            "t": round(float(_first(ln, "ts_sec", "t", "ts", default=0.0) or 0.0), 1),
            "s": str(_first(ln, "speaker", "s", default="") or ""),
            "j": str(_first(ln, "text_ja", "j", "text", default="") or ""),
        }
        zh = str(_first(ln, "text_zh", "z", default="") or "")
        if zh.strip():
            row["z"] = zh  # omit empty translations entirely
        out.append(row)
    return out


def embed_json(obj: Any) -> str:
    """JSON text safe to embed inside a ``<script>`` block.

    Escaping ``<``/``>``/``&``/line-separators means the literal sequence
    ``</script>`` can never appear, while ``JSON.parse``/the JS parser still
    decode the original characters correctly.
    """
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def load_inputs(plot_path: str, script_path: str, cross_path: str) -> Dict[str, Any]:
    """Load the three pipeline artifacts into one normalized data dict."""
    with open(plot_path, "r", encoding="utf-8") as f:
        plot = json.load(f)
    cross: Dict[str, Any] = {}
    if cross_path and os.path.exists(cross_path):
        with open(cross_path, "r", encoding="utf-8") as f:
            cross = json.load(f)

    lines: List[Dict[str, Any]] = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    bvid = _first(plot, "bvid", default="") or _first(cross, "bvid", default="")
    return assemble_data(plot, lines, cross, bvid=bvid)


def assemble_data(
    plot: Dict[str, Any],
    lines: List[Dict[str, Any]],
    cross: Optional[Dict[str, Any]] = None,
    bvid: str = "",
    title: str = "",
) -> Dict[str, Any]:
    """Normalize heterogeneous plot/cross schemas into the viewer data dict."""
    cross = cross or {}
    acts = _first(plot, "acts", "three_acts", "structure", default=[]) or []
    characters = _first(plot, "characters", "character_profiles", "chars", default=[]) or []
    scenes = _first(plot, "scenes", "scene_cards", default=[]) or []
    mermaid = _first(plot, "mermaid", "mermaid_graph", "relation_graph", default="") or ""
    causal = _first(plot, "causal_chains", "causal", "causal_chain", default=[]) or []
    coverage = _first(plot, "coverage_audit", "coverage", default={}) or {}
    cross_items = _first(cross, "items", "checks", "results", default=[]) or []

    return {
        "v": 4,
        "title": title or _first(plot, "work_title", "title", default="交互式剧情浏览器") or "交互式剧情浏览器",
        "bvid": bvid,
        "acts": acts,
        "characters": characters,
        "mermaid": mermaid if isinstance(mermaid, str) else "\n".join(map(str, mermaid)),
        "causal_chains": causal,
        "scenes": scenes,
        "coverage": coverage,
        "cross": {"summary": _first(cross, "summary", default=""), "items": cross_items},
        "trx": compress_transcript(lines),
    }


# --------------------------------------------------------------------------- #
# HTML template
# --------------------------------------------------------------------------- #

CSS = r"""
:root{--bg:#0f1116;--fg:#e8eaf0;--card:#1a1d26;--line:#2b2f3d;--accent:#7aa2f7;
--ok:#3fb950;--warn:#d29922;--bad:#f85149;--muted:#8b93a7}
body[data-theme=light]{--bg:#f6f7fb;--fg:#1c2030;--card:#fff;--line:#dfe3ee;
--accent:#2f6feb;--muted:#5c6478}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line);
position:sticky;top:0;background:var(--bg);z-index:5}
header h1{font-size:17px;margin:0;flex:1}
.tabs{display:flex;gap:6px}
.tab{padding:7px 14px;border:1px solid var(--line);border-radius:8px;cursor:pointer;background:var(--card);color:var(--fg)}
.tab.active{border-color:var(--accent);color:var(--accent)}
.panel{display:none;padding:20px;max-width:1200px;margin:0 auto}
.panel.active{display:block}
.btn{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.btn:disabled{opacity:.4;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h3{margin:0 0 6px;font-size:15px}
.muted{color:var(--muted)}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.badge.ok{background:rgba(63,185,80,.18);color:var(--ok)}
.badge.warn{background:rgba(210,153,34,.18);color:var(--warn)}
.badge.bad{background:rgba(248,81,73,.18);color:var(--bad)}
.act{border-left:3px solid var(--accent);padding-left:12px;margin:10px 0}
.offline-graph{overflow:auto}
.offline-graph svg{display:block;margin:0 auto}
.edge-label{font-size:10px;fill:var(--warn);paint-order:stroke;stroke:var(--bg);stroke-width:3px}
.chead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge.faction{background:rgba(122,162,247,.16);color:var(--accent);margin-left:auto}
.role-tag{color:var(--muted);font-size:12px;margin-bottom:6px}
.profile-summary{margin:6px 0;font-size:13px}
.profile-mot{margin:6px 0;font-size:12px}
.profile-sub{margin:8px 0;font-size:12px;color:var(--muted)}
.profile-sub ul{margin:4px 0 0;padding-left:18px}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0}
.chip{display:inline-block;padding:1px 7px;border:1px solid var(--line);border-radius:10px;font-size:11px;color:var(--muted)}
.filter-speaker{margin-top:8px}
.chain-steps{margin:6px 0;padding-left:20px}
.step-no{display:inline-block;min-width:18px;color:var(--accent);font-weight:600}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
input[type=text]{padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);min-width:240px}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.filters label{padding:3px 8px;border:1px solid var(--line);border-radius:12px;cursor:pointer}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{position:sticky;top:54px;background:var(--bg);color:var(--muted);font-size:12px}
td a{color:var(--accent);text-decoration:none;white-space:nowrap}
td .ja{font-size:13px}
td .zh{color:var(--muted);font-size:12px}
.pager{display:flex;gap:8px;align-items:center;justify-content:center;margin:16px 0}
.scene-quote{color:var(--muted);font-style:italic;margin:4px 0}
.dialog{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px;max-height:320px;overflow:auto}
#fatal-error{display:none;background:rgba(248,81,73,.15);border:1px solid var(--bad);color:var(--fg);
padding:12px 20px;margin:0;font-family:ui-monospace,monospace}
"""

# Main logic. Kept in one string so tests can extract & execute it verbatim.
JS_MAIN = r"""
(function(){
  "use strict";
  var errBox = document.getElementById("fatal-error");
  function fatal(e){
    var msg = (e && e.message) ? e.message : String(e);
    try{
      if(errBox){ errBox.style.display="block"; errBox.textContent="页面初始化失败（已降级为可见报错，非白屏）: "+msg; }
    }catch(_){}
  }
  try{
    /* ---- 2. safeStorage: never let localStorage throw ---- */
    var mem = {};
    var safeStorage = {
      get: function(k, d){
        try{
          var v = window.localStorage.getItem(k);   // getter itself may throw SecurityError
          return v === null ? d : v;
        }catch(e){
          return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : d;
        }
      },
      set: function(k, v){
        mem[k] = v;                                  // always remember in memory
        try{ window.localStorage.setItem(k, v); }catch(e){}
      }
    };

    var data = window.__VWDATA__ || {};
    var trx = data.trx || [];
    var bvid = data.bvid || "";
    function esc(s){ return String(s==null?"":s)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/"/g,"&quot;").replace(/'/g,"&#39;"); }
    function $(id){ return document.getElementById(id); }

    /* ---- theme + tabs (progress via safeStorage) ---- */
    var theme = safeStorage.get("vw_theme", "dark");
    document.body.setAttribute("data-theme", theme === "light" ? "light" : "dark");
    $("theme-btn") && $("theme-btn").addEventListener("click", function(){
      var cur = document.body.getAttribute("data-theme");
      var next = cur === "dark" ? "light" : "dark";
      document.body.setAttribute("data-theme", next);
      safeStorage.set("vw_theme", next);
    });
    function switchTab(name){
      var tabs = document.querySelectorAll ? document.querySelectorAll(".tab") : [];
      for(var i=0;i<tabs.length;i++){ tabs[i].className = "tab" + (tabs[i].getAttribute("data-tab")===name ? " active" : ""); }
      var panels = document.querySelectorAll ? document.querySelectorAll(".panel") : [];
      for(var j=0;j<panels.length;j++){ panels[j].className = "panel" + (panels[j].id==="panel-"+name ? " active" : ""); }
      if(name === "trx"){ ensureTrx(); }
      safeStorage.set("vw_tab", name);
    }
    var tabEls = document.querySelectorAll ? document.querySelectorAll(".tab") : [];
    for(var t=0;t<tabEls.length;t++){ (function(el){ el.addEventListener("click", function(){ switchTab(el.getAttribute("data-tab")); }); })(tabEls[t]); }

    /* ---- 1. offline SEMANTIC character graph (instant, no network) ---- */
    /* Parse Mermaid text emitted by summarize_mr v3 (semantic edge labels). */
    function parseGraph(){
      var nodes=[], edges=[], ids={};
      String(data.mermaid||"").split(/\r?\n/).forEach(function(line){
        var n=line.match(/^\s*(C\d+)\["([^"]+)"\]\s*$/);
        if(n){ ids[n[1]]=n[2]; nodes.push({id:n[1],name:n[2]}); return; }
        var e=line.match(/^\s*(C\d+)\s*---\|"([^"]*)"\|\s*(C\d+)\s*$/);
        if(e && ids[e[1]] && ids[e[3]]){
          edges.push({a:ids[e[1]],b:ids[e[3]],label:e[2]});
        }
      });
      if(!nodes.length){
        nodes=(data.characters||[]).slice(0,15).map(function(c,i){ return {id:"C"+i,name:_name(c)}; });
      }
      return {nodes:nodes.slice(0,15), edges:edges};
    }
    function renderOfflineGraph(){
      var host = $("graph");
      if(!host) return;
      var g = parseGraph(), nodes = g.nodes, edges = g.edges;
      if(!nodes.length){ host.innerHTML='<div class="muted">（暂无角色档案数据）</div>'; return; }
      var W=760, H=Math.max(320, nodes.length*44), cx=W/2, cy=H/2, R=Math.min(W,H)/2-110;
      var parts=['<svg width="100%" height="'+H+'" viewBox="0 0 '+W+' '+H+'" xmlns="http://www.w3.org/2000/svg">'];
      var pts=[], idx={};
      for(var i=0;i<nodes.length;i++){
        var a = (2*Math.PI*i)/nodes.length - Math.PI/2;
        pts.push([cx+R*Math.cos(a), cy+R*Math.sin(a)]);
        idx[nodes[i].name]=i;
      }
      edges.forEach(function(ed){
        var ia=idx[ed.a], ib=idx[ed.b];
        if(ia==null || ib==null || ia===ib) return;
        var x1=pts[ia][0], y1=pts[ia][1], x2=pts[ib][0], y2=pts[ib][1];
        parts.push('<line x1="'+x1.toFixed(1)+'" y1="'+y1.toFixed(1)+'" x2="'+x2.toFixed(1)+
                   '" y2="'+y2.toFixed(1)+'" stroke="var(--line)" stroke-width="1.5"/>');
        if(ed.label){
          var mx=(x1+x2)/2, my=(y1+y2)/2;
          parts.push('<text class="edge-label" x="'+mx.toFixed(1)+'" y="'+(my-3).toFixed(1)+
                     '" text-anchor="middle">'+esc(ed.label)+'</text>');
        }
      });
      for(var m=0;m<nodes.length;m++){
        var nm = nodes[m].name;
        var yy = pts[m][1]+4;
        parts.push('<circle cx="'+pts[m][0].toFixed(1)+'" cy="'+pts[m][1].toFixed(1)+'" r="5" fill="var(--accent)"/>'+
                   '<text x="'+(pts[m][0]>cx?pts[m][0]+10:pts[m][0]-10).toFixed(1)+'" y="'+yy.toFixed(1)+
                   '" text-anchor="'+(pts[m][0]>cx?'start':'end')+'" fill="var(--fg)" font-size="12">'+esc(nm)+'</text>');
      }
      parts.push('</svg>');
      host.innerHTML='<div class="offline-graph">'+parts.join("")+'</div>';
    }
    function _name(c){ if(typeof c==="string") return c; return c.name||c.original||c.character||c.title||"（未命名）"; }

    /* ---- 1b. OPTIONAL Mermaid: async injection only, never blocks first paint ---- */
    var mermaidTried=false;
    function loadMermaid(){
      if(mermaidTried) return; mermaidTried=true;
      var src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
      var s=document.createElement("script");
      s.src=src; s.async=true;
      var timer=setTimeout(function(){ /* 5s timeout: silently keep offline view */ }, 5000);
      s.onload=function(){ clearTimeout(timer); try{
        if(window.mermaid && data.mermaid){
          window.mermaid.initialize({startOnLoad:false, theme:(document.body.getAttribute("data-theme")==="dark"?"dark":"default")});
          window.mermaid.render("mermaid-svg", data.mermaid).then(function(res){
            var host=$("mermaid-host"); if(host){ host.innerHTML='<div class="card">'+res.svg+'</div>'; }
          }).catch(function(){});
        }
      }catch(e){} };
      s.onerror=function(){ clearTimeout(timer); };
      (document.head||document.body||document.documentElement).appendChild(s);
    }
    var mbtn=$("mermaid-btn"); if(mbtn) mbtn.addEventListener("click", loadMermaid);
    window.addEventListener("load", function(){ setTimeout(loadMermaid, 0); });

    /* ---- overview cards (Tab 1, v4) ---- */
    function renderOverview(){
      var actsHost=$("acts"), crHost=$("cross");
      if(actsHost){
        var acts=data.acts||[];
        actsHost.innerHTML = acts.length ? acts.map(function(a){
          var o=(typeof a==="string")?{text:a}:a;
          return '<div class="act"><h3>'+esc(o.title||o.name||"")+'</h3><div>'+
                 esc(o.summary_zh||o.text||o.summary||o.body||"")+'</div></div>';
        }).join("") : '<div class="muted">（暂无三幕总览数据）</div>';
      }
      renderCharacters();
      renderCausal();
      if(crHost){
        var cr=data.cross||{}, items=cr.items||[];
        var rows=items.map(function(it){
          var st=String(it.status||it.badge||"").toLowerCase();
          var cls = st.indexOf("conflict")>=0 ? "bad" : (st.indexOf("uncovered")>=0||st.indexOf("warn")>=0 ? "warn" : "ok");
          var lab = it.badge || it.label || it.status || "";
          return '<tr><td><span class="badge '+cls+'">'+esc(lab)+'</span></td><td>'+esc(it.title||it.item||it.name||"")+
                 '</td><td>'+esc(it.detail||it.text||it.note||"")+'</td><td class="muted">'+esc(it.source||it.source_line||"")+'</td></tr>';
        }).join("");
        crHost.innerHTML='<div class="card"><p class="muted">'+esc(cr.summary||"")+'</p>'+
          (rows?'<table><thead><tr><th>状态</th><th>核验项</th><th>说明</th><th>来源</th></tr></thead><tbody>'+rows+'</tbody></table>':
           '<div class="muted">（暂无交叉校验数据）</div>')+'</div>';
      }
    }

    /* ---- v4: deep character profile cards ---- */
    function renderCharacters(){
      var chHost=$("chars"); if(!chHost) return;
      var chars=data.characters||[];
      if(!chars.length){ chHost.innerHTML='<div class="muted">（暂无角色档案数据）</div>'; return; }
      chHost.innerHTML = chars.map(function(c){
        var nm=_name(c);
        var tag=c.role_tag||c.role_name||c.role||"";
        var fac=c.faction||"";
        var interactions=(c.key_interactions||[]).map(function(x){ return '<li>'+esc(x)+'</li>'; }).join("");
        var arcs=(c.arc_highlights||[]).map(function(x){ return '<span class="chip">'+esc(x)+'</span>'; }).join("");
        return '<div class="card profile">'+
          '<div class="chead"><h3>'+esc(nm)+'</h3>'+(fac?'<span class="badge faction">'+esc(fac)+'</span>':'')+'</div>'+
          (tag?'<div class="role-tag">'+esc(tag)+'</div>':'')+
          '<div class="profile-summary">'+esc(c.summary_zh||c.desc||c.summary||c.profile||"")+'</div>'+
          (c.motivation?'<div class="profile-mot"><b>核心动机</b> '+esc(c.motivation)+'</div>':'')+
          (interactions?'<div class="profile-sub"><b>关键人际交互</b><ul>'+interactions+'</ul></div>':'')+
          (arcs?'<div class="profile-sub"><b>分 P 高光</b><div class="chips">'+arcs+'</div></div>':'')+
          '<button class="btn filter-speaker" data-speaker="'+esc(nm)+'">筛选该角色台词 →</button>'+
        '</div>';
      }).join("");
      var btns=chHost.querySelectorAll?chHost.querySelectorAll("button.filter-speaker"):[];
      for(var i=0;i<btns.length;i++){
        (function(b){ b.addEventListener("click", function(){ filterBySpeaker(b.getAttribute("data-speaker")); }); })(btns[i]);
      }
    }

    /* ---- v4: cross-chapter causal chains / foreshadowing ---- */
    function renderCausal(){
      var host=$("causal"); if(!host) return;
      var chains=data.causal_chains||[];
      if(!chains.length){ host.innerHTML='<div class="muted">（暂无跨章因果链数据）</div>'; return; }
      host.innerHTML='<div class="grid">'+chains.map(function(ch){
        var steps=(ch.steps||[]).map(function(s,i){
          return '<li><span class="step-no">'+(i+1)+'.</span>'+esc(s)+'</li>';
        }).join("");
        var inv=(ch.involved||[]).map(function(n){ return '<span class="chip">'+esc(n)+'</span>'; }).join("");
        return '<div class="card chain"><h3>'+esc(ch.title||"")+'</h3>'+
          (steps?'<ol class="chain-steps">'+steps+'</ol>':'')+
          (inv?'<div class="chips">'+inv+'</div>':'')+'</div>';
      }).join("")+'</div>';
    }

    /* ---- v4: jump from a profile card to that character's lines in Tab 3 ---- */
    function filterBySpeaker(name){
      switchTab("trx");
      ensureTrx();
      var box=$("speaker-filters");
      if(box && box.querySelectorAll){
        var cs=box.querySelectorAll("input[type=checkbox]");
        for(var i=0;i<cs.length;i++){ cs[i].checked = (cs[i].value===name); }
      }
      var s=$("search"); if(s) s.value="";
      buildFiltered(); drawPage();
    }

    /* ---- scenes ---- */
    function renderScenes(){
      var host=$("scene-list"), sel=$("part-filter"); if(!host) return;
      var scenes=data.scenes||[];
      if(sel){
        var parts={}; scenes.forEach(function(s){ var p=_first3(s,"part","p",1); parts[p]=1; });
        sel.innerHTML='<option value="all">全部分 P</option>'+Object.keys(parts).sort(function(a,b){return a-b;})
          .map(function(p){return '<option value="'+esc(p)+'">P'+esc(p)+'</option>';}).join("");
        sel.addEventListener("change", function(){ draw(sel.value); });
      }
      function _first3(o,a,b,d){ return (o&&o[a]!=null)?o[a]:((o&&o[b]!=null)?o[b]:d); }
      function draw(filter){
        var list=scenes.filter(function(s){ return filter==="all" || String(_first3(s,"part","p",1))===String(filter); });
        host.innerHTML = list.length ? list.map(function(s,idx){
          var quotes=(s.quotes||[]).map(function(q){
            return '<div class="scene-quote">「'+esc(q.speaker||q.s||"")+'」'+esc(q.text_ja||q.ja||q.text||"")+'</div>';
          }).join("");
          return '<div class="card"><h3>'+esc(s.title||s.scene_id||s.id||("场景 "+(idx+1)))+'</h3>'+
            '<div class="muted">P'+esc(_first3(s,"part","p","?"))+' · '+esc(s.ts_start||s.ts||"")+'</div>'+
            '<div>'+esc(s.summary_zh||s.summary||s.text||"")+'</div>'+quotes+
            '<button class="btn" data-scene="'+idx+'" style="margin-top:8px">展开完整对话</button>'+
            '<div id="dialog-'+idx+'" class="dialog"></div></div>';
        }).join("") : '<div class="muted">（暂无场景卡数据）</div>';
        var btns=host.querySelectorAll?host.querySelectorAll("button[data-scene]"):[];
        for(var i=0;i<btns.length;i++){ (function(b){
          b.addEventListener("click", function(){
            var s=list[parseInt(b.getAttribute("data-scene"),10)];
            var box=$("dialog-"+b.getAttribute("data-scene"));
            if(box) box.innerHTML=sceneDialog(s);
          });
        })(btns[i]); }
      }
      function sceneDialog(s){
        var src=(s.source_lines||[]); var a=src[0]!=null?src[0]:-1, b=src[1]!=null?src[1]:-1;
        var picked = a<0 ? [] : trx.filter(function(r){ return r.i>=a && r.i<=b; });
        if(!picked.length) picked=trx.slice(0,80);
        return picked.map(function(r){ return '<div>'+jump(r)+' <b>'+esc(r.s)+'</b> '+esc(r.j)+
          (r.z?' <span class="zh">/ '+esc(r.z)+'</span>':'')+'</div>'; }).join("") || '<div class="muted">（无对话）</div>';
      }
      draw("all");
    }

    /* ---- 3. LAZY + PAGINATED transcript ---- */
    var PAGE_SIZE=100, curPage=1, filtered=[], trxReady=false;
    function jump(r){
      var t=Math.floor(Number(r.t)||0), mm=("0"+Math.floor(t/60)).slice(-2), ss=("0"+(t%60)).slice(-2);
      var href = bvid ? ("https://www.bilibili.com/video/"+encodeURIComponent(bvid)+"/?p="+encodeURIComponent(r.p)+"&amp;t="+t) : "#";
      return '<a href="'+href+'" target="_blank" rel="noopener">[P'+esc(r.p)+' '+mm+':'+ss+']</a>';
    }
    function selectedSpeakers(){
      var out=[]; var box=$("speaker-filters");
      if(box && box.querySelectorAll){ var cs=box.querySelectorAll("input[type=checkbox]:checked");
        for(var i=0;i<cs.length;i++) out.push(cs[i].value); }
      return out;
    }
    function buildFiltered(){
      var kw=(($("search")&&$("search").value)||"").trim().toLowerCase();
      var sel=selectedSpeakers();
      filtered=trx.filter(function(r){
        if(sel.length && sel.indexOf(r.s)<0) return false;
        if(!kw) return true;
        return (r.j||"").toLowerCase().indexOf(kw)>=0 || (r.z||"").toLowerCase().indexOf(kw)>=0 || (r.s||"").toLowerCase().indexOf(kw)>=0;
      });
      curPage=1;
    }
    function drawPage(){
      var body=$("trx-body"); if(!body) return;
      var total=filtered.length, pages=Math.max(1, Math.ceil(total/PAGE_SIZE));
      if(curPage>pages) curPage=pages; if(curPage<1) curPage=1;
      var slice=filtered.slice((curPage-1)*PAGE_SIZE, curPage*PAGE_SIZE);
      body.innerHTML = slice.map(function(r){
        return '<tr><td>'+esc(r.i)+'</td><td>'+esc(r.s)+'</td><td class="ja">'+esc(r.j)+
               (r.z?'<div class="zh">'+esc(r.z)+'</div>':'')+'</td><td>'+jump(r)+'</td></tr>';
      }).join("") || '<tr><td colspan="4" class="muted">（无匹配台词）</td></tr>';
      var info=$("page-info"); if(info) info.textContent="命中 "+total+" 条 · 第 "+curPage+" / "+pages+" 页";
      var prev=$("prev-page"), next=$("next-page"), jumpIn=$("jump-page");
      if(prev) prev.disabled = curPage<=1;
      if(next) next.disabled = curPage>=pages;
      if(jumpIn) jumpIn.max = pages;
    }
    function ensureTrx(){
      if(trxReady) return; trxReady=true;
      // speaker checkboxes built once, lazily
      var box=$("speaker-filters");
      if(box){
        var counts={}; trx.forEach(function(r){ if(r.s) counts[r.s]=(counts[r.s]||0)+1; });
        var names=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a];}).slice(0,40);
        box.innerHTML=names.map(function(n){ return '<label><input type="checkbox" value="'+esc(n)+'">'+esc(n)+' ('+counts[n]+')</label>'; }).join("");
        var cbs=box.querySelectorAll?box.querySelectorAll("input[type=checkbox]"):[];
        for(var i=0;i<cbs.length;i++) cbs[i].addEventListener("change", function(){ buildFiltered(); drawPage(); });
      }
      var s=$("search"); if(s) s.addEventListener("input", function(){ buildFiltered(); drawPage(); });
      var prev=$("prev-page"); if(prev) prev.addEventListener("click", function(){ curPage--; drawPage(); });
      var next=$("next-page"); if(next) next.addEventListener("click", function(){ curPage++; drawPage(); });
      var go=$("go-page"); if(go) go.addEventListener("click", function(){
        var v=parseInt(($("jump-page")&&$("jump-page").value)||"1",10); if(!isNaN(v)){ curPage=v; drawPage(); }
      });
      buildFiltered(); drawPage();
    }

    renderOfflineGraph(); renderOverview(); renderScenes();
    /* Tab 3 intentionally NOT rendered here — lazily on activation */
    var initial=safeStorage.get("vw_tab","overview");
    if(initial==="trx"){ switchTab("trx"); }

    /* test/debug hook (harmless in production) */
    window.__VW_TEST__={
      safeStorage:safeStorage, fatal:fatal, ensureTrx:ensureTrx, switchTab:switchTab,
      loadMermaid:loadMermaid, filterBySpeaker:filterBySpeaker, renderOfflineGraph:renderOfflineGraph,
      state:function(){ return {ready:trxReady, page:curPage, rows:filtered.length,
        pageRows:(($("trx-body")||{}).innerHTML||"").split("<tr").length-1,
        theme:document.body.getAttribute("data-theme")}; }
    };
  }catch(err){ fatal(err); }
})();
"""

HTML_TMPL = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body data-theme="dark">
<div id="fatal-error"></div>
<header>
  <h1>__TITLE__</h1>
  <div class="tabs">
    <button class="tab active" data-tab="overview">总览与校验</button>
    <button class="tab" data-tab="scenes">分场景精读</button>
    <button class="tab" data-tab="trx">完整台词</button>
  </div>
  <button class="btn" id="theme-btn" title="切换暗色/亮色">主题</button>
</header>
<div class="panel active" id="panel-overview">
  <h2>三幕总览</h2><div id="acts"></div>
  <h2>角色关系（离线内置 SVG，秒开）</h2><div id="graph"></div>
  <p><button class="btn" id="mermaid-btn">尝试渲染 Mermaid 图（需联网，失败自动保留离线视图）</button></p>
  <div id="mermaid-host"></div>
  <h2>核心角色档案</h2><div class="grid" id="chars"></div>
  <h2>跨章因果与关键伏笔</h2><div id="causal"></div>
  <h2>联网交叉校验</h2><div id="cross"></div>
</div>
<div class="panel" id="panel-scenes">
  <div class="toolbar"><label>分 P 过滤：<select id="part-filter" class="btn"></select></label></div>
  <div class="grid" id="scene-list"></div>
</div>
<div class="panel" id="panel-trx">
  <div class="toolbar">
    <input type="text" id="search" placeholder="搜索日文/中文/角色名…">
    <span class="muted" id="page-info"></span>
  </div>
  <div class="filters" id="speaker-filters"></div>
  <table><thead><tr><th>#</th><th>角色</th><th>台词（原文 / 译文）</th><th>跳转</th></tr></thead>
  <tbody id="trx-body"></tbody></table>
  <div class="pager">
    <button class="btn" id="prev-page">上一页</button>
    <button class="btn" id="next-page">下一页</button>
    <span>跳至 <input type="number" id="jump-page" min="1" value="1" style="width:70px"> 页</span>
    <button class="btn" id="go-page">Go</button>
  </div>
</div>
<script id="viewer-data">window.__VWDATA__ = __DATA__;</script>
<script id="viewer-main">__JS__</script>
</body>
</html>
"""


def build_html(data: Dict[str, Any], title: str = "", bvid: str = "") -> str:
    """Return the full single-file HTML string for ``data``."""
    if bvid:
        data = dict(data)
        data["bvid"] = bvid
    page_title = title or data.get("title") or "剧情浏览器"
    return (
        HTML_TMPL.replace("__TITLE__", esc_html(page_title))
        .replace("__CSS__", CSS)
        .replace("__JS__", JS_MAIN)
        .replace("__DATA__", embed_json(data))
    )


def esc_html(s: Any) -> str:
    return (
        str(s if s is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_viewer(
    plot_path: str,
    script_path: str,
    cross_path: str,
    output_path: str,
    bvid: str = "",
    title: str = "",
) -> str:
    data = load_inputs(plot_path, script_path, cross_path)
    html = build_html(data, title=title, bvid=bvid or data.get("bvid", ""))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build offline single-file HTML story viewer (v4)")
    ap.add_argument("--plot", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--cross", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bvid", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    out = write_viewer(args.plot, args.script, args.cross, args.output, args.bvid, args.title)
    print(f"[build_viewer v4] wrote {out} ({os.path.getsize(out)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
