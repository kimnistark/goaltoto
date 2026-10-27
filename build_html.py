"""
data/params.json → simulator.html (파라미터를 박아 넣은 단일 HTML, 시뮬 v3)
입력: data/params.json, players_{시즌}.json, shapes.json, chain.json
파라미터를 바꾸면 이 파일만 다시 실행하면 됨
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
params = json.loads((HERE / "data" / "params.json").read_text(encoding="utf-8"))
_pf = HERE / "data" / f"players_{params['meta']['season']}.json"
_sf = HERE / "data" / "shapes.json"
players = json.loads(_pf.read_text(encoding="utf-8"))["teams"] if _pf.exists() else {}
shapes = json.loads(_sf.read_text(encoding="utf-8"))["teams"] if _sf.exists() else {}
players = {k: [{kk: p[kk] for kk in ("pid", "name", "group", "start_prob", "finish", "gk_save")} for p in v]
           for k, v in players.items() if k in params["teams"]}
_cf = HERE / "data" / "chain.json"
if not _cf.exists():
    raise SystemExit("data/chain.json 이 없어. 먼저 py prepare_chain.py 를 돌려줘")
chain = json.loads(_cf.read_text(encoding="utf-8"))
_ids = {str(p["pid"]) for v in players.values() for p in v}
chain["players"] = {k: v for k, v in chain["players"].items() if k in _ids}
_kf = HERE / "data" / "corr.json"          # 전력 차 보정값 (없으면 보정 안 함)
corr_c1 = float(json.loads(_kf.read_text(encoding="utf-8")).get("c1", 0.0)) if _kf.exists() else 0.0
shapes = {k: v for k, v in shapes.items() if k in params["teams"]}
html = TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>골탐정 시뮬레이터</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --surface-2: #f3f2ef;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --home: #2a78d6; --away: #eb6834; --draw: #c3c2b7;
  --goal-bg: #e7f0fc;
  --h0: #f3f2ef; --h1: #cde2fb; --h2: #9ec5f4; --h3: #6da7ec; --h4: #3987e5; --h5: #256abf; --h6: #184f95;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --surface-2: #232321;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --line: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --home: #3987e5; --away: #d95926; --draw: #555550;
    --goal-bg: #1c2a3d;
    --h0: #232321; --h1: #17263a; --h2: #0d366b; --h3: #104281; --h4: #184f95; --h5: #1c5cab; --h6: #2a78d6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --surface-2: #232321;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --line: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --home: #3987e5; --away: #d95926; --draw: #555550;
  --goal-bg: #1c2a3d;
  --h0: #232321; --h1: #17263a; --h2: #0d366b; --h3: #104281; --h4: #184f95; --h5: #1c5cab; --h6: #2a78d6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
h1 { font-size: 22px; margin: 0; letter-spacing: -0.02em; }
.sub { color: var(--muted); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 14px; padding: 18px; }
.controls { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: end; }
.controls label { display: block; font-size: 12px; color: var(--ink-2); margin-bottom: 6px; }
select, button { font: inherit; color: var(--ink); }
select { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--surface-2); }
.vs { color: var(--muted); font-weight: 600; padding-bottom: 10px; }
.run-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
.run-row select { width: auto; }
button.primary { background: var(--ink); color: var(--page); border: 0; border-radius: 10px; padding: 10px 22px; font-weight: 600; cursor: pointer; }
button.ghost { background: transparent; border: 1px solid var(--line); border-radius: 10px; padding: 8px 14px; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.status { color: var(--muted); font-size: 13px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
.full { grid-column: 1 / -1; }
.grid > .card { min-width: 0; }
h2 { font-size: 14px; margin: 0 0 12px; color: var(--ink-2); font-weight: 600; }
.wdl { display: flex; height: 38px; border-radius: 8px; overflow: hidden; gap: 2px; background: var(--surface); }
.wdl div { display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 13px; min-width: 0; white-space: nowrap; }
.wdl .h { background: var(--home); color: #fff; } .wdl .d { background: var(--draw); color: var(--ink); } .wdl .a { background: var(--away); color: #fff; }
.wdl-legend { display: flex; flex-wrap: wrap; gap: 4px 12px; justify-content: space-between; margin-top: 8px; font-size: 13px; color: var(--ink-2); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }
.stat { background: var(--surface-2); border-radius: 10px; padding: 10px 12px; }
.stat .k { font-size: 12px; color: var(--muted); } .stat .v { font-size: 18px; font-weight: 650; }
.heat { display: grid; gap: 2px; font-variant-numeric: tabular-nums; }
.heat .cell { aspect-ratio: 1.35; display: flex; align-items: center; justify-content: center; font-size: 12px; border-radius: 4px; position: relative; }
.heat .axis { font-size: 12px; color: var(--muted); display: flex; align-items: center; justify-content: center; }
.heat .corner { font-size: 11px; color: var(--muted); line-height: 1.2; text-align: center; }
.cell:hover { outline: 2px solid var(--ink); outline-offset: -1px; }
.tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--page); font-size: 12px; padding: 6px 9px; border-radius: 6px; display: none; z-index: 5; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
td, th { padding: 7px 4px; border-bottom: 1px solid var(--line); text-align: right; font-size: 14px; }
td:first-child, th:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
.bar { height: 6px; border-radius: 3px; background: var(--home); display: inline-block; vertical-align: middle; margin-right: 8px; }
.log { max-height: 420px; overflow-y: auto; font-variant-numeric: tabular-nums; }
.ev { display: grid; grid-template-columns: 50px 1fr auto; gap: 8px; padding: 5px 8px; border-radius: 6px; font-size: 13px; color: var(--ink-2); }
.ev.goal { background: var(--goal-bg); color: var(--ink); font-weight: 600; }
.ev.sep { color: var(--muted); font-size: 12px; justify-content: center; grid-template-columns: 1fr; text-align: center; }
.ev .t { color: var(--muted); }
.head-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 8px; flex-wrap: wrap; }
button.ghost { white-space: nowrap; }
.ev > span:last-child { white-space: nowrap; }
.head-row h2 { margin: 0; }
.note { color: var(--muted); font-size: 12px; margin-top: 10px; }
.gd { display: flex; align-items: flex-end; gap: 6px; height: 190px; padding-top: 18px; }
.gd .col { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; min-width: 0; }
.gd .b { width: 100%; max-width: 56px; border-radius: 4px 4px 0 0; }
.gd .pv { font-size: 12px; color: var(--ink-2); margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.gd .x { font-size: 12px; color: var(--muted); margin-top: 6px; border-top: 1px solid var(--line); width: 100%; text-align: center; padding-top: 4px; }
.gd-legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-2); margin-top: 10px; }
.hc-inputs { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 14px; }
.hc-inputs label { font-size: 12px; color: var(--ink-2); display: block; }
.hc-inputs input { display: block; width: 100%; margin-top: 4px; font: inherit; color: var(--ink); padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface-2); }
.hc-wrap { overflow-x: auto; }
#hcTbl td, #hcTbl th { white-space: nowrap; padding: 8px 6px; }
.verdict { font-weight: 600; }
.v-good { color: #006300; } .v-mid { color: var(--ink-2); } .v-bad { color: var(--muted); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .v-good { color: #0ca30c; } }
:root[data-theme="dark"] .v-good { color: #0ca30c; }
.excl { margin-top: 14px; }
.excl summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
.excl-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }
.excl-col h3 { font-size: 13px; margin: 0 0 6px; }
.excl-col label { display: flex; gap: 8px; align-items: center; font-size: 13px; padding: 3px 0; color: var(--ink); }
.excl-col label span.m { color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
.excl-col .grp { font-size: 11px; color: var(--muted); margin: 8px 0 2px; }
.lineups { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px; font-size: 13px; }
.lineups .shape { font-weight: 600; margin-bottom: 4px; }
.lineups .ppl { color: var(--ink-2); line-height: 1.7; }
.hidden { display: none; }
@media (max-width: 720px) {
  .grid { grid-template-columns: 1fr; }
  .controls { grid-template-columns: 1fr; }
  .vs { display: none; }
  .stats { grid-template-columns: repeat(2, 1fr); }
  .hc-inputs { grid-template-columns: repeat(2, 1fr); }
  .excl-cols, .lineups { grid-template-columns: 1fr; }
  .gd { gap: 3px; } .gd .pv { font-size: 10px; }
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>골탐정 시뮬레이터</h1>
    <div class="sub" id="meta"></div>
  </header>

  <div class="card">
    <div class="controls">
      <div><label for="home">홈팀</label><select id="home"></select></div>
      <div class="vs">vs</div>
      <div><label for="away">원정팀</label><select id="away"></select></div>
    </div>
    <div class="run-row">
      <select id="n" aria-label="시뮬 횟수">
        <option value="1000">1,000회</option>
        <option value="10000" selected>10,000회</option>
        <option value="100000">100,000회</option>
      </select>
      <button class="primary" id="run">시뮬레이션</button>
      <span class="status" id="status"></span>
    </div>
    <details class="excl" id="exclBox">
      <summary>선수 제외 (부상·징계 반영) · 체크를 끄면 그 선수는 선발로 안 뽑힘</summary>
      <div class="excl-cols"><div class="excl-col" id="exclH"></div><div class="excl-col" id="exclA"></div></div>
    </details>
  </div>

  <div id="out" class="hidden">
    <div class="grid">
      <div class="card">
        <h2>승무패</h2>
        <div class="wdl" id="wdl"></div>
        <div class="wdl-legend" id="wdlLegend"></div>
        <div class="stats" id="stats"></div>
      </div>
      <div class="card">
        <h2>스코어 분포 <span class="sub">(행: 홈 골, 열: 원정 골)</span></h2>
        <div class="heat" id="heat"></div>
      </div>
      <div class="card">
        <h2>많이 나온 스코어 TOP 8</h2>
        <table id="top"></table>
      </div>
      <div class="card">
        <h2>팀 파라미터 <span class="sub">(1.00 = 리그 평균)</span></h2>
        <table id="teamTbl"></table>
      </div>
      <div class="card full">
        <h2>골 차 분포 <span class="sub">(홈 골 − 원정 골, 색은 아래 핸디 기준 결과)</span></h2>
        <div class="gd" id="gd"></div>
        <div class="gd-legend" id="gdLegend"></div>
      </div>
      <div class="card full">
        <h2>핸디캡 계산기 <span class="sub">(홈팀 기준 핸디 · 배당을 넣으면 기대값 계산)</span></h2>
        <div class="hc-inputs">
          <label>홈 핸디<input id="hcLine" type="number" step="0.5" value="-1"></label>
          <label>핸디승 배당<input id="oddW" type="number" step="0.01" min="1" placeholder="예: 2.10"></label>
          <label id="oddDWrap">핸디무 배당<input id="oddD" type="number" step="0.01" min="1" placeholder="예: 3.40"></label>
          <label>핸디패 배당<input id="oddL" type="number" step="0.01" min="1" placeholder="예: 2.80"></label>
          <label>안전마진 (%p)<input id="margin" type="number" step="1" min="0" value="5"></label>
        </div>
        <div class="hc-wrap"><table id="hcTbl"></table></div>
        <p class="note">가치 있음 = 기대값이 플러스이고, 시뮬 확률이 배당 내재확률(1 ÷ 배당)보다 안전마진 이상 높을 때. 공정 배당 = 1 ÷ 시뮬 확률.</p>
      </div>
      <div class="card full">
        <div class="head-row">
          <h2 id="sampleTitle">샘플 경기</h2>
          <button class="ghost" id="again">다른 경기 보기</button>
        </div>
        <div class="lineups" id="lineups"></div>
        <div class="log" id="log"></div>
      </div>
    </div>
    <p class="note" id="note"></p>
  </div>
</div>
<div class="tip" id="tip"></div>

<script>
const P = __PARAMS__;
const PROMOTED = "승격팀 (기본값)";
const BINS = [[0,15,"0-15"],[16,30,"16-30"],[31,45,"31-45"],[46,60,"46-60"],[61,75,"61-75"],[76,90,"76-90"]];
const SIT_KO = {OpenPlay:"오픈플레이", corner:"코너", ifk:"간접 프리킥", dfk:"직접 프리킥", Penalty:"페널티", OwnGoal:"자책골"};
const LEAGUE_KO = {EPL:"EPL", La_Liga:"라리가", Bundesliga:"분데스리가"};
const $ = id => document.getElementById(id);

// ---------- 준비 ----------
function tp(name) { return name === PROMOTED ? P.promoted_default : P.teams[name]; }
function pmf(obj) {
  const k = Object.keys(obj).map(Number), p = k.map(x => obj[x]);
  const s = p.reduce((a, b) => a + b, 0); let c = 0;
  return { k, cum: p.map(v => (c += v / s)) };
}
const INJ1 = pmf(P.added_time.first), INJ2 = pmf(P.added_time.second);
function pick(d) { const u = Math.random(); for (let i = 0; i < d.cum.length; i++) if (u <= d.cum[i]) return d.k ? d.k[i] : i; return d.k ? d.k[d.k.length - 1] : d.cum.length - 1; }
function poisson(lam) { if (lam <= 0) return 0; const L = Math.exp(-lam); let k = 0, p = Math.random(); while (p > L) { k++; p *= Math.random(); } return k; }
function drawXg(sit) {
  const q = P.xg_quantiles[sit], x = Math.random() * (q.length - 1), i = Math.floor(x);
  return i >= q.length - 1 ? q[q.length - 1] : q[i] + (q[i + 1] - q[i]) * (x - i);
}
function timeMult(half, m, stoppage) {
  if (stoppage) return half === 1 ? P.time_mult["46-60"] : P.time_mult["90+"];
  for (const [lo, hi, n] of BINS) if (m >= lo && m <= hi) return P.time_mult[n];
  return 1;
}

// ---------- 선수 단위 ----------
const PL = __PLAYERS__, SH = __SHAPES__, C = __CHAIN__, CORR_C1 = __CORR__;
const SHAPE_DECAY = 0.9, OPP_SIGMA = 0.35, HOME_BONUS = 1.5, GROUPS = ["DEF", "MID", "FWD"];
const GROUP_KO = { GK: "GK", DEF: "수비", MID: "미드", FWD: "공격" };
const GI = { DEF: 0, MID: 1, FWD: 2 }, NSP = C.sp_types.length;
const EXCL = {};   // 팀 → 제외한 선수 이름 Set
function ct(name) { return C.teams[name] || C.promoted_default; }
function power(t) { const T = ct(t); return (T.att_open * T.att_q) / (T.def_open * T.def_q); }
for (const ps of Object.values(PL)) for (const p of ps) if (p.group !== "GK") p.c = C.players[p.pid] || C.group_mean[p.group] || C.group_mean.MID;
function wpick(w) {   // 가중치 배열에서 하나 뽑기
  let tot = 0; for (const v of w) tot += v;
  let u = Math.random() * tot, i = 0;
  for (; i < w.length - 1; i++) { u -= w[i]; if (u < 0) break; }
  return i;
}
function squad(team, useExcl) {
  const ex = useExcl ? (EXCL[team] || new Set()) : new Set(), sq = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const p of PL[team] || []) if (!ex.has(p.name) && sq[p.group]) sq[p.group].push(p);
  return sq;
}
function shapeDist(team, opp, isHome, generic) {
  const lst = SH[team];
  if (!lst || !lst.length) return { keys: ["4-5-1"], cum: [1] };
  const lp = Math.log(power(opp)), agg = new Map();
  for (const e of lst) {
    let w = Math.pow(SHAPE_DECAY, e.k);
    if (!generic) { w *= Math.exp(-Math.abs(Math.log(power(e.opp)) - lp) / OPP_SIGMA); if (e.home === isHome) w *= HOME_BONUS; }
    const key = `${e.DEF}-${e.MID}-${e.FWD}`;
    agg.set(key, (agg.get(key) || 0) + w);
  }
  const keys = [...agg.keys()], tot = [...agg.values()].reduce((a, b) => a + b, 0);
  let c = 0;
  return { keys, cum: keys.map(k => (c += agg.get(k) / tot)) };
}
function pickKey(d) { const u = Math.random(); for (let i = 0; i < d.cum.length; i++) if (u <= d.cum[i]) return d.keys[i]; return d.keys[d.keys.length - 1]; }
function topk(arr, c, g) {   // 선발 확률 가중치로 중복 없이 c명 (Gumbel)
  const keyed = arr.map(p => [Math.log(Math.max(p.start_prob, 1e-4)) - Math.log(-Math.log(Math.random())), p]);
  keyed.sort((a, b) => b[0] - a[0]);
  const out = keyed.slice(0, c).map(x => x[1]);
  while (out.length < c) out.push({ name: "(대체 선수)", group: g, start_prob: 0, finish: 1, c: C.group_mean[g] });
  return out;
}
function drawLineup(sq, dist) {
  const shape = pickKey(dist), nums = shape.split("-").map(Number);
  const gk = sq.GK.length ? topk(sq.GK, 1, "GK")[0] : { name: "(대체 골키퍼)", gk_save: 1 };
  let field = [];
  GROUPS.forEach((g, i) => field.push(...topk(sq[g], nums[i], g)));
  field = field.slice(0, 10);
  let sumB = 0, sumS = 0;
  for (const p of field) { sumB += p.c.b; sumS += p.c.s; }
  return { shape, gk, field, sumB, sumS, kw: field.map(p => p.c.k + 1e-9), gi: field.map(p => GI[p.group]) };
}
function openPick(L) {   // 오픈플레이 슈팅 하나: 패스 받았나 → 패스한 선수 → 슈터
  const ast = Math.random() < C.p_assist, pj = wpick(L.kw), pg = L.gi[pj];
  const w = L.field.map((p, j) => ast ? (j === pj ? 0 : p.c.s * C.flow[pg][L.gi[j]]) : p.c.s);
  const sj = wpick(w.map(v => v + 1e-12));
  return { ast, pj, sj, m: L.field[sj].c.rs * (ast ? L.field[pj].c.rp : 1) };
}
const TYP = {};
function typical(team) {   // 평소 라인업의 빌드업 합, 슈팅 합, 질 배수 평균 (제외 없이, 구성 가중치 없이)
  if (!TYP[team]) {
    const sq = squad(team, false), d = shapeDist(team, team, true, true), N = 4000;
    let b = 0, s = 0, m = 0;
    for (let i = 0; i < N; i++) { const L = drawLineup(sq, d); b += L.sumB; s += L.sumS; for (let j = 0; j < 4; j++) m += openPick(L).m; }
    TYP[team] = { b: b / N, s: s / N, m: m / (N * 4) };
  }
  return TYP[team];
}

function setup(home, away) {
  const H = tp(home), A = tp(away), h = P.home_mult;
  const mk = (T, O, hm, team, opp, isHome) => {
    const TC = ct(team), OC = ct(opp);
    const X = { open: C.base_open * hm * TC.att_open * OC.def_open, q0: TC.att_q * OC.def_q,
                sp: C.sp_rate.map(r => r * hm * TC.att_sp * OC.def_sp),
                pen: P.penalty.per_min * hm * T.att_shots * O.def_shots,
                og: C.og_c !== undefined ? C.og_c * Math.pow(hm * TC.att_open * OC.def_open, C.og_gamma) : (P.own_goal_per_min || 0) };
    if (PL[team] && PL[team].length) { X.sq = squad(team, true); X.dist = shapeDist(team, opp, isHome, false); X.typ = typical(team); }
    return X;
  };
  return [mk(H, A, h, home, away, true), mk(A, H, 1 / h, away, home, false)];
}
function lineupFor(X) {   // 시뮬 1회마다: 선발 구성 추첨 → 선수 추첨 → 이번 판 공격 횟수·슈팅 질
  if (!X.sq) return { shot: X.open, q: X.q0, L: null };
  const L = drawLineup(X.sq, X.dist);
  const vol = Math.sqrt((L.sumB / X.typ.b) * (L.sumS / X.typ.s));
  return { shot: X.open * vol, q: X.q0 / X.typ.m, L };
}

// ---------- 한 경기 ----------
function playMatch(S, log) {
  const R = [lineupFor(S[0]), lineupFor(S[1])];   // 1~2. 구성·선수 추첨
  const gkOpp = [R[1].L ? (R[1].L.gk.gk_save || 1) : 1, R[0].L ? (R[0].L.gk.gk_save || 1) : 1];
  const g = [0, 0];
  function minute(half, m, st) {
    const tm = timeMult(half, m, st);
    const label = half === 1 ? (m > 45 ? `45+${m - 45}'` : `${m}'`) : (m > 90 ? `90+${m - 90}'` : `${m}'`);
    for (let s = 0; s < 2; s++) {
      const X = S[s], r = R[s], L = r.L;
      // 오픈플레이: 공격 발생 → 패스 → 슈터 → xG → 골
      for (let j = poisson(r.shot * tm); j > 0; j--) {
        let who = null, pass = null, build = null, fin = 1, m2 = 1;
        if (L) {
          const k = openPick(L); m2 = k.m;
          who = L.field[k.sj].name; fin = L.field[k.sj].finish || 1;
          if (k.ast) pass = L.field[k.pj].name;
          if (log) build = L.field[wpick(L.field.map((p, i) => (i === k.sj || (k.ast && i === k.pj)) ? 0 : p.c.b + 1e-9))].name;
        }
        const xg = Math.min(0.97, drawXg("OpenPlay") * r.q * m2);
        const goal = Math.random() < Math.min(0.97, xg * C.calib_open * fin * gkOpp[s]);
        if (goal) g[s]++;
        if (log) log.push({ label, s, sit: "OpenPlay", xg, goal, who, pass, build, score: [g[0], g[1]] });
      }
      // 세트피스 3종류: 발생 → 슈터 → 슈터의 세트피스 평균 xG → 골
      for (let ti = 0; ti < NSP; ti++) for (let j = poisson(X.sp[ti] * tm); j > 0; j--) {
        let who = null, xg = C.sp_xg[ti];
        if (L) { const sj = wpick(L.field.map(p => p.c.sw[ti] + 1e-9)); who = L.field[sj].name; xg = L.field[sj].c.sx[ti]; }
        const goal = Math.random() < Math.min(0.97, xg * C.sp_calib[ti] * gkOpp[s]);
        if (goal) g[s]++;
        if (log) log.push({ label, s, sit: C.sp_types[ti], xg, goal, who, score: [g[0], g[1]] });
      }
      if (poisson(X.pen * tm) > 0) {
        const goal = Math.random() < P.penalty.conv;
        if (goal) g[s]++;
        if (log) log.push({ label, s, sit: "Penalty", xg: P.penalty.conv, goal, who: null, score: [g[0], g[1]] });
      }
      if (poisson(X.og) > 0) {   // 상대 선수 자책골 → 이 팀 득점
        g[s]++;
        if (log) log.push({ label, s, sit: "OwnGoal", xg: null, goal: true, who: null, score: [g[0], g[1]] });
      }
    }
  }
  for (let m = 1; m <= 45; m++) minute(1, m, false);            // 3. 전반
  const inj1 = pick(INJ1);                                       // 4. 전반 추가시간 추첨
  if (log) log.push({ sep: `전반 추가시간 ${inj1}분` });
  for (let m = 46; m <= 45 + inj1; m++) minute(1, m, true);
  const ht = [g[0], g[1]];
  if (log) log.push({ sep: `하프타임 ${g[0]}-${g[1]}` });
  for (let m = 46; m <= 90; m++) minute(2, m, false);            // 5. 후반
  const inj2 = pick(INJ2);                                       // 6. 후반 추가시간 추첨
  if (log) log.push({ sep: `후반 추가시간 ${inj2}분` });
  for (let m = 91; m <= 90 + inj2; m++) minute(2, m, true);
  return { g, ht, inj1, inj2, lineups: [R[0].L, R[1].L] };       // 7. 1회 완료
}

// ---------- 대량 시뮬 (끊어서 돌려 화면 안 멈추게) ----------
function runMany(home, away, N, onDone) {
  const S = setup(home, away), MAX = 10;
  const mat = Array.from({ length: MAX + 1 }, () => new Float64Array(MAX + 1));
  let done = 0, sumH = 0, sumA = 0;
  const t0 = performance.now();
  // 전력 차 보정: 먼저 3000번 돌려 보정 전 평균 골차를 재고, 강팀 +c1·골차 / 약팀 −c1·골차
  let cor = null;
  if (CORR_C1 > 0) {
    const PN = 3000; let ph = 0, pa = 0;
    for (let i = 0; i < PN; i++) { const g = playMatch(S, null).g; ph += g[0]; pa += g[1]; }
    ph /= PN; pa /= PN;
    if (ph !== pa) {
      const s = ph > pa ? 0 : 1, delta = CORR_C1 * Math.abs(ph - pa);
      cor = { s, w: 1 - s, delta, keep: Math.max(0, 1 - delta / Math.max(s === 0 ? pa : ph, 1e-9)) };
    }
  }
  (function chunk() {
    const end = Math.min(N, done + 5000);
    for (; done < end; done++) {
      const { g } = playMatch(S, null);
      if (cor) {
        g[cor.s] += poisson(cor.delta);                                            // 강팀: 추가 골
        let k = 0; for (let j = 0; j < g[cor.w]; j++) if (Math.random() < cor.keep) k++;   // 약팀: 골마다 일정 확률로 취소
        g[cor.w] = k;
      }
      mat[Math.min(g[0], MAX)][Math.min(g[1], MAX)]++;
      sumH += g[0]; sumA += g[1];
    }
    $("status").textContent = `${done.toLocaleString()} / ${N.toLocaleString()}`;
    if (done < N) setTimeout(chunk, 0);
    else onDone({ mat, N, xgH: sumH / N, xgA: sumA / N, ms: performance.now() - t0 });
  })();
}

// ---------- 화면 ----------
const pct = x => (x * 100).toFixed(1) + "%";
function render(home, away, R) {
  const { mat, N } = R;
  let w = 0, d = 0, l = 0, o15 = 0, o25 = 0, o35 = 0, btts = 0;
  const cells = [];
  mat.forEach((row, i) => row.forEach((c, j) => {
    if (!c) return;
    const p = c / N; cells.push([i, j, p]);
    if (i > j) w += p; else if (i === j) d += p; else l += p;
    if (i + j > 1.5) o15 += p; if (i + j > 2.5) o25 += p; if (i + j > 3.5) o35 += p;
    if (i > 0 && j > 0) btts += p;
  }));

  $("wdl").innerHTML = `<div class="h" style="flex:${w}">${w > .08 ? pct(w) : ""}</div>` +
    `<div class="d" style="flex:${d}">${d > .08 ? pct(d) : ""}</div>` +
    `<div class="a" style="flex:${l}">${l > .08 ? pct(l) : ""}</div>`;
  $("wdlLegend").innerHTML = `<span><i class="dot" style="background:var(--home)"></i>${home} 승 ${pct(w)}</span>` +
    `<span>무 ${pct(d)}</span><span><i class="dot" style="background:var(--away)"></i>${away} 승 ${pct(l)}</span>`;
  const st = [["예상 골 (홈)", R.xgH.toFixed(2)], ["예상 골 (원정)", R.xgA.toFixed(2)], ["양팀 득점", pct(btts)],
              ["오버 1.5", pct(o15)], ["오버 2.5", pct(o25)], ["오버 3.5", pct(o35)]];
  $("stats").innerHTML = st.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  // 히트맵 0~5 + 6+
  const K = 6, g = (i, j) => {
    let s = 0;
    for (let a = 0; a < mat.length; a++) for (let b = 0; b < mat.length; b++)
      if ((i < K ? a === i : a >= K) && (j < K ? b === j : b >= K)) s += mat[a][b];
    return s / N;
  };
  const vals = []; for (let i = 0; i <= K; i++) for (let j = 0; j <= K; j++) vals.push(g(i, j));
  const vmax = Math.max(...vals);
  const heat = $("heat");
  heat.style.gridTemplateColumns = `34px repeat(${K + 1}, 1fr)`;
  let hx = `<div class="corner"></div>`;
  for (let j = 0; j <= K; j++) hx += `<div class="axis">${j < K ? j : "6+"}</div>`;
  for (let i = 0; i <= K; i++) {
    hx += `<div class="axis">${i < K ? i : "6+"}</div>`;
    for (let j = 0; j <= K; j++) {
      const p = vals[i * (K + 1) + j], lvl = p === 0 ? 0 : Math.min(6, 1 + Math.floor((p / vmax) * 5.999));
      const dark = lvl >= 4 ? "color:#fff" : "";
      hx += `<div class="cell" style="background:var(--h${lvl});${dark}" data-tip="${home} ${i < K ? i : "6+"} - ${j < K ? j : "6+"} ${away}: ${pct(p)}">${p >= 0.005 ? (p * 100).toFixed(1) : ""}</div>`;
    }
  }
  heat.innerHTML = hx;

  cells.sort((a, b) => b[2] - a[2]);
  const top = cells.slice(0, 8), tmax = top[0][2];
  $("top").innerHTML = `<tr><th>스코어</th><th>확률</th></tr>` + top.map(([i, j, p]) =>
    `<tr><td>${i} - ${j}</td><td><span class="bar" style="width:${(p / tmax) * 90}px;background:${i > j ? "var(--home)" : i < j ? "var(--away)" : "var(--draw)"}"></span>${pct(p)}</td></tr>`).join("");

  const H = ct(home), A = ct(away);
  const rows = [["오픈플레이 슈팅 생산", "att_open"], ["오픈플레이 슈팅 허용", "def_open"], ["슈팅 질", "att_q"], ["허용 슈팅 질", "def_q"],
                ["세트피스 슈팅 생산", "att_sp"], ["세트피스 슈팅 허용", "def_sp"]];
  $("teamTbl").innerHTML = `<tr><th></th><th>${home}</th><th>${away}</th></tr>` +
    rows.map(([k, c]) => `<tr><td>${k}</td><td>${(H[c] || 1).toFixed(2)}</td><td>${(A[c] || 1).toFixed(2)}</td></tr>`).join("");

  $("status").textContent = `${N.toLocaleString()}회 · ${(R.ms / 1000).toFixed(2)}초`;
  $("out").classList.remove("hidden");
  LAST = { mat, N }; renderGD(); renderHC();
  sample(home, away);
}

function sample(home, away) {
  const log = [], r = playMatch(setup(home, away), log);
  const names = [home, away];
  $("sampleTitle").textContent = `샘플 경기 · ${home} ${r.g[0]} - ${r.g[1]} ${away}`;
  $("lineups").innerHTML = r.lineups.map((L, i) => !L ? `<div><div class="shape">${names[i]} · 선수 데이터 없음 (팀 단위)</div></div>` :
    `<div><div class="shape">${names[i]} · ${L.shape} (수비-미드-공격)</div><div class="ppl">` +
    `<b>GK</b> ${L.gk.name}<br>` + GROUPS.map(g => `<b>${GROUP_KO[g]}</b> ` + L.field.filter(p => p.group === g).map(p => p.name).join(", ")).join("<br>") +
    `</div></div>`).join("");
  $("log").innerHTML = log.map(e => e.sep
    ? `<div class="ev sep">${e.sep}</div>`
    : `<div class="ev ${e.goal ? "goal" : ""}"><span class="t">${e.label}</span>` +
      `<span><i class="dot" style="background:${e.s ? "var(--away)" : "var(--home)"}"></i>${names[e.s]} · ${SIT_KO[e.sit] || e.sit}${e.who ? " · " + [e.build, e.pass, e.who].filter(Boolean).join(" → ") : ""}` +
      `${e.xg !== null ? ` · xG ${e.xg.toFixed(2)}` : ""}</span>` +
      `<span>${e.goal ? `골! ${e.score[0]}-${e.score[1]}` : ""}</span></div>`).join("") +
    `<div class="ev sep">종료 ${r.g[0]}-${r.g[1]}</div>`;
}

// ---------- 초기화 ----------
function fill(sel, def) {
  const by = {};
  for (const [t, v] of Object.entries(P.teams)) (by[v.league] = by[v.league] || []).push(t);
  let h = "";
  for (const lg of Object.keys(by).sort()) h += `<optgroup label="${LEAGUE_KO[lg] || lg}">` + by[lg].sort().map(t => `<option ${t === def ? "selected" : ""}>${t}</option>`).join("") + `</optgroup>`;
  h += `<optgroup label="기타"><option>${PROMOTED}</option></optgroup>`;
  sel.innerHTML = h;
}
const teams = Object.keys(P.teams);
fill($("home"), teams.includes("Arsenal") ? "Arsenal" : teams[0]);
fill($("away"), teams.includes("Chelsea") ? "Chelsea" : teams[1]);
$("meta").textContent = `${P.meta.season}/${(P.meta.season + 1) % 100} 시즌 · ${P.meta.leagues.map(l => LEAGUE_KO[l] || l).join(" + ")} · ${P.meta.matches.toLocaleString()}경기 기준`;
$("note").textContent = `v3 · 오픈플레이 분당 ${C.base_open} (빌드업 → 패스 → 슈터) · 세트피스 분당 ${C.sp_rate.reduce((a, b) => a + b, 0).toFixed(4)} (코너·간접·직접 프리킥 따로) · 홈 배수 ${P.home_mult} · 선수 결정력·골키퍼(최근 3시즌) 반영${CORR_C1 > 0 ? ` · 전력 차 보정 ${CORR_C1} × 예상 골차 (강팀 +, 약팀 −. 확률·분포에만 반영되고 샘플 경기는 보정 전)` : ""} · 샘플 경기의 이름 순서 = 빌드업 → 마지막 패스 → 슈터 · 게임상태·교체 미적용`;

$("run").onclick = () => {
  const home = $("home").value, away = $("away").value, N = +$("n").value;
  if (home === away && home !== PROMOTED) { $("status").textContent = "서로 다른 팀을 골라줘"; return; }
  $("run").disabled = true;
  runMany(home, away, N, R => { render(home, away, R); $("run").disabled = false; });
};

// ---------- 골 차 분포 + 핸디 계산기 ----------
let LAST = null;
function gdDist(mat, N) {
  const d = {};
  mat.forEach((row, i) => row.forEach((c, j) => { if (c) d[i - j] = (d[i - j] || 0) + c / N; }));
  return d;
}
function hcOutcome(gd, h) { const a = gd + h; return a > 1e-9 ? "W" : a < -1e-9 ? "L" : "D"; }
function renderGD() {
  if (!LAST) return;
  const d = gdDist(LAST.mat, LAST.N), h = +$("hcLine").value || 0;
  const cols = [];
  cols.push(["-4↓", Object.entries(d).filter(([k]) => +k <= -4).reduce((s, [, v]) => s + v, 0), -4]);
  for (let k = -3; k <= 3; k++) cols.push([k > 0 ? "+" + k : String(k), d[k] || 0, k]);
  cols.push(["+4↑", Object.entries(d).filter(([k]) => +k >= 4).reduce((s, [, v]) => s + v, 0), 4]);
  const vmax = Math.max(...cols.map(c => c[1]));
  const color = { W: "var(--home)", D: "var(--draw)", L: "var(--away)" };
  $("gd").innerHTML = cols.map(([lab, p, k]) =>
    `<div class="col" title="골 차 ${lab}: ${pct(p)}"><div class="pv">${pct(p)}</div>` +
    `<div class="b" style="height:${(p / vmax) * 130}px;background:${color[hcOutcome(k, h)]}"></div><div class="x">${lab}</div></div>`).join("");
  const hs = h > 0 ? "+" + h : String(h);
  $("gdLegend").innerHTML = `<span><i class="dot" style="background:var(--home)"></i>핸디승 (홈 ${hs})</span>` +
    (Number.isInteger(h) ? `<span><i class="dot" style="background:var(--draw)"></i>핸디무</span>` : "") +
    `<span><i class="dot" style="background:var(--away)"></i>핸디패</span>`;
}
function renderHC() {
  if (!LAST) return;
  const d = gdDist(LAST.mat, LAST.N), h = +$("hcLine").value || 0, integer = Number.isInteger(h);
  $("oddDWrap").style.display = integer ? "" : "none";
  const prob = { W: 0, D: 0, L: 0 };
  for (const [k, v] of Object.entries(d)) prob[hcOutcome(+k, h)] += v;
  const need = 1 - h;   // 핸디승에 필요한 골 차 (정수 핸디)
  const cond = {
    W: integer ? `골 차 ≥ ${need}` : `골 차 ≥ ${Math.ceil(-h)}`,
    D: `골 차 = ${-h}`,
    L: integer ? `골 차 ≤ ${-h - 1}` : `골 차 ≤ ${Math.floor(-h)}`,
  };
  const odds = { W: +$("oddW").value, D: +$("oddD").value, L: +$("oddL").value };
  const margin = (+$("margin").value || 0) / 100;
  const names = { W: "핸디승", D: "핸디무", L: "핸디패" };
  const keys = integer ? ["W", "D", "L"] : ["W", "L"];
  $("hcTbl").innerHTML = `<tr><th>결과</th><th>조건 (홈−원정)</th><th>시뮬 확률</th><th>공정 배당</th><th>입력 배당</th><th>내재확률</th><th>차이</th><th>기대값</th><th>판정</th></tr>` +
    keys.map(k => {
      const p = prob[k], o = odds[k], fair = p > 0 ? (1 / p).toFixed(2) : "-";
      if (!(o > 1)) return `<tr><td>${names[k]}</td><td>${cond[k]}</td><td>${pct(p)}</td><td>${fair}</td><td>-</td><td>-</td><td>-</td><td>-</td><td class="v-bad">배당 입력</td></tr>`;
      const imp = 1 / o, ev = p * o - 1, gap = p - imp;
      const v = ev > 0 && gap >= margin ? ["v-good", "✓ 가치 있음"] : ev > 0 ? ["v-mid", "△ 마진 미달"] : ["v-bad", "✗ 기대값 마이너스"];
      return `<tr><td>${names[k]}</td><td>${cond[k]}</td><td>${pct(p)}</td><td>${fair}</td><td>${o.toFixed(2)}</td><td>${pct(imp)}</td>` +
        `<td>${gap >= 0 ? "+" : ""}${(gap * 100).toFixed(1)}p</td><td>${ev >= 0 ? "+" : ""}${(ev * 100).toFixed(1)}%</td><td class="verdict ${v[0]}">${v[1]}</td></tr>`;
    }).join("");
}
["hcLine", "oddW", "oddD", "oddL", "margin"].forEach(id => $(id).addEventListener("input", () => { renderGD(); renderHC(); }));

// ---------- 선수 제외 패널 ----------
function fillExcl(boxId, team) {
  const box = $(boxId), ps = (PL[team] || []).filter(p => p.start_prob >= 0.02);
  if (!ps.length) { box.innerHTML = `<h3>${team}</h3><div class="note">선수 데이터 없음</div>`; return; }
  const ex = EXCL[team] || (EXCL[team] = new Set());
  let h = `<h3>${team}</h3>`;
  for (const g of ["GK", ...GROUPS]) {
    const arr = ps.filter(p => p.group === g).sort((a, b) => b.start_prob - a.start_prob);
    if (!arr.length) continue;
    h += `<div class="grp">${GROUP_KO[g]}</div>` + arr.map(p =>
      `<label><input type="checkbox" data-team="${team}" data-name="${p.name}" ${ex.has(p.name) ? "" : "checked"}>${p.name}<span class="m">선발 ${Math.round(p.start_prob * 100)}%</span></label>`).join("");
  }
  box.innerHTML = h;
}
function refreshExcl() { fillExcl("exclH", $("home").value); fillExcl("exclA", $("away").value); }
$("exclBox").addEventListener("change", e => {
  const c = e.target; if (!c.dataset || !c.dataset.team) return;
  const ex = EXCL[c.dataset.team] || (EXCL[c.dataset.team] = new Set());
  if (c.checked) ex.delete(c.dataset.name); else ex.add(c.dataset.name);
});
$("home").addEventListener("change", refreshExcl);
$("away").addEventListener("change", refreshExcl);
refreshExcl();
$("again").onclick = () => sample($("home").value, $("away").value);

const tip = $("tip");
$("heat").addEventListener("mousemove", e => {
  const c = e.target.closest(".cell"); if (!c) { tip.style.display = "none"; return; }
  tip.textContent = c.dataset.tip; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = (e.clientY + 14) + "px";
});
$("heat").addEventListener("mouseleave", () => tip.style.display = "none");
</script>
</body>
</html>
"""

out = HERE / "simulator.html"
out.write_text(TEMPLATE.replace("__PARAMS__", json.dumps(params, ensure_ascii=False))
               .replace("__PLAYERS__", json.dumps(players, ensure_ascii=False))
               .replace("__SHAPES__", json.dumps(shapes, ensure_ascii=False))
               .replace("__CHAIN__", json.dumps(chain, ensure_ascii=False))
               .replace("__CORR__", json.dumps(corr_c1)), encoding="utf-8")
print(f"생성: {out}  (팀 {len(params['teams'])}개, 선수 데이터 {len(players)}팀, 선발 구성 {len(shapes)}팀) → 더블클릭해서 크롬으로 열면 돼")
