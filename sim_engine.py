"""
시뮬 엔진 v3 (HTML과 같은 로직) - 검증 코드들이 공용으로 씀
시뮬 1회:
 1. 선발 구성 추첨 (수비-미드-공격 인원, 최근성 × 상대 강도 × 홈/원정 가중치)
 2. 선수 추첨 (그룹별 선발 확률, 중복 없이)
 3. 매 분 (전반 → 전반 추가시간 → 후반 → 후반 추가시간)
    [오픈플레이] 슈팅까지 간 공격 발생 → 마지막 패스한 선수 → 슈터 → xG → 골
                 골 확률 = xG × 보정 × 슈터 결정력(선수의 최근 3시즌 기록) × 상대 골키퍼(최근 3시즌 기록)
    [세트피스]   코너 / 간접FK / 직접FK 각각 발생 → 슈터 → 슈터의 세트피스 평균 xG → 골
    [페널티] [자책골: 밀어붙이는 팀일수록 상대 자책골을 더 얻음]
 4. 전력 차 보정 (data/corr.json 이 있을 때): 강팀 골을 조금 더하고 약팀 골을 조금 뺌
use_players=False 로 부르면 선수 없이 팀 값만으로 돌아감 (비교용)
"""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
P = json.loads((DATA / "params.json").read_text(encoding="utf-8"))
SEASON = P["meta"]["season"]
rng = np.random.default_rng(0)

_lt = DATA / "live_teams.json"
LIVE = _lt.exists() and (DATA / "players_live.json").exists() and (DATA / "chain_live.json").exists()   # 살아있는 값 3종이 다 있으면 지금 기준으로
_cf = DATA / ("chain_live.json" if LIVE else "chain.json")
if not _cf.exists():
    sys.exit("data/chain.json 이 없어. 먼저 py prepare_chain.py 를 돌려줘")
C = json.loads(_cf.read_text(encoding="utf-8"))
_pf = DATA / ("players_live.json" if LIVE else f"players_{SEASON}.json")
PL = json.loads(_pf.read_text(encoding="utf-8"))["teams"] if _pf.exists() else {}
_sf = DATA / ("shapes_live.json" if LIVE and (DATA / "shapes_live.json").exists() else "shapes.json")
SH = json.loads(_sf.read_text(encoding="utf-8"))["teams"] if _sf.exists() else {}

_kf = DATA / "corr.json"      # 전력 차 보정값 (fit_corr.py 가 만듦. 없으면 보정 안 함)
CORR_C1 = float(json.loads(_kf.read_text(encoding="utf-8")).get("c1", 0.0)) if _kf.exists() else 0.0

if LIVE:   # 팀 값을 살아있는 값(그 시점까지 기록)으로 덮어씀. 페널티용 att_shots/def_shots 는 오픈플레이 슈팅 값으로
    _live = json.loads(_lt.read_text(encoding="utf-8"))
    LIVE_META = _live.get("meta", {})
    for _t, _v in _live["teams"].items():
        C["teams"][_t] = {k: _v[k] for k in ("att_open", "def_open", "att_q", "def_q", "att_sp", "def_sp")}
        P["teams"][_t] = {"att_shots": _v["att_open"], "def_shots": _v["def_open"], "att_q": _v["att_q"], "def_q": _v["def_q"],
                          "league": _v.get("league")}
else:
    LIVE_META = {}

SHAPE_DECAY = 0.9     # 선발 구성: 한 경기 전마다 ×0.9
OPP_SIGMA = 0.35      # 상대 강도가 이만큼(로그) 다르면 가중치 1/e
HOME_BONUS = 1.5      # 홈/원정이 같은 경기 가중치
GROUPS = ("DEF", "MID", "FWD")
NSP = len(C["sp_types"])
FLOW = np.array(C["flow"])            # [패스 그룹][슈터 그룹] 궁합
N_TYP = 20000                         # '평소 라인업' 계산용 추첨 횟수

# ---------- 공통 ----------
def pmf(d):
    k = np.array([int(x) for x in d]); p = np.array([d[x] for x in d], float)
    return k, p / p.sum()

INJ1, INJ2 = pmf(P["added_time"]["first"]), pmf(P["added_time"]["second"])
XQ_OPEN = np.array(P["xg_quantiles"]["OpenPlay"])
BINS = [(0, 15, "0-15"), (16, 30, "16-30"), (31, 45, "31-45"), (46, 60, "46-60"), (61, 75, "61-75"), (76, 90, "76-90")]
TM = P["time_mult"]

def tmult(m):
    for lo, hi, n in BINS:
        if lo <= m <= hi:
            return TM[n]
    return 1.0

def draw_xg(t):
    x = rng.random(t) * (len(XQ_OPEN) - 1)
    j = np.floor(x).astype(int).clip(0, len(XQ_OPEN) - 2)
    return XQ_OPEN[j] + (XQ_OPEN[j + 1] - XQ_OPEN[j]) * (x - j)

def tp(name):
    return P["teams"].get(name, P["promoted_default"])

def ct(name):
    return C["teams"].get(name, C["promoted_default"])

def power(name):
    T = ct(name)
    return (T["att_open"] * T["att_q"]) / (T["def_open"] * T["def_q"])

# ---------- 선수 단위 ----------
def cstat(p):
    return C["players"].get(str(p.get("pid"))) or C["group_mean"].get(p["group"]) or C["group_mean"]["MID"]

def squad(team, excluded=()):
    ps = [p for p in PL.get(team, []) if p["name"] not in excluded]
    sq = {}
    for g in ("GK",) + GROUPS:
        arr = [p for p in ps if p["group"] == g]
        cs = [cstat(p) for p in arr] if g != "GK" else []
        col = lambda k: np.array([c[k] for c in cs], float)
        sq[g] = {"name": [p["name"] for p in arr],
                 "sp": np.array([p["start_prob"] for p in arr], float),
                 "fin": np.array([p.get("finish") or 1.0 for p in arr], float),
                 "gk": np.array([p.get("gk_save") or 1.0 for p in arr], float)}
        if g != "GK":
            sq[g].update(b=col("b"), k=col("k"), s=col("s"), rs=col("rs"), rp=col("rp"),
                         sw=np.array([c["sw"] for c in cs], float).reshape(-1, NSP),
                         sx=np.array([c["sx"] for c in cs], float).reshape(-1, NSP))
    return sq

def shape_dist(team, opp, is_home, generic=False):
    lst = SH.get(team)
    if not lst:
        return [(4, 5, 1)], np.array([1.0])
    lp = np.log(power(opp))
    agg = {}
    for e in lst:
        w = SHAPE_DECAY ** e["k"]
        if not generic:
            w *= np.exp(-abs(np.log(power(e["opp"])) - lp) / OPP_SIGMA)
            w *= HOME_BONUS if e["home"] == is_home else 1.0
        key = (e["DEF"], e["MID"], e["FWD"])
        agg[key] = agg.get(key, 0) + w
    keys = list(agg); p = np.array([agg[k] for k in keys]); return keys, p / p.sum()

def topk(w, c, n):
    """가중치 w로 중복 없이 c명 뽑기를 n번 (Gumbel 방식)"""
    g = -np.log(-np.log(rng.random((n, len(w)))))
    return np.argsort(-(np.log(np.maximum(w, 1e-4)) + g), axis=1)[:, :c]

def draw_lineups(sq, keys, probs, n):
    """n번의 선발 10명(필드) + 골키퍼. 선수별 수치를 (n, 10) 배열로 돌려줌"""
    choice = rng.choice(len(keys), n, p=probs)
    L = {k: np.zeros((n, 10)) for k in ("B", "K", "S")}
    L.update(RS=np.ones((n, 10)), RP=np.ones((n, 10)), F=np.ones((n, 10)), G=np.zeros((n, 10), int),
             SW=np.zeros((n, 10, NSP)), SX=np.full((n, 10, NSP), 0.1), GK=np.ones(n))
    if len(sq["GK"]["sp"]):
        L["GK"] = sq["GK"]["gk"][topk(sq["GK"]["sp"], 1, n)[:, 0]]
    for si, shape in enumerate(keys):
        rows = np.where(choice == si)[0]
        if not len(rows):
            continue
        col = 0
        for gi, (g, c) in enumerate(zip(GROUPS, shape)):
            if c == 0 or col >= 10:
                continue
            c = min(c, 10 - col)
            a, d = sq[g], C["group_mean"][g]
            # 인원이 모자라면 포지션 평균 '대체 선수'로 채움 (가중치 아주 낮게)
            ext = lambda arr, v: np.concatenate([arr, np.full(c, v)])
            sel = topk(ext(a["sp"], 1e-4), c, len(rows))
            sl = np.s_[rows[:, None], np.arange(col, col + c)[None, :]]
            L["B"][sl] = ext(a["b"], d["b"])[sel]; L["K"][sl] = ext(a["k"], d["k"])[sel]
            L["S"][sl] = ext(a["s"], d["s"])[sel]; L["RS"][sl] = ext(a["rs"], d["rs"])[sel]
            L["RP"][sl] = ext(a["rp"], d["rp"])[sel]; L["F"][sl] = ext(a["fin"], 1.0)[sel]
            L["G"][sl] = gi
            L["SW"][sl] = np.concatenate([a["sw"], np.tile(d["sw"], (c, 1))])[sel]
            L["SX"][sl] = np.concatenate([a["sx"], np.tile(d["sx"], (c, 1))])[sel]
            col += c
    L["sumB"], L["sumS"] = L["B"].sum(1), L["S"].sum(1)
    return L

def pick_col(W):
    """행마다 가중치 W로 열 하나 뽑기"""
    cw = np.cumsum(W, axis=1)
    return (rng.random(len(W))[:, None] * cw[:, -1:] > cw).sum(1).clip(0, W.shape[1] - 1)

def open_pick(L, idx):
    """오픈플레이 슈팅 하나마다: 패스 받았나 → 패스한 선수 → 슈터. (패스 열, 슈터 열, 패스 여부, 질 배수)"""
    t = len(idx); ar = np.arange(t)
    ast = rng.random(t) < C["p_assist"]
    pj = pick_col(L["K"][idx] + 1e-9)                          # 패스한 선수: 키패스 비율대로
    G = L["G"][idx]
    W = L["S"][idx] * np.where(ast[:, None], FLOW[G[ar, pj]][ar[:, None], G], 1.0)   # 슈터: 슈팅 × 궁합
    W[ar, pj] *= ~ast                                           # 패스한 본인은 슈터에서 제외
    sj = pick_col(W + 1e-12)
    m = L["RS"][idx, sj] * np.where(ast, L["RP"][idx, pj], 1.0)  # 슈터 슈팅 질 × 패스한 선수 패스 질
    return pj, sj, ast, m

_typ = {}
def typical(team):
    """이 팀 '평소 라인업'의 빌드업 합, 슈팅 합, 질 배수 평균 (제외 선수 없이, 구성 가중치 없이)"""
    if team not in _typ:
        k, p = shape_dist(team, team, True, generic=True)
        L = draw_lineups(squad(team), k, p, N_TYP)
        m = open_pick(L, np.arange(N_TYP))[3]
        _typ[team] = (L["sumB"].mean(), L["sumS"].mean(), m.mean())
    return _typ[team]

# ---------- 시뮬 ----------
LAST = {}     # 마지막 simulate의 경기당 평균 (확인용): 골 종류별 [홈, 원정] + 슈팅 수·xG 합 (오픈/세트피스)

def simulate(home, away, n, use_players=True, excluded=(), corr=True):
    H, A, h = tp(home), tp(away), P["home_mult"]
    sides, L = [], []
    for T, O, hm, team, opp, is_home in ((H, A, h, home, away, True), (A, H, 1 / h, away, home, False)):
        TC, OC = ct(team), ct(opp)
        shot = np.full(n, C["base_open"] * hm * TC["att_open"] * OC["def_open"])
        q = TC["att_q"] * OC["def_q"]
        lu = None
        if use_players and PL.get(team):
            k, p = shape_dist(team, opp, is_home)
            lu = draw_lineups(squad(team, excluded), k, p, n)
            tB, tS, tM = typical(team)
            shot = shot * np.sqrt((lu["sumB"] / tB) * (lu["sumS"] / tS))     # 라인업 → 공격 횟수
            q = q / tM
        sides.append({"shot": shot, "q": q,
                      "sp": [r * hm * TC["att_sp"] * OC["def_sp"] for r in C["sp_rate"]],
                      "pen": P["penalty"]["per_min"] * hm * T["att_shots"] * O["def_shots"],
                      # 자책골로 얻는 골: 밀어붙일수록 많아짐 (chain.json에 없으면 예전처럼 고정값)
                      "og": (C["og_c"] * (hm * TC["att_open"] * OC["def_open"]) ** C["og_gamma"]) if "og_c" in C
                            else P.get("own_goal_per_min", 0)})
        L.append(lu)
    for s in (0, 1):   # 상대 골키퍼
        o = L[1 - s]
        sides[s]["gk"] = o["GK"] if o is not None else np.ones(n)

    inj1 = rng.choice(INJ1[0], n, p=INJ1[1]); inj2 = rng.choice(INJ2[0], n, p=INJ2[1])
    g = np.zeros((2, n), int)
    src = {k: np.zeros(2) for k in ("open", "sp", "pen", "og", "n_open", "xg_open", "n_sp", "xg_sp")}
    first_goal = np.full(n, 999)
    steps = [(m, tmult(m), None) for m in range(1, 46)] + [(45, TM["46-60"], ("inj1", k)) for k in range(1, 11)] + \
            [(m, tmult(m), None) for m in range(46, 91)] + [(90, TM["90+"], ("inj2", k)) for k in range(1, 16)]
    for clock, tm, stp in steps:
        active = np.ones(n, bool) if stp is None else ((inj1 if stp[0] == "inj1" else inj2) >= stp[1])
        for s in (0, 1):
            X, lu = sides[s], L[s]
            scored = np.zeros(n, int)
            # --- 오픈플레이 ---
            k = rng.poisson(X["shot"] * tm) * active
            tot = k.sum()
            if tot:
                idx = np.repeat(np.arange(n), k)
                m, fin = np.ones(tot), np.ones(tot)
                if lu is not None:
                    _, sj, _, m = open_pick(lu, idx)
                    fin = lu["F"][idx, sj]
                xg = np.minimum(0.97, draw_xg(tot) * X["q"] * m)
                src["n_open"][s] += tot; src["xg_open"][s] += xg.sum()
                goal = rng.random(tot) < np.minimum(0.97, xg * C["calib_open"] * fin * X["gk"][idx])
                add = np.bincount(idx[goal], minlength=n); scored += add; src["open"][s] += add.sum()
            # --- 세트피스 3종류 ---
            for ti in range(NSP):
                k = rng.poisson(X["sp"][ti] * tm, n) * active
                tot = k.sum()
                if not tot:
                    continue
                idx = np.repeat(np.arange(n), k)
                if lu is not None:
                    sj = pick_col(lu["SW"][idx, :, ti] + 1e-9)        # 슈터: 그 종류 세트피스 슈팅 빈도대로
                    xg = lu["SX"][idx, sj, ti]                         # 슈터의 세트피스 평균 xG
                else:
                    xg = np.full(tot, C["sp_xg"][ti])
                src["n_sp"][s] += tot; src["xg_sp"][s] += xg.sum()
                goal = rng.random(tot) < np.minimum(0.97, xg * C["sp_calib"][ti] * X["gk"][idx])
                add = np.bincount(idx[goal], minlength=n); scored += add; src["sp"][s] += add.sum()
            # --- 페널티, 자책골 ---
            pen = (rng.poisson(X["pen"] * tm, n) > 0) & active & (rng.random(n) < P["penalty"]["conv"])
            og = (rng.poisson(X["og"], n) > 0) & active
            scored += pen; scored += og
            src["pen"][s] += pen.sum(); src["og"][s] += og.sum()
            new = (scored > 0) & (first_goal == 999)
            first_goal[new] = clock
            g[s] += scored
    LAST.clear(); LAST.update({k: (v / n).round(3).tolist() for k, v in src.items()})
    # --- 전력 차 보정: 보정 전 평균 골차 x 에 대해 강팀 +c1·x 골, 약팀 −c1·x 골 ---
    mh, ma = float(g[0].mean()), float(g[1].mean())
    LAST["raw"] = [mh, ma]                       # 보정 전 평균 골 (보정값을 다시 찾을 때 씀)
    if corr and CORR_C1 > 0 and mh != ma:
        s_, w_ = (0, 1) if mh > ma else (1, 0)
        delta = CORR_C1 * abs(mh - ma)
        g[s_] += rng.poisson(delta, n)                                   # 강팀: 평균 delta 골짜리 추가 골
        keep = max(0.0, 1 - delta / max(g[w_].mean(), 1e-9))
        g[w_] = rng.binomial(g[w_], keep)                                # 약팀: 골 하나하나를 일정 확률로 취소
    return g[0], g[1], first_goal
