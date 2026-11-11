"""
A. 스쿼드로 팀 값 보정 (11/11) — 시뮬 없이 xG로 채점. 몇 분이면 끝남
  팀 값(살아있는 값)은 지난 경기들로 쌓이니까 시즌 초·이적 직후엔 늦게 따라감
  선수 기록(xGChain, 다른 팀에서 뛴 것 포함)은 바로 따라감 → 이번 경기 선발 10명의 xGChain/90 합 = '스쿼드 기준 공격력'
  B = log(스쿼드 기준 공격력) − log(팀 값 공격력)  →  +면 '팀 값보다 지금 선수들이 더 세다'
  (11/11 수정) B 자체는 '선수 값이 평균 쪽으로 눌린 정도'가 섞여서(승격팀 ↑, 강팀 ↓) 안 됨
  → dB = 이번 B − 그 팀의 지난 경기들 B 평균 (반감기 10경기, 시즌 넘어가도 이어짐)
     = 팀 값이 따라오기 전에 스쿼드가 바뀐 만큼 (여름 이적이면 시즌 초에 크게 나옴). 승격팀은 지난 기록이 없어서 0
  팀 값이 덜 쌓인 시즌 초일수록 B 가 더 잘 맞을 거라는 가정 → 시즌 경기 수 구간별로 기울기를 따로 찾음

  채점: 팀 값 기대 xG 대비 실제 xG (잔차)를 B 가 얼마나 설명하나
        기울기는 23/24 ~ 24/25 로 찾고 25/26 ~ 26/27 로 채점 (처음 보는 경기), 선수 값은 그 경기 전 기록만
        선발 명단은 경기 직전에 안다고 가정

사용: py squad_test.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import live as L

DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"
HL = 180            # 선수 기록 반감기 (일). 반년 전 기록은 절반만 반영
K_MIN = 900         # 공격: 출전 900분만큼 포지션 평균을 섞음
K_DEF = 8           # 수비: 출전·결장 각각 8경기만큼 0 쪽으로 당김
POOL = 14           # 주전 후보: 최근 10경기 출전 시간 상위 14명
EPS = 0.15          # xG 비율 계산할 때 0 나누기 방지


def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"
    if pos.startswith("FW"): return "FWD"
    return "MID"


m, T = L.load_team_matches()
seasons = sorted(T["season"].unique())
latest = int(max(seasons))
TEST = [latest - 1, latest]
TRAIN = [latest - 3, latest - 2]
print(f"경기 {len(m)}개 · 기울기 찾기 {TRAIN} · 채점 {TEST}\n")

saved = L.load_params()
if saved and saved.get("test_season") in (latest - 1, latest):
    p = saved
else:
    print("  (α, carry 찾는 중...)")
    p = L.fit(T, latest - 1, log=lambda *a, **k: None)
const = p["const"]
d0 = T[T["season"] < latest - 1]
SPQ = d0["xg_sp"].sum() / max(d0["n_sp"].sum(), 1)

# ---------- 1. 팀 기대 xG (경기 전 값) ----------
exp_xg, lvl, ngames = {}, {}, {}
def before(rows, R):
    for r in rows:
        e = R.expect(r.team, r.opp, r.home)
        exp_xg[(r.match_id, r.team)] = e["open"] * e["q"] + e["sp"] * SPQ
        v = R.team_values(r.team)
        lvl[(r.match_id, r.team)] = v["att_open"] * v["att_q"]
        ngames[(r.match_id, r.team)] = R.n.get(r.team, 0)
L.run(T, p["alpha"], p["carry"], const, int(T["season"].min()), before)
T["exp"] = [exp_xg[(a, b)] for a, b in zip(T["match_id"], T["team"])]
T["act"] = T["xg_open"] + T["xg_sp"]
T["res"] = np.log((T["act"] + EPS) / (T["exp"] + EPS))     # +면 팀 값보다 xG를 더 만든 경기

# ---------- 2. 선발 / 출전 기록 ----------
lineups = {}      # (match_id, team) -> [(pid, group, mins, chain, start)]
has_chain = None
for mid in m["match_id"]:
    f = CACHE / f"{mid}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    ro = d.get("rosters") if isinstance(d.get("rosters"), dict) else {}
    for side in ("h", "a"):
        x = ro.get(side) or {}
        ps = []
        for q in (x.values() if isinstance(x, dict) else x):
            if not isinstance(q, dict):
                continue
            if has_chain is None:
                has_chain = "xGChain" in q
            chain = float(q.get("xGChain") if q.get("xGChain") is not None else
                          (float(q.get("xG") or 0) + float(q.get("xA") or 0) + float(q.get("xGBuildup") or 0)))
            pos = q.get("position") or "Sub"
            ps.append((int(q["player_id"]), group(pos) if pos != "Sub" else None, int(q.get("time") or 0), chain, pos != "Sub"))
        lineups[(mid, side)] = ps
print(f"선수 기록: {'xGChain 있음' if has_chain else 'xGChain 없음 → xG + xA + xGBuildup 으로 대신'}")

# ---------- 3. 날짜순으로 선수 값 쌓으면서 경기마다 특징 계산 ----------
att = defaultdict(lambda: [0.0, 0.0, None])            # pid -> [chain 합, 분 합, 마지막 날짜]
dfn = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, None])  # pid -> [출전 시 실점잔차 합, 출전 수, 결장 시 합, 결장 수, 날짜]
grp_of = {}                                            # pid -> 주 포지션
recent = defaultdict(list)                              # team -> 최근 경기 [(pid, mins)] 목록
g_rate = {"DEF": [0.0, 0.0], "MID": [0.0, 0.0], "FWD": [0.0, 0.0]}

def decay(v, now, idx_date, n_keep):
    if v[idx_date] is not None:
        w = 0.5 ** ((now - v[idx_date]).days / HL)
        for i in range(n_keep):
            v[i] *= w
    v[idx_date] = now

def att90(pid, now):
    v = att[pid]; decay(v, now, 2, 2)
    g = grp_of.get(pid, "MID")
    gm = g_rate[g][0] / max(g_rate[g][1], 1) * 90 if g in g_rate else 0.3
    return (v[0] + gm * K_MIN / 90) / (v[1] + K_MIN) * 90

def defval(pid, now):
    v = dfn[pid]; decay(v, now, 4, 4)
    return v[2] / (v[3] + K_DEF) - v[0] / (v[1] + K_DEF)    # +면 빠질 때 더 내줌

def pool_of(team):
    mins = defaultdict(float)
    for lst in recent[team][-10:]:
        for pid, mn in lst:
            mins[pid] += mn
    return sorted(mins, key=lambda k: -mins[k])

rows_out = []
Tm = T.set_index(["match_id", "side"])
for dt, day in m.groupby(m["dt"].dt.normalize(), sort=True):
    feats = []
    for r in day.itertuples():
        sides = {"h": (r.home, r.away), "a": (r.away, r.home)}
        if (r.match_id, "h") not in lineups or (r.match_id, "a") not in lineups:
            continue
        f = {}
        for side, (team, opp) in sides.items():
            ps = lineups[(r.match_id, side)]
            starters = [q for q in ps if q[4] and q[1] != "GK"]
            pool = [pid for pid in pool_of(team) if grp_of.get(pid) != "GK"]
            typical = pool[:10] if len(pool) >= 10 else []
            if len(starters) < 10:
                f[side] = None
                continue
            s_now = sum(att90(q[0], dt) for q in starters)
            s_typ = sum(att90(pid, dt) for pid in typical) if typical else 0.0
            start_ids = {q[0] for q in starters}
            miss = sum(defval(pid, dt) for pid in typical if pid not in start_ids)
            f[side] = {"A": np.log(s_now / s_typ) if s_typ > 0 else 0.0, "D": miss, "S": s_now}
        if f.get("h") is None or f.get("a") is None:
            feats.append((r, None))
            continue
        for side, o in (("h", "a"), ("a", "h")):
            t = Tm.loc[(r.match_id, side)]
            key = (r.match_id, t["team"])
            rows_out.append({"match_id": r.match_id, "season": r.season, "team": t["team"], "res": t["res"],
                             "A": f[side]["A"], "B": np.log(f[side]["S"]) - np.log(lvl[key]), "n": ngames[key],
                             "act": t["act"], "exp": t["exp"]})
        feats.append((r, f))
    # 이 날 경기 반영 (다음 날부터 쓰임)
    for r, _ in feats:
        for side, team in (("h", r.home), ("a", r.away)):
            ps = lineups[(r.match_id, side)]
            opp_side = "a" if side == "h" else "h"
            res_against = Tm.loc[(r.match_id, opp_side)]["res"]
            pool = pool_of(team)[:POOL]
            start_ids = {q[0] for q in ps if q[4]}
            for pid, g, mins, chain, st in ps:
                if g:
                    grp_of[pid] = g
                if mins > 0:
                    v = att[pid]; decay(v, dt, 2, 2); v[0] += chain; v[1] += mins
                    gg = grp_of.get(pid)
                    if gg in g_rate:
                        g_rate[gg][0] += chain; g_rate[gg][1] += mins
            for pid in set(pool) | start_ids:
                v = dfn[pid]; decay(v, dt, 4, 4)
                if pid in start_ids:
                    v[0] += res_against; v[1] += 1
                else:
                    v[2] += res_against; v[3] += 1
            recent[team].append([(q[0], q[2]) for q in ps if q[2] > 0])
            recent[team] = recent[team][-10:]

D = pd.DataFrame(rows_out)
print(f"특징 계산된 팀-경기 {len(D)}개\n")

# ---------- 4. 기울기 찾기 (TRAIN) → 채점 (TEST) ----------
BINS = [(0, 3, "시즌 1~3경기"), (4, 8, "4~8경기"), (9, 15, "9~15경기"), (16, 99, "16경기~")]
D["bin"] = 0
for i, (lo, hi, _) in enumerate(BINS):
    D.loc[(D["n"] >= lo) & (D["n"] <= hi), "bin"] = i
D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["B"]).reset_index(drop=True)
D["ref"] = D.groupby("team")["B"].transform(lambda x: x.shift(1).ewm(halflife=10).mean())
D["dB"] = (D["B"] - D["ref"]).fillna(0.0)                          # 지난 기록 없는 팀(첫 경기·승격팀)은 0
D["B"] = D["B"] - D.loc[D["season"].isin(TRAIN), "B"].mean()      # 평균 0 으로 (해석 쉽게)
tr, te = D[D["season"].isin(TRAIN)].copy(), D[D["season"].isin(TEST)].copy()


def design(d, kind):
    cols = [np.ones(len(d))]
    if kind in ("dB", "dBbin", "dBbin+A"):
        if kind == "dB":
            cols.append(d["dB"])
        else:
            for i in range(1, len(BINS)):
                cols.append((d["bin"] == i).astype(float))
            for i in range(len(BINS)):
                cols.append(d["dB"] * (d["bin"] == i))
        if kind == "dBbin+A":
            cols.append(d["A"])
        return np.column_stack(cols)
    if kind in ("bin", "bin+A"):
        for i in range(1, len(BINS)):
            cols.append((d["bin"] == i).astype(float))
        for i in range(len(BINS)):
            cols.append(d["B"] * (d["bin"] == i))
    if kind == "one":
        cols.append(d["B"])
    if kind == "bin+A":
        cols.append(d["A"])
    if kind == "A":
        cols.append(d["A"])
    return np.column_stack(cols)


def fit_pred(kind):
    b = np.linalg.lstsq(design(tr, kind), tr["res"], rcond=None)[0]
    return b, design(te, kind) @ b


base_pred = np.full(len(te), tr["res"].mean())
print("===== 팀 값이 못 맞힌 xG를 '스쿼드 기준 공격력'으로 얼마나 설명하나 =====")
print(f"  기울기: {TRAIN} {len(tr)}개로 찾음 / 채점: {TEST} {len(te)}개 (처음 보는 경기)")
print(f"  {'':<26}{'xG 오차 줄어든 %':>16}{'확실도':>8}{'시즌 1~8경기만':>16}{'확실도':>8}")
early = te["n"] <= 8
results = {}
for kind, name in (("dB", "dB 하나"), ("dBbin", "dB 시즌 경기 수 구간별"), ("dBbin+A", "dB 구간별 + 결장 A"),
                   ("A", "참고: 결장 효과 A 만"), ("one", "참고: 예전 B 하나")):
    b, pr = fit_pred(kind)
    results[kind] = b
    e0 = (te["res"] - base_pred) ** 2; e1 = (te["res"] - pr) ** 2; d = e0 - e1
    z = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
    de = d[early]; ze = de.mean() / (de.std() / np.sqrt(len(de)) + 1e-12)
    print(f"  {name:<24}{d.mean() / e0.mean():>15.2%}{z:>+8.1f}{de.mean() / e0[early].mean():>15.2%}{ze:>+8.1f}")
print("  확실도 +2 이상이면 맞다")

b = results["dBbin"]
print("\n  [dB 구간별 기울기] 1이면 '스쿼드가 평소보다 10% 세지면 실제 xG도 10% 많음', 0이면 무시해도 됨")
for i, (lo, hi, lab) in enumerate(BINS):
    nn = (tr["bin"] == i).sum(), (te["bin"] == i).sum()
    print(f"    {lab:<12} 기울기 {b[len(BINS) + i]:+.3f}   (찾은 경기 {nn[0]}, 채점 경기 {nn[1]})")

print("\n  [크기] 채점 경기, 시즌 1~8경기 구간에서 dB(스쿼드 변화) 별 실제 xG ÷ 팀 기대 xG")
g0 = te[early]
for lo, hi, lab in ((-9, -0.15, "스쿼드 15%+ 약해짐"), (-0.15, -0.05, "5~15% 약해짐"), (-0.05, 0.05, "비슷"),
                    (0.05, 0.15, "5~15% 세짐"), (0.15, 9, "15%+ 세짐")):
    g = g0[(g0["dB"] >= lo) & (g0["dB"] < hi)]
    if len(g) >= 15:
        r = np.exp(g["res"])
        print(f"    {lab:<20} {len(g):>4}개 → {r.mean():.3f} (±{2 * r.std() / np.sqrt(len(g)):.3f})")
print("\n  [예시] 26/27 시즌 1~5경기, 스쿼드가 지난 시즌보다 제일 세진 팀 / 약해진 팀 (dB)")
ex = te[(te["season"] == max(TEST)) & (te["n"] <= 5)].groupby("team")["dB"].mean().sort_values()
for t, v in list(ex.tail(5).items())[::-1] + list(ex.head(5).items()):
    print(f"    {t:<26} {np.exp(v):.2f}배")
D.to_csv(DATA / "squad_test.csv", index=False, encoding="utf-8-sig")
print("\n저장: data/squad_test.csv")
