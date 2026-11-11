"""
선수 결장 효과 (11/11) — 시뮬 없이 xG로 채점. 몇 분이면 끝남
  ③ 채점은 골 말고 '그 경기 실제 xG' (운이 훨씬 덜 섞임)
  ① 공격: 선발 11명이 평소 만들던 xGChain(관여한 공격의 xG) 합 ÷ 평소 베스트 라인업 합 → 팀 xG가 얼마나 바뀌나
  ② 수비: 선수마다 '그 선수가 빠졌을 때 팀이 xG를 얼마나 더 내줬나' (출전/결장 비교, 평균 쪽으로 세게 당김)
        → 이번 경기에 빠진 주전들 값 합 → 상대 xG가 얼마나 바뀌나

  방법
    1. 살아있는 팀 값으로 경기마다 '팀 기대 xG'를 냄 (그 경기 전 정보로만)
    2. 실제 xG ÷ 기대 xG = 팀 값이 못 맞힌 부분 (잔차)
    3. 그 잔차를 선발 정보(①②)로 얼마나 설명하나. 기울기는 23/24 ~ 24/25 로 찾고, 25/26 ~ 26/27 로 채점 (처음 보는 경기)
    선수 값도 전부 그 경기 '전' 기록으로만 계산 (미래 안 섞임). 선발 명단은 경기 직전에 안다고 가정

사용: py absence_test.py
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
exp_xg = {}
def before(rows, R):
    for r in rows:
        e = R.expect(r.team, r.opp, r.home)
        exp_xg[(r.match_id, r.team)] = e["open"] * e["q"] + e["sp"] * SPQ
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
            typical = pool[:10]
            if len(typical) < 10 or len(starters) < 10:
                f[side] = None
                continue
            s_now = sum(att90(q[0], dt) for q in starters)
            s_typ = sum(att90(pid, dt) for pid in typical)
            start_ids = {q[0] for q in starters}
            miss = sum(defval(pid, dt) for pid in typical if pid not in start_ids)
            f[side] = {"A": np.log(s_now / s_typ), "D": miss, "n_miss": sum(pid not in start_ids for pid in typical)}
        if f.get("h") is None or f.get("a") is None:
            feats.append((r, None))
            continue
        for side, o in (("h", "a"), ("a", "h")):
            t = Tm.loc[(r.match_id, side)]
            rows_out.append({"match_id": r.match_id, "season": r.season, "team": t["team"], "res": t["res"],
                             "A": f[side]["A"], "Dopp": f[o]["D"], "n_miss": f[side]["n_miss"], "act": t["act"], "exp": t["exp"]})
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
tr, te = D[D["season"].isin(TRAIN)], D[D["season"].isin(TEST)]
models = {"공격만 (①)": ["A"], "수비만 (②)": ["Dopp"], "공격 + 수비": ["A", "Dopp"]}
print(f"===== 팀 값이 못 맞힌 xG를 선발 정보로 얼마나 설명하나 =====")
print(f"  기울기: {TRAIN} {len(tr)}개로 찾음 / 채점: {TEST} {len(te)}개 (처음 보는 경기)")
print(f"  {'':<14}{'기울기':>24}{'xG 오차 줄어든 %':>16}{'확실도':>8}")
base_err = (te["res"] - tr["res"].mean()) ** 2
for name, cols in models.items():
    X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols])
    b = np.linalg.lstsq(X, tr["res"], rcond=None)[0]
    Xt = np.column_stack([np.ones(len(te))] + [te[c] for c in cols])
    err = (te["res"] - Xt @ b) ** 2
    d = base_err - err
    z = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
    bs = ", ".join(f"{c} {v:+.3f}" for c, v in zip(cols, b[1:]))
    print(f"  {name:<14}{bs:>24}{d.mean() / base_err.mean():>15.2%}{z:>+8.1f}")
print("  확실도 +2 이상이면 선발을 알면 팀 xG를 진짜 더 잘 맞힘. 기울기 A: 1이면 선발 xGChain 10% 많으면 팀 xG 10% 많음")

# 크기 감 잡기
print(f"\n  [크기] 채점 경기에서 A(공격 라인업 비율): 평소 그대로 {np.mean(np.abs(te['A']) < 0.02):.0%}, "
      f"10% 넘게 약한 라인업 {np.mean(te['A'] < np.log(0.9)):.0%}, 10% 넘게 강한 라인업 {np.mean(te['A'] > np.log(1.1)):.0%}")
for lo, hi, lab in ((-9, np.log(0.85), "15%+ 약함"), (np.log(0.85), np.log(0.95), "5~15% 약함"), (np.log(0.95), np.log(1.05), "평소"),
                    (np.log(1.05), 9, "5%+ 강함")):
    g = te[(te["A"] >= lo) & (te["A"] < hi)]
    if len(g) >= 20:
        print(f"    {lab:<10} {len(g):>5}개 → 실제 xG ÷ 팀 기대 xG 평균 {np.exp(g['res']).mean():.3f} (±{2 * np.exp(g['res']).std() / np.sqrt(len(g)):.3f})")
for lo, hi, lab in ((-9, 0.02, "상대 주전 결장 효과 거의 없음"), (0.02, 0.08, "조금"), (0.08, 9, "큼")):
    g = te[(te["Dopp"] >= lo) & (te["Dopp"] < hi)]
    if len(g) >= 20:
        print(f"    {lab:<20} {len(g):>5}개 → 실제 xG ÷ 기대 xG {np.exp(g['res']).mean():.3f} (±{2 * np.exp(g['res']).std() / np.sqrt(len(g)):.3f})")
D.to_csv(DATA / "absence_test.csv", index=False, encoding="utf-8-sig")
print("\n저장: data/absence_test.csv")
