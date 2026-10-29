"""
1위 스코어 적중률 확인
  매주 그 주 시작 전 정보로만 예측 (살아있는 값, 팀 단위 시뮬) → 시뮬 1위 스코어 = 실제 스코어인 비율
  비교: 리그 평균만으로 찍은 1위 스코어 (팀 정보 없이 홈·원정 평균 골, 그 시즌 이전 경기로 계산)

사용: py top1_check.py          (25/26 + 26/27 지금까지)
"""
import copy
import sys
from math import exp, factorial

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 2000
MAXG = 6
m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
pois = lambda k, lam: exp(-lam) * lam ** k / factorial(k)


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


def evaluate(test):
    p = season_params(test)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    prev = m[m["season"] < test]
    bh, ba = prev["FTHG"].mean(), prev["FTAG"].mean()
    base = np.outer([pois(i, bh) for i in range(MAXG + 1)], [pois(j, ba) for j in range(MAXG + 1)])
    b1 = np.unravel_index(base.argmax(), base.shape)
    S = {"week": None, "Rw": None}
    out = []

    def before(rows, R):
        r0 = rows[0]
        if r0.season != test:
            return
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
        for r in rows:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rows if r.home); a = next(r for r in rows if not r.home)
        for t in (h.team, a.team):
            v = S["Rw"].team_values(t)
            E.C["teams"][t] = v
            E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
        sh, sa, _ = E.simulate(h.team, a.team, N, use_players=False)
        mat = np.zeros((MAXG + 1, MAXG + 1))
        np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
        i, j = np.unravel_index(mat.argmax(), mat.shape)
        gh, ga = int(h.gf), int(a.gf)
        out.append({"league": h.league, "season": test, "result": f"{gh}-{ga}", "top1": f"{i}-{j}",
                    "top1_p": mat[i, j] / N, "hit": (i, j) == (min(gh, MAXG), min(ga, MAXG)),
                    "base_top1": f"{b1[0]}-{b1[1]}", "base_hit": (b1[0], b1[1]) == (gh, ga)})
        if len(out) % 200 == 0:
            print(f"    {len(out)}경기")

    print(f"===== [{test}/{(test + 1) % 100}] 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    return pd.DataFrame(out)


def line(g, label):
    n = len(g); r = g["hit"].mean(); se = np.sqrt(r * (1 - r) / n)
    b = g["base_hit"].mean()
    d = g["hit"].astype(float) - g["base_hit"].astype(float)
    z = d.mean() / (d.std() / np.sqrt(n) + 1e-12)
    print(f"  {label:<16}{n:>6}{r:>9.1%}   ({max(0, r - 2 * se):.1%} ~ {r + 2 * se:.1%}){b:>12.1%}{z:>+9.1f}")


D = pd.concat([evaluate(t) for t in tests], ignore_index=True)
print(f"\n===== 1위 스코어 적중률 =====")
print(f"  {'':<16}{'경기':>6}{'적중률':>9}   {'(오차범위)':<17}{'리그평균만':>8}{'확실도':>8}")
for s, g in D.groupby("season"):
    line(g, f"{s}/{(s + 1) % 100}")
for lg, g in D.groupby("league"):
    line(g, lg)
line(D, "전체")
print("  오차범위 = 운으로 흔들릴 수 있는 폭 (95%) / 리그평균만 = 팀 정보 없이 매 경기 같은 스코어 찍기"
      f" ({', '.join(sorted(D['base_top1'].unique()))}) / 확실도 +2 이상이면 시뮬이 진짜 나음")
print("\n  [시뮬 1위 스코어별]")
for k, g in D.groupby("top1"):
    if len(g) >= 10:
        print(f"    {k}로 찍은 경기 {len(g):>4}개 → 적중 {g['hit'].mean():.1%} (시뮬이 준 확률 평균 {g['top1_p'].mean():.1%})")
print("  ※ 팀 단위 시뮬 (선수 추첨 끔). 실전 예측은 선수까지 켜고 돌림")
