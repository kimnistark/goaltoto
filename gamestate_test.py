"""
스코어 상황 넣기 vs 안 넣기 (11/12)
  gamestate.py 가 최근 2시즌 '전' 경기로 잰 상황 배수를 켜고 끄고, 25/26 ~ 26/27 경기로 채점
  매주 그 주 전 정보로만, 팀만 시뮬, 경기당 3000번
  보는 것: 1-1·무승부·팀 0골 이 실제에 가까워졌나 (11/11 진단에서 덜 본 칸), 1위 적중, 스코어 손실, 골 오차, 2골차+ 확률

사용: py gamestate.py → py gamestate_test.py   (20~30분)
"""
import copy
import sys

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 3000
MAXG = 6
if E.GS is None:
    sys.exit("data/gamestate.json 이 없어. 먼저 py gamestate.py")
m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
if E.GS["meta"]["seasons"][1] >= min(tests):
    print(f"[주의] gamestate.json 이 {E.GS['meta']['seasons']} 시즌으로 잰 거라 채점 시즌이랑 겹침. py gamestate.py (all 없이) 로 다시 재는 게 맞음\n")
VAR = {"off": "상황 없음 (지금)", "on": "상황 넣음"}


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


rows = []
for test in tests:
    p = season_params(test)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    S = {"week": None, "Rw": None}

    def before(rs, R):
        r0 = rs[0]
        if r0.season != test:
            return
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
        for r in rs:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rs if r.home); a = next(r for r in rs if not r.home)
        for t in (h.team, a.team):
            v = S["Rw"].team_values(t)
            E.C["teams"][t] = v
            E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
        gh, ga = int(h.gf), int(a.gf)
        row = {"season": test, "gh": gh, "ga": ga}
        for k in VAR:
            E.GS_ON = k == "on"
            sh, sa, _ = E.simulate(h.team, a.team, N, use_players=False)
            E.GS_ON = False
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            mat /= N
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            gd = sh - sa
            row.update({f"{k}_hit": (i, j) == (min(gh, MAXG), min(ga, MAXG)), f"{k}_ll": -np.log(max(ps, 1e-4)),
                        f"{k}_dev": L.poisson_dev(gh, sh.mean()) + L.poisson_dev(ga, sa.mean()),
                        f"{k}_p11": mat[1, 1], f"{k}_draw": np.trace(mat), f"{k}_h0": mat[0].sum(), f"{k}_a0": mat[:, 0].sum(),
                        f"{k}_p10": mat[1, 0] + mat[0, 1], f"{k}_p21": mat[2, 1] + mat[1, 2],
                        f"{k}_h2": (gd >= 2).mean(), f"{k}_a2": (gd <= -2).mean(),
                        f"{k}_mh": sh.mean(), f"{k}_ma": sa.mean()})
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"    {len(rows)}경기")

    print(f"===== [{test}/{(test + 1) % 100}] 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)

D = pd.DataFrame(rows)
n = len(D)
print(f"\n===== 분포 칸 맞추기 ({n}경기): 시뮬 확률 평균 vs 실제 =====")
print(f"  {'':<14}{'실제':>8}{VAR['off']:>16}{VAR['on']:>12}")
act = {"p11": ((D.gh == 1) & (D.ga == 1)).mean(), "draw": (D.gh == D.ga).mean(),
       "p10": (((D.gh == 1) & (D.ga == 0)) | ((D.gh == 0) & (D.ga == 1))).mean(),
       "p21": (((D.gh == 2) & (D.ga == 1)) | ((D.gh == 1) & (D.ga == 2))).mean(),
       "h0": (D.gh == 0).mean(), "a0": (D.ga == 0).mean(), "h2": ((D.gh - D.ga) >= 2).mean(), "a2": ((D.gh - D.ga) <= -2).mean()}
for k, lab in (("p11", "1-1"), ("draw", "무승부"), ("p10", "1-0 + 0-1"), ("p21", "2-1 + 1-2"), ("h0", "홈 0골"), ("a0", "원정 0골"),
               ("h2", "홈 2골차+"), ("a2", "원정 2골차+")):
    print(f"  {lab:<14}{act[k]:>8.1%}{D[f'off_{k}'].mean():>16.1%}{D[f'on_{k}'].mean():>12.1%}")
print(f"  {'평균 골 (홈-원정)':<12}{D.gh.mean():>6.2f}-{D.ga.mean():.2f}{D.off_mh.mean():>11.2f}-{D.off_ma.mean():.2f}{D.on_mh.mean():>7.2f}-{D.on_ma.mean():.2f}")

print(f"\n===== 채점 =====")
print(f"  {'':<18}{VAR['off']:>16}{VAR['on']:>12}{'확실도':>8}")
for col, lab, hb, fmt in (("hit", "1위 스코어 적중률", True, "%"), ("ll", "스코어 손실", False, "f"), ("dev", "골 오차", False, "f")):
    a, b = D[f"off_{col}"].astype(float), D[f"on_{col}"].astype(float)
    d = (b - a) if hb else (a - b)
    z = d.mean() / (d.std() / np.sqrt(n) + 1e-12)
    f = (lambda x: f"{x:.1%}") if fmt == "%" else (lambda x: f"{x:.4f}")
    print(f"  {lab:<16}{f(a.mean()):>16}{f(b.mean()):>12}{z:>+8.1f}")
# 핸디 2골차+ 브라이어 (낮을수록 좋음)
for side, lab in (("h2", "홈 2골차+ 확률 오차"), ("a2", "원정 2골차+ 확률 오차")):
    y = ((D.gh - D.ga) >= 2) if side == "h2" else ((D.gh - D.ga) <= -2)
    e0, e1 = (D[f"off_{side}"] - y) ** 2, (D[f"on_{side}"] - y) ** 2
    d = e0 - e1
    print(f"  {lab:<16}{e0.mean():>16.4f}{e1.mean():>12.4f}{d.mean() / (d.std() / np.sqrt(n) + 1e-12):>+8.1f}")
print("  확실도: + 면 상황 넣은 게 나음, +2 이상이면 맞다. 오차·손실은 낮을수록 좋음")
