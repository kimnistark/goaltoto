"""
선수 켬 vs 팀만 (같은 경기, 같은 조건)
  매주 그 주 시작 전 정보로만 예측
    팀 값: 살아있는 값 (그 주 시작 시점)
    선수:  data/snap/그 주 시작일/ 에 찍어둔 선수 파일 (make_snaps.py 가 만듦)
  두 방식 다 같은 팀 값·같은 선수 파일(전개·세트피스 기준값 포함)을 쓰고, 선수 추첨만 켜고 끄는 차이

  비교 숫자: 1위 스코어 적중률, 평균 골 반올림 적중률, 언더오버 2.5 적중률, 골 오차, 실제 스코어 확률
  확실도: 선수 켬 − 팀만 (같은 경기끼리 비교). +2 이상이면 선수 켠 게 진짜 나음

사용: py make_snaps.py 먼저 → py players_test.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 2000
MAXG = 6
SNAP = Path(__file__).parent / "data" / "snap"
m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
VAR = {"team": "팀만", "players": "선수 켬"}
rnd = lambda x: int(np.floor(x + 0.5))


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
    TT = T[T["season"] <= test]
    S = {"week": None, "Rw": None, "missing": 0}
    out = []

    def set_week(wk, R):
        d = SNAP / f"{wk:%Y-%m-%d}"
        if not (d / "players_live.json").exists():
            S["missing"] += 1
            return False
        E.load_snapshot(d)
        E.C["base_open"] = const["base_open"]
        sp_tot = sum(E.C["sp_rate"])
        E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
        E.P["home_mult"] = const["home"]
        return True

    def before(rows, R):
        r0 = rows[0]
        if r0.season != test:
            return
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
            S["ok"] = set_week(wk, R)
        if not S["ok"]:
            return
        for r in rows:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rows if r.home); a = next(r for r in rows if not r.home)
        for t in (h.team, a.team):
            v = S["Rw"].team_values(t)
            E.C["teams"][t] = v
            E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
        gh, ga = int(h.gf), int(a.gf)
        row = {"league": h.league, "season": test, "home": h.team, "away": a.team, "result": f"{gh}-{ga}",
               "has_players": bool(E.PL.get(h.team)) and bool(E.PL.get(a.team))}
        for k in VAR:
            sh, sa, _ = E.simulate(h.team, a.team, N, use_players=(k == "players"))
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            mat /= N
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            mh, ma = sh.mean(), sa.mean()
            row.update({f"{k}_top1": f"{i}-{j}", f"{k}_hit": (i, j) == (min(gh, MAXG), min(ga, MAXG)),
                        f"{k}_rnd": (rnd(mh), rnd(ma)) == (gh, ga),
                        f"{k}_ou": ((((sh + sa) > 2.5).mean() > 0.5) == (gh + ga > 2.5)),
                        f"{k}_dev": L.poisson_dev(gh, mh) + L.poisson_dev(ga, ma),
                        f"{k}_p": ps, f"{k}_ll": -np.log(max(ps, 1e-3)), f"{k}_mh": mh, f"{k}_ma": ma})
        out.append(row)
        if len(out) % 100 == 0:
            print(f"    {len(out)}경기")

    print(f"===== [{test}/{(test + 1) % 100}] 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    if S["missing"]:
        print(f"  [주의] 선수 파일 없는 주 {S['missing']}개는 건너뜀 (py make_snaps.py 다시 돌리면 채워짐)")
    return pd.DataFrame(out)


def z(D, col, higher_better=True):
    d = D[f"players_{col}"].astype(float) - D[f"team_{col}"].astype(float)
    if not higher_better:
        d = -d
    return d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)


def report(D, label):
    n = len(D)
    print(f"\n  ----- {label}: {n}경기 -----")
    print(f"  {'':<22}{'팀만':>9}{'선수 켬':>10}{'확실도':>9}")
    rows = [("1위 스코어 적중률", "hit", True, "%"), ("반올림 스코어 적중률", "rnd", True, "%"),
            ("언더오버 2.5 적중률", "ou", True, "%"), ("실제 스코어 확률 (평균)", "p", True, "%"),
            ("골 오차 (낮을수록 좋음)", "dev", False, "f")]
    for name, col, hb, fmt in rows:
        a, b = D[f"team_{col}"].mean(), D[f"players_{col}"].mean()
        f = (lambda x: f"{x:.1%}") if fmt == "%" else (lambda x: f"{x:.4f}")
        print(f"  {name:<20}{f(a):>10}{f(b):>10}{z(D, col, hb):>+9.1f}")
    r = D["players_hit"].mean(); se = np.sqrt(r * (1 - r) / n)
    print(f"  → 선수 켬 1위 스코어 적중률 {r:.1%} (오차범위 {max(0, r - 2 * se):.1%} ~ {r + 2 * se:.1%})")
    same = (D["players_top1"] == D["team_top1"]).mean()
    print(f"  두 방식 1위 스코어가 같은 경기 {same:.0%}")


all_D = [evaluate(t) for t in tests]
D = pd.concat(all_D, ignore_index=True)
for s, g in D.groupby("season"):
    report(g, f"{s}/{(s + 1) % 100}")
if len(all_D) > 1:
    report(D, "전체")
print(f"\n  선수 데이터가 양 팀 다 있는 경기 {D['has_players'].mean():.0%} (없는 팀은 선수 켬도 팀 값으로만 돌아감)")
print("  확실도 = 선수 켬 − 팀만, 같은 경기끼리. +2 이상이면 선수 켠 게 진짜 나음, −2 이하면 진짜 못함, 그 사이면 차이 없음")
D.to_csv(Path(__file__).parent / "data" / "players_test.csv", index=False, encoding="utf-8-sig")
print("  저장: data/players_test.csv")
