"""
선수 켬 vs 팀만 (같은 경기, 같은 조건)
  매주 그 주 시작 전 정보로만 예측
    팀 값: 살아있는 값 (그 주 시작 시점)
    선수:  data/snap/그 주 시작일/ 에 찍어둔 선수 파일 (make_snaps.py 가 만듦)
  두 방식 다 같은 팀 값·같은 선수 파일(전개·세트피스 기준값 포함)을 쓰고, 선수 추첨만 켜고 끄는 차이

  비교 숫자: 1위 스코어 적중률, 평균 골 반올림 적중률, 언더오버 2.5 적중률, 골 오차, 실제 스코어 확률
  선발 앎: 그 경기 실제 선발 11명을 그대로 넣고 돌림 (경기 1시간 전 선발 발표를 보고 예측하는 상황)
           선수 스탯은 여전히 그 주 시작 전 기록. 선수 파일에 없는 선발(새로 온 선수 등)은 포지션 평균 선수로 채움
  확실도: 선수 켬 − 팀만, 선발 앎 − 팀만 (같은 경기끼리 비교). +2 이상이면 진짜 나음

사용: py make_snaps.py 먼저 → py players_test.py       (팀만 vs 선발 앎)
                             py players_test.py all   (선수 켬(추첨)까지 셋 다)
"""
import copy
import json
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
VAR = {"team": "팀만", "players": "선수 켬", "known": "선발 앎"}
if "all" not in sys.argv:          # 선수 켬(추첨)은 11/10에 이미 봤으니 기본은 건너뜀 → 시간 절약. 셋 다 보려면 py players_test.py all
    VAR.pop("players")
CACHE = Path(__file__).parent / "data" / "cache"


def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"
    if pos.startswith("FW"): return "FWD"
    return "MID"


def actual_lineups(mid, home, away):
    """캐시에서 실제 선발: {팀: (pid 집합, (수비, 미드, 공격))}. 선수 파일에 있는 선수는 그 파일 포지션으로 셈"""
    f = CACHE / f"{mid}.json"
    if not f.exists():
        return {}
    ro = json.loads(f.read_text(encoding="utf-8")).get("rosters")
    if not isinstance(ro, dict):
        return {}
    out = {}
    for side, team in (("h", home), ("a", away)):
        x = ro.get(side) or {}
        known_g = {p["pid"]: p["group"] for p in E.PL.get(team, [])}
        pids, cnt = set(), {"DEF": 0, "MID": 0, "FWD": 0}
        for p in (x.values() if isinstance(x, dict) else x):
            if not isinstance(p, dict) or p.get("position") in (None, "Sub"):
                continue
            pid = int(p["player_id"]); pids.add(pid)
            g = known_g.get(pid) or group(p["position"])
            if g in cnt:
                cnt[g] += 1
        if sum(cnt.values()) == 10:
            out[team] = (pids, (cnt["DEF"], cnt["MID"], cnt["FWD"]))
    return out
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
        kn = actual_lineups(h.match_id, h.team, a.team)
        row["has_lineup"] = len(kn) == 2
        for k in VAR:
            sh, sa, _ = E.simulate(h.team, a.team, N, use_players=(k != "team"), known=kn if k == "known" else None)
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


def z(D, col, higher_better=True, who="players"):
    d = D[f"{who}_{col}"].astype(float) - D[f"team_{col}"].astype(float)
    if not higher_better:
        d = -d
    return d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)


def report(D, label):
    n = len(D)
    print(f"\n  ----- {label}: {n}경기 -----")
    print(f"  {'':<22}{'팀만':>9}{'선수 켬':>10}{'확실도':>8}{'선발 앎':>10}{'확실도':>8}")
    rows = [("1위 스코어 적중률", "hit", True, "%"), ("반올림 스코어 적중률", "rnd", True, "%"),
            ("언더오버 2.5 적중률", "ou", True, "%"), ("실제 스코어 확률 (평균)", "p", True, "%"),
            ("골 오차 (낮을수록 좋음)", "dev", False, "f")]
    for name, col, hb, fmt in rows:
        f = (lambda x: f"{x:.1%}") if fmt == "%" else (lambda x: f"{x:.4f}")
        a, c = D[f"team_{col}"].mean(), D[f"known_{col}"].mean()
        mid = f"{f(D[f'players_{col}'].mean()):>10}{z(D, col, hb):>+8.1f}" if "players" in VAR else f"{'-':>10}{'':>8}"
        print(f"  {name:<20}{f(a):>10}{mid}{f(c):>10}{z(D, col, hb, 'known'):>+8.1f}")
    for who in [w for w in ("players", "known") if w in VAR]:
        r = D[f"{who}_hit"].mean(); se = np.sqrt(r * (1 - r) / n)
        print(f"  → {VAR[who]} 1위 스코어 적중률 {r:.1%} (오차범위 {max(0, r - 2 * se):.1%} ~ {r + 2 * se:.1%})")
    print(f"  팀만과 1위 스코어가 같은 경기 (선발 앎): {(D['known_top1'] == D['team_top1']).mean():.0%}")


all_D = [evaluate(t) for t in tests]
D = pd.concat(all_D, ignore_index=True)
for s, g in D.groupby("season"):
    report(g, f"{s}/{(s + 1) % 100}")
if len(all_D) > 1:
    report(D, "전체")
print(f"\n  선수 데이터가 양 팀 다 있는 경기 {D['has_players'].mean():.0%} (없는 팀은 선수 켬도 팀 값으로만 돌아감)")
print(f"  실제 선발을 양 팀 다 읽은 경기 {D['has_lineup'].mean():.0%} (못 읽은 팀은 선발 앎도 추첨으로 돌아감)")
print("  확실도 = 선수 켬 − 팀만 / 선발 앎 − 팀만, 같은 경기끼리. +2 이상이면 선수 켠 게 진짜 나음, −2 이하면 진짜 못함, 그 사이면 차이 없음")
D.to_csv(Path(__file__).parent / "data" / "players_test.csv", index=False, encoding="utf-8-sig")
print("  저장: data/players_test.csv")
