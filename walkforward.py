"""
지난 경기들을 '그 경기 이전 정보만으로' 다시 예측해서 채점 (1단계 검증 + 오답노트 1회차)
  [고정]  팀 값을 시즌 시작 때 값으로 고정 (지금 시뮬과 같은 방식. 승격팀은 기본값)
  [살아있는] 팀 값을 경기마다 갱신 (live.py)
두 방식을 같은 경기로 비교. 시뮬은 팀 단위 (선수 단위는 다음 단계)

사용: py walkforward.py            (25/26 전체 + 26/27 지금까지)
      py walkforward.py 2026       (26/27만)
      py walkforward.py --refit    (α, carry 를 다시 찾음. 처음엔 자동으로 찾음)
결과: data/live_params.json (α, carry), data/predictions_시즌.csv, data/live_teams.json (지금 시점 팀 값)
"""
import copy
import json
import sys

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 2000
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REFIT = "--refit" in sys.argv

m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in args] or [latest - 1, latest]
print(f"경기 {len(m)}개 (시즌 {int(T['season'].min())}~{latest}, {'+'.join(L.LEAGUES)}). 채점할 시즌: {tests}\n")

def season_params(test):
    saved = L.load_params()
    if saved and not REFIT and saved.get("test_season") == test:
        return saved
    print(f"===== [{test}/{(test + 1) % 100} 채점용] α, carry 찾기 ({test - 1}/{test % 100} 이전 시즌으로) =====")
    p = L.fit(T, test)
    p["test_season"] = test
    if test == latest:
        L.save_params(p)
    print()
    return p

def sim(R, home, away):
    for t in (home, away):
        v = R.team_values(t)
        E.C["teams"][t] = v
        E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
    gh, ga, _ = E.simulate(home, away, N, use_players=False)
    gd = gh - ga
    return {"mh": gh.mean(), "ma": ga.mean(), "pH": (gd > 0).mean(), "pD": (gd == 0).mean(), "pA": (gd < 0).mean(),
            "pH2": (gd >= 2).mean(), "pA2": (gd <= -2).mean()}

def evaluate(test):
    p = season_params(test)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    first = int(TT["season"].min())
    fixed = {"R": None}
    out = []

    def before(rows, R):
        r0 = rows[0]
        if r0.season != test:
            return
        if fixed["R"] is None:
            fixed["R"] = copy.deepcopy(R)                   # 시즌 첫 경기 직전 값으로 고정
        F = fixed["R"]
        for r in rows:
            F.touch(r.team, r.season)
        h = next(r for r in rows if r.home); a = next(r for r in rows if not r.home)
        lv, fx = sim(R, h.team, a.team), sim(F, h.team, a.team)
        eh, ea = R.expect(h.team, a.team, True), R.expect(a.team, h.team, False)
        out.append({"match_id": h.match_id, "dt": h.dt, "league": h.league, "home": h.team, "away": a.team,
                    "gh": h.gf, "ga": a.gf, "n_home_games": R.n.get(h.team, 0), "n_away_games": R.n.get(a.team, 0),
                    **{f"live_{k}": v for k, v in lv.items()}, **{f"fix_{k}": v for k, v in fx.items()},
                    "h_open": h.n_open, "h_open_e": eh["open"], "h_q": h.xg_open / h.n_open if h.n_open else np.nan, "h_q_e": eh["q"],
                    "h_xg": h.xg_open, "h_gopen": h.g_open,
                    "a_open": a.n_open, "a_open_e": ea["open"], "a_q": a.xg_open / a.n_open if a.n_open else np.nan, "a_q_e": ea["q"],
                    "a_xg": a.xg_open, "a_gopen": a.g_open})
        if len(out) % 100 == 0:
            print(f"    {len(out)}경기 예측")

    print(f"===== [{test}/{(test + 1) % 100}] 경기마다 그 이전 정보만으로 예측 중 (경기당 {N}번 시뮬) =====")
    R = L.run(TT, p["alpha"], p["carry"], const, first, before)
    D = pd.DataFrame(out)
    D.to_csv(L.DATA / f"predictions_{test}.csv", index=False, encoding="utf-8-sig")
    report(D, test)
    return R, D

def scores(D, pre):
    res = np.sign(D["gh"] - D["ga"])
    p = np.where(res > 0, D[f"{pre}_pH"], np.where(res == 0, D[f"{pre}_pD"], D[f"{pre}_pA"]))
    ll = -np.log(np.clip(p, 1e-3, 1))
    pick = np.select([D[f"{pre}_pH"] >= np.maximum(D[f"{pre}_pD"], D[f"{pre}_pA"]), D[f"{pre}_pA"] >= D[f"{pre}_pD"]], [1, -1], 0)
    hit = (pick == res).astype(float)
    dev = np.array([L.poisson_dev(y, mu) for y, mu in zip(D["gh"], D[f"{pre}_mh"])]) + \
          np.array([L.poisson_dev(y, mu) for y, mu in zip(D["ga"], D[f"{pre}_ma"])])
    h2 = ((D["gh"] - D["ga"]) >= 2).astype(float)
    br = (D[f"{pre}_pH2"] - h2) ** 2
    return ll, hit, dev, br

def report(D, test):
    S = {k: scores(D, k) for k in ("fix", "live")}
    names = ["승무패 로그손실 (낮을수록 좋음)", "승무패 적중률 (높을수록 좋음)", "골 오차 (낮을수록 좋음)", "홈 2골차+ 승 Brier (낮을수록 좋음)"]
    print(f"\n  ----- {test}/{(test + 1) % 100} 결과: {len(D)}경기 -----")
    print(f"  {'':<36}{'고정':>9}{'살아있는':>10}{'차이':>9}{'확실도':>8}")
    for i, nm in enumerate(names):
        a, b = S["fix"][i], S["live"][i]
        d = (b - a) if i == 1 else (a - b)          # +면 살아있는 쪽이 나음
        z = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
        fmt = "{:.1%}" if i == 1 else "{:.4f}"
        print(f"  {nm:<34}{fmt.format(a.mean()):>10}{fmt.format(b.mean()):>10}{('+' if d.mean() >= 0 else '') + fmt.format(d.mean()):>10}{z:>+8.1f}")
    print("  차이·확실도: +면 살아있는 쪽이 나음. 확실도 +2 이상이면 우연이 아닐 가능성이 큼")
    # 지난번 문제: 전력 차 큰 경기에서 강팀을 덜 쳐주는지
    for pre, lab in (("fix", "고정"), ("live", "살아있는")):
        g = D[f"{pre}_mh"] - D[f"{pre}_ma"]
        top = np.abs(g) >= np.quantile(np.abs(g), 0.8)
        s_ = np.sign(g[top])
        err = ((D["gh"] - D["ga"])[top] - g[top]) * s_
        print(f"  [{lab}] 전력 차 상위 20% 경기({top.sum()}개): 강팀 기준 실제 골차 − 예측 골차 = {err.mean():+.2f}  (0에 가까울수록 좋음, +면 강팀을 덜 쳐준 것)")

def notebook(D, test):
    print(f"\n===== [{test}/{(test + 1) % 100}] 오답노트 =====")
    D = D.copy()
    D["week"] = pd.to_datetime(D["dt"]).dt.to_period("W-MON").dt.start_time
    S = scores(D, "live"); F = scores(D, "fix")
    D["ll"], D["hit"], D["ll_fix"] = S[0], S[1], F[0]
    print(f"  {'주 (월요일 시작)':<16}{'경기':>5}{'적중':>8}{'로그손실 살아있는':>14}{'고정':>8}")
    for w, g in D.groupby("week"):
        print(f"  {w.strftime('%Y-%m-%d'):<18}{len(g):>5}{g['hit'].mean():>8.0%}{g['ll'].mean():>14.3f}{g['ll_fix'].mean():>9.3f}")
    D["err"] = (D["gh"] - D["ga"]) - (D["live_mh"] - D["live_ma"])
    print("\n  [가장 크게 틀린 10경기] 원인: 그 팀 숫자 중 기대에서 가장 크게 벗어난 것")
    def cause(s, r):
        z = {}
        if r[f"{s}_open_e"] > 0:
            z["슈팅 수"] = (r[f"{s}_open"] - r[f"{s}_open_e"]) / np.sqrt(r[f"{s}_open_e"])
        if r[f"{s}_open"] > 0:
            z["슈팅 질"] = (r[f"{s}_q"] - r[f"{s}_q_e"]) / (0.11 / np.sqrt(r[f"{s}_open"]))
        exp_g = r[f"{s}_xg"] * E.C["calib_open"]
        z["골 전환(운)"] = (r[f"{s}_gopen"] - exp_g) / np.sqrt(max(exp_g, 0.1))
        k = max(z, key=lambda t: abs(z[t]))
        return f"{k} {'↑' if z[k] > 0 else '↓'}"
    for _, r in D.reindex(D["err"].abs().sort_values(ascending=False).index).head(10).iterrows():
        print(f"  {pd.to_datetime(r['dt']):%m/%d} {r['home']} {int(r['gh'])}-{int(r['ga'])} {r['away']}  "
              f"(예측 {r['live_mh']:.1f}-{r['live_ma']:.1f}) | 홈: {cause('h', r)}, 슈팅 {int(r['h_open'])}/{r['h_open_e']:.1f}, xG {r['h_xg']:.2f} | "
              f"원정: {cause('a', r)}, 슈팅 {int(r['a_open'])}/{r['a_open_e']:.1f}, xG {r['a_xg']:.2f}")
    print("  (슈팅 a/b = 실제 오픈플레이 슈팅 / 기대. 골 전환(운)이 원인이면 시뮬보다는 그날 운인 경우가 많음)")

for t in tests:
    R, D = evaluate(t)
    if t == latest:
        notebook(D, t)
        cur, _ = L.write_current(R, T)
        pw = sorted(cur, key=lambda t_: -(cur[t_]["att_open"] * cur[t_]["att_q"] / (cur[t_]["def_open"] * cur[t_]["def_q"])))
        print(f"\n===== 지금 시점 팀 값 (살아있는) — 상위 8 / 하위 5 =====")
        print(f"  {'팀':<26}{'경기':>5}{'슈팅':>7}{'허용':>7}{'질':>7}{'허용질':>7}{'세트':>7}")
        for tm in pw[:8] + pw[-5:]:
            v = cur[tm]
            print(f"  {tm:<26}{v['games_this_season']:>5}{v['att_open']:>7.2f}{v['def_open']:>7.2f}{v['att_q']:>7.2f}{v['def_q']:>7.2f}{v['att_sp']:>7.2f}")
        print("  저장: data/live_teams.json, data/live_params.json")
