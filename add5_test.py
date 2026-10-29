"""
5번 가중치 테스트 (김니스타크 아이디어, 10/29)
  시뮬 5번 단계 = 오픈플레이 슈팅이 골이 되는 확률 (xG × 0.87 × 결정력 × 상대 GK)
  여기에 팀별 가중치를 '더함' (곱하기 아님)

  매주: 지지난주까지 정보로 지난주 경기 예측 → 팀마다 예상 골 vs 실제 골
  차이가 문턱을 넘으면:
    그 경기를 가중치 바꿔가며 다시 시뮬 → 평균 골이 실제 골과 같아지는 더하기 값 δ 를 찾음 (반씩 좁혀가기)
    δ 의 절반은 넣은 팀 공격 가중치에, 절반은 먹은 팀 수비 가중치에 더함
      (둘 다 통째로 더하면 다시 만났을 때 2배가 되니까 반반)
  가중치는 누적, 시즌 시작 때 0

  비교: 살아있는 (지금 방식) vs 살아있는 + 5번 가중치 (문턱 0.3 / 0.5 / 0.7 / 1.0)
  전부 그 주 시작 전 정보로만 예측

사용: py add5_test.py          (25/26 + 26/27 지금까지)
      py add5_test.py 2026     (26/27만)
"""
import copy
import sys

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 300          # 예측용 시뮬 횟수
N_FIND = 200     # 가중치 찾을 때 시뮬 횟수
ITER = 7         # 반씩 좁히는 횟수 (−0.3 ~ +0.6 범위를 0.007 단위까지)
THRS = [0.3, 0.5, 0.7, 1.0]
MAXG = 6
VARIANTS = [None] + THRS
BASE = None

def vname(v):
    return "살아있는 (지금)" if v is None else f"살아있는 + 5번 문턱 {v}"

m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
print(f"경기 {len(m)}개. 채점할 시즌: {tests} · 예측 {N}번 / 가중치 찾기 {N_FIND}번 × {ITER}회\n")


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


def setup(R, home, away):
    for t in (home, away):
        v = R.team_values(t)
        E.C["teams"][t] = v
        E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}


def sim(R, home, away, W, n, seed=None):
    setup(R, home, away)
    E.ADD5 = W
    if seed is not None:
        E.rng = np.random.default_rng(seed)
    gh, ga, _ = E.simulate(home, away, n, use_players=False)
    E.ADD5 = {}
    return gh, ga


def find_delta(R, home, away, W, side, target, seed):
    """side 팀(0 홈 / 1 원정)의 골 평균이 target 이 되게 하는 5번 더하기 값. 같은 난수로 돌려서 비교가 공평하게"""
    att = home if side == 0 else away
    opp = away if side == 0 else home
    lo, hi = -0.3, 0.6
    for _ in range(ITER):
        mid = (lo + hi) / 2
        W2 = dict(W)
        a = W2.get(att, (0.0, 0.0)); W2[att] = (a[0] + mid, a[1])
        g = sim(R, home, away, W2, N_FIND, seed)[side].mean()
        if g < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def evaluate(test):
    p = season_params(test)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    S = {"week": None, "Rw": None, "pending": [], "W": {v: {} for v in THRS}, "found": {v: [] for v in THRS}}
    out = []

    def close_week():
        for v, h, a, mh, ma, gh, ga, seed in S["pending"]:
            W = S["W"][v]
            for side, pred, act in ((0, mh, gh), (1, ma, ga)):
                if abs(act - pred) <= v:
                    continue
                d = find_delta(S["Rw"], h, a, W, side, act, seed)
                S["found"][v].append(d)
                att, opp = (h, a) if side == 0 else (a, h)
                wa = W.get(att, (0.0, 0.0)); W[att] = (wa[0] + d / 2, wa[1])
                wo = W.get(opp, (0.0, 0.0)); W[opp] = (wo[0], wo[1] + d / 2)
        S["pending"] = []

    def before(rows, R):
        r0 = rows[0]
        if r0.season != test:
            return
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            if S["Rw"] is not None:
                close_week()
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
        for r in rows:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rows if r.home); a = next(r for r in rows if not r.home)
        gh, ga = int(h.gf), int(a.gf)
        row = {"match_id": h.match_id, "dt": h.dt, "home": h.team, "away": a.team, "gh": gh, "ga": ga}
        seed = int(h.match_id)
        for v in VARIANTS:
            sh, sa = sim(S["Rw"], h.team, a.team, {} if v is None else S["W"][v], N)
            mh, ma = sh.mean(), sa.mean()
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            mat /= N
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            row[(v, "dev")] = L.poisson_dev(gh, mh) + L.poisson_dev(ga, ma)
            row[(v, "ll")] = -np.log(max(ps, 1e-3))
            row[(v, "p")] = ps
            row[(v, "top3")] = float((mat > ps).sum() < 3)
            if v is not None:
                S["pending"].append((v, h.team, a.team, mh, ma, gh, ga, seed))
        out.append(row)
        if len(out) % 100 == 0:
            print(f"    {len(out)}경기 예측")

    print(f"===== [{test}/{(test + 1) % 100}] 주 단위로 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    E.rng = np.random.default_rng()
    return pd.DataFrame(out), S["W"], S["found"]


def report(D, label):
    print(f"\n  ----- {label}: {len(D)}경기 -----")
    print(f"  {'방식':<24}{'골 오차':>9}{'스코어 손실':>11}{'실제스코어 확률':>14}{'TOP3':>7}{'골오차 확실도':>13}{'스코어 확실도':>13}")
    for v in VARIANTS:
        z = []
        for k in ("dev", "ll"):
            d = D[(BASE, k)] - D[(v, k)]
            z.append(0.0 if v == BASE else d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12))
        print(f"  {vname(v):<24}{D[(v, 'dev')].mean():>9.4f}{D[(v, 'll')].mean():>11.4f}{D[(v, 'p')].mean():>14.1%}"
              f"{D[(v, 'top3')].mean():>7.0%}{z[0]:>+13.1f}{z[1]:>+13.1f}")
    print("  골 오차·스코어 손실: 낮을수록 좋음 / 확실도: 지금 방식 대비. +2 이상이면 맞다, 그 밑이면 아니다")


all_D = []
for t in tests:
    D, W, found = evaluate(t)
    report(D, f"{t}/{(t + 1) % 100}")
    all_D.append(D)
    for v in THRS:
        f = np.array(found[v])
        if len(f):
            print(f"  [문턱 {v}] 가중치 찾은 횟수 {len(f)} · 찾은 값 평균 {f.mean():+.3f}, 크기 평균 {np.abs(f).mean():.3f}")
    Wv = W[0.5]
    pw = sorted(Wv, key=lambda k: -(Wv[k][0] - Wv[k][1]))
    print(f"  [{t}/{(t + 1) % 100} 끝났을 때 문턱 0.5 가중치] (슈팅당 골 확률에 더하는 값. 평균 골 확률은 0.107)")
    for k in pw[:5] + pw[-5:]:
        print(f"    {k:<26} 공격 {Wv[k][0]:+.3f}  수비 {Wv[k][1]:+.3f}")
if len(all_D) > 1:
    report(pd.concat(all_D, ignore_index=True), "전체 합계")
