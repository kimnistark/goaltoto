"""
문턱 가중치 테스트 (김니스타크 아이디어)
  지지난주까지 결과로 지난주 경기를 예측 → 팀마다 '예상 골'과 '실제 골' 비교
  차이가 문턱(예 0.5)을 넘으면 그 팀 가중치를 (예상 ± 문턱) / 예상 배 곱함
    예) 예상 1.4골, 실제 3골, 문턱 0.5 → 득점 가중치 × 1.9/1.4
        예상 2.1골, 실제 0골, 문턱 0.5 → 득점 가중치 × 1.6/2.1
    실점도 똑같이: 예상 실점보다 문턱 넘게 더 먹으면 실점 가중치 × (예상+문턱)/예상
  가중치는 매주 누적 (시즌 시작 때 1로 시작)
  기준: 골만 사용 (xG 안 씀)

비교하는 것 (전부 '그 주 시작 전 정보'로만 예측):
  고정            시즌 시작 값 그대로
  고정 + 문턱      시즌 시작 값 × 문턱 가중치
  살아있는         지금 쓰는 방식 (매 경기 슈팅·xG로 조금씩 갱신)
  살아있는 + 문턱   살아있는 값 × 문턱 가중치

사용: py threshold_test.py          (25/26 + 26/27 지금까지)
      py threshold_test.py 2026     (26/27만)
"""
import copy
import sys

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 1000
THRS = [0.3, 0.5, 0.7, 1.0]
MAXG = 6
VARIANTS = [("fix", None)] + [("fix", t) for t in THRS] + [("live", None)] + [("live", t) for t in THRS]
BASE = ("live", None)

def vname(v):
    return ("고정" if v[0] == "fix" else "살아있는") + (f" + 문턱 {v[1]}" if v[1] is not None else "")

m, T = L.load_team_matches()
latest = int(T["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
print(f"경기 {len(m)}개. 채점할 시즌: {tests} · 경기당 {N}번 시뮬 · 방식 {len(VARIANTS)}개\n")


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


def sim(R, home, away, mult):
    for t in (home, away):
        v = dict(R.team_values(t))
        ma, md = mult.get(t, (1.0, 1.0))
        v["att_open"] *= ma; v["att_sp"] *= ma
        v["def_open"] *= md; v["def_sp"] *= md
        E.C["teams"][t] = v
        E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
    gh, ga, _ = E.simulate(home, away, N, use_players=False)
    mat = np.zeros((MAXG + 1, MAXG + 1))
    np.add.at(mat, (np.minimum(gh, MAXG), np.minimum(ga, MAXG)), 1)
    return gh.mean(), ga.mean(), mat / N


def step(m_, pred, actual, thr):
    """예상과 실제 차이가 문턱을 넘으면 (예상 ± 문턱) / 예상 배"""
    pred = max(pred, 0.05)
    if actual - pred > thr:
        return m_ * (pred + thr) / pred
    if pred - actual > thr:
        return m_ * max(pred - thr, 0.5 * pred) / pred      # 예상이 문턱보다 작으면 최대 절반까지만
    return m_


def evaluate(test):
    p = season_params(test)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    S = {"week": None, "F": None, "Rw": None, "pending": [], "mult": {v: {} for v in VARIANTS if v[1] is not None}}
    out = []

    def close_week():
        # 지난주 예측 vs 실제 골 → 문턱 가중치 갱신 (다음 주 예측부터 적용)
        for v, h, a, ph, pa, gh, ga in S["pending"]:
            M = S["mult"][v]
            for team, pf, gf, pa_, ga_ in ((h, ph, gh, pa, ga), (a, pa, ga, ph, gh)):
                ma, md = M.get(team, (1.0, 1.0))
                M[team] = (step(ma, pf, gf, v[1]), step(md, pa_, ga_, v[1]))
        S["pending"] = []

    def before(rows, R):
        r0 = rows[0]
        if r0.season != test:
            return
        if S["F"] is None:
            S["F"] = copy.deepcopy(R)                     # 시즌 시작 값 (고정)
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            close_week()
            S["week"], S["Rw"] = wk, copy.deepcopy(R)     # 이번 주 예측은 지난주까지 정보로만
        for r in rows:
            S["F"].touch(r.team, r.season); S["Rw"].touch(r.team, r.season)
        h = next(r for r in rows if r.home); a = next(r for r in rows if not r.home)
        gh, ga = int(h.gf), int(a.gf)
        row = {"match_id": h.match_id, "dt": h.dt, "week": wk, "home": h.team, "away": a.team, "gh": gh, "ga": ga}
        for v in VARIANTS:
            base = S["F"] if v[0] == "fix" else S["Rw"]
            mh, ma, mat = sim(base, h.team, a.team, S["mult"].get(v, {}))
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            row[(v, "dev")] = L.poisson_dev(gh, mh) + L.poisson_dev(ga, ma)
            row[(v, "ll")] = -np.log(max(ps, 1e-4))
            row[(v, "p")] = ps
            row[(v, "top3")] = float((mat > ps).sum() < 3)
            if v[1] is not None:
                S["pending"].append((v, h.team, a.team, mh, ma, gh, ga))
        out.append(row)
        if len(out) % 100 == 0:
            print(f"    {len(out)}경기 예측")

    print(f"===== [{test}/{(test + 1) % 100}] 주 단위로 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    close_week()
    return pd.DataFrame(out), S["mult"]


def report(D, label):
    print(f"\n  ----- {label}: {len(D)}경기 -----")
    print(f"  {'방식':<22}{'골 오차':>9}{'스코어 손실':>11}{'실제스코어 확률':>14}{'TOP3':>7}{'골오차 확실도':>13}{'스코어 확실도':>13}")
    for v in VARIANTS:
        z = []
        for k in ("dev", "ll"):
            d = D[(BASE, k)] - D[(v, k)]          # +면 이 방식이 살아있는보다 나음
            z.append(0.0 if v == BASE else d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12))
        print(f"  {vname(v):<22}{D[(v, 'dev')].mean():>9.4f}{D[(v, 'll')].mean():>11.4f}{D[(v, 'p')].mean():>14.1%}"
              f"{D[(v, 'top3')].mean():>7.0%}{z[0]:>+13.1f}{z[1]:>+13.1f}")
    print("  골 오차·스코어 손실: 낮을수록 좋음 / 확실도: '살아있는'(지금 방식) 대비. +면 이 방식이 나음, +2 이상이면 우연 아닐 가능성 큼")


all_D = []
for t in tests:
    D, mult = evaluate(t)
    report(D, f"{t}/{(t + 1) % 100}")
    all_D.append(D)
    v = ("live", 0.5)
    M = mult[v]
    pw = sorted(M, key=lambda k: -M[k][0] / M[k][1])
    print(f"  [{t}/{(t + 1) % 100} 끝났을 때 '살아있는 + 문턱 0.5' 가중치] (득점 ×, 실점 ×)")
    for k in pw[:5] + pw[-5:]:
        print(f"    {k:<26} 득점 ×{M[k][0]:.2f}  실점 ×{M[k][1]:.2f}")
if len(all_D) > 1:
    report(pd.concat(all_D, ignore_index=True), "전체 합계")
