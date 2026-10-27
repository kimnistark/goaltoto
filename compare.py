"""
보정 전 vs 보정 후 비교 (같은 시뮬 결과에 보정만 얹어서 비교하므로 우연 차이 없음)
사용: py compare.py                          (강팀 상위 4 × 약팀 하위 4 자동, 강팀 홈)
      py compare.py "Manchester City" Burnley (홈팀 원정팀 직접 지정. 이름 일부만 써도 됨)
"""
import sys

import numpy as np

import sim_engine as E

N = 20000
teams = sorted(E.C["teams"], key=E.power, reverse=True)

def find(q):
    hit = [t for t in teams if q.lower() == t.lower()] or [t for t in teams if q.lower() in t.lower()]
    if not hit:
        sys.exit(f"'{q}' 팀을 못 찾았어. 팀 목록: " + ", ".join(sorted(teams)))
    return hit[0]

def stats(gh, ga):
    gd = gh - ga
    sc = {}
    for a, b in zip(gh, ga):
        sc[(a, b)] = sc.get((a, b), 0) + 1
    top = sorted(sc.items(), key=lambda kv: -kv[1])[:5]
    return {"h": gh.mean(), "a": ga.mean(), "w": (gd > 0).mean(), "d": (gd == 0).mean(), "l": (gd < 0).mean(),
            "w2": (gd >= 2).mean(), "w3": (gd >= 3).mean(), "top": "  ".join(f"{a}-{b} {c / len(gh):.1%}" for (a, b), c in top),
            "gd": {k: ((gd == k).mean() if abs(k) < 4 else ((gd >= 4).mean() if k > 0 else (gd <= -4).mean())) for k in range(-4, 5)}}

def run(home, away):
    gh, ga, _ = E.simulate(home, away, N, corr=False)
    mh, ma = gh.mean(), ga.mean()
    ch, ca = gh.copy(), ga.copy()
    c1 = E.CORR_C1
    if c1 > 0 and mh != ma:                         # sim_engine과 같은 보정을 같은 결과 위에 얹음
        delta = c1 * abs(mh - ma)
        if mh > ma:
            ch = ch + E.rng.poisson(delta, N); ca = E.rng.binomial(ca, max(0.0, 1 - delta / max(ma, 1e-9)))
        else:
            ca = ca + E.rng.poisson(delta, N); ch = E.rng.binomial(ch, max(0.0, 1 - delta / max(mh, 1e-9)))
    A, B = stats(gh, ga), stats(ch, ca)
    print(f"\n===== {home} (홈) vs {away} (원정) =====")
    print(f"  {'':<16}{'보정 전':>9}{'보정 후':>9}{'변화':>9}")
    rows = [("홈 평균 골", "h", "{:.2f}", 1), ("원정 평균 골", "a", "{:.2f}", 1), ("홈 승", "w", "{:.1%}", 100), ("무", "d", "{:.1%}", 100),
            ("원정 승", "l", "{:.1%}", 100), ("홈 2골차 이상 승", "w2", "{:.1%}", 100), ("홈 3골차 이상 승", "w3", "{:.1%}", 100)]
    for lab, k, f, sc in rows:
        d = (B[k] - A[k]) * sc
        print(f"  {lab:<14}{f.format(A[k]):>10}{f.format(B[k]):>10}{d:>+8.2f}" + ("" if sc == 1 else "p"))
    print("  골 차 분포 (홈 − 원정)  " + "".join(f"{('≤-4' if k == -4 else '≥+4' if k == 4 else f'{k:+d}'):>7}" for k in range(-4, 5)))
    print("    보정 전              " + "".join(f"{A['gd'][k]:>7.1%}" for k in range(-4, 5)))
    print("    보정 후              " + "".join(f"{B['gd'][k]:>7.1%}" for k in range(-4, 5)))
    print(f"  많이 나온 스코어  전: {A['top']}")
    print(f"                  후: {B['top']}")

print(f"전력 차 보정 c1 = {E.CORR_C1:.3f}" + ("" if E.CORR_C1 > 0 else "  ← 보정이 꺼져 있어 (py fit_corr.py 를 먼저 돌려줘)") + f" · 경기마다 {N:,}번 시뮬")
args = sys.argv[1:]
if len(args) >= 2:
    run(find(args[0]), find(args[1]))
else:
    strong, weak = teams[:4], teams[-4:]
    print("강팀 상위 4: " + ", ".join(strong) + "\n약팀 하위 4: " + ", ".join(weak))
    for s_, w_ in zip(strong, reversed(weak)):
        run(s_, w_)
    run(weak[-1], strong[0])                     # 반대로: 약팀 홈, 강팀 원정
