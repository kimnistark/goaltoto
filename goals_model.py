"""
② 골 기반 모델 + ④ 팀별 스코어 성향 (11/18) — 1위 학습기 재료
  시뮬은 xG만 봄. 근데 어떤 팀은 시즌 내내 xG보다 더 넣거나 덜 먹음 → 실제 골로 팀 값을 따로 매기는 모델을 만들어서 섞음
  ② 골 모델: 홈 골 ~ 포아송(exp(평균 + 홈 + 홈팀 공격 + 원정팀 수비 구멍)), 원정 골도 같은 식
     경기 하나 끝날 때마다 (실제 골 − 예상 골) × η 만큼 공격/수비 값을 움직임 (η 는 21/22~22/23 로 고름)
     시즌 바뀌면 값을 carry 만큼만 남김, 승격팀은 약하게 시작
     + 저득점 칸 보정 (0-0, 1-0, 0-1, 1-1) ρ
  ④ 팀 성향: 그 팀 경기에서 '무승부' / '합계 2골 이하' 가 골 모델 예상보다 얼마나 더 나왔나 (최근 경기일수록 크게, 경기 전 값)
  전부 그 경기 전 정보로만 → data/goals_model.json (top1_learn.py 가 읽음)

사용: py goals_model.py   (1분 안쪽)
"""
import json
from collections import defaultdict
from math import exp, factorial, log
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MAXG = 6
LEAGUES = ["EPL", "La_Liga", "Bundesliga"]
CLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3)]

m = pd.read_csv(DATA / "matches.csv")
m = m[m["league"].isin(LEAGUES)].copy()
m["dt"] = pd.to_datetime(m["datetime"])
m = m.sort_values("dt").reset_index(drop=True)
seasons = sorted(m["season"].unique())
FIT = [s for s in seasons[1:3]]                  # η 고르기용 (첫 시즌은 워밍업)
pmf = lambda k, lam: exp(-lam) * lam ** k / factorial(k)


def run(eta, carry=0.6, rho=-0.05, decay=0.03, keep=False):
    att = defaultdict(float); dfn = defaultdict(float); last = {}
    mu = {lg: log(1.35) for lg in LEAGUES}; home = {lg: 0.2 for lg in LEAGUES}
    tend = defaultdict(lambda: [0.0, 0.0, 0.0])      # 팀 -> [무승부 잔차 합, 저득점 잔차 합, 가중치 합]
    ll, out = 0.0, {}
    for r in m.itertuples():
        for t in (r.home, r.away):
            if last.get(t) != r.season:
                if t in last:
                    att[t] *= carry; dfn[t] *= carry
                else:
                    att[t], dfn[t] = (-0.15, 0.15) if r.season > seasons[0] else (0.0, 0.0)
                last[t] = r.season
        lh = exp(mu[r.league] + home[r.league] + att[r.home] + dfn[r.away])
        la = exp(mu[r.league] + att[r.away] + dfn[r.home])
        P = np.outer([pmf(i, lh) for i in range(MAXG + 1)], [pmf(j, la) for j in range(MAXG + 1)])
        P[0, 0] *= max(1 - lh * la * rho, 0); P[0, 1] *= max(1 + lh * rho, 0)
        P[1, 0] *= max(1 + la * rho, 0); P[1, 1] *= max(1 - rho, 0)
        P /= P.sum()
        gh, ga = min(int(r.FTHG), MAXG), min(int(r.FTAG), MAXG)
        if r.season in FIT:
            ll -= log(max(P[gh, ga], 1e-6))
        if keep:
            th, ta = tend[r.home], tend[r.away]
            out[int(r.match_id)] = {
                "dc": [float(P[i, j]) for i, j in CLS], "dc_mh": lh, "dc_ma": la,
                "t_draw": (th[0] + ta[0]) / (th[2] + ta[2] + 10),       # 10경기 분량만큼 0 쪽으로 당김
                "t_low": (th[1] + ta[1]) / (th[2] + ta[2] + 10)}
            draw_res = float(gh == ga) - float(np.trace(P))
            low = np.add.outer(np.arange(MAXG + 1), np.arange(MAXG + 1)) <= 2
            low_res = float(gh + ga <= 2) - float(P[low].sum())
            for t in (r.home, r.away):
                v = tend[t]; v[0] = v[0] * (1 - decay) + draw_res; v[1] = v[1] * (1 - decay) + low_res; v[2] = v[2] * (1 - decay) + 1
        # 업데이트
        eh, ea = int(r.FTHG) - lh, int(r.FTAG) - la
        att[r.home] += eta * eh; dfn[r.away] += eta * eh
        att[r.away] += eta * ea; dfn[r.home] += eta * ea
        mu[r.league] += 0.002 * (eh + ea); home[r.league] += 0.002 * (eh - ea)
    return ll, out


print(f"경기 {len(m)}개. η·carry 고르기: {FIT} 시즌 스코어 손실")
best = None
for eta in (0.02, 0.03, 0.045, 0.06, 0.08):
    for carry in (0.4, 0.6, 0.8):
        ll, _ = run(eta, carry)
        if best is None or ll < best[0]:
            best = (ll, eta, carry)
print(f"  → η {best[1]}, carry {best[2]}")
_, out = run(best[1], best[2], keep=True)
(DATA / "goals_model.json").write_text(json.dumps(out), encoding="utf-8")
td = pd.Series([v["t_draw"] for v in out.values()])
print(f"저장: data/goals_model.json ({len(out)}경기). 팀 무승부 성향 범위 {td.quantile(0.05):+.3f} ~ {td.quantile(0.95):+.3f}")
