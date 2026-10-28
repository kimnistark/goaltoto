"""
라운드 채점 (2단계: 라운드 기록장)
records/pred_*.csv 에 박제해둔 예측을 실제 결과(data/matches.csv)와 비교
  - 실제 스코어가 시뮬 TOP 1 / TOP 3 / TOP 5 안에 있었나
  - 실제 스코어에 시뮬이 준 확률, 순위
  - 실제 골차에 시뮬이 준 확률
  - 비교 기준: '아무 정보 없이 리그 평균 골로만 찍었을 때' 같은 숫자
먼저 py download_understat.py 로 결과를 받아야 함
사용: py score_round.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from math import exp, factorial

HERE = Path(__file__).parent
REC, DATA = HERE / "records", HERE / "data"
MAXG = 6
files = sorted(REC.glob("pred_*.csv"))
if not files:
    raise SystemExit("records/ 에 예측 파일이 없어. 경기 전에 py predict_round.py 를 먼저 돌려야 함")
res = pd.read_csv(DATA / "matches.csv").set_index("match_id")
avg_h, avg_a = res["FTHG"].mean(), res["FTAG"].mean()      # 기준: 리그 평균 골만 아는 경우

def pois(k, lam):
    return exp(-lam) * lam ** k / factorial(k)
base_mat = np.array([[pois(i, avg_h) for _ in range(MAXG + 1)] for i in range(MAXG + 1)]) * \
           np.array([[pois(j, avg_a) for j in range(MAXG + 1)] for _ in range(MAXG + 1)])

all_rows = []
for f in files:
    P = pd.read_csv(f)
    graded = []
    for r in P.itertuples():
        if r.match_id not in res.index:
            continue
        gh, ga = int(res.at[r.match_id, "FTHG"]), int(res.at[r.match_id, "FTAG"])
        mat = np.array(json.loads(r.matrix))
        i, j = min(gh, MAXG), min(ga, MAXG)
        p = mat[i, j]
        rank = int((mat > p).sum()) + 1
        gd = gh - ga
        gdk = f"gd_{max(-4, min(4, gd)):+d}"
        bp = base_mat[i, j]; brank = int((base_mat > bp).sum()) + 1
        bgd = sum(base_mat[a, b] for a in range(MAXG + 1) for b in range(MAXG + 1) if max(-4, min(4, a - b)) == max(-4, min(4, gd)))
        graded.append({"file": f.name, "match_id": r.match_id, "datetime": r.datetime, "league": r.league,
                       "home": r.home, "away": r.away, "result": f"{gh}-{ga}", "pred_top1": r.top1,
                       "mean": f"{r.mean_h:.1f}-{r.mean_a:.1f}", "p_score": p, "rank": rank,
                       "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5, "p_gd": float(P.loc[P["match_id"] == r.match_id, gdk].iloc[0]),
                       "base_p_score": bp, "base_rank": brank, "base_p_gd": bgd})
    G = pd.DataFrame(graded)
    print(f"===== {f.name}: 예측 {len(P)}경기 중 결과 나온 {len(G)}경기 =====")
    if G.empty:
        print("  아직 결과 나온 경기가 없음\n"); continue
    print(f"  {'':<26}{'시뮬':>8}{'리그 평균만':>12}")
    print(f"  {'실제 스코어가 TOP 1':<24}{G['top1'].mean():>9.0%}{(G['base_rank'] <= 1).mean():>12.0%}")
    print(f"  {'실제 스코어가 TOP 3 안':<23}{G['top3'].mean():>9.0%}{(G['base_rank'] <= 3).mean():>12.0%}")
    print(f"  {'실제 스코어가 TOP 5 안':<23}{G['top5'].mean():>9.0%}{(G['base_rank'] <= 5).mean():>12.0%}")
    print(f"  {'실제 스코어에 준 확률 (평균)':<20}{G['p_score'].mean():>9.1%}{G['base_p_score'].mean():>12.1%}")
    print(f"  {'실제 골차에 준 확률 (평균)':<21}{G['p_gd'].mean():>9.1%}{G['base_p_gd'].mean():>12.1%}")
    print("  (리그 평균만 = 팀 정보 없이 홈·원정 평균 골로만 찍은 경우. 시뮬이 이것보다 높아야 의미 있음)\n")
    print(f"  {'경기':<46}{'결과':>6}{'시뮬 1위':>9}{'평균 골':>9}{'결과 확률':>10}{'순위':>5}")
    for r in G.sort_values("datetime").itertuples():
        print(f"  {r.home[:20]:>20} vs {r.away[:21]:<21}{r.result:>6}{r.pred_top1:>9}{r.mean:>9}{r.p_score:>10.1%}{r.rank:>5}")
    print()
    all_rows.append(G)

if all_rows:
    A = pd.concat(all_rows)
    A.to_csv(REC / "scores.csv", index=False, encoding="utf-8-sig")
    if len(all_rows) > 1:
        print(f"===== 전체 누적 {len(A)}경기 =====")
        print(f"  TOP1 {A['top1'].mean():.0%} · TOP3 {A['top3'].mean():.0%} · TOP5 {A['top5'].mean():.0%} · "
              f"결과 스코어 확률 {A['p_score'].mean():.1%} (리그 평균만 {A['base_p_score'].mean():.1%}) · 결과 골차 확률 {A['p_gd'].mean():.1%}")
    print("저장: records/scores.csv")
