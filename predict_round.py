"""
라운드 예측 저장 (2단계: 라운드 기록장)
경기 시작 전에 돌려서 시뮬 예측을 records/ 에 박제 → 경기 끝나면 score_round.py 가 채점
  저장: 스코어 분포 전체(0~6골, 6 = 6골 이상), 많이 나온 스코어 TOP 5, 골차 분포, 양 팀 평균 골
  한 번 저장한 파일은 덮어쓰지 않음 (나중에 고치면 채점이 의미 없어지니까)

순서 (경기 전):
  py download_understat.py      (최신 결과 + 일정)
  py live.py
  py prepare_players.py live
  py prepare_shapes.py live
  py prepare_chain.py live
  py predict_round.py           (다가올 7일 경기. 기간 바꾸려면 py predict_round.py 4)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sim_engine as E

N = 20000
MAXG = 6
HERE = Path(__file__).parent
REC = HERE / "records"
days = int(next((a for a in sys.argv[1:] if a.isdigit()), 7))

fx_path = E.DATA / "fixtures.csv"
if not fx_path.exists():
    sys.exit("data/fixtures.csv 가 없어. 먼저 py download_understat.py 를 돌려줘 (일정도 같이 받게 바뀜)")
if not E.LIVE:
    print("[주의] live 파일이 없어서 '시즌 기준' 값으로 예측함. live.py 와 prepare_*.py live 를 먼저 돌리는 게 맞음\n")

fx = pd.read_csv(fx_path)
fx = fx[fx["league"].isin(E.P["meta"]["leagues"])].copy()
fx["dt"] = pd.to_datetime(fx["datetime"])
last_result = pd.to_datetime(pd.read_csv(E.DATA / "matches.csv", usecols=["datetime"])["datetime"]).max()
fx = fx[fx["dt"] > last_result].sort_values("dt")
if fx.empty:
    sys.exit("예측할 경기가 없어 (일정이 비었거나 데이터가 최신이 아님)")
start = fx["dt"].min()
fx = fx[fx["dt"] < start.normalize() + pd.Timedelta(days=days)]

REC.mkdir(exist_ok=True)
out_path = REC / f"pred_{start:%Y-%m-%d}.csv"
if out_path.exists() and "--force" not in sys.argv:
    sys.exit(f"{out_path.name} 이 이미 있어. 예측은 한 번 저장하면 안 고치는 게 원칙이라 멈춤 (정말 다시 하려면 --force)")

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
except Exception:
    commit = ""
asof = E.LIVE_META.get("asof", str(last_result.date()))
print(f"{len(fx)}경기 예측 ({start:%m/%d} ~ {fx['dt'].max():%m/%d}) · 데이터 {asof}까지 · 경기당 {N:,}번 · 보정 c1={E.CORR_C1} · 코드 {commit or '?'}\n")

rows = []
for r in fx.itertuples():
    unknown = [t for t in (r.home, r.away) if t not in E.C["teams"]]
    gh, ga, _ = E.simulate(r.home, r.away, N)
    gd = gh - ga
    mat = np.zeros((MAXG + 1, MAXG + 1))
    np.add.at(mat, (np.minimum(gh, MAXG), np.minimum(ga, MAXG)), 1)
    mat /= N
    order = np.dstack(np.unravel_index(np.argsort(-mat, axis=None), mat.shape))[0][:5]
    top = [f"{i}-{j}" for i, j in order]
    row = {"match_id": r.match_id, "datetime": r.datetime, "league": r.league, "home": r.home, "away": r.away,
           "mean_h": round(float(gh.mean()), 3), "mean_a": round(float(ga.mean()), 3)}
    for k, (i, j) in enumerate(order, 1):
        row[f"top{k}"] = f"{i}-{j}"
        row[f"top{k}_p"] = round(float(mat[i, j]), 4)
    for g in range(-4, 5):
        p = (gd <= -4).mean() if g == -4 else (gd >= 4).mean() if g == 4 else (gd == g).mean()
        row[f"gd_{g:+d}"] = round(float(p), 4)
    row["matrix"] = json.dumps(np.round(mat, 4).tolist())
    row.update({"data_asof": asof, "commit": commit, "c1": E.CORR_C1, "n_sims": N, "unknown_team": ",".join(unknown)})
    rows.append(row)
    print(f"  {pd.to_datetime(r.datetime):%m/%d %H:%M} {r.home:>24} {gh.mean():.2f} - {ga.mean():.2f} {r.away:<24} "
          f"TOP3 {top[0]} {mat[tuple(order[0])]:.0%} / {top[1]} {mat[tuple(order[1])]:.0%} / {top[2]} {mat[tuple(order[2])]:.0%}"
          f"  | 홈 2골차+ {(gd >= 2).mean():.0%} 원정 2골차+ {(gd <= -2).mean():.0%}" + (f"  [데이터 없는 팀: {', '.join(unknown)}]" if unknown else ""))

pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n저장: records/{out_path.name} ({len(rows)}경기)")
print("깃허브에 올려두면 '경기 전에 이렇게 예측했다'가 기록으로 남음:")
print(f'  git add records\n  git commit -m "예측 {start:%Y-%m-%d}"\n  git push')
