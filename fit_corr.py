"""
전력 차 보정값 확정 → data/corr.json 저장 (시뮬과 html이 이 파일을 읽음)
  보정 후 평균 골 = 시뮬 평균 골 + c1 × (그 팀 평균 골 − 상대 평균 골)     → 강팀은 조금 더, 약팀은 조금 덜
먼저 py run_seasons.py 로 시즌별 data/sim_means_20XX.csv 가 있어야 함
사용: py fit_corr.py          (찾아서 저장)
      py fit_corr.py off      (보정 끄기: corr.json 삭제)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = DATA / "corr.json"
if len(sys.argv) > 1 and sys.argv[1] == "off":
    OUT.unlink(missing_ok=True); sys.exit("보정을 껐어 (data/corr.json 삭제). py build_html.py 를 다시 돌리면 html에도 반영돼")
files = sorted(DATA.glob("sim_means_20*.csv"))
if len(files) < 2:
    raise SystemExit("시즌별 파일이 2개 이상 필요해. 먼저 py run_seasons.py 를 돌려줘")
parts = []
for f in files:
    d = pd.read_csv(f); d["season"] = int(f.stem.split("_")[-1]); parts.append(d)
M = pd.concat(parts, ignore_index=True)
last, seasons = int(M["season"].max()), [int(v) for v in sorted(M["season"].unique())]
n = len(M)
mu = np.concatenate([M["sim_h"].to_numpy(), M["sim_a"].to_numpy()])
x = mu - np.concatenate([M["sim_a"].to_numpy(), M["sim_h"].to_numpy()])
y = np.concatenate([M["real_h"].to_numpy(), M["real_a"].to_numpy()]).astype(float)
tr = np.tile((M["season"] < last).to_numpy(), 2); te = ~tr
GRID = np.round(np.arange(-0.05, 0.2001, 0.005), 3)

def dev(c1):
    p = np.maximum(mu + c1 * x, 0.05)
    return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / p), 0) - (y - p))
def best(mask):
    sc = [(dev(c)[mask].mean(), c) for c in GRID]
    return min(sc)[1]

d0 = dev(0.0)
c_tr = best(tr)
df = (d0 - dev(c_tr))[te]
z = df.mean() / (df.std() / np.sqrt(len(df)))
print(f"시즌 {seasons}, 경기 {n}개\n")
print(f"===== [1] 앞 시즌들({seasons[0]}~{last - 1})로 찾고 {last}/{(last + 1) % 100}으로 채점 =====")
print(f"  찾은 값 c1 = {c_tr:.3f}")
print(f"  {last}/{(last + 1) % 100} 오차 개선 {(1 - dev(c_tr)[te].mean() / d0[te].mean()) * 100:+.2f}%, 확실도 {z:+.1f}  (+2 이상이면 진짜일 가능성이 큼)")
print(f"\n===== [2] c1을 바꿔가며 본 {last}/{(last + 1) % 100} 오차 개선 (값이 조금 달라도 결과가 비슷한지) =====")
for c in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15]:
    dd = (d0 - dev(c))[te]
    print(f"  c1 = {c:.2f}: {(1 - dev(c)[te].mean() / d0[te].mean()) * 100:+.2f}%  확실도 {dd.mean() / (dd.std() / np.sqrt(len(dd)) + 1e-12):+.1f}")
print("\n===== [3] 시즌별로 따로 찾으면 (시즌마다 비슷하게 나와야 믿을 만함) =====")
for s in seasons:
    mk = np.tile((M["season"] == s).to_numpy(), 2)
    print(f"  {s}/{(s + 1) % 100}: c1 = {best(mk):.3f}")
c_all = best(np.ones(2 * n, bool))
print(f"\n===== [4] 전 시즌을 합쳐서 찾은 최종값: c1 = {c_all:.3f} =====")
if c_all <= 0 or z < 1.0:
    print("  채점에서 효과가 확인되지 않아서 저장하지 않음 (보정 없이 감)")
    OUT.unlink(missing_ok=True)
else:
    OUT.write_text(json.dumps({"c1": float(c_all), "found_on_prior_seasons": float(c_tr), "test_season": last, "test_z": round(float(z), 2),
                               "seasons": seasons}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  예: 예상 골차 " + ", ".join(f"{g:.1f} → 강팀 {c_all * g:+.3f} / 약팀 {-c_all * g:+.3f}" for g in (0.5, 1.0, 1.5, 2.0)))
    print("  저장: data/corr.json  → 이제 py build_html.py 와 py validate_handicap.py 를 돌리면 보정이 적용됨")
