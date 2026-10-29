"""
강팀 덜 넣나? 약팀 너무 넣나? 확인
walkforward.py 가 저장한 data/predictions_시즌.csv (경기마다 그 이전 정보만으로 한 예측) 를 실제 결과와 비교
시뮬 안 돌림. 몇 초면 끝남

  예측 골차 크기로 경기를 5칸으로 나눔 (전력 차 작은 경기 → 큰 경기)
  칸마다: 강팀 예측 골 vs 실제 골, 약팀 예측 골 vs 실제 골
  ± 는 운으로 흔들릴 수 있는 폭 (이 안이면 '차이 없음'으로 봄)

사용: py check_strong.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
files = sorted(DATA.glob("predictions_*.csv"))
if not files:
    raise SystemExit("data/predictions_*.csv 가 없어. 먼저 py walkforward.py 를 돌려야 함")
D = pd.concat([pd.read_csv(f).assign(season=f.stem.split("_")[1]) for f in files], ignore_index=True)
print(f"경기 {len(D)}개 ({', '.join(f.name for f in files)})\n")


def table(D, pre, label):
    gap = D[f"{pre}_mh"] - D[f"{pre}_ma"]
    home_strong = gap >= 0
    S = pd.DataFrame({
        "gap": gap.abs(),
        "s_pred": np.where(home_strong, D[f"{pre}_mh"], D[f"{pre}_ma"]),
        "s_real": np.where(home_strong, D["gh"], D["ga"]),
        "w_pred": np.where(home_strong, D[f"{pre}_ma"], D[f"{pre}_mh"]),
        "w_real": np.where(home_strong, D["ga"], D["gh"]),
        "s_home": home_strong,
    })
    S["bin"] = pd.qcut(S["gap"], 5, labels=False, duplicates="drop")
    print(f"===== [{label}] =====")
    print(f"  {'예측 골차':<14}{'경기':>5}{'강팀 홈':>8} │{'강팀 예측':>9}{'실제':>7}{'차이':>13} │{'약팀 예측':>9}{'실제':>7}{'차이':>13} │{'골차 차이':>9}")
    for b, g in S.groupby("bin"):
        n = len(g)
        ds = g["s_real"] - g["s_pred"]; dw = g["w_real"] - g["w_pred"]
        se_s = ds.std() / np.sqrt(n); se_w = dw.std() / np.sqrt(n)
        mark = lambda d, se: "  ←" if abs(d) > 2 * se else ""
        print(f"  {g['gap'].min():.2f}~{g['gap'].max():.2f}{'':<4}{n:>5}{g['s_home'].mean():>8.0%} │"
              f"{g['s_pred'].mean():>9.2f}{g['s_real'].mean():>7.2f}{ds.mean():>+7.2f} ±{2 * se_s:.2f}{mark(ds.mean(), se_s)} │"
              f"{g['w_pred'].mean():>9.2f}{g['w_real'].mean():>7.2f}{dw.mean():>+7.2f} ±{2 * se_w:.2f}{mark(dw.mean(), se_w)} │"
              f"{(ds - dw).mean():>+9.2f}")
    top = S["bin"] == S["bin"].max()
    ds = (S["s_real"] - S["s_pred"])[top]; dw = (S["w_real"] - S["w_pred"])[top]
    print(f"  → 전력 차 제일 큰 칸: 강팀 {ds.mean():+.2f}골 (±{2 * ds.std() / np.sqrt(top.sum()):.2f}), "
          f"약팀 {dw.mean():+.2f}골 (±{2 * dw.std() / np.sqrt(top.sum()):.2f})")
    print("  차이 = 실제 − 예측. 강팀 + 면 강팀을 덜 쳐준 것, 약팀 + 면 약팀이 예측보다 더 넣은 것. ← = 운으로 보기엔 큼\n")


for pre, lab in (("live", "살아있는 (지금 방식)"), ("fix", "고정 (시즌 시작 값)")):
    table(D, pre, lab + " · 전체")
for s, g in D.groupby("season"):
    table(g.reset_index(drop=True), "live", f"살아있는 · {s}/{(int(s) + 1) % 100} 시즌만")
