"""
1위 적중률 + 강팀 골 (11/12, 1순위 목표 두 개)
  shape_check.py 가 저장한 경기별 스코어 분포(data/shape_preds.json)를 그대로 씀 → 시뮬 안 돌림, 몇 초
  강팀 기준으로 스코어 칸을 뒤집어서 봄 (강팀 골 - 약팀 골). 전력 차(평균 골 차이) 구간별로
    1) 진단: 칸마다 시뮬 확률 vs 실제 비율, 강팀 평균 골 시뮬 vs 실제
    2) 보정표: 24/25 로 '구간 × 칸' 마다 실제 ÷ 시뮬 배수를 찾음 (경기 수 적은 칸은 1 쪽으로 당김)
       → 25/26 ~ 26/27 분포에 곱하고 다시 1위 스코어 뽑기 (처음 보는 경기로 채점)
    3) 강팀 골 보정: 강팀 평균 골에 + c × (예상 골차) 를 더하는 식으로 분포를 옮김 (c 를 24/25 로 찾음)

사용: py shape_check.py 를 한 번 돌렸으면 → py top1_tune.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MAXG = 6
K = 15          # 보정표: 칸마다 '15경기 분량'만큼 1 쪽으로 당김 (드문 칸이 튀지 않게)
f = DATA / "shape_preds.json"
if not f.exists():
    raise SystemExit("data/shape_preds.json 이 없어. 먼저 py shape_check.py")
rows = json.loads(f.read_text(encoding="utf-8"))
D = pd.DataFrame(rows)
M = np.array([np.array(x) for x in D["mat"]])
seasons = sorted(int(x) for x in D["season"].unique())
TRAIN, TEST = seasons[:1], seasons[1:]
tr = D["season"].isin(TRAIN).to_numpy(); te = D["season"].isin(TEST).to_numpy()

# 강팀 기준으로 뒤집기: F[경기, 강팀 골, 약팀 골]
home_fav = (D["mh"] >= D["ma"]).to_numpy()
F = np.where(home_fav[:, None, None], M, M.transpose(0, 2, 1))
fg = np.where(home_fav, D["gh"], D["ga"]).clip(max=MAXG); ug = np.where(home_fav, D["ga"], D["gh"]).clip(max=MAXG)
gap = (D["mh"] - D["ma"]).abs().to_numpy()
EDGES = [0, 0.3, 0.6, 1.0, 1.5, 9]
LAB = ["0~0.3 (비슷)", "0.3~0.6", "0.6~1.0", "1.0~1.5", "1.5~ (압도)"]
gb = np.digitize(gap, EDGES[1:-1])


def top1(P):
    t = P.reshape(len(P), -1).argmax(1)
    return t // (MAXG + 1), t % (MAXG + 1)


def evaluate(P, mask):
    i, j = top1(P[mask])
    hit = (i == fg[mask]) & (j == ug[mask])
    p = P[mask][np.arange(mask.sum()), fg[mask], ug[mask]]
    return hit, -np.log(np.maximum(p, 1e-4))


# ---------- 1. 진단 ----------
print(f"===== 진단 ({TEST} {te.sum()}경기, 강팀 기준) =====")
print(f"  {'전력 차':<14}{'경기':>5}{'강팀 골 시뮬':>11}{'실제':>7}{'약팀 골 시뮬':>11}{'실제':>7}{'1위 적중':>9}  가장 많이 찍은 1위")
P0 = F
for b in range(len(LAB)):
    mk = te & (gb == b)
    if mk.sum() < 10:
        continue
    efg = (P0[mk].sum(2) * np.arange(MAXG + 1)).sum(1).mean(); eug = (P0[mk].sum(1) * np.arange(MAXG + 1)).sum(1).mean()
    hit, _ = evaluate(P0, mk)
    i, j = top1(P0[mk])
    picks = pd.Series([f"{a}-{c}" for a, c in zip(i, j)]).value_counts().head(2)
    print(f"  {LAB[b]:<14}{mk.sum():>5}{efg:>11.2f}{fg[mk].mean():>7.2f}{eug:>11.2f}{ug[mk].mean():>7.2f}{hit.mean():>9.1%}  "
          + ", ".join(f"{k} {v}번" for k, v in picks.items()))

print(f"\n  [칸별] 시뮬 확률 평균 → 실제 비율 (강팀 골-약팀 골)")
cells = [(1, 1), (1, 0), (2, 0), (2, 1), (0, 0), (0, 1), (3, 0), (3, 1), (2, 2), (1, 2)]
print(f"  {'전력 차':<14}" + "".join(f"{f'{a}-{c}':>13}" for a, c in cells))
for b in range(len(LAB)):
    mk = te & (gb == b)
    if mk.sum() < 10:
        continue
    print(f"  {LAB[b]:<14}" + "".join(f"{P0[mk][:, a, c].mean():>6.1%}→{((fg[mk] == a) & (ug[mk] == c)).mean():<6.1%}" for a, c in cells))

# ---------- 2. 보정표 ----------
W = np.ones((len(LAB), MAXG + 1, MAXG + 1))
for b in range(len(LAB)):
    mk = tr & (gb == b)
    A = np.zeros((MAXG + 1, MAXG + 1)); np.add.at(A, (fg[mk], ug[mk]), 1)
    Pp = F[mk].sum(0)
    W[b] = np.clip((A + K) / (Pp + K), 0.7, 1.4)          # 실제 ÷ 시뮬, 드문 칸은 1 근처, 한 칸 최대 ±30~40%


def apply_w(P, w=W):
    Q = P * w[gb]
    return Q / Q.sum((1, 2), keepdims=True)


# ---------- 3. 강팀 골 보정 ----------
def shift(P, c):
    """강팀 골 분포를 c × 예상 골차 만큼 오른쪽으로 (강팀 한 골 더 확률을 섞음)"""
    if c == 0:
        return P
    a = np.clip(c * gap, 0, 0.9)[:, None, None]
    S = np.zeros_like(P)
    S[:, 1:, :] = P[:, :-1, :]; S[:, -1, :] += P[:, -1, :]
    return (1 - a) * P + a * S


hit0, ll0 = evaluate(F, te)
grid_c = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
best_c = min(grid_c, key=lambda c: evaluate(shift(F, c), tr)[1].mean())
best_c2 = min(grid_c, key=lambda c: evaluate(apply_w(shift(F, c)), tr)[1].mean())
print(f"\n===== 고치기: {TRAIN} 로 찾고 {TEST} {te.sum()}경기로 채점 =====")
print(f"  {'':<28}{'1위 적중':>9}{'확실도':>8}{'스코어 손실':>11}{'확실도':>8}{'강팀 골 시뮬/실제':>16}")
for lab, P in (("지금 그대로", F), (f"강팀 골 보정 c={best_c}", shift(F, best_c)), ("칸 보정표", apply_w(F)),
               (f"둘 다 (c={best_c2})", apply_w(shift(F, best_c2)))):
    hit, ll = evaluate(P, te)
    dh = hit.astype(float) - hit0; dl = ll0 - ll
    zh = dh.mean() / (dh.std() / np.sqrt(len(dh)) + 1e-12) if lab != "지금 그대로" else 0
    zl = dl.mean() / (dl.std() / np.sqrt(len(dl)) + 1e-12) if lab != "지금 그대로" else 0
    big = te & (gb >= 3)
    efg = (P[big].sum(2) * np.arange(MAXG + 1)).sum(1).mean()
    print(f"  {lab:<26}{hit.mean():>9.1%}{zh:>+8.1f}{ll.mean():>11.4f}{zl:>+8.1f}{efg:>10.2f}/{fg[big].mean():.2f}")
print("  강팀 골 시뮬/실제 = 전력 차 1.0 넘는 경기에서 강팀 평균 골. 확실도 +2 이상이면 맞다")

P1 = apply_w(shift(F, best_c2))
print(f"\n  [둘 다 적용했을 때 전력 차 구간별 1위 적중]")
for b in range(len(LAB)):
    mk = te & (gb == b)
    if mk.sum() < 10:
        continue
    h0, _ = evaluate(F, mk); h1, _ = evaluate(P1, mk)
    i, j = top1(P1[mk])
    picks = pd.Series([f"{a}-{c}" for a, c in zip(i, j)]).value_counts().head(2)
    print(f"  {LAB[b]:<14}{mk.sum():>5}경기  {h0.mean():.1%} → {h1.mean():.1%}   1위로 찍은 것: " + ", ".join(f"{k} {v}번" for k, v in picks.items()))
