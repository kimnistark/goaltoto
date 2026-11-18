"""
1위 스코어 고르는 놈 학습 (11/18, 1순위 목표)
  지금: 시뮬 분포에서 확률 제일 높은 칸을 기계적으로 찍음
  이거: '실제 스코어가 1-1 / 1-0 / 0-1 / 2-1 …  중 뭐였나' 를 과거 경기로 직접 학습해서 찍음
    재료: 시뮬이 그 칸에 준 확률 + 칸마다 (리그, 전력 차, 예상 총 골, 홈이 강팀인지) 에 따른 보정
    → 같은 전력 차라도 라리가는 1-0 / 0-0 쪽, 분데스는 2-1 쪽 같은 차이를 경기마다 반영
  학습: shape_check.py 가 저장한 분포 중 앞 시즌들 / 채점: 25/26 ~ 26/27 (처음 보는 경기)
  덜 튀게 보정값에 벌점(λ)을 줌. λ 는 학습 시즌 중 마지막 시즌으로 고름

사용: py shape_check.py 2023 (시뮬, 20분) → py top1_learn.py (몇 초)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MAXG = 6
f = DATA / "shape_preds.json"
if not f.exists():
    raise SystemExit("data/shape_preds.json 이 없어. 먼저 py shape_check.py 2023")
D = pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
M = np.array([np.array(x) for x in D["mat"]])
seasons = sorted(int(x) for x in D["season"].unique())
latest = max(seasons)
TEST = [latest - 1, latest]
TRAIN = [s for s in seasons if s < latest - 1]
if len(TRAIN) < 2:
    raise SystemExit(f"학습 시즌이 {TRAIN} 뿐이라 부족함. py shape_check.py 2023 로 앞 시즌까지 돌려야 함")

CLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3)]
K = len(CLS) + 1                                     # 마지막 = 그 외
P = np.stack([M[:, i, j] for i, j in CLS], 1)
P = np.column_stack([P, np.clip(1 - P.sum(1), 1e-4, 1)])
logP = np.log(np.clip(P, 1e-4, 1))
y = np.full(len(D), K - 1)
for k, (i, j) in enumerate(CLS):
    y[(D["gh"] == i).to_numpy() & (D["ga"] == j).to_numpy()] = k

# 경기 특징 z: [1, 라리가, 분데스, 전력 차(홈−원정), |전력 차|, 예상 총 골]
z = np.column_stack([np.ones(len(D)), (D["league"] == "La_Liga").astype(float), (D["league"] == "Bundesliga").astype(float),
                     D["mh"] - D["ma"], (D["mh"] - D["ma"]).abs(), D["mh"] + D["ma"]])
zm, zs = z[:, 1:].mean(0), z[:, 1:].std(0) + 1e-9
z[:, 1:] = (z[:, 1:] - zm) / zs
F = z.shape[1]
season = D["season"].to_numpy()


def fit(mask, lam, iters=3000, lr=0.05):
    """로짓 = w × log(시뮬 확률) + B[칸] · z.  B 에 벌점 λ"""
    w = 1.0; B = np.zeros((K, F))
    lp, zz, yy = logP[mask], z[mask], y[mask]
    n = mask.sum()
    mw = vw = 0.0; mB = np.zeros_like(B); vB = np.zeros_like(B)
    for t in range(1, iters + 1):
        L = w * lp + zz @ B.T
        L -= L.max(1, keepdims=True)
        Q = np.exp(L); Q /= Q.sum(1, keepdims=True)
        G = Q.copy(); G[np.arange(n), yy] -= 1                       # d(손실)/d(로짓)
        gw = (G * lp).sum() / n
        gB = G.T @ zz / n + lam * B
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw ** 2
        mB = 0.9 * mB + 0.1 * gB; vB = 0.999 * vB + 0.001 * gB ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        B -= lr * (mB / (1 - 0.9 ** t)) / (np.sqrt(vB / (1 - 0.999 ** t)) + 1e-8)
    return w, B


def predict(w, B, mask):
    L = w * logP[mask] + z[mask] @ B.T
    L -= L.max(1, keepdims=True)
    Q = np.exp(L); return Q / Q.sum(1, keepdims=True)


def top1_hit(Q, mask):
    pick = Q[:, :-1].argmax(1)                      # '그 외' 는 못 찍으니 칸 중에서
    return pick == y[mask], pick


# λ 고르기: 학습 시즌 중 마지막 시즌을 떼서
inner_tr = np.isin(season, TRAIN[:-1]); inner_va = season == TRAIN[-1]
base_va, _ = top1_hit(P[inner_va], inner_va)
print(f"λ 고르기: {TRAIN[:-1]} 로 학습 → {TRAIN[-1]} 로 확인 (지금 방식 1위 적중 {base_va.mean():.1%})")
best = None
for lam in (3.0, 1.0, 0.3, 0.1, 0.03, 0.01):
    w, B = fit(inner_tr, lam)
    h, _ = top1_hit(predict(w, B, inner_va), inner_va)
    ll = -np.log(predict(w, B, inner_va)[np.arange(inner_va.sum()), y[inner_va]]).mean()
    print(f"  λ {lam:<6} 1위 적중 {h.mean():.1%}   손실 {ll:.4f}")
    if best is None or ll < best[1]:
        best = (lam, ll)
lam = best[0]
print(f"  → λ {lam} (손실 기준)\n")

tr = np.isin(season, TRAIN); te = np.isin(season, TEST)
w, B = fit(tr, lam)
Q = predict(w, B, te)
h1, pick1 = top1_hit(Q, te)
h0, pick0 = top1_hit(P[te], te)
d = h1.astype(float) - h0
zsc = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
ll0 = -np.log(np.clip(P[te][np.arange(te.sum()), y[te]], 1e-4, 1)); ll1 = -np.log(Q[np.arange(te.sum()), y[te]])
dl = ll0 - ll1; zl = dl.mean() / (dl.std() / np.sqrt(len(dl)) + 1e-12)
print(f"===== 1위 적중: {TRAIN} 로 학습 → {TEST} {te.sum()}경기 채점 =====")
print(f"  지금 방식 (시뮬 확률 최대 칸)   {h0.mean():.1%}")
print(f"  학습한 놈                     {h1.mean():.1%}   확실도 {zsc:+.1f}   (스코어 손실 {ll0.mean():.4f} → {ll1.mean():.4f}, 확실도 {zl:+.1f})")
print(f"  w (시뮬 확률 믿는 정도) = {w:.2f}   (1 이면 시뮬 그대로, 1보다 크면 시뮬을 더 믿음)")
print(f"  1위 바뀐 경기 {np.mean(pick1 != pick0):.0%}")

name = lambda k: f"{CLS[k][0]}-{CLS[k][1]}"
print("\n  [1위로 찍은 스코어] 지금 → 학습한 놈")
c0 = pd.Series([name(k) for k in pick0]).value_counts(); c1 = pd.Series([name(k) for k in pick1]).value_counts()
for s in sorted(set(c0.index) | set(c1.index), key=lambda s: -(c0.get(s, 0) + c1.get(s, 0)))[:8]:
    g1 = pick1 == CLS.index(tuple(map(int, s.split("-"))))
    print(f"    {s:<5} {c0.get(s, 0):>5}번 → {c1.get(s, 0):>5}번   (학습한 놈이 찍었을 때 적중 {h1[g1].mean() if g1.any() else 0:.1%})")
print("\n  [리그별 1위 적중] 지금 → 학습한 놈")
lg = D.loc[te, "league"].to_numpy()
for L_ in ("EPL", "La_Liga", "Bundesliga"):
    mk = lg == L_
    print(f"    {L_:<11} {mk.sum():>4}경기  {h0[mk].mean():.1%} → {h1[mk].mean():.1%}")
print("\n  확실도 +2 이상이면 맞다")
