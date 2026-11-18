"""
1위 스코어 고르는 놈 학습 (11/18, 1순위 목표)
  지금: 시뮬 분포에서 확률 제일 높은 칸을 기계적으로 찍음
  이거: '실제 스코어가 1-1 / 1-0 / 0-1 / 2-1 …  중 뭐였나' 를 과거 경기로 직접 학습해서 찍음
    재료: 시뮬이 그 칸에 준 확률 + 칸마다 (리그, 전력 차, 예상 총 골, 홈이 강팀인지) 에 따른 보정
    → 같은 전력 차라도 라리가는 1-0 / 0-0 쪽, 분데스는 2-1 쪽 같은 차이를 경기마다 반영
  학습: shape_check.py 가 저장한 분포 중 앞 시즌들 / 채점: 25/26 ~ 26/27 (처음 보는 경기)
  덜 튀게 보정값에 벌점(λ)을 줌. λ 는 학습 시즌 중 마지막 시즌으로 고름

사용: py shape_check.py 2023 (시뮬, 20분) → py goals_model.py (1분) → py top1_learn.py (1~2분)
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

# 경기 특징 z: [1, 라리가, 분데스, 전력 차(홈−원정), |전력 차|, 예상 총 골] (+ 팀 성향)
season = D["season"].to_numpy()
z0 = np.column_stack([(D["league"] == "La_Liga").astype(float), (D["league"] == "Bundesliga").astype(float),
                      D["mh"] - D["ma"], (D["mh"] - D["ma"]).abs(), D["mh"] + D["ma"]])

# ② 골 모델 / ④ 팀 성향 (goals_model.py)
gm_f = DATA / "goals_model.json"
GM = json.loads(gm_f.read_text(encoding="utf-8")) if gm_f.exists() else None
if GM:
    mids = D["match_id"].astype(str)
    PD = np.array([GM[i]["dc"] if i in GM else list(P[k, :-1]) for k, i in enumerate(mids)])
    PD = np.column_stack([PD, np.clip(1 - PD.sum(1), 1e-4, 1)])
    logPD = np.log(np.clip(PD, 1e-4, 1))
    tend = np.array([[GM[i]["t_draw"], GM[i]["t_low"]] if i in GM else [0.0, 0.0] for i in mids])
    print(f"골 모델·팀 성향 붙음: {np.mean([i in GM for i in mids]):.0%} 경기")
else:
    print("data/goals_model.json 없음 → 시뮬만으로 학습 (py goals_model.py 먼저 돌리면 골 모델·팀 성향도 씀)")

# 선발 정보 (absence_test.py 가 저장한 A = 선발 xGChain ÷ 평소 베스트, 경기 직전 선발 발표를 안다고 가정)
ab_f = DATA / "absence_test.csv"
LU = None
if ab_f.exists():
    ab = pd.read_csv(ab_f)
    amap = {(int(r.match_id), r.team): float(r.A) for r in ab.itertuples()}
    Ah = np.array([amap.get((int(mid), h), 0.0) for mid, h in zip(D["match_id"], D["home"])])
    Aa = np.array([amap.get((int(mid), a_), 0.0) for mid, a_ in zip(D["match_id"], D["away"])])
    LU = np.column_stack([Ah, Aa, Ah - Aa])
    print(f"선발 정보 붙음: {np.mean([(int(mid), h) in amap for mid, h in zip(D['match_id'], D['home'])]):.0%} 경기")
else:
    print("data/absence_test.csv 없음 → 선발 정보 없이 (py absence_test.py 먼저 돌리면 씀)")


def zmat(extra=None):
    zz = z0 if extra is None else np.column_stack([z0, extra])
    zz = (zz - zz.mean(0)) / (zz.std(0) + 1e-9)
    return np.column_stack([np.ones(len(zz)), zz])


def fit(Ls, Z, mask, lam, iters=3000, lr=0.05):
    """로짓 = Σ w_i × log(확률_i) + B[칸] · z.  B 에 벌점 λ"""
    w = np.ones(len(Ls)) / len(Ls); B = np.zeros((K, Z.shape[1]))
    lp = [L_[mask] for L_ in Ls]; zz, yy = Z[mask], y[mask]; n = mask.sum()
    mw = np.zeros_like(w); vw = np.zeros_like(w); mB = np.zeros_like(B); vB = np.zeros_like(B)
    for t in range(1, iters + 1):
        L = sum(wi * li for wi, li in zip(w, lp)) + zz @ B.T
        L -= L.max(1, keepdims=True)
        Q = np.exp(L); Q /= Q.sum(1, keepdims=True)
        G = Q.copy(); G[np.arange(n), yy] -= 1
        gw = np.array([(G * li).sum() / n for li in lp])
        gB = G.T @ zz / n + lam * B
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw ** 2
        mB = 0.9 * mB + 0.1 * gB; vB = 0.999 * vB + 0.001 * gB ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        B -= lr * (mB / (1 - 0.9 ** t)) / (np.sqrt(vB / (1 - 0.999 ** t)) + 1e-8)
    return w, B


def predict(Ls, Z, w, B, mask):
    L = sum(wi * li[mask] for wi, li in zip(w, Ls)) + Z[mask] @ B.T
    L -= L.max(1, keepdims=True)
    Q = np.exp(L); return Q / Q.sum(1, keepdims=True)


def top1_hit(Q, mask):
    pick = Q[:, :-1].argmax(1)                      # '그 외' 는 못 찍으니 칸 중에서
    return pick == y[mask], pick


def zscore(d):
    return d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)


inner_tr = np.isin(season, TRAIN[:-1]); inner_va = season == TRAIN[-1]
tr = np.isin(season, TRAIN); te = np.isin(season, TEST)
h0, pick0 = top1_hit(P[te], te)
ll0 = -np.log(np.clip(P[te][np.arange(te.sum()), y[te]], 1e-4, 1))

VARS = [("시뮬 + 보정 (11/18 첫 버전)", [logP], zmat())]
if GM:
    VARS += [("시뮬 + 골 모델", [logP, logPD], zmat()),
             ("시뮬 + 골 모델 + 팀 성향", [logP, logPD], zmat(tend))]
if LU is not None:
    VARS += [("시뮬 + 선발", [logP], zmat(LU))]
    if GM:
        VARS += [("시뮬 + 골 모델 + 선발", [logP, logPD], zmat(LU))]
print(f"\n===== 1위 적중: {TRAIN} 로 학습 → {TEST} {te.sum()}경기 채점 (λ 는 {TRAIN[:-1]}→{TRAIN[-1]} 로 고름) =====")
print(f"  {'':<28}{'1위 적중':>9}{'확실도':>8}{'스코어 손실':>11}{'확실도':>8}   λ     믿는 정도 (시뮬, 골 모델)")
print(f"  {'지금 방식 (시뮬 최대 칸)':<26}{h0.mean():>9.1%}{'':>8}{ll0.mean():>11.4f}")
if GM:
    hd, _ = top1_hit(PD[te], te)
    print(f"  {'참고: 골 모델만 (최대 칸)':<26}{hd.mean():>9.1%}{zscore(hd.astype(float) - h0):>+8.1f}")
results = {}
for name, Ls, Z in VARS:
    best = None
    for lam in (3.0, 1.0, 0.3, 0.1, 0.03):
        w, B = fit(Ls, Z, inner_tr, lam, iters=1500)
        Qv = predict(Ls, Z, w, B, inner_va)
        llv = -np.log(Qv[np.arange(inner_va.sum()), y[inner_va]]).mean()
        if best is None or llv < best[1]:
            best = (lam, llv)
    w, B = fit(Ls, Z, tr, best[0])
    Q = predict(Ls, Z, w, B, te)
    h1, pick1 = top1_hit(Q, te)
    ll1 = -np.log(Q[np.arange(te.sum()), y[te]])
    results[name] = (h1, pick1)
    print(f"  {name:<26}{h1.mean():>9.1%}{zscore(h1.astype(float) - h0):>+8.1f}{ll1.mean():>11.4f}{zscore(ll0 - ll1):>+8.1f}   {best[0]:<5} {', '.join(f'{x:.2f}' for x in w)}")
print("  확실도: 지금 방식 대비. +2 이상이면 맞다 (스코어 손실은 낮을수록 좋음)")

name_best = max(results, key=lambda k: results[k][0].mean())
h1, pick1 = results[name_best]
nm = lambda k: f"{CLS[k][0]}-{CLS[k][1]}"
print(f"\n  [1위로 찍은 스코어] 지금 → {name_best}")
c0 = pd.Series([nm(k) for k in pick0]).value_counts(); c1 = pd.Series([nm(k) for k in pick1]).value_counts()
for sc in sorted(set(c0.index) | set(c1.index), key=lambda x: -(c0.get(x, 0) + c1.get(x, 0)))[:8]:
    g0 = pick0 == CLS.index(tuple(map(int, sc.split("-")))); g1 = pick1 == CLS.index(tuple(map(int, sc.split("-"))))
    print(f"    {sc:<5} {c0.get(sc, 0):>5}번 (적중 {h0[g0].mean() if g0.any() else 0:.1%}) → {c1.get(sc, 0):>5}번 (적중 {h1[g1].mean() if g1.any() else 0:.1%})")
print(f"\n  [리그별 1위 적중] 지금 → {name_best}")
lg = D.loc[te, "league"].to_numpy()
for L_ in ("EPL", "La_Liga", "Bundesliga"):
    mk = lg == L_
    print(f"    {L_:<11} {mk.sum():>4}경기  {h0[mk].mean():.1%} → {h1[mk].mean():.1%}")
