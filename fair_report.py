"""
공정한 채점 결과 보기 (fair_eval.py 가 만든 data/fair/season_*.csv)
  1. 1위 스코어 적중: 엔진별 + '매번 1-1 찍기' + '1-1 기본 규칙'(그 전 시즌들로 문턱 찾음)
  2. 스코어 손실 (실제 스코어에 준 확률), 골 오차
  3. 핸디: 강팀 −1 (강팀 2골차+), 강팀 −2 (3골차+), 약팀 +1 (약팀 지거나 비기거나… = 강팀 2골차 미만) 확률이 실제랑 맞나
  4. 언오버 2.5
  5. 압도 경기 강팀 골
  확실도: 지금 엔진(new) − 비교 대상, 같은 경기끼리. +2 이상이면 확실히 나음

사용: py fair_report.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "fair"
MAXG = 6
files = sorted(DATA.glob("season_*.csv"))
if not files:
    raise SystemExit("data/fair/ 에 결과가 없어. 먼저 py fair_eval.py 2023 2024")
D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
VAR = [v for v in ("old", "new", "new45") if f"{v}_mat" in D]
NAME = {"old": "예전 (마무리 끔, c1 0.045)", "new": "지금 (마무리 켬, c1 0.02)", "new45": "마무리 켬, c1 0.045"}
M = {v: np.array([json.loads(x) for x in D[f"{v}_mat"]]) for v in VAR}
gh, ga = D["gh"].clip(upper=MAXG).to_numpy(), D["ga"].clip(upper=MAXG).to_numpy()
idx = np.arange(len(D))
seasons = sorted(D["season"].unique())
n = len(D)
print(f"채점 경기 {n}개 ({', '.join(f'{s}/{(s + 1) % 100}' for s in seasons)}) — 전부 그 시즌 전 데이터로만 예측\n")


def z(d):
    d = np.asarray(d, float)
    return d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)


def ci(r, k):
    se = np.sqrt(r * (1 - r) / k)
    return f"{r:.1%} ({max(0, r - 2 * se):.1%}~{r + 2 * se:.1%})"


def top1(Mx):
    t = Mx.reshape(len(Mx), -1).argmax(1)
    return t // (MAXG + 1), t % (MAXG + 1)


# ---------- 1. 1위 적중 ----------
hit = {}
for v in VAR:
    i, j = top1(M[v]); hit[v] = (i == gh) & (j == ga)
hit["one"] = (gh == 1) & (ga == 1)
# 1-1 기본 규칙: 1-1 확률 × d 가 제일 크면 1-1. d 는 그 전 시즌들로 찾음 (첫 시즌은 d=1)
rule = np.zeros(n, bool); d_used = {}
for s in seasons:
    prev = (D["season"] < s).to_numpy(); cur = (D["season"] == s).to_numpy()
    best_d = 1.0
    if prev.any():
        def hits(dd, mask):
            A = M["new"][mask].copy(); A[:, 1, 1] *= dd
            i, j = top1(A); return ((i == gh[mask]) & (j == ga[mask])).mean()
        best_d = max([1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0], key=lambda dd: hits(dd, prev))
    A = M["new"][cur].copy(); A[:, 1, 1] *= best_d
    i, j = top1(A); rule[cur] = (i == gh[cur]) & (j == ga[cur]); d_used[s] = best_d
hit["rule"] = rule
print("===== 1. 1위 스코어 적중 =====")
print(f"  {'':<30}{'적중 (오차범위)':>22}{'지금 엔진 − 이것':>16}{'확실도':>8}")
for k, lab in [(v, NAME[v]) for v in VAR] + [("one", "매번 1-1 찍기"), ("rule", "1-1 기본 규칙 (지금 엔진)")]:
    dd = hit["new"].astype(float) - hit[k]
    print(f"  {lab:<28}{ci(hit[k].mean(), n):>22}{dd.mean():>+15.1%}{(z(dd) if k != 'new' else 0):>+8.1f}")
print(f"  1-1 기본 규칙 문턱 d (그 전 시즌으로 찾음): {', '.join(f'{s}/{(s + 1) % 100} ×{d_used[s]}' for s in seasons)}")
print("  시즌별:")
for s in seasons:
    mk = (D["season"] == s).to_numpy()
    print(f"    {s}/{(s + 1) % 100} {mk.sum():>5}경기  " + "  ".join(f"{k} {hit[k][mk].mean():.1%}" for k in VAR + ["one", "rule"]))

# ---------- 2. 스코어 손실 / 골 오차 ----------
print("\n===== 2. 스코어 손실 (실제 스코어에 준 확률) / 골 오차 =====")
ll = {v: -np.log(np.clip(M[v][idx, gh, ga], 1e-4, 1)) for v in VAR}
dev = {}
for v in VAR:
    mh, ma = D[f"{v}_mh"].to_numpy(), D[f"{v}_ma"].to_numpy()
    f = lambda y, mu: 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - (y - mu))
    dev[v] = f(D["gh"].to_numpy(), mh) + f(D["ga"].to_numpy(), ma)
for v in VAR:
    print(f"  {NAME[v]:<28} 스코어 손실 {ll[v].mean():.4f}  골 오차 {dev[v].mean():.4f}"
          + ("" if v == "new" else f"   지금 엔진이 나은 정도: 손실 {z(ll[v] - ll['new']):+.1f}, 골 {z(dev[v] - dev['new']):+.1f}"))

# ---------- 3. 핸디 ----------
print("\n===== 3. 핸디 (지금 엔진 기준 강팀 = 예상 평균 골 높은 쪽) =====")
fav_home = (D["new_mh"] >= D["new_ma"]).to_numpy()
gd_fav = np.where(fav_home, D["gh"] - D["ga"], D["ga"] - D["gh"])
gdm = np.subtract.outer(np.arange(MAXG + 1), np.arange(MAXG + 1))
for k, lab in ((2, "강팀 −1 (2골차+)"), (3, "강팀 −2 (3골차+)")):
    y = (gd_fav >= k).astype(float)
    print(f"  [{lab}] 실제 비율 {y.mean():.1%}")
    for v in VAR:
        Mf = np.where(fav_home[:, None, None], M[v], M[v].transpose(0, 2, 1))
        p = Mf[:, gdm >= k].sum(1)
        b = (p - y) ** 2
        line = f"    {NAME[v]:<28} 예측 평균 {p.mean():.1%}  오차(브라이어) {b.mean():.4f}"
        if v == "new":
            bins = pd.cut(p, [0, .1, .2, .3, .4, .5, .6, 1])
            cal = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
            line += "\n      구간별 (예측 → 실제): " + ", ".join(f"{r.p:.0%}→{r.y:.0%}({r.n})" for r in cal.itertuples() if r.n >= 20)
        else:
            pn = np.where(fav_home[:, None, None], M["new"], M["new"].transpose(0, 2, 1))[:, gdm >= k].sum(1)
            line += f"   지금 엔진이 나은 정도 {z(b - (pn - y) ** 2):+.1f}"
        print(line)

# ---------- 4. 언오버 ----------
print("\n===== 4. 언오버 2.5 =====")
tot = np.add.outer(np.arange(MAXG + 1), np.arange(MAXG + 1))
y = ((D["gh"] + D["ga"]) > 2.5).astype(float).to_numpy()
pn = M["new"][:, tot > 2.5].sum(1)
for v in VAR:
    p = M[v][:, tot > 2.5].sum(1)
    hit_ou = ((p > 0.5) == (y == 1)).mean()
    print(f"  {NAME[v]:<28} 적중 {hit_ou:.1%}  예측 평균 {p.mean():.1%} (실제 {y.mean():.1%})  오차 {((p - y) ** 2).mean():.4f}"
          + ("" if v == "new" else f"  지금 엔진이 나은 정도 {z((p - y) ** 2 - (pn - y) ** 2):+.1f}"))
print(f"  매번 {'오버' if y.mean() > 0.5 else '언더'} 찍기: {max(y.mean(), 1 - y.mean()):.1%}")

# ---------- 5. 압도 경기 강팀 골 ----------
print("\n===== 5. 전력 차 구간별 강팀 골 (실제 / 엔진별 예측) =====")
gap = (D["new_mh"] - D["new_ma"]).abs().to_numpy()
sg = np.where(fav_home, D["gh"], D["ga"])
for lo, hi, lab in ((0, 0.6, "0~0.6"), (0.6, 1.0, "0.6~1.0"), (1.0, 1.5, "1.0~1.5"), (1.5, 9, "1.5~ 압도")):
    mk = (gap >= lo) & (gap < hi)
    print(f"  {lab:<10}{mk.sum():>5}경기  실제 {sg[mk].mean():.2f}  " +
          "  ".join(f"{v} {np.where(fav_home, D[f'{v}_mh'], D[f'{v}_ma'])[mk].mean():.2f}" for v in VAR))
