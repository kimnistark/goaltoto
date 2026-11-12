"""
④ 강팀 마무리 (11/12)
  진단: 압도 경기 강팀은 찬스(xG)는 예상대로인데 골 ÷ xG 가 0.98 (리그 평균 0.85~0.89), 약팀은 0.71
  → 오픈플레이 마무리 배수 = (내 공격력)^a × (상대 수비 허용)^b   (공격력 = 슈팅 수 × 슈팅 질 팀 값, 경기 전 값)
     a, b 두 개만 21/22 ~ 24/25 로 찾음 (팀마다 따로 숫자 안 매김 → 운에 덜 휘둘림)
  → 시뮬에서 그 팀 슈팅 질(att_q)에 곱해서 25/26 ~ 26/27 채점 (매주 그 주 전 정보, 팀만, 경기당 3000번)

사용: py finish_test.py    (a, b 찾기 1~2분 + 시뮬 15~25분)
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 3000
MAXG = 6
DATA = Path(__file__).parent / "data"
m, T = L.load_team_matches()
latest = int(T["season"].max())
TEST = [latest - 1, latest]

# ---------- 1. a, b 찾기 ----------
saved = L.load_params()
p0 = saved if saved else L.fit(T, latest - 1, log=lambda *a, **k: None)
vals = {}
def grab(rows, R):
    for r in rows:
        v, o = R.team_values(r.team), R.team_values(r.opp)
        vals[(r.match_id, r.team)] = (v["att_open"] * v["att_q"], o["def_open"] * o["def_q"])
L.run(T, p0["alpha"], p0["carry"], p0["const"], int(T["season"].min()), grab)
T["A"] = [np.log(vals[(a, b)][0]) for a, b in zip(T["match_id"], T["team"])]
T["Dp"] = [np.log(vals[(a, b)][1]) for a, b in zip(T["match_id"], T["team"])]
tr = T[(T["season"] < min(TEST)) & (T["xg_open"] > 0)]
mA, mD = tr["A"].mean(), tr["Dp"].mean()

# 포아송 회귀: 오픈플레이 골 ~ xG × exp(b0 + a(A−평균) + b(D−평균))
X = np.column_stack([np.ones(len(tr)), tr["A"] - mA, tr["Dp"] - mD]); y = tr["g_open"].to_numpy(); off = tr["xg_open"].to_numpy()
beta = np.array([np.log(y.sum() / off.sum()), 0.0, 0.0])
for _ in range(25):
    mu = off * np.exp(X @ beta)
    H = X.T @ (X * mu[:, None]); g = X.T @ (y - mu)
    beta += np.linalg.solve(H, g)
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (off * np.exp(X @ beta))[:, None]))))
b0, a, b = beta
print(f"===== 마무리 공식 (21/22~{min(TEST) - 1}/{min(TEST) % 100} 팀-경기 {len(tr)}개) =====")
print(f"  평균 골 ÷ xG = {np.exp(b0):.3f}")
print(f"  a (내 공격력)     = {a:+.3f}  (±{2 * se[1]:.3f})   + 면 강팀일수록 xG보다 잘 넣음")
print(f"  b (상대 수비 허용) = {b:+.3f}  (±{2 * se[2]:.3f})   + 면 수비 구린 팀 상대로 잘 넣음 (골키퍼 포함)")
for lab, A_, D_ in (("강팀 vs 약팀", 0.5, 0.35), ("평균 vs 평균", 0, 0), ("약팀 vs 강팀", -0.4, -0.35)):
    print(f"    예) {lab}: 마무리 배수 ×{np.exp(a * A_ + b * D_):.2f}")


def fmult(Alog, Dlog):
    return float(np.exp(a * (Alog - mA) + b * (Dlog - mD)))


# ---------- 2. 시뮬 채점 ----------
rows = []
for test in TEST:
    saved = L.load_params()
    p = saved if (saved and saved.get("test_season") == test) else None
    if p is None:
        print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
        p = L.fit(T, test, log=lambda *a_, **k: None)
        p["test_season"] = test
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= test]
    S = {"week": None, "Rw": None}

    def before(rs, R):
        r0 = rs[0]
        if r0.season != test:
            return
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
        for r in rs:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rs if r.home); a_ = next(r for r in rs if not r.home)
        v = {t: S["Rw"].team_values(t) for t in (h.team, a_.team)}
        gh, ga = int(h.gf), int(a_.gf)
        row = {"season": test, "gh": gh, "ga": ga}
        for k in ("off", "fin"):
            for t, o in ((h.team, a_.team), (a_.team, h.team)):
                vv = dict(v[t])
                if k == "fin":
                    vv["att_q"] *= fmult(np.log(v[t]["att_open"] * v[t]["att_q"]), np.log(v[o]["def_open"] * v[o]["def_q"]))
                E.C["teams"][t] = vv
                E.P["teams"][t] = {"att_shots": vv["att_open"], "def_shots": vv["def_open"], "att_q": vv["att_q"], "def_q": vv["def_q"]}
            sh, sa, _ = E.simulate(h.team, a_.team, N, use_players=False)
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            mat /= N
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            row.update({f"{k}_mh": sh.mean(), f"{k}_ma": sa.mean(), f"{k}_hit": (i, j) == (min(gh, MAXG), min(ga, MAXG)),
                        f"{k}_ll": -np.log(max(ps, 1e-4)), f"{k}_dev": L.poisson_dev(gh, sh.mean()) + L.poisson_dev(ga, sa.mean()),
                        f"{k}_top": f"{i}-{j}"})
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"    {len(rows)}경기")

    print(f"===== [{test}/{(test + 1) % 100}] 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)

D = pd.DataFrame(rows); n = len(D)
D["gap"] = (D["off_mh"] - D["off_ma"]).abs()
hf = D["off_mh"] >= D["off_ma"]
D["s_g"] = np.where(hf, D["gh"], D["ga"]); D["w_g"] = np.where(hf, D["ga"], D["gh"])
for k in ("off", "fin"):
    D[f"{k}_s"] = np.where(hf, D[f"{k}_mh"], D[f"{k}_ma"]); D[f"{k}_w"] = np.where(hf, D[f"{k}_ma"], D[f"{k}_mh"])
print(f"\n===== 전력 차 구간별 강팀·약팀 골 ({n}경기): 실제 / 지금 / 마무리 넣음 =====")
for lo, hi, lab in ((0, 0.6, "0~0.6"), (0.6, 1.0, "0.6~1.0"), (1.0, 1.5, "1.0~1.5"), (1.5, 9, "1.5~ 압도")):
    g = D[(D["gap"] >= lo) & (D["gap"] < hi)]
    print(f"  {lab:<10}{len(g):>5}경기   강팀 {g['s_g'].mean():.2f} / {g['off_s'].mean():.2f} / {g['fin_s'].mean():.2f}"
          f"     약팀 {g['w_g'].mean():.2f} / {g['off_w'].mean():.2f} / {g['fin_w'].mean():.2f}")

print(f"\n===== 채점 =====")
print(f"  {'':<16}{'지금':>10}{'마무리 넣음':>12}{'확실도':>8}")
for col, lab, hb in (("hit", "1위 적중률", True), ("dev", "골 오차", False), ("ll", "스코어 손실", False)):
    x0, x1 = D[f"off_{col}"].astype(float), D[f"fin_{col}"].astype(float)
    d = (x1 - x0) if hb else (x0 - x1)
    z = d.mean() / (d.std() / np.sqrt(n) + 1e-12)
    fmt = (lambda x: f"{x:.1%}") if col == "hit" else (lambda x: f"{x:.4f}")
    print(f"  {lab:<14}{fmt(x0.mean()):>10}{fmt(x1.mean()):>12}{z:>+8.1f}")
big = D["gap"] >= 1.5
h0, h1 = D.loc[big, "off_hit"].mean(), D.loc[big, "fin_hit"].mean()
print(f"  압도 경기만 1위 적중  {h0:.1%} → {h1:.1%}   (1위로 찍은 것: {D.loc[big, 'fin_top'].value_counts().head(3).to_dict()})")
print("  확실도 +2 이상이면 맞다 (골 오차·스코어 손실은 낮을수록 좋음)")
(DATA / "finish.json").write_text(json.dumps({"a": a, "b": b, "mA": mA, "mD": mD}), encoding="utf-8")
print("\n저장: data/finish.json (마무리 공식 숫자)")
