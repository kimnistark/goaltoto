"""
마무리 배수 넣었을 때 c1(전력 차 보정)을 다시 맞추기 (11/12)
  11/12 finish_test: 마무리 배수가 압도 경기 강팀 골을 2.74 → 2.91 (실제 2.94) 로 맞춤
  근데 c1=0.045 도 '강팀 골 더 주기' 라서 둘이 겹칠 수 있음 (0.6~1.0 구간 강팀 1.85 → 1.89, 실제 1.74)
  → 마무리 켠 상태에서 c1 = 0 / 0.02 / 0.045 비교. c1 은 24/25 로 고르고 25/26 ~ 26/27 로 채점
  비교 기준: 지금 엔진 (마무리 끔, c1 0.045)

사용: py finish_test.py 를 한 번 돌려서 data/finish.json 이 있어야 함 → py fin_c1_test.py  (20분 정도)
"""
import copy
import sys

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 2000
MAXG = 6
if E.FIN is None:
    sys.exit("data/finish.json 이 없어. 먼저 py finish_test.py")
m, T = L.load_team_matches()
latest = int(T["season"].max())
TRAIN, TEST = [latest - 2], [latest - 1, latest]
VAR = {"base": ("지금 (마무리 끔, c1 0.045)", False, 0.045), "f045": ("마무리 + c1 0.045", True, 0.045),
       "f02": ("마무리 + c1 0.02", True, 0.02), "f0": ("마무리 + c1 0", True, 0.0)}


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


rows = []
for test in TRAIN + TEST:
    p = season_params(test)
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
        h = next(r for r in rs if r.home); a = next(r for r in rs if not r.home)
        for t in (h.team, a.team):
            v = S["Rw"].team_values(t)
            E.C["teams"][t] = v
            E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
        gh, ga = int(h.gf), int(a.gf)
        row = {"season": test, "gh": gh, "ga": ga}
        for k, (_, fin, c1) in VAR.items():
            E.FIN_ON, E.CORR_C1 = fin, c1
            sh, sa, _ = E.simulate(h.team, a.team, N, use_players=False)
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            mat /= N
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            ps = mat[min(gh, MAXG), min(ga, MAXG)]
            row.update({f"{k}_mh": sh.mean(), f"{k}_ma": sa.mean(), f"{k}_hit": (i, j) == (min(gh, MAXG), min(ga, MAXG)),
                        f"{k}_ll": -np.log(max(ps, 1e-4)), f"{k}_dev": L.poisson_dev(gh, sh.mean()) + L.poisson_dev(ga, sa.mean())})
        E.FIN_ON, E.CORR_C1 = False, 0.045
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f"    {len(rows)}경기")

    print(f"===== [{test}/{(test + 1) % 100}] 예측 중 =====")
    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)

D = pd.DataFrame(rows)
tr, te = D[D["season"].isin(TRAIN)], D[D["season"].isin(TEST)]
pick = min(["f045", "f02", "f0"], key=lambda k: tr[f"{k}_dev"].mean())
print(f"\n  {TRAIN} 에서 골 오차 제일 작은 c1: {VAR[pick][0]}  "
      "(" + ", ".join(f"{VAR[k][0]} {tr[k + '_dev'].mean():.4f}" for k in ("f045", "f02", "f0")) + ")")

gap = (te["base_mh"] - te["base_ma"]).abs(); hf = te["base_mh"] >= te["base_ma"]
sg = np.where(hf, te["gh"], te["ga"])
print(f"\n===== 전력 차 구간별 강팀 골 ({TEST} {len(te)}경기) =====")
print(f"  {'전력 차':<10}{'경기':>5}{'실제':>7}" + "".join(f"{VAR[k][0]:>22}" for k in VAR))
for lo, hi, lab in ((0, 0.6, "0~0.6"), (0.6, 1.0, "0.6~1.0"), (1.0, 1.5, "1.0~1.5"), (1.5, 9, "1.5~ 압도")):
    mk = ((gap >= lo) & (gap < hi)).to_numpy()
    print(f"  {lab:<10}{mk.sum():>5}{sg[mk].mean():>7.2f}" +
          "".join(f"{np.where(hf, te[f'{k}_mh'], te[f'{k}_ma'])[mk].mean():>22.2f}" for k in VAR))

print(f"\n===== 채점 ({TEST} {len(te)}경기, 기준 = 지금 엔진) =====")
print(f"  {'':<26}{'1위 적중':>9}{'확실도':>8}{'골 오차':>9}{'확실도':>8}{'스코어 손실':>11}{'확실도':>8}")
for k in VAR:
    out = []
    for col, hb in (("hit", True), ("dev", False), ("ll", False)):
        x0, x1 = te[f"base_{col}"].astype(float), te[f"{k}_{col}"].astype(float)
        d = (x1 - x0) if hb else (x0 - x1)
        z = 0.0 if k == "base" else d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
        out.append((x1.mean(), z))
    mark = "  ← 24/25 가 고른 것" if k == pick else ""
    print(f"  {VAR[k][0]:<24}{out[0][0]:>9.1%}{out[0][1]:>+8.1f}{out[1][0]:>9.4f}{out[1][1]:>+8.1f}{out[2][0]:>11.4f}{out[2][1]:>+8.1f}{mark}")
print("  확실도 +2 이상이면 맞다 (골 오차·스코어 손실은 낮을수록 좋음, 확실도는 + 가 좋아진 것)")
