"""
강팀이 덜 넣는 거 어디서 생기나 (11/12 진단)
  data/shape_preds.json (shape_check.py 가 저장한 시뮬 분포) + 실제 xG·골
  전력 차 구간별, 강팀/약팀 각각:
    시뮬 골 → 실제 골            : 최종 차이
    팀 값 기대 xG → 실제 xG       : 찬스 단계 (여기서 모자라면 ①②③ 찬스를 더 만듦)
    실제 xG → 실제 골 (페널티 빼고) : 마무리 단계 (여기서 모자라면 ④ 마무리를 더 잘함)
  + 압도 경기에서 강팀이 이기고 있을 때 슈팅을 얼마나 줄이나 (평균 팀은 +1 ×0.84, +2 ×0.82)

사용: py strong_diag.py   (몇 분)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import live as L

DATA = Path(__file__).parent / "data"
f = DATA / "shape_preds.json"
if not f.exists():
    raise SystemExit("data/shape_preds.json 이 없어. 먼저 py shape_check.py")
P = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))).drop(columns=["mat"])

m, T = L.load_team_matches()
saved = L.load_params()
p = saved if saved else L.fit(T, int(T["season"].max()) - 1, log=lambda *a, **k: None)
d0 = T[T["season"] < int(T["season"].max()) - 1]
SPQ = d0["xg_sp"].sum() / max(d0["n_sp"].sum(), 1)
ex = {}
def before(rows, R):
    for r in rows:
        e = R.expect(r.team, r.opp, r.home)
        ex[(r.match_id, r.team)] = e["open"] * e["q"] + e["sp"] * SPQ
L.run(T, p["alpha"], p["carry"], p["const"], int(T["season"].min()), before)
T["exp_xg"] = [ex[(a, b)] for a, b in zip(T["match_id"], T["team"])]
T["xg"] = T["xg_open"] + T["xg_sp"]
T["g_np"] = T["g_open"] + T["g_sp"]                    # 페널티·자책골 뺀 골
Tk = T.set_index(["match_id", "team"])

rows = []
for r in P.itertuples():
    hf = r.mh >= r.ma
    s_team, w_team = (r.home, r.away) if hf else (r.away, r.home)
    for role, team, sim, act in (("강팀", s_team, max(r.mh, r.ma), r.gh if hf else r.ga),
                                 ("약팀", w_team, min(r.mh, r.ma), r.ga if hf else r.gh)):
        t = Tk.loc[(r.match_id, team)]
        rows.append({"season": r.season, "gap": abs(r.mh - r.ma), "role": role, "sim_g": sim, "g": act,
                     "exp_xg": t["exp_xg"], "xg": t["xg"], "g_np": t["g_np"]})
D = pd.DataFrame(rows)
EDGES = [0, 0.6, 1.0, 1.5, 9]; LAB = ["0~0.6", "0.6~1.0", "1.0~1.5", "1.5~ 압도"]
D["bin"] = pd.cut(D["gap"], EDGES, labels=LAB, right=False)


def table(d, title):
    print(f"\n===== {title} =====")
    print(f"  {'전력 차':<10}{'':<5}{'경기':>5}{'시뮬골→실제골':>16}{'기대xG→실제xG':>17}{'찬스 배수':>9}{'실제xG→골':>13}{'마무리 배수':>11}")
    for b in LAB:
        for role in ("강팀", "약팀"):
            g = d[(d["bin"] == b) & (d["role"] == role)]
            if len(g) < 15:
                continue
            ch = g["xg"].sum() / g["exp_xg"].sum(); fin = g["g_np"].sum() / g["xg"].sum()
            se_ch = g["xg"].std() / np.sqrt(len(g)) / g["exp_xg"].mean()
            se_fin = np.sqrt(g["g_np"].sum()) / g["xg"].sum()
            print(f"  {b:<10}{role:<5}{len(g):>5}{g['sim_g'].mean():>9.2f}→{g['g'].mean():<6.2f}"
                  f"{g['exp_xg'].mean():>10.2f}→{g['xg'].mean():<6.2f}{ch:>7.2f}±{2 * se_ch:.2f}"
                  f"{g['xg'].mean():>8.2f}→{g['g_np'].mean():<5.2f}{fin:>6.2f}±{2 * se_fin:.2f}")
    print("  찬스 배수 = 실제 xG ÷ 팀 값 기대 xG (1보다 크면 찬스를 예상보다 많이 만듦)")
    print("  마무리 배수 = 페널티 뺀 골 ÷ 실제 xG (리그 평균이 0.9 근처. 강팀만 높으면 마무리 차이)")


seasons = sorted(int(x) for x in D["season"].unique())
table(D[D["season"].isin(seasons[1:])], f"{seasons[1:]} (11/12 채점 시즌)")
table(D[D["season"] == seasons[0]], f"{seasons[0]} (비교: 이 시즌엔 강팀 문제 없었음)")

# ② 압도 경기에서 강팀이 이기고 있을 때 슈팅
s = pd.read_csv(DATA / "shots.csv")
big = P[(P["mh"] - P["ma"]).abs() >= 1.5]
expo = {0: 0.0, 1: 0.0, 2: 0.0}; cnt = {0: 0, 1: 0, 2: 0}
for r in big.itertuples():
    side = "h" if r.mh >= r.ma else "a"
    ss = s[s["match_id"] == r.match_id]
    gl = ss[ss["result"].isin(["Goal", "OwnGoal"])][["minute", "goal_side"]].values.tolist()
    def state(mn):
        return int(np.clip(sum(1 if gs == side else -1 for m_, gs in gl if m_ < mn), -9, 2))
    for mn in range(1, 91):
        st = state(mn)
        if st >= 0:
            expo[st] += 1
    for sh in ss[(ss["side"] == side) & (ss["situation"] == "OpenPlay") & (ss["result"] != "OwnGoal")].itertuples():
        st = state(sh.minute)
        if 0 <= st <= 2 and sh.minute <= 90:
            cnt[st] += 1
r0 = cnt[0] / max(expo[0], 1)
print(f"\n===== ② 압도 경기({len(big)}개) 강팀: 이기고 있을 때 오픈플레이 슈팅 (동점일 때 = 1) =====")
for st, lab, avg in ((1, "1골 이김", 0.844 / 1.024), (2, "2골+ 이김", 0.821 / 1.024)):
    rr = cnt[st] / max(expo[st], 1) / r0
    print(f"  {lab}: ×{rr:.2f}  (슈팅 {cnt[st]}개, {expo[st]:.0f}분)   ← 전 팀 평균 ×{avg:.2f}")
print("  강팀 숫자가 평균보다 확실히 크면 '강팀은 이기고 있어도 안 내려앉음' (②)")
print("  (시간대 효과는 안 뺀 단순 비교라 참고용)")
