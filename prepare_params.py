"""
시뮬레이션 파라미터 준비 (2025/26, EPL + 라리가 + 분데스리가 통합)
입력: data/matches.csv, data/shots.csv, data/sofa_time.csv, data/match_time.csv
출력: data/params.json  (시뮬이 읽는 숫자 전부)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025      # py prepare_params.py 2024 → 24/25 시즌 기준
LEAGUES = ["EPL", "La_Liga", "Bundesliga"]
TIME_BINS = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 200)]
BIN_NAMES = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
N_QUANT = 200   # xG 분포를 저장할 분위수 개수

DATA = Path(__file__).parent / "data"

# ===== 0. 데이터 불러오기 =====
m = pd.read_csv(DATA / "matches.csv")
s = pd.read_csv(DATA / "shots.csv")
# SofaScore 추가시간 (없으면 2025/26 3개 리그에서 확정해둔 값으로 대체)
DEF_INJ1 = {0: 0.040, 1: 0.204, 2: 0.225, 3: 0.228, 4: 0.135, 5: 0.092, 6: 0.039, 7: 0.023, 8: 0.010, 9: 0.004}
DEF_INJ2 = {2: 0.014, 3: 0.089, 4: 0.175, 5: 0.260, 6: 0.188, 7: 0.126, 8: 0.073, 9: 0.039, 10: 0.017, 11: 0.012, 12: 0.007}
DEF_MEAN1, DEF_MEAN2 = 2.86, 5.65
HAS_SOFA = (DATA / "match_time.csv").exists() and (DATA / "sofa_time.csv").exists()
t = (pd.read_csv(DATA / "match_time.csv")[["match_id", "inj1", "inj2"]] if HAS_SOFA
     else pd.DataFrame(columns=["match_id", "inj1", "inj2"]))
if not HAS_SOFA:
    print("[알림] SofaScore 추가시간 파일이 없어서 저장해둔 확률표(전반 평균 2.86분, 후반 5.65분)를 씀\n")

m = m[(m["season"] == SEASON) & (m["league"].isin(LEAGUES))].copy()
s = s[s["match_id"].isin(m["match_id"])].copy()

# 골 수가 안 맞는 경기 제외 (자책골 처리 후 슈팅으로 센 골 ≠ 공식 골)
g = s[s["result"].isin(["Goal", "OwnGoal"])].groupby(["match_id", "goal_side"]).size().unstack(fill_value=0)
g = g.reindex(columns=["h", "a"], fill_value=0)
chk = m.set_index("match_id")[["FTHG", "FTAG"]].join(g).fillna(0)
bad = chk[(chk["h"] != chk["FTHG"]) | (chk["a"] != chk["FTAG"])].index
m = m[~m["match_id"].isin(bad)]
s = s[s["match_id"].isin(m["match_id"])]

# 경기 시간 = 90 + 전반 추가 + 후반 추가 (없으면 평균으로 채움)
m = m.merge(t, on="match_id", how="left")
n_fill = m["inj1"].isna().sum()
m["inj1"] = pd.to_numeric(m["inj1"], errors="coerce").clip(0, 10)
m["inj2"] = pd.to_numeric(m["inj2"], errors="coerce").clip(1, 15)
m["inj1"] = m["inj1"].fillna(m["inj1"].mean() if m["inj1"].notna().any() else DEF_MEAN1)
m["inj2"] = m["inj2"].fillna(m["inj2"].mean() if m["inj2"].notna().any() else DEF_MEAN2)
m["mins"] = 90 + m["inj1"] + m["inj2"]

shots = s[s["result"] != "OwnGoal"].copy()          # 자책골은 슈팅이 아님
npen = shots[shots["situation"] != "Penalty"]         # 페널티 제외 슈팅

print(f"사용 경기 {len(m)}개 (골 불일치 제외 {len(bad)}개, 추가시간 없어서 평균으로 채운 경기 {n_fill}개)")
print(f"슈팅 {len(shots)}개 (페널티 제외 {len(npen)}개)\n")

# ===== 1. 리그 기준 분당 슈팅 + 홈 배수 =====
total_min = m["mins"].sum()
cnt = shots.groupby("side").size()
rate_h, rate_a = cnt["h"] / total_min, cnt["a"] / total_min
base = (rate_h + rate_a) / 2                          # 팀 1개 기준 분당 슈팅
home = float(np.sqrt(rate_h / rate_a))                # 홈 ×home, 원정 ÷home
print("===== [1] 리그 기준 =====")
print(f"  분당 슈팅 {base:.4f} (경기당 약 {base * m['mins'].mean():.1f}개)")
print(f"  홈 배수 {home:.3f} (홈 분당 {rate_h:.4f} / 원정 분당 {rate_a:.4f})")

# ===== 2. 시간대 배수 =====
# Understat은 전반 추가시간을 46, 47분...으로 이어서 기록 → 46-60 구간 노출시간에 전반 추가시간 포함
share_over90 = (shots["minute"] > 90).mean()
second_half_continues = share_over90 > 0.005
print("\n===== [2] 시간대 배수 =====")
print(f"  90분 넘는 분으로 기록된 슈팅 비율 {share_over90:.1%} → "
      f"후반 추가시간 기록 방식: {'91, 92분...으로 이어짐' if second_half_continues else '90분에 몰림'}")

expo = {n: 0.0 for n in BIN_NAMES}
for _, r in m.iterrows():
    for (lo, hi), n in zip(TIME_BINS, BIN_NAMES):
        e = hi - lo + 1 if hi < 200 else 0
        if n == "0-15":
            e = 15
        if n == "46-60":
            e += r["inj1"]
        if n == "76-90" and not second_half_continues:
            e += r["inj2"]
        if n == "90+":
            e = r["inj2"] if second_half_continues else 0
        expo[n] += e

time_mult = {}
for (lo, hi), n in zip(TIME_BINS, BIN_NAMES):
    k = shots["minute"].between(lo, hi).sum()
    rate = k / expo[n] / 2 if expo[n] else 0
    time_mult[n] = round(rate / base, 3) if expo[n] else 1.0
    print(f"  {n:>6}분: 슈팅 {k:>5}개, 노출 {expo[n]:>8.0f}분 → 배수 {time_mult[n]:.3f}")

# ===== 3. 슈팅 → xG =====
print("\n===== [3] 슈팅 상황과 xG =====")
sit_share = npen["situation"].value_counts(normalize=True)
lg_xg = npen["xG"].mean()
xg_quant = {}
for sit, p in sit_share.items():
    x = npen.loc[npen["situation"] == sit, "xG"]
    xg_quant[sit] = np.quantile(x, np.linspace(0, 1, N_QUANT)).round(5).tolist()
    print(f"  {sit:<15} 비중 {p:5.1%}  평균 xG {x.mean():.3f}")
calib = npen["result"].eq("Goal").sum() / npen["xG"].sum()
print(f"  페널티 제외 슈팅 평균 xG {lg_xg:.4f}, 실제 골 / xG 합 = {calib:.3f} (1에 가까우면 xG가 잘 맞음)")

pen = shots[shots["situation"] == "Penalty"]
pen_rate = len(pen) / (2 * total_min)
pen_conv = pen["result"].eq("Goal").mean()
print(f"  페널티: 분당 {pen_rate:.5f} (경기당 팀 {pen_rate * m['mins'].mean():.3f}개), 성공률 {pen_conv:.1%}")
og_rate = (s["result"] == "OwnGoal").sum() / (2 * total_min)
print(f"  자책골: 분당 {og_rate:.5f} (경기당 팀 {og_rate * m['mins'].mean():.3f}개)")

# ===== 4. 팀별 4개 값 =====
side_team = pd.concat([
    m[["match_id", "home", "away", "mins"]].assign(side="h").rename(columns={"home": "team", "away": "opp"}),
    m[["match_id", "home", "away", "mins"]].assign(side="a").rename(columns={"away": "team", "home": "opp"}),
])
side_team["league"] = side_team["match_id"].map(m.set_index("match_id")["league"])
sh = shots.groupby(["match_id", "side"]).size().rename("n_shots")
q = npen.groupby(["match_id", "side"])["xG"].agg(["sum", "count"]).rename(columns={"sum": "xg", "count": "n_np"})
st = side_team.join(sh, on=["match_id", "side"]).join(q, on=["match_id", "side"]).fillna(0)

att = st.groupby(["team", "league"]).agg(games=("match_id", "count"), mins=("mins", "sum"),
                                          shots=("n_shots", "sum"), xg=("xg", "sum"), n_np=("n_np", "sum"))
dfn = st.groupby("opp").agg(shots_c=("n_shots", "sum"), xg_c=("xg", "sum"), n_np_c=("n_np", "sum"))
teams = att.reset_index().set_index("team").join(dfn)
teams["att_shots"] = teams["shots"] / teams["mins"] / base
teams["def_shots"] = teams["shots_c"] / teams["mins"] / base
teams["att_q"] = teams["xg"] / teams["n_np"] / lg_xg
teams["def_q"] = teams["xg_c"] / teams["n_np_c"] / lg_xg
teams["att_power"] = teams["att_shots"] * teams["att_q"]      # 참고용: 공격 종합
teams["def_power"] = teams["def_shots"] * teams["def_q"]      # 참고용: 수비 종합 (낮을수록 좋음)

# 승점으로 순위 → 리그별 하위 3팀 = 강등팀 (승격팀 대용 평균)
pts = pd.concat([
    m.assign(team=m["home"], p=np.select([m["FTHG"] > m["FTAG"], m["FTHG"] == m["FTAG"]], [3, 1], 0)),
    m.assign(team=m["away"], p=np.select([m["FTAG"] > m["FTHG"], m["FTAG"] == m["FTHG"]], [3, 1], 0)),
]).groupby("team")["p"].sum()
teams["pts"] = pts
relegated = teams.sort_values("pts").groupby("league").head(3).index.tolist()
cols = ["att_shots", "def_shots", "att_q", "def_q"]
promoted = teams.loc[relegated, cols].mean().round(3).to_dict()

print("\n===== [4] 팀별 값 (1.00 = 리그 평균) =====")
print("  att_shots: 슈팅 생산 / def_shots: 슈팅 허용 / att_q: 슈팅 질 / def_q: 허용 슈팅 질")
view = teams[["league", "games", "pts", *cols, "att_power", "def_power"]].sort_values("att_power", ascending=False)
print(view.round(2).to_string())
print(f"\n  강등팀 {relegated}")
print(f"  → 승격팀 대용 값: {promoted}")

# ===== 5. 추가시간 확률표 (3개 리그 통합) =====
if HAS_SOFA:
    sofa = pd.read_csv(DATA / "sofa_time.csv")
    sofa = sofa[(sofa["season"] == SEASON) & sofa["league"].isin(LEAGUES)].dropna(subset=["inj1", "inj2"])
    sofa = sofa[sofa["inj1"].between(0, 10) & sofa["inj2"].between(1, 15)]
    inj = {c: sofa[c].astype(int).value_counts(normalize=True).sort_index().round(4) for c in ("inj1", "inj2")}
    print(f"\n===== [5] 추가시간 ({len(sofa)}경기) =====")
    print(f"  전반 평균 {sofa['inj1'].mean():.2f}분, 후반 평균 {sofa['inj2'].mean():.2f}분")
else:
    inj = {"inj1": pd.Series(DEF_INJ1), "inj2": pd.Series(DEF_INJ2)}
    print("\n===== [5] 추가시간 (저장해둔 확률표 사용) =====")
    print(f"  전반 평균 {DEF_MEAN1}분, 후반 평균 {DEF_MEAN2}분")

# ===== 저장 =====
params = {
    "meta": {"season": SEASON, "leagues": LEAGUES, "matches": int(len(m))},
    "base_shot_per_min": round(float(base), 5),
    "home_mult": round(home, 4),
    "time_mult": time_mult,
    "situation_share": sit_share.round(4).to_dict(),
    "xg_quantiles": xg_quant,
    "league_xg_per_shot": round(float(lg_xg), 5),
    "xg_calib": round(float(calib), 4),   # 시뮬 골 확률 = xG × 이 값 (실제 골 / xG 합)
    "penalty": {"per_min": round(float(pen_rate), 6), "conv": round(float(pen_conv), 4)},
    "own_goal_per_min": round(float(og_rate), 6),
    "added_time": {"first": {int(k): float(v) for k, v in inj["inj1"].items()},
                   "second": {int(k): float(v) for k, v in inj["inj2"].items()}},
    "teams": {tm: {c: round(float(r[c]), 4) for c in cols} | {"league": r["league"]}
              for tm, r in teams.iterrows()},
    "promoted_default": promoted,
}
(DATA / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n저장: data/params.json (팀 {len(teams)}개)")
