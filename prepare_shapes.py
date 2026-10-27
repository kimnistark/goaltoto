"""
선발 구성(수비/미드/공격 인원) 준비
1) 경기마다 팀 선발 11명의 (수비, 미드, 공격) 인원 → 팀별 최근 기록 저장 (시뮬에서 가중치로 추첨)
2) 인원 효과: 같은 팀이 평소보다 수비 1명 / 공격 1명 더 세웠을 때
   슈팅 생산·허용, 슈팅 질·허용 슈팅 질이 몇 % 바뀌나 (팀 전력·상대 전력·홈 보정 후)
입력: data/params.json, data/matches.csv, data/shots.csv, data/match_time.csv, data/cache
출력: data/shapes.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
P = json.loads((DATA / "params.json").read_text(encoding="utf-8"))
SEASON, LEAGUES = P["meta"]["season"], P["meta"]["leagues"]
KEEP_RECENT = 20      # 팀별로 최근 몇 경기의 구성을 저장할지

def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"
    if pos.startswith("FW"): return "FWD"
    return "MID"

m = pd.read_csv(DATA / "matches.csv")
m = m[(m["season"] == SEASON) & m["league"].isin(LEAGUES)]
_mt = DATA / "match_time.csv"
t = (pd.read_csv(_mt)[["match_id", "inj1", "inj2"]] if _mt.exists()
     else pd.DataFrame(columns=["match_id", "inj1", "inj2"]))   # SofaScore 못 받았으면 평균값 사용
m = m.merge(t, on="match_id", how="left")
m["mins"] = (90 + pd.to_numeric(m["inj1"], errors="coerce").clip(0, 10).fillna(2.9)
             + pd.to_numeric(m["inj2"], errors="coerce").clip(1, 15).fillna(5.7))
s = pd.read_csv(DATA / "shots.csv")
np_s = s[(s["result"] != "OwnGoal") & (s["situation"] != "Penalty")]
q = np_s.groupby(["match_id", "side"])["xG"].agg(["sum", "count"])

# ---------- 1. 경기별 선발 구성 ----------
rows = []
for r in m.itertuples():
    f = DATA / "cache" / f"{r.match_id}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    ro = d.get("rosters") if isinstance(d.get("rosters"), dict) else {}
    cnt = {}
    for side in ("h", "a"):
        x = ro.get(side) or {}
        c = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in (x.values() if isinstance(x, dict) else x):
            if isinstance(p, dict) and p.get("position") not in (None, "Sub"):
                c[group(p["position"])] += 1
        cnt[side] = c
    for side, team, opp, hs, as_ in (("h", r.home, r.away, r.HS, r.AS), ("a", r.away, r.home, r.AS, r.HS)):
        c = cnt[side]
        if c["GK"] != 1 or sum(c.values()) != 11:
            continue
        o = "a" if side == "h" else "h"
        rows.append({"match_id": r.match_id, "dt": r.datetime, "team": team, "opp": opp, "home": side == "h",
                     "DEF": c["DEF"], "MID": c["MID"], "FWD": c["FWD"], "mins": r.mins,
                     "shots_for": hs, "shots_against": as_,
                     "xg_for": q["sum"].get((r.match_id, side), 0), "n_for": q["count"].get((r.match_id, side), 0),
                     "xg_ag": q["sum"].get((r.match_id, o), 0), "n_ag": q["count"].get((r.match_id, o), 0)})
D = pd.DataFrame(rows)
D["shape"] = D["DEF"].astype(str) + "-" + D["MID"].astype(str) + "-" + D["FWD"].astype(str)
print(f"{SEASON}/{(SEASON + 1) % 100} {'+'.join(LEAGUES)}: 팀-경기 {len(D)}개 (선발 11명·GK 1명 정상인 것만)\n")
print("===== [1] 선발 구성 (수비-미드-공격) 빈도 TOP 10 =====")
for sh, n in D["shape"].value_counts().head(10).items():
    print(f"  {sh}: {n:>5} ({n / len(D):5.1%})")

# ---------- 2. 인원 효과 (팀 안에서 평소보다 몇 명 더/덜) ----------
T = P["teams"]
def tv(team, k): return T.get(team, P["promoted_default"])[k]
hm = np.where(D["home"], P["home_mult"], 1 / P["home_mult"])
base, lg_xg = P["base_shot_per_min"], P["league_xg_per_shot"]
D["exp_for"] = base * hm * D["team"].map(lambda x: tv(x, "att_shots")) * D["opp"].map(lambda x: tv(x, "def_shots")) * D["mins"]
D["exp_ag"] = base / hm * D["opp"].map(lambda x: tv(x, "att_shots")) * D["team"].map(lambda x: tv(x, "def_shots")) * D["mins"]
D["dDEF"] = D["DEF"] - D.groupby("team")["DEF"].transform("mean")
D["dFWD"] = D["FWD"] - D.groupby("team")["FWD"].transform("mean")

def wls(y, w):
    X = np.column_stack([np.ones(len(D)), D["dDEF"], D["dFWD"]])
    W = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    res = (y - X @ b) * W
    cov = np.linalg.inv((X * W[:, None]).T @ (X * W[:, None])) * (res @ res) / (len(y) - 3)
    return b[1:], np.sqrt(np.diag(cov))[1:]

targets = {
    "shots_for": (np.log((D["shots_for"] + 0.5) / D["exp_for"]), np.ones(len(D)), "슈팅 생산"),
    "shots_ag": (np.log((D["shots_against"] + 0.5) / D["exp_ag"]), np.ones(len(D)), "슈팅 허용"),
    "q_for": (np.log(((D["xg_for"] + lg_xg) / (D["n_for"] + 1)) / (lg_xg * D["team"].map(lambda x: tv(x, "att_q")) * D["opp"].map(lambda x: tv(x, "def_q")))),
              D["n_for"] + 1, "슈팅 질"),
    "q_ag": (np.log(((D["xg_ag"] + lg_xg) / (D["n_ag"] + 1)) / (lg_xg * D["opp"].map(lambda x: tv(x, "att_q")) * D["team"].map(lambda x: tv(x, "def_q")))),
             D["n_ag"] + 1, "허용 슈팅 질"),
}
print("\n===== [2] 인원 효과: 평소보다 1명 더 세웠을 때 배수 (95% 범위) =====")
print(f"  {'항목':<10}{'수비 +1명':>24}{'공격 +1명':>24}")
effects = {}
for key, (y, w, label) in targets.items():
    b, se = wls(y.to_numpy(), np.asarray(w, float))
    effects[key] = {"per_def": round(float(np.exp(b[0])), 4), "per_fwd": round(float(np.exp(b[1])), 4)}
    cell = lambda i: f"×{np.exp(b[i]):.3f} ({np.exp(b[i] - 1.96 * se[i]):.3f}~{np.exp(b[i] + 1.96 * se[i]):.3f})"
    sig = lambda i: "*" if abs(b[i]) > 1.96 * se[i] else " "
    print(f"  {label:<10}{cell(0):>23}{sig(0)}{cell(1):>23}{sig(1)}")
print("  * = 95% 범위가 1을 안 넘어감 (우연이 아닐 가능성이 큼). 미드는 수비·공격의 나머지라 따로 없음")
print(f"  팀 안에서 구성이 바뀐 폭: 수비 표준편차 {D['dDEF'].std():.2f}명, 공격 {D['dFWD'].std():.2f}명")

# ---------- 3. 팀별 최근 구성 저장 ----------
D = D.sort_values("dt")
teams = {}
for team, g in D.groupby("team"):
    g = g.tail(KEEP_RECENT).iloc[::-1]     # 최근 순
    teams[team] = [{"k": k, "home": bool(r.home), "opp": r.opp, "DEF": int(r.DEF), "MID": int(r.MID), "FWD": int(r.FWD)}
                   for k, r in enumerate(g.itertuples())]
print("\n===== [3] 예시: 팀별 최근 구성 =====")
for ex in [x for x in ("Arsenal", "Manchester City", "Bayern Munich", "Getafe") if x in teams]:
    c = pd.Series([f"{x['DEF']}-{x['MID']}-{x['FWD']}" for x in teams[ex]]).value_counts()
    print(f"  {ex:<16} " + ", ".join(f"{k} {v}회" for k, v in c.items()))

(DATA / "shapes.json").write_text(json.dumps({"season": SEASON, "effects": effects, "teams": teams}, ensure_ascii=False), encoding="utf-8")
print(f"\n저장: data/shapes.json (팀 {len(teams)}개, 팀별 최근 {KEEP_RECENT}경기)")
