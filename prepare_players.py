"""
선수 특성치 준비
입력: data/cache/*.json (Understat 원본), data/matches.csv
출력: data/players.json (팀별 선수 목록 + 특성치), data/players.csv (확인용)
사용: py prepare_players.py          (2025/26 기준)
      py prepare_players.py 2024     (2024/25 기준 → 나중에 시즌 밖 검증용)

특성치
- start_prob : 선발 확률 (그 팀 최근 10경기, 최근일수록 가중치)
- shots90    : 90분당 슈팅 (페널티 제외, 포지션 평균 쪽으로 당김)
- xg90       : 90분당 xG (페널티 제외, 포지션 평균 쪽으로 당김)
- finish     : 결정력 배수 (최근 3시즌 골 ÷ 기대 골, 강하게 1.0 쪽으로 당김)
- gk_save    : 골키퍼 실점 배수 (최근 3시즌 실점 ÷ 기대 실점, 1보다 작을수록 잘 막음)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
HIST = [SEASON - 2, SEASON - 1, SEASON]     # 결정력/골키퍼는 3시즌 누적
RECENT_N, DECAY = 10, 0.85                  # 선발 확률: 최근 10경기, 한 경기 전마다 ×0.85
K_MIN = 900          # 슈팅 생산: 900분(10경기)만큼 포지션 평균을 섞음
A_FIN = 12.0         # 결정력: 평균(1.0)을 섞는 양의 기본값 (기대 골 기준). 아래에서 데이터로 다시 정함
A_GK = 12.0          # 골키퍼: 위와 같음 (기대 실점 기준)
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 200]

def best_k(tab, label, default):
    """지난 시즌들 기록으로 이번 시즌 골을 맞혀보고, 평균을 얼마나 섞을 때 제일 잘 맞는지 고름
    tab: [선수, season, g(실제 골), e(기대 골)]  /  배수 = (지난 골 + K) ÷ (지난 기대 골 + K)"""
    past = tab[tab["season"] < SEASON].groupby("key")[["g", "e"]].sum()
    cur = tab[tab["season"] == SEASON].groupby("key")[["g", "e"]].sum()
    J = past.join(cur, lsuffix="_p", rsuffix="_c", how="inner")
    J = J[(J["e_p"] > 0) & (J["e_c"] > 0)]
    print(f"\n[{label}: 평균을 얼마나 섞을까] 지난 시즌 기록이 있는 선수 {len(J)}명으로 {SEASON}/{(SEASON + 1) % 100} 골을 맞혀봄")
    if len(J) < 30:
        print(f"  지난 시즌 데이터가 부족해서 기본값 {default:g} 사용"); return default
    g = J["g_c"].to_numpy(float)
    dev = lambda mu: (2 * (np.where(g > 0, g * np.log(np.maximum(g, 1e-9) / mu), 0) - (g - mu))).sum()
    base = dev(J["e_c"].to_numpy())
    print(f"  {'섞는 양':>8}{'오차 개선':>10}   (기준 = 지난 기록을 아예 안 씀. 플러스면 지난 기록이 도움이 됨)")
    best, best_gain = None, 0.0
    for k in K_GRID:
        gain = (1 - dev(J["e_c"].to_numpy() * ((J["g_p"] + k) / (J["e_p"] + k)).to_numpy()) / base) * 100
        print(f"  {k:>8g}{gain:>+9.2f}%")
        if gain > best_gain:
            best, best_gain = k, gain
    if best is None:
        print("  → 어떤 양으로도 도움이 안 됨: 지난 기록으로 이번 시즌을 못 맞힘 → 전원 1.0 (반영 안 함)")
        return 1e9
    print(f"  → {best:g} 선택 (오차 {best_gain:+.2f}% 개선)")
    return float(best)

DATA = Path(__file__).parent / "data"
m = pd.read_csv(DATA / "matches.csv")
m["dt"] = pd.to_datetime(m["datetime"])
m = m[m["season"].isin(HIST)].set_index("match_id")

def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"   # DC, DL, DR + 윙백(DML, DMR)
    if pos.startswith("FW"): return "FWD"
    return "MID"

# ---------- 원본 읽기 ----------
apps = []                       # 선수 × 경기
shot_rows = []                  # 슈팅 (선수, 상대 골키퍼 포함)
for mid, r in m.iterrows():
    f = DATA / "cache" / f"{mid}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    rosters = d.get("rosters") if isinstance(d.get("rosters"), dict) else {}
    gk_of = {}
    for side in ("h", "a"):
        ro = rosters.get(side) or {}
        players = ro.values() if isinstance(ro, dict) else ro
        best_gk = (-1, None)
        for p in players:
            if not isinstance(p, dict):
                continue
            t = int(p.get("time", 0) or 0)
            pos = p.get("position", "Sub")
            apps.append({"match_id": mid, "season": r["season"], "dt": r["dt"], "league": r["league"],
                         "team": r["home"] if side == "h" else r["away"],
                         "pid": int(p["player_id"]), "name": p["player"], "pos": pos,
                         "start": pos != "Sub", "mins": t})
            if pos == "GK" and t > best_gk[0]:
                best_gk = (t, int(p["player_id"]))
        gk_of[side] = best_gk[1]
    shots = d.get("shots") if isinstance(d.get("shots"), dict) else {}
    for side in ("h", "a"):
        for s in shots.get(side) or []:
            if s["result"] == "OwnGoal":
                continue
            shot_rows.append({"match_id": mid, "season": r["season"], "pid": int(s["player_id"]),
                              "sit": s["situation"], "res": s["result"], "xg": float(s["xG"]),
                              "gk_faced": gk_of.get("a" if side == "h" else "h")})

A = pd.DataFrame(apps)
S = pd.DataFrame(shot_rows)
S["pen"] = S["sit"] == "Penalty"
S["goal"] = S["res"] == "Goal"
print(f"기준 시즌 {SEASON}/{(SEASON + 1) % 100} · 누적 시즌 {HIST} · 출전 기록 {len(A)}건 · 슈팅 {len(S)}개\n")

# ---------- 1. 이번 시즌 소속팀 + 주 포지션 ----------
cur = A[A["season"] == SEASON]
last_team = cur.sort_values("dt").groupby("pid").tail(1).set_index("pid")[["team", "league", "name"]]
starts_pos = cur[cur["start"]].groupby("pid")["pos"].agg(lambda x: x.value_counts().index[0])
last_team["group"] = starts_pos.map(group).reindex(last_team.index).fillna("MID")

# ---------- 2. 선발 확률 (소속팀 최근 10경기) ----------
start_prob = {}
for team, g in cur.groupby("team"):
    recent = sorted(g["match_id"].unique(), key=lambda x: m.loc[x, "dt"], reverse=True)[:RECENT_N]
    w = {mid: DECAY ** k for k, mid in enumerate(recent)}
    wsum = sum(w.values())
    gg = g[g["match_id"].isin(recent) & g["start"]].drop_duplicates(["match_id", "pid"])
    for pid, h in gg.groupby("pid"):
        start_prob[(team, pid)] = sum(w[x] for x in h["match_id"]) / wsum

# ---------- 3. 슈팅 생산 (이번 시즌, 페널티 제외) ----------
np_cur = S[(S["season"] == SEASON) & ~S["pen"]]
mins = cur.groupby("pid")["mins"].sum()
sh = np_cur.groupby("pid").agg(shots=("xg", "size"), xg=("xg", "sum"))
P = last_team.join(mins.rename("mins")).join(sh).fillna({"shots": 0, "xg": 0, "mins": 0})
grp_rate = P.groupby("group").apply(lambda g: pd.Series({
    "s90": g["shots"].sum() / max(g["mins"].sum(), 1) * 90,
    "x90": g["xg"].sum() / max(g["mins"].sum(), 1) * 90}))
P["shots90"] = (P["shots"] + K_MIN / 90 * P["group"].map(grp_rate["s90"])) / (P["mins"] + K_MIN) * 90
P["xg90"] = (P["xg"] + K_MIN / 90 * P["group"].map(grp_rate["x90"])) / (P["mins"] + K_MIN) * 90

# ---------- 4. 결정력 (3시즌, 페널티 제외) ----------
np_all = S[~S["pen"]]
c = np_all["goal"].sum() / np_all["xg"].sum()                  # 리그 전체 골/xG
_t = np_all.assign(key=np_all["pid"], g=np_all["goal"].astype(float), e=c * np_all["xg"])
A_FIN = best_k(_t.groupby(["key", "season"], as_index=False)[["g", "e"]].sum(), "결정력", A_FIN)
fin = np_all.groupby("pid").agg(g=("goal", "sum"), x=("xg", "sum"))
fin["finish"] = (fin["g"] + A_FIN) / (c * fin["x"] + A_FIN)
P = P.join(fin[["g", "x", "finish"]].rename(columns={"g": "np_goals3", "x": "np_xg3"}))
P["finish"] = P["finish"].fillna(1.0)

# ---------- 5. 골키퍼 (3시즌, 페널티 제외, 유효슈팅 기준) ----------
ot = np_all[np_all["res"].isin(["Goal", "SavedShot"])]
r_ot = ot["goal"].sum() / ot["xg"].sum()
_t = ot.dropna(subset=["gk_faced"]).assign(g=lambda d: d["goal"].astype(float), e=lambda d: r_ot * d["xg"])
_t["key"] = _t["gk_faced"]
A_GK = best_k(_t.groupby(["key", "season"], as_index=False)[["g", "e"]].sum(), "골키퍼", A_GK)
gk = ot.groupby("gk_faced").agg(ga=("goal", "sum"), xga=("xg", "sum"))
gk["gk_save"] = (gk["ga"] + A_GK) / (r_ot * gk["xga"] + A_GK)
P = P.join(gk["gk_save"])
P.loc[P["group"] == "GK", "gk_save"] = P.loc[P["group"] == "GK", "gk_save"].fillna(1.0)

P["start_prob"] = [start_prob.get((t, pid), 0.0) for pid, t in zip(P.index, P["team"])]
P = P.reset_index().rename(columns={"index": "pid"})

# ---------- 확인용 출력 ----------
pd.set_option("display.width", 160)
print(f"\n리그 기준: 페널티 제외 골/xG = {c:.3f}, 유효슈팅 실점/xG = {r_ot:.3f}")
print("\n[포지션 평균 90분당]"); print(grp_rate.round(3).to_string())
act = P[P["mins"] >= 900]
print("\n[90분당 xG TOP 10 (900분 이상)]")
print(act.nlargest(10, "xg90")[["name", "team", "group", "mins", "shots90", "xg90", "finish"]].round(2).to_string(index=False))
print("\n[결정력 TOP 10 / BOTTOM 5 (3시즌 기대 골 10 이상)]")
fz = P[P["np_xg3"] >= 10]
print(fz.nlargest(10, "finish")[["name", "team", "np_goals3", "np_xg3", "finish"]].round(2).to_string(index=False))
print(fz.nsmallest(5, "finish")[["name", "team", "np_goals3", "np_xg3", "finish"]].round(2).to_string(index=False))
print("\n[골키퍼 TOP 5 / BOTTOM 5 (900분 이상, 낮을수록 잘 막음)]")
g2 = P[(P["group"] == "GK") & (P["mins"] >= 900)]
print(g2.nsmallest(5, "gk_save")[["name", "team", "mins", "gk_save"]].round(2).to_string(index=False))
print(g2.nlargest(5, "gk_save")[["name", "team", "mins", "gk_save"]].round(2).to_string(index=False))

# 팀별로 보면: 선발들의 결정력을 (선발 확률 × 슈팅 수)로 가중 평균, 골키퍼는 선발 확률로 가중 평균
_f = P[(P["group"] != "GK") & (P["start_prob"] > 0)].assign(w=lambda d: d["start_prob"] * d["shots90"])
_g = P[(P["group"] == "GK") & (P["start_prob"] > 0)]
tv = pd.DataFrame({"결정력": _f.groupby("team").apply(lambda d: np.average(d["finish"], weights=d["w"] + 1e-9)),
                   "골키퍼": _g.groupby("team").apply(lambda d: np.average(d["gk_save"].fillna(1.0), weights=d["start_prob"]))})
cur_teams = set(m[m["season"] == SEASON]["home"])
tv = tv[tv.index.isin(cur_teams)].sort_values("결정력", ascending=False)
print(f"\n[팀별 선발 평균] 결정력: 1보다 크면 xG보다 잘 넣는 선수들 / 골키퍼: 1보다 작으면 잘 막음  (섞는 양: 결정력 {A_FIN:g}, 골키퍼 {A_GK:g})")
print(pd.concat([tv.head(8), tv.tail(5)]).round(3).to_string())
print(f"  팀 간 범위: 결정력 {tv['결정력'].min():.3f} ~ {tv['결정력'].max():.3f}, 골키퍼 {tv['골키퍼'].min():.3f} ~ {tv['골키퍼'].max():.3f}")

team_chk = P.groupby("team")["start_prob"].sum()
print(f"\n[팀별 선발 확률 합] 평균 {team_chk.mean():.2f} (11에 가까워야 정상), 최소 {team_chk.min():.2f}, 최대 {team_chk.max():.2f}")
ex = "Arsenal" if "Arsenal" in set(P["team"]) else P["team"].iloc[0]
print(f"\n[예시: {ex} 선발 확률 상위 14명]")
print(P[P["team"] == ex].nlargest(14, "start_prob")[["name", "group", "start_prob", "xg90", "finish", "gk_save"]].round(2).to_string(index=False))

# ---------- 저장 ----------
cols = ["pid", "name", "team", "league", "group", "mins", "start_prob", "shots90", "xg90", "finish", "gk_save"]
P[cols].to_csv(DATA / f"players_{SEASON}.csv", index=False, encoding="utf-8-sig")
out = {"season": SEASON, "teams": {}}
for team, g in P[P["start_prob"] > 0].groupby("team"):
    out["teams"][team] = [{k: (None if pd.isna(v) else (round(float(v), 4) if isinstance(v, (float, np.floating)) else v))
                           for k, v in row.items() if k in ("pid", "name", "group", "start_prob", "shots90", "xg90", "finish", "gk_save")}
                          for row in g[cols].to_dict("records")]
(DATA / f"players_{SEASON}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"\n저장: data/players_{SEASON}.json, data/players_{SEASON}.csv (선수 {len(P)}명, 팀 {len(out['teams'])}개)")
