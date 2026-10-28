"""
공격 전개(빌드업 → 마지막 패스 → 슈터) + 세트피스 준비  [시뮬 v3용]
입력: data/params.json, data/matches.csv, data/cache/*.json, (있으면) data/match_time.csv, data/players_{시즌}.json
출력: data/chain.json
사용: py prepare_chain.py

만드는 값
- 기준값  : 오픈플레이 분당 슈팅, 세트피스 3종류 분당 슈팅, 종류별 골/xG 보정, 패스 받아 쏜 비율
- 흐름표  : 패스한 선수 그룹 → 슈터 그룹 궁합 (1.00 = 평균, 높을수록 그 조합이 자주 나옴)
- 팀 값   : 오픈플레이 슈팅 생산·허용 / 슈팅 질·허용 질 / 세트피스 슈팅 생산·허용
- 선수 값 : 빌드업, 키패스, 오픈플레이 슈팅(90분당) / 슈팅 질, 패스 질 / 세트피스 종류별 슈팅 빈도·평균 xG
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

K_MIN = 900       # 90분당 수치: 900분만큼 포지션 평균을 섞음 (prepare_players와 같음)
K_Q = 20          # 슈팅 질·패스 질: 슈팅 20개만큼 평균을 섞음
K_SP = 10         # 세트피스 평균 xG: 슈팅 10개만큼 '같은 포지션·같은 종류' 평균을 섞음
SP = {"FromCorner": "corner", "SetPiece": "ifk", "DirectFreekick": "dfk"}
SP_KO = {"corner": "코너", "ifk": "간접FK", "dfk": "직접FK"}
GROUPS = ["DEF", "MID", "FWD"]

DATA = Path(__file__).parent / "data"
P = json.loads((DATA / "params.json").read_text(encoding="utf-8"))
SEASON, LEAGUES = P["meta"]["season"], P["meta"]["leagues"]
DEF_MINS = 90 + 2.86 + 5.65

def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"
    if pos.startswith("FW"): return "FWD"
    return "MID"

import sys
LIVE = len(sys.argv) > 1 and sys.argv[1] == "live"     # live: 최근 1년 경기로 계산 (선수 = players_live.json)
m = pd.read_csv(DATA / "matches.csv")
if LIVE:
    m["_dt"] = pd.to_datetime(m["datetime"])
    m = m[(m["_dt"] > m["_dt"].max() - pd.Timedelta(days=365)) & m["league"].isin(LEAGUES)].copy()
    SEASON = int(m["season"].max())
    print(f"[live] {m['_dt'].max():%Y-%m-%d} 기준 최근 1년 경기로 계산")
else:
    m = m[(m["season"] == SEASON) & m["league"].isin(LEAGUES)].copy()
_mt = DATA / "match_time.csv"
if _mt.exists():
    t = pd.read_csv(_mt)[["match_id", "inj1", "inj2"]]
    m = m.merge(t, on="match_id", how="left")
    m["mins"] = (90 + pd.to_numeric(m["inj1"], errors="coerce").clip(0, 10).fillna(2.86)
                 + pd.to_numeric(m["inj2"], errors="coerce").clip(1, 15).fillna(5.65))
else:
    m["mins"] = DEF_MINS

# ---------- 원본 읽기 ----------
apps, shots, ogs = [], [], []
used = []
for r in m.itertuples():
    f = DATA / "cache" / f"{r.match_id}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    ro = d.get("rosters") if isinstance(d.get("rosters"), dict) else {}
    sh = d.get("shots") if isinstance(d.get("shots"), dict) else {}
    used.append(r.match_id)
    for side, team, opp in (("h", r.home, r.away), ("a", r.away, r.home)):
        x = ro.get(side) or {}
        by_name = {}
        for p in (x.values() if isinstance(x, dict) else x):
            if not isinstance(p, dict):
                continue
            pid = int(p["player_id"])
            by_name[p.get("player")] = pid
            apps.append({"match_id": r.match_id, "dt": r.datetime, "team": team, "pid": pid, "name": p.get("player"),
                         "pos": p.get("position", "Sub"), "mins": int(p.get("time", 0) or 0),
                         "build": float(p.get("xGBuildup") or 0)})
        for s in sh.get(side) or []:
            sit = s.get("situation")
            if s.get("result") == "OwnGoal":          # 자책골은 넣은 선수 팀 쪽에 기록돼 있음 → 상대 팀 득점
                ogs.append({"match_id": r.match_id, "team": opp, "opp": team, "home": side == "a"})
                continue
            if sit == "Penalty":
                continue
            shots.append({"match_id": r.match_id, "team": team, "opp": opp, "pid": int(s["player_id"]),
                          "kind": "open" if sit == "OpenPlay" else SP.get(sit, "open"),
                          "xg": float(s["xG"]), "goal": s.get("result") == "Goal",
                          "passer": by_name.get(s.get("player_assisted")) if s.get("player_assisted") else None})
m = m[m["match_id"].isin(used)]
A = pd.DataFrame(apps).drop_duplicates(["match_id", "pid"])
S = pd.DataFrame(shots)
S["passer"] = S["passer"].astype("Int64")
print(f"{SEASON}/{(SEASON + 1) % 100} {'+'.join(LEAGUES)}: 경기 {len(m)}개, 페널티 제외 슈팅 {len(S)}개\n")

# ---------- 1. 기준값 ----------
tot_min = m["mins"].sum()                      # 경기 시간 합 (팀 1개 기준 분당 = 개수 ÷ (2 × 이 값))
O = S[S["kind"] == "open"]
base_open = len(O) / (2 * tot_min)
calib_open = O["goal"].sum() / O["xg"].sum()
p_assist = O["passer"].notna().mean()
lg_q_open = O["xg"].mean()
lg_qp = O.loc[O["passer"].notna(), "xg"].mean()
sp_rate, sp_calib, sp_xg = {}, {}, {}
print("===== [1] 기준값 (팀 1개 기준) =====")
print(f"  {'':<10}{'경기당':>8}{'분당':>9}{'평균xG':>9}{'골/xG':>8}")
print(f"  {'오픈플레이':<10}{base_open * m['mins'].mean():>8.2f}{base_open:>9.4f}{lg_q_open:>9.3f}{calib_open:>8.2f}")
for k in SP.values():
    g = S[S["kind"] == k]
    sp_rate[k] = len(g) / (2 * tot_min)
    sp_calib[k] = g["goal"].sum() / g["xg"].sum() if len(g) else 1.0
    sp_xg[k] = g["xg"].mean() if len(g) else 0.1
    print(f"  {SP_KO[k]:<10}{sp_rate[k] * m['mins'].mean():>8.2f}{sp_rate[k]:>9.4f}{sp_xg[k]:>9.3f}{sp_calib[k]:>8.2f}")
print(f"  오픈플레이 슈팅 중 패스 받아 쏜 비율 {p_assist:.1%}")

# ---------- 2. 선수 주 포지션 ----------
st = A[A["pos"] != "Sub"]
main_pos = st.groupby("pid")["pos"].agg(lambda x: x.value_counts().index[0])
grp = main_pos.map(group)
gof = lambda pid: grp.get(pid, "MID")

# ---------- 3. 흐름표 (패스 그룹 → 슈터 그룹) ----------
F = O[O["passer"].notna()].copy()
F["pg"] = F["passer"].map(gof); F["sg"] = F["pid"].map(gof)
F = F[F["pg"].isin(GROUPS) & F["sg"].isin(GROUPS)]
joint = pd.crosstab(F["pg"], F["sg"]).reindex(index=GROUPS, columns=GROUPS, fill_value=0) + 1.0
jp = joint / joint.values.sum()
R = jp / np.outer(jp.sum(axis=1), jp.sum(axis=0))
print("\n===== [2] 흐름표: 패스한 선수 → 슈터 =====")
print("  (왼쪽 %: 전체 중 그 조합 비율 / 오른쪽 ×: 궁합. 1.00보다 크면 그 조합이 평균보다 자주 나옴)")
print(f"  {'패스＼슈터':<8}" + "".join(f"{g:>16}" for g in GROUPS))
for pg in GROUPS:
    print(f"  {pg:<10}" + "".join(f"{jp.loc[pg, sg]:>9.1%} ×{R.loc[pg, sg]:.2f}" for sg in GROUPS))

# ---------- 4. 팀 값 ----------
gm_ = pd.concat([m[["home", "mins"]].rename(columns={"home": "team"}), m[["away", "mins"]].rename(columns={"away": "team"})])
tmin = gm_.groupby("team")["mins"].sum()
S["sp"] = S["kind"] != "open"
T = pd.DataFrame(index=tmin.index)
T["league"] = pd.concat([m.set_index("home")["league"], m.set_index("away")["league"]]).groupby(level=0).first()
base_sp = sum(sp_rate.values())
T["n_open"] = O.groupby("team").size(); T["c_open"] = O.groupby("opp").size()
T["n_sp"] = S[S["sp"]].groupby("team").size(); T["c_sp"] = S[S["sp"]].groupby("opp").size()
T = T.fillna({"n_open": 0, "c_open": 0, "n_sp": 0, "c_sp": 0})
T["att_open"] = T["n_open"] / tmin / base_open
T["def_open"] = T["c_open"] / tmin / base_open
T["att_q"] = O.groupby("team")["xg"].mean() / lg_q_open
T["def_q"] = O.groupby("opp")["xg"].mean() / lg_q_open
T[["att_q", "def_q"]] = T[["att_q", "def_q"]].fillna(1.0)

def shrink(ratio, n, label):
    """팀 간 차이 중 '진짜' 비율만큼만 남기고 1.0 쪽으로 당김 (세트피스는 팀당 표본이 적어서)"""
    noise = (1 / n.clip(lower=1)).mean()
    rel = float(np.clip(1 - noise / max(ratio.var(), 1e-9), 0, 1))
    print(f"  {label}: 팀 간 차이 중 진짜로 보이는 비율 {rel:.0%} → 그만큼만 반영")
    return 1 + rel * (ratio - 1)
print("\n===== [3] 팀 값 =====")
T["att_sp"] = shrink(T["n_sp"] / tmin / base_sp, T["n_sp"], "세트피스 슈팅 생산")
T["def_sp"] = shrink(T["c_sp"] / tmin / base_sp, T["c_sp"], "세트피스 슈팅 허용")
cols = ["att_open", "def_open", "att_q", "def_q", "att_sp", "def_sp"]
T["power"] = T["att_open"] * T["att_q"] / (T["def_open"] * T["def_q"])
print("  att_open/def_open: 오픈플레이 슈팅 생산/허용, att_q/def_q: 오픈플레이 슈팅 질/허용 질, att_sp/def_sp: 세트피스 슈팅 생산/허용")
view = T.sort_values("power", ascending=False)[["league", *cols]].round(2)
print(pd.concat([view.head(8), view.tail(5)]).to_string())

pts = pd.concat([
    m.assign(team=m["home"], p=np.select([m["FTHG"] > m["FTAG"], m["FTHG"] == m["FTAG"]], [3, 1], 0)),
    m.assign(team=m["away"], p=np.select([m["FTAG"] > m["FTHG"], m["FTAG"] == m["FTHG"]], [3, 1], 0)),
]).groupby("team")["p"].sum()
releg = T.assign(pts=pts).sort_values("pts").groupby("league").head(3).index
promoted = T.loc[releg, cols].mean().round(3).to_dict()
print(f"  승격팀 대용 (리그별 하위 3팀 평균): {promoted}")

# ---------- 4-2. 자책골: 밀어붙이는 정도에 따라 얼마나 늘어나나 ----------
# 자책골로 얻는 골 ~ c × (홈 배수 × 우리 오픈플레이 공격 × 상대 오픈플레이 수비)^γ.  γ=0이면 누구나 똑같음(예전 방식)
h_ = P["home_mult"]
tmm = pd.concat([m.assign(team=m["home"], opp=m["away"], hm=h_), m.assign(team=m["away"], opp=m["home"], hm=1 / h_)])[["match_id", "team", "opp", "hm", "mins"]]
OG = pd.DataFrame(ogs, columns=["match_id", "team", "opp", "home"])
tmm["og"] = tmm.set_index(["match_id", "team"]).index.map(OG.groupby(["match_id", "team"]).size()).fillna(0).to_numpy()
tmm["r"] = tmm["hm"] * tmm["team"].map(T["att_open"]) * tmm["opp"].map(T["def_open"])
print("\n===== [3-2] 자책골 =====")
print(f"  자책골 {int(tmm['og'].sum())}개 (팀당 경기당 {tmm['og'].mean():.3f}골)")
og_best = (-1e18, 0.0, 0.0); ll0 = None
print(f"  {'γ':>6}{'맞는 정도':>10}   (γ=0 대비. 클수록 실제 자책골 분포와 잘 맞음)")
for gam in np.round(np.arange(0, 3.01, 0.25), 2):
    e = tmm["r"] ** gam * tmm["mins"]
    c_ = tmm["og"].sum() / e.sum()
    ll_ = float((tmm["og"] * np.log(c_ * e) - c_ * e).sum())
    ll0 = ll_ if ll0 is None else ll0
    print(f"  {gam:>6.2f}{ll_ - ll0:>+10.2f}")
    if ll_ > og_best[0]:
        og_best = (ll_, float(gam), float(c_))
og_gamma, og_c = og_best[1], og_best[2]
q_ = tmm["r"].quantile([0.1, 0.5, 0.9])
print(f"  → γ = {og_gamma:.2f} 선택. 경기당 자책골로 얻는 골: 밀리는 쪽(하위 10%) {og_c * q_.iloc[0] ** og_gamma * DEF_MINS:.3f} / "
      f"중간 {og_c * q_.iloc[1] ** og_gamma * DEF_MINS:.3f} / 앞서는 쪽(상위 10%) {og_c * q_.iloc[2] ** og_gamma * DEF_MINS:.3f}")
print("  (맞는 정도 차이가 2 미만이면 그 γ들은 사실상 구분이 안 됨)")

# ---------- 5. 선수 값 ----------
PL = pd.DataFrame({"mins": A.groupby("pid")["mins"].sum(), "build": A.groupby("pid")["build"].sum(),
                   "name": A.groupby("pid")["name"].last()})
PL["group"] = [gof(i) for i in PL.index]
PL["team"] = A.sort_values("dt").groupby("pid")["team"].last()
PL["sh"] = O.groupby("pid").size(); PL["xg"] = O.groupby("pid")["xg"].sum()
AS = O[O["passer"].notna()]
PL["kp"] = AS.groupby("passer").size(); PL["xa"] = AS.groupby("passer")["xg"].sum()
for k in SP.values():
    g = S[S["kind"] == k]
    PL[f"n_{k}"] = g.groupby("pid").size(); PL[f"x_{k}"] = g.groupby("pid")["xg"].sum()
PL = PL.fillna(0)
PL = PL[PL["group"].isin(GROUPS)]

G = PL.groupby("group")
gsum = G[["mins", "build", "sh", "xg", "kp", "xa"] + [f"{a}_{k}" for k in SP.values() for a in ("n", "x")]].sum()
per90 = lambda col: gsum[col] / gsum["mins"].clip(lower=1) * 90
gmean = pd.DataFrame({"build90": per90("build"), "kp90": per90("kp"), "s90": per90("sh"),
                      "qs": gsum["xg"] / gsum["sh"].clip(lower=1)})
for k in SP.values():
    gmean[f"sp90_{k}"] = per90(f"n_{k}")
    gmean[f"spx_{k}"] = (gsum[f"x_{k}"] / gsum[f"n_{k}"].clip(lower=1)).where(gsum[f"n_{k}"] > 0, sp_xg[k])
gm = lambda col: PL["group"].map(gmean[col])
r90 = lambda num, col: (PL[num] + K_MIN / 90 * gm(col)) / (PL["mins"] + K_MIN) * 90
PL["build90"] = r90("build", "build90")
PL["kp90"] = r90("kp", "kp90")
PL["s90"] = r90("sh", "s90")
PL["rs"] = (PL["xg"] + K_Q * gm("qs")) / (PL["sh"] + K_Q) / lg_q_open          # 슈팅 질 (1.00 = 리그 평균)
PL["rp"] = (PL["xa"] + K_Q * lg_qp) / (PL["kp"] + K_Q) / lg_qp                  # 패스 질 (1.00 = 리그 평균)
for k in SP.values():
    PL[f"sp90_{k}"] = r90(f"n_{k}", f"sp90_{k}")
    PL[f"spx_{k}"] = (PL[f"x_{k}"] + K_SP * gm(f"spx_{k}")) / (PL[f"n_{k}"] + K_SP)

print("\n===== [4] 포지션 평균 (선수 1명, 90분당) =====")
print(f"  {'':<6}{'빌드업':>8}{'키패스':>8}{'슈팅':>8}{'슈팅질':>8}" + "".join(f"{SP_KO[k] + '슈팅':>10}{'xG':>7}" for k in SP.values()))
for g in GROUPS:
    r = gmean.loc[g]
    print(f"  {g:<6}{r['build90']:>8.3f}{r['kp90']:>8.2f}{r['s90']:>8.2f}{r['qs'] / lg_q_open:>8.2f}"
          + "".join(f"{r[f'sp90_{k}']:>10.2f}{r[f'spx_{k}']:>7.3f}" for k in SP.values()))

act = PL[PL["mins"] >= 900]
def show(title, col, cols_, n=7, sub=None):
    d = act if sub is None else act[sub(act)]
    print(f"\n  [{title}]")
    print(d.nlargest(n, col)[["name", "team", "group", "mins", *cols_]].round(3).to_string(index=False))
print("\n===== [5] 선수 예시 (900분 이상) — 아는 선수들이 맞게 나오는지 확인 =====")
show("빌드업 TOP (수비수)", "build90", ["build90"], sub=lambda d: d["group"] == "DEF")
show("빌드업 TOP (미드)", "build90", ["build90"], sub=lambda d: d["group"] == "MID")
show("키패스 TOP", "kp90", ["kp90", "rp"])
show("오픈플레이 슈팅 TOP", "s90", ["s90", "rs"])
show("패스 질 TOP (키패스 30개 이상)", "rp", ["kp", "rp"], sub=lambda d: d["kp"] >= 30)
show("코너 슈팅 TOP", "sp90_corner", ["n_corner", "sp90_corner", "spx_corner"])
show("직접 프리킥 TOP", "sp90_dfk", ["n_dfk", "sp90_dfk", "spx_dfk"])

# ---------- 저장 ----------
keep = None
_pf = DATA / ("players_live.json" if LIVE else f"players_{SEASON}.json")
if _pf.exists():
    keep = {p["pid"] for ps in json.loads(_pf.read_text(encoding="utf-8"))["teams"].values() for p in ps}
pcols = ["build90", "kp90", "s90", "rs", "rp"] + [f"{a}_{k}" for k in SP.values() for a in ("sp90", "spx")]
def pack(r):
    return {"b": round(float(r["build90"]), 4), "k": round(float(r["kp90"]), 4), "s": round(float(r["s90"]), 4),
            "rs": round(float(r["rs"]), 4), "rp": round(float(r["rp"]), 4),
            "sw": [round(float(r[f"sp90_{k}"]), 4) for k in SP.values()],
            "sx": [round(float(r[f"spx_{k}"]), 4) for k in SP.values()]}
gdef = gmean.copy(); gdef["rs"] = gdef["qs"] / lg_q_open; gdef["rp"] = 1.0
out = {
    "season": SEASON,
    "base_open": round(float(base_open), 5), "calib_open": round(float(calib_open), 4),
    "p_assist": round(float(p_assist), 4), "q_open": round(float(lg_q_open), 5),
    "sp_types": list(SP.values()),
    "sp_rate": [round(float(sp_rate[k]), 5) for k in SP.values()],
    "sp_calib": [round(float(sp_calib[k]), 4) for k in SP.values()],
    "sp_xg": [round(float(sp_xg[k]), 4) for k in SP.values()],
    "og_c": round(og_c, 7), "og_gamma": og_gamma,       # 자책골 분당 = og_c × (홈 × 공격 × 상대 수비)^og_gamma
    "flow": [[round(float(R.loc[a, b]), 4) for b in GROUPS] for a in GROUPS],
    "teams": {tm: {c: round(float(r[c]), 4) for c in cols} for tm, r in T.iterrows()},
    "promoted_default": promoted,
    "group_mean": {g: pack(gdef.loc[g]) for g in GROUPS},
    "players": {str(pid): pack(r) for pid, r in PL.iterrows() if keep is None or pid in keep},
}
OUT = "chain_live.json" if LIVE else "chain.json"
(DATA / OUT).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"\n저장: data/{OUT} (팀 {len(out['teams'])}개, 선수 {len(out['players'])}명)")
