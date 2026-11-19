"""
공정한 채점 틀 (11/19, 교차검증 1~4차 반영)
  채점하는 시즌 s 의 경기를 예측할 때 쓰는 건 전부 's 시즌 전' 데이터뿐:
    - 팀 값: 살아있는 값, 매주 그 주 시작 전 값 (α, carry 도 s 전 시즌으로 찾음)
    - 승격팀 기본값: s−1 시즌 강등팀의 시즌 끝 값 (예전엔 25/26 강등팀 값을 박아놨었음)
    - 기본값 파일 (마무리 기본값·세트피스 xG·자책골·패스 흐름 / xG 분포·시간대·추가시간):
      시즌 시작일 기준으로 prepare_* 를 다시 돌려 data/snap/시즌시작일/ 에 만들어 씀 (params 는 s−1 시즌으로)
    - 마무리 배수 a, b: s 전 시즌으로 다시 찾음
  비교하는 엔진 (같은 경기, 같은 난수):
    old   : 예전 엔진 (마무리 끔, c1 0.045)
    new   : 지금 엔진 (마무리 켬, c1 0.02)
    new45 : 마무리 켬, c1 0.045
  경기마다 7×7 스코어 분포를 저장 → fair_report.py 가 1위 적중 / 핸디 / 언오버 / 1-1 기본 규칙 채점
  저장: data/fair/season_시즌.csv  (이미 있는 시즌은 건너뜀 → 나눠서 돌려도 됨)

사용: py fair_eval.py 2023 2024        (23/24, 24/25 — 20분 정도)
      py fair_eval.py 2025 2026        (25/26, 26/27 지금까지 — 15분 정도)
      (21/22 은 워밍업, 22/23 은 α·carry 를 찾을 앞 시즌이 부족해서 채점 안 함)
      py fair_report.py                 (몇 초)
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import live as L
import sim_engine as E
from lineups import actual_lineups

N = 1200
MAXG = 6
HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "fair"
OUT.mkdir(exist_ok=True)
VARIANTS = {"old": (False, 0.045), "new": (True, 0.02), "new45": (True, 0.045)}
TAG = ""
KNOWN = "k" in sys.argv[1:]
if KNOWN:                       # 선발 앎 공정 비교: 매주 그 주 시점 파일(data/snap/주시작일, make_snaps.py) + 지금 엔진
    VARIANTS = {"t": (False, 0.0), "kn": (False, 0.0)}
    TAG = "_k"
if "b" in sys.argv[1:]:          # 2차 비교: c1 더 낮춰보기 (11/19 공정 채점에서 c1 0.02 가 0.045 보다 핸디 확실히 나음 → 더 낮추면?)
    VARIANTS = {"f0": (True, 0.0), "fm02": (True, -0.02), "o0": (False, 0.0)}
    TAG = "_b"

m, T = L.load_team_matches()
seasons_all = sorted(int(x) for x in T["season"].unique())
# α·carry 를 그 전 시즌으로 찾으려면 '워밍업 1시즌 + 채점 1시즌' 이 앞에 있어야 함 → 첫 두 시즌은 채점 못 함 (21/22, 22/23)
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or seasons_all[2:]
tests = [s for s in tests if s >= seasons_all[2] and not (OUT / f"season_{s}{TAG}.csv").exists()]
if not tests:
    sys.exit("채점할 시즌이 없음 (이미 다 했거나 첫 시즌만 남음). py fair_report.py 로 결과 보면 됨")
print(f"채점 시즌: {tests} · 엔진 {len(VARIANTS)}개 · 경기당 {N}번\n")

# ---------- 0. 시즌별 승격팀 기본값 (그 전 시즌 강등팀 끝 값) ----------
last_mid = T.sort_values("dt").groupby("season")["match_id"].last().to_dict()
teams_in = {s: set(T.loc[T["season"] == s, "team"]) for s in seasons_all}
prom = {}
def grab_rel(rows, R):
    s = rows[0].season
    if rows[0].match_id == last_mid.get(s) and s + 1 in teams_in:
        rel = [t for t in teams_in[s] - teams_in[s + 1] if t in R.acc]
        if rel:
            v = [R.team_values(t) for t in rel]
            prom[s + 1] = {"open": (np.mean([x["att_open"] for x in v]), np.mean([x["def_open"] for x in v])),
                           "q": (np.mean([x["att_q"] for x in v]), np.mean([x["def_q"] for x in v])),
                           "sp": (np.mean([x["att_sp"] for x in v]), np.mean([x["def_sp"] for x in v]))}
pre_alpha = {s_: 0.05 for s_ in L.STATS}; pre_carry = {s_: 0.6 for s_ in L.STATS}
L.run(T, pre_alpha, pre_carry, L.constants(T, seasons_all[1]), seasons_all[0], grab_rel)
L.PROMOTED_BY_SEASON.update(prom)
print("승격팀 기본값 (그 전 시즌 강등팀 끝 값, 공격 슈팅/수비 허용):")
for s, v in sorted(prom.items()):
    print(f"  {s}/{(s + 1) % 100}: 공격 {v['open'][0]:.2f} 수비 {v['open'][1]:.2f}")


def snapshot(s):
    """s 시즌 시작일 기준 기본값 파일 (그 날 전 경기만, params 는 s−1 시즌)"""
    start = pd.to_datetime(m.loc[m["season"] == s, "datetime"]).min().strftime("%Y-%m-%d")
    d = DATA / "snap" / start
    need = ["params.json", "players_live.json", "shapes_live.json", "chain_live.json"]
    if all((d / f).exists() for f in need):
        return d
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", GT_ASOF=start)
    for cmd in (["prepare_params.py", str(s - 1)], ["prepare_players.py", "live"], ["prepare_shapes.py", "live"], ["prepare_chain.py", "live"]):
        r = subprocess.run([sys.executable, str(HERE / cmd[0])] + cmd[1:], cwd=HERE, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cmd[0]}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            sys.exit(f"[{start}] {cmd[0]} 에러 → data/snap/{start}/{cmd[0]}.log 확인\n" + "\n".join(r.stderr.strip().splitlines()[-5:]))
    return d


def fit_finish(rows):
    """마무리 배수 a, b: 오픈플레이 골 ~ xG × exp(b0 + a(A−평균) + b(D−평균)), s 전 시즌 경기로"""
    D = pd.DataFrame(rows, columns=["A", "Dp", "xg", "g"])
    D = D[D["xg"] > 0]
    mA, mD = D["A"].mean(), D["Dp"].mean()
    X = np.column_stack([np.ones(len(D)), D["A"] - mA, D["Dp"] - mD]); y = D["g"].to_numpy(); off = D["xg"].to_numpy()
    beta = np.array([np.log(y.sum() / off.sum()), 0.0, 0.0])
    for _ in range(25):
        mu = off * np.exp(X @ beta)
        beta += np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
    return {"a": float(beta[1]), "b": float(beta[2]), "mA": float(mA), "mD": float(mD)}


for s in tests:
    print(f"\n===== {s}/{(s + 1) % 100} =====")
    d = snapshot(s)
    print(f"  기본값 파일: data/snap/{d.name}/ (그 날 전 경기만)")
    pf = OUT / f"params_{s}.json"
    if pf.exists():
        p = json.loads(pf.read_text(encoding="utf-8"))
    else:
        print("  α, carry 찾는 중 (그 전 시즌으로)...")
        p = L.fit(T, s, log=lambda *a, **k: None)
        pf.write_text(json.dumps({k: v for k, v in p.items() if k != "detail"}, default=float), encoding="utf-8")
    E.load_snapshot(d)
    const = p["const"]
    E.C["base_open"] = const["base_open"]
    sp_tot = sum(E.C["sp_rate"])
    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
    E.P["home_mult"] = const["home"]
    TT = T[T["season"] <= s]
    S = {"week": None, "Rw": None, "fin_rows": [], "fin": None, "wok": True}
    rows = []

    def before(rs, R):
        r0 = rs[0]
        if r0.season < s:                          # 마무리 공식 재료 모으기 (s 전 시즌)
            for r in rs:
                v, o = R.team_values(r.team), R.team_values(r.opp)
                S["fin_rows"].append((np.log(v["att_open"] * v["att_q"]), np.log(o["def_open"] * o["def_q"]), r.xg_open, r.g_open))
            return
        if S["fin"] is None:
            S["fin"] = fit_finish(S["fin_rows"])
            E.FIN = S["fin"]
            print(f"  마무리 a {S['fin']['a']:+.3f}, b {S['fin']['b']:+.3f} (그 전 시즌 {len(S['fin_rows'])} 팀-경기)")
        wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
        if wk != S["week"]:
            S["week"], S["Rw"] = wk, copy.deepcopy(R)
            if KNOWN:                              # 그 주 시점 선수·기본값 파일
                wd = DATA / "snap" / f"{wk:%Y-%m-%d}"
                S["wok"] = (wd / "players_live.json").exists() and (wd / "params.json").exists()
                if S["wok"]:
                    E.load_snapshot(wd)
                    E.C["base_open"] = const["base_open"]
                    sp_tot = sum(E.C["sp_rate"])
                    E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
                    E.P["home_mult"] = const["home"]
                else:
                    S["miss"] = S.get("miss", 0) + 1
        if KNOWN and not S["wok"]:
            return
        for r in rs:
            S["Rw"].touch(r.team, r.season)
        h = next(r for r in rs if r.home); a = next(r for r in rs if not r.home)
        for t in (h.team, a.team):
            v = S["Rw"].team_values(t)
            E.C["teams"][t] = v
            E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
        row = {"match_id": int(h.match_id), "season": s, "league": h.league, "home": h.team, "away": a.team,
               "gh": int(h.gf), "ga": int(a.gf), "n_home": S["Rw"].n.get(h.team, 0), "n_away": S["Rw"].n.get(a.team, 0)}
        kn = actual_lineups(int(h.match_id), h.team, a.team, E.PL) if KNOWN else {}
        if KNOWN:
            row["has_lineup"] = len(kn) == 2
        for k, (fin, c1) in VARIANTS.items():
            E.FIN_ON, E.CORR_C1 = fin, c1
            E.rng = np.random.default_rng(int(h.match_id))          # 엔진끼리 같은 난수
            if k == "kn":
                sh, sa, _ = E.simulate(h.team, a.team, N, use_players=True, known=kn)
            else:
                sh, sa, _ = E.simulate(h.team, a.team, N, use_players=False)
            mat = np.zeros((MAXG + 1, MAXG + 1))
            np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
            row[f"{k}_mh"] = round(float(sh.mean()), 4); row[f"{k}_ma"] = round(float(sa.mean()), 4)
            row[f"{k}_mat"] = json.dumps(np.round(mat / N, 5).tolist())
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f"    {len(rows)}경기")

    L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    E.rng = np.random.default_rng()
    if KNOWN and S.get("miss"):
        print(f"  [주의] 그 주 시점 파일 없는 주 {S['miss']}개 건너뜀 → py make_snaps.py {s} 먼저")
    pd.DataFrame(rows).to_csv(OUT / f"season_{s}{TAG}.csv", index=False, encoding="utf-8-sig")
    print(f"  저장: data/fair/season_{s}{TAG}.csv ({len(rows)}경기)")
print("\n다음: py fair_report.py")
