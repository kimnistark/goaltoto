"""
살아있는 팀 값 (1단계)
사용: py live.py      → 지금 시점 팀 값을 data/live_teams.json 에 저장 (α, carry 는 data/live_params.json, 없으면 찾음. 1~3분)
- 팀 값을 '시즌 평균' 대신 '그 경기 직전까지의 기록'으로 계산
- 경기마다 옛 기록의 비중을 (1 − α) 만큼 줄이고 새 경기를 더함  →  새 값 ≈ 기존 값 × (1 − α) + 이번 경기 × α
- 상대 전력도 그 시점 값으로 보정 (약팀 상대로 슈팅 많이 쏜 건 덜 쳐줌)
- 시즌이 바뀌면 지난 기록을 carry 만큼만 남기고 새 시즌에서 빨리 따라감. 처음 보는 팀(승격팀)은 승격팀 기본값에서 시작
- α, carry 는 지난 시즌들로 "이 값이면 다음 경기를 제일 잘 맞혔다"를 찾아서 정함 (walkforward.py)

값 3종 (각각 공격/수비)
  open : 오픈플레이 슈팅 수        (1.00 = 리그 평균)
  q    : 오픈플레이 슈팅당 xG       (1.00 = 리그 평균)
  sp   : 세트피스 슈팅 수          (1.00 = 리그 평균)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
LEAGUES = ["EPL", "La_Liga", "Bundesliga"]
MINS = 90 + 2.86 + 5.65
SPK = {"FromCorner", "SetPiece", "DirectFreekick"}
STATS = ("open", "q", "sp")
PRIOR_K = 3.0          # 처음 보는 팀은 '경기 3개 분량'의 기본값에서 시작
PROMOTED = {"open": (0.83, 1.17), "q": (0.90, 1.05), "sp": (0.93, 1.11)}   # 승격팀 기본값 (공격, 수비). 25/26 강등팀 평균


def load_team_matches(leagues=LEAGUES):
    """경기를 팀 기준 두 줄(홈 공격, 원정 공격)로 펼침. 날짜순"""
    m = pd.read_csv(DATA / "matches.csv")
    m = m[m["league"].isin(leagues)].copy()
    m["dt"] = pd.to_datetime(m["datetime"])
    s = pd.read_csv(DATA / "shots.csv")
    s = s[s["match_id"].isin(m["match_id"]) & (s["result"] != "OwnGoal")].copy()
    s["goal"] = (s["result"] == "Goal").astype(int)
    o = s[s["situation"] == "OpenPlay"].groupby(["match_id", "side"]).agg(n_open=("xG", "size"), xg_open=("xG", "sum"), g_open=("goal", "sum"))
    sp = s[s["situation"].isin(SPK)].groupby(["match_id", "side"]).agg(n_sp=("xG", "size"), xg_sp=("xG", "sum"), g_sp=("goal", "sum"))
    parts = []
    for side, tc, oc, gf, ga in (("h", "home", "away", "FTHG", "FTAG"), ("a", "away", "home", "FTAG", "FTHG")):
        d = m[["match_id", "dt", "season", "league", tc, oc, gf, ga]].rename(columns={tc: "team", oc: "opp", gf: "gf", ga: "ga"})
        d["side"] = side
        d["home"] = side == "h"
        parts.append(d)
    T = pd.concat(parts, ignore_index=True)
    T = T.join(o, on=["match_id", "side"]).join(sp, on=["match_id", "side"])
    T[["n_open", "xg_open", "g_open", "n_sp", "xg_sp", "g_sp"]] = T[["n_open", "xg_open", "g_open", "n_sp", "xg_sp", "g_sp"]].fillna(0)
    return m.sort_values("dt"), T.sort_values(["dt", "match_id", "side"]).reset_index(drop=True)


def constants(T, before_season):
    """리그 기준값 (그 시즌 이전 데이터만)"""
    d = T[T["season"] < before_season]
    if d.empty:
        d = T
    n = len(d)
    rate_h = d.loc[d["home"], "n_open"].sum() / (d["home"].sum() * MINS)
    rate_a = d.loc[~d["home"], "n_open"].sum() / ((~d["home"]).sum() * MINS)
    return {"base_open": d["n_open"].sum() / (n * MINS), "q": d["xg_open"].sum() / max(d["n_open"].sum(), 1),
            "base_sp": d["n_sp"].sum() / (n * MINS), "home": float(np.sqrt(rate_h / rate_a))}


class Ratings:
    """팀 값 계산기. 경기를 날짜순으로 하나씩 넣으면 값이 갱신됨"""

    def __init__(self, const, alpha, carry, first_season):
        self.c, self.alpha, self.carry, self.first = const, alpha, carry, first_season
        self.acc = {}       # team -> {stat: [공격 num, 공격 den, 수비 num, 수비 den]}
        self.season = {}    # team -> 마지막으로 뛴 시즌
        self.n = {}         # team -> 이번 시즌 경기 수

    def _unit(self, stat):
        c = self.c
        return {"open": c["base_open"] * MINS, "q": c["base_open"] * MINS * c["q"], "sp": c["base_sp"] * MINS}[stat]

    def _init(self, team, season):
        u = {s: self._unit(s) * PRIOR_K for s in STATS}
        fresh = season == self.first
        self.acc[team] = {s: [(1.0 if fresh else PROMOTED[s][0]) * u[s], u[s], (1.0 if fresh else PROMOTED[s][1]) * u[s], u[s]] for s in STATS}

    def touch(self, team, season):
        """시즌이 바뀌었으면 지난 기록 비중을 carry 만큼으로 줄임. 처음 보거나 1시즌 넘게 빠졌던 팀은 승격팀 기본값부터"""
        last = self.season.get(team)
        if last is None or season - last > 1:
            self._init(team, season)
        elif season != last:
            for s in STATS:
                cr = self.carry[s]
                self.acc[team][s] = [v * cr for v in self.acc[team][s]]
        if season != last:
            self.n[team] = 0
        self.season[team] = season

    def get(self, team, stat):
        a = self.acc[team][stat]
        return a[0] / a[1], a[2] / a[3]

    def team_values(self, team):
        """sim_engine 이 쓰는 형식"""
        (ao, do), (aq, dq), (asp, dsp) = (self.get(team, s) for s in STATS)
        return {"att_open": ao, "def_open": do, "att_q": aq, "def_q": dq, "att_sp": asp, "def_sp": dsp}

    def expect(self, team, opp, home):
        """이 팀의 이번 경기 기대치: 오픈플레이 슈팅 수, 슈팅당 xG, 세트피스 슈팅 수"""
        hm = self.c["home"] if home else 1 / self.c["home"]
        ta, oa = self.acc[team], self.acc[opp]
        return {"open": self.c["base_open"] * MINS * hm * (ta["open"][0] / ta["open"][1]) * (oa["open"][2] / oa["open"][3]),
                "q": self.c["q"] * (ta["q"][0] / ta["q"][1]) * (oa["q"][2] / oa["q"][3]),
                "sp": self.c["base_sp"] * MINS * hm * (ta["sp"][0] / ta["sp"][1]) * (oa["sp"][2] / oa["sp"][3])}

    def update_match(self, rows):
        """한 경기(팀 두 줄)를 반영. 두 팀 값 모두 '경기 전' 값으로 상대 보정한 뒤 한꺼번에 갱신"""
        snap = {r.team: {s: self.get(r.team, s) for s in STATS} for r in rows}
        adds = []
        for r in rows:
            hm = self.c["home"] if r.home else 1 / self.c["home"]
            (_, d_open), (_, d_q), (_, d_sp) = (snap[r.opp][s] for s in STATS)
            (a_open, _), (a_q, _), (a_sp, _) = (snap[r.team][s] for s in STATS)
            e_open, e_sp = self.c["base_open"] * MINS * hm, self.c["base_sp"] * MINS * hm
            # 공격 쪽 (이 팀이 쏜 것)
            adds.append((r.team, "open", 0, r.n_open, e_open * d_open))
            adds.append((r.team, "q", 0, r.xg_open, r.n_open * self.c["q"] * d_q))
            adds.append((r.team, "sp", 0, r.n_sp, e_sp * d_sp))
            # 수비 쪽 (상대가 이 팀 슈팅을 허용한 것)
            adds.append((r.opp, "open", 2, r.n_open, e_open * a_open))
            adds.append((r.opp, "q", 2, r.xg_open, r.n_open * self.c["q"] * a_q))
            adds.append((r.opp, "sp", 2, r.n_sp, e_sp * a_sp))
        for team, stat, k, num, den in adds:
            a = self.acc[team][stat]
            keep = 1 - self.alpha[stat]
            if stat == "q" and den == 0:
                continue          # 슈팅이 하나도 없으면 슈팅 질은 건드리지 않음
            a[k] = a[k] * keep + num
            a[k + 1] = a[k + 1] * keep + den
        for r in rows:
            self.n[r.team] = self.n.get(r.team, 0) + 1


def group_rows(T):
    """경기별로 묶은 줄 목록 (날짜순). 여러 번 돌릴 때 한 번만 만들어 두면 빠름"""
    return [list(g.itertuples()) for _, g in T.groupby("match_id", sort=False)]


def run(T, alpha, carry, const, first_season, on_before=None, groups=None):
    """날짜순으로 전 경기를 돌림. on_before(경기 줄들, Ratings) 는 각 경기 반영 직전에 불림 (예측·채점용)"""
    R = Ratings(const, alpha, carry, first_season)
    for rows in (groups if groups is not None else group_rows(T)):
        for r in rows:
            R.touch(r.team, r.season)
        if on_before is not None:
            on_before(rows, R)
        R.update_match(rows)
    return R


def poisson_dev(y, mu):
    mu = max(mu, 1e-6)
    return 2 * ((y * np.log(y / mu) if y > 0 else 0.0) - (y - mu))


def fit(T, test_season, grid_alpha=(0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20), grid_carry=(0.2, 0.4, 0.6, 0.8, 1.0), log=print):
    """α, carry 를 test_season 이전 시즌들로 찾음 (첫 시즌은 워밍업, 그다음 시즌부터 채점)"""
    T = T[T["season"] < test_season].sort_values(["dt", "match_id", "side"])
    first = int(T["season"].min())
    const = constants(T, test_season)
    groups = group_rows(T)
    best = {}
    for stat in STATS:
        res = []
        for a in grid_alpha:
            for cr in grid_carry:
                alpha = {s: (a if s == stat else 0.05) for s in STATS}
                carry = {s: (cr if s == stat else 0.6) for s in STATS}
                tot = [0.0, 0.0]

                def score(rows, R, stat=stat, tot=tot):
                    for r in rows:
                        if r.season == first:
                            continue
                        e = R.expect(r.team, r.opp, r.home)
                        if stat == "open":
                            tot[0] += poisson_dev(r.n_open, e["open"]); tot[1] += 1
                        elif stat == "sp":
                            tot[0] += poisson_dev(r.n_sp, e["sp"]); tot[1] += 1
                        elif r.n_open > 0:
                            tot[0] += r.n_open * (r.xg_open / r.n_open - e["q"]) ** 2; tot[1] += r.n_open
                run(T, alpha, carry, const, first, score, groups)
                res.append((tot[0] / max(tot[1], 1), a, cr))
        res.sort()
        base = [x for x in res if x[1] == 0.0 and x[2] == 1.0]
        best[stat] = {"alpha": res[0][1], "carry": res[0][2], "loss": res[0][0], "loss_frozen": base[0][0] if base else None}
        log(f"  {stat:<5} α = {res[0][1]:.2f}, carry = {res[0][2]:.1f}" +
            (f"   (옛 경기도 똑같이 반영할 때보다 다음 경기 오차 {(1 - res[0][0] / base[0][0]) * 100:+.1f}%)" if base else ""))
    return {"alpha": {s: best[s]["alpha"] for s in STATS}, "carry": {s: best[s]["carry"] for s in STATS},
            "detail": best, "const": const, "fit_seasons": sorted(int(x) for x in T["season"].unique())}


def save_params(p, path=DATA / "live_params.json"):
    path.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


def load_params(path=DATA / "live_params.json"):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_current(R, T, path=DATA / "live_teams.json"):
    """지금 시점(마지막 경기 뒤) 이번 시즌 팀들의 값 저장"""
    latest = int(T["season"].max())
    lg = T[T["season"] == latest].groupby("team")["league"].first()
    cur = {tm: {**{k: round(float(v), 4) for k, v in R.team_values(tm).items()}, "league": lg.get(tm),
                "games_this_season": R.n.get(tm, 0)}
           for tm in R.acc if R.season.get(tm) == latest}
    meta = {"asof": str(T["dt"].max().date()), "season": latest}
    path.write_text(json.dumps({"meta": meta, "teams": cur}, ensure_ascii=False, indent=1), encoding="utf-8")
    return cur, meta


if __name__ == "__main__":
    m, T = load_team_matches()
    latest = int(T["season"].max())
    p = load_params()
    if not p or p.get("test_season") != latest:
        print(f"α, carry 찾기 ({latest}/{(latest + 1) % 100} 이전 시즌으로)")
        p = fit(T, latest)
        p["test_season"] = latest
        save_params(p)
    R = run(T, p["alpha"], p["carry"], p["const"], int(T["season"].min()))
    cur, meta = write_current(R, T)
    print(f"저장: data/live_teams.json ({meta['asof']} 기준, 팀 {len(cur)}개)")
