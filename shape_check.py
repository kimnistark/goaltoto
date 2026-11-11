"""
B. 골 분포 모양 (11/11)
  평균 골은 그대로 두고, 스코어 분포 '모양'이 실제랑 맞나 → 안 맞는 데를 고치면 1위 스코어가 바뀜
  1) 시뮬(팀만, 매주 그 주 전 정보) 로 24/25 ~ 26/27 경기마다 스코어 분포 저장 (경기당 4000번)
  2) 진단: 스코어 칸별 (0-0, 1-0, 1-1 …) 시뮬 확률 평균 vs 실제 비율 / 팀 골 0·1·2·3+ / 무승부
           선수 구성별 (에이스 한 명 의존도, 팀 슈팅 질) 로 0골·3골+ 이 어긋나는지
  3) 고치기 (시뮬 다시 안 돌리고 분포만 손봄): 24/25 로 찾고 25/26 ~ 26/27 로 채점
       - 무승부 칸 키우기: 0-0, 1-1, 2-2 … 에 × d
       - 저득점 칸 보정 (Dixon-Coles 방식): 0-0, 1-0, 0-1, 1-1 을 ρ 로 같이 조정
     채점: 1위 스코어 적중률, 실제 스코어 확률(로그)

사용: py shape_check.py         (처음엔 시뮬 돌려서 data/shape_preds.json 저장, 20~30분)
      py shape_check.py again   (저장된 분포로 진단·고치기만 다시, 몇 초)
"""
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import live as L
import sim_engine as E

N = 4000
MAXG = 6
DATA = Path(__file__).parent / "data"
OUT = DATA / "shape_preds.json"
m, T = L.load_team_matches()
latest = int(T["season"].max())
SEASONS = [latest - 2, latest - 1, latest]
TRAIN, TEST = [latest - 2], [latest - 1, latest]


def season_params(test):
    saved = L.load_params()
    if saved and saved.get("test_season") == test:
        return saved
    print(f"  ({test}/{(test + 1) % 100} 채점용 α, carry 찾는 중...)")
    p = L.fit(T, test, log=lambda *a, **k: None)
    p["test_season"] = test
    return p


def star_share_tracker():
    """팀별 최근 10경기에서 '제일 많이 만든 선수 xG ÷ 팀 xG' (에이스 의존도). 경기 전 값"""
    hist = defaultdict(list)          # team -> [{pid: xg}]

    def get(team):
        tot = defaultdict(float)
        for d in hist[team][-10:]:
            for k, v in d.items():
                tot[k] += v
        s = sum(tot.values())
        return max(tot.values()) / s if s > 0 else np.nan

    def add(mid, home, away):
        f = DATA / "cache" / f"{mid}.json"
        if not f.exists():
            return
        ro = json.loads(f.read_text(encoding="utf-8")).get("rosters")
        if not isinstance(ro, dict):
            return
        for side, team in (("h", home), ("a", away)):
            x = ro.get(side) or {}
            hist[team].append({int(p["player_id"]): float(p.get("xG") or 0)
                               for p in (x.values() if isinstance(x, dict) else x) if isinstance(p, dict)})
            hist[team] = hist[team][-10:]
    return get, add


def simulate_all():
    rows = []
    for test in SEASONS:
        p = season_params(test)
        const = p["const"]
        E.C["base_open"] = const["base_open"]
        sp_tot = sum(E.C["sp_rate"])
        E.C["sp_rate"] = [r * const["base_sp"] / sp_tot for r in E.C["sp_rate"]]
        E.P["home_mult"] = const["home"]
        TT = T[T["season"] <= test]
        S = {"week": None, "Rw": None}
        get_star, add_star = star_share_tracker()

        def before(rs, R):
            r0 = rs[0]
            h = next(r for r in rs if r.home); a = next(r for r in rs if not r.home)
            if r0.season == test:
                wk = pd.Timestamp(r0.dt).to_period("W-MON").start_time
                if wk != S["week"]:
                    S["week"], S["Rw"] = wk, copy.deepcopy(R)
                for r in rs:
                    S["Rw"].touch(r.team, r.season)
                vals = {}
                for t in (h.team, a.team):
                    v = S["Rw"].team_values(t); vals[t] = v
                    E.C["teams"][t] = v
                    E.P["teams"][t] = {"att_shots": v["att_open"], "def_shots": v["def_open"], "att_q": v["att_q"], "def_q": v["def_q"]}
                sh, sa, _ = E.simulate(h.team, a.team, N, use_players=False)
                mat = np.zeros((MAXG + 1, MAXG + 1))
                np.add.at(mat, (np.minimum(sh, MAXG), np.minimum(sa, MAXG)), 1)
                rows.append({"match_id": int(h.match_id), "season": test, "league": h.league, "home": h.team, "away": a.team,
                             "gh": int(h.gf), "ga": int(a.gf), "mh": float(sh.mean()), "ma": float(sa.mean()),
                             "q_h": vals[h.team]["att_q"], "q_a": vals[a.team]["att_q"],
                             "star_h": get_star(h.team), "star_a": get_star(a.team),
                             "mat": (mat / N).round(5).tolist()})
                if len(rows) % 200 == 0:
                    print(f"    {len(rows)}경기")
            add_star(h.match_id, h.team, a.team)

        print(f"===== [{test}/{(test + 1) % 100}] 시뮬 중 =====")
        L.run(TT, p["alpha"], p["carry"], const, int(TT["season"].min()), before)
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    print(f"저장: {OUT.name} ({len(rows)}경기)\n")
    return rows


rows = json.loads(OUT.read_text(encoding="utf-8")) if ("again" in sys.argv and OUT.exists()) else simulate_all()
D = pd.DataFrame(rows)
M = np.array([np.array(x) for x in D["mat"]])            # (경기, 7, 7)
gh, ga = D["gh"].clip(upper=MAXG).to_numpy(), D["ga"].clip(upper=MAXG).to_numpy()
idx = np.arange(len(D))
te = D["season"].isin(TEST).to_numpy(); tr = D["season"].isin(TRAIN).to_numpy()


def band(pred, act, n):
    se = np.sqrt(max(pred * (1 - pred), 1e-9) / n)
    flag = "  ←" if abs(act - pred) > 2 * se else ""
    return f"{pred:>8.1%}{act:>8.1%}{act - pred:>+8.1%}  ±{2 * se:.1%}{flag}"


# ---------- 2. 진단 ----------
Mt = M[te]; n = te.sum()
print(f"===== 진단: 시뮬 확률 평균 vs 실제 ({TEST} {n}경기) =====")
print(f"  {'':<10}{'시뮬':>8}{'실제':>8}{'차이':>8}   운 범위   (← = 운으로 보기엔 큼)")
for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (3, 1), (1, 3)]:
    print(f"  {i}-{j:<8}{band(Mt[:, i, j].mean(), ((gh[te] == i) & (ga[te] == j)).mean(), n)}")
draw_p = np.trace(Mt, axis1=1, axis2=2).mean()
print(f"  {'무승부':<8}{band(draw_p, (gh[te] == ga[te]).mean(), n)}")
tot = np.add.outer(np.arange(MAXG + 1), np.arange(MAXG + 1))
for k, lab in ((0, "합계 0골"), (1, "합계 1골"), (2, "합계 2골"), (3, "합계 3골")):
    print(f"  {lab:<8}{band(Mt[:, tot == k].sum(1).mean(), ((gh[te] + ga[te]) == k).mean(), n)}")
print(f"  {'합계 4골+':<8}{band(Mt[:, tot >= 4].sum(1).mean(), ((gh[te] + ga[te]) >= 4).mean(), n)}")

print(f"\n  [팀 한쪽 골 분포] (홈·원정 합쳐서 {2 * n}개)")
team_p = np.concatenate([Mt.sum(2), Mt.sum(1)])             # (2n, 7) 팀 골 분포
team_g = np.concatenate([gh[te], ga[te]])
for k, lab in ((0, "0골"), (1, "1골"), (2, "2골")):
    print(f"  {lab:<10}{band(team_p[:, k].mean(), (team_g == k).mean(), 2 * n)}")
print(f"  {'3골+':<10}{band(team_p[:, 3:].sum(1).mean(), (team_g >= 3).mean(), 2 * n)}")

print("\n  [선수 구성별] 0골 / 3골+ 이 어긋나는지 (시뮬 − 실제 를 구간별로)")
Dt = D[te].reset_index(drop=True)
side = pd.DataFrame({"star": np.concatenate([Dt["star_h"], Dt["star_a"]]), "q": np.concatenate([Dt["q_h"], Dt["q_a"]]),
                     "p0": team_p[:, 0], "p3": team_p[:, 3:].sum(1), "g": team_g})
for col, lab in (("star", "에이스 의존도 (최근 10경기 xG 중 1명 비율)"), ("q", "팀 슈팅 질 (살아있는 값)")):
    print(f"   {lab}")
    s = side.dropna(subset=[col])
    if len(s) < 30:
        print("     (데이터 없음)")
        continue
    for qi, g in s.groupby(pd.qcut(s[col].rank(method="first"), 3, labels=["낮음", "중간", "높음"]), observed=True):
        print(f"     {qi}  {g[col].min():.2f}~{g[col].max():.2f}  0골 {band(g['p0'].mean(), (g['g'] == 0).mean(), len(g))}")
        print(f"     {'':<17}        3골+ {band(g['p3'].mean(), (g['g'] >= 3).mean(), len(g))}")


# ---------- 3. 분포만 손보기 ----------
def adjust(Mx, mh, ma, d=1.0, rho=0.0):
    A = Mx.copy()
    if d != 1.0:
        for k in range(MAXG + 1):
            A[:, k, k] *= d
    if rho != 0.0:
        A[:, 0, 0] *= np.maximum(1 - mh * ma * rho, 0)
        A[:, 0, 1] *= np.maximum(1 + mh * rho, 0)
        A[:, 1, 0] *= np.maximum(1 + ma * rho, 0)
        A[:, 1, 1] *= np.maximum(1 - rho, 0)
    return A / A.sum((1, 2), keepdims=True)


def score(A, mask):
    a = A[mask]; ii = idx[: mask.sum()]
    p = a[ii, gh[mask], ga[mask]]
    top = a.reshape(len(a), -1).argmax(1)
    hit = (top // (MAXG + 1) == gh[mask]) & (top % (MAXG + 1) == ga[mask])
    return -np.log(np.maximum(p, 1e-4)), hit


mh, ma = D["mh"].to_numpy(), D["ma"].to_numpy()
ll0_tr, _ = score(M, tr)
grid_d = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
grid_r = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1]
best_d = min(grid_d, key=lambda d: score(adjust(M, mh, ma, d=d), tr)[0].mean())
best_r = min(grid_r, key=lambda r: score(adjust(M, mh, ma, rho=r), tr)[0].mean())
best_b = min(((d, r) for d in grid_d for r in grid_r), key=lambda x: score(adjust(M, mh, ma, *x), tr)[0].mean())
print(f"\n===== 고치기: {TRAIN} 로 찾음 → {TEST} {te.sum()}경기로 채점 =====")
print(f"  찾은 값: 무승부 × {best_d} / 저득점 ρ {best_r:+.2f} / 같이 쓰면 × {best_b[0]}, ρ {best_b[1]:+.2f}")
ll0, hit0 = score(M, te)
print(f"  {'':<22}{'1위 적중':>9}{'확실도':>8}{'스코어 손실':>11}{'확실도':>8}")
print(f"  {'지금 그대로':<20}{hit0.mean():>9.1%}{'':>8}{ll0.mean():>11.4f}")
for lab, kw in (("무승부 키우기", {"d": best_d}), ("저득점 보정", {"rho": best_r}), ("둘 다", {"d": best_b[0], "rho": best_b[1]})):
    ll1, hit1 = score(adjust(M, mh, ma, **kw), te)
    dh = hit1.astype(float) - hit0; dl = ll0 - ll1
    zh = dh.mean() / (dh.std() / np.sqrt(len(dh)) + 1e-12); zl = dl.mean() / (dl.std() / np.sqrt(len(dl)) + 1e-12)
    print(f"  {lab:<20}{hit1.mean():>9.1%}{zh:>+8.1f}{ll1.mean():>11.4f}{zl:>+8.1f}")
print("  확실도 +2 이상이면 맞다 (스코어 손실은 낮을수록 좋음, 확실도는 + 가 좋아진 것)")
