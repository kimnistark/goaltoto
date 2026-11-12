"""
스코어 상황 배수 재기 (11/12)
  지고 있는 팀은 더 쏘고, 이기는 팀은 내려앉나? → 상황별(골차 −2 이하 / −1 / 0 / +1 / +2 이상) 슈팅 배수
  팀(시즌) 공격력 × 상대(시즌) 수비력 × 홈 × 시간대 를 같이 맞춰서 상황 효과만 뽑음
    (강팀이 원래 많이 이기고 있어서 생기는 착시를 뺌. 같은 경기 합계로 맞추면 다른 착시가 생겨서 시즌 단위로 함)
    오픈플레이 슈팅 수, 오픈플레이 슈팅당 xG, 세트피스 슈팅 수 각각
  기본: 테스트 시즌(최근 2시즌) 전 경기로만 잼 → 검증할 때 미래 안 섞임
결과: data/gamestate.json (시뮬이 GS_ON 일 때 씀)

사용: py gamestate.py          (최근 2시즌 빼고 잼)
      py gamestate.py all      (전 시즌으로 잼 — 실전용, 검증 끝난 다음에)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
LEAGUES = ["EPL", "La_Liga", "Bundesliga"]
SPK = {"FromCorner", "SetPiece", "DirectFreekick"}
STATES = ["−2↓", "−1", "0", "+1", "+2↑"]
TBINS = [(1, 15, "0-15"), (16, 30, "16-30"), (31, 45, "31-45"), (46, 60, "46-60"), (61, 75, "61-75"), (76, 90, "76-90"), (91, 200, "90+")]
EXTRA2 = 5.65          # 후반 추가시간 평균 (분)

m = pd.read_csv(DATA / "matches.csv")
m = m[m["league"].isin(LEAGUES)]
latest = int(m["season"].max())
if "all" not in sys.argv:
    m = m[m["season"] < latest - 1]
s = pd.read_csv(DATA / "shots.csv")
s = s[s["match_id"].isin(m["match_id"])].copy()
print(f"경기 {len(m)}개 ({int(m['season'].min())}~{int(m['season'].max())} 시즌), 슈팅 {len(s)}개로 잼\n")

goals = s[s["result"].isin(["Goal", "OwnGoal"])][["match_id", "minute", "goal_side"]]
gby = {k: g for k, g in goals.groupby("match_id")}
shots = s[s["result"] != "OwnGoal"]


def tbin(minute):
    for i, (lo, hi, _) in enumerate(TBINS):
        if lo <= minute <= hi:
            return i
    return 0


# 경기·팀별로 (시간대, 상황) 칸마다 노출 분, 오픈 슈팅, 오픈 xG, 세트피스 슈팅
expo = {}; cnt = {}
for mid in m["match_id"]:
    g = gby.get(mid)
    gm = [] if g is None else list(zip(g["minute"], g["goal_side"]))
    for side in ("h", "a"):
        key = (mid, side)
        E = np.zeros((len(TBINS), 5))
        for minute in range(1, 91):
            gd = sum((1 if gs == side else -1) for mn, gs in gm if mn < minute)
            E[tbin(minute), int(np.clip(gd, -2, 2)) + 2] += 1
        gd = sum((1 if gs == side else -1) for mn, gs in gm if mn <= 90)
        E[len(TBINS) - 1, int(np.clip(gd, -2, 2)) + 2] += EXTRA2
        expo[key] = E
        cnt[key] = {"open": np.zeros_like(E), "xg": np.zeros_like(E), "sp": np.zeros_like(E)}
for r in shots.itertuples():
    key = (r.match_id, r.side)
    if key not in cnt:
        continue
    g = gby.get(r.match_id)
    gd = 0 if g is None else sum((1 if gs == r.side else -1) for mn, gs in zip(g["minute"], g["goal_side"]) if mn < r.minute)
    i, j = tbin(r.minute), int(np.clip(gd, -2, 2)) + 2
    if r.situation == "OpenPlay":
        cnt[key]["open"][i, j] += 1; cnt[key]["xg"][i, j] += r.xG
    elif r.situation in SPK:
        cnt[key]["sp"][i, j] += 1

keys = list(expo)
EX = np.array([expo[k] for k in keys])                  # (팀경기, 시간대, 상황)
mm = m.set_index("match_id")
def ids(kind):
    lab = []
    for mid, side in keys:
        r = mm.loc[mid]
        team = r["home"] if side == "h" else r["away"]
        opp = r["away"] if side == "h" else r["home"]
        lab.append(f"{team}|{r['season']}" if kind == "att" else f"{opp}|{r['season']}" if kind == "def" else side)
    u = {v: i for i, v in enumerate(sorted(set(lab)))}
    return np.array([u[v] for v in lab]), len(u)
AI, NA = ids("att"); DI, ND = ids("def"); HI, NH = ids("home")


def ipf(C, iters=60):
    """C ≈ 공격팀시즌 × 상대수비팀시즌 × 홈 × 시간대 T × 상황 G × 노출
    (같은 경기 합계로 맞추면 '먼저 골 넣으려고 슈팅을 많이 쓴 경기'에서 뒤가 적게 나오는 착시가 생겨서, 팀 값은 시즌 단위로 둠)"""
    a = np.ones(NA); d = np.ones(ND); h = np.ones(NH); T = np.ones(len(TBINS)); G = np.ones(5)
    tot = C.sum((1, 2))
    for _ in range(iters):
        TG = T[:, None] * G[None, :]
        base = (EX * TG[None]).sum((1, 2))                       # 팀경기별 (시간·상황 반영한) 노출
        row = a[AI] * d[DI] * h[HI]
        a = np.bincount(AI, tot, NA) / np.maximum(np.bincount(AI, base * d[DI] * h[HI], NA), 1e-9)
        d = np.bincount(DI, tot, ND) / np.maximum(np.bincount(DI, base * a[AI] * h[HI], ND), 1e-9)
        h = np.bincount(HI, tot, NH) / np.maximum(np.bincount(HI, base * a[AI] * d[DI], NH), 1e-9)
        row = a[AI] * d[DI] * h[HI]
        T = C.sum((0, 2)) / np.maximum((EX * row[:, None, None] * G[None, None, :]).sum((0, 2)), 1e-9)
        G = C.sum((0, 1)) / np.maximum((EX * row[:, None, None] * T[None, :, None]).sum((0, 1)), 1e-9)
        T /= (EX.sum((0, 2)) * T).sum() / EX.sum()
        G /= (EX.sum((0, 1)) * G).sum() / EX.sum()
    se = 1 / np.sqrt(np.maximum(C.sum((0, 1)), 1))
    return T, G, se


out = {}
for stat, lab in (("open", "오픈플레이 슈팅 수"), ("sp", "세트피스 슈팅 수")):
    C = np.array([cnt[k][stat] for k in keys])
    T, G, se = ipf(C)
    out[stat] = G.round(4).tolist()
    if stat == "open":
        out["time"] = {TBINS[i][2]: round(float(T[i]), 4) for i in range(len(TBINS))}
    print(f"[{lab}] 상황별 배수 (1 = 평균)")
    for j in range(5):
        print(f"   골차 {STATES[j]:>4}  ×{G[j]:.3f}  (±{2 * se[j] * G[j]:.3f})   노출 {EX.sum((0, 1))[j] / EX.sum():.0%}")
    if stat == "open":
        print("   시간대 배수: " + ", ".join(f"{TBINS[i][2]} ×{T[i]:.2f}" for i in range(len(TBINS))))
    print()

# 슈팅당 xG: 같은 팀·경기의 평균 대비
O = np.array([cnt[k]["open"] for k in keys]); X = np.array([cnt[k]["xg"] for k in keys])
avg_q = (np.bincount(AI, X.sum((1, 2)), NA) / np.maximum(np.bincount(AI, O.sum((1, 2)), NA), 1))[AI]   # 그 팀 시즌 평균
q = X.sum((0, 1)) / np.maximum((O * avg_q[:, None, None]).sum((0, 1)), 1e-9)
q /= (O.sum((0, 1)) * q).sum() / O.sum()
out["q"] = q.round(4).tolist()
print("[오픈플레이 슈팅당 xG] 상황별 배수")
for j in range(5):
    print(f"   골차 {STATES[j]:>4}  ×{q[j]:.3f}")
out["meta"] = {"seasons": [int(m["season"].min()), int(m["season"].max())], "n_matches": int(len(m))}
(DATA / "gamestate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n저장: data/gamestate.json  (−1 = 한 골 지고 있음, +1 = 한 골 이기고 있음)")
