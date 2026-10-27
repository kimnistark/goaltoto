"""
핸디캡 보정 검증
"시뮬이 핸디승 30%라고 한 경기들이 실제로 30% 정도 핸디승이 났나?"
- 25/26 시즌 3개 리그 전 경기, 경기마다 N번 시뮬 → 핸디 이벤트별 시뮬 확률
- 시뮬 확률 구간별로 실제 적중률과 비교 (보정표)
사용: py validate_handicap.py                  (3개 리그, 선수 단위 v3)
      py validate_handicap.py --team           (3개 리그, 선수 없이 팀 값만 - 비교용)
      py validate_handicap.py EPL              (한 리그만)
"""
import sys

import numpy as np
import pandas as pd

from sim_engine import CORR_C1, DATA, LAST, P, SEASON, simulate

N = 1000
args = [a for a in sys.argv[1:] if not a.startswith("--")]
USE_PLAYERS = "--team" not in sys.argv
leagues = [args[0]] if args else P["meta"]["leagues"]

# 홈팀 기준 핸디 h (정수): 핸디승 = 골차 ≥ 1-h, 핸디패 = 골차 ≤ -h-1
EVENTS = [
    ("홈 -2 핸디승 (3골차+ 홈승)", lambda gd: gd >= 3),
    ("홈 -1 핸디승 (2골차+ 홈승)", lambda gd: gd >= 2),
    ("홈 -1 핸디패 (홈 무/패)", lambda gd: gd <= 0),
    ("홈 -2 핸디패 (홈 1골차승 이하)", lambda gd: gd <= 1),
    ("홈 +1 핸디승 (홈 무/승)", lambda gd: gd >= 0),
    ("홈 +1 핸디패 (2골차+ 원정승)", lambda gd: gd <= -2),
]
BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]

m = pd.read_csv(DATA / "matches.csv")
m = m[(m["season"] == SEASON) & m["league"].isin(leagues)].reset_index(drop=True)
print(f"전력 차 보정: {'c1 = ' + format(CORR_C1, '.3f') + ' 적용' if CORR_C1 > 0 else '없음 (data/corr.json 없음)'}")
print(f"{' + '.join(leagues)} {SEASON}/{(SEASON + 1) % 100}: 실제 {len(m)}경기, 경기마다 {N}번 시뮬 "
      f"[{'선수 단위 v3' if USE_PLAYERS else '팀 값만'}]\n")

pred = {name: np.zeros(len(m)) for name, _ in EVENTS}
sim_h, sim_a = np.zeros(len(m)), np.zeros(len(m))
raw_h, raw_a = np.zeros(len(m)), np.zeros(len(m))      # 보정 전 평균 골
KEYS = ["n_open", "xg_open", "open", "n_sp", "xg_sp", "sp", "pen", "og"]
SIM = {k: np.zeros((len(m), 2)) for k in KEYS}      # 경기별 시뮬 평균 [홈, 원정]
for i, r in m.iterrows():
    gh, ga, _ = simulate(r["home"], r["away"], N, use_players=USE_PLAYERS)
    gd = gh - ga
    sim_h[i], sim_a[i] = gh.mean(), ga.mean()
    raw_h[i], raw_a[i] = LAST["raw"]
    for k in KEYS:
        SIM[k][i] = LAST[k]
    for name, f in EVENTS:
        pred[name][i] = f(gd).mean()
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(m)}")

real_gd = (m["FTHG"] - m["FTAG"]).to_numpy()
# 경기별 시뮬 평균 골 저장 (보정값 찾기용: check_corr.py 가 읽음)
_sm = pd.DataFrame({"match_id": m["match_id"], "datetime": m["datetime"], "home": m["home"], "away": m["away"],
              "sim_h": raw_h, "sim_a": raw_a, "real_h": m["FTHG"], "real_a": m["FTAG"]})      # 보정 전 값을 저장 (보정값 찾기용)
_sm.to_csv(DATA / "sim_means.csv", index=False, encoding="utf-8-sig")
_sm.to_csv(DATA / f"sim_means_{SEASON}.csv", index=False, encoding="utf-8-sig")

# ----- 전력 차 구간별 평균 골: 시뮬이 강팀 골을 덜 주는지, 약팀 골을 더 주는지 -----
print("\n===== [0] 시뮬 예상 골차로 10등분: 평균 골 (시뮬 vs 실제) =====")
print(f"  {'구간 (시뮬 골차)':<16}{'경기':>5} |{'홈골 시뮬':>8}{'실제':>7}{'차이':>7} |{'원정골 시뮬':>9}{'실제':>7}{'차이':>7} |{'골차 시뮬':>8}{'실제':>7}{'차이':>7}")
T = pd.DataFrame({"sh": sim_h, "sa": sim_a, "rh": m["FTHG"].to_numpy(), "ra": m["FTAG"].to_numpy()})
T["bin"] = pd.qcut(T["sh"] - T["sa"], 10, duplicates="drop")
for b, g in T.groupby("bin", observed=True):
    sh, sa, rh, ra = g["sh"].mean(), g["sa"].mean(), g["rh"].mean(), g["ra"].mean()
    print(f"  {f'{b.left:+.2f} ~ {b.right:+.2f}':<18}{len(g):>5} |{sh:>9.2f}{rh:>8.2f}{rh - sh:>+8.2f} |{sa:>10.2f}{ra:>8.2f}{ra - sa:>+8.2f} |"
          f"{sh - sa:>+9.2f}{rh - ra:>+8.2f}{(rh - ra) - (sh - sa):>+8.2f}")
print("  위쪽 = 원정이 강한 경기, 아래쪽 = 홈이 강한 경기. 한 칸의 실제 평균은 ±0.2골 정도는 우연으로 흔들림")

# ----- 전력 차가 큰 경기만 골라서: 슈팅 → xG → 골을 종류별로 시뮬 vs 실제 -----
s = pd.read_csv(DATA / "shots.csv")
s = s[s["match_id"].isin(m["match_id"])]
SPK = ["FromCorner", "SetPiece", "DirectFreekick"]
REAL = {k: np.zeros((len(m), 2)) for k in KEYS}
pos = {mid: i for i, mid in enumerate(m["match_id"])}
for r in s.itertuples():
    i = pos[r.match_id]
    if r.result == "OwnGoal":
        REAL["og"][i, 0 if r.goal_side == "h" else 1] += 1; continue
    j, g = (0 if r.side == "h" else 1), r.result == "Goal"
    if r.situation == "Penalty": REAL["pen"][i, j] += g
    elif r.situation == "OpenPlay": REAL["n_open"][i, j] += 1; REAL["xg_open"][i, j] += r.xG; REAL["open"][i, j] += g
    elif r.situation in SPK: REAL["n_sp"][i, j] += 1; REAL["xg_sp"][i, j] += r.xG; REAL["sp"][i, j] += g
LAB = {"n_open": "오픈플레이 슈팅 수", "xg_open": "오픈플레이 xG 합", "open": "오픈플레이 골", "n_sp": "세트피스 슈팅 수",
       "xg_sp": "세트피스 xG 합", "sp": "세트피스 골", "pen": "페널티 골", "og": "자책골로 얻은 골"}
gdp = sim_h - sim_a
order = np.argsort(gdp)
n10 = len(m) // 10
print("\n===== [0-2] 전력 차가 큰 경기만: 어느 단계에서 새나 (경기당 평균, 시뮬 vs 실제) =====")
for title, idx, strong in ((f"홈이 가장 강한 {n10}경기", order[-n10:], 0), (f"그다음 {n10}경기 (홈이 강함)", order[-2 * n10:-n10], 0),
                           (f"원정이 가장 강한 {n10}경기", order[:n10], 1)):
    print(f"\n  [{title}]")
    print(f"  {'':<16} | {'강팀 시뮬':>8}{'실제':>7}{'차이':>7}{'비율':>6} | {'약팀 시뮬':>8}{'실제':>7}{'차이':>7}{'비율':>6}")
    tot = np.zeros((2, 2))
    for k in KEYS:
        cells = []
        for j, side in enumerate((strong, 1 - strong)):
            a, b = SIM[k][idx, side].mean(), REAL[k][idx, side].mean()
            if k in ("open", "sp", "pen", "og"): tot[j] += (a, b)
            cells.append(f"{a:>9.2f}{b:>8.2f}{b - a:>+8.2f}{b / a if a > 0 else 0:>7.2f}")
        print(f"  {LAB[k]:<14} | " + " | ".join(cells))
    print(f"  {'골 합계':<16} | " + " | ".join(f"{a:>9.2f}{b:>8.2f}{b - a:>+8.2f}{b / a:>7.2f}" for a, b in tot))
print("\n  읽는 법: 비율 = 실제 ÷ 시뮬. 1.00이면 맞는 것. 약팀 쪽에서 1보다 작은 줄, 강팀 쪽에서 1보다 큰 줄이 새는 곳")
print("  슈팅 수 → xG 합 → 골 순서로 보면 어느 단계에서 벌어지기 시작하는지 보임 (xG 합은 맞는데 골이 틀리면 골 전환 문제)")
print("\n※ 오차 범위(±)는 표본 수로 계산한 통계적 흔들림 (95%). 실제와 시뮬 차이가 이 안이면 '맞는다'고 봐도 됨\n")

rows = []
for name, f in EVENTS:
    p, y = pred[name], f(real_gd).astype(float)
    brier = np.mean((p - y) ** 2)
    base = np.mean((y.mean() - y) ** 2)
    print(f"===== {name} =====")
    print(f"  전체: 시뮬 평균 {p.mean():5.1%} / 실제 {y.mean():5.1%}   "
          f"예측력(Brier 개선) {1 - brier / base:+.1%}  (+면 '그냥 평균'보다 나음)")
    print(f"  {'시뮬 확률 구간':<14}{'경기수':>6}{'시뮬':>8}{'실제':>8}{'차이':>8}{'오차범위':>9}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (p >= lo) & (p < hi)
        n = mask.sum()
        if n < 10:
            continue
        a, b = p[mask].mean(), y[mask].mean()
        se = 1.96 * np.sqrt(max(a * (1 - a), 1e-9) / n)
        flag = "" if abs(b - a) <= se else "  ← 벗어남"
        print(f"  {f'{lo:.0%}~{min(hi, 1):.0%}':<14}{n:>6}{a:>8.1%}{b:>8.1%}{(b - a) * 100:>+7.1f}p{se * 100:>7.1f}p{flag}")
        rows.append({"event": name, "bin": f"{lo:.0%}-{min(hi, 1):.0%}", "n": n, "sim": a, "real": b})
    print()

pd.DataFrame(rows).to_csv(DATA / f"handicap_calibration_{'v3' if USE_PLAYERS else 'v3_team'}.csv", index=False, encoding="utf-8-sig")
print(f"저장: data/handicap_calibration_{'v3' if USE_PLAYERS else 'v3_team'}.csv")
print("참고: 파라미터를 같은 25/26 시즌으로 만들었기 때문에 실제보다 조금 유리하게 나오는 검증이야 (답을 보고 푼 셈).")
