"""
오답노트 페이지 (3단계)
records/pred_*.csv (경기 전에 박제한 예측) + data/matches.csv, data/shots.csv (실제 결과) → 오답노트.html
  - 결과 전: 경기별 스코어 분포 히트맵, TOP 5, 핸디(골차) 분포
  - 결과 후: 실제 스코어 위치·순위·확률, 리그 평균만으로 찍은 것과 비교, 왜 틀렸나(경기 내용 vs 골 운)
사용: py build_notes.py   → 오답노트.html 더블클릭
"""
import json
from datetime import timedelta
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REC, DATA = HERE / "records", HERE / "data"
OUT = HERE / "오답노트.html"
MAXG = 6

files = sorted(REC.glob("pred_*.csv"))
if not files:
    raise SystemExit("records/ 에 예측 파일이 없어. 경기 전에 py predict_round.py 먼저")

res = pd.read_csv(DATA / "matches.csv").set_index("match_id") if (DATA / "matches.csv").exists() else pd.DataFrame()
xg = {}
if (DATA / "shots.csv").exists():
    s = pd.read_csv(DATA / "shots.csv", usecols=["match_id", "side", "xG"])
    xg = s.groupby(["match_id", "side"])["xG"].sum().to_dict()

# 비교 기준: 팀 정보 없이 리그 평균 홈/원정 골로만 찍은 경우 (score_round.py 와 같은 방식)
if len(res):
    avg_h, avg_a = res["FTHG"].mean(), res["FTAG"].mean()
else:
    avg_h, avg_a = 1.55, 1.25
pois = lambda k, lam: exp(-lam) * lam ** k / factorial(k)
base = np.outer([pois(i, avg_h) for i in range(MAXG + 1)], [pois(j, avg_a) for j in range(MAXG + 1)])


def cause(mh, ma, gh, ga, xh, xa):
    """빗나간 이유: 실제 경기 내용(xG)이 예상과 달랐나, 내용은 맞았는데 골만 다르게 났나"""
    if xh is None:
        return "", ""
    miss_content = abs(xh - mh) + abs(xa - ma)       # 시뮬 평균 골 vs 실제 xG
    miss_luck = abs(gh - xh) + abs(ga - xa)          # 실제 xG vs 실제 골
    if miss_content + miss_luck < 1.5:
        return "ok", "예상대로"
    if miss_content >= miss_luck:
        side = "홈팀" if abs(xh - mh) >= abs(xa - ma) else "원정팀"
        more = (xh - mh) if side == "홈팀" else (xa - ma)
        return "model", f"경기 내용이 예상과 다름 ({side}이 찬스 {'더' if more > 0 else '덜'} 만듦)"
    side = "홈팀" if abs(gh - xh) >= abs(ga - xa) else "원정팀"
    more = (gh - xh) if side == "홈팀" else (ga - xa)
    return "luck", f"골 운 ({side}이 xG보다 {'많이' if more > 0 else '적게'} 넣음)"


rounds = []
for f in files:
    P = pd.read_csv(f)
    games = []
    for r in P.to_dict("records"):
        mat = np.array(json.loads(r["matrix"]))
        kst = pd.to_datetime(r["datetime"]) + timedelta(hours=9)   # Understat 시간은 UTC
        g = {"id": int(r["match_id"]), "kst": f"{kst:%m/%d(%a) %H:%M}".replace("Mon", "월").replace("Tue", "화")
             .replace("Wed", "수").replace("Thu", "목").replace("Fri", "금").replace("Sat", "토").replace("Sun", "일"),
             "ts": kst.isoformat(), "league": r["league"], "home": r["home"], "away": r["away"],
             "mh": r["mean_h"], "ma": r["mean_a"], "mat": np.round(mat, 4).tolist(),
             "top": [[r[f"top{k}"], r[f"top{k}_p"]] for k in range(1, 6)],
             "gd": [r[f"gd_{g:+d}"] for g in range(-4, 5)], "res": None}
        mid = int(r["match_id"])
        if len(res) and mid in res.index:
            gh, ga = int(res.at[mid, "FTHG"]), int(res.at[mid, "FTAG"])
            i, j = min(gh, MAXG), min(ga, MAXG)
            p = float(mat[i, j]); rank = int((mat > p).sum()) + 1
            bp = float(base[i, j]); brank = int((base > bp).sum()) + 1
            gdk = max(-4, min(4, gh - ga))
            bgd = float(sum(base[a, b] for a in range(MAXG + 1) for b in range(MAXG + 1) if max(-4, min(4, a - b)) == gdk))
            xh, xa = xg.get((mid, "h")), xg.get((mid, "a"))
            ck, ct = cause(r["mean_h"], r["mean_a"], gh, ga, xh, xa)
            g["res"] = {"gh": gh, "ga": ga, "p": p, "rank": rank, "bp": bp, "brank": brank,
                        "pgd": float(r[f"gd_{gdk:+d}"]), "bgd": bgd,
                        "xh": None if xh is None else round(xh, 2), "xa": None if xa is None else round(xa, 2),
                        "ck": ck, "ct": ct}
        games.append(g)
    games.sort(key=lambda g: g["ts"])
    r0 = P.iloc[0]
    rounds.append({"name": f.stem.replace("pred_", ""), "asof": str(r0["data_asof"]), "commit": str(r0["commit"]),
                   "c1": float(r0["c1"]), "n": int(r0["n_sims"]), "games": games})

payload = json.dumps(rounds, ensure_ascii=False)
html = (HERE / "notes_template.html").read_text(encoding="utf-8").replace("/*DATA*/[]", payload)
OUT.write_text(html, encoding="utf-8")
n_all = sum(len(r["games"]) for r in rounds)
n_done = sum(1 for r in rounds for g in r["games"] if g["res"])
print(f"생성: {OUT}  (라운드 {len(rounds)}개 · 예측 {n_all}경기 · 결과 나온 {n_done}경기) → 더블클릭해서 크롬으로 열면 돼")
