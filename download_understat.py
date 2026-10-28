"""
Understat 4대리그 원 데이터 다운로드
결과: data/matches.csv (경기 단위), data/shots.csv (슈팅 단위), data/red_cards.csv,
      data/fixtures.csv (아직 안 치른 경기 일정 → predict_round.py 가 씀)
중간에 끊겨도 data/cache에 받은 경기는 다시 안 받음 → 그냥 다시 실행하면 이어서 받음
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

# ===== 설정 =====
TEST_MODE = False  # True: EPL 2025/26 한 시즌만 (약 5분). 확인 후 False로 바꿔서 전체 실행
LEAGUES = ["EPL", "La_Liga", "Serie_A", "Bundesliga"]
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]   # 시작 연도 (2026 = 2026/27). 2020은 지금 계산에 안 쓰여서 뺌
DELAY = 0.6        # 요청 간격(초)

BASE = "https://understat.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}
DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"

session = requests.Session()
session.headers.update(HEADERS)


def get_json(path: str, retries: int = 3):
    for i in range(retries):
        try:
            r = session.get(BASE + path, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"   재시도 {i + 1}/{retries}: {path} ({e})")
            time.sleep(3 * (i + 1))
    return None


FIXTURES = []   # 아직 안 치른 경기


def league_matches(league: str, season: int) -> list[dict]:
    d = get_json(f"getLeagueData/{league}/{season}")
    if not d or "dates" not in d:
        print(f"  [실패] {league} {season}: 경기 목록 없음")
        return []
    rows = []
    for m in d["dates"]:
        if not m.get("isResult"):
            FIXTURES.append({"match_id": int(m["id"]), "league": league, "season": season, "datetime": m["datetime"],
                             "home": m["h"]["title"], "away": m["a"]["title"]})
            continue  # 아직 안 치른 경기 → 일정으로만 저장
        rows.append({
            "match_id": int(m["id"]), "league": league, "season": season,
            "datetime": m["datetime"],
            "home": m["h"]["title"], "away": m["a"]["title"],
            "FTHG": int(m["goals"]["h"]), "FTAG": int(m["goals"]["a"]),
        })
    return rows


def match_detail(mid: int):
    f = CACHE / f"{mid}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = get_json(f"getMatchData/{mid}")
    if d is None:
        return None
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    time.sleep(DELAY)
    return d


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    leagues, seasons = (["EPL"], [2025]) if TEST_MODE else (LEAGUES, SEASONS)

    # 1) 경기 목록
    matches = []
    for lg in leagues:
        for ss in seasons:
            rows = league_matches(lg, ss)
            print(f"  [목록] {lg:<11} {ss}: {len(rows)}경기")
            matches += rows
            time.sleep(DELAY)
    matches = pd.DataFrame(matches)
    if FIXTURES:
        pd.DataFrame(FIXTURES).sort_values("datetime").to_csv(DATA / "fixtures.csv", index=False, encoding="utf-8-sig")
        print(f"  [일정] 아직 안 치른 경기 {len(FIXTURES)}개 → data/fixtures.csv")
    print(f"\n경기 상세 {len(matches)}개 받는 중 (예상 {len(matches) * DELAY / 60:.0f}분 이상)")

    # 2) 경기별 슈팅 / 레드카드
    shots, reds, failed, empty = [], [], [], []
    for i, mid in enumerate(matches["match_id"], 1):
        d = match_detail(mid)
        if d is None:
            failed.append(mid)
            continue
        if i == 1 and "shots" not in d:
            print("  [경고] 응답 구조가 예상과 다름. 키:", list(d.keys()))
        shots_d = d.get("shots") if isinstance(d.get("shots"), dict) else {}
        rosters_d = d.get("rosters") if isinstance(d.get("rosters"), dict) else {}
        if not shots_d:
            empty.append(mid)
        for side in ("h", "a"):
            for s in shots_d.get(side) or []:
                shots.append({
                    "match_id": mid, "side": side, "minute": int(s["minute"]),
                    "result": s["result"], "xG": float(s["xG"]),
                    "situation": s.get("situation"),
                })
            roster = rosters_d.get(side) or {}
            players = roster.values() if isinstance(roster, dict) else roster
            for p in players:
                if not isinstance(p, dict):
                    continue
                if int(p.get("red_card", 0) or 0) > 0:
                    reds.append({
                        "match_id": mid, "side": side, "time": int(p.get("time", 0) or 0),
                        "starter": p.get("position") != "Sub",
                    })
        if i % 100 == 0:
            print(f"  {i}/{len(matches)}")

    shots = pd.DataFrame(shots)
    reds = pd.DataFrame(reds, columns=["match_id", "side", "time", "starter"])

    # 3) 경기 단위로 집계
    # Understat은 자책골을 자책한 선수의 팀(side)에 기록 → 득점 팀은 반대편
    own = shots["result"] == "OwnGoal"
    shots["goal_side"] = shots["side"]
    shots.loc[own, "goal_side"] = shots.loc[own, "side"].map({"h": "a", "a": "h"})

    shots["is_shot"] = ~own
    shots["is_sot"] = shots["result"].isin(["Goal", "SavedShot"])
    shots["is_goal"] = shots["result"].isin(["Goal", "OwnGoal"])
    shots["is_ht_goal"] = shots["is_goal"] & (shots["minute"] <= 45)

    # 슈팅/유효슈팅은 찬 팀 기준, 골은 득점 팀 기준으로 따로 집계
    shot_agg = shots.groupby(["match_id", "side"])[["is_shot", "is_sot"]].sum()
    goal_agg = (shots[shots["is_goal"]]
                .groupby(["match_id", "goal_side"])[["is_goal", "is_ht_goal"]].sum()
                .rename_axis(["match_id", "side"]))
    agg = shot_agg.join(goal_agg, how="outer").fillna(0).astype(int)
    agg = agg.unstack("side", fill_value=0)
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.rename(columns={
        "is_shot_h": "HS", "is_shot_a": "AS", "is_sot_h": "HST", "is_sot_a": "AST",
        "is_ht_goal_h": "HTHG", "is_ht_goal_a": "HTAG",
        "is_goal_h": "shotG_h", "is_goal_a": "shotG_a",
    }).reset_index()
    red_cnt = reds.groupby(["match_id", "side"]).size().unstack(fill_value=0)
    red_cnt = red_cnt.reindex(columns=["h", "a"], fill_value=0)
    red_cnt.columns = ["HR", "AR"]

    out = matches.merge(agg, on="match_id", how="left").merge(
        red_cnt, left_on="match_id", right_index=True, how="left")
    out = out.fillna(0)

    # 4) 검증: 슈팅에서 센 골 vs 공식 골
    mismatch = (out["shotG_h"] != out["FTHG"]) | (out["shotG_a"] != out["FTAG"])
    out = out.drop(columns=["shotG_h", "shotG_a"])

    out.to_csv(DATA / "matches.csv", index=False, encoding="utf-8-sig")
    shots.to_csv(DATA / "shots.csv", index=False, encoding="utf-8-sig")
    reds.to_csv(DATA / "red_cards.csv", index=False, encoding="utf-8-sig")

    print("\n===== 요약 =====")
    print(out.groupby(["league", "season"]).size().unstack(fill_value=0))
    print(f"\n경기당 평균 슈팅 {(out['HS'] + out['AS']).mean() / 2:.1f}, "
          f"유효슈팅 {(out['HST'] + out['AST']).mean() / 2:.1f}, "
          f"골 {(out['FTHG'] + out['FTAG']).mean() / 2:.2f} (팀당)")
    print(f"골 수 불일치 경기: {mismatch.sum()} / {len(out)}")
    print(f"상세 실패 경기: {len(failed)}  (다시 실행하면 재시도)")
    print(f"슈팅 데이터 없는 경기: {len(empty)}  {empty[:10]}")


if __name__ == "__main__":
    main()
