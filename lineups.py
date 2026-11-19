"""
실제 선발 읽기 (Understat 캐시 data/cache/경기.json 의 rosters) — players_test.py / fair_eval.py 가 같이 씀
  {팀: (선발 pid 집합, (수비, 미드, 공격) 인원)}. 선수 파일에 있는 선수는 그 파일 포지션으로 셈
"""
import json
from pathlib import Path

CACHE = Path(__file__).parent / "data" / "cache"


def group(pos):
    if pos == "GK": return "GK"
    if pos.startswith("D") and pos != "DMC": return "DEF"
    if pos.startswith("FW"): return "FWD"
    return "MID"


def actual_lineups(mid, home, away, PL):
    f = CACHE / f"{mid}.json"
    if not f.exists():
        return {}
    ro = json.loads(f.read_text(encoding="utf-8")).get("rosters")
    if not isinstance(ro, dict):
        return {}
    out = {}
    for side, team in (("h", home), ("a", away)):
        x = ro.get(side) or {}
        known_g = {p["pid"]: p["group"] for p in PL.get(team, [])}
        pids, cnt = set(), {"DEF": 0, "MID": 0, "FWD": 0}
        for p in (x.values() if isinstance(x, dict) else x):
            if not isinstance(p, dict) or p.get("position") in (None, "Sub"):
                continue
            pid = int(p["player_id"]); pids.add(pid)
            g = known_g.get(pid) or group(p["position"])
            if g in cnt:
                cnt[g] += 1
        if sum(cnt.values()) == 10:
            out[team] = (pids, (cnt["DEF"], cnt["MID"], cnt["FWD"]))
    return out
