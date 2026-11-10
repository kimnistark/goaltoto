"""
과거 시점별 선수 파일 찍기 (players_test.py 준비)
테스트할 시즌의 매주 시작일(화요일)마다, 그 날 '전' 경기만 가지고
  prepare_players.py live / prepare_shapes.py live / prepare_chain.py live 를 돌려서
  data/snap/날짜/ 에 저장
이미 찍어둔 주는 건너뜀 → 중간에 끊겨도 다시 돌리면 이어서 함

사용: py make_snaps.py          (25/26 + 26/27 지금까지)
      py make_snaps.py 2026     (26/27만)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
FILES = ("players_live.json", "shapes_live.json", "chain_live.json")

m = pd.read_csv(DATA / "matches.csv")
m["dt"] = pd.to_datetime(m["datetime"])
latest = int(m["season"].max())
tests = [int(a) for a in sys.argv[1:] if a.isdigit()] or [latest - 1, latest]
weeks = sorted(m[m["season"].isin(tests)]["dt"].dt.to_period("W-MON").dt.start_time.unique())
todo = [w for w in weeks if not all((DATA / "snap" / f"{w:%Y-%m-%d}" / f).exists() for f in FILES)]
print(f"시즌 {tests}: 경기 있는 주 {len(weeks)}개 · 이미 찍은 주 {len(weeks) - len(todo)}개 · 남은 주 {len(todo)}개\n")

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
t0 = time.time()
for i, w in enumerate(todo, 1):
    tag = f"{w:%Y-%m-%d}"
    env["GT_ASOF"] = tag
    log = DATA / "snap" / tag
    log.mkdir(parents=True, exist_ok=True)
    for script in ("prepare_players.py", "prepare_shapes.py", "prepare_chain.py"):
        r = subprocess.run([sys.executable, str(HERE / script), "live"], cwd=HERE, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        (log / f"{script}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            print(f"  [{tag}] {script} 에러 → data/snap/{tag}/{script}.log 확인")
            print("  " + "\n  ".join(r.stderr.strip().splitlines()[-5:]))
            sys.exit(1)
    el = time.time() - t0
    print(f"  {i}/{len(todo)}  {tag} 기준 완료 · {el / 60:.1f}분 지남 · 남은 시간 약 {el / i * (len(todo) - i) / 60:.0f}분")
print(f"\n끝: data/snap/ 에 {len(weeks)}주치 선수 파일. 다음: py players_test.py")
