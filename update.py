"""
새로 받은 골탐정 파일 정리 (다운로드 폴더 = 바탕화면 기준)
- 'sim_engine (1).py' 처럼 번호 붙어서 저장된 것 중 가장 최근 것을 원래 이름(sim_engine.py)으로 덮어쓰고
  나머지 번호 붙은 파일은 삭제
사용: py update.py
      py update.py "C:\다른\다운로드폴더"   (다운로드 폴더가 따로 있으면 거기서 가져옴)
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEST = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DEST
KNOWN = {
    "download_understat", "download_sofascore", "prepare_params", "prepare_players", "prepare_shapes", "prepare_chain",
    "build_html", "sim_engine", "validate_sim", "validate_handicap", "update",
    "fit_corr", "run_seasons", "compare",
}   # 새 골탐정 파일이 생기면 여기에 이름 추가
pat = re.compile(r"^(.*?)(?: \(\d+\))?\.py$")

if not SRC.exists():
    sys.exit(f"다운로드 폴더를 못 찾았어: {SRC}\n  py update.py \"폴더경로\" 로 직접 지정해줘")

groups = {}
for f in SRC.glob("*.py"):
    m = pat.match(f.name)
    if not m:
        continue
    base = m.group(1)
    if base in KNOWN:                 # 골탐정 파일 목록에 있는 것만 (다른 파일은 절대 안 건드림)
        groups.setdefault(base, []).append(f)


changed = 0
for base, files in sorted(groups.items()):
    target = DEST / f"{base}.py"
    newest = max(files, key=lambda x: x.stat().st_mtime)
    others = [f for f in files if f.resolve() != target.resolve()]
    if not others:
        continue                      # 번호 붙은 새 파일이 없음
    if newest.resolve() != target.resolve():
        shutil.copy2(newest, target)
        print(f"  덮어씀: {base}.py  ← {newest.name}  "
              f"{datetime.fromtimestamp(target.stat().st_mtime):%m/%d %H:%M}")
        changed += 1
    for f in others:
        if f.exists() and f.resolve() != target.resolve():
            f.unlink()
print(f"\n완료: {changed}개 덮어씀" if changed else "새로 받은 파일이 없어")
