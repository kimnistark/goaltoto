"""
여러 시즌을 차례로 준비 + 시뮬해서 시즌별 '경기별 시뮬 평균 골'을 저장 (보정값 찾기용)
시즌마다: 팀 값 → 선수 값 → 선발 구성 → 공격 전개·세트피스 → 전 경기 시뮬(경기당 1000번)
결과: data/sim_means_2023.csv, sim_means_2024.csv, sim_means_2025.csv
마지막에 2025/26으로 끝나므로, 다 돌고 나면 data 폴더는 지금과 같은 25/26 상태로 돌아옴
사용: py run_seasons.py            (23/24, 24/25, 25/26)
      py run_seasons.py 2022 2023 2024 2025   (시즌을 직접 지정. 마지막은 2025로 둘 것)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}   # 윈도우에서 선수 이름의 특수문자(é, ú 등) 출력 오류 방지
seasons = [int(a) for a in sys.argv[1:]] or [2023, 2024, 2025]
STEPS = [("팀 값", "prepare_params.py", True), ("선수 값", "prepare_players.py", True), ("선발 구성", "prepare_shapes.py", False),
         ("공격 전개·세트피스", "prepare_chain.py", False), ("전 경기 시뮬", "validate_handicap.py", False)]
for s in seasons:
    print(f"\n===== {s}/{(s + 1) % 100} 시즌 =====")
    for label, script, with_season in STEPS:
        t = time.time()
        print(f"  {label} ...", end=" ", flush=True)
        r = subprocess.run([sys.executable, str(HERE / script)] + ([str(s)] if with_season else []),
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=ENV)
        if r.returncode != 0:
            print("실패\n" + r.stderr.decode("utf-8", "replace")[-1500:])
            sys.exit(f"{script} 에서 멈춤 ({s} 시즌)")
        print(f"완료 ({time.time() - t:.0f}초)")
    print(f"  → data/sim_means_{s}.csv 저장")
print("\n끝. 이제 py check_fit.py 를 돌리면 돼")
