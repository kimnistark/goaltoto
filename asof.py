"""
과거 시점 기준으로 선수 파일 만들기용 (prepare_players / prepare_shapes / prepare_chain 이 같이 씀)
환경변수 GT_ASOF=2025-10-14 가 있으면:
  - matches.csv 에서 그 날짜 '전' 경기만 읽음 (그 날 이후 경기는 없는 것처럼)
  - 결과 파일은 data/snap/2025-10-14/ 에 저장 (평소 data/ 파일은 안 건드림)
없으면 평소랑 똑같음. 직접 쓸 일은 없고 make_snaps.py 가 알아서 씀
"""
import os
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
ASOF = os.environ.get("GT_ASOF")
OUTDIR = DATA / "snap" / ASOF if ASOF else DATA
if ASOF:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def read_matches(**kw):
    m = pd.read_csv(DATA / "matches.csv", **({k: v for k, v in kw.items() if k != "usecols"}))
    if ASOF:
        m = m[pd.to_datetime(m["datetime"]) < pd.Timestamp(ASOF)]
    if "usecols" in kw:
        m = m[kw["usecols"]]
    return m
