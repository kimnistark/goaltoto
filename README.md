# goaltoto (골탐정 시뮬레이터)

분 단위 축구 경기 시뮬레이터. 두 팀을 넣고 수천 번 돌려서 스코어 분포를 봄.
데이터: Understat (EPL · 라리가 · 분데스리가 통합)

## 처음 세팅 (데이터 받기 → 준비 → html)
```
py download_understat.py
py prepare_params.py
py prepare_players.py
py prepare_shapes.py
py prepare_chain.py
py build_html.py
```
→ `simulator.html` 을 브라우저로 열기

## 지금 기준으로 업데이트 (살아있는 팀·선수 값)
```
py live.py                      # 팀 값: 지금까지 경기 반영 → data/live_teams.json
py prepare_players.py live      # 선수: 최근 1년 기록, 지금 뛰는 팀 (이적생 자동 반영, 스탯은 예전 팀 기록까지 가져감)
py prepare_shapes.py live       # 선발 구성: 시즌 구분 없이 최근 20경기
py prepare_chain.py live        # 공격 전개·세트피스: 최근 1년
py build_html.py                # live 파일 3개가 있으면 '지금 기준' html
```
되돌리려면 data/ 의 live_teams.json, players_live.json, chain_live.json 중 하나를 지우면 시즌 기준으로 돌아감

## 파일
| 파일 | 하는 일 |
|---|---|
| download_understat.py | Understat 경기·슈팅·선수 원본 받기 → data/ |
| prepare_params.py | 팀 값, 분당 슈팅, 시간대·홈 배수, 추가시간 표 → data/params.json |
| prepare_players.py | 선수 선발 확률, 슈팅·xG, 결정력, 골키퍼 → data/players_시즌.json |
| prepare_shapes.py | 팀별 선발 구성(수비-미드-공격 인원) 기록 → data/shapes.json |
| prepare_chain.py | 빌드업 → 패스 → 슈터 흐름, 세트피스, 자책골 → data/chain.json |
| fit_corr.py | 전력 차 보정값 찾기 → data/corr.json (현재 c1 = 0.045) |
| sim_engine.py | 시뮬 엔진 (검증 스크립트들이 공용으로 씀) |
| build_html.py | 위 값들을 박아 넣은 simulator.html 생성 |
| validate_handicap.py | 시즌 전 경기 시뮬 → 핸디 확률 보정표 검증 |
| run_seasons.py | 여러 시즌 준비+시뮬 (보정값 다시 찾을 때) |
| compare.py | 보정 전/후 비교 |
| live.py | (1단계) 살아있는 팀 값: 경기마다 갱신, 옛 경기는 덜 반영 |
| walkforward.py | (1단계) 지난 경기들을 그 이전 정보만으로 다시 예측해서 채점 + 오답노트 |
