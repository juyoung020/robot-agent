# 스킬 `explore` — 집 탐사

**목표**: 2D 지도(scenemap 점유 격자)가 닿을 수 있는 집 전체를 덮을 때까지, **어디에도 닿지 않고**(1 순위), **빠짐없이**(2 순위),
**짧은 길로**(경로 길이 최소에 가깝게), **적은 LLM 호출·토큰·시뮬 시간으로** 베이스를 움직인다. 로봇은 LIMO(차동 2륜) + OMX-F 하나이고, 탐사에는 베이스만 쓴다(앞뒤로 가고 제자리에서 돎, 옆으로 못 감). LLM(Qwen3.5-9B, KAU)은 `move_robot` 도구 하나만 쓴다.

| 구성 | 내용 |
|---|---|
| 시스템 프롬프트 | `../../prompts/common.md` + `system.md`(판 `explore-vN`), 과제 `task.md`, 이 스킬에서 보일 도구 설명 `tool.md` — 실행할 때 읽음, 판·FNV-64 지문을 `summary.json`·`decisions.jsonl` 에 남김 |
| 도구 | `move_robot` 하나(base `go_to`·`probe`, 팔·그리퍼는 쓰지 않음) — [`../../tools/README.md`](../../tools/README.md) |
| 끝 조건 | LLM 이 글로 끝냄(지도 요약 `status` 가 "no reachable frontier left") / 런타임: 호출 80 번, 시뮬 880 s(15 분 안), 벽시계 1 시간. 프런티어가 남았는데 끝내면 한 번 되물음(`nudges`) — 데이터는 `skill.json` `end`·`limits` |
| 성공 지표 | 접촉 수(0 이어야), 정답 바닥 덮음(`gt_cov`), 덮음 50/80/90/95 % 에 닿은 경로 m·시뮬 s·호출 수, 총 경로 m, LLM 호출·토큰·지연, 벽시계 |
| 기준선 | LLM 없는 Yamauchi 프런티어 탐사(같은 도구·같은 관측에서 `path_m` 가장 짧은 프런티어로 `go_to`) = `move_robot` 모드 `explore`(`max_steps` 1, 도구 코드 `tools/move_robot/src/frontier.rs`) — `skill.json` `baseline`, 기록 이름 `frontier` |
| 설정 | `skill.json`(데이터): 도구 `move_robot` 과 LLM 에게 보일 mode(`go_to`·`probe`·`delta`·`absolute` — `explore` 는 숨김), 첫 관측, 기준선, 끝 조건·되묻기, 지표 칸(`_m` 의 `gt_cov` 등), 한도 |

## 구조

```
런타임 run-skill --skill explore                평가기 프로세스 (src/sim/explore/run_explore.py, 파이썬은 바이트만)
 ├ 원형 도구 호출 루프 / 기준선(src/agent/runtime) ├ SceneMemory(libsgrt): ObjectSAM + SigLIP 2 + objprob + scenemap 격자·물체, memory/ 에 1 s 마다 저장
 ├ 결과 줄이기(move_robot llm_view), 쌍 접기       ├ keyframe(6 스텝)마다 sgrt_map(view) → mr_set_map(robot, &view)  (포인터만 넘김)
 ├ decisions.jsonl · timeline.jsonl · trace.jsonl  ├ libmove_robot(Rust): 매 스텝 닫힌 고리(go_to / probe / delta / 관절)
 └ TCP 줄 JSON ───────────────────────────────▶  ├ 정답 바닥 기준(gt_trav.py) → mr_set_reference, 접촉(RigidContactAPI) → mr_set_contacts
                                                  └ memory/explore.json(뷰어 겹침: 지나온 길·계획 경로·목표·프런티어 id)
```
- 주행 층·지도 분석·관측 만들기는 전부 Rust(`tools/move_robot/src/{map,nav,robot_nav}.rs`)이고 평가기 안 같은 실행기에서 돈다. LLM 을 기다리는 동안 로봇은 멈춰 있다(0 속도 유지).
- 지도는 scenemap 격자 하나가 유일한 출처(바뀐 영역만 비교). 새로 나타난 장애물은 격자 갱신 사이에 얇은 깊이 정지가 막는다.

## 관측(베이스 결과의 `map`) — 아는 곳 / 모르는 곳을 나눠 말한다

```json
{"free_m2":61.2,"new_free_m2":5.3,
 "frontiers":[{"id":"F1","path_m":2.4,"dir":"40L","new_area_m2":12.6,"room":"R3"}, …최대 5],
 "around":{"F":"known 2.1m then wall, depth clear 1.8m","L":"known 0.4m then unknown", …8 방향},
 "rooms":[{"id":"R1","visited":true,"path_m":3.0}, …], "in_room":"R1",
 "status":"3 reachable frontier(s); all frontier goals are in KNOWN free space (go_to works)"}
```
- 프런티어 = 계획으로 닿는 아는 빈칸 중 모르는 칸과 붙은 덩어리(0.5 m 이상). 목표점은 늘 아는 공간 안(가능하면 제자리에서 돌 수 있는 넓은 칸)이라
  `go_to` 가 언제나 계획할 수 있다. `path_m` 은 직선이 아니라 아는 빈칸 경로 길이, `new_area_m2` 는 목표 둘레 2.5 m 모르는 넓이(정보 이득).
  순서 = 정보 이득 / (경로 + 2 m), id 는 결과마다 다시 매김(F1 = 가장 좋음). 두 번 실패한 목표 둘레는 뺌(`skipped_failed`).
- `around`: 방향마다 지도(아는 빈칸이 몸통 원 기준 어디까지, 끝이 벽인가 모름인가)와 지금 깊이(카메라 시야 안일 때만). 모르는 방향으로 probe 할지 판단용.
- LLM 에게 안 보이는 측정값은 `_m`(덮음·접촉·경로·시간·µs) — 에이전트가 지우고 기록에만 남긴다.

## 두 움직임 — 언제 무엇을

| 모드 | 언제 | 하는 일 |
|---|---|---|
| `go_to` + `target: "F1"` / `"R2"` | 목표가 **아는** 빈칸 안(프런티어·방은 늘 그렇다) — 거의 모든 움직임 | 부풀린 아는 빈칸 격자 Dijkstra(벽 벌점) → DWA(사각형 몸통 검사) → 회복(다시 계획 → 돌아보기 → 되짚기 → `blocked` + 이유). 모르는 점을 주면 `error`("UNKNOWN space … use probe") |
| `probe` + `[돌기°, 앞 m]` | 프런티어 id 가 없는 **모르는** 쪽을 보거나 짧게 엿볼 때(처음 둘러보기 `[120,0]`) | 제자리에서 돌고(모서리 검사, 막히면 반대로) 0.25 m/s 로 최대 1.5 m, 지금 깊이 프레임 장애물 앞에서 멈춤 |

`decisions.jsonl` 의 `target_space`(known/unknown)와 `label.tags`(`goto_into_unknown`, `wasted_probe`, …)로 모델의 아는/모르는 판단이 맞았는지 센다.

## VLA ↔ move_robot

탐사는 베이스만이라 `move_robot` 만 쓴다. 전환 접점(`{"executor":"vla","skill":…}` → libmove_robot VLA 실행기),
쓰려면 필요한 것은 [`../../decision_log.md`](../../decision_log.md) 끝.

## 결과

덮음 = 닿을 수 있는 정답 바닥(`gt_trav.py` `.reach.pgm` 중 시작에서 몸통 부풀림 원 0.241 m 로 이어진 곳 + 둘레 0.6 m) 중 지도가 빈칸으로 덮은 비율.
접촉 = 몸통이 바닥 아닌 것에 새로 닿은 횟수(가짜 집은 LIMO 사각형 0.36 × 0.22 m 가 벽 칸에 닿기 시작한 횟수).

### 가짜 집(LIMO 몸, 출발 6 개 = 두 집 × 방향 3, 기준선 `frontier`, `devtools/mock_eval.sh`)

| 집 | 출발 | 덮음 | 접촉 | 호출 | 경로 m | 시뮬 s | 90 % 까지 경로 m |
|---|---|---|---|---|---|---|---|
| 2 층 집 아래층(turning_on_radio) | s0 / s1 / s2 | 0.997 / 0.998 / 0.999 | 0 / 0 / 0 | 15 / 16 / 15 | 35.5 / 36.9 / 35.7 | 153 / 152 / 156 | 25.2 / 26.6 / 13.2 |
| 1 층 집(bringing_water) | s0 / s1 / s2 | 0.986 / 0.986 / 0.991 | 0 / 0 / 0 | 31 / 31 / 33 | 86.2 / 81.4 / 82.9 | 392 / 348 / 394 | 76.8 / 58.2 / 73.3 |

가짜 카메라는 LIMO eyes 화각(수평 67.9°)·0.25–6 m 이고, 몸통 접촉은 사각형으로 검사한다. LLM 판(`explore-v4` 이후)과 시뮬(OmniGibson) LIMO 판은 이 몸으로 다시 재야 한다 —
시뮬 LIMO 탐사 기록은 [`../../../scene_graph/runtime/README.md`](../../../scene_graph/runtime/README.md) 의 "LIMO 탐색 결과".

## 고친 것

| 본 것 | 고친 것 |
|---|---|
| 베이스가 2 cm·2° 덜 감(허용 오차 끝에서 멈춤) | 허용 1 cm·1°, 최소 접근 속도, 0 지령 뒤 멈출 때까지 재고 다시 접근 |
| 9B 가 프런티어를 F1→F2→…→F5 차례로 고름(경로 1.6 배) | 프롬프트 v2(매번 새 목록의 F1) |
| 좁은 곳에서 못 돌아 멈춤 | DWA 뒤로·제자리 돌기, 앞·뒤 빠져나오기, 되짚기 회복, 넓은 프런티어 목표 |
| 갑자기 나타난 장애물(지도에 아직 없음) 옆에서 돌다 닿음 | 지도 밖 장애물이 몸 둘레에 있으면 돌지 않고 물러남, 얇은 깊이 정지 |
| 못 가는 방에 go_to 를 15 번 | 두 번 실패한 목표 숨김 + 프롬프트 v3 |
| 지도가 닫힌 뒤 probe 되풀이 | 프롬프트 v4 → 스스로 done |
| KAU 429 | 기다렸다 다시(10·20·… s, 6 번) |
| 가짜 집에서 첫 관측에 프런티어가 없어 바로 끝남(빈칸 4.7 m²) | 가짜 카메라 최소 거리 0.4 m 가 출발 둘레 가정 반경 0.29 m 보다 커서 몸 둘레 지도가 끊겼다 → 0.25 m. 첫 관측 호출 `values` 도 베이스 delta 형식 `[0, 0]` 으로 |

## 남은 것

- 닫힌 문 뒤 방(1 층 집 정답의 24 %)은 베이스만으로는 못 간다 — 문 열기는 VLA 실행기 몫(`decision_log.md` 전환 접점).
- 맨 끝에 못 가는 방에 두 번은 시도한다(방 목표가 좁은 곳에 잡힘). 방 목표도 프런티어처럼 넓은 칸으로 잡거나 "path through doorway too narrow" 를 미리 알려 줄 것.
- 1 층 집 복도로 들어가는 길이 판마다 열리기도 안 열리기도 한다(문틀 너비 ≈ 몸통 + 여유, 지도 잡음) — PLACES 층(병목 여유가 변 무게, `sm_snap_place_path`)으로 계획하면 판단이 쉬워질 것.
- LIMO 몸으로 시뮬 LLM 판을 다시 재야 한다(`carto` 자세, 실제 로봇과 같게).
- 판 수가 적다(집당 시뮬 1–2 판). 시드·출발을 바꿔 더 돌릴 것.

## 돌리는 법

```bash
# 이 폴더에는 코드가 없다: 루프는 런타임, 기준선·관측은 move_robot, 집계는 devtools. 저장소 루트에서
tools/build_all.sh agent                                    # run-skill, libmove_robot.so
(cd src/agent/devtools && cargo build --release -j4)       # decisions-agg (개발 명령, LLM 도구 아님)
set -a; . ~/.config/robot-agent/kau.env; set +a             # LLM 키는 환경변수로만
# 가짜 집(정답 바닥 + 카메라 흉내), Isaac Sim 없이 몇 분 (정답 바닥 src/agent/devtools/mock_gt)
src/agent/devtools/mock_eval.sh <tag> [--style compact]     # → data/explore_mock_<tag>/ (POLS=frontier 면 기준선만, STARTS=1 이면 출발 하나)
build/bin/run-skill --skill explore --policy frontier --mock <gt.pgm> --gt <gt.json> --out <dir>   # 한 판
# 시뮬(OmniGibson, 머리 RGB-D, VRAM ≥ 9 GB·RAM ≥ 16 GB 될 때까지 기다림): 정답 바닥 gt 는 src/sim/explore/gt_trav.py 로 만든다
src/sim/explore/run_explore.sh frontier bringing_water <tag>     # 또는 llm → data/outputs/explore_<ts>_…/
tools/run_explore_live.sh frontier bringing_water <tag>          # 실시간 sgview(소켓 → SSE)
src/agent/devtools/target/release/decisions-agg data/outputs/explore_*/ --by policy
```
시뮬 판 기본은 `SGRT_POSE=carto`(실제 로봇과 같음 — Cartographer: 2D 라이다 + 바퀴 오도메트리, 위치 오차까지 시험). 시뮬은 실제 로봇과 같은 방식으로 돌아야 하므로 성능·점수는 `carto` 판에서 잰다. 정답 자세(map = world)는 지도·탐사를 떼어 보는 확인용으로 `SGRT_POSE=gt` 를 줄 때만.
