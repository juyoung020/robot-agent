# 스킬 `explore` — 집 탐사

**목표**: 2D 지도(scenemap 점유 격자)가 닿을 수 있는 집 전체를 덮을 때까지, **어디에도 닿지 않고**(1 순위), **빠짐없이**(2 순위),
**짧은 길로**(경로 길이 최소에 가깝게), **적은 LLM 호출·토큰·시뮬 시간으로** 베이스를 움직인다. LLM(Qwen3.5-9B, KAU)은 `move_robot` 도구 하나만 쓴다.

| 구성 | 내용 |
|---|---|
| 시스템 프롬프트 | `../../prompts/common.md` + `system.md`(판 `explore-vN`), 과제 `task.md`, 이 스킬에서 보일 도구 설명 `tool.md` — 실행할 때 읽음, 판·FNV-64 지문을 `summary.json`·`decisions.jsonl` 에 남김 |
| 도구 | `move_robot` 하나(base `go_to`·`probe`, 나머지 부분은 쓰지 않음) — [`../../tools/README.md`](../../tools/README.md) |
| 끝 조건 | LLM 이 글로 끝냄(지도 요약 `status` 가 "no reachable frontier left") / 실행기: 호출 80 번, 시뮬 880 s(15 분 안), 벽시계 1 시간. 프런티어가 남았는데 끝내면 한 번 되물음(`nudges`) |
| 성공 지표 | 접촉 수(0 이어야), 정답 바닥 덮음(`gt_cov`), 덮음 50/80/90/95 % 에 닿은 경로 m·시뮬 s·호출 수, 총 경로 m, LLM 호출·토큰·지연, 벽시계 |
| 기준선 | LLM 없는 Yamauchi 프런티어 탐사(같은 도구·같은 관측에서 `path_m` 가장 짧은 프런티어로 `go_to`) |

## 구조

```
explore (Rust, 이 폴더)                         평가기 프로세스 (behavior-2026/src/sim/explore/run_explore.py, 파이썬은 바이트만)
 ├ 원형 도구 호출 루프 / 기준선                   ├ SceneMemory(libsgrt): YOLOE + scenemap 격자·물체, memory/ 에 1 s 마다 저장
 ├ 결과 줄이기(compact), 오래된 쌍 한 줄 요약       ├ keyframe(6 스텝)마다 sgrt_map(view) → mr_set_map(robot, &view)  (포인터만 넘김)
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

## VLA(π0.5) ↔ move_robot

탐사는 베이스만이라 `move_robot` 만 쓴다. 전환 접점(`{"executor":"vla","skill":…}` → 평가기가 π0.5 엔진으로 `act()` 를 넘김)과 지금 상태(`unavailable`),
쓰려면 필요한 것은 [`../../decision_log.md`](../../decision_log.md) 끝.

## 결과

(아래 "결과 표" 절 — 가짜 집 반복 실험과 시뮬 판)

## 돌리는 법

```bash
cd src/agent/skills/explore && cargo build --release           # explore, decisions-agg (move_robot·planner llm.rs 경로 의존)
(cd ../../tools/move_robot && cargo build --release && cargo test --release)   # libmove_robot.so, 시험 26 개
set -a; . ~/.config/behavior-2026/kau.env; set +a              # 키는 환경변수로만
# 가짜 집(정답 바닥 + 카메라 흉내), Isaac Sim 없이 몇 분
(cd ../../../behavior-2026/src/sim/explore && python gt_trav.py --task bringing_water && python gt_trav.py --task turning_on_radio)
./mock_eval.sh <tag> [--style compact]                          # → behavior-2026/outputs/explore_mock_<tag>/
# 시뮬(OmniGibson, 머리 RGB-D, VRAM ≥ 9 GB·RAM ≥ 16 GB 될 때까지 기다림)
cmake -S ../../../behavior-2026/src/scene_graph/runtime -B ~/sgrt_build_explore && cmake --build ~/sgrt_build_explore -j
../../../behavior-2026/src/sim/explore/run_explore.sh frontier bringing_water <tag>    # 또는 llm
../../../behavior-2026/src/sim/explore/viewer_8080.sh <run dir>                        # 8080 뷰어를 이 판으로(하나만)
./target/release/decisions-agg ../../../behavior-2026/outputs/explore_*/ --by policy
```
시뮬 판은 `SGRT_POSE=gt`(정답 자세, map = world — 측정·시험용)로 돈다. 실제 로봇 기본은 `slam`.
