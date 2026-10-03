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

## 결과 (2026-10-03)

덮음 = 닿을 수 있는 정답 바닥(`gt_trav.py` `.reach.pgm`: 과제 방 중 정원 빼고, 닫힌 문은 막힘, 시작에서 몸통 0.37 m 원으로 이어진 곳 + 둘레 0.6 m) 중
지도가 빈칸으로 덮은 비율. 접촉 = 로봇 링크가 바닥 아닌 것에 새로 닿은 횟수(시뮬 RigidContactAPI, 가짜 집은 몸통 원이 벽 칸에 닿기 시작한 횟수).
경로 = base_qvel 적분 이동 거리. 시뮬은 `SGRT_POSE=gt`, 머리 RGB-D 720², 평가기 한 스텝 93–165 ms(실시간의 0.2–0.35 배).

### 시뮬(OmniGibson, public_test instance 0, 같은 출발)

| 집 / 정책 | 판 | 접촉 | 덮음 | 50 % 에 | 80 % 에 | 90 % 에 | 끝 | 경로 m | 시뮬 s | 호출 | LLM 호출 | 토큰 | 벽시계 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 층 집(bringing_water) 기준선 | `explore_20261003_093438_…frontier_reach` | **0** | 0.872 | 15.3 m / 80 s | 33.0 m / 150 s | — | 프런티어 없음 | 46.3 | 204 | 11 | 0 | 0 | 619 |
| 1 층 집 LLM `explore-v4` | `…094722_…llm_v4` | **0** | **0.953** | 15.2 m / 81 s | 37.4 m / 186 s | 46.2 m / 212 s | LLM done | 75.2 | 379 | 14 | 15 | 41.6 k | 1160 |
| 1 층 집 LLM `explore-v3` | `…085903_…llm_v3` | 0 | 0.950 (최고) | 14.9 m / 109 s | 36.9 m / 207 s | 44.9 m / 230 s | 사람이 끊음(probe 되풀이) | 74.7 | 445 | 20 | 20 | 55.9 k | 1326 |
| 1 층 집 LLM `explore-v2` | `…082657_…llm_c37` | 0 | 0.654 | 16.5 m / 98 s | — | — | 사람이 끊음(R5 15 번) | 58.5 | 546 | 25 | 25 | 69.6 k | 1599 |
| 2 층 집 아래층(turning_on_radio) 기준선 | `…092443_…frontier_c37` | 0 | 0.925 | 2.8 m / 18 s | 11.6 m / 51 s | 14.5 m / 64 s | 프런티어 없음 | 14.5 | 64 | 6 | 0 | 0 | 167 |
| 2 층 집 아래층 LLM `explore-v4` | `…092910_…llm_v4` | 0 | 0.941 | 2.8 m / 23 s | 12.4 m / 65 s | 16.4 m / 85 s | LLM done | 16.5 | 89 | 8 | 9 | 21.9 k | 197 |
| (참고) 1 층 집 기준선, 사각형 몸통 | `…074931_…frontier_v2` | **13**(바퀴–소파·탁자) | 0.537* | | | | | 39.2 | 154 | 11 | | | 637 |

\* 이 판은 방 안 바닥 전부(닫힌 문 뒤 방 포함) 분모. 같은 판을 닿을 수 있는 분모로 다시 재면 원 몸통 기준선(`…080349`)은 0.867.

- **LLM 만으로 집 탐사가 된다**: 두 집 모두 접촉 0, 덮음 0.94–0.95 로 기준선(0.87–0.93)보다 높다. 1 층 집에서 기준선은 가장 가까운 프런티어만 따라가다
  복도 끝 쪽을 남기고 프런티어가 0 이 됐고, LLM 은 "정보 이득 / (경로 + 2 m)" 1 등(F1)을 골라 넓은 쪽을 먼저 열어 그 쪽을 덮었다.
- 경로: 90 % 덮음까지 LLM 46.2 m(1 층)·16.4 m(2 층) — 1 층은 기준선이 90 % 에 못 닿아 비교 불가, 80 % 까지는 LLM 37.4 m vs 기준선 33.0 m(1.13 배),
  2 층 90 % 까지 16.4 vs 14.5 m(1.13 배). 끝까지의 총 경로는 LLM 이 길다(덮은 넓이가 더 넓고, 못 가는 방에 두 번 시도).
- 효율: LLM 한 번 4.6–5.3 s(KAU), 요청 하나 약 2.7 k 토큰(16 k 의 1/6, 오래된 호출은 한 줄 요약), 판당 9–15 번. 벽시계는 시뮬(0.2–0.35 배 실시간)이 지배.
- 실행기 시간(시뮬, `timing.jsonl`): 스텝마다 물체 기억 2–4.5 ms, 지도 넘기기 0.5–0.9 ms(keyframe 당 3–5.5 ms: 격자 복사·거리장·바뀐 칸), 닫힌 고리 0.2–5.5 ms(DWA 중),
  관측 만들기 1–2.5 ms, 다시 계획 0.4–1.4 ms. 평가기 스텝 93–165 ms 대비 실행기 몫 3–6 %.

### 가짜 집(정답 바닥 + 카메라 흉내, 출발 6 개 = 두 집 × 방향 3, `mock_eval.sh final`)

| 정책 | 덮음 평균 | 접촉 | 경로 m 평균 | 90 % 까지 경로 평균 | 호출 | LLM 호출 | 토큰 | 벽시계 s |
|---|---|---|---|---|---|---|---|---|
| 기준선 | 0.971 | 0 | 44.6 | 37.8 | 13.3 | 0 | 0 | 21 |
| LLM `explore-v2` | 0.976 | 0 | 48.9 | 38.1 (기준선의 1.01 배) | 12.7 | 13.7 | 34.8 k | 82 |
| LLM `explore-v3` | 0.980 | 0 | 53.4 | 39.5 | 13.3 | 14.3 | 37.0 k | 93 |

### 결정 기록 집계(시뮬 판 전부, `decisions-agg --by policy`)

| 정책·모드 | 결정 | reached | blocked | error | 그 밖(timeout) | 평균 이동 m | 평균 새 빈칸 m² | m² / m |
|---|---|---|---|---|---|---|---|---|
| 기준선 go_to | 44 | 84 % | 14 % | 0 % | 2 % | 3.39 | 9.78 | 2.89 |
| LLM go_to | 61 | 52 % | 0 % | 3 % | 44 % | 3.65 | 6.21 | 1.70 |
| LLM probe | 6 | 0 % | 83 % | 0 % | 17 % | 0.33 | −0.03 | — |

LLM go_to 의 timeout 대부분은 v2 판의 R5 되풀이(15 번)와 지도가 닫힌 뒤 못 가는 방(v3·v4 각 2 번). probe 는 v3·v4 판에서 지도가 닫힌 뒤에만 썼고 전부 헛걸음(`wasted_probe`)
— 9B 는 아는/모르는 판단을 go_to(아는 곳)로만 했고 모르는 곳으로 go_to 하려다 거절된 일(`goto_into_unknown`)은 0.

## 실패 → 고친 것

| 본 것 | 고친 것 |
|---|---|
| 베이스가 2 cm·2° 덜 감(허용 오차 끝에서 멈춤) | 허용 1 cm·1°, 최소 접근 속도, 0 지령 뒤 멈출 때까지 재고 다시 접근 |
| 9B 가 프런티어를 F1→F2→…→F5 차례로 고름(경로 1.6 배) | 프롬프트 v2(매번 새 목록의 F1) |
| 원 0.30 m 몸통: 시뮬에서 바 의자·소파에 11 번 접촉 | 사각형 → (바퀴가 밖이라 13 번 접촉) → 반지름 0.37 m 원 + 보수적 거리 + 속도 비례 여유: 접촉 0 |
| 좁은 곳에서 못 돌아 멈춤 | DWA 에 옆 속도·뒤·미끄러지기, 되짚기 회복, 넓은 프런티어 목표 |
| 갑자기 나타난 장애물(지도에 아직 없음) 옆에서 돌다 닿음 | 지도 밖 장애물이 몸 둘레에 있으면 돌지 않고 물러남, 얇은 깊이 정지 |
| 못 가는 방에 go_to 를 15 번 | 두 번 실패한 목표 숨김 + 프롬프트 v3 |
| 지도가 닫힌 뒤 probe 되풀이 | 프롬프트 v4 → 스스로 done |
| KAU 429 | 기다렸다 다시(10·20·… s, 6 번) |

## 남은 것

- 닫힌 문 뒤 방(1 층 집 정답의 24 %)은 베이스만으로는 못 간다 — 문 열기는 π0.5(VLA) 실행기 몫(`decision_log.md` 전환 접점).
- 맨 끝에 못 가는 방에 두 번은 시도한다(방 목표가 좁은 곳에 잡힘). 방 목표도 프런티어처럼 넓은 칸으로 잡거나 "path through doorway too narrow" 를 미리 알려 줄 것.
- 1 층 집 복도로 들어가는 길이 판마다 열리기도 안 열리기도 한다(문틀 너비 ≈ 몸통 + 여유, 지도 잡음) — PLACES 층(병목 여유가 변 무게, `sm_snap_place_path`)으로 계획하면 판단이 쉬워질 것.
- 시뮬은 `SGRT_POSE=gt` 로 쟀다. `slam` 자세(실제 로봇 기본)에서 같은 지표를 다시 재야 한다(시뮬 odom wz 가 약 40 % 크다는 보고).
- 판 수가 적다(집당 시뮬 1–2 판). 시드·출발을 바꿔 더 돌릴 것.

## 돌리는 법

```bash
cd src/agent/skills/explore && cargo build --release           # explore, decisions-agg (move_robot·planner llm.rs 경로 의존)
(cd ../../tools/move_robot && cargo build --release && cargo test --release)   # libmove_robot.so, 시험 29 개
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
시뮬 판은 `SGRT_POSE=gt`(정답 자세, map = world — 측정·시험용)로 돈다. 실제 로봇 기본은 `slam`. 시뮬은 실제 로봇과 같은 방식으로 돌아야 하므로 `gt` 는 지도·탐사를 떼어 보는 확인용이고, 성능·점수는 `slam` 판에서 잰다([계획](../../../../docs/plan.md)).
