# agent/tools

LLM 이 골라 부르는 도구를 둔다. 설계는 [`../plan.md`](../plan.md) 3.3절.

원칙: id 로 말하고 좌표로 말하지 않는다, 인자는 enum·필수로 좁힌다, 결과는 짧은 JSON + `hint`, 실패는 오류 관찰값(예외로 루프를 죽이지 않음). LLM 에 보이는 도구는 8개 이하.

| 도구 | 하는 일 |
|---|---|
| `find_object` | 물체 기억에서 물체 찾기 (방·자리·마지막으로 본 시각·상태) |
| `describe_object` | 물체 크기·관측 수·움직인 거리, 선택하면 best view 사진 |
| `list_place` | 방이나 가구 위·안의 물체 목록 |
| `set_plan` | 스킬 단계 계획을 검증하고 π0.5 에게 줄 문장으로 바꾸기 |
| `check` | 보이는지·잡았는지·놓였는지·도착했는지 확인 |
| `ask_user` | 사용자에게 되묻기 (턴 끝냄) |
| `report` | 진행·완료·실패 알리기 (턴 끝냄) |
| `move_robot` | 로봇 한 부분(베이스·몸통·팔·그리퍼)을 직접 움직이고 멈출 때까지 기다리기 — [`move_robot/`](move_robot/) |

## `move_robot` — 관절·베이스 직접 움직이기

π0.5 를 거치지 않고 LLM 이 로봇을 직접 움직이는 도구 하나. 하는 일은 이것뿐이다: 한 부분을 목표까지 안전하게 움직이고 결과를 짧게 돌려준다.
(plan.md 3.3 은 "이동·스킬 실행 도구는 LLM 에 주지 않는다"가 기본이다. 이 도구는 시연·디버깅·π0.5 가 못 하는 작은 보정용이고, 본 경로에 넣을지는 결정 필요.)

### 스키마 (9B 모델용으로 작게, 정의 JSON 약 1 KB)

```json
{"part": "base | torso | left_arm | right_arm | left_gripper | right_gripper",
 "mode": "delta | absolute",
 "values": [number, ...],
 "duration_s": number (선택)}
```
필수: `part`, `mode`, `values`. 부분마다 값 개수가 정해져 있다.

| part | values (순서·단위) | 행동 벡터 칸 (R1Pro 23) | 제어기 (`omnigibson/eval/r1pro.yaml`) |
|---|---|---|---|
| `base` | `[앞 m, 왼쪽 m, 왼쪽으로 돌기 °]` — 호출할 때의 로봇 기준, **delta 만** | 0..3 `vx, vy, wz` | 속도, [-1,1] → ±0.75 m/s, ±0.75 m/s, ±1 rad/s |
| `torso` | `[j1..j4]` ° | 3..7 | 절대 관절 위치 (rad) |
| `left_arm` | `[j1..j7]` ° (어깨 → 손목) | 7..14 | 절대 관절 위치 |
| `left_gripper` | `[벌림]` 0 = 닫힘 … 1 = 열림 (absolute 권장) | 14 | smooth, [-1,1] → 손가락 0…0.05 m |
| `right_arm` | `[j1..j7]` ° | 15..22 | 절대 관절 위치 |
| `right_gripper` | `[벌림]` | 22 | smooth |

- 행동·관측 칸은 BEHAVIOR-1K `eval_utils.py` 의 `ACTION_QPOS_INDICES` / `PROPRIOCEPTION_INDICES` 그대로. 머리(카메라)는 평가 설정에서 `NullJointController` 라 움직일 수 없어 enum 에 없다.
- `delta` 는 지금 측정값에 더하기, `absolute` 는 그 값으로 가기. **`delta` 에 0 만 주면 움직이지 않고 상태만 읽는다**(도구를 늘리지 않으려고).
- 한 번에 한 부분. 그동안 나머지 부분은 마지막 목표를 유지하고 베이스는 0 속도.

### 결과 (짧은 JSON)

```json
{"status":"reached","part":"left_arm","state":[0.0,20.0,0.0,-30.0,0.0,0.0,0.0],"target":[…],"units":"deg","steps":38,"time_s":1.27}
{"status":"blocked","part":"right_gripper","state":[0.4],"error":0.6,"hint":"gripper stopped before closing: probably holding an object",…}
{"status":"error","message":"left_arm needs exactly 7 values (deg), got 2","hint":"fix the arguments and call move_robot again"}
```
- `status`: `reached`(허용 오차 안) / `blocked`(멈췄는데 목표 못 감: 접촉·물체·장애물) / `timeout`(아직 가는 중) / `error`(인자 틀림 — 예외가 아니라 관찰값).
- `clamped`: 한계 밖이라 잘린 칸 번호, 관절이면 `limits`(그 칸의 허용 범위 °)도 — 부호를 고칠 수 있게. `slowed`: 요청한 `duration_s` 가 안전 속도보다 빨라서 늘렸다.
- 베이스 `state` 는 이번 호출 동안 움직인 양(base_qvel 적분 오도메트리, 시뮬 정답 자세는 안 씀), `error` 는 `[남은 m, 남은 °]`.

### 안전 한계 (`Safety::default`, `src/lib.rs`)

| 항목 | 값 |
|---|---|
| 관절 한계 | URDF `r1pro.urdf` 한계 안쪽 2° 로 자름 |
| 팔 / 몸통 속도 | 45 °/s / 20 °/s (최소 저크 보간 최고 속도 기준) |
| 그리퍼 | 1 초에 끝까지 (비율 1.0/s) |
| 베이스 | 0.3 m/s, 35 °/s, 가속 0.6 m/s²·90 °/s², 한 번에 축마다 2 m·180° 까지 |
| 막힘 판정 | 관절: 지령과 실제 차이 20° 넘게 0.2 s → 멈추고 그 자리 유지 / 보간 끝난 뒤 정지 0.27 s. 베이스: 지령의 20 % 미만으로 1 s |
| 도착 허용 | 관절 1.5°, 그리퍼 0.05, 베이스 2 cm·2° |
| 시간 초과 | 관절: 보간 + 1.5 s, 베이스: 예상 시간 × 1.5 + 3 s |

그리퍼가 닫다가 막히면(물체를 쥠) 지령은 그대로 둔다(계속 쥔다). 팔이 막히면 측정 위치로 지령을 바꿔 더 밀지 않는다.

### 실행 경로

```
LLM (Qwen3.5-9B, raw tool loop) ── tool_call ──▶ move_robot::link::run_tool ── 먼저 인자 검사(왕복 없이 오류 돌려줌)
     │                                              │
     │                                              └─ TcpSim ─ JSON 한 줄 ─▶ [평가기 프로세스]
     │                                                   behavior-2026/src/sim/move_robot/move_robot_sim.py (ctypes)
     │                                                   매 스텝: proprio 61 → mr_tick (libmove_robot.so, 같은 Rust 실행기) → 행동 23
     ◀──────────────────── 결과 JSON 한 줄 ◀─────────────────┘
```
- 닫힌 고리 실행기(검증·자르기·보간·판정)는 Rust 한 곳(`move_robot/src/lib.rs`). 시뮬 쪽 파이썬은 바이트만 옮긴다(numpy·torch 없어도 됨).
- 에이전트 쪽 의존성: `serde_json` 하나. `--features llm` 일 때만 기존 계획기 `llm.rs`(KAU HTTPS, curl)를 경로 의존성으로 쓴다.

### 쓰는 법

```bash
cd src/agent/tools/move_robot
cargo test --release                      # 단위 시험 26개 (Isaac Sim 없이)
cargo build --release --features llm      # libmove_robot.so + move-robot 명령
./target/release/move-robot schema
./target/release/move-robot mock '{"part":"left_arm","mode":"delta","values":[0,20,0,-30,0,0,0]}'   # 가짜 로봇
# 시뮬: 평가기 안에서 도구 호출을 받는다 (behavior-2026)
python src/behavior-2026/src/sim/move_robot/run_eval_move.py --listen 127.0.0.1:8771 -- --task-name turning_on_radio --max-steps 6000 --headless
./target/release/move-robot call '{"part":"base","mode":"delta","values":[0.3,0,0]}'
set -a; . ~/.config/behavior-2026/kau.env; set +a   # 키는 환경변수로만
./target/release/move-robot llm "오른쪽 그리퍼를 반쯤 닫고 앞으로 50cm 가" [--mock]
python3 src/behavior-2026/src/sim/move_robot/test_move_robot_sim.py   # 시뮬 접착부 시험 (Isaac Sim 없이)
```

## 2026-10-03 변경: 탐사용 베이스 모드·지도 관측·안전 정지 (스킬 [`explore`](../skills/explore/))

스키마(약 1.5 KB): `mode` enum 이 `go_to | probe | delta | absolute`, `target`(go_to 의 id) 추가, 필수는 `part`, `mode` 만.

| 베이스 모드 | values / target | 하는 일 |
|---|---|---|
| `go_to` | `target`: 지난 지도 요약의 `F1`(프런티어)·`R2`(방) — 또는 values `[앞 m, 왼쪽 m]`(아는 빈칸이어야 함) | **아는 빈칸만** 지나는 경로(부풀린 격자 Dijkstra)를 지역 제어기(DWA)가 따라감. 도착하면 프런티어의 모르는 쪽을 봄. 모르는 점·장애물 점이면 `error` + 고칠 문장 |
| `probe` | `[왼쪽으로 돌기 °, 앞 m]`(앞 ≤ 1.5 m) | 제자리에서 돌고 0.25 m/s 로 천천히 앞으로. 지금 깊이 프레임·지도 장애물 앞(몸통 원 + 12 cm)에서 멈춤 → `blocked` + `stopped_by`, `clear_m`. 앞 0 = 돌아보기만 |
| `delta` | `[앞, 왼쪽, 돌기°]` | 예전과 같음 + 지도가 있으면 진행 방향 안전 정지(지도 장애물·카메라 밖 모르는 곳·깊이) |

주행 층(`src/nav.rs`, `src/robot_nav.rs`, Nav2 와 같은 층 구조, ROS 없음)
- 비용 지도 = scenemap 2D 점유 격자 하나(로그 오즈라 치운 물체는 광선이 비움). 바뀐 칸은 scenemap 의 바뀐 영역(`sgrt_map_view.dirty_box`) 안에서만 비교,
  거리장은 장애물이 바뀔 때만 다시. 물체 기억의 옮길 수 있는 물체 둘레는 계획 벌점.
- 전역 경로: 바뀐 칸이 경로 통로(몸통 + 0.3 m)에 걸리고 실제로 막혔을 때, 또는 3 s 마다 다시 계획(ms 단위).
- 지역 제어: 30 Hz 마다 DWA(속도 6 × 회전 13 표본, 1.2 s 굴림, 몸통 원 0.30 m 충돌 검사, 멈춤 거리 v²/2a). 가야 할 쪽이 30° 넘게 옆이면 제자리 돌기.
- 얇은 안전 정지: 마지막 깊이 프레임 장애물 점 중 **지도에 아직 없는** 것(새로 나타남·발밑·움직임)으로 앞 속도를 줄이고 멈춤.
- 회복: 다시 계획 → 35° 돌아보기 → 0.25 m 뒤로 → `blocked`("path blocked (…)"). 밀었는데 안 움직이면(접촉) 0.15 m 물러난 뒤 `blocked`.
  두 번 실패한 목표 둘레 프런티어는 다음 관측에서 뺌(`skipped_failed`).

베이스 결과에 붙는 것
- `map`(LLM 관측): `free_m2`, `new_free_m2`, `frontiers`[`id`, `path_m`(아는 빈칸 경로 길이), `dir`(`ahead`·`40L`·`75R`·`back`), `new_area_m2`(둘레 2.5 m 모르는 넓이), `room`],
  `around`(8 방향 `F, FL, L, BL, B, BR, R, FR`: "known 2.1m then wall|unknown, depth clear 1.8m"), `rooms`[`id`, `visited`, `path_m`], `in_room`, `status`.
- `_m`(측정, 에이전트가 LLM 에게서 뺌): `gt_cov`, `free_m2`, `path_m`(오도메트리 누적), `sim_s`, `contacts`, `stalls`, `blocked`, `replans`, `min_clear_m`, `obs_us`, `plan_us`, `costmap_us`, `pose`.

베이스 덜 감 고침: 허용 오차 2 cm·2° → 1 cm·1°, 남은 거리가 있으면 최소 접근 속도(3 cm/s, 3°/s), 도착 뒤 0 지령으로 멈출 때까지 기다려 실제 자리를 재고
2 배 오차 밖이면 다시 접근(최대 2 번). 가짜 로봇 시험: 0.2 m → 오차 < 12 mm, 30° → < 1.2°.

속도: go_to 0.5 m/s·50°/s, probe 0.25 m/s, delta 0.3 m/s(예전 그대로). 가감속 0.6 m/s²·90°/s².

C ABI 추가(`include/move_robot.h`): `mr_set_map(r, &sgrt_map_view)`(평가기 접착부가 sgrt 구조체 포인터를 그대로 넘김), `mr_set_reference`(정답 바닥, 측정용),
`mr_set_contacts`, `mr_overlay_json`(뷰어 겹침). 가짜 집 `link::MockWorld`(정답 바닥 PGM + 머리 카메라 흉내 ±49.6°, 6 m, 로그 오즈 격자, 몸통 원 접촉).
시험 26개(`cargo test --release`): 지도 요약, 프런티어 go_to 무접촉, probe 벽 앞 정지, 모르는 점 go_to 거절, 옆(안 보이는 곳) delta 정지, 덜 감, 길에 나타나는 장애물.
