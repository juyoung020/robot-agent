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
cargo test --release                      # 단위 시험 19개 (Isaac Sim 없이)
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
