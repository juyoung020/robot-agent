# agent/tools

LLM 이 골라 부르는 도구 중 **코드가 있는 것**만 여기 적는다. 도구 = 실행되는 코드(크레이트 하나), 스킬([`../skills/`](../skills/README.md)) = 지시문뿐,
루프는 에이전트 런타임([`../runtime/`](../runtime/README.md)).

원칙: 대상은 id 로 말한다(`O12`·`R2`·`F1`), 인자는 enum·필수로 좁힌다, 결과는 짧은 JSON + `hint`, 실패는 오류 관찰값
`{"status":"error","message","hint"}`(예외로 루프를 죽이지 않음). LLM 이 못 보는 측정값은 결과의 `_m` 에 두고 런타임이 지운다.

| 도구 | 하는 일 | 폴더 |
|---|---|---|
| `search_objects` | 물체 기억에서 찾기: ① 이름·동의어·상위어 → 없거나 약하면 ② 이름 무시 생김새 재검색(벡터는 도구 안, 결과는 글) | [`search_objects/`](search_objects/) |
| `confirm_object` | 확인된 물체의 이름 고치기(이름 사후 베이즈 갱신 + 확인 기록) | [`search_objects/`](search_objects/) |
| `list_place` | 방의 물체, 또는 가구 id 상자에서 1.5 m 안 물체(거리·높이 숫자, 관계말 없음), 최대 15 | [`search_objects/`](search_objects/) |
| `move_robot` | 로봇 한 부분(베이스·몸통·팔·그리퍼)을 움직이고 멈출 때까지 기다리기. 베이스 모드 `go_to`·`probe`·`delta`, 에이전트 쪽 모드 `explore`(프런티어 탐사) | [`move_robot/`](move_robot/) |

- LLM 에 보이는 도구 수: 코드가 있는 것 4 개. 계획만 있는 도구와 전체 수(원칙 8 개, 계획까지 10 개)·합치는 안은 [plan.md 3.3](../plan.md).
- 프런티어 탐사(옛 스킬 explore 의 기준선 코드, 10-06)는 새 도구로 늘리지 않고 `move_robot` 의 베이스 모드 `explore` 로 넣었다 — 도구 수 그대로.
  스킬은 `skill.json` 의 `tools[].modes` 로 LLM 에게 보일 모드를 고른다(explore 스킬은 이 모드를 LLM 에게 숨기고 기준선으로만 씀).
- 도구가 아닌 것: 평가용 정답 기준(`move_robot::mock_eval`, LLM 에게 안 보임), 개발 명령 `decisions-agg`([`../devtools/`](../devtools/README.md)).

## `search_objects` · `confirm_object` · `list_place` — 물체 찾기, 이름 고치기, 자리 목록

자세한 것: [`search_objects/README.md`](search_objects/README.md). LLM 은 글만 받으므로 벡터 찾기는 도구 안의 공용 물체 색인
(behavior-2026 `src/scene_graph/clip/include/sgsearch.h`)이 하고, 도구는 색인된 이름·속성과 기억의 자리 정보를 글로 준다.

인자
```json
search_objects {"query": "라디오", "k": 5, "room": "kitchen", "state": "seen|moved|held|gone", "max_age_s": 60, "seen_after_s": 120}   (query 만 필수; 좌표·시간 필터만 — 옛 `near`(물체 기준 1.5 m)는 물체 간 관계라 10-06 에 뺌)
confirm_object {"id": "O27", "name": "radio", "source": "user|close_look"}                                       (셋 다 필수)
list_place     {"place": "kitchen" | "R2" | "O12"}
```
결과
- 찾기 후보 하나: `id, name, name_p, alt[{name,p}], match_type(name|appearance), registered?, p_query·p_registered(appearance 일 때), attrs[색·재질·크기],
  room, landmark{id,name,dist_m,dz_m}, state, last_seen_ago_s, pos[x,y,z], size[x,y,z], pos_sd?, rel{x,y,z,dist_m,bearing_deg}, match`.
  맨 위 `ask_user`(있으면 행동 전에 묻기)·`hint`·`searched`·`now_s`.
- `confirm_object`: `{"status":"ok","id","name","p_before","p_after","registered"}`(실시간이면 + `map`). `list_place`: 같은 줄 형식 + 가구면 `dist_m`·`dz_m`.
- 기억: 오프라인 `view.json` 또는 실시간 scenemap 스냅숏(`so_open_live` — 호출마다 새 스냅숏). 물체 사이 관계말(on/in)은 계산하지 않는다.

만들기·시험: `cd search_objects && cargo test --release`(9개), `search-objects live-check libsgrt.so MEM`(진짜 scenemap 합성 스트림),
`search-objects demo MEM 라디오`(각본), `search-objects llm MEM "라디오 가져와"`(KAU).

측정(BEHAVIOR 집 FastSAM 기억 283 물체): 이름만 R@5 0.22 → 이름 + 생김새 0.56, 이름으로 못 찾는 물체 R@5 0 → 0.33,
없는 물체 질의에 묻지 않고 행동할 만큼 나오는 것 0.03, 질의 ≈ 0.1 ms(자유 글 ≈ 1 ms).

## `move_robot` — 관절·베이스 직접 움직이기

한 부분을 목표까지 안전하게 움직이고 결과를 짧게 돌려준다. 닫힌 고리 실행기(검증·자르기·보간·판정)는 Rust 한 곳(`move_robot/src/lib.rs`,
주행은 `nav.rs`·`robot_nav.rs`·`map.rs`)이고, 시뮬에서는 같은 코드가 `libmove_robot.so` 로 평가기 안에서 돈다.

### 인자 (9B 모델용으로 작게, 정의 JSON 약 1.5 KB — `move_robot::definition()`, `move-robot schema`)

```json
{"part": "base | torso | left_arm | right_arm | left_gripper | right_gripper",
 "mode": "go_to | probe | delta | absolute",            (에이전트 쪽 "explore" 는 스킬이 고를 때만 보임 — 아래)
 "target": string (선택, go_to 만: 지난 지도 요약의 id "F1"·"R2"),
 "values": [number, ...] (1..7 개),
 "duration_s": number (선택, 0 보다 큼),
 "max_steps": int (explore 만, 1..50, 기본 1)}
```
필수: `part`, `mode`. `values` 는 go_to 에 `target` 을 줄 때 말고는 있어야 하고, 개수는 모드·부분마다 정해져 있다(probe 2, target 없는 go_to 2, base delta 3, torso 4, 팔 7, 그리퍼 1).
`go_to`·`probe`·`explore` 는 베이스 전용, 베이스에 `absolute` 는 없다(`error`).

| part | values (순서·단위) | 행동 벡터 칸 (R1Pro 23) | 제어기 (`omnigibson/eval/r1pro.yaml`) |
|---|---|---|---|
| `base` | delta `[앞 m, 왼쪽 m, 왼쪽으로 돌기 °]` — 호출할 때의 로봇 기준 / probe `[왼쪽으로 돌기 °, 앞 m]` / go_to `target` 또는 `[앞 m, 왼쪽 m]` | 0..3 `vx, vy, wz` | 속도, [-1,1] → ±0.75 m/s, ±0.75 m/s, ±1 rad/s |
| `torso` | `[j1..j4]` ° (delta·absolute) | 3..7 | 절대 관절 위치 (rad) |
| `left_arm` | `[j1..j7]` ° (어깨 → 손목) | 7..14 | 절대 관절 위치 |
| `left_gripper` | `[벌림]` 0 = 닫힘 … 1 = 열림 (absolute 권장) | 14 | smooth, [-1,1] → 손가락 0…0.05 m |
| `right_arm` | `[j1..j7]` ° | 15..22 | 절대 관절 위치 |
| `right_gripper` | `[벌림]` | 22 | smooth |

- 행동·관측 칸은 BEHAVIOR-1K `eval_utils.py` 의 `ACTION_QPOS_INDICES` / `PROPRIOCEPTION_INDICES` 그대로. 머리(카메라)는 평가 설정에서 `NullJointController` 라 enum 에 없다.
- `delta` 는 지금 측정값에 더하기, `absolute` 는 그 값으로 가기(관절·그리퍼만). **`delta` 에 0 만 주면 움직이지 않고 상태만 읽는다**(도구를 늘리지 않으려고).
- 한 번에 한 부분. 그동안 나머지 부분은 마지막 목표를 유지하고 베이스는 0 속도.

| 베이스 모드 | values / target | 하는 일 |
|---|---|---|
| `go_to` | `target`: 지난 지도 요약의 `F1`(프런티어)·`R2`(방) — 또는 values `[앞 m, 왼쪽 m]`(아는 빈칸이어야 함) | **아는 빈칸만** 지나는 경로(부풀린 격자 Dijkstra)를 지역 제어기(DWA)가 따라감. 도착하면 프런티어의 모르는 쪽을 봄. 모르는 점·장애물 점이면 `error` + 고칠 문장 |
| `probe` | `[왼쪽으로 돌기 °, 앞 m]`(앞 ≤ 1.5 m) | 제자리에서 돌고 0.25 m/s 로 천천히 앞으로. 깊이 프레임·지도 장애물 앞(몸통 + 12 cm)에서 멈춤 → `blocked` + `stopped_by`, `clear_m`. 앞 0 = 돌아보기만 |
| `delta` | `[앞, 왼쪽, 돌기°]` | 직진·돌기 + 지도가 있으면 진행 방향 안전 정지(지도 장애물·카메라 밖 모르는 곳·깊이) |
| `explore` (10-06) | `max_steps` | 프런티어 탐사(Yamauchi): 지난 지도 요약에서 `path_m` 가장 짧은 프런티어로 `go_to` 를 `max_steps` 번까지, 닿을 수 있는 프런티어가 없으면 멈춤 — `src/frontier.rs` |

`explore` 는 지난 관측이 필요해 시뮬 접착부(`mr_command`)가 아니라 에이전트 쪽(`frontier::run_tool_ctx`, 런타임이 부름)에서만 돈다.
기본 정의(`definition()`, `mr_tool_definition`)의 enum 에는 없고, 스킬이 `skill.json` `tools[].modes` 에 넣으면 `definition_modes` 가 보인다.

### 결과 (짧은 JSON)

```json
{"status":"reached","part":"left_arm","state":[0.0,20.0,0.0,-30.0,0.0,0.0,0.0],"target":[…],"units":"deg","steps":38,"time_s":1.27}
{"status":"blocked","part":"right_gripper","state":[0.4],"error":0.6,"hint":"gripper stopped before closing: probably holding an object",…}
{"status":"error","message":"left_arm needs exactly 7 values (deg), got 2","hint":"fix the arguments and call move_robot again"}
```
- `status`: `reached`(허용 오차 안) / `blocked`(멈췄는데 목표 못 감: 접촉·물체·장애물) / `timeout`(아직 가는 중) / `error`(인자 틀림 — 관찰값).
- `clamped`: 한계 밖이라 잘린 칸 번호, 관절이면 `limits`(그 칸의 허용 범위 °)도. `slowed`: 요청한 `duration_s` 가 안전 속도보다 빨라서 늘렸다.
- `error`: 관절·그리퍼는 가장 큰 남은 차이 하나(°·비율). 베이스 delta `state` 는 이번 호출 동안 움직인 양(base_qvel 적분 오도메트리), `error` 는 `[남은 m, 남은 °]`.
- 베이스 go_to·probe 결과는 `state` 대신 `mode`, `target`, `moved_m`, `turned_deg`, `time_s`, go_to 면 `planned_m`·`replans`. 안전 정지면 `stopped_by`, `clear_m`.
- 지도를 받은 뒤의 베이스 결과에는 `map`(LLM 관측): `free_m2`, `new_free_m2`, `frontiers`[`id`, `path_m`(아는 빈칸 경로 길이), `dir`, `new_area_m2`(둘레 2.5 m 모르는 넓이), `room`],
  `around`(8 방향 "known 2.1m then wall|unknown, depth clear 1.8m"), `rooms`[`id`, `visited`, `path_m`], `in_room`, `status`.
  그리고 `_m`(측정, 런타임이 LLM 에게서 뺌): `gt_cov`, `free_m2`, `path_m`, `sim_s`, `contacts`, `stalls`, `blocked`, `replans`, `min_clear_m`, `obs_us`, `plan_us`, `costmap_us`, `pose`.
- `explore` 결과: 마지막 go_to 결과 + `explore{steps, targets, end: no_frontier|max_steps|link_down}` + `_calls[{args, result}]`(실제 go_to 하나하나 — 런타임이 결정 기록을 하나씩 남김).
  한 번도 못 가면 `{"status":"done","mode":"explore","end":"no_frontier","message":"no reachable frontier left",…}`.
- 결과 → LLM 관측 줄이기·결정 기록 칸(`compact`, `obs_features`, `label`, …)은 도구 쪽 `src/llm_view.rs` 에 있고 런타임이 이름으로 부른다.

### 안전 한계 (`Safety::default`, `src/lib.rs`)

| 항목 | 값 |
|---|---|
| 관절 한계 | URDF `r1pro.urdf` 한계 안쪽 2° 로 자름 |
| 팔 / 몸통 속도 | 45 °/s / 20 °/s (최소 저크 보간 최고 속도 기준) |
| 그리퍼 | 1 초에 끝까지 (비율 1.0/s) |
| 베이스 | delta 0.3 m/s, 35 °/s, 한 번에 축마다 2 m·180° 까지 / go_to 0.5 m/s, 50 °/s / probe 앞으로 0.25 m/s, 한 번에 1.5 m·돌기 ±360° 까지. 가속 0.6 m/s²·90 °/s² |
| 막힘 판정 | 관절: 지령과 실제 차이 20° 넘게 0.2 s → 멈추고 그 자리 유지 / 보간 끝난 뒤 정지 0.27 s. 베이스: 지령의 20 % 미만으로 1 s |
| 도착 허용 | 관절 1.5°, 그리퍼 0.05, 베이스 1 cm·1° (멈춘 뒤 2 배 안이면 `reached`, 밖이면 다시 접근 최대 2 번) |
| 시간 초과 | 관절: 보간 + 1.5 s, 베이스 delta: 예상 시간 × 1.5 + 3 s (+ `duration_s`), go_to: 경로 길이 / 0.3 m/s + 12 s, probe: 돌기 / 50 °/s × 1.5 + 앞 / 0.25 m/s × 2 + 4 s |

그리퍼가 닫다가 막히면(물체를 쥠) 지령은 그대로 둔다(계속 쥔다). 팔이 막히면 측정 위치로 지령을 바꿔 더 밀지 않는다.

### 주행 층 (`src/nav.rs`, `src/robot_nav.rs`, Nav2 와 같은 층 구조, ROS 없음)

- 비용 지도 = scenemap 2D 점유 격자 하나(로그 오즈라 치운 물체는 광선이 비움). 바뀐 칸은 scenemap 의 바뀐 영역(`sgrt_map_view.dirty_box`) 안에서만 비교,
  거리장은 장애물이 바뀔 때만 다시. 물체 기억의 옮길 수 있는 물체 둘레는 계획 벌점.
- 전역 경로: 바뀐 칸이 경로 통로(몸통 + 0.3 m)에 걸리고 실제로 막혔을 때, 또는 3 s 마다 다시 계획(ms 단위).
- 몸통: 반지름 0.37 m 원(시뮬 base_link AABB 0.743 × 0.732 m). 사각형 0.57 × 0.54 m 로 했을 때 시뮬에서 바퀴가 소파·탁자에
  12 번 닿았고, 원으로 바꾼 뒤 0 번(같은 집·같은 출발). 계획 부풀림 0.40 m.
- 로봇별 몸 크기: 환경 변수 `MOVE_ROBOT_FOOTPRINT` — 없거나 `limo_omx`(10-06 부터 기본)면 LIMO 사각형 0.36 × 0.22 m·계획 부풀림 0.241 m, `r1pro` 면 위 값(옛 R1 기록·시험 전용),
  `rect:<길이>x<폭>`·`circle:<반경>` 도 된다(`nav::Body`). behavior-2026 `run_explore.sh` 가 `SGRT_ROBOT=limo_omx` 일 때 넣는다.
- 지역 제어: 매 스텝 DWA(앞 속도 6 + 뒤 2 × 회전 13 × 옆 속도 3 표본, 1.2 s 굴림, 몸통 둘레 44 점 검사, 여유 2 cm + 0.2·v)
  + 전방향 미끄러지기 7 방향 + 막혔을 때 16 방향 빠져나오기. 가야 할 쪽이 30° 넘게 옆이면 제자리 돌기.
- 얇은 안전 정지: 마지막 깊이 프레임 장애물 점 중 **지도에 아직 없는** 것으로 앞 속도를 줄이고 멈춤.
- 회복: 다시 계획 → 35° 돌아보기 → 되짚기(지나온 자리 0.7 m 를 거꾸로) → `blocked`. 밀었는데 안 움직이면(접촉) 0.15 m 물러난 뒤 `blocked`.
  두 번 실패한 목표 둘레 프런티어는 다음 관측에서 뺌(`skipped_failed`).

### 실행 경로

```
런타임(run-skill) ── 도구 호출 ──▶ move_robot::frontier::run_tool_ctx (explore 면 go_to 를 되풀이) → link::run_tool ── 먼저 인자 검사(왕복 없이 오류)
     │                                              │
     │                                              └─ TcpSim ─ JSON 한 줄 ─▶ [평가기 프로세스]
     │                                                   behavior-2026/src/sim/move_robot/move_robot_sim.py (ctypes)
     │                                                   매 스텝: proprio 61 → mr_tick (libmove_robot.so, 같은 Rust 실행기) → 행동 23
     ◀──────────────────── 결과 JSON 한 줄 ◀─────────────────┘
```
- C ABI(`move_robot/include/move_robot.h`, `src/ffi.rs`): `mr_new`, `mr_free`, `mr_tick`(0 대기 / 1 움직이는 중 / 2 이번에 끝남 / -1 proprio 이상 / -2 인자 이상), `mr_command`(0 시작 / 1 바로 끝남 / -2),
  `mr_busy`, `mr_reset`, `mr_take_result`, `mr_tool_definition`, 지도용 `mr_set_map`·`mr_set_reference`·`mr_set_contacts`·`mr_overlay_json`,
  `mr_set_gt_pose(r, x, y, yaw)`(`SGRT_POSE=gt` 일 때 평가기가 매 스텝 부름. 베이스 delta 의 도착 판정은 그래도 오도메트리 — 실제 로봇과 같게).
- 에이전트 쪽 의존성: `serde_json` 하나. `--features llm` 일 때만 기존 계획기 `llm.rs`(KAU HTTPS, curl)를 경로 의존성으로 쓴다.
- 가짜 집 `link::MockWorld`(정답 바닥 PGM + 머리 카메라 흉내 ±49.6°, 6 m, 로그 오즈 격자, 몸통 원 접촉). 덮음을 잴 정답 기준은
  `mock_eval::{mock_from_gt, reachable_reference}`(평가용, LLM 에게 안 보임 — 옛 스킬 explore `bin/explore.rs` 에서 옮김).

### `executor: "vla"` 호출 (코드: `src/vla.rs`·`robot_vla.rs`·`goal.rs`·`limo.rs`·`verify.rs`)

- 호출 `{"executor":"vla","skill":"pick up cup","goal":{"pick":{"id":"O12"},"place":{"id":"O3"} | {"point":[x,y(,z)]}},"max_s":30}` →
  결과 `{"status":"done|failed|timeout|handback","reason","evidence","steps","min_clear_m","contacts",…}`. `run_tool` 이 먼저 나눔(이동·탐사는 `route:"move_robot"` 로 거절)·물체 id·지점을 검사한다.
- 지점 검사(`goal.rs check_point`): 아는 빈 바닥 또는 0.05–0.52 m 윗면, 몸통이 서는 칸에서 닿음, 아니면 1.0 m 안 가장 가까운 맞는 자리 → `point.snap_m`, 없으면 `error(invalid_point)`.
- C ABI: `mr_vla_start`, `mr_vla_tick`, `mr_vla_tick_ext`, `mr_filter`, `mr_vla_set_objects(_json)`, `mr_vla_goal_entries`, `mr_vla_contacts`, `mr_vla_stop`, `mr_vla_busy`.
  안의 정책은 대역(`scripted` 기본, `replay:<jsonl>`, `external`). 설계는 [POLICY.md](../../../docs/map_vla/POLICY.md) 1.2–1.4.

### 만들기·시험

```bash
cd src/agent/tools/move_robot
cargo test --release -j4                  # lib 시험 60개 (Isaac Sim 없이; explore 모드 3개 포함)
cargo build --release -j4 --features llm  # libmove_robot.so + move-robot 명령
./target/release/move-robot schema
./target/release/move-robot mock '{"part":"left_arm","mode":"delta","values":[0,20,0,-30,0,0,0]}'   # 가짜 로봇
python ../../../behavior-2026/src/sim/move_robot/run_eval_move.py --listen 127.0.0.1:8771 -- --task-name turning_on_radio --max-steps 6000 --headless
./target/release/move-robot call '{"part":"base","mode":"delta","values":[0.3,0,0]}'
set -a; . ~/.config/behavior-2026/kau.env; set +a   # 키는 환경변수로만
./target/release/move-robot llm "오른쪽 그리퍼를 반쯤 닫고 앞으로 50cm 가" [--mock]
python3 ../../../behavior-2026/src/sim/move_robot/test_move_robot_sim.py   # 시뮬 접착부 시험 (Isaac Sim 없이)
```

### 측정

- 베이스 덜 감 고침: 허용 1 cm·1°, 최소 접근 속도 3 cm/s·3°/s, 멈춘 뒤 다시 접근. 가짜 로봇 0.2 m → 오차 < 12 mm, 30° → < 1.2°.
- 시간(시뮬, 1 층 집): 지도 받기 2–5 ms/keyframe, 관측 만들기 1–2.5 ms/호출, 다시 계획 0.4–1.4 ms, 닫힌 고리 한 스텝 0.2–5 ms(DWA 중). 평가기 한 스텝 130–165 ms 의 3 % 안팎.
- 탐사(스킬 [`explore`](../skills/explore/README.md)): 시뮬 두 집에서 go_to·probe 로 접촉 0, 덮음 0.94–0.95(LLM). 모드 `explore`(프런티어 탐사, `max_steps` 1 씩)는
  가짜 집 6 출발(`devtools/mock_eval.sh`)에서 접촉 0, 덮음 0.940–0.994, 호출 7–17 — 옮기기 전 기준선과 decisions·timeline·summary 가 벽시계·시각 칸 말고 모두 같음(10-06).
