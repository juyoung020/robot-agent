# AI Agent 계획 — 기억을 먼저 찾고, VLA 를 한 단계씩 부리는 에이전트

작성 2026-10-03 (5주차). 이 폴더(`src/agent/`)는 AI Agent(0536, 최영식 교수님) 개인 프로젝트 범위다.
질문: **"우리 AI agent 를 이 프로젝트에서 적극적으로 활용하려면 어떻게 해야 하나"** — 답은 이 문서 하나에 둔다.

근거로 읽은 것: 팀 `README.md`·`docs/plan.md`·`docs/model_selection.md`, 수업 강의계획서·labs(week02·03·05),
기존 Rust 계획기 `src/behavior-2026/src/agent/planner/`(설계 `docs/에이전트_설계.md`, 통합 `docs/통합_실시간.md`, `plan.md` 4.1),
물체 기억 C ABI `scenemap.h`·`sgrt.h`, 2025 2위 `refs/openpi-comet`, 팀 벤치마크 `refs/code/dynamic-object-mapping-benchmark`.
추정은 "(추정)", 사용자가 정할 것은 "(결정 필요)"로 적었다.

---

## 0. 한 줄 결론

1. **"기억 먼저, 행동은 결정적으로."** 질문은 LLM 없이도 답할 수 있게(빠른 길) 만들고, 명령은 LLM 이 **계획 한 번**만 세운다.
   단계 실행·확인·흔한 복구는 Rust 상태 기계가 한다. LLM 은 계획·애매함·복구 한도 초과 때만 다시 부른다(Plan-and-Execute + 예외 때 재계획).
2. **지시 문장은 한 곳(Rust `skillspec`)에서만 만든다.** 학습 데이터 주석도, 에이전트 실행도 같은 함수로 문장을 만든다.
   지금 VLA 가중치(Comet pt50/pt12, 1위 모델)는 단계 문장을 안 따르므로 **스킬 단위 재학습**이 필수이고, 그 전까지는 1위 모델 + 단계 번호 / 이동 제어기로 대신한다.
3. **"Anthropic 급"은 숫자로 정의한다.** 골든 세트 평가·회귀 게이트·기록(trace)·실패 분류표가 기능보다 먼저 들어간다.
4. 새 코드는 `src/agent/` 의 Rust 크레이트 `ragent` 하나. 이미 있는 `behavior-agent` 계획기(LLM 연결·결정론·trace·35 스킬 어휘·참조 풀기)를 **경로 의존성으로 재사용**하고 다시 짜지 않는다.

---

## 1. 목표 행동 (이게 제품이다)

| 사람이 | 에이전트가 하는 일 | 좋은 답의 모양 |
|---|---|---|
| "컵 어디 있어?" | 물체 기억 조회 → 방·가까운 기준물·최근성 → 한국어 답(위·안 같은 말은 위치·크기를 보고 LLM 이 고름) | "부엌 식탁 위에 있어요 (12분 전에 봤어요)" |
| (기억에 없음) | 없다고 정직하게 + 다음 행동 제안 | "아직 컵을 본 적이 없어요. 부엌부터 찾아볼까요?" |
| (오래됐거나 옮겨짐 표시) | 확신도 낮춰 말함 | "2시간 전엔 거실 소파 옆에 있었는데, 그 뒤로는 못 봤어요" |
| (컵이 여러 개) | 후보를 나눠 말하거나 되묻기 | "컵이 2개 있어요: 부엌 식탁 위, 거실 탁자 위. 어느 쪽이요?" |
| "컵을 쓰레기통에 넣어" | 계획 → `move to` → (보일 때까지) → `pick up` → `move to` 쓰레기통 → `place in` → 단계마다 확인·복구·보고 | "컵 쪽으로 가는 중이에요" → "컵을 집었어요" → "쓰레기통에 넣었어요 ✅" |

명령 실행 예(지시 문장은 2절 계약 그대로):

```
plan  : [-] move to dining table   [x] pick up cup from dining table   [x] move to trash can   [x] place cup in the trash can
step 1: "move to dining table"            → 확인: cup 이 머리 카메라에 보임(1 s 안에 검출)        → [o]
step 2: "pick up cup from dining table"   → 확인: SM_HELD 또는 그리퍼 닫힘+물체 들림            → [o]
step 3: "move to trash can"               → 확인: 쓰레기통이 보이고 1.2 m 안에서 멈춤             → [o]
step 4: "place cup in the trash can"      → 확인: 그리퍼 열림 + 컵이 쓰레기통 상자 안            → [o]  → 보고
```

---

## 2. 지시 계약 (Instruction Contract) — 학습과 실행이 같은 문장

### 2.1 무엇을 확인했나 (openpi-comet 코드)

| 사실 | 근거 |
|---|---|
| 공개 Comet 체크포인트는 모두 `fine_grained_level=0`(과제 문장 고정)으로 학습 | `src/openpi/training/config.py` (637~825행 전부 0) |
| 실행 때 단계 문장은 VLM reasoner 가 자유 글로 만든다: `generate_plan` → `plan_critique` → `generate_subtask`, 계획 상태 `[o]` 끝 / `[-]` 진행 / `[x]` 안 함 | `src/openpi/shared/client.py`, `eval_b1k_wrapper.py` (`fine_grained_level>0` 일 때만) |
| 주석 레벨 번호가 **README 와 코드가 다르다**: README 는 "0 global / 1 subtask / 2 skill", 로더 `load_orchestrators_data` 는 **1 = `skill_description`, 2 = `cot_subtask_description`**, 3 = event | `README.md:147` vs `src/behavior/learning/datas/dataset.py:692-750` |
| 로더는 `orchestrators/task-XXXX/episode_*/subtask_{i}_annotated.json` 을 읽어 프레임 구간마다 문장을 바꾼다 | 같은 파일 `load_orchestrators`, `_get_fine_grained_task` |
| 스킬 어휘는 과제 50개 기준 34종(`move to` 50, `pick up from` 50, `place on` 34, `place in` 31, `push to` 29, `open door` 24 …) | `scripts/task_mapping.json` 집계 |
| 우리 계획기는 2026 시연 주석 `skill_annotation[].skill_description` 35종(34 + `lift`)과 물체 칸 순서 틀을 이미 갖고 있다 | `planner/src/vocab.rs` |
| 지금 가중치(공식 radio, Comet pt50)는 문장을 바꿔도 행동 변화가 잡음 수준 | behavior-2026 `plan.md` 3절·4.1 |

→ **결론**: Comet 이 "학습한" 단계 문장 형식은 공개 가중치에 없다. 우리가 정한 형식으로 **같은 로더에 주석을 넣어 재학습**해야 하고, 레벨 번호 혼동을 피하려고 `skill_description` 과 `cot_subtask_description` **두 칸에 같은 문장**을 쓴다(레벨 1·2 어느 쪽으로 학습해도 같은 문장).

### 2.2 문장 틀 (실행에 쓰는 것만 먼저, 전체는 `vocab.rs` 35종)

| 스킬 | 틀 | 예 |
|---|---|---|
| `move to` | `move to {m}{target}` | `move to dining table` |
| `pick up from` | `pick up {m}{obj} from {support}` (받침 모르면 `pick up {obj}`) | `pick up cup from dining table` |
| `place in` | `place {obj} {b}in the {s}{container}` | `place cup in the trash can` |
| `place on` | `place {obj} {b}on {s}{support}` | `place cup on kitchen counter` |
| `place in next to` / `place on next to` | `… {s:next to} {ref}` | `place cup on table next to plate` |
| `open door` / `close door` | `open the {s}door of {furniture}` | `open the left door of cabinet` |
| `open drawer` / `close drawer` | `open the {s}drawer of {furniture}` | |
| `press` / `turn on switch` / `turn off switch` | `press {obj}` / `turn on {obj}` | `turn on light switch` |

규칙(지금 계획기 규칙을 그대로 계약으로 굳힌다):

- `{obj}` 등은 **정규화된 범주 이름**(`canon_name`: synset `.n.NN` 제거, `_`→공백, 소문자). 시뮬은 BDDL/주석 `object_id` → `canon_name`, 리모는 YOLO 라벨 → 같은 `canon_name`. 인스턴스 id(`cup_12`)는 문장에 절대 안 넣는다(기억·확인에만).
- `{m}` = `the other `, `{b}` = `back ` (주석 `memory_prefix`), `{s}` = 주석 `spatial_prefix`. 그 밖의 수식어 금지.
- **숫자 거리·각도 금지**(사용자 결정 09-29), 문장 90 토큰 이내(π0.5 `max_token_len=200`, 상태 몫 제외). 기존 `strip_numbers`·`tokens_with_margin` 그대로.
- 문장은 LLM 이 쓰지 않는다. LLM 은 `{skill, objects:[id…], spatial?, memory?}` 구조만 고르고, 문장은 `skillspec::render()` 가 만든다 → 학습 문장과 글자 하나까지 같다.

### 2.3 한 곳에서 만들기 — `skillspec`

```
skillspec (Rust, 순수 함수, 의존성 0)
  ├─ SKILLS: [Skill{name, id, kind, slots, templates}]      ← vocab.rs 를 옮겨 계약으로 고정, 버전 SPEC_VERSION
  ├─ canon_name(&str) -> String
  ├─ render(&SkillCall) -> Result<String, SpecError>         ← 칸 수·enum 검사, 숫자 제거, 토큰 예산
  └─ bin `skillspec annotate`                                ← 학습 주석 생성기(아래)
```

- 학습 쪽(Python)은 문장을 만들지 않는다. `skillspec annotate --demos <annotations/> --out <orchestrators/>` 가
  시연마다 `task_annotated.json`(`cot_task_description` = 과제 문장, `cot_subtask_description_list`)과
  `subtask_{i}_annotated.json`(`skill_description` = `cot_subtask_description` = 렌더한 문장, `start_frame`, `end_frame` = 주석 `frame_duration`)을 쓴다.
  Comet 로더를 **고치지 않고** `fine_grained_level=1` 로 학습할 수 있다.
- 계약 시험(골든 파일): `contract_golden.jsonl` 에 `(주석 항목 → 문장)` 1,000줄. Rust `cargo test` 와 학습 스크립트 둘 다 이 파일과 비교 → 한쪽만 바뀌면 바로 깨진다.
- 학습 혼합(추정, 실험으로 정함): 프레임의 70 % 는 스킬 문장, 30 % 는 과제 문장(레벨 0) — 과제 문장 보험 점수를 잃지 않게. 프롬프트 증강은 하지 않는다(정확히 같은 문장이 계약).

### 2.4 재학습 전 대체 경로 (지금 쓸 수 있는 것)

| 상황 | VLA 에 주는 것 | 이동 | 조작 |
|---|---|---|---|
| A. 시뮬, 1위 모델(과제 0~49) | 과제 문장 + **단계 번호**(`pi05_set_stage`, `--stage external`) | 1위 모델 | 1위 모델 |
| B. 시뮬, 이동 제어기(plan.md 4.1 (나)) | 조작 단계만 과제 문장 | `sm_snap_reachable` 경로 + 오도메트리 추종(베이스 행동 칸만 덮어씀) | π0.5 |
| C. 재학습 뒤 | 스킬 문장(`fine_grained_level=1`) | VLA `move to …` (또는 B 와 비교) | VLA |
| D. 리모 | 스킬 문장(리모 시연으로 학습) | **Nav2 제어기**(`move to` 는 VLA 로 안 함) | VLA (`pick up`·`place`) |

- 에이전트 코드는 A~D 어디서나 같다: 상태 기계는 `SkillCall` 을 내고, `Backend` 가 그것을 (문장 | 단계 번호 | 제어기 목표)로 바꾼다.
- 재학습 효과 판정(게이트): 같은 관측에 문장만 바꿨을 때 행동 변화가 잡음의 **3배 이상**(지금 1.0~1.6배, 오프라인 probe 그대로 재사용) + 스킬 구간 성공률.

---

## 3. 구조

### 3.1 전체

```
 📱 앱(WS) ──▶ ragent ─────────────────────────────────────────────────────────────┐
              │ Dialog ─▶ Router ─┬─ T0 빠른 답(LLM 없음) ──────────────▶ Answer   │
              │                   ├─ T1 한 번 호출(후보 미리 넣음) ───────▶ Answer   │
              │                   └─ T2 계획(도구 루프) ─▶ TaskMachine ─▶ Backend ──┼─▶ SimLink(behavior-2026) | Limo(ROS 2) | Mock
              │ MemView(C ABI sm_snapshot) ◀──────────── scenemap/sgrt ◀───────────┘
              │ Rooms · Lexicon · Ctx(16k) · Trace(JSONL) · Gate(승인) · Checkpoint
              └─▶ KAU vLLM (Qwen3.5-9B, OpenAI 호환, 스트리밍)
```

- 루프는 **프레임워크 없이** OpenAI Chat Completions 원형(수업 규칙). 기존 `llm.rs`(결정론 표본값, `<tool_call>` 글 복구, `<think>` 제거, 요청 지문)를 그대로 쓴다.
- LLM 은 이미지·관측을 매 스텝 보지 않는다. 매 스텝 경로(시뮬 30 Hz)는 기존 `link.rs`/`EnvSession` 이 맡는다.

### 3.2 크레이트 `ragent` 모듈

| 모듈 | 하는 일 | 재사용 |
|---|---|---|
| `main.rs` | 명령: `chat`, `serve`(앱 WS), `ask "<질문>" --scene scene.json`, `eval <set>`, `replay` | |
| `agent.rs` | 대화 한 턴: Router → (T0/T1/T2) → 답 스트리밍. 반복 한도 6, 도구 없는 글은 최종 답으로 | 수업 week02 루프 |
| `router.rs` | 의도 분류 `Where / Command / Status / Cancel / Chat` — 규칙(한국어 어미·동사 사전) 먼저, 애매하면 T1 에 맡김 | |
| `memview.rs` | `sm_snapshot` FFI 래퍼 `MemSnapshot`(수명 안전): `find`, `near`, `objects`, `movable`, `view`, `reachable`, `pose`. 오프라인용 `scene.json`(Spark-DSG) 읽기 구현도 같은 trait `Memory` | behavior-2026 `graph.rs` `SceneQuery` |
| `rooms.rs` | 방 추론(3.5절) `RoomMap::room_at(x,y) -> Option<RoomLabel>` | |
| `landmark.rs` | 가까운 기준물(고정 가구) 고르기 — 이름·거리·상대 높이만, 관계 계산 없음(3.6절) | |
| `lexicon.rs` | 한국어 ↔ 라벨: `컵/머그잔/텀블러 → cup`, `쓰레기통/휴지통 → trash can`, 방 `부엌/주방 → kitchen` … 표(TOML) + 표에 없으면 LLM 질의 1회 | |
| `answer.rs` | 한국어 답 틀: `{방} {기준물} … ({최근성})` — 기준물과의 말(위·안·옆)은 LLM 이 위치·크기로 고름, 최근성 `방금/N분 전/N시간 전/오늘 아침`, 확신도 문구 | |
| `task/plan.rs` | `TaskPlan{goal, steps: Vec<Step>, version}` 검증(스킬 enum·칸 수·id 존재·전제조건) | `plan.rs` 체크리스트 |
| `task/machine.rs` | 단계 상태 기계(3.4절), 예산·재시도·복구 표 | `planner.rs` 자동 증거 |
| `task/verify.rs` | `visible(id)`, `held(id)`, `placed(id, target)`, `at(target)` — 기억·그리퍼·검출로 | `Core::evidence` |
| `skillspec`(별도 크레이트) | 2.3절 계약 | `vocab.rs` `instruction.rs` |
| `backend/{mod,sim,limo,mock}.rs` | trait `Backend { exec(SkillCall)->Handle; poll(Handle)->StepStatus; stop(); }` | `link.rs` `mockworld.rs` |
| `ctx.rs` | 16k 맥락 조립·예산(3.7절) | `context.rs` `memory.rs` |
| `gate.rs` | 위험 행동 승인(5.5절) | |
| `checkpoint.rs` | `task.json` 매 단계 저장, 재시작 시 이어 하기 | |
| `trace.rs` | JSONL 기록 + HTML 재생기 | 기존 `trace.rs` `replay.rs` |
| `eval/` | 골든 세트 실행·채점·게이트(5절) | |

### 3.3 도구 (LLM 에 보이는 것은 8개 이하 — 지금 10 개, 아래 메모)

원칙: **계획·실행 호출은 id 로 말한다. 좌표는 보여 주되 VLA 로는 보내지 않는다**(10-05 바꿈). 찾기 결과에 map 좌표 `pos`·크기 `size`·확률 모드 위치
불확실도 `pos_sd`·지금 로봇 기준 `rel{x 앞, y 왼쪽, z, dist_m, bearing_deg}` 가 있어 LLM 이 숫자로 추론(어느 쪽이 가까운지, 높이 차 등)한다.
그래도 `set_plan`·VLA 호출의 물체는 id 로만 넘기고, map 좌표는 VLA 입력에 들어가지 않는다 — 실행기가 매 스텝 실시간 지도에서 id 를 풀어
로봇 기준 값으로 바꾼다(VLA_INPUT 0절 "어떤 집에서도 같은 뜻"). 인자는 enum·필수로 좁힌다, 결과는 짧은 JSON + `hint`, 실패는 오류 관찰값(예외로 루프를 죽이지 않음).

| 도구 | 인자 | 결과(요약) | 끝냄 |
|---|---|---|---|
| `search_objects` (10-05, 옛 `find_object`) | `query: string`(한국어 가능), `k: int=5`, `room?`, `state?`, `near?`, `max_age_s?`, `seen_after_s?` | 3 단계 중 ①② 를 도구 안에서: ① 이름·동의어·상위어 검색 → 없거나 약하면 ② **이름 무시 생김새 재검색**(물체별 시점 벡터로 P(질의어 \| 모습)). 결과는 글만: `{query, searched, ask_user?, hint?, matches:[{id, name, name_p, alt:[{name,p}], match_type: name/appearance, registered?, p_query·p_registered(appearance 일 때), attrs:[색·재질·크기], room, landmark{id,name,dist_m,dz_m}, state, last_seen_ago_s, pos, size, pos_sd?, rel{x,y,z,dist_m,bearing_deg}, match}], now_s}` — `ask_user` 는 후보마다가 아니라 결과 맨 위 하나(만든 것: [`tools/search_objects`](tools/search_objects/)), 없으면 `matches:[]` + `ask_user` + `hint` | |
| `confirm_object` (10-05) | `id`, `name`, `source: enum(user, close_look)` | ③ 이름 고치기 — 물체 이름 사후에 강한 관측으로 반영, 다음부터 ① 에서 바로 찾음. 확인 기록은 보정 데이터 | |
| `describe_object` | `id`, `with_image: bool=false` | 크기·높이·관측 수·처음 자리에서 움직인 거리 + (선택) best view RGB 256 px(`sm_snap_view`) | |
| `list_place` (10-05) | `place: string`(방 이름·`R2` 또는 가구 id `O12`) | 그 방 물체, 또는 그 가구 상자에서 수평 1.5 m 안 물체(최대 15, 옮길 수 있는 것 먼저). 줄은 찾기와 같은 칸 + 가구면 `dist_m`·`dz_m` — **관계말 없음**(만든 것: [`tools/search_objects`](tools/search_objects/)) | |
| `set_plan` | `goal: string`, `steps: [{skill: enum(35), objects: [id], spatial?: enum, memory?: enum}]` | 검증 결과 + 렌더한 문장 목록 → 승인 필요하면 `needs_approval` | |
| `check` | `kind: enum(visible, held, placed, at)`, `id`, `target?` | `{result: yes/no/unclear, evidence}` | |
| `ask_user` | `question`, `options?: [string]` | 사용자 답(다음 턴) | ✔ |
| `report` | `text`, `status: enum(progress, done, failed)` | — (스트리밍) | ✔ |
| `remember` | `fact` | 대화 기억에 저장(물체 기억과 별도) | |

- **물체 검색(10-05)**: 검색은 도구 안에서 벡터(SigLIP 2, 공용 물체 색인)로 하고 LLM 에는 **색인된 이름·속성 글만** 준다(LLM 은 API 라 벡터를 못 받음). 같은 색인을 RecallVLA 도 자기 질의 벡터로 검색하므로, `set_plan` 의 물체 id 는 VLA 에 **힌트**다. 예: "라디오 가져와" → ① 라디오 없음 → ② 소화기로 등록된 O27 이 라디오일 확률 2 등 → LLM 이 되묻거나 O27 을 힌트로 넘김 → 확인되면 `confirm_object`. 도구가 9 개가 되어 "8 개 이하" 원칙을 넘는다 — `confirm_object` 를 `check` 결과에서 자동으로 부르는 쪽도 후보. 설계 [model_selection 물체 찾기](../../docs/model_selection.md), [MAPVLA_SPEC 결정 기록](../../docs/map_vla/MAPVLA_SPEC.md).
- **실시간 기억(10-05)**: 같은 도구가 오프라인 `view.json` 과 실시간 scenemap 스냅숏(`so_open_live`, sgrt 는 `sgrt_scenemap`) 둘 다에서 돈다 — 3.2 `memview.rs` 의 `Memory` trait. 실시간 확인은 `sm_observe_object_name` 으로 지도 `objprob` 이름 사후에도 넣는다.
- **도구 수 메모(10-05)**: LLM 에 보이는 것 = `search_objects`·`confirm_object`·`list_place`·`describe_object`·`set_plan`·`check`·`ask_user`·`report`·`remember` 9 + 시연용 `move_robot` 1 = **10 개**(원칙 8 개). 합치는 안(아직 안 합침 — 결정 필요):
  1. `list_place` → `search_objects` 의 `place` 인자(있으면 `query` 생략 가능, 결과는 지금 list_place 형식) — 줄 형식이 이미 같아 쉬움. −1.
  2. `describe_object` → 접기: 크기·자리·불확실도는 이미 찾기·목록 줄에 있음. 남는 것(관측 수·처음 자리에서 움직인 거리·best view 사진)은 `search_objects` 에 `detail: true`(k = 1)로. −1.
  3. (선택) `confirm_object` 의 `close_look` 은 `check` 결과에서 자동으로 부르고, LLM 은 사용자 확인 때만 부름 — 도구 수는 그대로지만 호출이 줆.
  1 + 2 면 8 개.
- **프런티어 탐사(10-06)**: 옛 스킬 explore 의 LLM 없는 기준선 코드(경로 가장 짧은 프런티어로 go_to)를 도구로 옮기면서 새 도구 `explore_frontier` 로 두지 않고
  `move_robot` 의 베이스 모드 `explore`(`max_steps`, 에이전트 쪽 `tools/move_robot/src/frontier.rs`)로 넣었다 — 도구 수 10 그대로(새 도구면 11).
  스킬이 `skill.json` `tools[].modes` 로 이 모드를 보일지 고른다(explore 스킬은 LLM 에게 숨기고 기준선으로만). 권고: 위 1 + 2 합치기로 8 개를 맞추고, 탐사가 필요한 스킬은 이 모드를 보이게.
- **코드 자리(10-06)**: 스킬 폴더는 지시문만(프롬프트·`skill.json`). 루프·프롬프트 읽기·맥락 접기·기록은 에이전트 런타임 `runtime/`(`run-skill`), 실행 코드는 `tools/`, 개발 명령(`decisions-agg`)은 `devtools/`.
  위 3.2 의 `ctx.rs`·`trace.rs` 는 `runtime/` 에 먼저 생겼다(지금은 explore 의 16k 접기·decisions/timeline/trace 기록).
- 이동·스킬 실행 도구는 **LLM 에 주지 않는다**. `set_plan` 으로 계획을 넘기면 `TaskMachine` 이 실행한다. 복구 한도를 넘었을 때만 LLM 이 다시 불려 `set_plan`(남은 단계 교체) / `ask_user` / `report(failed)` 중 하나를 고른다.
- `set_plan` 예(컵 → 쓰레기통):

```json
{"goal": "put the cup in the trash can",
 "steps": [{"skill": "move to", "objects": ["dining_table_3"]},
           {"skill": "pick up from", "objects": ["cup_12", "dining_table_3"]},
           {"skill": "move to", "objects": ["trash_can_1"]},
           {"skill": "place in", "objects": ["cup_12", "trash_can_1"]}]}
```

- 검증 규칙(`task/plan.rs`): 스킬 enum 밖 → 거부, 칸 수 틀림 → 거부, id 가 기억에 없음 → 거부 + `find_object` 하라는 hint, `place *` 앞에 같은 물체 `pick up` 없음 → 거부, `pick up` 앞에 `move to` 없음 → 자동 삽입(규칙, LLM 안 부름).

### 3.4 단계 상태 기계 (`task/machine.rs`)

```
Pending ─▶ Approach(move to) ─▶ Acquire(보이나?) ─▶ Execute(VLA skill, 예산) ─▶ Verify ─┬─▶ Done ─▶ 다음 단계 / report(done)
                                    │                         │                         └─▶ Failed(reason) ─▶ Recover
                                    └── NotVisible ───────────┴── Timeout ───────────────────────────────────────┘
```

| 실패 | 알아채는 법 | 자동 복구(LLM 없음) | 한도 넘으면 |
|---|---|---|---|
| `NotVisible` (도착했는데 안 보임) | 검출 없음 2 keyframe | 제자리 회전 둘러보기 1번 → 기억 2순위 후보로 이동 → 같은 방 큰 가구 근처 | LLM 재계획 (`find_object`·`ask_user`) |
| `GraspFailed` | `held=no` (그리퍼 0.005 m 아래로 닫힘 / 물체 안 들림) | 같은 문장 재시도 2번(VLA 자체 복구 여지), 그다음 다시 다가가기 | LLM |
| `ObjectMoved` | 기억에서 대상 `state=moved` 또는 위치 0.5 m 넘게 바뀜 | 새 위치로 `move to` 다시 | 2번 넘게 바뀌면 사용자에게 보고 |
| `Dropped` | 이동 중 `held` 가 no 로 | 떨어진 자리 찾아 `pick up` 다시 1번 | LLM |
| `PlaceFailed` | 그리퍼 열렸는데 물체가 대상 상자 밖 | 다시 집어서 `place` 1번 | LLM |
| `Unreachable` | `sm_snap_reachable < 0` | 가까운 도달 가능 지점 | `ask_user` |
| `Timeout` | 스텝 예산(시연 p90 × 1.5, `catalog.rs`) 초과 | — | LLM |
| `VlaStalled` | 예산 1/3 동안 진척 없음(자세·그리퍼 변화 없음) | 문장 다시 보내고 flush 1번 | LLM |

- 확인 신호: 시뮬 = 기억 `SM_HELD`·물체 높이·그리퍼 폭(기존 자동 증거 임계값), 리모 = 손목 카메라 검출 + 그리퍼 폭/전류.
- 사용자 끼어들기: `Cancel`(“그만”) → `Backend::stop()` 즉시, `Status`(“어디까지 했어?”) → 체크리스트를 T0 로 답.

### 3.5 방 이름 얻기 (지금 DSG ROOMS 층은 비어 있음)

| 환경 | 방 이름 출처 | 쓰는 법 |
|---|---|---|
| 시뮬 개발 | OmniGibson 씬의 방 분할 지도(방 인스턴스 이름 `kitchen_0` 등) 한 번 내보내기 → `rooms.json`(격자 + 이름) | 개발·평가 정답용. **대회 평가 때 써도 되는지 모름**(결정 필요, plan.md 7절 "미리 만든 지도 허용") |
| 시뮬 평가(안전한 쪽) | BDDL `inroom` (과제 물체·가구가 어느 방인지) + 가구로 방 추론 | 가구 → 방 투표: 고정 물체(`movable=0`)마다 BDDL 방 또는 규칙(냉장고·오븐·싱크 → kitchen, 침대 → bedroom, 소파·TV → living room, 변기·욕조 → bathroom) → 격자 영역을 가장 가까운 가구 방으로 칠함(보로노이, 문 경계는 2D 격자 좁은 통로) |
| 리모 | **앱에서 사람이 2D 지도 위에 영역을 그리고 이름 붙임**(한 번) → `rooms.json` | 가장 정확·간단. 그리기 전에는 가구 규칙 추론 |

- `RoomMap` 은 `(x, y) → RoomLabel{id, name_ko, name_en, source: gt/bddl/rule/user}`. 답에서 `source=rule` 이면 "부엌 쪽" 처럼 흐리게 말한다.
- scenemap 팀 코드에 요청: `sm_set_rooms(ctx, grid, names)` 로 받아 DSG ROOMS 층과 `scene.json` 에 같이 저장(뷰어 sgview 에도 보이게). 그 전까지는 `ragent` 쪽에서만 붙인다.

### 3.6 기준물 (`landmark.rs`)

- **물체 간 전치사 관계(on/in/next to/near)는 계산·저장하지 않는다** — 장면 그래프에서도 뺐다(MAP_STATE_PLAN 3 절, 10-05 다시 확인). 위치·크기가 있으므로 "위에 / 안에 / 옆에" 는 LLM 이 추론한다.
- 기준물 = 고정 물체(`sm_snap_movable == 0`) 중 대상에서 1.5 m 안의 가장 가까운 것. 도구 결과에는 기준물 id·이름·거리·대상과의 상대 높이(m)만 넣는다.
- 최근성: `now - last_seen`. 확신도: `state`(seen/moved/gone) × 경과 시간 × `n_obs` — `gone` 이면 "있었는데 지금은 없어요".

### 3.7 맥락 예산 (16k, Qwen3.5-9B)

| 칸 | 토큰 | 내용 |
|---|---|---|
| system + 도구 정의 | 2,000 | 역할·규칙·답 틀·스킬 표(이름만) |
| 기억 요약 | ≤ 2,500 | **질문과 관련된 물체만**(find 결과 + 같은 방 가구 10개), 전체 지도 금지 |
| 작업 상태 | ≤ 800 | 계획 체크리스트 `[o]/[-]/[x]`(Comet reasoner 와 같은 표기), 실패 이유, 손에 든 것 |
| 대화 요약 + 사실 | ≤ 1,200 | 수업 week03 요약 규칙, `remember` 사실 최대 20 |
| 최근 턴 | ≤ 4,000 | 원문 3턴 |
| 이미지(선택) | ≤ 1,000 | best view 1장(256 px) — 애매할 때만 |
| 출력 여유 | 2,000 | |
| 남는 몫 | ~2,500 | 안전 여유(토큰 추정 오차 10 %+) |

넘치면 줄이는 순서: 이미지 → 기억 요약 10줄 → 오래된 턴 → 요약 다시. 요약은 답을 보낸 **뒤** 다른 스레드에서(지연 경로 밖).

### 3.7b 참고 논문 비교 — 장면 그래프 + LLM (10-05, 기록만, 아직 넣지 않음)

`refs/papers/` 다섯 편을 읽고 정리했다. **다섯 편 모두 LLM 에 벡터를 넣지 않는다** — 임베딩 검색은 전부 LLM 밖에서 돌고 LLM 은 글만 받는다(우리 `search_objects` 와 같은 가정).

| 논문 | LLM 에 주는 것 | 물체 찾기 | 메모 |
|---|---|---|---|
| SayPlan (CoRL'23, GPT-4) | 장면 그래프 JSON, 처음엔 방만 보이게 접어서(Office 6,731 → 878 토큰, Tab 2) | 이름·속성 글만 | LLM 이 `expand_node`·`contract_node` 로 필요한 방만 펼침, 펼친 목록을 Memory 로 넘겨 대화 기록 없이 진행. 시뮬레이터 검증 → 재계획 ≤ 5. GPT-3.5 는 탐색 0–6.6 %(Tab 1). 정답 그래프·정적 |
| MoMa-LLM (RA-L'24, GPT-4) | JSON 대신 정리된 문장: 방별 물체(같은 이름은 개수로, 상태를 이름 앞에), 거리는 very close/near/far, 미탐색 방, 최근 5 행동·결과. 매 스텝 새로 인코딩 | 이름 글만(정답 라벨) | 성공 97.7 %, SayPlan 식 JSON 은 86.3 %. 미탐색 방 정보를 빼면 79.4 %(가장 큰 하락) |
| DovSG (RA-L'25, GPT-4o) | 그래프를 안 줌. 지시만 받아 `{행동, 물체 이름}` 단계로 쪼갬 | LLM 밖 CLIP(이름 글 벡터 ↔ 물체 벡터) | 핵심은 지도 부분 갱신. 물체가 옮겨진 뒤 옛 CLIP 특징이 옛 위치로 이끄는 실패 |
| ConceptGraphs (ICRA'24, GPT-4) | 물체마다 `{id, 크기, 위치, 이름, 캡션}` JSON 전체, "이름은 틀릴 수 있음" 명시 | LLM 은 캡션 글로 고름(CLIP 은 비교용) | LLaVA 캡션을 GPT-4 가 요약. 1 등 정답 CLIP / LLM: 묘사형 0.59 / 0.61, 기능형 0.43 / 0.57, 부정형 0.26 / 0.80(Tab III). 가서 없으면 노드를 지우고 "있을 법한 곳" 재질의 |
| HOV-SG (RSS'24, GPT-3.5) | 그래프를 안 줌. 질의를 `[층, 방, 물체]` 로 나누기만 | LLM 밖 CLIP, 층 → 방 → 물체로 좁힘 | 이름은 CLIP 1 등일 뿐, 검색은 벡터라 이름 오류에 강함. 정적 |

**우리만 하는 것**: 이름 검색 → 생김새 재검색 두 단계, 이름 확률을 LLM 에 보여 주고 `confirm_object` 로 이름을 고쳐 쓰는 경로, 같은 색인을 RecallVLA 도 자기 벡터로 검색. 다섯 편은 모두 GPT-4 급이라 9B 로 될지는 검증 안 됨 → 출력 형식을 좁게, 결과 글을 짧게.

**가져올 후보 (보류 — 사용자 결정 10-05: 문서화만, 아직 추가하지 않음)**
1. MoMa 식 결과 표현: 거리를 very close/near/far 로, 상태를 이름 앞에(closed fridge), 같은 이름은 개수로 묶기, **미탐색 방** 목록.
2. ConceptGraphs 식 짧은 캡션 한 줄 + 시스템 글에 "이름은 틀릴 수 있다" — 기능·부정형 질의 보완.
3. 갔는데 못 찾으면 후보를 사라짐으로 낮추고 "있을 법한 곳" 을 다시 묻는 흐름(`last_seen`·`state` 와 연결).
4. SayPlan 식 접기·펼치기(방 요약 → `expand_room`) — 맥락 예산용. 단 작은 모델은 자유 탐색에서 거의 실패했으므로(GPT-3.5 0–6.6 %) 지금처럼 도구가 후보를 좁혀 주는 쪽이 9B 에 안전.
5. DovSG 교훈: 다시 본 물체는 바로 갱신해 옛 벡터가 옛 위치로 이끌지 않게(확률 모드 옮겨짐 처리).

### 3.8 지연 예산 — LLM 을 언제 부르나

| 경로 | LLM | 목표(p50 / p95) | 근거 |
|---|---|---|---|
| 접수 메시지("알겠어요, 찾아볼게요") | 0 | < 0.2 s | 바로 |
| T0 어디 질문(라벨 하나로 풀림) | **0** | < 0.3 s / 0.5 s | `find` + `answer.rs` 틀 |
| T1 애매한 질문(동의어·여러 후보·"아까 그거") | 1 | 첫 토큰 < 2 s, 끝 < 8 s / 15 s | 후보 미리 넣고 한 번 |
| T2 명령 계획 | 1~2 | < 12 s / 25 s | `find_object` → `set_plan` (KAU 실측 한 번 7~12 s) |
| 단계 실행 | **0** | VLA 시간 | 상태 기계 |
| 복구 한도 초과 | 1 | < 12 s | |
| 요약 | 1 | 답 뒤, 경로 밖 | |

- 명령 하나에 LLM 호출 **3번 이하**(정상 경로 1~2번)를 목표로 잰다. 스트리밍(SSE `stream:true`)으로 `report` 문장을 앱에 바로 흘린다.
- KAU 가 죽거나 60 s 넘으면: T0 는 그대로, T2 는 규칙 계획기(`PriorDecider` 방식: "A 를 B 에" 패턴 → 고정 4단계)로 대신하고 "지금은 간단한 명령만 할 수 있어요" 라고 알린다.

---

## 4. "Anthropic 급"의 뜻 — 운영 기준

| 기준 | 뜻(우리 말로) | 잴 것 / 합격선 |
|---|---|---|
| 단순함 우선 | 정해진 흐름(workflow)으로 되는 건 LLM 에 안 맡긴다. 에이전트 루프는 계획·복구에만 | 명령당 LLM 호출 ≤ 3, 질문의 T0 비율 ≥ 60 % |
| 도구 설계(ACI) | id·enum·필수 인자, 틀린 사용을 못 하게(poka-yoke), 오류는 고치는 방법까지 알려 줌 | 도구 인자 오류율 < 2 %, 오류 뒤 회복률 > 90 % |
| 맥락 공학 | 필요한 것만, 예산 안에서, 최근·관련 순 | 요청 토큰 p95 < 9k, 넘침 0 |
| 정직함 | 기억에 없는 것을 있다고 하지 않는다 | **환각 답 0 %**(골든 세트 "없음" 문항), 확신 표현 보정 |
| 검증 | 모든 단계는 확인 신호로 끝난다(LLM 말로 끝내지 않음) | 확인 없는 Done 0 |
| 평가 먼저 | 기능 전에 골든 세트·채점기 | 5절 게이트 통과 없이는 main 에 안 넣음 |
| 투명함 | 계획을 사용자에게 보여 주고, 진행을 보고 | 단계마다 보고 1줄 |
| 기록 | 모든 LLM 요청·응답·도구·결정·지연이 trace 에, 재생하면 같은 결정 | `replay --verify` 일치 100 % |
| 안전·승인 | 되돌리기 어려운 행동은 확인, 비상 정지 항상 | 5.5절 표, 레드팀 통과 |
| 비용·지연 | 3.8절 예산 | p50/p95 기록, 회귀 시 실패 |
| 내구성 | 죽어도 이어 한다 | 단계 중간 kill → 재시작 후 이어 하기 성공 |

실패 분류표(trace 마다 하나 붙임, 9주차 진단 기준):

| 코드 | 실패 | 대표 원인 |
|---|---|---|
| M1 | 기억에 없음 | 검출 못 함·안 가 봄 |
| M2 | 기억이 낡음 | 옮겨졌는데 갱신 안 됨 |
| M3 | 이름 못 이음 | 한국어 ↔ 라벨 사전 빈칸 |
| M4 | 방·기준물 틀림 | 방 지도·기준물 고르기 |
| L1 | 도구 호출 형식 오류 | 9B JSON 깨짐 |
| L2 | 잘못된 계획 | 순서·물체 고르기 |
| L3 | 증거와 반대 판단 | 자동 확인 무시 |
| L4 | 환각 답 | 기억 밖 사실 |
| A1 | 이동 실패/못 감 | 경로·제어기 |
| A2 | 도착했는데 안 보임 | 시야·기억 위치 오차 |
| A3 | 집기 실패 | VLA |
| A4 | 놓기 실패 | VLA |
| A5 | VLA 가 문장 무시 | 재학습 전 가중치 |
| S1 | 외부 API 지연·끊김 | KAU |
| S2 | 위험 요청 | 사용자 입력 |

---

## 5. 평가 — 무엇으로 재나

### 5.1 골든 세트

| 세트 | 크기 | 만드는 법 | 채점 |
|---|---|---|---|
| `qa_where` | 120 문항 | 저장된 `scene.json` 스냅숏(시뮬 장면 6개 + 벤치마크 Office 2바퀴 끝) 위의 질문. 정답 = (방, 기준물 허용 목록, 최근성 ±1분) | 규칙 채점: 방 일치, 기준물 ∈ 허용, "없음" 정직, 최근성 범위 |
| `qa_absent` | 30 | 기억에 없는 물건·옮겨져 `gone` 인 물건 | 환각 0 이어야 |
| `qa_ambig` | 30 | 같은 라벨 여러 개, 동의어("머그잔"), 지시어("아까 그거") | 후보 다 말하거나 되묻기 |
| `cmd_mock` | 30 명령 | `mockworld` 가짜 세계(실패 주입: 안 보임·집기 실패·옮겨짐·떨어뜨림) | 과제 성공, 단계 성공, 복구 성공, LLM 호출 수 |
| `cmd_sim` | 10 명령 × 3 씨앗 | BEHAVIOR(R1Pro) 실제 시뮬, A/B/C 경로 | 성공률, q_score(해당 과제), 시간 |
| `cmd_limo` | 10 명령 × 3 | 실제 리모 | 성공률, 사람 개입 수 |
| `redteam` | 40 | 프롬프트 주입·위험 요청·한국어 우회 | 거절/확인 정확도 |
| `answer_style` | 50 | 답 문장 자연스러움 | LLM 채점기(다른 프롬프트, 5점) + 사람 표본 20 |

### 5.2 지표·게이트 (`ragent eval --gate`)

| 지표 | 1차 합격선(P1 끝) | 최종(P5) |
|---|---|---|
| `qa_where` 정답률 | ≥ 85 % | ≥ 95 % |
| 환각 답(`qa_absent`) | 0 | 0 |
| `cmd_mock` 과제 성공 | ≥ 80 % | ≥ 95 % |
| 단계 성공 | ≥ 90 % | ≥ 97 % |
| 실패 주입 복구율 | ≥ 60 % | ≥ 85 % |
| 도구 인자 오류 | < 5 % | < 2 % |
| T0 지연 p95 | < 0.5 s | < 0.5 s |
| 명령당 LLM 호출 p50 | ≤ 3 | ≤ 2 |
| 레드팀 통과 | — | ≥ 95 % |

- 회귀 게이트: `cargo test`(계약 골든·단위) + `ragent eval qa_* cmd_mock` 이 위 선 아래로 떨어지면 커밋 전에 막는다. LLM 결과는 KAU 공유 서버라 흔들림이 있으므로(추정) `cmd_mock` 은 3번 돌려 평균, 기준선 대비 −3 %p 넘으면 실패.
- 기준선(baseline): ① LLM 없는 규칙 에이전트 ② 기억 없이 LLM 만(물체 기억 효과 ablation) ③ 계획만 LLM, 복구 없음 — 각각 표에 같이 적는다.
- trace 뷰어: 기존 `bagent replay --html` 을 확장 — 대화·계획 체크리스트·도구 결과·단계 상태·실패 코드·지연을 한 화면. 실패 trace 는 분류 코드로 모아 본다.

### 5.3 물체 기억 품질 — 팀 벤치마크

- scenemap 의 `tools/map_timeline` 으로 `map_timeline.csv` 를 내고 `dynamic-object-mapping score` 로 잰다(Office dynamic1/dynamic2/static).
- 합격선(제안): Static F1 ≥ 0.62(ConceptGraphs 수준), **moved·removed 재현율 > 0**(공개 방법은 모두 0 — 우리 기억이 "옮겨짐"을 알아야 "N분 전" 답이 맞음), static 판 change FP ≤ 10.
- 에이전트 연결 시험: 같은 Office 2바퀴 끝 스냅숏에서 "X 어디 있어?" 를 벤치마크 정답(옮겨진 물체는 새 자리)으로 채점 → **기억 점수와 QA 점수를 같이** 본다(기억이 틀려서인지 답 만들기가 틀려서인지 구분: M 코드 vs L 코드).

### 5.4 지연·비용

- 모든 LLM 호출의 입력/출력 토큰·지연을 trace 에. 주마다 표: p50/p95, 명령당 호출 수, KAU 실패율.
- 대회 평가용 대량 실행(1,000판 × 경계 수십 번)은 KAU 로 감당 못 할 수 있다 → 로컬 llama.cpp(8k 맥락, `--ctx-budget 6000`)로 같은 게이트를 돌려 둔다.

### 5.5 안전·승인 (`gate.rs`)

| 등급 | 예 | 처리 |
|---|---|---|
| 읽기 | 어디 있어, 뭐 있어 | 자유 |
| 일반 이동·조작 | 갖다 놔, 가져와 | 대상이 하나로 정해지면 바로. 계획을 먼저 보여 줌 |
| 되돌리기 어려움 | 버리기(쓰레기통), 액체 붓기, 문 열고 나가기 | 대상이 여러 개거나 최근 10분 안에 못 본 물체면 **확인**("이 컵을 버릴까요?"), 하나로 확실하면 계획 보고만 |
| 금지 | 칼·뜨거운 것 사람 쪽으로, 계단·출입 금지 영역, 사람에게 물건 던지기 | 거절 + 이유. 금지 영역은 지도 위 다각형 |

- 비상 정지: 앱 버튼·"멈춰" → LLM 거치지 않고 `Backend::stop()` (< 100 ms).
- 프롬프트 주입: 사용자 입력만 지시로 취급, 기억·검출 이름·도구 결과는 **데이터**로 감싸 넣는다(`<memory>…</memory>`), 라벨은 고정 어휘라 주입 통로가 좁다. 셸·파일 도구 없음.
- 앱 연결은 토큰 인증(로봇1 소유자만), API 키는 `~/.config/behavior-2026/kau.env` 환경변수로만(기록·인자에 없음 — 기존 규칙).

---

## 6. 수업 15주와 맞추기

각 주 주제마다 **에이전트에 실제로 하나를 더하고, 결과물 하나**를 남긴다. Project 1·2 시점은 강의계획서에 날짜가 없어 Project 1 = 8주차 무렵, Project 2 = 13~15주차로 가정했다(결정 필요: 수업 공지 확인).

| 주차 | 주제 | 에이전트에 더하는 것 | 결과물 |
|---|---|---|---|
| 1 | Agent vs Workflow | 질문 = workflow(T0), 명령 = agent(T2) 로 나눈 이유를 표로 | 3.8절 표 |
| 2 | Agent Loop & Tool Calling | `ragent` 뼈대: raw loop, 도구 8개 스키마, 오류 관찰값 | `ragent ask` 로 scene.json 질문 |
| 3 | Context Engineering & ReAct | `ctx.rs` 16k 예산, system 글, 도구 설명 다듬기 | 맥락 크기 실측 표 |
| 4 | State & Memory | 대화 요약·`remember`, 작업 상태 `task.json` | 요약 전후 정답률 |
| 5 | Knowledge & Retrieval | 물체 기억 = retrieval tool(`search_objects`: 이름 → 생김새 재검색, `confirm_object`), 한국어 사전 + 임베딩(Qwen3-Embedding, query 지시문 붙임 — week05 조사 결과) 후보 확장, 점수 하한 | `qa_where` v1 (120) |
| 6 | Planning & Reasoning | `set_plan` + 검증기 + 상태 기계, Plan-and-Execute vs ReAct 비교 | `cmd_mock` 성공률·호출 수 비교표 |
| 7 | Agent Framework / MCP | 같은 도구를 MCP 서버로도 노출(`ragent mcp`, stdio) — 수업 비교용, 본 경로는 raw | raw vs MCP 지연·코드량 비교 |
| 8 | Evaluation | 골든 세트·채점기·게이트·기준선 3개 | **Project 1**: 기억 QA 에이전트 + 평가 보고 |
| 9 | Failure Diagnosis | 실패 분류표(4절) 자동 태깅, trace 뷰어 | 실패 상위 5개와 고친 것 |
| 10 | Security & Safety | `gate.rs`, 금지 영역, 레드팀 40 | 레드팀 결과표 |
| 11 | Multi-Agent | agent-as-tool 실험: "기억 답변기" 를 하위 에이전트로 뺐을 때 비용·정확도 → 단일 에이전트 유지 여부 결정, LLM 채점기 | 조율 비용 측정표 |
| 12 | Production / Durable | `checkpoint.rs`, 재시작 이어 하기, 프롬프트·스펙 버전(`SPEC_VERSION`, 프롬프트 sha1 trace) | kill 시험 통과 |
| 13 | Build & Red-Team | 시뮬 R1Pro 한 판 + ablation(기억 없음/복구 없음) | ablation 표 |
| 14 | Evaluation & Freeze | 게이트 최종 수치, 데모 고정(태그) | 재현 스크립트 |
| 15 | Final Defense | 데모(질문 + 명령 + 실패 복구 1개) | **Project 2** 발표·구두 방어 |

---

## 7. 단계별 일정 (P0 ~ P5)

| 단계 | 기간 | 할 일 | 끝난 기준 |
|---|---|---|---|
| **P0 기억 QA** | 10-03 ~ 10-12 | `ragent` 크레이트, `Memory` trait(`scene.json` 읽기), `rooms.rs`(시뮬 GT·규칙), `landmark.rs`, `lexicon.rs`, `answer.rs`, T0/T1 | `qa_where` ≥ 85 %, `qa_absent` 환각 0, T0 p95 < 0.5 s |
| **P1 명령(가짜 세계)** | 10-13 ~ 10-26 | `skillspec` 분리·골든, `set_plan`·검증기·상태 기계·복구 표, `Backend::Mock`(behavior-2026 `mockworld` 재사용), 평가 게이트 | `cmd_mock` ≥ 80 %, 복구 ≥ 60 %, Project 1 제출 |
| **P2 시뮬 연결** | 10-27 ~ 11-16 | `MemView` C ABI(같은 프로세스, simlink), `Backend::Sim` 경로 A(1위 모델 + 단계 번호)·B(이동 제어기), 실패 진단·보안 | R1Pro 에서 "컵(물체)을 X 에 넣어" 류 3과제 중 1개 이상 성공, trace 로 실패 분류 |
| **P3 VLA 스킬 재학습** | P2 와 나란히(GPU 확보되면) | `skillspec annotate` → Comet 로더(`fine_grained_level=1`), 70/30 혼합, LoRA. probe 로 문장 따름 측정 | 문장 바꿈 행동 변화 ≥ 잡음 × 3, 스킬 구간 성공 ≥ 지금 + 10 %p |
| **P4 리모** | 11-17 ~ 12-07 | ROS 2 `Backend::Limo`(r2r: Nav2 목표·팔 VLA·그리퍼), 앱 WS 스트리밍, 방 그리기, 리모 시연 수집(같은 틀로 주석) | `cmd_limo` 10 × 3 중 ≥ 50 %, 질문 정답률 시뮬과 같은 수준 |
| **P5 고정·방어** | 12-08 ~ 12-14 | 게이트 최종, 데모 고정, 보고서 | 5.2 최종 열 |

리모 시연 주석: 텔레옵 중 앱/키보드로 스킬 시작·끝을 찍고(`skill_description`, `object_id`) → 같은 `skillspec annotate` 로 문장. 처음엔 `pick up from`·`place in`·`place on` 3스킬 × 50판(추정).

---

## 8. 위험과 열린 질문

| 위험 | 영향 | 대응 |
|---|---|---|
| Qwen3.5-9B 도구 호출 신뢰도(16k) | JSON 깨짐·엉뚱한 도구 | 도구 8개 이하·enum·짧은 설명, `<tool_call>` 글 복구(이미 있음), 오류 관찰값 1번 재시도 후 규칙 대체, 정상 경로에서 LLM 호출 자체를 줄임 |
| 9B 판단이 증거를 무시 | 같은 단계 반복 | 판정은 `verify.rs` 가 하고 LLM 은 판정을 바꾸지 못함(기존 "자동 증거 + 되묻기" 보다 한 단계 강하게) |
| KAU API 지연(7~12 s/호출)·공유 서버 | 명령 체감 느림, 대량 평가 불가 | T0 경로, 스트리밍, 접수 메시지, 로컬 llama.cpp 대체. (결정 필요) KAU 를 대량 평가에 써도 되나 |
| 4090 한 장(24 GB) | 시뮬 + π0.5 + 학습 동시 불가, LoRA 도 22.5 GB 초과 | 시간 나눔, 재학습은 외부 GPU(결정 필요: 클라우드/연구실) |
| VLA 가 문장을 안 따름 | 에이전트 계획이 행동에 안 닿음 | 2.4절 대체 경로 A/B/D, P3 재학습, 효과 게이트 |
| Comet 레벨 번호 혼동 | 잘못된 레벨로 학습 | 두 칸에 같은 문장(2.1절) |
| 방 이름 | "부엌" 답이 틀림 | 시뮬 GT 는 개발용만, 평가는 BDDL+규칙, 리모는 사람이 그림. (결정 필요) 대회에서 방 분할 지도 허용 여부 |
| 같은 라벨 여러 개 | "빨간 컵" 구분 못 함 | 되묻기 + `describe_object(with_image)` best view 로 VLM 판단, 색 속성은 scenemap 에 요청 |
| 시뮬 ↔ 리모 몸 차이 | R1Pro 두 팔·3카메라 vs 리모 한 팔·2카메라 | 계약은 몸과 무관(문장·확인 신호 trait), VLA 는 따로 학습 |
| 기억 갱신 실패(moved/removed 0) | "N분 전" 답이 틀림 | 벤치마크 게이트(5.3), M2 를 답에서 확신도로 드러냄 |
| 수업 규칙(프레임워크 금지, 개인 작성) | 재사용 범위 | 재사용하는 behavior-2026 계획기도 본인 코드. MCP 는 비교용으로만 |

열린 질문(사용자에게):

1. Project 1·2 마감일과 범위(평가 데모가 시뮬이어도 되는지).
2. 대회 평가에서 미리 만든 지도·방 분할 사용 허용 여부(Discord 질문 이어서).
3. 재학습 GPU(클라우드 4090×1 이상 / 연구실 서버)와 시점.
4. 리모 기본형/프로, 매니퓰레이터 모델 — `Backend::Limo` 와 확인 신호(그리퍼 전류 등)가 여기에 달림.
5. 쓰레기통에 넣기 같은 "버리기"를 확인 없이 해도 되는지(5.5절 기본값: 하나로 확실하면 확인 없음).
