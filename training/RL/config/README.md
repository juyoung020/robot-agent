# 설정 (G3)

Rust 실행기(`../ppo/driver`)가 읽는 JSON. 빠진 키는 실행기 기본값(`main.rs` `make_config`)을 쓴다.

| 파일 | 내용 |
|---|---|
| `ppo_a0a1.json` | G3: G1 CURRICULUM A0 → A1, G2 자라는 지도 토큰, 목표 참값(특권). 아래 값 |
| `ppo_a0a1_bound.json` | G3 와 같고 `bound_coef` 0.5 만 더함(벽 충돌 고침 확인용, 10 분) |
| `ppo_a0a1_coll.json` | 위 + 충돌 추가 벌 `shaping.coll` −20(벽 충돌 원인 확인 — 결정적 충돌 0.0298 → 0.0006) |
| `ppo_g4.json` | **G4**: 목표는 지도에서(`goal_from_map` 1), 처음 지도 커리큘럼 A0C0 → A1C0 → A1C1 → A1C2, `bound_coef` 0.5, 충돌 추가 벌 `shaping.coll` −20(합 −30, 기본) |
| `ppo_g4_notok.json` | G4 와 같고 지도 토큰만 끔(`use_map` 0) — 토큰 켬/끔 비교 |
| `ppo_a2.json` | A2(가구, `../env/README.md`): A0C0 → A1C0 → A2C0(≥ 0.85, CURRICULUM A2) → A2C1 → A2C2, 충돌 추가 벌 −20, 지도 토큰 + 안 본 곳 광선(`use_map` 2), 20 분 |
| `ppo_a2_notok.json` · `ppo_a2_nofront.json` | 같고 `use_map` 0(토큰 끔) · 1(안 본 곳 광선만 뺌) |
| `ppo_pnp.json` | **E6 잡기 물리**: B4 집기 → B5 놓기 → B6 가져오기(C0 → 섞음) → B1–B6 섞음 + 실패 판, 행동 8, 모양 잡기 끔. 아래 "ppo_pnp" |
| `ppo_b.json` | **E2 BEHAVIOR**(CURRICULUM_BEHAVIOR2026 3·3.1·5.4절): B0 = 상자 방 A0C0 → A1C0 → A2C0(회귀 단계) → B1 집 안 이동 C0 → C1 → C2 → B2 찾기 → B3 다가가기 → B1–B3 섞음. 아래 "ppo_b" |
| `ppo_a2_ft40.json` · `ppo_a2_ft20.json` | A2 켬 씨앗 1 체크포인트를 `--resume` 으로 A2C2 단계만 10 분 더(충돌 추가 벌 −40 · −20 대조). `../ppo/README.md` "A2 충돌 빠른 시험" |

`use_map`: 0 = 지도 입력 끔, 1 = G3/G4 지도 토큰, 2 = + 안 본 곳 광선 8(`../observation/README.md`). `ppo_run --seed S` 가 설정의 씨앗을 덮어쓴다.

| 키 | 값 | 근거 |
|---|---|---|
| `n_env` × `horizon` | 4,096 × 64 | 계획서 8.1 |
| `epochs`, `minibatches` | 5, 4 (미니배치 65,536) | (가정) 흔한 PPO 값 |
| `gamma`, `lam`, `clip`, `vclip`, `vf_coef`, `ent_coef` | 0.99, 0.95, 0.2, 0.2, 0.5, 0 | (가정) |
| `lr`, `adaptive_lr`, `kl_target`, `lr_min/max` | 3e-4, 켬(미니배치 KL > 2·목표면 ÷1.5, < ½·목표면 ×1.5), 0.01, 1e-5/1e-3 | (가정) 장치 값으로 조절 |
| `max_grad_norm` | 1.0 | (가정) |
| `init_logstd` | −0.5 | (가정) |
| `act_dims` | 2 | CURRICULUM 3절: approach 는 vx, wz 만 학습. G1 행동 8 중 나머지 6 은 0 고정 |
| `shaping` | coef 1, near 8, aim 1, zone 1, v 4, w 2 | `../reward/README.md` (가정) |
| `curriculum` | 단계 [A0, A1], 넘어가기 = 최근 20 바퀴 확률적 성공률 평균 ≥ 0.9, 멈춤 = 마지막 단계 ≥ 0.98 | CURRICULUM 4절(≥ 90 %). "최근 1,000 판" 대신 바퀴 평균 (가정) |
| `budget_minutes`, `ckpt_every`, `inflight` | 30, 200 바퀴, 3 | 띄워 둔 바퀴 수(기록 링 16 칸 안) |

G4 에서 더한 키:

| 키 | 값 | 근거 |
|---|---|---|
| `goal_from_map` | 1 | 계획서 5.4: 교사도 지도에 확정된 목표만 앎(`../observation/README.md`). 0 = G3 특권 |
| `shaping.coll` | −20(G4 두 설정 기본, `ppo_a0a1_coll`). G4 첫 학습 판(README 표)은 0 으로 돌렸음 | (가정) 충돌 스텝에 더하는 보상. `../reward/README.md` |
| `bound_coef` | 0.5 | (가정) 정책 평균이 ±1 밖이면 coef·(|μ|−1)². `../network/README.md` |
| `map_kmin`, `map_kmax`, `map_reveal_r` | 3, 6, 1.5 m | C1: 참 물체 9 개 중 3–6 개(33–67 %, 계획서 30–70 %) 미리 확정, 그 둘레 1.5 m 격자 공개 (반경은 가정) |
| `curriculum.stages[]` | `{name, env, map: [C0, C1], promote, metric}` | `env` = G1 단계(A0/A1), `map` = 처음 지도 비율(나머지 C2), `metric` = 넘어가기를 재는 처음 지도(0 C0, 1 C1, 2 C2, −1 전체). 정수만 쓰면 예전 꼴(G3) |
| 단계 | A0C0 [1,0] ≥ 0.9 → A1C0 [1,0] ≥ 0.8 → A1C1 [0.2, 0.8] ≥ 0.7 → A1C2 [0.1, 0.1] | 계획서 5.5: C0 ≥ 0.8, C1 ≥ 0.7. 앞 단계 약 20 % 유지(가정) — C1 단계는 C0 20 %, C2 단계는 C0·C1 각 10 %. A0 는 G3 처럼 ≥ 0.9 |
| `window` | 20 바퀴, 에피소드 가중 | 지도 비율을 바꾸면 이미 띄운 바퀴(예전 비율)의 기록은 넘어가기 판단에서 뺌 |
| `budget_minutes` | 35 | 토큰 켬/끔 두 판 + 확인 판 10 분 = GPU 약 80 분 |

## ppo_b — BEHAVIOR 집 장면(env 3, 커리큘럼 B1–B3) 키와 가정

env 3 단계가 하나라도 있으면 실행기가 `beh` 1 로 학습기를 만든다 → 장면 묶음(`env/src/bscene_host.cpp`, RASC `~/ra_b1k` 7 장면, 호스트 약 9 s·장치 61.2 MB)을 환경·지도에 붙인다. 상자 방 단계(A0–A2)는 장면 묶음이 붙어도 결과 비트가 같다(`ppo/README.md` E2 절).

| 키 | 값 | 근거 / 가정 |
|---|---|---|
| `beh.scenes` | `house_single_floor`·`house_double_floor_lower`·`restaurant_diner`·`office_cubicles_right` | CURRICULUM 3절 제안(학습 집 4, 평가 집 = `Rs_int`·`hotel_suite_large`·`house_double_floor_upper`). 이름 → 장면 묶음 비트는 학습기를 만든 뒤 `ppo_scene_mask`. 비우면 묶음 전부. 단계 `b.scenes` 가 덮어씀 |
| `beh.build` | 없음(= RASC 폴더 전부) | 만들 장면만 고르기(장치 메모리 줄이기) — 평가에 다른 집을 쓰려면 다 만든다 |
| `beh.mix` | [0.34, 0.33] (B1, B2; 나머지 B3) | `env_verify`·`kBCurrDefault` 와 같은 기본. 단계 `b.mix` 가 덮어씀 |
| `beh.split` | 0(학습 인스턴스) | CURRICULUM 3절: 공개 평가 인스턴스(1)는 평가만 |
| `beh.strict` | 0(느슨 거르개) | 3.1절 표. 엄격(1)은 E0 엄격 한도 판만 |
| `beh.eval_instr` | 0 | 1 = 지시문 heldout 문장(평가) |
| `beh.p_point` / `beh.p_goto` | 0.3 / 0.25 | **(가정)** 목표 점(CURRICULUM_BEHAVIOR2026 3.2, POLICY 4.6): 집기·놓기 판 놓을 곳을 점으로(+ "put the {o} here") / B1·B3 를 점으로 가기로. 단계 `b` 가 덮어씀. 0 = 예전 판 |
| `beh.yaw_jit` | 0.5 rad | **(가정)** B1 시작 yaw = 인스턴스 R1Pro 자세라 ±0.5 rad 흔듦(외우기 방지). 집기·놓기 판은 시작이 원래 무작위 |
| `beh.nav_k` | 10 | 지도 다가가기 거리장 주기(map README E2, `nav_tradeoff`: K 10 의 판 보상 합 차 0.12 / 8.25) |
| 단계 `b` | `{mix, scenes, split, strict, eval_instr, yaw_jit}` | env 3 단계에 들어갈 때 장치 값 `bsc::BCurr` 로(같은 환경이면 장치 커리큘럼 커널이 바로, 다른 환경이면 실행기가 `ppo_set_bcurr`) |
| `metric` | 3 / 4 / 5 = B1 / B2 / B3 에피소드 | 0–2(C0–C2)·−1(전체)는 예전 그대로. 끝난 판의 단계 = 환경 `I_B_LKIND` |
| 단계 문턱 | B0: A0C0 0.9 · A1C0 0.8 · A2C0 0.85, B1C0 0.85 · B1C1 0.7 · B1C2 0.7, B2 0.85, B3 0.85, 마지막 B123 멈춤 없음 | CURRICULUM 3절 표(B1 SR ≥ 85 %, B2 찾음률 ≥ 85 %, B3 SR ≥ 85 %)·5.5(C1 0.7). B1 의 C1·C2 문턱 0.7 은 G4 와 같게 **(가정)** |
| 앞 단계 섞기 | B2 = B1 20 %, B3 = B1·B2 각 10 %, 지도는 C2 단계에서 C0·C1 각 10 % | CURRICULUM_APPROACH 4절 "앞 단계 약 20 %" **(가정: 나누는 법)** |
| `act_dims` | 2(vx, wz) | B1–B3 모두 이동·다가가기(팔은 홈 자세 — B3 성공은 몸통 자세의 잡는 점 작업 공간) |
| `budget_minutes` | 60 | **(가정)** 학습 시간은 아직 안 잼(이 작업은 연결·검증만) |

## ppo_pnp — 잡기 물리 판(E6, 커리큘럼 B4–B6) 키와 가정 (2026-10-05)

env 3 그대로(장면 묶음), `beh.pnp`·`b.pnp` = [B4, B5, B6] 비율(B3 몫에서 뗌 — 단계 고르기 u < p1 → B1, < +p2 → B2, < +p4 → B4, < +p5 → B5, < +p6 → B6, 나머지 B3), `fail` = [p_slip, p_occ](실패 판). 판정·보상은 POLICY 4.8, 모형은 CURRICULUM_BEHAVIOR2026 5.5.

| 키 | 값 | 근거 / 가정 |
|---|---|---|
| `act_dims` / 단계 `act_mask` | 8 / 255 | 집기는 팔·그리퍼 6 행동이 필요(VLA_INPUT 5절) |
| `shaping.coef` | 0 | 다가가기 퍼텐셜(겉면 0.4–0.8 m 에 서기)은 팔 닿는 거리(0.2–0.35 m)와 맞섬 — B4–B6 의 손·들기·놓기 퍼텐셜은 환경 보상 안에 있음 |
| 단계 | B4(C0, ≥ 0.8, 지표 6) → B5(C0, ≥ 0.8, 7) → B6C0(≥ 0.7, 8) → B6C2(C0·C1 각 10 %, ≥ 0.6, 8) → B1-6 섞음(실패 판 p_slip 0.005·p_occ 0.2) | CURRICULUM 3절 문턱(SR ≥ 80 %). B6 문턱은 **(가정)** |
| `budget_minutes` | 60 | **(가정)** — 처음부터 PPO 로는 90 s 시험에서 B4 성공 0(`../ppo/README.md` E6 절). 대본 교사 시연은 BC 쪽(`../../BC/config/bc_pnp_lite.json`, `teacher_script` 1)에서 쓰임. BC 학생 → PPO 교사 변수로 옮기는 길은 아직 없음(남은 일) |
