# 남은 일 추적표 (설계·계획 문서 전체)

2026-10-04 기준. 문서 글이 아니라 **코드와 git log** 로 상태를 정했다.
기준 커밋: robot-agent `708db02`, 서브모듈 `src/behavior-2026` `c4548fe`, LIMO 작업 클론 `~/behavior-2026-limo` `c4548fe`.
작업 트리의 커밋 안 된 변경(G6 FP8 학생: `training/BC/*`, `training/RL/{network,ppo}/*`, `ppo_a0a1_g6_*.json`)은 "진행 중" 으로만 본다.

상태 표기
- **완료**: 코드가 있고 문서의 통과 기준을 넘음(증거에 경로·커밋)
- **일부**: 일부만 있음
- **안 함**: 시작 안 함
- **대체됨**: 다른 길로 바뀌어 더 할 필요 없음
- **결정 필요**: 사용자가 정해야 진행됨
- **진행 중**: 지금 다른 에이전트가 작업 중(이 표에서 다시 손대지 않음)

담당 영역: env(시뮬·학습 환경) / map(지도) / policy(RL·BC·VLA) / viewer / robot(로봇·에이전트·실기) / docs

---

## 1. GPU_TRAINING.md

### 11절 단계 G0–G7

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| G0 측정 틀: FP8 GEMM V3 | 완료 | `training/RL/network/tools/fp8_verify.cu`, `src/gemm_fp8.cuh` — `708db02` (V3 7/7, 음성 대조 3/3) | — | policy |
| G0 조건 노드·PDL 고정비, cuBLASLt 비교 | 안 함 | `training/RL/ppo/README.md:44` "조건 노드(WHILE)는 쓰지 않았다"(펼친 그래프로 동기 0). cuBLASLt 비교 코드 없음 | 필요 없으면 문서에서 "안 함(이유)" 로 닫기 | policy |
| G1 LIMO+OMX 환경 커널 + CPU 참조 + 뷰어 | 완료 | `training/RL/env/` (`env.h`, `env_verify.cu` 비트 동일, `env_view.cpp`), `training/RL/tools/urdf2hdr` — `0eddb72`, 렌즈 카메라 `436c2db` | 가정 값(a_v·a_w·서보 이득·XL330 속도, `env.h:29-42`)을 실측/사양으로 | env |
| G2 GPU 지도 근사 + scenemap LIMO FK + 비교 도구 | 완료 | `training/RL/map/` — `137daec`, `1bbd634`, `36b41f2`, `4f3accc`, `d447f5c`; `training/RL/map_cmp/` — `d1a0c50`, `f66c173`; scenemap LIMO FK — 서브모듈 `7bf4d19`, `7219187`, `48f9f90` (`a2abc45`, `886aa60`) | — | map |
| G3 RL 교사 PPO (그래프 둘, BF16) | 완료 | `training/RL/ppo/` — `dbc0b3b`, 속도 `5cf891a` (update 135 → 87 ms) | — | policy |
| G4 자라는 지도 커리큘럼 C0→C1→C2 + 완성도별 평가 | 완료 | `ec4e36d`, `e73f873`, `b24a76f`; A2 `bc23d06`, `c96c367`, `2d2939b`, `21c7d26` | — | policy |
| G5 BC 학생 + DAgger + 영상 학생 | 완료 | `training/BC/` — `31dc3f5`, `44b409c`, `b27aafd`, `3b09389` (DAgger 8 0.946 = 교사 0.944) | 남은 일은 아래 training/BC 절 | policy |
| G6 FP8 앞 → dgrad → wgrad, 9.1 BF16 대 FP8 학습 비교 | 진행 중 | GEMM `708db02`; 학생 FP8·`ppo_a0a1_g6_{bf16,fp8}.json` 커밋 안 됨 | (다른 에이전트) | policy |
| G7 BC 큰 VLA (얼린 Qwen FP8) | 안 함 | Qwen 코드 없음(`training/model/Qwen3.5-2B` 가중치만) | G6 끝난 뒤. 돌릴 곳 결정(VLA_INPUT 8절)에 달림 | policy |

### 본문 다른 항목

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 4.2 팀 학습기 호스트 동기 없애기 | 대체됨 | 팀 `pi05_train` 대신 자체 학습기(G3·G5)를 동기 0 으로 만듦 | — | policy |
| 4.5 렌더 / 4.6 BC 데이터 로더(NVDEC) | 대체됨 | 팀 `RenderBatch` 그래프 잡기로 학습 때 다시 렌더(`b27aafd`, `44b409c`). 저장 영상·NVDEC 안 씀 | — | env |
| 5.3 지도 토큰 시계열 기록 | 완료 | BC 기록이 스텝마다 본 지도 토큰 저장 — `31dc3f5` | — | map |
| 5.4 교사 특권 (나)·(다), 비대칭 가치 | 안 함 | 가치 머리도 가린 입력만 봄(`training/RL/ppo/README.md:147`) | 집기 스킬 시작 때 같이 | policy |
| 6 커널 합치기 | 일부 | 칸 MLP 묶음 커널·관측 모으기 `5cf891a` | 칸 MLP 부분 속도(ppo README 다음 5) | policy |

### 12절 열린 문제

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| FP8 혼합 형식 e5m2×e4m3 | 완료 | `708db02`, `fp8_verify probe` | — | policy |
| 8 비트 ldmatrix 전치 | 완료 | `708db02` | — | policy |
| 블록당 공유 메모리 한도 | 완료 | `708db02` (99 KB) | — | policy |
| CUDA 12.8 고정 (sm_120f·mxf4 는 12.9+) | 결정 필요 | 저장소 기준 12.8 | 사용자: 올릴지 | policy |
| 검출 실수·slam 오차 모델 | 일부 | LIMO 기록으로 잡음 보정 `101b36a`, `7a474a0`, `c6b5cc5`. 거리별 놓침은 아직 R1 기록, rec.bin 있는 LIMO 판 없음 | rec.bin 켠 LIMO 탐색 판 하나 더 → 놓침·검출 잡음 다시 맞춤 | map |
| 224² 렌더 속도 | 완료 | `44b409c` (싼 설정 21k·팀 기본 3.0k 판·프레임/s) → 온라인 렌더로 결정 | — | env |
| scenemap R1 Pro 전용 | 완료 | 서브모듈 `7bf4d19`, `7219187`, `3daa1fb`, `171e185` | — | map |
| Python 기준값 범위 | 일부 | 오프라인 한 번 덤프만 씀(`training/BC/tools/siglip_ref.py`) — 사실상 정해짐 | 문서에 확정으로 적기 | docs |

---

## 2. POLICY.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1.3 VLA 실행기 접점(objects·끝 신호·결과 형식·안전 거르개·LIMO 부분) | 진행 중 | 지금은 `move_robot` 몸 크기만 LIMO(`0705f51`, `nav.rs:331`), 시뮬 쪽 LIMO 어댑터(베이스만, 서브모듈 `50c971c`) | (다른 에이전트) | robot |
| 3.1 스킬: approach | 일부 | A0–A2 완료(위 G3·G4). A3–A5 안 함 | CURRICULUM 정리 뒤 A3 | policy |
| 3.1 스킬: pick / place / drawer / door / press | 안 함 | 환경은 몸통 (vx, wz) 만 학습, 팔 고정(`env.h`) | 집기 환경(팔 행동·물체 접촉) 설계부터 | env |
| 3.3 보상 기본 구조 | 일부 | approach 보상(경로 거리·에임·충돌 벌) `bc23d06`, `b24a76f`. pick 보상 없음 | pick 때 | policy |
| 3.4 무작위화 | 일부 | 지도 잡음·map_drift `c6b5cc5`, A2 가구 배치. 물리·렌더 무작위화 없음(`bc_render.cu` 조명 고정) | 환경 개선 때 렌더·물리 무작위화 | env |
| 4.2 관측 세 단계 (가)(나)(다) | 일부 | (가)만. 특권 (나)·비대칭 (다) 없음 | pick 때 | policy |
| 4.3 신경망 (칸 MLP + 셀프 어텐션) | 일부 | 칸 MLP + 풀링만(`training/RL/network/README.md:10`), 어텐션·GRU 없음 | 필요할 때 | policy |
| 4.5 커리큘럼 1–4 단계 | 대체됨 | GPU_TRAINING C0–C2 로 구현(`ec4e36d`). 이름 통일은 BEHAVIOR 2026 커리큘럼 문서에서 | — | docs |
| 6 학습 순서 0·1·2a·2b | 완료(approach 기준) | G1–G5 커밋들 | — | policy |
| 6 2c 섞기(BEHAVIOR R1 시연·다른 팔·VQA) | 안 함 | — | 커리큘럼 정한 뒤 | policy |
| 6 3 학생 RL 미세조정(잔차 PPO) | 안 함 | — | — | policy |
| 6 4 실제 로봇 미세조정 | 안 함 | 실기 없음 | 리모 받은 뒤 | robot |
| 7.1 `mr_filter` 행동 거르기 C ABI | 안 함 | `src/agent/tools/move_robot` 에 없음 | 실행기 접점(1.3) 작업과 묶음 | robot |
| 7.2 끝·돌려주기 조건 | 진행 중 | 1.3 과 같은 작업 | (다른 에이전트) | robot |
| 8 평가(처음 보는 이름·집·gt 대 slam·문장 반응) | 일부 | 완성도별 평가만(`e73f873`). 집·이름·문장 반응 없음 | BEHAVIOR 집에서 학습한 뒤 | policy |
| 9 OMX-F 작업 공간 뽑기 | 안 함 | 코드 없음 | URDF 한계로 손끝 닿는 범위 계산(집기 전 필수) | robot |
| 9 팔 행동 표현 / 바퀴 차동·메카넘 / 몸통 A·B / 특권 / GRU / 깊이 / 문장 바꾸기 / 렌더 / 학생 RL / 넘김 거리 / 돌릴 곳 | 결정 필요 | 바퀴는 env 가 차동으로 이미 구현, 렌더는 팀 RenderBatch 로 이미 씀 — 이 둘은 사실상 정해짐 | 사용자: 나머지 확정 | policy |

---

## 3. VLA_INPUT.md (8절 "아직 정할 것" + 본문)

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 어디서 돌릴지(리모 위 작은 몸통 / PC Qwen) | 결정 필요 | — | 사용자: 리모 판(기본형·프로) 확정 뒤 | policy |
| SigLIP 2 패치 토큰 내는 인코더 | 완료 | `training/BC/src/vit.cu` — `b27aafd` (코사인 0.9996) | — | policy |
| 리모 바퀴 모드 | 결정 필요(사실상 차동) | env 차동 구현 | 문서에 확정 | docs |
| OMX-F 작업 공간 | 안 함 | 위 POLICY 9 와 같음 | — | robot |
| 지금 보이는 물체의 즉시 3D C ABI | 일부 | `sm_last_assoc`·`sm_last_views`(id·품질만, `scenemap.h:192-195`). 지도 넣기 전 3D 값 내는 함수 없음 | 필요해지면 `sm_last_dets3d` 같은 함수 | map |
| 3절 물체 칸 이름 뜻·생김새 벡터 | 안 함 | 지금 one-hot(`training/RL/network/README.md:14` "(가정: 표가 아직 없음)") | embed `text128_*`·`head128` 표를 붙일지 결정 | policy |
| 7절 외웠는지 평가 | 안 함 | — | 이름 벡터 넣은 뒤 | policy |

---

## 4. TRAIN_VIEWER.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| V0–V5 전체(`trainfmt`, 서버, 성공 표, `.trp` 재생, SSE·비교, 영상) | 진행 중 | `training/viewer/` 아직 없음. 학습기는 `config.json`·`log.csv`·`results.json` 을 씀(`run.json`·`progress.jsonl` 아님) | (다른 에이전트) | viewer |
| 12절 정할 것 13 개 | 진행 중 | 위 작업에서 정함 | — | viewer |
| 학습기 쪽 쓰기(run.json·episodes·trp) | 안 함 | PPO·BC 드라이버(`ppo/driver/src/main.rs:287`, `BC/driver/src/main.rs:186`)가 옛 형식 | 뷰어 형식이 정해지면 드라이버에 쓰기 추가 | policy |

---

## 5. CURRICULUM_APPROACH.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| A0, A1 | 완료 | `dbc0b3b`, `b24a76f` (결정적 0.992) | — | policy |
| A2 (가구, ≥ 85 %·충돌 ≤ 5 %) | 완료(씨앗 1) | `bc23d06`, `c96c367` (0.946, 충돌 0.032), `21c7d26` (0.023). 씨앗 2 는 충돌 0.055 | 씨앗 2 충돌 줄이기(가구 가까움 벌) | policy |
| A3 BEHAVIOR 집 한 방 + slam 잡음 | 안 함 | 환경이 상자 방만 | **BEHAVIOR 2026 집 배치를 환경에 넣기** | env |
| A4 집 전체·문 지나기, A5 오래된 지도 | 안 함 | — | A3 뒤 | env |
| 5절 지표 SR·SPL·Soft SPL·끝 자세 오차 | 일부 | 성공·충돌·시간 초과만. SPL 없음 | 평가에 최단 경로 길이 기록 → SPL | policy |
| 6절 다음 커리큘럼(집기·들고 이동·놓기·잇기) | 안 함 | — | BEHAVIOR 2026 커리큘럼 문서에 따름 | policy |
| BEHAVIOR 2026 과 맞춘 커리큘럼(CURRICULUM_BEHAVIOR2026.md) | 진행 중 | 파일 아직 없음 | (다른 에이전트) | docs |

---

## 6. MAP_STATE_PLAN.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1 벤치마크 raw 3 개 받기·validate | 진행 중 | 어댑터 작업에 포함 | (다른 에이전트) | map |
| 2 어댑터: 시퀀스 → scenemap C ABI | 진행 중 | — | (다른 에이전트) | map |
| 3 실제 검출(FastSAM + SigLIP 2) | 일부 | SigLIP 2 는 `sgrt_clip` 에 붙음, 검출기는 아직 YOLOE(FastSAM 코드 없음) | 어댑터 뒤 | map |
| 4 지도 → state 변환기(슬롯 16 × 14) | 대체됨 | VLA_INPUT 칸 형식 + GPU 지도 토큰(`1bbd634`) 으로 바뀜 | — | map |
| 5 우리 로봇 데이터와 짝지어 학습 | 대체됨 | 시뮬 안 지도로 학습(G4·G5), 지도 켬/끔 비교 `c96c367` | — | policy |

---

## 7. SIM_PORTING.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| M0 URDF + FK (기준 TF 값) | 완료 | `training/RL/tools/urdf2hdr`, `limo_omx_model.h` — `0eddb72`; scenemap FK `7219187` | — | env |
| M1 관절 PD + 베이스 적분 (GPU 병렬) | 완료 | `env.h` — `0eddb72` (7.8e8 env-step/s) | — | env |
| M2 depth 광선 투사 + BVH (MJWarp 비교) | 대체됨 | 팀 `RenderBatch` 사용 `b27aafd`. 지도 근사는 자체 광선 | — | env |
| M3 패킹 + CUDA graph | 완료 | G3·G5 그래프, 호스트 동기 0 | — | policy |
| M4 물체 접촉 | 일부 | A2 상자 막힘 `bc23d06`. 팔-물체 접촉 없음 | 집기 때 | env |
| 4절 시뮬 용도·검증용 Python 허용 | 대체됨 | RL 에 씀, Python 은 오프라인 덤프만 | — | docs |
| 4절 LIMO·손목 카메라 위치 실측 전 | 일부 | 렌즈 프레임 `72a9f7a`, `391c04b`. 실측 없음 | 리모 받은 뒤 실측 | robot |

---

## 8. PORT_PLAN.md (MuJoCo Warp 포팅)

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| A–G 단계 전체, libmujoco 받기 | 대체됨 | BEHAVIOR_ENGINE_REVIEW 6절 제안대로 자체 env 커널(G1) + 팀 `RenderBatch` 로 감. MuJoCo 코드 없음 | — (표시 완료 `57bd99d`) | docs |

## 9. BEHAVIOR_ENGINE_REVIEW.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 6절 ① 우리 URDF 를 OmniGibson 에 | 완료 | `src/robot/og/` — `efcb433`, `4536d88`, `de769fa`, `8f36e85` | — | robot |
| 6절 ② 엔진에 LIMO 로봇 설정 | 대체됨 | 엔진 확장 대신 자체 env 커널(G1), 렌더만 엔진 `RenderBatch` | — | env |
| 6절 ③ 우리 VLA 학습기 | 완료(작은 학생) | G3·G5 | 큰 VLA 는 G7 | policy |
| 5절 과제 범위(64/100 지원)·보조 잡기·성공 판정 | 안 함(우리 쪽) | BEHAVIOR 과제를 우리 환경에서 아직 안 씀 | BEHAVIOR 2026 커리큘럼과 함께 | env |

## 10. VLA_QnA.md · PI05_TRAINING.md · untitled.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| VLA_QnA 요약 1–6 (SmolVLA/openpi 로 시작, 텍스트 지도) | 대체됨 | 자체 RL 교사 → BC 학생, 연속값 지도 토큰으로 감 | 기록용 | docs |
| PI05_TRAINING 6절: HL 단계 글·VI 다른 말 | 일부 | 지시 문장 표 8 개(`training/BC/tools/text_table.py`, `data/instr_a2.f32`) | 다른 말 섞기는 문장 바꾸기 결정 뒤 | policy |
| PI05_TRAINING 6절: 다른 로봇·VQA 데이터 섞기 | 안 함 | — | POLICY 6 2c 와 같음 | policy |
| PI05_TRAINING 6절: flow matching 행동 청크 | 완료 | 영상 학생 flow 청크 `b27aafd` | — | policy |
| π0.5 자체(학습·추론) | 대체됨 | π0.5 는 버림, 가중치는 일부러 지움(`~/behavior-2026/data/pi05_native` 없음) | 다시 받지 않음 | docs |
| untitled.md (Gemini 대화 원본) | 대체됨 | 정리본 VLA_QnA.md | 기록용 | docs |

---

## 11. src/agent/plan.md (에이전트)

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| P0 기억 질답(`ragent`, rooms·landmark·lexicon·answer) | 안 함 | `ragent` 크레이트 없음 | `qa_where`(scene.json 읽기)부터 | robot |
| P1 `skillspec` 분리·골든 파일·`set_plan`·상태 기계·모의 백엔드 | 안 함 | 문장 생성은 아직 서브모듈 planner `vocab.rs:139`·`instruction.rs:180` | `vocab.rs` 를 `skillspec` 크레이트로 | robot |
| P2 시뮬 연결(Backend::Sim) | 일부 | `move_robot` 시뮬 TCP·C ABI, explore 스킬 실제 시뮬에서 돎. `Backend` trait 없음 | — | robot |
| P3 VLA 스킬 재학습(Comet annotate) | 결정 필요 | 자체 지도 토큰 VLA 로 옮겨감 | 사용자: Comet 재학습을 버릴지 | policy |
| P4 LIMO(Backend::Limo, 앱 WS) | 일부 | 시뮬 LIMO 탐색 서브모듈 `50c971c`, `cdf55ae`; footprint `0705f51`; `tools/run_explore_live.sh` `8beecaa`, `127d10c` | 실기·ROS 2 백엔드는 리모 받은 뒤 | robot |
| P5 굳히기·발표 | 안 함 | — | — | docs |
| 3.3 LLM 도구 8 개(find_object·set_plan·check·ask_user·report·remember) | 결정 필요 | `move_robot` 하나만 있음 | 사용자: 3.3 을 유지할지 move_robot·스킬 모델로 바꿀지 | robot |
| 3.4 상태 기계·복구 표, `verify.rs` | 안 함 | `machine.rs`·`verify.rs` 없음(move_robot 안 재계획·후진만) | 실행기 접점과 함께 | robot |
| 3.5 방 이름 | 일부 | scenemap `rooms.cpp` 규칙표·`sm_set_room_name`. 투표·제로샷 없음 | SigLIP 이름이 방 규칙에 들어가는지 확인 | map |
| 5 평가 `ragent eval --gate`·골든 세트, 5.5 `gate.rs` | 안 함 | — | P1 뒤 | robot |
| decision_log(decisions.jsonl) | 완료 | `skills/explore/src/lib.rs`, `decisions-agg` — `4ad1781`, `f82dc18` | `label.vla_better` 는 VLA 실행기 생긴 뒤 | robot |
| 8 Qwen 도구 호출 안정성 | 일부 | explore 프롬프트 v2–v4, 429 재시도 | — | robot |
| 8 열린 질문 1–5 | 결정 필요 | Q4(팔)는 OMX-F 로 사실상 정해짐 | 사용자 | robot |

## 12. docs/plan.md (팀 계획 할 일)

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1 ROS 2 올리기 | 일부 | `src/robot` ROS 2 패키지, `fetch_upstream.sh`(limo_ros2). 실기 안 돌림 | 리모 받은 뒤 | robot |
| 1 SLAM(Cartographer) | 결정 필요 | 시뮬은 자체 `scenemap/src/slam2d.cpp`. Cartographer 없음 | 사용자: 실기에서 slam2d 쓸지 | robot |
| 1 이동(내비게이션) | 일부 | 시뮬 `move_robot` Dijkstra + DWA. Nav2 없음 | — | robot |
| 1 URDF 합치기 | 완료 | `src/robot/map_vla_description` — `72a9f7a`, `4536d88`, `0c4b83d` | — | robot |
| 1 캘리브레이션, 좌표 연결 확인, 팔 ROS 2, ros2 bag, 토픽 정리 | 안 함 | `scripts/` README 만 | 리모 받은 뒤 | robot |
| 1 TF 트리 | 일부 | `docs/map_vla/tf_tree.pdf`, URDF. 실기 `map→odom` 없음 | — | robot |
| 2 물체 인식 FastSAM-s + SigLIP 2 | 일부 | SigLIP 2 붙음(`sgrt_clip`, 서브모듈 `61eb0a2`, `976e217`, `8c1a7d6`). 검출기 아직 YOLOE | FastSAM-s 붙이고 이름 정답률·R@1 재측정 | map |
| 2 xyz·DA·지도 갱신·벡터 찾기·Spark-DSG 저장 | 완료 | scenemap `objmap.cpp`, `da/`, `sgrt_query_*`, `dsg_save.cpp`, sgview `c4548fe` | known_bugs 7·10·11 | map |
| 2 방 나누기·이름(제로샷) | 일부 | 규칙표만, 제로샷 없음 | — | map |
| 3 Qwen API | 완료 | planner `llm.rs` | — | robot |
| 3 상황 풀기·다시 계획·RAG 답·평가 | 안 함 | agent P0/P1 과 같음 | — | robot |
| 4 π0.5 LoRA·리모용 π0.5 | 대체됨 | π0.5 버림 | — | policy |
| 4 우리 VLA 설계·학습 | 일부 | 작은 학생 G5 완료, 큰 VLA G7 안 함, `src/vla` 는 README 만 | — | policy |
| 4 시뮬에 리모 + 팔 | 완료 | `src/robot/og` — `efcb433`, `de769fa` | — | robot |
| 4 리모에서 실행(Jetson/NPU) | 안 함 | — | 돌릴 곳 결정 뒤 | robot |
| 5 앱(iOS·Android·채팅·WS) | 안 함 | `src/app` README 만 | — | robot |
| 6 합치기: BEHAVIOR 전체 흐름 | 일부 | 탐사만 | — | robot |
| 6 합치기: 리모 시뮬·실기·점수 | 일부 / 안 함 | 시뮬 LIMO 탐사 `50c971c` | — | robot |

## 13. docs/known_bugs.md

모두 아직 코드에 있다(서브모듈·`~/behavior-2026-limo`·`src/scene_graph` 동기화 사본 모두).

| # | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1 ovdet_eval 옛 경로 | 안 함 | `ovdet/tools/ovdet_eval.py:33` | 서브모듈에서 고친 뒤 `tools/sync_scene_graph.sh` | map |
| 2 explore 주석 0.28 m | 안 함 | `explore.rs:91` | 주석 고침 | robot |
| 3–6 `/mnt/c/behavior-2026` 박힘 | 안 함 | 84 파일(문서는 약 80) | 스크립트 위치 기준 뿌리 | robot |
| 7 scene.json 3 층 대 2 층 결정 | 결정 필요 | `test_scene_json.cpp:81-82`, `scenemap.h:362-367`, `dsg_save.cpp:563` | 사용자: 2 층 확정이면 저장·시험·헤더 고침 | map |
| 8, 9 주석·메시지 틀림 | 안 함 | `scenemap/CMakeLists.txt:17,39`, `sgrt.cpp:65` | 문구 고침 | map |
| 10 DA `max_ext`·`kinds` 안 봄 | 안 함 | `da/src/merge.cpp:25` (문서 경로는 틀림: `src/`·`include/da/`) | 확인 넣기 | map |
| 11 이벤트 7 이름 없음 | 안 함 | `objmap.cpp:461`, `dsg_save.cpp:41-42` | "merged" 추가 | map |
| 12 walls 시험 변수 없음 | 안 함 | `scenemap/CMakeLists.txt:100` | 정하거나 뺌 | map |

## 14. training/**/README.md "남은 일"

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| RL/env 가정 값(가속 한도·서보 이득·XL330) | 안 함 | `env.h:29-42` (가정) | 실측·회사 값 | env |
| RL/map 남은 일 1 (5.2 맞추기) | 완료 | `d1a0c50`, `f66c173`, `a2abc45`, `886aa60` | — (README 표시 완료 `57bd99d`) | docs |
| RL/map 남은 일 2 (LIMO 기록으로 다시 맞춤) | 일부 | `7a474a0`, `c6b5cc5`. 거리별 놓침만 R1 | 위 "검출 실수" 와 같음 | map |
| RL/map 남은 일 3 keyframe 커널 970 µs | 안 함 | — | 필요할 때 | map |
| RL/map 남은 일 4 C1 격자 일부를 실제 탐사 경로로 | 안 함 | `map.h:185` 원판 공개 | 환경 개선 때 | map |
| RL/map_calib/limo: 고친 카메라 판 자세 비교, rec.bin 기록 | 안 함 | `~/behavior-2026-limo/outputs/explore_20261004_180748_*_limo_gt` 있음(rec.bin 없음) | 180748 판으로 자세 비교, rec.bin 켜고 한 판 | map |
| RL/map_cmp 잡음 켠 비교 | 안 함 | — | — | map |
| RL/ppo 다음 2 A2 씨앗 2 충돌 0.055 | 일부 | `21c7d26` 씨앗 1 만 | 위 A2 와 같음 | policy |
| RL/ppo 다음 3 광선 씨앗 더 / 4 정책 자기 지도로 경로 거리 / 3 씨앗 비교 | 안 함 | `env.h:255 path_prepare` 는 참 장면 | 환경 개선 때(경로 거리를 정책 지도로) | env |
| RL/network SigLIP 이름·생김새 표 | 결정 필요 | one-hot | VLA_INPUT 3절과 같음 | policy |
| RL/reward 탐색 보상 대조 | 안 함 | — | 선택 | policy |
| BC 남은 일 1 FP8 인코더 | 진행 중 | 커밋 안 됨 | (다른 에이전트) | policy |
| BC 남은 일 1 EMA·학습률 일정 | 안 함 | `BC/src` 에 ema 없음 | — | policy |
| BC 남은 일 2 렌더 무작위화·팀 기본 설정 섞기·로봇 그리기 | 안 함 | `bc_render.cu:90-99` 조명 고정 | **환경 개선**(학습 전) | env |
| BC 남은 일 3 작은 트랜스포머·처음 보는 이름 평가 | 안 함 | — | — | policy |
| embed `sb32_pe_sim` 판 | 일부 | `~/embed_work/runs/sb32_pe_sim/` 에 args.json 만 | 다시 돌리거나 지움 | map |
| embed 여러 집 시뮬 자르기·평가 B·한국어 평가·KorLex·Nano 측정 | 안 함 | `8425e32` 뒤 커밋 없음 | — | map |

## 15. README·src README·docs 기타

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| `src/vla` 폴더 | 안 함 | README 만, VLA 일은 `training/BC` | 가리키는 글 또는 계획 | docs |
| `src/app` ios·android | 안 함 | README 만 | — | robot |
| `src/robot` limo 라이선스 TODO | 결정 필요 | upstream `package.xml` | 사용자: 재배포 여부 | robot |
| model_selection DA 생김새 코사인 | 결정 필요 | `da/` 에 임베딩 비교 없음 | — | map |
| clip_candidates 8.6 Nano TRT 측정 | 안 함 | — | 실기 | robot |
| clip_candidates 8.6-5 한국어 학생 → `sgrt_query_embedding` | 일부 | API 있음(`sgrt.h:155`), 연결 안 됨 | 연결 | map |
| model_selection 앱 WS·VPN | 결정 필요 | — | — | robot |

---

## 문서 정리 필요 (내용이 낡았거나 틀린 문서)

이 추적표에서는 고치지 않았다. 각 문서 주인이 고친다.

| 문서:줄 | 무엇이 틀렸나 |
|---|---|
| `docs/map_vla/README.md` (SIM_PORTING 행) | "코드는 아직 없음" — G1 환경(`0eddb72`)이 있다. PORT_PLAN 행은 "대체됨" 으로. TODO_TRACKER 행 추가 필요. (다른 에이전트가 편집 중이라 손대지 않음) |
| `docs/map_vla/BEHAVIOR_ENGINE_REVIEW.md` 3절 | "π0.5 체크포인트들(`data/pi05_native` 23 GB)" — 일부러 지웠다. 5절 "학습기는 π0.5 용, 우리 학습기 없음" 도 G3·G5 로 낡음 — **완료** `57bd99d` |
| `docs/map_vla/PORT_PLAN.md` | 주의 문구만 있고 "대체됨" 표시가 없다 — **완료** `57bd99d` |
| `docs/map_vla/SIM_PORTING.md` 4·5절 | 용도 미정·Python 허용 미정·`refs/code/behavior-2026` 경로 — 모두 정해졌거나 바뀜 |
| `docs/map_vla/POLICY.md` 1.1 | "`move_robot` 부분·한계는 R1 Pro 기준, LIMO 없음" — 몸 크기 `limo_omx` 있음(`0705f51`). "`act()` 를 π0.5 엔진으로" — π0.5 버림 |
| `docs/map_vla/TRAIN_VIEWER.md` 12절 | "args.json 대 run.json" — 실제 학습기는 `config.json`·`log.csv`·`results.json`(`~/ra_*`) |
| `docs/map_vla/GPU_TRAINING.md` 12절 | "224² 렌더 안 쟀다"·"scenemap R1 Pro 전용" — 둘 다 끝남(`44b409c`, 서브모듈 `7219187`) — **완료** `57bd99d` |
| `README.md:50,57,76,113`, `src/README.md:9,13,15`, `src/agent/README.md:3,10`, `src/vla/README.md:3` | VLA = π0.5 — 버림 — **완료** `57bd99d` |
| `README.md:62` | "매니퓰레이터(모델 미정)" — OMX-F — **완료** `57bd99d` |
| `docs/plan.md` 4절·위험 표 | π0.5 LoRA·리모 π0.5·"매니퓰레이터 모델 미정" — 낡음 — **완료** `57bd99d` |
| `docs/model_selection.md:9,18,19,30,31,161-178` | Cartographer(실제는 자체 slam2d), π0.5 잠정/결정 — 변경 기록에 대체 행 없음 — **완료** `57bd99d` |
| `docs/perception_model_candidates.md:144,192,269,399,409` | π0.5 가 돈다고 가정한 메모리·NPU 예산 — **완료** `57bd99d` |
| `docs/known_bugs.md` 10 | 경로가 `da/merge.hpp`·`merge.cpp` 가 아니라 `da/include/da/merge.hpp`·`da/src/merge.cpp`. 3 번 파일 수 약 80 → 84 — 경로 **완료** `57bd99d`. 3 번 파일 수는 안 고침(세는 기준이 달라 다시 셀 것) |
| `src/scene_graph/README.md:18` | ctest "9/9" — 지금 12 개 — **완료** `57bd99d` |
| `training/README.md:10,24` | Qwen VLA(아직 G7), `~/<모델>_work/runs/args.json`(RL·BC 는 `~/ra_*`·`config.json`) — **완료** `57bd99d` |
| `training/RL/map/README.md:417-418` | "5.2 막혀 있다, FK R1 Pro 전용"·"R1 시뮬 기록" — 끝남/LIMO 기본 — **완료** `57bd99d` |
| `training/RL/map_cmp/README.md:73,135` | 벽 56 "근사판에 없음"(`1bbd634` 에 있음), 상자 충돌 없음(A2 에 있음) — **완료** `57bd99d` |
| `training/RL/map_calib/limo/README.md:106` | "고친 카메라 판이 아직 없어" — 18:07 판 있음 |
| `training/RL/ppo/README.md:127,385` | "C0 는 아직 없다"(G4 에 있음), "다음 1. G5"(끝남) — **완료** `57bd99d` |
| `training/BC/README.md:235,238,328,333-334` | "영상 2 장 TODO"·"flow 아님"·"영상 학생 없음" — `b27aafd`·`3b09389` 로 끝남(G6 에이전트가 이 파일 편집 중) |
| `docs/ai-agent/README.md:63` | VLA(π0.5) — git 무시 파일, 수업 저장소 쪽에서 고칠 것 |

---

## 남은 일 우선순위

사용자 우선순위: ① 커리큘럼을 BEHAVIOR 2026 에 맞추기 ② 학습 전 환경 개선 ③ 학습 뷰어.
"진행 중"(다른 에이전트) 항목은 따로 적었다.

**10-05 추가 (목표 칸 — 물체 id 와 지도 점 하나로, 앱 지도 두드리기)**
- 완료: 목표 칸 2 × 16(VLA_INPUT 2.1) — GPU 지도 토큰 v3·관측 X0 464·BC/RecallVLA 목표 토큰 48·실제 scenemap `sm_tok.h`·실행기 `move_robot` `goal{pick,place:{id|point}}`·`mr_vla_goal_entries`(5b1f40d), 점 판 B1/B3 점으로 가기·B2/B3 놓을 곳 = 점(POLICY 4.6, CURRICULUM 3.2), 지시문 pnp_v1 v2(1,428 행). 검증·해시 `training/RL/ppo/README.md` "목표 칸".
- 남음: B5 점에 놓기(잡기 물리 E6 뒤 — `pred_at_point` 정의만), 앱 위에서 본 지도 화면(plan.md 5절, 설계만), 실행기 id 풀기를 실시간 기억 풀이로(`goal::resolve` 한 곳), 점 판 학습 효과는 긴 학습 뒤에만 잴 수 있음.

**10-05 추가 (지도 인지·검색)**
- 진행 중: **A′** — FastSAM-s + SigLIP 2 후처리(벽·천장 기하 제거, 이름 없는 확률 DA, vMF 벡터·베이지안 이름, 위치 칼만) + scenemap 벽 축 정렬 버그(기운 SLAM 지도에서 정책 쪽 벽 0 개). 목표: A 의 재현율 + B(YOLO26s) 수준 깔끔함. 결과 뷰어 :8084 예정.
- 진행 중: **에이전트 물체 검색 도구** `search_objects`·`confirm_object` + 공용 물체 색인 + SigLIP 2 글 인코더 엔진.
- 다음: A′ 통과 뒤 시뮬 sgrt·LIMO 지도 시험을 FastSAM + SigLIP 2 + A′ 로 옮기기(지금은 보관 엔진 `~/ovdet_models/archive` 사용), GPU 지도 근사판을 A′ 규칙에 맞추기, RecallVLA 자체 검색(질의 벡터 → 상위 K 칸, 검색 InfoNCE, 힌트 지우기·틀린 이름 섞기) 스펙·구현.
- 기록: 검출기 비교 `~/datasets/sim_detcmp/README.md`(A FastSAM / B YOLO26s / C YOLOE, 영상 포함), 실제 bag 결과 MAP_STATE_PLAN 7 절.

**진행 중(손대지 않음)**: BEHAVIOR 2026 커리큘럼 설계(CURRICULUM_BEHAVIOR2026.md), 학습 뷰어(TRAIN_VIEWER V0–V5), VLA 실행기 접점(POLICY 1.3·7.2), LIMO SLAM 등록, MAP_STATE_PLAN 벤치마크 어댑터, G6 FP8 인코더.

**① 커리큘럼 (BEHAVIOR 2026 과 맞추기)** — 설계 문서가 나오면 바로
1. A3: BEHAVIOR 2026 집 배치(방·가구)를 env 커널 장면으로 가져오기 + slam 잡음 — env
2. OMX-F 작업 공간 뽑기(URDF 한계 → 손끝 닿는 높이·거리). 집기 과제 범위가 여기에 달림 — robot
3. 집기(pick) 환경: 팔 행동·물체 접촉·성공 판정(POLICY 3.1, 에이전트_설계 1.7 임계값) — env
4. 교사 특권 (나)·비대칭 가치 (다) — pick 부터 — policy
5. 지표 SPL·Soft SPL·끝 자세 오차를 평가에 — policy

**② 환경 개선 (학습 전)**
6. 경로 거리를 참 장면이 아니라 정책 지도로(`env.h:255`), C1 격자 공개를 실제 탐사 경로로(`map.h:185`) — env/map
7. 렌더·물리 무작위화(조명·노출 고정 `bc_render.cu:90-99`, 팀 기본 설정 섞기) — env
8. env 가정 값(가속 한도·서보 이득·XL330) 실측/사양으로 — env
9. rec.bin 켠 LIMO 탐색 판으로 거리별 놓침·검출 잡음 다시 맞춤, 180748 판 자세 비교 — map
10. A2 씨앗 2 충돌 0.055 → ≤ 0.05 — policy

**③ 학습 뷰어 쪽(뷰어 본체는 진행 중)**
11. 형식이 정해지면 PPO·BC 드라이버에 run.json·progress·episodes·`.trp` 쓰기 — policy

**그 다음**
12. 물체 칸 이름 뜻·생김새 벡터(embed 표) + 외웠는지 평가 — policy
13. known_bugs 1·2·8·9·10·11·12 (작은 고침) — map/robot
14. 위 "문서 정리 필요" 표 — docs
15. 에이전트 P0·P1(`ragent`, `skillspec`, 상태 기계, `verify.rs`) — robot
16. G7 큰 VLA, 학생 RL 미세조정, 데이터 섞기 — policy
17. 실기(ROS 2·캘리브레이션·bag·Jetson)·앱 — robot

**사용자 결정이 필요한 것**: CUDA 12.8 유지, VLA 돌릴 곳, POLICY 9 나머지(팔 행동 표현·몸통·GRU·깊이·문장 바꾸기·학생 RL), 에이전트 3.3 도구 모델 유지 여부, Comet 재학습(P3) 버릴지, 실기 SLAM(Cartographer 대 slam2d), scene.json 2 층 확정(known_bugs 7), DA 생김새 코사인, limo 라이선스, 앱 WS·VPN, 에이전트 plan 8 열린 질문.
