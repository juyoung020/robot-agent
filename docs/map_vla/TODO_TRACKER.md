# 남은 일 추적표 (설계·계획 문서 전체)

2026-10-04 처음 씀, **2026-10-06 다시 맞춤**(낡은 줄 정리). 문서 글이 아니라 **코드와 git log** 로 상태를 정했다.
기준 커밋: robot-agent `d93ede7`, 서브모듈 `src/behavior-2026` `34eb745`.
작업 트리의 커밋 안 된 변경(BC/DAgger 학생: `training/BC/*`, `training/RL/{env,map}/*`)은 "진행 중" 으로만 본다.

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
| G6 FP8 앞 → dgrad → wgrad, 9.1 BF16 대 FP8 학습 비교 | 완료 | GEMM `708db02`; `339bfb4` — SigLIP 2 인코더 FP8 은 코사인 기준 미달(0.952) → FP16 누산(k64) 0.99947 로 바꿈(인코더 0.201 → 0.159 ms/장), 학생·교사 FP8 길은 넣었으나 느려서 기본 끔(GPU_TRAINING 12절) | — | policy |
| G7 BC 큰 VLA (얼린 Qwen FP8) | 대체됨 → 일부 | 얼린 Qwen 대신 **RecallVLA**(Qwen3.5-0.8B 전부 학습 + SigLIP 2, MAPVLA_SPEC) 학습기 `training/vla` M1–M5: `c92898a`, `c494e46`, `11a085e`, 융합 `bac8c5d`…`10d8c0b`, 잰 값 `699bf6a`; 기억 요약 인코더 `e57acd7` | 실제 학습은 아직 — 입력 짜기(TRAINING_DESIGN W10)·인지 다지기 뒤 | policy |

### 본문 다른 항목

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 4.2 팀 학습기 호스트 동기 없애기 | 대체됨 | 팀 `pi05_train` 대신 자체 학습기(G3·G5)를 동기 0 으로 만듦 | — | policy |
| 4.5 렌더 / 4.6 BC 데이터 로더(NVDEC) | 대체됨 | 팀 `RenderBatch` 그래프 잡기로 학습 때 다시 렌더(`b27aafd`, `44b409c`). 저장 영상·NVDEC 안 씀 | — | env |
| 5.3 지도 토큰 시계열 기록 | 완료 | BC 기록이 스텝마다 본 지도 토큰 저장 — `31dc3f5` | — | map |
| 5.4 교사 특권 (나)·(다), 비대칭 가치 | 일부 | 특권 (나): 대본 특권 교사(E6 `564a19a`, 다시 씀 `3071753`, 상태 없는 교사 `a17bfee`). PPO 가치 머리는 아직 가린 입력만 봄(`training/RL/ppo/README.md` G4 절) | PPO 비대칭 가치 (다) | policy |
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
| 1.3 VLA 실행기 접점(objects·끝 신호·결과 형식·안전 거르개·LIMO 부분) | 완료 | `move_robot` 리모 + OMX-F VLA 실행기 `d4e8bd7`(`limo.rs`·`vla.rs`·`verify.rs`, C ABI `mr_vla_*`·`mr_filter`, 시험 19), 시뮬 쪽 서브모듈 `6377253`, 목표 칸 `goal{pick,place}` `5b1f40d` — POLICY 1.4 | 10-04 결정한 "바뀔 접점"(전체 목표 + objects)은 상태 기계와 함께 | robot |
| 3.1 스킬: approach | 완료(A0–A2) / A3–A5 대체됨 | A0–A2 위 G3·G4. A3–A5 는 BEHAVIOR 집 B1–B3 으로(E1 `e54ab3c`, E2 `de08944`, `8c0b42a`) | — | policy |
| 3.1 스킬: pick / place / drawer / door / press | 일부 | 집기·놓기·가져오기: E6 잡기 물리 B4·B5·B6 + 대본 특권 교사(`564a19a`, `3071753`, 빠르게 `e5aea37`, 상태 없는 교사 `a17bfee`). 서랍·문·스위치는 안 함(범위 밖 — CURRICULUM_BEHAVIOR2026 범위는 집기·놓기) | 학생 학습은 BC/DAgger 절(다른 에이전트) | env |
| 3.3 보상 기본 구조 | 일부 | approach 보상 `bc23d06`, `b24a76f`. B4–B6 판정·보상(POLICY 4.8, `564a19a`), PPO 설정 `ppo_pnp.json`(`c4fb346`) | PPO 로 집기 학습 결과는 아직 | policy |
| 3.4 무작위화 | 일부 | 지도 잡음·map_drift `c6b5cc5`, A2 가구 배치. 물리·렌더 무작위화 없음(`bc_render.cu` 조명 고정) | 환경 개선 때 렌더·물리 무작위화 | env |
| 4.2 관측 세 단계 (가)(나)(다) | 일부 | (가) PPO 교사 X0(10-05 목표 칸·탑뷰 격자까지, POLICY 4.2·4.7). 특권 (나)는 대본 교사만(`3071753`, `a17bfee`). 비대칭 (다) 없음 | PPO 교사 쪽 (나)·(다) | policy |
| 4.3 신경망 (칸 MLP + 셀프 어텐션) | 일부 | 칸 MLP + 풀링만(`training/RL/network/README.md:10`), 어텐션·GRU 없음 | 필요할 때 | policy |
| 4.5 커리큘럼 1–4 단계 | 대체됨 | GPU_TRAINING C0–C2 로 구현(`ec4e36d`). 이름 통일은 BEHAVIOR 2026 커리큘럼 문서에서 | — | docs |
| 6 학습 순서 0·1·2a·2b | 완료(approach 기준) | G1–G5 커밋들 | — | policy |
| 6 2c 섞기(BEHAVIOR R1 시연·다른 팔·VQA) | 안 함 | — | 커리큘럼 정한 뒤 | policy |
| 6 3 학생 RL 미세조정(잔차 PPO) | 안 함 | — | — | policy |
| 6 4 실제 로봇 미세조정 | 안 함 | 실기 없음 | 리모 받은 뒤 | robot |
| 7.1 `mr_filter` 행동 거르기 C ABI | 완료 | `src/agent/tools/move_robot/src/ffi.rs` `mr_filter` — `d4e8bd7` | — | robot |
| 7.2 끝·돌려주기 조건 | 완료 | 결과 done/failed/timeout/handback·자동 확인·예산 — `d4e8bd7` (POLICY 1.4) | — | robot |
| 8 평가(처음 보는 이름·집·gt 대 slam·문장 반응) | 일부 | 완성도별 평가만(`e73f873`). 집·이름·문장 반응 없음 | BEHAVIOR 집에서 학습한 뒤 | policy |
| 9 OMX-F 작업 공간 뽑기 | 완료 | `omx_workspace.h`·생성기 `training/RL/tools/omx_ws` — `7f6b529`; 잡는 점 기준 `omx_workspace_grasp.h` — `de08944` (VLA_INPUT 3·8절) | — | robot |
| 9 팔 행동 표현 / 바퀴 차동·메카넘 / 몸통 A·B / 특권 / GRU / 깊이 / 문장 바꾸기 / 렌더 / 학생 RL / 넘김 거리 / 돌릴 곳 | 결정 필요 | 바퀴는 env 가 차동으로 이미 구현, 렌더는 팀 RenderBatch 로 이미 씀 — 이 둘은 사실상 정해짐 | 사용자: 나머지 확정 | policy |

---

## 3. VLA_INPUT.md (8절 "아직 정할 것" + 본문)

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 어디서 돌릴지(리모 위 작은 몸통 / PC Qwen) | 결정 필요 | — | 사용자: 리모 판(기본형·프로) 확정 뒤 | policy |
| SigLIP 2 패치 토큰 내는 인코더 | 완료 | `training/BC/src/vit.cu` — `b27aafd` (코사인 0.9996) | — | policy |
| 리모 바퀴 모드 | 결정 필요(사실상 차동) | env 차동 구현 | 문서에 확정 | docs |
| OMX-F 작업 공간 | 완료 | 위 POLICY 9 와 같음(`7f6b529`) | — | robot |
| 지금 보이는 물체의 즉시 3D C ABI | 일부 | `sm_last_assoc`·`sm_last_views`(id·품질만, `scenemap.h:192-195`). 지도 넣기 전 3D 값 내는 함수 없음 | 필요해지면 `sm_last_dets3d` 같은 함수 | map |
| 3절 물체 칸 이름 뜻·생김새 벡터 | 완료(학습 쪽) | RL 관측·신경망 v2: 칸 = 숫자 33 + 이름 뜻 128 + 생김새 128(얼린 표) — `568f12c`, 칸 줄 버그 고침 `72c146d`. 실행기 `move_robot` `build_slots` 는 아직 256 칸 없음(POLICY 1.4) | 실행기 쪽 칸 채우기 | policy |
| 7절 외웠는지 평가 | 일부 | 처음 보는 이름 평가 흔들기(`568f12c`), 처음 보는 지시문 heldout(pnp_v1, `be92fcd`) | 긴 학습 뒤 처음 보는 집·이름 평가 | policy |

---

## 4. TRAIN_VIEWER.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| V0–V5 전체(`trainfmt`, 서버, 성공 표, 재생, SSE·비교, 평가 표) | 완료 | `training/viewer/`(trainview) — `343492d`, `ad26ccf`, `bda5f43`, `e44180b`(재생 = sgview 화면), `37f5a0b`(GPU 판도 진짜 scenemap), `8896574`(체크포인트마다 자동 기록·학습 상태 불), `9d0dfbb`(정리); README 그림 `3cc9ea3` | — | viewer |
| 12절 정할 것 13 개 | 완료 | 구현에서 정해짐 — TRAIN_VIEWER 12절에 정한 값 | std `TcpListener` 대기열 값만 안 잼 | viewer |
| 학습기 쪽 쓰기(run.json·progress·evals·재생 판) | 완료(본 줄기 판 기록 빼고) | `ppo_run`·`bc_run` 의 `runfolder.rs`(`training/RL/ppo/driver/src/`, `training/BC/driver/src/`) — `343492d`, 자동 재생 기록 `8896574`. 옛 `config.json`·`log.csv`·`results.json` 도 그대로 씀 | 학습 중 판마다 `episodes.jsonl`(장치 기록 링 변경 필요), PPO 에 없는 키(`train/explained_variance`·`reward/<스킬>/<항>` 등) | policy |

---

## 5. CURRICULUM_APPROACH.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| A0, A1 | 완료 | `dbc0b3b`, `b24a76f` (결정적 0.992) | — | policy |
| A2 (가구, ≥ 85 %·충돌 ≤ 5 %) | 완료(씨앗 1) | `bc23d06`, `c96c367` (0.946, 충돌 0.032), `21c7d26` (0.023). 씨앗 2 는 충돌 0.055 | 씨앗 2 충돌 줄이기(가구 가까움 벌) | policy |
| A3 BEHAVIOR 집 한 방 + slam 잡음 | 대체됨 | BEHAVIOR 2026 집을 환경에 넣음(E1 변환기 `e54ab3c`, E2 `de08944`, PPO `8c0b42a`) → 단계 B1–B3(CURRICULUM_BEHAVIOR2026) | — | env |
| A4 집 전체·문 지나기, A5 오래된 지도 | 대체됨 | B1–B3(집 전체), 기억 능력 판 C 낡은 기억(CURRICULUM_BEHAVIOR2026 3.3, `89dd0d1` 설계) | 판 종류는 TRAINING_DESIGN W5·W6 | env |
| 5절 지표 SR·SPL·Soft SPL·끝 자세 오차 | 일부 | 성공·충돌·시간 초과만. SPL 없음 | 평가에 최단 경로 길이 기록 → SPL | policy |
| 6절 다음 커리큘럼(집기·들고 이동·놓기·잇기) | 일부 | B4 집기·B5 놓기·B6 가져오기 환경·교사(`564a19a`, `3071753`) | 학생 학습(BC/DAgger 절) | policy |
| BEHAVIOR 2026 과 맞춘 커리큘럼(CURRICULUM_BEHAVIOR2026.md) | 완료 | `23149c6`(처음), 범위 확정 = 리모 집기·놓기만 `2500ce3`, 이후 E 단계 상태는 그 문서 5절 | — | docs |

---

## 6. MAP_STATE_PLAN.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1 벤치마크 raw 3 개 받기·validate | 완료 | 지도 벤치마크 1–4단계 `6790eda` | — | map |
| 2 어댑터: 시퀀스 → scenemap C ABI | 완료 | 서브모듈 어댑터(behavior-2026 `81f50b7`, 올림 `4a47dce`) | — | map |
| 3 실제 검출(ObjectSAM + SigLIP 2) | 완료(검출기 정함) / 수치는 옛것 | 처음 잰 것은 FastSAM-s + SigLIP 2(`6790eda`). 10-05 기본 = ObjectSAM + SigLIP 2 + objprob(`dom_bench_det`·`realbag_run`·libsgrt 기본, behavior-2026 `26cbdc4`, 올림 `ac1b671`) | ObjectSAM 기본으로 벤치마크 수치 다시 재기(MAP_STATE_PLAN 표는 FastSAM 때 값) | map |
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
| M4 물체 접촉 | 완료(잡기 모형 수준) | A2 상자 막힘 `bc23d06`. 팔-물체: E6 GPU 잡기 모형(그리퍼 폭·손가락 면·가반 하중·놓기 내려앉기·팔/손/든 물체 충돌 막기, CPU == GPU 비트 동일) `564a19a`, E7 대조 `3732b77` | OmniGibson 시험 틀 고쳐 E7 다시(CURRICULUM_BEHAVIOR2026 5.5) | env |
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
| 6절 ③ 우리 VLA 학습기 | 완료 | 작은 학생 G3·G5, RecallVLA 학습기 `training/vla`(위 G7 줄) | — | policy |
| 5절 과제 범위(64/100 지원)·보조 잡기·성공 판정 | 대체됨 | 우리 범위 = 리모 집기·놓기만(CURRICULUM_BEHAVIOR2026 `2500ce3`). BEHAVIOR 집·과제를 환경에 씀(E1 `e54ab3c`, 집기·놓기 판 고르기 `be92fcd`), 잡기·성공 판정은 우리 GPU 모형(E6 `564a19a`, POLICY 4.8) | — | env |

## 10. VLA_QnA.md · PI05_TRAINING.md · untitled.md

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| VLA_QnA 요약 1–6 (SmolVLA/openpi 로 시작, 텍스트 지도) | 대체됨 | 자체 RL 교사 → BC 학생, 연속값 지도 토큰으로 감 | 기록용 | docs |
| PI05_TRAINING 6절: HL 단계 글·VI 다른 말 | 일부 | 지시문 표 pnp_v1(조합 × 문장 12, 영·한·heldout·이름 흔들기 — `be92fcd`, 점 목표 더해 1,428 행 `1183a4d`). 단계 문장 출력은 RecallVLA 설계(MAPVLA_SPEC, TRAINING_DESIGN W4) | 단계 국면 열·라벨(W4) | policy |
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
| 3.3 LLM 도구 8 개(find_object·set_plan·check·ask_user·report·remember) | 일부 / 결정 필요 | 있음: `move_robot`, 물체 찾기 `search_objects`·`confirm_object`(`369cd3a`, 실시간 기억 `936c256`)·`list_place`. 없음: `describe_object`·`set_plan`·`check`·`ask_user`·`report` | 사용자: 도구 수(지금 계획 10 개 > 8) 합치는 안(plan.md 3.3 메모) | robot |
| 3.4 상태 기계·복구 표, `verify.rs` | 안 함 | `machine.rs`·`verify.rs` 없음(move_robot 안 재계획·후진만) | 실행기 접점과 함께 | robot |
| 3.5 방 이름 | 일부 | scenemap `rooms.cpp` 규칙표·`sm_set_room_name`. 투표·제로샷 없음 | SigLIP 이름이 방 규칙에 들어가는지 확인 | map |
| 5 평가 `ragent eval --gate`·골든 세트, 5.5 `gate.rs` | 안 함 | — | P1 뒤 | robot |
| decision_log(decisions.jsonl) | 완료 | `src/agent/runtime/src/lib.rs`(10-06 에 `skills/explore/src/lib.rs` 에서 옮김), `src/agent/devtools` `decisions-agg` — `4ad1781`, `f82dc18` | `label.vla_better` 는 VLA 실행기 생긴 뒤 | robot |
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
| 2 물체 인식 ObjectSAM + SigLIP 2 + objprob | 완료(PC) | SigLIP 2 붙음(`sgrt_clip`, 서브모듈 `61eb0a2`, `976e217`, `8c1a7d6`). 10-05 기본 검출 = ObjectSAM(YOLO26n 학생, `training/fastsam` `6b481de`) + objprob — libsgrt·realbag_run 기본(behavior-2026 `26cbdc4`, 올림 `ac1b671`). YOLOE 는 보관 | 이름 정답률·R@1 재측정, Jetson 기기 위 시간 | map |
| 2 xyz·DA·지도 갱신·벡터 찾기·Spark-DSG 저장 | 완료 | scenemap `objmap.cpp`, `da/`, `sgrt_query_*`, `dsg_save.cpp`, sgview `c4548fe` | known_bugs 7·10·11 | map |
| 2 방 나누기·이름(제로샷) | 일부 | 규칙표만, 제로샷 없음 | — | map |
| 3 Qwen API | 완료 | planner `llm.rs` | — | robot |
| 3 상황 풀기·다시 계획·RAG 답·평가 | 안 함 | agent P0/P1 과 같음 | — | robot |
| 4 π0.5 LoRA·리모용 π0.5 | 대체됨 | π0.5 버림 | — | policy |
| 4 우리 VLA 설계·학습 | 일부 | 작은 학생 G5 완료, RecallVLA 사양(MAPVLA_SPEC)·학습 설계(TRAINING_DESIGN `57e0d96`)·학습기 M1–M5(`training/vla`) — 큰 학습은 아직. `src/vla` 는 README 만(로봇 실행 코드 없음) | 환경 일 W1–W10 → RecallVLA 학습 | policy |
| 4 시뮬에 리모 + 팔 | 완료 | `src/robot/og` — `efcb433`, `de769fa` | — | robot |
| 4 리모에서 실행(Jetson/NPU) | 안 함 | — | 돌릴 곳 결정 뒤 | robot |
| 5 앱(iOS·Android·채팅·WS) | 안 함 | `src/app` README 만 | — | robot |
| 6 합치기: BEHAVIOR 전체 흐름 | 일부 | 탐사만 | — | robot |
| 6 합치기: 리모 시뮬·실기·점수 | 일부 / 안 함 | 시뮬 LIMO 탐사 `50c971c` | — | robot |

## 13. docs/known_bugs.md

대부분 고쳤다(`docs/known_bugs.md` 의 "고침" 칸). 남은 것은 3·7 뿐.

| # | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| 1 ovdet_eval 옛 경로 | 완료 | behavior-2026 `057689b` | — | map |
| 2 explore 주석 0.28 m | 완료 | `f139dfe` | — | robot |
| 3 `/mnt/c/behavior-2026` 박힘(84 파일) | 보류 | 일괄 고침이라 작은 고침 아님 | 스크립트 위치 기준 뿌리 | robot |
| 4–6 `/mnt/c` 박힘 셋 | 완료 | behavior-2026 `0c392f7` | — | robot |
| 7 scene.json 3 층 대 2 층 결정 | 결정 필요 | `test_scene_json.cpp:81-82`, `scenemap.h:362-367`, `dsg_save.cpp:563` | 사용자: 2 층 확정이면 저장·시험·헤더 고침 | map |
| 8–12 주석·메시지, DA `max_ext`, 이벤트 7 이름, walls 시험 변수 | 완료 | behavior-2026 `057689b` | — | map |

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
| RL/network SigLIP 이름·생김새 표 | 완료 | 얼린 이름·생김새 표 128 + 128(`568f12c`, 묶음 커널 `49b2149`) | — | policy |
| RL/reward 탐색 보상 대조 | 안 함 | — | 선택(교사 탐사 보상은 TRAINING_DESIGN W9) | policy |
| BC 남은 일 1 FP8 인코더 | 완료(FP16 누산으로) | `339bfb4` (FP8 은 기준 미달) | — | policy |
| BC 남은 일 1 EMA·학습률 일정 | 안 함 | `BC/src` 에 ema 없음 | — | policy |
| BC 남은 일 2 렌더 무작위화·팀 기본 설정 섞기·로봇 그리기 | 안 함 | `bc_render.cu:90-99` 조명 고정 | **환경 개선**(학습 전) | env |
| BC 남은 일 3 작은 트랜스포머·처음 보는 이름 평가 | 안 함 | — | — | policy |
| embed `sb32_pe_sim` 판 | 일부 | `~/embed_work/runs/sb32_pe_sim/` 에 args.json 만 | 다시 돌리거나 지움 | map |
| embed 여러 집 시뮬 자르기·평가 B·한국어 평가·KorLex·Nano 측정 | 안 함 | `8425e32` 뒤 커밋 없음 | — | map |

## 15. README·src README·docs 기타

| 항목 | 상태 | 증거 | 다음 할 일 | 영역 |
|---|---|---|---|---|
| `src/vla` 폴더 | 일부 | README 만(로봇 실행 코드 없음). README 가 학습 코드 `training/BC`(작은 학생)·`training/vla`(RecallVLA)를 가리킴 | 로봇 쪽 실행 코드는 돌릴 곳 결정 뒤 | docs |
| `src/app` ios·android | 안 함 | README 만 | — | robot |
| `src/robot` limo 라이선스 TODO | 결정 필요 | upstream `package.xml` | 사용자: 재배포 여부 | robot |
| model_selection DA 생김새 코사인 | 결정 필요 | `da/` 에 임베딩 비교 없음 | — | map |
| clip_candidates 8.6 Nano TRT 측정 | 안 함 | — | 실기 | robot |
| clip_candidates 8.6-5 한국어 학생 → `sgrt_query_embedding` | 일부 | API 있음(`sgrt.h:155`), 연결 안 됨 | 연결 | map |
| model_selection 앱 WS·VPN | 결정 필요 | — | — | robot |

---

## 문서 정리 필요 (내용이 낡았거나 틀린 문서)

남은 것만 적는다. 고친 것: `57bd99d`(π0.5·Cartographer·OMX-F·옛 경로 등 15 곳), 10-06 정리 커밋(아래 "고친 것").

| 문서:줄 | 무엇이 틀렸나 |
|---|---|
| `training/BC/README.md:3,355,463` | "Qwen 몸통 큰 모델(G7)은 아직 없고"(RecallVLA `training/vla` 있음), 표 "영상 RGB 2 장 아직 없음(TODO)"(`b27aafd`·`3b09389` 로 끝남), "G6: FP8 …, G7: 얼린 Qwen FP8"(G6 끝 `339bfb4`, G7 → RecallVLA) — BC 학생 작업 중인 에이전트가 이 파일을 편집 중이라 손대지 않음 |
| `docs/known_bugs.md` 3 | 파일 수 약 80 → 84(세는 기준이 달라 다시 셀 것) |
| `docs/map_vla/MAPVLA_SPEC.md` "자체 기억 검색" 절 | "설계, 구현 전" — 기억 요약 인코더(`e57acd7`)가 물체 InfoNCE·정밀 칸 고르기를 넣었다. `<recall>` 질의 방식 전체가 이걸로 끝났는지는 확인 못 함 |
| `docs/map_vla/POLICY.md` 1.3 "바뀔 접점" | "아직 구현 안 함" — `goal{pick,place}`(`5b1f40d`)만 들어감, 전체 목표 문장 + objects 호출은 아직인지 확인 필요 |
| `docs/ai-agent/README.md:63` | VLA(π0.5) — git 무시 파일, 수업 저장소 쪽에서 고칠 것 |

고친 것(10-06): `docs/map_vla/README.md`(SIM_PORTING "코드는 아직 없음", 빠진 문서 행), `SIM_PORTING.md` 4·5절, `POLICY.md` 머리·1.1, `TRAIN_VIEWER.md` 12절, `GPU_TRAINING.md`·`CURRICULUM_APPROACH.md`·`MAPVLA_SPEC.md` 머리, `BEHAVIOR_ENGINE_REVIEW.md` 학습기 행, `training/RL/map_calib/limo/README.md:140`, `training/RL/network/README.md` 칸 입력, `README.md`·`training/README.md`·`src/vla/README.md`(RecallVLA), `docs/model_selection.md`·`docs/plan.md`·`src/agent/skills/explore/README.md`(YOLOE → ObjectSAM), `src/agent/tools/README.md`(`set_plan` π0.5).

---

## 남은 일 우선순위

처음(10-04) 사용자 우선순위였던 ① 커리큘럼을 BEHAVIOR 2026 에 맞추기 ② 학습 전 환경 개선 ③ 학습 뷰어 중 ①·③ 은 끝났다. 아래 "지금 상태" 가 10-06 기준이고, 그 밑 날짜별 "추가" 는 기록이다.

**지금 상태 (10-06)**
- 끝남·정함: 검출 = ObjectSAM + SigLIP 2 + objprob(`ac1b671`, behavior-2026 `26cbdc4`). 대본 교사 빠르게(`e5aea37`, N 4,096 B4 2.7·B5 5.2·B6 2.5 ms/스텝) + 상태 없는 교사(`teacher_sl.h`, `a17bfee`). RecallVLA 신경망 융합·다시 계산 줄이기(M5 `699bf6a`, 마지막 융합 `10d8c0b`) + 기억 요약 인코더(`e57acd7`). 커리큘럼 설계(CURRICULUM_BEHAVIOR2026 — 리모 집기·놓기 B0–B6, 환경 E0–E7). 학습 뷰어 trainview(V0–V5, `training/viewer`). VLA 실행기 접점(`d4e8bd7`). 지도 벤치마크 어댑터(`6790eda`). G6(`339bfb4`).
- 안 됨: BC/DAgger 학생이 B4 를 아직 못 배움(0.00–0.04, CURRICULUM_BEHAVIOR2026 5.6.1) — 다른 에이전트가 작업 중.

**진행 중(다른 에이전트, 손대지 않음)**: BC/DAgger 학생(`training/BC`·`training/RL`, CURRICULUM_BEHAVIOR2026 5.6·POLICY 4.8), README 영상·설명 다시 찍기. 예전 목록의 "LIMO SLAM 등록" 은 10-06 에 확인 못 함.

**① 학생이 B4 를 배우게** — 다른 에이전트(아래 10-06 "남음" 줄).

**② RecallVLA 학습 전 환경 일 — TRAINING_DESIGN 6절 W1–W10 (10-06: 아무것도 시작 안 함)**
1. W1 판마다 장치 기억 표(N ≤ 256, 살펴본 정도 3) — map
2. W2 방 항목 ≤ 8·방향 구역 8 커널, W3 물체 줄 방 종류 — map
3. W4 단계 국면 열 + 라벨(`rec_tgt`, 단계 문장) — env
4. W5 두 부분 판 B·C, W6 목표 없음·가짜·확인 실패 판 — env
5. W7 실제 쪽 `sm_tok.h` 짝(scenemap 살펴본 정도 필드는 서브모듈에 들어감 — `852ee4c`) — map
6. W8 절차 생성 집(ProcTHOR-10K 가져오기) — env
7. W9 교사 X0 에 탐사 보상·방향 구역·방 항목 — policy
8. W10 BC 기록에 기억 표 열쇠 + 라벨, RecallVLA 입력 짜기 — policy

**③ 인지 다지기**(아래 표): ObjectSAM 기본으로 벤치마크·radio r3 다시 재기, Jetson 기기 위 시간, 실제 이름 정답률, SLAM 떠밀림, 학습용 지도 근사를 새 scenemap 에 다시 맞춤 — map

**④ 환경 개선 (남은 것)**
- 경로 거리를 참 장면이 아니라 정책 지도로, C1 격자 공개를 실제 탐사 경로로 — env/map
- 렌더·물리 무작위화(조명·노출 고정, 팀 기본 설정 섞기) — env
- env 가정 값(가속 한도·서보 이득·XL330) 실측/사양으로 — env
- rec.bin 켠 LIMO 탐색 판으로 거리별 놓침·검출 잡음 다시 맞춤, 180748 판 자세 비교 — map
- A2 씨앗 2 충돌 0.055 → ≤ 0.05 — policy

**그 다음**
- RecallVLA 큰 학습(②·③ 뒤), 자체 기억 검색 마무리 — policy
- PPO 교사 비대칭 가치 (다), 지표 SPL·Soft SPL·끝 자세 오차 — policy
- 학습 뷰어 남은 일: 학습 중 판마다 기록(장치 기록 링), PPO 에 없는 키 — viewer
- known_bugs 3(84 파일 일괄) — robot
- 위 "문서 정리 필요" 표 — docs
- 에이전트 P0·P1(`ragent`, `skillspec`, 상태 기계, `set_plan`, `verify.rs`) — robot
- 학생 RL 미세조정, 데이터 섞기 — policy
- 실기(ROS 2·캘리브레이션·bag·Jetson)·앱 — robot

**사용자 결정이 필요한 것**: CUDA 12.8 유지, VLA 돌릴 곳, POLICY 9 나머지(팔 행동 표현·몸통·GRU·깊이·문장 바꾸기·학생 RL), 에이전트 3.3 도구 모델 유지 여부, Comet 재학습(P3) 버릴지, 실기 SLAM(Cartographer 대 slam2d), scene.json 2 층 확정(known_bugs 7), DA 생김새 코사인, limo 라이선스, 앱 WS·VPN, 에이전트 plan 8 열린 질문.

### 기록 — 날짜별 추가

**10-06 추가 (교사 빠르게 + 상태 없는 교사 + PPO 집기 바퀴 재기)**
- 완료: 교사 계획 빠르게(정적 점유 표 `tocc`, 계획 = 블록 하나, 찾기 조각 BFS 재사용, 등급 1·2 찾은 조각에서 끝, 들고 다닐 때 팔 검사 캐시·지역 계획 비용 차례, `tch_extract` 이웃 버그) — N 4,096 B4 15.3 → 2.6, B5 27.5 → 5.9, B6 14.4 → 2.3, 섞음 22.9 → 4.3 ms/스텝, 성공 그대로. 상태 없는 교사(`teacher_sl.h`, BC `teacher_script 2`, `--sl-fresh` 로 라벨 = 상태의 함수 확인). PPO 집기 바퀴 nsys 표(`training/RL/ppo/README.md`). 잰 값 CURRICULUM_BEHAVIOR2026 5.6.1.
- 잰 값: 상태 없는 교사 성공 B4 0.920·B5 0.822·B6 0.638, 라벨 일치 0.61–0.71; BC + DAgger B4 학생 0.00–0.04(상태 있는·없는 교사 둘 다, 대조 "교사가 더 모음" 0.02–0.04).
- 남음: B5·섞음 교사 3 ms 넘음(놓기 서는 자리 찾기 조각 수), 학생이 B4 를 못 배움(학생 크기·서는 자리 관측 없음 — 짝마다 서는 자리 후보 여럿·가까운 것 고르기, arch 1 학생, DAgger β 섞기), 상태 없는 교사의 막힘 넘기(다시 하기 기억 없음), 면 4–6 cm 옆 잡기(D, 이번에 못 함), PPO 바퀴 지도 keyframe 0.73 ms/스텝·잡기 판 환경 1.27 ms/스텝.

**10-05 추가 (E6 대본 교사 다시 씀 + 잡기 가능 표)**
- 완료: 잡기 가능 표(`pnp_feasibility`, 장치 6.7 s) + `PF_FEAS` 고르기(PPO `beh.feas`·BC `b_feas`), 교사 = 계획(서는 자리 등급 찾기·잡기/놓기 웨이포인트·특권 점유 BFS 길)은 요청한 판만 워프로·나머지 스텝은 따르기, 지역 계획·다시 하기·B6 탐사(POLICY 4.2), 환경 고침 셋(받침에 앉혀 시작·닫힘 조건 ④·B5 시작 yaw). CPU == GPU 비트 동일(교사 버퍼까지) + 음성 대조, 예전 해시 그대로. 잰 값 CURRICULUM_BEHAVIOR2026 5.6.
- 잰 값: 교사 성공(잡기 가능 짝) B4 0.947·B5 0.911·B6 0.767; 잡기 가능 짝 B4 3,502·B5 6,685·B6 1,818 / 17,831; BC 짧은 판(B4) 학생 0.026.
- 남음: 계획 커널 비용(서는 자리 찾기 조각 10–16 ms·들고 다닐 때 행동 4.8 ms), 좁은 곳 나르기(나르는 자세가 0.2 m 앞), 면 4–6 cm 옆 잡기 0.49, 긴 BC/PPO 학습, DAgger 는 기억 없는 라벨 필요, 상자 안·가구 상자 근사로 못 잡는 물체(모형 한계).

**10-05 추가 (E6 잡기 물리 — B4 집기·B5 놓기·B6 가져오기)**
- 완료: GPU 잡기 모형(그리퍼 폭·손가락 면·가반 하중·놓기 내려앉기·팔/손/든 물체 충돌 막기, CPU == GPU 비트 동일 + 음성 대조 둘), 판 B4·B5·B6 + 점 목표 놓기(`pred_at_point`) + 실패 판(미끄러짐·막힌 자리), 대본 특권 교사(장치 커널, BC `teacher_script` 라벨), PPO 지표 6–8·`ppo_pnp.json`, 지도 근사가 움직이는 물체를 봄, 예전 해시 그대로. 잰 값·가정 CURRICULUM_BEHAVIOR2026 5.5, POLICY 4.8.
- 잰 상한: 교사 성공 B4 0.230·B5 0.124·B6 0.018(지도 C0), 잡기 계획이 있는 판 52.5 %(느슨)·79.0 %(엄격). E7: E0 OmniGibson 64 경우 결과 일치 0.750(닿음 0.936).
- 남음: 교사 길 찾기(가구 사이 돌기)·계획 비용, BC 학생 → PPO 시작점, 긴 학습으로 B4/B5 SR 재기, OmniGibson 시험 틀 고쳐 BEHAVIOR 물체 대조 다시, 막는 물체를 지도에, 넘어짐 회전, `nextto`·`under`, 쥠 닿음 관측(VLA_INPUT 8절 제안 — 결정 필요).

**10-05 추가 (위에서 본 지도 = 셋째 RGB 그림)**
- 완료: 하나의 정의 `topview.h`(VLA_INPUT 1.1) — GPU 지도 근사가 스텝마다 그림(교사 격자 MapTok v4 `tv`, 학생 RGB `tv_render`, GPU == CPU 비트 동일 + 음성 대조 둘), 실제 scenemap 변환기(`sm_tok.h sm_topview_rgb`), PPO 교사 X0 976(POLICY 4.7), BC 영상 학생 셋째 그림(시험용 깃발 `topview`), RecallVLA 그림 셋(B 24 29.4 표본/s, 11.1 GB).
- 남음: 얼린 SigLIP 2 가 합성 지도 그림을 쓰는지 BC 로 잼(깃발 켬/끔 비교 학습 — 긴 학습 필요), 앱 화면(설계만), 실제 LIMO scenemap 기록으로 그림 눈 확인, 점유 % 문턱 50 맞춤.

**10-05 추가 (목표 칸 — 물체 id 와 지도 점 하나로, 앱 지도 두드리기)**
- 완료: 목표 칸 2 × 16(VLA_INPUT 2.1) — GPU 지도 토큰 v3·관측 X0 464·BC/RecallVLA 목표 토큰 48·실제 scenemap `sm_tok.h`·실행기 `move_robot` `goal{pick,place:{id|point}}`·`mr_vla_goal_entries`(5b1f40d), 점 판 B1/B3 점으로 가기·B2/B3 놓을 곳 = 점(POLICY 4.6, CURRICULUM 3.2), 지시문 pnp_v1 v2(1,428 행). 검증·해시 `training/RL/ppo/README.md` "목표 칸".
- 남음: B5 점에 놓기(잡기 물리 E6 뒤 — `pred_at_point` 정의만), 앱 위에서 본 지도 화면(plan.md 5절, 설계만), 실행기 id 풀기를 실시간 기억 풀이로(`goal::resolve` 한 곳), 점 판 학습 효과는 긴 학습 뒤에만 잴 수 있음.

**10-05 추가 (지도 인지·검색)**
- 끝남: scenemap 확률 모드(`scenemap/src/objmap.cpp`·`objprob.cpp`, 확률론적 물체 수준 매핑 — [용어](../terms.md)) 1 차 + 벽 방향 수정, 조각 평면 맞춤 PCA → 랜색 교체(behavior-2026 `3b5543f`, 결과 거의 같음), 에이전트 물체 검색 도구 `search_objects`·`confirm_object`·`list_place` + 공용 물체 색인 + SigLIP 2 글 인코더(실시간 지도·좌표·시간 조건).
- 그 뒤 끝남(10-06 확인): 확률 모드 이름 정리(`objprob` — `563c09f`, 서브모듈 올림 `54b3618`), 벽 조각이 door·window·pillar 로 살아남는 문제(문·창 크기 확인 — 서브모듈 올림 `66e7929`), things 전용 분할(ObjectSAM — `6b481de`, `ac1b671`), 목표 칸 통일(`1183a4d`) → 탑뷰 그림(`a6c3a2f`).

**인지 다지기 단계 (RecallVLA 큰 학습 전에 통과, 10-05 결정)** — VLA 는 지도 기억을 입력으로 받으므로 인지·지도가 먼저다. 학습용 GPU 지도 근사(`training/RL/map`)는 실제 scenemap 을 흉내 내므로 scenemap 이 바뀌면 다시 맞춘다.

| 부분 | 위치 | 할 일 | 통과 기준(예시) |
|---|---|---|---|
| 분할 | `training/fastsam/` | 벽·천장·바닥을 자르지 않는 분할. **10-05: decided ObjectSAM (YOLO26n 학생) + SigLIP 2 + objprob** — 기본 엔진 `yolo26n-seg-obj-416`(behavior-2026 `26cbdc4`: libsgrt·realbag_run·시작 스크립트). FastSAM-s 재학습(`FastSAM-s-416-obj`)은 보관(`~/ovdet_models/archive/x86_sm120/`, 버림). 까닭: FastSAM-s 계산의 약 1/10 이라 LIMO 의 Jetson(특히 Nano)에 맞다 — 기기 위 시간은 아직 안 잼 | 물체·문·창·계단·처음 보는 종류 재현율이 기존 이상 |
| 물체 지도 | `scenemap/src/objmap.cpp`·`objprob.cpp` | 중복·잘못 합침·벽 오등록 줄이기 | BEHAVIOR radio r3: 찾음 ≥ 28/34, 중복·벽 오등록이 YOLO26s-seg 수준에 가깝게 |
| 바뀜 판정 | 같은 곳 | DOMB(이동·제거·추가·교환), OpenLORIS office1-6 채점 | DOMB 이동·교환 > 0, 변화 F1 > 0.25 / 안 바뀐 판 가짜 사라짐 최소 |
| SLAM | `scenemap/src/slam2d.cpp` | 고리 닫기·재위치 추정 | 실제 bag 출발 기준 떠밀림 줄이기(지금 4–22 cm) |
| 이름·어휘 | `clip` 라벨 표 | 다시점 투표 효과 측정, 어휘 하나로 | 실제 이름 정답률(지금 약 30 %) 개선 |
| 실행 경로 통일 | sgrt(시뮬 탐사)·LIMO 지도 시험 | ObjectSAM + 확률 모드로 — 10-05 끝(behavior-2026 `26cbdc4`: libsgrt objprob 앞단, `SGRT_ENGINE` 기본 학생) | 같은 결과 |
| 학습용 지도 근사 | `training/RL/map` | 새 scenemap 기준으로 다시 맞춤 | map_cmp 5.2 기준 통과 |

- 다음(그 뒤): RecallVLA 자체 검색(질의 벡터 → 상위 K 칸, 검색 InfoNCE, 힌트 지우기·틀린 이름 섞기) 구현.
- 기록: 검출기 비교 `~/datasets/sim_detcmp/README.md`(FastSAM-s-416 / YOLO26s-seg / YOLOE-11L, 영상 포함), 실제 bag 결과 MAP_STATE_PLAN 7 절.
