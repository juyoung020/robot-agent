# behavior-2026 엔진 검토 — 우리가 거기서 학습할 수 있나

대상: `~/behavior-2026` (= `~/robot-agent/src/behavior-2026`, BEHAVIOR Challenge 2026 작업 공간, 커밋 a1363d7).
방법: 문서와 소스를 읽고, 이 PC(RTX 5070 Ti)에서 GPU 시험을 GPU 잠금 규약대로 직접 돌렸다. 저장소는 수정하지 않았다.

## 1. "맵"이 CUDA인가
- **지도 `scene_graph/scenemap`(SLAM, 물체 기억)은 CUDA가 아니다.** C++ 파일 39개, `.cu` 0개. 키프레임 합 약 0.4 ms(Ryzen 한 코어)로 CPU에서 돈다 (`docs/scenemap_설계.md` 3.6.1). GPU는 검출기·임베딩(TensorRT)과 이미지 자르기(`crop.cu`)뿐이다.
- **CUDA로 포팅된 것은 `sim/engine`(시뮬레이터)이다.** `.cu/.cuh` 28개, 6.5k줄. "맵"이 시뮬레이션 장면을 뜻한다면 이쪽이다.

## 2. 직접 돌려서 확인한 것 (2026-10-03, 이 PC)
| 시험 | 결과 |
|---|---|
| `solver/test_free_bodies_gpu` | 판 1,024 × 강체 1,000 × 600 서브스텝. PhysX = C++ = CUDA 비트 불일치 **0 / 1,024,000**. 층 2(CUDA) 6.33e6 판·서브스텝/초 |
| `render/test_render_synth` | 판 64 × 카메라 3(720², 480², 480²) = 1152 ms → **56 판·프레임/초**. 층 1(CPU) = 층 2(CUDA) 비트 불일치 0 (판 4개 비교) |
- 나머지 GPU 시험(`test_contact_solver_gpu`, `test_art_solver_gpu`, `test_omni_gpu`)은 입력 파일이 필요해서 돌리지 않았다.
- 224² 해상도에서의 렌더 속도는 재지 않았다. 화소 수 비례로 약 10배 빠를 것으로 추정할 뿐이다(추정).
  - **(10-04 잼)** G5 에서 팀 `RenderBatch` 로 A2 장면·카메라 2 대 224² 를 쟀다: 팀 기본 설정 3,000, 싼 설정 21,027 판·프레임/s(`training/BC/README.md` 렌더 절, `44b409c`).

## 3. 이 PC에 있는 것
- 장면 에셋 `BEHAVIOR-1K/datasets` 37 GB, 시뮬레이터 env `behavior`(Isaac Sim 5.1), 데모 4.9 GB(과제 0만), π0.5 체크포인트들(`data/pi05_native` 23 GB).
  - **(10-04 바뀜)** π0.5 체크포인트(`data/pi05_native` 23 GB)는 일부러 지웠다. π0.5 를 쓰지 않기로 해서다. 다시 받지 않는다.
- 엔진 시험 바이너리는 `~/engine-build/tests`에 빌드되어 있다.

## 4. 이 엔진에 이미 있는 것 (문서 기준, `docs/엔진_자체구현.md`)
- GPU 강체·관절체(Featherstone)·접촉(PCM)·조인트·TGS 풀이, 비트 단위로 PhysX 5.6.1과 대조.
- OmniGibson 파이썬 층(제어기, 보조 잡기, 물체 상태, BDDL 판정) 이식.
- **판 N개 × 카메라 3대 RGB/깊이 배치 렌더러**(`RenderBatch`, 호스트 왕복 없음, 깊이는 공식과 중앙 1~2 ulp).
- 즉 이전에 세운 `docs/PORT_PLAN.md`(MuJoCo Warp 전체 포팅)가 만들려던 것 상당 부분이 이미 있다.

## 5. 우리가 학습하기에 막히는 점
| 항목 | 상태 | 근거 |
|---|---|---|
| 로봇이 R1 Pro에 고정 | 엔진 42개 파일, 지도 8개 파일이 R1 Pro를 직접 참조. 제어기는 `R1ProConfig/R1ProState`, 행동 23, 상태 61 | `core/omni/controllers.h:141`, `sim/configs/r1pro_openpi.yaml` |
| 우리 로봇(LIMO + OMX) 없음 | 엔진의 장면은 공식 OmniGibson 실행에서 뽑은 기록(OVD 등)을 정답지로 재생한다. 새 로봇은 먼저 OmniGibson에 올려야 한다 | `docs/엔진_자체구현.md` 2절, 15.3 |
| 폐루프 빈 곳 | 보조 잡기(시연 0)와 성공 판정이 native 루프에 없다. 렌더가 느리다(판 1, 원해상도 116 ms/프레임) | `docs/미해결과제.md` B19 |
| GPU 풀이 고정비 | 커널 한 번 3.2 ms(시험)·9 ms(radio). 판 64에서 CPU보다 느림 | 미해결과제 B16 |
| 과제 범위 | 100과제 중 지원 64, 부분 14, 막힘 22 | `docs/엔진_자체구현.md` 15.2 |
| 학습기 | `vla/pi05_train`은 π0.5용. 우리 VLA(SigLIP2 + Qwen3.5 + 액션 헤드)용 학습기는 없다. **(10-04 바뀜)** 우리 학습기는 `training/` 에 있다: G1 환경 커널, G2 GPU 지도, G3·G4 RL 교사 PPO, G5 BC 작은 VLA(얼린 SigLIP 2 + 지도 토큰 + flow matching 행동, DAgger). 큰 VLA(얼린 Qwen)는 G7 계획 | `src/vla/pi05_train`, `training/RL`, `training/BC`, [GPU_TRAINING.md](GPU_TRAINING.md) 11절 |
| π0.5 학습 메모리 | LoRA도 22.5 GB 초과라 이 PC(16 GB)에서는 불가. 학습은 보류 상태. **(10-04)** π0.5 는 쓰지 않기로 해서 이 행은 기록으로만 남긴다 | `docs/학습환경_가속.md`, 미해결과제 A7 |

## 6. 방향 제안 (결정은 사용자)
1. 새 엔진을 처음부터 포팅하지 말고 **`sim/engine`을 확장**한다. 물리, 렌더, 판 묶음이 이미 검증되어 있다.
2. 확장 순서(제안): ① 우리 URDF(`Map_Vla/src`)를 OmniGibson에 올려 정답지를 얻는다 → ② 엔진에 LIMO + OMX 로봇 설정(제어기, FK, 카메라)을 추가 → ③ 우리 VLA 학습기 작성.
3. 빌드 기준은 CUDA 12.8이다 (`CLAUDE.md`: 13.2는 설치돼 있어도 쓰지 않는다). `PORT_PLAN.md`의 nvcc 13.2 언급은 이 기준과 다르다.
