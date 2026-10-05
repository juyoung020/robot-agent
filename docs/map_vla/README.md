# docs/map_vla — Map_Vla (리모 + 매니퓰레이터 VLA) 설계 문서

Map_Vla 프로젝트에서 쓴 조사·설계 문서다. 작성 시점의 판단이라 일부는 바뀌었다(아래 표의 비고).

| 문서 | 내용 | 비고 |
|---|---|---|
| [TODO_TRACKER.md](TODO_TRACKER.md) | **남은 일 추적표**: 이 폴더 설계 문서·계획의 항목마다 상태(코드·git 기준)·증거 커밋·다음 할 일, 낡은 문서 목록, 우선순위 | 지금 상태는 여기서 본다 |
| [TRAINING_DESIGN.md](TRAINING_DESIGN.md) | **RecallVLA 학습 설계**: 최종 관측(기억 줄·방 항목·방향 구역), 신경망, 손실, 라벨·단계 문장, 교사 → BC/DAgger → S1–S3, 판 종류, 환경 일 W1–W10 | 2026-10-05 설계, W1–W10 은 아직 |
| [MAPVLA_SPEC.md](MAPVLA_SPEC.md) | **RecallVLA 사양**(지도 + VLA 파운데이션 모델: Qwen3.5-0.8B 전부 학습 + SigLIP 2, 단계 문장 + 행동), 결정 기록, 메모리·속도 계산 | 학습기 `training/vla` M1–M5 구현, 큰 학습은 아직 |
| [CURRICULUM_BEHAVIOR2026.md](CURRICULUM_BEHAVIOR2026.md) | **커리큘럼 2: BEHAVIOR 2026 기준** — 100 과제 목록(장면·물체·목표 술어·스킬), q_score 규칙, LIMO + OMX-F 로 되는 과제(q 상한 추정), 스킬 빈도, 새 단계 B0–B7(옛 A0–A5 대응), 학습 환경 차이 분석, 환경 개선 단계 E0–E7(장면·과제 변환기 포함), 열린 질문 | 2026-10-04 설계(범위: 리모 집기·놓기만, B0–B6). 커리큘럼 1 을 이 문서 기준으로 다시 짬. E 단계 상태는 5절 |
| [TRAIN_VIEWER.md](TRAIN_VIEWER.md) | **학습 뷰어 설계(Rust)**: sgview 방식 Rust 서버 + canvas 차트·three.js 재생, 학습·재생·비교 탭, 실행 종류(teacher·bc·dagger·rlft·eval·lab), 학습 고리가 쓰는 파일 규약(run.json·progress·episodes·이진 재생), 전투기 뷰어에서 가져온 규칙 20개, 단계 | 2026-10-03 설계, **구현 끝**(V0–V5, `training/viewer` trainview) |
| [GPU_TRAINING.md](GPU_TRAINING.md) | **GPU 만으로 학습하는 설계**(C++/CUDA/Rust): 5070 Ti(sm_120)에서 되는 FP8 명령, CUDA graph 로 CPU 없는 학습 고리, JSBSim GPU 포팅 방식의 환경 커널, 자라는 지도(GPU scenemap 근사), 커널 합치기, 정밀도, 16 GB 메모리, 검증 사다리, 예상 속도, 단계 | 2026-10-03 설계. G0–G6 구현, G7 은 RecallVLA(MAPVLA_SPEC)로 바뀜 |
| [POLICY.md](POLICY.md) | **정책 설계**: LLM → VLA·`move_robot` 역할 나누기, 교사 RL(스킬·보상·비대칭 actor-critic·자라는 지도·커리큘럼), 학생 VLA(몸통 두 안·행동 묶음·지연), 학습 순서(RL 교사 → BC + DAgger → VLA RL 다듬기 → 실제 리모), 안전·평가·정할 것 | 2026-10-03 설계. 실행기 접점·거르개 구현(1.4), 교사 PPO·대본 교사(4.8) |
| [VLA_INPUT.md](VLA_INPUT.md) | **우리 VLA 입력·출력 설계**: RGB 그림 3장(카메라 2 + 위에서 본 지도, 10-05) + 몸 상태 56 + 물체 칸 ≤ 16 × 289(상대 좌표, 이름 뜻·생김새 벡터, 손과의 거리, 출처·불확실도) + 벽·방 토큰, 일반화·외우지 않기 규칙과 평가 | 2026-10-03 설계, 절마다 구현 상태 적음 |

- 학습 구조(RL·대본 교사 → 시연 → 작은 학생 BC/DAgger, RecallVLA)는 [../../training/README.md](../../training/README.md) 의 `RL/`, `BC/`, `vla/`.
- 로봇 설명: [../../src/robot/README.md](../../src/robot/README.md).
- [MEMORY_ENCODER](MEMORY_ENCODER.md) — 기억 인코더 쉬운 설명(물체가 몇 개든 요약 32 개로, 상한·고르기, 교사·학생 같은 틀)
- [STATE_SPEC](STATE_SPEC.md) — 상태·입력 전체 목록(교사·학생, 칸 번호·단위·정규화·질의·행동)
