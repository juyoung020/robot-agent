# docs/map_vla — Map_Vla (리모 + 매니퓰레이터 VLA) 설계 문서

Map_Vla 프로젝트에서 쓴 조사·설계 문서다. 작성 시점의 판단이라 일부는 바뀌었다(아래 표의 비고).

| 문서 | 내용 | 비고 |
|---|---|---|
| [CURRICULUM_APPROACH.md](CURRICULUM_APPROACH.md) | **커리큘럼 1: 이동**("컵으로 가") — 성공(가까움·정면 에임·보임·멈춤)·충돌 = 실패, 보상(경로 거리·에임), 단계 A0–A5, 지표 SR·SPL·충돌률 | 2026-10-03 설계 |
| [CURRICULUM_BEHAVIOR2026.md](CURRICULUM_BEHAVIOR2026.md) | **커리큘럼 2: BEHAVIOR 2026 기준** — 100 과제 목록(장면·물체·목표 술어·스킬), q_score 규칙, LIMO + OMX-F 로 되는 과제(q 상한 추정), 스킬 빈도, 새 단계 B0–B7(옛 A0–A5 대응), 학습 환경 차이 분석, 환경 개선 단계 E0–E7(장면·과제 변환기 포함), 열린 질문 | 2026-10-04 설계. 커리큘럼 1 을 이 문서 기준으로 다시 짬 |
| [TRAIN_VIEWER.md](TRAIN_VIEWER.md) | **학습 뷰어 설계(Rust)**: sgview 방식 Rust 서버 + canvas 차트·three.js 재생, 학습·재생·비교 탭, 실행 종류(teacher·bc·dagger·rlft·eval·lab), 학습 고리가 쓰는 파일 규약(run.json·progress·episodes·이진 재생), 전투기 뷰어에서 가져온 규칙 20개, 단계 | 2026-10-03 설계 |
| [GPU_TRAINING.md](GPU_TRAINING.md) | **GPU 만으로 학습하는 설계**(C++/CUDA/Rust): 5070 Ti(sm_120)에서 되는 FP8 명령, CUDA graph 로 CPU 없는 학습 고리, JSBSim GPU 포팅 방식의 환경 커널, 자라는 지도(GPU scenemap 근사), 커널 합치기, 정밀도, 16 GB 메모리, 검증 사다리, 예상 속도, 단계 | 2026-10-03 설계 |
| [POLICY.md](POLICY.md) | **정책 설계**: LLM → VLA·`move_robot` 역할 나누기, 교사 RL(스킬·보상·비대칭 actor-critic·자라는 지도·커리큘럼), 학생 VLA(몸통 두 안·행동 묶음·지연), 학습 순서(RL 교사 → BC + DAgger → VLA RL 다듬기 → 실제 리모), 안전·평가·정할 것 | 2026-10-03 설계 |
| [VLA_INPUT.md](VLA_INPUT.md) | **우리 VLA 입력·출력 설계**: 카메라 2장 + 몸 상태 56 + 물체 칸 ≤ 16 × 289(상대 좌표, 이름 뜻·생김새 벡터, 손과의 거리, 출처·불확실도) + 벽·방 토큰, 일반화·외우지 않기 규칙과 평가 | 2026-10-03 설계 |
| [PI05_TRAINING.md](PI05_TRAINING.md) | π0.5 는 어떻게 학습됐나: 모델, 사전학습(28만 스텝, FAST 토큰, MM·ME·CE·HL·WD)·후학습(8만 스텝, flow matching 행동 전문가, VI), 제거 실험, 지식 격리, 우리 VLA 에 가져올 것 | 2026-10-03 조사 |
| [HUMAN_DEMOS.md](HUMAN_DEMOS.md) | 다른 로봇의 사람 원격조종 시연 + scenemap 재생: BEHAVIOR 2026 시연 20,000 판 메타데이터 거르기(E0 팔 한도, 태그 4 종), ep 0 재생, 작은 팔·DROID·OXE 조사, 채택 안 한 이유·재개 방법 | 2026-10-05 **보류(문서화만)** — 주 시연 출처는 GPU 시뮬 RL 교사 |
| [MAP_STATE_PLAN.md](MAP_STATE_PLAN.md) | 지도를 학습 state 수치로 넣는 계획(물체 슬롯 16 × 14, 벽 벡터 56) | 물체 간 전치사 관계는 쓰지 않음 — 물체 칸은 VLA_INPUT.md 가 바꿈 |
| [SIM_PORTING.md](SIM_PORTING.md) | 리모 + OMX 시뮬을 GPU 에서 수천 개 돌리는 설계, 카메라 외부 파라미터 표 | 코드는 아직 없음 |
| [PORT_PLAN.md](PORT_PLAN.md) | MuJoCo Warp 전체 포팅 계획 | 팀 `sim/engine` 과 겹침 — 먼저 BEHAVIOR_ENGINE_REVIEW 를 볼 것 |
| [BEHAVIOR_ENGINE_REVIEW.md](BEHAVIOR_ENGINE_REVIEW.md) | 팀 `behavior-2026` 엔진을 우리 로봇 학습에 쓸 수 있는지 검토 | |
| [VLA_QnA.md](VLA_QnA.md) | VLA 설계 질문과 답 | |
| [untitled.md](untitled.md) | Gemini 와 나눈 VLA 설계 대화 원본(중복 많음). 정리본이 VLA_QnA.md | 원본 |
| [tf_tree.pdf](tf_tree.pdf) | 리모 + OMX 통합 TF 트리 | |
| [move_robot_gt_pose.patch](move_robot_gt_pose.patch) | 시뮬 GT 자세를 `move_robot` 에 넘기는 패치 | 이미 적용됨(behavior-2026 `sim/`) |

- 학습 구조(RL 전문가 → 시연 → Qwen VLA 모방학습)는 [../../training/README.md](../../training/README.md) 의 `RL/`, `BC/`.
- 로봇 설명: [../../src/robot/README.md](../../src/robot/README.md).
