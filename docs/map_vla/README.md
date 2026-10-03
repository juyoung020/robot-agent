# docs/map_vla — Map_Vla (리모 + 매니퓰레이터 VLA) 설계 문서

Map_Vla 프로젝트에서 쓴 조사·설계 문서다. 작성 시점의 판단이라 일부는 바뀌었다(아래 표의 비고).

| 문서 | 내용 | 비고 |
|---|---|---|
| [MAP_STATE_PLAN.md](MAP_STATE_PLAN.md) | 지도를 학습 state 수치로 넣는 계획(물체 슬롯 16 × 14, 벽 벡터 56) | 물체 간 전치사 관계는 쓰지 않음 |
| [SIM_PORTING.md](SIM_PORTING.md) | 리모 + OMX 시뮬을 GPU 에서 수천 개 돌리는 설계, 카메라 외부 파라미터 표 | 코드는 아직 없음 |
| [PORT_PLAN.md](PORT_PLAN.md) | MuJoCo Warp 전체 포팅 계획 | 팀 `sim/engine` 과 겹침 — 먼저 BEHAVIOR_ENGINE_REVIEW 를 볼 것 |
| [BEHAVIOR_ENGINE_REVIEW.md](BEHAVIOR_ENGINE_REVIEW.md) | 팀 `behavior-2026` 엔진을 우리 로봇 학습에 쓸 수 있는지 검토 | |
| [VLA_QnA.md](VLA_QnA.md) | VLA 설계 질문과 답 | |
| [untitled.md](untitled.md) | Gemini 와 나눈 VLA 설계 대화 원본(중복 많음). 정리본이 VLA_QnA.md | 원본 |
| [tf_tree.pdf](tf_tree.pdf) | 리모 + OMX 통합 TF 트리 | |
| [move_robot_gt_pose.patch](move_robot_gt_pose.patch) | 시뮬 GT 자세를 `move_robot` 에 넘기는 패치 | 이미 적용됨(behavior-2026 `sim/`) |

- 학습 구조(RL 전문가 → 시연 → Qwen VLA 모방학습)는 [../../training/README.md](../../training/README.md) 의 `RL/`, `BC/`.
- 로봇 설명: [../../src/robot/README.md](../../src/robot/README.md).
