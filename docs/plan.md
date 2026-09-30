# 계획 (plan)

동적 3D scene graph + AI agent + VLA(π0.5) 결합.
참고 논문·코드는 [`refs/README.md`](../refs/README.md) 참고.

## 전제

**기본 리모(AgileX LIMO)로 가정.**

| 항목 | 기본 LIMO (가정) |
|---|---|
| 컴퓨터 | NVIDIA Jetson Nano (4GB) |
| 라이다 | EAI X2L |
| 카메라 | Orbbec DaBai (RGB-D) |
| 로봇 팔 | 없음 |
| 매니퓰레이터 | 추가로 달 예정 — **아직 모델 모름** |

**학습·추론용 서버: 학교 컴퓨터 사용 가능.**

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 4090 (24GB) |
| 용도 | π0.5 파인튜닝(LoRA, 22.5GB 이상 필요), 추론 서버 (리모는 원격으로 결과만 받음) |

## 1. Scene graph (로봇의 기억) — `src/scene_graph/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| SLAM | 로봇 위치(pose)와 3D 지도 만들기. RGB-D 카메라 기준으로 SLAM 고르고, ROS 2 로 pose·지도 토픽 나오는지 확인 | |
| SAM | 카메라 영상에서 물체 영역(mask) 잘라내기 | |
| CLIP | 잘라낸 물체마다 의미 특징(feature) 뽑기 → 글로 검색 가능하게 (open-vocabulary) | |
| DA (data association) | 새로 본 물체가 그래프에 이미 있는 물체인지, 새 물체인지 판단 | |
| 맵 업데이트 | 물체가 옮겨지거나 사라지면 그래프의 해당 부분만 고치기 (참고: DovSG) | |
| 그래프 내보내기 | agent 가 읽을 수 있는 형태(JSON 등)로 | |

## 2. 로봇

| 할 일 | 내용 | 담당 |
|---|---|---|
| 캘리브레이션 | 카메라 내부 파라미터(intrinsic), 카메라-로봇 위치 관계(hand-eye / extrinsic) | |
| 조작 | 팔·그리퍼를 ROS 2 로 움직이기, 기본 동작(집기·놓기) 확인 | |
| pose 토픽 따두기 | 로봇·팔 끝(end-effector)·카메라 pose 토픽 이름, 메시지 타입, 주기 정리 | |
| ros2 bag 녹화 | 카메라(RGB·depth)·pose·관절 상태 녹화 → 오프라인으로 scene graph·VLA 실험. 녹화 스크립트는 `scripts/` | |
| 사양 문서 | 로봇·센서 사양, 토픽 목록 → `docs/` | |

## 3. VLA (π0.5) — `src/vla/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| π0.5 돌려보기 | openpi 로 기본 추론(inference) 확인 | |
| 학습용 시뮬레이션 구하기 | LIBERO, BEHAVIOR-1K 등 비교 → 우리 로봇·작업과 비슷한 환경 고르기 (참고: 2025 BEHAVIOR Challenge 상위 팀) | |
| 파인튜닝 | 우리 작업 데이터(시뮬 + ros2 bag)로 π0.5 추가 학습 | |
| 양자화 | 로봇 컴퓨터에서 돌아갈 만큼 메모리·속도 줄이기 | |
| 지시 형식 맞추기 | agent 가 주는 단계 지시(subtask) 형식에 맞추기 | |

## 4. Agent (계획·실패 복구) — `src/agent/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| 작업 나누기 | scene graph 를 읽고 작업을 단계로 나누기 | |
| 실행·복구 | 단계마다 VLA 호출 → 성공 여부 확인 → 실패하면 재계획 (참고: BEHAVIOR 3위 SimpleAI 논문) | |

## 5. 통합

| 할 일 | 내용 | 담당 |
|---|---|---|
| 시뮬레이션 통합 | scene graph → agent → VLA 전체 흐름 확인 | |
| 실제 로봇 통합 | 실제 로봇에서 확인 | |
