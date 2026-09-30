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
| π0.5 실행 | 기본은 학교 4090 원격 추론. **필요하면** 로봇에 라즈베리파이 5 추가, 그래도 느리면 NPU(AI HAT+ 2) 포팅 |

**학습·추론용 서버: 학교 컴퓨터 사용 가능.**

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 4090 (24GB) |
| 용도 | π0.5 파인튜닝(LoRA, 22.5GB 이상 필요), 추론 서버 (리모는 원격으로 결과만 받음), agent LLM(Qwen3.5-9B) API 서버 |

## 한눈에 보기

| 어디서 | 무엇을 |
|---|---|
| 리모 (Jetson Nano) | 센서 드라이버, 2D SLAM (Cartographer, 위치 추정), 이동 → RGB-D 영상 + pose 를 서버로 보냄 |
| 학교 4090 서버 | YOLO-seg, 3D 위치 계산, DA·맵 업데이트, Spark-DSG, Qwen3.5-9B (agent), π0.5 (VLA) |
| 시뮬레이션 | 2025 BEHAVIOR Challenge 벤치마크 (정답 pose 제공 → SLAM 없이 개발 가능) |

부품별 선택 이유는 [`model_selection.md`](model_selection.md).

## 1. Scene graph (로봇의 기억) — `src/scene_graph/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| SLAM | 리모에서 **2D 라이다 SLAM (Cartographer)** 으로 지도 한 번 만들고, 평소엔 위치만 추정 | |
| 물체 인식 (YOLO-seg) | 카메라 영상에서 물체 영역(mask)과 이름을 한 번에 얻기. 80종 밖의 물건이 필요하면 YOLOE / YOLO-World | |
| 3D 위치 계산 | 마스크 픽셀 + depth → 카메라 좌표 → 로봇 좌표 → 월드 좌표 (tf2). 마스크를 살짝 줄이고 **중앙값**으로 물체 위치 하나 뽑기 | |
| DA (data association) | 새로 본 물체가 그래프에 이미 있는 물체인지 판단. **자체 제작**, 같은 이름끼리 위치로 비교 | |
| 맵 업데이트 | 옮겨짐·사라짐·생김을 찾아 바뀐 부분만 고치기. **자체 제작**, 변화 탐지 P/R/F1 로 평가 | |
| 저장·보기 | **Spark-DSG** 로 저장, `spark-dsg visualize` 뷰어로 확인. agent 에는 JSON 으로 넘기기 | |

## 2. 로봇

| 할 일 | 내용 | 담당 |
|---|---|---|
| ROS 1 ↔ ROS 2 연결 | 기본 리모는 Ubuntu 18.04 (ROS 1). ros1_bridge 또는 ROS 2 Docker 중 되는 방법 찾기 | |
| 캘리브레이션 | 카메라 내부 값(intrinsic), 카메라가 로봇 몸체 어디에 달렸는지(extrinsic) → 3D 위치 계산에 필요. 팔이 정해지면 hand-eye 도 | |
| 좌표 사슬 (tf) | `map → base_link → camera` 가 제대로 나오는지 확인. depth 를 컬러에 맞춰 정렬(align) | |
| 서버로 보내기 | RGB-D 영상 + pose 를 Wi-Fi 로 4090 서버에 보내기. 느리면 해상도·프레임 줄이거나 압축 | |
| 이동 | 저장한 2D 지도 위에서 목표 위치로 이동 (내비게이션) | |
| 조작 | 매니퓰레이터 모델이 정해지면 ROS 2 로 움직이기, 집기·놓기 확인 | |
| pose 토픽 따두기 | 로봇·팔 끝(end-effector)·카메라 pose 토픽 이름, 메시지 타입, 주기 정리 | |
| ros2 bag 녹화 | 카메라(RGB·depth)·pose·관절 상태 녹화 → 오프라인으로 scene graph·VLA 실험. 녹화 스크립트는 `scripts/` | |
| 사양 문서 | 로봇·센서 사양, 토픽 목록 → `docs/` | |

## 3. VLA (π0.5) — `src/vla/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| 원격 추론 서버 | 4090 에서 openpi 정책 서버 띄우기, 리모는 행동만 받기 | |
| 시뮬레이션 평가 | **2025 BEHAVIOR Challenge 벤치마크** (BEHAVIOR-1K, OmniGibson) 로 평가, 대회 상위 팀 점수와 비교 | |
| 파인튜닝 (시뮬) | BEHAVIOR 대회 데이터(축소판 약 260GB) + 공개 체크포인트(1·2위 팀)에서 시작해 4090 LoRA 학습 | |
| 파인튜닝 (실제) | 리모로 모은 시연 데이터로 따로 학습 (BEHAVIOR 로봇과 몸이 달라서) | |
| 양자화 | 4090 에서 Qwen 과 같이 올라가게 메모리 줄이기. (필요하면) 라즈베리파이 5·NPU 용 | |
| (필요하면) 로봇 위 실행 | 라즈베리파이 5 에서 π0.5 돌리기 → 느리면 NPU(AI HAT+ 2) 포팅 | |
| 지시 형식 맞추기 | agent 가 주는 단계 지시(subtask) 형식에 맞추기 | |

## 4. Agent (계획·실패 복구) — `src/agent/`

| 할 일 | 내용 | 담당 |
|---|---|---|
| LLM 서버 | 4090 에 **Qwen3.5-9B** API 서버 띄우기 (vLLM 등, FP8 또는 4bit) | |
| 작업 나누기 | scene graph(JSON)를 읽고 "컵을 식탁에" → 이동·집기·이동·놓기 로 나누기 | |
| 실행·복구 | 단계마다 이동(로봇) 또는 VLA 호출 → 성공 여부 확인 → 실패하면 재계획 (참고: BEHAVIOR 3위 SimpleAI 논문) | |
| 평가 세트 | 명령 10~20개로 계획 성공률·응답 시간 재기 | |

## 5. 통합

| 할 일 | 내용 | 담당 |
|---|---|---|
| 시뮬레이션 통합 | BEHAVIOR 에서 scene graph → agent → VLA 전체 흐름 확인 (정답 pose 사용) | |
| 실제 로봇 통합 | 리모에서 SLAM pose 로 같은 흐름 확인 | |
| 4090 나눠 쓰기 | π0.5 학습(22.5GB 이상) 중에는 Qwen·시뮬 끄기. 평소엔 Qwen(FP8/4bit) + π0.5 추론 같이 | |

---

<details>
<summary>변경 기록</summary>

| 날짜 | 바뀐 것 | 이유 |
|---|---|---|
| 2026-09-30 | 계획 처음 작성 (scene graph·로봇·VLA·agent·통합) | |
| 2026-09-30 | 전제 추가: 기본 리모(AgileX LIMO)로 가정, 매니퓰레이터는 모델 미정 | 로봇 사양이 아직 확정 안 됨 |
| 2026-09-30 | 전제 추가: 학교 RTX 4090 (24GB) 서버 사용 가능 | 리모(Jetson Nano 4GB)에서는 π0.5 를 못 돌림 → 학습·추론은 서버에서 |
| 2026-09-30 | 물체 인식: SAM + CLIP → **YOLO-seg** 하나로 | 영역과 이름을 한 번에 줘서 더 빠르고 가벼움. 80종 밖의 물건이 필요하면 YOLOE / YOLO-World |
| 2026-09-30 | DA: **자체 제작**, 같은 이름끼리 위치로 비교 | CLIP 특징을 안 쓰므로. 같은 종류가 여러 개일 때가 약점 |
| 2026-09-30 | 맵 업데이트: **자체 제작** | |
| 2026-09-30 | Scene graph 저장·보기: **Spark-DSG + 뷰어** | Hydra·Khronos 와 같은 형식 |
| 2026-09-30 | π0.5 실행: 기본은 4090 원격 추론, **필요하면** 라즈베리파이 5 → NPU 포팅 | 로봇 위에서 직접 돌려야 할 때를 대비 |
| 2026-09-30 | SLAM: **2D 라이다 SLAM (Cartographer)**, 3D SLAM 안 씀 | Jetson Nano 에 3D SLAM 은 무거움. 물체 3D 위치는 pose + depth 로 충분 |
| 2026-09-30 | 시뮬레이션: **2025 BEHAVIOR Challenge 벤치마크** | 대회 상위 팀과 점수 비교 가능 |
| 2026-09-30 | Agent LLM: **Qwen3.5-9B**, 학교 4090 에서 API 로 | 요금·외부 인터넷 불필요. π0.5 와 4090 을 나눠 쓰려면 FP8/4bit |
| 2026-09-30 | 계획 전체 갱신: 한눈에 보기, 3D 위치 계산, ROS 1↔2 연결·tf·영상 전송, 파인튜닝 시뮬/실제 분리, LLM 서버, 4090 나눠 쓰기 추가 | 위 결정들 반영 |

자세한 선택 과정은 [`model_selection.md`](model_selection.md) 참고.

</details>
