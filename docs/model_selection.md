# 모델 선택

로봇의 각 부품에 **무엇을 골랐고, 왜 골랐는지** 정리한 문서.
할 일과 전체 구조는 [계획](plan.md), 참고 논문·코드는 [참고 자료](../refs/README.md).

## 한눈에 보기

| 부품 | 선택 | 한 줄 이유 | 상태 |
|---|---|---|---|
| 지도·위치 (SLAM) | **Cartographer (2D 라이다)** | 리모에서 가볍게 돌고, 물체 높이는 depth 로 알 수 있음 | 결정 |
| 물체 인식 | **YOLO-seg (nano)**, 리모에서 TensorRT | 물체 영역과 이름을 한 번에 | 결정 |
| 같은 물체 판단 (DA) | **직접 만듦** | 같은 이름끼리 위치로 비교 | 결정 |
| 지도 갱신 | **직접 만듦** | 바뀐 부분만 고침 | 결정 |
| 물체 지도 저장·보기 | **Spark-DSG + 뷰어** | Hydra·Khronos 와 같은 형식, 웹 3D 뷰어 포함 | 결정 |
| 계획 (LLM) | **Qwen3.5-9B**, 학교 4090 API | 요금·외부 인터넷 없이 사용 | 결정 |
| 행동 (VLA) | **π0.5** | 2025 BEHAVIOR 대회 상위 팀이 모두 사용 | 잠정 |
| π0.5 실행 위치 | **리모**, 안 되면 라즈베리파이 5 + DEEPX DX-M1 (보유) | 로봇이 서버 없이 스스로 움직이게 | 결정 |
| 시뮬레이션 | **2025 BEHAVIOR Challenge 벤치마크** | 대회 상위 팀과 점수를 비교할 수 있음 | 결정 |
| 앱 | **iOS (SwiftUI) · Android (Compose)** | 네이티브, 카카오톡식 채팅 | 결정 |

## 고를 때 따지는 것

모든 부품에 공통으로 걸리는 조건이다.

| 조건 | 영향 |
|---|---|
| **우리 코드는 전부 리모에서 돈다** (Jetson Nano 4GB) | 모든 모델이 4GB 안에 같이 올라가야 한다 → 가벼운 것 우선 |
| 학교 4090 (24GB) 은 Qwen API 와 학습만 | 로봇이 쓰는 모델은 4090 에 기대지 않는다 |
| 리모 기본형은 Ubuntu 18.04 (ROS 1) | ROS 2 로 포팅된 리모 패키지를 찾아 쓴다 |

새로 고르거나 바꿀 때는 이렇게 한다.

1. 후보를 2~3개로 줄인다.
2. **같은 데이터**(우리 ros2 bag 녹화본이나 같은 시뮬 장면)로 비교한다.
3. 아래 "잴 것"을 숫자로 재서 남긴다 (실험은 `tests/sandbox/`).
4. 가장 중요한 조건을 통과한 것 중 가장 가벼운 걸 고르고, 위 표를 고친다.

---

## 부품별 자세히

### 지도·위치 (SLAM) — Cartographer (2D 라이다)

- 리모로 방을 한 번 돌며 2D 지도를 만들고, 평소엔 그 지도 위에서 **위치만** 추정한다.
- 물체의 3D 위치는 SLAM 이 아니라 `로봇 위치 + 카메라 장착 위치 + depth` 로 계산한다. 그래서 무거운 3D SLAM 은 필요 없다.
- 시뮬레이션은 정답 위치를 주므로, 물체 기억·계획 파트는 SLAM 이 끝나기를 기다리지 않아도 된다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| RTAB-Map (3D) | Jetson Nano 에 무거움 |
| slam_toolbox (2D) | 리모 공식 데모는 Cartographer 쪽 |

잴 것: 리모 CPU·메모리 사용량, 방을 한 바퀴 돌고 출발점에 왔을 때 어긋난 거리.

### 물체 인식 — YOLO-seg (nano)

- 카메라 영상에서 물체의 **영역과 이름을 한 번에** 준다.
- 컵·병·의자·식탁·소파·노트북·책 등 집 안 물건은 기본 목록(COCO 80종)에 대부분 들어 있다. 목록 밖의 물건이 필요해지면 원하는 단어로 찾는 YOLOE / YOLO-World 로 바꾼다.
- **리모에서 돌린다**: 가장 작은 nano 모델을 TensorRT FP16 엔진으로 바꿔 C++ 로 실행 (참고: [Qengineering/YoloV8-TensorRT-Jetson_Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano), `tensorrt8` 브랜치). 엔진은 리모와 같은 TensorRT 버전으로 만들어야 한다.
- 속도 참고치: Jetson Nano 에서 검출만 약 19 FPS (YOLOv8n, FP16). 영역까지 내면 더 느리므로 우리 리모에서 직접 잰다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| FastSAM | 빠르지만 이름이 없고 물체가 조각남 (처음엔 골랐다가 바꿈) |
| MobileSAM · EfficientViT-SAM · SAM 2 | 이름을 따로 붙여야 하고, 리모에는 무거움 |

잴 것: 리모에서 FPS, 우리 물체를 놓치거나 이름을 틀리는 비율, 메모리.

### 같은 물체 판단 (DA) — 직접 만듦

- 새로 본 물체가 이미 지도에 있는 물체인지 판단한다. **같은 이름끼리 위치로 비교**한다 (컵은 컵끼리만).
- 약점: 컵이 두 개일 때 어느 컵이 옮겨졌는지 구분이 어렵다 → 부족하면 색 분포 같은 가벼운 생김새 정보를 붙인다.
- 비교 기준: 위치 거리만 쓰는 가장 단순한 방법.

잴 것: 같은 물체가 두 번 등록된 수, 다른 물체가 하나로 합쳐진 수.

### 지도 갱신 — 직접 만듦

- 물체가 옮겨지거나·사라지거나·새로 생기면 지도에서 **바뀐 부분만** 고친다.
- 참고: DovSG (국소 갱신), Khronos (시간에 따른 변화 추적).

잴 것: 옮겨짐·사라짐·생김을 맞힌 비율 (P/R/F1), 갱신 속도.

### 물체 지도 저장·보기 — Spark-DSG + 뷰어

- Hydra·Khronos 가 쓰는 형식이라 참고 코드의 도구를 그대로 쓸 수 있다. 층·방·물체 계층을 지원한다.
- 뷰어가 들어 있어 웹 브라우저에서 3D 로 볼 수 있다.

```bash
pip install spark-dsg
spark-dsg visualize 지도.json   # http://localhost:8080
```

agent(LLM) 에게는 읽기 쉽게 JSON 으로 바꿔 넘긴다.

### 계획 (LLM) — Qwen3.5-9B, 학교 4090 API

- 리모의 agent 가 학교 4090 의 API 로 질문을 보낸다. 요금이 없고 외부 인터넷도 필요 없다.
- 4090 에 원래 크기(FP16, 약 18GB)로 혼자 올라간다. π0.5 학습(22.5GB 이상) 중에는 끈다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| Claude · GPT 등 유료 API | 요금과 외부 인터넷이 필요 |

잴 것: 명령 10~20개로 단계를 제대로 나누는 비율, 응답 시간.

### 행동 (VLA) — π0.5 (잠정)

- 단계 지시와 카메라 영상을 받아 로봇을 움직인다. 2025 BEHAVIOR 대회 1~3위 팀이 모두 π0.5 를 썼고, 공개된 모델과 코드가 많다.
- 학습은 4090 에서 LoRA (22.5GB 이상 필요, 4090 에 겨우 들어감).

| 다른 후보 | 크기 | 특징 |
|---|---|---|
| SmolVLA | 0.45B | 훨씬 가벼움 → π0.5 가 리모에서 끝내 안 돌면 대안 |
| GR00T N1.5 | 3B | NVIDIA, 실행에 16GB 이상 |

잴 것: 시뮬레이션에서 집기·놓기 성공률.

### π0.5 실행 위치 — 리모, 안 되면 라즈베리파이 5 + DEEPX DX-M1

| 순서 | 방법 | 확인할 것 |
|---|---|---|
| 1 | 리모(Jetson Nano) 에서 양자화한 π0.5 | 공식 기준 실행에 8GB 이상 → 4bit 로 줄여도 4GB 에 빠듯함. 먼저 재 본다 |
| 2 | 안 되면 리모에 라즈베리파이 5 + **DEEPX DX-M1** (보유, 25 TOPS INT8, 메모리 4GB, M.2) | DXNN SDK 로 ONNX 모델을 변환해 올린다. π0.5 는 3.3B 라 INT8 로도 약 3.3GB → **4GB 에 빠듯함**. transformer·행동 생성 부분이 변환되는지 확인 |

양자화 단계: bf16 (원래) → int8 → int4 순으로 줄이며 성공률이 얼마나 떨어지는지 잰다.

### 시뮬레이션 — 2025 BEHAVIOR Challenge 벤치마크

- 대회가 준 과제와 평가 방식을 그대로 쓴다 → 점수를 대회 상위 팀과 바로 비교할 수 있다.
- 사람이 조종한 시연 데이터 약 10,000개와 상위 팀의 학습된 모델이 공개돼 있어 학습도 여기서 시작한다.
- 대회 로봇(Galaxea R1 Pro, 바퀴 + 양팔) 은 리모와 몸이 다르다 → 실제 리모용은 따로 학습한다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| LIBERO | 책상 위 팔 조작만 있고 이동이 없음 |
| Gazebo | 리모 모델은 있지만 VLA 학습용으로는 약함 |

### 앱 — iOS (SwiftUI) · Android (Compose)

| 고를 것 | 선택 | 비고 |
|---|---|---|
| iOS | Swift / SwiftUI | 네이티브 |
| Android | Kotlin / Jetpack Compose | 네이티브 |
| 앱 ↔ 리모 연결 | WebSocket (추천, 미정) | 진행 상황을 실시간으로 받아야 해서 |
| 학교 밖 접속 | VPN 등 (미정) | 학교 네트워크 규칙 확인 필요. 로봇을 인터넷에 그대로 열면 위험 |
| 로봇 목록 | 앱에 고정하지 않음 | 지금은 로봇1 만, 나중에 로봇2… 를 앱 수정 없이 추가 |

---

## 출처

- 리모 사양·SLAM 데모: [AgileX LIMO 사양](https://www.wevolver.com/specs/agilex-limo), [LIMO ROS2 매핑·내비게이션](https://www.hackster.io/agilexrobotics/ros2-mapping-and-navigation-with-limo-ros2-1936a9), [Trossen LIMO 데모](https://docs.trossenrobotics.com/agilex_limo_docs/demos.html)
- 물체 인식: [SAM 계열 속도·정확도 비교 (2025)](https://scitepress.org/PublishedPapers/2025/137785), [MobileSAM](https://docs.ultralytics.com/ko/models/mobile-sam), [YOLOv8 TensorRT Jetson Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano)
- LLM: [Qwen3.5-9B GPU 가이드](https://www.spheron.network/tools/gpu-recommender/Qwen/Qwen3.5-9B/)
- VLA 크기·메모리: openpi README (`refs/code/openpi/README.md`), [VLA 비교 연구 (arXiv 2603.19233)](https://arxiv.org/pdf/2603.19233)
- NPU: [DEEPX DX-M1 사양 (DFRobot)](https://wiki.dfrobot.com/SKU_DFR1252_DX-M1%20AI%20Accelerator), [dx-all-suite (DEEPX SDK)](https://github.com/juyoung020/dx-all-suite)
- 시뮬레이션: [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/archive/2025/call_for_participation.html)
