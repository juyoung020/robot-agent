# 모델 선택

로봇의 각 부품에 **무엇을 골랐고, 왜 골랐는지** 정리한 문서.
할 일과 전체 구조는 [계획](plan.md), 참고 논문·코드는 [참고 자료](../refs/README.md).

## 한눈에 보기

| 부품 | 선택 | 한 줄 이유 | 상태 |
|---|---|---|---|
| 지도·위치 (SLAM) | **Cartographer (2D 라이다)** | 리모에서 가볍게 돌고, 물체 높이는 depth 로 알 수 있음 | 결정 |
| 물체 인식 | **FastSAM-s (입력 416) + SigLIP 2 B/32** — SAM + CLIP 구조로 돌아옴 (그 사이: YOLO-seg nano, 시뮬은 아직 YOLOE) | 임베딩 벡터 찾기를 택해서 물체마다 CLIP 류 임베딩이 필요 | 결정, 코드 작업 중 |
| 같은 물체 판단 (DA) | **직접 만듦** | 같은 이름끼리 위치로 비교 | 결정 |
| 물체 찾기 | **임베딩 벡터 찾기 + 이름(의미) 찾기** 둘 다 | 질의 글 ↔ 물체 벡터 코사인, 라벨 표 이름·상위어 | 결정 |
| 지도 갱신 | **직접 만듦** | 바뀐 부분만 고침 | 결정 |
| 물체 지도 저장·보기 | **Spark-DSG** 저장, 보기는 **2D 지도** | Hydra·Khronos 와 같은 형식. 웹 3D 뷰어는 안 씀(10-02) | 결정 |
| 큰 계획·대화 (LLM) | **Qwen3.5-9B**, AI agent 수업이 주는 KAU API | 요금 없이 사용 (인터넷 필요) | 결정 |
| 작은 계획·행동 (VLA) | **π0.5** | 2025 BEHAVIOR 대회 상위 팀이 모두 사용 | 잠정 |
| π0.5 실행 위치 | **리모**, 안 되면 라즈베리파이 5 + DEEPX DX-M1 (보유) | 로봇이 서버 없이 스스로 움직이게 | 결정 |
| 시뮬레이션 | **2026 BEHAVIOR Challenge 벤치마크** (참고 코드는 2025 상위 팀 것) | 대회 상위 팀과 점수를 비교할 수 있음 | 결정 |
| 앱 | **iOS (SwiftUI) · Android (Compose)** | 네이티브, 카카오톡식 채팅 | 결정 |

## 고를 때 따지는 것

모든 부품에 공통으로 걸리는 조건이다.

| 조건 | 영향 |
|---|---|
| **우리 코드는 전부 리모에서 돈다** (Jetson Nano 4GB) | 모든 모델이 4GB 안에 같이 올라가야 한다 → 가벼운 것 우선 |
| LLM 은 AI agent 수업(최영식 교수님)이 주는 Qwen API, 학교 4090 (24GB) 은 π0.5 LoRA 학습만 | 로봇이 쓰는 모델은 4090 에 기대지 않는다 |
| 시뮬 작업 PC `jy-desktop` (RTX 5070 Ti 16GB) | 시뮬 평가·π0.5 추론은 여기서 돈다. LoRA 학습은 16GB 에 안 들어가서 학교 4090 으로 |
| 리모 기본형은 Ubuntu 18.04 (ROS 1) | ROS 2 로 포팅된 리모 패키지를 찾아 쓴다 |
| **리모 프로를 받을 수도 있다** (Jetson Orin Nano 8GB, ROS 2 Foxy 공식 지원) | 받으면 메모리 한도가 8GB 로, CUDA·TensorRT 도 새 버전을 쓸 수 있다 → 아래 선택 중 "리모에 무거워서" 뺀 것들을 다시 볼 수 있다 |

새로 고르거나 바꿀 때는 이렇게 한다.

1. 후보를 2~3개로 줄인다.
2. **같은 데이터**(우리 ros2 bag 녹화본이나 같은 시뮬 장면)로 비교한다.
3. 아래 "잴 것"을 숫자로 재서 남긴다 (실험은 `tests/sandbox/`).
4. 가장 중요한 조건을 통과한 것 중 가장 가벼운 걸 고르고, 위 표를 고친다.

---

## 부품별 자세히

### 지도·위치 (SLAM) — Cartographer (2D 라이다)

- 리모로 방을 한 번 돌며 2D 지도를 만들고, 평소엔 그 지도 위에서 **위치만** 추정한다.
- 물체 위치(xyz)는 SLAM 이 아니라 `로봇 위치 + 카메라 장착 위치 + 물체 마스크(지금은 FastSAM-s) 무게중심의 depth` 로 계산해 기록한다. 그래서 무거운 3D SLAM 은 필요 없다.
- 시뮬레이션은 정답 위치를 주므로, 물체 기억·계획 파트는 SLAM 이 끝나기를 기다리지 않아도 된다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| RTAB-Map (3D) | Jetson Nano 에 무거움 |
| slam_toolbox (2D) | 리모 공식 데모는 Cartographer 쪽 |

잴 것: 리모 CPU·메모리 사용량, 방을 한 바퀴 돌고 출발점에 왔을 때 어긋난 거리.

### 물체 인식 — FastSAM-s (416) + SigLIP 2 B/32, SAM + CLIP 구조로 돌아옴 (작업 중)

흐름: 처음엔 SAM + CLIP → 09-30 에 YOLO-seg 하나로 바꿈 → **물체 찾기를 임베딩 벡터 찾기로 정하면서 다시 SAM + CLIP 구조**(분할 = FastSAM-s, 임베딩 = SigLIP 2). YOLO 는 이름만 주고 물체 벡터를 주지 않아서, 벡터로 찾으려면 CLIP 류 인코더가 따로 있어야 한다.

- **FastSAM-s**(입력 416)가 keyframe 에서만 이름 없는 물체 마스크를 낸다. 클래스가 없어서 목록 밖 물건도 빠짐없이 잡는다.
- **SigLIP 2 B/32** 가 물체 조각(원본 RGB 에서 상자 + 10 % 둘레, 정사각)마다 768-d 영상 임베딩을 낸다. 새 물체 / best view 가 바뀐 물체만, 묶어서, 비동기로 돈다.
- 이름은 그 임베딩과 미리 계산한 **라벨 표**(글 임베딩)의 코사인으로 고른다. 확신이 낮으면 WordNet 상위어로 올린다. 벽·바닥 같은 구조물은 표시만 하고 agent 목록에서 뺀다.
- 저장: 물체 벡터는 **원본 임베딩 그대로**(`objects/O<id>_emb.f16`), 이름은 다시 만들 수 있는 **캐시**(기억 폴더 `cache/names.json`).
- 코드는 서브모듈 `src/behavior-2026/src/scene_graph/clip`(작업 중). 지금 시뮬에서 돌아가는 검출기는 아직 YOLOE(`src/behavior-2026/src/scene_graph/ovdet`). 후보·측정: [CLIP 후보](clip_candidates.md), [물체 인식 모델 후보](perception_model_candidates.md).

**왜 SAM + CLIP 으로 돌아왔나 (YOLO-seg → FastSAM-s + SigLIP 2)**

- **임베딩 벡터 찾기를 택했다.** 질의 글 벡터와 물체 벡터의 코사인으로 찾으려면 물체마다 영상·글이 정렬된 임베딩이 있어야 한다 → CLIP 류(SigLIP 2) 인코더가 필요하고, 그러면 분할은 이름 없는 마스크(FastSAM-s)로 충분하다.
- **이름보다 임베딩이 중요했다.** 우리 데이터에서 이름 붙이기 정답률은 0.30–0.35 에 그쳤지만(라디오는 모든 모델이 0), 글 → 물체 찾기는 SigLIP 2 B/32 가 영어 R@1 0.70, 한국어 0.65 로 "라디오"·"빨간 라디오"를 1·2위 안에 찾았다. 그래서 물체마다 벡터를 저장하고 이름은 캐시로 둔다.
- **아무 말로나 찾기가 필요해졌다.** "빨간 컵"처럼 생김새로, 한국어로 찾으려면 영상·글이 정렬된 임베딩이 있어야 한다. YOLO 는 정해진 이름만 준다.
- **같은 이름 여러 개·목록 밖 물건**(처음 YOLO 를 고를 때 따로 챙길 것으로 적었던 약점)을 임베딩이 메운다.
- **리모 예산 안(추정)**: 영상 인코더 엔진 약 0.22–0.24 GB + 라벨 표 0.05 GB. Nano 에서 새 물체 5개 keyframe 약 0.35–0.5 s(비동기라 SLAM·주행은 안 막음).
- 주의: SigLIP 2 는 TensorRT FP16 그대로면 망가진다 → LayerNorm 을 FP32 로 고정한다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| YOLO-seg nano (09-30 첫 선택) | 영역과 이름을 한 번에 주지만 정해진 이름만, 목록 밖 물건·자유 글 찾기가 안 됨 |
| YOLOE (시뮬에서 지금 돌아감) | 단어로 찾을 수 있지만 프롬프트 어휘 안에서만. 물체 벡터가 없어 생김새로 못 찾음 |
| OpenAI CLIP B/32 + 한국어 글 인코더 | 바로 쓸 수 있는 2순위 대안. 한국어 찾기는 좋지만 이름 정답률이 낮음(0.28) |
| MobileCLIP 계열 | 이름 정확도는 가장 높지만 가중치가 비상업 라이선스 |
| MobileSAM · EfficientViT-SAM · SAM 2 | 리모에 무거움. 정밀한 영역은 π0.5 에 쓰이지 않음 |

잴 것: 리모에서 keyframe 지연·메모리, 이름 정답률, 글 → 물체 찾기 R@1(영어·한국어).

<details>
<summary>처음 결정(09-30): YOLO-seg nano 를 골랐던 근거</summary>

- 영역과 이름을 한 번에 줘서 모델 하나로 끝나고, nano + TensorRT FP16 이면 Jetson Nano 에서 실시간에 가깝다(검출만 약 19 FPS).
- 물체를 집는 건 π0.5 라 정밀한 영역이 필요 없고, 위치는 영역 안 depth 의 중앙값이라 대충 맞는 영역이면 충분하다(이 근거는 지금도 맞다).
- 그때는 "아무 단어로 찾기는 필요 없다"고 봤다 → 임베딩 벡터 찾기를 택하면서 SAM + CLIP 구조로 돌아왔다.

</details>

### 같은 물체 판단 (DA) — 직접 만듦

- 새로 본 물체가 이미 지도에 있는 물체인지 판단한다. **같은 이름끼리 위치로 비교**한다 (컵은 컵끼리만).
- 약점: 컵이 두 개일 때 어느 컵이 옮겨졌는지 구분이 어렵다 → 부족하면 생김새 정보를 붙인다. 물체마다 SigLIP 2 임베딩이 이미 있으므로 그 코사인 유사도가 첫 후보다(아직 안 정함).
- 비교 기준: 위치 거리만 쓰는 가장 단순한 방법.

잴 것: 같은 물체가 두 번 등록된 수, 다른 물체가 하나로 합쳐진 수.

### 물체 찾기 — 임베딩 벡터 찾기 + 이름(의미) 찾기, 둘 다 쓴다

- **임베딩 벡터 찾기**: 질의 글("빨간 컵", 한국어도)을 SigLIP 2 B/32 글 공간의 벡터로 바꾸고, 기억 속 물체 벡터(물체 조각의 원본 영상 임베딩)와 **코사인 유사도**(L2 정규화한 벡터의 내적)로 비교해 가까운 순으로 고른다. 이름표에 없는 말("물 마시는 거")로도 찾을 수 있다.
- **이름(의미) 찾기**: 물체마다 라벨 표에서 고른 이름·상위어(기억 폴더 `cache/`)를 붙여 두고, 이름이나 상위어("컵" ⊂ "식기")로 찾는다. 빠르고 결과를 설명하기 쉽다.
- 찾은 물체(이름·위치·상태)를 LLM 프롬프트에 넣어 답·계획을 만들게 하는 구조가 **RAG**(Retrieval-Augmented Generation, 검색 증강 생성)다. 우리 agent 는 물체 기억을 도구로 찾아 그 결과로 답하므로 이 방식에 해당한다.
- 임베딩·라벨 표·한국어 질의 모델은 [`training/embed`](../training/embed/README.md), 실행 쪽은 서브모듈 `src/behavior-2026/src/scene_graph/clip`.

### 지도 갱신 — 직접 만듦

- 물체가 옮겨지거나·사라지거나·새로 생기면 지도에서 **바뀐 부분만** 고친다.
- 참고: DovSG (국소 갱신), Khronos (시간에 따른 변화 추적).

잴 것: 옮겨짐·사라짐·생김을 맞힌 비율 (P/R/F1), 갱신 속도.

### 물체 지도 저장·보기 — Spark-DSG + 뷰어

- Hydra·Khronos 가 쓰는 형식이라 참고 코드의 도구를 그대로 쓸 수 있다. 층·방·물체 계층을 지원한다.
- 기억 폴더: `scene.json`(Spark-DSG), `objects/`(물체마다 RGB·depth·마스크·점구름·임베딩 = 원본), `cache/`(이름·찾기 색인 = 지워도 다시 만듦). 지도 자세는 `SGRT_POSE`(`slam`·`odom`·`gt`, 시뮬은 `gt`)로 고른다.
- 보기는 **2D 지도** 뷰어 sgviz(서브모듈 `src/behavior-2026/src/scene_graph/viewer`, Spark-DSG + viser). spark-dsg 의 웹 3D 뷰어는 안 쓴다(10-02).

agent(LLM) 에게는 읽기 쉽게 JSON 으로 바꿔 넘긴다.

### 큰 계획·대화 (LLM) — Qwen3.5-9B, KAU API (AI agent 수업)

- 리모의 agent 가 AI agent 수업(최영식 교수님)이 주는 Qwen API 로 질문을 보낸다: `https://agent.kau.ac.kr/v1`, 모델 `qwen3.5-9b` (vLLM). 요금이 없다. 학교 서버(`agent.kau.ac.kr`)에 HTTPS 로 붙으므로 인터넷이 필요하다.
- 서버는 수업 쪽이 운영한다. 우리가 띄우거나 끄지 않는다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| Claude · GPT 등 유료 API | 요금이 든다 |

잴 것: 명령 10~20개로 π0.5 에게 맞는 지시를 주는 비율, 물체를 놓쳤을 때 다시 찾아가는 비율, 응답 시간.

### 작은 계획·행동 (VLA) — π0.5 (잠정)

- LLM 의 지시와 카메라 영상을 받아, 할 일을 스스로 잘게 나눠 로봇을 움직이고 눈앞의 실패는 스스로 복구한다. 2025 BEHAVIOR 대회 1~3위 팀이 모두 π0.5 를 썼고, 공개된 모델과 코드가 많다.
- 학습(LoRA)은 학교 4090(24GB)에서 한다 (22.5GB 이상 필요, 4090 에 겨우 들어감). 시뮬 평가·추론은 시뮬 작업 PC(RTX 5070 Ti 16GB)에서 잘 돈다.

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

### 시뮬레이션 — 2026 BEHAVIOR Challenge 벤치마크

- 참고하는 코드·체크포인트는 2025 대회 상위 팀 것이다(서브모듈 `docs/2025상위팀_깃허브.md`).
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

- 리모 사양·SLAM 데모: [AgileX LIMO 사양](https://www.wevolver.com/specs/agilex-limo), [LIMO ROS2 매핑·내비게이션](https://www.hackster.io/agilexrobotics/ros2-mapping-and-navigation-with-limo-ros2-1936a9), [Trossen LIMO 데모](https://docs.trossenrobotics.com/agilex_limo_docs/demos.html), [LIMO Pro 공식](https://global.agilex.ai/products/limo-pro)
- 물체 인식: [CLIP 후보](clip_candidates.md)(SigLIP 2·FastSAM-s 측정), [SAM 계열 속도·정확도 비교 (2025)](https://scitepress.org/PublishedPapers/2025/137785), [MobileSAM](https://docs.ultralytics.com/ko/models/mobile-sam), [YOLOv8 TensorRT Jetson Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano)
- LLM: AI agent 수업 KAU API (`https://agent.kau.ac.kr/v1`, `qwen3.5-9b`), 사용법은 서브모듈 `docs/에이전트_설계.md`
- VLA 크기·메모리: openpi README (`refs/code/openpi/README.md`), [VLA 비교 연구 (arXiv 2603.19233)](https://arxiv.org/pdf/2603.19233)
- NPU: [DEEPX DX-M1 사양 (DFRobot)](https://wiki.dfrobot.com/SKU_DFR1252_DX-M1%20AI%20Accelerator), [dx-all-suite (DEEPX SDK)](https://github.com/juyoung020/dx-all-suite)
- 시뮬레이션: [2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html), 참고 [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/archive/2025/call_for_participation.html)
---

<details>
<summary>변경 기록</summary>

| 날짜 | 바뀐 것 | 이유 |
|---|---|---|
| 2026-09-30 | 문서 처음 작성: SLAM Cartographer, 물체 인식 SAM + CLIP → **YOLO-seg nano**, DA·지도 갱신 직접 만듦, Spark-DSG, LLM Qwen3.5-9B, VLA π0.5, 시뮬 BEHAVIOR, 앱 네이티브 | 리모(Jetson Nano 4GB)에서 가볍게, 영역과 이름을 한 번에 |
| 2026-09-30 | π0.5 실행 위치: **리모**, 안 되면 라즈베리파이 5 + DEEPX DX-M1 | 로봇이 서버 없이 스스로 움직이게 |
| 2026-10-02 | 보기: 3D → **2D 지도**(웹 3D 뷰어 안 씀). 리모 **프로**를 받을 수도 있음 | |
| 2026-10-03 | LLM: **AI agent 수업(최영식 교수님) KAU API**(Qwen3.5-9B, 인터넷 필요). 학교 4090 은 π0.5 LoRA 학습만 | 수업이 서버를 운영 |
| 2026-10-03 | 물체 인식: YOLO-seg → 다시 **SAM + CLIP 구조**(FastSAM-s 416 + SigLIP 2 B/32). 물체 벡터 = 원본 임베딩, 이름 = 기억 폴더 `cache/` | 물체 찾기를 임베딩 벡터 찾기로 정해서 물체마다 CLIP 류 임베딩이 필요 |
| 2026-10-03 | 물체 찾기 절 추가: **임베딩 벡터 찾기 + 이름(의미) 찾기** 둘 다, RAG 구조 | |
| 2026-10-03 | 시뮬레이션: **2026 BEHAVIOR Challenge**(참고 코드는 2025 상위 팀). 장비: 시뮬 평가·π0.5 추론 = `jy-desktop`(RTX 5070 Ti 16GB), LoRA 학습 = 학교 4090 | 16GB 에 LoRA 학습이 안 들어감 |

</details>
