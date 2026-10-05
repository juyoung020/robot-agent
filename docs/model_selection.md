# 모델 선택

로봇의 각 부품에 **무엇을 쓰는지, 왜 그것인지** 정리한 문서다. 본문은 지금 결정만 적고, 바뀐 과정은 맨 아래 [변경 기록](#변경-기록)에 둔다.
할 일과 전체 구조는 [계획](plan.md), 저장소 구조는 [LAYOUT](LAYOUT.md), 참고 논문·코드는 [참고 자료](../refs/README.md).

## 한눈에 보기

| 부품 | 선택 | 한 줄 이유 | 상태 |
|---|---|---|---|
| 위치 추정 (SLAM) | **Cartographer (2D 라이다)** | 리모 공식 데모 쪽이고 가볍다. 물체 높이는 깊이로 앎 | 결정 (09-30). 아직 우리 파이프라인에 안 붙음 — 지금 시뮬·bag 은 임시로 scenemap `slam2d` |
| 물체 분할 | **ObjectSAM** — FastSAM-s 에서 증류한 YOLO26n 학생(물체만, 입력 416) | FastSAM-s 만큼 많이 잡으면서 벽·천장·바닥은 안 자르고, 계산은 약 1/10 | 결정 (10-05), 기본 엔진 |
| 물체 이름·임베딩 | **SigLIP 2 B/32** (우리 C++/TensorRT 포팅) — 공간 하나 | 실제 로봇·학습·찾기가 같은 벡터 | 결정. 학습·이름표를 이 공간으로 옮기는 중 |
| 물체 지도 기억 | **scenemap 확률 모드 (`objprob`)** — 직접 만듦 | 조각을 이름 없이 3D·벡터로 합치고, 이름·위치를 확률로 | 결정, 기본 켬 |
| 물체 찾기 | 이름 찾기 + 임베딩 찾기, **에이전트·RecallVLA 공용 색인** | 이름이 틀려도 생김새로 찾음 | 결정, 에이전트 도구 구현 |
| 방 나누기·이름 | 직접 만듦 — 점유 격자로 방, 방 안 물체 이름으로 종류 | 검출기 클래스 없이도 방 종류 | 결정 |
| 지도 저장·보기 | Spark-DSG 형식 + **sgview**(우리가 고친 2D 뷰어) | 실시간·학습 리플레이가 같은 뷰어 | 결정 |
| 큰 계획·대화 (LLM) | Qwen3.5-9B, AI agent 수업 KAU API | 요금 없음 (인터넷 필요) | 결정 |
| 작은 계획·행동 (VLA) | **RecallVLA** — Qwen3.5-0.8B 전체 학습 + SigLIP 2, 단계 문장 + flow matching 행동 | 기억에서 물체를 스스로 찾아가고, 없으면 탐색 | 결정. 신경망 완료, 학습은 교사 준비 뒤 |
| 학습 중 인지 | **검출 흉내(인지 오차 모델) + GPU 로 옮긴 확률 모드** | 학습 속도를 지키면서 실제 인지와 같은 입력 | 결정 (10-06), 구현 중 |
| 시뮬·학습 환경 | BEHAVIOR-1K 집 장면(OmniGibson) + 우리 GPU 일괄 환경 | 대량 학습은 GPU, 확인·마지막 미세 조정은 OmniGibson | 결정 |
| VLA 실행 위치 | 리모, 안 되면 라즈베리파이 5 + DEEPX DX-M1 | 서버 없이 스스로 움직이게 | 결정, 아직 안 잼 |
| 앱 | iOS (SwiftUI) · Android (Compose) | 네이티브, 카카오톡식 채팅 | 결정 |

## 고를 때 따지는 것

| 조건 | 영향 |
|---|---|
| **로봇 코드는 전부 리모에서 돈다** (기본형 Jetson Nano 4 GB, 프로면 Orin Nano 8 GB) | 분할·SigLIP 2·지도·VLA 가 한 기기에 같이 올라가야 한다 → 가벼운 것 우선 |
| LLM 은 수업 KAU API | 로봇 쪽 모델은 외부 GPU 에 기대지 않는다 |
| 개발·학습 PC `jy-desktop` (RTX 5070 Ti 16 GB) | 시뮬, GPU 학습 환경, RecallVLA 학습이 여기서 돈다 |
| 학습과 실제가 같은 입력 | 정책이 받는 입력 형식은 하나(`VLA_INPUT.md`). 실제 로봇은 scenemap, 학습은 GPU 근사가 같은 형식을 낸다 |

새로 고르거나 바꿀 때: 후보를 2–3 개로 줄이고, **같은 데이터**(같은 시뮬 기록, OpenLORIS 실제 bag)로 아래 "잴 것"을 숫자로 재서, 가장 중요한 조건을 통과한 것 중 가장 가벼운 것을 고른다. 그리고 위 표와 변경 기록을 고친다.

---

## 부품별 자세히

### 위치 추정 (SLAM) — Cartographer (2D 라이다)

- **결정**: Cartographer(2D 라이다)로 방 지도를 만들고, 평소에는 그 위에서 위치만 추정한다. 리모 공식 데모 쪽이고 Jetson 에서 가볍다.
- 물체 위치는 `로봇 자세 + 카메라 장착 위치 + 마스크의 깊이 점`으로 계산한다. 그래서 SLAM 은 2D 로 충분하다.
- **지금 상태**: Cartographer 는 아직 우리 파이프라인에 붙지 않았다. 시뮬과 공개 bag 시험은 그동안 scenemap 안에 만든 임시 2D SLAM `slam2d`(바퀴 오도메트리 + 깊이 카메라 스캔 맞추기)로 돌았다. 이건 결정이 아니라 시뮬을 먼저 돌리려고 들어간 것이다 — Cartographer 로 바꿔야 한다(scenemap 은 외부 자세를 받을 수 있다: `SGRT_POSE`).

잴 것: 리모 CPU·메모리, 한 바퀴 돌고 왔을 때 어긋난 거리(ATE).

### 물체 분할 — ObjectSAM

- **무엇**: 클래스 없이 물체만 자르는 분할 모델. FastSAM-s 마스크로 자기 증류하고, 벽·천장·바닥은 빼고, 정답이 있으면 통째 마스크로 바꾼 라벨로 YOLO26n-seg 를 학습했다. 과소분할(의자 + 식탁) 벌점을 넣었다. 출력 모양이 FastSAM-s 와 같아 그대로 갈아 끼운다.
- **엔진**: `yolo26n-seg-obj-416` (TensorRT FP16, CUDA 그래프 한 번 호출 약 0.62 ms, 이 PC). 공개: [github.com/juyoung020/ObjectSAM](https://github.com/juyoung020/ObjectSAM) v1.0, AGPL-3.0(원 가중치가 Ultralytics AGPL).
- **결과**(처음 보는 시뮬 집 10 채, 원래 FastSAM-s 대비): 재현율 0.793 → 0.803, 벽·천장·바닥 가짜 마스크 4.44 → 1.55/장, 물체당 마스크 2.77 → 2.51. 재현율은 크기·문창계단·처음 보는 종류 어디에서도 뚜렷이 떨어지지 않음. 자세히 `training/fastsam/README.md`.

잴 것: 리모(Jetson) 위 지연·메모리 — 아직 안 잼. Orin 이면 INT8 판도 잰다.

### 물체 이름·임베딩 — SigLIP 2 B/32

- 물체 조각마다 **영상 벡터 768-d 하나**(마스크 풀링: 주의집중 logit 에 마스크 비율을 더해 물체 부분만). 이름은 SigLIP 2 글 벡터로 만든 이름표와 비교한 확률. 글 인코더도 TensorRT 로 돈다. 코드 `src/scene_graph/clip`.
- **공간은 하나다**: 실제 로봇의 이름 붙이기, 물체 찾기, 학습 입력(이름·생김새·지시 벡터)이 모두 SigLIP 2 공간을 쓴다. 학습 쪽 크기를 줄일 때는 저장소에 둔 고정 변환(768 → 128) 하나를 양쪽이 같이 쓴다.
- **한국어 지시**: SigLIP 2 글 인코더에 그대로 넣어 품질을 재고, 모자라면 한국어 → 이름 짝 표를 같이 쓴다.

### 물체 지도 기억 — scenemap 확률 모드 (`objprob`)

- **벽·천장·바닥**은 **기하로** 거른다(평면 맞추기 RANSAC, 지도 벽선과 겹치는 수직면, 천장·바닥 높이 수평면). 문·창은 크기 규칙으로 남긴다.
- **같은 물체 판단(DA)**: 이름을 쓰지 않는 베이지안 가설 검정 — 3D 맞닿음·겹침(가우시안)과 벡터 일치(vMF). 같은 물체 판정 가중치는 엔진별 매개변수 파일(`objprob_params/<엔진>.json`).
- **물체마다**: 벡터 = vMF 사후(r ← r + κ·z), 이름 = 베이지안 범주 사후(낮으면 상위어), 위치·크기 = 칼만. 덜 본 정도(가장 가까이 본 거리·본 횟수·윗면 본 비율)도 둔다.
- **물체 사이 관계(위·안·옆)는 저장하지 않는다** — 위치·크기 숫자로 충분하다.
- **학습과 같은 계산식**: 판정·합치기 계산식은 공용 헤더 하나로 두고 scenemap 과 GPU 학습 지도가 같이 쓴다(만드는 중).

잴 것: 정답 물체 찾음, 중복, 잘못 합침, 벽·천장 위 가짜, 이름 정답률 — `src/scene_graph/tools/realbag`(`realbag_run`·`objprob_eval`).

### 물체 찾기 — 이름 + 임베딩, 공용 색인

- 색인 하나(`src/scene_graph/clip/include/sgsearch.h`)를 에이전트와 RecallVLA 가 같이 쓴다.
- **에이전트**는 벡터를 못 받으므로 도구 안에서 찾고 글만 돌려준다: ① 이름·동의어·상위어 → ② 없거나 약하면 이름을 무시한 생김새 재검색 → ③ 사람 확인·가까이 본 결과로 이름 고치기(`search_objects`·`confirm_object`·`list_place`).
- **RecallVLA** 는 자기 질의 벡터로 같은 색인에서 상위 후보를 기억 토큰에 넣는다.
- 측정(이름이 틀린 물체 R@5): 이름만 0.00 → 이름 + 생김새 0.33.

### 방 나누기·이름 — 직접 만듦

- 점유 격자의 빈칸을 거리 변환으로 방으로 나누고(Hydra room finder 의 2D 판), 지도가 바뀌어도 방 id 를 유지한다.
- 방 종류는 방 안 물체의 SigLIP 2 이름으로 규칙 점수(냉장고 → kitchen …). 다음 단계는 임베딩 제로샷 분류("a kitchen" 글 벡터와 비교). LLM 이 정한 이름은 덮어쓴다.

잴 것: 시뮬 정답 방 종류와 맞는 비율, 방 이름이 흔들리는 횟수.

### 지도 갱신 — 직접 만듦

- 옮겨짐·사라짐·생김을 합친 물체 단위로 판정하고 바뀐 부분만 고친다. 참고: DovSG, Khronos.

잴 것: 바뀜 판정 P/R/F1, 갱신 속도.

### 지도 저장·보기 — Spark-DSG + sgview

- 기억 폴더: `scene.json`(Spark-DSG), `objects/`(물체마다 RGB·깊이·마스크·점구름·임베딩 = 원본), `cache/`(이름·색인 = 다시 만들 수 있음).
- 보기는 **sgview**(MIT Spark-DSG 를 고친 2D 지도 뷰어, Rust 서버 + three.js, `src/scene_graph/sgview`). 로봇이 본 곳만 그린다(모르는 칸은 투명). 실시간 로봇·기록 재생·**학습 리플레이**가 같은 뷰어를 쓴다 — 학습 뷰어는 재생 제어와 학습용 패널만 바깥에 더한다.

### 큰 계획·대화 (LLM) — Qwen3.5-9B

- AI agent 수업(최영식 교수님)이 운영하는 KAU API(`https://agent.kau.ac.kr/v1`, `qwen3.5-9b`). 요금 없음, 인터넷 필요.
- 에이전트 = 스킬(지시문) + 도구(코드). `src/agent/`.

### 작은 계획·행동 (VLA) — RecallVLA

- **입력**: RGB 3 장(리모 카메라, 손목 카메라, 위에서 본 지도 그림 — 같은 SigLIP 2), 몸 상태, 목표 칸 2(물체 또는 지도 지점), 물체 기억 전체(≤ 256 → 기억 요약 32 + 정밀 칸 8, "기억에 없음" 포함), 방 항목, 방향 구역 8, 지시 문장. 깊이는 그림으로 넣지 않는다. 정의 [VLA_INPUT](map_vla/VLA_INPUT.md).
- **신경망**: Qwen3.5-0.8B 전체 학습 + SigLIP 2 학습, 행동 전문가(flow matching, 행동 묶음), 다음 단계 문장. 약 0.98 B. 사양 [MAPVLA_SPEC](map_vla/MAPVLA_SPEC.md), 코드 `training/vla`.
- **학습**: 강화학습·대본 교사 → BC·DAgger → RecallVLA. 원칙: 정답 지도로 학습하지 않는다(빈 지도에서 자라는 지도), 교사도 학생과 같은 입력(특권은 critic 만), 명령에는 탐색으로 찾을 수 있는 물체만. 커리큘럼 [CURRICULUM_BEHAVIOR2026](map_vla/CURRICULUM_BEHAVIOR2026.md) 5.7.

잴 것: 탐색해서 찾기·집기·놓기 성공률(학습 집 / 처음 보는 집), 리모 위 지연.

### 학습 중 인지 — 검출 흉내 + GPU 확률 모드

- GPU 학습 환경은 수천 판을 동시에 돌려서 실제 렌더 + ObjectSAM + SigLIP 2 를 매번 돌릴 수 없다. 그래서 **검출 단계만 흉내 내고**(정답 물체 + 카메라 자세 → 진짜 파이프라인이 냈을 검출 목록: 놓침·가짜·붙음·이름 헷갈림·임베딩·위치 오차, 놓침은 키프레임을 넘어 이어짐), 그 뒤 합치기는 **진짜와 같은 확률 모드 규칙을 GPU 로** 돌린다.
- 흉내 값은 OmniGibson 에서 진짜 파이프라인을 돌린 결과로 맞추고, 파이프라인이 바뀌면 맞추는 스크립트만 다시 돌린다. 통계형으로 시작하고, 검증을 못 넘으면 학습형으로 올린다.
- 자율주행의 인지 오차 모델(PEM)과 같은 틀이다 — 조사 [PERCEPTION_EMULATION_SURVEY](map_vla/PERCEPTION_EMULATION_SURVEY.md), 설계 [GPU_MAP_PORT](map_vla/GPU_MAP_PORT.md).
- 진짜 파이프라인은 학생 DAgger 일부와 마지막 미세 조정에 쓴다.

### 시뮬·학습 환경 — BEHAVIOR-1K 장면 + GPU 환경

- 집 장면·물체·과제 인스턴스는 BEHAVIOR-1K(`third_party/BEHAVIOR-1K`), 렌더·물리 확인은 OmniGibson. 대량 학습은 GPU 일괄 환경(`training/RL`).
- 대회 점수는 목표가 아니다. 리모 + OMX-F 의 집기·놓기 학습장으로 쓴다.
- 학습 집 4 채(1층 집, 2층 집 아래층, 식당, 사무실) / 평가 집 3 채(Rs_int, 호텔 방, 2층 집 위층). 다음은 절차 생성 집(ProcTHOR) 섞기.

### VLA 실행 위치 — 리모, 안 되면 라즈베리파이 5 + DEEPX DX-M1

| 순서 | 방법 | 확인할 것 |
|---|---|---|
| 1 | 리모 위에서 양자화 | 메모리·지연 |
| 2 | 라즈베리파이 5 + DEEPX DX-M1(보유, 25 TOPS INT8, 4 GB) | DXNN SDK 로 변환되는지 |

### 앱 — iOS (SwiftUI) · Android (Compose)

| 고를 것 | 선택 | 비고 |
|---|---|---|
| iOS / Android | Swift·SwiftUI / Kotlin·Compose | 네이티브 |
| 앱 ↔ 리모 | WebSocket (미정) | 진행 상황 실시간 |
| 학교 밖 접속 | VPN 등 (미정) | 학교 네트워크 규칙 확인 |
| 위에서 본 지도 | VLA 의 셋째 그림과 같은 정의 | 물체를 누르면 id, 바닥을 누르면 지점 목표 |

---

## 검토했던 다른 후보

부품마다 비교했던 후보와 고르지 않은 이유. 다시 고를 때 같은 비교를 반복하지 않으려고 남긴다.

**위치 추정 (SLAM)**

| 후보 | 안 고른 이유 |
|---|---|
| RTAB-Map (3D) | Jetson Nano 에 무거움 |
| slam_toolbox (2D) | 리모 공식 데모는 Cartographer 쪽 |

**물체 분할**

| 후보 | 안 고른 이유 |
|---|---|
| FastSAM-s 그대로 | 벽·천장을 자르고 물체를 잘게 쪼갬(같은 기록에서 노드 290, 중복 140) |
| FastSAM-s 재학습판 | 마스크는 가장 깔끔했지만 계산이 10 배. 리모(특히 Nano)에 무거움 |
| YOLO26s-seg | 깔끔하지만 아는 종류만 잡아 물체 수가 적음(정답 34 중 14) |
| YOLOE-11L | 프롬프트 어휘 안에서만 잡고 무거움 |

**물체 이름·임베딩**

| 후보 | 안 고른 이유 |
|---|---|
| PE-L 글 공간 + 투영 머리 + 한국어 학생 | 실제 로봇과 학습이 다른 공간을 쓰게 되고, 모델이 하나 더 필요 |
| OpenAI CLIP B/32 + 한국어 글 인코더 | SigLIP 2 가 이름·찾기 측정에서 앞섬([clip_candidates.md](clip_candidates.md)) |

**큰 계획·대화 (LLM)**

| 후보 | 안 고른 이유 |
|---|---|
| 유료 API (Claude·GPT) | 요금 |

**작은 계획·행동 (VLA)**

| 후보 | 안 고른 이유 |
|---|---|
| π0.5 | 지도 기억을 넣을 자리가 없고 리모에 무거움. 코드·가중치까지 지움 |
| SmolVLA | 가볍지만 지도 기억·탐색을 따로 붙여야 함 |

**시뮬·학습 환경**

| 후보 | 안 고른 이유 |
|---|---|
| LIBERO | 책상 위 팔 조작만, 이동 없음 |
| Gazebo | VLA 학습용으로 약함 |

---

## 출처

- 리모: [AgileX LIMO 사양](https://www.wevolver.com/specs/agilex-limo), [LIMO ROS2 매핑·내비게이션](https://www.hackster.io/agilexrobotics/ros2-mapping-and-navigation-with-limo-ros2-1936a9), [LIMO Pro](https://global.agilex.ai/products/limo-pro)
- 분할·임베딩: [ObjectSAM](https://github.com/juyoung020/ObjectSAM), [CLIP 후보 측정](clip_candidates.md), FastSAM(Zhao 외 2023), SigLIP 2(Tschannen 외 2025)
- 학습 중 인지: [PERCEPTION_EMULATION_SURVEY](map_vla/PERCEPTION_EMULATION_SURVEY.md)
- NPU: [DEEPX DX-M1](https://wiki.dfrobot.com/SKU_DFR1252_DX-M1%20AI%20Accelerator)
- 장면: [BEHAVIOR](https://behavior.stanford.edu/)

---

## 변경 기록

| 날짜 | 바뀐 것 | 이유 |
|---|---|---|
| 2026-09-30 | 처음 작성: SLAM Cartographer, 분할 YOLO-seg nano, DA·지도 갱신 직접, Spark-DSG, LLM Qwen3.5-9B, VLA π0.5, 시뮬 BEHAVIOR, 앱 네이티브. VLA 실행 위치 리모 → 라즈베리파이 5 + DX-M1 | 리모에서 가볍게 |
| 2026-10-02 | 보기 3D → 2D 지도. 리모 프로를 받을 수도 있음 | |
| 2026-10-03 | LLM = 수업 KAU API. 분할 YOLO-seg → FastSAM-s + SigLIP 2(SAM + CLIP 구조). 물체 찾기(임베딩 + 이름)·방 나누기 절 추가. 시뮬 = 2026 BEHAVIOR Challenge | 물체마다 임베딩이 필요, FastSAM 은 클래스가 없어 방 이름에 SigLIP 2 이름 사용 |
| 2026-10-04 | VLA π0.5 버림 → 우리 VLA. SLAM 은 시뮬에서 자체 `slam2d` | 지도 기억을 넣을 자리, 리모 크기 |
| 2026-10-05 | 분할 비교(FastSAM-s / YOLO26s-seg / YOLOE) 뒤 FastSAM-s + scenemap 확률 모드로 방향, 그다음 **ObjectSAM(YOLO26n 학생)** 으로 결정·기본 엔진. DA 를 이름 없는 확률 DA 로. 찾기를 에이전트·RecallVLA 공용 색인으로. VLA = RecallVLA | FastSAM 조각·벽 문제, Jetson 에 맞는 계산량 |
| 2026-10-06 | 임베딩 공간을 SigLIP 2 하나로(옛 PE-L 투영 경로 버림). 학습 중 인지 = 검출 흉내 + GPU 확률 모드. 판정·합치기 계산식 공용 헤더. 정답 지도 학습 금지·새 커리큘럼. 시뮬은 BEHAVIOR-1K 장면 + GPU 환경(대회 저장소 분리). SLAM 은 Cartographer 그대로(지금 도는 `slam2d` 는 임시, 바꿀 것). sgview 를 학습 리플레이에도 | 학습과 실제가 같은 입력, 저장소 하나 |
