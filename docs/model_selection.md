# 모델 선택

로봇의 각 부품에 **무엇을 골랐고, 왜 골랐는지** 정리한 문서.
할 일과 전체 구조는 [계획](plan.md), 참고 논문·코드는 [참고 자료](../refs/README.md).

> **10-06 갱신**: 저장소는 robot-agent 하나(인지·뷰어 원본 `src/scene_graph`, 옛 시뮬 저장소는 분리), 임베딩은 SigLIP 2 공간 하나로 통일 중, 학습 중 인지는 "검출 흉내 + GPU 확률 모드", VLA 는 RecallVLA. 자세히는 아래 표와 각 절.

## 한눈에 보기

| 부품 | 선택 | 한 줄 이유 | 상태 |
|---|---|---|---|
| 지도·위치 (SLAM) | **Cartographer (2D 라이다)** — 시뮬에서는 실제로 자체 `slam2d`(scenemap)가 돈다 | 리모에서 가볍게 돌고, 물체 높이는 depth 로 알 수 있음 | 시뮬은 `slam2d`, 실기에서 무엇을 쓸지 다시 정해야 함 |
| 물체 인식 | **ObjectSAM(FastSAM-s 에서 증류한 YOLO26n 학생, things 만 — 엔진 `yolo26n-seg-obj-416`, [github.com/juyoung020/ObjectSAM](https://github.com/juyoung020/ObjectSAM) v1.0: `ObjectSAM-416.pt`·`.onnx`·`-int8-qdq.onnx`) + SigLIP 2 B/32 + objprob(scenemap 확률 모드, 기본 켬, 매개변수 `objprob_params/yolo26n-seg-obj-416.json`)** — 까닭: FastSAM-s 계산의 약 1/10 이라 LIMO 의 Jetson(특히 Nano)에 맞다 — 기기 위 시간은 아직 안 잼. 벽·천장 기하 제거, 이름 없는 3D·벡터 병합, vMF 벡터·베이지안 이름 (아래 "물체 인식" 절). YOLO26·YOLOE 는 비교 뒤 보관(`models/ovdet/archive`) | 가장 많이 잡고(목록 밖 포함) 벡터·단어 찾기 둘 다 됨. 목표: FastSAM-s 의 재현율 + YOLO26s-seg 수준의 깔끔함 | **10-05: decided ObjectSAM (YOLO26n 학생) + SigLIP 2 + objprob** — 기본 엔진(libsgrt 글루 `SGRT_ENGINE`·`realbag_run`·explore/LIMO 시작 스크립트, objprob 기본 켬). 원래 FastSAM-s-416 은 `--engine`/`SGRT_ENGINE` 으로 고를 수 있음 |
| 같은 물체 판단 (DA) | **직접 만듦 — 이름 없는 확률 DA (`objprob`)** | 3D 맞닿음(가우시안) + 벡터 일치(vMF)의 가설 검정. 처음엔 같은 이름끼리 위치로 비교했으나 FastSAM 조각이 안 합쳐져 바꿈 | 결정, 구현·기본 켬 (10-05). 판정·합치기 계산식은 scenemap 과 GPU 학습 지도가 공용 헤더 하나를 쓰게 바꾸는 중(10-06) |
| 물체 찾기 | **임베딩 벡터 찾기 + 이름(의미) 찾기** 둘 다, **에이전트·RecallVLA 공용 색인** | 이름 검색 → 생김새 재검색 → 확인 후 이름 고치기. 에이전트엔 글로, VLA 는 자기 질의 벡터로 | 결정 (10-05 갱신). 에이전트 도구 `search_objects`·`confirm_object`·`list_place` 구현(`369cd3a`, `936c256`), RecallVLA 자체 검색은 학습 전 |
| 물체 임베딩·이름 | **SigLIP 2 B/32 하나**(우리 C++/TensorRT 포팅, 영상 마스크 풀링 + 글 인코더) | 실제 로봇·학습·찾기가 같은 공간. 옛 PE-L 글 공간 + 투영 머리 `h` + 한국어 학생 경로는 버림 | **결정 (10-06)**, 바꾸는 중(GPU_MAP_PORT P5) — 바꾼 뒤 이름 품질 확인하고 옛 경로 보관. 한국어 지시: SigLIP 2 글 인코더에 그대로 넣어 재 보고, 모자라면 한국어 → 이름 짝 표 |
| 학습 중 인지 | **검출 흉내(인지 오차 모델, 통계형 → 필요하면 학습형) + GPU 로 옮긴 확률 모드 합치기** | 학습 속도를 지키면서 실제 ObjectSAM + SigLIP 2 + objprob 와 같은 입력. 자율주행 PEM 방식([PERCEPTION_EMULATION_SURVEY.md](map_vla/PERCEPTION_EMULATION_SURVEY.md)) | **결정 (10-06)**. 흉내 값은 OmniGibson + 진짜 파이프라인 결과로 맞춤. 학생 DAgger 일부와 마지막 미세 조정만 진짜 파이프라인 |
| 지도 갱신 | **직접 만듦** | 바뀐 부분만 고침 | 결정 |
| 방 나누기·이름 | **직접 만듦** — 방 안 물체의 SigLIP 2 이름 → 규칙, 다음은 임베딩 제로샷 분류 | 클래스 이름 없는 FastSAM-s 에서도 방 종류를 붙임 | 결정 |
| 물체 지도 저장·보기 | **Spark-DSG** 저장, 보기는 **2D 지도** | Hydra·Khronos 와 같은 형식. 웹 3D 뷰어는 안 씀(10-02) | 결정 |
| 큰 계획·대화 (LLM) | **Qwen3.5-9B**, AI agent 수업이 주는 KAU API | 요금 없이 사용 (인터넷 필요) | 결정 |
| 작은 계획·행동 (VLA) | ~~π0.5~~ → **RecallVLA**(Qwen3.5-0.8B 전체 학습 + SigLIP 2, 단계 문장 + flow matching 행동, 기억 요약 인코더, `training/vla`) | 지도 기억에서 물체를 스스로 찾아가고, 없으면 탐색. 리모 + OMX-F 에 맞춤. π0.5 는 코드·가중치까지 지움(10-06) | 결정. 신경망·융합 커널 완료, 학습은 교사(빈 지도 커리큘럼) 준비 뒤 |
| VLA 실행 위치 (π0.5 때 정함) | **리모**, 안 되면 라즈베리파이 5 + DEEPX DX-M1 (보유) | 로봇이 서버 없이 스스로 움직이게 | 결정 |
| 시뮬레이션 | **BEHAVIOR-1K 집 장면(OmniGibson) + 우리 GPU 일괄 환경** | 대회가 아니라 리모 집기·놓기 학습장. 대량 학습은 GPU 환경, 확인·미세 조정은 OmniGibson | 결정 (10-06 갱신: 대회 저장소는 분리, BEHAVIOR-1K 는 `third_party/`) |
| 앱 | **iOS (SwiftUI) · Android (Compose)** | 네이티브, 카카오톡식 채팅 | 결정 |

## 고를 때 따지는 것

모든 부품에 공통으로 걸리는 조건이다.

| 조건 | 영향 |
|---|---|
| **우리 코드는 전부 리모에서 돈다** (Jetson Nano 4GB) | 모든 모델이 4GB 안에 같이 올라가야 한다 → 가벼운 것 우선 |
| LLM 은 AI agent 수업(최영식 교수님)이 주는 Qwen API, 학교 4090 (24GB) 은 π0.5 LoRA 학습만(π0.5 버림, 10-04) | 로봇이 쓰는 모델은 4090 에 기대지 않는다 |
| 시뮬 작업 PC `jy-desktop` (RTX 5070 Ti 16GB) | 시뮬 평가와 우리 VLA 학습(RL 교사·BC 학생)이 여기서 돈다. (π0.5 때: LoRA 학습은 16GB 에 안 들어가서 학교 4090 으로 — 대체됨) |
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

> **10-04 메모**: 저장소에 Cartographer 는 없다. 시뮬에서는 scenemap 의 자체 2D SLAM(`src/scene_graph/scenemap/src/slam2d.cpp`)이 돈다. 실기에서 Cartographer 를 쓸지 `slam2d` 를 쓸지는 아직 안 정했다.

- 리모로 방을 한 번 돌며 2D 지도를 만들고, 평소엔 그 지도 위에서 **위치만** 추정한다.
- 물체 위치(xyz)는 SLAM 이 아니라 `로봇 위치 + 카메라 장착 위치 + 물체 마스크(지금은 ObjectSAM) 무게중심의 depth` 로 계산해 기록한다. 그래서 무거운 3D SLAM 은 필요 없다.
- 시뮬레이션은 정답 위치를 주므로, 물체 기억·계획 파트는 SLAM 이 끝나기를 기다리지 않아도 된다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| RTAB-Map (3D) | Jetson Nano 에 무거움 |
| slam_toolbox (2D) | 리모 공식 데모는 Cartographer 쪽 |

잴 것: 리모 CPU·메모리 사용량, 방을 한 바퀴 돌고 출발점에 왔을 때 어긋난 거리.

### 물체 인식 — ObjectSAM + SigLIP 2 + objprob (10-05 결정)

> **10-05: decided ObjectSAM (YOLO26n 학생) + SigLIP 2 + objprob.** 분할 = ObjectSAM(FastSAM-s 에서 증류한 YOLO26n 학생, things 만 — 엔진 `yolo26n-seg-obj-416`, [github.com/juyoung020/ObjectSAM](https://github.com/juyoung020/ObjectSAM) v1.0: `ObjectSAM-416.pt`·`.onnx`·`-int8-qdq.onnx`) + SigLIP 2 B/32 + objprob(scenemap 확률 모드, 기본 켬, 매개변수 `objprob_params/yolo26n-seg-obj-416.json`). 까닭: FastSAM-s 계산의 약 1/10 이라 LIMO 의 Jetson(특히 Nano)에 맞다 — 기기 위 시간은 아직 안 잼. FastSAM-s 재학습(`FastSAM-s-416-obj`)은 버리고 `models/ovdet/archive/x86_sm120/` 에 보관했다.
>
> **(기록) 갱신 (2026-10-05 저녁): 방향 결정 — FastSAM-s-416 + SigLIP 2 + scenemap 확률 모드(`objprob`).** 같은 BEHAVIOR 기록(house_double_floor_lower, LIMO r3)에서 셋을 비교(`~/datasets/sim_detcmp/README.md`, 정답 34개, slam 자세): FastSAM-s-416 노드 290·정답 찾음 27·중복 140·벽 위 가짜 108, YOLO26s-seg 28·14·9·4, YOLOE-11L 78·24·32·19. 사용자 관찰: "FastSAM-s 는 천장·벽이 안 걸러지고 병합이 잘 안 됨, YOLO26s-seg 는 다 잘 되는데 뽑은 수가 아쉬움, YOLOE 는 FastSAM-s 보다 벽·천장이 덜함". 목표는 **FastSAM-s 를 YOLO26s-seg 수준으로 깔끔하게 + 물체를 더 잘 잡게**. YOLO26s-seg·YOLOE 엔진은 보관했다.
>
> **(아래는 비교 전 기록)** 상태 (2026-10-05 낮): 미결정. 아래 "FastSAM-s + SigLIP 2" 는 09-30 이후의 안이었지만, 실제 로봇 데이터(OpenLORIS-Scene)에서 FastSAM 이 물체를 잘게 쪼개 중복이 많았고(물체 177 개, 0.5 m 안 같은 이름 쌍 72 개), 방 이름 규칙에는 일정한 물체 어휘가 필요해서 다시 고르는 중이다. 셋 중 하나로 **통일**해야 한다(지금은 시뮬 탐사 = YOLOE, LIMO 지도 시험 = YOLO26s-seg, 벤치마크·실제 bag = FastSAM + SigLIP 2 로 제각각).
>
> | 후보 | 분할 | 이름 | 물체 벡터 | 목록 밖 물체 |
> |---|---|---|---|---|
> | FastSAM-s + SigLIP 2 (09-30 안) | 이름 없는 조각 (잘게 쪼개지기 쉬움) | SigLIP 이 라벨 표 약 2.9 만 개에서 | SigLIP | 잡음 |
> | YOLO26s-seg + SigLIP 2 | YOLO 물체 단위 (깔끔) | YOLO 이름 + SigLIP 이 더 자세히 | SigLIP | 못 잡음 (기본 가중치 어휘 밖, COCO 80 종인지 확인 필요) |
> | YOLOE + SigLIP 2 | 열린 어휘 분할 | 프롬프트 어휘 안에서 (어휘는 우리가 정함) | SigLIP | 프롬프트 어휘 안에서만 |
>
> 정하는 법(제안): 같은 데이터(실제 bag OpenLORIS·TUM, dynamic-object-mapping 벤치마크, BEHAVIOR 시뮬)에서 셋을 나란히 — 기억에 들어간 물체 수(정답 대비)·중복·이름 정답률·찾기 R@1·책장 같은 구조물과 벽 구분·변화 탐지 F1·방 이름 정확도·리모 지연·메모리. 이름 어휘(방 이름 규칙에 쓰는 물체 어휘)는 고른 쪽에 맞춰 하나로 통일한다.

#### scenemap 확률 모드(`objprob`) 첫 구현 결과 (2026-10-05) — 아직 합격 아님

구현은 behavior-2026 `ca85b1a`(scenemap 확률 모드, 기본 꺼짐)·`3b37c0a`(정책 벽 방향)에 있다. 자세한 설명은 scenemap README "scenemap 확률 모드" 절에 있다.

**비교 조건**
- 같은 radio r3 기록, slam 자세, 정답 34 개다.
- 채점은 `tools/realbag/objprob_eval.py`(detcmp_eval 의 표 + 잘못 합침·문창계단·글 질의)로 했다.

| | 확률 모드 (FastSAM-s-416) | 옛 규칙 FastSAM-s-416 | 옛 규칙 YOLO26s-seg | 옛 규칙 YOLOE-11L |
|---|---|---|---|---|
| 살아 있는 노드 | 122 | 290 | 28 | 78 |
| 정답 34 중 찾음 | **28** | 27 | 14 | 24 |
| 중복 | 59 | 140 | **9** | 32 |
| 잘못 합침(다른 정답이 한 노드 — 작은 것+가구 / 같은 종류 이웃) | 11 (1 / 5) | 18 (2 / 3) | 5 (0 / 3) | 8 (0 / 3) |
| 벽·천장·바닥 위 헛노드 | 6 | 25 | 0 | 9 |
| 문 6·창 2·계단 1 을 맞는 이름으로 | 2·2·1 | 0 | 0 | 0 |
| 이름 정확도 노드 / 정답 | 0.45 / 0.88 | 0.21 / 0.56 | 0.68 / 0.92 | 0.52 / 0.86 |
| 글 질의 R@1(물체 벡터 μ) | 0.40 | – (벡터 없음, 이름 같음 0.20) | – (0.15) | – (0.30) |
| 검출 keyframe 당 GPU | 0.9 + 4.6 + 0.4 ms(검출 + SigLIP 조각 + 통째 다시 담기), 662 MiB | 5.7 ms, 662 MiB | 1.8 ms, 640 MiB | 6.1 ms, 890 MiB |

**gt 자세**
- 확률 모드는 찾음 27·중복 45·벽 장식 8/8 이다.
- 같은 검출 캐시인데도 slam 판과 차이가 난다. 원인은 탐욕 병합과 벽 선분의 차이로 보인다.

**OpenLORIS(확률 모드 / 옛 규칙, 둘 다 FastSAM-s-416)**
- office1-1 slam: 노드 120 / 293, 같은 이름 0.5 m 안 쌍 45 / 89.
- office1-5 slam: 노드 107 / 274, 같은 이름 0.5 m 안 쌍 12 / 29.
- 여러 판을 이은 gt 자세 판에서 사라짐·옮겨짐 사건 수:

  | 판 | 확률 모드 사라짐 / 옮겨짐 | 옛 규칙 사라짐 / 옮겨짐 | 비고 |
  |---|---|---|---|
  | 1-2 | 3 / 3 | 17 / 12 | 안 바뀐 판 |
  | 1-4 | 3 / 5 | 22 / 16 | 안 바뀐 판 |
  | 1-5 | 24 / 8 | 56 / 38 | 안 바뀐 판 |
  | 1-6 | 8 / 11 | 33 / 41 | 바뀐 판 |

  확률 모드는 노드가 적어서 사건 수도 같이 줄었다는 점을 감안해야 한다.

**평면 맞춤 PCA → 랜색 교체 (2026-10-05, behavior-2026 `3b5543f`)** — 왜·무엇은 [terms.md "평면 맞추기: PCA 와 랜색"](terms.md#평면-맞추기-pca-와-랜색). 같은 검출 캐시(FastSAM-s-416), PCA → 랜색:

| | slam 자세 | gt 자세 |
|---|---|---|
| 정답 34 중 찾음 | 28 → 28 | 27 → 27 |
| 벽·천장·바닥 위 헛노드 | 5·1·0 → 4·1·1 | 4·2·0 → 4·2·1 |
| 문·창·계단 위 헛노드 | 9·4·5 → 10·3·4 | 10·3·5 → 9·4·5 |
| 문 6·창 2·계단 1 찾음(맞는 이름) | 2·2·1 → 2·2·1 | 1·2·1 → 1·2·1 |
| 중복 / 잘못 합침 | 59 / 11 → 62 / 9 | 45 / 9 → 53 / 9 |
| keyframe 당 scenemap CPU | 약 30 ms → 약 30 ms(잡음 안) | |

OpenLORIS office1-1·1-5 노드 120 → 120, 107 → 111. 솔직히 랜색은 벽·천장·바닥 헛노드를 줄이지 못했다 — 남은 것은 평면 맞춤 실패가 아니라
벽 조각이 SigLIP 에서 door·window·pillar 이름을 받아 문·창 보호로 살아남는 것(크기 확인으로 고치는 중).

**YOLO26s-seg 보다 못한 곳**
- 중복: 소파·hall tree·커피 탁자 위 것들, 같은 이름 조각.
- 잘못 합침: 식탁 의자 묶음, 주방 줄.
- 노드 이름 정확도.
- 문 조각에 붙는 물체 이름(curtain·bag).

**다음**
- ~~FastSAM 재학습 엔진(`FastSAM-s-416-obj.plan`)으로 다시 돈다~~ → 10-05: ObjectSAM(YOLO26n 학생, `yolo26n-seg-obj-416`)이 기본 엔진(`realbag_run` 은 `--engine` 없이).
- 병합·식탁 둘레 잘못 합침을 줄인다.
- sgrt·LIMO 지도 시험을 확률 모드로 옮김 — 10-05 끝(behavior-2026 `26cbdc4`, libsgrt objprob 앞단: 아래 셋 모두).
  - sgrt 에서 조각 임베딩을 scenemap 에 넘기기.
  - 통째 다시 담기를 sgrt_clip 에서 돌리기.
  - O<id>_emb.f16 를 쓰는 곳을 하나로(scenemap 저장 ↔ sgrt_clip).

#### (09-30 안) FastSAM-s (416) + SigLIP 2 B/32, SAM + CLIP 구조

흐름: 처음엔 SAM + CLIP → 09-30 에 YOLO-seg 하나로 바꿈 → **물체 찾기를 임베딩 벡터 찾기로 정하면서 다시 SAM + CLIP 구조**(분할 = FastSAM-s, 임베딩 = SigLIP 2). YOLO 는 이름만 주고 물체 벡터를 주지 않아서, 벡터로 찾으려면 CLIP 류 인코더가 따로 있어야 한다.

- **FastSAM-s**(입력 416)가 keyframe 에서만 이름 없는 물체 마스크를 낸다. 클래스가 없어서 목록 밖 물건도 빠짐없이 잡는다.
- **SigLIP 2 B/32** 가 물체 조각(원본 RGB 에서 상자 + 10 % 둘레, 정사각)마다 768-d 영상 임베딩을 낸다. 새 물체 / best view 가 바뀐 물체만, 묶어서, 비동기로 돈다.
- 이름은 그 임베딩과 미리 계산한 **라벨 표**(글 임베딩)의 코사인으로 고른다. 확신이 낮으면 WordNet 상위어로 올린다. 벽·바닥 같은 구조물은 표시만 하고 agent 목록에서 뺀다.
- 저장: 물체 벡터는 **원본 임베딩 그대로**(`objects/O<id>_emb.f16`), 이름은 다시 만들 수 있는 **캐시**(기억 폴더 `cache/names.json`).
- 코드는 서브모듈 `src/scene_graph/clip`. (10-05 갱신: 시뮬 검출기도 ObjectSAM + SigLIP 2 + objprob — YOLOE 는 보관.) 후보·측정: [CLIP 후보](clip_candidates.md), [물체 인식 모델 후보](perception_model_candidates.md).

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
| YOLOE (10-05 까지 시뮬에서 돌던 것 — 지금은 보관, 기본은 ObjectSAM) | 단어로 찾을 수 있지만 프롬프트 어휘 안에서만. 물체 벡터가 없어 생김새로 못 찾음 |
| OpenAI CLIP B/32 + 한국어 글 인코더 | 바로 쓸 수 있는 2순위 대안. 한국어 찾기는 좋지만 이름 정답률이 낮음(0.28) |
| MobileCLIP 계열 | 이름 정확도는 가장 높지만 가중치가 비상업 라이선스 |
| MobileSAM · EfficientViT-SAM · SAM 2 | 리모에 무거움. 정밀한 영역은 VLA 에 쓰이지 않음 |

잴 것: 리모에서 keyframe 지연·메모리, 이름 정답률, 글 → 물체 찾기 R@1(영어·한국어).

**후보 B (비교용으로 적어 둠): YOLOE 그대로 + SigLIP 2 추가**

- 검출·이름은 지금처럼 YOLOE(`ovdet`), 물체 벡터만 SigLIP 2 로 더한다. `clip` 은 검출기가 준 상자·마스크로 자르므로 YOLOE 와도 그대로 돈다(시뮬은 `SGRT_CLIP=1` 만 켜면 됨). 그 위에 `training/embed` 머리(128-d 투영·도메인 보정·한국어 질의 학생)를 얹는다.
- 좋은 점: 방 이름 규칙이 YOLOE 이름으로 바로 돈다. 바꿀 것이 가장 적다. 이름(YOLOE)과 자유 글·한국어 찾기(벡터)를 둘 다 얻는다.
- 나쁜 점: YOLOE 가 못 잡은 물체(프롬프트 목록 밖, 라디오처럼 놓치는 것)는 마스크가 없어 기억에 아예 안 들어간다. 리모(Nano 4GB)에서 YOLOE + SigLIP 2 는 무겁다(작은 YOLOE 로 다시 잴 것). YOLOE 이름과 SigLIP 2 이름이 다를 때 고르는 규칙이 필요하다.
- 정하는 법: 같은 장면에서 FastSAM-s + SigLIP 2 와 나란히 — 기억에 들어간 물체 수(정답 대비), 이름 정답률, 찾기 R@1, 리모 지연·메모리.

<details>
<summary>처음 결정(09-30): YOLO-seg nano 를 골랐던 근거</summary>

- 영역과 이름을 한 번에 줘서 모델 하나로 끝나고, nano + TensorRT FP16 이면 Jetson Nano 에서 실시간에 가깝다(검출만 약 19 FPS).
- 물체를 집는 건 VLA 라 정밀한 영역이 필요 없고, 위치는 영역 안 depth 의 중앙값이라 대충 맞는 영역이면 충분하다(이 근거는 지금도 맞다).
- 그때는 "아무 단어로 찾기는 필요 없다"고 봤다 → 임베딩 벡터 찾기를 택하면서 SAM + CLIP 구조로 돌아왔다.

</details>

### 같은 물체 판단 (DA) — 직접 만듦

> **갱신 (2026-10-05): 이름 없는 확률 DA 로 바꾸는 중 (`objprob`).** 같은 이름 조건 때문에 FastSAM 조각(같은 물체인데 조각마다 이름이 다름)이 합쳐지지 않았다(BEHAVIOR 비교에서 소파 22 노드, 냉장고 17 노드). 확률 모드 설계:
> - **벽·천장·바닥은 이름이 아니라 기하로 거른다**: 지도 벽선과 겹치는 큰 수직 평면, 천장·바닥 높이의 수평 평면.
> - **합치기·DA = 베이지안 가설 검정**: 같은 물체 대 다른 물체의 우도비. 근거는 3D 맞닿음·겹침(가우시안)과 벡터 일치(vMF), 이름은 쓰지 않거나 약하게만 쓴다. 바뀜(사라짐·옮겨짐)도 합친 물체 단위로 판정한다.
> - **물체 벡터 = vMF 사후**: r ← r + κᵢ·zᵢ, 대표 μ = r/‖r‖, 확신 ‖r‖. κ(시점 품질)는 시뮬 정답으로 맞춘다. 합친 뒤에는 물체 전체를 다시 잘라 인코딩하고, 상위 5 개 시점 벡터를 함께 저장한다.
> - **이름 = 베이지안 범주 사후**: SigLIP 점수(시점 투표) + 이름별 크기 가우시안 (+ 방 사전확률). 문턱 아래면 상위어로 올린다. 이름은 캐시이고 원본은 벡터다.
> - **위치·크기 = 가우시안 + 칼만**: 불확실성(‖r‖, 이름 엔트로피, 위치 공분산)은 RecallVLA 지도 토큰의 불확실성 칸으로 이어진다.
>
> 아래는 처음 설계.

- 새로 본 물체가 이미 지도에 있는 물체인지 판단한다. **같은 이름끼리 위치로 비교**한다 (컵은 컵끼리만).
- 약점: 컵이 두 개일 때 어느 컵이 옮겨졌는지 구분이 어렵다 → 부족하면 생김새 정보를 붙인다. 물체마다 SigLIP 2 임베딩이 이미 있으므로 그 코사인 유사도가 첫 후보다(아직 안 정함).
- 비교 기준: 위치 거리만 쓰는 가장 단순한 방법.

잴 것: 같은 물체가 두 번 등록된 수, 다른 물체가 하나로 합쳐진 수.

### 물체 찾기 — 임베딩 벡터 찾기 + 이름(의미) 찾기, 둘 다 쓴다

> **갱신 (2026-10-05): 에이전트와 RecallVLA 가 같은 색인을 둘 다 검색한다.**
> - **공용 물체 색인 하나(C++)**: 물체별 대표 벡터 μ·시점 벡터·이름 확률·상태·위치·방·마지막으로 본 때. 라벨 표 검색 코드(`scene_graph/clip`)를 넓혀 만든다.
> - **에이전트(LLM, API)**: 벡터를 입력으로 못 받으므로 검색은 도구 안에서 하고 **글(색인된 이름·속성)만 돌려준다**. 도구 `search_objects(query)`·`confirm_object(id, name, source)`.
> - **3 단계 재검색**: ① 이름·동의어·상위어 검색 → ② 없거나 약하면 **이름을 무시한 생김새 재검색**(저장한 시점 벡터로 물체마다 P(질의어 | 모습)을 계산; 예: "라디오"가 없지만 "소화기"로 등록된 물체가 라디오일 확률 2 등) → ③ 사용자 확인·가까이 본 결과로 **이름 고치기**(베이지안 갱신, 다음부터 ①에서 바로 찾음).
> - 벡터는 속성 단어(색·재질·크기)로도 바꿔 글 결과에 붙인다. SigLIP 2 글 인코더 TensorRT 엔진을 새로 만든다(자유 문장 → 벡터).
> - **RecallVLA**: 몸통이 만든 질의 벡터 q(SigLIP 2 글 공간으로 학습)로 같은 색인을 검색해 상위 K 개를 16 칸에 넣는다. 에이전트의 후보 id 는 힌트일 뿐이다. 학습 때 힌트 지우기·틀린 힌트·일부러 틀린 이름을 섞어, 이름보다 생김새를 믿게 한다(MAPVLA_SPEC).

- **임베딩 벡터 찾기**: 질의 글("빨간 컵", 한국어도)을 SigLIP 2 B/32 글 공간의 벡터로 바꾸고, 기억 속 물체 벡터(물체 조각의 원본 영상 임베딩)와 **코사인 유사도**(L2 정규화한 벡터의 내적)로 비교해 가까운 순으로 고른다. 이름표에 없는 말("물 마시는 거")로도 찾을 수 있다.
- **이름(의미) 찾기**: 물체마다 라벨 표에서 고른 이름·상위어(기억 폴더 `cache/`)를 붙여 두고, 이름이나 상위어("컵" ⊂ "식기")로 찾는다. 빠르고 결과를 설명하기 쉽다.
- 찾은 물체(이름·위치·상태)를 LLM 프롬프트에 넣어 답·계획을 만들게 하는 구조가 **RAG**(Retrieval-Augmented Generation, 검색 증강 생성)다. 우리 agent 는 물체 기억을 도구로 찾아 그 결과로 답하므로 이 방식에 해당한다.
- 임베딩·라벨 표·한국어 질의 모델은 [`training/embed`](../training/embed/README.md), 실행 쪽은 서브모듈 `src/scene_graph/clip`.

### 방 나누기·이름 — 직접 만듦 (CLIP + SAM 구조로)

- **방 나누기**: 2D 점유 격자의 빈칸으로 방을 나눈다(Hydra room finder 의 2D 판 — 벽까지 거리 변환, 넓은 곳부터 채우다 좁은 통로(문)에서 만나면 각각 방, 넓은 문은 합침). 지도가 바뀌어도 칸이 많이 겹치는 이전 방과 짝지어 **방 id 를 유지**한다. 물체는 바닥 자리가 걸친 방 칸 다수결로 방에 배정.
- **방 이름 (1) 물체 이름 규칙**: 방 안 물체 이름의 머리 명사로 방 종류 점수를 더한다(냉장고·오븐 → kitchen 3, 침대 → bedroom 3, 소파 → living room 3, 변기 → bathroom 3, 책상 → office 2 …). 가장 높은 종류가 2 점 이상이면 그 이름("kitchen", 같은 종류가 또 있으면 "kitchen 2"), 아니면 "room N". 근거(어느 물체가 몇 점)를 같이 남긴다.
  - **물체 이름은 SigLIP 2 이름**(기억 폴더 `cache/names.json`: 라벨 표 1위 이름, 확신이 낮으면 상위어)을 쓴다. FastSAM-s 는 클래스 이름이 `object` 하나라 검출기 이름만 보면 모든 방이 "room N" 이 된다. 라벨 표(약 2.9만 개)가 검출기 어휘보다 넓어서 규칙에 맞는 물체도 많아진다.
- **방 이름 (2) 임베딩 제로샷 분류** (다음 단계): 이름을 거치지 않고 "a kitchen"·"a bedroom"·"a laundry room" 같은 글 벡터를 SigLIP 2 글 공간에서 만들어, 방 안 물체 벡터(점수 가중 평균)나 방에서 찍은 장면 영상과 코사인으로 비교해 방 종류를 고른다.
  - 규칙표에 없는 방(세탁실·차고·다용도실)과 물체가 적은 방도 맞출 수 있다. (1) 과 점수를 합쳐 쓴다.
- **외부 이름**: LLM 이 대화로 정한 이름("민수 방")은 `sm_set_room_name` 으로 규칙보다 우선해 덮어쓴다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| 검출기(YOLO-seg·YOLOE) 클래스 이름만 규칙에 | 이름표 밖 물건은 근거가 안 되고, FastSAM-s 로 바꾸면 이름이 없음 |
| LLM 에게 방 물체 목록을 주고 이름 묻기 | 키프레임마다 부르면 느리고(요청당 수 초) 인터넷이 필요. 사람과 대화로 정할 때만 쓴다 |

잴 것: 시뮬 정답 방 종류와 맞는 비율, 방 이름이 바뀌는(흔들리는) 횟수.

### 지도 갱신 — 직접 만듦

- 물체가 옮겨지거나·사라지거나·새로 생기면 지도에서 **바뀐 부분만** 고친다.
- 참고: DovSG (국소 갱신), Khronos (시간에 따른 변화 추적).

잴 것: 옮겨짐·사라짐·생김을 맞힌 비율 (P/R/F1), 갱신 속도.

### 물체 지도 저장·보기 — Spark-DSG + 뷰어

- Hydra·Khronos 가 쓰는 형식이라 참고 코드의 도구를 그대로 쓸 수 있다. 층·방·물체 계층을 지원한다.
- 기억 폴더: `scene.json`(Spark-DSG), `objects/`(물체마다 RGB·depth·마스크·점구름·임베딩 = 원본), `cache/`(이름·찾기 색인 = 지워도 다시 만듦). 지도 자세는 `SGRT_POSE`(`slam`·`odom`·`gt`, 시뮬은 `gt`)로 고른다.
- 보기는 **2D 지도** 뷰어 **sgview**(Rust 서버 + three.js, `src/scene_graph/sgview`, 실행 `tools/run_sgview.sh`·`tools/run_explore_live.sh`, sgrt 소켓 → SSE 60 Hz 이상). spark-dsg 의 웹 3D 뷰어는 안 쓴다(10-02). 처음 쓰던 Python 뷰어 sgviz(서브모듈 `scene_graph/viewer`, Spark-DSG + viser)는 파일 폴링이라 실시간이 아니어서 기록용으로만 남겼다.

agent(LLM) 에게는 읽기 쉽게 JSON 으로 바꿔 넘긴다.

### 큰 계획·대화 (LLM) — Qwen3.5-9B, KAU API (AI agent 수업)

- 리모의 agent 가 AI agent 수업(최영식 교수님)이 주는 Qwen API 로 질문을 보낸다: `https://agent.kau.ac.kr/v1`, 모델 `qwen3.5-9b` (vLLM). 요금이 없다. 학교 서버(`agent.kau.ac.kr`)에 HTTPS 로 붙으므로 인터넷이 필요하다.
- 서버는 수업 쪽이 운영한다. 우리가 띄우거나 끄지 않는다.

| 다른 후보 | 안 고른 이유 |
|---|---|
| Claude · GPT 등 유료 API | 요금이 든다 |

잴 것: 명령 10~20개로 VLA 에게 맞는 지시를 주는 비율, 물체를 놓쳤을 때 다시 찾아가는 비율, 응답 시간.

### 작은 계획·행동 (VLA) — RecallVLA (π0.5 는 버림)

> **10-06**: VLA 는 **RecallVLA** 다 — Qwen3.5-0.8B 전체 학습 + SigLIP 2(영상 3장: 리모 RGB·손목 RGB·위에서 본 지도), 기억 요약 인코더(물체 ≤ 256 → 32 + 정밀 칸 8, "기억에 없음"), 단계 문장 + flow matching 행동([MAPVLA_SPEC](map_vla/MAPVLA_SPEC.md), `training/vla`). 교사는 빈 지도에서 자라는 지도로 학습하는 강화학습·대본 교사(행동은 학생과 같은 입력, 특권은 critic 만), 학생은 BC·DAgger([CURRICULUM_BEHAVIOR2026](map_vla/CURRICULUM_BEHAVIOR2026.md) 5.7). π0.5 코드·가중치는 10-06 지움. 아래는 그 전 판단으로 남긴다.

- LLM 의 지시와 카메라 영상을 받아, 할 일을 스스로 잘게 나눠 로봇을 움직이고 눈앞의 실패는 스스로 복구한다. 2025 BEHAVIOR 대회 1~3위 팀이 모두 π0.5 를 썼고, 공개된 모델과 코드가 많다.
- 학습(LoRA)은 학교 4090(24GB)에서 한다 (22.5GB 이상 필요, 4090 에 겨우 들어감). 시뮬 평가·추론은 시뮬 작업 PC(RTX 5070 Ti 16GB)에서 잘 돈다.

| 다른 후보 | 크기 | 특징 |
|---|---|---|
| SmolVLA | 0.45B | 훨씬 가벼움 → π0.5 가 리모에서 끝내 안 돌면 대안 |
| GR00T N1.5 | 3B | NVIDIA, 실행에 16GB 이상 |

잴 것: 시뮬레이션에서 집기·놓기 성공률.

### π0.5 실행 위치 — 리모, 안 되면 라즈베리파이 5 + DEEPX DX-M1

> **10-04**: π0.5 를 버려서 아래 크기 계산(3.3B)은 기록이다. "리모, 안 되면 라즈베리파이 5 + DX-M1" 순서는 우리 VLA 에도 그대로 두되, 우리 VLA 를 리모·DX-M1 에서 잰 적은 없다.

| 순서 | 방법 | 확인할 것 |
|---|---|---|
| 1 | 리모(Jetson Nano) 에서 양자화한 π0.5 | 공식 기준 실행에 8GB 이상 → 4bit 로 줄여도 4GB 에 빠듯함. 먼저 재 본다 |
| 2 | 안 되면 리모에 라즈베리파이 5 + **DEEPX DX-M1** (보유, 25 TOPS INT8, 메모리 4GB, M.2) | DXNN SDK 로 ONNX 모델을 변환해 올린다. π0.5 는 3.3B 라 INT8 로도 약 3.3GB → **4GB 에 빠듯함**. transformer·행동 생성 부분이 변환되는지 확인 |

양자화 단계: bf16 (원래) → int8 → int4 순으로 줄이며 성공률이 얼마나 떨어지는지 잰다.

### 시뮬레이션 — 2026 BEHAVIOR Challenge 벤치마크 → BEHAVIOR-1K 장면 + GPU 환경 (10-06)

> **10-06**: 대회 점수는 목표가 아니다(리모 집기·놓기가 목표). 옛 대회 저장소는 분리했고, 쓰던 것(인지 코드, 리모 시뮬 실행, 플래너, 학습 렌더)은 robot-agent 로 옮겼다. 집 장면·물체·과제 인스턴스(BEHAVIOR-1K)는 `third_party/BEHAVIOR-1K` 에 두고 OmniGibson 으로 확인·재현한다. 대량 학습은 GPU 일괄 환경(`training/RL`). 학습 집 4 채(1층 집·2층 집 아래층·식당·사무실) / 평가 집 3 채(Rs_int·호텔 방·2층 집 위층)로 나눈다. 아래는 그 전 판단으로 남긴다.

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
| 2026-10-03 | 방 나누기·이름 절 추가: CLIP + SAM 구조로 — SigLIP 2 이름을 방 규칙에, 다음은 임베딩 제로샷 방 분류, LLM 이름은 덮어쓰기 | FastSAM-s 는 클래스 이름이 없어 검출기 이름만 쓰면 방 종류를 못 붙임 |
| 2026-10-03 | 물체 인식 후보 B 적어 둠: YOLOE 그대로 + SigLIP 2(+ training/embed 머리) — FastSAM-s + SigLIP 2 와 비교해서 정함 | |
| 2026-10-05 | 10-05: decided ObjectSAM (YOLO26n 학생) + SigLIP 2 + objprob — 분할 기본 엔진 `yolo26n-seg-obj-416`(FastSAM-s 에서 증류, things 만), 확률 모드 기본 켬 | 까닭: FastSAM-s 계산의 약 1/10 이라 LIMO 의 Jetson(특히 Nano)에 맞다 — 기기 위 시간은 아직 안 잼 |
| 2026-10-04 | VLA: **π0.5 버림** → 우리 작은 VLA(`training/BC`, G5). π0.5 절·실행 위치 절은 기록으로 남김. SLAM: 시뮬은 자체 `slam2d`, 실기 선택은 다시 정함 | π0.5 가중치는 일부러 지움. Cartographer 는 저장소에 없음 |

</details>
