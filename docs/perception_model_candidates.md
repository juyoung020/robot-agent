# 물체 인식 모델 후보 (① 물체 기억)

작성 2026-10-03. **추천만 하는 문서**다. 결정은 팀이 [모델 선택](model_selection.md) 에서 한다.

- 지금 쓰는 것: `src/behavior-2026/src/scene_graph/ovdet` = **YOLOE-11 text-prompt seg**, TensorRT FP16, C API, 어휘 272개.
  - 비교 기록: `src/behavior-2026/docs/ovdet_검출기.md`. 숫자는 10-02 에 "다시 잴 것" 표시가 붙어 있다. 그래서 방향 참고로만 쓴다.
- 표기
  - (추정): 출처 없이 비슷한 모델에서 미루어 본 값.
  - (출처 기기): 우리 기기가 아닌 곳에서 잰 값.
  - 숫자 옆 [n]: 5절 출처 번호. [D1]… 은 바로 아래 새 절의 출처.
- 문서 순서
  - **A′**: 리모 기본형 단독(Jetson Nano 만). **지금 기준이다.**
  - **A**: Nano + DX-M1(RPi5) 가정. A′ 의 근거로 남긴다.
  - **B**: 시뮬 추천(짧게).
  - **이전 가정**: 리모 프로와 4090 을 쓸 수 있다고 본 첫 판.

## A′. 리모 기본형 단독(Jetson Nano만) 추천 (10-03, 최종 가정)

**가정**
- 쓰는 것은 Jetson Nano 4 GB 하나다: JetPack 4.6, Ubuntu 18.04, TensorRT 8.2, Maxwell 128 코어(sm_53), FP16 약 472 GFLOPS, 텐서코어 없음.
- Nano 메모리를 Cartographer · ROS 2 · scenemap · agent 와 함께 쓴다.
- RPi5, 외부 인지 상자, 4090·원격은 **없다**.

**DX-M1 은 뺀다.**
- DX-RT 지원 OS 는 Ubuntu 20.04 이상이고, 18.04(커널 4.9)는 목록에 없다 [D8].
- Nano 의 M.2 는 Key E(PCIe ×1)인데 DX-M1 은 M.2 M-key 2280 이다(어댑터 필요).
- 그래서 Nano 단독으로 붙여 구동한다는 근거가 없다. 공식 지원이 생기면 다시 본다.

### A′.1 고른 것 (3개)

| # | 모델 | 역할 | 설정 | 속도·메모리 |
|---|---|---|---|---|
| 1 | **YOLOE-26n-seg**, text prompt 를 export 때 굳힘(272 + 동의어. 리모용으로는 집 물건 위주 100개 안팎으로 줄여도 됨) | 실시간 등록(마스크 + 이름), **열린 어휘 유지** | 416×416, FP16. PC 에서 ONNX `opset=12`, `end2end=False`, `simplify=True`, 정적 크기 → Nano 에서 `trtexec --fp16` | **6–9 FPS (추정)**. n 크기라 연산량이 YOLO11n-seg 와 비슷하다고 본다. 엔진 + 실행 메모리 0.3–0.45 GB(CUDA 문맥 포함, 추정) |
| 2 | **YOLO11n-seg**, 우리 물체로 미세조정(닫힌 어휘) | 1번이 TRT 8.2 에서 안 만들어지거나 너무 틀리면 대신 쓴다 | 416, FP16, 위와 같은 export | **8–10 FPS (추정)**. 근거: YOLOv8n-seg 가 Nano TRT FP16 4.2 FPS(640, 측정) [12], 416 은 연산량이 약 0.42배. 메모리 0.3–0.4 GB(추정) |
| 3 | **YOLOE 물체 임베딩**(1번 망의 머리 출력, 추가 망 없음) | crop 임베딩 대신. 같은 이름 물체 구분, 이름 다시 매기기 | 엔진 출력을 하나 더 낸다(512-d 안팎, 추정) | 추가 연산은 거의 0. 1번을 쓸 때만 가능하다 |

- **INT8 은 쓰지 않는다.** Nano(sm_53)는 INT8 을 지원하지 않는다 [D10]. 실제로 빨라지지 않고 정확도만 떨어진다 [12].
- 입력 크기: 320 은 더 빠르지만(10–14 FPS, 추정) 작은 물체(컵·캔)를 놓친다. 640 은 4–5 FPS 다. **416 을 기본으로 하고 첫 시험에서 셋 다 잰다.**
- export 는 **PC 에서만** 한다. Nano 안의 Ultralytics·PyTorch 는 너무 새것이라 안 된다 [D2]. TRT 8.2 는 opset 12 가 권장이다 [D3]. YOLO26 계열은 NMS 없는 출력(TopK)이 TRT 8.2 에서 될지 불확실해서 `end2end=False` 로 낸다 [D4]. NMS 는 ovdet 의 CUDA 후처리가 한다.

### A′.2 살아남는 역할 / 없어지는 역할

| 역할 | 결론 |
|---|---|
| ① 실시간 seg | **남음.** 1번(안 되면 2번). 5 Hz 로 묶는다(GPU 를 SLAM 쪽 여유로 남김) |
| ② crop 임베딩 (별도 CLIP 류) | **기본은 뺀다.** CLIP RN50 급은 Nano TRT FP16 에서 장당 40–60 ms(추정. Nano 측정 ResNet18 FP16 18 ms @256 [D10] 에서 미룸), 메모리 0.2–0.3 GB(추정). 계산은 best view 때만이면 버틸 수 있지만 메모리가 빠듯하다. 1번을 쓰면 3번(YOLOE 임베딩)이 공짜로 대신한다 |
| ②의 대체 | **이름 + 위치로 같은 물체 판단**(팀 문서 원래 방식) + **agent 의 한국어 동의어 사전**("머그잔"·"텀블러"→cup). 같은 이름 물체 여럿은 색 히스토그램(CPU, 마스크 안 HSV)으로 구분한다 |
| ③ 이름 확인 (VLM·SAM) | **없어진다.** 남는 것: 노드 안 이름 점수 가중 다수결. 이름이 갈리면 "불확실" 로 두고 agent 가 사용자에게 묻는다. 1번을 쓰면 YOLOE 임베딩으로 어휘 글 임베딩(미리 계산)과 다시 비교해 이름을 고른다 |

### A′.3 파이프라인 (Nano 하나)

```
Orbbec DaBai RGB-D ──▶ [GPU] YOLOE-26n-seg 416 FP16 (5 Hz) ── 마스크·이름·점수·(임베딩)
                                │
                     [CPU] scenemap: 마스크 무게중심 + depth 중앙값 → xyz, DA = 이름 다수결 + 위치(+색·임베딩)
                                │                     ▲
라이다 ──▶ [CPU] Cartographer ──┘ 로봇 자세            │
                                ▼                     │
                     Spark-DSG 기억 ◀──── agent(find_object · 한국어 사전)
손목 카메라: 집기 직전·직후에만 같은 엔진으로 1–2 Hz (따로 엔진을 두지 않음)
```

### A′.4 Nano 예산 (추정, 4 GB 공유)

| 항목 | 메모리 | 계산 |
|---|---|---|
| Ubuntu 18.04 (화면 끔) + 기본 서비스 | 0.5–0.7 GB | CPU 소량 |
| ROS 2 (노드 5–8개, 카메라·라이다 드라이버) | 0.3–0.5 GB | CPU 0.5–1 코어 |
| Cartographer 2D (위치 추정 모드) | 0.3–0.5 GB | CPU 1–2 코어 |
| 검출기 (CUDA 문맥 + TRT 엔진 + 버퍼) | 0.3–0.45 GB | GPU 5 Hz × 110–160 ms = **55–80%** |
| scenemap + Spark-DSG + best view PNG | 0.1–0.2 GB | CPU 0.5 코어 |
| agent (도구·계획 로직만) | 0.05–0.15 GB | CPU 소량 |
| **합계** | **1.6–2.5 GB** | 4 GB 중 1.5 GB 이상 남김. 스왑은 끄지 말고 zram 을 둔다 |

- **agent 의 LLM 은 Nano 에 못 올린다.** Qwen3.5-9B 는 물론이고 1–2B 급도 위 예산과 같이 4 GB 에 넣기 어렵다(추정). 이 가정이면 계획·대화를 규칙 기반으로 하거나 LLM 위치를 따로 정해야 한다 → **팀 결정이 필요하다**(이 문서 범위 밖).
- 전원 모드는 MAXN(10 W)을 쓴다. 열 때문에 속도가 떨어지면 검출 주기를 3 Hz 로 낮춘다.

### A′.5 뺀 것

- DX-M1(위 이유), RPi5·외부 상자, 4090.
- SAM 3 · SAM 계열 분할기.
- VLM 전부(Qwen·SmolVLM·Moondream·Florence-2), RAM++.
- YOLOE-11s 이상: 416 에서 2–3 FPS(추정). s 크기부터 연산량이 n 의 약 3배다.
- 별도 CLIP/SigLIP 임베딩(기본에서는 뺌).
- Grounding DINO · OWL · YOLO-World 등 트랜스포머·상자 전용 검출기.

### A′.6 실제 Nano 에서 먼저 할 것 3가지

1. **엔진 만들기 + 단독 속도.**
   - PC 에서 YOLOE-26n-seg(굳힌 어휘)와 YOLO11n-seg 를 `opset=12, end2end=False` ONNX 로 내보낸다.
   - Nano 에서 `trtexec --fp16` 으로 320/416/640 엔진을 만든다. TRT 8.2 에서 안 되는 연산이 있으면 그 이름을 적는다.
   - `trtexec` 지연과 `tegrastats` 메모리를 잰다.
2. **전부 같이 띄운 상태.**
   - Cartographer + ROS 2 + 카메라 + 검출기(5 Hz) + scenemap 을 동시에 30분 돌린다.
   - 최대 RAM, GPU 점유율, 온도·클럭 저하, 검출 지연 p99 를 잰다.
3. **우리 물체로 품질.**
   - 리모로 녹화한 bag(집·실험실 1바퀴, 물체 20–30개)을 쓴다.
   - 1번과 2번의 과제 물체 재현율, 프레임 사이 이름 바뀜 비율, 같은 물체가 둘로 등록된 수를 잰다.
   - 1번이 2번보다 크게 나쁘지 않으면 1번(미세조정 없이 열린 어휘)을 쓴다.

출처 추가: [D10] Jetson Nano TensorRT 벤치마크(ResNet18 FP16 18 ms @256, "Nano 는 INT8 미지원") — https://github.com/kentaroy47/benchmark-FP32-FP16-INT8-with-TensorRT. 나머지는 [12]·[D2]–[D4]·[D8].

---

## A. 리모 기본형·온디바이스 가정 추천 (10-03 갱신, A′ 이전 가정: Nano + RPi5·DX-M1)

**가정**
- 로봇: 리모 기본형(Jetson Nano 4 GB, JetPack 4.6, TensorRT 8.2, Maxwell GPU. bf16·텐서코어가 없다) + DEEPX DX-M1(25 TOPS INT8, 자체 LPDDR5 4 GB).
- **모두 로봇 안에서 돈다.** 학교 4090, KAU API, 원격 확인은 쓰지 않는다.
- Nano 메모리 4 GB 는 Cartographer, ROS 2, 물체 기억, agent 가 함께 쓴다.

### A.0 한눈에

| # | 모델 | 맡는 일 | 도는 곳 | 예상 속도·메모리 |
|---|---|---|---|---|
| 1 | **YOLO26-seg (s/m), 우리 물체로 미세조정** (닫힌 어휘 약 100–300개) | 실시간 등록. **확실한 경로** | DX-M1 (INT8) | RPi5 + DX-M1 에서 n–l 모두 end-to-end 약 55 FPS [5][D6]. NPU 메모리 수십 MB(추정) |
| 2 | **YOLOE-11s / 26s-seg, 어휘를 굳혀서(272 + 동의어) DX-COM 으로 컴파일** | 실시간 등록. 열린 어휘를 유지한다. **컴파일이 되면 1위** | DX-M1 (INT8) | 30–55 FPS(추정. 같은 크기 YOLO-seg 와 거의 같은 conv 망이다). 공식 지원 목록에는 없다 [6] |
| 3 | **YOLO26n-seg / YOLO11n-seg, TensorRT 8.2 FP16** | DX-M1 이 없을 때의 대체. 손목 카메라 보조 | Nano GPU | 640 입력 약 4–5 FPS(v8n-seg 4.2 FPS 측정 [12]), 416 입력 8–10 FPS(추정). 메모리 0.4–0.6 GB(CUDA 문맥 포함, 추정) |
| 4 | **CLIP ViT-B/32 영상 인코더** (OpenAI, MIT). 같은 자리의 비교 후보: SigLIP 2 B/16(Apache), MobileCLIP 2-S0(연구 전용) | best-view crop 임베딩 + **라벨 목록 zero-shot 이름 붙이기** | DX-M1. 글 임베딩은 미리 계산하고, 자유 질의만 CPU 에서 | crop 당 2–5 ms(추정), 모델 약 90 MB INT8(추정). best view 를 고칠 때만 돈다 |
| 5 | **SmolVLM-256M / 500M** (Apache-2.0, llama.cpp GGUF) | (선택) 새 물체를 **한가할 때 천천히** 확인 | RPi5 CPU (Nano 에는 올리지 않는다) | crop 하나에 10–30 s(추정), 메모리 0.3–0.8 GB(추정). 빠지면 3번 역할은 4번 CLIP zero-shot 만 남는다 |

**결론**
- **DX-M1 은 Nano 에 바로 꽂지 말고, 팀 문서대로 라즈베리파이 5 에 꽂아 "인지 상자"로 쓴다.** 이유는 A.1 의 DX-RT 지원 OS 다.
- 등록 + 임베딩 + 이름 고르기를 DX-M1 하나에서 처리하고, Nano 는 SLAM · 물체 기억 · agent 만 맡는다.
- 무거운 확인(SAM 3, 9B VLM)은 **뺀다**. 대신 다음 셋으로 메운다.
  - 같은 물체 판단(DA)에서 이름을 다수결로 정한다.
  - CLIP zero-shot 으로 이름을 다시 매긴다.
  - (선택) 작은 VLM 을 느리게 돌린다.

### A.1 기기 사실 (제약)

| 기기 | 사실 | 우리에게 뜻하는 것 |
|---|---|---|
| Jetson Nano | Maxwell 128 코어, FP16 은 되지만 INT8 가속·텐서코어가 없다. JetPack 4.6 = Ubuntu 18.04, Python 3.6, TensorRT 8.2 | YOLO 계열 n 크기만 실시간 근처로 돈다. 트랜스포머 영상 모델은 사실상 불가(추정) |
| TensorRT 8.2 export | Ultralytics 는 TRT ≤ 8.2 에 **opset 12** 를 권한다 [D3]. Nano 안에서는 export 가 안 된다(PyTorch·Ultralytics 가 너무 새것). PC 에서 ONNX 로 내보내고, 엔진은 Nano 에서 만든다 [D2][D3] | YOLO11-seg 는 opset 12 ONNX → `trtexec` 로 된다(Qengineering 이 v8 로 같은 경로를 썼다 [12]). YOLO26 은 `end2end=False`(NMS 있는 출력)로 내보내는 편이 안전하다. TopK·NMS 없는 머리가 TRT 8.2 에서 되는지는 확인해야 한다(추정) [D4] |
| DX-COM (컴파일러) | 컴파일은 x86-64 Linux 에서만 된다 [6]. 입력은 **정적 크기**, INT8 고정이다 | 프롬프트 없는 YOLOE(LRPC 의 동적 걸러내기)는 그대로 못 올린다 → 어휘를 굳힌 text-prompt 판으로 바꿔 올린다(추정) |
| DX-COM 연산자 | MatMul·Gemm 은 제한이 없다. Softmax 는 축 길이 ≤ 4080. Gelu 를 지원하고, LayerNorm 은 쪼개진 형태(Div)로 지원한다. Resize 는 `pytorch_half_pixel` 과 정수 배율만. 채널 방향 broadcast 는 안 된다. 동적 크기 불가 [D1] | ViT 주의집중(토큰 50–197개)은 연산자 조건 안에 든다. DEEPX 모델 동물원에 ViT/DeiT/BEiT · FastViT · SegFormer 가 있다 [26]. Advantech 의 DX-M1 CLIP 데모는 영상 인코더를 NPU 에, 글 인코더를 CPU 에 둔다 [D5]. 다만 Sixfab 문서는 "트랜스포머는 아직 안 된다"고 적는다 [D7] → **서로 어긋난다. 직접 컴파일해 봐야 한다** |
| DX-M1 INT8 정확도 | Sixfab 기준 FP32 대비 약 2% 손실 [D7]. 교정 이미지에 따라 달라진다 | 작은 물체·낮은 conf 쪽이 먼저 깨진다(추정). 교정 이미지는 우리 집·실험실 프레임으로 고른다 |
| DX-M1 속도 | RPi5 + DX-M1 YOLO26 seg: end-to-end 55 FPS(n). RPi5 는 PCIe Gen3 ×1 이 병목이라 **seg 는 large 까지 같은 FPS** 다 [D6]. m-seg 모델 처리량은 80 FPS [D6] | 크기를 키워도 거의 공짜다 → **s/m-seg 를 기본으로** |
| DX-RT (런타임) 지원 OS | x86_64·aarch64, **Ubuntu 20.04 이상**, Debian 12/13 [D8]. Ubuntu 18.04(JetPack 4.6, 커널 4.9)는 목록에 없다 | Nano 에 직접 꽂으면 커널 드라이버 빌드부터 막힐 가능성이 크다(추정). Nano 개발 키트의 M.2 는 Key E(PCIe ×1)라 어댑터도 필요하다. **RPi5 + DX-M1 을 따로 두고 Ethernet 으로 Nano 와 잇는다** |
| DX-M1 를 π0.5 와 나눔 | 팀 문서는 π0.5(INT8 약 3.3 GB)도 DX-M1 에 올릴 수 있다고 본다 | 인지 모델(합쳐 약 0.1–0.2 GB)은 메모리에 들어간다. 다만 **NPU 시간을 나눠 써서 π0.5 지연이 늘어난다**. π0.5 를 올리면 검출은 2–5 Hz 로 낮춘다 |

### A.2 Nano 메모리 예산 (추정)

| 항목 | 추정 |
|---|---|
| Ubuntu 18.04 (화면 없이) + ROS 2 노드들 | 0.8–1.2 GB |
| Cartographer (2D) | 0.3–0.6 GB |
| scenemap · Spark-DSG 기억 · best view PNG 캐시 | 0.1–0.3 GB |
| agent (LLM 이 로봇 안에 있다면 모델 크기만큼 더) | 이 문서 범위 밖. 4 GB 에서 남는 몫이 거의 없다 |
| **남는 것** | **약 1–2 GB** → Nano GPU 에 검출기를 올리면(3번, 0.4–0.6 GB) 빠듯하다. 그래서 인지는 DX-M1 쪽으로 뺀다 |

### A.3 역할별 다시 거르기

| 역할 | 후보 | Nano GPU | DX-M1 | 판정 |
|---|---|---|---|---|
| ① 실시간 seg | YOLO26/11-seg 미세조정 | n: 4–5 FPS | **공식 지원, 약 55 FPS** | **채택(1)** |
| | YOLOE-11/26 text prompt(굳힘) | s: 1–2 FPS(추정, 연산량이 v8n-seg 의 약 3배) | 공식 목록 밖. 구조는 YOLO-seg + 1×1 conv 대조 머리라 될 가능성이 높다(추정) | **채택(2), 컴파일 시험 1순위** |
| | YOLOE 프롬프트 없음(4,585) | 불가(느림) | LRPC 는 동적이라 불가. 4,585개를 text-prompt 로 굳히면 머리 채널 4,585개(제한 32,768 안) → 시험 대상(추정) | 2번의 변형으로 시험 |
| | RF-DETR-Seg (Apache) | DINOv2 백본이라 느림(추정) | 트랜스포머가 될지 불확실 | 뺌(라이선스 대안으로만 기억) |
| | FastSAM (+CLIP) | 느림 | **DX 동물원에 있다** [26] | 뺌. 우리 비교에서 이름 정밀도 0.21, 조각남 2.9 [ovdet_검출기.md]. 이름 없는 "무엇이든" 보조로만 |
| | NanoOWL · Grounding DINO · OWLv2 · YOLO-World | Orin 전용이거나 너무 느림 | 상자만 내고, 트랜스포머 검출기는 불확실 | 뺌 |
| ② 임베딩 | CLIP ViT-B/32 (MIT) | 1–3 FPS(추정) | DX-M1 CLIP 데모가 있다 [D5] | **채택(4)** |
| | SigLIP 2 B/16 (Apache, 다국어) | 느림 | ViT-B/16 196 토큰 → 연산자 조건 안(추정) | 4번 비교 후보. 한국어 글을 바로 넣을 수 있다. 다만 글 인코더(어휘 256k)는 CPU 에서 무겁다 |
| | MobileCLIP 2-S0 (FastViT) | 2–4 FPS(추정) | FastViT 가 동물원에 있다 → 가장 잘 될 것 | 성능은 가장 좋을 수 있지만 **연구 전용 라이선스** [27] → 공개 데모에 걸리면 뺀다 |
| | YOLOE 물체 임베딩 | 2번과 함께 | 2번이 되면 따로 돌 필요가 없다 | 2번이 되면 첫 시도 |
| ③ 이름 확인 | CLIP zero-shot (라벨 목록 = 272 + RAM++ 4,585 태그 + 한국어 사전) | — | 4번과 같은 모델, 글 임베딩은 미리 계산 | **채택(4번이 겸함)** |
| | SmolVLM-256M/500M | 메모리가 없다 | VLM 은 지원 근거가 없다 | **(5) 선택**, RPi5 CPU 에서 한가할 때만 |
| | Moondream 2 (1.9B), Qwen3-VL-2B, Florence-2 | 4 GB 에 SLAM 과 같이 못 올림 | 근거 없음 | 뺌 |
| | RAM++ (Swin-L), SAM 3, Qwen3.5-9B | 불가 | 불가 | 뺌 |

### A.4 온디바이스 파이프라인

```
[머리 RGB-D · 손목 RGB]──USB──▶ RPi5 + DX-M1 (인지 상자, Ubuntu 22.04/24.04)
     ① YOLO26-seg 또는 YOLOE-seg(굳힘) INT8  ── 머리 5–10 Hz(필요한 만큼만), 손목 2–5 Hz
     ② best view 가 바뀐 물체만 crop → CLIP B/32 영상 인코더 INT8 → 512-d
     ③ 512-d · (미리 계산한 라벨 글 임베딩) → zero-shot 이름 후보 상위 3개
     (선택) 새 물체 → SmolVLM 대기열(CPU, 느림)
        │ 검출(이름·점수·상자·마스크 비트·임베딩)을 ROS 2 토픽으로 (Ethernet)
        ▼
  Jetson Nano: Cartographer · scenemap(무게중심 + depth → xyz, DA = 이름 다수결 + 위치 + 임베딩 거리) · Spark-DSG · agent
```

| 항목 | 예상 (추정, [D6] 기준으로 미룸) |
|---|---|
| 머리 검출 | YOLO26s/m-seg 640: NPU 처리 12–20 ms, RPi5 쪽 전후처리 포함 end-to-end 20–30 ms → 30 FPS 이상 가능. **실제로는 5–10 Hz 로 묶어** π0.5·임베딩과 NPU 를 나눈다 |
| 임베딩 | crop 당 2–5 ms. 초당 수 개 |
| DX-M1 메모리 | seg 20–40 MB + CLIP 영상 약 90 MB → 약 0.15 GB. π0.5 를 올려도 4 GB 안 |
| RPi5 메모리 | 디코딩·전후처리·DX-RT 0.3–0.5 GB. SmolVLM 을 쓰면 +0.3–0.8 GB |
| Nano 추가 부담 | 검출 메시지를 받는 데 수십 MB, GPU 0 |
| DX-M1 이 없을 때 | Nano GPU 에 YOLO26n-seg 416 FP16 → 8–10 FPS(추정), +0.4–0.6 GB. 임베딩은 빼거나, best view 때만 CPU 에서 MobileCLIP 2-S0/CLIP(장당 0.5–1 s, 추정) |

- **한국어·동의어 검색**: CLIP 은 영어 글만 받는다. 그래서 agent 계획의 한국어 사전("머그잔"→mug→cup)으로 먼저 영어로 바꾼 뒤 글 임베딩과 비교한다.
  - 사전에 있는 단어는 글 임베딩을 미리 계산해 두므로 실행 중 글 인코더가 필요 없다.
  - 사전 밖 자유 질의만 RPi5 CPU 에서 CLIP 글 인코더를 돌린다(ONNX Runtime C++, 질의당 수십 ms, 추정).
- **이름 안정**: 프레임마다 나온 이름을 노드에서 점수 가중 다수결로 정한다. 이름 후보가 갈리면 CLIP zero-shot 상위 1개로 정한다. 그래도 갈리면 "불확실" 표시를 하고 agent 가 사용자에게 묻는다.

### A.5 확인할 것 (순서대로, 각 반나절–하루)

1. RPi5(Ubuntu 24.04)에 DX-RT·드라이버를 설치하고 `run_model` 로 YOLO26s-seg 동물원 모델을 돌린다(55 FPS 재현).
2. **YOLOE-11s / 26s text-prompt(272 + 동의어)를 opset 11+ 정적 ONNX 로 내보내고 `dxcom` 으로 컴파일**한다. 마스크 IoU·이름을 FP32 와 비교한다. 실패하면 1번(YOLO26-seg 미세조정)으로 확정한다.
3. CLIP B/32 · SigLIP 2 B/16 · MobileCLIP 2-S0 영상 인코더를 DX-COM 으로 컴파일한다. INT8 과 FP32 임베딩의 코사인 유사도, zero-shot top-1 을 잰다.
4. PC 에서 opset 12 로 YOLO26n-seg(end2end=False) · YOLO11n-seg 를 내보낸다. Nano TRT 8.2 로 엔진을 만들고, Cartographer 와 ROS 2 를 띄운 채 FPS·메모리를 잰다.
5. (선택) RPi5 에서 SmolVLM-256M llama.cpp 로 crop 하나에 걸리는 시간을 잰다. 30 s 를 넘으면 뺀다.
6. 위 결과를 이전 가정의 4.2 비교 계획(같은 지표, 팀 벤치마크)에 넣는다. 다만 "VRAM" 대신 **Nano RAM · DX-M1 메모리 · NPU 점유율**을 잰다.

### A.6 뺀 것 (이 가정에서)

| 뺀 것 | 이유 |
|---|---|
| SAM 3 / 3.1 | 848M. Nano·DX-M1 어디에도 안 들어간다. 오프라인 라벨 도구로는 개발 PC 에서 계속 쓸 수 있다(실행 경로가 아님) |
| Qwen3.5-9B 확인, Qwen3-VL-2B, Moondream 2, Florence-2 | 4090 이 없고, Nano 4 GB 에 SLAM 과 같이 못 올린다. VLM 이 DX-COM 에서 된다는 근거가 없다 |
| 무거운 열린 어휘 검출기 (Grounding DINO 계열, OWL, Mask Grounding DINO, OmDet) | 트랜스포머 검출기. Nano 에 너무 느리고 DX-COM 지원이 불확실하다. 상자만 내는 것도 많다 |
| EfficientViT-SAM · NanoSAM · EdgeTAM | 분할기를 하나 더 두는 메모리·시간이 없다. YOLO-seg 마스크로 충분하다(팀 문서 근거) |
| RAM++ | Swin-L 급. 태그 목록만 가져와 CLIP zero-shot 라벨로 쓴다 |
| 이전 판 5위 "가끔 하는 무거운 확인" 역할 | **축소**: CLIP zero-shot + DA 다수결 + (선택) SmolVLM |

## B. 시뮬 (BEHAVIOR, RTX 5070 Ti) 추천 — 짧게

| 역할 | 추천 | 비고 |
|---|---|---|
| 실시간 seg | YOLOE-26 s/m text prompt, TRT 10 FP16 (지금 11m 을 바꿀 후보) | 이전 가정 3.1 그대로. 2–5 ms(추정), 약 0.2 GB |
| 임베딩 | **로봇과 같은 모델**(CLIP B/32 또는 A.5-3 에서 고른 것)을 TRT FP16 으로 | 시뮬과 실물의 기억 형식·검색 방식이 같아진다 |
| 이름 확인 | 로봇과 같게 CLIP zero-shot + DA 다수결 | 대회 평가 중 원격 API 는 규칙 확인 전까지 쓰지 않는다 |
| 오프라인 | SAM 3 로 의사 라벨 → A.0-1 의 YOLO26-seg 미세조정 데이터 | Isaac Sim 이 꺼져 있을 때만 |

### A·B 출처

- [D1] DEEPX DX-COM 지원 ONNX 연산자 — https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-COM/Building_Models/
- [D2] Ultralytics Jetson 빠른 시작(JetPack 4 도커, Nano 안에서 export 불가) — https://docs.ultralytics.com/guides/nvidia-jetson/
- [D3] Ultralytics DeepStream on Jetson(JetPack 4.6.4 · DeepStream 6.0.1 · TRT 8.2 이하는 opset 12) — https://docs.ultralytics.com/guides/deepstream-nvidia-jetson
- [D4] Ultralytics end-to-end 검출 안내(YOLO26 end2end, TRT 에서 자동 해제 조건) — https://docs.ultralytics.com/guides/end2end-detection
- [D5] Advantech DEEPX NPU CLIP VLM 컨테이너(영상 인코더 NPU, 글 CPU) — https://catalog.advantech.com/en-us/containers/deepx-npu-clip-vlm , https://github.com/Advantech-Containers/DEEPX-NPU-demo
- [D6] DEEPX dx-benchmark YOLO26 분석(RPi5B_M1 seg 55.1 FPS, m-seg 처리량 80 FPS, seg 는 large 까지 같은 FPS) — `/tmp/claude-1000/dxas/dx-benchmark/docs/ANALYSIS_EN.md` · `ANALYSIS_KOR.md`, https://github.com/jyun69/dx-all-suite
- [D7] Sixfab DEEPX 지원 모델 목록(트랜스포머 미지원 표기, INT8 약 2% 손실) — https://docs.sixfab.com/docs/ai-model-deployment-supported-models
- [D8] DX-AllSuite 구조 개요(DX-RT 지원 OS·아키텍처, 모델 동물원 345개) — `/tmp/claude-1000/dxas/docs/source/01_DX-AllSuite_Architecture_Overview.md`
- [D9] SmolVLM — https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct

---

# 이전 가정: 리모 프로(Orin) · 학교 4090 사용 가능 (10-03 첫 판)

> 아래는 첫 판 분석이다. 후보 20개 표와 출처는 A 절의 근거로 그대로 쓴다. **추천은 A 절이 앞선다.**

## 0. 한눈에

| # | 모델 | 맡는 일 | 도는 곳 |
|---|---|---|---|
| 1 | **YOLOE-26-seg** (s/m, text prompt). 지금 쓰는 YOLOE-11m 을 바꿀 후보 | 실시간 등록(마스크 + 이름) | 시뮬 PC, Orin |
| 2 | **YOLO26-seg** (n/s), 우리 어휘로 미세조정한 닫힌 어휘 | 실시간 등록(가벼운 기기용) | DX-M1, Jetson Nano |
| 3 | **SigLIP 2** (B/16, 다국어) | best-view crop 임베딩. "머그잔" ↔ cup 찾기, 색·속성 구분 | 시뮬 PC, Orin, DX-M1(추정) |
| 4 | **SAM 3 / 3.1** | 가끔 하는 무거운 확인: 놓친 물체 찾기, 오프라인 자동 라벨 | 4090(밖), 시뮬 PC 는 오프라인일 때만 |
| 5 | **Qwen3.5-9B 비전 입력** (이미 4090 에 있음) | 새 물체·헷갈리는 물체의 이름 확인 | 4090 API |

추천 조합(4절 A): YOLOE-26 실시간 등록 → best view 갱신 때만 SigLIP 2 임베딩 → 새 물체나 이름이 흔들릴 때만 Qwen3.5-9B 확인.

## 1. 요구사항

| 구분 | 요구 | 우리 파이프라인에서 나온 이유 |
|---|---|---|
| 필수 | **인스턴스 마스크** | 마스크 무게중심 + depth 중앙값 → xyz. 상자만 내면 배경 깊이가 섞인다. 상자만 내는 모델은 분할기를 하나 더 붙여야 한다 |
| 필수 | **열린 어휘 또는 아주 큰 어휘** | BEHAVIOR-1K 물체 상당수가 COCO-80 밖이다(radio receiver, electric refrigerator, coffee table, pumpkin, candle, 공구, 음식). `vocab_all.txt` 에 272개 |
| 필수 | **프레임 사이 이름 안정** | 같은 물체 판단(DA)이 "같은 이름끼리 위치 비교"다. `cup` ↔ `bowl` 로 흔들리면 노드가 둘로 갈라진다. 팀 벤치마크는 시간에 걸친 obj_id 를 채점한다 |
| 필수 | **네이티브 실행** (TensorRT 또는 DX-COM, C/C++) | 실행 중에는 파이썬이 없다. 평가기 프로세스 안에서 C ABI 로 부른다 |
| 필수 | **메모리·속도** | 시뮬 PC 는 16 GB 중 Isaac Sim 이 약 8 GB, π0.5 가 약 6 GB 를 쓰므로 남는 게 **약 2 GB**다. 검출은 약 6 step 마다, step 예산 40 ms 안에서 **수 ms** 여야 한다. 로봇은 SLAM(+π0.5) 옆에서 **수 Hz** |
| 필수 | **공개 저장소와 맞는 라이선스** | 제출물은 AGPL 공개로 이미 정했다(09-30). 연구 전용(비상업) 가중치는 공개 데모·논문에 걸림돌이 된다 |
| 있으면 좋음 | crop 임베딩 | 노드의 `semantic_feature` 칸. 동의어·한국어 검색, 같은 이름 물체 여럿 구분(빨간 컵 / 파란 컵) |
| 있으면 좋음 | 프롬프트 없이 "보이는 건 다" | 과제 밖 물체도 기억해야 "리모컨 어디 있어?" 에 답한다 |
| 있으면 좋음 | 가끔 쓰는 무거운 확인 | 새 물체 이름 확인, 놓친 과제 물체 다시 찾기(라디오 문제) |
| 있으면 좋음 | 성숙도 | 내보내기(export) 경로가 공식이고, Jetson·NPU 실사용 사례가 있어야 한다 |

## 2. 후보 20개

### 2.1 먼저 거른 것 (우리 조건에 안 맞아 20개에 넣지 않음)

| 모델 | 뺀 이유 |
|---|---|
| Grounding DINO 1.5 / 1.6 Edge, DINO-X | 가중치를 공개하지 않고 API 로만 쓸 수 있다 [11]. 로봇 안·평가기 안에서 못 돈다 |
| Grounded-SAM / Grounded-SAM-2, Lang-SAM | Grounding DINO-T + SAM 을 파이썬으로 이은 것이다. 같은 조합은 아래 #8(Mask Grounding DINO)·#10(EfficientViT-SAM)으로 본다 |
| Detic | Detectron2 + CenterNet2(Swin-B) 구조다. TensorRT 로 가는 공식 경로가 없고 무겁다 |
| OpenSeeD · X-Decoder · SEEM · APE | 연구 코드이고 무겁다(Swin/ViT-L). TensorRT·Jetson 실사용 사례가 없다 |
| OV-DINO, OWLv2 | 상자만 낸다. OWLv2 는 Orin 에서 400 ms 이상이라는 보고가 있다 [9] |
| FastSAM (+CLIP) | 우리 비교에서 이미 뺐다. 이름 정밀도 0.21, 물체가 조각남(평균 2.9개) [ovdet_검출기.md] |
| SAM 2 base/large | 이름을 주지 않고 무겁다. 추적 용도는 #12 EdgeTAM 이 대신한다 |
| CLIP ViT-B (OpenAI) | SigLIP 2 · MobileCLIP 2 에 밀린다 |
| PaliGemma 2, Moondream | 이름 확인 역할은 #15·#16(Qwen)이 같은 일을 더 잘하고, 이미 4090 에 있다. PaliGemma 는 Gemma 라이선스다 |
| DINOv3 | 별도 라이선스이고, 우리 용도(외형 re-ID)에는 DINOv2 로 충분하다 |
| DOSOD | 상자만 낸다. 주로 Horizon RDK 용이고 사용자가 적다 [21] |

### 2.2 남긴 20개

역할은 세 가지다. ① 실시간 등록, ② crop 임베딩, ③ 가끔 하는 확인·이름 붙이기.

| # | 모델 | 종류 | 크기 | 마스크 | 어휘 | 속도 (출처) | TRT / ONNX / DX-COM | 라이선스 | 남김·뺌 한 줄 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **YOLOE-11 s/m/l** (지금 쓰는 것) | 열린 어휘 seg | 10.7–26M | 있음 | text 프롬프트(우리 272개) | 우리 측정: 11m p50 4.6 ms, 176 MB(다시 잴 것). Orin NX 에서 11s-seg 1280×720 TRT 추론 14 ms [8] | TRT 됨(ovdet). DX-COM 은 공식 목록에 없다 [6] | AGPL-3.0 | **기준선**. 라디오를 못 찾고 coffee table 을 floor 로 부른다 |
| 2 | **YOLOE-26 n/s/m/l/x** | 열린 어휘 seg, NMS 없음 | 26s 10.7M, 26l 25.5M | 있음 | text / visual / 프롬프트 없음 | LVIS mAP 26s 30.8, 26l 37.8(11s 는 27.5) [2]. YOLO26n 검출이 Orin Nano Super 에서 4.57 ms [7] | TRT·ONNX 공식 [2]. 프롬프트는 export 때 굳는다 | AGPL-3.0 | **남김(1위)**. 같은 크기에서 11보다 정확하고, NMS 가 없어 후처리가 짧다 |
| 3 | YOLOE-26 프롬프트 없음 모드 | 큰 어휘 seg | 위와 같음 | 있음 | 내장 4,585개(RAM++ 태그) | AP 가 text 모드보다 낮다(26x 31.1) [1] | TRT 됨(추정). 분류 머리가 4,585개라 커진다 | AGPL-3.0 | 남김. "과제 밖 물체도 기억" 용. 이름이 흔들리기 쉬워 #2 의 보조로 쓴다 |
| 4 | YOLO-World-v2 (+YOLO-World-Seg) | 열린 어휘 det(+seg) | S–L | Seg 판만 | text | TRT FP16 v2-S 1221 FPS(출처 기기) [10] | TRT 됨 | **GPL-3.0**(원본), Ultralytics 판은 AGPL | 뺌. YOLOE 가 같은 방식에 같은 크기에서 더 정확하고 빠르다 [1]. Ultralytics 판은 상자만 낸다 |
| 5 | **YOLO26-seg / YOLO11-seg** + 미세조정 | 닫힌 어휘 seg | n 2–3M, s 약 10M | 있음 | 우리가 학습한 N개 | RPi5 + DX-M1 에서 n-seg end-to-end 55 FPS(Full HD) [5]. Jetson Nano 에서 v8n-seg 4.2 FPS [12] | **DX-COM 공식**(`format=deepx`, seg 지원) [6]. TRT 8.2 도 됨 | AGPL-3.0 | **남김(2위)**. DX-M1·Nano 에서 유일하게 확실한 경로. 학습 데이터는 시뮬 정답 마스크 + YOLOE 의사 라벨 |
| 6 | RF-DETR-Seg | 닫힌 어휘 seg (DINOv2 백본) | Nano–2XL | 있음 | COCO → 미세조정 | Nano: T4 TRT 3.4 ms, COCO AP 40.3 [13]. Orin NX 에서 det-Nano 7.6 ms [14] | TRT·ONNX 공식. DX-COM 은 미확인 | **Apache-2.0** | 남김(예비). AGPL 을 피하고 싶어질 때 #5 를 대신한다. 다만 열린 어휘가 아니다 |
| 7 | NanoOWL (OWL-ViT B/32) | 열린 어휘 det | 약 150M (추정) | 없음 → #11 이 필요 | text, tree 프롬프트 | AGX Orin 95 FPS, Orin Nano 는 미측정(TBD) [9] | TRT(torch2trt) | Apache-2.0 | 뺌. 상자만 내고 COCO AP 28 이다. Orin Nano 숫자가 없다 |
| 8 | Mask Grounding DINO (NVIDIA TAO) | 열린 어휘 seg | Swin-T 급 | 있음 | text | AGX Orin 에서 GDINO-T 약 12 FPS [15] | TAO 가 TRT 엔진을 만든다 | NVIDIA 모델 라이선스(상용 가능) [16] | 뺌. 시뮬 PC 예산(수 ms)과 Orin Nano 에 무겁다. 4090 확인용으로는 #13 이 낫다 |
| 9 | OmDet-Turbo | 열린 어휘 det | B 약 170M (추정) | 없음 | text(언어 캐시) | TRT 로 100 FPS(출처 기기) [17] | ONNX·TRT | Apache-2.0 | 뺌. 상자만 내므로 마스크를 위해 망이 둘이 된다 |
| 10 | EfficientViT-SAM L0 | 프롬프트 분할기 | 34.8M | 있음(상자 프롬프트) | 이름 없음 | AGX Orin TRT FP16 8.2 ms [18] | TRT 공식 | Apache-2.0 | 남김(보조). 상자만 내는 검출기(#6 det, #9)와 짝지을 때 쓴다 |
| 11 | NanoSAM / MobileSAM | 프롬프트 분할기 | ResNet18 / 9.7M | 있음 | 이름 없음 | NanoSAM: Orin Nano 전체 27 ms. MobileSAM: 146 ms [19] | TRT | Apache-2.0 | 뺌. #10 과 역할이 겹치고, YOLOE 가 이미 마스크를 낸다 |
| 12 | EdgeTAM (SAM 2 경량) | 영상 추적 분할 | 약 14M (추정) | 있음 | 이름 없음 | iPhone 15 Pro Max 16 FPS [20] | ONNX(추정) | Apache-2.0 | 뺌. ID 안정성은 DA 로 푼다. 손목 카메라로 잡을 물체를 추적할 때 다시 본다 |
| 13 | **SAM 3 / 3.1** | 개념 프롬프트 seg + 추적 | 848M [3] | 있음 | 열린 개념(학습 4M 개념) | H200 에서 이미지당 30 ms, 물체 100개 이상(논문 주장) [3] | 공식 TRT 없음. Jetson AGX 양자화판 있음 [4] | SAM License(상용 가능, 가중치 신청 필요) | **남김(4위)**. 확인·오프라인 라벨용 최강. 실시간에는 크다 |
| 14 | Florence-2 base/large | 작은 VLM (OVD·영역 캡션·seg) | 0.23B / 0.77B | 다각형 seg | 자유 문장 | Jetson 미지원(Roboflow 기준) [22] | ONNX 있음(인코더-디코더라 TRT 는 손이 많이 감) | MIT | 남김(예비). 4090 없이 시뮬 PC 에서 이름을 확인할 대안 |
| 15 | **Qwen3.5-9B (비전 입력)** | VLM | 9B | 없음(이름·판단) | 자유 | 이미 4090 API 에서 돈다. 이미지 입력을 지원한다 [23] | 서버 쪽 | Apache-2.0 | **남김(5위)**. 새로 설치할 것이 없고 agent 와 같은 모델이다 |
| 16 | Qwen3-VL-2B / Qwen3.5 소형 | 작은 VLM | 2B | 없음 | 자유 | Orin Nano Super 에서 0.53–0.89 질의/s [24] | TRT Edge-LLM, llama.cpp | Apache-2.0 | 남김(예비). 4090 에 못 닿을 때 로봇 안에서 확인 |
| 17 | RAM++ (Recognize Anything) | 이미지 태거 | Swin-L 급 (추정) | 없음 | 4,585개 | 무겁다(추정) | ONNX(커뮤니티) | Apache-2.0 | 뺌. 같은 태그 목록을 #3 이 이미 품고 있다 |
| 18 | YOLOE 물체 임베딩 (#1·#2 의 머리 출력) | 검출마다 임베딩 | +0 | — | text 공간과 정렬 | 추가 비용 거의 없음 | 엔진 출력을 하나 더 내면 된다(추정) | AGPL | 남김(첫 시도). 따로 망을 두기 전에 이걸로 동의어 검색을 해 본다 |
| 19 | MobileCLIP 2 S0/S2 | 임베딩 | 영상 쪽 약 12M(S0) | — | 영어 | S0 영상 1.5 ms, 글 1.6 ms(iPhone) [25] | ONNX 있음. FastViT 계열은 DX 모델 동물원에 있다 [26] | **apple-amlr = 연구 전용, 비상업** [27] | 뺌. 공개 데모에 걸림돌이고 한국어 글을 못 넣는다 |
| 20 | **SigLIP 2** B/16 (+so400m) | 임베딩(다국어) | B 약 86M(영상 쪽) | — | 다국어 글 | so400m 의 TRT FP16 측정이 NVIDIA 에 있다 [28]. B/16 은 RTX 에서 crop 당 1–2 ms (추정) | ONNX·TRT(NVIDIA TAO 판) [28]. ViT 는 DX 동물원에 있다 [26] | Apache-2.0 | **남김(3위)**. 한국어 질의를 바로 쓰고 라이선스가 깨끗하다 |
| (참고) | DINOv2 ViT-S/14 | 외형 특징 | 21M | — | — | 빠르다 | TRT 됨 | Apache-2.0 | 같은 이름 물체 여러 개를 구분하는 re-ID. SigLIP 2 로 부족할 때만 쓴다 |

## 3. 최종 5개

### 3.1 YOLOE-26-seg (s 또는 m, text prompt) — 실시간 등록

| 항목 | 내용 |
|---|---|
| 역할 | 매 keyframe(약 6 step)마다 마스크와 이름을 낸다. 지금 ovdet 의 YOLOE-11m 을 바꿀 후보 |
| 대상 | 시뮬 PC(5070 Ti, TRT 10.16), Orin Nano 8GB(JetPack 6, TRT 10) |
| 지연·메모리 | 시뮬 PC: 26s 640 입력 2–3 ms, 26m 3–5 ms, 200 MB 안팎(추정. 11m 측정이 4.6 ms·176 MB). Orin Nano: 26s-seg 640 FP16 15–25 ms(추정. 근거는 YOLO26n det 4.57 ms [7], 11s-seg Orin NX 14 ms [8]) → 5–10 Hz |
| ovdet 연결 | `tools/export_yoloe.py` 에서 모델 이름만 바꾼다. 출력 배치가 다르다(NMS 없는 머리는 top-k 출력) → `ovdet.cu` 디코더에 분기를 넣거나 `end2end=False` 로 내보내 지금 배치를 유지한다(추정). 손이 덜 가는 편이다 |
| 어휘 | 272개 + **동의어 줄**: 'radio receiver' 에 'radio', 'portable radio' 를 더하고, 결과는 같은 prompt 칸으로 모은다. 라디오 미검출 대책([ovdet_검출기.md] 5절 1번)을 그대로 적용한다 |
| 덤 | 검출마다 물체 임베딩(#18)을 내 노드 `semantic_feature` 에 넣는다. 추가 비용은 거의 없다(추정) |
| 위험 | AGPL(이미 받아들임). 프롬프트가 export 때 굳으므로 어휘를 바꾸면 엔진을 다시 만들어야 한다. 낮은 conf 에서 조각남·이름 흔들림이 생긴다. YOLOE 글 인코더(MobileCLIP 계열)의 라이선스는 export 단계에만 걸린다 → 확인 필요 |

### 3.2 YOLO26-seg (n/s) + 우리 어휘로 미세조정 — DX-M1 · Jetson Nano

| 항목 | 내용 |
|---|---|
| 역할 | DX-M1(라즈베리파이 5) 또는 Jetson Nano 4GB 에서 실시간 등록. 열린 어휘 대신 **우리 물체 N개(약 100–300)로 학습한 닫힌 어휘** |
| 왜 | DEEPX 공식 export(`format=deepx`)는 YOLOv8·11·26 의 seg 를 지원하고 **YOLOE 는 목록에 없다** [6]. DX 모델 동물원의 인스턴스 분할도 YOLOv5/v8/26-Seg 뿐이다 [26] |
| 지연·메모리 | RPi5 + DX-M1: n-seg end-to-end 55 FPS(Full HD 영상 파이프라인) [5]. n 검출 34.6 ms/장(Ultralytics 측정) [6]. 모델 수 MB. Jetson Nano(TRT 8.2): v8n-seg 4.2 FPS [12] → 2–4 Hz |
| 학습 데이터 | 시뮬 정답 마스크(`gt_scene` 라벨) + 실내 실사진에 YOLOE-26l·SAM 3 를 돌려 만든 의사 라벨(#13). 4090 에서 학습 |
| ovdet 연결 | 백엔드를 하나 더 둔다(`OvdBackend`: TRT / DX-RT). DX-RT C++ API 로 `.dxnn` 을 부르고, 출력 해석은 지금 CUDA 후처리를 CPU 판으로 옮긴다. `sm_detections` 계약은 그대로다. 손이 중간쯤 간다 |
| 위험 | INT8 강제라 작은 물체 정확도가 떨어진다(교정 이미지가 중요). 학습하지 않은 물체는 못 잡는다(→ 4090 확인으로 보충). DX-COM export 는 x86-64 Linux 에서만 된다 [6]. Jetson Nano 는 TRT 8.2·opset 이 오래돼 YOLO26 export 가 될지 확인해야 한다(추정) |
| 대안 | 라이선스를 바꾸고 싶으면 RF-DETR-Seg Nano(Apache)를 쓴다. DX-COM 이 될지는 직접 시험해야 한다 |

### 3.3 SigLIP 2 B/16 (다국어) — crop 임베딩

| 항목 | 내용 |
|---|---|
| 역할 | best view 가 바뀐 노드만 crop 을 임베딩해 `semantic_feature` 에 넣는다. agent 의 `find_object` 가 "머그잔"·"빨간 컵" 을 글 임베딩과 비교해 후보를 넓힌다. 같은 이름 물체를 색·모양으로 구분하는 데도 쓴다 |
| 대상 | 시뮬 PC(TRT), Orin(TRT). DX-M1 은 ViT 계열이 동물원에 있어 될 것으로 본다(추정) [26] |
| 지연·메모리 | 영상 쪽만 올린다. RTX: crop 당 1–2 ms, 200 MB 안팎(추정). Orin Nano: crop 당 10–20 ms(추정). 매 프레임이 아니라 **best view 를 고칠 때만**(초당 몇 번) 돈다 |
| 글 쪽 | 어휘 272개 + 한국어 동의어 사전은 **오프라인에서 미리 임베딩해 파일로** 둔다. 자유 문장 질의는 4090 쪽(agent 서버)에서 글 인코더를 부르거나, agent 계획에 있는 Qwen3-Embedding 경로와 합친다 |
| 연결 | 새 작은 라이브러리(`crop_embed`, TRT 엔진 하나)를 만든다. scenemap 의 best view 갱신 hook 에서 부른다. 손이 중간쯤 간다 |
| 위험 | 시뮬 물체(렌더링)와 실사진의 영역 차이. crop 이 작으면(손목 카메라 밖) 품질이 떨어진다. 먼저 #18(YOLOE 임베딩)로 충분한지 재고, 모자라면 이걸 붙인다 |

### 3.4 SAM 3 / 3.1 — 가끔 하는 확인 · 오프라인 라벨

| 항목 | 내용 |
|---|---|
| 역할 | (a) 과제 물체를 N 초 동안 못 찾으면 그 이름으로 개념 분할을 한 번 돌린다(라디오 같은 경우). (b) 녹화본·시연 영상에 돌려 **의사 정답과 #5 학습 라벨**을 만든다 |
| 대상 | 학교 4090(밖) 또는 시뮬 PC 의 **오프라인** 시간(Isaac Sim 이 꺼져 있을 때). 평가 중 시뮬 PC 에는 남는 메모리(약 2 GB)가 모자란다(FP16 가중치만 1.7 GB, 추정) |
| 지연 | H200 에서 이미지당 30 ms(논문 주장) [3]. 4090 에서 이미지당 약 100 ms(추정). 실시간 경로 밖 |
| 연결 | 런타임에 직접 넣지 않는다. 4090 에 확인 서비스로 두거나(agent 의 도구 하나), 오프라인 라벨 도구로 쓴다(파이썬 가능) |
| 위험 | 가중치 신청이 필요하고 SAM License 다(배포 조건 확인). 848M 이라 Orin 에는 무리다. 대회 제출 Docker 안에서는 못 쓴다(메모리) → 대회에서는 오프라인 라벨 용도만 |

### 3.5 Qwen3.5-9B 비전 입력 — 이름 확인

| 항목 | 내용 |
|---|---|
| 역할 | 새로 등록한 노드, 또는 이름이 프레임마다 흔들리는 노드(`cup` ↔ `bowl`)의 best-view crop 을 보내 "이게 무엇인지, 우리 어휘 중 무엇인지" 를 묻는다. 결과는 노드 이름·메모로 고쳐 적는다(팀 문서의 "LLM 이 기억에 '사실 그릇'이라고 고쳐 적는다" 를 이미지로 근거 있게 한다) |
| 대상 | 학교 4090 API(이미 돌고 있음) [23]. 로봇이 4090 에 못 닿으면 Qwen3-VL-2B / Qwen3.5 소형을 Orin 에(#16) |
| 지연 | 질의당 0.3–1 s(추정). 비동기로 돌리고 실시간 경로 밖 |
| 연결 | agent 쪽에 `verify_object(id)` 도구를 하나 더한다. 기억에서 crop PNG 를 읽어 보낸다. 손이 거의 안 간다 |
| 위험 | 4090 하나를 계획 LLM 과 나눠 쓴다(대기 시간). 대회 평가 중에는 KAU API 사용 규칙을 확인해야 한다. VLM 도 작은 crop 에서 틀린다 → 확신도가 낮으면 이름을 바꾸지 않는다 |

### 3.6 대상별로 정리

| 대상 | 실시간 등록 | 임베딩 | 확인 |
|---|---|---|---|
| 시뮬 PC (5070 Ti) | YOLOE-26 s/m (지금 11m) | YOLOE 임베딩 → SigLIP 2 | Qwen3.5-9B(4090). SAM 3 는 오프라인 |
| LIMO Pro (Orin Nano 8GB) | YOLOE-26 s | SigLIP 2 (best view 때만) | Qwen3.5-9B(4090), 없으면 Qwen3-VL-2B |
| LIMO 기본 (Jetson Nano 4GB) | YOLO26n-seg 미세조정 (TRT 8.2) | 없음, 또는 4090 에서 | 4090 |
| RPi5 + DX-M1 | YOLO26 n/s-seg 미세조정 (DX-COM INT8) | SigLIP 2 DX-COM(추정) | 4090 |
| 4090 (밖) | — | 글 임베딩 | SAM 3, Qwen3.5-9B |

## 4. 추천 조합과 비교 계획

### 4.1 조합

| | A. 기본 (시뮬 PC · Orin) | B. 가벼운 기기 (DX-M1 · Nano) |
|---|---|---|
| 실시간 | YOLOE-26-seg text prompt(과제 물체 + 구조물 18 + 동의어) | YOLO26-seg 미세조정(우리 물체 N개), INT8 |
| 프롬프트 밖 | 가끔(예: 10 keyframe 마다) YOLOE-26 프롬프트 없음 모드로 "과제 밖 물체" 등록. 같은 엔진은 아니다 | 없음 |
| 임베딩 | YOLOE 물체 임베딩으로 시작한다. 모자라면 best view 갱신 때 SigLIP 2 | 4090 에서 crop 을 받아 SigLIP 2 |
| 확인 | 새 노드, 이름 흔들림, 과제 물체를 오래 못 찾음 → Qwen3.5-9B(4090). 라벨 만들기는 SAM 3 | 같음 |
| 메모리 | 시뮬 PC 기준 약 0.4 GB (추정) — 남는 2 GB 안 | DX-M1 4 GB 중 수십 MB (π0.5 와 나눔) |

### 4.2 비교 계획 (우리 데이터)

| 단계 | 내용 |
|---|---|
| 데이터 | ① `src/behavior-2026/outputs/*/videos/*.mp4` 머리 영상(turning_on_radio_301, bringing_water_301). ② `src/behavior-2026/data/2026-challenge-demos/videos/observation.rgb.zed_link_camera_0/chunk-000`(머리 zed, 같은 시연의 `depth_linear` 와 짝). ③ 지금 비교에 쓰는 ep0·ep200 프레임(정답 `gt_scene` 이 있음) |
| 후보 | YOLOE-11m(기준) · YOLOE-26s · YOLOE-26m · YOLOE-26 프롬프트 없음 · YOLO26s-seg 미세조정. 확인은 위 출력에 SAM 3 · Qwen3.5-9B 를 덧붙인 것 |
| 지표 (프레임) | 과제 물체 재현율(마스크 IoU ≥ 0.5 + 이름 맞음, BDDL 조상 synset 인정). 오검출 = 프레임당 정답과 안 겹치거나 이름이 틀린 비구조물 검출 수. 이름 정밀도. 조각남 |
| 지표 (시간) | **이름 안정성**: 한 정답 물체에 붙은 검출 이름이 프레임 사이에 바뀌는 비율. DA 를 거친 뒤 같은 물체가 노드 둘 이상이 된 수 |
| 지표 (비용) | ms/frame p50·p99, **Isaac Sim + π0.5 를 같이 띄운 상태**의 VRAM(nvidia-smi 최대치). Orin·DX-M1 은 같은 프레임을 오프라인으로 재생해 잰다 |
| 임베딩 | 질의 100개(과제 물체 × 영어 동의어 · 한국어 이름, 예: "머그잔"→cup, "휴지통"→ashcan). top-1/top-5 로 맞는 노드를 찾는 비율. YOLOE 임베딩과 SigLIP 2 를 비교한다 |
| 팀 벤치마크 | `refs/code/dynamic-object-mapping-benchmark` office 시퀀스(`scenarios/office_classes.json`)에 scenemap 을 돌려 `map_timeline.csv` 로 낸다. 검출기만 바꿔 obj_id 유지·위치·옮김/사라짐/생김 점수를 비교한다 → 프레임 지표가 좋아도 지도 점수가 좋은지 확인 |
| 고르기 | 조건: 시뮬 PC p99 ≤ 8 ms, VRAM ≤ 0.5 GB. 이걸 통과한 것 중 과제 물체 재현율·이름 안정성이 가장 좋은 것을 고른다. 차이가 작으면 가벼운 것을 고른다(팀 규칙) |
| 기록 | `tests/sandbox/perception_bench/` 에 결과 JSON 과 표를 남기고, 팀이 `model_selection.md` 를 고친다 |

## 5. 출처

1. YOLOE: Real-Time Seeing Anything (arXiv 2503.07465) — https://arxiv.org/abs/2503.07465
2. Ultralytics YOLOE 문서(YOLOE-26, 4,585 어휘, export, set_classes 굳히기) — https://docs.ultralytics.com/models/yoloe
   - YOLOE-26 논문: https://arxiv.org/abs/2602.00168
3. SAM 3: Segment Anything with Concepts (arXiv 2511.16719), 저장소(848M, SAM 3.1, SAM License) — https://arxiv.org/abs/2511.16719 , https://github.com/facebookresearch/sam3
4. SAM 3 Jetson 양자화판(Embedl) — https://huggingface.co/embedl/sam3
5. DEEPX dx-benchmark YOLO26 분석(RPi5B_M1 seg 55.1 FPS) — `/tmp/claude-1000/dxas/dx-benchmark/docs/ANALYSIS_EN.md`, https://github.com/jyun69/dx-all-suite
6. Ultralytics DeepX export(YOLOv8/11/26, seg, INT8, x86-64 에서만 export) — https://docs.ultralytics.com/integrations/deepx
7. Ultralytics NVIDIA Jetson 가이드(YOLO26n Orin Nano Super 4.57 ms) — https://docs.ultralytics.com/guides/nvidia-jetson/
8. YOLOE Jetson TensorRT 느림 문의(Orin NX, 11s-seg 추론 14 ms) — https://community.ultralytics.com/t/yoloe-inference-very-slow-on-jetson-with-tensorrt/1443
9. NanoOWL — https://github.com/NVIDIA-AI-IOT/nanoowl , OWLv2 속도 보고: https://forums.developer.nvidia.com/t/nanoowl-inference-takes-more-time-than-nanoowl-official-github-shows/293051
10. YOLO-World(GPL-3.0, Seg, TRT FPS) — https://github.com/AILab-CVC/YOLO-World
11. Grounding DINO 1.5/1.6 API(API 로만 제공) — https://github.com/IDEA-Research/Grounding-DINO-1.5-API
12. Qengineering YoloV8 TensorRT Jetson Nano — https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano
13. RF-DETR Segmentation — https://blog.roboflow.com/rf-detr-segmentation/
14. RF-DETR on the edge(Orin NX) — https://blog.roboflow.com/rf-detr-for-the-edge/
15. NVIDIA Jetson Platform Services GDINO — https://docs.nvidia.com/jetson/jps/inference-services/gdino.html
16. TAO Mask Grounding DINO — https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/instance_segmentation/mask_grounding_dino.html , https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino
17. OmDet-Turbo (arXiv 2403.06892) — https://arxiv.org/abs/2403.06892 , https://github.com/om-ai-lab/OmDet
18. EfficientViT-SAM (arXiv 2402.05008) — https://arxiv.org/abs/2402.05008
19. NanoSAM — https://github.com/NVIDIA-AI-IOT/nanosam
20. EdgeTAM (CVPR 2025) — https://github.com/facebookresearch/EdgeTAM
21. DOSOD (arXiv 2412.14680) — https://arxiv.org/abs/2412.14680
22. Florence-2(Roboflow inference 문서) — https://inference-models.roboflow.com/models/florence2/
23. Qwen3.5-9B(비전 입력) — https://www.jetson-ai-lab.com/models/qwen3-5-9b , https://lmstudio.ai/models/qwen/qwen3.5-9b
24. Qwen3-VL-2B Orin Nano Super 성능 문의 — https://forums.developer.nvidia.com/t/performance-inquiry-optimizing-qwen3-vl-2b-inference-for-2-qps-target-on-orin-nano-super/359639
25. MobileCLIP / MobileCLIP 2 — https://github.com/apple/ml-mobileclip , https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP
26. DX-AllSuite 구조 개요(모델 동물원: ViT/DeiT/FastViT 분류, YOLOv5/v8/26-Seg) — `/tmp/claude-1000/dxas/docs/source/01_DX-AllSuite_Architecture_Overview.md`
27. apple-amlr 라이선스(연구 전용) — https://huggingface.co/apple/mobileclip2_coca_dfn2b_s13b_context77/blob/main/README.md
28. NVIDIA TAO SigLIP 2 — https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/siglip_v2
29. Recognize Anything (RAM++, Apache-2.0) — https://github.com/xinyu1205/recognize-anything
30. 우리 측정: `src/behavior-2026/docs/ovdet_검출기.md`, `src/behavior-2026/src/scene_graph/ovdet/README.md`
