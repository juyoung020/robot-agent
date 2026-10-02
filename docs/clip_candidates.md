# CLIP 류 임베딩 모델 후보 (리모 기본형 = Jetson Nano 단독)

작성 2026-10-03. 처음엔 조사·측정 문서였다. **2026-10-03 결정: SigLIP 2 B/32-256 을 쓴다.** C++/CUDA·TensorRT 포팅과 모델 자체 효율 측정은 [8절](#8-포팅-결과와-모델-효율-2026-10-03).

- 앞 문서: [물체 인식 모델 후보](perception_model_candidates.md) A′(Nano 단독). 거기서는 "별도 CLIP 은 기본에서 뺌"이었다.
  - 이 문서는 **새 방향**(FastSAM-s 로 이름 없는 마스크 → CLIP 으로 이름·임베딩)을 전제로 다시 본다.
- 표기
  - (추정): 직접 재지 않고 미루어 본 값. **Nano 숫자는 전부 추정이다.** Nano 에서 CLIP 을 잰 공개 숫자는 찾지 못했다.
  - (PC): RTX 5070 Ti, TensorRT 10.16, FP16, `trtexec` GPU 시간 중앙값, CUDA graph.
  - GFLOPs = 곱셈·덧셈 따로 센 값(2 × MAC).
- 측정 스크립트·결과는 `~/clip_bench/` 에 있다(커밋하지 않음). 가상환경은 `~/clip_venv`.

## 0. 한눈에

| 항목 | 결론 |
|---|---|
| **결정(10-03)** | **SigLIP 2 B/32-256 채택, PC 포팅 끝(8절)**: 엔진 = 패치 GEMM + FP16(LayerNorm·GELU FP32) + 배치 칸 1·2·4·8, PyTorch FP32 대비 코사인 평균 0.99987(567 crop, 98.8 % ≥ 0.999), 이름·찾기 정확도 FP32 와 같음. Nano(TRT 8.2) 빌드는 남음 |
| **1순위 영상 인코더** | **SigLIP 2 B/32-256** (Apache-2.0, 영상 쪽 95M, 768-d). 마스크를 MAP 풀링 주의집중에 log 가중치로 넣는 **두 번째 출력(emb_mask)** 을 ONNX 에 굳힌다 |
| 2순위(바로 쓸 수 있는 대안) | **OpenAI CLIP ViT-B/32** (MIT, 지금 meridian 과 같은 영상 공간) + **`Bingsu/clip-vit-base-patch32-ko`** 한국어 글 인코더(MIT, 영상 탑이 OpenAI B/32 와 비트까지 같음을 확인) |
| 연구용으로만 | MobileCLIP-S1 / MobileCLIP 2-S0. 우리 데이터 이름 정확도가 가장 높은 축이지만 가중치가 apple-amlr(비상업) |
| 우리 데이터 이름 붙이기 | 어휘 360개 기준 정답률 0.30–0.35(정답이 3D 상자에서 와서 잡음이 있다. 깨끗한 부분집합은 0.33–0.41). **라디오는 모든 모델이 0** — 이름 붙이기만으로는 부족하다 |
| 우리 데이터 글 → 물체 찾기 | SigLIP 2 B/32: 영어 R@1 0.70, **한국어 그대로 0.65**. "라디오"·"빨간 라디오"는 1·2위 안에 찾는다 → **임베딩 저장이 이름보다 중요** |
| 자르기 방법 | 상자 + 10 % 둘레(정사각, 회색 채움)가 기본. **마스크 밖을 회색으로 지우면 크게 나빠진다**(0.30 → 0.11). 패치 풀링은 ViT CLIP 에선 쓸모없고(0.02), SigLIP MAP 풀링·FastViT 평균 풀링에서만 좋아진다 |
| PC 속도 | B/32 류 영상 인코더 배치 1/8/32 = 0.47/1.1/3.1–3.3 ms. FastViT(MobileCLIP)는 연산량이 반인데도 GPU 에서 더 빠르지 않다 |
| Nano 속도 (추정) | SigLIP 2 B/32: 물체당 50–70 ms(배치 8), OpenAI B/32: 37–53 ms. FastSAM-s 416: 80–150 ms. 새 물체 5개 keyframe ≈ 0.35–0.5 s, 20개 ≈ 1.1–1.6 s(비동기) |
| FP16 함정 | **SigLIP 2 는 TRT FP16 그대로면 망가진다**(코사인 0.64–0.74). LayerNorm 부분을 FP32 로 고정하면 0.9999, PC 지연 차이 없음. TRT 8.2 에서도 같은 처리가 필요하다 |
| 라벨 찾기 | 28.8k 라벨에서 128-d INT8 IVF + 2진 부호 → 상위 32개만 768-d 로 다시 매기면 **PC 단일 스레드 6–8 µs/물체**, 이름 정확도 손실 없음. Nano A57 20–100 µs(추정). keyframe 비용의 1 % 미만 |

## 1. 요구와 파이프라인 가정

- **FastSAM-s**(YOLOv8s-seg 구조, 이름 없음)가 keyframe 에서만 마스크를 낸다.
  - 엔진: `~/ovdet_models/x86_sm120/FastSAM-s-{640,416}.plan`. 출력 배치는 YOLO-seg 와 같다(4 상자 + 1 클래스 + 32 계수, proto 32 × H/4 × W/4). 이름 파일 = `object`.
  - ovdet 의 CUDA NMS 를 그대로 쓴다. PC 에서 416 엔진 전체 p50 0.82 ms, GPU 58 MB(이번 측정).
- scenemap 이 물체마다 best view 를 들고 있다. `png_dirty` 가 "best view 가 바뀜"을 알린다(`dsg_save.hpp`).
- CLIP 은 **새 물체 / best view 가 바뀐 물체만**, keyframe 마다 **묶어서, 비동기로** 돈다. 매 프레임 돌지 않는다.
- agent(Qwen3.5-9B, KAU API, 로봇 밖)가 쓰는 것 두 가지:
  1. **이름 붙이기**: 마스크마다 어휘 중 하나. 영상 임베딩 ↔ 미리 계산한 라벨 글 임베딩. 구조물(벽·바닥·천장·문·창)은 걸러낸다. 한국어 이름은 짝지은 표로.
  2. **임베딩 찾기**: 물체마다 영상 임베딩을 저장. "빨간 컵"·"radio"·"흰 의자" 같은 자유 글과 코사인으로 비교.
- 그래서 **영상·글 인코더가 정렬된 모델**이어야 한다. 글 인코더는 Nano 또는 로봇 밖 어디서 돌아도 된다(2.3).
- 기준선: 팀 `neoul-ro/meridian_frontend`
  - OpenAI CLIP ViT-B/32 영상 쪽, FP16 TRT, 224 입력, 512-d.
  - 출력 둘: CLS(`query_emb`)와 마스크 가중 패치 풀링(`id_emb`, 7 × 7 가중치 `wpatch` 입력).
  - GPU 에서 `roi_align` 으로 상자를 224 × 224 로 늘려 자른다(가로세로 비 무시, 둘레 없음).
  - keyframe 발화 때 새 tracklet 만 묶어 한 번에 넣는다. 워커 스레드 + 별도 스트림, TensorRT 10 API(`set_tensor_address`, `execute_async_v3`), Jetson Orin.

## 2. 후보

### 2.1 후보 표 (19개)

품질 숫자(IN = ImageNet zero-shot top-1, Avg38 = DataComp 38과제 평균)는 출처 [MC]·[OCr]·[S2]·[TC]·[PE]·[OV]. PC 지연은 이번 측정. 연산량은 open_clip 표[OCp] 또는 이번 측정(`torch.utils.flop_counter`), 둘 다 없으면 토큰 수로 셈(추정).

| # | 모델 | 입력 | 영상 쪽 크기 | GFLOPs | IN | Avg38 | 글 쪽 | 한국어 | 가중치 라이선스 | PC FP16 b1 / b8 / b32 ms | 남김·뺌 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenAI ViT-B/32 (기준선) | 224 | 88M | 8.8 | 63.3 | 52.5 | 63M, 영어 | 글만 바꾸면 됨(2.3) | MIT | 0.47 / 1.06 / 3.06 | **남김(2순위)** |
| 2 | DataComp-XL ViT-B/32 | 224 | 88M | 8.8 | 69.2 | 58.0 | 63M, 영어 | 없음 | MIT | (1번과 같은 구조) | 품질은 1번보다 낫지만 한국어 글이 없다 |
| 3 | DataComp ViT-B/32-256 (s34b) | 256 | 88M | 11.5 | 72.8 | 60.9 | 63M, 영어 | 없음 | MIT | 0.50 / 1.19 / 3.52 | DX-M1 모델 동물원에 있는 유일한 작은 CLIP(4.4) |
| 4 | OpenAI RN50 | 224 | 38M | 12.2 | 59.8 | 48.1 | 63M | 없음 | MIT | 0.36 / 1.05 / 3.85 | 뺌. 우리 데이터 최하위(0.12) |
| 5 | **SigLIP 2 B/32-256** | 256 | 95M | 11.4(추정, 64 토큰) | 74.0 | – | 282M(그중 197M 은 256k 어휘 표), Gemma 토크나이저 | **그대로 됨** | **Apache-2.0** | 0.48 / 1.12 / 3.30 | **남김(1순위)** |
| 6 | SigLIP 2 B/16-224 | 224 | 93M | 35.4 | 78.2 | – | 282M | 그대로 됨 | Apache-2.0 | 0.70 / 2.80 / 9.28 | 뺌. 연산량 3배, 우리 데이터 이득 작음 |
| 7 | MobileCLIP-S1 | 256 | 21.5M | 9.3 | 72.6 | 61.3 | 63M, 영어 | 없음 | apple-amlr(연구 전용) | 0.72 / 1.99 / 6.73 | 연구용만. 우리 데이터 이름 1위권 |
| 8 | MobileCLIP-S2 / MobileCLIP 2-S2 | 256 | 35.7M | 15.7 | 74.4 / 77.2 | 63.7 / 64.1 | 63M | 없음 | apple-amlr | 1.03 / 2.78 / 10.10 | 뺌(느림·라이선스) |
| 9 | **MobileCLIP 2-S0** | 256 | 11.4M | 4.8 | 71.5 | 59.7 | 63M | 없음 | apple-amlr | 0.42 / 1.14 / 3.59 | 연구용만. 연산량 최소인데 GPU 에선 B/32 보다 빠르지 않다 |
| 10 | MobileCLIP 2-B | 224 | 86M | 35.4 | 79.4 | 65.8 | 63M | 없음 | apple-amlr | 0.66 / 2.76 / 9.32 | 뺌 |
| 11 | TinyCLIP ViT-8M/16 | 224 | 8.3M | 3.6 | 41.1 | – | 15M | 없음 | MIT | 0.19 / 0.49 / 1.41 | 뺌. 우리 데이터 0.14 |
| 12 | TinyCLIP ViT-39M/16 | 224 | 39M | 16.0 | 63.5 | – | 44M | 없음 | MIT | 0.42 / 1.40 / 4.50 | 뺌 |
| 13 | TinyCLIP ViT-40M/32 | 224 | 40M | 4.0 | 59.8 | 51.2 | 44M | 없음 | MIT | 0.35 / 0.62 / 1.44 | 예비(속도 최후 수단) |
| 14 | TinyCLIP ViT-61M/32 | 224 | 61M | 6.2 | 62.4 | 53.0 | 54M | 없음 | MIT | 0.39 / 0.79 / 1.88 | 예비. B/32 보다 Nano 에서 약 30 % 빠를 것(추정) |
| 15 | EVA02-CLIP B/16 | 224 | 86M | 35.1 | 74.7 | 58.9 | 63M | 없음 | MIT | 0.69 / 3.21 / 10.96 | 뺌(연산량, RoPE) |
| 16 | PE-Core T/16-384 | 384 | 6M | 9.2(추정, 577 토큰) | 62.1 | – | 40M | 없음 | Apache-2.0 | 0.36 / 1.24 / 4.69 | 뺌. 384 고정이라 토큰이 많다 |
| 17 | PE-Core S/16-384 | 384 | 24M | 31(추정) | 72.7 | – | 40M | 없음 | Apache-2.0 | 0.58 / 2.96 / 11.64 | 뺌 |
| 18 | xlm-roberta-base ViT-B/32 (LAION-5B) | 224 | 88M | 8.8 | 62.4 | 56.4 | 278M XLM-R | 그대로 됨 | MIT | (1번과 같은 구조) | 뺌. 한국어는 되지만 SigLIP 2 보다 다 낮다 |
| 19 | NLLB-CLIP base(-SigLIP) | 224 | 88M | 8.8 | – | – | NLLB 600M | 됨(201 언어) | **CC-BY-NC** | – | 뺌(라이선스, 글 쪽 너무 큼) |

- 살펴보고 넣지 않은 것: jina-clip-v2·AltCLIP·SigLIP so400m(너무 큼), OpenVision Ti/S(ONNX 없음, IN 49.6/65.9), koclip(COCO 8만 장만으로 학습해 약함), NanoOWL(상자 검출기, Orin 전용).
- **Jetson 숫자**
  - Orin Nano(원판 → Super)에서 HF clip-vit-base-patch32 196 → 314 fps, patch16 95 → 161 fps [JAL].
  - 원판 Jetson Nano 의 CLIP 숫자는 없다. Nano TensorRT FP16 트랜스포머 숫자는 MobileViT-S(2.0 GMAC @256) 53 ms, EdgeNeXt-S 48.8 ms 뿐이다 [EN].
- 모든 후보 영상 인코더는 **opset 13 ONNX 로 내보내진다**(이번 확인). 쓰인 연산은 Conv·MatMul·Gemm·Softmax·ReduceMean·Pow·Sqrt·Erf/Tanh·Where·Gather 등 기본뿐이라 TRT 8.2 파서 범위 안이다(추정, Nano 에서 확인 필요).
  - 걸림돌 둘: torch 의 `scaled_dot_product_attention` 은 opset 13 으로 못 나간다 → 수동 주의집중으로 바꿔 내보냄(`sdpa_patch.py`). timm 은 `TIMM_FUSED_ATTN=0`.
  - MobileCLIP 은 export 전에 `reparameterize_model()` 이 필요하다.

### 2.2 우리 데이터 품질

**데이터**
- (가) FastSAM-s 416(ovdet C API, 클래스 무관) 를 BEHAVIOR 시연 영상(turning_on_radio ep 0·57·133, 135장)에 돌렸다.
  - 마스크가 60 % 이상 한 정답 물체에 들어가고 IoU ≥ 0.2, 면적 ≥ 600 px 인 것을 정답 이름으로 썼다. 범주당 최대 60개, 542개.
  - 정답은 `gt_scene` 3D 상자에서 온다(자세는 오도메트리 적분). 그래서 **잡음이 있다**(조명 상자에 로봇 손이 들어가는 등). IoU ≥ 0.5 부분집합(166개)을 "깨끗"으로 따로 센다.
  - 정답 이름 인정: synset + BDDL 조상 이름 + 별칭(couch→sofa, tv→television 등) + 끝 단어 일치.
- (나) 시뮬 기억 best view 25개(`outputs/mem_yolo26s_20261003_044921`, `mem_seg_20261003_042852`). 눈으로 정답을 붙였다(흰 S자 의자, 바 스툴, 탁자, 러그, 로봇 팔 …).
- 어휘: `vocab_all.txt`(272, 구조물 포함) + COCO-80 + 8개(stool, robot arm, carpet …) = 약 360개. 문장 틀 4개 평균.
- 글 → 물체 찾기: 물체마다 best crop 1장(53개) 갤러리에 질의 20개(영어 / 한국어 짝, 예 "radio"/"라디오", "white chair"/"흰 의자", "blue couch"/"파란 소파").
  - 질의 20개라 한 질의가 0.05 다. **방향만 본다.**

**결과 (모델별 가장 좋은 자르기)**

| 모델 | 자르기 | 이름: 시연 / 깨끗 / 기억 | 구조물 재현 / 오탐 | 찾기 영어 R@1 / MRR | 찾기 한국어(모델 자체 글) R@1 / MRR |
|---|---|---|---|---|---|
| OpenAI B/32 | 상자 | 0.28 / 0.30 / 0.20 | 0.11 / 0.19 | 0.50 / 0.62 | 0.05 (영어 전용) |
| OpenAI B/32 | 늘림(meridian) | 0.21 / 0.23 / 0.32 | 0.26 / 0.14 | 0.60 / 0.69 | 0.15 |
| DataComp B/32 | 상자 | 0.31 / 0.34 / 0.32 | 0.46 / 0.24 | 0.60 / 0.73 | 0.00 |
| DataComp B/32-256 | 늘림 | 0.30 / 0.34 / 0.36 | 0.25 / 0.16 | 0.65 / 0.75 | 0.05 |
| **SigLIP 2 B/32** | **마스크 MAP** | **0.34 / 0.40 / 0.40** | 0.55 / 0.17 | **0.70 / 0.83** | **0.65 / 0.75** |
| SigLIP 2 B/32 | 상자 / 늘림 | 0.30 / 0.33 / 0.40 · 0.24 / 0.27 / 0.56 | 0.45 / 0.25 | 0.60 / 0.75 · 0.75 / 0.85 | 0.60 / 0.73 |
| SigLIP 2 B/16 | 늘림 | 0.28 / 0.33 / 0.44 | 0.48 / 0.23 | 0.85 / 0.90 | 0.55 / 0.69 |
| MobileCLIP-S1 | 마스크 평균 | 0.35 / 0.41 / 0.52 | 0.45 / 0.15 | 0.80 / 0.86 | 0.05 |
| MobileCLIP 2-S0 | 마스크 평균 | 0.31 / 0.35 / 0.40 | 0.73 / 0.19 | 0.85 / 0.92 | 0.05 |
| MobileCLIP 2-S2 | 마스크 평균 | 0.32 / 0.36 / 0.48 | 0.59 / 0.21 | 0.85 / 0.90 | 0.05 |
| PE-Core S/16 | 상자 | 0.31 / 0.35 / 0.36 | 0.28 / 0.20 | 0.65 / 0.75 | 0.05 |
| PE-Core T/16 | 상자 | 0.29 / 0.34 / 0.20 | 0.24 / 0.14 | 0.60 / 0.74 | 0.05 |
| TinyCLIP 61M/32 | 상자 | 0.27 / 0.31 / 0.08 | 0.49 / 0.26 | 0.50 / 0.62 | 0.05 |
| TinyCLIP 40M/32 | 상자 | 0.24 / 0.24 / 0.12 | 0.54 / 0.29 | 0.55 / 0.68 | 0.05 |
| XLM-R B/32 | 상자 | 0.29 / 0.31 / 0.36 | 0.39 / 0.23 | 0.55 / 0.69 | 0.45 / 0.62 |
| RN50 | 상자 | 0.12 / 0.17 / 0.24 | – | 0.50 / 0.64 | – |
| TinyCLIP 8M/16 | 늘림 | 0.11 / 0.14 / 0.16 | – | 0.45 / 0.56 | – |

**자르기 방법 비교 (이름: 시연 / 깨끗)**

| 모델 | 상자+둘레 | 늘림(meridian) | 마스크 밖 회색 | 마스크 넓혀 흐리게 | 배경 반만 흐림 | 패치·마스크 풀링 |
|---|---|---|---|---|---|---|
| OpenAI B/32 | **0.28 / 0.30** | 0.21 / 0.23 | 0.05 / 0.09 | 0.19 / 0.22 | 0.17 / 0.23 | 0.02 / 0.01 |
| DataComp B/32 | **0.31 / 0.34** | 0.25 / 0.27 | 0.11 / 0.13 | 0.21 / 0.25 | 0.21 / 0.27 | 0.03 / 0.01 |
| SigLIP 2 B/32 | 0.30 / 0.33 | 0.24 / 0.27 | 0.11 / 0.14 | 0.19 / 0.27 | 0.22 / 0.28 | **0.34 / 0.40** (MAP 주의집중에 log 마스크) |
| MobileCLIP 2-S0 | 0.29 / 0.30 | 0.25 / 0.27 | 0.11 / 0.14 | 0.26 / 0.30 | 0.23 / 0.26 | **0.31 / 0.35** (마지막 특징 지도 가중 평균) |
| MobileCLIP-S1 | 0.35 / 0.39 | 0.29 / 0.34 | 0.14 / 0.17 | 0.28 / 0.32 | 0.28 / 0.36 | **0.35 / 0.41** |

**읽는 법**
- **마스크 밖을 지우면 나빠진다.** 문맥이 사라지고 FastSAM 마스크는 proto 격자(4 px)라 가장자리가 계단 모양이다. 계단 모양 때문에 'staircase' 가 자주 나왔다. 넓혀 흐리게 해도 상자보다 못하다.
- **ViT CLIP 의 패치 토큰은 글과 정렬되어 있지 않다.** 마지막 층 패치 토큰을 마스크로 평균해 투영하면 'dirt'·'tie'·'plywood' 가 나온다(0.02). meridian 의 `id_emb` 는 **같은 물체 판별(re-ID)에만** 쓰고, 이름·글 찾기에는 `query_emb`(CLS)를 써야 한다.
- **SigLIP 2 의 MAP 풀링에 마스크를 넣으면 공짜로 좋아진다.** 주의집중 logit 에 `log(max(w, 0.01))` 을 더한다(w = 8 × 8 격자 마스크 비율). w = 1 이면 원래 출력과 같다(확인). 상자 문맥은 패치에 남고, 풀링만 물체 쪽으로 기운다.
- FastViT(MobileCLIP)은 마지막이 전역 평균이라 마스크 가중 평균이 자연스럽게 된다.
- **늘림(meridian 방식)** 은 시연 crop 에서는 나쁘고, 길쭉한 기억 crop 에서는 좋을 때가 있다. 기본은 정사각 + 둘레로 한다.
- 범주별(SigLIP 2 마스크 MAP): sofa 0.93, room light 0.48, coffee table 0.33, TV 0.57, **radio 0.00, cabinet 0.00, fireplace 0.08**.
  - 라디오(빨간·흰 장난감 같은 모양, 로봇 손에 들림)는 모든 모델이 'stapler'·'candy cane'·'defibrillator' 로 부른다.
  - 서랍장은 'baseboard'(구조물 단어)로 끌려간다. 드문 구조물 단어 9개를 어휘에서 빼도 시연 정확도는 그대로, 기억 crop 만 +0.04–0.08.
- **흰 의자(YOLO26 이 'toilet' 으로 부른 O23·O25·O30)**: CLIP 은 'toilet' 이라 하지 않는다.
  - MobileCLIP-S2·S1(마스크 평균)은 'chair', SigLIP 2·XLM-R 은 'stool'·'footstool', OpenAI B/32 는 'coffee table'·'baseboard'.
  - 다만 좌판 확대 crop(O4·O12)은 대부분 'toilet tissue'(흰색 = 욕실 물건 쏠림)다 → 3절 (c) 도메인 보정의 동기.
- **찾기는 이름보다 잘 된다.** SigLIP 2 마스크 MAP 에서 "red radio" 1위, "radio" 2위, 한국어 "빨간 라디오" 1위. 이름이 틀려도 임베딩이 있으면 찾는다.

### 2.3 한국어

글 인코더만 한국어가 필요하다. 영상 공간은 그대로 둔다.

| 글 인코더 | 짝 영상 공간 | 글 크기 | 한국어 찾기 R@1 / R@5 / MRR (상자) | 한국어 라벨로 이름(HOME 87개) | 영어 라벨로 이름 | 글 1개 CPU(PC, 1 / 16 스레드) | 라이선스 |
|---|---|---|---|---|---|---|---|
| `Bingsu/clip-vit-base-patch32-ko` | OpenAI B/32 (영상 탑 동일 확인, 코사인 1.0000) | 63M | **0.75 / 0.85 / 0.79** | 0.24 | 0.28 | 11 / 5 ms | MIT |
| `Bingsu/vitB32_bert_ko_small_clip` (학생 = `lassl/bert-ko-small`) | OpenAI B/32 (동일) | **23M** | 0.45 / 0.75 / 0.57 | **0.29** | 0.17 | **5 / 4.5 ms** | MIT |
| sentence-transformers `clip-ViT-B-32-multilingual-v1` | OpenAI B/32 | 135M | 0.40 / 0.60 / 0.51 | 0.06 | 0.25 | 25 / 11 ms | Apache-2.0 |
| SigLIP 2 자체 글(다국어) | SigLIP 2 B/32 | 282M | 0.60 / 0.90 / 0.73 (마스크 MAP 0.65 / – / 0.75) | 0.32 | 0.31 | 130 / 32 ms | Apache-2.0 |
| `hyunlord/siglip2-base-patch16-224-ko` | SigLIP 2 **B/16** | 282M | 0.55 / 0.90 / 0.69 (늘림 0.70 / 0.90 / 0.78) | 0.23 | 0.31 | 132 / 35 ms | MIT |
| SigLIP 2 B/16 자체 글 | SigLIP 2 B/16 | 282M | 0.55 / 0.85 / 0.68 | 0.31 | 0.29 | 135 / 34 ms | Apache-2.0 |
| XLM-R B/32 자체 글 | XLM-R B/32 | 278M | 0.45 / 0.85 / 0.62 | 0.28 | 0.24 | 154 / 40 ms | MIT |

- CPU 시간은 다른 작업이 함께 돌던 때 잰 값이라 대략이다. HOME 어휘 = COCO + 집 물건 + 구조물 87개, 한국어 이름은 직접 붙였다(`~/clip_bench/home_vocab.py`).
- **이름은 영어 라벨로 붙이고, 한국어 이름은 짝 열로 보여 준다.** 한국어 라벨로 직접 비교해도 나아지지 않는다(0.24–0.32 대 0.28–0.31).
- 한국어 질의는 두 길이 있다.
  - 사전에 있는 말("머그잔"→mug): agent 사전으로 영어로 바꿔 미리 계산한 표에서 찾는다. 글 인코더가 필요 없다.
  - 자유 문장: 한국어 글 인코더가 필요하다. SigLIP 2 글(282M)은 Nano CPU 에서 질의당 1–2 s(추정)라 **로봇 밖(agent 서버)** 에서 돌리거나, 작은 한국어 학생(3절 b)을 만든다.
- Bingsu 두 모델은 SBERT 다국어 증류(arXiv 2004.09813)로 만들었고, 학습 코드는 github.com/Bing-su/KoCLIP_training_code 에 있다. 작은 쪽(23M)이 우리가 만들 학생과 거의 같은 크기다 → **같은 방법으로 SigLIP 2 B/32 글 공간에 다시 증류**하면 된다.

**다시 증류할 때 쓸 HF 데이터 (라이선스 확인, HF API)**

| 데이터 | 내용 | 라이선스 | 쓰임 |
|---|---|---|---|
| `lemon-mint/korean_parallel_sentences_v1.1` | 한–영 문장 10만–100만 | MIT | 주 학습 데이터 |
| `Moo/korean-parallel-corpora` | 한–영 문장 1만–10만 | CC BY-SA 3.0 | 추가 |
| `lemon-mint/korean_english_parallel_wiki_augmented_v1` | 위키 한–영 10만–100만 | CC BY-SA 3.0 | 추가 |
| `Bingsu/laion2b_multi_korean_subset` | 한국어 캡션 + 영상 URL | 표기 없음(LAION 원본 조건 따름, 확인 필요) | 영상에 붙은 학습(선택) |
| `floschne/xm3600` | Crossmodal-3600, 한국어 캡션 포함 | CC BY 4.0 | **평가** |
| `kms7530/ko-coco-bal` | 한–영 COCO 캡션 | "cc"(종류 불명) | 확인 전 보류 |
| `KORMo-VL/coco_captioning` | COCO 캡션 | 표기 없음 | 확인 전 보류 |

- 라이선스 없는 AI Hub 재업로드(예: `jeina/korean_image_caption`)는 쓰지 않는다.
- HF 에 **물체 이름 한–영 단어표는 없다** → 직접 만든다(3절 1).

### 2.4 FP16 정확도 (PC, TRT 10.16)

| 모델 | TRT FP16 대 torch FP32 코사인 (실제 crop 8장) |
|---|---|
| OpenAI B/32, DataComp B/32-256, TinyCLIP 61M/32 | 1.0000 / 0.9999 / 1.0000 |
| MobileCLIP-S1, MobileCLIP 2-S0(두 출력 판) | 0.9999 |
| **SigLIP 2 B/32, 전부 FP16** | **0.64–0.74 (망가짐)** |
| SigLIP 2 B/32, 이름에 'norm' 이 든 층(LayerNorm 분해 243개)만 FP32 고정 | **0.9998–1.0000**, b8 지연 1.16 ms(전부 FP16 1.17 ms) |

- opset 13 에서는 LayerNorm 이 ReduceMean·Pow·Sqrt 로 쪼개진다. SigLIP 은 활성값이 커서 FP16 에서 넘친다.
- 빌드 스크립트: `~/clip_bench/build_mixed.py`(`OBEY_PRECISION_CONSTRAINTS` + 층별 FP32). TRT 8.2 에도 같은 API(층 precision + `kOBEY_PRECISION_CONSTRAINTS`)가 있다(추정, Nano 에서 확인).
- Maxwell 에서 FP32 층은 FP16 의 절반 속도다. LayerNorm 은 연산량의 1 % 미만이라 Nano 에서도 영향이 작을 것이다(추정).

## 3. 우리 작은 모델 + 큰 어휘 설계 (SigLIP 2 B/32, 768-d 기준)

### 3.1 학습 없는 큰 라벨 표

- **표 두 개를 둔다.**
  - **주 표(이름 붙이기)**: 집 물건 1–3k 개. BEHAVIOR-1K synset 중 집 안 물건 + COCO + LVIS + Objects365 이름 + 구조물. 
  - **긴 꼬리 표**: 3–5만 개. WordNet 명사(artifact·food·plant·animal·natural object·structure 아래) + ImageNet-21k + Open Images.
- 이번에 만든 긴 꼬리 표: WordNet 3.0 해당 가지의 모든 lemma + 우리 어휘 = **28,838 줄**(synset 22,374개). SigLIP 2 글 인코딩 PC CPU 8분, 한 번만.
- **어휘가 크면 1위 이름이 나빠진다**(측정, SigLIP 2): 360개 표 0.30–0.34 → 28.8k 표 0.16–0.19.
  - 비슷한 말이 서로 1위를 다툰다('studio couch'·'covered couch').
  - WordNet 상위어로 올리면 회복한다: 1단계 0.19–0.21, 2단계 0.26–0.29.
  - → **이름 수준을 확신도에 맞춘다**: 1위와 2위 차가 작으면 상위어(머그잔 → 컵 → 그릇)로 올린다. 주 표 점수가 낮을 때만 긴 꼬리 표를 본다.
- 한국어 열: Wikidata 한국어 라벨(CC0, WordNet·ImageNet synset 연결이 있다) → 없으면 우리말샘 동의어(CC BY-SA) → 그래도 없으면 LLM 한 번 번역 후 사람이 주 표만 검수. KorLex 는 사용자가 접근권을 얻으면 쓴다.
- 속성 질의("빨간 컵", "흰 의자")는 라벨 표에 넣지 않는다. 물체 임베딩과 질의 글 임베딩을 바로 비교한다(3.4).

### 3.2 우리 작은 모델 4개

| # | 모델 | 크기 | 학습 데이터·비용 | 효과 | 위험 |
|---|---|---|---|---|---|
| a | **투영 768 → 128** (Matryoshka 꼴, 128·64·32 앞자리만 잘라도 되게) | 선형 1개, 약 0.1M | (영상, 라벨 글) 쌍: BEHAVIOR 자산 렌더 + 우리 crop + 라벨 표. 대조 손실. 4090 수십 분 | 라벨 찾기 1단계를 작고 정확하게. 지금의 PCA(학습 없음)는 128-d 만으로는 1위 일치 0.24 에 그친다(3.3) | 영상 쪽 분포가 바뀌면 다시 학습 |
| b | **작은 한국어 글 인코더** (`lassl/bert-ko-small` 학생 → SigLIP 2 영어 글 임베딩을 따라 하게, a 와 함께 학습) | 23M (FP16 46 MB) | 한–영 문장 수십만–100만 쌍(2.3 표) + 물체 이름 쌍. 교사 임베딩 미리 계산. 4090 2–6 시간(추정) | 로봇 안 한국어 자유 질의. Nano GPU 질의당 10–30 ms, CPU 50–100 ms(추정). 같은 방법의 Bingsu 23M 모델이 B/32 공간에서 이미 동작함(한국어 R@5 0.75) | 짧은 물체 이름 질의는 문장 데이터와 분포가 달라 이름 쌍을 꼭 섞는다 |
| c | **도메인 보정기** (얼린 영상 임베딩 위 잔차 MLP 768 → 256 → 768) | 약 0.4M | BEHAVIOR-1K 자산 렌더(범주 1–2k, 범주당 50–200 시점, 10–30만 crop) + 라벨 표를 얼린 분류 머리로. 원 임베딩에서 멀어지지 않게 정규화. 4090 1 시간 이내 | '흰 의자 → toilet tissue', '서랍장 → baseboard' 같은 쏠림 고치기 | 시뮬에 과적합하면 실물에서 나빠진다 → 실물 bag 으로 꼭 검증. 보정 전 임베딩도 같이 저장 |
| d | 영상 인코더 증류 (예: ViT-S/32 → SigLIP 2 공간) | 20–40M | 영상 1–5천만 장, GPU 여러 날 | Nano 에서 2–3배 빠름 | **열린 어휘 일반성을 잃는다.** Nano 속도가 정말 모자랄 때만 |

### 3.3 라벨 찾기를 마이크로초로 (Nano CPU 에서)

- **메모리 대역폭이 한계다.** Nano LPDDR4 25.6 GB/s. 28.8k × 768 FP16 = 44 MB 를 읽는 것만으로 약 2 ms(이론), 실제 4–7 ms(추정).
- 방법: 128-d INT8 + IVF(256 묶음, 묶음은 WordNet 상위어에 맞출 수도 있음) + 128-bit 2진 부호, 마지막에 상위 32–64개만 768-d 로 다시 매긴다.
- **CPU NEON** 에서 돈다. GPU 는 FastSAM·CLIP 에 남기고, 커널 실행·동기화 비용도 없다. keyframe 마다 새·바뀐 물체만 묶어서.

**측정 (PC Ryzen 9 9950X, 단일 스레드 `taskset`, 28,838 라벨, SigLIP 2 B/32 마스크 MAP 임베딩 567개 질의)**
- `~/clip_bench/lookup/`(`prep.py` 가 색인·정확도, `lookup.cpp` 가 시간). PCA 는 학습된 투영 대신 쓴 것이다(라벨 글 + 우리 crop 으로 맞춤).
- SSE 판(`-march=x86-64 -mno-avx -mpopcnt`)은 NEON(128-bit) 폭에 맞춘 것이다.

| 방법 | 후보 수 | µs/물체 AVX-512 / SSE | 정확 1위와 일치 | 정확 1위가 상위 5 안 | 이름 정답률 1위 / 상위 5 |
|---|---|---|---|---|---|
| 768-d FP32 전부 (기준) | 28,838 | 1,712 / 2,085 (88 MB 읽기, 대역폭 한계) | 1 | 1 | 0.19 / 0.33 |
| 128-d INT8 전부 | 28,838 | 108 / 270 | 0.24 | 0.58 | 0.13 / 0.31 |
| IVF 4 묶음 + INT8 | 518 | 9.1 / 11.1 | 0.21 | 0.49 | 0.14 / 0.33 |
| IVF 4 묶음 + 2진 → INT8 상위 32 | 518 | 6.7 / 7.5 | 0.19 | 0.43 | 0.16 / 0.35 |
| **IVF 4 + INT8 → 768-d 상위 32 다시 매김** | 518 | 7.8 / 12.4 | 0.69 | 0.69 | **0.20 / 0.35** |
| **IVF 4 + 2진 → 768-d 상위 32 다시 매김** | 518 | **6.1 / 8.0** | 0.54 | 0.54 | **0.19 / 0.37** |
| IVF 8 + INT8 → 768-d 상위 64 | 1,199 | 16 / 26 | 0.82 | 0.82 | 0.19 / 0.37 |

- **768-d 로 다시 매기면 이름 정답률은 그대로다**(0.19–0.20). 정확 1위 일치가 0.54–0.82 인 것은 비슷한 말(동의어 줄)끼리 바뀌는 경우가 대부분이다.
- PCA 를 라벨 글만으로 맞추면 128-d 는 거의 쓸모없다(1위 일치 0.13). 영상·글 공간 차이(modality gap) 때문이다 → 3.2 (a) 학습 투영이 필요한 이유. 256-d PCA 는 0.56.
- **Nano A57 추정 (1.43 GHz, NEON 128-bit, SDOT 없음, `vcnt` 있음)**
  - 2진 거리 518개: 1 µs 안팎. INT8 518 × 128: `vmull_s8`+`vpadalq` 로 6.6만 곱셈 → 8–15 µs. 묶음 중심 256 × 128 FP32: 3–6 µs. 768-d FP16 상위 32: 3–6 µs. 정렬·기타 5–10 µs.
  - → **2진 + 다시 매김 20–40 µs, INT8 + 다시 매김 30–60 µs, 넉넉히 20–100 µs/물체.** 읽는 양은 물체당 약 0.1 MB.
  - 새 물체 20개 keyframe 이라도 2 ms 이하 → **keyframe 비용(FastSAM + CLIP 수백 ms)의 1 % 미만**이다.
- 메모리: 128-d INT8 3.7 MB + 2진 0.46 MB + 768-d FP16 다시 매김용 44 MB(또는 256-d 로 줄여 15 MB).

### 3.4 물체 찾기 쪽 (질의 → 물체)

- 물체는 수백 개다. 768-d FP16 × 500 = 0.77 MB 를 다 훑어도 Nano CPU 수십–수백 µs(추정). 색인이 필요 없다.
- 비용은 **글 인코더**다.
  - SigLIP 2 글 282M: Nano CPU 1–2 s/질의(추정), GPU 에 올리면 0.56 GB 를 더 써야 한다 → 로봇 밖에서.
  - 작은 한국어 학생 23M(3.2 b): Nano 에서 10–100 ms(추정) → 로봇 안에서 된다. 그래서 (b) 가 중요하다.
- agent 는 질의마다 "물체 id, 점수, 이름, 위치" 상위 5개를 받는다. 이름이 틀려도(라디오) 임베딩으로 찾는다.

### 3.5 저장: 벡터가 원본, 이름은 캐시 (2026-10-03 결정)

- CLIP 입력은 **원본 RGB** 에서 물체 상자를 10 % 여유로 정사각형으로 잘라 그대로 넣는다(픽셀은 바꾸지 않는다). 마스크는 마지막 풀링의 가중치로만 쓴다.
- 물체 하나에 저장하는 원본은 **임베딩 벡터 1개**(768-d FP16)뿐이다.
- 이름(의미 단어)은 그 벡터와 라벨 표의 코사인으로 언제든 다시 뽑을 수 있다. 그래서 이름은 **캐시**로 둔다. 라벨 찾기가 µs 라 다시 뽑는 비용은 작지만, 캐시해 두면 LLM 이 scene graph 를 읽을 때 도구 호출 없이 이름이 보이고, 뷰어·`scene.json` 을 사람이 바로 읽는다.
- 두 가지 찾기 모두 코사인이다.
  - 질의 → 물체: 질의 글 벡터 ↔ 물체 벡터.
  - 물체 → 이름: 물체 벡터 ↔ 라벨 표 → 캐시.

기억 폴더 안 배치:

```
memory/
├── scene.json            # 노드 metadata.names = 캐시의 1위 이름(영·한) + 라벨 표 sha
├── objects/              # 원본: 물체마다 관측에서 나온 것
│   ├── O<id>_rgb.png · O<id>_depth.png · O<id>_mask.png · O<id>_points.ply
│   └── O<id>_emb.f16     # 임베딩 벡터 768 × FP16 (원본)
└── cache/                # 파생: 지워도 objects/ + 라벨 표로 다시 만든다
    ├── names.json        # {"table": {name, version, sha}, "objects": {"O12": {"emb_sha": …, "en": [["chair", 0.31], …], "ko": [["의자", 0.31], …], "level": "chair", "structural": false}}}
    └── index/            # 찾기 색인(2진 부호·IVF 등), 다시 만들 수 있음
```

- 다시 만드는 조건: 라벨 표 `sha` 가 바뀌었거나, 물체의 `emb_sha` 가 바뀌었을 때(best view 갱신)만 그 물체를 다시 뽑는다.
- 쓰기는 다른 파일처럼 임시 이름 → rename(원자적), 저장 주기에 맞춰 비동기로.

## 4. Nano 속도 추정

**가정 (모두 추정)**
- Nano FP16 이론 472 GFLOPS, 텐서코어 없음, INT8 가속 없음.
- ViT 처럼 큰 행렬곱 위주: 배치 8 에서 이론의 35–50 %(165–235 GFLOPS), 배치 1 에서 20–30 %.
- conv·depthwise 위주(FastViT, YOLO): 80–150 GFLOPS. 근거: Nano TRT FP16 ResNet18 18 ms @256(약 265 GFLOPS 꼴) [D10], MobileViT-S 53 ms(약 75 GFLOPS 꼴) [EN], YOLOv8n-seg 640 4.2 FPS(전후처리 포함) [12].
- 검산: PC b32 B/32 0.096 ms/장 × (PC FP16 텐서 연산력 / Nano ≈ 370배) ≈ 36 ms. 아래 범위와 맞는다.

| 모델 | GFLOPs | Nano 물체당 ms 배치 8 / 배치 1 (추정) | 엔진 메모리(추정) |
|---|---|---|---|
| SigLIP 2 B/32-256 (LayerNorm FP32) | 11.4 | **50–70 / 80–120** | 190 MB 가중치 + 30–50 MB |
| OpenAI B/32-224 | 8.8 | 37–53 / 63–93 | 176 MB + 30 MB |
| MobileCLIP 2-S0 | 4.8 | 32–60 / 40–70 | 23 MB + 30 MB |
| TinyCLIP 61M/32 | 6.2 | 26–38 / 45–65 | 123 MB + 20 MB |
| FastSAM-s 416 (YOLOv8s-seg 42.6 GF @640 → 18 GF) | 18 | 80–150 (한 장) + 후처리 5–10 | 0.3–0.4 GB(CUDA 문맥 포함) |

- GPU 에서 FastViT 의 FLOP 이점이 사라지는 것은 PC 에서 이미 보인다(S0 4.8 GF 가 B/32 8.8 GF 와 같은 속도). Nano 에서도 S0 ≈ B/32 일 가능성이 크다(추정).

**keyframe 하나 (FastSAM-s 416 + SigLIP 2 B/32, 같은 프로세스·CUDA 문맥)**

| 단계 | 새·바뀐 물체 5개 | 20개 |
|---|---|---|
| FastSAM-s 416 + NMS·마스크 | 85–160 ms | 85–160 ms |
| ROI 자르기·정규화 커널 | < 2 ms | < 2 ms |
| SigLIP 2 B/32 | 250–350 ms | 1.0–1.4 s |
| 라벨 찾기 (CPU) | < 0.5 ms | < 2 ms |
| **합** | **0.35–0.5 s** | **1.1–1.6 s** |

- CLIP 이 대부분이다. 그래서
  - keyframe 당 CLIP 은 **최대 8개**(새 물체 먼저, 그다음 best view 가 많이 좋아진 것). 나머지는 다음 keyframe 으로 미룬다.
  - 탐색 중 keyframe 은 1 Hz 이하. 이미 아는 물체만 보이면 CLIP 은 0개다.
  - OpenAI B/32 로 바꾸면 CLIP 부분이 약 25 % 준다. TinyCLIP 61M/32 는 약 45 % 준다(품질은 2.2 표).

## 5. 추천 스택 (구체)

| 부분 | 선택 | 도는 곳 | 비고 |
|---|---|---|---|
| 분할 | **FastSAM-s 416**, FP16, opset 13 정적 ONNX → Nano 에서 엔진 빌드 | Nano GPU, keyframe 만(≤ 1 Hz) | ovdet 이 이미 돌린다(클래스 무관, CUDA NMS) |
| 영상 인코더 | **SigLIP 2 B/32-256**, 두 출력(`emb`, `emb_mask`, 입력 `images` N×3×256×256 + `wpatch` N×64), LayerNorm FP32 고정 | Nano GPU, 같은 CUDA 문맥, 별도 스트림, 배치 ≤ 8 | 저장은 벡터 1개(FP16 768-d, `objects/O<id>_emb.f16`), 이름은 `cache/` (3.5절) |
| 영상 대안 | OpenAI B/32 + `Bingsu/clip-vit-base-patch32-ko` | 같은 자리 | meridian 과 같은 공간. 한국어 찾기 측정 1위(0.75). 이름은 0.28 로 낮다 |
| 한국어 글 | **처음**: SigLIP 2 자체 글(282M)을 agent 서버에서. **다음**: `lassl/bert-ko-small` 학생을 SigLIP 2 B/32 글 공간에 다시 증류(3.2 b) | 서버 → 나중에 Nano | 사전에 있는 말은 미리 계산한 표에서 바로 |
| 라벨 표 | 주 표 1–3k(BEHAVIOR 집 물건 + COCO/LVIS/O365 + 구조물) + 긴 꼬리 28.8k(WordNet, 3–5만으로 늘림), 영어 글 임베딩은 PC 에서 한 번, 한국어 이름 열 | 파일로 Nano 에 | 확신도 낮으면 WordNet 상위어로 |
| 투영 차원 | 1단계 128-d INT8 + 128-bit 부호(처음엔 PCA, 다음 학습 투영 a), 2단계 768-d FP16 | Nano CPU | |
| 찾기 방법 | IVF 256 묶음 × 4 → 2진 거리 → 상위 32 를 768-d 로 | Nano CPU NEON, 물체마다 | PC 6–8 µs, Nano 20–100 µs(추정) |
| 구조물 거르기 | 주 표의 구조물 행(벽·바닥·천장·문·창·걸레받이 …)이 1위면 "구조물" 표시, 기억에는 남기되 agent 목록에서 뺀다 | Nano CPU | 측정 재현 0.55 / 오탐 0.17 → 깊이·평면 단서(큰 수평·수직 면)와 함께 쓴다 |
| 이름 안정 | 노드 안 점수 가중 다수결(best view 바뀔 때마다 갱신), 갈리면 상위어 | scenemap | |
| 물체 찾기 | 질의 글 임베딩 ↔ 모든 물체 `emb_mask` 코사인 | Nano CPU(수백 개) | agent 도구 `find_object(text)` |

- **Nano 예산(추정)**: 영상 인코더 엔진 0.22–0.24 GB + 라벨 표 0.05 GB. A′ 예산(1.6–2.5 GB)에 더해도 4 GB 안이다. CUDA 문맥은 ovdet 과 나눠 써야 한다(문맥 하나에 0.15–0.25 GB).
- **keyframe 지연(추정)**: 새 물체 5개 0.35–0.5 s, 8개 상한이면 최악 0.5–0.75 s. 비동기라 SLAM·주행은 막지 않는다.

## 6. 네이티브 포팅 계획 (10-03 구현 — 결과는 8절)

**ovdet / sgrt 에 넣는 모양**
1. `clipenc` 모듈(ovdet 안, 같은 CUDA 문맥·핸들). 입력: ovdet 이 이미 GPU 에 든 원본 RGB, 물체 상자, proto 격자 마스크 비트.
2. **CUDA 커널 하나**: 물체마다 정사각 상자 + 10 % 둘레를 양선형으로 S × S 로 뽑고, 정규화(평균·표준편차)해 FP16 NCHW 로 쓴다. 같은 커널이 마스크 비트를 8 × 8 격자 비율(`wpatch`)로 줄인다. 이미지 밖은 평균 회색.
3. 묶음: 엔진 배치 1–8 동적(최적화 프로필 하나, opt = 8) 또는 1·8 정적 엔진 둘. 8 넘으면 다음 keyframe 으로.
4. 비동기: 전용 스트림 + 이벤트. scenemap 의 `png_dirty`(또는 새 노드)가 큐에 넣고, 결과가 오면 노드 메타데이터에 768-d FP16 두 개와 이름 상위 3개를 쓴다.
5. 라벨 찾기: C++ 작은 라이브러리(IVF·INT8·2진·다시 매김, NEON 판과 일반 판). 표 파일은 PC 에서 Python 으로 만든다.
6. 출력: Spark-DSG 노드에 `name`, `name_ko`, `name_score`, `semantic_feature`(768-d) 를 넣는다(뷰어 메타데이터와 같은 자리).

**TensorRT 8.2 (JetPack 4.6) 에서 달라지는 것 (meridian 은 TRT 10)**
- 실행: `enqueueV3` + `setTensorAddress` 대신 `enqueueV2(bindings, stream)` + 바인딩 배열. 동적 크기는 `setBindingDimensions`. ovdet 에 `#if NV_TENSORRT_MAJOR < 10` 분기를 둔다.
- 빌드: `setMemoryPoolLimit` 대신 `setMaxWorkspaceSize`. 층별 정밀도는 `ILayer::setPrecision` + `kOBEY_PRECISION_CONSTRAINTS`(8.2 에 있다고 봄, 없으면 `kSTRICT_TYPES`).
- ONNX: opset 13, 정적 배치, `onnxsim` 으로 Shape·Gather 정리. 엔진은 **Nano 에서** 만든다(엔진은 기기·버전에 묶인다).
- CUDA 10.2: `cudaMallocAsync` 없음 → 시작 때 버퍼를 한 번 잡는다. Nano 는 CPU·GPU 메모리가 같으므로 카메라 프레임은 `cudaHostAllocMapped`(0 복사).
- INT8 은 쓰지 않는다(sm_53 미지원).

**DX-M1**: A′ 와 같이 기본에서 뺀다(Ubuntu 18.04 미지원). 팀이 RPi5 + DX-M1 로 가면 DEEPX 동물원의 **DataComp ViT-B/32-256**(MIT)을 그대로 쓰고(NPU INT8, 글은 CPU), SigLIP 2 는 DX-COM 컴파일을 먼저 시험한다.

**손 (사람-일, 추정)**

| 일 | 일수 |
|---|---|
| 두 출력 ONNX(이번에 PC 에서 됨) 정리 + Nano 엔진 빌드 + LayerNorm FP32 + 정확도 확인 | 1–2 |
| ovdet `clipenc`: TRT 8.2/10 분기, ROI 커널, 배치·스트림 | 3–4 |
| 라벨 표 도구(WordNet·BEHAVIOR·COCO/LVIS + Wikidata 한국어) | 2–3 |
| 라벨 찾기 C++(NEON) + 시험 | 1–2 |
| scenemap 연결(`png_dirty` 큐, 노드 메타데이터), 뷰어 표시 | 1–2 |
| agent `find_object(text)`(글은 서버 SigLIP 2) | 1 |
| (b) 한국어 학생 증류(Bingsu 코드 바탕) + XM3600·우리 질의 평가 | 2–3 + GPU 반나절 |
| (a) 학습 투영, (c) 도메인 보정기(렌더 데이터 만들기 포함) | 3–5 |
| **합** | 핵심 9–14일, 작은 모델 포함 14–22일 |

## 7. 열린 질문

1. **Nano 실측**: SigLIP 2 B/32 · OpenAI B/32 · TinyCLIP 61M/32 의 TRT 8.2 엔진 빌드 가능 여부와 배치 1/8 지연, FastSAM-s 416 과 동시 실행 때 메모리.
2. SigLIP 2 B/32 를 224 입력(49 토큰)으로 돌리면 몇 % 빨라지고 얼마나 나빠지나(위치 임베딩 보간, 안 잼).
3. 라디오처럼 **이름이 전혀 안 맞는 과제 물체**: 동의어 줄("portable radio", "toy radio") 추가, 도메인 보정기(c), 또는 agent 가 찾기 결과로 확인하는 흐름 중 무엇을 기본으로 할지.
4. 정답 잡음: 시연 정답은 3D 상자 + 오도메트리라 순위만 믿을 수 있다. 원본 HDF5(정답 자세)가 있으면 다시 잰다. 실물 bag 평가는 A′.6 3번과 함께.
5. 한국어 질의 20개는 너무 적다. XM3600 한국어 + 우리 물체 질의 100개 이상으로 다시 잰다.
6. 라이선스: MobileCLIP(apple-amlr)을 연구 시연에만 쓸지. 공개 제출물(AGPL)에는 넣지 않는다고 본다.
7. 글 인코딩을 로봇 밖에서 할 때 네트워크가 끊기면: 사전 표 + 작은 학생(b) 으로 물러난다.


## 8. 포팅 결과와 모델 효율 (2026-10-03)

### 8.1 만든 것

| 부분 | 어디 | 내용 |
|---|---|---|
| 내보내기 | `src/behavior-2026/src/scene_graph/clip/tools/export_siglip2.py` | 영상 탑 → opset 13 ONNX. 입력 `images` N×3×256×256 + `wpatch` N×64, 출력 `emb` N×768(L2). 마스크 = MAP 주의집중 logit 에 `log(max(w, 0.01))`. w = 1 이면 원래 임베딩과 코사인 1.000000 |
| 엔진 | `clip/tools/build_engine.py` | FP16 + 이름에 `norm`·`mlp/act` 든 층 FP32 고정, 배치 칸 1·2·4·8 프로필, INT8 PTQ 선택. 엔진·ONNX 는 `~/ovdet_models/x86_sm120/siglip2_b32/`(git 밖) |
| 실행 | `clip/src/{crop.cu,encoder.cpp}` | CUDA 커널 하나(정사각 상자 + 10 % 둘레, 원본 RGB 양선형 256², 정규화 FP16, 8 × 8 마스크 비율), TensorRT 자기 스트림, 칸 2개 비동기 고리, 배치 칸별 CUDA graph, 결과 자리 풀. TRT 8.2 / 10 분기 |
| 라벨 찾기 | `clip/src/labels.cpp` | 표 = training/embed `labels/objects-v1`(30,533 줄, 영·한, WordNet 상위어, 구조물, main/tail). IVF 256 + 128-d FP16 1단계(+ 선택 128-bit 해밍) + 768-d 다시 매김, AVX2 / NEON / 일반 |
| 기억 폴더 | `clip/src/memstore.*`, `runtime/src/sgrt_clip.*` | `objects/O<id>_emb.f16`, `cache/names.json`, `cache/index/`, scene.json 노드 `emb`·`names`(sm_set_object_meta). 다시 만드는 조건 = 3.5 |
| 글 쪽 | `clip/tools/text_query.py` | SigLIP 2 글 탑(로봇 밖, CPU): 글 → 768-d, 기억 폴더 찾기, HTTP `/encode`·`/search` |
| 시험 | `ctest`(clip 4 개) | 커널 = CPU 기준, 엔진 = PyTorch FP32, 찾기 = 전부 훑기, 캐시 규칙 |

- 자세한 배치·ABI 는 서브모듈 [docs/scenemap_설계.md](../src/behavior-2026/docs/scenemap_설계.md) 3.7, [clip/README.md](../src/behavior-2026/src/scene_graph/clip/README.md).

### 8.2 정확도 맞춤 (PC, TRT 10.16)

| 엔진 | 코사인 대 PyTorch FP32 (평가 crop 567) | 이름 시연 / 깨끗 / 기억 | 찾기 R@1 영 / 한 |
|---|---|---|---|
| PyTorch FP32(같은 자르기 식, 양선형) | 1 | 0.351 / 0.416 / 0.40 | 0.65 / 0.65 |
| **채택 엔진**(패치 GEMM, LN·GELU FP32, 칸 1·2·4·8) | **평균 0.99987, 하위 1 % 0.99876, 최저 0.992, 98.8 % ≥ 0.999** | 0.351 / 0.416 / 0.40 | 0.65 / 0.65 |
| C++ 끝까지(원본 RGB → 커널 → 엔진, crop 64, ctest) | 평균 0.99990, 최저 0.99884 | | |

- 자르기 식: 2.2 의 연구 때는 bicubic 이었다. 런타임 커널은 양선형이고, 같은 모델에서 양선형 쪽이 조금 낫다(시연 0.351 대 0.337).
- 최저 0.992 인 crop 은 하나다(e57_f2340, 마스크가 작은 물체). 1–2 % crop 이 0.999 아래인 것은 FP16 행렬곱 자체 오차다 — FP32 엔진은 최저 0.9996.

### 8.3 모델 효율 변형 (요청: SigLIP 2 B/32 자체를 빠르게)

품질 = 평가셋 567 crop(이름 시연 / 깨끗 / 기억, 찾기 R@1 영 / 한, 기준 FP32 대 코사인 평균 / 최저). 시간 = **nsys 커널 합 ÷ 추론 수**(ms, PC).
이번 측정 동안 GPU 는 다른 일(학습·시뮬)이 100 % 가까이 쓰고 있었다. CUDA event 시간은 시간 조각 기다림까지 들어가 쓸 수 없어서
커널 실행 시간만 더했다. 그래도 ±15 % 흔들린다(3 번 중 최소). 빈 GPU 의 trtexec 값(2.1 표)은 b1 0.48 / b8 1.12 ms 였다.
Nano 는 연산량 비례 추정(4절 가정, FP16, 텐서 코어 없음).

| 변형 | 이름 시연 / 깨끗 / 기억 | R@1 영 / 한 | 코사인 평균 / 최저 | PC b1 / b8 ms | Nano 물체당 b8 (추정) | 판정 |
|---|---|---|---|---|---|---|
| FP16 전부 | – | – | 0.64–0.74 (2.4) | – | – | 망가짐 |
| FP16 + LayerNorm FP32 | 0.354 / 0.422 / 0.40 | 0.65 / 0.65 | 0.99987 / 0.983 | 1.28 / 1.94 | 50–70 ms | 기준 |
| + GELU(`mlp/act`) FP32 | 0.354 / 0.422 / 0.40 | 0.65 / 0.65 | 0.99989 / **0.996** | 같음(±잡음) | 같음 | **채택**(최저 코사인 좋아짐, 비용 없음) |
| + softmax FP32 | 0.351 / 0.416 / 0.40 | 0.65 / 0.65 | 0.99984 / 0.975 | 같음 | 같음 | 이득 없음 |
| 전부 FP32 | 0.351 / 0.416 / 0.40 | 0.65 / 0.65 | 0.999997 / 0.9996 | 2.73 / 6.01 | 2배(FP32 반 속도) | 느림 |
| INT8 PTQ(엔트로피, LVIS crop 512 보정) | 0.097 / 0.036 / 0.00 | 0.35 / 0.35 | 0.68 / 0.50 | 1.34 / 2.21 | Nano INT8 없음 | 망가짐 |
| INT8 MLP 만(주의집중·패치·머리 FP16) | 0.266 / 0.337 / 0.32 | 0.70 / 0.60 | 0.875 / 0.76 | 1.13 / 2.08 | – | 손해 큼, 빠르지도 않음 |
| 입력 224(49 토큰, 위치 보간) | 0.209 / 0.223 / 0.16 | 0.70 / 0.55 | 0.866 / 0.62 | (−23 % 연산) | −23 % | 미세 조정 없이 안 됨 |
| 입력 192(36 토큰) | 0.157 / 0.139 / 0.08 | 0.45 / 0.35 | 0.737 / 0.47 | (−44 % 연산) | −44 % | 안 됨 |
| 마지막 1층 빼기(11층) | 0.317 / 0.373 / 0.44 | 0.80 / 0.70 | 0.869 / 0.80 | 1.37 / 1.79 | −8 % | 이름 −0.04. 안 씀(찾기는 오히려 좋아 보이나 질의 20개) |
| 마지막 2·3층 빼기 | 0.214 / 0.241 / 0.28 · 0.194 / 0.241 / 0.08 | 0.75 / 0.60 · 0.70 / 0.60 | 0.83 · 0.79 | | −17 · −25 % | 안 됨 |
| 마스크로 토큰 버림 48개 @ 9층 | 0.354 / 0.416 / 0.40 | 0.70 / 0.65 | 0.973 / 0.76 | 1.17 / 2.16 | −6 % | 이득 작음, 안 씀 |
| 마스크로 토큰 버림 48 @ 6 · 32 @ 6 · 32 @ 3 | 0.214 · 0.100 · 0.043 | | 0.89 · 0.72 · 0.67 | | | 안 됨(문맥 잃음 — 2.2 의 "마스크 밖 지우기"와 같은 이유) |
| ToMe 4 토큰/층 @ 6층부터 | 0.354 / 0.428 / 0.36 | 0.75 / 0.65 | 0.994 / 0.975 | 1.73 / 2.47 | −8 % 연산, 정렬·모으기 비용 | PC 에서 더 느림(층 1837 → 4957), 안 씀 |
| ToMe 8 @ 4 · 4 @ 0 | 0.294 · 0.331 | | 0.85 · 0.93 | | | 안 됨 |
| **패치 임베딩 Conv → reshape + GEMM** | 같음 | 같음 | 0.99987 / 0.992 | 1.10 / 2.08(단독 측정 1.04 / 1.71 대 1.24 / 2.00) | 0 ~ −5 % | **채택**: TRT 가 32×32 stride 32 conv 에 느린 implicit-GEMM(127 µs, 엔진의 13.8 %)을 골랐다 |
| **배치 칸 1·2·4·8 정적 프로필**(+ 패치 GEMM) | 같음 | 같음 | 같음 | **0.69 / 1.84** | b1 이득 | **채택**: 동적 1–8 하나는 b1 에도 b8 용 전술을 씀. 엔진 185 → 324 MB(Nano 는 1·8 둘만 권장) |
| CUDA graph(칸·배치별) | 같음 | 같음 | 같음 | event 기준 −5–10 % | 커널 실행 비용 큰 Nano 에서 더 큼(추정) | 채택(TRT 10). TRT 8.2 는 끔 |

- 머리 가지치기·자기 증류(distill-to-self)는 하지 않았다. 층 빼기·토큰 줄이기가 학습 없이는 모두 손해라서, 되살리려면 학습이 필요하다
  → 하게 되면 training/embed(임베딩 학습 담당)와 같이 한다(큰 학습은 시작하지 않음).
- 퓨전: 채택 엔진 층 정보(`build_engine.py --layer-info`)를 보면 TRT 10 이 주의집중을 `_gemm_mha_v2` 하나로, LayerNorm·GELU 를
  `__myl_*` 퓨전 커널로 묶었다. FP32 섬 경계의 Cast 는 그 퓨전 안에 들어가 따로 재포맷 층이 없다. 따로 플러그인을 쓸 이유가 없었다.
- 입력 FP16(`--half-input`): 커널이 FP16 NCHW 로 바로 써서 입력 버퍼가 반(8 × 3 × 256² × 2 = 3 MB).
- 고정 메모리: 출력·작업 목록·마스크 비트는 `cudaHostAlloc`. 프레임마다 메모리를 잡지 않는다(마스크 격자가 처음보다 커질 때만).

**채택 기본값**: `export_siglip2.py`(패치 GEMM 기본) → `build_engine.py --half-input --pin norm,mlp/act --profiles 1,2,4,8`.
Nano 는 같은 ONNX 로 `--profiles 1,8`(메모리) + FP16.

### 8.4 단계별 시간 (PC 측정, Nano 추정)

| 단계 | PC (RTX 5070 Ti, 9950X) | Nano (추정) | 비고 |
|---|---|---|---|
| 자르기 커널(물체 8, 720² 원본) | **3–12 µs** 실행(nsys). 기다림 포함 event 0.02 ms(빈 GPU) – 6 ms(나눠 쓸 때) | 0.3–1 ms | `sgc_submit` 은 이것만 기다림 |
| SigLIP 2 엔진 b1 / b8 | 0.69 / 1.84 ms(커널 합, 나눠 쓰는 GPU), 빈 GPU trtexec 0.48 / 1.12 ms(2.1) | 80–120 ms / 400–560 ms(물체당 50–70) | 비동기 |
| 라벨 이름(30.5k 줄, 1 물체, 단일 스레드) | 25 µs(IVF 8 → 128-d top 32 → 768-d + 상위어), 전부 훑기 1.0 ms | 100–250 µs(NEON) | 저장 스레드, 바뀐 물체만 |
| 라벨 색인 만들기(처음 한 번) | 0.8–0.9 s(영상 표본 PCA + k-means), 캐시 읽기 1 ms | 10–20 s(한 번, PC 에서 만들어 옮겨도 됨) | `cache/index/` |
| 질의 → 물체(255 물체 768-d) | < 50 µs | < 0.5 ms | 색인 없이 전부 |
| 글 인코딩(SigLIP 2 글 탑, 로봇 밖 CPU) | 32 ms/질의(16 스레드, 2.3), 1 스레드 130 ms | (로봇 밖) | text_query.py |
| keyframe 하나 합(FastSAM-s 416 + CLIP 8 개) | 검출 7.4 ms(시뮬과 GPU 나눔) + CLIP 비동기 2 ms 안팎 | 0.5–0.75 s(4절과 같음, 비동기) | |

- 라벨 찾기 1단계: 라벨 글만으로 PCA 하면 영상 질의와 공간이 달라(modality gap) 1위 일치가 0.34 였다. 같은 엔진으로 뽑은
  LVIS crop 9,753 개(평가셋과 다른 데이터)의 PCA 로 투영하고 질의에서 표본 평균을 빼면 같은 비용에 **0.65**(IVF 8, top 32),
  IVF 16·top 64 면 0.76. 128-bit 해밍 거름은 이 투영에선 1단계로 약해서 기본은 끄고(선택 `prefilter`) 128-d FP16 점수를 쓴다.
- 이름 정답률(C++ 찾기, 평가셋): 표 전체 30.5k 에서 1위 0.20 / 0.24 / 0.28 → **주 표(main 4,091 줄)만 0.29 / 0.36 / 0.36**. 그래서
  이름은 주 표에서 뽑는다. 상위어 올림은 평가 점수를 조금 낮춰(0.29 → 0.28) 보여 주는 이름은 1위로 두고 상위어는 `general` 로 따로 둔다.

### 8.5 시뮬 확인

- bringing_water public_test 0, 3000 스텝, FastSAM-s 416 + SigLIP 2(`outputs/clip_fastsam_20261003_075313`, 서브모듈): 물체 255 개 모두 이름.
  "white chair"·"흰 의자" → chair, "sofa"·"소파" → couch / recliner, "kitchen sink" → sink, "오븐" → oven. 이 장면엔 라디오가 없다.
- turning_on_radio public_test 0, 3000 스텝(`outputs/clip_fastsam_radio_20261003_081437`): 물체 269 개 모두 이름, CLIP 773 번(버림 0),
  keyframe 묶음 1–8, 엔진 GPU 0.6–0.8 ms(event), `sgc_submit` 2.6–3.9 ms(시뮬과 GPU 를 나눠 써서 자르기 커널 차례를 기다림).
  - 라디오 물체(O39, 빨간·흰 장난감 라디오 crop)의 이름은 **"extinguisher / 소화기"** 로 틀린다(2.2 의 "라디오는 이름 0" 과 같음).
  - 그래도 찾기로는 잡힌다: "red radio"·"빨간 라디오"·"portable radio" **1위**(코사인 0.12–0.13), "라디오" 3위, "radio" 5위(242 물체 중).
    라디오가 놓인 커피 테이블 crop(O34)이 바로 2위. → 3.5 결정(벡터가 원본, 이름은 캐시)이 맞다.
  - 그 밖: "white chair"·"흰 의자" → folding chair / chair, "sofa"·"소파" → sofa / couch, "television" → wall mounted tv, "coffee table" 1위.

### 8.6 남은 일 (Nano, TRT 8.2)

1. Nano 에서 엔진 빌드: TRT 8.2 파이썬으로 `build_engine.py`(`max_workspace_size`·`STRICT_TYPES` 분기 있음) — LayerNorm·GELU FP32
   고정이 8.2 에서도 먹는지(코사인)와 `--profiles 1,8` 메모리.
2. C++ 빌드: `encoder.cpp` 의 TRT 8 분기(바인딩 번호·`enqueueV2`)와 NEON 찾기는 컴파일만 생각해 짰고 아직 돌려 보지 않았다. JetPack 4.6 의
   gcc 7 은 C++17 `<filesystem>` 에 `-lstdc++fs` 가 필요할 수 있다. nlohmann json 헤더(3.x)가 필요하다.
3. 라벨 색인은 PC 에서 만들어 옮겨도 된다(같은 리틀 엔디언). 투영 표본 파일(`img_sample_lvis10k.f16`, 15 MB)도 같이.
4. Nano 실측으로 4절·8.4 의 추정을 바꾼다. FastSAM 과 같은 문맥에서 메모리(엔진 + 활성 0.2–0.3 GB 추정).
5. 한국어 학생(training/embed)이 나오면 `sgrt_query_embedding` 에 그 768-d 를 넣는다.

## 출처

- [MC] MobileCLIP 논문 Table 7·11 — https://arxiv.org/pdf/2311.17049 ; [MCgh] https://github.com/apple/ml-mobileclip
- [OCp] open_clip model_profile.csv — https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv
- [OCr] open_clip openclip_results.csv — https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
- [TC] TinyCLIP — https://github.com/wkcn/TinyCLIP
- [S2] SigLIP 2 논문 Table 1 — https://arxiv.org/html/2502.14786
- [PE] Perception Encoder — https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md
- [OV] OpenVision — https://github.com/UCSC-VLAA/OpenVision
- [JAL] Jetson AI Lab 벤치마크(Orin Nano CLIP) — https://jetson-ai-lab.com/archive/benchmarks.html
- [EN] EdgeNeXt 논문 Table 3(Jetson Nano TRT FP16) — https://arxiv.org/pdf/2206.10589
- [D10]·[12] 앞 문서 출처(Nano ResNet18 FP16, YOLOv8n-seg Nano)
- 한국어: https://huggingface.co/Bingsu/clip-vit-base-patch32-ko , https://huggingface.co/Bingsu/vitB32_bert_ko_small_clip , https://huggingface.co/hyunlord/siglip2-base-patch16-224-ko , https://github.com/Bing-su/KoCLIP_training_code , SBERT 다국어 증류 https://arxiv.org/abs/2004.09813 , https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
- DEEPX 동물원 CLIP — https://github.com/DEEPX-AI/dx-modelzoo
- meridian: `neoul-ro/meridian_frontend` `meridian_frontend/clip/clip.py`, `sam/sam.py`, `tracker/tracker.py`
