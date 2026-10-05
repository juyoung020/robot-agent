# embed — 물체 이름 붙이기·찾기용 임베딩 (증류)

작성 2026-10-03. 1단계(작은 MLP 머리) 시범 학습까지 했다. 숫자는 시범 값이다.

## 목표와 구조

- 물체마다 저장하는 원본은 **SigLIP 2 B/32-256 영상 벡터 1개(768-d FP16)** 다([clip_candidates.md](../../docs/clip_candidates.md) 3.5).
  - 입력은 원본 RGB 를 10 % 여유로 정사각형으로 자른 것이다. 마스크는 MAP 풀링 가중치로만 쓴다(아래 `pool`).
- 그 벡터 위에 작은 머리를 학습한다(**1단계**).
  - (a) 128-d 투영. Matryoshka 라서 앞 64-d 만 써도 된다.
  - (b) 집 안 도메인 보정. 잔차 MLP 하나(1.67M 변수)이고, 큰 선생(PE-Core L/14) 영상 임베딩과 라벨 순위를 따라가게 한다.
  - (c) 한국어 질의 학생. 작은 한국어 BERT(23M)를 SigLIP 2 B/32 글 공간과 PE-L 글 공간으로 증류한다.
- 128-d 공간은 선생 공간에서 선형 투영 `P`(1024→128)로 정한다.
  - 라벨은 `l2(P · 선생글(라벨))` 이다. 새 라벨은 PC 에서 선생 글 탑 + `P` 로 만든다.
  - 머리 `h` 는 `SigLIP2 벡터 → P·선생영상` 을 배운다. 손실은 코사인 + 라벨 3만 개 순위 KL(128·64-d) + 배치 관계.
- **2단계**(우리 픽셀 CNN 학생)는 하지 않았다. 계획만 아래에 적었다.

## 결과 요약

- **MLP 머리만으로 SigLIP 2 B/32 를 선생(PE-L)만큼, 일부는 선생보다 낫게** 만든다(평가셋 A, 같은 집).
  - 시연 top-1: 0.340 → 0.377(선생 0.354).
  - 깨끗한 부분집합: 0.398 → 0.482(선생 0.295).
  - 라벨 3만 개에서 top-5: 0.249 → 0.391(선생 0.280).
- **다른 집(B)에서는 차이의 1/3 쯤만 메운다.**
  - top-1: 0.080 → 0.102–0.119(선생 0.190).
  - B 는 정답 잡음이 커서(아래) 방향만 본다.
- **선형 투영 하나(0.1M)로도 MLP 와 거의 같다.** 평가셋이 작아(물체 350개, 표준오차 ±2.5 %p) 차이는 잡음 안이다.
- 생 데이터를 6만 → 30만 개로 늘려도 숫자가 같다. 더 넣어야 할 것은 **양이 아니라 집 안 도메인 데이터**다.
- 한국어 학생(23M)은 처음 보는 한국어 이름 → 라벨 top-5 0.697 이다. SigLIP 2 자체 글 탑(282M)은 0.631. top-1 은 0.453 대 0.522.
- 라벨 찾기는 INT8 전수 + FP32 재순위 16개로 정확값과 top-1 이 99.8–100 % 같다.
  - PC 1스레드 30.5k 라벨 기준: 768-d 0.6 ms, 128-d 0.13 ms.
  - 2진 부호는 영상↔글(서로 다른 공간) 이름 붙이기에 쓸 수 없다.

## 평가셋

- **A**: `$CLIP_BENCH/evalset.json`(CLIP 조사 것, 읽기만). 567개.
  - 내용: turning_on_radio 시연 ep 0·57·133 FastSAM crop 542개 + 시뮬 기억 best view 25개. 집은 house_double_floor_lower.
  - 지표는 `clip_bench/score.py` 를 import 해서 그대로 쓴다. 우리 재현값(OpenAI B/32 box 0.280, SigLIP 2 B/32 pool 0.340)은 그쪽 값과 같다.
- **B**(다른 집): `make_evalset_other.py` → `training/data/embed/eval_other/`. 406개.
  - putting_up_Christmas_decorations_inside ep 1800–1802, house_single_floor.
  - 같은 규칙: 마스크 60 % 이상이 한 정답 물체, IoU ≥ 0.2, 범주당 ≤ 60.
  - 구조물이 아닌 것은 약 190개(소파·아침 식탁·스탠드·트리·의자 …).
  - 에피소드가 길어(9분) 오도메트리 정답이 밀리고, 'sofa' 정답이 둥근 오토만인 등 잡음이 크다. 찾기 질의는 5개뿐이라 무시한다.
- 열 이름
  - `v360`: 어휘 약 360개(clip_bench 와 같음). `big`: 라벨 표 전체 30,533개.
  - `demo` = 시연 물체(구조물 제외), `clean` = IoU ≥ 0.5, `mem` = 시뮬 기억 25개.
  - `en R@1`: 질의 20개. `kolk` = 한국어 질의를 라벨 표 한국어 이름으로 찾아 그 행의 글 벡터를 씀. `kost` = 한국어 학생.

### 선생 고르기 (A, 각 모델 자체 글 탑)

| 모델 | 자르기 | 시연 top-1 / top-5 | 깨끗 | 기억 | 찾기 영어 R@1 | 찾기 한국어 R@1 |
|---|---|---|---|---|---|---|
| OpenAI B/32 | box | 0.280 / 0.449 | 0.295 | 0.20 | 0.50 | 0.05 |
| MobileCLIP 2-S0 | box | 0.271 / 0.409 | 0.271 | 0.52 | 0.75 | 0.10 |
| **SigLIP 2 B/32-256 (기반)** | pool | 0.340 / 0.469 | 0.398 | 0.40 | 0.65 | 0.65 |
| SigLIP 2 B/16 | box | 0.283 / 0.451 | 0.295 | 0.52 | 0.75 | 0.55 |
| SigLIP 2 So400m/16-384 | box | 0.286 / 0.469 | 0.265 | 0.56 | 0.65 | 0.75 |
| **PE-Core L/14-336 (선생)** | box | 0.354 / 0.503 | 0.295 | 0.68 | 0.70 | 0.35 |
| EVA02-L/14-336 | box | 0.360 / 0.520 | 0.319 | 0.52 | 0.60 | 0.05 |
| PE-L + EVA-L 이어붙임 | box | 0.397 / 0.520 | 0.349 | 0.68 | 0.80 | – |

- 큰 선생도 시연 top-1 이 0.35–0.40 에 그친다. 이 평가셋은 정답 잡음과 엄격한 이름 맞추기가 상한을 정한다.
- 선생은 PE-Core L/14-336(Apache-2.0)으로 정했다. 기억 crop 에서 가장 좋다.

### 1단계 머리 (LVIS crop 30만, 12 epoch, 학습 37 s)

| 실행 | 평가셋 | 시연 top-1 / top-5 | 깨끗 | 기억 | big top-1 / top-5 | 영어 R@1 | 한국어 R@1 kolk / kost |
|---|---|---|---|---|---|---|---|
| 기반 SigLIP 2 B/32 pool(768) | A | 0.340 / 0.469 | 0.398 | 0.40 | 0.160 / 0.249 | 0.65 | 0.60 / 0.55 |
| 선생 PE-L box(1024) | A | 0.354 / 0.503 | 0.295 | 0.68 | 0.191 / 0.280 | 0.70 | 0.60 / – |
| **MLP 머리 128-d** `sb32_pe_300k` | A | **0.377** / 0.477 | **0.482** | 0.44 | 0.189 / **0.391** | 0.70 | 0.70 / 0.55 |
| MLP 머리 앞 64-d | A | 0.343 / 0.483 | 0.410 | 0.44 | 0.166 / 0.391 | 0.60 | 0.60 / 0.50 |
| 선형 128-d `sb32_pe_300k_lin` | A | 0.354 / 0.480 | 0.428 | 0.56 | 0.194 / 0.371 | 0.70 | 0.60 / – |
| 선형 앞 64-d | A | 0.366 / 0.477 | 0.446 | 0.52 | 0.183 / 0.371 | 0.75 | 0.55 / – |
| OpenAI B/32 + MLP 128 `b32_pe_300k` | A | 0.283 / 0.434 | 0.349 | 0.28 | 0.169 / 0.351 | 0.70 | – |
| 기반 SigLIP 2 B/32 pool | B | 0.080 / 0.190 | 0.155 | – | 0.035 / 0.071 | – | – |
| 선생 PE-L box | B | 0.190 / 0.239 | 0.262 | – | 0.084 / 0.173 | – | – |
| MLP 머리 128-d | B | 0.102 / 0.217 | 0.167 | – | 0.053 / 0.115 | – | – |
| 선형 128-d | B | 0.119 / 0.199 | 0.190 | – | 0.053 / 0.111 | – | – |

- 같은 설정을 6만 개(`sb32_pe_p60k`), 12만 개(`sb32_pe_p120k`)로 돌려도 A 시연 top-1 이 0.377 로 같다.
- 손실·크기 바꾸기(KL ×3, 2층 2048, pool 보기만, τ 0.02)도 0.32–0.38 안에서 움직인다. 잡음 안이다.
- "MLP 정도면 충분한가": 이 평가셋에서는 **충분하다. 선형으로도 거의 된다.**
  - 머리가 못 하는 것은 기반 벡터에 없는 정보다. 다른 집 B 에서 선생과의 차이 0.08–0.11 이 그 몫이다.
  - 이것을 메우려면 집 안 도메인 데이터(여러 집 시뮬 crop)를 넣거나, 2단계로 픽셀부터 다시 배워야 한다.

### 한국어 질의 학생 (`train_ko.py`, lassl/bert-ko-small 23M, 8 epoch 약 11분)

| 무엇 | 처음 보는 한국어 이름 439개 → 주 표 라벨 4,091개 top-1 / top-5 | 영상 찾기 20질의 R@1 (A, pool) |
|---|---|---|
| SigLIP 2 B/32 자체 글 탑(282M), 한국어 그대로 | 0.522 / 0.631 | 0.65 |
| **학생 `ko_small_ho`**(이름 1/5 학습에서 뺌) | 0.453 / **0.697** | 0.55 |
| 학생 `ko_small_v2`(모든 이름 학습, 참고) | 0.631 / 0.859 | 0.55 |

- 짝 데이터는 35.3만 개다.
  - 라벨 표 한국어 이름 + 틀 15.3만: "빨간 {ko}" → "a photo of a red {en}.", 색·재질·크기·장소.
  - 한–영 문장 20만: Moo/korean-parallel-corpora CC BY-SA 3.0, lemon-mint/korean_parallel_sentences_v1.1 MIT.
- 손실: MSE + 코사인 + 배치 대조(τ 0.05, 무게 1.0). 대조 무게 0.2(`ko_small`)일 때 영상 찾기는 0.45 였다.
- 출력은 두 개다. SigLIP 2 B/32 글 공간(768, 저장 벡터와 바로 비교)과 PE-L 글 공간(1024, `P` 를 곱하면 128-d).
- 평가 질의 20개 중 "소파"·"의자" 같은 이름은 학습 짝에 들어 있다. 그래서 영상 찾기 숫자는 새는 쪽(낙관)이다.

### 라벨 찾기 지연 (`lookup_bench.cpp`, PC 1스레드, 다른 일로 CPU 가 바쁨)

| 공간 | FP32 전수 | INT8 전수 + FP32 재순위 16 | 2진 부호 + 재순위 128 |
|---|---|---|---|
| 768-d, 라벨 30.5k | 4.4 ms | **0.60 ms**, top-1 일치 0.998 | 0.19 ms, 일치 0.868 |
| 128-d (머리) | 0.78 ms | **0.13 ms**, 일치 1.000 | 0.08 ms, 일치 0.335 |
| 64-d | 0.11 ms | 0.08 ms, 일치 1.000 | 0.07 ms, 일치 0.187 |

- 2진 부호는 영상과 글이 다른 원뿔에 있어 이름 붙이기에 맞지 않는다. 평균을 빼도 128-d 0.53 이다.
  - 물체↔물체(같은 영상 공간) 중복 찾기에만 쓴다.
- 3만 개에서는 IVF 가 필요 없다. 라벨이 30만 개를 넘으면 다시 본다.
- **Nano(A57, 추정)**: PC 대비 6–10배 느리다고 보면 768-d INT8 + 재순위 4–6 ms, 128-d 1 ms 안팎. 물체당 한 번이고 캐시되므로 충분하다.

### 머리 지연

- ONNX(opset 13, `export_head.py`)의 연산은 LayerNorm, MatMul, Erf, Div 이다.
- PC: TensorRT FP16 배치 1·8 모두 GPU 0.02 ms. onnxruntime CPU 1스레드 배치 1 0.06 ms, 배치 8 0.14 ms.
- **Nano(추정)**: CPU 로 물체당 0.5–1 ms. 기반 SigLIP 2 B/32(물체당 50–120 ms, clip_candidates 4절)에 비하면 0 에 가깝다. GPU 로 돌릴 이유가 없다.

## 데이터

| 무엇 | 양 | 라이선스 |
|---|---|---|
| LVIS v1 상자·마스크, COCO train2017 이미지 | crop 30만 개(이미지 8.1만 장, 범주 1,173개, 60 %는 집 안 가구가 있는 이미지). 288 px, 1.3배 여유, 마스크 png | 주석 CC BY 4.0, 이미지 Flickr(장마다 CC BY 계열) |
| BEHAVIOR 시연 FastSAM crop(라벨 없음) | 9.6만 개, radio 시연 196편, 5초마다 | MIT(데이터셋). **평가 A 와 같은 집** |
| 다른 집 시연(평가 B) | 3편, 0.9 GB | MIT |
| 라벨: WordNet 3.1 명사(artifact·food·plant·animal·natural object·plant part) | 이름 28.8k | WordNet 라이선스(자유) |
| 라벨: LVIS 1,203 + 동의어, Open Images 600, BEHAVIOR-1K 범주 2,416, ovdet 272, COCO 80 | 주 표 4,091 | CC BY 4.0 / MIT |
| 한국어 이름: Wikidata ko 이름·별칭(WordNet 3.1 id P8814, Freebase id P646) | 11.6k 이름 | CC0 |
| 한국어 이름 빈칸: NLLB-200-distilled-600M 번역 | 18.7k 이름 | **CC BY-NC**(연구용. 공개판 전에 사람 검수나 다른 번역으로 바꾼다) |

- Objects365 는 연구 전용이라 쓰지 않았다. KorLex 는 사용 허가를 받으면 동의어 보강에 쓴다(아직 없음).
- 우리말샘 동의어는 아직 넣지 않았다.

## 실행 순서

```
export PY=$EMBED_PY        # training/README.md 의 venv
$PY build_crops.py --n 300000            # training/data/embed/data/lvis_crops/shard_*.tar + .jsonl (이미지는 받아서 자르고 버림, 약 40분)
$PY build_sim_crops.py --every 150       # training/data/embed/data/sim_radio (FastSAM, 같은 집)
$PY make_evalset_other.py --every 75     # 평가 B
$PY evalset.py score siglip2_b32 pe_l14  # 평가셋 임베딩 + zero-shot (EVALSET=other 로 B)
$PY build_labels.py && $PY encode_labels.py --nllb siglip2_b32 pe_l14
$PY extract.py --base siglip2_b32 --teacher pe_l14       # 30만 개 약 2.3시간(GPU 를 Isaac Sim 과 나눠 씀), RAM ≤ 12 GB
$PY train_head.py --base siglip2_b32 --teacher pe_l14 --views pool,poolaug,box,aug --name sb32_pe_300k
$PY eval_head.py sb32_pe_300k            # KO_RUN=ko_small_ho 로 한국어 학생 열 추가
$PY train_ko.py --name ko_small_ho --holdout && $PY eval_ko.py ko_small_ho
$PY export_head.py sb32_pe_300k && $PY export_labels.py --head sb32_pe_300k
g++ -O3 -march=native -std=c++17 lookup_bench.cpp -o training/data/embed/lookup_bench
```

- `extract.py` 주의
  - 워커는 spawn 으로 띄운다. fork 하면 모델 무게가 워커마다 복사돼 RAM 이 터진다(10-03 06:58 PC 멈춤).
  - 워커는 6개 이하로 둔다. 끝난 shard 는 `done_shards.txt` 에 적혀서, 다시 돌리면 이어서 한다.

| 파일 | 하는 일 |
|---|---|
| `common.py` | 경로, open_clip 감싸개, 자르기(box / stretch / masked, clip_bench 와 같음), SigLIP 마스크 풀링(`image_pool`) |
| `evalset.py`, `eval_head.py`, `eval_ko.py` | 평가 |
| `build_crops.py`, `build_sim_crops.py`, `make_evalset_other.py` | 데이터 |
| `extract.py` | 기반·선생 임베딩 뽑기(FP16 .npy) |
| `build_labels.py`, `encode_labels.py`, `export_labels.py` | 라벨 표 |
| `train_head.py`, `export_head.py` | 1단계 머리 |
| `train_ko.py` | 한국어 학생 |
| `lookup_bench.cpp` | 라벨 찾기 지연·정확도 |

## 라벨 표 형식 (런타임과 맞추는 약속)

라벨 표 하나 = 폴더 하나: `labels/<name>-<version>/`. 지금 것은 `models/labels/objects-v1/`(65 MB)이다.

```
labels/objects-v1/
├── manifest.json
├── table.jsonl                 # 줄마다 이름 하나
├── text_siglip2_b32.f16        # K × 768 FP16, 행 우선, L2 정규화된 글 임베딩 (줄 순서 = table.jsonl)
└── text128_<head>.f16          # (선택) K × 128 FP16, 머리 공간. 앞 64 열을 다시 정규화하면 64-d
```

`manifest.json`

```json
{"name": "objects", "version": "v1", "sha": "<sha256 앞 16자>", "count": 30533,
 "model": "siglip2_b32", "dim": 768, "templates": ["a photo of a {}.", "a photo of the {}.", "a cropped photo of a {}.", "a {} in a room."],
 "files": {"table.jsonl": "<sha256>", "text_siglip2_b32.f16": "<sha256>"}, "created": "2026-10-03", "counts": {"main": 4091}}
```

- `sha` 는 `files` 의 sha256 값들을 이름 순으로 이어 붙여 다시 sha256 한 것의 앞 16자다. memory 캐시의 `cache/names.json` `table.sha` 에 이 값을 적는다.
- 글 임베딩은 문장 틀 4개의 L2 정규화 평균을 다시 L2 정규화한 것이다.
- 같은 입력이면 같은 표가 나온다(`build_labels.py` 는 정렬된 순서로 만든다).

`table.jsonl` 한 줄

```json
{"i": 215, "en": "sofa", "en_syn": ["couch", "lounge"], "ko": ["소파", "쇼파", "카우치"], "ko_src": "wikidata",
 "synset": "sofa.n.01", "wn31": "04263630-n", "hypernyms": ["seat.n.03", "furniture.n.01", "furnishing.n.02", "..."],
 "tier": "main", "structural": false, "src": ["lvis", "behavior1k", "wordnet"]}
```

| 열 | 뜻 |
|---|---|
| `i` | 줄 번호 = 임베딩 행 번호 |
| `en` | 영어 이름(소문자, 공백). 이름 하나가 한 줄이다. 동의어도 자기 줄을 가진다 |
| `en_syn` | 같은 synset 의 다른 영어 이름(표시용, 최대 8개) |
| `ko` | 한국어 이름. 첫 번째가 대표 이름이다. 없으면 `[]`(194줄) |
| `ko_src` | `wikidata` / `wikidata_synset`(같은 synset 의 다른 이름에서 옴) / `nllb`(기계 번역, 품질 낮음) / `manual` |
| `synset` | WordNet 이름(`chair.n.01`). WordNet 에 없는 BEHAVIOR 이름은 그쪽 synset 문자열 그대로, 없으면 `""` |
| `wn31` | WordNet 3.1 id(`03005231-n`) 또는 `""` |
| `hypernyms` | 가까운 상위어부터 `entity.n.01` 바로 아래까지의 synset 이름. 확신이 낮을 때 상위 이름으로 물러나는 데 쓴다 |
| `tier` | `main`(집 물건 주 표: BEHAVIOR·COCO·LVIS·Open Images·ovdet·평가 어휘, 4,091) / `tail`(WordNet 긴 꼬리) |
| `structural` | 벽·바닥·천장·문·창·지붕·계단·난간·걸레받이 같은 구조물이면 `true`(105줄) |
| `src` | 이름이 나온 곳 |

머리(128-d)를 쓰려면 런타임은 `head128.onnx`(입력 `emb` N×768 = 저장 벡터, 출력 `emb128` N×128, L2 정규화됨)와 `text128_<head>.f16` 을 쓴다.

## VLA 이름·생김새·지시 표 `vla_v1` (VLA_INPUT 3·6·7절)

작성 2026-10-04. 물체 칸의 **이름 뜻 벡터 128**, **생김새 벡터 128**, 지시 문장 벡터를 이 문서의 128-d 공간(얼린 `P`, `sb32_pe_300k`)에 둔다.
VLA_INPUT 의 "얼린 SigLIP 2 글 인코더 → 128-d 투영" 은 이 공간의 약속으로 구현한다: 영어 글 = PE-Core L/14 글 탑 + `P`, 한국어 글 = 한국어 학생(`ko_small_v2` 의 pe_l14 출력) + `P`,
영상 = SigLIP 2 B/32-256 풀링 벡터 + 머리 h. 셋 다 같은 얼린 `P` 공간이라 지시 ↔ 이름 ↔ 생김새를 바로 비교한다. 학습 중에는 아무 모델도 돌리지 않고 표만 읽는다.

```
HF_HUB_OFFLINE=1 $EMBED_PY training/embed/vla_tables.py           # 이름·지시 (CPU 약 12 s, 내려받기 없음)
$EMBED_PY training/embed/export_head_f32.py                       # 머리 h → training/data/embed/runs/sb32_pe_300k/head_h.f32 (git 밖)
training/runs/bc/build/app_table --views 32                                               # 생김새 (C++/CUDA, 1 s)
python3 training/embed/vla_vocab_gen.py                                          # 확신도 표 → training/RL/map/include/vla_vocab.h, manifest
training/runs/bc/build/app_table --views 12 --dump D [--negative 1|2|3] && $EMBED_PY training/BC/tools/app_ref.py D   # 검증
```

| 파일(`training/data/vla_v1/`, 합 440 KB) | 내용 |
|---|---|
| `names.jsonl`, `name128.f16` [584][128] | 이름 584 줄 = 동의어 묶음 395 개(묶음마다 줄이 붙어 있음). 시뮬 종류 6 묶음(gmap::Cls 차례) 24 줄, BEHAVIOR 2026 범주 252 묶음, 동의어 189 줄, 상위어 137 줄. 584 줄 모두 `objects-v1` 표의 text128 그대로(표에 없는 이름 0 — PE-L 로 다시 계산한 표 줄 16 개는 표와 코사인 최소 0.99995) |
| `name_aux.i32` [584][8] | 상위어 줄, 묶음 시작, 묶음 길이, 비슷한 다른 이름 3(코사인 상위, 다른 묶음 대표), 안 보인 이름(평가용) 표시, 종류(0 시뮬, 1 BEHAVIOR, 2 동의어, 3 상위어) |
| `instr.jsonl`, `instr128.f16` [40][128] | 과제 6 개(지금 과제 `go_to_cup` 13 문장 + 집기·놓기 둘·열기·닫기 틀) × 영어·한국어, 과제마다 안 보인 바꿔 말하기 표시 |
| `app128.f16` [7][128], `app_views.f16` [7][64][128] | 생김새: 줄 0–5 = 시뮬 종류, 줄 6 = 유령(가짜 검출 = 물체 없는 바닥·벽 조각). 줄마다 시점 64 개 평균 |

- 시뮬 종류의 이름: 컵 = cup(+ teacup / 평가용 mug·tumbler), 작은 물건 = **box**(env.h 의 작은 물건은 0.08–0.25 m 상자로 그려짐 → 생김새 그대로; + carton·package / crate), 의자 = chair(+ straight chair / side chair), 탁자 = table(+ dining·kitchen table / worktable), 장 = cabinet(+ cupboard·bottom cabinet / sideboard), 쓰레기통 = trash can(+ garbage can·wastebin·dustbin, BEHAVIOR 의 ashcan / wastebasket). 상위어: tableware, container, furniture, furniture, furniture, container.
- BEHAVIOR 2026 범주: 과제 100 개(`datasets/2026-challenge-task-instances/metadata/task.jsonl`)의 `problem0.bddl` `:objects` synset 259 개(agent 뺌) + `task_custom_lists.json` 의 허용 모델 범주 108 개를 그 synset 묶음에 넣음. 시뮬 종류와 같은 synset(cup·box·chair·table·cabinet·ashcan) 6 개는 시뮬 묶음에 합침. 상위어 = WordNet 첫 상위어 중 너무 추상적이지 않고 표에 있는 것, 없으면 "X of Y" → X. 상위어 없는 대표 19 개(container, dust, sand, lawn …).
- 지시 표(잰 값): 같은 과제 영어끼리 코사인 평균 0.87, 한국어끼리 0.65, 영어–한국어 0.59, 다른 과제 0.40(최대 0.957 = "open the cabinet" 대 "close the cabinet" — 글 탑이 열기·닫기를 거의 못 가름). 가장 가까운 다른 문장의 과제가 맞는 비율 0.75(40 개). `go_to_cup` 문장과 cup 이름 행 코사인 평균 0.747(chair 0.241).

### 생김새 경로(C++/CUDA) — `training/BC/tools/app_table.cu`, `app_head.{h,cpp}`
팀 `RenderBatch`(읽기만)로 종류마다 상자 하나를 bc_render 와 같은 재질·색·조명 방에 놓고 물체를 겨눈 조각(경계 구가 화면의 1/1.1, FOV 50°, 시선 −6°–52°, 싼 설정) 256² 을 그린다
→ 우리 C++ SigLIP 2 영상 탑(`vit::Encoder`, 끝 LN 패치 토큰 bf16) → C++ MAP 풀링 머리(safetensors `attn_pool`, double) + 머리 h(double) → L2.

| 검사 | 결과 |
|---|---|
| 같은 토큰, C++ 풀링+머리 대 PyTorch FP32(open_clip `attn_pool` + `train_head.Head`), 영상 168 장 | 풀링 768 코사인 최소 1.0000000(상대 L2 6.7e-7), 128 코사인 최소 1.0000000 |
| 처음부터 끝까지: 같은 RGB 를 PyTorch FP32 open_clip 영상 탑 + 머리 h 대 C++ 탑(FP16) + C++ 풀링·머리 | 128 코사인 평균 1.00000, 최저 0.99996. 표 줄(평균) 7 개 모두 1.00000 |
| 음성 대조(`--negative`): 1 어텐션 1/8 배율 빠뜨림 / 2 머리 LN 빠뜨림 / 3 MAP MLP 잔차 빠뜨림 | 128 코사인 최소 0.566 / 0.581 / 0.169 → 셋 다 실패(정상) |
| 다른 씨앗(시점) / 팀 기본 렌더 설정과 표 줄 코사인 | 0.997–0.999 / 0.934–0.987 |

**결과가 말하는 것(잰 값)**: 색만 다른 상자는 생김새로 거의 가려지지 않는다. 종류 평균끼리 코사인 0.92–0.98(유령과는 0.81–0.89), 종류 안 시점끼리 0.89–0.93. 라벨 어휘(아래)에서 1위 이름은 컵 줄 footstool, 나머지 상자 다섯 wastebin, 유령 furniture sink.
실제 물체 메시(BEHAVIOR 자산, CURRICULUM_BEHAVIOR2026 E4)로 그리면 다시 만들어야 한다 — 같은 도구에 상자 대신 메시를 넣으면 된다.

### 이름 확신도·상위어 문턱 — `training/embed/vla_vocab_gen.py` → `training/RL/map/include/vla_vocab.h`(생성, 손으로 고치지 않음)
- conf1[app][cls] = cos(app128[app], name128[cls 이름]), conf2 = conf1 − (이름 어휘에서 cls 동의어 묶음 밖 최대 코사인). 이름 어휘 = vla_v1 에서 상위어·평가용 줄을 뺀 440 줄(시뮬 + 동의어 + BEHAVIOR 범주) — 우리 커리큘럼에서 검출기가 낼 수 있는 이름.
- conf2 분포(시점별, 백분위 5/25/50/75/95): 맞는 이름 −0.191/−0.143/−0.100/−0.048/0.022, 틀린 이름 −0.191/−0.142/−0.105/−0.060/0.001, 유령 −0.194/−0.149/−0.124/−0.100/−0.061. 맞는 이름과 틀린 이름이 거의 같다(위 상자 렌더 때문).
- `kConfLow` = −0.08: 표(평균 줄)에서 유령 6/6 이 상위어로, 맞는 이름은 3/6(box, table, trash can) 유지, 틀린 이름 18/30 이 상위어로. −0.10 이면 유령 5/6·맞는 3/6·틀린 15/30, −0.06 이면 맞는 1/6 만 유지. **(가정)** 상자 렌더에서는 맞는 이름을 다 지키며 틀린 이름을 거르는 문턱이 없다 — 메시 렌더로 표를 다시 만든 뒤 다시 정한다.

## 2단계 계획 (우리 픽셀 CNN 학생, 아직 안 함)

- **할 조건**(둘 중 하나)
  - Nano 실측에서 SigLIP 2 B/32 가 keyframe 예산(새 물체 5개에 0.5 s)을 넘는다.
  - 집 안 데이터를 넣은 1단계가 다른 집 평가에서 선생과의 차이를 절반 이상 남긴다.
- **모양**
  - RepVGG/MobileNetV3 꼴 CNN, 5–8M 변수, 입력 160–192, 출력 768(SigLIP 2 공간, 저장 벡터를 그대로 대신함) + 128 머리.
  - 연산은 Conv·ReLU·Add·GlobalPool 만 쓴다(TRT 8.2 INT8 보정이 쉬움). 마스크는 4번째 입력 채널로 넣거나 마지막 풀링 가중치로 쓴다.
- **학습**
  - 목표는 SigLIP 2 B/32 `pool` 벡터(코사인)와 PE-L 의 128-d 라벨 순위 KL 이다.
  - 데이터는 지금 crop(LVIS 30만 + 시뮬 9.6만)을 그대로 쓴다. 픽셀이 필요해서 crop 을 지우지 않는다.
  - GPU 하나로 1–2일(추정).
- **Nano(추정)**: 0.3–0.6 GFLOPs 라 FP16 물체당 5–10 ms. B/32 의 50–120 ms 보다 약 10배 빠르다.
- **대가**: 저장 벡터가 SigLIP 2 가 아니게 되면 SigLIP 2 글 탑(다국어)을 그대로 못 쓴다. 학생 공간의 글은 128-d 라벨 표와 한국어 학생만 쓴다.

## 전체 규모로 키우면

- 숫자가 6만 → 30만에서 멈췄다. 그래서 일반 사진(LVIS·Open Images 전부)을 더 넣기보다 **집 안 시뮬 crop 을 여러 집에서** 모으는 쪽이 먼저다.
- 뽑기 속도(이번 측정): GPU 를 Isaac Sim 과 나눠 쓸 때 35 crop/s. 기반 6보기 + OpenAI B/32 4보기 + PE-L 2보기 기준이다.
  - 혼자 쓰고 기반 4보기 + PE-L 2보기면 약 60/s(추정).
- 200만 crop(LVIS 전부 약 70만 + Open Images 100만 + 시뮬 30만)이면 아래와 같다.
  - 뽑기 약 10시간(추정).
  - 디스크: crop 약 40 GB(20 KB/개) + 임베딩 약 20 GB(FP16, 6보기).
  - 머리 학습은 10분 안쪽이다.
- 지금 디스크(`training/data/embed`)는 18 GB 다(crop 9.2, 임베딩 5.0, 한국어 목표 2.6, 기타).
  - 따로 venv 7.5 GB, HF 캐시 약 8 GB(So400m 4.3, PE-L 2.6, EVA-L 0.8 …)를 쓴다.

## 남은 일

- [ ] 같은 집 시뮬 crop 을 넣은 변형(`sb32_pe_sim`). A 와 같은 집이라 따로 표시한다.
- [ ] 여러 집 시뮬 crop(다른 과제 시연 받기) → 다른 집 B 로 일반화 확인. 평가 B 는 정답 잡음을 줄여 다시 만든다(짧은 구간, 정답 자세).
- [ ] 한국어: 평가 질의 100개 이상(clip_candidates 7-5), NLLB 이름 검수, 우리말샘 동의어.
- [ ] Nano 실측(TRT 8.2): 기반 B/32, 머리 CPU, 라벨 찾기 NEON.
- [x] 런타임: `objects-v1` 표를 port 쪽 라벨 찾기·`cache/names.json` 과 맞춘다(형식은 위 약속).
