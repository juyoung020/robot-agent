# embed — 물체 이름 붙이기·찾기용 임베딩 (증류)

작성 2026-10-03. **작업 중**(학습 준비 단계). 숫자는 바뀔 수 있다.

## 목표

- 물체마다 저장하는 원본은 **SigLIP 2 B/32-256 영상 벡터 1개(768-d FP16)** 다([clip_candidates.md](../../docs/clip_candidates.md) 3.5).
  - 입력은 원본 RGB 를 10 % 여유로 정사각형으로 자른 것이고, 마스크는 MAP 풀링 가중치로만 쓴다(`emb_mask`).
- 그 위에 작은 MLP 를 학습한다.
  - (a) 128-d 투영. Matryoshka 라서 앞 64-d 만 써도 된다. 찾기·색인을 가볍게 하려는 것이다.
  - (b) 집 안 도메인 보정기(잔차 MLP). 큰 선생 모델의 영상 임베딩과 라벨 순위를 따라가게 한다.
  - (c) 한국어 질의 학생(작은 한국어 BERT). SigLIP 2 B/32 글 공간(또는 128-d 공간)으로 증류한다.
- 2단계(우리 픽셀 CNN 학생)는 1단계로 모자라거나 Nano 에서 기반 인코더가 너무 느릴 때만 한다. 지금은 계획만 있다.

## 선생 고르기 (평가셋 567개, `~/clip_bench/evalset.json`)

| 모델 | 자르기 | 이름 시연 top-1 / top-5 | 기억 top-1 | 찾기 영어 R@1 | 찾기 한국어 R@1 |
|---|---|---|---|---|---|
| OpenAI B/32 | box | 0.280 / 0.449 | 0.20 | 0.50 | 0.05 |
| MobileCLIP 2-S0 | box | 0.271 / 0.409 | 0.52 | 0.75 | 0.10 |
| **SigLIP 2 B/32-256 (기반)** | pool | 0.340 / 0.469 | 0.40 | 0.65 | 0.65 |
| SigLIP 2 So400m/16-384 | box | 0.286 / 0.469 | 0.56 | 0.65 | 0.75 |
| **PE-Core L/14-336** | box | 0.354 / 0.503 | 0.68 | 0.70 | 0.35 |
| PE-Core L/14-336 | stretch | 0.369 / 0.494 | 0.60 | 0.85 | 0.25 |
| EVA02-L/14-336 | box | 0.360 / 0.520 | 0.52 | 0.60 | 0.05 |
| PE-L + EVA-L 이어붙임 | box | 0.397 / 0.520 | 0.68 | 0.80 | – |

- 지표 정의는 `clip_bench/score.py` 를 그대로 가져다 쓴다. 우리 재현값(OpenAI B/32 box 0.280)은 그쪽 값과 같다.
- 큰 선생도 시연 top-1 이 0.35–0.40 에 그친다. 이 평가셋에서는 어휘(약 360개)와 엄격한 이름 맞추기가 상한을 정한다. 그래서 증류만으로 올릴 수 있는 폭이 작다.
- 선생은 PE-Core L/14-336(Apache-2.0)을 주로 쓴다. So400m(Apache-2.0, 다국어) 임베딩도 같이 저장한다.

## 데이터

| 무엇 | 양 | 라이선스 |
|---|---|---|
| LVIS v1 상자·마스크, COCO train2017 이미지 | crop 30만 개(이미지 8.1만 장, 범주 1,173개, 60 %는 집 안 가구가 있는 이미지) | 주석 CC BY 4.0, 이미지 Flickr(장마다 CC BY 계열) |
| 라벨: WordNet 3.1 명사(artifact·food·plant·animal·natural object·plant part) | 이름 28.8k | WordNet 라이선스(자유) |
| 라벨: LVIS 1,203 + 동의어, Open Images 600, BEHAVIOR-1K 범주 2,416, ovdet 272, COCO 80 | | CC BY 4.0 / MIT |
| 한국어 이름: Wikidata ko 이름·별칭(WordNet 3.1 id P8814, Freebase id P646) | 11.6k 이름 | CC0 |
| 한국어 이름 빈칸: NLLB-200-distilled-600M 번역 | 18.7k 이름 | **CC BY-NC**(연구용. 공개판 전에 바꾼다) |

- Objects365 는 연구 전용이라 쓰지 않았다.
- 시뮬 crop(BEHAVIOR 시연)은 평가 에피소드(e0·e57·e133)와 같은 집이다. 그래서 따로 표시한 변형으로만 쓴다(예정).

## 실행 순서

```
export PY=~/embed_venv/bin/python        # training/README.md 의 venv
$PY build_crops.py --n 300000            # ~/embed_work/data/lvis_crops/shard_*.tar + .jsonl (이미지는 받아서 자르고 버림)
$PY evalset.py score siglip2_b32 pe_l14  # 평가셋 임베딩 + zero-shot 숫자
$PY build_labels.py                      # ~/embed_work/labels/labels.jsonl
$PY encode_labels.py --nllb siglip2_b32 pe_l14
$PY extract.py                           # ~/embed_work/emb/lvis/<model>_<view>.npy (FP16)
$PY train_head.py --base siglip2_b32 --teacher pe_l14 --views pool,poolaug,box,aug
```

| 파일 | 하는 일 |
|---|---|
| `common.py` | 경로, open_clip 감싸개, 자르기(box / stretch / masked, clip_bench 와 같음), SigLIP 마스크 풀링 |
| `evalset.py` | 평가셋 임베딩과 점수 |
| `build_crops.py` | LVIS crop 만들기 |
| `extract.py` | 기반·선생 임베딩 뽑기 |
| `build_labels.py`, `encode_labels.py` | 라벨 표와 글 임베딩 |
| `train_head.py` | 1단계 MLP 머리 학습 |

## 라벨 표 형식 (런타임과 맞추는 약속)

라벨 표 하나 = 폴더 하나: `labels/<name>-<version>/`. 예 `labels/objects-v1/`.

```
labels/objects-v1/
├── manifest.json
├── table.jsonl          # 줄마다 이름 하나
├── text_siglip2_b32.f16 # K × 768 FP16, 행 우선, L2 정규화된 글 임베딩 (줄 순서 = table.jsonl)
└── text128_<head>.f16   # (선택) K × 128 FP16, 학습한 128-d 공간. 앞 64 열 = 64-d
```

`manifest.json`

```json
{"name": "objects", "version": "v1", "sha": "<sha256 앞 16자>", "count": 30533,
 "model": "siglip2_b32", "dim": 768, "templates": ["a photo of a {}.", "a photo of the {}.", "a cropped photo of a {}.", "a {} in a room."],
 "files": {"table.jsonl": "<sha256>", "text_siglip2_b32.f16": "<sha256>"}, "created": "2026-10-03"}
```

- `sha` 는 `files` 의 sha256 값들을 이름 순으로 이어 붙여 다시 sha256 한 것의 앞 16자다. memory 캐시의 `cache/names.json` `table.sha` 에 이 값을 적는다.
- 글 임베딩은 문장 틀 4개의 L2 정규화 평균을 다시 L2 정규화한 것이다.

`table.jsonl` 한 줄

```json
{"i": 215, "en": "sofa", "en_syn": ["couch", "lounge"], "ko": ["소파", "쇼파", "카우치"], "ko_src": "wikidata",
 "synset": "sofa.n.01", "wn31": "04263257-n", "hypernyms": ["seat.n.03", "furniture.n.01"],
 "tier": "main", "structural": false, "src": ["lvis", "behavior1k", "wordnet"]}
```

| 열 | 뜻 |
|---|---|
| `i` | 줄 번호 = 임베딩 행 번호 |
| `en` | 영어 이름(소문자, 공백). 이름 하나가 한 줄이다. 동의어도 자기 줄을 가진다 |
| `en_syn` | 같은 synset 의 다른 영어 이름(표시용) |
| `ko` | 한국어 이름. 첫 번째가 대표 이름이다. 없으면 `[]` |
| `ko_src` | `wikidata` / `wikidata_synset`(같은 synset 의 다른 이름에서 옴) / `nllb`(기계 번역, 품질 낮음) / `manual` |
| `synset` | WordNet 이름(`chair.n.01`). WordNet 에 없는 BEHAVIOR 이름은 그쪽 synset 문자열 그대로, 없으면 `""` |
| `wn31` | WordNet 3.1 id(`03005231-n`) 또는 `""` |
| `hypernyms` | 가까운 상위어부터 `entity.n.01` 바로 아래까지의 synset 이름. 확신이 낮을 때 상위 이름으로 물러나는 데 쓴다 |
| `tier` | `main`(집 물건 주 표: BEHAVIOR·COCO·LVIS·Open Images·ovdet·평가 어휘) / `tail`(WordNet 긴 꼬리) |
| `structural` | 벽·바닥·천장·문·창·지붕·계단·난간·걸레받이 같은 구조물이면 `true` |
| `src` | 이름이 나온 곳 |

## 진행

- [x] venv, 평가 하네스(clip_bench 숫자 재현), 선생 고르기
- [x] LVIS crop 30만 개, 라벨 표 30.5k(한국어 Wikidata 11.6k + NLLB)
- [ ] 임베딩 뽑기(기반 SigLIP 2 B/32 pool·box·aug, 선생 PE-L·So400m)
- [ ] 1단계 MLP 머리 학습과 평가(128/64-d, 큰 라벨 표 이름 붙이기 포함)
- [ ] 한국어 질의 학생(Bingsu/lassl BERT-small → SigLIP 2 B/32 글 공간)
- [ ] 라벨 표를 위 형식(`labels/objects-v1/`)으로 내보내기
- [ ] 같은 집 시뮬 변형, 다른 BEHAVIOR 집 평가셋
