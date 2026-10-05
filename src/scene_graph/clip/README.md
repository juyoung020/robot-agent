# sgclip — 물체 영상 임베딩(SigLIP 2 B/32-256) + 라벨 찾기 + 물체 찾기

물체 기억(scenemap)의 물체마다 **영상 벡터 1개(768-d)** 를 만들고, 그 벡터로 이름을 붙이고(라벨 표), 글로 찾는다.
**물체 찾기(`sgsearch.h`)** 는 에이전트 도구(`robot-agent/src/agent/tools/search_objects`, 글 결과)와 RecallVLA(자기 질의 벡터 → 상위 K 칸)가
같이 쓰는 공용 색인이다 — 아래 "물체 찾기".
설계·측정은 [docs/clip_candidates.md](../../../docs/clip_candidates.md)(상위 저장소) 3.5·6·8절.

- 모델: SigLIP 2 B/32-256(Apache-2.0, open_clip `ViT-B-32-SigLIP2-256` / `webli`). 영상 탑만 로봇에서 돈다.
- 마스크: MAP 풀링 주의집중 logit 에 `log(max(w, 0.01))` 를 더한다(w = 8 × 8 칸 마스크 비율). 출력은 물체당 1개(L2 정규화).
- 실행: C++/CUDA + TensorRT. 파이썬은 내보내기·라벨 표·평가만. 글 인코더(SigLIP 2 글 탑)도 TensorRT(`src/textenc.cpp`, 10-05).

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/sgclip.h` | C ABI: 인코더(`sgc_create`·`sgc_submit`·`sgc_poll`), 라벨 표(`sgc_labels_open_ex`·`sgc_labels_names`·`sgc_labels_find`), FP16 도구 |
| `src/crop.cu` | CUDA 커널 하나: 정사각 상자 + 10 % 둘레를 원본 RGB 에서 양선형 256 × 256, 정규화(FP16 NCHW), 8 × 8 마스크 비율(검출기 마스크 비트에서) |
| `src/encoder.cpp` | TensorRT 실행(8.2 / 10 분기 `NV_TENSORRT_MAJOR`), 자기 스트림, 칸 2개 고리(비동기), 배치 크기별 CUDA graph(TRT 10), 결과 자리 풀(프레임마다 메모리 안 잡음) |
| `src/labels.cpp` | 라벨 표 읽기(training/embed 형식), IVF 256 묶음 + 128-d FP16 1단계 + (선택) 128-bit 해밍 + 768-d 다시 매김, AVX2 / NEON / 일반, 상위어 올림, 색인 캐시 |
| `include/sgsearch.h` · `src/objindex.cpp` | **물체 찾기 색인**(10-05): 기억 폴더(view.json + 물체 벡터) × 라벨 표 → 이름 검색 → 생김새 재검색 → 이름 고치기(확인 기록), RecallVLA 질의 벡터 찾기 |
| `src/textenc.cpp` | **글 인코더**(10-05): Gemma BPE 토크나이저(HF 와 토큰 번호 같음) + 토큰 임베딩 모으기(CPU mmap) + TensorRT 글 탑 FP16 |
| `src/labels_impl.hpp` | 라벨 표 안쪽 자료(labels.cpp·objindex.cpp 공용) |
| `src/memstore.*` | 기억 폴더: `objects/O<id>_emb.f16`(원본), `cache/names.json`(이름 캐시, 표 sha·emb_sha 가 바뀐 것만 다시), 노드 메타 JSON |
| `tools/export_siglip2.py` | 영상 탑 → opset 13 정적 ONNX(입력 `images` N×3×S×S + `wpatch` N×G², 출력 `emb` N×768). 변형: `--res` `--layers` `--keep` `--tome` |
| `tools/build_engine.py` | TensorRT 엔진: FP16 + `norm`·`mlp/act`(GELU) 층 FP32 고정, 배치 1–8 동적 또는 1·2·4·8 칸, INT8 PTQ 선택, 층 정보 |
| `tools/eval_variants.py` · `study.sh` · `make_calib.py` | 효율 변형 비교(평가셋 이름·찾기·코사인, GPU 시간) |
| `tools/make_parity.py` | C++ 시험용 기준(평가 crop 64개 원본 RGB·마스크 비트·PyTorch FP32 임베딩) |
| `tools/export_siglip2_text.py` | 글 탑 → ONNX(입력 tok_emb N×64×768, 토큰 임베딩 표는 엔진 밖) + 토크나이저 파일 + 토큰 임베딩 FP16 + C++ 시험 기준 |
| `tools/sgsearch.cpp` | 물체 찾기 명령: `sgsearch MEM [--encode] search Q… / bench N Q… / confirm ID NAME SOURCE / object ID / stats` |
| `tools/eval_objsearch.py` | 물체 찾기 평가(정답 있는 realbag·detcmp 판): 이름만 대 이름 + 생김새 recall@1/5, 없는 물체 헛찾음, 지연 |
| `tools/text_query.py` | 글 → 768-d(SigLIP 2 글 탑, 로봇 밖). 기억 폴더 찾기 JSON, `--serve` HTTP(`/encode`, `/search`) |
| `tools/eval_names.py` | C++ 라벨 찾기 이름 정답률(평가셋, ctypes `libsgclip_c.so`) |
| `tools/sgclip_bench.cpp` · `sgclip_names.cpp` | 단계 시간 표 / 기억 폴더 이름 캐시 오프라인 새로 고침 |

## 만들기

```bash
# 1) ONNX(파이썬, ~/clip_venv) → 엔진(~/ovdet_venv 의 TensorRT 파이썬). 엔진·ONNX 는 git 밖
$CLIP_PY tools/export_siglip2.py --out models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask.onnx
$TRT_PY tools/build_engine.py models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask.onnx \
    models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan --half-input --pin norm,mlp/act
# 1b) 글 탑(물체 찾기 자유 글 질의): ONNX·토크나이저·토큰 임베딩 → 엔진. 층 이름이 영상 탑과 달라 고정 목록을 따로
$CLIP_PY tools/export_siglip2_text.py          # models/ovdet/x86_sm120/siglip2_b32/ 에 씀
$TRT_PY tools/build_engine.py models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_text.onnx \
    models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_text_fp16.plan --pin 'ln_,mlp/gelu,reducel2,^/div$'
# 2) C++ (CUDA 12.8, libpng) — 시험 6개
cmake -S src/scene_graph/clip -B build/sgclip && cmake --build build/sgclip -j && ctest --test-dir build/sgclip
# 3) sgrt 에 같이 들어감(runtime/CMakeLists.txt add_subdirectory) — objprob 이 마스크마다 SigLIP 2 이름·임베딩
```

- 라벨 표: `SGRT_LABELS`(기본 `data/embed_work/labels/objects-v1`, training/embed `export_labels.py` 가 만든다).
- 투영 표본: `SGC_IMG_SAMPLE`(기본 `models/ovdet/x86_sm120/siglip2_b32/img_sample_lvis10k.f16` — LVIS crop 9,753개를 같은 엔진으로 뽑은 FP16 임베딩). 없으면 라벨 글 PCA(1단계 순위가 나쁨).
- 시험 기준 파일: `tools/make_parity.py` → `parity.bin`, `queries_f32.bin`(없으면 그 시험은 건너뜀).

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `sgclip_crop` | 커널 = CPU 기준(같은 식): RGB·RGBA, 행 여백, 영상 밖 상자, 마스크 없음. FP32 ≤ 5e-4, FP16 ≤ 4e-3, 마스크 비율 같음 |
| `sgclip_encoder` | 원본 RGB → 커널 → 엔진 = PyTorch FP32: 평가 crop 64개 코사인 평균 ≥ 0.999·하위 1 % ≥ 0.99, 배치 1·3·8 같은 답, graph 켬/끔, 고리 결과 빠짐없음 |
| `sgclip_lookup` | 돌려준 점수 = 정확한 768-d 내적·점수 순, 전부 열면 전부 훑기와 같은 상위 5, 색인 캐시 다시 읽어도 같음, µs |
| `sgclip_textenc` | 글 인코더 = PyTorch: 질의 25 개(한·일·중·문장 부호·64 넘는 글) 토큰 번호 전부 같음, 코사인 최소 ≥ 0.998·평균 ≥ 0.9995, 배치 1 = 배치 n |
| `sgclip_objindex` | 물체 찾기 논리(GPU 없이, 가짜 기억): 아래말 이름("chair" → straight chair)·한국어·없는 물체 0 개·소화기로 등록된 라디오 → 생김새 후보·확인 뒤 이름으로·다시 열어도 확인 유지·RecallVLA 벡터 찾기 |
| `sgclip_cache` | 처음 전부 / 같으면 0 / emb 하나 바뀌면 1 / 표 sha 바뀌면 전부 / 없어진 물체 지움 / 원자적 쓰기 / emb 파일 왕복 |

## Jetson Nano (TensorRT 8.2, JetPack 4.6) 에서

- 엔진은 Nano 에서 만든다: `build_engine.py`(TRT 8.2 파이썬: `max_workspace_size`, `EXPLICIT_BATCH`, `STRICT_TYPES` 로 물러남) 또는 같은 설정의 `trtexec --fp16 --precisionConstraints=obey --layerPrecisions=...`.
- C++: `encoder.cpp` 의 `NV_TENSORRT_MAJOR < 10` 분기(바인딩 번호, `setBindingDimensions`, `enqueueV2`, CUDA graph 끔). `CMakeLists.txt` 는 aarch64 면 CUDA `/usr/local/cuda`, sm_53. 라벨 찾기는 NEON(`vcvt_f32_f16`, `vfmaq_f32`).
- 아직 Nano 에서 빌드·실행해 보지 않았다(PC 에 TRT 8.2 가 없음). 남은 일은 docs/clip_candidates.md 8.6.

## 글 인코더 (10-05)

- 나눔: 토큰 임베딩 표(256000 × 768, 전체 변수의 70 %)는 **엔진 밖** — `siglip2_b32_tokemb.f16`(375 MB)을 mmap 해 질의의 ≤ 64 줄만 CPU 에서 모은다.
  엔진은 변환기 12 층 + ln_final + 'last' 풀링 + 투영뿐(167 MB, GPU 174 MB) — Nano(4 GB 공유)에서도 들어갈 크기.
- 토크나이저: open_clip `HFTokenizer(timm/ViT-B-32-SigLIP2-256, clean="canonicalize")` 를 C++ 로(BPE merges 580,604 개, 바이트 대체). HF 와 비교:
  시험 질의 25/25, 라벨 표 이름·한국어 이름·프롬프트 4,502 개 중 2 개만 다름(ftfy NFKC 정규화 — 합자 'ﬁ'·전각 숫자, 질의에선 드묾).
- 정밀도: FP16 + LayerNorm·GELU·마지막 L2 FP32 고정 → PyTorch FP32 와 코사인 최소 0.99998·평균 0.999997. 질의 하나 0.79 ms(p50, RTX 5070 Ti, 토큰화·모으기 포함).

## 물체 찾기 (10-05, `include/sgsearch.h`)

공용 물체 색인. 에이전트(LLM 은 API 라 벡터를 못 받음)는 도구 `search_objects`·`confirm_object` 로 **글** 결과를 받고,
RecallVLA 실행기는 같은 색인에 자기 질의 벡터(SigLIP 2 글 공간)를 넣어 상위 K 물체를 칸에 불러온다(`sgs_search_vec`).

**색인 자료**

| 것 | 어디서 | 비고 |
|---|---|---|
| 물체 벡터 | objprob: `objects/O<id>_views.f16`(n × 768 FP16, 상위 시점) → `objects/O<id>_emb.f16`(μ). 없으면 **대체**: best view 사진(`O<id>_rgb.png`·`_mask.png`)을 영상 엔진으로 뽑아 `cache/objsearch/O<id>_view.f16`(사진 크기·시각이 바뀐 것만 다시) | objprob 판이 들어오면 저절로 그쪽을 씀 |
| 등록 이름 | view.json `name`, (objprob) `name_post: [[이름, p] …]` 이 있으면 그것을 이름 사후 바탕으로 | objprob 의 view.json 이름 칸 형식은 아직 미정 — 들어오면 맞춤 |
| 확인 기록 | 기억 폴더 `confirmations.jsonl`(한 줄 = 확인 하나: id·이름·synset·source·우도비·질의·그때 사후 전/후·생김새 확률·벡터 출처·표 sha) | **원본**(보정 데이터). 다시 열면 다시 적용 |
| 이름 캐시 | `cache/objsearch/names.json`(물체마다 이름·확률·대안·속성·벡터 출처) | 언제든 다시 셈 |
| 라벨 집합 U | 라벨 표 main synset 3,244 + 이 기억의 등록·확인 이름 | 표 줄 4,785 개 FP16 행렬 하나 |
| 속성 낱말 | 색 13·재질 9 프롬프트를 글 인코더로(캐시 `cache/objsearch/attr_words_<hash>.f16`) | 크기 낱말은 도구가 상자 크기로 |

**식**
- 생김새 분포(물체 안 상대 확률): 시점 v 마다 `s_vc = t · max_{c 의 줄} cos(z_v, 글)`(t = 111.8, SigLIP 2 logit scale), `m_c = 평균_v log softmax_c s_vc`,
  `P_app(c|o) = softmax_c m_c`. 날 코사인(작고 흔들림) 대신 같은 물체 안에서 이름끼리 견준다. U 밖 질의(자유 글·tail 이름)는 질의를 라벨 하나로 더한
  분포를 물체마다 시점 normalizer 로 바로 셈(µs).
- 이름 사후: `P_name(c|o) ∝ 바탕(c) · Λ_reg(c) · Λ_ext(c)`, 바탕 = objprob 사후 또는 P_app, `Λ_reg`(등록 이름) = 3, `Λ_ext` = 확인 우도비 곱(user 50, close_look 10).
  보여 주는 이름은 낱 라벨 1 위와 "말해진 이름(등록·확인)과 그 아래말" 묶음 질량 중 큰 것(등록 "chair" + 생김새 "folding chair" → chair).
- ① 이름: 질의 → 라벨 표 이름(영어·한국어 전부·동의어, main 먼저) → 뜻 전부(한국어 이름은 기계 번역이 섞여 "빗자루" → broom·awning·shredder —
  글 인코더로 질의와 cos 가 1 위에서 0.05 안인 뜻만) → 그 synset 과 아래말(너무 넓은 말 container·device … 는 아래말 안 씀).
  물체의 말해진 이름이 맞으면 `p_name` = 질의 묶음의 사후 질량, `p_name ≥ 0.05` 면 후보(검출기 헛이름 "book" 으로 등록된 변기는 생김새가 지움).
- ② 생김새(이름 무시): `p_name` 최고 < 0.5 면 자동. `p_query = P_app(질의 묶음|o) ≥ 0.1` 이고 그 물체 안 질의 순위(등록 이름 빼고) = 1 이면 후보.
  이름으로 확실한(p_name ≥ 0.6) 물체가 있으면 영상↔영상 cos 도 `σ(25 (cos − 0.85))`(시뮬 정답: 같은 종류 시점 cos 중앙 0.71, 다른 종류 99 % 0.82).
- 합친 점수 `match = 1 − (1 − p_name)(1 − p_query)(1 − p_img)`, `match_type` = 이름이 맞으면 name, 아니면 appearance.
- ③ `sgs_confirm(id, 이름, user|close_look)`: `Λ_ext ×= 우도비` → 다음 찾기는 ① 에서 바로. 이름이 U 밖이면 라벨을 더하고 모든 물체 분포를 다시(≈ 8 ms).
- (10-05) `sgs_confirm_ex(…, extra)`: 같은 확인 + 기록 한 줄에 덧붙일 JSON(에이전트 실시간 기억이 `{"map":"applied"|"not_objprob"|…,"map_label"}` — 지도 scenemap 에도
  `sm_observe_object_name` 으로 넣었는지). `"map":"applied"` 확인은 view.json objprob `name_post.external` 이 true 가 되면(지도가 이미 셈) 다시 열 때 그 우도비를 빼서
  두 번 세지 않는다. `sgs_label_of(이름)` = 라벨 표 영어 이름(지도 라벨과 맞추기).
- (10-05 고침) objprob `name_post` 는 `{"top":[[이름,p]…],"p","entropy","rolled","external"}` 객체인데 색인이 배열 형식만 읽어 objprob 이름 사후를 통째로 무시했다 — 두 형식 다 읽음.

**결과 JSON**(`sgs_search_json`, 도구가 기억 자리 정보를 붙여 LLM 글로): `{query, resolved{kind, label, ko, senses}, step2, best_name, n_name_hits,
hits:[{id, name, name_ko, name_p, registered, alt:[[이름, p]…], attrs, vec, nv, match, match_type, p_name, p_query, q_rank, p_registered, p_img?, like?}], n_objects, us}`

**측정**(`tools/eval_objsearch.py`, BEHAVIOR `house_double_floor_lower` LIMO 탐사 기억 = `data/datasets/sim_detcmp/A_fastsam`(FastSAM-s + SigLIP 2,
등록 이름 어휘 62 개 — 정답과 맞는 이름이 적음), 확률 모드 전이라 대체 벡터(best view 사진 1 장). 질의 = 지도에 있는 정답 종류 18 개를 사람이 부를 말로
("chair" → straight_chair, "refrigerator" → fridge …), 관련 물체 = 정답 짝 + 정답 상자 안에 중심이 든 물체(같은 물건의 조각·중복).
"이름 못 찾는 물체" = 이름만 찾기가 돌려주지 않는 관련 물체(18 질의 모두 있음). 없는 물체 질의 40 개(컵·노트북·자전거 …).
문턱은 gt 판에서 골랐다(같은 장면의 slam 판은 반쯤 따로 본 셈).

| 판(물체 수) | 질의 | 찾기 | R@1 | R@5 | 이름 못 찾는 물체 R@1 / R@5 | 없는 물체: 뭐라도 나옴 / 묻지 않고 행동할 만큼 | 지연 p50 / p95 |
|---|---|---|---|---|---|---|---|
| gt 자세 (283) | 영어 | 이름만 | 0.11 | 0.22 | 0.00 / 0.00 | 0.05 / 0.03 | 98 / 243 µs |
| | | **이름 + 생김새** | **0.39** | **0.56** | **0.28 / 0.33** | 0.15 / 0.03 | 103 / 277 µs |
| | 한국어 | 이름만 | 0.06 | 0.17 | 0.00 / 0.00 | 0.07 / 0.03 | 122 µs / 1.1 ms |
| | | 이름 + 생김새 | 0.28 | 0.44 | 0.22 / 0.28 | 0.15 / 0.03 | 134 µs / 1.1 ms |
| slam 자세 (296) | 영어 | 이름만 | 0.11 | 0.17 | 0.00 / 0.00 | 0.07 / 0.05 | 97 / 225 µs |
| | | **이름 + 생김새** | **0.39** | **0.50** | **0.28 / 0.33** | 0.15 / 0.05 | 107 / 239 µs |
| | 한국어 | 이름 + 생김새 | 0.28 | 0.39 | 0.22 / 0.28 | 0.15 / 0.05 | 138 µs / 1.1 ms |

- "묻지 않고 행동할 만큼" = 1 위가 이름 후보이고 match ≥ 0.5(도구가 `ask_user` 를 안 붙임). 남은 것은 소파 위 "pillow" 로 등록된 쿠션(정답 종류엔 없지만
  실제로 베개 모양)과 slam 판의 "umbrella" 로 등록된 물체. 생김새 후보는 늘 `ask_user` 가 붙으므로 헛찾음은 되묻기가 된다.
- 문턱 훑기(gt, 영어): `app_min` 0.02–0.25 × 순위 ≤ 1–3 → R@5 0.39–0.67 대 헛찾음 0.07–0.38. 기본(0.1, 순위 1) = R@5 0.56·헛찾음 0.15.
- 라디오(실제 사례): 정답 라디오 하나가 지도에서 4 조각 — O234 "speaker"(16 관측), O232 "bag", O294 "appliance", O1190 "kettle". 이름만: 0 개.
  이름 + 생김새: O234·O232 가 1·2 위(p_query 0.15·0.13, 각 물체 안 1 위), 작은 두 조각(4 cm·12 cm)은 생김새도 못 찾음.
  확인(user) 뒤 O234 의 radio 사후 0.13 → 0.88, 다음 "라디오" 는 ① 이름으로 1 위.
- 실제 OpenLORIS office1-1(298 물체, 정답 없음): 사무실에 없을 집 물건 10 개(욕조·변기·침대·기타·곰 인형·자전거·칫솔·빗자루·세탁기·우산) 헛찾음 0 개.
  "chair" → 등록 chair 5 개 중 생김새도 맞는 셋, "cup" → 1 위가 "bottle" 로 등록된 종이컵(생김새, 되묻기), "keyboard" → "cable" 로 등록된 물체(생김새).
- 지연(질의 하나, 물체 ≈ 300, 찾기는 한 스레드): 라벨 표 이름 질의 74–95 µs, 자유 글·뜻 여럿인 한국어 1.0–1.1 ms(글 인코더 0.8 ms).
  열기: 대체 벡터 뽑기 포함 170 ms(물체 ≈ 300, 영상 엔진), 캐시에서 13 ms. 이름 붙인 라벨 줄 4,785 × 물체 분포 셈 8 ms(8 스레드).
- 속성: 색·재질 확률 ≥ 0.6 만 씀 — sim 283 물체 중 216 개에 하나 이상(검정 67 이 가장 많음: 어두운 사진). 정답이 없어 정확도는 재지 않았다(빨간 라디오 → red,
  파란 천 소파 → blue·fabric 은 사진으로 확인).

**한계**: 대체 벡터는 시점 1 장(best view) — objprob 상위 5 시점이 오면 나아질 것(재지 않음). 등록 이름 우도비 3·확인 우도비 50/10·문턱은 손으로 정한 값(확인 기록이
쌓이면 맞출 것). 한국어는 라벨 표 한국어 이름이 기계 번역이 섞여 영어보다 낮다 — 에이전트가 영어 낱말로 바꿔 부르면 낫다(도구 설명에 적음).
영상↔영상 경로는 이번 평가 질의에선 거의 안 켜짐(이름으로 확실한 본보기가 드묾).

- `tools/gpu_time.sh PLAN BATCH`: GPU 를 다른 일과 나눠 쓸 때 nsys 로 커널 실행 시간만 더해 추론 한 번 시간을 잰다.
