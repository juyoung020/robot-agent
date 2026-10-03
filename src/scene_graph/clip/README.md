# sgclip — 물체 영상 임베딩(SigLIP 2 B/32-256) + 라벨 찾기

물체 기억(scenemap)의 물체마다 **영상 벡터 1개(768-d)** 를 만들고, 그 벡터로 이름을 붙이고(라벨 표), 글로 찾는다.
설계·측정은 [docs/clip_candidates.md](../../../../../docs/clip_candidates.md)(상위 저장소) 3.5·6·8절.

- 모델: SigLIP 2 B/32-256(Apache-2.0, open_clip `ViT-B-32-SigLIP2-256` / `webli`). 영상 탑만 로봇에서 돈다.
- 마스크: MAP 풀링 주의집중 logit 에 `log(max(w, 0.01))` 를 더한다(w = 8 × 8 칸 마스크 비율). 출력은 물체당 1개(L2 정규화).
- 실행: C++/CUDA + TensorRT. 파이썬은 내보내기·라벨 표·평가·글 인코더(로봇 밖)만.

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/sgclip.h` | C ABI: 인코더(`sgc_create`·`sgc_submit`·`sgc_poll`), 라벨 표(`sgc_labels_open_ex`·`sgc_labels_names`·`sgc_labels_find`), FP16 도구 |
| `src/crop.cu` | CUDA 커널 하나: 정사각 상자 + 10 % 둘레를 원본 RGB 에서 양선형 256 × 256, 정규화(FP16 NCHW), 8 × 8 마스크 비율(검출기 마스크 비트에서) |
| `src/encoder.cpp` | TensorRT 실행(8.2 / 10 분기 `NV_TENSORRT_MAJOR`), 자기 스트림, 칸 2개 고리(비동기), 배치 크기별 CUDA graph(TRT 10), 결과 자리 풀(프레임마다 메모리 안 잡음) |
| `src/labels.cpp` | 라벨 표 읽기(training/embed 형식), IVF 256 묶음 + 128-d FP16 1단계 + (선택) 128-bit 해밍 + 768-d 다시 매김, AVX2 / NEON / 일반, 상위어 올림, 색인 캐시 |
| `src/memstore.*` | 기억 폴더: `objects/O<id>_emb.f16`(원본), `cache/names.json`(이름 캐시, 표 sha·emb_sha 가 바뀐 것만 다시), 노드 메타 JSON |
| `tools/export_siglip2.py` | 영상 탑 → opset 13 정적 ONNX(입력 `images` N×3×S×S + `wpatch` N×G², 출력 `emb` N×768). 변형: `--res` `--layers` `--keep` `--tome` |
| `tools/build_engine.py` | TensorRT 엔진: FP16 + `norm`·`mlp/act`(GELU) 층 FP32 고정, 배치 1–8 동적 또는 1·2·4·8 칸, INT8 PTQ 선택, 층 정보 |
| `tools/eval_variants.py` · `study.sh` · `make_calib.py` | 효율 변형 비교(평가셋 이름·찾기·코사인, GPU 시간) |
| `tools/make_parity.py` | C++ 시험용 기준(평가 crop 64개 원본 RGB·마스크 비트·PyTorch FP32 임베딩) |
| `tools/text_query.py` | 글 → 768-d(SigLIP 2 글 탑, 로봇 밖). 기억 폴더 찾기 JSON, `--serve` HTTP(`/encode`, `/search`) |
| `tools/eval_names.py` | C++ 라벨 찾기 이름 정답률(평가셋, ctypes `libsgclip_c.so`) |
| `tools/sgclip_bench.cpp` · `sgclip_names.cpp` | 단계 시간 표 / 기억 폴더 이름 캐시 오프라인 새로 고침 |

## 만들기

```bash
# 1) ONNX(파이썬, ~/clip_venv) → 엔진(~/ovdet_venv 의 TensorRT 파이썬). 엔진·ONNX 는 git 밖
~/clip_venv/bin/python tools/export_siglip2.py --out ~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask.onnx
~/ovdet_venv/bin/python tools/build_engine.py ~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask.onnx \
    ~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan --half-input --pin norm,mlp/act
# 2) C++ (CUDA 12.8) — 시험 4개
cmake -S src/scene_graph/clip -B ~/sgclip_build && cmake --build ~/sgclip_build -j && ctest --test-dir ~/sgclip_build
# 3) sgrt 에 같이 들어감(runtime/CMakeLists.txt add_subdirectory). 켜기: SGRT_CLIP=1(기본 엔진 경로) 또는 plan 경로
```

- 라벨 표: `SGRT_LABELS`(기본 `~/embed_work/labels/objects-v1`, training/embed `export_labels.py` 가 만든다).
- 투영 표본: `SGC_IMG_SAMPLE`(기본 `~/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16` — LVIS crop 9,753개를 같은 엔진으로 뽑은 FP16 임베딩). 없으면 라벨 글 PCA(1단계 순위가 나쁨).
- 시험 기준 파일: `tools/make_parity.py` → `parity.bin`, `queries_f32.bin`(없으면 그 시험은 건너뜀).

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `sgclip_crop` | 커널 = CPU 기준(같은 식): RGB·RGBA, 행 여백, 영상 밖 상자, 마스크 없음. FP32 ≤ 5e-4, FP16 ≤ 4e-3, 마스크 비율 같음 |
| `sgclip_encoder` | 원본 RGB → 커널 → 엔진 = PyTorch FP32: 평가 crop 64개 코사인 평균 ≥ 0.999·하위 1 % ≥ 0.99, 배치 1·3·8 같은 답, graph 켬/끔, 고리 결과 빠짐없음 |
| `sgclip_lookup` | 돌려준 점수 = 정확한 768-d 내적·점수 순, 전부 열면 전부 훑기와 같은 상위 5, 색인 캐시 다시 읽어도 같음, µs |
| `sgclip_cache` | 처음 전부 / 같으면 0 / emb 하나 바뀌면 1 / 표 sha 바뀌면 전부 / 없어진 물체 지움 / 원자적 쓰기 / emb 파일 왕복 |

## Jetson Nano (TensorRT 8.2, JetPack 4.6) 에서

- 엔진은 Nano 에서 만든다: `build_engine.py`(TRT 8.2 파이썬: `max_workspace_size`, `EXPLICIT_BATCH`, `STRICT_TYPES` 로 물러남) 또는 같은 설정의 `trtexec --fp16 --precisionConstraints=obey --layerPrecisions=...`.
- C++: `encoder.cpp` 의 `NV_TENSORRT_MAJOR < 10` 분기(바인딩 번호, `setBindingDimensions`, `enqueueV2`, CUDA graph 끔). `CMakeLists.txt` 는 aarch64 면 CUDA `/usr/local/cuda`, sm_53. 라벨 찾기는 NEON(`vcvt_f32_f16`, `vfmaq_f32`).
- 아직 Nano 에서 빌드·실행해 보지 않았다(PC 에 TRT 8.2 가 없음). 남은 일은 docs/clip_candidates.md 8.6.
