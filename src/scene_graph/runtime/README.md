# sgrt — 장면 그래프 런타임(libsgrt.so): 검출 + 물체 기억을 한 C ABI 로

평가기(또는 로봇) 프로세스 안에서 물체 기억을 실시간으로 굴린다. 공유 라이브러리 하나에 세 가지를 묶는다.

- 검출: ovdet(YOLOE 또는 닫힌 어휘 YOLO-seg, TensorRT). 머리 RGB 를 장치 메모리에서 바로 읽는다.
- 지도·물체 기억: scenemap(slam2d·objmap·방·장면 그래프·저장). 깊이는 호스트 f32 미터.
- (선택) 물체 영상 임베딩: sgclip(SigLIP 2, `../clip`). `SGRT_CLIP` 이면 켜짐.

파이썬은 포인터만 넘긴다(`glue/sgrt_glue.py`). 한 스레드에서 부른다. 설계는 [docs/scenemap_설계.md](../../../docs/scenemap_설계.md) 3.2.2·3.5·3.6·3.7절.

```
매 스텝     sgrt_step(proprio)                    → scenemap 자세 적분, 든 물체 따라가기
keyframe    sgrt_step(proprio + 머리 RGB + 깊이)  → ovdet 검출 → scenemap 물체 지도 갱신(+ best view 자르기·점 구름 색)
지도 스텝   sgrt_step(proprio + 깊이만)           → slam2d 격자만 갱신(SGRT_MAP_EVERY, 글루가 정함)
주기 저장   시뮬 save_s 마다                       → 저장 스레드가 out_dir 에 scene.json · view.json · map.pgm …
```

keyframe 인지는 `sgrt_want_image()` 가 알려 준다(`kf_every` 스텝마다, 기본 6). 호출자는 그 스텝에만 영상 포인터를 넘긴다.

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/sgrt.h` | C ABI: `sgrt_create`·`sgrt_begin`·`sgrt_step`·`sgrt_save`·`sgrt_destroy`, 통계·시간(`sgrt_stats`·`sgrt_get_timing`·`sgrt_get_stage_timing`), 지도 보기(`sgrt_map`·`sgrt_map_snapshot`), 자세 원천(`sgrt_set_pose_mode`·`sgrt_push_pose`·`sgrt_get_pose_diag`), 이름 종류(`sgrt_set_kind_names`), 임베딩·이름 찾기(`sgrt_object_embedding`·`sgrt_query_embedding`·`sgrt_query_label`·`sgrt_object_names`·`sgrt_get_clip_stats`) |
| `src/sgrt.cpp` | 구현: ovdet + scenemap 연결, 프롬프트 표(`SGRT_PROMPT`), 영상 시각 늦춤(`SGRT_IMAGE_LAG`), 저장 스레드, 스트림 요약 스레드(`SGRT_STREAM`), 입력 기록(`SGRT_RECORD`), 단계 시간(det·step·map·record) |
| `src/crop.cu` · `crop.hpp` | CUDA 커널 둘: best view 상자 자르기(넓이 평균, 요청 32개씩 한 번에, 자른 것만 고정 메모리로 내려받기)와 점 구름 화소 색 모으기. 식은 `scenemap/src/bestview.cpp` 호스트 판과 같다 |
| `src/sgrt_clip.cpp` · `.hpp` | sgclip 연결: keyframe 마다 새 물체·best view 품질이 임베딩 때의 1.2배 이상인 물체를 최대 8개 비동기로 임베딩. 저장 때 `objects/O<id>_emb.f16`, `cache/names.json`, scene.json 노드 metadata(`sm_set_object_meta`). 라벨 표는 따로 스레드에서 읽음 |
| `glue/sgrt_glue.py` | 평가기 쪽 접착부 `SceneMemory(task, out_dir)` · `step(obs)` · `close()`. 과제 프롬프트(`../ovdet/config/task_prompts.txt` 의 과제 줄 + `_scene` 줄), GT 자세 넣기, GT 기록, 900 스텝마다 진단·시간 출력 |
| `tests/test_crop.cpp` | 장치 자르기·색 모으기 = 호스트 식, 시간 |
| `tools/sgrt_frames.cpp` | 실제 엔진으로 끝까지 확인: raw RGB 프레임을 장치에 올려 `sgrt_step` → 저장. 깊이는 평평한 2 m(가짜) — 검출 → 자르기·색 모으기 → PNG/PLY 경로 확인용 |

## 만들기

CUDA 12.8(`/usr/local/cuda-12.8`, `OVDET_CUDA_ROOT`), sm_120(`CMAKE_CUDA_ARCHITECTURES`). `../scenemap`·`../ovdet`·`../clip` 을 `add_subdirectory` 로 같이 빌드한다.

```bash
cmake -S src/scene_graph/runtime -B ~/sgrt_build && cmake --build ~/sgrt_build -j
ctest --test-dir ~/sgrt_build -R crop             # 장치 자르기 시험
~/sgrt_build/sgrt_frames <engine.plan> <out_dir> <w> <h> <frames.rgb ...>   # <engine.plan>.names.txt 가 옆에 있어야 함
```

글루는 `~/sgrt_build/libsgrt.so` 를 기본으로 읽는다(`SGRT_LIB`).

```python
from sgrt_glue import SceneMemory
mem = SceneMemory(task_name, out_dir)   # 프로세스에 한 번
mem.step(obs)                           # 평가기 스텝마다, 정책 앞에서
mem.close()                             # 이름 보고·진단 출력, 마지막 저장
```

- 머리 RGB 는 GPU 텐서 그대로 넘긴다. 깊이는 keyframe(과 지도 스텝)에만 고정 메모리 버퍼로 내려받는다.
- 관측에 머리 RGB-D 가 없으면 한 번 경고한다(`RGBDFullResWrapper` 를 쓸 것).
- 쓰는 곳: `src/sim/move_robot/run_eval_move.py`, `src/sim/explore/run_explore.py`, `src/vla/pi05_native/glue/run_eval_native.py`.

## 환경 변수

`src/sgrt.cpp`·`src/sgrt_clip.cpp`(라이브러리)와 `glue/sgrt_glue.py`(글루)가 읽는 것 전부.

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `SGRT_POSE` | `slam` | 자세 원천. `slam`(적분 + 스캔 맞추기) · `odom`(적분만) · `gt`(외부 정답 베이스 자세, map = 시뮬 world — 진단·시각화용, 대회 제출 금지). 그 밖의 값은 `slam`. 시뮬은 실제 로봇과 같게 `slam` 으로 재고, `gt` 는 확인용으로만 |
| `SGRT_MAP_POLICY` | `1` | 격자 넣기 정책. 1 = 사건 기반(서 있어도 바뀐 장애물을 넣고 지움), 0 = 옛 움직임 거르기 |
| `SGRT_IMAGE_LAG` | `1` | 영상 stamp 를 몇 스텝 앞 시각으로 둘지(0..7). 1 = 직전 스텝(평가기 영상 k = 장면 k-1). 글루도 GT 자세를 읽을 스텝을 고를 때 같은 값을 쓴다 |
| `SGRT_SAVE_SYNC` | `0` | 0 이 아니면 주기 저장을 스텝 안에서 한다. 0 이면 저장 스레드(앞 저장이 안 끝났으면 그 주기는 건너뜀) |
| `SGRT_STREAM` | 없음(끔) | `host:port`. sgview(`--ingest`)로 실시간 스트림. 자세·지도 변화분은 스텝 안에서 링 버퍼로, 요약은 따로 스레드가 만든다. 5 s 마다 stderr 에 스트림 통계 |
| `SGRT_STREAM_HZ` | `60` | 요약(view) 만드는 주기 Hz, 0.5–240 으로 자름 |
| `SGRT_RECORD` | 없음(끔) | 파일 경로. 받은 입력(proprio·외부 자세·keyframe 깊이·RGB·검출)을 이진(`SGRC` 판 1)으로 기록. `scenemap/tools/sm_bench`·`stage_bench` 가 재생 |
| `SGRT_PROMPT` | `auto` | 프롬프트 표. `task`(과제 이름만) · `all`(엔진 어휘 전부) · `auto`(어휘 200 이하 닫힌 어휘 엔진이면 `all`, 아니면 `task`). 프롬프트가 비면 늘 `all` |
| `SGRT_CLIP` | 없음(끔) | SigLIP 2 엔진 plan 경로. `1` = `~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan`, `0`·빈 값 = 끔 |
| `SGRT_CLIP_GRAPH` | sgclip 기본(1) | 배치 크기별 CUDA graph 쓰기(`sgc_config.use_graph`) |
| `SGRT_LABELS` | `~/embed_work/labels/objects-v1` | 라벨 표 폴더(이름 찾기). 색인 캐시는 `<out_dir>/cache/index` |
| `SGC_IMG_SAMPLE` | `~/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16`(있으면) | 라벨 표 투영을 맞출 영상 임베딩 표본(sgclip 변수, sgrt_clip 이 읽어 넘김) |
| `SGRT_LIB` | `~/sgrt_build/libsgrt.so` | 글루: 읽을 라이브러리 |
| `SGRT_ENGINE` | `~/ovdet_models/x86_sm120/yoloe-11l-all.plan` | 글루: 검출 엔진. 이름 표는 `<엔진>.names.txt` |
| `SGRT_GT_POSE` | `1` | 글루: 시뮬 로봇 베이스 정답 자세를 `sgrt_push_pose` 로 넣기(`0` = 끔). gt 모드면 map 자세, 아니면 떠밀림 진단에만 쓰임 |
| `SGRT_GT_EVERY` | `0` | 글루: `1` 이면 정답 자세를 매 스텝 읽음. 아니면 영상 시각·keyframe·지도 스텝에 필요한 스텝만 |
| `SGRT_MAP_EVERY` | `0` | 글루: n 스텝마다 깊이만으로 격자 갱신(검출·물체 지도·임베딩은 keyframe 그대로). 0 = keyframe 에서만 |
| `SGRT_GT_LOG` | 없음(끔) | 글루: CSV 경로. keyframe 마다 정답 베이스·머리 카메라 자세를 쓰고, 정답 물체(이름·범주·위치·AABB)를 `<csv>.objects.json` 에 씀(`scenemap/eval/score_map_gt.py` 가 읽음) |

`sgrt_config` 기본값(`sgrt_default_config`): `kf_every` 6, `save_s` 1.0(시뮬 초), `conf_th` 0.25. 글루도 같은 값을 넘긴다.

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `crop` | 720 × 720 RGBA 난수 영상에서 상자 6개(작은 것·줄이는 것·가장자리) 장치 자르기 = 호스트 `cropRgbHost`, 화소 3000개(영상 밖 좌표 포함) 색 모으기 = 호스트 `gatherRgbHost`. 장치 자르기·온 영상 내려받기 시간을 찍는다. GPU 가 없으면 실패 |

scenemap 시험은 [../scenemap/README.md](../scenemap/README.md), sgclip 시험은 [../clip/README.md](../clip/README.md).
