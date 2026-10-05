# 코드 지도 — 무엇이 어디서 어떻게 도는가

코드를 직접 읽으며 정리한 구조. 새 작업을 시작할 때 "어디를 고치면 무엇이 같이 바뀌나"를 찾는 용도.
배치(폴더·경로 변수)는 [LAYOUT.md](LAYOUT.md), 결정·까닭은 [model_selection.md](model_selection.md), 할 일은 [plan.md](plan.md).
파일별 읽기 점검 기록은 [file_review.md](file_review.md).

## 1. 큰 흐름

```
             ┌──────────── 실제 로봇 / OmniGibson 시뮬 / 실제 bag 재생 ────────────┐
 입력        │ 몸통 RGB-D(DaBai 0.3–3 m) · 2D 라이다(X2L) · 바퀴 오도메트리 · 팔 관절 │
             └───────────────────────────────┬────────────────────────────────────┘
                                             ▼
 인지(한 원본)   src/scene_graph/runtime  libsgrt.so  (sgrt_step·sgrt_push_scan)
                 ├ ovdet      ObjectSAM 분할(TensorRT) ─ 마스크
                 ├ clip       SigLIP 2 마스크 임베딩 ─ 이름·벡터   (objprob 앞단 objprob_front.hpp)
                 ├ slam_carto Cartographer ─ map←base 자세
                 └ scenemap   mapper2d 격자 · objmap(objprob) · 방 · 장면 그래프 · 저장 · 스트림
                                             ▼
 출력            memory/ (scene.json·view.json·map.pgm·objects/) · sgview 스트림(TCP)
                                             ▼
 쓰는 쪽         에이전트 툴(search_objects·move_robot) · VLA 지도 토큰 · 뷰어(sgview, 학습 뷰어)
```

학습은 같은 규칙을 GPU 로 흉내 낸다: `training/RL/map`(GPU 학습 지도)이 `scenemap/include/scenemap/objprob_math.h` 를 같이 include 하므로
**물체 판정 규칙을 바꾸면 실제·학습이 함께 바뀐다**. 깊이 한계·몸 크기 같은 값은 scenemap 이 원본이고 GPU 지도가 따른다.

## 2. 인지 — `src/scene_graph`

| 폴더 | 하는 일 | 진입점 | 주의 |
|---|---|---|---|
| `runtime/` | 전부를 묶는 C ABI(libsgrt). keyframe 마다 ObjectSAM → SigLIP 2(같은 스텝 안에서 기다림) → scenemap. 라이다 스캔·오도메트리 → Cartographer → `sm_push_ext_pose`. 저장 스레드·스트림 요약 스레드·입력 기록(`SGRT_RECORD`) | `include/sgrt.h`, `src/sgrt.cpp`, 파이썬 `glue/sgrt_glue.py`(OmniGibson 평가기 안) | 영상 stamp 는 직전 스텝(`SGRT_IMAGE_LAG`). 자세 원천 `SGRT_POSE=carto|odom|gt` |
| `runtime/src/objprob_front.hpp` | 낱말 표(kVocab·kVocabAp → 라벨), 라벨 크기·상위어(kApLabels), 엔진별 매개변수 파일 읽기 | realbag_run 과 libsgrt 가 같이 씀 | 라벨 표(`models/labels/objects-v1`)에 없는 낱말은 빠짐 |
| `scenemap/` | 지도·물체 기억 본체(CPU). 상태는 뮤텍스 하나, 스냅숏은 참조 카운트 사본 | `include/scenemap.h`, `src/capi.cpp` | 로봇은 LIMO + OMX-F 하나 — 기본값(`ScanParams`·`ObjParams`)이 곧 리모 값 |
| `scenemap/src/objmap.cpp` | 물체 지도 `update`: 관측 만들기(깊이·MAD·팔 캡슐·바닥 조각) → 구조물 거르기 → objprob 짝(P(같음)) → 칼만/상자 → 확정·사라짐·옮겨짐·들기 → `apMergePass`(다음 keyframe 앞 물체끼리 병합) | `ObjectMap::update` | 물체 지도의 유일한 규칙 = objprob. 검출마다 임베딩이 없으면 물체가 안 생김 |
| `scenemap/include/scenemap/objprob_math.h` | 같은 것 로지스틱·κ·이름 사후·칼만 수식(`__host__ __device__`) | scenemap + GPU 지도 | 여기를 바꾸면 GPU 학습 지도도 바뀜 — `training/RL/map/tools/objprob_parity` 로 확인 |
| `scenemap/src/mapper2d.cpp` · `scan.cpp` · `grid.cpp` | 외부 자세로 깊이 가상 스캔을 격자에 넣기(정책 1 = 사건 기반) | `Mapper2D::keyframe` | 자세는 밖(Cartographer). 스캔 높이 띠 0.05–0.50 m |
| `scenemap/src/rooms.cpp` · `sgraph.cpp` · `walls.cpp` | 방 나누기, 장면 그래프(물체·방·place·agent), 벽 선분 | 스냅숏 때 | 물체끼리 관계 변은 없음(결정) |
| `scenemap/src/stream.cpp` · `dsg_save.cpp` | sgview 스트림(링 버퍼 + 송신 스레드), 파일 저장(임시 → rename) | | 뷰어·학습 뷰어가 같은 스트림 형식(SGS) |
| `slam_carto/` | Cartographer(ROS 없음) C ABI `sc_*` | `include/slam_carto.h` | 설정 lua 는 `config/`. 떠밀림 측정 `calib/carto_drift.json` → GPU 지도 `drift_params.h` |
| `ovdet/` | 분할 엔진 TensorRT 실행(CUDA 그래프 한 번, 동기 한 번) | `include/ovdet.h` | ObjectSAM 은 이름 'object' 하나, 프롬프트 없음 |
| `clip/` | SigLIP 2 마스크 임베딩(TensorRT), 라벨 표, 물체 찾기 색인(`sgsearch`) | `include/sgclip.h` | |
| `da/` | 물체 둘 합치기·상자 겹침(objprob 병합이 씀) | `include/da/merge.hpp` | |
| `tools/realbag/` | 실제 bag(OpenLORIS) 재생: 같은 파이프라인 + 채점(`objprob_eval.py`) | `realbag_run` | **회귀 확인 기준**: 검출 캐시(`--dump/--load`) + `--pose odom` 이면 결과가 결정적 → 바이트 비교 |
| `sgview/` | 장면 그래프 뷰어(Rust 서버 + three.js) | `tools/run_sgview.sh` | |

## 3. 시뮬·로봇·에이전트

| 폴더 | 하는 일 | 진입점 |
|---|---|---|
| `src/sim/explore` | OmniGibson 에서 탐사 한 판: 평가기(`run_explore.py` + 글루) + 에이전트 런타임(`run-skill`) | `run_explore.sh`, 실시간은 `tools/run_explore_live.sh` |
| `src/sim/lidar` | 시뮬 X2L 라이다(광선 쏘기 → `sgrt_push_scan`) | 글루가 부름 |
| `src/sim/move_robot` | `move_robot` 툴 행동 8 → 시뮬 행동 9 칸 | 글루 |
| `src/robot` | LIMO + OMX-F URDF(xacro), OmniGibson 자산·평가기 연결(`og/eval_with_limo.py`) | `tools/build_urdf.sh` |
| `src/agent` | LLM 에이전트: 런타임(`runtime`), 툴(`tools/move_robot`·`search_objects`), 스킬 = 프롬프트(`skills`, `prompts`). `planner` 는 공부용 | `build/bin/run-skill` |

## 4. 학습 — `training/`

| 폴더 | 하는 일 |
|---|---|
| `RL/env` · `RL/map` · `RL/ppo` | GPU 환경(BEHAVIOR 장면 묶음), GPU 학습 지도(objprob 흉내 + 인지 흉내 `percept.h`), PPO 교사 |
| `RL/map_calib` | 인지 흉내 보정(`percept/`: 실제 og_cmp 기준값 → `percept_params.h`), Cartographer 떠밀림 → `drift_params.h` |
| `RL/map_cmp` | GPU 지도 ↔ 진짜 scenemap 같은 입력 비교 |
| `BC` · `vla` | 학생(BC·DAgger), RecallVLA 학습기 |
| `viewer` | 학습 뷰어(trainview): 실제 sgview 를 iframe 으로 + 재생 조종. 탐사 판은 `og2sg`(libsgrt 로 다시 돌림), GPU 판은 `record_replay`(GPU 지도 저장소를 그대로 그림) |
| `fastsam` · `embed` | ObjectSAM 학습, 이름 임베딩·라벨 표 |

## 5. 바꿀 때 같이 바뀌는 것(연결)

| 바꾸는 것 | 같이 봐야 할 곳 |
|---|---|
| objprob 수식(`objprob_math.h`) | GPU 지도(`training/RL/map`), `objprob_parity` 시험, realbag 바이트 비교 |
| 깊이·몸 크기(`ScanParams`·`ObjParams` 기본값) | GPU 지도의 같은 상수(`map.h`), 시험 장면(3 m 안) |
| 기록 형식 `SGRC`(sgrt.cpp `recImage`·'L' 등) | 읽는 곳 셋: `scenemap/tools/sgrec.hpp`, `training/viewer/tools/og2sg/og2sg.cpp`, 파이썬 `training/RL/map_calib` 쪽 읽기 |
| 스트림 형식(stream.hpp) | sgview, 학습 뷰어 재생(`sgs_play`), `og2sg`, `record_replay/gpu_sg.h` |
| 경로 | `config/paths.env` 하나(코드는 변수만 읽음) |

## 6. 알게 된 점·남은 문제

- 실제 bag 회귀 확인은 `realbag_run … --load <dets.gz> --pose odom` 로 결정적(Cartographer 는 스레드 때문에 매번 조금 다름).
- CLIP 조사 평가셋(`clip_bench`)이 저장소 어디에도 없다 → 이 자료를 읽는 도구(`clip/tools/eval_names.py`·`eval_variants.py`·`study.sh`·`make_parity.py`·`make_calib.py`·`export_siglip2_text.py`, `training/embed` 평가)는 지금 못 돈다. 엔진은 이미 있음. 다시 만들지·지울지 결정 필요.
- objprob 동작 확인 필요 항목은 [known_bugs.md](known_bugs.md) O1–O4.
