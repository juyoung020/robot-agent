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

## 7. 파일별 상세 (읽은 것부터 채움 — 하는 일 · 주요 함수 · 부르는 곳 · 최적화 여지 · 안 쓰는 것)

### `src/scene_graph/runtime`

**`include/sgrt.h` · `src/sgrt.cpp`** (829 줄) — libsgrt C ABI.
- `sgrt_create`: ovdet(분할 엔진) 만들기 → `sm_create`(SGRT_SM_CONFIG > SGRT_ROBOT) → `objprobInit`(SigLIP 2 엔진·라벨 표·엔진별 매개변수 파일) → 자세 원천(`SGRT_POSE`, carto 면 `sc_create`) → 저장 스레드·스트림 요약 스레드·기록 파일.
- `sgrt_begin`: `sm_reset` + Cartographer 새 궤적 + `objprobBegin`(낱말 표 + 과제 이름 → 라벨 → `sm_set_text_model`·`sm_set_label_stats`·`sm_set_obj_params`).
- `sgrt_step`(매 스텝): 기록 'P' → 바퀴 오도메트리를 Cartographer 에 넣고 그 시각 자세를 `sm_push_ext_pose` → `sm_push_proprio`. 영상이 오면: `ovd_detect` → `objprobName`(마스크마다 SigLIP 2, 낱말 표 최댓값 = 이름, `sm_set_det_embeddings`) → 기록 'I' → `sm_push_image_rgb`(best view 자르기·구름 색은 장치에서 `crop.cu`) → `objprobReenc`(통째 다시 담기). 깊이만 오면 격자만. `save_s` 마다 저장 스레드에 맡김.
- `sgrt_push_scan`: 기록 'L' → `sc_push_scan`.
- `sgrt_map`: 탐색·안전 정지용 격자·스캔·옮길 수 있는 물체 사본.
- 부르는 곳: `glue/sgrt_glue.py`(OmniGibson), `training/viewer/tools/og2sg`, `tools/sgrt_replay`·`sgrt_frames`, 에이전트 `search_objects`(sgrt_scenemap).
- 최적화 여지: `objprobName` 이 같은 스텝 안에서 SigLIP 2 결과를 기다림(검출 keyframe 당 약 4–5 ms) — 다음 영상 검출과 겹치면 지연을 줄일 수 있음(미측정). 낱말 최댓값은 CPU 768 × 낱말 수 내적 — 검출 20 × 낱말 ~150 이면 작음.
- 안 쓰는 것: 없음(C ABI 함수마다 부르는 곳 확인함).

**`src/objprob_front.hpp`** — 낱말 표 `kVocab`(글 → 지도 이름)·`kVocabAp`(구조물 배경 낱말), 라벨 크기·상위어 `kApLabels`, `engineStem`·`readParamsFile`, `buildTextTable`(라벨 표에서 글 임베딩 찾기), `enable`. realbag_run 과 같이 씀 — 이름 규칙을 바꾸면 둘이 같이 바뀜.

**`src/crop.cu`·`crop.hpp`** — best view 상자 자르기(넓이 평균, 요청 32 개씩 커널 한 번)·구름 점 색 모으기. 호스트 판(`scenemap/src/bestview.cpp`)과 같은 식, `tests/test_crop.cpp` 가 비교.

**`glue/sgrt_glue.py`** — OmniGibson 평가기 안 `SceneMemory`: 시뮬 proprio(24) → LIMO 12(바퀴 속도를 30 Hz 로 적분해 오도메트리), 몸통 카메라 RGB 는 GPU 텐서 그대로·깊이는 keyframe 만 고정 메모리로, 시뮬 라이다(`src/sim/lidar`)를 `sgrt_push_scan`, 정답 자세(`sgrt_push_pose`, 진단용)·정답 기록. 900 스텝마다 진단 출력.

**`tools/sgrt_replay.cpp`** — 기록을 libsgrt(dlopen)로 다시 굴림, 한 줄씩 읽음(`sgrec::Reader`). 두 빌드 바이트 비교용.
**`tools/sgrt_frames.cpp`** — raw RGB 프레임 + 평평한 2 m 깊이로 끝까지(자르기·저장 경로 확인용).

### `src/scene_graph/scenemap`

**`src/capi.cpp`** (2050 줄) — C ABI 전부. 상태 `sm_ctx`(뮤텍스 하나): `Mapper2D mapper`, `ObjectMap om`, 라벨·종류 표, proprio 대기열(`pending`, 영상 stamp 까지 적분), best view(`views`), 저장 캐시, 방(`RoomTracker`), 자세 큐(`gtq` 정답·`extq` Cartographer), 격자 i8 사본(`grid8`)·벽(`WallExtractor`), 스트림, 장면 그래프, objprob 입력(검출 임베딩·다시 담기 요청).
- 이름 종류: `kStructureNames`(구조물 — 노드 안 됨)·`kStaticNames`(고정 가구)·`kStructObjNames`(문·창·계단 — structural 노드)·`kFloorLevelNames`(러그 등). 비교는 머리 명사(`headMatch`).
- `sm_push_proprio`: 오도메트리 자세 차 → 속도(`Prop.v`), 뷰어 관절 스트림, 0.5 s 넘게 밀린 것은 적분.
- `sm_push_image_rgb`(핵심): 영상 stamp 까지 적분 → 자세(EXT = Cartographer 자세, 없으면 적분) → 순기구학 몸·카메라 → `mapper.keyframe`(격자) → 검출 있으면 ObjFrame(map ← 카메라, 팔 끝·그리퍼, 팔 캡슐, 임베딩, 벽 선분 + 창 틈 잇기) → `om.update` → 검출↔물체·best view 후보 → 그래프 → (잠금 밖) 구름 색·자르기 → 구름 넣기 → 통째 다시 담기 요청.
- `sm_snapshot`: 자세·격자 사본·벽·스캔·내보낼 물체(이름·movable·구름·inspect·best view) → 방 나누기(잠금 밖) → 그래프 사본.
- `refreshWalls`: 벽 방향 θ 를 2 s 마다, 기울면 돌린 격자에서 전부(0.5 s 에 한 번까지), 축에 맞으면 바뀐 행만.
- `sm_snap_reachable`: 격자 8방향 A*(점유 ≥ 65 % 를 0.30 m 부풀림).
- 저장(`sm_save_dsg_ex`): 스냅숏 → objprob 메타·벡터 파일(`apSaveMeta`·`apWriteFiles`) → `saveScene`(dsg_save.cpp). 스트림 요약(`sm_stream_view`)은 바뀐 게 없으면 안 만듦.
- 최적화 여지: 창 틈 잇기가 벽 선분 쌍 O(n²)(벽 수십 개라 작음). `sm_snap_reachable` 이 부를 때마다 W×H dist 배열을 새로 잡음 — 자주 부르면 재사용 가능. 스냅숏마다 물체 구름(shared)·이름 문자열 복사.
- 안 쓰는 것: `robotHands`·`robotBody` 의 robot 인자(로봇이 하나라 늘 같음). C ABI 함수는 모두 부르는 곳 있음(scenemap.h 선언마다 저장소 전체 검색).

**`src/objmap.cpp`** (1440 줄) — 물체 지도 본체(`ObjectMap::update`, keyframe 마다 한 번).
- 0. `updateHands`: 그리퍼가 닫혀 `grip_settle_s` 동안 멈추면 한 번, 잡는 점 `grasp_r` 안 가장 가까운 들 수 있는 확정 물체를 held(`holdable`: 큰 것·고정 종류 아님, 가운데 변 ≤ `grasp_max_w`, 손끝 틈(`gripGap` 표)이 폭과 맞음). 끝까지 닫히거나 열리면 `release`(놓은 자리 아래 받침 물체에 `parent` 로 붙임). 든 것·붙은 것은 손·받침을 따라 평행 이동(구름 포함).
- 1. `apMergePass`: 지난 keyframe 들 구름으로 물체 쌍 P(같음)(`apPairObj` — 접촉·틈·중심 거리·μ cos·겹침·받침·이름 분포 겹침 → 로지스틱) ≥ `merge_p` 인 쌍을 큰 P 부터 1:1 합침(`da::absorbObject` + `apMerge`).
- 2. 검출 → 관측(`Obs`): 상자 안 깊이 화소를 성기게(`max_pts` 기준 간격) 훑어 마스크(1 칸 깎기) 안 점만 map 으로, 팔 캡슐 안 점 버림, 깊이 중앙값 ± MAD 밖 버림, 손 가까운 점이 많으면(든 것) 버림, 10·50·90 백분위 상자. 바닥 조각·벽 너머(창 밖) 버림. `apObsStructural`: 조각 하나의 구조물 확률(`ps`)·구조 물체 확률(`pso`)·가장 그럴듯한 라벨, RANSAC 평면 → 천장·바닥·높고 넓은 벽·벽 선분 위 평면이면 버림(문·창 크기 안이면 남김). 천장 높이도 여기서 추정(`ceil_est_`).
- 3. 짝: 관측마다 P(같음) 가 가장 큰 물체(≥ `same_p`), 구조물 막기·이름 충돌(`name_veto`)·두 물체에 걸친 마스크(`bridge_drop`) 처리. 한 물체에 붙은 조각들은 가장 큰 조각에 합쳐 한 번 갱신.
- 4. 갱신: 작은 것은 축마다 칼만(`objprob_math.h kalman_gain`), 큰 것은 상자 합집합(면마다 `grow_max` 까지) + 분산만 칼만. 움직이는 중(사람이 옮김)이면 관측 자리로 바로(`snap`). 새 물체: id·ap 상태, `appeared`(전에 본 자리에 새로 나타남) 표시, 같은 영상 뒤 조각을 상자 맞닿음으로 같이 묶음(`frame_group`).
- 5. 이름: 이번에 본 물체마다 `apRename`(objprob 사후 최댓값 → `cls`). 구름 후보를 `points_` 로 내보냄(색은 capi 가 붙여 `addPoints`).
- 6. 부재: 확정·안 맞은 물체를 투영해 보일 만큼 보이는데 검출이 없으면 놓침(`absentEvidence` — 가림·너머 보임 구분), 카메라가 옮기거나 돌아야 새 근거로 셈, 검출률로 필요한 놓침 수, 독립 근거면 GONE. `relink`: 사라진 것 ↔ 새로 나타난 것(μ cos, 거리·시간)을 이어 옮겨짐.
- 7. 지우기: 구조물 덩어리(`apStructObject` — 구름 평면·천장 띠·주방향 폭·벽 선 근접, 문·창 크기 확인, 크면 숨김), 확정 안 된 오래된 후보, 몇 번 안 보이고 사라진 헛검출. 윗면 살펴본 정도(`inspTop`), 본 곳 기록(`markView`).
- `buildReencode`: 이번에 본 확정 물체의 구름을 영상에 투영해 마스크를 만들고, 모습 품질 κ 가 지금보다 `reenc_gain` 배 좋거나 합쳐진 물체면 통째 다시 담기 요청(상위 `reenc_max` 개).
- 진단 환경 변수: `SM_OBJ_PARAMS`(매개변수), `SM_OBJ_LOG`·`SM_AP_LOG`·`SM_MOVE_LOG`·`SM_ABS_LOG`·`SM_LINK_LOG`(stderr 로그).
- 최적화 여지: `apMergePass`·관측 짝이 물체 수 n 에 대해 O(n²)·O(관측×n) — 상자 틈 `gate` 로 바로 거르지만 물체가 수백 개면 격자 색인이 나음(지금 실제 bag 118 개라 측정 먼저). `buildReencode` 가 물체마다 마스크 크기 `grid` 를 새로 채움(마스크 160×160 이면 작음). `relink`·`updateHands` 의 id 찾기는 선형.
- 안 쓰는 것: 없음(모든 물체가 ap 상태를 가짐 — 확인함).

**`include/scenemap.h`** — 위 C ABI 선언·설명. 로봇은 LIMO + OMX-F 하나(`SM_ROBOT_LIMO_OMX`, proprio 12).

**`include/scenemap/objmap.hpp`** — `ObjParams`(기본값 = 리모: 깊이 0.15–3 m, 손 하나, grasp_r 0.12, 그리퍼 닫힘 0.6 rad, 잡기 확인 켬, 틈 표), `MapObject`, `ObjFrame`, `ObjectMap`. 규칙 요약이 머리말.
**`include/scenemap/scan.hpp`** — `ScanParams`(기본값 = 리모: 0.3–3 m, 높이 띠 0.05–0.50 m, self_r 0.22) · 붙은 것 거르기 · `makeScan`.
**`include/scenemap/mapper2d.hpp`** — 외부 자세로 격자 넣기, 넣기 정책 1(사건 기반).
**`include/scenemap/fk.hpp`** — 리모 순기구학(`computeLimoFk`: 깊이·손목 카메라, 잡는 점, 팔 끝, 팔 뼈대).

## 6. 알게 된 점·남은 문제

- 실제 bag 회귀 확인은 `realbag_run … --load <dets.gz> --pose odom` 로 결정적(Cartographer 는 스레드 때문에 매번 조금 다름).
- CLIP 조사 평가셋(`clip_bench`)이 저장소 어디에도 없다 → 이 자료를 읽는 도구(`clip/tools/eval_names.py`·`eval_variants.py`·`study.sh`·`make_parity.py`·`make_calib.py`·`export_siglip2_text.py`, `training/embed` 평가)는 지금 못 돈다. 엔진은 이미 있음. 다시 만들지·지울지 결정 필요.
- objprob 동작 확인 필요 항목은 [known_bugs.md](known_bugs.md) O1–O4.
