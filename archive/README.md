# archive — 더 안 쓰는 코드·설정 (원래 자리 그대로)

빌드·시험에서 뺀 것. 지우지 않고 `git mv` 로 옮겼다(원래 경로를 `archive/` 아래에 그대로). 다시 쓰려면 같은 경로로 되돌린다.

| 언제 | 무엇(원래 경로) | 까닭 |
|---|---|---|
| 2026-10-06 | `training/BC/config/bc_a2*.json`(7 개), `bc_pnp_b4_dagger.json`, `bc_pnp_lite.json` | 처음 지도 C0/C1(미리 채운 참 지도)로 학습하는 설정. 10-06 사용자 결정 — 학습·평가는 빈 지도 C2 에서 자라는 지도만(CURRICULUM_BEHAVIOR2026 5.7). `bc_run` 은 이제 `--debug-gt-map` 없이는 이 설정을 멈춤. 잰 값은 BC README·CURRICULUM 5.6.2 에 남음 |
| 2026-10-06 | `training/RL/config/ppo_a2.json`, `ppo_a2_notok.json`, `ppo_a2_nofront.json`, `ppo_b.json` | 같은 까닭(단계 `map` [p0, p1] > 0). `ppo_run` 도 `--debug-gt-map` 없이는 멈춤 |
| 2026-10-06 | `training/demos/`(전체), `docs/map_vla/HUMAN_DEMOS.md` | 다른 로봇(BEHAVIOR 2026 R1 Pro, DROID)의 사람 시연을 scenemap 으로 다시 돌려 RecallVLA 자료로 쓰는 시험(10-04, 한 판). 10-05 보류 결정 — 학습은 우리 로봇(LIMO + OMX-F, `map_vla.urdf`)의 GPU 시뮬 교사만. 학습 뷰어의 시험 판(`~/trainview_work/behavior_og/human_demos`)은 `~/trainview_work/archive/human_demos` 로 옮김 |
| 2026-10-06 | `training/RL/map/include/map.h` 의 옛 objmap 이름 규칙 조각 → `archive/training/RL/map/include/map_objmap_rules.h`(코드 조각, 빌드 안 함) | GPU 학습 지도를 objprob 구조로 다시 옮김(docs/map_vla/GPU_MAP_PORT.md P1): 이름 표 투표·같은 이름 탐욕 짝·관측 수 평균·이름 같음/IoU 병합 대신 인지 흉내 층(`percept.h`) + objprob 합치기(`objprob_gpu.h`) |
| 2026-10-06 | `tools/sync_scene_graph.sh` | scene_graph 를 behavior-2026 서브모듈에서 복사하던 도구. 10-06 결정으로 robot-agent `src/scene_graph` 가 유일한 원본(behavior-2026 에서 옮겨 옴) — 복사할 곳이 없음 |
| 2026-10-06 | `src/scene_graph/scenemap/eval/`(demo_data·export_episode·export_gtdet·gt_scene·gt_traj·score_objmap·score_slam), `src/scene_graph/ovdet/tools/make_task_prompts.py` | BEHAVIOR 2026 R1 Pro 사람 시연(`data/2026-challenge-demos`, 4.9 GB)으로 scenemap 을 채점하던 도구와 옛 YOLOE 과제 프롬프트 만들기. 시연 데이터는 behavior-2026 분리 때 지움. 지금 평가는 OpenLORIS 실제 기록(`realbag_run`·`objprob_eval`)과 리모 시뮬 기록. `score_map_gt.py` 는 남김 |
| 2026-10-06 | `training/RL/map_calib/limo/tools/det_score.py` | COCO-80 검출기(옛 서브모듈 ovdet `coco80.txt`)로 LIMO 기록 검출을 채점하던 도구. 검출기는 ObjectSAM `yolo26n-seg-obj-416`, 비교는 `map_cmp`·`og_cmp` |
| 2026-10-06 | `src/scene_graph/scenemap/src/slam2d.cpp`·`include/scenemap/slam2d.hpp`(깊이 가상 스캔 맞추기 `'A'`·`'B'`), `tools/slam2d_eval.cpp`·`objmap_eval.cpp`·`stage_bench.cpp`·`capi_replay.cpp`(slam2d 내부 API·R1 시연 `ep_*.bin` 을 쓰던 도구) | 사용자 결정: SLAM 은 Cartographer 하나(`src/scene_graph/slam_carto`, libsgrt·realbag_run 기본). slam2d 는 시뮬을 먼저 돌리려고 들어간 임시였다. 자세와 상관없는 부분(적분·깊이 가상 스캔·넣기 정책·격자)은 `scenemap/src/mapper2d.cpp` 로 남겼다. OpenLORIS 7 판 Cartographer 대 slam2d 표는 `slam_carto/README.md` |
| 2026-10-06 | `src/scene_graph/viewer/`(sgviz, 파이썬 Spark-DSG + viser 뷰어) | 파일을 주기적으로 읽는 옛 뷰어. 실시간·기록 재생·학습 리플레이는 모두 sgview(Rust)가 맡는다 |
| 2026-10-06 | `src/scene_graph/scenemap/include/scenemap/r1pro_fk_table.hpp`, `tools/gen_fk_table.py`, `tests/test_fk.cpp`, `src/fk_r1pro_computeBodyFk.cpp.txt`(fk.cpp 의 `computeBodyFk` 조각), `tests/test_objmem_held_r1.cpp.txt`(R1 팔 끝을 proprio 로 정하던 들기 시험 조각) | 사용자 결정: 로봇은 LIMO + OMX-F 하나(R1 Pro 뺌). scenemap 순기구학·proprio·`"robot"` 설정은 limo_omx 만. 들기 구름 따라감은 `test_limo_e2e` 가 봄 |
| 2026-10-06 | `src/scene_graph/scenemap/tools/map_timeline.cpp`·`dom_bench.cpp`·`dom_seq.hpp`(벤치마크 어댑터 실행기 부분), `src/scene_graph/runtime/tools/dom_bench_det.cpp` | R1 61 proprio·R1 몸 매개변수(스캔 높이 띠 0.10–1.80 m)로 dynamic-object-mapping-benchmark 사람 손 카메라 시퀀스를 넣던 도구. 로봇이 LIMO 하나라 쓸 곳 없음. PNG·CSV 읽기 부분은 `scenemap/tools/png_io.hpp`(realbag_run 이 씀)로 남김 |

아직 옮기지 않은 것(쓰는 곳이 남음 — 바꿀 것이 생기면 옮김): 상태 있는 대본 교사(`training/RL/env/include/teacher.h` — 잡기 가능 표·서는 자리 찾기·`pnp_teach` 기본 교사가 같은 함수를 씀), B6 제자리 돌기 탐사(`teacher_sl.h SLD_EXPLORE` — 프런티어 탐사 교사가 생기면 바꿈), student-lite·flow 머리(설정 `mlp_w`·`head` 로 아직 고름), 따로 된 RL 관측 코드(관측 하나로 합치면).
| `src/scene_graph/tools/realbag/objprob_params/FastSAM-s-416-obj*.json` | 버린 FastSAM-s 재학습 엔진의 objprob 매개변수(엔진도 보관) | 10-06 |
