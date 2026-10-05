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

아직 옮기지 않은 것(쓰는 곳이 남음 — 바꿀 것이 생기면 옮김): 상태 있는 대본 교사(`training/RL/env/include/teacher.h` — 잡기 가능 표·서는 자리 찾기·`pnp_teach` 기본 교사가 같은 함수를 씀), B6 제자리 돌기 탐사(`teacher_sl.h SLD_EXPLORE` — 프런티어 탐사 교사가 생기면 바꿈), student-lite·flow 머리(설정 `mlp_w`·`head` 로 아직 고름), 따로 된 RL 관측 코드(관측 하나로 합치면).
