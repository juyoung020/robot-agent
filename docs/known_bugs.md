# 알려진 버그 (코드)

문서를 코드와 맞추다가 찾은, **코드 쪽** 문제를 모은다. 고치면 이 표에서 지우거나 "고침" 으로 적는다.
처음 정리 2026-10-03. 2026-10-04 1·2·4–6·8–12 고침(behavior-2026 커밋은 서브모듈 포인터를 아직 안 올림). 경로는 이 저장소 기준(서브모듈은 `src/behavior-2026/...`).

| # | 어디 | 무엇 | 영향 | 고치는 법 |
|---|---|---|---|---|
| 1 | `src/scene_graph/ovdet/tools/ovdet_eval.py:33` (동기화 사본 `src/scene_graph/ovdet/tools/ovdet_eval.py` 도 같음) | `sys.path` 에 옛 경로 `src/scenemap/eval` 을 넣는다. 폴더 개편 뒤 경로는 `src/scene_graph/scenemap/eval` | ovdet 평가 스크립트가 `gt_scene` 을 못 불러와 실패할 것으로 보임 | **고침** behavior-2026 `057689b`(시험: 고치기 전 `ModuleNotFoundError: demo_data`, 뒤 `--help` 됨). 동기화 사본 `src/scene_graph/...` 은 서브모듈 포인터 올린 뒤 `tools/sync_scene_graph.sh` 로. `os.path.join(ROOT, 'src', 'scene_graph', 'scenemap', 'eval')` 로. 서브모듈에서 고친 뒤 `tools/sync_scene_graph.sh` |
| 2 | `src/agent/skills/explore/src/bin/explore.rs:91`(10-06 에 `src/agent/tools/move_robot/src/mock_eval.rs` 로 옮김) | 주석은 몸통 원 0.28 m 인데 코드(같은 파일 115 줄)는 0.37 m | 동작은 0.37 m. 주석만 틀림 | **고침** robot-agent `f139dfe`. 주석을 0.37 m 로 |
| 3 | 서브모듈 스크립트 약 80 개 (아래 목록) | 저장소 경로를 `/mnt/c/behavior-2026` 으로 박아 둠(Windows·WSL 시절 경로) | 작업 PC `jy-desktop` 에서는 심볼릭 링크 `/mnt/c/behavior-2026` → 서브모듈 이 있어서 돈다(서브모듈 `docs/Linux_설치.md` 3.1). 링크가 없는 PC(학교 4090 등)에서는 실패 | **보류**: 84 파일 일괄(작은 고침 아님). 4·5·6 만 고침. 스크립트 위치에서 저장소 뿌리를 찾거나(`$(cd "$(dirname "$0")/../.." && pwd)`) 환경 변수로 받는다. `tools/README.md` 의 리눅스판 스크립트가 이미 이렇게 한다 |
| 4 | `src/behavior-2026/src/sim/integ/build_simlink.sh:4,8` | 사용 예가 `wsl.exe -d Ubuntu-22.04 ...`, 본문이 `cd /mnt/c/behavior-2026/...` | 3 과 같음 | **고침** behavior-2026 `0c392f7`(스크립트 위치 기준 `cd`). 3 과 같이 |
| 5 | `src/behavior-2026/src/agent/planner/src/main.rs:214` | `build-assets --root` 기본값이 `/mnt/c/behavior-2026` | `--root` 를 안 주면 3 과 같음 | **고침** behavior-2026 `0c392f7`(기본 = 크레이트 `../../..`, 시험 `repo_root_is_this_checkout`). 기본값을 실행 파일 위치나 현재 폴더 기준으로 |
| 6 | `src/behavior-2026/tools/check_agent_hook.py:16` | `sys.path` 에 `/mnt/c/behavior-2026/tools` | 3 과 같음 | **고침** behavior-2026 `0c392f7`(`Path(__file__).resolve().parent`). 파일 위치 기준(`Path(__file__).parent`) |
| 7 | `src/scene_graph/scenemap/src/dsg_save.cpp` (`sceneJsonFast`), `tests/test_scene_json.cpp`, `include/scenemap.h` | 그래프는 물체 + 방 2 층(장소는 백엔드에서 계산)으로 정했는데, scene.json 쓰기가 아직 장소(층 3)·agent 노드를 쓰고, 시험이 `numLayers() == 3`·PLACES 층을 확인한다. `scenemap.h` 층 목록도 OBJECTS/AGENTS/PLACES/ROOMS/BUILDINGS | 저장 파일·시험이 2 층 결정과 다름 | **보류**: 사용자 결정(2 층 확정) 필요. scene.json 에서 장소 층을 빼고 시험·헤더를 2 층으로 |
| 8 | `src/scene_graph/scenemap/CMakeLists.txt` 메시지, `include/scenemap.h` 주석 | "Spark-DSG 가 없으면 scene.json 을 안 쓴다" 고 적혀 있지만 빠른 쓰기(`sceneJsonFast`)가 늘 쓴다 | 주석·메시지만 틀림 | **고침** behavior-2026 `057689b`(문구). 문구를 고친다 |
| 9 | `src/scene_graph/runtime/src/sgrt.cpp` | `SGRT_STREAM_HZ` 주석은 기본 5 인데 코드는 60(0.5–240 으로 자름) | 주석만 틀림 | **고침** behavior-2026 `057689b`(주석 60, 0.5–240). 주석을 60 으로 |
| 10 | `src/scene_graph/da/include/da/merge.hpp`·`da/src/merge.cpp` | 주석은 `max_ext` 를 넘는 쌍은 병합 안 한다는데 `merge.cpp` 가 `max_ext` 를 확인하지 않는다. `mergeable()` 이 `kinds` 를 안 본다 | 큰 물체끼리, 다른 종류끼리 병합될 수 있음 | **고침** behavior-2026 `057689b`: 합집합이 될 쌍(큰 가구·고정 종류)은 한 변이 `ObjParams::max_ext` 넘으면 안 합침. `kinds` 는 같은 이름 번호면 같아서 따로 볼 것 없음(주석). 시험 `test_merge` 2 경우 추가(고치기 전 실패). 실제 경로 `da/include/da/merge.hpp`·`da/src/merge.cpp`. `max_ext`·`kinds` 확인을 넣거나 주석을 고친다 |
| 11 | `src/scene_graph/scenemap/src/objmap.cpp`, `dsg_save.cpp` (`eventName`) | 병합을 이벤트 7 로 기록하는데 `eventName` 은 0–6 만 안다 | 기록·뷰어에 "?" 로 보임 | **고침** behavior-2026 `057689b`(`merged`, 시험 `test_scene_json` 이벤트 이름 — 고치기 전 실패). `eventName` 에 7(병합) 추가 |
| 12 | `src/scene_graph/scenemap/CMakeLists.txt` (`walls` 시험) | `${WALLS_REF_CELLS}`·`${WALLS_REF_JSON}` 를 넘기지만 어디서도 정하지 않는다 | ctest 가 파이썬 기준 비교 없이 돈다 | **고침** behavior-2026 `057689b`: `WALLS_REF_CELLS`·`WALLS_REF_JSON` 캐시 변수(기본 빈 값), 둘 다 있을 때만 인자로. 변수를 정하거나 인자를 뺀다 |
| 13 | `src/agent/tools/move_robot/src/lib.rs` (`Motion::Base`) | **버그 아님(메모)**: `SGRT_POSE=gt` 로 정답 자세를 넣어도 base delta 의 도착 판정·결과 `state`·`error` 는 바퀴 속도(base_qvel) 적분이다. 지도·go_to·probe·`moved_m` 만 정답 자세 | 실제 로봇에는 정답 자세가 없어 delta 는 오도메트리로 판정한다 → 시뮬에서도 같은 방식으로 시험하는 것이 맞다 | 그대로 둔다 |

## 3 번 목록 (`/mnt/c/behavior-2026` 을 쓰는 실행 줄이 있는 파일, 주석만 있는 것은 뺌)

서브모듈 `src/behavior-2026/` 기준.

| 폴더 | 파일 |
|---|---|
| `src/sim/engine/capture/` | `export_s3.py`, `extract_batch.sh`, `extract_instance.sh`, `run_capture_linux.sh` |
| `src/sim/engine/eval/` | `run_ported_engine.sh` |
| `src/sim/engine/scripts/` | `build_replay.sh`, `gen_aos.py`, `gen_aos_test.py`, `gpu_lock.sh`, `linux_env.sh`, `setup_linux_official.sh` |
| `src/sim/engine/tests/` | `articulation/` (`build.sh`, `l1_all.sh`, `ptxinfo.sh`, `run_gpu.sh`, `run_gpu_all.sh`, `run_gpu_regs.sh`, `snap.sh`, `snapneg.sh`, `sweep.sh`), `common/` (`run_sleef_trigf.sh`, `test_sleef_trigf.py`), `joints/` (`build.sh`, `run_cpu.sh`, `run_gpu.sh`, `run_gpu_sq.sh`), `omni/` (`build_omni.sh`, `gen_bddl_ref.py`, `run_gfmat.sh`, `run_obs.sh`, `test_gfmat.py`, `test_obs.py`), `particles/` (`build_particles.sh`, `capture_*.py`, `harvest_spawn.py`, `probe_load_order.py`, `run_*.sh`), `render/` (`build_render.sh`, `compare/build.sh`, `gpu_tests.sh`) |
| `src/sim/fasteval/` | `replaysrv/verify.sh`, `replaysrv/verify_vs_python.py`, `tracecmp/verify_vs_python.sh` |
| `src/sim/integ/` | `build_simlink.sh`, `fk/fit_cam_fk.py` |
| `src/agent/planner/src/` | `main.rs` (5 번) |

다시 뽑기(서브모듈 안에서):

```bash
git grep -n -E "/mnt/c/behavior-2026|wsl\.exe" -- 'src/*' 'tools/*' ':!*.md' ':!archive' \
  | grep -v -E ":[0-9]+:\s*(#|//|\"\"\"|\*)" | cut -d: -f1 | sort -u
```

## objprob 동작 확인 필요 (2026-10-06, objprob 단일화 때 찾음 — 코드는 안 바꿈)

| # | 어디 | 무엇 | 영향 | 다음 |
|---|---|---|---|---|
| O1 | `src/scene_graph/scenemap/src/objmap.cpp` `apObsStructural` | 바닥 높이의 수평 평면은 이름과 상관없이 바닥으로 거른다(`zhi < floor_z`). 러그·카펫·매트 예외(`floorLevel`)는 검출 하나 단위 `floor_h` 검사에만 남음 | 러그·카펫·매트가 물체 노드로 안 남음 | 살릴지 결정 → 살리면 이름 사후가 바닥 깔개면 예외 |
| O2 | `objmap.cpp` 조각 합치기(접촉 판정) + `capi.cpp` `robotParams` `body_r` 0.22 | 몸 반경 안의 구름 점을 빼서 조각끼리 닿음 판정이 안 섬 | 리모 바로 앞(약 0.2 m 안) 작은 물체(컵 4 cm)가 윗면·옆면 두 조각으로 남음 — **집기에 직접 영향** | 먼저 원인 확인: 몸 빼기를 접촉 판정 뒤로 미루거나 손목 카메라 점은 빼지 않기 |
| O3 | `objmap.cpp` `update` 맨 앞 `apMergePass` | 같은 영상에서 새로 생긴 조각은 다음 keyframe 에서야 합쳐짐 | 한 keyframe 동안 같은 물체가 둘로 보임(짧음) | 큰 문제 아니면 둠 |
| O4 | (시험 `test_objmem` boxes, 원인 모름) | 높은 위치(영상 행 280..400)의 넓은 소파는 물체가 생겨도 스냅숏에 안 나옴(행 350..440 이면 나옴) | 높이 있는 넓은 가구가 그래프에서 빠질 수 있음 | 원인 찾기 |
