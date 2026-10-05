# scene_graph

동적 scene graph. 로봇이 본 물체를 실시간 2D 지도에 등록·갱신한다 (로봇의 기억). **실제 로봇 쪽 코드**다.

> **보기(뷰어) = sgview** ([sgview/README.md](sgview/README.md)): 실시간 `tools/run_explore_live.sh`(탐사 한 판 + 뷰어, LIMO 는 `SGRT_ROBOT=limo_omx`), 폴더·스트림 `tools/run_sgview.sh <memory_dir> [--live]`(시뮬 쪽 `SGRT_STREAM=127.0.0.1:9001`). `scene_graph/viewer/`(sgviz, Python + viser)는 옛 뷰어로 실시간이 아니다 — 쓰지 않는다.

**robot-agent 의 `src/scene_graph/` 가 유일한 원본이다**(10-06 결정: 원래 behavior-2026 서브모듈에서 시험하고 `tools/sync_scene_graph.sh` 로 복사하던 것을 뒤집음. 복사 도구는 `archive/tools/` 로 옮김). 시뮬(OmniGibson) 쪽 실행기는 이 코드를 그대로 가져다 쓴다. `tools/realbag/`(실제 bag 재생·objprob 맞춤)·`viewer/`(옛 sgviz)도 여기 있다.

| 폴더 | 내용 |
|---|---|
| `scenemap/` | 2D SLAM, 물체 지도, 벽 2D 상태, Spark-DSG 저장, C ABI (C++/CUDA) |
| `da/` | 데이터 연관 — 프레임마다 나온 같은 물체 세그를 하나로 병합 |
| `spark_dsg/` | Spark-DSG 우리 수정본(물체·방 2층으로 줄임, `OUR_CHANGES.md`) |
| `sgview/` | **뷰어는 이것** — Spark-DSG 장면 그래프 보기 = sgview (Rust 서버 + three.js, 파이썬 없음, 실시간). [sgview/README.md](sgview/README.md) |
| `runtime/` | sgrt — scenemap + 검출기 + 임베딩을 한 C ABI 로 |
| `ovdet/`, `clip/` | 물체 검출기(YOLOE), 물체 이미지 임베딩(SigLIP 2) |

빌드·시험: `cmake -S src/scene_graph/scenemap -B build/scenemap && cmake --build build/scenemap -j && (cd build/scenemap && ctest)` (12/12, 10-04 확인 — `limo_fk`·`limo_e2e` 등 추가), 뷰어는 `cd src/scene_graph/sgview && cargo build --release`.
