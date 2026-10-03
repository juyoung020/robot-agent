# scene_graph

동적 scene graph. 로봇이 본 물체를 실시간 2D 지도에 등록·갱신한다 (로봇의 기억). **실제 로봇 쪽 코드**다.

시뮬 테스트 공간은 서브모듈 `src/behavior-2026`(BEHAVIOR Challenge 2026)이고, 그쪽 `src/scene_graph/` 에서 먼저 시험한 코드를 `tools/sync_scene_graph.sh` 로 여기에 맞춘다. **원본은 behavior-2026** — 여기서 직접 고치지 말고 거기서 고친 뒤 서브모듈을 갱신하고 동기화한다(`--check` 로 어긋남 확인).

| 폴더 | 내용 |
|---|---|
| `scenemap/` | 2D SLAM, 물체 지도, 벽 2D 상태, Spark-DSG 저장, C ABI (C++/CUDA) |
| `da/` | 데이터 연관 — 프레임마다 나온 같은 물체 세그를 하나로 병합 |
| `spark_dsg/` | Spark-DSG 우리 수정본(물체·방 2층으로 줄임, `OUR_CHANGES.md`) |
| `sgview/` | 실시간 뷰어(Rust 서버 + three.js, 파이썬 없음) |
| `runtime/` | sgrt — scenemap + 검출기 + 임베딩을 한 C ABI 로 |
| `ovdet/`, `clip/` | 물체 검출기(YOLOE), 물체 이미지 임베딩(SigLIP 2) |

빌드·시험: `cmake -S src/scene_graph/scenemap -B build/scenemap && cmake --build build/scenemap -j && (cd build/scenemap && ctest)` (9/9), 뷰어는 `cd src/scene_graph/sgview && cargo build --release`.
