# scene_graph

동적 scene graph. 로봇이 본 물체를 실시간 2D 지도에 등록·갱신한다 (로봇의 기억).

코드는 서브모듈 [`src/behavior-2026/src/scene_graph/`](../behavior-2026/src/scene_graph) 에 있다(`git submodule update --init src/behavior-2026`).

| 폴더 | 내용 |
|---|---|
| `scenemap/` | 2D SLAM, 물체 지도, 벽 2D 상태, Spark-DSG 저장, C ABI (C++/CUDA) |
| `da/` | 데이터 연관 — 프레임마다 나온 같은 물체 세그를 하나로 병합 |
| `spark_dsg/` | Spark-DSG 우리 수정본(물체·방 2층으로 줄임, `OUR_CHANGES.md`) |
| `sgview/` | 실시간 뷰어(Rust 서버 + three.js, 파이썬 없음) |
| `runtime/` | sgrt — scenemap + 검출기 + 임베딩을 한 C ABI 로 |
| `ovdet/`, `clip/` | 물체 검출기(YOLOE), 물체 이미지 임베딩(SigLIP 2) |
| `viewer/` | 옛 파이썬 뷰어(sgviz), 제거 예정 |
