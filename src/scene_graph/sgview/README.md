# sgview — 장면 그래프 뷰어 (Rust)

파이썬(viser)·Spark-DSG 없이 메모리 폴더를 브라우저에서 실시간으로 본다. Rust 서버(std만, 의존 crate 없음)가 `view.json`,
`map.pgm`/`map.yaml`, `objects/O<id>_*` 를 서빙하고, 그리기는 three.js(`assets/`에 같이 넣음, 인터넷 불필요).

```bash
cd src/scene_graph/sgview && cargo build --release
target/release/sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]
```
브라우저에서 http://localhost:8080 . 0.25 s 마다 `view.json` 이 바뀌었는지 보고 바뀌면 다시 그린다.

## 보이는 것 (파이썬 sgviz 와 같은 기능)
- 점유 지도, 로봇(원판 + 화살표), 물체 세그먼트 점구름(true/state colour), 이름표, 천장 숨김(기본), 구조물/gone 켜고 끄기.
- 공중에 뜬 그래프 3층: 물체(z 4 m, 상태 색 + 썸네일), 장소(z 6 m, 여유 거리 색), 방(z 8 m) + 에이전트 궤적, 간선(장소–장소, 방→물체). 전치사 관계·building·frontier 는 그리지 않는다.
- 2D 벽(하늘색 선) + 벽 상태 벡터(길이 56) + "Save wall state" 로 JSON 저장.
- 물체 클릭(그래프 노드 12 px 안 → 아니면 경계 상자 광선) 또는 드롭다운 → 정보 표 + RGB/마스크/깊이 조각.

## 구조
- `src/main.rs` 서버, `src/walls_ffi.cpp` 가 scenemap 의 `walls.cpp`(C++)를 부른다(`build.rs` 가 같이 컴파일). 벽 계산은 한 군데(C++)에만 있다.
- 벽은 `map.pgm` 이 바뀔 때만 다시 계산(mtime+크기 캐시).
- `assets/index.html` 이 화면 전부.

## 아직 없는 것 (파이썬 sgviz 에는 있던 것)
- 마스크 윤곽선을 RGB 위에 겹치기, 깊이 조각 회색조 정규화/통계(지금은 저장된 PNG 를 그대로 보여 준다).
- 텍스트 검색 패널, explore 오버레이(`memory/explore.json`).
- 점 PLY 고르기는 점이 아니라 경계 상자로 맞춘다.
