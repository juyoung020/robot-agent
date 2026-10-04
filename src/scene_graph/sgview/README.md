# sgview — 장면 그래프 뷰어 (Rust) · 이것이 지금 뷰어다

**Spark-DSG 장면 그래프 보기 = sgview.** 물체 기억(scene graph, Spark-DSG 형식 `scene.json`·`view.json`)을 보는 뷰어는 이것 하나다
(spark dsg viewer / scene graph viewer / 물체 기억 뷰어). Rust 서버 + three.js, 파이썬 없음.
옛 Python 뷰어 sgviz(`../viewer/`, Spark-DSG + viser)는 파일 폴링이라 실시간이 아니고 쓰지 않는다(기록용).

| 하고 싶은 것 | 명령 (robot-agent 저장소 루트) |
|---|---|
| 탐사 한 판 + 실시간 뷰어 한 번에 | `tools/run_explore_live.sh [policy] [task] [tag]` — LIMO 는 `SGRT_ROBOT=limo_omx tools/run_explore_live.sh …` |
| 실시간 스트림 받기(60 Hz 이상) | `tools/run_sgview.sh <memory_dir> --live` + 시뮬/로봇 쪽 `SGRT_STREAM=127.0.0.1:9001` |
| 녹화된 메모리 폴더 보기 | `tools/run_sgview.sh <memory_dir>` |
| behavior-2026 만 있을 때(파일 모드) | `src/sim/explore/viewer_8080.sh <run dir>` 또는 아래 직접 실행 |

실시간 경로: C++ libsgrt → 소켓(`SGRT_STREAM=127.0.0.1:9001`) → Rust 서버(`--ingest`) → SSE → 브라우저. 브라우저에서 http://localhost:8080 .

Rust 서버(std + `serde_json`·`flate2`)가 `view.json`,
`map.pgm`/`map.yaml`, `objects/O<id>_*` 를 서빙하고, 그리기는 three.js(`assets/`에 같이 넣음, 인터넷 불필요).

```bash
cd src/scene_graph/sgview && cargo build --release
target/release/sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]
```
브라우저에서 http://localhost:8080 . 0.25 s 마다 `view.json` 이 바뀌었는지 보고 바뀌면 다시 그린다.

## 보이는 것 (옛 파이썬 sgviz 의 기능을 옮김)
- 점유 지도, 로봇(원판 + 화살표), 물체 세그먼트 점구름(true/state colour), 이름표, 천장 숨김(기본), 구조물/gone 켜고 끄기.
- 공중에 뜬 그래프 2층: 물체(z 4 m, 상태 색 + 썸네일), 방(그 위 층 간격 2 m) + 에이전트 궤적(바닥), 간선(층 안·방→물체). 장소·전치사 관계·building·frontier 는 그리지 않는다.
- 2D 벽(하늘색 선) + 벽 상태 벡터(길이 56) + "Save wall state" 로 JSON 저장.
- slam 지도는 출발 자세 좌표라 벽이 기울어져 있다(예: turning_on_radio 출발 yaw 49°). 벽 선분은 `wallSegmentsAligned` 로 벽 방향 θ 에서 찾고(`/api/walls`·`walls` 이벤트에 `theta`), 페이지는 \|θ\| > 1° 면 점유 지도를 θ 로 돌린 격자로 다시 뽑아 그린다(칸 계단 → 곧은 벽, 보기만 바뀜). gt 지도(축에 맞음)는 예전과 같다. 선분은 원래 격자의 점유 띠에 다시 맞춰 벽 위에 놓인다(scenemap `wallSegmentsAligned`).
- 궤적 라벨은 로봇 이름: `view.json` 의 `robot`, 없으면 `/api/robot`(실행 폴더 `robot_footprint.json` 의 `robot`, 옛 파일은 AABB 로 R1 Pro·LIMO 구분), 모르면 "robot".
- 물체 클릭(그래프 노드 12 px 안 → 아니면 경계 상자 광선) 또는 드롭다운 → 정보 표 + RGB/마스크/깊이 조각.

## 구조
- `src/main.rs` 서버, `src/walls_ffi.cpp` 가 scenemap 의 `walls.cpp`(C++)를 부른다(`build.rs` 가 같이 컴파일). 벽 계산은 한 군데(C++)에만 있다.
- 벽은 `map.pgm` 이 바뀔 때만 다시 계산(mtime+크기 캐시).
- `assets/index.html` 이 화면 전부.

## 아직 없는 것 (옛 파이썬 sgviz 에는 있던 것)
- 마스크 윤곽선을 RGB 위에 겹치기, 깊이 조각 회색조 정규화/통계(지금은 저장된 PNG 를 그대로 보여 준다).
- 텍스트 검색 패널, explore 오버레이(`memory/explore.json`).
- 점 PLY 고르기는 점이 아니라 경계 상자로 맞춘다.

## 실시간 모드 (파일 없이, 60 Hz 이상)

시뮬/로봇 프로세스(`sgrt`)가 소켓으로 바로 보내고, 서버가 브라우저에 SSE 로 흘려 준다. 파일은 거치지 않는다(이미지·점구름 같은 큰 파일만 `/file/` 로).

```bash
sgview <memory_dir> --port 8081 --ingest 127.0.0.1:9001          # 1) 뷰어: 9001 로 프레임을 받는다. memory_dir 은 sgrt 의 out 폴더(이미지·점구름 파일 위치)
SGRT_STREAM=127.0.0.1:9001 SGRT_STREAM_HZ=60 SGRT_MAP_EVERY=1 <시뮬 실행>   # 2) sgrt 가 스트림을 연다(요약 주기 기본 60 Hz, 최대 240)
```

```
sgrt(C++) ──소켓(프레임)──▶ sgview(Rust) ──SSE──▶ 브라우저(three.js)
 스텝 스레드: 링 버퍼에 복사만            상태 보관 + 방송          60 fps+ 렌더, 자세 보간
```

- **스텝 스레드는 아무것도 기다리지 않는다** (`scenemap/stream.hpp`): 락·시스템 호출·할당 없는 단일 생산자 링 버퍼에 복사만 하고 별도 스레드가 비차단 소켓으로 보낸다. 자세 ≈ 0.2~0.3 µs, 지도 바뀐 영역 10 KB ≈ 0.5 µs. 뷰어가 느리거나 끊겨도 스텝은 안 막힌다(링이 차면 그 프레임만 버림). 다시 붙으면 송신 스레드가 들고 있는 그림자 지도·요약·자세로 전체 상태를 다시 보낸다.
- **보내는 것**: 자세(`pose`), 지도 바뀐 영역만(`map`, 격자 변화 추적), 물체·방·로봇 궤적 요약(`view`, PLACES 는 뺌, 안 바뀌면 안 보냄), 벽 선분·벽 상태 벡터(`walls`, 서버가 C++ `walls.cpp` 로 계산, 선분은 바뀔 때만).
- **브라우저**: 자세는 표시 주기(60 fps+)로 지수 보간, 지도는 바뀐 행만 텍스처에 쓰고, 요약은 기하에 영향 주는 값(위치·상태·점 버전 …)이 바뀔 때만 다시 그린다. 새 시뮬 실행이 붙으면 `reset` 으로 이전 화면을 비운다. 우상단 패널 맨 위에 단계별 비용(HUD)이 보인다.
- **합성 부하 시험**: `scenemap/tools/stream_sim <memory_dir> 127.0.0.1:9001 24 240 60 60` — 자세 240 Hz, 지도 60 Hz, 요약 60 Hz 를 보내 본다. 측정(GPU 브라우저): 받은 주기 pose 240/s·map 60/s·view 60/s, 이벤트 처리 0.00~0.27 ms, 렌더 60 fps(디스플레이 상한, 프레임 0.4 ms), 시작 첫 데이터 16 ms · 장면 완성 213 ms. 송신 쪽 pushPose 평균 227 ns, 버려진 프레임 0.
- 시뮬 자체가 자세를 7 Hz 정도로만 내보내면(Isaac Sim 렌더 속도) 뷰어는 그 사이를 보간한다. 추론 주기(10~20 Hz)가 얼마든 뷰어가 병목이 되지 않게 한 것.
- 파일 모드(`sgview <memory_dir>` 만)는 녹화된 폴더를 보는 용도로 그대로 있다.
- **지도 갱신 주기 분리 (`SGRT_MAP_EVERY=n`)**: 점유 지도를 검출 키프레임(`kf_every`, 기본 6 스텝)이 아니라 n 스텝마다 깊이만으로 갱신한다(스캔 ≈ 0.47 ms + 격자 ≈ 0.07 ms). 검출·물체 지도·임베딩은 키프레임 그대로. 0/미설정 = 예전 동작. 같은 시나리오 측정: 지도 갱신 초당 2~3번 → 14번(자세와 같은 주기), 스텝 평균 1.01 → 1.52 ms, 탐색 커버리지 0.906 → 0.927(나빠지지 않음).
- **관절·상태 벡터 프레임(`joints`)**: `sm_push_proprio` 의 벡터를 그대로 스트림에 싣는다(로봇별 순서는 URDF 가 정함). 지금은 받아서 보관만 하고(`jointsCur`), URDF 모델을 올릴 때 이 값으로 로봇을 움직인다.
- 카메라는 지도가 커지는 동안 자동으로 따라가고(직접 조작하면 멈춤, `Fit` 으로 다시 켬).
