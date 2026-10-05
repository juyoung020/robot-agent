# [옛 뷰어, 안 씀] sgviz — 물체 기억 Python 뷰어 (Spark-DSG + viser, 실시간 아님)

> **옛 Python 뷰어(실시간 아님). 지금 뷰어는 sgview(Rust) — tools/run_sgview.sh / tools/run_explore_live.sh**
>
> - 이 폴더(sgviz, Spark-DSG + viser)는 기록용으로만 남겨 둔다. `scene.json` 파일을 폴링할 뿐이라 실시간이 아니다. 새로 쓰지 말 것.
> - "Spark-DSG 장면 그래프 보기" = **sgview**: [`../sgview/README.md`](../sgview/README.md) (Rust 서버 + three.js, sgrt 소켓 → SSE 60 Hz 이상).
> - 실행(robot-agent 저장소): 녹화 폴더 `tools/run_sgview.sh <memory_dir>`, 실시간 `tools/run_sgview.sh <memory_dir> --live`(시뮬 쪽 `SGRT_STREAM=127.0.0.1:9001`),
>   탐사 한 판 + 실시간 뷰어 `tools/run_explore_live.sh` (LIMO 는 `SGRT_ROBOT=limo_omx`). 이 저장소만 있으면 `src/sim/explore/viewer_8080.sh <run dir>`(sgview 파일 모드).
> - 규칙: 런타임·학습·추론은 Rust/C++/CUDA, 파이썬은 오프라인 도구만. `walls2d.py` 는 scenemap `walls.cpp` 의 원본(값 비교용)이라 같이 남긴다.


메모리 런타임이 ~1 s 마다 다시 쓰는 디렉터리(`scene.json`, `map.pgm`/`map.yaml`,
`objects/O<id>_{rgb,depth}.png`)를 브라우저에서 실시간으로 본다.
`spark_dsg.viser.ViserRenderer` 위에 얇게 얹은 것(그래프 읽기·상자 모서리는 spark_dsg API).

## 실행

```bash
~/sdsg_venv/bin/python src/scene_graph/viewer/sgviz.py <memory_dir> [--port 8080]
# 예: outputs/mem_comet12_water_gui_*/memory
```

그다음 브라우저에서 http://localhost:8080 . (원격이면 `ssh -L 8080:localhost:8080`.)

환경: `~/sdsg_venv`(python 3.11, spark_dsg) 에 `viser`, `pillow` 추가
(`VIRTUAL_ENV=~/sdsg_venv uv pip install viser pillow`).

## 보이는 것

- 바닥: 2D 점유 지도(회색 단계 그대로, 205 = 미지 = 청회색).
- 로봇: 검은 원판 + 빨간 화살표(robot_pose x, y, yaw).
- 물체 형태: `metadata.points.path`(`objects/O<id>_points.ply`, binary_little_endian, float x,y,z 지도 좌표 m
  + uchar red,green,blue)의 세그먼트 점을 실제 색으로 그린다(점 크기 = voxel × 배율,
  "Point colour" 로 true colour / state colour 전환). PLY 는 numpy 로 직접 읽고,
  파일 mtime·크기가 바뀔 때만 다시 읽는다(scene.json 보다 늦게 써져도 다음 poll 에 잡힘).
- 점이 있는 물체는 경계 상자를 그리지 않는다. 상자는 점이 없는 물체에만, "Boxes" 를 켰을 때만
  (기본 꺼짐; 한 변이 "Max box side" 보다 크면 잘라 그림). bounding_box 자체는 scene.json 에 그대로 있다.
- `name#id` 이름표(점 위). 중심점은 선택 사항("Centre markers", 기본 꺼짐). 점이 없는 물체는 상태별 색 구
  (seen 초록, moved 주황, held 파랑, gone 회색 반투명).
- 물체끼리 관계(`on`/`in`/`near`)는 읽지도 그리지도 않는다. 물체 연결은 방 → 물체(부모)뿐이다.
- moved 물체: first_pos → pos 궤적.
- 고르기: 물체의 점(복셀) 아무 데나 누르면 그 물체가 선택된다. viser 장면 클릭
  (`server.scene.on_click`)의 광선(origin, direction)으로 numpy 에서 고른다: 물체마다 점 배열과
  (반경만큼 키운) AABB 를 들고, 광선이 AABB 를 지나는 물체만 점-광선 수직 거리 ≤ max(2×voxel, 2 cm)
  인 점을 찾아 광선 방향으로 가장 가까운 물체를 고른다. 점이 없는 물체는 구(반경 1.5×)로 맞춘다.
  빈 곳을 누르면 선택은 그대로(드래그는 카메라 회전). 50 물체 × 4000 점 최악 ~1.7 ms.
  선택된 점 구름은 노랗게 물들이고 조금 키워 보인다(구 물체는 노란 철망 구).
- 오른쪽 패널 "Object": 물체를 누르거나 드롭다운에서 고르면 name/id/state/pos/first_pos/n_obs/
  score/마지막 관측/bbox/room/점 수(SigLIP 2 이름·임베딩이 있으면 그것도), `metadata.rgbd` 가 있으면 box_px/depth_m 과 조각 그림:
  - RGB 조각 + 세그먼트(`rgbd.mask` = `objects/O<id>_mask.png`, 8-bit 255 = 안쪽, 같은 조각 상자):
    바깥은 어둡게, 경계는 노란 선.
  - 깊이 조각은 회색조(가까울수록 밝게). 범위는 마스크 안 유효 깊이의 2–98 백분위, 마스크 밖은 어둡게,
    0(무효)은 검정. 아래에 마스크 안 깊이 최소/중앙/최대 [m] (마스크가 없으면 조각 전체).
  - 작은 조각은 정수배로 키워 보여 준다.
  데이터가 바뀌면 패널도 갱신. 그림은 고를 때만 읽는다(경로+mtime 캐시).
- "Display" 폴더: 이름표/세그먼트 점/점 색/점 크기/중심점/상자/방 라벨/최소 관측 수/궤적/구조물/천장/2D 벽/gone/지도, 노드 크기, 상자 최대 변.

scene.json 의 mtime 을 0.25 s 마다 보고, 바뀐 노드·간선만 다시 그린다(카메라 유지).

## 시험

```bash
~/sdsg_venv/bin/python src/scene_graph/viewer/test_sgviz.py
```

합성 디렉터리(물체 3, 점 PLY·rgbd·마스크 1, 지지 간선 1)를 spark_dsg 로 만들어 읽기·PLY(이진/ascii)·
차이 계산·마스크 겹침·회색조 깊이와 통계·지도 텍스처·"점 있으면 상자 없음"·PLY 갱신 감지,
광선 고르기(맞음/앞의 것 우선/빗나감/시간)와 뷰어 안에서 클릭→패널을 확인하고, 실제 viser 서버를 띄워 선택 패널과 HTTP 200 을 확인한다(브라우저 없음).

## 참고

- `rgbd.rgb/depth/mask` 그림이 box_px 크기면 그대로, 전체 프레임이면 box_px 로 자른다(여백 8 px).
- spark_dsg 의 `ViserRenderer.draw()`(GraphHandle) 는 viser 1.x 에서 없어진 API
  (`server.add_folder` 등)를 써서 그대로는 안 돈다. 그래서 서버 소유만 물려받고 그리기는 여기서 한다.
- Rust 뷰어는 `src/scene_graph/sgview`(따로 README)에 있다. 이 폴더의 `target/` 은 git 무시 대상이다.

## Map_Vla 에서 추가한 것 (천장 숨김 · 2D 벽 · 벽 state)

원본(팀 `behavior-2026`)에 더한 부분이다. 원래 동작은 체크를 끄면 그대로 나온다.
- **Hide ceiling** (기본 켬) + **Ceiling cut height [m]** (기본 2.1): 이 높이보다 위의 세그먼트 점을 그리지 않고, 중심이 그보다 높은 물체와 이름이 `ceiling(s)`/`roof`로 끝나는 물체는 통째로 뺀다. 이름만으로는 열린 어휘 라벨이 흔들려서 높이 기준이 주력이다.
- **Walls as 2D lines** (기본 켬): 이름이 `wall(s)`/`baseboard`/`wainscoting`/`drywall`로 끝나는 물체(`wall mounted tv`는 해당 안 됨)의 3D 점을 빼고, 점유 지도에서 뽑은 벽 선분을 바닥에 하늘색 선으로 그린다(주황은 이동 궤적 색).
- **Wall state (2D walls -> numbers)** 패널: 로봇 기준 수치(길이 56). 레이아웃은 `walls2d.py` 맨 위 설명이 기준이다.
  - 앞쪽부터 반시계로 16개 방향의 첫 점유 칸까지 거리(÷4 m), 가장 가까운 벽 선분 8개의 로봇 프레임 끝점 + 유효 표시.
  - 단추로 `outputs/wall_state/<이름>_t<시각>.json/.npy` 저장(`SGVIZ_STATE_DIR`로 바꿈). 헤드리스: `python walls2d.py <memory_dir> [--pose x y yaw] [--out f.json]`.
- 한계: 벽 선분은 축에 평행한 선만 뽑는다(기울어진 벽은 놓침). 이 지도에서는 점유 칸의 약 59%가 선분 0.15 m 안에 든다. 파이썬 개발 도구이고, 학습·추론 경로에는 같은 레이아웃의 네이티브 버전(`sm_grid` 위 C++: scenemap `walls.cpp` `wallStateVector`, C ABI `sm_snap_wall_state`)을 쓴다.
- **물체끼리 관계**: 옛 `scene.json`에 들어 있는 물체끼리 `on/in/near` 간선은 무시한다. 새 `scenemap`은 이 간선을 만들지 않는다(방 → 물체만). `Min observations`(기본 3)는 두 번 이하 본 물체를 숨긴다.
- **방 → 물체**: 물체 패널에 `room` 행이 나오고, `Room in label`을 켜면 라벨에 `[room N]`이 붙는다.
- **Show graph layers**(Hydra 층 쌓기, 공중에 뜬 그래프)는 기본 켜짐이고 높이는 물체 4 m, 장소 6 m, 방 8 m(간격 2 m)이다. 건물 층은 뺐다(옛 파일에 있어도 그리지 않음).
- **frontier는 뷰어에 표시하지 않는다.** frontier는 탐사 스킬이 쓰는 내부 계산 값이고, Hydra의 파이썬·Spark-DSG 뷰어도 따로 그리지 않는다. 장소는 여유 거리 색으로만 칠하고 패널에도 frontier 개수를 적지 않는다(계산은 `scenemap`에 그대로 있음).
