# src/robot — 리모(AgileX LIMO) + 매니퓰레이터(ROBOTIS OMX-F) 로봇 설명

ROS 2 패키지(colcon). 큰 메시 때문에 업스트림 두 패키지는 저장소에 넣지 않고 받아서 쓴다.

| 패키지 | 어디에 있나 | 비고 |
|---|---|---|
| `map_vla_description` | 이 저장소 | 리모 위에 OMX-F 를 얹은 통합 URDF(xacro), RViz 실행, 손목 카메라 프레임 |
| `limo_description` | `fetch_upstream.sh` 가 받음 | agilexrobotics/limo_ros2 @ dcc5a86 + `limo_description.patch` (depth_camera_visual 상자 제거, 깊이 카메라 위치 (0.084,0,0.03)). 원본 라이선스 표기가 TODO 라 재배포하지 않음 |
| `open_manipulator_description` | `fetch_upstream.sh` 가 받음 | ROBOTIS-GIT/open_manipulator @ d31000d, 수정 없음(Apache-2.0) |

```bash
src/robot/fetch_upstream.sh                       # 업스트림 두 패키지 받기 + 패치
source /opt/ros/humble/setup.bash
PATH=/usr/bin:$PATH colcon build --base-paths src/robot --symlink-install -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch map_vla_description view.launch.py    # RViz 에서 TF 확인
```

- 카메라: 리모 깊이 카메라와 라이다는 원본, 손목 카메라(`wrist_cam_link`)만 추가. OMX 메시에 이미 들어 있는 카메라 위치를 측정해 맞췄다(별도 상자 없음).
- `xacro` 는 `pip install --user xacro`, colcon 은 시스템 python 으로 돌린다(miniconda python 이면 `catkin_pkg` 가 없어 실패).

## 뷰어용 모델 자산 (`src/scene_graph/sgview/assets/robot/`)

실시간 뷰어(sgview)가 로봇을 URDF 모델로 그린다. LIMO 메시는 75만 면(63 MB)이라 브라우저에 그대로 못 보내서, 오프라인으로 한 번 변환한 결과(GLB 10개 + `robot.json`, 합 3.4 MB)를 저장소에 넣었다. 서버 바이너리에 내장된다.

```bash
# xacro 를 펼친 URDF + 메시 → GLB(색 그룹별 병합·정점 클러스터링 단순화) + 링크·관절 트리
xacro map_vla_description/urdf/map_vla.urdf.xacro > map_vla.urdf
python3 src/robot/tools/build_viewer_assets.py map_vla.urdf src/robot src/scene_graph/sgview/assets/robot   # pip: trimesh numpy pycollada fast-simplification
```
- LIMO 본체 75.6만 → 8.6만 면(1.5 MB), 바퀴 14만 → 1.2만 면, OMX 8개 링크는 STL 을 mm → m 로 굽고 2.5만 면 상한(0.05~0.45 MB).
- 이 자산은 `robot-agent` 에만 있다(`behavior-2026` 의 뷰어는 없으면 상자 모양으로 대체). `tools/sync_scene_graph.sh` 가 이 폴더를 지우지 않는다.
- 뷰어에서 관절 움직임: 브라우저 콘솔 `setJoints({omx_joint2: 0.5, omx_joint3: -0.6})`. 스트림의 관절 벡터를 쓰려면 `robot.json` 의 `joint_order`(관절 이름 순서)를 채우면 된다 — 지금 시뮬의 로봇은 R1 Pro 라서 비어 있다.
