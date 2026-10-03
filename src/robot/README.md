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
