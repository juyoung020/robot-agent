# 시뮬레이션 CUDA 포팅 계획

목표: LIMO + OMX-F 시뮬레이션을 GPU에서 수천 개 병렬로 돌리고, **state와 depth 이미지**를 GPU 메모리에서 바로 학습 커널로 넘긴다.
제약: `CLAUDE.md`에 따라 C++/CUDA/Rust만 쓴다. 참조 코드는 전부 Python(Warp/PyTorch)이므로 **코드를 가져오지 않고 설계와 알고리즘만 옮긴다.**

환경 확인: nvcc 13.2, rustc/cargo, g++, cmake 있음. GPU는 5070 Ti(compute 12.0, 드라이버 580).

---

## 1. 참조에서 직접 확인한 것

### MuJoCo Warp (`refs/code/mujoco_warp`)
- **월드 배치 레이아웃**: 모든 상태가 첫 차원이 `nworld`인 배열이다. `Data.qpos (nworld, nq)`, `qvel (nworld, nv)` (`_src/types.py:2253`).
- **스텝 순서**: `fwd_position → fwd_velocity → fwd_actuation → fwd_acceleration → solver.solve` (`_src/forward.py:2059` forward, `:2097` step).
- **렌더링은 megakernel 1개**: `wp.launch(dim=(nworld, total_rays))`로 광선 하나당 스레드 하나를 돌린다 (`_src/render.py:1490` `render`, `:991` `_build_megakernel`). 지오메트리는 BVH로 가속한다 (`_src/bvh.py`, `_src/ray.py`).
- **depth 출력 버퍼**: `rc.depth_data`는 `(nworld, 카메라별 H*W의 합)` 연속 버퍼이고, `rc.depth_adr`가 카메라별 시작 오프셋이다 (`_src/render_util_test.py:219-224`). 호스트로 복사하지 않고 장치 메모리에 둔다.
- **CUDA graph**: 스텝 전체를 한 번 캡처해서 재생한다 (`_src/cli.py:264-290`). 커널 런치 오버헤드를 없애는 핵심이다.
- 규모: `solver.py` 약 4.4k줄, `smooth.py` 약 4.6k줄, `collision_*` 다수. 범용 엔진이라 크다.

### ManiSkill (`refs/code/ManiSkill`)
- **모바일 베이스는 바퀴 마찰을 시뮬레이션하지 않는다.** Fetch는 `root_x_axis_joint`, `root_y_axis_joint`, `root_z_rotation_joint` 가상 조인트 3개로 베이스를 움직인다 (`mani_skill/agents/robots/fetch/fetch.py:108`).
- 관측 모드가 `state`, `depth`, `rgbd` 등으로 분리되어 있다 (`mani_skill/envs/sapien_env.py:53`, `get_obs` `:502`, `_get_obs_sensor_data` `:579`).

### 조사 결과 (코드는 안 읽음)
- MJWarp 배치 렌더러는 레이트레이싱 기반이고, 이전의 Madrona-MJX는 유지보수가 끝나 MJWarp로 대체되었다고 한다.
- Newton은 MJWarp를 백엔드로 쓴다. Newton과 Madrona는 아직 열어 보지 않았다.

---

## 2. 우리 설계 (제안: 참조에서 확인한 사실이 아니라 제 판단)

### 2.1 물리 범위를 줄인다
MJWarp 전체를 포팅하는 것은 과하다. 우리에게 필요한 것은 다음뿐이다.
| 요소 | 방식 |
|---|---|
| LIMO 베이스 | **평면 키네마틱** `(x, y, yaw)`를 속도 명령으로 적분. 바퀴-지면 마찰은 안 돌림 (ManiSkill Fetch 방식) |
| OMX-F 5축 + 그리퍼 | 관절 PD 제어. 자유도가 작으므로 **월드당 스레드 1개**로 FK/역학을 계산 |
| 물체 | 처음엔 정적 + 그리퍼가 닫히면 부착(kinematic grasp). 접촉 solver는 나중에 필요할 때 추가 |
| 지도 | 정적 삼각형 메시 → BVH |

### 2.2 메모리 레이아웃 (SoA, 장치 상주)
```
base_pose   float[nworld][3]      x, y, yaw (map 프레임)
base_vel    float[nworld][3]
arm_q       float[nworld][6]      joint1~5 + gripper
arm_qd      float[nworld][6]
obj_pose    float[nworld][nobj][7]
depth       float[nworld][ncam*H*W]   // 미터, 무효=0, 카메라별 오프셋 테이블
state_out   float[nworld][S]          // 정규화 완료된 학습 입력
```

### 2.3 한 스텝 = CUDA graph 1개
`action 읽기 → 베이스 적분 → 관절 PD → FK → 카메라 포즈 → depth 렌더 → state 패킹`
호스트 동기화 없이 장치에서 끝낸다.

### 2.4 depth
- 광선 1개당 스레드 1개, 장면 BVH 순회 (MJWarp megakernel 구조).
- 동적 물체는 BVH를 매 스텝 갱신하거나 별도 소형 BVH로 둔다.
- SigLIP2에 넣으려면 depth를 정규화하고 3채널(colormap)로 바꾸는 커널을 같은 스트림에 둔다 (앞서 정한 방식).

### 2.4.1 카메라 외부/내부 파라미터
외부 파라미터는 URDF TF를 그대로 쓴다. 카메라 본체는 두 로봇의 **메시에 이미 들어 있으므로 별도 상자를 추가하지 않는다.**
`base_footprint` 기준 위치 (실행 중인 TF에서 읽은 값):
| 프레임 | 위치 (m) | 근거 |
|---|---|---|
| `depth_camera_link` (LIMO 깊이 카메라) | (0.084, 0, 0.180) | LIMO 메시의 카메라 본체(부품 856) 중심 (0.084, 0, 0.030, base_link 기준). 원본 URDF는 z=0.3이라 소수점이 어긋난 것으로 보여 0.03으로 고쳤다 |
| `laser_link` (LIMO 라이다) | (0.103, 0, 0.116) | 원본 그대로 |
| `wrist_cam_link` (OMX 손목 RGB 카메라, **깊이 없음**) | (0.231, 0, 0.396) | OMX-F `link5` 메시에 들어 있는 카메라. `omx_link5` 기준 렌즈 중심 (0.0498, 0, 0.0351), 광축은 앞쪽 아래 37.94°, 요 0° |
- 손목 카메라의 ROBOTIS 기본 이름은 `camera1`(`/camera1/image_raw`)이다 (physical_ai_tools `omx_f_config.yaml`). OMX 공식 URDF에는 카메라 링크가 없어서 메시에서 직접 측정했다.
- 내부 파라미터(fx, fy, cx, cy, 해상도)는 정하지 않았다. 팀 `scenemap`이 영상마다 `fx, fy, cx, cy`를 받으므로 같은 방식으로 넘긴다. OMX USB 카메라는 640×480 30fps가 문서의 예시이고, 시야각은 모른다.
- LIMO 깊이 카메라의 실제 모델과 시야각도 확인하지 못했다.

### 2.5 state 인터페이스 (VLA 입력)
- base pose(map), base 속도, 관절 6개, EE pose(base 프레임), 가까운 물체 K개의 **로봇 프레임 상대 좌표**(부족하면 패딩 + 마스크).
- 정규화 통계는 학습 데이터에서 계산해 저장한다. 시뮬과 실제 로봇이 같은 정규화를 쓰게 한다.

---

## 3. 단계와 검증 기준
| 단계 | 내용 | 통과 기준 |
|---|---|---|
| M0 | URDF 파서 + FK (C++ 또는 Rust) | zero pose에서 `base_footprint→omx_end_effector_link` = (0.273, −0.002, 0.361), `→depth_camera_link` = (0.080, 0, 0.190) (rviz TF에서 확인한 값) |
| M1 | 관절 PD + 베이스 적분 (GPU, nworld 병렬) | 수천 월드에서 스텝 시간 측정, CPU 참조와 상태 일치 |
| M2 | depth 광선 투사 + BVH | 같은 장면을 MJWarp로 렌더한 depth와 비교(오차 허용치는 정해야 함) |
| M3 | state/depth 패킹 + CUDA graph | 호스트 복사 0회, 학습 커널이 같은 버퍼를 직접 읽음 |
| M4 | 물체 접촉 | 필요하다고 확인된 뒤에만 |

---

## 4. 열린 문제와 위험
- **용도가 미정**: 시뮬을 (a) 데모 데이터 생성, (b) 폐루프 평가, (c) RL 중 무엇에 쓸지에 따라 접촉 정확도 요구가 크게 다르다.
- **MJWarp를 검증 오라클로 쓰려면 Python이 필요**하다. `CLAUDE.md`는 학습/추론 코드에 Python을 금지하는데, 검증용에도 허용할지 정해야 한다.
- LIMO 바퀴 역학(메카넘/차동 모드)을 생략하므로 실제 주행과 차이가 난다 (sim-to-real).
- URDF의 LIMO 카메라와 손목 카메라 위치는 실측 전 값이다.
- Newton, Madrona, Isaac Lab은 코드를 읽지 않았다. 필요하면 추가로 읽는다.

---

## 5. 팀 프로젝트의 지도 인터페이스 (`refs/code/behavior-2026/src/scene_graph/scenemap/include/scenemap.h`)
시뮬레이션이 내보내는 값과 지도가 받는 값을 같은 형식으로 맞춘다.
- 입력 영상 `sm_image`: `cam`(0 머리, 1 왼손목, 2 오른손목), `rgba` (w×h×4 u8), **`depth_m` (w×h float, 미터)**, `fx, fy, cx, cy`. 우리 depth 버퍼 정의(미터, float)와 같다.
- 입력 proprio: 스텝마다 f32 배열(`n_proprio`). 지금 팀 코드는 R1 Pro용 61개다. 우리 로봇(LIMO + OMX)의 구성은 새로 정해야 한다.
- 지도 출력: 물체 `sm_object` {id, name, pos[3], extent[3], n_obs, last_seen, state(SEEN/GONE/MOVED/HELD)} (map 프레임), 2D 점유 격자 `sm_grid` (int8, −1 모름, 0~100 점유%), 도달 거리 `sm_snap_reachable`. 이것이 VLA state의 "물체/금지 영역" 입력의 원천이다.
- 시뮬에서 지도를 직접 뽑을지(GT)는 Isaac Sim 쪽 작업이다. 팀의 `dynamic-object-mapping-benchmark`(Isaac Sim)가 `src/`에 있다. 내용은 아직 안 읽었다.
