# scenemap — 2D 지도 · 물체 기억 · 장면 그래프 (C++)

로봇의 proprio(R1 Pro 61 f32 또는 LIMO + OMX-F 12 f32 — 아래 "LIMO + OMX-F" 절)와 머리(LIMO: 몸통 앞) 깊이, 검출(마스크 + 이름 번호)로 실시간 기억을 만든다.
2D 점유 격자와 로봇 자세(slam2d), 물체 지도(objmap), 방 나누기, 장면 그래프, 파일 저장, 뷰어 스트림까지 한다.
밖에서는 C ABI 하나(`include/scenemap.h`)로 부른다. CUDA 는 쓰지 않는다(장치 쪽 자르기는 `../runtime`).
설계·측정은 [docs/scenemap_설계.md](../../../docs/scenemap_설계.md).

- 장면 그래프에 기억하는 층은 **물체**(OBJECTS)와 **방**(ROOMS) 두 층이다. 로봇 keyframe 자세(궤적)는 물체 층 번호의 agent partition 에 붙는다.
- 장소(place)는 `sgraph.cpp` 가 2D 격자에서 계산하는 백엔드 값이다. 길 찾기(`sm_snap_place_path`)·이동·탐색이 쓰고, 뷰어는 그리지 않는다.
- 물체끼리 관계(on/in/near)는 만들지 않는다. 위치·상자 메타데이터로 쓰는 쪽이 판단한다. 물체의 부모는 방, place.
- 보통은 `../runtime`(libsgrt)이 검출기와 묶어 부른다.

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/scenemap.h` | C ABI. 판(`sm_create`·`sm_set_labels`·`sm_reset`), 입력(`sm_push_proprio`·`sm_push_image`·`_ex`·`_rgb`·`sm_push_pose`), 스냅숏 질의(물체·이름 찾기·가까운 물체·격자·벽·방·가상 스캔·best view·점 구름·그래프·`sm_snap_reachable`), 저장(`sm_save_dsg`·`_ex`), 스트림(`sm_stream_*`), 자세 원천·격자 정책·단계 시간·이름 종류 |
| `src/capi.cpp` | C ABI 구현. 상태는 뮤텍스 하나 아래, 스냅숏은 그 순간의 자세·상태·격자를 통째로 복사(참조 카운트). 이름 종류 기본 표(구조물·고정), `sm_snap_reachable` = 격자 8방향 A*(점유 ≥ 65 % 칸을 0.30 m 부풀려 막음, 모름은 1.5배 비용) |
| `include/scenemap/fk.hpp` · `src/fk.cpp` · `r1pro_fk_table.hpp` · `limo_omx_fk_table.hpp` | 순기구학(proprio 관절값 + URDF 표) — R1 Pro 와 LIMO + OMX-F. 카메라·팔 끝 자세는 이것으로만 만든다. 표는 `tools/gen_fk_table.py`(R1)·`tools/gen_limo_fk_table.cpp`(LIMO)가 만든다(손으로 고치지 말 것) |
| `scan.*` | 깊이 → 베이스 기준 가상 2D 스캔. 높이 띠 안 점 = 장애물(방위 칸마다 가장 가까운 것), 띠 아래 바닥 점 = 빈 광선 끝. 로봇 몸·팔은 뺀다 |
| `grid.*` | 2D 점유 격자(로그 오즈), 필요하면 넓어짐. 보이는 값(−1 모름, 0..100 %)을 늘 고쳐 두고, 바뀐 영역(dirty)을 추적 |
| `slam2d.*` | `base_qvel` 적분 예측 + keyframe 가상 스캔 맞추기(`'A'` Cartographer 식 / `'B'` point-to-line, 기본 `'B'`) + 격자 넣기 |
| `objmap.*` | 검출 마스크 + 깊이 + 자세 → 물체 3D 위치·크기 → 같은 물체 판단 → 갱신. 확정·옮겨짐·사라짐·들기·받침 따라가기. 이름 종류(옮길 수 있음 / 구조물 — 노드 안 됨 / 고정 가구). 끝에 `../da` 로 중복 병합 |
| `bestview.*` | 물체별 best view: 품질 = 유효 마스크 넓이 × 점수가 가장 큰(같으면 최근) 모습. 상자 + 변마다 10 % 여유, 긴 변 최대 256 px. RGB 자르기는 호출자 함수 또는 호스트 RGBA |
| `cloud.*` | 물체 점 구름: 복셀(기본 0.02 m)마다 점 하나, 물체당 한도(기본 4000), 물체가 움직이면 원점만 옮김 |
| `rooms.*` | 방 나누기(Hydra room finder 의 2D 판): 빈칸 거리 변환 → 지속성 거름·붙이기 → 합치기 → 문(방–방 변). 방 id 유지, 물체 배정(바닥 자리 다수결), 들어 있는 물체로 규칙 이름(kitchen·bedroom …), 외부 이름 덮어쓰기 |
| `sgraph.*` | 살아 있는 장면 그래프: 물체·방 노드, agent(0.5 m·30° 또는 10 s 마다), place 계산(바뀐 격자 둘레 2 m 창만 다시). `publish()` 가 바뀐 때만 읽기 전용 사본을 만든다 |
| `walls.*` | 점유 격자 → 축에 맞는 벽 선분 → 벽 상태 벡터(길이 56: 16방향 거리 + 가까운 선분 8개). `viewer/walls2d.py` 의 C++ 판. sgview 도 이 파일을 컴파일해 쓴다. `wallSegmentsAligned`: slam 지도(지도 좌표 = 출발 자세라 벽이 기울어짐)용 — 벽 방향 θ 를 찾아(`wallAngle`, 투영 히스토그램) 돌린 격자에서 뽑고 되돌림, \|θ\| ≤ 1° 면 `wallSegments` 와 같은 결과(sgview 가 씀, capi 는 그대로) |
| `stream.*` | sgview 실시간 스트림. 스텝 스레드는 링 버퍼에 복사만, 송신 스레드가 비차단 소켓으로 보냄. 프레임 POSE · MAP_RECT · VIEW · JOINTS. 다시 붙으면 전체 상태를 다시 보냄 |
| `dsg_save.*` | 저장(아래 "저장 파일"). 모든 파일은 임시 이름 → rename. scene.json 은 Spark-DSG JSON 형식을 직접 문자열로 쓴다(바뀐 노드만 다시 만드는 캐시) |
| `png.*` | 최소 PNG 쓰기(8 비트 RGB, 16 비트 회색) |
| `timing.hpp` | 단계별 µs 막대그래프(할당·잠금 없음). `sm_get_timing` 이 읽음 |
| `geom.hpp` | 평면 자세·작은 선형대수 |
| `../da/` | 중복 물체 병합. 이 라이브러리에 같이 들어간다([../da/README.md](../da/README.md)) |

## 저장 파일 (`sm_save_dsg` · `sm_save_dsg_ex`)

| 파일 | 내용 |
|---|---|
| `scene.json` | Spark-DSG 형식 장면 그래프: 확정 물체 노드(`'O'<id>`, 이름·위치·상자·상태·movable 메타데이터, best view·점 구름 경로, `sm_set_object_meta` 로 덧붙인 것), 방 노드(`'R'<id>`)·방→물체 변·방–방 문 변, agent 노드 |
| `view.json` | 계획기·뷰어용 요약(자세, 물체 표, 최근 사건, 방·문) |
| `map.pgm` · `map.yaml` | 2D 점유 격자(ROS map_server 형식: 254 빈칸, 0 점유, 205 모름) |
| `rooms.pgm` | 방 칸 그림(방 나눔이 있을 때만) |
| `objects/O<id>_rgb.png` · `_depth.png` · `_mask.png` | best view(RGB 8 비트, 깊이 16 비트 mm, 마스크). 바뀐 것·없는 것만 다시 씀 |
| `objects/O<id>_points.ply` | 물체 점 구름(binary little-endian, x y z float + r g b). 바뀐 것·없는 것만 |

순서는 PNG·PLY → scene.json → view.json(뷰어는 view.json 이 바뀌면 다시 읽는다).

## 만들기

CUDA 없음. C++20, zlib 필요. sgrt 를 빌드하면 `../runtime/CMakeLists.txt` 가 이 폴더를 `add_subdirectory` 로 같이 빌드한다(그쪽은 CUDA 12.8).

```bash
cmake -S src/scene_graph/scenemap -B ~/scenemap_build && cmake --build ~/scenemap_build -j && ctest --test-dir ~/scenemap_build
```

- Spark-DSG: 우리 사본 `../spark_dsg` 가 있으면 그것을, 없으면 `~/.local` 설치본을 찾는다. 있으면 `SM_HAVE_SPARK_DSG` 를 켠다.
  scene.json 은 라이브러리 없이 직접 쓰므로 Spark-DSG 는 `SM_DSG_SAVE=spark`(라이브러리로 쓰기, 비교용)와 시험의 다시 읽기 확인에만 쓰인다.

### 도구 (`tools/`, 같이 빌드됨)

| 도구 | 하는 일 |
|---|---|
| `slam2d_eval <ep.bin> <out prefix> [--method A\|B] [--carto-prior] [--pgm]` | 학습 데모 한 판(`eval/export_episode.py`)을 slam2d 로 재생, 프레임별 추정 자세·keyframe 통계. 채점은 `eval/score_slam.py` |
| `objmap_eval <ep.bin> <det.bin> <out prefix> [--gt-pose] [--min-cells N]` | '완벽한 검출'(`eval/export_gtdet.py`)로 물체 지도를 만들어 물체 표·사건을 씀. 채점은 `eval/score_objmap.py` |
| `capi_replay <ep.bin> <est.bin>` | 같은 판을 C ABI 로 넣고 keyframe 자세를 `slam2d_eval` 결과와 비교, 격자 크기·reachable |
| `map_timeline <ep.bin> <det.bin> <out dir> [--min-cells N]` | 팀 벤치마크 형식 지도 시간표 `map_timeline.csv`, 물체 점 `map_points.npz` |
| `sm_bench <rec.bin> [--pose slam\|odom\|gt] [--lag 0\|1] [--policy 0\|1] …` | sgrt 기록(`SGRT_RECORD`)을 C ABI 로 재생: 자세 모드 비교(떠밀림), 단계별 µs 표. `--robot limo_omx`(또는 `--sm-config '<json>'`)로 LIMO 기록 — 기록에는 로봇이 안 적히므로 sgrt 의 `SGRT_ROBOT` 과 같게 준다(없으면 R1) |
| `stage_bench <rec.bin> [--loops K] [--frames N]` | 같은 기록을 내부 C++ API 로 재생해 fk·scan·match·insert·objmap 단계 µs(옛 판 소스로도 빌드되게 오래된 모양만 씀) |
| `rooms_pgm <memory dir> [출력 dir] [되풀이 수]` | 저장된 기억(map.pgm·map.yaml·view.json)에서 방을 나눠 표, `rooms.pgm`, `rooms_color.ppm`. 되풀이 수를 주면 시간 중앙값 |
| `stream_sim <memory_dir> <host:port> [초] [pose_hz] [map_hz] [view_hz]` | 시뮬 없이 sgview(`--ingest`)에 합성 프레임을 높은 주기로 보내는 부하 시험 |
| `gen_fk_table.py` | `src/sim/integ/fk/r1pro_cam_fk.json` → `include/scenemap/r1pro_fk_table.hpp` |
| `gen_limo_fk_table <map_vla.urdf> <out.hpp>` | LIMO + OMX-F URDF → `include/scenemap/limo_omx_fk_table.hpp`(같이 빌드됨, 아래 LIMO 절) |
| `sgrec.hpp` | sgrt 기록 읽기(`sm_bench`·`stage_bench` 공용) |

### 채점 (`eval/`, 파이썬, 로봇 밖)

정답은 채점에만 쓴다. 학습 데모는 `data/2026-challenge-demos`, 원본 HDF5 는 `data/2026-challenge-rawdata`.

| 파일 | 하는 일 |
|---|---|
| `demo_data.py` | 학습 데모(LeRobot v3) 읽기, 헤드 깊이 역양자화, 640 × 480 판 내부 파라미터 |
| `gt_traj.py` | 원본 HDF5 에서 프레임별 정답 자세(로봇·물체). `python gt_traj.py <에피소드>` 로 요약 |
| `gt_scene.py` | 과제 인스턴스에서 정답 물체 상자, map 프레임으로 옮김 |
| `export_episode.py <에피소드> [--stride 3] [--lag 1] [--out …]` | `slam2d_eval` 입력 `ep_<ep>.bin`(proprio·정답 자세·keyframe 깊이 160 × 120) |
| `export_gtdet.py <에피소드> [--every 9] [--lag 1] [--step 2]` | '완벽한 검출' `ep_<ep>_det.bin`(정답 상자로 깊이 점 라벨) |
| `score_slam.py <ep.bin> <out prefix>[,…] [--npz …]` | 이동 거리별 위치·yaw 오차, keyframe 시간 p50/p99 |
| `score_objmap.py <에피소드> <out prefix> [--lag 1]` | 찾음·상자 거리·중심 거리·중복·헛것·합쳐짐·옮겨짐 |
| `score_map_gt.py <memory_dir> --gt-map <pgm> [--gt-objects …] [--slam-start x y yaw]` | 저장된 기억을 시뮬 정답(world)과 비교: 빈칸 정밀도, 점유 근처 비율, 벽 어긋남, 물체 중심 ↔ 정답 AABB 거리 |

### 환경 변수 (진단용)

| 이름 | 뜻 |
|---|---|
| `SM_DSG_SAVE=spark` | scene.json 을 Spark-DSG 라이브러리로 쓴다(비교용, `SM_HAVE_SPARK_DSG` 빌드에서만) |
| `SM_NO_MERGE` | 있으면 중복 병합(da) 끔 |
| `SM_MERGE_LOG` | 있으면 병합마다 stderr 에 한 줄 |
| `SM_WALLS_CHECK` | 있으면 벽 증분 계산을 처음부터 계산한 것과 비교 |
| `SM_KEEP` | `test_scene_json` 이 출력 폴더를 지우지 않음 |

sgrt 쪽 `SGRT_*` 변수는 [../runtime/README.md](../runtime/README.md).

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `fk` | 순기구학이 시연 robot2cam 자세를 재현 |
| `objmem` | C ABI 만으로 합성 RGB-D + 검출: 구조물·movable, 상자 이상값, 큰 가구 상자 자람 한도, 사라짐, best view, PNG(자체 디코더로 화소 비교)·scene.json 다시 읽기, 점 구름(자리·색·한도·들기·옮겨짐), 시간 |
| `rooms` | 두 방 + 문, ㄱ자, 복도 + 방 셋, 잡음, 크기 다른 방, 이상한 설정, 자람(id 유지), 물체 배정·이름, 외부 이름, 저장, C ABI, 600 × 600 시간 |
| `posemap` | 자세 원천(GT/SLAM/ODOM), 서 있을 때 장애물이 생기고 없어지는 것(사건 기반 넣기), 단계 시간 |
| `relations` | 물체끼리 on/in/near 변이 없음, 물체 부모는 방·place 만 |
| `gt_traj` | GT 자세 모드에서 agent 노드가 `sm_push_pose` 정답 자세를 그대로 따라감(영상·proprio 없이) |
| `scene_json` | scene.json 에 frontier·mesh·건물 층·GVD 필드가 없음, 줄인 Spark-DSG 로 다시 읽힘 |
| `walls` | 합성 격자 속도, 증분 = 처음부터 계산, 물체 자리(소파) 빼기, 49.2° 기울어진 방(축 추출 0 개 → 돌려 뽑기 4 벽, θ 오차 < 0.3°), 축에 맞는 격자에서 돌려 뽑기 = 그대로. 인자로 파이썬 기준(`<cells.bin> <ref.json>`)을 주면 값 비교(ctest 는 인자 없이 돔) |
| `da_merge` | `../da/tests/test_merge.cpp` |
| `stream` | 루프백 TCP 로 프레임 내용, 다시 붙을 때 전체 상태 재전송, 스텝 스레드 비용 |
| `limo_fk` | LIMO 순기구학(깊이·손목 카메라 광학, 팔 끝)이 URDF 독립 계산(`tests/gen_limo_fk_ref.py`, 15 자세)과 위치 1e-5 m·회전 원소 1e-6 안, C ABI `sm_robot_fk` = 내부 값, R1 `sm_robot_fk` 머리 = `T_head` |
| `limo_e2e` | C ABI 만으로 LIMO proprio(odom 원점 ≠ map) + 합성 깊이(벽 둘·바닥·컵) + 컵 마스크: 벽 칸 점유·앞 빈칸·뒤 모름·몸 위 점유 없음, 컵 자리, 0.5 m·14° 주행 뒤 자세(twist 를 일부러 틀려도 오도메트리 자세 차로), 그리퍼 닫기 → 듦 → 따라감 → 놓기(옮겨짐) |

## LIMO + OMX-F

R1 Pro 가 기본이고(옛 동작 그대로 — 아래 회귀 확인), 로봇을 고르면 proprio 형식·순기구학·몸 크기 매개변수가 바뀐다.

```c
sm_ctx* c = sm_create("{\"robot\": \"limo_omx\"}");   // 또는 sm_create(NULL) 뒤 sm_set_robot(c, SM_ROBOT_LIMO_OMX)
// 선택 키: "odom": "pose"(기본) | "twist", "grip_closed": 0.35
```

| 함수 | 뜻 |
|---|---|
| `sm_set_robot(c, SM_ROBOT_R1PRO \| SM_ROBOT_LIMO_OMX)` | 그 로봇 기본 매개변수로 다시 놓고 `sm_reset`(labels·자세 모드·넣기 정책·구름 설정은 그대로) |
| `sm_get_robot(c)` · `sm_proprio_dim(robot)` | 지금 로봇, 최소 `n_proprio`(61 / 12) |
| `sm_robot_fk(robot, proprio, n, &out)` | ctx 없이 순기구학만: 카메라 광학 자세 `T_cam[k]`, 팔 끝 `T_eef[h]`, 그리퍼 값(GPU 근사판 맞추기·시험용) |

**proprio (f32, 12 개 — `SM_LIMO_*`)**

| 번호 | 값 | 쓰는 곳 |
|---|---|---|
| 0, 1, 2 | 바퀴 오도메트리 자세 x, y, yaw(odom 프레임, base_footprint) | 적분(기본): 지난 proprio 와의 자세 차를 지난 베이스 기준으로 → 그 구간 속도. 원점은 상관없음(차만 씀) |
| 3, 4, 5 | 베이스 속도 vx, vy, wz(base_footprint 기준) | `"odom": "twist"` 이거나 첫 표본·간격 이상일 때 적분 |
| 6..10 | omx_joint1..5 | 순기구학(손목 카메라, 팔 끝, 팔 뼈대 — 스캔에서 팔 빼기) |
| 11 | omx_gripper_joint_1(0 닫힘 .. 1.745 다 열림, joint_2 = −이 값) | 잡기 규칙: `grip_closed`(기본 0.35 rad, 손끝 틈 ≈ 5 cm) 아래로 내려가는 순간 팔 끝 0.12 m 안 확정 물체를 듦, 올라가면 놓음 |
| (12..15) | 바퀴 각 fl, fr, rl, rr — 선택 | 뷰어 스트림만 |

- 스트림(sgview `joints`): LIMO 는 뷰어 `robot.json` 의 `joint_order` 순서(omx_joint1..5, gripper_1, gripper_2 = −gripper_1, (바퀴 넷))로 바꿔 보낸다. R1 은 받은 벡터 그대로.
- 지도 자세: R1 과 같이 첫 proprio 의 베이스가 map 원점.

**순기구학** — `tools/gen_limo_fk_table.cpp` 가 `~/ra_ws/map_vla.urdf` 의 `<joint>` 에서 base_footprint → 목표 링크 사슬을 찾아 원 숫자(origin xyz·rpy, axis)를 표로 쓴다. 계산은 R1 과 같은 `walk`(fk.cpp).

```bash
~/scenemap_build/gen_limo_fk_table ~/ra_ws/map_vla.urdf src/scene_graph/scenemap/include/scenemap/limo_omx_fk_table.hpp
python src/scene_graph/scenemap/tests/gen_limo_fk_ref.py ~/ra_ws/map_vla.urdf   # 시험 기준값(numpy 필요, URDF 가 바뀌면 같이)
```

| 프레임 | 링크 | 뜻 |
|---|---|---|
| 베이스 | `base_footprint` | 바닥(z = 0), x 앞, y 왼쪽. 스캔 높이 띠·물체 z 가 이 기준 |
| cam 0 | `depth_camera_lens_optical_frame`(`depth_camera_link` +x 0.010 m 렌즈 + 광학 회전) | 몸통 앞 Orbbec Dabai 렌즈: base_footprint 에서 (0.094, 0, 0.18) m, 수평 앞(OmniGibson `eyes` 와 같은 자리, robot-agent 391c04b). `depth_link` 는 센서 몸체 중심(0.084)이라 쓰지 않는다. 지도(slam2d·objmap)에 쓰는 유일한 깊이 |
| cam 1 | `wrist_cam_optical_frame` | OMX-F link5 메시 안 RGB 카메라(37.9° 아래), 깊이 없음 → 지도에 안 씀(`sm_push_image` 가 cam ≠ 0 은 지나감) |
| 팔 끝 | `omx_end_effector_link` | 잡기 점(손가락 끝 근처). 팔 뼈대 = omx_link0, joint1..5 원점, 팔 끝 |

검증: URDF 독립 계산(xml.etree + numpy 4×4, 15 자세, cam 0 = 렌즈 광학 프레임 — 0·홈·관절 한계 양끝·임의 10)과 최대 차 위치 1.1e-16 m, 회전 원소 3.3e-16(`limo_fk`). RL 환경의 `limo_omx_model.h`(urdf2hdr, f32 상수)와도 위치 1e-16 m·회전 5e-8(f32 반올림) 안.

**몸 크기 매개변수**(R1 값 → LIMO 값, `capi.cpp` `robotParams`)

| 매개변수 | R1 | LIMO | 까닭 |
|---|---|---|---|
| 스캔 `self_r`(베이스 둘레 수평 반경, 이 안 점 버림) | 0.55 | 0.22 | 몸통 0.32 × 0.22 m 반대각선 0.19 + 여유 |
| 스캔 `eef_r` / 팔 캡슐 반경 / 손 캡슐 | 0.35 / 0.09 / 0.10 | 0.08 / 0.05 / 0.06 | OMX 링크 폭 3–4 cm |
| 몸통 캡슐 | 몸통 관절 1..4 → 머리 | 없음 | 몸은 `self_r` 원 |
| 스캔 높이 띠 `band_lo`–`band_hi` | 0.10–1.80 | 0.05–0.50 | 5 cm 턱도 못 넘음, 팔 접은 키 ≈ 0.35 m — 탁자 상판 밑은 지나감 |
| 붙은 것 거르기 반경 | 1.3 | 0.6 | 팔 닿는 거리 ≈ 0.4 m |
| objmap `hand_r` / `grasp_r` / `cloud_hand_r` / `body_r` | 0.40 / 0.25 / 0.10 / 0.30 | 0.10 / 0.12 / 0.05 / 0.22 | 작은 그리퍼·몸 |
| 손 수(`n_hands`) | 2 | 1 | |
| 그리퍼 닫힘 문턱 | 손가락 합 < 0.09 m | gripper_1 < 0.35 rad | 홈 자세는 0(닫힘) — 열었다 닫을 때만 잡기 |

이 값들은 실측 전 추정이다(실제 로봇 기록으로 맞출 것). 나머지(slam2d 맞추기·격자·objmap 확정/사라짐 규칙)는 R1 과 같다.

**R1 회귀**: 바꾸기 전 빌드와 sgrt 기록 3 개(`mem_pose_slam_*`, `mem_pose_gt_move*`) × 자세 모드 3 개(slam·gt·odom)를 `sm_bench --save --traj` 로 재생해 keyframe 자세 CSV·`map.pgm`·`scene.json`·`view.json`·물체 PNG/PLY 가 바이트까지 같음, ctest 기존 10 개 통과.

**아직 R1 전용**: `tools/`(slam2d_eval·objmap_eval·stage_bench·capi_replay·map_timeline 은 R1 61 proprio 기록·`computeBodyFk` 를 씀. sm_bench 는 `--robot` 으로 고름), `eval/`(BEHAVIOR 데모·평가기 형식), `test_fk`·`test_objmem`·`test_posemap` 의 proprio, `sm_object`·이름 종류 표의 "person = 로봇 팔 오검출" 규칙, 스캔의 `shoulder`(캡슐이 있으면 안 씀). sgrt(`../runtime`)는 `SGRT_ROBOT=limo_omx` 로 고른다(그쪽 README "로봇 고르기").
