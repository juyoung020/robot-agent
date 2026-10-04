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
| `objprob.*` | 확률 물체 모델(아래 "scenemap 확률 모드"): vMF 임베딩 사후(r = Σκz)·상위 K 모습·이름 범주 사후(상위어로 올림·엔트로피·바깥 관측)·같은 것 로지스틱 특징·평면 맞춤(PCA)·접촉 칸. `ObjParams::objprob` 일 때만 쓰임 |
| `bestview.*` | 물체별 best view: 품질 = 유효 마스크 넓이 × 점수가 가장 큰(같으면 최근) 모습. 상자 + 변마다 10 % 여유, 긴 변 최대 256 px. RGB 자르기는 호출자 함수 또는 호스트 RGBA |
| `cloud.*` | 물체 점 구름: 복셀(기본 0.02 m)마다 점 하나, 물체당 한도(기본 4000), 물체가 움직이면 원점만 옮김 |
| `rooms.*` | 방 나누기(Hydra room finder 의 2D 판): 빈칸 거리 변환 → 지속성 거름·붙이기 → 합치기 → 문(방–방 변). 방 id 유지, 물체 배정(바닥 자리 다수결), 들어 있는 물체로 규칙 이름(kitchen·bedroom …), 외부 이름 덮어쓰기 |
| `sgraph.*` | 살아 있는 장면 그래프: 물체·방 노드, agent(0.5 m·30° 또는 10 s 마다), place 계산(바뀐 격자 둘레 2 m 창만 다시). `publish()` 가 바뀐 때만 읽기 전용 사본을 만든다 |
| `walls.*` | 점유 격자 → 축에 맞는 벽 선분 → 벽 상태 벡터(길이 56: 16방향 거리 + 가까운 선분 8개). `viewer/walls2d.py` 의 C++ 판. sgview 도 이 파일을 컴파일해 쓴다. `wallSegmentsAligned`: slam 지도(지도 좌표 = 출발 자세라 벽이 기울어짐)용 — 벽 방향 θ 를 찾아(`wallAngle`, 투영 히스토그램) 돌린 격자에서 뽑고 되돌림, \|θ\| ≤ 1° 이거나 θ 로 돌려도 투영 점수가 8 % 미만으로 늘면(gt 지도의 −1°대 잡음) `wallSegments` 와 같은 결과. 돌린 격자에서 뽑은 선분은 원래 격자의 점유 띠에 다시 맞춤(±3 칸 안 가장 진한 띠 가운데로, 4 칸 넘는 끊김에서 나누고 양끝 자름, 점유 < 70 % 버림, 2 칸 안 겹치는 나란한 선 합침) — 돌린 격자 덩어리 가운데 행을 그대로 쓰면 slam 지도에서 선이 벽 옆 빈칸·두 줄 벽 사이에 놓이고 벽 끝을 지나 뻗음. sgview 가 쓰고, capi(정책이 받는 `sm_snap_wall_segments`·`wall_state`)도 10-05 부터 같은 길: θ 를 2 s 마다 재어 1° 넘게 바뀔 때만 바꾸고 \|θ\| > 1° 면 `wallSegmentsAtAngle`(0.5 s 에 한 번까지), 축에 맞으면 예전 증분 추출(`SM_WALLS_AXIS=1` = 옛 동작) |
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
| `dom_bench <seq dir> <pred root> [--min-px 100] [--every 1] [--bench-static]` | dynamic-object-mapping-benchmark 시퀀스(toolkit 배치)를 '완벽한 검출'(정답 인스턴스 마스크 + 범주)로 넣어 `<pred root>/<seq>/map_timeline.csv`·`map_points.npz`. 카메라만 있는 기록은 `sm_set_cam_extrinsic`(베이스 = 카메라 바닥 투영, GT 자세 모드)으로 넣는다(`tools/dom_seq.hpp`, libpng 있을 때만 빌드). 실제 검출판은 `../runtime/tools/dom_bench_det.cpp`(`--dump dets.gz` 로 검출·이름을 남기고 `--load` 로 GPU 없이 scenemap 만 다시 돌림 — 규칙 비교용). `moving` 열 = 든 것, 또는 옮겨짐 상태이고 두 프레임 잇달아 중심이 3 cm/프레임 넘게 옮겨 간 것(지도 출력만으로). 점수는 아래 "물체 바뀜 규칙" |
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
| `SM_SLAM_LOG=<파일>` | keyframe 마다 slam2d 예측·맞춤·정답·점 수 CSV(아래 LIMO SLAM) |
| `SM_OBJ_PARAMS="key=val,…"` | objmap 바뀜 판정 매개변수 덮어쓰기(이름은 `ObjParams` 그대로 — `objmap.cpp` `envOverrides`, 모르는 이름은 stderr 에 알림). A/B 비교용 |
| `SM_OBJ_LOG` | 있으면 objmap 사건(후보·확정·옮겨짐·사라짐·병합 …)과 옮겨짐 잇기(`[link]`)를 stderr 에 |
| `SM_WALLS_AXIS` | 있으면 capi 벽 추출을 예전처럼 축 정렬만(기운 slam 지도에서 벽이 거의 없음 — 비교용) |
| `SM_AP_LOG` | objprob: 물체 쌍 같은 것 확률(`[ap-pair]`, p > 0.2)·병합(`[ap-merge]`)을 stderr 에. 매개변수는 `SM_OBJ_PARAMS` 의 `objprob`·`ap_*`(`objmap.cpp` `envOverrides`) |
| `SM_ABS_LOG=<id>` · `SM_LINK_LOG=<id>` · `SM_MOVE_LOG=<id\|0>` | 물체 하나의 사라짐 근거(보인 표본 수·화소 크기·놓침) / 새 물체 하나의 잇기 후보 / 움직임 따라가기(0 = 전부) |

sgrt 쪽 `SGRT_*` 변수는 [../runtime/README.md](../runtime/README.md).

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `fk` | 순기구학이 시연 robot2cam 자세를 재현 |
| `objmem` | C ABI 만으로 합성 RGB-D + 검출: 구조물·movable, 상자 이상값, 큰 가구 상자 자람 한도, 사라짐(컵 2 s, 큰 소파 4 s), 다른 자리에 나타난 같은 이름을 사라진 물체 id 로 다시 잇기, best view, PNG(자체 디코더로 화소 비교)·scene.json 다시 읽기, 점 구름(자리·색·한도·들기·옮겨짐), 시간 |
| `rooms` | 두 방 + 문, ㄱ자, 복도 + 방 셋, 잡음, 크기 다른 방, 이상한 설정, 자람(id 유지), 물체 배정·이름, 외부 이름, 저장, C ABI, 600 × 600 시간 |
| `posemap` | 자세 원천(GT/SLAM/ODOM), 서 있을 때 장애물이 생기고 없어지는 것(사건 기반 넣기), 단계 시간 |
| `relations` | 물체끼리 on/in/near 변이 없음, 물체 부모는 방·place 만 |
| `gt_traj` | GT 자세 모드에서 agent 노드가 `sm_push_pose` 정답 자세를 그대로 따라감(영상·proprio 없이) |
| `scene_json` | scene.json 에 frontier·mesh·건물 층·GVD 필드가 없음, 줄인 Spark-DSG 로 다시 읽힘 |
| `walls` | 합성 격자 속도, 증분 = 처음부터 계산, 물체 자리(소파) 빼기, 49.2° 기울어진 방(축 추출 0 개 → 돌려 뽑기 4 벽, θ 오차 < 0.3°, 양끝이 벽 가운데선 1 cm 안·벽 밖으로 안 나감), 축에 맞는 격자에서 돌려 뽑기 = 그대로. 인자로 파이썬 기준(`<cells.bin> <ref.json>`)을 주면 값 비교(ctest 는 인자 없이 돔) |
| `da_merge` | `../da/tests/test_merge.cpp` |
| `objprob` | 확률 모드: r 합·μ·‖r‖, 같은 영상 조각은 덜 세지 않고 비슷한 시점은 temper 배, 애매한 이름 → 상위어·엔트로피, 바깥 이름 관측이 뒤 영상 모습에 덮이지 않음, 받침이면 같은 것 아님, 평면 맞춤(세운 얇은 평면·수평면), 접촉 칸 |
| `stream` | 루프백 TCP 로 프레임 내용, 다시 붙을 때 전체 상태 재전송, 스텝 스레드 비용 |
| `limo_fk` | LIMO 순기구학(깊이·손목 카메라 광학, 잡는 점, 팔 끝)이 URDF 독립 계산(`tests/gen_limo_fk_ref.py`, 15 자세)과 위치 1e-5 m·회전 원소 1e-6 안, 잡는 점이 E0 OmniGibson `get_eef_position`(omx_link5 기준 0.08003)과 1e-4 m 안, C ABI `sm_robot_fk` = 내부 값, R1 `sm_robot_fk` 머리 = `T_head` |
| `limo_e2e` | C ABI 만으로 LIMO proprio(odom 원점 ≠ map) + 합성 깊이(벽 둘·바닥·컵) + 컵 마스크: 벽 칸 점유·앞 빈칸·뒤 모름·몸 위 점유 없음, 컵 자리, 0.5 m·14° 주행 뒤 자세(twist 를 일부러 틀려도 오도메트리 자세 차로), 그리퍼를 4 cm 컵 폭(0.41 rad)에서 닫아 멈춤 → 듦 → 따라감 → 놓기(옮겨짐) |
| `limo_held` | LIMO 잡기 규칙·팔 가림(합성 깊이에 순기구학 팔 캡슐을 광선 추적, 검출기가 가린 팔 화소를 탁자 마스크에 넣음): 탁자 앞을 팔이 가리고 빈손으로 닫힘 → 탁자 held·옮겨짐·사라짐 아님, 자리·상자 그대로(대조: 옛 규칙 `SM_OBJ_PARAMS=grasp_check=0,self_mask=0` 이면 held). 4 cm 컵: 열린 채 → 아님, 0 rad(빈손) → 아님, 0.41 rad 멈춤 → held, 열면 놓음. 8 cm 컵 → 아님 |

## 물체 바뀜 규칙 (10-04, dynamic-object-mapping-benchmark 로 고침)

벤치마크 어댑터(`dom_bench`·`dom_bench_det`)로 진단한 실패를 objmap 규칙에서 고쳤다. 모든 값은 `ObjParams` 기본값이라 로봇(sgrt·sm_bench, R1·LIMO)도
같은 규칙을 쓴다(벤치마크 전용 설정 없음). 규칙 요약은 `include/scenemap/objmap.hpp` 머리말.

| 고친 것 | 전 | 후 |
|---|---|---|
| 사라짐 → 옮겨짐 잇기 | 안 맞은 관측이 나오는 순간 같은 이름 '사라짐' 중 가장 가까운 것과 이음(거리·시간 문턱 없음) | `relink`: 새 물체 n 이 확정·3 번 이상 보이고, n 처음 > m 마지막, 거리 ≤ min(8 m, 1 m + 1 m/s·시간 차), n 자리를 처음 검출한 거리 이하에서 5 s 넘게 전에 본 적 있음(처음 가 본 곳에서 찾은 것은 새 물체 — 2D 0.5 m 칸 × 거리 띠 1..5 m 의 처음 본 시각), n 에 더 가까운 같은 이름 물체가 n 이후 안 보였으면 30 s 기다림. 가까운 쌍부터 1:1, n 의 자리·모양·관측을 m 의 id 로 옮기고 n 은 지움. n 은 처음 본 뒤 180 s 까지만 후보 |
| 새 자리를 옛 자리보다 먼저 봄 | 새 id(옮겨짐이 추가 + 사라짐) | 새 자리는 먼저 새 id 로 생기고, 옛 자리가 사라짐이 되면 위 규칙으로 다시 이음(re-association) |
| 큰 것(한 변 > 0.5 m)·고정 종류 사라짐 | 판정 안 함 | 판정함. 놓침 6 번, 첫 놓침에서 4 s 또는 카메라 0.5 m 이동 |
| 사라짐 근거 | 물체 중심 한 점 + 3×3 깊이 | 구름 점(없으면 상자 27 점) 48 개 투영: 시야 안·안 가림 ≥ 50 %, 보이는 부분 ≥ 12 px, 카메라 거리 ≤ 이 물체를 검출한 가장 먼 거리 × 1.15 + 0.2 m. 새 근거만 셈(지난 놓침 뒤 카메라 0.1 m·5° 넘게 바뀜, 또는 자리 너머가 보임 — 서 있는 카메라의 같은 영상을 되풀이해 세지 않음). 물체 상자 안에 그 물체 이름 표에 있던 다른 이름 관측이 있으면 안 셈(이름 흔들림). 필요한 놓침 = max(3, 검출률 p 로 (1−p)^k < 0.02 인 k). 첫 놓침에서 2 s 또는 카메라 0.5 m 이동 |
| 헛검출 | 사라짐으로 남음 | 관측 5 번 미만이던 것이 사라짐이 되면 지움(잇기 후보가 안 되게) |
| 이름 | 관측마다 이름 하나, 같은 이름끼리만 이음 | 물체마다 이름 표(점수 합), 이름 = 최댓값(1.25 배 넘어야 바뀜). 관측 이름이 표에서 20 % 이상이면 짝 후보(이름 다르면 0.02 뒤로). 병합(da)은 이름이 달라도 3D IoU ≥ 0.5 면 합침 |
| 바닥 조각 | 물체로 만듦 | 점의 90 백분위 높이 < 0.05 m(map, 바닥 = 0)면 버림. 러그·카펫·매트(`capi.cpp kFloorLevelNames`)는 둠 |
| 움직이는 중 | 로봇이 든 것만 | 관측 중심이 3 번 잇달아 0.3 m/s 넘게 같은 쪽으로 가고 쉬던 상자를 벗어나면(상자 겹침 < 0.1, 0.25 m 또는 반 폭 넘게) 평균 대신 관측 자리로 따라감. 영상 가장자리에 닿은 관측·카메라가 0.6 rad/s 넘게 도는 keyframe 은 근거로 안 씀 |
| 조각 합치기(`frag_overlap`) | 없음 | 코드는 있으나 기본 끔(실제 검출에서 끈 쪽이 조금 나음) |

**점수**(세 시퀀스 합, class-agnostic 기본 채점, F1. 재현: `docs/map_vla/MAP_STATE_PLAN.md` §6 의 명령 — 로봇 에이전트 저장소)

| 입력 | static | change | moved | removed | added | swapped | dynamic | static 시퀀스 거짓 변화 |
|---|---|---|---|---|---|---|---|---|
| 완벽한 검출, 전 | 0.919 | 0.275 | 0.154 | 0.286 | 0.444 | 0 | 0 | 1 |
| 완벽한 검출, 후 | **0.928** | **0.591** | 0.667 | 0.600 | 0.800 | 0 | **0.431** | 1 |
| FastSAM-s + SigLIP 2 이름, 전 | 0.414 | 0.047 | 0 | 0.102 | 0.083 | 0 | 0 | 186 |
| FastSAM-s + SigLIP 2 이름, 후 | **0.687** | **0.152** | 0.195 | 0.063 | 0.203 | 0 | 0 | 48 |
| (참고) ConceptGraphs | 0.621 | 0.143 | 0 | 0 | | | | |

실제 검출 후: static P/R 0.675/0.699(전 0.299/0.670), 지도 FP(static 시퀀스) phantom 228 → 16, duplicate 99 → 28.
떼어 보기(실제 검출, 10-04 중간판 기준): 바닥 조각 거르기를 끄면 static 0.555·change 0.110, 이름 모으기를 끄면 static 0.688·change 0.156.
`gone_eps = 0.1` 이면 실제 검출 static 0.697·change 0.187 이지만 R1 에서 맞은 물체가 3 개 줄어 기본은 0.02.
swapped 는 여전히 0: 같은 이름 쌍(의자 ↔ 의자, 스탠드 둘 ↔ 모니터 둘)은 생김새 없이 구별이 안 되고, 9–20 m 떨어진 교환은 잇기 거리 8 m 밖.

**R1 회귀**(sgrt 기록 3 개 × 자세 slam·gt·odom, `sm_bench --labels yolo26s-seg 이름` 과 `sgrt_replay` 결과 같음): keyframe 자세 CSV·`map.pgm` 은 바이트까지 같다(objmap 만 바뀜).
물체는 바뀐다. 9 판 합(전 → 후): 확정 물체 275 → 249, 살아 있는 것 254 → 236, 사라짐 21 → 13, 옮겨짐 9 → 22(odom 판의 떠밀린 자리 다시 잇기가 대부분),
정답(`gt.csv.objects.json`, 같은 장면) 비교: 이름이 맞는 짝 49 → 49, 느슨한 짝 14 → 13, 다른 물체 위 59 → 52, 아무 물체에도 없음 132 → 122, 찾은 정답 물체 63 → 62
(빠진 하나는 의자 조각이 'bench' 이름 물체에 병합된 것). objmap 단계 평균 257 → 374 µs/keyframe.

**GPU 근사판(`training/RL/map`)에 옮길 것** — `map.h` 가 지금 따르는 objmap 규칙 중 바뀐 것:
1. 짝짓기(`map.h` 3c, `S.cls != D.cls`): 같은 이름 대신 이름 표 몫 ≥ `name_share` 0.2, 이름 다르면 키 +0.02. 칸에 이름 표(이름 번호별 점수 합)와 표 최댓값 이름(1.25 배 문턱) 필요.
2. 관측 거르기: 점 90 백분위 높이 < `floor_h` 0.05 m 면 버림(러그·카펫·매트 이름 제외).
3. 안 맞은 관측의 옮겨짐 잇기(`map.h` 1148–1160 근처, 사라짐 중 가장 가까운 것과 바로 이음): 없앰. 새 후보로 만들고 `appeared`(자리를 처음 검출한 거리 이하에서 5 s 넘게 전에 본 적 — 2D 0.5 m 칸 × 거리 띠 1..5 m 처음 본 시각) 표시, keyframe 끝에 `relink`(위 표의 조건)로 잇기.
4. 부재 확인(`map.h` 1194–1208): 고정 종류·큰 것도 판정(놓침 6, 4 s), 중심 광선 하나 대신 물체 점 48 개(근사판은 상자 27 점) 투영 — 시야 안·안 가림 ≥ 50 %, 보이는 부분 ≥ 12 px, 거리 ≤ 검출한 가장 먼 거리 × 1.15 + 0.2. 새 근거만(카메라 0.1 m·5° 또는 자리 너머가 보임), 다른 이름 관측 예외, 검출률 k(`gone_eps` 0.02), 시간 대신 카메라 0.5 m 이동도 됨. 칸에 `max_det_z`·`n_vis_miss`·마지막 놓침 카메라 자리·광축·첫 놓침 카메라 자리 필요.
5. 사라짐 때 관측 < 5 인 것은 지움(`spurious_obs`).
6. 움직임 따라가기(위 표). 칸에 마지막 관측 중심·시각·걸음·연속 수·`moving_t` 필요.
7. 병합(da): 이름이 달라도 3D IoU ≥ 0.5 면 합침, 이름 표 합.
8. 값은 그대로: `min_points`·`confirm`·`prune_s`·`moved_d`·`gone_misses` 3·`gone_min_s` 2·`occl`·`da_*`·`big`·`grow_max`·`max_ext`.

## scenemap 확률 모드(`objprob`) — 확률론적 물체 수준 매핑: FastSAM 조각 + SigLIP 2 (10-05, `ObjParams::objprob`, 기본 꺼짐)

용어(stuff·things, PCA 와 RANSAC 등)는 robot-agent `docs/terms.md`.

켜지 않으면(`sm_set_object_model` 을 안 부르면) 위의 이름 기준 규칙 그대로다(radio r3 FastSAM-s-416 검출로 objects·events 바이트 같음 확인).
켜는 길: `sm_set_labels` → `sm_set_text_model`(글 임베딩 줄 → 라벨, SigLIP logit 척도·치우침) → `sm_set_label_stats`(라벨 log 사전·크기 가우스·
상위어·"object") → `sm_set_object_model(c, 1)`. keyframe 마다 `sm_set_det_embeddings`(검출마다 768-d L2) → `sm_push_image_*` →
`sm_reencode_requests`(통째 다시 담기 마스크) → 호출자 SigLIP → `sm_set_object_embeddings`. 진단 셈 `sm_get_objprob_stats`.
예: `tools/realbag/realbag_run --objprob`(검출·SigLIP·다시 담기 모두 호출자).

**흐름(objmap.cpp `update`)**
1. 기하 구조물 거르기(조각마다 RANSAC 평면 맞춤 — 안쪽 문턱 1 cm + 0.25 cm·d², 안쪽 점 비율 ≥ 0.8, 합친 물체는 2 cm·≥ 0.92 일 때만 평면): 얇은 수평면이 천장 높이 위(`ceil_z` 2.0 m 와 정답 없이 잰 천장 − 0.35 m 중 낮은 쪽)면 천장, 바닥 높이면 바닥.
   얇은 세운 평면의 점 60 % 이상이 벽 선분 0.12 m 안이면 벽 선 위 평면: 크면(1 m) 벽, 작으면 조각의 구조물 확률(벽·바닥·천장 …) ≥ 0.5 일 때 벽.
   문·창·계단 확률 ≥ 0.5 인 조각은 지우지 않음. 카메라 → 조각 중심 선이 벽 선분을 지나 0.2 m 넘게 더 가면(창 밖) 버림.
2. 이름 없는 같은 것: 관측 ↔ 물체 P(같음) = 로지스틱(접촉 비율·상자 틈·중심 거리/크기·cos(가장 잘 맞는 μ·모습) − 0.75·상자 겹침·받침),
   가장 큰 물체 ≥ `same_p`(0.6). 여러 조각이 한 물체에 붙음(대표 조각에 상자 합쳐 한 번 갱신). 받침(작은 것이 큰 것 윗면 위)이면 같은 것이
   아님(막기 — 관계로 저장하지 않음). 구조물 사후 ≥ 0.6 인 물체에 구조물 확률 ≤ 0.2 인 조각은 안 붙음(벽이 액자를 삼키지 않게).
   물체끼리는 다음 keyframe 앞에서(쌓인 구름으로) 같은 식 + 이름 분포 겹침, ≥ `merge_p`(0.7) 쌍을 큰 것부터 합침.
   가중치는 `tools/realbag/objprob_fit.py`(시뮬 정답 쌍)가 맞춘 값(objprob.hpp 주석에 문턱 표).
3. 임베딩 = vMF 사후: r = Σ κ_i z_i(조각 r_frag·통째 r_whole 따로), μ = r/‖r‖, 확신 ‖r‖. κ = `viewKappa`(bestview.hpp, 맞춤: ≈ 4369·s/(s + 40),
   s = √마스크 넓이 px — 잘림·4 m 안 깊이는 차이 없었음). 다른 시각의 비슷한 시점(0.3 m·15° 안)은 0.3 배. 합치면 r = r1 + r2(통째도 조각으로) 후
   통째 다시 담기: 구름을 지금 영상에 투영(깊이로 가림 확인)한 마스크(검출 마스크 격자)로 호출자가 SigLIP — 합친 뒤·통째가 없을 때·지금 κ 가
   가장 좋은 통째의 1.3 배 넘을 때, keyframe 당 최대 8.
4. 이름 = 범주 사후: log P(c) = log 사전 + λ Σ w_i log p(c|z_i) + log N(log 크기; μ_c, σ_c) + 바깥 관측 + 기하 우도. p(c|z) = SigLIP 시그모이드
   (라벨마다 낱말 줄 최대)를 라벨 위 정규화, w = κ/3000, Σw > 6 이면 λ 로 눌러 과신 막기. 조각과 통째 우도를 합침(통째만 쓰면 모습 1–2 개라 사후가
   납작했음). 최대 ≥ 0.5 면 그 이름, 아니면 부모 사슬 확률 합이 0.5 넘는 가장 구체적인 상위어, 아니면 "object". 엔트로피 = 이름 불확실성.
   이름은 다시 셀 수 있는 캐시(벡터가 원본).
5. 위치: 작은 물체는 축마다 칼만(관측 잡음 σ = 0.02 + 0.01·깊이, 잘리면 + 반 폭), 큰 것은 상자 합집합 + 분산만. 저장 `pos_sd`.
6. 합친 물체 지우기·내보내기: 구름이 벽 크기 얇은 세운 평면이거나(폭 1.5 m 또는 위 끝 1.9 m·높이 1 m), 천장 띠 안 넓은 덩어리, 벽 선 위 아주 얇은 평면
   (1.2 cm, 납작한 물체·문·창 사후 < 0.5), 구조물 사후 ≥ 0.6(통째 모습 2 개 이상)이면 지움. 노드로는 이름이 정해졌고 벽·바닥·천장 쪽이 아닌 것만
   내보냄(`export_named`). 문·창·계단·난간·기둥은 구조 물체(`SM_KIND_STRUCT_OBJ`): 내보내되 structural 1·movable 0, 벽 추출에서 자리를 안 지움.
7. 바뀜(사라짐·옮겨짐)은 합친 물체 단위: 이번 영상의 어떤 조각이 물체에 닿으면(접촉 ≥ 0.3 또는 P ≥ 0.2) 놓침으로 안 셈. 옮겨짐 잇기는 이름 대신
   μ cos ≥ 0.8.

**저장 형식(확률 모드 물체만, 물체 검색 도구가 읽음)**
- `objects/O<id>_emb.f16`: μ, 768 × FP16 리틀 엔디언, L2 정규화(예전 sgrt 형식과 같음).
- `objects/O<id>_views.f16`: 상위 K(≤ 5) 모습 벡터, K × 768 FP16(각 L2), 순서 = 통째 먼저·κ 큰 순. K 는 metadata `emb.n_views`.
- scene.json 노드 metadata 와 view.json `objects[]` 에 같은 멤버:
  - `emb`: `{"path", "views", "dim": 768, "n_views", "conf": ‖r‖, "kappa_sum": Σκ, "source": "whole"|"frag", "n_whole", "kappa": [K], "whole": [K 0/1]}`
  - `name_post`: `{"top": [[이름, 확률] × 5], "p": 이름 확률(상위어면 그 합), "entropy": nat, "rolled": 상위어로 올렸나, "external": 바깥 관측 있나}`
  - `pos_sd`: `[σx, σy, σz]` m(칼만 분산의 √)
- 바깥 이름 관측: `sm_observe_object_name(c, id, name, log_lr)` — 라벨 `name` 에 로그 우도비를 더해 두고 줄이지 않는다(영상 모습이 더 와도 남음).
  RecallVLA 지도 토큰의 불확실성 칸(‖r‖·엔트로피·pos_sd)은 아직 안 씀(토큰 형식 그대로).

**잰 값(radio r3, LIMO 기록 731 keyframe, 정답 34, `tools/realbag/objprob_eval.py`)** — slam 자세 / gt 자세

| | 확률 모드(FastSAM-s-416 + SigLIP 2) | 옛 규칙: FastSAM-s-416 + 이름 | 옛 규칙: YOLO26s-seg | 옛 규칙: YOLOE-11L |
|---|---|---|---|---|
| 살아 있는 노드 | 122 / 109 | 290 / 274 | 28 / 27 | 78 / 77 |
| 정답 34 중 찾음 | **28** / 27 | 27 / 26 | 14 / 14 | 24 / 23 |
| 크기별: 작은 것·가구 25·벽 장식 8 | 1·22·5 / 1·18·8 | 1·21·5 | 0·14·0 | 1·20·3 |
| 중복(정답 하나 → 노드 여럿, 남는 수) | 59 / 45 | 140 / 133 | 9 / 8 | 32 / 32 |
| 잘못 합침(서로 다른 정답이 한 노드) — 작은 것+가구 / 같은 종류 이웃 | 11(1 / 5) / 9(0 / 3) | 18(2 / 3) | 5(0 / 3) | 8(0 / 3) |
| 벽·천장·바닥 위 헛노드 | 5·1·0 / 4·2·0 | 10·13·2 | 0·0·0 | 1·7·1 |
| 문 6·창 2·계단 1 찾음(맞는 이름) | 2·2·1 / 1·2·1 | 0·0·0 | 0·0·0 | 0·0·0 |
| 이름 정확도 노드 / 정답 | 0.45 / 0.88 | 0.21 / 0.56 | 0.68 / 0.92 | 0.52 / 0.86 |
| 글 질의 R@1(20 종류): μ / 이름 같음 | 0.40 / 0.30 | – / 0.20 | – / 0.15 | – / 0.30 |
| 검출 keyframe 당 GPU ms: 검출 + SigLIP(조각) + 통째 | 0.90 + 4.59 + 0.39 | 0.96 + 4.70 | 1.10 + 0.71 | 4.50 + 1.62 |
| SigLIP 자르기 / keyframe | 21.6 + 0.75 | 21.6 | 1.2 | 5.6 |
| 프로세스 GPU 최대 | 662 MiB | 662 MiB | 640 MiB | 890 MiB |

- YOLO26s-seg(옛 규칙)보다 못한 곳: 중복(59 vs 9 — 소파·hall tree·커피 탁자 위 것들·같은 이름 조각), 잘못 합침(식탁 의자 묶음·주방 줄), 노드 이름 정확도(0.45 vs 0.68),
  문 조각의 물체 이름(curtain·bag — 문 6 개 중 2 개만 door). 중복 셈은 중심이 정답 상자 안인 노드를 다 세므로 소파 위 쿠션·탁자 위 물건도 들어간다.
- 평면 맞춤 PCA → RANSAC(같은 검출 캐시, slam / gt 자세): 찾음 28 → 28 / 27 → 27, 벽·천장·바닥 위 헛노드 5·1·0 → 4·1·1 / 4·2·0 → 4·2·1,
  문·창·계단 위 헛노드 9·4·5 → 10·3·4, 문 6·창 2·계단 1 찾음 그대로, 중복 59 → 62 / 45 → 53, 잘못 합침 11 → 9 / 9 → 9, keyframe 당 CPU 차이 없음(약 30 ms, 잡음 안),
  OpenLORIS office1-1·1-5 노드 120 → 120·107 → 111. 남은 천장 쪽 헛것은 평면 맞춤 실패가 아니라(평면이 아닌 덩어리·작은 것) 줄지 않음.
- 구조물 후처리(10-05): 문·창·계단·기둥 이름 보호는 그 모양 크기 안일 때만(문·창 수평 폭 ≤ 2.2 m·두께 ≤ 0.45 m·높이 ≤ 2.8 m, 기둥 ≤ 1 m,
  계단 ≤ 6 m — `ApParams::so_*`), 크기 밖이고 벽 모양(평면·벽 선 근처·세운 평면 안쪽 ≥ 0.5·한도 1.5 배 넘음)이면 숨김(`ApState::hide` — 지우면 그 자리
  벽 조각이 새 작은 문·창이 되어 다시 남아서, 기억에 두고 노드로만 안 냄). 벽 선 근처 얇은 세운 평면 + 구조물 확률 ≥ 0.5 조각은 구조물 아닌 물체에
  안 붙고 버림(`struct_look_block`), 납작한 벽걸이 이름(액자·TV …)은 상자 폭 2.2 m 넘게 안 키움(`flat_max_w`), 벽 선 없이 높고 넓은 세운 평면은 벽(`tall_*`).
  radio r3 slam / gt(RANSAC 판 → 이 판): 노드 125 → 108 / 115 → 109, 찾음 28 → 28 / 27 → 28, 벽·천장·바닥 헛노드 4·1·1 → 3·1·1 / 4·2·1 → 4·2·1,
  문·창·계단 위 헛노드 10·3·4 → 7·2·4 / 9·4·5 → 11·3·5, 문 6·창 2·계단 1 찾음 2·2·1 → 1·1·1 / 1·2·1 → 1·2·1, 중복 62 → 57 / 53 → 46,
  잘못 합침 9 → 8 / 9 → 9, 문·창 이름 노드 7·17 → 2·11, 액자·TV 노드 가장 큰 폭 3.1 → 2.9 / 3.0 → 1.9 m(정답 최대 1.4), CPU 그대로.
  OpenLORIS office1-1·1-5 노드 120·111 → 121·110. 벽 선 없는 벽 규칙은 r3 에서 1 번만 맞음. 노드 상자는 관측 상자 합집합이라 구름보다 클 수 있음.
- scenemap 확률 모드 CPU: gt 판 731 keyframe 18.2 s vs 옛 규칙 11.9 s(keyframe 당 약 +8.6 ms, 통째 다시 담기 GPU 포함).

## LIMO + OMX-F

R1 Pro 가 기본이고(옛 동작 그대로 — 아래 회귀 확인), 로봇을 고르면 proprio 형식·순기구학·몸 크기 매개변수가 바뀐다.

```c
sm_ctx* c = sm_create("{\"robot\": \"limo_omx\"}");   // 또는 sm_create(NULL) 뒤 sm_set_robot(c, SM_ROBOT_LIMO_OMX)
// 선택 키: "odom": "pose"(기본) | "twist", "grip_closed": 0.6
```

| 함수 | 뜻 |
|---|---|
| `sm_set_robot(c, SM_ROBOT_R1PRO \| SM_ROBOT_LIMO_OMX)` | 그 로봇 기본 매개변수로 다시 놓고 `sm_reset`(labels·자세 모드·넣기 정책·구름 설정은 그대로) |
| `sm_get_robot(c)` · `sm_proprio_dim(robot)` | 지금 로봇, 최소 `n_proprio`(61 / 12) |
| `sm_robot_fk(robot, proprio, n, &out)` | ctx 없이 순기구학만: 카메라 광학 자세 `T_cam[k]`, 잡는 점 `T_eef[h]`(LIMO `grasp_point`), 그리퍼 값(GPU 근사판 맞추기·시험용) |

**proprio (f32, 12 개 — `SM_LIMO_*`)**

| 번호 | 값 | 쓰는 곳 |
|---|---|---|
| 0, 1, 2 | 바퀴 오도메트리 자세 x, y, yaw(odom 프레임, base_footprint) | 적분(기본): 지난 proprio 와의 자세 차를 지난 베이스 기준으로 → 그 구간 속도. 원점은 상관없음(차만 씀) |
| 3, 4, 5 | 베이스 속도 vx, vy, wz(base_footprint 기준) | `"odom": "twist"` 이거나 첫 표본·간격 이상일 때 적분 |
| 6..10 | omx_joint1..5 | 순기구학(손목 카메라, 팔 끝, 팔 뼈대 — 스캔에서 팔 빼기) |
| 11 | omx_gripper_joint_1(0 닫힘 .. 1.745 다 열림, joint_2 = −이 값) | 잡기 규칙(아래 "잡기 확인"): `grip_closed`(기본 0.6 rad) 아래에서 멈추면 잡는 점 0.12 m 안 들 수 있는 확정 물체를 듦, 올라가면 놓음 |
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
| 잡는 점 | `grasp_point`(omx_link5 x 0.08003) | `T_eef`·잡기 규칙. E0 실측(robot-agent 66fe1ee, `docs/map_vla/CURRICULUM_BEHAVIOR2026.md` 5.3): 물체가 실제로 쥐이는 자리 = OmniGibson `get_eef_position`, `omx_end_effector_link` 에서 손가락 축으로 0.0119 m 뒤. URDF(map_vla_description xacro)에 프레임으로 넣음 |
| 팔 끝 | `omx_end_effector_link` | `T_tip`(내부). 팔 뼈대 = omx_link0, joint1..5 원점, 팔 끝 — 스캔·objmap 팔 가리기 캡슐 |

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
| 그리퍼 닫힘 문턱 | 손가락 합 < 0.09 m | gripper_1 < 0.6 rad(틈 ≈ 6.6 cm) | 홈 자세는 0(닫힘) — 열었다 닫을 때만 잡기 |
| 잡기 확인(`grasp_check`) · objmap 팔 거르기 | 끔 | 켬 | 아래 |

**잡기 확인·팔 가림(10-04, LIMO 만 — R1 은 옛 규칙 그대로)**

원인: 시뮬 한 판(turning_on_radio, 대역 정책이 탁자를 "집기")에서 팔이 몸통 카메라 앞을 가리자 검출기(yolo26s-seg)가 가린 팔 화소를 탁자 마스크에
넣어 탁자(1.2 m, 고정 종류) 상자가 팔 쪽으로 자랐고(큰 물체 위치 = 상자 중심), 그리퍼가 빈손으로 끝까지(0 rad) 닫히는 순간 옛 잡기 규칙 —
"닫히는 순간 팔 끝 `grasp_r` 안 가장 가까운 확정 물체" — 이 크기·종류·그리퍼 벌림을 안 보고 탁자를 `held` 로 들어 0.49 m 옮겼다.

- 팔 거르기: 물체 관측 점 중 순기구학 팔 캡슐(스캔 몸 가리기와 같은 것, map 으로 옮김) 반경 + `self_pad` 0.01 m 안 점은 검출 마스크 안이어도 버린다.
  팔 화소가 물체 상자·자리·구름을 바꾸지 않는다(옮겨짐·사라짐 근거도 안 됨 — 가린 자리는 깊이가 더 가까워 '가림'으로 셈).
- 잡기: 그리퍼가 `grip_closed` 아래에서 0.2 s 동안 0.01 rad 안으로 멈춘 뒤 한 번 고른다. 큰 것(한 변 > 0.5 m)·고정 종류·지도 상자 가운데 변 > 0.06 m
  는 못 든다. 손끝 틈(그리퍼 각 → 틈 표: E0 쥔 각도 1·2·3·4 cm = 0.095·0.231·0.347·0.408 rad, 그 위 `finger_gap_hull` link5 x 0.08 틈)이
  5 mm 넘고(끝까지 닫힘 = 빈손), 가장 좁은 변 − 2.5 cm ≤ 틈 ≤ 가장 넓은 변 + 2.5 cm 이어야 든다. 든 뒤 끝까지 닫히면 놓친 것으로 놓는다.

이 값들은 실측 전 추정이다(실제 로봇 기록으로 맞출 것). 나머지(격자·objmap 확정/사라짐 규칙)는 R1 과 같고, slam2d 맞추기 가중·받기 문턱만 아래처럼 다르다.

**LIMO SLAM(slam2d 맞추기, 10-04)**

원인(시뮬 LIMO 기록 4 개 재생, `SM_SLAM_LOG` 로 keyframe 마다 예측·맞춘 결과·정답): 원래 가중(스캔 점 σ 1 cm, 사전항 2 cm + 10 %·이동)이면
스캔 점 100–700 개의 정보가 오도메트리 사전항을 수천 배 눌러 예측이 사실상 무시된다. 좁은 시야(67.9°)에 벽 하나·복도만 보이거나 지도가 덜 찬
keyframe(제자리 회전·출발 회전)에서 맞추기가 벽을 따라 미끄러지거나(복도 퇴화) 덜 찬 지도에 끌려 keyframe 하나에 4–10 cm 씩 튀었고
(받기 문턱 8 cm·4σ 안이라 받아짐 — keyframe 사이 오도메트리는 1 mm·0.1° 수준인데), 그 자세로 넣은 지도(이중 벽)에 뒤 스캔이 다시 맞아
오차가 고정 어긋남으로 굳었다. 맞추기는 평균으로 오차를 줄이지도 못했다(맞춘 뒤 정답에 가까워진 keyframe 43–51 %).
높이 띠(맞추기 점을 0.5 m 아래로만 — 오히려 나빠짐)·첫 대응 반경(0.20 → 0.10 m — 나빠짐)·인라이어 하한은 원인이 아니었다.

고침(`capi.cpp` `robotParams`, LIMO 만): 점 σ `sigma_r` 1 → 10 cm(점끼리 상관 — 스캔 정보 정규화), 사전항을 오도메트리 오차 크기로
(`prior_xy0` 5 mm, `prior_xy_k` 3 %, `prior_yaw0` 0.2°, `prior_yaw_k` 3 %), 받기 문턱 `gate_xy` 3 cm·`gate_yaw` 1.5°(또는 사전항 4σ).
퇴화 방향은 오도메트리를 따르고 잘 잡히는 방향(벽 법선·yaw — 회전 치우침 1.2 % 를 고침)만 스캔이 고친다. 코드 경로는 그대로라 R1 은 바이트까지 같다.

| 기록(`sm_bench --robot limo_omx --no-dets`) | 전 RMS / 최대 / 끝 cm | 뒤 RMS / 최대 / 끝 cm | keyframe 튐 최대(전 → 뒤) cm | 점유 ±5 cm(전 → 뒤, gt 자세) | 벽 선분 ±5 cm(전 → 뒤, gt 자세) |
|---|---|---|---|---|---|
| 제자리 360° 14 s (turning_on_radio) | 1.47 / 2.81 / 0.45 | 0.70 / 1.24 / 0.34 | 0.95 → 0.24 | 95.3 → 95.5 % (95.1) | 95.2 → 96.5 % (96.6) |
| explore 104 s (turning_on_radio) | 8.30 / 14.07 / 2.77 | 1.45 / 2.75 / 2.39 | 7.28 → 0.84 | 93.2 → 96.4 % (96.1) | 91.1 → 96.1 % (97.0) |
| explore 121 s (bringing_water) | 12.25 / 26.33 / 4.57 | 2.68 / 4.93 / 3.25 | 11.06 → 0.76 | 89.5 → 96.5 % (97.4) | 93.6 → 98.5 % (94.0) |
| explore 146 s (turning_on_radio, 고칠 때 안 씀) | 9.68 / 14.14 / 12.16 | 2.19 / 3.70 / 2.53 | 7.37 → 0.94 | 90.5 → 97.5 % (98.4) | 86.4 → 97.4 % (98.4) |

오차 = keyframe 자세와 정답(첫 keyframe 에서 맞춤)의 거리, 튐 = 이웃 keyframe 사이 오차 벡터 변화. 점유 ±5 cm = 점유 칸(map.pgm 0) 중심이
정답 지나갈 수 없는 곳(`gt_trav.py` .pgm, 1 cm 로 늘린 거리 변환)에서 5 cm 안인 비율, 벽 선분 = 저장 지도에 `wallSegmentsAligned` 를 돌린 선분을 1 cm 마다
같은 거리로. 오도메트리만(odom)은 RMS 2.7–23 cm(회전 치우침으로 yaw 3°대) — 맞추기는 꼭 필요하다.

진단: `SM_SLAM_LOG=<파일>`(sm_reset 때 열림)이면 keyframe 마다 한 줄 — stamp, 예측·맞춘 결과·쓴 자세, 정답(맞춤), 스캔 점 수, 인라이어, 맞춤·받음·제자리, 적분 이동.

**R1 회귀**(LIMO SLAM 고침 10-04 에도 다시: sm_bench 9 개 저장·자세 CSV, `sgrt_replay` 9 개 저장 디렉터리 전부 바이트 같음, ctest 12 개 통과): 바꾸기 전 빌드와 sgrt 기록 3 개(`mem_pose_slam_*`, `mem_pose_gt_move*`) × 자세 모드 3 개(slam·gt·odom)를 `sm_bench --save --traj` 로 재생해 keyframe 자세 CSV·`map.pgm`·`scene.json`·`view.json`·물체 PNG/PLY 가 바이트까지 같음, ctest 기존 10 개 통과.

**아직 R1 전용**: `tools/`(slam2d_eval·objmap_eval·stage_bench·capi_replay·map_timeline 은 R1 61 proprio 기록·`computeBodyFk` 를 씀. sm_bench 는 `--robot` 으로 고름), `eval/`(BEHAVIOR 데모·평가기 형식), `test_fk`·`test_objmem`·`test_posemap` 의 proprio, `sm_object`·이름 종류 표의 "person = 로봇 팔 오검출" 규칙, 스캔의 `shoulder`(캡슐이 있으면 안 씀). sgrt(`../runtime`)는 `SGRT_ROBOT=limo_omx` 로 고른다(그쪽 README "로봇 고르기").
