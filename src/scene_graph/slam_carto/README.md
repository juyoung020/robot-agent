# slam_carto — Cartographer 2D(ROS 없이)를 우리 파이프라인의 SLAM 으로

SLAM 결정은 Cartographer(2D 라이다)다(`docs/model_selection.md` "위치 추정"). 10-06 부터 이것만 쓴다. 시뮬을 먼저 돌리려고 넣었던 scenemap
`slam2d`(깊이 가상 스캔 맞추기)는 `archive/src/scene_graph/scenemap` 으로 옮겼다.

- 코어: Cartographer(Apache-2.0) `third_party/cartographer`, rev 877157a. `tools/build_all.sh cartographer` 가 받아서 `build/cartographer/install` 에
  정적 라이브러리로 설치한다. 의존성(abseil·ceres·protobuf·lua 5.2·eigen·cairo·glog·boost)은 apt 판을 쓴다. 자체 단위 시험 80 개가 통과했다.
- 이 폴더: Cartographer 를 감싼 작은 C ABI(`include/slam_carto.h`, `sc_*`), lua 설정, 도구. libsgrt·realbag_run 이 링크한다.

```
2D 스캔(sc_push_scan) ─┐
바퀴 오도메트리(sc_push_odom) ─┼→ Cartographer 지역 SLAM(스캔 맞추기·submap) + 전역 최적화(되돌아옴)
(IMU, 있으면) ─┘
sc_pose_at(t) = map ← base: 전역 ∘ 지역 맞춤(t 이하 최근 스캔) ∘ 오도메트리 차 — keyframe(영상) 시각마다
   → scenemap sm_push_ext_pose(SM_POSE_EXT) → 격자(mapper2d)·물체 지도가 그 자세로
```

map 원점은 첫 오도메트리 때 베이스다. Cartographer 지역 프레임(첫 스캔 때)과의 차이는 첫 지역 결과에서 한 번 맞춘다. 맞추기 전에는
scenemap 이 오도메트리 적분으로 이어 가므로 원점이 같아야 한다.

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/slam_carto.h` · `src/slam_carto.cpp` | C ABI: `sc_create`(lua·라이다 장착 자세·오도메트리/IMU 켬), `sc_push_scan`·`sc_push_odom`·`sc_push_imu`, `sc_pose_at`·`sc_last_pose`, `sc_get_stats`(노드·submap·되돌아옴 제약·µs), `sc_finish`(마지막 전역 최적화), `sc_nodes`(최적화 궤적), `sc_write_map`(submap 확률 격자 → PGM·yaml) |
| `config/carto_2d.lua` | 공통: 2D, 온라인 상관 맞추기 + Ceres, 움직임 거르기 5 cm·0.2°, submap 45 스캔, 5 cm 칸, 전역 최적화 30 노드마다, 백그라운드 스레드 2 |
| `config/limo_x2l.lua` | 리모 EAI X2L(실제·시뮬 공용): 0.15–8 m, 빠진 광선 3 m |
| `config/openloris_hokuyo.lua` | OpenLORIS Hokuyo UTM-30LX: 0.2–12 m |
| `tools/carto_run.cpp` | 스트림 폴더(`scans.bin` + `odom.csv` + `frames.csv`) → 궤적 `traj.csv`·최적화 노드·`carto_map.pgm`·`metrics.json`(ATE 첫 프레임 맞춤·SE(2), 오도메트리만 대비). GPU·검출 없이 몇 초 — 설정 맞출 때 |
| `tools/carto_vs_slam2d.sh` · `_table.py` | OpenLORIS 7 판을 realbag_run 기본(Cartographer)으로 돌리고(검출 캐시 하나), 옛 slam2d 판이 있으면 같이 표로 낸다 |
| `tools/carto_drift.py` | 떠밀림 통계 → GPU 학습 지도 SLAM 떠밀림 흉내 보정 json(아래) |
| `calib/carto_drift.json` | 그 결과(OpenLORIS 7 판 + 시뮬 LIMO 판) |

## 입력 만들기

- 실제 bag: `tools/realbag/bag2stream.py openloris <bag> <stream> --scan-only` 가 `scans.bin`(SCN1 — 스캔마다 마지막 광선 시각·각도·거리 f32),
  `gt.csv`, `meta.json` 의 `T_bl`(base ← 라이다, TF)·`scan_dt` 를 쓴다.
  `scan_dt` 는 라이다 시계의 어긋남이다. 정답 없이 잰다: 스캔끼리 거리 배열을 각도로 밀어 맞춘 yaw 속도와 바퀴 오도메트리 yaw 속도의
  상호상관이 가장 큰 지연을 쓴다. OpenLORIS 1-6·1-7 은 +0.72·+0.78 s 어긋나 있었다(1-1..1-5 는 −0.06..+0.10 s). 보정하지 않으면
  돌 때마다 yaw 가 6–20° 튄다.
- 시뮬: 시뮬 2D 라이다(`src/sim/lidar/limo_lidar.py`)를 sgrt 글루와 og_replay 가 부르고 `sgrt_push_scan` 으로 넘긴다. 기록(`SGRT_RECORD`)에는
  'L' 레코드로 남는다. `sgrec2stream.py` 가 그 기록을 같은 `scans.bin` 스트림으로 바꾼다.
- 실제 리모: X2L 드라이버의 LaserScan(마지막 광선 시각, `angle_min`·`angle_increment`·`time_increment`)을 그대로 `sgrt_push_scan` 에 넘긴다.
  장착 자세의 기본값은 URDF `laser_link`(base_link (0.103, 0, −0.034))다. 다르면 `SGRT_LASER`.

## 시뮬 라이다(EAI X2L 흉내)

자료표: 0.12–8 m, 360°, 5–8 Hz(권장 6 Hz), 3000 측정/s → 6 Hz 에서 0.72°·500 점, 상대 오차 ≤ 1 m 3 %·1–6 m 3.5 %. PhysX 광선 쏘기로 만든다.
**가정**:
- 자료표 오차를 약 2σ 로 보고 σ = 1.5 %·1.75 % × 거리로 둔다.
- 광선의 1 % 는 빠진다.
- 한 스캔의 광선은 한 순간에 쏜다(`time_inc` 0).
- 로봇 자기 몸에 맞은 광선은 버린다.
- 높이 0.116 m 의 수평 한 면만 본다.
OmniGibson `limo_omx` 자산에 `laser_link` 가 있어 그 자세를 쓴다. 파이썬 비용은 스캔 하나에 약 7 ms(스텝 평균 1.1 ms)다.

## 결과

### OpenLORIS office1-1..7 — realbag_run 기본(Cartographer) 대 옛 slam2d

같은 검출 캐시(ObjectSAM + SigLIP 2 + objprob, conf 0.25, 깊이 4 m)로 돌렸다. 지도 기준은 정답 자세로 만든 지도(gt 판)다.
ATE 는 카메라 xy 기준이다. "첫"은 첫 프레임만 맞춘 것(실시간 떠밀림)이다. yaw 는 `pose_diag` 값이다.

| 판 | 자세 | ATE se2 cm | ATE 첫 cm | yaw ° | 지도 정밀 5/10 cm | 재현 10 cm | 노드 전부 | 중복 쌍 | 작은 것 |
|---|---|---|---|---|---|---|---|---|---|
| 1-1 | slam2d | 4.8 | 9.6 | 2.91 | 0.921 / 0.977 | 0.943 | 110 | 16 | 42 |
| 1-1 | Cartographer | 1.4 | 2.2 | 0.50 | 0.979 / 0.998 | 0.995 | 112 | 27 | 38 |
| 1-2 | slam2d | 9.3 | 13.9 | 1.46 | 0.924 / 0.977 | 0.953 | 119 | 26 | 41 |
| 1-2 | Cartographer | 1.7 | 5.2 | 1.34 | 0.902 / 0.964 | 0.941 | 113 | 22 | 38 |
| 1-3 | slam2d | 0.9 | 4.2 | 1.13 | 0.807 / 0.890 | 0.845 | 60 | 57 | 37 |
| 1-3 | Cartographer | 2.5 | 3.6 | 0.38 | 0.977 / 0.999 | 0.994 | 57 | 31 | 32 |
| 1-4 | slam2d | 9.3 | 21.8 | 4.85 | 0.905 / 0.962 | 0.925 | 168 | 73 | 68 |
| 1-4 | Cartographer | 2.3 | 4.8 | 2.72 | 0.950 / 0.986 | 0.965 | 162 | 77 | 70 |
| 1-5 | slam2d | 8.3 | 17.7 | 1.80 | 0.940 / 0.983 | 0.964 | 126 | 15 | 21 |
| 1-5 | Cartographer | 4.2 | 11.4 | 0.65 | 0.894 / 0.962 | 0.944 | 143 | 27 | 27 |
| 1-6 | slam2d | 3.0 | 7.4 | 1.77 | 0.961 / 0.993 | 0.967 | 96 | 15 | 33 |
| 1-6 | Cartographer | 1.8 | 3.2 | 0.29 | 0.997 / 1.000 | 0.999 | 92 | 9 | 27 |
| 1-7 | slam2d | 7.7 | 11.1 | 2.30 | 0.948 / 0.988 | 0.975 | 95 | 23 | 44 |
| 1-7 | Cartographer | 1.5 | 9.7 | 0.53 | 0.936 / 0.977 | 0.969 | 95 | 17 | 41 |
| **평균** | slam2d | 6.2 | 12.2 | 2.32 | 0.915 / 0.967 | 0.939 | 110.6 | 32.1 | 40.9 |
| **평균** | **Cartographer** | **2.2** | **5.7** | **0.92** | **0.948 / 0.984** | **0.973** | 110.6 | 30.0 | 39.0 |

- 평균은 모든 열에서 Cartographer 가 같거나 낫다.
- 판별로 보면 몇 칸이 더 나쁘다:
  - 1-3 의 ATE se2: 2.5 cm 대 0.9 cm. 0.4 m 만 움직이고 거의 제자리에서 돈 판이다.
  - 1-2·1-5 의 지도 정밀 5 cm: 0.902 대 0.924, 0.894 대 0.940. ATE 는 더 좋은데도 그렇다.
  - 1-5 는 노드·중복도 늘었다: 143 대 126, 27 대 15.
- 정밀 5 cm 가 떨어지는 까닭은 깊이 카메라 점을 라이다 자세로 넣기 때문으로 본다. 라이다(바닥 위 1 m)와 카메라 사이 외부 자세(TF)의 작은
  오차가 그대로 남는다. slam2d 는 깊이를 깊이에 맞춰서 이 오차가 상쇄됐다.
- 맞춰 본 설정(1-2·1-3·1-5·1-1, realbag_run 재생)은 다음 넷이다. 어느 것도 이 칸을 바꾸지 못해 기본값을 그대로 둔다.
  - Ceres 회전 가중 ×10
  - 온라인 상관 맞추기 끔
  - submap 90 스캔
  - 적응 복셀 0.2 m
- carto_run 만으로 맞춘 것: 움직임 거르기를 0.2 m·1° 에서 5 cm·0.2° 로 줄이자 7 판 평균 ATE 가 내려갔다.
- OpenLORIS 에는 물체 정답이 없어 '찾음'은 시뮬에서 잰다(아래).

### 시뮬 LIMO(OmniGibson, turning_on_radio, 2400 스텝 = 80 s, 12.4 m·1008°)

- 판: `run_limo_map.sh`, `SGRT_POSE=carto`, `SGRT_RECORD`.
- 실시간: 정답 대비 keyframe 400 개, rms 2.9 cm / 0.31°, 최대 5.6 cm / 0.89°.
- 지도: 점유의 97.5 % 가 정답 비바닥 ±10 cm 안, 빈칸의 90 % 가 정답 바닥, 정답 벽 경계의 94.7 % 를 덮음.

같은 기록을 두 SLAM 으로 다시 돌린 비교:
- 방법: slam2d 는 archive 전 빌드의 `sgrt_replay`, Cartographer 는 지금 빌드의 `sgrt_replay`로 돌렸다.
- realbag 스트림(`sgrec2stream`, 검출 캐시 하나)으로도 같은 비교를 했다.

| | slam2d | Cartographer |
|---|---|---|
| sgrt_replay 실시간 자세 rms 첫 맞춤 / SE(2) | 1.3 / 0.7 cm, yaw 0.36° | 3.0 / 1.5 cm, yaw 0.31° |
| realbag 카메라 ATE se2 / 첫 맞춤, yaw | 0.6 / 1.9 cm, 0.63° | 1.0 / 1.7 cm, 0.30° |
| 지도 점유 ±10 cm(정답 비바닥) · 빈칸 정밀 | 96.5 % · 88.8 % | 97.4 % · 88.3 % |
| objprob(realbag, `objprob_eval`): 노드 · 찾음 · 중복 · 잘못 합침 | 73 · 19 · 33 · 12 | 79 · 19 · 34 · 13 |

- 시뮬에서는 xy 가 slam2d 쪽이 약간 낫다(1 cm 안팎 차이).
- 다만 조건이 다르다. slam2d 는 잡음 없는 시뮬 깊이를 맞췄고, Cartographer 는 자료표 잡음(σ 1.5–1.75 %)을 넣은 라이다를 쓴다.
- 찾음은 같다(19). 노드는 6 개 많다.

### 떠밀림 보정 json(GPU 학습 지도용) — `calib/carto_drift.json`

`openloris`·`sim`·`all` 세 묶음이 있고, 묶음마다 아래가 들어 있다.
- `runs[]`: 판마다 경로, 첫 keyframe 맞춤 rms·최대·끝(xy·yaw).
- `step_model`: keyframe(0.2 s) 한 걸음 상대 자세 오차의 분산을 맞춘 값. 식은 `c0 + c_d·Δd + c_r·|Δθ|`, 앞뒤·옆·합·yaw 와 치우침을 따로 둔다.
- `rpe_by_dist`: 구간 0.5·1·2·4·8 m 의 상대 오차 rms·90 %.
- `rpe_by_turn`: 45·90·180·360° 의 상대 오차 rms·90 %.
- `growth`: 출발 뒤 거리별 절대 오차 중앙값·90 %.

요약:
- OpenLORIS: xy 분산 6.6e-4·Δd + 7.5e-4·|Δθ| (m²), yaw 분산 4.4e-4·|Δθ| (rad²). 출발 뒤 4–8 m 에서 중앙값 11 cm.
- 시뮬: 걸음마다 거의 상수(xy σ ≈ 2.5 mm, yaw σ ≈ 0.23°). 8 m 넘어서 중앙값 3.6 cm(되돌아옴이 오차를 되돌림).

다시 만들기:
```bash
tools/build_all.sh slam_carto
build/bin/carto_run <stream> <out>        # OpenLORIS 판마다
python3 src/scene_graph/slam_carto/tools/carto_drift.py out.json <out>/traj.csv … <sim run>/poses.csv
```

## 아직

- 실제 리모(X2L) 기록이 없다. 장착 자세·시계 어긋남·CPU(Jetson)를 재야 한다. 이 PC 에서 스캔 넣기는 평균 10–20 µs 이고(맞추기는 Cartographer
  스레드), OpenLORIS 40 Hz 한 판은 실제 시간의 1/10 안팎에 끝난다.
- 판을 넘는 지도(저장·다시 위치 찾기, `pure_localization`)는 아직 쓰지 않는다. 한 판마다 새 궤적이다.
- 되돌아옴으로 바뀐 지난 자세는 이미 넣은 scenemap 격자에 반영되지 않는다. 지금 자세만 고쳐진다(slam2d 때와 같다).
