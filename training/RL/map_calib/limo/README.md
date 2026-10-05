# map 잡음 값 — LIMO 기록으로 다시 맞추기

`../map/include/map.h` 의 `MP` 값을 **LIMO(limo_omx) 시뮬 기록**으로 다시 맞춘 제안이다. 값은 `calib_limo.json` 에 있다. R1 보정(`../README.md`, `../calib.json`)은 그대로 둔다. 이 폴더는 map 코드를 바꾸지 않는다(적용은 map 담당이 한다).

**단서 두 가지**
1. **기록이 하나뿐이다.** SLAM 잔차는 탐색 판 하나(`explore_20261004_094010_turning_on_radio_frontier_limo`)로 맞췄다. 원 오도메트리와 검출 하나하나는 14 초짜리 지도 시험 판 하나(scratchpad `limo3`)에서 쟀다. 표본 분산은 모른다.
2. **카메라 FK 1 cm 오차가 들어 있다.** 세 LIMO 판 모두 커밋 391c04b(eyes 를 렌즈 위치 +x 0.010 m 로 옮김) 전에 찍었다. 그래서 SLAM 잔차 목표(rms 3.1 cm)에 이 오차가 섞여 있다. 고친 카메라로 다시 재면 `kf_corr`·`odo_*` 잡음이 조금 더 작아질 수 있다.

## 새 LIMO SLAM(behavior-2026 84b373c)으로 다시 맞춤 — 지금 `map.h` 기본값 (2026-10-04, 잰 값)

scenemap slam2d 가 LIMO 에서 오도메트리 가중 맞추기로 바뀌었다(점 σ 10 cm, 사전항 5 mm + 3 %·0.2° + 3 %, 받기 문턱 3 cm·1.5°). 옛 목표(아래 "요약" — 둘째 탐색 판 rms 9.9 cm, keyframe 튐 4–10 cm)는 더 이상 실제 SLAM 이 아니다.
**목표** = 새 SLAM 으로 다시 재생한 기록 넷(`~/datasets/limo_rec`, rec.bin): `sm_bench <r>.bin --robot limo_omx --pose slam --no-dets --traj`(서브모듈 d58c978 소스를 CPU 빌드), keyframe 자세 − 외부 GT(첫 keyframe 에서 맞춤).

| 기록 | 시간 | 길이 | 회전 | rms xy / yaw | max xy / yaw | keyframe 사이 오차 튐 최대 |
|---|---|---|---|---|---|---|
| r1 (turning_on_radio) | 103.4 s | 24.96 m | 1,788° | 1.45 cm / 0.365° | 2.75 cm / 1.847° | 0.84 cm |
| r2 (bringing_water) | 120.8 s | 30.36 m | 2,198° | 2.68 / 0.376 | 4.93 / 1.527 | 0.76 |
| r3 (turning_on_radio) | 145.8 s | 37.87 m | 2,663° | 2.19 / 0.242 | 3.70 / 0.891 | 0.94 |
| r4live (turning_on_radio, 실시간) | 123.2 s | 27.36 m | 2,139° | 1.72 / 0.254 | 3.60 / 1.112 | 0.21 |
| **평균(목표)** | | | | **2.01 / 0.309** | **3.75 / 1.344** | 0.69(넷 모두 < 1 cm) |

**모형**: `map_drift` 에 kind 5(= kind 2 동작, 길이 합 30.1 m — 기록 넷 비슷)를 더했고, CPU 판에 걸음마다 오차 벡터 튐 최대를 더했다. 치우침(`odo_bt/br/bw`)은 limo3 원 오도메트리 값 그대로다(SLAM 이 바뀌어도 오도메트리는 같음). yaw(`odo_rr` 0.04·`odo_rt` 0.02·`kf_corr_yaw` 0.15)는 이미 목표와 맞아(0.308° / 1.30°) 그대로 두고, xy 두 값만 격자(CPU 참조판 128 판, kind 5)로 찾았다.

| odo_t \ kf_corr_xy | 0.0015 | 0.002 | 0.003 | 0.01 | 0.03 |
|---|---|---|---|---|---|
| 0.005 | 2.07 / 3.63 | 1.96 / 3.49 | 1.80 / 3.30 | | |
| 0.01 | 2.18 / 3.87 | 2.07 / 3.72 | 1.89 / 3.49 | 1.38 / 2.84 | 0.96 / 2.14 |
| **0.015** | 2.36 / 4.21 | 2.23 / 4.03 | **2.03 / 3.78** | | |
| 0.03 | | 2.90 / 5.34 | | 1.80 / 3.85 | 1.20 / 2.92 |

(칸 = rms / max xy cm. 옛 기본 odo_t 0.10·kf_corr_xy 0.001 은 8.38 / 15.22.) **고른 값 `odo_t` 0.10 → 0.015, `kf_corr_xy` 0.001 → 0.003** (`map.h` `ODO_T_V`·`KF_CORR_XY_V`):

| kind 5 | rms xy / yaw | max xy / yaw | 걸음 튐 최대(판 평균, CPU) |
|---|---|---|---|
| 옛 값(CPU 256 판) | 8.58 cm / 0.288° | 15.43 cm / 1.212° | 1.32 cm |
| **새 값(GPU 1,024 판)** | **1.96 cm / 0.307°** | **3.69 cm / 1.260°** | 0.22 cm(CPU 256 판) |
| 목표(넷 평균) | 2.01 / 0.309 | 3.75 / 1.344 | < 1 cm(0.21–0.94) |

- 다른 궤적(GPU 1,024 판, 새 값): kind 4(38.7 m) 2.00 cm / 0.308°, max 3.89 / 1.301°; kind 2(13.4 m) 1.81 / 0.300°, 3.11 / 1.071°; kind 3(4.0 m) 0.89 / 0.298°, 1.83 / 0.892°. 실제 기록처럼 길이(25–38 m)에 따라 크게 자라지 않는다.
- 단서: 실제 오차는 keyframe 사이 튐이 작고(최대 0.21–0.94 cm) 천천히 변하는 치우침이다. 모형은 걸음 잡음(odo_t)이 작고 keyframe 되돌림이 조금 커서 같은 rms·max 를 낸다. 실제 기록은 모두 같은 두 집(turning_on_radio·bringing_water)이다.
- 재현: `sm_bench` 4 번(CPU, 기록마다 약 30 s), `g++ … -DDRIFT_CPU_ONLY -DODO_T_V=… -DKF_CORR_XY_V=… tools/map_drift.cpp src/map_ref.cpp` → `drift 5 128`, GPU 확인 `map_drift 5 1024`.

## 요약 (옛 SLAM 기준 — 기록)

| MP | R1 → **LIMO** | 근거 (기록, n, 방법) |
|---|---|---|
| `odo_bt` | 0.030 → **0.0082** | limo3 rec.bin 원 오도메트리 대 GT. 직진 1 구간 0.80 m 에서 거리 +0.82 % |
| `odo_br` | 0.13 → **0.0121** | 같은 기록. 제자리 회전 1 번(363°, keyframe 52 개)에서 +1.21 %/rad. 직선 맞춤의 잔차는 σ 3e-5 rad 로, 잡음 없이 치우침만 있다 |
| `odo_bw` | 0.031 → **0** | 같은 직진 구간. yaw 변화는 오도메트리 −4e-5, GT −1e-5 rad 로 5e-5 rad/m 보다 작다 |
| `odo_t` | 0.15 → **0.06** | 탐색 판 SLAM 잔차에 맞춘 **유효** 값(아래 2). 원 오도메트리에는 걸음 잡음이 없다 |
| `odo_rr` | 0.035 → **0.005** | 같음(격자 아래 끝. 0.005 와 0.01 은 차이가 없다) |
| `odo_rt` | 0.03 → **0.02** | 같음 |
| `kf_corr_xy` | 0.025 → **0.0075** | 같음 |
| `kf_corr_yaw` | 0.7 → **0.15** | 같음 |
| `p_conf` | 0.07 → **0.08** | 탐색 판 확정 물체에서 틀린 이름 3/16 = 0.19. map_drift 에서 0.08 이면 0.21 이 나온다. (COCO 검출 하나하나로는 0.24, n 132) |
| `p_ghost` (`n_ghost` 3 그대로) | 1.0 → **0.03** | limo3 에서 가짜 1/133 검출, 가짜가 있는 keyframe 1/80 = 0.013. 탐색 판 확정 물체의 가짜 0/16. 0.03 이면 keyframe 0.012, 확정 가짜 0.04 |
| `lat_n` | 0.04 → **0.023** | limo3 의 맞는 검출 중심, 강건 σ(x/y), n 100(물체 4 개). 높이 0.007 |
| `p_miss_near/mid/far` | 0.15/0.10/0.79 그대로 | 맞추지 못함(아래 3.1) |
| `ext_n`, `dn0/dn2`, `zmax` 등 | 그대로 | 재지 않음. Dabai 값은 로봇과 무관 |

### map_drift 결과 (GPU, 1,024 판의 판별 평균, 판별 rms·max)

| | 길이·회전·시간 | rms xy | rms yaw | max xy | max yaw | 가짜 있는 kf | 확정: 맞음/틀린 이름/가짜 |
|---|---|---|---|---|---|---|---|
| **실제 LIMO 탐색** | 13.17 m, 879°, 58 s | **3.05 cm** | **0.256°** | **5.34 cm** | **0.662°** | 0.013 (limo3) | 0.81 / 0.19 / 0 (n 16) |
| kind 2, R1 값 | 13.40 m, 760°, 55.5 s | 5.25 | 0.792 | 12.26 | 4.03 | 0.350 | 0.64 / 0.14 / 0.22 |
| **kind 2, LIMO 값** | 같음 | **2.82** | **0.233** | **5.45** | **0.680** | 0.012 | 0.75 / 0.21 / 0.04 |
| 실제 limo4 (40 s 지도 시험) | 5.02 m, 662°, 40 s | 1.32 | 0.206 | 2.72 | 0.372 | | |
| kind 3, R1 값 | 5.07 m, 728°, 30 s | 8.13 | 0.746 | 22.25 | 4.76 | | |
| kind 3, LIMO 값 | 같음 | 4.71 | 0.312 | 8.74 | 1.53 | | |

- 맞춘 대상인 kind 2 에서 rms 는 −8 %(xy)·−9 %(yaw), max 는 +2 %(xy)·+3 %(yaw) 차이다. R1 값으로는 xy 가 1.7–2.3 배, yaw 가 3–6 배 컸다.
- 확인용 kind 3(limo4 비슷: 제자리 한 바퀴 뒤 5 m)은 실제보다 xy 가 3.6 배, max yaw 가 4 배 크다. kind 3 는 G1 방 **가운데**에서 출발한다. 방 반치수가 최대 3.5 m 라 Dabai 3 m 안에 벽이 없을 때가 있고, 그동안 맞추기(맞은 열 ≥ `min_hits`)가 안 되어 오차가 쌓인다(R1 README 의 max 설명과 같은 원인). 실제 집에서는 limo4 판 내내 벽이 3 m 안에 있었다. 이 차이는 궤적·장면 탓이라 값으로 맞추지 않았다.

## 쓴 기록 (읽기만 함)

| 기록 | 내용 | 쓴 것 |
|---|---|---|
| `~/behavior-2026-limo/outputs/explore_20261004_094010_turning_on_radio_frontier_limo` | 첫 LIMO 탐색 판(frontier, SLAM 자세, 열린 어휘 검출 272·과제 이름 20). **rec.bin 없음** | `poses.csv`(keyframe 291 개, GT·SLAM 자세), `pose_diag.json`, `gt_poses.csv.objects.json`(GT 물체 244 개, AABB), `frame.json`(map_from_world), `memory/scene.json`(확정 물체 16 개) |
| scratchpad `limo3` (같은 장면·같은 시작 자세, 14 s) | 제자리 363° 회전 + 0.8 m 직진, COCO-80 검출, **rec.bin 있음** | 원 오도메트리(sm_bench 재생), 검출 하나하나, 확정 물체 14 개 |
| scratchpad `limo2` | limo3 와 같은 대본 | 오도메트리가 limo3 와 비트 단위로 같아 독립 표본이 아님 |
| scratchpad `limo4` (40 s, 5 m) | COCO-80, rec.bin 없음 | `summary.json` pose_diag(확인용 목표), 확정 물체 19 개 |

탐색 판에는 `base_qvel` 도 rec.bin 도 남지 않는다(sim.log 에는 proprio 배치와 t=900 의 pose diag 한 줄만 있다). 그래서 원 오도메트리와 검출 하나하나는 limo3 에서 쟀다. 세 판 모두 장면(turning_on_radio)과 시작 자세(4.902, 3.832, 0.858)가 같아서 탐색 판의 GT 물체 목록을 limo3·limo4 채점에도 썼다.

## 1. 오도메트리 치우침 (limo3, 원 오도메트리 vs GT)
`sm_bench <limo3>/rec.bin --robot limo_omx --pose odom --no-dets --traj`(`~/behavior-2026-limo` scenemap 빌드)로 `base_qvel` 적분만 다시 만들고, 기록의 외부 GT 자세('G')와 keyframe 마다 비교했다(첫 keyframe 에서 두 틀을 맞춤).
- 회전: GT yaw 에 대한 yaw 오차의 직선 기울기는 **+1.21 %**(제자리 6.34 rad 동안, keyframe 52 개)다. 잔차 σ 는 3e-5 rad 로, 걸음 잡음이 없는 순수 치우침이다. R1 의 +13 % 보다 10 배 작다.
- 이동: 회전이 끝난 뒤 직진 구간(10.77 s~)에서 오도메트리는 0.8084 m, GT 는 0.8018 m 로 **+0.82 %** 다. 직진 중 yaw 변화는 없다(< 5e-5 rad/m).
- 판 끝 오차는 6.4 cm / 4.4° 이고, 거의 다 회전 치우침에서 온다. 같은 기록을 `--pose slam` 으로 재생하면 끝 오차는 0.4 cm / 0.14° 다.
- MP 는 판마다 가우스로 치우침을 뽑고 부호를 모른다. 표본이 하나이므로 σ = |측정값| 으로 두었다(`odo_bt 0.0082`, `odo_br 0.0121`, `odo_bw 0`).

## 2. SLAM 잔차 → odo_* 잡음, kf_corr
**실제 값**(`poses.csv` 의 map_* − gt_map_*, keyframe 291 개, 0.2 s 마다): 13.17 m, 879°, 58.4 s 동안 rms 3.05 cm / 0.256°, max 5.34 cm / 0.662°, 끝 4.08 cm / 0.49° 이다(`pose_diag.json` 과 같음). 7.0 m 지점(스텝 900)까지는 rms 2.27 cm, max 3.5 cm 다. 오차는 천천히 변하는 치우침 꼴이고 max/rms 는 1.75 다. 몸 좌표 평균은 앞 +1.9 cm, 옆 −1.1 cm 다(FK 1 cm 오차와 크기가 비슷하다).

**모형 맞춤.** `map_drift` 를 내 빌드 폴더(scratchpad, `tools/map_drift_limo.patch`)에서 고쳐 썼다. map/ 소스는 건드리지 않았다.
- `map.h` 사본에서 `odo_t/rr/rt`, `odo_bt/br/bw`, `p_conf`, `p_miss_*` 를 `-D` 매크로로 바꿀 수 있게 했다(기본값은 원래 값 그대로).
- `map_drift` 에 궤적 두 가지를 더했다.
  - **kind 2 = LIMO 탐색 비슷**: kind 1 과 같은 동작에 길이 합 13.2 m, 직진 0.40 m/s, 회전 최대 0.8 rad/s. 실제 판은 움직일 때 평균 0.37 m/s, 최대 0.51 m/s, 회전 최대 0.87 rad/s 였다.
  - **kind 3 = limo4 비슷**: 제자리 한 바퀴(0.6 rad/s) 뒤 3.9 m.
- 치우침을 1. 의 LIMO 값으로 고정하고, (odo_t, odo_rr, odo_rt, kf_corr_xy, kf_corr_yaw) 를 격자 두 번으로 찾았다. 첫 격자는 6×4×4×5×5 = 2,400 점, 둘째는 둘레 360 점이고, 점마다 CPU 참조판 128 판이다. 손실은 R1 맞춤 B 와 같다: 로그 제곱 오차에 rms xy·yaw 가중 1, max 가중 0.5. 상위 4 개를 GPU 1,024 판으로 다시 재서 rms 와 max 가 고르게 맞는 것을 골랐다.
- 예: kf_corr_xy 0.005 는 rms 3.19 / max 5.90 cm, 0.01 은 2.56 / 5.13 cm 다. 둘 사이의 0.0075 를 골랐다. yaw 는 이 둘에 거의 무관하다(0.233° / 0.680°).
- 원 오도메트리에는 잡음이 없다. 그래서 `odo_t 0.06` 등은 "scan-matching 잔차를 내는 유효 잡음" 이다(R1 맞춤 B 와 같은 뜻). 실제 잔차는 거의 일정한 치우침(max/rms 1.75)이고, 모형은 1.9–2.9 다.

## 3. 검출

### 3.1 검출 하나하나 (limo3 rec.bin, COCO-80, keyframe 80, 검출 133)
R1(`../README.md` 3.1)과 같은 방법을 썼다. 카메라 자세는 기록의 GT 베이스 자세('G', 보간)에 LIMO eyes 외부 행렬을 붙인 것이다. eyes 는 몸 +x 0.094 m, 높이 0.1805 m, 수평 광축이고, 이 값은 탐색 판 `gt_poses.csv` 의 cam_* 에서 얻었다. 바닥 점 z 중앙값은 0.001 m(n 90,060)로 맞았다. 영상 아래쪽 약 190 줄은 로봇 몸(4 cm)이 가린다.
- 마스크 점을 MAD 로 거른 뒤 세계로 옮기고, 그 점의 30 % 이상이 들어가는 GT AABB(+5 cm, 벽·바닥·천장·잔디 등 뺌)에 붙였다.
- 결과: 맞음 100, 틀린 이름 32, 가짜 1. **틀린 이름 / 붙은 검출 = 0.24**(n 132). 가짜가 있는 keyframe 은 1/80 = **0.013** 이다.
- 틀린 이름의 대부분은 어휘 밖 물체에 되풀이되는 것이다: 창 → "potted plant" 12, coffee_table → "bench" 6, radio → "suitcase" 4, top_cabinet → "tv" 3. R1 에서 가짜로 세던 비침과 같은 종류인데, LIMO 장면에서는 GT 상자 안에 들어서 틀린 이름으로 셌다.
- **놓침은 맞추지 못했다.** 보이는 (프레임, 물체) 쌍은 R1 규칙(검출기 어휘의 종류, 투영 ≥ 20 px·넓이 ≥ 40², 표본점 50 % 이상 안 가림)으로 셌다.
  - < 1.5 m 에는 쌍이 0 개다.
  - 1.5–2.5 m 는 20/74 = 0.27 이다. 그러나 놓친 20 개 중 18 개가 벽걸이 TV 하나이고, 나머지는 2/52 = 0.04 다. 물체는 5 개뿐이다.
  - 2.5–3.0 m 에는 쌍이 0 개다. 3–6 m 는 개수대 10/10 을 놓쳤다(Dabai 3 m 밖이라 MP 는 쓰지 않음).
  - 제자리 회전 판이라 거리 폭이 좁다. 그래서 R1 계단(0.15/0.10/0.79)을 그대로 둔다.
- 위치 흩어짐: 같은 물체에 붙은 맞는 검출 중심의 강건 σ 는 x/y 0.023 m, 높이 0.007 m 다(n 100, 물체 4 개). R1 은 0.04 였다. 이 값으로 `lat_n 0.023` 을 제안한다.

### 3.2 확정 물체 (memory/scene.json vs GT AABB)
확정 물체를 map → 세계로 옮기고(`frame.json` map_from_world), 겹침 ≥ 20 % 또는 중심이 상자에서 0.10 m 안인 GT 를 찾았다. 이름 대응은 엄격(table→breakfast/coffee_table, lamp→room/track/downlight, radio receiver→radio, …)과 너그러움(shelf→top_cabinet·hall_tree)으로 했다.

| 판 | 검출기 | 확정 | 맞음 (엄격+너그러움) | 틀린 이름 | 가짜 | 맞는 것의 GT 상자 중심까지 거리 |
|---|---|---|---|---|---|---|
| **탐색 094010** | 열린 어휘 272(과제 20) | 16 | 13 (10+3) | 3 | 0 | n 13: 3D 중앙값 0.20 m, xy 중앙값 0.08 m, rms 0.53 m, max 1.56 m |
| limo3 | COCO-80 | 14 | 8 | 6 | 0 | n 8: 0.23 m / xy 0.21 m |
| limo4 | COCO-80 | 19 | 7 | 11 | 1 | n 7: 0.29 m / xy 0.29 m |

- 탐색 판의 틀린 이름은 lamp→stairs, picture frame→hall_tree, radio receiver→downlight 다. 위치 오차 max 1.56 m 는 hall_tree 의 GT 상자(5.4 m 길이)의 중심 문제라, 중앙값을 본다. 탁자 하나는 같은 coffee_table 에 확정이 둘이다(0.21 m, 0.68 m).
- 맞춤(map_drift kind 2, 512 판):
  - `p_ghost 0` 이면 `p_conf` 0.07/0.08/0.09 에서 확정 틀린 이름이 0.18/0.21/0.25 다. 실제 3/16 = 0.19 에 맞춰 **`p_conf 0.08`** 로 했다.
  - `p_ghost` 0.02/0.03/0.05 에서 가짜 있는 keyframe 은 0.008/0.012/0.021 이고, 확정 가짜는 0.02/0.04/0.06 이다. **`p_ghost 0.03`** 으로 했다.
- 교차 확인: COCO 의 검출별 틀린 이름 0.24 를 `p_conf` 로 넣으면 map_drift 의 확정 틀린 이름이 0.49 다. 실제 COCO 판은 0.43(limo3), 0.58(limo4)이라 모형이 검출별 값과 확정 값을 일관되게 잇는다. 학습에 COCO 검출기를 쓰면 `p_conf 0.24` 가 맞다.

## 맞추지 못한 것과 까닭
- **탐색 판의 원 오도메트리.** rec.bin 도 `base_qvel` 로그도 없다. 그래서 limo3(회전 1 번, 직진 0.8 m)에서만 쟀고, 치우침 σ 는 표본 1 개다.
- **열린 어휘 검출기의 검출 하나하나**(놓침, keyframe 가짜, 검출별 틀린 이름). 탐색 판에 rec.bin 이 없다. 검출별 값은 COCO(limo3)뿐이고, 열린 어휘 쪽은 확정 물체 16 개로만 맞췄다.
- **거리별 놓침.** limo3 의 보이는 쌍이 1.5–2.5 m 에만 있고(물체 5 개), 놓침도 TV 하나가 거의 다다. R1 값을 그대로 둔다.
- **kind 3(limo4) 대조.** 모형 장면(G1 방 가운데 출발, 3 m 범위)의 한계로 xy 가 3.6 배 크다.
- **`ext_n`, `min_hits`, `wall_h`, 깊이 잡음.** 재지 않았다. 시뮬 깊이에는 잡음이 없다.
- **FK 1 cm 오차의 몫.** 고친 카메라(391c04b 뒤)로 찍은 판이 아직 없어서 따로 떼어 내지 못했다. (10-06: 그 뒤 판 `~/behavior-2026-limo/outputs/explore_20261004_180748_turning_on_radio_frontier_limo_gt` 이 있다 — rec.bin 없음, 이 판으로 자세 비교는 아직 안 함.)

## 재현
```
# 1. 원 오도메트리 / SLAM 재생 (CPU)
<behavior-2026-limo scenemap 빌드>/sm_bench limo3/rec.bin --robot limo_omx --pose odom --no-dets --traj l3_odom.csv
# 3. 검출·확정 물체 채점 (numpy; GT = 탐색 판 gt_poses.csv.objects.json)
python tools/det_score.py limo3/rec.bin <explore>/gt_poses.csv.objects.json
python tools/confirmed_score.py <explore> <explore>/gt_poses.csv.objects.json [frame.json]
# 2. map_drift: map/ 를 빌드 폴더에 복사해 tools/map_drift_limo.patch 를 적용한 뒤
cmake -S <copy>/map -B b -DCMAKE_CXX_FLAGS="$F" -DCMAKE_CUDA_FLAGS="$F" && cmake --build b --target map_drift && b/map_drift 2 1024
#  F="-DODO_BT_V=0.0082f -DODO_BR_V=0.0121f -DODO_BW_V=0.0f -DODO_T_V=0.06f -DODO_RR_V=0.005f -DODO_RT_V=0.02f
#     -DKF_CORR_XY_V=0.0075f -DKF_CORR_YAW_V=0.15f -DP_GHOST_V=0.03f -DP_CONF_V=0.08f"
python tools/rank.py grid.txt   # 격자 결과 순위(손실)
```
패치는 map.h 작업 트리(md5 75b59997…, HEAD fbae36a 에 커밋 안 된 변경 포함)를 기준으로 만들었다. 다른 사람이 map.h 를 고치는 중이라 줄 번호는 맞지 않을 수 있다.
