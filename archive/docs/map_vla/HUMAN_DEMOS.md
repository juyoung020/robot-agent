# 사람 원격조종 시연 + scenemap 재생 — 조사·시험 기록 (보류, 문서화만)

작성 2026-10-05. **보류된 길의 기록**이다. 2026-10-04 에 "다른 로봇의 사람 시연을 우리 실시간 지도(scenemap)로 다시 돌려 주 학습 자료로 쓴다" 를 시험했고,
2026-10-05 사용자 결정으로 **주 시연 출처는 다시 GPU 시뮬 RL 교사**가 되었다([MAPVLA_SPEC.md](MAPVLA_SPEC.md) 결정 기록 2026-10-05, 4.3c). 이 길은 더 받거나 바꾸지 않는다.
코드는 [training/demos](../../training/demos/README.md)(보류 표시), 거르기 결과는 `~/datasets/human_demos/b1k_screen/`(큰 `segments.csv` 73 MB)와 저장소의 `training/demos/b1k/screen_results/`(작은 표).
표기: 사실은 출처·파일. **(추정)** = 재지 않은 판단. **(계산)** = 잰 값·파일 크기에서 산수로만.

## 0. 한눈에

1. **방법**: 메타데이터만으로 시연을 구간별로 거르고(E0 팔 한도) → 쓸 구간만 받아 → 우리 카메라 자리에서 다시 그리거나(계획) 시연 카메라 그대로 → scenemap 으로 다시 돌려 지도 토큰·단계 라벨·공통 행동을 만든다.
2. **BEHAVIOR 2026 사람 시연 20,000 판(1,953 h)** 중 OMX-F 팔 행동으로 쓸 수 있는 구간은 **28.5 h(1.5 %)**, 베이스 이동·탐색만 쓸 수 있는 구간 749 h(38 %), 단계 문장(높은 수준)만 1,087 h, 버림 91 h.
3. **보류 이유**: 팔 행동 자료가 너무 적음(1.5 %), 쓸 판만 받아도 원본 268 GB·LeRobot 머리 카메라 319 GB 로 디스크(여유 약 24 GB, 예산 ≤ 30 GB)에 안 들어감, 이동 자료는 많지만(749 h) 우리 GPU 시뮬이 이동·탐색 자료를 무한히·라벨째 만든다.
4. **재생 자체는 된다**: ep 0 을 R1 Pro scenemap 으로 다시 돌린 판이 학습 뷰어(Replay, group `human_demos`)에 있다. 변환 101 s(대부분 Python 상자 라벨), scenemap 재생 2.7 s.

## 1. 방법

### 1.1 메타데이터 거르기 (`training/demos/b1k/screen.py`)

- 입력(작은 파일만, 영상 없음): HF `behavior-1k/2026-challenge-demos` 의 `annotations/task-NNNN/episode_*.json`(판마다 스킬 구간: 설명·`object_id`[물체, 받침]·`frame_duration`·`skill_type` navigation/uncoordinated/coordinated), `meta/episodes`(판 → 과제·인스턴스·길이·원본 번호), `meta/tasks.jsonl`. 과제 인스턴스 장면 JSON(템플릿 = 장면 물체, `tro_state` = 과제 물체·인스턴스 시작 자세), 물체 `metadata.json` `bbox_size` × `scale`, `avg_category_specs.json`(종류 평균 질량).
- 주석 20,000 개는 **git sparse clone**(261 MB, 한 번)으로 받았다. 파일별 HF 받기는 3.5 파일/s 로 1.5 시간 넘게 걸리고 429 위험(`behavior-2026 docs/데이터셋.md` 2.8).
- 주석이 물체를 범주 이름(`can_of_soda`, `toy_figure`)으로 쓴 경우가 많다 → 그 범주의 과제 물체 인스턴스로 찾는다(표시 `catref`). 끊긴 스킬은 `frame_duration` 이 구간 목록(724 개).
- 파일 크기: `fetch_sizes.py` 가 HF 목록 API 로만(원본 HDF5 판별, LeRobot 영상·data 파일별). LeRobot 판 크기 = 과제 파일 합 × 판 길이 / 과제 길이 합 (계산).

### 1.2 E0 한도와 태그 (사용자 규칙 2026-10-04, [CURRICULUM_BEHAVIOR2026.md](CURRICULUM_BEHAVIOR2026.md) 3.1·5.3)

| 거르개 | 느슨 | 엄격 |
|---|---|---|
| 집을 물체 가로 최소 변 | ≤ 0.06 m | ≤ 0.04 m |
| 종류 평균 질량 | ≤ 0.40 kg | ≤ 0.25 kg |
| 집는 높이(받침 윗면, 받침 모르면 물체 바닥) | ≤ 0.50 m | ≤ 0.45 m |
| 위에서 잡기 / 옆 잡기 | 0.25 m 넘으면 옆 잡기만: 물체 중심 → 받침 가장자리 ≤ 0.10 m(받침 모르면 느슨만) | 같음 |
| 놓을 면 윗면(바닥은 됨, `place in` 은 +0.05) | ≤ 0.52 m | ≤ 0.48 m |
| 양손(`skill_type` coordinated) | 행동 학습 제외 | 같음 |

| 태그 | 구간 |
|---|---|
| `action_full` | 집기·놓기가 느슨 한도 통과(엄격 통과는 `grade=strict`) |
| `action_base_only` | `move to`, `turn to` — 베이스 (vx, vy, wz) 그대로, 팔은 가림 |
| `highlevel_only` | 한도 밖 집기·놓기(이유 기록), 양손, 열기·닫기·토글·누르기·밀기·건네기 — 단계 문장 학습에만 |
| `drop` | 도구·입자 스킬(자르기·닦기·쓸기·붓기·뿌리기·불붙이기 …), 주석 없는 프레임 |

아직 안 본 것(변환 때 할 일이었음): 팔 끝 궤적이 OMX 작업 공간 안인지(옆 0.27·앞 0.17 m, OMX IK), 어느 팔이 일하는지(그리퍼 닫힘·주석), 몸통 안 닿음.

### 1.3 R1 Pro → 한 팔 변환 규칙 (사용자 승인 2026-10-04, 구현 전)

1. 일하는 팔만: 구간마다 주석·그리퍼 열고 닫음·쥔 손으로 팔을 고르고 그 팔의 팔 끝 궤적 + 그리퍼만 쓴다. 쉬는 팔은 버림.
2. 양손 구간은 행동 학습에서 빼고 단계 문장 학습에만.
3. 몸통(torso 4 관절): 팔 끝 행동을 **로봇 베이스 기준**으로 적으면 몸통 움직임이 궤적에 흡수된다 → 몸통 관절은 안 씀.
4. 위 1.2 한도. 한도 밖은 단계 문장만.
5. 베이스 평면 속도(vx, vy, wz)는 공통 행동에 그대로. 이동·탐색 구간이 가장 값지다.
6. 구간 태그 4 종 + 이유(1.2 표).

공통 행동 제안(MAPVLA_SPEC 2.5 의 몸 무관 칸을 채우는 꼴, 구현 전): 팔 = 팔 끝 자세 변화 6D(베이스 기준, 위치 3 + 축각 3) + 그리퍼 벌림(0 닫힘 … 1 열림), 베이스 = (vx, vy, wz)(고정 팔이면 가림), 몸마다 가림. OMX 로 실행할 때는 URDF 잡는 점(`grasp_point`, omx_link5 x 0.08003) IK — 5 자유도라 위치 + 도구 축 + roll 만 맞춤. 관절 출력 머리는 대안으로 남김.

### 1.4 재생 (`demo2rec.py` → `demo2sg`)

- `demo2rec.py`(오프라인 형식 바꾸기, Python — LeRobot·HEVC·HDF5 도구가 Python 뿐): LeRobot 머리 zed RGB·`depth_linear`(gray12le 로그 역양자화, 가운데 720 × 540 → 640 × 480 → 320 × 240), proprio 61(평가기 형식 = R1 Pro scenemap 입력 그대로), 원본 HDF5 의 정답 베이스·물체 자세(시뮬 없이 state 행에서, scenemap `eval/gt_traj.py`), 검출 = 깊이 점을 담는 가장 작은 정답 물체 상자의 범주(`map_src: gt_box`, scenemap `eval/export_gtdet.py` 규칙, 구조물 제외, 마스크 160 × 120) → SGRC `rec.bin`(sgrt 기록 형식) + `labels.txt` + `demo.json`.
- `demo2sg`(C++, scenemap C ABI): `rec.bin` → scenemap(`robot r1pro`, slam 또는 gt 자세) → 학습 뷰어 재생 판(scenemap sgview 스트림 + 시뮬 시각, 끝에 `sm_save_dsg`, 머리 카메라 JPEG 5 Hz, BEHAVIOR 집 배치 바탕 층) — `training/viewer/tools/og2sg` 와 같은 방식, 실행 폴더 group `human_demos`.
- **우리 카메라 자리에서 다시 그리기(계획, 안 함)**: 시연 카메라(R1 머리 zed, 바닥 위 약 1.6 m (추정))는 LIMO 몸통 카메라(0.18 m)와 시점이 아주 다르다. OmniGibson 으로 원본 HDF5 를 재생하면서(`replay_obs.py` 방식) LIMO 카메라 자리(베이스 자세 + URDF 장착)에 카메라를 더 달아 RGB·깊이·인스턴스 분할을 그리고, 그 영상으로 scenemap 을 돌리면 배포와 같은 시점의 지도·영상이 나온다. 원본 HDF5 만 받으면 되어(판당 6–560 MB) LeRobot 영상(판당 약 100 MB)보다 싸다.

## 2. BEHAVIOR 2026 거르기 결과 (20,000 판, 2026-10-04, 66 s)

### 2.1 전체

| 태그 | 시간 | 비율 |
|---|---:|---:|
| action_full | 28.5 h | 1.5 % |
| action_base_only | 749.5 h | 38.4 % |
| highlevel_only | 1,087.3 h | 55.7 % |
| drop | 90.6 h | 4.6 % |
| 합 | 1,952.9 h | |

- 구간 406,306 개(action_full 4,563, action_base_only 144,798, highlevel_only 233,613, drop 23,332). 집기 93,085 중 action_full 3,098(엄격 통과 포함, 집기·놓기 엄격 통과 931). **폭 한도만 0.08 m 로 넓히면 집기 7,679**(2.5 배 — 캔 지름 0.066–0.076 m).
- action_full 이 하나라도 있는 판 **2,116**(18 과제). 양손 구간 29,136.
- highlevel 이유(집기·놓기 시간, 이유 겹침): 폭 311 h, 든 물체가 이미 집기 한도 밖 286 h, 높이 206 h, 옆 잡기 가장자리 176 h, 질량 172 h, 양손 64 h, 물체 못 찾음 18 h, 놓을 높이 4.5 h.

### 2.2 장면별 (최종 — 2026-10-04 처음 보낸 값은 범주 이름 찾기 전이라 틀렸음)

| 장면 | 시연 | action_full 있는 판 | action_full h | 이동 h | highlevel h |
|---|---:|---:|---:|---:|---:|
| house_single_floor | 6,800 | 1,130 | 13.55 | 268.7 | 477.8 |
| house_double_floor_lower | 6,400 | 662 | 6.26 | 248.1 | 302.3 |
| restaurant_diner | 1,600 | 257 | 7.96 | 62.5 | 76.1 |
| hotel_suite_large | 1,000 | 64 | 0.75 | 40.1 | 59.0 |
| house_double_floor_upper | 2,000 | 3 | 0.02 | 61.0 | 106.6 |
| office_cubicles_right | 1,000 | 0 | 0 | 43.5 | 30.2 |
| Rs_int | 1,200 | 0 | 0 | 25.6 | 35.3 |

### 2.3 작은 물체 집기·놓기 상위 과제

| 과제 | 장면 | action_full 있는 판 | action_full h | 이동 h | 쓸 판만 받기: 원본 HDF5 / LeRobot 머리 RGB+깊이 (계산) | 집는 물체 |
|---|---|---:|---:|---:|---|---|
| clean_up_broken_glass | restaurant_diner | 200 | 7.36 | 8.7 | 7.1 / 21.7 GB | broken_glass |
| clean_your_rusty_garden_tools | house_single_floor | 200 | 4.83 | 16.9 | 46.5 / 46.6 GB | trowel, scraper |
| put_together_a_basic_pruning_kit | house_double_floor_lower | 200 | 3.88 | 7.3 | 6.8 / 10.3 GB | shears |
| sorting_household_items | house_single_floor | 200 | 2.69 | 12.5 | 10.2 / 23.3 GB | toothpaste, toothbrush |
| organizing_school_stuff | house_single_floor | 200 | 2.62 | 10.5 | 13.8 / 18.4 GB | pen |
| putting_up_Christmas_decorations_inside | house_single_floor | 200 | 1.45 | 11.5 | 14.7 / 71.6 GB | candy_cane |
| spraying_for_bugs | house_double_floor_lower | 200 | 1.01 | 4.0 | 34.6 / 36.5 GB | insectifuge_atomizer |
| spraying_fruit_trees | house_double_floor_lower | 200 | 1.01 | 6.2 | 111.8 / 45.6 GB(입자 상태로 원본 판당 560 MB) | pesticide_atomizer |
| scrubbing_bathroom_floor | house_single_floor | 200 | 0.76 | 2.9 | 2.7 / 3.1 GB | scrub_brush |
| collecting_childrens_toys | house_single_floor | 102 | 0.99 | 9.0 | 6.9 / 13.5 GB | dice |

그다음 polishing_shoes 64 판, setting_the_fire 62, cook_brussels_sprouts 57, halve_an_egg 14, clean_up_your_desk 8.
가장 쓸 만한 판 예: ep 16128(clean_up_broken_glass, 151.6 s) — action_full 65 %, 이동 35 %, 원본 15.9 MB.

### 2.4 크기

| 묶음 | 원본 HDF5 | LeRobot 머리 RGB + 깊이 + data | LeRobot 전부 |
|---|---:|---:|---:|
| 20,000 판 전부 | 1,440 GB | 2,095 GB | 3.27 TB(HF 표기) |
| action_full 있는 2,116 판 | 268 GB | 319 GB | 451 GB |
| 위 2.3 의 1·3·4·5·9·10 번 과제 | 약 42 GB (계산) | | |

원본 판 크기는 과제마다 6–560 MB(입자 과제가 큼). 쓸 판만 받아도 이 PC 여유(약 24 GB)·예산(≤ 30 GB)을 넘는다.

### 2.5 버린 구간 예

| 판 | 과제 | 구간 | 태그 | 이유 |
|---|---|---|---|---|
| 0 | turning_on_radio | pick up from radio ← coffee_table | highlevel_only | 폭 0.138 m, 1.85 kg, 옆 잡기 가장자리 0.17 m |
| 0 | turning_on_radio | press radio | highlevel_only | 양손(coordinated) |
| 0 | turning_on_radio | place on radio → coffee_table | highlevel_only | 든 물체가 집기 한도 밖 |
| 200 | picking_up_trash | pick up from can_of_soda ← floors | highlevel_only | 폭 0.075 m(> 0.06) |
| 200 | picking_up_trash | pick up from trash_can ← floors | highlevel_only | 폭 0.23 m, 3.75 kg |
| 400 | putting_away_Halloween_decorations | open drawer bottom_cabinet | highlevel_only | 관절체 |
| 600 | cleaning_up_plates_and_food | pick up from plate ← breakfast_table | highlevel_only | 폭 0.41, 0.50 kg, 높이 0.67 m |
| 800 | can_meat | pick up from hinged_jar ← top_cabinet | highlevel_only | 높이 1.48 m |
- ep 0 판 전체: action_full 0 %, 이동 13.6 %, highlevel 77.3 %, drop 9.2 %. ep 1: 0 / 16.1 / 68.5 / 15.5 %.

## 3. ep 0 재생에서 찾은 것

- 학습 뷰어 `behavior_og/human_demos/b1k_ep0_turning_on_radio`(Replay 탭): 점유 지도·조각 점구름·물체 노드 11(radio, coffee_table, picture, electric_switch, room_light …)·방·궤적이 시간에 따라 자라고 시연 머리 카메라가 옆에 나온다. 1,956 스텝, keyframe 652, 스트림 프레임 3,787(버림 0), 판 12 MB.
- **들린 라디오**: 31–55 s 에 scenemap 이 radio(id 7, 지도 상자 0.21 × 0.14 × 0.24 m, 구름 806 점)를 HELD 로 둔다. 시연에서 사람이 실제로 라디오를 들었다(정답 radio_89 가 움직임, 상자 0.14 × 0.32 × 0.24 m). R1 모드는 옛 잡기 규칙(그리퍼가 닫히는 순간 팔 끝 0.25 m 안 가장 가까운 확정 물체, 크기·폭 확인 없음)을 쓴다 — LIMO 만 크기·손끝 틈 확인(`grasp_check`)을 켠다(behavior-2026 `d58c978`). R1 데모용으로 같은 확인을 켜려면 R1 손가락 틈 표가 필요하다(`grip_gap` 이 비면 틈 0 → 늘 빈손): scenemap 설정 키로 따로 켜는 것이 남은 일(기본 R1 동작은 그대로). 사용자가 본 "작은 덩어리" 모양: 들린 동안 구름은 806 점 그대로이고 손을 따라 평행 이동만 한다(지워지지 않음) — 화면에서 작게 보인 원인은 확인 못 함.
- **OmniGibson 재생**: `gm.HEADLESS = True` 가 없으면 센서 초기화 렌더에서 segfault. 켜면 RGB·깊이·proprio 재생은 된다. **`seg_instance`·`seg_instance_id` 모달리티를 넣으면 SyntheticData `_post_process_`(omni.syntheticdata 0.6.13) 안에서 segfault**(모달리티를 하나씩 빼서 확인, `og_probe.py`). 그래서 시뮬 정확한 인스턴스 마스크 대신 정답 상자 라벨을 썼다. 상자 라벨은 상자가 모양보다 커서 받침·벽 점이 물체로 붙을 수 있다.
- 시간: `demo2rec` 101 s/판(상자 라벨 97 s, Python numpy), `demo2sg` 2.7 s(scenemap 0.83 s). 중간 `rec.bin` 360 MB/판(지움).

## 4. 작은 팔 데이터셋 (조사 중 멈춤 — 모은 것만)

(2026-10-04, 메타데이터·카드·HF/GCS 목록만, 받은 파일 없음. "확인 안 함" 은 보지 않은 것, "(추정)" 은 확인 못 한 판단. 라이선스 "없음" = 카드에 표기 없음 = 허락 없음.)

결론: scenemap 이 필요한 세 가지(깊이 또는 스테레오, 내부 파라미터, 베이스 기준 카메라 자세)를 다 주는 것은 **DROID 원본과 BEHAVIOR 뿐**이다. OMX 같은 작은 팔 자료는 RGB 만, 보정 없음, 대부분 라이선스 표기 없음.

### 4.1 DROID (Franka, 고정 베이스, CC-BY-4.0)

| 판 | 크기 | 판 수 | 카메라·깊이 | 보정 | 행동 | 말 |
|---|---|---|---|---|---|---|
| 원본 1.0.1 `gs://gresearch/robotics/droid_raw/1.0.1/` | 8.7 TB(스테레오 포함), 판 폴더 0.33–0.65 GB | — | ZED 2 × 2 외부 + ZED Mini 손목, 15 Hz. 깊이 파일 없음, **스테레오 MP4**(나란히 2560 × 720, ZED SDK 없이 읽힘 — 정류 여부 (추정)) + SVO | `trajectory.h5` 프레임별 카메라 6-DoF(베이스 기준, 왼·오른 눈), 내부 파라미터는 HF 모델 저장소 `KarlP/droid` `intrinsics.json`(판·카메라별, 1280 × 720) | 관절 7·직교 6·그리퍼 1 | `KarlP/droid` 주석(95 % 넘는 성공 판에 3 개씩) |
| RLDS 1.0.1 | 1,866 GB | 95,658 | 왼 눈 180 × 320 × 3, 깊이 없음 | 없음 | 7 | 일부 |
| `lerobot/droid_1.0.1` | 약 412 GB | 95,658 | 같음(AV1) | 왼 외부 카메라 외부 자세만 | 관절 8 | 3 개 |
- 스테레오 MP4 가 없는 판이 많다(WEIRD 142/142, TRI·AUTOLab·RAIL 일부). 고른 시험 20 판(성공·말 3 개·보정 좋음·스테레오 있음, 실험실 11 곳)은 `training/demos/droid/pilot20.json`, 받을 크기 0.85 GB(스테레오 MP4 + `trajectory.h5` + 메타) + `KarlP/droid` JSON 0.18 GB. 깊이는 스테레오에서 직접(SGBM 등), 기선은 왼·오른 외부 자세 차(0.113–0.120 m, Mini 0.062 m).

### 4.2 Open X-Embodiment (RLDS 에 실제로 든 것 기준)

| 부분집합 | GB | 판 | 이동 | RLDS 깊이 | RLDS 보정 | 행동 |
|---|---:|---:|---|---|---|---|
| fractal (RT-1) | 119.3 | 87,212 | 예(베이스 행동) | 없음 | 없음 | 팔 끝 변화 + 베이스 |
| bridge (WidowX) | 416.1 | 28,935 | 아니오 | 없음 | 없음 | 팔 끝 변화 |
| taco_play | 51.3 | 3,603 | 아니오 | 있음 | 없음 | 팔 끝 |
| berkeley_autolab_ur5 | 82.0 | 1,000 | 아니오 | 있음 | 없음 | 팔 끝 변화 |
| nyu_franka_play | 5.6 | 456 | 아니오 | 있음(2 시점) | 없음 | 팔 끝 속도 |
| stanford_robocook | 133.8 | 2,460 | 아니오 | 있음(4) | 외부 자세 | 팔 끝 |
| io_ai_tech(사람 동작) | 307.6 | 3,847 | 아니오 | 있음 | 있음 | 7 |
| cmu_stretch | 0.76 | 135 | 예 | 없음 | 없음 | 8 |
| aloha_mobile | 98.5 | 276 | 예 | 없음 | 없음 | 관절 16 |
| nyu_door_opening | 7.7 | 484 | 예 | 없음 | 없음 | 팔 끝 변화 |
- OXE 표의 "깊이 카메라"·"보정" 칸은 하드웨어 설명이라 RLDS 에 든 것과 다르다(DROID·bridge 는 표에 깊이 카메라가 있지만 RLDS 에 깊이 없음). 부분집합별 라이선스는 확인 안 함(저장소는 자료 CC-BY 4.0).

### 4.3 작은 팔 (LeRobot HF, 전부 RGB, 깊이·보정 없음)

| 순위(6 cm 그리퍼·400 mm 5 자유도에 맞는 정도) | 자료 | 크기 | 판 | 라이선스 | 로봇 | 일 |
|---|---|---:|---:|---|---|---|
| 1 | Hadolking/openmanipulator-pickplace-dataset-1114ep | 2.16 GB | 1,114 | 없음 | OpenManipulator-X | pick & place, 관절 4 + 그리퍼, 30 fps |
| 1 | yechan3219/openmanipulator_x_18k_episode_topsideview | 3.61 GB | 1,800 | 없음 | OpenManipulator-X | 정육면체 집어 목표에(실제·시뮬 (추정)) |
| 2 | pseudolab/omx_f_PickUpDollWith2Cam | 1.33 GB | 94 | Apache-2.0 | OMX-F | 인형 |
| 2 | maximellerbach/omx_multicubes_annotated | 2.44 GB | 171 | Apache-2.0 | OMX follower | 정육면체, 말 사건 칸 |
| 3 | lerobot/svla_so100_pickplace·sorting·stacking, svla_so101_pickplace, koch_pick_place_*_lego | 0.09–2.1 GB | 50–102 | Apache-2.0 | SO-100/101, Koch | 정육면체·레고 |
| 4 | HuggingFaceVLA/community_dataset_v3 | 758 GB | 50,622(251.5 h, 791 묶음) | Apache-2.0 | 대부분 SO-100/101 | 여러 |
| 5 | LeKiwi(SO-101 + 3 바퀴 베이스) 묶음들 | 0.4–0.5 GB | 31–150 | 없음·Apache-2.0 섞임 | 이동 | 짧은 일, 베이스 칸 형식이 묶음마다 다름 |
| 6 | BridgeData V2 (WidowX 250) | 411 GB(사람 조종) | 60,096 | CC-BY-4.0 | WidowX | 집기·놓기·밀기·쓸기, 고정 RGBD 카메라 1 대만 깊이 |
- ROBOTIS 조직의 Task_0001–0006 은 OMX 가 아니라 AI Worker(양팔 7 자유도). SHEC3R/HEC3R-Embodiment-OMX 는 깊이·내부 파라미터·`T_world_base` 가 있으나 시연이 아니라 보정 자료로 보임 (추정). 그리퍼 벌림 사양은 어느 팔도 확인 안 함.
- 출처: <https://droid-dataset.github.io/>, <https://huggingface.co/KarlP/droid>, <https://huggingface.co/datasets/lerobot/droid_1.0.1>, OXE 표 <https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g>, GCS `gresearch/robotics` 목록, <https://rail-berkeley.github.io/bridgedata/>, <https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v3>, 각 HF 카드(2026-10-04).
- BEHAVIOR 원본 HDF5 판 크기 분포(HF 목록): 중앙값 48.9 MB, 0.67 MB–1.82 GB. 원본 저장소엔 카드·라이선스가 없다(데모 저장소는 MIT).

## 5. 왜 채택하지 않았나 (2026-10-05)

1. **팔 행동 자료가 너무 적다**: OMX-F 한도(폭 6 cm, 0.4 kg, 높이 0.5 m)를 지나는 구간은 1,953 h 중 28.5 h(1.5 %), 그나마 7 과제에 몰림(깨진 유리·원예 도구·가위·칫솔·펜·사탕 지팡이·분무기). 집 과제 대부분은 조리대·식탁(0.7–0.9 m) 위 물체이거나 크고 무겁다.
2. **디스크에 안 들어간다**: 쓸 판만 받아도 원본 268 GB / LeRobot 머리 카메라 319 GB. 이 PC 여유 약 24 GB.
3. **이동 자료(749 h)는 많지만** 우리 GPU 시뮬이 같은 BEHAVIOR 집에서 이동·탐색 자료를 무한히·라벨째(지도 토큰·단계 라벨 포함) 만든다(BC README G5). 사람 이동 자료의 값(사람다운 탐색·속도 분포)은 있으나 받기·재생 비용에 비해 작다 (추정).
4. 시점 차이: R1 머리 카메라(약 1.6 m (추정))와 LIMO 몸통 카메라(0.18 m). 우리 카메라 자리에서 다시 그리려면 OmniGibson 재생이 필요하고, 인스턴스 분할은 지금 이 PC 에서 segfault.

## 6. 다시 할 때 (재개 방법)

- **단계 문장·높은 수준 학습**: highlevel 1,087 h 의 주석(스킬 문장·물체·받침, 판당 약 3 KB, 이미 `~/datasets/b1k_ann_git` 261 MB)은 영상 없이도 "다음 단계 문장" 예측 자료가 된다(지도 토큰은 정답 장면에서 만들거나 GPU 시뮬로 같은 장면을 다시 돎). 받을 것 없음.
- **이동 사전학습**: `action_base_only` 구간만 고르고 원본 HDF5(판당 6–30 MB 인 과제 우선)만 받아 OmniGibson 으로 LIMO 카메라 자리에서 RGB·깊이를 다시 그림 → scenemap → 지도 토큰 + 베이스 속도. 디스크는 원본만(과제 하나 200 판 ≈ 1–7 GB).
- **팔 행동**: 2.3 의 작은 물체 과제부터(clean_up_broken_glass 7.1 GB, pruning_kit 6.8 GB). 남은 일: R1 손가락 틈 표 + scenemap R1 데모용 잡기 확인 스위치, 일하는 팔 고르기, 베이스 기준 팔 끝 변화 → OMX IK 작업 공간 판정, 인스턴스 분할 segfault 우회(예: OmniGibson 판 바꾸기, 또는 정답 메시 광선 투사).
- 다시 돌리는 명령: [training/demos/README.md](../../training/demos/README.md).

## 출처

- HF: <https://huggingface.co/datasets/behavior-1k/2026-challenge-demos>(카드: MIT, 20,000 판, 3.27 TB), <https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata>(1.44 TB, `replay_obs.py` 로 재생). 크기는 HF 목록 API(2026-10-04).
- behavior-2026 `docs/데이터셋.md`(형식·주석·깊이 양자화), scenemap `eval/{demo_data,gt_traj,gt_scene,export_gtdet}.py`(읽기만), `objmap.cpp` 잡기 규칙·`capi.cpp robotParams`.
