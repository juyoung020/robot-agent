# 커리큘럼 2: BEHAVIOR 2026 에 맞춘 커리큘럼과 학습 환경 개선 계획

작성 2026-10-04. **설계 문서**다. 코드·학습 실행은 없다(GPU 는 쓰지 않았다).
사용자 결정 두 가지를 따른다: ① 커리큘럼은 **BEHAVIOR Challenge 2026** 을 따른다. ② 학습을 더 돌리기 전에 **학습 환경부터** 고친다.

> **범위 확정 (2026-10-04, 사용자):** 대회에는 나가지 않는다. 목표는 **리모 + OMX-F 의 집기·놓기(pick & place)** 하나이고, [VLA_INPUT.md](VLA_INPUT.md) 설계대로 **지도를 관측(지도 토큰)으로 주면서** 학습한다. BEHAVIOR 2026 의 장면·물체는 **학습 무대**로만 쓴다. 그래서 아래에서 q_score·100 과제 상한·대회 전략(0절 일부, 8절 질문 1·2), B6(관절체·버튼)·B7(전체 과제), E5(관절체)는 **범위 밖**이다. 커리큘럼은 B0–B5(이동 → 찾기 → 다가가기 → 집기 → 놓기)만 쓰고, 집기·놓기 대상은 E1 변환기가 뽑는 "팔이 닿는 작은 물체 × 닿는 높이의 받침" 표에서 고른다. 높이 때문에 안 되는 배치는 학습 장면에서 빼거나 낮은 받침으로 옮긴다(8절 질문 3 → 예).
앞 문서: 이동 커리큘럼 [CURRICULUM_APPROACH.md](CURRICULUM_APPROACH.md)(A0–A5), GPU 학습 [GPU_TRAINING.md](GPU_TRAINING.md), 입력 [VLA_INPUT.md](VLA_INPUT.md), 정책 [POLICY.md](POLICY.md), 팀 엔진 검토 [BEHAVIOR_ENGINE_REVIEW.md](BEHAVIOR_ENGINE_REVIEW.md), 시뮬 포팅 [SIM_PORTING.md](SIM_PORTING.md).
환경 코드: [training/RL/env](../../training/RL/env/README.md)(G1), [training/RL/map](../../training/RL/map/README.md)(자라는 지도), [training/RL/ppo](../../training/RL/ppo/README.md), [training/BC](../../training/BC/README.md).

표기: 사실은 출처를 붙인다. **(추정)** = 재지 않은 판단. 서브모듈 경로는 `src/behavior-2026/` 를 줄여 `B26/` 로 쓴다.

## 0. 한눈에

1. **과제는 50 개가 아니라 100 개다.** 2026 대회는 2025 과제 50 개(0–49)에 새 과제 50 개(50–99)를 더했다. 장면 7 개, 과제마다 공개 평가 인스턴스 20 개(점수용 0–9), 학습 인스턴스 300 개, 사람 시연 200 개. 점수 = BDDL 목표 조건 부분 점수(q_score)의 100 과제 평균. 마감 2026-10-17 20:59 (한국).
2. **LIMO + OMX-F 로 닿는 과제는 아주 적다.** 목표 조건 550 개(접지한 문자 기준)를 하나씩 "이 로봇이 할 수 있나" 로 따지면, 공식 사양(도달 400 mm, 가반 하중 100–250 g)에 가까운 **엄격 기준에서 7 과제만 점수가 0 이 아니고 평균 q 상한은 0.027**, 느슨한 기준에서도 **14 과제, 평균 q 상한 0.087** 이다(1.5절). 막는 것은 높이(조리대 0.9 m·냉장고·윗장), 그리퍼 폭, 무게, 문 열기, 입자 상태(닦기·요리·자르기) 순이다.
3. 그래서 커리큘럼은 **BEHAVIOR 2026 의 장면·물체·술어·지표를 그대로 쓰되, 단계 순서는 이 로봇이 할 수 있는 스킬 순**으로 짠다: 이동 → 집 안 이동(문·방) → 찾기(자라는 지도·빈 지도) → 다가가기 → 집기 → 놓기 → 열기·토글 → 전체 과제(가능한 부분집합). 옛 A0–A2 는 회귀 시험으로 남기고, A3–A5 는 실제 BEHAVIOR 집으로 바꾼다(3절).
4. **지금 GPU 환경(G1 + 지도 근사)에 없는 것**: 실제 집 배치(벽·방·문), 실제 물체 종류·크기·자리, 관절체(문·서랍·장)·토글, 잡기 물리, BDDL 술어 판정, 평가기와 같은 관측(30 Hz·카메라·proprio). 가장 큰 일은 **장면·과제 변환기**와 **술어 판정 커널**이다(4절).
5. **환경 개선 단계 E0–E7**(5절). E0(측정·결정) → E1(변환기, Rust) → E2(실제 집 이동 장면) → E3(물체·술어) → E4(관측 동등) → E5(관절체·토글) → E6(잡기 물리) → E7(OmniGibson 대조). 학습은 E2·E3 통과 뒤 이동·찾기 단계부터 다시 연다.
6. **먼저 답이 필요한 질문 5 개**(6절). 가장 중요한 것: 이 로봇으로 대회 점수를 노릴지, BEHAVIOR 를 학습 무대로만 쓸지.

---

## 1. BEHAVIOR 2026 과제 목록 — 우리에게 필요한 것

### 1.1 대회 사실

| 항목 | 값 | 출처 |
|---|---|---|
| 과제 | 100 개. 0–49 = 2025 과제(문장 다듬음), 50–99 = 새 과제 | `B26/docs/과제목록.md`, updates.html 07/27 원문 |
| 장면 | 7 개: `house_single_floor` 34 과제, `house_double_floor_lower` 32, `house_double_floor_upper` 10, `restaurant_diner` 8, `Rs_int` 6, `hotel_suite_large` 5, `office_cubicles_right` 5 | 같은 곳 |
| BDDL 정의 | 과제마다 `problem0.bddl`(평가기 `activity_definition_id: 0`) | `B26/BEHAVIOR-1K/bddl3/bddl/activity_definitions/<과제>/problem0.bddl`, `OmniGibson/omnigibson/eval/utils/eval_utils.py:218` |
| 인스턴스 | 학습용 300 개/과제(합 30,000, `scenes/<장면>/json/*_instances/*-tro_state.json`), 공개 평가 20 개/과제(id 301–320, 점수용 0–9, 연습 10–19), 비공개 20 개(321–340, 배포 안 됨) | `B26/BEHAVIOR-1K/datasets/2026-challenge-task-instances/`, `평가규칙_원문.md` 7절 |
| 인스턴스 차이 | "initial object states and initial robot poses" 만 다름. 장면 배치(벽·가구)는 같다 | evaluation.html 원문 |
| 점수 | q_score = 완전 성공이면 1, 아니면 **처음엔 거짓이었다가 끝에 참인** 목표 조건 수 ÷ 목표 조건 수, 선택지(option)마다 계산해 최대. 과제 점수 = 인스턴스 10 개 평균, 전체 = 100 과제 평균 | `OmniGibson/omnigibson/metrics/task_metric.py:6-28` |
| 제한 시간 | 사람 시연 평균 길이 × 1.5 (30 Hz 스텝). 과제별 107–1,303 s, 중앙값 약 513 s | `metadata/task.jsonl`, `eval_utils.py` `EVAL_TIMEOUT_MULTIPLIER` |
| 동점 | 시뮬 시간, 베이스 이동 거리, 손끝 이동 거리(사람 평균으로 정규화) | `평가규칙_원문.md` 6.3 |
| 관측 | **RGB + 깊이 + proprio 만.** 정답 분할·물체 상태·목표 pose·전체 점군·**로봇 전역 자세 금지**. SLAM·LLM 허용 | evaluation.html 원문(`평가규칙_원문.md` 1·3절) |
| 평가기가 더 주는 것 | `task_id`, `<로봇>::cam_rel_poses`(베이스 기준 카메라 자세) | `evaluator.py` `_preprocess_obs` (`평가규칙_원문.md` 10절 4) |
| 로봇 | R1Pro 기본, **다른 OmniGibson 로봇 허용**(`--robot-config`, 설정 파일 제출). 시작 자세는 인스턴스 값으로 덮어씀 | 같은 문서 5절 |
| 주기 | 행동 30 Hz, 렌더 30 Hz, 물리 120 Hz(CPU PhysX) | `eval_utils.py` `generate_basic_environment_config` |
| 불러오는 방 | 과제마다 일부 방만(`B100_task_misc.csv` "Rooms to include") | `metadata/B100_task_misc.csv` |
| 지금 순위 | 자체 보고 1 위 Q 0.322(완전 성공 14.8 %) | `B26/docs/대회개요.md` 7절 |

### 1.2 장면 (우리 변환기가 다룰 것)

`B26/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/<장면>/json/<장면>_best.json`(물체 목록·자세·척도·방 태그)과 `layout/floor_{trav,trav_no_obj,trav_no_door,trav_open_door,semseg,insseg}_0.png`(0.01 m/화소)에서 센 값이다(2026-10-04, 읽기만).

| 장면 | 과제 | 물체 | 방 | 문(door 류) | 관절체 | 벽 물체 | 다닐 수 있는 넓이 | 계단 물체 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `house_single_floor` | 34 | 595 | 21 | 24 | 53 | 147 | 2,721 m²(정원 포함, 86 m 폭) | 0 |
| `house_double_floor_lower` | 32 | 260 | 6 | 7 | 20 | 39 | 840 m² | 1 |
| `house_double_floor_upper` | 10 | 134 | 6 | 6 | 11 | 13 | 62 m² | 1 |
| `restaurant_diner` | 8 | 184 | 5 | 3 | 20 | 24 | 219 m² | 0 |
| `Rs_int` | 6 | 80 | 5 | 2 | 24 | 7 | 24 m² | 0 |
| `hotel_suite_large` | 5 | 63 | 2 | 2 | 3 | 13 | 46 m² | 0 |
| `office_cubicles_right` | 5 | 297 | 12 | 15 | 37 | 66 | 776 m² | 0 |

- **층 이동은 없다.** 2 층 집은 `_lower`·`_upper` 가 **서로 다른 장면 모델**이고 과제 하나는 한 장면만 쓴다. 계단 물체는 장애물로만 나온다 → "계단 못 오름" 은 과제를 막지 않는다. 다만 방 사이 바닥 높이 차(문턱)·정원 지형은 LIMO 바퀴(반지름 약 0.045 m (추정))로 못 넘을 수 있다 — 바닥 물체 z 가 장면 안에서 최대 0.15 m 다르다(`restaurant_diner` −0.15 / 0.0). 메시 원점 차이인지 실제 턱인지는 E0 에서 잰다.
- 물체 크기: 물체 USD 는 암호화돼 있다(`objects/<종류>/<모델>/usd/*.encrypted.usd`). **상자 크기는 평문 JSON**(`objects/<종류>/<모델>/misc/metadata.json` 의 `bbox_size` × 인스턴스 `scale`)이라 Python·OmniGibson 없이 읽을 수 있다. 메시(정확한 모양)는 OmniGibson 복호화가 필요하다.
- 질량: 종류 평균 `metadata/avg_category_specs.json`(질량·부피·밀도). 인스턴스별 질량은 아니다(추정치로만 씀).

### 1.3 과제 100 개 표

만든 법(2026-10-04, 읽기 전용 일회성 스크립트 — 저장소에 넣지 않음. 같은 숫자를 E1 변환기가 다시 내야 한다 = E1 통과 기준):
`problem0.bddl` 의 목표를 접지(`forall`·`forpairs` 를 물체 수만큼 펼침, `exists` 는 하나, `or` 는 첫 선택지)해 **목표 조건 수**를 세고, 공개 평가 인스턴스 템플릿(`scene_test/public/<장면>/json/<장면>_task_<과제>_0_0_template[-partial_rooms].json` 의 `inst_to_name`·자세·척도)으로 물체 종류·크기·높이를 붙였다.

- 장면 줄임: HSF `house_single_floor`, HDL `house_double_floor_lower`, HDU `house_double_floor_upper`, RD `restaurant_diner`, RS `Rs_int`, HSL `hotel_suite_large`, OCR `office_cubicles_right`.
- 스킬(2절): 모든 과제 = 이동 + 찾기. 더해서 M 여러 방(목표 물체가 2 방 이상 또는 시작에서 5 m 넘게), P 집기·놓기, O 열기·닫기, T 토글, H 가열·냉각, S 자르기, C 닦기, F 채우기·뿌리기, A 붙이기, W 무거운 물체(> 2 kg).
- **q 상한**: 각 목표 조건을 LIMO + OMX-F 가 원리상 만족시킬 수 있나를 1.5절 기준으로 따져, 할 수 있는 조건 수 ÷ 전체. 시간·인식·실패는 빼고 본 **위쪽 한계 (추정)**. 엄격 = 집는 면 ≤ 0.45 m·놓는 면 ≤ 0.50 m·무게 ≤ 0.25 kg·물체 가로 최소 폭 ≤ 0.08 m·토글 ≤ 0.55 m. 느슨 = 0.60 m·0.62 m·0.5 kg·0.10 m·0.70 m.
- 막는 것: 높이(집을 물체가 높은 곳), 놓을높이, 무게, 폭(그리퍼), 열기(닫힌 관절체 안으로), 닫힌곳안(처음에 닫힌 관절체 안에 있음), 닫힘(처음부터 참 — 아래 1.4), 선반(책장·신발장 칸), 토글높이, 붙이기, 그리고 입자·전이 술어(covered·cooked·real·contains·filled·frozen·on_fire), ?(템플릿에서 물체를 못 찾음).

| # | 과제 | 장면 | 사람 s | 목표 조건 수 | 목표 술어 | 옮길 물체 → 목적지 | 스킬 | q 상한 엄격 / 느슨 | 엄격에서 막는 것(문자 수) |
|---:|---|---|---:|---:|---|---|---|---|---|
| 0 | `turning_on_radio` | HDL | 72 | 1 | toggled_on | — | T | 0.00 / 1.00 | 토글높이 1 |
| 1 | `picking_up_trash` | HDL | 176 | 3 | inside | can_of_soda → trash_can | MP | 0.00 / 1.00 | 무게 3 |
| 2 | `putting_away_Halloween_decorations` | HDL | 460 | 8 | inside, nextto, not open | cauldron, pillar_candle, pumpkin → bottom_cabinet, coffee_table | POW | 0.00 / 0.00 | 폭 6, 무게 6, 열기 5 |
| 3 | `cleaning_up_plates_and_food` | HDL | 457 | 8 | inside, not open, ontop | bowl, pizza → drop_in_sink, fridge, plate | PO | 0.00 / 0.00 | 폭 6, 높이 6, 무게 6 |
| 4 | `can_meat` | HSF | 395 | 9 | inside, not open | bratwurst, hinged_jar → top_cabinet | PO | 0.00 / 0.00 | 열기 6, 높이 6, 닫힘(처음부터 참) 3 |
| 5 | `setting_mousetraps` | HDU | 340 | 6 | ontop, under | mousetrap → floors, furniture_sink | P | 0.00 / 0.00 | 높이 6 |
| 6 | `hiding_Easter_eggs` | HDL | 254 | 9 | nextto, not inside, ontop | easter_egg → lawn, tree, wicker_basket | MP | 0.00 / 0.00 | 폭 9 |
| 7 | `picking_up_toys` | HSF | 630 | 6 | inside | board_game, jigsaw_puzzle, tennis_ball → toy_box | P | 0.00 / 0.00 | 놓을높이 6, 높이 6, 폭 5 |
| 8 | `rearranging_kitchen_furniture` | HDL | 298 | 4 | inside, not open | food_processor, french_press, toaster → top_cabinet | MPOW | 0.00 / 0.00 | 열기 3, 폭 3, 높이 3 |
| 9 | `putting_up_Christmas_decorations_inside` | HSF | 457 | 9 | nextto, ontop | candy_cane, gift_box, pillar_candle, wreath → breakfast_table, christmas_tree, sofa | MP | 0.00 / 0.22 | 놓을높이 6, 폭 4, 무게 3 |
| 10 | `set_up_a_coffee_station_in_your_kitchen` | HSF | 209 | 6 | nextto, ontop | bottle_of_coffee, coffee_cup, coffee_maker, electric_kettle → countertop, saucer | PW | 0.00 / 0.00 | 높이 6, 폭 5, 무게 4 |
| 11 | `putting_dishes_away_after_cleaning` | HSF | 365 | 10 | inside, not open | plate → bottom_cabinet, bottom_cabinet_no_top, top_cabinet | PO | 0.00 / 0.00 | 열기 8, 폭 8, 높이 8 |
| 12 | `preparing_lunch_box` | HSF | 275 | 6 | inside, not open | bottle_of_tea, chocolate_chip_cookie, club_sandwich, half_apple → packing_box | PO | 0.00 / 0.00 | 놓을높이 5, 높이 5, 폭 4 |
| 13 | `loading_the_car` | HDL | 641 | 3 | inside | digital_camera, tennis_racket, toy_box → car | MPO | 0.00 / 0.00 | 폭 3, 무게 3, 열기 2 |
| 14 | `carrying_in_groceries` | HDL | 476 | 4 | inside, not open | beefsteak_tomato, carton_of_milk → fridge | MPO | 0.00 / 0.00 | 열기 2, 높이 2, 닫힘(처음부터 참) 2 |
| 15 | `bringing_in_wood` | HDL | 451 | 3 | ontop | — → floors | MP | 0.00 / 0.00 | ? 3 |
| 16 | `moving_boxes_to_storage` | HDL | 487 | 2 | ontop | storage_box → floors | MP | 0.00 / 0.00 | 폭 2, 무게 2 |
| 17 | `bringing_water` | HSF | 315 | 3 | not open, ontop | beer_bottle → coffee_table | MPO | 0.00 / 0.00 | 닫힌곳안 2, 높이 2, 무게 2 |
| 18 | `tidying_bedroom` | HSF | 368 | 3 | nextto, ontop | hardback, sandal → bed, nightstand | P | 0.00 / 0.67 | 폭 3, 무게 3, 높이 1 |
| 19 | `outfit_a_basic_toolbox` | HSF | 355 | 7 | inside, not open, ontop | allen_wrench, drill, flashlight, plier → countertop, toolbox | POW | 0.00 / 0.00 | 높이 6, 열기 5, 무게 2 |
| 20 | `sorting_vegetables` | HSF | 397 | 13 | inside | bok_choy, broccoli, leek, sweet_corn → mixing_bowl | P | 0.00 / 0.00 | 놓을높이 13, 무게 9, 폭 5 |
| 21 | `collecting_childrens_toys` | HSF | 640 | 7 | inside | board_game, dice, teddy_bear, toy_train → bookcase | P | 0.00 / 0.00 | 선반 7, 무게 5, 폭 4 |
| 22 | `putting_shoes_on_rack` | HDL | 258 | 10 | nextto, not touching, touching | gym_shoe, sandal → floors, hall_tree | P | 0.00 / 0.00 | 폭 10, 무게 10, 선반 4 |
| 23 | `boxing_books_up_for_storage` | HDU | 808 | 6 | inside | hardback → storage_box | MP | 0.00 / 0.00 | 폭 6, 무게 6, 높이 2 |
| 24 | `storing_food` | HSF | 662 | 8 | inside | bag_of_chips, bottle_of_olive_oil, box_of_oatmeal, jar_of_sugar → bottom_cabinet, bottom_cabinet_no_top, top_cabinet | PO | 0.00 / 0.00 | 열기 8, 높이 8, 폭 6 |
| 25 | `clearing_food_from_table_into_fridge` | HDL | 436 | 5 | inside, not open | half_apple_pie, half_chicken, tupperware → fridge | PO | 0.00 / 0.00 | 폭 4, 높이 4, 무게 4 |
| 26 | `assembling_gift_baskets` | HDL | 869 | 16 | inside | bow, butter_cookie, pillar_candle, swiss_cheese → wicker_basket | P | 0.50 / 0.75 | 폭 8, 무게 4 |
| 27 | `sorting_household_items` | HSF | 527 | 8 | inside, nextto, ontop, under | bottle_of_detergent, box_of_sanitary_napkins, coffee_cup, soap_dispenser → multi_station_furniture_sink, shelf | MP | 0.12 / 0.12 | 폭 5, 무게 5, 놓을높이 4 |
| 28 | `getting_organized_for_work` | HDU | 522 | 10 | nextto, ontop, under | desktop_computer, folder, keyboard, monitor → desk, mouse, notebook | PW | 0.00 / 0.00 | 높이 8, 폭 7, 무게 6 |
| 29 | `clean_up_your_desk` | HSF | 714 | 11 | inside, not open, ontop | folder, laptop, paperback_book, pen → bookcase, desk, pencil_case | PO | 0.00 / 0.00 | 높이 10, 폭 6, 놓을높이 6 |
| 30 | `setting_the_fire` | HDL | 304 | 8 | inside, not toggled_on, on_fire, ontop | firewood → newspaper, wood_fireplace | PTH | 0.12 / 0.38 | 무게 4, on_fire 3, 놓을높이 2 |
| 31 | `clean_boxing_gloves` | HSF | 275 | 2 | not covered | — | C | 0.00 / 0.00 | covered 2 |
| 32 | `wash_a_baseball_cap` | HSF | 278 | 2 | not covered | — | C | 0.00 / 0.00 | covered 2 |
| 33 | `wash_dog_toys` | HSF | 374 | 6 | not covered | — | OC | 0.00 / 0.00 | covered 6 |
| 34 | `hanging_pictures` | HDL | 80 | 1 | attached | poster → wall_nail | MA | 0.00 / 0.00 | 붙이기 1, 높이 1 |
| 35 | `attach_a_camera_to_a_tripod` | HDU | 130 | 1 | attached | digital_camera → camera_tripod | A | 0.00 / 0.00 | 붙이기 1, 폭 1, 무게 1 |
| 36 | `clean_a_patio` | HDL | 402 | 1 | not covered | — | C | 0.00 / 0.00 | covered 1 |
| 37 | `clean_a_trumpet` | HDU | 177 | 1 | not covered | — | C | 0.00 / 0.00 | covered 1 |
| 38 | `spraying_for_bugs` | HDL | 216 | 2 | covered | — | F | 0.00 / 0.00 | covered 2 |
| 39 | `spraying_fruit_trees` | HDL | 278 | 2 | covered | — | F | 0.00 / 0.00 | covered 2 |
| 40 | `make_microwave_popcorn` | HDL | 108 | 2 | contains, real | — | SF | 0.00 / 0.00 | real 1, contains 1 |
| 41 | `cook_cabbage` | HSF | 471 | 4 | contains, real | — | OSF | 0.00 / 0.00 | real 2, contains 2 |
| 42 | `chop_an_onion` | HDL | 213 | 4 | contains, inside, real | cutting_board, parer → drop_in_sink | PSF | 0.00 / 0.00 | 놓을높이 2, 높이 2, real 1 |
| 43 | `slicing_vegetables` | HSF | 495 | 9 | not open, not real, real | — | OS | 0.00 / 0.00 | real 8, 닫힘(처음부터 참) 1 |
| 44 | `chopping_wood` | HDL | 358 | 8 | real | — | S | 0.00 / 0.00 | real 8 |
| 45 | `cook_hot_dogs` | HSF | 305 | 2 | cooked | — | OH | 0.00 / 0.00 | cooked 2 |
| 46 | `cook_bacon` | HSF | 256 | 7 | cooked, not open | — | OH | 0.00 / 0.00 | cooked 6, 닫힘(처음부터 참) 1 |
| 47 | `freeze_pies` | HSF | 415 | 7 | frozen, inside, not open | apple_pie, tupperware → fridge | POH | 0.00 / 0.00 | 폭 4, 무게 4, 높이 3 |
| 48 | `canning_food` | HSF | 766 | 10 | filled, inside, not contains, not open, real | bowl → bottom_cabinet | POSF | 0.00 / 0.00 | real 2, filled 2, contains 2 |
| 49 | `make_pizza` | HDL | 640 | 2 | ontop, real | — → baking_sheet | POS | 0.00 / 0.00 | real 1, ? 1, 놓을높이 1 |
| 50 | `freeze_fruit` | HSF | 421 | 7 | inside, not open | apple, strawberry, tupperware → fridge | PO | 0.00 / 0.00 | 높이 6, 놓을높이 4, 열기 2 |
| 51 | `cook_a_brisket` | HDL | 244 | 3 | cooked, ontop | brisket, frying_pan → bar, chopping_board | POH | 0.00 / 0.00 | 폭 2, 놓을높이 2, 높이 2 |
| 52 | `sorting_bottles_cans_and_paper` | HDL | 342 | 16 | inside, not inside | can_of_soda, magazine, newspaper, wine_bottle → bucket, ice_bucket | MP | 0.00 / 0.31 | 무게 13, 높이 11, 폭 6 |
| 53 | `tidying_living_room` | HDU | 419 | 4 | inside, ontop | hardback, newspaper, notebook, pot_plant → bookcase, coffee_table, desk | PW | 0.00 / 0.00 | 무게 4, 폭 3, ? 1 |
| 54 | `putting_away_toys` | HSF | 377 | 8 | inside | toy_figure → toy_box | MP | 0.25 / 0.62 | 폭 6 |
| 55 | `re_shelving_library_books` | HDU | 318 | 3 | inside | hardback, notebook → bookcase | P | 0.00 / 0.00 | 폭 3, ? 3, 높이 3 |
| 56 | `make_rose_centerpieces` | HDL | 151 | 4 | inside, ontop | rose, vase → coffee_table | P | 0.00 / 0.00 | 폭 4, 놓을높이 4, 무게 1 |
| 57 | `sweeping_garage` | HDL | 149 | 2 | not covered | — | C | 0.00 / 0.00 | covered 2 |
| 58 | `stacking_wood` | HSF | 531 | 6 | touching | log | MPW | 0.00 / 0.00 | 폭 6, 무게 6 |
| 59 | `organizing_art_supplies` | HDU | 206 | 5 | inside, ontop | glue_stick, marker, paintbrush, rubber_eraser → desk, tote | P | 0.00 / 0.00 | 열기 4, 높이 4, 폭 1 |
| 60 | `scrubbing_bathroom_floor` | HSF | 105 | 1 | not covered | — | C | 0.00 / 0.00 | covered 1 |
| 61 | `bringing_paper_to_recycling` | HDL | 374 | 3 | inside, not open | newspaper, paper_bag → recycling_bin | MPO | 0.00 / 0.00 | 열기 2, 폭 2, 높이 1 |
| 62 | `halve_an_egg` | HSF | 213 | 5 | inside, ontop, real | carving_knife → drop_in_sink, plate | POS | 0.00 / 0.00 | 놓을높이 3, real 2, ? 2 |
| 63 | `installing_smoke_detectors` | HDL | 86 | 1 | attached | fire_alarm → wall_nail | MA | 0.00 / 0.00 | 붙이기 1, 폭 1 |
| 64 | `setting_the_table` | HSF | 594 | 8 | nextto, ontop | cupcake, plate, table_knife, tablefork → breakfast_table | MPO | 0.00 / 0.00 | 높이 8, 폭 4, 놓을높이 4 |
| 65 | `unloading_the_car` | HDL | 377 | 2 | nextto | briefcase, satchel → sofa | MPO | 0.00 / 0.00 | 닫힌곳안 2, 폭 2, 높이 2 |
| 66 | `turning_out_all_lights_before_sleep` | HSF | 334 | 5 | not toggled_on | — | MT | 0.00 / 0.00 | 토글높이 5 |
| 67 | `boxing_food_after_dinner` | HSF | 223 | 6 | inside, not open, ontop | plate, taco, tupperware → drop_in_sink, fridge | PO | 0.00 / 0.00 | 높이 5, 놓을높이 4, 폭 3 |
| 68 | `cleaning_up_branches_and_twigs` | HDL | 418 | 5 | inside, ontop | branch, recycling_bin → floors | MPOW | 0.00 / 0.00 | 폭 5, 무게 5, 열기 4 |
| 69 | `vacuuming_floors` | HDU | 80 | 1 | not covered | — | C | 0.00 / 0.00 | covered 1 |
| 70 | `thawing_frozen_food` | HDL | 274 | 9 | inside, not frozen, not open, ontop, toggled_on | bread_slice, chicken_breast, plate → bar, microwave | POTH | 0.00 / 0.00 | 폭 4, 높이 4, 놓을높이 3 |
| 71 | `clean_your_rusty_garden_tools` | HSF | 505 | 5 | inside, not covered, not open | scraper, trowel → toolbox | MPOC | 0.00 / 0.00 | covered 2, 열기 2, 닫힘(처음부터 참) 1 |
| 72 | `cook_a_frozen_pie` | RD | 289 | 2 | cooked, ontop | apple_pie → tray | POH | 0.00 / 0.00 | 폭 1, 높이 1, 무게 1 |
| 73 | `organizing_school_stuff` | HSF | 486 | 6 | inside, ontop | calculator, folder, notebook, pen → bed, tote | P | 0.00 / 0.00 | 열기 5, 폭 4, 높이 3 |
| 74 | `carrying_out_garden_furniture` | HSF | 369 | 2 | ontop | garden_chair, wheelbarrow → floors, lawn | MPW | 0.00 / 0.00 | 폭 2, 무게 2 |
| 75 | `put_together_a_basic_pruning_kit` | HDL | 375 | 4 | inside, not open, ontop | pruner, shears, toolbox → floors | POW | 0.00 / 0.00 | 무게 3, 열기 2, 폭 1 |
| 76 | `dispose_of_glass` | HSL | 307 | 4 | inside | water_glass → trash_can | MP | 0.00 / 0.00 | 폭 4, 높이 4 |
| 77 | `installing_a_modem` | RS | 80 | 4 | nextto, ontop, toggled_on | modem, standing_tv → bottom_cabinet | PTW | 0.00 / 0.50 | 무게 3, 놓을높이 2, 토글높이 1 |
| 78 | `make_cabinet_doors` | RS | 115 | 1 | attached | cabinet_door → cabinet_base | AW | 0.00 / 0.00 | 붙이기 1, 무게 1 |
| 79 | `polishing_shoes` | HSL | 358 | 6 | nextto, not covered, ontop | scrub_brush, walker → nightstand, ottoman | MPC | 0.17 / 0.67 | 폭 3, 무게 3, covered 2 |
| 80 | `clean_up_broken_glass` | RD | 290 | 3 | inside | broken_glass → trash_can | MP | 1.00 / 1.00 | — |
| 81 | `packing_meal_for_delivery` | RD | 295 | 6 | inside, ontop | paper_bag, wrapped_hamburger → storage_box | P | 0.00 / 0.00 | 폭 6, 놓을높이 6, 높이 6 |
| 82 | `store_batteries` | RS | 257 | 3 | inside | battery → bottom_cabinet | MPO | 0.00 / 0.00 | 열기 3, 높이 1 |
| 83 | `store_honey` | RS | 226 | 1 | inside | jar_of_honey → bottom_cabinet | MPO | 0.00 / 0.00 | 열기 1, 높이 1, 무게 1 |
| 84 | `tidying_bathroom` | HSL | 433 | 4 | inside, ontop | bar_soap, cork, tissue_dispenser, toilet_paper → soap_dish, toilet, trash_can | P | 0.00 / 0.00 | 폭 3, 놓을높이 3, 높이 2 |
| 85 | `putting_dirty_dishes_in_sink` | RD | 405 | 4 | inside | bowl, plate → commercial_kitchen_sink | MP | 0.00 / 0.00 | 폭 4, 놓을높이 4, 높이 4 |
| 86 | `make_gift_bags_for_baby_showers` | RS | 322 | 6 | inside, ontop | paper_bag, toy_dice, wafer → coffee_table | PO | 0.00 / 0.00 | 높이 6, 놓을높이 4, 폭 2 |
| 87 | `collecting_aluminum_cans` | HSL | 340 | 6 | inside | can_of_soda → ice_bucket | P | 0.00 / 1.00 | 무게 6, 높이 3, 폭 2 |
| 88 | `rearrange_your_room` | HSL | 428 | 3 | ontop | pillow, tissue_dispenser → bed, stand | MP | 0.00 / 0.00 | 폭 3, 무게 3, ? 1 |
| 89 | `installing_a_fax_machine` | OCR | 136 | 2 | ontop, toggled_on | facsimile → cubicle | MPT | 0.50 / 0.50 | 폭 1, 놓을높이 1, 무게 1 |
| 90 | `composting_waste` | RS | 121 | 2 | inside | half_banana, half_pomegranate → trash_can | MP | 0.00 / 0.00 | 폭 2, 놓을높이 2, 높이 2 |
| 91 | `store_produce` | RD | 317 | 4 | inside | mango, pomegranate → fridge | PO | 0.00 / 0.00 | 열기 4, 무게 4, 폭 2 |
| 92 | `installing_a_scanner` | OCR | 142 | 2 | nextto, toggled_on | scanner → laptop | PT | 0.00 / 0.00 | 토글높이 1, 폭 1, 높이 1 |
| 93 | `clean_a_keyboard` | OCR | 129 | 1 | not covered | — | C | 0.00 / 0.00 | covered 1 |
| 94 | `dispose_of_batteries` | OCR | 481 | 4 | inside, ontop | battery, trash_can → floors | MPW | 0.00 / 0.00 | 높이 3, 폭 1, 무게 1 |
| 95 | `cook_brussels_sprouts` | RD | 522 | 26 | cooked, inside, not open, real | — → oven | POHS | 0.00 / 0.00 | real 8, cooked 8, 열기 8 |
| 96 | `cook_broccolini` | RD | 154 | 11 | cooked, ontop | broccolini, frying_pan, garlic_clove → stove | PH | 0.00 / 0.00 | 놓을높이 6, 높이 6, cooked 5 |
| 97 | `setup_a_bar_for_a_cocktail_party` | RD | 447 | 20 | inside, nextto, ontop | can_of_soda, corkscrew, ice_bucket, ice_cube → bar, wine_bottle, wineglass | MP | 0.00 / 0.00 | 높이 20, 놓을높이 14, 폭 13 |
| 98 | `laying_tile_floors` | OCR | 478 | 8 | nextto, ontop | ceramic_tile → floors | MP | 0.00 / 0.00 | 폭 8, 무게 8 |
| 99 | `sorting_books_on_shelf` | HDU | 258 | 11 | inside, ontop | comic_book, hardback, notebook → bookcase | MP | 0.00 / 0.00 | 폭 11, 선반 7, 무게 6 |

### 1.4 성공 술어와 부분 점수 — *q_score 는 범위 밖(대회 안 나감). 아래는 BDDL 해석 확인용으로만 남김*

목표 조건 550 개(위 접지 기준)의 종류:

| 술어 | 수 | 뜻(OmniGibson 물체 상태) | 우리 판정 난이도 |
|---|---:|---|---|
| `inside` | 232 | 물체가 용기 안(상자·광선 검사) | 상자 근사 쉬움, 정확히는 메시 |
| `ontop` | 103 | 위에 올려져 닿음 | 상자 + 접촉 |
| `nextto` | 41 | 옆(거리·크기 기준) | 상자만으로 됨 |
| `not open` | 31 | 관절체 닫힘 | 관절 위치 |
| `real`(+`not real` 5) | 28 | 자르기 등 전이로 새 물체가 생김 | 전이 규칙 — 안 함 |
| `cooked` | 23 | 온도 누적 | 안 함 |
| `not covered`(+`covered` 4) | 21 | 먼지·얼룩 입자 없음 | 시각 입자 — 안 함 |
| `not inside` | 13 | 꺼냄 | `inside` 와 같음 |
| `touching` / `not touching` | 10 / 4 | 접촉 | 접촉 정보 |
| `toggled_on` / `not toggled_on` | 5 / 6 | 켜짐 | 버튼 영역 닿음 |
| `under` | 5 | 아래 | 상자 |
| `attached` | 4 | 붙이기(못·삼각대) | 안 함 |
| `contains`·`filled`·`frozen`·`on_fire` 등 | 14 | 입자·온도 | 안 함 |

- 부분 점수는 **처음엔 거짓이던 조건만** 센다(`task_metric.py:26`). 그래서 "장을 모두 닫아라(`not open`)" 처럼 처음부터 참인 조건 31 개는 분모에는 들어가고 분자에는 **완전 성공 때만** 들어간다(22 과제 — BDDL 문법 기준. 실제 시작 상태로는 29 개 / 21 과제, 5.2절). 이런 과제는 다른 조건을 다 해도 1 이 안 된다 — 학습 보상도 같은 규칙으로 계산해야 한다.
- `exists`(아무 장이나)·`or` 는 선택지별 최대다. GPU 판정 커널은 선택지를 모두 계산해야 한다(팀 `B26/src/sim/engine/core/omni/bddl.h` 가 선택지까지 컴파일한다 — 4절 G7).
- 이 표의 판정은 OmniGibson `object_states`(`inside`·`ontop` 은 광선·접촉, `nextto` 는 상자 거리)를 근사한 것이 아니라 BDDL 문법만 센 것이다. 정확한 판정 규칙은 E3 에서 팀 엔진 `core/omni/states.h`(OmniGibson 을 옮긴 것)를 기준으로 맞춘다.

### 1.5 LIMO + OMX-F 로 되는 과제와 안 되는 과제 — *q 상한은 범위 밖. E1 은 이 숫자를 파서 확인용으로만 다시 냄*

로봇 제약(사실):

| 항목 | 값 | 출처 |
|---|---|---|
| OMX-F 도달 | 400 mm, **가반 하중 100 g(완전히 뻗음) / 250 g(보통)**, 무게 560 g, 그리퍼 0–100°(벌림 폭 문서에 없음) | ROBOTIS OMX-F 사양 https://docs.robotis.com/docs/systems/omx/specifications/hardware/ (2026-10-04 확인) |
| 어깨 높이 | 바닥에서 약 0.25 m: base_link 0.15 + joint1 0.034 + joint2 0.0635 | `training/RL/env/include/limo_omx_model.h`(URDF 생성) |
| 팔 끝 최대 높이 | 어깨 + 링크 합(0.1205 + 0.162 + 0.0287 + 0.0919 = 0.403) ≈ **0.65 m**(곧게 세웠을 때). 물체를 위에서 잡으려면 손목·그리퍼 길이만큼 낮아져 **면 높이 ≈ 0.45 m 안팎 (추정)** | 같은 표에서 계산 |
| 몸통 카메라 | 바닥 0.18 m, 가로 화각 67.9° | `env/README.md` 사양 출처 |
| 베이스 | 0.5 m/s·50 °/s(우리 한도), 바닥 이동만 | `src/robot/og/limo_omx_eval.yaml` |
| 평가기 실행 | LIMO + OMX 로 평가기 띄우기 됨(보기용, `grasping_mode: physical`) | `src/robot/og/eval_with_limo.py` |

BEHAVIOR 장면의 면 높이(위 템플릿 값): 조리대 0.89–0.92 m, 식탁·책상 0.74–0.99 m, 냉장고 1.8–2.2 m(문 손잡이 약 1 m (추정)), 윗장 바닥 1.45–1.49 m, 벽 못 1.73 m, 전등 스위치 0.9–1.48 m, 소파 0.6 m, 협탁 0.34–0.45 m, 커피 테이블 0.29–0.57 m, 쓰레기통 입구 0.27–0.44 m, 바닥 0.

결과(1.3절 표의 q 상한):

| 기준 | 점수 0 이 아닌 과제 | q 상한 1 인 과제 | 100 과제 평균 q 상한 |
|---|---|---|---|
| 엄격(공식 하중 250 g) | 7: 26, 27, 30, 54, 79, **80**, 89 | 80 `clean_up_broken_glass` | **0.027** |
| 느슨(하중 0.5 kg, 높이 +0.15 m, 폭 0.10 m) | 14: 0, 1, 9, 18, 26, 27, 30, 52, 54, 77, 79, 80, 87, 89 | 0 `turning_on_radio`, 1 `picking_up_trash`, 80, 87 `collecting_aluminum_cans` | **0.087** |

막는 이유(엄격, 과제 수 — 한 과제에 여럿): 폭 66, 무게 65, 높이 51, 놓을높이 35, 열기 25, 처음부터 참인 닫힘 22, 닫힌 곳 안 13, 시각 입자 13, 자르기 9, 요리 6, 토글 높이 5, 붙이기 4.

- **되는 과제의 모양**: 바닥이나 낮은 탁자에 있는 작고 가벼운 물체(캔·유리 조각·장난감 인형·과자)를 바닥에 둔 열린 용기(쓰레기통·얼음 통·바구니·장난감 상자)에 넣기, 낮은 버튼 누르기(라디오·모뎀·팩스, 느슨 기준).
- **안 되는 과제와 이유**:
  - 조리대·식탁·책상 위 물체(부엌 과제 거의 전부): 면이 팔 끝 최대 높이(0.65 m) 근처이거나 위다. 카메라 0.18 m 에서는 0.9 m 면 위 물체가 보이지도 않는다(올려다보는 각이 커서 면 가장자리에 가림 (추정)).
  - 냉장고·윗장·전자레인지·오븐·세탁기: 문 손잡이 높이·여는 힘(OMX 하중 250 g) 둘 다 안 됨 (추정). 낮은 장(바닥 가까운 문)만 가능성이 있다.
  - 무거운 것(장작 2.75 kg, 상자 1 kg, 정원 의자 22 kg, 공구함 9 kg)·큰 것(가로 > 0.1 m: 호박·책·접시·타일): 하중·그리퍼.
  - 벽 못·삼각대 붙이기(1.4–1.7 m), 전등 끄기(스위치 0.9–1.5 m).
  - 닦기·뿌리기·요리·자르기: 높이·도구 잡기 문제에 더해 입자·전이 상태가 필요하다(4절 G-c 에서 범위 밖으로 둠).
- 이 판정은 상자 크기·종류 평균 질량·문턱값으로 낸 **(추정)** 이다. 특히 그리퍼 폭(문서에 없음)과 집는 높이는 E0 에서 OmniGibson 에 띄운 LIMO + OMX 로 직접 잰다 — **잰 값과 새 한도는 5.3절**(폭 0.08 → 0.04 m 등, 이 절 숫자는 아직 옛 한도). 판정이 0.1 m 바뀌어도 위 결론(대부분 안 됨)은 그대로다(느슨 기준이 그 감도).

## 2. 스킬 분해와 빈도 — 커리큘럼 순서의 근거

과제 수(100 중, 1.3절 스킬 열)와 목표 조건 수(550 중):

| 스킬 | 과제 수 | 목표 조건 | 이 로봇으로 | 커리큘럼 단계 |
|---|---:|---:|---|---|
| 이동(navigate) | 100 | — | 됨 | B0–B1 |
| 찾기(find, 지도에 없는 물체) | 100 | — | 됨 | B2 |
| 여러 방 이동(문 지나기) | 36 | — | 됨(문턱 확인 필요) | B1 |
| 집기(pick) | 77 | — | 낮고 작고 가벼운 것만 | B4 |
| 놓기 — `inside` | 54 | 232 | 열린 낮은 용기만 | B5 |
| 놓기 — `ontop` | 40 | 103 | 낮은 면·바닥만 | B5 |
| 열기·닫기(관절체) | 38 | 31 + 숨은 것 | 거의 안 됨(낮은 장·상자 뚜껑만 (추정)) | B6 |
| 놓기 — `nextto`·`under`·`touching` | 16·4·3 | 41·5·14 | 바닥 쪽이면 됨 | B5 |
| 무거운 물체(> 2 kg) | 13 | — | 안 됨 | — |
| 닦기(`not covered`) | 11 | 21 | 범위 밖 | — |
| 가열·냉각(cooked 등) | 9 | 30 | 범위 밖 | — |
| 자르기(real/future) | 9 | 33 | 범위 밖 | — |
| 토글 | 7 | 11 | 낮은 버튼만 | B6 |
| 붙이기·채우기·뿌리기 | 4·4·2 | 4·6·4 | 범위 밖 | — |

- 순서 규칙: **모든 과제가 쓰는 것(이동·찾기) → 많이 쓰는 것(집기 77·놓기 inside 54·ontop 40) → 관절체(38) → 나머지**. 이 로봇의 한계 때문에 관절체 이후는 대부분 범위 밖이다.
- 사람 시연 이동 거리 중앙값 26.5 m, 최대 86 m(`metadata/task.jsonl` `distance_traveled`) — 이동·찾기가 과제 시간의 큰 몫이다. 과제 길이 중앙값 342 s(사람).

## 3. BEHAVIOR 2026 에 맞춘 새 커리큘럼

공통:
- **장면은 BEHAVIOR 7 개 집 그대로**(E1 변환기로 가져온 벽·방·문·가구·물체). 학습 집 / 안 쓴 집으로 나눠 평가한다(VLA_INPUT 7절). 제안: 학습 = `house_single_floor`·`house_double_floor_lower`·`restaurant_diner`·`office_cubicles_right`, 평가 = `Rs_int`·`hotel_suite_large`·`house_double_floor_upper` (추정 — 작고 다른 종류의 집을 평가로).
- **물체 집합은 과제 인스턴스에서**: 과제 범위 물체(`inst_to_name`)의 종류·크기·자리. 학습 인스턴스 300 개/과제를 리셋 분포로 쓰고, 공개 평가 0–19 는 평가만(2025 규칙은 평가 인스턴스 수집 금지 — 2026 은 문장이 없지만 같게 둔다).
- **성공은 BDDL 술어로**: 단계마다 성공 조건을 BDDL 문자(`ontop`·`inside`·`nextto`·`toggled_on`·`open`)로 쓰고, GPU 판정 커널(E3)이 OmniGibson 과 같은 규칙으로 계산한다. 이동 단계는 BDDL 에 없는 조건(도착·에임)을 쓴다.
- 지도 처음 상태 C0(다 앎)·C1(부분)·C2(빈 지도)는 지금 G4 그대로(`training/RL/ppo/README.md` G4). 관측 규칙: 평가 때 전역 자세 금지 → 정책 입력은 지금처럼 slam 자세 기준 상대 좌표만(VLA_INPUT 0절).
- 넘어가는 기준·앞 단계 섞기 20 %·SR/SPL/충돌률 지표는 CURRICULUM_APPROACH 4·5절 그대로. 전체 과제 단계에서만 q_score 를 주 지표로 쓴다.

| 단계 | 하는 것 | 장면 | 물체 / 목표 | 성공 (BDDL 또는 이동 조건) | 넘어가는 기준 (추정) |
|---|---|---|---|---|---|
| **B0** 상자 방 이동 | 지금 A0–A2 그대로(회귀 시험) | G1 상자 방 | 컵 1 + 상자 8 | CURRICULUM_APPROACH 1절(0.4–0.8 m·에임 10°·보임·멈춤, 충돌 = 실패) | 지금 기준(A2 ≥ 85 %, 충돌 ≤ 5 %) |
| **B1** 집 안 이동 | 한 방 → 다른 방(문 지나기), 지도 C0 → C1 → C2 | ① `Rs_int`·`hotel_suite_large`(작음) ② `house_double_floor_upper` ③ 나머지 | 목표 = 방(`inroom`) 또는 가구 앞 지점. 과제 인스턴스의 로봇 시작 자세에서 출발 | 목표 방 안 + 목표 지점 0.5 m 안, 충돌 0 | SR ≥ 85 %, 충돌 ≤ 5 %, SPL ≥ 0.7 |
| **B2** 찾기 | 지시된 물체 종류를 지도에 확정하고 카메라에 넣기. 지도 C2(빈 지도) 위주 | B1 집들 | 과제 범위 **옮길 물체 종류**(1.3절 표, 예: `can_of_soda`·`broken_glass`·`toy_figure`·`mango`). 이름은 BDDL synset 의 글 벡터 | 지도 목표 칸 확정(참 물체와 짝) + 몸통·손목 카메라에 보임 | 찾음률 ≥ 85 %, 시간 ≤ 사람 이동 시간 × 1.5 |
| **B3** 다가가기 | 집기 앞 자세로 서기(옛 approach) | 같음 | 같은 물체, **실제 자리**(바닥·낮은 탁자 위) | 손끝 → 물체 ≤ 팔 작업 공간 안(VLA_INPUT "팔이 닿는지" = 1) + 에임 10° + 멈춤 | SR ≥ 85 % |
| **B4** 집기 | 팔 열기, 잡고 들기 | 같음 | 1.5절 엄격 기준을 넘는 물체만(낮고 ≤ 0.25 kg, 폭 ≤ 0.08 m) | 든 상태(OmniGibson `is_grasping` 대응 = E6 규칙) + 처음 받침에서 `not ontop` | SR ≥ 80 % |
| **B5** 놓기 | 들고 이동 → 놓기 | 같음 | 목적지 = 과제의 낮은 용기·면(쓰레기통·얼음 통·바구니·장난감 상자·바닥·커피 테이블) | 과제의 BDDL 문자 그대로: `inside ?x ?용기`·`ontop ?x ?면`·`nextto`·`under` | 조건 만족률 ≥ 80 % |
| ~~**B6** 관절체·토글~~ **(범위 밖, 2026-10-04 범위 확정)** | 낮은 장 문·뚜껑 열고 닫기, 낮은 버튼 | 같음 | 1.3절에서 높이 ≤ 0.5 m 인 관절체·토글(`radio`·`modem`·`facsimile`·낮은 `bottom_cabinet`·`toolbox`) — 수가 적다 | `open` / `not open`, `toggled_on` | SR ≥ 70 % |
| ~~**B7** 전체 과제~~ **(범위 밖 — 대회·q_score 안 함)** | 지시문 → 단계 이어 하기(LLM 이 단계 문장) | 과제 장면, **학습 인스턴스** | 1.5절 느슨 기준 14 과제(80, 87, 1, 0, 54, 26, 18, 89, 77, 30, 52, 79, 27, 9) | 공식 q_score(1.4절 규칙) | 공개 인스턴스 0–19 q 평균 (목표는 6절 질문 1 의 답으로) |

옛 A0–A5 와의 관계:

| 옛 단계 | 처리 | 이유 |
|---|---|---|
| A0, A1(빈 방) | **남김** = B0 앞부분. 비트 같은 회귀 시험으로 둔다 | 환경 코드가 바뀌어도 이전 결과(성공 8,907 등)와 비트가 같아야 한다 |
| A2(상자 가구 8 개) | **남김** = B0 끝. 교사·BC·DAgger 결과(0.944 / 0.946)의 기준선 | 같은 이유 + 새 환경 성능 비교 기준 |
| A3(BEHAVIOR 집 한 방) | **바꿈** → B1 ①(실제 `Rs_int`·`hotel_suite_large` 방 배치) | 상자 생성기 대신 실제 벽·가구 |
| A4(집 전체·문) | **바꿈** → B1 ②③ | 실제 문·방 |
| A5(오래된 지도·잡음) | **바꿈** → B2 의 C1·옮겨진 물체 + E4 잡음 | 인스턴스마다 물체 자리가 다름 = 오래된 지도의 실제 판 |
| 목표 "컵" 하나 | **바꿈** → 과제 범위 물체 종류(BDDL synset) | VLA_INPUT 의 이름 뜻 벡터가 여기서 처음 의미를 가짐 |
| 다가가기 성공 조건(0.4–0.8 m) | **남기되** B3 에서 "팔이 닿는 거리" 로 좁힘 | 다음 단계가 집기 |

- POLICY 1.2 의 역할 나누기(멀리 가기는 `move_robot` `go_to`, VLA 는 마지막 1.5 m)를 B1 에서 정한다: B1 을 RL 교사가 못 넘으면 이동은 `go_to` 에 맡기고 B2 부터 VLA 가 맡는다.

### 3.1 B3–B5 집기·놓기 거르개 표 (단일 원본, 2026-10-04)

판마다 고르는 법(사용자 규칙): ① 쓰는 장면·split 의 **32,000 인스턴스 전부** 중 아래 거르개를 지나는 짝이 하나라도 있는 인스턴스를 고르게 하나, 물체 배치는 인스턴스 그대로(합성 배치 없음) ② 그 인스턴스에서 거르개를 지나는 **집을 물체**(인스턴스의 물체 — RASC PICKS. 장면 파일 물체는 v3 에서 지나는 것이 0 개라 지금은 과제 물체뿐) 하나를 고르게 ③ 그 물체에서 **닿는 놓을 곳**(면·열린 용기·방 바닥) 하나를 고르게 ④ **시작 = 무작위**: 창(12.8 m) 안 가장자리 0.8 m 안쪽의 로봇 중심 칸 중 잡는 자세 칸에 **창 안에서** 닿는 칸(이웃 바닥 높이 차 ≤ 문턱), 무작위 yaw, 몸통 안 닿음, 물체에서 ≥ 1 m — 64 번 뽑아 못 찾으면 표에 미리 찾아 둔 시작(인스턴스 로봇 자세는 쓰지 않음) ⑤ 지시문 = 짝에서(아래). 고르기는 모두 장치 난수(`env_beh.h reset_beh`·`pick_pnp`·`spawn_ok`)이고 CPU 참조판과 비트가 같다. 판마다 쓴 거르개(0 느슨, 1 엄격)를 상태 `I_B_FSET` 에 적는다. 장치 값 `BCurr::strict` 로 엄격만 고른다(기본 느슨).

값: **RASC 는 b1kconv `limits_default`(E0 5.3절)를 `LIMITS` 레코드로 파일에 쓰고, 환경(`bscene_host` `PnpFilter`)은 그 레코드를 읽는다** — 같은 표. 장면끼리 다르면 멈추고, 아래 기본값과 다르면 알린다. "환경" 줄은 환경만 더하는 거르개(`bscene_host.cpp` 집기·놓기 표, 확인은 `pnp_check`).

| 거르개 | 느슨(기본) | 엄격 | 어디서 |
|---|---|---|---|
| 집을 물체 가로 최소 폭 `max_w` | ≤ 0.06 m | ≤ 0.04 m | b1kconv |
| 종류 평균 질량 `max_mass` | ≤ 0.40 kg | ≤ 0.25 kg | b1kconv(평균 없으면 통과, 표시) |
| 물체 바닥 높이 `pick_z` | ≤ 0.50 m | ≤ 0.45 m | b1kconv |
| 위에서 잡기 / 옆 잡기 | 바닥 ≤ `topdown_z` 0.25 m 는 위에서. 그 위는 옆 잡기만: 물체 가운데 → 출발 면 가장자리 ≤ `edge_dist` 0.10 m(출발이 용기·모름이면 느슨만 통과) | 같음(출발이 면이어야) | b1kconv |
| 제외 | 벽·바닥·천장·문·창·계단·카펫·로봇·입자·와일드카드·고정 | 같음 | b1kconv |
| 관절체·고정 물체 집기 | 뺌 | 뺌 | 환경 |
| 닫힌 관절체 안(BDDL init) | 뺌(`PK_IN_CLOSED`) | 뺌 | 환경 |
| 놓을 면 높이 `place_top` / 최소 | 윗면 0.05–0.52 m, 작은 변 ≥ 0.15 m | 0.05–0.48 m | b1kconv |
| 열린 용기 | 윗면 ≤ 0.52 + 0.05 m, 닫힌 관절체·선반 종류 아님 | ≤ 0.48 + 0.05 m | b1kconv |
| 구조물 받침 | 걸레받이(`baseboard`) 뺌 | 같음 | 환경 |
| 놓을 곳 빈 넓이 | 받침 사각형 − 그 위(용기면 안)에 가운데가 있는 물체 바닥 자국 ≥ (물체 가로 + 2·0.02)(세로 + 2·0.02) | 같음 | 환경 |
| 팔 닿는 거리 | 낮은 것(≤ 0.25 m) 몸통 가운데에서 0.38 m(= 옆 0.11 + 0.27, 앞 0.21 + 0.17), 높은 옆 잡기 0.21 m(0.11 + 0.10) | 같음 | b1kconv(다닐 칸) |
| 잡는 자세 칸 | 물체 바닥 자국에서 0.6 m 안, 물체를 바라본 자세로 몸통 안 닿고 **잡는 점 작업 공간**(`omx_workspace_grasp.h`, E0 −0.0119 m)에 물체 상자가 걸림 | 같음 | 환경 |
| 놓을 곳 가장자리에 닿음 | 잡는 자세 칸과 창 안에서 이어진 칸 중 받침 바닥 자국에서 (0, 0.38 / 0.21] m 인 칸이 있음. 바닥은 그 방, 물체에서 0.6–4 m, 3 × 3 칸이 다 빈 자리 | 같음(엄격 문턱으로 이어짐) | 환경 |
| 문턱·바닥 높이 차 | 이웃 칸 바닥 높이(RASC FLOOR_Z) 차 ≤ 0.025 m 로만 이음(다른 층은 장면이 따로라 없음) | ≤ 0.02 m | b1kconv(성분)·환경(창 BFS) |
| 시작 자리 | 위 ④(창 안 닿는 칸, 몸통 안 닿음, 물체에서 ≥ 1 m) | 엄격 문턱 BFS | 환경 |
| 짝 상한 | 집을 물체마다 놓을 곳 ≤ 16(엄격 먼저) | | 환경 |

지시문(VLA_INPUT 6절): 짝의 (집을 것 이름, 출발 받침 이름, 놓을 곳 이름 — 바닥이면 방 종류 "the kitchen floor", 술어 on/in) 조합마다 문장 12 개 = 영어 6 + 한국어 4(학습) + 영어 1 + 한국어 1(평가용 heldout), 물체 이름은 같은 뜻 묶음 안에서 돌려 씀(heldout 문장은 heldout 이름). 벡터는 `training/embed/pnp_instr.py`(오프라인, HF 캐시만)가 vla_v1 지시문과 같은 규칙(영어 PE-Core L/14 글 + P, 한국어 학생 + P)으로 만든 `training/data/pnp_v1/instr128.f16`. 환경은 판마다 문장 하나를 장치 난수로 골라 행 번호를 `I_B_INSTR` 에 쓴다(`BCurr::eval_instr` 면 heldout 에서). 목표 물체 번호(POLICY 1.3 `objects:[…]`)는 시작 조건의 prim 0(집을 것)·1(놓을 곳, 바닥 아니면) — 지도 토큰의 "목표 물체" 칸 값이 둘 다에 켜진다.


## 4. 학습 환경 차이 분석

지금 환경: G1(`training/RL/env`) = 방 하나(반치수 2.5–3.5 m) + 축 정렬 상자 ≤ 8 + 컵, 평면 키네마틱 베이스, OMX 관절 PD, 몸통 사각형 대 상자 충돌, 10 Hz 제어(10 서브스텝), 2048 판 비트 동일·7.3e8 판·스텝/s. 지도 근사(`training/RL/map`) = 128×128 칸 0.10 m 고정 창, 물체 16 칸, 방은 긴 축을 잘라 만든 가짜 방, 진짜 scenemap 과 5.2 비교 통과. 렌더 = 팀 `RenderBatch` 를 읽기만 해서 상자 장면을 그림(`training/BC` G5).

두 길:
- **(가) 우리 커널 넓히기**: G1·지도 방식(판 하나 = 스레드/블록 하나, CPU 참조판 비트 동일) 그대로 장면만 실제 집으로. 접촉은 단순화(상자·키네마틱 잡기).
- **(나) 팀 `sim/engine`**: PhysX 5.6.1 과 비트 동일한 강체·관절체·접촉·보조 잡기·물체 상태·BDDL(`B26/src/sim/engine`, BEHAVIOR_ENGINE_REVIEW). 대신 장면은 **공식 OmniGibson 실행에서 뜬 기록(OVD)** 을 넘겨받아 쓴다(`엔진_자체구현.md` 15.3) → LIMO 로 OmniGibson 을 돌려 판마다 뜨기가 먼저다. 판당 메모리 약 2.9 MB(radio, 관절체 64 링크 칸 고정), GPU 풀이 고정비 3.2–9 ms/호출(`미해결과제.md` B16), 로봇은 R1 Pro 고정 코드(엔진 42 파일).

| # | 빠진 것 | 지금 | 필요한 것 | 길과 비용 (추정, 에이전트 작업일) | 검증 |
|---|---|---|---|---|---|
| G-a | **실제 집 배치**(벽·방·문) | 직사각형 방 1 개, 가짜 방 1–3 | 장면 JSON 의 `walls`·`floors`·`door`·`sliding_door` 물체 상자(회전 포함) + `floor_insseg/semseg`(방 번호·종류) + `floor_trav_0.png`(다닐 곳) | (가) 변환기(E1) 3–5 일 + 커널: 회전 상자(OBB) ≤ 600 개/장면을 장면당 한 벌 공유(판마다 복사 안 함), 균등 격자 가속, 지도 창 12.8 m → 장면 크기(최대 86 m — 창을 로봇 둘레로 옮기는 방식) 6–10 일. (나)는 이동 단계에 과함 | ① CPU 참조판 비트 동일(기존 `env_verify` 방식, `--negative`) ② 우리 래스터 다닐 곳 대 `floor_trav_0.png` IoU ≥ 0.95(몸통 반지름만큼 깎은 뒤) ③ 방 칸 대 `floor_insseg` 일치 ≥ 0.98 ④ OmniGibson LIMO 몸통 카메라 깊이와 같은 자세 50 곳 비교(중앙 절대 오차 ≤ 2 cm, 추정 기준) |
| G-b | **실제 물체**(종류·크기·자리) | 상자 종류 6(의자·탁자·장·쓰레기통·작은 것·컵) | 과제 인스턴스 `inst_to_name` + `tro_state`(자세) + `metadata.json` `bbox_size`×`scale`, 종류 → 이름 뜻 벡터 번호(SigLIP 글 표), 평균 질량 | (가) 변환기에 포함 2 일 + 판 리셋 때 인스턴스 고르기(장치 표에서) 2 일. 인스턴스 30,000 개 × 과제 물체 ≤ 30 개 × 32 B ≈ 29 MB — 장치에 다 올라감 | 변환기가 1.3절 표 숫자(목표 조건 수·물체 종류·높이)를 다시 냄. 인스턴스 표본 20 개를 OmniGibson 에 불러 물체 상자 중심 차 ≤ 1 cm |
| G-c | 입자·전이(닦기·요리·자르기) | 없음 | — | **하지 않음**(1.5절: 이 로봇은 어차피 못 함). 팀 엔진도 시각 입자 14·물리 입자 6 과제는 따로 작업 중 | — |
| G-d | **관절체**(문·서랍·장)·**토글** | 없음 | 장면 문(방 사이 지나기), 낮은 장 문·뚜껑, 버튼 | 이동용 문: (가) 문을 "열림/닫힘 두 상태 상자"로(장면 `floor_trav_open_door` 처럼 열린 채 시작이 기본) 2 일. 손으로 여는 관절체: (가) 1 자유도 회전 문 + 손잡이 점 + 끌기 힘 한도 5–8 일 (추정, PhysX 와 다름) 또는 (나) 팀 Featherstone(비트 동일) — LIMO 기록 뜨기 필요. 토글: 버튼 영역(`metadata.json` `meta_links` 의 togglebutton) 안에 손끝이 들어가면 켜짐 — 팀 `states.h` `toggle_marker_overlap` 을 include 해서 씀 2 일 | 토글: 팀 함수와 같은 입력 비트 동일. 문: OmniGibson 에서 같은 손끝 궤적으로 연 관절 각도 시계열과 비교(오차 ≤ 5°, 추정 기준) |
| G-e | **잡기·물체 동역학** | 팔 끝 0.12 m 안 물체를 닫힘 순간 붙임(scenemap 들기 규칙 흉내, 지도만) | 들기·떨어뜨리기·놓은 뒤 받침에 앉기, 하중 한도 | (가) 키네마틱 잡기: OmniGibson `assisted` 와 같은 판단(두 손가락 사이 광선·접촉) + 하중 한도 넘으면 실패, 놓을 때 아래 면으로 떨어뜨려 앉힘 4–6 일. (나) 팀 보조 잡기·접촉 풀이 — 정확하지만 LIMO 손가락 점 정의가 OmniGibson 에서도 아직 없음(`limo_omx_eval.yaml` 주석: assisted 는 KeyError → physical) | ① CPU 참조판 비트 동일 ② OmniGibson(physical 잡기) 표본 판 50 개: 같은 행동열로 잡힘 여부 일치 ≥ 90 %, 놓은 물체 최종 자리 차 ≤ 3 cm (추정 기준) ③ 음성 대조: 하중 한도를 끄면 무거운 물체 판에서 불일치가 나야 함 |
| G-f | **여러 방 이동·문**·계단 제외 | 방 하나 | 경로 거리(지금은 상자 꼭짓점 보임 그래프, 꼭짓점 32 개) | (가) 큰 장면에서 보임 그래프는 꼭짓점이 수천 개라 안 맞음 → 장면마다 **0.10 m 다닐 곳 격자에서 목표별 거리장을 오프라인으로**(변환기, 목표 후보 = 과제 물체·방) 미리 계산해 표로 3–4 일. 계단·문턱: 다닐 곳에서 뺌(E0 측정 결과로) | 거리장 대 CPU 다익스트라 비트 동일. OmniGibson 다닐 곳 지도의 최단 경로 길이와 표본 200 쌍 상대 오차 ≤ 5 % |
| G-g | **BDDL 술어 판정**(GPU) | 없음(성공 = 거리·각도) | `ontop`·`inside`·`nextto`·`under`·`touching`·`open`·`toggled_on` + 선택지별 q_score(1.4절 규칙) | 팀 `core/omni/bddl.h`(BDDL 파싱·컴파일, 선택지 포함)를 **변환기(오프라인)에서** 써서 과제마다 접지된 조건 표를 만들고, 장치 커널은 조건 표를 상자 판정으로 계산 4–6 일. 정확한 판정(광선·메시)은 팀 `states.h` 의 `OEHD` 함수 include | ① 접지 조건 수 = 1.3절 표 ② OmniGibson 표본 판에서 `env.task.get_goal_option_satisfaction` 과 술어별 일치 ≥ 0.98(상자 근사 때문에 남는 차이는 이유 기록) ③ 음성 대조(처음 참 조건 빼기 규칙을 끄면 q 가 달라져야 함) |
| G-h | **관측 동등** | 10 Hz, 몸통·손목 카메라 상자 장면 256², proprio 56(우리 설계), 지도 토큰 | 평가기 = 30 Hz 행동, 카메라 이름 `eyes`·`wrist_eye`(우리 설정), proprio = `limo_omx_eval.yaml` `proprio_obs`(베이스 속도·팔 관절·손끝·그리퍼), `cam_rel_poses`, **깊이는 지금 설정에 빠져 있음**(`obs_modalities: [proprio, rgb]`) | (가) ① 제어 주기를 평가기에 맞출지(30 Hz) 행동 반복(10 Hz × 3)으로 둘지 결정 ② proprio 를 평가기 이름·순서로 내는 표 ③ 렌더: 실제 물체는 메시가 암호화 → 상자(종류별 색) 렌더로 시작, 메시는 6절 질문 5 3–5 일 | 같은 상태에서 우리 proprio 대 OmniGibson `robot.get_proprioception()` 비트/허용오차 비교, 카메라 자세 대 `cam_rel_poses` ≤ 1 mm, SigLIP 패치 토큰 코사인(우리 상자 렌더 대 OmniGibson RGB) 분포 기록 — 통과 기준은 처음 잰 값으로 정함 |
| G-i | **지도 근사가 실제 집에서** | 128×128 창, 물체 16 칸, 가짜 방 | 큰 장면(창 이동), 실제 방 경계·문, 물체 종류 = 과제 synset, 검출 잡음은 LIMO 보정값 그대로 | (가) 창을 로봇 둘레로 옮기기(격자 원점 이동) 3–4 일, 방은 `floor_insseg` 정답 방 + 드러남 규칙(지금 30 %) 1 일 | map 5.2 비교를 **실제 집 장면으로 다시**: 확정 정밀도·재현율 ≥ 0.9, 위치 50/90 % 차 ≤ 5 cm, 격자 일치 ≥ 0.9(GPU_TRAINING 5.2 표 그대로) |
| G-j | 영역 차이(렌더·잡음) | 싼 렌더 설정, 팀 기본과 패치 코사인 0.72 | 색·조명 흔들기, 깊이 잡음(Dabai σ = 0.006 z²), 관절·바퀴 속도 잡음, OmniGibson RTX 영상 섞기 | (가) 2–3 일. 공식 RTX 영상은 평가기에서 뜨는 수밖에 없음(느림) | OmniGibson 영상 대 우리 영상 토큰 코사인 분포, 같은 정책 성공률 차(OmniGibson 몇 판) |

정리:
- **이동·찾기·다가가기(B1–B3)는 (가)로 충분하다.** 필요한 것은 실제 배치·물체 상자·문·거리장·지도 근사 확장이고, 이것은 지금 비트 동일 방식 그대로 된다.
- **집기·놓기·관절체(B4–B6)는 (가)의 키네마틱 잡기로 먼저**, OmniGibson 표본 대조(G-e ②)가 기준 미달이면 (나)로 옮긴다. (나)를 쓰려면 LIMO 로 OmniGibson 을 돌려 장면 기록(OVD)을 뜨는 길(팀 15.3 적재기)이 먼저 열려야 한다.
- 팀 코드는 지금처럼 **include 로 읽기만** 한다(BC 의 `RenderBatch` 와 같은 방식). 서브모듈은 고치지 않는다.

## 5. 환경 개선 계획 (학습 전에)

순서는 앞 단계의 통과 기준이 다음 단계의 입력이 되게 짰다. 각 단계는 CPU 참조판 비트 동일 + `--negative` 음성 대조(지금 G1·지도 규칙)를 기본으로 하고, 그 위에 OmniGibson 대조를 더한다.

| 단계 | 내용 | 통과 기준 | 비용 (추정) |
|---|---|---|---|
| **E0** 측정·결정 — **됨(5.3절)** | OmniGibson 에 LIMO + OMX 를 띄워(`src/robot/og/eval_with_limo.py`) ① 그리퍼 최대 벌림 폭 ② 위에서 잡을 수 있는 최고 면 높이·옆에서 잡는 높이 ③ 7 장면 방 사이 바닥 턱·정원 지형을 넘는지(과제 시작 자세에서 목표 방까지 `go_to`) ④ 하중(무게를 바꾼 상자 들기) 을 잰다. 1.5절 표를 잰 값으로 다시 낸다. 6절 질문에 답을 받는다 | 잰 값 4 개 + 갱신된 되는 과제 목록, 사용자 결정 | 2–3 일, GPU 는 평가기 가벼운 사용 |
| **E1** 장면·과제 변환기 (**됨**, 5.2절) | **Rust 오프라인 도구** `training/RL/tools/b1kconv`(처음 가칭 `scene_import`). 입력: 장면 `_best.json`·`layout/*.png`, 과제 `problem0.bddl`, 인스턴스 템플릿·`tro_state`(학습 300 + 공개 20), 물체 `metadata.json`, `avg_category_specs.json`. 출력: 작은 장치 형식 하나(아래) | ① 1.3절 표 숫자(목표 조건 수, 옮길 물체 종류, 높이)를 그대로 다시 냄 ② 출력 파일을 다시 읽어 원본 JSON 과 값 비교 0 다름 ③ 깨진 입력(물체 빠짐)에 오류로 멈춤(음성 대조) | 4–6 일 |
| **E2** 실제 집 이동 장면 (**됨**, 5.4절 — OmniGibson 대조 ③ 은 E7 로 남김) | G1 에 `stage` 하나 더(B1): 장면 공유 OBB + 방 격자 + 문 + 거리장, 리셋 = 인스턴스 로봇 시작 자세. 지도 근사 창 이동·실제 방. A0–A2 비트 그대로 | ① CPU = GPU 비트 동일, `--negative` 실패 ② A0/A1/A2 결과 비트가 지금과 같음 ③ G-a·G-f 의 OmniGibson 대조(IoU ≥ 0.95, 경로 길이 ≤ 5 %, 깊이 50 곳) ④ 처리량: N 4,096 에서 판·스텝/s 기록(기준: A2 의 0.172 ms/스텝의 5 배 안 (추정)) | 8–12 일 |
| **E3** 물체·BDDL 판정 | 과제 물체(종류·상자·질량)를 판에 싣고, 접지 조건 표 + 술어 커널 + q_score. 찾기(B2) 성공 판정 | ① 비트 동일 ② G-g ②: OmniGibson 표본 판 술어 일치 ≥ 0.98 ③ 지도 근사 5.2 비교를 실제 집에서 다시(G-i) | 6–8 일 |
| **E4** 관측 동등 | proprio 를 평가기 이름·순서로, `cam_rel_poses`, 30 Hz 결정(행동 반복 또는 3 배 스텝), 깊이 켜기(평가 설정 `obs_modalities` 에 depth — `src/robot/og` 쪽 일), 상자 렌더에 실제 물체 종류 색 | G-h 표의 비교(proprio·카메라 자세 허용오차 안), 렌더 토큰 코사인 분포 기록 | 4–6 일 |
| ~~**E5** 관절체·토글~~ **(범위 밖 — B6 을 안 하므로. 이동용 문 두 상태만 E2 에서)** | 이동용 문 두 상태, 낮은 장 문 1 자유도, 토글 버튼 영역(팀 `toggle_marker_overlap`) | 비트 동일, 토글은 팀 함수와 같은 입력 비트 동일, 문 각도 OmniGibson 대조 ≤ 5° | 5–8 일 |
| **E6** 잡기 물리 | 키네마틱 잡기(손가락 사이 판단 + 하중 한도) + 놓기 떨어뜨림 + 받침 앉힘 | G-e ①–③ | 4–6 일 |
| **E7** OmniGibson 표본 대조 묶음 | E2–E6 대조를 한 도구로: 과제마다 인스턴스 몇 개 × 같은 행동열(스크립트 정책)을 우리 환경과 OmniGibson(LIMO)에 넣어 상태·술어·관측을 비교. map 5.2 의 `map_cmp` 방식 | 단계마다 위 기준을 한 표로, 음성 대조 포함. 이 표가 통과해야 B1 학습을 연다 | 3–5 일 |

- 학습 재개 시점: **E2 + E3 + E7(이동·찾기 부분) 통과 → B1·B2 교사 학습**. E4–E6 은 그동안 이어서 하고, 통과하면 B3–B5(B6·B7·E5 는 범위 밖).
- 합 약 36–54 에이전트 작업일 (추정). 2026 마감(10-17) 전에 끝나지 않는다(6절 질문 2).

### 5.1 변환기(E1) 출력 형식 — 제안

Python 은 쓰지 않는다. 입력이 모두 평문 JSON·PNG·BDDL 이라 Rust(`serde_json`, `png`)로 읽는다. 메시가 필요해지면(6절 질문 5) 그때만 OmniGibson 쪽 일회성 Python 추출을 쓴다.

```
RASC v1  (little endian, 64 B 정렬, 머리에 구조체 크기 — 빌드가 다르면 읽기 거절: 팀 ENGSCN1 규칙과 같음)
머리       : 장면 이름, 원점·크기(m), 격자 해상도 0.10 m, 개수들
정적 OBB   : [n_static] {center xyz, half xyz, yaw, 종류 id, 방 id, 플래그(벽/바닥/가구/문/계단)}   — 장면당 한 벌, 판끼리 공유
방 격자    : u8 [H][W] 방 id (floor_insseg 0.01 m → 0.10 m 다수결), 방 종류 표(room_categories.txt 번호)
다닐 곳    : 비트 [H][W] (floor_trav_0 를 몸통 반지름으로 깎음), 문 칸 표
거리장     : u16 [n_goal][H][W] cm (목표 = 과제 물체 자리·방 중심, 인스턴스마다 다르면 판 리셋 때 고름)
종류 표    : [n_cat] {이름(synset·OmniGibson 종류), 평균 질량, 글 벡터 번호}
과제 표    : [n_task] {장면, 사람 길이, 제한 스텝, 접지 조건 오프셋}
조건 표    : [n_lit] {술어, 인자 칸 2, 부정, 선택지 번호}
인스턴스   : [n_inst] {과제, 로봇 시작 xy·yaw, 물체 칸 [≤ 32] {종류, OBB, 질량, 처음 받침 칸}}
```

- 크기 (추정): 가장 큰 `house_single_floor` 도 0.10 m 격자 863 × 863 ≈ 0.75 MB(방) + 0.09 MB(다닐 곳), OBB 600 × 48 B ≈ 29 KB. 인스턴스 32,000 × 약 1.1 KB ≈ 35 MB. 장치 16 GB 에 여유.
- 변환기는 1.3절 표(목표 조건 수·종류·높이·q 상한)를 텍스트로도 낸다 → 이 문서 숫자를 다시 만드는 것이 E1 통과 기준 ①.

### 5.2 E1 상태 (2026-10-04, 잰 값)

**됨.** 도구는 `training/RL/tools/b1kconv`(Rust 변환기 + 생성 C 헤더 `cpp/rasc_format.h` + C++ 로더 `cpp/rasc.h`·시험 `cpp/rasc_test.cpp`), 형식·사용법은 그 [README](../../training/RL/tools/b1kconv/README.md). 가칭 `training/RL/scene_import` 대신 `urdf2hdr` 옆에 두었다. 입력은 평문 JSON·PNG·BDDL 뿐이고 암호화 USD·Python 은 쓰지 않았다.

- 출력: 장면당 RASC 파일 하나(위 5.1 제안에서 거리장만 뺌 — E2 에서). v1 은 7 장면 합 15.3 MB, 집기·놓기 표를 더한 v2 는 24.1 MB, E0 한도·바닥 높이의 v3 는 22.0 MB(HSF 9.86·HDL 5.99·RD 1.71·HDU 1.46·OCR 1.19·HSL 0.92·RS 0.82 MB), 인스턴스 32,000 개(대부분이 인스턴스 자세 구역). 변환 7 장면 약 4 s·최대 RSS 약 240 MB(CPU).
- **기준 ① 통과**: 엄격 7 과제 [26, 27, 30, 54, 79, 80, 89]·q = 1 은 80·평균 0.0267, 느슨 14 과제·q = 1 [0, 1, 80, 87]·평균 0.0874, 목표 문자 550(술어별 수도 1.4절 표와 같음), `not open` 31 개 / 22 과제, 2절 스킬 과제 수 전부, 1.2절 장면 표(물체·문·관절체·벽·다닐 넓이·계단), 1.3절 표 100 줄 × 10 열 **0 다름**. 막는 이유 과제 수도 1.5절과 같다(1.5절 줄에 빠진 것: `?` 7·선반 4·`contains` 4·`frozen` 2·`on_fire` 1·`filled` 1).
- **기준 ② 통과**: 파일을 다시 읽어 원본 JSON·BDDL 과 842,695 값 비교 0 다름(`b1kconv verify`). C++ 로더도 변환기가 쓴 바이트(구역별 FNV-1a)와 같은 것을 보고, 왕복·불변식(물체 수, AABB, 방 덮기, 술어 인자 유효)을 7 장면 모두 통과.
- **기준 ③ 통과**: 인스턴스 파일에서 과제 물체 하나 지움, 목표가 없는 물체를 가리킴, 배치 해시 손상 → 모두 오류로 멈춤(`b1kconv negative` 3/3, C++ 손상 4 종 거절).

**집기·놓기 후보 표(B3–B5 입력, 형식 v3, 2026-10-04 — E0 잰 값 반영).** 범위 확정(집기·놓기만)에 따라 RASC 에 LIMITS·PICKS·PLACES·PAIRS·PNP_RANGES 를 더했고(v2), E0(5.3) 값이 들어와 v3 에서 한도 구조를 바꾸고 칸마다 바닥 높이(FLOOR_Z)를 더했다. 한도는 매개변수다(`--outer`/`--inner k=v,…`). 기본은 E0: 바깥(포함) = 느슨 `pick_z 0.50·place_top 0.52·max_mass 0.40·max_w 0.06·threshold 0.025`, 안(플래그) = 엄격 `0.45·0.48·0.25·0.04·0.02`, 공통 도달 옆 0.27·앞 0.17 m(몸통 가장자리 밖; 가장자리는 joint1 축에서 옆 0.11·앞 0.21 m, 칸 중심 = joint1 축으로 봄), 위에서 잡기 ≤ 0.25 m, 그 위는 옆 잡기만이고 면 가장자리에서 ≤ 0.10 m.
- 집을 물체: 고정 아님·벽 등 아님·바닥 높이 ≤ `pick_z`·종류 평균 질량 ≤ `max_mass`·작은 가로 변 ≤ `max_w`. 바닥 높이 > 0.25 m 면 옆 잡기: 받침이 면이면 물체 중심에서 면 가장자리까지 ≤ 0.10 m 여야 들고(`edge_d` 기록), 받침이 용기·모름이면 넣되 `EDGE_UNKNOWN`(안 한도에서는 빠짐). 놓을 곳: 열린 용기(`inside`, 윗면 ≤ `place_top` + 0.05), 면(`ontop`, 윗면 0.05–`place_top`, 작은 변 ≥ 0.15 m), 방마다 바닥. 닫힌 관절체·선반 칸은 뺌.
- 다가가기: `floor_trav_0` 0.10 m 칸 중, 물체(받침) 상자 바닥 사각형에서 0.38 m(위에서 잡기 = max(0.11 + 0.27, 0.21 + 0.17)) 또는 0.21 m(옆 잡기 = 0.11 + 0.10) 안인 빈 칸 수와 연결 성분. 성분은 이웃 칸의 바닥 높이 차가 문턱(바깥 0.025·안 0.02 m)보다 크면 끊는다. 바닥 높이 = 칸의 방을 `in_rooms` 로 가진 바닥 물체 윗면. 끊긴 칸 이음: HSF 0 / 24(바깥 / 안 문턱), HSL 12 / 12, 나머지 0. 짝의 "닿음" 비트: 같은 성분(바깥 문턱), 안 한도, 로봇 시작도 같은 성분, 안 문턱으로 로봇·물체·목표가 같은 성분.
- **장면 파일(`_best.json`)에는 집을 물체가 없다**(가구뿐). 집을 물체는 모두 과제 물체(인스턴스 자세)다.

장면별 수, v2(옛 한도: 1.5절 느슨/엄격, 도달 0.40 m, 문턱 없음) → v3(E0 한도). "닿음" = 로봇 시작 칸에서 닿는 짝이 있는 인스턴스, "엄격" = 그중 안 한도·안 문턱까지.

| 장면 | 인스턴스 | 장면 받침 면 / 용기 | 집을 물체 수 | 집을 과제 물체가 있는 인스턴스 | 닿음 | 엄격 | 짝 수 |
|---|---:|---|---:|---:|---:|---:|---:|
| HSF | 10,880 | 35 / 1 → 19 / 1 | 11,895 → 5,615 | 4,187 → 2,378 | 3,839 → 1,719 | 2,444 → 661 | 112,366 → 45,783 |
| HDL | 10,240 | 5 / 0 → 4 / 0 | 8,640 → 1,049 | 2,880 → 1,049 | 2,223 → 409 | 1,246 → 0 | 44,430 → 2,596 |
| HDU | 3,200 | 6 / 0 → 5 / 0 | 639 → 4 | 639 → 4 | 615 → 0 | 0 → 0 | 1,900 → 12 |
| RD | 2,560 | 0 / 0 → 0 / 0 | 1,968 → 999 | 732 → 412 | 712 → 329 | 392 → 0 | 2,951 → 1,650 |
| RS | 1,920 | 4 / 1 → 3 / 1 | 1,592 → 230 | 960 → 194 | 960 → 193 | 320 → 193 | 7,000 → 690 |
| HSL | 1,600 | 9 / 0 → 6 / 0 | 3,200 → 121 | 960 → 121 | 956 → 101 | 319 → 0 | 31,003 → 726 |
| OCR | 1,600 | 6 / 0 → 6 / 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 합 | 32,000 | | 27,934 → 8,018 | 10,358 → 4,158 | 9,305 → 2,751 | 4,721 → 854 | 199,650 → 51,457 |

- 가장 크게 줄인 것은 폭(0.10/0.08 → 0.06/0.04 m)이다. 엄격까지 남는 과제는 4 개뿐: 27 `sorting_household_items`(칫솔·치약 320), 73 `organizing_school_stuff`(펜·연필 319), 82 `store_batteries`(배터리 193), 29 `clean_up_your_desk`(스테이플러 22). 느슨에서 닿는 과제는 14 개(9, 20, 21, 27, 29, 30, 38, 62, 71, 73, 79, 80, 82, 95; 과제별 수는 `<장면>.pnp.tsv`). 캔(1, 52, 87)·장난감 인형(54)은 폭으로 빠지고, 유리 조각(80)은 느슨에서만 남는다.
- 같은 E0 한도로 1.3절 q 상한을 다시 내면(문서 상자 높이 규칙) 엄격 3 과제 [27, 30, 89]·평균 0.0075, 느슨 7 과제 [0, 27, 30, 77, 79, 80, 89]·평균 0.0283(대회 점수는 범위 밖, 참고).
- 바깥(마당 등 방이 아닌 칸)에는 바닥 받침이 없어 거기 물체(예: 39, 60, 75)는 닿는 짝이 0 이다 — E2 에서 바깥 영역을 방처럼 다룰지 정한다.


이 문서에서 고칠 점(잰 값):
- 1.4절 "처음부터 참인 `not open` 31 개(22 과제)" 는 **BDDL 문법만 센 값**이다. 실제 시작 상태(BDDL `:init` 과 인스턴스 320 개의 관절 값 둘 다)로 보면 29 개 / 21 과제가 참이다. 14 `carrying_in_groceries` 의 `not open car`(관절 2.53, 열린 채 시작)와 29 `clean_up_your_desk` 의 `not open laptop`(init `(open laptop)`, 관절 2.40)은 처음에 거짓이라 부분 점수에 들어간다. BDDL `:init`(닫힌 세계)으로 처음부터 참인 목표 문자를 모두 세면 77 개다(예: 52 의 `not inside` 10, 43 의 `not real` 5 — 학습 보상도 이 규칙을 따라야 함).
- 1.2절 HDU 방 6 → 방 PNG 에는 5 개. 6 번째 `bathroom_1` 은 문 하나의 `in_rooms` 에만 있다.
- 1.3절 15 `bringing_in_wood` 의 "? 3" 은 공개 `-partial_rooms` 템플릿만 옛 `firewood` 이고 BDDL·전체 템플릿·인스턴스는 `plywood` 라서 생긴 것이다(규칙대로 다시 내면 같은 값). 장치 형식은 전체 템플릿으로 채웠다.
- 1.3·1.5절 높이는 회전·상자 오프셋을 무시한 `루트 z ± 높이/2` 다. 세계 상자로 다시 재면 엄격은 그대로, 느슨은 13 과제·평균 0.0852(9 `putting_up_Christmas_decorations_inside` 가 0 이 됨).
- 새로 안 것: **2025 과제(0–49) 학습 인스턴스 15,000 개에는 일반 로봇 자세(`robot`)가 없고** R1Pro·Fetch·R1·Stretch·Tiago 별 자세만 있다. 평가기는 `robot` 이 없으면 로봇 모델 이름을 찾으므로(`eval/evaluator.py` 483–490) LIMO 로 이 파일을 그대로 띄우면 KeyError 다. 변환기는 R1Pro 자세를 넣고 표시했다. 공개 평가 2,000 개는 모두 `robot`. 또 `cook_a_frozen_pie` 공개 인스턴스에는 `tray` 가 없다(템플릿 자세로 채우고 표시).
- 다닐 곳 중 방이 아닌 칸이 많다(RD·OCR 약 50 %, 마당·주차장 등) — E2 의 방 기반 목표·지도 근사는 "방 아님" 칸을 다뤄야 한다.

### 5.3 E0 측정 결과 (2026-10-04, 잰 값)

헤드리스 OmniGibson(빈 장면, `limo_omx_eval.yaml` 제어기, 물리 120 Hz·행동 30 Hz)에 LIMO + OMX-F 를 띄워 쟀다. 범위 확정(집기·놓기만)에 따라 토글·관절체는 재지 않았다. 스크립트·원자료는 `src/robot/og/e0/`(`kin.py` 평면 기구학·몸통 충돌, `sim_grasp.py` 잡기·하중·문턱, `sim_view.py` 카메라, `finger_gap.py` 손가락 틈, 결과 `results/*.json|txt`). 시험 방식: 0.02 m 각 기둥 위 상자 → 열고 다가감 → 닫음 → 0.08 m 들기 → 2 s 버팀. "손에 듦" = 들린 뒤에도 손가락 사이.

**잡는 점(eef) 고침.** `eef_link` 는 `omx_end_effector_link`(= `omx_link5` + (0.09193, −0.0016, 0), URDF) 에 붙어 있었는데 이것은 **손가락 끝 바로 앞**이다. 손가락은 축(link5 x 0.0295)에서 돌아 열수록 끝이 뒤로 물러난다(닫힘 0.0945 → 45° 0.082 → 60° 0.070). 그래서 그 점에 둔 물체는 손가락 사이에 들지 않는다(시뮬 g = 0.0919 에서 3/3 실패 — 앞선 잡기 실패의 원인). 물체 중심을 link5 x 0.055–0.092 로 옮겨 잡아 보면 0.065–0.085 가 되고, **0.080 으로 정했다**(10–40 mm 상자 4/4). 오프셋 = `omx_end_effector_link` 에서 손가락 축(eef z, URDF x) 으로 **−0.0119 m**, y −0.0016 은 두 손가락 축의 가운데(0.0075, −0.0108)라 그대로.
- 소스: `limo_omx_source_config.yaml` 의 `eef_vis_links.offset.position = [-0.0119, 0, 0]`(다시 가져오면 USD 에 들어감). 설치된 USD 는 `behavior-2026` 아래라 고치지 않고, `src/robot/og/limo_eef_fix.py` 가 불러온 무대에서 eef 고정 관절 `localPos0` 를 옮긴다(`eval_with_limo.py`·E0 스크립트가 부름). 확인: `eef_link`·`get_eef_position()` 이 link5 프레임 (0.0800, −0.0016, 0).
- 영향: proprio `eef_0_pos` 가 손가락 축으로 0.0119 m 바뀐다. 시뮬 기록·FK 표(scenemap `sm_robot_fk` 의 손, `training/RL` 의 OMX 작업 공간 표)가 `omx_end_effector_link` 를 쓰면 잡는 점은 거기서 −0.0119 m 다.

**그리퍼 쥐는 힘(시뮬).** 그리퍼 드라이브(acceleration 형, `isaac_kp` 1e4)는 쥐는 힘이 ~0 이라(잰 관절 힘 < 1e-4 N·m) 어떤 물체도 못 들었다. `max_effort` 로는 힘을 자를 수 없다(0.52 를 넣으면 0). kp 4e5 = 0.4–0.6 N·m(실제 XL330-M288-T 정지 토크 0.52 N·m 와 같음) 는 100 g 을 들고 250 g 을 놓친다. kp 1e6 = 1.0–1.4 N·m 는 250 g 을 들고 500 g 을 놓친다(공식 가반 하중 250 g 과 맞음) → **`limo_omx_eval.yaml` `isaac_kp` 를 1e6 으로**. 시뮬 손가락 충돌 모양은 볼록 껍질이라 실제 손가락 안쪽의 오목한 주머니가 없다(같은 힘으로 덜 쥠). 1e6 에서 미믹 오차 ~4°, 빈손 닫힘에서 안 터짐. 시뮬 팔 관절 드라이브는 최대 1000 N·m 라 포화되지 않는다(0.5 kg 을 뻗어 들어도 위치 오차 < 1e-4 rad) — **하중 한계는 실제 모터로 따진다**. OmniGibson 은 로봇 링크 중력을 끄므로 시뮬 관절 힘은 짐 몫만이다(뻗은 자세 0.1 kg: 시뮬 joint2 0.350 N·m, 계산 0.358).

| 항목 | 시뮬(OmniGibson) | 공식 사양 | 고른 값 |
|---|---|---|---|
| 그리퍼 각 | joint_1 0–1.745 rad, joint_2 미믹 | 0–100° | 0–100° |
| 그리퍼 폭(쥘 수 있는 물체 가로) | 10–40 mm 성공, 50–80 mm 실패(볼록 껍질 V 자 손가락이 밀어냄). 40 mm 때 joint_1 0.42 rad | 문서에 없음(OpenMANIPULATOR-X 는 20–75 mm 지만 다른 그리퍼) | **엄격 ≤ 0.04 m**(시뮬 확인), **느슨 ≤ 0.06 m**(실제 메시: 30° 에서 끝 틈 62–67 mm, 손끝이 아직 물체 앞) |
| 손끝 끝 최대 벌림 | — | — | 실제 메시 끝끼리 60° 110 mm·75° 138 mm(손가락이 옆을 향해 잡기엔 못 씀) |
| 위에서 잡기(도구 수직) 잡는 점 높이 | 옆 몸통 가장자리 +0.03/+0.08 m: 0.15 ✔, 0.22 ✔(들기 여유 0.03–0.045), 0.26 IK 없음; 바닥(0.03): +0.08 ✔·+0.03 ✗; 앞: +0.03 에서만 0.15 ✔(라이다·카메라 위로 팔꿈치가 걸림) | — | 바닥 ~ **0.25 m**(옆, 가장자리 +0.02–0.15 m). 앞에선 +0.05 m 안만 |
| 옆에서 잡기(도구 수평) 잡는 점 높이 | 옆 +0.03/+0.08: 0.40 ✔, 0.45–0.51 손에 듦(들기 여유 0–0.015 m); 앞 +0.03: 0.30·0.40 ✔, 0.45·0.48 손에 듦(0.03 m 들림) | — | 잡는 점 ≤ **0.48 m**(들기 여유 ≥ 1 cm), 기구학 최대 0.53(옆 가장자리) |
| **집는 면 높이**(물체 높이 5 cm, 잡는 점 = 면 + 2.5 cm) | 위 줄에서 | 문서에 없음 | **엄격 0.45 m(그대로)**, 느슨 0.50 m(0.60 → 내려감) — 면 가장자리가 몸통 가장자리에서 ≤ 0.10 m(옆)·0.05 m(앞) 일 때 |
| **놓는 면 높이**(들 필요 없음) | 잡는 점 0.51 까지 손에 듦 | — | **엄격 0.48 m**(0.50 → 내려감), 느슨 0.52 m(0.62 → 내려감) |
| 수평 도달(잡는 점, 몸통 가장자리 밖) | 기구학: 앞 0.174–0.179 m·옆 0.274–0.279 m(높이 0.25, 도구 −10°); joint1 축에서 0.384–0.389 m | 완전히 뻗음 400 mm | 옆 **0.27 m**, 앞 **0.17 m**. 높이 0.45 의 옆 잡기는 가장자리 +0.10 m 까지 |
| 하중(실제 모터 정지 토크 기준, 정적 계산) | 그리퍼 kp 1e6: 위에서 0.1 kg ✔·0.25 ✗, 수평 0.5 kg ✔·1.0 ✗. 팔은 포화 없음 | **250 g(보통)·100 g(완전히 뻗음)**. 모터: joint1–3 XL430-W250-T 1.5 N·m @12 V, joint4–5·그리퍼 XL330-M288-T 0.52 N·m @5 V | **엄격 0.25 kg**(공식), **느슨 0.40 kg**(0.5 → 내려감). 계산: 가까운 위에서 잡기 정지 토크 한도 0.63 kg·절반 0.23 kg, 높이 0.45 수평 잡기 0.44 kg(joint4)·절반 0.20 kg, 완전히 뻗음 0.24 kg·절반 0.045 kg(joint2) |
| 문턱(높이 h 판, 0.3 m/s 전진) | ≤ 0.025 m 넘음(기울기 ≤ 7.9°), ≥ 0.03 m 앞 범퍼가 막힘 | 최소 지상고 24 mm, 등판 25°(4 륜 차동) | **≤ 0.02 m** 넘음(지상고 24 mm 보다 낮게). 장면별 턱은 E1/E2 변환기가 이 값으로 다닐 곳에서 뺀다 |
| 몸통 카메라(높이 0.18 m, 시뮬 320×240·H-FOV 67.9°) 로 5 cm 물체 보기 | 면 0.30: 앞 0.30–1.20 m 에서 보임, 0.05 m(잡을 거리)에서 안 보임. 면 0.45: 1.2 m 에서 겨우(31 화소), 0.3–0.7 m 안 보임. 면 0.60·0.75: 안 보임 | Dabai 깊이 H 67.9°·V 45.3°, 컬러 H 71°·V 43.7° | 낮은 가구(≤ 0.30 m)는 멀리서 몸통 카메라로 찾고, 잡을 거리에선 손목 카메라로 본다 |
| 손목 카메라(보는 자세: 잡는 점 0.50 m·수평, 카메라 38° 아래) | 면 0.30·0.45 를 앞 0.05 m 에서 보임(0.45 면 상자 4,833 화소), 0.60·0.75 안 보임 | 사양 없음(640×480 USB) | 집기 직전 확인은 손목 카메라로, 면 ≤ 0.45 m |

**엄격·느슨 기준이 바뀌는가.** 바뀐다. 1.5절·b1kconv 기본 한도를 이렇게 바꾼다(과제 표 재계산은 E1 변환기 몫 — **b1kconv v3 에 반영함**, 5.2절 표):

| 한도 | 엄격 옛 → 새 | 느슨 옛 → 새 |
|---|---|---|
| 집는 면 `pick_z` | 0.45 → **0.45** | 0.60 → **0.50** |
| 놓는 면 `place_top` | 0.50 → **0.48** | 0.62 → **0.52** |
| 무게 `max_mass` | 0.25 → **0.25** kg | 0.5 → **0.40** kg |
| 물체 작은 가로 `max_w` | 0.08 → **0.04** m | 0.10 → **0.06** m |
| 팔 닿는 거리(몸통 가장자리 밖) | 0.40(추정, joint1 축 기준) → 옆 **0.27**·앞 **0.17** m(joint1 축 기준 0.38) | 같음 |
| 문턱 | — → **0.02** m | — → 0.025 m(시뮬 넘음) |
| 토글 | 재지 않음(범위 밖) | 재지 않음 |

- 가장 크게 바뀌는 것은 **폭**(0.08 → 0.04 m): 캔(지름 ~0.066 m) 같은 물체는 엄격에서 빠지고 느슨에서도 경계다. 높이는 엄격 그대로지만 "옆에서 수평으로, 면 가장자리 가까이" 라는 조건이 붙는다(위에서 잡기는 면 ≤ 0.22 m 안팎).
- 결론(되는 것 = 바닥·낮은 탁자(≤ 0.45 m)의 작고(≤ 4 cm) 가벼운(≤ 250 g) 물체)은 그대로고, 되는 집합은 더 작아진다.

출처(2026-10-04 확인): ROBOTIS OMX 하드웨어 사양 https://docs.robotis.com/docs/systems/omx/specifications/hardware/ · https://ai.robotis.com/omx/hardware_omx.html (도달 400 mm, 가반 하중 100 g 완전히 뻗음/250 g 보통, ID 11–13 XL430-W250-T, ID 14–16 XL330-M288-T, 그리퍼 0–100°, 그리퍼 폭은 없음); XL430-W250-T 정지 토크 1.4 N·m @11.1 V·1.5 N·m @12 V https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/ ; XL330-M288-T 0.52 N·m @5 V https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/ ; OpenMANIPULATOR-X 그리퍼 20–75 mm(비교용) https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/ ; AgileX LIMO 사용 설명서(322×220×251 mm, 최소 지상고 24 mm, 무게 4.8 kg, Dabai FOV) https://github.com/agilexrobotics/limo-doc/blob/master/Limo%20user%20manual(EN).md .

### 5.4 E2 상태 (2026-10-04, 잰 값)

**됨(OmniGibson 대조 ③ 빼고 — E7 로).** 코드: `training/RL/env`(`bscene.h` 장치 장면, `bscene_host.{h,cpp}` 호스트 묶음 만들기, `env_beh.h` B1–B5 판, `omx_workspace_grasp.h`, 도구 `bscene_check`·`pnp_check`·`env_verify --stage 3`), `training/RL/map`(`map.h`·`map_tok.h` BEHAVIOR 갈래·거리장, `map_verify --stage 3`, `nav_tradeoff`, `map_bench MAP_STAGE=3`), `training/embed/pnp_instr.py` → `training/data/pnp_v1`. 사용법·API 변화는 [env README](../../training/RL/env/README.md)·[map README](../../training/RL/map/README.md) "E2" 절.

무엇을 붙였나:
- **장면 7 개를 장치에 한 벌**(모든 판이 같이, 장치 61.2 MB — 시작 조건 표 13.4 MB, 집기·놓기 짝마다 창 닿는 칸 비트 2–4 KB, 장면 격자·상자·성분): 정적 회전 상자(문 상자 빼고 열린 채, 카펫·납작한 것·바닥 덮개 빼고, **벽 상자에서 문 자리 잘라 냄 + 인방** — 벽은 메시 구멍이 없는 상자라 안 자르면 문이 막힘, 문 자리 68 곳 자름), 1 m 묶음, 방 격자, 문(양쪽 방은 문 얇은 축 ±0.5 m 의 방 칸), 지도 띠 [0.05, 0.50] 점유 래스터, 로봇 중심 칸 성분(느슨·엄격 문턱). 판 상태는 작다(SoA float +9·int +6, 지도 판마다 `BMapEnv` ~0.5 KB + 거리장 조각 1 KB).
- **판 좌표 = 창 좌표**: 판마다 12.8 m 창(지도 128 × 128 칸)을 시작·목표가 0.8 m 여유로 들어가게 장면 칸 경계에 맞춰 둔다. 몸통 충돌·벽 광선 16·보임·지도 깊이 광선·부재 확인·보임 점이 정적 상자(묶음 걷기) + 과제 물체 상자로. 방 토큰 = 장면 방 격자(창 안 방 ≤ 16, 드러냄 규칙 그대로), 문 = 장면 문(아는 쪽 방이 다 드러나면). 지도 물체 9 = 목표 + 과제 물체 + 창 안 가구, 이름 = vla_v1 행(과제 물체 synset·가구 종류 → 이름, 없으면 상위어 "furniture"), 확신도 = name128·app128 코사인 표(C++ 로 다시 냄, 파이썬 표와 시뮬 6 행 차 ≤ 4.5e-8), 생김새 행 = 상자(가정 — 종류별 생김새 행은 E4). C0/C1 처음 지도 = 장면 띠 점유 래스터.
- **B1**(방 표): 인스턴스 로봇 시작(2025 과제는 R1Pro 자세) → 다른 방의 바닥 가구 앞(0.30–0.60 m) 점. 성공 = 목표 방 안 + 점 0.5 m + 멈춤 1 s. 시작 조건 HSF 5,359·HDL 5,266·HDU 4,494·OCR 2,273, **RS·HSL·RD 는 0** — RS 는 BEHAVIOR 다닐 곳(TRAV, 0.1 m 칸)에서 거실이 다른 방과 안 이어짐, HSL 은 욕실 바닥이 34 mm 높아 문턱 0.025 m 를 못 넘음, RD 는 주방–식당 사이 이중 벽 틈 0.36 m 에 0.13 m 여유 칸이 없음.
- **B2·B3(·B4·B5)**: 3.1절 규칙·거르개로 고르는 집기·놓기 표. B2 성공 = 보임 + 지도에 목표 확정(지도 되먹임) 0.3 s, B3 성공 = 물체 상자가 **잡는 점 작업 공간**(E0 −0.0119 m 로 다시 만든 표 — 상자 방 지도 토큰의 `T_REACH` 는 B0 비트 동일 때문에 예전 표 그대로, 바꿀지는 결정 필요)에 걸림 + 에임 10° + 멈춤 1 s. B4·B5 판정은 정의만(`bscene.h pred_grasped·pred_ontop·pred_inside`, 놓을 자리 `Entry::dlo/dhi`) — 잡기 물리는 E6.
- 커리큘럼 장치 값 `bsc::BCurr`(판 리셋 때 읽음, 다시 잡기 없음): B1/B2 비율(나머지 B3), 장면 비트, split(학습/공개 평가/둘 다), B1 시작 yaw 흔들기, 엄격, 지시문 heldout, 음성 대조용 거르개 끄기. 찾기의 지도 전체/부분/빈 지도는 지도 쪽 `MapCurr`(C0/C1/C2) 그대로.

집기·놓기 거르개 지남(3.1절 표, 느슨 / 엄격 — `bscene_check`):

| 장면 | 인스턴스 | 집을 물체(인스턴스 × 물체) | 놓을 곳(서로 다른 받침) | 짝 |
|---|---|---|---|---|
| HSF | 1,350 / 509 (10,880 중) | 3,122 / 583 | 1,787 / 505 | 16,291 / 1,672 |
| HDL | 382 / 0 (10,240) | 382 / 0 | 661 / 0 | 844 / 0 |
| HDU | 0 / 0 (3,200) | 0 / 0 | 0 / 0 | 0 / 0 |
| RD | 8 / 0 (2,560) | 14 / 0 | 8 / 0 | 14 / 0 |
| RS | 138 / 138 (1,920) | 155 / 155 | 2 / 2 | 310 / 310 |
| HSL | 93 / 0 (1,600) | 93 / 0 | 4 / 0 | 372 / 0 |
| OCR | 0 / 0 (1,600) | 0 / 0 | 0 / 0 | 0 / 0 |
| 합 | **1,971 / 647** | 3,766 / 738 | 2,462 / 507 | **17,831 / 1,982** |

- 뺀 까닭(HSF): 걸레받이 받침 6,717 짝, 잡는 자세 없음 6,576 물체, 창 3,006, 빈 넓이 1,840, 닫힌 곳 안 105. HDL: 관절체 320, 다른 성분 1,298. RD: 잡는 자세 없음 1,469(높은 식탁). 장면 파일 물체 중 집을 것 0(v3). 지시문 조합 59 × 문장 12 = 708 문장, 같은 조합 코사인 평균 0.722.

**다가가기 거리(목표 2) — 고른 것**: 지도 단계가 판마다 **nav_k(기본 10) 스텝에 한 번**(판 리셋 때 바로) 믿는 점유 비트(1 칸 부풀림, 안 본 칸은 지나감)에서 목표 둘레 씨앗(B1 0.15 m, 물체 0.45 m)으로 BFS(4·8 이웃 번갈아 = 팔각 거리, GPU 워프 하나 = 판 하나), 그때 로봇 칸 둘레 32 × 32 조각만 저장하고 로봇 칸에 닿은 단계 + 여유에서 멈춘다. 환경은 앞 스텝 지도의 거리장으로(판 번호 확인) **지난 자리·지금 자리를 같은 거리장으로** 재서 진행 보상 = 차이, 관측 74 = 거리. 지도가 안 붙으면 직선. 잰 대가(`nav_tradeoff` N 4,096 × 400, 같은 행동 → 궤적 같음, K = 1 기준): C2(빈 지도)에서 판 보상 합 차 평균 K 5 0.052 · K 10 0.124 · K 20 0.295(판 보상 합 크기 8.25), 거리 관측 차 0.006 / 0.013 / 0.028 m. C0(다 앎) 0.005 / 0.011 / 0.018. 지도 단계 비용(N 32,768, 자연): 거리장 끔 2.06 ms, K 1 3.74, **K 10 2.42**, K 20 2.30 ms. 거리장 대 참 장면 최단 경로(같은 씨앗, 로봇 중심 칸 0.13 m 여유): C0 평균 −0.26 m(팔각 거리·부풀림 차), 섞음(C0/C1/C2) −1.20 m(안 본 곳을 지나가는 낙관 — 의도대로).

검증(모두 잰 값):
- **GPU == CPU 비트 동일**: `env_verify 2048 600 --stage 3 --follow`(짝수 판 직진, 홀수 판 참 장면 대본 — B1 성공 407·B2 312·B3 241 판까지 지나감), `--strict`, `--split 1`. `map_verify 2048 600 --stage 3` C2·C0·C1·섞음+팔·매 keyframe·`--nav-k 1/20`·엄격(환경 ← 지도 거리장 되먹임 포함). `pnp_check`: 같은 씨앗으로 GPU 와 CPU 가 뽑은 10,000 판 상태 비트 동일.
- **음성 대조(모두 실패 = 정상)**: env `--negative`(회전 부호) 30,059,004, `--negative-scene`(B1 목표 방 지움) 11,539,830; map `--negative`(확정 규칙) 1,021,077, `--negative-room`(방 토큰 1.5 m 밀림) 591, `--negative-nav`(거리장 늘 4 이웃) 1,199, `--negative-way` 121, `--negative-live` 599.
- **B0 회귀(옛 A0–A2) 바이트 그대로**: 바꾸기 전 소스(f5b4ec7)로 지은 같은 도구와 견줘 env 스트림·상태 해시 6 설정(A0/A1/A2 × 팔 묶음/풂) 같음, `map_verify` 4 설정 출력 전체(통계·해시) 같음, `map_tokrec` 파일 2 개 md5 같음. `env_verify 2048 600` A1 8,907 / 3,420 / 1,543 그대로. 상자 방 지도 커널은 `BEH=false` 판이라 레지스터·스택이 예전과 같다(kf 91 레지스터·48 B), 지도 처리량 0.237 / 1.572 ms(전 0.234 / 1.564, N 4,096 / 32,768).
- **장면 바름**: 시작 조건 35,223 판 모두 시작 자세 몸통 안 닿음, B1 경로 = 다시 잰 다익스트라(차 0), B1 목표 점 방 = RASC 방(0 다름), 창 밖 0. 방 토큰 대 RASC 방 격자(독립 로더) 1,228,800 판·스텝 0 다름. `pnp_check` 10,000 판(느슨·엄격 각각): 집을 것(제외·관절체·폭·질량·높이·옆 잡기 가장자리·닫힌 곳)·놓을 곳(높이·크기·종류·출발과 다름·빈 넓이)·시작(창·몸통·거리·창 안 BFS 로 잡는 자세와 놓을 곳 가장자리에 닿음)·엄격 플래그·기록한 거르개 **위반 0**. 거르개 끄기 음성 대조: 빈 넓이 979, 닫힌 곳 17, 시작 닿음 4,696, 시작 몸통 744, 놓을 곳 닿음 762, 잡는 자세 1,981 위반(관절체 끄기는 0 — 관절체 집을 것은 어차피 잡는 자세에서 빠짐). 띠 점유 래스터 대 TRAV_OPEN_DOOR(방 칸) 같은 칸 비율 0.744–0.967, IoU 0.44(HSF)–0.95(RS) — BEHAVIOR TRAV 는 가구 바닥 자국을 칸 단위로 막아 다름. B1 짝의 독립 이어짐(RASC TRAV_OPEN_DOOR 4 이웃, 0.1 m 칸) HSF 4,035/5,359·HDU 0/4,494 — 0.1 m 칸 축소가 문간을 막아서(우리 다닐 칸은 문간 0.2 m 를 살림).

처리량(RTX 5070 Ti, 다른 GPU 일 없음, ms/스텝):

| | N 4,096 | N 32,768 |
|---|---|---|
| 환경만, 상자 방 A2 | 0.176 | 0.235 |
| 환경만, BEHAVIOR B1–B3 섞음 | 0.336 | 0.444 |
| 지도 단계, 상자 방 A1(자연) | 0.237 | 1.572 |
| 지도 단계, BEHAVIOR(자연, 거리장 K 10) | 0.446 | 2.420 |
| 환경 + 대본 + 지도, 상자 방 | 1.53e7 판·스텝/s | 2.00e7 |
| 환경 + 대본 + 지도, BEHAVIOR K 10 | 5.04e6 | 1.13e7 |

- 기준(A2 0.172 ms/스텝의 5 배 안): 환경만 0.336 ms = 1.9 배. 몸통 충돌은 스텝마다 0.27 m 안 후보 상자를 모아 서브스텝에서 그것만 봄(결과 같음, N 32,768 0.864 → 0.480 ms). N 4,096 은 판마다 스레드 하나의 광선 16(묶음 걷기)·보임 지연에 묶임.

막힌 것·남은 것:
- **③ OmniGibson 대조**(IoU·경로 길이·깊이 50 곳)는 안 함 — E7 묶음으로. 
- 창 12.8 m 고정(지도 창 이동은 판마다 옮김만, 판 안 따라 옮기기 없음) → B1 은 창에 들어가는 방 짝만. RS·HSL·RD B1 0, HDU·OCR 집기·놓기 0.
- 정적 상자는 AABB 가 아닌 회전 상자지만 속이 찬 상자(탁자 밑으로 못 감), 문은 늘 열림, 판의 과제 물체는 지도 물체 9 칸 안 것만 부딪힘, 동역학은 바닥 높이 차를 무시(다닐 곳·시작 고르기만 문턱을 씀).
- ~~PPO 학습기가 stage 3 를 못 띄움~~ → **됨**(2026-10-04): 학습기·실행기가 env 3 을 띄우고 장치 커리큘럼이 `BCurr`(B1/B2/B3 비율·장면·split·엄격·지시문)·`MapCurr`(C0/C1/C2)를 함께 바꾼다, 설정 `training/RL/config/ppo_b.json`(B0 → B1 C0/C1/C2 → B2 → B3 → 섞음). 잰 값은 [ppo README](../../training/RL/ppo/README.md) "E2" 절. 관측 쪽(`obs.h`)에 지시문 행(`I_B_INSTR` → `pnp_v1/instr128.f16`)을 넣는 일은 남음.
- 이름 표에 없는 가구 종류(bench·ottoman)는 "furniture" 상위어, 생김새 행은 상자 하나.
- 상자 방(A0–A2) 지도 토큰 `T_REACH` 를 잡는 점 표로 바꿀지(B0 비트 동일이 깨짐) — 결정 필요.

## 6. 열린 질문 (사용자)

1. **과제 수**: 지시에는 "50 과제" 로 적혀 있지만 2026 대회는 100 과제다(0–49 = 2025 과제). 커리큘럼 기준을 100 개 전부로 할까, 0–49 만으로 할까? (이 문서는 100 개로 썼다. 0–49 만 보면 점수가 0 이 아닌 과제는 0, 1, 9, 18, 26, 27, 30 — 느슨 기준.)
2. **목표**: 이 로봇의 q 상한은 0.03–0.09 (추정, 1.5절)이고 마감(10-17)까지 13 일이다. ⓐ BEHAVIOR 를 **학습 무대**로만 쓰고(장면·물체·술어·지표), 대회 점수는 노리지 않는다 ⓑ 대회 제출은 팀 R1 Pro 경로로 하고 LIMO 커리큘럼은 따로 ⓒ 로봇을 바꾼다(예: 팔을 올리는 기둥·리프트 — 실제 로봇도 바뀜). 어느 쪽인가?
3. **되는 과제 밖 과제**: 높이·무게 때문에 못 하는 과제를 "낮은 곳 변형"(같은 장면·물체·술어, 물체 자리만 낮은 면으로 다시 뽑은 인스턴스)으로 학습에 넣을까? 넣으면 데이터는 늘지만 공식 인스턴스와 분포가 다르다.
4. **잡기 물리**: 집기·놓기(B4–B5)를 우리 키네마틱 잡기(빠름, PhysX 와 다름)로 먼저 하고 OmniGibson 대조가 기준 미달일 때만 팀 엔진으로 옮기는 순서에 동의하나?
5. **물체 메시**: 렌더·충돌을 상자로 시작한다. 물체 메시가 필요해지면 암호화 USD 를 OmniGibson(Python, 오프라인 한 번)으로 풀어 내보내야 한다. 그때 Python 일회성 추출을 허용하나?

## 출처

- 대회 규칙·과제·장면: `B26/docs/대회개요.md`, `과제목록.md`, `평가규칙_원문.md`(원문 보관본 `B26/docs/raw/`), https://behavior.stanford.edu/challenge/
- BDDL: `B26/BEHAVIOR-1K/bddl3/bddl/activity_definitions/*/problem0.bddl`
- 인스턴스·메타데이터: `B26/BEHAVIOR-1K/datasets/2026-challenge-task-instances/`(`metadata/task.jsonl`, `B100_task_misc.csv`, `scenes/`, `scene_test/public/`)
- 장면·물체: `B26/BEHAVIOR-1K/datasets/behavior-1k-assets/`(VERSION 3.9.0: `scenes/*/json`, `scenes/*/layout`, `objects/*/*/misc/metadata.json`, `metadata/avg_category_specs.json`)
- 점수 코드: `B26/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py`, `eval/utils/eval_utils.py`, `eval/evaluator.py`
- 팀 엔진: `B26/docs/엔진_자체구현.md`(0·2·15.2·15.3절), `B26/docs/미해결과제.md`(B16·B19), `B26/src/sim/engine/core/omni/{bddl.h,states.h,assisted_grasp.h}`
- 로봇: `src/robot/og/limo_omx_eval.yaml`, `src/robot/real_limits.json`, `training/RL/env/include/limo_omx_model.h`, ROBOTIS OMX-F 사양(위 링크), E0 측정 `src/robot/og/e0/`(5.3절, 사양 링크도 거기)
- 우리 환경: `training/RL/env/README.md`, `training/RL/map/README.md`, `training/RL/ppo/README.md`, `training/BC/README.md`
