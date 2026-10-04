# fastsam — 물체 분할 다시 학습(벽·천장·바닥 없이, 통째로)

작성 2026-10-05. **이 폴더의 코드는 AGPL-3.0 이다**([LICENSE](LICENSE), 아래 "라이선스"). 저장소의 나머지는 Apache-2.0.

## 목표

FastSAM-s 416(TensorRT, behavior-2026 `src/scene_graph/ovdet`)을 다시 학습한다.
- 벽·천장·바닥 마스크를 내지 않는다.
- 문·창·계단은 물체로 남긴다. 지도의 방·문 토큰과 계단 위험 판단에 쓴다.
- 조각 대신 물체 통째 마스크를 낸다.
- **진짜 물체를 찾는 수는 줄지 않는다.** 작은 것, 처음 보는 범주, 문·창·계단까지 버킷마다 옛 엔진 이상이어야 한다(아래 "게이트").
- 그대로 갈아 끼울 수 있어야 한다.
  - 입력 416, 클래스 하나 'object'.
  - 출력 모양은 옛 엔진과 같다: `output0` 1×37×3549, `output1` 1×32×104×104.
  - 같은 ovdet 경로, 같은 설정(conf 0.25)으로 돈다.
  - 지연 시간·GPU 메모리는 같거나 낮다.
- 실행 쪽에는 후처리 커널을 더하지 않는다.

같은 라벨로 Jetson Nano(4 GB)용 작은 학생 YOLO26n-seg(클래스 하나)도 학습한다. 아래 "Nano 안"을 본다.

## 결과 요약

**설치한 엔진**
- `~/ovdet_models/x86_sm120/FastSAM-s-416-obj.plan`: v2, 문턱 0.05 보정, 실행 conf 0.25 그대로. `.pt`·`.onnx` 는 `~/ovdet_models/pt/`.
- `~/ovdet_models/x86_sm120/yolo26n-seg-obj-416.plan`: Nano 학생, 문턱 0.04 보정.
- 옛 엔진 `FastSAM-s-416.plan` 은 그대로 있다. 기본값은 바꾸지 않았다.

**게이트(검출 단계, 버킷마다 옛 엔진보다 뚜렷이 나쁘지 않음)**
- FastSAM-s-416-obj(v2, t 0.05)는 시뮬·COCO·ADE 세 곳 모든 버킷에서 95 % 구간이 0 을 포함하거나 더 좋다.
- 다만 시뮬 문·창·계단은 평균 −2.1 %p(구간 [−5.2, +1.2]), 시뮬 처음 보는 범주는 −1.5 %p 다.
- "평균도 −1 %p 이상"까지 요구하는 더 엄격한 기준으로는 v2 가 시뮬 문·창·계단 하나로 떨어진다.

**끝에서 끝(radio r3, slam)**
- 옛 규칙: 찾음 27 → 31/34, 노드 290 → 250, 중복 140 → 112, 벽·천장·바닥 위 헛노드 25 → 11, 정답 이름 정확도 0.56 → 0.76.
- scenemap 확률 모드(objprob): 찾음 28 → 27/34(≥ 27 조건은 지킴), 노드 126 → 121, 중복 62 → 54, 벽·천장·바닥 헛노드 6 → 3, 문·창·계단 구조 물체 5/9 → 7/9.

**값(대가)**
- 덜 나눔이 늘었다. 프레임당 시뮬 1.7 → 2.2, COCO 3.3 → 5.3. 대부분 의자 + 식탁, 사람 + 의자다.
- 실제 사무실 영상(OpenLORIS)에서 노드가 늘었다(293 → 368, 274 → 409, 같은 이름 쌍 89 → 147).
- 시뮬에서 재현율을 맞추려면 문턱을 낮춰야 하는데, 실제 사진에서는 그 문턱이 과하다.
- 같은 v2 를 t 0.11 로 쓰면 OpenLORIS 는 깨끗하다(230·298 노드, 천장 띠 5·9). 하지만 재현율 게이트에서 떨어진다(시뮬 문·창·계단 −6 %p, COCO 문·창·계단 −7 %p).

**권고**
- 시뮬(BEHAVIOR) 실험과 옛 규칙 지도에는 FastSAM-s-416-obj 가 확실히 낫다.
- 실제 로봇 기본값으로 바꾸기 전에 두 가지가 필요하다.
  - ① objprob 매개변수를 새 엔진에 다시 맞춘다(label_prior·κ·합치기 로지스틱이 옛 조각 통계로 맞춰져 있음).
  - ② 실제 집 영상에서 노드·중복을 확인한다.
- Nano 학생은 게이트는 지나지만(t 0.04) 실제 사진에서 헛것·덜 나눔이 옛 엔진보다 많다. 끝에서 끝 objprob 찾음도 26/34 라 아직 권하지 않는다.

## 라벨: 자기 증류 + 정답 통째

이미지 한 장의 학습 라벨은 다음 둘을 합친 것이다(`build_data.py`).

1. **정답 통째 마스크 전부.** 출처는 BEHAVIOR 시뮬 instance, COCO things, LVIS, ADE20K instance다. 문·창·계단도 넣는다.
   - 정답끼리는 절대 합치지 않는다. 인스턴스마다 한 줄씩 쓴다.
   - 실행 면적 문턱(ovdet `area_min` 24 칸 ≈ 640×480 의 909 px)의 절반쯤보다 작은 정답(넓이 비율 < 0.0013)은 뺀다. 실행 때 결코 나올 수 없는 크기라 배울 것이 없고 점수만 낮춘다.
2. **원래 FastSAM-s 416 의 마스크**(옛 엔진, conf 0.25). 다음 경우에는 버린다.
   - 픽셀 50 % 이상이 벽·천장·바닥이면 버린다(구조물 헛것).
   - 픽셀 50 % 이상이 정답 하나 위이거나, 70 % 이상이 정답들의 합집합 위면 버린다. 그 정답의 조각이고, 정답 통째가 대신한다.
   - 정답 하나를 그 넓이의 20 % 이상 덮으면 버린다. 정답에 벽이나 이웃 물체를 붙인 덩어리라서 덜 나눔을 배우게 된다(액자 + 아래 벽, 식탁 + 의자).
   - 나머지는 그대로 남긴다. 정답이 없는 곳, 즉 라벨 안 된 물체나 처음 보는 범주다. 그래서 옛 엔진이 찾던 것은 계속 찾는다.

평균 라벨 수(장당):

| 출처 | 장 | 정답 | 남긴 옛 마스크 | 버림: 구조물 | 버림: 정답 조각 | 버림: 합친 덩어리 |
|---|---|---|---|---|---|---|
| COCO train | 12,000 | 11.2 | 5.9 | 6.7 | 17.7 | 0.7 |
| ADE20K train(실내) | 6,000 | 11.4 | 1.3 | 9.9 | 19.7 | 0.3 |
| BEHAVIOR 시뮬 학습 장면 | 6,424 | 4.4 | 0.1 | 6.5 | 8.1 | 0.2 |

**구조물(배경)**
- 시뮬: `walls`·`floors`·`ceilings`·`roof`·`baseboard`.
- COCO-panoptic: wall-brick·wall-stone·wall-tile·wall-wood·wall-other-merged·floor-wood·floor-other-merged·ceiling-merged.
- ADE20K: wall·floor·ceiling.
- 그 밖의 stuff 는 배경으로 만들지 않는다. cabinet·shelf·counter·curtain·rug·table-merged·light 등은 정답 인스턴스가 아니어서 옛 엔진 마스크가 남는다. 하늘·길 같은 바깥 stuff 도 마찬가지다.
- 문·창·계단:
  - COCO 의 door-stuff·window-other·stairs 는 정답('dws')으로 넣는다.
  - ADE 의 door·windowpane·stairs·stairway·screen door 인스턴스도 정답이다.
  - 시뮬의 door·sliding_door·garage_door·elevator_door·fixed_window·openable_window·stairs·railing 도 정답이다.
- 시뮬 `pillar` 는 물체로 둔다. 사용자 정의상 배경은 벽·천장·바닥뿐이다.

## 데이터

| 출처 | 쓴 것 | 고른 방법 | 위치(git 밖) |
|---|---|---|---|
| BEHAVIOR 2026 시뮬 | 51 장면 중 `*_garden` 을 뺀 46. 학습 36 장면 + 과제 템플릿 판, 평가 10 장면 | 아래 "시뮬 렌더" | `~/datasets/fastsam_obj/sim/` |
| COCO 2017 + COCO-panoptic | train 12,000 장(실내 9,000 + 그 밖 3,000), 평가 val 1,500 장(실내 1,000 + 500) | 실내 = 벽·바닥·천장 stuff 가 이미지의 10 % 이상. 평가는 LVIS v1 val 에도 있는 val2017 이미지 | `~/datasets/fastsam_obj/real/` |
| LVIS v1 | 위 COCO 이미지의 인스턴스(1,203 범주). panoptic thing 과 IoU > 0.7 이면 LVIS 쪽을 버림 | 학습에서 120 범주를 뺌(`heldout_lvis.txt`, COCO 80 과 안 겹치는 c·f 빈도에서 시드 0) | 같은 곳 |
| ADE20K SceneParsing 2016 + 2017 instance | train 실내 6,000 장, val 실내 1,021 장 | 실내 = 천장 ≥ 2 % 또는 벽 + 바닥 ≥ 15 % | `~/datasets/fastsam_obj/raw/` |

**'처음 보는 범주' 시험**
- LVIS 120 범주는 학습 라벨에서 지운다. 옛 엔진 마스크가 있으면 그것만 남는다. 실제로 배포한 뒤 처음 보는 물체가 놓이는 처지와 같다.
- 평가에서는 이 범주를 'unseen' 으로 따로 센다.
- 시뮬 평가 장면의 범주 중 학습 장면 json 에 없는 것도 'unseen' 이다.

### 시뮬 렌더(`sim_render.py`, `label_sim.py`, `render_all.sh`, `render_tasks.sh`)

**카메라**
- LIMO 몸통 카메라와 같다(robot-agent `src/robot/og/limo_omx_eval.yaml` + `eval_with_limo.py`, sim_detcmp `meta.json` 의 `T_bc`).
- 높이 0.18 m, 앞을 수평으로 본다. H-FOV 67.9°(Orbbec Dabai), 640×480.

**자세**
- 장면 traversability 지도(물체 포함)에서 LIMO 반폭 0.2 m 만큼 깎은 빈 곳을 고른다. 로봇 없이 카메라만 둔다.
- 비율:
  - 60 %: 그냥.
  - 20 %: 벽·가구 0.2–0.6 m 앞에서 가장 가까운 장애물 쪽 ±70°. 비스듬히·가까이 보는 판.
  - 10 %: 위로 12–35°. 천장·벽 윗부분.
  - 10 %: 높이 0.3–1.2 m(팔 카메라·다른 로봇), pitch −25…+10°.

**정답 instance**
- OG 의 `seg_instance`·`seg_instance_id`·`seg_semantic` 은 이 PC 에서 SyntheticData 후처리 segfault 가 난다(3 번째 캡처, training/demos 와 같은 증상).
- 그래서 RGB·깊이만 OG 로 렌더하고, 장면 visual mesh(월드 좌표, `mesh_prim_to_trimesh_mesh`)를 장면마다 한 번 꺼내 embree 광선으로 픽셀마다 물체 번호를 만든다.
- OG 깊이와 광선 깊이가 2 cm + 2 % 넘게 다른 픽셀은 '모름'(학습·평가에서 뺌)이다.
- 장면 전체 일치율은 99.3–99.6 % 다.
- 렌더가 가끔 앞 프레임 그림을 그대로 낸다. 그런 프레임(모름 > 15 %)과 바깥을 보는 프레임(광선이 아무것도 안 맞은 픽셀 > 15 %)은 버린다.

**속도와 장면 수**
- 렌더는 장면 로드에 30 s–4 min, 프레임당 0.03–0.2 s 다. 광선 라벨은 0.1 s/장이다.
- 장면당 장 수: 집·`*_int` 400, 그 밖 200, 평가 150, house_double_floor_lower 300.
- 과제 템플릿 판(작은 과제 물체가 놓인 `2026-challenge-task-instances` 의 `*_template.json`, 로봇 항목은 뺌):
  - 평가: house_double_floor_lower turning_on_radio(라디오, sim_detcmp 와 같은 판), house_double_floor_upper·office_cubicles_right 템플릿 몇 개.
  - 학습: Rs_int·hotel_suite_large·house_single_floor·restaurant_diner.

**자원**
- OG 는 `flock /tmp/claude-1000/og.lock` 을 장면마다 따로 잡는다.
- Kit 이 장면마다 0.6–1 GB 남기는 임시 폴더는 프로세스 전용 TMPDIR(`~/datasets/fastsam_obj/tmp/<scene>_<pid>`)로 모으고 끝에 지운다.

**장면 나누기(`scenes.sh`)**
- 평가 장면은 집 단위로 뺀다. 같은 건물의 다른 층·좌우 짝도 같이 뺀다.
- 평가: house_double_floor_lower·upper, Wainscott_0·1_int, office_cubicles_left·right, restaurant_brunch, grocery_store_cafe, hotel_suite_small, school_geography.

## 평가

### 검출 단계(`eval_det.py`, `sweep_t.py`)

**공통 조건**
- 같은 프레임에 엔진 여럿을 ovdet C API(ctypes `libovdet.so`)로 돌린다. 실행 때와 똑같은 후처리다: letterbox, conf, NMS 0.7, 면적 문턱, 마스크 중복 0.7.
- 비교는 언제나 옛 엔진 `FastSAM-s-416.plan` 과 짝을 지어 한다.

**지표(프레임마다, 정답 하나씩)**
- found: 검출 마스크 중 그 픽셀의 50 % 이상이 이 정답 위인 것이 하나라도 있음. 지도 파이프라인이 노드를 만드는 조건과 같은 뜻이라 조각이어도 찾은 것이다.
- iou50: 가장 잘 맞는 검출의 마스크 IoU ≥ 0.5, 즉 통째로 찾음.
- best IoU: found 인 정답의 가장 좋은 IoU 평균(마스크 질).
- 덜 나눔과 더 나눔: FastSAM 은 두 방향 모두 틀린다.
  - 더 나눔(over-segmentation) `frag`: found 인 정답 하나 위의 검출 수.
  - 덜 나눔(under-segmentation) `under`: 검출 하나가 정답 둘 이상을 각 넓이의 20 % 이상 덮은 경우. 다른 정답 안에 든 것은 빼고 센다. 또는 정답 하나를 20 % 이상 덮으면서 검출 넓이의 30 % 이상이 벽·천장·바닥인 경우(액자 + 아래 벽, 식탁 + 의자). 프레임당 수와 검출 중 비율로 본다.
- struct FP: 픽셀 50 % 이상이 벽·천장·바닥인 검출(프레임당).

**버킷**
- 크기:
  - 시뮬은 물체 상자 가장 긴 변 기준이다. small ≤ 0.35 m(집을 수 있는 크기), medium ≤ 1 m, furniture.
  - 실제 사진은 COCO 넓이 기준이다. small < 32², medium < 96², large.
- 그 밖의 버킷:
  - `dws`: 문·창·계단.
  - `unseen`: 학습에 없던 범주.
  - `nested`: 더 큰 다른 정답 안에 50 % 이상 든 정답(입은 옷 ⊂ 사람, 서랍 ⊂ 장). 통째 정책상 일부러 줄이는 것이라 따로 보고한다.
  - `whole` = all − nested, `unseen_whole` = unseen − nested.
  - `tiny`(시뮬 300–1,000 px)는 보고만 한다.

**게이트** — 버킷 `whole·small·medium·furniture·large·dws·unseen_whole` 마다:
- 프레임 짝 부트스트랩(2,000 번)으로 Δfound 의 95 % 구간을 낸다.
- 평균 Δfound ≥ −0.01 이고, 위 끝 ≥ 0 이어야 한다(뚜렷이 나빠지지 않음).
- 시뮬 평가·COCO val·ADE val 세 곳 모두에서 통과해야 한다.

**점수 문턱 보정(`calibrate.py`)**
- 다시 학습한 모델은 같은 물체를 조각 여럿이 아니라 마스크 하나로 내므로 점수가 낮게 나온다.
- 게이트를 지나는 가장 높은 문턱 t 를 `sweep_t.py` 로 고른다.
- 분류 로짓 bias 에 logit(0.25) − logit(t) 를 더해 가중치 안에 넣는다.
  - 시그모이드 앞 상수 이동이라 점수 순서가 그대로다. NMS·중복 제거 결과는 문턱만 바꾼 것과 같다(300 장에서 확인).
  - 실행 설정 conf 0.25 를 그대로 두고 갈아 끼울 수 있다.

### 끝에서 끝(`e2e.sh`)

**기록**
- sim_detcmp radio r3(LIMO + OMX frontier 탐사, house_double_floor_lower, 146 s, 정답 34).
- realbag_run 은 slam·gt 자세 두 판으로 돌린다.
- 옛 규칙(plain)과 scenemap 확률 모드(objprob, `--objprob --label-prior ~/datasets/objprob/fit1/label_prior.json`) 둘 다 본다.
- 채점은 behavior-2026 `tools/realbag/objprob_eval.py` 로 한다(detcmp_eval 표 + 잘못 합침·문창계단).

**빌드**
- behavior-2026 main(59ec923)의 realbag → `~/realbag_build_fsobj`.
- `realbag_run --engine` 은 원래 있는 옵션이라 behavior-2026 은 고치지 않았다.

### 실제 bag(`realcheck.sh`)

- OpenLORIS office1-1·1-5, slam, det-every 3, max-depth 4(sim_detcmp `openloris_fullcount` 와 같음).
- 정답이 없으므로 노드 수, 같은 이름 0.5 m 쌍, 구조물처럼 보이는 노드(천장 띠 z 위끝 > 2.3 m 등)를 기하 대용으로 센다.

## 결과

**판들**

| 판 | 시작 | 데이터(학습 목록) | epoch | 비고 |
|---|---|---|---|---|
| pilot | FastSAM-s.pt | COCO 12k + ADE 6k | 8 | 실제 사진만 |
| real30 | FastSAM-s.pt | 같음(작은 정답 뺀 라벨) | 30 | 실제만 길게 → 시뮬이 나빠짐 |
| mix_partial | FastSAM-s.pt | 시뮬 2.65k ×3 + 실제 18k | 15 | 시뮬을 넣자 크게 나아짐 |
| fs_v1 | FastSAM-s.pt | 시뮬 5.8k ×3 + 실제 18k(합친 덩어리 버림 규칙) | 30 | |
| **fs_v2** | fs_v1 | 시뮬 6.4k ×2 + 문·창·계단 보이는 2.85k ×2 + 실제 18k | 25 | **설치(t 0.05)** |
| fs_v3 | fs_v2 | 시뮬 ×4 + 문·창·계단 ×3 + 실제 | 10/25 에서 멈춤 | 문·창·계단이 v2 보다 나빠서 |
| n26_v0 | yolo26n-seg.pt(머리 1 클래스로 새로) | 시뮬 5.2k ×3 + 실제 18k | 60 | Nano 학생 |

**한 표 비교**
- 검출 단계는 2 장마다 한 장으로 쟀다. 칸은 옛 엔진 → 새 것 값이다.
- 버킷별 전체 값과 95 % 구간은 [results.md](results.md) 에 있다.

| | 원래 FastSAM-s-416 | fs_v1 (t 0.07) | **fs_v2 (t 0.05, 설치)** | YOLO26n (t 0.04) |
|---|---|---|---|---|
| 게이트(뚜렷이 나빠진 버킷 없음) | — | ✗ 시뮬 문·창·계단 | ✓(평균 −2.1 %p 하나) | ✓ |
| 시뮬 재현율 whole / 문·창·계단 / 처음 보는 범주 | 0.793 / 0.866 / 0.668 | 0.802 / 0.827 / 0.649 | 0.815 / 0.845 / 0.653 | 0.820 / 0.872 / 0.664 |
| COCO 재현율 whole / small / large / 문·창·계단 / 처음 보는 범주 | 0.499 / 0.055 / 0.962 / 0.863 / 0.346 | 0.532 / 0.071 / 0.963 / 0.865 / 0.353 | 0.540 / 0.078 / 0.969 / 0.876 / 0.365 | 0.534 / 0.073 / 0.969 / 0.876 / 0.348 |
| ADE 재현율 whole / 문·창·계단 | 0.581 / 0.785 | 0.612 / 0.795 | 0.625 / 0.808 | 0.626 / 0.806 |
| 벽·천장·바닥 헛검출/프레임: 시뮬 · COCO · ADE | 4.44 · 6.06 · 10.05 | 0.98 · 4.86 · 6.35 | 1.05 · 5.46 · 7.56 | 1.71 · 6.31 · 9.14 |
| 더 나눔: 찾은 물체당 검출, 시뮬 · COCO · ADE | 2.77 · 3.14 · 2.29 | 1.81 · 2.96 · 2.07 | 1.94 · 3.23 · 2.21 | 2.75 · 3.90 · 2.82 |
| 덜 나눔/프레임: 시뮬 · COCO · ADE | 1.66 · 3.33 · 3.75 | 1.98 · 4.84 · 5.02 | 2.20 · 5.34 · 5.74 | 3.42 · 7.33 · 8.22 |
| 통째 IoU ≥ 0.5: 시뮬 · COCO · ADE | 0.676 · 0.422 · 0.501 | 0.696 · 0.470 · 0.559 | 0.699 · 0.475 · 0.573 | 0.698 · 0.464 · 0.562 |
| 끝에서 끝 옛 규칙(slam): 찾음 / 노드 / 중복 / 벽·천장·바닥 헛노드 / 정답 이름 | 27 / 290 / 140 / 25 / 0.56 | 30 / 234 / 106 / 13 / 0.82 | 31 / 250 / 112 / 11 / 0.76 | 29 / 295 / 132 / 19 / 0.59 |
| 끝에서 끝 objprob(slam): 찾음 / 노드 / 중복 / 벽·천장·바닥 헛노드 / 문·창·계단 구조 물체 | 28 / 126 / 62 / 6 / 5/9 | 24 / 115 / 46 / 4 / 7/9 | 27 / 121 / 54 / 3 / 7/9 | 26 / 100 / 32 / 1 / 6/9 |
| 끝에서 끝 objprob(gt 자세): 찾음 | 27 | 27 | 29 | 26 |
| OpenLORIS 1-1 / 1-5: 노드 · 같은 이름 쌍 · 천장 띠 노드 | 293·89·11 / 274·29·16 | 320·131·9 / 349·49·14 | 368·147·6 / 409·66·12 | 339·101·8 / 364·38·15 |
| TensorRT 416(RTX 5070 Ti, ovdet 전체): ms / MiB | 0.871 / 58 | 0.844 / 54 | 0.855 / 54 | 0.911 / 48 |
| TensorRT 320 | — | — | 0.798 / 46 | 0.866 / 42 |

**참고**
- 같은 v2 를 t 0.11 로 보정하면 다음과 같다.
  - 실제 사진: 벽·천장·바닥 헛것 COCO 3.39·ADE 4.41, 덜 나눔 3.75·3.74(옛 엔진과 같음), OpenLORIS 230·298 노드.
  - 대신 재현율 게이트에서 떨어진다: 시뮬 whole −3 %p, 문·창·계단 −6 %p, COCO 문·창·계단 −7 %p, large −1.8 %p.
- 지연 시간: 엔진 넷 모두 1 ms 아래로 같은 수준이다. 큰 GPU 에서는 고정 비용이 대부분이라 YOLO26n(3.8 GFLOPs)이 FastSAM-s(약 40 GFLOPs)보다 빠르지 않다. Jetson Nano 에서는 다를 것이다(재지 않음).
- GPU 메모리(엔진 + 버퍼)는 새 엔진이 같거나 적다. realbag 프로세스 최대는 662 MiB 로 같다.

**어디서 졌나(시뮬 문)**
- 놓친 문의 대부분은 렌즈 바로 앞을 가득 채운 민무늬·어두운 문면이다. 옛 엔진은 화면 전체를 덮는 큰 마스크로 '찾음'에 들어갔다.
- 일부는 실제로 놓친 것이다. 화분에 반쯤 가린 나무 문이 그 예다.
- 문·창·계단 이미지를 더 넣은 v2·v3 로도 크게 나아지지 않았다.

## Nano 안(YOLO26n-seg, Jetson Nano 4 GB)

**학습**
- 시작점: YOLO26n-seg(`~/ovdet_models/archive/pt/yolo26n-seg.pt`, 2.7 M 변수).
- COCO 80 클래스 머리를 'object' 1 클래스로 새로 만든다(`single_cls`).
- 라벨은 FastSAM-s 와 같다. 합친 덩어리 버림과 작은 정답 뺌이 들어간 판이다. 시뮬은 그때까지 렌더된 5.2 천 장이다.
- 60 epoch, lr 0.01.

**출력**
- `end2end=False` 로 내보내면 옛 FastSAM 엔진과 같은 모양이다(1×37×3549 + 1×32×104×104).
- 그러면 ovdet 이 NMS 를 그대로 한다. YOLO26 의 NMS 없는 머리는 쓰지 않는다.
- 320 판: 1×37×2100 + 1×32×80×80.

**결과(위 표)**
- 문턱 0.04 에서 게이트는 지난다. 처음 보는 범주도 옛 엔진과 같다(COCO 0.348 대 0.346, 시뮬 0.664 대 0.668). 작은 학생도 라벨 밖 범주를 옛 엔진만큼은 찾는다.
- 하지만 실제 사진에서 벽·천장·바닥 헛것이 거의 줄지 않는다(COCO 6.31 대 6.06).
- 덜 나눔은 두 배다.
- 끝에서 끝 objprob 찾음은 26/34 다.
- FastSAM-s v2 보다 분명히 못하므로, Nano 용으로는 아직 권하지 않는다.
- 다음에 해 볼 것: v2 를 선생으로 한 증류(v2 마스크를 라벨로), 더 긴 학습, 320 입력 학습.

## 다시 만들기

```
# 0) venv: ~/fastsam_venv (uv, py3.11, torch 2.11.0+cu128 — ~/embed_venv 와 하드링크 공유, ultralytics 8.4.171,
#    pycocotools onnx onnxslim trimesh embreex)
cd ~/robot-agent/training/fastsam
# 1) 시뮬 렌더 + 광선 라벨(OG conda env behavior, og.lock)
bash render_all.sh eval; bash render_all.sh train; bash render_tasks.sh
# 2) 실제 사진 고르기·받기(raw: panoptic_annotations_trainval2017.zip, lvis_v1_{train,val}.json.zip,
#    ADEChallengeData2016.zip, ChallengeData2017/annotations_instance.tar → ~/datasets/fastsam_obj/raw)
~/fastsam_venv/bin/python select_real.py
# 3) 라벨(옛 엔진 자기 증류 + 정답), 조각으로 나눠 동시에
for k in 0 1 2 3; do ~/fastsam_venv/bin/python build_data.py --source coco_train --shard $k/4 & done
~/fastsam_venv/bin/python build_data.py --source ade_train; ~/fastsam_venv/bin/python build_data.py --source sim_train
~/fastsam_venv/bin/python build_data.py --source sim_val; ~/fastsam_venv/bin/python build_data.py --source coco_val
# 이 PC 경로·og.lock 은 env_local.sh(스크립트가 읽음). 파이썬을 직접 부를 때는 먼저 source env_local.sh
# 4) 학습 → 문턱 고르기 → 보정 → 엔진
~/fastsam_venv/bin/python train.py --name <run> --epochs 30 --sim-rep 3
bash cand_eval.sh <run> last        # 임시 엔진 + 옛 엔진과 짝 평가
~/fastsam_venv/bin/python sweep_t.py ~/datasets/fastsam_obj/cand/cand_<run>_last.plan --ts 0.25 0.18 0.14 0.11 0.09 0.07 0.05 \
    --out ~/datasets/fastsam_obj/eval/sweep_<run>.json
~/fastsam_venv/bin/python calibrate.py ~/datasets/fastsam_obj/runs/<run>/weights/last.pt <out.pt> --t <best_t>
bash export.sh <out.pt> FastSAM-s-416-obj      # → ~/ovdet_models/{pt,x86_sm120}/FastSAM-s-416-obj.*
# 5) 지연 시간(빈 GPU 에서)
~/fastsam_venv/bin/python bench_latency.py old=~/ovdet_models/x86_sm120/FastSAM-s-416.plan new=~/ovdet_models/x86_sm120/FastSAM-s-416-obj.plan \
    --images ~/datasets/sim_detcmp/streams/radio_limo_r3/rgb --n 300
# 6) 끝에서 끝·실제 bag
bash e2e.sh <이름> ~/ovdet_models/x86_sm120/FastSAM-s-416-obj.plan plain   # 그리고 objprob
bash realcheck.sh <이름> ~/ovdet_models/x86_sm120/FastSAM-s-416-obj.plan
```

## 라이선스

| 것 | 라이선스 | 우리에게 뜻하는 것 |
|---|---|---|
| FastSAM-s.pt(Ultralytics 배포, 체크포인트 메타 `license: AGPL-3.0`) | AGPL-3.0. 원 FastSAM 코드(CASIA-IVA-Lab)는 Apache-2.0 이지만, 쓰는 가중치는 Ultralytics YOLOv8-seg 로 학습·배포된 것 | 미세조정한 가중치·ONNX·TensorRT 엔진은 파생물로 보고 AGPL-3.0 으로 다룬다 |
| Ultralytics 8.4(학습·내보내기 코드) | AGPL-3.0 | 이 폴더의 학습 스크립트는 ultralytics 를 import 한다 → 폴더를 AGPL-3.0 으로 표시(`LICENSE`). 실행 쪽 ovdet(C++/CUDA)에는 Ultralytics 코드가 없다 |
| YOLO26n-seg.pt(Nano 학생 시작점) | AGPL-3.0 | 위와 같다 |
| SA-1B(원래 FastSAM 학습 데이터) | SA-1B 연구 라이선스 | 우리는 SA-1B 를 직접 쓰지 않는다. 원래 FastSAM 마스크를 라벨로 쓰므로 간접적으로 이어진다 |
| COCO 2017 주석·COCO-panoptic | CC BY 4.0 | 출처 표시 |
| COCO 이미지 | Flickr 이용 약관. 이미지마다 CC 라이선스(BY·BY-NC·BY-NC-SA·BY-ND 등, 주석 json `licenses`) | 상업 사용 제한이 있다. 비상업 연구는 괜찮다. 이미지는 저장소에 넣지 않는다 |
| LVIS v1 주석 | CC BY 4.0(이미지는 COCO) | 출처 표시. 이미지는 위와 같다 |
| ADE20K 이미지(SceneParsing 2016) | MIT CSAIL 이용 약관: 비상업 연구·교육만 | 비상업 연구로만 쓴다. 저장소에 넣지 않는다 |
| ADE20K 주석·도구 | BSD-3(CSAIL) | 출처 표시 |
| BEHAVIOR-1K 자산·장면(시뮬 렌더 원천) | BEHAVIOR 데이터셋 약관(비상업 연구, OmniGibson 키 동의) | 렌더 이미지·라벨은 저장소에 넣지 않는다 |
| OpenLORIS-Scene(실제 bag 확인) | 연구용(데이터셋 약관) | 평가만 한다 |

**AGPL 결정(2026-10-05, 사용자)**
- 이 분할 모델(작업 이름 **"ObjectSAM"**)은 RecallVLA(Apache-2.0) 저장소와 따로, **AGPL-3.0 연구용으로 공개**한다.
  - FastSAM(CASIA-IVA-Lab)과 Ultralytics 를 출처로 밝힌다.
- 다시 학습한 FastSAM-s 와 YOLO26n 학생 모두 AGPL-3.0 이다. Apache 구조 학생은 따로 만들지 않는다.
- 가중치·ONNX·엔진은 Apache-2.0 저장소에 넣지 않는다. `~/ovdet_models` 는 git 밖이다.
- 이 학습 폴더는 AGPL-3.0 으로 표시한다(`LICENSE`).
- 참고:
  - 이 가중치를 쓰는 제품이나 서비스를 남에게 제공하면 AGPL 의무가 생긴다. 해당 소스 공개, 네트워크 서비스면 13 조다.
  - behavior-2026 제출물은 이미 AGPL-3.0 공개로 정해져 있다(09-30, `docs/제출지침.md`).
  - 실행 쪽 ovdet(C++/CUDA)에는 Ultralytics 코드가 없다.

## 남은 일

1. objprob 매개변수를 새 엔진으로 다시 맞춘다. `objprob_fit.py` 의 label_prior·κ·합치기 로지스틱이 대상이다. 의자 + 식탁 같은 덜 나눔이 objprob 의 합치기와 겹친다.
2. 실제 집 영상(LIMO)으로 확인한다. OpenLORIS 사무실에서는 t 0.05 판이 노드를 늘렸다.
   - 실제 영상용 문턱(t ≈ 0.1)과 시뮬 재현율용 문턱(t ≈ 0.05) 사이를 고르거나, 도메인별 문턱을 둔다.
3. 덜 나눔을 줄인다. 이웃 정답 경계에서 마스크를 자르는 학습 신호를 만든다(예: 정답 쌍 사이 경계 픽셀을 음성으로 하는 마스크 손실). 또는 모자이크 없이 마지막 epoch 를 더 돈다.
4. 시뮬 문: 가까운 문면처럼 애매한 프레임을 평가에서 따로 표시한다. 문 라벨(문짝 + 문틀)도 검토한다.
5. 렌더가 끝나지 않은 학습 장면: school_chemistry·school_computer_lab_and_infirmary·school_gym 과 과제 템플릿 일부. v2 에는 들어가지 않았다.

## 공개 저장소

- 분할 모델과 학습 도구는 **ObjectSAM**(AGPL-3.0)으로 따로 공개한다: https://github.com/juyoung020/ObjectSAM
- 이 폴더가 원본이다. `publish_objectsam.sh` 가 이 PC 전용 경로를 빼고 `tools/` 로 복사한다.
- 공개용 가중치는 `sanitize_release.py` 로 만든다. 로컬 경로·옵티마이저·git 정보를 뺀다.
