# 지도를 학습 state에 수치로 넣기 — 계획

구상: ① `dynamic-object-mapping-benchmark`로 GT 지도를 만들고, ② 우리 지도 업데이터를 만들고, ③ 지도 자체를 학습 state에 수치로 넣는다.
이 문서는 각 단계가 가능한지와 순서를 정리한다. 1–4단계 진행 상황과 잰 값은 맨 아래 §6.

## 1. 벤치마크가 주는 것 / 안 주는 것 (`src/dynamic-object-mapping-benchmark`, 직접 읽음)
| 주는 것 | 안 주는 것 |
|---|---|
| Office 장면에서 카메라가 같은 경로를 두 바퀴 돈다. 바퀴 사이에 물체가 옮겨지고(moved), 사라지고(removed), 추가되고(added), 바뀐다(swapped) | **로봇도 행동(action)도 없다.** 카메라 자세만 있다 |
| 프레임마다 RGB(8bit), 깊이(uint16 mm, z-depth), 인스턴스 마스크, 카메라 자세(TUM, `world_T_camera`), 내부 파라미터. 10 Hz, 720×480, 시퀀스 3개(static, dynamic1, dynamic2) × 2838프레임 | 우리 로봇(LIMO + OMX)의 영상 |
| GT: `objects.csv`(클래스, 이동 가능 여부, 크기), `object_poses.csv`(시간별 물체 중심 위치), `changes.csv`(변화 종류와 시각) | 파이프라인 학습용 데모 |
| 채점: 예측 `map_timeline.csv`(`frame,obj_id,x,y,z[,moving]`)로 정적/변화/동적 P·R·F1 | |
- 공개된 다른 방법(ConceptGraphs, DualMap)은 moved/removed가 0점이다. 같은 `obj_id`를 이동 후에도 유지하는지를 재는 벤치마크다.
- (10-04 받음, §6) 데이터는 원래 이 PC에 없었다(씬 매니페스트만 있음). 받을 수는 있다: 팀 비공개 릴리스 `dataset-office-seed1-v1`, raw 3개 약 1.3 GB씩(합 3.9 GB). `gh`로 접근 확인함. 다시 렌더하려면 Isaac Sim **4.5.0**이 필요하다(이 PC `behavior` env는 5.1, 호환은 확인 안 함).

## 2. 단계별 판단
**① GT 지도** — 가능. `objects.csv`+`object_poses.csv`로 프레임별 "정답 지도"가 나온다. 단, 전체 GT는 안 본 물체까지 다 담는다. 학습 state의 GT는 "그 프레임까지 인스턴스 마스크에서 충분히 보인 물체만"으로 걸러야 업데이터가 실제로 낼 수 있는 값과 맞는다(벤치마크도 `>= 400 px in >= 10 frames`를 "well seen"으로 쓴다).

**② 우리 지도 업데이터** — 팀 `scenemap`(`~/behavior-2026/src/scene_graph/scenemap`, CPU C++)이 이미 물체 상태 SEEN/GONE/MOVED/HELD와 `map_timeline.csv` 출력 도구(`tools/map_timeline.cpp`)를 갖고 있다. 다만 그 도구는 BEHAVIOR 에피소드(`ep_*.bin`, R1 Pro proprio)만 읽는다. **벤치마크 시퀀스를 읽는 어댑터가 필요하다**: RGB/깊이/자세 → `sm_push_image`(`depth_m`, `fx fy cx cy`)와 `sm_push_pose`(GT 자세 모드). 검출은 처음엔 인스턴스 마스크(= '완벽한 검출')로 넣어 업데이터 로직만 따로 채점하고, 그다음 FastSAM + SigLIP2로 바꾼다.

**③ 지도를 state 수치로** — 가능. 아래 §3. 단, 이 단계는 벤치마크 밖의 데이터가 필요하다(§4).

## 3. state 수치 형식 (제안 — 4단계에서 학습 쪽 `MapTok` 으로 대신함, §6)
로봇 프레임(`base_link`) 기준으로 `map → base_link` TF를 곱해서 만든다.
- 물체 슬롯 K개(예: 16, 가까운 순, 부족하면 패딩 + 마스크): `[존재, 클래스 id, x, y, z, 크기 3, 상태 one-hot 4(SEEN/GONE/MOVED/HELD), 마지막 본 뒤 경과 시간, 방 id]` = 슬롯당 약 14개 → 224개.
- 물체끼리 `on/in/near` 관계는 state에 넣지 않는다(그래프에서도 뺐다). 위치와 크기가 있으므로 "위에 있다 / 안에 있다"는 LLM이 추론한다. 물체의 소속은 방(`room id`)만 둔다.
- 가까운 장애물: 방향별 거리 8개(점유 격자에서 광선 투사).
- (선택) 로컬 점유 격자 64×64, 5 cm.
- 정규화 통계는 학습 데이터에서 계산해 저장하고, 실제 로봇도 같은 값을 쓴다.
- 이 형식은 팀 `scenemap.h`의 `sm_object`, `sm_grid`와 필드가 대응된다. 같은 C ABI 구조체에서 바로 변환할 수 있다.

## 4. 놓치기 쉬운 점
- **학습에는 GT 지도가 아니라 업데이터가 낸 지도를 쓴다.** GT로 학습하고 배포 때 업데이터 출력을 넣으면 분포가 어긋난다(위치 오차, 누락, ID 바뀜). GT는 업데이터 채점과 상한 비교에만 쓴다. 그래서 순서는 ② → ③이다.
- 벤치마크에는 행동이 없으므로 "지도를 입력으로 하는 정책"의 모방 학습 데이터는 여기서 나오지 않는다. 우리 로봇의 (영상, 상태, 행동) 데이터와 짝을 지어야 한다. 그 데이터는 시뮬레이터(`behavior-2026` 엔진 확장 또는 Isaac Sim)나 텔레옵에서 만들어야 한다.
- 벤치마크는 카메라 1대, 로봇 없음, Office 한 장면이다. 일반화 평가에는 부족하다.

## 5. 순서와 통과 기준
| 단계 | 할 일 | 통과 기준 |
|---|---|---|
| 1 | 벤치마크 raw 3개 내려받기, `validate` 통과 | 형식 검사 전부 통과 |
| 2 | 어댑터: 시퀀스 → `scenemap` C ABI, 완벽한 검출 입력 | `check-submission` 통과, `score`가 나옴 |
| 3 | 실제 검출(FastSAM + SigLIP2)로 교체 | static F1, change F1(moved, removed)을 완벽 검출 때와 비교 |
| 4 | 지도 → state 변환기(§3) 작성, 단위 시험(회전/평행 이동 불변성 포함) | 같은 지도에 로봇 자세를 바꿔 넣으면 상대 좌표가 맞게 변함 |
| 5 | 우리 로봇 데이터와 짝지어 학습 | 지도 없음 대 있음 비교 |

## 6. 진행 상황과 측정 (10-04)
벤치마크 저장소는 `refs/code/dynamic-object-mapping-benchmark`(697cd68)에 있다(`src/` 아래가 아님). 채점기는 그 저장소의 Python toolkit 을 `~/dom_venv` 에 설치해 오프라인으로 돌렸다.

**1단계 — 받기·`validate`: 통과.** `dataset-office-seed1-v1` 의 raw tar 3개(static 1,360,936,960 B, dynamic1 1,368,401,920 B, dynamic2 1,368,309,760 B, 합 4.1 GB, SHA256 확인)만 받았다(ROS bag 은 안 받음). `~/datasets/dom_office_seed1/data/<seq>`(dataset/README 의 hardlink 배치, tar 는 풀고 지움). `validate` 세 시퀀스 모두 `PASS: 0 error(s), 1 warning(s)`(물체 1135 monitor 의 높이 중심이 자세보다 +0.18 m — 데이터 쪽 경고), cross-sequence PASS. 각 2838 프레임, lap1_end 1418, depth_consistency 0.998.

**2단계 — 어댑터(완벽한 검출): `check-submission` 0 error 0 warning, `score` 나옴.**
- scenemap 은 카메라 외부 자세를 로봇 순기구학에서만 받아서, 카메라만 있는 기록을 넣을 ABI 하나를 더했다: `sm_set_cam_extrinsic(ctx, 0, T_bc)`(behavior-2026 `81f50b7`). 넣는 법: 베이스 = 카메라의 바닥 투영(x, y, 광축 수평 yaw) → `sm_push_pose`(SM_POSE_GT), 베이스 ← 카메라 = Rz(−yaw)·R, z = 카메라 높이 → `sm_set_cam_extrinsic`. 다시 합친 자세 오차 최대 2.2e-16. proprio 는 R1 61 개 0(팔 끝 베이스 아래 100 m, 손가락 열림 — 잡기·손 거르기가 안 걸리게).
- 도구: `src/scene_graph/scenemap/tools/dom_bench.cpp`(완벽한 검출 = 정답 인스턴스 마스크 ≥ 100 px + objects.csv 범주, 매 프레임), 공용 `dom_seq.hpp`(읽기·넣기·`map_timeline.csv`·`map_points.npz` 쓰기). 사라짐(GONE)은 지도에서 뺌, moving = HELD.
- 시간: 시퀀스당 약 24 s(CPU, 2838 프레임).

**3단계 — 실제 검출과 비교.** `src/scene_graph/runtime/tools/dom_bench_det.cpp`: ovdet FastSAM-s 416(TensorRT, conf 0.25 — sgrt 와 같음), 매 프레임, 평균 35.8 검출/프레임. 두 가지:
- **FastSAM 만(지금 sgrt 와 같음)**: 모든 검출이 이름 'object' 하나.
- **FastSAM + SigLIP 2 이름**(`--classify`): 검출마다 SigLIP 2 B/32 마스크 임베딩 → 글 프롬프트 67개(벤치마크 범주 37 + 구조물 wall·floor 등, 라벨 표 objects-v1 의 글 벡터) 코사인 최대를 cls 로 → scenemap 이 구조물은 물체로 안 만들고 같은 이름끼리만 잇는다. 검출 1.4 ms/프레임, SigLIP 9.1 ms/프레임(RTX 5070 Ti, 다른 작업과 같이).

세 시퀀스 합(class-agnostic 기본 채점, P / R / F1):

| 입력 | static 물체 | change 전체 | moved | removed | added | swapped | static 시퀀스 거짓 변화 |
|---|---|---|---|---|---|---|---|
| 완벽한 검출 | 0.975 / 0.869 / **0.919** | 0.259 / 0.292 / **0.275** | 0.143 / 0.167 / 0.154 | 0.250 / 0.333 / 0.286 | 0.333 / 0.667 / 0.444 | n/a / 0 / 0 | 1 |
| FastSAM + SigLIP 2 이름 | 0.299 / 0.670 / **0.414** | 0.025 / 0.312 / **0.047** | 0 / 0 / 0 | 0.064 / 0.250 / 0.102 | 0.043 / 1.000 / 0.083 | 0 / 0 / 0 | 186 |
| FastSAM 만('object') | 0.200 / 0.007 / **0.013** | n/a / 0 / 0 | 0 | 0 | 0 | 0 | 0 |

dynamic(움직이는 중 잡기) 행은 셋 다 0 — moving 은 로봇 그리퍼로 든 것(HELD)만 표시하므로 사람이 옮기는 물체는 표시되지 않는다. 비교: README 의 ConceptGraphs static 0.621, change 0.143(moved·removed 0).

완벽한 검출에서 고정 종류 표를 벤치마크 범주표의 movable = 0 범주로 바꿔도(`--bench-static`) static 0.918, change 0.280 — 거의 같다.

**진단(`score -v`)**
- 완벽한 검출의 change 실패는 업데이터 규칙에서 온다(검출은 정답):
  1. 사라짐 → 옮겨짐 잇기에 거리·시간 문턱이 없다(`objmap.cpp` "안 맞은 관측: 같은 이름의 확정 물체가 '사라짐'이면 그것이 옮겨진 것으로"). 치워진 책(removed 9·10·12)이 다른 곳의 같은 이름 관측과 이어져 moved 로 나옴 → removed FN + moved FP.
  2. 순서 의존: 옮겨진 물체의 새 자리를 옛 자리가 '사라짐'이 되기 전에 보면 새 id 가 된다 → moved 1·3·5·6 이 added + removed.
  3. 한 변 > 0.5 m(`big`)이거나 고정 종류(lamp·plant·desk …)는 사라짐 판정을 안 한다 → 옛 자리에 남은 물체(stale): removed 11(책 0.56 m), swapped 6쌍 전부(의자·안락의자·스탠드·모니터 0.53 m·화분이 낀 쌍).
  4. 같은 결과가 dynamic1·dynamic2 에 거의 그대로 나온다(눈앞에서 옮겨도 연속으로 따라가는 규칙이 없음).
- 실제 검출에서 더해진 실패:
  - FastSAM 만: 이름이 하나라 붙어 있는 마스크가 다 이어진다(같은 이름 + 상자 틈 < 0.10 m 면 같은 물체, 큰 물체는 상자 합집합) → 지도 끝 물체 6개.
  - FastSAM + SigLIP 2(static 시퀀스): FP 지도 물체 phantom 230(그중 119 가 'flat_item' 으로 이름 붙은 바닥 조각, 높이 중앙값 z = 0.0 m), duplicate 99(같은 물체가 프레임마다 다른 이름 → 같은 이름끼리만 이어서 따로 생김), 구조물로 빠진 것 78. 거짓 변화 186(moved 102, added 58, removed 19 …) — 이름 흔들림 + 위 1번(문턱 없는 잇기).

**4단계 — 지도 → state 변환기: 학습 형식에 맞춤(새 형식 안 만듦).** 학습 쪽에 이미 `training/RL/map/include/map_tok.h` 의 `MapTok`(칸 16 × 33 값 + 이름·생김새 번호 + 벽 56 + 방 10 + 완성도 4 + 안 본 곳 8, 1,280 B, base_link 기준)이 있어 §3 의 칸당 14 개 제안 대신 그것을 쓴다. `training/RL/map/include/sm_tok.h`: 진짜 scenemap 스냅숏(또는 물체 표) → 같은 `MapTok`(칸 순서·자리·FP16 반올림이 `make_tokens` 와 같음). §3 과 다른 점: 방 id 대신 같은 방 여부·방 종류 one-hot·가까운 문, 팔 끝 상대 위치·닿음·처음 자리에서 옮긴 양·속도·점수 등이 더 있다. scenemap 에 출처가 없는 값은 0/기본값(T_UNC, comp 4, front 8, app_id = NCLS) — 머리말 표.
- 시험 `tools/sm_tok_test.cpp`(ctest `sm_tok`): 442 경우 0 실패 — 지도 전체를 같은 SE(2)로 옮기면 토큰 같음(회전에서는 T_EEF_S 제외: 팔 끝 ↔ 상자 거리가 map 축 상자라 map 을 돌리면 바뀜 — GPU 형식 그대로의 성질), 평행 이동만이면 모든 값 같음, 로봇만 θ 돌리면 T_POS 가 Rz(−θ)·방위 −θ·거리/크기/상태 그대로, 로봇만 옮기면 T_POS = Rᵀ(p − x), 목표 맨 앞·거리 순·빈 칸·문·속도.
- 아직: GPU `make_tokens` 와 같은 입력으로 비트 비교는 안 했다(GPU 쪽 입력이 `MapCore` 라 짝 만들기가 필요).

**남은 일**
- 업데이터 규칙 고치기(위 진단 1–3: 사라짐→옮겨짐 잇기 문턱, 새 자리를 먼저 본 경우 옛 물체와 잇기, 큰 물체의 사라짐 판정) 후 같은 어댑터로 다시 채점. 고치는 곳은 behavior-2026 `objmap.cpp`.
- 실제 검출: 이름을 프레임마다 정하지 말고 물체 단위로 모으기(이름 흔들림 → duplicate), 바닥 조각 거르기.
- dynamic 행: 사람이 옮기는 물체를 moving 으로 표시하는 규칙이 없다.
- 5단계(우리 로봇 데이터와 짝지어 학습)는 안 함.

재현:
```
cmake -S <behavior-2026>/src/scene_graph/scenemap -B build/sm && cmake --build build/sm --target dom_bench
cmake -S <behavior-2026>/src/scene_graph/runtime -B build/rt && cmake --build build/rt --target dom_bench_det
build/sm/dom_bench  ~/datasets/dom_office_seed1/data/<seq> preds/perfect
build/rt/dom_bench_det ~/datasets/dom_office_seed1/data/<seq> preds/fastsam_siglip --classify
~/dom_venv/bin/dynamic-object-mapping check-submission ~/datasets/dom_office_seed1/data preds/perfect
~/dom_venv/bin/dynamic-object-mapping score ~/datasets/dom_office_seed1/data preds/perfect
```
