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

**5단계 앞 — objmap 바뀜 규칙 고침(behavior-2026 `3ed710f`, 위 진단 1–3 과 실제 검출 실패).** 규칙은 `ObjParams` 기본값이라 로봇(sgrt·sm_bench, R1·LIMO)도 같은 규칙을 쓴다. 자세한 표는 behavior-2026 `src/scene_graph/scenemap/README.md` "물체 바뀜 규칙".
- 옮겨짐 잇기: 안 맞은 관측을 그 자리에서 사라짐과 잇지 않고 새 후보로 만든 뒤, keyframe 끝 `relink` 가 사라진 m ↔ 새 n 을 잇는다 — n 확정·3 번 이상, n 처음 > m 마지막(시간), 거리 ≤ min(8 m, 1 m + 1 m/s·시간 차), n 자리를 처음 검출한 거리 이하에서 5 s 넘게 전에 본 적 있음(처음 가 본 곳에서 찾은 것은 새 물체), 더 가까운 같은 이름이 n 이후 안 보였으면 30 s 기다림. 새 자리를 먼저 보면 새 id 로 생겼다가 옛 자리가 사라짐이 되면 옛 id 로 다시 이어진다.
- 사라짐: 큰 것·고정 종류도 판정(놓침 6, 4 s 또는 카메라 0.5 m 이동). 근거 = 물체 점 48 개 투영(시야·가림·크기, 이 물체를 검출한 가장 먼 거리 안), 새 시점의 놓침만(카메라 0.1 m·5° 또는 자리 너머가 보임), 다른 이름으로 검출되면 안 셈, 검출률로 필요한 놓침 수(`gone_eps` 0.02). 관측 5 번 미만은 사라질 때 지움.
- 이름 표(물체별 점수 합, 짝 후보 = 표 몫 ≥ 0.2, 병합은 IoU ≥ 0.5 면 이름 달라도), 바닥 조각 거르기(점 90 백분위 높이 < 0.05 m, 러그·카펫 제외), 움직이는 중 따라가기(3 번 잇달아 0.3 m/s 넘게 같은 쪽 + 쉬던 상자를 벗어남, 잘린 관측·0.6 rad/s 넘는 회전 제외; 어댑터 `moving` 열 = 옮겨짐 상태에서 두 프레임 잇달아 3 cm/프레임 넘게).
- `dom_bench_det --dump/--load`: 검출·이름 캐시(시퀀스당 6 MB) — GPU 없이 규칙만 다시 돌려 비교.

다시 잰 점수(세 시퀀스 합, F1; 괄호는 P / R):

| 입력 | static | change | moved | removed | added | swapped | dynamic | static 시퀀스 거짓 변화 |
|---|---|---|---|---|---|---|---|---|
| 완벽한 검출, 전 | 0.919 | 0.275 | 0.154 | 0.286 | 0.444 | 0 | 0 | 1 |
| 완벽한 검출, 후 | **0.928** (0.988 / 0.875) | **0.591** (0.650 / 0.542) | 0.667 | 0.600 | 0.800 | 0 | **0.431** (0.902 / 0.284) | 1 |
| FastSAM-s + SigLIP 2, 전 | 0.414 (0.299 / 0.670) | 0.047 | 0 | 0.102 | 0.083 | 0 | 0 | 186 |
| FastSAM-s + SigLIP 2, 후 | **0.687** (0.675 / 0.699) | **0.152** (0.095 / 0.375) | 0.195 | 0.063 | 0.203 | 0 | 0 | 48 |
| ConceptGraphs(참고) | 0.621 | 0.143 | 0 | 0 | | | | |

- 실제 검출 static 시퀀스 지도 FP: phantom 228 → 16, duplicate 99 → 28. 떼어 보기: 바닥 조각 거르기 끔 static 0.555·change 0.110, 이름 모으기 끔 0.688·0.156. `gone_eps 0.1` 이면 0.697·0.187 이지만 R1 에서 맞은 물체가 줄어 기본은 0.02.
- 남은 실패: swapped 0(같은 이름 쌍은 생김새 없이 구별 불가, 9–20 m 교환은 잇기 거리 8 m 밖), 실제 검출 removed 가 0.102 → 0.063(사라짐 근거를 엄하게 해 dynamic 시퀀스 stale 이 늘어남 — 대신 static 시퀀스 거짓 removed·실제 물체 놓침이 줄어 static R 0.670 → 0.699), 실제 검출 dynamic 0(FastSAM 마스크가 움직이는 동안 쪼개지고 이름이 흔들림).
- 우리 로봇(R1 기록 3 개 × slam·gt·odom, `sm_bench` 와 `sgrt_replay` 결과 같음): 자세 CSV·`map.pgm` 바이트 같음. 9 판 합 확정 물체 275 → 249, 사라짐 21 → 13, 옮겨짐 9 → 22(odom 떠밀림 다시 잇기 대부분), 정답(`gt.csv.objects.json`) 대비 이름 맞는 짝 49 → 49, 아무 물체에도 없음 132 → 122, 다른 물체 위 59 → 52, 찾은 정답 물체 63 → 62. objmap 257 → 374 µs/keyframe. LIMO explore 기록(`gt_poses.csv.objects.json` 있는 것)은 sgrt 기록(rec.bin)이 없어 재생 못 함 — 규칙·값은 LIMO 도 같다.
- GPU 근사판(`training/RL/map/include/map.h`)에 옮길 것(아직 안 고침): ① 짝짓기 이름 비교 → 이름 표 몫 ≥ 0.2(다르면 키 +0.02) ② 바닥 조각 거르기 ③ 안 맞은 관측의 즉시 옮겨짐 잇기(1148–1160 근처) 없애고 `appeared` + `relink` ④ 부재 확인(1194–1208): 고정·큰 것 포함, 표본 투영·검출 거리·새 시점·다른 이름·검출률 k·카메라 0.5 m ⑤ 관측 < 5 사라짐 지움 ⑥ 움직임 따라가기 ⑦ 병합 이름 달라도 IoU ≥ 0.5. 나머지 값(confirm·prune_s·moved_d·gone_misses 3·gone_min_s 2·occl·da_*·big·grow_max·max_ext)은 그대로.
- **GPU 근사판에 옮김(2026-10-04)**: ①–⑦ 모두 + LIMO 잡기 확인·잡는 점·팔 가림(d58c978) + 새 LIMO SLAM 잡음 맞춤(84b373c, `odo_t` 0.015·`kf_corr_xy` 0.003 — 기록 넷 평균 rms 2.01 cm / 0.309° 에 모형 1.96 cm / 0.307°) + 토큰 `T_REACH` 를 잡는 점 작업 공간 표로. 단순화: 이름 표 칸 4, 부재 표본 = 상자 27 점(점 구름 없음), 본 곳 칸 시각 0.5 s 단위. `map_verify` 모든 설정 GPU == CPU 비트 동일, 새 음성 대조 5 종(이름 표·움직임·사라짐 근거·잇기·병합) 실패(정상), `map_realcheck` 잡기 규칙 진짜 objmap 과 33 스텝 같음, `map_cmp` 5.2 통과 + 바뀜 비교(컵 0.8 m 밀기, 판 76: 새 자리 확정 근사 25 / 진짜 24, 옮겨짐 10 / 11, 같은 판 75). 정적 상자 방에서 거짓 사라짐 47 → 0. 지도 단계 처리량 N 32,768 1.57 → 2.34 ms/스텝. 자세한 표는 `training/RL/map/README.md` "scenemap 바뀜 규칙…", `training/RL/map_cmp/README.md`.

**남은 일**
- swapped·같은 이름 옮김을 가리려면 생김새(구름 모양·SigLIP 임베딩) 비교가 필요.
- 실제 검출 removed·dynamic: 검출 마스크 쪼개짐·이름 흔들림이 남은 원인.
- GPU 근사판을 위 ①–⑦ 로 맞추기(학습 쪽 작업).
- 5단계(우리 로봇 데이터와 짝지어 학습)는 안 함.

재현:
```
cmake -S <behavior-2026>/src/scene_graph/scenemap -B build/sm && cmake --build build/sm --target dom_bench
cmake -S <behavior-2026>/src/scene_graph/runtime -B build/rt && cmake --build build/rt --target dom_bench_det
build/sm/dom_bench  ~/datasets/dom_office_seed1/data/<seq> preds/perfect
build/rt/dom_bench_det ~/datasets/dom_office_seed1/data/<seq> preds/fastsam_siglip --classify [--dump dets/<seq>.gz]
build/rt/dom_bench_det ~/datasets/dom_office_seed1/data/<seq> preds/replay --load dets/<seq>.gz   # GPU 없이 규칙만
SM_OBJ_PARAMS="gone_eps=0.1" build/sm/dom_bench ...                                                   # 매개변수 비교
~/dom_venv/bin/dynamic-object-mapping check-submission ~/datasets/dom_office_seed1/data preds/perfect
~/dom_venv/bin/dynamic-object-mapping score ~/datasets/dom_office_seed1/data preds/perfect
```

## 7. 실제 데이터 (10-05, 공개 실제 로봇 bag)

지금까지는 전부 시뮬이었다. 처음으로 실제 로봇 기록에 같은 파이프라인(ovdet FastSAM-s 416 + SigLIP 2 이름 → scenemap `limo_omx` 매개변수, 자세 = 바퀴 오도메트리 + 깊이 스캔 맞추기)을 돌렸다.
도구는 behavior-2026 `src/scene_graph/tools/realbag/`(`9dca06e`): ROS 없이 `rosbags` 로 bag → 스트림 변환(파이썬은 변환만), `realbag_run`(C++), `sgs_play`(기록 스트림을 sgview 로 벽시계 재생).

**자료**(받은 것 12.0 GB, `~/datasets/realbags`)
| 자료 | 로봇·센서 | 정답 | 라이선스 |
|---|---|---|---|
| OpenLORIS-Scene office1-1..7(rosbag tar 9.27 GB, 225 s) | Segway 배달 로봇, RealSense D435i 848×480(카메라 높이 0.916 m, TF), 바퀴 `/odom` 20 Hz | OptiTrack(`/gt`) | CC BY-ND 4.0 |
| TUM RGB-D fr2 pioneer_360·pioneer_slam(2.74 GB) | Pioneer, Kinect 640×480, 바퀴 `/pose` 10 Hz | 모캡(groundtruth.txt, 카메라 자세) | CC BY 4.0 |
dynamic-object-mapping-benchmark 의 bag 은 시뮬(Isaac Sim)이라 실제 자료에서 뺐다.

**넣는 법**: 영상 15 Hz(깊이는 컬러에 맞춘 것), 검출 5 Hz(3 프레임마다, sgrt kf_every 6 @ 30 Hz 와 같음), 나머지 프레임은 깊이로 지도만. 카메라 외부 자세 `sm_set_cam_extrinsic` — OpenLORIS 는 TF 그대로,
TUM 은 TF 가 이름뿐(카메라 수평·0.30 m)이라 깊이 바닥 평면(점이 가장 많은 수평면)으로 높이 0.584 / 0.598 m·숙임 5.9° 를 정함(모캡 카메라 숙임 약 5–6°). 오도메트리는 LIMO proprio 0–5, 영상 시각에 보간한 것 하나를 더 넣음. 깊이는 4 m 넘는 것을 버림(아래).

**SLAM·지도**(카메라 xy ATE, cm. SE(2) 맞춤 = 궤적 전체 최소제곱 / 첫 프레임 맞춤 = 출발 자세만 맞춤 — 실제 쓸 때의 떠밀림 / 그 최대. 지도 = SLAM 지도 점유 칸이 같은 스트림을 정답 자세로 만든 지도의 점유 칸 ±5 / ±10 cm 안인 비율, 그 반대 재현율 ±10 cm)
| 판 | 길이 s | 정답 경로 m | SLAM ATE | 오도메트리만 ATE | yaw rms ° SLAM / odom | 지도 정밀 5 / 10, 재현 10 (SLAM) | 같은 값 odom 10 / 재현 | 아는 / 빈 / 점유 m² | 벽 선분(정답 자세 지도) | 물체 / 같은 이름 0.5 m 안 쌍 | 검출/프레임 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| office1-1 | 27 | 5.6 | 4.8 / 9.6 / 15.8 | 3.5 / 13.3 / 20.1 | 1.9 / 1.8 | 0.92 / 0.98 / 0.94 | 0.98 / 0.98 | 29 / 11 / 17 | 52 (54) | 176 / 70 | 37 |
| office1-2 | 30 | 5.9 | 9.3 / 13.9 / 24.8 | 6.9 / 17.2 / 28.3 | 1.3 / 1.3 | 0.92 / 0.98 / 0.95 | 0.98 / 0.98 | 24 / 11 / 12 | 19 (49) | 111 / 28 | 26 |
| office1-3 | 12 | 1.0 | 0.9 / 4.2 / 8.2 | 1.2 / 9.8 / 18.2 | 5.2 / 16.1 | 0.81 / 0.89 / 0.84 | 0.64 / 0.64 | 16 / 5 / 11 | 38 (49) | 81 / 23 | 32 |
| office1-4 | 29 | 6.6 | 9.3 / 21.8 / 39.7 | 7.9 / 27.7 / 51.0 | 1.6 / 3.6 | 0.91 / 0.96 / 0.92 | 0.94 / 0.92 | 33 / 12 / 19 | 23 (59) | 122 / 37 | 29 |
| office1-5 | 53 | 11.3 | 8.3 / 17.7 / 27.8 | 8.9 / 11.0 / 20.9 | 1.3 / 1.3 | 0.94 / 0.98 / 0.96 | 0.98 / 0.96 | 40 / 15 / 24 | 41 (73) | 127 / 21 | 19 |
| office1-6 | 36 | 5.8 | 3.0 / 7.4 / 13.9 | 3.7 / 8.2 / 15.1 | 1.5 / 1.9 | 0.96 / 0.99 / 0.97 | 0.99 / 0.98 | 29 / 10 / 19 | 60 (79) | 183 / 69 | 37 |
| office1-7 | 38 | 6.1 | 7.7 / 11.1 / 21.3 | 7.4 / 10.1 / 19.9 | 1.5 / 0.7 | 0.95 / 0.99 / 0.98 | 0.97 / 0.96 | 24 / 12 / 12 | 12 (36) | 119 / 39 | 27 |
| TUM pioneer_360 | 72 | 16.2 | 12.5 / 38.3 / 58.7 | 15.7 / 43.2 / 64.2 | 4.7 / 5.4 | 0.56 / 0.67 / 0.64 | 0.60 / 0.60 | 107 / 93 / 5 | 16 (13) | 15 / 1 | 14 |
| TUM pioneer_slam | 155 | 41.0 | 15.4 / 25.8 / 48.4 | 22.4 / 37.8 / 67.6 | 3.1 / 4.5 | 0.80 / 0.92 / 0.86 | 0.80 / 0.75 | 120 / 104 / 10 | 10 (38) | 78 / 62 | 15 |

- 시뮬 LIMO(scenemap README "LIMO SLAM", 정답 대비 RMS 0.7–2.7 cm, 점유 ±5 cm 95–97.5 %)보다 실제가 크게 나쁘다: 첫 프레임 맞춤 ATE 4–22 cm(사무실), 26–38 cm(TUM). 맞추기가 오도메트리보다 나은 판은 첫 프레임 맞춤으로 9 판 중 7 판(1-5·1-7 은 더 나쁨), 1-3(제자리 회전)에서 yaw 16.1° → 5.2°.
- 지도는 정답 자세로 만든 지도와 점유 칸이 ±10 cm 안에서 0.89–0.99(사무실) 맞지만 벽 선분은 정답 자세 지도의 23–76 %(떠밀림으로 벽이 두 겹·번져 곧은 선이 덜 뽑힘). TUM pioneer_360 은 0.67.
- 로봇 매개변수 비교(office1-1·1-5, 같은 검출): `r1pro`(높이 띠 0.10–1.80 m, 맞추기 σ 1 cm)는 ATE SE(2) 18.3 / 21.3 cm 로 `limo_omx`(4.8 / 8.3)보다 나쁨 — 띠를 바꾸지 않았다. 깊이 상한: 8 m 면 office1-1 점유 51 m² · 빈칸 10.7 m²(먼 깊이 잡음이 책장 앞을 점유로 칠함), 4 m 면 17 / 10.7 m², ATE 4.1 → 4.8 cm. 3 m 는 아는 영역 20 m² 로 줄어 4 m 를 기본으로.

**물체 기억**(office1-1, 확정 176 개 중 best view 90 개를 눈으로 확인): 이름이 맞는 것 약 27 개(30 %). 나머지는 대부분 실제 물체지만 이름이 틀렸거나(본체 → speaker·modem·phone, 유리 책장 → monitor·box), 같은 물체의 조각(쌓인 본체·모니터 더미, 책장 문)이다. 같은 이름 0.5 m 안 쌍 70 개. 시뮬 벤치마크(같은 검출, static P 0.675)와 비교하면 실제가 훨씬 나쁘다.

**바뀜**(office1-1..7 을 정답 자세로 한 지도에 이음 — 판 사이 재위치 추정이 없어서. 논문: 1-2 같은 장면 반대 경로, 1-4·1-5 조명 다름, 1-6 물체 바뀜, 1-7 사람). 판별 사건(gone / moved): 1-2 17 / 12, 1-3 1 / 4, 1-4 22 / 16, 1-5 56 / 38, 1-6 33 / 41, 1-7 25 / 30.
바뀜이 없는 판(1-2·1-4·1-5)에서도 같은 수준이 나와 실제 바뀜(1-6: 책장 앞 책상·의자·모니터·선 더미가 치워짐 — 같은 자리 영상 짝으로 확인)과 구별되지 않는다. 그 자리(1-6 첫 시점 2 m 안) 1-1 물체 19 개 중 사라짐 4·옮겨짐 3·지움 3·사건 없음 11. 시뮬(실제 검출 static 시퀀스 거짓 변화 48)보다 거짓 바뀜이 많다.

**실제에서 본 실패**
1. 검출 과분할: FastSAM 이 쌓인 본체·모니터·유리 책장 문을 조각으로 → 물체 수가 실제의 몇 배, 같은 이름 쌍 많음.
2. 이름: SigLIP 2 마스크 임베딩이 실제 영상(어두움·흐림·자동 노출)에서 흔들림 — 본체 = speaker/modem/phone, 의자 = mouse.
3. 깊이 잡음: D435 의 먼 깊이(> 4 m)가 넓은 점유 얼룩을 만듦, Kinect 는 유리·검은 면 구멍.
4. 높은 카메라·좁은 세로 화각(0.92 m, ±21°): 바닥이 2.4 m 밖에서만 보여 빈칸이 적다(office1-1 아는 29 m² 중 빈칸 11 m²).
5. 움직임 흐림·조명(1-4·1-5 어두움): 흐린 best view, 조명 판에서 거짓 사라짐.
6. SLAM 떠밀림: 좁은 시야·복도 퇴화 — 첫 프레임 맞춤 4–38 cm, 벽 두 겹.
7. 사람(1-7): person 은 구조물로 빠지지만 사람에 가린 자리가 사라짐 근거가 됨.

**고칠 것**: 검출 조각 병합(같은 깊이 면·인접 마스크 합치기) 또는 다른 검출기(선택은 `docs/model_selection.md` 에서 미정), 이름은 여러 시점 투표·확신 문턱, 깊이 상한·센서별 잡음 모델을 로봇 설정으로, 사라짐 판정에 조명 변화·가림(사람) 근거 거르기, SLAM 은 고리 닫기·재위치(여러 판 이어 붙이기에 필요), 높은 카메라 로봇은 빈칸 근거를 바닥 아닌 벽 광선으로도.

**보기**: 학습 뷰어 재생 판 `~/trainview_work/real_bags/{ol_office1-1,ol_office1-5}`(run.json group `real_bags`, `trainview --root ~/trainview_work/real_bags`), sgview 실시간 재생(`sgs_play` → `--ingest`, 자세 60 Hz·지도 15 Hz·요약 6 Hz @ 1×), README 맨 위 영상(office1-5, 3배속).
