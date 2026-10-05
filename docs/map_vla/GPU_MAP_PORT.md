# GPU 학습 지도 다시 옮기기 — objprob 구조로, 입력 하나로 (2026-10-06)

작성 2026-10-06. **설계 + 진행 기록**이다. 단계마다 끝에 "상태" 를 적는다(잰 값만, 추정은 **(추정)**, 가정은 **(가정)**).

## 0. 결정과 범위

사용자 결정(10-06):
- GPU 학습 지도(`training/RL/map`, 정책 지도 G2)와 토큰·관측 만들기를 **지금 진짜 파이프라인 구조**로 다시 옮긴다:
  ObjectSAM(`yolo26n-seg-obj-416`) 분할 → SigLIP 2 이름·벡터 → scenemap 확률 모드(objprob: vMF 벡터 합치기, 이름 베이즈 합치기,
  이름 없는 베이즈 같은 것 판정, 칼만 자리, RANSAC 평면 구조물 지우기, 문·창 크기 확인, 벽 높이 규칙, 살펴본 정도 3).
- 옛 규칙 잔재(objmap 이름 규칙, 16 칸 가까운 순, 교사 전용 격자, C0/C1, 고정 창)를 지운다(`archive/`).
- **입력 하나**: RL 교사 입력 = BC·RecallVLA 학생 입력 = 실제 로봇 입력. 예외는 RL 교사의 카메라 RGB 뿐(속도).
  PPO 행위자 = 그 입력에서 카메라 빼고, 비평자만 특권 값. 예전 RL 체크포인트는 다시 쓰지 않는다.
- GPU 포팅 유지(학습 고리 안에 CPU scenemap 없음). 속도: 지금 환경 + 지도 한 스텝의 **+10 % 안**(nsys 로 잼).
- 같이 맡은 점검 항목(`LEGACY_AUDIT_10-06.md`): 1·2(벡터 공간 하나 = SigLIP 2), 4(GT 지도 설정), 5(교사 특권), 6(창 → 집 전체),
  7(잡음 다시 잼), 8(옛 입력), 9(PPO 집 나누기), 10(레포 밖 표), 11(이름 표 빠진 24 개), 8절 방 종류(참값 → 예측 + 잡음),
  1절 머리 `h`(되도록 버림). R1 기본값·URDF 헤더·`near` 인자·YOLOE 흔적·`mock_eval.sh` 는 다른 에이전트 몫(손대지 않음).

원본(읽기만): behavior-2026 `src/scene_graph`(robot-agent `src/scene_graph` 사본 — `tools/sync_scene_graph.sh --check` 차이 없음, 서브모듈 `75ecd76`):
`scenemap/src/objprob.cpp`, `objmap.cpp`(`ObjectMap::update` objprob 길, `apMergePass`, `apStructObject`, `apRename`), `inspect.cpp`,
`bestview.cpp viewKappa`, `walls.cpp`, `grid.cpp`, `runtime/src/objprob_front.hpp`(낱말·라벨 표·사전·엔진별 매개변수),
`tools/realbag/objprob_params/yolo26n-seg-obj-416.json`(로지스틱 `ap_w*`·`ap_wm*`, 문턱 0.5/0.7, κ).

### 0.1 단계와 상태

| 단계 | 일 | 상태 |
|---|---|---|
| P0 | 이 문서 | 씀(10-06) |
| P1 | 물체 층 GPU 포팅: 검출원(가구 전부 + 과제 물체), 잡음 모형, objprob 같은 것 판정·합치기·이름·칼만·병합·구조물·살펴본 정도, 물체 저장 SoA(N ≤ 256) | 진행 |
| P2 | 기억 표(W1)·방 항목(W2)·방향 구역(W2)·물체 방 종류(W3) 만들기 — 장치 함수 하나(`map/include/mem_tok.h`) | 대기 |
| P3 | 입력 만들기 하나(`observation/include/inputs.h`): 교사 X0·학생·RecallVLA `VBatch`·실제 `sm_tok.h` 가 같은 함수, 설정 해시 | 대기 |
| P4 | PPO 길 바꾸기: 행위자 = 통일 입력(카메라 뺌), 비평자 = 특권, 참 벽 광선 16·참 목표 자세 뺌, 대본 교사 믿음 지도 | 대기 |
| P5 | 벡터 공간 하나(SigLIP 2 768 → 저장 고정 투영 128), 이름 표 다시 만들기, 생김새·확신도 한 정의 | 대기 |
| P6 | 고정 창 → 집 전체(집 좌표, 로봇 따라 움직이는 고운 창 + 집 전체 거친 층) | 대기 |
| P7 | 옛 것 `archive/`, 설정(C2·집 나누기), 기록 도구 가드 | 대기 |
| P8 | 잡음 맞춤(다른 에이전트의 입력 단위 비교 BASELINE) + nsys 속도 표 + 교사 C2 B4/B5 다시 잼 | 대기 |

### 0.2 결정(10-06, 사용자 — 코디네이터 전달)

- **인지 흉내 층(perception emulator)을 objprob 합치기와 나눈다.** 장치 인터페이스 하나: 입력 = 참 장면 물체·참 카메라 자세·믿는 자세·보임,
  출력 = keyframe 마다 **ObjectSAM + SigLIP 2 가 냈을 검출 목록**(`map/include/percept.h` 의 `PDet`: 보임 면 중앙값 자리·10–90 백분위 상자·카메라 깊이·κ·
  출처 꼬리표·섞임·이름 상위 4 의 log p(c|z)·나머지·잘림). 합치기(`objprob_gpu.h`)는 이 목록만 읽는다 — 흉내 층을 학습형 PEM(작은 MLP/GRU)·시점 캐시·
  진짜 검출로 바꿔 끼워도 합치기·입력·교사는 그대로. 배경: `PERCEPTION_EMULATION_SURVEY.md`.
- **지금은 통계 흉내**(값싸게, keyframe 커널 안 물체마다 산수만, 광선 더 없음): 보임 확률(거리·크기·시선 각·가림·높이), 이름 혼동 상위 k,
  물체의 SigLIP 2 벡터 둘레에서 뽑은 벡터(아래 R7 — 벡터 대신 출처·κ 로 나타내고 cos 는 진짜 벡터 표로), 합침·유령 비율, 거리별 자리 오차,
  **물체마다 2 상태 마르코프(보임/놓침) — 놓침이 keyframe 을 넘어 이어짐**, slam 표류는 AR/마르코프(지금 걸음 잡음 + 판 치우침 + keyframe 되돌림).
- **맞춤 자료**: 비교 에이전트의 OmniGibson + 진짜 파이프라인 판(정답 ↔ 검출 짝)으로 작은 오프라인 스크립트(`map_calib/tools/fit_percept.py`)가
  매개변수를 맞추고, 레포 json(`map_calib/percept/<엔진>.json`, 파이프라인 판 해시 포함)에 둔다. 파이프라인이 바뀌면 스크립트만 다시 돈다. `MP` 의 흉내 값은 이 json 에서 생성한 헤더로.
- **학습형 PEM 은 나중**: 통계 판이 검증을 못 넘을 때만. 인터페이스는 지금 준비.
- **검증(두 층)**: 입력 분포 대 진짜 파이프라인(검출률·혼동·오차·놓침 이어짐 길이), 뒤에 같은 정책을 흉내 입력 대 진짜 입력으로 평가한 성공 차(문턱 약 10 %p, 정할 것).
- PPO 롤아웃: 판·스텝마다 물체 스냅숏을 두고 갱신 때 **같은 입력 함수**로 기억 줄 256 을 다시 만든다(결정적 — 롤아웃 줄 == 다시 만든 줄 비트 검사). 실패하면 가까운 64 + 목표 줄.
- 생김새 원형: og_cmp 의 μ, 모자란 종류는 글 + 모달리티 차 — **문서·표에 가정으로 표시**.

## 1. objprob 규칙별 옮기기

표기: CPU = 진짜 scenemap 동작(파일:함수), GPU = 우리 설계, 근사 = 다른 점과 까닭. 모든 GPU 코드는 CPU 참조판과 **같은 소스**(`DEV`, `detmath.h`,
`--fmad=false`)라 `map_verify` 가 비트 비교한다. 관측(검출)은 keyframe 마다, 기억 표는 스텝마다(소비하는 커널 안에서) 만든다.

| # | 규칙 | CPU(진짜) | GPU 설계 | 근사·까닭 |
|---|---|---|---|---|
| R1 | 검출원 | ObjectSAM 마스크(사물만, 이름 없음 'object') → 마스크 깊이 점 중앙값·10–90 백분위 상자 (`objmap.cpp update` 1) | **과제 물체(prim, 움직이는 상자)** = 지금처럼 5 점 보임 광선 + 보이는 면 통계(`surf_stats`). **가구(장면 정적 상자 중 이름 있는 것, 집 전체)** = 깊이 광선 64 × 9 이 맞힌 상자 번호를 열마다 기록(`phase_cast` 가 이미 쏨 — 광선 더 없음), 맞은 광선 수 ≥ 2 인 상자를 검출 후보로, 보이는 넓이 ≈ 맞은 광선 수 × 화소 덩이 | 가구 보임을 거친 광선으로 셈(작은 가구 선반 위 물건은 prim 길). 까닭: 집 전체 가구(장면당 59–542 상자)를 5 점 광선으로 보면 keyframe 비용이 몇 배 |
| R2 | 바닥·천장·벽 조각 지우기(RANSAC 평면) | `apObsStructural`(조각 평면 맞춤: 얇은 수평면 천장/바닥, 세운 평면 + 벽 선 0.12 m + 크기 ≥ 1 m 또는 구조물 확률 ≥ 0.5, 높고 넓은 세운 평면), 합친 물체 `apStructObject`(벽 크기·천장 덩어리·구조물 사후 ≥ 0.6, 벽에 붙은 큰 덩어리 숨김) | 검출원이 사물(prim·가구)뿐이라 대부분의 벽·바닥 조각은 애초에 안 생김 = 진짜에서 지워진 것. **남는 헛것만 모형**: 벽 조각 유령(벽에 맞은 광선 열에서 `p_struct_ghost` 로, 크기·자리 = 벽 위 0.3–1.2 m 판), 그중 진짜 규칙을 통과해 남는 비율 `p_struct_keep`(BASELINE 의 벽·천장·바닥 헛노드 수로 맞춤). 남은 것은 구조물 사후가 높은 물체(이름 = 상위어 "structure" 쪽) | 평면 맞춤 자체는 안 함(점 구름 없음). 진짜 결과(남는 헛노드 비율)만 맞춤 |
| R3 | 벽 너머(창 밖) 버림 | `through_d` 0.20 m: 카메라 → 관측 중심이 벽 선분을 지나 0.2 m 넘게 | 장면 창(`BK_WINDOW`) 너머 물체는 광선이 창 상자에 맞아 애초에 안 봄. 규칙 없음 | 시뮬은 창을 막힌 상자로 둠 |
| R4 | 문·창·계단·기둥 크기 확인 | `apObsStructural so_keep`, `apStructObject`(so_dw_w 2.2, so_dw_thick 0.45, so_max_h 2.8, so_pillar 1.0, so_stairs 6.0, so_dw_min 0.3) | 문은 장면 문 목록(`SDoor`)에서: 문 자리를 광선이 보면 "구조 물체" 줄(이름 door, 가구 아님 표시)로 기억 표에 — 진짜도 문·창은 structural 노드로 냄(`exportable`). 크기는 장면 문 폭 × 2.1 m(크기 확인을 늘 통과) | 문짝 조각이 커튼·가방으로 불리는 실패는 이름 혼동(R9)에 둠 |
| R5 | 벽 높이 규칙 | 벽 선 없이 높고 넓은 세운 평면(`tall_h` 1.7·`tall_w` 1.5 m) = 벽 | R2 유령에 포함(키 큰 가구 앞판이 벽으로 지워지는 실패: 큰 가구 검출에 `p_tall_drop`, 냉장고·옷장 높이 ≥ 1.7·폭 ≥ 3 m 일 때만) | 비율만 맞춤 |
| R6 | 모습 신뢰도 κ | `viewKappa`: κ = k0·s/(s + s0)·(잘림)·1/(1 + (d/d0)²)·vis^occ, 매개변수 k0 4369·s0 40·trunc 1·d0 100(엔진 json) | **같은 식**, s = √(보이는 넓이 화소)(prim: 보이는 점 비율 × 투영 넓이, 가구: 맞은 광선 × 덩이 넓이) | s 를 화소에서 바로가 아니라 어림 넓이로 |
| R7 | 벡터 vMF 합치기 | `apAddView`: r += κ z(비슷한 시점 0.3 m·15° 안이면 κ × temper 0.3, 최근 시점 32), 조각/통째 따로, μ = r/‖r‖ | 물체마다 **벡터를 들지 않고** 두 출처 번호(주 출처 src, 섞인 출처 src2 와 무게 w2)와 합 K = Σκ(temper 그대로, 최근 시점 8 개 고리)만 둔다. μ 는 소비할 때 `app(src)`(출처 물체의 SigLIP 2 영상 원형 + 판마다 개체 흔들림) 를 섞고 κ 로 정한 잡음 방향을 더해 만든다: μ = l2((1−w2)·a(src) + w2·a(src2) + ε(K)·u), ε(K) = √((d−1)/(2K)) (vMF 평균 벗어남), u = 물체마다 고정 무작위 단위 벡터(번호 해시). 통째 다시 담기(`buildReencode`)는 "통째 κ = 지금 가장 좋은 모습" 으로 K 를 바꿈 | 벡터 128 × 물체 256 을 판마다 들면 판당 64 KB. 같은 물체의 모습은 평균 둘레 vMF 라 μ 가 K 로만 정해진다는 사실을 씀. 조각 모습의 벡터 차이(조각 vs 통째)는 ε 에 들어감 |
| R8 | 같은 것 판정(관측 ↔ 물체) | 로지스틱 P = σ(w·f), f = [접촉, 상자 틈, 중심 거리/크기, cos − 0.75, 겹침, 받침, 이름 겹침 0], 받침이면 −30, 문턱 same_p 0.5(엔진 json), 틈 > 0.30 m 는 안 봄, 구조물 막기(guard 0.6/0.2), 벽 같은 조각 막기, 납작한 벽걸이 2.2 m, 관측마다 P 최대 물체(≥ same_p) — 여러 조각이 한 물체에 붙음, P ≥ 0.2 또는 접촉 ≥ 0.3 이면 '보임'(`objmap.cpp update` 2 objprob 길) | **같은 식·같은 무게(엔진 json 값을 `MP` 로)**. 특징: 접촉 = 상자끼리 4 cm 칸 이웃 안 비율을 상자 꼴에서 닫힌 꼴로(관측 상자 표면 중 물체 상자 + 4 cm 안 비율), 틈·중심 거리·겹침(`da::boxOverlap` 0.05)·받침은 그대로, cos = 관측 z 와 물체 μ 의 cos(아래 R7 모형에서 닫힌 꼴: 같은 출처면 κ 로, 다른 출처면 두 원형 cos 표). 관측 ↔ 물체 쌍은 상자 틈 0.30 m 안만(물체 상자 256 개를 스레드가 나눠 거름) | 접촉을 점 구름 대신 상자 꼴로. cos 를 벡터 내적 대신 표로(출처 원형 cos 는 장치 표 `appcos[n_app][n_app]`, 물체 개체 흔들림 포함 닫힌 꼴) |
| R9 | 이름 베이즈 | `apLabelLogLik`: p(c|z) = σ(t·cos + b) 라벨 정규화, `apName`: log 사전(엔진 label_prior) + λΣw log p(c|z)(w = κ/3000, Σw ≤ 6) + 크기 가우스(`kApLabels`), 최대 ≥ 0.5 면 그 이름, 아니면 상위어 합 ≥ 0.5 인 가장 구체적인 것, 아니면 "object" | 라벨 수 C 가 64(+과제 이름)라 물체마다 C 개를 들 수 없음 → **관측마다 상위 4 라벨의 log p(c|z)** 를 장치 표 `labll[src_name][4]`(진짜 SigLIP 2 글 벡터로 미리 셈: 출처 이름의 원형 영상 벡터 vs 라벨 표의 글 줄 — 같은 `apLabelLogLik`) + 관측 잡음(κ 가 작을수록 넓게 — 아래 2절)으로 만들고, 물체마다 상위 6 라벨 + 나머지 합을 든다. 사후 = 사전 + λΣ + 크기 우도, 같은 문턱·상위어 규칙(`kApLabels` 부모 표를 장치에) | 상위 6 밖 라벨의 우도를 "나머지" 하나로 묶음. 표가 이름 혼동(문짝 → 커튼 등)을 진짜 글 벡터로 냄 |
| R10 | 칼만 자리 | 축마다 P += q·dt(q 1e-4), R = (0.02 + 0.01·깊이)² (+ 잘림이면 (반 폭)²), 작은 것만 자리·상자 갱신, 큰 것(> 0.5 m·고정·구조 물체)은 합집합 + 분산만 | **같은 식** | 없음 |
| R11 | 같은 영상 조각 묶기(frame_group), 덜 나뉜 마스크 버림(bridge_drop, 엔진 json 에 없음 = 끔), 조각 합치기 | `update` 3 끝, 2 | frame_group 같은 식(상자 2 cm 안, 접촉 1, cos = 두 관측 z). 조각 합치기 = 한 물체에 여러 관측이 붙으면 대표(점 수 최대)에 합집합 | 없음 |
| R12 | 물체끼리 병합 | `apMergePass`(다음 keyframe 앞): 틈 0.30 안·합집합 한 변 ≤ 4 m 쌍마다 wm 로지스틱 + 이름 분포 바타차리야, P ≥ merge_p 0.7, 큰 P 부터, 판마다 물체 한 번 | **같은 식**, 쌍 후보 = 이번 keyframe 에 본 물체 × 모든 물체(틈 안). 바타차리야는 상위 6 라벨 + 나머지로 | 나머지 묶음만 |
| R13 | 확정·버리기·사라짐·옮겨짐 잇기 | 서로 다른 keyframe 2 번 = 확정, 후보 10 s 버림, 부재 확인(objprob 길: 다른 조각이 닿으면 안 셈), relink(이름 대신 link_cos 0.8) | 지금 GPU 규칙(`obj_absence`·`obj_relink`·`obj_prune`) 그대로 옮기고 이름 조건을 objprob 길로(cos ≥ 0.8, touched) | 없음 |
| R14 | 살펴본 정도 3 | `inspect.cpp`: closest_view_m(붙은 관측의 카메라 ↔ 관측 중심 최소), n_views(0.3 m·15° 넘게 다른 시점 수, 32 에서 멈춤), top_seen(윗면 있는 물체 4 × 4 칸, 윗면 + 5 cm 점이 2 m·연직 80° 안에서 안 가리고 보인 비율) | closest = 같은 식. n_views = 시점 고리 8 개(자리 int16 cm·광축 각 int8)로 같은 판정 + 8 넘으면 그 뒤로는 "고리의 어느 시점과도 다름" 일 때 셈 → 8 까지 정확, 그 위 어림. top_seen = 16 점을 믿는 자세로 투영 → 그 화소의 참 깊이(참 카메라·참 장면 광선, 부재 확인과 같은 `cast_beh`)로 같은 판정. keyframe 마다 윗면 있는 물체 중 2 m 안·안 다 본 것만 | n_views 8 넘은 뒤 어림 |
| R15 | 이름 보기 확인(confirm_object) | `apObserveName`(바깥 이름 증거) | 학습에서는 쓰지 않음(에이전트 확인은 선택 — TRAINING_DESIGN D7) | — |

물체 저장(판마다, 전역 SoA, 판 수 N 이 가장 안쪽): 상자 lo·hi(float 6), 상태 바이트(valid·state·confirmed·moved·held·appeared·src kind),
시각(int: first·last_seen·last_kf·first_miss·trk_t·moving_t), 칼만 P[3](float), K(float), src·src2(int16)·w2(half), 이름 상위 6(int16 라벨 + float Σ),
살펴본 정도(closest half, n_views u8, 고리 8 × 4 B, top 비트 u16 + 칸 상자 4 half), 놓침·움직임 값. **물체당 약 180 B, 판당 256 칸 46 KB(가정: 칸 수)**,
N = 4,096 이면 189 MB. keyframe 블록은 이번 시야 근처 물체(상자가 카메라 3.5 m 원과 겹침)만 공유 메모리로 옮겨 다룬다(목록 ≤ 48, 넘치면 가까운 순).

### 1.1 P1 상태(2026-10-06, 잰 값) — 물체 칸 16 그대로, 검출원 = 과제 물체 prim + 유령

- 코드: `training/RL/map/include/percept.h`(흉내 층), `objprob_gpu.h`(합치기), `map.h`(Slot 에 objprob·살펴본 정도 값, 옛 이름 규칙 뺌 → `archive/training/RL/map/include/map_objmap_rules.h`).
  목표 칸·완성도의 "참 물체 짝" 은 이름 대신 **주 출처 꼬리표**(앱·에이전트가 물체 id 를 준 것과 같은 뜻 — 이름 혼동과 무관). 토큰 이름 = objprob 이름 사후(모름 −1), 이름 확신도 2 = 고른 이름 사후·1위 − 2위.
- `map_verify`(CPU == GPU 비트 동일, 모든 스텝): 상자 방 512 × 300 OK, BEHAVIOR `--stage 3 --curr 0.34,0.33` 512 × 300 OK, `--arm` OK. 음성 대조 모두 실패(정상):
  `--negative`(확정 규칙) 30,248, `--negative-name`(같은 것 판정에서 생김새 cos 끔) 19,215, `--negative-merge`(병합에서 생김새 cos 끔) 16,317 낱말 다름(256 × 200, BEHAVIOR).
- objprob 품질(정답 쪽, BEHAVIOR 섞음 판 끝 확정 칸 2,604): 유령 0.003, 같은 참 물체 중복 0.025, 섞임(w2 > 0.2) 0.003, 이름 맞음 0.930·상위어 0.010·모름 0.036·틀림 0.021,
  가까이 본 거리 평균 0.40 m(C0 미리 채운 칸 포함), 시점 0.93. 상자 방: 유령 0.018, 중복 0.061, 이름 맞음 0.775·모름 0.140·틀림 0.068.
- 속도(같은 GPU 에서 OmniGibson 비교가 도는 중 — 잡음 큼, nsys 표는 P8): BEHAVIOR N 4,096 자연 지도 단계 0.89–0.93 → 1.02–1.06 ms(약 +15 %), 환경+정책+지도 3.09–3.26e6 → 2.66–2.72e6 env-step/s.
  keyframe 블록 공유 메모리 11.3 → 약 16 KB(Slot 172 → 316 B × 16). 늘어난 몫(`map_prof` 상자 방): 병합 판정 11k 사이클(바뀐 쌍만 다시 셈 — 결과 같음을 확인), 짝짓기·갱신 7k → 16k, 흉내 8k → 14k.
  **+10 % 예산을 아직 넘는다** — P1b(물체를 전역 저장소로, keyframe 블록은 근처 물체만)에서 다시 줄이고 P8 에서 nsys 로 잰다.
- 아직 안 한 것(P1b): 물체 저장소 256(전역) + 가구 전부 검출(깊이 광선 맞은 상자 번호), top_seen, 구조물 유령(R2), 문 줄(R4). `sm_tok_test` 의 "teacher grid sees the wall ahead" 1 실패는 바꾸기 전 판에도 있음(교사 격자는 P2 에서 뺌).

### 1.2 objprob 계산 하나(10-06, 잰 값)

- `src/scene_graph/scenemap/include/scenemap/objprob_math.h`: 같은 것 로지스틱(`same_logit`)·기하 특징(`pair_geo` — 틈·중심 거리·cos·겹침·받침)·`sigmoid`·κ(`view_kappa`)·
  temper·이름 무게·λ·사후 정규화(`softmax_post`, 나머지 라벨 묶음 선택)·상위어 고르기(`better_hyper`)·이름 분포 겹침(`bhattacharyya`)·칼만(`kalman_gain`·`kalman_R`·`kalman_big`·`merge_var`).
  scenemap `objprob.cpp`·`objmap.cpp`·`bestview.cpp` 가 double(`OpmStd`)로, GPU 지도가 float(`OpmDet` — 결정적 exp·ln)으로 부른다. 접촉(점 구름 대 상자)·라벨 공간(전부 대 상위 6)·생김새(벡터 대 출처 꼬리표)만 쪽마다 다름.
- scenemap 그대로인지: `realbag_run` OpenLORIS office1-5(ObjectSAM yolo26n + SigLIP 2 + objprob + inspect, slam) 바꾸기 전후 — GPU 검출 판·`--load` 검출 캐시 판 모두
  출력 파일 전부 바이트 같음(metrics.json 은 시간·경로 값만 다름 — 나머지 같음). scenemap ctest 15/15.
- 맞춤 시험 `training/RL/map/tools/objprob_parity`(ctest `objprob_parity`, 음성 대조 `--negative` = GPU 쪽 cos 특징 빼면 실패): 합성 입력 20 만 — 같은 것 P 차 최대 1.0e-6·문턱 판정 0 다름,
  κ 상대 차 2.1e-7, 이름 사후(라벨 ≤ 6, 상위어 포함) 고른 이름 0 다름·사후 차 2.7e-7, 칼만 20 번 자리 차 1.2e-7 m. 근사 오차(판정 아님): 이름 분포 겹침 상위 6 + 나머지 대 전체 10 라벨 평균 0.031·최대 0.23.
  이 시험은 **규칙 단위**다(합성 관측 열을 진짜 ObjectMap 에 넣는 열 단위 비교는 깊이·마스크가 필요해 하지 않음 — 입력 단위 비교는 og_cmp 가 맡음).
- GPU 쪽 상위어 고르기를 한 단계 → 사슬 전체(scenemap 과 같은 규칙)로 바꿈. `map_verify` 다시 비트 동일(상자 방·BEHAVIOR).

## 2. 인지 잡음 모형(ObjectSAM + SigLIP 2) — 맞출 값

| 잡음 | 모형 | 처음 값(출처) | 맞출 것(BASELINE) |
|---|---|---|---|
| 놓침 | p_miss = σ(a0 + a1·ln(s/40) + a2·d + a3·잘림 + a4·시선 각) — s = √화소 넓이, d = 카메라 거리 | ObjectSAM results.md 크기별 재현율 (확인 뒤 적음) | 목표 물체 확정 비율·확정까지 keyframe 수 |
| 유령(헛것) | 물체 아닌 것에서 생기는 관측: 판마다 유령 자리(지금 `Ghost`) + 벽 조각 유령(R2) | objprob_refit 표(radio r3: 헛노드 3·1·1 / 노드 96) | 확정 물체 중 짝 없는 것 비율 |
| 쪼개짐(중복) | 큰 가구 관측을 조각 2–3 개로(상자 축 따라 자름, 확률 p_split·크기 ≥ 0.8 m) → 병합 규칙이 다시 합침 | radio r3 중복 45/96 (yolo26 json 의 "fit") | 중복 쌍 비율 |
| 덜 나뉨(합침) | 맞닿은 두 물체(식탁 + 의자, 상자 틈 < 5 cm)를 한 관측으로(p_under), src2·w2 = 두 출처 | 잘못 합침 12/96 | 잘못 합침 비율 |
| 이름 혼동 | R9 표(진짜 글 벡터) + 관측 잡음: log p(c|z) 에 가우스(σ = c0/√(κ/κ_ref)) | 표에서 | 확정 물체 이름 맞음 비율 |
| 벡터 잡음 | ε(K)(R7) + 개체 흔들림 σ_inst(같은 종류 다른 개체 cos ≈ 0.85 (추정)) | SigLIP 2 B/32 같은 종류 개체 cos(측정해 적음) | 같은 것 판정 P 분포 |
| 자리·크기 | 깊이 σ = 0.006 z², 옆 0.023 m, 크기 0.05 m(지금 값) | Dabai 데이터시트, limo3 | 위치 오차 50/90 % |

잡음 값은 **`MP` 한 곳**(지금과 같음)에 두고 출처를 주석에. BASELINE(다른 에이전트의 `og_cmp`·`map_cmp`, TRAINING_DESIGN 6.1 인수인계)이 오면 잡음만 다시 맞춘다(코드 구조는 그대로 — 6.1 규칙).

## 3. 기억 표·방 항목·방향 구역(P2) — `map/include/mem_tok.h` 한 곳

- **기억 줄**(W1): 확정 물체 전부(N ≤ 256, 가구·문 포함, 사라진 것은 마지막 자리 + 상태). 줄 = VLA_INPUT 3절 33 숫자(같은 순서·같은 정의) + 이름 행 + 생김새(출처·섞임·K) +
  **방 종류 6**(그 물체가 든 방의 예측 확률 — 아래) + **살펴본 정도 3**(closest, n_views, top_seen; 윗면 없음 = −1 → 0 + 표시) + 힌트·표시. 줄 순서는 물체 번호 순(뜻 없음),
  `mem_n`·`mem_meta`(들고 있음·목표·힌트)·`mem_near`(경로 거리 — 거친 층 BFS, P6). 숫자는 소비 커널이 그 스텝 믿는 자세로 바로 계산(따로 저장 없음).
- **방 항목 ≤ 8**(W2): 드러난 방(본 넓이 ≥ 2 m²·30 %) 중 가까운 8: 방 종류 6(예측), 중심 x·y, 직선 거리, 경로 거리, 경로 있음, 탐사 비율, 지금 방, 문 수.
- **방향 구역 8**(W2): 로봇 기준 45° 칸마다 안 본 넓이(거친 층 0.4 m 칸, 집 전체 — P6), 가장 가까운 안 본 칸 경로 거리, 닿음, 문·열린 곳 보임, 덜 살펴본 가구 점수(칸 안 가구의 1 − top_seen 최대), sin·cos.
- **방 종류 = 예측 + 잡음**(점검 8절): 진짜 쪽은 `sm_room.type` 하나 + `name_conf`. 한 정의: 6 칸 = type 칸에 conf, "모름" 칸에 1 − conf(type 없음이면 모름 1).
  GPU 는 참 종류를 바로 쓰지 않고 방마다 리셋 때 예측(맞을 확률 p_rt, 틀리면 혼동 표, conf ~ 분포)을 뽑아 판 안에서 고정 — 방을 더 볼수록 conf 가 오르게(본 비율 따라). 값은 BASELINE 의 방 이름 정확도로 맞춤.
- 옛 16 칸·방 토큰 1·안 본 곳 광선 8·교사 전용 격자 16 × 16 × 2 는 이 표로 바꾸고 `archive/` 로(P7).

## 4. 입력 만들기 하나(P3·P4)

```
지도 상태(GPU 근사: MapCore + 물체 SoA + 격자 / 실제: scenemap 스냅숏)
   └─ 같은 장치·호스트 함수(inputs.h) ─→ 몸 56 + 목표 칸 2 × 16 + 벽 56 + 방 항목 ≤ 8 + 방향 구역 8 + 기억 줄 N ≤ 256 + 지시 벡터
          ├─ PPO 행위자: 위 전부(카메라·위에서 본 그림 RGB 없음). 기억 줄은 집합 인코더(지금 칸 MLP + 평균·최대 → 주목 풀링)
          ├─ PPO 비평자: 위 + 특권(참 목표 자세·참 벽 광선 16·참 자세 오차·덮은 넓이 — 비대칭)
          ├─ BC 학생 / RecallVLA VBatch: 위 + 카메라 3 장(위에서 본 그림 포함)
          └─ 실제 로봇(sm_tok.h): 스냅숏 → 같은 줄 함수(물체 줄의 숫자 계산 함수를 공유 — 입력이 sm_object 냐 GPU 물체냐만 다름)
```
- 한 함수: 줄 숫자 계산은 `mem_row(const ObjView&, const Pose&, …)` 하나(장치·호스트 공용, `ObjView` = 상자·상태·시각·이름 상위·K·살펴본 정도의 얇은 꼴).
  GPU 물체 SoA 와 `sm_object` + `sm_inspect` + objprob 이름 값을 각각 `ObjView` 로 바꾸는 작은 어댑터 둘만 다르다.
- **설정 해시**: 입력 설정(포맷 판·정규화 표 해시·use_map·잡음 `MP` 해시·지도 판·벡터 표 해시)을 `config/stage_*.json` 공유 단계 설정에서 읽고,
  `ppo_run`·`bc_run` 이 시작 때 교사 체크포인트의 해시와 비교 — 다르면 멈춤(TRAINING_DESIGN 4.1). run.json 에 기록.
- **교사 특권 빼기**(점검 5): 행위자 X0 에서 G1 관측 56–71(참 360° 벽 광선 16)과 확정 목표의 참 자세(G1 53–55·72–79)를 뺌 → 지도 벽 56·목표 칸(지도 자리)만.
  비평자에만 둠. 대본 교사는 믿음 지도(믿는 점유 비트 + 부풀림)로 계획(`teacher.h tch_occ_full` → 지도 비트), `target_known` 을 B4·B5 에도.
- 예전 X0(976)·칸 줄(16 × 304) 배치는 새 판(번호 올림)으로 바뀐다. 옛 체크포인트는 안 씀(사용자 결정).

## 5. 벡터 공간 하나(P5) — 점검 1·2·10·11, 1절 머리 `h`

- **공간 = SigLIP 2 B/32**(진짜 파이프라인이 쓰는 것): 이름 = SigLIP 2 글 탑(768, `sgc_text_encode` — 어떤 이름이든 그때 계산하고 캐시), 생김새 = 물체 μ(SigLIP 2 영상 768,
  objprob 이 이미 저장 `O<id>_emb.f16`), 지시(영·한) = SigLIP 2 글 탑(다국어 — 한국어 학생 인코더 불필요).
- **작게**: 고정 투영 768 → 128(PCA, SigLIP 2 글 + 영상 표본으로 한 번 계산, `training/data/siglip2_proj/pca128.f16` 레포에 저장, 두 쪽 같은 파일). 학습된 머리 `h` 는 버림
  (`training/embed` 의 PE-L·P·`h`·한국어 학생 길은 `archive/`). RecallVLA φ 입력 폭(304) 그대로.
- **이름 확신도 2 의 한 정의**: conf1 = objprob 이름 사후 확률(고른 이름), conf2 = 1위 − 2위 사후. GPU 는 R9 사후에서, 실제는 스냅숏에서 —
  scenemap 에 물체별 이름 사후 상위 둘을 꺼내는 C ABI 가 필요(behavior-2026 별도 클론에서 `sm_snap_objprob`(id, name_p, second_p, K, n_whole) 더함 → 포인터 올림 → `sync_scene_graph.sh`).
- **생김새의 한 정의**: 투영(μ). 실제 = `O<id>_emb.f16` 의 μ, GPU = R7 모형의 μ(같은 투영). 출처 원형 a(src) = 그 종류의 SigLIP 2 영상 원형:
  (1) og_replay 진짜 파이프라인이 낸 μ 를 정답 종류로 묶은 평균(BEHAVIOR 물체, OmniGibson 렌더 — 다른 에이전트 도구 결과를 읽기만), (2) 없으면 글 벡터 + 영상–글 평균 차(모달리티 차) **(가정)**.
- **실제 런타임도 같이 바꿈**(코디네이터 10-06): 지금 실제 이름도 PE-L 길이다(`src/scene_graph/runtime/src/sgrt_clip.cpp`·`sgrt.cpp` 가 `embed_work/labels/objects-v1`(P 공간 라벨) + 머리 `h` 를 읽음).
  P5 는 두 쪽을 함께: 라벨 표 = SigLIP 2 글 탑 + 레포의 고정 PCA(자료는 `config/paths.env` 변수 아래), 실제 이름 = SigLIP 2 영상 대 SigLIP 2 글, 학습 표(`vla_vocab.h`·`vec_tab.h`·`net.h`·BC `app_head.h`).
  순서: ① 바꾼 뒤 실제 이름 품질 확인(OpenLORIS office1-5 `realbag_run` + `objprob_eval`: 찾음·중복·이름 정확도를 지금 값과) ② 그 뒤에만 PE-L 길을 `archive/`(training/embed 머리·P·한국어 학생 스크립트, 런타임 머리 `h` 코드). 가중치·작업 파일은 data/embed_work 에 둠(지우지 않음).
- **objprob 계산 하나**(사용자 결정 10-06): scenemap `objprob.cpp` 와 `objprob_gpu.h` 가 같은 헤더 `src/scene_graph/scenemap/include/scenemap/objprob_math.h`(`__host__ __device__`, STL 없음)의
  같은 것 로지스틱·이름 사후·상위어·κ·칼만·이름 분포 겹침을 부른다(자료 배치·묶음만 다름). 확인: realbag_run OpenLORIS office1-5 출력 바이트 같음(바꾸기 전후), 합성 검출 열로 CPU 대 GPU 맞춤 시험.
- **이름 표**: `kVocab`·`kApLabels`(진짜 64 라벨) + BEHAVIOR 과제 synset 이름 → 표 행 하나로(빠진 24 개 포함). 레포 밖 `~/embed_work/labels/objects-v1` 의존은 표를 레포(`training/data/names_v2`)로 옮기고 만드는 스크립트를 레포 안에서 돌게.

## 6. 고정 창 → 집 전체(P6) — 점검 6

- 판 좌표 = **집(장면) 좌표**. 판마다 창 가운데 `wx·wy` 는 "판 원점" 일 뿐이고 환경의 창 안 묶기(`env_beh.h:144` 창 여유 밖 = 끝)를 없앤다(충돌·광선은 이미 장면 전체 묶음).
- 지도 격자 두 층: (a) **고운 창** 128 × 128 · 0.10 m(지금 커널 그대로)을 **로봇을 따라 옮김** — 로봇이 창 가장자리 3.2 m 안에 오면 32 칸 단위로 다시 놓고(내용 옮김, 새 칸은 모름),
  벽·경유 지점·위에서 본 그림·거리장은 이 창으로(모두 로봇 둘레라 결과 같음). (b) **집 전체 거친 층** 0.4 m 칸: 본 칸·점유 비트(장면 크기 상한 `MAXW_C × MAXH_C` = 256 × 128 (가정: 86 m × 40 m 집까지), 판당 8 KB) —
  고운 창에서 빠지는 칸도 남는다. 방향 구역·방 탐사 비율·기억 줄 경로 거리·프런티어는 이 층.
- 물체 SoA 는 집 좌표 float(경계 없음). 특징에는 지도 크기·좌표·경계가 없다(로봇 기준만).
- 대본 교사·거리장(`teacher.h`·`teacher_sl.h` 의 WIN 크기 BFS)은 고운 창 기준 + 거친 층 프런티어로.

## 7. 옛 것 지우기(P7)

| 옛 것 | 자리 | 처리 |
|---|---|---|
| objmap 이름 규칙(`name_share`·`name_switch`·`name_key`·`name_merge_iou`, `vote*`, `name_ok`, `obj_keys` 탐욕) | `map.h` | objprob 길로 바꾸고 옛 함수는 `archive/training/RL/map/include/map_objmap_rules.h` 로 |
| 16 칸 가까운 순 `MapTok::slot`·`name_id`·`app_id`, 방 토큰 10, 완성도, 안 본 곳 광선 `front`, 교사 격자 `tv` | `map_tok.h`, `obs.h`, `net.h`, BC `tf.h`, `trainer.cu` | 새 입력으로 바꾸고 옛 배치는 archive |
| 처음 지도 C0/C1(`MapCurr p0/p1`, `curr_slots`·`curr_grid`) | `map.h` | 디버그 깃발 뒤에만 남기지 않고 지움(설정은 이미 archive) — `--debug-gt-map` 길도 함께 뺌 **(결정 필요 없음: 10-06 결정)** |
| 고정 창 | `bscene.h WIN`, `map.h GW` 의미 | 6절 |
| 잡음 상수(COCO-80·R1) | `map.h MP` | 2절 값으로, 옛 값은 `map_calib/README.md` 기록 |
| PE-L·P·`h`·한국어 학생·`vla_v1` 표 | `training/embed`, `training/data/vla_v1` | 5절, archive |
| PPO 설정 C0/C1, 기록 도구 기본 80 % GT | `RL/config/*.json`, `record_ppo.cu`·`record_bc.cu` | C2 단계 설정, 기록 도구 기본 0 + 같은 가드 |
| PPO 집 나누기 없음 | `ppo/driver/src/main.rs bcurr_of` | 학습 집 4 기본, 평가 집 이름 한 곳(공유 파일) |

## 8. 속도 계획(+10 % 안)

- 기준: `map_bench`/`env_bench` + nsys(같은 날 같은 GPU, 다른 GPU 일 없을 때 — 지금은 OmniGibson 비교 실행 중이라 그 뒤), N 4,096·16,384, BEHAVIOR B4 판.
- 더는 비용(추정)과 줄이는 법:
  1. 가구 검출은 이미 쏘는 깊이 광선의 맞은 상자 번호로(광선 더 없음, 열마다 int16 하나 더 씀).
  2. 같은 것 판정은 관측 ≤ 24 × 시야 근처 물체 ≤ 48 만(상자 틈 0.30 m 거름 먼저). 로지스틱은 곱셈 8 번.
  3. 이름 사후는 관측이 붙은 물체만 keyframe 끝에(상위 6 + 나머지, 라벨 64 의 log-sum-exp 없음).
  4. 살펴본 정도 top_seen 은 윗면 있는 물체 중 2 m 안·덜 본 것만, 16 점 광선(부재 확인 27 점과 같은 꼴) — 이번 keyframe 에 맞은 물체만.
  5. 기억 줄은 따로 패스로 만들지 않고 소비 커널(PPO·BC 입력 모으기) 안에서 줄마다 계산(저장·읽기 한 번).
  6. MapCore 에서 물체 칸 16(2.7 KB)이 빠져 시작 커널·토큰 커널이 가벼워짐.
- keyframe 커널 공유 메모리: 물체 칸 16(2.7 KB) → 시야 근처 물체 48 개 목록(약 8 KB) — SM 당 블록 수가 줄 수 있음(지금 10). 넘으면 근처 물체를 두 묶음으로.

## 9. 검증 계획

| # | 검증 | 도구 | 통과 기준 |
|---|---|---|---|
| V1 | GPU == CPU 비트 동일(새 커널 전부: 물체 SoA·이름 사후·살펴본 정도·기억 줄·방 항목·방향 구역) | `map_verify`(새 비교 항목), `env_verify` | 0 다름, 음성 대조(규칙 하나씩 GPU 만 끔: 로지스틱 무게·병합·칼만·top_seen·n_views·기억 줄 순서) 모두 실패 |
| V2 | CPU 참조 대 진짜 scenemap objprob 규칙(잡음 끔, 같은 합성 입력) | 새 `map/tools/objprob_parity.cpp`(서브모듈 소스를 읽기만 해 빌드 — `map_realcheck` 꼴): 같은 관측 열(상자·κ·이름 우도)을 진짜 `apLogit`·`apAddView`·`apName`·칼만·`inspObserve`/`inspTop` 과 우리 함수에 | 같은 짝·같은 이름·자리 차 < 1e-4 m, 로짓 차 < 1e-5 |
| V3 | 입력 단위 비교 대 진짜 파이프라인 | 다른 에이전트의 `og_cmp`·`map_cmp`(읽기·실행만) | BASELINE 표의 각 값이 문턱 안(인수인계 표를 따름) |
| V4 | 교사 성공 C2 B4/B5 | `pnp_teach --sl --slknown --curr 0,0` | 지금 값과 비교(B4 0.011 이 막힘 — 새 지도에서 다시 잼) |
| V5 | 속도 | nsys 표(전·후) | 환경 + 지도 스텝 +10 % 안 |
| V6 | 실제 쪽 짝 | `sm_tok_test` 확장(기억 줄·방 항목·방향 구역, 강체 불변, GPU 어댑터와 같은 값) | 0 실패 |
| V7 | 설정 해시 | `ppo_run`/`bc_run` 다른 설정으로 시작 → 멈춤 | 멈춤 |

## 10. 열린 질문·충돌

1. **집 전체(P6)는 환경·교사 코드까지 바뀐다**(`env_beh.h` 창 묶기, `teacher*.h` 창 크기 BFS, `Entry::rb` 창 비트). 프런티어·RL 교사 일이 바로 뒤에 시작하므로 P6 을 그 일과 같이(또는 앞서) 해야 한다. 이 문서는 P1–P5 를 먼저 하고 P6 은 바로 이어서 한다.
2. ~~PPO 행위자의 기억 줄 수~~ → 결정(0.2): 물체 스냅숏 + 같은 함수로 다시 만들기, 비트 검사.
3. ~~생김새 원형~~ → 결정(0.2): og_cmp μ + 가정 표시.
