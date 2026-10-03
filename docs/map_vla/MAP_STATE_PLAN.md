# 지도를 학습 state에 수치로 넣기 — 계획

구상: ① `dynamic-object-mapping-benchmark`로 GT 지도를 만들고, ② 우리 지도 업데이터를 만들고, ③ 지도 자체를 학습 state에 수치로 넣는다.
이 문서는 각 단계가 가능한지와 순서를 정리한다. 코드는 아직 없다.

## 1. 벤치마크가 주는 것 / 안 주는 것 (`src/dynamic-object-mapping-benchmark`, 직접 읽음)
| 주는 것 | 안 주는 것 |
|---|---|
| Office 장면에서 카메라가 같은 경로를 두 바퀴 돈다. 바퀴 사이에 물체가 옮겨지고(moved), 사라지고(removed), 추가되고(added), 바뀐다(swapped) | **로봇도 행동(action)도 없다.** 카메라 자세만 있다 |
| 프레임마다 RGB(8bit), 깊이(uint16 mm, z-depth), 인스턴스 마스크, 카메라 자세(TUM, `world_T_camera`), 내부 파라미터. 10 Hz, 720×480, 시퀀스 3개(static, dynamic1, dynamic2) × 2838프레임 | 우리 로봇(LIMO + OMX)의 영상 |
| GT: `objects.csv`(클래스, 이동 가능 여부, 크기), `object_poses.csv`(시간별 물체 중심 위치), `changes.csv`(변화 종류와 시각) | 파이프라인 학습용 데모 |
| 채점: 예측 `map_timeline.csv`(`frame,obj_id,x,y,z[,moving]`)로 정적/변화/동적 P·R·F1 | |
- 공개된 다른 방법(ConceptGraphs, DualMap)은 moved/removed가 0점이다. 같은 `obj_id`를 이동 후에도 유지하는지를 재는 벤치마크다.
- 데이터는 이 PC에 **없다**(씬 매니페스트만 있음). 받을 수는 있다: 팀 비공개 릴리스 `dataset-office-seed1-v1`, raw 3개 약 1.3 GB씩(합 3.9 GB). `gh`로 접근 확인함. 다시 렌더하려면 Isaac Sim **4.5.0**이 필요하다(이 PC `behavior` env는 5.1, 호환은 확인 안 함).

## 2. 단계별 판단
**① GT 지도** — 가능. `objects.csv`+`object_poses.csv`로 프레임별 "정답 지도"가 나온다. 단, 전체 GT는 안 본 물체까지 다 담는다. 학습 state의 GT는 "그 프레임까지 인스턴스 마스크에서 충분히 보인 물체만"으로 걸러야 업데이터가 실제로 낼 수 있는 값과 맞는다(벤치마크도 `>= 400 px in >= 10 frames`를 "well seen"으로 쓴다).

**② 우리 지도 업데이터** — 팀 `scenemap`(`~/behavior-2026/src/scene_graph/scenemap`, CPU C++)이 이미 물체 상태 SEEN/GONE/MOVED/HELD와 `map_timeline.csv` 출력 도구(`tools/map_timeline.cpp`)를 갖고 있다. 다만 그 도구는 BEHAVIOR 에피소드(`ep_*.bin`, R1 Pro proprio)만 읽는다. **벤치마크 시퀀스를 읽는 어댑터가 필요하다**: RGB/깊이/자세 → `sm_push_image`(`depth_m`, `fx fy cx cy`)와 `sm_push_pose`(GT 자세 모드). 검출은 처음엔 인스턴스 마스크(= '완벽한 검출')로 넣어 업데이터 로직만 따로 채점하고, 그다음 FastSAM + SigLIP2로 바꾼다.

**③ 지도를 state 수치로** — 가능. 아래 §3. 단, 이 단계는 벤치마크 밖의 데이터가 필요하다(§4).

## 3. state 수치 형식 (제안)
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
