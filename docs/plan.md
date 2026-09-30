# 계획 (plan)

동적 3D scene graph + AI agent + VLA(π0.5) 결합.
체크박스는 끝나면 `[x]` 로 바꾸고, 담당자를 `(@이름)` 으로 적는다.

참고 논문·코드는 [`refs/README.md`](../refs/README.md) 참고.

## 1. Scene graph (로봇의 기억) — `src/scene_graph/`

- [ ] **SLAM(위치 추정 + 지도 작성)**: 로봇 위치(pose)와 3D 지도 만들기
  - [ ] 사용할 SLAM 고르기 (RGB-D 카메라 기준)
  - [ ] ROS 2 로 pose·지도 토픽이 나오는지 확인
- [ ] **SAM(물체 분할)**: 카메라 영상에서 물체 영역(mask) 잘라내기
- [ ] **CLIP(이미지-글 임베딩)**: 잘라낸 물체마다 의미 특징(feature) 뽑기 → 글로 검색 가능하게 (open-vocabulary)
- [ ] **DA(data association, 데이터 연관)**: 새로 본 물체가 그래프에 이미 있는 물체인지, 새 물체인지 판단
- [ ] **맵 업데이트**: 물체가 옮겨지거나 사라지면 그래프의 해당 부분만 고치기 (참고: DovSG 국소 갱신)
- [ ] scene graph 를 agent 가 읽을 수 있는 형태(JSON 등)로 내보내기

## 2. 로봇

- [ ] **캘리브레이션(calibration, 보정)**
  - [ ] 카메라 내부 파라미터(intrinsic)
  - [ ] 카메라-로봇 사이 위치 관계(hand-eye / extrinsic)
- [ ] **조작(manipulation)**: 팔·그리퍼를 ROS 2 로 움직이기, 기본 동작(집기·놓기) 확인
- [ ] **pose 토픽 따두기**: 로봇·팔 끝(end-effector)·카메라 pose 토픽 이름, 메시지 타입, 주기 정리
- [ ] **ros2 bag 녹화**: 카메라(RGB·depth)·pose·관절 상태를 녹화해서 오프라인으로 scene graph·VLA 실험
  - [ ] 녹화할 토픽 목록 정하기
  - [ ] 녹화 스크립트 → `scripts/`
- [ ] 로봇·센서 사양, 토픽 목록 문서 → `docs/`

## 3. VLA (π0.5) — `src/vla/`

- [ ] **π0.5 돌려보기**: openpi 로 기본 추론(inference) 확인
- [ ] **학습용 시뮬레이션 구하기**
  - [ ] 후보 비교: LIBERO, BEHAVIOR-1K 등 (참고: 2025 BEHAVIOR Challenge 상위 팀)
  - [ ] 우리 로봇·작업과 비슷한 환경 고르기
- [ ] **파인튜닝(fine-tuning, 추가 학습)**: 우리 작업 데이터(시뮬 + ros2 bag)로 π0.5 학습
- [ ] **양자화(quantization, 모델 경량화)**: 로봇 컴퓨터에서 돌아갈 만큼 메모리·속도 줄이기
- [ ] agent 가 주는 단계 지시(subtask) 형식 맞추기

## 4. Agent (계획·실패 복구) — `src/agent/`

- [ ] scene graph 를 읽고 작업을 단계로 나누기
- [ ] 단계마다 VLA 호출 → 성공 여부 확인 → 실패하면 재계획 (참고: BEHAVIOR 3위 SimpleAI 논문)

## 5. 통합

- [ ] scene graph → agent → VLA 전체 흐름 시뮬레이션에서 확인
- [ ] 실제 로봇에서 확인
