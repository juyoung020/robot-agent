# 인식 스택 에뮬레이션 조사 (Perception Error Model / Surrogate)

조사 방식: 웹 검색으로 존재와 초록을 확인함. 본문 세부(아키텍처, 수치)는 초록 수준만 확인했고, 그 이상은 "확인 필요"로 표시함. Waymo/Zoox/Cruise/Aurora/Wayve/NVIDIA DRIVE Sim의 내부 방식은 공개 문헌에서 검증하지 못해 서술하지 않음.

## 요약

자율주행 쪽에서는 이 문제를 Perception Error Model(PEM, 일명 surrogate)이라 부른다. 센서와 detector를 시뮬레이션하지 않고 GT 객체 목록에 "실제 detector가 낼 법한 오류"(미검출, 위치/크기/heading 오차, 오검출, ID switch)를 직접 주입한다. 모델은 통계형(조건부 분포, ARMA, HMM)에서 학습형(NN surrogate, 트랜스포머 생성모델 Emperror)으로 발전했다. 핵심 교훈은 세 가지다. (1) 오류는 프레임 간 상관이 강해서 i.i.d. 노이즈로는 부족하며 거리/가림/클래스 조건과 시간 구조(Markov/AR)가 필요하다. (2) 학습형 surrogate는 downstream planner 성능을 잘 보존하지만 픽셀 특유의 실패는 못 잡는다. (3) 실무적으로는 "privileged teacher -> 노이즈/실제 인식 입력 student" 증류(Learning by Cheating, Miki 등)와 "통계적 PEM으로 대량 학습 + 소량 실제 파이프라인으로 마무리"가 표준 패턴이다. 우리 설정에는 (c) 학습형 PEM을 (b) 캐시로 보정/검증하고, (d)(e)로 마무리하는 혼합안을 권장한다. "어떤 방식이 sim2real에서 최선인가"를 직접 비교한 결정적 연구는 찾지 못했다(확인 필요).

## 접근법 비교표

| 접근 | 무엇 | 사용처(검증된 것) | 비용 | 충실도 | 장단점 | 출처 |
|---|---|---|---|---|---|---|
| (a) 통계 PEM | GT에 조건부 확률 오류 주입 (검출확률, 위치 오차 분포, FP) | Piazzoni 2020(가림 기반 미검출), PEM 2023 | 매우 낮음 | 낮음~중간 | 해석 쉬움, 커버리지 조절 가능 / 실제 분포와 어긋남, 시간 상관 약함 | [1][2] |
| (a') 시계열 통계 PEM | ARMA/NAR, HMM으로 오류 지속성 모델링 | Mitra 2018, HMM 기반 PEM(요약 수준) | 낮음 | 중간 | 지속 오류 재현 / 상태 공간 설계 필요 | [3][6] |
| (b) 시점별 캐시 | 포즈마다 실제 인식 출력 사전계산 후 조회 | 자율주행 직접 선례 못 찾음(확인 필요). NeRF/log-replay류 유사 | 사전계산 큼, 조회 0 | 해당 시점은 최고 | 실제 오류 그대로 / 정책이 캐시 밖 포즈/행동 시 불가, 저장 폭증, 상호작용 불가 | - |
| (c) 학습형 surrogate | NN이 detector 출력 분포 학습 | CARLA PIXOR/CenterPoint surrogate (NeurIPS-W 2021), Emperror(트랜스포머 생성) | 학습 1회, 추론 매우 낮음 | 중간~높음 | downstream 거동 보존, 계산 절감 / 학습 분포 밖 취약, 픽셀 특화 실패 누락 | [4][5] |
| (d) 실제 인식 in loop (student만) | student 학습/DAgger에서 실제 파이프라인 사용 | Learning by Cheating(vision student) | 높음 | 최고 | 오류 완전 반영 / 느림(~30 samples/s) | [7] |
| (e) 최종 real fine-tune | 마지막에 실제 출력으로 미세조정 | RMA 등 2단계 구조 일반 패턴 | 중간 | 높음 | 잔여 gap 제거 / 데이터 필요 | [8] |
| 노이즈 주입 teacher-student | privileged teacher, student는 노이즈 관측 | Miki 2022(높이맵 노이즈), RMA | 낮음 | 노이즈 모델 의존 | RL은 빠르게 / 노이즈 모델 품질이 상한 | [8][9] |

## 조사 항목별 정리

### 1. AD의 PEM
- 개념: 센서 합성 없이 GT에 오류를 주입. Piazzoni et al., IJCAI 2020은 가림으로 인한 미검출을 모델링하고 시뮬레이션에서 in-the-loop 주입함 [1]. 이후 PEM(2023) [2], CoPEM(협력 인식) [10]이 있음.
- 입출력(일반): 입력은 GT 객체 상태 + 거리/가림/클래스/환경, 출력은 검출 여부, 위치/크기/heading 오차, FP, 트랙 ID. latency는 논문별로 다룸 여부가 달라 확인 필요.
- Mitra et al., ITSC 2018은 카메라 detection 오류를 ARMA/NAR로 모델링, CarMaker에서 평가 [3].
- 학습형: CARLA에서 LiDAR detector(PIXOR, CenterPoint) surrogate로 계산시간 절감과 유사한 downstream 거동 [4]. Emperror는 트랜스포머 기반 생성형 PEM이며 detector를 더 충실히 모사하고 IL planner 충돌률을 최대 85% 높이는 입력을 생성함(초록 기준) [5].
- 상업 팀(Waymo 등)의 neural surrogate: 검증 못 함. 확인 필요.

### 2. 중간 표현(object list / BEV)
- 정책이 객체 목록이면 sim은 목록만 에뮬레이션하면 됨. Learning by Cheating은 privileged agent가 GT 레이아웃과 모든 참가자 위치를 보고 학습, 이후 vision student를 증류 [7]. Learning from All Vehicles [11], World on Rails [12]는 중간 표현/로그 기반 감독 사례. 이들은 detector 오류를 모사하기보다 vision student로 풀었다는 점에서 우리 (d)에 가까움.

### 3. 학습형 surrogate 설계
- 구조: GBDT/MLP/혼합밀도/트랜스포머 생성(Emperror). 조건: 거리, 가림, 크기, 클래스, 조명. 학습: GT와 detector 출력 매칭(Hungarian/IoU) 후 매칭 여부와 잔차 학습. 세부 아키텍처와 학습 쌍 구성은 논문 본문 미확인, 확인 필요.
- 검증: 분포 지표(잔차 히스토그램, 검출률 곡선) + downstream 지표(충돌률, planner 성능). Piazzoni 계열은 downstream 영향을 핵심 평가로 둠 [1][2].
- 실패 모드: 픽셀 특유 실패(반사, 특정 질감), 분포 밖 상황, 시간 상관 누락.

### 4. 시간 상관
- Markov/HMM 상태(검출됨/놓침)를 위치/가림 구간별로 분할하는 PEM, 오류 t-1 -> t 회귀가 보고됨 [3][6]. 클래스/환경별 전이확률 차이가 큼(검색 요약 수준, 확인 필요). Adaptive sampling으로 희귀 안전 위반을 PEM 상류에서 샘플링하는 연구도 있음 [13].

### 5. SLAM/위치 오류
- AD에서는 localization 오류 모델이 따로 있으나 이번 조사에서 대표 논문을 검증하지 못함. 확인 필요.
- 로봇: Habitat realistic PointNav는 LoCoBot 벤치마킹 기반 actuation noise와 RGB 가우시안, depth Redwood 노이즈를 도입 [14]. 이후 연구는 GPS+Compass 없이 visual odometry로 대체 [15]. Sim-vs-real 상관(SRCC)이 낮을 수 있음(초기 설정에서 0.18)이 보고됨 [14].

### 6. 로봇/체화 AI
- Habitat 노이즈 모델 [14], "Rethinking Sim2Real: 낮은 충실도의 시뮬이 더 나은 전이"(제목만 확인, 주장 세부는 확인 필요) [16].
- Teacher-student: Miki 2022는 student 입력에 높이샘플 노이즈 모델을 얹고 재귀 인코더로 신뢰 상태를 구성 [9]. RMA [8], DAgger [17].

### 7. 캐시/룩업
- 직접적 선례는 찾지 못함(확인 필요). 일반적으로 상호작용(정책 행동 -> 새 시점) 시 커버리지 문제로 한정적. 우리 경우 맵 토큰이 키프레임 누적이라 시점 이산화는 가능하나 행동 공간이 연속이면 격자 보간 필요.

### 8. 최선 접근 증거
- 방식 간 직접 비교 증거는 못 찾음. 실무적 패턴(문헌 종합, 추정): 통계/학습 PEM으로 대량 + 실제 인식 소량 마무리. "확인 필요".

## 우리에게 맞는 안

원칙: RL teacher는 privileged + 경량 노이즈, student/VLA에는 학습형 PEM으로 map-token을 생성, 마지막은 실제 파이프라인으로 검증과 소량 fine-tune.

**무엇을 에뮬레이션할지** (object-level map 출력 기준, 이미지가 아니라 맵 상태 수준)
- 객체 검출 여부: 시점/거리/가림/객체 크기/클래스 조건부 검출확률(키프레임 단위).
- 위치/크기 오차: Kalman 융합 이후의 공분산 수준 포함.
- 이름/임베딩: 정답 라벨 -> SigLIP 2 top-k 혼동 분포(클래스 혼동 행렬 + 신뢰도), 임베딩은 실제 임베딩 코드북에서 혼동 클래스 근처 샘플링(vMF 노이즈) 형태 제안.
- merge/split/ghost: association 실패(두 객체 병합, 한 객체 분할), 유령 객체. 이것이 맵 토큰에 직접 영향하므로 우선순위 높음.
- SLAM drift: 휠 오도메트리 + scan matching 잔차를 pose 랜덤워크 + 간헐적 점프(loop closure 유사)로.
- 시간 상관: 객체별 숨은 상태(검출/놓침/오분류 지속), 에피소드별 전역 편향(조명, 장면 재질).

**모델**
- 1단계: 통계 PEM(조건부 검출확률 + 혼동행렬 + 가우시안 오차 + 객체별 2상태 Markov). 빠른 베이스라인.
- 2단계: 학습형(작은 MLP/GBDT 혼합밀도 + 객체별 GRU 상태, 또는 맵 상태 -> 맵 상태 생성모델). 맵 수준 변환이므로 Emperror식 객체 집합 생성이 참고가 되나 우리 맵 상태에 맞는지 확인 필요.
- 캐시(b)는 모델 학습/검증용 데이터 소스로 활용, 정책 학습의 주 경로로는 비권장.

**학습 데이터 (OmniGibson + 실제 파이프라인)**
- OmniGibson에서 다양한 궤적(정책 롤아웃 + 랜덤 탐색)을 렌더링하여 실제 ObjectSAM + SigLIP 2 + 맵 융합을 돌리고, 같은 순간의 GT 객체 상태와 쌍으로 저장(GT 객체 <-> 맵 객체 매칭 포함).
- 가림/거리/조명/장면 다양성 확보. 실제 로봇/실제 집 데이터가 있으면 sim 렌더링과의 도메인 갭을 따로 측정(렌더링 인식 오류와 실제 인식 오류는 다를 수 있음).

**검증**
- 입력 수준: 검출률 vs 거리/가림, 혼동행렬, 위치 오차 분포, ID/merge/ghost 빈도, 오류 지속 길이 분포, 맵 토큰 분포(임베딩 거리) 비교. 학습에 쓰지 않은 장면으로 hold-out.
- Downstream: 같은 정책을 (PEM 입력, 실제 파이프라인 입력)에서 평가해 성공률/실패 유형 상관 확인. Habitat의 sim-real 상관 지표(SRCC) 방식 [14]을 정책 순위 일치도로 차용.

**실제 파이프라인으로 돌아갈 때**
- PEM vs 실제 평가 성공률 차이가 임계치(예: 10%p 이상, 임계값은 설정 필요)일 때, PEM이 못 보는 실패 유형이 보일 때, 새 장면/조명/객체 분포일 때, 최종 배포 전 (e) fine-tune. (d)는 student DAgger의 소량 라운드에만 사용.

## 열린 질문
- 렌더링된 OmniGibson 이미지에서의 인식 오류가 실제 카메라의 오류를 얼마나 대변하는가(sim 인식 오류 vs 실제 오류).
- 맵 수준(융합 후) 에뮬레이션 vs 프레임 수준 detection 에뮬레이션 후 실제 융합 코드 재사용. 후자가 충실도는 높고 비용은 CPU 융합이라 GPU 배치와 맞는지 확인 필요.
- 학습형 PEM의 분포 밖 안전성: 정책이 PEM의 약점을 exploit할 수 있음(RL이 노이즈 모델 허점을 악용). 대응으로 앙상블/랜덤화 필요.
- SLAM 오류 모델 대표 문헌, 상업 AD 팀의 surrogate 공개 자료, 캐시 방식의 선례는 이번에 확인 못함.

## 참고문헌
[1] Piazzoni et al., Modeling Perception Errors towards Robust Decision Making in AVs, IJCAI 2020 https://arxiv.org/abs/2001.11695
[2] Piazzoni et al., PEM: Perception Error Model for Virtual Testing of AVs https://arxiv.org/pdf/2302.11919
[3] Mitra et al., Towards Modeling of Perception Errors in AVs, ITSC 2018 https://dl.acm.org/doi/10.1109/ITSC.2018.8570015
[4] A Step Towards Efficient Evaluation of Complex Perception Tasks in Simulation (NeurIPS 2021 workshop) https://arxiv.org/abs/2110.02739
[5] Emperror: A Flexible Generative Perception Error Model for Probing Self-Driving Planners https://arxiv.org/abs/2411.07719
[6] Quantifying Error Propagation and Recovery in Object Detection for AVs: A Markovian Approach (Springer, 초록만 확인) https://link.springer.com/chapter/10.1007/978-3-032-18474-0_39
[7] Chen et al., Learning by Cheating, CoRL 2019 https://arxiv.org/abs/1912.12294
[8] Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, https://arxiv.org/abs/2107.04034 (검색으로 직접 확인하지 않음, 기억 기반, 확인 필요)
[9] Miki et al., Learning robust perceptive locomotion for quadrupedal robots in the wild, Science Robotics 2022 https://arxiv.org/abs/2201.08117
[10] CoPEM: Cooperative Perception Error Models for AD https://arxiv.org/pdf/2211.11175
[11] Learning from All Vehicles https://arxiv.org/abs/2203.11934
[12] Learning to drive from a world on rails https://arxiv.org/abs/2105.00636
[13] Testing Rare Downstream Safety Violations via Upstream Adaptive Sampling of PEMs https://arxiv.org/abs/2209.09674
[14] Kadian et al., Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance? https://www.researchgate.net/publication/343446764_Sim2Real_Predictivity_Does_Evaluation_in_Simulation_Predict_Real-World_Performance
[15] The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation https://arxiv.org/abs/2108.11550
[16] Rethinking Sim2Real: Lower Fidelity Simulation Leads to Higher Sim2Real Transfer in Navigation https://arxiv.org/abs/2207.10821
[17] Ross et al., DAgger https://arxiv.org/abs/1011.0686 (기억 기반, 확인 필요)
