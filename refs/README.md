# refs

참고 논문·코드·자료 모음. 레포가 **공개(public)** 이므로 논문 PDF는 깃에 올리지 않는다 (저작권).
논문 PDF(`refs/papers/`)와 참고 코드(`refs/code/`, 원본 레포 클론)는 `.gitignore` 로 제외되어 있으니, 각자 아래 명령으로 받는다:

```bash
bash refs/download.sh
```

## VLA (행동 모델)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| π0.5: a Vision-Language-Action Model with Open-World Generalization (CoRL 2025) | [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 우리가 쓸 VLA. 단계(subtask) 지시 + 카메라 → 행동 |
| π0: A Vision-Language-Action Flow Model for General Robot Control | [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5의 전신 |

## Scene Graph (로봇의 기억)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| DovSG: Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation | [arXiv 2410.11989](https://arxiv.org/abs/2410.11989) | [BJHYZJ/DovSG](https://github.com/BJHYZJ/DovSG) | **우리 아이디어와 가장 가까움.** 그래프 국소 갱신 + LLM 작업 계획 |
| Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization (RSS 2022) | [arXiv 2201.13360](https://arxiv.org/abs/2201.13360) | [MIT-SPARK/Hydra](https://github.com/MIT-SPARK/Hydra) | 실시간 계층형 scene graph (물체·장소·방·건물) |
| Khronos: Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments (RSS 2024) | [arXiv 2402.13817](https://arxiv.org/abs/2402.13817) | [MIT-SPARK/Khronos](https://github.com/MIT-SPARK/Khronos) | Hydra 후속. 시간에 따른 변화 감지 |
| ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning | [arXiv 2309.16650](https://arxiv.org/abs/2309.16650) | [concept-graphs/concept-graphs](https://github.com/concept-graphs/concept-graphs) | 2D 기반 모델을 3D로 융합, 열린 어휘(open-vocabulary) 물체 노드 |
| HOV-SG: Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation (RSS 2024) | [arXiv 2403.17846](https://arxiv.org/abs/2403.17846) | [hovsg/HOV-SG](https://github.com/hovsg/HOV-SG) | 층·방·물체 계층 + 열린 어휘 |
| HAMMER: Heterogeneous, Multi-Robot Semantic Gaussian Splatting | [arXiv 2501.14147](https://arxiv.org/abs/2501.14147) | 공개 안 됨 ([프로젝트 페이지](https://hammer-project.github.io)) | 여러 로봇의 SLAM 좌표를 하나로 맞추고, CLIP 의미 특징을 넣은 3DGS 지도를 실시간 학습 (ROS 기반) |
| Spark-DSG: 3D scene graph 자료구조 라이브러리 (C++·Python) | 논문 없음 | [MIT-SPARK/Spark-DSG](https://github.com/MIT-SPARK/Spark-DSG) | Hydra·Khronos 가 쓰는 scene graph 저장·읽기 API. `pip install` 로 파이썬에서 바로 사용 가능 |

## Agent (Scene graph + LLM 계획)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| SayPlan: Grounding LLMs using 3D Scene Graphs for Scalable Robot Task Planning (CoRL 2023) | [arXiv 2307.06135](https://arxiv.org/abs/2307.06135) | | 그래프에서 필요한 부분만 LLM이 탐색 + 반복 재계획 |
| MoMa-LLM: Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation | [arXiv 2403.08605](https://arxiv.org/abs/2403.08605) | [robot-learning-freiburg/MoMa-LLM](https://github.com/robot-learning-freiburg/MoMa-LLM) | 탐색하며 갱신되는 scene graph 위에서 LLM이 행동 선택 |

## 2025 BEHAVIOR Challenge 상위 팀

BEHAVIOR-1K 가정용 장기 작업 대회. 1·2·3위 모두 π0.5 기반.

참고 대회 공식 페이지: [behavior.stanford.edu/challenge](https://behavior.stanford.edu/challenge/index.html) — 규칙·과제·데이터·평가 방법·리더보드.

| 순위 | 팀 | 점수 | 코드 | 보고서 | 메모 |
|---|---|---|---|---|---|
| 1 | Robot Learning Collective (개인 팀) | 0.260 | [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution) | [arXiv 2512.06951](https://arxiv.org/abs/2512.06951) | Task adaptation of VLA |
| 2 | Comet (NVIDIA Research) | 0.251 | [mli0603/openpi-comet](https://github.com/mli0603/openpi-comet) | [arXiv 2512.10071](https://arxiv.org/abs/2512.10071) | openpi 기반 |
| 3 | SimpleAI Robot (베이징 스타트업 深朴智能) | 0.159 | 공개 안 됨 | 공개 안 됨 (대회 발표 제목 "SimBot-Agent") | 같은 팀 논문 [arXiv 2607.06256](https://arxiv.org/abs/2607.06256): agent가 π0.5 스킬을 호출·검증·재계획 → **우리 agent+VLA 구조와 가장 가까움** |
| 4 | The North Star (Huawei) | 0.120 | 공개 안 됨 | 공개 안 됨 | |

## 리모 ROS 2 (포팅된 것 찾기)

기본 리모는 Ubuntu 18.04 (ROS 1) 이라 ROS 2 로 포팅된 것을 찾아 쓴다. 우리 리모(Jetson Nano)에서 되는지는 확인 필요.
리모 **프로**(Jetson Orin Nano)를 받으면 ROS 2 Foxy 를 공식 지원하므로 이 절은 거의 필요 없다 ([LIMO Pro 공식 페이지](https://global.agilex.ai/products/limo-pro)).

| 저장소 | 내용 | 메모 |
|---|---|---|
| [agilexrobotics/limo_ros2](https://github.com/agilexrobotics/limo_ros2) | **공식** ROS 2 패키지. 브랜치 `foxy`, `humble`, `humble-dev` | humble 은 Ubuntu 22.04 기준 |
| [agilexrobotics/limo_ros2_doc](https://github.com/agilexrobotics/limo_ros2_doc) | 공식 ROS 2 사용 설명서 | |
| [LCAS/limo_platform](https://github.com/LCAS/limo_platform) | Docker 로 감싼 ROS 2 작업 공간 | NVIDIA Docker 런타임 필요 |
| [TechShare-inc/limo_ros2_docker](https://github.com/TechShare-inc/limo_ros2_docker) | Docker 로 리모를 ROS 2 (humble) 에서 움직이기 | NVIDIA Docker 사용 |
| [WeGo-Robotics/limo_ros2_ws](https://github.com/WeGo-Robotics/limo_ros2_ws) | 한국 WeGo 로보틱스의 리모 ROS 2 작업 공간 (한국어 설명) | |
| [Kazimbalti/limo_ros2](https://github.com/Kazimbalti/limo_ros2) | ROS 2 Humble: URDF, Gazebo 시뮬 등 | |
| [MoraesWilliam/Limo-Ros2-Gazebo-Slam-Cartographer](https://github.com/MoraesWilliam/Limo-Ros2-Gazebo-Slam-Cartographer) | ROS 2 Humble + Gazebo + Cartographer SLAM | 우리 SLAM 결정과 같은 조합 |
