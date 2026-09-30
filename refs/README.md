# refs

참고 논문·코드·자료 모음. 레포가 **공개(public)** 이므로 논문 PDF는 올리지 말고 링크만 적는다 (저작권).

## VLA (행동 모델)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| π0.5: a Vision-Language-Action Model with Open-World Generalization (CoRL 2025) | [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 우리가 쓸 VLA. 단계(subtask) 지시 + 카메라 → 행동 |
| π0: A Vision-Language-Action Flow Model for General Robot Control | [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5의 전신 |

## 3D Scene Graph (로봇의 기억)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| DovSG: Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation | [arXiv 2410.11989](https://arxiv.org/abs/2410.11989) | [BJHYZJ/DovSG](https://github.com/BJHYZJ/DovSG) | **우리 아이디어와 가장 가까움.** 그래프 국소 갱신 + LLM 작업 계획 |
| Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization (RSS 2022) | [arXiv 2201.13360](https://arxiv.org/abs/2201.13360) | [MIT-SPARK/Hydra](https://github.com/MIT-SPARK/Hydra) | 실시간 계층형 scene graph (물체·장소·방·건물) |
| Khronos: Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments (RSS 2024) | [arXiv 2402.13817](https://arxiv.org/abs/2402.13817) | | Hydra 후속. 시간에 따른 변화 감지 |
| ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning | [arXiv 2309.16650](https://arxiv.org/abs/2309.16650) | | 2D 기반 모델을 3D로 융합, 열린 어휘(open-vocabulary) 물체 노드 |
| HOV-SG: Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation (RSS 2024) | [arXiv 2403.17846](https://arxiv.org/abs/2403.17846) | | 층·방·물체 계층 + 열린 어휘 |

## Agent (Scene graph + LLM 계획)

| 제목 | 논문 | 코드 | 메모 |
|---|---|---|---|
| SayPlan: Grounding LLMs using 3D Scene Graphs for Scalable Robot Task Planning (CoRL 2023) | [arXiv 2307.06135](https://arxiv.org/abs/2307.06135) | | 그래프에서 필요한 부분만 LLM이 탐색 + 반복 재계획 |
| MoMa-LLM: Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation | [arXiv 2403.08605](https://arxiv.org/abs/2403.08605) | | 탐색하며 갱신되는 scene graph 위에서 LLM이 행동 선택 |
