# agent

큰 계획·대화 (LLM). 사람과 대화하고, scene graph 를 읽어 VLA 에게 상황을 풀어 준다. 물체가 화면 밖으로 벗어나 VLA 가 움직일 수 없으면 다시 계획한다. (VLA 는 π0.5 가 아니라 우리 작은 VLA — [`training/BC`](../../training/BC/README.md). π0.5 는 10-04 버림.)

> **AI agent(최영식 교수님) 개인 프로젝트 범위** — 이 폴더는 김주영(juyoung020)이 혼자 작성한다.
> 나머지 폴더(물체 기억·행동·앱)는 로봇프로그래밍 팀 프로젝트 코드다.

| 수업에서 배운 것 | 여기서 쓰는 곳 |
|---|---|
| Tool calling 기반 Agent Loop | 물체 기억 조회, 이동, VLA 호출을 도구로 두고 LLM 이 골라 부른다 |
| Agent State · Context 관리 | 물체 지도(JSON)와 진행 상황을 필요한 만큼만 골라 LLM 에 넣는다 |
| Memory | 물체 기억(scene graph)을 장기 기억으로 쓰고, 대화 내용은 요약해 둔다 |
