# robot-programming-team
로봇프로그래밍 팀 프로젝트

동적 3D scene graph + AI agent (또는 RL planner) + VLA 결합

## 폴더 구조

```
robot-programming-team/
├── docs/            # 회의 자료, 설계 문서, 발표 자료
├── src/             # 코드 (ROS 2 패키지)
│   ├── scene_graph/ # 3D scene graph: 로봇이 본 물체 기억
│   ├── agent/       # 장기 계획, 단계 추적, 실패 복구
│   └── vla/         # π0.5: 지시 + 카메라 영상 → 행동
└── scripts/         # 설치·실행 스크립트
```
