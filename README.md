<div align="center">

<h2>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-bot-dark.svg">
    <img src="docs/assets/icon-bot-light.svg" width="28" height="28" alt="">
  </picture>
  우리는 물체를 기억하는 로봇을 만든다 - 자유주제
</h2>

<a href="https://plausible-hallway-e4f.notion.site/3eb454b08de28190b2e8e321a33a9371">팀 노션</a> &nbsp;·&nbsp; <a href="docs/plan.md">계획</a> &nbsp;·&nbsp; <a href="docs/model_selection.md">모델 선택</a> &nbsp;·&nbsp; <a href="refs/README.md">참고 자료</a>

</div>

<br>

로봇이 집 안을 돌아다니며 본 물체를 기억해 둔다. 그래서 사람이 물체를 찾거나 옮겨 달라고 하면, 어디 있는지 다시 뒤지지 않고 기억을 떠올려 바로 움직인다.

## 이런 걸 할 수 있어요

| 사람이 말하면 | 로봇은 |
|---|---|
| "컵 어디 있었지?" | 기억에서 찾아 "주방 식탁 위에 있었어요" 라고 답한다 |
| "컵을 식탁에 갖다 놔" | 컵 위치로 이동 → 집기 → 식탁으로 이동 → 놓기 |
| (누가 컵을 옮겨 놓으면) | 다음에 봤을 때 기억을 새 위치로 고친다 |
| (중간에 집기를 실패하면) | 다시 확인하고 재시도한다 |

## 어떻게 동작하나요

```
카메라·라이다 ──▶ ① 물체 기억 (scene graph) ──▶ ② 큰 계획·대화 (LLM) ──▶ ③ 작은 계획·행동 (VLA, π0.5) ──▶ 로봇
                   "무엇이 어디에 있나"          "무엇을 어떤 순서로"           "지금 이 단계를 어떻게"
                          ▲                                                                   │
                          └─────────────────────── 움직이며 본 것으로 기억 갱신 ◀─────────────┘
```

1. **물체 기억** — 로봇이 본 물체를 2D 지도 위에 등록하고, 옮겨지거나 사라지면 고친다.
2. **큰 계획·대화 (LLM)** — 사람과 채팅으로 대화하고, 물체 기억을 읽어 π0.5 에게 상황을 풀어 준다 ("컵은 주방 식탁 위, 놓을 곳은 거실 식탁"). 물체가 화면 밖으로 벗어나 π0.5 가 움직일 수 없으면, 기억을 보고 다시 계획한다.
3. **작은 계획·행동 (VLA)** — LLM 의 지시와 카메라 영상을 받아, 할 일을 잘게 나눠(팔 뻗기 → 잡기 → 들기) 로봇 팔·바퀴를 실제로 움직인다. 집다가 놓치는 것처럼 눈앞에서 생긴 실패는 스스로 복구한다.

## 로봇

AgileX 리모(LIMO) 기본형(가정, **프로를 받을 수도 있음**) + 매니퓰레이터(모델 미정). 자세한 전제와 할 일은 [`docs/plan.md`](docs/plan.md).

## 폴더 구조

```
robot-programming-team/
├── docs/            # 계획(plan.md), 회의 자료, 설계 문서, 발표 자료
├── src/             # 코드 (ROS 2 패키지)
│   ├── scene_graph/ # ① 물체 기억
│   ├── agent/       # ② 큰 계획·대화 (LLM)·실패 복구
│   ├── vla/         # ③ 작은 계획·행동 (VLA, π0.5)
│   └── app/         # 휴대폰 앱 (iOS·Android, 채팅으로 명령)
├── scripts/         # 설치·실행 스크립트
├── tests/           # 테스트 (sandbox/ 는 AI·사람 실험 공간)
└── refs/            # 참고 논문·코드 목록
```

## 시작하기

```bash
git clone https://github.com/juyoung020/robot-programming-team.git
cd robot-programming-team
bash refs/download.sh   # 참고 논문 PDF·코드를 refs/ 에 받기 (깃에는 안 올라감)
python3 -m pytest tests/  # 테스트 실행
```
