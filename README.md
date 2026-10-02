<div align="center">

<h2>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-bot-dark.svg">
    <img src="docs/assets/icon-bot-light.svg" width="28" height="28" alt="">
  </picture>
  우리는 물체를 기억하는 로봇을 만든다 - 자유주제
</h2>

<a href="https://plausible-hallway-e4f.notion.site/3eb454b08de28190b2e8e321a33a9371">팀 노션</a> &nbsp;·&nbsp; <a href="docs/plan.md">계획</a> &nbsp;·&nbsp; <a href="docs/model_selection.md">모델 선택</a> &nbsp;·&nbsp; <a href="refs/README.md">참고 자료</a>

<br><br>

<img src="docs/assets/memory_viewer.png" width="760" alt="물체 기억 뷰어">

<sub>로봇의 물체 기억(Spark-DSG) 뷰어 — 2D 지도 위에 물체를 세그먼트 점구름으로 등록하고, 누르면 위치·상태와 그 물체를 본 순간의 RGB·깊이 조각을 보여 준다 (BEHAVIOR 시뮬레이션)</sub>

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
robot-agent/
├── docs/            # 계획(plan.md), 모델 선택, 후보 조사, 회의 자료, 발표 자료 (목록은 docs/README.md)
├── src/             # 코드 (ROS 2 패키지)
│   ├── scene_graph/ # ① 물체 기억
│   ├── agent/       # ② 큰 계획·대화 (LLM)·실패 복구
│   │   ├── skills/  #   스킬(한 가지 일을 끝까지 하는 단위, 지금은 explore)
│   │   ├── tools/   #   LLM 에게 보이는 도구(move_robot, Rust)
│   │   └── prompts/ #   공통 프롬프트
│   ├── vla/         # ③ 작은 계획·행동 (VLA, π0.5)
│   ├── app/         # 휴대폰 앱 (iOS·Android, 채팅으로 명령)
│   └── behavior-2026/ # 서브모듈: BEHAVIOR Challenge 2026 (시뮬레이터에서 같은 구조를 시험)
│       ├── src/scene_graph/ # scenemap·ovdet·clip(물체 영상 임베딩)·runtime(sgrt)·viewer
│       ├── third_party/spark_dsg/ # Spark-DSG 우리 사본 (BSD-3)
│       ├── tools/   #   실행·측정·검증 스크립트
│       └── archive/ #   지금 안 쓰는 모듈 (지우지 않고 옮겨 둠)
├── training/        # 로봇에 올릴 작은 모델 학습 (embed/: 영상–글 임베딩 증류)
├── scripts/         # 설치·실행 스크립트
├── tests/           # 테스트 (sandbox/ 는 AI·사람 실험 공간)
└── refs/            # 참고 논문·코드 목록
```

## 문서

- [docs/README.md](docs/README.md) — 이 저장소 문서 목록
- [docs/clip_candidates.md](docs/clip_candidates.md) — CLIP 류 임베딩 모델 후보·측정
- [training/README.md](training/README.md) — 작은 모델 학습
- [scenemap 설계](https://github.com/juyoung020/behavior-2026/blob/main/docs/scenemap_설계.md) — 물체 기억(2D SLAM·물체 지도·계획기 질의) 설계 (서브모듈)
- [archive/README.md](https://github.com/juyoung020/behavior-2026/blob/main/archive/README.md) — 지금 안 쓰는 모듈: 무엇을, 왜, 어떻게 되살리나 (서브모듈)
- [tools/README.md](https://github.com/juyoung020/behavior-2026/blob/main/tools/README.md) — 실행·측정·검증 스크립트 (서브모듈)

## 정한 것

- 이미지 임베딩: SigLIP 2 B/32.
- 분할: FastSAM-s, 입력 416.
- 물체 벡터는 원본 임베딩 그대로 두고, 이름은 기억 폴더의 `cache/` 에 둔다.
- CUDA 12.8.
- 지도 자세: `SGRT_POSE` 로 고른다(`slam`·`odom`·`gt`, 실제 로봇 기본 `slam`, 시뮬 시험은 `gt`).

## BEHAVIOR Challenge 2026 (서브모듈)

[`src/behavior-2026`](https://github.com/juyoung020/behavior-2026) 은 같은 구조(물체 기억 + LLM 계획 + π0.5)를 Stanford BEHAVIOR Challenge 2026 시뮬레이터(OmniGibson, Isaac Sim 5.1)에서 시험하는 저장소다. 물체 기억은 scenemap(2D SLAM + YOLOE 물체 지도), 행동은 π0.5 네이티브 CUDA 엔진. 실행 환경은 Ubuntu 22.04 + RTX 4090(자세히는 그 저장소의 `docs/Linux_설치.md`).

```bash
git submodule update --init src/behavior-2026   # 서브모듈 받기 (그 안의 BEHAVIOR-1K 등은 필요할 때 --recursive)
```

## 시작하기

```bash
git clone https://github.com/juyoung020/robot-agent.git
cd robot-agent
bash refs/download.sh   # 참고 논문 PDF·코드를 refs/ 에 받기 (깃에는 안 올라감)
python3 -m pytest tests/  # 테스트 실행
```
