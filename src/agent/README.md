# agent

큰 계획·대화 (LLM). 사람과 대화하고, scene graph 를 읽어 VLA 에게 상황을 풀어 준다. 물체가 화면 밖으로 벗어나 VLA 가 움직일 수 없으면 다시 계획한다. (VLA 는 π0.5 가 아니라 우리 작은 VLA — [`training/BC`](../../training/BC/README.md). π0.5 는 10-04 버림.)

> **AI agent(최영식 교수님) 개인 프로젝트 범위** — 이 폴더는 김주영(juyoung020)이 혼자 작성한다.
> 나머지 폴더(물체 기억·행동·앱)는 로봇프로그래밍 팀 프로젝트 코드다.

| 수업에서 배운 것 | 여기서 쓰는 곳 |
|---|---|
| Tool calling 기반 Agent Loop | 물체 기억 조회, 이동, VLA 호출을 도구로 두고 LLM 이 골라 부른다 |
| Agent State · Context 관리 | 물체 지도(JSON)와 진행 상황을 필요한 만큼만 골라 LLM 에 넣는다 |
| Memory | 물체 기억(scene graph)을 장기 기억으로 쓰고, 대화 내용은 요약해 둔다 |

## 도구 (만든 것)

| 도구 | 하는 일 | 폴더 |
|---|---|---|
| `search_objects` | 물체 기억 찾기: 이름·동의어·상위어 → 없거나 약하면 이름 무시 생김새 재검색(벡터는 도구 안, LLM 에는 이름·속성·방·기준물·상태 글), 물을지 `ask_user` | [`tools/search_objects`](tools/search_objects/) |
| `confirm_object` | 확인된 물체 이름 고치기(이름 사후 베이즈 갱신, 확인 기록 = 보정 데이터) | [`tools/search_objects`](tools/search_objects/) |
| `move_robot` | 로봇 한 부분(베이스·몸통·팔·그리퍼) 직접 움직이기, 탐사 베이스 모드, VLA 실행기 | [`tools/move_robot`](tools/move_robot/) |

- `move_robot` 의 베이스 모드 `explore` = 프런티어 탐사(옛 스킬 explore 기준선 코드, 10-06) — 새 도구가 아니라 모드라 LLM 에 보이는 도구 수는 그대로.

도구 설명은 [`tools/README.md`](tools/README.md), 계획만 있는 도구까지 전체 설계는 [`plan.md`](plan.md) 3.3.

## 폴더 — 코드는 도구·런타임에, 스킬은 지시문만

| 폴더 | 무엇 | LLM 에게 보이나 |
|---|---|---|
| [`tools/`](tools/README.md) | 도구(실행되는 코드, 크레이트마다 하나) | 보임(스킬이 고른 것만) |
| [`runtime/`](runtime/README.md) | 에이전트 런타임: 원형 도구 호출 루프, 프롬프트 읽기·판 지문, 결과 줄이기·16k 맥락 접기, 결정·시간·LLM 기록, 명령 `run-skill` | — |
| [`skills/`](skills/README.md) | 스킬 = 프롬프트(`system.md`·`task.md`·`tool.md`)·`skill.json`(도구·끝 조건·지표 데이터)·CHANGELOG·결과 표. 코드 없음 | 프롬프트로 |
| [`prompts/`](prompts/) | 모든 스킬 공통 프롬프트 조각 `common.md` | 프롬프트로 |
| [`devtools/`](devtools/README.md) | 개발 명령(`decisions-agg`) — LLM 도구 아님 | 안 보임 |

```bash
(cd runtime && cargo build --release -j4) && runtime/target/release/run-skill --skill explore --policy frontier|llm --mock <gt.pgm> --gt <gt.json> --out <dir>
```
