# agent/skills — 스킬

**스킬** = LLM 에이전트가 한 가지 일을 끝까지 해내는 단위. **스킬 = 프롬프트·지시문 + 어떤 도구를 쓰나. 코드는 `../tools/`(도구) 나 `../runtime/`(루프) 에 둔다** —
스킬 폴더에는 실행되는 코드를 두지 않는다(10-06). 스킬 폴더 = `README.md`(목적·도구·끝 조건·지표·결과 표), `system.md`, `task.md`, `tool.md`, `CHANGELOG.md`,
`skill.json`(설정 데이터), 런타임만 부르는 평가 스크립트. 스킬 하나는 다음으로 정해진다.

| 구성 | 뜻 | 파일 |
|---|---|---|
| 시스템 프롬프트 | 이 일의 목표·전략·도구 쓰는 법(실행할 때 파일에서 읽음, 판·지문을 기록에 남김) | `<skill>/system.md`, 과제 문장 `task.md`, 이 스킬에서 보일 도구 설명 `tool.md` |
| 쓰는 도구 | LLM 에게 보이는 도구(전체 8개 이하, 이 스킬에서는 더 적게) | `../tools/` |
| 끝 조건 | 언제 끝났다고 보나(LLM 이 글로 끝냄 / 런타임 한도) | `<skill>/README.md`, 데이터는 `skill.json` `end`(+ 되묻기 문장) |
| 성공 지표 | 무엇으로 잘했나를 재나(기준선과 같은 지표) | `<skill>/README.md`, 결과 표, 기록 칸은 `skill.json` `metrics` |
| 기준선 | LLM 없이 같은 도구로 하는 호출(도구 쪽 코드) | `skill.json` `baseline` |

공통 형식
- 모든 스킬은 공통 조각 `../prompts/common.md`(로봇 소개·도구 쓰는 규칙)를 시스템 프롬프트 앞에 붙인다.
- 프롬프트 파일 첫 줄은 `<!-- version: <이름>-vN -->`. 고칠 때마다 N 을 올리고 `<skill>/CHANGELOG.md` 에 이유·잰 효과를 적는다.
- 실행은 에이전트 런타임 [`../runtime/`](../runtime/README.md)(`run-skill --skill <이름> --policy llm|baseline`): 원형 도구 호출 루프(프레임워크 없음, 수업 규칙)와 같은 도구를 쓰는 LLM 없는 기준선.
- 결정마다 `decisions.jsonl` 한 줄(형식: [`../decision_log.md`](../decision_log.md)).

| 스킬 | 한국어 | 하는 일 | 도구 | 상태 |
|---|---|---|---|---|
| [`explore`](explore/) | 집 탐사 | 2D 지도가 닿을 수 있는 집 전체를 덮을 때까지 부딪히지 않고 짧게 돌아다니기 | `move_robot`(base `go_to`·`probe`, 기준선은 모드 `explore`) | 시뮬 두 집에서 접촉 0, 덮음 0.94–0.95 (LLM, `explore-v4`) |
