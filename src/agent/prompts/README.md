# agent/prompts — 여러 스킬이 같이 쓰는 프롬프트 조각

스킬마다의 시스템 프롬프트는 `../skills/<skill>/system.md` 에 있다. 여기에는 모든 스킬이 앞에 붙이는 공통 조각만 둔다.

| 파일 | 내용 |
|---|---|
| `common.md` | 로봇 소개(R1Pro, 앞만 보는 머리 카메라)와 도구 쓰는 규칙(한 번에 하나, 결과 id 만 쓰기, error/blocked 는 관측, 같은 실패 되풀이 금지, 끝나면 한 줄) |

판 관리
- 파일 첫 줄 `<!-- version: common-vN -->`. 에이전트 런타임(`runtime/`, `run-skill --skill <스킬>`)이 실행할 때 읽어 시스템 프롬프트 맨 앞에 붙이고, 판 이름들(`common-v1+explore-v1+…`)과
  모든 프롬프트 파일을 이은 바이트의 FNV-64 지문을 `summary.json`·`decisions.jsonl`(`prompt.version`, `prompt.sha`)에 남긴다.
- 고칠 때: 판 번호를 올리고, 쓰는 스킬의 `CHANGELOG.md` 에 무엇을 왜 고쳤는지와 잰 효과(같은 집·같은 출발에서 덮은 비율·호출 수·토큰)를 적는다.
- 하드코딩하지 않는다: 코드에는 기본 도구 설명(`move_robot::DESCRIPTION`)만 있고, 스킬은 `tool.md` 로 덮어쓴다.
