# agent/runtime — 에이전트 런타임

스킬과 무관한 원형 도구 호출 루프(프레임워크 없음, 수업 규칙): messages + tools → tool_calls 면 실행해 `role:"tool"` 로 붙이고 다시 → 글이면 끝.
도구도 스킬도 아니다 — 스킬([`../skills/`](../skills/README.md))은 지시문(데이터)뿐이고, 도구([`../tools/`](../tools/README.md))는 실행되는 코드다.
옛 스킬 explore 크레이트(`skills/explore/src/lib.rs`·`bin/explore.rs`)에서 옮김(10-06).

| 파일 | 하는 일 |
|---|---|
| `src/prompts.rs` | 공통 `../prompts/common.md` + 스킬 `system.md`·`task.md`·도구 설명(`tool.md`)을 실행할 때 읽기, 판 이름(`<!-- version: … -->`)·FNV-64 지문 |
| `src/skill.rs` | 스킬 설정 `<skill>/skill.json`: 쓰는 도구(이름·설명 파일·LLM 에게 보일 mode), 첫 관측, LLM 없는 기준선 호출, 끝 조건·되묻기 문장, 지표 칸, 한도 |
| `src/tools.rs` | 도구를 이름으로 고르기(`by_name`) — 결과 줄이기·결정 기록 칸은 각 도구 크레이트 함수(move_robot `src/llm_view.rs`) |
| `src/lib.rs` | 루프 `run`: LLM / 기준선 정책, 429·5xx 다시 시도, 16k 맥락(최근 `keep_pairs` 쌍 + 오래된 것 한 줄 요약), 되묻기, 기록 `decisions.jsonl`([형식](../decision_log.md))·`timeline.jsonl`·`trace.jsonl`·`summary.json` |
| `src/bin/run_skill.rs` | 명령 `run-skill` |

```bash
cd src/agent/runtime && cargo build --release -j4 && cargo test --release -j4      # 시험 2개(explore 설정 읽기·LLM 에게 보이는 정의, 두 방 기준선 탐사)
set -a; . ~/.config/robot-agent/kau.env; set +a                                   # LLM 키는 환경변수로만
./target/release/run-skill --skill explore --policy llm|baseline \
    [--addr 127.0.0.1:8771 | --mock <gt.pgm> --gt <gt.json> [--start x,y,yaw_deg]] --out <dir> \
    [--max-calls N] [--max-sim-s S] [--max-wall-s S] [--keep 3] [--temperature T] [--thinking] [--no-nudge] [--style compact] [--task NAME] [--run-id ID]
```
- `--policy baseline` = `skill.json` 의 `baseline` 호출을 되풀이(LLM 없음). 그 이름(explore 는 `frontier`)도 받고, 기록의 `policy` 는 그 이름.
- 기준선·LLM 모두 같은 도구·같은 관측. 묶음 호출(move_robot 모드 `explore`)은 결과의 `_calls` 로 실제 실행 하나마다 결정 기록 한 줄.
- 끝: 호출 수·시뮬 s·벽시계 한도 / LLM 이 글로 끝냄(끝 조건 참이면 `llm_done_<이름>`, 아니면 한 번 되묻고 `llm_done_early`) / 기준선은 끝 조건 참이면 `<이름>`.
