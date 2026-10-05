# agent/devtools — 개발용 명령 (LLM 도구 아님)

여기 것은 **LLM 에게 보이지 않는다**. 판이 끝난 뒤 사람이 돌리는 분석 명령이다. LLM 도구는 [`../tools/`](../tools/README.md).

| 명령 | 하는 일 |
|---|---|
| `mock_eval.sh <tag> [run-skill 인자]` | 가짜 집(정답 바닥 + 카메라 흉내, 두 집 × 방향 3)에서 LLM 과 기준선(frontier)을 같이 돌려 비교. 결과 `behavior-2026/outputs/explore_mock_<tag>/` (스킬 폴더는 프롬프트만 — 10-06 에 skills/explore/ 에서 옮김) |
| `decisions-agg` | `decisions.jsonl` 여러 개를 실행기·모드·정책·프롬프트 판별로 묶어 마크다운 표(결정 수, reached/blocked/error 비율, 평균 이동 m, 평균 새 빈칸 m², m 당 새 빈칸, 접촉 증가, LLM 지연·토큰, 표지 상위). 옛 스킬 explore 에서 옮김(10-06) |

```bash
cd src/agent/devtools && cargo build --release -j4
./target/release/decisions-agg ../../behavior-2026/outputs/explore_*/ --by policy|mode|prompt|executor
```
기록 형식: [`../decision_log.md`](../decision_log.md).
