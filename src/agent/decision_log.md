# 결정 기록 (`decisions.jsonl`) — 모든 스킬 공통

LLM(또는 기준선)이 도구를 한 번 고를 때마다 한 줄. 목적: "무엇을 보고 → 어떤 실행기·모드를 골랐고 → 어떻게 됐나"를 쌓아,
실행기 고르기(π0.5 VLA 문장 vs `move_robot` 직접 제어, `go_to` vs `probe`)와 프롬프트 판을 데이터로 판단한다.
쓰는 곳: `skills/explore/src/lib.rs`(`DecisionLog`, `obs_features`, `label`). 저장: `outputs/<run>/decisions.jsonl`.

## 필드

| 필드 | 뜻 |
|---|---|
| `run`, `task`, `skill`, `policy` | 판 이름, BEHAVIOR 과제, 스킬(`explore`), 정책(`llm` / `frontier` 기준선) |
| `step`, `t`, `sim_s` | 판 안 결정 번호, 벽시계, 결과 시점 시뮬 시각(s) |
| `prompt.version`, `prompt.sha` | 프롬프트 판 이름들(`common-v1+explore-v2+…`)과 파일 바이트 FNV-64 지문 |
| `obs` | 결정 직전 관측 특징: `free_m2`, `n_frontiers`, `best`(첫 프런티어 id·path_m·new_area_m2), `nearest`(경로 가장 짧은 것), `in_room`, `front`(앞 방향 여유 글), `last_status` |
| `executor` | `move_robot` / `vla`(π0.5 문장) |
| `mode`, `args` | `base:go_to`, `base:probe`, `base:delta`, `left_arm:absolute` … 와 도구 인자 그대로 |
| `target_space` | `known`(go_to: 아는 빈칸 경로) / `unknown`(probe) / `n/a` |
| `outcome.status` | `reached` / `blocked` / `timeout` / `error` / `unavailable`(VLA 없음) |
| `outcome.moved_m`, `time_s`, `stopped_by`, `replans`, `new_free_m2` | 움직인 거리, 걸린 시뮬 시간, 멈춘 이유(지도 장애물·카메라 장애물·모르는 곳·접촉), 다시 계획한 횟수, 새로 생긴 빈칸 |
| `outcome.gt_cov_before/after` | 정답 바닥 대비 지도 빈칸 비율(LLM 에게는 안 보임) |
| `outcome.contacts`, `stalls`, `min_clear_m` | 누적 접촉 수(시뮬: 로봇 링크 ↔ 바닥 아닌 것, 가짜 집: 몸통 원이 벽 칸에 닿음), 밀었는데 안 움직인 횟수, 움직이는 동안 가장 작은 장애물 거리 |
| `outcome.exec_wall_ms`, `message` | 도구 실행 벽시계 시간, 오류 문장 |
| `llm` | `latency_ms`, `prompt_tokens`, `completion_tokens`, `call_id`, `name`, `n_calls`(한 응답의 호출 수) — 기준선은 null |
| `label.tags` | 사후 표지: `invalid_call`, `goto_into_unknown`(모르는 곳으로 go_to — 아는/모르는 판단 틀림), `wasted_probe`(probe 앞으로가 0.3 m 도 못 가고 막힘), `no_gain_look`, `no_gain_move`, `delta_for_explore` |
| `label.rank` | go_to 로 고른 프런티어가 관측 목록 몇 번째였나(1 = 가장 좋다고 보인 것) |
| `label.vla_better` | 다른 실행기가 나았나 — 지금은 VLA 실행기가 없어 null(아래) |

## 집계

```bash
src/agent/skills/explore/target/release/decisions-agg outputs/explore_*/ --by mode     # 실행기/모드별
decisions-agg <dirs> --by policy    # 정책 × 모드
decisions-agg <dirs> --by prompt    # 프롬프트 판 × 정책
```
열: 결정 수, reached/blocked/error 비율, 평균 이동 m, 평균 새 빈칸 m², m 당 새 빈칸, 접촉 증가, LLM 지연·토큰, 표지 상위.

## VLA ↔ move_robot 전환 (상태)

- 접점: 평가기 쪽 `src/behavior-2026/src/sim/explore/run_explore.py` 가 도구 호출 `{"executor":"vla","skill":"<π0.5 문장>","max_s":20}` 를
  가로채 `act()` 를 π0.5 네이티브 엔진(`Pi05NativePolicy`)으로 돌리고, 끝나면 `move_robot` 실행기를 `reset`(유지 목표를 다음 관측에서 다시 잡음)해
  되돌린다. 결과 `{"status":"done","executor":"vla",…}`.
- 지금: `--vla-weights` 없이 돌려 `{"status":"unavailable"}` 를 돌려준다(탐사는 베이스만이라 필요 없음). LLM 에게는 아직 이 실행기를 도구로 보이지 않는다.
- 쓰려면: (1) 탐사 판에 `--vla-weights <…>.pi05w` 를 주고 VRAM(Isaac Sim + π0.5 ≈ 15 GB)을 확인, (2) 에이전트에 두 번째 도구 `run_skill(skill 문장)`
  을 더해 `{"executor":"vla"}` 로 보냄, (3) `label.vla_better` 를 채우는 규칙(같은 하위 목표를 두 실행기로 했을 때 성공·시간 비교) — 같은 장면 짝 실험이 필요.
