# explore 프롬프트·관측 고친 기록

같은 조건에서 잰다: 가짜 집 두 곳(`house_single_floor`·bringing_water 방 6개 161 m², `house_double_floor_lower`·turning_on_radio 방 3개 55 m², 정원 뺌),
과제 instance 시작 자리에서 방향 0°/120°/240° 세 출발(`devtools/mock_eval.sh`, 10-06 에 skills/explore/ 에서 옮김), Qwen3.5-9B(KAU, 온도 0, 생각 끔). 덮음 = 정답 바닥(닿을 수 있는 곳 + 0.6 m) 중 지도 빈칸 비율.

| 판 | 무엇을 | 왜 | 잰 효과 |
|---|---|---|---|
| `explore-v1` (`common-v1`, `explore-tool-v1`) | 지도 요약 설명, go_to(아는 곳)·probe(모르는 곳) 구분, 전략 한 줄("넓은 new_area 와 짧은 path_m 사이"), 프런티어 없을 때만 끝 | 첫 판 | 원 몸통(0.30 m) 제어기에서: 덮음 0.99·0.99(1 층 집 s0·s1), 0.996(2 층 집 아래층) 이지만 **경로가 기준선의 1.6 배**(1 층 s1 130 m vs 79 m). 결정 기록을 보니 모델이 F1→F2→F3→F4→F5→F1… 차례로 고름(id 를 할 일 목록으로 읽음, rank 1·2·3·4·5 반복) |
| `explore-v2` | "id 는 움직일 때마다 다시 매기고 가장 좋은 것이 F1. 목록을 차례로 돌지 말고 매번 새 목록에서 F1(막혔거나 훨씬 가까운 것이 있을 때만 다른 것)" | v1 의 차례 고르기 | 같은 제어기·같은 출발에서 1 층 경로 85→78 m(s0), 130→77 m(s1), 덮음 0.95–0.99 유지, 2 층 아래층 경로 44→24–32 m(기준선 22–34 m 와 같은 수준). LLM 호출 9–21 번, 토큰 2.2–5.8 만 |

관측 모양 실험(코드, `robot_nav.rs` `ObsStyle`)
- `full`(기본): 프런티어 5 개(id·path_m·dir·new_area_m2·room) + 8 방향 여유(지도 아는 거리·끝이 벽인가 모름인가·지금 깊이) + 방 목록 + status. 결과 한 번 약 1.2–1.6 k 글자.
- 맥락: 최근 3 쌍만 그대로, 그 전은 "n. go_to F1 -> reached, +5.3 m2" 한 줄로 접음 → 요청 하나 약 2.4–2.7 k 토큰(16 k 의 1/6)에서 멈춤.

관측 변형 비교(같은 제어기 f82dc18+, `explore-v2`, 출발 6 개, `mock_eval.sh v2full` / `v2compact --style compact`)

| 변형 | 덮음 평균(6) | 경로 m 합 | LLM 호출 합 | 토큰 합 | 접촉 |
|---|---|---|---|---|---|
| `full`(8 방향 여유 포함) | 0.909 | 248.5 | 97 | 252.4 k | 0 |
| `compact`(여유 뺌) | 0.905 | 240.0 | 97 | 213.3 k | 0 |

→ 결정은 거의 같고(같은 프런티어 고름) 토큰만 15 % 적다. 9B 는 go_to 만 쓰는 동안 8 방향 여유를 거의 읽지 않는다(probe 를 안 씀).
여유는 probe 를 쓸 때(지도 끝이 모름인 방향 판단)를 위해 기본(`full`)에 남겨 두고, 맥락이 빠듯한 스킬에서는 `compact` 를 쓴다.
덮음이 낮은 한 판(1 층 s0 0.55)은 LLM 이 아니라 제어기 문제(좁은 막다른 복도에서 사각형 몸통이 못 돌아 나옴 — 기준선도 같은 곳에서 0.40–0.61).

| 판 | 무엇을 | 왜 | 잰 효과 |
|---|---|---|---|
| `explore-v3` (+ 코드: 두 번 실패한 방 목표는 id 를 빼고 `note: go_to failed twice`) | "blocked·timeout 이면 같은 target 을 다시 보내지 말 것, 방은 visited=false 이고 path_m 이 숫자일 때만" | 시뮬 v2 판(`explore_20261003_082657_…llm_c37`): 프런티어가 다 떨어진 뒤 못 가는 방 R5 로 go_to 를 **15 번** 되풀이(전부 timeout, 덮음 0.654 에서 멈춤, 시뮬 300 s 낭비) | 시뮬 같은 집·같은 출발(`…085903_…llm_v3`): 덮음 **0.950**(기준선 0.867), 접촉 0, 실패 방은 두 번 뒤 포기. 가짜 집 6 판 덮음 평균 0.980(v2 0.979), 경로 합 320 m(v2 293 m) |
| `explore-v4` | "status 가 no reachable frontier left 이고 갈 방이 없으면 곧바로 done — 더 probe 해도 벽·닫힌 문은 못 지남" | v3 시뮬 판: 덮음 0.95 에 닿은 뒤(호출 13) 못 가는 방 2 번 + probe 5 번(덮음 증가 0)을 더 하다 사람이 끊음 | 시뮬 2 층 집 아래층(`…092910_turning_on_radio_llm_v4`): 프런티어가 0 이 되자 go_to R2(오류) → probe 1 번 → **스스로 done**(LLM 9 번, 2.2 만 토큰, 덮음 0.941 / 기준선 0.925) |

## 10-06 코드 옮김 (프롬프트 판 그대로)

이 폴더의 Rust 크레이트(`explore-skill`)를 없앴다 — 스킬은 지시문만. 루프·프롬프트 읽기·맥락 접기·기록 → 에이전트 런타임 `src/agent/runtime`(`run-skill --skill explore`),
기준선(프런티어 탐사) → `move_robot` 베이스 모드 `explore`(`tools/move_robot/src/frontier.rs`), 결과 줄이기·결정 기록 칸 → `tools/move_robot/src/llm_view.rs`,
가짜 집 정답 기준 → `tools/move_robot/src/mock_eval.rs`, `decisions-agg` → `src/agent/devtools`. 스킬 설정은 `skill.json`(데이터).
같은 조건 확인: `mock_eval.sh`(기준선 6 출발)의 decisions·timeline·summary 가 옛 바이너리와 벽시계·시각 칸 말고 모두 같고, LLM 짧은 판(2 층 집 s0, 6 호출)도 같은 프롬프트 지문 `3b1ee39244394fa6`·같은 고름·같은 토큰(13 655 + 312).
