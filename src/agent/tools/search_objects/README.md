# search_objects · confirm_object — 물체 기억 찾기와 이름 고치기

LLM(Qwen3.5-9B, KAU API)은 글만 주고받으므로 **벡터는 도구 안에서만** 쓴다. 찾기·이름 확률·속성 낱말은 공용 물체 색인
(behavior-2026 [`src/scene_graph/clip`](../../../behavior-2026/src/scene_graph/clip/README.md) `sgsearch.h`, C++)이 하고,
이 크레이트는 그 결과에 기억의 자리 정보(방·기준물·상태·마지막으로 본 때·로봇까지 거리)를 붙여 짧은 JSON 글로 돌려준다.
같은 색인을 RecallVLA 실행기는 자기 질의 벡터로 쓴다(`sgs_search_vec`) — 에이전트가 넘기는 물체 id 는 VLA 에 **힌트**다.
설계 기록: [plan.md 3.3](../../plan.md), [MAPVLA_SPEC 공용 물체 찾기](../../../../docs/map_vla/MAPVLA_SPEC.md#공용-물체-찾기-10-05).

## 3 단계

```
"라디오 가져와"
 ① 이름 검색   등록·확인된 이름 / 동의어 / 상위어("chair" → 등록 "straight chair"), 한국어 이름 전부   → 라디오 없음
 ② 생김새 재검색(자동: 이름 후보가 없거나 약할 때)  물체마다 저장된 시점 벡터로 P(질의어 | 모습)을 라벨 표 안에서
    등록 이름과 견줌(물체 안 상대 확률)  → O234: 등록 "fire extinguisher", p_query 0.15 vs p_registered 0.06
 ③ LLM 이 되묻거나(ask_user) O234 를 VLA 에 힌트로 → 확인되면 confirm_object(O234, radio, user)
    → 이름 사후에 강한 관측(베이즈 갱신) → 다음 "라디오" 는 ① 에서 바로. 확인은 모두 기록(보정 데이터)
```

## 스키마 (두 개 합쳐 약 2 KB — `search_objects::definitions()`, `search-objects schema`)

```json
search_objects {"query": string (필수, 짧은 낱말 — 영어가 가장 낫고 한국어도 됨),
                "k": int 1..10 (기본 5), "room": string (방 이름 또는 "R2"), "state": "seen|moved|held|gone",
                "near": string (물체 id — 그 물체 1.5 m 안만)}
confirm_object {"id": "O27" (필수), "name": string (필수, 실제로 무엇인지), "source": "user|close_look" (필수)}
```

## 결과

```json
{"query":"라디오","searched":"name+appearance",
 "ask_user":"nothing is registered as '라디오'; O234 is registered as 'fire extinguisher' but looks like '라디오' (p 0.15 vs 0.06)",
 "hint":"ask the user, or pass O234 as a hint and verify by a close look; then call confirm_object",
 "matches":[{"id":"O234","name":"fire extinguisher","name_p":0.17,"alt":[{"name":"radio","p":0.13},{"name":"fuel can","p":0.07}],
             "match_type":"appearance","p_query":0.15,"p_registered":0.06,"attrs":["red","small"],"room":"office",
             "landmark":{"id":"O191","name":"sofa","dist_m":0.0,"dz_m":-1.65},"state":"gone","last_seen_ago_s":89,"dist_m":1.49,"match":0.15}]}
```

| 칸 | 뜻 |
|---|---|
| `id` | 물체 id(`set_plan`·VLA 호출에 그대로) |
| `name`, `name_p`, `alt` | 이름 사후(생김새 + 등록 이름 + 확인)의 1 위와 확률, 다른 이름 ≤ 3 개(p ≥ 0.03). 등록 이름과 그 아래말은 한 묶음으로 셈 |
| `registered` | 기억에 등록된 이름이 `name` 과 다를 때만 |
| `match_type` | `name`(등록·확인 이름이 질의와 맞음) / `appearance`(이름은 안 맞는데 생김새가 맞음) |
| `p_query`, `p_registered` | appearance 일 때만: **이름 무시** 생김새 확률 — 질의어 대 등록 이름(같은 물체 안에서 견준 값) |
| `looks_like` | 이름으로 확실한 다른 물체와 사진끼리 닮아서 나왔으면 그 id |
| `attrs` | 색·재질(색인이 물체 벡터와 낱말 표를 견줘 확률 ≥ 0.6 인 것만) + 크기 낱말(가장 긴 변: tiny < 0.1 m, small < 0.3, medium < 0.8, large < 1.5, very large) |
| `room` | 방 이름(기억의 방 분할) |
| `landmark` | 1.5 m 안에서 가장 가까운 고정 가구: `id`, `name`, `dist_m`(상자까지 수평 거리), `dz_m`(물체 중심 − 가구 윗면 높이). **on / in 같은 관계말은 계산하지 않는다**(10-05) — LLM 이 숫자로 고름 |
| `state`, `last_seen_ago_s` | seen / moved / held / gone, 기억 시계 기준 경과 초 |
| `dist_m` | 로봇(view.json `pose`)까지 수평 거리 |
| `match` | 합친 점수 1 − (1 − p_name)(1 − p_query)(1 − p_img) |
| `ask_user`(맨 위) | 있으면 **행동 전에 묻거나 가까이 보고 확인**: 생김새로만 찾음 / 점수 비슷한 후보 둘이 1 m 넘게 떨어짐 / 1 위 match < 0.5 / 아무것도 없음 |
| `hint`(맨 위) | 다음에 할 일 한 줄(사라진 물체면 "was seen there but is gone now") |

틀린 인자·없는 id·없는 방은 `{"status":"error","message",…,"hint"}` 관찰값(예외로 루프를 죽이지 않음).
`confirm_object` 결과: `{"status":"ok","id":"O234","name":"radio","p_before":0.13,"p_after":0.88,"registered":"fire extinguisher"}`.

## 실행 경로

```
LLM ── tool_call ──▶ ObjectSearch::run_tool (src/lib.rs) ── 인자 검사 → view.json 바뀌었으면 다시 읽기(memview.rs)
                          │
                          ├─ sys::Index (src/sys.rs) ── C ABI ─▶ libsgclip_c.so: sgs_search_json / sgs_confirm   (벡터·이름 확률·속성)
                          │                                       └ 글 인코더(TensorRT, 0.8 ms), 대체 벡터용 영상 인코더
                          └─ format_search: + 방·기준물(landmark.rs)·상태·경과·거리·ask_user  → JSON 글
```

- 기억 보기는 plan.md 3.2 의 `Memory` trait(`src/memview.rs`) — 오프라인 구현 `ViewJson`(view.json). 실시간(`sm_snapshot`) 판은 같은 trait 를 구현하면 된다.
- 물체 벡터: A′(`objects/O<id>_views.f16`·`_emb.f16`)가 있으면 그것, 없으면 best view 사진을 영상 엔진으로 뽑아 `cache/objsearch/` 에 둔다(사진이 바뀐 것만 다시).
- 쓰는 파일(기억 폴더 안): `confirmations.jsonl`(확인 기록 = 원본, 다시 열면 다시 적용), `cache/objsearch/names.json`(이름 캐시, 다시 셀 수 있음), `cache/objsearch/O<id>_view.f16`.
- C ABI(`include/search_objects.h`, `src/ffi.rs`): `so_open(mem_dir)`, `so_call(h, tool, args_json)`, `so_definitions`, `so_close`.

## 경로

| 무엇 | 환경 변수 | 기본 |
|---|---|---|
| `libsgclip_c.so`(빌드 때) | `SGCLIP_LIB_DIR` | `~/sgclip_build` |
| 라벨 표 | `SGRT_LABELS` | `~/embed_work/labels/objects-v1` |
| 글 인코더(토크나이저·토큰 임베딩·엔진) | `SGC_TEXT_DIR` | `~/ovdet_models/x86_sm120/siglip2_b32` |
| 영상 엔진(대체 벡터) | `SGC_ENGINE` | `~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan`(없으면 대체 벡터 없이) |

## 쓰는 법

```bash
# 공용 색인(C++) — behavior-2026 서브모듈에서(엔진 만들기는 그쪽 README "만들기"·"글 인코더")
cmake -S ../../../behavior-2026/src/scene_graph/clip -B ~/sgclip_build && cmake --build ~/sgclip_build -j4 && ctest --test-dir ~/sgclip_build
cd src/agent/tools/search_objects
cargo test --release                     # 시험 5개(GPU 없이; 끝까지 시험은 라벨 표가 있어야)
cargo build --release --features llm
./target/release/search-objects schema
cp -r ~/datasets/sim_detcmp/A_fastsam/gt/memory /tmp/mem        # 기억 폴더에 캐시·확인 기록이 생기므로 복사본에서
./target/release/search-objects call /tmp/mem search_objects '{"query":"라디오"}' confirm_object '{"id":"O234","name":"radio","source":"user"}'
./target/release/search-objects demo /tmp/mem 라디오            # 각본(LLM 없이)
set -a; . ~/.config/behavior-2026/kau.env; set +a
./target/release/search-objects llm /tmp/mem "라디오 가져와"      # KAU Qwen 원형 루프(도구 두 개)
```

## 시험 (`cargo test --release`)

| 시험 | 확인 |
|---|---|
| `schemas_are_small_and_strict` | 이름·필수 인자·enum, 둘 합쳐 2.6 KB 아래 |
| `parse_tolerant_and_errors_fixable` | 문자열 인자·대소문자·"o12"·k 자르기, 고칠 수 있는 오류 문장 |
| `appearance_hit_asks_user_and_reads_compactly` | 생김새 후보 → `ask_user`, `p_query`·`p_registered`, 대안 거르기, 속성 + 크기, 방·기준물(id·이름·거리·높이), 관계말 칸 없음, 한 후보 700 B 아래 |
| `name_hit_no_question_filters_and_ties` | 비슷한 후보 둘(1 m 넘게 떨어짐) → 묻기, 방·state·near 거르기, 하나면 묻지 않음, 다 걸러지면 "nothing" |
| `end_to_end_radio_registered_as_fire_extinguisher` | 가짜 기억 폴더(A′ 형식 벡터) + 공용 색인: "라디오" → 소화기로 등록된 O27 생김새 후보 → 확인 → "radio" 는 이름으로·묻지 않음, `confirmations.jsonl` 에 질의까지, 오류 관찰값, view.json 바뀌면 다시 읽음 |

## 측정 (10-05)

공용 색인 평가(behavior-2026 `clip/tools/eval_objsearch.py`, 자세한 표는 [clip README "물체 찾기"](../../../behavior-2026/src/scene_graph/clip/README.md)):
BEHAVIOR 집 LIMO 탐사 기억(FastSAM + SigLIP 2, 283 물체, A′ 전이라 best view 사진 한 장), 정답 종류 질의 18 개, 없는 물체 질의 40 개.

| | R@1 | R@5 | 이름으로 못 찾는 물체 R@5 | 없는 물체: 뭐라도 나옴 / 묻지 않고 행동 | 지연(물체 ≈ 300) |
|---|---|---|---|---|---|
| 이름만 (영어) | 0.11 | 0.22 | 0.00 | 0.05 / 0.03 | ≈ 100 µs |
| **이름 + 생김새 (영어)** | **0.39** | **0.56** | **0.33** | 0.15 / 0.03 | ≈ 100 µs (자유 글 ≈ 1 ms) |
| 이름 + 생김새 (한국어) | 0.28 | 0.44 | 0.28 | 0.15 / 0.03 | ≈ 0.1–1.1 ms |

- 없는 물체에 나온 후보는 대부분 생김새 후보라 `ask_user` 가 붙는다(헛찾음 → 되묻기). 묻지 않고 행동할 만큼(1 위 이름 후보, match ≥ 0.5)은 소파 위 "pillow" 로 등록된 쿠션 하나.
- 실제 OpenLORIS office1-1(298 물체, 정답 없음): 사무실에 없을 집 물건 10 개 질의에 0 개.
- 라디오 실제 사례(위 기억): 정답 라디오가 지도에서 4 조각("speaker"·"bag"·"appliance"·"kettle"). 이름만 0 개 → 생김새로 O234·O232 가 1·2 위, 작은 두 조각은 못 찾음.

## 대화 기록

각본(`search-objects demo`, LLM 없이 — 에이전트 말은 결과 칸으로 만든 틀 문장). 기억은 위 sim 기억의 복사본에서 라디오 조각 O234 의 등록 이름을
"fire extinguisher" 로 바꾼 것(사용자 시나리오 그대로). 실제 이름("speaker") 그대로 돌려도 같은 흐름이다.

```
  user │ 라디오 가져와
  call │ search_objects {"query":"라디오"}
result │ {"ask_user":"nothing is registered as '라디오'; O234 is registered as 'fire extinguisher' but looks like '라디오' (p 0.15 vs 0.06)", …
         "matches":[{"id":"O234","name":"fire extinguisher","name_p":0.17,"alt":[{"name":"radio","p":0.13},…],"match_type":"appearance",
         "p_query":0.15,"p_registered":0.06,"attrs":["red","small"],"room":"office","landmark":{"id":"O191","name":"sofa",…},…},
         {"id":"O232","name":"radio","registered":"bag",…}]}
 agent │ 라디오(으)로 등록된 건 없는데, office의 sofa 근처에 'fire extinguisher'(으)로 등록된 O234가 라디오일 수 있어요(모습 확률 0.15). 가져올까요?
  user │ 응, 그거 라디오 맞아
  call │ confirm_object {"id":"O234","name":"라디오","source":"user"}
result │ {"id":"O234","name":"radio","p_after":0.884,"p_before":0.132,"registered":"fire extinguisher","status":"ok"}
 agent │ O234 를 라디오(으)로 기억할게요. 가져올게요.
  (VLA) │ {"executor":"vla","max_s":30,"objects":["O234"],"skill":"pick up radio"}
  user │ 라디오 어디 있어?
  call │ search_objects {"query":"라디오"}
result │ {"hint":"O234 was seen there but is gone now","matches":[{"id":"O234","name":"radio","name_p":0.88,"match_type":"name",
         "registered":"fire extinguisher","match":0.9,…}],"searched":"name"}
```

`confirmations.jsonl` 한 줄: `{"id":234,"name":"라디오","label":"radio","synset":"radio_receiver.n.01","source":"user","lr":50.0,"query":"라디오",
"registered":"fire extinguisher","name_before":"fire extinguisher","p_before":0.132,"p_after":0.884,"p_app":0.149,"vec":"encoded","nv":1,"table":"8f2d638bae436dc6",…}`

실제 KAU Qwen3.5-9B 루프 한 번(`search-objects llm`, 등록 이름 그대로인 기억): 첫 턴 2.1 s 에 `search_objects {"query":"라디오"}` 를 부르고,
둘째 턴 3.5 s 에 도구 호출 없이 되물음 — *"O234는 'speaker'로 등록되어 있지만 라디오처럼 보입니다. O232도 라디오처럼 보이지만 'bag'로 등록되어 있습니다. 어떤 것이 라디오인가요?"*

## 한계 · 남은 일

- 대체 벡터는 best view 사진 한 장. A′ 상위 5 시점 벡터·이름 사후(view.json 칸 형식 미정)가 들어오면 색인이 저절로 그쪽을 쓴다 — 그때 다시 잴 것.
- 우도비(등록 3, user 50, close_look 10)와 문턱은 손으로 정했다. `confirmations.jsonl` 이 쌓이면 맞춘다.
- 한국어 질의는 라벨 표의 한국어 이름에 기계 번역이 섞여 영어보다 낮다 — 도구 설명에 "영어 낱말이 가장 낫다" 고 적었다.
- 실시간 기억(`sm_snapshot`)용 `Memory` 구현, `close_look` 을 `check` 결과에서 자동으로 부르는 것(plan.md 3.3 도구 수 9 개 문제)은 아직.
