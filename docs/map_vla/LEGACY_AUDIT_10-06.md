# 잔재(legacy) 점검 보고서 — 2026-10-06

읽기 전용 점검(코드·문서 수정 없음, 이 파일만 추가). 기준 = 10-05/06 결정 8 가지. 대상 = `~/robot-agent`, 서브모듈 `src/behavior-2026`(커밋 `75ecd76`).
"확인 필요" = 코드로 끝까지 확인하지 못한 항목. 줄 번호는 점검 시점 기준.

## 요약 표 (영향 큰 순)

| # | 영향 | 위반한 결정 | 한 줄 요약 | 위치 |
|---|---|---|---|---|
| 1 | 높음 | 8 | 학습 이름·지시 벡터는 PE-Core-L 글 탑 + 투영 P 공간. 실제 파이프라인(SigLIP 2 B/32 글)과 다른 인코더이고, 실제 로봇 쪽에는 이 벡터를 만드는 코드가 없음 | `training/embed/vla_tables.py:5-10`, `training/data/vla_v1/manifest.json:6,35` |
| 2 | 높음 | 3, 8 | BEHAVIOR 판의 생김새 벡터가 모든 물체에서 "상자 한 행" 상수. 실제 변환기는 생김새를 아예 안 채우고(NCLS), 이름 확신도 의미도 다름 | `training/RL/map/include/map_tok.h:668-677`, `sm_tok.h:35,188-191` |
| 3 | 높음 | 6 | 실시간 탐사 진입점이 기본 로봇 R1 Pro 로 돈다 | `tools/run_explore_live.sh`(SGRT_ROBOT 없음), `src/behavior-2026/src/sim/explore/run_explore.sh:20` |
| 4 | 높음 | 4 | 배포된 모든 PPO 설정이 C0/C1(GT 지도)을 요구하고, 뷰어 기록 도구는 기본이 GT 지도 80 % | `training/RL/config/*.json`, `training/viewer/tools/record_replay/record_ppo.cu:37`, `record_bc.cu:62,86` |
| 5 | 높음 | 5, 3 | RL 교사 입력에 참값(360° 벽 광선 16, 확정 목표의 참 자세)이 들어감. 대본 교사는 참 점유 격자로 길을 계획하고 B4/B5 는 지도 확정과 무관하게 참 자세 사용 | `training/RL/observation/include/obs.h:20-27,274`, `env/include/teacher.h:1-2,22,602,1409` |
| 6 | 높음 | 4 | 지도·환경이 판마다 옮겨지는 고정 12.8 m 창(시작·목표를 덮게 배치). 집 전체·경계 없음이 아님 | `training/RL/env/include/bscene.h:4-5,20-21`, `map/include/map.h:85` |
| 7 | 중간 | 1 | GPU 지도 잡음 상수가 COCO-80 검출기·R1 기록으로 잰 것. ObjectSAM + SigLIP 2 + objprob 로 잰 값 아님 | `training/RL/map/include/map.h:168-175`, `map_calib/limo/README.md:55,80,125` |
| 8 | 중간 | 3 | 정책 입력 옛 규칙(16 칸·가까운 순, 안 본 곳 광선, 교사 전용 격자)이 BC/PPO 쪽 전부에 남음(다른 에이전트 담당, 위치만) | 9절 |
| 9 | 중간 | 4 | PPO 실행기는 `scenes` 를 안 쓰면 평가 집까지 학습에 씀. 집 나누기 강제는 BC 쪽에만 | `training/RL/ppo/driver/src/main.rs:251-253` |
| 10 | 중간 | 2 | 실제 런타임이 레포 밖 `~/embed_work/labels/objects-v1` 에 의존. 헤드 `h` 가중치도 레포 밖 | `runtime/src/sgrt.cpp:233`, `sgrt_clip.cpp:53`, `training/BC/tools/app_table.cu:152` |
| 11 | 중간 | 8 | 실제 objprob 이름 24 개가 학습 이름 표(`vla_v1`)에 행이 없음 | 6절 |
| 12 | 낮음 | 6 | GPU 로봇 모델 헤더가 현재 URDF 보다 낡음(렌즈·잡는 점 관절 없음) | `training/RL/env/include/limo_omx_model.h` |
| 13 | 낮음 | 7 | `search_objects` 의 `near` 입력(물체 기준 1.5 m)은 물체 사이 근접 관계 | `src/agent/tools/search_objects/src/lib.rs:70,168` |
| 14 | 낮음 | 7 | 스킬 폴더에 코드(`mock_eval.sh`) | `src/agent/skills/explore/mock_eval.sh` |
| 15 | 낮음 | 1,6 | 옛 설명·옛 경로(YOLOE, R1 기본, `SGRT_OBJPROB=0` 옛 규칙) 코드 주석·분기 잔존 | 10절 |

---

## 1. 이름·지시 벡터가 다른 인코더(PE-L)이고 실제 쪽에 없음 — 높음 (결정 8)

현재 동작:
- 학습 이름 벡터 = `l2(P · PE-Core-L/14-336 글 탑(이름, 템플릿 4))`. 지시 벡터(영어) = PE-L 글(템플릿 없음) + P, 한국어 = 한국어 학생(`ko_small_v2`) + P.
- 생김새 벡터 = SigLIP 2 B/32 풀링 768 → 머리 `h`(`sb32_pe_300k`, PE-L 영상 + P 를 따라가게 학습) → 128.

근거:
```
training/embed/vla_tables.py:5   The 128-d space is the embed contract: P = frozen linear PE-Core L/14 text/image (1024) -> 128 ...
training/embed/vla_tables.py:6   Names: l2(P · PE-L text(name, 4 templates)) — taken from objects-v1/text128_sb32_pe_300k.f16
training/data/vla_v1/manifest.json:6   "space": "embed P (runs/sb32_pe_300k/head.pt, PE-Core-L-14-336 1024 -> 128, frozen)"
training/data/vla_v1/manifest.json:35  "en_model": "PE-Core-L-14-336 text, no template"
```
- 실제 파이프라인(sgclip)의 글 쪽은 `text_siglip2_b32.f16`(SigLIP 2 B/32 글 768)이다(`src/scene_graph/clip/include/sgclip.h:10`, `labels.cpp:3`). 레포 `src/` 어디에도 PE-L 글 탑, 투영 P, 머리 `h`, 한국어 학생을 돌리는 코드가 없다. `src/vla/README.md` 는 "실행 코드 없음". `sm_tok.h` 는 학습 폴더에 있고 이름은 문자열 일치로 표 행 번호만 얻는다(`sm_tok.h:270`).
- `docs/map_vla/MAPVLA_SPEC.md:253`(이름·생김새 = PE-L + P 128-d)와 `:522`(검색 질의 q = SigLIP 2 글 공간 768)가 서로 다른 공간을 가리킨다. `VLA_INPUT.md:119` 는 "SigLIP 2 글 인코더 → 128 투영" 이라 적었지만 구현은 PE-L(`VLA_INPUT.md:129` 메모에 있음). `TRAINING_DESIGN.md` 1절 "지시 벡터 … 실행기가 같은 글 인코더로" 는 실행기 코드가 없어 지금은 불가능.
- 학습 안에서도 공간이 둘이다. BC 초기 경로 지시 벡터 = SigLIP 2 글 768(`training/BC/README.md:241`, `training/BC/data/instr_a2.f32`), RL `instr128` = PE-L + P(`training/data/pnp_v1/manifest.json:89`).
- 머리 `h` 는 LVIS crop 과 FastSAM crop 으로 학습했다(`training/embed/README.md` 결과 절, `build_sim_crops.py:8` 은 FastSAM-s-416 엔진 사용). 지금 검출기(ObjectSAM) crop 으로는 다시 안 맞춤(확인 필요).

영향: 이름·생김새·지시가 학습에서는 PE-L 계열 공간, 실제 로봇은 SigLIP 2 계열이라 실제 투입 시 같은 의미 비교가 깨진다. 이름 흔들기/상위어 규칙도 이 표 기준이다.
고침(한 줄): 128-d 공간을 SigLIP 2 B/32 글·영상 탑에서 직접 투영(P 의 선생을 SigLIP 2 로 교체)하거나, 실행기에 PE-L 글 탑 + P + `h` 를 넣고 둘 중 하나로 문서·표를 통일.

## 2. BEHAVIOR 판 생김새 상수 + 실제 변환기 불일치 — 높음 (결정 3, 8)

- `map_tok.h:668-677`(BEHAVIOR 분기): `app = S.src >= 0 ? C_ITEM : NCLS`. 모든 진짜 물체의 생김새 행이 "item 상자" 한 행, 유령만 NCLS. `app128.f16` 은 7 행(1,792 B)뿐.
  ```
  // BEHAVIOR: 종류 = 이름 표 행. 생김새 = 우리 렌더의 상자(행 1, 가정 — 종류별 생김새 행은 아직 없음), 유령 = NCLS
  const int app = S.src >= 0 ? (int)C_ITEM : NCLS;
  const float cf1 = ss.conf1[app * ss.nname + S.cls], ...
  ```
  따라서 생김새 128 칸은 사실상 상수이고, 이름 확신도(conf1/conf2)는 (상수 생김새 × 참 이름) 표 값이다. 정책이 생김새를 쓸 이유가 없어진다("이름이 틀려도 생김새로 찾는다" 학습이 안 됨).
- 실제 변환기 `sm_tok.h`: `SmTokIn::app_id` 기본 −1(`:35`), `sm_tok_from_snapshot` 은 `app_id` 를 한 번도 안 채움(`:262-278`) → 전부 NCLS. 확신도는 `f2h_soft(S.score)`, `2·score−1`(`:188-189`) — 학습의 코사인 기반 conf1/conf2 와 의미가 다름.
- `sm_tok.h:15` `front[8]`=0, `comp`=0, `T_UNC`=0 도 학습과 다름(주석에 "실제 로봇에는 출처 없음"으로 적혀 있음).

고침: 실제 쪽은 물체 `_emb.f16`(768) → 머리 `h` → 128 로 채우고 conf 를 같은 정의(코사인)로 계산. 학습 쪽은 BEHAVIOR 물체의 생김새를 종류별 실제 crop 임베딩(또는 학습 때 지도가 갖는 SigLIP 임베딩 모사)으로 대체.

## 3. 실시간 탐사 진입점이 R1 Pro 로 돈다 — 높음 (결정 6)

- `tools/run_explore_live.sh` 는 `src/sim/explore/run_explore.sh` 를 `SGRT_ROBOT` 설정 없이 호출한다(`grep SGRT_ROBOT tools/` = 없음).
- `run_explore.sh:20`: `ROBOT=${SGRT_ROBOT:-r1pro}` → 기본 R1 Pro(R1 평가기, proprio 61, `zed_link` 카메라). LIMO 는 `SGRT_ROBOT=limo_omx` 를 직접 줘야 켜짐.
- 기본값 뿌리: `scenemap.h:45`(R1 Pro 기본), `sgrt.cpp:379-382`(없으면 R1, `sm_create(NULL)`), `sgrt.h:120`, `sgrt_glue.py:131`.
- 뷰어 재생(`training/viewer/.../og_replay_lib.py:312`)과 `record_replay/sg_feed.h:97`, `map_cmp` 는 `limo_omx` 로 명시해 안전.

영향: 메모리 기준 "viewer = run_explore_live.sh" 를 사용자가 쓰면 R1 Pro 기록이 학습 뷰어 "REAL" 판에 섞일 수 있음(실제 실행 때 환경변수는 확인 필요).
고침: `run_explore_live.sh` 에서 `export SGRT_ROBOT=${SGRT_ROBOT:-limo_omx}`, 장기적으로 `run_explore.sh`·sgrt 기본값을 `limo_omx` 로.

## 4. GT 지도(C0/C1) 설정·기본값 잔존 — 높음 (결정 4)

- 드라이버(`BC/driver/src/main.rs:348-352`, `RL/ppo/driver/src/main.rs:504-506`)는 `--debug-gt-map` 없이 `map_p0/p1>0` 이면 종료(코드 2). 그런데 배포된 설정은 전부 막힘 대상:
  `ppo_g4.json`(map [1,0],[1,0],[0.2,0.8],[0.1,0.1] — 줄 46·56·66·76), `ppo_g4_notok.json`(같음), `ppo_a2_ft20/40.json`(줄 46, [0.1,0.1]), `ppo_pnp.json`(B4·B5·B6C0 는 [1,0], B6C2·B1-6 는 [0.1,0.1] — 줄 75-163). `[0.1,0.1]` 도 C0/C1 20 %.
- 뷰어 기록 도구에는 가드가 없다: `record_ppo.cu:37` `float p0 = 0.2f, p1 = 0.6f;`, `record_bc.cu:62` 같은 기본. 학습 설정에 `map_p0` 가 없으면 학습 뷰어 재생은 80 % 에서 GT 로 채운 지도로 굴러감.
- `ppo_verify.cu:61`, `bc_verify.cu:53` 의 검증 설정도 C0/C1 포함(검증용이라 낮음).
- `training/BC/config/` 는 비어 있는데 `BC/README.md` 는 `config/bc_a2*.json` 을 가리킴(설정 없음).

고침: 설정의 `map` 을 `[0,0]` 으로 바꾸고, record 도구 기본을 0 으로 하며 `--debug-gt-map` 가드를 넣는다.

## 5. 교사 입력·계획의 참값 사용 — 높음 (결정 5)

(a) RL 교사 입력(`X0`)에 학생에 없는 참값이 있음:
- 16 방향 벽 광선: `env.h:344 wall_rays(const Core&…)` — 방 벽 참값, 360°, 가림·시야각 무관. `obs.h:274` 에서 `use_map` 과 무관하게 `G1 관측 56–71` 로 X0 에 들어감. 학생 토큰(arch 1)은 이 값을 안 쓰고 지도 벽 56 만 씀.
- 목표 자세: `goal_mode 1` 이라도 "확정된" 목표에는 **참값**을 넣는다(`obs.h:20-27`: "이미 본 물체의 정확한 자세 = 교사 특권", G1 53–55·72–79). 실제 로봇은 지도 추정 위치(잡음 있음)라 교사·학생(점 외 물체 목표) 모두 참값 입력과 실제 값 사이에 차이가 있음.
- 가치 머리는 가린 입력만 본다고 문서화돼 있어(`observation/README.md:35`) 비대칭 평론가는 아님. 위 두 가지는 액터 입력이라 위반.

(b) 대본 교사(`env/include/teacher.h`):
- 길 계획이 지도가 아니라 **특권 점유**: `:22`("창 128×128 칸 특권 점유"), `:602 tch_occ_full`. 믿음 지도로 계획하지 않음.
- `:1409 target_known`: `kind != EK_B6 || fb.conf == nullptr` 이면 `true` → B4/B5 는 지도 확정과 무관하게 참 물체 자세를 씀(`:2` 주석은 "지도에 있는 물체만"이라 되어 있지만 코드는 B6 만 막음). 떨어뜨림 때도 "물체 참 자리로 다시 잡기".

고침: 광선 16·참 목표 값을 X0 에서 빼거나 지도 값으로 대체. 교사 계획 점유를 믿음 지도(`MapTok`/occ)에서 만들고 `target_known` 을 B4/B5 에도 적용.

## 6. 고정 12.8 m 창 — 높음 (결정 4, 문서에도 인정된 상태)

- `bscene.h:4-5`: 판은 "창 좌표"로 돌고, 창은 판마다 **시작·목표를 덮도록** 옮겨진다(목표 정보로 창 위치 결정). `bscene.h:20-21` `WIN=128`, `WIN_HALF=6.4`; `map.h:85` `GW=128`(고정 창). 환경(`MAXBR=16` 방, `MAXBD=16` 문 상한)·교사 탐사 격자 1.6 m(`teacher.h:91`)도 창 안.
- `CURRICULUM_BEHAVIOR2026.md:719` 에 "1·3 단계는 집 전체가 필요 → 설계 …" 로 이미 기록. 아직 구현 없음. 집이 12.8 m 보다 크거나 창 밖 물체·벽이 있으면 학습 분포에 없음.

## 7. 잡음 보정이 옛 검출기·로봇 기준 — 중간 (결정 1)

- `map.h:167-175`: 놓침 p_miss 계단 0.15/0.10/0.79 = "R1 시뮬 기록 보정"(주석에 "LIMO 기록으로는 맞추지 못해 그대로"), `ext_n 0.05` R1 기록, `p_conf`(틀린 이름)·`p_ghost`·`lat_n` = LIMO 기록이지만 검출기는 **COCO-80**(limo3, 검출 133, 확정 33 개; `map_calib/limo/README.md:55,80,125-132`). 열린 어휘 쪽은 확정 16 개로만 맞춤(`:136`).
- 현재 결정(ObjectSAM + SigLIP 2 + objprob)으로 다시 잰 값 없음. objprob 의 이름 모델(posterior·상위어)이 아니라 `name_share/name_switch` 규칙(`map.h:139-142`)과 `p_conf` 로 흉내냄.
- `map_cmp` 는 잡음 끔 + 완벽한 검출 + 기본 scenemap(objprob 앞단 설정 없음)이라 objprob 대 근사 비교가 아님(`map_cmp/README.md`).

고침: ObjectSAM + SigLIP 2 + objprob 기록(radio r3·limo 기록)으로 `map_drift`/`map_cmp` 재보정(GPU 근사 재이식 에이전트 담당과 합침).

## 8. (참고) 옛 정책 입력 규칙이 남은 곳 — 다른 에이전트 담당, 위치만

| 옛 규칙 | 위치 |
|---|---|
| 지도 물체 칸 16 + 가까운 순 + 목표 항상 | `training/RL/network/include/net.h:29`(`KSLOT=16`), `net.h:35`(`SLOT_IN 304`), `map_tok.h:63-65,222-226`(`slot[KSLOT]`, `sk`·`tk`), BC `include/tf.h`, `BC/src/bc.cu:85,379,496`, `ppo/src/trainer.cu:51,491,697-742` |
| 안 본 곳 광선 8(탐사 경계 옛 안) | `map_tok.h:20,71`(`N_FRONT`, `front[]`), `observation/include/obs.h:78,93,291`, `net.h:47`(`X0_FRONT`), `sm_tok.h:15`(0 으로 둠), `observation/tools/tok_stats.cu:60` |
| 교사 전용 격자(16×16×2, X0 [464,976)) | `map/include/topview.h`, `MapTok::tv`(`VLA_INPUT.md:41`) |
| 교사 탐사(창 1.6 m 격자점) | `env/include/teacher.h:28,91,1449,1638` |
| RecallVLA 기억 경로는 `mem=false` 기본, 데이터 생산자 없음 | `training/vla/include/model.h:23`(`{16,304,289}` OBJ 묶음), `:52-57`; BC/PPO 쪽에 `mem_n`·방 항목 8·방향 구역 8 을 채우는 코드 없음(grep 0) |
| 방 토큰 1(방 종류 한 칸 1) | `VLA_INPUT.md:151`, `sm_tok.h:199`(실제는 방 분류기 확률 자리) — 시뮬이 참 방 종류를 쓰는지 확인 필요 |
| GPU 지도의 objmap 옛 이름 규칙 | `map.h:139-142`(`name_share`, `name_switch`, `name_merge_iou`) — objprob 구조가 아님 |

## 9. 집 나누기 강제가 BC 쪽에만 — 중간 (결정 4)

- BC: `BC/src/bc.cu:606-611` 가 이름 3 개(`Rs_int`, `hotel_suite_large`, `house_double_floor_upper`)를 평가로 빼고 기본 `house_split=1`(`BC/driver/src/main.rs:346`).
- PPO: `ppo/driver/src/main.rs:251-253` `bcurr_of` — `scenes` 가 비면 `all`(평가 집 포함). 학습 집 4 개는 `ppo_pnp.json:43-48` 에서만 명시. 다른 BEHAVIOR 단계 설정이 `scenes` 를 안 쓰면 평가 집을 학습에 사용.
- 평가 집 이름이 `bc.cu:608` 에 하드코딩, `runfolder.rs:76` 에 또 하나(두 곳 중복).

고침: PPO 드라이버에도 같은 train/eval 이름 목록을 한 곳에서 공유하고 기본 `scenes` 를 학습 집으로.

## 10. 레포 밖 경로·낡은 산출물 — 중간~낮음 (결정 2)

- 실제 런타임 기본 `~/embed_work/labels/objects-v1`(`sgrt.cpp:233`, `sgrt_clip.cpp:53`, `clip/CMakeLists.txt:65`, `dom_bench_det.cpp:127`, `realbag_run.cpp:493`). 이 폴더(2026-10-03 생성, 65 MB)는 레포에 없고 만든 스크립트(`training/embed/*`)가 `~/embed_work` 에 의존. `app_table.cu:152`, `app_ref.py:19`, `export_head_f32.py` 도 `~/embed_work/runs/sb32_pe_300k/head*.pt` 필요 → 헤드 `h` 가중치가 레포·버전 밖.
- `tools/run_explore_live.sh:25` 기본 빌드 `~/sgrt_build_explore`(비공개 빌드 경로). 지금은 `CMAKE_HOME_DIRECTORY` 가 서브모듈 소스를 가리키고 `libsgrt.so`(10-05 20:39)가 최신 런타임 커밋(20:13) 이후라 최신으로 확인. 단 소스 일치를 스크립트가 검사하지 않는다(`build_deps.sh` 는 같은 폴더를 소스 일치 검사 후 링크함 — 이쪽은 안전).
- `training/RL/map/CMakeLists.txt:64` 는 `src/scene_graph/scenemap/include`(복사본), `:80` 는 `src/behavior-2026/.../scenemap`(원본). 현재 `tools/sync_scene_graph.sh --check` = 차이 없음이라 문제 없음, 동기가 깨지면 섞임.
- `src/robot/og/import_to_omnigibson.sh`(`URDF_IN` 기본 `$HOME/ra_ws/map_vla.urdf`), `og/e0/kin.py:16,191`, `finger_gap.py:3` 가 `data/ws/` 의 절대 경로 사용. 이 URDF 가 레포 xacro 와 같은지는 `limo_description` 패키지가 없어 `xacro` 로 재생성하지 못함 — 확인 필요(렌즈·`grasp_point` 는 들어 있음).
- `training/viewer/tools/scene_mesh/export_mesh.py:19` `~/datasets/fastsam_obj/sim`, `training/embed/build_sim_crops.py:8` `~/embed_work/ovdet_build`(자체 빌드, FastSAM-s-416 엔진).

## 11. 이름 표 어휘 불일치 — 중간 (결정 8)

objprob 앞단 `objprob_front.hpp`(kVocab 라벨 + kApLabels 이름)와 학습 이름 표 `training/data/vla_v1/names.jsonl`(en 기준 584 행)을 비교:
- 실제 objprob 라벨 63 개 중 **24 개가 학습 표에 행 없음**: appliance, baseboard, cable, ceiling, clock, clothing, curtain, fan, outdoors, paper, partition, phone, picture frame, pillar, pole, radiator, railing, rug, speaker, staircase, umbrella, wall, whiteboard, window.
- `sm_tok_from_snapshot`(`sm_tok.h:269-270`)는 `labels` 표(호출자가 줌)와 문자열이 일치할 때만 `name_id` 를 정함 → 위 이름의 실제 물체는 이름 벡터가 비는(−1) 쪽으로 가거나 상위어 처리됨(`labels` 를 호출자가 어떻게 만드는지는 확인 필요). 구조물(wall 등)은 노드가 안 되니 영향 작음; clock, phone, fan, speaker, umbrella, curtain, rug, picture frame, whiteboard, paper 는 옮길 수 있는 물건·가구라 영향 있음.

## 12. GPU 로봇 모델 헤더가 현재 URDF 보다 낡음 — 낮음 (결정 6)

- `training/RL/env/include/limo_omx_model.h`(10-04 09:24, 관절 21 개). 현재 `data/ws/map_vla.urdf` 로 `urdf2hdr` 를 다시 돌리면 관절 24 개: `depth_camera_lens_joint`(+x 0.010), `depth_camera_lens_optical_joint`, `grasp_point_joint`(0.08003, −0.0016) 가 더해지고 이후 관절 번호(`J_OMX_MOUNT_JOINT` 17→19 등)가 바뀜.
- 영향은 낮음: 카메라 x 는 `env.h:35 cam_x = 0.094` 로 따로 하드코딩되어 이미 렌즈 위치이고, 잡는 점은 `map.h grasp_off -0.0119`. 다만 이 두 값이 헤더에서 파생되지 않아 URDF 가 바뀌면 조용히 어긋남.
- 고침: `urdf2hdr` 재실행 후 `cam_x`·`grasp_off` 를 헤더에서 읽게.

## 13. 에이전트: 물체 사이 관계, 스킬 코드 — 낮음 (결정 7)

- `search_objects` 인자 `near`(물체 id 지정 → 그 물체 1.5 m 안)이 입력으로 남음(`lib.rs:70,168`, `src/agent/tools/README.md:28`, `memview.rs:58 near()`). 결과 필드에는 관계 단어가 없다는 테스트가 있음(`tests.rs:110,214`) 하지만 입력은 물체 간 근접 관계.
- `move_robot/src/goal.rs:363,446` 은 목표 점의 받침("on": floor/surface, support)을 냄 — 점–물체 관계라 물체 간 관계는 아님(참고).
- `scenemap/README.md:121`·`CMakeLists.txt:87` `test_relations` 는 "물체끼리 on/in/near 변이 없음"을 검사하는 음성 시험(정상).
- `src/agent/skills/explore/mock_eval.sh` — 스킬 폴더 안 스크립트(가짜 집 + 정답 바닥 평가). 결정은 "스킬 = 프롬프트만, 코드는 tools/·runtime/"; `skill.json` 의 metrics·baseline 도 설정 성격이라 경계. 옮기는 편이 맞음.

## 14. 보관된 것·옛 결정이 코드에 남은 곳 — 낮음 (결정 1, 6, 7)

- 주석·헤더: `sgrt.h:4,23,35`(YOLOE), `scenemap.h:20,89`(검출기 YOLOE), `ovdet/include/ovdet.h:4-18`, `scenemap_stub`·`simlink`(YOLOE), `sgrt_glue.py:24`("R1 Pro (default…)"), `objprob_front.hpp:29`.
- 분기: `SGRT_OBJPROB=0` = 옛 이름 규칙(`sgrt.cpp:222`, `sgrt_glue.py:10`), `FastSAM-s-obj` 매개변수 파일(`objprob_params/FastSAM-s-416-obj.*`), `training/fastsam/eval_det.py:3-4`·`e2e.sh:4`(plain 옛 규칙).
- R1 경로: `run_explore.sh` R1 분기(`MOVE_ROBOT_FOOTPRINT` 설명), `simlink_policy.py:397`(`CAMERA_INTRINSICS["R1Pro"]`), `sim/configs/r1pro_robot.yaml`, `sgrt_glue.py:49-50`. π0.5 는 이번 삭제 커밋(`a56aef0`)에서 코드가 빠졌고 주석(`instruction.rs:3`, `plan.rs:3`)만 남음(문제 없음).
- `training/embed/common.py:17` MobileCLIP2 항목은 선생 비교용 표(보관 대상, 영향 없음).
- `training/RL/tools/b1kconv`(`task.rs`) 의 R1Pro 는 BEHAVIOR 시연 데이터 읽기용(학습 로봇 아님, 영향 없음).

---

## 점검에서 이상 없던 것 (비교용)

- 검출 기본: `sgrt_glue.py:47`, `objprob_front.hpp:27`, `run_explore.sh:48`, `run_limo_map.sh:30`, `dom_bench_det.cpp:125` 모두 `yolo26n-seg-obj-416.plan`, objprob 는 클래스 없는 엔진이면 자동 켬(`sgrt.cpp:220-230`).
- 단일 원본: `tools/sync_scene_graph.sh --check` 차이 없음(서브모듈 `75ecd76`).
- 학습 뷰어 재생: `training/viewer/build_deps.sh` 가 libsgrt·sgview·sgs_play 를 레포 소스에서 빌드, `og_replay_lib.py:312` `SGRT_ROBOT=limo_omx`, URDF 기반 OG 로봇.
- GT 지도 가드: 두 드라이버에 `--debug-gt-map` 가드 존재(단, 설정이 이를 위반 — 4절).
- 위에서 본 지도 그림·몸 56·목표 2 칸·지시 128 은 구현이 문서와 일치(`obs.h`, `map_tok.h goal_fill`).
