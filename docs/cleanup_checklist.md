# 잔재 체크리스트

`python3 tools/audit.py` 가 만든다(폴더마다 다섯 검사). 0 이면 ✅. 허용 예외는 tools/audit.py ALLOW.

| 폴더 | 결정 키워드(코드) | 결정 키워드(문서) | 저장소 밖 경로 | 깨진 링크 | 안 부르는 파일 | 상태 |
|---|---|---|---|---|---|---|
| `(뿌리)` | · | 3 | · | · | · | ⬜ |
| `config` | · | · | · | · | · | ✅ |
| `docs` | · | · | · | · | · | ✅ |
| `scripts` | · | · | · | · | · | ✅ |
| `src` | · | · | · | · | · | ✅ |
| `src/agent` | · | · | · | · | · | ✅ |
| `src/agent/devtools` | · | · | · | · | · | ✅ |
| `src/agent/planner` | · | · | · | · | · | ✅ |
| `src/agent/prompts` | · | · | · | · | · | ✅ |
| `src/agent/runtime` | · | · | · | · | · | ✅ |
| `src/agent/skills` | · | · | · | · | · | ✅ |
| `src/agent/tools` | · | · | · | · | · | ✅ |
| `src/app` | · | · | · | · | · | ✅ |
| `src/robot` | · | · | · | · | · | ✅ |
| `src/scene_graph` | · | · | · | · | · | ✅ |
| `src/scene_graph/clip` | · | · | · | · | · | ✅ |
| `src/scene_graph/common` | · | · | · | · | · | ✅ |
| `src/scene_graph/da` | 2 | 1 | · | · | · | ⬜ |
| `src/scene_graph/ovdet` | · | · | · | · | · | ✅ |
| `src/scene_graph/runtime` | 8 | 5 | · | · | · | ⬜ |
| `src/scene_graph/scenemap` | 8 | 11 | · | · | · | ⬜ |
| `src/scene_graph/sgview` | · | · | · | · | · | ✅ |
| `src/scene_graph/slam_carto` | · | · | · | · | · | ✅ |
| `src/scene_graph/tools` | 1 | 2 | · | · | · | ⬜ |
| `src/sim` | · | · | · | · | · | ✅ |
| `src/vla` | · | · | · | · | · | ✅ |
| `tests` | · | · | · | · | · | ✅ |
| `tools` | · | · | · | · | · | ✅ |
| `training/BC` | · | · | · | · | · | ✅ |
| `training/README.md` | · | · | · | · | · | ✅ |
| `training/RL` | · | · | · | · | · | ✅ |
| `training/RL/config` | · | 1 | · | · | · | ⬜ |
| `training/RL/env` | 1 | · | · | · | · | ⬜ |
| `training/RL/map` | 3 | 6 | · | · | · | ⬜ |
| `training/RL/map_calib` | 2 | 5 | · | · | · | ⬜ |
| `training/RL/map_cmp` | 3 | 1 | · | · | · | ⬜ |
| `training/RL/network` | · | · | · | · | · | ✅ |
| `training/RL/observation` | · | · | · | · | · | ✅ |
| `training/RL/ppo` | · | · | · | · | · | ✅ |
| `training/RL/reward` | · | · | · | · | · | ✅ |
| `training/RL/tools` | · | 1 | · | · | · | ⬜ |
| `training/data` | · | · | · | · | · | ✅ |
| `training/embed` | · | · | · | · | · | ✅ |
| `training/fastsam` | · | · | · | · | · | ✅ |
| `training/model` | · | · | · | · | · | ✅ |
| `training/render_engine` | · | · | · | · | · | ✅ |
| `training/viewer` | · | · | · | · | · | ✅ |
| `training/vla` | · | · | · | · | · | ✅ |

## 남은 항목

### `(뿌리)`
- [ ] 결정 키워드(문서): CLAUDE.md:4: - 작업을 시작하기 전에: `git pull --rebase --autostash` 와 `git submodule update --init src/behavior-2026`.
- [ ] 결정 키워드(문서): CLAUDE.md:6: - 서브모듈(`src/behavior-2026`)을 고쳤으면 서브모듈 안에서 먼저 커밋·푸시하고(그 저장소의 `CLAUDE.md`), 상위 저장소에서 서브모듈 포인터를 커밋·푸시한다.
- [ ] 결정 키워드(문서): CLAUDE.md:13: - 서브모듈(`src/behavior-2026`)도 같은 규칙이다.

### `src/scene_graph/da`
- [ ] 결정 키워드(코드): src/scene_graph/da/include/da/merge.hpp:15: // drop 을 keep 에 합친다(mergeDuplicates 와 같은 규칙: 위치·상자·관측 수·이름 표·점 구름). drop 은 호출자가 지운다
- [ ] 결정 키워드(코드): src/scene_graph/da/src/merge.cpp:62: if (op.name_vote) {
- [ ] 결정 키워드(문서): src/scene_graph/da/README.md:4: 하나로 합치는 `absorbObject` 와 상자 겹침 비율 `boxOverlap`. 같은 것 판정은 여기 없다(objprob 하나). 옛 이름 기반 `mergeDuplicates`·`MergePa

### `src/scene_graph/runtime`
- [ ] 결정 키워드(코드): src/scene_graph/runtime/glue/sgrt_glue.py:17: "slam" is the same), odom, gt. The old scenemap depth scan matcher (slam2d) is archived (archive/src/scene_gra
- [ ] 결정 키워드(코드): src/scene_graph/runtime/include/sgrt.h:189: * keyframe 마다 검출 마스크마다 SigLIP 2 임베딩(SGRT_OBJPROB_CLIP, 기본 siglip2_b32_mask_fp16.plan) → 낱말 표(SGRT_LABELS) 이름,
- [ ] 결정 키워드(코드): src/scene_graph/runtime/include/sgrt.h:191: * 매개변수 SGRT_OBJPROB_PARAMS = 파일 | none(내장), 없으면 tools/realbag/objprob_params/<엔진 줄기>.json. 과제 이름은 낱말 표에 더함.
- [ ] 결정 키워드(코드): src/scene_graph/runtime/src/sgrt.cpp:225: const char* cp = std::getenv("SGRT_OBJPROB_CLIP");
- [ ] 결정 키워드(코드): src/scene_graph/runtime/src/sgrt.cpp:236: // 엔진별 매개변수: SGRT_OBJPROB_PARAMS = 파일 | none, 없으면 <매개변수 폴더>/<엔진 줄기>.json
- [ ] 결정 키워드(코드): src/scene_graph/runtime/src/sgrt.cpp:237: const char* pe = std::getenv("SGRT_OBJPROB_PARAMS");
- [ ] 결정 키워드(코드): src/scene_graph/runtime/src/sgrt.cpp:741: if (mode == SM_POSE_SLAM) mode = SM_POSE_EXT;   // 옛 값 0 = Cartographer(호환)
- [ ] 결정 키워드(코드): src/scene_graph/runtime/src/sgrt.cpp:760: if (mode == SM_POSE_EXT || mode == SM_POSE_SLAM) {
- [ ] 결정 키워드(문서): src/scene_graph/runtime/README.md:82: **LIMO 시뮬 확인 — Cartographer**(10-06, turning_on_radio 인스턴스 0, `run_limo_map.sh … 2400`, `SGRT_POSE=carto`, 시뮬 
- [ ] 결정 키워드(문서): src/scene_graph/runtime/README.md:103: **LIMO 탐색 결과**(10-04, turning_on_radio 인스턴스 0 = house_double_floor_lower, headless, frontier, `SGRT_POSE=slam`
- [ ] 결정 키워드(문서): src/scene_graph/runtime/README.md:115: | `SGRT_POSE` | `carto` | 자세 원천. `carto`(Cartographer: 2D 라이다 + 바퀴 오도메트리, `../slam_carto` — `slam` 도 같은 뜻) · `
- [ ] 결정 키워드(문서): src/scene_graph/runtime/README.md:132: | `SGRT_OBJPROB_PARAMS` | `tools/realbag/objprob_params/<엔진 줄기>.json` | objprob 엔진별 매개변수 파일, `none` = 내장 기본값 |
- [ ] 결정 키워드(문서): src/scene_graph/runtime/README.md:133: | `SGRT_OBJPROB_CLIP` | `models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan` | objprob 마스크 임베딩 SigL

### `src/scene_graph/scenemap`
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/include/scenemap.h:323: * SM_POSE_SLAM(0): SM_POSE_EXT 와 같다(옛 호출자 호환).
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/include/scenemap.h:329: enum { SM_POSE_SLAM = 0, SM_POSE_ODOM = 1, SM_POSE_GT = 2, SM_POSE_EXT = 3 };
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/include/scenemap/objmap.hpp:89: bool name_vote = true;
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/src/capi.cpp:1671: if (!c || mode < SM_POSE_SLAM || mode > SM_POSE_EXT) return -1;
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/src/capi.cpp:1673: c->pose_mode = mode == SM_POSE_SLAM ? SM_POSE_EXT : mode;   // SM_POSE_SLAM(옛 값) = EXT
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/src/objmap.cpp:142: const K keys[] = {{"floor_h", &p->floor_h, nullptr, nullptr},          {"name_vote", nullptr, nullptr, &p->nam
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/src/objmap.cpp:222: if (!p_.name_vote) return;
- [ ] 결정 키워드(코드): src/scene_graph/scenemap/tests/test_posemap.cpp:208: sm_set_pose_mode(r.c, SM_POSE_SLAM);   // 옛 값 = EXT
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:22: | `mapper2d.*` | 자세는 밖에서(SM_POSE_EXT = Cartographer 기본·GT·ODOM): keyframe 사이·외부 자세가 없을 때 `base_qvel` 적분, keyfr
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:210: | | 확률 모드(FastSAM-s-416 + SigLIP 2) | 옛 규칙: FastSAM-s-416 + 이름 | 옛 규칙: YOLO26s-seg | 옛 규칙: YOLOE-11L |
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:244: - **새 분할 엔진 `FastSAM-s-416-obj`(things 만 다시 학습, 보정 t 0.05 = conf 0.25) 다시 맞추기**(10-05, `tools/realbag/objprob_
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:252: | 새 엔진 · 다시 맞춤 0.5/0.7(파일 `FastSAM-s-416-obj.json`) | 128 / 124 | 28 / 27 | 53 / 49 | 11 / 7 | 2·1·0 / 3·1·0 |
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:264: `--engine …/FastSAM-s-416-obj.plan` 이면 `FastSAM-s-416-obj.json` 이 자동으로 실린다. 실제(OpenLORIS)는 conf 0.44 쪽이 노드·같은 
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:272: | 앞 학생(v0) · FastSAM-s-416-obj 매개변수 빌림 | 100 / — | 26 / 26 | 32 / — | — | — | — |
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:277: - FastSAM-s-416-obj 0.5/0.7(128 / 124 노드, 찾음 28 / 27, OL 120 · 135)보다 노드는 적고, 찾음은 1 낮거나 같다.
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:280: 버린 FastSAM-s 재학습(보관 `models/ovdet/archive/x86_sm120/FastSAM-s-416-obj.plan`)은 `--engine`·`SGRT_ENGINE` 으로 고를 수
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:281: 위 "기본 엔진은 그대로"(FastSAM-s-416-obj 표)는 그때 판단이다.
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:403: slam2d(깊이 가상 스캔 맞추기 — LIMO 가중 고침 10-04, 시뮬 기록 4 개 표)는 `archive/src/scene_graph/scenemap`(slam2d.cpp·.hpp, slam
- [ ] 결정 키워드(문서): src/scene_graph/scenemap/README.md:404: stage_bench·capi_replay)로 옮겼고, 그 맞춤 기록은 git 이력(이 README 10-04 판)에 남는다. Cartographer 대 slam2d 비교는 `../slam_cart

### `src/scene_graph/tools`
- [ ] 결정 키워드(코드): src/scene_graph/tools/realbag/objprob_eval.py:142: objprob(물체 벡터 objects/O<id>_emb.f16 = μ, _views.f16): cos(μ, 글) 와 max(모습들, 글) 두 가지. 벡터가 없는 판(옛 규칙 FastSAM-s-41
- [ ] 결정 키워드(문서): src/scene_graph/tools/realbag/README.md:9: | `realbag_run.cpp` | 스트림 → 검출(`--det fastsam`(기본) = 이름 없는 분할 엔진 + SigLIP 2 이름 — 엔진 기본은 ObjectSAM `yolo26n-seg
- [ ] 결정 키워드(문서): src/scene_graph/tools/realbag/README.md:17: | `objprob_params/<엔진>.json` | 엔진별 objprob 매개변수: `obj_params`(`sm_set_obj_params` 문자열 — 로지스틱 `ap_w*`·`ap_wm*`,

### `training/RL/config`
- [ ] 결정 키워드(문서): training/RL/config/README.md:59: | `beh.yaw_jit` | 0.5 rad | **(가정)** B1 시작 yaw = 인스턴스 R1Pro 자세라 ±0.5 rad 흔듦(외우기 방지). 집기·놓기 판은 시작이 원래 무작위 |

### `training/RL/env`
- [ ] 결정 키워드(코드): training/RL/env/tools/env_verify.cu:10: // --arm: GPU·CPU 모두 팔을 풀고(arm_free) 행동 8 을 모두 무작위로(팔·그리퍼 경로, VLA_INPUT 5절).

### `training/RL/map`
- [ ] 결정 키워드(코드): training/RL/map/include/map.h:22: // 자세 오차는 Cartographer 흉내(drift_params.h, GPU_MAP_PORT 0.3). 옛 slam2d 맞춤 값(odo_t 0.015·kf_corr_xy 등, map_drift
- [ ] 결정 키워드(코드): training/RL/map/include/percept.h:77: // keyframe 카메라 방향·회전 빠르기(움직임 근거). 자세 고침은 Cartographer 흉내(map.h phase_begin, 스텝마다) — slam2d 의 깊이 열 맞추기 되돌림(kf_
- [ ] 결정 키워드(코드): training/RL/map/tools/map_drift.cpp:8: // kind 5: 옛 LIMO SLAM(slam2d, 보관) 기록 넷(data/datasets/limo_rec r1–r3·r4live: 25.0–37.9 m, 1,788–2,663°, 103–14
- [ ] 결정 키워드(문서): training/RL/map/README.md:64: 1. **시작(판마다 한 스레드)**: 환경 판 번호가 바뀌면 지도 리셋(격자·물체 비움, 장면 다시 만듦). 아니면 참 증분에 Cartographer 걸음 오차(`drift_params.h`)를 
- [ ] 결정 키워드(문서): training/RL/map/README.md:67: - (slam2d 의 keyframe 맞추기 되돌림 `kf_corr`·`min_hits` 는 없앰 — Cartographer 흉내는 시작 단계에서 끝남, GPU_MAP_PORT 0.3)
- [ ] 결정 키워드(문서): training/RL/map/README.md:91: ## scenemap 바뀜 규칙·LIMO 잡기 확인·새 SLAM 옮김 (behavior-2026 3ed710f·d58c978·84b373c, 2026-10-04, 잰 값)
- [ ] 결정 키워드(문서): training/RL/map/README.md:102: | 중복 병합(da mergeDuplicates, 근사판은 예전에 병합이 없었음): 확정·안 든·사라짐 아닌 쌍, 같은 이름 또는 3D IoU ≥ 0.5, 축별 겹침 곱 ≥ 0.35, 큰 쌍은 합집
- [ ] 결정 키워드(문서): training/RL/map/README.md:485: scenemap 기본값에서 가져온 값: 격자 Q·l_hit·l_miss·l_min/max, scan 의 높이 띠, slam2d 의 mf_xy·mf_yaw·mf_kf·still_v/w, objmap 
- [ ] 결정 키워드(문서): training/RL/map/README.md:499: - ~~**5.2 진짜 scenemap 과 맞추기(V2, `map_verify --check` 의 진짜 판)**: 막혀 있다.~~ **(했다)**: scenemap 에 LIMO + OMX 순기구학이

### `training/RL/map_calib`
- [ ] 결정 키워드(코드): training/RL/map_calib/limo/calib_limo.json:3: "robot": "limo_omx in BEHAVIOR/OmniGibson (sim). eyes camera 720x720 fx 306 at z 0.18 m, diff-drive, odometry 
- [ ] 결정 키워드(코드): training/RL/map_calib/limo/calib_limo.json:343: "_doc": "옛 LIMO SLAM(slam2d, 오도메트리 가중 맞추기 — 보관됨)으로 다시 맞춘 값(2026-10-04) — 지금 map.h 는 Cartographer 흉내(drift_para
- [ ] 결정 키워드(문서): training/RL/map_calib/README.md:28: ## 쓴 기록 (읽기만 함, `src/behavior-2026/outputs/`)
- [ ] 결정 키워드(문서): training/RL/map_calib/README.md:33: | `mem_pose_slam_20261003_065114` | SLAM | rec.bin, gt.csv. pi05 정책이 조작하는 판이다(베이스 0.55 m) |
- [ ] 결정 키워드(문서): training/RL/map_calib/README.md:42: **원 오도메트리는 따로 저장되지 않는다.** 그러나 rec.bin 에 매 스텝 proprio(R1 `base_qvel` = proprio[0:3])와 외부 GT 자세('G')가 들어 있다. 그래서
- [ ] 결정 키워드(문서): training/RL/map_calib/limo/README.md:11: ## 옛 LIMO SLAM(slam2d, 보관)으로 다시 맞춤 — 지금 `map.h` 는 Cartographer 흉내(`drift_params.h`)라 지난 기록 (2026-10-04, 잰 값)
- [ ] 결정 키워드(문서): training/RL/map_calib/limo/README.md:13: scenemap slam2d 가 LIMO 에서 오도메트리 가중 맞추기로 바뀌었다(점 σ 10 cm, 사전항 5 mm + 3 %·0.2° + 3 %, 받기 문턱 3 cm·1.5°). 옛 목표(아래 "

### `training/RL/map_cmp`
- [ ] 결정 키워드(코드): training/RL/map_cmp/tools/map_cmp.cpp:213: int N = 50, policy = 0, pose_mode = SM_POSE_SLAM, trace = -1, stage = 1, T = 2000;
- [ ] 결정 키워드(코드): training/RL/map_cmp/tools/map_cmp.cpp:217: if (!std::strcmp(argv[a], "--mode") && a + 1 < argc) { ++a; pose_mode = !std::strcmp(argv[a], "odom") ? SM_POS
- [ ] 결정 키워드(코드): training/RL/map_cmp/tools/map_cmp.cpp:228: N, stage, pose_mode == SM_POSE_SLAM ? "SLAM" : "ODOM", policy, W, H, fx, 2 * std::atan(0.5 * W / fx) * 180 / M
- [ ] 결정 키워드(문서): training/RL/map_cmp/README.md:3: GPU 지도 근사판(`../map`)과 진짜 scenemap(behavior-2026 서브모듈 `src/scene_graph/scenemap`, LIMO + OMX, CPU)에 **같은 입력**을 

### `training/RL/tools`
- [ ] 결정 키워드(문서): training/RL/tools/b1kconv/README.md:6: 입력은 평문 JSON·PNG·BDDL 뿐이다(`src/behavior-2026` 서브모듈, 읽기만). **암호화 USD 는 읽지도 풀지도 않는다.** Python 은 쓰지 않는다.

