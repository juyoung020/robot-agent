# 학습 뷰어 설계 — RL 교사·BC 학생·DAgger·RL 다듬기 (Rust 서버 + 브라우저)

작성 2026-10-03. 설계 문서. **구현 상태는 13절**(2026-10-04: V0–V4 와 V5 의 평가 표 — 코드 `training/viewer/`, 사용법 [training/viewer/README.md](../../training/viewer/README.md)).
- 무엇을 학습하나는 [POLICY.md](POLICY.md), 학습 고리·로그 링 버퍼는 [GPU_TRAINING.md](GPU_TRAINING.md), 입력은 [VLA_INPUT.md](VLA_INPUT.md), 실행 기록 위치는 [training/README.md](../../training/README.md).
- 본보기 둘: 사용자의 전투기 RL 학습 뷰어(`~/aircombat-rl-private/student/viewer/`, 이하 **전투기 뷰어**)와 팀의 Rust 실시간 뷰어 [sgview](../../src/scene_graph/sgview/README.md).
- 표기: **(추정)** = 재지 않은 숫자·판단. 지금 있는 코드에 대한 말은 직접 열어 보고 확인한 것만 적었다(파일 경로를 붙임).

## 0. 한눈에

1. **무엇**: 학습이 도는 동안 브라우저로 본다. 탭 셋 — **학습**(카드·곡선·표), **재생**(판 하나를 3D 로), **비교**(실행 여럿).
2. **서버**: Rust. sgview 틀(표준 라이브러리 `TcpListener` + 연결마다 스레드, `serde_json`·`flate2`)을 그대로 쓴다. **읽기 전용**이다. 학습 프로세스와 말을 섞지 않고 실행 폴더의 파일만 읽는다.
3. **화면**: 순수 JS. 곡선은 캔버스에 직접 그린다(차트 라이브러리 없음). 3D 는 sgview 가 바이너리에 넣어 둔 three.js·GLTFLoader·OrbitControls·로봇 GLB 를 같이 쓴다.
4. **기록 규약**: 학습기(Rust 로그 스레드)가 쓴다. `run.json`(설정·상수), `progress.jsonl`(업데이트마다 한 줄, 모집단 값), `episodes.jsonl`(판마다 한 줄, 표본), `replays/*.trp`(판 하나 궤적, **바이너리**), `evals/*.json`(평가 표).
5. **학습을 막지 않는다**: 학습기는 GPU 로그 링 버퍼를 비동기로 읽어 파일에 덧붙이기만 한다(GPU_TRAINING 4.1). 뷰어가 늦거나 꺼져 있어도 학습은 그대로 돈다.
6. **전투기 뷰어에서 배운 것**(8절)을 처음부터 규약에 넣는다. 예: 안 잰 값은 0 으로 찍지 않고 키를 뺀다, 목적함수가 다른 판은 집계에서 뺀다, 기준선 숫자는 `run.json` 에서 읽는다.

## 1. 참고한 코드 (확인함)

### 1.1 전투기 뷰어 (`~/aircombat-rl-private/student/viewer/`)

| 부분 | 확인한 사실 |
|---|---|
| 서버 `server.py` (1,095 줄) | 표준 라이브러리 `ThreadingHTTPServer`. 읽기 전용. 경로: `/api/runs`, `meta`, `metrics?from=`, `pool`, `episodes?from=`, `opponents`, `replays`, `replay?id=` |
| 이어 읽기 `_jsonl` | 경로마다 `[읽은 바이트, 줄들]` 캐시. 늘어난 바이트만 읽는다. 줄끝까지 온 줄만 캐시에 넣는다. 파일이 줄면 처음부터. 주석: 전에는 요청마다 34 MB 를 다시 읽어 `/api/runs` 가 5–35 초 밀렸다 |
| 줄 수 세기 `_lines` | 같은 바이트 오프셋 캐시. 전에는 `/api/runs` 마다 9 만 줄을 처음부터 셌다 |
| 응답 | 64 KB 넘고 브라우저가 받으면 gzip(수준 1). 끊긴 연결은 조용히. 대기열 128(기본 5 에서 `ERR_CONNECTION_REFUSED` 가 났다) |
| 잘림 | 판 기록은 `MAX_ROWS` 20,000 줄까지. 잘리면 `truncated`·`total`·`cap` 을 같이 보낸다 |
| 화면 `app.js` (3,075 줄) | `CHARTS` 배열 하나가 화면을 정한다. 3 초 폴링, `from=` 로 이어 받기. DOM 은 한 번만 짓고 바뀐 글자만 바꾼다. EMA 슬라이더, x 축(iteration / total_timesteps), 시점 커서, 고장 무늬 검사 6 개(못 잰 것은 이유와 함께 따로) |
| 전적 표 | 슬라이더 칸 `WIN_STOPS = [10, 20, 50, 100, 200, 500, Infinity]`. 이터가 아니라 **상대별 최근 N 판**. 이유: 업데이트 하나에 판이 0–1 개라서(`README.md`, `계획.md`) |
| 재생 `replay.js` (829 줄) | three.js(`fetch_vendor.py` 로 0.160.0 을 받아 둠, 없으면 CDN). 좌표 변환은 이 파일 한 곳. 카메라 넷, 궤적 10 초, HUD, 타임라인, 속도 |
| 기록 `my_runlog.py` | `FRAME_COLS` 32 칸 = "뷰어와 공유하는 유일한 규격". 궤적은 `REPLAY_FRAMES` 2,000 프레임까지 균등 솎기. JSON 판당 약 237 KB(실측 중앙값). `.tmp` 에 쓰고 이름 바꾸기. 보관 상한 `REPLAY_KEEP` 200 · 못 박은 것 `REPLAY_PIN_KEEP` 60 · 착취자 `REPLAY_EXP_KEEP` 8, 각각 따로 센다 |
| `run.json` (`gpu/train_gpu.py` `_meta`) | `pid`, `started`, `decision_hz`, WEZ 상수, `target_kl`, `ent_coef_range`, `n_steps`, `rollout` 등 |
| 착취자 분리 | `gpu/vec.py` 가 판 기록에서 착취자 좌석을 뺀다(`seat_bnd[:ms + 1]`). 곡선은 `progress_exp_<성향>.json`, 궤적은 `exp_<성향>/` 하위 폴더 |
| `labs` | `runs/labs/<이름>/`. `run.json` 에 `pid` 가 있을 때만 "training" 배지 |

### 1.2 sgview (`src/scene_graph/sgview/`)

| 부분 | 확인한 사실 |
|---|---|
| 서버 `src/main.rs` (832 줄) | `TcpListener` + 연결마다 스레드, 요청 줄을 직접 읽음, GET 만, `Connection: close` |
| 의존 | `serde_json`, `flate2`(`rust_backend`), 빌드에 `cc`(scenemap `walls.cpp` 컴파일). 파일 머리 주석은 "std 말고 없음" 이라 적혀 있으나 `Cargo.toml` 과 README 가 맞다 |
| 넣어 둔 파일 | `index.html`(944 줄)·`three.min.js`(UMD, 2021 빌드)·`OrbitControls.js`·`GLTFLoader.js` 를 `include_bytes!`. `build.rs` 가 `assets/robot/*`(GLB 10 개 + `robot.json`, 약 3.4 MB)를 바이너리에 넣는다 |
| `robot.json` | `src/robot/tools/build_viewer_assets.py` 가 만든다. `links`, `joints`, `movable` 11 개(바퀴 4, `omx_joint1–5`, 그리퍼 2), `joint_order` 는 **비어 있다** |
| SSE `/stream` | 붙으면 스냅숏(지도·요약·자세·관절·벽)을 먼저, 그다음 갱신. 클라이언트마다 `sync_channel(1024)`, 방송은 `try_send` — 못 따라오는 클라이언트는 끊고 다시 붙으면 스냅숏을 받는다. 10 초마다 keepalive |
| 지도 | `/api/map?since=` 바뀐 행만(`MAP_HIST` 8). 화면은 `DataTexture` 에 바뀐 행만 쓴다. 장면은 **z 위**(`camera.up.set(0, 0, 1)`) — ROS 좌표 그대로 |
| 로봇 모델 | `loadRobotModel()` 이 `robot.json` + GLB 로 링크 트리를 짓고 `setJoints({이름: rad})` 로 움직인다 |
| 스트림 원본 `scenemap/stream.hpp` | 선 형식 `[u32 len][u8 type][payload]` LE. `1 POSE` f64 ×4, `2 MAP_RECT` 머리 48 B(`i32 w,h; f64 res,ox,oy; i32 x0,y0,x1,y1`) + `i8 cells`(−1 모름, 0–100 %), `3 VIEW` JSON, `4 JOINTS`. 단일 생산자 링, 차면 그 프레임만 버림 |

### 1.3 우리 학습 설계에서 가져오는 것

| 문서 | 이 뷰어에 주는 것 |
|---|---|
| GPU_TRAINING 4.1 | 커널이 손실·보상·완성도 합을 **장치 링 버퍼**에 쌓고, 호스트는 `cudaEventQuery` 로 끝난 칸만 읽는다. 로그·체크포인트는 Rust 몫 → 이 뷰어가 읽는 파일을 그 Rust 로그 스레드가 쓴다 |
| GPU_TRAINING 5.3 | 정책이 실제로 본 지도 토큰을 스텝마다 기록(약 1.3 KB/스텝, 추정) |
| GPU_TRAINING 5.5·5.6 | 커리큘럼 C0–C2, 시작 지도 완성도 칸 0 / 0–30 / 30–70 / 70–100 / 100 % 으로 나눠 평가 |
| GPU_TRAINING 9.1·11 | FP8 감시(층마다 amax, 넘침·밑넘침 비율, BF16 기울기와의 코사인 ≥ 0.99). 단계 G0–G7 |
| POLICY 3.1·3.3 | 스킬(`approach`, `pick`, `place`, `open/close drawer`, `open/close door`, `press`), 보상 = 진행 + 성공 − 벌점, 항목 이름 |
| POLICY 3.3.4 | "판마다 보상 항목별 합계를 따로 기록", "성공률·판 길이·접촉 수를 같이", "보상은 오르는데 성공률이 안 오르면 꼼수", "판을 직접 본다" → **이 뷰어가 그 확인 도구다** |
| POLICY 6·8 | 학습 단계 0 / 1 / 2a BC / 2b DAgger / 2c 섞기 / 3 학생 RL / 4 실제. 평가 표(처음 보는 이름·집, 지도 완성도, `gt` 대 `slam`, 문장 반응, 교사 대비) |
| training/README | 실행마다 `~/<모델>_work/runs/<이름>/`, git 에 안 넣음 |

## 2. 목표와 안 하는 것

| 목표 | 화면에서 답할 질문 |
|---|---|
| 잘 되고 있나 | 스킬별 성공률이 오르나. 보상 항목 중 무엇이 끌고 가나 |
| 고장 났나 | 탐색이 죽었나, KL 이 넘나, 가치 머리가 포화됐나, 로그가 버려지나 |
| 꼼수를 찾았나 | 리턴은 오르는데 성공은 그대로인가. 판을 3D 로 돌려 본다 |
| 지도가 도움이 되나 | 성공률을 스킬 × 집 × 시작 지도 완성도로 쪼갠다 |
| 무엇이 나은가 | FP8 대 BF16(씨앗 3 개씩), 교사 특권 두 판, GRU 유무를 겹쳐 본다 |

- **안 하는 것**: 학습 제어(멈춤·하이퍼파라미터 바꾸기), 파일 지우기, 체크포인트 관리. 뷰어는 읽기만 한다.
- 텐서보드를 쓰지 않는다. PyTorch 가 없는 학습기(GPU_TRAINING 0절)라 텐서보드 기록기도 없다.

## 3. 실행 종류 (갈래)

| `kind` | 무엇 (POLICY 6) | 집계 |
|---|---|---|
| `teacher` | RL 교사, 스킬별 또는 잇기 (단계 1) | 학습 판 |
| `bc` | 학생 BC (단계 2a·2c) | 검증 손실 + 시뮬 폐루프 평가 판 |
| `dagger` | DAgger 바퀴 (단계 2b) | **학생이 몬 판만**. 교사가 몬 판은 따로(아래) |
| `rlft` | 학생 RL 다듬기 (단계 3) | 학습 판. 지킴 평가 판은 따로 |
| `eval` | 체크포인트 평가만 (POLICY 8) | 평가 판 |
| `lab` | 실험 정책. `<root>/labs/<이름>/` | 본학습 목록과 섞지 않음 |

- 종류는 **`run.json` 의 `kind` 가 정한다.** 이름으로 짐작하지 않는다.
- 화면 위 실행 고르는 자리는 줄을 나눈다: `teacher` / `student`(bc·dagger·rlft·eval) / `labs`. 선택은 줄을 건너 **하나**다(전투기 뷰어와 같음). 비교 탭만 여럿을 고른다.
- 기본으로 여는 실행은 `latest.txt` 가 가리키는 본학습이다. `lab` 은 기본 선택 후보가 아니다(전투기 뷰어 `_latest`).

### 3.1 한 실행 안에서 목적이 다른 판 — 줄기(stream)로 가른다

전투기 뷰어의 착취자와 같은 문제다. 한 프로세스에서 돌지만 **집계에 섞으면 조용히 틀리는** 판들이 있다.

| 실행 | 섞으면 안 되는 판 | 이유 |
|---|---|---|
| `dagger` | β 로 **교사가 몬** 판 | 학생 성공률에 교사 성공이 섞인다 |
| `teacher`, `rlft` | 평균 행동으로 도는 **평가 판** | 표본 행동 판과 분포가 다르다 |
| `rlft` | **지킴 평가** 판(처음 보는 이름·집) | 그 자체가 따로 볼 지표다 |
| 모두 | 커리큘럼에서 **앞 단계를 유지하는** 판(GPU_TRAINING 5.5 의 20 %) | 지금 단계 성공률이 부풀려진다 → 집계에는 남기되 `stage` 칸으로 늘 쪼갤 수 있게 |

- 규칙: 본 줄기는 실행 폴더 바로 밑, 나머지는 **하위 폴더** `s_<줄기>/` 에 같은 모양으로 둔다. 파일 이름에도 줄기 이름을 넣는다(`s_eval/episodes_eval.jsonl`).
- 이유: 서버가 본 줄기의 판 수를 `episodes*.jsonl` 로 센다. 같은 자리에 두면 그 glob 에 걸려 집계가 오염된다(전투기 뷰어 `exp_<성향>/` 와 같은 이유).

## 4. 실행 폴더 규약 (학습기가 쓴다)

```
~/<모델>_work/runs/                      # 뷰어의 --root (여러 개 가능)
  latest.txt                             # 기본으로 열 실행 이름
  <이름>/
    run.json                             # 설정·상수. 작다
    progress.jsonl                       # 업데이트마다 한 줄 (모집단)
    episodes.jsonl [, episodes.1.jsonl …] # 판마다 한 줄 (표본). 1 GB 마다 다음 번호
    replays/ep_<판번호>_<스킬>_<결과>[_pin].trp
    evals/it<이터>.json                   # POLICY 8 평가 표
    s_eval/   episodes_eval.jsonl, replays/…
    s_teacher/episodes_teacher.jsonl, replays/…   # dagger 의 교사 몬 판
  labs/<이름>/                            # 같은 모양
```

- 전부 **선택**이다. 있는 것만 화면에 뜬다. 없는 것은 "무엇이 없어서 안 그렸다" 를 적는다.
- 학습기는 이 폴더에 **덧붙이기**(`progress`, `episodes`)와 **이름 바꾸기**(`replays`, `evals`)만 한다. 뷰어는 아무것도 쓰지 않는다.

### 4.1 `run.json`

화면의 기준선·이름·단위는 전부 여기서 읽는다. 뷰어 코드에 숫자를 박지 않는다(8절 3번).

| 묶음 | 칸 |
|---|---|
| 정체 | `schema` 1, `kind`, `name`, `group`(씨앗 묶음 이름, 비교 탭의 띠), `seed`, `pid`, `pid_start`(`/proc/<pid>/stat` 시작 시각), `segments`(재개마다 `{started, from_iter, from_steps}` 를 **덧붙임**), git 커밋(본 저장소·서브모듈) |
| 출신 | `init_from`(체크포인트), `teacher`(bc·dagger 가 쓴 교사 실행 이름), `parent`(rlft 의 BC 실행) |
| 환경 | `ctrl_hz` 10, `phys_hz`, `n_envs`, `rollout_T`, `log_envs`(판 기록을 남기는 환경 번호 수), `t_max` 스킬별, `homes`(장면 이름 표) |
| 스킬·보상 | `skills`, 스킬별 `reward_terms`(이름·무게·단위, POLICY 3.3.2 표 그대로), `success`(들림 0.1 m, 그리퍼 0.005 m, 유지 1 s 등), `budget_s`(p90 × 1.5) |
| 지도 | `grid_res`, `grid_wh`, `slots` 16, `completion_bins`(예: `[0, 0.3, 0.7, 1]` + 끝점 칸 여부), `map_modes`(`gt`/`slam`/`partial`/`none`) |
| 커리큘럼 | 단계 표(이름·처음 지도·넘어갈 기준). GPU_TRAINING 5.5 와 POLICY 4.5 의 표가 다르므로 **이 칸이 정답**이다(12절) |
| 학습 | PPO: `gamma`, `gae_lambda`, `clip_range`, `target_kl`, `lr`, `n_epochs`, `n_minibatch`, `max_grad_norm`, `log_std_init`. BC: `chunk_H`, `denoise_steps`, `batch`. DAgger: `beta_schedule`, `rounds`. rlft: `kl_to_bc`, `residual_max` |
| 정밀도 | 층마다 BF16/FP8 표(GPU_TRAINING 9.1 5 번) |
| 로봇 | `action_names` 8, `joint_map`(프레임 열 → `robot.json` 관절 이름), URDF 해시 |
| 재생 | `frame_cols` 판 번호(4.4), 리플레이 보관 상한 |

- `started` 는 재개할 때마다 바뀐다. 경과 시간은 `segments` 첫 값 또는 판 기록 첫 줄의 `ts` 로 잡는다(8절 10번).

### 4.2 `progress.jsonl` — 모집단, 업데이트마다 한 줄

GPU_TRAINING 4.1 의 장치 링 버퍼 한 칸이 한 줄이 된다. Rust 로그 스레드가 합 → 평균으로 바꿔 쓴다. 값은 **그 업데이트 동안 끝난 모든 판**(모집단)에서 낸다.

| 이름공간 | 키 (예) |
|---|---|
| `time/` | `iter`, `env_steps`(**필수**, 단조 증가), `wall`, `fps_env`, `iter_ms`, `rollout_ms`, `update_ms` |
| `rollout/` | `n_eps`(**필수** — 가중 평균의 분모), `ep_ret_mean`, `ep_len_mean`, `success/<스킬>`, `n_eps/<스킬>`, `timeout_rate`, `drop_rate`, `contacts_per_ep`, `joint_limit_steps` |
| `reward/` | `<스킬>/<항>` = 판당 그 항의 합 평균(POLICY 3.3.4) |
| `train/` | `policy_loss`, `value_loss`, `entropy`, `log_std_mean`, `approx_kl`, `clip_frac`, `explained_variance`, `grad_norm`, `lr`, `adv_std` |
| `bc/`, `val/` | `flow_loss`, `end_bce`, `val/flow_loss`, `val/action_mse` |
| `dagger/` | `round`, `beta`, `agree`(학생–교사 행동 거리), `new_samples` |
| `rlft/` | `kl_to_bc`, `residual_norm`, `bc_weight` |
| `curr/` | `stage_frac/<단계>`, `completion_start_mean` |
| `fp8/` | `amax/<층>`, `overflow/<층>`, `underflow/<층>`, `grad_cos` |
| `gpu/` | `mem_used_mb`, `mem_reserved_mb` |
| `log/` | `ring_dropped`, `replay_dropped`, `eps_dropped` — 로그를 버린 수. 0 이 아니면 화면에 적는다 |

- 이름은 SB3 꼴(`<묶음>/<이름>`)로 둔다. grep 으로 찾기 쉽고 전투기 뷰어 코드를 옮기기 쉽다.
- 다만 리턴은 `ep_ret_mean` 이다. SB3 의 `ep_rew_mean` 은 이름이 틀렸다(값이 판 합 = 리턴, 전투기 뷰어 `app.js` "Return" 칸 주석).
- **잰 것만 쓴다.** 그 업데이트에 그 스킬 판이 하나도 안 끝났으면 `success/<스킬>` 키를 **뺀다**. `null`·`0` 을 쓰지 않는다.
- 업데이트가 초당 1 번보다 잦으면 여러 업데이트를 합쳐 한 줄로 쓴다(`time/iter` 는 마지막 값, 합친 수 `time/iters_in_row`). 9절 예산 때문이다.
- 재개 때 학습기는 체크포인트 이터 뒤의 줄을 지운다(전투기 `RunLog` 의 trim 과 같음). 뷰어는 그래도 `env_steps` 가 줄어드는 줄을 걷어내고 그 수를 적는다(8절 9번).

### 4.3 `episodes.jsonl` — 표본, 판마다 한 줄

```json
{"ts":1759480000.1,"iter":812,"env_steps":53215232,"env":17,"ep":40211,
 "skill":"pick","home":"Rs_int","stage":"C1","map_mode":"slam","completion0":0.42,
 "driver":"student","success":true,"outcome":"success","t":18.3,"steps":183,
 "ret":31.2,"r":{"reach":6.1,"align":0.8,"lift":2.0,"success":20,"contact":-1,"time":-1.83},
 "contacts":1,"drops":0,"min_clear_m":0.07,"replay":"ep_040211_pick_success.trp"}
```

- **표본 뽑기**: `run.json` 의 `log_envs` 개 환경을 고정해 두고, 그 환경에서 끝난 판은 **전부** 쓴다. 판 길이·결과로 고르지 않는다.
  - 전투기 뷰어는 이터당 N 줄로 잘랐다. 판이 동기화돼 있어 짧은 판(격추)이 과대표집됐다 — 진짜 격추 0.1 % 를 표본은 53 % 로 셌다(`app.js` 주석, 실측). 같은 실수를 안 하려는 규칙이다.
- `completion0` 은 판 **시작** 때 과제 물체 중 확정 비율(GPU_TRAINING 5.6). 칸 나누기는 뷰어가 `completion_bins` 로 한다.
- `home` 은 장면 이름이다. 정책 입력이 아니라 사람이 보는 칸이라 이름을 써도 된다(VLA_INPUT 0절은 정책 입력 규칙).
- `r` 의 키 이름은 `run.json` 의 `reward_terms` 와 같다. 판 하나의 `ret` = `r` 값의 합이어야 한다. 서버가 첫 읽기 때 이것을 확인하고 어긋난 줄 수를 적는다.
- `driver`: `student` / `teacher` / `mix`. `split`: 줄기 이름. 둘 다 줄기 폴더와 겹치는 정보지만 줄 하나만 떠 있어도 알 수 있게 둔다.

### 4.4 `replays/*.trp` — 판 하나, 바이너리

#### 왜 바이너리인가

| | JSON (전투기 방식) | 바이너리 (이 설계) |
|---|---|---|
| 프레임 열 | 32 열 × ≤ 2,000 프레임에 약 237 KB(실측) | 우리는 열이 훨씬 많다(아래). 같은 값을 f32/f16 으로 2–3 배 작게 (추정) |
| 지도 | 격자를 숫자 배열로 쓰면 크다. base64 로 넣으면 다시 풀어야 한다 | stream.hpp `MAP_RECT` 와 **같은 바이트**를 그대로. sgview 해석 코드를 그대로 쓴다 |
| 영상 | base64 로 33 % 커진다 | JPEG 바이트 그대로 |
| 읽기 | 통째로 `JSON.parse` | `Float32Array` 를 파일 위에 바로 겹침(복사·해석 없음). 프레임 폭이 고정이라 k 번째 프레임에 바로 감 |
| 서버 | — | 바꾸지 않고 그대로 보낸다(gzip 만) |

- 작은 것(`run.json`, 줄 기록, 평가 표)은 JSON 으로 둔다. 사람이 `head`·`jq` 로 보는 것이 더 값지다.
- 바이너리라도 **머리에 열 이름을 싣는다.** 열을 더해도 옛 뷰어·옛 파일이 안 깨진다(전투기 `cols` 와 같은 생각).

#### 파일 모양 (모두 little-endian, 섹션 시작은 8 바이트 정렬)

```
"TRP1"  u32 head_len  head(JSON, UTF-8)  [채움]
섹션 frames : n_frames × n_cols × f32           (행 우선)
섹션 slots  : n_frames × 16 × n_slot_cols × f16
섹션 map    : 기록이 이어짐 — [u32 frame][MAP_RECT 페이로드(48 B 머리 + i8 cells)]
섹션 img    : (선택) [u32 frame][u8 cam][u32 len][JPEG]
```

머리 JSON: `meta`(그 판의 `episodes.jsonl` 줄 그대로), `cols`, `slot_cols`, `dt`, `stride`, `n_frames`, `sections`(이름·오프셋·길이), `objects`(슬롯 번호 → 물체 이름·범주), `joint_map`, `grid`(`res`, `ox`, `oy`, `w`, `h`).

- 판 자신의 메타가 머리에 있으므로 재생 목록은 `episodes.jsonl` 과 짝짓지 않아도 된다. 전투기 뷰어는 짝지을 판 기록이 잘림 창 밖이면 목록에 `?` 가 떴다(`replay.js` `replay-note` 주석, 실측 390 개 중 43 개).

#### 프레임 열 `frame_cols` 1 판 (f32)

| 묶음 | 열 | 개수 |
|---|---|---|
| 시간 | `t` | 1 |
| 참 자세 | `x`, `y`, `yaw` (`map`) | 3 |
| `slam` 자세 | `sx`, `sy`, `syaw` | 3 |
| 몸통 속도 | `vx`, `wz` (잰 값) | 2 |
| 팔 | `q1`–`q5`, `qg` | 6 |
| 손끝 | `ee_x`, `ee_y`, `ee_z` (`map`, 순기구학) | 3 |
| 행동 | `a_vx`, `a_wz`, `a_q1`–`a_q5`, `a_g` (정책 출력) | 8 |
| 거른 행동 | `f_*` 8 (학생, 안전 거르개 뒤 — POLICY 7.1). 없으면 열 자체를 뺌 | 0 / 8 |
| 보상 | `r_<항>` (`run.json` 의 항 이름), `ret_cum` | 항 수 + 1 |
| 정책 | `value`(가치), `end_p`(학생의 끝 신호) | 1–2 |
| 사건 | `ev` 비트(접촉, 쥠, 놓침, 성공, 거르개 개입, 리셋) | 1 |
| 목표 | `tgt_slot`, `tgt_x`, `tgt_y`, `tgt_z` | 4 |
| 지도 | `completion` | 1 |
| 합 | | 약 45–55 (추정) |

- 좌표는 전부 `map` 프레임(ROS: x 앞, y 왼쪽, z 위)이다. sgview 장면이 z 위라 **바꿀 것이 없다.** 바꾸는 곳은 격자 행 순서(행 0 = 최소 y) 하나뿐이고, 그것도 sgview `onMapEvent` 가 이미 한다.
- 참 자세와 `slam` 자세를 둘 다 싣는다. 화면에서 둘의 궤적을 겹치면 로컬라이제이션 오차가 보인다. 둘을 섞어 쓰지 않는다(8절 4번).

#### 슬롯 열 (f16, 칸마다)

`id`(머리의 `objects` 번호), `bx`, `by`, `bz`(지도가 믿는 위치), `px`, `py`, `pz`(참 위치), `ex`, `ey`, `ez`(크기), `src`(0 빔 / 1 지금 보는 중 / 2 기억), `unc`(위치 불확실도 m), `state`(보임·사라짐·옮겨짐·들고 있음), `is_tgt`, `age`(마지막 본 뒤 s). 15 개 × 2 B × 16 칸 = 480 B/프레임.

- 위치를 f16 으로 쓰면 16–32 m 구간에서 1/64 m ≈ 1.6 cm 단위다(f16 가수 10 비트). 32 m 를 넘으면 3.1 cm 다. 슬롯은 그 정도면 된다. 로봇·손끝은 f32 다.

#### 지도 기록

- 판 시작 때 전체 격자 하나, 그다음 keyframe 마다 **바뀐 사각형만**(stream.hpp `MAP_RECT`).
- GPU 격자는 int16 로그 오즈 + "본 적 있음" 이다(GPU_TRAINING 5.1). 쓰는 쪽이 `MAP_RECT` 형식(−1 모름, 0–100 %)으로 바꿔 쓴다. 변환식은 scenemap 과 같게.
- 뒤로 감기: 화면이 N 프레임마다(예: 50, 추정) 격자 사본을 메모리에 둔다. 가장 가까운 사본 + 그 뒤 변화분으로 맞춘다.

#### 영상 (선택)

- 교사는 영상이 없다. 학생(bc·dagger·rlft 의 평가 판)만 리모 카메라·손목 카메라 RGB 를 싣는다.
- 224², JPEG 품질 80, 2 Hz (추정). 판당 약 1 MB (추정). 기본은 끔, 평가 줄기에서만 켬.
- 부호화는 로그 스레드(CPU)에서 한다. 학습 고리 안에서 하지 않는다. 밀리면 그 판의 영상만 버리고 `log/replay_dropped` 를 센다.

#### 쓰기 규칙

- `.tmp` 에 다 쓴 뒤 이름 바꾸기 — 반쪽 파일이 안 보인다(전투기 `RunLog._replay`).
- 프레임 상한은 `run.json` 의 스킬별 `t_max × ctrl_hz` 에서 계산한다. 숫자를 따로 박지 않는다(8절 13번).
- 재개하면 판 번호를 이어서 센다. 0 부터 세면 파일 이름이 겹쳐 덮인다(전투기 `RunLog.__init__` 주석).

### 4.5 `evals/it<이터>.json`

POLICY 8 의 평가 한 번이 파일 하나다. `{"iter", "ckpt", "rows": [{"eval": "unseen_names", "split": …, "n", "success", "ref", "pass"}]}`. 기준(`ref`)은 학습기가 POLICY 8 표에서 계산해 싣는다. 뷰어는 표만 그린다.

### 4.6 쓰는 쪽 — 학습 고리를 막지 않는 법

```
GPU 롤아웃 그래프 ──(장치 링: 합계, 리플레이 환경의 프레임)──▶ 고정 메모리
Rust 로그 스레드: cudaEventQuery 로 끝난 칸만 → 평균 계산 → BufWriter 로 덧붙이기 / .trp 조립
뷰어: 파일만 읽음
```

- 리플레이 프레임은 고른 환경(줄기마다 몇 개)만 장치 버퍼에 쓴다. 판 64 개 × T 64 × 약 1 KB = 4 MB 쯤이라 GPU_TRAINING 8.1 예산에 비해 작다 (추정).
- 로그 스레드가 늦으면 **버린다**(세고 적는다). 학습을 기다리게 하지 않는다. sgview 스트림과 같은 원칙이다(`stream.hpp`: 링이 차면 그 프레임만 버림).
- 열 이름·파일 모양은 작은 Rust 크레이트 하나(`trainfmt`)에 한 번만 정의하고, 쓰는 쪽(학습기)과 읽는 쪽(뷰어 서버)이 같이 쓴다. 전투기 뷰어의 `FRAME_COLS` 가 "뷰어와 공유하는 유일한 규격" 인 것과 같다.

## 5. Rust 서버

### 5.1 자리와 실행

```bash
cd training/viewer && cargo build --release
target/release/trainview --root ~/rl_work/runs --root ~/bc_work/runs [--port 7810] [--bind 127.0.0.1]
```

- 자리는 `training/viewer/` 를 제안한다. 학습 도구라서다(12절).
- 기본 bind 는 127.0.0.1 이다. 원격(Tailscale 등)에서 보려면 `--bind 0.0.0.0`. sgview 기본은 0.0.0.0 이다.

### 5.2 크레이트 배치

```
training/viewer/
  Cargo.toml
  build.rs            assets/* 와 sgview/assets/robot/* 를 바이너리에 넣음 (sgview build.rs 방식)
  src/main.rs         인자, 리스너, 연결마다 스레드
  src/http.rs         요청 읽기·응답·gzip·안전한 경로 (sgview 의 respond / parse_query / percent_decode / safe_join)
  src/runs.rs         실행 찾기, kind, 줄기, live 판정, latest
  src/tail.rs         JSONL 이어 읽기 + 열 저장소
  src/table.rs        스킬 × 집 × 완성도 표
  src/replay.rs       .trp 머리 읽기, 목록
  src/live.rs         SSE (sgview 방송 방식)
  assets/index.html, app.js, charts.js, replay.js, style.css
trainfmt/             (쓰는 쪽과 같이 쓰는 형식 크레이트, 4.6)
```

### 5.3 의존 — 최소

| 크레이트 | 쓰나 | 이유 |
|---|---|---|
| std `TcpListener` + 스레드 | 씀 | 보는 사람이 1–3 명이다. 비동기 런타임(tokio·hyper·axum)이 줄 이득이 없다. sgview 가 같은 틀로 SSE 240 Hz 를 냈다(sgview README 측정) |
| `serde_json` | 씀 | `run.json`, 줄 기록, `.trp` 머리 해석. sgview 가 이미 쓴다 |
| `flate2` (`rust_backend`) | 씀 | 큰 응답 gzip. 전투기 뷰어가 34 MB 를 느린 길(Tailscale)로 보내다 첫 화면이 막혔다. C 라이브러리 없이 빌드된다. sgview 가 이미 쓴다 |
| `notify`(inotify) | 안 씀 | 실행마다 파일 몇 개라 1 초마다 `stat` 이 충분하다. 크레이트·플랫폼 차이를 늘리지 않는다 |
| `cc` | 안 씀 | 벽 계산(`walls.cpp`)이 필요 없다. 재생은 격자 그림만 쓴다 |
| `libc` | 안 씀 | 프로세스 살아 있나는 `/proc/<pid>` 로 본다(리눅스) |

### 5.4 경로

| 경로 | 돌려주는 것 |
|---|---|
| `GET /`, `/app.js`, `/charts.js`, `/replay.js`, `/style.css` | 화면(바이너리에 넣음) |
| `GET /three.min.js`, `/OrbitControls.js`, `/GLTFLoader.js`, `/robot/<파일>` | sgview 와 같은 파일 |
| `GET /api/runs` | 실행 목록: `name`, `root`, `kind`, `streams`, `live`, `age`, `iters`, `env_steps`, `eps`(줄 수), `replays`, `stage`, `first_ts`, `rewound`, `disk_mb`. 그리고 `latest` |
| `GET /api/meta?run=` | `run.json` 그대로 |
| `GET /api/progress?run=&keys=a,b&from=<줄>&sig=` | 고른 키만 **열로**: `{start, total, sig, x: {iter: [...], env_steps: [...]}, cols: {a: [...], b: [...]}}`. 없는 값은 `null`(그 줄에 키가 없었다). `keys` 를 비우면 키 목록과 키마다 처음·마지막 줄 |
| `GET /api/progress?…&max_points=2000` | 긴 실행: 픽셀 칸마다 최소·최대·마지막(그림이 안 거짓말하는 솎기) |
| `GET /api/episodes?run=&stream=&from=<줄>` | 줄 그대로. `{start, total, cap, truncated, rows}` |
| `GET /api/table?run=&stream=&last=<N>&rows=skill&cols=bin&home=` | 칸마다 최근 N 판의 `n`, 성공, 평균 `t`, 접촉, 리턴. `last` 비우면 전체 |
| `GET /api/replays?run=&stream=` | `.trp` 머리의 `meta` 목록(파일 크기·시각 포함) |
| `GET /api/replay?run=&stream=&id=` | `.trp` 바이트 그대로 (gzip 되면 gzip) |
| `GET /api/evals?run=` | `evals/*.json` 모음 |
| `GET /api/live?runs=a,b&from=…` | SSE. 이벤트 `progress`(새 줄의 고른 키), `episodes`, `replay`(새 파일 머리), `runs`(목록 바뀜), `reset`(파일이 새로 쓰임) |

- 실행 이름·파일 이름은 `[\w.=-]+` 만 받는다(전투기 `SAFE`). 경로는 sgview `safe_join` 으로 `..` 을 막는다.
- 없는 것은 지어내지 않는다. 빈 결과에 **왜 없는지**를 `why` 칸에 싣는다(전투기 `_api_exp`).

### 5.5 이어 읽기와 캐시

파일마다 다음을 들고 있다.

| 칸 | 뜻 |
|---|---|
| `ino`, `len` | 파일 정체. `ino` 가 바뀌거나 길이가 줄면 새 파일 → 처음부터, `sig` 바꿈 |
| `off` | 줄끝까지 읽은 바이트 |
| `line_off: Vec<u64>` | 줄 번호 → 바이트 위치 (`from=` 을 바로 찾음) |
| 열 저장소 | 키 이름 → `Vec<f32>`(줄 수만큼, 없으면 NaN). 키 이름은 한 번만 저장(인터닝) |
| 나쁜 줄 수 | 해석 못 한 줄(죽었다 재개해 중간에 반쪽 줄이 남은 경우) |

- 요청이 오면 `stat` 하고, 늘어난 바이트만 읽는다. 마지막 `\n` 뒤의 반쪽 줄은 **보내지 않는다.** 다음에 다시 읽는다.
  - 전투기 뷰어는 반쪽 줄을 그 응답에만 붙였다. 해석이 되는 반쪽이면 클라이언트가 그 줄까지 가졌다고 세어 다음 `from=` 에서 완성된 줄을 건너뛸 수 있다 (추정 — 코드로 본 가능성이고 재현은 안 함). 아예 안 보내면 이 문제가 없다.
- `progress` 는 들어올 때 한 번만 해석해 열로 쌓는다. 요청마다 해석하지 않는다. 전투기 뷰어가 요청마다 해석하다 3 만 줄에 0.33 s 를 썼다(`_progress_stat` 주석, 실측).
- 열 저장소 크기: 줄 86,400(하루, 초당 1 줄) × 키 150 × 4 B ≈ 52 MB/실행 (추정). 열린 실행만 들고, 10 분 안 본 실행은 내린다.
- 표(`/api/table`)는 `episodes` 를 들어올 때 칸 번호(스킬·집·완성도 칸·줄기)와 숫자 몇 개만 뽑아 판당 약 32 B 로 쌓는다. 100 만 판에 32 MB (추정). "최근 N 판" 은 칸마다 링으로 잘라 낸다.
- 큰 응답은 `flate2` 빠른 수준(1)으로 gzip. 64 KB 아래는 그대로.
- `/api/runs` 의 디렉터리 크기(`disk_mb`)는 1 분에 한 번만 다시 잰다.

### 5.6 학습을 막지 않는다

- 읽기 전용 `open` 만 한다. 잠금·`fsync`·쓰기 없음. 리눅스에서 읽는 쪽은 덧붙이는 쪽을 막지 않는다.
- 한 요청이 읽는 양에 상한을 둔다(예: 64 MB, 추정). 처음 여는 큰 파일은 그 상한씩 나눠 읽고 화면에 "읽는 중 n %" 를 적는다.
- 같은 GPU 를 쓰지 않는다. 뷰어는 CPU 만 쓴다.

### 5.7 "학습 중" 판정

- `run.json` 에 `pid` 가 있으면 `/proc/<pid>` 가 있고, `/proc/<pid>/stat` 의 시작 시각이 `pid_start` 와 같을 때만 살아 있다. pid 재사용을 거른다.
- `pid` 가 없는 실행: 마지막 기록 후 `LIVE_S`(전투기 180 s) 안이면 살아 있다고 본다. 우리 평가 공백 길이는 재서 정한다 (추정).
- `lab` 은 `pid` 가 있을 때만 "training" 이다(전투기 `_labs`). 한 번에 떨구고 끝나는 산출물이 "학습 중" 으로 뜨면 안 된다.

### 5.8 SSE

- sgview `serve_stream` 을 그대로 가져온다: 붙으면 스냅숏 먼저, 클라이언트마다 묶인 채널, `try_send` 로 방송, 못 따라오면 끊음, 10 초 keepalive.
- 스냅숏은 "클라이언트가 이미 가진 줄 수(`from`) 뒤" 다. 다시 붙어도 빠지거나 겹치는 줄이 없다.
- 감시 스레드 하나가 구독된 실행의 파일을 1 초마다 `stat` 한다 (추정 주기). 늘었으면 5.5 로 읽고 방송한다.
- SSE 가 끊기면 화면은 3 초 폴링으로 내려간다(전투기 방식). 둘 다 같은 `from=` 규칙을 쓴다.

## 6. 화면

### 6.1 공통

- 머리줄: 이름, 탭(학습·재생·비교), 실행 줄 셋(teacher / student / labs), 상태줄(이터·판 수·마지막 갱신 시각), "training" 배지.
- 3 초 폴링이나 SSE 가 와도 DOM 을 다시 짓지 않는다. 바뀐 글자만 바꾸고, 캔버스는 내용 서명이 바뀔 때만 다시 그린다(전투기 `setText`, `S.sig`).
- 보이지 않는 탭은 그리지 않는다. 재생 탭이 꺼져 있으면 WebGL 고리를 멈춘다(전투기 `tick`).

### 6.2 학습 탭

**요약 카드**

| 카드 | 출처 |
|---|---|
| iteration · env_steps | `progress` 마지막 줄 (되감긴 줄 뺌) |
| env steps/s · 경과 | `time/fps_env`, 첫 기록 시각부터 |
| 단계 | `run.json` `kind` + POLICY 6 단계 + 지금 커리큘럼 섞임 |
| 성공률 | 스킬별 미니 막대(`rollout/success/*`, 모집단) |
| 리턴 · 판 길이 | `rollout/ep_ret_mean`, `ep_len_mean` |
| 판 수 | 모집단 / 기록된 표본 |
| 탐색 | `train/log_std_mean`(가우스), 처음 값 대비 % |
| KL · EV | `train/approx_kl`(대 `target_kl`), `explained_variance` |
| GPU 메모리 | `gpu/mem_used_mb` / 16 GB |
| 로그 버림 | `log/*_dropped`. 0 이 아니면 빨강 |

**고장 무늬 검사** (전투기 `checks` 틀. 못 잰 검사는 이유와 함께 "not measurable" 로 따로 적는다)

| 무늬 | 조건 (문턱은 추정, `run.json` 에서 덮어씀) |
|---|---|
| 꼼수 의심 | 최근 창에서 리턴 기울기 > 0 인데 그 스킬 성공률 기울기 ≤ 0 (POLICY 3.3.4) |
| 탐색 죽음 | `log_std_mean` 이 처음 값보다 크게 낮음 |
| KL 넘침 | `approx_kl` 평균 > `target_kl` × 1.5 |
| 가치 머리 포화 | EV ≥ 0.99 이고 KL < 0.001 |
| 안 움직임 | 시간 초과 비율 ≥ 99 % 이고 판 길이 ≈ `budget_s × ctrl_hz` |
| 접촉 늘어남 | `contacts_per_ep` 가 창 안에서 오름 |
| 로그 버림 | `log/*_dropped` > 0 |
| FP8 위험 | `fp8/overflow/*` > 문턱 또는 `fp8/grad_cos` < 0.99 (GPU_TRAINING 9.1) |

**곡선 묶음** (`CHARTS` 배열 한 곳이 화면을 정한다. 묶음 순서 = 보는 순서)

| 묶음 | 칸 |
|---|---|
| 결과 (모집단) | 스킬별 성공률 · 스킬별 판 길이(+ `budget_s` 선) · 시간 초과·떨어뜨림·충돌 비율 |
| 보상 항목 | 스킬마다 캔버스 하나, 항목마다 선(`reward/<스킬>/<항>`). 리턴과 성공률을 오른쪽 축에 겹침 |
| 안전 | 판당 접촉 · 관절 한계 근처 스텝 · 거르개 개입(학생) · 최소 여유 거리 |
| 학습 건강 | 엔트로피·`log_std` · KL(+ `target_kl` 선) · clip 비율 · EV + 가치 손실(오른쪽) · 기울기 노름(+ `max_grad_norm` 선) · 학습률 |
| 학생 | flow 손실(학습·검증) · 행동 MSE · 끝 신호 BCE · DAgger β · 학생–교사 일치 · `kl_to_bc` · 잔차 크기 |
| 지도·커리큘럼 | 단계 섞임 · 시작 완성도 평균 · 완성도 칸별 성공률 |
| 속도·자원 | env steps/s · 이터 시간(롤아웃 / 갱신) · GPU 메모리 · 로그 버림 |
| FP8 | 층별 amax · 넘침·밑넘침 · BF16 기울기 코사인(+ 0.99 선) |
| 판 기록 ⚠ 표본 | 판 길이 분포, 최소 여유 거리 등 모집단에 없는 칸만 |

- 키가 없는 칸은 접는다. 묶음 머리에 "not logged: `rollout/success/pick`" 처럼 **무엇을 찍으면 되는지** 적는다(전투기 `collapseGroups`, `labKeyNote`).
- 기준선 값이 `run.json` 에 없으면 선을 안 긋고 칸 아래에 없다고 적는다.
- x 축: iteration / env_steps. 판 기록은 `env_steps` 로 얹는다. 이터 번호는 `progress` 에서 찾아 붙인다 — 롤아웃 크기로 나누지 않는다(롤아웃이 바뀔 수 있다. 전투기 `iterOfStep`, 실측 261 대 진짜 359).
- 첫 판이 끝나기 전에는 "첫 판은 약 이터 n 에 끝남" 을 적는다(`t_max × ctrl_hz / rollout_T`). 빈 그래프와 고장 난 그래프를 가른다(전투기 `first_ep_it`).
- EMA 슬라이더, 시점 커서(카드·표·재생 목록이 그 이터로 잘림)는 전투기 그대로.

**성공 표 — 스킬 × 집 × 시작 지도 완성도**

| | 0 % | 0–30 % | 30–70 % | 70–100 % | 100 % |
|---|---|---|---|---|---|
| `pick` | 성공 % · n · 평균 t | … | | | |
| `place on` | | | | | |

- 슬라이더: **칸마다 최근 N 판**(10, 20, 50, 100, 200, 500, 전체). 이터로 자르지 않는다. 칸마다 판이 드나드는 속도가 달라 이터로 자르면 어떤 칸은 n 이 몇 개뿐이다(전투기 `계획.md` 와 같은 이유).
- 고르기: 행(스킬 / 집), 열(완성도 칸 / 지도 방식 `gt`·`slam`·`none`), 거르기(집, 줄기, 단계). 칸 색은 성공률, n < 20 이면 흐리게 + "표본 적음" (추정 문턱).
- 이 표는 **표본**(`episodes`)이다. 머리에 "⚠ sample: log_envs n 개 환경의 모든 판" 을 적는다. 스킬별 전체 성공률은 위 곡선(모집단)과 맞춰 볼 수 있다.
- 평가 표(`evals/*.json`): 체크포인트마다 POLICY 8 의 줄, 기준 통과·실패 색.

### 6.2.1 끝까지 해냈나 — 행동 지표 (맨 위 카드와 표)

행동 학습의 "끝까지 성공" 은 F1 이 아니라 아래로 잰다(F1 은 지도 변화 맞히기 같은 분류 지표). 정의는 [CURRICULUM_APPROACH.md](CURRICULUM_APPROACH.md) 5절.

| 지표 | 식 | 어디에 |
|---|---|---|
| 성공률 (SR) | 성공 판 / 전체 판 | 요약 카드 맨 앞, 곡선, 표 |
| SPL | 평균 S × L* / max(L, L*) (S 성공 0/1, L* 최단 경로, L 실제 간 길이) | 요약 카드, 곡선, 표 |
| Soft SPL | 실패도 다가간 만큼 부분 점수 | 곡선 |
| 충돌 없는 성공률 | 충돌 0 인 성공 / 전체 | 곡선. 충돌 = 실패인 과제에서 SR 과 다르면 경고 표시 |
| 충돌률 · 시간 초과율 | | 곡선, 표 |
| 끝 자세 오차 | 성공 판의 마지막 거리·에임 각도 분포 | 히스토그램 |
| 과제 전체 성공률 · 단계별 성공률 · 부분 점수 | 가기 → 집기 → 놓기 이어서 할 때 | 집기·놓기 단계부터 |

- `episodes.jsonl` 판 줄에 `success`, `collided`, `timeout`, `path_len`, `path_len_opt`(L*), `final_dist`, `final_aim_deg`, `curriculum`(A0–A5), `home`, `map_completeness` 를 넣는다. 안 잰 값은 키를 뺀다(8절 교훈).
- 표: 커리큘럼 단계 × 집(학습 / 안 쓴 집) × 지도 완성도마다 SR·SPL·충돌률, 최근 N 판 슬라이더.

### 6.3 재생 탭

- 목록: 학습 탭에서 고른 실행을 따라온다. 그 실행 이름을 줄 왼쪽에 적는다. 거르기: 스킬, 결과, 집, 완성도 칸, 줄기, "이 이터 근처".
- 실행이 바뀌면 **화면부터 비운다.** 전투기 뷰어는 새 실행에 궤적이 없으면 앞 실행의 기체가 그대로 떠 있었다(`replay.js` `open` 주석).

| 그리는 것 | 어떻게 |
|---|---|
| 로봇 | sgview `loadRobotModel` + `setJoints`. `q1`–`q5` → `omx_joint1–5`, `qg` → 그리퍼 두 관절(나누는 식은 URDF 에서 확인, 12절). 바퀴는 `vx`·`wz` 를 적분해 돌림(보기용) |
| 몸통 궤적 | 참 자세(파랑)와 `slam` 자세(주황) 두 줄. 손끝 궤적은 켜고 끔 |
| 물체 칸 | 믿는 위치에 상자(크기 `ex,ey,ez`). 출처: 지금 보는 중 = 꽉 찬 상자, 기억 = 반투명 선 상자. 불확실도 = 바닥 원 반지름. 참 위치가 다르면 흐린 유령 상자 + 잇는 선. 목표 칸은 굵은 테. 사라짐·옮겨짐·들고 있음 색 |
| 자라는 지도 | `MAP_RECT` 기록을 프레임 따라 `DataTexture` 에 씀(sgview `onMapEvent` 코드) |
| 목표 | 표시 + 손끝에서 잇는 선 |
| 보상 띠 | 타임라인 아래 캔버스: 스텝마다 보상 항목을 쌓은 막대(+ 는 위, − 는 아래), 사건 눈금(접촉·쥠·놓침·성공), 가치 `value` 선, 끝 신호 `end_p` 선. 누르면 그 프레임으로 감 |
| 카메라 영상 | 있으면 작은 칸 둘(리모 카메라, 손목). 가장 가까운 프레임 |
| HUD | t, 스킬, 이번 스텝 보상과 항목, 누적 리턴, 쥠 상태, 접촉 수, 목표까지 거리, 완성도 |
| 시점 | 1 지도 고정(위에서) · 2 로봇 따라감 · 3 손목 · 4 자유(OrbitControls). 숫자 키 |
| 조작 | 재생·멈춤, 속도, 스크럽, 한 프레임씩 |

### 6.4 비교 탭

- 실행 여럿을 고른다. 색 = 실행, 선 모양 = 계열(전투기 `RUN_COLORS`, `DASH`).
- `run.json` 의 `group` 이 같은 실행(씨앗만 다름)은 평균 선 + 표준편차 띠로 묶는다. GPU_TRAINING 9.1 의 "FP8 평균이 BF16 씨앗 퍼짐 안" 을 눈으로 본다.
- x 축: env_steps / 벽시계 / 처음부터 지난 시간.
- 비교 표: 실행마다 마지막 창의 스킬별 성공률, 완성도 칸별 성공률, 학생/교사 비율.
- 키가 한쪽에만 있으면 그쪽만 그리고 "없음" 을 적는다.

## 7. 화면 기술

| 무엇 | 선택 | 이유 |
|---|---|---|
| JS | 순수 ES 모듈, 빌드 단계 없음 | 바이너리에 넣고 바로 연다. 전투기·sgview 둘 다 이렇다 |
| 곡선 | 캔버스 직접(전투기 `plot` 을 옮김: DPR, 두 축, 눈금, 기준선, 커서, 값 읽기) | 기준선·잘림 표시·못 잰 칸 설명을 마음대로 넣는다. 의존 0 |
| 긴 곡선 | 서버가 `max_points` 로 칸마다 최소·최대·마지막을 줌. 화면은 EMA 를 그 위에 | 하루치 8 만 줄을 다 보내지 않는다 |
| 3D | sgview 의 `three.min.js`·`OrbitControls.js`·`GLTFLoader.js` 를 같이 씀 | 로봇 모델 코드(`loadRobotModel`, `setJoints`)가 이 판에 맞춰 있다. 인터넷 없이 돈다(`fetch_vendor` 단계가 없다) |
| 로봇 GLB | `build.rs` 가 `src/scene_graph/sgview/assets/robot/` 를 가리킴 | 파일을 두 벌 두지 않는다. 만드는 곳은 `build_viewer_assets.py` 하나 |
| sgview 와 나눌 JS | 로봇 모델, 격자 텍스처, SSE 받기 | 지금 sgview 는 `index.html` 한 파일(944 줄) 안에 있다. 처음엔 옮겨 쓰고, 나중에 모듈로 뺄지 정한다(12절) |

## 8. 전투기 뷰어에서 가져오는 교훈

| # | 교훈 | 그 저장소에서 왜 생겼나 | 여기서는 |
|---|---|---|---|
| 1 | 안 잰 값은 0 으로 찍지 말고 **키를 뺀다** | 0 을 넣으면 화면이 "쟀는데 0" 으로 그렸다. 한 번 고쳤다(`README.md` labs 절) | 4.2. 스킬 판이 없는 업데이트는 그 키가 없다 |
| 2 | 목적함수가 다른 판은 **집계에서 빼고, 궤적은 따로 남긴다** | 착취자 판이 섞이면 Main 의 리턴이 두 보상의 평균이 되어 조용히 틀렸다(`vec.py` `seat_bnd[:ms + 1]`) | 3.1 줄기. DAgger 교사 판, 평가 판 |
| 3 | 줄기 파일은 본 glob 에 안 걸리는 **하위 폴더**에, 이름에도 줄기를 | 같은 자리에 두면 서버가 그 glob 으로 판 수를 세어 오염된다 | `s_<줄기>/…_<줄기>.jsonl` |
| 4 | 기준선 상수는 **`run.json` 에서** | WEZ 콘을 참고 뷰어 값(2°)으로 그리면 판정(30°)과 어긋났다. `target_kl` 이 0.02 로 박혀 있었는데 진짜는 0.03 | 보상 무게, 성공 문턱, 예산, `target_kl`, 격자 크기 전부 4.1 |
| 5 | **좌표 변환은 한 곳** | 위치 ENU·각도 NED 가 섞여 있어 각도까지 뒤집으면 조용히 틀렸다 | 전부 `map`(z 위)라 변환이 없다. 격자 행 순서만 한 함수. 참·`slam` 자세는 열 이름으로 갈라 섞지 않음 |
| 6 | `pid` 로 "training" 배지 | 없으면 mtime 규칙인데, 한 번에 떨군 실험 폴더가 전부 "학습 중" 으로 떴다 | 5.7. pid 재사용도 거름 |
| 7 | 보관 개수는 **근거와 함께**, 못 박은 것은 **따로** 센다 | 좌석 6 개뿐인 `ace` 궤적이 자가대전에 밀려 먼저 사라졌다. 착취자를 더하며 Main 몫을 10 → 6 으로 줄여 총량을 줄였다(450 → 318 개) | 9절. 스킬·결과별로 나눠 센다 |
| 8 | 비율은 **모집단**으로, 표본은 ⚠ 표시 | 이터당 잘린 표본이 격추 0.1 % 를 53 % 로 셌다 | 4.2 모집단, 4.3 은 환경 고정 표본 |
| 9 | 재개로 **되감긴 줄**을 걷어내고 진행도는 학습기가 센 스텝으로 | 줄 수 × 롤아웃으로 잡으니 32 % 부풀었다(실측 25.6 억 대 19.3 억) | 쓰는 쪽 trim + 뷰어 단조 거르기 + `rewound` 표시 |
| 10 | `started` 는 재개마다 바뀐다 | 19 억 스텝짜리가 "2 분 0 초" 로 떴다 | `segments` 덧붙이기, 첫 기록 시각을 하한으로 |
| 11 | **이어 받기** (`from=`, 바이트 오프셋) | 34 MB 를 3 초마다 다시 읽어 응답이 5–35 초 밀렸다 | 5.5. 해석도 들어올 때 한 번만 |
| 12 | 잘랐으면 **잘랐다고** 말한다 | `MAX_ROWS` 로 말없이 자르니 옛 구간이 "원래 저랬다" 로 읽혔다 | 응답마다 `total`·`cap`·`truncated` |
| 13 | 궤적 길이 상한이 판 전체를 덮어야 한다 | 2,500 프레임 상한이 200 s 판을 125 s 에서 끊었다(최근 200 개 중 50 개) | 상한을 `run.json` `t_max` 에서 계산 |
| 14 | 재개하면 판 번호를 이어 센다 | 0 부터 세니 리플레이 파일이 덮이고 짝이 틀렸다 | 4.4 쓰기 규칙 |
| 15 | 못 잰 검사를 "이상 없음" 으로 세지 않는다 | "6 가지에 걸린 것 없음" 이 재 보지도 않은 것을 포함했다 | 6.2 검사는 확인 n / 못 잰 n 을 따로 |
| 16 | 빈 그래프와 고장 난 그래프를 가른다 | 첫 판까지 8 분 빈 화면을 배선 고장으로 의심했다 | "첫 판 약 이터 n" 안내 |
| 17 | 재생 탭은 고른 실행을 **따라오고, 바뀌면 비운다** | 앞 실행의 기체가 남아 "안 따라온다" 로 보였다 | 6.3 |
| 18 | 이름이 틀린 표준 키는 바로잡는다 | `ep_rew_mean` 은 리턴인데 "보상" 으로 읽혔다 | `ep_ret_mean` |
| 19 | 뜻이 이웃한 두 계열은 색이 멀어야 한다 | 판정패 분홍과 피격 빨강이 겹쳐 안 갈렸다 | 성공·실패 종류 색표를 한 곳에 |
| 20 | 기본 선택은 본학습 | 실험·가상 실행이 기본 화면을 가져갈 수 있었다 | `latest.txt`, `lab`·줄기는 후보 아님 |

## 9. 보관과 디스크 예산 (전부 추정)

| 파일 | 크기 | 쓰는 빈도 | 하루 |
|---|---|---|---|
| `progress.jsonl` | 줄당 약 3 KB(키 약 150 개: 스킬 6 × 보상 항목 약 10 + 학습·시간 등) | ≤ 1 줄/s (4.2 합치기) | ≤ 260 MB |
| `episodes.jsonl` | 줄당 약 0.6 KB | `log_envs` 로 맞춰 ≤ 20 줄/s | ≤ 1 GB (1 GB 마다 다음 번호 파일) |
| 리플레이(영상 없음) | 판당 0.25–0.65 MB: 프레임 약 0.7 KB × 200–600 + 지도 변화분 2 KB × 40–120 | 보관 상한까지 | 상한 × 크기 |
| 리플레이(영상) | 판당 +약 1 MB | 평가 줄기만 | |
| `evals/` | 수 KB | 평가마다 | 작음 |

**리플레이 보관 상한 (제안)**

| 무리 | 개수 | 이유 |
|---|---|---|
| 스킬마다 최근 | 20 | 드문 스킬이 흔한 스킬에 밀려 사라지지 않게(교훈 7) |
| 스킬 × 실패 종류마다 | 5 | 꼼수·실패를 보려고 남기는 것이다. 성공만 남으면 볼 것이 없다 |
| 못 박음(평가 줄기, 처음 보는 집·이름) | 60 | 따로 센다 |
| 영상 있는 것 | 30 | 크다 |

- 스킬 6 개, 실패 종류 5 개면 120 + 150 + 60 ≈ 330 개 × 0.5 MB ≈ 165 MB, 영상 30 개 +30 MB. 실행 하나에 약 0.2 GB + 기록 하루 1–1.3 GB.
- 전투기 뷰어에서 사용자가 "너무 많아서 안 보인다" 며 390 개를 비웠다(`my_runlog.py` 주석). 그래서 수를 늘리지 않고 **나눠** 남긴다.
- 지우기는 학습기(보관 상한)와 사람이 한다. 뷰어는 목록에 실행 폴더 크기만 보여 준다.

## 10. 성능 목표 (전부 추정)

| 항목 | 목표 |
|---|---|
| 학습에 주는 부담 | 호스트 동기 0 회 추가. 로그 스레드 CPU 코어 1 개의 10 % 아래. 디스크 쓰기 평균 < 1 MB/s |
| 서버 쉴 때 | CPU < 1 %, 메모리 < 300 MB (실행 5 개 열림) |
| `/api/runs` | 실행 50 개에 < 50 ms |
| 하루치 실행 첫 화면 | 곡선 12 개(키 약 25)에 전송 < 3 MB(gzip), 1 s 안 (LAN) |
| 실시간 지연 | 학습기가 줄을 쓴 뒤 화면 반영 < 2 s |
| 리플레이 열기 | 0.5 MB 에 < 200 ms, 스크럽 60 fps, 지도 변화분 하나 < 1 ms |
| 표 슬라이더 | 100 만 판에서 `/api/table` < 100 ms |

## 11. 단계

| 단계 | 할 일 | 통과 기준 | 필요한 때 |
|---|---|---|---|
| V0 | `trainfmt` 형식 크레이트 + **가짜 실행 생성기**(`fake_run`): 재개 되감기, 반쪽 줄, 줄기, labs, 리플레이까지 흉내. sgview 의 `stream_sim` 같은 쓰임 | 생성기 출력이 형식 검사 통과 | 학습기보다 먼저 |
| V1 | 서버: 실행 목록·메타·이어 읽기·열 저장소. 학습 탭: 카드·곡선(전투기 `plot` 옮김) | 가짜 실행으로 되감기·반쪽 줄·파일 바뀜이 맞게 보임 | G3(첫 PPO) |
| V2 | 성공 표(스킬 × 집 × 완성도) + 슬라이더, 고장 무늬 검사 | 표 숫자가 `jq` 로 센 값과 같음 | G3 |
| V3 | `.trp` 쓰기(학습기 쪽) + 재생 탭: 로봇·궤적·물체 칸·자라는 지도·보상 띠 | 가짜 판의 관절 각을 화면과 숫자로 대조. 지도 변화분 적용 결과가 마지막 전체 격자와 같음 | G4(자라는 지도) |
| V4 | SSE 실시간, 비교 탭(씨앗 띠) | 학습 중 지연 < 2 s, 끊었다 붙어도 줄이 안 빠지고 안 겹침 | G6(FP8 비교) |
| V5 | 영상 칸, 평가 표, FP8 묶음 | 학생 평가 판에서 영상이 프레임과 맞음 | G5(BC) |
| (뒤) | 살아 있는 평가 판 하나를 scenemap 스트림(`MAP_RECT`·`POSE`·`JOINTS`)으로 바로 보기 — sgview `--ingest` 를 붙임 | | 필요하면 |

## 12. 정할 것

| 무엇 | 선택지 | 기본 제안 |
|---|---|---|
| 뷰어 자리 | `training/viewer/` / `src/` 아래 / sgview 안 | `training/viewer/` (학습 도구) |
| 실행 뿌리 | 모델마다(`~/rl_work/runs`, `~/bc_work/runs`) / 하나(`~/map_vla_work/runs`) | 모델마다 + 뷰어가 `--root` 여럿. dagger·rlft 는 학생이라 `bc_work` |
| `args.json` 대 `run.json` | training/README 는 `args.json` | `run.json` 하나에 인자까지. README 는 맞춰 고칠 것 |
| sgview 코드 나누기 | 옮겨 쓰기 / sgview 에서 Rust 모듈·JS 모듈로 빼기 | 옮겨 쓰고, 둘 다 돌면 뺀다 |
| three.js 판 | sgview 의 2021 UMD 빌드 / 새 모듈판(전투기 0.160.0) | sgview 것. 로봇 코드가 거기 맞춰 있다 |
| 그리퍼 값 → 두 관절 | `robot.json` 에 그리퍼 관절 둘(`omx_gripper_joint_1/2`). 값 하나를 나누는 식(부호·배율) | URDF 에서 확인 후 `joint_map` 에 적음 |
| 커리큘럼 이름 | GPU_TRAINING 5.5 의 C0–C2 / POLICY 4.5 의 단계 1–4 | 두 문서를 맞춘다. 뷰어는 `run.json` 의 표를 읽으므로 어느 쪽이든 된다 |
| 완성도 칸 | GPU_TRAINING 5.6(0 / 0–30 / 30–70 / 70–100 / 100 %) / POLICY 8(전부 / 일부 / 빈 / 없음) | 칸 경계를 `run.json` 에. 지도 방식(`none`·`gt`·`slam`)은 열로 따로 |
| 표를 모집단으로 | 학습기가 칸(스킬 × 집 × 완성도)마다 판 수·성공 수를 `progress` 에 실음 / 표본만 | 처음엔 표본(키가 너무 많아짐). 칸 수가 작으면 모집단 키 추가 |
| 영상 | JPEG(로그 스레드) / BC 데이터의 HEVC 파일 위치만 기록 / 없음 | JPEG, 평가 줄기만 |
| 슬롯 정밀도 | f16 / f32 | f16 (16–32 m 에서 1.6 cm 단위, 4.4) |
| 원격 보기 | 127.0.0.1 + SSH 터널 / 0.0.0.0 | 127.0.0.1 기본 |
| std `TcpListener` 대기열 | Rust std 가 `listen` 에 주는 값 | 확인할 것. 작으면 전투기 뷰어처럼 연결 거부가 난다 |

## 13. 구현 상태 (2026-10-04)

코드: `training/viewer/`(서버 `trainview`, 형식 크레이트 `trainfmt/`, 도구 `fake_run`·`csv2run`, 화면 `assets/`). 학습기 쪽: `training/RL/ppo/driver/src/runfolder.rs`, `training/BC/driver/src/runfolder.rs`(각 `main.rs` 에는 기록 경로에 몇 줄만 더함). 확인한 값은 [viewer README](../../training/viewer/README.md) "확인한 것".

| 단계 | 상태 | 비고 |
|---|---|---|
| V0 `trainfmt` + `fake_run` | **됨** | 되감기(trim 없는 재개), 반쪽 줄, 줄기, labs, .trp(프레임·f16 슬롯·MAP_RECT 지도), 평가 표, 씨앗 묶음. `"synthetic": true` |
| V1 서버(목록·메타·이어 읽기·열 저장소) + 학습 탭(카드·곡선) | **됨** | 되감긴 줄 걷기, 반쪽 줄 안 받음, ino·길이로 새 파일 판정(sig), 64 KB 넘으면 gzip, 10 분 안 본 실행 내림 |
| V2 성공 표 + 고장 무늬 검사 | **됨** | 표 숫자 = 직접 센 값. 검사 8 개, 못 잰 검사는 이유와 함께 따로 |
| V3 `.trp` 재생 탭 | **화면은 됨, 쓰는 쪽은 fake_run 뿐** | 로봇(sgview 모델)·참/slam 궤적·손끝·물체 칸·자라는 지도·보상 띠·HUD·시점 4. 진짜 학습기의 판 기록은 남은 일 |
| V4 SSE 실시간 + 비교 탭 | **됨** | 실제 `ppo_run` 짧은 실행으로 화면 갱신 확인. 씨앗 묶음 평균 ± σ 띠 |
| V5 평가 표 | **됨(bc_run)** | `bc_run` 평가마다 `evals/*.json`. 영상 칸은 화면만(쓰는 쪽 없음), FP8 묶음은 키가 오면 그림(지금 학습기는 `fp8/*` 감시 값을 안 냄) |

학습기가 지금 쓰는 것(4절 중):

| 파일 | `ppo_run` | `bc_run` |
|---|---|---|
| `run.json`(schema·kind·group·seed·pid·pid_start·segments·git·config·refs·curriculum) | ✓ | ✓ |
| `latest.txt` | ✓ | ✓ |
| `progress.jsonl`(≤ 1 줄/s 합침, 안 잰 키 뺌) | ✓ 기록 스레드 | ✓ `Run::drain`(끝난 기록 꺼낼 때) |
| 재개 trim + segments | ✓ | — (bc_run 에 재개 없음) |
| `evals/` | — | ✓ |
| `episodes.jsonl`, `replays/*.trp`, `s_<줄기>/` | — | — |

설계와 다르게 한 것:
- 곡선은 **축 하나**다. 6.2 의 "오른쪽 축에 겹침"(보상 항목 + 리턴·성공률, EV + 가치 손실)은 그림을 나눴다(이중 축은 눈금이 서로를 속인다).
- SSE 는 연결마다 한 스레드가 1 초마다 stat 한다(5.8 의 감시 스레드 하나 + 방송 대신). 클라이언트마다 줄 커서가 따로라 다시 붙어도 줄이 빠지거나 겹치지 않는다.
- 실행 찾기는 뿌리 밑 깊이 4 까지(`~/ra_ppoout/g5/t4/on_s1` 같은 지금의 실행 나무를 그대로 `--root` 로 볼 수 있게). 실행 이름 = `<뿌리 이름>/<상대 경로>`.
- PPO 는 `kind` 를 설정의 `"kind"`(없으면 teacher)로, `group` 은 설정의 `"group"`(없으면 설정 파일 이름)으로 정한다. BC 는 DAgger 바퀴가 있으면 `dagger`, 없으면 `bc`.
- 판마다 줄이 없는 학습기라 `rollout/success/<스킬>` 등은 장치 링의 바퀴 합계(모집단)다. DAgger 의 교사 몬 판은 줄기 폴더 대신 `rollout_teacher/*` 키로 갈랐다(판 줄이 없으므로).

남은 일:
1. 장치 쪽 판 기록(판 끝마다 줄, 고른 환경의 프레임 링) → `episodes.jsonl`·`.trp`·`s_eval/`. CUDA·그래프 변경이라 이번에 하지 않음(FP8 작업과 겹침).
2. PPO 장치 기록에 없는 키: `train/explained_variance`, `log/*_dropped`, `reward/<스킬>/<항>`, 접촉 수, `fp8/*` 감시 값.
3. 영상(JPEG) 쓰는 쪽, 큰 progress 첫 읽기의 "읽는 중 n %", 그리퍼 → 두 관절 식(URDF 확인), training/README 의 `args.json` → `run.json` 정리(12절)는 README 에 한 줄 더함.

## 출처

- 전투기 뷰어: `~/aircombat-rl-private/student/viewer/{README.md, 계획.md, 조사자료.md, server.py, static/index.html, static/app.js, static/replay.js, static/style.css, fetch_vendor.py}`, `student/my_runlog.py`, `student/wrappers.py`(`Shaped`), `student/gpu/vec.py`(`EP_COLS`, `REPLAY_COLS`, `seat_bnd`, `exp_replay_env`), `student/gpu/train_gpu.py`(`_meta`).
- sgview: `src/scene_graph/sgview/{README.md, Cargo.toml, build.rs, src/main.rs, assets/index.html, assets/robot/robot.json}`, `src/robot/tools/build_viewer_assets.py`.
- scenemap 스트림: `src/behavior-2026/src/scene_graph/scenemap/{include/scenemap/stream.hpp, src/stream.cpp}`.
- 우리 설계: [GPU_TRAINING.md](GPU_TRAINING.md), [POLICY.md](POLICY.md), [VLA_INPUT.md](VLA_INPUT.md), [training/README.md](../../training/README.md).
