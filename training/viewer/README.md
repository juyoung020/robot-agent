# trainview — 학습 뷰어 (RL 교사 · BC 학생 · DAgger · RL 다듬기)

설계는 [docs/map_vla/TRAIN_VIEWER.md](../../docs/map_vla/TRAIN_VIEWER.md). 학습이 도는 동안 브라우저로 본다. 탭 셋: **학습**(카드·검사·곡선·표), **재생**(판 하나를 3D 로), **비교**(실행 여럿, 씨앗 묶음 띠).

- 서버는 Rust(std `TcpListener` + 연결마다 스레드, `serde_json`·`flate2` — sgview 틀). **읽기 전용**: 실행 폴더의 파일만 읽고 아무것도 쓰지 않는다. 학습 프로세스와 말을 섞지 않는다.
- 화면은 순수 JS(ES 모듈, 빌드 단계 없음). 곡선은 캔버스에 직접 그린다. 3D 는 sgview 의 `three.min.js`·`OrbitControls.js`·`GLTFLoader.js`·로봇 GLB 를 `build.rs` 가 바이너리에 같이 넣는다(sgview 파일은 읽기만, 두 벌 두지 않음). 인터넷 없이 돈다.
- 쓰는 쪽(학습기)과 읽는 쪽이 같이 쓰는 형식은 `trainfmt/` 크레이트 한 곳에만 있다(TRAIN_VIEWER 4.6).

## 빌드·실행

```bash
cd training/viewer && cargo build --release -j 4
target/release/trainview --root ~/ra_ppoout --root ~/ra_bc/runs [--port 7810] [--bind 127.0.0.1]
```
- `--root` 는 여러 개. 뿌리 밑 깊이 4 까지에서 `run.json` 또는 `progress.jsonl` 이 있는 폴더가 실행이다(실행 폴더 안으로는 내려가지 않음). 실행 이름 = `<뿌리 폴더 이름>/<상대 경로>`.
- 기본 bind 는 127.0.0.1. 원격은 SSH 터널 또는 `--bind 0.0.0.0`.
- 기본으로 여는 실행: 뿌리의 `latest.txt`(가장 최근에 바뀐 것). `labs/` 밑 실행은 후보가 아니다.

## 실행 폴더 규약 (학습기가 쓴다 — TRAIN_VIEWER 4절)

```
<root>/latest.txt                                  기본으로 열 실행 이름 (학습기가 시작할 때 씀)
<root>/<이름>/run.json                             설정·상수(schema 1): kind, group, seed, pid, pid_start, segments, git, config 전부, refs(기준선) …
              progress.jsonl                       업데이트마다 → ≤ 1 줄/s 로 합침(time/iterations_merged), 모집단 값
              episodes.jsonl                       판마다 한 줄(표본, log_envs 환경의 모든 판)
              replays/ep_<판>_<스킬>_<결과>.trp      판 하나 궤적(바이너리, trainfmt::trp)
              evals/<이름>.json                     평가 표 {iter, ckpt, rows:[{eval, split, n, success, collision, timeout, ref, pass}]}
              s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/   집계에 섞으면 안 되는 판(평가, DAgger 교사 판)
```
- 덧붙이기(progress·episodes)와 이름 바꾸기(run.json·evals·replays: `.tmp` 에 쓰고 rename)만 한다.
- **잰 것만 쓴다**: 그 업데이트에 끝난 판이 없으면 비율 키를 **뺀다**(0·null 을 쓰지 않음).
- 키 이름은 **표준 RL 기록 이름**(Stable-Baselines3 꼴 `<묶음>/<이름>`)이다 — 아래 "용어" 표. 예: `time/{iterations,total_timesteps,time_elapsed,fps}`, `rollout/{ep_rew_mean,ep_len_mean,success_rate,n_episodes,collision_rate,timeout_rate}`, `train/{approx_kl,clip_fraction,entropy_loss,explained_variance,value_loss,policy_gradient_loss,learning_rate,std,log_std,grad_norm,loss}`, `eval/{success_rate,mean_ep_length,…}`, `eval_teacher/*`, `rollout_teacher/*`(교사가 몬 판), `curriculum/*`, `dagger/*`, `reward/<스킬>/<항>`, `fp8/*`, `gpu/mem_used_mb`, `log/*_dropped`. 스킬이 하나면 `rollout/success_rate`, 여럿이면 `rollout/success_rate/<스킬>`.
- 재개(`--resume`): 같은 폴더에 이어 쓰고 첫 줄을 쓰기 전에 그 이터 이상의 줄을 지운다(trim). `segments` 에 `{started, pid, from_iter, from_steps, trimmed_lines}` 를 덧붙인다. 뷰어도 env_steps 가 줄어든 줄을 따로 걷어낸다(`rewound`).
- "학습 중" 배지: `pid` 가 살아 있고 `/proc/<pid>/stat` 시작 시각이 `pid_start` 와 같을 때만. `ended` 가 있으면 끝난 실행.

### 지금 학습기가 내는 것

| 학습기 | run.json | progress.jsonl | evals | episodes / replays |
|---|---|---|---|---|
| `ppo_run` (`training/RL/ppo/driver/src/runfolder.rs`) | ✓ (kind teacher, 커리큘럼 단계, refs: target_kl·max_grad_norm) | ✓ 기록 스레드에서, 바퀴 합계 → 1 줄/s | — | — (장치 기록 링이 바퀴 합계만 담음) |
| `bc_run` (`training/BC/driver/src/runfolder.rs`) | ✓ (kind bc/dagger, teacher) | ✓ 기록 꺼낼 때(`Run::drain`), 단계가 바뀌면 줄을 끊음 | ✓ 평가마다(앞 eval_drop 롤아웃 뺀 표) | — |

- 둘 다 기존 `log.csv`·`events.txt`·`results.json`·체크포인트는 그대로 쓴다. CUDA·그래프 코드는 건드리지 않았다. 설정에 `"runfolder": false` 면 끈다. `"group"`(없으면 설정 파일 이름), `"kind"`, `"progress_every_s"`(기본 1) 로 바꿀 수 있다.

## 도구

| 바이너리 | 하는 일 |
|---|---|
| `trainview` | 서버 |
| `fake_run --root DIR [--live S] [--only-live]` | 가짜 실행(설계 11절 V0): 씨앗 묶음(bf16/fp8 × 3), 되감기(trim 안 한 재개), 끝의 반쪽 줄, 줄기(s_eval, s_teacher), labs, 리플레이(.trp: 프레임·슬롯·자라는 지도), 평가 표. `--live S` 는 pid 있는 실행 하나를 S 초 동안 초마다 덧붙임. run.json 에 `"synthetic": true` → 화면에 synthetic 배지 |
| `tools/record_replay/`: `record_ppo`, `record_bc`, `scene_b1k` (C++/CUDA) | **진짜 판 궤적**: 체크포인트를 돌려 `.trp` 를 실행 폴더 `s_eval/` 에 쓴다(아래 "재생 기록"). BEHAVIOR 장면 배치(RASC)도 프레임 하나짜리 `.trp` 로 |
| `csv2run --out DIR <실행 폴더>…` | 규약 이전 실행(`log.csv` + `config.json` + `events.txt`/`results.json`)을 규약 폴더로 옮김(원본은 읽기만). 키 이름은 runfolder.rs 와 같다 |

## 재생 기록 (record_replay) — 진짜 판을 진짜 장면에서

학습기와 따로 도는 독립 도구다. 학습기 코드·CUDA 그래프는 고치지 않고 **공개 헤더로만** 쓴다: `ppo::Trainer`(trainer.h)·`bc::Bc`(bc.h)를 그래프 없이 만들어 `rollout_step` 을 한 스텝씩 부르고, 스텝마다 `DeviceEnv::download`·`DeviceMap::download` 로 환경·지도를 내려받는다(ppo_verify eval 의 충돌 다시 보기와 같은 방식). 끝 프레임은 같은 `env.h` 를 호스트에서 한 스텝 다시 돌려 얻는다(장치 상태는 이미 새 판으로 리셋돼 있으므로). `.trp` 바이트는 `trpc`(Rust `trainfmt::trp` 의 C ABI 정적 라이브러리)가 쓴다 — 형식 정의는 trainfmt 한 곳.

```bash
cmake -S training/viewer/tools/record_replay -B ~/ra_recbuild [-DTRAIN_SRC=<학습기 소스 training/>] && cmake --build ~/ra_recbuild -j 4
~/ra_recbuild/record_ppo --ckpt ~/ra_ppoout/g5/t4/on_s1/ckpt_final.bin --out <실행 폴더> [--episodes 8] [--track 8] [--map 0.2 0.6] [--stage 2] [--stochastic]
~/ra_recbuild/record_bc  --student ~/ra_bc/runs/img_s1_cont/student_dagger4.bin --out <실행 폴더> [--episodes 8] [--no-images]
~/ra_recbuild/scene_b1k  --out <실행 폴더> ~/ra_b1k/*.rasc
```
- 설정(`use_map`, `goal_from_map`, 영상·글·flow 머리 등)은 체크포인트 옆 `config.json` 에서 읽는다. 기본은 결정적 정책(log σ → −12), 처음 지도 C0 20 %·C1 60 %·C2 20 %(`--map 0 0` = 모두 빈 지도).
- `TRAIN_SRC`: 신경망·관측 배치가 바뀌면 옛 체크포인트가 새 소스에 맞지 않는다(2026-10-04 v2 변경 — `bc_load_teacher` 가 거절). 그 체크포인트를 만든 커밋의 학습기 소스를 `git archive <커밋> training/RL training/BC` 로 내보내 `-DTRAIN_SRC` 로 준다(읽기만). 이번 기록은 343492d 판(v2 전)으로 했다.
- 판마다 `.trp` 에 담는 것: 프레임 열 39(참 자세·slam 자세(지도의 믿는 자세)·속도·바퀴 각·팔 6·손끝(순기구학)·행동 8·보상·누적·가치·사건·목표·지도 완성도), 물체 기억 칸 16(믿는 위치·참 위치·크기·지금 봄/기억·불확실도·상태·확정), 자라는 지도(로그 오즈 → MAP_RECT %, 바뀐 사각형만), 지도가 뽑은 벽 선분(`segs` 섹션), 머리 `scene`(방 벽 + 지도의 참 상자: 가구·작은 물건·컵), 영상 학생은 학생이 본 카메라 2 장(256², 2 Hz, 기준 JPEG — 도구 안의 작은 부호기). 판 줄은 `s_eval/episodes_eval.jsonl`.
- BEHAVIOR(`scene_b1k`): 벽·문·창·가구 상자(yaw), 방 범위·이름, 다닐 곳 격자(TRAV_NO_OBJ), 과제 인스턴스의 로봇 시작 자세와 옮길 과제 물체. 로더는 `training/RL/tools/b1kconv/cpp/rasc.h`(읽기만).
- `.trp` 에 더한 것(되돌림 호환): 덧붙인 섹션 `[u32 frame][u32 len][바이트]`(지금 `segs`), 머리 `scene`·`slot_z`("center")·`source`. 옛 뷰어는 모르는 섹션·머리 칸을 건너뛴다.

## 용어 (표준 이름) — 2026-10-04 바꿈

화면 글자와 progress 키를 표준 RL 도구의 이름으로 바꿨다. 한국어 설명은 마우스를 올리면 나오는 풍선(title)으로 남겼다. 옛 실행 폴더는 서버가 읽을 때 옛 키를 새 키로 바꾼다(`trainfmt::keys::canon` — 되돌림 호환, `train/entropy` 는 부호를 바꿔 `train/entropy_loss`).

| 옛 키 / 화면 말 | 새 키 / 화면 말 | 출처 |
|---|---|---|
| `time/iter` | `time/iterations` | SB3 logger |
| `time/env_steps` | `time/total_timesteps` | SB3 logger |
| `time/wall` | `time/time_elapsed` | SB3 logger |
| `time/fps_env` | `time/fps` | SB3 logger (CleanRL `charts/SPS`, rl_games `performance/step_fps`) |
| `time/iters_in_row` | `time/iterations_merged` | (우리 것, SB3 꼴) |
| `rollout/ep_ret_mean` ("리턴") | `rollout/ep_rew_mean` ("Episode return") | SB3 logger (CleanRL `charts/episodic_return`, RLlib `env_runners/episode_return_mean`) |
| `rollout/success/<스킬>` | `rollout/success_rate[/<스킬>]` | SB3 logger `rollout/success_rate` |
| `rollout/n_eps` ("판 수") | `rollout/n_episodes` ("Episodes") | SB3 `time/episodes` 꼴 |
| `rollout/step_rew_mean` | `rollout/step_reward_mean` | rl_games `rewards/step` 꼴 |
| `train/entropy` | `train/entropy_loss` (= −엔트로피) | SB3 logger |
| `train/clip_frac` | `train/clip_fraction` | SB3 logger (CleanRL `losses/clipfrac`) |
| `train/policy_loss` | `train/policy_gradient_loss` | SB3 logger |
| `train/lr` | `train/learning_rate` | SB3 logger |
| `train/log_std_mean` ("탐색 log σ") | `train/log_std` ("Policy log std"), `train/std` 더함 | SB3 logger `train/std` |
| `train/adv_std`, `train/adv_mean` | `train/advantage_std`, `train/advantage_mean` | PPO 논문 "advantage" |
| `bc/loss`, `bc/flow_loss`, `bc/end_bce` | `train/loss`, `train/flow_loss`, `train/end_bce` | SB3 `train/loss` |
| `eval/student/success/<스킬>`, `…/ep_len_mean` | `eval/success_rate`, `eval/mean_ep_length` | SB3 logger `eval/*` |
| `eval/teacher/*` | `eval_teacher/*` | (SB3 꼴) |
| `dagger/disagree` | `dagger/action_mse` | (뜻대로) |
| `data/samples` | `dagger/dataset_size` | (뜻대로) |
| `curr/*` (`stage_idx`, `success/C0`, `stage_frac/C0`, `goal_known` …) | `curriculum/*` (`stage`, `success_rate/C0`, `start_map_fraction/C0`, `goal_known_rate` …) | (SB3 꼴) |
| 판 | episode | 모든 도구 |
| 모집단 / 표본 | all episodes (rollout stats) / logged sample | SB3 `rollout/` |
| 줄기 | split (`main` / `eval` / `teacher`) | — (폴더 이름 `s_<split>/` 은 그대로) |
| 고장 무늬 검사 | Training health checks (Reward hacking, Entropy collapse, KL too high, …) | — |
| 학습 / 재생 / 비교 탭 | Training / Replay / Compare | — |
| EMA | Smoothing (기본 0.6) | TensorBoard·W&B |
| x 축: iteration / env_steps / 벽시계 | X-axis: `time/iterations` / `time/total_timesteps` / relative time | TensorBoard STEP·RELATIVE·WALL |
| 시점 커서 | Step cursor | — |
| 씨앗 묶음 띠 | Group runs (mean ± std) | W&B grouping |
| 비교 표 | Run comparer | W&B Run comparer |
| 성공 표 | Success rate table (logged sample) | — |
| 평가 표 | Evaluation (`evals/*.json`) | SB3 `eval/` |

출처: [SB3 logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html), [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/), [RLlib new API stack](https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html), [rl_games `a2c_common.py`](https://github.com/Denys88/rl_games) (Isaac Lab 기본 학습기), [W&B line plots](https://docs.wandb.ai/models/app/features/panels/line-plot/reference)·[smoothing](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing), [TensorBoard 가로축·평활](https://datahacker.rs/tensorboard-visualizing-learning/), PPO 논문(Schulman 2017: clipped surrogate objective, advantage).

## 경로 (서버)

| 경로 | 돌려주는 것 |
|---|---|
| `/api/runs` | 실행 목록(id, kind, group, streams, live, age, iters, env_steps, eps, replays, evals, stage, first_ts, rewound, disk_mb) + latest |
| `/api/meta?run=` | run.json 그대로 |
| `/api/progress?run=` | 키 목록(키마다 처음·마지막 줄, 개수) |
| `/api/progress?run=&keys=a,b\|*&from=<줄>&sig=&max_points=` | 고른 키를 열로(x: `iterations`, `total_timesteps`, `time_elapsed`, `ts`). 없는 값 null. 옛 키는 새 이름으로. `max_points` 보다 길면 칸마다 마지막 + `lo`/`hi`(최소·최대). `sig` 가 다르면 처음부터 |
| `/api/episodes?run=&stream=&from=` | 줄 그대로(최대 20,000, `truncated`·`cap`·`total`, `ret_mismatch` = ret ≠ Σr 줄 수) |
| `/api/table?run=&stream=&rows=skill\|home\|stage\|map_mode\|driver&cols=bin\|…&last=N&home=&stage=&upto=<env_steps>` | 칸마다 최근 N 판의 n, 성공, SPL, 충돌, 시간 초과, 평균 t, 접촉, 리턴 |
| `/api/replays?run=&stream=`, `/api/replay?…&id=` | .trp 머리 목록 / 바이트 그대로 |
| `/api/evals?run=`, `/api/events?run=` | evals/*.json 모음 / events.txt 끝 200 줄 |
| `/api/live?runs=a,b&from=&sig=&eps=&rep=` | SSE: `progress`(새 줄, 모든 키), `reset`, `episodes`, `replay`, `runs`, 10 초 keepalive. 끊기면 화면은 3 초 폴링 |

- 이어 읽기: 파일마다 (ino, 읽은 바이트). 줄끝까지 온 줄만 받는다(반쪽 줄은 다음에). ino 가 바뀌거나 짧아지면 처음부터(`sig` 바뀜). progress 는 들어올 때 한 번 해석해 열(f32)로 쌓는다. 10 분 안 본 실행은 내린다.
- 이름 검사: 실행은 찾은 목록에 있는 id 만, 줄기·파일 이름은 `[\w.=-]+`.
- 64 KB 넘는 응답은 gzip(수준 1).

## 화면

- **학습**: 카드(이터·스텝, 처리량·경과, 단계·섞임, 스킬별 성공 막대, 리턴·판 길이, 판 수 모집단/표본, 탐색 σ, KL·EV, GPU 메모리, 로그 버림, SR·SPL 표본, 평가 학생/교사), 고장 무늬 검사 8 개(못 잰 것은 이유와 함께 따로), 곡선 묶음(`chartsSpec()` 한 곳, 없는 키는 "not logged: …"), x 축(iteration/env_steps/벽시계), EMA(점 60 개 미만인 드문 값은 평활 안 함), 시점 커서(카드·검사·표·재생 목록이 그 시점까지), 성공 표(스킬 × 완성도 칸 등, 칸마다 최근 N 판 슬라이더, n < 20 흐리게), 평가 표, events.txt, run.json.
- **재생**: 목록(스킬·결과·집 거르기, 커서 근처), 로봇(sgview 모델 + `joint_map`, 바퀴는 vx 적분), 참/slam 궤적, 손끝 궤적, 물체 칸(보는 중 꽉 참 / 기억 반투명, 불확실도 원, 참 위치 유령 + 선, 목표 칸 빨간 테, 상태 색), 자라는 지도(MAP_RECT 를 프레임 따라, 50 프레임마다 사본), 목표 표시, 보상 띠(항목 쌓은 막대, 가치·끝 신호 선, 사건 눈금, 누르면 그 프레임), HUD, 시점 1–4(숫자 키), 재생·속도·한 프레임(←/→, 스페이스). 영상 칸은 .trp 에 img 섹션이 있으면.
- **비교**: 실행 체크 목록(group 별), 지표 두 개, x 축(env_steps / 처음부터 시간 / 벽시계 / iteration), 같은 group 은 평균 ± σ 띠(겹치는 구간 200 칸 보간), 한쪽에만 있는 키는 "없음", 비교 표(마지막 10 % 줄 평균).
- 곡선은 축 하나다(이중 축 안 씀 — 단위가 다르면 그림을 나눈다). 색은 고정 순서 8 색(dataviz 기본 팔레트), 밝음·어두움 둘 다.

## 확인한 것 (2026-10-04)

| 무엇 | 결과 |
|---|---|
| `ppo_run` 짧은 실행(N 1,024, 1 분, `ppo_a0a1.json` 바탕) | run.json · progress.jsonl 60 줄(1,548 바퀴 → 1 줄/s), log.csv 1,549 줄 그대로. 화면에 "training" 배지, A0 → A1 단계, SSE 로 6 초 동안 줄 36 → 41 |
| 재개(`--resume ckpt_000802.bin`, 12 초) | progress 줄 30 개 trim, segments 에 from_iter 803 덧붙임, 이터 단조, 뷰어 sig 바뀜 → 처음부터 다시 받음 |
| `bc_run` 짧은 실행(lite, N 1,024, DAgger 3 바퀴, 3.3 s) | progress 19 줄(교사 기록 → `rollout_teacher/*`, 학생 모으기 → `rollout/*`, 평가마다 `eval/*` 한 줄), evals 5 개, log.csv·results.json 그대로 |
| 성공 표 | `/api/table` 의 pick × 완성도 칸 (n, 성공) 이 파이썬으로 직접 센 값과 같음 |
| 되감기·반쪽 줄(fake_run) | 1,200 줄 중 199 줄 걷어냄(`rewound`), 반쪽 줄은 안 받음 |
| 옮긴 실행(csv2run, A2 토큰 켬/끔 씨앗 2 개씩) | 비교 표 마지막 10 % 성공률 0.949 / 0.910 / 0.612 / 0.718 — PPO README 의 끝 값(0.950 / 0.912 / 0.618 / 0.720)과 맞음 |
| 경로 막기 | `id=../../x.trp` → 400, 없는 실행 → 404 |

### 재생 기록 확인 (2026-10-04)

| 무엇 | 결과 |
|---|---|
| `record_ppo` A2 토큰 켬 씨앗 1 교사(`g5/t4/on_s1/ckpt_final.bin`, 343492d 소스) | 처음 지도 섞음 8 판 성공 8, 모두 빈 지도(C2) 12 판 성공 11·충돌 1 — 판당 70–126 프레임, 100–150 KB. 화면: 방 벽·가구·컵, 자라는 지도, 지도 벽 선분, 물체 칸, 참/slam 궤적 |
| `record_bc` G5 영상 학생(`img_s1_cont/student_dagger4.bin` = DAgger 8 번 0.946) | 8 판 성공 8, 카메라 JPEG 2 장 × 2 Hz(판당 240–620 KB). JPEG 은 `file` 로 baseline 256×256 확인, 화면에 표시 |
| `scene_b1k` BEHAVIOR 7 장면 | 상자 59–556, 방 2–21, 격자 83²–863², 17–793 KB |
| 옛 키 실행(smoke) | 서버가 새 이름으로 바꿔 냄(`rollout/ep_rew_mean`, `train/entropy_loss` 부호 등) |

## 남은 일

- **학습 중 판마다 기록(episodes.jsonl)·리플레이**: 학습기 안에서 하려면 장치 기록 링 변경이 필요하다(CUDA·그래프). 지금 진짜 판 궤적은 학습 뒤 `record_replay` 로 체크포인트를 돌려 얻는다(평가 줄기 `s_eval`). 학습 판 표본의 성공 표는 아직 fake_run 자료로만 확인.
- 보상 항목별 값(`r_<항>`): 환경 스텝이 합만 내서 `.trp` 에는 `r_total` 하나. 영상 학생의 `end_p`·가치 없음.
- BEHAVIOR 장면에서 로봇이 도는 판은 E2(환경을 RASC 로) 뒤에.
- PPO 의 `train/explained_variance`, `log/*_dropped`(장치 링 넘침 수), 보상 항목별 합(`reward/<스킬>/<항>`), 접촉 수는 장치 기록에 없어 키가 없다(화면은 "not logged").
- 영상 칸(img 섹션 JPEG): 화면은 읽지만 쓰는 쪽이 없다.
- `progress` 첫 읽기를 64 MB 씩 여러 요청에 나눠 "읽는 중 n %" 표시(지금은 한 요청 안에서 64 MB 씩 끝까지 읽음).
- SSE 는 연결마다 감시 스레드가 1 초마다 stat 한다(설계의 "감시 스레드 하나 + 방송" 대신 — 보는 사람 1–3 명이라 같은 비용, 연결마다 커서가 따로라 줄이 빠지거나 겹치지 않음).
- 그리퍼 값 → 두 관절 식(부호·배율)은 URDF 확인 전 가정(`joint_map` qg → `omx_gripper_joint_1` ×1, `_2` ×−1).
