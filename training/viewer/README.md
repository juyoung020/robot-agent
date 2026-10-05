# trainview — 학습 뷰어 (RL 교사 → 학생/VLA)

설계는 [docs/map_vla/TRAIN_VIEWER.md](../../docs/map_vla/TRAIN_VIEWER.md). 학습이 도는 동안 브라우저로 본다. 탭 셋: **Training**(카드·건강 검사·곡선·표), **Replay**(판 하나를 sgview 화면으로), **Compare**(실행 여럿, 씨앗 묶음 띠).

- 서버는 Rust(std `TcpListener` + 연결마다 스레드, `serde_json`·`flate2` — sgview 틀). **읽기 전용**: 실행 폴더의 파일만 읽는다.
- 화면은 순수 JS(ES 모듈, 빌드 단계 없음). 곡선은 캔버스에 직접. Replay 는 **진짜 sgview**(`src/scene_graph/sgview`, 실시간에 쓰는 같은 바이너리)를 세션마다 프로세스로 띄우고 iframe 으로 보여 준다(서버가 같은 출처로 역프록시). 화면은 sgview 그대로 — 재생 조종·판 목록·덧그림·입력 패널은 그 바깥(이 폴더)에서 더한다. 흉내 코드 없음.
- 쓰는 쪽(학습기)과 읽는 쪽이 같이 쓰는 형식은 `trainfmt/` 크레이트 한 곳(실행 폴더·progress 키·`.trp`·재생 자동 기록 hook).

## 빌드·실행

```bash
cd training/viewer && cargo build --release -j 4
target/release/trainview --root ~/trainview_work/smoke_v2 --root ~/trainview_work/behavior_og --root ~/trainview_work/behavior --root ~/trainview_work/imported --root ~/trainview_work/fake --port 8095
```
- `--root` 여러 개. 뿌리 밑 깊이 4 까지에서 `run.json` 또는 `progress.jsonl` 이 있는 폴더가 실행(이름 = `<뿌리 이름>/<상대 경로>`). 기본 bind 127.0.0.1(원격은 SSH 터널 또는 `--bind 0.0.0.0`).
- `--archive-before <커밋>`: 이 커밋의 자손이 아닌 git 커밋으로 돈 실행은 보관함(기본 = "관측·신경망 v2" 커밋을 git log 로 찾음).

## 화면

- **실행 고르기(머리)** — 학습에는 모델이 둘이다: RL 교사(PPO, 특권 입력 — 시연을 만듦)와 학생/VLA(BC·DAgger, 뒤에 RecallVLA — 결과물).
  - `pipelines`: 학생마다 한 줄 "학생 ← 교사". 학생이 주인공(eval SR), 교사는 핵심 건강만(SR·충돌·판 길이)과 학생이 쓴 교사 체크포인트(run.json `teacher` → 그 체크포인트가 든 실행). 교사를 누르면 교사 상세.
  - `teachers`(학생이 아직 없는 교사), `BEHAVIOR`(OmniGibson LIMO 탐사 판·집 배치), `archive`(v2 이전 코드·csv 로 옮긴 옛 실행 — 기본 숨김), `test data (synthetic)`(fake_run 가짜 자료·옛 kind "lab" — 기본 숨김, 보이면 빨간 띠 "가짜 시험 자료 — 실제 학습 결과 아님").
  - **학습 상태 불**(목록·파이프라인 양쪽·머리줄, SSE `status` 로 2 초마다): training 초록(숨쉼, pid·시작 시각이 맞고 60 s 안에 기록), stalled 노랑(살아 있는데 60 s 넘게 기록 없음), finished 회색(run.json `ended`), crashed 빨강(프로세스가 없는데 `ended` 없음), unknown(pid 없는 옛 실행). 풍선 = 마지막 기록 시각·이터. **REC** = 재생 자동 기록 뒷 프로세스가 도는 중(`s_eval/recording.json`, pid 확인).
- **Training**: 카드, Training health checks 8 개(못 잰 것은 이유와 따로), 곡선 묶음(`chartsSpec()` 한 곳, 없는 키는 "not logged"), X-axis(time/iterations · time/total_timesteps · relative), Smoothing, Step cursor, 성공 표(표본), Evaluation, Events, run.json.
- **Replay** = sgview 화면 그대로: 점유 지도(기운 벽 θ 격자), 하늘색 2D 벽, 물체 세그먼트 점구름(true colour)·이름표·썸네일, 장면 그래프 층(물체·방 노드와 간선), LIMO URDF(관절), SLAM 궤적 + GT 궤적(초록). 시간 막대·재생·속도·0.5 s 걸음, 체크포인트 고르기·학습 진행 막대(자동 기록 판), 몸통 카메라 그림(OmniGibson 판), 켜고 끔: `house layout`(방 바탕 층 — 방 바닥 색·벽·가구(풍선에 종류)·놓을 곳·집을 것·과제 물체), `policy map`(정책이 본 지도 = GPU 근사판 G2, 주황 겹침 — "진짜 scenemap" 과 차이).
  - 예전 3D 그리기는 `?debug=1` 에서만(`3D classic`) — 판마다 보상 띠와 `.trp` 안 카메라 JPEG 를 보는 데만 남김.
- **Compare**: 지표 둘, X-axis(time/total_timesteps · relative · wall clock · iterations), Group runs(같은 group = 평균 ± 표준편차), Run comparer(마지막 10 % 줄 평균).

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


## 재생 판 — 어디서 오나

| 판 | 만드는 것 | 내용 |
|---|---|---|
| GPU 환경 판(RL 교사·BC 학생) | `tools/record_replay` 의 `record_ppo` / `record_bc` (C++/CUDA). **학습 중 체크포인트마다 자동**(아래) 또는 손으로 | 체크포인트를 돌려 판 K 개. 판이 끝나면 그 입력으로 **진짜 scenemap**(limo_omx C ABI)을 처음부터 돌림: 스텝마다 proprio 12, 정책 지도 keyframe 마다 깊이 640×400 + id 버퍼(정책 지도와 같은 장면 상자를 같은 카메라에서 CPU 광선 추적 — `training/RL/map_cmp` 와 같은 방법) + 면 음영 RGB, 완벽한 검출 → scenemap 의 sgview 스트림·메모리(점구름 PLY)를 `<판>.sg/` 에, 정책이 본 지도는 `<판>.sg/episode.trp`. (팀 RenderBatch 는 RGB·깊이만 내고 id 버퍼가 없어 마스크를 못 만든다) |
| OmniGibson LIMO 탐사 | `tools/og2sg` (C++) | sgrt 기록 `rec.bin` → scenemap 재실행 → 스트림·메모리·몸통 카메라 JPEG·GT 궤적. `tools/run_explore_live.sh` 를 `TRAINVIEW_OG=1` 로 돌리면 판 끝에 자동(→ `~/trainview_work/behavior_og/<판>`) |
| BEHAVIOR 집 배치 | `og2sg --layout --rasc <장면>.rasc` | 다닐 곳 격자(집만)·방 노드·과제 첫 인스턴스 시작 자세·바탕 층(RASC v3) |

- 재생 서버(`src/sg.rs`): 세션(쿠키 `sgsess`)마다 `sgview <memory> --ingest` 한 개 + `sgs_play --ctl`(보내는 쪽, behavior-2026 `tools/realbag`) 한 개를 띄우고, 시간 조종(seek·재생·속도)은 sgs_play 표준입력으로(seek = 새 ingest 연결 → sgview 가 reset + 기록 앞부분을 다시 받음). 동시 2 세션, 90 s 안 쓰면 끔, 서버 시작 때 남은 프로세스는 `~/trainview_work/run/sgview_pids` 로 정리. `.trp` 만 있는 GPU 환경 판은 같은 스트림 형식(SGS1)으로 바꿔 같은 길로 재생(점구름 없는 물체는 상자).
- 짝(과제 → 장면): turning_on_radio = house_double_floor_lower, bringing_water = house_single_floor.

## BEHAVIOR 집기·놓기 판 (2026-10-05)

| 무엇 | 출처 | 화면 |
|---|---|---|
| `.trp` 판(record_bc, stage 3) | GPU 환경. 장면 머리 = 벽·문·창·가구(RASC 종류 이름)·집을 물체·놓을 곳 + `world`(장면·창 가운데·과제·인스턴스) + `pnp`(잡기 모형 0.06 m·질량 대 가반 하중·면 높이 대 팔 닿는 띠·잡기 가능 표·서는 자리 후보) + `picks`. 스텝마다 `inputs` 섹션(지도 토큰·관측 80·X0 일부·학생 μ·대본 교사 라벨·낸 행동·특권 상태, `beh_rec.h`) | 목록 "GT / policy map" — **우리 인지 아님** |
| `<판>_og.sg` | `tools/og_replay`: 같은 궤적·시작 자세·집을 물체를 OmniGibson 에서 다시 돌림 → LIMO eyes 640×480 RGB-D → libsgrt(ObjectSAM yolo26n-seg-obj-416 + SigLIP 2 + scenemap objprob). `episode.trp` = 정책 기록(창 → 세계로 옮긴 지도) | 목록 "REAL pipeline", 정책 지도(주황) 겹침 기본 켬 |
| 집 메시 | `tools/scene_mesh/export_mesh.py`(fastsam `scene_mesh.npz` → 줄인 `~/trainview_work/scene_mesh/<장면>.smsh`, `/api/scene_mesh`) | "house mesh" (GT) |

- 자동: record_bc 가 체크포인트마다 학생·교사 앞 판 `--og-per`(기본 2)개를 `~/trainview_work/og_queue` 에 넣고 일꾼 `og_queue.sh` 를 띄움(하나만, nice 19, og.lock, 램·디스크·GPU 여유 검사, GPU 빈 메모리 < 600 MB 면 OG 쪽을 끄고 줄 뒤로).
- 손으로: `tools/og_replay/og_replay.sh <판.trp>`.
- 재생 탭: 오른쪽 Inputs 패널(재생 시각에 맞춤, 과제 조건은 특권으로 따로), 덧그림 토글 house mesh · env boxes · stances & reach · graspable tint. 반복 재생·같은 실행의 다른 판에서 사람이 맞춘 시점 유지.

## 학습 중 재생 판 자동 기록

`trainfmt::replay_hook` — `ppo_run`·`bc_run` 의 실행 폴더 쓰개(`runfolder.rs`)가 체크포인트를 쓴 뒤(PPO `ckpt_*.bin`, BC 평가 직전의 `student_*.bin`) 부른다. 학습기 C++/CUDA 는 그대로.
- `setpriv --pdeathsig KILL -- nice -n 19 record_{ppo,bc} --tag it<이터> --episodes K --keep-fail F --n-env N --split eval` 뒷 프로세스. 하나씩만 — 앞 것이 돌면 이번 것은 건너뜀(run.json `replays.skipped_busy`). 학습기가 죽으면 같이 죽음. 끝 체크포인트는 학습이 끝난 뒤 기록(최대 120 s 기다림).
- 설정 `"replays": false` 면 끔, `{"episodes": 4, "failures": 2, "n_env": 64, "every": 1, "recorder": 경로}`. 도구 경로: 설정 → `TRAINVIEW_RECORD_PPO`/`TRAINVIEW_RECORD_BC` → `~/ra_recbuild/record_{ppo,bc}`(없으면 끔). 도구는 학습기와 **같은 소스 나무**로 빌드(신경망·관측 배치가 같아야 체크포인트가 읽힘).
- 결과: `s_eval/replays/ep_<n>_<체크포인트>_<스킬>_<결과>.sg`, 판 줄 `s_eval/episodes_eval.jsonl`(`ckpt`, `ckpt_iter`), 도구 출력 `s_eval/record.log`.

```bash
cmake -S training/viewer/tools/record_replay -B ~/ra_recbuild [-DTRAIN_SRC=<학습기 소스 training/>] && cmake --build ~/ra_recbuild -j 4
~/ra_recbuild/record_ppo --ckpt <ckpt> --out <실행 폴더> [--episodes 8] [--keep-fail 2] [--map 0.2 0.6] [--stage 2] [--tag it000200] [--no-sg]
~/ra_recbuild/record_bc  --student <student.bin> --out <실행 폴더> [--no-images]
cmake -S training/viewer/tools/og2sg -B ~/ra_og2sg && cmake --build ~/ra_og2sg -j 4
~/ra_og2sg/og2sg --rec <rec.bin> --run <OG 실행 폴더> --run-out ~/trainview_work/behavior_og/<이름> --rasc ~/ra_b1k/<장면>.rasc
```
- 옛 체크포인트(v2 이전)는 그때 소스로: `git archive <커밋> training/RL training/BC | tar -x -C DIR` → `-DTRAIN_SRC=DIR/training`(A2 교사·G5 학생은 343492d).

## 도구

| 도구 | 상태 |
|---|---|
| `trainview` | 서버 |
| `tools/record_replay` (`record_ppo`, `record_bc`, `trpc` = trainfmt::trp C ABI) | 쓰임 — 자동 기록·손 기록 |
| `tools/og2sg` | 쓰임 — OmniGibson 판·BEHAVIOR 배치 |
| `fake_run` | 시험용으로 남김 — 가짜 실행(test data), 서버 기능(되감기·반쪽 줄·줄기) 시험 |
| `csv2run` | 옛것 읽기용으로 남김 — 규약 이전 `log.csv` 실행을 보관함으로 옮김 |

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


## 서버 경로

| 경로 | 돌려주는 것 |
|---|---|
| `/api/runs` | 실행 목록: id·kind·group·streams·age·iters·env_steps·eps·replays·evals·stage·**status**(학습 상태 불)·**health**(SR·충돌·판 길이·eval SR)·teacher·**teacher_run**/**teacher_ckpt**·**archive** + latest |
| `/api/meta`, `/api/progress`, `/api/episodes`, `/api/table`, `/api/evals`, `/api/events` | run.json, 열(키 목록·고른 키·`max_points` 솎기, 옛 키는 새 이름으로), 판 줄, 성공 표, 평가 표, events.txt |
| `/api/replays`, `/api/replay` | 재생 판 목록(`.sg`·`.trp` 머리 meta), `.trp` 바이트(3D classic) |
| `/api/live` | SSE: progress·reset·episodes·replay·runs·**status** |
| `/sg/`, `/stream`, `/file/`, `/api/map`·`view`·`walls`·`depth`·`mode`·`robot` | 세션의 진짜 sgview 로 역프록시(쿠키 `sgsess`) |
| `/api/sg/ctl`, `/api/sg/info`, `/api/sg/pipeline`, `/api/sg/rerun` | 재생 조종(sgs_play), 부모 화면 정보, 지금 인지 소스 해시, "stale pipeline" 다시 돌리기(og_queue 맨 앞) |
| `/api/sg/ctl`, `/api/sg/info`, `/api/sg/cam`, `/api/sg/policymap` | 재생 세션(판·시각·재생·속도), 판 정보(GT 궤적·바탕 층·카메라·관절 순서), 카메라 그림, 정책 지도(시각 t) |

## 확인한 것 (2026-10-04)

| 무엇 | 결과 |
|---|---|
| 자동 재생 기록(PPO 1 분 시험, N 1,024) | 체크포인트 4 개(152·302·452·끝) 모두 판, 건너뜀 0. 이터 GPU 시간 중앙값 50.02 ms(끔) 대 49.93 ms(켬) — 그대로 |
| 자동 재생 기록(BC/DAgger v2 시험, 위 교사로) | 학생 체크포인트 4 개(bc·dagger1–3) 모두 판 |
| 학습 상태 불 | 학습 중 training(+ 기록 중 REC), 끝나면 finished, SIGKILL 하면 crashed(기록 뒷 프로세스도 같이 죽음) |
| GPU 판 → 진짜 scenemap | A2 교사(343492d 판) 빈 지도 판: 물체 3 → 4 → 5, 점 1.4 천 → 3.5 천(판 7.5 s), 스트림 201 프레임. v2 시험 체크포인트(it152 / 끝)도 같은 화면 |
| OmniGibson 판(og2sg) | 4 판 스트림 버림 0, 물체 80–94, radio_slamfix_live 8 / 60 / 123 s 에 점구름 9 / 49 / 80 개 |
| 진행 기록·성공 표·되감기·옛 키 별칭 | progress 1 줄/s, 재개 trim, 표 = 직접 센 값, 옛 키 → 새 이름 |

## 지운 것 (정리, 2026-10-04)

- `scene_b1k`(BEHAVIOR 장면 → 한 프레임 `.trp`): `og2sg --layout` 으로 대체.
- 실행 무리 `labs`(옛 kind "lab"): `test data (synthetic)` 로 합침(읽기는 그대로).
- 화면: 예전 3D 그리기를 기본에서 뺌(`?debug=1` 에서만), sgview 화면에서 3D 전용 조작(시점·손끝·SLAM 궤적 켜고 끔) 숨김, 섞인 한국어 화면 글자 정리(설명은 풍선으로).
- 일회용 시험 실행: `~/trainview_work/smoke`(ppo_smoke_s1·bc_smoke_s1, v2 이전), `smoke_v2/live_a·live_b·live_c`(상태 불 시험), `smoke_v2/ppo_rep_off_s1`(처리량 비교 기준 — 값은 위 표).

## 남은 일

- **학습 중 판마다 기록**(본 줄기 `episodes.jsonl`, 학습 판 표본의 성공 표)은 장치 기록 링 변경이 필요(CUDA). 지금 재생 판은 체크포인트를 따로 돌린 평가 판.
- BEHAVIOR 장면에서 **우리 정책**이 도는 판은 E2(B 단계) 체크포인트가 생기면 같은 자동 기록으로.
- GPU 판 RGB 는 면 음영(물체 종류 색) — 팀 RenderBatch 에 id 버퍼가 생기면 그 RGB 로.
- 원래 BEHAVIOR 레이아웃 PNG 바닥 텍스처는 안 씀(RASC 방 격자 색칠).
- og2sg: 원래 실행 프롬프트 표가 기록에 없어 짝 못 지은 검출은 `cls<k>`.

## 장면 그래프 소스를 따라가는 규칙 (build_deps.sh)

인지(scenemap·objprob·ObjectSAM·SigLIP 2·libsgrt)와 sgview 는 **behavior-2026 `src/scene_graph` 가 원본**이고, 학습 뷰어·OG 다시 돌리기는 그걸 그대로 쓴다 — 복사·갈래·옛 빌드 없음.

- 진입점: **`training/viewer/build_deps.sh`** — libsgrt·sgview·sgs_play·og2sg·trainview 를 robot-agent 소스에서 `-j4` 로 빌드해 `$TRAINVIEW_DEPS`(기본 `~/trainview_work/deps`)에 링크한다. `og_replay.py`/`og_replay.sh`/trainview 는 거기서만 찾는다(개인 빌드 경로 박지 않음). 엔진·objprob 기본값은 `runtime`(`objprob_front.hpp` kDefaultEngine · `sgrt_glue.py` ENGINE)에서 읽고 여기서 정하지 않는다.
- 소스를 고친 뒤: behavior-2026 에서 고침 → 서브모듈 포인터 올림 → `tools/sync_scene_graph.sh` → `build_deps.sh` → trainview 다시 띄움. 그 뒤 재생 화면은 새 sgview 로, 새로 도는 OG 판은 새 인지로 나온다.
- 각 `_og.sg/meta.json` 의 `pipeline` 에 만든 소스(git 해시, scenemap·runtime·ovdet·clip·da 트리 해시), 엔진, objprob 매개변수 파일·해시, libsgrt 시각을 적는다. 재생 화면 HUD 에 표시하고, 지금 소스와 트리 해시가 다르거나 기록이 없으면 **"stale pipeline — re-run"** 단추: 눌러 og_queue 맨 앞에 다시 넣는다(옛 판은 `*.sg.prev` 로 비켜 둠).
- 손목 카메라: `og_replay.py` 가 LIMO RGB(`cam/NNNNNN.jpg`, `meta.cams`)와 OMX-F 손목 RGB(`cam/wNNNNNN.jpg`, `meta.wcams`, 로봇 모델의 `wrist_eye` 센서 = URDF `wrist_cam_link`, 화각은 모델 기본값)를 2 Hz 로 함께 기록한다. Replay 에서 `LIMO RGB`·`wrist RGB` 그림 위 그림(켜고 끔, 모서리 끌어 크기 조절).

## Training 탭 머리 (단순화)
- 기본은 **현재/최신 실행 하나**: 한 줄 머리(실행 이름·단계·상태 불과 말·걸린 시간·핵심 결과 + 작은 곡선·배운 교사 링크) → 큰 카드 4(성공·충돌·판 길이·학습 속도, "More stats" 접힘) → 건강 칩 "Health: OK / N warnings"(누르면 검사 펼침) → 핵심 곡선 4 + "More charts" 접힘.
- 실행 바꾸기: 위 줄 **Runs ▾**(검색, 단계별·최신순 한 줄 요약). Teachers (RL)·BEHAVIOR explore records·Archive 는 그 안의 접힌 구역. 시험(synthetic) 자료는 `?debug=1` 에서만 보인다.
- 실행이 하나도 없으면 빈 화면 안내(감시 중인 --root, 시작 방법, 문서 경로).
