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
              progress.jsonl                       업데이트마다 → ≤ 1 줄/s 로 합침(time/iters_in_row), 모집단 값
              episodes.jsonl                       판마다 한 줄(표본, log_envs 환경의 모든 판)
              replays/ep_<판>_<스킬>_<결과>.trp      판 하나 궤적(바이너리, trainfmt::trp)
              evals/<이름>.json                     평가 표 {iter, ckpt, rows:[{eval, split, n, success, collision, timeout, ref, pass}]}
              s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/   집계에 섞으면 안 되는 판(평가, DAgger 교사 판)
```
- 덧붙이기(progress·episodes)와 이름 바꾸기(run.json·evals·replays: `.tmp` 에 쓰고 rename)만 한다.
- **잰 것만 쓴다**: 그 업데이트에 끝난 판이 없으면 비율 키를 **뺀다**(0·null 을 쓰지 않음).
- 키 이름 `<묶음>/<이름>`: `time/{iter,env_steps,wall,fps_env,fps_gpu,rollout_ms,update_ms,iter_ms,iters_in_row}`, `rollout/{n_eps,ep_ret_mean,ep_len_mean,success/<스킬>,collision_rate,timeout_rate,…}`, `reward/<스킬>/<항>`, `train/{approx_kl,clip_frac,entropy,log_std_mean,std/<행동>,policy_loss,value_loss,grad_norm,lr,…}`, `curr/{stage_idx,success/C0..C2,stage_frac/C0..C2,…}`, `bc/loss`, `dagger/{round,disagree}`, `rollout_teacher/*`(교사가 몬 판), `eval/<teacher|student>/*`, `fp8/*`, `gpu/mem_used_mb`, `log/*_dropped`.
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
| `csv2run --out DIR <실행 폴더>…` | 규약 이전 실행(`log.csv` + `config.json` + `events.txt`/`results.json`)을 규약 폴더로 옮김(원본은 읽기만). 키 이름은 runfolder.rs 와 같다 |

## 경로 (서버)

| 경로 | 돌려주는 것 |
|---|---|
| `/api/runs` | 실행 목록(id, kind, group, streams, live, age, iters, env_steps, eps, replays, evals, stage, first_ts, rewound, disk_mb) + latest |
| `/api/meta?run=` | run.json 그대로 |
| `/api/progress?run=` | 키 목록(키마다 처음·마지막 줄, 개수) |
| `/api/progress?run=&keys=a,b\|*&from=<줄>&sig=&max_points=` | 고른 키를 열로. 없는 값 null. `max_points` 보다 길면 칸마다 마지막 + `lo`/`hi`(최소·최대). `sig` 가 다르면 처음부터 |
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

## 남은 일

- **판마다 기록(episodes.jsonl)·리플레이(.trp)를 진짜 학습기에서**: 장치에서 판 끝마다 줄 하나, 고른 환경의 프레임을 링에 쓰는 커널·그래프 변경이 필요하다(이번 작업은 CUDA·그래프를 건드리지 않음). 지금 성공 표·재생 탭은 fake_run 자료로만 확인했다.
- PPO 의 `train/explained_variance`, `log/*_dropped`(장치 링 넘침 수), 보상 항목별 합(`reward/<스킬>/<항>`), 접촉 수는 장치 기록에 없어 키가 없다(화면은 "not logged").
- 영상 칸(img 섹션 JPEG): 화면은 읽지만 쓰는 쪽이 없다.
- `progress` 첫 읽기를 64 MB 씩 여러 요청에 나눠 "읽는 중 n %" 표시(지금은 한 요청 안에서 64 MB 씩 끝까지 읽음).
- SSE 는 연결마다 감시 스레드가 1 초마다 stat 한다(설계의 "감시 스레드 하나 + 방송" 대신 — 보는 사람 1–3 명이라 같은 비용, 연결마다 커서가 따로라 줄이 빠지거나 겹치지 않음).
- 그리퍼 값 → 두 관절 식(부호·배율)은 URDF 확인 전 가정(`joint_map` qg → `omx_gripper_joint_1` ×1, `_2` ×−1).
