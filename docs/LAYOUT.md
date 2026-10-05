# 저장소 배치 · 빌드 · 외부 경로

## 배치 (전부 `~/robot-agent` 안)
```
src/          실행 코드: scene_graph(인지의 유일한 원본: scenemap·da·spark_dsg·sgview·runtime·ovdet·clip·slam_carto·tools/realbag), agent, robot, sim, vla
training/     학습 코드: RL(env·map·ppo·map_cmp)·BC·vla·embed·fastsam·viewer(학습 뷰어·og_replay)
  data/       학습 입력 — 작은 표(pnp_v1·vla_v1)만 git, 큰 것은 git 밖:
              b1k_scenes(GPU 환경 장면 묶음) · fastsam_obj(ObjectSAM 학습) · embed(임베딩 학습) ·
              map_cal·og_cmp(GPU 지도 인지 흉내 보정) · objprob(objprob 매개변수 맞추기) · sim_detcmp(검출 평가)
  runs/       (git 밖) 학습 결과·체크포인트(ppo/·bc/ …)
  model/      (git 밖) 학습할 때 불러오는 기본 가중치 — Qwen3.5-0.8B·SigLIP 2(씀), Qwen3.5-2B·smolvla_base(나중을 위해)
models/       (git 밖) 실행할 때 쓰는 것만 — 로봇에는 이 폴더만 복사:
              ovdet/(ObjectSAM·SigLIP 2 TensorRT 엔진) · labels/objects-v1(라벨 표)
data/         (git 밖) 실행·평가 기록: realbags(실제 bag) · limo_rec(리모·시뮬 기록) · outputs(탐사 판) · trainview_work(학습 뷰어)
tools/        build_all.sh(빌드)·check_env.sh·check_paths.sh·audit.py(잔재 점검)·run_explore_live.sh·run_sgview.sh
config/       paths.env(경로 기본값, git) · paths.local.env(내 PC, git 밖) · paths.py
docs/ refs/   설계 문서 · 참고 목록(논문·코드·데이터시트는 refs/download.sh 로 받음)
build/        (git 밖) 모든 빌드 결과 — cmake build/<이름>, cargo build/cargo/<이름>, 실행 파일 링크 build/bin/
third_party/  (git 밖) BEHAVIOR-1K(B1K_ROOT)·cartographer 등 큰 외부 코드·자산
```
conda·venv(`behavior`, `fastsam_venv` 등)와 시스템 캐시(`~/.cache`)만 홈에 둔다 — `config/paths.env` 의 `CONDA_SH`·`OG_CONDA_ENV`·`*_PY` 로 가리킨다.

## 빌드
`tools/build_all.sh [대상...]` 하나로 전부(증분, `-j4`, `JOBS=` 로 조절). 대상: `tools/build_all.sh --list`.
시험: `ctest --test-dir build/scenemap` 등. 학습 뷰어용 링크: `training/viewer/build_deps.sh`.

## 경로 변수
모든 코드는 `config/paths.env` 의 변수만 읽는다(bash: `. config/paths.env`, 파이썬: `config/paths.py`). 내 PC 만 다르면 `config/paths.local.env`.

| 변수 | 기본 |
|---|---|
| `RA_MODELS` · `OVDET_MODELS` · `RA_LABELS` | `models` · `models/ovdet` · `models/labels/objects-v1` |
| `RA_TRAIN_DATA` · `RA_EMBED_WORK` · `RA_B1K_SCENES` · `RA_CHECKPOINTS` | `training/data` · `training/data/embed` · `training/data/b1k_scenes` · `training/runs` |
| `RA_DATASETS` · `RA_TRAINVIEW_WORK` | `data` · `data/trainview_work` |
| `RA_BUILD` · `B1K_ROOT` | `build` · `third_party/BEHAVIOR-1K` |

`tools/check_env.sh` 가 빠진 것을, `tools/check_paths.sh`(커밋 전 훅)가 저장소 밖 경로를 박은 코드를 알려 준다.
