# 저장소 배치 · 빌드 · 외부 경로

## 배치 (전부 `~/robot-agent` 안)
```
src/        실행 코드: scene_graph(유일한 원본: scenemap·da·spark_dsg·sgview·runtime·ovdet·clip·tools/realbag), agent, robot, vla
training/   학습: RL(env·map·ppo·map_cmp)·BC·vla·embed·fastsam·viewer(학습 뷰어·og_replay)
tools/      build_all.sh(빌드 한 진입점)·check_env.sh·run_explore_live.sh·run_sgview.sh
config/     paths.env(외부 경로 기본값, git) · paths.local.env(내 PC, git 밖) · paths.py
docs/ archive/ refs/ tests/ scripts/
build/      (git 밖) 모든 빌드 결과 — cmake 폴더 build/<이름>, cargo build/cargo/<이름>, 실행 파일 링크 build/bin/
models/     (git 밖) 로봇·파이프라인이 실행할 때 쓰는 엔진·가중치 (OVDET_MODELS = models/ovdet)
training/model/ (git 밖) 학습할 때 불러오는 베이스 가중치 — Qwen3.5-0.8B·SigLIP 2(쓰는 것), Qwen3.5-2B·smolvla_base(나중을 위해 보관)
data/       (git 밖) 데이터셋·기록·실행 폴더 (datasets/ embed_work/ trainview_work/)
third_party/(git 밖) BEHAVIOR-1K(B1K_ROOT)·cartographer(Cartographer 코어, build_all.sh cartographer 가 받음) 등 큰 외부 코드·자산
```
conda·venv(`~/miniconda3` 의 `behavior`, `~/fastsam_venv` 등)와 시스템 캐시는 홈에 둔다 — `config/paths.env` 의 `OG_CONDA_ENV`/`OG_PYTHON` 으로 가리킨다.

## 빌드
`tools/build_all.sh [대상...]` 하나로 전부(증분, `-j4`, `JOBS=` 로 조절). 대상: `tools/build_all.sh --list`.
`cartographer slam_carto scenemap sgrt sgclip realbag sgview og2sg env map map_cmp ppo bc vla record_replay trainview agent ppo_driver bc_driver`.
시험: `ctest --test-dir build/scenemap` 등. 학습 뷰어용 링크: `training/viewer/build_deps.sh`.

## 외부 경로
모든 코드는 `config/paths.env` 의 변수(`RA_BUILD RA_MODELS OVDET_MODELS RA_DATASETS RA_EMBED_WORK RA_LABELS RA_TRAINVIEW_WORK B1K_ROOT OG_CONDA_ENV OG_PYTHON`)만 읽는다.
bash: `. config/paths.env`, 파이썬: `config/paths.py`. 내 PC 만 다르면 `config/paths.local.env`.
`tools/check_env.sh` 가 빠진 것을 알려 준다.

## 큰 데이터(약관·크기 때문에 git 밖)
- BEHAVIOR-1K(≈38 GB 자산, 약관상 재배포 금지) → `third_party/BEHAVIOR-1K`
- ObjectSAM·FastSAM·SigLIP 2 엔진 → `models/ovdet/x86_sm120/`
- 라벨 표 objects-v1(65 MB) → `data/embed_work/labels/objects-v1`
