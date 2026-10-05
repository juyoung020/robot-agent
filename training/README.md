# training/ — AI 모델 학습

로봇에 올릴 작은 모델을 우리가 직접 학습하는 곳이다. 실행(런타임) 코드는 여기 두지 않는다. 런타임 쪽은 `src/scene_graph/` 에 둔다(원본은 `src/behavior-2026/src/scene_graph/`).

## 배치

- 모델 하나에 폴더 하나.
  - `embed/`: 물체 이름 붙이기·검색용 영상–글 임베딩(증류). 설계와 결과는 [embed/README.md](embed/README.md).
  - `RL/`: 시뮬에서 RL 전문가(교사) 정책 학습 — 관측·보상·신경망·설정. [RL/README.md](RL/README.md)
  - `BC/`: 전문가 궤적 + 자연어 단계 지시로 우리 작은 VLA 모방학습(증류) — 얼린 SigLIP 2 영상 탑 + 지도·글 토큰 + flow matching 행동. 큰 모델은 아래 `vla/`(RecallVLA). G5: 교사 기록·BF16 BC 학습기·DAgger 를 GPU 그래프로. 영상 학생 = 학습 때 다시 렌더(팀 RenderBatch 그래프) → 얼린 SigLIP 2 패치 토큰(C++/CUDA) + 지도·글 토큰 → flow matching 청크, DAgger 8 번 0.946(교사 0.944). [BC/README.md](BC/README.md)
  - `vla/`: **RecallVLA** 학습기(C++/CUDA) — 몸통 Qwen3.5-0.8B 전부 학습 + SigLIP 2 B/32 영상 + 지도·몸 인코더 + 행동 전문가, 단계 문장 출력, 기억 요약 인코더. M1–M5(정확도 검증·메모리·속도·커널 융합) 끝, 큰 학습은 아직(입력은 TRAINING_DESIGN W10). 사양 [MAPVLA_SPEC.md](../docs/map_vla/MAPVLA_SPEC.md). [vla/README.md](vla/README.md)
  - `fastsam/`: 물체 분할 ObjectSAM(10-05 결정 — FastSAM-s 에서 증류한 YOLO26n 학생, things 만, `yolo26n-seg-obj-416`; FastSAM-s 계산의 약 1/10 이라 LIMO Jetson 에 맞음, 기기 위 시간은 아직 안 잼). 처음엔 FastSAM-s 416 다시 학습(벽·천장·바닥은 배경, 문·창·계단은 물체, 조각 대신 통째) — 원래 FastSAM 마스크 자기 증류 + BEHAVIOR 시뮬 정답 + COCO·LVIS·ADE20K, 같은 라벨로 Jetson Nano 용 YOLO26n-seg 학생. **이 폴더만 AGPL-3.0**(Ultralytics). 공개판은 별도 저장소 [ObjectSAM](https://github.com/juyoung020/ObjectSAM). [fastsam/README.md](fastsam/README.md)
  - `viewer/`: 학습 뷰어(Rust 서버 + 브라우저, 실행 폴더를 읽기만). [viewer/README.md](viewer/README.md)
  - `model/`: 베이스 모델 가중치(저장소에 없음, 받는 방법은 [model/README.md](model/README.md)).
- 폴더마다 `README.md` 를 둔다(한국어). 내용은 목표, 데이터와 라이선스, 실행 순서, 결과 표, 남은 일.

## 공통 규칙

- **git 에 넣지 않는 것**: 데이터셋, crop, 임베딩, 가중치, ONNX, TensorRT 엔진, 로그.
  - 이것들은 `~/<모델>_work/` 에 둔다(예: `~/embed_work/{data,emb,runs,logs,models}`).
  - 경로는 환경 변수로 바꿀 수 있다(`EMBED_WORK`).
- **venv**: 모델마다 `~/<모델>_venv`(예: `~/embed_venv`).
  - torch 는 cu128 판(2.11.0+cu128)을 쓴다. 저장소 기준이 CUDA 12.8 이다.
  - `uv venv -p 3.11 ~/embed_venv` → `uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128` → 나머지(`open_clip_torch timm transformers datasets onnx onnxruntime pycocotools nltk`).
- **GPU 를 같이 쓴다**. 긴 작업 전에 `nvidia-smi` 를 본다. 우리 작업은 VRAM 8 GB 아래로 잡는다.
- **결과 기록**
  - 실행마다 `~/<모델>_work/runs/<이름>/` 에 설정(`args.json`), 로그, 평가 json 을 남긴다(embed).
  - 학습 뷰어 규약(RL·BC): 실행 폴더에 `run.json`(설정·상수 — args 대신) + `progress.jsonl` + `evals/` — `ppo_run`·`bc_run` 이 쓴다. 보기는 `training/viewer/`([README](viewer/README.md)).
  - RL·BC 는 `--out` 으로 준 폴더(`~/ra_*` 아래, 예: `~/ra_bc/runs/<이름>/`)에 설정 `config.json`, 로그 `log.csv` 를 남긴다. RL 은 `events.txt`·체크포인트, BC 는 평가 `results.json` 도.
  - 모델 폴더 README 의 결과 표에는 실행 이름과 숫자를 옮겨 적는다.
  - 평가셋은 남의 것을 읽기만 한다(예: `~/clip_bench/evalset.json`).
- 커밋은 `training/<모델>/` 경로만 지정해서 한다(`git add -A` 금지).
