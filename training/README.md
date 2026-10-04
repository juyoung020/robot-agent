# training/ — AI 모델 학습

로봇에 올릴 작은 모델을 우리가 직접 학습하는 곳이다. 실행(런타임) 코드는 여기 두지 않는다. 런타임 쪽은 `src/scene_graph/` 에 둔다(원본은 `src/behavior-2026/src/scene_graph/`).

## 배치

- 모델 하나에 폴더 하나.
  - `embed/`: 물체 이름 붙이기·검색용 영상–글 임베딩(증류). 설계와 결과는 [embed/README.md](embed/README.md).
  - `RL/`: 시뮬에서 RL 전문가(교사) 정책 학습 — 관측·보상·신경망·설정. [RL/README.md](RL/README.md)
  - `BC/`: 전문가 궤적 + 자연어 단계 지시로 Qwen 기반 VLA 모방학습(증류). G5: 교사 기록·BF16 BC 학습기·DAgger 를 GPU 그래프로. 영상 학생 = 학습 때 다시 렌더(팀 RenderBatch 그래프) → 얼린 SigLIP 2 패치 토큰(C++/CUDA) + 지도·글 토큰 → flow matching 청크, DAgger 8 번 0.946(교사 0.944). [BC/README.md](BC/README.md)
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
  - 실행마다 `~/<모델>_work/runs/<이름>/` 에 설정(`args.json`), 로그, 평가 json 을 남긴다.
  - 모델 폴더 README 의 결과 표에는 실행 이름과 숫자를 옮겨 적는다.
  - 평가셋은 남의 것을 읽기만 한다(예: `~/clip_bench/evalset.json`).
- 커밋은 `training/<모델>/` 경로만 지정해서 한다(`git add -A` 금지).
