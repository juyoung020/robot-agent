# RL — 시뮬에서 RL 전문가(교사) 정책 학습

작은 정책(영상 없음)을 시뮬 수천 개 병렬로 학습해서, 성공한 궤적을 BC 폴더의 VLA 학습 데이터로 쓴다. C++/CUDA/Rust 네이티브만 쓴다(PyTorch·JAX 등 프레임워크와 Python 학습·추론 코드 금지. 층별 정확성 검증용 오프라인 기준값 덤프만 예외).

| 폴더 | 내용 |
|---|---|
| `env/` | LIMO + OMX GPU 환경 커널(G1), CPU 참조판과 비트 동일 |
| `map/` | GPU 안 자라는 지도(scenemap 근사, G2 앞부분), CPU 참조판과 비트 동일 |
| `observation/` | 관측: G1 관측 80 + 지도 토큰(물체 칸·벽 56·방·완성도) → 신경망 입력(G3, CPU·GPU 같은 소스) |
| `reward/` | 단계별 보상: G1 환경 보상 + 학습기 쪽 퍼텐셜 모양 잡기(G3) |
| `network/` | 정책·가치 신경망(칸 MLP + 집합 + MLP 머리), 손 BF16 GEMM·손실·Adam 커널, CPU FP64 참조판(G3) |
| `ppo/` | RL 교사 PPO(G3): 롤아웃·GAE·갱신을 CUDA 그래프 둘로, 검증 V4–V7(`tools/ppo_verify`), Rust 실행기(`driver/`) |
| `config/` | 하이퍼파라미터·커리큘럼 설정(JSON) |
