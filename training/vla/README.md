# RecallVLA — 학습기 (C++/CUDA)

사양·결정: [docs/map_vla/MAPVLA_SPEC.md](../../docs/map_vla/MAPVLA_SPEC.md). 몸통 = Qwen3.5-0.8B(전부 학습), 영상 = SigLIP 2 B/32-256, 지도·몸 인코더, π0.5 꼴 행동 전문가, 단계 문장 출력.
PyTorch·JAX 없음. Python 은 오프라인 기준값 덤프(`tools/qwen_ref.py`)만.

## 빌드
```
cmake -S training/vla -B ~/ra_vla/build && cmake --build ~/ra_vla/build -j4
~/clip_venv/bin/python training/vla/tools/qwen_ref.py training/model/Qwen3.5-0.8B ~/ra_vla/ref/q08   # 한 번(HF FP32 CPU, 약 20 s)
~/ra_vla/build/qwen_verify cos   training/model/Qwen3.5-0.8B ~/ra_vla/ref/q08 --neg
~/ra_vla/build/qwen_verify gen   training/model/Qwen3.5-0.8B ~/ra_vla/ref/q08
~/ra_vla/build/qwen_verify v67   training/model/Qwen3.5-0.8B
~/ra_vla/build/qwen_verify bench training/model/Qwen3.5-0.8B 32 232
```

## M1 — Qwen3.5 앞 계산·디코딩 (잰 값, 2026-10-04)

구현(`src/qwen.cu`, `src/qkern.cu`): safetensors 직접 읽기(`src/st.cpp`), config.json 그대로. 층 24 = Gated DeltaNet 18(합성곱 4 + SiLU, q·k L2 정규화, β = σ(b),
g = −e^{A_log}·softplus(a + dt_bias), 델타 규칙 재귀 — 블록 = (열, 머리), 상태 열마다 레지스터, 게이트 RMSNorm) + 게이트 풀 어텐션 6(q·k 머리 RMSNorm(1 + w),
앞 64 차원 RoPE θ 10⁷, GQA 8/2, 인과 온라인 softmax, 출력 ⊙ σ(gate)), SwiGLU MLP, RMSNorm(1 + w). GEMM = RL network `gemm2_k`(BF16 입력, FP32 누산),
나머지(잔차·정규화·합성곱·재귀·softmax) FP32. 디코딩: 풀 층 K·V 캐시, 선형 층 합성곱 기록 + 재귀 상태, LM 머리 = 임베딩 전치, 탐욕(장치 argmax)·top-p(호스트).
토크나이저: 디코딩 = 오프라인 표(`vocab.bin`, 바이트 수준 BPE 를 바이트로 풀어 둠), 인코딩 = 오프라인으로 만든 토큰 번호(학습 문장 표도 같은 방식으로 만든다).

| 검사 (HF transformers 5.18 FP32 CPU 기준, 프롬프트 5 개 = 토큰 272 개: 영어 대화 틀·한국어·평문·긴 166 토큰) | 결과 |
|---|---|
| 층별 은닉 상태 코사인(토큰마다), 0.8B | 임베딩 1.000000, 층 1–24 평균 0.999992–0.999997, 하위 1 % ≥ 0.999919, **끝 RMSNorm 뒤 평균 0.999992 / 하위 1 % 0.999921 / 최저 0.999901** (기준 ≥ 0.999 / ≥ 0.99 통과) |
| 같은 검사, 2B(같은 코드) | 끝 평균 0.999995 / 하위 1 % 0.999964, 층 최악 평균 0.999994 → 통과 |
| 음성 대조(실패해야 함) | RoPE θ 1e4(옛 기본값) 평균 0.9946·하위 1 % 0.967, MLP 앞 RMSNorm 빠뜨림 0.234, DeltaNet 감쇠 빠뜨림 0.542, q·k 정규화 빠뜨림 0.464 → **4 / 4 잡힘**. 참고: θ × 10 은 0.99959 / 0.99455 로 **기준을 통과해 버린다**(272 토큰 안의 짧은 위치에서는 낮은 주파수가 거의 안 돎 — 이 기준으로는 작은 θ 오류를 못 잡음) |
| 탐욕 디코딩 24 토큰 | 자유 실행 5 개 중 4 개 같음. 교사 강요(HF 토큰 열에서 스텝마다 argmax) **119 / 120 같음**. 다른 한 스텝(긴 프롬프트 19 번째)은 HF 로짓 차 **0.027** 인 거의 같은 값끼리(우리 쪽 차 0.025, 반대 방향) — BF16 GEMM 입력 반올림으로 뒤집힘. 규칙: 다른 스텝은 모두 HF 차 < 0.05 → 통과 |
| V7 같은 입력 두 번 / 묶음 무관 | B 4 × L 200: 819,200 낱말 중 0 다름, 열 하나 따로 == 묶음 안 0 다름 |
| V6 CUDA 그래프 == 즉시 | 0 다름 |
| 속도(B 32 × L 232 prefix, GPU 다른 일 없음) | **185.3 ms = 5.79 ms/표본**, 몸통 GEMM FLOP 기준 39.9 TFLOPS(GEMM 만은 74–87 — 나머지는 어텐션·재귀·정규화, M3 에서 나눠 잼), 모델 + 작업 공간 2.20 GB |
| 탐욕 디코딩 | B 1: 5.18 ms/토큰(193 토큰/s), B 8: 5.59 ms/스텝(1,430 토큰/s). 가중치 읽기 하한(1.5 GB / 896 GB/s ≈ 1.7 ms)의 3 배 — 커널 실행 수가 많음(그래프·GEMV 는 M3) |
