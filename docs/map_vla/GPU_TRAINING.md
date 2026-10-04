# GPU 안에서만 도는 학습 — RL 교사와 VLA (C++/CUDA/Rust)

작성 2026-10-03. **설계 문서**다. 코드는 아직 없다.
- 입력·출력은 [VLA_INPUT.md](VLA_INPUT.md), 시뮬 범위는 [SIM_PORTING.md](SIM_PORTING.md), 팀 엔진 검토는 [BEHAVIOR_ENGINE_REVIEW.md](BEHAVIOR_ENGINE_REVIEW.md), 학습 흐름은 [training/README.md](../../training/README.md).
- 표기: **(추정)** = 재지 않았거나 원문으로 확인하지 못한 숫자·판단. 표시 없는 숫자는 코드·문서·NVIDIA 자료에서 확인한 것이다.
- 팀 코드 경로는 `src/behavior-2026/` 아래 기준이다. JSBSim 포팅은 `~/juyoung020/jsbsim-f16-cuda` 이다.

## 0. 한눈에

1. **목표**: 시뮬 → 지도 → 관측 → 정책 → 롤아웃 버퍼 → 이득 계산 → 역전파 → 옵티마이저를 **전부 GPU 안**에서 돈다. 호스트는 그래프를 띄우고 로그를 비동기로 읽기만 한다.
2. **언어**: 커널은 CUDA, 그래프·메모리는 C++, 설정·실행 흐름·로그·체크포인트는 Rust. PyTorch·JAX 는 학습·추론에 없다. Python 은 층별 검증용 기준값을 한 번 뽑을 때만 쓴다.
3. **이미 있는 것**: JSBSim F-16 커널(스레드 하나 = 기체 하나, 상태를 레지스터에 두고 여러 프레임), 팀의 π0.5 네이티브 학습기(`src/vla/pi05_train`), 네이티브 로더(`src/vla/fasttrain`), 손 GEMM·CUDA 그래프(`src/vla/pi05_native`), PhysX 와 비트 동일한 GPU 물리·렌더(`src/sim/engine`).
4. **새로 짤 것**: LIMO + OMX 환경 커널, **GPU 안에서 자라는 지도(scenemap 근사)**, 작은 정책용 합친 커널, 장치 안에서 끝나는 PPO·BC 학습 그래프, FP8 GEMM.
5. **FP8**: RTX 5070 Ti(sm_120)는 `mma.sync ... kind::f8f6f4` 로 FP8 을 쓴다. `tcgen05`·TMEM 은 없다(2절). FP8(FP32 누산)의 이론 최고치는 BF16 의 2 배다.
6. **지도**: scenemap 은 CPU 다. 학습 중에는 GPU 근사판으로 지도를 **에피소드 안에서 키운다**. 근사판은 표본 에피소드에서 진짜 scenemap 과 맞춰 본다(5절).
7. **속도**: 시뮬만 빨라지면 학습 전체는 시뮬 몫만큼만 빨라진다(JSBSim 측정). 그래서 정책·학습 쪽 합치기와 호스트 동기 제거가 먼저다(10절).

## 1. 원칙

1. **한 바퀴 동안 호스트 동기 0 회.** `cudaStreamSynchronize`, 장치 → 호스트 복사 후 대기, 호스트 값을 커널 인자로 넘기기를 하지 않는다.
2. **주소는 고정.** 모든 상태는 제자리에서 갱신한다. 그래야 CUDA 그래프로 잡힌다. JSBSim 판이 그렇다(`docs/api.md` "State is only ever updated in place").
3. **판(env) 끼리 상호작용 없음.** 판 하나 = 스레드 하나(가벼운 것) 또는 블록 하나(지도·접촉). 팀 규칙도 같다(`docs/엔진_자체구현.md` 12.2 7번).
4. **기준 먼저.** 커널마다 CPU C++ 참조판을 같이 만들고 대조한 뒤 다음으로 간다(9절).
5. **잰 것만 숫자로 쓴다.** 나머지는 (추정).

### 1.1 언어 나눔

| 언어 | 맡는 일 | 팀 코드의 예 |
|---|---|---|
| CUDA | 환경 스텝, 지도 갱신, 관측, 정책 앞·뒤, 손실, 옵티마이저, 렌더, 디코딩 뒤 처리 | `pi05_train/src/tkern.cu`, `sim/engine/cuda/` |
| C++ | 장치 메모리 배치, 그래프 잡기·실행, 가중치 파일, C ABI | `pi05_native/src/model.cu` `capture_graph`, `fasttrain/csrc/capi.cpp` |
| Rust | 설정 읽기, 실행 순서, 커리큘럼 값 쓰기, 로그·체크포인트 관리, 데이터 표 만들기 | `fasttrain/ftprep`(Rust 표 만들기) |
| Python | 오프라인 기준값 덤프만 (한 번) | `pi05_train/tools/train_ref.py` |

## 2. 대상 하드웨어 — RTX 5070 Ti (sm_120), CUDA 12.8

### 2.1 사양 (NVIDIA RTX Blackwell 백서, 부록 B 표 5, 밀집 값)

| 항목 | 값 |
|---|---|
| SM | 70, 텐서 코어 280 (5세대) |
| BF16 텐서 (FP32 누산) | 87.9 TFLOPS |
| FP16 텐서 (FP16 누산 / FP32 누산) | 175.8 / 87.9 TFLOPS |
| FP8 텐서 (FP16 누산 / FP32 누산) | 351.5 / 175.8 TFLOPS |
| FP4 텐서 (FP32 누산) | 703 TFLOPS |
| FP32 (텐서 아님) | 43.9 TFLOPS |
| 메모리 | 16 GB GDDR7, 256-bit, 896 GB/s |
| L2 | 48 MB (49,152 KB) |
| L1/공유 메모리 | GPU 전체 8,960 KB (SM 당 128 KB) |
| 레지스터 | GPU 전체 17,920 KB |
| NVDEC | **1 개** (6세대) |

- GeForce 는 **FP32 누산이 FP16 누산의 절반**이다. 그래서 학습에서 쓸 수 있는 FP8(FP32 누산)은 BF16 의 2 배다. FP8 FP16 누산은 4 배지만 정확도 위험이 있다(7절).
- 팀 손 GEMM(`pi05_native/src/gemm.cuh`)은 `mma.sync.m16n8k16 bf16, f32 누산` 이다. 상한은 87.9 TFLOPS 다. π0.5 접두부는 GEMM 상한의 약 90 % 에서 돈다(`pi05_native/README.md`).

### 2.2 sm_120 에서 되는 명령과 안 되는 명령

| 기능 | sm_120 / sm_120a | 근거 |
|---|---|---|
| `mma.sync` BF16/FP16 `m16n8k16`, `ldmatrix`, `cp.async` | 됨 (팀 코드가 씀) | `pi05_native/src/gemm.cuh`, `pi05_train/src/tgemm.cuh` |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32` (e4m3, e5m2, e3m2, e2m3, e2m1 입력, f32 누산) | 됨. **`sm_120a` 대상**으로 컴파일해야 함. CUDA 12.8(PTX 8.7)부터 | PTX ISA 8.7: "sm_120a ... supports specialized accelerated features", mma 의 `.kind`·`.block_scale` 은 sm_120a 기능. CUTLASS `cute/arch/config.hpp`: SM120 + nvcc ≥ 12.8 이면 `CUTE_ARCH_F8F6F4_MMA_ENABLED` |
| FP8 e4m3/e5m2 `m16n8k16` + **f16 누산** | 됨 (PTX 8.7 에서 추가) | PTX ISA 8.7 새 기능 목록 |
| 블록 배율 FP8 `kind::mxf8f6f4.block_scale` (UE8M0, 32 개마다 배율) | 됨 (sm_120a) | CUTLASS `CUTE_ARCH_MXF8F6F4_MMA_ENABLED` (SM120) |
| NVFP4 `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... ue4m3` | 됨 (sm_120a). NVFP4 블록 배율의 유일한 모양이 m16n8k64 | Colfax SM12x NVFP4 글, CUTLASS `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` |
| `scale_vec::4X` + UE8M0 (mxf4nvf4) | CUDA 13.1 부터 | CUTLASS config(`__CUDACC_VER_MINOR__ >= 1` 조건), PTX 9.1 |
| sm_89 옛 FP8 형식 `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` (`kind` 없음) | CUTLASS 는 SM120 에서 이 경로를 켜지 않음. 하드웨어가 받는지는 확인 못 함 → **쓰지 않는다** | CUTLASS 이슈 #3044 |
| `tcgen05.mma`, TMEM, 2-SM UMMA | **없음**. sm_100a/101a/103a/107a/110 전용 | CUTLASS config: `CUTE_ARCH_TCGEN05_*`·`TCGEN05_TMEM` 은 SM100/101/103/107/110 에만. Colfax: "SM12x does not use tcgen05 or TMEM" |
| `wgmma` | 없음 (sm_90a 전용 명령) | PTX ISA (sm_90a 기능). SM12x 는 워프 단위 `mma.sync` (Colfax) |
| TMA (`cp.async.bulk.tensor`) | 됨 | CUTLASS `CUTE_ARCH_TMA_SM90_ENABLED`(SM120A), `CUTE_ARCH_TMA_SM120_ENABLED`, Colfax |
| PDL (`griddepcontrol`) | 됨 (sm_90+) | 팀 `pi05_native/src/common.cuh` 가 씀 |
| `sm_120f` (계열 대상) | CUDA 12.9 부터 → 12.8 기준에서는 못 씀 | CUTLASS `cutlass/arch/config.h` (`CUTLASS_ARCH_MMA_SM120F_SUPPORTED` 는 12.9 이상) |
| cuBLASLt FP8 | 됨. 단 Blackwell GeForce(12.x)는 **TN 배치만** | cuBLAS 문서 문장(PyTorch 이슈 #198126 인용). 우리는 cuBLAS 를 쓰지 않고(팀 방침) 속도 비교용으로만 쓴다 |
| CUTLASS sm_120 | 됨. `mma.sync` 기반 블록 배율 GEMM 빌더(128x32xK, 128x64xK 타일 추가) | CUTLASS 문서, Colfax. 우리는 링크하지 않고 구조만 참고 |

- 정리: **우리 FP8 은 `mma.sync` + `ldmatrix`/`cp.async`(또는 TMA) 손 GEMM** 이다. 팀 손 GEMM 틀에 FP8 경로를 더한다.
- `sm_120a` 큐빈은 정확히 cc 12.0 에서만 돈다. 이 PC 하나만 쓰므로 괜찮다. 팀 빌드 기본값은 `sm_120` 이므로 FP8 커널은 `-gencode arch=compute_120a,code=sm_120a` 로 따로 만든다.
- FP8 피연산자 A·B 를 서로 다른 형식(e5m2 × e4m3)으로 줄 수 있는지는 PTX 형식상 `.atype`·`.btype` 가 따로 있다. 실제 컴파일로 확인할 것(12절).
- 8 비트 `ldmatrix` 전치 모양(`CUTE_ARCH_LDSM_SM100A_ENABLED` 가 SM120A 에도 켜짐)이 있는지도 컴파일로 확인한다. 없으면 전치는 양자화 커널이 FP8 사본을 따로 써서 해결한다(7.3).

## 3. 무엇을 학습하나

| 학습 | 모델 (추정 크기) | 입력 | 데이터 |
|---|---|---|---|
| RL 교사 (`training/RL`) | 영상 없음. 물체 칸 16 개를 같은 MLP 로 토큰화 → 작은 집합 인코더 → 정책·가치 머리. 1–5M 변수 (추정) | 몸 상태 56 + 물체 칸 + 벽·방 + 목표 (VLA_INPUT 2–4절). **지도는 자라는 지도** | GPU 시뮬 수천 판, PPO |
| BC 작은 VLA (로봇용) | SigLIP 2 B/32 패치 토큰 + 트랜스포머 6–8 층 + flow matching 액션 전문가. 100–200M (추정) | VLA_INPUT 전체 | RL 교사 궤적 + DAgger + 외부 시연 |
| BC 큰 VLA (PC) | Qwen3.5-2B 얼림(파일 4.3 GB) + 액션 전문가 + 지도 토큰 인코더. 학습 대상 100–300M (추정) | 같음 | 같음 |

## 4. 한 GPU 안 학습 고리

```
호스트 (Rust 실행기 → C++ C ABI)
  매 바퀴: cudaGraphLaunch(rollout) ; cudaGraphLaunch(update) ; 로그 링 버퍼를 이벤트로 확인 (기다리지 않음)

GPU — rollout 그래프 (WHILE 조건 노드, T 스텝)
  ┌ K1 env_step  : 행동 풀기 → 베이스 적분 → 팔 PD → 순기구학 → 잡기 → 보상 → 끝 판정 → 끝난 판 리셋
  │ K2 map_update: (keyframe 인 판만) 깊이 광선 → 격자, 물체 보임 → 물체 기억, slam 자세 오차
  │ K3 obs_policy: 칸 고르기·정규화 → 칸 MLP → 정책·가치 → 행동 표본(장치 난수) → logp
  └ 버퍼 쓰기   : 관측(지도 토큰 시계열), 행동, logp, 가치, 보상, 끝, 지도 완성도
GPU — update 그래프
  K4 gae_norm  : 판마다 뒤에서 앞으로 GAE, 이득 정규화 통계(장치 합)
  에포크 × 미니배치 (WHILE 노드): 앞 → K5 손실 → 뒤 → K9 옵티마이저(기울기 노름·자르기·AdamW·EMA·저정밀 사본)
```

### 4.1 그래프와 장치 쪽 제어

- 바퀴 전체를 그래프 둘로 잡는다. 팀 `pi05_native` 는 추론 전체를 그래프 하나로 잡는다(`model.cu:655` `capture_graph`).
- 반복은 **조건 노드**로 장치 안에서 돈다. 조건 노드(IF·WHILE)는 CUDA 12.3 부터, IF/ELSE·SWITCH 는 12.8 부터 있다. 조건 값은 커널이 `cudaGraphSetConditional` 로 쓴다(NVIDIA 블로그 "Dynamic Control Flow in CUDA Graphs with Conditional Nodes"). → T 스텝, 에포크, 미니배치 수, "keyframe 인가" 를 호스트가 몰라도 된다.
- 커널 사이 꼬리는 PDL 로 겹친다. 팀 `launch_k` 가 이미 한다(`common.cuh:54-97`).
- 판 리셋은 장치에서 한다. JSBSim `reset(..., mask=)` 는 호스트 동기 없이 그래프 안에서 돈다(`docs/api.md`). 우리 리셋도 끝난 판 마스크만 보고 장면·물체 배치·지도를 장치 난수로 다시 만든다.
- 커리큘럼 값·학습률 일정은 장치 메모리 값이다. 학습률은 장치의 스텝 카운터로 커널이 계산한다. Rust 는 바뀔 때만 고정(pinned) 버퍼에 쓰고, 그래프 앞의 작은 복사 노드가 옮긴다.
- 로그: 커널이 손실·보상·완성도 합을 장치 링 버퍼에 쌓는다. 호스트는 이벤트(`cudaEventQuery`)로 끝난 칸만 읽는다. 기다리지 않는다.

### 4.2 지금 팀 학습기에 있는 호스트 동기 (없앨 것)

| 곳 | 무엇 | 바꿀 것 |
|---|---|---|
| `pi05_train/src/tparams.cu:128-134` `grad_norm` | 기울기 제곱합을 호스트로 복사하고 기다린 뒤 `adamw_step` 인자로 넘김 | 노름을 장치 값으로 두고 옵티마이저 커널이 포인터로 읽음 (K9) |
| `tparams.cu:136-167` `step` | 학습률·편향 보정을 호스트에서 계산, 텐서마다 커널 하나, 끝에 동기 | 장치 스텝 카운터, 파라미터 전체를 평평한 버퍼 하나로 → 커널 하나 |
| `trainer.cu:113-116`, `lora.cu:603-606` | 샘플마다 잡음·행동·시간 호스트 → 장치 | 장치 난수(`trng.h` 의 threefry 를 장치에서), 데이터는 로더 슬롯에서 바로 |
| `tkern.cu:400` `flow_inputs_k(..., float t, ...)` | 시간 t 가 호스트 값 인자 → 그래프에서 바꿀 수 없음 | t 를 장치 배열로 (샘플마다 다른 t) |
| `trainer.cu:258-259`, `lora.cu:667-668` | 손실을 호스트로 읽고 기다림 | 장치 링 버퍼 + 비동기 읽기 |
| `trainer`/`lora` "샘플 하나씩(미세배치 1)" | 샘플마다 앞·뒤 (`π05_네이티브엔진.md` 12.1) | 배치 GEMM (M = 배치 × 토큰) |

- 팀 학습기는 openpi 와 비트 단위로 맞추려고 일부러 이렇게 짰다. 우리 모델은 JAX 를 흉내 낼 필요가 없으므로 이 제약이 없다.

### 4.3 환경 커널 (LIMO + OMX) — JSBSim 방식 그대로

SIM_PORTING 2.1 범위(평면 키네마틱 베이스, 팔 PD, 잡으면 붙이기)로 시작한다. 커널 짜는 법은 JSBSim 포팅에서 가져온다.

| JSBSim 기법 (파일) | 우리 환경 커널에서 |
|---|---|
| 스레드 하나 = 기체 하나, `__launch_bounds__(128)` (`fused_core.py` `stick_step`) | 스레드 하나 = 판 하나 |
| 상태를 `Core` 구조체로 레지스터에 올리고(`core_load`), `n_sub` 프레임을 돈 뒤 한 번 씀(`core_store`) | 베이스·팔·잡은 물체를 레지스터에 두고 제어 주기 안의 물리 서브스텝을 다 돈 뒤 한 번 씀 |
| 상태 배치: 짧은 벡터는 판마다 붙여(`pos[3*i]`), 긴 상태는 성분별로 판을 이어(`f[k*N + i]`) → 이웃 스레드가 이웃 주소를 읽음 | 같은 배치. `arm_q[k*N + i]` 꼴 |
| 상수·표를 모델 데이터에서 **소스로 생성**, dtype 별로 정확한 리터럴(`lit()`), 숫자를 두 군데 두지 않음 | URDF → 순기구학 표·관절 한계·카메라 장착을 Rust 도구가 헤더로 생성 (scenemap 의 `r1pro_fk_table.hpp` 처럼 손으로 고치지 않음) |
| NVRTC 실행 중 컴파일 + 소스·옵션·아키텍처 해시로 디스크 캐시 (`nvrtc.py`) | 기본은 nvcc 미리 빌드. 장면별 상수를 박아 넣고 싶을 때만 NVRTC (`libnvrtc` 를 C++ 에서 직접) |
| `--fmad=false` 로 참조판과 같게 반올림 (비용 약 2 %) | 같은 옵션 + 팀 규칙 `-prec-div=true -prec-sqrt=true -ftz=true` (`엔진_자체구현.md` 12.2) |
| `dbg` 포인터: 켜면 프레임마다 중간값 55 개를 씀 (`DBG_KEYS`), 끄면 비용 없음 | 같은 탭. 검증 도구가 모든 중간값을 비교 |
| `reset(mask=)` 호스트 동기 없음, 트림 표를 장치에서 보간 | 끝난 판만 장치에서 다시 배치 |
| 처리량은 약 65,536 기체부터 GPU 를 채움 (`benchmarks.md`) | 판 수를 4,096 → 65,536 으로 늘려 가며 잼 |

- 접촉이 필요해지면 팀 `sim/engine` 의 층 2 모듈(PhysX 5.6.1 과 비트 동일)을 쓴다. 다만 GPU 풀이 진입 고정비가 판 수와 무관하게 3.2 ms(시험)·9 ms(radio)라는 미해결 문제가 있다(`docs/미해결과제.md` B16).

### 4.4 관측 만들기

- 지도 토큰은 5절의 장치 지도에서 만든다. scenemap 은 CUDA 를 쓰지 않는 C++ 다(`scenemap/README.md`).
- 이름 뜻 벡터·생김새 벡터(각 128)는 학습 중에 SigLIP 2 를 돌리지 않는다. 물체 범주·인스턴스마다 미리 계산한 표를 장치에 두고, 관측에는 **번호만** 넣는다. 칸 MLP 가 표에서 모아 쓴다.
  - 이름 흔들기·지우기·틀린 이름(VLA_INPUT 6절)은 번호 바꾸기로 장치에서 한다.
- 벽 상태 56(`walls.cpp`)은 CUDA 로 옮긴다(판 하나 = 블록 하나). 방은 5.2 의 근사로.
- 다음 경유 지점(`sm_snap_place_path`, 격자 A*)은 비싸다. 판마다 거친 격자에서 K 스텝마다 파면 BFS 로 거리장을 다시 만든다(블록 하나) (추정 — 비용은 잴 것).

### 4.5 렌더

- RL 교사는 영상이 없다. 지도 만들기에는 낮은 해상도 깊이 광선만 쓴다(5.2).
- BC VLA 는 RGB 2 장이 필요하다. 팀 `RenderBatch`(`sim/engine/cuda/render/render_batch.cuh`)는 판 E 개의 자세 → 카메라 RGB·깊이를 호스트 왕복 없이 만든다. 층 1 = 층 2 비트 동일이다(`엔진_자체구현.md` 14.6.4).
- 속도는 아직 느리다. 판 64 × 카메라 3(720², 480², 480²)에 1,152 ms = 56 판·프레임/초였다(BEHAVIOR_ENGINE_REVIEW 2절). 224² 속도는 안 쟀다.
- 그래서 BC 데이터는 처음엔 **롤아웃 때 저장한 상태로 학습 시점에 다시 렌더**하거나, 한 번 렌더해 영상으로 저장한다(4.6). 어느 쪽이 빠른지는 잰다.

### 4.6 BC 데이터 로더

| 데이터 | 경로 |
|---|---|
| 우리 GPU 시뮬 궤적 | 상태·지도 토큰 시계열·행동을 저장. 영상은 (a) 학습 때 GPU 렌더 또는 (b) HEVC 로 저장 후 NVDEC |
| 외부 시연 (BEHAVIOR 데모, 다른 로봇) | fasttrain 방식 |

fasttrain 에서 가져오는 것(`docs/학습환경_가속.md`, `fasttrain/README.md`):
- 필요한 패킷만 `pread` → NVDEC(cuvid 직접) → 색 변환 표 커널 → PIL 과 같은 정수 크기 조정 커널 → GPU 슬롯에 바로. 이미지는 호스트를 거치지 않는다.
- 섞기 순서·행동 창은 C++ 생산 스레드, 프레임마다 정해지는 값은 Rust 표(mmap).
- 슬롯 링(기본 6)이 다음 배치를 미리 만든다. 학습기는 C API 로 슬롯을 받는다(`pi05_train` 이 이미 이렇게 함).
- 잰 값: 로더만 289 샘플/s(배치 32, 스레드 4), 원래 openpi 36.1 의 8.0 배. 남은 병목은 NVDEC 이고 사용률은 평균 약 50 % 다.
- 5070 Ti 의 NVDEC 는 **1 개**다(백서). 그래서 외부 영상 데이터는 디코딩이 상한이다. 우리 시뮬 데이터는 (a) GPU 렌더로 디코딩을 피할 수 있다.
- 학습 스텝이 약 160 ms 보다 길면 GPU 가 데이터를 기다리지 않는다(학습환경_가속 4.3, 추정).

## 5. 자라는 지도 아래에서 학습

실제 로봇은 빈 지도(또는 일부 아는 지도)에서 시작해 scenemap 이 돌면서 지도가 커진다. 정책은 **그 순간의 지도**를 본다. 그래서 학습도 에피소드 안에서 지도가 자라야 한다.

### 5.1 GPU 안 지도 만들기 (scenemap 근사)

scenemap 은 CPU 라 학습 고리에 넣을 수 없다. 판마다 장치 메모리에 작은 지도를 두고 scenemap 의 규칙을 흉내 낸다. 규칙과 기본값은 scenemap 코드에서 가져온다.

| 장치 지도 (판마다) | 내용 | scenemap 원본 |
|---|---|---|
| 점유 격자 | int16 로그 오즈(1/256 고정소수점), 칸마다 "본 적 있음" 표시 | `grid.hpp`: `kQ = 256`, `l_hit 0.85`, `l_miss -0.4`, `l_min/max ±4`, 칸 0.05 m |
| 물체 기억 | 최대 K 개: `pos`, `extent`, `first_pos`, `n_obs`, `score`, `last_seen`, `state`, `confirmed`, 후보 여부 | `scenemap.h` `sm_object`, `objmap.hpp` |
| slam 자세 | 참 자세 + 오차 상태(누적 표류 + keyframe 보정) | `slam2d.hpp` |
| 지도 완성도 | 과제 물체 중 확정된 비율, 본 칸 비율 | 새로 (평가용) |

keyframe 마다(스텝마다가 아님, slam2d 의 움직임 거르기처럼 움직였을 때만) K2 커널이 판 하나 = 블록 하나로 한다.

1. **깊이 광선 → 격자.** 깊이 카메라 시야 안에서 거친 광선(예: 가로 64 개, 추정)을 정적 장면에 쏜다. 높이 띠 안에서 맞은 점은 점유, 띠 아래 바닥은 빈 광선 끝(`scan.cpp` 의 가상 2D 스캔 규칙). 광선이 지난 칸은 `l_miss`, 끝 칸은 `l_hit`.
   - 정수 덧셈이라 원자 연산 순서와 무관하게 결과가 같다. 판 안 결정성이 공짜로 생긴다.
   - 칸 0.05 m 에 20 m × 20 m 집이면 400 × 400 이다. 판 4,096 개면 1.3 GB 라 학습용은 0.10 m 칸으로 시작한다(8절, 추정).
2. **물체 보임.** 시야 안 물체마다 상자 위 점 몇 개로 광선을 쏴 가려졌는지 본다. 보이는 화소 수를 깊이·크기로 어림한다.
   - `min_px 6` 보다 작으면 안 보인 것으로 친다. 점 수 `min_points 20` 도 같이 본다.
   - 보이면 위치 = 참 위치 + 깊이 잡음 + 그 순간 slam 자세 오차. 크기 = 참 상자 + 잡음.
   - 서로 다른 keyframe 에서 `confirm 2` 번 보이면 확정. 후보가 `prune_s 10 s` 동안 안 보이면 버림.
   - 사라짐: 시야 안·가림 없음인데 `gone_misses 3` 번 연속, 첫 놓침부터 `gone_min_s 2 s` 넘게. 옮겨짐: `moved_d 0.15 m` 넘게 떨어진 자리에서 보임. 들기: 그리퍼가 닫힐 때 `grasp_r 0.25 m` 안 가장 가까운 확정 물체.
3. **검출 실수.** 놓침 확률, 가짜 물체, 틀린 이름(혼동 표)을 장치 난수로 넣는다. 확률은 실제 검출기 기록에서 맞춘다(추정 — 아직 표 없음).
4. **slam 자세 오차.** 바퀴 오도메트리 표류가 이동 거리·회전에 따라 커지고, keyframe 맞추기가 일부 되돌린다. 기억 물체 위치는 **본 순간의** 자세 오차를 물려받는다(실제와 같음). 오차 크기는 실제 scenemap 기록에서 맞춘다. 참고로 지금 slam2d 는 긴 판에서 최대 7.8·25 cm 였다(`미해결과제.md` B16).
5. **방.** 방 나누기(`rooms.cpp`, 거리 변환 + 지속성)는 무겁다. 처음엔 장면의 참 방 표를 쓰고, 그 방 칸을 일정 비율 이상 봤을 때만 드러낸다(근사).

- 처음 상태는 커리큘럼(5.5)이 정한다: 빈 지도, 일부(무작위 물체 미리 확정, 격자 일부 공개), 전체.
- 이 커널은 scenemap 을 **옮긴 것이 아니라 흉내 낸 것**이다. 그래서 5.2 로 맞는지 잰다.

### 5.2 근사판을 진짜 scenemap 과 맞추기

1. 표본 에피소드 N 개(예: 50, 추정)를 GPU 시뮬에서 돌리며 매 keyframe 의 깊이 영상, 완벽한 검출 마스크(렌더 id 버퍼), proprio 를 저장한다.
2. 같은 입력을 CPU 에서 진짜 scenemap 에 넣는다(C ABI `sm_push_proprio`, `sm_push_image`). 근사판의 지도 기록도 같이 저장한다.
3. keyframe 마다 비교한다.

| 비교 | 지표 | 통과 기준 (추정, 처음 값) |
|---|---|---|
| 확정 물체 집합 | 근사 대 진짜 정밀도·재현율 | 둘 다 0.9 이상 |
| 확정까지 걸린 keyframe | 분포 차이 | 중앙값 차 < 1 keyframe, 같은 참 물체끼리 짝지은 차도 < 1 (≤ 1 이면 확정 규칙을 끈 음성 대조가 통과해 버림 — `training/RL/map_cmp/README.md`) |
| 위치 오차 | 근사−참, 진짜−참 분포 | 분위수 50·90 % 차 ≤ 5 cm |
| 격자 | 점유/빈/모름 3 종 칸 일치율 | ≥ 0.9 |
| 벽 벡터 56 | L1 차 | 진짜끼리(같은 입력 두 번)와 비교해 정함 |
| 지도 토큰 | 칸 토큰 289 값 차 | 위 항목의 결과로 정함 |

- 음성 대조: 근사판의 확정 규칙을 일부러 끄면(1 번 보이면 확정) 비교가 실패해야 한다(JSBSim `--negative` 와 같은 생각).
- 정책 단위 확인: 근사 지도로 학습한 교사를 진짜 scenemap 고리(CPU, 느림)로 몇 판 돌려 성공률이 비슷한지 본다.
- 막힌 것: scenemap 순기구학은 R1 Pro 전용(`fk.cpp`, `r1pro_fk_table.hpp`)이고 proprio 도 R1 Pro 61 개다. LIMO + OMX 순기구학 표를 먼저 넣어야 이 비교를 할 수 있다(BEHAVIOR_ENGINE_REVIEW 5절).

### 5.3 지도 토큰을 시계열로 기록

- 롤아웃 버퍼와 BC 데이터에는 **스텝마다 정책이 실제로 본 지도 토큰**을 저장한다. 마지막 지도 하나가 아니다.
- 저장 형식(작게): 칸 16 개 × 숫자 33(FP16) + 물체 번호(이름·생김새 표 찾기용 int16) + 벽 56 + 방 + 지도 완성도. 스텝당 약 1.3 KB (추정).
- 이렇게 하면 BC 학생이 "그 순간 무엇을 알았는가" 를 그대로 배운다. DAgger 때도 같은 형식을 쓴다.

### 5.4 교사·학생 차이

- 전체 지도를 보는 특권 교사는 **탐색을 배우지 않는다**. 이미 다 아니까 바로 간다. 학생은 같은 정보를 못 받으니 그 행동을 따라 할 수 없다.
- 그래서 교사도 **같은 자라는 지도**를 본다. 교사의 특권은 다음으로 한정한다.
  - 정확한 물리 상태(접촉, 잡기 상태, 관절 토크).
  - **이미 본 물체**의 정확한 자세(잡음·slam 오차 없는 값).
  - 아직 안 본 물체의 위치는 교사도 모른다.
- 탐색은 따로 된 기술로도 둔다. LLM 이 "컵을 찾아" 같은 단계를 주면 탐색 정책이, "컵을 집어" 면 조작 정책이 맡는다. 같은 정책에 단계 종류로 넣을지 따로 둘지는 잰 뒤 정한다.
- 남은 차이는 **DAgger** 로 메운다. 학생이 GPU 시뮬에서 돌고, 교사가 같은 상태에서 행동 표를 단다. 교사 앞 계산은 작으므로 같은 롤아웃 그래프에 넣는다.

### 5.5 커리큘럼: 전체 지도 → 부분 → 빈 지도

| 단계 | 처음 지도 | 넘어가는 조건 (추정) |
|---|---|---|
| C0 | 전체(모든 물체 확정, 격자 공개, 자세 오차만) | 성공률 ≥ 0.8 |
| C1 | 부분(물체 30–70 % 미리 확정, 격자 일부) | 성공률 ≥ 0.7 |
| C2 | 빈 지도 | — |

- 판마다 단계를 섞는다(예: 20 % 는 앞 단계 유지, 추정). 앞에서 배운 것을 잊지 않게 한다.
- 단계 비율은 장치 값이라 다시 컴파일·다시 잡기 없이 바꾼다(4.1).

### 5.6 지도 완성도로 나눠 평가

- 에피소드 시작 때 완성도(과제 물체 중 확정 비율)를 0, 0–30 %, 30–70 %, 70–100 %, 100 % 로 나눈다.
- 칸마다 교사·학생 성공률과 걸린 스텝을 낸다. 빈 지도에서 크게 떨어지면 탐색이 약한 것이다.
- VLA_INPUT 7절의 "지도 없음 대 있음", "`gt` 지도 대 `slam` 지도" 도 같은 표에 넣는다.

## 6. 커널 합치기 계획

| 커널 | 합치는 것 | 이유 · 근거 |
|---|---|---|
| K1 `env_step` | 행동 풀기, 베이스 적분, 팔 PD, 순기구학, 잡기, 보상, 끝 판정, 리셋 | 중간값을 전역 메모리에 안 쓴다. JSBSim 은 같은 식을 torch + CUDA 그래프로 돌린 것보다 커널이 87–227 배 빨랐다(`benchmarks.md`, 그래프는 실행 비용만 없애고 메모리 왕복은 남음) |
| K2 `map_update` | 깊이 광선, 격자, 물체 보임, 물체 기억, 벽 벡터 | 판 하나 = 블록 하나. 격자 창을 공유 메모리에 올려 광선이 여러 번 읽음 (SM 당 128 KB, 블록 한도는 확인할 것) |
| K3 `obs_policy` | 칸 고르기·정규화, 칸 MLP, 집합 인코더, 정책·가치, 표본, logp | 교사는 작다(1–5M, 추정). 가중치가 L2(48 MB)에 다 들어간다. 블록 하나가 판 64 개를 맡아 64 행 타일로 텐서 코어 MLP |
| K1+K3 | **시뮬 스텝 + 관측 + 정책 MLP 한 커널** | 블록 하나가 판 64 개: 스레드마다 판 하나로 물리를 돌고, 관측을 공유 메모리 타일에 쓰고, 같은 블록이 `mma.sync` 로 MLP. 전역 메모리 왕복 0 (추정 — K1·K3 따로 잰 뒤 합칠 가치가 있으면) |
| K4 `gae_norm` | GAE 역방향 훑기, 반환값, 이득 정규화 통계 | 판마다 스레드 하나가 T 스텝을 거꾸로. 통계는 블록 합 + 장치 원자 합 |
| K5 `loss` | PPO 자르기 + 가치 + 엔트로피 (RL), flow matching (BC) | 마지막 층 끝단에서 바로 dlogits·dv 를 냄. 팀 `flow_loss_k` 가 이미 손실과 dv 를 한 커널로 냄 |
| K6 GEMM 끝단 | 편향 + 활성 + 잔차 + **다음 층 FP8 양자화 + amax** | 팀 손 GEMM 이 이미 끝단에 RoPE+KV 쓰기, gelu·up, 게이트 잔차를 합침(`π05_네이티브엔진.md` 4절) |
| K7 정규화 + 양자화 | RMSNorm/LayerNorm(FP32 통계) → FP8 출력 + 배율 | 정규화 출력을 따로 쓰지 않음 |
| K8 어텐션 | QKᵀ, 온라인 softmax(FP32), PV, 역전파 | 팀 학습기는 softmax 를 따로(`softmax_fwd_k`, `smb_fwd_k`), dq·dk 를 bf16 세 조각으로 정확히(JAX 비트 맞추기용) 계산한다. 우리 모델은 그럴 필요가 없어 flash 방식 하나로 |
| K9 옵티마이저 | 기울기 제곱합(2 단계: 블록 합 → 마지막 블록이 노름·자르기 배율 계산), AdamW, EMA, BF16/FP8 사본 + amax | 지금은 텐서마다 커널 + 호스트 동기(4.2). 평평한 버퍼 하나 + 커널 둘 |
| K10 난수 | 행동 표본, flow 잡음·시간, 커리큘럼, 검출 실수 | threefry 를 장치에서. 팀 `trng.h` 는 JAX 와 비트 같은 키 사슬이 있다 |
| K11 렌더 끝단 (BC) | 렌더 → 정규화 → 패치로 자르기 → 첫 선형층 입력 | 영상을 따로 쓰지 않음 |

## 7. 정밀도

### 7.1 층마다

| 부분 | 앞 | 뒤 | 이유 |
|---|---|---|---|
| 몸통 선형층(어텐션 사영, MLP) | E4M3 × E4M3, FP32 누산 | dgrad: dy E5M2 × W E4M3. wgrad: 처음엔 BF16 | 큰 GEMM 이 FP8 이득의 거의 전부. wgrad 는 전치 피연산자가 필요해 나중에(7.3) |
| 첫 층(패치 임베딩, 칸 MLP 입력 289), 마지막 층(행동 출력, 가치, logits) | BF16 | BF16 | 작고 예민하다 |
| LayerNorm/RMSNorm | 통계·배율 FP32 | FP32 | SigLIP 2 는 TRT FP16 그대로면 코사인 0.64–0.74 로 망가졌고 LayerNorm 을 FP32 로 고정하면 0.9999 였다(`docs/clip_candidates.md` 2.4, `src/scene_graph/clip/tools/build_engine.py`). 팀 학습기도 LayerNorm 변수는 f32 |
| softmax | FP32 | FP32 | 지수·합. 팀 학습기도 f32 softmax |
| 어텐션 PV | BF16 (처음) | BF16 | 확률 값 범위가 좁아 E4M3 로 잃는 것이 큼 (추정) |
| flow matching 시간 임베딩(sin/cos), adaRMS 변조, x_t = t·잡음 + (1−t)·행동, 손실 | FP32 | FP32 | 작은 계산이고 행동 정밀도를 바로 정함. 팀 `flow_inputs_k`·`flow_loss_k` 도 f32 |
| 잔차 흐름 | BF16 (작은 모델은 FP32) | | |
| 기울기 누적·감소 | FP32 | | 팀 학습기도 f32 로 누적 |
| 원본 가중치, Adam m·v, EMA | FP32 | | 팀 `adamw_k` 와 같음. 16 GB 에 들어감(8절) |
| RL 교사 MLP | BF16 | BF16 | GEMM 이 작아 지연 시간에 묶인다. FP8 이득이 거의 없다 (추정) |
| 큰 VLA 의 얼린 Qwen 몸통 | 가중치 E4M3(채널별 배율), 앞만 | 없음 | 앞 GEMM 상한 2 배. 출력 코사인을 BF16 과 비교(9절) |

### 7.2 배율

1. **텐서별 배율, 장치 안에서.** 앞 커널 끝단이 amax 를 장치 값으로 남기고, 다음 커널이 그 값으로 양자화한다. 호스트는 모른다. 지연 배율(amax 기록 16 칸, 추정)도 같은 방식으로 그래프 안에서 된다.
2. 이상값 때문에 손실이 흔들리면 **블록 배율 MXFP8**(`kind::mxf8f6f4.block_scale`, 32 개마다 UE8M0)로 바꾼다. sm_120a 에서 된다(2.2).
3. NVFP4(`m16n8k64`, 16 개마다 UE4M3)는 얼린 몸통 추론 가중치 실험에만 쓴다. 학습에는 쓰지 않는다.
4. FP8 + **FP16 누산**(351.5 TFLOPS)은 K 가 길면 누산 오차가 크다. 기준판이 아니다. 9절 사다리를 통과한 층에만 시험한다.

### 7.3 FP8 피연산자 배치

- `mma.sync` 8 비트 피연산자는 K 연속 배치가 기본이다. cuBLASLt 도 Blackwell GeForce 에서 FP8 을 TN 배치만 받는다(2.2).
- 팀 `tgemm.cuh` 는 bf16 에서 `ldmatrix.trans` 로 네 배치를 다 처리한다. FP8 에서는 다음 둘 중 하나.
  - 양자화 커널(K6·K7)이 FP8 값과 **전치한 FP8 사본**을 같이 쓴다. 메모리는 늘지만 단순하다.
  - sm_120a 의 8 비트 `ldmatrix` 전치 모양을 쓴다(있는지 컴파일로 확인, 2.2).

## 8. 메모리 (16 GB)

모든 값 (추정). 변수 하나당 학습 비용 = FP32 원본 4 + FP32 기울기 4 + Adam 8 + EMA 4 + BF16 사본 2 = 22 B.

### 8.1 RL 교사 (판 4,096, 롤아웃 64 스텝)

| 항목 | 계산 | 크기 |
|---|---|---|
| 시뮬 상태 | 판당 약 1 KB (베이스·팔·물체 64 개 자세) | 4 MB |
| 점유 격자 | 0.10 m 칸, 20 m × 20 m = 200 × 200 × int16 = 80 KB/판 | 330 MB |
| 물체 기억 | 64 개 × 64 B | 1 MB |
| 정적 장면 (광선용 BVH, 판끼리 공유) | 장면 수에 따라 | 0.2–1 GB |
| 롤아웃 버퍼 | 판 × 스텝 × (지도 토큰 1.3 KB + 몸 상태·행동·logp·가치·보상 0.2 KB) | 4,096 × 64 × 1.5 KB ≈ 0.4 GB |
| 정책 | 5M × 22 B | 0.1 GB |
| 미니배치 활성값 | 65,536 행 × 512 폭 × 6 층 × 2 B × 3 | 1.2 GB |
| 합 | | **약 2.5–3 GB** |

- 남는 메모리로 판 수를 늘리거나 BC 학습을 같이 돌린다(DAgger).
- 격자를 0.05 m 로 하면 판당 320 KB, 4,096 판에 1.3 GB 다.

### 8.2 BC

| 항목 | 작은 VLA (150M 학습) | 큰 VLA (Qwen3.5-2B 얼림 + 200M 학습) |
|---|---|---|
| 얼린 가중치 | — | BF16 4.3 GB (파일 크기) 또는 FP8 약 2.2 GB |
| 학습 변수 × 22 B | 3.3 GB | 4.4 GB |
| 활성값 (층 재계산 켬) | 1–2 GB | 2–4 GB |
| 로더 | +0.4–0.7 GB (fasttrain 실측) | 같음 |
| 렌더 (온라인일 때) | 정적 장면 + 판당 작업 공간 (팀 렌더: 정적 1.87 GiB, 판당 224 × 3 에 12 MB) | 같음 |
| 합 | **약 6–8 GB** | **약 10–14 GB** |

- 참고 실측: 팀 π0.5 expert 모드(3.3B 중 4.3억 학습)는 배치 32, 옵티마이저 상태를 호스트에 둔 `--offload 2` 로 10.7 GiB, 스텝 3.0 s 였다(`pi05_train/README.md`).
- 큰 VLA 를 RL 시뮬과 같이 돌리면 빠듯하다. DAgger 때는 롤아웃 판 수를 줄인다.
- 호스트로 내리기(offload)는 PCIe 를 매 스텝 쓴다. 그래프 안에 잡을 수는 있지만 느리다. 16 GB 에 들어가면 쓰지 않는다.

## 9. 검증 사다리 (JSBSim 방식)

JSBSim 포팅의 방법: **정답의 한가운데 상태를 통째로 심고 한 프레임만 돌려** 모든 중간값을 비교한다. 누적이 없으니 다른 것은 모델 차이다(`docs/verification.md` 2절). 긴 궤적 오차는 충실도 시험이 아니다(같은 JSBSim 도 1e-6 kt 차이로 수 km 갈라짐). 팀도 같은 틀을 쓴다(`π05_네이티브엔진.md` 3절, `학습환경_가속.md` 3절).

| 층 | 무엇 | 정답 | 기준 | 도구 |
|---|---|---|---|---|
| V0 수학 | sinf·atan2f 등 | glibc | 비트 동일, float 2³² 전수 | 팀 `test_glibc_trig` 방식 |
| V1 환경 커널 | 한 스텝, 모든 중간값(dbg 탭) | CPU C++ 참조판 | `-fmad=false` 로 비트 동일 목표. 안 되면 상대 오차 + 이유 기록 | `env_verify --check` (넘으면 종료 코드 1) |
| V2 지도 근사 | keyframe 마다 지도 | 진짜 scenemap | 5.2 표 | `map_verify --check` |
| V3 GEMM | BF16, FP8 GEMM | CPU FP64 | ① **같은 양자화 입력**으로 FP64 와 비교 → 누산 오차만 남음(명령 확인) ② 양자화 안 한 입력과 비교 → 양자화 오차 | 팀 `tgemm_test.cu`(CPU 배정밀도 대비 1.8e-7) 확장 |
| V4 층 앞·뒤 | 층 하나 | CPU FP64 참조판 + FP64 유한 차분 | 우리 오차 ≤ 2 × 바닥 | `layer_verify` |
| V5 학습 한 스텝 | 손실, 모든 기울기, AdamW 1·10 스텝 뒤 갱신량, EMA | CPU FP64 참조판 | 우리 오차 ≤ 2 × 바닥 | 팀 `train_verify.cpp` 틀 |
| V6 그래프 = 즉시 실행 | 같은 입력 | 그래프 없이 돈 결과 | 비트 동일 | |
| V7 결정성 | 같은 씨앗 두 번 | 자기 자신 | 비트 동일 (정수 원자, 고정 합산 순서, 커널 안 결정적 split-K — 팀 `gemm.cuh` 방식) | |
| V8 FP8 대 BF16 학습 | 손실·성공률 곡선 | BF16 같은 설정 | 9.1 | |
| V9 폐루프 | 성공률 | 진짜 scenemap 고리, 이후 실제 로봇 | 두 표본 검정 | |

- **바닥(노이즈 바닥)** = 같은 계산을 정밀도만 바꿔 돌린 차이. 우리 판에서는 "BF16 GPU 대 FP64 CPU" 를 바닥으로 하고 FP8 은 이것과 견준다. 팀 규칙 "우리 ≤ 2 × 바닥" 을 그대로 쓴다.
- **음성 대조**: 층마다 일부러 버그 하나(배율 하나 빼기, 확정 규칙 끄기, 부호 하나 뒤집기)를 넣은 판이 반드시 실패해야 한다(JSBSim `fdm_verify --negative`).
- **Python 기준값**: CPU FP64 참조판 자체가 맞는지는, 같은 작은 신경망을 PyTorch/JAX 로 한 번 짜서 기울기 덤프를 뽑아 맞춘다. 학습·추론 경로에는 들어가지 않는다. 팀의 `train_ref.py` 와 같은 쓰임이다.
- 1 스텝 Adam 은 기울기가 0 근처인 원소의 부호가 바뀌면 그 원소 갱신이 ±lr 로 뒤집힌다. 팀도 이 예외를 봤다(`π05_네이티브엔진.md` 12.3). 10 스텝 비교를 같이 본다.

### 9.1 FP8 이 학습을 해치지 않는지

1. 같은 설정·같은 씨앗 3 개씩 BF16 과 FP8 을 돌린다.
2. RL: 환경 스텝 대비 반환값·성공률 곡선. BC: 검증 flow 손실, 행동 MSE, 시뮬 폐루프 성공률.
3. 통과: 정해진 예산 끝에서 FP8 평균이 BF16 씨앗 퍼짐(표준편차 1 배) 안 (추정 기준).
4. 학습 중 감시(장치 카운터, 로그 링 버퍼로): 층마다 amax, FP8 넘침·밑넘침 비율, 고정 체크포인트에서 BF16 기울기와의 코사인(≥ 0.99, 추정 기준).
5. 문제가 생긴 층은 BF16 으로 되돌린다. 층마다 켜고 끄는 표를 설정에 둔다.

## 10. 예상 속도와 근거

### 10.1 근거가 되는 측정

| 측정 | 값 | 출처 |
|---|---|---|
| JSBSim CUDA 커널 대 torch GPU + CUDA 그래프 (262,144 기체) | 2.95 G 대 33.8 M 기체·프레임/s = 87 배 | `benchmarks.md` |
| 같은 비교, 1,024 기체 | 157 M 대 0.77 M = 204 배 | 같음 |
| torch 즉시 실행 → CUDA 그래프 (1,024 기체) | 0.09 M → 0.77 M = 8.6 배 | 같음 (작은 배치는 실행 비용이 지배) |
| CPU SB3 PPO → GPU 시뮬 PPO (같은 정책 256×2) | 약 4,200 → 525,692 env-step/s = 125 배, 대부분 병렬 판 수·큰 미니배치 덕 | 같음 |
| 물리만 torch → 커널로 바꿈 (PPO, 1024×2 정책) | 한 바퀴 19.0 s → 15.6 s (−18 %). 물리는 바퀴의 0.2 % 가 됨 | 같음 |
| FP8 (FP32 누산) 대 BF16 이론 상한 | 175.8 / 87.9 = 2 배 | 백서 |
| fasttrain 로더 | 원래의 7.3–8.0 배 | `학습환경_가속.md` |

### 10.2 그래서 우리 경우

- **Amdahl**: 시뮬이 학습 시간의 비율 p 면 시뮬을 아무리 빠르게 해도 최대 1 / (1 − p) 다(`benchmarks.md`). JSBSim 예에서 커널 뒤 병목은 정책 추론·관측·보상·갱신이었다.
- 그러므로 차례는 (1) 판 수와 미니배치를 키울 수 있게 모든 것을 GPU 로, (2) 호스트 동기 제거와 그래프, (3) 정책·학습 쪽 합치기, (4) 그 뒤에 FP8 이다.
- RL 교사 (전부 추정):
  - 정책 5M 변수면 앞 계산은 환경 스텝당 약 10 MFLOP, 학습까지 약 3 배. 초당 10⁶ 환경 스텝이면 약 30 TFLOPS 가 필요하다. BF16 상한 87.9 의 3 분의 1 이라, 작은 GEMM 효율(20–40 %)을 생각하면 **10⁵–10⁶ 환경 스텝/s** 가 범위다.
  - 지도 갱신은 keyframe 에만(예: 5 스텝마다) 광선 64 개 × 칸 100 개 = 판당 6,400 칸. 판 4,096 개면 2,600 만 칸 갱신이라 1 ms 안팎으로 본다.
  - 여기서 FP8 이득은 작다(7.1).
- BC 큰 VLA (전부 추정): 얼린 Qwen 앞 GEMM 이 대부분이면 FP8 가중치로 GEMM 부분이 최대 2 배. 정규화·어텐션·옵티마이저·로더는 그대로라 전체 1.3–1.6 배.
- BC 작은 VLA: 렌더 또는 NVDEC 가 상한일 수 있다. 먼저 잰다.

### 10.3 먼저 잴 것

1. `sm_120a` 로 빌드한 FP8 `kind::f8f6f4` 손 GEMM 대 팀 BF16 손 GEMM, 우리 모델 모양에서 (그리고 cuBLASLt FP8 TN 을 비교용으로).
2. 조건 노드 WHILE 한 바퀴의 고정비, PDL 켬/끔.
3. LIMO + OMX 환경 커널 하나의 판·스텝/s (4,096 → 65,536 판).
4. 지도 갱신 커널 시간 (keyframe 비율별).
5. RL 한 바퀴의 구간별 시간: 롤아웃(시뮬·지도·정책) 대 갱신. 이것으로 p 를 안다.
6. 224² 렌더 속도(`RenderBatch`)와 NVDEC 1 개의 디코딩 상한.
7. 바퀴당 호스트 동기 수 (Nsight Systems 로 0 인지 확인).

## 11. 단계

| 단계 | 할 일 | 통과 기준 |
|---|---|---|
| G0 | 측정 틀: FP8 GEMM 시험(V3), 조건 노드·PDL 고정비, cuBLASLt 비교 | V3 통과, 10.3 의 1–2 숫자 |
| G1 | LIMO + OMX 환경 커널 + CPU 참조판 + dbg 탭, 장치 리셋 | V1 비트 동일(또는 이유 기록), 판·스텝/s |
| G2 | GPU 지도 근사(5.1) + scenemap 에 LIMO + OMX 순기구학 + 비교 도구(5.2) | 5.2 표 통과, 음성 대조 실패 |
| G3 | RL 교사 PPO 전체를 그래프 둘로, BF16 | V4–V7 통과, 바퀴당 호스트 동기 0, 환경 스텝/s |
| G4 | 자라는 지도 커리큘럼(5.5) + 완성도별 평가(5.6) | 빈 지도 성공률이 목표 이상 (목표는 G3 결과 보고 정함) |
| G5 | BC 작은 VLA 학습기(BF16), 데이터 경로(GPU 렌더 또는 NVDEC), DAgger | V5 통과, 폐루프 성공률 |
| G6 | FP8: 앞 → dgrad → wgrad 순서 | 9.1 통과 |
| G7 | BC 큰 VLA (얼린 Qwen FP8 가중치) | 몸통 출력 코사인 평균 ≥ 0.999·하위 1 % ≥ 0.99 (팀 `sgclip_encoder` 기준과 같게), 9.1 |

## 12. 열린 문제

- **FP8 혼합 형식**: `kind::f8f6f4` 에서 e5m2 × e4m3 를 한 명령에 줄 수 있는지 컴파일로 확인한다. 안 되면 dgrad 의 dy 도 E4M3 + 배율로 한다.
- **8 비트 `ldmatrix` 전치**: sm_120a 에 있는지 확인. 없으면 전치 사본(7.3).
- **블록당 공유 메모리 한도**: SM 당 128 KB 는 백서 값이다. 블록 하나가 쓸 수 있는 최대값은 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 으로 잰다. 격자 창 크기가 여기에 달렸다.
- **CUDA 12.8 고정**: `sm_120f` 와 mxf4nvf4 4X UE8M0 는 12.9·13.1 이 필요하다. 저장소 기준(12.8)을 바꿀지는 사용자 결정이다.
- **검출 실수·slam 오차 모델**: 실제 기록에서 맞출 값이 아직 없다.
- **224² 렌더 속도**: 안 쟀다. BC 를 온라인 렌더로 할지 저장 영상으로 할지가 이것에 달렸다.
- **scenemap 이 R1 Pro 전용**: 비교(5.2) 전에 LIMO + OMX 순기구학과 proprio 형식을 넣어야 한다.
- **Python 기준값 범위**: SIM_PORTING 4절에서 물은 것과 같다. 이 문서는 "오프라인 한 번 덤프만" 으로 가정했다.

## 출처

- 팀·사용자 코드: `~/juyoung020/jsbsim-f16-cuda` (`README.md`, `README.ko.md`, `docs/porting_notes.md`, `docs/verification.md`, `docs/benchmarks.md`, `docs/api.md`, `jsbsim_f16_cuda/fused_core.py`, `nvrtc.py`), `src/behavior-2026/src/vla/{pi05_train,fasttrain,pi05_native}`, `src/behavior-2026/src/sim/engine`, `src/behavior-2026/docs/{학습환경_가속.md, π05_네이티브엔진.md, 엔진_자체구현.md, 미해결과제.md}`, `src/scene_graph/scenemap` (`README.md`, `include/scenemap/{objmap,grid,slam2d}.hpp`), `src/scene_graph/clip/tools/build_engine.py`, `docs/clip_candidates.md`.
- NVIDIA RTX Blackwell GPU Architecture 백서, 부록 B 표 5 (RTX 5070 Ti): <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>
- PTX ISA 8.7 (CUDA 12.8) 새 기능 — sm_120/sm_120a, FP8 m16n8k16 f16 누산: <https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html>
- CUTLASS 아키텍처 매크로: <https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/config.hpp>, <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/config.h>
- CUTLASS 이슈 #3044 (SM120 FP8 `kind::f8f6f4`, SM89 경로 없음): <https://github.com/NVIDIA/cutlass/issues/3044>
- Colfax, "CUTLASS Tutorial: NVFP4 Blockscaled GEMM on NVIDIA RTX PRO Blackwell GPUs (SM12x)": <https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/>
- cuBLAS FP8 TN 제약(Blackwell GeForce 12.x) 문장 인용: <https://github.com/pytorch/pytorch/issues/198126>, <https://github.com/pytorch/ao/issues/4932>
- CUDA 그래프 조건 노드: <https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes>, <https://docs.nvidia.com/cuda/archive/12.3.2/cuda-runtime-api/structcudaConditionalNodeParams.html>
