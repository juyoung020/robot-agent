// 신경망 연산 실행 함수(C++ 쪽). 모든 함수는 비동기이고 호스트 값은 모양·하이퍼파라미터 상수뿐이다(그래프로 잡을 때 고정).
// 바퀴마다 바뀌는 값(학습률, 바퀴 번호, 이득 통계, 기울기 노름)은 장치의 TrainState 에서 커널이 읽는다(4.1, 4.2).
#pragma once
#include <cstdint>

#include <cuda_runtime_api.h>

#include "net.h"

namespace net {

// 장치에 하나 있는 학습 상태(바퀴마다 바뀌는 값 + 이번 바퀴 통계 누적)
struct TrainState {
  long long iter;        // 끝난 바퀴 수(난수 열쇠, 기록 칸)
  long long adam_t;      // Adam 스텝 수
  float lr, gnorm, clip_coef, bc1, bc2;
  float adv_mean, adv_std;
  // 이번 바퀴 누적(로그 커널이 기록하고 0 으로)
  float s_pg, s_vl, s_kl, s_clip, s_ent, s_gnorm;
  int n_mb;
  float r_sum, ep_ret_sum, ep_len_sum, v_sum;
  int n_succ, n_coll, n_tout;
  float comp_sum;        // 지도 완성도(과제 물체 확정) 합
  uint32_t act_mask;     // 학습하는 행동 비트(커리큘럼 단계마다 장치 값 — 다시 잡기 없이 바꿈). 꺼진 행동은 0 고정(표본·logp·엔트로피에서 뺌)
  int act_pad;
};

struct LossHyper {
  float clip, vclip, vf_coef, ent_coef;
  int adaptive_lr;
  float kl_target, lr_min, lr_max;
  int act_dims;   // (예전 값, 커널은 쓰지 않음) 학습하는 행동은 장치 값 TrainState::act_mask
  float bound_coef;   // 정책 평균이 행동 자르기(±1) 밖이면 coef·(|μ|−1)² (행 평균). 0 이면 끔(G3 와 비트 같음)
};
struct AdamHyper {
  float b1, b2, eps, max_norm;
};

// ---- G6 FP8 ----
// 켬 표(비트): FP8_FWD 1 = 앞(E4M3×E4M3), FP8_DGRAD 2 = dX(E5M2 dZ × E4M3 W), FP8_WGRAD 4 = dW(E5M2 dZᵀ × E4M3 X). LayerDesc::fp8 인 층만.
// 호스트 값이고 그래프를 잡을 때 고정된다(잡기 전에 정함). 기본 0 = 모두 BF16(G3–G5 와 비트 같음).
enum Fp8Bits : int { FP8_FWD = 1, FP8_DGRAD = 2, FP8_WGRAD = 4 };
void set_fp8(int mask);
int fp8_mask();
// role 0 앞 / 1 dgrad / 2 wgrad. 다룰 수 없는 모양이면 false(그때는 BF16 길)
bool fp8_gemm(int role, const struct GemmP* ps, int np, int gz, int epi, cudaStream_t st);

// ---- GEMM ----
// 앞: X[M][L.K] → out. 숨은 층은 bf16 [M][L.ldo] (ELU), 머리(ACT_LIN)는 f32 [M][L.ldo]
void gemm_fwd(const LayerDesc& L, const uint16_t* X, int M, const uint16_t* Wb, void* out, cudaStream_t st);
// dX 를 이전 층 dZ 로: dZprev[M][Np] = (dZ[M][L.N] · W[:, 0:Np]) ⊙ elu'(Xin[M][0:Np])   (Xin = 이 층 입력 = 이전 층 출력)
void gemm_dx_dact(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, const uint16_t* Xin, int Np, uint16_t* dZprev, int bug,
                  cudaStream_t st);
// 첫 몸통 층의 dX 중 집합 칸 [0, POOL_W) 만: dpool[M][POOL_W] (=, 또는 += )
void gemm_dx_pool(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, float* dpool, bool accumulate, cudaStream_t st);
// dW 조각 부분합: ws[split][L.N][L.K], 조각 = kchunk 행
void gemm_dw(const LayerDesc& L, const uint16_t* dZ, const uint16_t* X, int M, float* ws, int kchunk, cudaStream_t st);
// 정책·가치 사슬의 같은 모양 층 둘을 한 번에(결과는 따로 부른 것과 같음 — 실행 수만 줄임)
void gemm_fwd2(const LayerDesc& L, const uint16_t* XA, const uint16_t* XC, int M, const uint16_t* WA, const uint16_t* WC, void* outA, void* outC,
               cudaStream_t st);
void gemm_dx_dact2(const LayerDesc& L, const uint16_t* dZA, const uint16_t* dZC, int M, const uint16_t* WA, const uint16_t* WC, const uint16_t* XinA,
                   const uint16_t* XinC, int Np, uint16_t* dZprevA, uint16_t* dZprevC, int bugA, cudaStream_t st);
void gemm_dw2(const LayerDesc& L, const uint16_t* dZA, const uint16_t* dZC, const uint16_t* XA, const uint16_t* XC, int M, float* wsA, float* wsC,
              int kchunk, cudaStream_t st);
inline int dw_splits(int M, int kchunk) { return (M + kchunk - 1) / kchunk; }
// 모든 층의 조각을 고정 순서로 더해 기울기 버퍼(평평)에
struct DwJob { const float* ws; float* g; int n; int splits; };
void dw_reduce(const DwJob* jobs, int njobs, cudaStream_t st);

// ---- 집합(칸 16 개, 빈 칸 가림) ----
// s2o[M*16][S_H] → x0[M][X0_W] 의 [0,64) 평균, [64,128) 최댓값, amax[M][S_H] (최댓값 칸 번호, 없으면 255)
void pool_fwd(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax, cudaStream_t st);
// dpool[M][POOL_W] → dZ_S2[M*16][S_H] = (평균 몫 + 최댓값 몫) ⊙ elu'(s2o)
void pool_bwd(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2, cudaStream_t st);
// 칸 MLP 앞 묶음: S1 → S2 → 집합을 한 커널로(따로 커널 gemm·pool 과 같은 결과). s1o·s2o·x0 의 집합 칸·amax 를 씀. sin = 304 칸 줄 [M·16][SLOT_IN]
void slot_fwd(const uint16_t* sin, const uint32_t* mask, const uint16_t* W1, const uint16_t* W2, int M, uint16_t* s1o, uint16_t* s2o, uint16_t* x0,
              uint8_t* amax, cudaStream_t st);
// 칸 MLP 뒤 묶음: pool_bwd → dW S2 · dX S2 → dW S1 을 한 커널로(같은 결과). dW 부분합 ws2·ws1 은 gemm_dw 와 같은 자리.
// dz2_out·dz1_out 이 nullptr 이 아니면 dZ S2·dZ S1 도 전역에 씀(검증용). kchunk 는 32 의 배수
void slot_bwd(const float* dpool, const uint16_t* s2o, const uint16_t* s1o, const uint16_t* sin, const uint32_t* mask, const uint8_t* amax,
              const uint16_t* W2, int M, int kchunk, float* ws2, float* ws1, uint16_t* dz2_out, uint16_t* dz1_out, cudaStream_t st);
// 줄인 칸 줄(net.h SLOT_C) + 얼린 이름·생김새 표(장치 bf16 [행][128]) — 묶음 커널이 공유 메모리에서 304 칸 줄로 펼침(같은 값 → 같은 결과)
struct SlotC { const uint16_t* sc; const uint16_t* name; const uint16_t* app; };
void slot_fwd_c(const SlotC& in, const uint32_t* mask, const uint16_t* W1, const uint16_t* W2, int M, uint16_t* s1o, uint16_t* s2o, uint16_t* x0,
                uint8_t* amax, cudaStream_t st);
void slot_bwd_c(const float* dpool, const uint16_t* s2o, const uint16_t* s1o, const SlotC& in, const uint32_t* mask, const uint8_t* amax,
                const uint16_t* W2, int M, int kchunk, float* ws2, float* ws1, uint16_t* dz2_out, uint16_t* dz1_out, cudaStream_t st);
// 줄인 칸 줄 rows 개 → 304 칸 줄(예전 따로 커널 길 NET_SLOT_OLD·검증용)
void slot_expand(const SlotC& in, long long rows, uint16_t* sin, cudaStream_t st);

// ---- PPO 손실(K5): 평균·가치 머리 출력 → dZ(bf16) + log σ 기울기·통계 부분합 ----
struct LossIn {
  const float* mean;     // [M][8]
  const float* val;      // [M][8], 0 번
  const float* logstd;   // 변수 버퍼 안 log σ [8]
  const float* act;      // [M][8]
  const float* oldlogp;  // [M]
  const float* oldv;     // [M]
  const float* adv;      // [M] (정규화 전)
  const float* ret;      // [M]
};
constexpr int LOSS_NT = 256;
constexpr int LOSS_NQ = 16;   // 블록 부분합 칸: dlogσ 8, pg, vl, kl, clip
inline int loss_blocks(int M) { return (M + LOSS_NT - 1) / LOSS_NT; }
void ppo_loss(const LossIn& in, int M, const TrainState* ts, const LossHyper& h, uint16_t* dzA, uint16_t* dzC, float* partial, int bug, cudaStream_t st);
// 부분합을 고정 순서로 → log σ 기울기(g_logstd), 통계 누적, (켜면) KL 로 학습률 조절
void ppo_loss_reduce(const float* partial, int nblk, int M, const LossHyper& h, float* g_logstd, const float* logstd, TrainState* ts, cudaStream_t st);

// ---- 옵티마이저(K9): 기울기 노름(2 단계, 고정 순서) → 자르기·Adam → bf16 사본 ----
constexpr int GN_BLOCKS = 240;
void adam_step(float* P, const float* G, float* m, float* v, uint16_t* Pb, long long n, float* gn_partial, TrainState* ts, const AdamHyper& h,
               cudaStream_t st);
void to_bf16(const float* P, uint16_t* Pb, long long n, cudaStream_t st);

}  // namespace net
