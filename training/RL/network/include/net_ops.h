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
};

struct LossHyper {
  float clip, vclip, vf_coef, ent_coef;
  int adaptive_lr;
  float kl_target, lr_min, lr_max;
  int act_dims;   // 앞에서부터 이만큼만 학습(나머지 행동은 0 고정, logp·엔트로피에서 뺌)
};
struct AdamHyper {
  float b1, b2, eps, max_norm;
};

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
inline int dw_splits(int M, int kchunk) { return (M + kchunk - 1) / kchunk; }
// 모든 층의 조각을 고정 순서로 더해 기울기 버퍼(평평)에
struct DwJob { const float* ws; float* g; int n; int splits; };
void dw_reduce(const DwJob* jobs, int njobs, cudaStream_t st);

// ---- 집합(칸 16 개, 빈 칸 가림) ----
// s2o[M*16][S_H] → x0[M][X0_W] 의 [0,64) 평균, [64,128) 최댓값, amax[M][S_H] (최댓값 칸 번호, 없으면 255)
void pool_fwd(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax, cudaStream_t st);
// dpool[M][POOL_W] → dZ_S2[M*16][S_H] = (평균 몫 + 최댓값 몫) ⊙ elu'(s2o)
void pool_bwd(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2, cudaStream_t st);

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
