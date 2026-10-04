// 생성: training/RL/observation/tools/tok_stats.cu (손으로 고치지 않는다). 지도 토큰 특징 111 개의 obs.h feat_pre 값 평균·표준편차.
// 자료: G1 A1·A2 각 N 512 × T 400 스텝(접근 제어 + 넷 중 하나 돌며 달림, 팔 풂), 처음 지도 C0 0.34·C1 0.33·C2. env-step 409600 개.
// 0/1·확률 값은 μ 0·σ 1(정규화 안 함). 표본이 없거나 σ < 1e-3 이면 σ 1. 특징 번호: 칸 0..32 | 벽 33..88 | 안 본 곳 89..96 | 방 97..106 | 경유 지점 107..110
#pragma once
namespace tokn {
constexpr int N_FEAT = 111;
#define TOKN_MU {0x1.23c478p-2, -0x1.f64bd8p-5, 0x1.3eee5ap-3, 0x1.fb208ep-5, -0x1.ec68b8p-5, -0x1.a9c7ap-4, \
  0x1.a3e4ccp+0, -0x1.cde2e8p-6, 0x1.b7228ap-2, 0x1.b6b5a8p-2, 0x1.38c52ep-1, 0x1.9e5c6ap+0, \
  0x1.898056p+0, 0x0p+0, -0x1.4d899ap-7, 0x1.b27042p-10, 0x1.a1acc8p-12, -0x1.30da1cp-10, \
  -0x1.b97598p-14, 0x1.6fdc9p-14, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, \
  0x1.029b0ep+0, 0x1.c97f2ep-1, 0x1.aa82eep+0, 0x0p+0, 0x1.3e4472p-2, 0x0p+0, \
  0x0p+0, 0x1.069894p-3, -0x1.461158p-4, 0x1.633bccp+0, 0x1.8b1f3p+0, 0x1.b54ea4p+0, \
  0x1.c100ccp+0, 0x1.c6c762p+0, 0x1.cc07ep+0, 0x1.d189ecp+0, 0x1.d72654p+0, 0x1.daa0dep+0, \
  0x1.d4be7p+0, 0x1.c8d7ap+0, 0x1.bfca42p+0, 0x1.b7d0c2p+0, 0x1.ae58d6p+0, 0x1.a04358p+0, \
  0x1.83435ep+0, 0x1.b17bfap-3, 0x1.e2a172p-7, 0x1.9a077ap-3, -0x1.903eaep-8, 0x0p+0, \
  0x1.c2649cp-3, -0x1.9c4c38p-6, 0x1.c1c54ep-3, -0x1.deb3c4p-6, 0x0p+0, 0x1.62f6c4p-3, \
  -0x1.345c88p-5, 0x1.5c8d32p-3, -0x1.ae551ep-5, 0x0p+0, 0x1.49566cp-3, -0x1.4cb658p-5, \
  0x1.49d30ep-3, -0x1.74bb26p-5, 0x0p+0, 0x1.588e8p-3, -0x1.d7b734p-5, 0x1.59a2a6p-3, \
  -0x1.0879f6p-4, 0x0p+0, 0x1.84fb7cp-3, -0x1.ee9ffep-5, 0x1.62842cp-3, -0x1.e0708p-5, \
  0x0p+0, 0x1.9b808cp-3, -0x1.d549dcp-5, 0x1.763616p-3, -0x1.8dfe5cp-5, 0x0p+0, \
  0x1.be1842p-4, 0x1.c3e018p-9, 0x1.82da1ap-4, -0x1.ae1ffap-6, 0x0p+0, 0x1.15968ep+1, \
  0x1.c359d4p+0, 0x1.9d0706p+0, 0x1.8a38fcp+0, 0x1.8992d2p+0, 0x1.99ad36p+0, 0x1.ba6cfcp+0, \
  0x1.e8ca18p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, \
  0x0p+0, 0x1.1c9e88p-3, -0x1.dbefbp-6, 0x1.694b48p+0, 0x0p+0, 0x1.2a58ep-1, \
  -0x1.5b8334p-3, 0x1.75b15p+0, 0x0p+0}
#define TOKN_SD {0x1.5b0b7p+0, 0x1.565f6cp+0, 0x1.15b66ap-2, 0x1.5bcb0ep+0, 0x1.565fb6p+0, 0x1.259b08p-2, \
  0x1.f6347cp-2, 0x1.049fc6p-1, 0x1.130114p-2, 0x1.137e14p-2, 0x1.5d66d8p-2, 0x1.07d10cp-1, \
  0x1.1f7ffep-1, 0x1p+0, 0x1.14a304p-4, 0x1.f6f8fap-5, 0x1.9ed054p-6, 0x1.1b9df4p-3, \
  0x1.01549ap-3, 0x1.5c155cp-6, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, \
  0x1.7bfc84p-1, 0x1.cacb72p-5, 0x1.89a83cp-1, 0x1p+0, 0x1.4df916p-2, 0x1p+0, \
  0x1p+0, 0x1.5852eep-5, 0x1.836e1cp-5, 0x1.2453cap-1, 0x1.18be04p-1, 0x1.15bc8p-1, \
  0x1.09d9ecp-1, 0x1.0346e8p-1, 0x1.fb699ap-2, 0x1.ed77aep-2, 0x1.dbf494p-2, 0x1.cf92bp-2, \
  0x1.e75d3cp-2, 0x1.08383ep-1, 0x1.13307ep-1, 0x1.19979ep-1, 0x1.209958p-1, 0x1.26e7dp-1, \
  0x1.232c4ap-1, 0x1.5e278p+0, 0x1.55197p+0, 0x1.5d962ep+0, 0x1.552a16p+0, 0x1p+0, \
  0x1.78a772p+0, 0x1.705b5cp+0, 0x1.77b976p+0, 0x1.6ed3eep+0, 0x1p+0, 0x1.99339ap+0, \
  0x1.914292p+0, 0x1.9a2724p+0, 0x1.9099a6p+0, 0x1p+0, 0x1.b38bf6p+0, 0x1.ada24ep+0, \
  0x1.b4429cp+0, 0x1.ada54ep+0, 0x1p+0, 0x1.c4ed9ap+0, 0x1.c33102p+0, 0x1.c52accp+0, \
  0x1.c394e6p+0, 0x1p+0, 0x1.ce9b98p+0, 0x1.d0da7ap+0, 0x1.cff688p+0, 0x1.d01f6p+0, \
  0x1p+0, 0x1.d5801cp+0, 0x1.d767f6p+0, 0x1.d5f88p+0, 0x1.d91b66p+0, 0x1p+0, \
  0x1.da1458p+0, 0x1.db5db4p+0, 0x1.d9418cp+0, 0x1.dd7d48p+0, 0x1p+0, 0x1.65a3cep-4, \
  0x1.674d48p-1, 0x1.8e5866p-1, 0x1.92a186p-1, 0x1.85a09p-1, 0x1.82b924p-1, 0x1.6a516p-1, \
  0x1.2b64b8p-1, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, \
  0x1p+0, 0x1.25382ep+0, 0x1.29174cp+0, 0x1.b2a378p-2, 0x1p+0, 0x1.58edf6p-1, \
  0x1.5a237cp-1, 0x1.194edcp-1, 0x1p+0}
// 표본 수: 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 2566765 0 2566765 2566765 2566765 2566765 2566765 2566765 0 0 0 0 2566765 2566765 2566765 0 2566765 0 0 2566765 2566765 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 409600 401828 401828 401828 401828 0 367549 367549 367549 367549 0 323381 323381 323381 323381 0 288376 288376 288376 288376 0 254299 254299 254299 254299 0 206459 206459 206459 206459 0 136791 136791 136791 136791 0 65328 65328 65328 65328 0 409600 409600 409600 409600 409600 409600 409600 409600 0 0 0 0 0 0 267287 267287 267287 0 292588 292588 292588 0
constexpr float kMu[N_FEAT] = TOKN_MU;
constexpr float kSd[N_FEAT] = TOKN_SD;
#ifdef __CUDACC__
static __constant__ float kMuDev[N_FEAT] = TOKN_MU;
static __constant__ float kSdDev[N_FEAT] = TOKN_SD;
#define TOKN_FN __host__ __device__ __forceinline__
#else
#define TOKN_FN inline
#endif
TOKN_FN float mu(int f) {
#ifdef __CUDA_ARCH__
  return kMuDev[f];
#else
  return kMu[f];
#endif
}
TOKN_FN float sd(int f) {
#ifdef __CUDA_ARCH__
  return kSdDev[f];
#else
  return kSd[f];
#endif
}
}  // namespace tokn
