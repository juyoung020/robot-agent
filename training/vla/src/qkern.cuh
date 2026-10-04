// Qwen3.5 앞 계산 커널 실행 함수(qkern.cu). 모두 비동기·결정적(부동소수 원자 없음, 고정 합산 순서).
#pragma once
#include <cstdint>

#include <cuda_runtime_api.h>

namespace rvla {
struct MT;
namespace qk {

void embed(const int* ids, const uint16_t* E, int R, int H, float* X, cudaStream_t st);
// out_b(bf16, 줄 간격 H) 또는 out_f(FP32)에 RMSN(X)·(1 + w). norm = false 면 정규화 없이 X 를 그대로(음성 대조)
void rmsnorm(const float* X, int R, int H, const float* w, bool norm, float eps, uint16_t* out_b, float* out_f, float* rs, cudaStream_t st);
// out[M][ldo] (=, 또는 +=) A[M][lda]·W[N][K]ᵀ, FP32 출력
void gemm_f32(const uint16_t* A, int lda, int M, const uint16_t* Wb, const MT& w, float* out, int ldo, bool acc, cudaStream_t st);
void full_prep(const float* T0, int ldT, int R, int B, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
               const float* kn, bool skipnorm, float* Q, float* Kc, float* Vc, int Lc, cudaStream_t st);
void attn(const float* Q, const float* Kc, const float* Vc, int B, int n, int pos0, int Lc, int nq, int nkv, int hd, float* O, float* lse, cudaStream_t st);
void gate(const float* O, const float* T0, int ldT, int R, int nq, int hd, uint16_t* A, cudaStream_t st);
void hist_put(const float* T0, int ld, int B, int n, int pos0, int C, float* hist, int Lmax, cudaStream_t st);
void conv_silu(const float* src, int ld, int Ls, int B, int n, int pos0, int C, int K, const float* w, float* out, cudaStream_t st);
void lin_prep(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb, bool skipnorm,
              float* Qn, float* Kn, float* G, float* Beta, cudaStream_t st);
void deltanet(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, int B, int n, int lh, int dk, int dv,
              const float* S0, float* S1, bool nodecay, float* O, cudaStream_t st);
void gnorm(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, uint16_t* A, cudaStream_t st);
void swiglu(const float* T0, int R, int I, uint16_t* A, cudaStream_t st);
void f2bf_rows(const float* X, int R, int H, uint16_t* A, cudaStream_t st);
void gather_last(const float* hid, int B, int n, int H, float* out, cudaStream_t st);
void argmax_rows(const float* L, int R, int V, int* out, cudaStream_t st);

}  // namespace qk
}  // namespace rvla
