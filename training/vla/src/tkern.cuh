// 학습용 커널(앞·뒤) 실행 함수(tkern.cu). 모두 비동기·결정적(부동소수 원자 없음, 고정 순서 합).
// 변수 기울기 중 "행마다 더하는" 것(정규화 w, 합성곱, A_log 등)은 행마다 기여를 임시 버퍼에 쓴 뒤 colsum 으로 고정 순서 합을 낸다.
#pragma once
#include <cstdint>

#include <cuda_runtime_api.h>

namespace rvla {
namespace tk {

// ---- 일반 ----
// out[c] (=, += acc) Σ_r T[r·ld + c], r < R, c < C. 작업 공간 part ≥ ceil(R/256)·C
void colsum(const float* T, int R, int C, int ld, float* part, float* out, bool acc, cudaStream_t st);
void f2bf(const float* x, long long n, uint16_t* y, cudaStream_t st);
void add(float* y, const float* x, long long n, cudaStream_t st);   // y += x
// 줄 간격 있는 행렬 복사·더하기: dst[r·ldd + c] (=|+=) src[r·lds + c], c < C
void copy2d(const float* src, int lds, float* dst, int ldd, int R, int C, bool acc, cudaStream_t st);

// ---- GEMM(bf16 입력, FP32 누산) ----
// 앞: out[M][ldo] (=|+=) A[M][lda]·W[N][K]ᵀ
void mm(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, cudaStream_t st);
// dX: out[M][ldo] (=|+=) dZ[M][ldz]·W[N][K]  (N 이 합 방향, 출력 K 열)
void mm_dx(const uint16_t* dZ, int ldz, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, cudaStream_t st);
// dW: G[N][K] (bf16, =, 또는 FP32 gacc 에 +=) Σ_m dZ[m][n]·X[m][k], split-K 조각(ws ≥ ceil(M/chunk)·N·K) → 고정 순서 합
void mm_dw(const uint16_t* dZ, int ldz, const uint16_t* X, int ldx, int M, int N, int K, float* ws, int chunk, uint16_t* G, float* gacc, cudaStream_t st);

// ---- 끝단 붙인 GEMM(vgemm.cu): 본 계산은 mm·mm_dx 와 비트까지 같고 끝단만 다르다 ----
enum EpiKind { EK_RES = 0, EK_BIAS = 1, EK_SWI = 2, EK_DSWI = 3, EK_DW = 4 };
struct Epi {
  int kind = EK_RES;
  float* C = nullptr; long long ldc = 0;          // F32 출력
  const float* R = nullptr; long long ldr = 0;    // EK_RES: 잔차 입력(C 와 같아도 됨)
  const float* bias = nullptr;                    // [N]
  uint16_t* Cb = nullptr; long long ldcb = 0;     // bf16 출력
  int gelu = 0;                                   // EK_BIAS: Cb = bf16(gelu(acc + b))
  const float* GU = nullptr; long long ldgu = 0;  // EK_DSWI: 끼운 꼴 GU [R][2I]
  int I = 0, bug = 0;
  float* gacc = nullptr;                          // EK_DW: FP32 += (nullptr 이면 Cb bf16 =)
};
// 앞: A·Wᵀ → 끝단. EK_RES: C = R + acc (+ b). EK_BIAS: C = acc + b, Cb = bf16(gelu?(·)). EK_SWI: W = [gate I | up I] 를 짝 끼운 순서로 읽어
// C(선택) = 끼운 꼴 GU [M][2I] F32, Cb = Hh [M][I] bf16 = silu(g)·u
void mme(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, const Epi& e, cudaStream_t st);
// dX: dZ·W → 끝단. EK_DSWI: acc = dHh [M][I], GU 끼운 꼴 → Cb = dGU bf16 [M][2I] 원래 꼴(swiglu_bwd 와 같은 식). EK_RES: C = R + acc
void mme_dx(const uint16_t* dZ, int ldz, int M, const uint16_t* W, int N, int K, const Epi& e, cudaStream_t st);
// dW 조각 하나(split 1): mm_dw 의 GEMM + dwred 를 끝단 하나로
void mme_dw1(const uint16_t* dZ, int ldz, const uint16_t* X, int ldx, int M, int N, int K, uint16_t* G, float* gacc, cudaStream_t st);

// ---- RMSNorm ----
// y = x·r·(w1 ? (1 + w) : w), r = (mean x² + ε)^−½. 벡터 NV 개(길이 D), 벡터 v 의 자리 = base + (v / per)·ld + (v % per)·D
// 뒤: dX (=|+=), wt[NV][D] = dy·x·r (w 기울기 기여, nullptr 이면 안 씀). bug 1: g = dy 로(w 곱 빠뜨림)
// gw(nullptr 아니면) = Σ_v dy·x·r — 커널 안 조각(npart ≥ npart_floats(NV, D)) + colred(D). dXb(선택) = bf16(dX 최종값)
void rms_bwd(const float* dY, int lddy, const float* X, int ldx, int NV, int per, int D, const float* w, bool w1, float eps, float* dX, int lddx,
             bool acc, float* npart, float* gw, int bug, cudaStream_t st, uint16_t* dXb = nullptr);
long long npart_floats(long long rows, int D);   // 조각 작업 공간(벡터 32 개당 D)

// ---- LayerNorm(SigLIP) ----
void ln_fwd(const float* X, int R, int D, const float* g, const float* b, float eps, uint16_t* out, float* outf, cudaStream_t st);
// dX (+=), gt[R][D] = dy·x̂ (γ 기여), β 기여는 dy 그대로(colsum)
// dX += , gg = Σ dy·x̂, gb = Σ dy(커널 안 조각, npart ≥ 2·npart_floats(R, D)), dXb(선택) = bf16(dX)
void ln_bwd(const float* dY, const float* X, int R, int D, const float* g, float eps, float* dX, float* npart, float* gg, float* gb, cudaStream_t st,
            uint16_t* dXb = nullptr);
void bias_add(float* Y, int R, int N, int ld, const float* b, cudaStream_t st);
// GELU tanh: out bf16 = gelu(x) (x F32 [R][N]); 뒤: dx = dy·gelu'(x) → bf16
void gelu_fwd(const float* X, long long n, uint16_t* out, cudaStream_t st);
void gelu_bwd(const float* dY, const float* X, long long n, uint16_t* dX, float* dXf, cudaStream_t st);   // dXf(FP32 사본) 는 nullptr 가능, dY 와 같아도 됨

// ---- 어텐션(일반): 질의 = 행 (b, t), 머리 nq; 키·값 = 구간 1 [B][L1][ldk1] (판마다 유효 len1, causal 이면 j ≤ t + qoff) + 구간 2 [B][n2][ldk2] ----
struct AttP {
  const float* Q = nullptr; int ldq = 0;
  const float *K1 = nullptr, *V1 = nullptr; int ldk1 = 0, L1 = 0;
  const int* len1 = nullptr; int n1c = 0;
  const float *K2 = nullptr, *V2 = nullptr; int ldk2 = 0, n2 = 0;
  int causal = 0, qoff = 0;
  int B = 0, n = 0, nq = 0, nkv = 0, hd = 0;
  float scale = 1.f;
  float* O = nullptr; int ldo = 0;
  float* lse = nullptr;            // [B·n][nq]
  // 뒤
  const float* dO = nullptr; int lddo = 0;
  float* Dd = nullptr;             // [B·n][nq]
  float* dQ = nullptr; int lddq = 0;
  float *dK1 = nullptr, *dV1 = nullptr;   // K1 과 같은 배치(쓰기, nullptr 이면 안 함 — 지식 격리)
  float *dK2 = nullptr, *dV2 = nullptr;
  int bug = 0;                     // 1: 뒤에서 scale 빠뜨림
};
void att_fwd(const AttP& p, cudaStream_t st);
void att_bwd(const AttP& p, cudaStream_t st);
// 텐서 코어 판(flash.cu). 모양이 안 맞으면(키 > 512, hd ∉ {32, 64, 256}) false — att_fwd/att_bwd 가 예전 FP32 커널로.
// 환경 변수 RVLA_ATT_OLD=1 이면 늘 예전 커널(비교용).
bool fa_fwd(const AttP& p, cudaStream_t st);
bool fa_bwd(const AttP& p, cudaStream_t st);
bool att_old();

// ---- Qwen 풀 어텐션 준비의 뒤: dQ(정규화·RoPE 뒤 q 기울기 [R][nq·hd]), dK·dV(캐시 배치 [B][Lc][kvw]) → dT0([q|gate],k,v 칸), qn·kn 기여 ----
// 어텐션 출력 게이트의 뒤: dG(게이트 곱 뒤 기울기) → dO(어텐션 출력 기울기), dT0 의 gate 칸
void gate_bwd(const float* dG, const float* O, const float* T0, int ldT, int R, int nq, int hd, float* dOut, float* dT0, cudaStream_t st, uint16_t* dT0b = nullptr);
void full_prep_bwd(const float* T0, int ldT, int R, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
                   const float* kn, const float* dQ, const float* dKc, const float* dVc, int Lc, float* dT0, float* qnt, float* knt, int bug,
                   cudaStream_t st, const int* poff = nullptr, uint16_t* dT0b = nullptr);
// ---- DeltaNet ----
void lin_prep_bwd(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb,
                  const float* dQn, const float* dKn, const float* dG, const float* dBeta, float* dT1, float* dT0, float* alt, float* dtt,
                  cudaStream_t st, uint16_t* dT0b = nullptr);
// 합성곱 + SiLU 의 뒤(학습: 열 시작 0, 캐시 없음): dT1 [R][C] → dX(=, dT0 의 qkv 칸, 줄 간격 ld), dpre 를 dp [R][C] 에 남김
void conv_bwd(const float* X, int ld, int B, int n, int C, int K, const float* w, const float* dT1, float* dp, float* dX, cudaStream_t st, uint16_t* dXb = nullptr);
// 합성곱 가중치 기울기 [C][K] = Σ_(b,t) dp[t][c]·x[t−K+1+k][c]
void convw_grad(const float* X, int ld, const float* dp, int B, int n, int C, int K, float* part, float* out, cudaStream_t st);
// 덩이 꼴(WY/UT 변환, 텐서 코어) 덩이 크기: dk 64 이상 = 64, 작은 구성 = 16(덩이 여럿을 시험하게)
inline int dn_chunk(int dk) { return dk >= 64 ? 64 : 16; }
// 덩이 꼴 앞(dnchunk.cu): Qn·Kn [R][lh·dk], V [R][ldv], G·Beta [R][lh] → O [R][lh·dv]; S0·S1 [B][lh][dk][dv](nullptr 가능).
// ws: dnc_ws_floats(B, n, …) 개. keep: 뒤가 쓸 검문점(덩이 시작 상태)·Δ 도 남김(같은 ws 로 dnc_bwd 를 부르기 전에 같은 입력으로 앞을 돌려 둘 것)
long long dnc_ws_floats(int B, int n, int lh, int dk, int dv);
void dnc_fwd(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, int B, int n, int lh, int dk, int dv, const float* S0,
             float* S1, bool nodecay, float* O, float* ws, bool keep, cudaStream_t st);
// 덩이 꼴 뒤(열 시작 상태 0, 끝 상태 기울기 0). bug 3: dS 감쇠 빠뜨림(음성 대조)
void dnc_bwd(const float* V, int ldv, const float* dO, int B, int n, int lh, int dk, int dv, bool nodecay, float* ws, float* dQn, float* dKn, float* dV, int lddv,
             float* dG, float* dBeta, int bug, cudaStream_t st);
// (예전) 재귀의 뒤(검문점 + 구간 다시 계산). ws: dn_ws_floats(...) 개
long long dn_ws_floats(int B, int n, int lh, int dk, int dv);
// hasck: 앞 계산(qk::deltanet 에 ws 를 검문점 자리로 준 것)이 검문점을 이미 써 둠
void deltanet_bwd(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, const float* dO, int B, int n, int lh, int dk,
                  int dv, bool nodecay, bool hasck, float* ws, float* dQn, float* dKn, float* dV, int lddv, float* dG, float* dBeta, int bug, cudaStream_t st);
void gnorm_bwd(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, const float* dY, float* dO, float* dZ, int lddz,
               float* npart, float* gw, cudaStream_t st, uint16_t* dZb = nullptr);
// SwiGLU 뒤: dH [R][I] F32 → dGU bf16 [R][2I]. bug 1: silu' 빠뜨림
void swiglu_bwd(const float* T0, const float* dH, int R, int I, uint16_t* dGU, int bug, cudaStream_t st);

}  // namespace tk
}  // namespace rvla
