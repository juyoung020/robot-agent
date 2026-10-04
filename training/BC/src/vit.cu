// 얼린 SigLIP 2 B/32-256 영상 탑(패치 토큰) — 설명은 include/vit.h.
// 속도판(Encoder::kern 1, 기본): GEMM = vit_gemm.cuh(128 × 256 × k64, 2 단 cp.async, XOR 섞기, 같은 누산 차례 — FP16 누산 결과 비트가 예전과 같음)
// + 끝단(편향 FP32, GELU tanh 를 x·σ(2u) 꼴로, 위치 임베딩, 잔차 FP32 더하기), 어텐션 = 텐서 코어(attn_tc_k), LayerNorm FP32.
// 예전 판(kern 0): training/RL/network/src/gemm.cuh 조각으로 짠 3 단 커널(vgemm_k)·f8::tn_k<TN_F16H>·스칼라 FP32 어텐션(attn_k) — 비교용으로 남김.
// FP8 GEMM(f8 표)은 두 판 모두 f8::tn_k. 부동소수 원자 연산 없음(결정적).
#include <dirent.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_fp16.h>

#include "gemm.cuh"
#include "gemm_fp8.cuh"
#include "vit.h"
#include "vit_gemm.cuh"

namespace vit {

using net::bf2f;
using net::f2bf;
// float → FP16 비트(가장 가까운 짝수) — 호스트 가중치 변환용
static uint16_t net_f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

#define VTK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// 가중치 읽기: safetensors = [u64 머리 길이][JSON 머리][자료]. 머리에서 이름 → dtype·shape·data_offsets 만 찾는다(작은 손 파서).
static bool find_tensor(const std::string& hdr, const std::string& name, size_t& a, size_t& b, std::string& dtype) {
  const std::string key = "\"" + name + "\"";
  const size_t k = hdr.find(key);
  if (k == std::string::npos) return false;
  const size_t e = hdr.find('}', k);
  const std::string obj = hdr.substr(k, e - k);
  const size_t dt = obj.find("\"dtype\"");
  const size_t q1 = obj.find('"', obj.find(':', dt) + 1), q2 = obj.find('"', q1 + 1);
  dtype = obj.substr(q1 + 1, q2 - q1 - 1);
  const size_t off = obj.find("\"data_offsets\"");
  const size_t lb = obj.find('[', off);
  a = std::strtoull(obj.c_str() + lb + 1, nullptr, 10);
  b = std::strtoull(obj.c_str() + obj.find(',', lb) + 1, nullptr, 10);
  return true;
}
static std::string default_path() {
  const char* home = std::getenv("HOME");
  const std::string base = std::string(home ? home : "") + "/.cache/huggingface/hub/models--timm--ViT-B-32-SigLIP2-256/snapshots";
  DIR* d = opendir(base.c_str());
  if (!d) return "";
  std::string out;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    const std::string p = base + "/" + e->d_name + "/open_clip_model.safetensors";
    if (FILE* f = std::fopen(p.c_str(), "rb")) { std::fclose(f); out = p; break; }
  }
  closedir(d);
  return out;
}
bool load_weights(const std::string& path_in, HostWeights& w, std::string* used) {
  const std::string path = path_in.empty() ? default_path() : path_in;
  if (used) *used = path;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "vit: cannot open weights '%s'\n", path.c_str()); return false; }
  uint64_t hl = 0;
  if (std::fread(&hl, 8, 1, f) != 1) { std::fclose(f); return false; }
  std::string hdr(hl, '\0');
  if (std::fread(&hdr[0], 1, hl, f) != hl) { std::fclose(f); return false; }
  const long long base = 8 + (long long)hl;
  bool ok = true;
  auto rd = [&](const std::string& name, std::vector<float>& v, size_t n) {
    size_t a, b;
    std::string dt;
    if (!find_tensor(hdr, "visual.trunk." + name, a, b, dt) || dt != "F32" || b - a != n * 4) {
      std::fprintf(stderr, "vit: tensor visual.trunk.%s missing or bad (dtype %s, bytes %zu, want %zu)\n", name.c_str(), dt.c_str(), b - a, n * 4);
      ok = false;
      return;
    }
    v.resize(n);
    std::fseek(f, base + (long long)a, SEEK_SET);
    if (std::fread(v.data(), 4, n, f) != n) ok = false;
  };
  rd("patch_embed.proj.weight", w.patch_w, (size_t)D * KP);
  rd("patch_embed.proj.bias", w.patch_b, D);
  rd("pos_embed", w.pos, (size_t)NTOK * D);
  for (int l = 0; l < LAYERS; ++l) {
    const std::string p = "blocks." + std::to_string(l) + ".";
    auto& B = w.blk[l];
    rd(p + "norm1.weight", B.ln1_g, D); rd(p + "norm1.bias", B.ln1_b, D);
    rd(p + "attn.qkv.weight", B.qkv_w, (size_t)3 * D * D); rd(p + "attn.qkv.bias", B.qkv_b, 3 * D);
    rd(p + "attn.proj.weight", B.proj_w, (size_t)D * D); rd(p + "attn.proj.bias", B.proj_b, D);
    rd(p + "norm2.weight", B.ln2_g, D); rd(p + "norm2.bias", B.ln2_b, D);
    rd(p + "mlp.fc1.weight", B.fc1_w, (size_t)MLP * D); rd(p + "mlp.fc1.bias", B.fc1_b, MLP);
    rd(p + "mlp.fc2.weight", B.fc2_w, (size_t)D * MLP); rd(p + "mlp.fc2.bias", B.fc2_b, D);
  }
  rd("norm.weight", w.norm_g, D);
  rd("norm.bias", w.norm_b, D);
  std::fclose(f);
  return ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM: C[M][N] = A[M][K] · W[N][K]ᵀ (둘 다 K 연속 bf16) + 끝단. K 는 32 의 배수, N 은 BN 의 배수, M 은 64 의 배수(영상 단위)라 경계 검사 없음.
enum VEpi { VE_BF16 = 0, VE_GELU = 1, VE_F32_POS = 2, VE_F32_ACC = 3 };
struct VG {
  const uint16_t* A; const uint16_t* B; void* C; const float* bias; const float* pos;
  int M, N, K, lda, ldb, ldc, bug;
};
constexpr int VST = 3;   // cp.async 단 수

// 저장 형식: H = true 면 FP16(기본), false 면 BF16. 피연산자 둘 다 같은 형식, 누산 FP32.
template <bool H>
__device__ __forceinline__ uint16_t st16(float f) { return H ? __half_as_ushort(__float2half_rn(f)) : f2bf(f); }
template <bool H>
__device__ __forceinline__ float ld16(uint16_t v) { return H ? __half2float(__ushort_as_half(v)) : bf2f(v); }
template <bool H>
__device__ __forceinline__ void mma16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  if (H)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    net::mma_bf16(d, a, b);
}

__device__ __forceinline__ float gelu_tanh(float x) {   // 예전 식(FP8 끝단과 vgemm_k 가 씀)
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + tanhf(u));
}
// 같은 함수의 빠른 꼴: 0.5·x·(1 + tanh u) = x · σ(2u) = x / (1 + e^{−2u}). ex2.approx(상대 2^-22 수준) + 나눗셈 근사 —
// 결과를 FP16 으로 저장하므로(상대 2^-11) 차이가 묻힌다. 큰 음수 x 에서도 상대 정밀도가 유지된다(1 + tanh 빼기 없음).
__device__ __forceinline__ float gelu_fast(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return __fdividef(x, 1.f + __expf(-2.f * u));
}

template <bool HF, int EPI, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) vgemm_k(const __grid_constant__ VG p) {
  constexpr int NWM = BM / WM, NT = NWM * (BN / WN) * 32, MI = WM / 16, NI = WN / 8;
  using CA = net::G2Tile<BM, false>;
  using CB = net::G2Tile<BN, false>;
  extern __shared__ __align__(128) uint16_t vs[];
  uint16_t (*sA)[CA::ELEMS] = reinterpret_cast<uint16_t (*)[CA::ELEMS]>(vs);
  uint16_t (*sB)[CB::ELEMS] = reinterpret_cast<uint16_t (*)[CB::ELEMS]>(vs + VST * CA::ELEMS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int wm = (warp % NWM) * WM, wn = (warp / NWM) * WN;
  float acc[MI][NI][4];
#pragma unroll
  for (int a = 0; a < MI; ++a)
#pragma unroll
    for (int b = 0; b < NI; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = p.K / net::G2K;
#pragma unroll
  for (int s = 0; s < VST - 1; ++s) {
    if (s < nk) {
      net::g2_load<BM, false, NT>(sA[s], p.A, p.lda, m0, p.M, s * net::G2K, p.K, tid);
      net::g2_load<BN, false, NT>(sB[s], p.B, p.ldb, n0, p.N, s * net::G2K, p.K, tid);
    }
    net::cp_commit();
  }
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  for (int kt = 0; kt < nk; ++kt) {
    const int nx = kt + VST - 1;
    if (nx < nk) {
      net::g2_load<BM, false, NT>(sA[nx % VST], p.A, p.lda, m0, p.M, nx * net::G2K, p.K, tid);
      net::g2_load<BN, false, NT>(sB[nx % VST], p.B, p.ldb, n0, p.N, nx * net::G2K, p.K, tid);
    }
    net::cp_commit();
    net::cp_wait<VST - 1>();
    __syncthreads();
    const uint16_t* A = sA[kt % VST];
    const uint16_t* B = sB[kt % VST];
#pragma unroll
    for (int kk = 0; kk < net::G2K; kk += 16) {
      uint32_t af[MI][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) net::ldsm4(af[mi], A + (wm + mi * 16 + lr + j0 * 8) * CA::LD + kk + j1 * 8);
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        uint32_t r[4];
        net::ldsm4(r, B + (wn + nj * 16 + lr + j1 * 8) * CB::LD + kk + j0 * 8);
        bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma16<HF>(acc[mi][ni], af[mi], bfr[ni]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r >= p.M || c >= p.N) continue;
        float x0 = acc[mi][ni][2 * h] + p.bias[c], x1 = acc[mi][ni][2 * h + 1] + p.bias[c + 1];
        if (EPI == VE_BF16 || EPI == VE_GELU) {
          if (EPI == VE_GELU) { x0 = gelu_tanh(x0); x1 = gelu_tanh(x1); }
          reinterpret_cast<uint32_t*>(p.C)[((long long)r * p.ldc + c) >> 1] = (uint32_t)st16<HF>(x0) | ((uint32_t)st16<HF>(x1) << 16);
        } else if (EPI == VE_F32_POS) {
          if (p.bug != 3) {
            const float* ps = p.pos + (size_t)(r % NTOK) * D + c;
            x0 = x0 + ps[0];
            x1 = x1 + ps[1];
          }
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c) = make_float2(x0, x1);
        } else {
          float2* d = reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c);
          float2 o = *d;
          o.x = o.x + x0;
          o.y = o.y + x1;
          *d = o;
        }
      }
}

template <bool HF, int EPI, int BM, int BN, int WM, int WN>
static void vgemm_attr() {
  constexpr int smem = VST * 2 * (net::G2Tile<BM, false>::ELEMS + net::G2Tile<BN, false>::ELEMS);
  VTK(cudaFuncSetAttribute(vgemm_k<HF, EPI, BM, BN, WM, WN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
}
template <bool HF, int EPI>
static void vgemm(const VG& p, cudaStream_t st) {
  constexpr int BM = 128, BN = 128, WM = 64, WN = 32;
  constexpr int smem = VST * 2 * (net::G2Tile<BM, false>::ELEMS + net::G2Tile<BN, false>::ELEMS);
  if (p.K % net::G2K || p.N % BN || p.M % 64) { std::fprintf(stderr, "vgemm: bad shape M %d N %d K %d\n", p.M, p.N, p.K); std::abort(); }
  vgemm_k<HF, EPI, BM, BN, WM, WN><<<dim3(p.N / BN, (p.M + BM - 1) / BM), (BM / WM) * (BN / WN) * 32, smem, st>>>(p);
  VTK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// LayerNorm(FP32 통계): 워프 하나 = 행 하나(768 = 레인당 24). 합은 xor 나비(모든 레인 같은 값, 결정적). out bf16 [rows][ldo]
template <bool HF>
__global__ void __launch_bounds__(256) ln_k(const float* X, int rows, const float* gm, const float* bt, uint16_t* out, int ldo, int bug) {
  const int r = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (r >= rows) return;
  const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)r * D);
  float v[24];
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const float4 q = x4[l + 32 * k];
    v[4 * k] = q.x; v[4 * k + 1] = q.y; v[4 * k + 2] = q.z; v[4 * k + 3] = q.w;
  }
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 24; ++k) s = s + v[k];
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, m);
  const float mean = s / (float)D;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < 24; ++k) { const float d = v[k] - mean; q = q + d * d; }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) q = q + __shfl_xor_sync(0xffffffffu, q, m);
  const float rstd = 1.f / sqrtf(q / (float)D + LN_EPS);
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const int c = 4 * (l + 32 * k);
    uint16_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float xc = bug == 2 ? v[4 * k + e] : v[4 * k + e] - mean;
      o[e] = st16<HF>(xc * rstd * gm[c + e] + bt[c + e]);
    }
    uint2 w;
    w.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
    w.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
    *reinterpret_cast<uint2*>(out + (size_t)r * ldo + c) = w;
  }
}

// 어텐션: 블록 하나 = (영상, 머리), 스레드 하나 = 질의 토큰 하나. S = QKᵀ/8, softmax, O = PV 모두 FP32(곱셈-덧셈은 __fmaf_rn, 차례 고정).
template <bool HF>
__global__ void __launch_bounds__(NTOK) attn_k(const uint16_t* qkv, uint16_t* out, int bug) {
  __shared__ float Ks[NTOK][HD], Vs[NTOK][HD];
  const int img = blockIdx.x / HEADS, hh = blockIdx.x % HEADS, i = threadIdx.x;
  const size_t base = (size_t)img * NTOK;
  for (int idx = i; idx < NTOK * HD / 8; idx += NTOK) {
    const int t = idx / (HD / 8), d0 = (idx % (HD / 8)) * 8;
    const uint4 kq = *reinterpret_cast<const uint4*>(qkv + (base + t) * 3 * D + D + hh * HD + d0);
    const uint4 vq = *reinterpret_cast<const uint4*>(qkv + (base + t) * 3 * D + 2 * D + hh * HD + d0);
    const uint32_t kw[4] = {kq.x, kq.y, kq.z, kq.w}, vw[4] = {vq.x, vq.y, vq.z, vq.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      Ks[t][d0 + 2 * e] = ld16<HF>((uint16_t)(kw[e] & 0xffffu)); Ks[t][d0 + 2 * e + 1] = ld16<HF>((uint16_t)(kw[e] >> 16));
      Vs[t][d0 + 2 * e] = ld16<HF>((uint16_t)(vw[e] & 0xffffu)); Vs[t][d0 + 2 * e + 1] = ld16<HF>((uint16_t)(vw[e] >> 16));
    }
  }
  float q[HD];
  {
    const uint4* qp = reinterpret_cast<const uint4*>(qkv + (base + i) * 3 * D + hh * HD);
#pragma unroll
    for (int c = 0; c < HD / 8; ++c) {
      const uint4 w = qp[c];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { q[8 * c + 2 * e] = ld16<HF>((uint16_t)(ww[e] & 0xffffu)); q[8 * c + 2 * e + 1] = ld16<HF>((uint16_t)(ww[e] >> 16)); }
    }
  }
  __syncthreads();
  const float scale = bug == 1 ? 1.f : 0.125f;
  float s[NTOK];
  float mx = -3.0e38f;
#pragma unroll
  for (int j = 0; j < NTOK; ++j) {
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < HD; ++d) a = __fmaf_rn(q[d], Ks[j][d], a);
    s[j] = a * scale;
    mx = fmaxf(mx, s[j]);
  }
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < NTOK; ++j) { s[j] = expf(s[j] - mx); sum = sum + s[j]; }
  const float inv = 1.f / sum;
#pragma unroll
  for (int d0 = 0; d0 < HD; d0 += 8) {
    float o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = 0.f;
#pragma unroll
    for (int j = 0; j < NTOK; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = __fmaf_rn(s[j], Vs[j][d0 + e], o[e]);
    uint4 w;
    w.x = (uint32_t)st16<HF>(o[0] * inv) | ((uint32_t)st16<HF>(o[1] * inv) << 16);
    w.y = (uint32_t)st16<HF>(o[2] * inv) | ((uint32_t)st16<HF>(o[3] * inv) << 16);
    w.z = (uint32_t)st16<HF>(o[4] * inv) | ((uint32_t)st16<HF>(o[5] * inv) << 16);
    w.w = (uint32_t)st16<HF>(o[6] * inv) | ((uint32_t)st16<HF>(o[7] * inv) << 16);
    *reinterpret_cast<uint4*>(out + (base + i) * D + hh * HD + d0) = w;
  }
}

// 텐서 코어 어텐션(속도판): 블록 하나 = (영상, 머리), 워프 4 개 × 질의 16 행. Q·K·V [64 토큰][64] 를 공유 메모리로(cp.async, 128 B 줄 XOR 섞기).
//   S = QKᵀ: mma m16n8k16 (16 비트 입력, FP32 누산) → × 1/8 → 행 최대(쿼드 나비) → p = exp(s − max) FP32 → p 를 16 비트로(같은 형식 HF)
//   → 행 합 = 16 비트로 반올림한 p 의 FP32 합(나비, 차례 고정) → O = P·V (mma, V 는 ldmatrix.trans, FP32 누산) → O / 합 → 16 비트.
// 토큰 64 개가 한 타일이라 flash 의 온라인 최대 갱신이 필요 없다(한 번에 정확한 최대). 부동소수 원자 없음 → 결정적, 묶음 크기 무관.
// 예전 스칼라 커널(attn_k)과 다른 점은 P 를 16 비트로 반올림하는 것뿐(상대 2^-11, CPU EMUL 도 같게 — tools/vit_ref.cpp).
// 역전파(RecallVLA 학습)용: 행 logsumexp = max + log(합) 을 저장하면 FA2 식 dQ·dK·dV 를 같은 타일로 다시 계산할 수 있다(남은 일, 지금은 앞만).
template <bool HF>
__global__ void __launch_bounds__(128) attn_tc_k(const uint16_t* __restrict__ qkv, uint16_t* __restrict__ out, int bug) {
  __shared__ __align__(128) uint8_t sm[2 * NTOK * 128];   // K, V (Q 는 워프마다 자기 16 행만 쓰므로 전역에서 바로 조각으로)
  const int img = blockIdx.x / HEADS, hh = blockIdx.x % HEADS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const size_t base = (size_t)img * NTOK;
  const unsigned s0 = (unsigned)__cvta_generic_to_shared(sm);
#pragma unroll
  for (int q = tid; q < 2 * NTOK * 8; q += 128) {
    const int m = q / (NTOK * 8), row = (q / 8) % NTOK, c = q % 8;
    hk::cp16(s0 + m * NTOK * 128 + hk::swz<64>(row, c), qkv + (base + row) * 3 * D + (m + 1) * D + hh * HD + c * 8);
  }
  hk::commit();
  uint32_t qf[HD / 16][4];   // m16n8k16 A 조각: (행 g, k 2t4) (행 g+8, k 2t4) (행 g, k 8+2t4) (행 g+8, k 8+2t4)
  {
    const uint32_t* q0 = reinterpret_cast<const uint32_t*>(qkv + (base + warp * 16 + g) * 3 * D + hh * HD + 2 * t4);
    const uint32_t* q1 = q0 + 8 * 3 * D / 2;
#pragma unroll
    for (int kk = 0; kk < HD / 16; ++kk) {
      qf[kk][0] = q0[kk * 8]; qf[kk][1] = q1[kk * 8]; qf[kk][2] = q0[kk * 8 + 4]; qf[kk][3] = q1[kk * 8 + 4];
    }
  }
  hk::wait<0>();
  __syncthreads();
  const unsigned sk = s0, sv = s0 + NTOK * 128;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  float S[8][4];
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) S[a][c] = 0.f;
#pragma unroll
  for (int kk = 0; kk < HD / 16; ++kk) {
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      uint32_t r[4];
      hk::ldsm4(r, sk + hk::swz<64>(nj * 16 + lr + j1 * 8, kk * 2 + j0));
      hk::mma32<HF ? hk::HK_F16 : hk::HK_BF16>(S[2 * nj], qf[kk], r[0], r[1]);
      hk::mma32<HF ? hk::HK_F16 : hk::HK_BF16>(S[2 * nj + 1], qf[kk], r[2], r[3]);
    }
  }
  const float scale = bug == 1 ? 1.f : 0.125f;
  float mx[2] = {-3.0e38f, -3.0e38f};
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) { S[a][c] = S[a][c] * scale; mx[c >> 1] = fmaxf(mx[c >> 1], S[a][c]); }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
    mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
  }
  uint32_t P[8][2];   // n 조각마다 (행 g 의 두 값, 행 g+8 의 두 값) 16 비트 쌍
  float sum[2] = {0.f, 0.f};
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint16_t p0 = st16<HF>(expf(S[a][2 * h] - mx[h])), p1 = st16<HF>(expf(S[a][2 * h + 1] - mx[h]));
      sum[h] = sum[h] + ld16<HF>(p0);
      sum[h] = sum[h] + ld16<HF>(p1);
      P[a][h] = (uint32_t)p0 | ((uint32_t)p1 << 16);
    }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sum[h] = sum[h] + __shfl_xor_sync(0xffffffffu, sum[h], 1);
    sum[h] = sum[h] + __shfl_xor_sync(0xffffffffu, sum[h], 2);
  }
  float O[8][4];
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) O[a][c] = 0.f;
#pragma unroll
  for (int j = 0; j < NTOK / 16; ++j) {
    const uint32_t af[4] = {P[2 * j][0], P[2 * j][1], P[2 * j + 1][0], P[2 * j + 1][1]};
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      uint32_t r[4];
      const unsigned a = sv + hk::swz<64>(j * 16 + lr + j0 * 8, nj * 2 + j1);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
      hk::mma32<HF ? hk::HK_F16 : hk::HK_BF16>(O[2 * nj], af, r[0], r[1]);
      hk::mma32<HF ? hk::HK_F16 : hk::HK_BF16>(O[2 * nj + 1], af, r[2], r[3]);
    }
  }
  const float inv[2] = {1.f / sum[0], 1.f / sum[1]};
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const size_t row = base + warp * 16 + g + h * 8;
      reinterpret_cast<uint32_t*>(out)[(row * D + hh * HD + a * 8 + 2 * t4) >> 1] =
          (uint32_t)st16<HF>(O[a][2 * h] * inv[h]) | ((uint32_t)st16<HF>(O[a][2 * h + 1] * inv[h]) << 16);
    }
}

// ---- G6 FP8 경로 ---------------------------------------------------------------------------------------------------
// LayerNorm(FP32 통계, ln_k 와 같은 식) → 행 amax → E4M3 + 행 되돌림 배율. 워프 하나 = 행 하나
__global__ void __launch_bounds__(256) ln8_k(const float* X, int rows, const float* gm, const float* bt, uint8_t* out, float* srow, int bug) {
  const int r = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (r >= rows) return;
  const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)r * D);
  float v[24];
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const float4 q = x4[l + 32 * k];
    v[4 * k] = q.x; v[4 * k + 1] = q.y; v[4 * k + 2] = q.z; v[4 * k + 3] = q.w;
  }
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 24; ++k) s = s + v[k];
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, m);
  const float mean = s / (float)D;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < 24; ++k) { const float d = v[k] - mean; q = q + d * d; }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) q = q + __shfl_xor_sync(0xffffffffu, q, m);
  const float rstd = 1.f / sqrtf(q / (float)D + LN_EPS);
  float am = 0.f;
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const int c = 4 * (l + 32 * k);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float xc = bug == 2 ? v[4 * k + e] : v[4 * k + e] - mean;
      v[4 * k + e] = xc * rstd * gm[c + e] + bt[c + e];
      am = fmaxf(am, fabsf(v[4 * k + e]));
    }
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
  const float inv = f8::pow2_inv(am, f8::E4M3);
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const int c = 4 * (l + 32 * k);
    uint32_t w = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) w |= f8::q8<f8::E4M3>(v[4 * k + e] * inv) << (8 * e);
    *reinterpret_cast<uint32_t*>(out + (size_t)r * D + c) = w;
  }
  if (l == 0) srow[r] = bug == 4 ? 1.f : 1.f / inv;
}
// FP16 행(폭 W) → E4M3 + 행 배율. 워프 하나 = 행 하나, 레인마다 8 개 덩이
template <int W>
__global__ void __launch_bounds__(256) rowq_k(const uint16_t* X, int rows, uint8_t* out, float* srow, int bug) {
  constexpr int PER = W / 256;   // 레인당 덩이(8 개) 수: 768 → 3, 3072 → 12
  const int r = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (r >= rows) return;
  float v[PER][8];
  float am = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const uint4 q = *reinterpret_cast<const uint4*>(X + (size_t)r * W + (l + 32 * j) * 8);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[j][e] = __half2float(__ushort_as_half((uint16_t)(w[e >> 1] >> ((e & 1) * 16))));
      am = fmaxf(am, fabsf(v[j][e]));
    }
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
  const float inv = f8::pow2_inv(am, f8::E4M3);
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    uint32_t o[2] = {0u, 0u};
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e >> 2] |= f8::q8<f8::E4M3>(v[j][e] * inv) << ((e & 3) * 8);
    *reinterpret_cast<uint2*>(out + (size_t)r * W + (l + 32 * j) * 8) = make_uint2(o[0], o[1]);
  }
  if (l == 0) srow[r] = bug == 4 ? 1.f : 1.f / inv;
}
// FP8 GEMM 끝단: x = acc · 행 배율 · 채널 배율 + 편향 → (FP16 | GELU FP16 | FP32 잔차에 더하기)
template <int EPI>
struct VEpi8 {
  void* C; int ldc; const float* bias; const float* sa; const float* sw;
  __device__ __forceinline__ void operator()(int r, int c, float a0, float a1) const {
    const float s = sa[r];
    float x0 = a0 * s * sw[c] + bias[c], x1 = a1 * s * sw[c + 1] + bias[c + 1];
    if (EPI == VE_BF16 || EPI == VE_GELU) {
      if (EPI == VE_GELU) { x0 = gelu_tanh(x0); x1 = gelu_tanh(x1); }
      reinterpret_cast<uint32_t*>(C)[((long long)r * ldc + c) >> 1] = (uint32_t)st16<true>(x0) | ((uint32_t)st16<true>(x1) << 16);
    } else {
      float2* d = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (long long)r * ldc + c);
      float2 o = *d;
      o.x = o.x + x0;
      o.y = o.y + x1;
      *d = o;
    }
  }
};
// FP16 누산 GEMM 끝단(vgemm_k 끝단과 같은 식)
template <int EPI>
struct VEpiH {
  void* C; int ldc; const float* bias; const float* pos; int bug;
  __device__ __forceinline__ void operator()(int r, int c, float a0, float a1) const {
    float x0 = a0 + bias[c], x1 = a1 + bias[c + 1];
    if (EPI == VE_BF16 || EPI == VE_GELU) {
      if (EPI == VE_GELU) { x0 = gelu_tanh(x0); x1 = gelu_tanh(x1); }
      reinterpret_cast<uint32_t*>(C)[((long long)r * ldc + c) >> 1] = (uint32_t)st16<true>(x0) | ((uint32_t)st16<true>(x1) << 16);
    } else if (EPI == VE_F32_POS) {
      if (bug != 3) { const float* ps = pos + (size_t)(r % NTOK) * D + c; x0 = x0 + ps[0]; x1 = x1 + ps[1]; }
      *reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (long long)r * ldc + c) = make_float2(x0, x1);
    } else {
      float2* d = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (long long)r * ldc + c);
      float2 o = *d;
      o.x = o.x + x0;
      o.y = o.y + x1;
      *d = o;
    }
  }
};
// GEMM v2(vit_gemm.cuh) 끝단: VEpiH 와 같은 식(x = acc + 편향 → 16 비트 | GELU 16 비트 | FP32 + 위치 | FP32 잔차에 더함), 16 비트 형식 HF, GELU 는 빠른 꼴
template <bool HF, int EPI>
struct VEpi2 {
  void* C; int ldc; const float* bias; const float* pos; int bug;
  static constexpr bool RMW = EPI == VE_F32_ACC || EPI == VE_F32_POS;
  __device__ __forceinline__ float2 colv(int c) const { return *reinterpret_cast<const float2*>(bias + c); }
  __device__ __forceinline__ float2 ld(int r, int c) const {
    if (EPI == VE_F32_POS) return bug == 3 ? make_float2(0.f, 0.f) : *reinterpret_cast<const float2*>(pos + (size_t)(r % NTOK) * D + c);
    return *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(C) + (long long)r * ldc + c);
  }
  __device__ __forceinline__ void st(int r, int c, float x0, float x1, float2 o) const {
    if (EPI == VE_BF16 || EPI == VE_GELU) {
      if (EPI == VE_GELU) { x0 = gelu_fast(x0); x1 = gelu_fast(x1); }
      reinterpret_cast<uint32_t*>(C)[((long long)r * ldc + c) >> 1] = (uint32_t)st16<HF>(x0) | ((uint32_t)st16<HF>(x1) << 16);
    } else {   // 잔차: X + (acc + b) / 위치: (acc + b) + pos — 예전 커널과 같은 덧셈 차례
      const float2 v = EPI == VE_F32_ACC ? make_float2(o.x + x0, o.y + x1) : make_float2(x0 + o.x, x1 + o.y);
      *reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (long long)r * ldc + c) = v;
    }
  }
};
// 타일: 블록 128 × 256 × k64, 워프 8 개 × 64 × 64, 2 단(공유 96 KB), SM 당 블록 1 — 잰 값(README "인코더 속도판"): 영상 2,048 장 블록 GEMM 넷이
// 끝단 포함 FP16 누산 qkv 135·proj 100·fc1(GELU) 117·fc2 131 TFLOPS(예전 128 × 128 × k32 4 단: 86–96). 128 × 128 SM 당 2 블록·k32 3–4 단은 합이 같거나 느림.
constexpr int H2BM = 128, H2BN = 256, H2WM = 64, H2WN = 64, H2BK = 64, H2ST = 2;
template <int KIND, bool HF, int EPI>
static void hgemm_attr() {
  VTK(cudaFuncSetAttribute(hk::gemm_k<KIND, f8::HPROMO, H2BM, H2BN, H2WM, H2WN, H2BK, H2ST, 1, VEpi2<HF, EPI>>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           hk::smem_bytes<H2BM, H2BN, H2BK>(H2ST)));
}
template <int KIND, bool HF, int EPI>
static void hgemm(const uint16_t* A, const uint16_t* W, int M, int N, int K, const VEpi2<HF, EPI>& e, cudaStream_t st) {
  if (K % H2BK || N % H2BN || M < 1) { std::fprintf(stderr, "hgemm: bad shape M %d N %d K %d\n", M, N, K); std::abort(); }
  hk::gemm_k<KIND, f8::HPROMO, H2BM, H2BN, H2WM, H2WN, H2BK, H2ST, 1, VEpi2<HF, EPI>>
      <<<dim3(N / H2BN, (M + H2BM - 1) / H2BM), (H2BM / H2WM) * (H2BN / H2WN) * 32, hk::smem_bytes<H2BM, H2BN, H2BK>(H2ST), st>>>(A, K, W, K, M, K, e);
  VTK(cudaGetLastError());
}
// 16 비트 GEMM 하나: h16 = FP16 누산(HF 일 때만), 아니면 FP32 누산(HF: FP16, 아니면 BF16 피연산자)
template <bool HF, int EPI>
static void gemm16(bool h16, const uint16_t* A, const uint16_t* W, int M, int N, int K, const VEpi2<HF, EPI>& e, cudaStream_t st) {
  if (HF && h16) hgemm<hk::HK_F16H, HF, EPI>(A, W, M, N, K, e, st);
  else if (HF) hgemm<hk::HK_F16, HF, EPI>(A, W, M, N, K, e, st);
  else hgemm<hk::HK_BF16, HF, EPI>(A, W, M, N, K, e, st);
}
template <bool HF>
static void gemm16_attr_all() {
  if (HF) {
    hgemm_attr<hk::HK_F16H, HF, VE_BF16>(); hgemm_attr<hk::HK_F16H, HF, VE_GELU>(); hgemm_attr<hk::HK_F16H, HF, VE_F32_ACC>(); hgemm_attr<hk::HK_F16H, HF, VE_F32_POS>();
    hgemm_attr<hk::HK_F16, HF, VE_BF16>(); hgemm_attr<hk::HK_F16, HF, VE_GELU>(); hgemm_attr<hk::HK_F16, HF, VE_F32_ACC>(); hgemm_attr<hk::HK_F16, HF, VE_F32_POS>();
  } else {
    hgemm_attr<hk::HK_BF16, HF, VE_BF16>(); hgemm_attr<hk::HK_BF16, HF, VE_GELU>(); hgemm_attr<hk::HK_BF16, HF, VE_F32_ACC>(); hgemm_attr<hk::HK_BF16, HF, VE_F32_POS>();
  }
}
constexpr int V8BM = 128, V8BN = 128, V8WM = 64, V8WN = 32, V8ST = 4;
template <int EPI>
static void vgemmh_attr() {
  VTK(cudaFuncSetAttribute(f8::tn_k<f8::TN_F16H, V8BM, V8BN, V8WM, V8WN, V8ST, VEpiH<EPI>>, cudaFuncAttributeMaxDynamicSharedMemorySize, f8::tn_smem<V8BM, V8BN>(V8ST)));
}
template <int EPI>
static void vgemmh(const uint16_t* A, const uint16_t* W, int M, int N, int K, const VEpiH<EPI>& e, cudaStream_t st) {
  if (K % 32 || N % V8BN || M % 64) { std::fprintf(stderr, "vgemmh: bad shape M %d N %d K %d\n", M, N, K); std::abort(); }
  f8::tn_k<f8::TN_F16H, V8BM, V8BN, V8WM, V8WN, V8ST, VEpiH<EPI>><<<dim3(N / V8BN, (M + V8BM - 1) / V8BM), (V8BM / V8WM) * (V8BN / V8WN) * 32,
                                                                   f8::tn_smem<V8BM, V8BN>(V8ST), st>>>(A, K, W, K, M, N, K, e);
  VTK(cudaGetLastError());
}
template <int EPI>
static void vgemm8_attr() {
  VTK(cudaFuncSetAttribute(f8::tn_k<f8::TN_E4E4, V8BM, V8BN, V8WM, V8WN, V8ST, VEpi8<EPI>>, cudaFuncAttributeMaxDynamicSharedMemorySize, f8::tn_smem<V8BM, V8BN>(V8ST)));
}
// C[M][N] = A8[M][K] · W8[N][K]ᵀ (바이트, K 연속)
template <int EPI>
static void vgemm8(const uint8_t* A8, const uint8_t* W8, int M, int N, int K, const VEpi8<EPI>& e, cudaStream_t st) {
  if (K % 64 || N % V8BN || M % 64) { std::fprintf(stderr, "vgemm8: bad shape M %d N %d K %d\n", M, N, K); std::abort(); }
  f8::tn_k<f8::TN_E4E4, V8BM, V8BN, V8WM, V8WN, V8ST, VEpi8<EPI>><<<dim3(N / V8BN, (M + V8BM - 1) / V8BM), (V8BM / V8WM) * (V8BN / V8WN) * 32,
                                                                  f8::tn_smem<V8BM, V8BN>(V8ST), st>>>(
      reinterpret_cast<const uint16_t*>(A8), K / 2, reinterpret_cast<const uint16_t*>(W8), K / 2, M, N, K / 2, e);
  VTK(cudaGetLastError());
}

// K11 패치 자르기: 스레드 하나 = (영상 행, k 8 개). v = (u8/255 − 0.5)/0.5 (torchvision ToTensor + Normalize(0.5, 0.5) 와 같은 식)
template <bool HF>
__global__ void patchify_k(const uint8_t* cam0, const uint8_t* cam1, int n, int row0, uint16_t* P) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)n * 2 * NTOK * (KP / 8);
  if (q >= total) return;
  const int kc = (int)(q % (KP / 8));
  const long long rr = q / (KP / 8);
  const int p = (int)(rr % NTOK), ci = (int)((rr / NTOK) % 2), e = (int)(rr / (2 * NTOK));
  const int k0 = kc * 8, ch = k0 / (PATCH * PATCH), ky = (k0 % (PATCH * PATCH)) / PATCH, kx0 = k0 % PATCH;
  const int y = (p / GRID) * PATCH + ky, x = (p % GRID) * PATCH + kx0;
  const uint8_t* src = (ci ? cam1 : cam0) + ((size_t)e * IMG * IMG + (size_t)y * IMG + x) * 3 + ch;
  uint16_t o[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float v = (float)src[3 * k] / 255.f;
    o[k] = st16<HF>((v - 0.5f) / 0.5f);
  }
  uint4 w;
  w.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
  w.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
  w.z = (uint32_t)o[4] | ((uint32_t)o[5] << 16);
  w.w = (uint32_t)o[6] | ((uint32_t)o[7] << 16);
  const long long img = (long long)(row0 + e) * 2 + ci;
  *reinterpret_cast<uint4*>(P + ((size_t)img * NTOK + p) * KP + k0) = w;
}

__global__ void tokbuf_k(uint16_t* tok, long long rows) {
  const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  for (int c = D; c < TOK_LD; ++c) tok[r * TOK_LD + c] = c == D ? (uint16_t)0x3f80 : (uint16_t)0;
}
void init_token_buffer(uint16_t* tok, long long rows, cudaStream_t st) {
  tokbuf_k<<<(unsigned)((rows + 255) / 256), 256, 0, st>>>(tok, rows);
  VTK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
template <class T_>
static T_* dalloc(DevWeights& W, size_t n) {
  void* p = nullptr;
  VTK(cudaMalloc(&p, n * sizeof(T_)));
  W.allocs.push_back(p);
  W.bytes += n * sizeof(T_);
  return reinterpret_cast<T_*>(p);
}
static float* up_f(DevWeights& W, const std::vector<float>& v) {
  float* d = dalloc<float>(W, v.size());
  VTK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
static uint16_t* up16(DevWeights& W, const std::vector<float>& v, bool half) {
  std::vector<uint16_t> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = half ? net_f2h(v[i]) : f2bf(v[i]);
  uint16_t* d = dalloc<uint16_t>(W, v.size());
  VTK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return d;
}

// 가중치 [N][K] FP32 → E4M3 바이트(출력 채널마다 2 의 거듭제곱 배율, 호스트 반올림 = cuda_fp8 호스트 함수 __nv_cvt_float_to_fp8) + 되돌림 배율
static void up8(DevWeights& W, const std::vector<float>& w, int N, int K, uint8_t*& q, float*& s) {
  std::vector<uint8_t> hq(w.size());
  std::vector<float> hs(N);
  for (int n = 0; n < N; ++n) {
    float am = 0.f;
    for (int k = 0; k < K; ++k) am = std::fmax(am, std::fabs(w[(size_t)n * K + k]));
    const float inv = f8::pow2_inv(am, f8::E4M3);
    hs[n] = 1.f / inv;
    for (int k = 0; k < K; ++k) hq[(size_t)n * K + k] = (uint8_t)__nv_cvt_float_to_fp8(w[(size_t)n * K + k] * inv, __NV_SATFINITE, __NV_E4M3);
  }
  q = dalloc<uint8_t>(W, hq.size());
  VTK(cudaMemcpy(q, hq.data(), hq.size(), cudaMemcpyHostToDevice));
  s = up_f(W, hs);
}
int h16_promo() { return f8::HPROMO; }
bool parse_f8(const std::string& spec, uint8_t (&f8)[LAYERS]) {
  for (int l = 0; l < LAYERS; ++l) f8[l] = 0;
  if (spec.empty() || spec == "none" || spec == "0") return true;
  if (spec == "all") { for (int l = 0; l < LAYERS; ++l) f8[l] = F8_ALL; return true; }
  if (spec.rfind("all-but:", 0) == 0) {
    for (int l = 0; l < LAYERS; ++l) f8[l] = F8_ALL;
    size_t p = 8;
    while (p < spec.size()) {
      const int l = std::atoi(spec.c_str() + p);
      if (l < 0 || l >= LAYERS) return false;
      f8[l] = 0;
      const size_t c = spec.find(',', p);
      if (c == std::string::npos) break;
      p = c + 1;
    }
    return true;
  }
  if (spec.size() != LAYERS) return false;
  for (int l = 0; l < LAYERS; ++l) {
    const char ch = spec[l];
    const int v = ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
    if (v < 0) return false;
    f8[l] = (uint8_t)v;
  }
  return true;
}

void Encoder::init(const HostWeights& hw, int max_images, bool fp16) {
  free_all();
  half = fp16;
  max_img = max_images;
  W.patch_w = up16(W, hw.patch_w, half);
  W.patch_b = up_f(W, hw.patch_b);
  W.pos = up_f(W, hw.pos);
  for (int l = 0; l < LAYERS; ++l) {
    const auto& B = hw.blk[l];
    auto& O = W.blk[l];
    O.ln1_g = up_f(W, B.ln1_g); O.ln1_b = up_f(W, B.ln1_b);
    O.qkv_w = up16(W, B.qkv_w, half); O.qkv_b = up_f(W, B.qkv_b);
    O.proj_w = up16(W, B.proj_w, half); O.proj_b = up_f(W, B.proj_b);
    O.ln2_g = up_f(W, B.ln2_g); O.ln2_b = up_f(W, B.ln2_b);
    O.fc1_w = up16(W, B.fc1_w, half); O.fc1_b = up_f(W, B.fc1_b);
    O.fc2_w = up16(W, B.fc2_w, half); O.fc2_b = up_f(W, B.fc2_b);
  }
  W.norm_g = up_f(W, hw.norm_g);
  W.norm_b = up_f(W, hw.norm_b);
  bool any8 = false;
  for (int l = 0; l < LAYERS; ++l) any8 = any8 || f8[l];
  if (any8 && !half) { std::fprintf(stderr, "vit: FP8 needs the FP16 path (half = true)\n"); std::abort(); }
  if (any8) {
    for (int l = 0; l < LAYERS; ++l) {
      const auto& B = hw.blk[l];
      auto& O = W.blk[l];
      up8(W, B.qkv_w, 3 * D, D, O.qkv_w8, O.qkv_s);
      up8(W, B.proj_w, D, D, O.proj_w8, O.proj_s);
      up8(W, B.fc1_w, MLP, D, O.fc1_w8, O.fc1_s);
      up8(W, B.fc2_w, D, MLP, O.fc2_w8, O.fc2_s);
    }
    vgemm8_attr<VE_BF16>(); vgemm8_attr<VE_GELU>(); vgemm8_attr<VE_F32_ACC>();
  }
  if (half) { vgemmh_attr<VE_BF16>(); vgemmh_attr<VE_GELU>(); vgemmh_attr<VE_F32_ACC>(); vgemmh_attr<VE_F32_POS>(); }
  if (half) gemm16_attr_all<true>();
  else gemm16_attr_all<false>();
  const size_t T = (size_t)max_img * NTOK;
  auto mk = [&](size_t nb) { void* p = nullptr; VTK(cudaMalloc(&p, nb)); VTK(cudaMemset(p, 0, nb)); this->bytes += nb; return p; };
  X = (float*)mk(T * D * 4);
  ln = (uint16_t*)mk(T * D * 2);
  qkv = (uint16_t*)mk(T * 3 * D * 2);
  h = (uint16_t*)mk(T * MLP * 2);
  patches = h;
  if (any8) {
    a8 = (uint8_t*)mk(T * MLP);
    srow = (float*)mk(T * 4);
  }
  for (bool hf : {false, true}) {
    if (hf) {
      vgemm_attr<true, VE_BF16, 128, 128, 64, 32>(); vgemm_attr<true, VE_GELU, 128, 128, 64, 32>();
      vgemm_attr<true, VE_F32_POS, 128, 128, 64, 32>(); vgemm_attr<true, VE_F32_ACC, 128, 128, 64, 32>();
    } else {
      vgemm_attr<false, VE_BF16, 128, 128, 64, 32>(); vgemm_attr<false, VE_GELU, 128, 128, 64, 32>();
      vgemm_attr<false, VE_F32_POS, 128, 128, 64, 32>(); vgemm_attr<false, VE_F32_ACC, 128, 128, 64, 32>();
    }
  }
  VTK(cudaDeviceSynchronize());
}
void Encoder::free_all() {
  for (void* p : W.allocs) cudaFree(p);
  W.allocs.clear();
  W.bytes = 0;
  for (void* p : {(void*)X, (void*)ln, (void*)qkv, (void*)h, (void*)a8, (void*)srow}) if (p) cudaFree(p);
  X = nullptr; ln = qkv = h = patches = nullptr;
  a8 = nullptr;
  srow = nullptr;
  bytes = 0;
}

void Encoder::patchify(const uint8_t* cam0, const uint8_t* cam1, int n, int row0, cudaStream_t st) {
  const long long total = (long long)n * 2 * NTOK * (KP / 8);
  if (half) patchify_k<true><<<(unsigned)((total + 255) / 256), 256, 0, st>>>(cam0, cam1, n, row0, patches);
  else patchify_k<false><<<(unsigned)((total + 255) / 256), 256, 0, st>>>(cam0, cam1, n, row0, patches);
  VTK(cudaGetLastError());
}

template <bool HF>
static void block_t(Encoder& e, int b, int n_img, cudaStream_t st, cudaEvent_t* ev = nullptr) {
  const int T = n_img * NTOK, bug = e.bug;
  int ne = 0;
  auto mark = [&] { if (ev) VTK(cudaEventRecord(ev[ne++], st)); };
  mark();
  const auto& B = e.W.blk[b];
  const uint8_t m = HF ? e.f8[b] : 0, hm = HF ? (uint8_t)(e.h16[b] & ~m) : 0;
  const unsigned g8 = (T + 7) / 8;
  const bool k2 = e.kern != 0;
  if (m & F8_QKV) {
    ln8_k<<<g8, 256, 0, st>>>(e.X, T, B.ln1_g, B.ln1_b, e.a8, e.srow, bug);
    mark();
    vgemm8<VE_BF16>(e.a8, B.qkv_w8, T, 3 * D, D, VEpi8<VE_BF16>{e.qkv, 3 * D, B.qkv_b, e.srow, B.qkv_s}, st);
  } else {
    ln_k<HF><<<g8, 256, 0, st>>>(e.X, T, B.ln1_g, B.ln1_b, e.ln, D, bug);
    mark();
    if (k2) gemm16<HF, VE_BF16>(hm & F8_QKV, e.ln, B.qkv_w, T, 3 * D, D, VEpi2<HF, VE_BF16>{e.qkv, 3 * D, B.qkv_b, nullptr, bug}, st);
    else if (hm & F8_QKV) vgemmh<VE_BF16>(e.ln, B.qkv_w, T, 3 * D, D, VEpiH<VE_BF16>{e.qkv, 3 * D, B.qkv_b, nullptr, bug}, st);
    else vgemm<HF, VE_BF16>(VG{e.ln, B.qkv_w, e.qkv, B.qkv_b, nullptr, T, 3 * D, D, D, D, 3 * D, bug}, st);
  }
  mark();
  if (k2) attn_tc_k<HF><<<n_img * HEADS, 128, 0, st>>>(e.qkv, e.ln, bug);
  else attn_k<HF><<<n_img * HEADS, NTOK, 0, st>>>(e.qkv, e.ln, bug);
  mark();
  if (m & F8_PROJ) {
    rowq_k<D><<<g8, 256, 0, st>>>(e.ln, T, e.a8, e.srow, bug);
    vgemm8<VE_F32_ACC>(e.a8, B.proj_w8, T, D, D, VEpi8<VE_F32_ACC>{e.X, D, B.proj_b, e.srow, B.proj_s}, st);
  } else if (k2) {
    gemm16<HF, VE_F32_ACC>(hm & F8_PROJ, e.ln, B.proj_w, T, D, D, VEpi2<HF, VE_F32_ACC>{e.X, D, B.proj_b, nullptr, bug}, st);
  } else if (hm & F8_PROJ) {
    vgemmh<VE_F32_ACC>(e.ln, B.proj_w, T, D, D, VEpiH<VE_F32_ACC>{e.X, D, B.proj_b, nullptr, bug}, st);
  } else {
    vgemm<HF, VE_F32_ACC>(VG{e.ln, B.proj_w, e.X, B.proj_b, nullptr, T, D, D, D, D, D, bug}, st);
  }
  mark();
  if (m & F8_FC1) {
    ln8_k<<<g8, 256, 0, st>>>(e.X, T, B.ln2_g, B.ln2_b, e.a8, e.srow, bug);
    mark();
    vgemm8<VE_GELU>(e.a8, B.fc1_w8, T, MLP, D, VEpi8<VE_GELU>{e.h, MLP, B.fc1_b, e.srow, B.fc1_s}, st);
  } else {
    ln_k<HF><<<g8, 256, 0, st>>>(e.X, T, B.ln2_g, B.ln2_b, e.ln, D, bug);
    mark();
    if (k2) gemm16<HF, VE_GELU>(hm & F8_FC1, e.ln, B.fc1_w, T, MLP, D, VEpi2<HF, VE_GELU>{e.h, MLP, B.fc1_b, nullptr, bug}, st);
    else if (hm & F8_FC1) vgemmh<VE_GELU>(e.ln, B.fc1_w, T, MLP, D, VEpiH<VE_GELU>{e.h, MLP, B.fc1_b, nullptr, bug}, st);
    else vgemm<HF, VE_GELU>(VG{e.ln, B.fc1_w, e.h, B.fc1_b, nullptr, T, MLP, D, D, D, MLP, bug}, st);
  }
  mark();
  if (m & F8_FC2) {
    rowq_k<MLP><<<g8, 256, 0, st>>>(e.h, T, e.a8, e.srow, bug);
    vgemm8<VE_F32_ACC>(e.a8, B.fc2_w8, T, D, MLP, VEpi8<VE_F32_ACC>{e.X, D, B.fc2_b, e.srow, B.fc2_s}, st);
  } else if (k2) {
    gemm16<HF, VE_F32_ACC>(hm & F8_FC2, e.h, B.fc2_w, T, D, MLP, VEpi2<HF, VE_F32_ACC>{e.X, D, B.fc2_b, nullptr, bug}, st);
  } else if (hm & F8_FC2) {
    vgemmh<VE_F32_ACC>(e.h, B.fc2_w, T, D, MLP, VEpiH<VE_F32_ACC>{e.X, D, B.fc2_b, nullptr, bug}, st);
  } else {
    vgemm<HF, VE_F32_ACC>(VG{e.h, B.fc2_w, e.X, B.fc2_b, nullptr, T, D, MLP, MLP, MLP, D, bug}, st);
  }
  mark();
  VTK(cudaGetLastError());
}
void Encoder::block_stage_ms(int b, int n_img, cudaStream_t st, float (&ms)[7]) {
  // FP8 이 켜진 층은 LN 이 GEMM 앞 양자화와 합쳐져(ln8_k) 단계 경계가 조금 다르다(LN1 칸 = LN1 + FP8 양자화)
  cudaEvent_t ev[10];
  for (auto& x : ev) VTK(cudaEventCreate(&x));
  if (half) block_t<true>(*this, b, n_img, st, ev);
  else block_t<false>(*this, b, n_img, st, ev);
  VTK(cudaEventSynchronize(ev[7]));
  for (int i = 0; i < 7; ++i) VTK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
  for (auto& x : ev) VTK(cudaEventDestroy(x));
}
void Encoder::block(int b, int n_img, cudaStream_t st) {
  if (half) block_t<true>(*this, b, n_img, st);
  else block_t<false>(*this, b, n_img, st);
}

void Encoder::run(int n_img, uint16_t* out, cudaStream_t st, int layers) {
  if (n_img > max_img) { std::fprintf(stderr, "vit: %d images > max %d\n", n_img, max_img); std::abort(); }
  const int T = n_img * NTOK;
  const VG pg{patches, W.patch_w, X, W.patch_b, W.pos, T, D, KP, KP, KP, D, bug};
  if (kern && half) gemm16<true, VE_F32_POS>(h16_patch, patches, W.patch_w, T, D, KP, VEpi2<true, VE_F32_POS>{X, D, W.patch_b, W.pos, bug}, st);
  else if (kern) gemm16<false, VE_F32_POS>(false, patches, W.patch_w, T, D, KP, VEpi2<false, VE_F32_POS>{X, D, W.patch_b, W.pos, bug}, st);
  else if (half && h16_patch) vgemmh<VE_F32_POS>(patches, W.patch_w, T, D, KP, VEpiH<VE_F32_POS>{X, D, W.patch_b, W.pos, bug}, st);
  else if (half) vgemm<true, VE_F32_POS>(pg, st);
  else vgemm<false, VE_F32_POS>(pg, st);
  for (int b = 0; b < layers; ++b) block(b, n_img, st);
  ln_k<false><<<(T + 7) / 8, 256, 0, st>>>(X, T, W.norm_g, W.norm_b, out, TOK_LD, bug);   // 학생 입력 토큰은 bf16 (RL 신경망 규칙)
  VTK(cudaGetLastError());
}

}  // namespace vit
