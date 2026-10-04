// 얼린 SigLIP 2 B/32-256 영상 탑(패치 토큰) — 설명은 include/vit.h.
// GEMM 은 training/RL/network/src/gemm.cuh 의 조각(ldmatrix·cp.async·mma.sync bf16, 읽기만 include)으로 짠 3 단 파이프라인 커널에
// 이 인코더 끝단(편향 FP32, GELU tanh, 위치 임베딩, 잔차 FP32 더하기)을 붙였다. LayerNorm·softmax 는 FP32. 부동소수 원자 연산 없음(결정적).
#include <dirent.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_fp16.h>

#include "gemm.cuh"
#include "vit.h"

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

__device__ __forceinline__ float gelu_tanh(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + tanhf(u));
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
  const size_t T = (size_t)max_img * NTOK;
  auto mk = [&](size_t nb) { void* p = nullptr; VTK(cudaMalloc(&p, nb)); VTK(cudaMemset(p, 0, nb)); this->bytes += nb; return p; };
  X = (float*)mk(T * D * 4);
  ln = (uint16_t*)mk(T * D * 2);
  qkv = (uint16_t*)mk(T * 3 * D * 2);
  h = (uint16_t*)mk(T * MLP * 2);
  patches = h;
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
  for (void* p : {(void*)X, (void*)ln, (void*)qkv, (void*)h}) if (p) cudaFree(p);
  X = nullptr; ln = qkv = h = patches = nullptr;
  bytes = 0;
}

void Encoder::patchify(const uint8_t* cam0, const uint8_t* cam1, int n, int row0, cudaStream_t st) {
  const long long total = (long long)n * 2 * NTOK * (KP / 8);
  if (half) patchify_k<true><<<(unsigned)((total + 255) / 256), 256, 0, st>>>(cam0, cam1, n, row0, patches);
  else patchify_k<false><<<(unsigned)((total + 255) / 256), 256, 0, st>>>(cam0, cam1, n, row0, patches);
  VTK(cudaGetLastError());
}

template <bool HF>
static void block_t(Encoder& e, int b, int n_img, cudaStream_t st) {
  const int T = n_img * NTOK, bug = e.bug;
  const auto& B = e.W.blk[b];
  ln_k<HF><<<(T + 7) / 8, 256, 0, st>>>(e.X, T, B.ln1_g, B.ln1_b, e.ln, D, bug);
  vgemm<HF, VE_BF16>(VG{e.ln, B.qkv_w, e.qkv, B.qkv_b, nullptr, T, 3 * D, D, D, D, 3 * D, bug}, st);
  attn_k<HF><<<n_img * HEADS, NTOK, 0, st>>>(e.qkv, e.ln, bug);
  vgemm<HF, VE_F32_ACC>(VG{e.ln, B.proj_w, e.X, B.proj_b, nullptr, T, D, D, D, D, D, bug}, st);
  ln_k<HF><<<(T + 7) / 8, 256, 0, st>>>(e.X, T, B.ln2_g, B.ln2_b, e.ln, D, bug);
  vgemm<HF, VE_GELU>(VG{e.ln, B.fc1_w, e.h, B.fc1_b, nullptr, T, MLP, D, D, D, MLP, bug}, st);
  vgemm<HF, VE_F32_ACC>(VG{e.h, B.fc2_w, e.X, B.fc2_b, nullptr, T, D, MLP, MLP, MLP, D, bug}, st);
  VTK(cudaGetLastError());
}
void Encoder::block(int b, int n_img, cudaStream_t st) {
  if (half) block_t<true>(*this, b, n_img, st);
  else block_t<false>(*this, b, n_img, st);
}

void Encoder::run(int n_img, uint16_t* out, cudaStream_t st, int layers) {
  if (n_img > max_img) { std::fprintf(stderr, "vit: %d images > max %d\n", n_img, max_img); std::abort(); }
  const int T = n_img * NTOK;
  const VG pg{patches, W.patch_w, X, W.patch_b, W.pos, T, D, KP, KP, KP, D, bug};
  if (half) vgemm<true, VE_F32_POS>(pg, st);
  else vgemm<false, VE_F32_POS>(pg, st);
  for (int b = 0; b < layers; ++b) block(b, n_img, st);
  ln_k<false><<<(T + 7) / 8, 256, 0, st>>>(X, T, W.norm_g, W.norm_b, out, TOK_LD, bug);   // 학생 입력 토큰은 bf16 (RL 신경망 규칙)
  VTK(cudaGetLastError());
}

}  // namespace vit
