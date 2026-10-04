// 옵티마이저 — 설명은 include/optim.h. 원소 갱신 식은 호스트·장치 같은 함수(OPT_HD)라 CPU 흉내와 비트가 같다(--fmad=false, 정확한 √).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "net.h"
#include "optim.h"

namespace rvla {

#define OCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
#define OPT_HD __host__ __device__ __forceinline__
constexpr int QB = 256;   // 양자화 블록
constexpr int GNB = 240;  // 노름 조각

OPT_HD float dq_m(int8_t c, float amax) {
  const int u = (uint8_t)c, mag = u & 127;
  if (mag == 127 || amax == 0.f) return 0.f;
  const int e = mag >> 3, f = mag & 7;
  const float v = amax * ldexpf(1.f - (float)f / 16.f, -e);
  return (u & 128) ? -v : v;
}
OPT_HD int8_t q_m(float x, float amax) {
  if (amax == 0.f) return (int8_t)127;
  const float y = fabsf(x) / amax;
  if (y == 0.f) return (int8_t)127;
  int ex;
  frexpf(y, &ex);                     // y = z·2^ex, z ∈ [0.5, 1). 우리 꼴: y = 2^−e(1 − f/16), f 0..7
  int e, f;
  if (y >= 1.f) { e = 0; f = 0; }
  else {
    // y ∈ [2^−(e'+1), 2^−e'): e' = −ex, z = y·2^e' ∈ [0.5, 1)
    e = -ex;
    const float z = ldexpf(y, e);
    f = (int)rintf((1.f - z) * 16.f);
    if (f >= 8) { e += 1; f = 0; }
  }
  if (e > 15 || (e == 15 && f > 6)) return (int8_t)127;
  if (e < 0) { e = 0; f = 0; }
  const int code = (e << 3) | f;
  return (int8_t)(x < 0.f ? (code | 128) : code);
}
OPT_HD float dq_r(uint8_t c, float amax) {
  if (c == 255 || amax == 0.f) return 0.f;
  const int e = c >> 4, f = c & 15;
  return amax * ldexpf(1.f - (float)f / 32.f, -e);
}
OPT_HD uint8_t q_r(float x, float amax) {
  if (amax == 0.f || x <= 0.f) return 255;
  const float y = x / amax;
  int ex;
  frexpf(y, &ex);
  int e, f;
  if (y >= 1.f) { e = 0; f = 0; }
  else {
    e = -ex;
    const float z = ldexpf(y, e);
    f = (int)rintf((1.f - z) * 32.f);
    if (f >= 16) { e += 1; f = 0; }
  }
  if (e > 15 || (e == 15 && f > 14)) return 255;
  return (uint8_t)((e << 4) | f);
}
OPT_HD uint32_t hmix(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t z = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ c * 0x94D049BB133111EBull;
  z ^= z >> 31; z *= 0xD6E8FEB86655AC2Bull; z ^= z >> 29;
  return (uint32_t)z;
}
OPT_HD uint16_t bf_sr(float x, uint32_t rnd, int bug) {
  uint32_t u;
  memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
  if (bug != 1) u = u + (rnd & 0xffffu);
  return (uint16_t)(u >> 16);
}
// 한 원소: 새 w(FP32), 새 m, 새 r(양자화 전)
OPT_HD void adam_elem(const OptCfg& c, float p1, float p2, float coef, float w, float g, float m, float r, int bug, float* wn, float* mn, float* rn) {
  g = g * coef;
  const float m2 = c.b1 * m + (1.f - c.b1) * g;
  const float v2 = c.b2 * (bug == 2 ? r : r * r) + (1.f - c.b2) * g * g;
  const float mh = m2 / (1.f - p1), vh = v2 / (1.f - p2);
  *wn = w - c.lr * (mh / (sqrtf(vh) + c.eps) + c.wd * w);
  *mn = m2;
  *rn = bug == 2 ? v2 : sqrtf(v2);
}

// 장치 상태: st[0] = 자르기 배율, st[1] = β1^t, st[2] = β2^t (t = 이번 스텝 번호, 1 부터)
__global__ void adam8_k(uint16_t* W, const uint16_t* G, int8_t* m8, uint8_t* r8, float* ms, float* rs, long long n, const float* stt, OptCfg c,
                        long long sk0, long long sk1, const long long* t, int bug) {
  __shared__ float sa[2][8];
  const long long blk = blockIdx.x, i = blk * QB + threadIdx.x;
  const long long b0 = blk * QB, b1 = b0 + QB;
  if (b0 >= sk0 && b1 <= sk1) return;   // 얼린 블록
  const bool in = i < n;
  float wn = 0.f, mn = 0.f, rn = 0.f;
  if (in) {
    const float w = net::bf2f(W[i]), g = net::bf2f(G[i]);
    adam_elem(c, stt[1], stt[2], stt[0], w, g, dq_m(m8[i], ms[blk]), dq_r(r8[i], rs[blk]), bug, &wn, &mn, &rn);
  }
  float am = fabsf(mn), ar = rn;
  for (int o = 16; o > 0; o >>= 1) { am = fmaxf(am, __shfl_xor_sync(~0u, am, o)); ar = fmaxf(ar, __shfl_xor_sync(~0u, ar, o)); }
  if ((threadIdx.x & 31) == 0) { sa[0][threadIdx.x >> 5] = am; sa[1][threadIdx.x >> 5] = ar; }
  __syncthreads();
  float AM = 0.f, AR = 0.f;
  for (int k = 0; k < QB / 32; ++k) { AM = fmaxf(AM, sa[0][k]); AR = fmaxf(AR, sa[1][k]); }
  if (in) {
    W[i] = bf_sr(wn, hmix(c.seed, (uint64_t)t[0], (uint64_t)i), bug);
    m8[i] = q_m(mn, AM);
    r8[i] = q_r(rn, AR);
  }
  if (threadIdx.x == 0) { ms[blk] = AM; rs[blk] = AR; }
}
__global__ void adamf_k(float* P, const float* G, float* m, float* v, long long n, const float* stt, OptCfg c, float wd) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  OptCfg cc = c;
  cc.wd = wd;
  float wn, mn, rn;
  adam_elem(cc, stt[1], stt[2], stt[0], P[i], G[i], m[i], sqrtf(v[i]), 0, &wn, &mn, &rn);
  P[i] = wn; m[i] = mn; v[i] = rn * rn;
}
// 비교용 FP32 상태 행렬: FP32 원본 사본 wf 를 갱신하고 bf16 은 가장 가까운 값
__global__ void adam32m_k(uint16_t* W, float* wf, const uint16_t* G, float* m, float* v, long long n, const float* stt, OptCfg c) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float wn, mn, rn;
  adam_elem(c, stt[1], stt[2], stt[0], wf[i], net::bf2f(G[i]), m[i], sqrtf(v[i]), 0, &wn, &mn, &rn);
  wf[i] = wn; m[i] = mn; v[i] = rn * rn; W[i] = net::f2bf(wn);
}
__global__ void bf2f_k(const uint16_t* a, long long n, float* b) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = net::bf2f(a[i]);
}
// 제곱합 조각(고정 순서): 블록 k 가 [k·n/GNB, (k+1)·n/GNB)
__global__ void sq_bf_k(const uint16_t* g, long long n, float* part) {
  __shared__ float sh[256];
  const long long a = n * blockIdx.x / GNB, b = n * (blockIdx.x + 1) / GNB;
  float s = 0.f;
  for (long long i = a + threadIdx.x; i < b; i += 256) { const float x = net::bf2f(g[i]); s = s + x * x; }
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = sh[threadIdx.x] + sh[threadIdx.x + k]; __syncthreads(); }
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
__global__ void sq_f_k(const float* g, long long n, float* part) {
  __shared__ float sh[256];
  const long long a = n * blockIdx.x / GNB, b = n * (blockIdx.x + 1) / GNB;
  float s = 0.f;
  for (long long i = a + threadIdx.x; i < b; i += 256) s = s + g[i] * g[i];
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = sh[threadIdx.x] + sh[threadIdx.x + k]; __syncthreads(); }
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
// 노름 → 배율, β^t 갱신, t++
__global__ void opt_pre_k(const float* part, int np, float clip, float b1, float b2, float* out, float* stt, long long* t) {
  if (threadIdx.x != 0) return;
  float s = 0.f;
  for (int k = 0; k < np; ++k) s = s + part[k];
  const float nrm = sqrtf(s);
  out[0] = nrm;
  stt[0] = clip > 0.f && nrm > clip ? clip / (nrm + 1e-6f) : 1.f;
  t[0] = t[0] + 1;
  stt[1] = stt[1] * b1;
  stt[2] = stt[2] * b2;
}

void Opt::init(const OptCfg& cfg, const std::vector<PSet*>& sets) {
  c = cfg;
  auto al = [&](size_t b) { void* p; OCK(cudaMalloc(&p, b + 256)); OCK(cudaMemset(p, 0, b + 256)); allocs.push_back(p); bytes += b; return p; };
  for (PSet* p : sets) {
    Buf b;
    b.p = p;
    if (c.fp32_states) {
      b.mf = (float*)al(p->nW * 4); b.vf = (float*)al(p->nW * 4); b.wf = (float*)al(p->nW * 4);
      bf2f_k<<<(unsigned)((p->nW + 255) / 256), 256>>>(p->W, p->nW, b.wf);
    } else {
      const long long nbk = (p->nW + QB - 1) / QB;
      b.m8 = (int8_t*)al(p->nW); b.r8 = (uint8_t*)al(p->nW); b.ms = (float*)al(nbk * 4); b.rs = (float*)al(nbk * 4);
    }
    b.vm = (float*)al(p->nV * 4); b.vv = (float*)al(p->nV * 4);
    bufs.push_back(b);
  }
  gn = (float*)al(sizeof(float) * (bufs.size() * 2 * GNB + 8));
  t = (long long*)al(sizeof(long long) * 2);
  const float st0[3] = {1.f, 1.f, 1.f};
  OCK(cudaMemcpy(gn + bufs.size() * 2 * GNB + 1, st0, 12, cudaMemcpyHostToDevice));
  OCK(cudaDeviceSynchronize());
}

void Opt::step(cudaStream_t st) {
  const int nb2 = (int)bufs.size() * 2;
  float* part = gn + 0;   // 앞 칸은 조각, 끝에 [노름, 배율, β1^t, β2^t]
  float* tail = gn + (size_t)nb2 * GNB;
  for (size_t k = 0; k < bufs.size(); ++k) {
    sq_bf_k<<<GNB, 256, 0, st>>>(bufs[k].p->GW, bufs[k].p->nW, part + (2 * k) * GNB);
    sq_f_k<<<GNB, 256, 0, st>>>(bufs[k].p->GV, bufs[k].p->nV, part + (2 * k + 1) * GNB);
  }
  float* stt = tail + 1;
  opt_pre_k<<<1, 32, 0, st>>>(part, nb2 * GNB, c.clip, c.b1, c.b2, tail, stt, t);
  for (auto& b : bufs) {
    const long long n = b.p->nW;
    if (c.fp32_states) adam32m_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(b.p->W, b.wf, b.p->GW, b.mf, b.vf, n, stt, c);
    else adam8_k<<<(unsigned)((n + QB - 1) / QB), QB, 0, st>>>(b.p->W, b.p->GW, b.m8, b.r8, b.ms, b.rs, n, stt, c, b.skip0, b.skip1, t, bug);
    adamf_k<<<(unsigned)((b.p->nV + 255) / 256), 256, 0, st>>>(b.p->V, b.p->GV, b.vm, b.vv, b.p->nV, stt, c, 0.f);
  }
  OCK(cudaGetLastError());
}

void Opt::free_all() {
  for (void* p : allocs) cudaFree(p);
  allocs.clear();
}

void opt8_ref_block(const OptCfg& c, long long t, float coef, int n, uint16_t* w, int8_t* m8, uint8_t* r8, float* ms, float* rs, const uint16_t* g,
                    long long idx0) {
  float p1 = 1.f, p2 = 1.f;
  for (long long k = 0; k < t; ++k) { p1 = p1 * c.b1; p2 = p2 * c.b2; }
  std::vector<float> wn(n), mn(n), rn(n);
  float AM = 0.f, AR = 0.f;
  for (int i = 0; i < n; ++i) {
    adam_elem(c, p1, p2, coef, net::bf2f(w[i]), net::bf2f(g[i]), dq_m(m8[i], *ms), dq_r(r8[i], *rs), 0, &wn[i], &mn[i], &rn[i]);
    AM = std::fmax(AM, std::fabs(mn[i]));
    AR = std::fmax(AR, rn[i]);
  }
  for (int i = 0; i < n; ++i) {
    w[i] = bf_sr(wn[i], hmix(c.seed, (uint64_t)t, (uint64_t)(idx0 + i)), 0);
    m8[i] = q_m(mn[i], AM);
    r8[i] = q_r(rn[i], AR);
  }
  *ms = AM; *rs = AR;
}

}  // namespace rvla
