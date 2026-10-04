// 환경 커널: 스레드 하나 = 판 하나. 호스트 쪽은 장치 메모리와 실행만(상태는 제자리 갱신 — CUDA 그래프로 잡을 수 있게).
#include <cstdio>
#include <vector>

#include "env_api.h"
#include "env_soa.h"

namespace env {

// FURN: A2(가구) 판. A0/A1 은 가구 코드가 없는 판(FURN = false)을 띄운다 — 결과는 같고 레지스터·넘침이 예전과 같음
template <bool FURN>
__global__ void __launch_bounds__(128) init_kernel(Soa s, uint64_t seed, int stage) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) init_env<FURN>(s, i, seed, stage);
}
template <bool FURN>
__global__ void __launch_bounds__(128) step_kernel(Soa s, const float* act, float* obs, float* rew, int* done, int stage, int arm_free, int bug) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) step_env<FURN>(s, i, act, obs, rew, done, stage, arm_free != 0, bug);
}

// A2 리셋의 경로 꼭짓점 거리(path_prepare)를 워프가 나눠 한다: 리셋하는 판마다 차례로, 레인 j = 꼭짓점 j(N_PN = 32 = 워프).
// 순서·식은 path_prepare 그대로(같으면 앞 번호, 같은 path_clear 인자) — 바깥 고리(꼭짓점 v)만 레인으로 펼쳤으므로 결과 비트가 같다.
// 판 하나를 스레드 하나가 하면(다익스트라 32 × 32 × 상자 8) 리셋 판 하나가 스텝 커널 전체를 0.3 ms 붙잡았다.
static_assert(N_PN == 32, "one warp lane per path node");
__device__ void path_prepare_warp(const Core& c, bool need) {
  constexpr unsigned FULL = 0xffffffffu;
  const int j = threadIdx.x & 31;
  unsigned m = __ballot_sync(FULL, need);
  while (m) {
    const int L = __ffs((int)m) - 1;
    m &= m - 1u;
    Core::Furn f;
    f.b = reinterpret_cast<float*>(__shfl_sync(FULL, reinterpret_cast<unsigned long long>(c.fu.b), L));
    f.c = reinterpret_cast<int*>(__shfl_sync(FULL, reinterpret_cast<unsigned long long>(c.fu.c), L));
    f.p = reinterpret_cast<float*>(__shfl_sync(FULL, reinterpret_cast<unsigned long long>(c.fu.p), L));
    f.s = __shfl_sync(FULL, c.fu.s, L);
    const int nf = __shfl_sync(FULL, c.nf, L);
    const float rhx = __shfl_sync(FULL, c.rhx, L), rhy = __shfl_sync(FULL, c.rhy, L);
    const float tx = __shfl_sync(FULL, c.tx, L), ty = __shfl_sync(FULL, c.ty, L);
    float bx[N_FURN][4];
    for (int k = 0; k < nf; ++k) for (int q = 0; q < 4; ++q) bx[k][q] = f.box(k, q);
    float nx = 0.f, ny = 0.f, pd = kFar;
    bool done = true;
    if (j < 4 * nf) {
      path_node(bx, j, nx, ny);
      done = !path_node_ok(rhx, rhy, nf, bx, nx, ny);
      if (!done && path_clear(bx, nf, nx, ny, tx, ty)) pd = hyp(tx - nx, ty - ny);
    }
    for (int it = 0; it < N_PN; ++it) {
      float kp = done ? __int_as_float(0x7f800000) : pd;   // 끝난 꼭짓점은 +inf(고르지 않음)
      int ki = done ? (1 << 30) : j;
      for (int off = 16; off; off >>= 1) {
        const float op = __shfl_xor_sync(FULL, kp, off);
        const int oi = __shfl_xor_sync(FULL, ki, off);
        if (op < kp || (op == kp && oi < ki)) { kp = op; ki = oi; }
      }
      if (ki == (1 << 30) || !(kp < kFar)) break;   // 워프 전체가 같은 값
      if (j == ki) done = true;
      const float ux = __shfl_sync(FULL, nx, ki), uy = __shfl_sync(FULL, ny, ki);
      if (!done) {
        const float nd = kp + hyp(nx - ux, ny - uy);
        if (nd < pd && path_clear(bx, nf, ux, uy, nx, ny)) pd = nd;
      }
    }
    f.pd(j) = pd;
  }
}
// A2 스텝 커널: step_env<true> 와 같은 일. 리셋의 장면 만들기·끝은 판마다, 경로 꼭짓점은 워프가(위). 길 없음 대비 고리(reset_a2_paths)도 같은 순서
__global__ void __launch_bounds__(128) step_kernel_a2(Soa s, const float* act, float* obs, float* rew, int* done, int arm_free, int bug) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = i < s.N;
  Core c{};
  bool rs = false;
  if (live) {
    rs = step_env_body<true>(s, i, act, obs, rew, done, arm_free != 0, bug, c);
    if (rs) reset_a2_scene(c);
  }
  path_prepare_warp(c, rs);
  int kfb = 0;
  bool chk = rs;
  for (;;) {
    const bool need = chk && kfb < N_FURN && path_dist(c, c.x, c.y) < 0.f;
    if (!__any_sync(0xffffffffu, need)) break;
    if (need) { corner_box(c, kfb); ++kfb; }
    path_prepare_warp(c, need);
    chk = need;
  }
  if (rs) reset_tail(c);
  if (live) store(s, i, c);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceEnv::DeviceEnv(int N, int stage, uint64_t seed, bool arm_free) : N_(N), stage_(stage), arm_free_(arm_free) {
  CK(cudaMalloc(&f_, sizeof(float) * NUM_F * (size_t)N));
  CK(cudaMalloc(&iv_, sizeof(int) * NUM_I * (size_t)N));
  CK(cudaMalloc(&rng_, sizeof(uint64_t) * (size_t)N));
  CK(cudaMemset(f_, 0, sizeof(float) * NUM_F * (size_t)N));   // A0/A1 는 가구 칸을 쓰지 않으므로 CPU 참조판(0)과 같게
  CK(cudaMemset(iv_, 0, sizeof(int) * NUM_I * (size_t)N));
  Soa s{f_, iv_, rng_, N};
  if (stage >= 2) init_kernel<true><<<(N + 127) / 128, 128>>>(s, seed, stage);
  else init_kernel<false><<<(N + 127) / 128, 128>>>(s, seed, stage);
  CK(cudaGetLastError());
}
DeviceEnv::~DeviceEnv() { cudaFree(f_); cudaFree(iv_); cudaFree(rng_); }

void DeviceEnv::step(const float* act, float* obs, float* rew, int* done, int bug) {
  Soa s{f_, iv_, rng_, N_};
  if (stage_ >= 2) step_kernel_a2<<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug);
  else step_kernel<false><<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, stage_, arm_free_ ? 1 : 0, bug);
}

void DeviceEnv::download(std::vector<float>& f, std::vector<int>& iv, std::vector<uint64_t>& rng) const {
  f.resize((size_t)NUM_F * N_); iv.resize((size_t)NUM_I * N_); rng.resize(N_);
  CK(cudaMemcpy(f.data(), f_, sizeof(float) * f.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(iv.data(), iv_, sizeof(int) * iv.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rng.data(), rng_, sizeof(uint64_t) * rng.size(), cudaMemcpyDeviceToHost));
}

}  // namespace env
