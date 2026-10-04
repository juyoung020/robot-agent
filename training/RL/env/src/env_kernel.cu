// 환경 커널: 스레드 하나 = 판 하나. 호스트 쪽은 장치 메모리와 실행만(상태는 제자리 갱신 — CUDA 그래프로 잡을 수 있게).
#include <cstddef>
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
// ctl != nullptr(장치 단계 판): 단계를 장치 값에서 읽고, 이 커널 무리의 단계가 아니면 바로 끝남(블록 전체 같은 값)
template <bool FURN>
__global__ void __launch_bounds__(128) step_kernel(Soa s, const float* act, float* obs, float* rew, int* done, int stage, int arm_free, int bug, const EnvCtl* ctl) {
  if (ctl) { stage = ctl->stage; if (stage > 1) return; }
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
__global__ void __launch_bounds__(128) step_kernel_a2(Soa s, const float* act, float* obs, float* rew, int* done, int arm_free, int bug, const EnvCtl* ctl) {
  if (ctl && ctl->stage != 2) return;   // 장치 단계 판: 커널 전체(모든 워프) 같은 값 — 아래 워프 동기 전에 끝남
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

// BEHAVIOR 판(E2): 스레드 하나 = 판 하나. 장면·시작 조건 표는 모든 판이 같이 읽는다(장치 SceneSet 하나)
__global__ void __launch_bounds__(128) init_kernel_beh(Soa s, uint64_t seed, const bsc::SceneSet* ss, const bsc::BCurr* cu) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) init_env_beh(s, i, seed, *ss, *cu);
}
__global__ void __launch_bounds__(128) step_kernel_beh(Soa s, const float* act, float* obs, float* rew, int* done, int arm_free, int bug,
                                                       const bsc::SceneSet* ss, const bsc::BCurr* cu, bsc::NavFb fb, const EnvCtl* ctl) {
  if (ctl && ctl->stage < kStageBeh) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef ENV_PROF
  const long long k0 = clock64();
#endif
  if (i < s.N) step_env_beh<1>(s, i, act, obs, rew, done, arm_free != 0, bug, *ss, cu, fb);
#ifdef ENV_PROF
  const unsigned long long kc = (unsigned long long)(clock64() - k0);
  const unsigned long long km = __reduce_max_sync(0xffffffffu, (unsigned)kc);
  if ((threadIdx.x & 31) == 0) atomicAdd(&g_env_prof[1], km);
#endif
}

// 대본 교사(E6): 판마다 스레드 하나
__global__ void __launch_bounds__(128) teacher_kernel(Soa s, const bsc::SceneSet* ss, bsc::NavFb fb, float* act) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) teacher_step(s, i, *ss, fb, act);
}
void DeviceEnv::teacher(float* act) const {
  if (!ss_) return;
  teacher_kernel<<<(N_ + 127) / 128, 128>>>(Soa{f_, iv_, rng_, N_}, ss_, nav_, act);
}

// 잡기 물리 판(B4–B6, E6) 커널: 예전 판 커널 뒤에 띄움(위 SEL 규칙)
__global__ void __launch_bounds__(128) step_kernel_pnp(Soa s, const float* act, float* obs, float* rew, int* done, int arm_free, int bug,
                                                       const bsc::SceneSet* ss, const bsc::BCurr* cu, bsc::NavFb fb, const EnvCtl* ctl) {
  if (ctl && ctl->stage < kStageBeh) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) step_env_beh<2>(s, i, act, obs, rew, done, arm_free != 0, bug, *ss, cu, fb);
}

// 장치 단계 바꾸기(apply): 요청이 있으면 판마다 생성자와 같은 상태로 — 상태 칸을 모두 0 으로(생성자의 memset) 채우고 그 단계 init.
// 요청이 없으면 바로 끝남(그래프에 늘 넣어 둠). 단계·요청 지우기는 다음 커널(env_commit_k)이 — 모든 판이 같은 요청을 읽은 뒤
__global__ void __launch_bounds__(128) env_apply_k(Soa s, const EnvCtl* ctl, const bsc::SceneSet* ss, const bsc::BCurr* cu) {
  const int pend = ctl->pend;
  if (pend < 0) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x, N = s.N;
  if (i >= N) return;
  const uint64_t seed = ctl->seed;
  for (int k = 0; k < NUM_F; ++k) s.f[(size_t)k * N + i] = 0.f;
  for (int k = 0; k < NUM_I; ++k) s.iv[(size_t)k * N + i] = 0;
  s.rng[i] = 0;
  if (pend >= kStageBeh) init_env_beh(s, i, seed, *ss, *cu);
  else if (pend >= 2) init_env<true>(s, i, seed, pend);
  else init_env<false>(s, i, seed, pend);
}
__global__ void env_commit_k(EnvCtl* ctl) {
  if (ctl->pend >= 0) { ctl->stage = ctl->pend; ctl->pend = -1; }
}

#ifdef ENV_PROF
void env_prof_read(unsigned long long out[8]) { cudaMemcpyFromSymbol(out, g_env_prof, sizeof(unsigned long long) * 8); }
void env_prof_reset() { const unsigned long long z[8] = {}; cudaMemcpyToSymbol(g_env_prof, z, sizeof z); }
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceEnv::DeviceEnv(int N, int stage, uint64_t seed, bool arm_free, const bsc::SceneSet* ss_dev, const bsc::BCurr& cu0)
    : N_(N), stage_(stage), arm_free_(arm_free), ss_(ss_dev) {
  if (stage >= kStageBeh && !ss_dev) { std::fprintf(stderr, "DeviceEnv: BEHAVIOR stage %d needs a device SceneSet\n", stage); std::abort(); }
  CK(cudaMalloc(&bcurr_, sizeof(bsc::BCurr)));
  {
    const EnvCtl c0{stage, -1, (unsigned long long)seed};
    CK(cudaMalloc(&ctl_, sizeof(EnvCtl)));
    CK(cudaMemcpy(ctl_, &c0, sizeof c0, cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(bcurr_, &cu0, sizeof cu0, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&f_, sizeof(float) * NUM_F * (size_t)N));
  CK(cudaMalloc(&iv_, sizeof(int) * NUM_I * (size_t)N));
  CK(cudaMalloc(&rng_, sizeof(uint64_t) * (size_t)N));
  CK(cudaMemset(f_, 0, sizeof(float) * NUM_F * (size_t)N));   // A0/A1 는 가구 칸을 쓰지 않으므로 CPU 참조판(0)과 같게
  CK(cudaMemset(iv_, 0, sizeof(int) * NUM_I * (size_t)N));
  Soa s{f_, iv_, rng_, N};
  if (stage >= kStageBeh) init_kernel_beh<<<(N + 127) / 128, 128>>>(s, seed, ss_dev, bcurr_);
  else if (stage >= 2) init_kernel<true><<<(N + 127) / 128, 128>>>(s, seed, stage);
  else init_kernel<false><<<(N + 127) / 128, 128>>>(s, seed, stage);
  CK(cudaGetLastError());
}
DeviceEnv::~DeviceEnv() {
  cudaFree(f_); cudaFree(iv_); cudaFree(rng_); cudaFree(bcurr_); cudaFree(ctl_);
  if (ctl_h_) {
    for (void* e : ctl_ev_) if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
    cudaFreeHost(ctl_h_);
  }
}

void DeviceEnv::set_dynamic(uint32_t families) {
  fam_ = families | stage_family(stage_);
  if ((fam_ & kFamBeh) && !ss_) { std::fprintf(stderr, "DeviceEnv: BEHAVIOR family needs a device SceneSet\n"); std::abort(); }
  dyn_ = true;
  if (!ctl_h_) {
    CK(cudaHostAlloc(&ctl_h_, sizeof(EnvCtl) * 8, cudaHostAllocDefault));
    for (auto& e : ctl_ev_) { cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)); e = ev; }
  }
}

int DeviceEnv::request_stage(int stage, uint64_t seed) {
  if (!dyn_ || !(fam_ & stage_family(stage))) return -1;
  const int k = ctl_slot_++ % 8;
  cudaEvent_t ev = static_cast<cudaEvent_t>(ctl_ev_[k]);
  if (cudaEventQuery(ev) == cudaErrorNotReady) CK(cudaEventSynchronize(ev));   // 8 번 전 복사가 아직이면(드묾)
  ctl_h_[k] = EnvCtl{stage_, stage, (unsigned long long)seed};
  // pend·seed 만 덮어씀(stage 는 장치 값 그대로 — apply 가 바꿈)
  CK(cudaMemcpyAsync(&ctl_->pend, &ctl_h_[k].pend, sizeof(EnvCtl) - offsetof(EnvCtl, pend), cudaMemcpyHostToDevice, 0));
  CK(cudaEventRecord(ev, 0));
  stage_ = stage;
  return 0;
}

void DeviceEnv::apply() {
  Soa s{f_, iv_, rng_, N_};
  env_apply_k<<<(N_ + 127) / 128, 128>>>(s, ctl_, ss_, bcurr_src_ ? bcurr_src_ : bcurr_);
  env_commit_k<<<1, 1>>>(ctl_);
}

void DeviceEnv::step(const float* act, float* obs, float* rew, int* done, int bug) {
  Soa s{f_, iv_, rng_, N_};
  if (dyn_) {   // 장치 단계: 무리마다 띄우고 커널이 단계를 읽어 맞지 않으면 바로 끝남
    const unsigned g = (N_ + 127) / 128;
    if (fam_ & kFamBox) step_kernel<false><<<g, 128>>>(s, act, obs, rew, done, 0, arm_free_ ? 1 : 0, bug, ctl_);
    if (fam_ & kFamA2) step_kernel_a2<<<g, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, ctl_);
    if (fam_ & kFamBeh) {
      step_kernel_beh<<<g, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, ss_, bcurr_src_ ? bcurr_src_ : bcurr_, nav_, ctl_);
      step_kernel_pnp<<<g, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, ss_, bcurr_src_ ? bcurr_src_ : bcurr_, nav_, ctl_);
    }
    return;
  }
  if (stage_ >= kStageBeh) {
    step_kernel_beh<<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, ss_, bcurr_src_ ? bcurr_src_ : bcurr_, nav_, nullptr);
    step_kernel_pnp<<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, ss_, bcurr_src_ ? bcurr_src_ : bcurr_, nav_, nullptr);
  }
  else if (stage_ >= 2) step_kernel_a2<<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, arm_free_ ? 1 : 0, bug, nullptr);
  else step_kernel<false><<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, stage_, arm_free_ ? 1 : 0, bug, nullptr);
}

void DeviceEnv::download(std::vector<float>& f, std::vector<int>& iv, std::vector<uint64_t>& rng) const {
  f.resize((size_t)NUM_F * N_); iv.resize((size_t)NUM_I * N_); rng.resize(N_);
  CK(cudaMemcpy(f.data(), f_, sizeof(float) * f.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(iv.data(), iv_, sizeof(int) * iv.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rng.data(), rng_, sizeof(uint64_t) * rng.size(), cudaMemcpyDeviceToHost));
}

}  // namespace env
