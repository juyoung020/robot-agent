// 환경 커널: 스레드 하나 = 판 하나. 호스트 쪽은 장치 메모리와 실행만(상태는 제자리 갱신 — CUDA 그래프로 잡을 수 있게).
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <chrono>

#include "bscene_host.h"
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

// 대본 교사(E6): 앞(판마다) → 계획(요청한 판만 — 장치 목록, 고정 격자가 목록을 나눠 돎, 스레드마다 작업 메모리 한 칸) → 행동(판마다).
// 목록 차례는 원자 더하기라 매번 다르지만 판끼리 서로 안 보므로 결과는 같다(CPU 참조판은 판 번호 차례)
#ifdef ENV_PROF
__device__ unsigned long long g_tprof[64][3];   // 계획 요청 비트마다 [수, 사이클 합, 최대]
#endif
constexpr int kTeachSlots = 256;   // 계획 커널 블록 수 = 작업 메모리 칸 수(T_SCR 42 KB × 256 = 11 MB). 판 하나 = 블록 하나(T_CHUNK 레인이 후보·BFS 행을 나눔)
__global__ void __launch_bounds__(128) teacher_pre_k(Soa s, TBuf tb, const bsc::SceneSet* ss, bsc::NavFb fb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N && teacher_pre(s, tb, i, *ss, fb)) {
    const int k = atomicAdd(tb.list, 1);
    tb.list[1 + k] = i;
  }
}
__global__ void __launch_bounds__(T_CHUNK) teacher_plan_k(Soa s, TBuf tb, const bsc::SceneSet* ss) {
  __shared__ int sh[16];
  const WCtx w{(int)threadIdx.x, (int)blockDim.x, sh};
  const int warp = blockIdx.x, nwarp = gridDim.x;   // 블록 = 작업 메모리 칸
  const int n = tb.list[0];
  for (int k = warp; k < n; k += nwarp) {
#ifdef ENV_PROF
    const int need = tb.iv[TI_NEED * s.N + tb.list[1 + k]] & 63;
    const long long c0 = clock64();
#endif
    teacher_plan(s, tb, tb.list[1 + k], *ss, tb.scr + (size_t)warp * T_SCR, w);
    __syncthreads();   // 작업 메모리를 다음 판이 다시 씀
#ifdef ENV_PROF
    if (w.lane == 0) { const unsigned long long dc = (unsigned long long)(clock64() - c0); atomicAdd(&g_tprof[need][0], 1ull); atomicAdd(&g_tprof[need][1], dc); atomicMax(&g_tprof[need][2], dc); }
#endif
  }
}
__global__ void __launch_bounds__(128) teacher_act_k(Soa s, TBuf tb, const bsc::SceneSet* ss, float* act) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) teacher_act(s, tb, i, *ss, act);
  if (i == 0) tb.list[0] = 0;   // 다음 스텝 목록(계획 커널은 끝남)
}
void DeviceEnv::teacher_pre(const bsc::NavFb& fb) const { teacher_pre_k<<<(N_ + 127) / 128, 128>>>(Soa{f_, iv_, rng_, N_}, tb_, ss_, fb); }
void DeviceEnv::teacher_plan() const { teacher_plan_k<<<tb_.nslot, T_CHUNK>>>(Soa{f_, iv_, rng_, N_}, tb_, ss_); }
void DeviceEnv::teacher_act(float* act) const { teacher_act_k<<<(N_ + 127) / 128, 128>>>(Soa{f_, iv_, rng_, N_}, tb_, ss_, act); }
void DeviceEnv::teacher(float* act) const {
  if (!ss_ || !tb_.f) return;
  teacher_pre(nav_);
  teacher_plan();
  teacher_act(act);
}

// 잡기 가능 표(짝마다 스레드 하나)
__global__ void __launch_bounds__(128) feas_k(const bsc::SceneSet* ss, const int* idx, int n, FeasOut* out) {   // 짝 하나 = 워프 하나
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const WCtx w{(int)(threadIdx.x & 31), 32, nullptr};
  if (k >= n) return;   // 워프 전체가 같은 값
  FeasOut o;
  feas_entry(*ss, idx[k], o, w);
  if (w.lane == 0) out[k] = o;
}

// 교사 정적 점유 표(짝 하나 = 워프 하나): 칸 성분·과제 물체·정적 상자를 미리 칠함(teacher.h tch_occ_static)
__global__ void __launch_bounds__(128) tocc_k(const bsc::SceneSet* ss, const int* idx, int n, uint64_t* out) {
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const WCtx w{(int)(threadIdx.x & 31), 32, nullptr};
  if (k >= n) return;
  TR* o = reinterpret_cast<TR*>(out + (size_t)k * 4 * bsc::WIN);   // 표 번호 = 물체 짝 차례 k
  for (int r = w.lane; r < 2 * bsc::WIN; r += 32) o[r] = TR{0ull, 0ull};
  __syncwarp();
  tch_occ_static(*ss, idx[k], o, w);
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
void tseg_read(unsigned long long out[16][2]) { cudaMemcpyFromSymbol(out, g_tseg, sizeof(unsigned long long) * 32); }
void tact_read(unsigned long long out[16][3]) { cudaMemcpyFromSymbol(out, g_tact, sizeof(unsigned long long) * 48); }
void tdbg_read(unsigned long long out[32]) { cudaMemcpyFromSymbol(out, g_tdbgd, sizeof(unsigned long long) * 32); }
void tkc_read(unsigned long long out[3][5]) { cudaMemcpyFromSymbol(out, g_tkc, sizeof(unsigned long long) * 15); }
void tbin_read(unsigned long long out[3][8]) { cudaMemcpyFromSymbol(out, g_tbin, sizeof(unsigned long long) * 24); }
void tcause_read(unsigned long long out[2][16]) { cudaMemcpyFromSymbol(out, g_tcause, sizeof(unsigned long long) * 32); }
void tprof_read(unsigned long long out[64][3]) { cudaMemcpyFromSymbol(out, g_tprof, sizeof(unsigned long long) * 64 * 3); }
void env_prof_read(unsigned long long out[8]) { cudaMemcpyFromSymbol(out, g_env_prof, sizeof(unsigned long long) * 8); }
void env_prof_reset() {
  static const unsigned long long z[64 * 3] = {};
  cudaMemcpyToSymbol(g_env_prof, z, sizeof(unsigned long long) * 8);
  cudaMemcpyToSymbol(g_tprof, z, sizeof(unsigned long long) * 64 * 3);
  cudaMemcpyToSymbol(g_tseg, z, sizeof(unsigned long long) * 32);
  cudaMemcpyToSymbol(g_tcause, z, sizeof(unsigned long long) * 32);
  cudaMemcpyToSymbol(g_tbin, z, sizeof(unsigned long long) * 24);
  cudaMemcpyToSymbol(g_tact, z, sizeof(unsigned long long) * 48);
  cudaMemcpyToSymbol(g_tdbgd, z, sizeof(unsigned long long) * 32);
  cudaMemcpyToSymbol(g_tkc, z, sizeof(unsigned long long) * 15);
}
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
  if (ss_dev) {   // 대본 교사 버퍼(장면 묶음이 있을 때 — 그래프 잡기 전에 잡아 둠)
    tb_.N = N;
    tb_.nslot = N < kTeachSlots ? N : kTeachSlots;   // 블록 수
    CK(cudaMalloc(&tb_.f, sizeof(float) * NTF * (size_t)N));
    CK(cudaMalloc(&tb_.iv, sizeof(int) * NTI * (size_t)N));
    CK(cudaMalloc(&tb_.list, sizeof(int) * ((size_t)N + 1)));
    CK(cudaMalloc(&tb_.scr, (size_t)T_SCR * tb_.nslot));
    CK(cudaMemset(tb_.f, 0, sizeof(float) * NTF * (size_t)N));
    CK(cudaMemset(tb_.iv, 0, sizeof(int) * NTI * (size_t)N));
    CK(cudaMemset(tb_.list, 0, sizeof(int) * ((size_t)N + 1)));
    CK(cudaMalloc(&tb_.rb, sizeof(uint32_t) * T_RBW * (size_t)N));
    CK(cudaMemset(tb_.rb, 0, sizeof(uint32_t) * T_RBW * (size_t)N));
  }
  Soa s{f_, iv_, rng_, N};
  if (stage >= kStageBeh) init_kernel_beh<<<(N + 127) / 128, 128>>>(s, seed, ss_dev, bcurr_);
  else if (stage >= 2) init_kernel<true><<<(N + 127) / 128, 128>>>(s, seed, stage);
  else init_kernel<false><<<(N + 127) / 128, 128>>>(s, seed, stage);
  CK(cudaGetLastError());
}
DeviceEnv::~DeviceEnv() {
  cudaFree(f_); cudaFree(iv_); cudaFree(rng_); cudaFree(bcurr_); cudaFree(ctl_);
  if (tb_.f) { cudaFree(tb_.f); cudaFree(tb_.iv); cudaFree(tb_.list); cudaFree(tb_.scr); cudaFree(tb_.rb); }
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

double pnp_feasibility(bsc::SceneBuild& b, bool quiet) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<int> idx;
  for (size_t k = 0; k < b.ent.size(); ++k) if (b.ent[k].list == bsc::L_OBJ) idx.push_back((int)k);
  const int n = (int)idx.size();
  int* d_idx = nullptr;
  FeasOut* d_out = nullptr;
  CK(cudaMalloc(&d_idx, sizeof(int) * (n > 0 ? n : 1)));
  CK(cudaMalloc(&d_out, sizeof(FeasOut) * (n > 0 ? n : 1)));
  if (n) CK(cudaMemcpy(d_idx, idx.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  size_t st0 = 0;
  CK(cudaDeviceGetLimit(&st0, cudaLimitStackSize));
  if (n) feas_k<<<(n * 32 + 127) / 128, 128>>>(b.dev, d_idx, n, d_out);
  CK(cudaGetLastError());
  std::vector<FeasOut> out(n);
  if (n) CK(cudaMemcpy(out.data(), d_out, sizeof(FeasOut) * n, cudaMemcpyDeviceToHost));
  cudaFree(d_idx); cudaFree(d_out);
  for (int k = 0; k < n; ++k) {
    bsc::Entry& e = b.ent[idx[k]];
    e.feas = out[k].feas;
    for (int a = 0; a < 4; ++a) { e.gst4[a] = out[k].gst4[a]; e.gst[a] = out[k].gst[a]; e.pst5[a] = out[k].pst5[a]; e.pst6[a] = out[k].pst6[a]; e.grel[a] = out[k].grel[a]; }
  }
  b.host.has_feas = 1;
  // 교사 정적 점유 표(짝마다 4 KB — 계획마다 칠하던 것): 장치에서 만들고 호스트에도(CPU 참조판)
  uint64_t* d_tocc = nullptr;
  int* d_toccix = nullptr;
  const size_t tocc_n = (size_t)n * 4 * bsc::WIN;
  if (!b.host.tocc) {
    CK(cudaMalloc(&d_tocc, sizeof(uint64_t) * (tocc_n ? tocc_n : 1)));
    b.toccix.assign(b.ent.size(), -1);
    for (int k = 0; k < n; ++k) b.toccix[idx[k]] = k;
    CK(cudaMalloc(&d_toccix, sizeof(int) * (b.ent.empty() ? 1 : b.ent.size())));
    if (!b.ent.empty()) CK(cudaMemcpy(d_toccix, b.toccix.data(), sizeof(int) * b.ent.size(), cudaMemcpyHostToDevice));
    if (n) {
      int* d_idx2 = nullptr;
      CK(cudaMalloc(&d_idx2, sizeof(int) * n));
      CK(cudaMemcpy(d_idx2, idx.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
      tocc_k<<<(n * 32 + 127) / 128, 128>>>(b.dev, d_idx2, n, d_tocc);
      CK(cudaGetLastError());
      cudaFree(d_idx2);
    }
    b.tocc.resize(tocc_n);
    if (tocc_n) CK(cudaMemcpy(b.tocc.data(), d_tocc, sizeof(uint64_t) * tocc_n, cudaMemcpyDeviceToHost));
    b.dbuf.push_back(d_tocc);
    b.dbuf.push_back(d_toccix);
    b.dev_bytes += sizeof(uint64_t) * tocc_n + sizeof(int) * b.ent.size();
    b.host.tocc = b.tocc.data();
    b.host.toccix = b.toccix.data();
  }
  bsc::SceneSet D;
  CK(cudaMemcpy(&D, b.dev, sizeof D, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(const_cast<bsc::Entry*>(D.ent), b.ent.data(), sizeof(bsc::Entry) * b.ent.size(), cudaMemcpyHostToDevice));
  D.has_feas = 1;
  if (d_tocc && !std::getenv("TEACH_NO_TOCC")) { D.tocc = d_tocc; D.toccix = d_toccix; }   // TEACH_NO_TOCC: 장치는 예전처럼 칠함(같은 비트 확인용)
  CK(cudaMemcpy(b.dev, &D, sizeof D, cudaMemcpyHostToDevice));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!quiet) {
    int ng = 0, n5 = 0, n6 = 0;
    for (int k = 0; k < n; ++k) { ng += out[k].feas & 1; n5 += (out[k].feas >> 1) & 1; n6 += (out[k].feas >> 2) & 1; }
    std::fprintf(stderr, "pnp_feasibility: %d pick-and-place entries, B4 graspable %d, B5 placeable %d, B6 (grasp + place) %d — %.1f s\n", n, ng, n5, n6, secs);
  }
  return secs;
}

}  // namespace env
