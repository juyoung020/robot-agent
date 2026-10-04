// BC 학생 학습기 본체(G5): 롤아웃(교사·학생 앞 → 행동 고르기·기록 → K1 환경 → K2 지도) 과 갱신(K 번 [모으기 → 앞 → MSE(K5) → 뒤 → Adam(K9)]).
// 그래프 둘. 누가 움직이나·기록하나(장치 Mode), 자료 쓰기 자리·표본 수(장치 Data), 미니배치 난수 열쇠(TrainState.iter)는 모두 장치 값 → 호스트 동기 0.
// 신경망 커널(손 BF16 GEMM, 칸 MLP 묶음, dW 조각 합, Adam)은 training/RL/network 것을 그대로 쓴다(소스를 읽기만).
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bc.h"

namespace bc {

using namespace net;

#define BCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

template <class T_>
T_* Bc::alloc(size_t n) {
  void* p = nullptr;
  BCK(cudaMalloc(&p, sizeof(T_) * n + 16));
  BCK(cudaMemset(p, 0, sizeof(T_) * n + 16));
  allocs.push_back(p);
  dev_bytes += sizeof(T_) * n;
  return reinterpret_cast<T_*>(p);
}

constexpr int AS_L = 16, AS_E = 8;   // 판(행) 하나를 16 레인이, 블록에 8 판
constexpr int TAB_Q = 6, TAB_C = 10;

// ---- 관측 모으기(롤아웃: 판 e 그대로). role 0 = 교사(goal_mode 1), 1 = 학생 ----
__global__ void __launch_bounds__(AS_L * AS_E) assemble_k(const float* obs_col, const gmap::MapTok* tok, int N, int use_map, int role, int student_goal,
                                                          uint16_t* x0, uint16_t* sin, uint32_t* mask) {
  const int e = blockIdx.x * AS_E + threadIdx.x / AS_L, lane = threadIdx.x % AS_L;
  if (e >= N) return;   // N 은 8 의 배수 → 같은 워프의 두 판은 함께 돌아감(__syncwarp 안전)
  uint32_t mk;
  if (role == 0) mk = obsv::assemble(obs_col, N, e, tok[e], x0 + (size_t)e * X0_W, sin + (size_t)e * KSLOT * SLOT_IN, use_map, 1, lane, AS_L);
  else mk = assemble_student(obs_col, N, e, tok[e], x0 + (size_t)e * X0_W, sin + (size_t)e * KSLOT * SLOT_IN, use_map, student_goal, lane, AS_L);
  if (lane == 0) mask[e] = mk;
}

// ---- 행동 고르기 + 기록(5.3): 워프 하나 = 판 하나 ----
struct ActP {
  const float *meanT, *meanS; const Mode* md; const Data* dd; int t, N; long long cap;
  const float* obs_col; const gmap::MapTok* tok; env::Soa s; const float* met_init; const int* cur_len;
  float* act_env; float* dis; uint16_t* d_obs; gmap::MapTok* d_tok; float* d_lab; uint32_t* d_meta; RenderState* d_rs;
};
__global__ void __launch_bounds__(256) act_rec_k(ActP p) {
  const int i = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (i >= p.N) return;
  const int actor = p.md->actor, rec = p.md->record;
  if (l < N_ACT) {
    const int k = l;
    float a = 0.f;
    if (k < N_LAB) a = actor ? p.meanS[(size_t)i * N_ACT + k] : p.meanT[(size_t)i * N_ACT + k];
    p.act_env[(size_t)k * p.N + i] = a;   // 환경이 ±1 로 자름. 학습하지 않는 행동 6 개는 0
  }
  float lab[N_LAB];
  for (int k = 0; k < N_LAB; ++k) lab[k] = fminf(fmaxf(p.meanT[(size_t)i * N_ACT + k], -1.f), 1.f);
  if (l == 0) {
    float d = 0.f;
    for (int k = 0; k < N_LAB; ++k) { const float e = p.meanS[(size_t)i * N_ACT + k] - lab[k]; d = d + e * e; }
    p.dis[(size_t)p.t * p.N + i] = d;
  }
  if (!rec) return;
  const long long slot = (p.dd->cursor + (long long)p.t * p.N + i) % p.cap;
  for (int c = l; c < env::N_OBS; c += 32) p.d_obs[slot * env::N_OBS + c] = obs_rec(p.obs_col[(size_t)c * p.N + i]);
  {
    const uint4* src = reinterpret_cast<const uint4*>(p.tok + i);
    uint4* dst = reinterpret_cast<uint4*>(p.d_tok + slot);
    for (int q = l; q < (int)(sizeof(gmap::MapTok) / 16); q += 32) dst[q] = src[q];
  }
  if (l == 0) {
    for (int k = 0; k < N_LAB; ++k) p.d_lab[slot * N_LAB + k] = lab[k];
    const int len = p.cur_len[i] < 65535 ? p.cur_len[i] : 65535;
    p.d_meta[slot] = (uint32_t)len | ((uint32_t)((int)p.met_init[i] & 0xff) << 16) | ((uint32_t)actor << 24);
    if (p.d_rs) {
      RenderState r;
      render_state(p.s, i, r);
      p.d_rs[slot] = r;
    }
  }
}

// ---- 처음 지도(완성도)별 에피소드 통계: 환경 스텝 뒤·지도 스텝 앞(ppo epstat_k 와 같은 부호·배치) ----
__global__ void epstat_k(const int* done, const float* met_init, int N, int* cur_len, int* it_stat, int* it_out, unsigned long long* tab) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int len = cur_len[i] + 1;
  const int d = done[i];
  if (d != env::kRunning) {
    const int code = (int)met_init[i];
    int st = (code >> 5) & 3, c = code & 15;
    st = st > 2 ? 2 : st;
    c = c > TAB_C - 1 ? TAB_C - 1 : c;
    const int g = (code >> 4) & 1;
    atomicAdd(&it_stat[st * 3], 1);
    if (d == env::kSuccess) atomicAdd(&it_stat[st * 3 + 1], 1);
    if (d == env::kCollision) atomicAdd(&it_stat[st * 3 + 2], 1);
    atomicAdd(&it_out[d == env::kSuccess ? 0 : d == env::kCollision ? 1 : 2], 1);
    unsigned long long* q = tab + (size_t)((st * 2 + g) * TAB_C + c) * TAB_Q;
    atomicAdd(q, 1ull);
    atomicAdd(q + (d == env::kSuccess ? 1 : d == env::kCollision ? 2 : 3), 1ull);
    atomicAdd(q + 4, (unsigned long long)len);
    if (d == env::kSuccess) atomicAdd(q + 5, (unsigned long long)len);
    len = 0;
  }
  cur_len[i] = len;
}

// ---- 블록 안 고정 나무 합 ----
template <int NQ>
__device__ void bsum(float (&q)[NQ], float* sh) {
  const int t = threadIdx.x, nt = blockDim.x;
  for (int k = 0; k < NQ; ++k) sh[k * nt + t] = q[k];
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (t < s)
      for (int k = 0; k < NQ; ++k) sh[k * nt + t] = sh[k * nt + t] + sh[k * nt + t + s];
    __syncthreads();
  }
  for (int k = 0; k < NQ; ++k) q[k] = sh[k * nt];
}

// ---- 롤아웃 끝: 어긋남 평균(고정 순서), 에피소드 수, 자료 자리 옮기기, 기록 ----
__global__ void __launch_bounds__(256) rollout_end_k(const float* dis, long long n, int N, int T, const Mode* md, Data* dd, long long cap, int* it_stat, int* it_out,
                                                     const TrainState* ts, int ring, BcLog* out) {
  __shared__ float sh[256];
  float q[1] = {0.f};
  for (long long j = threadIdx.x; j < n; j += 256) q[0] = q[0] + dis[j];
  bsum<1>(q, sh);
  if (threadIdx.x != 0) return;
  BcLog L;
  memset(&L, 0, sizeof L);
  L.kind = 0;
  L.actor = md->actor;
  L.record = md->record;
  if (md->record) {
    dd->cursor = (dd->cursor + (long long)N * T) % cap;
    const long long c = dd->count + (long long)N * T;
    dd->count = c < cap ? c : cap;
  }
  dd->rollouts = dd->rollouts + 1;
  L.count = dd->count;
  L.adam_t = ts->adam_t;
  L.disagree = q[0] / (float)n;
  const float ne = (float)(it_out[0] + it_out[1] + it_out[2]);
  L.n_eps = ne;
  const float inv = ne > 0.f ? 1.f / ne : 0.f;
  L.succ = it_out[0] * inv;
  L.coll = it_out[1] * inv;
  L.tout = it_out[2] * inv;
  it_out[0] = it_out[1] = it_out[2] = 0;
  for (int k = 0; k < 3; ++k) {
    const int nk = it_stat[k * 3];
    L.n_c[k] = (float)nk;
    L.s_c[k] = nk > 0 ? (float)it_stat[k * 3 + 1] / (float)nk : 0.f;
    L.k_c[k] = nk > 0 ? (float)it_stat[k * 3 + 2] / (float)nk : 0.f;
    it_stat[k * 3] = it_stat[k * 3 + 1] = it_stat[k * 3 + 2] = 0;
  }
  dd->pad = dd->pad + 1;
  L.seq = dd->pad;
  out[(L.seq - 1) % ring] = L;
}

// ---- 미니배치 모으기: 자료 버퍼에서 균등 무작위(복원 추출, 장치 열쇠 = (씨앗, 갱신 스텝, 행)) → 학생 입력 ----
struct GatherP {
  const uint16_t* d_obs; const gmap::MapTok* d_tok; const float* d_lab; const Data* dd; const TrainState* ts;
  int MB, use_map, student_goal; uint64_t seed;
  uint16_t* x0; uint16_t* sin; uint32_t* mask; float* lab;
};
__global__ void __launch_bounds__(AS_L * AS_E) gather_k(GatherP g) {
  __shared__ float so[AS_E][env::N_OBS];
  const int rl = threadIdx.x / AS_L, lane = threadIdx.x % AS_L;
  const int r = blockIdx.x * AS_E + rl;
  long long idx = 0;
  if (r < g.MB) {
    const long long cnt = g.dd->count;
    idx = cnt > 0 ? (long long)(hash4(g.seed, (uint64_t)g.ts->iter, (uint64_t)r, 0x42434d42ull) % (uint64_t)cnt) : 0;
    for (int c = lane; c < env::N_OBS; c += AS_L) so[rl][c] = bf2f(g.d_obs[idx * env::N_OBS + c]);
  }
  __syncthreads();
  if (r >= g.MB) return;   // MB 는 8 의 배수 → 워프 단위로 같이 빠짐
  const uint32_t mk = assemble_student(so[rl], 1, 0, g.d_tok[idx], g.x0 + (size_t)r * X0_W, g.sin + (size_t)r * KSLOT * SLOT_IN, g.use_map, g.student_goal,
                                       lane, AS_L);
  if (lane < N_LAB) g.lab[(size_t)r * N_LAB + lane] = g.d_lab[idx * N_LAB + lane];
  if (lane == 0) g.mask[r] = mk;
}

// ---- BC 손실(K5): L = (1/M) Σ_r Σ_{k<2} (μ − 라벨)², dZ_A4 = 2(μ − 라벨)/M (bf16), 나머지 행동 0 ----
constexpr int LOSS_T = 256;
__global__ void __launch_bounds__(LOSS_T) bc_loss_k(const float* mean, const float* lab, int M, int bug, uint16_t* dz, float* part) {
  __shared__ float sh[LOSS_T];
  const int r = blockIdx.x * LOSS_T + threadIdx.x;
  float q[1] = {0.f};
  if (r < M) {
    const float inv = 1.f / (float)M, two = bug == 1 ? 1.f : 2.f;
    for (int k = 0; k < N_ACT; ++k) {
      float g = 0.f;
      if (k < N_LAB) {
        const float d = mean[(size_t)r * N_ACT + k] - lab[(size_t)r * N_LAB + k];
        q[0] = q[0] + d * d;
        g = two * d * inv;
      }
      dz[(size_t)r * N_ACT + k] = f2bf(g);
    }
  }
  bsum<1>(q, sh);
  if (threadIdx.x == 0) part[blockIdx.x] = q[0];
}
__global__ void __launch_bounds__(256) bc_loss_reduce_k(const float* part, int nb, int M, TrainState* ts) {
  __shared__ float sh[256];
  float q[1] = {0.f};
  for (int b = threadIdx.x; b < nb; b += 256) q[0] = q[0] + part[b];
  bsum<1>(q, sh);
  if (threadIdx.x != 0) return;
  ts->s_pg = ts->s_pg + q[0] / (float)M;
  ts->n_mb = ts->n_mb + 1;
  ts->iter = ts->iter + 1;   // 다음 스텝의 미니배치 열쇠
}
__global__ void update_end_k(TrainState* ts, Data* dd, int ring, BcLog* out) {
  BcLog L;
  memset(&L, 0, sizeof L);
  L.kind = 1;
  L.count = dd->count;
  L.adam_t = ts->adam_t;
  L.loss = ts->n_mb > 0 ? ts->s_pg / (float)ts->n_mb : 0.f;
  L.grad_norm = ts->gnorm;
  ts->s_pg = 0.f;
  ts->n_mb = 0;
  dd->pad = dd->pad + 1;
  L.seq = dd->pad;
  out[(L.seq - 1) % ring] = L;
}

__global__ void set_col_k(uint16_t* buf, long long rows, int ld, int col, uint16_t v) {
  const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) buf[r * ld + col] = v;
}

// ---------------------------------------------------------------------------------------------------------------------
static void alloc_net(Bc& b, NetBufs& n, int M, bool grad) {
  n.M = M;
  const size_t MS = (size_t)M * KSLOT;
  n.x0 = b.alloc<uint16_t>((size_t)M * X0_W);
  n.sin = b.alloc<uint16_t>(MS * SLOT_IN);
  n.mask = b.alloc<uint32_t>(M);
  n.s1o = b.alloc<uint16_t>(MS * kLayers[L_S1].ldo);
  n.s2o = b.alloc<uint16_t>(MS * kLayers[L_S2].ldo);
  n.amax = b.alloc<uint8_t>((size_t)M * S_H);
  n.ho[L_S1] = n.s1o;
  n.ho[L_S2] = n.s2o;
  for (int l : {L_A1, L_A2, L_A3}) n.ho[l] = b.alloc<uint16_t>((size_t)M * kLayers[l].ldo);
  n.mean = b.alloc<float>((size_t)M * N_ACT);
  for (int l : {L_S1, L_A1, L_A2, L_A3}) {   // 1 칸(다음 층 편향 입력)
    const LayerDesc& L = kLayers[l];
    const long long rows = l <= L_S2 ? (long long)MS : (long long)M;
    set_col_k<<<(unsigned)((rows + 255) / 256), 256>>>(n.ho[l], rows, L.ldo, L.N, (uint16_t)0x3f80);
  }
  if (grad) {
    for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4}) n.dz[l] = b.alloc<uint16_t>((l <= L_S2 ? MS : (size_t)M) * kLayers[l].N);
    n.dpool = b.alloc<float>((size_t)M * POOL_W);
  }
}

Bc::Bc(const BcConfig& c) : cfg(c) {
  lay = param_layout();
  N = cfg.n_env;
  T = cfg.horizon;
  MB = cfg.mb;
  if (N % 8 || MB % 8 || T % 2 || cfg.dw_chunk % 64) { std::fprintf(stderr, "bc: N, mb must be multiples of 8, horizon even, dw_chunk multiple of 64\n"); std::abort(); }
  cap = cfg.cap;
  ah = AdamHyper{cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.max_grad_norm};

  obs_col = alloc<float>((size_t)2 * env::N_OBS * N);
  act_env = alloc<float>((size_t)N_ACT * N);
  rew = alloc<float>(N);
  done = alloc<int>(N);
  cur_len = alloc<int>(N);
  it_stat = alloc<int>(9);
  it_out = alloc<int>(3);
  tab = alloc<unsigned long long>((size_t)3 * 2 * TAB_C * TAB_Q);
  dis = alloc<float>((size_t)T * N);
  curr_d = alloc<gmap::MapCurr>(1);
  mode_d = alloc<Mode>(1);
  data_d = alloc<Data>(1);
  BCK(cudaHostAlloc(&stage_h, 64 * 16, cudaHostAllocDefault));
  for (auto& e : stage_ev) BCK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  {
    const gmap::MapCurr c0{cfg.map_p0, cfg.map_p1, cfg.map_kmin, cfg.map_kmax, cfg.map_reveal_r, 0};
    BCK(cudaMemcpy(curr_d, &c0, sizeof c0, cudaMemcpyHostToDevice));
  }

  d_obs = alloc<uint16_t>((size_t)cap * env::N_OBS);
  d_tok = alloc<gmap::MapTok>((size_t)cap);
  d_lab = alloc<float>((size_t)cap * N_LAB);
  d_meta = alloc<uint32_t>((size_t)cap);
  if (cfg.store_render) d_rs = alloc<RenderState>((size_t)cap);

  PbT = alloc<uint16_t>(lay.total);
  P = alloc<float>(lay.total);
  G = alloc<float>(lay.total);
  Am = alloc<float>(lay.total);
  Av = alloc<float>(lay.total);
  Pb = alloc<uint16_t>(lay.total);
  alloc_net(*this, nt, N, false);
  alloc_net(*this, ns, N > MB ? N : MB, true);
  lab_mb = alloc<float>((size_t)MB * N_LAB);
  for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4}) {
    const int rows = l <= L_S2 ? MB * KSLOT : MB;
    ws[l] = alloc<float>((size_t)dw_splits(rows, cfg.dw_chunk) * kLayers[l].N * kLayers[l].K);
  }
  gn_part = alloc<float>(GN_BLOCKS);
  loss_part = alloc<float>((size_t)(MB + LOSS_T - 1) / LOSS_T);
  ts = alloc<TrainState>(1);

  // 학생 초기화(호스트, 결정적): ppo 학습기와 같은 규칙(균등 ±gain·√(3/fan_in)), 정책 사슬과 칸 MLP 만 씀(가치 사슬 C 는 0 그대로 — 기울기 0)
  {
    std::vector<float> hp(lay.total, 0.f);
    uint64_t s = cfg.seed * 0x9E3779B97F4A7C15ull + 11;
    for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4}) {
      const LayerDesc& L = kLayers[l];
      const float a = L.gain * std::sqrt(3.f / (float)L.bias);
      for (int n = 0; n < L.N; ++n)
        for (int k = 0; k < L.bias; ++k) hp[lay.off[l] + (size_t)n * L.K + k] = dm::rand_range(s, -a, a);
    }
    BCK(cudaMemcpy(P, hp.data(), sizeof(float) * lay.total, cudaMemcpyHostToDevice));
    to_bf16(P, Pb, lay.total, 0);
    TrainState h{};
    h.lr = cfg.lr;
    BCK(cudaMemcpy(ts, &h, sizeof h, cudaMemcpyHostToDevice));
    const Mode m{0, 0, 0, 0};
    BCK(cudaMemcpy(mode_d, &m, sizeof m, cudaMemcpyHostToDevice));
  }

  const int R = cfg.log_ring > 1 ? cfg.log_ring : 16;
  cfg.log_ring = R;
  BCK(cudaHostAlloc(&ring_h, sizeof(BcLog) * R, cudaHostAllocMapped));
  std::memset(ring_h, 0, sizeof(BcLog) * R);
  BCK(cudaHostGetDevicePointer(&ring_d, ring_h, 0));
  ev_a.resize(R); ev_b.resize(R);
  for (int k = 0; k < R; ++k) { BCK(cudaEventCreate(&ev_a[k])); BCK(cudaEventCreate(&ev_b[k])); }

  make_env(cfg.env_seed);
  BCK(cudaDeviceSynchronize());
  if (cfg.use_graphs) capture();
}

Bc::~Bc() {
  cudaDeviceSynchronize();
  if (g_roll) cudaGraphExecDestroy(g_roll);
  if (g_upd) cudaGraphExecDestroy(g_upd);
  tok.reset(); map.reset(); env.reset();
  for (void* p : allocs) cudaFree(p);
  for (size_t k = 0; k < ev_a.size(); ++k) { cudaEventDestroy(ev_a[k]); cudaEventDestroy(ev_b[k]); }
  for (auto& e : stage_ev) if (e) cudaEventDestroy(e);
  if (ring_h) cudaFreeHost(ring_h);
  if (stage_h) cudaFreeHost(stage_h);
}

void Bc::make_env(uint64_t env_seed) {
  cfg.env_seed = env_seed;
  tok.reset(); map.reset(); env.reset();
  env = std::make_unique<env::DeviceEnv>(N, cfg.stage, env_seed * 1000003ull + 17ull + (uint64_t)cfg.stage);
  map = std::make_unique<gmap::DeviceMap>(N, env_seed * 7919ull + 3ull + (uint64_t)cfg.stage);
  tok = std::make_unique<gmap::TokenRecorder>(N, 2);
  BCK(cudaMemset(obs_col, 0, sizeof(float) * 2 * env::N_OBS * N));
  BCK(cudaMemset(tok->at(0), 0, sizeof(gmap::MapTok) * N));
  BCK(cudaMemset(tok->at(1), 0, sizeof(gmap::MapTok) * N));
  BCK(cudaMemset(cur_len, 0, sizeof(int) * N));
  BCK(cudaMemset(it_stat, 0, sizeof(int) * 9));
  BCK(cudaMemset(it_out, 0, sizeof(int) * 3));
}

void Bc::set_dev(void* dst, const void* src, size_t n) {
  if (n > 64) std::abort();
  const int k = stage_slot++ % 16;
  if (cudaEventQuery(stage_ev[k]) == cudaErrorNotReady) BCK(cudaEventSynchronize(stage_ev[k]));   // 16 번 전 복사가 아직이면(드묾)
  std::memcpy(stage_h + 64 * k, src, n);
  BCK(cudaMemcpyAsync(dst, stage_h + 64 * k, n, cudaMemcpyHostToDevice, 0));
  BCK(cudaEventRecord(stage_ev[k], 0));
}

static cudaGraphExec_t capture_one(Bc* b, void (Bc::*body)()) {
  cudaGraph_t g;
  BCK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
  (b->*body)();
  BCK(cudaStreamEndCapture(cudaStreamPerThread, &g));
  cudaGraphExec_t ex;
  BCK(cudaGraphInstantiate(&ex, g, 0));
  size_t nn = 0;
  cudaGraphGetNodes(g, nullptr, &nn);
  std::fprintf(stderr, "bc: captured graph with %zu nodes\n", nn);
  cudaGraphDestroy(g);
  return ex;
}
void Bc::capture() {
  if (g_roll) { cudaGraphExecDestroy(g_roll); g_roll = nullptr; }
  if (g_upd) { cudaGraphExecDestroy(g_upd); g_upd = nullptr; }
  g_roll = capture_one(this, &Bc::rollout_body);
  g_upd = capture_one(this, &Bc::update_body);
}

// ---- 신경망(정책 사슬만) ----
void Bc::forward(const uint16_t* Wb, NetBufs& b, int M) {
  auto Wl = [&](int l) { return Wb + lay.off[l]; };
  slot_fwd(b.sin, b.mask, Wl(L_S1), Wl(L_S2), M, b.s1o, b.s2o, b.x0, b.amax, 0);
  gemm_fwd(kLayers[L_A1], b.x0, M, Wl(L_A1), b.ho[L_A1], 0);
  gemm_fwd(kLayers[L_A2], b.ho[L_A1], M, Wl(L_A2), b.ho[L_A2], 0);
  gemm_fwd(kLayers[L_A3], b.ho[L_A2], M, Wl(L_A3), b.ho[L_A3], 0);
  gemm_fwd(kLayers[L_A4], b.ho[L_A3], M, Wl(L_A4), b.mean, 0);
}

void Bc::gather() {
  GatherP g{d_obs, d_tok, d_lab, data_d, ts, MB, cfg.use_map, cfg.student_goal, cfg.seed, ns.x0, ns.sin, ns.mask, lab_mb};
  gather_k<<<(MB + AS_E - 1) / AS_E, AS_L * AS_E>>>(g);
  BCK(cudaGetLastError());
}

void Bc::loss(int M) {
  const int nb = (M + LOSS_T - 1) / LOSS_T;
  bc_loss_k<<<nb, LOSS_T>>>(ns.mean, lab_mb, M, bug, ns.dz[L_A4], loss_part);
  bc_loss_reduce_k<<<1, 256>>>(loss_part, nb, M, ts);
  BCK(cudaGetLastError());
}

void Bc::backward(int M) {
  auto Wl = [&](int l) { return Pb + lay.off[l]; };
  const int ch = cfg.dw_chunk;
  NetBufs& b = ns;
  gemm_dw(kLayers[L_A4], b.dz[L_A4], b.ho[L_A3], M, ws[L_A4], ch, 0);
  gemm_dx_dact(kLayers[L_A4], b.dz[L_A4], M, Wl(L_A4), b.ho[L_A3], kLayers[L_A3].N, b.dz[L_A3], 0, 0);
  gemm_dw(kLayers[L_A3], b.dz[L_A3], b.ho[L_A2], M, ws[L_A3], ch, 0);
  gemm_dx_dact(kLayers[L_A3], b.dz[L_A3], M, Wl(L_A3), b.ho[L_A2], kLayers[L_A2].N, b.dz[L_A2], bug == 2 ? 1 : 0, 0);
  gemm_dw(kLayers[L_A2], b.dz[L_A2], b.ho[L_A1], M, ws[L_A2], ch, 0);
  gemm_dx_dact(kLayers[L_A2], b.dz[L_A2], M, Wl(L_A2), b.ho[L_A1], kLayers[L_A1].N, b.dz[L_A1], 0, 0);
  gemm_dw(kLayers[L_A1], b.dz[L_A1], b.x0, M, ws[L_A1], ch, 0);
  gemm_dx_pool(kLayers[L_A1], b.dz[L_A1], M, Wl(L_A1), b.dpool, false, 0);
  slot_bwd(b.dpool, b.s2o, b.s1o, b.sin, b.mask, b.amax, Wl(L_S2), M, ch, ws[L_S2], ws[L_S1], keep_slot_bufs ? b.dz[L_S2] : nullptr,
           keep_slot_bufs ? b.dz[L_S1] : nullptr, 0);
  DwJob jobs[6];
  int nj = 0;
  for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4}) {
    const int rows = l <= L_S2 ? M * KSLOT : M;
    jobs[nj++] = DwJob{ws[l], G + lay.off[l], kLayers[l].N * kLayers[l].K, dw_splits(rows, ch)};
  }
  dw_reduce(jobs, nj, 0);
}

void Bc::optimizer() { adam_step(P, G, Am, Av, Pb, lay.total, gn_part, ts, ah, 0); }

void Bc::update_step() {
  gather();
  forward(Pb, ns, MB);
  loss(MB);
  backward(MB);
  optimizer();
}

void Bc::update_body() {
  for (int k = 0; k < cfg.upd_steps; ++k) update_step();
  update_end_k<<<1, 1>>>(ts, data_d, cfg.log_ring, ring_d);
  BCK(cudaGetLastError());
}

void Bc::rollout_body() {
  for (int t = 0; t < T; ++t) rollout_step(t);
  rollout_end_k<<<1, 256>>>(dis, (long long)T * N, N, T, mode_d, data_d, cap, it_stat, it_out, ts, cfg.log_ring, ring_d);
  BCK(cudaGetLastError());
}

// 롤아웃 한 스텝: 교사 앞(라벨) → 학생 앞 → 행동 고르기·기록 → 환경 → 에피소드 표 → 지도(다음 줄 토큰)
void Bc::rollout_step(int t) {
  const int ab = (N + AS_E - 1) / AS_E;
  float* obs_t = obs_col + (size_t)(t % 2) * env::N_OBS * N;
  float* obs_n = obs_col + (size_t)((t + 1) % 2) * env::N_OBS * N;
  assemble_k<<<ab, AS_L * AS_E>>>(obs_t, tok->at(t), N, cfg.teacher_use_map, 0, 0, nt.x0, nt.sin, nt.mask);
  forward(PbT, nt, N);
  assemble_k<<<ab, AS_L * AS_E>>>(obs_t, tok->at(t), N, cfg.use_map, 1, cfg.student_goal, ns.x0, ns.sin, ns.mask);
  forward(Pb, ns, N);
  const float* met_init = map->metrics() + (size_t)gmap::M_INIT * N;
  ActP p{nt.mean, ns.mean, mode_d, data_d, t, N, cap, obs_t, tok->at(t), env->soa(), met_init, cur_len, act_env, dis, d_obs, d_tok, d_lab, d_meta, d_rs};
  act_rec_k<<<(N + 7) / 8, 256>>>(p);
  env->step(act_env, obs_n, rew, done);
  epstat_k<<<(N + 127) / 128, 128>>>(done, met_init, N, cur_len, it_stat, it_out, tab);
  map->step(env->soa(), 0, 0, 0, tok->at(t + 1), curr_d);
  BCK(cudaGetLastError());
}

int Bc::launch(int kind) {
  const int R = cfg.log_ring;
  if (issued - polled >= R) return 1;
  const int slot = (int)(issued % R);
  BCK(cudaEventRecord(ev_a[slot], 0));
  if (kind == 0) { if (g_roll) BCK(cudaGraphLaunch(g_roll, 0)); else rollout_body(); }
  else { if (g_upd) BCK(cudaGraphLaunch(g_upd, 0)); else update_body(); }
  BCK(cudaEventRecord(ev_b[slot], 0));
  ++issued;
  return 0;
}

int Bc::poll(BcLog* out) {
  if (polled == issued) return 0;
  const int R = cfg.log_ring, slot = (int)(polled % R);
  const cudaError_t q = cudaEventQuery(ev_b[slot]);
  if (q == cudaErrorNotReady) return 0;
  BCK(q);
  std::memcpy(out, (const void*)&ring_h[slot], sizeof(BcLog));
  float a = 0.f;
  BCK(cudaEventElapsedTime(&a, ev_a[slot], ev_b[slot]));
  out->gpu_ms = a;
  ++polled;
  return 1;
}

}  // namespace bc

// ---- C ABI ----
using bc::Bc;
extern "C" {
void* bc_create(const BcConfig* cfg) { return new Bc(*cfg); }
void bc_destroy(void* h) { delete static_cast<Bc*>(h); }
int bc_load_teacher(void* h, const char* path) {
  auto* b = static_cast<Bc*>(h);
  FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  const long long n = b->lay.total;
  std::vector<uint8_t> hd(64);
  std::vector<float> P((size_t)n);
  const bool ok = std::fread(hd.data(), 1, 64, f) == 64 && std::fread(P.data(), sizeof(float), (size_t)n, f) == (size_t)n;
  std::fclose(f);
  long long nn = 0;
  std::memcpy(&nn, hd.data() + 8, 8);
  if (!ok || std::memcmp(hd.data(), "PPOCKPT1", 8) || nn != n) return -2;
  float* tmp = nullptr;
  BCK(cudaDeviceSynchronize());
  BCK(cudaMalloc(&tmp, sizeof(float) * n));
  BCK(cudaMemcpy(tmp, P.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  net::to_bf16(tmp, b->PbT, n, 0);
  BCK(cudaDeviceSynchronize());
  cudaFree(tmp);
  return 0;
}
int bc_reset_env(void* h, uint64_t env_seed) {
  auto* b = static_cast<Bc*>(h);
  if (b->issued != b->polled) return -1;
  BCK(cudaDeviceSynchronize());
  b->make_env(env_seed);
  BCK(cudaDeviceSynchronize());
  if (b->cfg.use_graphs) b->capture();
  return 0;
}
int bc_set_mode(void* h, int32_t actor, int32_t record) {
  auto* b = static_cast<Bc*>(h);
  const bc::Mode m{actor, record, 0, 0};
  b->set_dev(b->mode_d, &m, sizeof m);
  return 0;
}
int bc_set_map_curriculum(void* h, float p0, float p1, int32_t kmin, int32_t kmax, float reveal_r) {
  auto* b = static_cast<Bc*>(h);
  const gmap::MapCurr c{p0, p1, kmin, kmax, reveal_r, 0};
  b->set_dev(b->curr_d, &c, sizeof c);
  return 0;
}
int bc_set_lr(void* h, float lr) {
  auto* b = static_cast<Bc*>(h);
  b->set_dev(reinterpret_cast<uint8_t*>(b->ts) + offsetof(net::TrainState, lr), &lr, sizeof lr);
  return 0;
}
int bc_rollout(void* h) { return static_cast<Bc*>(h)->launch(0); }
int bc_update(void* h) { return static_cast<Bc*>(h)->launch(1); }
int bc_poll(void* h, BcLog* out) { return static_cast<Bc*>(h)->poll(out); }
int bc_inflight(void* h) { auto* b = static_cast<Bc*>(h); return (int)(b->issued - b->polled); }
int bc_table(void* h, uint64_t* out) {
  auto* b = static_cast<Bc*>(h);
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(out, b->tab, sizeof(unsigned long long) * 3 * 2 * bc::TAB_C * bc::TAB_Q, cudaMemcpyDeviceToHost));
  return 0;
}
int bc_clear_table(void* h) {
  auto* b = static_cast<Bc*>(h);
  BCK(cudaMemsetAsync(b->tab, 0, sizeof(unsigned long long) * 3 * 2 * bc::TAB_C * bc::TAB_Q, 0));
  return 0;
}
int bc_save_student(void* h, const char* path) {
  auto* b = static_cast<Bc*>(h);
  const long long n = b->lay.total;
  std::vector<float> P((size_t)n);
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(P.data(), b->P, sizeof(float) * n, cudaMemcpyDeviceToHost));
  FILE* f = std::fopen(path, "wb");
  if (!f) return -1;
  uint8_t hd[64] = {0};
  std::memcpy(hd, "BCSTUD01", 8);
  std::memcpy(hd + 8, &n, 8);
  std::fwrite(hd, 1, 64, f);
  std::fwrite(P.data(), sizeof(float), (size_t)n, f);
  std::fclose(f);
  return 0;
}
int bc_load_student(void* h, const char* path) {
  auto* b = static_cast<Bc*>(h);
  const long long n = b->lay.total;
  FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  uint8_t hd[64];
  std::vector<float> P((size_t)n);
  const bool ok = std::fread(hd, 1, 64, f) == 64 && std::fread(P.data(), sizeof(float), (size_t)n, f) == (size_t)n;
  std::fclose(f);
  if (!ok || std::memcmp(hd, "BCSTUD01", 8)) return -2;
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(b->P, P.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  net::to_bf16(b->P, b->Pb, n, 0);
  BCK(cudaDeviceSynchronize());
  return 0;
}
int64_t bc_num_params(void*) {
  long long n = 0;
  for (int l : {net::L_S1, net::L_S2, net::L_A1, net::L_A2, net::L_A3, net::L_A4}) n += (long long)net::kLayers[l].N * net::kLayers[l].K;
  return n;
}
int64_t bc_device_bytes(void* h) {
  auto* b = static_cast<Bc*>(h);
  return (int64_t)(b->dev_bytes + b->map->bytes() + b->tok->bytes() + sizeof(float) * (env::NUM_F + 3) * (size_t)b->N);
}
int bc_sync(void*) { BCK(cudaDeviceSynchronize()); return 0; }
}
