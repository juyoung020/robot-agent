// PPO 학습기 본체: 롤아웃(K1 환경 · K2 지도 · K3 관측·정책·표본) → GAE(K4) → 에포크 × 미니배치(앞 → K5 손실 → 뒤 → K9 옵티마이저).
// 한 바퀴를 그래프 둘(rollout, update)로 잡는다. 반복 수(T, 에포크, 미니배치)는 잡을 때 펼친 고정 모양이고,
// 바퀴마다 바뀌는 값(바퀴 번호 → 난수 열쇠·섞기 열쇠, 학습률, 이득 통계)은 장치의 TrainState 에서 커널이 읽는다. 호스트 동기 0.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "obs.h"
#include "shaping.h"
#include "trainer.h"

namespace ppo {

using namespace net;

#define PCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

template <class T_>
T_* Trainer::alloc(size_t n) {
  void* p = nullptr;
  PCK(cudaMalloc(&p, sizeof(T_) * n + 16));
  PCK(cudaMemset(p, 0, sizeof(T_) * n + 16));
  allocs.push_back(p);
  dev_bytes += sizeof(T_) * n;
  return reinterpret_cast<T_*>(p);
}

static bool slot_old() {   // NET_SLOT_OLD=1: 칸 MLP 를 예전 따로 커널(gemm·pool)로 — 비교·측정용
  static const bool v = [] { const char* e = std::getenv("NET_SLOT_OLD"); return e && std::atoi(e) != 0; }();
  return v;
}

// ---- 관측 모으기(롤아웃: 판 i 그대로 / 갱신: 섞은 표본) ----
constexpr int AS_L = 16, AS_E = 8;
// 흔들기 열쇠 = (바퀴, 스텝 << 32 | 판): 롤아웃과 갱신 모으기가 같은 (스텝, 판)에 같은 입력을 만든다
__global__ void __launch_bounds__(AS_L * AS_E) assemble_k(const float* obs_row, const gmap::MapTok* tok_row, int N, int use_map, int goal_mode,
                                                          uint16_t* x0, uint16_t* sc, uint32_t* mask, obsv::VecTab vt, const obsv::ObsAug* aug,
                                                          const TrainState* ts, int t) {
  const int e = blockIdx.x * AS_E + threadIdx.x / AS_L, lane = threadIdx.x % AS_L;
  if (e >= N) return;
  const uint32_t mk = obsv::assemble(obs_row + (size_t)e * N_OBS_G1, 1, 0, tok_row[e], x0 + (size_t)e * X0_W, sc + (size_t)e * KSLOT * SLOT_C, use_map, goal_mode, lane, AS_L,
                                     vt, aug, (uint64_t)ts->iter, ((uint64_t)t << 32) | (uint64_t)e, true);
  if (lane == 0) mask[e] = mk;
}

// 관측 [80][N] → [N][80] (판 32 개씩 공유 메모리로 뒤집음 — 읽기·쓰기 모두 이어서). 값은 그대로 복사
__global__ void __launch_bounds__(256) obs_rows_k(const float* col, int N, float* rows) {
  __shared__ float s[32][N_OBS_G1 + 1];
  const int i0 = blockIdx.x * 32;
  for (int q = threadIdx.x; q < 32 * N_OBS_G1; q += 256) {
    const int c = q / 32, e = q % 32;
    if (i0 + e < N) s[e][c] = col[(size_t)c * N + i0 + e];
  }
  __syncthreads();
  for (int q = threadIdx.x; q < 32 * N_OBS_G1; q += 256) {
    const int e = q / N_OBS_G1, c = q % N_OBS_G1;
    if (i0 + e < N) rows[(size_t)(i0 + e) * N_OBS_G1 + c] = s[e][c];
  }
}

// 섞기: 정의역 2^(2hb) 위 4 단 Feistel + 순환 걷기 → [0, S) 의 순열. 열쇠는 (씨앗, 바퀴, 에포크) — 장치에서 계산
__host__ __device__ inline uint32_t feistel_perm(uint32_t x, uint32_t S, int hb, uint64_t key) {
  const uint32_t m = (1u << hb) - 1u;
  do {
    uint32_t L = x >> hb, R = x & m;
    for (int r = 0; r < 4; ++r) {
      const uint32_t F = (uint32_t)mix64(key + ((uint64_t)r << 40) + R) & m;
      const uint32_t nL = R;
      R = L ^ F;
      L = nL;
    }
    x = (L << hb) | R;
  } while (x >= S);
  return x;
}
inline int half_bits(uint32_t S) {
  int b = 0;
  while ((1ull << b) < S) ++b;
  return (b + 1) / 2;
}

struct GatherP {
  const float* obs_buf /* 행 판 obs_rows [T+1][N][80] */; const gmap::MapTok* tok0; const float* act_buf; const float* logp_buf; const float* val_buf; const float* adv_buf;
  const float* ret_buf; int N, T, MB, base, epoch, hb, use_map, goal_mode; uint64_t seed; const TrainState* ts;
  uint16_t* x0; uint16_t* sc; uint32_t* mask; float *act, *oldlogp, *oldv, *adv, *ret;
  obsv::VecTab vt; const obsv::ObsAug* aug;
};
__global__ void __launch_bounds__(AS_L * AS_E) gather_k(GatherP g) {
  const int r = blockIdx.x * AS_E + threadIdx.x / AS_L, lane = threadIdx.x % AS_L;
  if (r >= g.MB) return;
  const uint64_t key = hash4(g.seed, (uint64_t)g.ts->iter, (uint64_t)g.epoch, 0x5045524dull);
  const uint32_t S = (uint32_t)g.N * (uint32_t)g.T;
  const uint32_t p = feistel_perm((uint32_t)(g.base + r), S, g.hb, key);
  const int t = (int)(p / (uint32_t)g.N), i = (int)(p % (uint32_t)g.N);
  const uint32_t mk = obsv::assemble(g.obs_buf + ((size_t)t * g.N + i) * N_OBS_G1, 1, 0, g.tok0[(size_t)t * g.N + i], g.x0 + (size_t)r * X0_W,
                                     g.sc + (size_t)r * KSLOT * SLOT_C, g.use_map, g.goal_mode, lane, AS_L, g.vt, g.aug, (uint64_t)g.ts->iter,
                                     ((uint64_t)t << 32) | (uint64_t)i, true);
  if (lane < N_ACT) g.act[(size_t)r * N_ACT + lane] = g.act_buf[((size_t)t * g.N + i) * N_ACT + lane];
  if (lane == 0) {
    g.mask[r] = mk;
    g.oldlogp[r] = g.logp_buf[(size_t)t * g.N + i];
    g.oldv[r] = g.val_buf[(size_t)t * g.N + i];
    g.adv[r] = g.adv_buf[(size_t)t * g.N + i];
    g.ret[r] = g.ret_buf[(size_t)t * g.N + i];
  }
}

// ---- 행동 표본(장치 난수) · logp · 가치 ----
__global__ void sample_k(const float* mean, const float* val, const float* logstd, const TrainState* ts, uint64_t seed, int t, int N,
                         float* act_env, float* act_buf, float* logp_buf, float* val_buf) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const uint64_t it = (uint64_t)ts->iter;
  const uint32_t am = ts->act_mask;   // 커리큘럼 가림(장치 값)
  float logp = 0.f;
  for (int k = 0; k < N_ACT; ++k) {
    const uint64_t h = hash4(seed, it, ((uint64_t)t << 8) | (uint64_t)k, (uint64_t)i);
    const float u1 = (float)((h >> 40) + 1ull) * (1.f / 16777216.f);   // (0, 1]
    const float u2 = (float)((h >> 16) & 0xffffffull) * (1.f / 16777216.f);
    const float eps = sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
    const float ls = logstd[k];
    const float mu = mean[(size_t)i * N_ACT + k];
    float a = 0.f;   // 학습하지 않는 행동은 0 고정(CURRICULUM 3절) — 팔·그리퍼 0 = 홈 자세
    if ((am >> k) & 1u) {
      a = mu + expf(ls) * eps;
      const float inv = expf(-ls);
      const float z = (a - mu) * inv;   // ppo_loss_k 와 같은 식(첫 미니배치 비율이 정확히 1)
      logp = logp + (-0.5f * z * z - ls - 0.5f * kLog2Pi);
    }
    act_env[(size_t)k * N + i] = a;
    act_buf[((size_t)t * N + i) * N_ACT + k] = a;
  }
  logp_buf[(size_t)t * N + i] = logp;
  val_buf[(size_t)t * N + i] = val[(size_t)i * 8];
}
__global__ void value_k(const float* val, int N, float* val_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) val_row[i] = val[(size_t)i * 8];
}

// ---- 처음 지도(완성도)별 에피소드 통계(G4, 5.6): 환경 스텝 뒤·지도 스텝 앞. met_init 은 아직 이 스텝의 판(끝난 판)의 처음 지도 부호 ----
constexpr int TAB_Q = 6, TAB_C = 10;
__global__ void epstat_k(const int* done, const float* met_init, int N, int* cur_len, int* it_stat, unsigned long long* tab) {
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
    unsigned long long* q = tab + (size_t)((st * 2 + g) * TAB_C + c) * TAB_Q;
    atomicAdd(q, 1ull);
    atomicAdd(q + (d == env::kSuccess ? 1 : d == env::kCollision ? 2 : 3), 1ull);
    atomicAdd(q + 4, (unsigned long long)len);
    if (d == env::kSuccess) atomicAdd(q + 5, (unsigned long long)len);
    len = 0;
  }
  cur_len[i] = len;
}

// ---- GAE(K4): 판마다 스레드 하나가 T 스텝을 거꾸로. 에피소드 통계도 같이 ----
constexpr int GAE_NQ = 12;
enum GaeQ { Q_RSUM, Q_SUCC, Q_COLL, Q_TOUT, Q_RET, Q_LEN, Q_ASUM, Q_ASQ, Q_VSUM };
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
struct GaeP { const float* rew; const int* done; const float* val; const float* obs; int* first; float* adv; float* ret; float* ep_ret; int* ep_len; float* part; int N, T; float gamma, lam, rs; rw::ShapeP sp; };
__global__ void __launch_bounds__(256) gae_k(GaeP g) {
  __shared__ float sh[GAE_NQ * 256];
  const int i = blockIdx.x * 256 + threadIdx.x;
  float q[GAE_NQ];
  for (int k = 0; k < GAE_NQ; ++k) q[k] = 0.f;
  if (i < g.N) {
    float er = g.ep_ret[i];
    int el = g.ep_len[i];
    for (int t = 0; t < g.T; ++t) {
      const float r = g.rew[(size_t)t * g.N + i];
      const int d = g.done[(size_t)t * g.N + i];
      er = er + r;
      el += 1;
      q[Q_RSUM] = q[Q_RSUM] + r;
      if (d != env::kRunning) {
        q[d == env::kSuccess ? Q_SUCC : (d == env::kCollision ? Q_COLL : Q_TOUT)] += 1.f;
        q[Q_RET] = q[Q_RET] + er;
        q[Q_LEN] = q[Q_LEN] + (float)el;
        er = 0.f;
        el = 0;
      }
    }
    g.ep_ret[i] = er;
    g.ep_len[i] = el;
    const int first0 = g.first[i];
    g.first[i] = g.done[(size_t)(g.T - 1) * g.N + i] != env::kRunning;
    const size_t os = (size_t)N_OBS_G1 * g.N;   // 관측 줄 간격
    float last = 0.f;
    for (int t = g.T - 1; t >= 0; --t) {
      const size_t o = (size_t)t * g.N + i;
      const float v = g.val[o], nv = g.val[o + g.N];
      const int d = g.done[o];
      float r = g.rew[o] * g.rs;
      if (d == env::kCollision && g.sp.coll != 0.f) r = r + g.sp.coll * g.rs;   // 충돌 추가 벌(G4, 기록 반환값에는 안 들어감)
      // 퍼텐셜 모양 잡기: 진짜 끝이면 Φ(s') = 0. 에피소드 첫 스텝은 관측이 지난 판 끝 값이라(G1 은 끝난 스텝의 관측을 돌려줌) 더하지 않음
      const bool first = t > 0 ? g.done[o - g.N] != env::kRunning : first0 != 0;
      if (g.sp.coef != 0.f && !first) {
        const float p0 = rw::potential(g.obs + (size_t)t * os, g.N, i, g.sp);
        const float p1 = (d == env::kSuccess || d == env::kCollision) ? 0.f : rw::potential(g.obs + (size_t)(t + 1) * os, g.N, i, g.sp);
        r = r + g.sp.coef * (g.gamma * p1 - p0);
      }
      float delta;
      if (d == env::kSuccess || d == env::kCollision) { delta = r - v; last = delta; }               // 진짜 끝: 뒤를 잇지 않음
      else if (d == env::kTimeout) { delta = r + g.gamma * nv - v; last = delta; }                   // 시간 끝: 가치로 이음(아래 README 의 한계 참고)
      else { delta = r + g.gamma * nv - v; last = delta + g.gamma * g.lam * last; }
      g.adv[o] = last;
      g.ret[o] = last + v;
      q[Q_ASUM] = q[Q_ASUM] + last;
      q[Q_ASQ] = q[Q_ASQ] + last * last;
      q[Q_VSUM] = q[Q_VSUM] + v;
    }
  }
  bsum<GAE_NQ>(q, sh);
  if (threadIdx.x == 0)
    for (int k = 0; k < GAE_NQ; ++k) g.part[(size_t)blockIdx.x * GAE_NQ + k] = q[k];
}
__global__ void __launch_bounds__(256) gae_reduce_k(const float* part, int nb, int N, int T, TrainState* ts) {
  __shared__ float sh[GAE_NQ * 256];
  float q[GAE_NQ];
  for (int k = 0; k < GAE_NQ; ++k) {
    float s = 0.f;
    for (int b = threadIdx.x; b < nb; b += 256) s = s + part[(size_t)b * GAE_NQ + k];
    q[k] = s;
  }
  bsum<GAE_NQ>(q, sh);
  if (threadIdx.x != 0) return;
  const float n = (float)N * (float)T;
  const float m = q[Q_ASUM] / n;
  ts->adv_mean = m;
  ts->adv_std = sqrtf(fmaxf(q[Q_ASQ] / n - m * m, 0.f));
  ts->r_sum = q[Q_RSUM];
  ts->n_succ = (int)q[Q_SUCC];
  ts->n_coll = (int)q[Q_COLL];
  ts->n_tout = (int)q[Q_TOUT];
  ts->ep_ret_sum = q[Q_RET];
  ts->ep_len_sum = q[Q_LEN];
  ts->v_sum = q[Q_VSUM];
}

// ---- 기록(바퀴 끝): 매핑된 호스트 링에 직접 쓰고 누적을 0 으로, 바퀴 번호 +1 ----
__global__ void __launch_bounds__(256) log_k(TrainState* ts, const float* met_task, const gmap::MapTok* tok_end, int* it_stat, const float* logstd, int N, int T,
                                             int ring, int stage, PpoLog* out) {
  __shared__ float sh[2 * 256];
  float q[2] = {0.f, 0.f};
  for (int i = threadIdx.x; i < N; i += 256) { q[0] = q[0] + met_task[i]; q[1] = q[1] + (obsv::goal_known(tok_end[i]) ? 1.f : 0.f); }
  bsum<2>(q, sh);
  if (threadIdx.x != 0) return;
  const long long it = ts->iter;
  PpoLog L;
  memset(&L, 0, sizeof L);
  L.iter = it + 1;
  L.env_steps = (it + 1) * (long long)N * T;
  L.rew_mean = ts->r_sum / ((float)N * T);
  const float ne = (float)(ts->n_succ + ts->n_coll + ts->n_tout);
  L.n_eps = ne;
  const float inv = ne > 0.f ? 1.f / ne : 0.f;
  L.ep_ret = ts->ep_ret_sum * inv;
  L.ep_len = ts->ep_len_sum * inv;
  L.succ = ts->n_succ * inv;
  L.coll = ts->n_coll * inv;
  L.tout = ts->n_tout * inv;
  const float im = ts->n_mb > 0 ? 1.f / (float)ts->n_mb : 0.f;
  L.kl = ts->s_kl * im;
  L.clipfrac = ts->s_clip * im;
  L.entropy = ts->s_ent * im;
  L.pg_loss = ts->s_pg * im;
  L.v_loss = ts->s_vl * im;
  L.grad_norm = ts->s_gnorm * im;
  L.lr = ts->lr;
  L.adv_mean = ts->adv_mean;
  L.adv_std = ts->adv_std;
  L.value_mean = ts->v_sum / ((float)N * T);
  L.std0 = expf(logstd[0]);
  L.std1 = expf(logstd[1]);
  L.map_task = q[0] / (float)N;
  L.goal_known = q[1] / (float)N;
  for (int k = 0; k < 3; ++k) {
    const int n = it_stat[k * 3];
    L.n_c[k] = (float)n;
    L.s_c[k] = n > 0 ? (float)it_stat[k * 3 + 1] / (float)n : 0.f;
    L.k_c[k] = n > 0 ? (float)it_stat[k * 3 + 2] / (float)n : 0.f;
    it_stat[k * 3] = it_stat[k * 3 + 1] = it_stat[k * 3 + 2] = 0;
  }
  L.stage = stage;
  out[it % ring] = L;
  ts->s_pg = ts->s_vl = ts->s_kl = ts->s_clip = ts->s_ent = ts->s_gnorm = 0.f;
  ts->n_mb = 0;
  ts->iter = it + 1;
}

__global__ void set_col_k(uint16_t* buf, long long rows, int ld, int col, uint16_t v) {
  const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) buf[r * ld + col] = v;
}

// ---------------------------------------------------------------------------------------------------------------------
Trainer::Trainer(const PpoConfig& c) : cfg(c) {
  lay = param_layout();
  N = cfg.n_env;
  T = cfg.horizon;
  const long long S = (long long)N * T;
  if (S % cfg.minibatches) { std::fprintf(stderr, "ppo: N*T (%lld) must be divisible by minibatches (%d)\n", S, cfg.minibatches); std::abort(); }
  MB = (int)(S / cfg.minibatches);
  if (MB % 8 || N % 8) { std::fprintf(stderr, "ppo: minibatch and N must be multiples of 8\n"); std::abort(); }
  Mmax = MB > N ? MB : N;
  if (cfg.act_dims <= 0 || cfg.act_dims > N_ACT) cfg.act_dims = N_ACT;
  net::set_fp8(cfg.fp8);   // G6: 그래프를 잡기 전에(잡을 때 고정되는 호스트 값)
  lh = LossHyper{cfg.clip, cfg.vclip, cfg.vf_coef, cfg.ent_coef, cfg.adaptive_lr, cfg.kl_target, cfg.lr_min, cfg.lr_max, cfg.act_dims, cfg.bound_coef};
  ah = AdamHyper{cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.max_grad_norm};

  obs_buf = alloc<float>((size_t)(T + 1) * N_OBS_G1 * N);
  obs_rows = alloc<float>((size_t)(T + 1) * N_OBS_G1 * N);
  act_env = alloc<float>((size_t)N_ACT * N);
  act_buf = alloc<float>((size_t)T * N * N_ACT);
  logp_buf = alloc<float>((size_t)T * N);
  val_buf = alloc<float>((size_t)(T + 1) * N);
  rew_buf = alloc<float>((size_t)T * N);
  done_buf = alloc<int>((size_t)T * N);
  adv_buf = alloc<float>((size_t)T * N);
  ret_buf = alloc<float>((size_t)T * N);
  ep_ret = alloc<float>(N);
  ep_len = alloc<int>(N);
  gae_part = alloc<float>((size_t)((N + 255) / 256) * GAE_NQ);
  first = alloc<int>(N);
  cur_len = alloc<int>(N);
  it_stat = alloc<int>(9);
  tab = alloc<unsigned long long>((size_t)3 * 2 * TAB_C * TAB_Q);
  curr_d = alloc<gmap::MapCurr>(1);
  PCK(cudaHostAlloc(&curr_h, sizeof(gmap::MapCurr) * 8, cudaHostAllocDefault));
  for (int k = 0; k < 8; ++k) PCK(cudaEventCreateWithFlags(&curr_ev[k], cudaEventDisableTiming));
  {
    const gmap::MapCurr c0{cfg.map_p0, cfg.map_p1, cfg.map_kmin, cfg.map_kmax, cfg.map_reveal_r, 0};
    PCK(cudaMemcpy(curr_d, &c0, sizeof c0, cudaMemcpyHostToDevice));
  }

  const size_t M = Mmax, MS = (size_t)Mmax * KSLOT;
  x0 = alloc<uint16_t>(M * X0_W);
  sc = alloc<uint16_t>(MS * SLOT_C);   // 줄인 칸 줄(표 행은 번호) — 묶음 커널이 펼침
  if (slot_old()) sin = alloc<uint16_t>(MS * SLOT_IN);   // 예전 따로 커널 길만 304 칸 줄을 씀(검증은 sin_full 이 그때 만듦)
  mask = alloc<uint32_t>(M);
  s1o = alloc<uint16_t>(MS * kLayers[L_S1].ldo);
  s2o = alloc<uint16_t>(MS * kLayers[L_S2].ldo);
  amax = alloc<uint8_t>(M * S_H);
  ho[L_S1] = s1o;
  ho[L_S2] = s2o;
  for (int l : {L_A1, L_A2, L_A3, L_C1, L_C2, L_C3}) ho[l] = alloc<uint16_t>(M * kLayers[l].ldo);
  mean = alloc<float>(M * N_ACT);
  val = alloc<float>(M * 8);
  for (int l = 0; l < N_LAYER; ++l) dz[l] = alloc<uint16_t>((l <= L_S2 ? MS : M) * kLayers[l].N);
  dpool = alloc<float>(M * POOL_W);
  mb_act = alloc<float>((size_t)MB * N_ACT);
  mb_oldlogp = alloc<float>(MB);
  mb_oldv = alloc<float>(MB);
  mb_adv = alloc<float>(MB);
  mb_ret = alloc<float>(MB);
  // 1 칸(편향 입력)
  for (int l = 0; l < N_LAYER; ++l) {
    const LayerDesc& L = kLayers[l];
    if (L.ldo > L.N && ho[l]) {
      const long long rows = l <= L_S2 ? (long long)MS : (long long)M;
      set_col_k<<<(unsigned)((rows + 255) / 256), 256>>>(ho[l], rows, L.ldo, L.N, (uint16_t)0x3f80);
    }
  }
  P = alloc<float>(lay.total);
  G = alloc<float>(lay.total);
  Am = alloc<float>(lay.total);
  Av = alloc<float>(lay.total);
  Pb = alloc<uint16_t>(lay.total);
  for (int l = 0; l < N_LAYER; ++l) {
    const int rows = l <= L_S2 ? MB * KSLOT : MB;
    splits[l] = dw_splits(rows, cfg.dw_chunk);
    ws[l] = alloc<float>((size_t)splits[l] * kLayers[l].N * kLayers[l].K);
  }
  gn_part = alloc<float>(GN_BLOCKS);
  loss_part = alloc<float>((size_t)loss_blocks(MB) * LOSS_NQ);
  ts = alloc<TrainState>(1);

  // 변수 초기화(호스트, 결정적): 균등 ±gain·√(3/fan_in), 편향 0, log σ = init
  std::vector<float> hp(lay.total, 0.f);
  uint64_t s = cfg.seed * 0x9E3779B97F4A7C15ull + 7;
  for (int l = 0; l < N_LAYER; ++l) {
    const LayerDesc& L = kLayers[l];
    const int nout = l == L_C4 ? kValueOut : L.N;
    const float a = L.gain * std::sqrt(3.f / (float)L.bias);
    for (int n = 0; n < nout; ++n)
      for (int k = 0; k < L.bias; ++k) hp[lay.off[l] + (size_t)n * L.K + k] = dm::rand_range(s, -a, a);
  }
  for (int k = 0; k < N_ACT; ++k) hp[lay.logstd + k] = cfg.init_logstd;
  PCK(cudaMemcpy(P, hp.data(), sizeof(float) * lay.total, cudaMemcpyHostToDevice));
  to_bf16(P, Pb, lay.total, 0);
  TrainState h{};
  h.lr = cfg.lr;
  h.adv_std = 1.f;
  h.act_mask = cfg.act_mask ? cfg.act_mask : ((1u << cfg.act_dims) - 1u);
  if (!vt.load()) std::abort();
  vt.upload();
  dev_bytes += vt.dev_bytes();
  aug_d = alloc<obsv::ObsAug>(1);
  {
    obsv::ObsAug a = obsv::kAugOff;
    a.on = cfg.aug_on; a.eval_unseen = cfg.aug_eval_unseen; a.vel_sigma = cfg.aug_vel_sigma; a.prev_drop = cfg.aug_prev_drop; a.prev_sigma = cfg.aug_prev_sigma;
    a.p_erase = cfg.aug_p_erase; a.p_syn = cfg.aug_p_syn; a.p_hyper = cfg.aug_p_hyper; a.p_wrong = cfg.aug_p_wrong; a.p_slot_drop = cfg.aug_p_slot_drop;
    a.p_map_off = cfg.aug_p_map_off; a.seed = cfg.seed * 0x2545F4914F6CDD1Dull + 0xA06ull;
    PCK(cudaMemcpy(aug_d, &a, sizeof a, cudaMemcpyHostToDevice));
  }
  PCK(cudaMemcpy(ts, &h, sizeof h, cudaMemcpyHostToDevice));

  const int R = cfg.log_ring > 1 ? cfg.log_ring : 16;
  cfg.log_ring = R;
  PCK(cudaHostAlloc(&ring_h, sizeof(PpoLog) * R, cudaHostAllocMapped));
  std::memset(ring_h, 0, sizeof(PpoLog) * R);
  PCK(cudaHostGetDevicePointer(&ring_d, ring_h, 0));
  ev_a.resize(R); ev_b.resize(R); ev_c.resize(R);
  for (int k = 0; k < R; ++k) { PCK(cudaEventCreate(&ev_a[k])); PCK(cudaEventCreate(&ev_b[k])); PCK(cudaEventCreate(&ev_c[k])); }
  ckpt_bytes = 64 + sizeof(float) * 3 * lay.total + sizeof(TrainState);
  PCK(cudaHostAlloc(&ckpt_h, ckpt_bytes, cudaHostAllocDefault));
  PCK(cudaEventCreateWithFlags(&ev_ckpt, cudaEventDisableTiming));

  make_env(cfg.stage);
  PCK(cudaDeviceSynchronize());
  if (cfg.use_graphs) capture();
}

Trainer::~Trainer() {
  cudaDeviceSynchronize();
  if (g_roll) cudaGraphExecDestroy(g_roll);
  if (g_upd) cudaGraphExecDestroy(g_upd);
  for (void* p : allocs) cudaFree(p);
  for (size_t k = 0; k < ev_a.size(); ++k) { cudaEventDestroy(ev_a[k]); cudaEventDestroy(ev_b[k]); cudaEventDestroy(ev_c[k]); }
  if (ev_ckpt) cudaEventDestroy(ev_ckpt);
  if (ring_h) cudaFreeHost(ring_h);
  if (ckpt_h) cudaFreeHost(ckpt_h);
  if (curr_h) cudaFreeHost(curr_h);
  vt.free_dev();
  for (auto& e : curr_ev) if (e) cudaEventDestroy(e);
}

// 처음 지도 비율 바꾸기: 고정 호스트 링 한 칸에 쓰고 스트림 순서대로 비동기 복사(이미 띄운 바퀴 뒤, 다음 바퀴 앞). 동기 없음.
// 칸은 8 번 바뀐 뒤에야 다시 쓴다 — 그 칸의 복사가 아직 안 끝났을 때만(드묾) 기다린다
void Trainer::set_map_curr(const gmap::MapCurr& c) {
  const int k = curr_slot++ % 8;
  if (cudaEventQuery(curr_ev[k]) == cudaErrorNotReady) PCK(cudaEventSynchronize(curr_ev[k]));
  curr_h[k] = c;
  PCK(cudaMemcpyAsync(curr_d, &curr_h[k], sizeof c, cudaMemcpyHostToDevice, 0));
  PCK(cudaEventRecord(curr_ev[k], 0));
}

// 학습하는 행동 비트(TrainState::act_mask) 바꾸기: 같은 고정 호스트 링으로 4 B 복사 하나(다음 바퀴부터, 다시 잡기 없음)
void Trainer::set_act_mask(uint32_t m) {
  const int k = curr_slot++ % 8;
  if (cudaEventQuery(curr_ev[k]) == cudaErrorNotReady) PCK(cudaEventSynchronize(curr_ev[k]));
  std::memcpy(&curr_h[k], &m, sizeof m);
  PCK(cudaMemcpyAsync(&ts->act_mask, &curr_h[k], sizeof m, cudaMemcpyHostToDevice, 0));
  PCK(cudaEventRecord(curr_ev[k], 0));
}

void Trainer::make_env(int stage) {
  cfg.stage = stage;
  tok.reset();
  map.reset();
  env.reset();
  // 팔을 늘 풀어 둔다(행동 8, VLA_INPUT 5절): 꺼진 행동은 표본이 0 → 관절 목표 = 홈 + 0·범위 = 홈 자세라 예전(팔 묶음)과 비트가 같다(env_verify --arm-zero)
  env = std::make_unique<env::DeviceEnv>(N, stage, cfg.seed * 1000003ull + 17ull + (uint64_t)stage, true);
  map = std::make_unique<gmap::DeviceMap>(N, cfg.seed * 7919ull + 3ull + (uint64_t)stage);
  tok = std::make_unique<gmap::TokenRecorder>(N, T + 1);
  PCK(cudaMemset(obs_buf, 0, sizeof(float) * (size_t)(T + 1) * N_OBS_G1 * N));
  PCK(cudaMemset(obs_rows, 0, sizeof(float) * (size_t)(T + 1) * N_OBS_G1 * N));
  PCK(cudaMemset(ep_ret, 0, sizeof(float) * N));
  PCK(cudaMemset(ep_len, 0, sizeof(int) * N));
  PCK(cudaMemset(cur_len, 0, sizeof(int) * N));
  std::vector<int> ones(N, 1);
  PCK(cudaMemcpy(first, ones.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
}

static cudaGraphExec_t capture_one(Trainer* tr, void (Trainer::*body)()) {
  cudaGraph_t g;
  PCK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
  (tr->*body)();
  PCK(cudaStreamEndCapture(cudaStreamPerThread, &g));
  cudaGraphExec_t ex;
  PCK(cudaGraphInstantiate(&ex, g, 0));
  size_t nn = 0;
  cudaGraphGetNodes(g, nullptr, &nn);
  std::fprintf(stderr, "ppo: captured graph with %zu nodes\n", nn);
  cudaGraphDestroy(g);
  return ex;
}
void Trainer::capture() {
  if (g_roll) { cudaGraphExecDestroy(g_roll); g_roll = nullptr; }
  if (g_upd) { cudaGraphExecDestroy(g_upd); g_upd = nullptr; }
  g_roll = capture_one(this, &Trainer::rollout_body);
  g_upd = capture_one(this, &Trainer::update_body);
}


// ---- 몸통 ----
void Trainer::forward(int M) {
  const uint16_t* W = Pb;
  auto Wl = [&](int l) { return W + lay.off[l]; };
  if (!slot_old()) {   // 칸 MLP 앞 묶음(한 커널, 같은 결과). 칸 입력은 줄인 칸 줄 + 얼린 표에서 커널 안에서 펼침
    slot_fwd_c(slot_c(), mask, Wl(L_S1), Wl(L_S2), M, s1o, s2o, x0, amax, 0);
  } else {
    slot_expand(slot_c(), (long long)M * KSLOT, sin, 0);
    gemm_fwd(kLayers[L_S1], sin, M * KSLOT, Wl(L_S1), s1o, 0);
    gemm_fwd(kLayers[L_S2], s1o, M * KSLOT, Wl(L_S2), s2o, 0);
    pool_fwd(s2o, mask, M, x0, amax, 0);
  }
  // 정책·가치 사슬의 같은 층을 한 번에(gemm_fwd2: 실행 수만 반, 결과 같음)
  gemm_fwd2(kLayers[L_A1], x0, x0, M, Wl(L_A1), Wl(L_C1), ho[L_A1], ho[L_C1], 0);
  gemm_fwd2(kLayers[L_A2], ho[L_A1], ho[L_C1], M, Wl(L_A2), Wl(L_C2), ho[L_A2], ho[L_C2], 0);
  gemm_fwd2(kLayers[L_A3], ho[L_A2], ho[L_C2], M, Wl(L_A3), Wl(L_C3), ho[L_A3], ho[L_C3], 0);
  gemm_fwd2(kLayers[L_A4], ho[L_A3], ho[L_C3], M, Wl(L_A4), Wl(L_C4), mean, val, 0);
}

void Trainer::backward(int M) {
  auto Wl = [&](int l) { return Pb + lay.off[l]; };
  const int ch = cfg.dw_chunk;
  // 정책 사슬(A)·가치 사슬(C)의 같은 층을 한 번에(*2: 실행 수만 반, 결과 같음). dpool 은 A 가 쓰고 C 가 더함(예전 순서)
  const int a4 = L_A4, a3 = L_A3, a2 = L_A2, a1 = L_A1, c4 = L_C4, c3 = L_C3, c2 = L_C2, c1 = L_C1;
  gemm_dw2(kLayers[a4], dz[a4], dz[c4], ho[a3], ho[c3], M, ws[a4], ws[c4], ch, 0);
  gemm_dx_dact2(kLayers[a4], dz[a4], dz[c4], M, Wl(a4), Wl(c4), ho[a3], ho[c3], kLayers[a3].N, dz[a3], dz[c3], 0, 0);
  gemm_dw2(kLayers[a3], dz[a3], dz[c3], ho[a2], ho[c2], M, ws[a3], ws[c3], ch, 0);
  gemm_dx_dact2(kLayers[a3], dz[a3], dz[c3], M, Wl(a3), Wl(c3), ho[a2], ho[c2], kLayers[a2].N, dz[a2], dz[c2], bug == 1 ? 1 : 0, 0);
  gemm_dw2(kLayers[a2], dz[a2], dz[c2], ho[a1], ho[c1], M, ws[a2], ws[c2], ch, 0);
  gemm_dx_dact2(kLayers[a2], dz[a2], dz[c2], M, Wl(a2), Wl(c2), ho[a1], ho[c1], kLayers[a1].N, dz[a1], dz[c1], 0, 0);
  gemm_dw2(kLayers[a1], dz[a1], dz[c1], x0, x0, M, ws[a1], ws[c1], ch, 0);
  gemm_dx_pool(kLayers[a1], dz[a1], M, Wl(a1), dpool, false, 0);
  gemm_dx_pool(kLayers[c1], dz[c1], M, Wl(c1), dpool, true, 0);
  if (!slot_old()) {   // 칸 MLP 뒤 묶음(한 커널, 같은 결과)
    slot_bwd_c(dpool, s2o, s1o, slot_c(), mask, amax, Wl(L_S2), M, ch, ws[L_S2], ws[L_S1], keep_slot_bufs ? dz[L_S2] : nullptr, keep_slot_bufs ? dz[L_S1] : nullptr, 0);
  } else {
    pool_bwd(dpool, s2o, mask, amax, M, dz[L_S2], 0);
    gemm_dw(kLayers[L_S2], dz[L_S2], s1o, M * KSLOT, ws[L_S2], ch, 0);
    gemm_dx_dact(kLayers[L_S2], dz[L_S2], M * KSLOT, Wl(L_S2), s1o, kLayers[L_S1].N, dz[L_S1], 0, 0);
    gemm_dw(kLayers[L_S1], dz[L_S1], sin, M * KSLOT, ws[L_S1], ch, 0);
  }
  DwJob jobs[N_LAYER];
  for (int l = 0; l < N_LAYER; ++l) {
    const int rows = l <= L_S2 ? M * KSLOT : M;
    jobs[l] = DwJob{ws[l], G + lay.off[l], kLayers[l].N * kLayers[l].K, dw_splits(rows, ch)};
  }
  dw_reduce(jobs, N_LAYER, 0);
}

// 검증용: 지금 줄인 칸 줄(행 rows 개)을 304 칸 줄로 펼친 버퍼(처음 부를 때 할당 — 그래프 잡기 밖에서만)
uint16_t* Trainer::sin_full(int rows) {
  if (!sin) sin = alloc<uint16_t>((size_t)Mmax * KSLOT * SLOT_IN);
  slot_expand(slot_c(), (long long)rows * KSLOT, sin, 0);
  return sin;
}

void Trainer::gather(int epoch, int mb) {
  GatherP g{obs_rows, tok->at(0), act_buf, logp_buf, val_buf, adv_buf, ret_buf, N, T, MB, mb * MB, epoch, half_bits((uint32_t)(N * T)), cfg.use_map,
            cfg.goal_from_map, cfg.seed, ts, x0, sc, mask, mb_act, mb_oldlogp, mb_oldv, mb_adv, mb_ret, vt.dev(), aug_d};
  gather_k<<<(MB + AS_E - 1) / AS_E, AS_L * AS_E>>>(g);
  PCK(cudaGetLastError());
}

void Trainer::loss(int M) {
  LossIn in{mean, val, P + lay.logstd, mb_act, mb_oldlogp, mb_oldv, mb_adv, mb_ret};
  ppo_loss(in, M, ts, lh, dz[L_A4], dz[L_C4], loss_part, bug, 0);
  ppo_loss_reduce(loss_part, loss_blocks(M), M, lh, G + lay.logstd, P + lay.logstd, ts, 0);
}

void Trainer::optimizer() { adam_step(P, G, Am, Av, Pb, lay.total, gn_part, ts, ah, 0); }

void Trainer::gae() {
  GaeP g{rew_buf, done_buf, val_buf, obs_buf, first, adv_buf, ret_buf, ep_ret, ep_len, gae_part, N, T, cfg.gamma, cfg.lam, cfg.reward_scale,
         rw::ShapeP{cfg.shape_coef, cfg.shape_near, cfg.shape_aim, cfg.shape_zone, cfg.shape_v, cfg.shape_w, cfg.coll_extra}};
  const int nb = (N + 255) / 256;
  gae_k<<<nb, 256>>>(g);
  gae_reduce_k<<<1, 256>>>(gae_part, nb, N, T, ts);
  PCK(cudaGetLastError());
}

void Trainer::log_iter() {
  log_k<<<1, 256>>>(ts, map->metrics() + (size_t)gmap::M_TASK * N, tok->at(T), it_stat, P + lay.logstd, N, T, cfg.log_ring, cfg.stage, ring_d);
  PCK(cudaGetLastError());
}

void Trainer::rollout_body() {
  // 지난 바퀴 끝 줄(관측·지도 토큰)을 0 줄로
  PCK(cudaMemcpyAsync(obs_buf, obs_buf + (size_t)T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice, 0));
  PCK(cudaMemcpyAsync(obs_rows, obs_rows + (size_t)T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice, 0));
  PCK(cudaMemcpyAsync(tok->at(0), tok->at(T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice, 0));
  for (int t = 0; t <= T; ++t) rollout_step(t);
  PCK(cudaGetLastError());
}

// 롤아웃 한 스텝(t < T): 관측 모으기 → 정책 앞 → 표본 → 환경 → 지도. t == T: 마지막 가치(부트스트랩)만
void Trainer::rollout_step(int t) {
  const int ab = (N + AS_E - 1) / AS_E, sb = (N + 127) / 128;
  assemble_k<<<ab, AS_L * AS_E>>>(obs_rows + (size_t)t * N_OBS_G1 * N, tok->at(t), N, cfg.use_map, cfg.goal_from_map, x0, sc, mask, vt.dev(), aug_d, ts, t);
  forward(N);
  if (t == T) { value_k<<<sb, 128>>>(val, N, val_buf + (size_t)T * N); return; }
  sample_k<<<sb, 128>>>(mean, val, P + lay.logstd, ts, cfg.seed, t, N, act_env, act_buf, logp_buf, val_buf);
  env->step(act_env, obs_buf + (size_t)(t + 1) * N_OBS_G1 * N, rew_buf + (size_t)t * N, done_buf + (size_t)t * N);
  obs_rows_k<<<(N + 31) / 32, 256>>>(obs_buf + (size_t)(t + 1) * N_OBS_G1 * N, N, obs_rows + (size_t)(t + 1) * N_OBS_G1 * N);
  epstat_k<<<sb, 128>>>(done_buf + (size_t)t * N, map->metrics() + (size_t)gmap::M_INIT * N, N, cur_len, it_stat, tab);
  map->step(env->soa(), 0, 0, 0, tok->at(t + 1), curr_d);
}

void Trainer::update_body() {
  gae();
  for (int e = 0; e < cfg.epochs; ++e)
    for (int b = 0; b < cfg.minibatches; ++b) {
      gather(e, b);
      forward(MB);
      loss(MB);
      backward(MB);
      optimizer();
    }
  log_iter();
}

int Trainer::iterate() {
  const int R = cfg.log_ring;
  if (issued - polled >= R) return 1;
  const int slot = (int)(issued % R);
  PCK(cudaEventRecord(ev_a[slot], 0));
  if (g_roll) PCK(cudaGraphLaunch(g_roll, 0)); else rollout_body();
  PCK(cudaEventRecord(ev_b[slot], 0));
  if (g_upd) PCK(cudaGraphLaunch(g_upd, 0)); else update_body();
  PCK(cudaEventRecord(ev_c[slot], 0));
  ++issued;
  return 0;
}

int Trainer::eval_iterate() {
  const int R = cfg.log_ring;
  if (issued - polled >= R) return 1;
  const int slot = (int)(issued % R);
  PCK(cudaEventRecord(ev_a[slot], 0));
  if (g_roll) PCK(cudaGraphLaunch(g_roll, 0)); else rollout_body();
  PCK(cudaEventRecord(ev_b[slot], 0));
  gae();
  log_iter();
  PCK(cudaEventRecord(ev_c[slot], 0));
  ++issued;
  return 0;
}

int Trainer::poll(PpoLog* out) {
  if (polled == issued) return 0;
  const int R = cfg.log_ring, slot = (int)(polled % R);
  const cudaError_t q = cudaEventQuery(ev_c[slot]);
  if (q == cudaErrorNotReady) return 0;
  PCK(q);
  std::memcpy(out, (const void*)&ring_h[slot], sizeof(PpoLog));
  float a = 0.f, b = 0.f;
  PCK(cudaEventElapsedTime(&a, ev_a[slot], ev_b[slot]));
  PCK(cudaEventElapsedTime(&b, ev_b[slot], ev_c[slot]));
  out->rollout_ms = a;
  out->update_ms = b;
  ++polled;
  return 1;
}

}  // namespace ppo

// ---- C ABI ----
using ppo::Trainer;
extern "C" {
void* ppo_create(const PpoConfig* cfg) { return new Trainer(*cfg); }
void ppo_destroy(void* h) { delete static_cast<Trainer*>(h); }
int ppo_iterate(void* h) { return static_cast<Trainer*>(h)->iterate(); }
int ppo_poll(void* h, PpoLog* out) { return static_cast<Trainer*>(h)->poll(out); }
int ppo_inflight(void* h) { auto* t = static_cast<Trainer*>(h); return (int)(t->issued - t->polled); }
int ppo_set_stage(void* h, int stage) {
  auto* t = static_cast<Trainer*>(h);
  if (t->issued != t->polled) return -1;
  PCK(cudaDeviceSynchronize());
  t->make_env(stage);
  PCK(cudaDeviceSynchronize());
  if (t->cfg.use_graphs) t->capture();
  return 0;
}
int ppo_ckpt_begin(void* h) {
  auto* t = static_cast<Trainer*>(h);
  if (t->ckpt_pending) return -1;
  const long long n = t->lay.total;
  uint8_t* p = t->ckpt_h;
  std::memset(p, 0, 64);
  std::memcpy(p, "PPOCKPT1", 8);
  std::memcpy(p + 8, &n, 8);
  const int32_t st = t->cfg.stage;
  std::memcpy(p + 16, &st, 4);
  PCK(cudaMemcpyAsync(p + 64, t->P, sizeof(float) * n, cudaMemcpyDeviceToHost, 0));
  PCK(cudaMemcpyAsync(p + 64 + sizeof(float) * n, t->Am, sizeof(float) * n, cudaMemcpyDeviceToHost, 0));
  PCK(cudaMemcpyAsync(p + 64 + 2 * sizeof(float) * n, t->Av, sizeof(float) * n, cudaMemcpyDeviceToHost, 0));
  PCK(cudaMemcpyAsync(p + 64 + 3 * sizeof(float) * n, t->ts, sizeof(net::TrainState), cudaMemcpyDeviceToHost, 0));
  PCK(cudaEventRecord(t->ev_ckpt, 0));
  t->ckpt_pending = true;
  return 0;
}
int ppo_ckpt_poll(void* h, const uint8_t** data, int64_t* nbytes) {
  auto* t = static_cast<Trainer*>(h);
  if (!t->ckpt_pending) return -1;
  if (cudaEventQuery(t->ev_ckpt) != cudaSuccess) return 0;
  t->ckpt_pending = false;
  *data = t->ckpt_h;
  *nbytes = (int64_t)t->ckpt_bytes;
  return 1;
}
int ppo_load(void* h, const uint8_t* data, int64_t nbytes) {
  auto* t = static_cast<Trainer*>(h);
  const long long n = t->lay.total;
  if (nbytes != (int64_t)t->ckpt_bytes || std::memcmp(data, "PPOCKPT1", 8)) return -1;
  PCK(cudaDeviceSynchronize());
  PCK(cudaMemcpy(t->P, data + 64, sizeof(float) * n, cudaMemcpyHostToDevice));
  PCK(cudaMemcpy(t->Am, data + 64 + sizeof(float) * n, sizeof(float) * n, cudaMemcpyHostToDevice));
  PCK(cudaMemcpy(t->Av, data + 64 + 2 * sizeof(float) * n, sizeof(float) * n, cudaMemcpyHostToDevice));
  net::TrainState s;
  std::memcpy(&s, data + 64 + 3 * sizeof(float) * n, sizeof s);
  s.s_pg = s.s_vl = s.s_kl = s.s_clip = s.s_ent = s.s_gnorm = 0.f;
  s.n_mb = 0;
  PCK(cudaMemcpy(t->ts, &s, sizeof s, cudaMemcpyHostToDevice));
  net::to_bf16(t->P, t->Pb, n, 0);
  PCK(cudaDeviceSynchronize());
  t->issued = t->polled = s.iter;
  return 0;
}
int ppo_set_map_curriculum(void* h, float p0, float p1, int32_t kmin, int32_t kmax, float reveal_r) {
  static_cast<Trainer*>(h)->set_map_curr(gmap::MapCurr{p0, p1, kmin, kmax, reveal_r, 0});
  return 0;
}
int ppo_set_act_mask(void* h, uint32_t mask) {
  static_cast<Trainer*>(h)->set_act_mask(mask);
  return 0;
}
int64_t ppo_issued(void* h) { return static_cast<Trainer*>(h)->issued; }
int64_t ppo_num_params(void* h) { return static_cast<Trainer*>(h)->lay.total; }
int64_t ppo_device_bytes(void* h) {
  auto* t = static_cast<Trainer*>(h);
  return (int64_t)(t->dev_bytes + t->map->bytes() + t->tok->bytes() + sizeof(float) * (env::NUM_F + 3) * (size_t)t->N);
}
}
