// BC 학생 학습기 본체(G5): 롤아웃(교사 앞 → [학생: 렌더 → 얼린 SigLIP 2 → 학생 앞 → 머리] → 행동 고르기·기록 → K1 환경 → K2 지도) 과
// 갱신(K 번 [모으기(표본·청크 라벨·렌더 상태) → 다시 렌더 → 인코더 → 앞 → MSE 또는 flow 손실(K5) → 뒤 → Adam(K9)]).
// 그래프 셋(교사가 움직이는 롤아웃, 학생이 움직이는 롤아웃, 갱신). 누가 기록하나(장치 Mode), 자료 쓰기 자리·표본 수(장치 Data),
// 미니배치·flow 잡음·시간 난수 열쇠(TrainState.iter, Data.rollouts)는 모두 장치 값 → 단계 안 호스트 동기 0.
// 신경망 커널(손 BF16 GEMM, 칸 MLP 묶음, dW 조각 합, Adam)은 training/RL/network 것을 그대로 쓴다(소스를 읽기만).
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bc.h"
#include "bc_render.h"

namespace bc {

using namespace net;

#define BCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

static const cudaStream_t ST = cudaStreamPerThread;

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

// ---- 학생 신경망 모양 ----
StudentNet student_net(const BcConfig& c) {
  StudentNet s;
  s.vision = c.vision != 0;
  s.text = c.text != 0;
  s.head = c.head != 0;
  s.H = s.head ? (c.chunk < 1 ? 1 : c.chunk > MAX_H ? MAX_H : c.chunk) : 1;
  s.x_img = X0_W;
  s.x_txt = X0_W + (s.vision ? IMG_W : 0);
  s.k1 = s.x_txt + (s.text ? TXT_W : 0);
  const float g2 = 1.41421356f;
  s.L[SL_S1] = kLayers[L_S1];
  s.L[SL_S2] = kLayers[L_S2];
  s.L[SL_P1] = LayerDesc{vit::TOK_LD, IMG_D, IMG_D, vit::D, ACT_ELU, g2};
  s.L[SL_A1] = LayerDesc{s.k1, 256, 272, X0_BIAS, ACT_ELU, g2};
  s.L[SL_A2] = kLayers[L_A2];
  s.L[SL_A3] = LayerDesc{272, 128, s.head ? E_IN : 144, 256, ACT_ELU, g2};
  s.L[SL_A4] = kLayers[L_A4];
  s.L[SL_E1] = LayerDesc{E_IN, 256, 272, 128, ACT_ELU, g2};
  s.L[SL_E2] = LayerDesc{272, 256, 272, 256, ACT_ELU, g2};
  s.L[SL_E3] = LayerDesc{272, FLOW_W, FLOW_W, 256, ACT_LIN, 0.01f};
  for (int l = 0; l < SL_N; ++l) s.on[l] = false;
  s.on[SL_S1] = s.on[SL_S2] = s.on[SL_A1] = s.on[SL_A2] = s.on[SL_A3] = true;
  s.on[SL_P1] = s.vision;
  s.on[SL_A4] = !s.head;
  s.on[SL_E1] = s.on[SL_E2] = s.on[SL_E3] = s.head;
  long long o = 0;
  for (int l : {SL_S1, SL_S2, SL_A1, SL_A2, SL_A3, SL_A4, SL_E1, SL_E2, SL_E3, SL_P1}) {
    s.off[l] = o;
    if (!s.on[l]) continue;
    o += (long long)s.L[l].N * s.L[l].K;
    o = (o + 63) / 64 * 64;
  }
  s.total = o;
  return s;
}

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

// ---- 스텝 처음: 새 에피소드면 번호를 매기고, 학생 렌더 상태·지시 번호 ----
__global__ void step_begin_k(env::Soa s, const int* cur_len, uint32_t* ep_uid, const Data* dd, int t, int T, int n_txt, RenderState* rs, int* tid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s.N) return;
  uint32_t u = ep_uid[i];
  if (cur_len[i] == 0) {
    u = (uint32_t)(((unsigned long long)dd->rollouts * (unsigned long long)T + (unsigned long long)t) * (unsigned long long)s.N + (unsigned long long)i) + 1u;
    if (u == 0) u = 1;
    ep_uid[i] = u;
  }
  if (tid) tid[i] = text_id(u, n_txt);
  if (rs) { RenderState r; render_state(s, i, r); rs[i] = r; }
}

// ---- A1 입력 이어 붙이기: [X0 288 | 영상 2048 | 글 768] (16 B 덩이 하나 = 스레드 하나) ----
__global__ void concat_k(const uint16_t* x0, const uint16_t* imgf, const uint16_t* txt, const int* tid, int M, int k1, int x_img, int x_txt, uint16_t* out) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int nc = k1 / 8;
  if (q >= (long long)M * nc) return;
  const int r = (int)(q / nc), c = (int)(q % nc) * 8;
  uint4 v;
  if (c < X0_W) v = *reinterpret_cast<const uint4*>(x0 + (size_t)r * X0_W + c);
  else if (c < x_txt) v = *reinterpret_cast<const uint4*>(imgf + (size_t)r * IMG_W + (c - x_img));
  else v = *reinterpret_cast<const uint4*>(txt + (size_t)tid[r] * TXT_W + (c - x_txt));
  *reinterpret_cast<uint4*>(out + (size_t)r * k1 + c) = v;
}

// ---- flow matching(K5·K10): 시간 임베딩, 잡음, 학습 입력·손실, 추론 오일러 ----
// 학습 입력: 행 r 의 τ = U(0,1), ε = N(0,1) 32 개, x_τ = τ ε + (1 − τ) a, 목표 u = ε − a (가린 칸은 a = 0, 가림 0)
__global__ void flow_train_in_k(const float* chunk, const float* fm, const TrainState* ts, uint64_t seed, int M, uint16_t* ein, float* u) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  const uint64_t it = (uint64_t)ts->iter;
  const float tau = u01(hash4(seed ^ 0xF10Full, it, (uint64_t)r, 0x7a75ull));
  uint16_t* e = ein + (size_t)r * E_IN;
  for (int j = 0; j < FLOW_W; ++j) {
    const float a = fm[(size_t)r * MAX_H + j / N_LAB] > 0.f ? chunk[(size_t)r * FLOW_W + j] : 0.f;
    const float eps = gauss(hash4(seed ^ 0xF10Full, it, (uint64_t)r, 0x1000ull + (uint64_t)j));
    e[E_X + j] = f2bf(tau * eps + (1.f - tau) * a);
    u[(size_t)r * FLOW_W + j] = eps - a;
  }
  temb_write(tau, e + E_T);
}
constexpr int LOSS_T = 256;
template <int NQ>
__device__ void bsum(float (&q)[NQ], float* sh);
// L = (1/M) Σ_r Σ_j m (v − u)², dZ_E3 = 2 m (v − u)/M (bf16)
__global__ void __launch_bounds__(LOSS_T) flow_loss_k(const float* v, const float* u, const float* fm, int M, int bug, uint16_t* dz, float* part) {
  __shared__ float sh[LOSS_T];
  const int r = blockIdx.x * LOSS_T + threadIdx.x;
  float q[1] = {0.f};
  if (r < M) {
    const float inv = 1.f / (float)M, two = bug == 3 ? 1.f : 2.f;
    for (int j = 0; j < FLOW_W; ++j) {
      const float m = fm[(size_t)r * MAX_H + j / N_LAB];
      const float d = v[(size_t)r * FLOW_W + j] - u[(size_t)r * FLOW_W + j];
      q[0] = q[0] + m * d * d;
      dz[(size_t)r * FLOW_W + j] = f2bf(two * m * d * inv);
    }
  }
  bsum<1>(q, sh);
  if (threadIdx.x == 0) part[blockIdx.x] = q[0];
}
// 추론: x = ε (판·롤아웃·스텝마다), τ_s = 1 − s/S 에서 v 를 구해 x ← x − v/S. 끝에서 행동 = x 의 첫 칸 둘(청크 첫 스텝)
__global__ void flow_noise_k(const Data* dd, uint64_t seed, int t, int M, float* x) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  for (int j = 0; j < FLOW_W; ++j) x[(size_t)r * FLOW_W + j] = gauss(hash4(seed ^ 0x1AF5ull, (uint64_t)dd->rollouts, (uint64_t)t * 65536ull + (uint64_t)r, (uint64_t)j));
}
__global__ void flow_step_in_k(const float* x, float tau, int M, uint16_t* ein) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  uint16_t* e = ein + (size_t)r * E_IN;
  for (int j = 0; j < FLOW_W; ++j) e[E_X + j] = f2bf(x[(size_t)r * FLOW_W + j]);
  temb_write(tau, e + E_T);
}
__global__ void flow_euler_k(const float* v, float dt, int M, int last, float* x, float* act) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  for (int j = 0; j < FLOW_W; ++j) x[(size_t)r * FLOW_W + j] = x[(size_t)r * FLOW_W + j] - dt * v[(size_t)r * FLOW_W + j];
  if (last)
    for (int k = 0; k < N_ACT; ++k) act[(size_t)r * N_ACT + k] = k < N_LAB ? x[(size_t)r * FLOW_W + k] : 0.f;
}
__global__ void copy_mean_k(const float* mean, int M, float* act) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < M * N_ACT) act[q] = mean[q];
}

// ---- 행동 고르기 + 기록(5.3): 워프 하나 = 판 하나 ----
struct ActP {
  const float *meanT, *meanS; const Mode* md; const Data* dd; int t, N; long long cap;
  const float* obs_col; const gmap::MapTok* tok; env::Soa s; const float* met_init; const int* cur_len; const uint32_t* ep_uid;
  float* act_env; float* dis; uint16_t* d_obs; gmap::MapTok* d_tok; float* d_lab; uint32_t* d_meta; uint32_t* d_epi; RenderState* d_rs;
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
    p.d_epi[slot] = p.ep_uid[i];
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

// ---- 미니배치 모으기: 자료 버퍼에서 균등 무작위(복원 추출, 장치 열쇠 = (씨앗, 갱신 스텝, 행)) → 학생 입력·라벨·청크·렌더 상태·지시 번호 ----
// 청크 칸 h(< H): 같은 판의 h 스텝 뒤 표본 = 자리 (idx + h·N) % cap. 같은 에피소드(번호)이고 에피소드 스텝이 h 만큼 뒤일 때만 씀(가림 1), 아니면 0.
struct GatherP {
  const uint16_t* d_obs; const gmap::MapTok* d_tok; const float* d_lab; const uint32_t* d_meta; const uint32_t* d_epi; const RenderState* d_rs;
  const Data* dd; const TrainState* ts;
  int MB, use_map, student_goal, N, H, n_txt; long long cap; uint64_t seed;
  uint16_t* x0; uint16_t* sin; uint32_t* mask; float* lab; float* chunk; float* fm; RenderState* rs; int* tid;
};
NDEV long long gather_index(uint64_t seed, long long iter, int r, long long cnt) {
  return cnt > 0 ? (long long)(hash4(seed, (uint64_t)iter, (uint64_t)r, 0x42434d42ull) % (uint64_t)cnt) : 0;
}
__global__ void __launch_bounds__(AS_L * AS_E) gather_k(GatherP g) {
  __shared__ float so[AS_E][env::N_OBS];
  const int rl = threadIdx.x / AS_L, lane = threadIdx.x % AS_L;
  const int r = blockIdx.x * AS_E + rl;
  long long idx = 0;
  if (r < g.MB) {
    idx = gather_index(g.seed, g.ts->iter, r, g.dd->count);
    for (int c = lane; c < env::N_OBS; c += AS_L) so[rl][c] = bf2f(g.d_obs[idx * env::N_OBS + c]);
  }
  __syncthreads();
  if (r >= g.MB) return;   // MB 는 8 의 배수 → 워프 단위로 같이 빠짐
  const uint32_t mk = assemble_student(so[rl], 1, 0, g.d_tok[idx], g.x0 + (size_t)r * X0_W, g.sin + (size_t)r * KSLOT * SLOT_IN, g.use_map, g.student_goal,
                                       lane, AS_L);
  if (lane < N_LAB) g.lab[(size_t)r * N_LAB + lane] = g.d_lab[idx * N_LAB + lane];
  if (lane == 0) g.mask[r] = mk;
  const uint32_t e0 = g.d_epi[idx];
  if (g.chunk) {
    const int h = lane;   // MAX_H = AS_L
    const long long sh = (idx + (long long)h * g.N) % g.cap;
    const bool ok = h < g.H && g.d_epi[sh] == e0 && (g.d_meta[sh] & 0xffffu) == ((g.d_meta[idx] & 0xffffu) + (uint32_t)h);
    for (int k = 0; k < N_LAB; ++k) g.chunk[((size_t)r * MAX_H + h) * N_LAB + k] = ok ? g.d_lab[sh * N_LAB + k] : 0.f;
    g.fm[(size_t)r * MAX_H + h] = ok ? 1.f : 0.f;
  }
  if (g.rs) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(g.d_rs + idx);
    uint32_t* dst = reinterpret_cast<uint32_t*>(g.rs + r);
    for (int q = lane; q < (int)(sizeof(RenderState) / 4); q += AS_L) dst[q] = src[q];
  }
  if (g.tid && lane == 0) g.tid[r] = text_id(e0, g.n_txt);
}

// ---- BC 손실(K5): L = (1/M) Σ_r Σ_{k<2} (μ − 라벨)², dZ_A4 = 2(μ − 라벨)/M (bf16), 나머지 행동 0 ----
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
static void set_col(uint16_t* buf, long long rows, int ld, int col) {
  set_col_k<<<(unsigned)((rows + 255) / 256), 256>>>(buf, rows, ld, col, (uint16_t)0x3f80);
}

// ---------------------------------------------------------------------------------------------------------------------
static void alloc_teacher(Bc& b, NetBufs& n, int M) {
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
    set_col(n.ho[l], l <= L_S2 ? (long long)MS : (long long)M, L.ldo, L.N);
  }
}
static void alloc_student(Bc& b, int M) {
  const StudentNet& s = b.sn;
  SBufs& n = b.sb;
  const size_t MS = (size_t)M * KSLOT;
  n.x0 = b.alloc<uint16_t>((size_t)M * X0_W);
  n.sin = b.alloc<uint16_t>(MS * SLOT_IN);
  n.mask = b.alloc<uint32_t>(M);
  n.s1o = b.alloc<uint16_t>(MS * kLayers[L_S1].ldo);
  n.s2o = b.alloc<uint16_t>(MS * kLayers[L_S2].ldo);
  n.amax = b.alloc<uint8_t>((size_t)M * S_H);
  n.tid = b.alloc<int>(M);
  n.h[SL_S1] = n.s1o;
  n.h[SL_S2] = n.s2o;
  set_col(n.s1o, (long long)MS, kLayers[L_S1].ldo, kLayers[L_S1].N);
  if (s.ext()) n.x0e = b.alloc<uint16_t>((size_t)M * s.k1);
  if (s.vision) {
    n.tok = b.alloc<uint16_t>((size_t)M * IMG_TOK * vit::TOK_LD);
    vit::init_token_buffer(n.tok, (long long)M * IMG_TOK, ST);
    n.imgf = b.alloc<uint16_t>((size_t)M * IMG_W);
    n.h[SL_P1] = n.imgf;
  }
  for (int l : {SL_A1, SL_A2, SL_A3, SL_E1, SL_E2}) {
    if (!s.on[l]) continue;
    n.h[l] = b.alloc<uint16_t>((size_t)M * s.L[l].ldo);
    set_col(n.h[l], M, s.L[l].ldo, s.L[l].N);
  }
  n.out = b.alloc<float>((size_t)M * FLOW_W);
  n.act = b.alloc<float>((size_t)M * N_ACT);
  n.xf = b.alloc<float>((size_t)M * FLOW_W);
  n.fm = b.alloc<float>((size_t)M * MAX_H);
  for (int l = 0; l < SL_N; ++l) {
    if (!s.on[l]) continue;
    const size_t rows = l <= SL_S2 ? MS : l == SL_P1 ? (size_t)M * IMG_TOK : (size_t)M;
    n.dz[l] = b.alloc<uint16_t>(rows * s.L[l].N);
  }
  n.dpool = b.alloc<float>((size_t)M * POOL_W);
}

Bc::Bc(const BcConfig& c) : cfg(c) {
  lay = param_layout();
  sn = student_net(cfg);
  N = cfg.n_env;
  T = cfg.horizon;
  MB = cfg.mb;
  if (N % 8 || MB % 8 || T % 2 || cfg.dw_chunk % 64) { std::fprintf(stderr, "bc: N, mb must be multiples of 8, horizon even, dw_chunk multiple of 64\n"); std::abort(); }
  if (sn.vision && !cfg.store_render) { std::fprintf(stderr, "bc: vision needs store_render\n"); std::abort(); }
  cap = cfg.cap;
  ah = AdamHyper{cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.max_grad_norm};

  obs_col = alloc<float>((size_t)2 * env::N_OBS * N);
  act_env = alloc<float>((size_t)N_ACT * N);
  rew = alloc<float>(N);
  done = alloc<int>(N);
  cur_len = alloc<int>(N);
  ep_uid = alloc<uint32_t>(N);
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
  d_epi = alloc<uint32_t>((size_t)cap);
  if (cfg.store_render) d_rs = alloc<RenderState>((size_t)cap);

  PbT = alloc<uint16_t>(lay.total);
  P = alloc<float>(sn.total);
  G = alloc<float>(sn.total);
  Am = alloc<float>(sn.total);
  Av = alloc<float>(sn.total);
  Pb = alloc<uint16_t>(sn.total);
  alloc_teacher(*this, nt, N);
  SM = N > MB ? N : MB;
  alloc_student(*this, SM);
  lab_mb = alloc<float>((size_t)MB * N_LAB);
  chunk_mb = alloc<float>((size_t)MB * FLOW_W);
  for (int l = 0; l < SL_N; ++l) {
    if (!sn.on[l]) continue;
    const int rows = l <= SL_S2 ? MB * KSLOT : l == SL_P1 ? MB * IMG_TOK : MB;
    ws[l] = alloc<float>((size_t)dw_splits(rows, cfg.dw_chunk) * sn.L[l].N * sn.L[l].K);
  }
  gn_part = alloc<float>(GN_BLOCKS);
  loss_part = alloc<float>((size_t)(MB + LOSS_T - 1) / LOSS_T);
  ts = alloc<TrainState>(1);
  txt = alloc<uint16_t>((size_t)MAX_TXT * TXT_W);

  if (sn.vision) {
    vit::HostWeights hw;
    if (!vit::load_weights("", hw)) { std::fprintf(stderr, "bc: SigLIP 2 weights not found\n"); std::abort(); }
    enc.init(hw, 2 * SM);
    const int rb = cfg.render_batch > 0 ? cfg.render_batch : 256;
    rnd = bcr::create(rb < SM ? rb : SM, cfg.render_profile);
    rs_roll = alloc<RenderState>(N);
    rs_mb = alloc<RenderState>(MB);
    dev_bytes += enc.bytes + enc.W.bytes + bcr::bytes(rnd);
  }

  // 학생 초기화(호스트, 결정적): ppo 학습기와 같은 규칙(균등 ±gain·√(3/fan_in)). 차례 S1, S2, A1(X0 칸), A2, A3, A4 는 student-lite 와 같아
  // 영상·글·flow 를 다 끄면 예전 학생과 같은 값. 그 뒤 E1(몸통 128 + x_τ·시간 64 칸, fan_in 192), E2, E3, P1, A1 의 영상·글 칸(fan_in = 그 칸 수).
  {
    std::vector<float> hp(sn.total, 0.f);
    uint64_t s = cfg.seed * 0x9E3779B97F4A7C15ull + 11;
    auto fill = [&](int l, int k0, int k1, int fan) {
      const LayerDesc& L = sn.L[l];
      const float a = L.gain * std::sqrt(3.f / (float)fan);
      for (int n = 0; n < L.N; ++n)
        for (int k = k0; k < k1; ++k) {
          if (l == SL_E1 && k == L.bias) continue;
          hp[sn.off[l] + (size_t)n * L.K + k] = dm::rand_range(s, -a, a);
        }
    };
    for (int l : {SL_S1, SL_S2, SL_A1, SL_A2, SL_A3, SL_A4})
      if (sn.on[l]) fill(l, 0, sn.L[l].bias, sn.L[l].bias);
    if (sn.head) {
      fill(SL_E1, 0, E_T + TEMB, E_T + TEMB - 1);
      fill(SL_E2, 0, sn.L[SL_E2].bias, sn.L[SL_E2].bias);
      fill(SL_E3, 0, sn.L[SL_E3].bias, sn.L[SL_E3].bias);
    }
    if (sn.vision) fill(SL_P1, 0, vit::D, vit::D);
    if (sn.ext()) fill(SL_A1, X0_W, sn.k1, sn.k1 - X0_W);
    BCK(cudaMemcpy(P, hp.data(), sizeof(float) * sn.total, cudaMemcpyHostToDevice));
    to_bf16(P, Pb, sn.total, 0);
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
  for (auto* g : {&g_roll, &g_roll_s, &g_upd}) if (*g) cudaGraphExecDestroy(*g);
  tok.reset(); map.reset(); env.reset();
  if (rnd) bcr::destroy(rnd);
  enc.free_all();
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
  for (auto* g : {&g_roll, &g_roll_s, &g_upd}) if (*g) { cudaGraphExecDestroy(*g); *g = nullptr; }
  g_roll = capture_one(this, &Bc::rollout_body_t);
  g_roll_s = capture_one(this, &Bc::rollout_body_s);
  g_upd = capture_one(this, &Bc::update_body);
}

// ---- 신경망 ----
void Bc::forward_teacher(NetBufs& b, int M) {
  auto Wl = [&](int l) { return PbT + lay.off[l]; };
  slot_fwd(b.sin, b.mask, Wl(L_S1), Wl(L_S2), M, b.s1o, b.s2o, b.x0, b.amax, 0);
  gemm_fwd(kLayers[L_A1], b.x0, M, Wl(L_A1), b.ho[L_A1], 0);
  gemm_fwd(kLayers[L_A2], b.ho[L_A1], M, Wl(L_A2), b.ho[L_A2], 0);
  gemm_fwd(kLayers[L_A3], b.ho[L_A2], M, Wl(L_A3), b.ho[L_A3], 0);
  gemm_fwd(kLayers[L_A4], b.ho[L_A3], M, Wl(L_A4), b.mean, 0);
}

// 렌더(render_batch 판씩) → K11 패치 → 얼린 인코더 → sb.tok [M × 128][784]
void Bc::vis_encode(const RenderState* rs, int M) {
  const int E = bcr::batch(rnd);
  for (int e0 = 0; e0 < M; e0 += E) {
    const int m = M - e0 < E ? M - e0 : E;
    bcr::render(rnd, rs + e0, m, ST);
    enc.patchify(bcr::rgb(rnd, 0), bcr::rgb(rnd, 1), m, e0, ST);
  }
  enc.run(2 * M, sb.tok, ST);
}

void Bc::student_trunk(int M, const RenderState* rs) {
  auto W = [&](int l) { return Pb + sn.off[l]; };
  slot_fwd(sb.sin, sb.mask, W(SL_S1), W(SL_S2), M, sb.s1o, sb.s2o, sb.x0, sb.amax, 0);
  const uint16_t* X = sb.x0;
  if (sn.ext()) {
    if (sn.vision) {
      vis_encode(rs, M);
      gemm_fwd(sn.L[SL_P1], sb.tok, M * IMG_TOK, W(SL_P1), sb.imgf, 0);
    }
    const long long nq = (long long)M * (sn.k1 / 8);
    concat_k<<<(unsigned)((nq + 255) / 256), 256>>>(sb.x0, sb.imgf, txt, sb.tid, M, sn.k1, sn.x_img, sn.x_txt, sb.x0e);
    X = sb.x0e;
  }
  gemm_fwd(sn.L[SL_A1], X, M, W(SL_A1), sb.h[SL_A1], 0);
  gemm_fwd(sn.L[SL_A2], sb.h[SL_A1], M, W(SL_A2), sb.h[SL_A2], 0);
  gemm_fwd(sn.L[SL_A3], sb.h[SL_A2], M, W(SL_A3), sb.h[SL_A3], 0);
  BCK(cudaGetLastError());
}

static void e_chain(Bc& b, int M) {
  auto W = [&](int l) { return b.Pb + b.sn.off[l]; };
  gemm_fwd(b.sn.L[SL_E1], b.sb.h[SL_A3], M, W(SL_E1), b.sb.h[SL_E1], 0);
  gemm_fwd(b.sn.L[SL_E2], b.sb.h[SL_E1], M, W(SL_E2), b.sb.h[SL_E2], 0);
  gemm_fwd(b.sn.L[SL_E3], b.sb.h[SL_E2], M, W(SL_E3), b.sb.out, 0);
}

// 추론 머리: MSE 는 A4 평균, flow 는 오일러 S 스텝(τ = 1 → 0) 뒤 청크 첫 행동
void Bc::student_act(int M) {
  const unsigned g = (unsigned)((M + 127) / 128);
  if (!sn.head) {
    gemm_fwd(sn.L[SL_A4], sb.h[SL_A3], M, Pb + sn.off[SL_A4], sb.out, 0);
    copy_mean_k<<<(unsigned)((M * N_ACT + 255) / 256), 256>>>(sb.out, M, sb.act);
  } else {
    const int S = cfg.flow_steps > 0 ? cfg.flow_steps : 10;
    for (int s = 0; s < S; ++s) {
      flow_step_in_k<<<g, 128>>>(sb.xf, 1.f - (float)s / (float)S, M, sb.h[SL_A3]);
      e_chain(*this, M);
      flow_euler_k<<<g, 128>>>(sb.out, 1.f / (float)S, M, s == S - 1, sb.xf, sb.act);
    }
  }
  BCK(cudaGetLastError());
}

void Bc::gather() {
  GatherP g{d_obs, d_tok, d_lab, d_meta, d_epi, sn.vision ? d_rs : nullptr, data_d, ts, MB, cfg.use_map, cfg.student_goal, N, sn.H, n_txt, cap, cfg.seed,
            sb.x0, sb.sin, sb.mask, lab_mb, sn.head ? chunk_mb : nullptr, sb.fm, sn.vision ? rs_mb : nullptr, sn.text ? sb.tid : nullptr};
  gather_k<<<(MB + AS_E - 1) / AS_E, AS_L * AS_E>>>(g);
  BCK(cudaGetLastError());
}

void Bc::flow_inputs(int M) {
  flow_train_in_k<<<(unsigned)((M + 127) / 128), 128>>>(chunk_mb, sb.fm, ts, cfg.seed, M, sb.h[SL_A3], sb.xf);
  BCK(cudaGetLastError());
}

void Bc::head_forward(int M) {
  if (!sn.head) gemm_fwd(sn.L[SL_A4], sb.h[SL_A3], M, Pb + sn.off[SL_A4], sb.out, 0);
  else { flow_inputs(M); e_chain(*this, M); }
}

void Bc::loss(int M) {
  const int nb = (M + LOSS_T - 1) / LOSS_T;
  if (!sn.head) bc_loss_k<<<nb, LOSS_T>>>(sb.out, lab_mb, M, bug, sb.dz[SL_A4], loss_part);
  else flow_loss_k<<<nb, LOSS_T>>>(sb.out, sb.xf, sb.fm, M, bug, sb.dz[SL_E3], loss_part);
  bc_loss_reduce_k<<<1, 256>>>(loss_part, nb, M, ts);
  BCK(cudaGetLastError());
}

void Bc::backward(int M) {
  auto W = [&](int l) { return Pb + sn.off[l]; };
  const int ch = cfg.dw_chunk;
  const LayerDesc* L = sn.L;
  SBufs& b = sb;
  if (!sn.head) {
    gemm_dw(L[SL_A4], b.dz[SL_A4], b.h[SL_A3], M, ws[SL_A4], ch, 0);
    gemm_dx_dact(L[SL_A4], b.dz[SL_A4], M, W(SL_A4), b.h[SL_A3], L[SL_A3].N, b.dz[SL_A3], 0, 0);
  } else {
    gemm_dw(L[SL_E3], b.dz[SL_E3], b.h[SL_E2], M, ws[SL_E3], ch, 0);
    gemm_dx_dact(L[SL_E3], b.dz[SL_E3], M, W(SL_E3), b.h[SL_E2], L[SL_E2].N, b.dz[SL_E2], 0, 0);
    gemm_dw(L[SL_E2], b.dz[SL_E2], b.h[SL_E1], M, ws[SL_E2], ch, 0);
    gemm_dx_dact(L[SL_E2], b.dz[SL_E2], M, W(SL_E2), b.h[SL_E1], L[SL_E1].N, b.dz[SL_E1], 0, 0);
    gemm_dw(L[SL_E1], b.dz[SL_E1], b.h[SL_A3], M, ws[SL_E1], ch, 0);
    gemm_dx_dact(L[SL_E1], b.dz[SL_E1], M, W(SL_E1), b.h[SL_A3], L[SL_A3].N, b.dz[SL_A3], bug == 5 ? 1 : 0, 0);
  }
  gemm_dw(L[SL_A3], b.dz[SL_A3], b.h[SL_A2], M, ws[SL_A3], ch, 0);
  gemm_dx_dact(L[SL_A3], b.dz[SL_A3], M, W(SL_A3), b.h[SL_A2], L[SL_A2].N, b.dz[SL_A2], bug == 2 ? 1 : 0, 0);
  gemm_dw(L[SL_A2], b.dz[SL_A2], b.h[SL_A1], M, ws[SL_A2], ch, 0);
  gemm_dx_dact(L[SL_A2], b.dz[SL_A2], M, W(SL_A2), b.h[SL_A1], L[SL_A1].N, b.dz[SL_A1], 0, 0);
  const uint16_t* X1 = sn.ext() ? b.x0e : b.x0;
  gemm_dw(L[SL_A1], b.dz[SL_A1], X1, M, ws[SL_A1], ch, 0);
  gemm_dx_pool(L[SL_A1], b.dz[SL_A1], M, W(SL_A1), b.dpool, false, 0);
  if (sn.vision) {   // 영상 칸 dX → dZ_P1 [M][2048] = [M × 128][16] (P1 출력 ELU' 곱) → dW P1 (토큰 행 M × 128)
    gemm_dx_dact(L[SL_A1], b.dz[SL_A1], M, W(SL_A1) + sn.x_img, b.x0e + sn.x_img, IMG_W, b.dz[SL_P1], bug == 4 ? 1 : 0, 0);
    gemm_dw(L[SL_P1], b.dz[SL_P1], b.tok, M * IMG_TOK, ws[SL_P1], ch, 0);
  }
  slot_bwd(b.dpool, b.s2o, b.s1o, b.sin, b.mask, b.amax, W(SL_S2), M, ch, ws[SL_S2], ws[SL_S1], keep_slot_bufs ? b.dz[SL_S2] : nullptr,
           keep_slot_bufs ? b.dz[SL_S1] : nullptr, 0);
  DwJob jobs[SL_N];
  int nj = 0;
  for (int l = 0; l < SL_N; ++l) {
    if (!sn.on[l]) continue;
    const int rows = l <= SL_S2 ? M * KSLOT : l == SL_P1 ? M * IMG_TOK : M;
    jobs[nj++] = DwJob{ws[l], G + sn.off[l], L[l].N * L[l].K, dw_splits(rows, ch)};
  }
  dw_reduce(jobs, nj, 0);
}

void Bc::optimizer() { adam_step(P, G, Am, Av, Pb, sn.total, gn_part, ts, ah, 0); }

void Bc::update_step() {
  gather();
  student_trunk(MB, rs_mb);
  head_forward(MB);
  loss(MB);
  backward(MB);
  optimizer();
}

void Bc::update_body() {
  for (int k = 0; k < cfg.upd_steps; ++k) update_step();
  update_end_k<<<1, 1>>>(ts, data_d, cfg.log_ring, ring_d);
  BCK(cudaGetLastError());
}

void Bc::rollout_body(bool student) {
  for (int t = 0; t < T; ++t) rollout_step(t, student);
  rollout_end_k<<<1, 256>>>(dis, (long long)T * N, N, T, mode_d, data_d, cap, it_stat, it_out, ts, cfg.log_ring, ring_d);
  BCK(cudaGetLastError());
}

// 롤아웃 한 스텝: 에피소드 번호·렌더 상태 → 교사 앞(라벨) → [학생 앞 → 머리] → 행동 고르기·기록 → 환경 → 에피소드 표 → 지도(다음 줄 토큰)
// student = false(교사가 움직이는 그래프)면 학생 계산을 통째로 뺀다(영상 학생의 렌더·인코더 비용이 교사 기록에 들지 않게). 이때 어긋남 기록은 0.
void Bc::rollout_step(int t, bool student) {
  const int ab = (N + AS_E - 1) / AS_E;
  float* obs_t = obs_col + (size_t)(t % 2) * env::N_OBS * N;
  float* obs_n = obs_col + (size_t)((t + 1) % 2) * env::N_OBS * N;
  step_begin_k<<<(N + 127) / 128, 128>>>(env->soa(), cur_len, ep_uid, data_d, t, T, n_txt, student && sn.vision ? rs_roll : nullptr,
                                         student && sn.text ? sb.tid : nullptr);
  assemble_k<<<ab, AS_L * AS_E>>>(obs_t, tok->at(t), N, cfg.teacher_use_map, 0, 0, nt.x0, nt.sin, nt.mask);
  forward_teacher(nt, N);
  const float* meanS = nt.mean;
  if (student) {
    assemble_k<<<ab, AS_L * AS_E>>>(obs_t, tok->at(t), N, cfg.use_map, 1, cfg.student_goal, sb.x0, sb.sin, sb.mask);
    student_trunk(N, rs_roll);
    if (sn.head) flow_noise_k<<<(N + 127) / 128, 128>>>(data_d, cfg.seed, t, N, sb.xf);
    student_act(N);
    meanS = sb.act;
  }
  const float* met_init = map->metrics() + (size_t)gmap::M_INIT * N;
  ActP p{nt.mean, meanS, mode_d, data_d, t, N, cap, obs_t, tok->at(t), env->soa(), met_init, cur_len, ep_uid, act_env, dis, d_obs, d_tok, d_lab, d_meta, d_epi, d_rs};
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
  if (kind == 0) {
    cudaGraphExec_t g = host_actor ? g_roll_s : g_roll;
    if (g) BCK(cudaGraphLaunch(g, 0)); else rollout_body(host_actor != 0);
  } else {
    if (g_upd) BCK(cudaGraphLaunch(g_upd, 0)); else update_body();
  }
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
int bc_load_text_table(void* h, const char* path) {
  auto* b = static_cast<Bc*>(h);
  FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  const int k = (int)(bytes / (4 * bc::TXT_W));
  if (k < 1 || k > bc::MAX_TXT || bytes != (long)k * 4 * bc::TXT_W) { std::fclose(f); return -2; }
  std::vector<float> v((size_t)k * bc::TXT_W);
  const bool ok = std::fread(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  if (!ok) return -3;
  std::vector<uint16_t> hb(v.size());
  for (size_t i = 0; i < v.size(); ++i) hb[i] = net::f2bf(v[i]);
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(b->txt, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  b->n_txt = k;
  if (b->cfg.use_graphs) b->capture();   // 문장 수는 그래프에 박히는 호스트 값
  return k;
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
  b->host_actor = actor;
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
  const long long n = b->sn.total;
  std::vector<float> P((size_t)n);
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(P.data(), b->P, sizeof(float) * n, cudaMemcpyDeviceToHost));
  FILE* f = std::fopen(path, "wb");
  if (!f) return -1;
  uint8_t hd[64] = {0};
  std::memcpy(hd, "BCSTUD02", 8);
  std::memcpy(hd + 8, &n, 8);
  const int32_t shape[4] = {b->sn.vision, b->sn.text, b->sn.head, b->sn.H};
  std::memcpy(hd + 16, shape, sizeof shape);
  std::fwrite(hd, 1, 64, f);
  std::fwrite(P.data(), sizeof(float), (size_t)n, f);
  std::fclose(f);
  return 0;
}
int bc_load_student(void* h, const char* path) {
  auto* b = static_cast<Bc*>(h);
  const long long n = b->sn.total;
  FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  uint8_t hd[64];
  std::vector<float> P((size_t)n);
  const bool ok = std::fread(hd, 1, 64, f) == 64 && std::fread(P.data(), sizeof(float), (size_t)n, f) == (size_t)n;
  std::fclose(f);
  long long nn = 0;
  std::memcpy(&nn, hd + 8, 8);
  if (!ok || (std::memcmp(hd, "BCSTUD01", 8) && std::memcmp(hd, "BCSTUD02", 8)) || nn != n) return -2;
  BCK(cudaDeviceSynchronize());
  BCK(cudaMemcpy(b->P, P.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  net::to_bf16(b->P, b->Pb, n, 0);
  BCK(cudaDeviceSynchronize());
  return 0;
}
int64_t bc_num_params(void* h) {
  auto* b = static_cast<Bc*>(h);
  long long n = 0;
  for (int l = 0; l < bc::SL_N; ++l)
    if (b->sn.on[l]) n += (long long)b->sn.L[l].N * b->sn.L[l].K;
  return n;
}
int64_t bc_device_bytes(void* h) {
  auto* b = static_cast<Bc*>(h);
  return (int64_t)(b->dev_bytes + b->map->bytes() + b->tok->bytes() + sizeof(float) * (env::NUM_F + 3) * (size_t)b->N);
}
int bc_sync(void*) { BCK(cudaDeviceSynchronize()); return 0; }
}
