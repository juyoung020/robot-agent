// 지도 갱신 커널(K2 map_update). 두 단계: 판마다 한 스레드가 시작(움직임 거르기)을 하고, keyframe 인 판만 블록 하나(NT 스레드)가
// 갱신한다. 상태는 제자리 갱신(CUDA 그래프로 잡을 수 있게), 호스트 동기 없음.
#include <cstdio>
#include <cstdlib>

#include "map_api.h"

namespace gmap {

struct BlockSync { __device__ void operator()() const { __syncthreads(); } };

__global__ void __launch_bounds__(NT) map_init_kernel(MapCore* core, int N, uint64_t seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) init_core(core[i], seed, i);
}

// 1 단계(판마다 한 스레드): 시작(리셋·오도메트리·움직임 거르기). keyframe·리셋인 판은 목록(장치 안, 원자 덧셈)에 넣고,
// 아닌 판은 여기서 완성도만 쓰고 끝난다. 목록 순서는 실행마다 달라도 판끼리 독립이라 결과는 같다.
constexpr int BEGIN_NT = 128;
// BEH: BEHAVIOR 장면 묶음이 있는 판(장면 코드 포함). 상자 방 지도는 BEH = false 로 띄워 장면 갈래가 컴파일에서 빠진다(레지스터·스택이 예전과 같게)
template <bool BEH>
__global__ void __launch_bounds__(BEGIN_NT) map_begin_kernel(env::Soa s, MapCore* core, float* met, uint32_t* list, int* count, int force_kf,
                                                             const bsc::SceneSet* ss_in, BMapEnv* bm) {
  const int i = blockIdx.x * BEGIN_NT + threadIdx.x, N = s.N;
  if (i >= N) return;
  const EnvView e = read_env(s, i, BEH);
  MapCore& m = core[i];
  const bsc::SceneSet* ss = BEH ? ss_in : nullptr;
  BMapEnv* bmi = BEH ? bm + i : nullptr;
  const int f = phase_begin(m, e, force_kf, ss, bmi);
#ifdef MAP_PROF
  atomicAdd(&g_prof[P_NSEC + ((f & B_KF) ? 1 : 0)], 1ull);
#endif
  if (f) list[atomicAdd(count, 1)] = (uint32_t)i | ((uint32_t)f << LIST_SHIFT);
  else write_metrics(m, e, met, N, i, (bmi && bmi->on) ? bmi->nprim : N_PRIM);
}

// 2 단계(목록의 판 하나 = 블록 하나, NT 스레드): keyframe 갱신. 블록 수는 N 으로 고정하고 목록 길이 밖 블록은 바로 끝난다(호스트 동기 없음).
// 블록이 대략 번호 순서로 SM 에 올라가므로, 한 물결(SM 수 × 상주 블록) 뒤 판의 MapCore 를 L2 로 미리 당긴다(값은 안 바뀜).
constexpr int PF_AHEAD = 1024;
// read_env 가 읽는 SoA 값의 자리(미리 당기기용): x, y, yaw, v, w, tx, ty, rhx, rhy, q[N_Q], ep
constexpr int ENV_PF_N = 9 + env::N_Q + 1;
__device__ __forceinline__ const void* env_pf_addr(const env::Soa& s, int i, int k) {
  const int N = s.N;
  constexpr int F[9] = {env::F_X, env::F_Y, env::F_YAW, env::F_V, env::F_W, env::F_TX, env::F_TY, env::F_RHX, env::F_RHY};
  if (k < 9) return s.f + (size_t)F[k] * N + i;
  if (k < 9 + env::N_Q) return s.f + (size_t)(env::F_Q0 + k - 9) * N + i;
  return s.iv + (size_t)env::I_EP * N + i;
}
template <bool BEH>
__global__ void __launch_bounds__(NT, 10) map_kf_kernel(env::Soa s, MapCore* core, int16_t* L, uint32_t* seen, uint32_t* occ,
                                                        int16_t* segs, float* met, const uint32_t* list, const int* count, int bug,
                                                        const MapCurr* curr, const bsc::SceneSet* ss, BMapEnv* bm, uint8_t* view) {
  __shared__ __align__(16) MapCore m;
  __shared__ KfShared u;
  const int j = blockIdx.x, tid = threadIdx.x, N = s.N;
  const int n = *count;
  if (j >= n) return;
  PROF_START();
  const uint32_t ent = list[j];
  constexpr uint32_t IMASK = (1u << LIST_SHIFT) - 1u;
  const int i = (int)(ent & IMASK), flags = (int)(ent >> LIST_SHIFT);
  if (j + PF_AHEAD < n) {   // 한 물결 뒤 판: MapCore 와 환경 SoA 값(판마다 따로 떨어진 줄)을 L2 로
    const uint32_t ia = list[j + PF_AHEAD] & IMASK;
    if (tid < (int)((sizeof(MapCore) + 127) / 128))
      asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(&core[ia]) + tid * 128));
    else if (tid >= 32 && tid < 32 + ENV_PF_N)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(env_pf_addr(s, (int)ia, tid - 32)));
  }
  if (tid < (int)(sizeof(uint32_t) * NWORD / 128) && (flags & (B_KF | B_WALL)))   // 벽 단계가 읽을 점유 비트 2 KB 를 L2 로 미리
    asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(occ + (size_t)i * NWORD) + tid * 128));
  static_assert(sizeof(MapCore) % 16 == 0, "MapCore copied in 16-byte pieces");
  const uint4* src = reinterpret_cast<const uint4*>(&core[i]);
  uint4* dst = reinterpret_cast<uint4*>(&m);
  for (int k = tid; k < (int)(sizeof(MapCore) / 16); k += NT) dst[k] = src[k];
  const EnvView e = read_env(s, i, BEH);   // 코어 읽기와 겹침
  __syncthreads();
  PROF_MARK(P_LOAD);
  const MapGrid g{L + (size_t)i * NCELL, seen + (size_t)i * NWORD, occ + (size_t)i * NWORD, segs + (size_t)i * SEGW, view + (size_t)i * VIEW_BYTES};
  const MapCurr cu = *curr;   // 장치 값(바퀴 사이에 바뀔 수 있음 — 다시 잡기 없이)
  map_rest(m, u, e, g, met, N, i, tid, NT, bug, flags, cu, BlockSync{}, BEH ? ss : nullptr, BEH ? bm + i : nullptr);
  __syncthreads();
  uint4* out = reinterpret_cast<uint4*>(&core[i]);
  for (int k = tid; k < (int)(sizeof(MapCore) / 16); k += NT) out[k] = dst[k];
  PROF_MARK(P_STORE);
}

// 3 단계(판마다 레인 16, 블록 = 판 8): 지도 토큰. 지도 갱신이 끝난 뒤 모든 판. 레인이 광선·선분·칸을 나눠 하고 반 워프 동기
constexpr int TOK_NL = 16, TOK_EPB = 8;
struct HalfSync {
  unsigned mask;
  __device__ void operator()() const { __syncwarp(mask); }
  __device__ bool any(bool v) const { return __any_sync(mask, v); }   // 같은 판 레인끼리 하나라도(경유 지점 BFS 끝 판정)
};
template <bool BEH>
__global__ void __launch_bounds__(TOK_NL * TOK_EPB) map_tok_kernel(int N, const MapCore* core, const uint32_t* occ, const uint32_t* seen, const int16_t* segs,
                                                                   TPrev* tprev, MapTok* out, int tbug, const bsc::SceneSet* ss, BMapEnv* bm) {
  __shared__ TokScratch ts[TOK_EPB];
  const int sub = threadIdx.x / TOK_NL, lane = threadIdx.x % TOK_NL;
  const int i0 = blockIdx.x * TOK_EPB + sub;
  const bool live = i0 < N;
  const int i = live ? i0 : N - 1;   // 남는 레인도 같은 동기를 지나도록 마지막 판을 읽기만 함
  const HalfSync hs{0xffffu << (16 * (sub & 1))};
  PROF_START();
  const BCtx bx = bctx(BEH ? ss : nullptr, BEH ? bm + i : nullptr);
  make_tokens_n<TOK_NL>(core[i], occ + (size_t)i * NWORD, seen + (size_t)i * NWORD, segs + (size_t)i * SEGW, tprev + (size_t)i * KSLOT, ts[sub], lane, TOK_NL, live, hs, tbug,
                        &bx);
  hs();
  PROF_MARK(TK_ROOM);
  if (!live) return;
  const uint4* src = reinterpret_cast<const uint4*>(&ts[sub].out);
  uint4* dst = reinterpret_cast<uint4*>(out + i);
  for (int k = lane; k < (int)(sizeof(MapTok) / 16); k += TOK_NL) dst[k] = src[k];
}

// 4 단계(BEHAVIOR 판만, 판 하나 = 워프 하나): 다가가기 거리장(map.h 8 절). 레인 l 이 행 4l..4l+3 을 레지스터에 두고, 위·아래 이웃 행은 셔플로.
// 단계 집합은 nav_bfs_ref(CPU, 행 차례)와 같다(집합 연산이라 차례와 무관). 목표 확정(conf)은 매 스텝 쓴다
constexpr int NAV_WPB = 4;   // 블록 = 워프 4 = 판 4
__global__ void __launch_bounds__(32 * NAV_WPB) map_nav_kernel(env::Soa s, const MapCore* core, const uint32_t* occ, const BMapEnv* bm, uint8_t* lev, int* org,
                                                               int* tag, int* conf, const MapCurr* curr, int bug) {
  const int lane = threadIdx.x & 31, i = blockIdx.x * NAV_WPB + (threadIdx.x >> 5), N = s.N;
  if (i >= N || !bm[i].on) return;   // 워프 전체 같은 판
  const MapCore& m = core[i];
  const int ep = m.ep, t = m.t, tg = tag[i];
  if (lane == 0) conf[i] = m.n_task_conf;
  const MapCurr cu = *curr;
  if (!(tg != ep || t % nav_period(cu) == 0)) return;
  const float tx = s.f[env::F_TX * N + i], ty = s.f[env::F_TY * N + i];
  const float rx = s.f[env::F_X * N + i], ry = s.f[env::F_Y * N + i];
  const float rad = env::seed_r_of(s.iv[env::I_B_KIND * N + i]);
  const int K = nav_period(cu), M = nav_margin(K);
  const int rc = nav_cell(rx), rr = nav_cell(ry), pc0 = rc - NAV_PH, pr0 = rr - NAV_PH;
  const uint32_t* oc = occ + (size_t)i * NWORD;
  uint8_t* lv = lev + (size_t)i * (NAV_P * NAV_P);
  constexpr unsigned F = 0xffffffffu;
  auto sh_up = [&](R128 a) -> R128 {   // 위 레인(l−1)의 행 3 = 내 행 −1
    R128 o{__shfl_up_sync(F, a.lo, 1), __shfl_up_sync(F, a.hi, 1)};
    if (lane == 0) { o.lo = 0ull; o.hi = 0ull; }
    return o;
  };
  auto sh_dn = [&](R128 a) -> R128 {   // 아래 레인(l+1)의 행 0 = 내 행 +4
    R128 o{__shfl_down_sync(F, a.lo, 1), __shfl_down_sync(F, a.hi, 1)};
    if (lane == 31) { o.lo = 0ull; o.hi = 0ull; }
    return o;
  };
  reinterpret_cast<uint4*>(lv)[lane * 2] = make_uint4(~0u, ~0u, ~0u, ~0u);       // 조각 1 KB = 레인마다 32 B
  reinterpret_cast<uint4*>(lv)[lane * 2 + 1] = make_uint4(~0u, ~0u, ~0u, ~0u);
  __syncwarp();
  R128 blk[4], fr[4], vis[4], h[4], rob[4];
  for (int q = 0; q < 4; ++q) {
    const int r = 4 * lane + q;
    h[q] = r_h3(occ_row128(oc, r));
    rob[q] = (r >= rr - 2 && r <= rr + 2) ? r_cols(rc - 2, rc + 2) : R128{0ull, 0ull};
  }
  bool hit0 = false;
  {
    const R128 a = sh_up(h[3]), b = sh_dn(h[0]);
    for (int q = 0; q < 4; ++q) {
      R128 x = h[q];
      x = r_or(x, q > 0 ? h[q - 1] : a);
      x = r_or(x, q < 3 ? h[q + 1] : b);
      const R128 sd = nav_seed_row(4 * lane + q, tx, ty, rad);
      blk[q] = r_andn(x, sd);
      fr[q] = sd;
      vis[q] = sd;
      nav_write_row(lv, pc0, pr0, 4 * lane + q, sd, 0);
      hit0 = hit0 || r_any(r_and(sd, rob[q]));
    }
  }
  int Lr = __any_sync(F, hit0) ? 0 : -1;
  for (int L = 1; L <= NAV_LMAX && !(Lr >= 0 && L > Lr + M); ++L) {
    const bool eight = (L & 1) && bug != 5;   // bug 5: 음성 대조(늘 4 이웃)
    for (int q = 0; q < 4; ++q) h[q] = eight ? r_h3(fr[q]) : fr[q];
    const R128 a = sh_up(h[3]), b = sh_dn(h[0]);
    R128 nw[4];
    bool any = false, hitr = false;
    for (int q = 0; q < 4; ++q) {
      R128 x = eight ? h[q] : r_or(r_up(fr[q]), r_dn(fr[q]));
      x = r_or(x, q > 0 ? h[q - 1] : a);
      x = r_or(x, q < 3 ? h[q + 1] : b);
      nw[q] = r_andn(r_andn(x, blk[q]), vis[q]);
      any = any || r_any(nw[q]);
      hitr = hitr || r_any(r_and(nw[q], rob[q]));
    }
    if (!__any_sync(F, any)) break;
    if (Lr < 0 && __any_sync(F, hitr)) Lr = L;
    for (int q = 0; q < 4; ++q) { vis[q] = r_or(vis[q], nw[q]); fr[q] = nw[q]; nav_write_row(lv, pc0, pr0, 4 * lane + q, nw[q], L); }
  }
  __syncwarp();
  if (lane == 0) { tag[i] = ep; org[i] = nav_org(pc0, pr0); }
}

// 다시 시작(apply): 판 하나 = 블록 하나. 요청이 있으면 생성자와 같은 상태(배열 0, 거리장 0xff·판 번호 −1, init_core). 없으면 바로 끝남
struct MapBufs { MapCore* core; int16_t* L; uint32_t* seen; float* met; uint32_t* occ; int16_t* segs; TPrev* tprev; MapTok* tok; BMapEnv* bm; uint8_t* lev; int* navorg; int* navtag; int* navconf; uint8_t* view; };
__global__ void __launch_bounds__(256) map_apply_k(MapBufs b, int N, const MapCtl* ctl) {
  if (ctl->pend == 0) return;
  const int i = blockIdx.x, t = threadIdx.x;
  if (i >= N) return;
  auto z32 = [&](void* p, size_t bytes) {   // 4 B 낱말로 0(모든 배열이 4 B 정렬·크기)
    uint32_t* w = reinterpret_cast<uint32_t*>(p);
    for (size_t k = t; k < bytes / 4; k += 256) w[k] = 0u;
  };
  z32(b.L + (size_t)i * NCELL, sizeof(int16_t) * NCELL);
  z32(b.seen + (size_t)i * NWORD, sizeof(uint32_t) * NWORD);
  z32(b.occ + (size_t)i * NWORD, sizeof(uint32_t) * NWORD);
  z32(b.segs + (size_t)i * SEGW, sizeof(int16_t) * SEGW);
  z32(b.tprev + (size_t)i * KSLOT, sizeof(TPrev) * KSLOT);
  z32(b.tok + i, sizeof(MapTok));
  z32(b.view + (size_t)i * VIEW_BYTES, VIEW_BYTES);   // 본 곳 칸(나타남 판정)
  for (int k = t; k < N_MET; k += 256) b.met[(size_t)k * N + i] = 0.f;
  if (b.bm) {
    z32(b.bm + i, sizeof(BMapEnv));
    uint32_t* lv = reinterpret_cast<uint32_t*>(b.lev + (size_t)i * NAV_P * NAV_P);
    for (int k = t; k < NAV_P * NAV_P / 4; k += 256) lv[k] = 0xffffffffu;
    if (t == 0) { b.navorg[i] = 0; b.navtag[i] = -1; b.navconf[i] = 0; }
  }
  if (t == 0) init_core(b.core[i], ctl->seed, i);
}
static_assert(sizeof(int16_t) * NCELL % 4 == 0 && sizeof(int16_t) * SEGW % 4 == 0 && sizeof(TPrev) % 4 == 0 && sizeof(MapTok) % 4 == 0 && sizeof(BMapEnv) % 4 == 0 &&
                  NAV_P * NAV_P % 4 == 0 && VIEW_BYTES % 4 == 0, "map_apply_k clears 4-byte words");
__global__ void map_commit_k(MapCtl* ctl) { ctl->pend = 0; }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceMap::DeviceMap(int N, uint64_t seed, const bsc::SceneSet* ss_dev) : N_(N), ss_(ss_dev) {
  if (ss_dev) {
    CK(cudaMalloc(&bm_, sizeof(BMapEnv) * (size_t)N));
    CK(cudaMemset(bm_, 0, sizeof(BMapEnv) * (size_t)N));
    CK(cudaMalloc(&lev_, (size_t)NAV_P * NAV_P * N));
    CK(cudaMemset(lev_, 0xff, (size_t)NAV_P * NAV_P * N));
    CK(cudaMalloc(&navorg_, sizeof(int) * (size_t)N));
    CK(cudaMemset(navorg_, 0, sizeof(int) * (size_t)N));
    CK(cudaMalloc(&navtag_, sizeof(int) * (size_t)N));
    CK(cudaMemset(navtag_, 0xff, sizeof(int) * (size_t)N));   // −1 = 거리장 없음
    CK(cudaMalloc(&navconf_, sizeof(int) * (size_t)N));
    CK(cudaMemset(navconf_, 0, sizeof(int) * (size_t)N));
  }
  CK(cudaMalloc(&core_, sizeof(MapCore) * (size_t)N));
  CK(cudaMalloc(&view_, (size_t)VIEW_BYTES * N));
  CK(cudaMemset(view_, 0, (size_t)VIEW_BYTES * N));
  CK(cudaMalloc(&L_, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMalloc(&seen_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&met_, sizeof(float) * N_MET * (size_t)N));
  CK(cudaMalloc(&occ_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&segs_, sizeof(int16_t) * SEGW * (size_t)N));
  CK(cudaMalloc(&tprev_, sizeof(TPrev) * KSLOT * (size_t)N));
  CK(cudaMalloc(&tok_, sizeof(MapTok) * (size_t)N));
  CK(cudaMemset(occ_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(segs_, 0, sizeof(int16_t) * SEGW * (size_t)N));
  CK(cudaMemset(tprev_, 0, sizeof(TPrev) * KSLOT * (size_t)N));
  CK(cudaMemset(tok_, 0, sizeof(MapTok) * (size_t)N));
  CK(cudaMalloc(&list_, sizeof(uint32_t) * (size_t)N));
  CK(cudaMalloc(&count_, sizeof(int)));
  CK(cudaMalloc(&curr_, sizeof(MapCurr)));
  CK(cudaMalloc(&ctl_, sizeof(MapCtl)));
  CK(cudaMemset(ctl_, 0, sizeof(MapCtl)));
  CK(cudaMemcpy(curr_, &kCurrEmpty, sizeof(MapCurr), cudaMemcpyHostToDevice));
  CK(cudaMemset(L_, 0, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMemset(seen_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(met_, 0, sizeof(float) * N_MET * (size_t)N));
  map_init_kernel<<<(N + NT - 1) / NT, NT>>>(core_, N, seed);
  CK(cudaGetLastError());
}
void DeviceMap::request_reset(uint64_t seed) {
  if (!ctl_h_) {
    CK(cudaHostAlloc(&ctl_h_, sizeof(MapCtl) * 8, cudaHostAllocDefault));
    for (auto& e : ctl_ev_) { cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)); e = ev; }
  }
  const int k = ctl_slot_++ % 8;
  cudaEvent_t ev = static_cast<cudaEvent_t>(ctl_ev_[k]);
  if (cudaEventQuery(ev) == cudaErrorNotReady) CK(cudaEventSynchronize(ev));
  ctl_h_[k] = MapCtl{1, 0, (unsigned long long)seed};
  CK(cudaMemcpyAsync(ctl_, &ctl_h_[k], sizeof(MapCtl), cudaMemcpyHostToDevice, 0));
  CK(cudaEventRecord(ev, 0));
}
void DeviceMap::apply() {
  const MapBufs b{core_, L_, seen_, met_, occ_, segs_, tprev_, tok_, bm_, lev_, navorg_, navtag_, navconf_, view_};
  map_apply_k<<<N_, 256>>>(b, N_, ctl_);
  map_commit_k<<<1, 1>>>(ctl_);
}

DeviceMap::~DeviceMap() { cudaFree(ctl_); if (ctl_h_) { for (void* e : ctl_ev_) if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e)); cudaFreeHost(ctl_h_); }
                         cudaFree(core_); cudaFree(L_); cudaFree(seen_); cudaFree(met_); cudaFree(list_); cudaFree(count_);
                         cudaFree(occ_); cudaFree(segs_); cudaFree(tprev_); cudaFree(tok_); cudaFree(curr_); cudaFree(view_); if (bm_) { cudaFree(bm_); cudaFree(lev_); cudaFree(navorg_); cudaFree(navtag_); cudaFree(navconf_); } }

size_t DeviceMap::bytes() const {
  return (size_t)N_ * (sizeof(MapCore) + sizeof(int16_t) * NCELL + 2 * sizeof(uint32_t) * NWORD + sizeof(float) * N_MET + sizeof(int16_t) * SEGW +
                       sizeof(TPrev) * KSLOT + sizeof(MapTok) + sizeof(uint32_t) + VIEW_BYTES + (ss_ ? sizeof(BMapEnv) + NAV_P * NAV_P + 3 * sizeof(int) : 0));
}

void DeviceMap::step(const env::Soa& s, int force_kf, int bug, cudaStream_t st, MapTok* tok, const MapCurr* curr) {
  // 목록 길이를 0 으로(비동기, 그래프로 잡힘) → 시작 커널(판마다 스레드) → keyframe 커널(목록의 판만 일함)
  CK(cudaMemsetAsync(count_, 0, sizeof(int), st));
  const int kbug = (bug == 1 || bug >= 6) ? bug : 0;   // 지도 갱신 음성 대조(1 확정 규칙, 6–10 바뀜 판정 규칙). 2–4 토큰, 5 거리장
  if (ss_) {
    map_begin_kernel<true><<<(N_ + BEGIN_NT - 1) / BEGIN_NT, BEGIN_NT, 0, st>>>(s, core_, met_, list_, count_, force_kf, ss_, bm_);
    map_kf_kernel<true><<<N_, NT, 0, st>>>(s, core_, L_, seen_, occ_, segs_, met_, list_, count_, kbug, curr ? curr : curr_, ss_, bm_, view_);
  } else {
    map_begin_kernel<false><<<(N_ + BEGIN_NT - 1) / BEGIN_NT, BEGIN_NT, 0, st>>>(s, core_, met_, list_, count_, force_kf, nullptr, nullptr);
    map_kf_kernel<false><<<N_, NT, 0, st>>>(s, core_, L_, seen_, occ_, segs_, met_, list_, count_, kbug, curr ? curr : curr_, nullptr, nullptr, view_);
  }
  if (ss_ && nav_on_)
    map_nav_kernel<<<(N_ + NAV_WPB - 1) / NAV_WPB, 32 * NAV_WPB, 0, st>>>(s, core_, occ_, bm_, lev_, navorg_, navtag_, navconf_, curr ? curr : curr_, bug == 5 ? 5 : 0);
  if (tok_on_) {
    if (ss_) map_tok_kernel<true><<<(N_ + TOK_EPB - 1) / TOK_EPB, TOK_NL * TOK_EPB, 0, st>>>(N_, core_, occ_, seen_, segs_, tprev_, tok ? tok : tok_, bug >= 2 ? bug : 0, ss_, bm_);
    else map_tok_kernel<false><<<(N_ + TOK_EPB - 1) / TOK_EPB, TOK_NL * TOK_EPB, 0, st>>>(N_, core_, occ_, seen_, segs_, tprev_, tok ? tok : tok_, bug >= 2 ? bug : 0, nullptr, nullptr);
  }
}

// 위에서 본 지도: 판 하나 = 블록 하나(128 스레드가 비트를 옮김, 스레드 0 이 입력)
__global__ void __launch_bounds__(128) tv_state_kernel(int N, const MapCore* core, const uint32_t* occ, const uint32_t* seen, const bsc::SceneSet* ss, BMapEnv* bm, TopState* out) {
  const int i = blockIdx.x;
  if (i >= N) return;
  const BCtx bx = bctx(ss, bm ? bm + i : nullptr);
  tv_topstate(core[i], &bx, occ + (size_t)i * NWORD, seen + (size_t)i * NWORD, out[i], threadIdx.x, blockDim.x);
}
void DeviceMap::topstate(TopState* out, cudaStream_t st) const {
  tv_state_kernel<<<N_, 128, 0, st>>>(N_, core_, occ_, seen_, ss_, bm_, out);
}
// RGB: 블록 = (그림, 행 16 줄), 스레드 256 = 열. 입력(TvIn 352 B)은 공유 메모리로, 비트는 전역(캐시)에서
constexpr int TV_ROWS = 16;
__global__ void __launch_bounds__(TV_PX) tv_render_kernel(const TopState* ts, const int* rows, const uint8_t* hide, uint8_t* rgb, int bug) {
  const int k = blockIdx.x, v0 = blockIdx.y * TV_ROWS, u = threadIdx.x;
  const TopState& t = ts[rows ? rows[k] : k];
  __shared__ TvIn in;
  if (u < (int)(sizeof(TvIn) / 4)) reinterpret_cast<uint32_t*>(&in)[u] = reinterpret_cast<const uint32_t*>(&t.in)[u];
  __syncthreads();
  const bool hd = hide && hide[k];
  const TvWinCell cell{t.occ, t.seen};
  for (int v = v0; v < v0 + TV_ROWS; ++v) {
    uint8_t c[3];
    tv_color(bug == 1 ? tv_class(in, cell, v, u, true, hd) : tv_class(in, cell, u, v, true, hd), c);
    uint8_t* p = rgb + (((size_t)k * TV_PX + v) * TV_PX + u) * 3;
    p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
  }
}
static_assert(sizeof(TvIn) % 4 == 0 && sizeof(TvIn) / 4 <= TV_PX, "TvIn copy by one row of threads");
void tv_render(const TopState* ts, const int* rows, int n, const uint8_t* hide, uint8_t* rgb, cudaStream_t st, int bug) {
  if (n <= 0) return;
  tv_render_kernel<<<dim3(n, TV_PX / TV_ROWS), TV_PX, 0, st>>>(ts, rows, hide, rgb, bug);
}

void prof_reset() {
#ifdef MAP_PROF
  const unsigned long long z[P_NSEC + 3] = {};
  CK(cudaMemcpyToSymbol(g_prof, z, sizeof z));
#endif
}
void prof_read(unsigned long long out[P_NSEC + 3]) {
  for (int k = 0; k < P_NSEC + 3; ++k) out[k] = 0;
#ifdef MAP_PROF
  CK(cudaMemcpyFromSymbol(out, g_prof, sizeof(unsigned long long) * (P_NSEC + 3)));
#endif
}

void DeviceMap::download(MapHost& h, const MapTok* tok) const {
  h.core.resize(N_); h.L.resize((size_t)NCELL * N_); h.seen.resize((size_t)NWORD * N_); h.met.resize((size_t)N_MET * N_);
  h.occ.resize((size_t)NWORD * N_); h.segs.resize((size_t)SEGW * N_); h.tprev.resize((size_t)KSLOT * N_); h.tok.resize(N_);
  CK(cudaMemcpy(h.occ.data(), occ_, sizeof(uint32_t) * h.occ.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.segs.data(), segs_, sizeof(int16_t) * h.segs.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.tprev.data(), tprev_, sizeof(TPrev) * h.tprev.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.tok.data(), tok ? tok : tok_, sizeof(MapTok) * h.tok.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.core.data(), core_, sizeof(MapCore) * h.core.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.L.data(), L_, sizeof(int16_t) * h.L.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.seen.data(), seen_, sizeof(uint32_t) * h.seen.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.met.data(), met_, sizeof(float) * h.met.size(), cudaMemcpyDeviceToHost));
  h.view.resize((size_t)VIEW_BYTES * N_);
  CK(cudaMemcpy(h.view.data(), view_, h.view.size(), cudaMemcpyDeviceToHost));
  if (bm_) {
    h.bm.resize(N_);
    CK(cudaMemcpy(h.bm.data(), bm_, sizeof(BMapEnv) * h.bm.size(), cudaMemcpyDeviceToHost));
    h.lev.resize((size_t)NAV_P * NAV_P * N_); h.navorg.resize(N_); h.navtag.resize(N_); h.navconf.resize(N_);
    CK(cudaMemcpy(h.lev.data(), lev_, h.lev.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.navorg.data(), navorg_, sizeof(int) * N_, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.navtag.data(), navtag_, sizeof(int) * N_, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.navconf.data(), navconf_, sizeof(int) * N_, cudaMemcpyDeviceToHost));
  }
}

TokenRecorder::TokenRecorder(int N, int T) : N_(N), T_(T) {
  CK(cudaMalloc(&buf_, bytes()));
  CK(cudaMemset(buf_, 0, bytes()));
}
TokenRecorder::~TokenRecorder() { cudaFree(buf_); }
void TokenRecorder::download(std::vector<MapTok>& out) const {
  out.resize((size_t)N_ * T_);
  CK(cudaMemcpy(out.data(), buf_, bytes(), cudaMemcpyDeviceToHost));
}

}  // namespace gmap
