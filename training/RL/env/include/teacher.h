// 대본 특권 교사(E6, 2026-10-05; 계획·따르기로 다시 씀): 잡기 물리 판(B4 집기·B5 놓기·B6 가져오기)의 결정적 시연 정책. 장치에서(스레드 하나 = 판 하나),
// CPU 참조판과 같은 소스. 참 물체 자세·크기·무게(특권, 지도에 있는 물체만 — B6 은 지도 확정 전에는 물체 자리를 안 씀: POLICY 4.2)와
// OMX-F 역기구학(grasp.h)·같은 충돌 함수(env_pnp.h)로 계획한다.
//
// 비싼 계획은 필요할 때만(판 시작·단계 바뀜·다시 계획 신호: 막힘·떨어뜨림·물체 움직임·팔 계획 실패) 하고, 그 밖의 스텝은 저장한 계획을 따르기만 한다:
//   teacher_pre (판마다, 가벼움: 판 시작·사건 처리, 계획 요청 비트) → 요청한 판만 모은 목록 → teacher_plan (목록의 판만, 무거움) → teacher_act (판마다, 가벼움: 행동)
// GPU 는 셋을 따로 커널로(목록 = 장치 원자 더하기, 호스트 동기 없음 — 그래프에 넣을 수 있음), CPU 참조판은 판마다 차례로 같은 셋. 판끼리 서로 안 봄 → 비트 같음.
// 2026-10-06 빠르게(결과 성공률 그대로, CURRICULUM 5.7): 계획 = 판 하나에 블록 하나(T_CHUNK 128 레인), 정적 점유는 짝마다 미리 칠한 표(SceneSet::tocc),
// 서는 자리 찾기 이어 하는 조각은 BFS 없이 첫 조각의 닿는 칸 비트(TBuf::rb), 등급 1·2 를 찾은 조각에서 끝냄, 들고 다닐 때 팔 검사 캐시(CarryCache)·
// 지역 계획 비용 차례 검사. tch_extract 의 (+1, +1) 이웃 버그 고침(헛 NOPATH → 다시 찾기).
//
// 계획(모두 이 파일):
//   서는 자리 찾기(stance_search): 목표(물체 또는 놓을 가운데) 둘레 원호 — 방향 α 36(10°), 팔 방향 β 7(0, ±0.45, ±0.9, ±1.3 rad),
//     joint1 축 거리 r(역기구학 띠에서 0.01 m 간격 — 목표 높이·기울기·팔꿈치마다 미리 훑음)에서 베이스 자세를 만들고, 싼 검사(창 안·설 칸 성분·몸통 안 닿음·
//     앞 물러난 자리 P(0.25/0.40/0.15 m 뒤)가 닿는 칸·P → 자리 직선 몸통 안 닿음·P 에서 제자리 돌기(엄격 단계))를 지난 것만 팔 계획(아래) + 들어가는 길에서
//     팔(잡기 전 자세)이 안 닿음 + (엄격 단계) 자리 ±1.5 cm·±0.03 rad 에서도 팔 계획이 됨. 비용 구간(이동 거리 + 0.1·|β|) 차례로, 처음 되는 것.
//   잡기 계획(arm_try_grasp): 위에서 잡기(기울기 −90°…−56°) / 옆 잡기(−26°…+17°) 중 물체 윗면 ≤ 0.30 m 면 위에서 먼저, 닫는 축 = 물체 좁은 가로 축
//     (위에서: roll 로 맞춤, 다른 축도 해 봄; 옆: roll 0, 팔 방향이 고름), 폭 ≤ max_w, 손가락 사이(grasp_width), 무게, 잡기·잡기 전(6 cm)·가운데(3 cm)·
//     들기(3·7 cm, 든 물체 포함) 웨이포인트마다 역기구학 + 충돌.
//   놓기 계획(arm_try_place): 든 물체 가운데가 놓을 점(막는 물체가 있으면 옆 빈 자리, 안 되면 둘레 4 점)에 오게, 놓은 뒤 내려앉힌 자리에서 판정(at_goal)이 참,
//     떨어진 높이 ≤ 5 cm(용기 빼고), 놓기 전(6 cm 위)·가운데·놓기·물러나기(6 cm 뒤 + 3 cm 위, 손 ≥ 5.5 cm) 역기구학 + 충돌 + 무게.
//   길(tch_bfs): 창 128 × 128 칸(0.1 m) 특권 점유 — 설 칸 성분(중심 0.13 m 안 충돌 상자 없음, 바닥 높이 문턱) + 과제 물체·집을 물체·막는 물체(0.13 m 부풀림),
//     그리고 더 넓은 판(정적 상자·과제 물체 0.20 m 부풀림 — 제자리 돌기 여유)을 로봇 칸에서 BFS(8·4 이웃 번갈아). 넓은 판에서 닿으면 그것, 아니면 좁은 판.
//     목표에서 거꾸로 내려와 칸 열 → 직선이 빈 칸만 지나는 가장 먼 칸으로 줄인 웨이포인트 ≤ 12.
//   따르기: 웨이포인트 순수 추종(크게 틀어지면 제자리 돌기), P 에서 자리 yaw 로 돌고 팔을 잡기 전 자세로 → 자리까지 곧게(1 cm·0.01 rad) → 그 자리 팔 계획 →
//     웨이포인트 차례(잡기 전 → 가운데 → 잡기 → 닫기 → 들기). B5·B6 는 들고 P 로 곧게 물러나 나르는 자세로 접고 놓을 곳 쪽으로 같은 순서.
//   다시 하기: 막힘(40 스텝에 남은 길 5 cm 안 줄면 길 다시, 3 번이면 자리 다시), 떨어뜨림(물체 참 자리로 다시 잡기), 닫아도 안 잡힘(팔 다시), 시도 > max_try 면 포기(까닭 적음).
//   B6 이고 지도가 붙었는데 목표가 지도에 아직 확정 안 됨: 탐사(창 1.6 m 격자점 중 안 가 본 가장 가까운 곳으로, 가서 한 바퀴) — 물체 자리를 쓰지 않음.
#pragma once
#include <cstdlib>
#include <cstdio>
#include "env_soa.h"

namespace env {

#if defined(ENV_PROF) && defined(__CUDACC__)
__device__ unsigned long long g_tdbgd[32];   // 측정 빌드(장치): 계획 거절 까닭 수(TDBG)
#endif
#if defined(TEACH_DBG) && !defined(__CUDA_ARCH__)
extern long g_tdbg[32];   // 진단 빌드만(호스트): 계획 거절 까닭 수
#define TDBG(k) (++g_tdbg[k])
#elif defined(ENV_PROF) && defined(__CUDA_ARCH__)
#define TDBG(k) atomicAdd(&g_tdbgd[k], 1ull)
#else
#define TDBG(k) ((void)0)
#endif
// 계획 구간 시간(측정 빌드 -DENV_PROF 만, 장치 레인 0): g_tseg[구간][수, 사이클]
#if defined(ENV_PROF) && defined(__CUDACC__)
__device__ unsigned long long g_tseg[16][2];
__device__ unsigned long long g_tcause[2][16];   // 새 서는 자리 찾기의 까닭(t.fail) — [잡기/놓기][까닭]
__device__ unsigned long long g_tbin[3][8];      // 찾은 서는 자리: [등급][비용 구간 0..6, 7 = 바로 가는 자리]
__device__ unsigned long long g_tact[16][3];     // 행동 커널: 단계마다 [수, 사이클 합, 최대]
__device__ unsigned long long g_tkc[3][5];       // 놓기 서는 자리 조각 결과: [없음·찾음·이어서][놓을 가운데 후보]
#endif
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
#define TBIN(g, k) do { if (w.lane == 0) atomicAdd(&g_tbin[g][k], 1ull); } while (0)
#else
#define TBIN(g, k) ((void)0)
#endif
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
#define TSEG_INIT long long ts_ = clock64();
#define TSEG(k) do { const long long tn_ = clock64(); if (w.lane == 0) { atomicAdd(&g_tseg[k][0], 1ull); atomicAdd(&g_tseg[k][1], (unsigned long long)(tn_ - ts_)); } ts_ = tn_; } while (0)
#else
#define TSEG_INIT
#define TSEG(k) ((void)0)
#endif
#if defined(__CUDACC__)
#define TDEV static __host__ __device__ __noinline__
#else
#define TDEV static inline
#endif

enum TPhase {
  TP_NAV = 0, TP_APP = 1, TP_PRE = 2, TP_DOWN = 3, TP_CLOSE = 4, TP_LIFT = 5, TP_BACK = 6, TP_FOLD = 7,
  TP_NAV2 = 8, TP_APP2 = 9, TP_PREPL = 10, TP_OPEN = 11, TP_RETREAT = 12, TP_DONE = 13, TP_EXPLORE = 14
};
struct KT {
  static constexpr int plan_budget = 192;   // 한 스텝에 서는 자리 후보(싼 검사를 지난 것) 이만큼까지(넘으면 다음 스텝에 이어서 — 한 스텝 계획 시간 상한, 로봇은 멈춰 기다림)
  static constexpr float pre_d = 0.06f;        // 잡기 전·놓기 전 거리(다가가는 축·위로)
  static constexpr float lift1 = 0.03f, lift2 = 0.07f;   // 들기 웨이포인트(잡는 점 위로) — B4 성공 0.05 m 보다 위
  static constexpr float open_extra = 0.02f;   // 벌림 = 물체 폭 + 2 cm(행동 상한 0.6 rad)
  static constexpr float q_tol = 0.03f, q_tol_fine = 0.012f;
  static constexpr int t_pre = 40, t_wp = 20, t_close = 15, t_lift = 25, t_open = 15, t_retreat = 25, t_app = 220, max_try = 6;
  static constexpr float topdown_top = 0.30f;  // 물체 윗면이 이 아래면 위에서 잡기 먼저(E0: 위에서 잡기 잡는 점 ≤ 0.25 m)
  static constexpr float nav_v = 0.35f, app_v = 0.12f, wp_r = 0.12f, end_r = 0.05f;   // end_r: 마지막 점(P) 이 안이면 다가가기 단계로(거기서 3 cm 까지)
  static constexpr float arrive = 0.0015f, arrive_yaw = 0.004f;
  static constexpr int stuck_T = 40, max_stk = 3;
  static constexpr float stuck_dd = 0.05f;
  static constexpr float r_nav0 = 0.13f, r_nav1 = 0.20f;   // 길 칸 부풀림: 좁은 판(곧게 지남, = bscene_host R_FREE), 넓은 판(제자리 돌기, 몸통 외접원 0.194)
  static constexpr float r_esc = 0.35f;        // 로봇 둘레 이 안은 좁은 판으로(가구 옆에서 빠져나오기)
  static constexpr float exp_sp = 1.6f;        // 탐사 격자 간격
};
constexpr int T_NWP = 12;   // 웨이포인트 수
constexpr int T_NAQ = 5;    // 팔 웨이포인트 수(잡기: 잡기 전·가운데·잡기·들기 1·들기 2, 놓기: 놓기 전·가운데·놓기·물러나기)
// 교사 기억(환경 상태 밖 — 장치 버퍼, 판 N 개 열 배치 f[k*N + i]). 환경 상태의 I_T_EP·PH·TM·TRY·SOK, F_T_SX·SY·SYAW·D 도 씀
enum TFl { TF_CPHI, TF_PX, TF_PY, TF_DPRE, TF_OX, TF_OY, TF_OZ, TF_OPEN, TF_W, TF_TCX, TF_TCY, TF_TCZ, TF_AX, TF_AY, TF_AYAW, TF_EXY,
           TF_WP0, TF_Q0 = TF_WP0 + 2 * T_NWP, NTF = TF_Q0 + 5 * T_NAQ };
enum TIn { TI_SKC, TI_SPOS, TI_SG1, TI_SG2, TI_CFG, TI_NEED, TI_NWP, TI_KWP, TI_NAQ, TI_KAQ, TI_FAIL, TI_STK, TI_PART, TI_EXP0, TI_EXP1, TI_HELD0, TI_PH0, TI_NPLAN, TI_SNP, NTI };
enum TNeed { TN_SG = 1, TN_SP = 2, TN_AG = 4, TN_AP = 8, TN_PATH = 16, TN_EXPL = 32, TN_CONT = 64 };   // CONT: 서는 자리 찾기를 이어서(진행 SPos 그대로)
constexpr int T_SCR = 2 * bsc::WIN * bsc::WIN + 5 * bsc::WIN * 16;   // 계획 한 판의 작업 메모리(바이트): 단계 칸 둘(넓은·좁은 판) + 128 비트 행 5 묶음
struct TBuf {
  float* f;          // NTF * N
  int* iv;           // NTI * N
  uint8_t* scr;      // nslot * T_SCR
  int* list;         // [N + 1]: [0] = 수, [1..] = 계획할 판(GPU 원자 더하기 — 차례는 결과와 무관)
  int N, nslot;
  uint32_t* rb;      // [N][T_RBW] 서는 자리 찾기 첫 조각의 닿는 칸 비트(3 × 3 이웃 넓힘 = lev_near) — 이어 하는 조각은 BFS 없이 이것(로봇은 멈춰 기다림)
};
constexpr int T_RBW = bsc::WIN * bsc::WIN / 32;

// 나눠 하기(계획 커널 = 판 하나에 블록 하나, 2026-10-06 — 예전 워프 하나): 레인마다 후보를 하나씩 보고, "처음 되는 후보" 는 레인들 중 가장 작은 번호
// (= CPU 차례대로 처음 되는 것과 같음). nl == 32 는 워프(잡기 가능 표 feas_k — 블록에 짝 넷), nl > 32 는 블록(sh = 블록 공유 int ≥ 16).
// CPU 참조판·후보 평가 안쪽은 레인 하나(nl 1) — 같은 함수, 같은 결과. 서는 자리 예산은 T_CHUNK 묶음마다 셈(CPU·GPU 같은 곳에서 끊김)
struct WCtx { int lane, nl; int* sh; };
constexpr int T_CHUNK = 128;   // 계획 커널 블록 크기 = 예산 묶음
DEV WCtx wseq() { return WCtx{0, 1, nullptr}; }
DEV void w_sync(const WCtx& w) {
#ifdef __CUDA_ARCH__
  if (w.nl > 32) __syncthreads();
  else if (w.nl > 1) __syncwarp();
#endif
  (void)w;
}
DEV bool w_any(const WCtx& w, bool v) {
#ifdef __CUDA_ARCH__
  if (w.nl > 32) return __syncthreads_or(v ? 1 : 0) != 0;
  if (w.nl > 1) return __any_sync(0xffffffffu, v);
#endif
  (void)w;
  return v;
}
template <class F>
DEV int w_first(const WCtx& w, int n, const F& f) {
#ifdef __CUDA_ARCH__
  if (w.nl > 32) {
    for (int base = 0; base < n; base += w.nl) {
      const int k = base + w.lane;
      const bool ok = k < n && f(k);
      __syncthreads();
      if (w.lane == 0) w.sh[0] = 0x7fffffff;
      __syncthreads();
      if (ok) atomicMin(&w.sh[0], k);
      __syncthreads();
      const int m = w.sh[0];
      if (m != 0x7fffffff) return m;
    }
    return -1;
  }
  if (w.nl > 1) {
    for (int base = 0; base < n; base += 32) {
      const int k = base + w.lane;
      const bool ok = k < n && f(k);
      const unsigned m = __ballot_sync(0xffffffffu, ok);
      if (m) return base + __ffs((int)m) - 1;
    }
    return -1;
  }
#endif
  (void)w;
  for (int k = 0; k < n; ++k) if (f(k)) return k;
  return -1;
}
// 가장 작은 값(f 가 1e30 이상이면 없음), 같으면 앞 번호
template <class F>
DEV int w_argmin(const WCtx& w, int n, const F& f) {
  float best = 1e30f;
  int bi = -1;
  for (int k = w.lane; k < n; k += w.nl) {
    const float v = f(k);
    if (v < best) { best = v; bi = k; }
  }
#ifdef __CUDA_ARCH__
  if (w.nl > 1)
    for (int off = 16; off; off >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (ob < best || (ob == best && oi >= 0 && (bi < 0 || oi < bi))) { best = ob; bi = oi; }
    }
  if (w.nl > 32) {   // 워프마다 (값, 번호) → 레인 0 이 워프 차례로 합침
    __syncthreads();
    if ((w.lane & 31) == 0) { w.sh[2 * (w.lane >> 5)] = __float_as_int(best); w.sh[2 * (w.lane >> 5) + 1] = bi; }
    __syncthreads();
    if (w.lane == 0) {
      for (int q = 1; q < (w.nl >> 5); ++q) {
        const float ob = __int_as_float(w.sh[2 * q]);
        const int oi = w.sh[2 * q + 1];
        if (ob < best || (ob == best && oi >= 0 && (bi < 0 || oi < bi))) { best = ob; bi = oi; }
      }
      w.sh[0] = __float_as_int(best); w.sh[1] = bi;
    }
    __syncthreads();
    best = __int_as_float(w.sh[0]); bi = w.sh[1];
    __syncthreads();
  }
#endif
  return best < 1e30f ? bi : -1;
}

struct TState {
  int ep, ph, tm, tr, sok;
  float sx, sy, syaw, sd;
  int skc, spos, sg1, sg2, cfg, need, nwp, kwp, naq, kaq, fail, stk, part, exp0, exp1, held0, ph0, nplan, snp;   // snp: 이 서는 자리 찾기의 조각 수
  float cphi, px, py, dpre, ox, oy, oz, open, w, tcx, tcy, tcz, ax, ay, ayaw, exy;
  float wp[2 * T_NWP], q[5 * T_NAQ];
};
DEV void load_t(const Soa& s, const TBuf& tb, int i, TState& t) {
  const int N = s.N;
  t.ep = s.iv[I_T_EP * N + i]; t.ph = s.iv[I_T_PH * N + i]; t.tm = s.iv[I_T_TM * N + i]; t.tr = s.iv[I_T_TRY * N + i]; t.sok = s.iv[I_T_SOK * N + i];
  t.sx = s.f[F_T_SX * N + i]; t.sy = s.f[F_T_SY * N + i]; t.syaw = s.f[F_T_SYAW * N + i]; t.sd = s.f[F_T_D * N + i];
  const int* iv = tb.iv;
  t.skc = iv[TI_SKC * N + i]; t.spos = iv[TI_SPOS * N + i]; t.sg1 = iv[TI_SG1 * N + i]; t.sg2 = iv[TI_SG2 * N + i];
  t.cfg = iv[TI_CFG * N + i]; t.cphi = tb.f[TF_CPHI * N + i];
  t.need = iv[TI_NEED * N + i]; t.nwp = iv[TI_NWP * N + i]; t.kwp = iv[TI_KWP * N + i]; t.naq = iv[TI_NAQ * N + i]; t.kaq = iv[TI_KAQ * N + i];
  t.fail = iv[TI_FAIL * N + i]; t.stk = iv[TI_STK * N + i]; t.part = iv[TI_PART * N + i]; t.exp0 = iv[TI_EXP0 * N + i]; t.exp1 = iv[TI_EXP1 * N + i];
  t.held0 = iv[TI_HELD0 * N + i]; t.ph0 = iv[TI_PH0 * N + i]; t.nplan = iv[TI_NPLAN * N + i]; t.snp = iv[TI_SNP * N + i];
  const float* f = tb.f;
  t.px = f[TF_PX * N + i]; t.py = f[TF_PY * N + i]; t.dpre = f[TF_DPRE * N + i]; t.ox = f[TF_OX * N + i]; t.oy = f[TF_OY * N + i]; t.oz = f[TF_OZ * N + i];
  t.open = f[TF_OPEN * N + i]; t.w = f[TF_W * N + i]; t.tcx = f[TF_TCX * N + i]; t.tcy = f[TF_TCY * N + i]; t.tcz = f[TF_TCZ * N + i];
  t.ax = f[TF_AX * N + i]; t.ay = f[TF_AY * N + i]; t.ayaw = f[TF_AYAW * N + i]; t.exy = f[TF_EXY * N + i];
  for (int k = 0; k < 2 * T_NWP; ++k) t.wp[k] = f[(TF_WP0 + k) * N + i];
  for (int k = 0; k < 5 * T_NAQ; ++k) t.q[k] = f[(TF_Q0 + k) * N + i];
}
DEV void store_t(const Soa& s, const TBuf& tb, int i, const TState& t) {
  const int N = s.N;
  s.iv[I_T_EP * N + i] = t.ep; s.iv[I_T_PH * N + i] = t.ph; s.iv[I_T_TM * N + i] = t.tm; s.iv[I_T_TRY * N + i] = t.tr; s.iv[I_T_SOK * N + i] = t.sok;
  s.f[F_T_SX * N + i] = t.sx; s.f[F_T_SY * N + i] = t.sy; s.f[F_T_SYAW * N + i] = t.syaw; s.f[F_T_D * N + i] = t.sd;
  int* iv = tb.iv;
  iv[TI_SKC * N + i] = t.skc; iv[TI_SPOS * N + i] = t.spos; iv[TI_SG1 * N + i] = t.sg1; iv[TI_SG2 * N + i] = t.sg2;
  iv[TI_CFG * N + i] = t.cfg; tb.f[TF_CPHI * N + i] = t.cphi;
  iv[TI_NEED * N + i] = t.need; iv[TI_NWP * N + i] = t.nwp; iv[TI_KWP * N + i] = t.kwp; iv[TI_NAQ * N + i] = t.naq; iv[TI_KAQ * N + i] = t.kaq;
  iv[TI_FAIL * N + i] = t.fail; iv[TI_STK * N + i] = t.stk; iv[TI_PART * N + i] = t.part; iv[TI_EXP0 * N + i] = t.exp0; iv[TI_EXP1 * N + i] = t.exp1;
  iv[TI_HELD0 * N + i] = t.held0; iv[TI_PH0 * N + i] = t.ph0; iv[TI_NPLAN * N + i] = t.nplan; iv[TI_SNP * N + i] = t.snp;
  float* f = tb.f;
  f[TF_PX * N + i] = t.px; f[TF_PY * N + i] = t.py; f[TF_DPRE * N + i] = t.dpre; f[TF_OX * N + i] = t.ox; f[TF_OY * N + i] = t.oy; f[TF_OZ * N + i] = t.oz;
  f[TF_OPEN * N + i] = t.open; f[TF_W * N + i] = t.w; f[TF_TCX * N + i] = t.tcx; f[TF_TCY * N + i] = t.tcy; f[TF_TCZ * N + i] = t.tcz;
  f[TF_AX * N + i] = t.ax; f[TF_AY * N + i] = t.ay; f[TF_AYAW * N + i] = t.ayaw; f[TF_EXY * N + i] = t.exy;
  for (int k = 0; k < 2 * T_NWP; ++k) f[(TF_WP0 + k) * N + i] = t.wp[k];
  for (int k = 0; k < 5 * T_NAQ; ++k) f[(TF_Q0 + k) * N + i] = t.q[k];
}

// ---- 팔 자세 ----
// 접은 자세(빈손으로 다닐 때): 팔 점이 모두 차대 앞 가장자리(0.16 m) 뒤(앞 끝 base_link x 0.023 m), 맨 위 세계 0.60 m, 관절 한계 5° 밖(OF_LIMIT 안 켬) —
// 행동 범위 안에서 앞으로 가장 덜 나오는 자세(0.05 rad 격자 훑기) (가정: 높은 가구 밑은 피함)
DEV void tuck_q(float q[5]) { q[0] = 0.f; q[1] = 0.10f; q[2] = -2.00f; q[3] = -0.5f; q[4] = 0.f; }
// cfg: 이 계획의 자세(팔꿈치 | roll 고름 << 1 | 위에서 << 2 | 놓을 가운데 후보 << 3), cphi = 기울기 — 자리에 와서 같은 자세를 먼저 해 봄
struct ArmPlan { float q[T_NAQ][5]; int n; float open, w; int cfg; float cphi; };
DEV void base_of(const Core& c, const float pw[3], float pb[3]) {   // 창 점 → base_link
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  const float dx = pw[0] - c.x, dy = pw[1] - c.y;
  pb[0] = cs * dx + sn * dy; pb[1] = -sn * dx + cs * dy; pb[2] = pw[2] - 0.15f;
}
DEV void hand_of(const Core& c, const float q5[5], float g, Fk& f, Hand& h) {
  float qq[N_Q], qd[N_Q];
  for (int k = 0; k < N_Q; ++k) { qq[k] = k < 5 ? q5[k] : g; qd[k] = 0.f; }
  fk(qq, qd, f);
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  hand_world(f, c.x, c.y, sn, cs, h);
}
// 기울기 phi 근처(±dmax, 0.05 간격, 가까운 것부터)에서 역기구학
DEV bool ik_near(const float tb[3], float phi, float roll, int el, float dmax, float q[5]) {
  const int n = (int)(dmax * 20.f + 0.5f);
  for (int dj = 0; dj <= 2 * n; ++dj) {
    const float dphi = (dj & 1) ? -0.05f * (float)((dj + 1) >> 1) : 0.05f * (float)(dj >> 1);
    if (ik_grasp(tb, phi + dphi, roll, el, q)) return true;
  }
  return false;
}
// 팔 자세 q(그리퍼 각 g)에서 팔·손·(든) 물체가 닿나 — 베이스 c
DEV bool arm_hits(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, const float q5[5], float g) {
  Core t = c;
  for (int k = 0; k < 5; ++k) t.q[k] = q5[k];
  t.q[5] = g;
  float qd[N_Q];
  for (int k = 0; k < N_Q; ++k) qd[k] = 0.f;
  Fk f;
  fk(t.q, qd, f);
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  return arm_collides(t, b, ss, E, p, E.odim, f, ac, sn, cs);
}
// 든 물체를 손 자세에 붙인 상태(잡을 때와 같은 식: 가운데 = 잡는 점 + rel·축, yaw = 로봇 yaw + joint1 + ryaw)
DEV void held_at(const Core& c, const float q5[5], const PState& p, PState& ph) {
  Fk f;
  Hand h;
  hand_of(c, q5, 0.f, f, h);
  ph = p;
  for (int a = 0; a < 3; ++a) ph.o[a] = h.p[a] + p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a];
  set_yaw(ph, c.yaw + q5[0] + p.ryaw);
}
DEV float grasp_z(bool down, const float lo[3], const float hi[3], float sup, float oz) {   // 잡는 점 높이(창 z)
  float z = down ? hi[2] - minf(0.02f, 0.5f * (hi[2] - lo[2])) : oz;
  return maxf(z, sup + (down ? KG::tip_front - KG::tip_in + KG::r_tip + 0.002f : KG::r_tip + 0.004f));
}
DEV int n_phi(bool down) { return down ? 5 : 6; }
DEV float phi_of(bool down, int k) {   // 위에서: −90° → −56°(0.15 rad), 옆: 0, −0.15, −0.3, −0.45, +0.15, +0.3
  if (down) return -1.5707963f + 0.15f * (float)k;
  return k < 4 ? -0.15f * (float)k : 0.15f * (float)(k - 3);
}

// 한 잡기 자세(베이스 c, 위에서/옆, 기울기 phi, 팔꿈치 el, rsel: 위에서 0 = 좁은 축·1 = 다른 축)가 되는가 — 되면 웨이포인트 5 개와 쥘 자리(rel, ryaw)
TDEV bool arm_try_grasp(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, float zg, bool down, float phi,
                        int el, int rsel, ArmPlan& ap, float rel[4]) {
  const float* e = E.odim;
  const float gw[3] = {p.o[0], p.o[1], zg};
  float gb[3];
  base_of(c, gw, gb);
  if (!down && rsel) return false;
  float q[5];
  if (!ik_grasp(gb, phi, 0.f, el, q)) { TDBG(0); return false; }
  float roll = 0.f;
  if (down) {   // 닫는 축의 수평 각 θ(roll 0); roll d 면 θ − d. 원하는 = 물체 좁은 축(rsel 0) 또는 다른 축
    Fk f0;
    Hand h0;
    hand_of(c, q, 0.f, f0, h0);
    const float th = atan2f_d(h0.n[1], h0.n[0]);
    const float ya = atan2f_d(p.os_, p.oc_);
    const bool n0 = (e[0] <= e[1]) == (rsel == 0);
    float d = wrap_pi(th - (n0 ? ya : ya + 1.5707963f));
    if (d > 1.5707963f) d = d - kPi; else if (d < -1.5707963f) d = d + kPi;
    roll = d;
    if (!ik_grasp(gb, phi, roll, el, q)) { TDBG(1); return false; }
  }
  Fk f;
  Hand h;
  hand_of(c, q, 0.f, f, h);
  {   // 무게(특권): 잡는 점이 joint1 축에서 r·이 기울기면 가반 하중 안
    const float rx = gb[0] - KIK::j1x, ry = gb[1];
    float sp, cp;
    sincosf_d(phi, &sp, &cp);
    if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), sp)) { TDBG(2); return false; }
  }
  const float wp = 2.f * proj_half(h.n, e, p.oc_, p.os_);
  if (!(wp <= KG::max_w)) { TDBG(3); return false; }
  const float open = minf(wp + KT::open_extra, grip_gap_of(0.6f)), go = grip_angle_of(open);
  float dn;
  if (!(grasp_width(h, p.o, e, p.oc_, p.os_, open, dn) > 0.f)) { TDBG(4); return false; }
  if (arm_hits(c, b, ss, E, p, ac, q, go)) { TDBG(5); return false; }
  // 다가가는 웨이포인트: 잡는 점에서 다가가는 축으로 pre_d·pre_d/2 뒤(같은 roll·팔꿈치, 기울기 ±0.3). 안 되면 위에서 곧게 내려옴(+z — 옆 잡기는
  // 가까운 자리에서 다가가는 축 뒤가 역기구학 띠 밖이 잦음; 손가락은 닫는 축이 수평이라 물체 양옆으로 내려감)
  float qa[2][5];
  bool oka = false;
  for (int mode = 0; mode < 2 && !oka; ++mode) {
    if (mode == 1 && h.a[2] < -0.9f) break;   // 위에서 잡기는 이미 위에서 내려옴
    oka = true;
    for (int k = 0; k < 2 && oka; ++k) {
      const float dd = (k == 0 ? KT::pre_d : 0.5f * KT::pre_d) * (mode ? 0.8f : 1.f);   // 위에서 내려올 때는 4.8·2.4 cm
      const float pw[3] = {mode ? h.p[0] : h.p[0] - dd * h.a[0], mode ? h.p[1] : h.p[1] - dd * h.a[1], mode ? h.p[2] + dd : h.p[2] - dd * h.a[2]};
      float pb[3];
      base_of(c, pw, pb);
      if (!ik_near(pb, phi, roll, el, mode ? 0.6f : 0.4f, qa[k])) { TDBG(6); oka = false; }
      else if (arm_hits(c, b, ss, E, p, ac, qa[k], go)) { TDBG(7); oka = false; }
    }
  }
  if (!oka) return false;
  // 들기: 닫히면 물체가 닫는 축으로 가운데로 밀림(dn) → 그 자리를 손에 붙여 3·7 cm 위(역기구학 ±0.35, 든 물체 충돌·무게)
  PState ph = p;
  for (int a = 0; a < 3; ++a) ph.o[a] = p.o[a] - dn * h.n[a];
  {
    const float d[3] = {ph.o[0] - h.p[0], ph.o[1] - h.p[1], ph.o[2] - h.p[2]};
    ph.rel[0] = dot3(d, h.a); ph.rel[1] = dot3(d, h.n); ph.rel[2] = dot3(d, h.b);
  }
  ph.ryaw = wrap_pi(p.yaw - (c.yaw + q[0]));
  ph.w = wp;
  ph.st = OS_HELD;
  float ql[2][5];
  // 들기 점: 곧게 위, 안 되면 joint1 축에서 바깥으로 3 cm·안으로 3 cm 같이(높은 면 옆 잡기는 띠 안쪽 끝이라 곧게 위가 자주 안 풀림)
  float rdx = gb[0] - KIK::j1x, rdy = gb[1];
  {
    const float rl = sqrtf(rdx * rdx + rdy * rdy);
    rdx = rl > 1e-4f ? rdx / rl : 1.f; rdy = rl > 1e-4f ? rdy / rl : 0.f;
  }
  for (int k = 0; k < 2; ++k) {
    bool okl = false;
    for (int m = 0; m < 3 && !okl; ++m) {
      const float dr = m == 0 ? 0.f : m == 1 ? 0.03f : -0.03f;
      const float tb2[3] = {gb[0] + dr * rdx, gb[1] + dr * rdy, gb[2] + (k == 0 ? KT::lift1 : KT::lift2)};
      okl = ik_near(tb2, phi, roll, el, 0.5f, ql[k]);
    }
    if (!okl) { TDBG(8); return false; }
    PState pl;
    held_at(c, ql[k], ph, pl);
    Fk f2;
    Hand h2;
    hand_of(c, ql[k], 0.f, f2, h2);
    float g2[3];
    grasp_point_base(f2, g2);
    const float rx = g2[0] - KIK::j1x, ry = g2[1];
    if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), h2.a[2])) { TDBG(9); return false; }
    if (arm_hits(c, b, ss, E, pl, ac, ql[k], grip_angle_of(wp))) { TDBG(10); return false; }
  }
  TDBG(11);
  for (int k = 0; k < 5; ++k) { ap.q[0][k] = qa[0][k]; ap.q[1][k] = qa[1][k]; ap.q[2][k] = q[k]; ap.q[3][k] = ql[0][k]; ap.q[4][k] = ql[1][k]; }
  ap.n = 5; ap.open = open; ap.w = wp; ap.cfg = el | (rsel << 1) | (down ? 4 : 0); ap.cphi = phi;
  rel[0] = ph.rel[0]; rel[1] = ph.rel[1]; rel[2] = ph.rel[2]; rel[3] = ph.ryaw;
  return true;
}
// 지금 베이스 자세에서 잡기 계획(모든 자세 차례: 위에서/옆 선호 → 기울기 → 팔꿈치 → roll). near: 지금 팔(c.q)에서 잡기 전 자세까지 관절 거리가
// 가장 작은 것(같으면 앞 차례) — 자리에 와서 다시 계획할 때 들어오며 든 자세와 같은 꼴을 고름(팔꿈치 뒤집기로 쓸고 지나가며 닿는 것을 피함).
// 자세 번호 = (꼴, 기울기, 팔꿈치, roll) 차례 — 워프면 레인이 나눠 봄(결과는 차례대로 본 것과 같음)
TDEV bool arm_grasp_here(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, ArmPlan& ap, float rel[4], bool near,
                         const WCtx& w) {
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  float lo[3], hi[3];
  obj_box(p, E.odim, lo, hi);
  const float sup = support_z(ss, b, E, p, p.o[0], p.o[1], lo[2], E.odim);
  const bool td_first = hi[2] <= KT::topdown_top;
  // 자리에 와서(near): 기울기를 0.05 rad 간격으로 촘촘히(위에서 −90°…−50°, 옆 −0.5…+0.4) — 역기구학 띠가 mm 단위로 좁은 높은 면 옆 잡기
  int nf[2];
  for (int fam = 0; fam < 2; ++fam) {
    const bool down = (fam == 0) == td_first;
    nf[fam] = (near ? (down ? 15 : 21) : n_phi(down)) * 2 * (down ? 2 : 1);
  }
  auto try_k = [&](int i, ArmPlan& a2, float r2[4]) -> bool {
    const int fam = i < nf[0] ? 0 : 1;
    const int j = fam ? i - nf[0] : i;
    const bool down = (fam == 0) == td_first;
    const int rsn = down ? 2 : 1;
    const int k = j / (2 * rsn), el = (j / rsn) & 1, rs = j % rsn;
    const float phi = !near ? phi_of(down, k) : down ? -1.5707963f + 0.05f * (float)k : (k & 1 ? -0.05f : 0.05f) * (float)((k + 1) >> 1);
    if (near && !down && (phi < -0.6f || phi > 0.4f)) return false;
    const float zg = grasp_z(down, lo, hi, sup, p.o[2]);
    return arm_try_grasp(c, b, ss, E, p, ac, zg, down, phi, el, rs, a2, r2);
  };
  const int n = nf[0] + nf[1];
  int ks;
  if (!near) ks = w_first(w, n, [&](int i) { ArmPlan a2; float r2[4]; return try_k(i, a2, r2); });
  else
    ks = w_argmin(w, n, [&](int i) -> float {
      ArmPlan a2;
      float r2[4];
      if (!try_k(i, a2, r2)) return 1e30f;
      float d = 0.f;
      for (int j = 0; j < 5; ++j) d = maxf(d, absf(a2.q[0][j] - c.q[j]));
      return d;
    });
  if (ks < 0) return false;
  return try_k(ks, ap, rel);   // 고른 자세로 다시(모든 레인 같은 값)
}

// 놓을 가운데(창): 놓을 목표 + 1 cm 위. 막는 물체가 있으면 로봇 쪽 옆 빈 자리(특권: 막는 물체를 앎). k > 0: 둘레 점(±x, ±y — 놓을 꼴마다 반경)
DEV bool place_center_k(const Core& c, const BState& b, const bsc::Entry& E, const PState& p, int k, float t[3]) {
  place_target(E, b, E.odim, t);
  t[2] = t[2] + 0.01f;
  if (p.oc[2] > 0.f) {
    float ux = c.x - p.oc[0], uy = c.y - p.oc[1];
    const float l = sqrtf(ux * ux + uy * uy);
    ux = l > 1e-4f ? ux / l : 1.f; uy = l > 1e-4f ? uy / l : 0.f;
    const float d = p.oc[2] + 0.5f * maxf(E.odim[0], E.odim[1]) + 0.03f;
    t[0] = p.oc[0] + d * ux; t[1] = p.oc[1] + d * uy;
  }
  if (k == 0) return true;
  float rad;
  const bool pt = (b.gmode & bsc::GM_PLACE_PT) || (E.dkind == bsc::DK_ONTOP && E.ppt_ok);
  if (pt) rad = p.oc[2] > 0.f ? 0.04f : 0.03f;
  else if (E.dkind == bsc::DK_FLOOR) rad = 0.10f;
  else if (E.dkind == bsc::DK_INSIDE) {
    const float m = minf(E.dhi[0] - E.dlo[0], E.dhi[1] - E.dlo[1]) * 0.5f - 0.5f * maxf(E.odim[0], E.odim[1]) - 0.01f;
    if (!(m > 0.01f)) return false;
    rad = minf(0.5f * m, 0.08f);
  } else rad = 0.08f;
  const float dx[4] = {1.f, -1.f, 0.f, 0.f}, dy[4] = {0.f, 0.f, 1.f, -1.f};
  t[0] = t[0] + rad * dx[k - 1]; t[1] = t[1] + rad * dy[k - 1];
  return true;
}
// 한 놓기 자세(베이스 c, 기울기 phi, 팔꿈치 el, roll 0 — 든 쥠 그대로)가 되는가: 든 물체 가운데 → tc
TDEV bool arm_try_place(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, const float tc[3], float phi,
                        int el, ArmPlan& ap) {
  const float* e = E.odim;
  float gw[3] = {tc[0], tc[1], tc[2]}, q[5];
  for (int it = 0; it < 2; ++it) {   // 손 축은 계획한 자세에서: 잡는 점 = 가운데 − rel·축
    float gb[3];
    base_of(c, gw, gb);
    if (!ik_grasp(gb, phi, 0.f, el, q)) { TDBG(17); return false; }
    Fk f;
    Hand h;
    hand_of(c, q, 0.f, f, h);
    for (int a = 0; a < 3; ++a) gw[a] = tc[a] - (p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a]);
  }
  float gb[3];
  base_of(c, gw, gb);
  if (!ik_grasp(gb, phi, 0.f, el, q)) { TDBG(18); return false; }
  {
    const float rx = gb[0] - KIK::j1x, ry = gb[1];
    float sp, cp;
    sincosf_d(phi, &sp, &cp);
    if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), sp)) { TDBG(19); return false; }
  }
  PState pp;
  held_at(c, q, p, pp);
  const float gh = grip_angle_of(p.w);
  if (arm_hits(c, b, ss, E, pp, ac, q, gh)) { TDBG(20); return false; }
  // 놓은 뒤: 수직으로 내려앉힌 자리에서 판정이 참, 떨어진 높이 ≤ 5 cm(용기 넣기는 빼고)
  PState pr = pp;
  pr.st = OS_REST;
  const float dh = settle(ss, b, E, pr, e);
  if (!(dh <= KG::drop_pen_h || E.dkind == bsc::DK_INSIDE)) { TDBG(21); return false; }
  if (!at_goal(ss, b, E, pr, e)) { TDBG(22); return false; }
  float qa[2][5];
  for (int k = 0; k < 2; ++k) {   // 놓기 전(6 cm 위)·가운데(3 cm 위)
    const float tb2[3] = {gb[0], gb[1], gb[2] + (k == 0 ? KT::pre_d : 0.5f * KT::pre_d)};
    if (!ik_near(tb2, phi, 0.f, el, 0.3f, qa[k])) { TDBG(23); return false; }
    PState pk;
    held_at(c, qa[k], p, pk);
    Fk f2;
    Hand h2;
    hand_of(c, qa[k], 0.f, f2, h2);
    float g2[3];
    grasp_point_base(f2, g2);
    const float rx = g2[0] - KIK::j1x, ry = g2[1];
    if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), h2.a[2])) return false;
    if (arm_hits(c, b, ss, E, pk, ac, qa[k], gh)) { TDBG(24); return false; }
  }
  // 물러나기: 열고 다가가는 축 뒤로 6 cm(안 되면 9 cm) + 3 cm 위, 손 ≥ retreat + 5 mm(놓인 물체는 그 자리)
  const float open = minf(p.w + KT::open_extra, grip_gap_of(0.6f)), go = grip_angle_of(open);
  Fk f;
  Hand h;
  hand_of(c, q, 0.f, f, h);
  float lo[3], hi[3];
  obj_box(pr, e, lo, hi);
  float qr[5];
  bool okr = false;
  for (int k = 0; k < 2 && !okr; ++k) {
    const float dd = k == 0 ? 0.06f : 0.09f;
    const float pw[3] = {h.p[0] - dd * h.a[0], h.p[1] - dd * h.a[1], h.p[2] - dd * h.a[2] + 0.03f};
    float pb[3];
    base_of(c, pw, pb);
    if (!ik_near(pb, phi, 0.f, el, 0.4f, qr)) continue;
    Fk f3;
    Hand h3;
    hand_of(c, qr, go, f3, h3);
    if (!(pt_box_dist(lo, hi, h3.p) >= KG::retreat + 0.005f)) continue;
    okr = !arm_hits(c, b, ss, E, pr, ac, qr, go);
  }
  if (!okr) { TDBG(25); return false; }
  TDBG(26);
  for (int k = 0; k < 5; ++k) { ap.q[0][k] = qa[0][k]; ap.q[1][k] = qa[1][k]; ap.q[2][k] = q[k]; ap.q[3][k] = qr[k]; ap.q[4][k] = qr[k]; }
  ap.n = 4; ap.open = open; ap.w = p.w; ap.cfg = el; ap.cphi = phi;
  return true;
}
DEV int n_phi_place() { return 8; }
DEV float phi_place(int k) { const float v[8] = {-1.5707963f, -1.2f, -0.9f, -0.6f, -0.3f, 0.f, 0.25f, -1.4f}; return v[k]; }
// 지금 베이스 자세에서 놓기 계획(놓을 가운데 후보 차례 → 기울기 → 팔꿈치 — 번호 = 그 차례, 워프면 레인이 나눠 봄)
TDEV bool arm_place_here(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, ArmPlan& ap, float tc_out[3], const WCtx& w) {
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  const int per = n_phi_place() * 2;
  auto try_k = [&](int i, ArmPlan& a2, float tc[3]) -> bool {
    const int kc = i / per, k = (i % per) / 2, el = i & 1;
    if (!place_center_k(c, b, E, p, kc, tc)) return false;
    if (!arm_try_place(c, b, ss, E, p, ac, tc, phi_place(k), el, a2)) return false;
    a2.cfg |= kc << 3;
    return true;
  };
  const int ks = w_first(w, 5 * per, [&](int i) { ArmPlan a2; float tc[3]; return try_k(i, a2, tc); });
  if (ks < 0) return false;
  return try_k(ks, ap, tc_out);
}

// ---- 128 비트 행(창 한 행) ----
struct TR { uint64_t lo, hi; };
DEV int tctz(uint64_t v) {
#ifdef __CUDA_ARCH__
  return __ffsll((long long)v) - 1;
#else
  return __builtin_ctzll(v);
#endif
}
DEV bool tr_bit(const TR* m, int c, int r) {   // 창 밖 = 막힘
  if (c < 0 || r < 0 || c >= bsc::WIN || r >= bsc::WIN) return true;
  return c < 64 ? ((m[r].lo >> c) & 1ull) != 0 : ((m[r].hi >> (c - 64)) & 1ull) != 0;
}
DEV void tr_set(TR* m, int c, int r, bool v) {
  if (c < 0 || r < 0 || c >= bsc::WIN || r >= bsc::WIN) return;
  uint64_t& w = c < 64 ? m[r].lo : m[r].hi;
  const uint64_t bit = 1ull << (c < 64 ? c : c - 64);
  w = v ? (w | bit) : (w & ~bit);
}
// 비트 켜기(여러 레인이 같은 낱말을 칠할 수 있음 — GPU 는 원자 OR, 켜기만이라 차례와 무관)
DEV void tr_or(TR* m, int c, int r) {
  if (c < 0 || r < 0 || c >= bsc::WIN || r >= bsc::WIN) return;
  uint64_t* w = c < 64 ? &m[r].lo : &m[r].hi;
  const uint64_t bit = 1ull << (c < 64 ? c : c - 64);
#ifdef __CUDA_ARCH__
  atomicOr(reinterpret_cast<unsigned long long*>(w), (unsigned long long)bit);
#else
  *w = *w | bit;
#endif
}
DEV int wcell(float v) { return (int)floorf(v * bsc::INV_CELL) + bsc::WIN / 2; }
DEV float wctr(int k) { return ((float)k + 0.5f) * bsc::CELL - bsc::WIN_HALF; }
// 막힘 칸 그리기: 칸 가운데가 dist(x, y) < r 인 칸(축 정렬 범위 lo..hi 안만)
template <class D>
DEV void tr_paint(TR* m, float x0, float y0, float x1, float y1, float r, const D& dist) {
  const int c0 = wcell(x0 - r), c1 = wcell(x1 + r), r0 = wcell(y0 - r), r1 = wcell(y1 + r);
  for (int rr = r0 < 0 ? 0 : r0; rr <= r1 && rr < bsc::WIN; ++rr)
    for (int cc = c0 < 0 ? 0 : c0; cc <= c1 && cc < bsc::WIN; ++cc)
      if (dist(wctr(cc), wctr(rr)) < r) tr_or(m, cc, rr);
}
// 특권 점유(창): 좁은 판 blk0 = 설 칸 성분이 이 짝의 잡는 자세 칸 성분과 다름 + 과제 물체(바닥 H_COLL 아래)·놓인 집을 물체·막는 물체를 r_nav0 부풀림.
// 넓은 판 blk1 = blk0 + 정적 충돌 상자·과제 물체를 r_nav1 부풀림
TDEV void tch_occ_full(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, TR* blk0, TR* blk1, const WCtx& w) {
  const bsc::SceneDev& d = ss.sc[b.scene];
  const int c0 = (int)floorf((E.wx - bsc::WIN_HALF - d.ox) * bsc::INV_CELL + 0.5f), r0 = (int)floorf((E.wy - bsc::WIN_HALF - d.oy) * bsc::INV_CELL + 0.5f);
  for (int r = w.lane; r < bsc::WIN; r += w.nl) {   // 행을 레인이 나눔
    TR row{0ull, 0ull};
    for (int c = 0; c < bsc::WIN; ++c) {
      const int sc = c0 + c, sr = r0 + r;
      const bool blk = sc < 0 || sr < 0 || sc >= d.W || sr >= d.H || d.comp[(size_t)sr * d.W + sc] != E.comp;
      if (blk) { if (c < 64) row.lo |= 1ull << c; else row.hi |= 1ull << (c - 64); }
    }
    blk0[r] = row;
  }
  w_sync(w);
  for (int pass = 0; pass < 2; ++pass) {
    TR* m = pass ? blk1 : blk0;
    const float rr = pass ? KT::r_nav1 : KT::r_nav0;
    if (pass) { for (int r = w.lane; r < bsc::WIN; r += w.nl) blk1[r] = blk0[r]; w_sync(w); }
    int item = 0;   // 칠할 것 번호(레인 = 번호 % nl)
    for (int k = 1; k < E.nprim; ++k) {
      const bsc::BPrim& P = E.prim[k];
      if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
      if ((item++ % w.nl) != w.lane) continue;
      tr_paint(m, P.lo[0], P.lo[1], P.hi[0], P.hi[1], rr, [&](float x, float y) {
        const float dx = maxf(maxf(P.lo[0] - x, x - P.hi[0]), 0.f), dy = maxf(maxf(P.lo[1] - y, y - P.hi[1]), 0.f);
        return sqrtf(dx * dx + dy * dy);
      });
    }
    if (p.st != OS_HELD && (item++ % w.nl) == w.lane) {
      float lo[3], hi[3];
      obj_box(p, E.odim, lo, hi);
      if (lo[2] < bsc::H_COLL) {
        const bsc::SBox ob = obj_sbox(p, E.odim, 0.f, 0.f);
        tr_paint(m, lo[0], lo[1], hi[0], hi[1], rr, [&](float x, float y) { return bsc::dist_pt_obb2(ob, x, y); });
      }
    }
    if (p.oc[2] > 0.f && (item++ % w.nl) == w.lane) {
      float lo[3], hi[3];
      occ_box(p, lo, hi);
      tr_paint(m, lo[0], lo[1], hi[0], hi[1], rr, [&](float x, float y) {
        const float dx = maxf(maxf(lo[0] - x, x - hi[0]), 0.f), dy = maxf(maxf(lo[1] - y, y - hi[1]), 0.f);
        return sqrtf(dx * dx + dy * dy);
      });
    }
    if (pass) {   // 정적 충돌 상자(창을 덮는 묶음)
      int bx0, by0, bx1, by1;
      bsc::bin_of(d, E.wx - bsc::WIN_HALF - rr, E.wy - bsc::WIN_HALF - rr, bx0, by0);
      bsc::bin_of(d, E.wx + bsc::WIN_HALF + rr, E.wy + bsc::WIN_HALF + rr, bx1, by1);
      bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
      bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
      for (int by = by0; by <= by1; ++by)
        for (int bx = bx0; bx <= bx1; ++bx) {
          const int bi = by * d.BW + bx;
          for (uint32_t k = d.bstart[bi]; k < d.bstart[bi + 1]; ++k) {
            const int j = d.bitem[k];
            if (!(d.bkind[j] & bsc::BK_COLL)) continue;
            if ((item++ % w.nl) != w.lane) continue;
            const bsc::SBox& B = d.box[j];
            const float ex = absf(B.c) * B.hx + absf(B.s) * B.hy, ey = absf(B.s) * B.hx + absf(B.c) * B.hy;
            // 이 묶음 안 부분만(상자가 여러 묶음에 걸치면 묶음마다 제 몫 — 같은 칸을 두 번 칠해도 같음)
            const float bxl = d.ox + (float)bx * bsc::BIN - E.wx, byl = d.oy + (float)by * bsc::BIN - E.wy;
            const float x0 = maxf(B.cx - E.wx - ex, bxl - rr), x1 = minf(B.cx - E.wx + ex, bxl + bsc::BIN + rr);
            const float y0 = maxf(B.cy - E.wy - ey, byl - rr), y1 = minf(B.cy - E.wy + ey, byl + bsc::BIN + rr);
            if (x0 > x1 || y0 > y1) continue;
            tr_paint(m, x0, y0, x1, y1, rr, [&](float x, float y) { return bsc::dist_pt_obb2(B, x + E.wx, y + E.wy); });
          }
        }
    }
    w_sync(w);
  }
}
// 움직이는 부분만(집을 물체 — 들지 않았으면, 막는 물체)을 반경 rr 로 칠함(tch_occ_full 의 그 부분과 같은 식)
DEV void tch_occ_dyn(const bsc::Entry& E, const PState& p, TR* m, float rr, const WCtx& w) {
  if (p.st != OS_HELD && w.lane == 0) {
    float lo[3], hi[3];
    obj_box(p, E.odim, lo, hi);
    if (lo[2] < bsc::H_COLL) {
      const bsc::SBox ob = obj_sbox(p, E.odim, 0.f, 0.f);
      tr_paint(m, lo[0], lo[1], hi[0], hi[1], rr, [&](float x, float y) { return bsc::dist_pt_obb2(ob, x, y); });
    }
  }
  if (p.oc[2] > 0.f && w.lane == (w.nl > 1 ? 1 : 0)) {
    float lo[3], hi[3];
    occ_box(p, lo, hi);
    tr_paint(m, lo[0], lo[1], hi[0], hi[1], rr, [&](float x, float y) {
      const float dx = maxf(maxf(lo[0] - x, x - hi[0]), 0.f), dy = maxf(maxf(lo[1] - y, y - hi[1]), 0.f);
      return sqrtf(dx * dx + dy * dy);
    });
  }
}
// 특권 점유: 정적 표(tocc — 짝마다 칸 성분·과제 물체·정적 상자를 미리 칠함)가 있으면 그것 + 움직이는 부분, 없으면 다 칠함. 칠하기는 켜기(OR)만이라 같은 비트
// (넓은 판 = 정적 넓은 판 ∪ 움직이는 것 r_nav0 ∪ r_nav1 = 정적 넓은 판 ∪ 움직이는 것 r_nav1 — 반경이 큰 칸 집합이 작은 것을 품음)
TDEV void tch_occ(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, TR* blk0, TR* blk1, const WCtx& w) {
  if (!ss.tocc) { tch_occ_full(ss, b, E, p, blk0, blk1, w); return; }
  const int ix = ss.toccix[b.ent];
  if (ix < 0) { tch_occ_full(ss, b, E, p, blk0, blk1, w); return; }
  const TR* T = reinterpret_cast<const TR*>(ss.tocc + (size_t)ix * 4 * bsc::WIN);
  for (int r = w.lane; r < bsc::WIN; r += w.nl) { blk0[r] = T[r]; blk1[r] = T[bsc::WIN + r]; }
  w_sync(w);
  tch_occ_dyn(E, p, blk0, KT::r_nav0, w);
  tch_occ_dyn(E, p, blk1, KT::r_nav1, w);
  w_sync(w);
}
// 정적 표 한 짝(pnp_feasibility): 물체를 든 것으로·막는 물체 없음으로 두고 다 칠함 → [좁은 판 WIN 행][넓은 판 WIN 행]
TDEV void tch_occ_static(const bsc::SceneSet& ss, int ent, TR* out, const WCtx& w) {
  const bsc::Entry& E = ss.ent[ent];
  BState b{};
  b.scene = E.scene; b.ent = ent; b.wx = E.wx; b.wy = E.wy;
  PState p;
  clear_p(p);
  p.st = OS_HELD;
  tch_occ_full(ss, b, E, p, out, out + bsc::WIN, w);
}
// BFS(로봇 칸에서, 8·4 이웃 번갈아 = 팔각 거리 ≈ 단계 × 0.1 m). lev[WIN²] 단계(255 = 못 감)
// 목표 칸(gc, gr ≥ 0)을 주면 그 칸 또는 3 × 3 이웃이 닿은 단계에서 멈춤(길 뽑기에는 그것으로 충분 — 결과 같음)
TDEV void tch_bfs(const TR* blk, int rc, int rr, uint8_t* lev, TR* fr, TR* vis, TR* nw, const WCtx& w, int gc = -1, int gr = -1) {
  for (int k = w.lane; k < bsc::WIN * bsc::WIN; k += w.nl) lev[k] = 255;
  for (int r = w.lane; r < bsc::WIN; r += w.nl) { fr[r] = TR{0ull, 0ull}; vis[r] = TR{0ull, 0ull}; }
  w_sync(w);
  if (rc < 0 || rr < 0 || rc >= bsc::WIN || rr >= bsc::WIN) return;
  if (w.lane == 0) {
    tr_set(fr, rc, rr, true);
    tr_set(vis, rc, rr, true);
    lev[rr * bsc::WIN + rc] = 0;
  }
  w_sync(w);
  auto up = [](TR a) { return TR{a.lo << 1, (a.hi << 1) | (a.lo >> 63)}; };
  auto dn = [](TR a) { return TR{(a.lo >> 1) | (a.hi << 63), a.hi >> 1}; };
  auto h3 = [&](TR a) { const TR u = up(a), v = dn(a); return TR{a.lo | u.lo | v.lo, a.hi | u.hi | v.hi}; };
  for (int L = 1; L <= 254; ++L) {
    bool any = false;
    for (int r = w.lane; r < bsc::WIN; r += w.nl) {   // 단계 L 새 칸(앞 단계 경계에서) — 행을 레인이 나눔
      TR x;
      if (L & 1) {   // 8 이웃
        x = h3(fr[r]);
        if (r > 0) { const TR q = h3(fr[r - 1]); x.lo |= q.lo; x.hi |= q.hi; }
        if (r < bsc::WIN - 1) { const TR q = h3(fr[r + 1]); x.lo |= q.lo; x.hi |= q.hi; }
      } else {       // 4 이웃
        const TR u = up(fr[r]), v = dn(fr[r]);
        x = TR{u.lo | v.lo, u.hi | v.hi};
        if (r > 0) { x.lo |= fr[r - 1].lo; x.hi |= fr[r - 1].hi; }
        if (r < bsc::WIN - 1) { x.lo |= fr[r + 1].lo; x.hi |= fr[r + 1].hi; }
      }
      nw[r] = TR{x.lo & ~blk[r].lo & ~vis[r].lo, x.hi & ~blk[r].hi & ~vis[r].hi};
      any = any || (nw[r].lo | nw[r].hi) != 0ull;
    }
    any = w_any(w, any);
    w_sync(w);
    if (!any) break;
    for (int r = w.lane; r < bsc::WIN; r += w.nl) {
      vis[r].lo |= nw[r].lo; vis[r].hi |= nw[r].hi; fr[r] = nw[r];
      for (uint64_t v = nw[r].lo; v; v &= v - 1ull) lev[r * bsc::WIN + tctz(v)] = (uint8_t)L;
      for (uint64_t v = nw[r].hi; v; v &= v - 1ull) lev[r * bsc::WIN + 64 + tctz(v)] = (uint8_t)L;
    }
    w_sync(w);
    if (gc >= 0) {
      bool hit = false;
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc) {
          const int c2 = gc + dc, r2 = gr + dr;
          hit = hit || (c2 >= 0 && r2 >= 0 && c2 < bsc::WIN && r2 < bsc::WIN && lev[r2 * bsc::WIN + c2] < 255);
        }
      if (hit) break;
    }
  }
}
// 창 직선 (x0,y0) → (x1,y1) 이 막힌 칸을 안 지나나(0.05 m 표본, 시작·끝 칸은 봐줌)
DEV bool seg_open(const TR* blk, float x0, float y0, float x1, float y1) {
  const float dx = x1 - x0, dy = y1 - y0, L = sqrtf(dx * dx + dy * dy);
  const int n = (int)(L * 20.f) + 1;
  const int cs = wcell(x0), rs = wcell(y0), ce = wcell(x1), re = wcell(y1);
  for (int k = 0; k <= n; ++k) {
    const float u = (float)k / (float)n;
    const int c = wcell(x0 + dx * u), r = wcell(y0 + dy * u);
    if ((c == cs && r == rs) || (c == ce && r == re)) continue;
    if (tr_bit(blk, c, r)) return false;
  }
  return true;
}
// 길 뽑기: 단계 칸 lev(로봇 칸 0)에서 목표 (gx, gy) 칸(또는 닿은 이웃 칸 중 가장 낮은 것)부터 내려가 로봇까지, 뒤집어 직선 줄이기. 웨이포인트 수(0 = 실패)
TDEV int tch_extract(const uint8_t* lev, const TR* blk, float rx, float ry, float gx, float gy, float* wp, int& part) {
  part = 0;
  int gc = wcell(gx), gr = wcell(gy);
  int best = -1, bl = 255;
  for (int k = 0; k < 9; ++k) {   // 목표 칸, 아니면 3 × 3 이웃 중 가장 낮은 단계(같으면 앞 차례). (2026-10-06 고침: 예전엔 (+1, +1) 이웃을 안 봄 —
    const int kk = k == 0 ? 4 : k <= 4 ? k - 1 : k;   //  lev_near 는 닿았다는데 길 뽑기가 실패(NOPATH)할 수 있었음)
    const int c = gc + kk % 3 - 1, r = gr + kk / 3 - 1;
    if (c < 0 || r < 0 || c >= bsc::WIN || r >= bsc::WIN) continue;
    const int L = lev[r * bsc::WIN + c];
    if (L < bl) { bl = L; best = r * bsc::WIN + c; }
    if (k == 0 && L < 255) break;
  }
  if (best < 0 || bl == 255) return 0;
  uint16_t seq[256];
  int n = 0, cur = best;
  seq[n++] = (uint16_t)cur;
  const int dc[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dr[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (lev[cur] > 0 && n < 256) {
    const int c = cur % bsc::WIN, r = cur / bsc::WIN;
    int nb = -1, nl = lev[cur];
    for (int k = 0; k < 8; ++k) {
      const int c2 = c + dc[k], r2 = r + dr[k];
      if (c2 < 0 || r2 < 0 || c2 >= bsc::WIN || r2 >= bsc::WIN) continue;
      const int L = lev[r2 * bsc::WIN + c2];
      if (L < nl) { nl = L; nb = r2 * bsc::WIN + c2; }
    }
    if (nb < 0) break;
    cur = nb;
    seq[n++] = (uint16_t)cur;
  }
  if (lev[cur] != 0) return 0;
  // seq[n−1] = 로봇 칸 … seq[0] = 목표 칸. 로봇 자리부터 직선으로 갈 수 있는 가장 먼 칸으로
  float px = rx, py = ry;
  int idx = n - 1, m = 0;
  while (idx > 0 && m < T_NWP) {
    const int lo = idx - 40 > 0 ? idx - 40 : 0;
    int pick = idx - 1;
    for (int k = lo; k < idx - 1; ++k) {
      const float x = k == 0 ? gx : wctr(seq[k] % bsc::WIN), y = k == 0 ? gy : wctr(seq[k] / bsc::WIN);
      if (seg_open(blk, px, py, x, y)) { pick = k; break; }
    }
    const float x = pick == 0 ? gx : wctr(seq[pick] % bsc::WIN), y = pick == 0 ? gy : wctr(seq[pick] / bsc::WIN);
    wp[2 * m] = x; wp[2 * m + 1] = y;
    ++m;
    px = x; py = y;
    idx = pick;
  }
  if (n == 1) { wp[0] = gx; wp[1] = gy; m = 1; }   // 이미 목표 칸
  part = idx > 0;
  return m;
}
// 계획 작업 메모리 나누기
struct TScr { uint8_t* lev1; uint8_t* lev0; TR* blk0; TR* blk1; TR* fr; TR* vis; TR* nw; };
DEV TScr scr_of(uint8_t* s) {
  TScr o;
  o.lev1 = s; o.lev0 = s + bsc::WIN * bsc::WIN;
  TR* r = reinterpret_cast<TR*>(s + 2 * bsc::WIN * bsc::WIN);
  o.blk0 = r; o.blk1 = r + bsc::WIN; o.fr = r + 2 * bsc::WIN; o.vis = r + 3 * bsc::WIN; o.nw = r + 4 * bsc::WIN;
  return o;
}
// 로봇에서 BFS 둘(넓은 판·좁은 판). 로봇 칸은 늘 열고, 로봇 둘레 r_esc 안은 넓은 판도 좁은 판 값으로(가구 옆에서 빠져나오기)
TDEV void tch_reach(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const TScr& S, const WCtx& w, float gx = 1e9f, float gy = 1e9f) {
  TSEG_INIT
  tch_occ(ss, b, E, p, S.blk0, S.blk1, w);
  TSEG(8);
  const int rc = wcell(c.x), rr = wcell(c.y);
  if (w.lane == 0) {
    const int re = (int)ceilf(KT::r_esc * bsc::INV_CELL);
    for (int dr = -re; dr <= re; ++dr)
      for (int dc = -re; dc <= re; ++dc) {
        const float ex = wctr(rc + dc) - c.x, ey = wctr(rr + dr) - c.y;
        if (ex * ex + ey * ey > KT::r_esc * KT::r_esc) continue;
        tr_set(S.blk1, rc + dc, rr + dr, tr_bit(S.blk0, rc + dc, rr + dr) && !(dc == 0 && dr == 0));
      }
    tr_set(S.blk0, rc, rr, false);
  }
  w_sync(w);
  const int gc = gx < 1e8f ? wcell(gx) : -1, gr = gx < 1e8f ? wcell(gy) : -1;
  tch_bfs(S.blk1, rc, rr, S.lev1, S.fr, S.vis, S.nw, w, gc, gr);
  tch_bfs(S.blk0, rc, rr, S.lev0, S.fr, S.vis, S.nw, w, gc, gr);
  TSEG(9);
}
DEV bool lev_near(const uint8_t* lev, float x, float y) {   // 칸 또는 3 × 3 이웃이 닿음
  const int c = wcell(x), r = wcell(y);
  for (int dr = -1; dr <= 1; ++dr)
    for (int dc = -1; dc <= 1; ++dc) {
      const int c2 = c + dc, r2 = r + dr;
      if (c2 >= 0 && r2 >= 0 && c2 < bsc::WIN && r2 < bsc::WIN && lev[r2 * bsc::WIN + c2] < 255) return true;
    }
  return false;
}

// ---- 서는 자리 찾기 ----
// 닿음 판단: 계획 때 BFS(lev0, 런타임) 또는 창 닿는 칸 비트(rb — 잡기 가능 표: 잡는 자세 칸에서 BFS)
struct NavCtx { const uint8_t* lev; const uint32_t* rb; };
DEV bool nav_reach(const NavCtx& nc, float x, float y) {
  if (nc.lev) return lev_near(nc.lev, x, y);
  if (nc.rb) {
    const int c = wcell(x), r = wcell(y);
    if (c < 0 || r < 0 || c >= bsc::WIN || r >= bsc::WIN) return false;
    const int wc = r * bsc::WIN + c;
    return ((nc.rb[wc >> 5] >> (wc & 31)) & 1u) != 0;
  }
  return true;
}
// 몸통(반 크기 + m)이 (x, y, yaw) 에서 안 닿나 — body_free_pnp 와 같은 규칙
DEV bool body_free_m(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, float x, float y, float yaw, float m) {
  if (m == 0.f) return body_free_pnp(ss, b, E, p, x, y, yaw);
  float s, co;
  sincosf_d(yaw, &s, &co);
  const float hl = K::half_len + m, hw = K::half_wid + m;
  if (bsc::body_hits_scene(ss.sc[b.scene], x + b.wx, y + b.wy, s, co, hl, hw)) return false;
  for (int k = 1; k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(x, y, s, co, hl, hw, P.lo, P.hi)) return false;
  }
  if (p.st != OS_HELD) {
    float lo[3], hi[3];
    obj_box(p, E.odim, lo, hi);
    if (lo[2] < bsc::H_COLL && bsc::rect_hits_obb(x, y, s, co, hl, hw, obj_sbox(p, E.odim, 0.f, 0.f))) return false;
  }
  if (p.oc[2] > 0.f) {
    float lo[3], hi[3];
    occ_box(p, lo, hi);
    if (lo[2] < bsc::H_COLL && bsc::rect_hits_aabb(x, y, s, co, hl, hw, lo, hi)) return false;
  }
  return true;
}
// 몸통 + (들고 있으면) 나르는 자세 팔·든 물체가 (x, y, yaw) 에서 안 닿나
DEV bool pose_free_held(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, float x, float y, float yaw) {
  if (!body_free_pnp(ss, b, E, p, x, y, yaw)) return false;
  if (p.st != OS_HELD) return true;
  float qc[5];
  if (!carry_q(qc)) home_q(qc);
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, x, y, ac);
  Core u{};
  u.x = x; u.y = y; u.yaw = yaw;
  PState ph;
  held_at(u, qc, p, ph);
  return !arm_hits(u, b, ss, E, ph, ac, qc, grip_angle_of(p.w));
}
// 앞 물러난 자리 P(자리에서 yaw 뒤로 dp): 창 안·설 칸·몸통·P → 자리 직선·(엄격) P 에서 제자리 돌기·닿음
DEV bool pre_ok(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, const NavCtx& nc, float x, float y, float yaw, float dp, bool tier0,
                float& px, float& py) {
  float sn, cs;
  sincosf_d(yaw, &sn, &cs);
  px = x - dp * cs; py = y - dp * sn;
  if (!(absf(px) < bsc::WIN_HALF - 0.3f && absf(py) < bsc::WIN_HALF - 0.3f)) return false;
  const bsc::SceneDev& d = ss.sc[b.scene];
  const int ci = pnp_cell(d, E, px, py);
  if (ci < 0 || d.comp[ci] != E.comp) return false;
  const int n = (int)(dp * 20.f + 0.5f);
  for (int k = 0; k < n; ++k) {   // P 부터 자리 앞까지(자리는 따로 봄)
    const float u = (float)k / (float)n;
    if (!body_free_pnp(ss, b, E, p, px + (x - px) * u, py + (y - py) * u, yaw)) return false;
  }
  if (tier0) {   // 몸통은 π 대칭이라 반 바퀴만(0.15 rad 간격). 들고 있으면 나르는 자세 팔·든 물체도 한 바퀴(0.3 rad 간격)
    for (int k = 1; k < 21; ++k)
      if (!body_free_pnp(ss, b, E, p, px, py, yaw + 0.15f * (float)k)) return false;
    if (p.st == OS_HELD) {
      float qc[5];
      if (!carry_q(qc)) home_q(qc);
      ArmCand ac;
      arm_gather(ss.sc[b.scene], E, b.wx, b.wy, px, py, ac);
      Core u{};
      u.x = px; u.y = py;
      for (int k = 0; k < 21; ++k) {
        u.yaw = yaw + 0.3f * (float)k;
        PState ph;
        held_at(u, qc, p, ph);
        if (arm_hits(u, b, ss, E, ph, ac, qc, grip_angle_of(p.w))) return false;
      }
    }
  }
  return nav_reach(nc, px, py);
}
// 들어가는 길(P → 자리, 0.05 m 간격)에서 팔이 q(그리퍼 g)로 안 닿음 — 든 물체는 손에 붙여서
DEV bool drive_in_free(const Core& t, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, const float q[5], float g, float dp) {
  float sn, cs;
  sincosf_d(t.yaw, &sn, &cs);
  const int n = (int)(absf(dp) * 20.f + 0.5f);
  for (int k = 1; k <= n; ++k) {
    Core u = t;
    u.x = t.x - cs * dp * (float)k / (float)n; u.y = t.y - sn * dp * (float)k / (float)n;
    PState pu;
    if (p.st == OS_HELD) held_at(u, q, p, pu); else pu = p;
    if (arm_hits(u, b, ss, E, pu, ac, q, g)) return false;
  }
  return true;
}
struct StanceOut { float x, y, yaw, px, py, dp; ArmPlan ap; float rel[4]; float tc[3]; };
DEV float dpre_of(int k) { return k == 0 ? 0.25f : k == 1 ? 0.40f : 0.15f; }
DEV bool is_avoid(float ax, float ay, float ayaw, float x, float y, float yaw) {
  return (ax - x) * (ax - x) + (ay - y) * (ay - y) < 0.05f * 0.05f && absf(wrap_pi(ayaw - yaw)) < 0.15f;
}
constexpr int T_NR = 31;          // r 띠: 0.12 + 0.01 k
constexpr float T_R0 = 0.12f;
// 공통 고리: 목표 (tx, ty) 둘레 후보를 비용 구간 차례로. bands[k] = 팔 자세 k 의 r 비트, full(t, ac, ir, dp) = 그 자리 팔 계획(되면 true, out 채움)
// 등급(작을수록 좋음): 0 = 몸통 여유 2 cm + P 제자리 돌기 + 자리 오차에 버팀(같은 팔 자세가 앞뒤·옆 ±1.5 cm 에서도), 1 = 버팀만, 2 = 그냥 됨, 3 = 안 됨.
// 고르기 = 등급 0 중 (비용 구간, 후보 번호) 차례로 처음, 없으면 등급 1 처음, 없으면 2 처음(예전 단계 셋 고리와 같은 답 — 후보마다 한 번만 봄)
template <class Full, class Rob>
DEV int stance_grade(const Core& t, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, int ir, float dp, bool rot_ok,
                     const Full& full, const Rob& rob, StanceOut& out) {
  if (!full(t, ac, ir, dp, out)) return 3;
  if (!rob(t, out)) { TDBG(16); return 2; }
  if (!rot_ok || !body_free_m(ss, b, E, p, t.x, t.y, t.yaw, 0.02f)) return 1;
  return 0;
}
// 여러 단계 후보 고르기(워프면 레인이 후보를 나눔): ev(i, out) = 등급(4 = 싼 검사에서 빠짐 — 예산에 안 셈). [k0, n) 에서 budget 개까지 — 등급 0 이면 그 번호, 아니면 −1 이고 next = 다음에 볼 번호.
// g1·g2 = 지금까지 처음 본 등급 1·2 번호(나눠 보며 이어짐). budget 은 본 수만큼 줄어듦
template <class Ev>
DEV int pick_graded(const WCtx& w, int k0, int n, const Ev& ev, int& g1, int& g2, int& budget, int& next) {
#ifdef __CUDA_ARCH__
  if (w.nl > 32) {   // 블록: 묶음 = 레인 수(T_CHUNK)
    int base = k0;
    for (; base < n && budget > 0; base += w.nl) {
      const int k = base + w.lane;
      StanceOut t2;
      const int q = k < n ? ev(k, t2) : 4;
      __syncthreads();
      if (w.lane == 0) { w.sh[0] = 0x7fffffff; w.sh[1] = 0x7fffffff; w.sh[2] = 0x7fffffff; w.sh[3] = 0; }
      __syncthreads();
      if (q == 0) atomicMin(&w.sh[0], k);
      if (q == 1) atomicMin(&w.sh[1], k);
      if (q == 2) atomicMin(&w.sh[2], k);
      if (q < 4) atomicAdd(&w.sh[3], 1);
      __syncthreads();
      const int m0 = w.sh[0], m1 = w.sh[1], m2 = w.sh[2], cnt = w.sh[3];
      budget -= cnt;   // 비싼 검사까지 간 후보만 셈
      if (m0 != 0x7fffffff) { next = n; return m0; }
      if (g1 < 0 && m1 != 0x7fffffff) g1 = m1;
      if (g2 < 0 && m2 != 0x7fffffff) g2 = m2;
    }
    next = base < n ? base : n;
    return -1;
  }
  if (w.nl > 1) {   // 워프(잡기 가능 표: 예산 무한 — 묶음 크기와 무관한 답)
    int base = k0;
    for (; base < n && budget > 0; base += 32) {
      const int k = base + w.lane;
      StanceOut t2;
      const int q = k < n ? ev(k, t2) : 4;
      const unsigned m0 = __ballot_sync(0xffffffffu, q == 0), m1 = __ballot_sync(0xffffffffu, q == 1), m2 = __ballot_sync(0xffffffffu, q == 2);
      budget -= __popc(__ballot_sync(0xffffffffu, q < 4));
      if (m0) { next = n; return base + __ffs((int)m0) - 1; }
      if (g1 < 0 && m1) g1 = base + __ffs((int)m1) - 1;
      if (g2 < 0 && m2) g2 = base + __ffs((int)m2) - 1;
    }
    next = base < n ? base : n;
    return -1;
  }
#endif
  (void)w;
  // CPU: 블록과 같은 T_CHUNK 묶음 단위로 예산을 씀(GPU 계획 커널과 같은 곳에서 끊김)
  int base = k0;
  for (; base < n && budget > 0; base += T_CHUNK) {
    int c1 = -1, c2 = -1;   // 이 묶음의 등급 1·2(등급 0 이 없을 때만 씀 — GPU 와 같게)
    for (int k = base; k < base + T_CHUNK && k < n; ++k) {
      StanceOut t2;
      const int q = ev(k, t2);
      budget -= q < 4 ? 1 : 0;
      if (q == 0) { next = n; return k; }
      if (c1 < 0 && q == 1) c1 = k;
      if (c2 < 0 && q == 2) c2 = k;
    }
    if (g1 < 0) g1 = c1;
    if (g2 < 0) g2 = c2;
  }
  next = base < n ? base : n;
  return -1;
}
// 서는 자리 찾기 진행(여러 스텝에 나눠 함 — 한 스텝 계획 시간 상한): 놓을 가운데 후보 kc, 후보 번호 pos(바로 갈 자리 앞, 원호 뒤), 등급 1·2 처음 번호
struct SPos { int kc, pos, g1, g2; };
template <class Full, class Rob>
DEV int stance_loop(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const NavCtx& nc, float tx, float ty,
                     uint32_t rmask, float avx, float avy, float avyaw, bool direct, float slack, const Full& full, const Rob& rob, StanceOut& o, const WCtx& w,
                     SPos& sp, int& budget) {   // 반환 1 찾음(o), 0 없음, 2 이어서(다음 스텝)
  const float a0 = atan2f_d(c.y - ty, c.x - tx);
  const float bins[7] = {0.3f, 0.6f, 1.0f, 1.6f, 2.5f, 4.f, 1e30f};
  const bsc::SceneDev& d = ss.sc[b.scene];
  // 바로 갈 수 있는 자리 먼저(런타임만): 지금 자리에서 제자리로 δ(0, ±0.1, ±0.2 rad) 돌고 그 방향으로 곧게 d(0, ±0.02 … ±0.30 m, 뒤로도) 간 자세 — P = 지금 자리.
  // 후보 번호 = (d 차례, δ 차례). slack > 0(표): 판 시작 yaw 가 지금 yaw ± slack 로 흔들리므로 그 범위 제자리 돌기도 안 닿아야
  bool slack_ok = direct;
  for (int k = -(int)(slack * 20.f + 0.5f); k <= (int)(slack * 20.f + 0.5f) && slack_ok; ++k) slack_ok = pose_free_held(ss, b, E, p, c.x, c.y, c.yaw + 0.05f * (float)k);
  auto direct_k = [&](int idx, StanceOut& out) -> int {
    const int dk = idx / 5, kd = idx % 5;
    const float dd = (dk & 1) ? 0.02f * (float)((dk + 1) >> 1) : -0.02f * (float)(dk >> 1);
    const float del = kd == 0 ? 0.f : (kd & 1 ? 0.1f : -0.1f) * (float)((kd + 1) >> 1);
    const float yaw = wrap_pi(c.yaw + del);
    float sy2, cy2;
    sincosf_d(yaw, &sy2, &cy2);
    const float x = c.x + dd * cy2, y = c.y + dd * sy2;
    if (!(absf(x) < bsc::WIN_HALF - 0.3f && absf(y) < bsc::WIN_HALF - 0.3f)) return 4;
    if (is_avoid(avx, avy, avyaw, x, y, yaw)) return 4;
    // 팔: joint1 축 → 목표 거리 r(띠 비트), 팔 방향 |β| ≤ 1.45
    const float jx = x + KIK::j1x * cy2, jy = y + KIK::j1x * sy2;
    const float rr = sqrtf((tx - jx) * (tx - jx) + (ty - jy) * (ty - jy));
    const int ir = (int)floorf((rr - T_R0) * 100.f + 0.5f);
    if (ir < 0 || ir >= T_NR || !((rmask >> ir) & 1u)) return 4;
    if (absf(wrap_pi(atan2f_d(ty - jy, tx - jx) - yaw)) > 1.45f) return 4;
    const int ci = pnp_cell(d, E, x, y);
    if (ci < 0 || d.comp[ci] == 0) return 3;
    if (!body_free_pnp(ss, b, E, p, x, y, yaw)) return 3;
    const int nr = (int)(absf(del) * 20.f + 0.5f);
    for (int k = 1; k <= nr; ++k) if (!pose_free_held(ss, b, E, p, c.x, c.y, c.yaw + del * (float)k / (float)nr)) return 3;
    const int ns = (int)(absf(dd) * 20.f + 0.5f);
    for (int k = 1; k < ns; ++k) if (!body_free_pnp(ss, b, E, p, c.x + dd * cy2 * (float)k / (float)ns, c.y + dd * sy2 * (float)k / (float)ns, yaw)) return 3;
    Core t = c;
    t.x = x; t.y = y; t.yaw = yaw;
    ArmCand ac;
    arm_gather(d, E, b.wx, b.wy, x, y, ac);
    const int q = stance_grade(t, b, ss, E, p, ac, ir, dd, true, full, rob, out);
    if (q < 3) { out.x = x; out.y = y; out.yaw = yaw; out.px = c.x; out.py = c.y; out.dp = dd; }
    return q;
  };
  const int nd = slack_ok ? 31 * 5 : 0;
  const int ncand = 36 * 7 * T_NR;
  if (sp.pos < nd) {
    int next = nd;
    const int k0 = pick_graded(w, sp.pos, nd, direct_k, sp.g1, sp.g2, budget, next);
    if (k0 >= 0) { TBIN(0, 7); return direct_k(k0, o) < 3 ? 1 : 0; }
    if (next < nd && sp.g1 < 0 && sp.g2 < 0) { sp.pos = next; return 2; }
    const int ks = sp.g1 >= 0 ? sp.g1 : sp.g2;
    if (ks >= 0) { TBIN(sp.g1 >= 0 ? 1 : 2, 7); return direct_k(ks, o) < 3 ? 1 : 0; }
    sp.pos = nd; sp.g1 = -1; sp.g2 = -1;
  }
  // 둘레 원호: 후보 번호 = (비용 구간, 방향 k, 팔 방향 ib, r) 차례 — 구간 밖이면 그 구간에서 건너뜀
  auto gen_k = [&](int gidx, StanceOut& out) -> int {
    const int pass = gidx / ncand, idx = gidx % ncand;
    const float clo = pass ? bins[pass - 1] : -1.f, chi = bins[pass];
    const int ir = idx % T_NR, ib = (idx / T_NR) % 7, k = idx / (T_NR * 7);
    if (!((rmask >> ir) & 1u)) return 4;
    const float bsv[7] = {0.f, 0.45f, -0.45f, 0.9f, -0.9f, 1.3f, -1.3f};
    const float bs = bsv[ib];
    const int kk = (k & 1) ? -((k + 1) >> 1) : (k >> 1);
    const float ang = a0 + 0.17453293f * (float)kk;   // 목표 → joint1 축 방향
    float sa, ca;
    sincosf_d(ang, &sa, &ca);
    const float yaw = wrap_pi(ang + kPi - bs);
    float sy2, cy2;
    sincosf_d(yaw, &sy2, &cy2);
    const float r = T_R0 + 0.01f * (float)ir;
    const float x = tx + r * ca - KIK::j1x * cy2, y = ty + r * sa - KIK::j1x * sy2;
    // 비용 = 지금 → 앞 물러난 자리(기본 0.25 m) 거리 + 팔 방향(옆으로 뻗을수록 조금)
    const float p0x = x - 0.25f * cy2, p0y = y - 0.25f * sy2;
    const float cost = sqrtf((p0x - c.x) * (p0x - c.x) + (p0y - c.y) * (p0y - c.y)) + 0.1f * absf(bs);
    if (!(cost > clo && cost <= chi)) return 4;
    if (!(absf(x) < bsc::WIN_HALF - 0.3f && absf(y) < bsc::WIN_HALF - 0.3f)) return 4;
    if (is_avoid(avx, avy, avyaw, x, y, yaw)) return 4;
    const int ci = pnp_cell(d, E, x, y);
    if (ci < 0 || d.comp[ci] == 0) { TDBG(12); return 3; }
    if (!body_free_pnp(ss, b, E, p, x, y, yaw)) { TDBG(13); return 3; }
    float px = 0.f, py = 0.f, dp = 0.f;
    bool okp = false;
    for (int kd = 0; kd < 3 && !okp; ++kd) { dp = dpre_of(kd); okp = pre_ok(ss, b, E, p, nc, x, y, yaw, dp, false, px, py); }
    if (!okp) { TDBG(14); return 3; }
    Core t = c;
    t.x = x; t.y = y; t.yaw = yaw;
    ArmCand ac;
    arm_gather(d, E, b.wx, b.wy, x, y, ac);
    // 등급 0 의 P 제자리 돌기는 버팀까지 된 것만 봄(비쌈)
    auto full_rot = [&](const Core& t2, const ArmCand& ac2, int ir2, float dp2, StanceOut& o2) -> bool { return full(t2, ac2, ir2, dp2, o2); };
    int q = stance_grade(t, b, ss, E, p, ac, ir, dp, true, full_rot, rob, out);
    if (q == 0) { float qx, qy; if (!pre_ok(ss, b, E, p, nc, x, y, yaw, dp, true, qx, qy)) q = 1; }
    if (q < 3) { out.x = x; out.y = y; out.yaw = yaw; out.px = px; out.py = py; out.dp = dp; }
    return q;
  };
  // 비용 구간 하나씩. 등급 0 은 바로 고름. 등급 1·2 를 찾았으면 그 조각(스텝 예산)·그 구간이 끝날 때 고름(2026-10-06: 예전엔 등급 0 을 바라고
  // 끝까지 훑음 — 런타임 찾기는 거의 늘 등급 2 로 끝나(잰 값: B4 745 중 0·B5 795 중 3) 후보 수천 개를 헛되이 봄)
  for (int pos = sp.pos - nd; pos < 7 * ncand;) {
    const int bend = (pos / ncand + 1) * ncand;
    int next = bend;
    const int k0 = pick_graded(w, pos, bend, gen_k, sp.g1, sp.g2, budget, next);
    if (k0 >= 0) { TBIN(0, k0 / ncand); return gen_k(k0, o) < 3 ? 1 : 0; }
    if (sp.g1 >= 0 || sp.g2 >= 0) break;
    if (next < bend) { sp.pos = nd + next; return 2; }
    pos = bend;
    if (budget <= 0 && pos < 7 * ncand) { sp.pos = nd + pos; return 2; }
  }
  const int ks = sp.g1 >= 0 ? sp.g1 : sp.g2;
  if (ks < 0) return 0;
  TBIN(sp.g1 >= 0 ? 1 : 2, ks / ncand);
  return gen_k(ks, o) < 3 ? 1 : 0;
}
// r 띠(joint1 축에서 잡는 점까지 수평 거리)를 훑음: 잡는 점 창 높이 zg, 기울기 phi, 팔꿈치 el 로 역기구학이 되는 r 비트
DEV uint32_t r_band(float zg, float phi, int el, float drad) {
  uint32_t m = 0u;
  for (int ir = 0; ir < T_NR; ++ir) {
    const float r = T_R0 + 0.01f * (float)ir - drad;
    const float tb[3] = {KIK::j1x + r, 0.f, zg - 0.15f};
    float q[5];
    if (ik_grasp(tb, phi, 0.f, el, q)) m |= 1u << ir;
  }
  return m;
}
// 잡기 서는 자리. 가구 옆 자리 ±1.5 cm·±0.03 rad 에서도 팔 계획이 되면(엄격) 먼저
TDEV int stance_grasp(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const NavCtx& nc, float avx, float avy,
                       float avyaw, bool direct, float slack, StanceOut& o, const WCtx& w, SPos& sp, int budget) {   // 반환 1·0·2(stance_loop)
  float lo[3], hi[3];
  obj_box(p, E.odim, lo, hi);
  const float sup = support_z(ss, b, E, p, p.o[0], p.o[1], lo[2], E.odim);
  const bool td_first = hi[2] <= KT::topdown_top;
  uint32_t band[2][6][2], rmask = 0u;
  float zg[2];
  for (int fam = 0; fam < 2; ++fam) {
    const bool down = (fam == 0) == td_first;
    zg[fam] = grasp_z(down, lo, hi, sup, p.o[2]);
    for (int k = 0; k < 6; ++k)
      for (int el = 0; el < 2; ++el) {
        band[fam][k][el] = k < n_phi(down) ? r_band(zg[fam], phi_of(down, k), el, 0.f) : 0u;
        rmask |= band[fam][k][el];
      }
  }
  auto full = [&](const Core& t, const ArmCand& ac, int ir, float dp, StanceOut& out) -> bool {
    for (int fam = 0; fam < 2; ++fam) {
      const bool down = (fam == 0) == td_first;
      for (int k = 0; k < n_phi(down); ++k)
        for (int el = 0; el < 2; ++el) {
          if (!((band[fam][k][el] >> ir) & 1u)) continue;
          for (int rs = 0; rs < (down ? 2 : 1); ++rs) {
            if (!arm_try_grasp(t, b, ss, E, p, ac, zg[fam], down, phi_of(down, k), el, rs, out.ap, out.rel)) continue;
            if (!drive_in_free(t, b, ss, E, p, ac, out.ap.q[0], grip_angle_of(out.ap.open), dp)) { TDBG(15); continue; }
            return true;
          }
        }
    }
    return false;
  };
  auto rob = [&](const Core& t, const StanceOut& out) -> bool {   // 자리 오차에 버팀: 같은 팔 자세가 앞뒤 ±1.5 cm(yaw ±0.03)·옆 ±1.5 cm 에서도
    const bool down = (out.ap.cfg & 4) != 0;
    const float zz = zg[down == td_first ? 0 : 1];
    for (int s2 = 0; s2 < 4; ++s2) {
      Core u = t;
      float sn, cs;
      sincosf_d(t.yaw, &sn, &cs);
      const float df = s2 == 0 ? 0.015f : s2 == 1 ? -0.015f : 0.f, dl = s2 == 2 ? 0.015f : s2 == 3 ? -0.015f : 0.f;
      u.x = t.x + df * cs - dl * sn; u.y = t.y + df * sn + dl * cs; u.yaw = wrap_pi(t.yaw + (s2 == 0 ? 0.03f : s2 == 1 ? -0.03f : 0.f));
      ArmCand acu;
      arm_gather(ss.sc[b.scene], E, b.wx, b.wy, u.x, u.y, acu);
      ArmPlan ap2;
      float rel2[4];
      if (!arm_try_grasp(u, b, ss, E, p, acu, zz, down, out.ap.cphi, out.ap.cfg & 1, (out.ap.cfg >> 1) & 1, ap2, rel2)) return false;
    }
    return true;
  };
  return stance_loop(c, b, ss, E, p, nc, p.o[0], p.o[1], rmask, avx, avy, avyaw, direct, slack, full, rob, o, w, sp, budget);
}
// 놓기 서는 자리(든 쥠 p.rel·ryaw·w 그대로). 놓을 가운데 후보(가운데, 둘레 4) 차례
TDEV int stance_place(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const NavCtx& nc, float avx, float avy,
                       float avyaw, bool direct, float slack, StanceOut& o, const WCtx& w, SPos& sp, int budget) {   // 반환 1·0·2
  for (; sp.kc < 5; ++sp.kc, sp.pos = 0, sp.g1 = -1, sp.g2 = -1) {
    const int kc = sp.kc;
    float tc[3];
    if (!place_center_k(c, b, E, p, kc, tc)) continue;
    uint32_t band[8][2], rmask = 0u;
    for (int k = 0; k < n_phi_place(); ++k) {
      // 든 물체 가운데 − 잡는 점(손 축 → base_link, joint1 0·roll 0): 이 기울기에서 축은 고정 — 띠는 물체 가운데 기준 r
      const float phi = phi_place(k);
      float qs[5];
      float rr = 0.f, rz = 0.f;
      bool okq = false;
      for (int j = 0; j < 6 && !okq; ++j) {   // 축을 읽을 아무 역기구학 해(r 0.20·0.25·0.30, 높이 0.20 m)
        const float tb0[3] = {KIK::j1x + 0.20f + 0.05f * (float)(j >> 1), 0.f, 0.05f};
        okq = ik_grasp(tb0, phi, 0.f, j & 1, qs);
      }
      if (okq) {
        float qq[N_Q], qd[N_Q];
        for (int j = 0; j < N_Q; ++j) { qq[j] = j < 5 ? qs[j] : 0.f; qd[j] = 0.f; }
        qq[0] = 0.f;
        Fk f;
        fk(qq, qd, f);
        rr = p.rel[0] * f.ee_R[0] + p.rel[1] * f.ee_R[1] + p.rel[2] * f.ee_R[2];
        rz = p.rel[0] * f.ee_R[6] + p.rel[1] * f.ee_R[7] + p.rel[2] * f.ee_R[8];
      } else {
        float sp, cp;
        sincosf_d(phi, &sp, &cp);
        rr = p.rel[0] * cp; rz = p.rel[0] * sp;
      }
      for (int el = 0; el < 2; ++el) { band[k][el] = r_band(tc[2] - rz, phi, el, rr); rmask |= band[k][el]; }
    }
    auto full = [&](const Core& t, const ArmCand& ac, int ir, float dp, StanceOut& out) -> bool {
      for (int k = 0; k < n_phi_place(); ++k)
        for (int el = 0; el < 2; ++el) {
          if (!((band[k][el] >> ir) & 1u)) continue;
          if (!arm_try_place(t, b, ss, E, p, ac, tc, phi_place(k), el, out.ap)) continue;
          out.ap.cfg |= kc << 3;
          if (!drive_in_free(t, b, ss, E, p, ac, out.ap.q[0], grip_angle_of(p.w), dp)) continue;
          out.tc[0] = tc[0]; out.tc[1] = tc[1]; out.tc[2] = tc[2];
          return true;
        }
      return false;
    };
    auto rob = [&](const Core& t, const StanceOut& out) -> bool {   // 같은 팔 자세(같은 놓을 가운데)가 앞뒤·옆 ±1.5 cm 에서도
      for (int s2 = 0; s2 < 4; ++s2) {
        Core u = t;
        float sn, cs;
        sincosf_d(t.yaw, &sn, &cs);
        const float df = s2 == 0 ? 0.015f : s2 == 1 ? -0.015f : 0.f, dl = s2 == 2 ? 0.015f : s2 == 3 ? -0.015f : 0.f;
        u.x = t.x + df * cs - dl * sn; u.y = t.y + df * sn + dl * cs; u.yaw = wrap_pi(t.yaw + (s2 == 0 ? 0.03f : s2 == 1 ? -0.03f : 0.f));
        ArmCand acu;
        arm_gather(ss.sc[b.scene], E, b.wx, b.wy, u.x, u.y, acu);
        ArmPlan ap2;
        if (!arm_try_place(u, b, ss, E, p, acu, tc, out.ap.cphi, out.ap.cfg & 1, ap2)) return false;
      }
      return true;
    };
    const int r = stance_loop(c, b, ss, E, p, nc, tc[0], tc[1], rmask, avx, avy, avyaw, direct, slack, full, rob, o, w, sp, budget);
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
    if (w.nl > 32 && w.lane == 0) atomicAdd(&g_tkc[r][kc], 1ull);
#endif
    if (r != 0) return r;
    if (budget <= 0) { ++sp.kc; sp.pos = 0; sp.g1 = -1; sp.g2 = -1; return sp.kc < 5 ? 2 : 0; }
  }
  return 0;
}

// ---- 잡기 가능 표(짝 하나, 판과 무관 — 장치 커널·CPU 확인이 같은 함수) ----
// B4: 처음 물체 자리에서 잡는 자세 칸(st, 물체를 봄)을 기준으로 잡기 서는 자리 + 팔 계획(닿음 = 창 닿는 칸 비트 rb).
// B5: 시작 쥠(reset_pnp_start 와 같은 식: 나르는 자세, 좁은 가로 폭, 물체 가운데 = 잡는 점)으로 놓기 서는 자리 + 나르는 자세 가반 하중.
// B6: 잡기 계획의 쥠(rel·ryaw)으로, gst 에서 들고 나르는 자세로 접은 상태에서 놓기 서는 자리.
struct FeasOut { int feas; float gst4[4], gst[4], pst5[4], pst6[4], grel[4]; };
DEV void feas_start(const bsc::SceneSet& ss, const bsc::Entry& E, int ent, int kind, Core& c, BState& b, PState& p) {
  c = Core{};
  b = BState{};
  b.wx = E.wx; b.wy = E.wy; b.scene = E.scene; b.ent = ent; b.kind = kind;
  b.gmode = 0;
  if (E.ppt_ok && (E.dkind == bsc::DK_FLOOR || E.dkind == bsc::DK_ONTOP)) {   // 점이 있으면 점 판(더 좁은 판정 0.05 m)으로 잼
    b.gmode = bsc::GM_PLACE_PT;
    for (int a = 0; a < 3; ++a) b.gp[a] = E.ppt[a];
  }
  clear_p(p);
  rest_obj(ss, b, E, p);
  c.x = E.st[0]; c.y = E.st[1]; c.yaw = atan2f_d(E.gy - c.y, E.gx - c.x);
  float qc[5];
  if (carry_q(qc)) for (int k = 0; k < 5; ++k) c.q[k] = qc[k];
}
TDEV void feas_entry(const bsc::SceneSet& ss, int ent, FeasOut& o, const WCtx& w) {
  o.feas = 0;
  for (int k = 0; k < 4; ++k) { o.gst4[k] = 0.f; o.gst[k] = 0.f; o.pst5[k] = 0.f; o.pst6[k] = 0.f; o.grel[k] = 0.f; }
  const bsc::Entry& E = ss.ent[ent];
  if (E.list != bsc::L_OBJ) return;
  const int sr = obj_static_feas(E.odim, E.mass);
  if (sr) { o.feas = (sr << 8) | (sr << 16) | (sr << 24); return; }
  const NavCtx nc{nullptr, E.rb >= 0 ? ss.rbits + E.rb : nullptr};
  Core c;
  BState b;
  PState p;
  // B4: 잡는 자세 칸(시작 — yaw ±0.3 흔들림)에서 바로 가는 자리 먼저
  feas_start(ss, E, ent, bsc::EK_B4, c, b, p);
  StanceOut s4;
  SPos sp{0, 0, -1, -1};
  if (stance_grasp(c, b, ss, E, p, nc, 1e9f, 1e9f, 0.f, true, 0.3f, s4, w, sp, 1 << 30) == 1) {
    o.feas |= bsc::FE_GRASP;
    o.gst4[0] = s4.x; o.gst4[1] = s4.y; o.gst4[2] = s4.yaw; o.gst4[3] = s4.dp;
  } else o.feas |= FR_NOSTANCE << 8;
  // B6 잡기: 어디서 와도(P 로 길 → 돌기 → 들어감)
  StanceOut so;
  sp = SPos{0, 0, -1, -1};
  const bool g = stance_grasp(c, b, ss, E, p, nc, 1e9f, 1e9f, 0.f, false, 0.f, so, w, sp, 1 << 30) == 1;
  if (g) {
    o.gst[0] = so.x; o.gst[1] = so.y; o.gst[2] = so.yaw; o.gst[3] = so.dp;
    for (int k = 0; k < 4; ++k) o.grel[k] = so.rel[k];
  }
  // B5 시작 쥠
  feas_start(ss, E, ent, bsc::EK_B5, c, b, p);
  if (mass_of(E.mass) > payload_max(0.25f, 0.f)) o.feas |= FR_HEAVY << 16;
  else {
    p.w = minf(minf(E.odim[0], E.odim[1]), KG::max_w);
    p.ryaw = E.odim[0] <= E.odim[1] ? 1.5707963f : 0.f;
    c.q[5] = grip_angle_of(p.w);
    p.st = OS_HELD;
    p.fl = OF_PICKED;
    b5_start_yaw(c, b, ss, E, p);   // 환경 B5 시작과 같은 yaw 고르기
    StanceOut s5;
    sp = SPos{0, 0, -1, -1};
    if (stance_place(c, b, ss, E, p, nc, 1e9f, 1e9f, 0.f, true, 0.3f, s5, w, sp, 1 << 30) == 1) {
      o.feas |= bsc::FE_PLACE5;
      o.pst5[0] = s5.x; o.pst5[1] = s5.y; o.pst5[2] = s5.yaw; o.pst5[3] = s5.dp;
    } else o.feas |= FR_NOPLACE << 16;
  }
  // B6: 잡은 쥠으로 gst 에서
  if (g) {
    feas_start(ss, E, ent, bsc::EK_B6, c, b, p);
    c.x = so.x; c.y = so.y; c.yaw = so.yaw;
    p.rel[0] = so.rel[0]; p.rel[1] = so.rel[1]; p.rel[2] = so.rel[2]; p.ryaw = so.rel[3];
    p.w = so.ap.w;
    p.st = OS_HELD;
    p.fl = OF_PICKED;
    PState ph;
    held_at(c, c.q, p, ph);
    StanceOut s6;
    sp = SPos{0, 0, -1, -1};
    if (stance_place(c, b, ss, E, ph, nc, 1e9f, 1e9f, 0.f, false, 0.f, s6, w, sp, 1 << 30) == 1) {
      o.feas |= bsc::FE_PLACE6;
      o.pst6[0] = s6.x; o.pst6[1] = s6.y; o.pst6[2] = s6.yaw; o.pst6[3] = s6.dp;
    } else o.feas |= FR_NOPLACE << 24;
  } else o.feas |= FR_NOSTANCE << 24;
}

// ---- 판마다: 1. 앞(사건·요청) ----
DEV bool target_known(const bsc::NavFb& fb, int i, int ep, int kind) {   // B6 이고 지도가 붙었으면 목표가 지도에 확정된 뒤에만 물체 자리를 씀(POLICY 4.2)
  if (kind != bsc::EK_B6 || fb.conf == nullptr) return true;
  return fb.tag[i] == ep && fb.conf[i] != 0;
}
DEV void set_stance(TState& t, const float s[4]) {
  t.naq = 0; t.cfg = -1;   // 팔 계획은 아직(계획에서 그 자리로 다시)
  t.sx = s[0]; t.sy = s[1]; t.syaw = s[2]; t.dpre = s[3];
  float sn, cs;
  sincosf_d(s[2], &sn, &cs);
  t.px = s[0] - s[3] * cs; t.py = s[1] - s[3] * sn;
}
// 반환 = 계획이 필요함
DEV bool teacher_pre(const Soa& s, const TBuf& tb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb) {
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  TState t;
  load_t(s, tb, i, t);
  if (!is_pnp(b.kind)) {
    if (t.ep != c.ep) { t.ep = c.ep; t.ph = TP_DONE; t.need = 0; t.nwp = 0; store_t(s, tb, i, t); }
    return false;
  }
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  const bool held = p.st == OS_HELD;
  if (t.ep != c.ep) {   // 판 시작: 표의 서는 자리가 있으면 길만, 없으면 찾기
    t.ep = c.ep; t.tm = 0; t.tr = 0; t.sok = 0; t.sd = 1e9f;
    t.need = 0; t.nwp = 0; t.kwp = 0; t.naq = 0; t.kaq = 0; t.fail = 0; t.stk = 0; t.part = 0; t.exp0 = 0; t.exp1 = 0; t.nplan = 0; t.cfg = -1; t.cphi = 0.f;
    t.ax = 1e9f; t.ay = 1e9f; t.ayaw = 0.f; t.exy = 0.f;
    t.ox = p.o[0]; t.oy = p.o[1]; t.oz = p.o[2];
    t.held0 = held ? 1 : 0;
    t.ph0 = -1;
    const bool tab = ss.has_feas != 0;
    // 표의 자리(B4 gst4·B5 pst5 = 시작 칸에서 바로 가는 자리 먼저, B6 gst = 어디서 와도)로 길·그 자리 팔 계획만. 표가 없으면 찾기
    if (b.kind == bsc::EK_B5) {
      t.ph = TP_NAV2;
      if (tab && (E.feas & bsc::FE_PLACE5)) { set_stance(t, E.pst5); t.sok = 2; t.need = TN_PATH; }
      else t.need = TN_SP;
    } else if (!target_known(fb, i, c.ep, b.kind)) { t.ph = TP_EXPLORE; t.need = TN_EXPL; }
    else {
      t.ph = TP_NAV;
      if (tab && b.kind == bsc::EK_B4 && (E.feas & bsc::FE_GRASP)) { set_stance(t, E.gst4); t.sok = 1; t.need = TN_PATH; }
      else if (tab && b.kind == bsc::EK_B6 && (E.feas & bsc::FE_PLACE6)) { set_stance(t, E.gst); t.sok = 1; t.need = TN_PATH; }
      else t.need = TN_SG;
    }
  } else {
    const bool goal = !held && (p.fl & OF_PICKED) && at_goal(ss, b, E, p, E.odim);
    const bool carrying = t.ph == TP_LIFT || t.ph == TP_BACK || t.ph == TP_FOLD || t.ph == TP_NAV2 || t.ph == TP_APP2 || t.ph == TP_PREPL;
    if (t.ph != TP_DONE && t.ph != TP_RETREAT && t.ph != TP_OPEN) {
      if (carrying && !held && !goal) {   // 떨어뜨림·미끄러짐: 물체 참 자리에서 다시 잡기
        ++t.tr; t.fail = FR_DROPS;
        t.ph = TP_NAV; t.sok = 0; t.nwp = 0; t.need = TN_SG; t.ax = 1e9f;
        t.ox = p.o[0]; t.oy = p.o[1]; t.oz = p.o[2];
      } else if (goal) { t.ph = TP_RETREAT; t.kaq = 0; t.need = 0; }
      else if (held && (t.ph <= TP_CLOSE || t.ph == TP_EXPLORE)) { t.ph = TP_LIFT; t.kaq = 3; t.need = 0; }
      else if (t.ph == TP_EXPLORE && target_known(fb, i, c.ep, b.kind)) {   // 지도에 확정됨: 물체로
        t.ph = TP_NAV; t.nwp = 0;
        const bool same = absf(p.o[0] - t.ox) < 1e-4f && absf(p.o[1] - t.oy) < 1e-4f;
        if (ss.has_feas && (E.feas & bsc::FE_PLACE6) && same) { set_stance(t, E.gst); t.sok = 1; t.need = TN_PATH; }
        else t.need = TN_SG;
      } else if (!held && (t.ph == TP_NAV || t.ph == TP_APP || t.ph == TP_PRE || t.ph == TP_DOWN)) {   // 물체가 움직임(밀림 등)
        const float dx = p.o[0] - t.ox, dy = p.o[1] - t.oy, dz = p.o[2] - t.oz;
        if (dx * dx + dy * dy + dz * dz > 0.02f * 0.02f) { t.ph = TP_NAV; t.sok = 0; t.nwp = 0; t.need = TN_SG; t.ox = p.o[0]; t.oy = p.o[1]; t.oz = p.o[2]; }
      }
    }
    if (t.tr > KT::max_try && t.ph != TP_DONE && !held) { t.ph = TP_DONE; t.need = 0; }
  }
  t.held0 = held ? 1 : 0;
  store_t(s, tb, i, t);
  return t.need != 0;
}

// ---- 판마다: 2. 계획(요청한 판만) ----
TDEV void teacher_plan(const Soa& s, const TBuf& tb, int i, const bsc::SceneSet& ss, uint8_t* scr, const WCtx& w) {
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  TState t;
  load_t(s, tb, i, t);
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  int need = t.need;
  t.need = 0;
  ++t.nplan;
  TSEG_INIT
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
  if (w.lane == 0 && (need & (TN_SG | TN_SP)) && !(need & TN_CONT)) atomicAdd(&g_tseg[10 + ((need & TN_SP) ? 1 : 0)][0], 1ull), atomicAdd(&g_tcause[(need & TN_SP) ? 1 : 0][t.fail & 15], 1ull);
#endif
  auto put_arm = [&](const ArmPlan& ap) {
    for (int k = 0; k < T_NAQ; ++k) for (int j = 0; j < 5; ++j) t.q[5 * k + j] = ap.q[k][j];
    t.naq = ap.n; t.open = ap.open; t.w = ap.w; t.cfg = ap.cfg; t.cphi = ap.cphi;
  };
  auto fail_stance = [&]() { t.ax = t.sx; t.ay = t.sy; t.ayaw = t.syaw; };
  if (need & TN_AG) {   // 자리에 옴: 그 자리 팔 계획 — 서는 자리 계획과 같은 자세를 먼저(대개 이것), 안 되면 모든 자세(촘촘한 기울기, 지금 팔에 가까운 것)
    ArmPlan ap;
    float rel[4];
    bool quick = false;
    if (t.cfg >= 0 && t.naq > 0) {
      ArmCand ac;
      arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
      float lo[3], hi[3];
      obj_box(p, E.odim, lo, hi);
      const float sup = support_z(ss, b, E, p, p.o[0], p.o[1], lo[2], E.odim);
      const bool down = (t.cfg & 4) != 0;
      for (int dj = 0; dj < 5 && !quick; ++dj)   // 같은 자세, 기울기 0, ±0.05, ±0.1
        quick = arm_try_grasp(c, b, ss, E, p, ac, grasp_z(down, lo, hi, sup, p.o[2]), down, t.cphi + (dj & 1 ? 0.05f : -0.05f) * (float)((dj + 1) >> 1),
                              t.cfg & 1, (t.cfg >> 1) & 1, ap, rel);
    }
    if (quick || arm_grasp_here(c, b, ss, E, p, ap, rel, true, w)) { put_arm(ap); t.ph = TP_PRE; t.kaq = 0; t.ox = p.o[0]; t.oy = p.o[1]; t.oz = p.o[2]; }
    else { ++t.tr; t.fail = FR_ARM; fail_stance(); t.ph = TP_NAV; t.sok = 0; t.nwp = 0; need |= TN_SG; }
    TSEG(0);
  }
  if (need & TN_AP) {   // 같은 자세(같은 놓을 가운데 후보) 먼저
    ArmPlan ap;
    float tc[3];
    bool quick = false;
    if (t.cfg >= 0 && t.naq > 0 && place_center_k(c, b, E, p, t.cfg >> 3, tc)) {
      ArmCand ac;
      arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
      for (int dj = 0; dj < 5 && !quick; ++dj)   // 같은 자세, 기울기 0, ±0.05, ±0.1
        quick = arm_try_place(c, b, ss, E, p, ac, tc, t.cphi + (dj & 1 ? 0.05f : -0.05f) * (float)((dj + 1) >> 1), t.cfg & 1, ap);
      if (quick) ap.cfg |= (t.cfg >> 3) << 3;
    }
    if (quick || arm_place_here(c, b, ss, E, p, ap, tc, w)) { put_arm(ap); t.ph = TP_PREPL; t.kaq = 0; t.tcx = tc[0]; t.tcy = tc[1]; t.tcz = tc[2]; }
    else { ++t.tr; t.fail = FR_ARM; fail_stance(); t.ph = TP_NAV2; t.sok = 0; t.nwp = 0; need |= TN_SP; }
    TSEG(1);
  }
  if (t.tr > KT::max_try && !(p.st == OS_HELD)) { t.ph = TP_DONE; if (w.lane == 0) store_t(s, tb, i, t); return; }
  if (!(need & (TN_SG | TN_SP | TN_PATH | TN_EXPL))) { if (w.lane == 0) store_t(s, tb, i, t); return; }
  const TScr S = scr_of(scr);
  // 길만 필요하고 P 가 로봇 자리(5 cm 안 — 표의 "바로 가는 자리")면 BFS 없이
  const bool trivial = !(need & (TN_SG | TN_SP | TN_EXPL)) && (t.px - c.x) * (t.px - c.x) + (t.py - c.y) * (t.py - c.y) < 0.05f * 0.05f;
  // 서는 자리 찾기를 이어 하는 조각: 첫 조각이 적어 둔 닿는 칸 비트(tb.rb)를 씀 — BFS 없음
  const bool scont = (need & (TN_SG | TN_SP)) && (need & TN_CONT) && tb.rb;
  uint32_t* rbi = tb.rb ? tb.rb + (size_t)i * T_RBW : nullptr;
  // 길만이면 목표 칸에 닿으면 BFS 를 멈춤(서는 자리 찾기는 창 전체 단계가 필요)
  bool have_lev = false;
  if (!trivial && !scont) {
    if (need & (TN_SG | TN_SP | TN_EXPL)) tch_reach(c, b, ss, E, p, S, w);
    else tch_reach(c, b, ss, E, p, S, w, t.px, t.py);
    have_lev = true;
    if ((need & (TN_SG | TN_SP)) && rbi) {   // 닿는 칸(3 × 3 이웃 넓힘) 비트를 적어 둠 — 이어 하는 조각용
      for (int k = w.lane; k < T_RBW; k += w.nl) {
        uint32_t m = 0u;
        for (int j = 0; j < 32; ++j) {
          const int wc = 32 * k + j, cc = wc % bsc::WIN, rr = wc / bsc::WIN;
          bool any = false;
          for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
              const int c2 = cc + dc, r2 = rr + dr;
              any = any || (c2 >= 0 && r2 >= 0 && c2 < bsc::WIN && r2 < bsc::WIN && S.lev0[r2 * bsc::WIN + c2] < 255);
            }
          if (any) m |= 1u << j;
        }
        rbi[k] = m;
      }
      w_sync(w);
    }
  }
  TSEG(2);
  const NavCtx nc = scont ? NavCtx{nullptr, rbi} : NavCtx{S.lev0, nullptr};
  if (need & (TN_SG | TN_SP)) {
    StanceOut so;
    const bool pl = (need & TN_SP) != 0;
    if (!(need & TN_CONT)) { t.skc = 0; t.spos = 0; t.sg1 = -1; t.sg2 = -1; t.snp = 0; }   // 새 찾기
    ++t.snp;
    SPos sp{t.skc, t.spos, t.sg1, t.sg2};
    const int r = pl ? stance_place(c, b, ss, E, p, nc, t.ax, t.ay, t.ayaw, true, 0.f, so, w, sp, KT::plan_budget)
                     : stance_grasp(c, b, ss, E, p, nc, t.ax, t.ay, t.ayaw, true, 0.f, so, w, sp, KT::plan_budget);
    t.skc = sp.kc; t.spos = sp.pos; t.sg1 = sp.g1; t.sg2 = sp.g2;
    TSEG(pl ? 4 : 3);
    if (r == 2) {   // 다음 스텝에 이어서(멈춰 기다림)
      t.need = (pl ? TN_SP : TN_SG) | TN_CONT;
      if (w.lane == 0) store_t(s, tb, i, t);
      return;
    }
    const bool ok = r == 1;
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
    if (w.lane == 0) { atomicAdd(&g_tseg[12 + (ok ? 0 : 1)][0], 1ull); atomicAdd(&g_tseg[12 + (ok ? 0 : 1)][1], (unsigned long long)t.snp); }
#endif
    if (ok) {
      const float st4[4] = {so.x, so.y, so.yaw, so.dp};
      set_stance(t, st4);
      put_arm(so.ap);
      t.stk = 0; t.sd = 1e9f;
      t.sok = pl ? 2 : 1;
      t.ox = p.o[0]; t.oy = p.o[1]; t.oz = p.o[2];
      need |= TN_PATH;
    } else {   // 지금 상태로는 같은 답 — 포기(까닭)
      t.fail = pl ? FR_NOPLACE : FR_NOSTANCE;
      t.ph = TP_DONE; t.tr = KT::max_try + 1;
      if (w.lane == 0) store_t(s, tb, i, t);
      return;
    }
  } else if ((need & TN_PATH) && t.sok && t.naq == 0) {   // 표의 자리: 그 자리 팔 계획으로 잡기 전 자세(들어갈 때 팔)
    Core u = c;
    u.x = t.sx; u.y = t.sy; u.yaw = t.syaw;
    ArmPlan ap;
    float tmp[4];
    const bool ok = t.sok == 2 ? arm_place_here(u, b, ss, E, p, ap, tmp, w) : arm_grasp_here(u, b, ss, E, p, ap, tmp, false, w);
    TSEG(5);
    if (ok) put_arm(ap);
    else { ++t.tr; t.fail = FR_ARM; fail_stance(); t.nwp = 0; t.need = t.sok == 2 ? TN_SP : TN_SG; t.sok = 0; if (w.lane == 0) store_t(s, tb, i, t); return; }
  }
  if ((need & TN_PATH) && !trivial && !have_lev) tch_reach(c, b, ss, E, p, S, w, t.px, t.py);   // 이어 한 찾기가 끝남: 길 BFS(목표에서 멈춤)
  if ((need & TN_PATH) && trivial) {
    t.nwp = 1; t.kwp = 0; t.part = 0; t.sd = 1e9f; t.stk = 0; t.wp[0] = t.px; t.wp[1] = t.py;
    t.ph = t.sok == 2 ? TP_NAV2 : TP_NAV;
  } else if (need & TN_PATH) {
    int part = 0;
    const bool wide = lev_near(S.lev1, t.px, t.py);
    int m = tch_extract(wide ? S.lev1 : S.lev0, wide ? S.blk1 : S.blk0, c.x, c.y, t.px, t.py, t.wp, part);
    if (m > 0) {
      t.nwp = m; t.kwp = 0; t.part = part;   // 막힘 수(stk)·남은 길 기준(sd)은 그대로 — 길 다시가 거듭되면 자리 다시
      t.ph = t.sok == 2 ? TP_NAV2 : TP_NAV;
    } else {
      ++t.tr; t.fail = FR_NOPATH; fail_stance(); t.nwp = 0;
      t.need = t.sok == 2 ? TN_SP : TN_SG;
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
      if (w.lane == 0) atomicAdd(&g_tseg[14 + (have_lev ? 0 : 1)][0], 1ull);
#endif
      t.sok = 0;
    }
    TSEG(6);
  }
  if (need & TN_EXPL) {   // 탐사: 창 격자점(1.6 m) 중 안 가 본 것 중 가장 가까운(길) 곳 — 물체 자리는 안 씀
    int best = -1, bl = 255;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
      if (pass) { t.exp0 = 0; t.exp1 = 0; }
      for (int k = 0; k < 64; ++k) {
        if (((k < 32 ? t.exp0 : t.exp1) >> (k & 31)) & 1) continue;
        const float gx = -bsc::WIN_HALF + KT::exp_sp * (0.5f + (float)(k & 7)), gy = -bsc::WIN_HALF + KT::exp_sp * (0.5f + (float)(k >> 3));
        const int c2 = wcell(gx), r2 = wcell(gy);
        const int L = S.lev1[r2 * bsc::WIN + c2];
        if (L < bl && L > 3) { bl = L; best = k; }
      }
    }
    if (best >= 0) {
      if (best < 32) t.exp0 |= 1 << best; else t.exp1 |= 1 << (best - 32);
      const float gx = -bsc::WIN_HALF + KT::exp_sp * (0.5f + (float)(best & 7)), gy = -bsc::WIN_HALF + KT::exp_sp * (0.5f + (float)(best >> 3));
      int part = 0;
      t.nwp = tch_extract(S.lev1, S.blk1, c.x, c.y, gx, gy, t.wp, part);
      t.kwp = 0; t.part = 0; t.kaq = 0; t.sd = 1e9f; t.stk = 0;
      t.ph = TP_EXPLORE;
    } else { t.ph = TP_EXPLORE; t.nwp = 0; t.kaq = 1; t.exy = c.yaw; }
  }
  if (w.lane == 0) store_t(s, tb, i, t);
}

// ---- 판마다: 3. 행동(저장한 계획 따르기) ----
// 웨이포인트 따르기(순수 추종): 반환 = 마지막 점에 옴
DEV bool follow_wp(const Core& c, TState& t, float& v, float& w, float end_r = KT::end_r) {
  v = 0.f; w = 0.f;
  if (t.nwp <= 0) return false;
  int k = t.kwp;
  while (k < t.nwp - 1) {
    const float dx = t.wp[2 * k] - c.x, dy = t.wp[2 * k + 1] - c.y;
    if (dx * dx + dy * dy < KT::wp_r * KT::wp_r) ++k; else break;
  }
  t.kwp = k;
  const float dx = t.wp[2 * k] - c.x, dy = t.wp[2 * k + 1] - c.y, d = sqrtf(dx * dx + dy * dy);
  const bool last = k == t.nwp - 1;
  if (last && d < end_r) return true;
  float h = wrap_pi(atan2f_d(dy, dx) - c.yaw);
  const bool back = absf(h) > 1.9f && d < 0.8f;   // 가까운 뒤쪽 점은 뒤로(좁은 곳에서 제자리 돌기를 줄임)
  if (back) h = wrap_pi(h + kPi);
  w = clampf(2.5f * h, -1.f, 1.f) * K::w_max;
  if (absf(h) > 0.5f) return false;
  const float vt = last ? minf(KT::nav_v, maxf(0.05f, 1.2f * d)) : KT::nav_v;
  v = vt * (1.f - 1.2f * absf(h)) * (back ? -0.6f : 1.f);
  return false;
}
DEV float path_left(const Core& c, const TState& t) {   // 남은 길(지금 → 웨이포인트들)
  if (t.nwp <= 0) return 0.f;
  float s = 0.f, x = c.x, y = c.y;
  for (int k = t.kwp; k < t.nwp; ++k) {
    const float dx = t.wp[2 * k] - x, dy = t.wp[2 * k + 1] - y;
    s = s + sqrtf(dx * dx + dy * dy);
    x = t.wp[2 * k]; y = t.wp[2 * k + 1];
  }
  return s;
}
// 몸통이 0.3 s 뒤 닿을 것 같으면 멈추고(돌기만), 돌기도 막히면 뒤로
DEV void teacher_guard(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, float& v, float& w) {
  auto free_at = [&](float vv, float ww) {
    float s2, c2;
    sincosf_d(c.yaw + 0.15f * ww, &s2, &c2);
    return body_free_pnp(ss, b, E, p, c.x + 0.3f * vv * c2, c.y + 0.3f * vv * s2, c.yaw + 0.3f * ww);
  };
  if (free_at(v, w)) return;
  const float wa = w != 0.f ? w : 0.3f * K::w_max;
  const float cand[7][2] = {{0.f, w}, {v, 0.f}, {0.f, -wa}, {-0.1f, 0.f}, {-0.1f, wa}, {-0.1f, -wa}, {0.08f, 0.f}};
  for (int k = 0; k < 7; ++k)
    if ((cand[k][0] != 0.f || cand[k][1] != 0.f) && free_at(cand[k][0], cand[k][1])) { v = cand[k][0]; w = cand[k][1]; return; }
  v = 0.f; w = 0.f;
}
// 들고 다닐 때 팔 검사 캐시(2026-10-06): 팔 자세(c.q)는 이 스텝 동안 그대로라 순기구학 둘(손 축 g = 0 — held_at, 팔 점 g = c.q[5] — arm_hits)을 한 번만,
// 후보 상자는 팔 점·든 물체 높이 띠와 겹치는 것만(높이는 베이스 자세와 무관 — 빠진 상자는 어떤 자세에서도 닿을 수 없음 → 같은 답)
struct CarryCache { Fk f0, fa; ArmCand ac; float R; };   // R: 베이스 가운데에서 팔 점(반경 × √2)·든 물체 바닥 자국이 닿는 평면 거리 상한 + 1 cm
DEV void carry_cache(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, const ArmCand& ac, CarryCache& cc) {
  float qq[N_Q], qd[N_Q];
  for (int k = 0; k < N_Q; ++k) { qq[k] = k < 5 ? c.q[k] : 0.f; qd[k] = 0.f; }
  fk(qq, qd, cc.f0);
  fk(c.q, qd, cc.fa);
  cc.ac = ac;
  if (ac.overflow) return;
  // 높이 띠: 팔 점(arm_collides 와 같은 여덟 점) + 든 물체 상자(지금 자세에서 — 세계 높이는 평면 이동과 무관)
  float zl = 1e9f, zh = -1e9f;
  {
    const Fk& f = cc.fa;
    float gp[3];
    grasp_point_base(f, gp);
    const float hg = 0.5f * grip_gap_of(c.q[5]);
    const float tip = gp[2] + (KG::tip_front - KG::tip_in) * f.ee_R[6];
    const float zz[8] = {0.5f * (f.p[1][2] + f.p[2][2]), f.p[2][2], 0.5f * (f.p[2][2] + f.p[3][2]), f.p[3][2], f.p[4][2], f.p[4][2] + KG::palm_d * f.ee_R[6],
                         tip + hg * f.ee_R[7], tip - hg * f.ee_R[7]};
    for (int k = 0; k < 8; ++k) { zl = minf(zl, zz[k] + 0.15f); zh = maxf(zh, zz[k] + 0.15f); }
  }
  {
    Hand h;
    hand_world(cc.f0, c.x, c.y, 0.f, 1.f, h);
    PState ph = p;
    for (int a = 0; a < 3; ++a) ph.o[a] = h.p[a] + p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a];
    zl = minf(zl, ph.o[2] - 0.5f * E.odim[2]); zh = maxf(zh, ph.o[2] + 0.5f * E.odim[2]);
  }
  zl = zl - KG::r_link - 0.03f; zh = zh + KG::r_link + 0.03f;   // 반경 + 3 cm 여유(넣는 쪽으로만 틀림 — 같은 답)
  {   // 평면 반경(베이스 틀): 점 |xy| + r·√2(pt_in_obb 는 네모로 넓힘), 든 물체 가운데 |xy| + 바닥 자국 반 대각
    const Fk& f = cc.fa;
    float gp[3];
    grasp_point_base(f, gp);
    const float hg = 0.5f * grip_gap_of(c.q[5]);
    float R = 0.f;
    auto pr = [&](float x, float y, float r) { R = maxf(R, sqrtf(x * x + y * y) + 1.4143f * r); };
    for (int k = 1; k <= 4; ++k) pr(f.p[k][0], f.p[k][1], KG::r_link);
    pr(f.p[4][0] + KG::palm_d * f.ee_R[0], f.p[4][1] + KG::palm_d * f.ee_R[3], KG::r_link);
    const float tx = gp[0] + (KG::tip_front - KG::tip_in) * f.ee_R[0], ty = gp[1] + (KG::tip_front - KG::tip_in) * f.ee_R[3];
    pr(tx + hg * f.ee_R[1], ty + hg * f.ee_R[4], KG::r_tip);
    pr(tx - hg * f.ee_R[1], ty - hg * f.ee_R[4], KG::r_tip);
    Hand h;
    hand_world(cc.f0, 0.f, 0.f, 0.f, 1.f, h);
    const float ox = h.p[0] + p.rel[0] * h.a[0] + p.rel[1] * h.n[0] + p.rel[2] * h.b[0], oy = h.p[1] + p.rel[0] * h.a[1] + p.rel[1] * h.n[1] + p.rel[2] * h.b[1];
    R = maxf(R, sqrtf(ox * ox + oy * oy) + 0.5f * sqrtf(E.odim[0] * E.odim[0] + E.odim[1] * E.odim[1]));
    cc.R = R + 0.01f;
  }
  const bsc::SceneDev& d = ss.sc[b.scene];
  int n = 0;
  for (int q = 0; q < ac.n; ++q) {
    const bsc::SBox& B = d.box[ac.idx[q]];
    if (B.z1 > zl && B.z0 < zh) cc.ac.idx[n++] = ac.idx[q];
  }
  cc.ac.n = n;
  int np = 0;
  for (int q = 0; q < ac.np; ++q) {
    const bsc::BPrim& P = E.prim[ac.pidx[q]];
    if (P.hi[2] > zl && P.lo[2] < zh) cc.ac.pidx[np++] = ac.pidx[q];
  }
  cc.ac.np = np;
}
// 베이스 자세 (x, y, yaw) 에서 몸통이 안 닿고, 들고 있으면(cc != nullptr) 지금 팔 자세 + 든 물체도 안 닿나(held_at·arm_hits 와 같은 식, 순기구학은 캐시)
DEV bool nav_free(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, const CarryCache* cc, float x, float y, float yaw) {
  if (!body_free_pnp(ss, b, E, p, x, y, yaw)) return false;
  if (!cc) return true;
  Core u = c;
  u.x = x; u.y = y; u.yaw = yaw;
  float sn, cs;
  sincosf_d(yaw, &sn, &cs);
  Hand h;
  hand_world(cc->f0, x, y, sn, cs, h);
  PState ph = p;
  for (int a = 0; a < 3; ++a) ph.o[a] = h.p[a] + p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a];
  set_yaw(ph, yaw + c.q[0] + p.ryaw);
  // 이 자세에서 반경 R 안에 드는 후보만(밖의 상자에는 어떤 팔 점·든 물체도 닿을 수 없음 → 같은 답)
  ArmCand an;
  an.overflow = cc->ac.overflow;
  if (an.overflow) an = cc->ac;
  else {
    an.n = 0; an.np = 0;
    const bsc::SceneDev& d = ss.sc[b.scene];
    for (int q = 0; q < cc->ac.n; ++q)
      if (bsc::dist_pt_obb2(d.box[cc->ac.idx[q]], x + b.wx, y + b.wy) < cc->R) an.idx[an.n++] = cc->ac.idx[q];
    for (int q = 0; q < cc->ac.np; ++q) {
      const bsc::BPrim& P = E.prim[cc->ac.pidx[q]];
      const float dx = maxf(maxf(P.lo[0] - x, x - P.hi[0]), 0.f), dy = maxf(maxf(P.lo[1] - y, y - P.hi[1]), 0.f);
      if (dx * dx + dy * dy < cc->R * cc->R) an.pidx[an.np++] = cc->ac.pidx[q];
    }
  }
  const bool hit = arm_collides(u, b, ss, E, ph, E.odim, cc->fa, an, sn, cs);
#if !defined(__CUDA_ARCH__)
  static const bool dbg_nf = std::getenv("TDBG_NAVFREE") != nullptr;
  if (dbg_nf) {   // 확인(CPU): 예전 식(held_at + arm_hits, 모은 후보 전부)과 같은 답
    ArmCand a0;
    arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, a0);
    PState p0;
    held_at(u, c.q, p, p0);
    const bool h0 = arm_hits(u, b, ss, E, p0, a0, c.q, c.q[5]);
    static long nchk = 0, nbad = 0;
    ++nchk;
    if (h0 != hit) { ++nbad; std::printf("NAVFREE MISMATCH %ld/%ld\n", nbad, nchk); }
    if ((nchk & ((1 << 14) - 1)) == 0) std::printf("navfree checks %ld mismatches %ld\n", nchk, nbad);
  }
#endif
  return !hit;
}
// 명령 (v, w) 를 0.2 s 따른 뒤 멈추는 동안(가속 한도 그대로 굴림, 0.05 s 표본) 안 닿나 — 지금 속도·관성까지 봄
DEV bool cmd_safe(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, const CarryCache* ac, float v_cmd, float w_cmd) {
  float x = c.x, y = c.y, yaw = c.yaw, v = c.v, w = c.w;
  const float dt = 0.05f;
  for (int k = 0; k < 16; ++k) {
    const float vt = k < 4 ? v_cmd : 0.f, wt = k < 4 ? w_cmd : 0.f;
    v = v + clampf(vt - v, -K::a_v * dt, K::a_v * dt);
    w = w + clampf(wt - w, -K::a_w * dt, K::a_w * dt);
    float sn, cs;
    sincosf_d(yaw + 0.5f * w * dt, &sn, &cs);
    x = x + v * cs * dt; y = y + v * sn * dt; yaw = yaw + w * dt;
    if ((k & 1) && !nav_free(c, b, p, ss, E, ac, x, y, yaw)) return false;
    if (k >= 4 && absf(v) < 1e-3f && absf(w) < 1e-3f) break;
  }
  return true;
}
// 지금에서 (v, w) 로 0.6 s(0.2 s 셋) 간 끝 자리(arc_free 와 같은 식, 검사 없음)
DEV void arc_end(const Core& c, float v, float w, float& ex, float& ey) {
  float x = c.x, y = c.y, yaw = c.yaw;
  for (int k = 0; k < 3; ++k) {
    float sn, cs;
    sincosf_d(yaw + 0.1f * w, &sn, &cs);
    x = x + 0.2f * v * cs; y = y + 0.2f * v * sn; yaw = yaw + 0.2f * w;
  }
  ex = x; ey = y;
}
// 지금에서 (v, w) 로 0.6 s(0.2 s 셋) 가는 동안 안 닿나
DEV bool arc_free(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, const CarryCache* ac, float v, float w, float& ex, float& ey) {
  float x = c.x, y = c.y, yaw = c.yaw;
  for (int k = 0; k < 3; ++k) {
    float sn, cs;
    sincosf_d(yaw + 0.1f * w, &sn, &cs);
    x = x + 0.2f * v * cs; y = y + 0.2f * v * sn; yaw = yaw + 0.2f * w;
    if (!nav_free(c, b, p, ss, E, ac, x, y, yaw)) return false;
  }
  ex = x; ey = y;
  return true;
}
// 막힌 곳 둘레 지역 계획(작은 DWA): 순수 추종 명령이 0.3 s 안에 닿으면 — (1) 제자리 돌기가 막혔으면 곧게 앞·뒤로 빠져(0.6 s 뒤 그 방향 0.3 rad 돌기가 되는 쪽)
// (2) 아니면 속도 5 × 회전 5 묶음을 0.6 s 굴려 안 닿는 것 중 (목표 거리 + 0.05·목표 방향 어긋남)이 가장 작은 것
DEV void local_cmd(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, const CarryCache* ac, float gx, float gy, float& v, float& w) {
  {
    float ex, ey;
    float s2, c2;
    sincosf_d(c.yaw + 0.15f * w, &s2, &c2);
    if (cmd_safe(c, b, p, ss, E, ac, v, w) && arc_free(c, b, p, ss, E, ac, 0.5f * v, 0.5f * w, ex, ey)) return;
  }
  if (v == 0.f && w != 0.f) {   // 돌기만 하려는데 막힘: 곧게 빠져 돌 자리 만들기
    const float sg = w > 0.f ? 1.f : -1.f;
    float bv = 0.f, bd = 1e30f;
    for (int k = 0; k < 2; ++k) {
      const float vv = k == 0 ? -0.12f : 0.12f;
      float ex, ey;
      if (!arc_free(c, b, p, ss, E, ac, vv, 0.f, ex, ey)) continue;
      if (!nav_free(c, b, p, ss, E, ac, ex, ey, c.yaw + 0.3f * sg)) continue;
      const float dd = (ex - gx) * (ex - gx) + (ey - gy) * (ey - gy);
      if (dd < bd) { bd = dd; bv = vv; }
    }
    if (bd < 1e30f && cmd_safe(c, b, p, ss, E, ac, bv, 0.f)) { v = bv; w = 0.f; return; }
  }
  const float vs[5] = {0.25f, 0.12f, 0.f, -0.1f, -0.2f}, ws[5] = {0.f, 0.5f, -0.5f, 1.f, -1.f};
  // 비용이 가장 작은, 굴림(arc_free)·멈춤(cmd_safe) 둘 다 안 닿는 묶음. 굴림 끝 자리·비용을 먼저 모두 재고(검사 없이), 비용 차례(같으면 앞 번호)로
  // 두 검사를 처음 되는 것까지만(2026-10-06 — 예전엔 25 개 모두 검사: 비용은 검사와 무관해 고른 답은 같음)
  float cost[25];
  for (int a = 0; a < 5; ++a)
    for (int k = 0; k < 5; ++k) {
      float& co = cost[a * 5 + k];
      co = 1e30f;
      if (vs[a] == 0.f && ws[k] == 0.f) continue;
      float ex, ey;
      arc_end(c, vs[a], ws[k] * K::w_max, ex, ey);   // 굴림 끝 자리(검사 없이 — arc_free 와 같은 식)
      const float yaw2 = c.yaw + 0.6f * ws[k] * K::w_max;
      const float he = absf(wrap_pi(atan2f_d(gy - ey, gx - ex) - yaw2));
      co = sqrtf((ex - gx) * (ex - gx) + (ey - gy) * (ey - gy)) + 0.05f * minf(he, kPi - he) + (vs[a] < 0.f ? 0.01f : 0.f);
    }
  float bv = 0.f, bw = 0.f;
  for (int it = 0; it < 25; ++it) {
    int bi = -1;
    for (int j = 0; j < 25; ++j) if (cost[j] < 1e30f && (bi < 0 || cost[j] < cost[bi])) bi = j;
    if (bi < 0) break;
    const float vv = vs[bi / 5], ww = ws[bi % 5] * K::w_max;
    float ex, ey;
    if (arc_free(c, b, p, ss, E, ac, vv, ww, ex, ey) && cmd_safe(c, b, p, ss, E, ac, vv, ww)) { bv = vv; bw = ww; break; }
    cost[bi] = 1e30f;
  }
#if !defined(__CUDA_ARCH__)
  static const bool dbg_dwa = std::getenv("TDBG_DWA") != nullptr;
  if (dbg_dwa) {   // 확인(CPU): 예전 고리(25 개 모두 검사)와 같은 답
    float best = 1e30f, ov = 0.f, ow = 0.f;
    for (int a = 0; a < 5; ++a)
      for (int k = 0; k < 5; ++k) {
        if (vs[a] == 0.f && ws[k] == 0.f) continue;
        float ex, ey;
        if (!arc_free(c, b, p, ss, E, ac, vs[a], ws[k] * K::w_max, ex, ey) || !cmd_safe(c, b, p, ss, E, ac, vs[a], ws[k] * K::w_max)) continue;
        const float yaw2 = c.yaw + 0.6f * ws[k] * K::w_max;
        const float he = absf(wrap_pi(atan2f_d(gy - ey, gx - ex) - yaw2));
        const float co = sqrtf((ex - gx) * (ex - gx) + (ey - gy) * (ey - gy)) + 0.05f * minf(he, kPi - he) + (vs[a] < 0.f ? 0.01f : 0.f);
        if (co < best) { best = co; ov = vs[a]; ow = ws[k] * K::w_max; }
      }
    static long nd = 0, nb = 0;
    ++nd;
    if (ov != bv || ow != bw) { ++nb; std::printf("DWA MISMATCH %ld/%ld\n", nb, nd); }
    if ((nd & 4095) == 0) std::printf("dwa checks %ld mismatches %ld\n", nd, nb);
  }
#endif
  v = bv; w = bw;
}
DEV float qerr(const Core& c, const float* q) {
  float e = 0.f;
  for (int k = 0; k < 5; ++k) e = maxf(e, absf(c.q[k] - q[k]));
  return e;
}
DEV void teacher_act(const Soa& s, const TBuf& tb, int i, const bsc::SceneSet& ss, float* act) {
  const int N = s.N;
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
  const long long ta0 = clock64();
  int ph_in = -1;
#endif
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  float a[N_ACT];
  for (int k = 0; k < N_ACT; ++k) a[k] = 0.f;
  if (!is_pnp(b.kind)) { for (int k = 0; k < N_ACT; ++k) act[k * N + i] = 0.f; return; }
  TState t;
  load_t(s, tb, i, t);
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  const bool held = p.st == OS_HELD;
  float qc[5], qtk[5];
  if (!carry_q(qc)) home_q(qc);
  tuck_q(qtk);
  float qt[5];
  for (int k = 0; k < 5; ++k) qt[k] = held ? qc[k] : qtk[k];
  float g = held ? 0.f : c.q[5];
  float v = 0.f, w = 0.f;
  const float* Q = t.q;
  bool guard = false;
  CarryCache ccn;   // 들고 다닐 때 지역 계획이 팔·든 물체 충돌도 봄
  const CarryCache* acp = nullptr;
  if (held && (t.ph == TP_NAV2 || t.ph == TP_APP2 || t.ph == TP_NAV)) {   // 둘레(0.65 m)에 아무 상자도 없으면 팔 검사 생략(바닥은 나르는 높이라 안 닿음)
    ArmCand acn;
    arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, acn);
    if (acn.n > 0 || acn.np > 0 || acn.overflow || p.oc[2] > 0.f) { carry_cache(c, b, p, E, ss, acn, ccn); acp = &ccn; }
  }
  if (t.ph != t.ph0) { t.tm = 0; t.ph0 = t.ph; }   // 앞·계획에서 단계가 바뀜: 단계 안 스텝 0 부터
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
  ph_in = t.ph;
#endif
  switch (t.ph) {
    case TP_NAV: case TP_NAV2: case TP_EXPLORE: {
      if (t.need || t.nwp <= 0) {
        if (t.ph == TP_EXPLORE && t.kaq == 1) {   // 탐사 점에서 한 바퀴(보기)
          w = 0.6f * K::w_max;
          if (t.tm > 120) { t.kaq = 0; t.need = TN_EXPL; }
        }
        break;
      }
      const bool end = follow_wp(c, t, v, w);
      if (!end && t.nwp > 0) local_cmd(c, b, p, ss, E, acp, t.wp[2 * t.kwp], t.wp[2 * t.kwp + 1], v, w);
      if (end) {
        v = 0.f; w = 0.f;
        if (t.ph == TP_EXPLORE) { t.nwp = 0; t.kaq = 1; t.tm = 0; t.exy = c.yaw; break; }
        if (t.part) { t.need = TN_PATH; break; }
        t.ph = t.ph == TP_NAV ? TP_APP : TP_APP2; t.kaq = 0;
        break;
      }
      if (t.tm > 0 && t.tm % KT::stuck_T == 0) {   // 막힘: 남은 길이 안 줄면 길 다시(3 번이면 자리 다시)
        const float rem = path_left(c, t);
        if (rem < 0.15f && t.kwp == t.nwp - 1 && !t.part && t.ph != TP_EXPLORE) {   // P 바로 앞에서 못 들어감: 다가가기 단계로(거기서 돌기·다시 찾기)
          t.ph = t.ph == TP_NAV ? TP_APP : TP_APP2; t.kaq = 0;
        } else if (rem > t.sd - KT::stuck_dd) {
          ++t.stk; t.sd = rem;
          if (t.stk > KT::max_stk) {
            ++t.tr; t.fail = FR_STUCK; t.stk = 0; t.ax = t.sx; t.ay = t.sy; t.ayaw = t.syaw; TDBG(27);
            t.need = t.ph == TP_EXPLORE ? TN_EXPL : t.ph == TP_NAV2 ? TN_SP : TN_SG;
          } else t.need = t.ph == TP_EXPLORE ? TN_EXPL : TN_PATH;
        } else t.sd = rem;
      }
      if (p.fl & OF_CONTACT) { v = -0.1f; w = 0.f; }   // 팔·든 물체가 닿아 막힘: 조금 물러남
      break;
    }
    case TP_APP: case TP_APP2: {   // P 에서 자리 yaw 로 돌기 → 팔을 잡기(놓기) 전 자세로 → 자리까지 곧게 → yaw 맞춤 → 그 자리 팔 계획
      const bool pl = t.ph == TP_APP2;
      if (t.tm > KT::t_app) { TDBG(28); ++t.tr; t.fail = FR_STUCK; t.ax = t.sx; t.ay = t.sy; t.ayaw = t.syaw; t.need = pl ? TN_SP : TN_SG; t.sok = 0; t.nwp = 0; t.ph = pl ? TP_NAV2 : TP_NAV; break; }
      float sn, cs;
      sincosf_d(t.syaw, &sn, &cs);
      const float ex = t.sx - c.x, ey = t.sy - c.y;
      const float along = ex * cs + ey * sn, lat = -ex * sn + ey * cs;
      const float eyaw = wrap_pi(t.syaw - c.yaw);
      const float pdx = t.px - c.x, pdy = t.py - c.y, pd2 = pdx * pdx + pdy * pdy;
      if (t.kaq == 0) {   // P 로(3 cm 안까지 — 막히면 지역 계획)
        if (pd2 > 0.03f * 0.03f && t.tm < 120) {
          TState u = t;
          u.nwp = 1; u.kwp = 0; u.wp[0] = t.px; u.wp[1] = t.py;
          follow_wp(c, u, v, w, 0.025f);
          local_cmd(c, b, p, ss, E, acp, t.px, t.py, v, w);
        } else t.kaq = 5;
      } else if (t.kaq == 5) {   // 제자리 돌기(막히면 곧게 빠져 돌 자리 만들기 — P 에서 12 cm 넘게 벗어나면 다시 P 로)
        w = clampf(2.5f * eyaw, -1.f, 1.f) * K::w_max;
        if (absf(eyaw) < 0.03f) t.kaq = 1;
        else if (t.tm > 70) { TDBG(29); ++t.tr; t.fail = FR_STUCK; t.ax = t.sx; t.ay = t.sy; t.ayaw = t.syaw; t.need = pl ? TN_SP : TN_SG; t.sok = 0; t.nwp = 0; t.ph = pl ? TP_NAV2 : TP_NAV; }
        else if (pd2 > 0.12f * 0.12f) t.kaq = 0;
        else local_cmd(c, b, p, ss, E, acp, t.px, t.py, v, w);
      } else if (t.kaq == 1) {
        for (int k = 0; k < 5; ++k) qt[k] = Q[k];
        g = held ? 0.f : grip_angle_of(t.open);
        w = clampf(2.f * eyaw, -0.5f, 0.5f) * K::w_max;
        if (qerr(c, Q) < 0.05f || t.tm > 45) t.kaq = 2;
      } else if (t.kaq == 2) {
        for (int k = 0; k < 5; ++k) qt[k] = Q[k];
        g = held ? 0.f : grip_angle_of(t.open);
        if (t.dpre >= 0.f ? along > KT::arrive : along < -KT::arrive) {   // 앞으로(또는 바로 갈 자리면 뒤로도) 곧게
          const float lc = t.dpre >= 0.f && along > 0.01f ? clampf(3.f * lat, -0.3f, 0.3f) : 0.f;
          const float hd = wrap_pi(t.syaw + lc - c.yaw);
          w = clampf(3.f * hd, -0.6f, 0.6f) * K::w_max;
          v = t.dpre >= 0.f ? clampf(2.5f * along, 0.02f, KT::app_v) : clampf(2.5f * along, -KT::app_v, -0.02f);
        } else t.kaq = 3;
        guard = true;
      } else {
        for (int k = 0; k < 5; ++k) qt[k] = Q[k];
        g = held ? 0.f : grip_angle_of(t.open);
        if (absf(eyaw) > KT::arrive_yaw) w = clampf(2.f * eyaw, -0.4f, 0.4f) * K::w_max;
        else if (absf(c.v) < 0.01f && absf(c.w) < 0.02f) t.need = pl ? TN_AP : TN_AG;
      }
      break;
    }
    case TP_PRE: case TP_DOWN: case TP_CLOSE: {
      const float go = grip_angle_of(t.open);
      if (t.ph == TP_PRE) {
        for (int k = 0; k < 5; ++k) qt[k] = Q[k];
        g = go;
        if (qerr(c, Q) < KT::q_tol && absf(c.q[5] - go) < KT::q_tol) { t.ph = TP_DOWN; t.kaq = 1; }
        else if (t.tm > KT::t_pre) { ++t.tr; t.fail = FR_PRETO; t.need = TN_AG; }
      } else if (t.ph == TP_DOWN) {
        const float* qq = Q + 5 * t.kaq;
        for (int k = 0; k < 5; ++k) qt[k] = qq[k];
        g = go;
        if (qerr(c, qq) < (t.kaq == 2 ? KT::q_tol_fine : KT::q_tol) || t.tm > KT::t_wp) {
          if (t.kaq < 2) { ++t.kaq; t.tm = -1; } else t.ph = TP_CLOSE;
        }
      } else {
        for (int k = 0; k < 5; ++k) qt[k] = Q[10 + k];
        g = 0.f;
        if (t.tm > KT::t_close) { ++t.tr; t.fail = FR_MISS; t.ph = TP_PRE; t.need = TN_AG; }   // 안 잡힘: 손을 다시 잡기 전 자리로·팔 다시
      }
      break;
    }
    case TP_LIFT: {   // 들기 1 → 들기 2(B4 는 거기서 멈춤 — 성공 판정), B5·B6 은 P 로 물러남
      if (t.kaq < 3) t.kaq = 3;
      const float* qq = Q + 5 * t.kaq;
      for (int k = 0; k < 5; ++k) qt[k] = qq[k];
      g = 0.f;
      if (qerr(c, qq) < KT::q_tol || t.tm > KT::t_lift) {
        if (t.kaq < 4) { t.kaq = 4; t.tm = -1; }
        else if (b.kind != bsc::EK_B4 && (p.fl & OF_PICKED)) { t.ph = TP_BACK; }
      }
      break;
    }
    case TP_BACK: {   // 들고 P 로 곧게 뒤로(들어온 길), 팔은 들기 2
      for (int k = 0; k < 5; ++k) qt[k] = Q[20 + k];
      g = 0.f;
      float sn, cs;
      sincosf_d(t.syaw, &sn, &cs);
      const float back = (c.x - t.px) * cs + (c.y - t.py) * sn;
      if (back > 0.01f && t.tm < 80) {
        v = -clampf(1.5f * back, 0.03f, 0.15f);
        w = clampf(2.f * wrap_pi(t.syaw - c.yaw), -0.4f, 0.4f) * K::w_max;
      } else t.ph = TP_FOLD;
      break;
    }
    case TP_FOLD: {   // 나르는 자세로 접음(관절 직선 5 점에서 잡는 점이 가반 하중 밖이면 그 자리 그대로 — 무게 특권). 다 접으면 놓기 쪽
      bool safe = true;
      float qd0[N_Q];
      for (int k = 0; k < N_Q; ++k) qd0[k] = 0.f;
      for (int j = 1; j <= 5 && safe; ++j) {
        float qq[N_Q];
        for (int k = 0; k < 5; ++k) qq[k] = c.q[k] + (qc[k] - c.q[k]) * (0.2f * (float)j);
        qq[5] = c.q[5];
        Fk f;
        fk(qq, qd0, f);
        float gp[3];
        grasp_point_base(f, gp);
        const float rx = gp[0] - KIK::j1x, ry = gp[1];
        safe = !(mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), f.ee_R[6]));
      }
      for (int k = 0; k < 5; ++k) qt[k] = safe ? qc[k] : c.q[k];
      g = 0.f;
      if (qerr(c, qc) < 0.05f || !safe || t.tm > 40) {
        t.ph = TP_NAV2; t.nwp = 0; t.sok = 0;
        const bool same = ss.has_feas && (E.feas & bsc::FE_PLACE6) && absf(p.rel[0] - E.grel[0]) < 0.01f && absf(p.rel[1] - E.grel[1]) < 0.01f &&
                          absf(p.rel[2] - E.grel[2]) < 0.01f && absf(wrap_pi(p.ryaw - E.grel[3])) < 0.1f && b.kind == bsc::EK_B6;
        if (same) { set_stance(t, E.pst6); t.sok = 2; t.need = TN_PATH; }
        else t.need = TN_SP;
      }
      break;
    }
    case TP_PREPL: {   // 놓기 전 → 가운데 → 놓기, 그다음 열기
      const float* qq = Q + 5 * t.kaq;
      for (int k = 0; k < 5; ++k) qt[k] = qq[k];
      g = 0.f;
      if (qerr(c, qq) < (t.kaq == 2 ? KT::q_tol_fine : KT::q_tol) || t.tm > KT::t_wp + (t.kaq == 0 ? 20 : 0)) {
        if (t.kaq < 2) { ++t.kaq; t.tm = -1; } else t.ph = TP_OPEN;
      }
      break;
    }
    case TP_OPEN: {
      for (int k = 0; k < 5; ++k) qt[k] = Q[10 + k];
      g = grip_angle_of(t.open);
      if (!held || t.tm > KT::t_open) { t.ph = TP_RETREAT; t.kaq = 0; }
      break;
    }
    case TP_RETREAT: {   // 물러나기 웨이포인트(열린 채), 다 오면 멈춤
      if (t.naq >= 4) for (int k = 0; k < 5; ++k) qt[k] = Q[15 + k];
      else for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = maxf(c.q[5], grip_angle_of(t.open));
      if (qerr(c, qt) < KT::q_tol || t.tm > KT::t_retreat) t.ph = TP_DONE;
      break;
    }
    default: {   // 끝: 지금 자세 그대로(놓은 뒤면 열린 채, 들었으면 쥔 채), 멈춤
      for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = held ? 0.f : c.q[5];
      break;
    }
  }
  if (guard) teacher_guard(c, b, p, ss, E, v, w);
  if (t.ph != t.ph0) { t.tm = 0; t.ph0 = t.ph; }   // 단계 안 스텝(앞·계획·행동 어디서 바뀌어도 0; 하위 단계는 −1 로 두어 0 부터)
  else t.tm = t.tm + 1;
  if (t.tm < 0) t.tm = 0;
  a[0] = clampf(v / K::v_max, -1.f, 1.f);
  a[1] = clampf(w / K::w_max, -1.f, 1.f);
  for (int k = 0; k < 5; ++k) a[2 + k] = act_of_q(k, qt[k]);
  a[7] = act_of_q(5, g);
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
  store_t(s, tb, i, t);
#if defined(ENV_PROF) && defined(__CUDA_ARCH__)
  const unsigned long long dc = (unsigned long long)(clock64() - ta0);
  atomicAdd(&g_tact[ph_in & 15][0], 1ull); atomicAdd(&g_tact[ph_in & 15][1], dc); atomicMax(&g_tact[ph_in & 15][2], dc);
#endif
}

// 한 판 전체(CPU 참조판·확인 도구): 앞 → (필요하면) 계획 → 행동. scr = T_SCR 바이트
DEV void teacher_step(const Soa& s, const TBuf& tb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb, uint8_t* scr, float* act) {
  if (teacher_pre(s, tb, i, ss, fb)) teacher_plan(s, tb, i, ss, scr, wseq());
  teacher_act(s, tb, i, ss, act);
}

}  // namespace env
