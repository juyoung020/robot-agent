// GPU 안에서 자라는 지도(scenemap 근사, 계획서 GPU_TRAINING.md 5.1·4.4, 단계 G2 앞부분).
// CPU 참조판과 GPU 커널이 **같은 소스**를 쓴다(DEV = __host__ __device__). 판 하나 = 블록 하나(NT 스레드). CPU 는 같은 함수를
// tid = 0, nt = 1 로 부른다. 결과가 스레드 순서와 무관하도록:
//   - 격자: 광선이 지나는 칸은 공유 메모리 비트표(맞음·빈칸)에 atomicOr 로 표시(순서 무관) → 칸마다 한 번 정수 덧셈.
//     scenemap grid.cpp 의 "한 스캔에서 칸마다 한 번, 맞음 우선" 규칙과 같다.
//   - 물체 보임 광선은 쌍마다 따로 칸에 쓰고, 검출·물체 기억(순서가 있는 탐욕 짝짓기)은 스레드 0 이 순서대로 한다.
//   - 난수는 detmath.h 의 splitmix64(판마다 따로, 스레드 0 만 뽑음).
//
// 규칙·기본값의 출처(읽기 전용): src/behavior-2026/src/scene_graph/scenemap
//   grid.hpp GridParams / grid.cpp insert, scan.hpp ScanParams / scan.cpp makeScan, slam2d.hpp SlamParams(움직임 거르기),
//   objmap.hpp ObjParams / objmap.cpp update(짝짓기·확정·옮겨짐·사라짐·버림), scenemap.h sm_object(SM_SEEN..).
// (가정) 표시 값은 실제 기록·사양으로 맞출 값이다. 한 곳(MP)에만 둔다.
#pragma once
#include "env.h"
#include "env_soa.h"

namespace gmap {
using namespace dm;

// 구간별 시간 재기(map_prof 빌드만, -DMAP_PROF). 블록의 스레드 0 이 clock64 차이를 구간 칸에 더한다. 보통 빌드에서는 아무것도 안 함
enum ProfSec { P_LOAD, P_BEGIN, P_RESET, P_CAST, P_POSE_DET, P_ASSOC, P_ABSENCE, P_OBJPRE, P_MARK, P_APPLY, P_FINISH, P_STORE, P_NSEC };
#if defined(MAP_PROF) && defined(__CUDACC__)
__device__ unsigned long long g_prof[P_NSEC + 3];   // + 블록 수(keyframe 아님, keyframe), 표시된 격자 낱말 수
#endif
#if defined(MAP_PROF) && defined(__CUDA_ARCH__)
__device__ __forceinline__ long long& prof_last() { __shared__ long long t; return t; }
#define PROF_START() do { if (threadIdx.x == 0) prof_last() = clock64(); } while (0)
#define PROF_MARK(k) do { if (threadIdx.x == 0) { const long long n_ = clock64(); atomicAdd(&g_prof[k], (unsigned long long)(n_ - prof_last())); prof_last() = n_; } } while (0)
#define PROF_COUNT(k) do { if (threadIdx.x == 0) atomicAdd(&g_prof[P_NSEC + (k)], 1ull); } while (0)
#else
#define PROF_START() do {} while (0)
#define PROF_MARK(k) do {} while (0)
#define PROF_COUNT(k) do {} while (0)
#endif

// ---- 크기 -------------------------------------------------------------------------------------------------------------
constexpr int GW = 128;                 // 격자 창 한 변 칸 수(고정 창, 원점 중심). G1 방은 최대 7 m × 7 m 라 12.8 m 창
constexpr float RES = 0.10f;            // 칸 0.10 m (계획서 8절; scenemap 기본 0.05)
constexpr float INV_RES = 10.0f;
constexpr int GX0 = -GW / 2;            // 창 첫 칸의 전역 칸 번호(칸 = floor(x / RES))
constexpr int NCELL = GW * GW;
constexpr int NWORD = NCELL / 32;
constexpr int NT = 64;                  // 블록 스레드 수
constexpr int NCOL = 64, NROW = 8;      // 거친 깊이 광선: 가로 64 (계획서 5.1 예) × 세로 8 (가정)
constexpr int M_SCN = 8;                // 장면 상자(가구 5 + 작은 물건 3) — G1 빈 방에 지도용으로 더함
constexpr int N_PRIM = M_SCN + 1;       // + 컵(과제 물체, 0 번)
constexpr int NPT = 9;                  // 물체마다 보임 광선 점(중심 + 0.8 배 줄인 모서리 8)
constexpr int KSLOT = 16;               // 물체 기억 칸 (가정; 계획서 8절 예는 64)
constexpr int MAXDET = N_PRIM + 1;      // 한 keyframe 검출 최대(참 물체 + 가짜 1)
constexpr int NCLS = 6;
constexpr int N_MET = 6;
enum Met { M_TASK, M_OBJ, M_SEEN, M_ERR_XY, M_ERR_YAW, M_KF };
enum Cls { C_CUP = 0, C_ITEM = 1, C_CHAIR = 2, C_TABLE = 3, C_CABINET = 4, C_BIN = 5 };
enum State { S_SEEN = 0, S_GONE = 1, S_MOVED = 2, S_HELD = 3 };   // scenemap.h SM_SEEN..SM_HELD

DEV constexpr int qround(float x) { return x >= 0.f ? (int)(x * 256.f + 0.5f) : -(int)(-x * 256.f + 0.5f); }   // lround(x · kQ)

// ---- 상수(한 곳) --------------------------------------------------------------------------------------------------------
struct MP {
  // grid.hpp GridParams + OccGrid::kQ = 256
  static constexpr int q_hit = qround(0.85f), q_miss = qround(-0.4f), q_min = qround(-4.f), q_max = qround(4.f);
  // scan.hpp ScanParams (광학 z 범위, 높이 띠)
  static constexpr float zmin = 0.3f, zmax = 8.0f, band_lo = 0.10f, band_hi = 1.80f;
  // slam2d.hpp SlamParams (움직임 거르기 update_policy 0, 제자리 규칙)
  static constexpr float mf_xy = 0.05f, mf_yaw = 0.034906585f /*2도*/, still_v = 0.01f, still_w = 0.01f;
  static constexpr int mf_kf = 50;
  // objmap.hpp ObjParams
  static constexpr int min_points = 20, min_px = 6, confirm = 2, gone_misses = 3;
  static constexpr float ozmin = 0.15f, ozmax = 5.0f, da_min = 0.30f, da_k = 0.5f, da_gap = 0.10f, big = 0.5f;
  static constexpr float moved_d = 0.15f, occl = 0.10f, grow_max = 0.25f, max_ext = 4.0f;
  static constexpr int prune_steps = 100, gone_min_steps = 20;   // prune_s 10 s, gone_min_s 2 s (제어 10 Hz, 영상 = 제어 스텝 (가정))
  // ---- (가정) ----
  static constexpr int img_w = 640, img_h = 400;                 // 깊이 영상 크기(가정: Orbbec Dabai 깊이 640×400), 정사각 화소
  static constexpr float wall_h = 2.5f;                          // 벽 높이(가정)
  static constexpr int min_hits = 5;                             // 맞추기 최소 맞은 줄 (가정: min_inliers 50 / 720 칸 × 64 줄 ≈ 5)
  static constexpr float kf_corr = 0.5f;                         // keyframe 맞추기가 되돌리는 오차 비율(가정)
  static constexpr float odo_t = 0.02f;                          // 이동 잡음 σ = 2 % × 거리(가정)
  static constexpr float odo_rr = 0.05f, odo_rt = 0.01f;         // 회전 잡음 σ = 0.05·|dθ| + 0.01 rad/m·거리(가정)
  static constexpr float p_miss = 0.10f, p_fp = 0.05f, p_conf = 0.03f;   // 놓침·가짜(keyframe 당)·틀린 이름 확률(가정)
  static constexpr float dn0 = 0.005f, dn2 = 0.004f;             // 깊이 잡음 σ = 0.005 + 0.004·z² m (가정)
  static constexpr float lat_n = 0.01f, ext_n = 0.02f;           // 옆·높이 잡음, 크기 잡음 σ m (가정)
  static constexpr float cup_h = 2.f * env::K::tgt_z;            // 컵 높이 0.10 m (가정: tgt_z 를 중심 높이로 봄)
};
DEV bool is_static(int cls) { return cls == C_TABLE || cls == C_CABINET; }   // capi.cpp kStaticNames 의 table·cabinet

// ---- 판마다 지도 상태 -------------------------------------------------------------------------------------------------
struct Prim {   // 정적 장면 상자(축 정렬), 바닥에 놓임
  int cls;
  float lo[3], hi[3];
};
struct Slot {   // 물체 기억 한 칸 (scenemap.h sm_object / objmap.hpp MapObject 에서)
  int valid, id, cls, n_obs, last_seen, last_kf, state, confirmed, moved, misses, first_miss;   // 시각 = 판 시작 뒤 제어 스텝
  float pos[3], ext[3], first_pos[3], score;
};
struct MapCore {
  uint64_t rng;
  int ep, t, first, since, n_kf, n_kf_total, next_id, room_cells, n_seen_room, n_task_conf, n_obj_conf, n_dropped, kf_flag, n_fp_total;
  float ex, ey, eyaw;      // slam 이 믿는 자세(참 + 오차)
  float px, py, pyaw;      // 지난 스텝 참 자세(오도메트리 증분)
  float lx, ly, lyaw;      // 지난 keyframe 자세(움직임 거르기)
  float vmax, wmax;        // 지난 keyframe 뒤 속도 최대(제자리 규칙)
  float rhx, rhy;          // 방 반치수(완성도 계산)
  Prim prim[N_PRIM];
  Slot slot[KSLOT];
};
static_assert(sizeof(MapCore) % 8 == 0, "MapCore must be whole 8-byte words (no tail padding)");
constexpr int CORE_WORDS = (int)(sizeof(MapCore) / 4);

struct Det { int cls; float pos[3], ext[3], score; };   // keyframe 한 번의 검출 하나
struct DetGeo { float ctr[3], ext[3], rx, ry, fwd, left, up, rh, af; };
constexpr int NPART = (NT + 31) / 32;

// 블록 안 작업 공간(GPU 공유 메모리, CPU 지역 변수)
struct Scratch {
  uint32_t hitb[NWORD], missb[NWORD];
  float colx[NCOL], coly[NCOL];     // 줄 끝(베이스 기준)
  int8_t colt[NCOL];                // 0 없음, 1 맞음(띠 안), 2 빈 광선 끝(바닥)
  uint32_t vism[N_PRIM];            // 물체마다 보이는 점 비트(NPT 개)
  int part[NPART];                  // 워프별 부분합(GPU) / 스레드 0 값(CPU): 맞은 열 수, 새로 본 칸 수, 짝 후보 번호(CPU)
  float partf[NPART];               // 짝 후보 열쇠(CPU 일반판)
  int pcand[N_PRIM];                // 보임 점과 무관한 검출 판정을 넘은 물체(후보)
  DetGeo geo[N_PRIM];               // 그 판정의 기하(검출이 다시 씀)
  float tc, ts, to[3];              // 참 yaw 의 cos·sin, 참 카메라 위치
  int found[N_PRIM];                // 완성도: 참 물체마다 확정 칸이 있나
  Det det[MAXDET];
  float key[MAXDET * KSLOT];        // 짝 열쇠(관측 × 칸), 짝 아님 = -1
  int obs_to[MAXDET], hit[KSLOT];
  int nd, more;
  int flags;                        // phase_begin 결과(B_KF, B_RESET)
  float ec, es;                     // 믿는 yaw 의 cos·sin
};

struct EnvView { float x, y, yaw, v, w, tx, ty, rhx, rhy; int ep; };
DEV EnvView read_env(const env::Soa& s, int i) {
  const int N = s.N;
  EnvView e;
  e.x = s.f[env::F_X * N + i]; e.y = s.f[env::F_Y * N + i]; e.yaw = s.f[env::F_YAW * N + i];
  e.v = s.f[env::F_V * N + i]; e.w = s.f[env::F_W * N + i];
  e.tx = s.f[env::F_TX * N + i]; e.ty = s.f[env::F_TY * N + i];
  e.rhx = s.f[env::F_RHX * N + i]; e.rhy = s.f[env::F_RHY * N + i];
  e.ep = s.iv[env::I_EP * N + i];
  return e;
}

// ---- 카메라(깊이): 베이스 앞 cam_x, 높이 cam_z, 기울기 0 (가정), 가로 시야 = env 의 71 도 -------------------------------
struct Cam { float tanh, tanv, fx, cxp, cyp; };
DEV Cam cam_consts() {
  float s, c;
  sincosf_d(0.5f * env::K::cam_hfov, &s, &c);
  Cam k;
  k.tanh = s / c;
  k.tanv = k.tanh * (float)MP::img_h / (float)MP::img_w;
  k.fx = 0.5f * (float)MP::img_w / k.tanh;
  k.cxp = 0.5f * (float)MP::img_w;
  k.cyp = 0.5f * (float)MP::img_h;
  return k;
}

DEV float gauss(uint64_t& r) {   // Irwin–Hall 4 개(분산 1 로 맞춤) — 초월함수 없이 CPU·GPU 같게
  float a = rand01(r);
  a = a + rand01(r);
  a = a + rand01(r);
  a = a + rand01(r);
  return (a - 2.f) * 1.7320508f;
}

DEV void or_bits(uint32_t* p, uint32_t v) {   // 공유 메모리 비트 OR(순서 무관)
#ifdef __CUDA_ARCH__
  atomicOr(p, v);
#else
  *p |= v;
#endif
}
// 블록 부분합: GPU 는 워프 합을 part[워프] 에(워프의 모든 레인이 불러야 함), CPU(nt 1) 는 part[0] 에
DEV void put_part(Scratch& sh, int tid, int v) {
#ifdef __CUDA_ARCH__
  v = (int)__reduce_add_sync(0xffffffffu, (unsigned)v);
  if ((tid & 31) == 0) sh.part[tid >> 5] = v;
#else
  sh.part[tid] = v;
#endif
}
DEV int sum_part(const Scratch& sh, int nt) {
  int s = 0;
  for (int k = 0; k < (nt + 31) / 32; ++k) s += sh.part[k];
  return s;
}

// ---- 광선 -------------------------------------------------------------------------------------------------------------
constexpr float kInf = 1e30f;
// 판(slab) 한 축: 들어가는·나가는 t 구간을 [t0, t1] 에 좁힌다. 축에 나란하고 밖이면 false.
// inv = 1/d 는 광선마다 한 번만 구한다(상자마다 다시 나눠도 같은 값이라 결과는 그대로). max·min 은 정확한 연산이라
// 축 순서·묶는 순서를 바꿔도 비트가 같다 — 그래서 열의 xy 판을 줄 8 개가 같이 쓸 수 있다.
DEV bool slab(float o, float d, float inv, float lo, float hi, float& t0, float& t1) {
  if (absf(d) < 1e-12f) return !(o < lo || o > hi);
  float ta = (lo - o) * inv, tb = (hi - o) * inv;
  if (ta > tb) { const float x = ta; ta = tb; tb = x; }
  t0 = maxf(t0, ta);
  t1 = minf(t1, tb);
  return true;
}
DEV float slab_end(float t0, float t1) { return (t0 > t1 || t0 <= 0.f) ? kInf : t0; }
DEV void ray_inv(const float d[3], float inv[3]) {
  for (int a = 0; a < 3; ++a) inv[a] = absf(d[a]) < 1e-12f ? 0.f : 1.f / d[a];
}
DEV float ray_box_inv(const float o[3], const float d[3], const float inv[3], const Prim& b) {   // 들어가는 t (> 0), 없으면 kInf
  float t0 = -kInf, t1 = kInf;
  for (int a = 0; a < 3; ++a)
    if (!slab(o[a], d[a], inv[a], b.lo[a], b.hi[a], t0, t1)) return kInf;
  return slab_end(t0, t1);
}
DEV float ray_room(const float o[3], const float d[3], float rhx, float rhy) {   // 방 안에서 벽·바닥·천장까지
  float t = kInf;
  if (d[0] > 0.f) t = minf(t, (rhx - o[0]) / d[0]); else if (d[0] < 0.f) t = minf(t, (-rhx - o[0]) / d[0]);
  if (d[1] > 0.f) t = minf(t, (rhy - o[1]) / d[1]); else if (d[1] < 0.f) t = minf(t, (-rhy - o[1]) / d[1]);
  if (d[2] < 0.f) t = minf(t, -o[2] / d[2]); else if (d[2] > 0.f) t = minf(t, (MP::wall_h - o[2]) / d[2]);
  return t;
}
DEV float cast(const MapCore& m, const float o[3], const float d[3]) {
  float inv[3];
  ray_inv(d, inv);
  float t = ray_room(o, d, m.rhx, m.rhy);
  for (int p = 0; p < N_PRIM; ++p) t = minf(t, ray_box_inv(o, d, inv, m.prim[p]));
  return t;
}
DEV void cam_world(float x, float y, float c, float s, float o[3]) {
  o[0] = x + c * env::K::cam_x;
  o[1] = y + s * env::K::cam_x;
  o[2] = env::K::cam_z;
}
// 화소 방향(정규화 좌표 xn 오른쪽, yn 아래) → 세계 방향. 광학 z 로 매개(t = 깊이)
DEV void pix_dir(float c, float s, float xn, float yn, float d[3]) {
  d[0] = c + s * xn;    // 베이스 (1, -xn, -yn) 을 yaw 로
  d[1] = s - c * xn;
  d[2] = -yn;
}

// ---- 장면 만들기(판 리셋 때) -------------------------------------------------------------------------------------------
DEV bool box_overlap(const Prim& a, const Prim& b, float m) {
  return a.lo[0] - m < b.hi[0] && b.lo[0] - m < a.hi[0] && a.lo[1] - m < b.hi[1] && b.lo[1] - m < a.hi[1];
}
DEV bool box_has(const Prim& a, float x, float y, float m) {
  return x > a.lo[0] - m && x < a.hi[0] + m && y > a.lo[1] - m && y < a.hi[1] + m;
}
DEV void make_scene(MapCore& m, const EnvView& e) {
  Prim& cup = m.prim[0];
  cup.cls = C_CUP;
  cup.lo[0] = e.tx - env::K::tgt_r; cup.hi[0] = e.tx + env::K::tgt_r;
  cup.lo[1] = e.ty - env::K::tgt_r; cup.hi[1] = e.ty + env::K::tgt_r;
  cup.lo[2] = 0.f; cup.hi[2] = MP::cup_h;
  // 가구 5: 벽에 붙임. 크기 = 기본 × [0.8, 1.2] (가정)
  for (int k = 1; k <= 5; ++k) {
    Prim b{};
    for (int tries = 0; tries < 8; ++tries) {
      const int ci = (int)(rand01(m.rng) * 4.f);                 // 0..3 → 의자·탁자·장·쓰레기통
      const int cls = ci == 0 ? C_CHAIR : ci == 1 ? C_TABLE : ci == 2 ? C_CABINET : C_BIN;
      float w = cls == C_CHAIR ? 0.45f : cls == C_TABLE ? 0.8f : cls == C_CABINET ? 0.5f : 0.3f;   // 벽 따라
      float dp = cls == C_CHAIR ? 0.45f : cls == C_TABLE ? 0.5f : cls == C_CABINET ? 0.4f : 0.3f;  // 벽에서
      float h = cls == C_CHAIR ? 0.9f : cls == C_TABLE ? 0.75f : cls == C_CABINET ? 1.0f : 0.4f;
      w = w * rand_range(m.rng, 0.8f, 1.2f);
      dp = dp * rand_range(m.rng, 0.8f, 1.2f);
      h = h * rand_range(m.rng, 0.8f, 1.2f);
      const int side = (int)(rand01(m.rng) * 4.f);
      const bool xw = side < 2;                                  // x = ±rhx 벽
      const float along_r = (xw ? e.rhy : e.rhx) - 0.5f * w - 0.05f;
      const float u = rand_range(m.rng, -along_r, along_r);
      const float sg = (side & 1) ? -1.f : 1.f;
      const float perp = sg * ((xw ? e.rhx : e.rhy) - 0.5f * dp - 0.02f);
      const float cx = xw ? perp : u, cy = xw ? u : perp;
      const float hx = 0.5f * (xw ? dp : w), hy = 0.5f * (xw ? w : dp);
      b.cls = cls;
      b.lo[0] = cx - hx; b.hi[0] = cx + hx; b.lo[1] = cy - hy; b.hi[1] = cy + hy; b.lo[2] = 0.f; b.hi[2] = h;
      bool ok = !box_has(b, e.x, e.y, 0.35f) && !box_has(b, e.tx, e.ty, 0.15f);
      for (int j = 1; j < k && ok; ++j) ok = !box_overlap(b, m.prim[j], 0.05f);
      if (ok) break;
    }
    m.prim[k] = b;
  }
  // 작은 물건 3: 방 안 아무 데나(로봇·컵에서 떨어지게)
  for (int k = 6; k < N_PRIM; ++k) {
    Prim b{};
    for (int tries = 0; tries < 8; ++tries) {
      const float hx = 0.5f * rand_range(m.rng, 0.08f, 0.25f), hy = 0.5f * rand_range(m.rng, 0.08f, 0.25f);
      const float h = rand_range(m.rng, 0.12f, 0.35f);
      const float cx = rand_range(m.rng, -e.rhx + 0.5f, e.rhx - 0.5f), cy = rand_range(m.rng, -e.rhy + 0.5f, e.rhy - 0.5f);
      b.cls = C_ITEM;
      b.lo[0] = cx - hx; b.hi[0] = cx + hx; b.lo[1] = cy - hy; b.hi[1] = cy + hy; b.lo[2] = 0.f; b.hi[2] = h;
      bool ok = !box_has(b, e.x, e.y, 0.4f) && !box_has(b, e.tx, e.ty, 0.3f);
      for (int j = 1; j < k && ok; ++j) ok = !box_overlap(b, m.prim[j], 0.05f);
      if (ok) break;
    }
    m.prim[k] = b;
  }
}

DEV void init_core(MapCore& m, uint64_t seed, int i) {
  uint32_t* w = reinterpret_cast<uint32_t*>(&m);
  for (int k = 0; k < CORE_WORDS; ++k) w[k] = 0u;
  m.rng = seed * 0xD1B54A32D192ED03ull + (uint64_t)i * 0x9E3779B97F4A7C15ull + 0xABCDEFull;
  m.ep = -1;   // 첫 스텝에서 리셋
}

DEV void reset_core(MapCore& m, const EnvView& e) {
  m.ep = e.ep;
  m.t = 0; m.first = 1; m.since = 0; m.n_kf = 0; m.next_id = 1;
  m.n_seen_room = 0; m.n_task_conf = 0; m.n_obj_conf = 0; m.n_dropped = 0;
  m.ex = e.x; m.ey = e.y; m.eyaw = e.yaw;   // 판 시작 자세는 안다(scenemap: 원점에서 시작)
  m.px = e.x; m.py = e.y; m.pyaw = e.yaw;
  m.lx = e.x; m.ly = e.y; m.lyaw = e.yaw;
  m.vmax = 0.f; m.wmax = 0.f;
  m.rhx = e.rhx; m.rhy = e.rhy;
  for (int k = 0; k < KSLOT; ++k) {
    uint32_t* w = reinterpret_cast<uint32_t*>(&m.slot[k]);
    for (int j = 0; j < (int)(sizeof(Slot) / 4); ++j) w[j] = 0u;
  }
  make_scene(m, e);
  int nx = 0, ny = 0;
  for (int l = 0; l < GW; ++l) {
    const float cc = ((float)(l + GX0) + 0.5f) * RES;
    nx += absf(cc) < e.rhx;
    ny += absf(cc) < e.rhy;
  }
  m.room_cells = nx * ny;
}

// ---- 1. 시작(판마다 한 스레드): 리셋, 오도메트리 표류, 움직임 거르기 -------------------------------------------------------------
// 돌려주는 값: B_KF(이번 스텝 keyframe) | B_RESET(판 리셋). MapCore 의 머리(prim 앞)만 바꾸고, 리셋이면 MapCore 전체를 새로 쓴다.
enum BeginFlag { B_KF = 1, B_RESET = 2 };
DEV int phase_begin(MapCore& m, const EnvView& e, int force_kf) {
  int reset = 0;
  if (e.ep != m.ep) {
    reset_core(m, e);
    reset = B_RESET;
  } else {
    // 참 증분(지난 참 자세 기준 몸 좌표) → 잡음 → 믿는 자세에 붙임 (slam2d pushVelocity 적분의 오차 흉내)
    float s0, c0;
    sincosf_d(m.pyaw, &s0, &c0);
    const float dxw = e.x - m.px, dyw = e.y - m.py;
    const float dxb = c0 * dxw + s0 * dyw, dyb = -s0 * dxw + c0 * dyw;
    const float dth = wrap_pi(e.yaw - m.pyaw);
    const float dist = sqrtf(dxb * dxb + dyb * dyb);
    const float n1 = gauss(m.rng), n2 = gauss(m.rng), n3 = gauss(m.rng);
    const float sx = MP::odo_t * dist;
    const float nxb = dxb + n1 * sx, nyb = dyb + n2 * sx;
    const float nth = dth + n3 * (MP::odo_rr * absf(dth) + MP::odo_rt * dist);
    float se, ce;
    sincosf_d(m.eyaw, &se, &ce);
    m.ex = m.ex + (ce * nxb - se * nyb);
    m.ey = m.ey + (se * nxb + ce * nyb);
    m.eyaw = wrap_pi(m.eyaw + nth);
    m.t += 1;
  }
  m.px = e.x; m.py = e.y; m.pyaw = e.yaw;
  m.vmax = maxf(m.vmax, absf(e.v));
  m.wmax = maxf(m.wmax, absf(e.w));
  // slam2d insertStage update_policy 0: 처음, 또는 mf_xy·mf_yaw 넘게 움직임, 또는 mf_kf 번째
  m.since += 1;
  const float dx = m.ex - m.lx, dy = m.ey - m.ly;
  const bool moved = dx * dx + dy * dy >= MP::mf_xy * MP::mf_xy || absf(wrap_pi(m.eyaw - m.lyaw)) >= MP::mf_yaw;
  const int kf = (m.first || force_kf || moved || m.since >= MP::mf_kf) ? 1 : 0;
  m.kf_flag = kf;
  return (kf ? B_KF : 0) | reset;
}

DEV void clear_grid(int16_t* L, uint32_t* seen, int tid, int nt) {
  uint32_t* Lw = reinterpret_cast<uint32_t*>(L);
  for (int k = tid; k < NCELL / 2; k += nt) Lw[k] = 0u;
  for (int k = tid; k < NWORD; k += nt) seen[k] = 0u;
}

// ---- 2. 광선(모든 스레드): 깊이 줄 + 물체 점 보임. 참 장면·참 카메라에서 쏜다 ----------------------------------------------
DEV int vis_point(const MapCore& m, const float o[3], float c, float s, const Cam& k, int p) {
  const int pi = p / NPT, q = p % NPT;
  const Prim& b = m.prim[pi];
  float pt[3];
  for (int a = 0; a < 3; ++a) pt[a] = 0.5f * (b.lo[a] + b.hi[a]);
  if (q > 0) {
    const int bits = q - 1;
    for (int a = 0; a < 3; ++a) {
      const float h = 0.5f * (b.hi[a] - b.lo[a]) * 0.8f;
      pt[a] = pt[a] + ((bits >> a) & 1 ? h : -h);
    }
  }
  const float r[3] = {pt[0] - o[0], pt[1] - o[1], pt[2] - o[2]};
  const float fwd = c * r[0] + s * r[1], left = -s * r[0] + c * r[1];
  if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) return 0;
  const float xn = -left / fwd, yn = -r[2] / fwd;
  if (absf(xn) > k.tanh || absf(yn) > k.tanv) return 0;
  float inv[3];
  ray_inv(r, inv);
  for (int j = 0; j < N_PRIM; ++j)
    if (j != pi && ray_box_inv(o, r, inv, m.prim[j]) < 1.f) return 0;   // 다른 상자가 사이를 가림
  return 1;
}

DEV float max3(const float v[3]) { return maxf(v[0], maxf(v[1], v[2])); }

// 3b 검출 판정 중 보임 점과 무관한 부분(참 장면·참 카메라). ozmin/ozmax·min_px 를 넘고, 9 점이 다 보여도 넓이가 min_points 를 못 넘으면
// 검출될 수 없으므로 false — 그 물체의 보임 광선을 쏘지 않는다(넓이는 보이는 점 수에 단조라 결과는 그대로). 넓이 = af · nv / NPT
DEV bool det_prefilter(const MapCore& m, const float o[3], float c, float s, const Cam& k, int p, DetGeo& g) {
  const Prim& b = m.prim[p];
  for (int a = 0; a < 3; ++a) { g.ctr[a] = 0.5f * (b.lo[a] + b.hi[a]); g.ext[a] = b.hi[a] - b.lo[a]; }
  g.rx = g.ctr[0] - o[0]; g.ry = g.ctr[1] - o[1];
  g.fwd = c * g.rx + s * g.ry; g.left = -s * g.rx + c * g.ry; g.up = g.ctr[2] - o[2];
  if (!(g.fwd >= MP::ozmin && g.fwd <= MP::ozmax)) return false;
  const float size_px = k.fx * max3(g.ext) / g.fwd;                     // objmap 부재 확인과 같은 식
  g.rh = sqrtf(g.rx * g.rx + g.ry * g.ry);
  const float ux = g.rx / g.rh, uy = g.ry / g.rh;
  const float wsil = g.ext[0] * absf(uy) + g.ext[1] * absf(ux);          // 수평 실루엣 폭
  g.af = (k.fx * wsil / g.fwd) * (k.fx * g.ext[2] / g.fwd);
  return !(size_px < (float)MP::min_px || g.af < (float)MP::min_points);   // af · NPT/NPT = af
}

DEV void phase_cast(const MapCore& m, Scratch& sh, const EnvView& e, int tid, int nt) {
  for (int w = tid; w < NWORD; w += nt) { sh.hitb[w] = 0u; sh.missb[w] = 0u; }
  for (int p = tid; p < N_PRIM; p += nt) sh.vism[p] = 0u;   // 보임 점 비트(obj_pre 가 동기 뒤에 OR)
  const Cam k = cam_consts();
  float s, c;
  sincosf_d(e.yaw, &s, &c);
  float o[3];
  cam_world(e.x, e.y, c, s, o);
  if (tid == 0) { sh.tc = c; sh.ts = s; sh.to[0] = o[0]; sh.to[1] = o[1]; sh.to[2] = o[2]; }   // 뒤 단계가 같은 값을 다시 씀
  // 줄마다 같은 값(열과 무관): 세로 방향 d2 = -yn, 1/d2, 바닥·천장까지 t
  float invz[NROW], troomz[NROW];
  for (int r = 0; r < NROW; ++r) {
    const float yn = ((float)(2 * r + 1 - NROW) / (float)NROW) * k.tanv;
    const float d2 = -yn;
    invz[r] = absf(d2) < 1e-12f ? 0.f : 1.f / d2;
    troomz[r] = d2 < 0.f ? -o[2] / d2 : d2 > 0.f ? (MP::wall_h - o[2]) / d2 : kInf;
  }
  // scan.cpp makeScan: 띠 [band_lo, band_hi] 안 가장 가까운 점 = 장애물, 띠 아래(바닥) 가장 먼 점 = 빈 광선 끝. 기울기 0 이라 열 = 방위 칸
  // 한 열의 줄 8 개는 수평 방향이 같다: 방 벽·상자의 xy 판은 열마다 한 번, z 판만 줄마다(ray_room·ray_box 와 비트 같음)
  for (int col = tid; col < NCOL; col += nt) {
    const float xn = ((float)(2 * col + 1 - NCOL) / (float)NCOL) * k.tanh;
    const float d0 = c + s * xn, d1 = s - c * xn;   // pix_dir
    float txy = kInf;
    if (d0 > 0.f) txy = minf(txy, (m.rhx - o[0]) / d0); else if (d0 < 0.f) txy = minf(txy, (-m.rhx - o[0]) / d0);
    if (d1 > 0.f) txy = minf(txy, (m.rhy - o[1]) / d1); else if (d1 < 0.f) txy = minf(txy, (-m.rhy - o[1]) / d1);
    float tr[NROW];
    for (int r = 0; r < NROW; ++r) tr[r] = minf(txy, troomz[r]);
    float trmax = tr[0];
    for (int r = 1; r < NROW; ++r) trmax = maxf(trmax, tr[r]);
    const float inv0 = absf(d0) < 1e-12f ? 0.f : 1.f / d0, inv1 = absf(d1) < 1e-12f ? 0.f : 1.f / d1;
    for (int p = 0; p < N_PRIM; ++p) {
      const Prim& b = m.prim[p];
      float t0 = -kInf, t1 = kInf;
      if (!slab(o[0], d0, inv0, b.lo[0], b.hi[0], t0, t1) || !slab(o[1], d1, inv1, b.lo[1], b.hi[1], t0, t1)) continue;
      if (t0 > t1) continue;   // z 판은 구간을 좁히기만 하므로 이미 비면 어느 줄도 안 맞음
      if (t0 >= trmax) continue;  // 들어가는 t ≥ 모든 줄의 지금 값: 어느 줄도 줄일 수 없음(minf 결과 같음)
      for (int r = 0; r < NROW; ++r) {
        const float yn = ((float)(2 * r + 1 - NROW) / (float)NROW) * k.tanv;
        float u0 = t0, u1 = t1;
        if (!slab(o[2], -yn, invz[r], b.lo[2], b.hi[2], u0, u1)) continue;
        tr[r] = minf(tr[r], slab_end(u0, u1));
      }
      trmax = tr[0];
      for (int r = 1; r < NROW; ++r) trmax = maxf(trmax, tr[r]);
    }
    float best = kInf, hit_t = 0.f, floor_r = 0.f, floor_t = 0.f;
    for (int r = 0; r < NROW; ++r) {
      const float yn = ((float)(2 * r + 1 - NROW) / (float)NROW) * k.tanv;
      float d[3];
      pix_dir(c, s, xn, yn, d);
      const float t = tr[r];
      if (!(t >= MP::zmin && t <= MP::zmax)) continue;   // 깊이 없음
      const float pz = o[2] + d[2] * t;
      if (pz > MP::band_hi) continue;
      const float rr2 = t * t * (1.f + xn * xn);
      if (pz < MP::band_lo) {
        if (rr2 > floor_r) { floor_r = rr2; floor_t = t; }
      } else if (rr2 < best) {
        best = rr2; hit_t = t;
      }
    }
    if (best < kInf) { sh.colt[col] = 1; sh.colx[col] = env::K::cam_x + hit_t; sh.coly[col] = -xn * hit_t; }
    else if (floor_r > 0.f) { sh.colt[col] = 2; sh.colx[col] = env::K::cam_x + floor_t; sh.coly[col] = -xn * floor_t; }
    else { sh.colt[col] = 0; sh.colx[col] = 0.f; sh.coly[col] = 0.f; }
  }
  for (int p = tid; p < N_PRIM; p += nt) sh.pcand[p] = det_prefilter(m, o, c, s, k, p, sh.geo[p]) ? 1 : 0;
}

// ---- 3. 자세 보정 + 검출 + 물체 기억 ---------------------------------------------------------------------------------------
// 순서가 있는 일(난수 뽑기, 새 칸 고르기)만 스레드 0 이 하고, 판정·짝 열쇠·최솟값 찾기·부재 확인·완성도는 스레드가 나눠 한다.
// 나눈 일은 서로 다른 칸만 쓰고, 최솟값은 (열쇠, 번호) 순으로 골라 스레드 수와 무관하게 같은 답이다(CPU nt = 1 과 비트 같음).
// 3-0(모든 스레드): 맞은 열 수 부분합, 후보 물체(det_prefilter 통과)의 보임 점만 쏜다
DEV void obj_pre(const MapCore& m, Scratch& sh, int tid, int nt) {
  int nh = 0;
  for (int col = tid; col < NCOL; col += nt) nh += sh.colt[col] == 1;
  put_part(sh, tid, nh);
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  int ncand = 0;
  for (int p = 0; p < N_PRIM; ++p) ncand += sh.pcand[p];
  for (int i = tid; i < ncand * NPT; i += nt) {
    int p = 0;
    for (int j = i / NPT; ; ++p) if (sh.pcand[p] && j-- == 0) break;   // (i / NPT) 번째 후보
    if (vis_point(m, o, c, s, k, p * NPT + i % NPT)) or_bits(&sh.vism[p], 1u << (i % NPT));   // vism 은 phase_begin 이 비움
  }
}

// 3a·3b(스레드 0): 자세 보정, 검출(난수 순서 그대로), 가짜 물체
DEV void obj_detect(MapCore& m, Scratch& sh, const EnvView& e, int nt) {
  // 3a. keyframe 맞추기: 맞은 줄이 충분하고 제자리가 아니면 오차를 kf_corr 만큼 되돌림(slam2d keyframe 의 보정 흉내)
  const int n_hits = sum_part(sh, nt);
  const bool still = !m.first && m.vmax < MP::still_v && m.wmax < MP::still_w;
  if (!m.first && !still && n_hits >= MP::min_hits) {
    const float keep = 1.f - MP::kf_corr;
    m.ex = e.x + (m.ex - e.x) * keep;
    m.ey = e.y + (m.ey - e.y) * keep;
    m.eyaw = wrap_pi(e.yaw + wrap_pi(m.eyaw - e.yaw) * keep);
  }
  float es, ec;
  sincosf_d(m.eyaw, &es, &ec);
  sh.ec = ec; sh.es = es;
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey;
  uint64_t rng = m.rng;   // 레지스터에(공유 메모리 sh.det 쓰기와 겹칠까 봐 매번 다시 읽지 않게)

  // 3b. 검출: 판정(det_prefilter + 보이는 점 비율) → 놓침 → 잡음(본 순간의 slam 오차를 물려받음) → 틀린 이름
  int nd = 0;
  for (int p = 0; p < N_PRIM; ++p) {
    if (!sh.pcand[p]) continue;
    const DetGeo& g = sh.geo[p];
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += (int)((sh.vism[p] >> q) & 1u);
    if (nv == 0) continue;
    if (g.af * ((float)nv / (float)NPT) < (float)MP::min_points) continue;   // 깊이 점 수(간격 1)
    if (rand01(rng) < MP::p_miss) continue;
    const float fwd = g.fwd, left = g.left, rh = g.rh;
    const float sig_d = MP::dn0 + MP::dn2 * fwd * fwd;
    const float gd = gauss(rng) * sig_d, gl = gauss(rng) * MP::lat_n, gz = gauss(rng) * MP::lat_n;
    // 몸 좌표의 수평 시선 단위(fwd, left)/rh 와 그 수직
    const float bu = fwd / rh, bv = left / rh;
    const float f2 = fwd + bu * gd - bv * gl, l2 = left + bv * gd + bu * gl;
    const float bx = env::K::cam_x + f2, by = l2;
    Det& D = sh.det[nd++];
    D.pos[0] = ex + (ec * bx - es * by);
    D.pos[1] = ey + (es * bx + ec * by);
    D.pos[2] = o2 + g.up + gz;
    for (int a = 0; a < 3; ++a) D.ext[a] = maxf(0.01f, g.ext[a] + gauss(rng) * MP::ext_n);
    int cls = m.prim[p].cls;
    if (rand01(rng) < MP::p_conf) cls = (cls + 1 + (int)(rand01(rng) * (float)(NCLS - 1))) % NCLS;
    D.cls = cls;
    D.score = 0.9f;
  }
  // 가짜 물체: 맞은 줄 하나 위에 작은 상자(가정)
  if (rand01(rng) < MP::p_fp) {
    const int col = (int)(rand01(rng) * (float)NCOL);
    const float zc = rand_range(rng, 0.05f, 0.5f), sz = rand_range(rng, 0.05f, 0.2f);
    const int cls = (int)(rand01(rng) * (float)NCLS);
    if (sh.colt[col] != 0 && nd < MAXDET) {
      const float bx = sh.colx[col], by = sh.coly[col];
      Det& D = sh.det[nd++];
      D.pos[0] = ex + (ec * bx - es * by);
      D.pos[1] = ey + (es * bx + ec * by);
      D.pos[2] = zc;
      D.ext[0] = sz; D.ext[1] = sz; D.ext[2] = sz;
      D.cls = cls;
      D.score = 0.4f;
      m.n_fp_total += 1;
    }
  }
  sh.nd = nd;
  m.rng = rng;
}

// 3c. 같은 물체(objmap.cpp 2): 같은 이름끼리, 중심 거리 < max(da_min, da_k·큰 쪽 크기) 또는 상자 틈 < da_gap. (틈 + 1e-3·거리) 순 1:1 탐욕
// 짝 열쇠(모든 스레드, 쌍마다)
DEV void obj_keys(const MapCore& m, Scratch& sh, int tid, int nt) {
  const int nd = sh.nd;
  for (int b = tid; b < KSLOT; b += nt) sh.hit[b] = 0;
  for (int a = tid; a < MAXDET; a += nt) sh.obs_to[a] = -1;
  for (int idx = tid; idx < nd * KSLOT; idx += nt) {
    const int a = idx / KSLOT, b = idx % KSLOT;
    float key = -1.f;
    const Slot& S = m.slot[b];
    const Det& D = sh.det[a];
    if (S.valid && S.cls == D.cls) {
      const float ee = maxf(max3(D.ext), max3(S.ext));
      const float thr = maxf(MP::da_min, MP::da_k * ee);
      float d2 = 0.f, g2 = 0.f;
      for (int q = 0; q < 3; ++q) {
        const float dd = D.pos[q] - S.pos[q];
        d2 = d2 + dd * dd;
        const float olo = D.pos[q] - 0.5f * D.ext[q], ohi = D.pos[q] + 0.5f * D.ext[q];
        const float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float gk = maxf(0.f, maxf(olo - mhi, mlo - ohi));
        g2 = g2 + gk * gk;
      }
      const float d = sqrtf(d2), gap = sqrtf(g2);
      if (d < thr || gap < MP::da_gap) key = gap + 1e-3f * d;
    }
    sh.key[idx] = key;
  }
}
// 탐욕 한 바퀴의 앞(모든 스레드): 남은 쌍 중 스레드 몫의 최소(같으면 앞 번호 = 원래 (관측, 칸) 이중 고리의 첫 최소)
DEV void obj_argmin_local(Scratch& sh, int tid, int nt) {
  const int n = sh.nd * KSLOT;
  int bi = -1;
  float bk = 0.f;
  for (int idx = tid; idx < n; idx += nt) {
    const float key = sh.key[idx];
    if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
    if (bi < 0 || key < bk) { bi = idx; bk = key; }
  }
  sh.part[tid] = bi;
  sh.partf[tid] = bk;
}
// 탐욕 한 바퀴의 뒤(스레드 0): (열쇠, 번호) 최소를 짝으로. 더 없으면 sh.more = 0
DEV void obj_argmin_pick(Scratch& sh, int nt) {
  int bi = -1;
  float bk = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int i = sh.part[t];
    if (i < 0) continue;
    const float k = sh.partf[t];
    if (bi < 0 || k < bk || (k == bk && i < bi)) { bi = i; bk = k; }
  }
  sh.more = bi >= 0;
  if (bi >= 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
}

#ifdef __CUDA_ARCH__
// 탐욕 짝짓기 GPU 판(워프 0 만): 레인마다 몫의 최소 → 셔플로 (열쇠, 번호) 최소 → 레인 0 이 짝 표시. 남은 쌍이 없으면 끝
__device__ __forceinline__ void obj_greedy_warp(Scratch& sh, int lane) {
  const int n = sh.nd * KSLOT;
  for (;;) {
    int bi = -1;
    float bk = 0.f;
    for (int idx = lane; idx < n; idx += 32) {
      const float key = sh.key[idx];
      if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
      if (bi < 0 || key < bk) { bi = idx; bk = key; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      const float ok = __shfl_xor_sync(0xffffffffu, bk, off);
      if (oi >= 0 && (bi < 0 || ok < bk || (ok == bk && oi < bi))) { bi = oi; bk = ok; }
    }
    if (bi < 0) break;   // 워프 안 모두 같은 값
    if (lane == 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
    __syncwarp();
  }
}
#endif

// 3d. 갱신(objmap.cpp 3). 짝지은 관측(모든 스레드, 관측마다 다른 칸)
DEV void obj_update_matched(MapCore& m, Scratch& sh, int tid, int nt, int bug) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;   // 음성 대조: 확정 규칙 끔(한 번 보이면 확정)
  const int t = m.t;
  for (int a = tid; a < sh.nd; a += nt) {
    if (sh.obs_to[a] < 0) continue;
    const Det& D = sh.det[a];
    Slot& S = m.slot[sh.obs_to[a]];
    if (S.last_kf != t) S.n_obs += 1;
    S.last_kf = t;
    const float w = (float)(S.n_obs < 20 ? S.n_obs : 20);
    const bool bigo = is_static(S.cls) || maxf(maxf(S.ext[0], S.ext[1]), maxf(D.ext[0], D.ext[1])) > MP::big;
    for (int q = 0; q < 3; ++q) {
      if (bigo) {   // 합집합, keyframe 마다 면마다 grow_max·한 변 max_ext 까지
        float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float olo = D.pos[q] - 0.5f * D.ext[q], ohi = D.pos[q] + 0.5f * D.ext[q];
        const float lo = maxf(minf(mlo, olo), mlo - MP::grow_max);
        const float hi = minf(maxf(mhi, ohi), mhi + MP::grow_max);
        if (hi - lo <= MP::max_ext) { mlo = lo; mhi = hi; }
        S.pos[q] = 0.5f * (mlo + mhi);
        S.ext[q] = mhi - mlo;
      } else {
        S.pos[q] = (S.pos[q] * (w - 1.f) + D.pos[q]) / w;
        S.ext[q] = (S.ext[q] * (w - 1.f) + D.ext[q]) / w;
      }
    }
    S.score = maxf(S.score, D.score);
    S.last_seen = t;
    S.misses = 0;
    if (S.state == S_GONE) S.state = S.moved ? S_MOVED : S_SEEN;
    if (!S.confirmed && S.n_obs >= confirm_n) S.confirmed = 1;
  }
}
// 안 맞은 관측(스레드 0, 관측 순서대로). 짝지은 칸은 hit 이라 여기서 보지 않으므로 위와 나눠 해도 원래 순서와 같다
DEV void obj_update_new(MapCore& m, Scratch& sh, int bug) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;
  const int t = m.t;
  for (int a = 0; a < sh.nd; ++a) {
    if (sh.obs_to[a] >= 0) continue;
    const Det& D = sh.det[a];
    // 같은 이름의 확정·사라짐 물체가 있으면 옮겨진 것으로 잇는다
    int mf = -1;
    float best = 1e9f;
    for (int b = 0; b < KSLOT; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || S.cls != D.cls || !S.confirmed || sh.hit[b] || S.state != S_GONE) continue;
      float d2 = 0.f;
      for (int q = 0; q < 3; ++q) { const float dd = D.pos[q] - S.pos[q]; d2 = d2 + dd * dd; }
      if (d2 < best) { best = d2; mf = b; }
    }
    if (mf >= 0) {
      Slot& S = m.slot[mf];
      for (int q = 0; q < 3; ++q) { S.pos[q] = D.pos[q]; S.ext[q] = D.ext[q]; }
      S.moved = best > MP::moved_d * MP::moved_d ? 1 : S.moved;   // 짝 문턱(≥ 0.30 m) 밖이라 사실상 항상 옮겨짐
      S.state = S.moved ? S_MOVED : S_SEEN;
      S.misses = 0;
      S.last_seen = t;
      S.last_kf = t;
      S.n_obs += 1;
      sh.hit[mf] = 1;
      continue;
    }
    int fs = -1;
    for (int b = 0; b < KSLOT && fs < 0; ++b) if (!m.slot[b].valid) fs = b;
    if (fs < 0) { m.n_dropped += 1; continue; }   // 칸이 다 참(가정: 버림)
    Slot& S = m.slot[fs];
    S.valid = 1; S.id = m.next_id++; S.cls = D.cls;
    for (int q = 0; q < 3; ++q) { S.pos[q] = D.pos[q]; S.first_pos[q] = D.pos[q]; S.ext[q] = D.ext[q]; }
    S.n_obs = 1; S.last_seen = t; S.last_kf = t; S.score = D.score;
    S.confirmed = confirm_n <= 1 ? 1 : 0;
    S.state = S_SEEN; S.moved = 0; S.misses = 0; S.first_miss = 0;
    sh.hit[fs] = 1;
  }
}

// 3e. 부재 확인(objmap.cpp 4) + 3f. 오래된 후보 버리기. 칸마다 따로(모든 스레드, 칸마다 한 스레드)
//     확정·이번에 안 맞음·사라짐 아님·고정 종류 아님·작은 것. 믿는 자세로 화소에 투영하고
//     그 화소의 깊이(참 장면)가 물체보다 occl 넘게 가까우면 가려짐. 아니면 놓침 +1
DEV void obj_absence(MapCore& m, const Scratch& sh, const EnvView& e, int tid, int nt) {
  const Cam k = cam_consts();
  const float ec = sh.ec, es = sh.es;
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  const int t = m.t;
  for (int b = tid; b < KSLOT; b += nt) {
    Slot& S = m.slot[b];
    if (S.valid && S.confirmed && !sh.hit[b] && S.state != S_GONE && !is_static(S.cls) && !(max3(S.ext) > MP::big)) {
      const float rx = S.pos[0] - (m.ex + ec * env::K::cam_x), ry = S.pos[1] - (m.ey + es * env::K::cam_x);
      const float zc = ec * rx + es * ry, left = -es * rx + ec * ry, up = S.pos[2] - env::K::cam_z;
      if (!(zc < 0.3f || zc > MP::ozmax)) {
        const float xn = -left / zc, yn = -up / zc;
        const float u = k.fx * xn + k.cxp, v = k.fx * yn + k.cyp;
        const float size_px = k.fx * max3(S.ext) / zc;
        if (!(size_px < (float)MP::min_px || u < 2.f || v < 2.f || u >= (float)(MP::img_w - 2) || v >= (float)(MP::img_h - 2))) {
          float d[3];
          pix_dir(c, s, xn, yn, d);
          const float tz = cast(m, o, d);
          if (tz >= MP::zmin && tz <= MP::zmax && !(tz < zc - MP::occl)) {   // 깊이 있음, 가려지지 않음
            if (S.misses++ == 0) S.first_miss = t;
            if (S.misses >= MP::gone_misses && t - S.first_miss >= MP::gone_min_steps) S.state = S_GONE;
          }
        }
      }
    }
    if (S.valid && !S.confirmed && t - S.last_seen > MP::prune_steps) S.valid = 0;
  }
}

// 3g. 완성도(모든 스레드, 참 물체마다): 같은 이름의 확정(사라짐 아님) 칸이 짝 문턱 안에 있나 → sh.found[p]
DEV void obj_complete(const MapCore& m, Scratch& sh, int tid, int nt) {
  for (int p = tid; p < N_PRIM; p += nt) {
    const Prim& P = m.prim[p];
    const float cx = 0.5f * (P.lo[0] + P.hi[0]), cy = 0.5f * (P.lo[1] + P.hi[1]);
    float ext[3];
    for (int a = 0; a < 3; ++a) ext[a] = P.hi[a] - P.lo[a];
    const float thr = maxf(MP::da_min, MP::da_k * max3(ext));
    int found = 0;
    for (int b = 0; b < KSLOT && !found; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || !S.confirmed || S.state == S_GONE || S.cls != P.cls) continue;
      const float dx = S.pos[0] - cx, dy = S.pos[1] - cy;
      found = dx * dx + dy * dy < thr * thr;
    }
    sh.found[p] = found;
  }
}

// ---- 4. 격자 표시(모든 스레드): 믿는 자세로 줄을 격자에. 정수 Bresenham, 끝 칸 제외 → 맞음 끝은 hit, 빈 광선 끝은 miss ----------
DEV bool in_win(int ix, int iy) { return ix - GX0 >= 0 && iy - GX0 >= 0 && ix - GX0 < GW && iy - GX0 < GW; }
// 창 안 칸 표시. GPU 는 낱말이 바뀔 때 그 낱말의 로그 오즈 64 B·본 칸 낱말을 L2 로 미리 당긴다(격자 갱신의 DRAM 대기를 줄임, 값은 안 바뀜)
DEV void mark_in(uint32_t* bits, int ix, int iy, const int16_t* L, const uint32_t* seen, int& lastw) {
  const int idx = (iy - GX0) * GW + (ix - GX0), w = idx >> 5;
  or_bits(&bits[w], 1u << (idx & 31));
#ifdef __CUDA_ARCH__
  if (w != lastw) {
    lastw = w;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(L + (size_t)w * 32));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(seen + w));
  }
#else
  (void)L; (void)seen; (void)lastw;
#endif
}
DEV void mark(uint32_t* bits, int ix, int iy) {
  const int lx = ix - GX0, ly = iy - GX0;
  if (lx < 0 || ly < 0 || lx >= GW || ly >= GW) return;
  const int idx = ly * GW + lx;
  or_bits(&bits[idx >> 5], 1u << (idx & 31));
}
DEV void phase_mark(const MapCore& m, Scratch& sh, const int16_t* L, const uint32_t* seen, int tid, int nt) {
  const float ec = sh.ec, es = sh.es;
  const float ox = m.ex + ec * env::K::cam_x, oy = m.ey + es * env::K::cam_x;
  const int cx = (int)floorf(ox * INV_RES), cy = (int)floorf(oy * INV_RES);
  for (int col = tid; col < NCOL; col += nt) {
    const int ty = sh.colt[col];
    if (ty == 0) continue;
    const float bx = sh.colx[col], by = sh.coly[col];
    const float wx = m.ex + (ec * bx - es * by), wy = m.ey + (es * bx + ec * by);
    const int x1 = (int)floorf(wx * INV_RES), y1 = (int)floorf(wy * INV_RES);
    int x = cx, y = cy;
    const int dx = x1 > x ? x1 - x : x - x1, dy = -(y1 > y ? y1 - y : y - y1), sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1;
    int err = dx + dy;
    if (in_win(cx, cy) && in_win(x1, y1)) {   // 두 끝이 창 안이면 지나는 칸(두 끝의 상자 안)도 모두 창 안: 칸마다 검사 안 함
      int lastw = -1;
      while (!(x == x1 && y == y1)) {
        mark_in(sh.missb, x, y, L, seen, lastw);
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x += sx; }
        if (e2 <= dx) { err += dx; y += sy; }
      }
      mark_in(ty == 1 ? sh.hitb : sh.missb, x1, y1, L, seen, lastw);
      continue;
    }
    while (!(x == x1 && y == y1)) {
      mark(sh.missb, x, y);
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
    mark(ty == 1 ? sh.hitb : sh.missb, x1, y1);
  }
}

// ---- 5. 격자 갱신(모든 스레드, 낱말마다 한 스레드): 맞음 우선, 정수 덧셈 + 자르기, 본 적 표시, 방 안 새로 본 칸 수 ------------
DEV void phase_apply(const MapCore& m, Scratch& sh, int16_t* L, uint32_t* seen, int tid, int nt) {
  int cnt = 0;
#ifdef __CUDA_ARCH__
  // GPU: 워프가 낱말 32 개씩 훑어 표시된 낱말만 골라(ballot), 그 낱말의 칸 32 개를 레인 32 개가 맡는다(로그 오즈 64 B 를 한 번에 읽음).
  // 낱말 4 개까지 읽기를 먼저 내고 계산한다(DRAM 대기를 겹침). 칸마다 연산은 아래 CPU 고리와 같다.
  const int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
  const uint32_t bit = 1u << lane;
  for (int base = warp * 32; base < NWORD; base += nwarp * 32) {
    uint32_t mask = __ballot_sync(0xffffffffu, (sh.hitb[base + lane] | sh.missb[base + lane]) != 0u);
#ifdef MAP_PROF
    if (lane == 0) atomicAdd(&g_prof[P_NSEC + 2], (unsigned long long)__popc(mask));
#endif
    while (mask) {
      int wv[4], nb = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wv[j] = -1;
        if (mask) { wv[j] = base + __ffs(mask) - 1; mask &= mask - 1; ++nb; }
      }
      uint32_t hh[4], aa[4], oo[4];
      int lv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j >= nb) continue;
        const int w = wv[j];
        hh[j] = sh.hitb[w];
        aa[j] = hh[j] | (sh.missb[w] & ~hh[j]);
        oo[j] = seen[w];
        lv[j] = (aa[j] & bit) ? (int)L[(size_t)w * 32 + lane] : 0;
      }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j >= nb) continue;
        const int w = wv[j];
        if (aa[j] & bit) {
          int Lv = lv[j];
          Lv = (hh[j] & bit) ? (Lv + MP::q_hit < MP::q_max ? Lv + MP::q_hit : MP::q_max) : (Lv + MP::q_miss > MP::q_min ? Lv + MP::q_miss : MP::q_min);
          L[(size_t)w * 32 + lane] = (int16_t)Lv;
          if (!(oo[j] & bit)) {   // 새로 본 칸 중 방 안
            const int idx = w * 32 + lane;
            const float cxw = ((float)(idx % GW + GX0) + 0.5f) * RES, cyw = ((float)(idx / GW + GX0) + 0.5f) * RES;
            cnt += absf(cxw) < m.rhx && absf(cyw) < m.rhy;
          }
        }
        if (lane == 0) seen[w] = oo[j] | aa[j];
      }
    }
  }
  put_part(sh, tid, cnt);
#else
  // CPU(참조판): 낱말마다 칸을 차례로. 맞음 우선, 정수 덧셈 + 자르기
  for (int w = tid; w < NWORD; w += nt) {
    const uint32_t h = sh.hitb[w], ms = sh.missb[w] & ~h, any = h | ms;
    if (!any) continue;
    const uint32_t old = seen[w], nw = any & ~old;
    seen[w] = old | any;
    for (int b = 0; b < 32; ++b) {
      const uint32_t bit = 1u << b;
      if (!(any & bit)) continue;
      const int idx = w * 32 + b;
      int Lv = L[idx];
      Lv = (h & bit) ? (Lv + MP::q_hit < MP::q_max ? Lv + MP::q_hit : MP::q_max) : (Lv + MP::q_miss > MP::q_min ? Lv + MP::q_miss : MP::q_min);
      L[idx] = (int16_t)Lv;
      if (nw & bit) {   // 새로 본 칸 중 방 안
        const float cxw = ((float)(idx % GW + GX0) + 0.5f) * RES, cyw = ((float)(idx / GW + GX0) + 0.5f) * RES;
        cnt += absf(cxw) < m.rhx && absf(cyw) < m.rhy;
      }
    }
  }
  put_part(sh, tid, cnt);
#endif
}

DEV void phase_finish(MapCore& m, const Scratch& sh, int nt) {
  const int s = sum_part(sh, nt);
  int n_obj = 0;
  for (int p = 0; p < N_PRIM; ++p) n_obj += sh.found[p];
  m.n_obj_conf = n_obj;
  m.n_task_conf = sh.found[0];
  m.n_seen_room += s;
  m.n_kf += 1;
  m.n_kf_total += 1;
  m.first = 0;
  m.since = 0;
  m.lx = m.ex; m.ly = m.ey; m.lyaw = m.eyaw;
  m.vmax = 0.f; m.wmax = 0.f;
}

DEV void write_metrics(const MapCore& m, const EnvView& e, float* met, int N, int i) {
  const float dx = m.ex - e.x, dy = m.ey - e.y;
  met[M_TASK * N + i] = (float)m.n_task_conf;
  met[M_OBJ * N + i] = (float)m.n_obj_conf / (float)N_PRIM;
  met[M_SEEN * N + i] = m.room_cells > 0 ? (float)m.n_seen_room / (float)m.room_cells : 0.f;
  met[M_ERR_XY * N + i] = sqrtf(dx * dx + dy * dy);
  met[M_ERR_YAW * N + i] = absf(wrap_pi(m.eyaw - e.yaw));
  met[M_KF * N + i] = (float)m.kf_flag;
}

// ---- 한 판 한 스텝. Sync = GPU __syncthreads / CPU 아무것도 안 함(tid 0, nt 1) --------------------------------------------------
// keyframe 갱신 본체(시작 단계 뒤, B_KF 일 때만). 블록(GPU) 또는 스레드 하나(CPU)
template <class Sync>
DEV void map_keyframe(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, int tid, int nt, int bug, const Sync& sync) {
  phase_cast(m, sh, e, tid, nt);
  sync();
  PROF_MARK(P_CAST);
  obj_pre(m, sh, tid, nt);
  sync();
  PROF_MARK(P_OBJPRE);
  if (tid == 0) obj_detect(m, sh, e, nt);
  sync();
  PROF_MARK(P_POSE_DET);
  obj_keys(m, sh, tid, nt);
  sync();
#ifdef __CUDA_ARCH__
  if (tid < 32) obj_greedy_warp(sh, tid);   // GPU: 워프 0 이 셔플로(바퀴마다 블록 동기 없음). 고르는 규칙은 아래 일반판과 같음
  sync();
#else
  for (;;) {   // 탐욕 짝짓기: 한 바퀴에 한 쌍
    obj_argmin_local(sh, tid, nt);
    sync();
    if (tid == 0) obj_argmin_pick(sh, nt);
    sync();
    if (!sh.more) break;
  }
#endif
  obj_update_matched(m, sh, tid, nt, bug);
  sync();
  if (tid == 0) obj_update_new(m, sh, bug);
  sync();
  PROF_MARK(P_ASSOC);
  obj_absence(m, sh, e, tid, nt);
  sync();
  PROF_MARK(P_ABSENCE);
  obj_complete(m, sh, tid, nt);
  phase_mark(m, sh, L, seen, tid, nt);
  sync();
  PROF_MARK(P_MARK);
  phase_apply(m, sh, L, seen, tid, nt);
  sync();
  PROF_MARK(P_APPLY);
  if (tid == 0) phase_finish(m, sh, nt);
}

// 시작 단계 뒤의 나머지(flags = phase_begin 결과): 리셋이면 격자 비움, keyframe 이면 갱신, 마지막에 완성도 쓰기.
// GPU 는 keyframe·리셋인 판만 블록으로 이 함수를 부르고(map_kf_kernel), 나머지 판은 시작 커널이 완성도만 쓴다.
template <class Sync>
DEV void map_rest(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, float* met, int N, int i,
                  int tid, int nt, int bug, int flags, const Sync& sync) {
  if (flags & B_RESET) { clear_grid(L, seen, tid, nt); PROF_MARK(P_RESET); }
  if (flags & B_KF) map_keyframe(m, sh, e, L, seen, tid, nt, bug, sync);
  if (tid == 0) write_metrics(m, e, met, N, i);
  PROF_MARK(P_FINISH);
}

// 한 판 한 스텝 전체(CPU 참조판: tid 0, nt 1). GPU 는 같은 phase_begin → map_rest 를 두 커널로 나눠 부른다.
template <class Sync>
DEV void map_block(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, float* met, int N, int i,
                   int tid, int nt, int bug, int force_kf, const Sync& sync) {
  const int flags = phase_begin(m, e, force_kf);
  map_rest(m, sh, e, L, seen, met, N, i, tid, nt, bug, flags, sync);
}

}  // namespace gmap
