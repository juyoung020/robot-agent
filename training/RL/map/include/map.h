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

// 블록 안 작업 공간(GPU 공유 메모리, CPU 지역 변수)
struct Scratch {
  uint32_t hitb[NWORD], missb[NWORD];
  float colx[NCOL], coly[NCOL];     // 줄 끝(베이스 기준)
  int colt[NCOL];                   // 0 없음, 1 맞음(띠 안), 2 빈 광선 끝(바닥)
  int vis[N_PRIM * NPT];
  int part[NT];
  int do_kf, do_reset;
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

// ---- 광선 -------------------------------------------------------------------------------------------------------------
constexpr float kInf = 1e30f;
DEV float ray_box(const float o[3], const float d[3], const Prim& b) {   // 들어가는 t (> 0), 없으면 kInf. 시작점이 안이면 무시
  float t0 = -kInf, t1 = kInf;
  for (int a = 0; a < 3; ++a) {
    if (absf(d[a]) < 1e-12f) {
      if (o[a] < b.lo[a] || o[a] > b.hi[a]) return kInf;
    } else {
      const float inv = 1.f / d[a];
      float ta = (b.lo[a] - o[a]) * inv, tb = (b.hi[a] - o[a]) * inv;
      if (ta > tb) { const float x = ta; ta = tb; tb = x; }
      t0 = maxf(t0, ta);
      t1 = minf(t1, tb);
    }
  }
  if (t0 > t1 || t0 <= 0.f) return kInf;
  return t0;
}
DEV float ray_room(const float o[3], const float d[3], float rhx, float rhy) {   // 방 안에서 벽·바닥·천장까지
  float t = kInf;
  if (d[0] > 0.f) t = minf(t, (rhx - o[0]) / d[0]); else if (d[0] < 0.f) t = minf(t, (-rhx - o[0]) / d[0]);
  if (d[1] > 0.f) t = minf(t, (rhy - o[1]) / d[1]); else if (d[1] < 0.f) t = minf(t, (-rhy - o[1]) / d[1]);
  if (d[2] < 0.f) t = minf(t, -o[2] / d[2]); else if (d[2] > 0.f) t = minf(t, (MP::wall_h - o[2]) / d[2]);
  return t;
}
DEV float cast(const MapCore& m, const float o[3], const float d[3], int skip) {
  float t = ray_room(o, d, m.rhx, m.rhy);
  for (int p = 0; p < N_PRIM; ++p)
    if (p != skip) t = minf(t, ray_box(o, d, m.prim[p]));
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

// ---- 1. 시작(스레드 0): 리셋, 오도메트리 표류, 움직임 거르기 ---------------------------------------------------------------
DEV void phase_begin(MapCore& m, Scratch& sh, const EnvView& e, int force_kf) {
  sh.do_reset = 0;
  if (e.ep != m.ep) {
    reset_core(m, e);
    sh.do_reset = 1;
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
  sh.do_kf = kf;
  m.kf_flag = kf;
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
  for (int j = 0; j < N_PRIM; ++j)
    if (j != pi && ray_box(o, r, m.prim[j]) < 1.f) return 0;   // 다른 상자가 사이를 가림
  return 1;
}

DEV void phase_cast(const MapCore& m, Scratch& sh, const EnvView& e, int tid, int nt) {
  for (int w = tid; w < NWORD; w += nt) { sh.hitb[w] = 0u; sh.missb[w] = 0u; }
  const Cam k = cam_consts();
  float s, c;
  sincosf_d(e.yaw, &s, &c);
  float o[3];
  cam_world(e.x, e.y, c, s, o);
  // scan.cpp makeScan: 띠 [band_lo, band_hi] 안 가장 가까운 점 = 장애물, 띠 아래(바닥) 가장 먼 점 = 빈 광선 끝. 기울기 0 이라 열 = 방위 칸
  for (int col = tid; col < NCOL; col += nt) {
    const float xn = ((float)(2 * col + 1 - NCOL) / (float)NCOL) * k.tanh;
    float best = kInf, hit_t = 0.f, floor_r = 0.f, floor_t = 0.f;
    for (int r = 0; r < NROW; ++r) {
      const float yn = ((float)(2 * r + 1 - NROW) / (float)NROW) * k.tanv;
      float d[3];
      pix_dir(c, s, xn, yn, d);
      const float t = cast(m, o, d, -1);
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
  for (int p = tid; p < N_PRIM * NPT; p += nt) sh.vis[p] = vis_point(m, o, c, s, k, p);
}

// ---- 3. 자세 보정 + 검출 + 물체 기억(스레드 0, 순서대로) -----------------------------------------------------------------
struct Det { int cls; float pos[3], ext[3], score; };

DEV float max3(const float v[3]) { return maxf(v[0], maxf(v[1], v[2])); }

DEV void phase_objects(MapCore& m, Scratch& sh, const EnvView& e, int bug) {
  const Cam k = cam_consts();
  // 3a. keyframe 맞추기: 맞은 줄이 충분하고 제자리가 아니면 오차를 kf_corr 만큼 되돌림(slam2d keyframe 의 보정 흉내)
  int n_hits = 0;
  for (int col = 0; col < NCOL; ++col) n_hits += sh.colt[col] == 1;
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
  float s, c;
  sincosf_d(e.yaw, &s, &c);
  float o[3];
  cam_world(e.x, e.y, c, s, o);

  // 3b. 검출: 보이는 점 비율로 화소 넓이 어림 → min_px·min_points → 놓침 → 잡음(본 순간의 slam 오차를 물려받음) → 틀린 이름
  Det det[MAXDET];
  int nd = 0;
  for (int p = 0; p < N_PRIM; ++p) {
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += sh.vis[p * NPT + q];
    if (nv == 0) continue;
    const Prim& b = m.prim[p];
    float ctr[3], ext[3];
    for (int a = 0; a < 3; ++a) { ctr[a] = 0.5f * (b.lo[a] + b.hi[a]); ext[a] = b.hi[a] - b.lo[a]; }
    const float rx = ctr[0] - o[0], ry = ctr[1] - o[1];
    const float fwd = c * rx + s * ry, left = -s * rx + c * ry, up = ctr[2] - o[2];
    if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) continue;
    const float size_px = k.fx * max3(ext) / fwd;                        // objmap 부재 확인과 같은 식
    const float rh = sqrtf(rx * rx + ry * ry);
    const float ux = rx / rh, uy = ry / rh;
    const float wsil = ext[0] * absf(uy) + ext[1] * absf(ux);             // 수평 실루엣 폭
    const float area = (k.fx * wsil / fwd) * (k.fx * ext[2] / fwd) * ((float)nv / (float)NPT);   // 깊이 점 수(간격 1)
    if (size_px < (float)MP::min_px || area < (float)MP::min_points) continue;
    if (rand01(m.rng) < MP::p_miss) continue;
    const float sig_d = MP::dn0 + MP::dn2 * fwd * fwd;
    const float gd = gauss(m.rng) * sig_d, gl = gauss(m.rng) * MP::lat_n, gz = gauss(m.rng) * MP::lat_n;
    // 몸 좌표의 수평 시선 단위(fwd, left)/rh 와 그 수직
    const float bu = fwd / rh, bv = left / rh;
    const float f2 = fwd + bu * gd - bv * gl, l2 = left + bv * gd + bu * gl;
    const float bx = env::K::cam_x + f2, by = l2;
    Det& D = det[nd++];
    D.pos[0] = m.ex + (ec * bx - es * by);
    D.pos[1] = m.ey + (es * bx + ec * by);
    D.pos[2] = o[2] + up + gz;
    for (int a = 0; a < 3; ++a) D.ext[a] = maxf(0.01f, ext[a] + gauss(m.rng) * MP::ext_n);
    int cls = b.cls;
    if (rand01(m.rng) < MP::p_conf) cls = (cls + 1 + (int)(rand01(m.rng) * (float)(NCLS - 1))) % NCLS;
    D.cls = cls;
    D.score = 0.9f;
  }
  // 가짜 물체: 맞은 줄 하나 위에 작은 상자(가정)
  if (rand01(m.rng) < MP::p_fp) {
    const int col = (int)(rand01(m.rng) * (float)NCOL);
    const float zc = rand_range(m.rng, 0.05f, 0.5f), sz = rand_range(m.rng, 0.05f, 0.2f);
    const int cls = (int)(rand01(m.rng) * (float)NCLS);
    if (sh.colt[col] != 0 && nd < MAXDET) {
      const float bx = sh.colx[col], by = sh.coly[col];
      Det& D = det[nd++];
      D.pos[0] = m.ex + (ec * bx - es * by);
      D.pos[1] = m.ey + (es * bx + ec * by);
      D.pos[2] = zc;
      D.ext[0] = sz; D.ext[1] = sz; D.ext[2] = sz;
      D.cls = cls;
      D.score = 0.4f;
      m.n_fp_total += 1;
    }
  }

  // 3c. 같은 물체(objmap.cpp 2): 같은 이름끼리, 중심 거리 < max(da_min, da_k·큰 쪽 크기) 또는 상자 틈 < da_gap. (틈 + 1e-3·거리) 순 1:1 탐욕
  const int confirm_n = bug == 1 ? 1 : MP::confirm;   // 음성 대조: 확정 규칙 끔(한 번 보이면 확정)
  float key[MAXDET][KSLOT];
  int obs_to[MAXDET], hit[KSLOT];
  for (int b = 0; b < KSLOT; ++b) hit[b] = 0;
  for (int a = 0; a < nd; ++a) {
    obs_to[a] = -1;
    for (int b = 0; b < KSLOT; ++b) {
      key[a][b] = -1.f;
      const Slot& S = m.slot[b];
      if (!S.valid || S.cls != det[a].cls) continue;
      const float ee = maxf(max3(det[a].ext), max3(S.ext));
      const float thr = maxf(MP::da_min, MP::da_k * ee);
      float d2 = 0.f, g2 = 0.f;
      for (int q = 0; q < 3; ++q) {
        const float dd = det[a].pos[q] - S.pos[q];
        d2 = d2 + dd * dd;
        const float olo = det[a].pos[q] - 0.5f * det[a].ext[q], ohi = det[a].pos[q] + 0.5f * det[a].ext[q];
        const float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float gk = maxf(0.f, maxf(olo - mhi, mlo - ohi));
        g2 = g2 + gk * gk;
      }
      const float d = sqrtf(d2), gap = sqrtf(g2);
      if (d < thr || gap < MP::da_gap) key[a][b] = gap + 1e-3f * d;
    }
  }
  for (;;) {
    int ba = -1, bb = -1;
    float bk = 0.f;
    for (int a = 0; a < nd; ++a) {
      if (obs_to[a] >= 0) continue;
      for (int b = 0; b < KSLOT; ++b) {
        if (hit[b] || key[a][b] < 0.f) continue;
        if (ba < 0 || key[a][b] < bk) { ba = a; bb = b; bk = key[a][b]; }
      }
    }
    if (ba < 0) break;
    obs_to[ba] = bb;
    hit[bb] = 1;
  }
  // 3d. 갱신(objmap.cpp 3)
  const int t = m.t;
  for (int a = 0; a < nd; ++a) {
    const Det& D = det[a];
    if (obs_to[a] >= 0) {
      Slot& S = m.slot[obs_to[a]];
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
      continue;
    }
    // 안 맞은 관측: 같은 이름의 확정·사라짐 물체가 있으면 옮겨진 것으로 잇는다
    int mf = -1;
    float best = 1e9f;
    for (int b = 0; b < KSLOT; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || S.cls != D.cls || !S.confirmed || hit[b] || S.state != S_GONE) continue;
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
      hit[mf] = 1;
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
    hit[fs] = 1;
  }
  // 3e. 부재 확인(objmap.cpp 4): 확정·이번에 안 맞음·사라짐 아님·고정 종류 아님·작은 것. 믿는 자세로 화소에 투영하고
  //     그 화소의 깊이(참 장면)가 물체보다 occl 넘게 가까우면 가려짐. 아니면 놓침 +1
  for (int b = 0; b < KSLOT; ++b) {
    Slot& S = m.slot[b];
    if (!S.valid || !S.confirmed || hit[b] || S.state == S_GONE || is_static(S.cls)) continue;
    if (max3(S.ext) > MP::big) continue;
    const float rx = S.pos[0] - (m.ex + ec * env::K::cam_x), ry = S.pos[1] - (m.ey + es * env::K::cam_x);
    const float zc = ec * rx + es * ry, left = -es * rx + ec * ry, up = S.pos[2] - env::K::cam_z;
    if (zc < 0.3f || zc > MP::ozmax) continue;
    const float xn = -left / zc, yn = -up / zc;
    const float u = k.fx * xn + k.cxp, v = k.fx * yn + k.cyp;
    const float size_px = k.fx * max3(S.ext) / zc;
    if (size_px < (float)MP::min_px || u < 2.f || v < 2.f || u >= (float)(MP::img_w - 2) || v >= (float)(MP::img_h - 2)) continue;
    float d[3];
    pix_dir(c, s, xn, yn, d);
    const float tz = cast(m, o, d, -1);
    if (!(tz >= MP::zmin && tz <= MP::zmax)) continue;   // 깊이 없음
    if (tz < zc - MP::occl) continue;                    // 가려짐
    if (S.misses++ == 0) S.first_miss = t;
    if (S.misses >= MP::gone_misses && t - S.first_miss >= MP::gone_min_steps) S.state = S_GONE;
  }
  // 3f. 오래된 후보 버리기
  for (int b = 0; b < KSLOT; ++b) {
    Slot& S = m.slot[b];
    if (S.valid && !S.confirmed && t - S.last_seen > MP::prune_steps) S.valid = 0;
  }
  // 3g. 완성도: 참 물체마다 같은 이름의 확정(사라짐 아님) 칸이 짝 문턱 안에 있나
  int n_obj = 0, n_task = 0;
  for (int p = 0; p < N_PRIM; ++p) {
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
    n_obj += found;
    if (p == 0) n_task += found;
  }
  m.n_obj_conf = n_obj;
  m.n_task_conf = n_task;
}

// ---- 4. 격자 표시(모든 스레드): 믿는 자세로 줄을 격자에. 정수 Bresenham, 끝 칸 제외 → 맞음 끝은 hit, 빈 광선 끝은 miss ----------
DEV void or_bits(uint32_t* p, uint32_t v) {
#ifdef __CUDA_ARCH__
  atomicOr(p, v);
#else
  *p |= v;
#endif
}
DEV void mark(uint32_t* bits, int ix, int iy) {
  const int lx = ix - GX0, ly = iy - GX0;
  if (lx < 0 || ly < 0 || lx >= GW || ly >= GW) return;
  const int idx = ly * GW + lx;
  or_bits(&bits[idx >> 5], 1u << (idx & 31));
}
DEV void phase_mark(const MapCore& m, Scratch& sh, int tid, int nt) {
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
      if (nw & bit) {
        const float cxw = ((float)(idx % GW + GX0) + 0.5f) * RES, cyw = ((float)(idx / GW + GX0) + 0.5f) * RES;
        cnt += absf(cxw) < m.rhx && absf(cyw) < m.rhy;
      }
    }
  }
  sh.part[tid] = cnt;
}

DEV void phase_finish(MapCore& m, const Scratch& sh, int nt) {
  int s = 0;
  for (int k = 0; k < nt; ++k) s += sh.part[k];
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

// ---- 한 판 한 스텝(블록 하나). Sync = GPU __syncthreads / CPU 아무것도 안 함(tid 0, nt 1) ------------------------------------
template <class Sync>
DEV void map_block(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, float* met, int N, int i,
                   int tid, int nt, int bug, int force_kf, const Sync& sync) {
  if (tid == 0) phase_begin(m, sh, e, force_kf);
  sync();
  if (sh.do_reset) clear_grid(L, seen, tid, nt);
  if (sh.do_kf) {   // 블록 안에서 같은 값(공유)
    phase_cast(m, sh, e, tid, nt);
    sync();
    if (tid == 0) phase_objects(m, sh, e, bug);
    sync();
    phase_mark(m, sh, tid, nt);
    sync();
    phase_apply(m, sh, L, seen, tid, nt);
    sync();
    if (tid == 0) phase_finish(m, sh, nt);
  }
  if (tid == 0) write_metrics(m, e, met, N, i);
}

}  // namespace gmap
