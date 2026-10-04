// BEHAVIOR 집 장면(계획서 CURRICULUM_BEHAVIOR2026.md 5절 E2): 장치에 올린 장면 7 개를 모든 판이 같이 쓴다(판마다 복사 안 함).
// CPU 참조판과 GPU 커널이 같은 소스를 쓴다(DEV). 이 헤더는 혼자 선다 — RASC 로더(rasc.h)는 호스트 쪽 bscene_host.cpp 만 쓴다.
//
// 좌표: 장면은 세계 좌표(RASC). 판은 **창 좌표**(세계 − 창 가운데 (wx, wy))로 돈다 — 지도(../map)의 128 × 128 칸 0.10 m 고정 창이
// 판마다 그 판의 시작·목표를 덮도록 옮겨진 것과 같다. 창 가운데는 장면 격자 칸 경계에 맞춰 둔다(창 칸 = 장면 칸, 정수로 대응).
// 정적 상자 = RASC BOXES 중 문(열린 채로 둠)·카펫·보이기만 하는 것·납작한 것을 뺀 회전 상자(yaw). 1 m 칸 묶음(bin)으로 찾는다.
// 과제 물체(인스턴스 자세)는 판의 "움직이는 상자"(축 정렬, Entry::prim 중 sbox < 0)로 몸통·광선과 부딪힌다.
#pragma once
#include "detmath.h"

namespace bsc {
using namespace dm;

constexpr int MAXSC = 8;           // 장면 수 상한(BEHAVIOR 2026 은 7)
constexpr float CELL = 0.1f;       // 장면 격자(RASC) = 지도 칸
constexpr float BIN = 1.0f;        // 상자 찾기 칸 묶음 m
constexpr float INV_BIN = 1.0f;
constexpr int WIN = 128;           // 창 한 변 칸 수 = gmap::GW
constexpr float WIN_HALF = 6.4f;   // 창 반 변 m = GW·RES/2
constexpr int NPRIM = 9;           // 판의 지도 물체 = gmap::N_PRIM (0 = 목표)
constexpr float H_COLL = 0.35f;    // 바닥 z0 이 이보다 낮은 상자만 몸통과 부딪힘 (가정: 몸통 0.251 m + 접은 팔)
constexpr int N_RTYPE_TOK = 6;     // 방 토큰 종류(kitchen·bathroom·bedroom·living room·office·모름)
constexpr int MAXBR = 16;          // 창 안 방 수 상한(지도 토큰용)
constexpr int MAXBD = 16;          // 창 안 문 수 상한

enum BoxKind { BK_WALL = 1, BK_FURN = 2, BK_WINDOW = 4, BK_COLL = 8, BK_BAND = 16 };   // BK_COLL: z0 < H_COLL, BK_BAND: z 범위가 지도 높이 띠 [0.05, 0.50] 와 겹침
enum EntKind { EK_NONE = 0, EK_B1 = 1, EK_B2 = 2, EK_B3 = 3 };                      // 판 단계: B1 집 안 이동, B2 찾기, B3 다가가기
enum ListKind { L_ROOM = 0, L_OBJ = 1 };                                             // B1 은 방 표, B2·B3 는 물체 표

struct SBox { float cx, cy, hx, hy, c, s, z0, z1; };   // 세계 회전 상자(yaw 의 cos·sin), 32 B
struct SDoor { float x, y; int16_t ra, rb; };          // 문 가운데(세계), 양쪽 방(장면 방 번호, 없으면 −1)

struct SceneDev {
  float ox, oy;               // 칸 (0, 0) 의 왼쪽 아래 모서리(세계)
  int W, H;                   // 칸 수(0.10 m)
  int BW, BH;                 // 묶음 수(1 m, 같은 원점)
  const uint32_t* bstart;     // [BW·BH + 1] 묶음마다 상자 목록 시작
  const uint16_t* bitem;      // 상자 번호
  const SBox* box;            // [nbox]
  const uint8_t* bkind;       // BoxKind 비트
  const int16_t* bname;       // 이름 표 행(vla_v1), 가구 종류
  int nbox;
  const uint8_t* room;        // [H][W] 방 번호 + 1 (0 = 방 아님)
  const uint32_t* occ;        // [H][W] 비트: 띠 [0.05, 0.50] 와 겹치는 정적 상자가 칸을 덮음(처음 지도 C0·C1 의 참 점유)
  const int8_t* rtype;        // [nroom] 방 토큰 종류 0..5
  int nroom;
  const SDoor* door;
  int ndoor;
};

struct BPrim { float lo[3], hi[3]; int16_t name; int16_t sbox; };   // 창 좌표 축 정렬 상자, 이름 표 행, 정적 상자 번호(−1 = 과제 물체, 움직이는 상자)
struct Entry {            // 판 하나의 시작 조건(호스트가 미리 만들고 검사한 표)
  int16_t scene, list;    // 장면 번호, ListKind
  int16_t split, groom;   // 0 학습 / 1 공개 평가 인스턴스, 목표 방(장면 방 번호, B1) 또는 목표 물체가 있는 방(−1 없음)
  int32_t inst;           // RASC 인스턴스 번호
  int16_t nprim, task;    // 채운 물체 수(≤ NPRIM), RASC 과제 번호
  float wx, wy;           // 창 가운데(세계)
  float sx, sy, syaw;     // 로봇 시작(창 좌표)
  float gx, gy, gz;       // 목표: B1 = 가구 앞 바닥 점(gz 0), B2·B3 = 목표 물체 상자 가운데(창 좌표)
  float path;             // 시작 → 목표(B1: 점 0.5 m 안, 물체: 팔 닿는 원 안)까지 참 장면 최단 경로 m(호스트 8 이웃 다익스트라, 창 안)
  float ext[3];           // 목표 물체 상자 크기(B1 은 목표 가구)
  BPrim prim[NPRIM];      // 0 = 목표(B1 은 목표 가구, B2·B3 은 목표 물체), 다음 과제 물체, 다음 창 안 가구
};

struct SceneSet {         // 장치 메모리에 하나(커널은 포인터로 읽음). CPU 참조판은 같은 구조체를 호스트 포인터로
  SceneDev sc[MAXSC];
  int nsc;
  const Entry* ent;
  int nent;
  int loff[MAXSC][2][2], lcnt[MAXSC][2][2];   // [장면][ListKind][split] → ent 안 범위
  int lcnt_in[MAXSC][2][2];                   // 그 범위 앞쪽의 안 한도(엄격) 판 수 — 물체 표는 엄격 판을 앞에 둔다
  // 이름 표(vla_v1): 생김새 행 a(0..6) × 이름 행 n 의 확신도(bscene_host 가 name128·app128 로 계산), 비슷한 다른 이름 3, 상위어
  const float* conf1;     // [7][nname]
  const float* conf2;
  const int16_t* sim3;    // [nname][3]
  const int16_t* hyper;   // [nname]
  int nname;
};

// 커리큘럼(장치 값, 판 리셋 때 읽음 — 바꿔도 다시 컴파일·그래프 다시 잡기 없음)
struct BCurr {
  float p1, p2;           // B1·B2 비율(나머지 B3)
  uint32_t scene_mask;    // 쓸 장면 비트(SceneSet 번호)
  int split;              // 0 학습 인스턴스, 1 공개 평가, 2 둘 다
  float yaw_jit;          // 시작 yaw 흔들기 ±rad (기본 0 = 인스턴스 그대로)
  int strict;             // 1 = 물체 표(B2·B3)를 안 한도(RASC inner: E0 엄격 폭·무게·높이 + 엄격 문턱 성분) 판만, 0 = 바깥 한도(느슨, 기본)
  int pad[2];
};
constexpr BCurr kBCurrDefault = {0.34f, 0.33f, 0xffu, 0, 0.f, 0, {0, 0}};

// 지도 → 환경(앞 스텝의 지도 값): 정책이 아는 지도(자라는 지도)의 거리장과 목표 확정 여부
struct NavFb {
  const uint8_t* lev;     // [N][WIN·WIN] 거리장 단계(0.1 m 칸, 4·8 이웃 번갈아 BFS = 팔각 거리 ≈ 단계 × 0.1 m), 255 = 못 감
  const int* tag;         // [N] 거리장을 만든 판 번호(env ep), 다르면 쓰지 않음
  const int* conf;        // [N] 목표 물체가 지도에 확정(B2 성공 조건)
};

// ---------------------------------------------------------------------------------------------------------------------
DEV int bin_of(const SceneDev& d, float x, float y, int& bx, int& by) {   // 세계 → 묶음, 밖이면 0
  bx = (int)floorf((x - d.ox) * INV_BIN);
  by = (int)floorf((y - d.oy) * INV_BIN);
  return bx >= 0 && by >= 0 && bx < d.BW && by < d.BH;
}
DEV int room_at(const SceneDev& d, float x, float y) {   // 세계 점의 방(장면 방 번호), 없으면 −1
  const int c = (int)floorf((x - d.ox) / CELL), r = (int)floorf((y - d.oy) / CELL);
  if (c < 0 || r < 0 || c >= d.W || r >= d.H) return -1;
  return (int)d.room[(size_t)r * d.W + c] - 1;
}
// 몸통 직사각형(중심, yaw sin·cos, 반 길이·반 폭)과 회전 상자의 분리축 4 개(열린 겹침)
DEV bool rect_hits_obb(float x, float y, float s, float co, float hl, float hw, const SBox& b) {
  const float dx = b.cx - x, dy = b.cy - y;
  // 몸 축
  const float bc = co * b.c + s * b.s, bs = -s * b.c + co * b.s;   // 상자 축 x 를 몸 좌표로: (bc, bs)
  const float abc = absf(bc), abs_ = absf(bs);
  const float px = co * dx + s * dy, py = -s * dx + co * dy;
  if (!(absf(px) < hl + b.hx * abc + b.hy * abs_)) return false;
  if (!(absf(py) < hw + b.hx * abs_ + b.hy * abc)) return false;
  // 상자 축
  const float qx = b.c * dx + b.s * dy, qy = -b.s * dx + b.c * dy;
  if (!(absf(qx) < b.hx + hl * abc + hw * abs_)) return false;
  return absf(qy) < b.hy + hl * abs_ + hw * abc;
}
// 몸통과 축 정렬 상자(lo, hi xy)
DEV bool rect_hits_aabb(float x, float y, float s, float co, float hl, float hw, const float lo[3], const float hi[3]) {
  const float ex = 0.5f * (hi[0] - lo[0]), ey = 0.5f * (hi[1] - lo[1]);
  const float dx = 0.5f * (lo[0] + hi[0]) - x, dy = 0.5f * (lo[1] + hi[1]) - y;
  const float ac = absf(co), as = absf(s);
  if (!(absf(dx) < ex + ac * hl + as * hw)) return false;
  if (!(absf(dy) < ey + as * hl + ac * hw)) return false;
  if (!(absf(co * dx + s * dy) < hl + ex * ac + ey * as)) return false;
  return absf(-s * dx + co * dy) < hw + ex * as + ey * ac;
}
// 몸통(세계 중심)이 BK_COLL 정적 상자에 닿나: 몸통 외접원이 덮는 묶음만
DEV bool body_hits_scene(const SceneDev& d, float x, float y, float s, float co, float hl, float hw) {
  const float r = sqrtf(hl * hl + hw * hw);
  int bx0, by0, bx1, by1;
  bin_of(d, x - r, y - r, bx0, by0);
  bin_of(d, x + r, y + r, bx1, by1);
  bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
  bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
  for (int by = by0; by <= by1; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) {
        const int j = d.bitem[k];
        if (!(d.bkind[j] & BK_COLL)) continue;
        if (rect_hits_obb(x, y, s, co, hl, hw, d.box[j])) return true;
      }
    }
  return false;
}

// 2D 반직선 o + t·(dx, dy) 이 회전 상자에 들어가는 t(안이면 0), 없으면 tmax
DEV float ray_obb2(float ox, float oy, float dx, float dy, const SBox& b, float tmax) {
  const float rx = ox - b.cx, ry = oy - b.cy;
  const float lx = b.c * rx + b.s * ry, ly = -b.s * rx + b.c * ry;
  const float ux = b.c * dx + b.s * dy, uy = -b.s * dx + b.c * dy;
  float t0 = 0.f, t1 = tmax;
  if (ux > 1e-6f || ux < -1e-6f) {
    float ta = (-b.hx - lx) / ux, tb = (b.hx - lx) / ux;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(lx > -b.hx && lx < b.hx)) return tmax;
  if (uy > 1e-6f || uy < -1e-6f) {
    float ta = (-b.hy - ly) / uy, tb = (b.hy - ly) / uy;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(ly > -b.hy && ly < b.hy)) return tmax;
  return t0 <= t1 ? t0 : tmax;
}

// 묶음 걷기(Amanatides–Woo, 2D): 세계 반직선 o + t·d 가 지나는 묶음을 들어가는 t 순으로. f(상자 번호, 묶음 들어간 t) 를 부르고,
// 들어가는 t 가 lim() 이상이면 멈춘다(그 뒤 상자는 더 가까운 맞음을 낼 수 없음 — 결과 같음). 상자가 여러 묶음에 있으면 여러 번 불림(최솟값이라 같음)
template <class F, class L>
DEV void bin_walk(const SceneDev& d, float ox, float oy, float dx, float dy, const F& f, const L& lim) {
  const float fx = (ox - d.ox) * INV_BIN, fy = (oy - d.oy) * INV_BIN;
  int bx = (int)floorf(fx), by = (int)floorf(fy);
  const int sx = dx > 0.f ? 1 : -1, sy = dy > 0.f ? 1 : -1;
  const float adx = absf(dx), ady = absf(dy);
  const float tdx = adx > 1e-12f ? BIN / adx : 1e30f, tdy = ady > 1e-12f ? BIN / ady : 1e30f;
  float tx = adx > 1e-12f ? ((sx > 0 ? (float)(bx + 1) - fx : fx - (float)bx) * BIN) / adx : 1e30f;
  float ty = ady > 1e-12f ? ((sy > 0 ? (float)(by + 1) - fy : fy - (float)by) * BIN) / ady : 1e30f;
  // 격자 밖에서 시작하면 들어올 때까지 걷는다(창이 장면 밖까지 덮을 수 있음)
  float t = 0.f;
  for (int it = 0; it < 4096; ++it) {
    if (!(t < lim())) return;
    if (bx >= 0 && by >= 0 && bx < d.BW && by < d.BH) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) f((int)d.bitem[k], t);
    } else if ((bx < 0 && sx < 0) || (by < 0 && sy < 0) || (bx >= d.BW && sx > 0) || (by >= d.BH && sy > 0)) {
      return;   // 격자 밖으로 멀어짐
    }
    if (tx < ty) { t = tx; tx = tx + tdx; bx += sx; } else { t = ty; ty = ty + tdy; by += sy; }
  }
}

// 3D 선분 p → p + v (매개 0..1) 이 회전 상자를 지나나(열린 영역)
DEV bool seg_hits_obb3(const float p[3], const float v[3], const SBox& b) {
  const float rx = p[0] - b.cx, ry = p[1] - b.cy;
  const float lo3[3] = {b.c * rx + b.s * ry, -b.s * rx + b.c * ry, p[2]};
  const float u3[3] = {b.c * v[0] + b.s * v[1], -b.s * v[0] + b.c * v[1], v[2]};
  const float blo[3] = {-b.hx, -b.hy, b.z0}, bhi[3] = {b.hx, b.hy, b.z1};
  float t0 = 0.f, t1 = 1.f;
  for (int a = 0; a < 3; ++a) {
    if (u3[a] > 1e-9f || u3[a] < -1e-9f) {
      float ta = (blo[a] - lo3[a]) / u3[a], tb = (bhi[a] - lo3[a]) / u3[a];
      if (ta > tb) { const float q = ta; ta = tb; tb = q; }
      t0 = maxf(t0, ta); t1 = minf(t1, tb);
    } else if (!(lo3[a] > blo[a] && lo3[a] < bhi[a])) return false;
  }
  return t0 < t1;
}
DEV bool seg_hits_aabb3(const float p[3], const float v[3], const float lo[3], const float hi[3]) {
  float t0 = 0.f, t1 = 1.f;
  for (int a = 0; a < 3; ++a) {
    if (v[a] > 1e-9f || v[a] < -1e-9f) {
      float ta = (lo[a] - p[a]) / v[a], tb = (hi[a] - p[a]) / v[a];
      if (ta > tb) { const float q = ta; ta = tb; tb = q; }
      t0 = maxf(t0, ta); t1 = minf(t1, tb);
    } else if (!(p[a] > lo[a] && p[a] < hi[a])) return false;
  }
  return t0 < t1;
}
// 세계 선분 p → q 를 정적 상자(모든 높이, except 번호 하나 빼고)가 가리나
DEV bool seg_blocked_scene(const SceneDev& d, const float p[3], const float q[3], int except) {
  const float v[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  bool hit = false;
  bin_walk(d, p[0], p[1], v[0], v[1],
           [&](int j, float) { if (!hit && j != except && seg_hits_obb3(p, v, d.box[j])) hit = true; },
           [&]() { return hit ? -1.f : 1.f; });
  return hit;
}

// ---- 창 거리장(지도 → 환경): 칸 (x, y) 둘레 5 × 5 칸 중 닿은 칸의 (단계 × 0.1 m + 칸 가운데까지 거리) 최소. 없으면 −1 ----
DEV float field_dist(const uint8_t* lev, float x, float y) {
  const float fx = (x + WIN_HALF) / CELL, fy = (y + WIN_HALF) / CELL;
  const int cx = (int)floorf(fx), cy = (int)floorf(fy);
  float best = 1e30f;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const int ix = cx + dx, iy = cy + dy;
      if (ix < 0 || iy < 0 || ix >= WIN || iy >= WIN) continue;
      const int L = lev[iy * WIN + ix];
      if (L == 255) continue;
      const float ccx = ((float)ix + 0.5f) * CELL - WIN_HALF, ccy = ((float)iy + 0.5f) * CELL - WIN_HALF;
      const float ex = x - ccx, ey = y - ccy;
      const float dd = (float)L * CELL + sqrtf(ex * ex + ey * ey);
      if (dd < best) best = dd;
    }
  return best < 1e30f ? best : -1.f;
}

// ---- B4·B5 성공 판정(정의만 — 잡기 물리는 E6). 상자는 축 정렬 lo/hi(창 또는 세계 좌표, 같은 좌표계) ----
// B4 집기: 든 상태(E6 규칙의 is_grasping 대응 — 인자로) + 처음 받침 윗면에서 lift 이상 떨어짐(not ontop)
DEV bool pred_grasped(bool held, const float obj_lo[3], float support_top, float lift = 0.02f) { return held && obj_lo[2] >= support_top + lift; }
// ontop(가정, OmniGibson 의 접촉 + 위쪽 판정을 상자로): 물체 바닥이 받침 윗면 ± tol 안, 물체 가운데 xy 가 받침 윗면 사각형 안
DEV bool pred_ontop(const float o_lo[3], const float o_hi[3], const float s_lo[3], const float s_hi[3], float tol = 0.02f) {
  const float cx = 0.5f * (o_lo[0] + o_hi[0]), cy = 0.5f * (o_lo[1] + o_hi[1]);
  return absf(o_lo[2] - s_hi[2]) <= tol && cx > s_lo[0] && cx < s_hi[0] && cy > s_lo[1] && cy < s_hi[1];
}
// inside(가정): 물체 상자 가운데가 용기 상자 안이고 물체 xy 가 용기 xy 안에 들어감(위 열린 용기)
DEV bool pred_inside(const float o_lo[3], const float o_hi[3], const float c_lo[3], const float c_hi[3]) {
  const float cz = 0.5f * (o_lo[2] + o_hi[2]);
  return o_lo[0] >= c_lo[0] && o_hi[0] <= c_hi[0] && o_lo[1] >= c_lo[1] && o_hi[1] <= c_hi[1] && cz > c_lo[2] && cz < c_hi[2];
}

}  // namespace bsc
