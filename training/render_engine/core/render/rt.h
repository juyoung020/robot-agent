// 렌더 모듈 기본: 아핀 변환, 2단 BVH 노드, 광선-상자·광선-삼각형, BLAS/TLAS 순회 (손으로 짬, 라이브러리 없음).
// 같은 코드가 층 1(C++ 호스트)과 층 2(CUDA)에서 돈다(EHD). 두 층이 비트까지 같으려면:
//  - 식 순서를 바꾸지 않는다(-ffp-contract=off / -fmad=false). min/max 는 fminf 대신 비교식(NaN 처리까지 같게).
//  - 순회 순서는 부동소수 비교로만 정해진다 -> 같은 트리면 같은 순서, 같은 최근접 결과(동률은 먼저 찾은 것).
//  - FTZ/DAZ: GPU -ftz=true, 호스트는 FtzScope(MXCSR FTZ+DAZ) 안에서 돈다.
#pragma once
#include <cstdint>

#include "core/common/pmath.h"

namespace eng {
namespace rnd {

constexpr float kInf = 3.402823466e+38f;  // FLT_MAX (진짜 inf 는 0*inf=NaN 위험이 있어 쓰지 않는다)

EHD float fmn(float a, float b) { return a < b ? a : b; }
EHD float fmx(float a, float b) { return a > b ? a : b; }
EHD float fab(float a) { return a < 0.0f ? -a : a; }

// ---- 3x4 아핀 (열 벡터 규약, 행 우선 저장): p' = R p + t,  m = [r00 r01 r02 t0 | r10 r11 r12 t1 | r20 r21 r22 t2]
// USD 행렬(행 벡터, p_w = p * M)을 옮길 때는 전치: m[r*4+c] = M[c][r] (c<3), m[r*4+3] = M[3][r].
struct Aff {
  float m[12];
};

EHD V3 xpoint(const Aff& a, const V3& p) {
  return V3{a.m[0] * p.x + a.m[1] * p.y + a.m[2] * p.z + a.m[3], a.m[4] * p.x + a.m[5] * p.y + a.m[6] * p.z + a.m[7],
            a.m[8] * p.x + a.m[9] * p.y + a.m[10] * p.z + a.m[11]};
}
EHD V3 xvec(const Aff& a, const V3& v) {
  return V3{a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[4] * v.x + a.m[5] * v.y + a.m[6] * v.z,
            a.m[8] * v.x + a.m[9] * v.y + a.m[10] * v.z};
}
// 법선: 역행렬의 전치로 (inv 를 넘긴다) -> n' = inv^T n
EHD V3 xnormal_inv(const Aff& inv, const V3& n) {
  return V3{inv.m[0] * n.x + inv.m[4] * n.y + inv.m[8] * n.z, inv.m[1] * n.x + inv.m[5] * n.y + inv.m[9] * n.z,
            inv.m[2] * n.x + inv.m[6] * n.y + inv.m[10] * n.z};
}
// a ∘ b (먼저 b, 다음 a)
EHD Aff aff_mul(const Aff& a, const Aff& b) {
  Aff r;
  for (int i = 0; i < 3; ++i) {
    const float a0 = a.m[i * 4 + 0], a1 = a.m[i * 4 + 1], a2 = a.m[i * 4 + 2];
    for (int j = 0; j < 4; ++j) r.m[i * 4 + j] = a0 * b.m[0 * 4 + j] + a1 * b.m[1 * 4 + j] + a2 * b.m[2 * 4 + j];
    r.m[i * 4 + 3] = r.m[i * 4 + 3] + a.m[i * 4 + 3];
  }
  return r;
}
// 일반 3x4 역행렬 (축척·기울임 포함): 수반행렬 / 행렬식
EHD Aff aff_inv(const Aff& a) {
  const float* m = a.m;
  const float c00 = m[5] * m[10] - m[6] * m[9], c01 = m[6] * m[8] - m[4] * m[10], c02 = m[4] * m[9] - m[5] * m[8];
  const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  const float id = 1.0f / det;
  Aff r;
  r.m[0] = c00 * id;
  r.m[1] = (m[2] * m[9] - m[1] * m[10]) * id;
  r.m[2] = (m[1] * m[6] - m[2] * m[5]) * id;
  r.m[4] = c01 * id;
  r.m[5] = (m[0] * m[10] - m[2] * m[8]) * id;
  r.m[6] = (m[2] * m[4] - m[0] * m[6]) * id;
  r.m[8] = c02 * id;
  r.m[9] = (m[1] * m[8] - m[0] * m[9]) * id;
  r.m[10] = (m[0] * m[5] - m[1] * m[4]) * id;
  const float tx = m[3], ty = m[7], tz = m[11];
  r.m[3] = -(r.m[0] * tx + r.m[1] * ty + r.m[2] * tz);
  r.m[7] = -(r.m[4] * tx + r.m[5] * ty + r.m[6] * tz);
  r.m[11] = -(r.m[8] * tx + r.m[9] * ty + r.m[10] * tz);
  return r;
}

// ---- BVH 노드 (BLAS·TLAS 공용): 자식 둘의 상자를 부모가 들고 있다(부모에서 두 자식을 한꺼번에 시험).
// 자식 번호 c: c >= 0 내부 노드, c == kEmpty 없음, 그 밖(음수) 잎 = ~((first << 3) | (count - 1)), count 1..8
struct Node2 {
  float lo0[3], hi0[3], lo1[3], hi1[3];
  int32_t c0, c1, pad0, pad1;
};
constexpr int32_t kEmpty = INT32_MIN;
EHD int32_t leaf_code(uint32_t first, uint32_t count) { return ~int32_t((first << 3) | (count - 1u)); }
EHD uint32_t leaf_first(int32_t c) { return uint32_t(~c) >> 3; }
EHD uint32_t leaf_count(int32_t c) { return (uint32_t(~c) & 7u) + 1u; }

// 교차용 삼각형 (굽기 때 v1-v0, v2-v0 를 미리 계산해 둔다 -> 두 층 모두 같은 저장값을 읽는다)
struct TriX {
  float v0[3], e1[3], e2[3];
};

struct Ray {
  V3 o, d, inv;  // inv = 1/d (d 의 0 성분은 굽기 전에 ±1e-20 으로 바꾼다)
};
EHD Ray make_ray(const V3& o, const V3& d_in) {
  V3 d = d_in;
  const float tiny = 1e-20f;
  if (fab(d.x) < tiny) d.x = d.x < 0.0f ? -tiny : tiny;
  if (fab(d.y) < tiny) d.y = d.y < 0.0f ? -tiny : tiny;
  if (fab(d.z) < tiny) d.z = d.z < 0.0f ? -tiny : tiny;
  return Ray{o, d, V3{1.0f / d.x, 1.0f / d.y, 1.0f / d.z}};
}

// 광선-상자 (slab). 맞으면 들어가는 t, 아니면 kInf. 반올림으로 스치는 광선을 놓치지 않게 먼 쪽을 약간 넓힌다.
EHD float box_t(const float* lo, const float* hi, const Ray& r, float tmin, float tmax) {
  const float t0x = (lo[0] - r.o.x) * r.inv.x, t1x = (hi[0] - r.o.x) * r.inv.x;
  const float t0y = (lo[1] - r.o.y) * r.inv.y, t1y = (hi[1] - r.o.y) * r.inv.y;
  const float t0z = (lo[2] - r.o.z) * r.inv.z, t1z = (hi[2] - r.o.z) * r.inv.z;
  const float tn = fmx(fmx(fmn(t0x, t1x), fmn(t0y, t1y)), fmx(fmn(t0z, t1z), tmin));
  const float tf = fmn(fmn(fmx(t0x, t1x), fmx(t0y, t1y)), fmn(fmx(t0z, t1z), tmax)) * 1.0000004f;
  return tn <= tf ? tn : kInf;
}

// Möller–Trumbore, 양면. 맞으면 true 와 t,u,v
EHD bool tri_hit(const TriX& T, const Ray& r, float tmin, float tbest, float& t, float& u, float& v) {
  const V3 e1{T.e1[0], T.e1[1], T.e1[2]}, e2{T.e2[0], T.e2[1], T.e2[2]};
  const V3 p = cross(r.d, e2);
  const float det = dot(e1, p);
  if (det == 0.0f) return false;
  const float id = 1.0f / det;
  const V3 s{r.o.x - T.v0[0], r.o.y - T.v0[1], r.o.z - T.v0[2]};
  const float uu = dot(s, p) * id;
  if (uu < 0.0f || uu > 1.0f) return false;
  const V3 q = cross(s, e1);
  const float vv = dot(r.d, q) * id;
  if (vv < 0.0f || uu + vv > 1.0f) return false;
  const float tt = dot(e2, q) * id;
  if (!(tt > tmin && tt < tbest)) return false;
  t = tt;
  u = uu;
  v = vv;
  return true;
}

struct Hit {
  float t, u, v;
  int32_t inst;  // 인스턴스(메시 prim) 번호, -1 = 못 맞춤
  int32_t tri;   // 그 기하의 굽힌(재배열된) 삼각형 번호 (전역 번호)
};

#ifndef RENDER_STACK
#define RENDER_STACK 32
#endif
constexpr int kStack = RENDER_STACK;  // 순회 스택 (TLAS·BLAS 가 하나를 같이 씀, scene.h trace). radio 실측 최대 20 칸(render_stats)
constexpr int kBlasMaxDepth = 32;     // BLAS 굽기 깊이 한계: TLAS 깊이 + 이것 < kStack 이어야 칸이 안 넘친다

// 순회 통계 (호스트 진단 도구 tests/render/render_stats.cpp 만 -DRENDER_STATS 로 켬). 켜지 않으면 코드 없음.
#ifdef RENDER_STATS
struct RStats { uint64_t tlas_nodes, inst_leaves, blas_nodes, tri_tests, max_sp; };
inline thread_local RStats g_rstats{};
#define RSTAT(x) (g_rstats.x++)
#define RSTAT_SP(v) (g_rstats.max_sp = (v) > g_rstats.max_sp ? (v) : g_rstats.max_sp)
#else
#define RSTAT(x) ((void)0)
#define RSTAT_SP(v) ((void)0)
#endif

// BLAS 순회 (지역 좌표 광선). nodes/tris 는 이 기하의 시작을 가리킨다. 전역 삼각형 번호 = tri_base + 지역 번호.
// any = true: 그림자·가림 광선 — 처음 맞으면 바로 끝(결과는 "맞은 것이 있나" 하나라 순서와 무관). 반환 = 끝냄 여부.
EHD bool blas_trace(const Node2* nodes, const TriX* tris, int32_t tri_base, const Ray& r, float tmin, Hit& h, int32_t inst,
                    bool any = false) {
  int32_t stack[kStack];
  float stack_t[kStack];  // 넣을 때의 들어가는 t: 꺼낼 때 이미 더 가까운 것을 찾았으면 건너뜀
  int sp = 0;
  int32_t cur = 0;
  for (;;) {
    if (cur >= 0) {
      RSTAT(blas_nodes);
      const Node2& n = nodes[cur];
      const float ta = n.c0 != kEmpty ? box_t(n.lo0, n.hi0, r, tmin, h.t) : kInf;
      const float tb = n.c1 != kEmpty ? box_t(n.lo1, n.hi1, r, tmin, h.t) : kInf;
      const bool ha = ta < kInf, hb = tb < kInf;
      if (ha && hb) {
        const bool a_first = ta <= tb;
        const int32_t nearc = a_first ? n.c0 : n.c1, farc = a_first ? n.c1 : n.c0;
        if (sp < kStack) {
          stack[sp] = farc;
          stack_t[sp] = a_first ? tb : ta;
          ++sp;
        }
        cur = nearc;
        continue;
      }
      if (ha) { cur = n.c0; continue; }
      if (hb) { cur = n.c1; continue; }
    } else {
      const uint32_t f = leaf_first(cur), c = leaf_count(cur);
      for (uint32_t k = 0; k < c; ++k) {
        float t, u, v;
        RSTAT(tri_tests);
        if (tri_hit(tris[f + k], r, tmin, h.t, t, u, v)) {
          h.t = t;
          h.u = u;
          h.v = v;
          h.inst = inst;
          h.tri = tri_base + int32_t(f + k);
          if (any) return true;
        }
      }
    }
    // 꺼내기: 넣을 때의 들어가는 t 가 이미 찾은 것보다 먼 칸은 건너뛴다
    bool got = false;
    while (sp > 0) {
      --sp;
      if (stack_t[sp] <= h.t) {
        cur = stack[sp];
        got = true;
        break;
      }
    }
    if (!got) break;
  }
  return false;
}

}  // namespace rnd
}  // namespace eng
