// 판마다 프레임마다 새로 만드는 TLAS (LBVH, Karras 2012 "Maximizing Parallelism in the Construction of BVHs...").
// CPU(층 1)와 GPU(층 2)가 같은 트리를 만들게 짠다:
//  - 인스턴스 월드 행렬 = anchor ∘ rel, 역행렬, 월드 AABB: 같은 EHD 식.
//  - 중심 경계: min/max (순서 무관, 정확).  Morton 30 비트 + 인스턴스 번호 = 64 비트 키 -> 키가 모두 달라 정렬 결과가 하나뿐.
//  - 내부 노드 i 의 범위·분할: 키만 보는 정수 연산.  상자: 자식 상자의 min/max (순서 무관, 정확).
#pragma once
#include <cstdint>

#include "core/render/scene.h"

namespace eng {
namespace rnd {

// 인스턴스 하나: 월드·역행렬·AABB (Arvo: 중심·반폭을 |R| 로 옮김). 반올림에도 보수적이게 반폭을 조금 키운다.
EHD void inst_prepare(const SceneView& S, const EnvView& E, int32_t i) {
  const InstInfo& in = S.insts[i];
  const Aff w = aff_mul(E.anchor[in.anchor], in.rel);
  E.inst_world[i] = w;
  E.inst_inv[i] = aff_inv(w);
  const GeomInfo& g = S.geoms[in.geom];
  const float cx = (g.lo[0] + g.hi[0]) * 0.5f, cy = (g.lo[1] + g.hi[1]) * 0.5f, cz = (g.lo[2] + g.hi[2]) * 0.5f;
  const float ex = (g.hi[0] - g.lo[0]) * 0.5f, ey = (g.hi[1] - g.lo[1]) * 0.5f, ez = (g.hi[2] - g.lo[2]) * 0.5f;
  float* b = E.inst_box + 6 * i;
  for (int r = 0; r < 3; ++r) {
    const float* m = w.m + 4 * r;
    const float c = m[0] * cx + m[1] * cy + m[2] * cz + m[3];
    const float e = (fab(m[0]) * ex + fab(m[1]) * ey + fab(m[2]) * ez) * 1.000002f + 1e-6f;
    b[r] = c - e;
    b[3 + r] = c + e;
  }
}

EHD uint32_t expand10(uint32_t v) {  // 10 비트를 3 칸 간격으로
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}
EHD uint32_t quant10(float x) {
  const float s = fmn(fmx(x * 1024.0f, 0.0f), 1023.0f);
  return uint32_t(s);
}
// 키 = (Morton(중심) << 32) | 인스턴스 번호.  clo/cinv = 중심 경계의 최솟값과 1/폭(폭 0 이면 0)
EHD uint64_t inst_key(const float* box, const float* clo, const float* cinv, int32_t i) {
  const float x = ((box[0] + box[3]) * 0.5f - clo[0]) * cinv[0];
  const float y = ((box[1] + box[4]) * 0.5f - clo[1]) * cinv[1];
  const float z = ((box[2] + box[5]) * 0.5f - clo[2]) * cinv[2];
  const uint32_t m = (expand10(quant10(x)) << 2) | (expand10(quant10(y)) << 1) | expand10(quant10(z));
  return (uint64_t(m) << 32) | uint64_t(uint32_t(i));
}

EHD int clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
  return __clzll((long long)x);
#else
  return x ? __builtin_clzll(x) : 64;
#endif
}
EHD int lbvh_delta(const uint64_t* k, int n, int i, int j) {
  if (j < 0 || j >= n) return -1;
  return clz64(k[i] ^ k[j]);
}
// 내부 노드 i (0..n-2) 의 두 자식. 자식 부호: >=0 내부 노드, 음수 = leaf_code(정렬 칸, 1)
EHD void lbvh_node(const uint64_t* k, int n, int i, int32_t& left, int32_t& right) {
  const int d = (lbvh_delta(k, n, i, i + 1) - lbvh_delta(k, n, i, i - 1)) >= 0 ? 1 : -1;
  const int dmin = lbvh_delta(k, n, i, i - d);
  int lmax = 2;
  while (lbvh_delta(k, n, i, i + lmax * d) > dmin) lmax <<= 1;
  int l = 0;
  for (int t = lmax >> 1; t >= 1; t >>= 1)
    if (lbvh_delta(k, n, i, i + (l + t) * d) > dmin) l += t;
  const int j = i + l * d;
  const int dnode = lbvh_delta(k, n, i, j);
  int s = 0;
  int t = l;
  do {
    t = (t + 1) >> 1;
    if (lbvh_delta(k, n, i, i + (s + t) * d) > dnode) s += t;
  } while (t > 1);
  const int gamma = i + s * d + (d < 0 ? d : 0);
  const int lo = i < j ? i : j, hi = i < j ? j : i;
  left = (lo == gamma) ? leaf_code(uint32_t(gamma), 1u) : int32_t(gamma);
  right = (hi == gamma + 1) ? leaf_code(uint32_t(gamma + 1), 1u) : int32_t(gamma + 1);
}

EHD void box_union(const float* a, const float* b, float* o) {
  o[0] = fmn(a[0], b[0]);
  o[1] = fmn(a[1], b[1]);
  o[2] = fmn(a[2], b[2]);
  o[3] = fmx(a[3], b[3]);
  o[4] = fmx(a[4], b[4]);
  o[5] = fmx(a[5], b[5]);
}
EHD void node_set_child(Node2& n, int which, const float* b) {
  float* lo = which == 0 ? n.lo0 : n.lo1;
  float* hi = which == 0 ? n.hi0 : n.hi1;
  lo[0] = b[0]; lo[1] = b[1]; lo[2] = b[2];
  hi[0] = b[3]; hi[1] = b[4]; hi[2] = b[5];
}

}  // namespace rnd
}  // namespace eng

// 층 1 호스트 빌더 (nvcc 장치 단계에서도 구문은 읽히지만 호출되지 않는다)
#include <algorithm>
#include <vector>
namespace eng {
namespace rnd {
// 층 1: 호스트에서 판 하나의 TLAS. keys/nodebox 는 작업 공간.
inline void tlas_build_host(const SceneView& S, const EnvView& E, std::vector<uint64_t>& keys,
                            std::vector<float>& nodebox, std::vector<int32_t>& kids) {
  const int n = S.n_inst;
  for (int i = 0; i < n; ++i) inst_prepare(S, E, i);
  for (int i = 0; i < S.n_lights; ++i) {
    const Light& L = S.lights[i];
    const_cast<Aff*>(E.light_world)[i] = L.anchor >= 0 ? aff_mul(E.anchor[L.anchor], L.rel) : L.rel;
  }
  if (n == 0) {
    E.tlas[0].c0 = E.tlas[0].c1 = kEmpty;
    return;
  }
  if (n == 1) {
    E.order[0] = 0;
    Node2& r = E.tlas[0];
    node_set_child(r, 0, E.inst_box);
    r.c0 = leaf_code(0, 1);
    r.c1 = kEmpty;
    return;
  }
  float clo[3] = {kInf, kInf, kInf}, chi[3] = {-kInf, -kInf, -kInf};
  for (int i = 0; i < n; ++i) {
    const float* b = E.inst_box + 6 * i;
    for (int a = 0; a < 3; ++a) {
      const float c = (b[a] + b[3 + a]) * 0.5f;
      clo[a] = fmn(clo[a], c);
      chi[a] = fmx(chi[a], c);
    }
  }
  float cinv[3];
  for (int a = 0; a < 3; ++a) cinv[a] = chi[a] > clo[a] ? 1.0f / (chi[a] - clo[a]) : 0.0f;
  keys.resize(n);
  for (int i = 0; i < n; ++i) keys[i] = inst_key(E.inst_box + 6 * i, clo, cinv, i);
  std::sort(keys.begin(), keys.end());
  for (int i = 0; i < n; ++i) E.order[i] = int32_t(uint32_t(keys[i]));
  kids.resize(2 * (n - 1));
  for (int i = 0; i < n - 1; ++i) lbvh_node(keys.data(), n, i, kids[2 * i], kids[2 * i + 1]);
  nodebox.resize(6 * (n - 1));
  // 상자: 뒤에서부터가 위상 순서가 아니므로 명시적 후위 순회
  std::vector<int32_t> st;
  std::vector<uint8_t> state(n - 1, 0);
  st.push_back(0);
  auto child_box = [&](int32_t c) -> const float* {
    return c >= 0 ? &nodebox[6 * c] : E.inst_box + 6 * E.order[leaf_first(c)];
  };
  while (!st.empty()) {
    const int32_t i = st.back();
    if (state[i] == 0) {
      state[i] = 1;
      if (kids[2 * i] >= 0) st.push_back(kids[2 * i]);
      if (kids[2 * i + 1] >= 0) st.push_back(kids[2 * i + 1]);
      continue;
    }
    st.pop_back();
    box_union(child_box(kids[2 * i]), child_box(kids[2 * i + 1]), &nodebox[6 * i]);
  }
  for (int i = 0; i < n - 1; ++i) {
    Node2& nd = E.tlas[i];
    nd.c0 = kids[2 * i];
    nd.c1 = kids[2 * i + 1];
    node_set_child(nd, 0, child_box(nd.c0));
    node_set_child(nd, 1, child_box(nd.c1));
    nd.pad0 = nd.pad1 = 0;
  }
}
}  // namespace rnd
}  // namespace eng
