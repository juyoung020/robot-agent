// BLAS 굽기 (호스트 전용, 한 번): 기하 하나의 삼각형 -> 이분 BVH (구간 나눈 SAH, 세 축 × 16 칸), 잎 최대 8 삼각형.
// 결과(노드·재배열 순서)는 자료로 GPU 에 올라가 두 층이 같은 배열을 읽는다 -> 굽기 자체는 GPU 로 옮길 필요 없음.
#pragma once
#include <cstdint>
#include <vector>

#include "core/render/rt.h"

namespace eng {
namespace rnd {

struct BlasOut {
  std::vector<Node2> nodes;
  std::vector<uint32_t> order;  // 굽힌 칸 -> 원래 삼각형 번호
  float lo[3], hi[3];
};

inline BlasOut blas_build(const float* v /* ntri*9 */, uint32_t ntri, uint32_t max_leaf = 4) {
  struct Ref { float lo[3], hi[3], c[3]; };
  std::vector<Ref> refs(ntri);
  BlasOut out;
  for (int a = 0; a < 3; ++a) { out.lo[a] = kInf; out.hi[a] = -kInf; }
  for (uint32_t i = 0; i < ntri; ++i) {
    Ref& r = refs[i];
    for (int a = 0; a < 3; ++a) {
      const float x0 = v[9 * i + a], x1 = v[9 * i + 3 + a], x2 = v[9 * i + 6 + a];
      r.lo[a] = fmn(fmn(x0, x1), x2);
      r.hi[a] = fmx(fmx(x0, x1), x2);
      r.c[a] = (r.lo[a] + r.hi[a]) * 0.5f;
      out.lo[a] = fmn(out.lo[a], r.lo[a]);
      out.hi[a] = fmx(out.hi[a], r.hi[a]);
    }
  }
  out.order.resize(ntri);
  for (uint32_t i = 0; i < ntri; ++i) out.order[i] = i;
  if (ntri == 0) {
    Node2 n{};
    n.c0 = n.c1 = kEmpty;
    out.nodes.push_back(n);
    return out;
  }
  // 임시 트리 (깊이 우선, 명시적 스택). 노드 = 구간 [b, e)
  struct Tmp { float lo[3], hi[3]; uint32_t b, e; int32_t l, r; };
  std::vector<Tmp> tmp;
  tmp.reserve(2 * ntri / max_leaf + 2);
  auto bounds = [&](uint32_t b, uint32_t e, Tmp& t) {
    for (int a = 0; a < 3; ++a) { t.lo[a] = kInf; t.hi[a] = -kInf; }
    for (uint32_t k = b; k < e; ++k) {
      const Ref& r = refs[out.order[k]];
      for (int a = 0; a < 3; ++a) { t.lo[a] = fmn(t.lo[a], r.lo[a]); t.hi[a] = fmx(t.hi[a], r.hi[a]); }
    }
  };
  auto area = [](const float* lo, const float* hi) {
    const float dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
    return dx < 0.0f ? 0.0f : dx * dy + dy * dz + dz * dx;
  };
  Tmp root{};
  root.b = 0; root.e = ntri; root.l = root.r = -1;
  bounds(0, ntri, root);
  tmp.push_back(root);
  std::vector<std::pair<uint32_t, int>> work{{0u, 0}};
  constexpr int NB = 16;
  while (!work.empty()) {
    const uint32_t ni = work.back().first;
    const int depth = work.back().second;
    work.pop_back();
    const uint32_t b = tmp[ni].b, e = tmp[ni].e, cnt = e - b;
    if (cnt <= max_leaf) continue;
    float clo[3] = {kInf, kInf, kInf}, chi[3] = {-kInf, -kInf, -kInf};
    for (uint32_t k = b; k < e; ++k)
      for (int a = 0; a < 3; ++a) { clo[a] = fmn(clo[a], refs[out.order[k]].c[a]); chi[a] = fmx(chi[a], refs[out.order[k]].c[a]); }
    int best_axis = -1, best_split = -1;
    float best_cost = kInf;
    for (int a = 0; a < 3; ++a) {
      const float ext = chi[a] - clo[a];
      if (!(ext > 0.0f)) continue;
      float blo[NB][3], bhi[NB][3];
      uint32_t bc[NB] = {0};
      for (int q = 0; q < NB; ++q) for (int c = 0; c < 3; ++c) { blo[q][c] = kInf; bhi[q][c] = -kInf; }
      const float sc = float(NB) / ext;
      for (uint32_t k = b; k < e; ++k) {
        const Ref& r = refs[out.order[k]];
        int q = int((r.c[a] - clo[a]) * sc);
        q = q < 0 ? 0 : (q >= NB ? NB - 1 : q);
        bc[q]++;
        for (int c = 0; c < 3; ++c) { blo[q][c] = fmn(blo[q][c], r.lo[c]); bhi[q][c] = fmx(bhi[q][c], r.hi[c]); }
      }
      float rlo[NB][3], rhi[NB][3];
      uint32_t rc[NB];
      float alo[3] = {kInf, kInf, kInf}, ahi[3] = {-kInf, -kInf, -kInf};
      uint32_t ac = 0;
      for (int q = NB - 1; q >= 1; --q) {
        for (int c = 0; c < 3; ++c) { alo[c] = fmn(alo[c], blo[q][c]); ahi[c] = fmx(ahi[c], bhi[q][c]); }
        ac += bc[q];
        for (int c = 0; c < 3; ++c) { rlo[q][c] = alo[c]; rhi[q][c] = ahi[c]; }
        rc[q] = ac;
      }
      float llo[3] = {kInf, kInf, kInf}, lhi[3] = {-kInf, -kInf, -kInf};
      uint32_t lc = 0;
      for (int q = 0; q < NB - 1; ++q) {
        for (int c = 0; c < 3; ++c) { llo[c] = fmn(llo[c], blo[q][c]); lhi[c] = fmx(lhi[c], bhi[q][c]); }
        lc += bc[q];
        if (lc == 0 || rc[q + 1] == 0) continue;
        const float cost = area(llo, lhi) * float(lc) + area(rlo[q + 1], rhi[q + 1]) * float(rc[q + 1]);
        if (cost < best_cost) { best_cost = cost; best_axis = a; best_split = q; }
      }
    }
    uint32_t mid;
    const float leaf_cost = area(tmp[ni].lo, tmp[ni].hi) * float(cnt);
    if (best_axis >= 0) {
      if (cnt <= 8 && best_cost + area(tmp[ni].lo, tmp[ni].hi) * 0.5f >= leaf_cost) continue;  // 잎이 더 쌈
      const float ext = chi[best_axis] - clo[best_axis];
      const float sc = float(NB) / ext;
      uint32_t i = b, j = e;
      while (i < j) {
        const Ref& r = refs[out.order[i]];
        int q = int((r.c[best_axis] - clo[best_axis]) * sc);
        q = q < 0 ? 0 : (q >= NB ? NB - 1 : q);
        if (q <= best_split) ++i;
        else std::swap(out.order[i], out.order[--j]);
      }
      mid = i;
      if (mid == b || mid == e) mid = b + cnt / 2;
    } else {
      if (cnt <= 8) continue;  // 중심이 모두 같다: 8 개 이하면 잎
      mid = b + cnt / 2;
    }
    if (depth >= kBlasMaxDepth - 4) {  // 순회 스택을 넘지 않게: 깊이 한계에서는 반으로만 (잎 8 제한은 지킨다)
      mid = b + cnt / 2;
    }
    Tmp L{}, R{};
    L.b = b; L.e = mid; L.l = L.r = -1;
    R.b = mid; R.e = e; R.l = R.r = -1;
    bounds(L.b, L.e, L);
    bounds(R.b, R.e, R);
    tmp[ni].l = int32_t(tmp.size());
    tmp.push_back(L);
    tmp[ni].r = int32_t(tmp.size());
    tmp.push_back(R);
    work.push_back({uint32_t(tmp[ni].r), depth + 1});
    work.push_back({uint32_t(tmp[ni].l), depth + 1});
  }
  // Node2 로: 내부 임시 노드마다 하나. 번호는 깊이 우선(왼쪽 먼저) 순서.
  std::vector<int32_t> nid(tmp.size(), -1);
  std::vector<uint32_t> st{0};
  int32_t next = 0;
  std::vector<uint32_t> seq;
  while (!st.empty()) {
    const uint32_t i = st.back();
    st.pop_back();
    if (tmp[i].l < 0) continue;
    nid[i] = next++;
    seq.push_back(i);
    st.push_back(uint32_t(tmp[i].r));
    st.push_back(uint32_t(tmp[i].l));
  }
  auto code = [&](int32_t c) -> int32_t {
    const Tmp& t = tmp[c];
    if (t.l >= 0) return nid[c];
    return leaf_code(t.b, t.e - t.b);
  };
  if (seq.empty()) {  // 뿌리가 잎
    Node2 n{};
    for (int a = 0; a < 3; ++a) { n.lo0[a] = tmp[0].lo[a]; n.hi0[a] = tmp[0].hi[a]; }
    n.c0 = leaf_code(0, ntri);
    n.c1 = kEmpty;
    out.nodes.push_back(n);
    return out;
  }
  out.nodes.resize(seq.size());
  for (uint32_t i : seq) {
    Node2& n = out.nodes[nid[i]];
    n = Node2{};
    const Tmp& L = tmp[tmp[i].l];
    const Tmp& R = tmp[tmp[i].r];
    for (int a = 0; a < 3; ++a) { n.lo0[a] = L.lo[a]; n.hi0[a] = L.hi[a]; n.lo1[a] = R.lo[a]; n.hi1[a] = R.hi[a]; }
    n.c0 = code(tmp[i].l);
    n.c1 = code(tmp[i].r);
  }
  return out;
}

}  // namespace rnd
}  // namespace eng
