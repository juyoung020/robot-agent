// 살펴본 정도(inspect.hpp 머리말이 정의)
#include "scenemap/inspect.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace scenemap {

namespace {

constexpr int kG = 4;   // 윗면 칸 4 × 4

void boxOf(const double lo[3], const double hi[3], float b[4]) {
  b[0] = float(lo[0]); b[1] = float(lo[1]); b[2] = float(hi[0]); b[3] = float(hi[1]);
}

bool sameBox(const float a[4], const float b[4]) {
  for (int k = 0; k < 4; ++k)
    if (std::fabs(a[k] - b[k]) > 0.01f) return false;
  return true;
}

// 옛 상자 ob 의 칸 비트 → 새 상자 nb 의 칸(새 칸 가운데가 든 옛 칸)
uint16_t remapBits(uint16_t bits, const float ob[4], const float nb[4]) {
  if (!bits) return 0;
  const float ow = ob[2] - ob[0], oh = ob[3] - ob[1];
  if (!(ow > 1e-6f && oh > 1e-6f)) return 0;
  uint16_t out = 0;
  for (int j = 0; j < kG; ++j)
    for (int i = 0; i < kG; ++i) {
      const float x = nb[0] + (i + 0.5f) * (nb[2] - nb[0]) / kG, y = nb[1] + (j + 0.5f) * (nb[3] - nb[1]) / kG;
      if (x < ob[0] || x > ob[2] || y < ob[1] || y > ob[3]) continue;
      const int oi = std::clamp(int(std::floor((x - ob[0]) / ow * kG)), 0, kG - 1);
      const int oj = std::clamp(int(std::floor((y - ob[1]) / oh * kG)), 0, kG - 1);
      if ((bits >> (kG * oj + oi)) & 1u) out = uint16_t(out | (1u << (kG * j + i)));
    }
  return out;
}

bool sameView(const float* a, const double* b, double d2max, double cosmin) {
  const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz <= d2max && a[3] * b[3] + a[4] * b[4] + a[5] * b[5] >= cosmin;
}

void addView(InspectState& s, const double v[6], const InspectParams& p) {
  if (s.nViews() >= p.view_cap) return;
  const double d2 = p.view_d * p.view_d, cs = std::cos(p.view_deg * M_PI / 180.0);
  for (size_t k = 0; k + 6 <= s.views.size(); k += 6)
    if (sameView(s.views.data() + k, v, d2, cs)) return;
  for (int k = 0; k < 6; ++k) s.views.push_back(float(v[k]));
}

}  // namespace

bool inspHasTop(const double lo[3], const double hi[3], const InspectParams& p) {
  return hi[0] - lo[0] >= p.top_min_side && hi[1] - lo[1] >= p.top_min_side && hi[2] >= p.top_zmin && hi[2] <= p.top_zmax;
}

void inspObserve(InspectState& s, const double cam6[6], double dist, const InspectParams& p) {
  if (dist >= 0 && (s.closest < 0 || dist < s.closest)) s.closest = float(dist);
  double n = std::sqrt(cam6[3] * cam6[3] + cam6[4] * cam6[4] + cam6[5] * cam6[5]);
  if (n < 1e-9) n = 1;
  const double v[6] = {cam6[0], cam6[1], cam6[2], cam6[3] / n, cam6[4] / n, cam6[5] / n};
  addView(s, v, p);
}

void inspAlign(InspectState& s, const double lo[3], const double hi[3]) {
  float nb[4];
  boxOf(lo, hi, nb);
  if (!s.top_set) {
    std::copy(nb, nb + 4, s.top_box);
    s.top_bits = 0;
    s.top_set = true;
    return;
  }
  if (sameBox(s.top_box, nb)) return;
  s.top_bits = remapBits(s.top_bits, s.top_box, nb);
  std::copy(nb, nb + 4, s.top_box);
}

void inspTop(InspectState& s, const double lo[3], const double hi[3], const InspectFrame& f, const InspectParams& p) {
  inspAlign(s, lo, hi);
  if (s.top_bits == 0xFFFFu || !f.T_mc || (!f.depth_m && !f.depth_mm)) return;
  const double* T = f.T_mc;
  const double cx0 = T[3], cy0 = T[7], cz0 = T[11];
  const double zt = hi[2] + p.top_probe_h;
  {   // 윗면 사각형까지 가장 가까운 거리가 한도 밖이면 볼 것 없음
    const double dx = std::max({lo[0] - cx0, 0.0, cx0 - hi[0]}), dy = std::max({lo[1] - cy0, 0.0, cy0 - hi[1]}), dz = zt - cz0;
    if (dx * dx + dy * dy + dz * dz > p.top_range * p.top_range) return;
  }
  const double cinc = std::cos(p.top_inc_deg * M_PI / 180.0);
  const double m = p.top_margin_px;
  for (int j = 0; j < kG; ++j)
    for (int i = 0; i < kG; ++i) {
      const int bitn = kG * j + i;
      if ((s.top_bits >> bitn) & 1u) continue;
      const double x = lo[0] + (i + 0.5) * (hi[0] - lo[0]) / kG, y = lo[1] + (j + 0.5) * (hi[1] - lo[1]) / kG;
      const double vx = x - cx0, vy = y - cy0, vz = zt - cz0;
      const double r = std::sqrt(vx * vx + vy * vy + vz * vz);
      if (r > p.top_range || r < 1e-6 || std::fabs(vz) < cinc * r) continue;
      const double xc = T[0] * vx + T[4] * vy + T[8] * vz, yc = T[1] * vx + T[5] * vy + T[9] * vz, zc = T[2] * vx + T[6] * vy + T[10] * vz;
      if (zc <= f.zmin) continue;
      const double u = f.fx * xc / zc + f.cx, v = f.fy * yc / zc + f.cy;
      if (u < m || v < m || u >= f.w - m || v >= f.h - m) continue;
      const size_t px = size_t(int(v)) * size_t(f.w) + size_t(int(u));
      const double d = f.depth_m ? double(f.depth_m[px]) : f.depth_mm[px] * 1e-3;
      if (!(d > f.zmin)) continue;   // 0·NaN = 깊이 없음(모름). zmax 넘는 값은 점 너머(봄)
      if (d >= zc - (p.top_tol0 + p.top_tol_k * zc)) s.top_bits = uint16_t(s.top_bits | (1u << bitn));
    }
}

void inspMerge(InspectState& keep, const InspectState& drop, const double lo[3], const double hi[3], const InspectParams& p) {
  if (drop.closest >= 0 && (keep.closest < 0 || drop.closest < keep.closest)) keep.closest = drop.closest;
  for (size_t k = 0; k + 6 <= drop.views.size(); k += 6) {
    const double v[6] = {drop.views[k], drop.views[k + 1], drop.views[k + 2], drop.views[k + 3], drop.views[k + 4], drop.views[k + 5]};
    addView(keep, v, p);
  }
  if (!keep.top_set && !drop.top_set) return;
  float nb[4];
  boxOf(lo, hi, nb);
  uint16_t b = 0;
  if (keep.top_set) b = sameBox(keep.top_box, nb) ? keep.top_bits : remapBits(keep.top_bits, keep.top_box, nb);
  if (drop.top_set) b = uint16_t(b | (sameBox(drop.top_box, nb) ? drop.top_bits : remapBits(drop.top_bits, drop.top_box, nb)));
  keep.top_bits = b;
  std::copy(nb, nb + 4, keep.top_box);
  keep.top_set = true;
}

float inspTopSeen(const InspectState& s, const double lo[3], const double hi[3], const InspectParams& p) {
  if (!inspHasTop(lo, hi, p)) return -1.f;
  if (!s.top_set) return 0.f;
  float nb[4];
  boxOf(lo, hi, nb);
  const uint16_t b = sameBox(s.top_box, nb) ? s.top_bits : remapBits(s.top_bits, s.top_box, nb);
  return float(std::popcount(unsigned(b))) / float(kG * kG);
}

}  // namespace scenemap
