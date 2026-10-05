#include "scenemap/scan.hpp"

#include "scenemap/fk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace scenemap {

static inline float dist2(float ax, float ay, float az, const float* b) {
  const float dx = ax - b[0], dy = ay - b[1], dz = az - b[2];
  return dx * dx + dy * dy + dz * dz;
}

// atan2 근사(최대 오차 약 1e-5 rad — 방위 칸 0.5° = 8.7e-3 rad 보다 훨씬 작음). 칸 경계에 걸친 점만 이웃 칸으로 갈 수 있음
static inline float fastAtan2(float y, float x) {
  const float ax = std::fabs(x), ay = std::fabs(y);
  const float mx = std::fmax(ax, ay), mn = std::fmin(ax, ay);
  if (mx == 0.f) return 0.f;
  const float a = mn / mx, s = a * a;
  // 7차 최소 최대 다항식(atan on [0,1])
  float r = ((((-0.0117212f * s + 0.05265332f) * s - 0.11643287f) * s + 0.19354346f) * s - 0.33262347f) * s + 0.99997726f;
  r *= a;
  if (ay > ax) r = 1.57079637f - r;
  if (x < 0) r = 3.14159274f - r;
  return y < 0 ? -r : r;
}

static inline float segDist2(float px, float py, float pz, const float* a, const float* b) {
  const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const float l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  float t = l2 > 1e-12f ? ((px - a[0]) * ab[0] + (py - a[1]) * ab[1] + (pz - a[2]) * ab[2]) / l2 : 0.f;
  t = std::fmin(std::fmax(t, 0.f), 1.f);
  const float c[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
  return dist2(px, py, pz, c);
}

BodyState bodyFromFk(const BodyFk& fk, const float eef[2][3], float arm_r, float hand_r) {
  BodyState b;
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 3; ++k) b.eef[s][k] = eef[s][k];
  for (int s = 0; s < fk.n_arms; ++s)
    for (int k = 0; k + 1 < BodyFk::kArmPts; ++k) {
      Capsule c;
      for (int i = 0; i < 3; ++i) { c.a[i] = fk.arm[s][k][i]; c.b[i] = fk.arm[s][k + 1][i]; }
      if (c.a[0] == c.b[0] && c.a[1] == c.b[1] && c.a[2] == c.b[2]) continue;
      c.r = k >= BodyFk::kArmPts - 3 ? hand_r : arm_r;
      b.caps.push_back(c);
    }
  return b;
}

void AttachFilter::update(const std::vector<int64_t>& cur, const Pose2& odom) {
  if (!p_.enabled) return;
  if (!have_) {
    score_.clear();
    for (int64_t k : cur) score_[k] = 0;
    check_ = odom;
    have_ = true;
    return;
  }
  const double dxy = std::hypot(odom.x - check_.x, odom.y - check_.y), dth = std::fabs(wrapAngle(odom.th - check_.th));
  if (dxy < p_.move_xy && dth < p_.move_yaw) return;
  auto unpack = [](int64_t k, int* i, int* j, int* l) {
    auto sx = [](int64_t v) { return int((v ^ 0x80000) - 0x80000); };   // 20비트 부호 확장
    *i = sx((k >> 40) & 0xfffff); *j = sx((k >> 20) & 0xfffff); *l = sx(k & 0xfffff);
  };
  std::unordered_map<int64_t, int> next;
  next.reserve(cur.size());
  for (int64_t k : cur) {
    int i, j, l;
    unpack(k, &i, &j, &l);
    int best = -1;
    for (int a = -1; a <= 1; ++a)
      for (int b = -1; b <= 1; ++b)
        for (int c = -1; c <= 1; ++c) {
          auto it = score_.find(key3(i + a, j + b, l + c));
          if (it != score_.end() && it->second > best) best = it->second;
        }
    next[k] = best + 1;   // 지난번 근처에 있었으면 +1, 없었으면 0
  }
  score_.swap(next);
  check_ = odom;
}

size_t AttachFilter::nAttached() const {
  size_t n = 0;
  for (const auto& kv : score_) n += kv.second >= p_.min_score;
  return n;
}

void makeScan(const DepthView& d, const BodyState& b, const ScanParams& p, Scan2* out, const AttachFilter* att,
              std::vector<int64_t>* vox_out) {
  ScanWork w;
  makeScan(d, b, p, out, att, vox_out, &w, 0.f);
}

void makeScan(const DepthView& d, const BodyState& b, const ScanParams& p, Scan2* out, const AttachFilter* att,
              std::vector<int64_t>* vox_out, ScanWork* W, float sig_cell) {
  const int gw = (d.w + d.step - 1) / d.step, gh = (d.h + d.step - 1) / d.step;
  const size_t ng = size_t(gw) * gh;
  // 1. 격자 화소마다 베이스 기준 점. 유효 = Zo > 0(광학 z), 아니면 P 는 쓰지 않음(채우기 없음)
  if (W->P.size() < ng * 3) W->P.resize(ng * 3);
  W->Zo.assign(ng, 0.f);
  float* P = W->P.data();
  float* Zo = W->Zo.data();
  const float* T = d.T_bc;
  const float ifx = 1.f / d.fx, ify = 1.f / d.fy;
  for (int j = 0; j < gh; ++j) {
    const int v = j * d.step;
    const uint16_t* row = d.m ? nullptr : d.mm + size_t(v) * d.w;
    const float* rowf = d.m ? d.m + size_t(v) * d.w : nullptr;
    const float yn = (v - d.cy) / d.fy;
    // 행마다 상수: Y 항
    const float ry0 = T[1] * yn + T[2], ry1 = T[5] * yn + T[6], ry2 = T[9] * yn + T[10];
    for (int i = 0; i < gw; ++i) {
      const int u = i * d.step;
      const float z = rowf ? rowf[u] : row[u] * 1e-3f;
      if (!(z >= p.zmin && z <= p.zmax)) continue;   // NaN·0·범위 밖
      const float xn = (u - d.cx) * ifx;
      float* q = &P[(size_t(j) * gw + i) * 3];
      q[0] = (T[0] * xn + ry0) * z + T[3];
      q[1] = (T[4] * xn + ry1) * z + T[7];
      q[2] = (T[8] * xn + ry2) * z + T[11];
      Zo[size_t(j) * gw + i] = z;
    }
  }
  (void)ify;
  // 2. 분류
  const int nb = p.bins;
  // 화소 방위 칸·수평 배율 캐시: 카메라 쪽 수평 성분 (a, b)·z = (점 − 카메라) 이라 방위는 깊이와 무관.
  // 머리 회전 행렬 원소가 2e-4 안(≈ 0.01°, 방위 칸 0.5° 의 2 %)으로 같으면 다시 씀
  {
    const float key[16] = {T[0], T[1], T[2], T[4], T[5], T[6], d.fx, d.fy, d.cx, d.cy, float(d.w), float(d.h), float(d.step), float(nb),
                           0, 0};
    bool same = W->pvalid && W->pbin.size() == ng;
    for (int k = 0; k < 14 && same; ++k) same = std::fabs(key[k] - W->pkey[k]) <= (k < 6 ? 2e-4f : 0.f);
    if (!same) {
      W->pbin.resize(ng);
      W->phs2.resize(ng);
      const float bsc = nb / (2.f * float(M_PI));
      for (int j = 0; j < gh; ++j) {
        const float yn = (j * d.step - d.cy) / d.fy;
        for (int i = 0; i < gw; ++i) {
          const float xn = (i * d.step - d.cx) * ifx;
          const float a = T[0] * xn + T[1] * yn + T[2], bq = T[4] * xn + T[5] * yn + T[6];
          int k = int((fastAtan2(bq, a) + float(M_PI)) * bsc);
          k = k < 0 ? 0 : (k >= nb ? nb - 1 : k);
          W->pbin[size_t(j) * gw + i] = int16_t(k);
          W->phs2[size_t(j) * gw + i] = a * a + bq * bq;
        }
      }
      std::copy(key, key + 16, W->pkey);
      W->pvalid = true;
      ++W->n_cache_miss;
    } else {
      ++W->n_cache_hit;
    }
  }
  const int16_t* pbin = W->pbin.data();
  const float* phs2 = W->phs2.data();
  W->hit_r.assign(nb, std::numeric_limits<float>::infinity());
  W->hx.resize(nb);
  W->hy.resize(nb);
  W->floor_r.assign(nb, 0.f);
  float* hit_r = W->hit_r.data();
  float* hx = W->hx.data();
  float* hy = W->hy.data();
  float* floor_r = W->floor_r.data();
  const float bin_scale = nb / (2.f * float(M_PI));
  float ab[2][3], ab2[2];
  for (int s = 0; s < 2; ++s) {
    for (int k = 0; k < 3; ++k) ab[s][k] = b.eef[s][k] - p.shoulder[s][k];
    ab2[s] = std::max(ab[s][0] * ab[s][0] + ab[s][1] * ab[s][1] + ab[s][2] * ab[s][2], 1e-9f);
  }
  const float self2 = p.self_r * p.self_r, eef2 = p.eef_r * p.eef_r, arm2 = p.arm_r * p.arm_r;
  // 캡슐 전체를 감싼 상자(밖의 점은 캡슐 검사를 건너뜀)
  float bb0[3] = {1e9f, 1e9f, 1e9f}, bb1[3] = {-1e9f, -1e9f, -1e9f};
  for (const Capsule& c : b.caps)
    for (int i = 0; i < 3; ++i) {
      bb0[i] = std::fmin(bb0[i], std::fmin(c.a[i], c.b[i]) - c.r);
      bb1[i] = std::fmax(bb1[i], std::fmax(c.a[i], c.b[i]) + c.r);
    }
  const size_t ncap = b.caps.size();
  float cbox[6 * 64];
  const size_t ncap_box = std::min<size_t>(ncap, 64);
  for (size_t ci = 0; ci < ncap_box; ++ci) {
    const Capsule& c = b.caps[ci];
    for (int i = 0; i < 3; ++i) {
      cbox[ci * 6 + i] = std::fmin(c.a[i], c.b[i]) - c.r;
      cbox[ci * 6 + 3 + i] = std::fmax(c.a[i], c.b[i]) + c.r;
    }
  }
  // 맞추기 칸 해시(세대 번호로 비움 — 지우기 없음)
  constexpr size_t kH = 1 << 14;
  if (W->hkey.size() != kH) {
    W->hkey.assign(kH, 0);
    W->hgen.assign(kH, 0);
    W->hidx.assign(kH, 0);
    W->gen = 0;
  }
  if (++W->gen == 0) {
    std::fill(W->hgen.begin(), W->hgen.end(), 0u);
    W->gen = 1;
  }
  const uint32_t gen = W->gen;
  W->acc.clear();
  const float inv_mc = 1.f / p.match_cell;
  const bool use_att = att != nullptr;
  const bool any_caps = !b.caps.empty();
  for (int j = 0; j < gh; ++j)
    for (int i = 0; i < gw; ++i) {
      const size_t c0 = size_t(j) * gw + i;
      if (Zo[c0] == 0.f) continue;
      const float* q = &P[c0 * 3];
      const float px = q[0], py = q[1], pz = q[2];
      const float r2 = px * px + py * py;
      if (r2 < self2) continue;
      bool self = false;
      if (px >= bb0[0] && px <= bb1[0] && py >= bb0[1] && py <= bb1[1] && pz >= bb0[2] && pz <= bb1[2])
        for (size_t ci = 0; ci < ncap_box; ++ci) {   // 캡슐마다 상자로 먼저 거름
          const float* q = &cbox[ci * 6];
          if (px < q[0] || px > q[3] || py < q[1] || py > q[4] || pz < q[2] || pz > q[5]) continue;
          const Capsule& c = b.caps[ci];
          if (segDist2(px, py, pz, c.a, c.b) < c.r * c.r) { self = true; break; }
        }
      for (int s = 0; s < 2 && !self; ++s) {
        if (any_caps) {
          if (dist2(px, py, pz, b.eef[s]) < eef2) self = true;
          continue;
        }
        if (dist2(px, py, pz, b.eef[s]) < eef2) { self = true; break; }
        const float* a = p.shoulder[s];
        float t = ((px - a[0]) * ab[s][0] + (py - a[1]) * ab[s][1] + (pz - a[2]) * ab[s][2]) / ab2[s];
        t = std::fmin(std::fmax(t, 0.f), 1.f);
        const float c[3] = {a[0] + t * ab[s][0], a[1] + t * ab[s][1], a[2] + t * ab[s][2]};
        if (dist2(px, py, pz, c) < arm2) self = true;
      }
      if (self) continue;
      if (use_att && pz >= p.band_lo && att->near(px, py)) {
        if (vox_out) vox_out->push_back(att->key(px, py, pz));
        if (att->attached(px, py, pz)) continue;
      }
      // 레이저 한 줄(띠 안 가장 가까운 것) · 바닥 빈칸
      if (pz <= p.band_hi) {
        const int k = pbin[c0];
        const float zz = Zo[c0];
        const float rr2 = zz * zz * phs2[c0];   // 칸마다 거리²로 비교, 끝에서 한 번 sqrt
        if (pz < p.band_lo) {
          if (rr2 > floor_r[k]) floor_r[k] = rr2;
        } else if (rr2 < hit_r[k]) {
          hit_r[k] = rr2; hx[k] = px; hy[k] = py;
        }
      }
      // 맞추기 점: 수직면 쪽
      if (!p.dense || pz < p.band_lo || pz > p.match_hi || i == 0 || j == 0 || i == gw - 1 || j == gh - 1) continue;
      if (Zo[c0 - 1] == 0.f || Zo[c0 + 1] == 0.f || Zo[c0 - gw] == 0.f || Zo[c0 + gw] == 0.f) continue;
      const float z = Zo[c0];
      if (std::fabs(Zo[c0 + 1] - Zo[c0 - 1]) >= 0.1f * z || std::fabs(Zo[c0 + gw] - Zo[c0 - gw]) >= 0.1f * z) continue;
      const float* l = &P[(c0 - 1) * 3];
      const float* rr = &P[(c0 + 1) * 3];
      const float* up = &P[(c0 - gw) * 3];
      const float* dn = &P[(c0 + gw) * 3];
      const float ax = rr[0] - l[0], ay = rr[1] - l[1], az = rr[2] - l[2];
      const float bx = dn[0] - up[0], by = dn[1] - up[1], bz = dn[2] - up[2];
      const float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const float nn2 = nx * nx + ny * ny + nz * nz;
      const float vz = p.vert_nz;
      if (nn2 <= 0 || nz * nz >= vz * vz * nn2) continue;
      const int64_t key = (int64_t(std::floor(px * inv_mc)) << 32) ^ (int64_t(std::floor(py * inv_mc)) & 0xffffffff);
      size_t h = size_t((uint64_t(key) * 0x9E3779B97F4A7C15ull) >> 50) & (kH - 1);
      uint32_t ai = UINT32_MAX;
      for (;;) {
        if (W->hgen[h] != gen) {
          if (W->acc.size() >= kH * 3 / 4) break;   // 가득(2.5 cm 칸 1.2 만 개 넘음 — 실제로는 수천): 새 칸은 버림
          W->hgen[h] = gen;
          W->hkey[h] = key;
          ai = uint32_t(W->acc.size());
          W->hidx[h] = ai;
          W->acc.push_back(ScanWork::Acc{0, 0, 0, 0, 0});
          break;
        }
        if (W->hkey[h] == key) { ai = W->hidx[h]; break; }
        h = (h + 1) & (kH - 1);
      }
      if (ai == UINT32_MAX) continue;
      ScanWork::Acc& a = W->acc[ai];
      const float hn = std::sqrt(nx * nx + ny * ny);
      float ux = nx / hn, uy = ny / hn;
      if (ux * (T[3] - px) + uy * (T[7] - py) < 0) { ux = -ux; uy = -uy; }   // 카메라 쪽
      a.x += px; a.y += py; a.nx += ux; a.ny += uy; ++a.n;
    }
  if (vox_out) {
    std::sort(vox_out->begin(), vox_out->end());
    vox_out->erase(std::unique(vox_out->begin(), vox_out->end()), vox_out->end());
  }
  out->ox = T[3];
  out->oy = T[7];
  out->hx.clear(); out->hy.clear(); out->fx.clear(); out->fy.clear(); out->mx.clear(); out->my.clear(); out->mnx.clear(); out->mny.clear();
  const bool want_sig = sig_cell > 0.f;
  if (want_sig) out->sig.assign(nb, 0);
  const float isc = want_sig ? 1.f / sig_cell : 0.f;
  for (int k = 0; k < nb; ++k) {
    if (std::isfinite(hit_r[k])) hit_r[k] = std::sqrt(hit_r[k]);
    if (floor_r[k] > 0) floor_r[k] = std::sqrt(floor_r[k]);
    if (std::isfinite(hit_r[k])) {
      out->hx.push_back(hx[k]);
      out->hy.push_back(hy[k]);
      if (want_sig) out->sig[k] = int16_t(std::min(32000.f, hit_r[k] * isc + 1.f));
    } else if (floor_r[k] > 0) {
      const float a = (k + 0.5f) / bin_scale - float(M_PI);
      out->fx.push_back(T[3] + floor_r[k] * std::cos(a));
      out->fy.push_back(T[7] + floor_r[k] * std::sin(a));
      if (want_sig) out->sig[k] = int16_t(-std::min(32000.f, floor_r[k] * isc + 1.f));
    }
  }
  const size_t na = W->acc.size();
  out->mx.resize(na); out->my.resize(na); out->mnx.resize(na); out->mny.resize(na);
  for (size_t k = 0; k < na; ++k) {
    const ScanWork::Acc& a = W->acc[k];
    out->mx[k] = a.x / a.n;
    out->my[k] = a.y / a.n;
    const float nn = std::sqrt(a.nx * a.nx + a.ny * a.ny);
    out->mnx[k] = nn > 1e-6f ? a.nx / nn : 0.f;
    out->mny[k] = nn > 1e-6f ? a.ny / nn : 0.f;
  }
}

}  // namespace scenemap
