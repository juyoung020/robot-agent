#include "scenemap/objmap.hpp"

#include "da/merge.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <tuple>

namespace scenemap {
namespace {

inline bool maskBit(const sm_detections* d, int k, int i, int j) {
  if (i < 0 || j < 0 || i >= d->mask_w || j >= d->mask_h) return false;
  const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
  const size_t c = size_t(j) * d->mask_w + i;
  return (d->mask_bits[k * words + (c >> 5)] >> (c & 31)) & 1u;
}

double pct(std::vector<double>& v, double q) {
  const size_t k = std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// 10·50·90 백분위 한 번에(pct 세 번과 같은 값): 중앙값으로 나눈 뒤 아래쪽·위쪽에서만 찾음 — 3n → 약 2n
void pct3(std::vector<double>& v, double* lo, double* med, double* hi) {
  const size_t n = v.size(), last = n - 1;
  auto at = [&](double q) { return std::min(last, size_t(q * last + 0.5)); };
  const size_t k5 = at(0.5), k1 = at(0.1), k9 = at(0.9);
  std::nth_element(v.begin(), v.begin() + k5, v.end());
  *med = v[k5];
  if (k1 < k5) std::nth_element(v.begin(), v.begin() + k1, v.begin() + k5);
  *lo = v[k1];
  if (k9 > k5) std::nth_element(v.begin() + k5 + 1, v.begin() + k9, v.end());
  *hi = v[k9];
}

// 백분위에 쓰는 점 수 한도: 넘으면 고른 간격으로 골라 out 에(아니면 그대로 복사)
constexpr size_t kPctMax = 2048;
void subsample(const std::vector<double>& v, std::vector<double>* out) {
  const size_t n = v.size();
  if (n <= kPctMax) { out->assign(v.begin(), v.end()); return; }
  out->resize(kPctMax);
  for (size_t i = 0; i < kPctMax; ++i) (*out)[i] = v[(i * n) / kPctMax];
}

double d2(const double* a, const double* b) {
  return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

double dist3(const double* a, const double* b) {
  return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

struct Obs {
  std::vector<float> cxyz;        // 구름 후보(관측 안 복셀마다 하나)
  std::vector<int32_t> cpx;
  int det;                        // 검출 번호
  int sk;                         // 훑은 화소 간격
  double zmed;                    // 카메라 깊이 중앙값
  int cls;
  float score;
  double pos[3], ext[3], lo[3], hi[3];
  int n;
};

}  // namespace

void ObjectMap::event(double t, const MapObject& o, int kind) {
  ev_.push_back(ObjEvent{t, o.id, kind, {o.pos[0], o.pos[1], o.pos[2]}});
}

void ObjectMap::updateHands(double t, const double eef[2][3], const float grip[2], double yaw) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  for (int h = 0; h < 2; ++h) {
    const bool closed = grip[h] < p_.grip_closed;
    if (closed && !closed_[h]) {   // 잡기
      MapObject* best = nullptr;
      double bd = p_.grasp_r;
      for (auto& o : objs_) {
        if (!o.confirmed || o.held_by >= 0 || o.state == SM_GONE) continue;
        const double d = dist3(o.pos, eef[h]);
        if (d < bd) { bd = d; best = &o; }
      }
      if (best) {
        best->held_by = h;
        best->parent = 0;
        const double d[3] = {best->pos[0] - eef[h][0], best->pos[1] - eef[h][1], best->pos[2] - eef[h][2]};
        best->held_rel[0] = cy * d[0] + sy * d[1];
        best->held_rel[1] = -sy * d[0] + cy * d[1];
        best->held_rel[2] = d[2];
        for (int k = 0; k < 3; ++k) best->grasp_pos[k] = best->pos[k];
        best->state = SM_HELD;
        event(t, *best, 4);
      }
    } else if (!closed && closed_[h]) {   // 놓기
      for (auto& o : objs_) {
        if (o.held_by != h) continue;
        o.held_by = -1;
        const bool mv = dist3(o.pos, o.grasp_pos) > p_.moved_d;
        o.moved = o.moved || mv;
        o.state = o.moved ? SM_MOVED : SM_SEEN;
        o.misses = 0;
        // 놓은 자리가 다른 물체 위·안이면 붙이기(그 물체와 함께 움직임)
        const MapObject* best = nullptr;
        double bv = 1e18;
        for (const auto& p : objs_) {
          if (&p == &o || !p.confirmed || p.state == SM_GONE) continue;
          if (o.pos[0] < p.lo[0] - 0.1 || o.pos[0] > p.hi[0] + 0.1 || o.pos[1] < p.lo[1] - 0.1 || o.pos[1] > p.hi[1] + 0.1) continue;
          if (o.pos[2] < p.lo[2] - 0.05) continue;                  // 받침은 놓은 점 아래에서 시작
          const double v = o.pos[2] - std::min(p.hi[2], o.pos[2]);   // 떨어질 높이: 가장 높은 받침(바로 아래)
          if (v < bv) { bv = v; best = &p; }
        }
        if (best) {
          o.parent = best->id;
          for (int k = 0; k < 3; ++k) o.parent_rel[k] = o.pos[k] - best->pos[k];
        }
        event(t, o, 5);
      }
    }
    closed_[h] = closed;
  }
  for (auto& o : objs_)
    if (o.held_by >= 0)
      for (int k = 0; k < 3; ++k) {
        const double* r = o.held_rel;
        const double rel = k == 0 ? cy * r[0] - sy * r[1] : (k == 1 ? sy * r[0] + cy * r[1] : r[2]);
        const double np = eef[o.held_by][k] + rel, dk = np - o.pos[k];
        o.pos[k] = np;
        o.lo[k] += dk;
        o.hi[k] += dk;
        double d3[3] = {0, 0, 0};
        d3[k] = dk;
        o.cloud.translate(d3, t);   // 구름도 손을 따라감(평행 이동만)
      }
  // 붙은 물체는 받침을 따라간다(받침이 움직였으면 옮겨짐)
  for (auto& o : objs_) {
    if (!o.parent) continue;
    const MapObject* p = nullptr;
    for (const auto& q : objs_)
      if (q.id == o.parent) { p = &q; break; }
    if (!p) { o.parent = 0; continue; }
    for (int k = 0; k < 3; ++k) {
      const double np = p->pos[k] + o.parent_rel[k], dk = np - o.pos[k];
      if (std::fabs(dk) > 1e-9) o.moved = true;
      o.pos[k] = np;
      o.lo[k] += dk;
      o.hi[k] += dk;
      double d3[3] = {0, 0, 0};
      d3[k] = dk;
      o.cloud.translate(d3, t);
    }
    if (o.moved && o.state == SM_SEEN) o.state = SM_MOVED;
  }
}

void ObjectMap::addPoints(uint32_t id, const float* xyz, const uint8_t* rgb, int n, double stamp) {
  for (auto& m : objs_)
    if (m.id == id) {
      m.cloud.add(xyz, rgb, n, p_.voxel, p_.cloud_cap, stamp);
      return;
    }
}

void ObjectMap::update(const ObjFrame& f) {
  updateHands(f.stamp, f.eef, f.grip, f.base_yaw);
  const double* T = f.T_mc;
  auto depthAt = [&](int u, int v) -> float {
    if (u < 0 || v < 0 || u >= f.w || v >= f.h) return 0.f;
    const size_t i = size_t(v) * f.w + u;
    return f.depth_m ? f.depth_m[i] : f.depth_mm[i] * 1e-3f;
  };
  // 1. 검출 → 관측
  std::vector<Obs> obs;
  const sm_detections* D = f.dets;
  assoc_.assign(D && D->n > 0 ? D->n : 0, DetAssoc{});
  points_.clear();
  const float inv_vox = float(1.0 / std::max(1e-3, p_.voxel));
  VoxelIndex& seen = wseen_;
  const int st = std::max(1, p_.step);
  if (D && D->n > 0) {
    const float sxu = D->img_w > 0 ? float(f.w) / D->img_w : 1.f;   // 검출 영상 화소 ↔ 깊이 화소(크기가 다르면)
    const float syv = D->img_h > 0 ? float(f.h) / D->img_h : 1.f;
    std::vector<double>& X = wx_;
    std::vector<double>& Y = wy_;
    std::vector<double>& Z = wz_;
    std::vector<double>& ZC = wzc_;
    std::vector<double>& ZS = wzs_;
    std::vector<int32_t>& PU = wpu_;
    std::vector<int32_t>& PV = wpv_;
    const size_t words = (size_t(D->mask_w) * D->mask_h + 31) / 32;
    for (int k = 0; k < D->n; ++k) {
      if (kindOf(D->cls[k]) == kKindStructure) continue;   // 벽·바닥·문 등: 격자만(물체 아님)
      X.clear(); Y.clear(); Z.clear(); ZC.clear(); PU.clear(); PV.clear();
      // 상자 안만 훑는다(검출 영상 화소 → 깊이 화소)
      const float* b = D->box + 4 * k;
      const int u0 = std::max(0, int(b[0] * sxu) - 1), u1 = std::min(f.w - 1, int(b[2] * sxu) + 1);
      const int v0 = std::max(0, int(b[1] * syv) - 1), v1 = std::min(f.h - 1, int(b[3] * syv) + 1);
      const double box_px = double(std::max(0, u1 - u0 + 1)) * std::max(0, v1 - v0 + 1);
      const int sk = std::max(st, int(std::ceil(std::sqrt(box_px / std::max(1, p_.max_pts)))));
      // 열마다 마스크 칸 i·검출 화소 x 를 한 번만(안쪽 고리에 나눗셈 없음)
      wcol_.clear();
      for (int u = u0; u <= u1; u += sk) {
        const float xi = (u + 0.5f) / sxu;
        wcol_.push_back(int(std::floor((xi - D->mask_ox) / D->mask_sx)));
        wcol_.push_back(int32_t(xi));
      }
      const uint32_t* bits = D->mask_bits + size_t(k) * words;
      const int MW = D->mask_w, MH = D->mask_h;
      auto bit = [&](int i, int j) -> bool {
        if (i < 0 || j < 0 || i >= MW || j >= MH) return false;
        const size_t c = size_t(j) * MW + i;
        return (bits[c >> 5] >> (c & 31)) & 1u;
      };
      for (int v = v0; v <= v1; v += sk) {
        const float yi = (v + 0.5f) / syv;
        const int j = int(std::floor((yi - D->mask_oy) / D->mask_sy));
        if (j < 1 || j >= MH - 1) continue;
        const float* drow = f.depth_m ? f.depth_m + size_t(v) * f.w : nullptr;
        const uint16_t* drow16 = f.depth_m ? nullptr : f.depth_mm + size_t(v) * f.w;
        int ci = 0;
        for (int u = u0; u <= u1; u += sk, ci += 2) {
          // 깊이 화소 중심 → 검출 영상 화소 → 마스크 칸, 1 칸 깎기(네 이웃도 마스크)
          const int i = wcol_[ci];
          if (!bit(i, j) || !bit(i - 1, j) || !bit(i + 1, j) || !bit(i, j - 1) || !bit(i, j + 1)) continue;
          const float z = drow ? drow[u] : drow16[u] * 1e-3f;
          if (!(z > p_.zmin && z < p_.zmax)) continue;
          const double xc = (u - f.cx) / f.fx * z, yc = (v - f.cy) / f.fy * z;
          const double px = T[0] * xc + T[1] * yc + T[2] * z + T[3];
          const double py = T[4] * xc + T[5] * yc + T[6] * z + T[7];
          const double pz = T[8] * xc + T[9] * yc + T[10] * z + T[11];
          X.push_back(px); Y.push_back(py); Z.push_back(pz); ZC.push_back(z);
          PU.push_back(wcol_[ci + 1]); PV.push_back(int32_t(yi));
        }
      }
      if (int(X.size()) < p_.min_points) continue;
      // 깊이 이상값(마스크 가장자리로 뒤 벽·바닥이 비침): 카메라 깊이 중앙값 ± max(k·1.4826·MAD, floor) 밖 점 버림
      // 깊이 중앙값·MAD: 점이 kPctMax 넘으면 고른 간격 표본으로(백분위 오차 ≪ 칸 크기, 계산 약 1/3)
      subsample(ZC, &ZS);
      const double zmed = pct(ZS, 0.5);
      for (double& z : ZS) z = std::fabs(z - zmed);
      const double band = std::max(p_.mad_k * 1.4826 * pct(ZS, 0.5), p_.mad_floor);
      int near_hand = 0;
      const double hand2 = p_.hand_r * p_.hand_r, chand2 = p_.cloud_hand_r * p_.cloud_hand_r, body2 = p_.body_r * p_.body_r;
      size_t w = 0;
      Obs o;
      seen.clear();
      o.cxyz.reserve(3 * X.size());
      o.cpx.reserve(2 * X.size());
      for (size_t i = 0; i < X.size(); ++i) {
        if (std::fabs(ZC[i] - zmed) > band) continue;
        const double pp[3] = {X[i], Y[i], Z[i]};
        const double dh2 = std::min(d2(pp, f.eef[0]), d2(pp, f.eef[1]));
        if (dh2 < hand2) ++near_hand;
        // 구름 후보: 손·몸 가까운 점은 빼고, 이 관측 안에서 복셀마다 처음 점 하나
        uint32_t old;
        const double bx = X[i] - f.base_xy[0], by = Y[i] - f.base_xy[1];
        if (dh2 >= chand2 && bx * bx + by * by >= body2 &&
            seen.insert(voxelKey(float(X[i]), float(Y[i]), float(Z[i]), inv_vox), 0, &old)) {
          o.cxyz.insert(o.cxyz.end(), {float(X[i]), float(Y[i]), float(Z[i])});
          o.cpx.insert(o.cpx.end(), {PU[i], PV[i]});
        }
        X[w] = X[i]; Y[w] = Y[i]; Z[w] = Z[i];
        ++w;
      }
      X.resize(w); Y.resize(w); Z.resize(w);
      const int np = int(w);
      if (np < p_.min_points) continue;
      if (near_hand >= p_.hand_frac * np) continue;   // 손에 든 것
      o.det = k;
      o.zmed = zmed;
      o.sk = sk;
      o.cls = D->cls[k];
      o.score = D->score ? D->score[k] : 1.f;
      o.n = np;
      std::vector<double>* ax[3] = {&X, &Y, &Z};
      for (int a = 0; a < 3; ++a) {
        double lo, hi;
        subsample(*ax[a], &ZS);
        pct3(ZS, &lo, &o.pos[a], &hi);
        o.ext[a] = hi - lo;
        o.lo[a] = lo;
        o.hi[a] = hi;
      }
      obs.push_back(std::move(o));
    }
  }
  // 2. 같은 물체: 같은 이름 번호끼리 가까운 쌍부터 1:1
  std::vector<std::tuple<double, int, int>> pairs;
  for (int a = 0; a < int(obs.size()); ++a)
    for (int b = 0; b < int(objs_.size()); ++b) {
      const MapObject& m = objs_[b];
      if (m.cls != obs[a].cls || m.held_by >= 0) continue;
      const double e = std::max({obs[a].ext[0], obs[a].ext[1], obs[a].ext[2], m.ext[0], m.ext[1], m.ext[2]});
      const double thr = std::max(p_.da_min, p_.da_k * e);
      const double d = dist3(obs[a].pos, m.pos);
      double gap2 = 0;
      for (int k = 0; k < 3; ++k) {
        const double gk = std::max({0.0, obs[a].lo[k] - m.hi[k], m.lo[k] - obs[a].hi[k]});
        gap2 += gk * gk;
      }
      const double gap = std::sqrt(gap2);
      if (d < thr || gap < p_.da_gap) pairs.emplace_back(gap + 1e-3 * d, a, b);
    }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> obs_to(obs.size(), -1), obj_hit(objs_.size(), 0);
  for (auto& [d, a, b] : pairs) {
    if (obs_to[a] >= 0 || obj_hit[b]) continue;
    obs_to[a] = b;
    obj_hit[b] = 1;
  }
  // 3. 갱신
  for (int a = 0; a < int(obs.size()); ++a) {
    const Obs& o = obs[a];
    DetAssoc& as = assoc_[o.det];
    as.n_valid = o.n;
    as.area_px = float(o.n) * o.sk * o.sk;
    as.depth_med = float(o.zmed);
    if (obs_to[a] >= 0) {
      MapObject& m = objs_[obs_to[a]];
      as.obj_id = m.id;
      if (m.parent && dist3(m.pos, o.pos) > 0.3) m.parent = 0;
      if (m.last_kf != f.stamp) ++m.n_obs;
      m.last_kf = f.stamp;
      const double w = std::min<double>(m.n_obs, 20);
      const bool big = kindOf(m.cls) == kKindStatic || std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], o.ext[0], o.ext[1]}) > p_.big;
      for (int k = 0; k < 3; ++k) {
        if (big) {
          // 합집합, 단 keyframe 마다 면마다 grow_max 까지·한 변 max_ext 까지(이상값·잘못 붙은 관측이 끝없이 키우지 않게)
          const double lo = std::max(std::min(m.lo[k], o.lo[k]), m.lo[k] - p_.grow_max);
          const double hi = std::min(std::max(m.hi[k], o.hi[k]), m.hi[k] + p_.grow_max);
          if (hi - lo <= p_.max_ext) {
            m.lo[k] = lo;
            m.hi[k] = hi;
          }
          m.pos[k] = 0.5 * (m.lo[k] + m.hi[k]);
          m.ext[k] = m.hi[k] - m.lo[k];
        } else {
          m.pos[k] = (m.pos[k] * (w - 1) + o.pos[k]) / w;
          m.ext[k] = (m.ext[k] * (w - 1) + o.ext[k]) / w;
          m.lo[k] = (m.lo[k] * (w - 1) + o.lo[k]) / w;
          m.hi[k] = (m.hi[k] * (w - 1) + o.hi[k]) / w;
        }
      }
      m.score = std::max(m.score, o.score);
      m.last_seen = f.stamp;
      m.misses = 0;
      if (m.state == SM_GONE) {
        m.state = m.moved ? SM_MOVED : SM_SEEN;
        event(f.stamp, m, 6);
      }
      if (!m.confirmed && int(m.n_obs) >= p_.confirm) {
        m.confirmed = true;
        event(f.stamp, m, 1);
      }
      continue;
    }
    // 안 맞은 관측: 같은 이름의 확정 물체가 '사라짐'이면 그것이 옮겨진 것으로 잇는다(Khronos 식 이력)
    MapObject* moved_from = nullptr;
    double best = 1e9;
    for (auto& m : objs_) {
      if (m.cls != o.cls || !m.confirmed || m.held_by >= 0 || obj_hit[&m - objs_.data()]) continue;
      if (m.state != SM_GONE) continue;   // 사라짐으로 판정된 것만(같은 이름이 새로 하나 더 생긴 것과 헷갈리지 않게)
      const double d = dist3(o.pos, m.pos);
      if (d < best) { best = d; moved_from = &m; }
    }
    if (moved_from) {
      MapObject& m = *moved_from;
      as.obj_id = m.id;
      m.cloud.clear(f.stamp);   // 다른 자리에서 다시 찾음: 옛 구름은 비우고 새 관측으로 다시 쌓음
      for (int k = 0; k < 3; ++k) { m.pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k]; }
      m.moved = true;
      m.state = SM_MOVED;
      m.misses = 0;
      m.last_seen = f.stamp;
      m.last_kf = f.stamp;
      ++m.n_obs;
      obj_hit[moved_from - objs_.data()] = 1;
      event(f.stamp, m, 2);
      continue;
    }
    MapObject m;
    m.id = next_id_++;
    m.cls = o.cls;
    for (int k = 0; k < 3; ++k) { m.pos[k] = m.first_pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k]; }
    m.n_obs = 1;
    m.first_seen = m.last_seen = m.last_kf = f.stamp;
    m.score = o.score;
    m.confirmed = p_.confirm <= 1;
    as.obj_id = m.id;
    objs_.push_back(m);
    obj_hit.push_back(1);
    event(f.stamp, objs_.back(), 0);
  }
  // 구름 후보를 물체 id 와 함께 내놓음(색은 호출자가 붙여 addPoints)
  for (Obs& o : obs) {
    const uint32_t id = assoc_[o.det].obj_id;
    if (!id || o.cxyz.empty()) continue;
    ObsPoints q;
    q.obj_id = id;
    q.det = o.det;
    q.xyz = std::move(o.cxyz);
    q.px = std::move(o.cpx);
    points_.push_back(std::move(q));
  }
  // 4. 부재 확인(확정·안 든 것·이번에 안 맞은 것)
  if (f.depth_m || f.depth_mm) {
    for (size_t b = 0; b < objs_.size(); ++b) {
      MapObject& m = objs_[b];
      if (!m.confirmed || m.held_by >= 0 || obj_hit[b] || m.state == SM_GONE) continue;
      if (m.parent) continue;   // 통 안에 넣은 것은 안 보여도 그대로 있다고 본다
      if (kindOf(m.cls) == kKindStatic) continue;   // 가구·가전·붙박이는 사라지지 않음
      // 큰 가구(한 변 > big)는 사라짐 판정을 하지 않는다: 부분만 보이고 중심 한 점으로 가림을 판단하기 어렵고, 과제 중 없어지지 않는다
      if (std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]}) > p_.big) continue;
      // 손 가까이는 판단하지 않는다: 손에 든 관측은 거르므로(위) 잡으러 다가가는 동안 '안 보임'으로 셈하면 안 됨
      if (dist3(m.pos, f.eef[0]) < p_.hand_r + 0.1 || dist3(m.pos, f.eef[1]) < p_.hand_r + 0.1) continue;
      // map → 카메라
      const double d[3] = {m.pos[0] - T[3], m.pos[1] - T[7], m.pos[2] - T[11]};
      const double xc = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
      const double yc = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
      const double zc = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
      if (zc < 0.3 || zc > p_.zmax) continue;
      const int u = int(f.fx * xc / zc + f.cx), v = int(f.fy * yc / zc + f.cy);
      const double size_px = f.fx * std::max({m.ext[0], m.ext[1], m.ext[2]}) / zc;
      if (size_px < p_.min_px || u < 2 || v < 2 || u >= f.w - 2 || v >= f.h - 2) continue;
      // 3×3 깊이 중앙값이 물체보다 occl 넘게 가까우면 가려진 것
      float ds[9];
      int nd = 0;
      for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du) {
          const float z = depthAt(u + du, v + dv);
          if (z > 0) ds[nd++] = z;
        }
      if (nd < 5) continue;
      std::nth_element(ds, ds + nd / 2, ds + nd);
      if (ds[nd / 2] < zc - p_.occl) continue;
      if (m.misses++ == 0) m.first_miss = f.stamp;
      if (m.misses >= p_.gone_misses && f.stamp - m.first_miss >= p_.gone_min_s - 1e-9) {
        m.state = SM_GONE;
        event(f.stamp, m, 3);
      }
    }
  }
  // 5. 오래된 후보 버리기
  objs_.erase(std::remove_if(objs_.begin(), objs_.end(),
                             [&](const MapObject& m) { return !m.confirmed && f.stamp - m.last_seen > p_.prune_s; }),
              objs_.end());
  // 6. 중복 병합(da): 한 프레임에 일부만 보였거나 마스크가 쪼개져 따로 확정된 같은 물체를 하나로
  static const bool no_merge = std::getenv("SM_NO_MERGE") != nullptr;   // A/B 비교용
  static const bool log_merge = std::getenv("SM_MERGE_LOG") != nullptr;
  if (p_.merge && !no_merge) {
    da::MergeParams mp;
    mp.overlap_min = p_.merge_overlap;
    mp.min_ext = p_.merge_min_ext;
    for (const da::MergeResult& r : da::mergeDuplicates(objs_, mp, p_, f.stamp, &kinds_)) {
      if (log_merge) std::fprintf(stderr, "[da] t=%.1f merge keep O%u drop O%u overlap %.2f\n", f.stamp, r.keep, r.drop, r.overlap);
      for (DetAssoc& as : assoc_) if (as.obj_id == r.drop) as.obj_id = r.keep;
      for (ObsPoints& op : points_) if (op.obj_id == r.drop) op.obj_id = r.keep;
      for (const MapObject& m : objs_) if (m.id == r.keep) { event(f.stamp, m, 7); break; }
    }
  }
}

}  // namespace scenemap
