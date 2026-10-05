#include "da/merge.hpp"

#include <algorithm>
#include <cmath>

namespace scenemap::da {

double boxOverlap(const double alo[3], const double ahi[3], const double blo[3], const double bhi[3], double min_ext) {
  double r = 1.0;
  for (int k = 0; k < 3; ++k) {
    // 얇은 변은 중심 둘레로 min_ext 까지 부풀려서(러그·액자) 비율이 0/0 이 되지 않게
    const double ca = 0.5 * (alo[k] + ahi[k]), cb = 0.5 * (blo[k] + bhi[k]);
    const double ha = 0.5 * std::max(ahi[k] - alo[k], min_ext), hb = 0.5 * std::max(bhi[k] - blo[k], min_ext);
    const double ov = std::min(ca + ha, cb + hb) - std::max(ca - ha, cb - hb);
    const double f = ov <= 0 ? 0.0 : std::min(1.0, ov / (2.0 * std::min(ha, hb)));
    r *= f;
    if (r <= 0) return 0.0;
  }
  return r;
}

namespace {

// 큰 가구(고정 종류이거나 한 변 > big)면 상자 합집합으로 합친다
bool bigPair(const MapObject& a, const MapObject& b, const ObjParams& op, const std::vector<uint8_t>* kinds) {
  const bool stat = kinds && a.cls >= 0 && size_t(a.cls) < kinds->size() && (*kinds)[a.cls] == kKindStatic;
  const double ext_max = std::max({a.hi[0] - a.lo[0], a.hi[1] - a.lo[1], b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]});
  return stat || ext_max > op.big;
}

// b 를 a 에 합친다(a 가 남음)
void absorb(MapObject& a, MapObject& b, const ObjParams& op, double stamp, const std::vector<uint8_t>* kinds) {
  const bool big = bigPair(a, b, op, kinds);
  const double wa = std::max<double>(a.n_obs, 1), wb = std::max<double>(b.n_obs, 1);
  for (int k = 0; k < 3; ++k) {
    if (big) {
      a.lo[k] = std::min(a.lo[k], b.lo[k]);
      a.hi[k] = std::max(a.hi[k], b.hi[k]);
      a.pos[k] = 0.5 * (a.lo[k] + a.hi[k]);
      a.ext[k] = a.hi[k] - a.lo[k];
    } else {
      a.pos[k] = (a.pos[k] * wa + b.pos[k] * wb) / (wa + wb);
      a.ext[k] = (a.ext[k] * wa + b.ext[k] * wb) / (wa + wb);
      a.lo[k] = (a.lo[k] * wa + b.lo[k] * wb) / (wa + wb);
      a.hi[k] = (a.hi[k] * wa + b.hi[k] * wb) / (wa + wb);
    }
  }
  if (b.first_seen < a.first_seen) {
    a.first_seen = b.first_seen;
    for (int k = 0; k < 3; ++k) a.first_pos[k] = b.first_pos[k];
  }
  a.n_obs += b.n_obs;
  a.last_seen = std::max(a.last_seen, b.last_seen);
  a.last_kf = std::max(a.last_kf, b.last_kf);
  a.score = std::max(a.score, b.score);
  for (const auto& [c, w] : b.votes) {   // 이름 표 합치기(이름 = 최댓값)
    bool f = false;
    for (auto& [ac, aw] : a.votes)
      if (ac == c) { aw += w; f = true; break; }
    if (!f) a.votes.emplace_back(c, w);
  }
  a.max_det_z = std::max(a.max_det_z, b.max_det_z);
  a.moved = a.moved || b.moved;
  a.misses = std::min(a.misses, b.misses);
  a.n_vis_miss += b.n_vis_miss;
  if (a.state != SM_SEEN && b.state == SM_SEEN) a.state = SM_SEEN;
  if (!a.parent) { a.parent = b.parent; for (int k = 0; k < 3; ++k) a.parent_rel[k] = b.parent_rel[k]; }
  if (op.insp.on) inspMerge(a.insp, b.insp, a.lo, a.hi, op.insp);   // 살펴본 정도(합친 상자 기준)
  // 점 구름: b 의 점(org + 점)을 a 에 넣는다
  if (b.cloud.data && !b.cloud.data->pts.empty()) {
    const auto& pts = b.cloud.data->pts;
    std::vector<float> xyz(3 * pts.size());
    std::vector<uint8_t> rgb(3 * pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      xyz[3 * i] = float(b.cloud.org[0] + pts[i].x);
      xyz[3 * i + 1] = float(b.cloud.org[1] + pts[i].y);
      xyz[3 * i + 2] = float(b.cloud.org[2] + pts[i].z);
      rgb[3 * i] = pts[i].r; rgb[3 * i + 1] = pts[i].g; rgb[3 * i + 2] = pts[i].b;
    }
    a.cloud.add(xyz.data(), rgb.data(), int(pts.size()), op.voxel, op.cloud_cap, stamp);
  }
}

}  // namespace

void absorbObject(MapObject& keep, MapObject& drop, const ObjParams& op, double stamp, const std::vector<uint8_t>* kinds) {
  absorb(keep, drop, op, stamp, kinds);
}

}  // namespace scenemap::da
