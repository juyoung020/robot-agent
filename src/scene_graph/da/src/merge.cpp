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

// 같은 이름 번호(= 같은 종류)끼리만. 합집합이 될 쌍은 한 변이 max_ext 를 넘으면 안 합침(objmap 상자 키우기와 같은 한도)
// 이름이 다르면(이름 모으기 켬): 상자가 거의 같아야(두 상자의 IoU ≥ name_merge_iou) — 한 물체가 프레임마다 다른 이름으로 따로 생긴 것
double boxIou(const MapObject& a, const MapObject& b, double min_ext) {
  double ia = 1, va = 1, vb = 1;
  for (int k = 0; k < 3; ++k) {
    const double ea = std::max(a.hi[k] - a.lo[k], min_ext), eb = std::max(b.hi[k] - b.lo[k], min_ext);
    const double ca = 0.5 * (a.lo[k] + a.hi[k]), cb = 0.5 * (b.lo[k] + b.hi[k]);
    const double ov = std::min(ca + 0.5 * ea, cb + 0.5 * eb) - std::max(ca - 0.5 * ea, cb - 0.5 * eb);
    if (ov <= 0) return 0.0;
    ia *= ov; va *= ea; vb *= eb;
  }
  return ia / (va + vb - ia);
}

bool mergeable(const MapObject& a, const MapObject& b, const ObjParams& op, const std::vector<uint8_t>* kinds) {
  if (!(a.confirmed && b.confirmed && a.held_by < 0 && b.held_by < 0 && a.state != SM_GONE && b.state != SM_GONE)) return false;
  if (a.cls != b.cls && !(op.name_vote && boxIou(a, b, 0.05) >= op.name_merge_iou)) return false;
  if (bigPair(a, b, op, kinds))
    for (int k = 0; k < 3; ++k)
      if (std::max(a.hi[k], b.hi[k]) - std::min(a.lo[k], b.lo[k]) > op.max_ext) return false;
  return true;
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
  if (op.name_vote) {
    float best = -1;
    for (const auto& [c, w] : a.votes)
      if (w > best) { best = w; a.cls = c; }
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

std::vector<MergeResult> mergeDuplicates(std::vector<MapObject>& objs, const MergeParams& mp, const ObjParams& op, double stamp,
                                         const std::vector<uint8_t>* kinds) {
  std::vector<MergeResult> out;
  if (!mp.enable) return out;
  // 가장 많이 겹치는 쌍부터 하나씩 합치고 다시 센다(합치면 상자가 바뀌므로). 합칠 때마다 물체가 하나 줄어 반드시 끝난다.
  while (objs.size() >= 2) {
    double best = 0;
    int bi = -1, bj = -1;
    for (size_t i = 0; i < objs.size(); ++i)
      for (size_t j = i + 1; j < objs.size(); ++j) {
        if (!mergeable(objs[i], objs[j], op, kinds)) continue;
        const double ov = boxOverlap(objs[i].lo, objs[i].hi, objs[j].lo, objs[j].hi, mp.min_ext);
        if (ov >= mp.overlap_min && ov > best) { best = ov; bi = int(i); bj = int(j); }
      }
    if (bi < 0) break;
    // 관측이 많은 쪽을 남김, 같으면 먼저 본 쪽(id 가 작은 쪽)
    int keep = bi, drop = bj;
    if (objs[bj].n_obs > objs[bi].n_obs || (objs[bj].n_obs == objs[bi].n_obs && objs[bj].id < objs[bi].id)) std::swap(keep, drop);
    out.push_back({objs[keep].id, objs[drop].id, best});
    absorb(objs[keep], objs[drop], op, stamp, kinds);
    objs.erase(objs.begin() + drop);
  }
  return out;
}

}  // namespace scenemap::da
