// 물체 점 구름(include/scenemap/cloud.hpp).
#include "scenemap/cloud.hpp"

#include <algorithm>
#include <cmath>

namespace scenemap {
namespace {
constexpr uint64_t kEmpty = ~uint64_t(0);
inline size_t hash64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  return size_t(k);
}
}  // namespace

void VoxelIndex::reserve(size_t n) {
  size_t cap = 64;
  while (cap < 2 * n) cap <<= 1;
  if (cap <= keys_.size()) return;
  std::vector<uint64_t> ok;
  std::vector<uint32_t> ov, og;
  ok.swap(keys_);
  ov.swap(vals_);
  og.swap(gen_);
  const uint32_t oc = cur_;
  keys_.assign(cap, kEmpty);
  vals_.assign(cap, 0);
  gen_.assign(cap, 0);
  cur_ = 1;
  n_ = 0;
  uint32_t dummy;
  for (size_t i = 0; i < ok.size(); ++i)
    if (og[i] == oc) insert(ok[i], ov[i], &dummy);
}

void VoxelIndex::clear() {
  n_ = 0;
  if (++cur_ == 0) {   // 세대 한 바퀴: 진짜로 비움
    std::fill(gen_.begin(), gen_.end(), 0u);
    cur_ = 1;
  }
}

void VoxelIndex::grow() { reserve(std::max<size_t>(32, n_ * 2)); }

bool VoxelIndex::insert(uint64_t key, uint32_t val, uint32_t* old) {
  if (keys_.empty() || 2 * (n_ + 1) > keys_.size()) grow();
  const size_t mask = keys_.size() - 1;
  for (size_t i = hash64(key) & mask;; i = (i + 1) & mask) {
    if (gen_[i] != cur_) {
      gen_[i] = cur_;
      keys_[i] = key;
      vals_[i] = val;
      ++n_;
      return true;
    }
    if (keys_[i] == key) {
      *old = vals_[i];
      return false;
    }
  }
}

uint64_t voxelKey(float x, float y, float z, float inv) {
  auto q = [&](float v) { return uint64_t(int64_t(std::floor(v * inv)) + (1 << 20)) & 0x1FFFFF; };
  return q(x) | q(y) << 21 | q(z) << 42;
}

void ObjCloud::clear(double t) {
  if (!data && !has_org) return;
  data.reset();
  has_org = false;
  ++version;
  stamp = t;
}

void ObjCloud::translate(const double d[3], double t) {
  if (!data || (d[0] == 0 && d[1] == 0 && d[2] == 0)) return;
  for (int k = 0; k < 3; ++k) org[k] += d[k];
  ++version;
  stamp = t;
}

void ObjCloud::add(const float* xyz, const uint8_t* rgb, int n, double voxel, int cap, double t) {
  if (n <= 0) return;
  if (!has_org) {   // 처음 점을 원점으로(float 정밀도: 원점 가까이 작은 값)
    for (int k = 0; k < 3; ++k) org[k] = xyz[k];
    has_org = true;
  }
  // 쓸 때 복사: 스냅숏이 들고 있으면 새로 만든다
  std::shared_ptr<CloudData> d = data && data.use_count() == 1 ? std::const_pointer_cast<CloudData>(data)
                                                               : std::make_shared<CloudData>(data ? *data : CloudData{});
  const float inv = float(1.0 / voxel);
  const uint32_t seq = ++d->seq;
  d->index.reserve(d->pts.size() + size_t(n));
  for (int i = 0; i < n; ++i) {
    CloudPt p;
    p.x = float(xyz[3 * i] - org[0]);
    p.y = float(xyz[3 * i + 1] - org[1]);
    p.z = float(xyz[3 * i + 2] - org[2]);
    p.r = rgb ? rgb[3 * i] : 128;
    p.g = rgb ? rgb[3 * i + 1] : 128;
    p.b = rgb ? rgb[3 * i + 2] : 128;
    p.a = 0;
    p.seq = seq;
    uint32_t old;
    if (d->index.insert(voxelKey(p.x, p.y, p.z, inv), uint32_t(d->pts.size()), &old))
      d->pts.push_back(p);
    else
      d->pts[old] = p;
  }
  if (cap > 0 && d->pts.size() > size_t(cap)) {   // 오래된 것부터 버려 cap 의 90 % 로
    const size_t keep = std::max<size_t>(1, size_t(cap) * 9 / 10);
    std::nth_element(d->pts.begin(), d->pts.begin() + (d->pts.size() - keep), d->pts.end(),
                     [](const CloudPt& a, const CloudPt& b) { return a.seq < b.seq; });
    d->pts.erase(d->pts.begin(), d->pts.begin() + (d->pts.size() - keep));
    d->index.clear();
    d->index.reserve(d->pts.size());
    uint32_t old;
    for (size_t i = 0; i < d->pts.size(); ++i) d->index.insert(voxelKey(d->pts[i].x, d->pts[i].y, d->pts[i].z, inv), uint32_t(i), &old);
  }
  data = std::move(d);
  ++version;
  stamp = t;
}

}  // namespace scenemap
