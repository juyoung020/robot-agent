#include "scenemap/grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace scenemap {

OccGrid::OccGrid(const GridParams& p) : p_(p), inv_res_(1.f / p.res) {
  q_hit_ = int16_t(std::lround(p.l_hit * kQ));
  q_miss_ = int16_t(std::lround(p.l_miss * kQ));
  q_min_ = int16_t(std::lround(p.l_min * kQ));
  q_max_ = int16_t(std::lround(p.l_max * kQ));
  lut_off_ = -q_min_;
  const int n = q_max_ - q_min_ + 1;
  lut8v_.resize(n);
  lutp_.resize(n);
  for (int k = 0; k < n; ++k) {
    const float L = float(k - lut_off_) / kQ;
    const float pr = 1.f / (1.f + std::exp(-L));
    lutp_[k] = pr;
    lut8v_[k] = int8_t(std::lround(100.f * pr));
  }
}

void OccGrid::ensure(int ix0, int iy0, int ix1, int iy1) {
  if (w_ && ix0 >= x0_ && iy0 >= y0_ && ix1 < x0_ + w_ && iy1 < y0_ + h_) return;
  const int pad = 64;
  int nx0 = w_ ? std::min(x0_, ix0 - pad) : ix0 - pad;
  int ny0 = w_ ? std::min(y0_, iy0 - pad) : iy0 - pad;
  int nx1 = w_ ? std::max(x0_ + w_, ix1 + pad) : ix1 + pad;
  int ny1 = w_ ? std::max(y0_ + h_, iy1 + pad) : iy1 + pad;
  const int nw = nx1 - nx0, nh = ny1 - ny0;
  std::vector<Hot> hot(size_t(nw) * nh, Hot{0, -1, 0, 0});
  std::vector<Cold> cold(hot.size(), Cold{0, 0, 0, 0, 0, 0});
  for (int y = 0; y < h_; ++y) {
    const size_t o = size_t(y) * w_, n = size_t(y + y0_ - ny0) * nw + (x0_ - nx0);
    std::memcpy(&hot[n], &hot_[o], sizeof(Hot) * w_);
    std::memcpy(&cold[n], &cold_[o], sizeof(Cold) * w_);
  }
  hot_.swap(hot);
  cold_.swap(cold);
  x0_ = nx0; y0_ = ny0; w_ = nw; h_ = nh;
  ++cells_ver_;
}

inline void OccGrid::mark(int ix, int iy, int8_t old_c8, int8_t new_c8) {
  if (old_c8 == new_c8) return;
  cells_changed_ = true;
  for (int k = 0; k < 4; ++k) {
    int* b = db_[k];
    if (!dirty_[k]) { b[0] = b[2] = ix; b[1] = b[3] = iy; dirty_[k] = true; continue; }
    b[0] = std::min(b[0], ix); b[1] = std::min(b[1], iy); b[2] = std::max(b[2], ix); b[3] = std::max(b[3], iy);
  }
}

inline void OccGrid::hit(size_t i, float hx, float hy, float nx, float ny, int ix, int iy) {
  Hot& h = hot_[i];
  if (h.stamp == scan_id_) return;
  h.stamp = scan_id_;
  const int16_t L = h.L;
  const int16_t nL = int16_t(std::min<int>(L + q_hit_, q_max_));
  const bool was_seen = h.flags & kSeen;
  h.flags |= kSeen | kSum;
  changed_ += (nL != L) | !was_seen;
  h.L = nL;
  const int8_t c8 = lut8v_[nL + lut_off_];
  mark(ix, iy, h.c8, c8);
  h.c8 = c8;
  Cold& c = cold_[i];
  if (c.cnt < 60000) { c.sx += hx; c.sy += hy; ++c.cnt; }
  if ((nx != 0 || ny != 0) && c.cntn < 60000) { c.nx += nx; c.ny += ny; ++c.cntn; }
}

inline void OccGrid::miss(size_t i, int ix, int iy) {
  Hot& h = hot_[i];
  if (h.stamp == scan_id_) return;
  h.stamp = scan_id_;
  const int16_t L = h.L;
  const int16_t nL = int16_t(std::max<int>(L + q_miss_, q_min_));
  const bool was_seen = h.flags & kSeen;
  h.flags |= kSeen;
  if (nL == L && was_seen) return;   // 이미 한계(빈칸 쪽): 아무것도 안 바뀜
  ++changed_;
  h.L = nL;
  const int8_t c8 = lut8v_[nL + lut_off_];
  mark(ix, iy, h.c8, c8);
  h.c8 = c8;
  if (nL <= 0 && (h.flags & kSum)) {
    h.flags &= uint8_t(~kSum);
    cold_[i] = Cold{0, 0, 0, 0, 0, 0};
  }
}

int OccGrid::insert(const Scan2& s, const Pose2& pose) {
  if (++scan_id_ == 0) {   // 2^32 번째 스캔: 표시 다시 시작
    for (Hot& h : hot_) h.stamp = 0;
    scan_id_ = 1;
  }
  changed_ = 0;
  cells_changed_ = false;
  const double c = std::cos(pose.th), sn = std::sin(pose.th);
  const double ox = pose.x + c * s.ox - sn * s.oy, oy = pose.y + sn * s.ox + c * s.oy;
  const int cx = cellOf(ox), cy = cellOf(oy);
  // 점마다 한 번 변환: map 좌표(맞은 점 합용)와 칸. 순서 = 맞추기 점, 레이저 한 줄, 빈 광선 끝
  const size_t nm = s.mx.size(), nh = s.hx.size(), nf = s.fx.size(), n = nm + nh + nf;
  wc_.resize(2 * n);
  wf_.resize(2 * n);
  int bx0 = cx, by0 = cy, bx1 = cx, by1 = cy;
  auto put = [&](size_t k, float x, float y) {
    const double wx = pose.x + c * x - sn * y, wy = pose.y + sn * x + c * y;
    const int ix = cellOf(wx), iy = cellOf(wy);
    wf_[2 * k] = float(wx); wf_[2 * k + 1] = float(wy);
    wc_[2 * k] = ix; wc_[2 * k + 1] = iy;
    bx0 = std::min(bx0, ix); by0 = std::min(by0, iy); bx1 = std::max(bx1, ix); by1 = std::max(by1, iy);
  };
  for (size_t k = 0; k < nm; ++k) put(k, s.mx[k], s.my[k]);
  for (size_t k = 0; k < nh; ++k) put(nm + k, s.hx[k], s.hy[k]);
  for (size_t k = 0; k < nf; ++k) put(nm + nh + k, s.fx[k], s.fy[k]);
  ensure(bx0, by0, bx1, by1);
  ++version_;
  // 맞음 먼저(한 스캔에서 맞은 칸은 빈칸으로 덮이지 않게)
  // 맞추기 점 먼저(법선이 있는 쪽이 칸의 첫 맞음이 되게), 그다음 레이저 한 줄
  const bool have_n = s.mnx.size() >= nm;
  for (size_t k = 0; k < nm; ++k) {
    const float nx = have_n ? float(c * s.mnx[k] - sn * s.mny[k]) : 0.f;
    const float ny = have_n ? float(sn * s.mnx[k] + c * s.mny[k]) : 0.f;
    const int ix = wc_[2 * k], iy = wc_[2 * k + 1];
    hit(idx(ix, iy), wf_[2 * k], wf_[2 * k + 1], nx, ny, ix, iy);
  }
  for (size_t k = nm; k < nm + nh; ++k) {
    const int ix = wc_[2 * k], iy = wc_[2 * k + 1];
    hit(idx(ix, iy), wf_[2 * k], wf_[2 * k + 1], 0.f, 0.f, ix, iy);
  }
  // Bresenham(정수), 끝 칸 제외. 칸 번호는 더하기로만 움직임
  const long W = w_;
  const size_t i0 = idx(cx, cy);
  auto ray = [&](int x1, int y1) {
    int x = cx, y = cy;
    size_t i = i0;
    const int dx = std::abs(x1 - x), dy = -std::abs(y1 - y), sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1;
    const long di = sx, dj = sy * W;
    int err = dx + dy;
    while (!(x == x1 && y == y1)) {
      miss(i, x, y);
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; i += di; }
      if (e2 <= dx) { err += dx; y += sy; i += dj; }
    }
  };
  for (size_t k = nm; k < nm + nh; ++k) ray(wc_[2 * k], wc_[2 * k + 1]);
  for (size_t k = nm + nh; k < n; ++k) {
    const int ix = wc_[2 * k], iy = wc_[2 * k + 1];
    ray(ix, iy);
    miss(idx(ix, iy), ix, iy);
  }
  if (cells_changed_) ++cells_ver_;
  return changed_;
}

bool OccGrid::normal(int ix, int iy, float* nx, float* ny) const {
  if (!inside(ix, iy)) return false;
  const size_t i = idx(ix, iy);
  const Cold& c = cold_[i];
  if (!c.cntn || hot_[i].L <= 0) return false;
  const float n = std::sqrt(c.nx * c.nx + c.ny * c.ny);
  if (n < 0.5f * c.cntn) return false;
  *nx = c.nx / n;
  *ny = c.ny / n;
  return true;
}

bool OccGrid::mean(int ix, int iy, float* mx, float* my, int* n) const {
  if (!inside(ix, iy)) return false;
  const size_t i = idx(ix, iy);
  const Hot& h = hot_[i];
  if (!(h.flags & kSum) || h.L <= 0) return false;
  const Cold& c = cold_[i];
  if (!c.cnt) return false;
  *mx = c.sx / c.cnt;
  *my = c.sy / c.cnt;
  if (n) *n = c.cnt;
  return true;
}

float OccGrid::prob(int ix, int iy) const {
  if (!inside(ix, iy)) return pmin;
  const Hot& h = hot_[idx(ix, iy)];
  if (!(h.flags & kSeen)) return pmin;
  return std::clamp(lutp_[h.L + lut_off_], pmin, pmax);
}

bool OccGrid::takeDirty(int* ix0, int* iy0, int* ix1, int* iy1, int consumer) {
  const int k = consumer < 0 ? 0 : consumer > 3 ? 3 : consumer;
  if (!dirty_[k]) return false;
  *ix0 = db_[k][0]; *iy0 = db_[k][1]; *ix1 = db_[k][2]; *iy1 = db_[k][3];
  dirty_[k] = false;
  return true;
}

void OccGrid::exportRect(int lx0, int ly0, int lx1, int ly1, int8_t* o) const {
  const int rw = lx1 - lx0 + 1;
  for (int y = ly0; y <= ly1; ++y) {
    const Hot* row = hot_.data() + size_t(y) * w_ + lx0;
    for (int x = 0; x < rw; ++x) o[size_t(y - ly0) * rw + x] = row[x].c8;
  }
}

void OccGrid::export8(int8_t* o) const {
  const size_t n = hot_.size();
  const Hot* h = hot_.data();
  for (size_t i = 0; i < n; ++i) o[i] = h[i].c8;
}

std::vector<int8_t> OccGrid::export8() const {
  std::vector<int8_t> o(hot_.size());
  export8(o.data());
  return o;
}

}  // namespace scenemap
