#include "scenemap/walls.hpp"

#include <algorithm>
#include <memory>
#include <cmath>
#include <limits>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace scenemap {
namespace {

struct Run { int a, b; };
struct Group { int r0, r1, x0, x1; };

// 점유 칸을 행마다 비트(64칸 = 워드 1개)로 묶는다. 파이썬(PGM) 행 순서: 행 0 = 맨 위(최대 y) — 같은 선분 순서를 내려고 맞춘다.
// SSE2(x86-64 기본)로 16칸씩 비교해 movemask 로 바로 비트를 만든다.
struct Bits {
  int w = 0, h = 0, nw = 0;
  std::vector<uint64_t> v;   // [iy·nw + k]
  // 파이썬(PGM) 행 iy 의 비트를 다시 만든다(iy0..iy1 끝 포함). 크기가 다르면 전부 새로.
  void build(const WallGrid& g, int iy0, int iy1) {
    if (g.w != w || g.h != h) {
      w = g.w; h = g.h; nw = (w + 63) / 64;
      v.assign(size_t(h) * nw, 0);
      iy0 = 0; iy1 = h - 1;
    }
    for (int iy = std::max(iy0, 0); iy <= std::min(iy1, h - 1); ++iy) {
      const int8_t* src = g.cells + size_t(h - 1 - iy) * w;
      uint64_t* dst = &v[size_t(iy) * nw];
      std::fill(dst, dst + nw, 0);
      int x = 0;
#if defined(__SSE2__)
      const __m128i thr = _mm_set1_epi8(kOccMin - 1);
      for (; x + 64 <= w; x += 64) {
        uint64_t m = 0;
        for (int q = 0; q < 4; ++q) {
          __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + x + 16 * q));
          m |= uint64_t(uint32_t(_mm_movemask_epi8(_mm_cmpgt_epi8(c, thr)))) << (16 * q);
        }
        dst[x >> 6] = m;
      }
#endif
      for (; x < w; ++x) if (src[x] >= kOccMin) dst[x >> 6] |= uint64_t(1) << (x & 63);
    }
  }
  const uint64_t* row(int iy) const { return &v[size_t(iy) * nw]; }
};

// 한 행(워드 nw 개)의 이어진 점유 구간을 앞에서부터 out 에 더한다. 시작·끝은 이웃 비트와 비교해 워드째 구한다(빈 워드는 건너뜀).
void rowRuns(const uint64_t* row, int nw, int min_run, std::vector<Run>& out) {
  int open = -1;
  for (int k = 0; k < nw; ++k) {
    const uint64_t cur = row[k];
    const uint64_t before = (k ? row[k - 1] >> 63 : 0), after = (k + 1 < nw ? row[k + 1] << 63 : 0);
    if (!cur) continue;
    uint64_t st = cur & ~((cur << 1) | before), en = cur & ~((cur >> 1) | after);
    while (st | en) {
      const int sb = st ? __builtin_ctzll(st) : 64, eb = en ? __builtin_ctzll(en) : 64;
      if (sb <= eb) { open = k * 64 + sb; st &= st - 1; }   // 한 비트 구간은 시작과 끝이 같은 비트: 시작 먼저
      if (sb > eb || (sb == eb)) {
        const int e = k * 64 + eb;
        if (open >= 0 && e - open + 1 >= min_run) out.push_back({open, e});
        open = -1;
        en &= en - 1;
      }
    }
  }
}

// rows 행의 run(offset 으로 나눠 담음)을 위 행과 겹치면 한 덩어리로 묶는다.
void groupRuns(const std::vector<Run>& runs, const std::vector<int>& off, int rows, double overlap, std::vector<Group>& done) {
  std::vector<Group> active, nxt;
  std::vector<char> used;
  for (int r = 0; r < rows; ++r) {
    nxt.clear();
    used.assign(active.size(), 0);
    for (int i = off[r]; i < off[r + 1]; ++i) {
      const Run& rn = runs[i];
      int hit = -1;
      for (size_t gi = 0; gi < active.size(); ++gi) {
        if (used[gi] || active[gi].r1 != r - 1) continue;
        int ov = std::min(rn.b, active[gi].x1) - std::max(rn.a, active[gi].x0) + 1;
        if (ov >= overlap * std::min(rn.b - rn.a + 1, active[gi].x1 - active[gi].x0 + 1)) { hit = int(gi); break; }
      }
      if (hit < 0) {
        nxt.push_back({r, r, rn.a, rn.b});
      } else {
        Group g = active[hit];
        used[hit] = 1;
        g.r1 = r;
        g.x0 = std::min(g.x0, rn.a);
        g.x1 = std::max(g.x1, rn.b);
        nxt.push_back(g);
      }
    }
    for (size_t gi = 0; gi < active.size(); ++gi) if (!used[gi]) done.push_back(active[gi]);
    active.swap(nxt);
  }
  done.insert(done.end(), active.begin(), active.end());
}

}  // namespace

struct WallExtractor::Impl {
  Bits bits;
  std::vector<Run> runs, vruns;
  std::vector<int> off, voff, start, pos;
  struct VRun { int x, a, b; };
  std::vector<VRun> vr;
  std::vector<uint64_t> zeros;
  std::vector<Group> groups;
};

WallExtractor::WallExtractor() : p_(new Impl) {}
WallExtractor::~WallExtractor() = default;
void WallExtractor::reset() { p_.reset(new Impl); segs_.clear(); }

const std::vector<WallSeg>& WallExtractor::update(const WallGrid& g, int y_lo, int y_hi, const std::vector<WallRect>* ignore,
                                                  double min_len, double max_thick, double overlap) {
  std::vector<WallSeg>& out = segs_;
  out.clear();
  if (g.w <= 0 || g.h <= 0) return out;
  const int min_run = std::max(1, int(std::lround(min_len / g.res)));
  Impl& I = *p_;
  // 격자 y(아래→위) 행 y_lo..y_hi 가 바뀜 → 파이썬 행 순서(위→아래)로 옮겨 그 행만 다시 비트로. y_lo > y_hi 면 전부.
  static const std::vector<WallRect> kNone;
  const std::vector<WallRect>& ig = ignore ? *ignore : kNone;
  const bool ig_changed = !(ig == last_ignore_);   // 영역이 바뀌면 그 둘레 행이 달라지므로 전부 다시
  if (ig_changed) { y_lo = 0; y_hi = -1; last_ignore_ = ig; }
  I.bits.build(g, y_lo > y_hi ? 0 : g.h - 1 - y_hi, y_lo > y_hi ? g.h - 1 : g.h - 1 - y_lo);
  if (!ig.empty()) {
    // 무시 영역 안의 점유 비트를 지운다(다시 만든 행만이 아니라 영역 전체 — 안 만든 행은 이미 지워져 있으니 같은 결과)
    Bits& B = I.bits;
    for (const WallRect& r : ig) {
      const int cx0 = std::max(0, int(std::floor((r.x0 - g.ox) / g.res))), cx1 = std::min(g.w - 1, int(std::floor((r.x1 - g.ox) / g.res)));
      const int cy0 = std::max(0, int(std::floor((r.y0 - g.oy) / g.res))), cy1 = std::min(g.h - 1, int(std::floor((r.y1 - g.oy) / g.res)));
      for (int y = cy0; y <= cy1; ++y) {
        uint64_t* row = &B.v[size_t(g.h - 1 - y) * B.nw];
        for (int x = cx0; x <= cx1;) {
          const int k = x >> 6, b0 = x & 63, n = std::min(64 - b0, cx1 - x + 1);
          const uint64_t mask = (n == 64 ? ~uint64_t(0) : ((uint64_t(1) << n) - 1)) << b0;
          row[k] &= ~mask;
          x += n;
        }
      }
    }
  }
  const Bits& bits = I.bits;
  auto emit = [&](const Group& gr, bool transpose) {
    if ((gr.r1 - gr.r0 + 1) * g.res > max_thick) return;
    double rc = (gr.r0 + gr.r1) / 2.0;
    double p[2][2];
    int k = 0;
    for (int c : {gr.x0, gr.x1}) {
      double iy = transpose ? c : rc, ix = transpose ? rc : c;
      p[k][0] = g.ox + (ix + 0.5) * g.res;
      p[k][1] = g.oy + (g.h - 1 - iy + 0.5) * g.res;
      ++k;
    }
    out.push_back({p[0][0], p[0][1], p[1][0], p[1][1]});
  };
  std::vector<Group>& groups = I.groups;
  groups.clear();
  // 가로 벽: 행마다 구간
  {
    std::vector<Run>& runs = I.runs;
    std::vector<int>& off = I.off;
    runs.clear();
    off.assign(g.h + 1, 0);
    for (int iy = 0; iy < g.h; ++iy) {
      rowRuns(bits.row(iy), bits.nw, min_run, runs);
      off[iy + 1] = int(runs.size());
    }
    groupRuns(runs, off, g.h, overlap, groups);
    for (const Group& gr : groups) emit(gr, false);
  }
  // 세로 벽: 전치 없이 열마다 구간. 행을 내려가며 "이번 행에서 시작/끝난 열"만 비트로 골라 낸다.
  {
    auto& vr = I.vr;
    vr.clear();
    std::vector<int>& start = I.start;
    start.assign(g.w + 64, 0);
    std::vector<uint64_t>& zeros = I.zeros;
    zeros.assign(bits.nw, 0);
    for (int iy = 0; iy <= g.h; ++iy) {
      const uint64_t* cur = iy < g.h ? bits.row(iy) : zeros.data();
      const uint64_t* prev = iy > 0 ? bits.row(iy - 1) : zeros.data();
      for (int k = 0; k < bits.nw; ++k) {
        uint64_t st = cur[k] & ~prev[k], en = prev[k] & ~cur[k];
        while (st) { start[k * 64 + __builtin_ctzll(st)] = iy; st &= st - 1; }
        while (en) {
          const int x = k * 64 + __builtin_ctzll(en);
          if (iy - start[x] >= min_run) vr.push_back({x, start[x], iy - 1});
          en &= en - 1;
        }
      }
    }
    // 열 번호 순(전치 격자의 행)으로 모은다 — 같은 열 안에서는 이미 위에서 아래로 정렬돼 있다
    std::vector<int>& voff = I.voff;
    voff.assign(g.w + 1, 0);
    for (const auto& r : vr) ++voff[r.x + 1];
    for (int x = 0; x < g.w; ++x) voff[x + 1] += voff[x];
    std::vector<Run>& vruns = I.vruns;
    vruns.resize(vr.size());
    I.pos.assign(voff.begin(), voff.end() - 1);
    for (const auto& r : vr) vruns[I.pos[r.x]++] = {r.a, r.b};
    groups.clear();
    groupRuns(vruns, voff, g.w, overlap, groups);
    for (const Group& gr : groups) emit(gr, true);
  }
  return out;
}

std::vector<WallSeg> wallSegments(const WallGrid& g, double min_len, double max_thick, double overlap, const std::vector<WallRect>* ignore) {
  WallExtractor e;
  return e.update(g, 0, -1, ignore, min_len, max_thick, overlap);
}

double wallAngle(const WallGrid& g) {
  std::vector<float> px, py;
  for (int y = 0; y < g.h; ++y)
    for (int x = 0; x < g.w; ++x)
      if (g.cells[size_t(y) * g.w + x] >= kOccMin) { px.push_back(float(x)); py.push_back(float(y)); }
  if (px.size() < 20) return 0.0;
  // 점들을 −θ 돌려 x·y 축으로 투영한 히스토그램(칸 폭)의 제곱합: 벽이 축과 나란할수록 크다. 90° 주기라 −45°..45° 만 본다.
  const int D = int(std::ceil(std::hypot(g.w, g.h))) + 2;
  std::vector<int> hx(2 * D + 1), hy(2 * D + 1);
  auto score = [&](double th) {
    std::fill(hx.begin(), hx.end(), 0); std::fill(hy.begin(), hy.end(), 0);
    const float c = float(std::cos(th)), s = float(std::sin(th)), d = float(D) + 0.5f;   // d: 음수 없이 반올림(int 자르기)
    for (size_t i = 0; i < px.size(); ++i) {
      ++hx[int(c * px[i] + s * py[i] + d)];
      ++hy[int(-s * px[i] + c * py[i] + d)];
    }
    double v = 0;
    for (int i = 0; i <= 2 * D; ++i) v += double(hx[i]) * hx[i] + double(hy[i]) * hy[i];
    return v;
  };
  constexpr double deg = M_PI / 180.0;
  double best = 0.0, bv = score(0.0);
  // 1° 간격 → 0.25° → 0.05° 로 좁힌다
  for (int k = -45; k < 45; ++k) { const double th = k * deg, v = score(th); if (v > bv) { bv = v; best = th; } }
  for (const double step : {0.25 * deg, 0.05 * deg}) {
    const double c0 = best;
    for (int k = -4; k <= 4; ++k) { const double th = c0 + k * step, v = score(th); if (v > bv) { bv = v; best = th; } }
  }
  if (best >= M_PI / 4) best -= M_PI / 2;
  if (best < -M_PI / 4) best += M_PI / 2;
  return best;
}

std::vector<WallSeg> wallSegmentsAligned(const WallGrid& g, double min_len, double max_thick, double overlap,
                                         const std::vector<WallRect>* ignore, double* angle_out) {
  if (g.w <= 0 || g.h <= 0) { if (angle_out) *angle_out = 0; return {}; }
  // ignore 영역은 원래 좌표 축에 맞는 상자라 먼저 지운다(각 판정에도 가구가 안 끼게)
  std::vector<int8_t> masked;
  const int8_t* src = g.cells;
  if (ignore && !ignore->empty()) {
    masked.assign(g.cells, g.cells + size_t(g.w) * g.h);
    for (const WallRect& r : *ignore) {
      const int cx0 = std::max(0, int(std::floor((r.x0 - g.ox) / g.res))), cx1 = std::min(g.w - 1, int(std::floor((r.x1 - g.ox) / g.res)));
      const int cy0 = std::max(0, int(std::floor((r.y0 - g.oy) / g.res))), cy1 = std::min(g.h - 1, int(std::floor((r.y1 - g.oy) / g.res)));
      for (int y = cy0; y <= cy1; ++y) for (int x = cx0; x <= cx1; ++x) if (masked[size_t(y) * g.w + x] >= kOccMin) masked[size_t(y) * g.w + x] = 0;
    }
    src = masked.data();
  }
  const WallGrid gm{src, g.w, g.h, g.res, g.ox, g.oy};
  const double th = wallAngle(gm);
  if (angle_out) *angle_out = th;
  if (std::abs(th) <= kAlignTol) return wallSegments(g, min_len, max_thick, overlap, ignore);   // 축에 맞는 지도: 예전과 같은 결과
  // 돌린 좌표 u = R(−θ)·p (지도 좌표 p 를 −θ 돌림). 원래 지도의 네 모서리를 돌려 새 격자 범위를 잡는다.
  const double c = std::cos(th), s = std::sin(th), res = g.res;
  double u0 = 1e300, v0 = 1e300, u1 = -1e300, v1 = -1e300;
  for (int k = 0; k < 4; ++k) {
    const double x = g.ox + ((k & 1) ? g.w : 0) * res, y = g.oy + ((k & 2) ? g.h : 0) * res;
    const double u = c * x + s * y, v = -s * x + c * y;
    u0 = std::min(u0, u); u1 = std::max(u1, u); v0 = std::min(v0, v); v1 = std::max(v1, v);
  }
  const int W = int(std::ceil((u1 - u0) / res)) + 1, H = int(std::ceil((v1 - v0) / res)) + 1;
  std::vector<int8_t> rot(size_t(W) * H, -1);
  // 새 칸 중심을 원래 격자로 돌려 놓고, 둘러싼 원래 칸 4 개(쌍선형 발자국) 중 하나라도 점유면 점유, 아니면 가장 가까운 칸 값.
  // 가장 가까운 칸만 보면 계단 모양의 대각선 벽이 돌린 격자에서 끊긴다.
  for (int j = 0; j < H; ++j) {
    const double v = v0 + (j + 0.5) * res;
    for (int i = 0; i < W; ++i) {
      const double u = u0 + (i + 0.5) * res;
      const double fx = (c * u - s * v - g.ox) / res - 0.5, fy = (s * u + c * v - g.oy) / res - 0.5;
      const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
      int8_t val = -1;
      bool occ = false;
      for (int q = 0; q < 4; ++q) {
        const int x = x0 + (q & 1), y = y0 + (q >> 1);
        if (x < 0 || y < 0 || x >= g.w || y >= g.h) continue;
        if (src[size_t(y) * g.w + x] >= kOccMin) occ = true;
      }
      if (occ) val = 100;
      else {
        const int x = int(std::lround(fx)), y = int(std::lround(fy));
        if (x >= 0 && y >= 0 && x < g.w && y < g.h) val = src[size_t(y) * g.w + x] >= 0 ? 0 : -1;
      }
      rot[size_t(j) * W + i] = val;
    }
  }
  const WallGrid gr{rot.data(), W, H, res, u0, v0};
  std::vector<WallSeg> all = wallSegments(gr, min_len, max_thick, overlap, nullptr);
  // 돌린 격자의 벽 가장자리는 계단 모양이라, 두꺼운 벽 한 덩어리 옆에 짧은 평행 조각이 따로 남는다(groupRuns 는 행마다 run 하나만 잇는다).
  // 더 긴 평행 선분과 수직 거리 ≤ max_thick/2 이고 그 구간 안(양끝 1 칸 여유)에 들어가면 버린다.
  std::vector<WallSeg> segs;
  {
    const double tol = max_thick / 2, slack = res;
    auto horiz = [](const WallSeg& s) { return s.ay == s.by; };
    for (size_t i = 0; i < all.size(); ++i) {
      const WallSeg& a = all[i];
      const bool ha = horiz(a);
      const double alen = ha ? std::abs(a.bx - a.ax) : std::abs(a.by - a.ay);
      const double a0 = ha ? std::min(a.ax, a.bx) : std::min(a.ay, a.by), a1 = ha ? std::max(a.ax, a.bx) : std::max(a.ay, a.by);
      bool shadow = false;
      for (size_t j = 0; j < all.size() && !shadow; ++j) {
        const WallSeg& b = all[j];
        if (j == i || horiz(b) != ha) continue;
        const double blen = ha ? std::abs(b.bx - b.ax) : std::abs(b.by - b.ay);
        if (blen < alen || (blen == alen && j > i)) continue;
        const double off = ha ? std::abs(a.ay - b.ay) : std::abs(a.ax - b.ax);
        const double b0 = ha ? std::min(b.ax, b.bx) : std::min(b.ay, b.by), b1 = ha ? std::max(b.ax, b.bx) : std::max(b.ay, b.by);
        shadow = off <= tol && a0 >= b0 - slack && a1 <= b1 + slack;
      }
      if (!shadow) segs.push_back(a);
    }
  }
  for (WallSeg& w : segs) {   // p = R(θ)·u
    const double ax = c * w.ax - s * w.ay, ay = s * w.ax + c * w.ay, bx = c * w.bx - s * w.by, by = s * w.bx + c * w.by;
    w = {ax, ay, bx, by};
  }
  return segs;
}

void rayDistances(const WallGrid& g, const double pose[3], float* out, int n, double max_range) {
  const double x = pose[0], y = pose[1], yaw = pose[2];
  for (int i = 0; i < n; ++i) {
    const double th = yaw + 2.0 * M_PI * i / n;
    const double dx = std::cos(th), dy = std::sin(th);
    // 격자 DDA (Amanatides–Woo): 칸 경계마다 한 번 — 샘플링 없이 칸 하나씩 정확히
    double fx = (x - g.ox) / g.res, fy = (y - g.oy) / g.res;
    int cx = int(std::floor(fx)), cy = int(std::floor(fy));
    const int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
    const double inf = std::numeric_limits<double>::infinity();
    const double tdx = dx != 0 ? std::abs(g.res / dx) : inf, tdy = dy != 0 ? std::abs(g.res / dy) : inf;
    double tx = dx != 0 ? ((sx > 0 ? cx + 1 - fx : fx - cx) * g.res) / std::abs(dx) : inf;
    double ty = dy != 0 ? ((sy > 0 ? cy + 1 - fy : fy - cy) * g.res) / std::abs(dy) : inf;
    float best = float(max_range);
    double t = 0;
    while (t <= max_range) {
      if (cx >= 0 && cx < g.w && cy >= 0 && cy < g.h && g.cells[size_t(cy) * g.w + cx] >= kOccMin) {
        best = float(std::max(t, 0.0));
        break;
      }
      if (tx < ty) { t = tx; tx += tdx; cx += sx; } else { t = ty; ty += tdy; cy += sy; }
    }
    out[i] = best;
  }
}

int segmentsRobotFrame(const std::vector<WallSeg>& segs, const double pose[3], int k, float* out, float* dist) {
  std::fill(out, out + 4 * k, 0.f);
  std::fill(dist, dist + k, std::numeric_limits<float>::infinity());
  if (segs.empty()) return 0;
  const double c = std::cos(-pose[2]), s = std::sin(-pose[2]);
  struct Cand { double d; float a[4]; };
  std::vector<Cand> cand(segs.size());
  for (size_t i = 0; i < segs.size(); ++i) {
    const WallSeg& w = segs[i];
    double ax = w.ax - pose[0], ay = w.ay - pose[1], bx = w.bx - pose[0], by = w.by - pose[1];
    double rax = c * ax - s * ay, ray = s * ax + c * ay, rbx = c * bx - s * by, rby = s * bx + c * by;
    double abx = rbx - rax, aby = rby - ray;
    double t = std::clamp(-(rax * abx + ray * aby) / std::max(abx * abx + aby * aby, 1e-12), 0.0, 1.0);
    cand[i].d = std::hypot(rax + abx * t, ray + aby * t);
    cand[i].a[0] = float(rax); cand[i].a[1] = float(ray); cand[i].a[2] = float(rbx); cand[i].a[3] = float(rby);
  }
  const int m = std::min<int>(k, int(cand.size()));
  std::partial_sort(cand.begin(), cand.begin() + m, cand.end(), [](const Cand& a, const Cand& b) { return a.d < b.d; });
  for (int j = 0; j < m; ++j) {
    std::copy(cand[j].a, cand[j].a + 4, out + 4 * j);
    dist[j] = float(cand[j].d);
  }
  return m;
}

void wallStateVector(const WallGrid& g, const std::vector<WallSeg>& segs, const double pose[3], float* out) {
  float rays[kSectors];
  rayDistances(g, pose, rays, kSectors, kMaxRange);
  for (int i = 0; i < kSectors; ++i) out[i] = float(rays[i] / kMaxRange);
  float seg[kSegments * 4], dist[kSegments];
  segmentsRobotFrame(segs, pose, kSegments, seg, dist);
  for (int j = 0; j < kSegments; ++j) {
    float* o = out + kSectors + 5 * j;
    const bool valid = std::isfinite(dist[j]) && dist[j] <= kMaxRange * 1.5;
    for (int q = 0; q < 4; ++q) o[q] = valid ? std::clamp(float(seg[4 * j + q] / kMaxRange), -1.f, 1.f) : 0.f;
    o[4] = valid ? 1.f : 0.f;
  }
}

}  // namespace scenemap
