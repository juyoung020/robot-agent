// 살아 있는 장면 그래프(include/scenemap/sgraph.hpp).
#include "scenemap/sgraph.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace scenemap {
static_assert(sizeof(GEdge) == sizeof(sm_gedge), "GEdge == sm_gedge");
namespace {
using Clk = std::chrono::steady_clock;
double usSince(Clk::time_point t) { return std::chrono::duration<double, std::micro>(Clk::now() - t).count(); }

constexpr int kObjLayer = 2, kPlaceLayer = 3, kRoomLayer = 4, kBuildLayer = 5, kAgentPart = 'a';

int layerOrder(const GNode& n) {
  if (n.layer == kObjLayer) return n.partition == kAgentPart ? 1 : 0;
  if (n.layer == kPlaceLayer) return 2;
  if (n.layer == kRoomLayer) return 3;
  return 4;
}

// 1D 제곱 거리 변환(Felzenszwalb–Huttenlocher). f: 0 = 막힘, kInf = 빈칸(double — 큰 값끼리 빼도 정확)
constexpr double kInf = 1e12;
void dt1(const double* f, int n, double* d, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -1e300;
  z[1] = 1e300;
  for (int q = 1; q < n; ++q) {
    double s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) / (2.0 * (q - v[k]));
    while (s <= z[k]) {
      --k;
      s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) / (2.0 * (q - v[k]));
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e300;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const double dq = double(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}
}  // namespace

void SceneGraph::reset() { *this = SceneGraph(p_); }

GNode& SceneGraph::upsert(uint64_t id, int layer, int partition) {
  auto it = nodes_.find(id);
  if (it != nodes_.end()) return it->second;
  GNode& n = nodes_[id];
  n.id = id;
  n.layer = layer;
  n.partition = partition;
  changed_ = true;
  return n;
}

void SceneGraph::eraseEdgesOf(uint64_t id, int rel_mask) {
  auto it = adj_.find(id);
  if (it == adj_.end()) return;
  std::vector<GEdge>& v = it->second;
  for (size_t k = 0; k < v.size();) {
    const GEdge e = v[k];
    if (!((rel_mask >> e.rel) & 1)) { ++k; continue; }
    const uint64_t o = e.a == id ? e.b : e.a;
    auto jt = adj_.find(o);
    if (jt != adj_.end()) {
      auto& w = jt->second;
      for (size_t m = 0; m < w.size(); ++m)
        if (w[m].a == e.a && w[m].b == e.b && w[m].rel == e.rel) { w[m] = w.back(); w.pop_back(); break; }
    }
    v[k] = v.back();
    v.pop_back();
    changed_ = true;
  }
}

void SceneGraph::erase(uint64_t id) {
  eraseEdgesOf(id, ~0);
  adj_.erase(id);
  if (nodes_.erase(id)) changed_ = true;
}

void SceneGraph::addEdge(uint64_t a, uint64_t b, float w, int rel, const float* pos) {
  if (a == b || !nodes_.count(a) || !nodes_.count(b)) return;
  auto& va = adj_[a];
  for (const GEdge& e : va)
    if (e.rel == rel && ((e.a == a && e.b == b) || (e.a == b && e.b == a))) return;
  GEdge e;
  e.a = a; e.b = b; e.weight = w; e.rel = rel;
  if (pos) { e.pos[0] = pos[0]; e.pos[1] = pos[1]; }
  va.push_back(e);
  adj_[b].push_back(e);
  changed_ = true;
}

uint64_t SceneGraph::nearestPlace(double x, double y, double rmax) const {
  uint64_t best = 0;
  double bd = rmax * rmax;
  for (const auto& [id, n] : nodes_) {
    if (n.layer != kPlaceLayer) continue;
    const double d = (n.pos[0] - x) * (n.pos[0] - x) + (n.pos[1] - y) * (n.pos[1] - y);
    if (d < bd) { bd = d; best = id; }
  }
  return best;
}

void SceneGraph::updateObjects(const std::vector<ObjIn>& objs) {
  const auto t0 = Clk::now();
  bool any = false;
  std::vector<uint64_t> keep;
  keep.reserve(objs.size());
  for (const ObjIn& o : objs) {
    const uint64_t id = nodeSym('O', o.id);
    keep.push_back(id);
    const bool isnew = !nodes_.count(id);
    GNode& n = upsert(id, kObjLayer, 0);
    double dm = 0;
    for (int k = 0; k < 3; ++k)
      dm = std::max({dm, std::fabs(n.pos[k] - o.pos[k]), std::fabs(n.lo[k] - o.lo[k]), std::fabs(n.hi[k] - o.hi[k])});
    if (isnew || dm > 1e-3 || n.state != o.state || n.movable != o.movable || n.name != o.name) {
      for (int k = 0; k < 3; ++k) { n.pos[k] = o.pos[k]; n.lo[k] = o.lo[k]; n.hi[k] = o.hi[k]; }
      n.state = o.state;
      n.movable = o.movable;
      n.name = o.name;
      ++n.ver;
      any = true;
      changed_ = true;
    }
  }
  // 없어진 물체
  std::vector<uint64_t> gone;
  for (const auto& [id, n] : nodes_)
    if (n.layer == kObjLayer && n.partition == 0 && std::find(keep.begin(), keep.end(), id) == keep.end()) gone.push_back(id);
  for (uint64_t id : gone) erase(id), any = true;
  if (any) {
    // 물체끼리 관계(on/in/near 전치사 규칙)는 만들지 않는다(Map_Vla): 물체마다 위치·상자가 메타데이터로 남으므로 '위에 있다/안에 있다'는
    // 소비자(LLM)가 추론한다. 그래프에 남는 물체 연결은 부모 쪽뿐이다: 방 → 물체(updateRooms), place → 물체(linkObjectsToPlaces).
    linkObjectsToPlaces();
  }
  us_objects = usSince(t0);
}

void SceneGraph::linkObjectsToPlaces() {
  for (auto& [id, n] : nodes_) {
    if (n.layer != kObjLayer || n.partition != 0) continue;
    // 물체 → place 변만 다시(방 → 물체 변은 그대로)
    auto it = adj_.find(id);
    if (it != adj_.end()) {
      std::vector<uint64_t> drop;
      for (const GEdge& e : it->second)
        if (e.rel == kRelGeneric && symChar(e.b == id ? e.a : e.b) == 'p') drop.push_back(e.b == id ? e.a : e.b);
      for (uint64_t o : drop) {
        auto& v = adj_[id];
        for (size_t k = 0; k < v.size(); ++k)
          if (v[k].rel == kRelGeneric && (v[k].a == o || v[k].b == o)) { v[k] = v.back(); v.pop_back(); break; }
        auto& w = adj_[o];
        for (size_t k = 0; k < w.size(); ++k)
          if (w[k].rel == kRelGeneric && (w[k].a == id || w[k].b == id)) { w[k] = w.back(); w.pop_back(); break; }
        changed_ = true;
      }
    }
    if (const uint64_t pl = nearestPlace(n.pos[0], n.pos[1], 3.0)) addEdge(pl, id, 1.f, kRelGeneric);
  }
}

void SceneGraph::updateAgent(double stamp, const Pose2& P) {
  if (have_agent_ && std::hypot(P.x - last_agent_.x, P.y - last_agent_.y) < p_.agent_xy &&
      std::fabs(wrapAngle(P.th - last_agent_.th)) < p_.agent_yaw && stamp - last_agent_t_ < p_.agent_s)
    return;
  const uint64_t id = nodeSym('a', next_agent_++);
  GNode& n = upsert(id, kObjLayer, kAgentPart);
  n.pos[0] = n.lo[0] = n.hi[0] = P.x;
  n.pos[1] = n.lo[1] = n.hi[1] = P.y;
  n.yaw = P.th;
  n.stamp = stamp;
  n.name = "robot";
  ++n.ver;
  if (next_agent_ >= 2) addEdge(nodeSym('a', next_agent_ - 2), id, 1.f, kRelAgent);
  if (const uint64_t pl = nearestPlace(P.x, P.y, 2.0)) addEdge(pl, id, 1.f, kRelGeneric);
  last_agent_ = P;
  last_agent_t_ = stamp;
  have_agent_ = true;
}

void SceneGraph::linkAgentsToPlaces(double x0, double y0, double x1, double y1) {
  for (auto& [id, n] : nodes_) {
    if (n.partition != kAgentPart || n.pos[0] < x0 || n.pos[0] > x1 || n.pos[1] < y0 || n.pos[1] > y1) continue;
    auto& v = adj_[id];
    for (size_t k = 0; k < v.size();) {
      const uint64_t o = v[k].a == id ? v[k].b : v[k].a;
      if (v[k].rel == kRelGeneric && symChar(o) == 'p') {
        auto& w = adj_[o];
        for (size_t m = 0; m < w.size(); ++m)
          if (w[m].rel == kRelGeneric && (w[m].a == id || w[m].b == id)) { w[m] = w.back(); w.pop_back(); break; }
        v[k] = v.back();
        v.pop_back();
        changed_ = true;
      } else {
        ++k;
      }
    }
    if (const uint64_t pl = nearestPlace(n.pos[0], n.pos[1], 2.0)) addEdge(pl, id, 1.f, kRelGeneric);
  }
}

void SceneGraph::updatePlaces(const int8_t* cells, int w, int h, int gx0, int gy0, double res, double stamp, const int* dirty,
                              bool force) {
  if (dirty) {   // 다음 계산까지 쌓아 둠
    if (!pending_dirty_) { std::copy(dirty, dirty + 4, pd_); pending_dirty_ = true; }
    else { pd_[0] = std::min(pd_[0], dirty[0]); pd_[1] = std::min(pd_[1], dirty[1]); pd_[2] = std::max(pd_[2], dirty[2]); pd_[3] = std::max(pd_[3], dirty[3]); }
  }
  if (!cells || w <= 0 || h <= 0) return;
  const bool reshaped = w != cw_ || h != ch_ || gx0 != cgx0_ || gy0 != cgy0_ || std::fabs(res - cres_) > 1e-9;
  if (!force && !reshaped && (!pending_dirty_ || stamp - last_places_ < p_.period_s)) return;
  const auto t0 = Clk::now();
  last_places_ = stamp;
  // 격자 사본(시야 검사·방 배정) — 모양이 바뀌면 전부 다시
  if (reshaped) force = true;
  cells_.assign(cells, cells + size_t(w) * h);
  cw_ = w; ch_ = h; cgx0_ = gx0; cgy0_ = gy0; cres_ = res;
  // 창(이 격자 칸): 바뀐 상자 + 여백
  const int mg = int(std::ceil(p_.win_margin / res)), cap = int(std::ceil(p_.max_clear / res)) + 1;
  int wx0 = 0, wy0 = 0, wx1 = w - 1, wy1 = h - 1;
  if (!force && pending_dirty_) {
    wx0 = std::max(0, pd_[0] - gx0 - mg); wy0 = std::max(0, pd_[1] - gy0 - mg);
    wx1 = std::min(w - 1, pd_[2] - gx0 + mg); wy1 = std::min(h - 1, pd_[3] - gy0 + mg);
  }
  pending_dirty_ = false;
  if (wx0 > wx1 || wy0 > wy1) return;
  // 창을 그 안 빈칸 상자로 줄임(모름·점유만인 곳은 place 가 없음 — 대부분이 모름일 때 거리 변환이 작아짐)
  {
    int fx0 = wx1 + 1, fy0 = wy1 + 1, fx1 = wx0 - 1, fy1 = wy0 - 1;
    for (int y = wy0; y <= wy1; ++y) {
      const int8_t* row = cells + size_t(y) * w;
      for (int x = wx0; x <= wx1; ++x)
        if (row[x] >= 0 && row[x] < 50) { fx0 = std::min(fx0, x); fx1 = std::max(fx1, x); fy0 = std::min(fy0, y); fy1 = std::max(fy1, y); }
    }
    if (fx0 <= fx1) {
      // place 지우기는 원래 창 전체(빈칸이 없어진 곳의 옛 place 도 지움), 거리 변환·후보는 빈칸 상자만
      win_[0] = wx0; win_[1] = wy0; win_[2] = wx1; win_[3] = wy1;
      wx0 = fx0; wy0 = fy0; wx1 = fx1; wy1 = fy1;
    } else {
      win_[0] = wx0; win_[1] = wy0; win_[2] = wx1; win_[3] = wy1;
      wx1 = wx0 - 1;   // 빈칸 없음: 옛 place 만 지움
    }
  }
  // 거리 변환 영역 = 창 + 상한(창 안 값이 정확하게). 막힘 = 점유·모름·격자 밖, 모름 거리는 따로
  const int rx0 = wx0 - cap, ry0 = wy0 - cap, rx1 = wx1 + cap, ry1 = wy1 + cap;
  const int RW = rx1 - rx0 + 1, RH = ry1 - ry0 + 1;
  edt_.assign(size_t(RW) * RH, kInf);
  for (int y = ry0; y <= ry1; ++y)
    for (int x = rx0; x <= rx1; ++x) {
      const bool in = x >= 0 && y >= 0 && x < w && y < h;
      const int v = in ? cells[size_t(y) * w + x] : -1;
      const size_t i = size_t(y - ry0) * RW + (x - rx0);
      if (v < 0 || v >= 50) edt_[i] = 0;
    }
  const int nmax = std::max(RW, RH);
  z_.resize(nmax + 1);
  v_.resize(nmax);
  std::vector<double> col(nmax), out(nmax);
  for (std::vector<double>* E : {&edt_}) {
    double* D = E->data();
    for (int x = 0; x < RW; ++x) {
      for (int y = 0; y < RH; ++y) col[y] = D[size_t(y) * RW + x];
      dt1(col.data(), RH, out.data(), v_.data(), z_.data());
      for (int y = 0; y < RH; ++y) D[size_t(y) * RW + x] = out[y];
    }
    for (int y = 0; y < RH; ++y) {
      double* row = D + size_t(y) * RW;
      dt1(row, RW, out.data(), v_.data(), z_.data());
      std::copy(out.begin(), out.begin() + RW, row);
    }
  }
  // 여유(m) 한 번만: sqrt 를 칸마다 한 번
  clr_.resize(size_t(RW) * RH);
  for (size_t i = 0; i < clr_.size(); ++i) clr_[i] = float(std::sqrt(edt_[i]) * res - 0.5 * res);
  auto clr = [&](int x, int y) { return clr_[size_t(y - ry0) * RW + (x - rx0)]; };
  // frontier: 여유 + tol 원 위 16 점 중 모름(격자 밖 포함)이 있으면 — 가장 가까운 막힘이 모름 쪽
  auto frontier = [&](int x, int y, float c) {
    const double r = (c + p_.frontier_tol) / res + 0.5;
    for (int k = 0; k < 16; ++k) {
      const double a = k * (M_PI / 8);
      const int qx = x + int(std::lround(r * std::cos(a))), qy = y + int(std::lround(r * std::sin(a)));
      if (qx < 0 || qy < 0 || qx >= w || qy >= h || cells[size_t(qy) * w + qx] < 0) return true;
    }
    return false;
  };
  const double mx0 = (gx0 + win_[0]) * res, my0 = (gy0 + win_[1]) * res, mx1 = (gx0 + win_[2] + 1) * res, my1 = (gy0 + win_[3] + 1) * res;
  // 창 안 옛 place 지우기(자리는 id 다시 쓰기용으로 남김)
  std::vector<std::pair<uint64_t, std::array<double, 2>>> old;
  std::vector<uint64_t> rm;
  for (const auto& [id, n] : nodes_)
    if (n.layer == kPlaceLayer && n.pos[0] >= mx0 && n.pos[0] < mx1 && n.pos[1] >= my0 && n.pos[1] < my1) {
      old.push_back({id, {n.pos[0], n.pos[1]}});
      rm.push_back(id);
    }
  for (uint64_t id : rm) erase(id);
  // 능선 후보: 빈칸, 여유 ≥ min_clear, 네 방향 중 하나에서 양 옆보다 작지 않음
  struct Cand { float c; int x, y; };
  std::vector<Cand> cand;
  const float cmax = float(p_.max_clear);
  for (int y = wy0; y <= wy1; ++y)
    for (int x = wx0; x <= wx1; ++x) {
      const int v = cells[size_t(y) * w + x];
      if (v < 0 || v >= 50) continue;
      const float c = std::min(clr(x, y), cmax);
      if (c < p_.min_clear) continue;
      static const int DX[4] = {1, 0, 1, 1}, DY[4] = {0, 1, 1, -1};
      bool ridge = false;
      for (int k = 0; k < 4 && !ridge; ++k) {
        const float a = std::min(clr(x + DX[k], y + DY[k]), cmax), b = std::min(clr(x - DX[k], y - DY[k]), cmax);
        ridge = c >= a && c >= b;
      }
      if (ridge) cand.push_back({c, x, y});
    }
  std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
    return a.c != b.c ? a.c > b.c : (a.y != b.y ? a.y < b.y : a.x < b.x);
  });
  // 성글게 고르기: 1 m 통 해시로 이웃 place 찾기(창 밖 place 포함)
  std::unordered_map<int64_t, std::vector<std::array<double, 3>>> bucket;   // x, y, 간격
  auto bkey = [](double x, double y) { return (int64_t(std::floor(x)) << 32) ^ (int64_t(std::floor(y)) & 0xffffffff); };
  auto spacingOf = [&](double c) { return std::clamp(1.5 * c, p_.spacing_min, p_.spacing_max); };
  for (const auto& [id, n] : nodes_)
    if (n.layer == kPlaceLayer) bucket[bkey(n.pos[0], n.pos[1])].push_back({n.pos[0], n.pos[1], spacingOf(n.clearance)});
  std::vector<uint8_t> reused(old.size(), 0);
  std::vector<uint64_t> added;
  for (const Cand& c : cand) {
    const double px = (gx0 + c.x + 0.5) * res, py = (gy0 + c.y + 0.5) * res, sp = spacingOf(c.c);
    bool ok = true;
    const int r = int(std::ceil(p_.spacing_max));
    for (int by = -r; by <= r && ok; ++by)
      for (int bx = -r; bx <= r && ok; ++bx) {
        auto it = bucket.find(bkey(px + bx, py + by));
        if (it == bucket.end()) continue;
        for (const auto& q : it->second) {
          const double s = std::min(sp, q[2]);
          if ((q[0] - px) * (q[0] - px) + (q[1] - py) * (q[1] - py) < s * s) { ok = false; break; }
        }
      }
    if (!ok) continue;
    bucket[bkey(px, py)].push_back({px, py, sp});
    // id: 0.3 m 안 옛 place 가 있으면 그것
    uint64_t id = 0;
    double bd = 0.3 * 0.3;
    int bi = -1;
    for (size_t k = 0; k < old.size(); ++k) {
      if (reused[k]) continue;
      const double d = (old[k].second[0] - px) * (old[k].second[0] - px) + (old[k].second[1] - py) * (old[k].second[1] - py);
      if (d < bd) { bd = d; bi = int(k); }
    }
    if (bi >= 0) { reused[bi] = 1; id = old[bi].first; }
    else id = nodeSym('p', next_place_++);
    GNode& n = upsert(id, kPlaceLayer, 0);
    n.pos[0] = n.lo[0] = n.hi[0] = px;
    n.pos[1] = n.lo[1] = n.hi[1] = py;
    n.clearance = c.c;
    n.state = frontier(c.x, c.y, c.c) ? 1 : 0;
    n.stamp = stamp;
    ++n.ver;
    added.push_back(id);
  }
  // 변: 창 + edge_r 안 place 마다 다시
  const double er = p_.edge_r;
  std::vector<GNode*> act, all;
  for (auto& [id, n] : nodes_) {
    if (n.layer != kPlaceLayer) continue;
    all.push_back(&n);
    if (n.pos[0] >= mx0 - er && n.pos[0] < mx1 + er && n.pos[1] >= my0 - er && n.pos[1] < my1 + er) act.push_back(&n);
  }
  for (GNode* n : act) eraseEdgesOf(n->id, 1 << kRelPlace);
  // 시야: 칸 직선(Bresenham) 위 가장 작은 여유. 창 거리 변환 밖은 칸 값만(빈칸이면 여유 = 상한으로 봄 — 창 안 끝점이 정함)
  auto los = [&](const GNode& a, const GNode& b, float* minc) {
    int x = int(std::floor(a.pos[0] / res)) - gx0, y = int(std::floor(a.pos[1] / res)) - gy0;
    const int x1 = int(std::floor(b.pos[0] / res)) - gx0, y1 = int(std::floor(b.pos[1] / res)) - gy0;
    const int dx = std::abs(x1 - x), dy = -std::abs(y1 - y), sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1;
    int err = dx + dy;
    float m = std::min(a.clearance, b.clearance);
    for (;;) {
      if (x < 0 || y < 0 || x >= w || y >= h) return false;
      const int v = cells[size_t(y) * w + x];
      if (v < 0 || v >= 50) return false;
      if (x >= rx0 && x <= rx1 && y >= ry0 && y <= ry1) m = std::min(m, clr(x, y));
      if (x == x1 && y == y1) break;
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
    *minc = m;
    return m >= p_.edge_min_clear;
  };
  for (GNode* a : act) {
    std::vector<std::pair<double, GNode*>> nb;
    for (GNode* b : all) {
      if (a == b) continue;
      const double d = std::hypot(a->pos[0] - b->pos[0], a->pos[1] - b->pos[1]);
      if (d <= er) nb.push_back({d, b});
    }
    std::sort(nb.begin(), nb.end(), [](auto& x, auto& y) { return x.first < y.first; });
    int k = 0;
    for (auto& [d, b] : nb) {
      if (k >= p_.edge_k) break;
      float m;
      if (!los(*a, *b, &m)) continue;
      addEdge(a->id, b->id, m, kRelPlace);
      ++k;
    }
  }
  linkObjectsToPlaces();
  linkAgentsToPlaces(mx0 - er, my0 - er, mx1 + er, my1 + er);
  linkPlacesToRooms();
  ++n_place_updates;
  us_places = usSince(t0);
}

void SceneGraph::linkPlacesToRooms() {
  if (!rooms_) return;
  for (auto& [id, n] : nodes_) {
    if (n.layer != kPlaceLayer) continue;
    const uint32_t rid = rooms_->at(n.pos[0], n.pos[1]);
    const uint64_t want = rid ? nodeSym('R', rid) : 0;
    uint64_t have = 0;
    for (const GEdge& e : adj_[id])
      if (e.rel == kRelGeneric && symChar(e.a) == 'R') have = e.a;
    if (have == want) continue;
    if (have) {
      auto& v = adj_[id];
      for (size_t k = 0; k < v.size(); ++k)
        if (v[k].rel == kRelGeneric && v[k].a == have) { v[k] = v.back(); v.pop_back(); break; }
      auto& r = adj_[have];
      for (size_t k = 0; k < r.size(); ++k)
        if (r[k].rel == kRelGeneric && r[k].b == id) { r[k] = r.back(); r.pop_back(); break; }
      changed_ = true;
    }
    if (want && nodes_.count(want)) addEdge(want, id, 1.f, kRelGeneric);
  }
}

void SceneGraph::updateRooms(const std::shared_ptr<const RoomSeg>& rs, const std::vector<std::pair<uint32_t, uint32_t>>& obj_room,
                             const std::vector<std::string>& names) {
  const bool new_seg = rs != rooms_;
  if (new_seg) {
    rooms_ = rs;
    std::vector<uint64_t> rm;
    for (const auto& [id, n] : nodes_)
      if (n.layer == kRoomLayer || n.layer == kBuildLayer) rm.push_back(id);
    for (uint64_t id : rm) erase(id);
    // Map_Vla: three layers only (OBJECTS, PLACES, ROOMS). The single BUILDINGS node 'B0' (and building -> room edges) is no longer made.
    if (rs && !rs->rooms.empty()) {
      for (size_t k = 0; k < rs->rooms.size(); ++k) {
        const RoomGeom& r = rs->rooms[k];
        GNode& n = upsert(nodeSym('R', r.id), kRoomLayer, 0);
        n.pos[0] = r.centroid[0]; n.pos[1] = r.centroid[1]; n.pos[2] = 0;
        n.lo[0] = r.bmin[0]; n.lo[1] = r.bmin[1]; n.hi[0] = r.bmax[0]; n.hi[1] = r.bmax[1];
        n.clearance = float(r.max_clear);
        n.stamp = r.area_m2;   // 넓이(저장 메타데이터)
        n.name = k < names.size() ? names[k] : "room " + std::to_string(r.id);
        ++n.ver;
      }
      for (const RoomDoor& d : rs->doors) {
        const float pos[2] = {float(d.pos[0]), float(d.pos[1])};
        addEdge(nodeSym('R', d.a), nodeSym('R', d.b), float(d.width), kRelDoor, pos);
      }
    }
    linkPlacesToRooms();
  } else if (rs) {   // 같은 나눔: 이름만 바뀌었을 수 있음
    for (size_t k = 0; k < rs->rooms.size() && k < names.size(); ++k) {
      auto it = nodes_.find(nodeSym('R', rs->rooms[k].id));
      if (it != nodes_.end() && it->second.name != names[k]) { it->second.name = names[k]; ++it->second.ver; changed_ = true; }
    }
  }
  // 방 → 물체(바뀐 것만)
  for (const auto& [oid, rid] : obj_room) {
    const uint64_t o = nodeSym('O', oid);
    if (!nodes_.count(o)) continue;
    uint64_t have = 0;
    for (const GEdge& e : adj_[o])
      if (e.rel == kRelGeneric && symChar(e.a) == 'R') have = e.a;
    const uint64_t want = rid ? nodeSym('R', rid) : 0;
    if (have == want) continue;
    if (have) {
      auto& v = adj_[o];
      for (size_t k = 0; k < v.size(); ++k)
        if (v[k].rel == kRelGeneric && v[k].a == have) { v[k] = v.back(); v.pop_back(); break; }
      auto& r = adj_[have];
      for (size_t k = 0; k < r.size(); ++k)
        if (r[k].rel == kRelGeneric && r[k].b == o) { r[k] = r.back(); r.pop_back(); break; }
      changed_ = true;
    }
    if (want) addEdge(want, o, 1.f, kRelGeneric);
  }
}

std::shared_ptr<const GraphView> SceneGraph::publish() {
  if (!changed_ && view_) return view_;
  const auto t0 = Clk::now();
  auto v = std::make_shared<GraphView>();
  v->nodes.reserve(nodes_.size());
  for (const auto& [id, n] : nodes_) v->nodes.push_back(n);
  std::sort(v->nodes.begin(), v->nodes.end(), [](const GNode& a, const GNode& b) {
    const int la = layerOrder(a), lb = layerOrder(b);
    return la != lb ? la < lb : a.id < b.id;
  });
  int li = 0;
  for (size_t i = 0; i <= v->nodes.size(); ++i) {
    const int l = i < v->nodes.size() ? layerOrder(v->nodes[i]) : 5;
    while (li <= l) v->layer_off[li++] = int32_t(i);
  }
  while (li < 6) v->layer_off[li++] = int32_t(v->nodes.size());
  v->index.reserve(v->nodes.size());
  for (size_t i = 0; i < v->nodes.size(); ++i) v->index[v->nodes[i].id] = uint32_t(i);
  for (const auto& [id, es] : adj_)
    for (const GEdge& e : es)
      if (e.a == id) v->edges.push_back(e);
  std::sort(v->edges.begin(), v->edges.end(), [](const GEdge& a, const GEdge& b) {
    return a.a != b.a ? a.a < b.a : (a.b != b.b ? a.b < b.b : a.rel < b.rel);
  });
  // CSR
  v->adj_off.assign(v->nodes.size() + 1, 0);
  for (const GEdge& e : v->edges) {
    auto ia = v->index.find(e.a), ib = v->index.find(e.b);
    if (ia != v->index.end()) ++v->adj_off[ia->second + 1];
    if (ib != v->index.end()) ++v->adj_off[ib->second + 1];
  }
  for (size_t i = 1; i < v->adj_off.size(); ++i) v->adj_off[i] += v->adj_off[i - 1];
  v->adj.resize(v->adj_off.back());
  std::vector<uint32_t> fill(v->adj_off.begin(), v->adj_off.end() - 1);
  for (uint32_t k = 0; k < v->edges.size(); ++k) {
    const GEdge& e = v->edges[k];
    auto ia = v->index.find(e.a), ib = v->index.find(e.b);
    if (ia != v->index.end()) v->adj[fill[ia->second]++] = k;
    if (ib != v->index.end()) v->adj[fill[ib->second]++] = k;
  }
  v->cn.resize(v->nodes.size());
  for (size_t i = 0; i < v->nodes.size(); ++i) {
    const GNode& n = v->nodes[i];
    sm_gnode& c = v->cn[i];
    c = sm_gnode{};
    c.id = n.id; c.layer = n.layer; c.partition = n.partition; c.group = layerOrder(n);
    for (int k = 0; k < 3; ++k) { c.pos[k] = n.pos[k]; c.bbox_min[k] = n.lo[k]; c.bbox_max[k] = n.hi[k]; }
    c.name = n.name.c_str();
    c.state = n.state; c.clearance = n.clearance; c.yaw = n.yaw; c.stamp = n.stamp; c.movable = n.movable;
  }
  v->version = ++version_;
  view_ = std::move(v);
  changed_ = false;
  us_publish = usSince(t0);
  return view_;
}

}  // namespace scenemap
