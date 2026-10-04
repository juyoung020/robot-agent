#include "scenemap/objmap.hpp"

#include "da/merge.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
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
  bool trunc;                     // 상자가 영상 가장자리에 닿음(잘려 보임 — 중심을 움직임 판단에 안 씀)
  float score;
  double pos[3], ext[3], lo[3], hi[3];
  int n;
};

}  // namespace

void ObjectMap::event(double t, const MapObject& o, int kind) {
  ev_.push_back(ObjEvent{t, o.id, kind, {o.pos[0], o.pos[1], o.pos[2]}});
  static const bool log_ev = std::getenv("SM_OBJ_LOG") != nullptr;
  if (log_ev)
    std::fprintf(stderr, "[ev] t=%.1f id=%u kind=%d cls=%d pos=%.2f %.2f %.2f ext=%.2f %.2f %.2f nobs=%u\n", t, o.id, kind, o.cls,
                 o.pos[0], o.pos[1], o.pos[2], o.ext[0], o.ext[1], o.ext[2], o.n_obs);
}

namespace {
// 점이 로봇 팔 캡슐(map 기준, 반경 + pad) 안인가
bool inSelfCaps(const Capsule* caps, int n, double pad, double px, double py, double pz) {
  for (int c = 0; c < n; ++c) {
    const Capsule& k = caps[c];
    const double ab[3] = {double(k.b[0]) - k.a[0], double(k.b[1]) - k.a[1], double(k.b[2]) - k.a[2]};
    const double ap[3] = {px - k.a[0], py - k.a[1], pz - k.a[2]};
    const double l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    double u = l2 > 1e-12 ? (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / l2 : 0.0;
    u = std::clamp(u, 0.0, 1.0);
    const double d[3] = {ap[0] - u * ab[0], ap[1] - u * ab[1], ap[2] - u * ab[2]};
    const double r = k.r + pad;
    if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < r * r) return true;
  }
  return false;
}
}  // namespace

void ObjectMap::envOverrides(ObjParams* p) {
  const char* e = std::getenv("SM_OBJ_PARAMS");
  if (!e || !*e) return;
  struct K { const char* n; double* d; int* i; bool* b; };
  const K keys[] = {{"floor_h", &p->floor_h, nullptr, nullptr},          {"name_vote", nullptr, nullptr, &p->name_vote},
                    {"name_share", &p->name_share, nullptr, nullptr},    {"name_switch", &p->name_switch, nullptr, nullptr},
                    {"name_merge_iou", &p->name_merge_iou, nullptr, nullptr}, {"frag_overlap", &p->frag_overlap, nullptr, nullptr},
                    {"absent_samples", nullptr, &p->absent_samples, nullptr}, {"absent_vis", &p->absent_vis, nullptr, nullptr},
                    {"absent_min_px", &p->absent_min_px, nullptr, nullptr}, {"absent_det_k", &p->absent_det_k, nullptr, nullptr},
                    {"gone_misses", nullptr, &p->gone_misses, nullptr},  {"gone_min_s", &p->gone_min_s, nullptr, nullptr},
                    {"gone_misses_big", nullptr, &p->gone_misses_big, nullptr}, {"gone_min_s_big", &p->gone_min_s_big, nullptr, nullptr},
                    {"gone_view_d", &p->gone_view_d, nullptr, nullptr},  {"gone_eps", &p->gone_eps, nullptr, nullptr},  {"gone_step_d", &p->gone_step_d, nullptr, nullptr},
                    {"gone_step_deg", &p->gone_step_deg, nullptr, nullptr},  {"spurious_obs", nullptr, &p->spurious_obs, nullptr},
                    {"link_v", &p->link_v, nullptr, nullptr},            {"link_d0", &p->link_d0, nullptr, nullptr},
                    {"link_max_d", &p->link_max_d, nullptr, nullptr},    {"link_min_obs", nullptr, &p->link_min_obs, nullptr},
                    {"link_wait_s", &p->link_wait_s, nullptr, nullptr},  {"link_window_s", &p->link_window_s, nullptr, nullptr},  {"link_view_gap_s", &p->link_view_gap_s, nullptr, nullptr},
                    {"view_cell", &p->view_cell, nullptr, nullptr},      {"confirm", nullptr, &p->confirm, nullptr},  {"move_v", &p->move_v, nullptr, nullptr},
                    {"move_n", nullptr, &p->move_n, nullptr},  {"move_min_d", &p->move_min_d, nullptr, nullptr},  {"move_max_cam_w", &p->move_max_cam_w, nullptr, nullptr},
                    {"da_min", &p->da_min, nullptr, nullptr},            {"da_k", &p->da_k, nullptr, nullptr},
                    {"merge_overlap", &p->merge_overlap, nullptr, nullptr},
                    {"grasp_check", nullptr, nullptr, &p->grasp_check},  {"self_pad", &p->self_pad, nullptr, nullptr},
                    {"self_mask", nullptr, nullptr, &p->self_mask}};
  std::string s(e);
  size_t a = 0;
  while (a < s.size()) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    const std::string kv = s.substr(a, b - a);
    a = b + 1;
    const size_t q = kv.find('=');
    if (q == std::string::npos) continue;
    const std::string k = kv.substr(0, q);
    const double v = std::atof(kv.c_str() + q + 1);
    bool ok = false;
    for (const K& x : keys)
      if (k == x.n) {
        if (x.d) *x.d = v;
        if (x.i) *x.i = int(v);
        if (x.b) *x.b = v != 0;
        ok = true;
      }
    std::fprintf(stderr, "[objmap] SM_OBJ_PARAMS %s = %g%s\n", k.c_str(), v, ok ? "" : " (unknown key)");
  }
}

double ObjectMap::nameShare(const MapObject& m, int cls) {
  double tot = 0, mine = 0;
  for (const auto& [c, w] : m.votes) {
    tot += w;
    if (c == cls) mine = w;
  }
  return tot > 0 ? mine / tot : (cls == m.cls ? 1.0 : 0.0);
}

void ObjectMap::vote(MapObject& m, int cls, float w) const {
  w = std::max(w, 1e-3f);
  bool found = false;
  for (auto& [c, v] : m.votes)
    if (c == cls) { v += w; found = true; break; }
  if (!found) m.votes.emplace_back(cls, w);
  if (!p_.name_vote) return;
  // 이름 = 표의 최댓값. 지금 이름보다 name_switch 배 넘게 많아야 바꿈(흔들림 막기)
  float cur = 0, best = 0;
  int bc = m.cls;
  for (const auto& [c, v] : m.votes) {
    if (c == m.cls) cur = v;
    if (v > best) { best = v; bc = c; }
  }
  if (bc != m.cls && best > p_.name_switch * cur) m.cls = bc;
}

bool ObjectMap::nameOk(const MapObject& m, int cls) const {
  if (m.cls == cls) return true;
  return p_.name_vote && nameShare(m, cls) >= p_.name_share;
}

bool ObjectMap::isBig(const MapObject& m) const {
  return kindOf(m.cls) == kKindStatic || std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]}) > p_.big;
}

namespace {
inline int64_t cellKey(double x, double y, double cell) {
  const int64_t i = int64_t(std::floor(x / cell)), j = int64_t(std::floor(y / cell));
  return (i << 32) ^ (j & 0xffffffffll);
}
}  // namespace

// 본 곳(2D 칸): 깊이를 성기게(가로 32 점) 훑어 카메라 → 점 수평 선분이 지나는 칸에, 거리 띠(1·2·3·4·5 m 이하)마다 처음 본 시각
void ObjectMap::markView(const ObjFrame& f) {
  if (!(f.depth_m || f.depth_mm) || f.w <= 0 || f.h <= 0) return;
  const double* T = f.T_mc;
  const double cell = std::max(0.05, p_.view_cell);
  const int stp = std::max(1, f.w / 32);
  for (int v = stp / 2; v < f.h; v += stp)
    for (int u = stp / 2; u < f.w; u += stp) {
      const size_t i = size_t(v) * f.w + u;
      const float z = f.depth_m ? f.depth_m[i] : f.depth_mm[i] * 1e-3f;
      if (!(z > p_.zmin && z < p_.zmax)) continue;
      const double xc = (u - f.cx) / f.fx * z, yc = (v - f.cy) / f.fy * z;
      const double px = T[0] * xc + T[1] * yc + T[2] * z + T[3];
      const double py = T[4] * xc + T[5] * yc + T[6] * z + T[7];
      const double dx = px - T[3], dy = py - T[7];
      const double L = std::sqrt(dx * dx + dy * dy), Lm = std::min(L, double(kViewBands));
      if (L < 1e-6) continue;
      const int ns = int(std::ceil(Lm / (0.5 * cell)));
      for (int s = 0; s <= ns; ++s) {
        const double r = Lm * s / std::max(1, ns);
        auto [it, fresh] = view_first_.try_emplace(cellKey(T[3] + r / L * dx, T[7] + r / L * dy, cell));
        auto& e = it->second;
        if (fresh) e.fill(-1e300);   // 새 칸: 아직 못 봄
        for (int b = std::max(0, int(std::ceil(r)) - 1); b < kViewBands; ++b)
          if (e[size_t(b)] < -1e299) e[size_t(b)] = f.stamp;
      }
    }
}

double ObjectMap::firstView(double x, double y, double range) const {
  auto it = view_first_.find(cellKey(x, y, std::max(0.05, p_.view_cell)));
  if (it == view_first_.end()) return 1e300;
  const int b = std::clamp(int(std::ceil(range)) - 1, 0, kViewBands - 1);
  return it->second[size_t(b)] < -1e299 ? 1e300 : it->second[size_t(b)];
}

int ObjectMap::absentEvidence(const MapObject& m, const ObjFrame& f) const {
  const double* T = f.T_mc;
  auto toCam = [&](const double p[3], double c[3]) {
    const double d[3] = {p[0] - T[3], p[1] - T[7], p[2] - T[11]};
    c[0] = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
    c[1] = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
    c[2] = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
  };
  // 검출할 수 있는 거리 안인가: 이 물체를 검출했던 가장 먼 깊이 기준
  double cc[3];
  toCam(m.pos, cc);
  if (cc[2] < 0.3 || cc[2] > p_.zmax) return 0;
  if (m.max_det_z > 0 && cc[2] > p_.absent_det_k * m.max_det_z + 0.2) return 0;
  auto depthAt = [&](int u, int v) -> float {
    const size_t i = size_t(v) * f.w + u;
    return f.depth_m ? f.depth_m[i] : f.depth_mm[i] * 1e-3f;
  };
  // 표본 점: 구름(물체 겉면)이 있으면 고른 간격으로, 없으면 상자 3×3×3 격자
  thread_local std::vector<double> S;
  S.clear();
  const size_t nc = m.cloud.size();
  const int K = std::max(8, p_.absent_samples);
  if (nc >= 8) {
    const auto& pts = m.cloud.data->pts;
    const size_t stride = std::max<size_t>(1, nc / size_t(K));
    for (size_t i = 0; i < nc && S.size() < size_t(3 * K); i += stride)
      S.insert(S.end(), {m.cloud.org[0] + pts[i].x, m.cloud.org[1] + pts[i].y, m.cloud.org[2] + pts[i].z});
  } else {
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          S.insert(S.end(), {m.lo[0] + 0.5 * a * (m.hi[0] - m.lo[0]), m.lo[1] + 0.5 * b * (m.hi[1] - m.lo[1]),
                             m.lo[2] + 0.5 * c * (m.hi[2] - m.lo[2])});
  }
  const int n = int(S.size() / 3);
  int vis = 0, thru = 0;
  double u0 = 1e9, v0 = 1e9, u1 = -1e9, v1 = -1e9;
  for (int i = 0; i < n; ++i) {
    double c[3];
    toCam(&S[3 * size_t(i)], c);
    if (c[2] < 0.2) continue;
    const double uf = f.fx * c[0] / c[2] + f.cx, vf = f.fy * c[1] / c[2] + f.cy;
    const int u = int(uf), v = int(vf);
    if (u < 2 || v < 2 || u >= f.w - 2 || v >= f.h - 2) continue;
    const float d = depthAt(u, v);
    if (!(d > 0)) continue;
    if (d < c[2] - p_.occl) continue;   // 앞에 다른 것이 있음(가림)
    ++vis;
    if (d > c[2] + std::max(p_.occl, 0.1)) ++thru;   // 물체 자리 너머가 보임
    u0 = std::min(u0, uf); u1 = std::max(u1, uf); v0 = std::min(v0, vf); v1 = std::max(v1, vf);
  }
  static const long dbg_id = std::getenv("SM_ABS_LOG") ? std::atol(std::getenv("SM_ABS_LOG")) : -1;
  if (long(m.id) == dbg_id)
    std::fprintf(stderr, "[abs] t=%.1f O%u z=%.2f maxz=%.2f n=%d vis=%d px=%.1f misses=%d\n", f.stamp, m.id, cc[2], m.max_det_z, n, vis,
                 std::sqrt(std::max(0.0, u1 - u0) * std::max(0.0, v1 - v0)), m.misses);
  if (vis < std::max(3, int(p_.absent_vis * n + 0.5))) return 0;
  if (std::sqrt(std::max(0.0, u1 - u0) * std::max(0.0, v1 - v0)) < p_.absent_min_px) return 0;
  return thru * 10 >= vis * 3 ? 2 : 1;   // 2: 보이는 점의 30 % 이상에서 그 너머가 보임(자리가 비었음)
}

void ObjectMap::remapId(uint32_t from, uint32_t to) {
  for (DetAssoc& as : assoc_) if (as.obj_id == from) as.obj_id = to;
  for (ObsPoints& op : points_) if (op.obj_id == from) op.obj_id = to;
  for (MapObject& o : objs_) if (o.parent == from) o.parent = to;
}

// 옮겨짐 잇기: 사라진 m ↔ 새로 나타난 n(같은 이름). 가까운 쌍부터 1:1. n 의 관측·모양을 m 의 id 로 옮기고 n 은 지운다
void ObjectMap::relink(double t) {
  std::vector<std::tuple<double, uint32_t, uint32_t>> pairs;
  for (const MapObject& n : objs_) {
    if (!n.appeared || !n.confirmed || n.held_by >= 0 || n.state == SM_GONE || int(n.n_obs) < p_.link_min_obs) continue;
    if (t - n.first_seen > p_.link_window_s) continue;   // 오래 자리 잡은 새 물체는 더 잇지 않음
    for (const MapObject& m : objs_) {
      if (&m == &n || m.state != SM_GONE || !m.confirmed || m.held_by >= 0 || (!m.moved && int(m.n_obs) < p_.spurious_obs)) continue;
      if (m.cls != n.cls && !(p_.name_vote && (nameShare(m, n.cls) >= p_.name_share || nameShare(n, m.cls) >= p_.name_share))) continue;
      if (!(n.first_seen > m.last_seen)) continue;   // 둘이 같이 있던 적이 있으면 다른 물체
      const double d = dist3(n.pos, m.pos);
      if (d > std::min(p_.link_max_d, p_.link_d0 + p_.link_v * (n.first_seen - m.last_seen))) continue;
      // 애매함: n 에 더 가까운 같은 이름 물체가 n 이 나타난 뒤로 아직 안 보였으면 그것이 사라졌는지 알 때까지 기다림
      bool wait = false;
      if (t - n.first_seen < p_.link_wait_s)
        for (const MapObject& q : objs_) {
          if (&q == &n || &q == &m || !q.confirmed || q.held_by >= 0 || q.state == SM_GONE || q.cls != n.cls) continue;
          if (q.last_seen < n.first_seen && dist3(q.pos, n.pos) < d) { wait = true; break; }
        }
      static const long dbg_n = std::getenv("SM_LINK_LOG") ? std::atol(std::getenv("SM_LINK_LOG")) : -1;
      if (long(n.id) == dbg_n) std::fprintf(stderr, "[relink] t=%.1f n=O%u m=O%u d=%.2f wait=%d\n", t, n.id, m.id, d, int(wait));
      if (!wait) pairs.emplace_back(d, m.id, n.id);
    }
  }
  if (pairs.empty()) return;
  std::sort(pairs.begin(), pairs.end());
  std::vector<uint32_t> used;
  for (auto& [d, mid, nid] : pairs) {
    if (std::find(used.begin(), used.end(), mid) != used.end() || std::find(used.begin(), used.end(), nid) != used.end()) continue;
    used.push_back(mid);
    used.push_back(nid);
    auto mi = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& o) { return o.id == mid; });
    auto ni = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& o) { return o.id == nid; });
    MapObject& m = *mi;
    MapObject& n = *ni;
    static const bool log_link = std::getenv("SM_OBJ_LOG") != nullptr;
    if (log_link)
      std::fprintf(stderr, "[link] t=%.1f O%u <- O%u d=%.2f cls=%d from %.2f %.2f last %.1f to %.2f %.2f first %.1f\n", t, m.id, n.id, d,
                   m.cls, m.pos[0], m.pos[1], m.last_seen, n.pos[0], n.pos[1], n.first_seen);
    for (int k = 0; k < 3; ++k) { m.pos[k] = n.pos[k]; m.ext[k] = n.ext[k]; m.lo[k] = n.lo[k]; m.hi[k] = n.hi[k]; }
    m.cloud = n.cloud;   // 새 자리의 모양
    m.cloud.version = std::max(m.cloud.version, n.cloud.version) + 1;
    m.n_obs += n.n_obs;
    m.last_seen = n.last_seen;
    m.last_kf = n.last_kf;
    m.score = std::max(m.score, n.score);
    for (const auto& [c, w] : n.votes) vote(m, c, w);
    m.max_det_z = std::max(m.max_det_z, n.max_det_z);
    for (int k = 0; k < 3; ++k) m.trk_pos[k] = n.trk_pos[k];
    m.trk_t = n.trk_t;
    m.mv_cnt = 0;
    m.appeared = false;
    m.moved = true;
    m.state = SM_MOVED;
    m.misses = 0;
    m.parent = 0;

    remapId(nid, mid);
    event(t, m, 2);
    objs_.erase(ni);
  }
}

double ObjectMap::gripGap(double g) const {
  const auto& T = p_.grip_gap;
  if (T.empty()) return 0;
  if (g <= T.front().first) return T.front().second;
  for (size_t i = 1; i < T.size(); ++i)
    if (g <= T[i].first) {
      const double a = (g - T[i - 1].first) / std::max(1e-12, T[i].first - T[i - 1].first);
      return T[i - 1].second + a * (T[i].second - T[i - 1].second);
    }
  return T.back().second;
}

// grasp_check: 틈 gap 으로 쥔 그리퍼 사이에 o 가 있을 수 있나(크기·종류·틈과 폭)
bool ObjectMap::holdable(const MapObject& o, double gap) const {
  if (isBig(o)) return false;   // 고정 종류(가구·가전) 또는 한 변 > big
  double e[3];
  for (int k = 0; k < 3; ++k) e[k] = std::max(0.0, o.hi[k] - o.lo[k]);
  std::sort(e, e + 3);
  // 폭 = 가운데 변: 한 축만 긴 것(펜·병)은 들 수 있고, 한쪽 면만 본 물체는 깊이 쪽 변이 거의 0 이라 가장 좁은 변은 폭이 못 됨
  const double wmin = e[0], wmid = e[1], wmax = e[2];
  if (wmid > p_.grasp_max_w) return false;
  if (gap < p_.grasp_min_gap) return false;   // 끝까지 닫힘: 손가락 사이에 아무것도 없음
  return gap >= wmin - p_.grasp_w_tol && gap <= wmax + p_.grasp_w_tol;
}

void ObjectMap::release(double t, MapObject& o) {
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

void ObjectMap::updateHands(double t, const double eef[2][3], const float grip[2], double yaw) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  auto grab = [&](MapObject* best, int h) {
    best->held_by = h;
    best->parent = 0;
    const double d[3] = {best->pos[0] - eef[h][0], best->pos[1] - eef[h][1], best->pos[2] - eef[h][2]};
    best->held_rel[0] = cy * d[0] + sy * d[1];
    best->held_rel[1] = -sy * d[0] + cy * d[1];
    best->held_rel[2] = d[2];
    for (int k = 0; k < 3; ++k) best->grasp_pos[k] = best->pos[k];
    best->state = SM_HELD;
    event(t, *best, 4);
  };
  for (int h = 0; h < p_.n_hands && h < 2; ++h) {
    const bool closed = grip[h] < p_.grip_closed;
    if (p_.grasp_check) {
      // 쥠이 끝났나: 그리퍼 값이 grip_settle_s 동안 grip_settle_eps 안
      if (gref_t_[h] < -1e299 || std::fabs(grip[h] - gref_[h]) > p_.grip_settle_eps) { gref_[h] = grip[h]; gref_t_[h] = t; }
      const bool settled = t - gref_t_[h] >= p_.grip_settle_s - 1e-9;
      if (closed && !closed_[h]) tried_[h] = false;   // 새로 닫힘
      const double gap = gripGap(grip[h]);
      if (closed && settled && !tried_[h]) {          // 잡기: 닫힌 채 멈춘 뒤 한 번
        tried_[h] = true;
        MapObject* best = nullptr;
        double bd = p_.grasp_r;
        for (auto& o : objs_) {
          if (!o.confirmed || o.held_by >= 0 || o.state == SM_GONE) continue;
          const double d = dist3(o.pos, eef[h]);
          if (d < bd && holdable(o, gap)) { bd = d; best = &o; }
        }
        if (best) grab(best, h);
      } else if (closed && settled && gap < p_.grasp_min_gap) {   // 든 뒤 끝까지 닫힘: 놓침
        for (auto& o : objs_)
          if (o.held_by == h) release(t, o);
      }
      if (!closed && closed_[h])
        for (auto& o : objs_)
          if (o.held_by == h) release(t, o);
      closed_[h] = closed;
      continue;
    }
    if (closed && !closed_[h]) {   // 잡기
      MapObject* best = nullptr;
      double bd = p_.grasp_r;
      for (auto& o : objs_) {
        if (!o.confirmed || o.held_by >= 0 || o.state == SM_GONE) continue;
        const double d = dist3(o.pos, eef[h]);
        if (d < bd) { bd = d; best = &o; }
      }
      if (best) grab(best, h);
    } else if (!closed && closed_[h]) {   // 놓기
      for (auto& o : objs_)
        if (o.held_by == h) release(t, o);
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
  // 카메라가 빨리 돌 때는 움직임 근거를 안 씀(자세 오차가 물체를 쓸어 가는 것처럼 보이게 함)
  double cam_w = 0;
  {
    const double fwd[3] = {T[2], T[6], T[10]};
    const double dt = f.stamp - last_cam_t_;
    if (last_cam_t_ > -1e299 && dt > 1e-6) {
      const double c = std::clamp(fwd[0] * last_cam_fwd_[0] + fwd[1] * last_cam_fwd_[1] + fwd[2] * last_cam_fwd_[2], -1.0, 1.0);
      cam_w = std::acos(c) / dt;
    }
    for (int k = 0; k < 3; ++k) last_cam_fwd_[k] = fwd[k];
    last_cam_t_ = f.stamp;
  }
  const bool cam_steady = cam_w <= p_.move_max_cam_w;
  auto inSelf = [&](const ObjFrame& fr, double px, double py, double pz) {
    return inSelfCaps(fr.self_caps, fr.n_self_caps, p_.self_pad, px, py, pz);
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
          if (f.n_self_caps > 0 && p_.self_mask && inSelf(f, px, py, pz)) continue;   // 로봇 팔 화소(카메라 앞을 가림)
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
      o.trunc = D->img_w > 0 && D->img_h > 0 && (b[0] <= 2 || b[1] <= 2 || b[2] >= D->img_w - 3 || b[3] >= D->img_h - 3);
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
      // 바닥 조각(점이 거의 다 바닥 높이): 물체 아님. 바닥에 깔리는 이름(러그·카펫·매트 — setFloorClasses)은 둠
      if (o.hi[2] < p_.floor_h && !floorLevel(o.cls)) continue;
      obs.push_back(std::move(o));
    }
  }
  // 2. 같은 물체: 같은 이름(이름 표에서 name_share 이상인 이름 포함)끼리 가까운 쌍부터 1:1. 이름이 표 최댓값과 다르면 조금 뒤로
  std::vector<std::tuple<double, int, int>> pairs;
  for (int a = 0; a < int(obs.size()); ++a)
    for (int b = 0; b < int(objs_.size()); ++b) {
      const MapObject& m = objs_[b];
      if (m.held_by >= 0 || !nameOk(m, obs[a].cls)) continue;
      const double e = std::max({obs[a].ext[0], obs[a].ext[1], obs[a].ext[2], m.ext[0], m.ext[1], m.ext[2]});
      const double thr = std::max(p_.da_min, p_.da_k * e);
      const double d = dist3(obs[a].pos, m.pos);
      double gap2 = 0;
      for (int k = 0; k < 3; ++k) {
        const double gk = std::max({0.0, obs[a].lo[k] - m.hi[k], m.lo[k] - obs[a].hi[k]});
        gap2 += gk * gk;
      }
      const double gap = std::sqrt(gap2);
      if (d < thr || gap < p_.da_gap) pairs.emplace_back(gap + 1e-3 * d + (m.cls != obs[a].cls ? 0.02 : 0.0), a, b);
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
      // 움직이는 중(사람이 옮김): 관측 중심(날 것)이 잇달아 move_v 넘는 빠르기로 같은 쪽으로 가고 쉬던 자리에서 move_min_d(또는 상자
      // 반 폭) 넘게 벗어나면 그동안 평균·합집합 대신 관측 자리로 바로 옮긴다(평균이 뒤처져 놓치지 않게)
      bool snap = false;
      if (p_.move_n > 0 && m.trk_t > 0 && m.held_by < 0 && !o.trunc && cam_steady) {
        const double dt = f.stamp - m.trk_t;
        const double sx = o.pos[0] - m.trk_pos[0], sy = o.pos[1] - m.trk_pos[1];
        const double sp = dt > 1e-6 && dt <= 0.6 ? std::sqrt(sx * sx + sy * sy) / dt : 0.0;
        const bool same_dir = sx * m.trk_step[0] + sy * m.trk_step[1] > 0;
        m.mv_cnt = sp > p_.move_v && (same_dir || m.mv_cnt == 0) ? m.mv_cnt + 1 : 0;
        m.trk_step[0] = sx;
        m.trk_step[1] = sy;
        const double rx = o.pos[0] - m.pos[0], ry = o.pos[1] - m.pos[1];
        const double away = std::sqrt(rx * rx + ry * ry);
        const bool moving = f.stamp - m.moving_t < 1.0;
        // 시작: 관측 상자가 쉬던 상자와 거의 안 겹쳐야(부분만 보여 중심이 흔들리는 것은 같은 상자 안)
        const bool left = away > std::max(p_.move_min_d, 0.5 * std::max(o.ext[0], o.ext[1])) &&
                          da::boxOverlap(o.lo, o.hi, m.lo, m.hi, p_.merge_min_ext) < 0.1;
        if (m.mv_cnt >= p_.move_n && (moving || left)) {
          if (!moving) m.move_from_t = f.stamp;
          m.moving_t = f.stamp;
          snap = true;
        } else if (moving && sp > 0.5 * p_.move_v) {
          snap = true;   // 움직이는 중에는 조금 느려져도 따라감
          m.moving_t = f.stamp;
        }
      }
      if (!o.trunc) {
        for (int k = 0; k < 3; ++k) m.trk_pos[k] = o.pos[k];
        m.trk_t = f.stamp;
      }
      static const long dbg_mv = std::getenv("SM_MOVE_LOG") ? std::atol(std::getenv("SM_MOVE_LOG")) : -1;
      if (snap && (dbg_mv == 0 || long(m.id) == dbg_mv))
        std::fprintf(stderr, "[move] t=%.1f O%u cls=%d from %.2f %.2f %.2f to %.2f %.2f %.2f ext %.2f %.2f %.2f cnt %d\n", f.stamp, m.id, m.cls,
                     m.pos[0], m.pos[1], m.pos[2], o.pos[0], o.pos[1], o.pos[2], o.ext[0], o.ext[1], o.ext[2], m.mv_cnt);
      if (snap) {
        double d3[3];
        for (int k = 0; k < 3; ++k) {
          d3[k] = o.pos[k] - m.pos[k];
          m.pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k];
        }
        m.cloud.translate(d3, f.stamp);
        if (dist3(m.pos, m.first_pos) > p_.moved_d) m.moved = true;
        if (m.moved && m.state == SM_SEEN) m.state = SM_MOVED;
        m.parent = 0;
      }
      for (int k = 0; k < 3 && !snap; ++k) {
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
      vote(m, o.cls, o.score);
      m.max_det_z = std::max(m.max_det_z, o.zmed);
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
    // 안 맞은 관측: 새 후보. 전에 (view_r 안에서) 본 자리에 새로 나타났으면 옮겨짐 잇기 후보(relink)
    MapObject m;
    m.id = next_id_++;
    m.cls = o.cls;
    for (int k = 0; k < 3; ++k) { m.pos[k] = m.first_pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k]; }
    m.n_obs = 1;
    m.first_seen = m.last_seen = m.last_kf = f.stamp;
    for (int k = 0; k < 3; ++k) m.trk_pos[k] = o.pos[k];
    m.trk_t = f.stamp;
    m.score = o.score;
    vote(m, o.cls, o.score);
    m.max_det_z = o.zmed;
    {   // 처음 검출한 거리(수평) 이하에서 그 자리를 link_view_gap_s 넘게 전에 본 적 있으면 '나타남'
      const double hx = o.pos[0] - T[3], hy = o.pos[1] - T[7];
      m.appeared = firstView(o.pos[0], o.pos[1], std::sqrt(hx * hx + hy * hy) + p_.view_cell) < f.stamp - p_.link_view_gap_s;
    }
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
  // 3b. 조각 합치기: 이번 관측 하나(검출 하나 = 물체 하나)의 상자 안에 다른 같은 이름 물체가 거의 다 들어 있으면 같은 물체의 조각
  if (p_.frag_overlap > 0) {
    std::vector<std::pair<uint32_t, uint32_t>> fr;   // (남김, 지움)
    for (int a = 0; a < int(obs.size()); ++a) {
      const uint32_t keep = assoc_[obs[a].det].obj_id;
      if (!keep) continue;
      const Obs& o = obs[a];
      const double oext = std::max({o.ext[0], o.ext[1], o.ext[2]});
      for (size_t c = 0; c < objs_.size(); ++c) {
        const MapObject& q = objs_[c];
        if (q.id == keep || obj_hit[c] || q.held_by >= 0 || !nameOk(q, o.cls)) continue;
        bool inside = true;
        for (int k = 0; k < 3; ++k) inside = inside && q.pos[k] >= o.lo[k] - 0.05 && q.pos[k] <= o.hi[k] + 0.05;
        if (!inside || std::max({q.ext[0], q.ext[1], q.ext[2]}) > oext) continue;
        if (da::boxOverlap(q.lo, q.hi, o.lo, o.hi, p_.merge_min_ext) < p_.frag_overlap) continue;
        fr.emplace_back(keep, q.id);
      }
    }
    std::vector<std::pair<uint32_t, uint32_t>> gone_to;   // 지운 id → 남은 id
    auto cur = [&](uint32_t id) {
      for (bool ch = true; ch;) {
        ch = false;
        for (auto& [g, t] : gone_to) if (g == id) { id = t; ch = true; break; }
      }
      return id;
    };
    for (auto [kid, did] : fr) {
      kid = cur(kid);
      did = cur(did);
      if (kid == did) continue;
      auto ki = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& x) { return x.id == kid; });
      auto di = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& x) { return x.id == did; });
      if (ki == objs_.end() || di == objs_.end()) continue;
      if (di->n_obs > ki->n_obs || (di->n_obs == ki->n_obs && did < kid)) {   // 관측이 많은(같으면 먼저 본) 쪽 id 를 남김
        std::swap(ki, di);
        std::swap(kid, did);
      }
      gone_to.emplace_back(did, kid);
      da::absorbObject(*ki, *di, p_, f.stamp, &kinds_);
      ki->confirmed = ki->confirmed || di->confirmed;
      const size_t dpos = size_t(di - objs_.begin());
      obj_hit.erase(obj_hit.begin() + long(dpos));
      objs_.erase(di);
      remapId(did, kid);
      for (const MapObject& x : objs_) if (x.id == kid) { event(f.stamp, x, 7); break; }
    }
  }
  // 4. 부재 확인(확정·안 든 것·이번에 안 맞은 것): 물체 점을 투영해 보일 만큼 보이는데 검출이 없으면 놓침
  if (f.depth_m || f.depth_mm) {
    for (size_t b = 0; b < objs_.size(); ++b) {
      MapObject& m = objs_[b];
      if (!m.confirmed || m.held_by >= 0 || obj_hit[b] || m.state == SM_GONE) continue;
      if (m.parent) continue;   // 통 안에 넣은 것은 안 보여도 그대로 있다고 본다
      // 손 가까이는 판단하지 않는다: 손에 든 관측은 거르므로(위) 잡으러 다가가는 동안 '안 보임'으로 셈하면 안 됨
      if (dist3(m.pos, f.eef[0]) < p_.hand_r + 0.1 || dist3(m.pos, f.eef[1]) < p_.hand_r + 0.1) continue;
      const int ev = absentEvidence(m, f);
      if (!ev) continue;
      // 다른 이름으로 검출됨: 이 물체 상자 안에 중심이 있는 관측의 이름이 이 물체 이름 표에 있으면(전에 그 이름으로도 불림)
      // 이름 흔들림이지 없어진 것이 아니다. 표에 없던 이름이면(다른 물체가 그 자리에 놓임) 놓침으로 센다
      bool renamed = false;
      for (const Obs& o : obs) {
        if (o.cls == m.cls || nameShare(m, o.cls) <= 0) continue;
        bool in = true;
        for (int k = 0; k < 3; ++k) in = in && o.pos[k] >= m.lo[k] - 0.1 && o.pos[k] <= m.hi[k] + 0.1;
        if (in) { renamed = true; break; }
      }
      if (renamed) continue;
      // 새 근거만 센다: 지난 놓침 뒤 카메라가 gone_step_d 넘게 옮기거나 gone_step_deg 넘게 돌았거나, 자리 너머가 보일 때.
      // 서 있는 카메라의 같은 영상을 되풀이해 세지 않는다(검출기는 같은 영상에서 같은 것을 놓친다)
      const double cam_now[3] = {T[3], T[7], T[11]}, fwd[3] = {T[2], T[6], T[10]};
      if (m.misses > 0 && ev != 2) {
        const double cosang = fwd[0] * m.last_miss_fwd[0] + fwd[1] * m.last_miss_fwd[1] + fwd[2] * m.last_miss_fwd[2];
        if (dist3(cam_now, m.last_miss_cam) < p_.gone_step_d && cosang > std::cos(p_.gone_step_deg * M_PI / 180.0)) continue;
      }
      for (int k = 0; k < 3; ++k) { m.last_miss_cam[k] = cam_now[k]; m.last_miss_fwd[k] = fwd[k]; }
      const bool big = isBig(m);
      ++m.n_vis_miss;
      if (m.misses++ == 0) {
        m.first_miss = f.stamp;
        for (int k = 0; k < 3; ++k) m.miss_cam[k] = T[4 * k + 3];
      }
      // 검출률을 아는 만큼: 보일 때 검출된 비율 p(관측 수 / (관측 수 + 보이는데 놓친 수), 라플라스)로 '있는데 k 번 연속 놓칠'
      // 확률 (1 − p)^k 가 gone_eps 아래가 되는 k 이상 놓쳐야(검출이 드문 물체일수록 더 기다림)
      const double pr = (double(m.n_obs) + 1.0) / (double(m.n_obs) + double(m.n_vis_miss) + 2.0);
      const int k_rate = pr >= 1.0 ? 0 : int(std::ceil(std::log(p_.gone_eps) / std::log(1.0 - pr)));
      const int need = std::min(std::max(big ? p_.gone_misses_big : p_.gone_misses, k_rate), 60);
      const double need_s = big ? p_.gone_min_s_big : p_.gone_min_s;
      // 놓침이 서로 다른 근거여야: 첫 놓침부터 need_s 넘게 지났거나 카메라가 gone_view_d 넘게 옮겨 다른 자리에서도 안 보임
      const double cam[3] = {T[3], T[7], T[11]};
      const bool indep = f.stamp - m.first_miss >= need_s - 1e-9 || dist3(cam, m.miss_cam) >= p_.gone_view_d;
      if (m.misses >= need && indep) {
        m.state = SM_GONE;
        m.gone_t = f.stamp;
        m.n_vis_miss -= uint32_t(std::min<int>(m.misses, int(m.n_vis_miss)));   // 이 연속 놓침은 진짜 없음이었다(검출률에서 뺌)
        event(f.stamp, m, 3);
      }
    }
  }
  relink(f.stamp);
  // 5. 오래된 후보 버리기
  objs_.erase(std::remove_if(objs_.begin(), objs_.end(),
                             [&](const MapObject& m) {
                               if (!m.confirmed) return f.stamp - m.last_seen > p_.prune_s;
                               // 몇 번 안 보이고 사라진 것은 헛검출(조각)로 보고 지움 — 옮겨짐 잇기 후보가 되지 않게
                               return m.state == SM_GONE && !m.moved && int(m.n_obs) < p_.spurious_obs;
                             }),
              objs_.end());
  markView(f);
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
