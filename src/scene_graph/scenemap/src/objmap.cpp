#include "scenemap/objmap.hpp"

#include "da/merge.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  // objprob
  const float* z = nullptr;       // SigLIP 임베딩(검출 k)
  double kappa = 0;               // vMF 집중도(viewKappa)
  bool wall_like = false;         // 벽 선 위 세운 얇은 평면(작아서 구조물로 안 버린 것)
  bool struct_look = false;       // 구조물 같은 조각(구조물 확률 ≥ 0.5 인 얇은 세운 평면) — 구조물 아닌 물체에 안 붙음
  bool sec = false;               // 같은 물체에 붙은 이번 영상의 다른 조각(첫 조각에 합쳐 갱신 — 짝·모습만)
  double ps = -1;                 // 이 조각의 (지울) 구조물 확률(라벨 위 p(c|z) 의 벽·바닥·천장 … 합, 임베딩 없으면 −1)
  double pso = 0;                 // 이 조각의 구조 물체(문·창·계단) 확률
  int top_lab = -1;               // 이 조각 하나의 가장 그럴듯한 라벨·확률(이름 충돌 막기)
  double top_p = 0;
  double camd = -1;               // 카메라 광학 중심 ↔ 관측 중심 거리(살펴본 정도)
};

// 점 p(xy)와 선분 사이 수평 거리
double segDist(double px, double py, const double* s) {
  const double ax = s[0], ay = s[1], bx = s[2], by = s[3];
  const double dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  double u = l2 > 1e-12 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0;
  u = std::clamp(u, 0.0, 1.0);
  return std::hypot(px - ax - u * dx, py - ay - u * dy);
}

// objprob 같은 것 특징(a = 관측 또는 작은 쪽 물체, b = 물체): a 의 점 표본 중 b 접촉 칸에 닿는 비율, 상자 틈, 중심 거리/크기, cos, 겹침, 받침
ApPair pairFeatures(const double alo[3], const double ahi[3], const double apos[3], const float* apts, int na, const double blo[3],
                    const double bhi[3], const double bpos[3], const std::vector<uint64_t>& bidx, double cos, const ApParams& P,
                    bool merge = false) {
  ApPair q;
  // 틈·중심 거리·cos·겹침·받침은 objprob_math.h(GPU 학습 지도와 같은 식), 접촉은 점 구름 색인으로 여기서
  q.f[0] = apContact(apts, na, bidx, P.contact_cell);
  opm::pair_geo<OpmStd>(alo, ahi, apos, blo, bhi, bpos, cos, P.cos0, q.f);
  q.logit = apLogit(q, P, merge);
  q.p = opm::sigmoid<OpmStd>(q.logit);
  return q;
}

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
  applyParams(p, e, true);
}

int ObjectMap::applyParams(ObjParams* p, const char* e, bool log) {
  if (!e || !*e) return 0;
  struct K { const char* n; double* d; int* i; bool* b; };
  const K keys[] = {{"floor_h", &p->floor_h, nullptr, nullptr},          {"name_vote", nullptr, nullptr, &p->name_vote},
                    {"name_switch", &p->name_switch, nullptr, nullptr},
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
                    {"grasp_check", nullptr, nullptr, &p->grasp_check},  {"self_pad", &p->self_pad, nullptr, nullptr},
                    {"self_mask", nullptr, nullptr, &p->self_mask},  {"inspect", nullptr, nullptr, &p->insp.on},
                    // objprob(objprob.hpp ApParams — 앞에 ap_)
                    {"ap_same_p", &p->ap.same_p, nullptr, nullptr},  {"ap_merge_p", &p->ap.merge_p, nullptr, nullptr},
                    {"ap_gate", &p->ap.gate, nullptr, nullptr},  {"ap_name_tau", &p->ap.name_tau, nullptr, nullptr},
                    {"ap_name_wmax", &p->ap.name_wmax, nullptr, nullptr},  {"ap_size_w", &p->ap.size_w, nullptr, nullptr},
                    {"ap_kappa_ref", &p->ap.kappa_ref, nullptr, nullptr},  {"ap_temper", &p->ap.temper, nullptr, nullptr},
                    {"ap_wall_big", &p->ap.wall_big, nullptr, nullptr},  {"ap_struct_p", &p->ap.struct_p, nullptr, nullptr},
                    {"ap_ceil_z", &p->ap.ceil_z, nullptr, nullptr},  {"ap_struct_obj_p", &p->ap.struct_obj_p, nullptr, nullptr},
                    {"ap_obj_wall_span", &p->ap.obj_wall_span, nullptr, nullptr},  {"ap_obj_wall_top", &p->ap.obj_wall_top, nullptr, nullptr},
                    {"ap_reenc_max", nullptr, &p->ap.reenc_max, nullptr},  {"ap_reenc_gain", &p->ap.reenc_gain, nullptr, nullptr},
                    {"ap_link_cos", &p->ap.link_cos, nullptr, nullptr},  {"ap_whole_w", &p->ap.whole_w, nullptr, nullptr},
                    {"ap_export_named", nullptr, nullptr, &p->ap.export_named},  {"ap_guard_obj", &p->ap.guard_obj, nullptr, nullptr},
                    {"ap_guard_obs", &p->ap.guard_obs, nullptr, nullptr},  {"ap_geo_w", &p->ap.geo_w, nullptr, nullptr},
                    {"ap_name_veto", &p->ap.name_veto, nullptr, nullptr},  {"ap_through_d", &p->ap.through_d, nullptr, nullptr},  {"ap_bridge_max", &p->ap.bridge_max, nullptr, nullptr},  {"ap_frame_group", nullptr, nullptr, &p->ap.frame_group},  {"ap_struct_obj_min_views", nullptr, &p->ap.struct_obj_min_views, nullptr},
                    {"ap_plane_inl", &p->ap.plane_inl, nullptr, nullptr},  {"ap_obj_plane_inl", &p->ap.obj_plane_inl, nullptr, nullptr},  {"ap_ransac_tau0", &p->ap.ransac_tau0, nullptr, nullptr},
                    {"ap_ransac_tau_k", &p->ap.ransac_tau_k, nullptr, nullptr},  {"ap_obj_tau", &p->ap.obj_tau, nullptr, nullptr},
                    {"ap_ransac_iters", nullptr, &p->ap.ransac_iters, nullptr},
                    {"ap_so_dw_w", &p->ap.so_dw_w, nullptr, nullptr},  {"ap_so_dw_thick", &p->ap.so_dw_thick, nullptr, nullptr},
                    {"ap_so_max_h", &p->ap.so_max_h, nullptr, nullptr},  {"ap_so_pillar", &p->ap.so_pillar, nullptr, nullptr},
                    {"ap_so_stairs", &p->ap.so_stairs, nullptr, nullptr},  {"ap_wallhug_frac", &p->ap.wallhug_frac, nullptr, nullptr},
                    {"ap_wallhug_w", &p->ap.wallhug_w, nullptr, nullptr},
                    {"ap_struct_look_block", nullptr, nullptr, &p->ap.struct_look_block},  {"ap_flat_max_w", &p->ap.flat_max_w, nullptr, nullptr},
                    {"ap_tall_h", &p->ap.tall_h, nullptr, nullptr},  {"ap_tall_w", &p->ap.tall_w, nullptr, nullptr},  {"ap_tall_ps", &p->ap.tall_ps, nullptr, nullptr},
                    {"ap_tall_w_any", &p->ap.tall_w_any, nullptr, nullptr},  {"ap_big_vinl", &p->ap.big_vinl, nullptr, nullptr},  {"ap_so_dw_min", &p->ap.so_dw_min, nullptr, nullptr},  {"ap_so_hide_k", &p->ap.so_hide_k, nullptr, nullptr},
                    // 같은 것 로지스틱 가중치(관측 ↔ 물체 ap_w0..7, 물체 ↔ 물체 ap_wm0..7 — ApParams::w·wm 순서)·κ(KappaParams) — 엔진별 매개변수 파일
                    {"ap_w0", &p->ap.w[0], nullptr, nullptr}, {"ap_w1", &p->ap.w[1], nullptr, nullptr}, {"ap_w2", &p->ap.w[2], nullptr, nullptr},
                    {"ap_w3", &p->ap.w[3], nullptr, nullptr}, {"ap_w4", &p->ap.w[4], nullptr, nullptr}, {"ap_w5", &p->ap.w[5], nullptr, nullptr},
                    {"ap_w6", &p->ap.w[6], nullptr, nullptr}, {"ap_w7", &p->ap.w[7], nullptr, nullptr},
                    {"ap_wm0", &p->ap.wm[0], nullptr, nullptr}, {"ap_wm1", &p->ap.wm[1], nullptr, nullptr}, {"ap_wm2", &p->ap.wm[2], nullptr, nullptr},
                    {"ap_wm3", &p->ap.wm[3], nullptr, nullptr}, {"ap_wm4", &p->ap.wm[4], nullptr, nullptr}, {"ap_wm5", &p->ap.wm[5], nullptr, nullptr},
                    {"ap_wm6", &p->ap.wm[6], nullptr, nullptr}, {"ap_wm7", &p->ap.wm[7], nullptr, nullptr},
                    {"ap_cos0", &p->ap.cos0, nullptr, nullptr},  {"ap_bridge_drop", nullptr, nullptr, &p->ap.bridge_drop},
                    {"kap_k0", &p->kap.k0, nullptr, nullptr}, {"kap_s0", &p->kap.s0, nullptr, nullptr}, {"kap_trunc", &p->kap.trunc, nullptr, nullptr},
                    {"kap_d0", &p->kap.d0, nullptr, nullptr}};
  int n_bad = 0;
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
    if (!ok) ++n_bad;
    if (log || !ok) std::fprintf(stderr, "[objmap] %s %s = %g%s\n", log ? "SM_OBJ_PARAMS" : "params", k.c_str(), v, ok ? "" : " (unknown key)");
  }
  return n_bad;
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

bool ObjectMap::isBig(const MapObject& m) const {
  return kindOf(m.cls) == kKindStatic || kindOf(m.cls) == kKindStructObj || std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]}) > p_.big;
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
      if (!m.ap || !n.ap) continue;
      {   // objprob: 임베딩(μ 끼리 가장 잘 맞는 cos)
        std::vector<float> mu;
        if (!apMu(*n.ap, &mu) || apCosMax(*m.ap, mu.data(), int(mu.size())) < p_.ap.link_cos) continue;
      }
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
    m.insp = n.insp;   // 살펴본 정도는 새 자리 것
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
  cam_w_ = cam_w;
  // objprob: 지난 keyframe 들의 구름이 쌓인 뒤 물체끼리 같은 것 판정(이름 없이) — 합친 물체를 이번 관측이 바로 받게 먼저
  if (f.wall_segs) wsegs_.assign(f.wall_segs, f.wall_segs + 4 * f.n_wall_segs);
  apMergePass(f.stamp);
  const double cam6[6] = {T[3], T[7], T[11], T[2], T[6], T[10]};
  // objprob 기하 구조물: 관측 점(복셀 후보)의 평면 맞춤 — 얇은 수평면이 천장 높이(ceil_z) 위면 천장, 바닥 높이면 바닥,
  // 얇은 세운 평면의 점 wall_frac 이상이 벽 선분 wall_d 안이면 벽 선 위 평면: 크면(wall_big) 벽, 작으면 이 조각 하나의
  // 구조물(벽·문·창 …) 확률이 struct_p 이상일 때만 벽(벽에 건 액자·간판은 남김) — 남긴 것은 wall_like(벽 추출에서 자리를 안 지움)
  std::vector<uint8_t> smask, omask;   // 라벨이 지울 구조물(벽·바닥·천장 …) / 구조 물체(문·창·계단)
  smask.assign(size_t(std::max(0, text_.n_labels)), 0);
  omask.assign(size_t(std::max(0, text_.n_labels)), 0);
  for (int l = 0; l < text_.n_labels; ++l) { smask[size_t(l)] = kindOf(l) == kKindStructure; omask[size_t(l)] = kindOf(l) == kKindStructObj; }
  std::vector<uint8_t> gmask(smask.size());   // 구조물 막기용: 지울 구조물 + 구조 물체
  for (size_t l = 0; l < gmask.size(); ++l) gmask[l] = smask[l] || omask[l];
  std::vector<float> llz;
  auto apObsStructural = [&](Obs& o) -> bool {
    double qsh[4] = {0, 0, 0, 0};   // 구조 물체 모양 묶음별 확률(문·창 / 계단 / 기둥)
    if (o.z && text_.ready() && text_.dim == f.emb_dim) {
      apLabelLogLik(text_, o.z, &llz);
      double q = 0, qo = 0;
      for (size_t l = 0; l < llz.size() && l < smask.size(); ++l) {
        if (llz[l] <= -1e29f) continue;
        if (smask[l]) q += std::exp(llz[l]);
        if (omask[l]) {
          qo += std::exp(llz[l]);
          qsh[l < text_.so_shape.size() ? text_.so_shape[l] & 3 : 0] += std::exp(llz[l]);
        }
      }
      o.ps = q;
      o.pso = qo;
      for (size_t l = 0; l < llz.size(); ++l)   // 이 조각 하나의 이름(사전 없이 p(c|z) 최대)
        if (llz[l] > -1e29f && std::exp(llz[l]) > o.top_p) { o.top_p = std::exp(llz[l]); o.top_lab = int(l); }
    }
    const int nc = int(o.cxyz.size() / 3);
    const ApParams& A = p_.ap;
    if (nc < 8) return false;
    double cen[3] = {0, 0, 0};
    for (int i = 0; i < nc; ++i) for (int k = 0; k < 3; ++k) cen[k] += o.cxyz[size_t(3 * i + k)];
    const double dc = std::hypot(cen[0] / nc - f.T_mc[3], cen[1] / nc - f.T_mc[7], cen[2] / nc - f.T_mc[11]);
    const double tau = std::min(A.ransac_tau_max, A.ransac_tau0 + A.ransac_tau_k * dc * dc);
    uint64_t sb;
    std::memcpy(&sb, &f.stamp, 8);
    const ApPlane pl = apPlaneFit(o.cxyz.data(), nc, tau, apSeed(sb, uint64_t(o.det)), A.ransac_iters);
    if (!pl.ok || pl.inl < A.plane_inl || pl.thick > A.plane_thick) return false;
    const double nz = std::fabs(pl.n[2]);
    if (nz > A.horiz) {
      if (pl.zmed > 1.8) {   // 천장 높이 추정에 씀(정답 없이)
        ceil_obs_.push_back(pl.zmed);
        if (ceil_obs_.size() > 400) ceil_obs_.erase(ceil_obs_.begin());
        if (ceil_obs_.size() >= 20) {
          std::vector<double> v = ceil_obs_;
          std::nth_element(v.begin(), v.begin() + long(v.size() / 2), v.end());
          ceil_est_ = v[v.size() / 2];
        }
      }
      const double cz = ceil_est_ > 0 ? std::min(A.ceil_z, ceil_est_ - A.ceil_band) : A.ceil_z;
      if (pl.zmed > cz) { ++aps_.n_ceil; return true; }
      if (pl.zhi < A.floor_z) { ++aps_.n_floor; return true; }
      return false;
    }
    if (nz > A.wall_vert) return false;
    // 얇은 세운 평면이고 모습도 구조물 쪽이면 벽 조각 후보(벽 선분이 아직 없어도) — 그 물체 자리로 벽을 지우지 않음
    if (o.ps >= 0.5) o.wall_like = o.struct_look = true;
    // 문·창·기둥 이름이고 그 모양 크기 안인가(아래 두 벽 규칙의 예외)
    const int sh = int(std::max_element(qsh + 1, qsh + 4) - qsh);
    const bool so_keep = o.pso >= 0.5 && (sh == 1 ? pl.hspan <= A.so_dw_w && pl.zhi - pl.zlo <= A.so_max_h : sh == 3 ? pl.hspan <= A.so_pillar : pl.hspan <= A.so_stairs);
    // 벽 선 없이 벽: 높고 넓은 세운 평면(가구가 벽 아래를 가리거나 문 둘레라 벽 선이 없는 곳)
    {
      const double cz = ceil_est_ > 0 ? ceil_est_ : A.ceil_z + 0.4;
      const bool tall = pl.zhi - pl.zlo >= A.tall_h || (pl.zlo <= A.tall_floor && pl.zhi >= cz - A.tall_ceil);
      const bool wide = pl.hspan >= (o.ps >= A.tall_ps ? A.tall_w : A.tall_w_any);
      if (tall && wide && !so_keep) { ++aps_.n_wall_tall; return true; }
    }
    if (!f.wall_segs || f.n_wall_segs <= 0) return false;
    const int stp = std::max(1, nc / 64);
    int near = 0, tot = 0;
    for (int i = 0; i < nc; i += stp, ++tot) {
      double dm = 1e9;
      for (int k = 0; k < f.n_wall_segs && dm > A.wall_d; ++k) dm = std::min(dm, segDist(o.cxyz[3 * i], o.cxyz[3 * i + 1], f.wall_segs + 4 * k));
      near += dm <= A.wall_d;
    }
    if (near < A.wall_frac * tot) return false;
    if (o.pso >= 0.5) {   // 문·창 조각(벽 선 위 평면이지만 지우지 않음) — 그 모양 크기 안일 때만
      if (so_keep) return false;
      ++aps_.n_so_big;
    }
    if (std::max(pl.hspan, pl.zhi - pl.zlo) >= A.wall_big) { ++aps_.n_wall; return true; }
    if (o.ps >= A.struct_p) { ++aps_.n_wall_name; return true; }
    o.wall_like = true;
    return false;
  };
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
      o.camd = std::sqrt((o.pos[0] - T[3]) * (o.pos[0] - T[3]) + (o.pos[1] - T[7]) * (o.pos[1] - T[7]) + (o.pos[2] - T[11]) * (o.pos[2] - T[11]));
      // 바닥 조각(점이 거의 다 바닥 높이): 물체 아님. 바닥에 깔리는 이름(러그·카펫·매트 — setFloorClasses)은 둠
      if (o.hi[2] < p_.floor_h && !floorLevel(o.cls)) { ++aps_.n_floor; continue; }
      if (f.wall_segs && f.n_wall_segs > 0) {
        // 벽 너머(창·유리문으로 본 바깥): 카메라 → 관측 중심 수평 선분이 벽 선분을 지나고 그 뒤로 through_d 넘게 더 가면 집 안 물체가 아님
        const double cx0 = T[3], cy0 = T[7], dx = o.pos[0] - cx0, dy = o.pos[1] - cy0, L = std::hypot(dx, dy);
        bool behind = false;
        for (int s2 = 0; s2 < f.n_wall_segs && !behind && L > 1e-6; ++s2) {
          const double* w = f.wall_segs + 4 * s2;
          const double ex = w[2] - w[0], ey = w[3] - w[1], den = dx * ey - dy * ex;
          if (std::fabs(den) < 1e-9) continue;
          const double t = ((w[0] - cx0) * ey - (w[1] - cy0) * ex) / den, u = ((w[0] - cx0) * dy - (w[1] - cy0) * dx) / den;
          behind = u >= 0 && u <= 1 && t > 0 && (1 - t) * L > p_.ap.through_d;
        }
        if (behind) { ++aps_.n_through; continue; }
      }
      ++aps_.n_obs;
      if (f.emb && f.emb_dim > 0) o.z = f.emb + size_t(k) * size_t(f.emb_dim);
      ViewQuality vq;
      vq.size_px = float(std::sqrt(double(np) * sk * sk / std::max(1e-9, double(sxu) * syv)));
      vq.trunc = o.trunc;
      vq.depth_m = float(zmed);
      vq.cam_w = float(cam_w);
      o.kappa = viewKappa(vq, p_.kap);
      if (apObsStructural(o)) continue;
      obs.push_back(std::move(o));
    }
  }
  // 2. 같은 물체(objprob): 관측마다 P(같은 물체)가 가장 큰 물체
  std::vector<int> obs_to(obs.size(), -1), obj_hit(objs_.size(), 0), touched(objs_.size(), 0);
  // objprob: 이름 없이 관측마다 P(같은 물체) 가 가장 큰 물체(≥ same_p). 여러 조각이 한 물체에 붙을 수 있음(FastSAM 은 한 물체를
  // 여러 마스크로 나눔) — 한 물체에 붙은 조각들은 첫(가장 큰) 조각에 합쳐 한 번 갱신. 접촉 ≥ 0.3 또는 P ≥ 0.2 인 물체는
  // '이번에 보임'(놓침으로 안 셈 — 조각 하나를 놓쳐도 물체는 보임)
  const ApParams& A = p_.ap;
  std::vector<float> sp;
  std::vector<float> mu;
  for (int a = 0; a < int(obs.size()); ++a) {
    Obs& o = obs[a];
    const int nc = int(o.cxyz.size() / 3);
    sp.clear();
    const int stp = std::max(1, nc / std::max(8, A.contact_samples));
    for (int i = 0; i < nc; i += stp) sp.insert(sp.end(), {o.cxyz[3 * i], o.cxyz[3 * i + 1], o.cxyz[3 * i + 2]});
    double best = A.same_p;
    bool blocked = false;
    int n_same = 0;   // P ≥ same_p 인 물체 수(둘 이상이면 두 물체에 걸친 덜 나뉜 마스크 — bridge_drop)
    for (int b = 0; b < int(objs_.size()); ++b) {
      MapObject& m = objs_[b];
      if (m.held_by >= 0 || !m.ap) continue;
      double g2 = 0;
      for (int k = 0; k < 3; ++k) {
        const double g = std::max({0.0, o.lo[k] - m.hi[k], m.lo[k] - o.hi[k]});
        g2 += g * g;
      }
      if (g2 > A.gate * A.gate) continue;
      if (o.ps >= 0 && !m.ap->post.empty() && apGroupProb(*m.ap, gmask) >= A.guard_obj && o.ps + o.pso <= A.guard_obs) continue;   // 구조물 막기(문·창 포함)
      // 벽 같은 조각 → 구조물 아닌 물체, 납작한 벽걸이 이름 크기 밖(액자 + 벽): 붙이지 않음. 붙을 만했으면(P ≥ same_p) 그 조각은 버림(새 물체 아님)
      const bool blk = (A.struct_look_block && o.struct_look && !m.ap->post.empty() && apGroupProb(*m.ap, gmask) < 0.5) ||
                       (flatNamed(m) && std::max(std::max(o.hi[0], m.hi[0]) - std::min(o.lo[0], m.lo[0]),
                                                 std::max(o.hi[1], m.hi[1]) - std::min(o.lo[1], m.lo[1])) > A.flat_max_w);
      if (A.name_veto > 0 && o.top_lab >= 0 && o.top_p >= A.name_veto && m.ap->top_lab >= 0 && m.ap->top_p >= A.name_veto &&
          !labelRelated(o.top_lab, m.ap->top_lab))
        continue;   // 이름 충돌
      const double cs = o.z ? apCosMax(*m.ap, o.z, f.emb_dim) : -2.0;
      const ApPair q = pairFeatures(o.lo, o.hi, o.pos, sp.data(), int(sp.size() / 3), m.lo, m.hi, m.pos, apContactIdx(m), cs, A);
      if (q.f[0] >= 0.3 || q.p >= 0.2) touched[b] = 1;
      if (blk) { blocked = blocked || q.p >= A.same_p; continue; }
      if (q.p >= A.same_p) ++n_same;
      if (q.p > best) { best = q.p; obs_to[a] = b; }
    }
    if (A.bridge_drop && n_same >= 2 && obs_to[a] >= 0) { obs_to[a] = -2; ++aps_.n_bridge; continue; }   // 버림(두 물체 모두 '보임')
    if (obs_to[a] >= 0) { obj_hit[obs_to[a]] = 1; ++aps_.n_assoc; }
    else if (blocked) { obs_to[a] = -2; ++aps_.n_blocked; }   // 버림
  }
  // 한 물체에 붙은 조각 합치기: 가장 큰(점 많은) 조각을 대표로, 상자 = 합집합, 중심 = 점 수 가중 평균
  std::vector<int> head(objs_.size(), -1);
  for (int a = 0; a < int(obs.size()); ++a) {
    const int b = obs_to[a];
    if (b < 0) continue;
    if (head[b] < 0 || obs[a].n > obs[head[b]].n) head[b] = a;
  }
  for (int a = 0; a < int(obs.size()); ++a) {
    const int b = obs_to[a];
    if (b < 0 || head[b] == a) continue;
    Obs& h = obs[head[b]];
    const Obs& o = obs[a];
    const double wh = h.n, wo = o.n;
    for (int k = 0; k < 3; ++k) {
      h.lo[k] = std::min(h.lo[k], o.lo[k]);
      h.hi[k] = std::max(h.hi[k], o.hi[k]);
      h.pos[k] = (h.pos[k] * wh + o.pos[k] * wo) / (wh + wo);
      h.ext[k] = h.hi[k] - h.lo[k];
    }
    h.n += o.n;
    h.trunc = h.trunc || o.trunc;
    h.zmed = std::min(h.zmed, o.zmed);
    h.score = std::max(h.score, o.score);
    obs[a].sec = true;
  }
  // 3. 갱신
  for (int a = 0; a < int(obs.size()); ++a) {
    const Obs& o = obs[a];
    DetAssoc& as = assoc_[o.det];
    as.n_valid = o.n;
    as.area_px = float(o.n) * o.sk * o.sk;
    as.depth_med = float(o.zmed);
    if (obs_to[a] == -2) continue;   // 벽 같은 조각·크기 밖(버림)
    if (obs_to[a] >= 0) {
      MapObject& m = objs_[obs_to[a]];
      as.obj_id = m.id;
      if (m.ap) {   // 조각마다 모습 하나(vMF r·이름 우도), 벽 선 위 평면 셈
        if (o.z) apAddView(*m.ap, &text_, p_.ap, o.z, f.emb_dim, o.kappa, f.stamp, cam6, false);
        ++m.ap->n_ap_obs;
        if (o.wall_like) ++m.ap->n_wall_obs;
        if (o.struct_look) ++m.ap->n_struct_look;
        m.ap->wall_like = 2 * m.ap->n_wall_obs >= m.ap->n_ap_obs;
        if (o.sec) continue;   // 갱신은 대표 조각(합친 상자)으로 한 번
      }
      const double dt_seen = f.stamp - m.last_seen;
      if (m.parent && dist3(m.pos, o.pos) > 0.3) m.parent = 0;
      if (m.last_kf != f.stamp) ++m.n_obs;
      m.last_kf = f.stamp;
      const double w = std::min<double>(m.n_obs, 20);
      const bool big = kindOf(m.cls) == kKindStatic || kindOf(m.cls) == kKindStructObj || std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], o.ext[0], o.ext[1]}) > p_.big;
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
      const bool fresh = m.first_seen == f.stamp;   // 이번 영상에 생긴 물체에 붙은 다음 조각: 상자 합집합
      for (int k = 0; k < 3 && !snap; ++k) {
        if (fresh) {
          m.lo[k] = std::min(m.lo[k], o.lo[k]);
          m.hi[k] = std::max(m.hi[k], o.hi[k]);
          m.pos[k] = 0.5 * (m.lo[k] + m.hi[k]);
          m.ext[k] = m.hi[k] - m.lo[k];
          continue;
        }
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
        } else if (m.ap) {
          // 칼만(축마다): 예측 P += q·dt, 관측 잡음 R = (r0 + r1·깊이)² (+ 잘렸으면 (반 폭)²), 이득 K = P/(P+R)
          const double K = opm::kalman_gain(m.ap->P[k], p_.ap_q_pos, dt_seen, opm::kalman_R(p_.ap_r0, p_.ap_r1, o.zmed, o.trunc, o.ext[k]));   // objprob_math.h
          m.pos[k] += K * (o.pos[k] - m.pos[k]);
          m.lo[k] += K * (o.lo[k] - m.lo[k]);
          m.hi[k] += K * (o.hi[k] - m.hi[k]);
          m.ext[k] = m.hi[k] - m.lo[k];
        } else {
          m.pos[k] = (m.pos[k] * (w - 1) + o.pos[k]) / w;
          m.ext[k] = (m.ext[k] * (w - 1) + o.ext[k]) / w;
          m.lo[k] = (m.lo[k] * (w - 1) + o.lo[k]) / w;
          m.hi[k] = (m.hi[k] * (w - 1) + o.hi[k]) / w;
        }
        if (big && m.ap) {   // 큰 것: 자리는 상자 합집합, 분산만 칼만(상자 폭의 1/4 을 관측 잡음에)
          opm::kalman_big(m.ap->P[k], p_.ap_q_pos, dt_seen, p_.ap_r0 + p_.ap_r1 * o.zmed, m.ext[k]);   // objprob_math.h
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
    m.ap = std::make_shared<ApState>();
    const double sd = p_.ap_r0 + p_.ap_r1 * o.zmed;
    for (int k = 0; k < 3; ++k) m.ap->P[k] = sd * sd + (o.trunc ? 0.25 * o.ext[k] * o.ext[k] : 0.0);
    if (o.z) apAddView(*m.ap, &text_, p_.ap, o.z, f.emb_dim, o.kappa, f.stamp, cam6, false);
    m.ap->n_ap_obs = 1;
    m.ap->n_wall_obs = o.wall_like;
    m.ap->n_struct_look = o.struct_look;
    m.ap->wall_like = o.wall_like;
    ++aps_.n_new;
    objs_.push_back(m);
    obj_hit.push_back(1);
    touched.push_back(1);
    // objprob: 같은 영상의 뒤 조각이 이 새 물체에 붙을 수 있게(아직 구름이 없으니 상자·임베딩으로만) — 앞 관측이 만든 새 물체도 후보
    if (p_.ap.frame_group)
      for (int b2 = a + 1; b2 < int(obs.size()); ++b2) {
        if (obs_to[b2] != -1) continue;
        const Obs& q = obs[b2];
        double g2 = 0;
        for (int k = 0; k < 3; ++k) {
          const double g = std::max({0.0, q.lo[k] - o.hi[k], o.lo[k] - q.hi[k]});
          g2 += g * g;
        }
        if (g2 > 0.02 * 0.02 || !q.z || !o.z) continue;
        // 닿은 두 조각: 이웃 조각 접촉을 1 로 보고 같은 로지스틱(구름 대신 상자 맞닿음)
        static const std::vector<uint64_t> none;
        ApPair pr = pairFeatures(q.lo, q.hi, q.pos, nullptr, 0, o.lo, o.hi, o.pos, none, apDot(q.z, o.z, f.emb_dim), p_.ap);
        pr.f[0] = 1.0;
        if (1.0 / (1.0 + std::exp(-apLogit(pr, p_.ap))) >= p_.ap.same_p) obs_to[b2] = int(objs_.size()) - 1;
      }
    event(f.stamp, objs_.back(), 0);
  }
  // 살펴본 정도: 물체에 붙은 관측마다(조각 포함) 가장 가까이 본 거리·시점
  if (p_.insp.on)
    for (const Obs& o : obs) {
      const uint32_t id = assoc_[o.det].obj_id;
      if (!id) continue;
      for (MapObject& m : objs_)
        if (m.id == id) { inspObserve(m.insp, cam6, o.camd, p_.insp); break; }
    }
  // objprob: 이번에 본 물체의 이름 사후를 다시(이름 = 다시 셀 수 있는 캐시, 벡터가 원본)
  for (MapObject& m : objs_)
    if (m.ap && m.last_kf == f.stamp) apRename(m);
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
      if (touched[b]) continue;   // objprob: 다른 조각이 이 물체에 닿음(물체는 보임)
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
  // objprob: 합친 덩어리가 구조물(벽 크기 평면·천장·구조물 이름)이면 지움
  for (MapObject& m : objs_) if (m.ap) m.ap->drop = apStructObject(m);
  // 5. 오래된 후보 버리기
  objs_.erase(std::remove_if(objs_.begin(), objs_.end(),
                             [&](const MapObject& m) {
                               if (m.ap && m.ap->drop) { ++aps_.n_obj_struct; return true; }
                               if (!m.confirmed) return f.stamp - m.last_seen > p_.prune_s;
                               // 몇 번 안 보이고 사라진 것은 헛검출(조각)로 보고 지움 — 옮겨짐 잇기 후보가 되지 않게
                               return m.state == SM_GONE && !m.moved && int(m.n_obs) < p_.spurious_obs;
                             }),
              objs_.end());
  // 살펴본 정도: 윗면 있는 물체(든 것·사라짐 빼고)의 윗면 칸을 이 깊이 영상으로 표시
  if (p_.insp.on && (f.depth_m || f.depth_mm)) {
    InspectFrame inf;
    inf.w = f.w; inf.h = f.h; inf.depth_m = f.depth_m; inf.depth_mm = f.depth_mm;
    inf.fx = f.fx; inf.fy = f.fy; inf.cx = f.cx; inf.cy = f.cy;
    inf.T_mc = f.T_mc;
    inf.zmin = p_.zmin; inf.zmax = p_.zmax;
    for (MapObject& m : objs_)
      if (m.held_by < 0 && m.state != SM_GONE && inspHasTop(m.lo, m.hi, p_.insp)) inspTop(m.insp, m.lo, m.hi, inf, p_.insp);
  }
  markView(f);
  // 6. 중복 병합은 다음 keyframe 앞 apMergePass(objprob, 이름 없이)
}

// ---------------- objprob ----------------

const std::vector<uint64_t>& ObjectMap::apContactIdx(MapObject& m) {
  ApState& s = *m.ap;
  if (s.cidx_ver != m.cloud.version) {
    std::vector<float> xyz;
    if (m.cloud.data) {
      const auto& pts = m.cloud.data->pts;
      xyz.reserve(3 * pts.size());
      for (const CloudPt& q : pts) xyz.insert(xyz.end(), {float(m.cloud.org[0] + q.x), float(m.cloud.org[1] + q.y), float(m.cloud.org[2] + q.z)});
    }
    apBuildContact(xyz.data(), int(xyz.size() / 3), p_.ap.contact_cell, &s.cidx);
    s.cidx_ver = m.cloud.version;
  }
  return s.cidx;
}

namespace {
// 구름 점(map) 표본 최대 n 개
void cloudSample(const MapObject& m, int n, std::vector<float>* out) {
  out->clear();
  if (!m.cloud.data) return;
  const auto& pts = m.cloud.data->pts;
  const size_t stp = std::max<size_t>(1, pts.size() / size_t(std::max(1, n)));
  for (size_t i = 0; i < pts.size(); i += stp)
    out->insert(out->end(), {float(m.cloud.org[0] + pts[i].x), float(m.cloud.org[1] + pts[i].y), float(m.cloud.org[2] + pts[i].z)});
}
}  // namespace

// 물체 쌍: 구름이 작은 쪽을 표본으로 큰 쪽 접촉 칸에(구름이 아직 없으면 상자만), cos = 서로의 μ 를 상대 모습들에 맞춘 최대
ApPair ObjectMap::apPairObj(MapObject& a, MapObject& b) {
  MapObject& sm = a.cloud.size() <= b.cloud.size() ? a : b;
  MapObject& lg = &sm == &a ? b : a;
  thread_local std::vector<float> sp, mu;
  cloudSample(sm, p_.ap.contact_samples, &sp);
  double cs = -2;
  if (apMu(*a.ap, &mu)) cs = std::max(cs, apCosMax(*b.ap, mu.data(), int(mu.size())));
  if (apMu(*b.ap, &mu)) cs = std::max(cs, apCosMax(*a.ap, mu.data(), int(mu.size())));
  ApPair q = pairFeatures(sm.lo, sm.hi, sm.pos, sp.data(), int(sp.size() / 3), lg.lo, lg.hi, lg.pos, apContactIdx(lg), cs, p_.ap, true);
  if (!a.ap->post.empty() && a.ap->post.size() == b.ap->post.size()) {   // 이름 분포 겹침
    q.f[6] = opm::bhattacharyya<OpmStd, double>(a.ap->post.data(), b.ap->post.data(), int(a.ap->post.size()), 0.0, 0.0, 0);   // objprob_math.h
    q.logit = apLogit(q, p_.ap, true);
    q.p = opm::sigmoid<OpmStd>(q.logit);
  }
  return q;
}

// 물체끼리 같은 것 판정(이름 없이): 상자 틈 gate 안 쌍마다 P(같음), 큰 것부터 하나씩 합침(한 판에 물체 하나는 한 번만 — 바뀐 상자로
// 다음 keyframe 에 다시). 합친 물체는 통째 다시 담기를 기다림(need_whole)
void ObjectMap::apMergePass(double t) {
  const ApParams& A = p_.ap;
  smask_.assign(size_t(std::max(0, text_.n_labels)), 0);
  for (int l = 0; l < text_.n_labels; ++l) smask_[size_t(l)] = kindOf(l) == kKindStructure || kindOf(l) == kKindStructObj;   // 막기: 구조물 쪽 전부
  std::vector<std::tuple<double, uint32_t, uint32_t>> cand;
  for (size_t i = 0; i < objs_.size(); ++i) {
    MapObject& a = objs_[i];
    if (!a.ap || a.held_by >= 0 || a.state == SM_GONE) continue;
    for (size_t j = i + 1; j < objs_.size(); ++j) {
      MapObject& b = objs_[j];
      if (!b.ap || b.held_by >= 0 || b.state == SM_GONE) continue;
      double g2 = 0;
      bool too_big = false;
      for (int k = 0; k < 3; ++k) {
        const double g = std::max({0.0, a.lo[k] - b.hi[k], b.lo[k] - a.hi[k]});
        g2 += g * g;
        too_big = too_big || std::max(a.hi[k], b.hi[k]) - std::min(a.lo[k], b.lo[k]) > p_.max_ext;
      }
      if (g2 > A.gate * A.gate || too_big) continue;
      if (!a.ap->post.empty() && !b.ap->post.empty()) {   // 구조물 막기(한쪽은 구조물, 다른 쪽은 아님)
        const double sa = apGroupProb(*a.ap, smask_), sb = apGroupProb(*b.ap, smask_);
        if ((sa >= A.guard_obj && sb <= A.guard_obs) || (sb >= A.guard_obj && sa <= A.guard_obs)) continue;
        if (A.struct_look_block) {   // 벽 같은 관측이 절반 넘는 것 ↔ 구조물 아닌 물체
          const bool la = 2 * a.ap->n_struct_look > a.ap->n_ap_obs, lb = 2 * b.ap->n_struct_look > b.ap->n_ap_obs;
          if ((la && !lb && sb < 0.5) || (lb && !la && sa < 0.5)) continue;
        }
      }
      if ((flatNamed(a) || flatNamed(b)) &&
          std::max(std::max(a.hi[0], b.hi[0]) - std::min(a.lo[0], b.lo[0]), std::max(a.hi[1], b.hi[1]) - std::min(a.lo[1], b.lo[1])) > A.flat_max_w)
        continue;   // 납작한 벽걸이 이름 크기 밖
      const ApPair q = apPairObj(a, b);
      static const bool log_m = std::getenv("SM_AP_LOG") != nullptr;
      if (log_m && q.p > 0.2)
        std::fprintf(stderr, "[ap-pair] t=%.1f O%u O%u contact %.2f gap %.2f cdist %.2f cos %.3f ov %.2f sup %.0f p %.2f\n", t, a.id, b.id, q.f[0], q.f[1],
                     q.f[2], q.f[3] + A.cos0, q.f[4], q.f[5], q.p);
      if (q.p >= A.merge_p) cand.emplace_back(q.p, a.id, b.id);
    }
  }
  if (cand.empty()) return;
  std::sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) { return std::get<0>(x) > std::get<0>(y); });
  std::vector<uint32_t> used;
  for (auto& [pp, ia, ib] : cand) {
    if (std::find(used.begin(), used.end(), ia) != used.end() || std::find(used.begin(), used.end(), ib) != used.end()) continue;
    auto ai = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& o) { return o.id == ia; });
    auto bi = std::find_if(objs_.begin(), objs_.end(), [&](const MapObject& o) { return o.id == ib; });
    if (ai == objs_.end() || bi == objs_.end()) continue;
    if (bi->n_obs > ai->n_obs || (bi->n_obs == ai->n_obs && ib < ia)) std::swap(ai, bi);   // 관측이 많은(같으면 먼저 본) 쪽을 남김
    used.push_back(ia);
    used.push_back(ib);
    const uint32_t kid = ai->id, did = bi->id;
    da::absorbObject(*ai, *bi, p_, t, &kinds_);
    apMerge(*ai->ap, *bi->ap, A);
    ai->confirmed = ai->confirmed || bi->confirmed;
    apRename(*ai);
    ++aps_.n_merge;
    static const bool log_m = std::getenv("SM_AP_LOG") != nullptr;
    if (log_m) std::fprintf(stderr, "[ap-merge] t=%.1f keep O%u drop O%u p %.2f\n", t, kid, did, pp);
    objs_.erase(bi);
    remapId(did, kid);
    for (const MapObject& x : objs_) if (x.id == kid) { event(t, x, 7); break; }
  }
}

void ObjectMap::apRename(MapObject& m) {
  if (!m.ap || !text_.ready()) return;
  const double size = std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]});
  ApState& s = *m.ap;
  if (s.geo == 3) {   // 벽 선 위 아주 얇은 평면: 문·창·계단·납작한 물체(액자·TV …) 쪽이 그럴듯함
    s.L_geo.assign(size_t(text_.n_labels), 0.f);
    for (int l = 0; l < text_.n_labels; ++l)
      if (kindOf(l) == kKindStructObj || (size_t(l) < text_.flat_ok.size() && text_.flat_ok[size_t(l)])) s.L_geo[size_t(l)] = float(p_.ap.geo_w);
  } else {
    s.L_geo.clear();
  }
  apName(s, text_, p_.ap, size);
  if (m.ap->name_lab >= 0) m.cls = m.ap->name_lab;
}

// 라벨 a 와 b 가 같거나 한쪽이 다른 쪽의 상위어(부모 사슬)
bool ObjectMap::labelRelated(int a, int b) const {
  if (a == b) return true;
  auto up = [&](int x, int y) {
    for (int g = 0; x >= 0 && g < 8; ++g) {
      x = size_t(x) < text_.parent.size() ? text_.parent[size_t(x)] : -1;
      if (x == y) return true;
    }
    return false;
  };
  return up(a, b) || up(b, a);
}

bool ObjectMap::apStructObject(MapObject& m) {
  if (!m.ap || m.held_by >= 0) return false;
  ApState& s = *m.ap;
  const ApParams& A = p_.ap;
  s.hide = false;
  // 기하(구름이 바뀌었을 때만 다시)
  if (s.geo_ver != m.cloud.version && m.cloud.size() >= 30) {
    s.geo_ver = m.cloud.version;
    thread_local std::vector<float> P;
    cloudSample(m, 800, &P);
    const int n = int(P.size() / 3);
    const ApPlane pl = apPlaneFit(P.data(), n, A.obj_tau, apSeed(m.id, m.cloud.version), A.ransac_iters);
    s.geo = 0;
    const double nz = std::fabs(pl.n[2]);
    const bool plane = pl.ok && pl.inl >= A.obj_plane_inl;   // 지배 평면
    s.vinl = pl.ok && nz < A.wall_vert ? float(pl.inl) : 0.f;   // 세운 평면 안쪽 비율(크기 밖 이름 물체를 벽으로 볼 때)
    if (plane && pl.thick < A.obj_thick) {
      if (nz < A.wall_vert && (pl.hspan >= A.obj_wall_span || (pl.zhi >= A.obj_wall_top && pl.zhi - pl.zlo >= A.obj_wall_h))) s.geo = 1;
      else if (nz > A.horiz && pl.zmed > A.ceil_z) s.geo = 2;
    }
    // 천장 덩어리: 점 대부분이 천장 띠 안이고 수평으로 넓음(평면 맞춤이 나빠도 — 낮은 카메라로 비스듬히 본 천장·벽 모서리)
    if (!s.geo && ceil_est_ > 0 && n > 0) {
      int up = 0;
      for (int i = 0; i < n; ++i) up += P[size_t(3 * i + 2)] > ceil_est_ - A.ceil_band;
      if (up >= A.ceil_frac * n && std::max(m.hi[0] - m.lo[0], m.hi[1] - m.lo[1]) >= A.ceil_wide) s.geo = 2;
    }
    // 모양(크기 확인·벽에 붙은 덩어리): 수평 주방향 긴·짧은 폭(5~95 백분위), 높이, 벽 선분 wall_d 안 점 비율
    if (n > 0) {
      double mx = 0, my = 0;
      for (int i = 0; i < n; ++i) { mx += P[size_t(3 * i)]; my += P[size_t(3 * i + 1)]; }
      mx /= n; my /= n;
      double sxx = 0, sxy = 0, syy = 0;
      for (int i = 0; i < n; ++i) {
        const double dx = P[size_t(3 * i)] - mx, dy = P[size_t(3 * i + 1)] - my;
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
      }
      const double th = 0.5 * std::atan2(2 * sxy, sxx - syy), c = std::cos(th), sn = std::sin(th);
      std::vector<double> u(static_cast<size_t>(n)), v(static_cast<size_t>(n)), z(static_cast<size_t>(n));
      int nw = 0;
      for (int i = 0; i < n; ++i) {
        const double dx = P[size_t(3 * i)] - mx, dy = P[size_t(3 * i + 1)] - my;
        u[size_t(i)] = dx * c + dy * sn;
        v[size_t(i)] = -dx * sn + dy * c;
        z[size_t(i)] = P[size_t(3 * i + 2)];
        double dm = 1e9;
        for (size_t k = 0; k + 3 < wsegs_.size() && dm > A.wall_d; k += 4) dm = std::min(dm, segDist(P[size_t(3 * i)], P[size_t(3 * i + 1)], &wsegs_[k]));
        nw += dm <= A.wall_d;
      }
      auto pct = [](std::vector<double>& x, double q) {
        const size_t k = std::min(x.size() - 1, size_t(q * double(x.size() - 1) + 0.5));
        std::nth_element(x.begin(), x.begin() + long(k), x.end());
        return x[k];
      };
      const double su = pct(u, 0.95) - pct(u, 0.05), sv = pct(v, 0.95) - pct(v, 0.05);
      s.maj = float(std::max(su, sv));
      s.minr = float(std::min(su, sv));
      s.hgt = float(pct(z, 0.95) - pct(z, 0.05));
      s.fw = float(double(nw) / n);
    }
    // 벽 선 위 아주 얇은 평면(문짝·창유리)
    if (!s.geo && plane && pl.thick < A.obj_flat_thick && nz < A.wall_vert && !wsegs_.empty() && n > 0) {
      int near = 0;
      for (int i = 0; i < n; ++i) {
        double dm = 1e9;
        for (size_t k = 0; k + 3 < wsegs_.size() && dm > A.obj_flat_d; k += 4) dm = std::min(dm, segDist(P[size_t(3 * i)], P[size_t(3 * i + 1)], &wsegs_[k]));
        near += dm <= A.obj_flat_d;
      }
      if (near >= A.obj_flat_frac * n) s.geo = 3;
    }
  }
  double qso = 0;   // 구조 물체(문·창·계단) 사후 — 벽 크기 평면·벽 선 위 평면이어도 남김
  for (size_t l = 0; l < s.post.size(); ++l) if (kindOf(int(l)) == kKindStructObj) qso += s.post[l];
  if (qso >= 0.5 && s.geo != 2) {   // 그 모양 크기 안일 때만 남김(벽 조각이 door·window·pillar 로 불린 것은 아래 규칙으로)
    double qsh[4] = {0, 0, 0, 0};
    for (size_t l = 0; l < s.post.size(); ++l)
      if (kindOf(int(l)) == kKindStructObj) qsh[l < text_.so_shape.size() ? text_.so_shape[l] & 3 : 0] += s.post[l];
    const int sh = int(std::max_element(qsh + 1, qsh + 4) - qsh);
    const bool fits = sh == 1   ? s.maj <= A.so_dw_w && s.minr <= A.so_dw_thick && s.hgt <= A.so_max_h
                      : sh == 3 ? s.maj <= A.so_pillar
                                : s.maj <= A.so_stairs;
    if (sh == 1 && s.maj < A.so_dw_min && int(m.n_obs) >= A.so_min_obs) return true;   // 문·창 이름인 가는 조각(문틀·창틀 모서리) — 문·창 노드 아님
    if (fits) return false;
    // 너무 큼: 벽 조각 모양(평면·천장·벽에 붙음·세운 평면이 반 넘음)이면 숨김(노드로 안 내보냄). 지우지 않는 까닭: 지우면 그 자리의 다음 벽
    // 조각들이 새 작은 문·창 물체가 되어 다시 남음 — 숨긴 덩어리가 계속 받아 둔다
    const double lim = sh == 1 ? A.so_dw_w : sh == 3 ? A.so_pillar : A.so_stairs;
    s.hide = s.geo != 0 || (s.fw >= A.wallhug_frac && s.maj >= A.wallhug_w) || s.vinl >= A.big_vinl || s.maj > A.so_hide_k * lim;
    return false;
  }
  if (flatNamed(m) && s.maj > A.flat_max_w && (s.geo != 0 || s.fw >= 0.5 || s.vinl >= A.big_vinl)) { s.hide = true; return false; }   // 납작한 벽걸이 이름인데 벽 크기
  if (s.fw >= A.wallhug_frac && s.maj >= A.wallhug_w) { s.hide = true; return false; }   // 벽에 붙은 큰 덩어리(벽 모서리 L 자 등)
  if (s.geo == 3) {   // 납작한 물체 이름(액자·TV …)이면 남김(문·창은 위에서 남김)
    double q = qso;
    for (size_t l = 0; l < s.post.size() && l < text_.flat_ok.size(); ++l) if (text_.flat_ok[l]) q += s.post[l];
    if (q < A.flat_keep_p) return true;
  }
  if (s.geo == 1 || s.geo == 2) return true;
  if (s.n_whole < A.struct_obj_min_views || s.post.empty()) return false;
  double q = 0;
  for (size_t l = 0; l < s.post.size(); ++l) if (kindOf(int(l)) == kKindStructure) q += s.post[l];
  return q >= A.struct_obj_p;
}

void ObjectMap::observeName(uint32_t id, int lab, double log_lr) {
  for (MapObject& m : objs_)
    if (m.id == id && m.ap) {
      apObserveName(*m.ap, text_.n_labels, lab, log_lr);
      apRename(m);
      return;
    }
}

void ObjectMap::addWholeView(uint32_t id, const float* z, int dim, double kappa, double stamp, const double cam[6]) {
  for (MapObject& m : objs_)
    if (m.id == id && m.ap) {
      apAddView(*m.ap, &text_, p_.ap, z, dim, kappa, stamp, cam, true);
      apRename(m);
      ++aps_.n_reenc_done;
      return;
    }
}

void ObjectMap::buildReencode(const ObjFrame& f, int img_w, int img_h, int mw, int mh, float msx, float msy, float mox, float moy,
                              std::vector<ReencReq>* out) {
  out->clear();
  if (!(f.depth_m || f.depth_mm) || mw <= 0 || mh <= 0 || img_w <= 0 || img_h <= 0) return;
  const ApParams& A = p_.ap;
  const double* T = f.T_mc;
  const double sxu = double(f.w) / img_w, syv = double(f.h) / img_h;   // 검출 화소 → 깊이 화소
  const size_t words = (size_t(mw) * mh + 31) / 32;
  struct Cand { double prio; ReencReq r; };
  std::vector<Cand> cands;
  std::vector<uint8_t> grid;
  for (MapObject& m : objs_) {
    if (!m.ap || !m.confirmed || m.held_by >= 0 || m.state == SM_GONE || m.last_kf != f.stamp || m.cloud.size() < 30) continue;
    const auto& pts = m.cloud.data->pts;
    const size_t stp = std::max<size_t>(1, pts.size() / 3000);
    grid.assign(size_t(mw) * mh, 0);
    int tot = 0, vis = 0;
    bool trunc = false;
    std::vector<float> zs;
    double u0 = 1e9, v0 = 1e9, u1 = -1e9, v1 = -1e9;
    for (size_t i = 0; i < pts.size(); i += stp) {
      ++tot;
      const double p[3] = {m.cloud.org[0] + pts[i].x - T[3], m.cloud.org[1] + pts[i].y - T[7], m.cloud.org[2] + pts[i].z - T[11]};
      const double cx = T[0] * p[0] + T[4] * p[1] + T[8] * p[2], cy = T[1] * p[0] + T[5] * p[1] + T[9] * p[2],
                   cz = T[2] * p[0] + T[6] * p[1] + T[10] * p[2];
      if (cz < 0.15) continue;
      const double ud = f.fx * cx / cz + f.cx, vd = f.fy * cy / cz + f.cy;   // 깊이 화소
      if (ud < 0 || vd < 0 || ud >= f.w || vd >= f.h) { trunc = true; continue; }
      const size_t di = size_t(int(vd)) * f.w + size_t(int(ud));
      const float d = f.depth_m ? f.depth_m[di] : f.depth_mm[di] * 1e-3f;
      if (!(d > 0) || d < cz - 0.08) continue;   // 앞에 다른 것(가림)·깊이 없음
      ++vis;
      zs.push_back(float(cz));
      // 검출 화소 → 마스크 칸, 복셀 크기만큼 칠함
      const double ui = ud / sxu, vi = vd / syv;
      const double rpx = 0.6 * p_.voxel * f.fx / cz / sxu;
      const int i0 = int(std::floor((ui - rpx - mox) / msx)), i1 = int(std::floor((ui + rpx - mox) / msx));
      const int j0 = int(std::floor((vi - rpx - moy) / msy)), j1 = int(std::floor((vi + rpx - moy) / msy));
      for (int jj = std::max(0, j0); jj <= std::min(mh - 1, j1); ++jj)
        for (int ii = std::max(0, i0); ii <= std::min(mw - 1, i1); ++ii) grid[size_t(jj) * mw + ii] = 1;
      u0 = std::min(u0, ui); u1 = std::max(u1, ui); v0 = std::min(v0, vi); v1 = std::max(v1, vi);
      if (ui <= 2 || vi <= 2 || ui >= img_w - 3 || vi >= img_h - 3) trunc = true;
    }
    if (tot == 0 || vis < 12) continue;
    const double vfrac = double(vis) / tot;
    if (vfrac < A.reenc_min_vis) continue;
    // 닫기(이웃 칸 하나 넓힘): 점 사이 틈을 메움
    std::vector<uint8_t> g2 = grid;
    long area = 0;
    for (int jj = 0; jj < mh; ++jj)
      for (int ii = 0; ii < mw; ++ii) {
        if (grid[size_t(jj) * mw + ii]) continue;
        int nb = 0;
        for (int dj = -1; dj <= 1; ++dj)
          for (int di2 = -1; di2 <= 1; ++di2) {
            const int a = ii + di2, b = jj + dj;
            nb += a >= 0 && b >= 0 && a < mw && b < mh && grid[size_t(b) * mw + a];
          }
        if (nb >= 3) g2[size_t(jj) * mw + ii] = 1;
      }
    for (uint8_t v : g2) area += v;
    const double size_px = std::sqrt(double(area) * msx * msy);
    if (size_px < A.reenc_min_px) continue;
    std::nth_element(zs.begin(), zs.begin() + long(zs.size() / 2), zs.end());
    ViewQuality vq;
    vq.size_px = float(size_px);
    vq.trunc = trunc;
    vq.depth_m = zs[zs.size() / 2];
    vq.vis = float(vfrac);
    vq.cam_w = float(cam_w_);
    const double kap = viewKappa(vq, p_.kap);
    const bool need = m.ap->need_whole;
    if (!need && kap < A.reenc_gain * m.ap->whole_kappa) continue;
    Cand c;
    c.prio = (need ? 1e6 : 0) + kap / std::max(1.0, m.ap->whole_kappa);
    c.r.id = m.id;
    c.r.kappa = float(kap);
    c.r.box[0] = float(std::max(0.0, u0 - 2)); c.r.box[1] = float(std::max(0.0, v0 - 2));
    c.r.box[2] = float(std::min(double(img_w), u1 + 2)); c.r.box[3] = float(std::min(double(img_h), v1 + 2));
    for (int k = 0; k < 6; ++k) c.r.cam[k] = k < 3 ? T[4 * k + 3] : T[4 * (k - 3) + 2];
    c.r.bits.assign(words, 0u);
    for (size_t q = 0; q < g2.size(); ++q) if (g2[q]) c.r.bits[q >> 5] |= 1u << (q & 31);
    cands.push_back(std::move(c));
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.prio > b.prio; });
  for (size_t i = 0; i < cands.size() && int(i) < A.reenc_max; ++i) out->push_back(std::move(cands[i].r));
  aps_.n_reenc_req += long(out->size());
}

}  // namespace scenemap
