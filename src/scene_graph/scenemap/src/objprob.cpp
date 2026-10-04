// A′ 물체 모델(include/scenemap/objprob.hpp): vMF 임베딩 사후·이름 범주 사후·같은 것 로지스틱·평면 맞춤.
#include "scenemap/objprob.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace scenemap {
namespace {

uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (e <= 0) {   // 작은 수: 0 으로(임베딩 성분에는 충분)
    if (e < -10) return uint16_t(sign);
    m |= 0x800000u;
    const int sh = 14 - e;
    return uint16_t(sign | ((m + (1u << (sh - 1))) >> sh));
  }
  if (e >= 31) return uint16_t(sign | 0x7c00u);
  const uint32_t r = m + 0x1000u;   // 반올림
  if (r & 0x800000u) { m = 0; ++e; if (e >= 31) return uint16_t(sign | 0x7c00u); }
  else m = r;
  return uint16_t(sign | (uint32_t(e) << 10) | (m >> 13));
}

float h2f(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
  if (e == 0) {
    if (m == 0) x = sign;
    else {   // 비정규
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3ffu;
      x = sign | (e << 23) | (m << 13);
    }
  } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
  else x = sign | ((e + 127 - 15) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// 3×3 대칭 고유 분해(야코비): 고유값 오름차순 ev, 고유 벡터 열 V
void eigSym3(double A[3][3], double ev[3], double V[3][3]) {
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) V[i][j] = i == j;
  for (int it = 0; it < 32; ++it) {
    int p = 0, q = 1;
    double best = std::fabs(A[0][1]);
    if (std::fabs(A[0][2]) > best) { best = std::fabs(A[0][2]); p = 0; q = 2; }
    if (std::fabs(A[1][2]) > best) { best = std::fabs(A[1][2]); p = 1; q = 2; }
    if (best < 1e-12) break;
    const double th = 0.5 * std::atan2(2 * A[p][q], A[q][q] - A[p][p]);
    const double c = std::cos(th), s = std::sin(th);
    for (int k = 0; k < 3; ++k) {   // A ← Jᵀ A J
      const double akp = A[k][p], akq = A[k][q];
      A[k][p] = c * akp - s * akq;
      A[k][q] = s * akp + c * akq;
    }
    for (int k = 0; k < 3; ++k) {
      const double apk = A[p][k], aqk = A[q][k];
      A[p][k] = c * apk - s * aqk;
      A[q][k] = s * apk + c * aqk;
    }
    for (int k = 0; k < 3; ++k) {
      const double vkp = V[k][p], vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }
  int o[3] = {0, 1, 2};
  std::sort(o, o + 3, [&](int a, int b) { return A[a][a] < A[b][b]; });
  double W[3][3];
  for (int k = 0; k < 3; ++k) {
    ev[k] = A[o[k]][o[k]];
    for (int i = 0; i < 3; ++i) W[i][k] = V[i][o[k]];
  }
  std::memcpy(V, W, sizeof W);
}

double pctOf(std::vector<double>& v, double q) {
  const size_t k = std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + long(k), v.end());
  return v[k];
}

inline uint64_t cellKey3(int64_t i, int64_t j, int64_t k) {
  auto q = [](int64_t v) { return uint64_t(v + (1 << 20)) & 0x1FFFFF; };
  return q(i) | q(j) << 21 | q(k) << 42;
}

}  // namespace

ApPlane apPlaneFit(const float* xyz, int n) {
  ApPlane P;
  if (n < 8) return P;
  double m[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) m[k] += xyz[3 * i + k];
  for (double& v : m) v /= n;
  double C[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int i = 0; i < n; ++i) {
    const double d[3] = {xyz[3 * i] - m[0], xyz[3 * i + 1] - m[1], xyz[3 * i + 2] - m[2]};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a][b] += d[a] * d[b];
  }
  for (auto& r : C) for (double& v : r) v /= n;
  double ev[3], V[3][3];
  eigSym3(C, ev, V);
  for (int k = 0; k < 3; ++k) P.n[k] = V[k][0];
  P.thick = std::sqrt(std::max(0.0, ev[0]));
  // 주축 두 방향 폭(10~90 백분위)·높이·수평 폭
  std::vector<double> a(static_cast<size_t>(n)), b(static_cast<size_t>(n)), z(static_cast<size_t>(n)), h(static_cast<size_t>(n));
  // 수평 주방향: 가로 성분 공분산의 큰 고유 방향
  double sxx = 0, sxy = 0, syy = 0;
  for (int i = 0; i < n; ++i) {
    const double dx = xyz[3 * i] - m[0], dy = xyz[3 * i + 1] - m[1];
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const double th = 0.5 * std::atan2(2 * sxy, sxx - syy), hc = std::cos(th), hs = std::sin(th);
  for (int i = 0; i < n; ++i) {
    const double d[3] = {xyz[3 * i] - m[0], xyz[3 * i + 1] - m[1], xyz[3 * i + 2] - m[2]};
    a[size_t(i)] = d[0] * V[0][2] + d[1] * V[1][2] + d[2] * V[2][2];
    b[size_t(i)] = d[0] * V[0][1] + d[1] * V[1][1] + d[2] * V[2][1];
    z[size_t(i)] = xyz[3 * i + 2];
    h[size_t(i)] = d[0] * hc + d[1] * hs;
  }
  P.span1 = pctOf(a, 0.9) - pctOf(a, 0.1);
  P.span2 = pctOf(b, 0.9) - pctOf(b, 0.1);
  P.hspan = pctOf(h, 0.9) - pctOf(h, 0.1);
  P.zlo = pctOf(z, 0.1);
  P.zhi = pctOf(z, 0.9);
  P.zmed = pctOf(z, 0.5);
  P.ok = true;
  return P;
}

void apToF16(const float* in, uint16_t* out, int n) {
  for (int i = 0; i < n; ++i) out[i] = f2h(in[i]);
}

void apNormalize(std::vector<float>& v) {
  double s = 0;
  for (float x : v) s += double(x) * x;
  s = std::sqrt(std::max(s, 1e-30));
  for (float& x : v) x = float(x / s);
}

double apDot(const float* a, const float* b, int d) {
  double s = 0;
  for (int i = 0; i < d; ++i) s += double(a[i]) * b[i];
  return s;
}

bool apMu(const ApState& s, std::vector<float>* mu, double* conf) {
  const std::vector<float>& r = s.k_whole > 0 ? s.r_whole : s.r_frag;
  if (r.empty()) return false;
  double n = 0;
  for (float x : r) n += double(x) * x;
  n = std::sqrt(n);
  if (!(n > 0)) return false;
  mu->resize(r.size());
  for (size_t i = 0; i < r.size(); ++i) (*mu)[i] = float(r[i] / n);
  if (conf) *conf = n;
  return true;
}

double apCosMax(const ApState& s, const float* z, int d) {
  double best = -1;
  for (const std::vector<float>* r : {&s.r_frag, &s.r_whole}) {
    if (int(r->size()) != d) continue;
    double n = 0, dot = 0;
    for (int i = 0; i < d; ++i) { n += double((*r)[i]) * (*r)[i]; dot += double((*r)[i]) * z[i]; }
    if (n > 0) best = std::max(best, dot / std::sqrt(n));
  }
  for (const ApView& v : s.views) {
    if (int(v.z.size()) != d) continue;
    double dot = 0;
    for (int i = 0; i < d; ++i) dot += double(h2f(v.z[size_t(i)])) * z[i];
    best = std::max(best, dot);
  }
  return best;
}

void apLabelLogLik(const ApText& T, const float* z, std::vector<float>* out) {
  out->assign(size_t(T.n_labels), -1e30f);
  const int rows = int(T.row_label.size());
  std::vector<float> best(size_t(T.n_labels), -2.f);
  for (int r = 0; r < rows; ++r) {
    const int l = T.row_label[size_t(r)];
    if (l < 0 || l >= T.n_labels) continue;
    const float c = float(apDot(T.text.data() + size_t(r) * T.dim, z, T.dim));
    best[size_t(l)] = std::max(best[size_t(l)], c);
  }
  // 라벨마다 log σ(t·cos + b), 라벨 위로 정규화(log-sum-exp) — 낱말 줄이 없는 라벨(상위어)은 −∞
  double mx = -1e300;
  for (int l = 0; l < T.n_labels; ++l) {
    if (best[size_t(l)] < -1.5f) continue;
    const double x = double(best[size_t(l)]) * T.scale + T.bias;
    const double ls = x >= 0 ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));
    (*out)[size_t(l)] = float(ls);
    mx = std::max(mx, ls);
  }
  double se = 0;
  for (int l = 0; l < T.n_labels; ++l) if ((*out)[size_t(l)] > -1e29f) se += std::exp((*out)[size_t(l)] - mx);
  const double lz = mx + std::log(std::max(se, 1e-300));
  for (int l = 0; l < T.n_labels; ++l) if ((*out)[size_t(l)] > -1e29f) (*out)[size_t(l)] = float((*out)[size_t(l)] - lz);
}

double apAddView(ApState& s, const ApText* T, const ApParams& p, const float* z, int d, double kappa, double stamp, const double cam[6],
                 bool whole) {
  // 비슷한 시점(자리 temper_d 안·광축 temper_deg 안)에서 이미 셌으면 temper 배로
  double k = kappa;
  if (cam) {
    const double cmin = std::cos(p.temper_deg * M_PI / 180.0);
    for (const auto& c : s.cams) {
      if (std::fabs(c[6] - stamp) < 1e-6) continue;   // 같은 영상의 다른 조각
      const double dd = std::hypot(c[0] - cam[0], c[1] - cam[1], c[2] - cam[2]);
      const double ca = c[3] * cam[3] + c[4] * cam[4] + c[5] * cam[5];
      if (dd < p.temper_d && ca > cmin) { k = kappa * p.temper; break; }
    }
    s.cams.push_back({cam[0], cam[1], cam[2], cam[3], cam[4], cam[5], stamp});
    if (s.cams.size() > 32) s.cams.erase(s.cams.begin());
  }
  ++s.ver;
  std::vector<float>& r = whole ? s.r_whole : s.r_frag;
  if (r.size() != size_t(d)) r.assign(size_t(d), 0.f);
  for (int i = 0; i < d; ++i) r[size_t(i)] += float(k * z[i]);
  (whole ? s.k_whole : s.k_frag) += k;
  if (whole) { ++s.n_whole; s.whole_kappa = std::max(s.whole_kappa, kappa); s.need_whole = false; }
  if (T && T->ready() && T->dim == d) {
    std::vector<float> ll;
    apLabelLogLik(*T, z, &ll);
    std::vector<float>& L = whole ? s.L_whole : s.L_frag;
    if (L.size() != ll.size()) L.assign(ll.size(), 0.f);
    const double w = k / std::max(1e-6, p.kappa_ref);
    for (size_t l = 0; l < ll.size(); ++l) L[l] += float(w * std::max(ll[l], -60.f));
    (whole ? s.lw_whole : s.lw_frag) += w;
  }
  // 상위 topk 모습(κ 순). 통째가 생기면 조각 모습은 밀려남(통째 κ 가 보통 큼 — 같으면 통째 먼저)
  ApView v;
  v.z.resize(size_t(d));
  for (int i = 0; i < d; ++i) v.z[size_t(i)] = f2h(z[i]);
  v.kappa = float(kappa);
  v.stamp = stamp;
  v.whole = whole;
  s.views.push_back(std::move(v));
  std::stable_sort(s.views.begin(), s.views.end(), [](const ApView& a, const ApView& b) {
    if (a.whole != b.whole) return a.whole > b.whole;
    return a.kappa > b.kappa;
  });
  if (int(s.views.size()) > p.topk) s.views.resize(size_t(p.topk));
  return k;
}

void apMerge(ApState& a, const ApState& b, const ApParams& p) {
  auto add = [](std::vector<float>& dst, const std::vector<float>& src) {
    if (src.empty()) return;
    if (dst.size() != src.size()) dst.assign(src.size(), 0.f);
    for (size_t i = 0; i < src.size(); ++i) dst[i] += src[i];
  };
  ++a.ver;
  // 통째도 이제는 일부였던 것 → 조각으로
  add(a.r_frag, a.r_whole);
  add(a.r_frag, b.r_frag);
  add(a.r_frag, b.r_whole);
  a.k_frag += a.k_whole + b.k_frag + b.k_whole;
  a.r_whole.clear();
  a.k_whole = 0;
  add(a.L_frag, a.L_whole);
  add(a.L_frag, b.L_frag);
  add(a.L_frag, b.L_whole);
  a.lw_frag += a.lw_whole + b.lw_frag + b.lw_whole;
  a.L_whole.clear();
  a.lw_whole = 0;
  add(a.L_ext, b.L_ext);
  a.n_whole = 0;
  a.whole_kappa = 0;
  a.need_whole = true;
  for (const ApView& v : b.views) a.views.push_back(v);
  for (ApView& v : a.views) v.whole = 0;
  std::stable_sort(a.views.begin(), a.views.end(), [](const ApView& x, const ApView& y) { return x.kappa > y.kappa; });
  if (int(a.views.size()) > p.topk) a.views.resize(size_t(p.topk));
  for (const auto& c : b.cams) a.cams.push_back(c);
  while (a.cams.size() > 32) a.cams.erase(a.cams.begin());
  for (int k = 0; k < 3; ++k) a.P[k] = 1.0 / (1.0 / std::max(a.P[k], 1e-9) + 1.0 / std::max(b.P[k], 1e-9));
  a.cidx_ver = ~0u;
  a.n_wall_obs += b.n_wall_obs;
  a.n_ap_obs += b.n_ap_obs;
  a.wall_like = a.wall_like && b.wall_like;
}

void apName(ApState& s, const ApText& T, const ApParams& p, double size) {
  if (!T.ready()) return;
  // 이름 우도 = 조각 + 통째(통째 무게 whole_w). 통째만 쓰면 모습 1–2 개라 사후가 납작했다(radio r3: 상위 0.1–0.3, 이름 대부분 "object") —
  // 정답 물체마다 조각만 모아도 이름이 31 개 중 26 개 맞음(aprime_fit). 합친 뒤 예전 통째는 조각으로 넘어가 있다(apMerge)
  const int C = T.n_labels;
  thread_local std::vector<float> L;
  L.assign(size_t(C), 0.f);
  double lw = 0;
  if (int(s.L_frag.size()) == C) { for (int c = 0; c < C; ++c) L[size_t(c)] += s.L_frag[size_t(c)]; lw += s.lw_frag; }
  if (int(s.L_whole.size()) == C) { for (int c = 0; c < C; ++c) L[size_t(c)] += float(p.whole_w * s.L_whole[size_t(c)]); lw += p.whole_w * s.lw_whole; }
  if (!(lw > 0)) return;   // 영상 모습이 하나는 있어야(바깥 관측만으로는 이름을 새로 만들지 않음)
  const double lam = lw > p.name_wmax ? p.name_wmax / lw : 1.0;   // 과신 막기(이어진 모습은 독립이 아님)
  std::vector<double> lp(static_cast<size_t>(C));
  const double ls = std::log(std::max(size, 0.02));
  double mx = -1e300;
  for (int c = 0; c < C; ++c) {
    double v = lam * L[size_t(c)];
    if (L[size_t(c)] < -1e29f || L[size_t(c)] <= -60.0 * lw) v = -1e300;   // 낱말 없는 라벨(상위어): 직접 이름 안 됨
    if (size_t(c) < T.log_prior.size()) v += T.log_prior[size_t(c)];
    if (size_t(c) < s.L_ext.size()) v += s.L_ext[size_t(c)];
    if (size_t(c) < s.L_geo.size()) v += s.L_geo[size_t(c)];
    if (size_t(c) < T.ls_sd.size() && T.ls_sd[size_t(c)] > 0) {
      const double zz = (ls - T.ls_mu[size_t(c)]) / T.ls_sd[size_t(c)];
      v += p.size_w * (-0.5 * zz * zz - std::log(T.ls_sd[size_t(c)]));
    }
    lp[size_t(c)] = v;
    mx = std::max(mx, v);
  }
  double se = 0;
  for (double v : lp) if (v > -1e299) se += std::exp(v - mx);
  s.post.assign(size_t(C), 0.f);
  double H = 0;
  int best = -1;
  for (int c = 0; c < C; ++c) {
    if (lp[size_t(c)] <= -1e299) continue;
    const double q = std::exp(lp[size_t(c)] - mx) / se;
    s.post[size_t(c)] = float(q);
    if (q > 1e-12) H -= q * std::log(q);
    if (best < 0 || q > s.post[size_t(best)]) best = c;
  }
  s.name_H = float(H);
  s.top_lab = best;
  s.top_p = best >= 0 ? s.post[size_t(best)] : 0.f;
  s.rolled = false;
  if (best >= 0 && s.top_p >= p.name_tau) { s.name_lab = best; s.name_p = s.top_p; return; }
  // 상위어로: 부모 사슬마다 확률 합, name_tau 를 넘는 가장 낮은(가장 구체적인) 상위어
  std::vector<double> up(size_t(C), 0.0);
  std::vector<int> depth(size_t(C), 0);
  for (int c = 0; c < C; ++c) {
    int a = size_t(c) < T.parent.size() ? T.parent[size_t(c)] : -1;
    for (int g = 0; a >= 0 && a < C && g < 8; ++g, a = size_t(a) < T.parent.size() ? T.parent[size_t(a)] : -1) up[size_t(a)] += s.post[size_t(c)];
  }
  for (int c = 0; c < C; ++c) {   // 상위어 깊이(뿌리에서 먼 것 = 구체적)
    int a = size_t(c) < T.parent.size() ? T.parent[size_t(c)] : -1, dpt = 0;
    while (a >= 0 && a < C && dpt < 8) { ++dpt; a = size_t(a) < T.parent.size() ? T.parent[size_t(a)] : -1; }
    depth[size_t(c)] = dpt;
  }
  int g = -1;
  for (int c = 0; c < C; ++c)
    if (up[size_t(c)] >= p.name_tau && (g < 0 || depth[size_t(c)] > depth[size_t(g)] || (depth[size_t(c)] == depth[size_t(g)] && up[size_t(c)] > up[size_t(g)]))) g = c;
  if (g >= 0) { s.name_lab = g; s.name_p = float(up[size_t(g)]); s.rolled = true; return; }
  if (T.object_label >= 0) { s.name_lab = T.object_label; s.name_p = s.top_p; s.rolled = true; return; }
  s.name_lab = best;
  s.name_p = s.top_p;
}

void apObserveName(ApState& s, int n_labels, int lab, double log_lr) {
  if (lab < 0 || lab >= n_labels) return;
  if (int(s.L_ext.size()) != n_labels) s.L_ext.assign(size_t(n_labels), 0.f);
  s.L_ext[size_t(lab)] += float(log_lr);
  ++s.ver;
}

double apGroupProb(const ApState& s, const std::vector<uint8_t>& mask) {
  double q = 0;
  for (size_t c = 0; c < s.post.size() && c < mask.size(); ++c) if (mask[c]) q += s.post[c];
  return q;
}

void apBuildContact(const float* xyz, int n, double cell, std::vector<uint64_t>* keys) {
  keys->clear();
  keys->reserve(size_t(n));
  const double inv = 1.0 / cell;
  for (int i = 0; i < n; ++i)
    keys->push_back(cellKey3(int64_t(std::floor(xyz[3 * i] * inv)), int64_t(std::floor(xyz[3 * i + 1] * inv)), int64_t(std::floor(xyz[3 * i + 2] * inv))));
  std::sort(keys->begin(), keys->end());
  keys->erase(std::unique(keys->begin(), keys->end()), keys->end());
}

double apContact(const float* xyz, int n, const std::vector<uint64_t>& keys, double cell) {
  if (n <= 0 || keys.empty()) return 0;
  const double inv = 1.0 / cell;
  int hit = 0;
  for (int i = 0; i < n; ++i) {
    const int64_t a = int64_t(std::floor(xyz[3 * i] * inv)), b = int64_t(std::floor(xyz[3 * i + 1] * inv)), c = int64_t(std::floor(xyz[3 * i + 2] * inv));
    bool h = false;
    for (int dx = -1; dx <= 1 && !h; ++dx)
      for (int dy = -1; dy <= 1 && !h; ++dy)
        for (int dz = -1; dz <= 1 && !h; ++dz) h = std::binary_search(keys.begin(), keys.end(), cellKey3(a + dx, b + dy, c + dz));
    hit += h;
  }
  return double(hit) / n;
}

double apLogit(const ApPair& q, const ApParams& p, bool merge) {
  const double* w = merge ? p.wm : p.w;
  double v = w[0];
  for (int k = 0; k < 7; ++k) v += w[k + 1] * q.f[k];
  if (q.f[5] > 0) v = -30;   // 받침(작은 것이 큰 것 윗면 위): 같은 물체가 아님(펜·탁자)
  return v;
}

}  // namespace scenemap
