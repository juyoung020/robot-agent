// sgsearch 물체 색인 + 찾기(include/sgsearch.h). 식과 근거는 헤더 머리말, 측정은 README "물체 찾기".
//
// 자료
//   U(라벨 집합)  : main synset 전부 + 등록·확인·objprob 이름. 라벨마다 표 줄들(점수 = 줄 중 최대 cos) 또는 자유 이름 벡터 하나.
//                   모든 줄을 FP16 행렬 R(nr × 768) 하나로 — 시점 하나 = nr 번 SIMD 내적(sgc_dot_f16).
//   물체          : 시점 벡터 V(nv × 768), μ, 시점별 logZ_v = logsumexp_c t·s_vc, m_c = 평균_v(t·s_vc − logZ_v), N = logsumexp_c m_c
//                   (P_app(c) = exp(m_c − N)), m 상위 8 개(순위·대안 이름), 색·재질 낱말.
//   U 밖 질의 q    : s_vq(질의 벡터 cos, 또는 질의 synset 줄 최대) 하나를 라벨로 더한 분포 —
//                   logZ'_v = logaddexp(logZ_v, t·s_vq), m_q = 평균(t·s_vq − logZ'_v), δ = 평균(logZ_v − logZ'_v),
//                   P(q) = 1 / (1 + exp(N + δ − m_q)), 순위 = #{c : m_c + δ > m_q} + 1.
#if SGS_HAVE_PNG
#include <png.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "labels_impl.hpp"
#include "sgsearch.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr int D = SGC_DIM;
constexpr float kT = 111.83257f;   // SigLIP 2 B/32-256 logit scale(labels.cpp 와 같음)
constexpr int kTop = 8;
constexpr float kSenseDelta = 0.05f;   // 뜻 고르기: 질의 글 cos 가 1 위 뜻에서 이만큼 안

// 색·재질 낱말(속성). 글 = 프롬프트, 이름 = 결과에 쓰는 말
struct Word {
  const char* group;
  const char* name;
  const char* prompt;
};
const Word kWords[] = {
    {"color", "red", "a photo of a red object."},       {"color", "orange", "a photo of an orange object."},
    {"color", "yellow", "a photo of a yellow object."}, {"color", "green", "a photo of a green object."},
    {"color", "blue", "a photo of a blue object."},     {"color", "purple", "a photo of a purple object."},
    {"color", "pink", "a photo of a pink object."},     {"color", "brown", "a photo of a brown object."},
    {"color", "black", "a photo of a black object."},   {"color", "white", "a photo of a white object."},
    {"color", "gray", "a photo of a gray object."},     {"color", "silver", "a photo of a silver object."},
    {"color", "beige", "a photo of a beige object."},
    {"material", "wood", "a photo of an object made of wood."},       {"material", "metal", "a photo of an object made of metal."},
    {"material", "plastic", "a photo of an object made of plastic."}, {"material", "glass", "a photo of an object made of glass."},
    {"material", "fabric", "a photo of an object made of fabric."},   {"material", "leather", "a photo of an object made of leather."},
    {"material", "ceramic", "a photo of an object made of ceramic."}, {"material", "paper", "a photo of an object made of paper."},
    {"material", "stone", "a photo of an object made of stone."},
};
constexpr int kNW = int(sizeof(kWords) / sizeof(kWords[0]));

float lae(float a, float b) {   // log(exp a + exp b)
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  const float m = std::max(a, b);
  return m + std::log1p(std::exp(-std::fabs(a - b)));
}

std::string lower(std::string s) {
  size_t a = s.find_first_not_of(" \t\n"), b = s.find_last_not_of(" \t\n");
  s = a == std::string::npos ? "" : s.substr(a, b - a + 1);
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return s;
}

bool asciiShort(const std::string& q) {   // text_query.py 와 같은 규칙: 영어 3 낱말 이하면 라벨 표 프롬프트를 씌움
  if (q.empty()) return false;
  int words = 1;
  for (char c : q) {
    if (c == ' ') ++words;
    else if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-')) return false;
  }
  return words <= 3;
}

double r3(double v) { return std::round(v * 1000.0) / 1000.0; }

struct Label {
  std::string en, ko;
  int syn = -1;               // 라벨 표 synset 번호(-1 = 자유 이름)
  std::vector<int> chain;     // [자기, 상위어 …] synset 번호
  int r0 = 0, r1 = 0;         // R 행렬의 줄 범위
  bool named = false;         // 보여 줄 이름을 기억이 쓴 말로 바꿨음
};

struct Obj {
  uint32_t id = 0;
  std::string reg, reg_l;     // 등록 이름(view.json name), 소문자
  int reg_c = -1;
  std::vector<std::pair<int, float>> ap_post;   // objprob 이름 사후(라벨, p) — 있으면 바탕
  std::string vsrc = "none";
  int nv = 0;
  std::vector<float> V, mu;   // nv × D, D
  std::vector<float> logZ, m; // nv, |U|
  float N = 0;
  std::vector<std::pair<float, int>> top;   // m 상위(내림차순)
  std::vector<std::string> attrs;
  std::map<int, float> lext;  // 라벨 → Σ log 우도비(확인)
  std::map<int, float> lext_map;   // 그중 지도에도 넣은 것(sm_observe_object_name 성공, 기록 "map":"applied")
  bool ap_external = false;   // objprob 사후에 바깥 관측이 이미 들어 있음(view.json name_post.external)
  std::string rgb, mask;      // best view 사진(상대 경로)
};

struct NameView {               // 이름 사후(보여 줄 것)
  int c = -1;
  float p = 0;
  std::vector<std::pair<int, float>> alt;
};

}  // namespace

struct sgs_index {
  sgs_config cfg{};
  std::string mem, cache;
  const sgc_labels* L = nullptr;
  std::vector<std::vector<int>> syn_rows;   // synset → 표 줄
  std::vector<Label> U;
  std::unordered_map<int, int> lab_of_syn;
  std::unordered_map<std::string, int> lab_of_free;   // 자유 이름(소문자) → 라벨
  std::vector<uint16_t> R;                  // nr × D
  std::vector<int> row_lab;
  std::vector<float> W;                     // 속성 낱말 벡터 kNW × D(없으면 빔)
  std::vector<Obj> obj;
  std::unordered_map<uint32_t, int> by_id;
  json stats;
  int nthreads = 4;
};

namespace {

template <class F>
void parallel(int n, int nt, F f) {
  if (n <= 0) return;
  nt = std::max(1, std::min(nt, n));
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) f(i);
    });
  for (auto& x : th) x.join();
}

// ---- 라벨 ----
void appendRows(sgs_index* X, Label& lab, const std::vector<int>& rows) {
  lab.r0 = int(X->row_lab.size());
  for (int r : rows) {
    X->R.insert(X->R.end(), X->L->text.begin() + ptrdiff_t(size_t(r) * D), X->L->text.begin() + ptrdiff_t(size_t(r + 1) * D));
    X->row_lab.push_back(int(X->U.size()));
  }
  lab.r1 = int(X->row_lab.size());
}

int addSynLabel(sgs_index* X, int syn, int row_hint) {
  auto it = X->lab_of_syn.find(syn);
  if (it != X->lab_of_syn.end()) return it->second;
  Label lab;
  lab.syn = syn;
  // 보여 줄 이름: 그 synset 의 main 줄 중 가장 짧은 영어 이름("radio receiver" 보다 "radio"), main 이 없으면 아무 줄
  int show = row_hint;
  for (int r : X->syn_rows[size_t(syn)]) {
    const auto& a = X->L->rows[size_t(r)];
    const auto& b = X->L->rows[size_t(show)];
    if ((a.main && !b.main) || (a.main == b.main && a.en.size() < b.en.size())) show = r;
  }
  lab.en = X->L->rows[size_t(show)].en;
  lab.ko = X->L->rows[size_t(show)].ko;
  lab.chain = X->L->rows[size_t(show)].chain;
  appendRows(X, lab, X->syn_rows[size_t(syn)]);
  X->lab_of_syn[syn] = int(X->U.size());
  X->U.push_back(std::move(lab));
  return int(X->U.size()) - 1;
}

// 이름 → 라벨(표에 있으면 그 synset, 없으면 자유 이름: 글 인코더가 있으면 그 벡터, 없으면 줄 없음 = 이름으로만)
int labelFor(sgs_index* X, const std::string& name, bool* added) {
  const std::string l = lower(name);
  if (l.empty()) return -1;
  const int row = sgc_labels_find_name(X->L, l.c_str());
  const size_t n0 = X->U.size();
  int c;
  if (row >= 0 && X->L->rows[size_t(row)].syn >= 0) {
    c = addSynLabel(X, X->L->rows[size_t(row)].syn, row);
    if (!X->U[size_t(c)].named) {   // 보여 줄 이름은 기억이 처음 쓴 말("trash can" 등록 → "ashcan" 아니라 "trash can")
      X->U[size_t(c)].en = X->L->rows[size_t(row)].en;
      if (!X->L->rows[size_t(row)].ko.empty()) X->U[size_t(c)].ko = X->L->rows[size_t(row)].ko;
      X->U[size_t(c)].named = true;
    }
  } else if (row >= 0) {   // synset 없는 줄: 그 줄 하나
    auto it = X->lab_of_free.find(l);
    if (it != X->lab_of_free.end()) return it->second;
    Label lab;
    lab.en = X->L->rows[size_t(row)].en;
    lab.ko = X->L->rows[size_t(row)].ko;
    appendRows(X, lab, {row});
    X->lab_of_free[l] = int(X->U.size());
    X->U.push_back(std::move(lab));
    c = int(X->U.size()) - 1;
  } else {
    auto it = X->lab_of_free.find(l);
    if (it != X->lab_of_free.end()) return it->second;
    Label lab;
    lab.en = name;
    lab.r0 = lab.r1 = int(X->row_lab.size());
    if (X->cfg.text) {
      const std::string p = asciiShort(l) ? "a photo of a " + l + "." : l;
      const char* t = p.c_str();
      float v[D];
      if (sgc_text_encode(X->cfg.text, &t, 1, v) == 0) {
        std::vector<uint16_t> h(D);
        sgc_f32_to_f16(v, h.data(), D);
        X->R.insert(X->R.end(), h.begin(), h.end());
        X->row_lab.push_back(int(X->U.size()));
        lab.r1 = lab.r0 + 1;
      }
    }
    X->lab_of_free[l] = int(X->U.size());
    X->U.push_back(std::move(lab));
    c = int(X->U.size()) - 1;
  }
  if (added) *added = *added || X->U.size() != n0;
  return c;
}

// ---- 물체 생김새 분포 ----
void computeApp(const sgs_index* X, Obj& o) {
  const int nu = int(X->U.size()), nr = int(X->row_lab.size());
  o.m.assign(size_t(nu), 0.f);
  o.logZ.assign(size_t(o.nv), 0.f);
  o.top.clear();
  if (o.nv == 0) return;
  std::vector<float> s(static_cast<size_t>(nu));
  for (int v = 0; v < o.nv; ++v) {
    const float* z = &o.V[size_t(v) * D];
    std::fill(s.begin(), s.end(), -INFINITY);
    for (int r = 0; r < nr; ++r) {
      const float x = kT * sgc_dot_f16(&X->R[size_t(r) * D], z, D);
      float& y = s[size_t(X->row_lab[size_t(r)])];
      if (x > y) y = x;
    }
    float mx = -INFINITY;
    for (float x : s) mx = std::max(mx, x);
    double acc = 0;
    for (float x : s)
      if (x != -INFINITY) acc += std::exp(double(x - mx));
    const float lz = mx + float(std::log(acc));
    o.logZ[size_t(v)] = lz;
    for (int c = 0; c < nu; ++c) o.m[size_t(c)] += (s[size_t(c)] == -INFINITY ? -1e30f : s[size_t(c)] - lz) / float(o.nv);
  }
  float mx = -INFINITY;
  for (float x : o.m) mx = std::max(mx, x);
  double acc = 0;
  for (float x : o.m) acc += std::exp(double(x - mx));
  o.N = mx + float(std::log(acc));
  std::vector<std::pair<float, int>> t;
  t.reserve(size_t(nu));
  for (int c = 0; c < nu; ++c) t.emplace_back(o.m[size_t(c)], c);
  const int k = std::min(kTop, nu);
  std::partial_sort(t.begin(), t.begin() + k, t.end(), [](auto& a, auto& b) { return a.first > b.first; });
  o.top.assign(t.begin(), t.begin() + k);
}

void computeAttrs(const sgs_index* X, Obj& o) {
  o.attrs.clear();
  if (X->W.empty() || o.nv == 0) return;
  for (const char* g : {"color", "material"}) {
    float mx = -INFINITY;
    std::vector<std::pair<float, int>> s;
    for (int w = 0; w < kNW; ++w)
      if (!std::strcmp(kWords[w].group, g)) {
        double c = 0;
        for (int d = 0; d < D; ++d) c += double(X->W[size_t(w) * D + size_t(d)]) * o.mu[size_t(d)];
        s.emplace_back(kT * float(c), w);
        mx = std::max(mx, kT * float(c));
      }
    double z = 0;
    for (auto& [v, w] : s) z += std::exp(double(v - mx));
    for (auto& [v, w] : s)
      if (std::exp(double(v - mx)) / z >= X->cfg.attr_min) o.attrs.push_back(kWords[w].name);
  }
}

double pApp(const Obj& o, int c) { return o.nv && c >= 0 && c < int(o.m.size()) ? std::exp(double(o.m[size_t(c)] - o.N)) : 0.0; }

// 이름 사후: 바탕(c)·Λ(c) / Z,  Z = 1 + Σ_{Λ≠1} 바탕(c)(Λ(c) − 1)
struct Post {
  std::map<int, double> base, lam;
  double Z = 1;
  double p(int c) const {
    auto b = base.find(c);
    if (b == base.end()) return 0;
    auto l = lam.find(c);
    return b->second * (l == lam.end() ? 1.0 : l->second) / Z;
  }
};

Post posterior(const sgs_index* X, const Obj& o) {
  Post P;
  if (!o.ap_post.empty()) {
    for (auto& [c, p] : o.ap_post) P.base[c] += p;
  } else if (o.nv) {
    for (auto& [m, c] : o.top) P.base[c] = std::exp(double(m - o.N));
    if (o.reg_c >= 0) P.base[o.reg_c] = pApp(o, o.reg_c);
    for (auto& [c, l] : o.lext) P.base[c] = pApp(o, c);
  } else if (o.reg_c >= 0) {
    P.base[o.reg_c] = 0.5;   // 벡터 없음: 등록 이름 반, 나머지 "모름" 반
  }
  for (auto& [c, l] : o.lext) P.base.emplace(c, 0.0);   // 바탕에 없던 이름도 확인으로 생길 수 있게(아래 최소)
  if (o.ap_post.empty() && o.reg_c >= 0) P.lam[o.reg_c] = X->cfg.reg_lr;
  for (auto& [c, l] : o.lext) P.lam[c] = (P.lam.count(c) ? P.lam[c] : 1.0) * std::exp(double(l));
  for (auto& [c, b] : P.base)
    if (P.lam.count(c) && b < 1e-4) b = 1e-4;   // 확인한 이름이 바탕에서 0 이어도 사후가 움직이게
  for (auto& [c, l] : P.lam) P.Z += P.base[c] * (l - 1.0);
  return P;
}

// 라벨 묶음(예: 질의 synset 과 그 아래말)의 사후 질량 Σ 바탕(c)·Λ(c) / Z. 바탕 = objprob 사후 / P_app / (벡터 없으면) 등록 이름 0.5
double postMass(const Obj& o, const Post& P, const std::vector<int>& cs) {
  double s = 0;
  for (int c : cs) {
    double b = 0;
    if (!o.ap_post.empty() || !o.nv) {
      auto it = P.base.find(c);
      b = it == P.base.end() ? 0 : it->second;
    } else {
      b = pApp(o, c);
      if (P.lam.count(c)) b = std::max(b, 1e-4);
    }
    auto l = P.lam.find(c);
    s += b * (l == P.lam.end() ? 1.0 : l->second);
  }
  return std::min(1.0, s / P.Z);
}

// 라벨 a 와 그 아래말(U 안)
std::vector<int> subtree(const sgs_index* X, int a) {
  std::vector<int> v;
  const int syn = X->U[size_t(a)].syn;
  if (syn < 0) return {a};
  for (int c = 0; c < int(X->U.size()); ++c) {
    const auto& ch = X->U[size_t(c)].chain;
    if (c == a || std::find(ch.begin(), ch.end(), syn) != ch.end()) v.push_back(c);
  }
  return v;
}

std::vector<int> asserted(const Obj& o);

// 보여 줄 이름: 낱 라벨 사후 1 위와, 말해진 이름(등록·확인)의 아래말까지 묶은 질량 중 큰 것
// (등록 "chair" + 생김새 "folding chair" → chair 묶음이 이김 — 서로 맞는 말)
NameView nameView(const sgs_index* X, const Obj& o) {
  const Post P = posterior(X, o);
  std::vector<std::pair<double, int>> v;
  for (auto& [c, b] : P.base) v.emplace_back(P.p(c), c);
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
  NameView n;
  if (v.empty()) return n;
  n.c = v[0].second;
  n.p = float(v[0].first);
  for (int a : asserted(o)) {
    const double m = postMass(o, P, subtree(X, a));
    if (m > n.p) n.c = a, n.p = float(m);
  }
  const std::vector<int> st = subtree(X, n.c);
  for (size_t i = 0; i < v.size() && n.alt.size() < 3; ++i)
    if (v[i].first >= 0.02 && std::find(st.begin(), st.end(), v[i].second) == st.end()) n.alt.emplace_back(v[i].second, float(v[i].first));
  return n;
}

// 말해진 이름: 등록·확인(+)·objprob 사후 0.2 이상
std::vector<int> asserted(const Obj& o) {
  std::vector<int> a;
  if (o.reg_c >= 0) a.push_back(o.reg_c);
  for (auto& [c, l] : o.lext)
    if (l > 0) a.push_back(c);
  for (auto& [c, p] : o.ap_post)
    if (p >= 0.2f) a.push_back(c);
  std::sort(a.begin(), a.end());
  a.erase(std::unique(a.begin(), a.end()), a.end());
  return a;
}

// ---- 파일 ----
bool readPng(const std::string& path, int want_channels, std::vector<uint8_t>* px, int* w, int* h) {
#if !SGS_HAVE_PNG
  (void)path, (void)want_channels, (void)px, (void)w, (void)h;
  return false;   // libpng 없이 빌드: 대체 벡터 뽑기 꺼짐(objprob 벡터·캐시는 그대로)
#else
  png_image im;
  std::memset(&im, 0, sizeof(im));
  im.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&im, path.c_str())) return false;
  im.format = want_channels == 1 ? PNG_FORMAT_GRAY : PNG_FORMAT_RGB;
  px->resize(PNG_IMAGE_SIZE(im));
  if (!png_image_finish_read(&im, nullptr, px->data(), 0, nullptr)) {
    png_image_free(&im);
    return false;
  }
  *w = int(im.width), *h = int(im.height);
  return true;
#endif
}

bool readF16(const std::string& path, std::vector<float>* out, int* n) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const size_t b = size_t(f.tellg());
  if (b == 0 || b % (size_t(D) * 2)) return false;
  std::vector<uint16_t> h(b / 2);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(h.data()), std::streamsize(b));
  *n = int(b / (size_t(D) * 2));
  out->resize(h.size());
  sgc_f16_to_f32(h.data(), out->data(), int32_t(h.size()));
  for (int v = 0; v < *n; ++v) {   // FP16 반올림 뒤 다시 정규화
    float* z = &(*out)[size_t(v) * D];
    double s = 0;
    for (int d = 0; d < D; ++d) s += double(z[d]) * z[d];
    const float inv = s > 0 ? float(1 / std::sqrt(s)) : 0.f;
    for (int d = 0; d < D; ++d) z[d] *= inv;
  }
  return true;
}

bool writeAtomic(const std::string& path, const std::string& data) {
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), std::streamsize(data.size()));
    if (!f) return false;
  }
  fs::rename(tmp, path, ec);
  return !ec;
}

std::string fileKey(const std::string& p) {
  std::error_code ec;
  const auto sz = fs::file_size(p, ec);
  if (ec) return "";
  const auto t = fs::last_write_time(p, ec).time_since_epoch().count();
  return std::to_string(sz) + ":" + std::to_string(t);
}

// 대체 벡터: best view 사진 → sgclip 영상 인코더(마스크 MAP 풀링), cache/objsearch/O<id>_view.f16. 바뀐 것만
int encodeFallback(sgs_index* X, std::vector<Obj*>& need, json& keys) {
  if (!X->cfg.encoder || need.empty()) return 0;
  int done = 0;
  std::vector<sgc_result> res(16);
  auto drain = [&](int wait) {
    const int r = sgc_poll(X->cfg.encoder, res.data(), int(res.size()), wait);
    for (int i = 0; i < r; ++i) {
      auto it = X->by_id.find(res[size_t(i)].id);
      if (it == X->by_id.end()) continue;
      Obj& o = X->obj[size_t(it->second)];
      o.V.assign(res[size_t(i)].emb, res[size_t(i)].emb + D);
      o.nv = 1;
      o.vsrc = "encoded";
      std::vector<uint16_t> h(D);
      sgc_f32_to_f16(o.V.data(), h.data(), D);
      writeAtomic(X->cache + "/O" + std::to_string(o.id) + "_view.f16", std::string(reinterpret_cast<const char*>(h.data()), D * 2));
      keys["O" + std::to_string(o.id)] = fileKey(X->mem + "/" + o.rgb) + "|" + fileKey(X->mem + "/" + o.mask);
      ++done;
    }
    return r;
  };
  for (Obj* o : need) {
    std::vector<uint8_t> rgb, m;
    int w = 0, h = 0, mw = 0, mh = 0;
    if (!readPng(X->mem + "/" + o->rgb, 3, &rgb, &w, &h)) continue;
    std::vector<uint32_t> bits;
    const bool have_mask = !o->mask.empty() && readPng(X->mem + "/" + o->mask, 1, &m, &mw, &mh) && mw == w && mh == h;
    if (have_mask) {
      bits.assign((size_t(w) * h + 31) / 32, 0);
      for (size_t i = 0; i < size_t(w) * h; ++i)
        if (m[i] > 127) bits[i >> 5] |= 1u << (i & 31);
    }
    sgc_frame f{};
    f.rgb = rgb.data();
    f.row_stride = int64_t(w) * 3;
    f.pix_stride = 3;
    f.w = w, f.h = h;
    if (have_mask) {
      f.mask_w = w, f.mask_h = h, f.mask_sx = f.mask_sy = 1.f;
      f.mask_bits = bits.data();
    }
    sgc_item it{o->id, have_mask ? 0 : -1, {0.f, 0.f, float(w), float(h)}, 1.f};
    while (sgc_submit(X->cfg.encoder, 0.0, &f, &it, 1) == 0) drain(1);
    drain(0);
  }
  while (sgc_pending(X->cfg.encoder) > 0)
    if (drain(1) <= 0) break;
  return done;
}

void loadConfirmations(sgs_index* X) {
  std::ifstream f(X->mem + "/confirmations.jsonl");
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    try {
      const json j = json::parse(line);
      auto it = X->by_id.find(j.value("id", 0u));
      if (it == X->by_id.end()) continue;
      bool add = false;
      const int c = labelFor(X, j.value("name", ""), &add);
      if (c < 0) continue;
      const float l = float(std::log(std::max(1.0, j.value("lr", 1.0))));
      X->obj[size_t(it->second)].lext[c] += l;
      if (j.value("map", "") == "applied") X->obj[size_t(it->second)].lext_map[c] += l;
    } catch (...) {
    }
  }
}

json nameJson(const sgs_index* X, const Obj& o) {
  const NameView n = nameView(X, o);
  json a = json::array();
  for (auto& [c, p] : n.alt) a.push_back({X->U[size_t(c)].en, r3(p)});
  json j = {{"id", o.id}, {"name", n.c >= 0 ? X->U[size_t(n.c)].en : o.reg}, {"name_p", r3(n.p)}, {"registered", o.reg}, {"alt", a},
            {"attrs", o.attrs}, {"vec", o.vsrc}, {"nv", o.nv}};
  if (n.c >= 0 && !X->U[size_t(n.c)].ko.empty()) j["name_ko"] = X->U[size_t(n.c)].ko;
  return j;
}

bool saveNames(const sgs_index* X) {
  json j;
  j["table"] = {{"name", sgc_labels_name(X->L)}, {"sha", sgc_labels_sha(X->L)}};
  j["note"] = "derived cache (sgsearch): recomputed from object vectors + view.json names + confirmations.jsonl";
  json o = json::object();
  for (const Obj& x : X->obj) o["O" + std::to_string(x.id)] = nameJson(X, x);
  j["objects"] = o;
  return writeAtomic(X->cache + "/names.json", j.dump(1));
}

void build(sgs_index* X) {
  const auto t0 = std::chrono::steady_clock::now();
  std::ifstream vf(X->mem + "/view.json");
  if (!vf) throw std::runtime_error("no view.json in " + X->mem);
  const json v = json::parse(vf);
  X->obj.clear();
  X->by_id.clear();
  json keys = json::object();
  {
    std::ifstream kf(X->cache + "/views.json");
    if (kf) try { keys = json::parse(kf); } catch (...) {}
  }
  int n_ap = 0, n_mu = 0, n_cache = 0, n_none = 0;
  std::vector<Obj*> need;
  for (const json& jo : v.value("objects", json::array())) {
    Obj o;
    o.id = jo.value("id", 0u);
    o.reg = jo.value("name", "");
    o.reg_l = lower(o.reg);
    if (jo.contains("rgbd")) {
      o.rgb = jo["rgbd"].value("rgb", "");
      o.mask = jo["rgbd"].value("mask", "");
    }
    X->by_id[o.id] = int(X->obj.size());
    X->obj.push_back(std::move(o));
  }
  for (size_t i = 0; i < X->obj.size(); ++i) {
    Obj& o = X->obj[i];
    const std::string b = X->mem + "/objects/O" + std::to_string(o.id);
    int n = 0;
    if (readF16(b + "_views.f16", &o.V, &n)) o.nv = n, o.vsrc = "objprob_views", ++n_ap;
    else if (readF16(b + "_emb.f16", &o.V, &n)) o.V.resize(D), o.nv = 1, o.vsrc = "objprob_mu", ++n_mu;
    else {
      const std::string k = fileKey(X->mem + "/" + o.rgb) + "|" + fileKey(X->mem + "/" + o.mask);
      const std::string key = "O" + std::to_string(o.id);
      if (!o.rgb.empty() && keys.value(key, "") == k && readF16(X->cache + "/" + key + "_view.f16", &o.V, &n)) o.nv = 1, o.vsrc = "cache", ++n_cache;
      else if (!o.rgb.empty()) need.push_back(&o);
      else ++n_none;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  const int n_enc = encodeFallback(X, need, keys);
  if (n_enc) writeAtomic(X->cache + "/views.json", keys.dump(1));
  for (Obj* o : need)
    if (o->nv == 0) ++n_none;
  const auto t2 = std::chrono::steady_clock::now();
  // μ
  for (Obj& o : X->obj) {
    o.mu.assign(D, 0.f);
    for (int v = 0; v < o.nv; ++v)
      for (int d = 0; d < D; ++d) o.mu[size_t(d)] += o.V[size_t(v) * D + size_t(d)];
    double s = 0;
    for (float x : o.mu) s += double(x) * x;
    if (s > 0)
      for (float& x : o.mu) x = float(x / std::sqrt(s));
  }
  // 등록 이름·objprob 사후 → 라벨(U 에 더함)
  for (size_t i = 0; i < X->obj.size(); ++i) {
    Obj& o = X->obj[i];
    bool add = false;
    o.reg_c = labelFor(X, o.reg, &add);
  }
  for (const json& jo : v.value("objects", json::array())) {
    // objprob 형식 {"top": [[이름, p] …], "p", "entropy", "rolled", "external"}(scenemap README "확률 모드" 저장 형식), 옛 형식 [[이름, p] …]
    const json* np = jo.contains("name_post") ? &jo["name_post"] : nullptr;
    if (np && np->is_object()) {
      if (np->value("external", false)) X->obj[size_t(X->by_id[jo.value("id", 0u)])].ap_external = true;
      np = np->contains("top") ? &(*np)["top"] : nullptr;
    }
    if (!np || !np->is_array()) continue;
    Obj& o = X->obj[size_t(X->by_id[jo.value("id", 0u)])];
    for (const json& e : *np)
      if (e.is_array() && e.size() >= 2) {
        bool add = false;
        const int c = labelFor(X, e[0].get<std::string>(), &add);
        if (c >= 0) o.ap_post.emplace_back(c, e[1].get<float>());
      }
  }
  loadConfirmations(X);
  // 지도(objprob)가 이미 받은 확인은 name_post 에 들어 있으므로 두 번 세지 않는다(external 이 아니면 아직 저장 전 — 여기서 셈)
  for (Obj& o : X->obj) {
    if (!o.ap_external || o.ap_post.empty()) continue;
    for (auto& [c, l] : o.lext_map) {
      auto it = o.lext.find(c);
      if (it == o.lext.end()) continue;
      it->second -= l;
      if (std::fabs(it->second) < 1e-6f) o.lext.erase(it);
    }
  }
  const auto t3 = std::chrono::steady_clock::now();
  parallel(int(X->obj.size()), X->nthreads, [&](int i) {
    computeApp(X, X->obj[size_t(i)]);
    computeAttrs(X, X->obj[size_t(i)]);
  });
  const auto t4 = std::chrono::steady_clock::now();
  saveNames(X);
  auto ms = [](auto a, auto b) { return r3(std::chrono::duration<double, std::milli>(b - a).count()); };
  X->stats = {{"objects", X->obj.size()}, {"vec_objprob_views", n_ap}, {"vec_objprob_mu", n_mu}, {"vec_cache", n_cache}, {"vec_encoded", n_enc},
              {"vec_none", n_none}, {"labels", X->U.size()}, {"label_rows", X->row_lab.size()}, {"attr_words", X->W.empty() ? 0 : kNW},
              {"ms_read", ms(t0, t1)}, {"ms_encode", ms(t1, t2)}, {"ms_labels", ms(t2, t3)}, {"ms_appearance", ms(t3, t4)},
              {"ms_total", ms(t0, std::chrono::steady_clock::now())}, {"threads", X->nthreads}, {"simd", sgc_simd()}};
}

// 라벨이 늘면(확인 이름이 U 밖) 모든 물체 다시
void rebuildApp(sgs_index* X) {
  parallel(int(X->obj.size()), X->nthreads, [&](int i) { computeApp(X, X->obj[size_t(i)]); });
}

void loadAttrWords(sgs_index* X) {
  X->W.clear();
  std::string tag;
  for (const Word& w : kWords) tag += std::string(w.prompt) + "\n";
  char h[17];
  uint64_t fh = 1469598103934665603ull;
  for (char c : tag) fh = (fh ^ uint8_t(c)) * 1099511628211ull;
  std::snprintf(h, sizeof(h), "%016llx", static_cast<unsigned long long>(fh));
  const std::string p = X->cache + "/attr_words_" + h + ".f16";
  int n = 0;
  if (readF16(p, &X->W, &n) && n == kNW) return;
  X->W.clear();
  if (!X->cfg.text) return;
  std::vector<const char*> t;
  for (const Word& w : kWords) t.push_back(w.prompt);
  std::vector<float> e(size_t(kNW) * D);
  if (sgc_text_encode(X->cfg.text, t.data(), kNW, e.data()) != 0) return;
  X->W = e;
  std::vector<uint16_t> hh(e.size());
  sgc_f32_to_f16(e.data(), hh.data(), int32_t(e.size()));
  writeAtomic(p, std::string(reinterpret_cast<const char*>(hh.data()), hh.size() * 2));
}

int32_t emit(const std::string& s, char* out, int32_t cap) {
  if (int64_t(s.size()) + 1 > cap || !out) return -int32_t(s.size() + 1);
  std::memcpy(out, s.c_str(), s.size() + 1);
  return int32_t(s.size());
}

// 질의 하나의 물체별 생김새 점수(U 밖 질의: 시점 점수 sv(v) 로)
struct AppQ {
  double p = 0;
  int rank = 99;
  float cos = -1;
};

AppQ appExtra(const Obj& o, const std::vector<float>& sv) {   // sv: 시점마다 cos
  AppQ a;
  if (!o.nv) return a;
  double mq = 0, del = 0;
  for (int v = 0; v < o.nv; ++v) {
    const float s = kT * sv[size_t(v)];
    const float lz2 = lae(o.logZ[size_t(v)], s);
    mq += (s - lz2) / o.nv;
    del += (o.logZ[size_t(v)] - lz2) / o.nv;
    a.cos = std::max(a.cos, sv[size_t(v)]);
  }
  a.p = 1.0 / (1.0 + std::exp(double(o.N) + del - mq));
  a.rank = 1;
  for (auto& [m, c] : o.top)
    if (m + del > mq && c != o.reg_c) ++a.rank;   // 순위는 등록 이름을 빼고 셈: "등록 이름 아니면 질의" 도 1 위(사용자 시나리오)
  return a;
}

AppQ appIn(const Obj& o, const std::vector<int>& Q) {   // Q ⊂ U
  AppQ a;
  if (!o.nv) return a;
  float best = -INFINITY;
  for (int c : Q) a.p += std::exp(double(o.m[size_t(c)] - o.N)), best = std::max(best, o.m[size_t(c)]);
  a.rank = 1;
  for (auto& [m, c] : o.top)
    if (m > best && c != o.reg_c) ++a.rank;
  return a;
}

}  // namespace

extern "C" {

void sgs_default_config(sgs_config* c) {
  if (!c) return;
  *c = sgs_config{};
  c->reg_lr = 3.f;
  c->user_lr = 50.f;
  c->look_lr = 10.f;
  c->weak_name = 0.5f;
  c->name_min = 0.05f;
  c->app_min = 0.1f;    // tools/eval_objsearch.py: BEHAVIOR detcmp A(FastSAM) 기억에서 고름 — README "물체 찾기" 표
  c->app_rank = 1;
  c->exemplar_min = 0.6f;
  c->img_a = 25.f;
  c->img_c0 = 0.85f;   // 시뮬 정답: 같은 종류 시점 cos 중앙 0.71·90 % 0.84, 다른 종류 99 % 0.82 → 0.85 넘으면 반반 이상
  c->img_min = 0.5f;
  c->attr_min = 0.6f;
}

sgs_index* sgs_open(const sgs_config* c, char* err, size_t err_len) {
  auto X = std::make_unique<sgs_index>();
  try {
    if (!c || !c->mem_dir || !c->labels) throw std::runtime_error("sgs_open: mem_dir and labels required");
    X->cfg = *c;
    X->mem = c->mem_dir;
    X->cache = c->cache_dir && *c->cache_dir ? c->cache_dir : X->mem + "/cache/objsearch";
    X->L = c->labels;
    X->nthreads = c->threads > 0 ? c->threads : std::max(1, std::min(8, int(std::thread::hardware_concurrency())));
    X->syn_rows.assign(X->L->syn_names.size(), {});
    for (int r = 0; r < X->L->K; ++r)
      if (X->L->rows[size_t(r)].syn >= 0) X->syn_rows[size_t(X->L->rows[size_t(r)].syn)].push_back(r);
    for (int r = 0; r < X->L->K; ++r) {   // main synset 전부
      const auto& row = X->L->rows[size_t(r)];
      if (row.main && row.syn >= 0) addSynLabel(X.get(), row.syn, r);
    }
    loadAttrWords(X.get());
    build(X.get());
    return X.release();
  } catch (const std::exception& x) {
    if (err && err_len) std::snprintf(err, err_len, "%s", x.what());
    return nullptr;
  }
}

void sgs_close(sgs_index* X) { delete X; }
int32_t sgs_count(const sgs_index* X) { return X ? int32_t(X->obj.size()) : 0; }

int32_t sgs_reload(sgs_index* X, char* err, size_t err_len) {
  if (!X) return -1;
  try {
    build(X);
    return 0;
  } catch (const std::exception& x) {
    if (err && err_len) std::snprintf(err, err_len, "%s", x.what());
    return -1;
  }
}

int32_t sgs_search_json(sgs_index* X, const char* query, int32_t k, int32_t force_app, char* out, int32_t cap) {
  if (!X || !query) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  const sgs_config& C = X->cfg;
  const std::string q = query, ql = lower(q);
  json res;
  res["query"] = q;
  // ---- 질의 풀기 ----
  const int row = sgc_labels_find_name(X->L, ql.c_str());
  std::vector<int> Q;          // U 안의 맞는 라벨(그 synset + 아래말)
  int qsyn = -1;
  std::vector<float> qv;       // U 밖 질의의 벡터(자유 글) — 또는 표 줄
  std::vector<int> qrows;      // U 밖 synset 질의: 그 synset 의 표 줄
  if (row >= 0) {
    // 말의 뜻 전부(같은 글을 이름으로 가진 main 줄들의 synset) + 그 아래말
    const auto& r = X->L->rows[size_t(row)];
    qsyn = r.syn;
    int32_t rr[64];
    const int nr = std::max(0, sgc_labels_find_names(X->L, ql.c_str(), rr, 64));
    std::vector<int> syns;
    json senses = json::array();
    for (int t = 0; t < nr; ++t) {
      const int s = X->L->rows[size_t(rr[t])].syn;
      if (s >= 0 && std::find(syns.begin(), syns.end(), s) == syns.end()) syns.push_back(s), senses.push_back(X->L->syn_names[size_t(s)]);
    }
    if (syns.empty() && qsyn >= 0) syns.push_back(qsyn);
    // 뜻이 여럿이면(표의 한국어 이름은 기계 번역이 섞임: "빗자루" → broom·awning·shredder) 글 인코더로 고름 — 질의 글 벡터와 각 뜻의
    // 표 줄(영어 프롬프트 평균) cos 최대가 1 위에서 sense_delta 안인 뜻만. 글 인코더가 없으면 전부
    if (syns.size() > 1 && C.text) {
      const std::string p = asciiShort(ql) ? "a photo of a " + ql + "." : q;
      const char* t = p.c_str();
      float v[D];
      if (sgc_text_encode(C.text, &t, 1, v) == 0) {
        std::vector<float> sc;
        for (int s : syns) {
          float b = -1;
          for (int rr2 : X->syn_rows[size_t(s)]) b = std::max(b, sgc_dot_f16(&X->L->text[size_t(rr2) * D], v, D));
          sc.push_back(b);
        }
        const float best = *std::max_element(sc.begin(), sc.end());
        std::vector<int> keep;
        senses = json::array();
        for (size_t i = 0; i < syns.size(); ++i)
          if (sc[i] >= best - kSenseDelta) keep.push_back(syns[i]), senses.push_back(X->L->syn_names[size_t(syns[i])]);
        syns = keep;
      }
    }
    res["resolved"] = {{"kind", "label"}, {"label", r.en}, {"ko", r.ko}, {"senses", senses}};
    for (int c = 0; c < int(X->U.size()); ++c) {
      const Label& lab = X->U[size_t(c)];
      bool hit = qsyn < 0 && lab.en == r.en;
      for (int s : syns) {
        // 아래말까지: 뜻이 너무 넓은 말(container·device … labels.cpp generic)이면 그 synset 만
        const bool generic = s < int(X->L->generic.size()) && X->L->generic[size_t(s)];
        hit = hit || (generic ? lab.syn == s : std::find(lab.chain.begin(), lab.chain.end(), s) != lab.chain.end());
      }
      if (hit) Q.push_back(c);
    }
    for (int s : syns)   // U 밖 뜻(tail synset): 그 줄들을 질의 라벨 하나로
      if (!X->lab_of_syn.count(s)) qrows.insert(qrows.end(), X->syn_rows[size_t(s)].begin(), X->syn_rows[size_t(s)].end());
    if (qsyn < 0 && Q.empty()) qrows = {row};
  } else {
    res["resolved"] = {{"kind", "free"}};
    auto it = X->lab_of_free.find(ql);
    if (it != X->lab_of_free.end()) Q.push_back(it->second);   // 이미 라벨(등록·확인된 자유 이름) — 같은 벡터를 두 번 세지 않음
    else if (C.text) {
      const std::string p = asciiShort(ql) ? "a photo of a " + ql + "." : q;
      const char* t = p.c_str();
      qv.resize(D);
      if (sgc_text_encode(C.text, &t, 1, qv.data()) != 0) qv.clear();
    }
    if (qv.empty() && Q.empty()) res["note"] = "no text encoder: only label-table names can be searched";
  }
  const auto t1 = std::chrono::steady_clock::now();
  // ---- ① 이름 ----
  const int n = int(X->obj.size());
  std::vector<double> pn(size_t(n), 0), pa(size_t(n), 0), pi(size_t(n), 0), preg(size_t(n), 0);
  std::vector<int> rk(size_t(n), 99), like(size_t(n), -1);
  std::vector<char> named(size_t(n), 0);   // 말해진 이름이 질의와 맞음(확률이 낮아도 이름 후보 — 기억이 그렇게 부름)
  std::vector<float> cs(size_t(n), -1);
  double best_name = 0;
  for (int i = 0; i < n; ++i) {
    const Obj& o = X->obj[size_t(i)];
    const Post P = posterior(X, o);
    for (int c : asserted(o))
      if (std::find(Q.begin(), Q.end(), c) != Q.end() || (c == o.reg_c && !o.reg_l.empty() && o.reg_l == ql)) named[size_t(i)] = 1;
    // 이름이 맞으면 p_name = 질의 묶음(뜻 + 아래말) 전체의 사후 질량: 등록 "chair" 에 생김새 "folding chair" 면 둘 다 의자
    if (named[size_t(i)]) {
      std::vector<int> cs = Q;
      if (o.reg_c >= 0 && std::find(cs.begin(), cs.end(), o.reg_c) == cs.end() && o.reg_l == ql) cs.push_back(o.reg_c);
      pn[size_t(i)] = postMass(o, P, cs);
    }
    best_name = std::max(best_name, pn[size_t(i)]);
    preg[size_t(i)] = pApp(o, o.reg_c);
  }
  int n_name = 0;
  for (int i = 0; i < n; ++i) n_name += named[size_t(i)];
  const bool step2 = force_app || best_name < C.weak_name;
  // ---- ② 생김새(이름 무시) — 이름 맞은 물체의 질의 확률도 보여 주려고 늘 셈(µs) ----
  std::vector<float> sv;
  for (int i = 0; i < n; ++i) {
    const Obj& o = X->obj[size_t(i)];
    AppQ a;
    if (!Q.empty() && qrows.empty() && qv.empty()) a = appIn(o, Q);
    else if (!qrows.empty() || !qv.empty()) {
      sv.assign(size_t(o.nv), -1.f);
      for (int v = 0; v < o.nv; ++v) {
        const float* z = &o.V[size_t(v) * D];
        if (!qv.empty()) {
          double s = 0;
          for (int d = 0; d < D; ++d) s += double(qv[size_t(d)]) * z[d];
          sv[size_t(v)] = float(s);
        }
        for (int r : qrows) sv[size_t(v)] = std::max(sv[size_t(v)], sgc_dot_f16(&X->L->text[size_t(r) * D], z, D));
      }
      a = appExtra(o, sv);
      if (!Q.empty()) {   // 아래말 라벨이 U 안에도 있으면(예: "chair" 질의 + straight chair) 더함
        const AppQ b = appIn(o, Q);
        a.p = std::min(1.0, a.p + b.p);
        a.rank = std::min(a.rank, b.rank);
      }
    }
    pa[size_t(i)] = a.p;
    rk[size_t(i)] = a.rank;
    cs[size_t(i)] = a.cos;
  }
  // 영상↔영상: 이름으로 확실한 물체가 본보기
  std::vector<int> ex;
  for (int i = 0; i < n; ++i)
    if (pn[size_t(i)] >= C.exemplar_min && X->obj[size_t(i)].nv) ex.push_back(i);
  if (step2 && !ex.empty())
    for (int i = 0; i < n; ++i) {
      const Obj& o = X->obj[size_t(i)];
      float best = -1;
      int who = -1;
      for (int e : ex) {
        if (e == i) continue;
        const Obj& b = X->obj[size_t(e)];
        for (int u = 0; u < o.nv; ++u)
          for (int w = 0; w < b.nv; ++w) {
            double s = 0;
            for (int d = 0; d < D; ++d) s += double(o.V[size_t(u) * D + size_t(d)]) * b.V[size_t(w) * D + size_t(d)];
            if (s > best) best = float(s), who = e;
          }
      }
      if (who >= 0) {
        pi[size_t(i)] = 1.0 / (1.0 + std::exp(-double(C.img_a) * (best - C.img_c0)));
        like[size_t(i)] = who;
      }
    }
  // ---- 합치기 ----
  struct H {
    int i;
    double match;
    const char* type;
  };
  std::vector<H> hits;
  for (int i = 0; i < n; ++i) {
    const bool by_name = named[size_t(i)] && pn[size_t(i)] >= C.name_min;   // 이름은 맞는데 생김새가 거의 확실히 아니면(< name_min) 뺌
    const bool by_app = step2 && pa[size_t(i)] >= C.app_min && rk[size_t(i)] <= C.app_rank;
    const bool by_img = step2 && pi[size_t(i)] >= C.img_min;
    if (!by_name && !by_app && !by_img) continue;
    const double m = 1 - (1 - pn[size_t(i)]) * (1 - (step2 || by_name ? pa[size_t(i)] : 0)) * (1 - pi[size_t(i)]);
    const char* type = by_name ? "name" : "appearance";   // 이름이 맞으면 이름(생김새는 p_query 로 같이 보임)
    hits.push_back({i, m, type});
  }
  std::sort(hits.begin(), hits.end(), [](const H& a, const H& b) { return a.match > b.match; });
  if (k > 0 && int(hits.size()) > k) hits.resize(size_t(k));
  json arr = json::array();
  for (const H& h : hits) {
    const Obj& o = X->obj[size_t(h.i)];
    json j = nameJson(X, o);
    j["match"] = r3(h.match);
    j["match_type"] = h.type;
    j["p_name"] = r3(pn[size_t(h.i)]);
    j["p_query"] = r3(pa[size_t(h.i)]);
    j["q_rank"] = rk[size_t(h.i)];
    j["p_registered"] = r3(preg[size_t(h.i)]);
    if (pi[size_t(h.i)] > 0) {
      j["p_img"] = r3(pi[size_t(h.i)]);
      j["like"] = X->obj[size_t(like[size_t(h.i)])].id;
    }
    arr.push_back(j);
  }
  res["step2"] = step2;
  res["best_name"] = r3(best_name);
  res["n_name_hits"] = n_name;
  res["hits"] = arr;
  res["n_objects"] = n;
  res["us"] = {{"encode", r3(std::chrono::duration<double, std::micro>(t1 - t0).count())},
               {"total", r3(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count())}};
  return emit(res.dump(), out, cap);
}

int32_t sgs_search_vec(const sgs_index* X, const float* q, int32_t k, sgs_slot* out) {
  if (!X || !q || !out || k <= 0) return -1;
  std::vector<sgs_slot> all;
  std::vector<float> sv;
  for (const Obj& o : X->obj) {
    if (!o.nv) continue;
    sv.assign(size_t(o.nv), -1.f);
    for (int v = 0; v < o.nv; ++v) {
      double s = 0;
      for (int d = 0; d < D; ++d) s += double(q[d]) * o.V[size_t(v) * D + size_t(d)];
      sv[size_t(v)] = float(s);
    }
    const AppQ a = appExtra(o, sv);
    all.push_back({o.id, float(a.p), a.cos});
  }
  const int m = std::min<int>(k, int(all.size()));
  std::partial_sort(all.begin(), all.begin() + m, all.end(), [](const sgs_slot& a, const sgs_slot& b) { return a.score > b.score; });
  std::copy(all.begin(), all.begin() + m, out);
  return m;
}

int32_t sgs_object_json(const sgs_index* X, uint32_t id, char* out, int32_t cap) {
  if (!X) return -1;
  auto it = X->by_id.find(id);
  if (it == X->by_id.end()) return emit("{\"status\":\"error\",\"message\":\"unknown id\"}", out, cap);
  return emit(nameJson(X, X->obj[size_t(it->second)]).dump(), out, cap);
}

int32_t sgs_confirm(sgs_index* X, uint32_t id, const char* name, const char* source, const char* query, char* out, int32_t cap) {
  return sgs_confirm_ex(X, id, name, source, query, nullptr, out, cap);
}

int32_t sgs_label_of(const sgs_index* X, const char* name, char* out, int32_t cap) {
  if (!X || !name) return -1;
  const std::string l = lower(name);
  if (l.empty()) return -1;
  const int row = sgc_labels_find_name(X->L, l.c_str());
  return emit(row >= 0 ? X->L->rows[size_t(row)].en : l, out, cap);
}

int32_t sgs_confirm_ex(sgs_index* X, uint32_t id, const char* name, const char* source, const char* query, const char* extra, char* out,
                       int32_t cap) {
  if (!X) return -1;
  auto err = [&](const std::string& m) { return emit(json{{"status", "error"}, {"message", m}}.dump(), out, cap); };
  auto it = X->by_id.find(id);
  if (it == X->by_id.end()) return err("unknown object id O" + std::to_string(id));
  const std::string src = source ? source : "";
  if (src != "user" && src != "close_look") return err("source must be user or close_look");
  if (!name || !*name) return err("name required");
  bool added = false;
  const int c = labelFor(X, name, &added);
  if (c < 0) return err("empty name");
  if (added) rebuildApp(X);
  Obj& o = X->obj[size_t(it->second)];
  const double before = posterior(X, o).p(c), app = pApp(o, c);
  const NameView nb = nameView(X, o);
  const double lr = src == "user" ? X->cfg.user_lr : X->cfg.look_lr;
  json ex = json::object();
  if (extra && *extra) {
    try {
      ex = json::parse(extra);
    } catch (...) {
      return err("extra is not a JSON object");
    }
    if (!ex.is_object()) return err("extra is not a JSON object");
  }
  o.lext[c] += float(std::log(lr));
  if (ex.value("map", "") == "applied") o.lext_map[c] += float(std::log(lr));
  const double after = posterior(X, o).p(c);
  json rec = {{"t", double(std::time(nullptr))}, {"id", o.id},          {"name", name},
                    {"label", X->U[size_t(c)].en},     {"synset", X->U[size_t(c)].syn >= 0 ? X->L->syn_names[size_t(X->U[size_t(c)].syn)] : ""},
                    {"source", src},                   {"lr", lr},             {"query", query ? query : ""},
                    {"registered", o.reg},             {"name_before", nb.c >= 0 ? X->U[size_t(nb.c)].en : ""},
                    {"p_before", r3(before)},          {"p_after", r3(after)}, {"p_app", r3(app)},
                    {"vec", o.vsrc},                   {"nv", o.nv},           {"table", sgc_labels_sha(X->L)}};
  for (auto& [k, val] : ex.items())
    if (!rec.contains(k)) rec[k] = val;
  const std::string log = X->mem + "/confirmations.jsonl";
  {
    std::ofstream f(log, std::ios::app);
    if (!f) return err("cannot append " + log);
    f << rec.dump() << "\n";
  }
  saveNames(X);
  return emit(json{{"status", "ok"}, {"id", o.id}, {"name", X->U[size_t(c)].en}, {"p_before", r3(before)}, {"p_after", r3(after)},
                   {"registered", o.reg}, {"logged", log}}
                  .dump(),
              out, cap);
}

int32_t sgs_stats_json(const sgs_index* X, char* out, int32_t cap) {
  if (!X) return -1;
  return emit(X->stats.dump(), out, cap);
}

}  // extern "C"
