// sgclip 라벨 표 + 찾기(include/sgclip.h). docs/clip_candidates.md 3.3.
//
//   표   : manifest.json · table.jsonl · text_siglip2_b32.f16(K × 768 FP16, L2) — training/embed/README.md "라벨 표 형식".
//   색인 : 라벨 글 임베딩을 평균 빼고 PCA 128-d(부분 공간 반복, 표본 12k 줄) → k-means 256 묶음(IVF) + 128-bit 부호.
//          질의 q(768)는 평균을 빼지 않고 같은 P 로 투영한다(q·(l − μ) 순위 = q·l 순위, 상수 q·μ 차이뿐).
//          표 sha 별 파일(labels_<sha>.idx)로 캐시. 만들기는 PC 단일 스레드 1–3 s(한 번).
//   찾기 : 묶음 중심 점수 상위 nprobe → 그 묶음 줄들 해밍 거리 → 상위 rerank 개만 768-d FP16 내적 → 점수 순 k 개.
//   SIMD : x86 AVX2+F16C(_mm256_cvtph_ps), aarch64 NEON(vcvt_f32_f16 — Nano A57 도 됨), 아니면 일반 C++.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define SGC_AVX2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SGC_NEON 1
#endif

#include "labels_impl.hpp"

using sgclip_detail::Row;

namespace {

constexpr int D = SGC_DIM, PD = 128, NC = 256, kMagic = 0x32584449;   // "IDX2"
constexpr float kLogitScale = 111.83257f, kLogitBias = -16.766876f;  // SigLIP 2 B/32-256 (open_clip webli)

float h2f(uint16_t h) {
  const uint32_t s = uint32_t(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, f;
  if (e == 0) {
    if (m == 0) f = s;
    else {
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3ffu;
      f = s | (e << 23) | (m << 13);
    }
  } else if (e == 31) f = s | 0x7f800000u | (m << 13);
  else f = s | ((e + 127 - 15) << 23) | (m << 13);
  float r;
  std::memcpy(&r, &f, 4);
  return r;
}

// ---- SIMD 내적 ----
inline float dotF16(const uint16_t* a, const float* b, int n) {
#if SGC_AVX2
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i))), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 8))), _mm256_loadu_ps(b + i + 8), s1);
  }
  s0 = _mm256_add_ps(s0, s1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  float r = _mm_cvtss_f32(h);
  for (; i < n; ++i) r += h2f(a[i]) * b[i];
  return r;
#elif SGC_NEON
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(a + i));
    s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(h)), vld1q_f32(b + i + 4));
  }
  float r = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < n; ++i) r += h2f(a[i]) * b[i];
  return r;
#else
  float r = 0;
  for (int i = 0; i < n; ++i) r += h2f(a[i]) * b[i];
  return r;
#endif
}

inline float dotF32(const float* a, const float* b, int n) {
#if SGC_AVX2
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  s0 = _mm256_add_ps(s0, s1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  float r = _mm_cvtss_f32(h);
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
#elif SGC_NEON
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float r = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
#else
  float r = 0;
  for (int i = 0; i < n; ++i) r += a[i] * b[i];
  return r;
#endif
}

inline int popc(uint64_t x) { return __builtin_popcountll(x); }   // x86 popcnt / NEON vcnt

std::string lower(std::string s) {
  size_t a = s.find_first_not_of(" \t\n"), b = s.find_last_not_of(" \t\n");
  s = a == std::string::npos ? "" : s.substr(a, b - a + 1);
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return s;
}



}  // namespace



namespace {

void project(const sgc_labels* L, const float* q, float* qp) {
  for (int k = 0; k < PD; ++k) qp[k] = dotF32(L->P.data() + size_t(k) * D, q, D);
}

void buildIndex(sgc_labels* L, const std::vector<float>& img) {
  const int K = L->K;
  std::vector<float> X(size_t(K) * D);
  for (size_t i = 0; i < X.size(); ++i) X[i] = h2f(L->text[i]);
  L->mu.assign(D, 0.f);
  for (int i = 0; i < K; ++i)
    for (int d = 0; d < D; ++d) L->mu[d] += X[size_t(i) * D + d];
  for (float& m : L->mu) m /= float(K);
  for (int i = 0; i < K; ++i)
    for (int d = 0; d < D; ++d) X[size_t(i) * D + d] -= L->mu[d];
  // 공분산(표본 ≤ 12k 줄, 위 삼각). 영상 표본이 있으면 그 분포(질의 쪽)로 맞춤 — 글 PCA 보다 1단계 순위가 훨씬 낫다
  // (평가 crop 567 개, 128-d 점수 상위 32 → 768-d: 글 PCA 0.55, 영상 PCA 0.77 = 전부 훑기 1위와 일치)
  const int NI = int(img.size() / D);
  L->qc.assign(D, 0.f);
  std::vector<float> Xi;
  if (NI > 0) {
    for (int i = 0; i < NI; ++i)
      for (int d = 0; d < D; ++d) L->qc[d] += img[size_t(i) * D + d] / float(NI);
    Xi.resize(img.size());
    for (int i = 0; i < NI; ++i)
      for (int d = 0; d < D; ++d) Xi[size_t(i) * D + d] = img[size_t(i) * D + d] - L->qc[d];
  }
  const std::vector<float>& S = NI > 0 ? Xi : X;
  const int NS = NI > 0 ? NI : K;
  const int stride = std::max(1, NS / 12000);
  std::vector<float> C(size_t(D) * D, 0.f);
  int ns = 0;
  for (int i = 0; i < NS; i += stride, ++ns) {
    const float* x = &S[size_t(i) * D];
    for (int a = 0; a < D; ++a) {
      const float xa = x[a];
      float* c = &C[size_t(a) * D];
      for (int b = a; b < D; ++b) c[b] += xa * x[b];
    }
  }
  for (int a = 0; a < D; ++a)
    for (int b = a; b < D; ++b) C[size_t(b) * D + a] = C[size_t(a) * D + b] /= float(ns);
  // 부분 공간 반복: Q(D × PD) 열 정규 직교, 25 번
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd;
  std::vector<float> Q(size_t(PD) * D), Z(size_t(PD) * D);   // 행 = 기저 벡터
  for (float& v : Q) v = nd(rng);
  auto orth = [&](std::vector<float>& M) {
    for (int k = 0; k < PD; ++k) {
      float* v = &M[size_t(k) * D];
      for (int j = 0; j < k; ++j) {
        const float* u = &M[size_t(j) * D];
        const float p = dotF32(u, v, D);
        for (int d = 0; d < D; ++d) v[d] -= p * u[d];
      }
      const float n = std::sqrt(std::max(dotF32(v, v, D), 1e-20f));
      for (int d = 0; d < D; ++d) v[d] /= n;
    }
  };
  orth(Q);
  for (int it = 0; it < 25; ++it) {
    for (int k = 0; k < PD; ++k)
      for (int a = 0; a < D; ++a) Z[size_t(k) * D + a] = dotF32(&C[size_t(a) * D], &Q[size_t(k) * D], D);
    orth(Z);
    Q.swap(Z);
  }
  L->P = Q;
  // 투영
  std::vector<float> Y(size_t(K) * PD);
  for (int i = 0; i < K; ++i)
    for (int k = 0; k < PD; ++k) Y[size_t(i) * PD + k] = dotF32(&L->P[size_t(k) * D], &X[size_t(i) * D], D);
  // k-means(점곱 − ½|c|²), 12 번
  std::vector<float> Cn(size_t(NC) * PD), cn(NC);
  for (int c = 0; c < NC; ++c) std::memcpy(&Cn[size_t(c) * PD], &Y[size_t((size_t(c) * K) / NC) * PD], PD * 4);
  std::vector<int> as(K, 0);
  for (int it = 0; it < 12; ++it) {
    for (int c = 0; c < NC; ++c) cn[c] = 0.5f * dotF32(&Cn[size_t(c) * PD], &Cn[size_t(c) * PD], PD);
    for (int i = 0; i < K; ++i) {
      float best = -1e30f;
      for (int c = 0; c < NC; ++c) {
        const float s = dotF32(&Y[size_t(i) * PD], &Cn[size_t(c) * PD], PD) - cn[c];
        if (s > best) best = s, as[i] = c;
      }
    }
    std::vector<float> sum(size_t(NC) * PD, 0.f);
    std::vector<int> cnt(NC, 0);
    for (int i = 0; i < K; ++i) {
      ++cnt[as[i]];
      for (int k = 0; k < PD; ++k) sum[size_t(as[i]) * PD + k] += Y[size_t(i) * PD + k];
    }
    for (int c = 0; c < NC; ++c)
      if (cnt[c])
        for (int k = 0; k < PD; ++k) Cn[size_t(c) * PD + k] = sum[size_t(c) * PD + k] / float(cnt[c]);
  }
  L->cent = Cn;
  L->off.assign(NC + 1, 0);
  for (int i = 0; i < K; ++i) ++L->off[as[i] + 1];
  for (int c = 0; c < NC; ++c) L->off[c + 1] += L->off[c];
  L->ids.assign(K, 0);
  std::vector<int> pos(L->off.begin(), L->off.end() - 1);
  for (int i = 0; i < K; ++i) L->ids[pos[as[i]]++] = i;
  L->codes.assign(size_t(K) * 2, 0);
  L->lp.assign(size_t(K) * PD, 0);
  for (int j = 0; j < K; ++j) {
    const float* y = &Y[size_t(L->ids[j]) * PD];
    sgc_f32_to_f16(y, &L->lp[size_t(j) * PD], PD);
    for (int k = 0; k < PD; ++k)
      if (y[k] > 0) L->codes[size_t(j) * 2 + (k >> 6)] |= 1ull << (k & 63);
  }
}

template <class T>
void wr(std::ofstream& f, const std::vector<T>& v) {
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}
template <class T>
bool rd(std::ifstream& f, std::vector<T>& v, size_t n) {
  v.resize(n);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
  return bool(f);
}

bool loadIndex(sgc_labels* L, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  int32_t h[5];
  char sha[16];
  f.read(reinterpret_cast<char*>(h), sizeof(h));
  f.read(sha, 16);
  if (!f || h[0] != kMagic || h[1] != L->K || h[2] != D || h[3] != PD || h[4] != NC || std::string(sha, 16) != L->sha) return false;
  return rd(f, L->mu, D) && rd(f, L->qc, D) && rd(f, L->P, size_t(PD) * D) && rd(f, L->cent, size_t(NC) * PD) && rd(f, L->off, NC + 1) &&
         rd(f, L->ids, L->K) && rd(f, L->codes, size_t(L->K) * 2) && rd(f, L->lp, size_t(L->K) * PD);
}

void saveIndex(const sgc_labels* L, const std::string& path) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return;
    const int32_t h[5] = {kMagic, L->K, D, PD, NC};
    f.write(reinterpret_cast<const char*>(h), sizeof(h));
    char sha[16] = {0};
    std::memcpy(sha, L->sha.data(), std::min<size_t>(16, L->sha.size()));
    f.write(sha, 16);
    wr(f, L->mu); wr(f, L->qc); wr(f, L->P); wr(f, L->cent); wr(f, L->off); wr(f, L->ids); wr(f, L->codes); wr(f, L->lp);
  }
  std::rename(tmp.c_str(), path.c_str());
}

int internSyn(sgc_labels* L, const std::string& s) {
  auto it = L->syn_id.find(s);
  if (it != L->syn_id.end()) return it->second;
  const int id = int(L->syn_names.size());
  L->syn_names.push_back(s);
  L->syn_id.emplace(s, id);
  return id;
}

struct Cand {
  float s;
  int row;
};

// 공용 찾기: 점수 순 상위 k(out 은 k 칸)
int lookupImpl(const sgc_labels* L, const float* q, int k, Cand* out, const sgc_lookup_params& p) {
  thread_local std::vector<Cand> buf;
  thread_local std::vector<std::pair<int, int>> ham;   // (거리, 목록 자리)
  buf.clear();
  if (p.exact || L->cent.empty()) {
    for (int i = 0; i < L->K; ++i) {
      if (p.main_only && !L->rows[i].main) continue;
      buf.push_back({dotF16(&L->text[size_t(i) * D], q, D), i});
    }
  } else {
    float qp[PD], qq[D];
    for (int d = 0; d < D; ++d) qq[d] = q[d] - L->qc[d];
    project(L, qq, qp);
    uint64_t qc[2] = {0, 0};
    for (int j = 0; j < PD; ++j)
      if (qp[j] > 0) qc[j >> 6] |= 1ull << (j & 63);
    float cs[NC];
    int ci[NC];
    for (int c = 0; c < NC; ++c) cs[c] = dotF32(&L->cent[size_t(c) * PD], qp, PD), ci[c] = c;
    const int np = std::min(std::max(1, p.nprobe), NC);
    std::sort(ci, ci + NC, [&](int a, int b) { return cs[a] > cs[b]; });
    ham.clear();
    const size_t want = size_t(std::max(k, p.rerank));
    for (int t = 0; t < NC && (t < np || ham.size() < want); ++t)   // 묶음 nprobe 개, 후보가 모자라면 더
      for (int j = L->off[ci[t]]; j < L->off[ci[t] + 1]; ++j) {
        if (p.main_only && !L->is_main[j]) continue;
        ham.emplace_back(p.prefilter > 0 ? popc(L->codes[size_t(j) * 2] ^ qc[0]) + popc(L->codes[size_t(j) * 2 + 1] ^ qc[1]) : 0, j);
      }
    // (선택) 128-bit 해밍으로 prefilter 개까지 거름 → 128-d FP16 점수 상위 rerank → 768-d
    if (p.prefilter > 0 && int(ham.size()) > p.prefilter) {
      std::nth_element(ham.begin(), ham.begin() + p.prefilter, ham.end());
      ham.resize(size_t(p.prefilter));
    }
    thread_local std::vector<Cand> c128;
    c128.clear();
    for (const auto& h : ham) c128.push_back({dotF16(&L->lp[size_t(h.second) * PD], qp, PD), h.second});
    const int R = std::min<int>(int(want), int(c128.size()));
    std::nth_element(c128.begin(), c128.begin() + R, c128.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    for (int t = 0; t < R; ++t) {
      const int row = L->ids[c128[t].row];
      buf.push_back({dotF16(&L->text[size_t(row) * D], q, D), row});
    }
  }
  const int n = std::min<int>(k, int(buf.size()));
  std::partial_sort(buf.begin(), buf.begin() + n, buf.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
  std::copy(buf.begin(), buf.begin() + n, out);
  return n;
}

sgc_hit hitOf(const sgc_labels* L, const Cand& c) {
  const Row& r = L->rows[c.row];
  return sgc_hit{c.row, c.s, r.en.c_str(), r.ko.c_str(), r.structural ? 1 : 0};
}

float rollDelta() {
  static const float d = [] {
    const char* e = std::getenv("SGC_ROLLUP_DELTA");
    return e ? float(std::atof(e)) : 0.004f;
  }();
  return d;
}

}  // namespace

extern "C" {

void sgc_default_lookup(sgc_lookup_params* p) {
  if (!p) return;
  *p = sgc_lookup_params{8, 32, 0, 0, 0};
}

sgc_labels* sgc_labels_open(const char* dir, const char* index_dir, char* err, size_t err_len) {
  const char* s = std::getenv("SGC_IMG_SAMPLE");
  return sgc_labels_open_ex(dir, index_dir, s, err, err_len);
}

sgc_labels* sgc_labels_open_ex(const char* dir, const char* index_dir, const char* img_sample, char* err, size_t err_len) {
  auto L = std::make_unique<sgc_labels>();
  try {
    if (!dir) throw std::runtime_error("label dir required");
    L->dir = dir;
    std::ifstream mf(L->dir + "/manifest.json");
    if (!mf) throw std::runtime_error("no manifest.json in " + L->dir);
    const nlohmann::json man = nlohmann::json::parse(mf);
    L->sha = man.value("sha", "");
    L->name = man.value("name", "labels") + "-" + man.value("version", "");
    const std::string model = man.value("model", "siglip2_b32");
    if (man.value("dim", D) != D) throw std::runtime_error("label table dim != 768");
    std::ifstream tf(L->dir + "/table.jsonl");
    if (!tf) throw std::runtime_error("no table.jsonl");
    std::string line;
    std::vector<std::vector<std::string>> hyp;
    while (std::getline(tf, line)) {
      if (line.empty()) continue;
      const nlohmann::json j = nlohmann::json::parse(line);
      Row r;
      r.en = j.value("en", "");
      if (j.contains("ko") && j["ko"].is_array() && !j["ko"].empty()) {
        r.ko = j["ko"][0].get<std::string>();
        for (const auto& x : j["ko"]) r.ko_all.push_back(x.get<std::string>());
      }
      if (j.contains("en_syn") && j["en_syn"].is_array())
        for (const auto& x : j["en_syn"]) r.en_syn.push_back(x.get<std::string>());
      r.synset = j.value("synset", "");
      r.main = j.value("tier", "tail") == "main";
      r.structural = j.value("structural", false);
      std::vector<std::string> h;
      if (j.contains("hypernyms"))
        for (const auto& x : j["hypernyms"]) h.push_back(x.get<std::string>());
      hyp.push_back(std::move(h));
      L->rows.push_back(std::move(r));
    }
    L->K = int(L->rows.size());
    if (!L->K) throw std::runtime_error("empty table.jsonl");
    for (int i = 0; i < L->K; ++i) {
      Row& r = L->rows[i];
      if (!r.synset.empty()) {
        r.syn = internSyn(L.get(), r.synset);
        r.chain.push_back(r.syn);
      }
      for (const std::string& h : hyp[i]) r.chain.push_back(internSyn(L.get(), h));
    }
    L->generic.assign(L->syn_names.size(), 0);
    for (const char* g : {"artifact.n.01", "instrumentality.n.03", "whole.n.02", "object.n.01", "physical_entity.n.01", "matter.n.03",
                          "commodity.n.01", "consumer_goods.n.01", "durables.n.01", "structure.n.01", "covering.n.02", "device.n.01",
                          "container.n.01", "equipment.n.01", "implement.n.01", "material.n.01", "substance.n.07", "solid.n.01",
                          "natural_object.n.01", "organism.n.01", "living_thing.n.01", "part.n.02", "creation.n.02", "unit.n.05"}) {
      auto it = L->syn_id.find(g);
      if (it != L->syn_id.end()) L->generic[size_t(it->second)] = 1;
    }
    L->syn_row.assign(L->syn_names.size(), -1);
    for (int pass = 0; pass < 2; ++pass)
      for (int i = 0; i < L->K; ++i) {
        const Row& r = L->rows[i];
        if (r.syn >= 0 && L->syn_row[r.syn] < 0 && (pass == 1 || r.main)) L->syn_row[r.syn] = i;
      }
    for (int i = L->K - 1; i >= 0; --i) {   // 앞 줄이 이김
      L->by_text[lower(L->rows[i].en)] = i;
      if (!L->rows[i].ko.empty()) L->by_text[lower(L->rows[i].ko)] = i;
    }
    for (int i = L->K - 1; i >= 0; --i)
      if (L->rows[i].main) L->by_text[lower(L->rows[i].en)] = i;   // 같은 글이면 main 줄
    // 이름 검색용(sgc_labels_find_name, 물체 찾기 objindex): main 줄의 영어·한국어 전부·동의어가 먼저, 그다음 tail 줄.
    // ("라디오" 는 tail 줄 radiocommunication 의 첫 한국어 이름이기도 해서 by_text 로는 그 줄이 나옴 — 물건 이름은 main 이 맞다)
    // 순서: main 영어 이름 → main 한국어·동의어 → tail 영어 → tail 나머지(앞이 이김: "radio" 는 en "radio" 줄, en_syn 에 radio 가 있는 줄이 아님)
    for (int pass = 0; pass < 4; ++pass)
      for (int i = 0; i < L->K; ++i) {
        const Row& r = L->rows[i];
        if (r.main != (pass < 2)) continue;
        if (pass % 2 == 0) {
          L->by_name.emplace(lower(r.en), i);
          continue;
        }
        for (const std::string& t : r.ko_all) L->by_name.emplace(lower(t), i);
        for (const std::string& t : r.en_syn) L->by_name.emplace(lower(t), i);
      }
    // 뜻 전부(sgc_labels_find_names): 같은 글의 main 줄 전부("의자" → chair·armchair, "전등" → light bulb·lamp …)
    for (int i = 0; i < L->K; ++i) {
      const Row& r = L->rows[i];
      auto& m = r.main ? L->names_main : L->names_tail;
      std::vector<std::string> ts{lower(r.en)};
      for (const std::string& t : r.ko_all) ts.push_back(lower(t));
      for (const std::string& t : r.en_syn) ts.push_back(lower(t));
      std::sort(ts.begin(), ts.end());
      ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
      for (const std::string& t : ts) m[t].push_back(i);
    }
    std::ifstream ef(L->dir + "/text_" + model + ".f16", std::ios::binary | std::ios::ate);
    if (!ef) throw std::runtime_error("no text_" + model + ".f16");
    const size_t bytes = size_t(ef.tellg());
    if (bytes != size_t(L->K) * D * 2) throw std::runtime_error("text embedding size != count x 768 x 2");
    ef.seekg(0);
    L->text.resize(size_t(L->K) * D);
    ef.read(reinterpret_cast<char*>(L->text.data()), std::streamsize(bytes));
    // 색인: 캐시 → 없으면 만들고 씀
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> img;
    std::string tag = "t";
    if (img_sample && *img_sample) {   // N × 768 FP16 영상 임베딩 표본(질의 분포)
      std::ifstream sf(img_sample, std::ios::binary | std::ios::ate);
      if (sf) {
        const size_t nb = size_t(sf.tellg());
        std::vector<uint16_t> h(nb / 2);
        sf.seekg(0);
        sf.read(reinterpret_cast<char*>(h.data()), std::streamsize(nb));
        const size_t n = h.size() / D;
        img.resize(n * D);
        sgc_f16_to_f32(h.data(), img.data(), int32_t(n * D));
        uint64_t fh = 1469598103934665603ull;
        for (uint16_t v : h) fh = (fh ^ v) * 1099511628211ull;
        char b[9];
        std::snprintf(b, sizeof(b), "%08x", unsigned(fh >> 32));
        tag = std::string("i") + b;
        L->proj = std::string("img:") + b;
      }
    }
    std::string ip;
    if (index_dir && *index_dir) {
      std::error_code ec;
      std::filesystem::create_directories(index_dir, ec);
      ip = std::string(index_dir) + "/labels_" + L->sha + "_" + tag + ".idx";
    }
    if (!ip.empty() && loadIndex(L.get(), ip)) {
      L->from_cache = true;
    } else {
      buildIndex(L.get(), img);
      if (!ip.empty()) saveIndex(L.get(), ip);
    }
    L->is_main.resize(L->K);
    for (int j = 0; j < L->K; ++j) L->is_main[j] = L->rows[L->ids[j]].main;
    L->build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[sgclip] labels %s (%s): %d rows, index %s %.0f ms (projection %s), simd %s\n", L->name.c_str(), L->sha.c_str(),
                 L->K, L->from_cache ? "cache" : "built", L->build_ms, L->proj.c_str(), sgc_simd());
    return L.release();
  } catch (const std::exception& x) {
    if (err && err_len) std::snprintf(err, err_len, "%s", x.what());
    return nullptr;
  }
}

void sgc_labels_close(sgc_labels* L) { delete L; }
const char* sgc_labels_sha(const sgc_labels* L) { return L ? L->sha.c_str() : ""; }
const char* sgc_labels_name(const sgc_labels* L) { return L ? L->name.c_str() : ""; }
int32_t sgc_labels_count(const sgc_labels* L) { return L ? L->K : 0; }

int32_t sgc_labels_lookup(const sgc_labels* L, const float* q, int32_t k, sgc_hit* out, const sgc_lookup_params* pp) {
  if (!L || !q || !out || k <= 0) return -1;
  sgc_lookup_params p;
  sgc_default_lookup(&p);
  if (pp) p = *pp;
  thread_local std::vector<Cand> c;
  c.resize(size_t(k));
  const int n = lookupImpl(L, q, k, c.data(), p);
  for (int i = 0; i < n; ++i) out[i] = hitOf(L, c[i]);
  return n;
}

int32_t sgc_labels_names(const sgc_labels* L, const float* q, sgc_names* o, const sgc_lookup_params* pp) {
  if (!L || !q || !o) return -1;
  sgc_lookup_params p;
  sgc_default_lookup(&p);
  if (pp) p = *pp;
  Cand c[16];
  const int n = lookupImpl(L, q, 16, c, p);
  *o = sgc_names{};
  if (n <= 0) return -1;
  o->n = std::min(n, 5);
  for (int i = 0; i < o->n; ++i) o->top[i] = hitOf(L, c[i]);
  const Row& r0 = L->rows[c[0].row];
  o->level_en = r0.en.c_str();
  o->level_ko = r0.ko.c_str();
  o->level_score = c[0].s;
  o->prob = 1.f / (1.f + std::exp(-(c[0].s * kLogitScale + kLogitBias)));
  o->structural = r0.structural ? 1 : 0;
  auto same = [&](int a, int b) { return a == b || (L->rows[a].syn >= 0 && L->rows[a].syn == L->rows[b].syn); };
  o->margin = c[0].s;
  for (int i = 1; i < n; ++i)
    if (!same(c[0].row, c[i].row)) { o->margin = c[0].s - c[i].s; break; }
  // 확신이 낮으면(1위와 delta 안의 다른 synset 들) 공통 상위어로
  const float delta = rollDelta();
  std::vector<int> close;
  for (int i = 0; i < n && c[i].s >= c[0].s - delta; ++i) {
    bool dup = false;
    for (int j : close) dup = dup || same(j, c[i].row);
    if (!dup) close.push_back(c[i].row);
  }
  if (close.size() < 2) return 0;
  const std::vector<int>& ch0 = L->rows[close[0]].chain;
  for (size_t a = 0; a < ch0.size(); ++a) {
    bool all = true;
    for (size_t t = 1; t < close.size() && all; ++t) {
      const auto& ch = L->rows[close[t]].chain;
      all = std::find(ch.begin(), ch.end(), ch0[a]) != ch.end();
    }
    if (!all) continue;
    if (ch0.size() - a < 4 || L->generic[size_t(ch0[a])]) break;   // 뿌리 가까운 말(artifact·object·whole …)은 이름이 아님
    const int syn = ch0[a];
    const int row = syn < int(L->syn_row.size()) ? L->syn_row[syn] : -1;
    if (row >= 0) {
      o->level_en = L->rows[row].en.c_str();
      o->level_ko = L->rows[row].ko.c_str();
    } else {   // 표에 줄이 없는 상위어: synset 이름의 머리말("seat.n.03" → "seat")
      static thread_local std::string tmp;
      tmp = L->syn_names[syn].substr(0, L->syn_names[syn].find('.'));
      for (char& ch : tmp)
        if (ch == '_') ch = ' ';
      o->level_en = tmp.c_str();
      o->level_ko = "";
    }
    o->rolled = 1;
    break;
  }
  return 0;
}

int32_t sgc_labels_find(const sgc_labels* L, const char* text) {
  if (!L || !text) return -1;
  auto it = L->by_text.find(lower(text));
  return it == L->by_text.end() ? -1 : it->second;
}

int32_t sgc_labels_find_name(const sgc_labels* L, const char* text) {
  if (!L || !text) return -1;
  auto it = L->by_name.find(lower(text));
  return it == L->by_name.end() ? -1 : it->second;
}

int32_t sgc_labels_find_names(const sgc_labels* L, const char* text, int32_t* rows, int32_t cap) {
  if (!L || !text || !rows || cap <= 0) return -1;
  const std::string t = lower(text);
  auto it = L->names_main.find(t);
  if (it == L->names_main.end()) {
    it = L->names_tail.find(t);
    if (it == L->names_tail.end()) return 0;
  }
  const int n = std::min<int>(cap, int(it->second.size()));
  for (int i = 0; i < n; ++i) rows[i] = it->second[size_t(i)];
  return n;
}

int32_t sgc_labels_text_emb(const sgc_labels* L, int32_t row, float* out) {
  if (!L || row < 0 || row >= L->K || !out) return -1;
  for (int d = 0; d < D; ++d) out[d] = h2f(L->text[size_t(row) * D + d]);
  return 0;
}

float sgc_dot_f16(const uint16_t* a, const float* b, int32_t n) { return dotF16(a, b, n); }

const char* sgc_simd(void) {
#if SGC_AVX2
  return "avx2";
#elif SGC_NEON
  return "neon";
#else
  return "scalar";
#endif
}

}  // extern "C"
