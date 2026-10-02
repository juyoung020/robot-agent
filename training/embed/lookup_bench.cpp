// Label/object lookup micro-benchmark: exact FP32 vs INT8 (+FP32 re-rank of top 16) vs binary sign bits (+FP32 re-rank of top R).
//   g++ -O3 -march=native -std=c++17 lookup_bench.cpp -o lookup_bench
//   ./lookup_bench <bank.f16> <queries.f16> <dim> [rerank=32]
// bank: K x dim FP16 (L2-normalised rows), queries: Q x dim FP16. Prints per-query latency and agreement with exact
// top-1 / top-5 (recall of exact top-1 inside the approximate top-5). Single thread, no SIMD intrinsics: the compiler
// auto-vectorises the int8 dot (SSE/AVX here, NEON on aarch64). Nano numbers are estimated from this, not measured.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static float h2f(uint16_t h) {
  uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
  if (e == 0) {
    if (m == 0) f = s;
    else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; --e; } m &= 0x3ff; f = s | (e << 23) | (m << 13); }
  } else if (e == 31) f = s | 0x7f800000u | (m << 13);
  else f = s | ((e + 127 - 15) << 23) | (m << 13);
  float r; std::memcpy(&r, &f, 4); return r;
}

static std::vector<float> load(const char* p, int dim, int& n) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  size_t bytes = f.tellg(); f.seekg(0);
  std::vector<uint16_t> h(bytes / 2); f.read((char*)h.data(), bytes);
  n = int(h.size() / dim);
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) o[i] = h2f(h[i]);
  return o;
}

struct Int8 { std::vector<int8_t> q; std::vector<float> s; };
static Int8 quant(const std::vector<float>& x, int n, int d) {   // per-row symmetric scale
  Int8 r; r.q.resize(size_t(n) * d); r.s.resize(n);
  for (int i = 0; i < n; ++i) {
    float m = 1e-12f; for (int j = 0; j < d; ++j) m = std::max(m, std::fabs(x[size_t(i) * d + j]));
    r.s[i] = m / 127.f;
    for (int j = 0; j < d; ++j) r.q[size_t(i) * d + j] = int8_t(std::lrint(x[size_t(i) * d + j] / r.s[i]));
  }
  return r;
}
static inline int32_t dot8(const int8_t* a, const int8_t* b, int d) {
  int32_t s = 0; for (int j = 0; j < d; ++j) s += int32_t(a[j]) * int32_t(b[j]); return s;
}
static inline float dotf(const float* a, const float* b, int d) {
  float s[8] = {0}; int j = 0;                       // 8 partial sums so the compiler vectorises without -ffast-math
  for (; j + 8 <= d; j += 8) for (int k = 0; k < 8; ++k) s[k] += a[j + k] * b[j + k];
  float t = 0; for (; j < d; ++j) t += a[j] * b[j];
  for (int k = 0; k < 8; ++k) t += s[k]; return t;
}
static std::vector<uint64_t> bits(const std::vector<float>& x, int n, int d) {
  int w = (d + 63) / 64; std::vector<uint64_t> b(size_t(n) * w, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < d; ++j) if (x[size_t(i) * d + j] > 0) b[size_t(i) * w + j / 64] |= 1ull << (j % 64);
  return b;
}

template <class F> static std::vector<int> topk(int n, int k, F score) {
  std::vector<std::pair<float, int>> h; h.reserve(k + 1);
  for (int i = 0; i < n; ++i) {
    float s = score(i);
    if ((int)h.size() < k) { h.push_back({s, i}); std::push_heap(h.begin(), h.end(), std::greater<>()); }
    else if (s > h.front().first) { std::pop_heap(h.begin(), h.end(), std::greater<>()); h.back() = {s, i}; std::push_heap(h.begin(), h.end(), std::greater<>()); }
  }
  std::sort(h.begin(), h.end(), std::greater<>());
  std::vector<int> o; for (auto& p : h) o.push_back(p.second); return o;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s bank.f16 queries.f16 dim [rerank]\n", argv[0]); return 1; }
  int d = std::atoi(argv[3]), R = argc > 4 ? std::atoi(argv[4]) : 32, K, Q;
  auto B = load(argv[1], d, K); auto X = load(argv[2], d, Q);
  auto Bq = quant(B, K, d); auto Xq = quant(X, Q, d);
  auto Bb = bits(B, K, d); auto Xb = bits(X, Q, d); int w = (d + 63) / 64;
  using clk = std::chrono::steady_clock;
  std::vector<std::vector<int>> ex(Q);
  double t_ex = 0, t_i8 = 0, t_bin = 0; int a1_i8 = 0, a5_i8 = 0, a1_bin = 0, a5_bin = 0;
  for (int q = 0; q < Q; ++q) {
    const float* x = &X[size_t(q) * d];
    auto t0 = clk::now();
    ex[q] = topk(K, 5, [&](int i) { return dotf(x, &B[size_t(i) * d], d); });
    auto t1 = clk::now();
    const int8_t* xq = &Xq.q[size_t(q) * d];
    auto c8 = topk(K, 16, [&](int i) { return float(dot8(xq, &Bq.q[size_t(i) * d], d)) * Bq.s[i]; });
    auto i8 = topk((int)c8.size(), 5, [&](int r) { return dotf(x, &B[size_t(c8[r]) * d], d); });   // fp32 re-rank of 16
    for (auto& v : i8) v = c8[v];
    auto t2 = clk::now();
    const uint64_t* xb = &Xb[size_t(q) * w];
    auto cand = topk(K, R, [&](int i) { int c = 0; for (int j = 0; j < w; ++j) c += __builtin_popcountll(xb[j] ^ Bb[size_t(i) * w + j]); return float(-c); });
    auto bin = topk((int)cand.size(), 5, [&](int r) { return dotf(x, &B[size_t(cand[r]) * d], d); });
    for (auto& v : bin) v = cand[v];
    auto t3 = clk::now();
    t_ex += std::chrono::duration<double, std::micro>(t1 - t0).count();
    t_i8 += std::chrono::duration<double, std::micro>(t2 - t1).count();
    t_bin += std::chrono::duration<double, std::micro>(t3 - t2).count();
    a1_i8 += i8[0] == ex[q][0]; a5_i8 += std::count(i8.begin(), i8.end(), ex[q][0]);
    a1_bin += bin[0] == ex[q][0]; a5_bin += std::count(bin.begin(), bin.end(), ex[q][0]);
  }
  std::printf("K %d dim %d Q %d | us/query: fp32 %.1f  int8+rr16 %.1f  bin%d+rr %.1f | top1 agree int8 %.3f bin %.3f | exact-top1 in top5 int8 %.3f bin %.3f\n",
              K, d, Q, t_ex / Q, t_i8 / Q, R, t_bin / Q, double(a1_i8) / Q, double(a1_bin) / Q, double(a5_i8) / Q, double(a5_bin) / Q);
  return 0;
}
