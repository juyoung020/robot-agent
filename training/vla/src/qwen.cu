// Qwen3.5 글 몸통 앞 계산·디코딩 — 설명은 include/qwen.h.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>

#include "gemm.cuh"
#include "qwen.h"
#include "qkern.cuh"
#include "st.h"

namespace rvla {

using namespace net;

QCfg tiny_cfg() {
  QCfg c;
  c.H = 64; c.I = 128; c.layers = 4;
  c.nq = 2; c.nkv = 1; c.hd = 32; c.rot = 8; c.theta = 10000.f;
  c.lh = 4; c.dk = 16; c.dv = 16; c.conv = 4; c.vocab = 256;   // lin_all = 200 (GEMM 은 8 의 배수)
  c.full = {0, 0, 0, 1};
  return c;
}

bool load_cfg(const std::string& dir, QCfg& c) {
  JsonLite j;
  if (!j.load(dir + "/config.json")) return false;
  const std::string t = j.section("text_config");
  c.H = (int)JsonLite::num(t, "hidden_size", 0);
  c.I = (int)JsonLite::num(t, "intermediate_size", 0);
  c.layers = (int)JsonLite::num(t, "num_hidden_layers", 0);
  c.nq = (int)JsonLite::num(t, "num_attention_heads", 0);
  c.nkv = (int)JsonLite::num(t, "num_key_value_heads", 0);
  c.hd = (int)JsonLite::num(t, "head_dim", 0);
  c.rot = (int)(c.hd * JsonLite::num(t, "partial_rotary_factor", 1.0));
  c.theta = (float)JsonLite::num(t, "rope_theta", 10000);
  c.eps = (float)JsonLite::num(t, "rms_norm_eps", 1e-6);
  c.lh = (int)JsonLite::num(t, "linear_num_value_heads", 0);
  const int lk = (int)JsonLite::num(t, "linear_num_key_heads", 0);
  c.dk = (int)JsonLite::num(t, "linear_key_head_dim", 0);
  c.dv = (int)JsonLite::num(t, "linear_value_head_dim", 0);
  c.conv = (int)JsonLite::num(t, "linear_conv_kernel_dim", 4);
  c.vocab = (int)JsonLite::num(t, "vocab_size", 0);
  const auto lt = JsonLite::strs(t, "layer_types");
  c.full.clear();
  for (auto& s : lt) c.full.push_back(s == "full_attention" ? 1 : 0);
  if (lk != c.lh || (int)c.full.size() != c.layers || t.find("\"attn_output_gate\": true") == std::string::npos) {
    std::fprintf(stderr, "qwen cfg: unsupported (lk %d lh %d, layer types %zu)\n", lk, c.lh, c.full.size());
    return false;
  }
  return true;
}

static long long al8(long long o) { return (o + 63) / 64 * 64; }
QLayout q_layout(const QCfg& c) {
  QLayout t;
  long long w = 0, v = 0;
  auto mt = [&](int N, int K) { MT m{w, N, K}; w = al8(w + (long long)N * K); return m; };
  auto vv = [&](long long n) { long long r = v; v = al8(v + n); return r; };
  t.emb = mt(c.vocab, c.H);
  for (int i = 0; i < c.layers; ++i) {
    QLayout::L L{};
    L.ln1 = vv(c.H); L.ln2 = vv(c.H);
    if (c.full[i]) {
      L.wqkv = mt(c.qg() + 2 * c.kvw(), c.H);
      L.wo = mt(c.H, c.nq * c.hd);
      L.qn = vv(c.hd); L.kn = vv(c.hd);
    } else {
      L.win = mt(c.lin_all(), c.H);
      L.wout = mt(c.H, c.lh * c.dv);
      L.convw = vv((long long)c.lin_in() * c.conv); L.alog = vv(c.lh); L.dtb = vv(c.lh); L.gnw = vv(c.dv);
    }
    L.wgu = mt(2 * c.I, c.H);
    L.wdn = mt(c.H, c.I);
    t.l.push_back(L);
  }
  t.lnf = vv(c.H);
  t.nW = w; t.nV = v;
  return t;
}

#define QCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

template <class T_>
T_* Qwen::alloc(size_t n) {
  void* p = nullptr;
  QCK(cudaMalloc(&p, sizeof(T_) * n + 256));
  QCK(cudaMemset(p, 0, sizeof(T_) * n + 256));
  allocs.push_back(p);
  bytes += sizeof(T_) * n;
  return reinterpret_cast<T_*>(p);
}
template float* Qwen::alloc<float>(size_t);
template uint16_t* Qwen::alloc<uint16_t>(size_t);

void Qwen::free_all() {
  for (void* p : allocs) cudaFree(p);
  allocs.clear();
  bytes = 0;
}

void Qwen::alloc_work(int rmax) {
  Rmax = rmax;
  const long long R = rmax;
  const long long wide = std::max<long long>({(long long)c.qg() + 2 * c.kvw(), c.lin_all(), 2LL * c.I});
  const long long ak = std::max<long long>({(long long)c.H, c.I, (long long)c.nq * c.hd, (long long)c.lh * c.dv});
  const long long qw = std::max<long long>((long long)c.nq * c.hd, (long long)c.lh * c.dk);
  X = alloc<float>(R * c.H);
  T0 = alloc<float>(R * wide);
  T1 = alloc<float>(R * std::max<long long>(c.lin_in(), 2 * c.kvw()));
  Qb = alloc<float>(R * qw * 2);           // Q | K(선형)
  Gb = alloc<float>(R * c.lh * 2);         // g | β
  Bb = alloc<float>(R * std::max<long long>((long long)c.nq * c.hd, (long long)c.lh * c.dv));   // 어텐션·재귀 출력
  lse = alloc<float>(R * c.nq);
  A = alloc<uint16_t>(R * ak);
  rs = alloc<float>(R);
}

bool Qwen::load(const std::string& dir, int rmax, std::string* err) {
  if (!load_cfg(dir, c)) { if (err) *err = "config"; return false; }
  lay = q_layout(c);
  StFile f;
  std::string path = dir + "/model.safetensors-00001-of-00001.safetensors";
  if (!f.open(path, err)) return false;
  std::vector<uint16_t> hw(lay.nW, 0);
  std::vector<float> hv(lay.nV, 0.f);
  const std::string P = "model.language_model.";
  auto putm = [&](const MT& m, long long row0, const std::string& name) {
    const StTensor* t = f.get(P + name);
    if (!t || t->dtype != "BF16") { std::fprintf(stderr, "qwen load: %s\n", name.c_str()); std::abort(); }
    memcpy(hw.data() + m.off + row0 * m.K, t->data, t->bytes);
    return (long long)(t->numel() / m.K);
  };
  auto putv = [&](long long off, const std::string& name) {
    auto x = f.f32(P + name);
    memcpy(hv.data() + off, x.data(), x.size() * 4);
  };
  putm(lay.emb, 0, "embed_tokens.weight");
  for (int i = 0; i < c.layers; ++i) {
    const auto& L = lay.l[i];
    const std::string p = "layers." + std::to_string(i) + ".";
    putv(L.ln1, p + "input_layernorm.weight");
    putv(L.ln2, p + "post_attention_layernorm.weight");
    if (c.full[i]) {
      long long r = putm(L.wqkv, 0, p + "self_attn.q_proj.weight");
      r += putm(L.wqkv, r, p + "self_attn.k_proj.weight");
      putm(L.wqkv, r, p + "self_attn.v_proj.weight");
      putm(L.wo, 0, p + "self_attn.o_proj.weight");
      putv(L.qn, p + "self_attn.q_norm.weight");
      putv(L.kn, p + "self_attn.k_norm.weight");
    } else {
      long long r = putm(L.win, 0, p + "linear_attn.in_proj_qkv.weight");
      r += putm(L.win, r, p + "linear_attn.in_proj_z.weight");
      r += putm(L.win, r, p + "linear_attn.in_proj_b.weight");
      putm(L.win, r, p + "linear_attn.in_proj_a.weight");
      putm(L.wout, 0, p + "linear_attn.out_proj.weight");
      putv(L.convw, p + "linear_attn.conv1d.weight");
      putv(L.alog, p + "linear_attn.A_log");
      putv(L.dtb, p + "linear_attn.dt_bias");
      putv(L.gnw, p + "linear_attn.norm.weight");
    }
    long long r = putm(L.wgu, 0, p + "mlp.gate_proj.weight");
    putm(L.wgu, r, p + "mlp.up_proj.weight");
    putm(L.wdn, 0, p + "mlp.down_proj.weight");
  }
  putv(lay.lnf, "norm.weight");
  Wb = alloc<uint16_t>(lay.nW);
  Pv = alloc<float>(lay.nV);
  QCK(cudaMemcpy(Wb, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  QCK(cudaMemcpy(Pv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
  alloc_work(rmax);
  return true;
}

void Qwen::init_random(const QCfg& cfg, int rmax, uint64_t seed) {
  c = cfg;
  lay = q_layout(c);
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<uint16_t> hw(lay.nW, 0);
  std::vector<float> hv(lay.nV, 0.f);
  auto fm = [&](const MT& m, float a) { for (long long i = 0; i < (long long)m.N * m.K; ++i) hw[m.off + i] = f2bf(a * u(g)); };
  auto fv = [&](long long off, int n, float base, float a) { for (int i = 0; i < n; ++i) hv[off + i] = base + a * u(g); };
  fm(lay.emb, 0.5f);
  const float s = 0.5f / std::sqrt((float)c.H);   // 작은 구성은 잔차 흐름이 잘 조건 지어지게 작은 가중치(bf16 바닥이 작아야 V5 가 날카로움)
  for (int i = 0; i < c.layers; ++i) {
    const auto& L = lay.l[i];
    fv(L.ln1, c.H, 0.f, 0.3f); fv(L.ln2, c.H, 0.f, 0.3f);
    if (c.full[i]) { fm(L.wqkv, 2 * s); fm(L.wo, 0.5f / std::sqrt((float)(c.nq * c.hd))); fv(L.qn, c.hd, 0.f, 0.3f); fv(L.kn, c.hd, 0.f, 0.3f); }
    else {
      fm(L.win, 2 * s); fm(L.wout, 0.5f / std::sqrt((float)(c.lh * c.dv)));
      fv(L.convw, c.lin_in() * c.conv, 0.f, 0.5f); fv(L.alog, c.lh, 0.5f, 0.5f); fv(L.dtb, c.lh, 0.f, 1.f); fv(L.gnw, c.dv, 1.f, 0.3f);
    }
    fm(L.wgu, 2 * s); fm(L.wdn, 0.5f / std::sqrt((float)c.I));
  }
  fv(lay.lnf, c.H, 0.f, 0.3f);
  Wb = alloc<uint16_t>(lay.nW);
  Pv = alloc<float>(lay.nV);
  QCK(cudaMemcpy(Wb, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  QCK(cudaMemcpy(Pv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
  alloc_work(rmax);
}

QCache Qwen::make_cache(int B, int Lmax) {
  QCache k;
  k.B = B; k.Lmax = Lmax;
  for (int i = 0; i < c.layers; ++i) {
    if (c.full[i]) {
      k.K.push_back(alloc<float>((size_t)B * Lmax * c.kvw()));
      k.V.push_back(alloc<float>((size_t)B * Lmax * c.kvw()));
      k.hist.push_back(nullptr); k.S.push_back(nullptr);
    } else {
      k.K.push_back(nullptr); k.V.push_back(nullptr);
      k.hist.push_back(alloc<float>((size_t)B * Lmax * c.lin_in()));
      k.S.push_back(alloc<float>((size_t)B * c.lh * c.dk * c.dv));
    }
  }
  return k;
}
void Qwen::free_cache(QCache& k) {
  auto fr = [&](float* p) {
    if (!p) return;
    for (size_t i = 0; i < allocs.size(); ++i) if (allocs[i] == p) { cudaFree(p); allocs.erase(allocs.begin() + i); return; }
  };
  for (auto p : k.K) fr(p);
  for (auto p : k.V) fr(p);
  for (auto p : k.hist) fr(p);
  for (auto p : k.S) fr(p);
  k = QCache{};
}

void Qwen::forward(const int* ids, const float* emb, int B, int n, int pos0, QCache* cache, float* hidden, float* taps, cudaStream_t st) {
  const int R = B * n;
  if (R > Rmax) { std::fprintf(stderr, "qwen forward: rows %d > Rmax %d\n", R, Rmax); std::abort(); }
  if (!cache && pos0 != 0) { std::fprintf(stderr, "qwen forward: pos0 needs cache\n"); std::abort(); }
  const size_t xb = sizeof(float) * (size_t)R * c.H;
  if (ids) qk::embed(ids, Wb + lay.emb.off, R, c.H, X, st);
  else QCK(cudaMemcpyAsync(X, emb, xb, cudaMemcpyDeviceToDevice, st));
  if (taps) QCK(cudaMemcpyAsync(taps, X, xb, cudaMemcpyDeviceToDevice, st));
  int fi = 0;
  for (int i = 0; i < c.layers; ++i) {
    const auto& L = lay.l[i];
    qk::rmsnorm(X, R, c.H, Pv + L.ln1, true, c.eps, A, nullptr, rs, st);
    if (c.full[i]) {
      const int kvw = c.kvw();
      qk::gemm_f32(A, c.H, R, Wb, L.wqkv, T0, L.wqkv.N, false, st);
      float *Kc, *Vc;
      int Lc;
      if (cache) { Kc = cache->K[i]; Vc = cache->V[i]; Lc = cache->Lmax; }
      else { Kc = T1; Vc = T1 + (size_t)R * kvw; Lc = n; }
      const float theta = bug == 1 ? 1e4f : bug == 5 ? c.theta * 10.f : c.theta;
      qk::full_prep(T0, L.wqkv.N, R, B, n, pos0, c.nq, c.nkv, c.hd, c.rot, theta, c.eps, Pv + L.qn, Pv + L.kn, bug == 4, Qb, Kc, Vc, Lc, st);
      qk::attn(Qb, Kc, Vc, B, n, pos0, Lc, c.nq, c.nkv, c.hd, Bb, lse, st);
      qk::gate(Bb, T0, L.wqkv.N, R, c.nq, c.hd, A, st);
      qk::gemm_f32(A, c.nq * c.hd, R, Wb, L.wo, X, c.H, true, st);
      ++fi;
    } else {
      const int li = c.lin_in();
      qk::gemm_f32(A, c.H, R, Wb, L.win, T0, c.lin_all(), false, st);
      const float* src = T0;
      int ld = c.lin_all(), Ls = n;
      float* S = nullptr;
      if (cache) {
        qk::hist_put(T0, c.lin_all(), B, n, pos0, li, cache->hist[i], cache->Lmax, st);
        src = cache->hist[i]; ld = li; Ls = cache->Lmax; S = cache->S[i];
      }
      qk::conv_silu(src, ld, Ls, B, n, pos0, li, c.conv, Pv + L.convw, T1, st);
      float* Qn = Qb;
      float* Kn = Qb + (size_t)R * c.lh * c.dk;
      qk::lin_prep(T1, li, T0, c.lin_all(), li + c.lh * c.dv, R, c.lh, c.dk, Pv + L.alog, Pv + L.dtb, bug == 4, Qn, Kn, Gb, Gb + (size_t)R * c.lh, st);
      qk::deltanet(Qn, Kn, T1 + 2 * c.lh * c.dk, li, Gb, Gb + (size_t)R * c.lh, B, n, c.lh, c.dk, c.dv, (cache && pos0 > 0) ? S : nullptr, S, bug == 3, Bb, st);
      qk::gnorm(Bb, T0 + li, c.lin_all(), R, c.lh, c.dv, Pv + L.gnw, c.eps, A, st);
      qk::gemm_f32(A, c.lh * c.dv, R, Wb, L.wout, X, c.H, true, st);
    }
    qk::rmsnorm(X, R, c.H, Pv + L.ln2, bug != 2, c.eps, A, nullptr, rs, st);
    qk::gemm_f32(A, c.H, R, Wb, L.wgu, T0, 2 * c.I, false, st);
    qk::swiglu(T0, R, c.I, A, st);
    qk::gemm_f32(A, c.I, R, Wb, L.wdn, X, c.H, true, st);
    if (taps) QCK(cudaMemcpyAsync(taps + (size_t)(i + 1) * R * c.H, X, xb, cudaMemcpyDeviceToDevice, st));
  }
  qk::rmsnorm(X, R, c.H, Pv + lay.lnf, true, c.eps, nullptr, hidden, rs, st);
  if (taps) QCK(cudaMemcpyAsync(taps + (size_t)(c.layers + 1) * R * c.H, hidden, xb, cudaMemcpyDeviceToDevice, st));
  (void)fi;
}

void Qwen::logits(const float* hidden, int rows, float* out, cudaStream_t st) {
  qk::f2bf_rows(hidden, rows, c.H, A, st);
  qk::gemm_f32(A, c.H, rows, Wb, lay.emb, out, c.vocab, false, st);
}

std::vector<std::vector<int>> generate(Qwen& m, const std::vector<std::vector<int>>& prompts, int max_new, int eos, float top_p, float temp,
                                       uint64_t seed, double* ms_per_tok) {
  const int B = (int)prompts.size(), n = (int)prompts[0].size();
  std::vector<int> flat;
  for (auto& p : prompts) flat.insert(flat.end(), p.begin(), p.end());
  int *dids, *dnext;
  float *hid, *lg, *last;
  QCK(cudaMalloc(&dids, sizeof(int) * flat.size()));
  QCK(cudaMalloc(&dnext, sizeof(int) * B));
  QCK(cudaMalloc(&hid, sizeof(float) * flat.size() * m.c.H));
  QCK(cudaMalloc(&last, sizeof(float) * B * m.c.H));
  QCK(cudaMalloc(&lg, sizeof(float) * (size_t)B * m.c.vocab));
  QCK(cudaMemcpy(dids, flat.data(), sizeof(int) * flat.size(), cudaMemcpyHostToDevice));
  QCache k = m.make_cache(B, n + max_new + 1);
  cudaStream_t st = 0;
  std::vector<std::vector<int>> out(B);
  std::vector<int> nx(B);
  std::vector<float> hl;
  std::mt19937_64 rng(seed);
  std::vector<char> done(B, 0);
  m.forward(dids, nullptr, B, n, 0, &k, hid, nullptr, st);
  qk::gather_last(hid, B, n, m.c.H, last, st);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  int steps = 0;
  for (int s = 0; s < max_new; ++s) {
    m.logits(last, B, lg, st);
    if (top_p <= 0.f) {
      qk::argmax_rows(lg, B, m.c.vocab, dnext, st);
      QCK(cudaMemcpy(nx.data(), dnext, sizeof(int) * B, cudaMemcpyDeviceToHost));
    } else {
      hl.resize((size_t)B * m.c.vocab);
      QCK(cudaMemcpy(hl.data(), lg, sizeof(float) * hl.size(), cudaMemcpyDeviceToHost));
      for (int b = 0; b < B; ++b) {
        const float* l = hl.data() + (size_t)b * m.c.vocab;
        float mx = -1e30f;
        for (int v = 0; v < m.c.vocab; ++v) mx = std::max(mx, l[v]);
        std::vector<std::pair<float, int>> pr;
        double z = 0;
        for (int v = 0; v < m.c.vocab; ++v) { const double e = std::exp((l[v] - mx) / temp); z += e; pr.push_back({(float)e, v}); }
        std::sort(pr.begin(), pr.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double cum = 0;
        size_t kk = 0;
        for (; kk < pr.size(); ++kk) { cum += pr[kk].first / z; if (cum >= top_p) { ++kk; break; } }
        double tot = 0;
        for (size_t i = 0; i < kk; ++i) tot += pr[i].first;
        double r = std::uniform_real_distribution<double>(0, tot)(rng);
        size_t i = 0;
        for (; i + 1 < kk; ++i) { r -= pr[i].first; if (r <= 0) break; }
        nx[b] = pr[i].second;
      }
      QCK(cudaMemcpy(dnext, nx.data(), sizeof(int) * B, cudaMemcpyHostToDevice));
    }
    bool all = true;
    for (int b = 0; b < B; ++b) { if (!done[b]) { out[b].push_back(nx[b]); if (nx[b] == eos) done[b] = 1; } all = all && done[b]; }
    if (all || s + 1 == max_new) break;
    if (s == 1) cudaEventRecord(e0, st);
    m.forward(dnext, nullptr, B, 1, n + s, &k, last, nullptr, st);
    ++steps;
  }
  cudaEventRecord(e1, st);
  QCK(cudaEventSynchronize(e1));
  if (ms_per_tok) {
    float ms = 0;
    if (steps > 2) { cudaEventElapsedTime(&ms, e0, e1); *ms_per_tok = ms / (steps - 1); } else *ms_per_tok = 0;
  }
  m.free_cache(k);
  cudaFree(dids); cudaFree(dnext); cudaFree(hid); cudaFree(last); cudaFree(lg);
  return out;
}

bool Vocab::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t n;
  f.read((char*)&n, 4);
  tok.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t l;
    f.read((char*)&l, 4);
    tok[i].resize(l);
    if (l) f.read(&tok[i][0], l);
  }
  return (bool)f;
}
std::string Vocab::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int i : ids) if (i >= 0 && i < (int)tok.size()) s += tok[i];
  return s;
}

}  // namespace rvla
