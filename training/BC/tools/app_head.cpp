// MAP 풀링 + embed 머리 h — 설명은 app_head.h
#include "app_head.h"

#include <dirent.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace apph {

static bool find_tensor(const std::string& hdr, const std::string& name, size_t& a, size_t& b, std::string& dtype) {
  const std::string key = "\"" + name + "\"";
  const size_t k = hdr.find(key);
  if (k == std::string::npos) return false;
  const size_t e = hdr.find('}', k);
  const std::string obj = hdr.substr(k, e - k);
  const size_t dt = obj.find("\"dtype\"");
  const size_t q1 = obj.find('"', obj.find(':', dt) + 1), q2 = obj.find('"', q1 + 1);
  dtype = obj.substr(q1 + 1, q2 - q1 - 1);
  const size_t lb = obj.find('[', obj.find("\"data_offsets\""));
  a = std::strtoull(obj.c_str() + lb + 1, nullptr, 10);
  b = std::strtoull(obj.c_str() + obj.find(',', lb) + 1, nullptr, 10);
  return true;
}
static std::string default_path() {
  const char* home = std::getenv("HOME");
  const std::string base = std::string(home ? home : "") + "/.cache/huggingface/hub/models--timm--ViT-B-32-SigLIP2-256/snapshots";
  DIR* d = opendir(base.c_str());
  if (!d) return "";
  std::string out;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    const std::string p = base + "/" + e->d_name + "/open_clip_model.safetensors";
    if (FILE* f = std::fopen(p.c_str(), "rb")) { std::fclose(f); out = p; break; }
  }
  closedir(d);
  return out;
}

bool load(const std::string& st_in, const std::string& head_path, Weights& w) {
  const std::string path = st_in.empty() ? default_path() : st_in;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "apph: cannot open %s\n", path.c_str()); return false; }
  uint64_t hl = 0;
  if (std::fread(&hl, 8, 1, f) != 1) return false;
  std::string hdr(hl, '\0');
  if (std::fread(&hdr[0], 1, hl, f) != hl) return false;
  bool ok = true;
  auto rd = [&](const char* name, std::vector<float>& v, size_t n) {
    size_t a, b;
    std::string dt;
    const std::string full = std::string("visual.trunk.attn_pool.") + name;
    if (!find_tensor(hdr, full, a, b, dt) || dt != "F32" || b - a != n * 4) { std::fprintf(stderr, "apph: bad %s\n", full.c_str()); ok = false; return; }
    v.resize(n);
    std::fseek(f, 8 + (long long)hl + (long long)a, SEEK_SET);
    if (std::fread(v.data(), 4, n, f) != n) ok = false;
  };
  rd("latent", w.latent, D);
  rd("q.weight", w.q_w, (size_t)D * D); rd("q.bias", w.q_b, D);
  rd("kv.weight", w.kv_w, (size_t)2 * D * D); rd("kv.bias", w.kv_b, 2 * D);
  rd("proj.weight", w.proj_w, (size_t)D * D); rd("proj.bias", w.proj_b, D);
  rd("norm.weight", w.ln_g, D); rd("norm.bias", w.ln_b, D);
  rd("mlp.fc1.weight", w.fc1_w, (size_t)MLP * D); rd("mlp.fc1.bias", w.fc1_b, MLP);
  rd("mlp.fc2.weight", w.fc2_w, (size_t)D * MLP); rd("mlp.fc2.bias", w.fc2_b, D);
  std::fclose(f);
  // 머리 h: 차례대로 ln.weight ln.bias fc1.w fc1.b fc2.w fc2.b out.w out.b (FP32, export_head_f32.py)
  FILE* g = std::fopen(head_path.c_str(), "rb");
  if (!g) { std::fprintf(stderr, "apph: cannot open head %s\n", head_path.c_str()); return false; }
  auto rh = [&](std::vector<float>& v, size_t n) { v.resize(n); if (std::fread(v.data(), 4, n, g) != n) ok = false; };
  rh(w.h_ln_g, D); rh(w.h_ln_b, D);
  rh(w.h_fc1_w, (size_t)HID * D); rh(w.h_fc1_b, HID);
  rh(w.h_fc2_w, (size_t)D * HID); rh(w.h_fc2_b, D);
  rh(w.h_out_w, (size_t)OUT * D); rh(w.h_out_b, OUT);
  char extra;
  if (std::fread(&extra, 1, 1, g) == 1) { std::fprintf(stderr, "apph: head file longer than expected\n"); ok = false; }
  std::fclose(g);
  return ok;
}

static void linear(const double* x, int K, const std::vector<float>& W, const std::vector<float>& b, int N, double* y) {
  for (int n = 0; n < N; ++n) {
    double s = b[n];
    const float* wr = W.data() + (size_t)n * K;
    for (int k = 0; k < K; ++k) s += (double)wr[k] * x[k];
    y[n] = s;
  }
}
static void layernorm(const double* x, int n, const std::vector<float>& g, const std::vector<float>& b, double eps, double* y) {
  double m = 0, v = 0;
  for (int i = 0; i < n; ++i) m += x[i];
  m /= n;
  for (int i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
  v /= n;
  const double r = 1.0 / std::sqrt(v + eps);
  for (int i = 0; i < n; ++i) y[i] = (x[i] - m) * r * g[i] + b[i];
}

void encode(const Weights& w, const float* tok, int ld, double* pooled, double* out, int bug) {
  std::vector<double> x((size_t)NTOK * D), kv((size_t)NTOK * 2 * D), lat(D), q(D), att(D), y(D), t(D), h(MLP);
  for (int i = 0; i < NTOK; ++i)
    for (int c = 0; c < D; ++c) x[(size_t)i * D + c] = tok[(size_t)i * ld + c];
  for (int i = 0; i < NTOK; ++i) linear(&x[(size_t)i * D], D, w.kv_w, w.kv_b, 2 * D, &kv[(size_t)i * 2 * D]);
  for (int c = 0; c < D; ++c) lat[c] = w.latent[c];
  linear(lat.data(), D, w.q_w, w.q_b, D, q.data());
  const double scale = bug == 1 ? 1.0 : 1.0 / std::sqrt((double)HD);
  for (int hh = 0; hh < HEADS; ++hh) {
    double s[NTOK], mx = -1e300;
    for (int i = 0; i < NTOK; ++i) {
      double d = 0;
      for (int k = 0; k < HD; ++k) d += q[hh * HD + k] * kv[(size_t)i * 2 * D + hh * HD + k];
      s[i] = d * scale;
      mx = s[i] > mx ? s[i] : mx;
    }
    double z = 0;
    for (int i = 0; i < NTOK; ++i) { s[i] = std::exp(s[i] - mx); z += s[i]; }
    for (int k = 0; k < HD; ++k) {
      double a = 0;
      for (int i = 0; i < NTOK; ++i) a += s[i] * kv[(size_t)i * 2 * D + D + hh * HD + k];
      att[hh * HD + k] = a / z;
    }
  }
  linear(att.data(), D, w.proj_w, w.proj_b, D, y.data());
  layernorm(y.data(), D, w.ln_g, w.ln_b, 1e-6, t.data());
  linear(t.data(), D, w.fc1_w, w.fc1_b, MLP, h.data());
  for (int j = 0; j < MLP; ++j) {   // GELU tanh
    const double u = h[j];
    h[j] = 0.5 * u * (1.0 + std::tanh(0.7978845608028654 * (u + 0.044715 * u * u * u)));
  }
  linear(h.data(), MLP, w.fc2_w, w.fc2_b, D, t.data());
  for (int c = 0; c < D; ++c) pooled[c] = bug == 3 ? t[c] : y[c] + t[c];
  // 머리 h
  std::vector<double> a(D), hb(HID), r(D);
  if (bug == 2) for (int c = 0; c < D; ++c) a[c] = pooled[c];
  else layernorm(pooled, D, w.h_ln_g, w.h_ln_b, 1e-5, a.data());
  linear(a.data(), D, w.h_fc1_w, w.h_fc1_b, HID, hb.data());
  for (int j = 0; j < HID; ++j) hb[j] = 0.5 * hb[j] * (1.0 + std::erf(hb[j] / std::sqrt(2.0)));
  linear(hb.data(), HID, w.h_fc2_w, w.h_fc2_b, D, r.data());
  for (int c = 0; c < D; ++c) a[c] += r[c];
  linear(a.data(), D, w.h_out_w, w.h_out_b, OUT, out);
  double n2 = 0;
  for (int c = 0; c < OUT; ++c) n2 += out[c] * out[c];
  const double inv = 1.0 / std::sqrt(n2 > 1e-24 ? n2 : 1e-24);
  for (int c = 0; c < OUT; ++c) out[c] *= inv;
}

}  // namespace apph
