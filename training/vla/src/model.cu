// RecallVLA 모델 앞·뒤 — 설명은 include/model.h.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "model.h"
#include "net.h"
#include "qkern.cuh"
#include "st.h"
#include "tkern.cuh"

namespace rvla {

using net::bf2f;
using net::f2bf;
#define MCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
#define MKC() MCK(cudaGetLastError())
static unsigned nb(long long n, int t = 256) { return (unsigned)((n + t - 1) / t); }
constexpr int TEMB = 32;
constexpr int DWCH = 1024;   // dW split-K 조각 행 수
// 비교용: RVLA_DN_OLD=1 이면 학습 DeltaNet 을 예전 재귀 꼴(FP32, 검문점 16 스텝)로
static bool dn_old() {
  static const bool o = [] { const char* e = getenv("RVLA_DN_OLD"); return e && e[0] == '1'; }();
  return o;
}

VCfg tiny_vcfg() {
  VCfg c;
  c.q = tiny_cfg();
  c.vD = 64; c.vHeads = 2; c.vMLP = 128; c.vL = 2; c.vT = 4; c.vK = 192; c.cams = 3;
  c.obj_hid = 64;
  c.De = 64; c.Ie = 128; c.Hc = 4; c.A = 8;
  c.Bmax = 2; c.Lmax = 48; c.Mtmax = 16;
  c.vocab_chunk = 96;
  return c;
}

struct Model::WS {
  int R = 0, Rv = 0, RA = 0, nf = 0, W0 = 0, QKW = 0, OW = 0;
  std::vector<float*> Xs, Kf, Vf, dKx, dVx, vX, eX;
  float *hn, *T0, *T1, *QK, *O, *lse, *Gb, *Bb, *Xmid, *GU;
  uint16_t *A1, *A2, *Ag, *Hh;
  float *dR, *dA, *dHh, *dAg, *dO, *dT0, *dT1, *dp, *dQK, *dK, *dV, *Dd, *dG, *dBt, *wt, *part, *dnws, *dnws0 = nullptr, *ws, *dHn;
  uint16_t *dRb, *dGU, *dT0b;
  // 영상
  float *vQKV, *vO, *vH1, *vlse, *vXm, *vOut, *vdR, *vdA, *vdQKV, *vdO, *vdH, *vDd, *vdOut;
  uint16_t *vA1, *vAo, *vA2, *vAg, *vTok, *vdb;
  // 묶음
  float* gOut[N_VG];
  float* dgOut[N_VG];
  float *gH1, *dgH;
  uint16_t *gH, *gdb;
  // 글
  uint16_t *hnT, *dLg;
  float *lg, *dHnT, *rm, *rs, *tl, *dEc, *lpart;
  // 전문가
  uint16_t *ain, *eA1, *eAg, *eA2, *eHh, *Ao, *dz;
  float *tau, *eps, *u, *eT, *eQ, *eK, *eV, *eO, *else_, *eXm, *eGU, *edR, *edA, *edHh, *edAg, *edO, *edT0, *edQ, *edK, *edV, *eDd;
  uint16_t *edRb, *edGU, *edT0b;
};

template <class T_>
T_* Model::alloc(size_t n) {
  void* p = nullptr;
  MCK(cudaMalloc(&p, sizeof(T_) * n + 256));
  MCK(cudaMemset(p, 0, sizeof(T_) * n + 256));
  allocs.push_back(p);
  bytes += sizeof(T_) * n;
  return reinterpret_cast<T_*>(p);
}

void Model::free_all() {
  for (void* p : allocs) cudaFree(p);
  allocs.clear();
  delete w;
  w = nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// 초기화
bool Model::init(const VCfg& cfg, const std::string& qdir, const std::string& sigpath, std::string* err) {
  c = cfg;
  const int H = c.q.H;
  if (qdir.empty()) q.init_random(c.q, 64, c.seed);
  else {
    if (!q.load(qdir, 256, err)) return false;
    c.q = q.c;
  }
  const QCfg& Q = q.c;
  const int nf = Q.n_full();
  // 몸통 변수 묶음
  qp.W = q.Wb; qp.V = q.Pv; qp.nW = q.lay.nW; qp.nV = q.lay.nV;
  qp.GW = alloc<uint16_t>(qp.nW);
  qp.GV = alloc<float>(qp.nV);
  // 그 밖 변수 자리
  auto MTo = [&](int N, int K) { MT m{ap.mat(N, K), N, K}; return m; };
  v_patch = MTo(c.vD, c.vK); v_patchb = ap.vec(c.vD); v_pos = ap.vec((long long)c.vT * c.vD);
  for (int l = 0; l < c.vL; ++l) {
    VBlk b{};
    b.ln1g = ap.vec(c.vD); b.ln1b = ap.vec(c.vD);
    b.qkv = MTo(3 * c.vD, c.vD); b.qkvb = ap.vec(3 * c.vD);
    b.proj = MTo(c.vD, c.vD); b.projb = ap.vec(c.vD);
    b.ln2g = ap.vec(c.vD); b.ln2b = ap.vec(c.vD);
    b.fc1 = MTo(c.vMLP, c.vD); b.fc1b = ap.vec(c.vMLP);
    b.fc2 = MTo(c.vD, c.vMLP); b.fc2b = ap.vec(c.vD);
    vb.push_back(b);
  }
  v_lnfg = ap.vec(c.vD); v_lnfb = ap.vec(c.vD);
  v_proj = MTo(H, c.vD); v_projb = ap.vec(H);
  for (int g = 0; g < N_VG; ++g) {
    g_w1[g] = MTo(g == VG_OBJ ? c.obj_hid : H, kVGrp[g].K);
    g_type[g] = ap.vec(H);
  }
  g_w2 = MTo(H, c.obj_hid); g_b2 = ap.vec(H);
  e_in = MTo(c.De, c.kpad());
  for (int f = 0; f < nf; ++f) {
    EBlk e{};
    e.ln1 = ap.vec(c.De); e.ln2 = ap.vec(c.De); e.qn = ap.vec(Q.hd); e.kn = ap.vec(Q.hd);
    e.qkv = MTo(Q.qg() + 2 * Q.kvw(), c.De);
    e.o = MTo(c.De, Q.nq * Q.hd);
    e.gu = MTo(2 * c.Ie, c.De);
    e.dn = MTo(c.De, c.Ie);
    eb.push_back(e);
  }
  e_lnf = ap.vec(c.De);
  e_out = MTo(c.A, c.De);
  ap.W = alloc<uint16_t>(ap.nW); ap.GW = alloc<uint16_t>(ap.nW);
  ap.V = alloc<float>(ap.nV); ap.GV = alloc<float>(ap.nV);
  // 값: 무작위(결정적) + SigLIP 2 가중치(있으면)
  std::vector<uint16_t> hw(ap.nW, 0);
  std::vector<float> hv(ap.nV, 0.f);
  std::mt19937_64 rg(c.seed * 7919 + 17);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  auto fill = [&](const MT& m, int kreal, float gain) {
    const float a = gain * std::sqrt(3.f / (float)kreal);
    for (int n = 0; n < m.N; ++n)
      for (int k = 0; k < kreal; ++k) hw[m.off + (long long)n * m.K + k] = f2bf(a * U(rg));
  };
  // 임베딩 크기(사영·묶음 출력을 같은 크기로)
  double es = 0;
  {
    std::vector<uint16_t> e((size_t)std::min<long long>((long long)Q.vocab * H, 4LL << 20));
    MCK(cudaMemcpy(e.data(), q.Wb + q.lay.emb.off, e.size() * 2, cudaMemcpyDeviceToHost));
    for (auto x : e) { const double v = bf2f(x); es += v * v; }
    es = std::sqrt(es / e.size());
  }
  bool sig = false;
  if (!sigpath.empty()) {
    StFile f;
    if (!f.open(sigpath, err)) return false;
    auto pm = [&](const MT& m, const std::string& nm) { auto x = f.f32("visual.trunk." + nm); for (size_t i = 0; i < x.size(); ++i) hw[m.off + i] = f2bf(x[i]); };
    auto pv = [&](long long o, const std::string& nm) { auto x = f.f32("visual.trunk." + nm); memcpy(hv.data() + o, x.data(), x.size() * 4); };
    pm(v_patch, "patch_embed.proj.weight"); pv(v_patchb, "patch_embed.proj.bias"); pv(v_pos, "pos_embed");
    for (int l = 0; l < c.vL; ++l) {
      const std::string p = "blocks." + std::to_string(l) + ".";
      const VBlk& b = vb[l];
      pv(b.ln1g, p + "norm1.weight"); pv(b.ln1b, p + "norm1.bias");
      pm(b.qkv, p + "attn.qkv.weight"); pv(b.qkvb, p + "attn.qkv.bias");
      pm(b.proj, p + "attn.proj.weight"); pv(b.projb, p + "attn.proj.bias");
      pv(b.ln2g, p + "norm2.weight"); pv(b.ln2b, p + "norm2.bias");
      pm(b.fc1, p + "mlp.fc1.weight"); pv(b.fc1b, p + "mlp.fc1.bias");
      pm(b.fc2, p + "mlp.fc2.weight"); pv(b.fc2b, p + "mlp.fc2.bias");
    }
    pv(v_lnfg, "norm.weight"); pv(v_lnfb, "norm.bias");
    sig = true;
  }
  if (!sig) {
    fill(v_patch, c.vK, 1.f);
    for (int i = 0; i < c.vT * c.vD; ++i) hv[v_pos + i] = 0.1f * U(rg);
    for (auto& b : vb) {
      for (int i = 0; i < c.vD; ++i) { hv[b.ln1g + i] = 1.f + 0.2f * U(rg); hv[b.ln2g + i] = 1.f + 0.2f * U(rg); hv[b.ln1b + i] = 0.1f * U(rg); hv[b.ln2b + i] = 0.1f * U(rg); }
      fill(b.qkv, c.vD, 0.5f); fill(b.proj, c.vD, 0.3f); fill(b.fc1, c.vD, 0.5f); fill(b.fc2, c.vMLP, 0.3f);
      for (int i = 0; i < 3 * c.vD; ++i) hv[b.qkvb + i] = 0.1f * U(rg);
      for (int i = 0; i < c.vMLP; ++i) hv[b.fc1b + i] = 0.1f * U(rg);
      for (int i = 0; i < c.vD; ++i) { hv[b.projb + i] = 0.1f * U(rg); hv[b.fc2b + i] = 0.1f * U(rg); hv[v_patchb + i] = 0.1f * U(rg); }
    }
    for (int i = 0; i < c.vD; ++i) { hv[v_lnfg + i] = 1.f + 0.2f * U(rg); hv[v_lnfb + i] = 0.1f * U(rg); }
  }
  // 사영·묶음: 출력 표준편차 ≈ 임베딩 크기(es)
  const float pg = (float)es;
  fill(v_proj, c.vD, pg);
  for (int g = 0; g < N_VG; ++g) {
    if (g == VG_OBJ) fill(g_w1[g], kVGrp[g].k_real, 1.f);
    else fill(g_w1[g], kVGrp[g].k_real, pg);
    for (int i = 0; i < H; ++i) hv[g_type[g] + i] = 0.1f * pg * U(rg);
  }
  fill(g_w2, c.obj_hid, pg);
  fill(e_in, c.A + TEMB, 1.f);
  const float res = 1.f / std::sqrt(2.f * (float)std::max(1, nf));
  const float tg = qdir.empty() ? 0.5f : 1.f;   // 작은 구성(V5): 작은 가중치
  for (auto& e : eb) {
    fill(e.qkv, c.De, tg); fill(e.o, Q.nq * Q.hd, res * tg); fill(e.gu, c.De, tg); fill(e.dn, c.Ie, res * tg);
    if (qdir.empty())
      for (int i = 0; i < c.De; ++i) { hv[e.ln1 + i] = 0.2f * U(rg); hv[e.ln2 + i] = 0.2f * U(rg); }
    if (qdir.empty())
      for (int i = 0; i < Q.hd; ++i) { hv[e.qn + i] = 0.2f * U(rg); hv[e.kn + i] = 0.2f * U(rg); }
  }
  if (qdir.empty()) for (int i = 0; i < c.De; ++i) hv[e_lnf + i] = 0.2f * U(rg);
  fill(e_out, c.De, qdir.empty() ? 1.f : 0.01f);
  MCK(cudaMemcpy(ap.W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  MCK(cudaMemcpy(ap.V, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));

  // ---- 작업 버퍼 ----
  w = new WS();
  WS& s = *w;
  const int R = c.Bmax * c.Lmax, Rv = c.Bmax * c.cams * c.vT, RA = c.Bmax * c.Hc;
  s.R = R; s.Rv = Rv; s.RA = RA; s.nf = nf;
  s.W0 = std::max(Q.qg() + 2 * Q.kvw(), Q.lin_all());
  s.QKW = std::max(Q.nq * Q.hd, 2 * Q.lh * Q.dk);
  s.OW = std::max(Q.nq * Q.hd, Q.lh * Q.dv);
  for (int l = 0; l <= Q.layers; ++l) s.Xs.push_back(alloc<float>((size_t)R * H));
  for (int f = 0; f < nf; ++f) {
    s.Kf.push_back(alloc<float>((size_t)R * Q.kvw())); s.Vf.push_back(alloc<float>((size_t)R * Q.kvw()));
    if (!c.ki) { s.dKx.push_back(alloc<float>((size_t)R * Q.kvw())); s.dVx.push_back(alloc<float>((size_t)R * Q.kvw())); }
  }
  s.hn = alloc<float>((size_t)R * H); hidden = s.hn;
  s.T0 = alloc<float>((size_t)R * s.W0); s.T1 = alloc<float>((size_t)R * Q.lin_in());
  s.QK = alloc<float>((size_t)R * s.QKW); s.O = alloc<float>((size_t)R * s.OW); s.lse = alloc<float>((size_t)R * Q.nq);
  s.Gb = alloc<float>((size_t)R * Q.lh); s.Bb = alloc<float>((size_t)R * Q.lh);
  s.Xmid = alloc<float>((size_t)R * H); s.GU = alloc<float>((size_t)R * 2 * Q.I);
  s.A1 = alloc<uint16_t>((size_t)R * H); s.A2 = alloc<uint16_t>((size_t)R * H); s.Ag = alloc<uint16_t>((size_t)R * s.OW); s.Hh = alloc<uint16_t>((size_t)R * Q.I);
  s.dR = alloc<float>((size_t)R * H); s.dA = alloc<float>((size_t)R * H); s.dHh = alloc<float>((size_t)R * Q.I); s.dAg = alloc<float>((size_t)R * s.OW);
  s.dO = alloc<float>((size_t)R * s.OW); s.dT0 = alloc<float>((size_t)R * s.W0); s.dT1 = alloc<float>((size_t)R * Q.lin_in()); s.dp = alloc<float>((size_t)R * Q.lin_in());
  s.dQK = alloc<float>((size_t)R * s.QKW); s.dK = alloc<float>((size_t)R * Q.kvw()); s.dV = alloc<float>((size_t)R * Q.kvw());
  s.Dd = alloc<float>((size_t)R * Q.nq); s.dG = alloc<float>((size_t)R * Q.lh); s.dBt = alloc<float>((size_t)R * Q.lh);
  s.dHn = alloc<float>((size_t)R * H);
  const long long wtn = std::max<long long>({(long long)R * H, (long long)R * (Q.nq + Q.nkv) * Q.hd, (long long)R * Q.lh * Q.dv, 2LL * R * Q.lh,
                                            (long long)Rv * 3 * c.vD, (long long)RA * std::max(c.De, (Q.nq + Q.nkv) * Q.hd), (long long)c.Bmax * 16 * H});
  s.wt = alloc<float>(wtn);
  s.part = alloc<float>((size_t)((std::max({R, Rv, RA}) + 255) / 256 + 1) * std::max<long long>({(long long)s.W0, (long long)Q.lin_in() * Q.conv, 3LL * c.vD, (long long)c.vMLP, (long long)c.vT * c.vD, (long long)H, 2LL * c.Ie}));
  s.dnws = alloc<float>(tk::dnc_ws_floats(c.Bmax, c.Lmax, Q.lh, Q.dk, Q.dv));
  if (dn_old()) s.dnws0 = alloc<float>(tk::dn_ws_floats(c.Bmax, c.Lmax, Q.lh, Q.dk, Q.dv));
  long long wsn = 0;
  auto need = [&](long long rows, long long N, long long K) { wsn = std::max(wsn, (rows + DWCH - 1) / DWCH * N * K); };
  for (int i = 0; i < Q.layers; ++i) { need(R, s.W0, H); need(R, 2 * Q.I, H); need(R, H, Q.I); need(R, H, s.OW); }
  need(Rv, 3 * c.vD, c.vD); need(Rv, c.vMLP, c.vD); need(Rv, c.vD, c.vMLP); need(Rv, c.vD, c.vK); need(Rv, H, c.vD);
  for (int g = 0; g < N_VG; ++g) need((long long)c.Bmax * kVGrp[g].n_tok, g_w1[g].N, g_w1[g].K);
  need((long long)c.Bmax * 16, H, c.obj_hid);
  need(RA, Q.qg() + 2 * Q.kvw(), c.De); need(RA, c.De, Q.nq * Q.hd); need(RA, 2 * c.Ie, c.De); need(RA, c.De, c.Ie); need(RA, c.De, c.kpad()); need(RA, c.A, c.De);
  need(c.Mtmax, c.vocab_chunk, H);
  s.ws = alloc<float>(wsn);
  s.dRb = alloc<uint16_t>((size_t)R * H); s.dGU = alloc<uint16_t>((size_t)R * 2 * Q.I); s.dT0b = alloc<uint16_t>((size_t)R * s.W0);
  // 영상
  for (int l = 0; l <= c.vL; ++l) s.vX.push_back(alloc<float>((size_t)Rv * c.vD));
  s.vQKV = alloc<float>((size_t)Rv * 3 * c.vD); s.vO = alloc<float>((size_t)Rv * c.vD); s.vH1 = alloc<float>((size_t)Rv * c.vMLP);
  s.vlse = alloc<float>((size_t)Rv * c.vHeads); s.vXm = alloc<float>((size_t)Rv * c.vD); s.vOut = alloc<float>((size_t)Rv * H);
  s.vdR = alloc<float>((size_t)Rv * c.vD); s.vdA = alloc<float>((size_t)Rv * c.vD); s.vdQKV = alloc<float>((size_t)Rv * 3 * c.vD);
  s.vdO = alloc<float>((size_t)Rv * c.vD); s.vdH = alloc<float>((size_t)Rv * c.vMLP); s.vDd = alloc<float>((size_t)Rv * c.vHeads); s.vdOut = alloc<float>((size_t)Rv * H);
  s.vA1 = alloc<uint16_t>((size_t)Rv * c.vD); s.vAo = alloc<uint16_t>((size_t)Rv * c.vD); s.vA2 = alloc<uint16_t>((size_t)Rv * c.vD);
  s.vAg = alloc<uint16_t>((size_t)Rv * c.vMLP); s.vTok = alloc<uint16_t>((size_t)Rv * c.vD);
  s.vdb = alloc<uint16_t>((size_t)Rv * std::max(std::max(3 * c.vD, c.vMLP), H));
  // 묶음
  for (int g = 0; g < N_VG; ++g) {
    s.gOut[g] = alloc<float>((size_t)c.Bmax * kVGrp[g].n_tok * H);
    s.dgOut[g] = alloc<float>((size_t)c.Bmax * kVGrp[g].n_tok * H);
  }
  s.gH1 = alloc<float>((size_t)c.Bmax * 16 * c.obj_hid); s.dgH = alloc<float>((size_t)c.Bmax * 16 * c.obj_hid);
  s.gH = alloc<uint16_t>((size_t)c.Bmax * 16 * c.obj_hid); s.gdb = alloc<uint16_t>((size_t)c.Bmax * 16 * std::max(c.obj_hid, H));
  // 글
  s.hnT = alloc<uint16_t>((size_t)c.Mtmax * H); s.dLg = alloc<uint16_t>((size_t)c.Mtmax * Q.vocab);
  s.lg = alloc<float>((size_t)c.Mtmax * c.vocab_chunk); s.dHnT = alloc<float>((size_t)c.Mtmax * H);
  s.rm = alloc<float>(c.Mtmax); s.rs = alloc<float>(c.Mtmax); s.tl = alloc<float>(c.Mtmax);
  s.dEc = alloc<float>((size_t)c.vocab_chunk * H); s.lpart = alloc<float>(c.Mtmax + RA + 64);
  // 전문가
  const int eW0 = Q.qg() + 2 * Q.kvw();
  s.ain = alloc<uint16_t>((size_t)RA * c.kpad());
  s.tau = alloc<float>(c.Bmax); s.eps = alloc<float>((size_t)RA * c.A); s.u = alloc<float>((size_t)RA * c.A);
  for (int f = 0; f <= nf; ++f) s.eX.push_back(alloc<float>((size_t)RA * c.De));
  s.eA1 = alloc<uint16_t>((size_t)RA * c.De); s.eAg = alloc<uint16_t>((size_t)RA * Q.nq * Q.hd); s.eA2 = alloc<uint16_t>((size_t)RA * c.De);
  s.eHh = alloc<uint16_t>((size_t)RA * c.Ie); s.Ao = alloc<uint16_t>((size_t)RA * c.De); s.dz = alloc<uint16_t>((size_t)RA * c.A);
  s.eT = alloc<float>((size_t)RA * eW0); s.eQ = alloc<float>((size_t)RA * Q.nq * Q.hd); s.eK = alloc<float>((size_t)RA * Q.kvw()); s.eV = alloc<float>((size_t)RA * Q.kvw());
  s.eO = alloc<float>((size_t)RA * Q.nq * Q.hd); s.else_ = alloc<float>((size_t)RA * Q.nq); s.eXm = alloc<float>((size_t)RA * c.De); s.eGU = alloc<float>((size_t)RA * 2 * c.Ie);
  s.edR = alloc<float>((size_t)RA * c.De); s.edA = alloc<float>((size_t)RA * c.De); s.edHh = alloc<float>((size_t)RA * c.Ie); s.edAg = alloc<float>((size_t)RA * Q.nq * Q.hd);
  s.edO = alloc<float>((size_t)RA * Q.nq * Q.hd); s.edT0 = alloc<float>((size_t)RA * eW0); s.edQ = alloc<float>((size_t)RA * Q.nq * Q.hd);
  s.edK = alloc<float>((size_t)RA * Q.kvw()); s.edV = alloc<float>((size_t)RA * Q.kvw()); s.eDd = alloc<float>((size_t)RA * Q.nq);
  s.edRb = alloc<uint16_t>((size_t)RA * c.De); s.edGU = alloc<uint16_t>((size_t)RA * 2 * c.Ie); s.edT0b = alloc<uint16_t>((size_t)RA * eW0);
  vel = alloc<float>((size_t)RA * c.A);
  loss = alloc<float>(8);
  // 1 칸(편향) — 행동 입력 줄
  {
    std::vector<uint16_t> h((size_t)RA * c.kpad(), 0);
    for (int r = 0; r < RA; ++r) h[(size_t)r * c.kpad() + c.A + TEMB] = 0x3f80;
    MCK(cudaMemcpy(s.ain, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  }
  MCK(cudaDeviceSynchronize());
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// 작은 커널
__global__ void addpos_k(float* X, long long n, int T, int D, const float* pos) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long r = i / D;
  X[i] = X[i] + pos[(r % T) * D + i % D];
}
struct GPtr { const float* p[N_VG]; long long typ[N_VG]; int ntok[N_VG]; };
__global__ void assemble_k(const int* src, long long R, int H, const uint16_t* E, const float* vOut, GPtr g, const float* V, float* X) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= R * H) return;
  const long long r = q / H;
  const int c = (int)(q % H), code = src[r], kind = (code >> 28) & 15, idx = code & 0x0fffffff;
  float v = 0.f;
  if (kind == SK_TXT) v = bf2f(E[(long long)idx * H + c]);
  else if (kind == SK_IMG) v = vOut[(long long)idx * H + c];
  else if (kind >= SK_GRP) { const int gg = kind - SK_GRP; v = g.p[gg][(long long)idx * H + c] + V[g.typ[gg] + c]; }
  X[q] = v;
}
struct GPtrW { float* p[N_VG]; };
__global__ void scatter_k(const int* src, long long R, int H, const float* dX, float* dvOut, GPtrW g) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= R * H) return;
  const long long r = q / H;
  const int c = (int)(q % H), code = src[r], kind = (code >> 28) & 15, idx = code & 0x0fffffff;
  if (kind == SK_IMG) dvOut[(long long)idx * H + c] = dX[q];
  else if (kind >= SK_GRP) g.p[kind - SK_GRP][(long long)idx * H + c] = dX[q];
}
// 임베딩 행 기울기(입력 쪽): 스레드 = 열, 자리를 차례로(같은 토큰 여러 번이어도 고정 순서)
__global__ void emb_sparse_k(const int* src, long long R, int H, const float* dX, int v0, int vc, float* dEc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  for (long long r = 0; r < R; ++r) {
    const int code = src[r], kind = (code >> 28) & 15, id = code & 0x0fffffff;
    if (kind == SK_TXT && id >= v0 && id < v0 + vc) dEc[(long long)(id - v0) * H + c] += dX[r * H + c];
  }
}
__global__ void gather_rows_k(const float* X, const int* rows, int M, int H, uint16_t* out) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * H) return;
  const int m = (int)(q / H), c = (int)(q % H);
  out[q] = f2bf(X[(long long)rows[m] * H + c]);
}
__global__ void scatter_rows_k(const float* dT, const int* rows, int M, int H, float* dX) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * H) return;
  const int m = (int)(q / H), c = (int)(q % H);
  dX[(long long)rows[m] * H + c] = dT[q];
}
// 어휘 조각 통계: 블록 = 행
__global__ void ce_stats_k(const float* lg, int M, int vc, int v0, const int* tid, float* rm, float* rs, float* tl, int first) {
  __shared__ float sh[256];
  const int m = blockIdx.x;
  const float* l = lg + (long long)m * vc;
  float mx = -INFINITY;
  for (int v = threadIdx.x; v < vc; v += 256) mx = fmaxf(mx, l[v]);
  sh[threadIdx.x] = mx;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
  const float cm = sh[0];
  __syncthreads();
  float se = 0.f;
  for (int v = threadIdx.x; v < vc; v += 256) se = se + expf(l[v] - cm);
  sh[threadIdx.x] = se;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] = sh[threadIdx.x] + sh[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) {
    const float cs = sh[0];
    if (first) { rm[m] = cm; rs[m] = cs; }
    else {
      const float nm = fmaxf(rm[m], cm);
      rs[m] = rs[m] * expf(rm[m] - nm) + cs * expf(cm - nm);
      rm[m] = nm;
    }
    const int t = tid[m];
    if (t >= v0 && t < v0 + vc) tl[m] = l[t - v0];
  }
}
__global__ void ce_grad_k(const float* lg, int M, int vc, int v0, int V, const int* tid, const float* w, const float* rm, const float* rs, float lam, int bug,
                          uint16_t* dLg) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * vc) return;
  const int m = (int)(q / vc), v = (int)(q % vc);
  const float p = bug == 4 ? 0.f : expf(lg[q] - rm[m]) / rs[m];
  const float g = lam * w[m] * (p - (v0 + v == tid[m] ? 1.f : 0.f));
  dLg[(long long)m * V + v0 + v] = f2bf(g);
}
__global__ void ce_loss_k(int M, const float* w, const float* rm, const float* rs, const float* tl, float* out) {
  if (threadIdx.x != 0) return;
  float s = 0.f;
  for (int m = 0; m < M; ++m) s = s + w[m] * (rm[m] + logf(rs[m]) - tl[m]);
  out[0] = s;
}
// flow matching(BC tf.cu 와 같은 식)
__device__ __forceinline__ float u01(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.f / 16777216.f); }
__device__ __forceinline__ float gaussf(uint64_t h) {
  const float a = u01(h), b = u01(net::mix64(h ^ 0x5bd1e995ull));
  return sqrtf(-2.f * logf(a)) * cosf(6.283185307179586f * b);
}
__device__ void temb(float tau, float* dst) {
  for (int i = 0; i < TEMB / 2; ++i) {
    const float period = 0.004f * powf(1000.f, (float)i / (float)(TEMB / 2 - 1));
    const float a = tau * (6.283185307179586f / period);
    dst[i] = sinf(a);
    dst[TEMB / 2 + i] = cosf(a);
  }
}
__global__ void flow_in_k(const float* chunk, const float* cmask, const long long* iter, uint64_t seed, int B, int H, int A, int KA, uint16_t* ain, float* tau,
                          float* eps, float* u) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const uint64_t it = (uint64_t)iter[0];
  const float tt = u01(net::hash4(seed ^ 0xF10Full, it, (uint64_t)b, 0x7a75ull));
  tau[b] = tt;
  float te[TEMB];
  temb(tt, te);
  for (int h = 0; h < H; ++h) {
    uint16_t* row = ain + ((long long)b * H + h) * KA;
    const bool on = cmask[(long long)b * H + h] > 0.f;
    for (int k = 0; k < A; ++k) {
      const long long q = ((long long)b * H + h) * A + k;
      const float a = on ? chunk[q] : 0.f;
      const float e = gaussf(net::hash4(seed ^ 0xF10Full, it, (uint64_t)b, 0x1000ull + (uint64_t)(h * A + k)));
      eps[q] = e;
      row[k] = f2bf(tt * e + (1.f - tt) * a);
      u[q] = e - a;
    }
    for (int k = 0; k < TEMB; ++k) row[A + k] = f2bf(te[k]);
  }
}
__global__ void fm_loss_k(const float* vel, const float* u, const float* cmask, const uint32_t* adim, int B, int Hc, int A, float lam, uint16_t* dz, float* part) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= B * Hc) return;
  const uint32_t am = adim[0];
  const float m = cmask[r], inv = 1.f / (float)B;
  float q = 0.f;
  for (int k = 0; k < A; ++k) {
    const float on = ((am >> k) & 1u) ? m : 0.f;
    const float dd = vel[(long long)r * A + k] - u[(long long)r * A + k];
    q = q + on * dd * dd;
    dz[(long long)r * A + k] = f2bf(lam * 2.f * on * dd * inv);
  }
  part[r] = q;
}
__global__ void fm_red_k(const float* part, int n, int B, float* out) {
  if (threadIdx.x != 0) return;
  float s = 0.f;
  for (int k = 0; k < n; ++k) s = s + part[k];
  out[0] = s / (float)B;
}
__global__ void loss_tot_k(float* L, float lt, float lf) { if (threadIdx.x == 0) L[0] = lt * L[1] + lf * L[2]; }

// ---------------------------------------------------------------------------------------------------------------------
// 몸통 한 층(학습, 열 시작 0). 중간값은 작업 버퍼에(뒤에서 다시 계산)
static void q_layer_fwd(Model& m, int l, int B, int L, const float* Xin, float* Xout, cudaStream_t st, bool ck = false) {
  Model::WS& s = *m.w;
  const QCfg& c = m.q.c;
  const auto& Ly = m.q.lay.l[l];
  const int R = B * L, H = c.H;
  const uint16_t* W = m.q.Wb;
  const float* P = m.q.Pv;
  qk::rmsnorm(Xin, R, H, P + Ly.ln1, true, c.eps, s.A1, nullptr, nullptr, st);
  if (c.full[l]) {
    int f = 0;
    for (int i = 0; i < l; ++i) f += c.full[i];
    tk::mm(s.A1, H, R, W + Ly.wqkv.off, Ly.wqkv.N, H, s.T0, Ly.wqkv.N, false, st);
    qk::full_prep(s.T0, Ly.wqkv.N, R, B, L, 0, c.nq, c.nkv, c.hd, c.rot, c.theta, c.eps, P + Ly.qn, P + Ly.kn, false, s.QK, s.Kf[f], s.Vf[f], L, st);
    tk::AttP a;
    a.Q = s.QK; a.ldq = c.nq * c.hd; a.K1 = s.Kf[f]; a.V1 = s.Vf[f]; a.ldk1 = c.kvw(); a.L1 = L; a.n1c = L; a.causal = 1;
    a.B = B; a.n = L; a.nq = c.nq; a.nkv = c.nkv; a.hd = c.hd; a.scale = 1.f / std::sqrt((float)c.hd); a.O = s.O; a.ldo = c.nq * c.hd; a.lse = s.lse;
    tk::att_fwd(a, st);
    qk::gate(s.O, s.T0, Ly.wqkv.N, R, c.nq, c.hd, s.Ag, st);
    MCK(cudaMemcpyAsync(s.Xmid, Xin, sizeof(float) * (size_t)R * H, cudaMemcpyDeviceToDevice, st));
    tk::mm(s.Ag, c.nq * c.hd, R, W + Ly.wo.off, H, c.nq * c.hd, s.Xmid, H, true, st);
  } else {
    const int li = c.lin_in(), la = c.lin_all();
    tk::mm(s.A1, H, R, W + Ly.win.off, la, H, s.T0, la, false, st);
    qk::conv_silu(s.T0, la, L, B, L, 0, li, c.conv, P + Ly.convw, s.T1, st);
    float* Qn = s.QK;
    float* Kn = s.QK + (size_t)R * c.lh * c.dk;
    qk::lin_prep(s.T1, li, s.T0, la, li + c.lh * c.dv, R, c.lh, c.dk, P + Ly.alog, P + Ly.dtb, false, Qn, Kn, s.Gb, s.Bb, st);
    if (dn_old()) qk::deltanet(Qn, Kn, s.T1 + 2 * c.lh * c.dk, li, s.Gb, s.Bb, B, L, c.lh, c.dk, c.dv, nullptr, nullptr, false, s.O, st, ck ? s.dnws0 : nullptr);
    else tk::dnc_fwd(Qn, Kn, s.T1 + 2 * c.lh * c.dk, li, s.Gb, s.Bb, B, L, c.lh, c.dk, c.dv, nullptr, nullptr, false, s.O, s.dnws, ck, st);
    qk::gnorm(s.O, s.T0 + li, la, R, c.lh, c.dv, P + Ly.gnw, c.eps, s.Ag, st);
    MCK(cudaMemcpyAsync(s.Xmid, Xin, sizeof(float) * (size_t)R * H, cudaMemcpyDeviceToDevice, st));
    tk::mm(s.Ag, c.lh * c.dv, R, W + Ly.wout.off, H, c.lh * c.dv, s.Xmid, H, true, st);
  }
  qk::rmsnorm(s.Xmid, R, H, P + Ly.ln2, true, c.eps, s.A2, nullptr, nullptr, st);
  tk::mm(s.A2, H, R, W + Ly.wgu.off, 2 * c.I, H, s.GU, 2 * c.I, false, st);
  qk::swiglu(s.GU, R, c.I, s.Hh, st);
  MCK(cudaMemcpyAsync(Xout, s.Xmid, sizeof(float) * (size_t)R * H, cudaMemcpyDeviceToDevice, st));
  tk::mm(s.Hh, c.I, R, W + Ly.wdn.off, H, c.I, Xout, H, true, st);
}
// 뒤: s.dR = 층 출력 기울기 → 층 입력 기울기(제자리). 앞 중간값은 q_layer_fwd 를 다시 불러 만든다
static void q_layer_bwd(Model& m, int l, int B, int L, cudaStream_t st) {
  Model::WS& s = *m.w;
  const QCfg& c = m.q.c;
  const auto& Ly = m.q.lay.l[l];
  const int R = B * L, H = c.H;
  const uint16_t* W = m.q.Wb;
  const float* P = m.q.Pv;
  uint16_t* GW = m.qp.GW;
  float* GV = m.qp.GV;
  // 다시 계산(Xout 은 버림 — dA 를 임시로)
  q_layer_fwd(m, l, B, L, s.Xs[l], s.dA, st, true);   // 검문점도 씀
  // MLP
  tk::f2bf(s.dR, (long long)R * H, s.dRb, st);
  tk::mm_dw(s.dRb, H, s.Hh, c.I, R, H, c.I, s.ws, DWCH, GW + Ly.wdn.off, nullptr, st);
  tk::mm_dx(s.dRb, H, R, W + Ly.wdn.off, H, c.I, s.dHh, c.I, false, st);
  tk::swiglu_bwd(s.GU, s.dHh, R, c.I, s.dGU, m.bug == 1 ? 1 : 0, st);
  tk::mm_dw(s.dGU, 2 * c.I, s.A2, H, R, 2 * c.I, H, s.ws, DWCH, GW + Ly.wgu.off, nullptr, st);
  tk::mm_dx(s.dGU, 2 * c.I, R, W + Ly.wgu.off, 2 * c.I, H, s.dA, H, false, st);
  tk::rms_bwd(s.dA, H, s.Xmid, H, R, 1, H, P + Ly.ln2, true, c.eps, s.dR, H, true, s.wt, 0, st);
  tk::colsum(s.wt, R, H, H, s.part, GV + Ly.ln2, false, st);
  // 섞개
  tk::f2bf(s.dR, (long long)R * H, s.dRb, st);
  if (c.full[l]) {
    int f = 0;
    for (int i = 0; i < l; ++i) f += c.full[i];
    const int ldT = Ly.wqkv.N, QW = c.nq * c.hd;
    tk::mm_dw(s.dRb, H, s.Ag, QW, R, H, QW, s.ws, DWCH, GW + Ly.wo.off, nullptr, st);
    tk::mm_dx(s.dRb, H, R, W + Ly.wo.off, H, QW, s.dAg, QW, false, st);
    tk::gate_bwd(s.dAg, s.O, s.T0, ldT, R, c.nq, c.hd, s.dO, s.dT0, st);
    tk::AttP a;
    a.Q = s.QK; a.ldq = QW; a.K1 = s.Kf[f]; a.V1 = s.Vf[f]; a.ldk1 = c.kvw(); a.L1 = L; a.n1c = L; a.causal = 1;
    a.B = B; a.n = L; a.nq = c.nq; a.nkv = c.nkv; a.hd = c.hd; a.scale = 1.f / std::sqrt((float)c.hd); a.O = s.O; a.ldo = QW; a.lse = s.lse;
    a.dO = s.dO; a.lddo = QW; a.Dd = s.Dd; a.dQ = s.dQK; a.lddq = QW; a.dK1 = s.dK; a.dV1 = s.dV; a.bug = m.bug == 5 ? 1 : 0;
    tk::att_bwd(a, st);
    if (!m.c.ki) {
      tk::add(s.dK, s.dKx[f], (long long)R * c.kvw(), st);
      tk::add(s.dV, s.dVx[f], (long long)R * c.kvw(), st);
    }
    float* qnt = s.wt;
    float* knt = s.wt + (size_t)R * c.nq * c.hd;
    tk::full_prep_bwd(s.T0, ldT, R, L, 0, c.nq, c.nkv, c.hd, c.rot, c.theta, c.eps, P + Ly.qn, P + Ly.kn, s.dQK, s.dK, s.dV, L, s.dT0, qnt, knt,
                      m.bug == 2 ? 2 : 0, st);
    tk::colsum(qnt, R * c.nq, c.hd, c.hd, s.part, GV + Ly.qn, false, st);
    tk::colsum(knt, R * c.nkv, c.hd, c.hd, s.part, GV + Ly.kn, false, st);
    tk::f2bf(s.dT0, (long long)R * ldT, s.dT0b, st);
    tk::mm_dw(s.dT0b, ldT, s.A1, H, R, ldT, H, s.ws, DWCH, GW + Ly.wqkv.off, nullptr, st);
    tk::mm_dx(s.dT0b, ldT, R, W + Ly.wqkv.off, ldT, H, s.dA, H, false, st);
  } else {
    const int li = c.lin_in(), la = c.lin_all(), VW = c.lh * c.dv;
    tk::mm_dw(s.dRb, H, s.Ag, VW, R, H, VW, s.ws, DWCH, GW + Ly.wout.off, nullptr, st);
    tk::mm_dx(s.dRb, H, R, W + Ly.wout.off, H, VW, s.dAg, VW, false, st);
    tk::gnorm_bwd(s.O, s.T0 + li, la, R, c.lh, c.dv, P + Ly.gnw, c.eps, s.dAg, s.dO, s.dT0 + li, la, s.wt, st);
    tk::colsum(s.wt, R * c.lh, c.dv, c.dv, s.part, GV + Ly.gnw, false, st);
    float* Qn = s.QK;
    float* Kn = s.QK + (size_t)R * c.lh * c.dk;
    float* dQn = s.dQK;
    float* dKn = s.dQK + (size_t)R * c.lh * c.dk;
    if (dn_old())
      tk::deltanet_bwd(Qn, Kn, s.T1 + 2 * c.lh * c.dk, li, s.Gb, s.Bb, s.dO, B, L, c.lh, c.dk, c.dv, false, true, s.dnws0, dQn, dKn, s.dT1 + 2 * c.lh * c.dk, li,
                       s.dG, s.dBt, m.bug == 3 ? 3 : 0, st);
    else tk::dnc_bwd(s.T1 + 2 * c.lh * c.dk, li, s.dO, B, L, c.lh, c.dk, c.dv, false, s.dnws, dQn, dKn, s.dT1 + 2 * c.lh * c.dk, li, s.dG, s.dBt, m.bug == 3 ? 3 : 0, st);
    float* alt = s.wt;
    float* dtt = s.wt + (size_t)R * c.lh;
    tk::lin_prep_bwd(s.T1, li, s.T0, la, li + VW, R, c.lh, c.dk, P + Ly.alog, P + Ly.dtb, dQn, dKn, s.dG, s.dBt, s.dT1, s.dT0, alt, dtt, st);
    tk::colsum(alt, R, c.lh, c.lh, s.part, GV + Ly.alog, false, st);
    tk::colsum(dtt, R, c.lh, c.lh, s.part, GV + Ly.dtb, false, st);
    tk::conv_bwd(s.T0, la, B, L, li, c.conv, P + Ly.convw, s.dT1, s.dp, s.dT0, st);
    tk::convw_grad(s.T0, la, s.dp, B, L, li, c.conv, s.part, GV + Ly.convw, st);
    tk::f2bf(s.dT0, (long long)R * la, s.dT0b, st);
    tk::mm_dw(s.dT0b, la, s.A1, H, R, la, H, s.ws, DWCH, GW + Ly.win.off, nullptr, st);
    tk::mm_dx(s.dT0b, la, R, W + Ly.win.off, la, H, s.dA, H, false, st);
  }
  tk::rms_bwd(s.dA, H, s.Xs[l], H, R, 1, H, P + Ly.ln1, true, c.eps, s.dR, H, true, s.wt, 0, st);
  tk::colsum(s.wt, R, H, H, s.part, GV + Ly.ln1, false, st);
}

// ---- 영상 블록 ----
static void v_block_fwd(Model& m, int l, int Rv, const float* Xin, float* Xout, cudaStream_t st) {
  Model::WS& s = *m.w;
  const VCfg& c = m.c;
  const auto& b = m.vb[l];
  const uint16_t* W = m.ap.W;
  const float* P = m.ap.V;
  const int D = c.vD;
  tk::ln_fwd(Xin, Rv, D, P + b.ln1g, P + b.ln1b, c.vEps, s.vA1, nullptr, st);
  tk::mm(s.vA1, D, Rv, W + b.qkv.off, 3 * D, D, s.vQKV, 3 * D, false, st);
  tk::bias_add(s.vQKV, Rv, 3 * D, 3 * D, P + b.qkvb, st);
  tk::AttP a;
  a.Q = s.vQKV; a.ldq = 3 * D; a.K1 = s.vQKV + D; a.V1 = s.vQKV + 2 * D; a.ldk1 = 3 * D; a.L1 = c.vT; a.n1c = c.vT;
  a.B = Rv / c.vT; a.n = c.vT; a.nq = c.vHeads; a.nkv = c.vHeads; a.hd = D / c.vHeads; a.scale = 1.f / std::sqrt((float)(D / c.vHeads));
  a.O = s.vO; a.ldo = D; a.lse = s.vlse;
  tk::att_fwd(a, st);
  tk::f2bf(s.vO, (long long)Rv * D, s.vAo, st);
  MCK(cudaMemcpyAsync(s.vXm, Xin, sizeof(float) * (size_t)Rv * D, cudaMemcpyDeviceToDevice, st));
  tk::mm(s.vAo, D, Rv, W + b.proj.off, D, D, s.vXm, D, true, st);
  tk::bias_add(s.vXm, Rv, D, D, P + b.projb, st);
  tk::ln_fwd(s.vXm, Rv, D, P + b.ln2g, P + b.ln2b, c.vEps, s.vA2, nullptr, st);
  tk::mm(s.vA2, D, Rv, W + b.fc1.off, c.vMLP, D, s.vH1, c.vMLP, false, st);
  tk::bias_add(s.vH1, Rv, c.vMLP, c.vMLP, P + b.fc1b, st);
  tk::gelu_fwd(s.vH1, (long long)Rv * c.vMLP, s.vAg, st);
  MCK(cudaMemcpyAsync(Xout, s.vXm, sizeof(float) * (size_t)Rv * D, cudaMemcpyDeviceToDevice, st));
  tk::mm(s.vAg, c.vMLP, Rv, W + b.fc2.off, D, c.vMLP, Xout, D, true, st);
  tk::bias_add(Xout, Rv, D, D, P + b.fc2b, st);
}
static void v_block_bwd(Model& m, int l, int Rv, cudaStream_t st) {
  Model::WS& s = *m.w;
  const VCfg& c = m.c;
  const auto& b = m.vb[l];
  const uint16_t* W = m.ap.W;
  const float* P = m.ap.V;
  uint16_t* GW = m.ap.GW;
  float* GV = m.ap.GV;
  const int D = c.vD, M = c.vMLP;
  v_block_fwd(m, l, Rv, s.vX[l], s.vdA, st);   // 다시 계산(출력은 버림)
  tk::f2bf(s.vdR, (long long)Rv * D, s.vdb, st);
  tk::mm_dw(s.vdb, D, s.vAg, M, Rv, D, M, s.ws, DWCH, GW + b.fc2.off, nullptr, st);
  tk::colsum(s.vdR, Rv, D, D, s.part, GV + b.fc2b, false, st);
  tk::mm_dx(s.vdb, D, Rv, W + b.fc2.off, D, M, s.vdH, M, false, st);
  tk::gelu_bwd(s.vdH, s.vH1, (long long)Rv * M, s.vdb, s.vdH, st);
  tk::colsum(s.vdH, Rv, M, M, s.part, GV + b.fc1b, false, st);
  tk::mm_dw(s.vdb, M, s.vA2, D, Rv, M, D, s.ws, DWCH, GW + b.fc1.off, nullptr, st);
  tk::mm_dx(s.vdb, M, Rv, W + b.fc1.off, M, D, s.vdA, D, false, st);
  tk::ln_bwd(s.vdA, s.vXm, Rv, D, P + b.ln2g, c.vEps, s.vdR, s.wt, st);
  tk::colsum(s.wt, Rv, D, D, s.part, GV + b.ln2g, false, st);
  tk::colsum(s.vdA, Rv, D, D, s.part, GV + b.ln2b, false, st);
  tk::f2bf(s.vdR, (long long)Rv * D, s.vdb, st);
  tk::mm_dw(s.vdb, D, s.vAo, D, Rv, D, D, s.ws, DWCH, GW + b.proj.off, nullptr, st);
  tk::colsum(s.vdR, Rv, D, D, s.part, GV + b.projb, false, st);
  tk::mm_dx(s.vdb, D, Rv, W + b.proj.off, D, D, s.vdO, D, false, st);
  tk::AttP a;
  a.Q = s.vQKV; a.ldq = 3 * D; a.K1 = s.vQKV + D; a.V1 = s.vQKV + 2 * D; a.ldk1 = 3 * D; a.L1 = c.vT; a.n1c = c.vT;
  a.B = Rv / c.vT; a.n = c.vT; a.nq = c.vHeads; a.nkv = c.vHeads; a.hd = D / c.vHeads; a.scale = 1.f / std::sqrt((float)(D / c.vHeads));
  a.O = s.vO; a.ldo = D; a.lse = s.vlse; a.dO = s.vdO; a.lddo = D; a.Dd = s.vDd; a.dQ = s.vdQKV; a.lddq = 3 * D; a.dK1 = s.vdQKV + D; a.dV1 = s.vdQKV + 2 * D;
  a.bug = m.bug == 5 ? 1 : 0;
  tk::att_bwd(a, st);
  tk::colsum(s.vdQKV, Rv, 3 * D, 3 * D, s.part, GV + b.qkvb, false, st);
  tk::f2bf(s.vdQKV, (long long)Rv * 3 * D, s.vdb, st);
  tk::mm_dw(s.vdb, 3 * D, s.vA1, D, Rv, 3 * D, D, s.ws, DWCH, GW + b.qkv.off, nullptr, st);
  tk::mm_dx(s.vdb, 3 * D, Rv, W + b.qkv.off, 3 * D, D, s.vdA, D, false, st);
  tk::ln_bwd(s.vdA, s.vX[l], Rv, D, P + b.ln1g, c.vEps, s.vdR, s.wt, st);
  tk::colsum(s.wt, Rv, D, D, s.part, GV + b.ln1g, false, st);
  tk::colsum(s.vdA, Rv, D, D, s.part, GV + b.ln1b, false, st);
}

// ---- 전문가 블록 ----
static void e_block_fwd(Model& m, int f, const VBatch& bt, const float* Xin, float* Xout, cudaStream_t st) {
  Model::WS& s = *m.w;
  const VCfg& c = m.c;
  const QCfg& Q = m.q.c;
  const auto& e = m.eb[f];
  const uint16_t* W = m.ap.W;
  const float* P = m.ap.V;
  const int B = bt.B, RA = B * c.Hc, De = c.De, ldT = e.qkv.N, QW = Q.nq * Q.hd;
  qk::rmsnorm(Xin, RA, De, P + e.ln1, true, Q.eps, s.eA1, nullptr, nullptr, st);
  tk::mm(s.eA1, De, RA, W + e.qkv.off, ldT, De, s.eT, ldT, false, st);
  qk::full_prep(s.eT, ldT, RA, B, c.Hc, 0, Q.nq, Q.nkv, Q.hd, Q.rot, Q.theta, Q.eps, P + e.qn, P + e.kn, false, s.eQ, s.eK, s.eV, c.Hc, st, bt.plen);
  tk::AttP a;
  a.Q = s.eQ; a.ldq = QW; a.K1 = s.Kf[f]; a.V1 = s.Vf[f]; a.ldk1 = Q.kvw(); a.L1 = bt.L; a.len1 = bt.plen;
  a.K2 = s.eK; a.V2 = s.eV; a.ldk2 = Q.kvw(); a.n2 = c.Hc;
  a.B = B; a.n = c.Hc; a.nq = Q.nq; a.nkv = Q.nkv; a.hd = Q.hd; a.scale = 1.f / std::sqrt((float)Q.hd); a.O = s.eO; a.ldo = QW; a.lse = s.else_;
  tk::att_fwd(a, st);
  qk::gate(s.eO, s.eT, ldT, RA, Q.nq, Q.hd, s.eAg, st);
  MCK(cudaMemcpyAsync(s.eXm, Xin, sizeof(float) * (size_t)RA * De, cudaMemcpyDeviceToDevice, st));
  tk::mm(s.eAg, QW, RA, W + e.o.off, De, QW, s.eXm, De, true, st);
  qk::rmsnorm(s.eXm, RA, De, P + e.ln2, true, Q.eps, s.eA2, nullptr, nullptr, st);
  tk::mm(s.eA2, De, RA, W + e.gu.off, 2 * c.Ie, De, s.eGU, 2 * c.Ie, false, st);
  qk::swiglu(s.eGU, RA, c.Ie, s.eHh, st);
  MCK(cudaMemcpyAsync(Xout, s.eXm, sizeof(float) * (size_t)RA * De, cudaMemcpyDeviceToDevice, st));
  tk::mm(s.eHh, c.Ie, RA, W + e.dn.off, De, c.Ie, Xout, De, true, st);
}
static void e_block_bwd(Model& m, int f, const VBatch& bt, cudaStream_t st) {
  Model::WS& s = *m.w;
  const VCfg& c = m.c;
  const QCfg& Q = m.q.c;
  const auto& e = m.eb[f];
  const uint16_t* W = m.ap.W;
  const float* P = m.ap.V;
  uint16_t* GW = m.ap.GW;
  float* GV = m.ap.GV;
  const int B = bt.B, RA = B * c.Hc, De = c.De, ldT = e.qkv.N, QW = Q.nq * Q.hd;
  e_block_fwd(m, f, bt, s.eX[f], s.edA, st);
  tk::f2bf(s.edR, (long long)RA * De, s.edRb, st);
  tk::mm_dw(s.edRb, De, s.eHh, c.Ie, RA, De, c.Ie, s.ws, DWCH, GW + e.dn.off, nullptr, st);
  tk::mm_dx(s.edRb, De, RA, W + e.dn.off, De, c.Ie, s.edHh, c.Ie, false, st);
  tk::swiglu_bwd(s.eGU, s.edHh, RA, c.Ie, s.edGU, m.bug == 1 ? 1 : 0, st);
  tk::mm_dw(s.edGU, 2 * c.Ie, s.eA2, De, RA, 2 * c.Ie, De, s.ws, DWCH, GW + e.gu.off, nullptr, st);
  tk::mm_dx(s.edGU, 2 * c.Ie, RA, W + e.gu.off, 2 * c.Ie, De, s.edA, De, false, st);
  tk::rms_bwd(s.edA, De, s.eXm, De, RA, 1, De, P + e.ln2, true, Q.eps, s.edR, De, true, s.wt, 0, st);
  tk::colsum(s.wt, RA, De, De, s.part, GV + e.ln2, false, st);
  tk::f2bf(s.edR, (long long)RA * De, s.edRb, st);
  tk::mm_dw(s.edRb, De, s.eAg, QW, RA, De, QW, s.ws, DWCH, GW + e.o.off, nullptr, st);
  tk::mm_dx(s.edRb, De, RA, W + e.o.off, De, QW, s.edAg, QW, false, st);
  tk::gate_bwd(s.edAg, s.eO, s.eT, ldT, RA, Q.nq, Q.hd, s.edO, s.edT0, st);
  tk::AttP a;
  a.Q = s.eQ; a.ldq = QW; a.K1 = s.Kf[f]; a.V1 = s.Vf[f]; a.ldk1 = Q.kvw(); a.L1 = bt.L; a.len1 = bt.plen;
  a.K2 = s.eK; a.V2 = s.eV; a.ldk2 = Q.kvw(); a.n2 = c.Hc;
  a.B = B; a.n = c.Hc; a.nq = Q.nq; a.nkv = Q.nkv; a.hd = Q.hd; a.scale = 1.f / std::sqrt((float)Q.hd); a.O = s.eO; a.ldo = QW; a.lse = s.else_;
  a.dO = s.edO; a.lddo = QW; a.Dd = s.eDd; a.dQ = s.edQ; a.lddq = QW; a.dK2 = s.edK; a.dV2 = s.edV; a.bug = m.bug == 5 ? 1 : 0;
  if (!c.ki) { a.dK1 = s.dKx[f]; a.dV1 = s.dVx[f]; }
  tk::att_bwd(a, st);
  float* qnt = s.wt;
  float* knt = s.wt + (size_t)RA * QW;
  tk::full_prep_bwd(s.eT, ldT, RA, c.Hc, 0, Q.nq, Q.nkv, Q.hd, Q.rot, Q.theta, Q.eps, P + e.qn, P + e.kn, s.edQ, s.edK, s.edV, c.Hc, s.edT0, qnt, knt,
                    m.bug == 2 ? 2 : 0, st, bt.plen);
  tk::colsum(qnt, RA * Q.nq, Q.hd, Q.hd, s.part, GV + e.qn, false, st);
  tk::colsum(knt, RA * Q.nkv, Q.hd, Q.hd, s.part, GV + e.kn, false, st);
  tk::f2bf(s.edT0, (long long)RA * ldT, s.edT0b, st);
  tk::mm_dw(s.edT0b, ldT, s.eA1, De, RA, ldT, De, s.ws, DWCH, GW + e.qkv.off, nullptr, st);
  tk::mm_dx(s.edT0b, ldT, RA, W + e.qkv.off, ldT, De, s.edA, De, false, st);
  tk::rms_bwd(s.edA, De, s.eX[f], De, RA, 1, De, P + e.ln1, true, Q.eps, s.edR, De, true, s.wt, 0, st);
  tk::colsum(s.wt, RA, De, De, s.part, GV + e.ln1, false, st);
}

// ---------------------------------------------------------------------------------------------------------------------
static void forward_all(Model& m, const VBatch& bt, bool train, cudaStream_t st) {
  Model::WS& s = *m.w;
  const VCfg& c = m.c;
  const QCfg& Q = m.q.c;
  const int B = bt.B, L = bt.L, R = B * L, H = Q.H, Rv = B * c.cams * c.vT, RA = B * c.Hc, nf = s.nf;
  if (B > c.Bmax || L > c.Lmax || bt.Mt > c.Mtmax) { std::fprintf(stderr, "batch too large\n"); std::abort(); }
  const uint16_t* W = m.ap.W;
  const float* P = m.ap.V;
  // 영상 탑
  tk::mm(bt.patches, c.vK, Rv, W + m.v_patch.off, c.vD, c.vK, s.vX[0], c.vD, false, st);
  tk::bias_add(s.vX[0], Rv, c.vD, c.vD, P + m.v_patchb, st);
  addpos_k<<<nb((long long)Rv * c.vD), 256, 0, st>>>(s.vX[0], (long long)Rv * c.vD, c.vT, c.vD, P + m.v_pos);
  MKC();
  for (int l = 0; l < c.vL; ++l) v_block_fwd(m, l, Rv, s.vX[l], s.vX[l + 1], st);
  tk::ln_fwd(s.vX[c.vL], Rv, c.vD, P + m.v_lnfg, P + m.v_lnfb, c.vEps, s.vTok, nullptr, st);
  tk::mm(s.vTok, c.vD, Rv, W + m.v_proj.off, H, c.vD, s.vOut, H, false, st);
  tk::bias_add(s.vOut, Rv, H, H, P + m.v_projb, st);
  // 묶음
  for (int g = 0; g < N_VG; ++g) {
    const int rows = B * kVGrp[g].n_tok;
    if (g == VG_OBJ) {
      tk::mm(bt.grp[g], kVGrp[g].K, rows, W + m.g_w1[g].off, c.obj_hid, kVGrp[g].K, s.gH1, c.obj_hid, false, st);
      tk::gelu_fwd(s.gH1, (long long)rows * c.obj_hid, s.gH, st);
      tk::mm(s.gH, c.obj_hid, rows, W + m.g_w2.off, H, c.obj_hid, s.gOut[g], H, false, st);
      tk::bias_add(s.gOut[g], rows, H, H, P + m.g_b2, st);
    } else {
      tk::mm(bt.grp[g], kVGrp[g].K, rows, W + m.g_w1[g].off, H, kVGrp[g].K, s.gOut[g], H, false, st);
    }
  }
  GPtr gp;
  for (int g = 0; g < N_VG; ++g) { gp.p[g] = s.gOut[g]; gp.typ[g] = m.g_type[g]; gp.ntok[g] = kVGrp[g].n_tok; }
  assemble_k<<<nb((long long)R * H), 256, 0, st>>>(bt.src, R, H, m.q.Wb + m.q.lay.emb.off, s.vOut, gp, P, s.Xs[0]);
  MKC();
  // 몸통
  for (int l = 0; l < Q.layers; ++l) q_layer_fwd(m, l, B, L, s.Xs[l], s.Xs[l + 1], st);
  qk::rmsnorm(s.Xs[Q.layers], R, H, m.q.Pv + m.q.lay.lnf, true, Q.eps, nullptr, s.hn, nullptr, st);
  // 글: 어휘 조각 통계(+ 학습이면 dlogits 저장, dHnT)
  const int Mt = bt.Mt, V = Q.vocab, VC = c.vocab_chunk;
  if (Mt > 0) {
    gather_rows_k<<<nb((long long)Mt * H), 256, 0, st>>>(s.hn, bt.tgt_row, Mt, H, s.hnT);
    MKC();
    for (int pass = 0; pass < (train ? 2 : 1); ++pass) {
      for (int v0 = 0; v0 < V; v0 += VC) {
        const int vc = std::min(VC, V - v0);
        tk::mm(s.hnT, H, Mt, m.q.Wb + m.q.lay.emb.off + (long long)v0 * H, vc, H, s.lg, vc, false, st);
        if (pass == 0) ce_stats_k<<<Mt, 256, 0, st>>>(s.lg, Mt, vc, v0, bt.tgt_id, s.rm, s.rs, s.tl, v0 == 0 ? 1 : 0);
        else {
          ce_grad_k<<<nb((long long)Mt * vc), 256, 0, st>>>(s.lg, Mt, vc, v0, V, bt.tgt_id, bt.tgt_w, s.rm, s.rs, c.lam_txt, m.bug, s.dLg);
          tk::mm_dx(s.dLg + v0, V, Mt, m.q.Wb + m.q.lay.emb.off + (long long)v0 * H, vc, H, s.dHnT, H, v0 > 0, st);
        }
        MKC();
      }
    }
    ce_loss_k<<<1, 32, 0, st>>>(Mt, bt.tgt_w, s.rm, s.rs, s.tl, m.loss + 1);
  } else MCK(cudaMemsetAsync(m.loss + 1, 0, 4, st));
  // 전문가
  flow_in_k<<<nb(B, 64), 64, 0, st>>>(bt.chunk, bt.cmask, bt.iter, c.seed, B, c.Hc, c.A, c.kpad(), s.ain, s.tau, s.eps, s.u);
  tk::mm(s.ain, c.kpad(), RA, W + m.e_in.off, c.De, c.kpad(), s.eX[0], c.De, false, st);
  for (int f = 0; f < nf; ++f) e_block_fwd(m, f, bt, s.eX[f], s.eX[f + 1], st);
  qk::rmsnorm(s.eX[nf], RA, c.De, P + m.e_lnf, true, Q.eps, s.Ao, nullptr, nullptr, st);
  tk::mm(s.Ao, c.De, RA, W + m.e_out.off, c.A, c.De, m.vel, c.A, false, st);
  fm_loss_k<<<nb(RA, 128), 128, 0, st>>>(m.vel, s.u, bt.cmask, bt.adim, B, c.Hc, c.A, c.lam_fm, s.dz, s.lpart);
  fm_red_k<<<1, 32, 0, st>>>(s.lpart, RA, B, m.loss + 2);
  loss_tot_k<<<1, 32, 0, st>>>(m.loss, c.lam_txt, c.lam_fm);
  MKC();
}

void Model::forward_loss(const VBatch& b, cudaStream_t st) { forward_all(*this, b, false, st); }
const uint16_t* Model::flow_ain() const { return w->ain; }
const float* Model::flow_u() const { return w->u; }

void Model::step_grads(const VBatch& bt, cudaStream_t st) {
  if (!c.vis_train) {   // 얼린 영상 탑: 기울기 0(옵티마이저는 Opt::Buf::skip 범위로 건너뜀)
    MCK(cudaMemsetAsync(ap.GW, 0, sizeof(uint16_t) * (size_t)ap.nW, st));
    MCK(cudaMemsetAsync(ap.GV, 0, sizeof(float) * (size_t)ap.nV, st));
  }
  forward_all(*this, bt, true, st);
  WS& s = *w;
  const QCfg& Q = q.c;
  const int B = bt.B, L = bt.L, R = B * L, H = Q.H, Rv = B * c.cams * c.vT, RA = B * c.Hc, nf = s.nf;
  // 전문가 뒤
  tk::mm_dw(s.dz, c.A, s.Ao, c.De, RA, c.A, c.De, s.ws, DWCH, ap.GW + e_out.off, nullptr, st);
  tk::mm_dx(s.dz, c.A, RA, ap.W + e_out.off, c.A, c.De, s.edA, c.De, false, st);
  MCK(cudaMemsetAsync(s.edR, 0, sizeof(float) * (size_t)RA * c.De, st));
  tk::rms_bwd(s.edA, c.De, s.eX[nf], c.De, RA, 1, c.De, ap.V + e_lnf, true, Q.eps, s.edR, c.De, true, s.wt, 0, st);
  tk::colsum(s.wt, RA, c.De, c.De, s.part, ap.GV + e_lnf, false, st);
  for (int f = nf - 1; f >= 0; --f) e_block_bwd(*this, f, bt, st);
  tk::f2bf(s.edR, (long long)RA * c.De, s.edRb, st);
  tk::mm_dw(s.edRb, c.De, s.ain, c.kpad(), RA, c.De, c.kpad(), s.ws, DWCH, ap.GW + e_in.off, nullptr, st);
  // 몸통 뒤: 끝 RMSN
  MCK(cudaMemsetAsync(s.dHn, 0, sizeof(float) * (size_t)R * H, st));
  if (bt.Mt > 0) {
    scatter_rows_k<<<nb((long long)bt.Mt * H), 256, 0, st>>>(s.dHnT, bt.tgt_row, bt.Mt, H, s.dHn);
    MKC();
  }
  MCK(cudaMemsetAsync(s.dR, 0, sizeof(float) * (size_t)R * H, st));
  tk::rms_bwd(s.dHn, H, s.Xs[Q.layers], H, R, 1, H, q.Pv + q.lay.lnf, true, Q.eps, s.dR, H, true, s.wt, 0, st);
  tk::colsum(s.wt, R, H, H, s.part, qp.GV + q.lay.lnf, false, st);
  for (int l = Q.layers - 1; l >= 0; --l) q_layer_bwd(*this, l, B, L, st);
  // 임베딩(= LM 머리): 어휘 조각마다 dlogitsᵀ·hnT + 입력 자리 기울기
  const int V = Q.vocab, VC = c.vocab_chunk;
  if (c.emb_train) {
    for (int v0 = 0; v0 < V; v0 += VC) {
      const int vc = std::min(VC, V - v0);
      MCK(cudaMemsetAsync(s.dEc, 0, sizeof(float) * (size_t)vc * H, st));
      if (bt.Mt > 0) tk::mm_dw(s.dLg + v0, V, s.hnT, H, bt.Mt, vc, H, s.ws, DWCH, nullptr, s.dEc, st);
      emb_sparse_k<<<nb(H, 128), 128, 0, st>>>(bt.src, R, H, s.dR, v0, vc, s.dEc);
      tk::f2bf(s.dEc, (long long)vc * H, qp.GW + q.lay.emb.off + (long long)v0 * H, st);
      MKC();
    }
  } else MCK(cudaMemsetAsync(qp.GW + q.lay.emb.off, 0, sizeof(uint16_t) * (size_t)V * H, st));
  // 입력 열 흩기
  MCK(cudaMemsetAsync(s.vdOut, 0, sizeof(float) * (size_t)Rv * H, st));
  GPtrW gw;
  for (int g = 0; g < N_VG; ++g) { MCK(cudaMemsetAsync(s.dgOut[g], 0, sizeof(float) * (size_t)B * kVGrp[g].n_tok * H, st)); gw.p[g] = s.dgOut[g]; }
  scatter_k<<<nb((long long)R * H), 256, 0, st>>>(bt.src, R, H, s.dR, s.vdOut, gw);
  MKC();
  // 묶음 인코더
  for (int g = 0; g < N_VG; ++g) {
    const int rows = B * kVGrp[g].n_tok;
    tk::colsum(s.dgOut[g], rows, H, H, s.part, ap.GV + g_type[g], false, st);
    tk::f2bf(s.dgOut[g], (long long)rows * H, s.gdb, st);
    if (g == VG_OBJ) {
      tk::colsum(s.dgOut[g], rows, H, H, s.part, ap.GV + g_b2, false, st);
      tk::mm_dw(s.gdb, H, s.gH, c.obj_hid, rows, H, c.obj_hid, s.ws, DWCH, ap.GW + g_w2.off, nullptr, st);
      tk::mm_dx(s.gdb, H, rows, ap.W + g_w2.off, H, c.obj_hid, s.dgH, c.obj_hid, false, st);
      tk::gelu_bwd(s.dgH, s.gH1, (long long)rows * c.obj_hid, s.gdb, nullptr, st);
      tk::mm_dw(s.gdb, c.obj_hid, bt.grp[g], kVGrp[g].K, rows, c.obj_hid, kVGrp[g].K, s.ws, DWCH, ap.GW + g_w1[g].off, nullptr, st);
    } else {
      tk::mm_dw(s.gdb, H, bt.grp[g], kVGrp[g].K, rows, H, kVGrp[g].K, s.ws, DWCH, ap.GW + g_w1[g].off, nullptr, st);
    }
  }
  // 영상 사영·탑
  tk::colsum(s.vdOut, Rv, H, H, s.part, ap.GV + v_projb, false, st);
  tk::f2bf(s.vdOut, (long long)Rv * H, s.vdb, st);
  tk::mm_dw(s.vdb, H, s.vTok, c.vD, Rv, H, c.vD, s.ws, DWCH, ap.GW + v_proj.off, nullptr, st);
  if (c.vis_train) {
    tk::mm_dx(s.vdb, H, Rv, ap.W + v_proj.off, H, c.vD, s.vdA, c.vD, false, st);
    MCK(cudaMemsetAsync(s.vdR, 0, sizeof(float) * (size_t)Rv * c.vD, st));
    tk::ln_bwd(s.vdA, s.vX[c.vL], Rv, c.vD, ap.V + v_lnfg, c.vEps, s.vdR, s.wt, st);
    tk::colsum(s.wt, Rv, c.vD, c.vD, s.part, ap.GV + v_lnfg, false, st);
    tk::colsum(s.vdA, Rv, c.vD, c.vD, s.part, ap.GV + v_lnfb, false, st);
    for (int l = c.vL - 1; l >= 0; --l) v_block_bwd(*this, l, Rv, st);
    tk::colsum(s.vdR, Rv, c.vD, c.vD, s.part, ap.GV + v_patchb, false, st);
    tk::colsum(s.vdR, Rv / c.vT, c.vT * c.vD, c.vT * c.vD, s.part, ap.GV + v_pos, false, st);
    tk::f2bf(s.vdR, (long long)Rv * c.vD, s.vdb, st);
    tk::mm_dw(s.vdb, c.vD, bt.patches, c.vK, Rv, c.vD, c.vK, s.ws, DWCH, ap.GW + v_patch.off, nullptr, st);
  }
}

}  // namespace rvla
