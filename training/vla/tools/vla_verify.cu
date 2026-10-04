// RecallVLA 학습 검증·측정(MAPVLA_SPEC M2·M3, GPU_TRAINING 9절).
//   vla_verify v5 [--ki0] [--frozen-vis] [--seed S] [--nb N]  작은 구성: 손실·모든 변수 기울기를 CPU FP64 와 비교(바닥 = EMUL), 묶음 N 개(기본 4)
//                                         제곱 평균 제곱근으로 모아 2 배 규칙, FP64 참조판 대 유한 차분(첫 묶음)
//   vla_verify neg                        뒤 계산 버그 1–5 가 v5 를 실패시켜야 통과
//   vla_verify v67 [real B L]             같은 입력 두 번 비트 동일(V7), 그래프 == 즉시(V6), 다른 난수 열쇠 → 다름
//   vla_verify opt                        8 비트 Adam GPU == CPU 흉내(비트), 8 비트 대 FP32 상태 궤적 차이
//   vla_verify bench B L [--no-vis] [--fp32opt]   실제 크기(Qwen3.5-0.8B + SigLIP 2 B/32): 메모리·스텝 시간·표본/s
//   vla_verify smoke B L STEPS            실제 크기 짧은 학습(같은 배치 반복) — 손실이 내려가는지(≤ 2 분)
//   vla_verify dnref                      DeltaNet 덩이 꼴 CPU 참조판 == 재귀 꼴(FP64), 재귀 뒤 == 유한 차분
//   vla_verify dn                         DeltaNet 덩이 꼴 GPU 앞·뒤 대 CPU(덩이 EMUL — 같은 반올림 자리, 재귀 FP64)
//   vla_verify dnbench B n                DeltaNet 한 층 시간(덩이 꼴 대 예전 재귀 꼴)
//   vla_verify att                        텐서 코어 어텐션 앞·뒤 대 CPU(EMUL·FP64): 인과 GQA·전문가(구간 1 유효 길이 + 구간 2)·영상
// 환경 변수(비교·진단): RVLA_DN_OLD=1 학습 DeltaNet 예전 재귀 꼴, RVLA_ATT_OLD=1 예전 FP32 어텐션, VREF_ATT_EXACT=1 EMUL 어텐션 반올림 흉내 끔,
//   VREF_DN_REC=1 EMUL DeltaNet 을 재귀 꼴(반올림 흉내 없음)로, BENCH_NOFWD=1 bench 에서 앞만 재는 것 생략
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "model.h"
#include "net.h"
#include "optim.h"
#include "qkern.cuh"
#include "tkern.cuh"
#include "vref.h"

using namespace rvla;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static std::string HOME() { const char* h = getenv("HOME"); return h ? h : "."; }
static std::string QDIR() { return HOME() + "/robot-agent/training/model/Qwen3.5-0.8B"; }
static std::string SIGP() {
  return HOME() + "/.cache/huggingface/hub/models--timm--ViT-B-32-SigLIP2-256/snapshots/" +
         std::string("") ;
}
static std::string sig_path() {
  const std::string base = HOME() + "/.cache/huggingface/hub/models--timm--ViT-B-32-SigLIP2-256/snapshots";
  FILE* p = popen(("ls -d " + base + "/*/open_clip_model.safetensors 2>/dev/null | head -1").c_str(), "r");
  char buf[1024] = {0};
  if (p) { if (!fgets(buf, sizeof buf, p)) buf[0] = 0; pclose(p); }
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s;
}

// ---- 글 표 ----
struct TTab { std::vector<int> pre, vs, ve, suf, end; std::vector<std::vector<int>> instr, sub; bool ok = false; };
static TTab load_tab(const std::string& p) {
  TTab t;
  std::ifstream f(p, std::ios::binary);
  if (!f) return t;
  uint32_t n;
  f.read((char*)&n, 4);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t k, l;
    f.read((char*)&k, 4); f.read((char*)&l, 4);
    std::vector<int> ids(l);
    f.read((char*)ids.data(), 4 * l);
    if (k == 0) t.pre = ids; else if (k == 1) t.vs = ids; else if (k == 2) t.ve = ids; else if (k == 3) t.suf = ids; else if (k == 4) t.end = ids;
    else if (k == 10) t.instr.push_back(ids); else if (k == 11) t.sub.push_back(ids);
  }
  t.ok = !t.pre.empty() && !t.sub.empty();
  return t;
}

// ---- 미니배치(호스트 + 장치) ----
struct HB {
  int B = 0, L = 0, Mt = 0;
  std::vector<uint16_t> patches, grp[N_VG];
  std::vector<int> src, plen, trow, tid;
  std::vector<float> tw, chunk, cmask;
  uint32_t adim = 0xff;
  long long iter = 5;
  std::vector<void*> dev;
  VBatch vb;
  template <class T> T* up(const std::vector<T>& v) {
    void* p;
    CK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(1, v.size()) + 64));
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    dev.push_back(p);
    return (T*)p;
  }
  void upload() {
    vb.B = B; vb.L = L; vb.Mt = Mt;
    vb.patches = up(patches);
    for (int g = 0; g < N_VG; ++g) vb.grp[g] = up(grp[g]);
    vb.src = up(src); vb.plen = up(plen); vb.tgt_row = up(trow); vb.tgt_id = up(tid); vb.tgt_w = up(tw);
    vb.chunk = up(chunk); vb.cmask = up(cmask);
    vb.adim = up(std::vector<uint32_t>{adim});
    vb.iter = up(std::vector<long long>{iter});
  }
  void set_iter(long long it) { CK(cudaMemcpy((void*)vb.iter, &it, 8, cudaMemcpyHostToDevice)); iter = it; }
  ~HB() { for (void* p : dev) cudaFree(p); }
};
static void make_batch(HB& h, const VCfg& c, int B, int Lmin, uint64_t seed, const TTab* tt) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  const int Vq = c.q.vocab;
  auto rid = [&](int n) { std::vector<int> v(n); for (auto& x : v) x = (int)(g() % (uint64_t)Vq); return v; };
  h.B = B;
  const int Rv = B * c.cams * c.vT;
  h.patches.resize((size_t)Rv * c.vK);
  for (auto& x : h.patches) x = net::f2bf(U(g));
  for (int gg = 0; gg < N_VG; ++gg) {
    const auto& d = kVGrp[gg];
    h.grp[gg].assign((size_t)B * d.n_tok * d.K, 0);
    for (int r = 0; r < B * d.n_tok; ++r) {
      for (int k = 0; k < d.k_real; ++k) h.grp[gg][(size_t)r * d.K + k] = net::f2bf(U(g));
      h.grp[gg][(size_t)r * d.K + d.k_real] = 0x3f80;
    }
  }
  std::vector<std::vector<int>> seqs(B), kinds(B);
  std::vector<int> sub0(B), subn(B);
  for (int b = 0; b < B; ++b) {
    auto& s = seqs[b];
    auto txt = [&](const std::vector<int>& ids) { for (int i : ids) s.push_back(src_code(SK_TXT, i)); };
    const std::vector<int> pre = tt ? tt->pre : rid(3), vs = tt ? tt->vs : rid(1), ve = tt ? tt->ve : rid(1), suf = tt ? tt->suf : rid(5), end = tt ? tt->end : rid(1);
    txt(pre);
    for (int cam = 0; cam < c.cams; ++cam) {
      txt(vs);
      for (int i = 0; i < c.vT; ++i) s.push_back(src_code(SK_IMG, (b * c.cams + cam) * c.vT + i));
      txt(ve);
    }
    s.push_back(src_code(SK_GRP + VG_ARM, b)); s.push_back(src_code(SK_GRP + VG_BASE, b)); s.push_back(src_code(SK_GRP + VG_GOAL, b));
    const int nobj = 2 + (int)((b * 3 + seed) % 6);
    for (int j = 0; j < nobj; ++j) s.push_back(src_code(SK_GRP + VG_OBJ, b * 16 + j));
    s.push_back(src_code(SK_GRP + VG_WALL, b)); s.push_back(src_code(SK_GRP + VG_ROOM, b));
    txt(tt ? tt->instr[(b + seed) % tt->instr.size()] : rid(4 + b % 3));
    txt(suf);
    sub0[b] = (int)s.size();
    std::vector<int> sb = tt ? tt->sub[(b * 7 + seed) % tt->sub.size()] : rid(3 + b % 2);
    sb.insert(sb.end(), end.begin(), end.end());
    txt(sb);
    subn[b] = (int)sb.size();
  }
  int L = Lmin;
  for (auto& s : seqs) L = std::max<int>(L, (int)s.size());
  L = (L + 7) / 8 * 8;
  h.L = L;
  h.src.assign((size_t)B * L, src_code(SK_PAD, 0));
  h.plen.resize(B);
  h.trow.clear(); h.tid.clear();
  for (int b = 0; b < B; ++b) {
    for (size_t t = 0; t < seqs[b].size(); ++t) h.src[(size_t)b * L + t] = seqs[b][t];
    h.plen[b] = (int)seqs[b].size();
    for (int k = 0; k < subn[b]; ++k) {
      const int p = sub0[b] + k;
      h.trow.push_back(b * L + p - 1);
      h.tid.push_back(seqs[b][p] & 0x0fffffff);
    }
  }
  h.Mt = (int)h.trow.size();
  h.tw.assign(h.Mt, 1.f / (float)h.Mt);
  h.chunk.resize((size_t)B * c.Hc * c.A);
  for (auto& x : h.chunk) x = U(g);
  h.cmask.resize((size_t)B * c.Hc);
  for (size_t i = 0; i < h.cmask.size(); ++i) h.cmask[i] = (i % 5 == 4) ? 0.f : 1.f;
  h.upload();
}

template <class T>
static std::vector<T> down(const T* p, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost)); return v; }
static vref::V tod(const std::vector<uint16_t>& v) { vref::V o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = net::bf2f(v[i]); return o; }
static vref::V tod(const std::vector<float>& v) { return vref::V(v.begin(), v.end()); }

struct TRef { std::string name; int set; bool mat; long long off, n; };   // set 0 = 몸통, 1 = 그 밖
static std::vector<TRef> tensors(const Model& m) {
  std::vector<TRef> t;
  const auto& ql = m.q.lay;
  const QCfg& Q = m.q.c;
  auto M = [&](const std::string& nm, int s, const MT& x) { t.push_back({nm, s, true, x.off, (long long)x.N * x.K}); };
  auto Vv = [&](const std::string& nm, int s, long long off, long long n) { t.push_back({nm, s, false, off, n}); };
  M("q.embed(=LM head)", 0, ql.emb);
  for (int l = 0; l < Q.layers; ++l) {
    const auto& L = ql.l[l];
    const std::string p = "q.L" + std::to_string(l) + (Q.full[l] ? "F." : "D.");
    Vv(p + "ln1", 0, L.ln1, Q.H); Vv(p + "ln2", 0, L.ln2, Q.H);
    if (Q.full[l]) { M(p + "wqkv", 0, L.wqkv); M(p + "wo", 0, L.wo); Vv(p + "qn", 0, L.qn, Q.hd); Vv(p + "kn", 0, L.kn, Q.hd); }
    else {
      M(p + "win", 0, L.win); M(p + "wout", 0, L.wout); Vv(p + "conv", 0, L.convw, (long long)Q.lin_in() * Q.conv);
      Vv(p + "A_log", 0, L.alog, Q.lh); Vv(p + "dt_bias", 0, L.dtb, Q.lh); Vv(p + "gnorm", 0, L.gnw, Q.dv);
    }
    M(p + "wgu", 0, L.wgu); M(p + "wdn", 0, L.wdn);
  }
  Vv("q.lnf", 0, ql.lnf, Q.H);
  const VCfg& c = m.c;
  M("v.patch", 1, m.v_patch); Vv("v.patch_b", 1, m.v_patchb, c.vD); Vv("v.pos", 1, m.v_pos, (long long)c.vT * c.vD);
  for (int l = 0; l < c.vL; ++l) {
    const auto& b = m.vb[l];
    const std::string p = "v.B" + std::to_string(l) + ".";
    Vv(p + "ln1g", 1, b.ln1g, c.vD); Vv(p + "ln1b", 1, b.ln1b, c.vD); M(p + "qkv", 1, b.qkv); Vv(p + "qkv_b", 1, b.qkvb, 3 * c.vD);
    M(p + "proj", 1, b.proj); Vv(p + "proj_b", 1, b.projb, c.vD); Vv(p + "ln2g", 1, b.ln2g, c.vD); Vv(p + "ln2b", 1, b.ln2b, c.vD);
    M(p + "fc1", 1, b.fc1); Vv(p + "fc1_b", 1, b.fc1b, c.vMLP); M(p + "fc2", 1, b.fc2); Vv(p + "fc2_b", 1, b.fc2b, c.vD);
  }
  Vv("v.lnf_g", 1, m.v_lnfg, c.vD); Vv("v.lnf_b", 1, m.v_lnfb, c.vD);
  M("v.proj(768->H)", 1, m.v_proj); Vv("v.proj_b", 1, m.v_projb, Q.H);
  const char* gn[N_VG] = {"arm", "base", "goal", "obj", "wall", "room"};
  for (int g = 0; g < N_VG; ++g) { M(std::string("g.") + gn[g] + ".w1", 1, m.g_w1[g]); Vv(std::string("g.") + gn[g] + ".type", 1, m.g_type[g], Q.H); }
  M("g.obj.w2", 1, m.g_w2); Vv("g.obj.b2", 1, m.g_b2, Q.H);
  M("e.in", 1, m.e_in);
  for (size_t f = 0; f < m.eb.size(); ++f) {
    const auto& e = m.eb[f];
    const std::string p = "e.B" + std::to_string(f) + ".";
    Vv(p + "ln1", 1, e.ln1, c.De); Vv(p + "ln2", 1, e.ln2, c.De); Vv(p + "qn", 1, e.qn, Q.hd); Vv(p + "kn", 1, e.kn, Q.hd);
    M(p + "qkv", 1, e.qkv); M(p + "o", 1, e.o); M(p + "gu", 1, e.gu); M(p + "dn", 1, e.dn);
  }
  Vv("e.lnf", 1, m.e_lnf, c.De);
  M("e.out", 1, m.e_out);
  return t;
}
static const vref::V& pick(const vref::Params& p, const TRef& t) { return t.set == 0 ? (t.mat ? p.qW : p.qV) : (t.mat ? p.aW : p.aV); }
static vref::V& pickm(vref::Params& p, const TRef& t) { return t.set == 0 ? (t.mat ? p.qW : p.qV) : (t.mat ? p.aW : p.aV); }
static double rel(const vref::V& a, const vref::V& r, long long off, long long n) {
  double e = 0, z = 0;
  for (long long i = 0; i < n; ++i) { const double d = a[off + i] - r[off + i]; e += d * d; z += r[off + i] * r[off + i]; }
  return z > 0 ? std::sqrt(e / z) : (e > 0 ? 1e9 : 0);
}

struct V5Res { int pass = 0, tot = 0; double worst_ratio = 0; };
// V5: 작은 구성에서 손실·몸통 출력·변수 기울기 텐서마다 GPU 오차(대 FP64) ≤ 2 × 바닥(EMUL 대 FP64).
// 바닥 하나는 반올림 잡음의 한 표본이라 묶음(입력)에 따라 크게 흔들린다(같은 뿌리를 가진 텐서 무리 — 지도 인코더 g.*, 영상 ln1 — 가 함께 2 배를 넘나듦).
// 그래서 묶음 V5_NB 개(씨앗 11, 12, …)의 오차·바닥을 각각 제곱 평균 제곱근으로 모은 뒤 같은 2 배 규칙을 쓴다(검사 수는 그대로). 유한 차분은 첫 묶음에서.
static int g_v5seed = 11, g_v5nb = 4;
struct V5One { std::vector<std::string> nm; std::vector<double> eg, ef, ge; double fdw = 0; std::string fdn; int nfd = 0; };
static V5One run_v5_one(Model& m, const VCfg& c, bool frozen, uint64_t seed, bool fd) {
  V5One r;
  HB h;
  make_batch(h, m.c, 2, 0, seed, nullptr);
  m.step_grads(h.vb, 0);
  CK(cudaDeviceSynchronize());
  const auto lossg = down(m.loss, 3);
  vref::Params P{tod(down(m.qp.W, m.qp.nW)), tod(down(m.qp.V, m.qp.nV)), tod(down(m.ap.W, m.ap.nW)), tod(down(m.ap.V, m.ap.nV))};
  vref::Params G{tod(down(m.qp.GW, m.qp.nW)), tod(down(m.qp.GV, m.qp.nV)), tod(down(m.ap.GW, m.ap.nW)), tod(down(m.ap.GV, m.ap.nV))};
  vref::In in;
  in.B = h.B; in.L = h.L; in.Mt = h.Mt;
  in.patches = tod(h.patches);
  for (int g = 0; g < N_VG; ++g) in.grp[g] = tod(h.grp[g]);
  in.src = h.src; in.plen = h.plen; in.tgt_row = h.trow; in.tgt_id = h.tid; in.tgt_w = tod(h.tw);
  const int RA = h.B * c.Hc;
  in.ain = tod(down(m.flow_ain(), (size_t)RA * c.kpad()));
  in.u = tod(down(m.flow_u(), (size_t)RA * c.A));
  in.cmask = tod(h.cmask);
  in.adim = h.adim;
  vref::Out o64, oem;
  vref::run(vref::FP64, m, P, in, o64, true);
  vref::run(vref::EMUL, m, P, in, oem, true);
  auto put = [&](const std::string& nm, double eg, double ef, double ge) { r.nm.push_back(nm); r.eg.push_back(eg); r.ef.push_back(ef); r.ge.push_back(ge); };
  auto sc = [&](const char* nm, double g, double r64, double rem) {
    const double d = std::max(1e-12, std::fabs(r64));
    put(nm, std::fabs(g - r64) / d, std::fabs(rem - r64) / d, std::fabs(g - rem) / d);
  };
  sc("loss total", lossg[0], o64.loss, oem.loss);
  sc("loss text CE", lossg[1], o64.ltxt, oem.ltxt);
  sc("loss flow", lossg[2], o64.lfm, oem.lfm);
  {
    const auto hg = tod(down(m.hidden, (size_t)h.B * h.L * c.q.H));
    double e = 0, f = 0, z = 0, x = 0;
    for (int b = 0; b < h.B; ++b)
      for (int t = 0; t < h.plen[b]; ++t)
        for (int k = 0; k < c.q.H; ++k) {
          const size_t i = ((size_t)b * h.L + t) * c.q.H + k;
          e += (hg[i] - o64.hidden[i]) * (hg[i] - o64.hidden[i]); f += (oem.hidden[i] - o64.hidden[i]) * (oem.hidden[i] - o64.hidden[i]);
          x += (hg[i] - oem.hidden[i]) * (hg[i] - oem.hidden[i]); z += o64.hidden[i] * o64.hidden[i];
        }
    put("fwd backbone hidden", std::sqrt(e / z), std::sqrt(f / z), std::sqrt(x / z));
  }
  for (const auto& t : tensors(m)) {
    if (frozen && t.name.rfind("v.", 0) == 0 && t.name.rfind("v.proj", 0) != 0) continue;
    put("grad " + t.name, rel(pick(G, t), pick(o64.g, t), t.off, t.n), rel(pick(oem.g, t), pick(o64.g, t), t.off, t.n), rel(pick(G, t), pick(oem.g, t), t.off, t.n));
  }
  if (fd) {
    // FP64 참조판 대 유한 차분: 텐서마다 원소 3 개
    std::mt19937_64 g(5);
    for (const auto& t : tensors(m)) {
      if (frozen && t.name.rfind("v.", 0) == 0 && t.name.rfind("v.proj", 0) != 0) continue;
      for (int k = 0; k < 3; ++k) {
        const long long i = t.off + (long long)(g() % (uint64_t)t.n);
        vref::Params Pp = P;
        std::vector<vref::V> kvfix[2] = {o64.Kf, o64.Vf};
        in.kvfix = c.ki ? kvfix : nullptr;   // 지식 격리: 전문가가 보는 prefix K·V 는 상수(stop-gradient)
        const double p0 = pick(P, t)[i], hh = 2e-5 * std::max(1.0, std::fabs(p0));
        vref::Out a, b;
        pickm(Pp, t)[i] = p0 + hh;
        vref::run(vref::FP64, m, Pp, in, a, false);
        pickm(Pp, t)[i] = p0 - hh;
        vref::run(vref::FP64, m, Pp, in, b, false);
        in.kvfix = nullptr;
        const double fdv = (a.loss - b.loss) / (2 * hh), an = pick(o64.g, t)[i];
        const double e = std::fabs(fdv - an) / std::max(1e-6, std::fabs(an) + std::fabs(fdv));
        if (std::fabs(fdv - an) > 1e-9 && e > r.fdw) { r.fdw = e; r.fdn = t.name; }
        if (getenv("FDV") && e > 1e-4) std::printf("    fd %s[%lld]: fd %.6e analytic %.6e\n", t.name.c_str(), i - t.off, fdv, an);
        ++r.nfd;
      }
    }
  }
  return r;
}
static V5Res run_v5(bool ki0, bool frozen, int bug, bool fd, bool verbose) {
  VCfg c = tiny_vcfg();
  c.ki = !ki0;
  c.vis_train = !frozen;
  Model m;
  std::string err;
  if (!m.init(c, "", "", &err)) { std::fprintf(stderr, "init %s\n", err.c_str()); std::exit(2); }
  m.bug = bug;
  std::vector<V5One> rs;
  for (int k = 0; k < g_v5nb; ++k) rs.push_back(run_v5_one(m, c, frozen, (uint64_t)(g_v5seed + k), fd && k == 0));
  V5Res res;
  int tfail = 0;
  const size_t nc = rs[0].nm.size();
  std::vector<double> wseed(rs.size(), 0.0);
  for (size_t i = 0; i < nc; ++i) {
    double e2 = 0, f2 = 0, x2 = 0;
    for (size_t k = 0; k < rs.size(); ++k) {
      e2 += rs[k].eg[i] * rs[k].eg[i]; f2 += rs[k].ef[i] * rs[k].ef[i]; x2 += rs[k].ge[i] * rs[k].ge[i];
      wseed[k] = std::max(wseed[k], rs[k].eg[i] / std::max(rs[k].ef[i], 1e-12));
    }
    const double eg = std::sqrt(e2 / rs.size()), ef = std::sqrt(f2 / rs.size()), ge = std::sqrt(x2 / rs.size());
    const bool ok = eg <= std::max(2 * ef, 1e-5);
    res.pass += ok; res.tot++;
    res.worst_ratio = std::max(res.worst_ratio, eg / std::max(ef, 1e-12));
    if (!ok && rs[0].nm[i].rfind("grad", 0) == 0) ++tfail;
    if (verbose) std::printf("  %-27s err %.2e  floor %.2e  ratio %5.2f  GPU-EMUL %.2e %s\n", rs[0].nm[i].c_str(), eg, ef, eg / std::max(ef, 1e-12), ge, ok ? "" : "FAIL");
  }
  if (fd) {
    const bool ok = rs[0].fdw < 1e-4;
    std::printf("  FD (FP64 ref vs central differences, batch seed %d, %d params, h = 2e-5%s): worst rel err %.2e (%s) %s\n", g_v5seed, rs[0].nfd,
                c.ki ? ", expert prefix K/V held fixed = stop-gradient" : ", full gradient", rs[0].fdw, rs[0].fdn.c_str(), ok ? "" : "FAIL");
    res.pass += ok; res.tot++;
  }
  if (verbose) {
    std::printf("  per-batch worst err/floor ratio (seeds %d..%d):", g_v5seed, g_v5seed + g_v5nb - 1);
    for (double w : wseed) std::printf(" %.2f", w);
    std::printf("\n[v5] ki %d vis_train %d bug %d, %d batches (RMS): %d / %d checks pass, %d tensor grads fail, worst err/floor ratio %.2f\n", c.ki, c.vis_train,
                bug, g_v5nb, res.pass, res.tot, tfail, res.worst_ratio);
  }
  return res;
}

// ---- V6·V7 ----
static int run_v67(bool real, int B, int Lmin) {
  VCfg c = real ? VCfg() : tiny_vcfg();
  if (real) { c.Bmax = B; c.Lmax = std::max(Lmin, 240); c.Mtmax = B * 16; }
  Model m;
  std::string err;
  if (!m.init(c, real ? QDIR() : "", real ? sig_path() : "", &err)) { std::fprintf(stderr, "init %s\n", err.c_str()); return 2; }
  TTab tt = load_tab(HOME() + "/ra_vla/text/table.bin");
  HB h;
  make_batch(h, m.c, real ? B : 2, real ? Lmin : 0, 3, real && tt.ok ? &tt : nullptr);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto snap = [&]() {
    CK(cudaStreamSynchronize(st));
    auto a = down(m.qp.GW, m.qp.nW);
    auto b = down(m.ap.GW, m.ap.nW);
    auto cg = down((uint16_t*)m.qp.GV, m.qp.nV * 2);
    auto d = down((uint16_t*)m.ap.GV, m.ap.nV * 2);
    auto l = down((uint16_t*)m.loss, 6);
    a.insert(a.end(), b.begin(), b.end()); a.insert(a.end(), cg.begin(), cg.end()); a.insert(a.end(), d.begin(), d.end()); a.insert(a.end(), l.begin(), l.end());
    return a;
  };
  auto diff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { size_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i]; return n; };
  m.step_grads(h.vb, st);
  auto r0 = snap();
  m.step_grads(h.vb, st);
  auto r1 = snap();
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  m.step_grads(h.vb, st);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaMemsetAsync(m.qp.GW, 0, m.qp.nW * 2, st));
  CK(cudaGraphLaunch(ge, st));
  auto r2 = snap();
  h.set_iter(6);
  m.step_grads(h.vb, st);
  auto r3 = snap();
  const size_t d7 = diff(r0, r1), d6 = diff(r0, r2), dk = diff(r0, r3);
  std::printf("[v7] %s: same input twice %zu / %zu words differ; [v6] graph vs eager %zu differ; other noise key -> %zu differ (should be > 0)\n",
              real ? "real" : "tiny", d7, r0.size(), d6, dk);
  return (d7 == 0 && d6 == 0 && dk > 0) ? 0 : 1;
}

// ---- 옵티마이저 ----
static int run_opt() {
  const long long n = 3 * 256 + 77;
  PSet p;
  p.nW = n; p.nV = 100;
  std::vector<uint16_t> w(n), g(n);
  std::mt19937_64 rg(1);
  std::normal_distribution<float> N01(0.f, 1.f);
  for (long long i = 0; i < n; ++i) w[i] = net::f2bf(0.05f * N01(rg));
  CK(cudaMalloc(&p.W, n * 2)); CK(cudaMalloc(&p.GW, n * 2)); CK(cudaMalloc(&p.V, 400)); CK(cudaMalloc(&p.GV, 400));
  CK(cudaMemset(p.V, 0, 400)); CK(cudaMemset(p.GV, 0, 400));
  CK(cudaMemcpy(p.W, w.data(), n * 2, cudaMemcpyHostToDevice));
  OptCfg oc;
  oc.lr = 1e-3f; oc.clip = 1e9f;
  Opt o;
  o.init(oc, {&p});
  // CPU 흉내 상태
  std::vector<int8_t> m8(n, 127);
  std::vector<uint8_t> r8(n, 255);
  const long long nb = (n + 255) / 256;
  std::vector<float> ms(nb, 0.f), rs(nb, 0.f);
  std::vector<uint16_t> wc = w;
  size_t bad = 0;
  for (int t = 1; t <= 10; ++t) {
    for (long long i = 0; i < n; ++i) g[i] = net::f2bf(0.01f * N01(rg) * (i % 97 == 0 ? 50.f : 1.f));
    CK(cudaMemcpy(p.GW, g.data(), n * 2, cudaMemcpyHostToDevice));
    o.step(0);
    CK(cudaDeviceSynchronize());
    for (long long b = 0; b < nb; ++b) {
      const int len = (int)std::min<long long>(256, n - b * 256);
      opt8_ref_block(oc, t, 1.f, len, &wc[b * 256], &m8[b * 256], &r8[b * 256], &ms[b], &rs[b], &g[b * 256], b * 256);
    }
    auto wg = down(p.W, n);
    for (long long i = 0; i < n; ++i) bad += wg[i] != wc[i];
  }
  std::printf("[opt] 8-bit Adam GPU vs CPU emulation, 10 steps x %lld params: %zu bf16 words differ\n", n, bad);
  // 품질: 8 비트 대 FP32 상태(같은 기울기 열), 2 차 손실 0.5|w − w*|² 에서 100 스텝
  const long long n2 = 256 * 64;
  std::vector<float> wstar(n2);
  for (auto& x : wstar) x = 0.05f * N01(rg);
  double dmove = 0, dz = 0, l8 = 0, l32 = 0;
  for (int mode = 0; mode < 2; ++mode) {
    PSet q;
    q.nW = n2; q.nV = 64;
    CK(cudaMalloc(&q.W, n2 * 2)); CK(cudaMalloc(&q.GW, n2 * 2)); CK(cudaMalloc(&q.V, 256)); CK(cudaMalloc(&q.GV, 256));
    CK(cudaMemset(q.V, 0, 256)); CK(cudaMemset(q.GV, 0, 256));
    std::vector<uint16_t> w0(n2, 0);
    CK(cudaMemcpy(q.W, w0.data(), n2 * 2, cudaMemcpyHostToDevice));
    OptCfg qc;
    qc.lr = 1e-3f; qc.clip = 1e9f; qc.wd = 0.f; qc.fp32_states = mode == 1;
    Opt oq;
    oq.init(qc, {&q});
    std::mt19937_64 r2(9);
    std::vector<uint16_t> gg(n2);
    for (int t = 0; t < 100; ++t) {
      auto wc2 = down(q.W, n2);
      for (long long i = 0; i < n2; ++i) gg[i] = net::f2bf(net::bf2f(wc2[i]) - wstar[i] + 0.02f * N01(r2));
      CK(cudaMemcpy(q.GW, gg.data(), n2 * 2, cudaMemcpyHostToDevice));
      oq.step(0);
    }
    auto wf = down(q.W, n2);
    double l = 0;
    for (long long i = 0; i < n2; ++i) { const double d = net::bf2f(wf[i]) - wstar[i]; l += 0.5 * d * d; }
    static std::vector<uint16_t> w8;
    if (mode == 0) { w8 = wf; l8 = l; }
    else {
      l32 = l;
      for (long long i = 0; i < n2; ++i) { const double a = net::bf2f(w8[i]), b = net::bf2f(wf[i]); dmove += (a - b) * (a - b); dz += b * b; }
    }
    cudaFree(q.W); cudaFree(q.GW); cudaFree(q.V); cudaFree(q.GV);
  }
  std::printf("[opt] quadratic, 100 steps, %lld params: loss 8-bit %.5f vs FP32 states %.5f (start %.5f); weights rel diff %.3e\n", n2, l8, l32,
              [&] { double s = 0; for (auto x : wstar) s += 0.5 * x * x; return s; }(), std::sqrt(dmove / dz));
  // 음성 대조: 0 쪽 자르기(확률 반올림 빠뜨림)
  int caught = 0;
  {
    Opt ob;
    PSet q2 = p;
    CK(cudaMemcpy(p.W, w.data(), n * 2, cudaMemcpyHostToDevice));
    ob.init(oc, {&q2});
    ob.bug = 1;
    std::vector<int8_t> m8b(n, 127);
    std::vector<uint8_t> r8b(n, 255);
    std::vector<float> msb(nb, 0.f), rsb(nb, 0.f);
    std::vector<uint16_t> wcb = w;
    for (long long i = 0; i < n; ++i) g[i] = net::f2bf(0.01f * N01(rg));
    CK(cudaMemcpy(p.GW, g.data(), n * 2, cudaMemcpyHostToDevice));
    ob.step(0);
    for (long long b = 0; b < nb; ++b) {
      const int len = (int)std::min<long long>(256, n - b * 256);
      opt8_ref_block(oc, 1, 1.f, len, &wcb[b * 256], &m8b[b * 256], &r8b[b * 256], &msb[b], &rsb[b], &g[b * 256], b * 256);
    }
    auto wg = down(p.W, n);
    size_t d = 0;
    for (long long i = 0; i < n; ++i) d += wg[i] != wcb[i];
    caught = d > 0;
    std::printf("[opt] negative (truncate instead of stochastic rounding): %zu words differ -> %s\n", d, caught ? "caught" : "NOT caught");
  }
  return (bad == 0 && caught) ? 0 : 1;
}


// ---- DeltaNet 덩이 꼴: CPU 참조판끼리(덩이 FP64 == 재귀 FP64) + GPU 대 참조 ----
static double relv(const vref::V& a, const vref::V& r) {
  double e = 0, z = 0;
  for (size_t i = 0; i < a.size(); ++i) { e += (a[i] - r[i]) * (a[i] - r[i]); z += r[i] * r[i]; }
  return std::sqrt(e / std::max(z, 1e-300));
}
struct DnIn { int n, dk, dv; vref::V q, k, v, g, be, S0, dO, dS1; };
static DnIn dn_rand(int n, int dk, int dv, uint64_t seed, bool s0) {
  DnIn d{n, dk, dv};
  std::mt19937_64 rg(seed);
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(0, 1);
  d.q.resize((size_t)n * dk); d.k.resize((size_t)n * dk); d.v.resize((size_t)n * dv); d.g.resize(n); d.be.resize(n);
  for (int t = 0; t < n; ++t) {
    double sq = 0, sk = 0;
    std::vector<double> a(dk), b(dk);
    for (int i = 0; i < dk; ++i) { a[i] = N(rg); b[i] = N(rg); sq += a[i] * a[i]; sk += b[i] * b[i]; }
    for (int i = 0; i < dk; ++i) { d.q[(size_t)t * dk + i] = a[i] / std::sqrt(sq) / std::sqrt((double)dk); d.k[(size_t)t * dk + i] = b[i] / std::sqrt(sk); }
    for (int j = 0; j < dv; ++j) d.v[(size_t)t * dv + j] = N(rg);
    d.g[t] = -std::exp(-1.5 + 1.5 * N(rg)) * 0.5;
    d.be[t] = 1.0 / (1.0 + std::exp(-1.5 * N(rg)));
  }
  d.dO.resize((size_t)n * dv);
  for (auto& x : d.dO) x = N(rg);
  if (s0) {
    d.S0.resize((size_t)dk * dv); d.dS1.resize((size_t)dk * dv);
    for (auto& x : d.S0) x = 0.3 * N(rg);
    for (auto& x : d.dS1) x = 0.3 * N(rg);
  }
  return d;
}
static int run_dnref() {
  int bad = 0;
  for (int cfg = 0; cfg < 4; ++cfg) {
    const int dk = cfg < 2 ? 16 : 128, C = cfg < 2 ? 16 : 64, n = cfg < 2 ? 45 : 150;
    const bool s0 = cfg % 2 == 1;
    DnIn d = dn_rand(n, dk, dk, 100 + cfg, s0);
    const double* S0 = s0 ? d.S0.data() : nullptr;
    const double* dS1 = s0 ? d.dS1.data() : nullptr;
    vref::V o1((size_t)n * dk), o2((size_t)n * dk), S1a((size_t)dk * dk), S1b((size_t)dk * dk);
    vref::dn_rec_fwd(n, dk, dk, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), S0, o1.data(), S1a.data());
    vref::dn_chk_fwd(n, dk, dk, C, false, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), S0, o2.data(), S1b.data());
    vref::V g1[6], g2[6];
    const size_t sz[6] = {(size_t)n * dk, (size_t)n * dk, (size_t)n * dk, (size_t)n, (size_t)n, (size_t)dk * dk};
    for (int i = 0; i < 6; ++i) { g1[i].assign(sz[i], 0); g2[i].assign(sz[i], 0); }
    vref::dn_rec_bwd(n, dk, dk, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), S0, d.dO.data(), dS1, g1[0].data(), g1[1].data(), g1[2].data(),
                     g1[3].data(), g1[4].data(), g1[5].data());
    vref::dn_chk_bwd(n, dk, dk, C, false, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), S0, d.dO.data(), dS1, g2[0].data(), g2[1].data(),
                     g2[2].data(), g2[3].data(), g2[4].data(), g2[5].data());
    const char* nm[6] = {"dq", "dk", "dv", "dg", "dbeta", "dS0"};
    double w = std::max(relv(o2, o1), relv(S1b, S1a));
    std::printf("[dnref] dk %d C %d n %d S0 %d: o %.2e S1 %.2e", dk, C, n, (int)s0, relv(o2, o1), relv(S1b, S1a));
    for (int i = 0; i < 6; ++i) { const double e = relv(g2[i], g1[i]); w = std::max(w, e); std::printf(" %s %.2e", nm[i], e); }
    // 재귀 뒤 대 유한 차분(손실 = Σ dO·o + Σ dS1·S1)
    auto loss = [&](const DnIn& x) {
      vref::V o((size_t)n * dk), S1((size_t)dk * dk);
      vref::dn_rec_fwd(n, dk, dk, x.q.data(), x.k.data(), x.v.data(), x.g.data(), x.be.data(), s0 ? x.S0.data() : nullptr, o.data(), S1.data());
      double L = 0;
      for (size_t i = 0; i < o.size(); ++i) L += o[i] * d.dO[i];
      if (s0) for (size_t i = 0; i < S1.size(); ++i) L += S1[i] * d.dS1[i];
      return L;
    };
    double wf = 0;
    std::mt19937_64 rg(7);
    for (int which = 0; which < 5; ++which)
      for (int rep = 0; rep < 4; ++rep) {
        DnIn a = d, b = d;
        vref::V* pa[5] = {&a.q, &a.k, &a.v, &a.g, &a.be};
        vref::V* pb[5] = {&b.q, &b.k, &b.v, &b.g, &b.be};
        const size_t i = rg() % pa[which]->size();
        const double h = 1e-6;
        (*pa[which])[i] += h; (*pb[which])[i] -= h;
        const double fd = (loss(a) - loss(b)) / (2 * h), an = g1[which][i];
        wf = std::max(wf, std::fabs(fd - an) / std::max(1e-6, std::fabs(fd) + std::fabs(an)));
      }
    std::printf(" | rec vs FD %.1e\n", wf);
    if (w > 1e-9 || wf > 1e-5) ++bad;
  }
  std::printf("[dnref] %s\n", bad ? "FAIL" : "PASS (chunked FP64 == recurrent FP64, recurrent bwd == FD)");
  return bad ? 1 : 0;
}

// GPU 덩이 꼴 대 CPU(덩이 EMUL = 같은 반올림 자리, 재귀 FP64 = 정답). 비율 = GPU 대 FP64 오차 / EMUL 대 FP64 오차
static int run_dngpu(bool verbose) {
  int bad = 0;
  for (int cfg = 0; cfg < 2; ++cfg) {
    const int dk = cfg == 0 ? 16 : 128, dv = dk, lh = cfg == 0 ? 3 : 2, B = 2, n = cfg == 0 ? 45 : 150, C = tk::dn_chunk(dk), R = B * n;
    std::vector<DnIn> ins;
    for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) ins.push_back(dn_rand(n, dk, dv, 1000 + cfg * 10 + b * lh + h, false));
    std::vector<float> q((size_t)R * lh * dk), k(q.size()), v((size_t)R * lh * dv), g((size_t)R * lh), be(g.size()), dO(v.size());
    for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) {
      const DnIn& d = ins[b * lh + h];
      for (int t = 0; t < n; ++t) {
        const size_t r = (size_t)b * n + t;
        for (int i = 0; i < dk; ++i) { q[(r * lh + h) * dk + i] = (float)d.q[(size_t)t * dk + i]; k[(r * lh + h) * dk + i] = (float)d.k[(size_t)t * dk + i]; }
        for (int j = 0; j < dv; ++j) { v[(r * lh + h) * dv + j] = (float)d.v[(size_t)t * dv + j]; dO[(r * lh + h) * dv + j] = (float)d.dO[(size_t)t * dv + j]; }
        g[r * lh + h] = (float)d.g[t]; be[r * lh + h] = (float)d.be[t];
      }
    }
    // 참조판은 GPU 가 받은 FP32 값 그대로
    for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) {
      DnIn& d = ins[b * lh + h];
      for (int t = 0; t < n; ++t) {
        const size_t r = (size_t)b * n + t;
        for (int i = 0; i < dk; ++i) { d.q[(size_t)t * dk + i] = q[(r * lh + h) * dk + i]; d.k[(size_t)t * dk + i] = k[(r * lh + h) * dk + i]; }
        for (int j = 0; j < dv; ++j) { d.v[(size_t)t * dv + j] = v[(r * lh + h) * dv + j]; d.dO[(size_t)t * dv + j] = dO[(r * lh + h) * dv + j]; }
        d.g[t] = g[r * lh + h]; d.be[t] = be[r * lh + h];
      }
    }
    auto upf = [](const std::vector<float>& x) { float* p; CK(cudaMalloc(&p, x.size() * 4 + 64)); CK(cudaMemcpy(p, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); return p; };
    float *dq = upf(q), *dkk = upf(k), *dvv = upf(v), *dg = upf(g), *db = upf(be), *ddo = upf(dO);
    float *O, *gq, *gk, *gv, *gg, *gb, *ws;
    CK(cudaMalloc(&O, v.size() * 4)); CK(cudaMalloc(&gq, q.size() * 4)); CK(cudaMalloc(&gk, q.size() * 4)); CK(cudaMalloc(&gv, v.size() * 4));
    CK(cudaMalloc(&gg, g.size() * 4)); CK(cudaMalloc(&gb, g.size() * 4));
    CK(cudaMalloc(&ws, tk::dnc_ws_floats(B, n, lh, dk, dv) * 4));
    tk::dnc_fwd(dq, dkk, dvv, lh * dv, dg, db, B, n, lh, dk, dv, nullptr, nullptr, false, O, ws, true, 0);
    tk::dnc_bwd(dvv, lh * dv, ddo, B, n, lh, dk, dv, false, ws, gq, gk, gv, lh * dv, gg, gb, 0, 0);
    CK(cudaDeviceSynchronize());
    auto dl = [](const float* p, size_t nn) { std::vector<float> x(nn); CK(cudaMemcpy(x.data(), p, nn * 4, cudaMemcpyDeviceToHost)); return x; };
    std::vector<float> hO = dl(O, v.size()), hq = dl(gq, q.size()), hk = dl(gk, q.size()), hv = dl(gv, v.size()), hg = dl(gg, g.size()), hb = dl(gb, g.size());
    // 모아서 비교(텐서 하나 = 모든 판·머리)
    const char* nm[6] = {"o", "dq", "dk", "dv", "dg", "dbeta"};
    vref::V G6[6], E6[6], F6[6];
    for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) {
      const DnIn& d = ins[b * lh + h];
      vref::V oe((size_t)n * dv), of((size_t)n * dv), x[3][5];
      for (int m = 0; m < 3; ++m) { x[m][0].assign((size_t)n * dk, 0); x[m][1].assign((size_t)n * dk, 0); x[m][2].assign((size_t)n * dv, 0); x[m][3].assign(n, 0); x[m][4].assign(n, 0); }
      vref::dn_chk_fwd(n, dk, dv, C, true, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), nullptr, oe.data(), nullptr);
      vref::dn_rec_fwd(n, dk, dv, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), nullptr, of.data(), nullptr);
      vref::dn_chk_bwd(n, dk, dv, C, true, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), nullptr, d.dO.data(), nullptr, x[0][0].data(), x[0][1].data(),
                       x[0][2].data(), x[0][3].data(), x[0][4].data(), nullptr);
      vref::dn_rec_bwd(n, dk, dv, d.q.data(), d.k.data(), d.v.data(), d.g.data(), d.be.data(), nullptr, d.dO.data(), nullptr, x[1][0].data(), x[1][1].data(),
                       x[1][2].data(), x[1][3].data(), x[1][4].data(), nullptr);
      for (int t = 0; t < n; ++t) {
        const size_t r = (size_t)b * n + t;
        for (int j = 0; j < dv; ++j) { G6[0].push_back(hO[(r * lh + h) * dv + j]); E6[0].push_back(oe[(size_t)t * dv + j]); F6[0].push_back(of[(size_t)t * dv + j]); }
        for (int i = 0; i < dk; ++i) {
          G6[1].push_back(hq[(r * lh + h) * dk + i]); E6[1].push_back(x[0][0][(size_t)t * dk + i]); F6[1].push_back(x[1][0][(size_t)t * dk + i]);
          G6[2].push_back(hk[(r * lh + h) * dk + i]); E6[2].push_back(x[0][1][(size_t)t * dk + i]); F6[2].push_back(x[1][1][(size_t)t * dk + i]);
        }
        for (int j = 0; j < dv; ++j) { G6[3].push_back(hv[(r * lh + h) * dv + j]); E6[3].push_back(x[0][2][(size_t)t * dv + j]); F6[3].push_back(x[1][2][(size_t)t * dv + j]); }
        G6[4].push_back(hg[r * lh + h]); E6[4].push_back(x[0][3][t]); F6[4].push_back(x[1][3][t]);
        G6[5].push_back(hb[r * lh + h]); E6[5].push_back(x[0][4][t]); F6[5].push_back(x[1][4][t]);
      }
    }
    std::printf("[dn] dk %d C %d B %d lh %d n %d:", dk, C, B, lh, n);
    for (int i = 0; i < 6; ++i) {
      const double ge = relv(G6[i], E6[i]), gf = relv(G6[i], F6[i]), ef = relv(E6[i], F6[i]);
      const bool ok = ge < 2e-3 && gf <= std::max(2 * ef, 1e-5);
      bad += !ok;
      std::printf(" %s GPU-EMUL %.1e GPU-FP64 %.1e EMUL-FP64 %.1e%s;", nm[i], ge, gf, ef, ok ? "" : " FAIL");
    }
    std::printf("\n");
    (void)verbose;
    cudaFree(dq); cudaFree(dkk); cudaFree(dvv); cudaFree(dg); cudaFree(db); cudaFree(ddo); cudaFree(O); cudaFree(gq); cudaFree(gk); cudaFree(gv); cudaFree(gg); cudaFree(gb); cudaFree(ws);
  }
  std::printf("[dn] %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

// DeltaNet 한 층 시간(실제 모양: lh 16, dk = dv = 128): 덩이 꼴 앞·뒤 대 예전 재귀 꼴
static int run_dnbench(int B, int n) {
  const int lh = 16, dk = 128, dv = 128, R = B * n;
  std::mt19937_64 rg(3);
  std::normal_distribution<float> N(0.f, 1.f);
  std::vector<float> q((size_t)R * lh * dk), v((size_t)R * lh * dv), g((size_t)R * lh), be(g.size());
  for (auto& x : q) x = N(rg) * 0.0884f;
  for (auto& x : v) x = N(rg);
  for (size_t i = 0; i < g.size(); ++i) { g[i] = -0.1f * std::fabs(N(rg)); be[i] = 0.5f; }
  auto upf = [](const std::vector<float>& x) { float* p; CK(cudaMalloc(&p, x.size() * 4 + 64)); CK(cudaMemcpy(p, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); return p; };
  float *dq = upf(q), *dk2 = upf(q), *dv2 = upf(v), *dg = upf(g), *db = upf(be), *ddo = upf(v);
  float *O, *gq, *gk, *gv, *gg, *gb, *ws, *ws0;
  CK(cudaMalloc(&O, v.size() * 4)); CK(cudaMalloc(&gq, q.size() * 4)); CK(cudaMalloc(&gk, q.size() * 4)); CK(cudaMalloc(&gv, v.size() * 4));
  CK(cudaMalloc(&gg, g.size() * 4)); CK(cudaMalloc(&gb, g.size() * 4));
  CK(cudaMalloc(&ws, tk::dnc_ws_floats(B, n, lh, dk, dv) * 4));
  CK(cudaMalloc(&ws0, tk::dn_ws_floats(B, n, lh, dk, dv) * 4));
  cudaEvent_t e[5];
  for (auto& x : e) cudaEventCreate(&x);
  float t[4] = {0, 0, 0, 0};
  const int reps = 5;
  for (int r = 0; r <= reps; ++r) {
    cudaEventRecord(e[0]);
    tk::dnc_fwd(dq, dk2, dv2, lh * dv, dg, db, B, n, lh, dk, dv, nullptr, nullptr, false, O, ws, true, 0);
    cudaEventRecord(e[1]);
    tk::dnc_bwd(dv2, lh * dv, ddo, B, n, lh, dk, dv, false, ws, gq, gk, gv, lh * dv, gg, gb, 0, 0);
    cudaEventRecord(e[2]);
    qk::deltanet(dq, dk2, dv2, lh * dv, dg, db, B, n, lh, dk, dv, nullptr, nullptr, false, O, 0, ws0);
    cudaEventRecord(e[3]);
    tk::deltanet_bwd(dq, dk2, dv2, lh * dv, dg, db, ddo, B, n, lh, dk, dv, false, true, ws0, gq, gk, gv, lh * dv, gg, gb, 0, 0);
    cudaEventRecord(e[4]);
    CK(cudaEventSynchronize(e[4]));
    if (r == 0) continue;
    for (int k = 0; k < 4; ++k) { float ms; cudaEventElapsedTime(&ms, e[k], e[k + 1]); t[k] += ms / reps; }
  }
  std::printf("[dnbench] B %d n %d lh %d: chunked fwd %.3f ms bwd %.3f ms | recurrent fwd(+ck) %.3f ms bwd %.3f ms\n", B, n, lh, t[0], t[1], t[2], t[3]);
  return 0;
}

// 텐서 코어 어텐션 대 CPU(EMUL = 같은 반올림 자리, FP64): 인과(GQA) / 구간 1 유효 길이 + 구간 2(전문가) / 양방향(영상)
static int run_attgpu() {
  int bad = 0;
  struct Cs { const char* nm; int hd, nq, nkv, n, L1, n2, causal, ki; };
  const Cs cs[] = {{"causal hd32", 32, 2, 1, 40, 40, 0, 1, 0}, {"causal hd256 GQA", 256, 8, 2, 100, 100, 0, 1, 0}, {"expert hd32", 32, 2, 1, 4, 40, 4, 0, 0},
                   {"expert hd256", 256, 8, 2, 16, 120, 16, 0, 0}, {"vision hd64", 64, 12, 12, 64, 64, 0, 0, 0}, {"vision hd32", 32, 2, 2, 4, 4, 0, 0, 0}};
  for (const Cs& c : cs) {
    const int B = 2;
    std::mt19937_64 rg(77);
    std::normal_distribution<double> N(0, 1);
    const int qw = c.nq * c.hd, kw = c.nkv * c.hd;
    std::vector<float> q((size_t)B * c.n * qw), k1((size_t)B * c.L1 * kw), v1(k1.size()), k2((size_t)B * c.n2 * kw + 4), v2(k2.size()), dO(q.size());
    for (auto& x : q) x = (float)N(rg);
    for (auto& x : k1) x = (float)N(rg);
    for (auto& x : v1) x = (float)N(rg);
    for (auto& x : k2) x = (float)N(rg);
    for (auto& x : v2) x = (float)N(rg);
    for (auto& x : dO) x = (float)N(rg);
    std::vector<int> len = {c.L1 - 7, c.L1};
    if (c.causal || c.n2 == 0) len = {c.L1, c.L1};
    auto upf = [](const std::vector<float>& x) { float* p; CK(cudaMalloc(&p, x.size() * 4 + 64)); CK(cudaMemcpy(p, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); return p; };
    float *dq = upf(q), *dk1 = upf(k1), *dv1 = upf(v1), *dk2 = upf(k2), *dv2 = upf(v2), *ddo = upf(dO);
    int* dlen;
    CK(cudaMalloc(&dlen, 8)); CK(cudaMemcpy(dlen, len.data(), 8, cudaMemcpyHostToDevice));
    std::vector<float> z(q.size(), 0.f), zk1(k1.size(), 0.f), zk2(k2.size(), 0.f);
    float *O = upf(z), *gq = upf(z), *gk1 = upf(zk1), *gv1 = upf(zk1), *gk2 = upf(zk2), *gv2 = upf(zk2), *lse = upf(z), *Dd = upf(z);
    tk::AttP a;
    a.Q = dq; a.ldq = qw; a.K1 = dk1; a.V1 = dv1; a.ldk1 = kw; a.L1 = c.L1; a.n1c = c.L1; a.len1 = c.n2 ? dlen : nullptr;
    if (c.n2) { a.K2 = dk2; a.V2 = dv2; a.ldk2 = kw; a.n2 = c.n2; }
    a.causal = c.causal; a.B = B; a.n = c.n; a.nq = c.nq; a.nkv = c.nkv; a.hd = c.hd; a.scale = 1.f / std::sqrt((float)c.hd); a.O = O; a.ldo = qw; a.lse = lse;
    a.dO = ddo; a.lddo = qw; a.Dd = Dd; a.dQ = gq; a.lddq = qw; a.dK1 = gk1; a.dV1 = gv1;
    if (c.n2) { a.dK2 = gk2; a.dV2 = gv2; }
    tk::att_fwd(a, 0);
    tk::att_bwd(a, 0);
    CK(cudaDeviceSynchronize());
    auto dl = [](const float* p, size_t nn) { std::vector<float> x(nn); CK(cudaMemcpy(x.data(), p, nn * 4, cudaMemcpyDeviceToHost)); return x; };
    std::vector<float> hO = dl(O, q.size()), hq = dl(gq, q.size()), hk1 = dl(gk1, k1.size()), hv1 = dl(gv1, k1.size()), hk2 = dl(gk2, k2.size()), hv2 = dl(gv2, k2.size());
    vref::V G[4], E[4], F[4];
    for (int b = 0; b < B; ++b) {
      const int nk = c.L1 + c.n2, pl = len[b], L1 = c.L1;
      vref::V Q((size_t)c.n * qw), K((size_t)nk * kw), Vv((size_t)nk * kw), D((size_t)c.n * qw);
      for (size_t i = 0; i < Q.size(); ++i) { Q[i] = q[(size_t)b * c.n * qw + i]; D[i] = dO[(size_t)b * c.n * qw + i]; }
      for (int j = 0; j < nk; ++j)
        for (int e = 0; e < kw; ++e) {
          K[(size_t)j * kw + e] = j < L1 ? k1[((size_t)b * L1 + j) * kw + e] : k2[((size_t)b * c.n2 + j - L1) * kw + e];
          Vv[(size_t)j * kw + e] = j < L1 ? v1[((size_t)b * L1 + j) * kw + e] : v2[((size_t)b * c.n2 + j - L1) * kw + e];
        }
      const bool causal = c.causal;
      auto ok = [=](int t, int j) { return j < L1 ? (j < pl && (!causal || j <= t)) : true; };
      for (int m = 0; m < 2; ++m) {
        vref::V o((size_t)c.n * qw), gq2((size_t)c.n * qw, 0), gk((size_t)nk * kw, 0), gv((size_t)nk * kw, 0);
        vref::att_ref(m == 0, c.n, nk, c.nq, c.nkv, c.hd, ok, Q.data(), K.data(), Vv.data(), D.data(), o.data(), gq2.data(), gk.data(), gv.data());
        vref::V* T = m == 0 ? E : F;
        T[0].insert(T[0].end(), o.begin(), o.end()); T[1].insert(T[1].end(), gq2.begin(), gq2.end());
        T[2].insert(T[2].end(), gk.begin(), gk.end()); T[3].insert(T[3].end(), gv.begin(), gv.end());
      }
      for (size_t i = 0; i < Q.size(); ++i) { G[0].push_back(hO[(size_t)b * c.n * qw + i]); G[1].push_back(hq[(size_t)b * c.n * qw + i]); }
      for (int j = 0; j < nk; ++j)
        for (int e = 0; e < kw; ++e) {
          G[2].push_back(j < L1 ? hk1[((size_t)b * L1 + j) * kw + e] : hk2[((size_t)b * c.n2 + j - L1) * kw + e]);
          G[3].push_back(j < L1 ? hv1[((size_t)b * L1 + j) * kw + e] : hv2[((size_t)b * c.n2 + j - L1) * kw + e]);
        }
    }
    const char* nm[4] = {"O", "dQ", "dK", "dV"};
    std::printf("[att] %-18s", c.nm);
    for (int i = 0; i < 4; ++i) {
      const double ge = relv(G[i], E[i]), gf = relv(G[i], F[i]), ef = relv(E[i], F[i]);
      const bool okk = ge < 1e-4 && gf <= std::max(2 * ef, 1e-5);
      bad += !okk;
      std::printf(" %s G-E %.1e G-F %.1e E-F %.1e%s;", nm[i], ge, gf, ef, okk ? "" : " FAIL");
    }
    std::printf("\n");
  }
  std::printf("[att] %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
// ---- 측정 ----
static size_t used_bytes() { size_t f, t; cudaMemGetInfo(&f, &t); return t - f; }
static int run_bench(int B, int Lmin, bool novis, bool fp32opt, int steps, bool smoke) {
  const size_t base = used_bytes();
  VCfg c;
  c.Bmax = B; c.Lmax = std::max(Lmin, 64); c.Mtmax = B * 16;
  c.vis_train = !novis;
  Model m;
  std::string err;
  const std::string sp = sig_path();
  if (!m.init(c, QDIR(), sp, &err)) { std::fprintf(stderr, "init %s\n", err.c_str()); return 2; }
  const size_t after_model = used_bytes();
  OptCfg oc;
  oc.fp32_states = fp32opt;
  oc.lr = smoke ? 1e-4f : 1e-5f;
  Opt o;
  o.init(oc, {&m.qp, &m.ap});
  if (novis) { o.bufs[1].skip0 = 0; o.bufs[1].skip1 = m.v_proj.off; }
  const size_t after_opt = used_bytes();
  TTab tt = load_tab(HOME() + "/ra_vla/text/table.bin");
  HB h;
  make_batch(h, m.c, B, Lmin, 1, tt.ok ? &tt : nullptr);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const long long nparam = m.qp.nW + m.qp.nV + m.ap.nW + m.ap.nV;
  std::printf("[bench] params %.1f M (backbone %.1f M, aux %.1f M: vision+proj %.1f M, expert+enc %.1f M); B %d L %d Mt %d; SigLIP weights %s\n", nparam / 1e6,
              (m.qp.nW + m.qp.nV) / 1e6, (m.ap.nW + m.ap.nV) / 1e6, (m.v_proj.off + (long long)m.v_proj.N * m.v_proj.K) / 1e6,
              (m.ap.nW - m.v_proj.off - (long long)m.v_proj.N * m.v_proj.K) / 1e6, B, h.L, h.Mt, sp.empty() ? "MISSING (random)" : "loaded");
  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  float tf = 0, tg = 0, to = 0;
  std::vector<float> losses;
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    h.set_iter(100 + s);
    if (!smoke && s >= 1 && !getenv("BENCH_NOFWD")) { cudaEventRecord(e0, st); m.forward_loss(h.vb, st); cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1)); float a; cudaEventElapsedTime(&a, e0, e1); tf += a; }
    cudaEventRecord(e0, st);
    m.step_grads(h.vb, st);
    cudaEventRecord(e1, st);
    o.step(st);
    cudaEventRecord(e2, st);
    CK(cudaEventSynchronize(e2));
    float a, b;
    cudaEventElapsedTime(&a, e0, e1);
    cudaEventElapsedTime(&b, e1, e2);
    if (s >= 1) { tg += a; to += b; }
    auto l = down(m.loss, 3);
    losses.push_back(l[0]);
    if (smoke) std::printf("  step %2d loss %.4f (text %.4f, flow %.4f) grad-norm %.3f  %.0f ms\n", s, l[0], l[1], l[2], down(o.gn + (o.bufs.size() * 2 * 240), 1)[0], a + b);
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 100) { std::printf("  (time limit)\n"); steps = s + 1; break; }
  }
  const int n = std::max(1, steps - 1);
  const size_t peak = used_bytes();
  std::printf("[bench] device: model+work %.2f GB (params+grads+activation work), optimizer states %.2f GB, total used %.2f GB (incl. %.2f GB before start)\n",
              (after_model - base) / 1e9, (after_opt - after_model) / 1e9, peak / 1e9, base / 1e9);
  std::printf("[bench] model alloc %.2f GB (of which weights %.2f GB bf16 + %.3f GB f32, grads %.2f GB), optimizer %.2f GB\n", m.bytes / 1e9 + m.q.bytes / 1e9,
              (m.qp.nW + m.ap.nW) * 2 / 1e9, (m.qp.nV + m.ap.nV) * 4 / 1e9, (m.qp.nW + m.ap.nW) * 2 / 1e9 + (m.qp.nV + m.ap.nV) * 4 / 1e9, o.bytes / 1e9);
  if (!smoke)
    std::printf("[bench] per step (B %d): forward only %.1f ms, forward+backward %.1f ms, optimizer %.1f ms -> %.2f samples/s (%.1f ms/sample)\n", B, tf / n,
                tg / n, to / n, B * 1000.0 / ((tg + to) / n), (tg + to) / n / B);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: vla_verify v5|neg|v67|opt|bench|smoke ...\n"); return 2; }
  const std::string cmd = argv[1];
  auto has = [&](const char* f) { for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == f) return true; return false; };
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--seed") g_v5seed = std::atoi(argv[i + 1]);
    if (std::string(argv[i]) == "--nb") g_v5nb = std::atoi(argv[i + 1]);
  }
  if (cmd == "v5") {
    V5Res r = run_v5(has("--ki0"), has("--frozen-vis"), 0, true, true);
    return r.pass == r.tot ? 0 : 1;
  }
  if (cmd == "neg") {
    const char* nm[] = {"", "SwiGLU silu' dropped", "RoPE forward instead of transpose", "DeltaNet dS decay dropped", "CE softmax dropped", "attention bwd scale dropped"};
    int caught = 0;
    for (int b = 1; b <= 5; ++b) {
      V5Res r = run_v5(false, false, b, false, false);
      const bool c = r.pass < r.tot;
      caught += c;
      std::printf("negative %d (%s): %d / %d checks fail -> %s\n", b, nm[b], r.tot - r.pass, r.tot, c ? "caught" : "NOT caught");
    }
    std::printf("[neg] caught %d / 5\n", caught);
    return caught == 5 ? 0 : 1;
  }
  if (cmd == "v67") {
    const bool real = has("real");
    int B = 2, L = 0;
    if (real) { B = std::atoi(argv[3]); L = std::atoi(argv[4]); }
    return run_v67(real, B, L);
  }
  if (cmd == "opt") return run_opt();
  if (cmd == "dnref") return run_dnref();
  if (cmd == "att") return run_attgpu();
  if (cmd == "dn") return run_dngpu(true);
  if (cmd == "dnbench") return run_dnbench(std::atoi(argv[2]), std::atoi(argv[3]));
  if (cmd == "bench") return run_bench(std::atoi(argv[2]), std::atoi(argv[3]), has("--no-vis"), has("--fp32opt"), 4, false);
  if (cmd == "smoke") return run_bench(std::atoi(argv[2]), std::atoi(argv[3]), has("--no-vis"), false, std::atoi(argv[4]), true);
  return 2;
}
