// objprob 맞춤 시험(CPU, GPU_MAP_PORT.md 9절 V2): 진짜 scenemap 확률 모드(src/scene_graph/scenemap objprob.cpp·bestview.cpp, double)와
// GPU 학습 지도 쪽 규칙(training/RL/map — 같은 소스가 GPU 커널과 비트 같음: map_verify)을 같은 합성 입력 열에 넣고 맞춘다. 잡음 없음.
//   objprob_parity [n=200000] [--negative]
// 맞추는 것(모두 objprob_math.h 를 두 쪽이 부름 — 남는 차이는 double 대 float·쪽마다 다른 자료 배치):
//   1 같은 것 확률(관측 ↔ 물체, 물체 ↔ 물체): 상자 둘 + cos + 접촉 → P, 문턱 판정 같음(문턱 1e-4 안 제외)
//   2 모습 신뢰도 κ(viewKappa): 상대 차
//   3 이름 사후(apName): 라벨 수 ≤ 6(GPU 가 다 듦 — 근사 없음) — 고른 이름·사후·상위어
//   4 칼만 열(관측 20 번): 자리 차
//   5 이름 분포 겹침: GPU 상위 6 + 나머지 묶음 근사 대 전체(라벨 10 — 근사 오차를 잼, 판정 아님)
// --negative: GPU 쪽 같은 것 로지스틱에서 cos 특징을 빼고 셈 → 1 이 실패해야 정상
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "map.h"
#include "scenemap/bestview.hpp"
#include "scenemap/objprob.hpp"

using namespace gmap;

static uint64_t rs = 0x1234567ull;
static double U() { return (double)(splitmix64(rs) >> 11) * (1.0 / 9007199254740992.0); }
static double R(double a, double b) { return a + (b - a) * U(); }

int main(int argc, char** argv) {
  int n = 200000;
  bool neg = false;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) neg = true;
    else n = std::atoi(argv[a]);
  }
  int fails = 0;
  // ---- 1 같은 것 확률 ----
  {
    scenemap::ApParams P;   // 엔진 json(yolo26n-seg-obj-416) 무게 = MP 값
    const double w[8] = {MP::ap_w0, MP::ap_w1, MP::ap_w2, MP::ap_w3, MP::ap_w4, MP::ap_w5, MP::ap_w6, MP::ap_w7};
    const double wm[8] = {MP::ap_wm0, MP::ap_wm1, MP::ap_wm2, MP::ap_wm3, MP::ap_wm4, MP::ap_wm5, MP::ap_wm6, MP::ap_wm7};
    for (int k = 0; k < 8; ++k) { P.w[k] = w[k]; P.wm[k] = wm[k]; }
    P.cos0 = MP::ap_cos0;
    double maxd = 0;
    long dis = 0, near_thr = 0;
    for (int i = 0; i < n; ++i) {
      double alo[3], ahi[3], apos[3], blo[3], bhi[3], bpos[3];
      for (int k = 0; k < 3; ++k) {
        const double c = R(-1, 1), e = R(0.02, k == 2 ? 1.2 : 1.6);
        blo[k] = c - 0.5 * e; bhi[k] = c + 0.5 * e; bpos[k] = R(blo[k], bhi[k]);
        const double ca = c + R(-0.6, 0.6), ea = R(0.02, 1.2);
        alo[k] = ca - 0.5 * ea; ahi[k] = ca + 0.5 * ea; apos[k] = R(alo[k], ahi[k]);
      }
      const bool merge = (i & 1) != 0;
      const double cs = (i % 7 == 0) ? -2.0 : R(0.3, 1.0), contact = R(0, 1), f6 = merge ? R(-0.5, 0.5) : 0.0;
      // scenemap(double)
      scenemap::ApPair q;
      q.f[0] = contact;
      opm::pair_geo<scenemap::OpmStd>(alo, ahi, apos, blo, bhi, bpos, cs, P.cos0, q.f);
      q.f[6] = f6;
      const double pc = opm::sigmoid<scenemap::OpmStd>(scenemap::apLogit(q, P, merge));
      // GPU 지도(float, 같은 소스가 장치에서 비트 같음)
      float fa[3], fb[3], fp[3], ga[3], gb[3], gp[3], f[7];
      for (int k = 0; k < 3; ++k) { fa[k] = (float)alo[k]; fb[k] = (float)ahi[k]; fp[k] = (float)apos[k]; ga[k] = (float)blo[k]; gb[k] = (float)bhi[k]; gp[k] = (float)bpos[k]; }
      const float pg = sigm_d(ap_feat_logit(fa, fb, fp, ga, gb, gp, neg ? -2.f : (float)cs, (float)contact, merge, (float)f6, f));
      const double d = std::fabs(pc - (double)pg);
      maxd = std::max(maxd, d);
      const double thr = merge ? MP::ap_merge_p : MP::ap_same_p;
      if (std::fabs(pc - thr) < 1e-4) { ++near_thr; continue; }
      dis += (pc >= thr) != ((double)pg >= thr);
    }
    const bool ok = dis == 0 && maxd < 1e-4;
    std::printf("1 same-object P: %d cases, max |P_scenemap − P_gpu| %.2e, threshold decisions differ %ld (skipped %ld within 1e-4 of threshold)  %s\n", n, maxd, dis,
                near_thr, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  // ---- 2 κ ----
  {
    scenemap::KappaParams kp;
    kp.k0 = MP::kap_k0; kp.s0 = MP::kap_s0; kp.trunc = 1.0; kp.d0 = MP::kap_d0; kp.occ = 1.0; kp.blur_w = 0.0;
    double maxr = 0;
    for (int i = 0; i < n; ++i) {
      const float npx = (float)R(4, 200000), dep = (float)R(0.2, 6);
      scenemap::ViewQuality vq;
      vq.size_px = std::sqrt(npx); vq.trunc = false; vq.depth_m = dep; vq.vis = 1; vq.cam_w = 0;
      const double kc = scenemap::viewKappa(vq, kp), kg = view_kappa(npx, dep);
      maxr = std::max(maxr, std::fabs(kc - kg) / kc);
    }
    const bool ok = maxr < 1e-5;
    std::printf("2 view kappa: %d cases, max relative diff %.2e  %s\n", n, maxr, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  // ---- 3 이름 사후(라벨 6, 상위어 둘) ----
  {
    constexpr int C = 7;   // 0..3 낱말 있는 라벨, 4·5 상위어(0·1 → 4, 2 → 5), 6 = "object"(못 정함 — GPU 는 −1)
    scenemap::ApText T;
    T.dim = 1; T.text = {1.f}; T.row_label = {0}; T.n_labels = C;
    T.parent = {4, 4, 5, -1, -1, -1, -1};
    T.object_label = 6;
    scenemap::ApParams P;
    P.name_tau = MP::ap_name_tau; P.name_wmax = MP::ap_name_wmax;
    // GPU 쪽 맥락: 상위어 표만 쓰는 장면 묶음
    bsc::SceneSet ss{};
    int16_t hyper[C] = {4, 4, 5, -1, -1, -1, -1};
    ss.hyper = hyper; ss.nname = C;
    BMapEnv bm{};
    bm.on = 1;
    BCtx bx{};
    bx.on = 1; bx.ss = &ss; bx.bm = &bm;
    MapCore m{};
    m.nlab = C;
    long bad_name = 0, hyp = 0;
    double maxp = 0;
    for (int i = 0; i < n / 4; ++i) {
      scenemap::ApState s;
      s.L_frag.assign(C, 0.f);
      const double lw = R(0.1, 12);
      s.lw_frag = lw;
      Slot S{};
      ap_init(S);
      S.lw = (float)lw;
      for (int c = 0; c < C; ++c) {
        const float L = c >= 4 ? -1e30f : (float)(R(-3, 0) * lw);   // 4·5 = 낱말 없는 상위어 라벨(scenemap: 직접 이름 안 됨)
        s.L_frag[c] = L;
        if (c < 4) { S.lab[c] = (int16_t)c; S.L[c] = L; }
      }
      S.Lrest = -1e30f;   // 상위어 라벨 둘 = GPU 의 나머지 묶음(직접 이름 안 됨)
      scenemap::apName(s, T, P, 0.5);
      m.nlab = 4;   // GPU: 낱말 있는 라벨 4 개 다 듦, 나머지 0
      ap_name(S, m, bx);
      const int nc = s.name_lab == 6 ? -1 : s.name_lab, ng = S.cls;
      bad_name += nc != ng;
      hyp += s.rolled;
      maxp = std::max(maxp, std::fabs((double)s.name_p - (double)S.name_p));
    }
    const bool ok = bad_name == 0 && maxp < 1e-4;
    std::printf("3 name posterior: %d cases, chosen name differs %ld (hypernym chosen %ld), max |name_p diff| %.2e  %s\n", n / 4, bad_name, hyp, maxp, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  // ---- 4 칼만 열 ----
  {
    double maxd = 0;
    for (int i = 0; i < n / 20; ++i) {
      double Pc = R(1e-4, 0.05), xc = R(-1, 1);
      float Pg = (float)Pc, xg = (float)xc;
      for (int t = 0; t < 20; ++t) {
        const double z = R(0.3, 3), o = xc + R(-0.05, 0.05), dt = R(0, 2), e = R(0.02, 0.5);
        const bool tr = U() < 0.2;
        const double K = opm::kalman_gain(Pc, 1e-4, dt, opm::kalman_R(0.02, 0.01, z, tr, e));
        xc += K * (o - xc);
        const float Kg = opm::kalman_gain(Pg, MP::ap_q_pos, (float)dt, opm::kalman_R(MP::ap_r0, MP::ap_r1, (float)z, tr, (float)e));
        xg = xg + Kg * ((float)o - xg);
      }
      maxd = std::max(maxd, std::fabs(xc - (double)xg));
    }
    const bool ok = maxd < 1e-5;
    std::printf("4 Kalman 20 updates: %d sequences, max position diff %.2e m  %s\n", n / 20, maxd, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  // ---- 5 이름 분포 겹침(근사 오차) ----
  {
    constexpr int C = 10;
    double maxd = 0, sum = 0;
    for (int i = 0; i < n / 10; ++i) {
      float pa[C], pb[C];
      double za = 0, zb = 0;
      for (int c = 0; c < C; ++c) { pa[c] = (float)std::pow(U(), 4); pb[c] = (float)std::pow(U(), 4); za += pa[c]; zb += pb[c]; }
      for (int c = 0; c < C; ++c) { pa[c] = (float)(pa[c] / za); pb[c] = (float)(pb[c] / zb); }
      const double full = opm::bhattacharyya<scenemap::OpmStd, double>(pa, pb, C, 0.0, 0.0, 0);
      // GPU 근사: 큰 것 6 개 + 나머지 4 개를 같은 값(평균)으로
      Slot A{}, B{};
      ap_init(A); ap_init(B);
      auto top6 = [&](const float* p, Slot& S) {
        int idx[C];
        for (int c = 0; c < C; ++c) idx[c] = c;
        for (int a2 = 0; a2 < C; ++a2) for (int b2 = a2 + 1; b2 < C; ++b2) if (p[idx[b2]] > p[idx[a2]]) { const int t = idx[a2]; idx[a2] = idx[b2]; idx[b2] = t; }
        float rest = 0.f;
        for (int k = 0; k < C; ++k) { if (k < NLAB) { S.lab[k] = (int16_t)idx[k]; S.post[k] = p[idx[k]]; } else rest += p[idx[k]]; }
        S.prest = rest / (float)(C - NLAB);
      };
      top6(pa, A);
      top6(pb, B);
      const double appr = ap_bhat(A, B, C);
      maxd = std::max(maxd, std::fabs(full - appr));
      sum += std::fabs(full - appr);
    }
    std::printf("5 name-distribution overlap (top-6 + rest approximation vs full 10 labels): mean |diff| %.4f, max %.4f  (measure, not a pass/fail)\n",
                sum / (n / 10), maxd);
  }
  std::printf(fails ? "FAIL: %d checks\n" : "OK: objprob rules match scenemap (%d checks failed)\n", fails);
  return neg ? (fails ? 0 : 1) : (fails ? 1 : 0);
}
