// CPU 참조판: GPU 커널과 같은 map_block(map.h)을 tid 0, nt 1 로 부른다. 판끼리는 독립이라 판 단위로만 나눠 돈다(OpenMP).
#include <cstring>

#include "map_api.h"

namespace gmap {

struct NoSync { void operator()() const {} bool any(bool v) const { return v; } };

CpuMap::CpuMap(int N_, uint64_t seed, const bsc::SceneSet* ss_host) : N(N_), ss(ss_host) {
  if (ss) {
    h.bm.assign((size_t)N, BMapEnv{});
    h.lev.assign((size_t)NAV_P * NAV_P * N, 255);
    h.navorg.assign(N, 0);
    h.navtag.assign(N, -1);
    h.navconf.assign(N, 0);
  }
  h.core.resize(N);
  h.L.assign((size_t)NCELL * N, 0);
  h.seen.assign((size_t)NWORD * N, 0u);
  h.met.assign((size_t)N_MET * N, 0.f);
  h.occ.assign((size_t)NWORD * N, 0u);
  h.segs.assign((size_t)SEGW * N, 0);
  h.tprev.assign((size_t)KSLOT * N, TPrev{{0.f, 0.f, 0.f}, 0});
  h.tok.assign((size_t)N, MapTok{});
  h.view.assign((size_t)VIEW_BYTES * N, 0);
  for (int i = 0; i < N; ++i) init_core(h.core[i], seed, i);
}

void CpuMap::step(const env::Soa& s, int force_kf, const MapCurr& cu) {
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N; ++i) {
    KfShared u;
    const EnvView e = read_env(s, i, ss != nullptr);
    const MapGrid g{h.L.data() + (size_t)i * NCELL, h.seen.data() + (size_t)i * NWORD, h.occ.data() + (size_t)i * NWORD, h.segs.data() + (size_t)i * SEGW,
                      h.view.data() + (size_t)i * VIEW_BYTES};
    BMapEnv* bm = ss ? &h.bm[i] : nullptr;
    map_block(h.core[i], u, e, g, h.met.data(), N, i, 0, 1, 0, force_kf, cu, NoSync{}, ss, bm);
    TokScratch ts;
    const BCtx bx = bctx(ss, bm);
    make_tokens_n<1>(h.core[i], g.occ, g.seen, g.segs, h.tprev.data() + (size_t)i * KSLOT, ts, 0, 1, true, NoSync{}, 0, &bx);
    h.tok[i] = ts.out;
    if (bx.on) {   // 다가가기 거리장(GPU map_nav_kernel 과 같은 조건·같은 단계 집합)
      const MapCore& m = h.core[i];
      h.navconf[i] = m.n_task_conf;
      if (nav_due(m, h.navtag[i], cu)) {
        nav_bfs_ref(g.occ, s.f[env::F_TX * N + i], s.f[env::F_TY * N + i], env::seed_r_of(s.iv[env::I_B_KIND * N + i]), s.f[env::F_X * N + i], s.f[env::F_Y * N + i],
                    nav_period(cu), h.lev.data() + (size_t)i * NAV_P * NAV_P, h.navorg[i]);
        h.navtag[i] = m.ep;
      }
    }
  }
}

void tv_topstate_cpu(const MapHost& h, const bsc::SceneSet* ss, int i, TopState& out) {
  BMapEnv* bm = (ss && !h.bm.empty()) ? const_cast<BMapEnv*>(&h.bm[i]) : nullptr;
  const BCtx bx = bctx(ss, bm);
  tv_topstate(h.core[i], &bx, h.occ.data() + (size_t)i * NWORD, h.seen.data() + (size_t)i * NWORD, out, 0, 1);
}
void tv_render_cpu(const TopState& t, bool hide, uint8_t* rgb) {
  for (int v = 0; v < TV_PX; ++v)
    for (int u = 0; u < TV_PX; ++u) tv_pixel(t, u, v, hide, rgb + ((size_t)v * TV_PX + u) * 3);
}

float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) x = sign;
    else { float f = (float)m * (1.f / 16777216.f); std::memcpy(&x, &f, 4); x |= sign; }
  } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
  else x = sign | ((e + 112) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

}  // namespace gmap
