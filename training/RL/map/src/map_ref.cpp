// CPU 참조판: GPU 커널과 같은 map_block(map.h)을 tid 0, nt 1 로 부른다. 판끼리는 독립이라 판 단위로만 나눠 돈다(OpenMP).
#include <cstring>

#include "map_api.h"

namespace gmap {

struct NoSync { void operator()() const {} bool any(bool v) const { return v; } };

CpuMap::CpuMap(int N_, uint64_t seed) : N(N_) {
  h.core.resize(N);
  h.L.assign((size_t)NCELL * N, 0);
  h.seen.assign((size_t)NWORD * N, 0u);
  h.met.assign((size_t)N_MET * N, 0.f);
  h.occ.assign((size_t)NWORD * N, 0u);
  h.segs.assign((size_t)SEGW * N, 0);
  h.tprev.assign((size_t)KSLOT * N, TPrev{{0.f, 0.f, 0.f}, 0});
  h.tok.assign((size_t)N, MapTok{});
  for (int i = 0; i < N; ++i) init_core(h.core[i], seed, i);
}

void CpuMap::step(const env::Soa& s, int force_kf, const MapCurr& cu) {
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N; ++i) {
    KfShared u;
    const EnvView e = read_env(s, i);
    const MapGrid g{h.L.data() + (size_t)i * NCELL, h.seen.data() + (size_t)i * NWORD, h.occ.data() + (size_t)i * NWORD, h.segs.data() + (size_t)i * SEGW};
    map_block(h.core[i], u, e, g, h.met.data(), N, i, 0, 1, 0, force_kf, cu, NoSync{});
    TokScratch ts;
    make_tokens(h.core[i], g.occ, g.seen, g.segs, h.tprev.data() + (size_t)i * KSLOT, ts, 0, 1, true, NoSync{});
    h.tok[i] = ts.out;
  }
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
