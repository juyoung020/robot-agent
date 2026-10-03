// CPU 참조판: GPU 커널과 같은 map_block(map.h)을 tid 0, nt 1 로 부른다. 판끼리는 독립이라 판 단위로만 나눠 돈다(OpenMP).
#include "map_api.h"

namespace gmap {

struct NoSync { void operator()() const {} };

CpuMap::CpuMap(int N_, uint64_t seed) : N(N_) {
  h.core.resize(N);
  h.L.assign((size_t)NCELL * N, 0);
  h.seen.assign((size_t)NWORD * N, 0u);
  h.met.assign((size_t)N_MET * N, 0.f);
  for (int i = 0; i < N; ++i) init_core(h.core[i], seed, i);
}

void CpuMap::step(const env::Soa& s, int force_kf) {
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N; ++i) {
    Scratch sh;
    const EnvView e = read_env(s, i);
    map_block(h.core[i], sh, e, h.L.data() + (size_t)i * NCELL, h.seen.data() + (size_t)i * NWORD, h.met.data(), N, i, 0, 1, 0, force_kf, NoSync{});
  }
}

}  // namespace gmap
