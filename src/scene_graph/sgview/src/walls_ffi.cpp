// C 연결(Rust FFI) — scenemap/walls.hpp 를 그대로 부른다. 격자는 sm_grid 배치(행 0 = 최소 y, 0..100 점유 %, −1 모름).
#include <cstdint>
#include <vector>

#include "scenemap/walls.hpp"

extern "C" {
// 선분 개수를 돌려준다(cap 보다 클 수 있음). out: 선분당 ax ay bx by (map, m)
// ignore: 벽으로 치지 않을 영역 n_ignore 개 (x0 y0 x1 y1 …, map m) — 바닥에 놓인 가구
int sgv_wall_segments(const int8_t* cells, int w, int h, double res, double ox, double oy, const double* ignore, int n_ignore, double* out, int cap) {
  scenemap::WallGrid g{cells, w, h, res, ox, oy};
  std::vector<scenemap::WallRect> ig(n_ignore);
  for (int i = 0; i < n_ignore; ++i) ig[i] = {ignore[4 * i], ignore[4 * i + 1], ignore[4 * i + 2], ignore[4 * i + 3]};
  const auto segs = scenemap::wallSegments(g, scenemap::kMinLen, scenemap::kMaxThick, 0.6, &ig);
  for (int i = 0; i < cap && i < int(segs.size()); ++i) {
    out[4 * i] = segs[i].ax; out[4 * i + 1] = segs[i].ay; out[4 * i + 2] = segs[i].bx; out[4 * i + 3] = segs[i].by;
  }
  return int(segs.size());
}

// 선분(map)과 로봇 자세로 벽 상태 벡터(길이 56) 를 만든다
void sgv_wall_state(const int8_t* cells, int w, int h, double res, double ox, double oy, const double* segs, int n,
                    const double pose[3], float* out) {
  scenemap::WallGrid g{cells, w, h, res, ox, oy};
  std::vector<scenemap::WallSeg> v(n);
  for (int i = 0; i < n; ++i) v[i] = {segs[4 * i], segs[4 * i + 1], segs[4 * i + 2], segs[4 * i + 3]};
  scenemap::wallStateVector(g, v, pose, out);
}
}
