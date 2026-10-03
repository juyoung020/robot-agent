// C ABI 재생 시험: ep_*.bin 을 sm_push_proprio(매 프레임) · sm_push_image(keyframe, 깊이 m) 로 넣고
// keyframe 마다 sm_snapshot 자세를 slam2d_eval 결과(<prefix>_est.bin)와 비교한다. 격자 크기·reachable 도 한 번 찍는다.
//
//   capi_replay <ep.bin> <slam2d_eval est.bin>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "scenemap.h"

#pragma pack(push, 1)
struct Row {
  float qvel[3], eefL[3], eefR[3], gl, gr;
  double gt[3];
  float prop[61];
};
#pragma pack(pop)

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: capi_replay <ep.bin> <est.bin>\n"); return 2; }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) return 1;
  char magic[4];
  uint32_t hdr[5];
  float K[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5 || std::fread(K, 4, 4, f) != 4 || hdr[0] != 2) return 1;
  const uint32_t n = hdr[1], w = hdr[3], h = hdr[4];
  std::vector<Row> rows(n);
  if (std::fread(rows.data(), sizeof(Row), n, f) != n) return 1;
  uint32_t m;
  if (std::fread(&m, 4, 1, f) != 1) return 1;
  std::vector<double> ref(size_t(n) * 3);
  FILE* e = std::fopen(argv[2], "rb");
  if (!e || std::fread(ref.data(), 8, ref.size(), e) != ref.size()) { std::fprintf(stderr, "est.bin\n"); return 1; }
  std::fclose(e);

  sm_ctx* c = sm_create(nullptr);
  std::vector<uint16_t> mm(size_t(w) * h);
  std::vector<float> dm(mm.size());
  float T[12];
  uint32_t kf = 0, next = 0, done = 0;
  auto readKf = [&]() -> bool {
    if (done >= m) return false;
    if (std::fread(&next, 4, 1, f) != 1 || std::fread(T, 4, 12, f) != 12 || std::fread(mm.data(), 2, mm.size(), f) != mm.size())
      return false;
    ++done;
    return true;
  };
  bool have = readKf();
  double worst = 0, snap_us = 0;
  int nsnap = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const double stamp = i / 30.0;
    sm_proprio p{stamp, rows[i].prop, 61};
    sm_push_proprio(c, &p);
    if (have && next == i) {
      for (size_t k = 0; k < mm.size(); ++k) dm[k] = mm[k] * 1e-3f;
      sm_image im{stamp, 0, int(w), int(h), nullptr, dm.data(), K[0], K[1], K[2], K[3]};
      sm_push_image(c, &im, nullptr);
      const auto t0 = std::chrono::steady_clock::now();
      sm_snapshot_t* s;
      sm_snapshot(c, &s);
      snap_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      ++nsnap;
      const sm_pose2 q = sm_snap_pose(s);
      const double d = std::hypot(q.x - ref[i * 3], q.y - ref[i * 3 + 1]);
      worst = std::fmax(worst, d);
      sm_snapshot_release(s);
      ++kf;
      have = readKf();
    }
  }
  sm_snapshot_t* s;
  sm_snapshot(c, &s);
  sm_grid g;
  sm_snap_map(s, &g);
  const sm_pose2 q = sm_snap_pose(s);
  const double from[2] = {q.x, q.y}, to[2] = {0, 0};
  const auto t0 = std::chrono::steady_clock::now();
  const double r = sm_snap_reachable(s, from, to);
  const double r_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  std::printf("keyframes %u, C ABI vs slam2d_eval 최대 위치 차 %.3e m, 스냅숏 평균 %.0f us, 격자 %dx%d res %.2f origin (%.2f, %.2f)\n",
              kf, worst, snap_us / std::max(1, nsnap), g.width, g.height, g.resolution, g.origin[0], g.origin[1]);
  std::printf("끝 자세 (%.2f, %.2f) → 원점 경로 %.2f m (직선 %.2f m), A* %.0f us\n", q.x, q.y, r, std::hypot(q.x, q.y), r_us);
  sm_snapshot_release(s);
  sm_destroy(c);
  std::fclose(f);
  return worst < 1e-6 ? 0 : 1;
}
