// sgrt 끝까지(실제 엔진·장치 RGB): 실제 머리 영상 프레임(raw RGB)들을 장치에 올려 sgrt_step(keyframe 마다 영상)으로 넣고
// 저장까지. 깊이는 없어서 평평한 2 m(모양·자리는 가짜 — 검출 → best view 자르기·구름 색 모으기·PNG/PLY 저장 경로 확인용).
//   sgrt_frames <engine.plan> <out_dir> <w> <h> <frames.rgb ...>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sgrt.h"

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: sgrt_frames <engine.plan> <out_dir> <w> <h> <frames.rgb ...>\n");
    return 2;
  }
  const std::string eng = argv[1], names = eng + ".names.txt";
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  sgrt_config c;
  sgrt_default_config(&c);
  c.engine = eng.c_str();
  c.names = names.c_str();
  c.out_dir = argv[2];
  c.kf_every = 1;
  c.save_s = 1e9;
  char err[1024] = {0};
  sgrt* s = sgrt_create(&c, err, sizeof err);
  if (!s) { std::fprintf(stderr, "create: %s\n", err); return 1; }
  const char* prompt[] = {"radio receiver", "sofa", "table", "cup"};   // 닫힌 어휘면 auto → 어휘 전부
  sgrt_begin(s, prompt, 4, err, sizeof err);
  if (err[0]) std::printf("prompt: %s\n", err);
  std::vector<float> q(61, 0.f), depth(size_t(W) * H, 2.0f);
  q[17] = q[42] = -2.f;
  q[24] = q[25] = q[49] = q[50] = 0.05f;
  std::vector<uint8_t> rgb(size_t(W) * H * 3);
  uint8_t* d_rgb = nullptr;
  cudaMalloc(&d_rgb, rgb.size());
  double kf = 0, crop = 0, gather = 0, det = 0;
  int n = 0, npts = 0, ncrop = 0;
  for (int f = 5; f < argc; ++f) {
    FILE* fp = std::fopen(argv[f], "rb");
    if (!fp || std::fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) { std::fprintf(stderr, "read %s\n", argv[f]); return 1; }
    std::fclose(fp);
    cudaMemcpy(d_rgb, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
    for (int rep = 0; rep < 2; ++rep) {   // 같은 프레임 두 번(물체 확정 = 서로 다른 keyframe 2 번)
      const double t = (2 * (f - 5) + rep) / 30.0;
      sgrt_step(s, t, q.data(), 61, d_rgb, 1, int64_t(W) * 3, 3, W, H, depth.data(), 306, 306, W / 2.0, H / 2.0);
      sgrt_timing tm;
      sgrt_get_timing(s, &tm);
      det += tm.det_ms; kf += tm.kf_ms; crop += tm.crop_ms; gather += tm.gather_ms;
      npts += tm.n_points; ncrop += tm.n_crops;
      ++n;
    }
  }
  const int rc = sgrt_save(s);
  sgrt_timing tm;
  sgrt_get_timing(s, &tm);
  int32_t nkf, ndet, nobj;
  float dms, sms;
  sgrt_stats(s, &nkf, &ndet, &nobj, &dms, &sms);
  std::printf("keyframes %d, objects %d, mean det %.2f ms, scenemap kf %.2f ms (crop %.3f ms ×%.1f, gather %.3f ms ×%.0f pts), save rc %d %.2f ms "
              "(png %d, ply %d)\n",
              n, nobj, det / n, kf / n, crop / n, double(ncrop) / n, gather / n, double(npts) / n, rc, tm.save_ms, tm.n_png, tm.n_ply);
  cudaFree(d_rgb);
  sgrt_destroy(s);
  return rc == 0 ? 0 : 1;
}
