// sgrt 기록(SGRT_RECORD) 을 libsgrt 로 다시 굴리기 — 검출(ovdet)부터 저장까지 sgrt 전체를 같은 입력으로.
//
//   sgrt_replay <libsgrt.so> <engine.plan> <rec.bin> <out_dir> [--frames N] [--kf K] [--traj out.csv]
//
// 라이브러리는 dlopen 으로 읽는다(옛 빌드와 새 빌드를 같은 도구로 비교 — 바이트 회귀 확인용). 옛 ABI 만 쓴다.
// 환경 변수는 라이브러리가 읽는 그대로(SGRT_POSE·SGRT_ROBOT·SGRT_SM_CONFIG·SGRT_SAVE_SYNC …). 비교할 때는 SGRT_SAVE_SYNC=1.
// 재생 순서 = 기록 순서: 'G' → sgrt_push_pose, 'P' → sgrt_step(그 뒤 같은 stamp 'I' 가 오면 그 RGB·깊이·K 를 같이).
// 기록 영상 RGB 는 호스트 RGBA(sgrec) 로 넘긴다. 프롬프트는 엔진 어휘 전부(sgrt_begin(NULL, 0)).
// --traj: keyframe 마다 sgrt_map 자세와 자세 진단(외부 자세가 있으면 그 차)을 CSV 로.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sgrec.hpp"
#include "sgrt.h"

namespace {
template <class F>
F sym(void* h, const char* n) {
  auto* p = reinterpret_cast<F>(dlsym(h, n));
  if (!p) { std::fprintf(stderr, "missing %s\n", n); std::exit(1); }
  return p;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: sgrt_replay <libsgrt.so> <engine.plan> <rec.bin> <out_dir> [--frames N] [--kf K] [--traj out.csv]\n");
    return 2;
  }
  int frames = 0, kf = 6;
  std::string traj;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--frames") frames = std::atoi(nx());
    else if (a == "--kf") kf = std::atoi(nx());
    else if (a == "--traj") traj = nx();
  }
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  auto create = sym<sgrt* (*)(const sgrt_config*, char*, size_t)>(h, "sgrt_create");
  auto begin = sym<int (*)(sgrt*, const char* const*, int32_t, char*, size_t)>(h, "sgrt_begin");
  auto step = sym<int (*)(sgrt*, double, const float*, int32_t, const uint8_t*, int32_t, int64_t, int32_t, int32_t, int32_t, const float*,
                          double, double, double, double)>(h, "sgrt_step");
  auto push_pose = sym<int (*)(sgrt*, double, double, double, double)>(h, "sgrt_push_pose");
  auto save = sym<int (*)(sgrt*)>(h, "sgrt_save");
  auto destroy = sym<void (*)(sgrt*)>(h, "sgrt_destroy");
  auto diag = sym<int (*)(const sgrt*, sgrt_pose_diag*)>(h, "sgrt_get_pose_diag");
  auto map = sym<int (*)(sgrt*, sgrt_map_view*)>(h, "sgrt_map");
  auto stats = sym<void (*)(const sgrt*, int32_t*, int32_t*, int32_t*, float*, float*)>(h, "sgrt_stats");

  std::vector<sgrec::Rec> recs;
  if (!sgrec::load(argv[3], &recs, frames)) { std::fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
  const std::string engine = argv[2], names = engine + ".names.txt";
  sgrt_config cfg{engine.c_str(), names.c_str(), argv[4], kf, 1.0, 0.25f};
  char err[512] = {0};
  sgrt* s = create(&cfg, err, sizeof err);
  if (!s) { std::fprintf(stderr, "sgrt_create: %s\n", err); return 1; }
  begin(s, nullptr, 0, err, sizeof err);

  FILE* tf = traj.empty() ? nullptr : std::fopen(traj.c_str(), "w");
  if (tf) std::fprintf(tf, "stamp,map_x,map_y,map_yaw,diag_n,err_xy,err_yaw,ref_x,ref_y,ref_yaw\n");
  const sgrec::Rec* pend = nullptr;
  size_t n_step = 0, n_img = 0;
  auto run = [&](const sgrec::Rec* p, const sgrec::Rec* im) {
    if (!im) {
      step(s, p->stamp, p->f.data(), int32_t(p->f.size()), nullptr, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0);
    } else {
      const uint8_t* rgb = im->rgba.empty() ? nullptr : im->rgba.data();
      step(s, p->stamp, p->f.data(), int32_t(p->f.size()), rgb, 0, int64_t(im->w) * 4, 4, im->w, im->h, im->f.data(), im->K[0], im->K[1],
           im->K[2], im->K[3]);
      ++n_img;
      if (tf) {
        sgrt_map_view v{};
        sgrt_pose_diag d{};
        map(s, &v);
        diag(s, &d);
        std::fprintf(tf, "%.4f,%.5f,%.5f,%.6f,%d,%.5f,%.6f,%.5f,%.5f,%.6f\n", p->stamp, v.pose[0], v.pose[1], v.pose[2], d.n, d.last_xy,
                     d.last_yaw, d.ref[0], d.ref[1], d.ref[2]);
      }
    }
    ++n_step;
  };
  for (const auto& r : recs) {
    if (r.tag == 'P') {
      if (pend) run(pend, nullptr);
      pend = &r;
    } else if (r.tag == 'I') {
      if (pend && pend->stamp == r.stamp) run(pend, &r);
      pend = nullptr;
    } else if (r.tag == 'G') {
      if (pend) run(pend, nullptr);
      pend = nullptr;
      push_pose(s, r.stamp, r.g[0], r.g[1], r.g[2]);
    }
  }
  if (pend) run(pend, nullptr);
  if (tf) std::fclose(tf);
  save(s);
  sgrt_pose_diag d{};
  diag(s, &d);
  int32_t nkf = 0, ndet = 0, nobj = 0;
  float dms = 0, sms = 0;
  stats(s, &nkf, &ndet, &nobj, &dms, &sms);
  std::printf("replayed %zu steps, %zu keyframes -> %s: %d objects; pose diag n %d rms_xy %.4f max_xy %.4f rms_yaw %.5f max_yaw %.5f\n",
              n_step, n_img, argv[4], nobj, d.n, d.rms_xy, d.max_xy, d.rms_yaw, d.max_yaw);
  destroy(s);
  return 0;
}
