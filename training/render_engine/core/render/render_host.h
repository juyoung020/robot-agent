// 층 1: 호스트(C++) 렌더. 판 하나 × 카메라 여러 대. 픽셀마다 같은 EHD 함수(shade.h)를 부른다 -> 층 2 와 비트 비교.
// 픽셀끼리 독립이라 스레드 수는 결과에 영향 없다.
#pragma once
#include <cstdint>
#include <thread>
#include <vector>
#include <xmmintrin.h>

#include "core/render/denoise.h"
#include "core/render/tlas.h"

namespace eng {
namespace rnd {

EHD uint32_t pixel_seed(int32_t env, int32_t cam, int32_t px, int32_t py, int32_t frame) {
  return pcg(uint32_t(frame) * 0x9E3779B1u ^ pcg(uint32_t(env) * 0x85EBCA77u ^ pcg(uint32_t(cam) * 0xC2B2AE3Du ^
             pcg(uint32_t(py) * 0x27D4EB2Fu ^ uint32_t(px)))));
}

// 픽셀 하나: depth(f32) 와 RGB(u8 ×3). 층 1·2 가 똑같이 부르는 입구.
EHD void render_pixel(const SceneView& S, const EnvView& E, const Camera& c, int32_t env, int32_t cam, int32_t frame,
                      int px, int py, float* depth_out, uint8_t* rgb_out) {
  float d;
  V3 L;
  shade_pixel(S, E, c, px, py, pixel_seed(env, cam, px, py, frame), d, L);
  depth_out[py * c.w + px] = d;
  tonemap(S.sp, L, rgb_out + 3 * (py * c.w + px));
}

struct HostEnv {
  std::vector<Aff> anchor, light_world, inst_world, inst_inv;
  std::vector<float> inst_box;
  std::vector<Node2> tlas;
  std::vector<int32_t> order;
  std::vector<uint32_t> vis;
  std::vector<uint64_t> keys;
  std::vector<float> nodebox;
  std::vector<int32_t> kids;
  void resize(const SceneView& S) {
    anchor.resize(S.n_anchor);
    light_world.resize(S.n_lights > 0 ? S.n_lights : 1);
    inst_world.resize(S.n_inst);
    inst_inv.resize(S.n_inst);
    inst_box.resize(6 * size_t(S.n_inst));
    tlas.assign(S.n_inst > 1 ? S.n_inst - 1 : 1, Node2{});
    order.resize(S.n_inst > 0 ? S.n_inst : 1);
    vis.assign((S.n_inst + 31) / 32, 0xFFFFFFFFu);
  }
  EnvView view() {
    return EnvView{anchor.data(), vis.data(), light_world.data(), inst_world.data(), inst_inv.data(), inst_box.data(),
                   tlas.data(), order.data()};
  }
  void build(const SceneView& S) {
    EnvView v = view();
    const unsigned old = _mm_getcsr();  // GPU -ftz=true 와 같게 FTZ+DAZ
    _mm_setcsr(old | 0x8040u);
    tlas_build_host(S, v, keys, nodebox, kids);
    _mm_setcsr(old);
  }
};

// 층 1 카메라 한 대: (1) G 버퍼·조도 (2) à-trous sp.denoise 번 (3) 합치기·톤매핑. 단계 사이는 스레드 합류(= GPU 커널 경계).
// lin != nullptr 이면 톤매핑 전 선형 휘도(픽셀당 3)도 적는다(노출 맞추기용, 비트 비교 대상 아님).
inline void render_host(const SceneView& S, HostEnv& E, const Camera& c, int32_t env, int32_t cam, int32_t frame,
                        float* depth, uint8_t* rgb, int nthreads = 0, float* lin = nullptr) {
  if (nthreads <= 0) nthreads = int(std::thread::hardware_concurrency());
  const EnvView ev = E.view();
  const int n = c.w * c.h;
  std::vector<GPix> G(n);
  std::vector<float> A(size_t(n) * 6), B(size_t(n) * 6);
  auto par = [&](auto&& fn) {
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t)
      th.emplace_back([&, t] {
        const unsigned old = _mm_getcsr();  // GPU -ftz=true 와 같게: FTZ(0x8000)+DAZ(0x40)
        _mm_setcsr(old | 0x8040u);
        for (int py = t; py < c.h; py += nthreads)
          for (int px = 0; px < c.w; ++px) fn(px, py);
        _mm_setcsr(old);
      });
    for (auto& x : th) x.join();
  };
  par([&](int px, int py) {
    const int i = py * c.w + px;
    shade_gbuf(S, ev, c, px, py, pixel_seed(env, cam, px, py, frame), depth[i], G[i], &A[6 * size_t(i)]);
  });
  float* in = A.data();
  float* out = B.data();
  for (int it = 0; it < S.sp.denoise; ++it) {
    par([&](int px, int py) { atrous_pixel(c, G.data(), in, out, px, py, 1 << it, S.sp.dn_plane); });
    float* t = in; in = out; out = t;
  }
  par([&](int px, int py) { compose_pixel(S.sp, G.data(), in, py * c.w + px, rgb); });
  if (lin)
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < 3; ++k) lin[3 * i + k] = G[i].a[k] * in[6 * i + k] + in[6 * i + 3 + k];
}

}  // namespace rnd
}  // namespace eng
