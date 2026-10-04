// 224² 렌더 속도(계획서 4.5·10.3 의 6, 12절 열린 문제): 팀 RenderBatch(src/behavior-2026/src/sim/engine/cuda/render, **읽기만** — include 로 씀)로
// G1 A2 장면(방 벽 4 + 바닥 + 가구 상자 8 + 컵)을 판 E 개 × 카메라 2 대(머리 RGB-D 67.9°, 손목) 224² 로 그린다.
//   render_bench [E=256] [reps=10] [--dump DIR]
// 장면 상태는 GPU 환경(DeviceEnv, A2)을 몇 스텝 돌린 판들에서 가져온다(BC 기록의 RenderState 와 같은 값: 자세·컵·방·가구).
// 상자 하나 = 인스턴스 하나 = 기준 prim 하나(판마다 축척을 행렬에 넣음 — set_anchors), 안 쓰는 가구는 보임 비트로 끔.
// 두 설정: 팀 기본(spp 1, 튕김 1, 반사 1, 잡음 제거 4 — test_render_synth 기본값)과 싼 설정(튕김 0, 반사 0, 잡음 제거 0).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/render/render_batch.cuh"
#include "env_api.h"

using namespace eng::rnd;

static void push_tri(std::vector<float>& tv, std::vector<float>& tn, std::vector<float>& tu, std::vector<int32_t>& ts, const float* a, const float* b,
                     const float* c, const float* n, int slot) {
  const float* p[3] = {a, b, c};
  const float uvs[6] = {0, 0, 1, 0, 1, 1};
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tv.push_back(p[k][q]);
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tn.push_back(n[q]);
  for (int k = 0; k < 6; ++k) tu.push_back(uvs[k]);
  ts.push_back(slot);
}
static int32_t make_box(HostScene& S) {   // 단위 상자(중심 원점, 변 1)
  std::vector<float> tv, tn, tu;
  std::vector<int32_t> ts;
  const float P[8][3] = {{-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {.5f, .5f, -.5f}, {-.5f, .5f, -.5f}, {-.5f, -.5f, .5f}, {.5f, -.5f, .5f}, {.5f, .5f, .5f}, {-.5f, .5f, .5f}};
  const int F[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
  const float Nn[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {1, 0, 0}, {-1, 0, 0}};
  for (int f = 0; f < 6; ++f) {
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][1]], P[F[f][2]], Nn[f], 0);
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][2]], P[F[f][3]], Nn[f], 0);
  }
  return add_geometry(S, tv.data(), tn.data(), tu.data(), ts.data(), uint32_t(ts.size()), 3);
}
static Aff aff_box(float cx, float cy, float cz, float sx, float sy, float sz) { return Aff{{sx, 0, 0, cx, 0, sy, 0, cy, 0, 0, sz, cz}}; }
static Aff aff_yaw(float x, float y, float yaw) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  return Aff{{c, -s, 0, x, s, c, 0, y, 0, 0, 1, 0}};
}

constexpr int I_FLOOR = 0, I_WALL = 1, I_FURN = 5, I_CUP = 13, N_INST = 14, A_ROBOT = 14, N_ANCH = 15;

static HostScene make_scene(int profile) {
  HostScene H;
  const int gbox = make_box(H);
  const float col[7][3] = {{0.55f, 0.52f, 0.48f}, {0.85f, 0.82f, 0.75f}, {0.75f, 0.15f, 0.12f}, {0.45f, 0.30f, 0.18f}, {0.35f, 0.40f, 0.55f},
                           {0.30f, 0.55f, 0.35f}, {0.60f, 0.60f, 0.20f}};   // 바닥, 벽, 컵, 가구 종류 4 개 (가정: 색)
  for (auto& c : col) {
    Material M{};
    M.albedo[0] = c[0]; M.albedo[1] = c[1]; M.albedo[2] = c[2];
    M.tex_albedo = -1;
    M.uv_scale[0] = M.uv_scale[1] = 1.0f;
    M.albedo_brightness = 1.0f;
    M.opacity = 1.0f;
    M.roughness = 0.6f;
    H.mats.push_back(M);
  }
  H.n_anchor = N_ANCH;
  auto add_inst = [&](int anc, int mat) {
    InstInfo in{};
    in.geom = gbox; in.anchor = anc; in.slot_base = int32_t(H.slot_mat.size());
    in.rel = Aff{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};
    H.slot_mat.push_back(mat);
    H.insts.push_back(in);
  };
  add_inst(0, 0);
  for (int k = 0; k < 4; ++k) add_inst(1 + k, 1);
  for (int k = 0; k < 8; ++k) add_inst(5 + k, 3 + k % 4);
  add_inst(13, 2);
  Light L{};
  L.type = kLightDistant; L.anchor = -1; L.visible = 1;
  L.radiance[0] = L.radiance[1] = L.radiance[2] = 3000.f;
  L.angle = 0.53f;
  L.rel = Aff{{1, 0, 0, 0, 0, 0.8f, -0.6f, 0, 0, 0.6f, 0.8f, 0}};
  H.lights.push_back(L);
  Light S{};
  S.type = kLightSphere; S.anchor = -1; S.visible = 1;
  S.radiance[0] = 600.f; S.radiance[1] = 560.f; S.radiance[2] = 500.f;
  S.radius = 0.15f;
  S.rel = Aff{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.4f}};
  H.lights.push_back(S);
  H.sp.ambient[0] = H.sp.ambient[1] = H.sp.ambient[2] = 0.08f;
  H.sp.exposure = 1.0f; H.sp.spp = 1; H.sp.shadow_lights = 1; H.sp.tonemap = 2;
  H.sp.white_scale = 8.0f; H.sp.dome_light = -1; H.sp.dn_plane = 0.01f; H.sp.clamp_ind = 0.0f; H.sp.tex_aniso = 2; H.sp.lod_bias = -0.5f; H.sp.spec_f0 = 0.04f;
  if (profile == 0) { H.sp.bounces = 1; H.sp.denoise = 4; H.sp.spec = 1; }   // 팀 기본(test_render_synth)
  else { H.sp.bounces = 0; H.sp.denoise = 0; H.sp.spec = 0; }                // 싼 설정
  return H;
}

// 카메라 rel(로봇 기준): 열 = 오른쪽, 위, 뒤(−시선), 위치
static CamRig rig(float px, float pz, float pitch_down, float hfov, int w, int h) {
  const float c = std::cos(pitch_down), s = std::sin(pitch_down);
  // 시선 f = (c, 0, −s), 오른쪽 r = (0, −1, 0), 위 u = r × f = (s, 0, c), 뒤 = −f
  CamRig g{};
  g.anchor = A_ROBOT; g.w = w; g.h = h;
  g.rel = Aff{{0, s, -c, px, -1, 0, 0, 0, 0, c, s, pz}};
  g.tanx = std::tan(hfov * 0.5f);
  g.tany = g.tanx * (float)h / (float)w;
  g.znear = 0.01f; g.zfar = 50.f;
  return g;
}

int main(int argc, char** argv) {
  int E = 256, reps = 10;
  std::string dump;
  std::vector<int> pos;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else pos.push_back(std::atoi(argv[i]));
  }
  if (pos.size() > 0) E = pos[0];
  if (pos.size() > 1) reps = pos[1];
  // 장면 상태: A2 환경을 행동 0 으로 몇 스텝(판마다 다른 방·가구·컵, 리셋 직후 자세)
  env::DeviceEnv de(E, 2, 4242);
  float *act, *obs, *rew;
  int* done;
  cudaMalloc(&act, sizeof(float) * 8 * E); cudaMemset(act, 0, sizeof(float) * 8 * E);
  cudaMalloc(&obs, sizeof(float) * env::N_OBS * E); cudaMalloc(&rew, sizeof(float) * E); cudaMalloc(&done, sizeof(int) * E);
  for (int k = 0; k < 3; ++k) de.step(act, obs, rew, done);
  std::vector<float> f;
  std::vector<int> iv;
  std::vector<uint64_t> rng;
  de.download(f, iv, rng);
  std::vector<Aff> anch((size_t)E * N_ANCH);
  std::vector<uint32_t> vis((size_t)E, 0);
  for (int e = 0; e < E; ++e) {
    auto F = [&](int k) { return f[(size_t)k * E + e]; };
    const float rx = F(env::F_RHX), ry = F(env::F_RHY), wt = 0.1f, wh = 1.0f;
    Aff* a = &anch[(size_t)e * N_ANCH];
    a[0] = aff_box(0, 0, -0.05f, 2 * rx + 1, 2 * ry + 1, 0.1f);
    a[1] = aff_box(rx + wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
    a[2] = aff_box(-rx - wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
    a[3] = aff_box(0, ry + wt / 2, wh / 2, 2 * rx, wt, wh);
    a[4] = aff_box(0, -ry - wt / 2, wh / 2, 2 * rx, wt, wh);
    const int nf = iv[(size_t)env::I_NF * E + e];
    uint32_t v = (1u << I_FLOOR) | (0xfu << I_WALL) | (1u << I_CUP);
    for (int k = 0; k < env::N_FURN; ++k) {
      if (k < nf) {
        const float lx = F(env::F_FB0 + 5 * k), ly = F(env::F_FB0 + 5 * k + 1), hx = F(env::F_FB0 + 5 * k + 2), hy = F(env::F_FB0 + 5 * k + 3), hz = F(env::F_FB0 + 5 * k + 4);
        a[I_FURN + k] = aff_box((lx + hx) / 2, (ly + hy) / 2, hz / 2, hx - lx, hy - ly, hz);
        v |= 1u << (I_FURN + k);
      } else {
        a[I_FURN + k] = aff_box(0, 0, -10, 0.01f, 0.01f, 0.01f);
      }
    }
    a[I_CUP] = aff_box(F(env::F_TX), F(env::F_TY), 0.05f, 0.08f, 0.08f, 0.10f);
    a[A_ROBOT] = aff_yaw(F(env::F_X), F(env::F_Y), F(env::F_YAW));
    vis[e] = v;
  }
  Aff* d_anch;
  uint32_t* d_vis;
  cudaMalloc(&d_anch, sizeof(Aff) * anch.size());
  cudaMemcpy(d_anch, anch.data(), sizeof(Aff) * anch.size(), cudaMemcpyHostToDevice);
  cudaMalloc(&d_vis, sizeof(uint32_t) * vis.size());
  cudaMemcpy(d_vis, vis.data(), sizeof(uint32_t) * vis.size(), cudaMemcpyHostToDevice);

  const int W = 224, Hh = 224;
  // 머리: 깊이 카메라 렌즈 (0.094, 0, 0.18) m, 수평, H-FOV 67.9°(env K::cam_hfov). 손목: (0.25, 0, 0.25) m, 45° 아래, 87° (가정 — 팔 FK 로 정해야 함, 속도 측정용 고정)
  const std::vector<CamRig> rigs = {rig(0.094f, 0.18f, 0.f, 1.18508f, W, Hh), rig(0.25f, 0.25f, 0.785398f, 1.5184f, W, Hh)};
  for (int profile : {0, 1}) {
    HostScene Hs = make_scene(profile);
    gpu::RenderBatch rb;
    rb.init(Hs, E, rigs);
    rb.set_anchors(d_anch);
    rb.set_visibility(d_vis);
    rb.render(0);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) rb.render(1 + r);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    ms /= reps;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    std::printf("render %s: E %d x 2 cams %dx%d: %.2f ms/frame-batch -> %.0f env-frames/s (2 images each), %.0f images/s; GPU mem used %.2f GB\n",
                profile == 0 ? "team-default (spp1 bounce1 spec1 denoise4)" : "cheap (spp1 bounce0 spec0 denoise0)", E, W, Hh, ms, E / (ms * 1e-3),
                2 * E / (ms * 1e-3), (tot - fr) / 1e9);
    if (!dump.empty()) {
      for (int c = 0; c < 2; ++c) {
        std::vector<uint8_t> img((size_t)W * Hh * 3 * 4);
        cudaMemcpy(img.data(), rb.rgb(c), img.size(), cudaMemcpyDeviceToHost);
        for (int e = 0; e < 4; ++e) {
          const std::string p = dump + "/p" + std::to_string(profile) + "_e" + std::to_string(e) + "_c" + std::to_string(c) + ".ppm";
          FILE* fp = std::fopen(p.c_str(), "wb");
          if (!fp) continue;
          std::fprintf(fp, "P6\n%d %d\n255\n", W, Hh);
          std::fwrite(img.data() + (size_t)e * W * Hh * 3, 1, (size_t)W * Hh * 3, fp);
          std::fclose(fp);
        }
      }
    }
  }
  return 0;
}
