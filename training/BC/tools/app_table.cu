// 생김새 벡터 표(VLA_INPUT 3절 "생김새 벡터 128 = 물체 조각의 SigLIP 2 영상 벡터 → 같은 128-d 투영") — 우리 GPU 렌더 + 우리 C++ SigLIP 2 영상 탑.
//   app_table [--out DIR] [--views V] [--profile 1|0] [--dump DIR] [--seed S] [--negative B]
// 시뮬 물체 종류(gmap::Cls: 컵·작은 상자·의자·탁자·장·쓰레기통)마다 상자 하나를 bc_render 와 같은 재질·색·조명의 방에 놓고,
// 물체 중심을 겨눈 카메라로 "조각"(물체 경계 구가 화면의 1/1.1 — embed 의 box 자르기 10 % 여유와 같은 꼴)을 256² 로 그린다.
// 시점: 방위 무작위, 시선 각 −6°–52°(몸통 카메라 수평 ~ 손목 내려다봄), 물체 자리·yaw 무작위, 작은 상자는 크기도 env.h 범위에서 무작위.
// 줄 6(유령 = 가짜 검출, map.h Ghost)은 물체 없는 바닥·벽 조각이다: 가짜 검출 자리에는 실제로 아무것도 없으므로 그 조각의 생김새.
// 렌더(RenderBatch) → vit::Encoder(끝 LN 패치 토큰) → apph::encode(MAP 풀링 + embed 머리 h, double) → 시점마다 128 →
// 종류마다 평균 → L2 정규화 = app128.f16 [7][128]. 시점별 벡터는 app_views.f16 [7][2V][128](확신도 분포·문턱 근거용).
// --dump DIR: 시점별 토큰(FP32 로 바꾼 bf16)·RGB·C++ 출력(풀링 768·128)을 써서 tools/app_ref.py(PyTorch 기준값)와 맞춘다.
// --negative B: apph 음성 대조(1 어텐션 배율 빠뜨림, 2 머리 LN 빠뜨림, 3 MAP MLP 잔차 빠뜨림) — dump 와 같이 써서 기준값과 어긋나야 정상.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "app_head.h"
#include "cuda/render/render_batch.cuh"
#include "vit.h"

using namespace eng::rnd;

namespace {

constexpr int NCLS = 6, NROW = 7, RES = 256;
constexpr int I_FLOOR = 0, I_WALL = 1, I_OBJ = 5, N_INST = I_OBJ + NCLS, A_CAM0 = N_INST, A_CAM1 = N_INST + 1, N_ANCH = N_INST + 2;
constexpr float FOV = 0.8726646f;   // 50° 조각 카메라 (가정)

void push_tri(std::vector<float>& tv, std::vector<float>& tn, std::vector<float>& tu, std::vector<int32_t>& ts, const float* a, const float* b,
              const float* c, const float* n) {
  const float* p[3] = {a, b, c};
  const float uvs[6] = {0, 0, 1, 0, 1, 1};
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tv.push_back(p[k][q]);
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tn.push_back(n[q]);
  for (int k = 0; k < 6; ++k) tu.push_back(uvs[k]);
  ts.push_back(0);
}
int32_t make_box(HostScene& S) {
  std::vector<float> tv, tn, tu;
  std::vector<int32_t> ts;
  const float P[8][3] = {{-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {.5f, .5f, -.5f}, {-.5f, .5f, -.5f}, {-.5f, -.5f, .5f}, {.5f, -.5f, .5f}, {.5f, .5f, .5f}, {-.5f, .5f, .5f}};
  const int F[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
  const float Nn[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {1, 0, 0}, {-1, 0, 0}};
  for (int f = 0; f < 6; ++f) {
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][1]], P[F[f][2]], Nn[f]);
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][2]], P[F[f][3]], Nn[f]);
  }
  return add_geometry(S, tv.data(), tn.data(), tu.data(), ts.data(), uint32_t(ts.size()), 3);
}

// 재질·조명은 src/bc_render.cu make_scene 과 같은 값(그 파일은 고치지 않고 값을 옮김): 바닥, 벽, 컵, 종류 item·chair·table·cabinet·bin
HostScene make_scene(int profile) {
  HostScene H;
  const int gbox = make_box(H);
  const float col[8][3] = {{0.55f, 0.52f, 0.48f}, {0.85f, 0.82f, 0.75f}, {0.75f, 0.15f, 0.12f}, {0.60f, 0.60f, 0.20f}, {0.45f, 0.30f, 0.18f},
                           {0.35f, 0.40f, 0.55f}, {0.30f, 0.55f, 0.35f}, {0.25f, 0.25f, 0.28f}};
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
  add_inst(I_FLOOR, 0);
  for (int k = 0; k < 4; ++k) add_inst(I_WALL + k, 1);
  add_inst(I_OBJ + 0, 2);                                       // 컵
  for (int c = 1; c < NCLS; ++c) add_inst(I_OBJ + c, 2 + c);    // item(3)·chair(4)·table(5)·cabinet(6)·bin(7) — bc_render 의 3 + (cls − 1)
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
  if (profile == 0) { H.sp.bounces = 1; H.sp.denoise = 4; H.sp.spec = 1; }
  else { H.sp.bounces = 0; H.sp.denoise = 0; H.sp.spec = 0; }
  return H;
}
CamRig rig(int anchor) {   // 기준 prim 틀 = 카메라 링크(x 시선, z 위) — bc_render rig() 와 같은 rel
  CamRig g{};
  g.anchor = anchor; g.w = RES; g.h = RES;
  g.rel = Aff{{0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0}};
  g.tanx = std::tan(FOV * 0.5f);
  g.tany = g.tanx;
  g.znear = 0.01f; g.zfar = 50.f;
  return g;
}
Aff box_aff(float cx, float cy, float cz, float sx, float sy, float sz, float yaw) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  return Aff{{c * sx, -s * sy, 0, cx, s * sx, c * sy, 0, cy, 0, 0, sz, cz}};
}
Aff look_at(const float p[3], const float t[3]) {   // 링크 틀: x = 시선, y = 왼쪽, z = 위
  float f[3] = {t[0] - p[0], t[1] - p[1], t[2] - p[2]};
  const float fn = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (float& v : f) v /= fn;
  float y[3] = {-f[1], f[0], 0.f};   // z × f
  const float yn = std::sqrt(y[0] * y[0] + y[1] * y[1]);
  for (float& v : y) v /= yn;
  const float z[3] = {f[1] * y[2] - f[2] * y[1], f[2] * y[0] - f[0] * y[2], f[0] * y[1] - f[1] * y[0]};
  return Aff{{f[0], y[0], z[0], p[0], f[1], y[1], z[1], p[1], f[2], y[2], z[2], p[2]}};
}

struct Rng {
  uint64_t s;
  double u() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (double)(s >> 11) * (1.0 / 9007199254740992.0); }
  float r(float a, float b) { return a + (b - a) * (float)u(); }
};

// 종류별 상자 크기(m): 컵 0.08×0.08×0.10(bc_render), 작은 상자 env.h 범위, 가구 env.h reset_a2_scene 의 기준 크기 × 0.8–1.2
void cls_size(int c, Rng& g, float s[3]) {
  if (c == 0) { s[0] = 0.08f; s[1] = 0.08f; s[2] = 0.10f; return; }
  if (c == 1) { s[0] = g.r(0.08f, 0.25f); s[1] = g.r(0.08f, 0.25f); s[2] = g.r(0.12f, 0.35f); return; }
  const float w[4] = {0.45f, 0.8f, 0.5f, 0.3f}, d[4] = {0.45f, 0.5f, 0.4f, 0.3f}, h[4] = {0.9f, 0.75f, 1.0f, 0.4f};
  s[0] = w[c - 2] * g.r(0.8f, 1.2f); s[1] = d[c - 2] * g.r(0.8f, 1.2f); s[2] = h[c - 2] * g.r(0.8f, 1.2f);
}

void put16(const std::string& p, const std::vector<float>& v) {
  std::vector<uint16_t> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) { const __half x = __float2half_rn(v[i]); std::memcpy(&h[i], &x, 2); }
  FILE* f = std::fopen(p.c_str(), "wb");
  std::fwrite(h.data(), 2, h.size(), f);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = "training/data/vla_v1", dump, head = std::string(std::getenv("RA_EMBED_WORK") ? std::getenv("RA_EMBED_WORK") : "training/data/embed") + "/runs/sb32_pe_300k/head_h.f32";
  int V = 12, profile = 1, bug = 0;
  uint64_t seed = 20261004;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out") out = argv[++i];
    else if (a == "--views") V = std::atoi(argv[++i]);
    else if (a == "--profile") profile = std::atoi(argv[++i]);
    else if (a == "--dump") dump = argv[++i];
    else if (a == "--seed") seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--negative") bug = std::atoi(argv[++i]);
    else if (a == "--head") head = argv[++i];
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  apph::Weights aw;
  if (!apph::load("", head, aw)) return 1;
  vit::HostWeights hw;
  if (!vit::load_weights("", hw)) return 1;
  const int E = NROW * V;   // 판 = (종류, 시점 쌍), 카메라 2 대 → 영상 2E
  vit::Encoder enc;
  enc.init(hw, 2 * E);
  uint16_t* dtok = nullptr;
  RCK(cudaMalloc(&dtok, sizeof(uint16_t) * (size_t)2 * E * vit::NTOK * vit::TOK_LD));
  vit::init_token_buffer(dtok, (long long)2 * E * vit::NTOK, 0);

  gpu::RenderBatch rb;
  HostScene H = make_scene(profile);
  rb.init(H, E, {rig(A_CAM0), rig(A_CAM1)});
  const int A = rb.B.A, W = rb.B.W;
  std::vector<Aff> anc((size_t)E * A);
  std::vector<uint32_t> vis((size_t)E * W, 0u);
  Rng g{seed};
  for (int e = 0; e < E; ++e) {
    const int c = e / V;   // 줄 0..5 = 종류, 6 = 유령(물체 없음)
    Aff* a = anc.data() + (size_t)e * A;
    const float rx = 3.f, ry = 3.f, wt = 0.1f, wh = 1.0f;
    a[I_FLOOR] = box_aff(0, 0, -0.05f, 2 * rx + 1, 2 * ry + 1, 0.1f, 0);
    a[I_WALL + 0] = box_aff(rx + wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh, 0);
    a[I_WALL + 1] = box_aff(-rx - wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh, 0);
    a[I_WALL + 2] = box_aff(0, ry + wt / 2, wh / 2, 2 * rx, wt, wh, 0);
    a[I_WALL + 3] = box_aff(0, -ry - wt / 2, wh / 2, 2 * rx, wt, wh, 0);
    for (int k = 0; k < 5; ++k) vis[(size_t)e * W + ((I_FLOOR + k) >> 5)] |= 1u << ((I_FLOOR + k) & 31);
    float sz[3] = {0.3f, 0.3f, 0.3f};
    const float ox = g.r(-1.2f, 1.2f), oy = g.r(-1.2f, 1.2f), yaw = g.r(-3.14159f, 3.14159f);
    if (c < NCLS) cls_size(c, g, sz);
    for (int k = 0; k < NCLS; ++k)
      a[I_OBJ + k] = (k == c) ? box_aff(ox, oy, sz[2] * 0.5f, sz[0], sz[1], sz[2], yaw) : box_aff(0, 0, -10.f, 0.01f, 0.01f, 0.01f, 0);
    if (c < NCLS) vis[(size_t)e * W + ((I_OBJ + c) >> 5)] |= 1u << ((I_OBJ + c) & 31);
    for (int cam = 0; cam < 2; ++cam) {
      float t[3] = {ox, oy, sz[2] * 0.5f}, p[3];
      const float az = g.r(-3.14159f, 3.14159f), h = g.r(0.15f, 0.75f);
      if (c < NCLS) {
        const float rho = 0.5f * std::sqrt(sz[0] * sz[0] + sz[1] * sz[1] + sz[2] * sz[2]);
        const float d = rho * 1.1f / std::tan(FOV * 0.5f);
        (void)h;   // 높이는 시선 올려다봄·내려다봄 각으로(−6°–52°, 몸통 카메라 수평 ~ 손목 내려다봄), 바닥 아래로는 안 감
        const float el = g.r(-0.1f, 0.9f), ce = std::cos(el);
        p[0] = ox + d * ce * std::cos(az); p[1] = oy + d * ce * std::sin(az); p[2] = std::fmax(t[2] + d * std::sin(el), 0.05f);
      } else {   // 유령: 바닥·벽 쪽 아무 자리(0.6–1.5 m 앞, 높이 0–0.4 m)를 겨눔
        const float d = g.r(0.6f, 1.5f);
        p[0] = ox; p[1] = oy; p[2] = h;
        t[0] = ox + d * std::cos(az); t[1] = oy + d * std::sin(az); t[2] = g.r(0.0f, 0.4f);
      }
      a[cam == 0 ? A_CAM0 : A_CAM1] = look_at(p, t);
    }
    for (int k = N_ANCH; k < A; ++k) a[k] = a[A_CAM0];
  }
  Aff* danc = nullptr;
  uint32_t* dvis = nullptr;
  RCK(cudaMalloc(&danc, sizeof(Aff) * anc.size()));
  RCK(cudaMalloc(&dvis, 4 * vis.size()));
  RCK(cudaMemcpy(danc, anc.data(), sizeof(Aff) * anc.size(), cudaMemcpyHostToDevice));
  RCK(cudaMemcpy(dvis, vis.data(), 4 * vis.size(), cudaMemcpyHostToDevice));
  rb.set_anchors(danc, 0);
  rb.set_visibility(dvis, 0);
  rb.render(0, 0);
  enc.patchify(rb.rgb(0), rb.rgb(1), E, 0, 0);
  enc.run(2 * E, dtok, 0);
  RCK(cudaDeviceSynchronize());
  const int NI = 2 * E;
  std::vector<uint16_t> tb((size_t)NI * vit::NTOK * vit::TOK_LD);
  RCK(cudaMemcpy(tb.data(), dtok, 2 * tb.size(), cudaMemcpyDeviceToHost));
  std::vector<float> tf((size_t)NI * vit::NTOK * vit::D);
  for (int r = 0; r < NI * vit::NTOK; ++r)
    for (int c = 0; c < vit::D; ++c) {
      const uint32_t u = (uint32_t)tb[(size_t)r * vit::TOK_LD + c] << 16;
      std::memcpy(&tf[(size_t)r * vit::D + c], &u, 4);
    }
  // 영상 i = 판 e × 2 + 카메라 (patchify 규칙) → 줄 c = e / V
  std::vector<double> pooled((size_t)NI * apph::D), o128((size_t)NI * apph::OUT);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < NI; ++i)
    apph::encode(aw, tf.data() + (size_t)i * vit::NTOK * vit::D, vit::D, &pooled[(size_t)i * apph::D], &o128[(size_t)i * apph::OUT], bug);
  std::vector<float> app((size_t)NROW * apph::OUT, 0.f), views((size_t)NROW * 2 * V * apph::OUT);
  for (int row = 0; row < NROW; ++row) {
    double m[apph::OUT] = {};
    for (int j = 0; j < 2 * V; ++j) {
      const int i = row * 2 * V + j;
      for (int k = 0; k < apph::OUT; ++k) { m[k] += o128[(size_t)i * apph::OUT + k]; views[((size_t)row * 2 * V + j) * apph::OUT + k] = (float)o128[(size_t)i * apph::OUT + k]; }
    }
    double n2 = 0;
    for (double v : m) n2 += v * v;
    for (int k = 0; k < apph::OUT; ++k) app[(size_t)row * apph::OUT + k] = (float)(m[k] / std::sqrt(n2));
  }
  // 요약: 종류 안 시점끼리 코사인 평균, 종류 평균끼리 코사인
  const char* nm[NROW] = {"cup", "box(item)", "chair", "table", "cabinet", "trash can", "ghost"};
  for (int r = 0; r < NROW; ++r) {
    double s = 0;
    int n = 0;
    for (int a = 0; a < 2 * V; ++a)
      for (int b = a + 1; b < 2 * V; ++b, ++n)
        for (int k = 0; k < apph::OUT; ++k) s += (double)views[((size_t)r * 2 * V + a) * apph::OUT + k] * views[((size_t)r * 2 * V + b) * apph::OUT + k];
    std::printf("row %d %-10s within-row view cos mean %.3f | mean-vector cos to rows:", r, nm[r], s / n);
    for (int q = 0; q < NROW; ++q) {
      double d = 0;
      for (int k = 0; k < apph::OUT; ++k) d += (double)app[(size_t)r * apph::OUT + k] * app[(size_t)q * apph::OUT + k];
      std::printf(" %.3f", d);
    }
    std::printf("\n");
  }
  if (!dump.empty()) {
    mkdir(dump.c_str(), 0755);
    auto wr = [&](const std::string& f, const void* p, size_t n) { FILE* q = std::fopen((dump + "/" + f).c_str(), "wb"); std::fwrite(p, 1, n, q); std::fclose(q); };
    wr("tokens.f32", tf.data(), 4 * tf.size());
    std::vector<float> pf(pooled.begin(), pooled.end()), of(o128.begin(), o128.end());
    wr("pooled_cpp.f32", pf.data(), 4 * pf.size());
    wr("out_cpp.f32", of.data(), 4 * of.size());
    std::vector<uint8_t> rgb((size_t)E * RES * RES * 3);
    for (int cam = 0; cam < 2; ++cam) {
      RCK(cudaMemcpy(rgb.data(), rb.rgb(cam), rgb.size(), cudaMemcpyDeviceToHost));
      wr("rgb" + std::to_string(cam) + ".u8", rgb.data(), rgb.size());
    }
    FILE* q = std::fopen((dump + "/meta.txt").c_str(), "w");
    std::fprintf(q, "%d %d %d %d\n", NI, E, V, bug);
    std::fclose(q);
    std::printf("dump: %d images -> %s\n", NI, dump.c_str());
  }
  if (bug == 0 && dump.empty()) {
    mkdir(out.c_str(), 0755);
    put16(out + "/app128.f16", app);
    put16(out + "/app_views.f16", views);
    std::printf("wrote %s/app128.f16 [%d][128], app_views.f16 [%d][%d][128] (profile %d, seed %llu)\n", out.c_str(), NROW, NROW, 2 * V, profile,
                (unsigned long long)seed);
  }
  cudaFree(dtok); cudaFree(danc); cudaFree(dvis);
  return 0;
}
