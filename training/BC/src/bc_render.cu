// BC 영상 학생 렌더 — 설명은 include/bc_render.h. 팀 RenderBatch 는 include 로 읽기만 한다(src/behavior-2026 서브모듈, 고친 것 없음).
// 이 파일만 팀 렌더 빌드 규칙(-fmad=false -prec-div=true -prec-sqrt=true -ftz=true)으로 컴파일한다(CMakeLists).
//
// 그래프로 잡기(작업 1): RenderBatch::render 안의 일은 커널 실행뿐(kBuild, kRigs, kShade/kAtrous/kCompose)이고 호스트 값(판 수, 잡음 제거 횟수,
// frame)은 잡을 때 고정된다. 단 하나의 걸림돌은 render_cam → Scratch::ensure 의 cudaMalloc(처음 부를 때·판 수가 늘 때)이다.
// → create() 가 잡기 전에 한 번 그려 작업 공간을 잡아 두면 그 뒤 render 는 cudaMalloc 이 없어 그대로 잡힌다(서브모듈을 고치거나 베낄 필요 없음).
// capture_cold_probe() 는 그 반대(작업 공간 없이 바로 잡기)가 실패하는지 확인하는 시험용이다.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "bc_data.h"
#include "bc_render.h"
#include "cuda/render/render_batch.cuh"
#include "limo_omx_model.h"

using namespace eng::rnd;

namespace bcr {

// 인스턴스 배치(색 변형 V 개 — 렌더 흔들기 끔이면 V = 1 로 예전과 같은 장면): 바닥 V, 벽 4 × V, 가구 칸 8 × 종류 5 × V, 컵 V.
// 기준 prim = 인스턴스마다 하나 + 몸(머리 카메라) + 손목 카메라 (+ 흔들기 켬: 조명 하나)
constexpr int N_FCLS = 5, N_BASECOL = 8, MAXV = 4;
struct Lay {
  int V, n_inst, n_anch, vis_w, a_robot, a_wrist, a_light;
  __host__ __device__ int floor_i(int v) const { return v; }
  __host__ __device__ int wall_i(int w, int v) const { return V + w * V + v; }
  __host__ __device__ int furn_i(int k, int c, int v) const { return 5 * V + (k * N_FCLS + c) * V + v; }
  __host__ __device__ int cup_i(int v) const { return 5 * V + env::N_FURN * N_FCLS * V + v; }
};
static Lay make_lay(int V, bool light) {
  Lay L{};
  L.V = V;
  L.n_inst = (5 + env::N_FURN * N_FCLS + 1) * V;
  L.a_robot = L.n_inst; L.a_wrist = L.n_inst + 1; L.a_light = light ? L.n_inst + 2 : -1;
  L.n_anch = L.n_inst + (light ? 3 : 2);
  L.vis_w = (L.n_inst + 31) / 32;
  return L;
}
// 렌더 흔들기 세기(만들 때 정하는 호스트 값; 판마다 값은 RenderState 씨앗 pad[0..3] 에서)
struct AugK { int on; float color, light, expo; };

struct Renderer {
  gpu::RenderBatch rb;
  int E = 0;
  Lay lay{};
  AugK aug{};
};

static void push_tri(std::vector<float>& tv, std::vector<float>& tn, std::vector<float>& tu, std::vector<int32_t>& ts, const float* a, const float* b,
                     const float* c, const float* n, int slot) {
  const float* p[3] = {a, b, c};
  const float uvs[6] = {0, 0, 1, 0, 1, 1};
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tv.push_back(p[k][q]);
  for (int k = 0; k < 3; ++k) for (int q = 0; q < 3; ++q) tn.push_back(n[q]);
  for (int k = 0; k < 6; ++k) tu.push_back(uvs[k]);
  ts.push_back(slot);
}
static int32_t make_box(HostScene& S) {   // 단위 상자(중심 원점, 변 1) — render_bench 와 같음
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

static HostScene make_scene(int profile, const Lay& Ly) {
  HostScene H;
  const int gbox = make_box(H);
  // 바닥, 벽, 컵, 가구 종류 5 개(item, chair, table, cabinet, bin) — 색은 가정. 변형 v ≥ 1 은 기본 색을 흔든 색(고정 씨앗, 결정적):
  // 채널마다 × U(0.55, 1.45) + 다른 색상 쪽으로 섞기 U(0, 0.35) — 실제 집의 같은 종류 물건 색 퍼짐 흉내(가정)
  const float col[N_BASECOL][3] = {{0.55f, 0.52f, 0.48f}, {0.85f, 0.82f, 0.75f}, {0.75f, 0.15f, 0.12f}, {0.60f, 0.60f, 0.20f}, {0.45f, 0.30f, 0.18f},
                                   {0.35f, 0.40f, 0.55f}, {0.30f, 0.55f, 0.35f}, {0.25f, 0.25f, 0.28f}};
  uint64_t cs = 0xC0105EEDull;
  auto u01 = [&] { cs = cs * 6364136223846793005ull + 1442695040888963407ull; return (float)(cs >> 40) * (1.f / 16777216.f); };
  for (int b = 0; b < N_BASECOL; ++b)
    for (int v = 0; v < Ly.V; ++v) {
      float c[3] = {col[b][0], col[b][1], col[b][2]};
      if (v > 0) {
        const float hue[3] = {u01(), u01(), u01()}, mix = 0.35f * u01();
        for (int q = 0; q < 3; ++q) c[q] = std::fmin(0.95f, std::fmax(0.03f, (1.f - mix) * c[q] * (0.55f + 0.9f * u01()) + mix * hue[q]));
      }
      Material M{};
      M.albedo[0] = c[0]; M.albedo[1] = c[1]; M.albedo[2] = c[2];
      M.tex_albedo = -1;
      M.uv_scale[0] = M.uv_scale[1] = 1.0f;
      M.albedo_brightness = 1.0f;
      M.opacity = 1.0f;
      M.roughness = 0.6f;
      H.mats.push_back(M);
    }
  auto mat = [&](int b, int v) { return b * Ly.V + v; };
  H.n_anchor = Ly.n_anch;
  auto add_inst = [&](int anc, int mat) {
    InstInfo in{};
    in.geom = gbox; in.anchor = anc; in.slot_base = int32_t(H.slot_mat.size());
    in.rel = Aff{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};
    H.slot_mat.push_back(mat);
    H.insts.push_back(in);
  };
  // 인스턴스 차례 = Lay 의 번호(인스턴스 i 의 기준 prim = i)
  for (int v = 0; v < Ly.V; ++v) add_inst(Ly.floor_i(v), mat(0, v));
  for (int k = 0; k < 4; ++k) for (int v = 0; v < Ly.V; ++v) add_inst(Ly.wall_i(k, v), mat(1, v));
  for (int k = 0; k < env::N_FURN; ++k)
    for (int c = 0; c < N_FCLS; ++c) for (int v = 0; v < Ly.V; ++v) add_inst(Ly.furn_i(k, c, v), mat(3 + c, v));
  for (int v = 0; v < Ly.V; ++v) add_inst(Ly.cup_i(v), mat(2, v));
  Light L{};
  L.type = kLightDistant; L.anchor = Ly.a_light; L.visible = 1;   // 흔들기 켬이면 판마다 조명 기준 prim(방향·자리), 끔이면 −1(예전 그대로)
  L.radiance[0] = L.radiance[1] = L.radiance[2] = 3000.f;
  L.angle = 0.53f;
  L.rel = Aff{{1, 0, 0, 0, 0, 0.8f, -0.6f, 0, 0, 0.6f, 0.8f, 0}};
  H.lights.push_back(L);
  Light S{};
  S.type = kLightSphere; S.anchor = Ly.a_light; S.visible = 1;
  S.radiance[0] = 600.f; S.radiance[1] = 560.f; S.radiance[2] = 500.f;
  S.radius = 0.15f;
  S.rel = Aff{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.4f}};
  H.lights.push_back(S);
  H.sp.ambient[0] = H.sp.ambient[1] = H.sp.ambient[2] = 0.08f;
  H.sp.exposure = 1.0f; H.sp.spp = 1; H.sp.shadow_lights = 1; H.sp.tonemap = 2;
  H.sp.white_scale = 8.0f; H.sp.dome_light = -1; H.sp.dn_plane = 0.01f; H.sp.clamp_ind = 0.0f; H.sp.tex_aniso = 2; H.sp.lod_bias = -0.5f; H.sp.spec_f0 = 0.04f;
  if (profile == TEAM_DEFAULT) { H.sp.bounces = 1; H.sp.denoise = 4; H.sp.spec = 1; }
  else { H.sp.bounces = 0; H.sp.denoise = 0; H.sp.spec = 0; }
  return H;
}

// 카메라 rel(기준 prim 틀): 열 = 오른쪽, 위, 뒤(−시선), 위치. 시선 = 틀의 +x, 위 = +z (ROS 링크 틀)
static CamRig rig(int anchor, float px, float pz, float hfov) {
  CamRig g{};
  g.anchor = anchor; g.w = RES; g.h = RES;
  g.rel = Aff{{0, 0, -1, px, -1, 0, 0, 0, 0, 1, 0, pz}};
  g.tanx = std::tan(hfov * 0.5f);
  g.tany = g.tanx;
  g.znear = 0.01f; g.zfar = 50.f;
  return g;
}

__device__ inline float h2f_dev(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ inline Aff box_aff(float cx, float cy, float cz, float sx, float sy, float sz) { return Aff{{sx, 0, 0, cx, 0, sy, 0, cy, 0, 0, sz, cz}}; }

// 손목 카메라 자세(footprint 기준): base_joint → omx_mount → joint1..5(q) → wrist_cam_joint (env fk 와 같은 합성 순서)
__device__ void wrist_pose(const float* q, float R[9], float p[3]) {
  using namespace limo_omx;
  for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0) ? 1.f : 0.f;
  p[0] = p[1] = p[2] = 0.f;
  auto compose = [&](int j, float ang) {
    const JointConst& J = joint(j);
    for (int i = 0; i < 3; ++i) p[i] = p[i] + (R[3 * i] * J.t[0] + R[3 * i + 1] * J.t[1] + R[3 * i + 2] * J.t[2]);
    float R1[9];
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) R1[3 * i + k] = R[3 * i] * J.R[k] + R[3 * i + 1] * J.R[3 + k] + R[3 * i + 2] * J.R[6 + k];
    if (J.kind != kFixed) {
      float s, c;
      sincosf(ang, &s, &c);
      const float* a = J.axis;
      const float t = 1.f - c;
      const float Ra[9] = {c + a[0] * a[0] * t, a[0] * a[1] * t - a[2] * s, a[0] * a[2] * t + a[1] * s,
                           a[1] * a[0] * t + a[2] * s, c + a[1] * a[1] * t, a[1] * a[2] * t - a[0] * s,
                           a[2] * a[0] * t - a[1] * s, a[2] * a[1] * t + a[0] * s, c + a[2] * a[2] * t};
      for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) R[3 * i + k] = R1[3 * i] * Ra[k] + R1[3 * i + 1] * Ra[3 + k] + R1[3 * i + 2] * Ra[6 + k];
    } else {
      for (int i = 0; i < 9; ++i) R[i] = R1[i];
    }
  };
  compose(J_BASE_JOINT, 0.f);
  compose(J_OMX_MOUNT_JOINT, 0.f);
  const int jid[5] = {J_OMX_JOINT1, J_OMX_JOINT2, J_OMX_JOINT3, J_OMX_JOINT4, J_OMX_JOINT5};
  for (int k = 0; k < 5; ++k) compose(jid[k], q[k]);
  compose(J_WRIST_CAM_JOINT, 0.f);
}

// RenderState → 기준 prim 행렬 + 보임 비트 (스레드 하나 = 판 하나). 흔들기 켬: 씨앗(pad[0..3])으로 무리마다 색 변형(확률 color 로 1..V−1),
// 조명 기준 prim = z 축 회전 ±light·π · 수평 이동 ±light·1.5 m(구 조명 자리·해 방향이 바뀜)
__device__ inline uint64_t aug_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ inline float aug_u(uint32_t seed, int k) { return (float)(aug_mix(((uint64_t)seed << 16) + 0x5EEDull * (uint64_t)(k + 1)) >> 40) * (1.f / 16777216.f); }
__global__ void rs_anchor_k(const bc::RenderState* rs, int n, int A, int Wv, Lay Ly, AugK ak, Aff* anchor, uint32_t* vis) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  const bc::RenderState s = rs[e];
  const uint32_t seed = (uint32_t)s.pad[0] | ((uint32_t)s.pad[1] << 8) | ((uint32_t)s.pad[2] << 16) | ((uint32_t)s.pad[3] << 24);
  auto var = [&](int g) -> int {   // 무리 g 의 색 변형
    if (!ak.on || Ly.V < 2) return 0;
    if (aug_u(seed, 2 * g) >= ak.color) return 0;
    const int v = 1 + (int)(aug_u(seed, 2 * g + 1) * (float)(Ly.V - 1));
    return v > Ly.V - 1 ? Ly.V - 1 : v;
  };
  Aff* a = anchor + (size_t)e * A;
  const float rx = h2f_dev(s.rh[0]), ry = h2f_dev(s.rh[1]), wt = 0.1f, wh = 1.0f;
  uint32_t v[(46 * MAXV + 31) / 32];
  for (int k = 0; k < Ly.vis_w; ++k) v[k] = 0u;
  auto on = [&](int i) { v[i >> 5] |= 1u << (i & 31); };
  const Aff hidden = box_aff(0, 0, -10.f, 0.01f, 0.01f, 0.01f);
  const int vf = var(0), vw = var(1), vc = var(2);
  for (int q = 0; q < Ly.V; ++q) {
    a[Ly.floor_i(q)] = box_aff(0, 0, -0.05f, 2 * rx + 1, 2 * ry + 1, 0.1f);
    a[Ly.wall_i(0, q)] = box_aff(rx + wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
    a[Ly.wall_i(1, q)] = box_aff(-rx - wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
    a[Ly.wall_i(2, q)] = box_aff(0, ry + wt / 2, wh / 2, 2 * rx, wt, wh);
    a[Ly.wall_i(3, q)] = box_aff(0, -ry - wt / 2, wh / 2, 2 * rx, wt, wh);
    a[Ly.cup_i(q)] = box_aff(s.tx, s.ty, 0.05f, 0.08f, 0.08f, 0.10f);
  }
  on(Ly.floor_i(vf));
  for (int k = 0; k < 4; ++k) on(Ly.wall_i(k, vw));
  on(Ly.cup_i(vc));
  for (int k = 0; k < env::N_FURN; ++k) {
    const int cls = (int)s.fc[k];
    const bool use = k < (int)s.nf && cls >= 1 && cls <= N_FCLS;
    const int vk = var(3 + k);
    for (int c = 0; c < N_FCLS; ++c)
      for (int q = 0; q < Ly.V; ++q) {
        const int i = Ly.furn_i(k, c, q);
        if (use && c == cls - 1) {
          const float lx = h2f_dev(s.fb[k * 5]), ly = h2f_dev(s.fb[k * 5 + 1]), hx = h2f_dev(s.fb[k * 5 + 2]), hy = h2f_dev(s.fb[k * 5 + 3]), hz = h2f_dev(s.fb[k * 5 + 4]);
          a[i] = box_aff((lx + hx) * 0.5f, (ly + hy) * 0.5f, hz * 0.5f, hx - lx, hy - ly, hz);
          if (q == vk) on(i);
        } else {
          a[i] = hidden;
        }
      }
  }
  float sn, cs;
  sincosf(s.yaw, &sn, &cs);
  a[Ly.a_robot] = Aff{{cs, -sn, 0, s.x, sn, cs, 0, s.y, 0, 0, 1, 0}};
  float R[9], p[3];
  wrist_pose(s.q, R, p);
  const float Rz[9] = {cs, -sn, 0, sn, cs, 0, 0, 0, 1};
  Aff w;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) w.m[4 * i + k] = Rz[3 * i] * R[k] + Rz[3 * i + 1] * R[3 + k] + Rz[3 * i + 2] * R[6 + k];
    w.m[4 * i + 3] = (i == 0 ? s.x : i == 1 ? s.y : 0.f) + Rz[3 * i] * p[0] + Rz[3 * i + 1] * p[1] + Rz[3 * i + 2] * p[2];
  }
  a[Ly.a_wrist] = w;
  if (Ly.a_light >= 0) {
    const float th = ak.light * 3.14159265f * (2.f * aug_u(seed, 40) - 1.f);
    const float dx = ak.light * 1.5f * (2.f * aug_u(seed, 41) - 1.f), dy = ak.light * 1.5f * (2.f * aug_u(seed, 42) - 1.f);
    float ls, lc;
    sincosf(th, &ls, &lc);
    a[Ly.a_light] = Aff{{lc, -ls, 0, dx, ls, lc, 0, dy, 0, 0, 1, 0}};
  }
  for (int k = 0; k < A - Ly.n_anch; ++k) a[Ly.n_anch + k] = a[Ly.a_robot];
  for (int k = 0; k < Wv; ++k) vis[(size_t)e * Wv + k] = k < Ly.vis_w ? v[k] : 0u;
}
// 노출·대비·채널 이득·감마 흔들기(판마다, 두 카메라 같은 값): y = clamp(((x/255)^γ − ½)·c + ½)·b·g_ch (가정: 실제 카메라 자동 노출·흰색 맞춤 퍼짐)
__global__ void expo_k(const bc::RenderState* rs, int n, float expo, uint8_t* rgb) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long per = (long long)RES * RES * 3;
  if (q >= (long long)n * per) return;
  const int e = (int)(q / per), ch = (int)(q % 3);
  const bc::RenderState& s = rs[e];
  const uint32_t seed = (uint32_t)s.pad[0] | ((uint32_t)s.pad[1] << 8) | ((uint32_t)s.pad[2] << 16) | ((uint32_t)s.pad[3] << 24);
  const float b = 1.f + expo * 0.4f * (2.f * aug_u(seed, 50) - 1.f), c = 1.f + expo * 0.3f * (2.f * aug_u(seed, 51) - 1.f);
  const float g = 1.f + expo * 0.15f * (2.f * aug_u(seed, 52 + ch) - 1.f), gm = 1.f + expo * 0.3f * (2.f * aug_u(seed, 55) - 1.f);
  float x = powf((float)rgb[q] * (1.f / 255.f), gm);
  x = ((x - 0.5f) * c + 0.5f) * b * g;
  x = fminf(fmaxf(x, 0.f), 1.f);
  rgb[q] = (uint8_t)(x * 255.f + 0.5f);
}

static std::vector<CamRig> rigs(const Lay& Ly) {
  return {rig(Ly.a_robot, 0.094f, 0.18f, 1.18508f), rig(Ly.a_wrist, 0.f, 0.f, 1.5184f)};
}

static void default_states(std::vector<bc::RenderState>& v, int E) {   // pad 0 = 씨앗 0
  v.assign((size_t)E, bc::RenderState{});
  for (auto& s : v) {
    s.x = 0.5f; s.y = 0.f; s.yaw = 0.f; s.tx = -1.f; s.ty = 0.5f;
    const float q[6] = {0.f, 1.3f, -1.9f, 0.7f, 0.f, 0.f};
    for (int k = 0; k < 6; ++k) s.q[k] = q[k];
    s.rh[0] = s.rh[1] = 0x4000;   // 2.0
    s.nf = 0;
    for (int k = 0; k < env::N_FURN; ++k) s.fc[k] = -1;
  }
}

Renderer* create(int E, int profile, const RenderAug* aug) {
  auto* r = new Renderer;
  r->E = E;
  const bool ao = aug && aug->on;
  r->aug = AugK{ao ? 1 : 0, ao ? aug->color : 0.f, ao ? aug->light : 0.f, ao ? aug->expo : 0.f};
  r->lay = make_lay(ao && aug->color > 0.f ? MAXV : 1, ao && aug->light > 0.f);
  HostScene H = make_scene(profile, r->lay);
  r->rb.init(H, E, rigs(r->lay));
  // 작업 공간을 잡기 전에 한 번(기본 장면) — 이 뒤 render 는 cudaMalloc 없이 그래프로 잡힌다
  std::vector<bc::RenderState> hs;
  default_states(hs, E);
  bc::RenderState* d = nullptr;
  RCK(cudaMalloc(&d, sizeof(bc::RenderState) * E));
  RCK(cudaMemcpy(d, hs.data(), sizeof(bc::RenderState) * E, cudaMemcpyHostToDevice));
  render(r, d, E, cudaStreamPerThread);
  RCK(cudaStreamSynchronize(cudaStreamPerThread));
  cudaFree(d);
  return r;
}
void destroy(Renderer* r) { delete r; }

void render(Renderer* r, const bc::RenderState* rs, int n, cudaStream_t st) {
  if (n > r->E) { std::fprintf(stderr, "bcr::render: n %d > E %d\n", n, r->E); std::abort(); }
  rs_anchor_k<<<(n + 127) / 128, 128, 0, st>>>(rs, n, r->rb.B.A, r->rb.B.W, r->lay, r->aug, r->rb.B.anchor, r->rb.B.vis);
  RCK(cudaGetLastError());
  r->rb.render(0, st);
  if (r->aug.on && r->aug.expo > 0.f)
    for (int c = 0; c < 2; ++c) {
      const long long nq = (long long)n * RES * RES * 3;
      expo_k<<<(unsigned)((nq + 255) / 256), 256, 0, st>>>(rs, n, r->aug.expo, r->rb.rgb(c));
    }
}

int capture_cold_probe(int E, int profile) {
  gpu::RenderBatch rb;
  HostScene H = make_scene(profile, make_lay(1, false));
  rb.init(H, E, rigs(make_lay(1, false)));
  cudaStream_t st;
  RCK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  RCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  // render_cam → Scratch::ensure → cudaMalloc. RCK 가 실패를 잡으면 이 과정은 exit(1) 한다(부르는 쪽이 fork 해서 본다)
  rb.render(0, st);
  cudaGraph_t g;
  const cudaError_t e = cudaStreamEndCapture(st, &g);
  return (int)e;
}

const uint8_t* rgb(Renderer* r, int cam) { return r->rb.rgb(cam); }
int batch(Renderer* r) { return r->E; }
size_t bytes(Renderer* r) {
  const size_t px = (size_t)RES * RES * r->E;
  return r->rb.scene.bytes + px * 2 * (4 + 3) + r->rb.work.cap * (sizeof(GPix) + 48) + (size_t)r->E * (r->lay.n_anch * sizeof(Aff) * 3 + 4096);
}

}  // namespace bcr
