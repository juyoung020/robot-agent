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

constexpr int I_FLOOR = 0, I_WALL = 1, I_FURN = 5, N_FCLS = 5, I_CUP = I_FURN + env::N_FURN * N_FCLS, N_INST = I_CUP + 1;
constexpr int A_ROBOT = N_INST, A_WRIST = N_INST + 1, N_ANCH = N_INST + 2;
constexpr int VIS_W = (N_INST + 31) / 32;
static_assert(N_INST == 46 && VIS_W == 2, "instances");

struct Renderer {
  gpu::RenderBatch rb;
  int E = 0;
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

static HostScene make_scene(int profile) {
  HostScene H;
  const int gbox = make_box(H);
  // 바닥, 벽, 컵, 가구 종류 5 개(item, chair, table, cabinet, bin) — 색은 가정
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
  for (int k = 0; k < env::N_FURN; ++k)
    for (int c = 0; c < N_FCLS; ++c) add_inst(I_FURN + k * N_FCLS + c, 3 + c);
  add_inst(I_CUP, 2);
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

// RenderState → 기준 prim 행렬 48 개 + 보임 비트 2 낱말 (스레드 하나 = 판 하나)
__global__ void rs_anchor_k(const bc::RenderState* rs, int n, int A, int Wv, Aff* anchor, uint32_t* vis) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  const bc::RenderState s = rs[e];
  Aff* a = anchor + (size_t)e * A;
  const float rx = h2f_dev(s.rh[0]), ry = h2f_dev(s.rh[1]), wt = 0.1f, wh = 1.0f;
  a[I_FLOOR] = box_aff(0, 0, -0.05f, 2 * rx + 1, 2 * ry + 1, 0.1f);
  a[I_WALL + 0] = box_aff(rx + wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
  a[I_WALL + 1] = box_aff(-rx - wt / 2, 0, wh / 2, wt, 2 * ry + 2 * wt, wh);
  a[I_WALL + 2] = box_aff(0, ry + wt / 2, wh / 2, 2 * rx, wt, wh);
  a[I_WALL + 3] = box_aff(0, -ry - wt / 2, wh / 2, 2 * rx, wt, wh);
  uint32_t v[VIS_W] = {0u, 0u};
  auto on = [&](int i) { v[i >> 5] |= 1u << (i & 31); };
  on(I_FLOOR);
  for (int k = 0; k < 4; ++k) on(I_WALL + k);
  on(I_CUP);
  for (int k = 0; k < env::N_FURN; ++k) {
    const int cls = (int)s.fc[k];
    const bool use = k < (int)s.nf && cls >= 1 && cls <= N_FCLS;
    for (int c = 0; c < N_FCLS; ++c) {
      const int i = I_FURN + k * N_FCLS + c;
      if (use && c == cls - 1) {
        const float lx = h2f_dev(s.fb[k * 5]), ly = h2f_dev(s.fb[k * 5 + 1]), hx = h2f_dev(s.fb[k * 5 + 2]), hy = h2f_dev(s.fb[k * 5 + 3]), hz = h2f_dev(s.fb[k * 5 + 4]);
        a[i] = box_aff((lx + hx) * 0.5f, (ly + hy) * 0.5f, hz * 0.5f, hx - lx, hy - ly, hz);
        on(i);
      } else {
        a[i] = box_aff(0, 0, -10.f, 0.01f, 0.01f, 0.01f);
      }
    }
  }
  a[I_CUP] = box_aff(s.tx, s.ty, 0.05f, 0.08f, 0.08f, 0.10f);
  float sn, cs;
  sincosf(s.yaw, &sn, &cs);
  a[A_ROBOT] = Aff{{cs, -sn, 0, s.x, sn, cs, 0, s.y, 0, 0, 1, 0}};
  float R[9], p[3];
  wrist_pose(s.q, R, p);
  const float Rz[9] = {cs, -sn, 0, sn, cs, 0, 0, 0, 1};
  Aff w;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) w.m[4 * i + k] = Rz[3 * i] * R[k] + Rz[3 * i + 1] * R[3 + k] + Rz[3 * i + 2] * R[6 + k];
    w.m[4 * i + 3] = (i == 0 ? s.x : i == 1 ? s.y : 0.f) + Rz[3 * i] * p[0] + Rz[3 * i + 1] * p[1] + Rz[3 * i + 2] * p[2];
  }
  a[A_WRIST] = w;
  for (int k = 0; k < A - N_ANCH; ++k) a[N_ANCH + k] = a[A_ROBOT];
  for (int k = 0; k < Wv; ++k) vis[(size_t)e * Wv + k] = k < VIS_W ? v[k] : 0u;
}

static std::vector<CamRig> rigs() {
  return {rig(A_ROBOT, 0.094f, 0.18f, 1.18508f), rig(A_WRIST, 0.f, 0.f, 1.5184f)};
}

static void default_states(std::vector<bc::RenderState>& v, int E) {
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

Renderer* create(int E, int profile) {
  auto* r = new Renderer;
  r->E = E;
  HostScene H = make_scene(profile);
  r->rb.init(H, E, rigs());
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
  rs_anchor_k<<<(n + 127) / 128, 128, 0, st>>>(rs, n, r->rb.B.A, r->rb.B.W, r->rb.B.anchor, r->rb.B.vis);
  RCK(cudaGetLastError());
  r->rb.render(0, st);
}

int capture_cold_probe(int E, int profile) {
  gpu::RenderBatch rb;
  HostScene H = make_scene(profile);
  rb.init(H, E, rigs());
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
  return r->rb.scene.bytes + px * 2 * (4 + 3) + r->rb.work.cap * (sizeof(GPix) + 48) + (size_t)r->E * (N_ANCH * sizeof(Aff) * 3 + 4096);
}

}  // namespace bcr
