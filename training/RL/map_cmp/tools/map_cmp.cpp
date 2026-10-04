// 계획서 GPU_TRAINING.md 5.2: GPU 지도 근사판(../map, 잡음 끔 빌드) 대 진짜 scenemap(LIMO + OMX, CPU, C ABI)을 같은 입력으로.
//
//   map_cmp [N=50] [--mode slam|odom] [--policy 0|1] [--seed S] [--trace lane] [--stage 0|1]
//
// 한 판(N 개, 판마다 첫 에피소드만):
//   G1 환경(GPU) → 근사판 지도 A(GPU, 확정 규칙 그대로) + 음성 대조 B(GPU, 확정 규칙 끔 = map_verify --negative 와 같은 bug 1).
//   스텝마다 진짜 scenemap 에 LIMO proprio 12(근사판이 믿는 자세 = 오도메트리, 속도, 팔 5, 그리퍼).
//   근사판 keyframe 마다 진짜 scenemap 에 같은 순간의 입력:
//     - 깊이 640×400(Dabai, 근사판 MP::img_w/h·depth_hfov 의 fx, 정사각 화소): 근사판이 쏘는 같은 장면 상자(MapCore::prim + 방 벽·바닥·천장)를
//       근사판 카메라(베이스 + (cam_x, 0, cam_z), 기울기 0)에서 CPU 로 화소마다 렌더. 깊이 범위 밖(0.3–3.0 m)은 0(없음)
//     - 완벽한 검출: 렌더 id 버퍼의 물체마다 마스크(640×400 칸 그대로) + 상자, 이름 = 근사판 종류, 점수 0.9
//   → sm_snapshot 을 근사판 상태와 비교(5.2 표). 카메라 외부 자세는 sm_robot_fk cam 0 과 근사판 식을 keyframe 마다 비교.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"
#include "scenemap.h"

using namespace env;

namespace {

constexpr int W = gmap::MP::img_w, H = gmap::MP::img_h;
constexpr int NP = gmap::N_PRIM;
constexpr int MASK_WORDS = (W * H + 31) / 32;
const char* const kLabels[gmap::NCLS] = {"cup", "item", "chair", "table", "cabinet", "bin"};
enum CellCls { CU = 0, CF = 1, CO = 2 };   // 모름·빈·점유

struct Obj {
  int cls, prim, state;
  double pos[3], ext[3];
};

// 비교 한 쪽(근사 A 또는 음성 대조 B) 대 진짜 R 의 누적
struct Acc {
  long tp = 0, na = 0, nr = 0;                  // 확정 물체 집합(keyframe 마다 합)
  long tp_end = 0, na_end = 0, nr_end = 0;      // 판 끝 스냅숏
  long dup_a = 0, unm_a = 0;                    // 같은 참 물체에 둘 이상, 어느 참 물체와도 안 맞음
  std::vector<double> err;                      // 확정·맞은 물체 위치 오차(3D), keyframe × 물체
  std::vector<double> err_xy;
  long grid[3][3] = {{0}};                      // [이쪽 칸][진짜 칸], 방 + 0.15 m 칸
  long gridp[3][3] = {{0}};
  // 진단(모은 칸): 근사 모름·진짜 빈칸 칸의 카메라 거리 [<1.15, 1.15–2, 2–3, ≥3 m], 진짜만 점유인 칸의 자리 [벽 띠, 작은 물체(컵·물건), 가구, 빈 바닥]
  long uf_dist[4] = {0}, ro_where[4] = {0};                     // 같은 칸, 진짜 쪽은 0.05 m 칸 넷을 모아(점유 하나라도 → 점유, 아니면 빈칸 하나라도 → 빈칸)
  std::vector<int> kfc, kff, kfc_rel;           // 확정까지: 판 시작부터 keyframe 수, 처음 붙은 keyframe, (확정 − 처음)
  std::vector<int> pair_dc;                     // 둘 다 확정된 참 물체: 이쪽 확정 keyframe − 진짜 확정 keyframe
  long n_gone = 0;
  void merge(const Acc& o) {
    tp += o.tp; na += o.na; nr += o.nr; tp_end += o.tp_end; na_end += o.na_end; nr_end += o.nr_end;
    dup_a += o.dup_a; unm_a += o.unm_a; n_gone += o.n_gone;
    err.insert(err.end(), o.err.begin(), o.err.end());
    err_xy.insert(err_xy.end(), o.err_xy.begin(), o.err_xy.end());
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { grid[a][b] += o.grid[a][b]; gridp[a][b] += o.gridp[a][b]; }
    for (int k = 0; k < 4; ++k) { uf_dist[k] += o.uf_dist[k]; ro_where[k] += o.ro_where[k]; }
    kfc.insert(kfc.end(), o.kfc.begin(), o.kfc.end());
    kff.insert(kff.end(), o.kff.begin(), o.kff.end());
    kfc_rel.insert(kfc_rel.end(), o.kfc_rel.begin(), o.kfc_rel.end());
    pair_dc.insert(pair_dc.end(), o.pair_dc.begin(), o.pair_dc.end());
  }
};
struct RAcc {   // 진짜 쪽 혼자 값
  std::vector<double> err, err_xy;
  std::vector<int> kfc, kff, kfc_rel;
  long dup = 0, unm = 0, n_gone = 0;
  long n_kf = 0, n_det = 0, n_det_acc = 0;      // keyframe, 넣은 검출, objmap 이 받은 검출
  long obs_tp = 0, obs_a = 0, obs_r = 0;        // 이번 keyframe 에 관측된 참 물체: 근사(last_kf == t) 대 진짜(assoc ≠ 0)
  long obs_a_only[gmap::NCLS] = {0}, obs_r_only[gmap::NCLS] = {0};
  double pose_xy_max = 0, pose_yaw_max = 0, pose_xy_s2 = 0, pose_yaw_s2 = 0;
  long n_pose = 0;
  double ext_t_max = 0, ext_r_max = 0;          // 카메라 외부 자세: sm_robot_fk cam 0 − 근사판
  long n_ext = 0;
  long confirm_only_a[gmap::NCLS] = {0}, confirm_only_r[gmap::NCLS] = {0};   // 판 끝 확정이 한쪽에만(종류별)
  // 진단: 진짜만 관측한 참 물체의 근사판 쪽 까닭 — 0 중심 깊이 [ozmin, ozmax] 밖, 1 min_px·넓이(9 점 다 보여도), 2 보이는 점 0(시야 밖·가림),
  //       3 넓이 × 보이는 점 비율 < min_points, 4 근사판 검출 규칙은 넘음(기타)
  long why_a[5] = {0};
  long why_a_cls[5][gmap::NCLS] = {{0}};
  // 근사판만 관측한 참 물체의 진짜 쪽 까닭 — 0 렌더에 없음, 1 마스크 1 칸 깎은 안쪽 유효 깊이 화소 < min_points, 2 기타
  long why_r[3] = {0};
  std::vector<double> err_cls[gmap::NCLS], dz, drange;   // 진짜 위치 오차: 종류별 3D, 높이(물체 − 참), 수평 시선 방향 성분(+ = 카메라에서 멀어짐)
  void merge(const RAcc& o) {
    for (int k = 0; k < 5; ++k) { why_a[k] += o.why_a[k]; for (int c = 0; c < gmap::NCLS; ++c) why_a_cls[k][c] += o.why_a_cls[k][c]; }
    for (int k = 0; k < 3; ++k) why_r[k] += o.why_r[k];
    for (int c = 0; c < gmap::NCLS; ++c) err_cls[c].insert(err_cls[c].end(), o.err_cls[c].begin(), o.err_cls[c].end());
    dz.insert(dz.end(), o.dz.begin(), o.dz.end());
    drange.insert(drange.end(), o.drange.begin(), o.drange.end());
    err.insert(err.end(), o.err.begin(), o.err.end());
    err_xy.insert(err_xy.end(), o.err_xy.begin(), o.err_xy.end());
    kfc.insert(kfc.end(), o.kfc.begin(), o.kfc.end());
    kff.insert(kff.end(), o.kff.begin(), o.kff.end());
    kfc_rel.insert(kfc_rel.end(), o.kfc_rel.begin(), o.kfc_rel.end());
    dup += o.dup; unm += o.unm; n_gone += o.n_gone; n_kf += o.n_kf; n_det += o.n_det; n_det_acc += o.n_det_acc;
    obs_tp += o.obs_tp; obs_a += o.obs_a; obs_r += o.obs_r;
    for (int c = 0; c < gmap::NCLS; ++c) {
      obs_a_only[c] += o.obs_a_only[c]; obs_r_only[c] += o.obs_r_only[c];
      confirm_only_a[c] += o.confirm_only_a[c]; confirm_only_r[c] += o.confirm_only_r[c];
    }
    pose_xy_max = std::max(pose_xy_max, o.pose_xy_max); pose_yaw_max = std::max(pose_yaw_max, o.pose_yaw_max);
    pose_xy_s2 += o.pose_xy_s2; pose_yaw_s2 += o.pose_yaw_s2; n_pose += o.n_pose;
    ext_t_max = std::max(ext_t_max, o.ext_t_max); ext_r_max = std::max(ext_r_max, o.ext_r_max); n_ext += o.n_ext;
  }
};

struct Lane {
  sm_ctx* ctx = nullptr;
  int ep0 = -1;
  bool active = true, started = false;
  double ox = 0, oy = 0, oyaw = 0;   // 진짜 지도 원점(첫 proprio 의 오도메트리 자세 = 근사판이 믿는 자세)
  int kf = 0;
  int firstA[NP], confA[NP], firstB[NP], confB[NP], firstR[NP], confR[NP];
  gmap::Prim prim[NP];
  float rhx = 0, rhy = 0;
  // 작업 버퍼
  std::vector<float> depth;
  std::vector<int8_t> idb;
  std::vector<uint32_t> bits;
  Acc a, b;
  RAcc r;
  Lane() {
    for (int p = 0; p < NP; ++p) firstA[p] = confA[p] = firstB[p] = confB[p] = firstR[p] = confR[p] = -1;
  }
};

// 물체 → 참 물체(같은 종류, 중심이 상자 xy + 0.2 m 안이거나 중심 거리 < max(da_min, da_k·최대 변)). 가장 가까운 것
int match_prim(const Lane& L, int cls, const double pos[3]) {
  int best = -1;
  double bd = 1e30;
  for (int p = 0; p < NP; ++p) {
    const gmap::Prim& P = L.prim[p];
    if (P.cls != cls) continue;
    double c[3], e = 0;
    for (int a = 0; a < 3; ++a) { c[a] = 0.5 * (P.lo[a] + P.hi[a]); e = std::max(e, double(P.hi[a] - P.lo[a])); }
    const double dx = pos[0] - c[0], dy = pos[1] - c[1], dz = pos[2] - c[2];
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    const bool in_box = pos[0] > P.lo[0] - 0.2 && pos[0] < P.hi[0] + 0.2 && pos[1] > P.lo[1] - 0.2 && pos[1] < P.hi[1] + 0.2;
    if (!(in_box || d < std::max(double(gmap::MP::da_min), gmap::MP::da_k * e))) continue;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

void prim_center(const gmap::Prim& P, double c[3]) { for (int a = 0; a < 3; ++a) c[a] = 0.5 * (P.lo[a] + P.hi[a]); }

// 근사판 칸 → 3 종
int approx_cell(const int16_t* Lg, const uint32_t* seen, int idx) {
  if (!((seen[idx >> 5] >> (idx & 31)) & 1u)) return CU;
  const int v = Lg[idx];
  return v > 0 ? CO : v < 0 ? CF : CU;
}
int real_cell(const sm_grid& g, double mx, double my) {
  if (!g.cells || g.width <= 0) return CU;
  const int ix = (int)std::floor((mx - g.origin[0]) / g.resolution), iy = (int)std::floor((my - g.origin[1]) / g.resolution);
  if (ix < 0 || iy < 0 || ix >= g.width || iy >= g.height) return CU;
  const int v = g.cells[(size_t)iy * g.width + ix];
  return v < 0 ? CU : v > 50 ? CO : v < 50 ? CF : CU;
}

// 렌더: 화소 (u, v) 의 광학 방향 xn = (u − cx)/fx, yn = (v − cy)/fy, 근사판 pix_dir 로 세계 방향(전방 성분 1 → t = 광학 깊이).
// 근사판 cast() 와 같은 판 계산(ray_room + ray_box_inv)에 맞은 상자 번호를 더함
void render(const gmap::MapCore& m, const gmap::EnvView& e, float fx, float cx, float cy, std::vector<float>& depth, std::vector<int8_t>& idb) {
  float s, c;
  dm::sincosf_d(e.yaw, &s, &c);
  float o[3];
  gmap::cam_world(e.x, e.y, c, s, o);
  depth.assign((size_t)W * H, 0.f);
  idb.assign((size_t)W * H, -1);
  for (int v = 0; v < H; ++v) {
    const float yn = ((float)v - cy) / fx;
    for (int u = 0; u < W; ++u) {
      const float xn = ((float)u - cx) / fx;
      float d[3];
      gmap::pix_dir(c, s, xn, yn, d);
      float inv[3];
      gmap::ray_inv(d, inv);
      float t = gmap::ray_room(o, d, m.rhx, m.rhy);
      int id = -1;
      for (int p = 0; p < NP; ++p) {
        const float tb = gmap::ray_box_inv(o, d, inv, m.prim[p]);
        if (tb < t) { t = tb; id = p; }
      }
      const size_t i = (size_t)v * W + u;
      idb[i] = (int8_t)id;
      depth[i] = (t >= gmap::MP::zmin && t <= gmap::MP::zmax) ? t : 0.f;
    }
  }
}

double quant(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const size_t k = std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5));
  return v[k];
}
double median_i(const std::vector<int>& v) {
  std::vector<double> d(v.begin(), v.end());
  return quant(d, 0.5);
}

}  // namespace

int main(int argc, char** argv) {
  int N = 50, policy = 0, pose_mode = SM_POSE_SLAM, trace = -1, stage = 1, T = 2000;
  uint64_t seed = 20261004, mseed = 99;
  int pos = 0;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--mode") && a + 1 < argc) { ++a; pose_mode = !std::strcmp(argv[a], "odom") ? SM_POSE_ODOM : SM_POSE_SLAM; }
    else if (!std::strcmp(argv[a], "--policy") && a + 1 < argc) policy = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--seed") && a + 1 < argc) { seed = std::strtoull(argv[++a], nullptr, 10); mseed = seed * 7 + 1; }
    else if (!std::strcmp(argv[a], "--trace") && a + 1 < argc) trace = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--stage") && a + 1 < argc) stage = std::atoi(argv[++a]);
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
  }
  const gmap::Cam K = gmap::cam_consts();
  const float fx = K.fx, cx = 0.5f * (W - 1), cy = 0.5f * (H - 1);   // 화소 u 의 중심 = (u − cx)/fx, 좌우 대칭
  std::printf("map_cmp: N=%d episodes, stage %d, real scenemap mode %s, map update policy %d, depth %dx%d fx=fy=%.3f (H-FOV %.1f deg), range %.1f-%.1f m\n",
              N, stage, pose_mode == SM_POSE_SLAM ? "SLAM" : "ODOM", policy, W, H, fx, 2 * std::atan(0.5 * W / fx) * 180 / M_PI,
              gmap::MP::zmin, gmap::MP::zmax);

  DeviceEnv genv(N, stage, seed);
  gmap::DeviceMap amap(N, mseed), bmap(N, mseed);
  float *d_act, *d_obs, *d_rew;
  int* d_done;
  cudaMalloc((void**)&d_act, sizeof(float) * N_ACT * N);
  cudaMalloc((void**)&d_obs, sizeof(float) * N_OBS * N);
  cudaMalloc((void**)&d_rew, sizeof(float) * N);
  cudaMalloc((void**)&d_done, sizeof(int) * N);
  std::vector<float> act((size_t)N_ACT * N), obs((size_t)N_OBS * N), fg;
  std::vector<int> ig;
  std::vector<uint64_t> rg;
  gmap::MapHost ha, hb;

  std::vector<Lane> lanes(N);
  for (int i = 0; i < N; ++i) {
    Lane& L = lanes[i];
    const std::string cfg = "{\"robot\": \"limo_omx\", \"odom\": \"pose\"}";
    L.ctx = sm_create(cfg.c_str());
    if (!L.ctx) { std::fprintf(stderr, "sm_create failed\n"); return 2; }
    sm_set_labels(L.ctx, kLabels, gmap::NCLS);
    sm_set_pose_mode(L.ctx, pose_mode);
    sm_set_map_update(L.ctx, policy, 0);
  }
  uint64_t arng = 777;
  int n_active = N;
  long prim_mismatch = 0;
  for (int t = 0; t < T && n_active > 0; ++t) {
    // 행동: map_verify 와 같은 규칙(무작위 + 짝수 판은 접근 제어)
    for (size_t k = 0; k < act.size(); ++k) act[k] = dm::rand_range(arng, -1.f, 1.f);
    for (int i = 0; i < N; ++i) { act[0 * N + i] = dm::rand_range(arng, -0.2f, 1.f); act[1 * N + i] = dm::rand_range(arng, -0.6f, 0.6f); }
    if (t > 0) for (int i = 0; i < N; i += 2) { float a[N_ACT]; approach_action(obs.data(), N, i, a); act[0 * N + i] = a[0]; act[1 * N + i] = a[1]; }
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    genv.step(d_act, d_obs, d_rew, d_done);
    amap.step(genv.soa(), 0, 0);
    bmap.step(genv.soa(), 0, 1);   // 음성 대조: 확정 규칙 끔
    cudaDeviceSynchronize();
    cudaMemcpy(obs.data(), d_obs, sizeof(float) * obs.size(), cudaMemcpyDeviceToHost);
    genv.download(fg, ig, rg);
    amap.download(ha);
    bmap.download(hb);
    const Soa hs{fg.data(), ig.data(), rg.data(), N};

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : prim_mismatch)
    for (int i = 0; i < N; ++i) {
      Lane& L = lanes[i];
      if (!L.active) continue;
      const gmap::MapCore& A = ha.core[i];
      const gmap::MapCore& B = hb.core[i];
      if (L.ep0 < 0) L.ep0 = A.ep;
      if (A.ep != L.ep0) { L.active = false; continue; }   // 판마다 첫 에피소드만
      const gmap::EnvView e = gmap::read_env(hs, i);
      Core c;
      load(hs, i, c);
      if (!L.started) {
        std::memcpy(L.prim, A.prim, sizeof(L.prim));
        L.rhx = A.rhx; L.rhy = A.rhy;
        L.ox = A.ex; L.oy = A.ey; L.oyaw = A.eyaw;
        L.started = true;
      }
      if (std::memcmp(A.prim, B.prim, sizeof(A.prim))) ++prim_mismatch;
      // LIMO proprio 12: 오도메트리 자세(근사판이 믿는 자세), 속도, omx_joint1..5, 그리퍼
      float q[SM_LIMO_PROPRIO_DIM];
      q[SM_LIMO_ODOM_X] = A.ex; q[SM_LIMO_ODOM_Y] = A.ey; q[SM_LIMO_ODOM_YAW] = A.eyaw;
      q[SM_LIMO_VX] = e.v; q[SM_LIMO_VY] = 0.f; q[SM_LIMO_WZ] = e.w;
      for (int k = 0; k < 5; ++k) q[SM_LIMO_ARM_Q + k] = c.q[k];
      q[SM_LIMO_GRIPPER] = c.q[5];
      const double stamp = 0.1 * A.t;
      sm_proprio pp{stamp, q, SM_LIMO_PROPRIO_DIM};
      sm_push_proprio(L.ctx, &pp);
      if (!A.kf_flag) continue;
      if (!B.kf_flag) ++prim_mismatch;   // keyframe 은 확정 규칙과 무관해야 함
      L.kf += 1;
      const int kf = L.kf;
      L.r.n_kf += 1;

      // 카메라 외부 자세: sm_robot_fk cam 0(베이스 ← 광학) 대 근사판(베이스 + (cam_x, 0, cam_z), 광학 z = 베이스 x, x = −y, y = −z)
      {
        sm_body_fk fk;
        if (sm_robot_fk(SM_ROBOT_LIMO_OMX, q, SM_LIMO_PROPRIO_DIM, &fk) == 0 && fk.cam_valid[0]) {
          const double ref[12] = {0, 0, 1, env::K::cam_x, -1, 0, 0, 0, 0, -1, 0, env::K::cam_z};
          for (int k = 0; k < 12; ++k) {
            const double d = std::fabs(fk.T_cam[0][k] - ref[k]);
            if (k % 4 == 3) L.r.ext_t_max = std::max(L.r.ext_t_max, d);
            else L.r.ext_r_max = std::max(L.r.ext_r_max, d);
          }
          L.r.n_ext += 1;
        }
      }
      // 깊이 + 완벽한 검출
      render(A, e, fx, cx, cy, L.depth, L.idb);
      int cnt[NP] = {0}, bx0[NP], by0[NP], bx1[NP], by1[NP];
      for (int p = 0; p < NP; ++p) { bx0[p] = by0[p] = 1 << 30; bx1[p] = by1[p] = -1; }
      for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u) {
          const int p = L.idb[(size_t)v * W + u];
          if (p < 0) continue;
          ++cnt[p];
          bx0[p] = std::min(bx0[p], u); bx1[p] = std::max(bx1[p], u);
          by0[p] = std::min(by0[p], v); by1[p] = std::max(by1[p], v);
        }
      int nd = 0, det_prim[NP];
      int32_t dcls[NP];
      float dscore[NP], dbox[4 * NP];
      L.bits.assign((size_t)MASK_WORDS * NP, 0u);
      for (int p = 0; p < NP; ++p) {
        if (!cnt[p]) continue;
        det_prim[nd] = p;
        dcls[nd] = A.prim[p].cls;
        dscore[nd] = 0.9f;
        dbox[4 * nd + 0] = (float)bx0[p]; dbox[4 * nd + 1] = (float)by0[p];
        dbox[4 * nd + 2] = (float)(bx1[p] + 1); dbox[4 * nd + 3] = (float)(by1[p] + 1);
        uint32_t* mb = L.bits.data() + (size_t)nd * MASK_WORDS;
        for (int v = by0[p]; v <= by1[p]; ++v)
          for (int u = bx0[p]; u <= bx1[p]; ++u)
            if (L.idb[(size_t)v * W + u] == p) { const int k = v * W + u; mb[k >> 5] |= 1u << (k & 31); }
        ++nd;
      }
      // 마스크 배열을 검출 순서로 촘촘히(위에서 nd 번째 칸에 바로 썼음)
      sm_detections D{};
      D.stamp = stamp; D.cam = 0; D.img_w = W; D.img_h = H; D.n = nd;
      D.cls = dcls; D.score = dscore; D.box = dbox;
      D.mask_w = W; D.mask_h = H; D.mask_sx = 1.f; D.mask_sy = 1.f; D.mask_ox = 0.f; D.mask_oy = 0.f;
      D.mask_bits = L.bits.data();
      sm_image im{};
      im.stamp = stamp; im.cam = 0; im.w = W; im.h = H; im.rgba = nullptr; im.depth_m = L.depth.data();
      im.fx = fx; im.fy = fx; im.cx = cx; im.cy = cy;
      sm_push_image(L.ctx, &im, nd ? &D : nullptr);
      L.r.n_det += nd;
      uint32_t assoc[NP] = {0};
      const int na = nd ? sm_last_assoc(L.ctx, assoc, NP) : 0;
      bool obsR[NP] = {false}, obsA[NP] = {false};
      for (int k = 0; k < na && k < nd; ++k) if (assoc[k]) { obsR[det_prim[k]] = true; L.r.n_det_acc += 1; }

      sm_snapshot_t* snap = nullptr;
      sm_snapshot(L.ctx, &snap);
      const double c0 = std::cos(L.oyaw), s0 = std::sin(L.oyaw);
      auto to_world = [&](const double* p, double* w) {
        w[0] = L.ox + c0 * p[0] - s0 * p[1];
        w[1] = L.oy + s0 * p[0] + c0 * p[1];
        w[2] = p[2];
      };
      // 진짜 자세 대 참(근사판이 믿는 자세 = 잡음 끔이면 참)
      {
        const sm_pose2 P = sm_snap_pose(snap);
        const double mp[3] = {P.x, P.y, 0};
        double wp[3];
        to_world(mp, wp);
        const double exy = std::hypot(wp[0] - e.x, wp[1] - e.y);
        double dy = std::fmod(L.oyaw + P.yaw - e.yaw + 3 * M_PI, 2 * M_PI) - M_PI;
        dy = std::fabs(dy);
        L.r.pose_xy_max = std::max(L.r.pose_xy_max, exy); L.r.pose_yaw_max = std::max(L.r.pose_yaw_max, dy);
        L.r.pose_xy_s2 += exy * exy; L.r.pose_yaw_s2 += dy * dy; L.r.n_pose += 1;
      }
      // 진짜 확정 물체
      std::vector<Obj> R;
      {
        const sm_object* so = nullptr;
        const int n = sm_snap_objects(snap, &so);
        for (int k = 0; k < n; ++k) {
          Obj o;
          o.cls = -1;
          for (int cc = 0; cc < gmap::NCLS; ++cc) if (!std::strcmp(so[k].name, kLabels[cc])) o.cls = cc;
          to_world(so[k].pos, o.pos);
          for (int a = 0; a < 3; ++a) o.ext[a] = so[k].extent[a];
          o.state = so[k].state;
          o.prim = o.cls >= 0 ? match_prim(L, o.cls, o.pos) : -1;
          if (o.state == SM_GONE) { L.r.n_gone += 1; continue; }
          R.push_back(o);
        }
      }
      // 근사판 확정 물체(A, B)와 이번 keyframe 관측(last_kf == t)
      auto approx_objs = [&](const gmap::MapCore& M, std::vector<Obj>& out, bool* obs, long* gone, int* first) {
        for (int b = 0; b < gmap::KSLOT; ++b) {
          const gmap::Slot& S = M.slot[b];
          if (!S.valid) continue;
          Obj o;
          o.cls = S.cls;
          for (int a = 0; a < 3; ++a) { o.pos[a] = S.pos[a]; o.ext[a] = S.ext[a]; }
          o.state = S.state;
          o.prim = match_prim(L, o.cls, o.pos);
          if (o.prim >= 0 && first[o.prim] < 0) first[o.prim] = kf;
          if (obs && S.last_kf == M.t && o.prim >= 0) obs[o.prim] = true;
          if (!S.confirmed) continue;
          if (S.state == gmap::S_GONE) { *gone += 1; continue; }
          out.push_back(o);
        }
      };
      std::vector<Obj> OA, OB;
      approx_objs(A, OA, obsA, &L.a.n_gone, L.firstA);
      approx_objs(B, OB, nullptr, &L.b.n_gone, L.firstB);
      for (int p = 0; p < NP; ++p) if (obsR[p] && L.firstR[p] < 0) L.firstR[p] = kf;
      // 관측 일치(검출 규칙 차이 진단): 이번 keyframe 에 물체 기억에 들어간 참 물체
      for (int p = 0; p < NP; ++p) {
        L.r.obs_a += obsA[p]; L.r.obs_r += obsR[p]; L.r.obs_tp += obsA[p] && obsR[p];
        if (obsA[p] && !obsR[p]) L.r.obs_a_only[L.prim[p].cls] += 1;
        if (obsR[p] && !obsA[p]) L.r.obs_r_only[L.prim[p].cls] += 1;
      }
      {  // 진단: 관측이 한쪽에만 있는 까닭
        float sy, cyw;
        dm::sincosf_d(e.yaw, &sy, &cyw);
        float o3[3];
        gmap::cam_world(e.x, e.y, cyw, sy, o3);
        for (int p = 0; p < NP; ++p) {
          if (obsR[p] && !obsA[p]) {
            gmap::DetGeo g;
            int why = 4;
            const bool pre = gmap::det_prefilter(A, o3, cyw, sy, K, p, g);
            if (!(g.fwd >= gmap::MP::ozmin && g.fwd <= gmap::MP::ozmax)) why = 0;
            else if (!pre) why = 1;
            else {
              int nv = 0;
              for (int qq = 0; qq < gmap::NPT; ++qq) nv += gmap::vis_point(A, o3, cyw, sy, K, p * gmap::NPT + qq);
              if (nv == 0) why = 2;
              else if (g.af * ((float)nv / (float)gmap::NPT) < (float)gmap::MP::min_points) why = 3;
            }
            L.r.why_a[why] += 1;
            L.r.why_a_cls[why][L.prim[p].cls] += 1;
          }
          if (obsA[p] && !obsR[p]) {
            int why = 2;
            if (!cnt[p]) why = 0;
            else {
              int ne = 0;
              auto in = [&](int u, int v) { return u >= 0 && v >= 0 && u < W && v < H && L.idb[(size_t)v * W + u] == p; };
              for (int v = by0[p]; v <= by1[p]; ++v)
                for (int u = bx0[p]; u <= bx1[p]; ++u)
                  if (in(u, v) && in(u - 1, v) && in(u + 1, v) && in(u, v - 1) && in(u, v + 1) && L.depth[(size_t)v * W + u] > 0.f) ++ne;
              if (ne < gmap::MP::min_points) why = 1;
            }
            L.r.why_r[why] += 1;
          }
        }
      }
      // 집합: 맞은 참 물체 번호(중복 하나로) + 안 맞은 물체(각각 하나)
      auto set_of = [&](const std::vector<Obj>& O, bool* in, long* dup, long* unm) {
        int n = 0;
        for (const Obj& o : O) {
          if (o.prim < 0) { ++n; *unm += 1; continue; }
          if (in[o.prim]) { *dup += 1; continue; }
          in[o.prim] = true;
          ++n;
        }
        return n;
      };
      bool inR[NP] = {false}, inA[NP] = {false}, inB[NP] = {false};
      const int nR = set_of(R, inR, &L.r.dup, &L.r.unm);
      long dB = 0, uB = 0;
      const int nA = set_of(OA, inA, &L.a.dup_a, &L.a.unm_a);
      const int nB = set_of(OB, inB, &dB, &uB);
      L.b.dup_a += dB; L.b.unm_a += uB;
      int tpA = 0, tpB = 0;
      for (int p = 0; p < NP; ++p) { tpA += inA[p] && inR[p]; tpB += inB[p] && inR[p]; }
      L.a.tp += tpA; L.a.na += nA; L.a.nr += nR;
      L.b.tp += tpB; L.b.na += nB; L.b.nr += nR;
      // 판 끝 값은 마지막 keyframe 값으로 덮어씀(아래에서 판이 끝날 때 남은 값)
      L.a.tp_end = tpA; L.a.na_end = nA; L.a.nr_end = nR;
      L.b.tp_end = tpB; L.b.na_end = nB; L.b.nr_end = nR;
      for (int p = 0; p < NP; ++p) {
        if (inA[p] && L.confA[p] < 0) L.confA[p] = kf;
        if (inB[p] && L.confB[p] < 0) L.confB[p] = kf;
        if (inR[p] && L.confR[p] < 0) L.confR[p] = kf;
      }
      // 위치 오차(확정·맞은 물체, 3D 와 xy)
      auto pos_err = [&](const std::vector<Obj>& O, std::vector<double>& e3, std::vector<double>& e2) {
        for (const Obj& o : O) {
          if (o.prim < 0) continue;
          double c3[3];
          prim_center(L.prim[o.prim], c3);
          const double dx = o.pos[0] - c3[0], dy = o.pos[1] - c3[1], dz = o.pos[2] - c3[2];
          e3.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
          e2.push_back(std::sqrt(dx * dx + dy * dy));
        }
      };
      pos_err(R, L.r.err, L.r.err_xy);
      for (const Obj& o : R) {
        if (o.prim < 0) continue;
        double c3[3];
        prim_center(L.prim[o.prim], c3);
        const double dx = o.pos[0] - c3[0], dy = o.pos[1] - c3[1], dz = o.pos[2] - c3[2];
        L.r.err_cls[o.cls].push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
        L.r.dz.push_back(dz);
        const double vx = c3[0] - e.x, vy = c3[1] - e.y, vn = std::hypot(vx, vy);
        if (vn > 1e-6) L.r.drange.push_back((dx * vx + dy * vy) / vn);
      }
      pos_err(OA, L.a.err, L.a.err_xy);
      pos_err(OB, L.b.err, L.b.err_xy);
      // 격자: 근사판 칸(0.10 m) 중심이 방 + 0.15 m 안인 칸, 진짜 격자(0.05 m)는 그 중심 점을 진짜 map 좌표로 옮겨 읽음
      {
        sm_grid g{};
        sm_snap_map(snap, &g);
        const int16_t* LA = ha.L.data() + (size_t)i * gmap::NCELL;
        const uint32_t* SA = ha.seen.data() + (size_t)i * gmap::NWORD;
        const int16_t* LB = hb.L.data() + (size_t)i * gmap::NCELL;
        const uint32_t* SB = hb.seen.data() + (size_t)i * gmap::NWORD;
        for (int ly = 0; ly < gmap::GW; ++ly)
          for (int lx = 0; lx < gmap::GW; ++lx) {
            const double wx = ((double)(lx + gmap::GX0) + 0.5) * gmap::RES, wy = ((double)(ly + gmap::GX0) + 0.5) * gmap::RES;
            if (std::fabs(wx) > L.rhx + 0.15 || std::fabs(wy) > L.rhy + 0.15) continue;
            const double dx = wx - L.ox, dy = wy - L.oy;
            const int rc = real_cell(g, c0 * dx + s0 * dy, -s0 * dx + c0 * dy);
            int rp = CU;
            for (int k = 0; k < 4; ++k) {
              const double ddx = dx + ((k & 1) ? 0.025 : -0.025), ddy = dy + ((k & 2) ? 0.025 : -0.025);
              const int r4 = real_cell(g, c0 * ddx + s0 * ddy, -s0 * ddx + c0 * ddy);
              if (r4 == CO) rp = CO;
              else if (r4 == CF && rp == CU) rp = CF;
            }
            const int idx = ly * gmap::GW + lx;
            const int ca = approx_cell(LA, SA, idx), cb = approx_cell(LB, SB, idx);
            L.a.grid[ca][rc] += 1;
            L.b.grid[cb][rc] += 1;
            L.a.gridp[ca][rp] += 1;
            L.b.gridp[cb][rp] += 1;
            if (ca == CU && rp == CF) {
              float sy2, cy2;
              dm::sincosf_d(e.yaw, &sy2, &cy2);
              const double dd = std::hypot(wx - (e.x + cy2 * env::K::cam_x), wy - (e.y + sy2 * env::K::cam_x));
              L.a.uf_dist[dd < 1.15 ? 0 : dd < 2.0 ? 1 : dd < 3.0 ? 2 : 3] += 1;
            }
            if (ca != CO && rp == CO) {
              int where = 3;
              if (std::fabs(wx) > L.rhx - 0.1 || std::fabs(wy) > L.rhy - 0.1) where = 0;
              else
                for (int pp = 0; pp < NP; ++pp) {
                  const gmap::Prim& P = L.prim[pp];
                  if (wx > P.lo[0] - 0.1 && wx < P.hi[0] + 0.1 && wy > P.lo[1] - 0.1 && wy < P.hi[1] + 0.1) {
                    where = (P.cls == gmap::C_CUP || P.cls == gmap::C_ITEM) ? 1 : 2;
                    break;
                  }
                }
              L.a.ro_where[where] += 1;
            }
          }
      }
      if (i == trace) {
        std::printf("[trace lane %d kf %d t %d] true (%.2f %.2f %.2f) dets %d accepted:", i, kf, A.t, e.x, e.y, e.yaw, nd);
        for (int p = 0; p < NP; ++p) if (cnt[p]) std::printf(" p%d(%s,%dpx,A%d,R%d)", p, kLabels[L.prim[p].cls], cnt[p], (int)obsA[p], (int)obsR[p]);
        std::printf("\n   A conf:");
        for (const Obj& o : OA) std::printf(" %s->p%d(%.2f,%.2f,%.2f)", kLabels[o.cls], o.prim, o.pos[0], o.pos[1], o.pos[2]);
        std::printf("\n   R conf:");
        for (const Obj& o : R) std::printf(" %s->p%d(%.2f,%.2f,%.2f)", o.cls >= 0 ? kLabels[o.cls] : "?", o.prim, o.pos[0], o.pos[1], o.pos[2]);
        std::printf("\n");
      }
      sm_snapshot_release(snap);
    }
    n_active = 0;
    for (const Lane& L : lanes) n_active += L.active;
  }
  // 판 끝: 확정까지 걸린 keyframe(참 물체마다)
  Acc A, B;
  RAcc R;
  for (Lane& L : lanes) {
    for (int p = 0; p < NP; ++p) {
      if (L.confA[p] >= 0) { L.a.kfc.push_back(L.confA[p]); L.a.kff.push_back(L.firstA[p]); L.a.kfc_rel.push_back(L.confA[p] - L.firstA[p]); }
      if (L.confB[p] >= 0) { L.b.kfc.push_back(L.confB[p]); L.b.kff.push_back(L.firstB[p]); L.b.kfc_rel.push_back(L.confB[p] - L.firstB[p]); }
      if (L.confR[p] >= 0) { L.r.kfc.push_back(L.confR[p]); L.r.kff.push_back(L.firstR[p]); L.r.kfc_rel.push_back(L.confR[p] - (L.firstR[p] >= 0 ? L.firstR[p] : L.confR[p])); }
      if (L.confA[p] >= 0 && L.confR[p] >= 0) L.a.pair_dc.push_back(L.confA[p] - L.confR[p]);
      if (L.confB[p] >= 0 && L.confR[p] >= 0) L.b.pair_dc.push_back(L.confB[p] - L.confR[p]);
      if (L.confA[p] >= 0 && L.confR[p] < 0) L.r.confirm_only_a[L.prim[p].cls] += 1;
      if (L.confR[p] >= 0 && L.confA[p] < 0) L.r.confirm_only_r[L.prim[p].cls] += 1;
    }
    A.merge(L.a);
    B.merge(L.b);
    R.merge(L.r);
    sm_destroy(L.ctx);
  }
  if (prim_mismatch) std::printf("WARNING: A/B scene or keyframe mismatch %ld\n", prim_mismatch);

  std::printf("\nkeyframes %ld, detections pushed %ld, accepted by real objmap %ld\n", R.n_kf, R.n_det, R.n_det_acc);
  std::printf("camera extrinsics sm_robot_fk cam0 vs approx (base+(%.3f,0,%.3f), level): max |dt| %.3g m, max |dR| %.3g over %ld keyframes\n",
              env::K::cam_x, env::K::cam_z, R.ext_t_max, R.ext_r_max, R.n_ext);
  std::printf("real pose vs truth at keyframes: rms xy %.4f m, yaw %.4f deg; max xy %.4f m, yaw %.3f deg\n",
              std::sqrt(R.pose_xy_s2 / std::max(1L, R.n_pose)), std::sqrt(R.pose_yaw_s2 / std::max(1L, R.n_pose)) * 180 / M_PI,
              R.pose_xy_max, R.pose_yaw_max * 180 / M_PI);
  std::printf("observed-this-keyframe (true objects entering object memory): approx %ld, real %ld, both %ld -> approx precision %.3f recall %.3f\n",
              R.obs_a, R.obs_r, R.obs_tp, R.obs_tp / std::max(1.0, (double)R.obs_a), R.obs_tp / std::max(1.0, (double)R.obs_r));
  std::printf("  approx-only by class:");
  for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %ld", kLabels[c], R.obs_a_only[c]);
  std::printf("\n  real-only by class:  ");
  for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %ld", kLabels[c], R.obs_r_only[c]);
  std::printf("\nconfirmed by episode end only in approx (by class):");
  for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %ld", kLabels[c], R.confirm_only_a[c]);
  std::printf("\nconfirmed by episode end only in real (by class):  ");
  for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %ld", kLabels[c], R.confirm_only_r[c]);
  std::printf("\nreal: duplicates %ld, unmatched confirmed %ld, gone %ld (keyframe x object)\n", R.dup, R.unm, R.n_gone);
  const char* why_a_name[5] = {"center depth outside [ozmin,ozmax]", "min_px / area prefilter", "no visible point of 9 (FOV/occlusion)",
                               "area x visible share < min_points", "passes approx rule (other)"};
  std::printf("why approx did not observe (real did):\n");
  for (int k = 0; k < 5; ++k) {
    std::printf("  %-40s %5ld  [", why_a_name[k], R.why_a[k]);
    for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %ld", kLabels[c], R.why_a_cls[k][c]);
    std::printf(" ]\n");
  }
  std::printf("why real did not observe (approx did): not rendered %ld, eroded valid-depth px < min_points %ld, other %ld\n", R.why_r[0], R.why_r[1], R.why_r[2]);
  std::printf("real position error by class q50/q90 [m]:");
  for (int c = 0; c < gmap::NCLS; ++c) std::printf(" %s %.3f/%.3f (n %zu)", kLabels[c], quant(R.err_cls[c], 0.5), quant(R.err_cls[c], 0.9), R.err_cls[c].size());
  std::printf("\nreal position error components: height (obj - true) q10/q50/q90 %.3f/%.3f/%.3f m; along horizontal view ray q10/q50/q90 %.3f/%.3f/%.3f m\n",
              quant(R.dz, 0.1), quant(R.dz, 0.5), quant(R.dz, 0.9), quant(R.drange, 0.1), quant(R.drange, 0.5), quant(R.drange, 0.9));

  auto report = [&](const char* name, const Acc& X) {
    const double prec = X.tp / std::max(1.0, (double)X.na), rec = X.tp / std::max(1.0, (double)X.nr);
    const double mA = median_i(X.kfc_rel), mR = median_i(R.kfc_rel);
    const double qa50 = quant(X.err, 0.5), qa90 = quant(X.err, 0.9), qr50 = quant(R.err, 0.5), qr90 = quant(R.err, 0.9);
    long gs = 0, ge = 0, ks = 0, ke = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        gs += X.grid[a][b];
        ge += a == b ? X.grid[a][b] : 0;
        if (a != CU || b != CU) { ks += X.grid[a][b]; ke += a == b ? X.grid[a][b] : 0; }
      }
    const double gag = ge / std::max(1.0, (double)gs), gak = ke / std::max(1.0, (double)ks);
    long ps = 0, pe = 0, pks = 0, pke = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        ps += X.gridp[a][b];
        pe += a == b ? X.gridp[a][b] : 0;
        if (a != CU || b != CU) { pks += X.gridp[a][b]; pke += a == b ? X.gridp[a][b] : 0; }
      }
    std::vector<double> pdc(X.pair_dc.begin(), X.pair_dc.end());
    long same = 0;
    for (int d : X.pair_dc) same += d == 0;
    std::printf("\n== %s vs real ==\n", name);
    std::printf("| metric | approx | real | value | criterion | result |\n|---|---|---|---|---|---|\n");
    std::printf("| confirmed set precision (keyframe sum) | %ld | %ld | %.3f | >= 0.9 | %s |\n", X.na, X.nr, prec, prec >= 0.9 ? "PASS" : "FAIL");
    std::printf("| confirmed set recall (keyframe sum) | | | %.3f | >= 0.9 | %s |\n", rec, rec >= 0.9 ? "PASS" : "FAIL");
    std::printf("| confirmed set precision / recall at episode end | %ld | %ld | %.3f / %.3f | (info) | |\n", X.na_end, X.nr_end,
                X.tp_end / std::max(1.0, (double)X.na_end), X.tp_end / std::max(1.0, (double)X.nr_end));
    std::printf("| keyframes first-seen -> confirmed, median | %.1f (n %zu) | %.1f (n %zu) | %.1f | <= 1 | %s |\n", mA, X.kfc_rel.size(), mR,
                R.kfc_rel.size(), std::fabs(mA - mR), std::fabs(mA - mR) <= 1 ? "PASS" : "FAIL");
    std::printf("| keyframes from episode start to confirmed, median | %.1f | %.1f | %.1f | (info) | |\n", median_i(X.kfc), median_i(R.kfc),
                std::fabs(median_i(X.kfc) - median_i(R.kfc)));
    const double pmed = quant(pdc, 0.5);
    {
      long lt = 0, eq = 0, gt = 0;
      for (int d : X.pair_dc) (d < 0 ? lt : d > 0 ? gt : eq) += 1;
      std::printf("paired confirm keyframe (approx - real): < 0: %ld, = 0: %ld, > 0: %ld\n", lt, eq, gt);
    }
    std::printf("| same true object: confirm kf (approx - real), median / share equal | | | %.1f / %.3f (n %zu) | abs(median) < 1 (tightened) | %s |\n",
                pmed, same / std::max(1.0, (double)X.pair_dc.size()), X.pair_dc.size(), std::fabs(pmed) < 1 ? "PASS" : "FAIL");
    std::printf("| keyframes first-seen -> confirmed, median difference, tightened | | | %.1f | < 1 | %s |\n", std::fabs(mA - mR), std::fabs(mA - mR) < 1 ? "PASS" : "FAIL");
    std::printf("| position error 3D q50 [m] | %.3f | %.3f | %.3f | <= 0.05 | %s |\n", qa50, qr50, std::fabs(qa50 - qr50), std::fabs(qa50 - qr50) <= 0.05 ? "PASS" : "FAIL");
    std::printf("| position error 3D q90 [m] | %.3f | %.3f | %.3f | <= 0.05 | %s |\n", qa90, qr90, std::fabs(qa90 - qr90), std::fabs(qa90 - qr90) <= 0.05 ? "PASS" : "FAIL");
    std::printf("| position error xy q50 / q90 [m] | %.3f / %.3f | %.3f / %.3f | | (info) | |\n", quant(X.err_xy, 0.5), quant(X.err_xy, 0.9),
                quant(R.err_xy, 0.5), quant(R.err_xy, 0.9));
    std::printf("| grid 3-class agreement (room + 0.15 m cells) | | | %.3f | >= 0.9 | %s |\n", gag, gag >= 0.9 ? "PASS" : "FAIL");
    std::printf("| grid agreement, cells known in either | | | %.3f | (info) | |\n", gak);
    std::printf("| grid agreement, real 0.05 m cells pooled to 0.10 m: all / known in either | | | %.3f / %.3f | (info) | |\n",
                pe / std::max(1.0, (double)ps), pke / std::max(1.0, (double)pks));
    std::printf("grid confusion [approx row: unknown free occ] x [real col: unknown free occ] (cell x keyframe), center sample | pooled:\n");
    for (int a = 0; a < 3; ++a)
      std::printf("  %10ld %10ld %10ld   | %10ld %10ld %10ld\n", X.grid[a][0], X.grid[a][1], X.grid[a][2], X.gridp[a][0], X.gridp[a][1], X.gridp[a][2]);
    std::printf("approx: duplicates %ld, unmatched confirmed %ld, gone %ld\n", X.dup_a, X.unm_a, X.n_gone);
    if (&X == &A) std::printf("pooled approx-unknown/real-free cells by distance from camera: <1.15 m %ld, 1.15-2 m %ld, 2-3 m %ld, >=3 m %ld\n",
                X.uf_dist[0], X.uf_dist[1], X.uf_dist[2], X.uf_dist[3]);
    if (&X == &A) std::printf("pooled real-occupied/approx-not cells: wall band %ld, small object (cup/item) %ld, furniture %ld, open floor %ld\n",
                X.ro_where[0], X.ro_where[1], X.ro_where[2], X.ro_where[3]);
    const bool rule = prec >= 0.9 && rec >= 0.9 && std::fabs(mA - mR) <= 1 && gag >= 0.9;
    const bool pos = std::fabs(qa50 - qr50) <= 0.05 && std::fabs(qa90 - qr90) <= 0.05;
    const bool tight = std::fabs(mA - mR) < 1 && std::fabs(pmed) < 1;
    std::printf("%s: doc 5.2 criteria without position: %s; position: %s; tightened confirm-timing: %s\n", name, rule ? "PASS" : "FAIL",
                pos ? "PASS" : "FAIL", tight ? "PASS" : "FAIL");
    return rule && tight;
  };
  const bool apass = report("approx (confirm rule on)", A);
  const bool negpass = report("negative control (approx confirm rule off)", B);
  std::printf("\nrule comparison (doc criteria except position + tightened confirm timing): approx %s, negative control %s -> %s\n",
              apass ? "PASS" : "FAIL", negpass ? "PASS" : "FAIL",
              apass && !negpass ? "OK (only the broken rule fails)" : "NOT OK");
  return 0;
}
