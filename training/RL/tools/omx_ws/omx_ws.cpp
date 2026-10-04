// OMX-F 작업 공간 표 만들기(VLA_INPUT 3절 "팔이 닿는지": 리모를 안 움직이고 OMX 만으로 닿는지, URDF 관절 한계로 미리 계산).
// 같은 순기구학(env.h fk = limo_omx_model.h, urdf2hdr 가 URDF 에서 생성)과 같은 관절 한계(env::K::q_lo/q_hi = src/robot/real_limits.json)로
// joint2·3·4·5 를 촘촘히 훑어 손끝(omx_end_effector_link) 자리를 (r, z) 칸에 찍는다. joint1 은 세로축(base_link z)이고 한계가 360° 를 덮으므로
// (r = joint1 축에서 수평 거리, z = base_link 높이) 2D 표면 충분하다(손끝의 y 0.0016 m 치우침은 joint5 를 같이 훑어 칸에 들어감).
// 바닥 아래(z < −base_z, base_footprint 아래)는 뺀다. 표본 사이 틈은 닫힘(팽창 1 칸 → 침식 1 칸)으로 메운다.
// 출력: training/RL/map/include/omx_workspace.h (손으로 고치지 않는다). 빌드·실행:
//   g++ -std=c++17 -O2 -ffp-contract=off -I training/RL/env/include training/RL/tools/omx_ws/omx_ws.cpp -o /tmp/omx_ws && /tmp/omx_ws > training/RL/map/include/omx_workspace.h
// --grasp: 잡는 점(E0, CURRICULUM_BEHAVIOR2026 5.3) = omx_end_effector_link 에서 손가락 축(그 링크 x)으로 −0.0119 m 로 같은 표를 만든다
//   (namespace omxwsg, 매크로 OMX_WSG_ROWS) → training/RL/env/include/omx_workspace_grasp.h. B3 다가가기 성공 판정(env_beh.h)이 쓴다
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "env.h"

int main(int argc, char** argv) {
  const bool grasp = argc > 1 && std::string(argv[1]) == "--grasp";
  constexpr float kGraspOff = -0.0119f;   // E0 잰 값(link5 x 0.080 = ee 링크 −0.0119 m)
  constexpr float RES = 0.01f, R_MAX = 0.50f, Z_LO = -0.15f, Z_HI = 0.55f;
  constexpr int NR = 50, NZ = 70;
  static_assert(NR <= 64, "one uint64 row");
  std::vector<uint8_t> g((size_t)NR * NZ, 0);
  // joint1 축의 base_link 자리(관절각과 무관): mount + R_mount·t_j1
  const auto& Jm = limo_omx::joint(limo_omx::J_OMX_MOUNT_JOINT);
  const auto& J1 = limo_omx::joint(limo_omx::J_OMX_JOINT1);
  float j1[3];
  env::mat_vec(Jm.R, J1.t, j1);
  const float ax = Jm.t[0] + j1[0], ay = Jm.t[1] + j1[1];
  const int S2 = 241, S3 = 241, S4 = 241, S5 = 5;
  long long n_in = 0, n_floor = 0, n_out = 0;
  for (int a = 0; a < S2; ++a)
    for (int b = 0; b < S3; ++b)
      for (int c = 0; c < S4; ++c)
        for (int d = 0; d < S5; ++d) {
          float q[env::N_Q] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, qd[env::N_Q] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          q[1] = env::K::q_lo(1) + (env::K::q_hi(1) - env::K::q_lo(1)) * (float)a / (float)(S2 - 1);
          q[2] = env::K::q_lo(2) + (env::K::q_hi(2) - env::K::q_lo(2)) * (float)b / (float)(S3 - 1);
          q[3] = env::K::q_lo(3) + (env::K::q_hi(3) - env::K::q_lo(3)) * (float)c / (float)(S4 - 1);
          q[4] = -1.5707964f + 3.1415927f * (float)d / (float)(S5 - 1);
          env::Fk f;
          env::fk(q, qd, f);
          float P[3] = {f.ee_p[0], f.ee_p[1], f.ee_p[2]};
          if (grasp) for (int k = 0; k < 3; ++k) P[k] = P[k] + kGraspOff * f.ee_R[3 * k];   // 링크 x 축 = ee_R 첫 열
          const float r = std::sqrt((P[0] - ax) * (P[0] - ax) + (P[1] - ay) * (P[1] - ay)), z = P[2];
          if (z < -0.15f) { ++n_floor; continue; }   // base_footprint 아래 = 바닥 속
          const int ir = (int)std::floor(r / RES), iz = (int)std::floor((z - Z_LO) / RES);
          if (ir < 0 || ir >= NR || iz < 0 || iz >= NZ) { ++n_out; continue; }
          g[(size_t)iz * NR + ir] = 1;
          ++n_in;
        }
  auto at = [&](const std::vector<uint8_t>& v, int iz, int ir) { return (iz >= 0 && iz < NZ && ir >= 0 && ir < NR) ? v[(size_t)iz * NR + ir] : (uint8_t)0; };
  std::vector<uint8_t> dil(g.size(), 0), clo(g.size(), 0);
  for (int iz = 0; iz < NZ; ++iz)
    for (int ir = 0; ir < NR; ++ir) {
      uint8_t v = 0;
      for (int dz = -1; dz <= 1; ++dz) for (int dr = -1; dr <= 1; ++dr) v |= at(g, iz + dz, ir + dr);
      dil[(size_t)iz * NR + ir] = v;
    }
  int n_cells = 0, n_fill = 0;
  for (int iz = 0; iz < NZ; ++iz)
    for (int ir = 0; ir < NR; ++ir) {
      uint8_t v = 1;
      for (int dz = -1; dz <= 1; ++dz) for (int dr = -1; dr <= 1; ++dr) {
        const int z2 = iz + dz, r2 = ir + dr;
        if (z2 < 0 || z2 >= NZ || r2 < 0 || r2 >= NR) continue;   // 표 밖은 침식에 쓰지 않음(경계 칸을 깎지 않게)
        v &= dil[(size_t)z2 * NR + r2];
      }
      v |= g[(size_t)iz * NR + ir];
      clo[(size_t)iz * NR + ir] = v;
      n_cells += v;
      n_fill += v && !g[(size_t)iz * NR + ir];
    }
  float rmax = 0.f, zmin = 1e9f, zmax = -1e9f;
  for (int iz = 0; iz < NZ; ++iz) for (int ir = 0; ir < NR; ++ir) if (clo[(size_t)iz * NR + ir]) {
    rmax = std::fmax(rmax, (ir + 1) * RES); zmin = std::fmin(zmin, Z_LO + iz * RES); zmax = std::fmax(zmax, Z_LO + (iz + 1) * RES);
  }
  const char* ns = grasp ? "omxwsg" : "omxws";
  const char* mac = grasp ? "OMX_WSG_ROWS" : "OMX_WS_ROWS";
  if (grasp) std::printf("// 생성: training/RL/tools/omx_ws/omx_ws.cpp --grasp (손으로 고치지 않는다). OMX-F **잡는 점**(omx_end_effector_link 에서 링크 x 로 −0.0119 m, E0) 작업 공간, URDF 순기구학 +\n");
  else std::printf("// 생성: training/RL/tools/omx_ws/omx_ws.cpp (손으로 고치지 않는다). OMX-F 손끝(omx_end_effector_link) 작업 공간, URDF 순기구학 +\n");
  std::printf("// 관절 한계 env::K::q_lo/q_hi(src/robot/real_limits.json). joint2·3·4 %d×%d×%d, joint5 %d 값(±90°) 훑음, 바닥(z < −0.15 m) 뺌, 닫힘 1 칸.\n", S2, S3, S4, S5);
  std::printf("// 표: (r = joint1 축에서 수평 거리, z = base_link 높이) %d × %d 칸, %.2f m. 칸 %d 개(그중 닫힘으로 메운 칸 %d), 표본 %lld 개(바닥 아래 %lld, 표 밖 %lld).\n",
              NR, NZ, RES, n_cells, n_fill, n_in, n_floor, n_out);
  std::printf("// 범위: r ≤ %.2f m, z %.2f – %.2f m (base_link). joint1 축 base_link 자리 (%.5f, %.5f).\n", rmax, zmin, zmax, ax, ay);
  std::printf("#pragma once\n#include <cstdint>\nnamespace %s {\n", ns);
  std::printf("constexpr float RES = %#.9gf, Z_LO = %#.9gf, AX = %#.9gf, AY = %#.9gf;\nconstexpr int NR = %d, NZ = %d;\n", RES, Z_LO, ax, ay, NR, NZ);
  std::printf("#define %s \\\n  { \\\n", mac);
  for (int iz = 0; iz < NZ; ++iz) {
    uint64_t w = 0;
    for (int ir = 0; ir < NR; ++ir) if (clo[(size_t)iz * NR + ir]) w |= 1ull << ir;
    std::printf("    0x%016llxull,%s \\\n", (unsigned long long)w, "");
  }
  std::printf("  }\n");
  std::printf("constexpr uint64_t kRows[NZ] = %s;   // 호스트\n#ifdef __CUDACC__\nstatic __constant__ uint64_t kRowsDev[NZ] = %s;   // 장치(같은 표)\n#endif\n", mac, mac);
  std::printf("#ifdef __CUDACC__\n__host__ __device__ __forceinline__\n#else\ninline\n#endif\nuint64_t row(int iz) {\n#ifdef __CUDA_ARCH__\n  return kRowsDev[iz];\n#else\n  return kRows[iz];\n#endif\n}\n");
  std::printf("}  // namespace %s\n", ns);
  return 0;
}
