// 생성: training/RL/tools/omx_ws/omx_ws.cpp (손으로 고치지 않는다). OMX-F 손끝(omx_end_effector_link) 작업 공간, URDF 순기구학 +
// 관절 한계 env::K::q_lo/q_hi(src/robot/real_limits.json). joint2·3·4 241×241×241, joint5 5 값(±90°) 훑음, 바닥(z < −0.15 m) 뺌, 닫힘 1 칸.
// 표: (r = joint1 축에서 수평 거리, z = base_link 높이) 50 × 70 칸, 0.01 m. 칸 2201 개(그중 닫힘으로 메운 칸 2), 표본 68173389 개(바닥 아래 1814216, 표 밖 0).
// 범위: r ≤ 0.41 m, z -0.15 – 0.51 m (base_link). joint1 축 base_link 자리 (-0.05125, 0.00000).
#pragma once
#include <cstdint>
namespace omxws {
constexpr float RES = 0.00999999978f, Z_LO = -0.150000006f, AX = -0.0512499996f, AY = 0.00000000f;
constexpr int NR = 50, NZ = 70;
#define OMX_WS_ROWS \
  { \
    0x00000001ffffffffull, \
    0x00000001ffffffffull, \
    0x00000003ffffffffull, \
    0x00000007ffffffffull, \
    0x00000007ffffffffull, \
    0x0000000fffffffffull, \
    0x0000001fffffffffull, \
    0x0000001fffffffffull, \
    0x0000003fffffffffull, \
    0x0000003fffffffffull, \
    0x0000003fffffffffull, \
    0x0000007fffffffffull, \
    0x0000007fffffffffull, \
    0x0000007fffffffffull, \
    0x000000ffffffffffull, \
    0x000000ffffffffffull, \
    0x000000ffffffffffull, \
    0x000000ffffffffffull, \
    0x000000ffffffffffull, \
    0x000001fffffffffcull, \
    0x000001fffffffff0ull, \
    0x000001ffffffffe0ull, \
    0x000001ffffffffe0ull, \
    0x000001ffffffffc0ull, \
    0x000001ffffffffc0ull, \
    0x000001ffffffffc0ull, \
    0x000001ffffffffe0ull, \
    0x000001ffffffffe0ull, \
    0x000001fffffffff0ull, \
    0x000001fffffffff0ull, \
    0x000000fffffffff8ull, \
    0x000000fffffffff8ull, \
    0x000000fffffffffcull, \
    0x000000fffffffffcull, \
    0x000000ffffffffffull, \
    0x0000007fffffffffull, \
    0x0000007fffffffffull, \
    0x0000007fffffffffull, \
    0x0000007fffffffffull, \
    0x0000003fffffffffull, \
    0x0000003fffffffffull, \
    0x0000001fffffffffull, \
    0x0000001fffffffffull, \
    0x0000000fffffffffull, \
    0x0000000fffffffffull, \
    0x00000007ffffffffull, \
    0x00000007ffffffffull, \
    0x00000003ffffffffull, \
    0x00000001ffffffffull, \
    0x00000001ffffffffull, \
    0x00000000ffffffffull, \
    0x000000007fffffffull, \
    0x000000003fffffffull, \
    0x000000001fffffffull, \
    0x000000000fffffffull, \
    0x0000000007ffffffull, \
    0x0000000003ffffffull, \
    0x0000000001ffffffull, \
    0x00000000007fffffull, \
    0x00000000003fffffull, \
    0x00000000000fffffull, \
    0x000000000003ffffull, \
    0x000000000000ffffull, \
    0x0000000000001fffull, \
    0x00000000000003ffull, \
    0x0000000000000007ull, \
    0x0000000000000000ull, \
    0x0000000000000000ull, \
    0x0000000000000000ull, \
    0x0000000000000000ull, \
  }
constexpr uint64_t kRows[NZ] = OMX_WS_ROWS;   // 호스트
#ifdef __CUDACC__
static __constant__ uint64_t kRowsDev[NZ] = OMX_WS_ROWS;   // 장치(같은 표)
#endif
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
uint64_t row(int iz) {
#ifdef __CUDA_ARCH__
  return kRowsDev[iz];
#else
  return kRows[iz];
#endif
}
}  // namespace omxws
