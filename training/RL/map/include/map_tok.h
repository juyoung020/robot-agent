// 지도 토큰(계획서 GPU_TRAINING.md 5.3·4.4, VLA_INPUT.md 3·4절) — 스텝마다 판마다 1,360 B(v3: v2 1,296 B + 목표 칸 2 × 16, 2026-10-05). map.h 끝에서 include 된다.
// CPU 참조판과 GPU 커널이 같은 make_tokens 를 쓴다(GPU 는 판 하나 = 레인 16, CPU 는 레인 1). 모든 값은 믿는(slam) 자세의 base_link 기준이다.
//   물체 칸 16 × 숫자 33 (FP16) + 이름 표 번호·생김새 표 번호(int16, 4.4: 벡터는 장치 표에, 관측에는 번호만) + 벽 56 (FP16) + 방 10 (FP16)
//   + 완성도 4 (FP16) + 칸 수·표시 + 안 본 곳 광선 8 + 다음 경유 지점 4 (FP16). 값은 m·rad 그대로이고 정규화(5 m 자르기·로그·자료 통계)는 관측 쪽(obs.h)이 한다.
// 칸 고르기: 확정 물체(scenemap 스냅숏처럼), 목표 물체가 맨 앞, 나머지는 로봇에서 가까운 순(VLA_INPUT 3절). 빈 칸 = 0, 번호 −1.
#pragma once
#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif
#include "omx_workspace_grasp.h"   // 팔이 닿는지: URDF 로 미리 계산한 OMX 잡는 점 작업 공간(tools/omx_ws --grasp, env 헤더)
#include "vla_vocab.h"       // 이름 표 행·상위어·이름 확신도(training/data/vla_v1 에서 생성)
#include "topview.h"         // 위에서 본 지도 그림 — 하나의 정의(교사 격자 MapTok::tv, 학생 RGB)

namespace gmap {

constexpr int TOK_SLOT_VALS = 33;
constexpr int N_WALL = 56;   // walls.hpp kStateLen = 16 + 8·5
constexpr int N_ROOMTOK = 10;
constexpr int N_COMP = 4;
constexpr int N_FRONT = 8;   // 안 본 곳 광선 8(45° 간격) — VLA_INPUT 에 없는 값을 더함(가정)
constexpr int N_WAY = 4;     // 다음 경유 지점: x, y, 경로 길이, 있음
// 목표 칸(VLA_INPUT 2.1, 결정 2026-10-05 — 앱에서 물체를 누르면 id, 바닥·면을 누르면 점): 집을 것(0)·놓을 곳(1, 점으로 가기의 점도 여기) 둘.
// 칸마다 16 값, m 그대로(정규화는 obs.h), 믿는(slam) 자세의 base_link 기준 — 지도 좌표·id·종류 번호 없음. 물체 목표는 그 물체 칸의 "목표인지" 도 켜짐(일부러 겹침)
constexpr int N_GENT = 2, N_GV = 16;
enum GoalEnt { GE_PICK = 0, GE_PLACE = 1 };
enum GoalVal {
  GV_PRESENT = 0,   // 이 목표가 주어짐
  GV_KOBJ = 1,      // 꼴: 물체(id)
  GV_KPT = 2,       // 꼴: 점
  GV_KNOWN = 3,     // 위치 값이 있음(물체가 지도에 있음 — 지금 또는 마지막 본 자리, 또는 점)
  GV_LOST = 4,      // 물체 목표의 위치가 마지막으로 알던 자리(지도에서 사라짐 S_GONE / 실제: 지도에서 빠짐)
  GV_POS = 5,       // 3 x 앞, y 왼쪽, z (base_link, z = 세계 z − base_z — 칸 T_POS 와 같음)
  GV_DIST = 8,      // 수평 거리
  GV_SIN = 9,       // 방위 sin = y / 거리 (거리 < 1e-4 면 0)
  GV_COS = 10,      // 방위 cos = x / 거리 (거리 < 1e-4 면 1)
  GV_EEF = 11,      // 3 위치 − 손끝(base_link 축)
  // 14, 15 = 0
};
// 칸 숫자 33 의 자리(VLA_INPUT 3절 표 순서)
enum TokSlot {
  T_POS = 0,        // 3 위치 xyz (base_link)
  T_POS_EEF = 3,    // 3 위치 − 팔 끝 (base_link 축)
  T_DIST = 6,       // 수평 거리
  T_BEAR = 7,       // 방위 rad
  T_EXT = 8,        // 3 크기
  T_EEF_C = 11,     // 팔 끝 ↔ 중심 거리
  T_EEF_S = 12,     // 팔 끝 ↔ 상자 겉면 거리(안이면 0)
  T_REACH = 13,     // 팔이 닿는지(0/1): 물체 상자의 (r, z) 범위가 OMX 잡는 점 작업 공간과 겹치나(omx_workspace_grasp.h)
  T_DISP = 14,      // 3 처음 자리에서 옮겨진 양(base_link 축)
  T_VEL = 17,       // 3 물체 속도 m/s (지난 스텝 지도 자리와의 차, base_link 축)
  T_STATE = 20,     // 4 보임·사라짐·옮겨짐·들고 있음
  T_AGE = 24,       // 마지막 본 뒤 s
  T_SCORE = 25,     // 검출 점수
  T_NOBS = 26,      // 본 횟수
  T_SRC = 27,       // 1 = 지금 보는 중(마지막 keyframe 에서 봄 — 위치 = 그 관측 그대로 Slot::meas), 0 = 기억(지도 pos)
  T_UNC = 28,       // 위치 불확실도 m
  T_SAMEROOM = 29,  // 로봇과 같은 (드러난) 방
  T_TARGET = 30,    // 목표 물체
  T_CONF1 = 31,     // 이름 확신도: 라벨 표 1위 점수 = cos(생김새 128, 이름 128) (vla_vocab.h, 생김새 행 × 이름 종류)
  T_CONF2 = 32,     // 1위 − 2위(이름 어휘 안 다른 이름 최댓값과의 차). 이 값이 kConfLow 아래면 이름 행 = 상위어
};
struct alignas(16) MapTok {
  uint16_t slot[KSLOT][TOK_SLOT_VALS];   // FP16
  int16_t name_id[KSLOT];                // 이름 뜻 표 행(training/data/vla_v1 names) = 지도 이름(틀린 이름 그대로)의 라벨 행, 확신 낮으면 상위어 행. −1 빈 칸
  int16_t app_id[KSLOT];                 // 생김새 표 행 = 출처 참 물체의 종류(0..5), 유령 = NCLS(배경·잡동사니). −1 빈 칸
  uint16_t wall[N_WALL];                 // FP16, walls.hpp wallStateVector 와 같은 배치
  uint16_t room[N_ROOMTOK];              // FP16: 방 종류 6(kitchen·bathroom·bedroom·living room·office·모름), 가까운 문 x·y·거리 m, 문 있음
  uint16_t comp[N_COMP];                 // FP16: 과제 물체 확정, 장면 물체 확정 비율, 방 칸 본 비율, 드러난 방 비율
  int16_t n_slot;                        // 채운 칸 수
  int16_t flags;                         // 비트 0 = 이번 스텝 keyframe
  uint16_t front[N_FRONT];               // FP16: 안 본 곳 광선(로봇 앞부터 반시계 45°), 첫 안 본 칸까지 /4 m. 점유 칸에 먼저 막히거나 4 m 안에 없으면 1 (예전 pad 자리)
  uint16_t way[N_WAY];                   // FP16: 다음 경유 지점(VLA_INPUT 4절 목표 토큰) base_link x·y m, 목표까지 경로 길이 m, 있음(1/0). 목표 칸이 없거나 길이 없으면 0
  uint16_t instr1;                       // 이 판의 지시문 행 + 1(training/data/pnp_v1 instr128, 환경 I_B_INSTR — 집기·놓기 판만), 0 = 없음(상자 방·B1). 관측(obs.h)이 표에서 128 칸으로
  uint16_t bkind;                        // 이 판의 BEHAVIOR 단계(1 B1, 2 B2, 3 B3), 0 = 상자 방 — 교사 스킬 표시(POLICY 3.2)
  uint16_t pad2[2];                      // 0
  uint16_t goal[N_GENT][N_GV];           // FP16 목표 칸 2(GoalEnt × GoalVal). flags 비트 1 = 지금 가는 목표가 점(점으로 가기)
  uint8_t tv[TV_NCH][TV_B][TV_B];        // 교사 격자(topview.h): 위에서 본 지도 그림 16 × 16 덩이 [장애물 / 안 본 칸][행 = 그림 위→아래][열 = 왼→오], 표본 수 0..16
};
static_assert(sizeof(MapTok) == 1872, "map token v4 = 1872 B per env-step (v3 1360 B + top-view teacher grid 2 x 16 x 16 u8)");
static_assert(sizeof(MapTok) % 16 == 0, "16 B copies");
struct TPrev { float p[3]; int tag; };   // 칸마다 지난 스텝 지도 자리, tag = 물체 번호 << 16 | 스텝 & 0xffff (물체 속도용, 확정 칸만 씀)

// float → FP16 (가장 가까운 짝수로 반올림, NaN 은 0x7fff). CPU 는 정수 연산(f2h_soft), GPU 는 하드웨어 __float2half_rn —
// 둘이 float 2^32 개 전부에서 같은 비트임을 map_verify 가 시작 때 확인한다.
DEV uint16_t f2h_soft(float f) {
#ifdef __CUDA_ARCH__
  const uint32_t x = (uint32_t)__float_as_uint(f);
#else
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
#endif
  const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
  if (ax > 0x7f800000u) return (uint16_t)0x7fffu;   // NaN
  if (ax == 0x7f800000u) return (uint16_t)(sign | 0x7c00u);
  if (ax >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   // ≥ 65520 → inf
  if (ax < 0x38800000u) {                                     // 정규 FP16 아래
    if (ax < 0x33000000u) return (uint16_t)sign;
    const uint32_t mm = (ax & 0x7fffffu) | 0x800000u;
    const int sh = 126 - (int)(ax >> 23);
    uint32_t h = mm >> sh;
    const uint32_t rem = mm & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = (ax - 0x38000000u) >> 13;
  const uint32_t rem = ax & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return (uint16_t)(sign | h);
}
DEV uint16_t f2h(float f) {
#ifdef __CUDA_ARCH__
  return __half_as_ushort(__float2half_rn(f));
#else
  return f2h_soft(f);
#endif
}

// ---- 벽 상태 56(walls.cpp wallStateVector) ----------------------------------------------------------------------------
// 광선(rayDistances): 격자 DDA(Amanatides–Woo, walls.cpp 와 같은 꼴: 처음 경계 t 를 구하고 칸마다 tdx·tdy 를 더함, tx < ty 일 때만 x),
// 첫 점유 칸에 들어간 거리(시작 칸이 점유면 0), wall_range 넘으면 wall_range. walls.cpp 는 double, 여기는 float.
// 광선은 4 m(wall_range) 안 칸만 닿는다: 칸마다 t 가 RES 이상 늘므로 시작 칸에서 행·열로 41 칸까지. 그래서 로봇 칸 둘레 행 TOK_SROWS(84) 개 ×
// 열 96 개(시작 열 − 41 부터, 낱말 3)만 공유 메모리(GPU)·지역 배열(CPU)로 옮겨 읽는다(CPU·GPU 같은 함수). 창 밖 칸은 닿지 않으므로 결과는 전체 격자와 같다.
constexpr int TOK_SROWS = 84, TOK_SWORDS = 3, TOK_REACH = 41;
DEV int tok_cell(float v) { return (int)floorf((v - (float)GX0 * RES) / RES); }   // wall_ray 의 시작 칸과 같은 식
DEV int tok_row0(float ey) {   // 짝수 행에서 시작(32 B 조각 맞춤), 로봇 행 − 41 ~ + 41 을 덮음
  const int cy = tok_cell(ey);
  const int r0 = (cy - TOK_REACH) & ~1;
  return r0 < 0 ? 0 : r0 > GW - TOK_SROWS ? GW - TOK_SROWS : r0;
}
DEV int tok_col0(float ex) { return tok_cell(ex) - TOK_REACH; }   // 창 첫 열(음수·창 밖일 수 있음 — 그 칸은 0)
DEV uint32_t tok_word(const uint32_t w[4], int b) {   // 한 행(낱말 4)의 열 b ~ b+31 비트(창 밖 0)
  const int wi = b >> 5, sh = b & 31;
  const uint32_t lo = (wi >= 0 && wi < 4) ? w[wi] : 0u, hi = (wi + 1 >= 0 && wi + 1 < 4) ? w[wi + 1] : 0u;
#ifdef __CUDA_ARCH__
  return __funnelshift_r(lo, hi, sh);
#else
  return sh ? (lo >> sh) | (hi << (32 - sh)) : lo;
#endif
}
DEV void tok_stage(const uint32_t* occ_full, int row0, int col0, uint32_t* so, int lane, int nl) {
  for (int k = lane; k < TOK_SROWS; k += nl) {
    uint32_t w[4];
#ifdef __CUDA_ARCH__
    const uint4 v = reinterpret_cast<const uint4*>(occ_full)[row0 + k];
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
#else
    for (int q = 0; q < 4; ++q) w[q] = occ_full[(row0 + k) * 4 + q];
#endif
    for (int q = 0; q < TOK_SWORDS; ++q) so[k * TOK_SWORDS + q] = tok_word(w, col0 + 32 * q);
  }
}
DEV float wall_ray(const uint32_t* so, int row0, int col0, float x, float y, float th) {
  constexpr float OX = (float)GX0 * RES;
  float s, c;
  sincosf_d(th, &s, &c);
  const float fx = (x - OX) / RES, fy = (y - OX) / RES;
  int cx = (int)floorf(fx), cy = (int)floorf(fy);
  const int sx = c > 0.f ? 1 : -1, sy = s > 0.f ? 1 : -1;
  const float tdx = c != 0.f ? absf(RES / c) : kInf, tdy = s != 0.f ? absf(RES / s) : kInf;
  float tx = c != 0.f ? ((sx > 0 ? (float)(cx + 1) - fx : fx - (float)cx) * RES) / absf(c) : kInf;
  float ty = s != 0.f ? ((sy > 0 ? (float)(cy + 1) - fy : fy - (float)cy) * RES) / absf(s) : kInf;
  float t = 0.f;
  while (t <= MP::wall_range) {
    const unsigned lx = (unsigned)(cx - col0), ly = (unsigned)(cy - row0);
    if ((unsigned)cx < (unsigned)GW && ly < (unsigned)TOK_SROWS && lx < 32u * TOK_SWORDS &&
        ((so[ly * TOK_SWORDS + (lx >> 5)] >> (lx & 31)) & 1u))
      return maxf(t, 0.f);
    if (tx < ty) { t = tx; tx = tx + tdx; cx += sx; } else { t = ty; ty = ty + tdy; cy += sy; }
  }
  return MP::wall_range;
}
// 안 본 곳 광선(가정, VLA_INPUT 에 없음 — "어디를 아직 안 봤나"): wall_ray 와 같은 DDA 로 점유 비트 so·본 칸 비트 ss 를 같이 읽어,
// 점유 칸을 만나면 wall_range(그 방향 4 m 안에 이어진 안 본 곳 없음), 본 적 없는 칸을 만나면 그 칸에 들어간 거리. 로봇 몸통 자리(front_r0 안)의
// 안 본 칸은 건너뛴다(깊이 카메라가 몸 아래·바로 옆을 못 봄). 창 밖 칸(격자 밖)은 안 본 칸이다
constexpr float kFrontR0 = 0.2f;   // (가정) 몸통 외접원 0.194 m
DEV float front_ray(const uint32_t* so, const uint32_t* ss, int row0, int col0, float x, float y, float th) {
  constexpr float OX = (float)GX0 * RES;
  float s, c;
  sincosf_d(th, &s, &c);
  const float fx = (x - OX) / RES, fy = (y - OX) / RES;
  int cx = (int)floorf(fx), cy = (int)floorf(fy);
  const int sx = c > 0.f ? 1 : -1, sy = s > 0.f ? 1 : -1;
  const float tdx = c != 0.f ? absf(RES / c) : kInf, tdy = s != 0.f ? absf(RES / s) : kInf;
  float tx = c != 0.f ? ((sx > 0 ? (float)(cx + 1) - fx : fx - (float)cx) * RES) / absf(c) : kInf;
  float ty = s != 0.f ? ((sy > 0 ? (float)(cy + 1) - fy : fy - (float)cy) * RES) / absf(s) : kInf;
  float t = 0.f;
  while (t <= MP::wall_range) {
    const unsigned lx = (unsigned)(cx - col0), ly = (unsigned)(cy - row0);
    const bool in = (unsigned)cx < (unsigned)GW && ly < (unsigned)TOK_SROWS && lx < 32u * TOK_SWORDS;
    const int wi = in ? (int)(ly * TOK_SWORDS + (lx >> 5)) : 0;
    if (in && ((so[wi] >> (lx & 31)) & 1u)) return MP::wall_range;
    if (t >= kFrontR0 && !(in && ((ss[wi] >> (lx & 31)) & 1u))) return t;
    if (tx < ty) { t = tx; tx = tx + tdx; cx += sx; } else { t = ty; ty = ty + tdy; cy += sy; }
  }
  return MP::wall_range;
}
// 선분 하나를 로봇 기준으로(segmentsRobotFrame): a·b 끝(m) 과 로봇에서 선분까지 거리
DEV float wall_seg_robot(const int16_t* q, float px, float py, float c, float s, float r[4]) {
  float w[4];
  for (int k = 0; k < 4; ++k) w[k] = ((float)q[k] * 0.5f + (float)GX0) * RES;
  const float ax = w[0] - px, ay = w[1] - py, bx = w[2] - px, by = w[3] - py;
  r[0] = c * ax + s * ay; r[1] = -s * ax + c * ay;
  r[2] = c * bx + s * by; r[3] = -s * bx + c * by;
  const float abx = r[2] - r[0], aby = r[3] - r[1];
  const float t = clampf(-(r[0] * abx + r[1] * aby) / maxf(abx * abx + aby * aby, 1e-12f), 0.f, 1.f);
  const float dx = r[0] + abx * t, dy = r[1] + aby * t;
  return sqrtf(dx * dx + dy * dy);
}
DEV const int16_t* seg_at(const int16_t* segs, const MapCore& m, int k) {   // 가로 nseg_h 개 다음 세로
  return k < m.nseg_h ? segs + 4 * k : segs + 4 * (MAXSEG + k - m.nseg_h);
}

constexpr int WG = GW / 2;   // 경유 지점 거친 격자 한 변(0.2 m 칸 64 개)
static_assert(WG == 64 && MP::way_res == 2.f * RES, "waypoint grid = 2x2 fine cells, one uint64 per row");
struct WayScratch {   // 경유 지점 BFS(거친 격자 행마다 64 비트)
  uint64_t blk[WG];     // 막힌 칸(점유 2×2 → 1 칸 부풀림)
  uint64_t fr[WG];      // 이번 단계 새로 닿은 칸
  uint64_t l0[WG], l1[WG];   // BFS 단계 mod 3 (두 비트, 3 = 안 닿음)
};
struct TokScratch {   // 판 하나의 작업 공간(GPU 공유 메모리). 광선 거리·칸의 지난 자리는 그 일을 맡은 레인의 레지스터(make_tokens)
  float sd[2 * MAXSEG];   // 선분 거리
  float sk[KSLOT];        // 칸 순서 열쇠(수평 거리), 안 넣는 칸 −1
  float tk[KSLOT];        // 목표 후보 열쇠, 아님 −1
  float wy[N_WAY];        // 경유 지점 결과(레인 0 이 씀, 끝에 out 으로)
  float tk2[KSLOT];       // 둘째 목표(BEHAVIOR 놓을 곳) 후보 열쇠, 아님 −1
  float tg[KSLOT], tg2[KSLOT];   // 같은 짝인데 사라진(S_GONE) 칸 — 목표 칸의 "마지막으로 알던 자리"(GV_LOST), 아님 −1
  TvIn tv;                // 위에서 본 지도 입력(레인 0 이 만들고 모든 레인이 덩이를 나눠 셈)
  union {
    WayScratch w;                            // 1) 경유 지점 BFS(창 옮기기 전)
    struct {
      uint32_t so[TOK_SROWS * TOK_SWORDS];   // 로봇 둘레 창의 점유 비트(광선이 읽음). 광선 뒤에는 out 이 덮어씀
      uint32_t ss[TOK_SROWS * TOK_SWORDS];   // 같은 창의 본 칸 비트(안 본 곳 광선)
    };
    MapTok out;
  };
};

// ---- 팔이 닿는지(VLA_INPUT 3절): 물체를 joint1 축 기준 (r, z) 범위로 바꿔 작업 공간 표와 겹치는지 ------------------------------------
// 표 = **잡는 점** 작업 공간(omx_workspace_grasp.h, E0 −0.0119 m — env_beh.h grasp_reach_box·B3 성공 판정과 같은 표, 2026-10-04 바꿈.
// 예전 omx_end_effector_link 표 omx_workspace.h 는 B0 비트 동일 때문에 남겨 두었었음)
// r 범위 = joint1 축(믿는 자세의 base_link 에서 (AX, AY))에서 물체 중심까지 수평 거리 ∓ 바닥 자국 외접원 반지름(map 축 정렬 상자라도 회전에 불변이게 —
// 강체 불변 시험 sm_tok_test), z 범위 = 상자 높이(base_link). 표는 joint1 이 360° 를 돌 수 있어 방위와 무관하다. 범위 안 칸 중 하나라도 작업 공간이면 1
DEV bool omx_reach_box(const float ctr[3], const float ext[3], float px, float py, float c, float s) {
  const float axw = px + (c * omxwsg::AX - s * omxwsg::AY), ayw = py + (s * omxwsg::AX + c * omxwsg::AY);
  const float dx = ctr[0] - axw, dy = ctr[1] - ayw, d = sqrtf(dx * dx + dy * dy);
  const float rho = 0.5f * sqrtf(ext[0] * ext[0] + ext[1] * ext[1]);
  const float rmin = maxf(0.f, d - rho), rmax = d + rho;
  const float zlo = ctr[2] - 0.5f * ext[2] - MP::base_z, zhi = ctr[2] + 0.5f * ext[2] - MP::base_z;
  int r0 = (int)floorf(rmin / omxwsg::RES), r1 = (int)floorf(rmax / omxwsg::RES);
  int z0 = (int)floorf((zlo - omxwsg::Z_LO) / omxwsg::RES), z1 = (int)floorf((zhi - omxwsg::Z_LO) / omxwsg::RES);
  if (r0 >= omxwsg::NR || z1 < 0 || z0 >= omxwsg::NZ) return false;
  r1 = r1 >= omxwsg::NR ? omxwsg::NR - 1 : r1;
  z0 = z0 < 0 ? 0 : z0;
  z1 = z1 >= omxwsg::NZ ? omxwsg::NZ - 1 : z1;
  const uint64_t hiw = r1 >= 63 ? ~0ull : ((1ull << (r1 + 1)) - 1ull), mask = hiw & ~((1ull << r0) - 1ull);
  for (int z = z0; z <= z1; ++z)
    if (omxwsg::row(z) & mask) return true;
  return false;
}

// ---- 다음 경유 지점(VLA_INPUT 4절, sm_snap_place_path 대신 — 계획서 4.4 "거친 격자 파면 BFS") ---------------------------------------
// 믿는 점유 비트(occ, 0.1 m) → 0.2 m 거친 격자(2×2 중 하나라도 점유) → 1 칸 부풀림(8 이웃) = 막힘. 안 본 칸은 지나갈 수 있다(가정).
// 목표 칸(지도의 목표 칸 자리) 둘레 1 칸을 씨앗(단계 0)으로 8 이웃 BFS — 단계는 mod 3 두 비트로만 둔다(이웃 칸의 단계 차는 1 이하라 내리막에 충분).
// 로봇 칸에 닿으면 멈춘다. 로봇 칸 둘레 1 칸과 씨앗은 막힘에서 뺀다(벽 옆에서 시작·목표 물체 자체가 점유). 레인이 행을 나누고 단계마다 동기 2 번.
// 레인 0 이 로봇 칸에서 내리막(단계 −1 이웃 중 목표 칸에 가장 가까운 것, 같으면 고정 차례)으로 끝까지 가며 경로 길이를 더하고, way_look 칸 간 자리를
// 경유 지점으로. 경로가 way_look 보다 짧으면 목표 자리. 결과 ts.wy = {x, y (base_link), 경로 길이 m, 있음}
DEV uint32_t compact16(uint32_t w) {   // 32 칸 → 짝지은 OR 16 칸
  uint32_t t = (w | (w >> 1)) & 0x55555555u;
  t = (t | (t >> 1)) & 0x33333333u;
  t = (t | (t >> 2)) & 0x0f0f0f0fu;
  t = (t | (t >> 4)) & 0x00ff00ffu;
  t = (t | (t >> 8)) & 0x0000ffffu;
  return t;
}
DEV uint64_t dil_row(uint64_t x) { return x | (x << 1) | (x >> 1); }
template <int NLC, class Sync>
DEV void waypoint(const uint32_t* occ, float gx, float gy, float px, float py, float c, float s, WayScratch& w, float* wy, int lane, int nl,
                  const Sync& sync, int tbug = 0) {
  constexpr float OX = (float)GX0 * RES;
  const int gfx = (int)floorf((gx - OX) / RES), gfy = (int)floorf((gy - OX) / RES);
  const int rfx = (int)floorf((px - OX) / RES), rfy = (int)floorf((py - OX) / RES);
  const bool inside = gfx >= 0 && gfx < GW && gfy >= 0 && gfy < GW && rfx >= 0 && rfx < GW && rfy >= 0 && rfy < GW;
  const int gcx = gfx >> 1, gcy = gfy >> 1, rcx = rfx >> 1, rcy = rfy >> 1;   // inside 일 때만 씀(같은 값을 모든 레인이 앎)
  if (!inside) {
    if (lane == 0) for (int q = 0; q < N_WAY; ++q) wy[q] = 0.f;
    sync();
    return;
  }
  // 거친 점유
  for (int r = lane; r < WG; r += nl) {
    uint64_t row = 0;
    for (int q = 0; q < 4; ++q) {
      const uint32_t a = occ[(2 * r) * 4 + q] | occ[(2 * r + 1) * 4 + q];
      row |= (uint64_t)compact16(a) << (16 * q);
    }
    w.fr[r] = row;
  }
  sync();
  auto near1 = [](int r, int cx, int cy) -> uint64_t { return (r < cy - 1 || r > cy + 1) ? 0ull : dil_row(1ull << cx); };   // 행 r 에서 (cx, cy) 둘레 1 칸
  for (int r = lane; r < WG; r += nl) {
    uint64_t d = dil_row(w.fr[r]);
    if (r > 0) d |= dil_row(w.fr[r - 1]);
    if (r < WG - 1) d |= dil_row(w.fr[r + 1]);
    d &= ~near1(r, rcx, rcy);
    d &= ~near1(r, gcx, gcy);
    w.blk[r] = d;
    const uint64_t seed = near1(r, gcx, gcy);
    w.l0[r] = ~seed;   // 씨앗 = 단계 0(두 비트 0), 나머지 = 3(안 닿음)
    w.l1[r] = ~seed;
  }
  sync();
  for (int r = lane; r < WG; r += nl) w.fr[r] = near1(r, gcx, gcy);
  sync();
  int L = -1;
  if (rcy >= gcy - 1 && rcy <= gcy + 1 && rcx >= gcx - 1 && rcx <= gcx + 1) L = 0;
  constexpr int PER = (WG + NLC - 1) / NLC;   // 레인마다 맡는 행 수의 최댓값(nl ≥ NLC)
  for (int lev = 1; L < 0 && lev <= MP::way_iter; ++lev) {
    uint64_t nw[PER];
    bool any = false, hit = false;
    int k = 0;
    for (int r = lane; r < WG; r += nl, ++k) {
      uint64_t x = w.fr[r];
      if (r > 0) x |= w.fr[r - 1];
      if (r < WG - 1) x |= w.fr[r + 1];
      x = dil_row(x) & ~w.blk[r] & (w.l0[r] & w.l1[r]);
      nw[k] = x;
      any = any || x != 0ull;
      hit = hit || (r == rcy && ((x >> rcx) & 1ull));
    }
    sync();
    const int cd = lev % 3;
    k = 0;
    for (int r = lane; r < WG; r += nl, ++k) {
      const uint64_t x = nw[k];
      w.fr[r] = x;
      w.l0[r] = (w.l0[r] & ~x) | ((cd & 1) ? x : 0ull);
      w.l1[r] = (w.l1[r] & ~x) | ((cd & 2) ? x : 0ull);
    }
    any = sync.any(any);
    hit = sync.any(hit);
    if (hit) L = lev;
    else if (!any) break;
  }
  if (lane == 0) {
    if (L < 0) {
      for (int q = 0; q < N_WAY; ++q) wy[q] = 0.f;
    } else {
      auto code = [&](int x, int y) -> int { return (int)((w.l0[y] >> x) & 1ull) | ((int)((w.l1[y] >> x) & 1ull) << 1); };
      int cx = rcx, cy = rcy, cd = L % 3;
      float len = 0.f;
      int wx = -1, wyc = -1;
      bool ok = true;
      const int dxs[8] = {1, 0, -1, 0, 1, -1, -1, 1}, dys[8] = {0, 1, 0, -1, 1, 1, -1, -1};
      for (int st = 1; st <= L; ++st) {
        const int want = (cd + 2) % 3;
        int bx = -1, by = -1, bd = 0, bk = -1;
        for (int k2 = 0; k2 < 8; ++k2) {
          const int nx = cx + dxs[k2], ny = cy + dys[k2];
          if (nx < 0 || nx >= WG || ny < 0 || ny >= WG || code(nx, ny) != want) continue;
          const int ddx = nx - gcx, ddy = ny - gcy, dd = ddx * ddx + ddy * ddy;
          if (bk < 0 || dd < bd || (tbug == 2 && dd == bd)) { bx = nx; by = ny; bd = dd; bk = k2; }   // tbug 2(음성 대조): 같으면 뒤 차례
        }
        if (bk < 0) { ok = false; break; }
        len = len + (bk >= 4 ? 1.41421356f : 1.f) * MP::way_res;
        cx = bx; cy = by; cd = want;
        if (st == MP::way_look) { wx = cx; wyc = cy; }
      }
      if (!ok) {
        for (int q = 0; q < N_WAY; ++q) wy[q] = 0.f;
      } else {
        const float ccx = OX + ((float)cx + 0.5f) * MP::way_res, ccy = OX + ((float)cy + 0.5f) * MP::way_res;
        len = len + sqrtf((gx - ccx) * (gx - ccx) + (gy - ccy) * (gy - ccy));
        if (L == 0) len = sqrtf((gx - px) * (gx - px) + (gy - py) * (gy - py));
        const float tx = wx >= 0 ? OX + ((float)wx + 0.5f) * MP::way_res : gx, ty = wyc >= 0 ? OX + ((float)wyc + 0.5f) * MP::way_res : gy;
        const float dx = tx - px, dy = ty - py;
        wy[0] = c * dx + s * dy;
        wy[1] = -s * dx + c * dy;
        wy[2] = len;
        wy[3] = 1.f;
      }
    }
  }
  sync();
}

// 목표 물체 칸 고르기(make_tokens 1)·1b) 와 같은 규칙, 한 스레드판 — 위에서 본 지도 그림이 씀): g0 = prim 0 짝(사라지지 않은 것 중 가장 가까움, 없으면 사라진 것),
// g1 = prim 1 짝(BEHAVIOR goal 비트 1). lost0/lost1 = 사라진 칸을 고름. 없으면 −1
DEV void tv_goal_slots(const MapCore& m, const BCtx* bxp, int& g0, int& g1) {
  const bool beh = bxp != nullptr && bxp->on;
  const Prim& P = m.prim[0];
  const float tcx = 0.5f * (P.lo[0] + P.hi[0]), tcy = 0.5f * (P.lo[1] + P.hi[1]);
  float pext[3];
  for (int a = 0; a < 3; ++a) pext[a] = P.hi[a] - P.lo[a];
  const float thr = maxf(MP::da_min, MP::da_k * max3(pext));
  int t0 = -1, l0 = -1, t1 = -1, l1 = -1;
  float k0 = 0.f, kl0 = 0.f, k1 = 0.f, kl1 = 0.f;
  for (int b = 0; b < KSLOT; ++b) {
    if (!((m.conf_mask >> b) & 1)) continue;
    const Slot& S = m.slot[b];
    const float tx = S.pos[0] - tcx, ty = S.pos[1] - tcy, d2 = tx * tx + ty * ty;
    if ((!beh || (bxp->bm->goal & 1)) && S.cls == (beh ? P.cls : (int)C_CUP) && d2 < thr * thr) {
      if (S.state != S_GONE) { if (t0 < 0 || d2 < k0) { t0 = b; k0 = d2; } }
      else if (l0 < 0 || d2 < kl0) { l0 = b; kl0 = d2; }
    }
  }
  if (beh && (bxp->bm->goal & 2)) {
    const Prim& P1 = m.prim[1];
    float e1[3];
    for (int a = 0; a < 3; ++a) e1[a] = P1.hi[a] - P1.lo[a];
    const float th1 = maxf(MP::da_min, MP::da_k * max3(e1));
    for (int b = 0; b < KSLOT; ++b) {
      if (!((m.conf_mask >> b) & 1)) continue;
      const Slot& S = m.slot[b];
      const float ux = S.pos[0] - 0.5f * (P1.lo[0] + P1.hi[0]), uy = S.pos[1] - 0.5f * (P1.lo[1] + P1.hi[1]), u2 = ux * ux + uy * uy;
      if (!(S.cls == P1.cls && u2 < th1 * th1)) continue;
      if (S.state != S_GONE) { if (b != t0 && (t1 < 0 || u2 < k1)) { t1 = b; k1 = u2; } }
      else if (b != l0 && (l1 < 0 || u2 < kl1)) { l1 = b; kl1 = u2; }
    }
  }
  g0 = t0 >= 0 ? t0 : l0;
  g1 = t1 >= 0 ? t1 : l1;
}
// 위에서 본 지도 입력(topview.h TvIn): 믿는 자세, 확정 물체 상자(사라진 것 빼고 — 사라진 목표는 마지막 자리로 넣음), 목표 색, 놓을 점.
// teacher = true: 교사 격자용(목표 색·점 없음 — 목표는 목표 칸 값으로). hide_obj = 학생 목표 감추기(물체 목표 색만 장애물로, 점은 그대로)
DEV void tv_input(const MapCore& m, const BCtx* bxp, bool teacher, bool hide_obj, TvIn& in) {
  const bool beh = bxp != nullptr && bxp->on;
  float s, c;
  sincosf_d(m.eyaw, &s, &c);
  in.px = m.ex; in.py = m.ey; in.c = c; in.s = s;
  int g0 = -1, g1 = -1;
  if (!teacher) tv_goal_slots(m, bxp, g0, g1);
  const int k0 = (beh && bxp->bm->kind == bsc::EK_B1) ? (int)TV_PLACE : (int)TV_PICK;   // prim 0 = B1 이면 가는 곳(놓을 곳 칸)
  int n = 0;
  for (int b = 0; b < KSLOT && n < TV_MAXBOX; ++b) {
    if (!((m.conf_mask >> b) & 1)) continue;
    const Slot& S = m.slot[b];
    const bool goal = b == g0 || b == g1;
    if (S.state == S_GONE && !goal) continue;
    TvBox& B = in.box[n++];
    for (int a = 0; a < 2; ++a) { B.lo[a] = S.pos[a] - 0.5f * S.ext[a]; B.hi[a] = S.pos[a] + 0.5f * S.ext[a]; }
    B.kind = (!goal || hide_obj) ? (int)TV_OBST : (b == g0 ? k0 : (int)TV_PLACE);
  }
  in.nbox = n;
  for (int k = n; k < TV_MAXBOX; ++k) { in.box[k].lo[0] = in.box[k].lo[1] = in.box[k].hi[0] = in.box[k].hi[1] = 0.f; in.box[k].kind = 0; }   // 빈 자리 0(바이트 비교)
  in.has_pt = 0; in.gx = 0.f; in.gy = 0.f;
  if (!teacher && beh && (bxp->bm->gmode & bsc::GM_PLACE_PT)) { in.has_pt = 1; in.gx = bxp->bm->gp[0]; in.gy = bxp->bm->gp[1]; }
}

// 학생 그림 상태 한 판(레인 nl 개가 비트를 나눠 옮김, 레인 0 이 입력)
DEV void tv_topstate(const MapCore& m, const BCtx* bxp, const uint32_t* occ, const uint32_t* seen, TopState& t, int lane, int nl) {
  for (int k = lane; k < NWORD; k += nl) { t.occ[k] = occ[k]; t.seen[k] = seen[k]; }
  if (lane == 0) tv_input(m, bxp, false, false, t.in);
}

// 목표 칸 하나(GoalVal 16 값 FP16): kind 0 = 없음, 1 = 물체, 2 = 점. 위치 P(지도 좌표)는 known 일 때만. (px, py, c, s) = 믿는 자세, eef_b = base_link 손끝
DEV void goal_fill(uint16_t* g, int kind, bool known, bool lost, const float P[3], float px, float py, float c, float s, const float eef_b[3]) {
  float v[N_GV];
  for (int q = 0; q < N_GV; ++q) v[q] = 0.f;
  if (kind) {
    v[GV_PRESENT] = 1.f;
    v[kind == 1 ? GV_KOBJ : GV_KPT] = 1.f;
    if (known) {
      v[GV_KNOWN] = 1.f;
      v[GV_LOST] = lost ? 1.f : 0.f;
      const float dx = P[0] - px, dy = P[1] - py;
      const float x = c * dx + s * dy, y = -s * dx + c * dy, z = P[2] - MP::base_z;
      const float d = sqrtf(x * x + y * y);
      v[GV_POS] = x; v[GV_POS + 1] = y; v[GV_POS + 2] = z;
      v[GV_DIST] = d;
      v[GV_SIN] = d >= 1e-4f ? y / d : 0.f;
      v[GV_COS] = d >= 1e-4f ? x / d : 1.f;
      v[GV_EEF] = x - eef_b[0]; v[GV_EEF + 1] = y - eef_b[1]; v[GV_EEF + 2] = z - eef_b[2];
    }
  }
  for (int q = 0; q < N_GV; ++q) g[q] = f2h(v[q]);
}

// 판 하나의 토큰. 레인 lane / nl 개가 나눠 하고 sync 로 맞춘다(CPU: lane 0, nl 1). write = false 면 tprev 를 쓰지 않음(GPU 남는 레인)
// occ: 판의 점유 비트 전체. 로봇 둘레 행을 ts.so 로 옮긴 뒤 광선을 쏘고, out 은 그다음 동기부터 쓴다(so 와 같은 자리)
// NLC: nl 의 최솟값(컴파일 때). 레인마다 맡는 칸·광선 수가 (16 + NLC − 1) / NLC 이하라 지난 자리·광선 거리를 레지스터 배열에 둔다(GPU NLC = 16 → 하나씩)
// tbug(음성 대조, 검증용): 2 = 경유 지점 내리막의 같은 거리 이웃을 뒤 차례로, 3 = 지금 보는 중 칸도 지도 자리(관측 자리 무시), 4 = BEHAVIOR 방 자리 1.5 m 밀림,
//   11 = 목표 칸을 회전 없이(지도 차), 12 = 위에서 본 지도(교사 격자)를 거꾸로 돌림
template <int NLC, class Sync>
// bxp: BEHAVIOR 판 맥락(map.h BCtx — 방·문은 장면 방 격자, 이름·확신도는 장면 묶음의 이름 표 표, 목표 칸 = prim 0 의 이름). nullptr = 상자 방
DEV void make_tokens_n(const MapCore& m, const uint32_t* occ, const uint32_t* seen, const int16_t* segs, TPrev* tprev, TokScratch& ts, int lane, int nl,
                       bool write, const Sync& sync, int tbug = 0, const BCtx* bxp = nullptr) {
  const bool beh = bxp != nullptr && bxp->on;
  constexpr int PER = (KSLOT + NLC - 1) / NLC;
  static_assert(KSLOT == 16, "rays and slots share the per-lane count");
  TPrev tpl[PER];
  float ray[PER], fr[PER];
  MapTok& o = ts.out;
  float s, c;
  sincosf_d(m.eyaw, &s, &c);
  const float px = m.ex, py = m.ey;
  const int nseg = m.nseg_h + m.nseg_v;
  const int rk = beh ? broom_local(*bxp, px + (tbug == 4 ? 1.5f : 0.f), py) : room_of(m, px, py);   // tbug 4: 음성 대조(BEHAVIOR 방 토큰 자리 틀림)
  const int rrk = (rk >= 0 && ((m.rrev >> rk) & 1)) ? rk : -1;   // 로봇이 있는 드러난 방
  // 1) 서로 무관한 전역 읽기를 먼저 다 낸다(지연을 겹침): 칸 열쇠·지난 자리, 선분 거리, 로봇 둘레 점유 행
  {
    const Prim& P = m.prim[0];   // 참 컵(과제 물체): 목표 칸 = 같은 이름의 확정 칸 중 짝 문턱 안 가장 가까운 것(가정: 교사 쪽 정답으로 고름)
    const float tcx = 0.5f * (P.lo[0] + P.hi[0]), tcy = 0.5f * (P.lo[1] + P.hi[1]);
    float pext[3];
    for (int a = 0; a < 3; ++a) pext[a] = P.hi[a] - P.lo[a];
    const float thr = maxf(MP::da_min, MP::da_k * max3(pext));
    const int cm = m.conf_mask;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
      const int b = lane + q * nl;
      if (b >= KSLOT) break;
      if (!((cm >> b) & 1)) { ts.sk[b] = -1.f; ts.tk[b] = -1.f; ts.tk2[b] = -1.f; ts.tg[b] = -1.f; ts.tg2[b] = -1.f; continue; }
      const Slot& S = m.slot[b];
#ifdef __CUDA_ARCH__
      {  // 칸 기록(100 B)을 L1 로 미리 — 3) 이 광선 뒤에 다시 읽음
        const char* sp = reinterpret_cast<const char*>(&S);
        for (int q = 0; q < (int)sizeof(Slot); q += 32) asm volatile("prefetch.global.L1 [%0];" ::"l"(sp + q));
      }
#endif
      tpl[q] = tprev[b];
      const float dx = S.pos[0] - px, dy = S.pos[1] - py;
      ts.sk[b] = sqrtf(dx * dx + dy * dy);
      const float tx = S.pos[0] - tcx, ty = S.pos[1] - tcy, d2 = tx * tx + ty * ty;
      // 목표 물체(prim 0) 짝: BEHAVIOR 는 goal 비트 0 일 때만(점으로 가기면 물체 목표 없음). 사라진 칸은 따로(목표 칸의 마지막 자리)
      const bool m0 = (!beh || (bxp->bm->goal & 1)) && S.cls == (beh ? P.cls : (int)C_CUP) && d2 < thr * thr;
      ts.tk[b] = (m0 && S.state != S_GONE) ? d2 : -1.f;
      ts.tg[b] = (m0 && S.state == S_GONE) ? d2 : -1.f;
      ts.tk2[b] = -1.f;
      ts.tg2[b] = -1.f;
      if (beh && (bxp->bm->goal & 2)) {   // 놓을 곳(prim 1): 같은 이름의 확정 칸 중 짝 문턱 안
        const Prim& P1 = m.prim[1];
        float e1[3];
        for (int a = 0; a < 3; ++a) e1[a] = P1.hi[a] - P1.lo[a];
        const float th1 = maxf(MP::da_min, MP::da_k * max3(e1));
        const float ux = S.pos[0] - 0.5f * (P1.lo[0] + P1.hi[0]), uy = S.pos[1] - 0.5f * (P1.lo[1] + P1.hi[1]), u2 = ux * ux + uy * uy;
        const bool m1 = S.cls == P1.cls && u2 < th1 * th1;
        ts.tk2[b] = (m1 && S.state != S_GONE) ? u2 : -1.f;
        ts.tg2[b] = (m1 && S.state == S_GONE) ? u2 : -1.f;
      }
    }
  }
  for (int k = lane; k < nseg; k += nl) {
    float r[4];
    ts.sd[k] = wall_seg_robot(seg_at(segs, m, k), px, py, c, s, r);
  }
  sync();
  // 1b) 목표 칸(모든 레인이 같은 값) → 다음 경유 지점(목표 칸이 있을 때만 BFS). 작업 공간은 창 옮기기 전의 union 자리
  int tgt = -1, tgt2 = -1, nslot = 0, gone = -1, gone2 = -1;
  for (int b = 0; b < KSLOT; ++b) {
    nslot += ts.sk[b] >= 0.f;
    if (ts.tk[b] >= 0.f && (tgt < 0 || ts.tk[b] < ts.tk[tgt])) tgt = b;
    if (ts.tg[b] >= 0.f && (gone < 0 || ts.tg[b] < ts.tg[gone])) gone = b;
  }
  if (beh)
    for (int b = 0; b < KSLOT; ++b) {
      if (ts.tk2[b] >= 0.f && b != tgt && (tgt2 < 0 || ts.tk2[b] < ts.tk2[tgt2])) tgt2 = b;
      if (ts.tg2[b] >= 0.f && b != gone && (gone2 < 0 || ts.tg2[b] < ts.tg2[gone2])) gone2 = b;
    }
  const bool goto_pt = beh && (bxp->bm->gmode & bsc::GM_GOTO);   // 지금 가는 목표 = 놓을 점(물체 목표 없음)
  if (tgt >= 0) {
    waypoint<NLC>(occ, m.slot[tgt].pos[0], m.slot[tgt].pos[1], px, py, c, s, ts.w, ts.wy, lane, nl, sync, tbug);
  } else if (goto_pt) {   // 점으로 가기: 경유 지점도 점으로(점은 늘 앎)
    waypoint<NLC>(occ, bxp->bm->gp[0], bxp->bm->gp[1], px, py, c, s, ts.w, ts.wy, lane, nl, sync, tbug);
  } else if (lane == 0) {
    for (int q = 0; q < N_WAY; ++q) ts.wy[q] = 0.f;
  }
  const int row0 = tok_row0(py), col0 = tok_col0(px);
  tok_stage(occ, row0, col0, ts.so, lane, nl);
  tok_stage(seen, row0, col0, ts.ss, lane, nl);
  sync();
  PROF_MARK(TK_STAGE);
  // 2) 광선 16(레인마다 레지스터에)
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    const int i = lane + q * nl;
    if (i < 16) ray[q] = wall_ray(ts.so, row0, col0, px, py, m.eyaw + kTwoPi * (float)i * (1.0f / 16.f));
    if (i < N_FRONT) fr[q] = front_ray(ts.so, ts.ss, row0, col0, px, py, m.eyaw + kTwoPi * (float)i * (1.0f / (float)N_FRONT));
  }
  sync();
  PROF_MARK(TK_A);
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    const int i = lane + q * nl;
    if (i < 16) o.wall[i] = f2h(ray[q] / MP::wall_range);
    if (i < N_FRONT) o.front[i] = f2h(fr[q] / MP::wall_range);
  }
  PROF_MARK(TK_SEG);
  // 3) 가까운 선분 8(거리 순, 같으면 앞 번호)
  for (int k = lane; k < nseg; k += nl) {
    int rank = 0;
    for (int j = 0; j < nseg; ++j) rank += ts.sd[j] < ts.sd[k] || (ts.sd[j] == ts.sd[k] && j < k);
    if (rank >= 8) continue;
    float r[4];
    const float d = wall_seg_robot(seg_at(segs, m, k), px, py, c, s, r);
    const bool valid = d <= MP::wall_valid;
    uint16_t* w = o.wall + 16 + 5 * rank;
    for (int q = 0; q < 4; ++q) w[q] = f2h(valid ? clampf(r[q] / MP::wall_range, -1.f, 1.f) : 0.f);
    w[4] = f2h(valid ? 1.f : 0.f);
  }
  for (int j = lane; j < 8; j += nl)
    if (j >= nseg) for (int q = 0; q < 5; ++q) o.wall[16 + 5 * j + q] = 0;
  // 4) 물체 칸
  const int t_kf = m.t - m.since;   // 마지막 keyframe 시각
  const int tag_now = m.t & 0xffff, tag_prev = (m.t - 1) & 0xffff;
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    const int b = lane + q * nl;
    if (b >= KSLOT) break;
    if (ts.sk[b] < 0.f) continue;
    const Slot& S = m.slot[b];
    const TPrev tp = tpl[q];
    {
      const float kb = b == tgt ? -1.f : ts.sk[b];
      int rank = 0;
      for (int j = 0; j < KSLOT; ++j) {
        if (ts.sk[j] < 0.f || j == b) continue;
        const float kj = j == tgt ? -1.f : ts.sk[j];
        rank += kj < kb || (kj == kb && j < b);
      }
      uint16_t* v = o.slot[rank];   // 바로 FP16 으로(지역 배열 없이)
      // 지금 보는 중이면 그 keyframe 의 관측 자리(Slot::meas — 이번 프레임 깊이 → 카메라 → 몸에 해당), 아니면 지도 자리 − slam 자세(VLA_INPUT 3절)
      const bool live = !S.held && S.last_seen == t_kf;
      const float* P = (live && tbug != 3) ? S.meas : S.pos;
      const float dx = P[0] - px, dy = P[1] - py;
      const float pr0 = c * dx + s * dy, pr1 = -s * dx + c * dy, pr2 = P[2] - MP::base_z;
      const float pe0 = pr0 - m.eef_b[0], pe1 = pr1 - m.eef_b[1], pe2 = pr2 - m.eef_b[2];
      const float dist = sqrtf(pr0 * pr0 + pr1 * pr1);
      v[T_POS] = f2h(pr0); v[T_POS + 1] = f2h(pr1); v[T_POS + 2] = f2h(pr2);
      v[T_POS_EEF] = f2h(pe0); v[T_POS_EEF + 1] = f2h(pe1); v[T_POS_EEF + 2] = f2h(pe2);
      v[T_DIST] = f2h(dist);
      v[T_BEAR] = f2h(atan2f_d(pr1, pr0));
      for (int a = 0; a < 3; ++a) v[T_EXT + a] = f2h(S.ext[a]);
      v[T_EEF_C] = f2h(sqrtf(pe0 * pe0 + pe1 * pe1 + pe2 * pe2));
      float g2 = 0.f, blo[3], bhi[3];
      for (int a = 0; a < 3; ++a) {
        blo[a] = P[a] - 0.5f * S.ext[a];
        bhi[a] = P[a] + 0.5f * S.ext[a];
        const float gk = maxf(0.f, maxf(blo[a] - m.eef_m[a], m.eef_m[a] - bhi[a]));
        g2 = g2 + gk * gk;
      }
      v[T_EEF_S] = f2h(sqrtf(g2));
      v[T_REACH] = omx_reach_box(P, S.ext, px, py, c, s) ? (uint16_t)0x3c00u : (uint16_t)0u;
      const float fdx = S.pos[0] - S.first_pos[0], fdy = S.pos[1] - S.first_pos[1];
      v[T_DISP] = f2h(c * fdx + s * fdy);
      v[T_DISP + 1] = f2h(-s * fdx + c * fdy);
      v[T_DISP + 2] = f2h(S.pos[2] - S.first_pos[2]);
      if (m.t > 0 && tp.tag == ((S.id << 16) | tag_prev)) {
        const float vx = (S.pos[0] - tp.p[0]) / MP::tok_dt, vy = (S.pos[1] - tp.p[1]) / MP::tok_dt;
        v[T_VEL] = f2h(c * vx + s * vy);
        v[T_VEL + 1] = f2h(-s * vx + c * vy);
        v[T_VEL + 2] = f2h((S.pos[2] - tp.p[2]) / MP::tok_dt);
      } else {
        v[T_VEL] = 0; v[T_VEL + 1] = 0; v[T_VEL + 2] = 0;
      }
      for (int q = 0; q < 4; ++q) v[T_STATE + q] = S.state == q ? (uint16_t)0x3c00u : (uint16_t)0u;   // FP16 1.0 / 0
      v[T_AGE] = f2h((float)(m.t - S.last_seen) * MP::tok_dt);
      v[T_SCORE] = f2h(S.score);
      v[T_NOBS] = f2h((float)S.n_obs);
      v[T_SRC] = live ? (uint16_t)0x3c00u : (uint16_t)0u;
      {  // 마지막 본 뒤 믿는 오도메트리 이동·회전에 오도메트리 잡음 규칙을 곱함(가정: σ = odo_t·Δs + (odo_rr·Δθ + odo_rt·Δs)·거리)
        const float dl = m.plen - S.seen_len, dr = m.prot - S.seen_rot;
        v[T_UNC] = f2h(S.held ? 0.f : MP::odo_t * dl + (MP::odo_rr * dr + MP::odo_rt * dl) * dist);
      }
      const int ok = beh ? broom_local(*bxp, S.pos[0], S.pos[1]) : room_of(m, S.pos[0], S.pos[1]);
      v[T_SAMEROOM] = (rrk >= 0 && ok == rrk) ? (uint16_t)0x3c00u : (uint16_t)0u;
      v[T_TARGET] = (b == tgt || b == tgt2) ? (uint16_t)0x3c00u : (uint16_t)0u;   // BEHAVIOR 집기·놓기: 집을 물체 + 놓을 곳
      // 이름: 라벨 표 1위(지도 이름 = 틀린 이름 그대로)의 확신도 = 생김새(출처 참 물체 종류, 유령 = NCLS)와 이름 벡터의 코사인·1위 − 2위(vla_vocab.h).
      // 1위 − 2위가 낮으면 상위어 행(VLA_INPUT 3절 "확신 낮으면 상위어")
      if (beh) {   // BEHAVIOR: 종류 = 이름 표 행. 생김새 = 우리 렌더의 상자(행 1, 가정 — 종류별 생김새 행은 아직 없음), 유령 = NCLS
        const bsc::SceneSet& ss = *bxp->ss;
        const int app = S.src >= 0 ? (int)C_ITEM : NCLS;
        const float cf1 = ss.conf1[app * ss.nname + S.cls], cf2 = ss.conf2[app * ss.nname + S.cls];
        v[T_CONF1] = f2h(cf1);
        v[T_CONF2] = f2h(cf2);
        const int hy = ss.hyper[S.cls];
        o.name_id[rank] = (int16_t)((cf2 < vlav::kConfLow && hy >= 0) ? hy : S.cls);
        o.app_id[rank] = (int16_t)app;
      } else {
      const int app = S.src >= 0 ? m.prim[S.src].cls : NCLS;
      const float cf1 = vlav::conf1(app, S.cls), cf2 = vlav::conf2(app, S.cls);
      v[T_CONF1] = f2h(cf1);
      v[T_CONF2] = f2h(cf2);
      o.name_id[rank] = cf2 < vlav::kConfLow ? vlav::sim_hyper(S.cls) : vlav::sim_name(S.cls);
      o.app_id[rank] = (int16_t)app;
      }
    }
    if (write) tprev[b] = TPrev{{S.pos[0], S.pos[1], S.pos[2]}, (S.id << 16) | tag_now};
  }
  for (int j = lane; j < KSLOT; j += nl)
    if (j >= nslot) {
      for (int q = 0; q < TOK_SLOT_VALS; ++q) o.slot[j][q] = 0;
      o.name_id[j] = -1;
      o.app_id[j] = -1;
    }
  PROF_MARK(TK_SLOT);
  // 5) 방·완성도·표시
  if (lane == 0) {
    float r[N_ROOMTOK];
    for (int q = 0; q < N_ROOMTOK; ++q) r[q] = 0.f;
    float bd = kInf;
    if (beh) {   // 방 종류 = 장면 방 이름의 종류, 문 = 창 안 문 중 아는 쪽 방이 모두 드러난 것
      const BMapEnv& B = *bxp->bm;
      r[rrk >= 0 ? (int)bxp->sd->rtype[B.rid[rrk]] : N_RTYPE] = 1.f;
      for (int j = 0; j < B.ndoor; ++j) {
        const int a = B.da[j], c2 = B.db[j];
        if ((a >= 0 && !((m.rrev >> a) & 1)) || (c2 >= 0 && !((m.rrev >> c2) & 1))) continue;
        const float dxw = B.dx[j] - px, dyw = B.dy[j] - py;
        const float d = sqrtf(dxw * dxw + dyw * dyw);
        if (d < bd) {
          bd = d;
          r[6] = c * dxw + s * dyw;
          r[7] = -s * dxw + c * dyw;
          r[8] = d;
          r[9] = 1.f;
        }
      }
    } else
    r[rrk >= 0 ? m.rtype[rrk] : N_RTYPE] = 1.f;
    for (int j = 0; !beh && j < m.n_room - 1; ++j) {   // 문 = 양쪽 방이 다 드러난 자르는 선(scenemap: 두 방 사이 이음매)
      if (!((m.rrev >> j) & 1) || !((m.rrev >> (j + 1)) & 1)) continue;
      const float dxw = (m.raxis ? m.rdoor[j] : m.rcut[j]) - px, dyw = (m.raxis ? m.rcut[j] : m.rdoor[j]) - py;
      const float d = sqrtf(dxw * dxw + dyw * dyw);
      if (d < bd) {
        bd = d;
        r[6] = c * dxw + s * dyw;
        r[7] = -s * dxw + c * dyw;
        r[8] = d;
        r[9] = 1.f;
      }
    }
    for (int q = 0; q < N_ROOMTOK; ++q) o.room[q] = f2h(r[q]);
    o.comp[0] = f2h((float)m.n_task_conf);
    o.comp[1] = f2h((float)m.n_obj_conf / (float)(beh ? bxp->bm->nprim : N_PRIM));
    o.comp[2] = f2h(m.room_cells > 0 ? (float)m.n_seen_room / (float)m.room_cells : 0.f);
    o.comp[3] = f2h((float)popc32((uint32_t)m.rrev) / (float)m.n_room);
    o.n_slot = (int16_t)nslot;
    o.flags = (int16_t)((m.kf_flag ? 1 : 0) | (goto_pt ? 2 : 0));
    for (int q = 0; q < N_WAY; ++q) o.way[q] = f2h(ts.wy[q]);
    o.instr1 = (uint16_t)(beh ? bxp->bm->instr + 1 : 0);   // 상자 방은 0(예전 pad 와 같은 바이트)
    o.bkind = (uint16_t)(beh ? bxp->bm->kind : 0);
    for (int q = 0; q < 2; ++q) o.pad2[q] = 0;
    // 목표 칸 2(GoalEnt): 물체 = 목표 칸(T_TARGET 과 같은 칸)의 자리(지금 보는 중이면 관측 자리 — 칸 위치와 같은 규칙), 없으면 같은 짝의 사라진 칸
    // 마지막 자리(GV_LOST), 그것도 없으면 위치 모름. 상자 방: 집을 것 = 컵. BEHAVIOR: prim 0 = B1 은 놓을 곳(가는 곳)·그 밖은 집을 것, prim 1 = 놓을 곳, 점 = 놓을 곳
    const float gc = tbug == 11 ? 1.f : c, gs = tbug == 11 ? 0.f : s;   // tbug 11(음성 대조): 목표 칸을 지도 차(회전 없음)로
    for (int k = 0; k < N_GENT; ++k) goal_fill(o.goal[k], 0, false, false, nullptr, px, py, gc, gs, m.eef_b);
    auto obj_entry = [&](int k, int b0, int bg) {
      const int b = b0 >= 0 ? b0 : bg;
      if (b < 0) { goal_fill(o.goal[k], 1, false, false, nullptr, px, py, gc, gs, m.eef_b); return; }
      const Slot& S = m.slot[b];
      const bool live = !S.held && S.last_seen == t_kf;
      goal_fill(o.goal[k], 1, true, b0 < 0, (live && tbug != 3 && b0 >= 0) ? S.meas : S.pos, px, py, gc, gs, m.eef_b);
    };
    if (!beh) obj_entry(GE_PICK, tgt, gone);
    else {
      const BMapEnv& B = *bxp->bm;
      if (B.goal & 1) obj_entry(B.kind == bsc::EK_B1 ? GE_PLACE : GE_PICK, tgt, gone);
      if (B.goal & 2) obj_entry(GE_PLACE, tgt2, gone2);
      if (B.gmode & bsc::GM_PLACE_PT) goal_fill(o.goal[GE_PLACE], 2, true, false, B.gp, px, py, gc, gs, m.eef_b);
    }
    tv_input(m, bxp, true, false, ts.tv);   // 교사 격자 입력(점·목표 색 없음)
    if (tbug == 12) ts.tv.s = -ts.tv.s;     // 음성 대조: 그림을 거꾸로 돌림
  }
  sync();
  // 6) 교사 격자(topview.h): 덩이 256 개를 레인이 나눔
  const TvWinCell cell{occ, seen};
  for (int k = lane; k < TV_B * TV_B; k += nl) {
    uint8_t no, nu;
    tv_block(ts.tv, cell, k % TV_B, k / TV_B, no, nu);
    o.tv[0][k / TV_B][k % TV_B] = no;
    o.tv[1][k / TV_B][k % TV_B] = nu;
  }
}

// 예전 꼴(레인 수를 실행 때만 앎): 아무 nl ≥ 1 에서 같은 결과
template <class Sync>
DEV void make_tokens(const MapCore& m, const uint32_t* occ, const uint32_t* seen, const int16_t* segs, TPrev* tprev, TokScratch& ts, int lane, int nl,
                     bool write, const Sync& sync) {
  make_tokens_n<1>(m, occ, seen, segs, tprev, ts, lane, nl, write, sync);
}

}  // namespace gmap
