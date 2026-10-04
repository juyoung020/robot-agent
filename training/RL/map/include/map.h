// GPU 안에서 자라는 지도(scenemap 근사, 계획서 GPU_TRAINING.md 5.1·4.4, 단계 G2 앞부분).
// CPU 참조판과 GPU 커널이 **같은 소스**를 쓴다(DEV = __host__ __device__). 판 하나 = 블록 하나(NT 스레드). CPU 는 같은 함수를
// tid = 0, nt = 1 로 부른다. 결과가 스레드 순서와 무관하도록:
//   - 격자: 광선이 지나는 칸은 공유 메모리 비트표(맞음·빈칸)에 atomicOr 로 표시(순서 무관) → 칸마다 한 번 정수 덧셈.
//     scenemap grid.cpp 의 "한 스캔에서 칸마다 한 번, 맞음 우선" 규칙과 같다.
//   - 물체 보임 광선은 쌍마다 따로 칸에 쓰고, 검출·물체 기억(순서가 있는 탐욕 짝짓기)은 스레드 0 이 순서대로 한다.
//   - 난수는 detmath.h 의 splitmix64(판마다 따로, 스레드 0 만 뽑음).
//
// 규칙·기본값의 출처(읽기 전용): src/behavior-2026/src/scene_graph/scenemap
//   grid.hpp GridParams / grid.cpp insert, scan.hpp ScanParams / scan.cpp makeScan, slam2d.hpp SlamParams(움직임 거르기),
//   objmap.hpp ObjParams / objmap.cpp update(짝짓기·확정·옮겨짐·사라짐·버림), scenemap.h sm_object(SM_SEEN..).
// (가정) 표시 값은 실제 기록·사양으로 맞출 값이다. 한 곳(MP)에만 둔다.
#pragma once
#include "env.h"
#include "env_soa.h"
// 잡음·맞춤 값의 기본(MP 가 씀). 기본은 LIMO 탐색 기록 보정(../map_calib/limo, calib_limo.json) + 이 모형에 다시 맞춘 값(README "LIMO 보정").
// map_drift 로 다시 맞출 때만 -D 로 바꾼다. R1 값(예전 기본)은 ../map_calib/README.md 에 기록으로 남김
#ifndef ODO_T_V
#define ODO_T_V 0.10f
#endif
#ifndef ODO_RR_V
#define ODO_RR_V 0.04f
#endif
#ifndef ODO_RT_V
#define ODO_RT_V 0.02f
#endif
#ifndef ODO_BT_V
#define ODO_BT_V 0.0082f
#endif
#ifndef ODO_BR_V
#define ODO_BR_V 0.0121f
#endif
#ifndef ODO_BW_V
#define ODO_BW_V 0.0f
#endif
#ifndef KF_CORR_XY_V
#define KF_CORR_XY_V 0.001f
#endif
#ifndef KF_CORR_YAW_V
#define KF_CORR_YAW_V 0.15f
#endif
#ifndef P_CONF_V
#define P_CONF_V 0.08f
#endif
#ifndef LAT_N_V
#define LAT_N_V 0.023f
#endif
#ifndef N_GHOST_V
#define N_GHOST_V 3
#endif
#ifndef P_GHOST_V
#define P_GHOST_V 0.03f
#endif
// 잡음 끄기(map_cmp 의 진짜 scenemap 비교용, -DMAP_NOISE_V=0): 오도메트리 치우침·걸음 잡음, 놓침, 틀린 이름, 유령, 깊이·옆·크기 잡음을
// 모두 끈다. 기본 1 = 지금 모형 그대로(비트 같음)
#ifndef MAP_NOISE_V
#define MAP_NOISE_V 1
#endif


namespace gmap {
using namespace dm;

// 구간별 시간 재기(map_prof 빌드만, -DMAP_PROF). 블록의 스레드 0 이 clock64 차이를 구간 칸에 더한다. 보통 빌드에서는 아무것도 안 함
enum ProfSec { P_LOAD, P_BEGIN, P_RESET, P_CAST, P_POSE_DET, P_ASSOC, P_ABSENCE, P_OBJPRE, P_MARK, P_APPLY, P_FINISH, P_STORE, P_WALLS, PW1, PW3, PW4, PW5, PW6, PW7, TK_STAGE, TK_A, TK_SEG, TK_SLOT, TK_ROOM, P_NSEC };
// PW* = 벽 단계 안 구간, TK_* = 토큰 커널 구간(토큰 블록 = 판 8 개라 판당 값은 표의 "keyframe 블록당" 의 8 배 / 8 판)
#if defined(MAP_PROF) && defined(__CUDACC__)
__device__ unsigned long long g_prof[P_NSEC + 3];   // + 블록 수(keyframe 아님, keyframe), 표시된 격자 낱말 수
#endif
#if defined(MAP_PROF) && defined(__CUDA_ARCH__)
__device__ __forceinline__ long long& prof_last() { __shared__ long long t; return t; }
#define PROF_START() do { if (threadIdx.x == 0) prof_last() = clock64(); } while (0)
#define PROF_MARK(k) do { if (threadIdx.x == 0) { const long long n_ = clock64(); atomicAdd(&g_prof[k], (unsigned long long)(n_ - prof_last())); prof_last() = n_; } } while (0)
#define PROF_COUNT(k) do { if (threadIdx.x == 0) atomicAdd(&g_prof[P_NSEC + (k)], 1ull); } while (0)
#else
#define PROF_START() do {} while (0)
#define PROF_MARK(k) do {} while (0)
#define PROF_COUNT(k) do {} while (0)
#endif

// ---- 크기 -------------------------------------------------------------------------------------------------------------
constexpr int GW = 128;                 // 격자 창 한 변 칸 수(고정 창, 원점 중심). G1 방은 최대 7 m × 7 m 라 12.8 m 창
constexpr float RES = 0.10f;            // 칸 0.10 m (계획서 8절; scenemap 기본 0.05)
constexpr float INV_RES = 10.0f;
constexpr int GX0 = -GW / 2;            // 창 첫 칸의 전역 칸 번호(칸 = floor(x / RES))
constexpr int NCELL = GW * GW;
constexpr int NWORD = NCELL / 32;
constexpr int NT = 64;                  // 블록 스레드 수
constexpr int NCOL = 64, NROW = 8;      // 거친 깊이 광선: 가로 64 (계획서 5.1 예) × 세로 8 (가정). 128 열은 map_cmp 격자 일치율이 같아(0.982 대 0.983) 64 로 둠
constexpr int NROWC = NROW + 1;         // + 빈칸 줄 하나: 깊이 범위 안 가장 먼 바닥 점(scan.cpp 빈 광선 끝) — MP::scan_step 참고
constexpr int M_SCN = 8;                // 장면 상자(가구 5 + 작은 물건 3) — G1 빈 방에 지도용으로 더함
constexpr int N_PRIM = M_SCN + 1;       // + 컵(과제 물체, 0 번)
constexpr int NPT = 5;                  // 물체마다 보임 광선: 시야로 자른 실루엣 안 5 점(가운데 + 가로·세로 0.8 배 안쪽 넷)
constexpr int KSLOT = 16;               // 물체 기억 칸 (가정; 계획서 8절 예는 64)
constexpr int MAXDET = N_PRIM + N_GHOST_V;   // 한 keyframe 검출 최대(참 물체 + 유령 자리)
constexpr int NCLS = 6;
constexpr int N_MET = 8;
// M_INIT: 이 판의 처음 지도(커리큘럼 5.5) 부호 = 미리 확정한 참 물체 수 | 과제 물체 미리 확정 << 4 | 단계(0 C0, 1 C1, 2 C2) << 5 (float 로)
enum Met { M_TASK, M_OBJ, M_SEEN, M_ERR_XY, M_ERR_YAW, M_KF, M_ROOM, M_INIT };
// 벽(walls.hpp)·방(5.1 의 5 번 근사)
constexpr int MAXSEG = 32;              // 판마다 벽 선분 칸, 가로·세로 따로 (가정; 넘치면 버리고 n_wall_ovf 에 셈)
constexpr int SEGW = 2 * MAXSEG * 4;    // 판마다 선분 int16 수: [가로|세로][칸][X2a, Y2a, X2b, Y2b] (반 칸 단위, 창 원점 기준)
constexpr int NRUN = 128;               // 벽 추출 한 방향의 구간 칸 (가정; 넘치면 그 뒤 행의 구간을 버림)
constexpr int NACT = 24;                // 한 행 구간 최대(128 칸, 구간 ≥ 5 + 틈 1 → 22)
constexpr int MAXROOM = 3;
enum RoomType { R_KITCHEN, R_BATHROOM, R_BEDROOM, R_LIVING, R_OFFICE, N_RTYPE };   // VLA_INPUT 4절 순서(+ 모름)
enum Cls { C_CUP = 0, C_ITEM = 1, C_CHAIR = 2, C_TABLE = 3, C_CABINET = 4, C_BIN = 5 };
static_assert(env::N_FURN == M_SCN && (int)env::FC_ITEM == (int)C_ITEM && (int)env::FC_CHAIR == (int)C_CHAIR && (int)env::FC_TABLE == (int)C_TABLE && (int)env::FC_CABINET == (int)C_CABINET &&
              (int)env::FC_BIN == (int)C_BIN, "A2 furniture boxes = map scene boxes (same classes)");
enum State { S_SEEN = 0, S_GONE = 1, S_MOVED = 2, S_HELD = 3 };   // scenemap.h SM_SEEN..SM_HELD

DEV constexpr int qround(float x) { return x >= 0.f ? (int)(x * 256.f + 0.5f) : -(int)(-x * 256.f + 0.5f); }   // lround(x · kQ)

// ---- 상수(한 곳) --------------------------------------------------------------------------------------------------------
// 출처 표시: (scenemap) 기본값, (Dabai 데이터시트) Orbbec DaBai Datasheet V1.5 2.1.2 절, (R1 시뮬 기록 보정) ../map_calib(101b36a,
// BEHAVIOR 의 R1 기록 — LIMO 가 아님), (맞춤) 아래 모형으로 R1 탐색 판 SLAM 잔차에 맞춘 값(README), (가정) 근거 없음.
struct MP {
  // grid.hpp GridParams + OccGrid::kQ = 256 (scenemap)
  static constexpr int q_hit = qround(0.85f), q_miss = qround(-0.4f), q_min = qround(-4.f), q_max = qround(4.f);
  // 깊이 범위(Dabai 데이터시트: 0.3–3.0 m). 높이 띠는 scan.hpp ScanParams (scenemap)
  // 높이 띠는 LIMO 값 0.05–0.50 m (scenemap capi.cpp robotParams·README LIMO 절: 5 cm 턱, 팔 접은 키 ≈ 0.35 m), 맞추기 점 위 끝 match_hi 3.0 (scan.hpp)
  static constexpr float zmin = 0.3f, zmax = 3.0f, band_lo = 0.05f, band_hi = 0.50f, match_hi = 3.0f;
  // 빈칸 줄: scan.cpp 는 깊이 화소 간격 scan_step 의 모든 바닥 점 중 가장 먼 것을 빈 광선 끝으로 쓴다. 그 줄 = 바닥 깊이가 zmax 안인
  // 첫 화소 줄(v = scan_step 의 배수) (가정: sgrt 깊이 간격 4, 640 × 400 → 160 × 100 점)
  static constexpr int scan_step = 4;
  static constexpr float depth_hfov = env::K::cam_hfov;        // 깊이 가로 FOV 67.9° (Dabai 데이터시트) — 환경 K::cam_hfov 와 같은 값(시뮬 RGB·깊이 모두 67.9°)
  // slam2d.hpp SlamParams (움직임 거르기 update_policy 0, 제자리 규칙) (scenemap)
  static constexpr float mf_xy = 0.05f, mf_yaw = 0.034906585f /*2도*/, still_v = 0.01f, still_w = 0.01f;
  static constexpr int mf_kf = 50;
  // objmap.hpp ObjParams (scenemap). ozmax 만 Dabai 깊이 범위 3.0 m (Dabai 데이터시트; scenemap 기본 5)
  static constexpr int min_points = 20, min_px = 6, confirm = 2, gone_misses = 3;
  static constexpr float ozmin = 0.15f, ozmax = 3.0f, da_min = 0.30f, da_k = 0.5f, da_gap = 0.10f, big = 0.5f;
  static constexpr float moved_d = 0.15f, occl = 0.10f, grow_max = 0.25f, max_ext = 4.0f;
  static constexpr int prune_steps = 100, gone_min_steps = 20;   // prune_s 10 s, gone_min_s 2 s (제어 10 Hz, 영상 = 제어 스텝 (가정))
  // ---- 오도메트리·slam ---- (LIMO 탐색 기록 보정 — ../map_calib/limo. 단서: 기록 하나(탐색 판 094010 + limo3), 카메라 FK 1 cm 오차가 든 판)
  // 걸음마다 랜덤워크: 이동 σ = odo_t·거리, 회전 σ = odo_rr·|dθ| + odo_rt·거리. 원 오도메트리에는 걸음 잡음이 없어 SLAM 잔차에 맞춘 유효 값
  static constexpr float odo_t = ODO_T_V, odo_rr = ODO_RR_V, odo_rt = ODO_RT_V;
  // 판마다 뽑는 배율 치우침(판 리셋 때 가우스, 판 안에서 고정): 이동 배율 σ 0.82 %, 회전 배율 σ 1.21 %, 직진 중 yaw 0
  // (LIMO 탐색 기록 보정: limo3 원 오도메트리 대 GT — 직진 0.80 m 에서 +0.82 %, 제자리 363° 에서 +1.21 %/rad. 표본 하나라 σ = |측정값|, 부호는 ± 대칭)
  static constexpr float odo_bt = ODO_BT_V, odo_br = ODO_BR_V, odo_bw = ODO_BW_V;
  // keyframe 맞추기가 되돌리는 오차 비율, xy 와 yaw 따로 (LIMO 탐색 기록 보정 — 이 모형에서 map_drift kind 2 로 맞춤)
  static constexpr float kf_corr_xy = KF_CORR_XY_V, kf_corr_yaw = KF_CORR_YAW_V;
  // ---- 검출 ----
  // 놓침 확률, 카메라–물체 중심 거리별 계단(R1 시뮬 기록 보정: < 1.5 m 0.15, 1.5–2.5 m 0.10, ≥ 2.5 m 0.79. LIMO 기록으로는 맞추지 못해 그대로)
  static constexpr float miss_d1 = 1.5f, miss_d2 = 2.5f, p_miss_near = 0.15f, p_miss_mid = 0.10f, p_miss_far = 0.79f;
  static constexpr float p_conf = P_CONF_V;                      // 틀린 이름 (LIMO 탐색 기록 보정: 확정 물체 틀린 이름 3/16 = 0.19 에 맞춤)
  // 가짜 물체: 판마다 정해진 유령 자리 n_ghost 개가 시야에 들면 keyframe 마다 p_ghost 로 검출된다
  // (LIMO 탐색 기록 보정: limo3 가짜가 있는 keyframe 1/80 = 0.013, 탐색 판 확정 가짜 0/16)
  static constexpr int n_ghost = N_GHOST_V;
  static constexpr float p_ghost = P_GHOST_V;
  static constexpr float dn0 = 0.0f, dn2 = 0.006f;               // 깊이 잡음 σ = 0 + 0.006·z² m (Dabai 데이터시트: 1 m 에서 6 mm, z² 법칙)
  static constexpr float lat_n = LAT_N_V, ext_n = 0.05f;         // 옆·높이 잡음 σ m (LIMO 탐색 기록 보정: limo3 맞는 검출 중심 강건 σ 0.023, n 100), 크기 잡음 (R1 시뮬 기록 보정)
  // ---- (가정) ----
  static constexpr int img_w = 640, img_h = 400;                 // 깊이 영상 640×400 (Dabai 데이터시트), 정사각 화소(가정: 세로 FOV 45.6°, 사양 45.3°)
  static constexpr float wall_h = 2.5f;                          // 벽 높이(가정)
  static constexpr bool noise = MAP_NOISE_V != 0;                // 잡음 켬(기본). 0 이면 위 잡음·실수가 모두 꺼짐(map_cmp 비교용)
  static constexpr int min_hits = 5;                             // 맞추기 최소 맞은 열 (가정: min_inliers 50 / 720 칸 × 64 열 ≈ 5)
  static constexpr float cup_h = 2.f * env::K::tgt_z;            // 컵 높이 0.10 m (가정: tgt_z 를 중심 높이로 봄)
  // ---- 벽 상태 56 (walls.hpp, scenemap) ----
  // 점유 = export8 값 ≥ kOccMin 65. grid.cpp 표: lround(100·σ(L/256)) ≥ 65 ⇔ L ≥ 153 (map_realcheck 가 진짜 OccGrid 로 확인)
  static constexpr int q_occ = 153;
  static constexpr float wall_range = 4.0f;                      // kMaxRange
  static constexpr int wall_min_run = 5;                         // max(1, lround(kMinLen 0.5 / res)) — res = double(0.10f)
  static constexpr int wall_max_rows = 4;                        // (행 수 · res) > kMaxThick 0.5 면 버림: 5 · double(0.10f) = 0.50000000745 > 0.5
  static constexpr float wall_ign_m = 0.1f, wall_ign_z = 0.4f, wall_ign_max = 5.0f;   // capi.cpp refreshWalls: 바닥 위 확정 물체 상자 + 0.1 m
  static constexpr float wall_valid = 6.0f;                      // wallStateVector: 선분 거리 ≤ kMaxRange · 1.5
  // ---- 들기(objmap.cpp updateHands, LIMO 값 capi.cpp robotParams) ----
  static constexpr float grasp_r = 0.12f, grip_closed = 0.35f, hand_r = 0.10f;   // omx_gripper_joint_1 < 0.35 rad 면 닫힘
  static constexpr float base_z = 0.15f;                         // base_footprint → base_link (URDF base_joint)
  // ---- 방(계획서 5.1 의 5 번: 참 방 표 + 본 비율로 드러냄) ----
  static constexpr float min_room_m2 = 2.0f;                     // rooms.hpp RoomParams min_room_m2 (scenemap)
  static constexpr float room_reveal = 0.3f;                     // 방 칸 중 본 비율이 이 이상이면 드러냄 (가정)
  static constexpr float room_l2 = 4.5f, room_l3 = 6.0f;         // 긴 변이 이 이상이면 방 2·3 개 (가정)
  static constexpr float room_wmin = 1.2f;                       // 방 최소 폭 (가정)
  // ---- 지도 토큰(VLA_INPUT 3절) ----
  static constexpr float arm_reach = 0.40f;                      // 팔이 닿는지: omx_joint1 원점에서 구 (가정: ROBOTIS 도달 400 mm)
  static constexpr float tok_dt = 0.1f;                          // 제어 스텝 s(물체 속도·마지막 본 뒤 시간)
  // ---- 커리큘럼 처음 지도(계획서 5.5) ----
  // 공개한 격자 칸의 로그 오즈: 맞음·빈칸 3 번 본 값 (가정: "예전에 몇 번 본 지도")
  static constexpr int curr_occ = 3 * q_hit, curr_free = 3 * q_miss;
};

// 커리큘럼 처음 지도(5.5): 판 리셋 때 이 판의 단계를 장치 난수로 고른다. **장치 값**이라 바꿔도 다시 컴파일·그래프 다시 잡기가 없다(4.1).
//   C0 전체: 참 물체 모두 확정(참 자리·크기 — 자세 오차만), 방 안 격자 전부 공개
//   C1 부분: 참 물체 kmin..kmax 개(고르게) 확정, 그 물체 둘레 reveal_r 안 격자 공개
//   C2 빈 지도(예전 그대로)
struct MapCurr {
  float p0, p1;     // C0·C1 비율(나머지 C2)
  int kmin, kmax;   // C1 미리 확정할 참 물체 수(N_PRIM 개 중). 계획서 30–70 % → 9 개 중 3–6 개
  float reveal_r;   // C1 격자 공개 반경 m (가정)
  int pad;
};
// 기본 = 모두 C2(빈 지도): G2·G3 와 같은 지도
constexpr MapCurr kCurrEmpty = {0.f, 0.f, 3, 6, 1.5f, 0};
DEV bool is_static(int cls) { return cls == C_TABLE || cls == C_CABINET; }   // capi.cpp kStaticNames 의 table·cabinet

// ---- 판마다 지도 상태 -------------------------------------------------------------------------------------------------
struct Prim {   // 정적 장면 상자(축 정렬), 바닥에 놓임
  int cls;
  float lo[3], hi[3];
};
struct Slot {   // 물체 기억 한 칸 (scenemap.h sm_object / objmap.hpp MapObject 에서)
  int valid, id, cls, n_obs, last_seen, last_kf, state, confirmed, moved, misses, first_miss;   // 시각 = 판 시작 뒤 제어 스텝
  int held;                // 팔 끝이 들고 있음(objmap held_by ≥ 0). 짝짓기·옮겨짐 잇기·부재 확인·벽 무시 영역에서 뺀다
  int src;                 // 마지막 관측의 출처: 참 물체 prim 번호, 유령 g 는 −1−g (토큰의 생김새 표 번호)
  float pos[3], ext[3], first_pos[3], score;
  float seen_len, seen_rot;   // 마지막으로 본 때의 믿는 오도메트리 누적 이동·회전(토큰의 위치 불확실도)
};
struct Ghost {  // 판마다 정해진 가짜 물체 자리(유령): 시야에 들면 keyframe 마다 p_ghost 로 검출된다
  int cls;
  float pos[3], sz;
};
struct MapCore {
  uint64_t rng;
  int ep, t, first, since, n_kf, n_kf_total, next_id, room_cells, n_seen_room, n_task_conf, n_obj_conf, n_dropped, kf_flag, n_fp_total;
  int n_kf_fp;             // 가짜 검출이 하나라도 있던 keyframe 수(누적, 판 리셋에 안 지움 — n_fp_total·n_kf_total 처럼)
  float ex, ey, eyaw;      // slam 이 믿는 자세(참 + 오차)
  float px, py, pyaw;      // 지난 스텝 참 자세(오도메트리 증분)
  float lx, ly, lyaw;      // 지난 keyframe 의 참 자세(움직임 거르기는 참 운동으로)
  float vmax, wmax;        // 지난 keyframe 뒤 속도 최대(제자리 규칙)
  float rhx, rhy;          // 방 반치수(완성도 계산)
  float bt, br, bw;        // 이 판의 오도메트리 치우침: 이동 배율 −1, 회전 배율 −1, 직진 중 yaw rad/m
  // 들기(objmap updateHands): 지난 스텝 그리퍼 닫힘, 든 칸(−1 없음), 팔 끝(base_link·map), 든 물체의 팔 끝 기준 자리(베이스 축)·잡은 자리
  int closed, held_slot, n_grasp_total;   // n_grasp_total: 들기 누적(리셋에 안 지움)
  float eef_b[3], eef_m[3], held_rel[3], grasp_pos[3];
  float fk_q[5];           // eef_b 를 구한 팔 관절(같으면 순기구학을 다시 하지 않음 — 결과 같음)
  float plen, prot;        // 믿는 오도메트리 누적 이동 m·회전 rad(판 안)
  // 벽 선분: 가로·세로 개수(선분은 따로 장치 배열), 넘침 누적
  int nseg_h, nseg_v, n_wall_ovf, n_wall_runs;   // n_wall_runs: 벽 선분을 다시 계산한 횟수(누적, 통계)
  int wnrect;                      // 지난 벽 계산의 무시 영역 수(−1 = 다시 계산해야 함)
  int conf_mask;                   // 확정 칸 비트(valid && confirmed) — keyframe 끝에 다시 셈. 토큰이 이 칸만 읽음
  uint32_t wrect[KSLOT];           // 지난 벽 계산의 무시 영역(창 칸 cx0 | cx1 << 8 | cy0 << 16 | cy1 << 24) — 점유 비트도 그대로면 선분이 같으므로 건너뜀
  // 방(참 방 표, 판 리셋 때): 긴 축(raxis 0 = x 를 자름)을 rcut 에서 자름, 문 = 자르는 선 위 rdoor. 드러난 방 비트 rrev
  int n_room, raxis, rrev;
  // 커리큘럼 처음 지도(판 리셋 때, curr_slots): 단계 0/1/2, 미리 확정한 참 물체 수, 과제 물체(컵)를 미리 확정했나
  int init_stage, init_conf, init_goal, init_pad;
  int rtype[MAXROOM], rcells[MAXROOM], rseen[MAXROOM];
  float rcut[MAXROOM - 1], rdoor[MAXROOM - 1];
  Prim prim[N_PRIM];
  Ghost ghost[N_GHOST_V];
  Slot slot[KSLOT];
};
static_assert(sizeof(MapCore) % 8 == 0, "MapCore must be whole 8-byte words (no tail padding)");
constexpr int CORE_WORDS = (int)(sizeof(MapCore) / 4);

struct Det { int cls, src; float pos[3], ext[3], score, bc[3]; };   // keyframe 한 번의 검출 하나(src: Slot.src 와 같음)
// 검출 기하(참 카메라 기준). med·bc: 보이는 면 점의 축별 중앙값·10–90 백분위 상자 중심(카메라 기준 앞·왼쪽·위), pe: 백분위 폭
struct DetGeo { float ctr[3], ext[3], rx, ry, fwd, left, up, rh, af; float med[3], bc[3], pe[3]; float xr[2], yr[2]; int inr; };   // inr: 일부라도 깊이 범위 안(진단용)
constexpr int NPART = (NT + 31) / 32;

// 블록 안 작업 공간(GPU 공유 메모리, CPU 지역 변수)
struct Scratch {
  // ---- 앞 부분(geo ~ hit)은 물체 단계(obj_absence)까지만 쓴다. GPU 는 그 뒤(격자 표시 동안) 이 자리에 점유 비트를 비동기로 옮겨 두고
  //      격자 갱신이 바뀐 낱말을 고쳐 쓴다 — 벽 단계(WallScratch::occ, 같은 자리 = 0)가 전역에서 다시 읽지 않게
  alignas(16) DetGeo geo[N_PRIM];   // 보임 점과 무관한 판정의 기하(검출이 다시 씀)
  Det det[MAXDET];
  float key[MAXDET * KSLOT];        // 짝 열쇠(관측 × 칸), 짝 아님 = -1
  int obs_to[MAXDET], hit[KSLOT];
  // ---- 뒤 부분은 격자 갱신·끝까지 쓴다
  union {
    struct { uint32_t hitb[NWORD], missb[NWORD]; };   // 격자 표시(phase_mark 부터)
    struct {                                           // 열 끝(phase_cast → phase_mark 앞까지). 열 방향은 열 번호로 다시 구함
      float colt_t[NCOL];           // 맞음·빈 광선 끝의 광학 깊이 t
      float cold0[NCOL], cold1[NCOL];   // 수직면 맞추기 점(띠 아래 끝 ~ match_hi) 중 가장 가까운·먼 t, 없으면 0
      int8_t colt[NCOL];            // 0 없음, 1 맞음(띠 안), 2 빈 광선 끝(바닥)
    };
  };
  uint32_t vism[N_PRIM];            // 물체마다 보이는 점 비트(NPT 개)
  int part[NPART];                  // 워프별 부분합(GPU) / 스레드 0 값(CPU): 맞은 열 수, 새로 본 칸 수, 짝 후보 번호(CPU)
  float partf[NPART];               // 짝 후보 열쇠(CPU 일반판)
  int part2[NPART];                 // 둘째 부분합: 새로 본 방 칸 중 방 1·2 (방 1 | 방 2 << 16)
  int pcand[N_PRIM];                // 보임 점과 무관한 검출 판정을 넘은 물체(후보)
  float tc, ts, to[3];              // 참 yaw 의 cos·sin, 참 카메라 위치
  int found[N_PRIM];                // 완성도: 참 물체마다 확정 칸이 있나
  int nd, more;
  int flags;                        // phase_begin 결과(B_KF, B_RESET)
  int occ_chg;                      // 이번 keyframe 에 점유 비트가 바뀐 낱말이 있나(벽 선분 다시 할지)
  float ec, es;                     // 믿는 yaw 의 cos·sin
};
static_assert(offsetof(Scratch, hitb) >= sizeof(uint32_t) * NWORD, "occupancy copy (offset 0) must fit in the dead object-phase region");
DEV uint32_t* scr_occ(Scratch& sh) { return reinterpret_cast<uint32_t*>(&sh); }   // 점유 비트 사본(GPU, 격자 표시 뒤) = WallScratch::occ

// furn: A2 가구 상자(환경 SoA 의 이 판 자리, 줄 간격 N). 판 리셋 때만 읽는다. nullptr(손으로 만든 EnvView) 이면 가구 없음
struct EnvView { float x, y, yaw, v, w, tx, ty, rhx, rhy; float q[env::N_Q]; int ep; const float* fb; const int* fi; int fs; };
DEV EnvView read_env(const env::Soa& s, int i) {
  const int N = s.N;
  EnvView e;
  e.x = s.f[env::F_X * N + i]; e.y = s.f[env::F_Y * N + i]; e.yaw = s.f[env::F_YAW * N + i];
  e.v = s.f[env::F_V * N + i]; e.w = s.f[env::F_W * N + i];
  e.tx = s.f[env::F_TX * N + i]; e.ty = s.f[env::F_TY * N + i];
  e.rhx = s.f[env::F_RHX * N + i]; e.rhy = s.f[env::F_RHY * N + i];
  for (int k = 0; k < env::N_Q; ++k) e.q[k] = s.f[(env::F_Q0 + k) * N + i];   // 팔 관절(순기구학 → 팔 끝, 들기)
  e.ep = s.iv[env::I_EP * N + i];
  e.fb = s.f + (size_t)env::F_FB0 * N + i;
  e.fi = s.iv + (size_t)env::I_NF * N + i;
  e.fs = N;
  return e;
}

// ---- 카메라(깊이): 베이스 앞 cam_x, 높이 cam_z, 기울기 0 (가정), 가로 시야 = 깊이 FOV 67.9° (MP::depth_hfov) ----------------------
struct Cam { float tanh, tanv, fx, cxp, cyp; };
DEV Cam cam_consts() {
  float s, c;
  sincosf_d(0.5f * MP::depth_hfov, &s, &c);
  Cam k;
  k.tanh = s / c;
  k.tanv = k.tanh * (float)MP::img_h / (float)MP::img_w;
  k.fx = 0.5f * (float)MP::img_w / k.tanh;
  k.cxp = 0.5f * (float)MP::img_w;
  k.cyp = 0.5f * (float)MP::img_h;
  return k;
}

DEV float gauss(uint64_t& r) {   // Irwin–Hall 4 개(분산 1 로 맞춤) — 초월함수 없이 CPU·GPU 같게
  float a = rand01(r);
  a = a + rand01(r);
  a = a + rand01(r);
  a = a + rand01(r);
  return (a - 2.f) * 1.7320508f;
}

DEV void or_bits(uint32_t* p, uint32_t v) {   // 공유 메모리 비트 OR(순서 무관)
#ifdef __CUDA_ARCH__
  atomicOr(p, v);
#else
  *p |= v;
#endif
}
DEV int popc32(uint32_t v) {
#ifdef __CUDA_ARCH__
  return __popc(v);
#else
  return __builtin_popcount(v);
#endif
}
DEV int ctz32(uint32_t v) {   // v != 0
#ifdef __CUDA_ARCH__
  return __ffs((int)v) - 1;
#else
  return __builtin_ctz(v);
#endif
}
// 블록 부분합: GPU 는 워프 합을 part[워프] 에(워프의 모든 레인이 불러야 함), CPU(nt 1) 는 part[0] 에
DEV void put_part(Scratch& sh, int tid, int v) {
#ifdef __CUDA_ARCH__
  v = (int)__reduce_add_sync(0xffffffffu, (unsigned)v);
  if ((tid & 31) == 0) sh.part[tid >> 5] = v;
#else
  sh.part[tid] = v;
#endif
}
DEV int sum_part(const Scratch& sh, int nt) {
  int s = 0;
  for (int k = 0; k < (nt + 31) / 32; ++k) s += sh.part[k];
  return s;
}
DEV void put_part2(Scratch& sh, int tid, int v) {   // 둘째 부분합(같은 규칙, part2)
#ifdef __CUDA_ARCH__
  v = (int)__reduce_add_sync(0xffffffffu, (unsigned)v);
  if ((tid & 31) == 0) sh.part2[tid >> 5] = v;
#else
  sh.part2[tid] = v;
#endif
}
DEV int sum_part2(const Scratch& sh, int nt) {
  int s = 0;
  for (int k = 0; k < (nt + 31) / 32; ++k) s += sh.part2[k];
  return s;
}

// ---- 광선 -------------------------------------------------------------------------------------------------------------
constexpr float kInf = 1e30f;
// 판(slab) 한 축: 들어가는·나가는 t 구간을 [t0, t1] 에 좁힌다. 축에 나란하고 밖이면 false.
// inv = 1/d 는 광선마다 한 번만 구한다(상자마다 다시 나눠도 같은 값이라 결과는 그대로). max·min 은 정확한 연산이라
// 축 순서·묶는 순서를 바꿔도 비트가 같다 — 그래서 열의 xy 판을 줄 8 개가 같이 쓸 수 있다.
DEV bool slab(float o, float d, float inv, float lo, float hi, float& t0, float& t1) {
  if (absf(d) < 1e-12f) return !(o < lo || o > hi);
  float ta = (lo - o) * inv, tb = (hi - o) * inv;
  if (ta > tb) { const float x = ta; ta = tb; tb = x; }
  t0 = maxf(t0, ta);
  t1 = minf(t1, tb);
  return true;
}
DEV float slab_end(float t0, float t1) { return (t0 > t1 || t0 <= 0.f) ? kInf : t0; }
DEV void ray_inv(const float d[3], float inv[3]) {
  for (int a = 0; a < 3; ++a) inv[a] = absf(d[a]) < 1e-12f ? 0.f : 1.f / d[a];
}
DEV float ray_box_inv(const float o[3], const float d[3], const float inv[3], const Prim& b) {   // 들어가는 t (> 0), 없으면 kInf
  float t0 = -kInf, t1 = kInf;
  for (int a = 0; a < 3; ++a)
    if (!slab(o[a], d[a], inv[a], b.lo[a], b.hi[a], t0, t1)) return kInf;
  return slab_end(t0, t1);
}
DEV float ray_room(const float o[3], const float d[3], float rhx, float rhy) {   // 방 안에서 벽·바닥·천장까지
  float t = kInf;
  if (d[0] > 0.f) t = minf(t, (rhx - o[0]) / d[0]); else if (d[0] < 0.f) t = minf(t, (-rhx - o[0]) / d[0]);
  if (d[1] > 0.f) t = minf(t, (rhy - o[1]) / d[1]); else if (d[1] < 0.f) t = minf(t, (-rhy - o[1]) / d[1]);
  if (d[2] < 0.f) t = minf(t, -o[2] / d[2]); else if (d[2] > 0.f) t = minf(t, (MP::wall_h - o[2]) / d[2]);
  return t;
}
DEV float cast(const MapCore& m, const float o[3], const float d[3]) {
  float inv[3];
  ray_inv(d, inv);
  float t = ray_room(o, d, m.rhx, m.rhy);
  for (int p = 0; p < N_PRIM; ++p) t = minf(t, ray_box_inv(o, d, inv, m.prim[p]));
  return t;
}
DEV void cam_world(float x, float y, float c, float s, float o[3]) {
  o[0] = x + c * env::K::cam_x;
  o[1] = y + s * env::K::cam_x;
  o[2] = env::K::cam_z;
}
// 화소 방향(정규화 좌표 xn 오른쪽, yn 아래) → 세계 방향. 광학 z 로 매개(t = 깊이)
DEV void pix_dir(float c, float s, float xn, float yn, float d[3]) {
  d[0] = c + s * xn;    // 베이스 (1, -xn, -yn) 을 yaw 로
  d[1] = s - c * xn;
  d[2] = -yn;
}

// ---- 장면 만들기(판 리셋 때) -------------------------------------------------------------------------------------------
DEV bool box_overlap(const Prim& a, const Prim& b, float m) {
  return a.lo[0] - m < b.hi[0] && b.lo[0] - m < a.hi[0] && a.lo[1] - m < b.hi[1] && b.lo[1] - m < a.hi[1];
}
DEV bool box_has(const Prim& a, float x, float y, float m) {
  return x > a.lo[0] - m && x < a.hi[0] + m && y > a.lo[1] - m && y < a.hi[1] + m;
}
DEV void make_scene(MapCore& m, const EnvView& e) {
  Prim& cup = m.prim[0];
  cup.cls = C_CUP;
  cup.lo[0] = e.tx - env::K::tgt_r; cup.hi[0] = e.tx + env::K::tgt_r;
  cup.lo[1] = e.ty - env::K::tgt_r; cup.hi[1] = e.ty + env::K::tgt_r;
  cup.lo[2] = 0.f; cup.hi[2] = MP::cup_h;
  // A2: 환경이 가구를 갖고 있으면(몸통이 실제로 부딪힘) 그 상자를 그대로 — 광선·가림·검출이 동역학과 같은 장면. m.rng 는 쓰지 않음
  if (e.fi != nullptr && e.fi[0] > 0) {
    for (int k = 1; k <= M_SCN; ++k) {
      Prim& b = m.prim[k];
      b.cls = e.fi[(size_t)k * e.fs];   // I_FC0 + (k − 1) = I_NF + k
      const float* f = e.fb + (size_t)(5 * (k - 1)) * e.fs;
      b.lo[0] = f[0]; b.lo[1] = f[(size_t)e.fs]; b.hi[0] = f[2 * (size_t)e.fs]; b.hi[1] = f[3 * (size_t)e.fs];
      b.lo[2] = 0.f; b.hi[2] = f[4 * (size_t)e.fs];
    }
    return;
  }
  // 가구 5: 벽에 붙임. 크기 = 기본 × [0.8, 1.2] (가정)
  for (int k = 1; k <= 5; ++k) {
    Prim b{};
    for (int tries = 0; tries < 8; ++tries) {
      const int ci = (int)(rand01(m.rng) * 4.f);                 // 0..3 → 의자·탁자·장·쓰레기통
      const int cls = ci == 0 ? C_CHAIR : ci == 1 ? C_TABLE : ci == 2 ? C_CABINET : C_BIN;
      float w = cls == C_CHAIR ? 0.45f : cls == C_TABLE ? 0.8f : cls == C_CABINET ? 0.5f : 0.3f;   // 벽 따라
      float dp = cls == C_CHAIR ? 0.45f : cls == C_TABLE ? 0.5f : cls == C_CABINET ? 0.4f : 0.3f;  // 벽에서
      float h = cls == C_CHAIR ? 0.9f : cls == C_TABLE ? 0.75f : cls == C_CABINET ? 1.0f : 0.4f;
      w = w * rand_range(m.rng, 0.8f, 1.2f);
      dp = dp * rand_range(m.rng, 0.8f, 1.2f);
      h = h * rand_range(m.rng, 0.8f, 1.2f);
      const int side = (int)(rand01(m.rng) * 4.f);
      const bool xw = side < 2;                                  // x = ±rhx 벽
      const float along_r = (xw ? e.rhy : e.rhx) - 0.5f * w - 0.05f;
      const float u = rand_range(m.rng, -along_r, along_r);
      const float sg = (side & 1) ? -1.f : 1.f;
      const float perp = sg * ((xw ? e.rhx : e.rhy) - 0.5f * dp - 0.02f);
      const float cx = xw ? perp : u, cy = xw ? u : perp;
      const float hx = 0.5f * (xw ? dp : w), hy = 0.5f * (xw ? w : dp);
      b.cls = cls;
      b.lo[0] = cx - hx; b.hi[0] = cx + hx; b.lo[1] = cy - hy; b.hi[1] = cy + hy; b.lo[2] = 0.f; b.hi[2] = h;
      bool ok = !box_has(b, e.x, e.y, 0.35f) && !box_has(b, e.tx, e.ty, 0.15f);
      for (int j = 1; j < k && ok; ++j) ok = !box_overlap(b, m.prim[j], 0.05f);
      if (ok) break;
    }
    m.prim[k] = b;
  }
  // 작은 물건 3: 방 안 아무 데나(로봇·컵에서 떨어지게)
  for (int k = 6; k < N_PRIM; ++k) {
    Prim b{};
    for (int tries = 0; tries < 8; ++tries) {
      const float hx = 0.5f * rand_range(m.rng, 0.08f, 0.25f), hy = 0.5f * rand_range(m.rng, 0.08f, 0.25f);
      const float h = rand_range(m.rng, 0.12f, 0.35f);
      const float cx = rand_range(m.rng, -e.rhx + 0.5f, e.rhx - 0.5f), cy = rand_range(m.rng, -e.rhy + 0.5f, e.rhy - 0.5f);
      b.cls = C_ITEM;
      b.lo[0] = cx - hx; b.hi[0] = cx + hx; b.lo[1] = cy - hy; b.hi[1] = cy + hy; b.lo[2] = 0.f; b.hi[2] = h;
      bool ok = !box_has(b, e.x, e.y, 0.4f) && !box_has(b, e.tx, e.ty, 0.3f);
      for (int j = 1; j < k && ok; ++j) ok = !box_overlap(b, m.prim[j], 0.05f);
      if (ok) break;
    }
    m.prim[k] = b;
  }
}

// ---- 방(계획서 5.1 의 5 번 근사): 참 방 표 ----------------------------------------------------------------------------------
// G1 은 칸막이 없는 직사각형 방 하나라, 긴 축을 1–2 번 잘라 방 1–3 개로 나눈다(열린 구획). 자르는 선은 장면 상자(가구·작은 물건·컵)를
// 지나지 않게 고른다. 문 = 자르는 선 위 한 점. 방 종류는 아무거나(가정). 칸막이 벽은 장면에 넣지 않는다(G1 동역학·광선 그대로).
// 난수는 m.rng 를 섞은 따로 된 값으로 뽑아 m.rng 를 움직이지 않는다(기존 지도 결과가 그대로).
DEV int room_of(const MapCore& m, float x, float y) {   // 칸 중심·자리 → 방 번호(0..n_room−1), 방 밖 −1
  if (!(absf(x) < m.rhx && absf(y) < m.rhy)) return -1;
  const float u = m.raxis ? y : x;
  int k = 0;
  for (int j = 0; j < m.n_room - 1; ++j) k += u >= m.rcut[j];
  return k;
}
DEV void make_rooms(MapCore& m, const EnvView& e) {
  uint64_t rr = m.rng ^ 0x2545F4914F6CDD1Dull;
  const int ax = e.rhx >= e.rhy ? 0 : 1;
  const float H = ax == 0 ? e.rhx : e.rhy, S = ax == 0 ? e.rhy : e.rhx;   // 자르는 축 반길이, 옆 축 반길이
  const float L = 2.f * H;
  const int want = 1 + (L >= MP::room_l2 ? 1 : 0) + (L >= MP::room_l3 ? 1 : 0);
  m.raxis = ax;
  m.n_room = 1;
  m.rrev = 0;
  float prev = -H;
  for (int k = 1; k < want; ++k) {
    float cut = 0.f;
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
      cut = -H + L * ((float)k / (float)want) + (rand01(rr) - 0.5f) * 0.3f * L / (float)want;
      ok = cut - prev >= MP::room_wmin && H - cut >= MP::room_wmin * (float)(want - k);
      for (int p = 0; p < N_PRIM && ok; ++p) ok = !(m.prim[p].lo[ax] < cut + 0.05f && m.prim[p].hi[ax] > cut - 0.05f);
    }
    if (!ok) break;
    m.rcut[k - 1] = cut;
    m.rdoor[k - 1] = rand_range(rr, -S + 0.5f, S - 0.5f);
    m.n_room = k + 1;
    prev = cut;
  }
  for (int k = 0; k < MAXROOM; ++k) {
    const int ty = (int)(rand01(rr) * (float)N_RTYPE);
    m.rtype[k] = ty < N_RTYPE ? ty : N_RTYPE - 1;
    m.rcells[k] = 0;
    m.rseen[k] = 0;
  }
  int nside = 0;
  for (int l = 0; l < GW; ++l) nside += absf(((float)(l + GX0) + 0.5f) * RES) < S;
  for (int l = 0; l < GW; ++l) {
    const float cc = ((float)(l + GX0) + 0.5f) * RES;
    if (!(absf(cc) < H)) continue;
    int k = 0;
    for (int j = 0; j < m.n_room - 1; ++j) k += cc >= m.rcut[j];
    m.rcells[k] += nside;
  }
}

DEV void init_core(MapCore& m, uint64_t seed, int i) {
  uint32_t* w = reinterpret_cast<uint32_t*>(&m);
  for (int k = 0; k < CORE_WORDS; ++k) w[k] = 0u;
  m.rng = seed * 0xD1B54A32D192ED03ull + (uint64_t)i * 0x9E3779B97F4A7C15ull + 0xABCDEFull;
  m.ep = -1;   // 첫 스텝에서 리셋
}

DEV void reset_core(MapCore& m, const EnvView& e) {
  m.ep = e.ep;
  m.t = 0; m.first = 1; m.since = 0; m.n_kf = 0; m.next_id = 1;
  m.n_seen_room = 0; m.n_task_conf = 0; m.n_obj_conf = 0; m.n_dropped = 0;
  m.ex = e.x; m.ey = e.y; m.eyaw = e.yaw;   // 판 시작 자세는 안다(scenemap: 원점에서 시작)
  m.px = e.x; m.py = e.y; m.pyaw = e.yaw;
  m.lx = e.x; m.ly = e.y; m.lyaw = e.yaw;
  m.vmax = 0.f; m.wmax = 0.f;
  m.rhx = e.rhx; m.rhy = e.rhy;
  for (int k = 0; k < KSLOT; ++k) {
    uint32_t* w = reinterpret_cast<uint32_t*>(&m.slot[k]);
    for (int j = 0; j < (int)(sizeof(Slot) / 4); ++j) w[j] = 0u;
  }
  make_scene(m, e);
  // 이 판의 오도메트리 치우침(판 안에서 고정): 이동 배율, 회전 배율, 직진 중 yaw 표류
  m.bt = (MP::noise ? gauss(m.rng) : 0.f) * MP::odo_bt;
  m.br = (MP::noise ? gauss(m.rng) : 0.f) * MP::odo_br;
  m.bw = (MP::noise ? gauss(m.rng) : 0.f) * MP::odo_bw;
  // 유령 자리: 방 안 아무 데나, 높이·크기·이름은 예전 가짜 물체와 같은 범위(가정)
  for (int g = 0; g < MP::n_ghost; ++g) {
    Ghost& G = m.ghost[g];
    G.pos[0] = rand_range(m.rng, -e.rhx + 0.3f, e.rhx - 0.3f);
    G.pos[1] = rand_range(m.rng, -e.rhy + 0.3f, e.rhy - 0.3f);
    G.pos[2] = rand_range(m.rng, 0.05f, 0.5f);
    G.sz = rand_range(m.rng, 0.05f, 0.2f);
    G.cls = (int)(rand01(m.rng) * (float)NCLS);
  }
  int nx = 0, ny = 0;
  for (int l = 0; l < GW; ++l) {
    const float cc = ((float)(l + GX0) + 0.5f) * RES;
    nx += absf(cc) < e.rhx;
    ny += absf(cc) < e.rhy;
  }
  m.room_cells = nx * ny;
  // 들기·벽·방(새 판)
  m.closed = 0; m.held_slot = -1;
  for (int a = 0; a < 3; ++a) { m.eef_b[a] = 0.f; m.eef_m[a] = 0.f; m.held_rel[a] = 0.f; m.grasp_pos[a] = 0.f; }
  m.plen = 0.f; m.prot = 0.f;
  m.nseg_h = 0; m.nseg_v = 0; m.wnrect = -1; m.conf_mask = 0;
  make_rooms(m, e);
}

// ---- 1. 시작(판마다 한 스레드): 리셋, 오도메트리 표류, 움직임 거르기 -------------------------------------------------------------
// 돌려주는 값: B_KF(이번 스텝 keyframe) | B_RESET(판 리셋). MapCore 의 머리(prim 앞)만 바꾸고, 리셋이면 MapCore 전체를 새로 쓴다.
enum BeginFlag { B_KF = 1, B_RESET = 2, B_WALL = 4 };   // B_WALL: 들기·놓기로 벽 무시 영역이 바뀜 → 벽 선분 다시
constexpr int LIST_SHIFT = 29;                           // 장치 목록 낱말 = 판 번호 | 시작 결과 << 29

DEV float dist3(const float a[3], const float b[3]) {
  const float d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
  return sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
}
// 들기·놓기(objmap.cpp updateHands, 손 하나 = LIMO). 팔 끝 = G1 순기구학 omx_end_effector_link(base_link) → base_footprint → map(믿는 자세).
// 그리퍼 omx_gripper_joint_1 < grip_closed 가 되는 순간 팔 끝 grasp_r 안 가장 가까운 확정·안 든·사라짐 아닌 물체를 들고(팔 끝 기준 자리를
// 베이스 축으로 기억), 열리는 순간 놓는다(잡은 자리에서 moved_d 넘게 옮겼으면 옮겨짐). 든 동안 매 스텝 팔 끝을 따라간다.
// 놓을 때 받침에 붙이기(parent)는 하지 않는다. 돌려주는 값: 들기·놓기가 있었으면 1(벽 무시 영역이 바뀜)
DEV int hands_step(MapCore& m, const EnvView& e) {
  bool same_q = m.t > 0;   // 판 첫 스텝은 늘 계산
  for (int k = 0; k < 5; ++k) same_q = same_q && m.fk_q[k] == e.q[k];
  if (!same_q) {
    float qd[env::N_Q];
    for (int k = 0; k < env::N_Q; ++k) qd[k] = 0.f;
    env::Fk f;
    env::fk(e.q, qd, f);
    for (int a = 0; a < 3; ++a) m.eef_b[a] = f.ee_p[a];
    for (int k = 0; k < 5; ++k) m.fk_q[k] = e.q[k];
  }
  float s, c;
  sincosf_d(m.eyaw, &s, &c);
  const float bx = m.eef_b[0], by = m.eef_b[1], bz = m.eef_b[2] + MP::base_z;
  m.eef_m[0] = m.ex + (c * bx - s * by);
  m.eef_m[1] = m.ey + (s * bx + c * by);
  m.eef_m[2] = bz;
  const int closed = e.q[5] < MP::grip_closed ? 1 : 0;
  int ev = 0;
  if (closed && !m.closed) {   // 잡기
    int best = -1;
    float bd = MP::grasp_r;
    for (int b = 0; b < KSLOT; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || !S.confirmed || S.held || S.state == S_GONE) continue;
      const float d = dist3(S.pos, m.eef_m);
      if (d < bd) { bd = d; best = b; }
    }
    if (best >= 0) {
      Slot& S = m.slot[best];
      S.held = 1;
      m.held_slot = best;
      const float d0 = S.pos[0] - m.eef_m[0], d1 = S.pos[1] - m.eef_m[1], d2 = S.pos[2] - m.eef_m[2];
      m.held_rel[0] = c * d0 + s * d1;
      m.held_rel[1] = -s * d0 + c * d1;
      m.held_rel[2] = d2;
      for (int a = 0; a < 3; ++a) m.grasp_pos[a] = S.pos[a];
      S.state = S_HELD;
      m.n_grasp_total += 1;
      ev = 1;
    }
  } else if (!closed && m.closed && m.held_slot >= 0) {   // 놓기
    Slot& S = m.slot[m.held_slot];
    S.held = 0;
    const bool mv = dist3(S.pos, m.grasp_pos) > MP::moved_d;
    S.moved = (S.moved || mv) ? 1 : 0;
    S.state = S.moved ? S_MOVED : S_SEEN;
    S.misses = 0;
    m.held_slot = -1;
    ev = 1;
  }
  m.closed = closed;
  if (m.held_slot >= 0) {   // 든 물체는 팔 끝을 따라간다(팔 끝 기준 자리를 지금 yaw 로 돌림)
    Slot& S = m.slot[m.held_slot];
    const float* r = m.held_rel;
    S.pos[0] = m.eef_m[0] + (c * r[0] - s * r[1]);
    S.pos[1] = m.eef_m[1] + (s * r[0] + c * r[1]);
    S.pos[2] = m.eef_m[2] + r[2];
  }
  return ev;
}

DEV int phase_begin(MapCore& m, const EnvView& e, int force_kf) {
  int reset = 0;
  if (e.ep != m.ep) {
    reset_core(m, e);
    reset = B_RESET;
  } else {
    // 참 증분(지난 참 자세 기준 몸 좌표) → 이 판의 배율 치우침 + 걸음마다 잡음 → 믿는 자세에 붙임 (slam2d pushVelocity 적분의 오차 흉내)
    float s0, c0;
    sincosf_d(m.pyaw, &s0, &c0);
    const float dxw = e.x - m.px, dyw = e.y - m.py;
    const float dxb = c0 * dxw + s0 * dyw, dyb = -s0 * dxw + c0 * dyw;
    const float dth = wrap_pi(e.yaw - m.pyaw);
    const float dist = sqrtf(dxb * dxb + dyb * dyb);
    const float n1 = MP::noise ? gauss(m.rng) : 0.f, n2 = MP::noise ? gauss(m.rng) : 0.f, n3 = MP::noise ? gauss(m.rng) : 0.f;
    const float sx = MP::odo_t * dist, st = 1.f + m.bt;
    const float nxb = dxb * st + n1 * sx, nyb = dyb * st + n2 * sx;
    const float nth = dth * (1.f + m.br) + m.bw * dist + n3 * (MP::odo_rr * absf(dth) + MP::odo_rt * dist);
    float se, ce;
    sincosf_d(m.eyaw, &se, &ce);
    m.ex = m.ex + (ce * nxb - se * nyb);
    m.ey = m.ey + (se * nxb + ce * nyb);
    m.eyaw = wrap_pi(m.eyaw + nth);
    m.plen = m.plen + sqrtf(nxb * nxb + nyb * nyb);
    m.prot = m.prot + absf(nth);
    m.t += 1;
  }
  const int wall = hands_step(m, e) ? B_WALL : 0;   // scenemap integrate: 자세 적분 뒤 updateHands
  m.px = e.x; m.py = e.y; m.pyaw = e.yaw;
  m.vmax = maxf(m.vmax, absf(e.v));
  m.wmax = maxf(m.wmax, absf(e.w));
  // slam2d insertStage update_policy 0: 처음, 또는 mf_xy·mf_yaw 넘게 움직임, 또는 mf_kf 번째.
  // 움직임은 **참 자세**로 잰다(lx·ly·lyaw = 지난 keyframe 의 참 자세). 믿는 자세로 재면 회전 중 yaw 잡음이 참 회전을 지워
  // keyframe 이 빠지고 보정도 빠진다(map_calib README 2.3). 실제 slam 은 맞춘 자세로 거르므로 참 운동에 더 가깝다
  m.since += 1;
  const float dx = e.x - m.lx, dy = e.y - m.ly;
  const bool moved = dx * dx + dy * dy >= MP::mf_xy * MP::mf_xy || absf(wrap_pi(e.yaw - m.lyaw)) >= MP::mf_yaw;
  const int kf = (m.first || force_kf || moved || m.since >= MP::mf_kf) ? 1 : 0;
  m.kf_flag = kf;
  return (kf ? B_KF : 0) | reset | wall;
}

DEV void clear_grid(int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt) {
  uint32_t* Lw = reinterpret_cast<uint32_t*>(L);
  for (int k = tid; k < NCELL / 2; k += nt) Lw[k] = 0u;
  for (int k = tid; k < NWORD; k += nt) { seen[k] = 0u; occ[k] = 0u; }
}

// ---- 2. 광선(모든 스레드): 깊이 줄 + 물체 점 보임. 참 장면·참 카메라에서 쏜다 ----------------------------------------------
// 물체 pi 의 보임 광선 q: 시야로 자른 실루엣(정규화 화면 좌표 xr × yr) 안 점 방향으로 쏴서, 자기 상자에 먼저 닿고(다른 상자가 가리지 않고)
// 그 깊이가 [ozmin, ozmax] 안이면 보임. 실루엣은 상자 모서리의 투영 범위로 어림한다(모서리 사이로 빗나가면 안 보임으로 셈)
DEV int vis_point(const MapCore& m, const float o[3], float c, float s, const DetGeo& g, int pi, int q) {
  const float xc = 0.5f * (g.xr[0] + g.xr[1]), yc = 0.5f * (g.yr[0] + g.yr[1]);
  const float hx = 0.4f * (g.xr[1] - g.xr[0]), hy = 0.4f * (g.yr[1] - g.yr[0]);
  const float xn = xc + (q == 1 ? -hx : q == 2 ? hx : 0.f), yn = yc + (q == 3 ? -hy : q == 4 ? hy : 0.f);
  float d[3];
  pix_dir(c, s, xn, yn, d);
  float inv[3];
  ray_inv(d, inv);
  const float t = ray_box_inv(o, d, inv, m.prim[pi]);   // 광학 깊이(d 의 앞 성분 = 1)
  if (!(t >= MP::ozmin && t <= MP::ozmax)) return 0;
  for (int j = 0; j < N_PRIM; ++j)
    if (j != pi && ray_box_inv(o, d, inv, m.prim[j]) < t) return 0;   // 다른 상자가 사이를 가림
  return 1;
}

DEV float max3(const float v[3]) { return maxf(v[0], maxf(v[1], v[2])); }

// 보이는 면 점의 통계(objmap: 마스크 안 깊이 점의 축별 중앙값 = 위치, 10–90 백분위 = 상자). 상자에서 카메라 쪽을 보는 면(x 면·y 면 하나씩,
// 카메라가 윗면보다 높으면 윗면)마다 화소 수 ∝ 넓이·cos / 거리² 로 무게를 주고, 축 a 의 값은 "a 면의 점 = 그 면 평면 한 값(무게 w_a)" +
// "다른 면의 점 = [lo, hi] 고르게(나머지 무게)" 의 섞임으로 본다. 분위수는 닫힌 꼴이라 CPU·GPU 같은 연산. 가림·시야·깊이 범위 자르기와
// 마스크 깎기·MAD 거르기는 보지 않는다(가정). 면이 안 보이면(카메라가 상자 안) 상자 중심·폭
DEV float mix_quantile(float lo, float hi, int side, float wa, float W, float q) {
  // side −1: 면이 lo 쪽, +1: hi 쪽, 0: 그 축 면이 안 보임(고르게만)
  if (side == 0 || W <= 0.f) return lo + (hi - lo) * q;
  const float rest = W - wa;
  if (side < 0) { const float m = q * W; return m <= wa || rest <= 0.f ? lo : lo + (hi - lo) * ((m - wa) / rest); }
  const float m = (1.f - q) * W;
  return m <= wa || rest <= 0.f ? hi : hi - (hi - lo) * ((m - wa) / rest);
}
DEV void surf_stats(const Prim& b, const float o[3], float med[3], float plo[3], float phi[3]) {
  float w[3], W = 0.f;
  int side[3];
  for (int a = 0; a < 3; ++a) {
    side[a] = o[a] < b.lo[a] ? -1 : o[a] > b.hi[a] ? 1 : 0;
    if (a == 2 && side[a] < 0) side[a] = 0;   // 바닥 면은 안 보임
    w[a] = 0.f;
    if (!side[a]) continue;
    const int a1 = (a + 1) % 3, a2 = (a + 2) % 3;
    const float f = side[a] < 0 ? b.lo[a] : b.hi[a];
    float fc[3];
    fc[a] = f; fc[a1] = 0.5f * (b.lo[a1] + b.hi[a1]); fc[a2] = 0.5f * (b.lo[a2] + b.hi[a2]);
    const float dx = fc[0] - o[0], dy = fc[1] - o[1], dz = fc[2] - o[2];
    const float d2 = dx * dx + dy * dy + dz * dz;
    const float area = (b.hi[a1] - b.lo[a1]) * (b.hi[a2] - b.lo[a2]);
    w[a] = area * absf(o[a] - f) / (d2 * sqrtf(d2));
    W = W + w[a];
  }
  for (int a = 0; a < 3; ++a) {
    med[a] = mix_quantile(b.lo[a], b.hi[a], side[a], w[a], W, 0.5f);
    plo[a] = mix_quantile(b.lo[a], b.hi[a], side[a], w[a], W, 0.1f);
    phi[a] = mix_quantile(b.lo[a], b.hi[a], side[a], w[a], W, 0.9f);
  }
}

// 3b 검출 판정 중 보임 점과 무관한 부분(참 장면·참 카메라). 상자의 일부라도 깊이 범위 [ozmin, ozmax] 안(모서리 8 개의 앞 거리로)이고
// min_px 를 넘고, 보임 점이 다 보여도 넓이가 min_points 를 못 넘으면 검출될 수 없으므로 false — 그 물체의 보임 광선을 쏘지 않는다
// (넓이는 보이는 점 수에 단조라 결과는 그대로). 넓이 = af · nv / NPT. 크기·넓이는 중심 앞 거리를 깊이 범위로 자른 값으로 어림
DEV bool det_prefilter(const MapCore& m, const float o[3], float c, float s, const Cam& k, int p, DetGeo& g) {
  const Prim& b = m.prim[p];
  for (int a = 0; a < 3; ++a) { g.ctr[a] = 0.5f * (b.lo[a] + b.hi[a]); g.ext[a] = b.hi[a] - b.lo[a]; }
  g.rx = g.ctr[0] - o[0]; g.ry = g.ctr[1] - o[1];
  g.fwd = c * g.rx + s * g.ry; g.left = -s * g.rx + c * g.ry; g.up = g.ctr[2] - o[2];
  float fmin = kInf, fmax = -kInf;
  for (int q = 0; q < 4; ++q) {
    const float x = ((q & 1) ? b.hi[0] : b.lo[0]) - o[0], y = ((q & 2) ? b.hi[1] : b.lo[1]) - o[1];
    const float f = c * x + s * y;
    fmin = minf(fmin, f); fmax = maxf(fmax, f);
  }
  g.inr = fmax >= MP::ozmin && fmin <= MP::ozmax;
  if (!g.inr) return false;
  const float fe = minf(maxf(g.fwd, maxf(fmin, MP::ozmin)), MP::ozmax);   // 깊이 범위 안 앞 거리
  const float size_px = k.fx * max3(g.ext) / fe;                          // objmap 부재 확인과 같은 식
  g.rh = sqrtf(g.rx * g.rx + g.ry * g.ry);
  const float ux = g.rx / g.rh, uy = g.ry / g.rh;
  const float wsil = g.ext[0] * absf(uy) + g.ext[1] * absf(ux);          // 수평 실루엣 폭
  g.af = (k.fx * wsil / fe) * (k.fx * g.ext[2] / fe);
  {  // 실루엣의 화면 범위(모서리 8 개 투영, 카메라 뒤 모서리가 있으면 그쪽 끝은 시야 끝) → 시야로 자름. 넓이에 시야 안 비율을 곱함
    float x0 = kInf, x1 = -kInf, y0 = kInf, y1 = -kInf;
    bool behind = false;
    for (int q = 0; q < 8; ++q) {
      const float x = ((q & 1) ? b.hi[0] : b.lo[0]) - o[0], y = ((q & 2) ? b.hi[1] : b.lo[1]) - o[1], z = ((q & 4) ? b.hi[2] : b.lo[2]) - o[2];
      const float f = c * x + s * y, l = -s * x + c * y;
      if (f < 0.05f) { behind = true; continue; }
      const float iv = 1.f / f, xq = -l * iv, yq = -z * iv;
      x0 = minf(x0, xq); x1 = maxf(x1, xq); y0 = minf(y0, yq); y1 = maxf(y1, yq);
    }
    if (behind) { x0 = -kInf; x1 = kInf; y0 = minf(y0, -k.tanv); y1 = maxf(y1, k.tanv); }
    g.xr[0] = maxf(x0, -k.tanh); g.xr[1] = minf(x1, k.tanh);
    g.yr[0] = maxf(y0, -k.tanv); g.yr[1] = minf(y1, k.tanv);
    if (!(g.xr[0] < g.xr[1] && g.yr[0] < g.yr[1])) return false;   // 시야 밖
    const float fx = behind ? 1.f : (g.xr[1] - g.xr[0]) / (x1 - x0), fy = behind ? 1.f : (g.yr[1] - g.yr[0]) / (y1 - y0);
    g.af = g.af * fx * fy;
  }
  if (size_px < (float)MP::min_px || g.af < (float)MP::min_points) return false;   // af · NPT/NPT = af
  float med[3], plo[3], phi[3];
  surf_stats(b, o, med, plo, phi);
  const float mx = med[0] - o[0], my = med[1] - o[1];
  g.med[0] = c * mx + s * my; g.med[1] = -s * mx + c * my; g.med[2] = med[2] - o[2];
  const float bx = 0.5f * (plo[0] + phi[0]) - o[0], by = 0.5f * (plo[1] + phi[1]) - o[1];
  g.bc[0] = c * bx + s * by; g.bc[1] = -s * bx + c * by; g.bc[2] = 0.5f * (plo[2] + phi[2]) - o[2];
  for (int a = 0; a < 3; ++a) g.pe[a] = phi[a] - plo[a];
  return true;
}

// 줄 r 의 세로 정규화 좌표 yn(아래 +). 줄 0..NROW−1 은 세로 시야를 고르게, 줄 NROW 는 빈칸 줄(MP::scan_step)
DEV void row_yn(const Cam& k, float yn[NROWC]) {
  for (int r = 0; r < NROW; ++r) yn[r] = ((float)(2 * r + 1 - NROW) / (float)NROW) * k.tanv;
  // 바닥 깊이가 zmax 안인 첫 화소 줄: v = cy + cam_z/zmax · fy 를 scan_step 배수로 올림(화소 중심 cy = h/2 − 0.5)
  const float cy = 0.5f * (float)MP::img_h - 0.5f;
  const float v = cy + (env::K::cam_z / MP::zmax) * k.fx;
  const float vs = ceilf(v / (float)MP::scan_step) * (float)MP::scan_step;
  yn[NROW] = (vs - cy) / k.fx;
}

DEV void phase_cast(const MapCore& m, Scratch& sh, const EnvView& e, int tid, int nt) {
  for (int p = tid; p < N_PRIM; p += nt) sh.vism[p] = 0u;   // 보임 점 비트(obj_pre 가 동기 뒤에 OR)
  const Cam k = cam_consts();
  float s, c;
  sincosf_d(e.yaw, &s, &c);
  float o[3];
  cam_world(e.x, e.y, c, s, o);
  if (tid == 0) { sh.tc = c; sh.ts = s; sh.to[0] = o[0]; sh.to[1] = o[1]; sh.to[2] = o[2]; sh.occ_chg = 0; }   // 뒤 단계가 같은 값을 다시 씀
  // 줄마다 같은 값(열과 무관): 세로 방향 d2 = -yn, 1/d2, 바닥·천장까지 t
  float yn[NROWC], invz[NROWC], troomz[NROWC];
  row_yn(k, yn);
  for (int r = 0; r < NROWC; ++r) {
    const float d2 = -yn[r];
    invz[r] = absf(d2) < 1e-12f ? 0.f : 1.f / d2;
    troomz[r] = d2 < 0.f ? -o[2] / d2 : d2 > 0.f ? (MP::wall_h - o[2]) / d2 : kInf;
  }
  // scan.cpp makeScan: 띠 [band_lo, band_hi] 안 가장 가까운 점 = 장애물, 띠 아래(바닥) 가장 먼 점 = 빈 광선 끝. 기울기 0 이라 열 = 방위 칸.
  // 수직면(방 벽·상자 옆면)에 맞은 줄 중 높이 [band_lo, match_hi] 는 맞추기 점(dense) — 지도에 맞음으로만 넣는다(가장 가까운·먼 것 둘).
  // 한 열의 줄은 수평 방향이 같다: 방 벽·상자의 xy 판은 열마다 한 번, z 판만 줄마다(ray_room·ray_box 와 비트 같음)
  for (int col = tid; col < NCOL; col += nt) {
    const float xn = ((float)(2 * col + 1 - NCOL) / (float)NCOL) * k.tanh;
    const float d0 = c + s * xn, d1 = s - c * xn;   // pix_dir
    float txy = kInf;
    if (d0 > 0.f) txy = minf(txy, (m.rhx - o[0]) / d0); else if (d0 < 0.f) txy = minf(txy, (-m.rhx - o[0]) / d0);
    if (d1 > 0.f) txy = minf(txy, (m.rhy - o[1]) / d1); else if (d1 < 0.f) txy = minf(txy, (-m.rhy - o[1]) / d1);
    float tr[NROWC];
    uint32_t vert = 0u;   // 줄마다 수직면에 맞았나
    for (int r = 0; r < NROWC; ++r) { tr[r] = minf(txy, troomz[r]); vert |= (txy < troomz[r] ? 1u : 0u) << r; }
    float trmax = tr[0];
    for (int r = 1; r < NROWC; ++r) trmax = maxf(trmax, tr[r]);
    const float inv0 = absf(d0) < 1e-12f ? 0.f : 1.f / d0, inv1 = absf(d1) < 1e-12f ? 0.f : 1.f / d1;
    for (int p = 0; p < N_PRIM; ++p) {
      const Prim& b = m.prim[p];
      float t0 = -kInf, t1 = kInf;
      if (!slab(o[0], d0, inv0, b.lo[0], b.hi[0], t0, t1) || !slab(o[1], d1, inv1, b.lo[1], b.hi[1], t0, t1)) continue;
      if (t0 > t1) continue;   // z 판은 구간을 좁히기만 하므로 이미 비면 어느 줄도 안 맞음
      if (t0 >= trmax) continue;  // 들어가는 t ≥ 모든 줄의 지금 값: 어느 줄도 줄일 수 없음(minf 결과 같음)
      for (int r = 0; r < NROWC; ++r) {
        float u0 = t0, u1 = t1;
        if (!slab(o[2], -yn[r], invz[r], b.lo[2], b.hi[2], u0, u1)) continue;
        const float te = slab_end(u0, u1);
        if (te < tr[r]) {
          tr[r] = te;
          vert = (vert & ~(1u << r)) | ((u0 == t0 ? 1u : 0u) << r);   // xy 판으로 들어감 = 옆면
        }
      }
      trmax = tr[0];
      for (int r = 1; r < NROWC; ++r) trmax = maxf(trmax, tr[r]);
    }
    float best = kInf, hit_t = 0.f, floor_r = 0.f, floor_t = 0.f, dv0 = 0.f, dv1 = 0.f;
    for (int r = 0; r < NROWC; ++r) {
      const float t = tr[r];
      if (!(t >= MP::zmin && t <= MP::zmax)) continue;   // 깊이 없음
      const float pz = o[2] - yn[r] * t;
      if (((vert >> r) & 1u) && pz >= MP::band_lo && pz <= MP::match_hi) {
        dv0 = dv0 == 0.f ? t : minf(dv0, t);
        dv1 = maxf(dv1, t);
      }
      if (pz > MP::band_hi) continue;
      const float rr2 = t * t * (1.f + xn * xn);
      if (pz < MP::band_lo) {
        if (rr2 > floor_r) { floor_r = rr2; floor_t = t; }
      } else if (rr2 < best) {
        best = rr2; hit_t = t;
      }
    }
    sh.colt[col] = best < kInf ? 1 : floor_r > 0.f ? 2 : 0;
    sh.colt_t[col] = best < kInf ? hit_t : floor_r > 0.f ? floor_t : 0.f;
    sh.cold0[col] = dv0;
    sh.cold1[col] = dv1;
  }
  for (int p = tid; p < N_PRIM; p += nt) sh.pcand[p] = det_prefilter(m, o, c, s, k, p, sh.geo[p]) ? 1 : 0;
}

// ---- 3. 자세 보정 + 검출 + 물체 기억 ---------------------------------------------------------------------------------------
// 순서가 있는 일(난수 뽑기, 새 칸 고르기)만 스레드 0 이 하고, 판정·짝 열쇠·최솟값 찾기·부재 확인·완성도는 스레드가 나눠 한다.
// 나눈 일은 서로 다른 칸만 쓰고, 최솟값은 (열쇠, 번호) 순으로 골라 스레드 수와 무관하게 같은 답이다(CPU nt = 1 과 비트 같음).
// 3-0(모든 스레드): 맞은 열 수 부분합, 후보 물체(det_prefilter 통과)의 보임 점만 쏜다
DEV void obj_pre(const MapCore& m, Scratch& sh, int tid, int nt) {
  int nh = 0;
  for (int col = tid; col < NCOL; col += nt) nh += sh.colt[col] == 1;
  put_part(sh, tid, nh);
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  int ncand = 0;
  for (int p = 0; p < N_PRIM; ++p) ncand += sh.pcand[p];
  for (int i = tid; i < ncand * NPT; i += nt) {
    int p = 0;
    for (int j = i / NPT; ; ++p) if (sh.pcand[p] && j-- == 0) break;   // (i / NPT) 번째 후보
    if (vis_point(m, o, c, s, sh.geo[p], p, i % NPT)) or_bits(&sh.vism[p], 1u << (i % NPT));   // vism 은 phase_cast 가 비움
  }
}

// 놓침 확률: 카메라–물체 중심 거리 계단 (MP 참고)
DEV float p_miss_at(float d) { return d < MP::miss_d1 ? MP::p_miss_near : d < MP::miss_d2 ? MP::p_miss_mid : MP::p_miss_far; }

// 검출 하나를 지도에: 참 몸 좌표의 앞·옆·위(fwd, left, up, 수평 거리 rh)에 깊이·옆·높이 잡음을 넣고, 본 순간의 믿는 자세로 세계에 놓는다
// 검출 하나를 지도에: 보이는 면 중앙값(카메라 기준 앞·왼쪽·위 med)에 깊이·옆·높이 잡음을 넣고, 본 순간의 믿는 자세로 세계에 놓는다.
// 백분위 상자 중심 bc 도 같은 잡음으로 옮긴다(큰 물체의 합집합 상자용). 난수 순서는 전과 같음
DEV void put_det(Det& D, uint64_t& rng, const float med[3], const float bc[3], const float ext[3], float o2, float ex, float ey, float ec, float es) {
  const float fwd = med[0], left = med[1], rh = sqrtf(fwd * fwd + left * left);
  const float sig_d = MP::dn0 + MP::dn2 * fwd * fwd;
  const float gd = (MP::noise ? gauss(rng) : 0.f) * sig_d, gl = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n, gz = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n;
  // 몸 좌표의 수평 시선 단위(fwd, left)/rh 와 그 수직
  const float bu = fwd / rh, bv = left / rh;
  const float nf = bu * gd - bv * gl, nl = bv * gd + bu * gl;
  {
    const float bx = env::K::cam_x + fwd + nf, by = left + nl;
    D.pos[0] = ex + (ec * bx - es * by);
    D.pos[1] = ey + (es * bx + ec * by);
    D.pos[2] = o2 + med[2] + gz;
  }
  {
    const float bx = env::K::cam_x + bc[0] + nf, by = bc[1] + nl;
    D.bc[0] = ex + (ec * bx - es * by);
    D.bc[1] = ey + (es * bx + ec * by);
    D.bc[2] = o2 + bc[2] + gz;
  }
  for (int a = 0; a < 3; ++a) D.ext[a] = maxf(0.01f, ext[a] + (MP::noise ? gauss(rng) : 0.f) * MP::ext_n);
}

// 3a·3b(스레드 0): 자세 보정, 검출(난수 순서 그대로), 유령 자리
DEV void obj_detect(MapCore& m, Scratch& sh, const EnvView& e, int nt) {
  // 3a. keyframe 맞추기: 맞은 줄이 충분하고 제자리가 아니면 오차를 xy 는 kf_corr_xy, yaw 는 kf_corr_yaw 만큼 되돌림(slam2d keyframe 의 보정 흉내)
  const int n_hits = sum_part(sh, nt);
  const bool still = !m.first && m.vmax < MP::still_v && m.wmax < MP::still_w;
  if (!m.first && !still && n_hits >= MP::min_hits) {
    const float kxy = 1.f - MP::kf_corr_xy, kyaw = 1.f - MP::kf_corr_yaw;
    m.ex = e.x + (m.ex - e.x) * kxy;
    m.ey = e.y + (m.ey - e.y) * kxy;
    m.eyaw = wrap_pi(e.yaw + wrap_pi(m.eyaw - e.yaw) * kyaw);
  }
  float es, ec;
  sincosf_d(m.eyaw, &es, &ec);
  sh.ec = ec; sh.es = es;
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey;
  uint64_t rng = m.rng;   // 레지스터에(공유 메모리 sh.det 쓰기와 겹칠까 봐 매번 다시 읽지 않게)

  // 3b. 검출: 판정(det_prefilter + 보이는 점 비율) → 거리별 놓침 → 잡음(본 순간의 slam 오차를 물려받음) → 틀린 이름
  int nd = 0;
  for (int p = 0; p < N_PRIM; ++p) {
    if (!sh.pcand[p]) continue;
    const DetGeo& g = sh.geo[p];
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += (int)((sh.vism[p] >> q) & 1u);
    if (nv == 0) continue;
    if (g.af * ((float)nv / (float)NPT) < (float)MP::min_points) continue;   // 깊이 점 수(간격 1)
    if (MP::noise && rand01(rng) < p_miss_at(sqrtf(g.rh * g.rh + g.up * g.up))) continue;
    Det& D = sh.det[nd++];
    put_det(D, rng, g.med, g.bc, g.pe, o2, ex, ey, ec, es);
    int cls = m.prim[p].cls;
    if (MP::noise && rand01(rng) < MP::p_conf) cls = (cls + 1 + (int)(rand01(rng) * (float)(NCLS - 1))) % NCLS;
    D.cls = cls;
    D.src = p;
    D.score = 0.9f;
    if (dist3(D.pos, m.eef_m) < MP::hand_r) --nd;   // 손에 든 것 거르기(objmap: 점의 절반 이상이 팔 끝 hand_r 안 — 여기서는 중심으로)
  }
  // 유령 자리(가짜 물체): 중심이 깊이 범위·시야 안이면 p_ghost 로 검출된다. 가림은 보지 않는다(비침·잘못 분할 흉내)
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts;
  int nfp = 0;
  for (int gi = 0; gi < (MP::noise ? MP::n_ghost : 0); ++gi) {
    const Ghost& G = m.ghost[gi];
    const float rx = G.pos[0] - sh.to[0], ry = G.pos[1] - sh.to[1], up = G.pos[2] - o2;
    const float fwd = c * rx + s * ry, left = -s * rx + c * ry;
    if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) continue;
    if (absf(left / fwd) > k.tanh || absf(up / fwd) > k.tanv) continue;
    if (!(rand01(rng) < MP::p_ghost)) continue;
    const float ext[3] = {G.sz, G.sz, G.sz};
    Det& D = sh.det[nd++];
    const float rel[3] = {fwd, left, up};
    put_det(D, rng, rel, rel, ext, o2, ex, ey, ec, es);
    D.cls = G.cls;
    D.src = -1 - gi;
    D.score = 0.4f;
    if (dist3(D.pos, m.eef_m) < MP::hand_r) { --nd; continue; }
    ++nfp;
  }
  m.n_fp_total += nfp;
  m.n_kf_fp += nfp > 0;
  sh.nd = nd;
  m.rng = rng;
}

// 3c. 같은 물체(objmap.cpp 2): 같은 이름끼리, 중심 거리 < max(da_min, da_k·큰 쪽 크기) 또는 상자 틈 < da_gap. (틈 + 1e-3·거리) 순 1:1 탐욕
// 짝 열쇠(모든 스레드, 쌍마다)
DEV void obj_keys(const MapCore& m, Scratch& sh, int tid, int nt) {
  const int nd = sh.nd;
  for (int b = tid; b < KSLOT; b += nt) sh.hit[b] = 0;
  for (int a = tid; a < MAXDET; a += nt) sh.obs_to[a] = -1;
  for (int idx = tid; idx < nd * KSLOT; idx += nt) {
    const int a = idx / KSLOT, b = idx % KSLOT;
    float key = -1.f;
    const Slot& S = m.slot[b];
    const Det& D = sh.det[a];
    if (S.valid && !S.held && S.cls == D.cls) {   // 든 물체는 짝짓지 않음(objmap: held_by ≥ 0 건너뜀)
      const float ee = maxf(max3(D.ext), max3(S.ext));
      const float thr = maxf(MP::da_min, MP::da_k * ee);
      float d2 = 0.f, g2 = 0.f;
      for (int q = 0; q < 3; ++q) {
        const float dd = D.pos[q] - S.pos[q];
        d2 = d2 + dd * dd;
        const float olo = D.bc[q] - 0.5f * D.ext[q], ohi = D.bc[q] + 0.5f * D.ext[q];   // 관측 상자 = 백분위 상자
        const float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float gk = maxf(0.f, maxf(olo - mhi, mlo - ohi));
        g2 = g2 + gk * gk;
      }
      const float d = sqrtf(d2), gap = sqrtf(g2);
      if (d < thr || gap < MP::da_gap) key = gap + 1e-3f * d;
    }
    sh.key[idx] = key;
  }
}
// 탐욕 한 바퀴의 앞(모든 스레드): 남은 쌍 중 스레드 몫의 최소(같으면 앞 번호 = 원래 (관측, 칸) 이중 고리의 첫 최소)
DEV void obj_argmin_local(Scratch& sh, int tid, int nt) {
  const int n = sh.nd * KSLOT;
  int bi = -1;
  float bk = 0.f;
  for (int idx = tid; idx < n; idx += nt) {
    const float key = sh.key[idx];
    if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
    if (bi < 0 || key < bk) { bi = idx; bk = key; }
  }
  sh.part[tid] = bi;
  sh.partf[tid] = bk;
}
// 탐욕 한 바퀴의 뒤(스레드 0): (열쇠, 번호) 최소를 짝으로. 더 없으면 sh.more = 0
DEV void obj_argmin_pick(Scratch& sh, int nt) {
  int bi = -1;
  float bk = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int i = sh.part[t];
    if (i < 0) continue;
    const float k = sh.partf[t];
    if (bi < 0 || k < bk || (k == bk && i < bi)) { bi = i; bk = k; }
  }
  sh.more = bi >= 0;
  if (bi >= 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
}

#ifdef __CUDA_ARCH__
// 탐욕 짝짓기 GPU 판(워프 0 만): 레인마다 몫의 최소 → 셔플로 (열쇠, 번호) 최소 → 레인 0 이 짝 표시. 남은 쌍이 없으면 끝
__device__ __forceinline__ void obj_greedy_warp(Scratch& sh, int lane) {
  const int n = sh.nd * KSLOT;
  for (;;) {
    int bi = -1;
    float bk = 0.f;
    for (int idx = lane; idx < n; idx += 32) {
      const float key = sh.key[idx];
      if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
      if (bi < 0 || key < bk) { bi = idx; bk = key; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      const float ok = __shfl_xor_sync(0xffffffffu, bk, off);
      if (oi >= 0 && (bi < 0 || ok < bk || (ok == bk && oi < bi))) { bi = oi; bk = ok; }
    }
    if (bi < 0) break;   // 워프 안 모두 같은 값
    if (lane == 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
    __syncwarp();
  }
}
#endif

// 3d. 갱신(objmap.cpp 3). 짝지은 관측(모든 스레드, 관측마다 다른 칸)
DEV void obj_update_matched(MapCore& m, Scratch& sh, int tid, int nt, int bug) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;   // 음성 대조: 확정 규칙 끔(한 번 보이면 확정)
  const int t = m.t;
  for (int a = tid; a < sh.nd; a += nt) {
    if (sh.obs_to[a] < 0) continue;
    const Det& D = sh.det[a];
    Slot& S = m.slot[sh.obs_to[a]];
    if (S.last_kf != t) S.n_obs += 1;
    S.last_kf = t;
    const float w = (float)(S.n_obs < 20 ? S.n_obs : 20);
    const bool bigo = is_static(S.cls) || maxf(maxf(S.ext[0], S.ext[1]), maxf(D.ext[0], D.ext[1])) > MP::big;
    for (int q = 0; q < 3; ++q) {
      if (bigo) {   // 합집합, keyframe 마다 면마다 grow_max·한 변 max_ext 까지
        float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float olo = D.bc[q] - 0.5f * D.ext[q], ohi = D.bc[q] + 0.5f * D.ext[q];   // 관측 상자 = 백분위 상자
        const float lo = maxf(minf(mlo, olo), mlo - MP::grow_max);
        const float hi = minf(maxf(mhi, ohi), mhi + MP::grow_max);
        if (hi - lo <= MP::max_ext) { mlo = lo; mhi = hi; }
        S.pos[q] = 0.5f * (mlo + mhi);
        S.ext[q] = mhi - mlo;
      } else {
        S.pos[q] = (S.pos[q] * (w - 1.f) + D.pos[q]) / w;
        S.ext[q] = (S.ext[q] * (w - 1.f) + D.ext[q]) / w;
      }
    }
    S.score = maxf(S.score, D.score);
    S.last_seen = t;
    S.src = D.src;
    S.seen_len = m.plen; S.seen_rot = m.prot;
    S.misses = 0;
    if (S.state == S_GONE) S.state = S.moved ? S_MOVED : S_SEEN;
    if (!S.confirmed && S.n_obs >= confirm_n) S.confirmed = 1;
  }
}
// 안 맞은 관측(스레드 0, 관측 순서대로). 짝지은 칸은 hit 이라 여기서 보지 않으므로 위와 나눠 해도 원래 순서와 같다
DEV void obj_update_new(MapCore& m, Scratch& sh, int bug) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;
  const int t = m.t;
  for (int a = 0; a < sh.nd; ++a) {
    if (sh.obs_to[a] >= 0) continue;
    const Det& D = sh.det[a];
    // 같은 이름의 확정·사라짐 물체가 있으면 옮겨진 것으로 잇는다
    int mf = -1;
    float best = 1e9f;
    for (int b = 0; b < KSLOT; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || S.cls != D.cls || !S.confirmed || sh.hit[b] || S.state != S_GONE) continue;
      float d2 = 0.f;
      for (int q = 0; q < 3; ++q) { const float dd = D.pos[q] - S.pos[q]; d2 = d2 + dd * dd; }
      if (d2 < best) { best = d2; mf = b; }
    }
    if (mf >= 0) {
      Slot& S = m.slot[mf];
      const bool bigd = is_static(D.cls) || maxf(D.ext[0], D.ext[1]) > MP::big;   // 큰 것: 자리 = 상자 중심(근사판 칸은 상자를 자리 ± 크기/2 로 둠)
      for (int q = 0; q < 3; ++q) { S.pos[q] = bigd ? D.bc[q] : D.pos[q]; S.ext[q] = D.ext[q]; }
      S.moved = best > MP::moved_d * MP::moved_d ? 1 : S.moved;   // 짝 문턱(≥ 0.30 m) 밖이라 사실상 항상 옮겨짐
      S.state = S.moved ? S_MOVED : S_SEEN;
      S.misses = 0;
      S.last_seen = t;
      S.last_kf = t;
      S.n_obs += 1;
      S.src = D.src;
      S.seen_len = m.plen; S.seen_rot = m.prot;
      sh.hit[mf] = 1;
      continue;
    }
    int fs = -1;
    for (int b = 0; b < KSLOT && fs < 0; ++b) if (!m.slot[b].valid) fs = b;
    if (fs < 0) { m.n_dropped += 1; continue; }   // 칸이 다 참(가정: 버림)
    Slot& S = m.slot[fs];
    S.valid = 1; S.id = m.next_id++; S.cls = D.cls;
    const bool bigd = is_static(D.cls) || maxf(D.ext[0], D.ext[1]) > MP::big;   // 큰 것: 자리 = 백분위 상자 중심(다음 합집합과 같은 상자)
    for (int q = 0; q < 3; ++q) { S.pos[q] = bigd ? D.bc[q] : D.pos[q]; S.first_pos[q] = S.pos[q]; S.ext[q] = D.ext[q]; }
    S.n_obs = 1; S.last_seen = t; S.last_kf = t; S.score = D.score;
    S.held = 0; S.src = D.src; S.seen_len = m.plen; S.seen_rot = m.prot;
    S.confirmed = confirm_n <= 1 ? 1 : 0;
    S.state = S_SEEN; S.moved = 0; S.misses = 0; S.first_miss = 0;
    sh.hit[fs] = 1;
  }
}

// 3e. 부재 확인(objmap.cpp 4) + 3f. 오래된 후보 버리기. 칸마다 따로(모든 스레드, 칸마다 한 스레드)
//     확정·이번에 안 맞음·사라짐 아님·고정 종류 아님·작은 것. 믿는 자세로 화소에 투영하고
//     그 화소의 깊이(참 장면)가 물체보다 occl 넘게 가까우면 가려짐. 아니면 놓침 +1
DEV void obj_absence(MapCore& m, const Scratch& sh, const EnvView& e, int tid, int nt) {
  const Cam k = cam_consts();
  const float ec = sh.ec, es = sh.es;
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  const int t = m.t;
  for (int b = tid; b < KSLOT; b += nt) {
    Slot& S = m.slot[b];
    // 든 것·팔 끝 hand_r + 0.1 안은 판단하지 않음(objmap.cpp 4)
    if (S.valid && S.confirmed && !S.held && !sh.hit[b] && S.state != S_GONE && !is_static(S.cls) && !(max3(S.ext) > MP::big) &&
        !(dist3(S.pos, m.eef_m) < MP::hand_r + 0.1f)) {
      const float rx = S.pos[0] - (m.ex + ec * env::K::cam_x), ry = S.pos[1] - (m.ey + es * env::K::cam_x);
      const float zc = ec * rx + es * ry, left = -es * rx + ec * ry, up = S.pos[2] - env::K::cam_z;
      if (!(zc < 0.3f || zc > MP::ozmax)) {
        const float xn = -left / zc, yn = -up / zc;
        const float u = k.fx * xn + k.cxp, v = k.fx * yn + k.cyp;
        const float size_px = k.fx * max3(S.ext) / zc;
        if (!(size_px < (float)MP::min_px || u < 2.f || v < 2.f || u >= (float)(MP::img_w - 2) || v >= (float)(MP::img_h - 2))) {
          float d[3];
          pix_dir(c, s, xn, yn, d);
          const float tz = cast(m, o, d);
          if (tz >= MP::zmin && tz <= MP::zmax && !(tz < zc - MP::occl)) {   // 깊이 있음, 가려지지 않음
            if (S.misses++ == 0) S.first_miss = t;
            if (S.misses >= MP::gone_misses && t - S.first_miss >= MP::gone_min_steps) S.state = S_GONE;
          }
        }
      }
    }
    if (S.valid && !S.confirmed && t - S.last_seen > MP::prune_steps) S.valid = 0;
  }
}

// 3g. 완성도(모든 스레드, 참 물체마다): 같은 이름의 확정(사라짐 아님) 칸이 짝 문턱 안에 있나 → sh.found[p]
DEV void obj_complete(const MapCore& m, Scratch& sh, int tid, int nt) {
  for (int p = tid; p < N_PRIM; p += nt) {
    const Prim& P = m.prim[p];
    const float cx = 0.5f * (P.lo[0] + P.hi[0]), cy = 0.5f * (P.lo[1] + P.hi[1]);
    float ext[3];
    for (int a = 0; a < 3; ++a) ext[a] = P.hi[a] - P.lo[a];
    const float thr = maxf(MP::da_min, MP::da_k * max3(ext));
    int found = 0;
    for (int b = 0; b < KSLOT && !found; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || !S.confirmed || S.state == S_GONE || S.cls != P.cls) continue;
      const float dx = S.pos[0] - cx, dy = S.pos[1] - cy;
      found = dx * dx + dy * dy < thr * thr;
    }
    sh.found[p] = found;
  }
}

// ---- 4. 격자 표시(모든 스레드): 믿는 자세로 줄을 격자에. 정수 Bresenham, 끝 칸 제외 → 맞음 끝은 hit, 빈 광선 끝은 miss ----------
DEV bool in_win(int ix, int iy) { return ix - GX0 >= 0 && iy - GX0 >= 0 && ix - GX0 < GW && iy - GX0 < GW; }
// 창 안 칸 표시. GPU 는 낱말이 바뀔 때 그 낱말의 로그 오즈 64 B·본 칸 낱말을 L2 로 미리 당긴다(격자 갱신의 DRAM 대기를 줄임, 값은 안 바뀜)
DEV void mark_in(uint32_t* bits, int ix, int iy, const int16_t* L, const uint32_t* seen, int& lastw) {
  const int idx = (iy - GX0) * GW + (ix - GX0), w = idx >> 5;
  or_bits(&bits[w], 1u << (idx & 31));
#ifdef __CUDA_ARCH__
  if (w != lastw) {
    lastw = w;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(L + (size_t)w * 32));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(seen + w));
  }
#else
  (void)L; (void)seen; (void)lastw;
#endif
}
DEV void mark(uint32_t* bits, int ix, int iy) {
  const int lx = ix - GX0, ly = iy - GX0;
  if (lx < 0 || ly < 0 || lx >= GW || ly >= GW) return;
  const int idx = ly * GW + lx;
  or_bits(&bits[idx >> 5], 1u << (idx & 31));
}
// 열 끝은 hitb·missb 와 같은 자리(공용체)에 있으므로 먼저 레지스터로 옮기고, 동기 뒤 비트표를 비우고, 다시 동기 뒤 표시한다
#ifdef __CUDA_ARCH__
constexpr int MARK_CPT = (NCOL + NT - 1) / NT;   // 스레드마다 열 수(GPU)
#else
constexpr int MARK_CPT = NCOL;                   // CPU(nt 1)
#endif
DEV void mark_cell(Scratch& sh, uint32_t* bits, float wx, float wy) { mark(bits, (int)floorf(wx * INV_RES), (int)floorf(wy * INV_RES)); }
// GPU: 점유 비트 2 KB 를 Scratch 앞(물체 단계가 끝난 자리, scr_occ)으로 비동기 복사(cp.async, L2 경유). 격자 표시 동안 오고,
// phase_mark 끝의 기다림 + 블록 동기 뒤 격자 갱신이 바뀐 낱말을 고쳐 쓴다. 벽 단계는 이 사본을 쓴다(전역 다시 읽기 없음)
DEV void occ_copy_issue(Scratch& sh, const uint32_t* occ, int tid, int nt) {
#ifdef __CUDA_ARCH__
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(scr_occ(sh));
  for (int k = tid; k < NWORD / 4; k += nt)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(base + 16u * (uint32_t)k), "l"(occ + 4 * k) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
#else
  (void)sh; (void)occ; (void)tid; (void)nt;
#endif
}
DEV void occ_copy_wait() {
#ifdef __CUDA_ARCH__
  asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}
template <class Sync>
DEV void phase_mark(const MapCore& m, Scratch& sh, const int16_t* L, const uint32_t* seen, const uint32_t* occ, int tid, int nt, const Sync& sync) {
  occ_copy_issue(sh, occ, tid, nt);
  const float ec = sh.ec, es = sh.es;
  int8_t ty[MARK_CPT];
  float tt[MARK_CPT], t0[MARK_CPT], t1[MARK_CPT];
#pragma unroll
  for (int j = 0; j < MARK_CPT; ++j) {
    const int col = tid + j * nt;
    ty[j] = col < NCOL ? sh.colt[col] : (int8_t)0;
    tt[j] = col < NCOL ? sh.colt_t[col] : 0.f;
    t0[j] = col < NCOL ? sh.cold0[col] : 0.f;
    t1[j] = col < NCOL ? sh.cold1[col] : 0.f;
  }
  sync();
  for (int w = tid; w < NWORD; w += nt) { sh.hitb[w] = 0u; sh.missb[w] = 0u; }
  sync();
  const Cam k = cam_consts();
  const float ox = m.ex + ec * env::K::cam_x, oy = m.ey + es * env::K::cam_x;
  const int cx = (int)floorf(ox * INV_RES), cy = (int)floorf(oy * INV_RES);
#pragma unroll
  for (int j = 0; j < MARK_CPT; ++j) {
    const int col = tid + j * nt;
    if (col >= NCOL) continue;
    const float xn = ((float)(2 * col + 1 - NCOL) / (float)NCOL) * k.tanh;
    for (int d = 0; d < 2; ++d) {   // 맞추기 점(수직면): 맞음만, 광선 없음(grid.cpp: 맞추기 점 먼저)
      const float t = d ? t1[j] : t0[j];
      if (t == 0.f || (d && t1[j] == t0[j])) continue;
      const float bx = env::K::cam_x + t, by = -xn * t;
      mark_cell(sh, sh.hitb, m.ex + (ec * bx - es * by), m.ey + (es * bx + ec * by));
    }
    if (ty[j] == 0) continue;
    const float bx = env::K::cam_x + tt[j], by = -xn * tt[j];
    const float wx = m.ex + (ec * bx - es * by), wy = m.ey + (es * bx + ec * by);
    const int x1 = (int)floorf(wx * INV_RES), y1 = (int)floorf(wy * INV_RES);
    int x = cx, y = cy;
    const int dx = x1 > x ? x1 - x : x - x1, dy = -(y1 > y ? y1 - y : y - y1), sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1;
    int err = dx + dy;
    if (in_win(cx, cy) && in_win(x1, y1)) {   // 두 끝이 창 안이면 지나는 칸(두 끝의 상자 안)도 모두 창 안: 칸마다 검사 안 함
      int lastw = -1;
      while (!(x == x1 && y == y1)) {
        mark_in(sh.missb, x, y, L, seen, lastw);
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x += sx; }
        if (e2 <= dx) { err += dx; y += sy; }
      }
      mark_in(ty[j] == 1 ? sh.hitb : sh.missb, x1, y1, L, seen, lastw);
      continue;
    }
    while (!(x == x1 && y == y1)) {
      mark(sh.missb, x, y);
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
    mark(ty[j] == 1 ? sh.hitb : sh.missb, x1, y1);
  }
  occ_copy_wait();   // 이 스레드의 복사 끝(다음 블록 동기 뒤 모두에게 보임)
}

// ---- 5. 격자 갱신(모든 스레드, 낱말마다 한 스레드): 맞음 우선, 정수 덧셈 + 자르기, 본 적 표시, 방 안 새로 본 칸 수 ------------
// 점유 비트(occ, 벽 상태용): 로그 오즈 ≥ q_occ(= export8 ≥ 65 %). 바뀐 칸만 다시 씀. 새로 본 방 칸은 방마다 센다(방 1 | 방 2 << 16 은 part2)
DEV void count_new_seen(const MapCore& m, int idx, int& cnt, int& cnt2) {
  const float cxw = ((float)(idx % GW + GX0) + 0.5f) * RES, cyw = ((float)(idx / GW + GX0) + 0.5f) * RES;
  const int k = room_of(m, cxw, cyw);
  cnt += k >= 0;
  cnt2 += k == 1 ? 1 : k == 2 ? (1 << 16) : 0;
}
DEV void phase_apply(const MapCore& m, Scratch& sh, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt) {
  int cnt = 0, cnt2 = 0;
#ifdef __CUDA_ARCH__
  // GPU: 워프가 낱말 32 개씩 훑어 표시된 낱말만 골라(ballot), 그 낱말의 칸 32 개를 레인 32 개가 맡는다(로그 오즈 64 B 를 한 번에 읽음).
  // 낱말 4 개까지 읽기를 먼저 내고 계산한다(DRAM 대기를 겹침). 칸마다 연산은 아래 CPU 고리와 같다.
  const int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
  const uint32_t bit = 1u << lane;
  for (int base = warp * 32; base < NWORD; base += nwarp * 32) {
    uint32_t mask = __ballot_sync(0xffffffffu, (sh.hitb[base + lane] | sh.missb[base + lane]) != 0u);
#ifdef MAP_PROF
    if (lane == 0) atomicAdd(&g_prof[P_NSEC + 2], (unsigned long long)__popc(mask));
#endif
    while (mask) {
      int wv[4], nb = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wv[j] = -1;
        if (mask) { wv[j] = base + __ffs(mask) - 1; mask &= mask - 1; ++nb; }
      }
      uint32_t hh[4], aa[4], oo[4], oc[4];
      int lv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j >= nb) continue;
        const int w = wv[j];
        hh[j] = sh.hitb[w];
        aa[j] = hh[j] | (sh.missb[w] & ~hh[j]);
        oo[j] = seen[w];
        oc[j] = scr_occ(sh)[w];   // 점유 비트는 공유 사본에서(phase_mark 의 복사)
        lv[j] = (aa[j] & bit) ? (int)L[(size_t)w * 32 + lane] : 0;
      }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j >= nb) continue;
        const int w = wv[j];
        int Lv = lv[j];
        if (aa[j] & bit) {
          Lv = (hh[j] & bit) ? (Lv + MP::q_hit < MP::q_max ? Lv + MP::q_hit : MP::q_max) : (Lv + MP::q_miss > MP::q_min ? Lv + MP::q_miss : MP::q_min);
          L[(size_t)w * 32 + lane] = (int16_t)Lv;
          if (!(oo[j] & bit)) count_new_seen(m, w * 32 + lane, cnt, cnt2);   // 새로 본 칸 중 방 안
        }
        const uint32_t ob = __ballot_sync(0xffffffffu, (aa[j] & bit) && Lv >= MP::q_occ);
        if (lane == 0) {
          seen[w] = oo[j] | aa[j];
          const uint32_t no = (oc[j] & ~aa[j]) | ob;
          occ[w] = no;
          scr_occ(sh)[w] = no;
          if (no != oc[j]) sh.occ_chg = 1;
        }
      }
    }
  }
  put_part(sh, tid, cnt);
  put_part2(sh, tid, cnt2);
#else
  // CPU(참조판): 낱말마다 칸을 차례로. 맞음 우선, 정수 덧셈 + 자르기
  for (int w = tid; w < NWORD; w += nt) {
    const uint32_t h = sh.hitb[w], ms = sh.missb[w] & ~h, any = h | ms;
    if (!any) continue;
    const uint32_t old = seen[w], nw = any & ~old;
    seen[w] = old | any;
    uint32_t ob = 0u;
    for (int b = 0; b < 32; ++b) {
      const uint32_t bit = 1u << b;
      if (!(any & bit)) continue;
      const int idx = w * 32 + b;
      int Lv = L[idx];
      Lv = (h & bit) ? (Lv + MP::q_hit < MP::q_max ? Lv + MP::q_hit : MP::q_max) : (Lv + MP::q_miss > MP::q_min ? Lv + MP::q_miss : MP::q_min);
      L[idx] = (int16_t)Lv;
      if (Lv >= MP::q_occ) ob |= bit;
      if (nw & bit) count_new_seen(m, idx, cnt, cnt2);   // 새로 본 칸 중 방 안
    }
    const uint32_t no = (occ[w] & ~any) | ob;
    if (no != occ[w]) sh.occ_chg = 1;
    occ[w] = no;
  }
  put_part(sh, tid, cnt);
  put_part2(sh, tid, cnt2);
#endif
}

DEV void phase_finish(MapCore& m, const Scratch& sh, const EnvView& e, int nt) {
  const int s = sum_part(sh, nt);
  int n_obj = 0;
  for (int p = 0; p < N_PRIM; ++p) n_obj += sh.found[p];
  m.n_obj_conf = n_obj;
  m.n_task_conf = sh.found[0];
  m.n_seen_room += s;
  {  // 방마다 본 칸 → 드러냄: 본 넓이 ≥ min_room_m2 이고 방 칸의 room_reveal 이상(한 번 드러나면 그대로)
    const int s2 = sum_part2(sh, nt), r1 = s2 & 0xffff, r2 = s2 >> 16;
    m.rseen[0] += s - r1 - r2;
    m.rseen[1] += r1;
    m.rseen[2] += r2;
    for (int k = 0; k < m.n_room; ++k)
      if (!((m.rrev >> k) & 1) && (float)m.rseen[k] * (RES * RES) >= MP::min_room_m2 && (float)m.rseen[k] >= MP::room_reveal * (float)m.rcells[k])
        m.rrev |= 1 << k;
  }
  {
    int cm = 0;
    for (int b = 0; b < KSLOT; ++b) cm |= (m.slot[b].valid && m.slot[b].confirmed) ? (1 << b) : 0;
    m.conf_mask = cm;
  }
  m.n_kf += 1;
  m.n_kf_total += 1;
  m.first = 0;
  m.since = 0;
  m.lx = e.x; m.ly = e.y; m.lyaw = e.yaw;   // 움직임 거르기는 참 자세로
  m.vmax = 0.f; m.wmax = 0.f;
}

DEV void write_metrics(const MapCore& m, const EnvView& e, float* met, int N, int i) {
  const float dx = m.ex - e.x, dy = m.ey - e.y;
  met[M_TASK * N + i] = (float)m.n_task_conf;
  met[M_OBJ * N + i] = (float)m.n_obj_conf / (float)N_PRIM;
  met[M_SEEN * N + i] = m.room_cells > 0 ? (float)m.n_seen_room / (float)m.room_cells : 0.f;
  met[M_ERR_XY * N + i] = sqrtf(dx * dx + dy * dy);
  met[M_ERR_YAW * N + i] = absf(wrap_pi(m.eyaw - e.yaw));
  met[M_KF * N + i] = (float)m.kf_flag;
  met[M_ROOM * N + i] = (float)popc32((uint32_t)m.rrev) / (float)m.n_room;
  met[M_INIT * N + i] = (float)(m.init_conf | (m.init_goal << 4) | (m.init_stage << 5));
}

// ---- 6. 벽 선분(walls.cpp wallSegments, 판 하나 = 블록 하나) ---------------------------------------------------------------
// 점유 비트(occ)에서 바닥 위 확정 물체 상자 + 0.1 m(무시 영역, capi.cpp refreshWalls)를 지우고, 가로 벽(행마다 구간)과 세로 벽(열마다
// 구간)을 따로 찾는다. 행 순서는 walls.cpp 와 같이 "파이썬 행"(행 0 = 가장 큰 y). 구간 ≥ wall_min_run 칸만, 위 행 묶음과 겹침
// ≥ 0.6·짧은 쪽(정수로 5·ov ≥ 3·min — 0.6·m 의 double 반올림과 같은 판정)이면 이어 붙이고, 두께가 wall_max_rows 넘는 묶음은 버린다.
// 선분 순서도 walls.cpp 와 같다(묶음이 끝나는 행 순, 그 안은 위 행 구간 순 → 가로 다음 세로). 행마다 구간 찾기는 스레드가 나눠 하고,
// 묶기(행을 차례로)는 가로 = 스레드 0, 세로 = 스레드 32(CPU 는 스레드 0 이 둘 다).
// 선분은 반 칸 정수 좌표(창 원점 기준): 세계 x = (X2 / 2 + GX0)·RES. 장치 배열 segs[판][가로|세로][MAXSEG][4] (int16)
struct WallScratch {
  alignas(16) uint32_t occ[NWORD];   // 행 y(아래 → 위) 우선, 무시 영역 지움. Scratch 앞의 점유 비트 사본(scr_occ)과 같은 자리
  uint32_t occT[NWORD];         // 열 x 마다 4 낱말, 비트 = 파이썬 행 iy(위 → 아래)
  uint16_t run[2][NRUN];        // 구간 a | b << 8 (칸)
  uint16_t off[2][GW + 1];      // 행마다 구간 시작 번호(NRUN 에서 자름)
  uint8_t cnt[2][GW];           // 행마다 구간 수
  uint8_t act[2][2][NACT][3];   // 묶기: 이어지는 묶음 r0, x0, x1 (두 벌 번갈아)
  uint32_t rect[KSLOT];         // 무시 영역 칸 cx0 | cx1 << 8 | cy0 << 16 | cy1 << 24
  uint32_t nz[2][4];            // 구간이 있는 행 비트
  int nrect, skip;
  int ovf[2];
};
static_assert(sizeof(WallScratch) <= sizeof(Scratch), "WallScratch shares the keyframe block's Scratch");

// 128 칸 한 행(낱말 4)의 이어진 1 구간 중 길이 ≥ wall_min_run 인 것(앞에서부터). out 이 있으면 cap 개까지 쓴다. 돌려주는 값 = 개수
DEV int row_runs(const uint32_t* w, uint16_t* out, int cap) {
  int n = 0, x = 0;
  while (x < GW) {
    int k = x >> 5;
    uint32_t cur = w[k] & (~0u << (x & 31));
    while (!cur && ++k < 4) cur = w[k];
    if (k >= 4) break;
    const int a = k * 32 + ctz32(cur);
    k = a >> 5;
    uint32_t inv = ~w[k] & (~0u << (a & 31));
    while (!inv && ++k < 4) inv = ~w[k];
    const int b = (k >= 4 ? GW : k * 32 + ctz32(inv)) - 1;
    if (b - a + 1 >= MP::wall_min_run) {
      if (out && n < cap) out[n] = (uint16_t)(a | (b << 8));
      ++n;
    }
    x = b + 1;
  }
  return n;
}
// 같은 결과를 비트 연산으로(GPU): 구간 시작 st = 1 이고 앞 칸이 0, 길이 ≥ 5 의 시작 = st 이고 뒤 4 칸도 1(x5). 시작마다 끝은 다음 0 칸
DEV int row_runs_bits(const uint32_t* w, uint16_t* out, int cap) {
  static_assert(MP::wall_min_run == 5, "x5 mask below assumes runs of 5");
  uint32_t v[4];
  for (int k = 0; k < 4; ++k) v[k] = w[k];
  uint32_t sel[4];
  int n = 0;
  for (int k = 0; k < 4; ++k) {
    const uint32_t prev = (v[k] << 1) | (k ? v[k - 1] >> 31 : 0u);   // 비트 x = 칸 x−1
    const uint32_t nx = k < 3 ? v[k + 1] : 0u;
    uint32_t x5 = v[k];
    for (int j = 1; j <= 4; ++j) x5 &= (v[k] >> j) | (nx << (32 - j));   // 비트 x = 칸 x+j
    sel[k] = v[k] & ~prev & x5;
    n += popc32(sel[k]);
  }
  if (!out) return n;
  int q = 0;
  for (int k = 0; k < 4 && q < cap; ++k) {
    uint32_t b = sel[k];
    while (b && q < cap) {
      const int a = k * 32 + ctz32(b);
      b &= b - 1u;
      int kk = a >> 5;
      uint32_t inv = ~v[kk] & (~0u << (a & 31));
      while (!inv && ++kk < 4) inv = ~v[kk];
      const int e = (kk >= 4 ? GW : kk * 32 + ctz32(inv)) - 1;
      out[q++] = (uint16_t)(a | (e << 8));
    }
  }
  return n;
}
DEV int row_runs_dev(const uint32_t* w, uint16_t* out, int cap) {   // GPU 는 비트판, CPU 참조판은 위 고리판(같은 결과를 map_verify 가 비교)
#ifdef __CUDA_ARCH__
  return row_runs_bits(w, out, cap);
#else
  return row_runs(w, out, cap);
#endif
}
// 묶기 한 방향(P 0 가로, 1 세로) — walls.cpp groupRuns + emit 와 같은 순서. 선분을 segs(MAXSEG 칸)에 쓰고 개수를 돌려준다
DEV int wall_group(WallScratch& ws, int P, int16_t* segs) {
  uint8_t(*A)[3] = ws.act[P][0];
  uint8_t(*B)[3] = ws.act[P][1];
  int na = 0, ns = 0;
  auto emit = [&](int r0, int r1, int x0, int x1) {
    if (r1 - r0 + 1 > MP::wall_max_rows) return;
    if (ns >= MAXSEG) { ws.ovf[P] += 1; return; }
    const int rc2 = r0 + r1;   // 가운데 행 × 2
    int16_t* o = segs + 4 * ns;
    if (P == 0) {   // 행 = 파이썬 행, 끝 = 열 x0, x1
      o[0] = (int16_t)(2 * x0 + 1); o[1] = (int16_t)(2 * GW - 1 - rc2);
      o[2] = (int16_t)(2 * x1 + 1); o[3] = (int16_t)(2 * GW - 1 - rc2);
    } else {        // 행 = 열 x, 끝 = 파이썬 행 x0, x1
      o[0] = (int16_t)(rc2 + 1); o[1] = (int16_t)(2 * GW - 1 - 2 * x0);
      o[2] = (int16_t)(rc2 + 1); o[3] = (int16_t)(2 * GW - 1 - 2 * x1);
    }
    ++ns;
  };
  // 구간이 있는 행만 차례로. 빈 행을 건너뛰면 그 앞의 묶음은 모두 끝난 것(walls.cpp: 빈 행에서 active 가 모두 done 으로)
  int prev = -2;
  for (int wd = 0; wd < 4; ++wd) {
    uint32_t bits = ws.nz[P][wd];
    while (bits) {
      const int r = wd * 32 + ctz32(bits);
      bits &= bits - 1u;
      if (r != prev + 1) {
        for (int g = 0; g < na; ++g) emit(A[g][0], prev, A[g][1], A[g][2]);
        na = 0;
      }
      int nn = 0;
      uint32_t used = 0u;
      for (int i = ws.off[P][r]; i < ws.off[P][r + 1]; ++i) {
        const int a = ws.run[P][i] & 0xff, b = ws.run[P][i] >> 8;
        int hit = -1;
        for (int g = 0; g < na; ++g) {
          if ((used >> g) & 1u) continue;
          const int gx0 = A[g][1], gx1 = A[g][2];
          const int ov = (b < gx1 ? b : gx1) - (a > gx0 ? a : gx0) + 1;
          const int la = b - a + 1, lg = gx1 - gx0 + 1;
          if (5 * ov >= 3 * (la < lg ? la : lg)) { hit = g; break; }
        }
        if (hit < 0) {
          B[nn][0] = (uint8_t)r; B[nn][1] = (uint8_t)a; B[nn][2] = (uint8_t)b;
        } else {
          used |= 1u << hit;
          B[nn][0] = A[hit][0];
          B[nn][1] = (uint8_t)(a < A[hit][1] ? a : A[hit][1]);
          B[nn][2] = (uint8_t)(b > A[hit][2] ? b : A[hit][2]);
        }
        ++nn;
      }
      for (int g = 0; g < na; ++g)
        if (!((used >> g) & 1u)) emit(A[g][0], r - 1, A[g][1], A[g][2]);
      uint8_t(*T)[3] = A; A = B; B = T;
      na = nn;
      prev = r;
    }
  }
  for (int g = 0; g < na; ++g) emit(A[g][0], prev, A[g][1], A[g][2]);
  return ns;
}
DEV const uint32_t* wall_row(const WallScratch& ws, int P, int r) { return P == 0 ? &ws.occ[(GW - 1 - r) * 4] : &ws.occT[r * 4]; }

// 물체 칸 하나의 무시 영역(창 칸, 묶음 낱말). 없으면 0xffffffff
DEV uint32_t wall_rect(const Slot& S) {
  constexpr float OX = (float)GX0 * RES;   // 창 왼쪽 아래 모서리(x·y 같음)
  if (!S.valid || !S.confirmed || S.held || S.state == S_GONE) return 0xffffffffu;
  float lo[3], hi[3];
  for (int a = 0; a < 3; ++a) { lo[a] = S.pos[a] - 0.5f * S.ext[a]; hi[a] = S.pos[a] + 0.5f * S.ext[a]; }
  if (lo[2] > MP::wall_ign_z) return 0xffffffffu;
  if (hi[0] - lo[0] > MP::wall_ign_max || hi[1] - lo[1] > MP::wall_ign_max) return 0xffffffffu;
  int cx0 = (int)floorf((lo[0] - MP::wall_ign_m - OX) * INV_RES), cx1 = (int)floorf((hi[0] + MP::wall_ign_m - OX) * INV_RES);
  int cy0 = (int)floorf((lo[1] - MP::wall_ign_m - OX) * INV_RES), cy1 = (int)floorf((hi[1] + MP::wall_ign_m - OX) * INV_RES);
  cx0 = cx0 < 0 ? 0 : cx0; cy0 = cy0 < 0 ? 0 : cy0;
  cx1 = cx1 > GW - 1 ? GW - 1 : cx1; cy1 = cy1 > GW - 1 ? GW - 1 : cy1;
  if (cx0 > cx1 || cy0 > cy1) return 0xffffffffu;
  return (uint32_t)cx0 | ((uint32_t)cx1 << 8) | ((uint32_t)cy0 << 16) | ((uint32_t)cy1 << 24);
}
// occ_ready: ws.occ 에 이미 점유 비트가 있음(GPU keyframe: phase_mark 의 복사 + phase_apply 의 고침). 아니면 전역에서 읽는다
template <class Sync>
DEV void phase_walls(MapCore& m, WallScratch& ws, const uint32_t* occ_g, int16_t* segs_g, int occ_chg, int occ_ready, int tid, int nt, const Sync& sync) {
  // 무시 영역: 확정·안 든·사라짐 아님·바닥이 0.4 m 아래·한 변 ≤ 5 m 물체 상자 + 0.1 m(칸 순서로 모음).
  // 점유 비트도 무시 영역도 지난 계산과 같으면 선분이 같다 → 건너뜀(결과 같음)
#ifdef __CUDA_ARCH__
  if (!occ_ready)
#else
  (void)occ_ready;
#endif
    for (int w = tid; w < NWORD; w += nt) ws.occ[w] = occ_g[w];   // 건너뛸지 모르지만 미리 읽음(지연을 겹침)
#ifdef __CUDA_ARCH__
  static_assert(KSLOT <= 32, "rect gather uses one warp");
  if (tid < 32) {
    const uint32_t R = tid < KSLOT ? wall_rect(m.slot[tid]) : 0xffffffffu;
    const uint32_t bal = __ballot_sync(0xffffffffu, R != 0xffffffffu);
    const int n = __popc(bal), pos = __popc(bal & ((1u << tid) - 1u));
    const bool diff = R != 0xffffffffu && (pos >= m.wnrect || m.wrect[pos] != R);
    const bool any = __any_sync(0xffffffffu, diff);
    if (R != 0xffffffffu) { ws.rect[pos] = R; m.wrect[pos] = R; }
    if (tid == 0) {
      ws.skip = (!occ_chg && !any && n == m.wnrect) ? 1 : 0;
      ws.nrect = n;
      m.wnrect = n;
      ws.ovf[0] = 0; ws.ovf[1] = 0;
    }
  }
#else
  if (tid == 0) {
    int n = 0;
    bool same = !occ_chg;
    for (int b = 0; b < KSLOT; ++b) {
      const uint32_t R = wall_rect(m.slot[b]);
      if (R == 0xffffffffu) continue;
      same = same && n < m.wnrect && m.wrect[n] == R;
      ws.rect[n] = R;
      m.wrect[n] = R;
      ++n;
    }
    ws.skip = (same && n == m.wnrect) ? 1 : 0;
    ws.nrect = n;
    m.wnrect = n;
    ws.ovf[0] = 0; ws.ovf[1] = 0;
  }
#endif
  sync();
  PROF_MARK(PW1);
  if (ws.skip) return;
  for (int r = 0; r < ws.nrect; ++r) {   // 무시 영역 지우기: 영역마다 덮는 낱말을 스레드가 나눠(AND 라 순서 무관)
    const uint32_t R = ws.rect[r];
    const int cx0 = R & 0xff, cx1 = (R >> 8) & 0xff, cy0 = (R >> 16) & 0xff, cy1 = R >> 24;
    const int k0 = cx0 >> 5, nk = (cx1 >> 5) - k0 + 1;
    for (int j = tid; j < (cy1 - cy0 + 1) * nk; j += nt) {
      const int y = cy0 + j / nk, k = k0 + j % nk, x0 = k * 32;
      const int a = cx0 > x0 ? cx0 - x0 : 0, b = cx1 < x0 + 31 ? cx1 - x0 : 31;
      const uint32_t mk = (b - a == 31 ? ~0u : ((1u << (b - a + 1)) - 1u)) << a;
#ifdef __CUDA_ARCH__
      atomicAnd(&ws.occ[y * 4 + k], ~mk);
#else
      ws.occ[y * 4 + k] &= ~mk;
#endif
    }
  }
  sync();
  PROF_MARK(PW3);
  // 세로용 전치: 열 x 의 비트 iy = 칸 (x, GW − 1 − iy)
#ifdef __CUDA_ARCH__
  {  // 32 × 32 조각마다 워프 안 비트 전치(셔플 5 번, Hacker's Delight transpose32 의 레인판). 레인 = iy. 워프 둘이 조각 16 개를 나눔
    const int lane = tid & 31;
    for (int tile = tid >> 5; tile < 16; tile += nt >> 5) {
      const int xb = tile & 3, ib = tile >> 2;
      uint32_t v = ws.occ[(GW - 1 - (ib * 32 + lane)) * 4 + xb];
      uint32_t mk = 0x0000ffffu;
#pragma unroll
      for (int j = 16; j > 0; j >>= 1, mk ^= mk << j) {
        const uint32_t p = __shfl_xor_sync(0xffffffffu, v, j);
        if (!(lane & j)) v ^= (((v >> j) ^ p) & mk) << j;
        else v ^= ((p >> j) ^ v) & mk;
      }
      ws.occT[(xb * 32 + lane) * 4 + ib] = v;   // 레인 l = 열 xb·32 + l, 비트 c = iy ib·32 + c
    }
  }
#else
  for (int x = tid; x < GW; x += nt) {
    uint32_t t4[4] = {0u, 0u, 0u, 0u};
    for (int iy = 0; iy < GW; ++iy)
      if ((ws.occ[(GW - 1 - iy) * 4 + (x >> 5)] >> (x & 31)) & 1u) t4[iy >> 5] |= 1u << (iy & 31);
    for (int k = 0; k < 4; ++k) ws.occT[x * 4 + k] = t4[k];
  }
#endif
  sync();
  PROF_MARK(PW4);
  for (int r = tid; r < 2 * GW; r += nt) ws.cnt[r / GW][r % GW] = (uint8_t)row_runs_dev(wall_row(ws, r / GW, r % GW), nullptr, 0);
  sync();
  PROF_MARK(PW5);
  // 행마다 구간 시작 번호(누적, NRUN 에서 자름)와 구간 있는 행 비트. GPU: 워프 P 가 방향 P(레인마다 행 4)
#ifdef __CUDA_ARCH__
  {
    const int lane = tid & 31, P = tid >> 5;
    if (P < 2) {
      int c4[4], loc = 0;
      for (int k = 0; k < 4; ++k) { c4[k] = ws.cnt[P][4 * lane + k]; loc += c4[k]; }
      int inc = loc;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, d);
        if (lane >= d) inc += y;
      }
      int o = inc - loc;
      for (int k = 0; k < 4; ++k) { ws.off[P][4 * lane + k] = (uint16_t)(o < NRUN ? o : NRUN); o += c4[k]; }
      const int tot = __shfl_sync(0xffffffffu, inc, 31);
      if (lane == 31) { ws.off[P][GW] = (uint16_t)(tot < NRUN ? tot : NRUN); ws.ovf[P] += tot > NRUN ? tot - NRUN : 0; }
      const uint32_t nb = (uint32_t)((c4[0] > 0) | ((c4[1] > 0) << 1) | ((c4[2] > 0) << 2) | ((c4[3] > 0) << 3)) << (4 * (lane & 7));
      for (int w = 0; w < 4; ++w) {
        const uint32_t v = __reduce_or_sync(0xffffffffu, (lane >> 3) == w ? nb : 0u);
        if (lane == 0) ws.nz[P][w] = v;
      }
    }
  }
#else
  for (int P = 0; P < 2; ++P) {
    if (tid != 0) continue;
    int o = 0;
    for (int w = 0; w < 4; ++w) ws.nz[P][w] = 0u;
    for (int r = 0; r < GW; ++r) {
      ws.off[P][r] = (uint16_t)(o < NRUN ? o : NRUN);
      o += ws.cnt[P][r];
      if (ws.cnt[P][r]) ws.nz[P][r >> 5] |= 1u << (r & 31);
    }
    ws.off[P][GW] = (uint16_t)(o < NRUN ? o : NRUN);
    ws.ovf[P] += o > NRUN ? o - NRUN : 0;
  }
#endif
  sync();
  PROF_MARK(PW6);
  for (int r = tid; r < 2 * GW; r += nt) {
    const int P = r / GW, rr = r % GW;
    if (ws.cnt[P][rr]) row_runs_dev(wall_row(ws, P, rr), &ws.run[P][ws.off[P][rr]], ws.off[P][rr + 1] - ws.off[P][rr]);
  }
  sync();
  PROF_MARK(PW7);
  const int t1 = nt > 32 ? 32 : 0;   // 세로를 맡는 스레드(GPU 둘째 워프)
  if (tid == 0) m.nseg_h = wall_group(ws, 0, segs_g);
  if (tid == t1) m.nseg_v = wall_group(ws, 1, segs_g + MAXSEG * 4);
  sync();
  if (tid == 0) { m.n_wall_ovf += ws.ovf[0] + ws.ovf[1]; m.n_wall_runs += 1; }
}

// ---- 한 판 한 스텝. Sync = GPU __syncthreads / CPU 아무것도 안 함(tid 0, nt 1) --------------------------------------------------
// keyframe 갱신 본체(시작 단계 뒤, B_KF 일 때만). 블록(GPU) 또는 스레드 하나(CPU)
template <class Sync>
DEV void map_keyframe(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt, int bug, const Sync& sync) {
  phase_cast(m, sh, e, tid, nt);
  sync();
  PROF_MARK(P_CAST);
  obj_pre(m, sh, tid, nt);
  sync();
  PROF_MARK(P_OBJPRE);
  if (tid == 0) obj_detect(m, sh, e, nt);
  sync();
  PROF_MARK(P_POSE_DET);
  obj_keys(m, sh, tid, nt);
  sync();
#ifdef __CUDA_ARCH__
  if (tid < 32) obj_greedy_warp(sh, tid);   // GPU: 워프 0 이 셔플로(바퀴마다 블록 동기 없음). 고르는 규칙은 아래 일반판과 같음
  sync();
#else
  for (;;) {   // 탐욕 짝짓기: 한 바퀴에 한 쌍
    obj_argmin_local(sh, tid, nt);
    sync();
    if (tid == 0) obj_argmin_pick(sh, nt);
    sync();
    if (!sh.more) break;
  }
#endif
  obj_update_matched(m, sh, tid, nt, bug);
  sync();
  if (tid == 0) obj_update_new(m, sh, bug);
  sync();
  PROF_MARK(P_ASSOC);
  obj_absence(m, sh, e, tid, nt);
  sync();
  PROF_MARK(P_ABSENCE);
  obj_complete(m, sh, tid, nt);
  phase_mark(m, sh, L, seen, occ, tid, nt, sync);
  sync();
  PROF_MARK(P_MARK);
  phase_apply(m, sh, L, seen, occ, tid, nt);
  sync();
  PROF_MARK(P_APPLY);
  if (tid == 0) phase_finish(m, sh, e, nt);
}

// ---- 7. 커리큘럼 처음 지도(계획서 5.5, 판 리셋 때만) ---------------------------------------------------------------------------
// 난수는 m.rng 를 섞은 따로 된 값(m.rng 를 움직이지 않음) — 모두 C2 면 지도 결과가 예전과 같다.
// (스레드 0) 단계 고르기 + 미리 확정할 참 물체를 칸에. 자리·크기는 참 상자(중심, 크기) 그대로 = 교사 지도의 "자세 오차만"
DEV void curr_slots(MapCore& m, const MapCurr& cu) {
  uint64_t rr = m.rng ^ 0x6A09E667F3BCC909ull;
  const float u = rand01(rr);
  const int st = u < cu.p0 ? 0 : (u < cu.p0 + cu.p1 ? 1 : 2);
  m.init_stage = st;
  m.init_conf = 0;
  m.init_goal = 0;
  if (st == 2) return;
  int k = N_PRIM;
  if (st == 1) {
    const int lo = cu.kmin < 0 ? 0 : (cu.kmin > N_PRIM ? N_PRIM : cu.kmin);
    const int hi = cu.kmax < lo ? lo : (cu.kmax > N_PRIM ? N_PRIM : cu.kmax);
    k = lo + (int)(rand01(rr) * (float)(hi - lo + 1));
    k = k > hi ? hi : k;
  }
  int idx[N_PRIM];
  for (int p = 0; p < N_PRIM; ++p) idx[p] = p;
  uint32_t pick = 0u;
  for (int j = 0; j < k; ++j) {   // 부분 Fisher–Yates: 서로 다른 k 개
    int r = j + (int)(rand01(rr) * (float)(N_PRIM - j));
    r = r > N_PRIM - 1 ? N_PRIM - 1 : r;
    const int x = idx[j]; idx[j] = idx[r]; idx[r] = x;
    pick |= 1u << idx[j];
  }
  int b = 0;
  for (int p = 0; p < N_PRIM; ++p) {   // 칸은 참 물체 번호 순(빈 칸 앞에서부터)
    if (!((pick >> p) & 1u)) continue;
    const Prim& P = m.prim[p];
    Slot& S = m.slot[b++];
    S.valid = 1; S.id = m.next_id++; S.cls = P.cls;
    for (int a = 0; a < 3; ++a) { S.pos[a] = 0.5f * (P.lo[a] + P.hi[a]); S.ext[a] = P.hi[a] - P.lo[a]; S.first_pos[a] = S.pos[a]; }
    S.n_obs = MP::confirm; S.last_seen = 0; S.last_kf = -1; S.score = 0.9f;
    S.state = S_SEEN; S.confirmed = 1; S.moved = 0; S.misses = 0; S.first_miss = 0; S.held = 0; S.src = p;
    S.seen_len = 0.f; S.seen_rot = 0.f;
  }
  m.init_conf = k;
  m.init_goal = (int)(pick & 1u);
}
// 공개한 칸인가(C0: 방 안·벽 칸 전부, C1: 미리 확정한 물체 중심 reveal_r 안)와 점유인가(방 벽 칸 또는 장면 상자 바닥 자국과 겹침).
// 벽 칸 = 방 경계 ±rh 를 담은 칸(광선이 벽에 맞는 칸과 같은 floor 규칙)
DEV int curr_cell(const MapCore& m, const MapCurr& cu, int ix, int iy, bool& occ_out) {
  const float x0 = (float)ix * RES, y0 = (float)iy * RES, x1 = x0 + RES, y1 = y0 + RES;
  const float cx = x0 + 0.5f * RES, cy = y0 + 0.5f * RES;
  const bool in_x = x1 > -m.rhx && x0 <= m.rhx, in_y = y1 > -m.rhy && y0 <= m.rhy;   // 방(+ 경계 칸)과 겹침
  if (!(in_x && in_y)) return 0;
  const bool wall = (x0 <= -m.rhx && -m.rhx < x1) || (x0 <= m.rhx && m.rhx < x1) || (y0 <= -m.rhy && -m.rhy < y1) || (y0 <= m.rhy && m.rhy < y1);
  if (m.init_stage == 1) {
    bool near = false;
    const float r2 = cu.reveal_r * cu.reveal_r;
    for (int b = 0; b < m.init_conf && !near; ++b) {
      const float dx = cx - m.slot[b].pos[0], dy = cy - m.slot[b].pos[1];
      near = dx * dx + dy * dy <= r2;
    }
    if (!near) return 0;
  }
  bool occ = wall;
  for (int p = 0; p < N_PRIM && !occ; ++p) {
    const Prim& P = m.prim[p];
    occ = P.lo[0] < x1 && P.hi[0] >= x0 && P.lo[1] < y1 && P.hi[1] >= y0;
  }
  occ_out = occ;
  return 1;
}
// (모든 스레드, 낱말마다 한 스레드) 공개한 칸에 로그 오즈·본 칸·점유 비트를 쓰고, 방 안 공개 칸 수를 부분합(part·part2)으로.
// clear_grid 뒤(동기 뒤)에 부른다. 낱말끼리 독립이라 스레드 순서와 무관
DEV void curr_grid(const MapCore& m, const MapCurr& cu, Scratch& sh, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt) {
  int cnt = 0, cnt2 = 0;
  // 방(+ 경계 칸)과 겹치는 행·열 칸 범위 — curr_cell 의 겹침 판정과 같은 식. 밖의 낱말은 clear_grid 가 이미 0 으로 둠(같은 결과, 일만 줄임)
  int ylo = GW, yhi = -1, xlo = GW, xhi = -1;
  for (int l = 0; l < GW; ++l) {
    const float c0 = (float)(l + GX0) * RES, c1 = c0 + RES;
    if (c1 > -m.rhy && c0 <= m.rhy) { ylo = l < ylo ? l : ylo; yhi = l; }
    if (c1 > -m.rhx && c0 <= m.rhx) { xlo = l < xlo ? l : xlo; xhi = l; }
  }
  for (int w = tid; w < NWORD; w += nt) {
    const int row = w / (GW / 32), cx0 = (w % (GW / 32)) * 32;
    if (row < ylo || row > yhi || cx0 + 31 < xlo || cx0 > xhi) continue;
    uint32_t sb = 0u, ob = 0u;
    for (int b = 0; b < 32; ++b) {
      const int idx = w * 32 + b;
      const int ix = idx % GW + GX0, iy = idx / GW + GX0;
      bool o = false;
      if (!curr_cell(m, cu, ix, iy, o)) continue;
      sb |= 1u << b;
      const int Lv = o ? MP::curr_occ : MP::curr_free;
      L[idx] = (int16_t)Lv;
      if (Lv >= MP::q_occ) ob |= 1u << b;
      count_new_seen(m, idx, cnt, cnt2);
    }
    seen[w] = sb;
    occ[w] = ob;
  }
  put_part(sh, tid, cnt);
  put_part2(sh, tid, cnt2);
}
// (스레드 0) 공개한 방 칸 수를 본 칸 수에 더함. 방 드러냄은 같은 스텝의 keyframe(phase_finish)이 같은 규칙으로 정함
DEV void curr_grid_finish(MapCore& m, const Scratch& sh, int nt) {
  const int s = sum_part(sh, nt), s2 = sum_part2(sh, nt), r1 = s2 & 0xffff, r2 = s2 >> 16;
  m.n_seen_room += s;
  m.rseen[0] += s - r1 - r2;
  m.rseen[1] += r1;
  m.rseen[2] += r2;
}

// 시작 단계 뒤의 나머지(flags = phase_begin 결과): 리셋이면 격자 비움, keyframe 이면 갱신, keyframe·들기/놓기면 벽 선분, 마지막에 완성도 쓰기.
// GPU 는 목록(keyframe·리셋·벽)의 판만 블록으로 이 함수를 부르고(map_kf_kernel), 나머지 판은 시작 커널이 완성도만 쓴다.
// sh 와 ws 는 같은 자리(공용체): 벽 단계는 keyframe 단계가 끝난 뒤 Scratch 를 덮어쓴다.
struct MapGrid { int16_t* L; uint32_t* seen; uint32_t* occ; int16_t* segs; };   // 한 판의 장치(또는 CPU) 배열
union KfShared { Scratch sh; WallScratch ws; };
template <class Sync>
DEV void map_rest(MapCore& m, KfShared& u, const EnvView& e, const MapGrid& g, float* met, int N, int i,
                  int tid, int nt, int bug, int flags, const MapCurr& cu, const Sync& sync) {
  if (flags & B_RESET) {
    clear_grid(g.L, g.seen, g.occ, tid, nt);
    if (tid == 0) curr_slots(m, cu);   // 처음 지도 단계(5.5)
    sync();
    if (m.init_stage < 2) {
      curr_grid(m, cu, u.sh, g.L, g.seen, g.occ, tid, nt);
      sync();
      if (tid == 0) curr_grid_finish(m, u.sh, nt);
      sync();
    }
    PROF_MARK(P_RESET);
  }
  if (flags & B_KF) map_keyframe(m, u.sh, e, g.L, g.seen, g.occ, tid, nt, bug, sync);
  if (flags & (B_KF | B_WALL)) {
    const int chg = (flags & B_KF) ? u.sh.occ_chg : 0;   // ws 가 sh 를 덮기 전에 읽음
    sync();
    phase_walls(m, u.ws, g.occ, g.segs, chg, flags & B_KF, tid, nt, sync);
    PROF_MARK(P_WALLS);
  }
  if (tid == 0) write_metrics(m, e, met, N, i);
  PROF_MARK(P_FINISH);
}

// 한 판 한 스텝 전체(CPU 참조판: tid 0, nt 1). GPU 는 같은 phase_begin → map_rest 를 두 커널로 나눠 부른다.
template <class Sync>
DEV void map_block(MapCore& m, KfShared& u, const EnvView& e, const MapGrid& g, float* met, int N, int i,
                   int tid, int nt, int bug, int force_kf, const MapCurr& cu, const Sync& sync) {
  const int flags = phase_begin(m, e, force_kf);
  map_rest(m, u, e, g, met, N, i, tid, nt, bug, flags, cu, sync);
}

}  // namespace gmap

#include "map_tok.h"
