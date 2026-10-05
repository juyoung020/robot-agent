// GPU 안에서 자라는 지도(scenemap 근사, 계획서 GPU_TRAINING.md 5.1·4.4, 단계 G2 앞부분).
// CPU 참조판과 GPU 커널이 **같은 소스**를 쓴다(DEV = __host__ __device__). 판 하나 = 블록 하나(NT 스레드). CPU 는 같은 함수를
// tid = 0, nt = 1 로 부른다. 결과가 스레드 순서와 무관하도록:
//   - 격자: 광선이 지나는 칸은 공유 메모리 비트표(맞음·빈칸)에 atomicOr 로 표시(순서 무관) → 칸마다 한 번 정수 덧셈.
//     scenemap grid.cpp 의 "한 스캔에서 칸마다 한 번, 맞음 우선" 규칙과 같다.
//   - 물체 보임 광선은 쌍마다 따로 칸에 쓰고, 검출·물체 기억(순서가 있는 탐욕 짝짓기)은 스레드 0 이 순서대로 한다.
//   - 난수는 detmath.h 의 splitmix64(판마다 따로, 스레드 0 만 뽑음).
//
// 규칙·기본값의 출처(읽기 전용): src/scene_graph/scenemap
//   grid.hpp GridParams / grid.cpp insert, scan.hpp ScanParams / scan.cpp makeScan, slam2d.hpp SlamParams(움직임 거르기),
//   objmap.hpp ObjParams / objmap.cpp update(짝짓기·확정·옮겨짐·사라짐·버림), scenemap.h sm_object(SM_SEEN..).
// (가정) 표시 값은 실제 기록·사양으로 맞출 값이다. 한 곳(MP)에만 둔다.
#pragma once
#include <cstring>
#include "env.h"
#include "env_soa.h"
// objprob 계산 하나(scenemap 과 같은 헤더 — 같은 것 로지스틱·이름 사후·상위어·κ·칼만·이름 분포 겹침). 저장소 안 상대 경로(빌드마다 include 경로를 안 더하게)
#include "../../../../src/scene_graph/scenemap/include/scenemap/objprob_math.h"
// 잡음·맞춤 값의 기본(MP 가 씀). 기본은 LIMO 탐색 기록 보정(../map_calib/limo, calib_limo.json) + 이 모형에 다시 맞춘 값(README "LIMO 보정").
// odo_t·kf_corr_xy 는 새 LIMO SLAM(behavior-2026 84b373c, 오도메트리 가중 맞추기)의 기록 넷(~/datasets/limo_rec r1–r3·r4live)에 다시 맞춤
// (2026-10-04, map_drift kind 5 — ../map_calib/limo/README.md "새 SLAM"). 옛 값(옛 SLAM 둘째 판에 맞춤): odo_t 0.10, kf_corr_xy 0.001.
// map_drift 로 다시 맞출 때만 -D 로 바꾼다. R1 값(예전 기본)은 ../map_calib/README.md 에 기록으로 남김
#ifndef ODO_T_V
#define ODO_T_V 0.015f
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
#define KF_CORR_XY_V 0.003f
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
enum ProfSec { P_LOAD, P_BEGIN, P_RESET, P_CAST, P_POSE_DET, P_ASSOC, P_ABSENCE, P_OBJPRE, P_MARK, P_APPLY, P_FINISH, P_STORE, P_WALLS, PW1, PW3, PW4, PW5, PW6, PW7, TK_STAGE, TK_A, TK_SEG, TK_SLOT, TK_ROOM, P_MERGE, P_NSEC };
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
constexpr int MAXDET = 2 * N_PRIM + N_GHOST_V;   // 한 keyframe 검출 최대(참 물체 — 큰 것은 조각 둘까지 — + 유령 자리)
constexpr int NCLS = 6;
constexpr int NSRC = N_PRIM + N_GHOST_V;   // 인지 흉내 출처 수(참 물체 + 유령 자리)
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
  // ---- 바뀜 판정(scenemap 3ed710f objmap.hpp ObjParams 기본값 — R1·LIMO 같은 값. 시각은 제어 스텝 0.1 s) ----
  static constexpr float floor_h = 0.05f;                        // 바닥 조각: 관측 상자(10–90 백분위) 윗면 < 0.05 m 면 버림(러그·카펫 이름은 우리 이름 표에 없음)
  static constexpr float name_share = 0.2f, name_switch = 1.25f, name_key = 0.02f;   // 이름 표 몫 ≥ 0.2 면 짝 후보, 다른 이름이면 열쇠 +0.02, 1.25 배 넘어야 이름 바뀜
  static constexpr float name_merge_iou = 0.5f;                  // 병합(da): 이름이 달라도 3D IoU ≥ 0.5
  static constexpr float merge_overlap = 0.35f, merge_min_ext = 0.05f;   // da/merge.hpp MergeParams
  static constexpr float absent_vis = 0.5f, absent_min_px = 12.f, absent_det_k = 1.15f;   // 사라짐 근거(표본 27 = 상자 3×3×3 격자 — 근사판은 점 구름 없음)
  static constexpr int gone_misses_big = 6, gone_min_steps_big = 40, spurious_obs = 5;   // 큰 것 놓침 6·4 s, 관측 < 5 사라짐은 지움
  static constexpr float gone_view_d = 0.5f, gone_step_d = 0.10f, gone_step_rad = 0.087266463f /*5도*/;
  static constexpr float ln_gone_eps = -3.912023f;               // ln(gone_eps 0.02): (1 − p)^k < 0.02 인 k
  static constexpr float link_v = 1.0f, link_d0 = 1.0f, link_max_d = 8.0f;   // 옮겨짐 잇기 거리 ≤ min(8, 1 + 1 m/s · 시간 차)
  static constexpr int link_min_obs = 3, link_wait_steps = 300, link_window_steps = 1800, link_view_gap_steps = 50;   // 30 s, 180 s, 5 s
  static constexpr float view_cell = 0.5f;                       // 본 곳 칸(나타남 판정): 0.5 m × 거리 띠 1..5 m
  static constexpr int move_n = 3, move_dt_steps = 6, moving_steps = 10;   // 움직임 따라가기: 3 번 잇달아, 관측 간격 ≤ 0.6 s, 마지막 따라감 뒤 1 s 안 = 움직이는 중
  static constexpr float move_v = 0.3f, move_min_d = 0.25f, move_max_cam_w = 0.6f;
  // ---- 잡기 확인(scenemap d58c978 capi.cpp robotParams LIMO) ----
  static constexpr float grasp_max_w = 0.06f, grasp_min_gap = 0.005f, grasp_w_tol = 0.025f;   // 가운데 변 ≤ 6 cm, 빈손 틈 < 5 mm, 틈 ± 2.5 cm
  static constexpr int grip_settle_steps = 2;                    // 그리퍼 값이 0.2 s 동안 0.01 rad 안이면 쥠 끝
  static constexpr float grip_settle_eps = 0.01f;
  static constexpr float grasp_off = -0.0119f;                   // 잡는 점 = omx_end_effector_link 에서 링크 x 로 −0.0119 m(E0, URDF grasp_point)
  static constexpr float self_r = 0.05f, self_pad = 0.01f;       // 팔 캡슐 반경(OMX 링크, capi robotBody 0.05) + 거르기 여유
  // ---- 오도메트리·slam ---- (LIMO 탐색 기록 보정 — ../map_calib/limo. 단서: 기록 하나(탐색 판 094010 + limo3), 카메라 FK 1 cm 오차가 든 판)
  // 걸음마다 랜덤워크: 이동 σ = odo_t·거리, 회전 σ = odo_rr·|dθ| + odo_rt·거리. 원 오도메트리에는 걸음 잡음이 없어 SLAM 잔차에 맞춘 유효 값
  static constexpr float odo_t = ODO_T_V, odo_rr = ODO_RR_V, odo_rt = ODO_RT_V;
  // 판마다 뽑는 배율 치우침(판 리셋 때 가우스, 판 안에서 고정): 이동 배율 σ 0.82 %, 회전 배율 σ 1.21 %, 직진 중 yaw 0
  // (LIMO 탐색 기록 보정: limo3 원 오도메트리 대 GT — 직진 0.80 m 에서 +0.82 %, 제자리 363° 에서 +1.21 %/rad. 표본 하나라 σ = |측정값|, 부호는 ± 대칭)
  static constexpr float odo_bt = ODO_BT_V, odo_br = ODO_BR_V, odo_bw = ODO_BW_V;
  // keyframe 맞추기가 되돌리는 오차 비율, xy 와 yaw 따로 (LIMO 탐색 기록 보정 — 이 모형에서 map_drift kind 2 로 맞춤)
  static constexpr float kf_corr_xy = KF_CORR_XY_V, kf_corr_yaw = KF_CORR_YAW_V;
  // ---- 검출 ----
  // TODO: R1/COCO-era — to be re-measured with ObjectSAM pipeline (GPU_MAP_PORT)
  // 놓침 확률, 카메라–물체 중심 거리별 계단(R1 시뮬 기록 보정: < 1.5 m 0.15, 1.5–2.5 m 0.10, ≥ 2.5 m 0.79. LIMO 기록으로는 맞추지 못해 그대로)
  static constexpr float miss_d1 = 1.5f, miss_d2 = 2.5f, p_miss_near = 0.15f, p_miss_mid = 0.10f, p_miss_far = 0.79f;
  // TODO: COCO-80 detector era — to be re-measured with ObjectSAM pipeline (GPU_MAP_PORT)
  static constexpr float p_conf = P_CONF_V;                      // 틀린 이름 (LIMO 탐색 기록 보정: 확정 물체 틀린 이름 3/16 = 0.19 에 맞춤)
  // 가짜 물체: 판마다 정해진 유령 자리 n_ghost 개가 시야에 들면 keyframe 마다 p_ghost 로 검출된다
  // (LIMO 탐색 기록 보정: limo3 가짜가 있는 keyframe 1/80 = 0.013, 탐색 판 확정 가짜 0/16)
  static constexpr int n_ghost = N_GHOST_V;
  static constexpr float p_ghost = P_GHOST_V;
  static constexpr float dn0 = 0.0f, dn2 = 0.006f;               // 깊이 잡음 σ = 0 + 0.006·z² m (Dabai 데이터시트: 1 m 에서 6 mm, z² 법칙)
  // TODO: R1/COCO-era — to be re-measured with ObjectSAM pipeline (GPU_MAP_PORT)
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
  static constexpr float grasp_r = 0.12f, grip_closed = 0.6f, hand_r = 0.10f;   // omx_gripper_joint_1 < 0.6 rad 면 닫힘(d58c978: 옛 0.35 는 4 cm 를 쥔 0.41 rad 를 못 봄)
  static constexpr float base_z = 0.15f;                         // base_footprint → base_link (URDF base_joint)
  // ---- 방(계획서 5.1 의 5 번: 참 방 표 + 본 비율로 드러냄) ----
  static constexpr float min_room_m2 = 2.0f;                     // rooms.hpp RoomParams min_room_m2 (scenemap)
  static constexpr float room_reveal = 0.3f;                     // 방 칸 중 본 비율이 이 이상이면 드러냄 (가정)
  static constexpr float room_l2 = 4.5f, room_l3 = 6.0f;         // 긴 변이 이 이상이면 방 2·3 개 (가정)
  static constexpr float room_wmin = 1.2f;                       // 방 최소 폭 (가정)
  // ---- 지도 토큰(VLA_INPUT 3절) ----
  // 팔이 닿는지: omx_workspace_grasp.h (URDF 순기구학 + 관절 한계로 미리 계산한 잡는 점 (r, z) 작업 공간, tools/omx_ws --grasp) — 예전 0.40 m 구·팔 끝 표는 뺐다
  static constexpr float tok_dt = 0.1f;                          // 제어 스텝 s(물체 속도·마지막 본 뒤 시간)
  // 다음 경유 지점(VLA_INPUT 4절, sm_snap_place_path 대신): 믿는 점유 격자를 2×2 로 묶은 0.2 m 거친 격자(점유 = 2×2 중 하나라도 점유,
  // 1 칸 부풀림 — 몸통 외접원 0.194 m), 안 본 칸은 지나갈 수 있음. 목표 칸에서 8 이웃 BFS → 로봇 칸에서 내리막으로 way_look 칸 간 자리 (가정: 칸·앞보기)
  static constexpr float way_res = 0.2f;
  static constexpr int way_look = 5, way_iter = 160;
  // ---- objprob 합치기(scenemap objprob.hpp ApParams + 엔진 매개변수 tools/realbag/objprob_params/yolo26n-seg-obj-416.json, 10-05) ----
  // 같은 것 로지스틱: 특징 [접촉, 상자 틈, 중심 거리/크기, cos − cos0, 겹침, 받침, 이름 분포 겹침 − 0.5]. w = 관측 ↔ 물체, wm = 물체 ↔ 물체(병합)
  static constexpr float ap_w0 = -4.235f, ap_w1 = 5.063f, ap_w2 = -5.289f, ap_w3 = -0.3241f, ap_w4 = 5.02f, ap_w5 = 1.417f, ap_w6 = -0.013f, ap_w7 = 0.f;
  static constexpr float ap_wm0 = -2.671f, ap_wm1 = 1.209f, ap_wm2 = -4.964f, ap_wm3 = -0.3121f, ap_wm4 = 6.248f, ap_wm5 = 1.648f, ap_wm6 = -0.5892f, ap_wm7 = 2.04f;
  static constexpr float ap_same_p = 0.5f, ap_merge_p = 0.7f, ap_cos0 = 0.75f, ap_gate = 0.30f;   // 엔진 json(same_p·merge_p), ApParams(cos0·gate)
  static constexpr float ap_contact_d = 0.06f;   // 접촉 특징(4 cm 칸 이웃 27 칸 안 비율)을 상자 표본 27 점의 "물체 상자에서 이 거리 안" 비율로 (가정: 칸 대각 반)
  static constexpr float ap_touch_c = 0.3f, ap_touch_p = 0.2f;   // 접촉 ≥ 0.3 또는 P ≥ 0.2 면 '이번에 보임'(objmap update 2)
  static constexpr float kap_k0 = 4369.f, kap_s0 = 40.f, kap_d0 = 100.f;   // viewKappa(엔진 json; trunc 1·occ 1·blur 끔)
  static constexpr float ap_temper = 0.3f, ap_temper_d = 0.30f, ap_temper_deg = 15.f, ap_kappa_ref = 3000.f;   // ApParams
  static constexpr float ap_name_tau = 0.5f, ap_name_wmax = 6.0f;   // 이름 사후 문턱·Σw 상한
  static constexpr float ap_q_pos = 1e-4f, ap_r0 = 0.02f, ap_r1 = 0.01f;   // 칼만(ObjParams ap_q_pos m²/s, 관측 σ = r0 + r1·깊이)
  static constexpr float ap_link_cos = 0.8f;     // 옮겨짐 잇기 생김새 cos 하한(ApParams link_cos)
  static constexpr float emb_d = 768.f;          // SigLIP 2 B/32 벡터 차원(vMF ε² = (d − 1)/(2κ))
  // ---- 살펴본 정도(scenemap inspect.hpp InspectParams) ----
  static constexpr float insp_view_d = 0.30f, insp_view_deg = 15.f;
  static constexpr int insp_view_cap = 32;
  static constexpr float insp_top_range = 2.0f, insp_top_inc_deg = 80.f, insp_top_probe_h = 0.05f, insp_top_min_side = 0.25f, insp_top_zmin = 0.20f,
                         insp_top_zmax = 1.50f, insp_top_tol0 = 0.05f, insp_top_tol_k = 0.02f;
  // ---- 인지 흉내(percept.h, 통계판 — 값은 맞춤 전 처음 값, GPU_MAP_PORT.md 2절. BASELINE 이 오면 map_calib/percept json 으로) ----
  static constexpr float pe_p_mm = 0.5f;         // 지난 keyframe 에 놓친 물체를 또 놓칠 확률(2 상태 마르코프, 가정)
  static constexpr float pe_q_top = 0.55f;       // 좋은 모습(κ = κ_ref)에서 맞는(또는 체계적으로 틀린) 이름의 p(c|z) (가정)
  static constexpr float pe_q_rest = 0.02f;      // 상위 4 밖 라벨 전체 몫(가정)
  static constexpr float pe_ll_sig = 0.3f;       // 관측 log p 잡음 σ(κ_ref 기준, √(κ_ref/κ) 배) (가정)
  static constexpr float pe_cos_same = 0.85f, pe_cos_sim = 0.75f, pe_cos_diff = 0.55f;   // 출처 원형 cos: 같은 종류 다른 개체 / 비슷한 이름 / 그 밖 (가정 — P5 SigLIP 2 표로)
  static constexpr float pe_cos_jit = 0.02f;     // 관측 cos 흔들림 σ (가정)
  static constexpr float pe_p_split = 0.10f, pe_split_min = 0.8f;   // 큰 물체(수평 긴 변 ≥ 0.8 m) 조각 둘로 (가정, radio r3 중복 45/96 로 맞출 것)
  static constexpr float pe_p_under = 0.05f, pe_under_gap = 0.05f;  // 맞닿은 두 물체 한 마스크로 (가정, 잘못 합침 12/96)
  static constexpr int pe_n_lab = 4;             // 관측 이름 상위 k
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
  int nav_k;        // BEHAVIOR 다가가기 거리장 다시 계산 주기(스텝, 판마다 m.t % nav_k == 0 과 판 리셋 때). 0 = 기본 NAV_K_DEF (예전 pad 자리)
};
// 기본 = 모두 C2(빈 지도): G2·G3 와 같은 지도
constexpr MapCurr kCurrEmpty = {0.f, 0.f, 3, 6, 1.5f, 0};
DEV bool is_static(int cls) { return cls == C_TABLE || cls == C_CABINET; }   // capi.cpp kStaticNames 의 table·cabinet
// BEHAVIOR 판(MapCore::init_pad = 1)은 종류 = 이름 표 행이라 고정 종류 목록을 쓰지 않는다(큰 것 > big 규칙만)
#define is_static_m(m, cls) ((m).init_pad == 0 && is_static(cls))

// 본 곳 칸(나타남 판정, objmap markView): 창 0.5 m 칸 VC × VC, 거리 띠 VB(1..5 m)
constexpr int VC = (int)(GW * RES / MP::view_cell + 0.5f) + 2;   // 26 + 2(창 가장자리 칸이 반 칸 걸침)
constexpr int VB = 5;
// ---- 판마다 지도 상태 -------------------------------------------------------------------------------------------------
struct Prim {   // 정적 장면 상자(축 정렬), 바닥에 놓임
  int cls;
  float lo[3], hi[3];
};
constexpr int NLAB = 6;    // 물체마다 이름 우도를 드는 라벨 수(objprob 은 라벨 전부 — 근사: 상위 6 + 나머지 하나, GPU_MAP_PORT R9)
constexpr int NVIEW = 8;   // 살펴본 정도 시점 고리(inspect.cpp 는 32 개 — 8 까지 정확, 그 뒤 어림, GPU_MAP_PORT R14)
struct Slot {   // 물체 기억 한 칸 (scenemap.h sm_object / objmap.hpp MapObject + objprob ApState + inspect InspectState). 공유 메모리에 올리므로 작은 값은 좁은 정수로
  int id, n_obs, last_seen, last_kf, first_miss;   // 시각 = 판 시작 뒤 제어 스텝
  int first_seen;             // 처음 본 스텝(옮겨짐 잇기: n 처음 > m 마지막)
  int trk_t, moving_t;        // 움직임: 마지막 관측 스텝(−1 없음), 마지막으로 따라간 스텝
  int16_t cls;                // 이름 = objprob 이름 사후(apName): 라벨 행(문턱 넘음) 또는 상위어 행, −1 = 모름("object")
  int16_t src;                // 생김새 주 출처(인지 흉내 꼬리표): 참 물체 prim 번호, 유령 g 는 −1−g. 합치기는 cos 계산(emb_cos)으로만 씀
  int16_t src2;               // 둘째 출처(덜 나뉜 마스크·합침), −32768 = 없음
  int16_t misses, n_vis_miss; // 연속 놓침, 보일 만한데 놓친 keyframe 수(검출률)
  int16_t lab[NLAB];          // 이름 우도를 든 라벨(−1 빈 칸)
  int8_t valid, state, confirmed, moved;
  int8_t held;                // 잡는 점이 들고 있음(objmap held_by ≥ 0). 짝짓기·옮겨짐 잇기·부재 확인·벽 무시 영역에서 뺀다
  int8_t appeared;            // 전에 본 자리(처음 검출 거리 이하에서 5 s 넘게 전)에 새로 나타남 — 잇기 후보
  int8_t mv_cnt;              // 잇달아 빠르게 같은 쪽으로 간 관측 수
  uint8_t vn;                 // 살펴본 정도: 고리 안 시점 수(≤ NVIEW)
  uint8_t n_views;            // 살펴본 정도 n_views(≤ insp_view_cap)
  uint8_t vnew;               // 이번 keyframe 에 더한 고리 칸 비트(같은 영상의 조각끼리는 temper 안 함)
  uint8_t tset;               // 윗면 칸 상자를 정했나
  uint8_t dirty;              // 지난 병합 판정(ap_merge_keys) 뒤 바뀜 — 안 바뀐 쌍은 같은 P 라 다시 셈하지 않음(결과 같음)
  uint16_t top_bits;          // 윗면 4 × 4 칸 본 비트(칸 (i, j) = 비트 4j + i)
  int16_t vx[NVIEW], vy[NVIEW];   // 시점 고리: 카메라 xy cm
  int8_t vyaw[NVIEW];         // 광축 yaw(rad × 40)
  float pos[3], ext[3], first_pos[3], score;
  float seen_len, seen_rot;   // 마지막으로 본 때의 믿는 오도메트리 누적 이동·회전(토큰의 위치 불확실도)
  float meas[3];              // 마지막 관측 자리(그 keyframe 의 검출 그대로, map 좌표) — 토큰의 "지금 보는 중" 칸 위치(VLA_INPUT 3절: 보이면 이번 프레임 깊이)
  float max_det_z;            // 검출한 가장 먼 카메라 깊이(사라짐 판정은 이 안에서만)
  float miss_cam[2];          // 연속 놓침 시작 때 (믿는) 카메라 xy
  float lmiss_cam[2], lmiss_yaw;   // 마지막으로 센 놓침의 카메라 xy·광축 yaw
  float trk_pos[2], trk_step[2];   // 마지막 관측 중심 xy(날 것)·한 걸음
  float L[NLAB], Lrest, lw;   // 라벨마다 Σ w log p(c|z), 그 밖 라벨 하나의 Σ, Σ w (objprob L_frag·lw_frag)
  float K, w2;                // 모습 κ 합(temper 뒤, vMF ‖r‖ 쪽), 둘째 출처 몫
  float P[3];                 // 칼만 분산 m²(축마다)
  float name_p, name_p2;      // 고른 이름의 사후 확률, 둘째 라벨 사후(이름 확신도 2 — GPU_MAP_PORT 5절)
  float post[NLAB], prest;    // 이름 사후(든 라벨, 나머지 라벨 하나) — ap_name 이 셈(병합 판정이 다시 쓰게)
  float closest;              // 살펴본 정도 closest_view_m(−1 = 없음)
  float tbox[4];              // 윗면 칸을 정한 상자 x0 y0 x1 y1
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
  // 잡기 확인(d58c978, LIMO): 잡는 점(map), 그리퍼 멈춤 기준 값·스텝, 이번 닫힘에 골랐나. 팔 뼈대(base_footprint, 거르기 캡슐)
  float gp_b[3];           // 잡는 점 base_link(eef_b 와 같은 때 다시 계산)
  float gp_m[3];           // 잡는 점(URDF grasp_point = 팔 끝 − 0.0119 m 링크 x) — 잡기·손에 든 것 거르기·손 둘레 판단(scenemap T_eef)
  float gref;              // 멈춤 판정 기준 그리퍼 값
  int gref_t, tried;       // 기준을 잡은 스텝(−1 없음), 이번 닫힘에 이미 골랐나
  float arm[7][3];         // omx_link0(mount), 관절 1..5 원점, 팔 끝 — base_footprint(scenemap LimoFk pts). 캡슐 반경 self_r + self_pad
  float arm_xmax;          // 팔 뼈대 x 최댓값(카메라 앞에 없으면 거르기 건너뜀 — 결과 같음)
  float arm_zmin;          // 카메라 앞으로 나온 캡슐(끝점 x + 반경 > cam_x)의 끝점 z 최솟값 — 시선이 이보다 반경 넘게 아래면 건너뜀(결과 같음)
  float cam_yaw_kf;        // 지난 keyframe 의 믿는 카메라 yaw(움직임 근거: 카메라가 0.6 rad/s 넘게 돌면 안 씀)
  int cam_t_kf;            // 그 스텝(−1 없음)
  int n_relink_total, n_merge_total;   // 옮겨짐 잇기·중복 병합 누적(리셋에 안 지움, 통계)
  uint32_t pe_miss;        // 인지 흉내 2 상태 마르코프: 지난 keyframe 에 후보였는데 놓친 prim 비트(percept.h)
  int nlab;                // 이름 라벨 수(BEHAVIOR 이름 표 행 수, 상자 방 NCLS) — 판 리셋 때
  int16_t snm[NSRC];       // 출처(prim 0..N_PRIM−1, 유령 N_PRIM + g)의 이름 행 — 판 리셋 때(장면 이름 표를 전역에서 다시 읽지 않게)
  int16_t ssim[NSRC][3];   // 그 이름의 비슷한 이름 3(BEHAVIOR 장면 묶음 sim3, 상자 방 다음 종류들)
  int pad_core[6];         // 16 B 배수(장치 복사)
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

// 인지 흉내 층의 출력(percept.h) = objprob 합치기의 입력: keyframe 검출 하나 — ObjectSAM 마스크 + 깊이 점 + SigLIP 2 가 냈을 값(GPU_MAP_PORT 0.2).
// 합치기는 이 구조체만 읽는다(흉내 층을 학습형 PEM·진짜 검출로 바꿔도 그대로). 자리는 그 순간 믿는(slam) 자세로 놓은 map 좌표.
// 생김새 벡터는 출처 꼬리표(src·src2·w2) + 모습 κ + 흔들림 jit 로 나타내고 cos 는 emb_cos 가 원형 표에서 구한다(GPU_MAP_PORT R7)
constexpr int NLO = MP::pe_n_lab;
struct Det {
  float pos[3];     // 보이는 면 점 중앙값(objmap 관측 자리)
  float ext[3];     // 10–90 백분위 폭
  float bc[3];      // 백분위 상자 가운데
  float zmed;       // 카메라 깊이 중앙값(사라짐 판정 거리)
  float kappa;      // 모습 신뢰도 κ(bestview viewKappa)
  float camd;       // 카메라 광학 중심 ↔ 관측 중심 거리(살펴본 정도 closest)
  float jit;        // 생김새 cos 흔들림(흉내 난수)
  float w2;         // 둘째 출처 몫(덜 나뉜 마스크)
  float ll[NLO];    // 이름 상위 k 라벨의 log p(c|z)(라벨 위로 정규화)
  float llrest;     // 그 밖 라벨 하나의 log p(c|z)
  float score;      // 검출 점수(토큰 T_SCORE)
  float npx;        // 보이는 넓이 화소(조각 합치기의 대표 = 가장 큰 것, κ 의 크기)
  int16_t lab[NLO]; // 상위 k 라벨(−1 빈 칸)
  int16_t src, src2, trunc, pad;   // 출처(참 prim, 유령 −1−g), 둘째 출처(−32768 없음), 영상 가장자리에 닿음
};
// 검출 기하(참 카메라 기준). med·bc: 보이는 면 점의 축별 중앙값·10–90 백분위 상자 중심(카메라 기준 앞·왼쪽·위), pe: 백분위 폭
struct DetGeo { float ctr[3], ext[3], rx, ry, fwd, left, up, rh, af; float med[3], bc[3], pe[3]; float xr[2], yr[2]; int inr; };   // inr 비트 0: 일부라도 깊이 범위 안, 비트 1: 상자가 영상 가장자리에 닿음(trunc)   // inr: 일부라도 깊이 범위 안(진단용)
constexpr int NPART = (NT + 31) / 32;

// 블록 안 작업 공간(GPU 공유 메모리, CPU 지역 변수)
struct Scratch {
  // ---- 앞 부분(geo ~ hit)은 물체 단계(obj_absence)까지만 쓴다. GPU 는 그 뒤(격자 표시 동안) 이 자리에 점유 비트를 비동기로 옮겨 두고
  //      격자 갱신이 바뀐 낱말을 고쳐 쓴다 — 벽 단계(WallScratch::occ, 같은 자리 = 0)가 전역에서 다시 읽지 않게
  alignas(16) DetGeo geo[N_PRIM];   // 보임 점과 무관한 판정의 기하(검출이 다시 씀)
  Det det[MAXDET];
  union {
    float key[MAXDET * KSLOT];      // 짝 열쇠(관측 × 칸), 짝 아님 = -1 (짝짓기까지)
    struct { int ab_ok[KSLOT], ab_vis[KSLOT], ab_thru[KSLOT], ab_u0[KSLOT], ab_u1[KSLOT], ab_v0[KSLOT], ab_v1[KSLOT]; };   // 사라짐 근거(부재 확인 동안)
  };
  int obs_to[MAXDET], hit[KSLOT];
  int head[KSLOT];                  // 칸마다 붙은 관측 중 대표(가장 큰 것), 없으면 −1
  uint32_t touch;                   // 칸 비트: 이번 관측이 닿음(접촉 ≥ 0.3 또는 P ≥ 0.2 — 놓침으로 안 셈)
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
  int steady;                       // 이번 keyframe 카메라가 move_max_cam_w 보다 천천히 돎(움직임 근거로 씀)
};
static_assert(offsetof(Scratch, hitb) >= sizeof(uint32_t) * NWORD, "occupancy copy (offset 0) must fit in the dead object-phase region");
DEV uint32_t* scr_occ(Scratch& sh) { return reinterpret_cast<uint32_t*>(&sh); }   // 점유 비트 사본(GPU, 격자 표시 뒤) = WallScratch::occ

// furn: A2 가구 상자(환경 SoA 의 이 판 자리, 줄 간격 N). 판 리셋 때만 읽는다. nullptr(손으로 만든 EnvView) 이면 가구 없음
struct EnvView { float x, y, yaw, v, w, tx, ty, rhx, rhy; float q[env::N_Q]; int ep; const float* fb; const int* fi; int fs;
                 int bkind, bscene, bent; float bwx, bwy; int binstr;     // BEHAVIOR 판(E2): 단계(0 = 상자 방), 장면, 시작 조건 번호, 창 가운데(세계), 지시문 행(−1 없음)
                 int bgmode; float bgp[3];                                 // 목표 꼴(bsc::GoalMode)·놓을 점(창 좌표) — 목표 칸(map_tok.h)
                 int bpnp; float bo[4]; };                                 // 잡기 물리 판(E6, 단계 ≥ B4): 집을 물체의 참 자세(창 좌표 가운데 xyz, yaw) — 환경이 옮김
DEV EnvView read_env(const env::Soa& s, int i, bool beh = false) {   // beh: BEHAVIOR 판 값도 읽음(장면 묶음이 있는 지도만)
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
  e.bkind = beh ? s.iv[env::I_B_KIND * N + i] : 0;
  e.bscene = beh ? s.iv[env::I_B_SCENE * N + i] : 0;
  e.bent = beh ? s.iv[env::I_B_ENT * N + i] : 0;
  e.bwx = beh ? s.f[env::F_B_WX * N + i] : 0.f;
  e.bwy = beh ? s.f[env::F_B_WY * N + i] : 0.f;
  e.binstr = beh ? s.iv[env::I_B_INSTR * N + i] : -1;
  e.bgmode = beh ? s.iv[env::I_B_GMODE * N + i] : 0;
  for (int a = 0; a < 3; ++a) e.bgp[a] = beh ? s.f[(env::F_B_GPX + a) * N + i] : 0.f;
  e.bpnp = e.bkind >= bsc::EK_B4 ? 1 : 0;   // 예전 판(≤ B3)은 아래 값을 읽지 않음
  for (int a = 0; a < 3; ++a) e.bo[a] = e.bpnp ? s.f[(env::F_O_X + a) * N + i] : 0.f;
  e.bo[3] = e.bpnp ? s.f[env::F_O_YAW * N + i] : 0.f;
  return e;
}

// ---- BEHAVIOR 장면 판(E2): 판마다 작은 덧붙임(BMapEnv, 장치 배열 [N]) + 장면 묶음(bsc::SceneSet, 모든 판이 같이) -------------------------
// 상자 방(A0–A2) 판은 on = 0 이라 아래 코드를 하나도 지나지 않는다(예전 결과 그대로). 판 좌표 = 환경의 창 좌표(env_beh.h) = 지도 창.
// 지도 물체(prim) = 시작 조건 표의 물체 9(목표 + 과제 물체 + 가구, 이름 = vla_v1 이름 표 행). 정적 가구 물체는 정적 상자(회전)와 같은 것이라
// 광선은 정적 상자로 맞히고(prim 상자는 검출 판정·지도 칸 크기에만), 과제 물체(움직이는 상자)만 prim 상자로 맞힌다.
constexpr int NO_PRIM_Z = -10;   // 빈 prim 자리: 바닥 밑 먼 곳(어떤 광선·검출에도 안 걸림)
struct BMapEnv {
  int on, scene, ent, nprim;
  float wx, wy;                  // 창 가운데(세계)
  int c0, r0;                    // 창 칸 (0, 0) = 장면 칸 (c0, r0)
  int nroom, ndoor, room_cells, kind;
  int goal;                      // 목표 물체 prim 비트(POLICY 1.3 objects:[…] — 칸의 "목표인지"): 0 = 집을 물체, 1 = 놓을 곳(바닥이 아닐 때)
  int instr;                     // 판의 지시문 행(환경 I_B_INSTR, −1 없음) — 지도 토큰 instr1 로(관측이 표에서 벡터로)
  int gmode;                     // 목표 꼴(bsc::GoalMode 비트: GM_PLACE_PT 놓을 칸 = 점, GM_GOTO 점으로 가기 — 집을 칸 없음)
  float gp[3];                   // 놓을 점(창 좌표 = 지도 좌표, z = 놓이는 바닥 높이 m)
  int16_t sbox[N_PRIM];          // prim 의 정적 상자 번호(−1 = 과제 물체)
  int16_t pad16;
  int8_t lut[32];                // 장면 방 번호 → 창 방 번호(−1 = 창에 없음)
  int16_t rid[bsc::MAXBR];       // 창 방 → 장면 방
  int rcells[bsc::MAXBR], rseen[bsc::MAXBR];
  int rcnt[32];                  // 리셋 때 장면 방별 칸 수(원자 덧셈 작업 공간)
  float dx[bsc::MAXBD], dy[bsc::MAXBD];   // 창 안 문(창 좌표)
  int8_t da[bsc::MAXBD], db[bsc::MAXBD];  // 문 양쪽 창 방(−1 = 없음/창 밖)
};
struct BCtx {   // 한 판의 BEHAVIOR 맥락(값으로). on = 0 이면 상자 방
  int on;
  const bsc::SceneSet* ss;
  const bsc::SceneDev* sd;
  BMapEnv* bm;
  float wx, wy;
};
DEV BCtx bctx(const bsc::SceneSet* ss, BMapEnv* bm) {
  BCtx x;
  x.on = 0; x.ss = ss; x.sd = nullptr; x.bm = bm; x.wx = 0.f; x.wy = 0.f;
  if (ss && bm && bm->on) { x.on = 1; x.sd = &ss->sc[bm->scene]; x.wx = bm->wx; x.wy = bm->wy; }
  return x;
}
DEV int broom_local(const BCtx& bx, float x, float y) {   // 창 좌표 점 → 창 방 번호(−1)
  const int r = bsc::room_at(*bx.sd, x + bx.wx, y + bx.wy);
  return (r >= 0 && r < 32) ? (int)bx.bm->lut[r] : -1;
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
// ---- BEHAVIOR 판 광선: 바닥·천장 평면 + 정적 회전 상자(1 m 묶음 걷기, 세계) + 과제 물체 상자(prim, 창 좌표) ----
// 회전 상자 3D: xy 는 상자 축으로 돌린 판, z 판은 그대로. o 는 세계, 들어가는 t(> 0), 없으면 kInf
DEV float ray_obb_inv(const float o[3], const float d[3], const bsc::SBox& b) {
  const float rx = o[0] - b.cx, ry = o[1] - b.cy;
  const float lx = b.c * rx + b.s * ry, ly = -b.s * rx + b.c * ry;
  const float ux = b.c * d[0] + b.s * d[1], uy = -b.s * d[0] + b.c * d[1];
  const float ix = absf(ux) < 1e-12f ? 0.f : 1.f / ux, iy = absf(uy) < 1e-12f ? 0.f : 1.f / uy, iz = absf(d[2]) < 1e-12f ? 0.f : 1.f / d[2];
  float t0 = -kInf, t1 = kInf;
  if (!slab(lx, ux, ix, -b.hx, b.hx, t0, t1) || !slab(ly, uy, iy, -b.hy, b.hy, t0, t1) || !slab(o[2], d[2], iz, b.z0, b.z1, t0, t1)) return kInf;
  return slab_end(t0, t1);
}
DEV float ray_planes(const float o[3], const float d[3]) {   // 바닥(0)·천장(wall_h)
  return d[2] < 0.f ? -o[2] / d[2] : d[2] > 0.f ? (MP::wall_h - o[2]) / d[2] : kInf;
}
DEV bool prim_dyn(const BCtx& bx, int p) { return p < bx.bm->nprim && bx.bm->sbox[p] < 0; }
DEV float cast_beh(const BCtx& bx, const MapCore& m, const float o[3], const float d[3]) {
  float inv[3];
  ray_inv(d, inv);
  float t = ray_planes(o, d);
  const float ow[3] = {o[0] + bx.wx, o[1] + bx.wy, o[2]};
  const bsc::SceneDev& sd = *bx.sd;
  bsc::bin_walk(sd, ow[0], ow[1], d[0], d[1], [&](int j, float) { t = minf(t, ray_obb_inv(ow, d, sd.box[j])); }, [&]() { return t; });
  for (int p = 0; p < N_PRIM; ++p) if (prim_dyn(bx, p)) t = minf(t, ray_box_inv(o, d, inv, m.prim[p]));
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

// BEHAVIOR 판 장면(판 리셋 때, 스레드 하나): prim = 시작 조건 표의 물체, 빈 자리는 바닥 밑 먼 곳. BMapEnv 머리(방 세기는 map_rest 리셋 갈래가)
DEV void make_scene_beh(MapCore& m, const EnvView& e, const bsc::SceneSet& ss, BMapEnv& bm) {
  const bsc::Entry& E = ss.ent[e.bent];
  const bsc::SceneDev& d = ss.sc[e.bscene];
  bm.on = 1; bm.scene = e.bscene; bm.ent = e.bent; bm.nprim = E.nprim; bm.kind = e.bkind; bm.instr = e.binstr;
  // 목표 칸 표시(T_TARGET)와 목표 칸(map_tok.h goal): 점으로 가기면 물체 목표 없음, 놓을 곳이 점이면 받침 물체 표시 없음
  bm.gmode = e.bgmode;
  for (int a = 0; a < 3; ++a) bm.gp[a] = e.bgp[a];
  bm.goal = ((e.bgmode & bsc::GM_GOTO) ? 0 : 1) |
            ((E.list == bsc::L_OBJ && E.dkind != bsc::DK_FLOOR && E.nprim > 1 && !(e.bgmode & bsc::GM_PLACE_PT)) ? 2 : 0);
  bm.wx = e.bwx; bm.wy = e.bwy;
  bm.c0 = (int)floorf((e.bwx - bsc::WIN_HALF - d.ox) / bsc::CELL + 0.5f);
  bm.r0 = (int)floorf((e.bwy - bsc::WIN_HALF - d.oy) / bsc::CELL + 0.5f);
  for (int p = 0; p < N_PRIM; ++p) {
    Prim& P = m.prim[p];
    if (p < E.nprim) {
      const bsc::BPrim& B = E.prim[p];
      P.cls = B.name;
      for (int a = 0; a < 3; ++a) { P.lo[a] = B.lo[a]; P.hi[a] = B.hi[a]; }
      bm.sbox[p] = B.sbox;
    } else {
      P.cls = -1;
      P.lo[0] = 1e4f; P.lo[1] = 1e4f; P.lo[2] = (float)NO_PRIM_Z;
      P.hi[0] = 1e4f; P.hi[1] = 1e4f; P.hi[2] = (float)NO_PRIM_Z;
      bm.sbox[p] = -1;
    }
  }
}

DEV void reset_core(MapCore& m, const EnvView& e, const bsc::SceneSet* ss = nullptr, BMapEnv* bm = nullptr) {
  const bool beh = ss != nullptr && bm != nullptr && e.bkind != 0;
  if (bm) bm->on = 0;
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
  if (beh) make_scene_beh(m, e, *ss, *bm); else make_scene(m, e);
  m.init_pad = beh ? 1 : 0;   // BEHAVIOR 판 표시(종류 = 이름 표 행)
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
    G.cls = (int)(rand01(m.rng) * (float)(beh ? ss->nname : NCLS));   // BEHAVIOR: 아무 이름 표 행
  }
  // 인지 흉내 출처 캐시: 이름 행·비슷한 이름 3, 라벨 수
  m.nlab = beh ? ss->nname : NCLS;
  for (int q = 0; q < NSRC; ++q) {
    const int c = q < N_PRIM ? m.prim[q].cls : m.ghost[q - N_PRIM].cls;
    m.snm[q] = (int16_t)c;
    for (int k = 0; k < 3; ++k) m.ssim[q][k] = (int16_t)(c < 0 ? -1 : beh ? (int)ss->sim3[c * 3 + k] : (c + 1 + k) % NCLS);
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
  m.gref = 0.f; m.gref_t = -1; m.tried = 0; m.cam_yaw_kf = 0.f; m.cam_t_kf = -1;
  for (int a = 0; a < 3; ++a) { m.eef_b[a] = 0.f; m.eef_m[a] = 0.f; m.held_rel[a] = 0.f; m.grasp_pos[a] = 0.f; }
  m.plen = 0.f; m.prot = 0.f;
  m.nseg_h = 0; m.nseg_v = 0; m.wnrect = -1; m.conf_mask = 0;
  if (beh) {   // 방은 map_rest 리셋 갈래(블록)가 장면 방 격자로 센다(room_setup_beh). 그전까지 방 하나·안 드러남
    m.raxis = 0; m.n_room = 1; m.rrev = 0;
    for (int k = 0; k < MAXROOM; ++k) { m.rtype[k] = N_RTYPE; m.rcells[k] = 0; m.rseen[k] = 0; }
    for (int k = 0; k < MAXROOM - 1; ++k) { m.rcut[k] = 0.f; m.rdoor[k] = 0.f; }
  } else {
    make_rooms(m, e);
  }
}

// ---- 물체 기억 도움 함수(scenemap 3ed710f·d58c978 규칙, CPU·GPU 같은 소스) --------------------------------------------------
// 두 상자(가운데 ± 크기/2)의 축별 겹침 비율 곱(da boxOverlap)·3D IoU(da boxIou) — 얇은 변은 min_ext 로 부풀림
DEV float box_ov(const float ac[3], const float ae[3], const float bc[3], const float be[3], float mn) {
  float r = 1.f;
  for (int k = 0; k < 3; ++k) {
    const float ha = 0.5f * maxf(ae[k], mn), hb = 0.5f * maxf(be[k], mn);
    const float ov = minf(ac[k] + ha, bc[k] + hb) - maxf(ac[k] - ha, bc[k] - hb);
    const float f = ov <= 0.f ? 0.f : minf(1.f, ov / (2.f * minf(ha, hb)));
    r = r * f;
    if (r <= 0.f) return 0.f;
  }
  return r;
}
DEV float box_iou(const float ac[3], const float ae[3], const float bc[3], const float be[3], float mn) {
  float ia = 1.f, va = 1.f, vb = 1.f;
  for (int k = 0; k < 3; ++k) {
    const float ea = maxf(ae[k], mn), eb = maxf(be[k], mn);
    const float ov = minf(ac[k] + 0.5f * ea, bc[k] + 0.5f * eb) - maxf(ac[k] - 0.5f * ea, bc[k] - 0.5f * eb);
    if (ov <= 0.f) return 0.f;
    ia = ia * ov; va = va * ea; vb = vb * eb;
  }
  return ia / (va + vb - ia);
}
// 그리퍼 각 → 잡는 점 손끝 틈 m(capi.cpp robotParams grip_gap: E0 쥔 각 1·2·3·4 cm, 그 위 finger_gap_hull), 사이는 선형, 밖은 끝값
DEV float grip_gap(float g) {
  constexpr int NG = 7;
  const float ga[NG] = {0.f, 0.095f, 0.231f, 0.347f, 0.408f, 0.5236f, 0.7854f};
  const float gw[NG] = {0.f, 0.01f, 0.02f, 0.03f, 0.04f, 0.0551f, 0.0933f};
  if (g <= ga[0]) return gw[0];
  for (int i = 1; i < NG; ++i)
    if (g <= ga[i]) return gw[i - 1] + (g - ga[i - 1]) / maxf(1e-12f, ga[i] - ga[i - 1]) * (gw[i] - gw[i - 1]);
  return gw[NG - 1];
}
DEV bool slot_big(const MapCore& m, const Slot& S) { return is_static_m(m, S.cls) || maxf(S.ext[0], maxf(S.ext[1], S.ext[2])) > MP::big; }   // objmap isBig
// 틈 gap 으로 쥔 그리퍼 사이에 S 가 있을 수 있나(objmap holdable): 큰 것·고정 종류 아님, 가운데 변 ≤ grasp_max_w, 빈손 아님, 틈이 폭과 맞음
DEV bool holdable(const MapCore& m, const Slot& S, float gap) {
  if (slot_big(m, S)) return false;
  float e0 = maxf(0.f, S.ext[0]), e1 = maxf(0.f, S.ext[1]), e2 = maxf(0.f, S.ext[2]), q;
  if (e0 > e1) { q = e0; e0 = e1; e1 = q; }
  if (e1 > e2) { q = e1; e1 = e2; e2 = q; }
  if (e0 > e1) { q = e0; e0 = e1; e1 = q; }
  if (e1 > MP::grasp_max_w) return false;
  if (gap < MP::grasp_min_gap) return false;
  return gap >= e0 - MP::grasp_w_tol && gap <= e2 + MP::grasp_w_tol;
}
// 선분 p0–p1 과 선분 q0–q1 의 거리² (Ericson 5.1.9, 겹치는 차례 고정 — CPU·GPU 같은 연산)
DEV float seg_seg_d2(const float p0[3], const float p1[3], const float q0[3], const float q1[3]) {
  float d1[3], d2[3], r[3];
  for (int k = 0; k < 3; ++k) { d1[k] = p1[k] - p0[k]; d2[k] = q1[k] - q0[k]; r[k] = p0[k] - q0[k]; }
  const float a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2], e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
  const float f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2];
  float sv = 0.f, tv = 0.f;
  if (a <= 1e-12f && e <= 1e-12f) { sv = 0.f; tv = 0.f; }
  else if (a <= 1e-12f) { sv = 0.f; tv = clampf(f / e, 0.f, 1.f); }
  else {
    const float c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2];
    if (e <= 1e-12f) { tv = 0.f; sv = clampf(-c / a, 0.f, 1.f); }
    else {
      const float b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
      const float den = a * e - b * b;
      sv = den > 1e-12f ? clampf((b * f - c * e) / den, 0.f, 1.f) : 0.f;
      tv = (b * sv + f) / e;
      if (tv < 0.f) { tv = 0.f; sv = clampf(-c / a, 0.f, 1.f); }
      else if (tv > 1.f) { tv = 1.f; sv = clampf((b - c) / a, 0.f, 1.f); }
    }
  }
  float d2s = 0.f;
  for (int k = 0; k < 3; ++k) { const float x = (p0[k] + d1[k] * sv) - (q0[k] + d2[k] * tv); d2s = d2s + x * x; }
  return d2s;
}
// 카메라(base_footprint 앞 cam_x, 높이 cam_z) → 점 b(base_footprint) 선분이 팔 캡슐을 지나나(scenemap objmap self_mask: 팔 캡슐 안 깊이 점은 버림 —
// 근사판은 팔을 그리지 않으므로, 팔이 가리는 시선을 보이지 않는 점·가림으로 셈)
DEV bool arm_blocks(const MapCore& m, const float b[3]) {
  if (!(m.arm_xmax + MP::self_r + MP::self_pad > env::K::cam_x)) return false;   // 팔이 카메라 앞에 없음
  if (m.arm_zmin - (MP::self_r + MP::self_pad) > maxf(env::K::cam_z, b[2])) return false;   // 카메라 앞 팔이 시선 선분보다 다 위(홈 자세 등)
  const float c[3] = {env::K::cam_x, 0.f, env::K::cam_z};
  const float rr = (MP::self_r + MP::self_pad) * (MP::self_r + MP::self_pad);
  for (int k = 0; k < 6; ++k) {
    const float* p = m.arm[k];
    const float* q = m.arm[k + 1];
    if (p[0] == q[0] && p[1] == q[1] && p[2] == q[2]) continue;
    if (seg_seg_d2(c, b, p, q) < rr) return true;
  }
  return false;
}

// ---- 1. 시작(판마다 한 스레드): 리셋, 오도메트리 표류, 움직임 거르기 -------------------------------------------------------------
// 돌려주는 값: B_KF(이번 스텝 keyframe) | B_RESET(판 리셋). MapCore 의 머리(prim 앞)만 바꾸고, 리셋이면 MapCore 전체를 새로 쓴다.
enum BeginFlag { B_KF = 1, B_RESET = 2, B_WALL = 4 };   // B_WALL: 들기·놓기로 벽 무시 영역이 바뀜 → 벽 선분 다시
constexpr int LIST_SHIFT = 29;                           // 장치 목록 낱말 = 판 번호 | 시작 결과 << 29

DEV float dist3(const float a[3], const float b[3]) {
  const float d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
  return sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
}
// ---- objprob 합치기 도움 함수(scenemap objprob.cpp·objmap.cpp objprob 길, CPU·GPU 같은 소스 — GPU_MAP_PORT 1절 R7–R12) ---------------
// exp: x = n·ln2 + r(|r| ≤ ln2/2), e^r 테일러 8 차, 2^n 은 지수 비트로. +·×·÷ 만(fma 없음) → CPU·GPU 비트 같음. |x| ≤ 80 으로 자름
DEV float expf_d(float x) {
  x = clampf(x, -80.f, 80.f);
  const float k = x * 1.44269504f;
  const int n = (int)(k < 0.f ? k - 0.5f : k + 0.5f);
  const float r = (x - (float)n * 0.693145752f) - (float)n * 1.42860677e-6f;
  float p = 2.48015873e-5f;
  p = p * r; p = p + 1.98412698e-4f;
  p = p * r; p = p + 1.38888889e-3f;
  p = p * r; p = p + 8.33333333e-3f;
  p = p * r; p = p + 4.16666667e-2f;
  p = p * r; p = p + 1.66666667e-1f;
  p = p * r; p = p + 0.5f;
  p = p * r; p = p + 1.f;
  p = p * r; p = p + 1.f;
  const uint32_t eb = (uint32_t)(n + 127) << 23;
  float sc;
#ifdef __CUDA_ARCH__
  sc = __uint_as_float(eb);
#else
  __builtin_memcpy(&sc, &eb, 4);
#endif
  return p * sc;
}
// objprob_math.h 의 수학(GPU·CPU 참조판 같은 비트): 결정적 다항식 exp·ln, 하드웨어 IEEE sqrt(--prec-sqrt), pow 는 지수 1 만(viewKappa occ 1)
struct OpmDet {
  DEV static float exp(float x) { return expf_d(x); }
  DEV static float log(float x) { return lnf_d(x); }
  DEV static float sqrt(float x) { return sqrtf(x); }
  DEV static float pow(float x, float y) { return y == 1.f ? x : expf_d(y * lnf_d(x)); }
};
DEV float sigm_d(float x) { return opm::sigmoid<OpmDet>(x); }
DEV int src_idx(int src) { return src >= 0 ? src : N_PRIM + (-1 - src); }   // 출처 → 캐시 자리(prim, 유령)
DEV int lab_parent(const BCtx& bx, int c) { return (bx.on && c >= 0 && c < bx.ss->nname) ? (int)bx.ss->hyper[c] : -1; }
// 비슷한 이름 k(0..2): BEHAVIOR = 이름 표의 비슷한 다른 이름 3(장면 묶음 sim3 — 이름 벡터가 가까운 것), 상자 방 = 다음 종류들 (가정)
DEV int lab_sim(const BCtx& bx, int c, int k) {
  if (bx.on) return (c >= 0 && c < bx.ss->nname) ? (int)bx.ss->sim3[c * 3 + k] : -1;
  return c >= 0 ? (c + 1 + k) % NCLS : -1;
}
// 출처(src: prim p ≥ 0, 유령 −1−g)의 이름 행(판 리셋 때 캐시)
DEV int src_name(const MapCore& m, int src) { return src > -32768 ? (int)m.snm[src_idx(src)] : -1; }
// 출처 둘의 원형 생김새 cos(인지 흉내 쪽 모형 — 진짜 SigLIP 2 원형 표는 P5): 같은 출처 1, 같은 이름 pe_cos_same, 비슷한 이름 pe_cos_sim, 그 밖 pe_cos_diff
DEV float proto_cos(const MapCore& m, const BCtx& bx, int a, int b) {
  if (a == b) return 1.f;
  (void)bx;
  if (a <= -32768 || b <= -32768) return MP::pe_cos_diff;
  const int ia = src_idx(a), ib = src_idx(b), na = m.snm[ia], nb = m.snm[ib];
  if (na < 0 || nb < 0) return MP::pe_cos_diff;
  if (na == nb) return MP::pe_cos_same;
  for (int k = 0; k < 3; ++k) if (m.ssim[ia][k] == nb || m.ssim[ib][k] == na) return MP::pe_cos_sim;
  return MP::pe_cos_diff;
}
DEV float mix_cos(const MapCore& m, const BCtx& bx, int a1, int a2, float wa, int b1, int b2, float wb) {
  float c = (1.f - wa) * (1.f - wb) * proto_cos(m, bx, a1, b1);
  if (wb > 0.f) c = c + (1.f - wa) * wb * proto_cos(m, bx, a1, b2);
  if (wa > 0.f) c = c + wa * (1.f - wb) * proto_cos(m, bx, a2, b1);
  if (wa > 0.f && wb > 0.f) c = c + wa * wb * proto_cos(m, bx, a2, b2);
  return c;
}
// vMF: 집중도 κ 인 평균의 원형 방향 cos 기대 ≈ 1/√(1 + (d − 1)/(2κ))
DEV float vmf_shrink(float K) { return K > 0.f ? 1.f / sqrtf(1.f + (MP::emb_d - 1.f) / (2.f * K)) : 0.f; }
// apCosMax(관측 z, 물체): 원형 섞임 cos × 두 쪽 κ 줄임 + 관측 흔들림
DEV float emb_cos_det(const MapCore& m, const BCtx& bx, const Det& D, const Slot& S) {
  return clampf(mix_cos(m, bx, D.src, D.src2, D.w2, S.src, S.src2, S.w2) * vmf_shrink(D.kappa) * vmf_shrink(S.K) + D.jit, -1.f, 1.f);
}
DEV float emb_cos_dets(const MapCore& m, const BCtx& bx, const Det& A, const Det& B) {   // apDot(q.z, o.z): 같은 영상 두 조각
  return clampf(mix_cos(m, bx, A.src, A.src2, A.w2, B.src, B.src2, B.w2) * vmf_shrink(A.kappa) * vmf_shrink(B.kappa) + A.jit + B.jit, -1.f, 1.f);
}
DEV float emb_cos_slots(const MapCore& m, const BCtx& bx, const Slot& A, const Slot& B) {   // apPairObj: μ 끼리
  return clampf(mix_cos(m, bx, A.src, A.src2, A.w2, B.src, B.src2, B.w2) * vmf_shrink(A.K) * vmf_shrink(B.K), -1.f, 1.f);
}
// 상자 둘의 같은 것 특징(objmap pairFeatures): a = 관측(또는 작은 쪽), b = 물체. contact < 0 이면 상자 꼴 접촉으로 셈
DEV float ap_contact(const float alo[3], const float ahi[3], const float blo[3], const float bhi[3]) {
  // a 상자 3 × 3 × 3 격자 표본 중 b 상자 ± ap_contact_d 안(축마다 따로 — 체비셰프 거리) 비율 = 축마다 {lo, 가운데, hi} 중 든 수의 곱 / 27
  int c = 1;
  for (int k = 0; k < 3; ++k) {
    const float l = blo[k] - MP::ap_contact_d, h = bhi[k] + MP::ap_contact_d, mid = 0.5f * (alo[k] + ahi[k]);
    c *= (int)(alo[k] >= l && alo[k] <= h) + (int)(mid >= l && mid <= h) + (int)(ahi[k] >= l && ahi[k] <= h);
  }
  return (float)c * (1.f / 27.f);
}
// f[7] = 접촉, 틈, 중심 거리/크기, cos − cos0(없으면 0), 겹침, 받침, 이름 겹침(0) → 로짓(받침이면 −30)
DEV float ap_feat_logit(const float alo[3], const float ahi[3], const float apos[3], const float blo[3], const float bhi[3], const float bpos[3], float cs,
                        float contact, bool merge, float f6, float f[7]) {
  f[0] = contact >= 0.f ? contact : ap_contact(alo, ahi, blo, bhi);
  float g2 = 0.f;
  for (int k = 0; k < 3; ++k) { const float g = maxf(0.f, maxf(alo[k] - bhi[k], blo[k] - ahi[k])); g2 = g2 + g * g; }
  opm::pair_geo<OpmDet>(alo, ahi, apos, blo, bhi, bpos, cs, MP::ap_cos0, f);   // 틈·중심 거리·cos·겹침·받침(scenemap 과 같은 식)
  f[6] = f6;
  const float w[8] = {MP::ap_w0, MP::ap_w1, MP::ap_w2, MP::ap_w3, MP::ap_w4, MP::ap_w5, MP::ap_w6, MP::ap_w7};
  const float wm[8] = {MP::ap_wm0, MP::ap_wm1, MP::ap_wm2, MP::ap_wm3, MP::ap_wm4, MP::ap_wm5, MP::ap_wm6, MP::ap_wm7};
  return opm::same_logit(merge ? wm : w, f);
}
DEV void slot_box(const Slot& S, float lo[3], float hi[3]) { for (int k = 0; k < 3; ++k) { lo[k] = S.pos[k] - 0.5f * S.ext[k]; hi[k] = S.pos[k] + 0.5f * S.ext[k]; } }
DEV void det_box(const Det& D, float lo[3], float hi[3]) { for (int k = 0; k < 3; ++k) { lo[k] = D.bc[k] - 0.5f * D.ext[k]; hi[k] = D.bc[k] + 0.5f * D.ext[k]; } }
// 이름 우도 더하기(apAddView 의 L += w·log p(c|z)): 든 라벨은 관측 값(상위 k 밖이면 llrest), 관측의 새 라벨은 "그동안 나머지 Σ + 이번 값"으로 넣고
// 칸이 차면 가장 작은 칸을 더 클 때만 바꿈. 나머지 Σ 에도 w·llrest
DEV void lab_add(Slot& S, const Det& D, float w) {
  for (int k = 0; k < NLAB; ++k) {
    if (S.lab[k] < 0) continue;
    float v = D.llrest;
    for (int j = 0; j < NLO; ++j) if (D.lab[j] == S.lab[k]) v = D.ll[j];
    S.L[k] = S.L[k] + opm::label_term(w, v);
  }
  const float rest0 = S.Lrest;
  for (int j = 0; j < NLO; ++j) {
    const int c = D.lab[j];
    if (c < 0) continue;
    bool have = false;
    for (int k = 0; k < NLAB; ++k) have = have || S.lab[k] == c;
    if (have) continue;
    const float val = rest0 + opm::label_term(w, D.ll[j]);
    int e = -1, lo = -1;
    for (int k = 0; k < NLAB; ++k) {
      if (S.lab[k] < 0) { if (e < 0) e = k; }
      else if (lo < 0 || S.L[k] < S.L[lo]) lo = k;
    }
    if (e >= 0) { S.lab[e] = (int16_t)c; S.L[e] = val; }
    else if (val > S.L[lo]) { S.lab[lo] = (int16_t)c; S.L[lo] = val; }
  }
  S.Lrest = rest0 + opm::label_term(w, D.llrest);
  S.lw = S.lw + w;
}
// 사후 확률(apName 앞부분, 사전 고름·크기 우도 없음 — 이름 표에 통계 없음): 든 라벨 + 나머지 라벨(C − 든 수 개, 같은 값)
DEV void ap_post(const Slot& S, int nlab, float post[NLAB], float& prest) {
  const float lam = opm::name_lambda(S.lw, MP::ap_name_wmax);
  float lp[NLAB];
  int nt = 0;
  for (int k = 0; k < NLAB; ++k) { lp[k] = S.lab[k] >= 0 ? lam * S.L[k] : -1e31f; nt += S.lab[k] >= 0; }
  opm::softmax_post<OpmDet>(lp, NLAB, lam * S.Lrest, nlab - nt, -1e30f, post, &prest);   // scenemap apName 과 같은 정규화(나머지 라벨 묶음 더함)
}
// 이름(apName 뒤: 최대 ≥ name_tau 면 그 라벨, 아니면 상위어 합 ≥ name_tau 인 것, 아니면 모름 −1). 이름 확신도 = 고른 이름 사후, 둘째 사후
DEV void ap_name(Slot& S, const MapCore& m, const BCtx& bx) {
  if (!(S.lw > 0.f)) return;
  float post[NLAB], pr;
  ap_post(S, m.nlab, post, pr);
  for (int k = 0; k < NLAB; ++k) S.post[k] = post[k];
  S.prest = pr;
  int b1 = -1, b2 = -1;
  for (int k = 0; k < NLAB; ++k) {
    if (S.lab[k] < 0) continue;
    if (b1 < 0 || post[k] > post[b1]) { b2 = b1; b1 = k; }
    else if (b2 < 0 || post[k] > post[b2]) b2 = k;
  }
  S.name_p2 = b2 >= 0 ? post[b2] : 0.f;
  if (b1 >= 0 && post[b1] >= MP::ap_name_tau) { S.cls = S.lab[b1]; S.name_p = post[b1]; return; }
  // 상위어 사슬(apName 과 같은 고르기 — objprob_math.h better_hyper): 든 라벨의 조상 a 마다 a 를 조상으로 둔 든 라벨 사후 합, 깊이(뿌리에서 거리)가 깊은 것, 같으면 합이 큰 것.
  // 든 라벨만 셈(나머지 라벨 묶음은 어느 조상 밑인지 모름 — 근사)
  auto depth_of = [&](int c) { int d = 0; for (int a = lab_parent(bx, c); a >= 0 && d < 8; a = lab_parent(bx, a)) ++d; return d; };
  int g = -1, gd = 0;
  float gp = 0.f;
  for (int k = 0; k < NLAB; ++k) {
    if (S.lab[k] < 0) continue;
    int a = lab_parent(bx, S.lab[k]);
    for (int lv = 0; a >= 0 && lv < 8; ++lv, a = lab_parent(bx, a)) {
      float u = 0.f;
      for (int j = 0; j < NLAB; ++j) {
        if (S.lab[j] < 0) continue;
        bool under = false;
        int b = lab_parent(bx, S.lab[j]);
        for (int l2 = 0; b >= 0 && l2 < 8 && !under; ++l2, b = lab_parent(bx, b)) under = b == a;
        if (under) u = u + post[j];
      }
      const int da = depth_of(a);
      if (opm::better_hyper(u, da, gp, gd, g >= 0, MP::ap_name_tau)) { g = a; gp = u; gd = da; }
    }
  }
  if (g >= 0) { S.cls = (int16_t)g; S.name_p = gp; return; }
  S.cls = -1;
  S.name_p = b1 >= 0 ? post[b1] : 0.f;
}
// 이름 분포 겹침(apPairObj f[6] = Σ√(p_a p_b) − 0.5): 같은 라벨끼리 + 나머지 라벨(둘 다 안 든 것)은 나머지 값끼리 (근사)
DEV float ap_bhat(const Slot& A, const Slot& B, int nlab) {   // 사후는 ap_name 이 둔 값
  const float *pa = A.post, *pb = B.post;
  const float ra = A.prest, rb = B.prest;
  // 두 쪽 라벨을 한 줄로 맞춤(A 의 라벨 다음 B 에만 있는 라벨, 없는 쪽은 나머지 값) → objprob_math.h bhattacharyya(나머지 라벨 묶음 포함)
  float xa[2 * NLAB], xb[2 * NLAB];
  int n = 0, only = 0;
  for (int k = 0; k < NLAB; ++k) {
    if (A.lab[k] < 0) continue;
    float q = rb;
    for (int j = 0; j < NLAB; ++j) if (B.lab[j] == A.lab[k]) q = pb[j];
    xa[n] = pa[k]; xb[n] = q; ++n;
  }
  const int na = n;
  for (int j = 0; j < NLAB; ++j) {
    if (B.lab[j] < 0) continue;
    bool in = false;
    for (int k = 0; k < NLAB; ++k) in = in || A.lab[k] == B.lab[j];
    if (!in) { xa[n] = ra; xb[n] = pb[j]; ++n; ++only; }
  }
  return opm::bhattacharyya<OpmDet, float, float>(xa, xb, n, ra, rb, nlab - (na + only));
}
// 생김새 출처 섞임 갱신(apAddView 의 r += κz 를 출처 몫으로): 관측 몫 (1 − w2)·k → src, w2·k → src2. 두 출처까지 들고, 큰 쪽이 주 출처
DEV void src_add(Slot& S, int src, float k) {
  if (!(k > 0.f)) return;
  const float m1 = (1.f - S.w2) * S.K, m2 = S.w2 * S.K;
  float a = m1, b = m2;
  int s1 = S.src, s2 = S.src2;
  if (src == s1) a = a + k;
  else if (src == s2) b = b + k;
  else if (b < k) { s2 = (int16_t)src; b = k; }   // 둘째 칸을 바꿈(작은 몫은 버림 — 근사)
  if (b > a) { const float t = a; a = b; b = t; const int ts = s1; s1 = s2; s2 = ts; }
  S.src = (int16_t)s1; S.src2 = (int16_t)s2;
  S.K = a + b;
  S.w2 = S.K > 0.f ? b / S.K : 0.f;
}
// 시점(카메라 xy, 광축 yaw)이 고리의 시점과 같나(inspect sameView: 0.3 m 안 그리고 15° 안)
DEV bool view_same(const Slot& S, int k, float cx, float cy, float yaw) {
  const float dx = cx - 0.01f * (float)S.vx[k], dy = cy - 0.01f * (float)S.vy[k];
  const float da = absf(wrap_pi(yaw - (float)S.vyaw[k] * 0.025f));
  return dx * dx + dy * dy <= MP::insp_view_d * MP::insp_view_d && da <= MP::insp_view_deg * 0.017453293f;
}
// 관측 하나를 물체에(apAddView + inspObserve): temper(같은 영상이 아닌 비슷한 시점이면 κ × 0.3), κ 합·출처 섞임·이름 우도, 가장 가까이 본 거리·시점 수
DEV void view_add(Slot& S, const Det& D, float cx, float cy, float yaw) {
  float k = D.kappa;
  int same = -1;
  for (int q = 0; q < S.vn && same < 0; ++q) if (view_same(S, q, cx, cy, yaw)) same = q;
  if (same >= 0 && !((S.vnew >> same) & 1u)) k = k * MP::ap_temper;
  src_add(S, D.src, (1.f - D.w2) * k);
  if (D.w2 > 0.f) src_add(S, D.src2, D.w2 * k);
  lab_add(S, D, k / MP::ap_kappa_ref);
  if (D.camd >= 0.f && (S.closest < 0.f || D.camd < S.closest)) S.closest = D.camd;
  if (same < 0 && S.n_views < MP::insp_view_cap) {   // 새 시점: 셈, 고리에(차면 가장 오래된 칸 — 8 넘으면 어림)
    S.n_views += 1;
    const int q = S.vn < NVIEW ? S.vn : (S.n_views - 1) % NVIEW;
    S.vx[q] = (int16_t)(cx * 100.f + (cx >= 0.f ? 0.5f : -0.5f));
    S.vy[q] = (int16_t)(cy * 100.f + (cy >= 0.f ? 0.5f : -0.5f));
    const float y40 = wrap_pi(yaw) * 40.f;
    S.vyaw[q] = (int8_t)(y40 >= 0.f ? y40 + 0.5f : y40 - 0.5f);
    S.vnew = (uint8_t)(S.vnew | (1u << q));
    if (S.vn < NVIEW) S.vn += 1;
  }
}
// 새 물체 칸의 objprob·살펴본 정도 값 비우기
DEV void ap_init(Slot& S) {
  for (int k = 0; k < NLAB; ++k) { S.lab[k] = -1; S.L[k] = 0.f; }
  S.Lrest = 0.f; S.lw = 0.f; S.K = 0.f; S.w2 = 0.f; S.src2 = -32768;
  S.name_p = 0.f; S.name_p2 = 0.f; S.closest = -1.f;
  S.vn = 0; S.n_views = 0; S.vnew = 0; S.tset = 0; S.top_bits = 0;
  for (int k = 0; k < NVIEW; ++k) { S.vx[k] = 0; S.vy[k] = 0; S.vyaw[k] = 0; }
  for (int k = 0; k < 4; ++k) S.tbox[k] = 0.f;
  for (int k = 0; k < 3; ++k) S.P[k] = 1.f;
  for (int k = 0; k < NLAB; ++k) S.post[k] = 0.f;
  S.prest = 0.f;
}
// 들기·놓기(objmap.cpp updateHands, 손 하나 = LIMO, grasp_check — scenemap d58c978). 잡는 점 = G1 순기구학 omx_end_effector_link 에서 링크 x 로
// −0.0119 m(URDF grasp_point, E0) → base_footprint → map(믿는 자세). 그리퍼 omx_gripper_joint_1 < grip_closed 인 채 grip_settle 동안 멈추면 한 번만
// 고른다: 잡는 점 grasp_r 안 가장 가까운 확정·안 든·사라짐 아닌 물체 중 들 수 있는 것(holdable: 큰 것·고정 종류·가운데 변 > 6 cm 아님,
// 손끝 틈(grip_gap 표) ≥ 5 mm 이고 폭 ± 2.5 cm). 든 뒤 끝까지 닫히면(틈 < 5 mm) 놓침, 열리면 놓는다(잡은 자리에서 moved_d 넘게 옮겼으면 옮겨짐).
// 든 동안 매 스텝 잡는 점을 따라간다. 받침 붙이기(parent)는 하지 않는다. 팔 뼈대(캡슐)도 여기서 base_footprint 로 둔다.
// 돌려주는 값: 들기·놓기가 있었으면 1(벽 무시 영역이 바뀜)
DEV void hand_release(MapCore& m) {
  Slot& S = m.slot[m.held_slot];
  S.held = 0;
  S.dirty = 1;
  const bool mv = dist3(S.pos, m.grasp_pos) > MP::moved_d;
  S.moved = (S.moved || mv) ? 1 : 0;
  S.state = S.moved ? S_MOVED : S_SEEN;
  S.misses = 0;
  m.held_slot = -1;
}
DEV int hands_step(MapCore& m, const EnvView& e) {
  bool same_q = m.t > 0;   // 판 첫 스텝은 늘 계산
  for (int k = 0; k < 5; ++k) same_q = same_q && m.fk_q[k] == e.q[k];
  if (!same_q) {
    float qd[env::N_Q];
    for (int k = 0; k < env::N_Q; ++k) qd[k] = 0.f;
    env::Fk f;
    env::fk(e.q, qd, f);
    for (int a = 0; a < 3; ++a) {
      m.eef_b[a] = f.ee_p[a];
      m.gp_b[a] = f.ee_p[a] + MP::grasp_off * f.ee_R[3 * a];   // 링크 x 축 = ee_R 첫 열(tools/omx_ws --grasp 와 같은 식)
    }
    const limo_omx::JointConst& J = limo_omx::joint(limo_omx::J_OMX_MOUNT_JOINT);
    for (int a = 0; a < 3; ++a) {
      m.arm[0][a] = J.t[a] + (a == 2 ? MP::base_z : 0.f);
      for (int k = 0; k < 5; ++k) m.arm[1 + k][a] = f.p[k][a] + (a == 2 ? MP::base_z : 0.f);
      m.arm[6][a] = f.ee_p[a] + (a == 2 ? MP::base_z : 0.f);
    }
    float xm = m.arm[0][0], zm = 1e30f;
    for (int k = 1; k < 7; ++k) xm = maxf(xm, m.arm[k][0]);
    for (int k = 0; k < 6; ++k)
      if (maxf(m.arm[k][0], m.arm[k + 1][0]) + MP::self_r + MP::self_pad > env::K::cam_x) zm = minf(zm, minf(m.arm[k][2], m.arm[k + 1][2]));
    m.arm_xmax = xm;
    m.arm_zmin = zm;
    for (int k = 0; k < 5; ++k) m.fk_q[k] = e.q[k];
  }
  float s, c;
  sincosf_d(m.eyaw, &s, &c);
  {
    const float bx = m.eef_b[0], by = m.eef_b[1], bz = m.eef_b[2] + MP::base_z;
    m.eef_m[0] = m.ex + (c * bx - s * by);
    m.eef_m[1] = m.ey + (s * bx + c * by);
    m.eef_m[2] = bz;
  }
  {
    const float bx = m.gp_b[0], by = m.gp_b[1], bz = m.gp_b[2] + MP::base_z;
    m.gp_m[0] = m.ex + (c * bx - s * by);
    m.gp_m[1] = m.ey + (s * bx + c * by);
    m.gp_m[2] = bz;
  }
  const float g = e.q[5];
  const int closed = g < MP::grip_closed ? 1 : 0;
  if (m.gref_t < 0 || absf(g - m.gref) > MP::grip_settle_eps) { m.gref = g; m.gref_t = m.t; }   // 쥠이 끝났나: grip_settle 동안 eps 안
  const bool settled = m.t - m.gref_t >= MP::grip_settle_steps;
  if (closed && !m.closed) m.tried = 0;   // 새로 닫힘
  const float gap = grip_gap(g);
  int ev = 0;
  if (closed && settled && !m.tried) {   // 잡기: 닫힌 채 멈춘 뒤 한 번
    m.tried = 1;
    int best = -1;
    float bd = MP::grasp_r;
    for (int b = 0; b < KSLOT; ++b) {
      const Slot& S = m.slot[b];
      if (!S.valid || !S.confirmed || S.held || S.state == S_GONE) continue;
      const float d = dist3(S.pos, m.gp_m);
      if (d < bd && holdable(m, S, gap)) { bd = d; best = b; }
    }
    if (best >= 0 && m.held_slot < 0) {
      Slot& S = m.slot[best];
      S.held = 1;
      m.held_slot = best;
      const float d0 = S.pos[0] - m.gp_m[0], d1 = S.pos[1] - m.gp_m[1], d2 = S.pos[2] - m.gp_m[2];
      m.held_rel[0] = c * d0 + s * d1;
      m.held_rel[1] = -s * d0 + c * d1;
      m.held_rel[2] = d2;
      for (int a = 0; a < 3; ++a) m.grasp_pos[a] = S.pos[a];
      S.state = S_HELD;
      m.n_grasp_total += 1;
      ev = 1;
    }
  } else if (closed && settled && gap < MP::grasp_min_gap && m.held_slot >= 0) {   // 든 뒤 끝까지 닫힘: 놓침
    hand_release(m);
    ev = 1;
  }
  if (!closed && m.closed && m.held_slot >= 0) {   // 놓기
    hand_release(m);
    ev = 1;
  }
  m.closed = closed;
  if (m.held_slot >= 0) {   // 든 물체는 잡는 점을 따라간다(잡는 점 기준 자리를 지금 yaw 로 돌림)
    Slot& S = m.slot[m.held_slot];
    const float* r = m.held_rel;
    S.pos[0] = m.gp_m[0] + (c * r[0] - s * r[1]);
    S.pos[1] = m.gp_m[1] + (s * r[0] + c * r[1]);
    S.pos[2] = m.gp_m[2] + r[2];
  }
  return ev;
}

DEV int phase_begin(MapCore& m, const EnvView& e, int force_kf, const bsc::SceneSet* ss = nullptr, BMapEnv* bm = nullptr) {
  int reset = 0;
  if (e.ep != m.ep) {
    reset_core(m, e, ss, bm);
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
  // 검증용 물체 옮기기(force_kf 의 8 비트 위 = 시작 스텝 S, 0 = 끔): 스텝 [S, S + 20) 동안 과제 물체(prim 0, BEHAVIOR 는 움직이는 상자일 때만)를
  // 방 가운데 쪽으로 0.04 m/스텝(0.4 m/s) 민다 — 지도만의 장면이라 동역학과 무관. 움직임 따라가기·사라짐·옮겨짐 잇기를 지나게(map_verify --shuffle)
  {
    const int S0 = force_kf >> 8;
    if (S0 > 0 && m.t >= S0 && m.t < S0 + 20 && (!bm || !bm->on || bm->sbox[0] < 0)) {
      Prim& P = m.prim[0];
      const float dx = 0.5f * (P.lo[0] + P.hi[0]) > 0.f ? -0.04f : 0.04f;
      P.lo[0] = P.lo[0] + dx; P.hi[0] = P.hi[0] + dx;
    }
    force_kf &= 0xff;
  }
  // 잡기 물리 판(E6): 집을 물체(prim 0, 움직이는 상자)를 환경의 참 자세로 — 들리면 손을 따라, 놓으면 앉은 자리(바깥 축 정렬 상자, Entry::odim·yaw)
  if (e.bpnp && ss && bm && bm->on && bm->sbox[0] < 0) {
    const bsc::Entry& E = ss->ent[e.bent];
    float so, co;
    sincosf_d(e.bo[3], &so, &co);
    const float hx = 0.5f * (absf(co) * E.odim[0] + absf(so) * E.odim[1]), hy = 0.5f * (absf(so) * E.odim[0] + absf(co) * E.odim[1]);
    Prim& P = m.prim[0];
    P.lo[0] = e.bo[0] - hx; P.hi[0] = e.bo[0] + hx;
    P.lo[1] = e.bo[1] - hy; P.hi[1] = e.bo[1] + hy;
    P.lo[2] = e.bo[2] - 0.5f * E.odim[2]; P.hi[2] = e.bo[2] + 0.5f * E.odim[2];
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
DEV int vis_point(const MapCore& m, const float o[3], float c, float s, const DetGeo& g, int pi, int q, const BCtx& bx) {
  const float xc = 0.5f * (g.xr[0] + g.xr[1]), yc = 0.5f * (g.yr[0] + g.yr[1]);
  const float hx = 0.4f * (g.xr[1] - g.xr[0]), hy = 0.4f * (g.yr[1] - g.yr[0]);
  const float xn = xc + (q == 1 ? -hx : q == 2 ? hx : 0.f), yn = yc + (q == 3 ? -hy : q == 4 ? hy : 0.f);
  float d[3];
  pix_dir(c, s, xn, yn, d);
  float inv[3];
  ray_inv(d, inv);
  if (bx.on) {   // BEHAVIOR: 자기 상자(정적이면 회전 상자) 앞을 정적 상자·다른 과제 물체가 가리나
    const bsc::SceneDev& sd = *bx.sd;
    const float ow[3] = {o[0] + bx.wx, o[1] + bx.wy, o[2]};
    const int own = bx.bm->sbox[pi];
    const float t = own >= 0 ? ray_obb_inv(ow, d, sd.box[own]) : ray_box_inv(o, d, inv, m.prim[pi]);
    if (!(t >= MP::ozmin && t <= MP::ozmax)) return 0;
    { const float pb[3] = {env::K::cam_x + t, -xn * t, env::K::cam_z - yn * t}; if (arm_blocks(m, pb)) return 0; }   // 팔이 가린 화소(self_mask)
    bool occ = false;
    bsc::bin_walk(sd, ow[0], ow[1], d[0], d[1], [&](int j, float) { if (!occ && j != own && ray_obb_inv(ow, d, sd.box[j]) < t) occ = true; },
                  [&]() { return occ ? -1.f : t; });
    if (occ) return 0;
    for (int j = 0; j < N_PRIM; ++j)
      if (j != pi && prim_dyn(bx, j) && ray_box_inv(o, d, inv, m.prim[j]) < t) return 0;
    return 1;
  }
  const float t = ray_box_inv(o, d, inv, m.prim[pi]);   // 광학 깊이(d 의 앞 성분 = 1)
  if (!(t >= MP::ozmin && t <= MP::ozmax)) return 0;
  { const float pb[3] = {env::K::cam_x + t, -xn * t, env::K::cam_z - yn * t}; if (arm_blocks(m, pb)) return 0; }   // 팔이 가린 화소(self_mask)
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
    if (x0 < -k.tanh || x1 > k.tanh || y0 < -k.tanv || y1 > k.tanv) g.inr |= 2;   // 상자가 영상 가장자리에 닿음(objmap Obs::trunc)
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

DEV void phase_cast(const MapCore& m, Scratch& sh, const EnvView& e, int tid, int nt, const BCtx& bx) {
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
    float tr[NROWC];
    uint32_t vert = 0u;   // 줄마다 수직면에 맞았나
    // 상자 하나(xy 판 구간 [t0, t1], z 범위)를 모든 줄에 — 줄마다 z 판만(상자 방·BEHAVIOR 같은 식)
    auto box_rows = [&](float t0, float t1, float z0, float z1, float& trmax) {
      if (t0 > t1) return;      // z 판은 구간을 좁히기만 하므로 이미 비면 어느 줄도 안 맞음
      if (t0 >= trmax) return;  // 들어가는 t ≥ 모든 줄의 지금 값: 어느 줄도 줄일 수 없음(minf 결과 같음)
      for (int r = 0; r < NROWC; ++r) {
        float u0 = t0, u1 = t1;
        if (!slab(o[2], -yn[r], invz[r], z0, z1, u0, u1)) continue;
        const float te = slab_end(u0, u1);
        if (te < tr[r]) {
          tr[r] = te;
          vert = (vert & ~(1u << r)) | ((u0 == t0 ? 1u : 0u) << r);   // xy 판으로 들어감 = 옆면
        }
      }
      trmax = tr[0];
      for (int r = 1; r < NROWC; ++r) trmax = maxf(trmax, tr[r]);
    };
    if (bx.on) {   // BEHAVIOR: 바닥·천장 + 정적 회전 상자(묶음 걷기, 깊이 범위 zmax 넘는 것은 결과가 같아 안 봄) + 과제 물체 상자
      for (int r = 0; r < NROWC; ++r) tr[r] = troomz[r];
      float trmax = tr[0];
      for (int r = 1; r < NROWC; ++r) trmax = maxf(trmax, tr[r]);
      const bsc::SceneDev& sd = *bx.sd;
      const float ow0 = o[0] + bx.wx, ow1 = o[1] + bx.wy;
      bsc::bin_walk(sd, ow0, ow1, d0, d1, [&](int j, float) {
        const bsc::SBox& b = sd.box[j];
        const float rx = ow0 - b.cx, ry = ow1 - b.cy;
        const float lx = b.c * rx + b.s * ry, ly = -b.s * rx + b.c * ry;
        const float ux = b.c * d0 + b.s * d1, uy = -b.s * d0 + b.c * d1;
        const float ix = absf(ux) < 1e-12f ? 0.f : 1.f / ux, iy = absf(uy) < 1e-12f ? 0.f : 1.f / uy;
        float t0 = -kInf, t1 = kInf;
        if (!slab(lx, ux, ix, -b.hx, b.hx, t0, t1) || !slab(ly, uy, iy, -b.hy, b.hy, t0, t1)) return;
        box_rows(t0, t1, b.z0, b.z1, trmax);
      }, [&]() { return minf(trmax, MP::zmax); });
      const float inv0 = absf(d0) < 1e-12f ? 0.f : 1.f / d0, inv1 = absf(d1) < 1e-12f ? 0.f : 1.f / d1;
      for (int p = 0; p < N_PRIM; ++p) {
        if (!prim_dyn(bx, p)) continue;
        const Prim& b = m.prim[p];
        float t0 = -kInf, t1 = kInf;
        if (!slab(o[0], d0, inv0, b.lo[0], b.hi[0], t0, t1) || !slab(o[1], d1, inv1, b.lo[1], b.hi[1], t0, t1)) continue;
        box_rows(t0, t1, b.lo[2], b.hi[2], trmax);
      }
    } else {
    float txy = kInf;
    if (d0 > 0.f) txy = minf(txy, (m.rhx - o[0]) / d0); else if (d0 < 0.f) txy = minf(txy, (-m.rhx - o[0]) / d0);
    if (d1 > 0.f) txy = minf(txy, (m.rhy - o[1]) / d1); else if (d1 < 0.f) txy = minf(txy, (-m.rhy - o[1]) / d1);
    for (int r = 0; r < NROWC; ++r) { tr[r] = minf(txy, troomz[r]); vert |= (txy < troomz[r] ? 1u : 0u) << r; }
    float trmax = tr[0];
    for (int r = 1; r < NROWC; ++r) trmax = maxf(trmax, tr[r]);
    const float inv0 = absf(d0) < 1e-12f ? 0.f : 1.f / d0, inv1 = absf(d1) < 1e-12f ? 0.f : 1.f / d1;
    for (int p = 0; p < N_PRIM; ++p) {
      const Prim& b = m.prim[p];
      float t0 = -kInf, t1 = kInf;
      if (!slab(o[0], d0, inv0, b.lo[0], b.hi[0], t0, t1) || !slab(o[1], d1, inv1, b.lo[1], b.hi[1], t0, t1)) continue;
      box_rows(t0, t1, b.lo[2], b.hi[2], trmax);
    }
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
DEV void obj_pre(const MapCore& m, Scratch& sh, int tid, int nt, const BCtx& bx) {
  int nh = 0;
  if (tid == 0) sh.more = 0;   // ap_merge_keys 가 OR
  for (int col = tid; col < NCOL; col += nt) nh += sh.colt[col] == 1;
  put_part(sh, tid, nh);
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  int ncand = 0;
  for (int p = 0; p < N_PRIM; ++p) ncand += sh.pcand[p];
  for (int i = tid; i < ncand * NPT; i += nt) {
    int p = 0;
    for (int j = i / NPT; ; ++p) if (sh.pcand[p] && j-- == 0) break;   // (i / NPT) 번째 후보
    if (vis_point(m, o, c, s, sh.geo[p], p, i % NPT, bx)) or_bits(&sh.vism[p], 1u << (i % NPT));   // vism 은 phase_cast 가 비움
  }
}

#include "percept.h"   // 인지 흉내 층(통계판): p_miss_at·put_det·view_kappa·slam_kf_correct·percept_stat

// 본 곳 칸(objmap markView·firstView): 창 0.5 m 칸 × 거리 띠 1..5 m 마다 처음 본 시각(0 = 아직, 아니면 min(255, 스텝/5 + 1) — 0.5 s 단위, 127 s 넘으면 255)
constexpr int VIEW_BYTES = ((VC * VC * VB + 15) / 16) * 16;
constexpr int VNS = 20;   // 열마다 표본 수 − 1 의 최댓값 = 띠 끝 5 m / 반 칸 0.25 m
static_assert(VNS * 0.5f * MP::view_cell == (float)VB, "view samples per column");
DEV int view_cell(float v) { return (int)floorf(v * (1.f / MP::view_cell)) + VC / 2; }
DEV int view_code(int t) { const int q = t / 5 + 1; return q > 255 ? 255 : q; }
// (x, y) 칸을 range(수평 m) 이하에서 처음 본 스텝(아래로 어림), 없으면 큰 값
DEV int first_view(const uint8_t* view, float x, float y, float range) {
  if (!view) return 1 << 30;
  const int cx = view_cell(x), cy = view_cell(y);
  if (cx < 0 || cy < 0 || cx >= VC || cy >= VC) return 1 << 30;
  int b = (int)ceilf(range) - 1;
  b = b < 0 ? 0 : (b > VB - 1 ? VB - 1 : b);
  const int q = view[(cy * VC + cx) * VB + b];
  return q == 0 ? (1 << 30) : (q - 1) * 5;
}
#include "objprob_gpu.h"   // objprob 합치기: ap_keys·ap_pick·ap_attach·ap_new·ap_merge_pass

// 3e. 부재 확인(objmap.cpp 4 + absentEvidence, scenemap 3ed710f). 칸마다 따로(모든 스레드, 칸마다 한 스레드).
//     확정·안 든·이번에 안 맞음·사라짐 아님(큰 것·고정 종류 포함), 잡는 점 hand_r + 0.1 밖. 근거: 상자 3×3×3 격자 27 점(근사판은 점 구름 없음)을
//     믿는 자세로 화소에 투영 → 그 화소의 참 깊이(참 카메라·참 장면, 팔이 가린 시선은 가림)로 시야 안·안 가림 ≥ 50 %, 보이는 부분 ≥ 12 px,
//     물체 카메라 깊이 ∈ [0.3, ozmax] 이고 검출한 가장 먼 깊이 × 1.15 + 0.2 안. 보이는 점의 30 % 이상에서 너머가 보이면 "자리 비었음"(2).
//     다른 이름(이름 표에 있던)으로 상자 안에 검출되면 안 셈. 새 근거만(지난 놓침 뒤 카메라 0.1 m·5° 넘게 바뀜 또는 2).
//     필요한 놓침 = min(60, max(gone_misses(큰 것 6), 검출률 p 로 (1−p)^k < 0.02 인 k)), 첫 놓침에서 2 s(큰 것 4 s) 또는 카메라 0.5 m 이동
DEV int fbits(float x) {   // 양수 float 의 비트(정수 순서 = 크기 순서) — 공유 메모리 최소·최댓값용
#ifdef __CUDA_ARCH__
  return __float_as_int(x);
#else
  int i; std::memcpy(&i, &x, 4); return i;
#endif
}
DEV float bitsf(int i) {
#ifdef __CUDA_ARCH__
  return __int_as_float(i);
#else
  float x; std::memcpy(&x, &i, 4); return x;
#endif
}
DEV void sh_add(int* p, int v) {
#ifdef __CUDA_ARCH__
  atomicAdd(p, v);
#else
  *p += v;
#endif
}
DEV void sh_min(int* p, int v) {
#ifdef __CUDA_ARCH__
  atomicMin(p, v);
#else
  *p = v < *p ? v : *p;
#endif
}
DEV void sh_max(int* p, int v) {
#ifdef __CUDA_ARCH__
  atomicMax(p, v);
#else
  *p = v > *p ? v : *p;
#endif
}
constexpr int AB_NS = 27;   // 상자 3 × 3 × 3 격자 표본
// (모든 스레드, 칸마다) 판단할 칸인가 + 가운데 깊이 문턱 → ab_ok, 모으기 칸 비움
DEV void absence_gate(const MapCore& m, Scratch& sh, int tid, int nt, int bug) {
  const float ec = sh.ec, es = sh.es;
  const float cxw = m.ex + ec * env::K::cam_x, cyw = m.ey + es * env::K::cam_x;
  for (int b = tid; b < KSLOT; b += nt) {
    const Slot& S = m.slot[b];
    int ok = S.valid && S.confirmed && !S.held && !sh.hit[b] && S.state != S_GONE && !(dist3(S.pos, m.gp_m) < MP::hand_r + 0.1f);
    if (ok) {
      const float rx = S.pos[0] - cxw, ry = S.pos[1] - cyw;
      const float zc = ec * rx + es * ry;
      if (zc < 0.3f || zc > MP::ozmax) ok = 0;
      else if (bug != 8 && S.max_det_z > 0.f && zc > MP::absent_det_k * S.max_det_z + 0.2f) ok = 0;   // 검출한 가장 먼 거리 밖
      else {   // 상자를 감싸는 구가 시야 밖이면 표본 27 개 모두 화면 밖(보이는 점 0 → 근거 없음과 같은 결과) — 투영을 건너뜀
        const Cam k = cam_consts();
        const float rho = 0.5f * sqrtf(S.ext[0] * S.ext[0] + S.ext[1] * S.ext[1] + S.ext[2] * S.ext[2]);
        const float left = -es * rx + ec * ry, up = S.pos[2] - env::K::cam_z;
        if (absf(left) - rho > (zc + rho) * k.tanh || absf(up) - rho > (zc + rho) * k.tanv) ok = 0;
      }
    }
    sh.ab_ok[b] = ok;
    sh.ab_vis[b] = 0; sh.ab_thru[b] = 0;
    sh.ab_u0[b] = fbits(3e38f); sh.ab_u1[b] = 0; sh.ab_v0[b] = fbits(3e38f); sh.ab_v1[b] = 0;
  }
}
// (모든 스레드, (칸, 표본) 쌍마다) 표본 하나를 믿는 자세로 화소에 투영 → 그 화소의 참 깊이(참 카메라·참 장면). 모으기는 정수 원자 덧셈·최소·최대(순서 무관)
DEV void absence_samples(const MapCore& m, Scratch& sh, int tid, int nt, const BCtx& bx) {
  const Cam k = cam_consts();
  const float ec = sh.ec, es = sh.es;
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  const float cxw = m.ex + ec * env::K::cam_x, cyw = m.ey + es * env::K::cam_x;
  for (int w = tid; w < KSLOT * AB_NS; w += nt) {
    const int b = w / AB_NS, q = w % AB_NS;
    if (!sh.ab_ok[b]) continue;
    const Slot& S = m.slot[b];
    const int ia = q / 9, ib = (q / 3) % 3, ic = q % 3;
    const float px = S.pos[0] - 0.5f * S.ext[0] + 0.5f * (float)ia * S.ext[0];
    const float py = S.pos[1] - 0.5f * S.ext[1] + 0.5f * (float)ib * S.ext[1];
    const float pz = S.pos[2] - 0.5f * S.ext[2] + 0.5f * (float)ic * S.ext[2];
    const float rx = px - cxw, ry = py - cyw;
    const float zc = ec * rx + es * ry, left = -es * rx + ec * ry, up = pz - env::K::cam_z;
    if (zc < 0.2f) continue;
    const float xn = -left / zc, yn = -up / zc;
    const float uf = k.fx * xn + k.cxp, vf = k.fx * yn + k.cyp;
    const int u = (int)uf, v = (int)vf;
    if (u < 2 || v < 2 || u >= MP::img_w - 2 || v >= MP::img_h - 2) continue;
    float d[3];
    pix_dir(c, s, xn, yn, d);
    const float tz = bx.on ? cast_beh(bx, m, o, d) : cast(m, o, d);
    if (!(tz >= MP::zmin && tz <= MP::zmax)) continue;   // 깊이 없음
    if (tz < zc - MP::occl) continue;                    // 앞에 다른 것(가림)
    { const float pb[3] = {env::K::cam_x + tz, -xn * tz, env::K::cam_z - yn * tz}; if (arm_blocks(m, pb)) continue; }   // 팔이 가림(팔 화소는 더 가까움)
    sh_add(&sh.ab_vis[b], 1);
    if (tz > zc + maxf(MP::occl, 0.1f)) sh_add(&sh.ab_thru[b], 1);   // 물체 자리 너머가 보임
    sh_min(&sh.ab_u0[b], fbits(uf)); sh_max(&sh.ab_u1[b], fbits(uf));
    sh_min(&sh.ab_v0[b], fbits(vf)); sh_max(&sh.ab_v1[b], fbits(vf));
  }
}
// (모든 스레드, 칸마다) 근거 → 놓침·사라짐
DEV void obj_absence(MapCore& m, const Scratch& sh, int tid, int nt, int bug) {   // objprob 길
  const int t = m.t;
  const float cxw = m.ex + sh.ec * env::K::cam_x, cyw = m.ey + sh.es * env::K::cam_x;
  for (int b = tid; b < KSLOT; b += nt) {
    if (!sh.ab_ok[b]) continue;
    Slot& S = m.slot[b];
    const int vis = sh.ab_vis[b], thru = sh.ab_thru[b];
    const int need_vis = (int)(MP::absent_vis * (float)AB_NS + 0.5f);
    if (vis < (need_vis > 3 ? need_vis : 3)) continue;
    const float du = bitsf(sh.ab_u1[b]) - bitsf(sh.ab_u0[b]), dv = bitsf(sh.ab_v1[b]) - bitsf(sh.ab_v0[b]);
    if (sqrtf(maxf(0.f, du) * maxf(0.f, dv)) < MP::absent_min_px) continue;
    const int ev = thru * 10 >= vis * 3 ? 2 : 1;   // 2: 보이는 점의 30 % 이상에서 너머가 보임(자리가 비었음)
    if ((sh.touch >> b) & 1u) continue;   // objprob: 다른 조각이 이 물체에 닿음(물체는 보임 — objmap update 4)
    if (bug != 8 && S.misses > 0 && ev != 2) {   // 새 근거만: 지난 놓침 뒤 카메라가 옮기거나 돌았을 때
      const float dx = cxw - S.lmiss_cam[0], dy = cyw - S.lmiss_cam[1];
      if (dx * dx + dy * dy < MP::gone_step_d * MP::gone_step_d && absf(wrap_pi(m.eyaw - S.lmiss_yaw)) < MP::gone_step_rad) continue;
    }
    S.lmiss_cam[0] = cxw; S.lmiss_cam[1] = cyw; S.lmiss_yaw = m.eyaw;
    const bool big = slot_big(m, S);
    S.n_vis_miss += 1;
    if (S.misses++ == 0) { S.first_miss = t; S.miss_cam[0] = cxw; S.miss_cam[1] = cyw; }
    // 검출률 p = (관측 + 1) / (관측 + 보이는데 놓침 + 2): (1 − p)^k < gone_eps 인 k 이상 놓쳐야
    const float pr = ((float)S.n_obs + 1.f) / ((float)S.n_obs + (float)S.n_vis_miss + 2.f);
    const int k_rate = pr >= 1.f ? 0 : (int)ceilf(MP::ln_gone_eps / lnf_d(1.f - pr));
    int need = big ? MP::gone_misses_big : MP::gone_misses;
    need = k_rate > need ? k_rate : need;
    need = need > 60 ? 60 : need;
    const int need_t = big ? MP::gone_min_steps_big : MP::gone_min_steps;
    const float mx = cxw - S.miss_cam[0], my = cyw - S.miss_cam[1];
    const bool indep = t - S.first_miss >= need_t || mx * mx + my * my >= MP::gone_view_d * MP::gone_view_d;
    if (S.misses >= need && indep) {
      S.state = S_GONE;
      S.n_vis_miss -= S.misses < S.n_vis_miss ? S.misses : S.n_vis_miss;   // 이 연속 놓침은 진짜 없음이었다(검출률에서 뺌)
    }
  }
}
// 3f(스레드 0): 옮겨짐 잇기(objmap relink — objprob 길은 생김새 cos) → 오래된 후보·헛검출 지우기. 물체끼리 병합은 다음 keyframe 앞(ap_merge_pass)
DEV void slot_free(MapCore& m, int b) {
  m.slot[b].valid = 0;
  m.slot[b].confirmed = 0;
}
DEV void obj_relink(MapCore& m, int bug, const BCtx& bx) {
  if (bug == 9) return;   // 음성 대조: 잇기 끔
  const int t = m.t;
  int usedm = 0, usedn = 0;
  for (;;) {   // 가까운 쌍부터 1:1 (거리, m id, n id 순 — scenemap 의 정렬과 같은 차례)
    int bm = -1, bn = -1;
    float bd = 0.f;
    for (int ni = 0; ni < KSLOT; ++ni) {
      const Slot& N = m.slot[ni];
      if (((usedn >> ni) & 1) || !N.valid || !N.appeared || !N.confirmed || N.held || N.state == S_GONE || N.n_obs < MP::link_min_obs) continue;
      if (t - N.first_seen > MP::link_window_steps) continue;
      for (int mi = 0; mi < KSLOT; ++mi) {
        const Slot& M = m.slot[mi];
        if (mi == ni || ((usedm >> mi) & 1) || !M.valid || M.state != S_GONE || !M.confirmed || M.held || (!M.moved && M.n_obs < MP::spurious_obs)) continue;
        if (emb_cos_slots(m, bx, M, N) < MP::ap_link_cos) continue;   // objprob: 이름 대신 생김새(μ 끼리 cos ≥ link_cos)
        if (!(N.first_seen > M.last_seen)) continue;
        const float d = dist3(N.pos, M.pos);
        if (d > minf(MP::link_max_d, MP::link_d0 + MP::link_v * (float)(N.first_seen - M.last_seen) * MP::tok_dt)) continue;
        bool wait = false;   // n 에 더 가까운 같은 이름 물체가 n 이 나타난 뒤 아직 안 보였으면 기다림
        if (t - N.first_seen < MP::link_wait_steps)
          for (int qi = 0; qi < KSLOT && !wait; ++qi) {
            const Slot& Q = m.slot[qi];
            if (qi == ni || qi == mi || !Q.valid || !Q.confirmed || Q.held || Q.state == S_GONE || Q.cls != N.cls) continue;
            wait = Q.last_seen < N.first_seen && dist3(Q.pos, N.pos) < d;
          }
        if (wait) continue;
        const bool better = bm < 0 || d < bd || (d == bd && (M.id < m.slot[bm].id || (M.id == m.slot[bm].id && N.id < m.slot[bn].id)));
        if (better) { bm = mi; bn = ni; bd = d; }
      }
    }
    if (bm < 0) break;
    usedm |= 1 << bm;
    usedn |= 1 << bn;
    Slot& M = m.slot[bm];
    const Slot& N = m.slot[bn];
    for (int q = 0; q < 3; ++q) { M.pos[q] = N.pos[q]; M.ext[q] = N.ext[q]; M.meas[q] = N.meas[q]; }
    M.n_obs += N.n_obs;
    M.last_seen = N.last_seen;
    M.last_kf = N.last_kf;
    M.score = maxf(M.score, N.score);
    M.max_det_z = maxf(M.max_det_z, N.max_det_z);
    M.trk_pos[0] = N.trk_pos[0]; M.trk_pos[1] = N.trk_pos[1]; M.trk_t = N.trk_t;
    M.mv_cnt = 0;
    M.appeared = 0;
    M.moved = 1;
    M.state = S_MOVED;
    M.misses = 0;
    M.dirty = 1;
    M.seen_len = N.seen_len; M.seen_rot = N.seen_rot;   // 생김새·이름(objprob 상태)은 M 그대로
    M.closest = N.closest; M.n_views = N.n_views; M.vn = N.vn; M.top_bits = N.top_bits; M.tset = N.tset;   // 살펴본 정도는 새 자리 것
    for (int q = 0; q < NVIEW; ++q) { M.vx[q] = N.vx[q]; M.vy[q] = N.vy[q]; M.vyaw[q] = N.vyaw[q]; }
    for (int q = 0; q < 4; ++q) M.tbox[q] = N.tbox[q];
    slot_free(m, bn);
    m.n_relink_total += 1;
  }
}
DEV void obj_prune(MapCore& m) {
  const int t = m.t;
  for (int b = 0; b < KSLOT; ++b) {
    const Slot& S = m.slot[b];
    if (!S.valid) continue;
    if (!S.confirmed ? t - S.last_seen > MP::prune_steps : (S.state == S_GONE && !S.moved && S.n_obs < MP::spurious_obs)) slot_free(m, b);
  }
}
// 3h(모든 스레드, 열마다): 본 곳 칸 표시(objmap markView). 열의 가장 먼 깊이 점까지(수평 ≤ 5 m) 믿는 카메라에서 반 칸 간격으로 칸을 지나며
// 거리 띠마다 처음 본 시각을 쓴다(이미 있으면 그대로 — 같은 스텝의 쓰기는 모두 같은 값이라 스레드 순서와 무관)
DEV void obj_view(const MapCore& m, const Scratch& sh, uint8_t* view, int tid, int nt) {
  if (!view) return;
  const Cam k = cam_consts();
  const float ec = sh.ec, es = sh.es;
  const float cxw = m.ex + ec * env::K::cam_x, cyw = m.ey + es * env::K::cam_x;
  const uint8_t code = (uint8_t)view_code(m.t);
  // 가로 32 점(scenemap markView: 깊이 영상 가로 32 간격) = 거친 열 64 중 하나 걸러. GPU 는 스레드 32.. 이 맡음(스레드 0 은 잇기·병합 중 — 결과 같음)
  constexpr int NVC = NCOL / 2;
  const int c0 = nt > NVC ? tid - (nt - NVC) : tid, cs = nt > NVC ? NVC : nt;
  for (int vc = c0; vc >= 0 && vc < NVC; vc += cs) {
    const int col = 2 * vc + 1;
    const float xn = ((float)(2 * col + 1 - NCOL) / (float)NCOL) * k.tanh;
    float tm = sh.colt[col] ? sh.colt_t[col] : 0.f;
    tm = maxf(tm, sh.cold1[col]);
    if (!(tm > 0.f)) continue;
    const float L = tm * sqrtf(1.f + xn * xn), Lm = minf(L, (float)VB);
    const float ux = (ec + es * xn) / sqrtf(1.f + xn * xn), uy = (es - ec * xn) / sqrtf(1.f + xn * xn);   // 믿는 자세의 수평 방향(단위)
    const int ns = (int)ceilf(Lm / (0.5f * MP::view_cell));   // ≤ 20 (Lm ≤ 5 m, 반 칸 0.25 m)
    // 1) 읽기만(서로 무관한 전역 읽기를 먼저 다 냄): 띠 b0 칸이 비었나 → 비트. 2) 빈 것만 씀. 같은 칸을 두 번 지나도 쓰는 값이 같아 결과 같음
    uint32_t need = 0u;
    for (int q = 0; q <= ns; ++q) {
      const float r = Lm * (float)q / (float)(ns > 1 ? ns : 1);
      const int cx = view_cell(cxw + r * ux), cy = view_cell(cyw + r * uy);
      if (cx < 0 || cy < 0 || cx >= VC || cy >= VC) continue;
      int b0 = (int)ceilf(r) - 1;
      b0 = b0 < 0 ? 0 : b0;
      if (view[(cy * VC + cx) * VB + b0] == 0) need |= 1u << q;   // 띠 b 를 쓰면 더 넓은 띠도 늘 같이(또는 먼저) 써짐 — 있으면 위도 다 있음
    }
    while (need) {
      const int q = ctz32(need);
      need &= need - 1u;
      const float r = Lm * (float)q / (float)(ns > 1 ? ns : 1);
      const int cx = view_cell(cxw + r * ux), cy = view_cell(cyw + r * uy);
      int b0 = (int)ceilf(r) - 1;
      b0 = b0 < 0 ? 0 : b0;
      uint8_t* e = view + (cy * VC + cx) * VB;
      for (int b = b0; b < VB; ++b) if (e[b] == 0) e[b] = code;
    }
  }
}

// 3g. 완성도(모든 스레드, 참 물체마다): 그 물체가 주 출처인 확정(사라짐 아님) 칸이 짝 문턱 안에 있나 → sh.found[p]
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
      if (!S.valid || !S.confirmed || S.state == S_GONE || S.src != p) continue;   // 정답 쪽 셈: 그 참 물체가 주 출처(인지 흉내 꼬리표)인 확정 칸
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
DEV void count_new_seen(const MapCore& m, int idx, int& cnt, int& cnt2, const BCtx& bx) {
  const float cxw = ((float)(idx % GW + GX0) + 0.5f) * RES, cyw = ((float)(idx / GW + GX0) + 0.5f) * RES;
  if (bx.on) {   // BEHAVIOR: 장면 방 격자(창 칸 = 장면 칸). 방마다 수는 BMapEnv 에 정수 원자 덧셈(순서 무관)
    const int sc = bx.bm->c0 + idx % GW, sr = bx.bm->r0 + idx / GW;
    const bsc::SceneDev& sd = *bx.sd;
    if (sc < 0 || sr < 0 || sc >= sd.W || sr >= sd.H) return;
    const int r = (int)sd.room[(size_t)sr * sd.W + sc] - 1;
    const int k = (r >= 0 && r < 32) ? (int)bx.bm->lut[r] : -1;
    if (k < 0) return;
    cnt += 1;
#ifdef __CUDA_ARCH__
    atomicAdd(&bx.bm->rseen[k], 1);
#else
    bx.bm->rseen[k] += 1;
#endif
    return;
  }
  const int k = room_of(m, cxw, cyw);
  cnt += k >= 0;
  cnt2 += k == 1 ? 1 : k == 2 ? (1 << 16) : 0;
}
DEV void phase_apply(const MapCore& m, Scratch& sh, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt, const BCtx& bx) {
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
          if (!(oo[j] & bit)) count_new_seen(m, w * 32 + lane, cnt, cnt2, bx);   // 새로 본 칸 중 방 안
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
      if (nw & bit) count_new_seen(m, idx, cnt, cnt2, bx);   // 새로 본 칸 중 방 안
    }
    const uint32_t no = (occ[w] & ~any) | ob;
    if (no != occ[w]) sh.occ_chg = 1;
    occ[w] = no;
  }
  put_part(sh, tid, cnt);
  put_part2(sh, tid, cnt2);
#endif
}

// 방 드러냄: 본 넓이 ≥ min_room_m2 이고 방 칸의 room_reveal 이상(한 번 드러나면 그대로). BEHAVIOR 는 창 방(BMapEnv)
DEV void reveal_beh(MapCore& m, const BCtx& bx) {
  const BMapEnv& b = *bx.bm;
  for (int k = 0; k < b.nroom; ++k)
    if (!((m.rrev >> k) & 1) && (float)b.rseen[k] * (RES * RES) >= MP::min_room_m2 && (float)b.rseen[k] >= MP::room_reveal * (float)b.rcells[k])
      m.rrev |= 1 << k;
}
DEV void phase_finish(MapCore& m, const Scratch& sh, const EnvView& e, int nt, const BCtx& bx) {
  const int s = sum_part(sh, nt);
  int n_obj = 0;
  for (int p = 0; p < N_PRIM; ++p) n_obj += sh.found[p];
  m.n_obj_conf = n_obj;
  m.n_task_conf = sh.found[0];
  m.n_seen_room += s;
  if (bx.on) reveal_beh(m, bx);
  else {  // 방마다 본 칸 → 드러냄: 본 넓이 ≥ min_room_m2 이고 방 칸의 room_reveal 이상(한 번 드러나면 그대로)
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

DEV void write_metrics(const MapCore& m, const EnvView& e, float* met, int N, int i, int nprim = N_PRIM) {
  const float dx = m.ex - e.x, dy = m.ey - e.y;
  met[M_TASK * N + i] = (float)m.n_task_conf;
  met[M_OBJ * N + i] = (float)m.n_obj_conf / (float)nprim;
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
DEV void map_keyframe(MapCore& m, Scratch& sh, const EnvView& e, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt, int bug, const Sync& sync,
                      const BCtx& bx, uint8_t* view = nullptr) {
  phase_cast(m, sh, e, tid, nt, bx);
  sync();
  PROF_MARK(P_CAST);
  obj_pre(m, sh, tid, nt, bx);
  sync();
  PROF_MARK(P_OBJPRE);
  ap_merge_keys(m, sh, tid, nt, bx, bug);   // objprob: 지난 keyframe 들이 쌓인 뒤 물체끼리 같은 것(이름 없이) — 합친 물체를 이번 관측이 바로 받게 먼저
  sync();
  if (tid == 0) {
    ap_merge_apply(m, sh, bx);
    PROF_MARK(P_MERGE);
    slam_kf_correct(m, sh, e, nt);
    percept_stat(m, sh, e, bx);      // 인지 흉내 층 → 검출 목록 sh.det
  }
  sync();
  PROF_MARK(P_POSE_DET);
  ap_keys(m, sh, tid, nt, bx, bug);
  sync();
  ap_pick(sh, tid, nt);
  sync();
  ap_attach(m, sh, tid, nt, bug, bx);
  sync();
  if (tid == 0) ap_new(m, sh, bug, view, bx);
  sync();
  PROF_MARK(P_ASSOC);
  absence_gate(m, sh, tid, nt, bug);
  sync();
  absence_samples(m, sh, tid, nt, bx);
  sync();
  obj_absence(m, sh, tid, nt, bug);
  sync();
  PROF_MARK(P_ABSENCE);
  // 순서: 사라짐 → 잇기 → 지우기(scenemap). 병합은 다음 keyframe 앞(ap_merge_pass)
  if (tid == 0) { obj_relink(m, bug, bx); obj_prune(m); }
  obj_view(m, sh, view, tid, nt);   // 칸 기억은 안 건드림(믿는 자세·시각만 읽음)
  sync();
  PROF_MARK(P_BEGIN);
  obj_complete(m, sh, tid, nt);
  phase_mark(m, sh, L, seen, occ, tid, nt, sync);
  sync();
  PROF_MARK(P_MARK);
  phase_apply(m, sh, L, seen, occ, tid, nt, bx);
  sync();
  PROF_MARK(P_APPLY);
  if (tid == 0) phase_finish(m, sh, e, nt, bx);
}

// ---- 7. 커리큘럼 처음 지도(계획서 5.5, 판 리셋 때만) ---------------------------------------------------------------------------
// 난수는 m.rng 를 섞은 따로 된 값(m.rng 를 움직이지 않음) — 모두 C2 면 지도 결과가 예전과 같다.
// (스레드 0) 단계 고르기 + 미리 확정할 참 물체를 칸에. 자리·크기는 참 상자(중심, 크기) 그대로 = 교사 지도의 "자세 오차만"
DEV void curr_slots(MapCore& m, const MapCurr& cu, int np = N_PRIM) {   // np: 쓰는 prim 수(BEHAVIOR 판은 시작 조건의 물체 수)
  uint64_t rr = m.rng ^ 0x6A09E667F3BCC909ull;
  const float u = rand01(rr);
  const int st = u < cu.p0 ? 0 : (u < cu.p0 + cu.p1 ? 1 : 2);
  m.init_stage = st;
  m.init_conf = 0;
  m.init_goal = 0;
  if (st == 2) return;
  int k = np;
  if (st == 1) {
    const int lo = cu.kmin < 0 ? 0 : (cu.kmin > np ? np : cu.kmin);
    const int hi = cu.kmax < lo ? lo : (cu.kmax > np ? np : cu.kmax);
    k = lo + (int)(rand01(rr) * (float)(hi - lo + 1));
    k = k > hi ? hi : k;
  }
  int idx[N_PRIM];
  for (int p = 0; p < N_PRIM; ++p) idx[p] = p;
  uint32_t pick = 0u;
  for (int j = 0; j < k; ++j) {   // 부분 Fisher–Yates: 서로 다른 k 개
    int r = j + (int)(rand01(rr) * (float)(np - j));
    r = r > np - 1 ? np - 1 : r;
    const int x = idx[j]; idx[j] = idx[r]; idx[r] = x;
    pick |= 1u << idx[j];
  }
  int b = 0;
  for (int p = 0; p < N_PRIM; ++p) {   // 칸은 참 물체 번호 순(빈 칸 앞에서부터)
    if (!((pick >> p) & 1u)) continue;
    const Prim& P = m.prim[p];
    Slot& S = m.slot[b++];
    S.valid = 1; S.id = m.next_id++; S.cls = P.cls;
    for (int a = 0; a < 3; ++a) { S.pos[a] = 0.5f * (P.lo[a] + P.hi[a]); S.ext[a] = P.hi[a] - P.lo[a]; S.first_pos[a] = S.pos[a]; S.meas[a] = S.pos[a]; }
    S.n_obs = MP::confirm; S.last_seen = 0; S.last_kf = -1; S.score = 0.9f;
    S.state = S_SEEN; S.confirmed = 1; S.moved = 0; S.misses = 0; S.first_miss = 0; S.held = 0; S.src = p;
    S.seen_len = 0.f; S.seen_rot = 0.f;
    S.first_seen = 0; S.appeared = 0; S.n_vis_miss = 0; S.mv_cnt = 0; S.trk_t = -1; S.moving_t = -100000; S.max_det_z = 0.f;
    ap_init(S);
    S.src = (int16_t)p; S.K = MP::confirm * MP::ap_kappa_ref; S.name_p = 1.f; S.dirty = 1;   // 이름·생김새는 참값(디버그 처음 지도)
  }
  m.init_conf = k;
  m.init_goal = (int)(pick & 1u);
}
// 공개한 칸인가(C0: 방 안·벽 칸 전부, C1: 미리 확정한 물체 중심 reveal_r 안)와 점유인가(방 벽 칸 또는 장면 상자 바닥 자국과 겹침).
// 벽 칸 = 방 경계 ±rh 를 담은 칸(광선이 벽에 맞는 칸과 같은 floor 규칙)
DEV int curr_cell(const MapCore& m, const MapCurr& cu, int ix, int iy, bool& occ_out, const BCtx& bx) {
  const float x0 = (float)ix * RES, y0 = (float)iy * RES, x1 = x0 + RES, y1 = y0 + RES;
  const float cx = x0 + 0.5f * RES, cy = y0 + 0.5f * RES;
  if (bx.on) {   // BEHAVIOR: 공개 = 장면 방 칸(C0) 또는 미리 확정한 물체 둘레(C1). 점유 = 장면 띠 점유 래스터(정적 상자) 또는 과제 물체 바닥 자국
    const bsc::SceneDev& sd = *bx.sd;
    const int sc = bx.bm->c0 + (ix - GX0), sr = bx.bm->r0 + (iy - GX0);
    if (sc < 0 || sr < 0 || sc >= sd.W || sr >= sd.H) return 0;
    const size_t si = (size_t)sr * sd.W + sc;
    if (sd.room[si] == 0) return 0;
    if (m.init_stage == 1) {
      bool near = false;
      const float r2 = cu.reveal_r * cu.reveal_r;
      for (int b = 0; b < m.init_conf && !near; ++b) {
        const float dx = cx - m.slot[b].pos[0], dy = cy - m.slot[b].pos[1];
        near = dx * dx + dy * dy <= r2;
      }
      if (!near) return 0;
    }
    bool occ = (sd.occ[si >> 5] >> (si & 31)) & 1u;
    for (int p = 0; p < N_PRIM && !occ; ++p) {
      if (!prim_dyn(bx, p)) continue;
      const Prim& P = m.prim[p];
      occ = P.lo[0] < x1 && P.hi[0] >= x0 && P.lo[1] < y1 && P.hi[1] >= y0 && P.hi[2] > MP::band_lo && P.lo[2] < MP::band_hi;
    }
    occ_out = occ;
    return 1;
  }
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
DEV void curr_grid(const MapCore& m, const MapCurr& cu, Scratch& sh, int16_t* L, uint32_t* seen, uint32_t* occ, int tid, int nt, const BCtx& bx) {
  int cnt = 0, cnt2 = 0;
  // 방(+ 경계 칸)과 겹치는 행·열 칸 범위 — curr_cell 의 겹침 판정과 같은 식. 밖의 낱말은 clear_grid 가 이미 0 으로 둠(같은 결과, 일만 줄임)
  int ylo = GW, yhi = -1, xlo = GW, xhi = -1;
  if (bx.on) { ylo = 0; yhi = GW - 1; xlo = 0; xhi = GW - 1; }   // BEHAVIOR: 창 전체(장면 칸으로 가림)
  else
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
      if (!curr_cell(m, cu, ix, iy, o, bx)) continue;
      sb |= 1u << b;
      const int Lv = o ? MP::curr_occ : MP::curr_free;
      L[idx] = (int16_t)Lv;
      if (Lv >= MP::q_occ) ob |= 1u << b;
      count_new_seen(m, idx, cnt, cnt2, bx);
    }
    seen[w] = sb;
    occ[w] = ob;
  }
  put_part(sh, tid, cnt);
  put_part2(sh, tid, cnt2);
}
// (스레드 0) 공개한 방 칸 수를 본 칸 수에 더함. 방 드러냄은 같은 스텝의 keyframe(phase_finish)이 같은 규칙으로 정함
DEV void curr_grid_finish(MapCore& m, const Scratch& sh, int nt, const BCtx& bx) {
  const int s = sum_part(sh, nt), s2 = sum_part2(sh, nt), r1 = s2 & 0xffff, r2 = s2 >> 16;
  m.n_seen_room += s;
  if (bx.on) return;   // BEHAVIOR: 방마다 본 칸은 count_new_seen 이 BMapEnv 에 이미 더함
  m.rseen[0] += s - r1 - r2;
  m.rseen[1] += r1;
  m.rseen[2] += r2;
}

// 시작 단계 뒤의 나머지(flags = phase_begin 결과): 리셋이면 격자 비움, keyframe 이면 갱신, keyframe·들기/놓기면 벽 선분, 마지막에 완성도 쓰기.
// GPU 는 목록(keyframe·리셋·벽)의 판만 블록으로 이 함수를 부르고(map_kf_kernel), 나머지 판은 시작 커널이 완성도만 쓴다.
// sh 와 ws 는 같은 자리(공용체): 벽 단계는 keyframe 단계가 끝난 뒤 Scratch 를 덮어쓴다.
struct MapGrid { int16_t* L; uint32_t* seen; uint32_t* occ; int16_t* segs; uint8_t* view = nullptr; };   // 한 판의 장치(또는 CPU) 배열. view: 본 곳 칸(VIEW_BYTES, 없으면 나타남 판정 끔)

// BEHAVIOR 판 리셋(블록): 창 칸의 장면 방을 세어 창 방 목록(≤ MAXBR, 장면 방 번호 순)·방 칸 수, 창 안 문(양쪽 창 방)을 만든다
template <class Sync>
DEV void room_setup_beh(MapCore& m, const BCtx& bx, int tid, int nt, const Sync& sync) {
  BMapEnv& b = *bx.bm;
  const bsc::SceneDev& sd = *bx.sd;
  for (int k = tid; k < 32; k += nt) b.rcnt[k] = 0;
  sync();
  for (int idx = tid; idx < NCELL; idx += nt) {
    const int sc = b.c0 + idx % GW, sr = b.r0 + idx / GW;
    if (sc < 0 || sr < 0 || sc >= sd.W || sr >= sd.H) continue;
    const int r = (int)sd.room[(size_t)sr * sd.W + sc] - 1;
    if (r < 0 || r >= 32) continue;
#ifdef __CUDA_ARCH__
    atomicAdd(&b.rcnt[r], 1);
#else
    b.rcnt[r] += 1;
#endif
  }
  sync();
  if (tid == 0) {
    int n = 0, tot = 0;
    for (int r = 0; r < 32; ++r) {
      b.lut[r] = -1;
      if (r < sd.nroom && b.rcnt[r] > 0 && n < bsc::MAXBR) {
        b.lut[r] = (int8_t)n; b.rid[n] = (int16_t)r; b.rcells[n] = b.rcnt[r]; b.rseen[n] = 0;
        tot += b.rcnt[r];
        ++n;
      }
    }
    for (int k = n; k < bsc::MAXBR; ++k) { b.rid[k] = -1; b.rcells[k] = 0; b.rseen[k] = 0; }
    b.nroom = n;
    m.n_room = n > 0 ? n : 1;
    m.room_cells = tot;
    m.rrev = 0;
    int nd = 0;
    for (int k = 0; k < sd.ndoor && nd < bsc::MAXBD; ++k) {
      const bsc::SDoor& D = sd.door[k];
      const float x = D.x - bx.wx, y = D.y - bx.wy;
      if (!(absf(x) < bsc::WIN_HALF && absf(y) < bsc::WIN_HALF)) continue;
      const int a = (D.ra >= 0 && D.ra < 32) ? (int)b.lut[D.ra] : -1, c = (D.rb >= 0 && D.rb < 32) ? (int)b.lut[D.rb] : -1;
      if (a < 0 && c < 0) continue;
      b.dx[nd] = x; b.dy[nd] = y; b.da[nd] = (int8_t)a; b.db[nd] = (int8_t)c;
      ++nd;
    }
    b.ndoor = nd;
  }
  sync();
}
union KfShared { Scratch sh; WallScratch ws; };
template <class Sync>
DEV void map_rest(MapCore& m, KfShared& u, const EnvView& e, const MapGrid& g, float* met, int N, int i,
                  int tid, int nt, int bug, int flags, const MapCurr& cu, const Sync& sync, const bsc::SceneSet* ss = nullptr, BMapEnv* bm = nullptr) {
  const BCtx bx = bctx(ss, bm);
  if (flags & B_RESET) {
    clear_grid(g.L, g.seen, g.occ, tid, nt);
    if (g.view) for (int k = tid; k < VIEW_BYTES / 4; k += nt) reinterpret_cast<uint32_t*>(g.view)[k] = 0u;
    if (bx.on) room_setup_beh(m, bx, tid, nt, sync);
    if (tid == 0) curr_slots(m, cu, bx.on ? bx.bm->nprim : N_PRIM);   // 처음 지도 단계(5.5)
    sync();
    if (m.init_stage < 2) {
      curr_grid(m, cu, u.sh, g.L, g.seen, g.occ, tid, nt, bx);
      sync();
      if (tid == 0) curr_grid_finish(m, u.sh, nt, bx);
      sync();
    }
    PROF_MARK(P_RESET);
  }
  if (flags & B_KF) map_keyframe(m, u.sh, e, g.L, g.seen, g.occ, tid, nt, bug, sync, bx, g.view);
  if (flags & (B_KF | B_WALL)) {
    const int chg = (flags & B_KF) ? u.sh.occ_chg : 0;   // ws 가 sh 를 덮기 전에 읽음
    sync();
    phase_walls(m, u.ws, g.occ, g.segs, chg, flags & B_KF, tid, nt, sync);
    PROF_MARK(P_WALLS);
  }
  if (tid == 0) write_metrics(m, e, met, N, i, bx.on ? bx.bm->nprim : N_PRIM);
  PROF_MARK(P_FINISH);
}

// 한 판 한 스텝 전체(CPU 참조판: tid 0, nt 1). GPU 는 같은 phase_begin → map_rest 를 두 커널로 나눠 부른다.
template <class Sync>
DEV void map_block(MapCore& m, KfShared& u, const EnvView& e, const MapGrid& g, float* met, int N, int i,
                   int tid, int nt, int bug, int force_kf, const MapCurr& cu, const Sync& sync, const bsc::SceneSet* ss = nullptr, BMapEnv* bm = nullptr) {
  const int flags = phase_begin(m, e, force_kf, ss, bm);
  map_rest(m, u, e, g, met, N, i, tid, nt, bug, flags, cu, sync, ss, bm);
}

// ---- 8. 다가가기 거리장(E2, BEHAVIOR 판만): **정책이 아는 지도**(믿는 점유 비트)로 목표에서 거꾸로 BFS ------------------------------
// 계획서 CURRICULUM_BEHAVIOR2026 2절: 다가가기 보상의 경로 거리는 그 순간 자란 지도로 잰다. 막힘 = 점유 칸을 8 이웃 1 칸 부풀림(몸통 반 폭 0.11 m ≈
// 1 칸 — 문이 0.8 m 면 0.6 m 길이 남음), 안 본 칸은 지나감. 씨앗 = 참 목표 둘레 반경(env_beh.h seed_r_of: B1 점 0.15, 물체 0.45 m) 안 칸 + 목표 칸(막힘에서 뺌).
// 단계는 4 이웃·8 이웃을 번갈아(홀수 단계 8, 짝수 4) 넓혀 팔각 거리 ≈ 단계 × 0.1 m. 판마다 nav_k 스텝에 한 번(판 리셋 때 바로).
// 저장은 그때 로봇 칸 둘레 NAV_P × NAV_P 조각(u8, 255 = 못 감/조각 밖)만: 로봇은 nav_k 스텝에 v_max·dt·nav_k 넘게 못 움직이므로(nav_k ≤ NAV_KMAX)
// 다음 계산까지 읽을 칸(로봇 둘레 5 × 5)이 조각 안에 있다. 같은 까닭으로 BFS 는 로봇 칸 둘레에 처음 닿은 단계 L_r + 여유(NAV_MARGIN) 에서 멈춘다.
// GPU 는 판 하나 = 워프 하나(레인마다 행 4 개, 행 = 128 비트), CPU 는 같은 집합 연산을 행마다. 단계 집합·멈춤은 차례와 무관
constexpr int NAV_K_DEF = 10;   // (가정) 측정은 README E2(nav_tradeoff, map_bench MAP_NAVK)
constexpr int NAV_KMAX = 20;
constexpr int NAV_LMAX = 254;
constexpr int NAV_P = 32;       // 조각 한 변(3.2 m)
constexpr int NAV_PH = NAV_P / 2;
static_assert(NAV_P == bsc::NAV_P, "patch size shared with env (bscene.h field_dist)");
DEV int nav_period(const MapCurr& cu) { const int k = cu.nav_k > 0 ? cu.nav_k : NAV_K_DEF; return k > NAV_KMAX ? NAV_KMAX : k; }
DEV bool nav_due(const MapCore& m, int tag, const MapCurr& cu) { return tag != m.ep || m.t % nav_period(cu) == 0; }
// 여유 단계: 로봇이 K 스텝에 갈 수 있는 거리(v_max·dt·K, 대각 1.1 배) + 5 × 5 읽기 반 폭 2 + 2
DEV int nav_margin(int K) { return (int)ceilf((float)K * env::K::v_max * 0.1f * 1.1f * INV_RES) + 4; }
struct R128 { uint64_t lo, hi; };   // 창 한 행 128 칸(열 c = 비트 c)
DEV R128 r_or(R128 a, R128 b) { return R128{a.lo | b.lo, a.hi | b.hi}; }
DEV R128 r_and(R128 a, R128 b) { return R128{a.lo & b.lo, a.hi & b.hi}; }
DEV R128 r_andn(R128 a, R128 b) { return R128{a.lo & ~b.lo, a.hi & ~b.hi}; }
DEV R128 r_up(R128 a) { return R128{a.lo << 1, (a.hi << 1) | (a.lo >> 63)}; }     // 열 + 1 쪽으로
DEV R128 r_dn(R128 a) { return R128{(a.lo >> 1) | (a.hi << 63), a.hi >> 1}; }     // 열 − 1 쪽으로
DEV R128 r_h3(R128 a) { return r_or(a, r_or(r_up(a), r_dn(a))); }
DEV bool r_any(R128 a) { return (a.lo | a.hi) != 0ull; }
DEV R128 r_cols_fast(int c0, int n);
DEV R128 r_cols(int c0, int c1) { return c1 < c0 ? R128{0ull, 0ull} : r_cols_fast(c0, c1 - c0 + 1); }   // 열 c0..c1 (창 안으로 자름)
DEV R128 occ_row128(const uint32_t* occ, int r) {
  return R128{(uint64_t)occ[r * 4] | ((uint64_t)occ[r * 4 + 1] << 32), (uint64_t)occ[r * 4 + 2] | ((uint64_t)occ[r * 4 + 3] << 32)};
}
// 행 r 의 씨앗: 칸 가운데가 (tx, ty) 에서 rad 안, + 목표가 든 칸
DEV R128 nav_seed_row(int r, float tx, float ty, float rad) {
  R128 o{0ull, 0ull};
  const float cy = ((float)r + 0.5f) * RES - (float)GW * 0.5f * RES, dy = cy - ty;
  if (dy * dy <= rad * rad) {
    const float hw = sqrtf(rad * rad - dy * dy);
    o = r_cols((int)ceilf((tx - hw) * INV_RES + (float)(GW / 2) - 0.5f), (int)floorf((tx + hw) * INV_RES + (float)(GW / 2) - 0.5f));
  }
  const int tc = (int)floorf(tx * INV_RES) + GW / 2, trow = (int)floorf(ty * INV_RES) + GW / 2;
  if (trow == r && tc >= 0 && tc < GW) { if (tc < 64) o.lo |= 1ull << tc; else o.hi |= 1ull << (tc - 64); }
  return o;
}
DEV int nav_cell(float v) { return (int)floorf(v * INV_RES) + GW / 2; }   // 창 좌표 → 창 칸(bsc::field_dist 와 같은 식)
DEV int ctz64(uint64_t v) {   // v != 0
#ifdef __CUDA_ARCH__
  return __ffsll((long long)v) - 1;
#else
  return __builtin_ctzll(v);
#endif
}
// 조각(원점 칸 pc0, pr0)에 행 r 의 비트 b 를 단계 L 로
DEV R128 r_cols_fast(int c0, int n) {   // 열 c0..c0+n−1 (n ≤ 64, 창 안으로 자름) — 고리 없이
  R128 o{0ull, 0ull};
  int a = c0 < 0 ? 0 : c0, b = c0 + n - 1 > GW - 1 ? GW - 1 : c0 + n - 1;
  if (a > b) return o;
  auto span = [](int lo, int hi) -> uint64_t {   // 비트 lo..hi (0 ≤ lo ≤ hi ≤ 63)
    const uint64_t up = hi >= 63 ? ~0ull : ((1ull << (hi + 1)) - 1ull);
    return up & ~((1ull << lo) - 1ull);
  };
  if (a < 64) o.lo = span(a, b < 64 ? b : 63);
  if (b >= 64) o.hi = span(a < 64 ? 0 : a - 64, b - 64);
  return o;
}
DEV void nav_write_row(uint8_t* lev, int pc0, int pr0, int r, R128 b, int L) {
  const int pr = r - pr0;
  if (pr < 0 || pr >= NAV_P) return;
  b = r_and(b, r_cols_fast(pc0, NAV_P));
  for (uint64_t x = b.lo; x; x &= x - 1ull) lev[pr * NAV_P + ctz64(x) - pc0] = (uint8_t)L;
  for (uint64_t x = b.hi; x; x &= x - 1ull) lev[pr * NAV_P + 64 + ctz64(x) - pc0] = (uint8_t)L;
}
// CPU 참조판(행마다 차례로). (rx, ry): 지금 로봇(참) 자리, 조각 원점은 돌려줌(org = 열 | 행 << 16, 음수면 16 비트 2 의 보수)
DEV int nav_org(int pc0, int pr0) { return (pc0 & 0xffff) | (pr0 << 16); }
DEV void nav_bfs_ref(const uint32_t* occ, float tx, float ty, float rad, float rx, float ry, int K, uint8_t* lev, int& org) {
  R128 blk[GW], fr[GW], vis[GW], h[GW], rob[GW];
  const int rc = nav_cell(rx), rr = nav_cell(ry), pc0 = rc - NAV_PH, pr0 = rr - NAV_PH, M = nav_margin(K);
  org = nav_org(pc0, pr0);
  for (int k = 0; k < NAV_P * NAV_P; ++k) lev[k] = 255;
  for (int r = 0; r < GW; ++r) {
    h[r] = r_h3(occ_row128(occ, r));
    rob[r] = (r >= rr - 2 && r <= rr + 2) ? r_cols(rc - 2, rc + 2) : R128{0ull, 0ull};   // 로봇 칸 둘레 5 × 5(읽는 칸)
  }
  int Lr = -1;
  for (int r = 0; r < GW; ++r) {
    R128 b = h[r];
    if (r > 0) b = r_or(b, h[r - 1]);
    if (r < GW - 1) b = r_or(b, h[r + 1]);
    const R128 sd = nav_seed_row(r, tx, ty, rad);
    blk[r] = r_andn(b, sd);
    fr[r] = sd;
    vis[r] = sd;
    nav_write_row(lev, pc0, pr0, r, sd, 0);
    if (r_any(r_and(sd, rob[r]))) Lr = 0;
  }
  for (int L = 1; L <= NAV_LMAX && !(Lr >= 0 && L > Lr + M); ++L) {
    R128 nw[GW];
    bool any = false, hitr = false;
    for (int r = 0; r < GW; ++r) h[r] = (L & 1) ? r_h3(fr[r]) : fr[r];
    for (int r = 0; r < GW; ++r) {
      R128 x = (L & 1) ? h[r] : r_or(r_up(fr[r]), r_dn(fr[r]));
      if (r > 0) x = r_or(x, h[r - 1]);
      if (r < GW - 1) x = r_or(x, h[r + 1]);
      nw[r] = r_andn(r_andn(x, blk[r]), vis[r]);
      any = any || r_any(nw[r]);
      hitr = hitr || r_any(r_and(nw[r], rob[r]));
    }
    if (!any) break;
    if (hitr && Lr < 0) Lr = L;
    for (int r = 0; r < GW; ++r) { vis[r] = r_or(vis[r], nw[r]); fr[r] = nw[r]; nav_write_row(lev, pc0, pr0, r, nw[r], L); }
  }
}

}  // namespace gmap

#include "map_tok.h"
