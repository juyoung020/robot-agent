// 관측 만들기(계획서 4.4, VLA_INPUT 2–4·6절): G1 관측 80 + 지도 토큰(MapTok v2 1,296 B) + 얼린 이름·생김새 표 → 신경망 입력.
// CPU 참조판과 GPU 커널이 같은 소스를 쓴다(비트 동일). 롤아웃(정책 앞 계산)과 갱신(미니배치 모으기)이 같은 함수·같은 열쇠를 써서
// 같은 (스텝, 판)이면 같은 입력 비트가 된다(학습 때 흔들기 포함).
//
//  X0 줄(432, bf16): [0,128) 집합(여기서 안 씀, 칸 MLP 뒤 집합 커널이 씀) | [128,208) G1 관측 80 | [208,264) 벽 56 | [264,274) 방 10 |
//                    [274,278) 완성도 4 | 278 = 1 | [279,287) 안 본 곳 광선 8 | [287,291) 다음 경유 지점 4 (둘 다 use_map 2 만) |
//                    [291,294) 스킬 B1·B2·B3 원-핫(지도 토큰 bkind, 상자 방 0) | 0 | [304,432) 지시문 128(지도 토큰 instr1 → pnp_v1 표 행, 없으면 0)
//                    [432,464) 목표 칸 2 × 16(지도 토큰 goal — 집을 것·놓을 곳, VLA_INPUT 2.1, goal_val 정규화)
//                    [464,976) 위에서 본 지도 교사 격자 2 × 16 × 16(지도 토큰 tv / 16 — 장애물·안 본 칸 비율, topview.h; 지도 값 — use_map 0·지도 끄기면 0)
//  스킬·지시문·목표 칸은 지도 값이 아니라 과제 값이라 use_map·지도 끄기 흔들기와 무관하게 늘 넣는다(교사·학생 같음 — POLICY 4.2 (가), README "지시문")
//  use_map: 0 = 지도 입력 모두 0, 1 = 지도 토큰(안 본 곳 광선·경유 지점 칸은 0), 2 = + 안 본 곳 광선 + 경유 지점
//  칸 줄 16 × 304(bf16): 숫자 33(정규화 v2) | 이름 뜻 128 | 생김새 128 | 289 = 1 | 0.  빈 칸은 모두 0
//  돌려주는 값: 채운 칸 비트(집합 평균·최댓값의 가림 — 칸 지우기 흔들기 뒤)
//
// 정규화 v2(VLA_INPUT 4절 "5 m 쯤에서 자르고 먼 값은 로그로 눌러 정규화, 통계는 학습 데이터로, 실제 로봇도 같은 값"):
//   길이(m) 값 x → sign(x)·ln(1 + min(|x|, 5) / 0.5) → (− μ) / σ. μ·σ 는 tok_norm.h(tools/tok_stats 가 시뮬 자료로 계산해 만든 헤더).
//   각도 /π, 시간·횟수 sign·ln(1 + |x|), 점수·속도는 표준화만, 0/1 값·확률은 그대로. 없는 값(빈 칸·안 맞은 선분·문 없음·경유 지점 없음)은 0.
//   G1 관측 80 은 예전처럼 ±10 자르기만(몸 상태는 VLA_INPUT 에 정규화 규칙이 없음 — 단위 그대로).
//
// 목표 출처(goal_mode, 계획서 5.4 교사 특권 한정):
//  0 = 특권(G3): G1 관측의 목표 값(참값)을 늘 넣는다.
//  1 = 지도: 목표 물체(컵)가 **지도에 확정된 경우만** 참값을 넣는다(이미 본 물체의 정확한 자세 = 교사 특권).
//      확정 여부 = 지도 토큰의 목표 칸(T_TARGET, 참 컵에 짝 문턱 안 확정 "컵" 칸 — map_tok.h). 아니면 아래 칸을 0 으로:
//      몸 상태의 "손끝 → 목표" 3 (G1 관측 53–55), 목표 칸 8 (72–79: 몸 기준 xy, 거리, 방위, 보임, 겉면, 카메라 거리, 표시).
//      목표 칸 79(표시)는 "목표를 앎" 1 / 0 이 된다. 지도 토큰 끔(use_map 0)이어도 목표는 지도에서 온다(지도는 늘 돈다).
//      점으로 가기(지도 토큰 flags 비트 1)는 점을 늘 안다(사용자가 준 점 — 특권 아님) → 늘 넣는다.
//  G1 관측 배치(env.h step_core)는 그대로이고 롤아웃 버퍼에는 참값을 둔다(보상·모양 잡기는 참값으로 — 관측이 아님).
//
// 학습 때 흔들기(VLA_INPUT 6절, 장치 값 ObsAug — 바꿔도 다시 잡기 없음, on 0 이면 예전과 같은 입력):
//   속도 잡음(관절·손끝·몸통 속도), 직전 명령 지우기·잡음, 이름 흔들기(같은 뜻 다른 말·상위어·비슷한 틀린 이름·지우기),
//   칸 지우기, 지도 통째로 비우기, 처음 보는 이름 평가(eval_unseen: 학습에 안 쓴 동의어로 바꿈 — VLA_INPUT 7절).
//   잡음은 균등 4 개 합(Irwin–Hall, 초월함수 없음)이라 CPU·GPU 비트가 같다. 열쇠 = (씨앗, key0, key1) — PPO 는 (바퀴, 스텝·판), BC 는 (갱신 스텝, 행).
#pragma once
#include "map.h"   // gmap::MapTok (map_tok.h), env 상수
#include "net.h"
#include "tok_norm.h"

namespace obsv {
using net::f2bf;

// ---- 얼린 128-d 표(장치 또는 호스트 배열 — 같은 구조체를 CPU 참조판도 씀) ----
// name: [n_name][128] bf16, app: [n_app][128] bf16, aux: [n_name][8] int32 = {상위어 행(−1), 동의어 무리 시작, 무리 크기, 비슷한 이름 3(−1), heldout, 종류}
struct VecTab {
  const uint16_t* name;
  const uint16_t* app;
  const int32_t* aux;
  int n_name, n_app;
  const uint16_t* pinstr;   // [n_pinstr][128] bf16: 집기·놓기 지시문(training/data/pnp_v1 instr128, 조합 × 문장 12) — X0 지시문 칸
  int n_pinstr;
};
constexpr int AUX_W = 8;
enum AuxCol { AX_HYPER = 0, AX_GS = 1, AX_GL = 2, AX_SIM = 3, AX_HELD = 6, AX_KIND = 7 };

// ---- 학습 때 흔들기(장치 값) ----
struct ObsAug {
  int on;                // 0 = 모두 끔(검증·교사 라벨·평가 기본)
  int eval_unseen;       // 1 = 이름 행을 같은 무리의 학습에 안 쓴(heldout) 이름으로(없으면 그대로) — 7절 처음 보는 이름 평가
  float vel_sigma;       // 속도 칸(관절 6·손끝 6·몸통 3) 잡음 σ(그 칸 단위: rad/s, m/s)
  float prev_drop;       // 직전 명령 8 을 통째로 0 으로 할 확률
  float prev_sigma;      // 직전 명령 잡음 σ(정규화 행동 단위)
  float p_erase;         // 이름 뜻 벡터를 0 으로(생김새만으로)
  float p_syn;           // 같은 무리의 다른 이름(학습용, heldout 뺌)
  float p_hyper;         // 상위어
  float p_wrong;         // 비슷한 틀린 이름(이름 벡터가 가까운 다른 무리)
  float p_slot_drop;     // 칸 지우기(덜 만들어진 지도)
  float p_map_off;       // 지도 토큰 통째로 비우기(카메라만으로)
  float p_goal_drop;     // 학생만(BC): 판·스텝마다 이 확률로 "목표인지" 칸 표시(T_TARGET)·목표 특권 값(G1 53–55·72–79)·경유 지점을 감춤 —
                         // 학생이 지시문(128-d)과 칸 이름 벡터(같은 공간)를 맞춰 목표 물체를 찾게. on 과 무관(0 = 끔). 교사(PPO)는 늘 0
  unsigned long long seed;
};
constexpr ObsAug kAugOff = {0, 0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0ull};

// 정규화(G1 관측·완성도): ±10 자름
NDEV float clamp10(float x) { return x < -10.f ? -10.f : (x > 10.f ? 10.f : (x != x ? 0.f : x)); }

// ---- 정규화 v2: 지도 토큰 값 하나 ----
// 특징 번호: 칸 숫자 0..32 | 벽 33..88 | 안 본 곳 광선 89..96 | 방 97..106 | 경유 지점 107..110 (tok_norm.h 의 μ·σ 자리)
constexpr int F_SLOT = 0, F_WALL = F_SLOT + gmap::TOK_SLOT_VALS, F_FRONT = F_WALL + gmap::N_WALL, F_ROOM = F_FRONT + gmap::N_FRONT,
              F_WAY = F_ROOM + gmap::N_ROOMTOK, N_FEAT = F_WAY + gmap::N_WAY;
static_assert(N_FEAT == tokn::N_FEAT, "token feature count = tok_norm.h");
enum FKind { K_RAW = 0, K_M = 1, K_Z = 2, K_LOG = 3, K_ANG = 4 };
constexpr float kDClip = 5.f, kDScale = 0.5f;
NDEV int feat_kind(int f) {
  using namespace gmap;
  if (f < F_WALL) {
    const int k = f - F_SLOT;
    if (k < T_DIST + 1 || (k >= T_EXT && k < T_EEF_S + 1) || (k >= T_DISP && k < T_DISP + 3) || k == T_UNC) return K_M;   // 위치·거리·크기·옮겨진 양·불확실도
    if (k == T_BEAR) return K_ANG;
    if ((k >= T_VEL && k < T_VEL + 3) || k == T_SCORE || k == T_CONF1 || k == T_CONF2) return K_Z;
    if (k == T_AGE || k == T_NOBS) return K_LOG;
    return K_RAW;   // 닿는지, 상태 4, 출처, 같은 방, 목표
  }
  if (f < F_FRONT) {
    const int k = f - F_WALL;
    if (k < 16) return K_M;                     // 광선(저장 /4 m)
    return ((k - 16) % 5) == 4 ? K_RAW : K_M;   // 선분 끝 둘(저장 /4 m) · 있음
  }
  if (f < F_ROOM) return K_M;                   // 안 본 곳 광선(저장 /4 m)
  if (f < F_WAY) { const int k = f - F_ROOM; return (k >= 6 && k <= 8) ? K_M : K_RAW; }   // 방 종류 확률 6, 문 x·y·거리, 문 있음
  return (f - F_WAY) < 3 ? K_M : K_RAW;         // 경유 지점 x·y·길이, 있음
}
NDEV float feat_scale(int f) {   // 저장 단위 → m (벽·안 본 곳 광선은 /4 m 로 저장됨)
  return (f >= F_WALL && f < F_ROOM) ? gmap::MP::wall_range : 1.f;
}
NDEV float clip_log(float x) {   // sign(x)·ln(1 + min(|x|, 5) / 0.5)
  const float a = dm::minf(dm::absf(x), kDClip);
  const float y = dm::lnf_d(1.f + a / kDScale);
  return x < 0.f ? -y : y;
}
// 표준화 전 값(통계를 낼 때도 이 값 — tools/tok_stats)
NDEV float feat_pre(int f, float raw) {
  const int k = feat_kind(f);
  if (raw != raw) return 0.f;
  if (k == K_M) return clip_log(raw * feat_scale(f));
  if (k == K_ANG) return raw * 0.31830988f;
  if (k == K_LOG) { const float y = dm::lnf_d(1.f + dm::absf(raw)); return raw < 0.f ? -y : y; }
  return raw;
}
NDEV float feat_norm(int f, float raw) {
  const int k = feat_kind(f);
  if (k == K_RAW) return clamp10(raw != raw ? 0.f : raw);
  return clamp10((feat_pre(f, raw) - tokn::mu(f)) / tokn::sd(f));
}
// 값이 "없음" 인가(정규화하지 않고 0): 선분 안 맞음·문 없음·경유 지점 없음
NDEV bool feat_absent(const gmap::MapTok& tok, int f) {
  if (f >= F_WALL + 16 && f < F_FRONT) { const int j = (f - F_WALL - 16) / 5; return net::h2f(tok.wall[16 + 5 * j + 4]) < 0.5f; }
  if (f >= F_ROOM + 6 && f <= F_ROOM + 8) return net::h2f(tok.room[9]) < 0.5f;
  if (f >= F_WAY && f < F_WAY + 3) return net::h2f(tok.way[3]) < 0.5f;
  return false;
}

static_assert(net::KSLOT == gmap::KSLOT, "slot count");
static_assert(net::SLOT_VALS == gmap::TOK_SLOT_VALS, "slot values");
static_assert(net::N_OBS_G1 == env::N_OBS, "G1 obs");
static_assert(net::OBS_W == env::N_OBS + gmap::N_WALL + gmap::N_ROOMTOK + gmap::N_COMP, "obs width");
static_assert(net::N_FRONT == gmap::N_FRONT && net::N_WAY == gmap::N_WAY, "unseen rays, waypoint");
static_assert(net::VEC_D == vlav::DIM, "name/appearance vector width = embed 128-d projection");
constexpr int G_EE_TGT = env::N_BODY - 3;               // 53: 손끝 → 목표 3
constexpr int G_TGT = env::N_BODY + env::N_RAYS;        // 72: 목표 칸 8 (79 = 표시)
constexpr int G_QD = env::N_Q;                          // 6..11 관절 속도
constexpr int G_EEV = G_QD + env::N_Q + 18 + 6;         // 36..41 손끝 선속도·각속도
constexpr int G_BV = G_EEV + 6;                         // 42..44 몸통 vx·vy·wz
constexpr int G_PREV = G_BV + 3;                        // 45..52 직전 명령
static_assert(G_PREV + env::N_ACT == G_EE_TGT, "body state 56 layout (VLA_INPUT 2절)");
static_assert(G_TGT + env::N_TGT == env::N_OBS, "target cell is the tail of the G1 obs");
NDEV bool vel_col(int c) { return (c >= G_QD && c < G_QD + env::N_Q) || (c >= G_EEV && c < G_BV + 3); }
// 지도 토큰에 목표 칸이 있나(목표 칸은 맨 앞 칸, T_TARGET = FP16 1.0)
NDEV bool goal_point_active(const gmap::MapTok& tok) { return (tok.flags & 2) != 0; }   // 지금 가는 목표 = 점(점으로 가기)
NDEV bool goal_known(const gmap::MapTok& tok) { return (tok.n_slot > 0 && tok.slot[0][gmap::T_TARGET] == 0x3c00u) || goal_point_active(tok); }
NDEV bool goal_col(int c) { return (c >= G_EE_TGT && c < G_EE_TGT + 3) || (c >= G_TGT && c < G_TGT + env::N_TGT); }

// ---- 흔들기 난수(열쇠 = 씨앗, key0, key1, 칸) ----
NDEV uint64_t aug_hash(const ObsAug& a, uint64_t k0, uint64_t k1, uint64_t c) { return net::hash4(a.seed, k0, k1, c); }
NDEV float aug_u(uint64_t h) { return (float)(h >> 40) * (1.0f / 16777216.0f); }   // [0, 1)
NDEV float aug_gauss(uint64_t h) {   // 균등 4 개 합(Irwin–Hall) → 평균 0, 분산 1
  float s = 0.f;
  for (int q = 0; q < 4; ++q) s = s + (float)((h >> (16 * q)) & 0xffffu) * (1.0f / 65536.0f);
  return (s - 2.f) * 1.7320508f;
}
// 목표 표시 감추기 결정(학생만, on 과 무관). 장치에서는 따로 부르는 함수(__noinline__): 이 해시를 assemble 안에 펼치면
// nvcc 12.8 -O3 가 같은 함수의 속도 잡음 칸을 틀리게 만들었다(실행마다 다른 값, -G 면 맞음 — README "지시문" 절, ppo_verify obs --aug 로 잡음)
#ifdef __CUDACC__
static __host__ __device__ __noinline__
#else
static inline
#endif
bool goal_off_hash(const ObsAug* a, uint64_t k0, uint64_t k1) {
  return a->p_goal_drop > 0.f && aug_u(aug_hash(*a, k0, k1, 0x47445250ull)) < a->p_goal_drop;
}
// 판(행) 하나의 흔들기 결정: 지도 끄기·직전 명령 지우기
struct AugRow { bool map_off, prev_off, goal_off; };
NDEV AugRow aug_row(const ObsAug* a, uint64_t k0, uint64_t k1) {
  AugRow r{false, false, false};
  if (!a) return r;
  // 목표 표시 감추기(학생만, on 과 무관): 감춘 판은 목표를 모르는 것과 같은 입력 + 칸 목표 표시 0 + 경유 지점 0
  r.goal_off = goal_off_hash(a, k0, k1);
  if (!a->on) return r;
  r.map_off = aug_u(aug_hash(*a, k0, k1, 0x4d4f4646ull)) < a->p_map_off;
  r.prev_off = aug_u(aug_hash(*a, k0, k1, 0x50524556ull)) < a->prev_drop;
  return r;
}
// 이름 행 흔들기(칸 b): 돌려주는 값 = 쓸 이름 행, −1 = 지우기(0 벡터)
NDEV int aug_name(const ObsAug* a, const VecTab& vt, int row, uint64_t k0, uint64_t k1, int b) {
  if (row < 0 || row >= vt.n_name) return -1;
  if (!a || !a->on) return row;
  const int32_t* x = vt.aux + (size_t)row * AUX_W;
  const uint64_t h = aug_hash(*a, k0, k1, 0x4e414d45ull + (uint64_t)b);
  const int gs = x[AX_GS], gl = x[AX_GL];
  if (a->eval_unseen) {   // 같은 무리의 heldout 이름 중 하나(열쇠로 고름), 없으면 그대로
    int nh = 0;
    for (int j = gs; j < gs + gl; ++j) nh += vt.aux[(size_t)j * AUX_W + AX_HELD] != 0;
    if (nh == 0) return row;
    int pick = (int)((h >> 8) % (uint64_t)nh);
    for (int j = gs; j < gs + gl; ++j)
      if (vt.aux[(size_t)j * AUX_W + AX_HELD] != 0 && pick-- == 0) return j;
    return row;
  }
  float u = aug_u(h);
  if (u < a->p_erase) return -1;
  u = u - a->p_erase;
  if (u < a->p_syn) {   // 같은 무리, heldout 뺌
    int ns = 0;
    for (int j = gs; j < gs + gl; ++j) ns += vt.aux[(size_t)j * AUX_W + AX_HELD] == 0;
    if (ns == 0) return row;
    int pick = (int)((h >> 8) % (uint64_t)ns);
    for (int j = gs; j < gs + gl; ++j)
      if (vt.aux[(size_t)j * AUX_W + AX_HELD] == 0 && pick-- == 0) return j;
    return row;
  }
  u = u - a->p_syn;
  if (u < a->p_hyper) return x[AX_HYPER] >= 0 ? x[AX_HYPER] : row;
  u = u - a->p_hyper;
  if (u < a->p_wrong) {
    const int j = x[AX_SIM + (int)((h >> 8) % 3ull)];
    return j >= 0 ? j : row;
  }
  return row;
}

// 목표 칸 값 하나(k = 칸 × 16 + GoalVal, 0..31) → 정규화 값(VLA_INPUT 2.1): 표시·꼴·앎·잃음·sin·cos 는 그대로, 위치 xyz·거리·손끝 기준 xyz 는
// 물체 칸과 같은 특징(T_POS·T_DIST·T_POS_EEF — tok_norm.h 같은 μ·σ, 길이 로그 누름). 위치를 모르면(GV_KNOWN 0) 위치 값 0, 칸이 없으면 모두 0.
// gdrop(학생 목표 감추기): **물체** 목표 칸은 통째로 0(목표 없음과 같은 입력), **점** 목표 칸은 감추지 않는다(점은 지시문에서 알 수 없음 — POLICY 4.4)
NDEV float goal_val(const gmap::MapTok& tok, int k, bool gdrop) {
  using namespace gmap;
  const int e = k / N_GV, q = k % N_GV;
  const uint16_t* g = tok.goal[e];
  if (g[GV_PRESENT] != 0x3c00u || (gdrop && g[GV_KOBJ] == 0x3c00u) || q >= GV_EEF + 3) return 0.f;
  const float raw = net::h2f(g[q]);
  if (q < GV_POS || q == GV_SIN || q == GV_COS) return clamp10(raw);
  if (g[GV_KNOWN] != 0x3c00u) return 0.f;
  const int f = q == GV_DIST ? F_SLOT + T_DIST : q < GV_DIST ? F_SLOT + T_POS + (q - GV_POS) : F_SLOT + T_POS_EEF + (q - GV_EEF);
  return feat_norm(f, raw);
}
static_assert(net::N_GOAL_E == gmap::N_GENT && net::N_GOAL_V == gmap::N_GV, "goal entries in X0 = map token goal entries");
static_assert(net::N_TV == gmap::TV_NCH * gmap::TV_B * gmap::TV_B, "top-view teacher grid in X0 = map token tv");

// 칸 숫자 하나(칸 b, 숫자 c < 33) → 정규화 값
NDEV float slot_num(const gmap::MapTok& tok, int b, int c, bool gdrop = false) {
  return (gdrop && c == gmap::T_TARGET) ? 0.f : feat_norm(F_SLOT + c, net::h2f(tok.slot[b][c]));
}
// 목표 표시 감추기(학생 흔들기, 열쇠 = 판·스텝) — aug_row().goal_off 와 같은 값(검증 도구용)
NDEV bool goal_drop(const ObsAug* a, uint64_t k0, uint64_t k1) { return aug_row(a, k0, k1).goal_off; }

// 지도 칸 수(use_map 0 이면 0)
NDEV int slot_count(const gmap::MapTok& tok, int use_map) { return use_map ? (tok.n_slot < 0 ? 0 : (tok.n_slot > net::KSLOT ? net::KSLOT : tok.n_slot)) : 0; }

// bf16 8 개를 한 번에(16 B 정렬 자리)
NDEV void put8(uint16_t* d, const uint16_t h[8]) {
#ifdef __CUDA_ARCH__
  *reinterpret_cast<uint4*>(d) = make_uint4(h[0] | ((uint32_t)h[1] << 16), h[2] | ((uint32_t)h[3] << 16), h[4] | ((uint32_t)h[5] << 16), h[6] | ((uint32_t)h[7] << 16));
#else
  for (int e = 0; e < 8; ++e) d[e] = h[e];
#endif
}
static_assert((net::X0_W - net::X0_OBS) % 8 == 0 && net::X0_OBS % 8 == 0 && net::SLOT_IN % 8 == 0, "16 B pieces");
static_assert(net::SLOT_NAME + net::VEC_D == net::SLOT_APP && net::SLOT_APP + net::VEC_D == net::SLOT_BIAS, "slot row layout");

// 판 하나의 입력을 레인 nl 개가 나눠 쓴다(lane = 0..nl-1). CPU 는 nl = 1.
// obs: G1 관측 [k*stride + i]. x0: 이 판의 X0 줄. srows: 이 판의 칸 줄 16 × 304 (nullptr 이면 칸 줄은 쓰지 않음 — 채운 칸 비트만 돌려줌).
// vt: 얼린 이름·생김새 표. aug/k0/k1: 흔들기(nullptr = 끔)와 열쇠. use_map = 0 이면 지도 입력을 모두 0 으로
// compact: srows 를 칸 16 × SLOT_C(40) 줄인 줄로(표 행은 번호만, net.h) — PPO 학습기가 씀. 펼친 값은 304 칸 줄과 같다
NDEV uint32_t assemble(const float* obs, int stride, int i, const gmap::MapTok& tok, uint16_t* x0, uint16_t* srows, int use_map, int goal_mode,
                       int lane, int nl, const VecTab& vt, const ObsAug* aug = nullptr, uint64_t k0 = 0, uint64_t k1 = 0, bool compact = false) {
  const AugRow ar = aug_row(aug, k0, k1);
  const bool gdrop = ar.goal_off;
  const bool gdrop_g1 = gdrop && !goal_point_active(tok);   // 점으로 가기면 G1 목표 값·경유 지점은 점 쪽이라 감추지 않음
  const bool show = !gdrop_g1 && (goal_mode == 0 || goal_known(tok));
  const int um = ar.map_off ? 0 : use_map;
  const bool on = aug && aug->on;
  // 8 칸(16 B)씩: 레인마다 덩이 하나를 계산해 한 번에 씀
  auto x0v = [&](int c) -> uint16_t {
    const int col = net::X0_OBS + c;
    float v = 0.f;
    if (c < net::N_OBS_G1) {
      v = (show || !goal_col(c)) ? obs[(size_t)c * stride + i] : 0.f;
      if (on) {
        const bool pv = c >= G_PREV && c < G_PREV + env::N_ACT, vl = vel_col(c);
        const float sg = pv ? aug->prev_sigma : vl ? aug->vel_sigma : 0.f;
        const float nz = sg * aug_gauss(aug_hash(*aug, k0, k1, (pv ? 0x50524500ull : 0x56454c00ull) + (uint64_t)c));
        v = (pv && ar.prev_off) ? 0.f : v + nz;
      }
      v = clamp10(v);
    } else if (um && c < net::N_OBS_G1 + gmap::N_WALL) {
      const int f = F_WALL + c - net::N_OBS_G1;
      v = feat_absent(tok, f) ? 0.f : feat_norm(f, net::h2f(tok.wall[c - net::N_OBS_G1]));
    } else if (um && c < net::N_OBS_G1 + gmap::N_WALL + gmap::N_ROOMTOK) {
      const int k = c - net::N_OBS_G1 - gmap::N_WALL, f = F_ROOM + k;
      v = feat_absent(tok, f) ? 0.f : feat_norm(f, net::h2f(tok.room[k]));
    } else if (um && c < net::OBS_W) v = clamp10(net::h2f(tok.comp[c - net::N_OBS_G1 - gmap::N_WALL - gmap::N_ROOMTOK]));
    else if (col == net::X0_BIAS) v = 1.f;
    else if (um >= 2 && col >= net::X0_FRONT && col < net::X0_FRONT + net::N_FRONT) v = feat_norm(F_FRONT + col - net::X0_FRONT, net::h2f(tok.front[col - net::X0_FRONT]));
    else if (um >= 2 && col >= net::X0_WAY && col < net::X0_WAY + net::N_WAY) {
      const int f = F_WAY + col - net::X0_WAY;
      v = feat_absent(tok, f) ? 0.f : feat_norm(f, net::h2f(tok.way[col - net::X0_WAY]));
    }
    return f2bf(v);
  };
  // 지시문 행(표 밖이면 없음)·스킬
  const int irow = (int)tok.instr1 - 1;
  const uint16_t* isrc = (irow >= 0 && irow < vt.n_pinstr && vt.pinstr) ? vt.pinstr + (size_t)irow * net::VEC_D : nullptr;
  const int skill = (int)tok.bkind;
  for (int q = lane; q < (net::X0_W - net::X0_OBS) / 8; q += nl) {
    uint16_t h[8];
    const int col0 = net::X0_OBS + q * 8;
    for (int e = 0; e < 8; ++e) h[e] = x0v(q * 8 + e);
    if (col0 >= net::X0_TV) {   // 교사 격자: 표본 수 / 4(bf16 에 정확)
      for (int e = 0; e < 8; ++e) {
        const int k = col0 - net::X0_TV + e, ch = k / (gmap::TV_B * gmap::TV_B), b = k % (gmap::TV_B * gmap::TV_B);
        h[e] = um ? f2bf((float)tok.tv[ch][b / gmap::TV_B][b % gmap::TV_B] * (1.f / (float)(gmap::TV_BS * gmap::TV_BS))) : (uint16_t)0;
      }
    } else if (col0 >= net::X0_GOAL) {   // 목표 칸 2 × 16
      for (int e = 0; e < 8; ++e) h[e] = f2bf(goal_val(tok, col0 - net::X0_GOAL + e, gdrop));
    } else if (col0 >= net::X0_INSTR) {   // 지시문 128: 표 행 bf16 그대로
      for (int e = 0; e < 8; ++e) h[e] = isrc ? isrc[col0 - net::X0_INSTR + e] : (uint16_t)0;
    } else if (col0 + 8 > net::X0_SKILL && col0 < net::X0_SKILL + net::N_SKILL) {   // 스킬 원-핫
      for (int e = 0; e < 8; ++e) {
        const int col = col0 + e;
        if (col >= net::X0_SKILL && col < net::X0_SKILL + net::N_SKILL) h[e] = f2bf(skill == 1 + col - net::X0_SKILL ? 1.f : 0.f);
      }
    }
    if (gdrop_g1 && col0 + 8 > net::X0_WAY && col0 < net::X0_WAY + net::N_WAY)   // 목표 감춤: 경유 지점 0(덩이 둘에 걸침)
      for (int e = 0; e < 8; ++e) if (col0 + e >= net::X0_WAY && col0 + e < net::X0_WAY + net::N_WAY) h[e] = 0;
    put8(x0 + net::X0_OBS + q * 8, h);
  }
  const int ns = slot_count(tok, um);
  uint32_t mk = ns >= 32 ? 0xffffffffu : ((1u << ns) - 1u);
  if (on && aug->p_slot_drop > 0.f)
    for (int b = 0; b < ns; ++b)
      if (aug_u(aug_hash(*aug, k0, k1, 0x534c4f54ull + (uint64_t)b)) < aug->p_slot_drop) mk &= ~(1u << b);
  if (srows && compact)   // 줄인 칸 줄(net.h SLOT_C): 펼치면(net::slot_col) 아래 304 칸 줄과 비트가 같다
    for (int q = lane; q < net::KSLOT * net::SLOT_C / 8; q += nl) {
      const int b = q / (net::SLOT_C / 8), c0 = (q % (net::SLOT_C / 8)) * 8;
      const bool live = (mk >> b) & 1u;
      uint16_t h[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (live) {
        if (c0 + 8 <= net::SLOT_VALS) {
          for (int e = 0; e < 8; ++e) h[e] = f2bf(slot_num(tok, b, c0 + e, gdrop));
        } else {
          static_assert(net::SLOT_VALS == 33 && net::SLOT_C == 40, "compact row: chunk 4 = num 32, bias, name, app");
          h[0] = f2bf(slot_num(tok, b, net::SLOT_VALS - 1, gdrop));
          h[1] = f2bf(1.f);
          const int nr = aug_name(aug, vt, tok.name_id[b], k0, k1, b);
          h[2] = (uint16_t)(nr >= 0 ? nr + 1 : 0);
          const int ar2 = tok.app_id[b];
          h[3] = (uint16_t)(ar2 >= 0 && ar2 < vt.n_app ? ar2 + 1 : 0);
        }
      }
      put8(srows + q * 8, h);
    }
  else if (srows)
    for (int q = lane; q < net::KSLOT * net::SLOT_IN / 8; q += nl) {
      const int b = q / (net::SLOT_IN / 8), c0 = (q % (net::SLOT_IN / 8)) * 8;
      uint16_t h[8];
      const bool live = (mk >> b) & 1u;
      // 칸마다 출처를 칸 번호로 고른다(16 B 덩이가 표 경계 33·161 에 걸쳐도 맞게 — v2 첫 판은 덩이 단위로 골라 칸 33..39 가 0,
      // 161..167 이 이름 표 다음 행이었음, README)
      const bool has_n = live && c0 + 8 > net::SLOT_NAME && c0 < net::SLOT_APP, has_a = live && c0 + 8 > net::SLOT_APP && c0 < net::SLOT_BIAS;
      const int nr = has_n ? aug_name(aug, vt, tok.name_id[b], k0, k1, b) : -1;
      const int ar2 = has_a ? tok.app_id[b] : -1;
      const uint16_t* nsrc = nr >= 0 ? vt.name + (size_t)nr * net::VEC_D : nullptr;
      const uint16_t* asrc = ar2 >= 0 && ar2 < vt.n_app ? vt.app + (size_t)ar2 * net::VEC_D : nullptr;
      for (int e = 0; e < 8; ++e) {
        const int c = c0 + e;
        uint16_t hb = 0;
        if (live) {
          if (c < net::SLOT_VALS) hb = f2bf(slot_num(tok, b, c, gdrop));
          else if (c < net::SLOT_APP) hb = nsrc ? nsrc[c - net::SLOT_NAME] : (uint16_t)0;
          else if (c < net::SLOT_BIAS) hb = asrc ? asrc[c - net::SLOT_APP] : (uint16_t)0;
          else if (c == net::SLOT_BIAS) hb = f2bf(1.f);
        }
        h[e] = hb;
      }
      put8(srows + q * 8, h);
    }
  return mk;
}

}  // namespace obsv
