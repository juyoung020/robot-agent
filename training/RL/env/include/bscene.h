// BEHAVIOR 집 장면(계획서 CURRICULUM_BEHAVIOR2026.md 5절 E2): 장치에 올린 장면 7 개를 모든 판이 같이 쓴다(판마다 복사 안 함).
// CPU 참조판과 GPU 커널이 같은 소스를 쓴다(DEV). 이 헤더는 혼자 선다 — RASC 로더(rasc.h)는 호스트 쪽 bscene_host.cpp 만 쓴다.
//
// 좌표: 장면은 세계 좌표(RASC). 판은 **창 좌표**(세계 − 창 가운데 (wx, wy))로 돈다 — 지도(../map)의 128 × 128 칸 0.10 m 고정 창이
// 판마다 그 판의 시작·목표를 덮도록 옮겨진 것과 같다. 창 가운데는 장면 격자 칸 경계에 맞춰 둔다(창 칸 = 장면 칸, 정수로 대응).
// 정적 상자 = RASC BOXES 중 문(열린 채로 둠)·카펫·보이기만 하는 것·납작한 것을 뺀 회전 상자(yaw). 1 m 칸 묶음(bin)으로 찾는다.
// 과제 물체(인스턴스 자세)는 판의 "움직이는 상자"(축 정렬, Entry::prim 중 sbox < 0)로 몸통·광선과 부딪힌다.
#pragma once
#include "detmath.h"

namespace bsc {
using namespace dm;

constexpr int MAXSC = 8;           // 장면 수 상한(BEHAVIOR 2026 은 7)
constexpr float CELL = 0.1f;       // 장면 격자(RASC) = 지도 칸
constexpr float INV_CELL = 10.0f;
constexpr int NAV_P = 32;          // 지도 거리장 조각 한 변(칸) — gmap::NAV_P
constexpr float BIN = 1.0f;        // 상자 찾기 칸 묶음 m
constexpr float INV_BIN = 1.0f;
constexpr int WIN = 128;           // 창 한 변 칸 수 = gmap::GW
constexpr float WIN_HALF = 6.4f;   // 창 반 변 m = GW·RES/2
constexpr int NPRIM = 9;           // 판의 지도 물체 = gmap::N_PRIM (0 = 목표)
constexpr float H_COLL = 0.35f;    // 바닥 z0 이 이보다 낮은 상자만 몸통과 부딪힘 (가정: 몸통 0.251 m + 접은 팔)
constexpr int N_RTYPE_TOK = 6;     // 방 토큰 종류(kitchen·bathroom·bedroom·living room·office·모름)
constexpr int MAXBR = 16;          // 창 안 방 수 상한(지도 토큰용)
constexpr int MAXBD = 16;          // 창 안 문 수 상한

enum BoxKind { BK_WALL = 1, BK_FURN = 2, BK_WINDOW = 4, BK_COLL = 8, BK_BAND = 16 };   // BK_COLL: z0 < H_COLL, BK_BAND: z 범위가 지도 높이 띠 [0.05, 0.50] 와 겹침
// 판 단계: B1 집 안 이동, B2 찾기, B3 다가가기, B4 집기, B5 놓기(든 채 시작), B6 가져오기(찾기 → 다가가기 → 집기 → 나르기 → 놓기, E6 2026-10-05)
enum EntKind { EK_NONE = 0, EK_B1 = 1, EK_B2 = 2, EK_B3 = 3, EK_B4 = 4, EK_B5 = 5, EK_B6 = 6 };
constexpr int N_EK = 7;   // 단계 번호 수(표 크기)
enum ListKind { L_ROOM = 0, L_OBJ = 1 };                                             // B1 은 방 표, B2–B5 는 집기·놓기 표(L_OBJ = 짝 하나 = 판 하나)
enum DstKind { DK_ONTOP = 1, DK_INSIDE = 2, DK_FLOOR = 3 };                           // 놓을 곳(RASC PlaceRec kind)
enum NoFilter { NF_SPAWN_REACH = 1, NF_SPAWN_FREE = 2, NF_FREE_AREA = 4, NF_IN_CLOSED = 8, NF_ARTIC = 16, NF_DST_REACH = 32, NF_STANCE = 64 };   // 음성 대조용 거르개 끄기

struct SBox { float cx, cy, hx, hy, c, s, z0, z1; };   // 세계 회전 상자(yaw 의 cos·sin), 32 B
struct SDoor { float x, y; int16_t ra, rb; };          // 문 가운데(세계), 양쪽 방(장면 방 번호, 없으면 −1)

struct SceneDev {
  float ox, oy;               // 칸 (0, 0) 의 왼쪽 아래 모서리(세계)
  int W, H;                   // 칸 수(0.10 m)
  int BW, BH;                 // 묶음 수(1 m, 같은 원점)
  const uint32_t* bstart;     // [BW·BH + 1] 묶음마다 상자 목록 시작
  const uint16_t* bitem;      // 상자 번호
  const SBox* box;            // [nbox]
  const uint8_t* bkind;       // BoxKind 비트
  const int16_t* bname;       // 이름 표 행(vla_v1), 가구 종류
  int nbox;
  const uint8_t* room;        // [H][W] 방 번호 + 1 (0 = 방 아님)
  const uint32_t* occ;        // [H][W] 비트: 띠 [0.05, 0.50] 와 겹치는 정적 상자가 칸을 덮음(처음 지도 C0·C1 의 참 점유)
  const int8_t* rtype;        // [nroom] 방 토큰 종류 0..5
  int nroom;
  const SDoor* door;
  int ndoor;
  const uint16_t* comp;       // [H][W] 로봇 중심 칸의 이어진 성분(0 = 설 수 없음), 바닥 높이 차 ≤ 느슨 문턱만 이음
  const uint16_t* comp_in;    // 같은 것, 엄격 문턱
};

struct BPrim { float lo[3], hi[3]; int16_t name; int16_t sbox; };   // 창 좌표 축 정렬 상자, 이름 표 행, 정적 상자 번호(−1 = 과제 물체, 움직이는 상자)
struct Entry {            // 판 하나의 시작 조건(호스트가 미리 만들고 검사한 표)
  int16_t scene, list;    // 장면 번호, ListKind
  int16_t split, groom;   // 0 학습 / 1 공개 평가 인스턴스, 목표 방(장면 방 번호, B1) 또는 목표 물체가 있는 방(−1 없음)
  int32_t inst;           // RASC 인스턴스 번호
  int16_t nprim, task;    // 채운 물체 수(≤ NPRIM), RASC 과제 번호
  float wx, wy;           // 창 가운데(세계)
  float sx, sy, syaw;     // 로봇 시작(창 좌표)
  float gx, gy, gz;       // 목표: B1 = 가구 앞 바닥 점(gz 0), B2·B3 = 목표 물체 상자 가운데(창 좌표)
  float path;             // 시작 → 목표(B1: 점 0.5 m 안, 물체: 팔 닿는 원 안)까지 참 장면 최단 경로 m(호스트 8 이웃 다익스트라, 창 안)
  float ext[3];           // 목표 물체 상자 크기(B1 은 목표 가구)
  BPrim prim[NPRIM];      // 0 = 목표(B1 은 목표 가구, B2–B5 는 집을 물체), 집기·놓기 표는 1 = 놓을 곳(바닥이 아니면), 다음 과제 물체, 창 안 가구
  // ---- 집기·놓기 표(L_OBJ) 만: 짝 = (집을 물체, 출발 받침, 놓을 곳). sx·sy·syaw 는 대신 쓸 시작(무작위 시작을 못 찾을 때) ----
  int16_t dkind, rel;     // DstKind, RASC 술어(ontop / inside)
  int16_t src_name, dst_name;   // 이름 표 행(−2 = 바닥, 놓을 곳 ≤ −1000 = 방 종류 −1000−값의 바닥)
  int16_t fset, dst_room; // fset 1 = 엄격 거르개도 지남, 놓을 곳 방
  uint16_t comp, comp_in; // 잡는 자세 칸의 성분(시작 칸이 같아야 함) — 느슨/엄격 문턱
  int32_t combo;          // 지시문 조합 번호(training/data/pnp_v1, 없으면 −1)
  int32_t pick_rec, dst_rec, src_rec;   // RASC PICKS·PLACES 번호(확인 도구용)
  int32_t rb, rb_in;      // 창 안 닿는 칸 비트(창 128² 칸 = 낱말 512) 자리: 잡는 자세 칸에서 창 안 BFS(느슨/엄격 문턱), −1 없음
  float dlo[3], dhi[3];   // 놓을 자리(창 좌표 축 정렬): 면·용기 = 상자, 바닥 = 고른 0.4 m 자리
  // 놓을 점(목표 점, 2026-10-05 — 앱에서 바닥·면을 눌러 "여기에 놔"): 창 좌표 xyz(z = 물체가 놓일 바닥 높이). 바닥 = 고른 자리 가운데(z 0),
  // 면(ontop) = 호스트가 고른 윗면 점(물체 바닥 자국이 면 안·다른 물체와 안 겹침·닿는 칸에서 팔 닿는 거리 — bscene_host place_point), 용기 = 없음
  float ppt[3];
  int32_t ppt_ok;         // 1 = ppt 가 놓을 수 있는 점, 0 = 없음(용기·못 찾음)
  // ---- 잡기 물리(E6, 2026-10-05 — 집기·놓기 표만, 뒤에 붙임: 앞 자리 그대로) ----
  float mass;             // 집을 물체 종류 평균 질량 kg(RASC PICKS, 없으면 NaN → 환경이 KG::mass_unknown 으로)
  float st[2];            // 잡는 자세 칸 가운데(창 좌표, 호스트가 BFS 시작으로 쓴 칸 — 물체를 바라보면 몸통 안 닿고 잡는 점 작업 공간) — B4·B5 시작
  float src_top;          // 집을 물체의 처음 받침 윗면 z(RASC PLACES top, 바닥이면 0 근처, 모르면 물체 바닥 z)
  float dst_st[2];        // 놓을 곳에 닿는 칸 가운데(창 좌표; 바닥이면 고른 자리 가운데) — 대본 교사·확인용
  float oyaw, odim[3];    // 집을 물체의 회전 상자: yaw(세계), 크기(yaw 축 가로·세로, 높이) — 바닥 자국을 가장 작게 덮는 직사각형(물체 축 투영). ext 는 그 AABB
};
// 집기·놓기 판 고르기 표(호스트가 만듦): 인스턴스 → 집을 물체 → 짝(Entry). 각 단계에서 엄격 판을 앞에 둠
struct PnpPick { int ent_off, n, n_in; };
struct PnpInst { int pick_off, npick, npick_in, scene, split; };

struct SceneSet {         // 장치 메모리에 하나(커널은 포인터로 읽음). CPU 참조판은 같은 구조체를 호스트 포인터로
  SceneDev sc[MAXSC];
  int nsc;
  const Entry* ent;
  int nent;
  int loff[MAXSC][2][2], lcnt[MAXSC][2][2];   // [장면][ListKind][split] → ent 안 범위
  int lcnt_in[MAXSC][2][2];                   // 그 범위 앞쪽의 안 한도(엄격) 판 수 — 물체 표는 엄격 판을 앞에 둔다
  const PnpInst* pinst;                       // 집기·놓기: 인스턴스 목록, [장면][split] 범위(엄격 가능한 것이 앞)
  const PnpPick* ppick;
  int ioff[MAXSC][2], icnt[MAXSC][2], icnt_in[MAXSC][2];
  int ntpl, ntpl_train;                       // 지시문 조합마다 문장 수(앞 ntpl_train = 학습, 나머지 = 평가용 heldout)
  int ncombo;                                 // 지시문 조합 수(pnp_v1 combos.tsv 줄 수)
  int iblocks;                                // 지시문 표 묶음: 1 = 조합만(pnp_v1 v1), 3 = 조합 | 점에 놓기("put the {o} here") | 점으로 가기("go here") — v2
  const uint32_t* rbits;                      // Entry::rb·rb_in 이 가리키는 창 닿는 칸 비트 묶음
  // 이름 표(vla_v1): 생김새 행 a(0..6) × 이름 행 n 의 확신도(bscene_host 가 name128·app128 로 계산), 비슷한 다른 이름 3, 상위어
  const float* conf1;     // [7][nname]
  const float* conf2;
  const int16_t* sim3;    // [nname][3]
  const int16_t* hyper;   // [nname]
  int nname;
};

// 커리큘럼(장치 값, 판 리셋 때 읽음 — 바꿔도 다시 컴파일·그래프 다시 잡기 없음)
struct BCurr {
  float p1, p2;           // B1·B2 비율(나머지 B3)
  uint32_t scene_mask;    // 쓸 장면 비트(SceneSet 번호)
  int split;              // 0 학습 인스턴스, 1 공개 평가, 2 둘 다
  float yaw_jit;          // 시작 yaw 흔들기 ±rad (기본 0 = 인스턴스 그대로)
  int strict;             // 1 = 집기·놓기 표(B2–B5)를 엄격 거르개 판만, 0 = 느슨(기본). 판마다 쓴 거르개를 상태에 적음(I_B_FSET)
  int nofilter;           // NoFilter 비트(시작 고르기 거르개 끄기 — 음성 대조만)
  int eval_instr;         // 1 = 지시문을 평가용(heldout) 문장에서
  // 목표 점(2026-10-05, VLA_INPUT 2.1 목표 칸 — 0 이면 예전과 같은 난수 흐름·같은 판)
  float p_point;          // 집기·놓기 판(B2·B3)에서 놓을 곳을 받침 물체 대신 놓을 점(Entry::ppt)으로 + 지시문 "put the {o} here" 묶음. 바닥 놓을 곳은 늘 점(이 확률은 지시문만)
  float p_goto;           // B1·B3 판을 "점으로 가기" 로: 집을 칸 없음, 놓을 칸 = 점(B1 = 방 목표 점, B3 = Entry::ppt), 지시문 "go here" 묶음
  // 잡기 물리 판(E6, 2026-10-05 — 모두 0 이면 예전과 같은 난수 흐름·같은 판). 단계 고르기: u < p1 → B1, < p1+p2 → B2, < +p4 → B4, < +p5 → B5, < +p6 → B6, 나머지 B3
  float p4, p5, p6;       // B4 집기(잡는 자세에서 시작) · B5 놓기(든 채 잡는 자세에서 시작) · B6 가져오기(무작위 시작, 찾기 → 놓기 전체)
  float p_slip;           // 실패 판: 들고 있는 동안(받침에서 뜸) 제어 스텝마다 이 확률로 미끄러져 떨어짐 — 다시 잡기 연습(가정 값은 설정에서)
  float p_occ;            // 실패 판: 판 리셋 때 이 확률로 놓을 자리(점·면 점)에 다른 물체(작은 상자)가 이미 있음 — 옆 빈 자리에 놓기
  int phys;               // 물리 비트(PhysFlag): 음성 대조·실험용 끄기
};
constexpr BCurr kBCurrDefault = {0.34f, 0.33f, 0xffu, 0, 0.f, 0, 0, 0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0};
// 물리 끄기 비트(BCurr::phys, 음성 대조·실험): 무게 미끄러짐 끔, 폭 검사 끔(손가락 사이 아무 폭이나 잡힘), 팔 충돌 막기 끔
enum PhysFlag { PF_NO_SLIP = 1, PF_NO_WIDTH = 2, PF_NO_ARMCOLL = 4 };
// 판의 목표 꼴(env I_B_GMODE 비트, 지도 BMapEnv::gmode)
enum GoalMode { GM_PLACE_PT = 1,   // 놓을 칸(목표 칸 1)이 점(F_B_GPX..Z)
                GM_GOTO = 2 };     // 점으로 가기: 집을 칸 없음, 지금 가는 목표 = 놓을 점(B1·B3 변형)
// 점에 놓기 성공(B5 점 판, 잡기 물리 E6 뒤 — 정의만): 놓은 물체(손 안 아님)의 바닥 자국 가운데가 점에서 수평 place_r 안, 바닥 높이가 점 z ± place_tz
struct KPt {
  static constexpr float place_r = 0.05f;    // 2D 반경 m (가정: 앱 지도 칸 0.05 m·OMX 위치 오차 ~1 cm 보다 넉넉히)
  static constexpr float place_tz = 0.02f;   // 높이 허용 m (pred_ontop tol 과 같음)
};

// 지도 → 환경(앞 스텝의 지도 값): 정책이 아는 지도(자라는 지도)의 거리장과 목표 확정 여부
struct NavFb {
  const uint8_t* lev;     // [N][NAV_P·NAV_P] 거리장 단계 조각(계산 때 로봇 칸 둘레, 0.1 m 칸, 4·8 이웃 번갈아 BFS = 팔각 거리 ≈ 단계 × 0.1 m), 255 = 못 감/멈춤 뒤
  const int* org;         // [N] 조각 원점 창 칸(열 & 0xffff | 행 << 16, 16 비트 부호)
  const int* tag;         // [N] 거리장을 만든 판 번호(env ep), 다르면 쓰지 않음
  const int* conf;        // [N] 목표 물체가 지도에 확정(B2 성공 조건)
};

// ---------------------------------------------------------------------------------------------------------------------
DEV int bin_of(const SceneDev& d, float x, float y, int& bx, int& by) {   // 세계 → 묶음, 밖이면 0
  bx = (int)floorf((x - d.ox) * INV_BIN);
  by = (int)floorf((y - d.oy) * INV_BIN);
  return bx >= 0 && by >= 0 && bx < d.BW && by < d.BH;
}
DEV int room_at(const SceneDev& d, float x, float y) {   // 세계 점의 방(장면 방 번호), 없으면 −1
  const int c = (int)floorf((x - d.ox) / CELL), r = (int)floorf((y - d.oy) / CELL);
  if (c < 0 || r < 0 || c >= d.W || r >= d.H) return -1;
  return (int)d.room[(size_t)r * d.W + c] - 1;
}
// 몸통 직사각형(중심, yaw sin·cos, 반 길이·반 폭)과 회전 상자의 분리축 4 개(열린 겹침)
DEV bool rect_hits_obb(float x, float y, float s, float co, float hl, float hw, const SBox& b) {
  const float dx = b.cx - x, dy = b.cy - y;
  // 몸 축
  const float bc = co * b.c + s * b.s, bs = -s * b.c + co * b.s;   // 상자 축 x 를 몸 좌표로: (bc, bs)
  const float abc = absf(bc), abs_ = absf(bs);
  const float px = co * dx + s * dy, py = -s * dx + co * dy;
  if (!(absf(px) < hl + b.hx * abc + b.hy * abs_)) return false;
  if (!(absf(py) < hw + b.hx * abs_ + b.hy * abc)) return false;
  // 상자 축
  const float qx = b.c * dx + b.s * dy, qy = -b.s * dx + b.c * dy;
  if (!(absf(qx) < b.hx + hl * abc + hw * abs_)) return false;
  return absf(qy) < b.hy + hl * abs_ + hw * abc;
}
// 몸통과 축 정렬 상자(lo, hi xy)
DEV bool rect_hits_aabb(float x, float y, float s, float co, float hl, float hw, const float lo[3], const float hi[3]) {
  const float ex = 0.5f * (hi[0] - lo[0]), ey = 0.5f * (hi[1] - lo[1]);
  const float dx = 0.5f * (lo[0] + hi[0]) - x, dy = 0.5f * (lo[1] + hi[1]) - y;
  const float ac = absf(co), as = absf(s);
  if (!(absf(dx) < ex + ac * hl + as * hw)) return false;
  if (!(absf(dy) < ey + as * hl + ac * hw)) return false;
  if (!(absf(co * dx + s * dy) < hl + ex * ac + ey * as)) return false;
  return absf(-s * dx + co * dy) < hw + ex * as + ey * ac;
}
// 몸통(세계 중심)이 BK_COLL 정적 상자에 닿나: 몸통 외접원이 덮는 묶음만
DEV bool body_hits_scene(const SceneDev& d, float x, float y, float s, float co, float hl, float hw) {
  const float r = sqrtf(hl * hl + hw * hw);
  int bx0, by0, bx1, by1;
  bin_of(d, x - r, y - r, bx0, by0);
  bin_of(d, x + r, y + r, bx1, by1);
  bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
  bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
  for (int by = by0; by <= by1; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) {
        const int j = d.bitem[k];
        if (!(d.bkind[j] & BK_COLL)) continue;
        if (rect_hits_obb(x, y, s, co, hl, hw, d.box[j])) return true;
      }
    }
  return false;
}

// 점 → 회전 상자 바닥 자국 거리(안이면 0)
DEV float dist_pt_obb2(const SBox& b, float x, float y) {
  const float dx = x - b.cx, dy = y - b.cy;
  const float lx = b.c * dx + b.s * dy, ly = -b.s * dx + b.c * dy;
  const float ex = maxf(absf(lx) - b.hx, 0.f), ey = maxf(absf(ly) - b.hy, 0.f);
  return sqrtf(ex * ex + ey * ey);
}
// 한 제어 스텝의 충돌 후보(넓은 단계): 스텝 시작 자리에서 r 안의 BK_COLL 정적 상자(겹치면 한 번). 넘치면 overflow(모든 묶음 검사로)
constexpr int NCAND = 8;
struct CollCand { int n, overflow; int idx[NCAND]; };
DEV void coll_gather(const SceneDev& d, float x, float y, float r, CollCand& cc) {
  cc.n = 0; cc.overflow = 0;
  int bx0, by0, bx1, by1;
  bin_of(d, x - r, y - r, bx0, by0);
  bin_of(d, x + r, y + r, bx1, by1);
  bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
  bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
  for (int by = by0; by <= by1; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) {
        const int j = d.bitem[k];
        if (!(d.bkind[j] & BK_COLL) || !(dist_pt_obb2(d.box[j], x, y) < r)) continue;
        bool dup = false;
        for (int q = 0; q < cc.n; ++q) dup = dup || cc.idx[q] == j;
        if (dup) continue;
        if (cc.n < NCAND) cc.idx[cc.n++] = j; else cc.overflow = 1;
      }
    }
}

// 2D 반직선 o + t·(dx, dy) 이 회전 상자에 들어가는 t(안이면 0), 없으면 tmax
DEV float ray_obb2(float ox, float oy, float dx, float dy, const SBox& b, float tmax) {
  const float rx = ox - b.cx, ry = oy - b.cy;
  const float lx = b.c * rx + b.s * ry, ly = -b.s * rx + b.c * ry;
  const float ux = b.c * dx + b.s * dy, uy = -b.s * dx + b.c * dy;
  float t0 = 0.f, t1 = tmax;
  if (ux > 1e-6f || ux < -1e-6f) {
    float ta = (-b.hx - lx) / ux, tb = (b.hx - lx) / ux;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(lx > -b.hx && lx < b.hx)) return tmax;
  if (uy > 1e-6f || uy < -1e-6f) {
    float ta = (-b.hy - ly) / uy, tb = (b.hy - ly) / uy;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(ly > -b.hy && ly < b.hy)) return tmax;
  return t0 <= t1 ? t0 : tmax;
}

// 묶음 걷기(Amanatides–Woo, 2D): 세계 반직선 o + t·d 가 지나는 묶음을 들어가는 t 순으로. f(상자 번호, 묶음 들어간 t) 를 부르고,
// 들어가는 t 가 lim() 이상이면 멈춘다(그 뒤 상자는 더 가까운 맞음을 낼 수 없음 — 결과 같음). 상자가 여러 묶음에 있으면 여러 번 불림(최솟값이라 같음)
template <class F, class L>
DEV void bin_walk(const SceneDev& d, float ox, float oy, float dx, float dy, const F& f, const L& lim) {
  const float fx = (ox - d.ox) * INV_BIN, fy = (oy - d.oy) * INV_BIN;
  int bx = (int)floorf(fx), by = (int)floorf(fy);
  const int sx = dx > 0.f ? 1 : -1, sy = dy > 0.f ? 1 : -1;
  const float adx = absf(dx), ady = absf(dy);
  const float tdx = adx > 1e-12f ? BIN / adx : 1e30f, tdy = ady > 1e-12f ? BIN / ady : 1e30f;
  float tx = adx > 1e-12f ? ((sx > 0 ? (float)(bx + 1) - fx : fx - (float)bx) * BIN) / adx : 1e30f;
  float ty = ady > 1e-12f ? ((sy > 0 ? (float)(by + 1) - fy : fy - (float)by) * BIN) / ady : 1e30f;
  // 격자 밖에서 시작하면 들어올 때까지 걷는다(창이 장면 밖까지 덮을 수 있음)
  float t = 0.f;
  for (int it = 0; it < 4096; ++it) {
    if (!(t < lim())) return;
    if (bx >= 0 && by >= 0 && bx < d.BW && by < d.BH) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) f((int)d.bitem[k], t);
    } else if ((bx < 0 && sx < 0) || (by < 0 && sy < 0) || (bx >= d.BW && sx > 0) || (by >= d.BH && sy > 0)) {
      return;   // 격자 밖으로 멀어짐
    }
    if (tx < ty) { t = tx; tx = tx + tdx; bx += sx; } else { t = ty; ty = ty + tdy; by += sy; }
  }
}

// 3D 선분 p → p + v (매개 0..1) 이 회전 상자를 지나나(열린 영역)
DEV bool seg_hits_obb3(const float p[3], const float v[3], const SBox& b) {
  const float rx = p[0] - b.cx, ry = p[1] - b.cy;
  const float lo3[3] = {b.c * rx + b.s * ry, -b.s * rx + b.c * ry, p[2]};
  const float u3[3] = {b.c * v[0] + b.s * v[1], -b.s * v[0] + b.c * v[1], v[2]};
  const float blo[3] = {-b.hx, -b.hy, b.z0}, bhi[3] = {b.hx, b.hy, b.z1};
  float t0 = 0.f, t1 = 1.f;
  for (int a = 0; a < 3; ++a) {
    if (u3[a] > 1e-9f || u3[a] < -1e-9f) {
      float ta = (blo[a] - lo3[a]) / u3[a], tb = (bhi[a] - lo3[a]) / u3[a];
      if (ta > tb) { const float q = ta; ta = tb; tb = q; }
      t0 = maxf(t0, ta); t1 = minf(t1, tb);
    } else if (!(lo3[a] > blo[a] && lo3[a] < bhi[a])) return false;
  }
  return t0 < t1;
}
DEV bool seg_hits_aabb3(const float p[3], const float v[3], const float lo[3], const float hi[3]) {
  float t0 = 0.f, t1 = 1.f;
  for (int a = 0; a < 3; ++a) {
    if (v[a] > 1e-9f || v[a] < -1e-9f) {
      float ta = (lo[a] - p[a]) / v[a], tb = (hi[a] - p[a]) / v[a];
      if (ta > tb) { const float q = ta; ta = tb; tb = q; }
      t0 = maxf(t0, ta); t1 = minf(t1, tb);
    } else if (!(p[a] > lo[a] && p[a] < hi[a])) return false;
  }
  return t0 < t1;
}
// 세계 선분 p → q 를 정적 상자(모든 높이, except 번호 하나 빼고)가 가리나
DEV bool seg_blocked_scene(const SceneDev& d, const float p[3], const float q[3], int except) {
  const float v[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  bool hit = false;
  bin_walk(d, p[0], p[1], v[0], v[1],
           [&](int j, float) { if (!hit && j != except && seg_hits_obb3(p, v, d.box[j])) hit = true; },
           [&]() { return hit ? -1.f : 1.f; });
  return hit;
}

// ---- 창 거리장(지도 → 환경): 칸 (x, y) 둘레 5 × 5 칸 중 닿은 칸의 (단계 × 0.1 m + 칸 가운데까지 거리) 최소. 없으면 −1 ----
DEV float field_dist(const uint8_t* lev, int org, float x, float y) {
  const int pc0 = (int)(int16_t)(org & 0xffff), pr0 = org >> 16;
  const int cx = (int)floorf(x * INV_CELL) + WIN / 2, cy = (int)floorf(y * INV_CELL) + WIN / 2;   // gmap::nav_cell 과 같은 식
  float best = 1e30f;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      const int ix = cx + dx, iy = cy + dy, px = ix - pc0, py = iy - pr0;
      if (ix < 0 || iy < 0 || ix >= WIN || iy >= WIN || px < 0 || py < 0 || px >= NAV_P || py >= NAV_P) continue;
      const int L = lev[py * NAV_P + px];
      if (L == 255) continue;
      const float ccx = ((float)ix + 0.5f) * CELL - WIN_HALF, ccy = ((float)iy + 0.5f) * CELL - WIN_HALF;
      const float ex = x - ccx, ey = y - ccy;
      const float dd = (float)L * CELL + sqrtf(ex * ex + ey * ey);
      if (dd < best) best = dd;
    }
  return best < 1e30f ? best : -1.f;
}

// ---- B4·B5 성공 판정(정의만 — 잡기 물리는 E6). 상자는 축 정렬 lo/hi(창 또는 세계 좌표, 같은 좌표계) ----
// B4 집기: 든 상태(E6 규칙의 is_grasping 대응 — 인자로) + 처음 받침 윗면에서 lift 이상 떨어짐(not ontop). 환경(env_pnp.h)은 KG::lift_h 로 부름
DEV bool pred_grasped(bool held, const float obj_lo[3], float support_top, float lift = 0.02f) { return held && obj_lo[2] >= support_top + lift; }
// ontop(가정, OmniGibson 의 접촉 + 위쪽 판정을 상자로): 물체 바닥이 받침 윗면 ± tol 안, 물체 가운데 xy 가 받침 윗면 사각형 안
DEV bool pred_ontop(const float o_lo[3], const float o_hi[3], const float s_lo[3], const float s_hi[3], float tol = 0.02f) {
  const float cx = 0.5f * (o_lo[0] + o_hi[0]), cy = 0.5f * (o_lo[1] + o_hi[1]);
  return absf(o_lo[2] - s_hi[2]) <= tol && cx > s_lo[0] && cx < s_hi[0] && cy > s_lo[1] && cy < s_hi[1];
}
// 점에 놓기(KPt): 손에서 놓였고, 바닥 자국 가운데가 점 pt 에서 수평 place_r 안, 물체 바닥이 pt z ± place_tz
DEV bool pred_at_point(bool held, const float o_lo[3], const float o_hi[3], const float pt[3]) {
  const float dx = 0.5f * (o_lo[0] + o_hi[0]) - pt[0], dy = 0.5f * (o_lo[1] + o_hi[1]) - pt[1];
  return !held && dx * dx + dy * dy <= KPt::place_r * KPt::place_r && absf(o_lo[2] - pt[2]) <= KPt::place_tz;
}
// inside(가정): 물체 상자 가운데가 용기 상자 안이고 물체 xy 가 용기 xy 안에 들어감(위 열린 용기)
DEV bool pred_inside(const float o_lo[3], const float o_hi[3], const float c_lo[3], const float c_hi[3]) {
  const float cz = 0.5f * (o_lo[2] + o_hi[2]);
  return o_lo[0] >= c_lo[0] && o_hi[0] <= c_hi[0] && o_lo[1] >= c_lo[1] && o_hi[1] <= c_hi[1] && cz > c_lo[2] && cz < c_hi[2];
}

}  // namespace bsc
