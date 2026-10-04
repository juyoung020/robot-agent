// scenemap ② objmap — 검출(마스크 + 이름 번호) + 깊이 + 자세 → 물체 3D 위치 → 같은 물체 판단 → 바뀐 부분만 갱신.
//
// docs/scenemap_설계.md 3.2. 규칙 요약
//   종류   : 이름 번호마다 물체 / 구조물(벽·바닥·천장·문·창·기둥·칸막이·계단·난간·걸레받이 — 물체가 안 되고 격자만) /
//            고정(가구·가전·붙박이 — 물체 노드지만 movable=false, 사라짐은 큰 것 기준으로 판정). 표는 capi(sm_set_kind_names)가 정함
//   위치   : 마스크를 1 칸 깎은 안쪽 깊이 점(map)의 축별 중앙값, 크기 = 10~90 백분위 폭.
//            그 전에 카메라 깊이가 마스크 안 중앙값에서 max(mad_k·1.4826·MAD, mad_floor) 넘게 떨어진 점은 버림(뒤 벽이 비친 것)
//   거르기 : 로봇 팔 캡슐(ObjFrame.self_caps — 순기구학, LIMO) 안(반경 + self_pad) 깊이 점은 검출 마스크 안이어도 버림(팔이 카메라 앞을
//            가릴 때 팔 화소가 물체 상자를 키우거나 옮기지 않게). 점 min_points 미만 버림. 점의 hand_frac 이상이 팔 끝 hand_r 안이면(손에 든 것) 버림. 점의 90 백분위 높이 < floor_h 면
//            바닥 조각(바닥에 깔리는 이름 — 러그·카펫·매트 — 은 둠)
//   이름   : 물체마다 이름 표(이름 번호별 점수 합). 이름 = 최댓값(지금 이름보다 name_switch 배 넘어야 바뀜)
//   같은 것: 이름이 같거나 관측 이름이 물체 이름 표에서 name_share 이상. 중심 거리 < max(da_min, da_k·큰 쪽 크기) 이거나
//            map 축 상자 사이 틈 < da_gap. (틈 + 이름 다르면 0.02)·중심 거리 순으로 1:1(탐욕). 큰 가구(한 변 > big 또는 고정 종류)는
//            상자를 합집합으로 키우고 위치 = 상자 중심 — 단 keyframe 마다 면마다 grow_max 까지만, 한 변 max_ext 넘게는 안 키움
//   확정   : 서로 다른 keyframe 에서 confirm 번 보이면 물체. 후보가 prune_s 동안 다시 안 보이면 버림
//   움직임 : 관측 중심(영상 가장자리에 닿지 않은 것, 카메라가 move_max_cam_w 보다 천천히 돌 때)이 move_n 번 잇달아 move_v 넘게
//            같은 쪽으로 가고 쉬던 상자를 벗어나면 평균 대신 관측 자리로 바로 따라감(옮겨짐 상태)
//   사라짐 : 확정·안 든·통에 안 넣은 물체(큰 것·고정 종류 포함). 물체 점(구름, 없으면 상자 격자)을 투영해 시야 안·안 가림 비율 ≥
//            absent_vis, 보이는 부분 ≥ absent_min_px, 카메라 거리 ≤ 이 물체를 검출한 가장 먼 거리·absent_det_k + 0.2, 팔 끝
//            hand_r+0.1 밖인데 검출이 없으면 놓침 +1 — 단 지난 놓침 뒤 카메라가 gone_step_d·gone_step_deg 넘게 바뀌었거나 자리 너머가
//            보일 때만(같은 영상 되풀이는 안 셈), 물체 상자 안에 전에 그 물체 이름 표에 있던 다른 이름 관측이 있으면 안 셈.
//            필요한 놓침 수 = max(gone_misses(큰 것 gone_misses_big), 검출률 p 로 (1−p)^k < gone_eps 인 k), 그리고 첫 놓침에서
//            gone_min_s(큰 것 gone_min_s_big) 지났거나 카메라가 gone_view_d 넘게 옮김. 관측이 spurious_obs 보다 적던 것은 지움(헛검출)
//   옮겨짐 : 사라진 m ↔ 새로 나타난 n(같은 이름, n 확정·관측 link_min_obs 이상, n 처음 > m 마지막, 거리 ≤ min(link_max_d,
//            link_d0 + link_v·시간 차), n 자리를 처음 검출한 거리 이하에서 link_view_gap_s 넘게 전에 본 적 있음, n 에 더 가까운 같은
//            이름 물체가 n 이후 아직 안 보였으면 link_wait_s 까지 기다림) 를 가까운 쌍부터 이어 n 을 m 의 id 로(relink)
//   들기   : 그리퍼가 닫히는 순간 팔 끝 grasp_r 안 가장 가까운 확정 물체를 든 것으로 — 드는 동안 팔 끝을 따라가고,
//            (grasp_check — LIMO: 그리퍼가 닫힌 채 멈췄을 때(grip_settle_s 동안 grip_settle_eps 안) 한 번만 고르고, 큰 것·고정 종류·지도
//            상자 가운데 변 > grasp_max_w 는 못 듦, 손끝 틈(grip_gap 표) ≥ grasp_min_gap(빈손이면 끝까지 닫힘)이고 틈이 물체 폭과 맞아야
//            (가장 좁은 변 − grasp_w_tol ≤ 틈 ≤ 가장 넓은 변 + grasp_w_tol). 든 뒤 끝까지 닫히면(놓침) 놓기)
//            열리는 순간 그 자리에 놓는다(moved_d 넘게 옮겼으면 옮겨짐). 놓은 점 아래에 xy 가 겹치는(0.1 m 여유) 다른 물체
//            상자가 있으면 그중 윗면이 가장 높은 것(떨어져 닿을 받침)에 붙인다 — 들고 있는 쓰레기통에 넣은 캔이 통을 따라가게
#pragma once
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/cloud.hpp"
#include "scenemap/objprob.hpp"
#include "scenemap/scan.hpp"

namespace scenemap {

struct ObjParams {
  int min_points = 20;
  float zmin = 0.15f, zmax = 5.0f;
  float hand_r = 0.40f, hand_frac = 0.5f;
  double da_min = 0.30, da_k = 0.5;
  double da_gap = 0.10;           // 상자끼리 이 안으로 붙어 있으면 같은 것(부분만 보이는 큰 가구)
  double big = 0.5;               // 상자 한 변이 이보다 크면 합집합으로 키움(작은 물체는 평균)
  int confirm = 2;
  double prune_s = 10.0;
  double moved_d = 0.15;
  int gone_misses = 3;
  double gone_min_s = 2.0;        // 사라짐: 첫 놓침부터 이 시간(시뮬 s) 넘게 연속으로 안 보여야
  double mad_k = 3.0;             // 마스크 안 깊이 이상값 띠: 중앙값 ± max(mad_k·1.4826·MAD, mad_floor)
  double mad_floor = 0.10;
  double grow_max = 0.25;         // 큰 가구 상자: keyframe 마다 면마다 최대 자람 m
  double max_ext = 4.0;           // 큰 가구 상자 한 변 최대 m
  double occl = 0.10;
  int min_px = 6;                 // 시야 안 판정: 물체가 이 화소보다 작게 보이면 부재 증거로 안 씀
  double grasp_r = 0.25;
  float grip_closed = 0.09f;      // 손가락 합이 이보다 작으면 닫힘(열림 0.1). LIMO: omx_gripper_joint_1 rad(0 닫힘 .. 1.745 열림)
  int n_hands = 2;                // 잡기 규칙을 보는 손 수(R1 2, LIMO 1 — 둘째 칸은 첫째와 같은 값을 받지만 잡기에는 안 씀)
  // 잡기 확인(10-04, LIMO 만 — capi robotParams 가 켬. 끄면 옛 규칙 그대로: 닫히는 순간 grasp_r 안 가장 가까운 확정 물체)
  bool grasp_check = false;
  double grasp_max_w = 0.06;      // 물체 지도 상자(10~90 백분위)의 가운데 변이 이보다 크면 못 듦(그리퍼 한도. 한 축만 긴 것은 됨)
  double grasp_min_gap = 0.005;   // 손끝 틈이 이보다 작으면(끝까지 닫힘) 빈손
  double grasp_w_tol = 0.025;     // 틈과 물체 폭이 맞음: 가장 좁은 변 − tol ≤ 틈 ≤ 가장 넓은 변 + tol(지도 상자는 10~90 백분위·일부만 보임)
  double grip_settle_s = 0.2, grip_settle_eps = 0.01;   // 그리퍼 값이 이 시간 동안 eps 안이면 쥠이 끝남(멈춤)
  std::vector<std::pair<double, double>> grip_gap;      // 그리퍼 값 → 잡는 점의 손끝 틈 m(오름차순, 사이는 선형, 밖은 끝값)
  bool self_mask = true;          // ObjFrame.self_caps 가 있으면 그 안 점을 버림(진단: SM_OBJ_PARAMS self_mask=0 으로 끔)
  double self_pad = 0.01;         // 팔 캡슐 거르기 여유(m)
  int step = 1;                   // 깊이 화소 간격(최소)
  int max_pts = 6000;             // 검출 하나에서 훑는 화소 수 한도: 큰 상자는 간격을 넓힘(백분위·중앙값에는 충분)
  // 점 구름(모양): 위치·크기에 쓴 점(MAD 띠 안) 중 팔 끝 cloud_hand_r 안·베이스 수평 body_r 안 점은 뺌
  double voxel = 0.02;
  int cloud_cap = 4000;
  double cloud_hand_r = 0.10;
  double body_r = 0.30;
  // 병합(da/merge.hpp): 확정된 같은 이름 물체끼리 상자가 이만큼(축별 겹침 비율의 곱) 겹치면 하나로
  bool merge = true;
  double merge_overlap = 0.35;
  double merge_min_ext = 0.05;
  // ---- 바뀜 판정(10-04, dynamic-object-mapping-benchmark 로 고침 — README "바뀜 규칙") ----
  // 바닥 조각: 관측 점의 90 백분위 높이가 이보다 낮으면(map z, 바닥 = 0) 물체가 아님
  double floor_h = 0.05;
  // 이름 모으기: 물체마다 이름 번호별 점수 합(표). 이름 = 표의 최댓값(지금 이름보다 name_switch 배 넘어야 바꿈).
  // 관측 이름이 표에서 name_share 이상이면 같은 물체 후보. 병합(da)은 이름이 달라도 상자가 거의 같으면(name_merge_iou) 합침
  // 조각 합치기: 이번 관측 상자 안에 다른 같은 이름 물체 상자가 이 비율(축별 겹침 곱) 넘게 들어 있고 더 작으면 한 물체로(0 = 끔)
  double frag_overlap = 0.0;      // 켜면 0.6 정도. 10-04 실제 검출에서 끈 쪽이 조금 나음(static 0.697 vs 0.684) — 기본 끔
  bool name_vote = true;
  double name_share = 0.2;
  double name_switch = 1.25;
  double name_merge_iou = 0.5;
  // 사라짐(보임 근거): 물체 점(구름, 없으면 상자 격자) absent_samples 개를 투영해 시야 안·안 가림 비율이 absent_vis 이상,
  // 보이는 부분 화소 크기 ≥ absent_min_px, 카메라 거리 ≤ 이 물체를 검출했던 가장 먼 거리·absent_det_k + 0.2 일 때만 놓침 +1.
  // 큰 것(한 변 > big)·고정 종류도 판정하되 gone_misses_big 번·gone_min_s_big 초. 시간 대신 카메라가 gone_view_d 넘게 옮긴 시점의 놓침도 됨
  int absent_samples = 48;
  double absent_vis = 0.5;
  double absent_min_px = 12;
  double absent_det_k = 1.15;
  int gone_misses_big = 6;
  double gone_min_s_big = 4.0;
  int spurious_obs = 5;            // 관측이 이보다 적은 확정 물체가 사라짐이 되면 헛검출로 보고 지움
  double gone_view_d = 0.5;
  double gone_step_d = 0.10, gone_step_deg = 5.0;   // 새 놓침 근거: 지난 놓침 뒤 카메라가 이만큼 옮기거나 돌았을 때(또는 자리 너머가 보일 때)만 셈
  double gone_eps = 0.02;         // 검출률 p 인 물체가 있는데 연속 k 번 놓칠 확률 (1−p)^k 가 이보다 작아야 사라짐(k 하한)       // 또는 첫 놓침 자리에서 카메라가 이만큼(m) 옮긴 뒤에도 놓침이면(다른 시점의 근거) 시간을 안 기다림
  // 옮겨짐 잇기(다시 잇기): '사라짐' m 과 새로 나타난 같은 이름 물체 n 을 잇는다(n 의 관측을 m 의 id 로). 조건
  //   시간: n 을 처음 본 때 > m 을 마지막으로 본 때, n 이 확정·관측 link_min_obs 번 이상
  //   거리: |n − m| ≤ min(link_max_d, link_d0 + link_v·(n 처음 − m 마지막))
  //   나타남: n 자리(view_cell 칸)를 n 을 처음 보기 link_view_gap_s 넘게 전에, n 을 처음 검출한 거리(수평, + view_cell) 이하에서
  //           본 적 있음(그때 있었으면 검출했을 것). 처음 가 본 곳에서 찾은 것은 새 물체(잇지 않음)
  //   애매함: n 에 더 가까운 같은 이름 물체가 n 이후로 아직 안 보였으면(사라졌을지 모름) link_wait_s 까지 기다림
  double link_v = 1.0, link_d0 = 1.0, link_max_d = 8.0;
  int link_min_obs = 3;
  double link_wait_s = 30.0;
  double link_window_s = 180.0;   // n 을 처음 본 뒤 이만큼 지나면 더는 잇기 후보가 아님(새 물체로 자리 잡음)
  double link_view_gap_s = 5.0;
  double view_cell = 0.5;
  // 움직이는 중: 관측 중심이 move_n 번 잇달아 move_v(m/s) 넘게 같은 쪽으로 가고 쉬던 자리에서 move_min_d 넘게(또는 상자 반 폭)
  // 벗어나면 관측 자리로 바로 따라감(평균 안 함). 마지막으로 그렇게 따라간 뒤 1 s 안이면 '움직이는 중'(moving_t). 0 = 끔
  int move_n = 3;
  double move_v = 0.3, move_min_d = 0.25;
  double move_max_cam_w = 0.6;    // 카메라 광축이 이보다 빨리(rad/s) 돌면 그 keyframe 은 움직임 근거로 안 씀(자세 오차가 물체를 쓸어 감).
                                  // 상자가 영상 가장자리에 닿은 관측(잘림)도 안 씀
  // ---- objprob(10-05, objprob.hpp): 이름 없는 같은 것 판정·기하 구조물 거르기·물체 임베딩 vMF·이름 사후. 끄면(기본) 위 규칙 그대로.
  // 켜려면 검출마다 임베딩(ObjFrame.emb)이 있어야 한다(capi sm_set_object_model·sm_set_det_embeddings)
  bool objprob = false;
  bool ap_name_struct_skip = false; // true: 검출 하나의 이름(cls)이 구조물이면 버림(옛 규칙). 끔(기본): FastSAM 조각의 이름은 자주 틀려
                                    // (소파 조각 → partition·baseboard) 기하·합친 물체의 이름 사후로만 거름
  ApParams ap;
  KappaParams kap;
  double ap_q_pos = 1e-4;         // 칼만: 물체 자리 과정 잡음(m²/s, 가만히 있는 물체)
  double ap_r0 = 0.02, ap_r1 = 0.01;   // 관측 중심 잡음 σ = r0 + r1·깊이(m) — 조각·잘림이면 반 폭을 더함
};

// objprob 진단 셈(ObjectMap::apStats)
struct ApStats {
  long n_through = 0, n_obs = 0, n_wall = 0, n_wall_name = 0, n_ceil = 0, n_floor = 0, n_name_struct = 0;
  long n_assoc = 0, n_new = 0, n_merge = 0, n_obj_struct = 0, n_reenc_req = 0, n_reenc_done = 0;
  long n_blocked = 0;     // 벽 같은 조각·납작한 이름 크기 밖이라 안 붙이고 버린 관측
  long n_wall_tall = 0;   // 벽 선 없이 높고 넓은 세운 평면이라 벽
  long n_so_big = 0;   // 문·창·기둥 이름 조각이지만 그 모양보다 커서 보호 안 함
};

// objprob 통째 다시 담기 요청(검출 마스크 격자와 같은 배치의 마스크)
struct ReencReq {
  uint32_t id = 0;
  float box[4] = {0, 0, 0, 0};    // 검출 영상 화소
  float kappa = 0;
  double cam[6] = {0, 0, 0, 0, 0, 1};
  std::vector<uint32_t> bits;     // ceil(mask_w·mask_h / 32) 단어
};

struct ObjEvent {
  double t;
  uint32_t id;
  int kind;                       // 0 새 후보, 1 확정, 2 옮겨짐, 3 사라짐, 4 들기, 5 놓기, 6 다시 보임, 7 병합(id = 남은 물체)
  double pos[3];
};

struct MapObject {
  uint32_t id = 0;
  int cls = -1;
  double pos[3] = {0, 0, 0}, ext[3] = {0, 0, 0}, first_pos[3] = {0, 0, 0};
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // map 축 상자(관측 10~90 백분위). 큰 물체는 합집합으로 자란다
  uint32_t n_obs = 0;
  double first_seen = 0, last_seen = 0;
  int32_t state = SM_SEEN;
  bool confirmed = false;
  bool moved = false;
  int misses = 0;
  double first_miss = 0;          // 연속 놓침이 시작된 시각
  double miss_cam[3] = {0, 0, 0}; // 그때 카메라 자리(map)
  double last_miss_cam[3] = {0, 0, 0}, last_miss_fwd[3] = {0, 0, 0};   // 마지막으로 센 놓침의 카메라 자리·광축
  uint32_t n_vis_miss = 0;
  double trk_pos[3] = {0, 0, 0}, trk_t = 0, trk_step[2] = {0, 0};   // 마지막 관측 중심(날 것)·시각·한 걸음
  int mv_cnt = 0;                 // 잇달아 빠르게 같은 쪽으로 간 관측 수
  double moving_t = -1e300, move_from_t = 0;   // 마지막으로 움직임을 따라간 시각, 이번 움직임 시작 시각        // 보일 만한데 검출이 없던 keyframe 수(평생) — 검출률 추정
  int held_by = -1;               // 0 왼손, 1 오른손
  double held_rel[3] = {0, 0, 0}, grasp_pos[3] = {0, 0, 0};   // held_rel: 팔 끝 → 물체, 베이스 축(로봇이 돌면 같이 돔)
  uint32_t parent = 0;            // 놓을 때 다른 물체(들고 있는 통·탁자) 위·안이면 그 물체에 붙어 같이 움직임
  double parent_rel[3] = {0, 0, 0};
  float score = 0;
  double last_kf = -1;
  ObjCloud cloud;                 // 모양(점 구름): 옮겨짐 잇기(사라짐 → 다른 자리)면 새 자리 물체의 것으로, 들기·받침은 org 를 옮김
  std::vector<std::pair<int32_t, float>> votes;   // 이름 표: (이름 번호, 점수 합)
  double max_det_z = 0;           // 이 물체를 검출한 가장 먼 카메라 깊이(사라짐 판정은 이 거리 안에서만)
  double gone_t = 0;              // 사라짐 판정 시각
  bool appeared = false;          // 전에 본 자리에 새로 나타남(옮겨짐 잇기 후보)
  ApStatePtr ap;                  // objprob 상태(임베딩·이름 사후·칼만 분산) — objprob 일 때만
};

// 이름 번호의 종류
// kKindStructObj(objprob 만): 문·창·계단 — 지우지 않고 노드(구조 물체: structural·안 옮김)로 내보냄. 벽·바닥·천장만 kKindStructure(지움)
enum ClassKind : uint8_t { kKindObject = 0, kKindStructure = 1, kKindStatic = 2, kKindStructObj = 3 };

// 한 keyframe 의 검출 k → 물체(update 가 채움, lastAssoc()). obj_id 0 = 물체에 안 붙음(점 부족·손에 든 것 등)
struct DetAssoc {
  uint32_t obj_id = 0;
  int n_valid = 0;                // 마스크 안 유효 깊이 점(step 간격 표본)
  float area_px = 0;              // 유효 마스크 넓이(깊이 화소) = n_valid·step²
  float depth_med = 0;            // 마스크 안 깊이 중앙값 m(카메라 z)
};

// 한 keyframe 입력
struct ObjFrame {
  double stamp = 0;
  int w = 0, h = 0;
  const float* depth_m = nullptr;     // 또는
  const uint16_t* depth_mm = nullptr;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  double T_mc[12] = {0};              // map ← 카메라 광학(행 우선 3×4)
  const sm_detections* dets = nullptr;
  double eef[2][3] = {{0}};           // map 기준 팔 끝
  float grip[2] = {0.1f, 0.1f};       // 손가락 합
  double base_yaw = 0;                // map 기준 베이스 yaw
  double base_xy[2] = {0, 0};         // map 기준 베이스 위치(몸 점 거르기)
  const Capsule* self_caps = nullptr; // map 기준 로봇 팔 캡슐(순기구학) — 이 안 깊이 점은 버림. NULL = 안 거름(R1)
  int n_self_caps = 0;
  // objprob: 검출마다 SigLIP 임베딩(dets->n × emb_dim, L2 정규화), 벽 선분(map, ax ay bx by — 기하 구조물 거르기)
  const float* emb = nullptr;
  int emb_dim = 0;
  const double* wall_segs = nullptr;
  int n_wall_segs = 0;
};

// 마지막 update 에서 물체에 붙은 관측의 점 구름 후보(관측 안에서 복셀마다 하나, map 좌표). 색은 호출자가 영상에서 골라
// addPoints 로 넣는다(RGB 가 장치에 있을 수 있어서 — sgrt 는 이 화소만 장치에서 모음).
struct ObsPoints {
  uint32_t obj_id = 0;
  int det = -1;
  std::vector<float> xyz;             // 3n, map
  std::vector<int32_t> px;            // 2n, 검출 영상 화소 x, y
};

class ObjectMap {
 public:
  // 진단: 환경 변수 SM_OBJ_PARAMS="key=val,key=val"(바뀜 판정 매개변수 이름 — objmap.cpp kEnvKeys)이 있으면 덮어씀
  explicit ObjectMap(const ObjParams& p = {}) : p_(p) { envOverrides(&p_); }
  static void envOverrides(ObjParams* p);
  void update(const ObjFrame& f);
  // 팔 끝·그리퍼만 바뀐 스텝(영상 없음)에도 들고 있는 물체를 따라가게
  void updateHands(double stamp, const double eef[2][3], const float grip[2], double base_yaw);
  const std::vector<MapObject>& objects() const { return objs_; }
  const std::vector<ObjEvent>& events() const { return ev_; }
  // 이름 번호 → 종류(ClassKind). 표 밖 번호는 물체
  void setClassKinds(std::vector<uint8_t> k) { kinds_ = std::move(k); }
  int kindOf(int cls) const { return cls >= 0 && size_t(cls) < kinds_.size() ? kinds_[cls] : int(kKindObject); }
  // 바닥에 깔리는 이름(러그·카펫·매트): 바닥 조각 거르기(floor_h)에서 뺌. 표 밖 번호는 아님
  void setFloorClasses(std::vector<uint8_t> f) { floor_cls_ = std::move(f); }
  bool floorLevel(int cls) const { return cls >= 0 && size_t(cls) < floor_cls_.size() && floor_cls_[cls]; }
  // 이름 표에서 cls 의 몫(0..1). 표가 비면 cls == m.cls 일 때 1
  static double nameShare(const MapObject& m, int cls);
  // 마지막 update 의 검출별 짝(크기 = dets->n, 검출 순서)
  const std::vector<DetAssoc>& lastAssoc() const { return assoc_; }
  std::vector<ObsPoints>& lastPoints() { return points_; }
  // 물체 id 의 구름에 map 좌표 점 n 개(rgb 3n, NULL = 회색)
  void addPoints(uint32_t id, const float* xyz, const uint8_t* rgb, int n, double stamp);
  void setCloudParams(double voxel, int cap) {
    if (voxel > 0) p_.voxel = voxel;
    if (cap > 0) p_.cloud_cap = cap;
  }
  const ObjParams& params() const { return p_; }
  ObjParams& paramsMut() { return p_; }
  // ---- objprob ----
  void setTextModel(ApText t) { text_ = std::move(t); }
  const ApText& textModel() const { return text_; }
  const ApStats& apStats() const { return aps_; }
  double ceilingEstimate() const { return ceil_est_; }
  // 통째 다시 담기 요청: 이번 keyframe 에 본 물체 중 아직 통째가 없거나(합친 뒤) 지금 모습이 훨씬 좋은 것. 구름을 지금 영상에 투영
  // (깊이로 가림 확인)해 검출 마스크 격자(mw × mh, 화소 = 칸 × s + o)에 칠한다. 이 keyframe 의 update·addPoints 뒤에 부른다
  void buildReencode(const ObjFrame& f, int img_w, int img_h, int mw, int mh, float msx, float msy, float mox, float moy,
                     std::vector<ReencReq>* out);
  // 통째 임베딩 하나(z: dim, L2) — 조각 벡터 대신 μ·이름에 쓰임
  void addWholeView(uint32_t id, const float* z, int dim, double kappa, double stamp, const double cam[6]);
  // 노드로 내보낼 물체인가: 확정 + (objprob export_named 면) 이름이 정해졌고 구조물 이름이 아님
  bool exportable(const MapObject& m) const {
    if (!m.confirmed) return false;
    if (!p_.objprob || !p_.ap.export_named || !m.ap) return true;
    if (m.ap->post.empty() || m.ap->hide) return false;
    if (text_.object_label >= 0 && m.cls == text_.object_label) return false;
    return kindOf(m.cls) != kKindStructure;   // 문·창·계단(kKindStructObj)은 내보냄
  }
  // 바깥 이름 관측(confirm_object 등): 라벨 lab 에 로그 우도비 log_lr 를 더하고 이름을 다시 셈(영상 모습이 와도 남음)
  void observeName(uint32_t id, int lab, double log_lr);

 private:
  void event(double t, const MapObject& o, int kind);
  void vote(MapObject& m, int cls, float w) const;
  bool nameOk(const MapObject& m, int cls) const;
  bool isBig(const MapObject& m) const;
  // 사라짐 근거: 2 = 안 보이고 자리 너머가 보임, 1 = 보여야 하는데 안 보임(놓침), 0 = 판단 못 함(시야 밖·가림·멀다·작다)
  int absentEvidence(const MapObject& m, const ObjFrame& f) const;
  void markView(const ObjFrame& f);
  double firstView(double x, double y, double range) const;   // range(수평 m) 이하에서 처음 본 시각(없으면 1e300)
  void relink(double t);
  void remapId(uint32_t from, uint32_t to);
  // objprob
  void apMergePass(double t);
  void apRename(MapObject& m);
  bool apStructObject(MapObject& m);
  bool labelRelated(int a, int b) const;
  ApPair apPairObj(MapObject& a, MapObject& b);
  const std::vector<uint64_t>& apContactIdx(MapObject& m);
  ApText text_;
  ApStats aps_;
  std::vector<uint8_t> smask_;        // 라벨이 구조물 종류인가(apMergePass)
  // 이름(사후 최대)이 납작한 벽걸이(액자·TV …)인가
  bool flatNamed(const MapObject& m) const {
    return m.ap && m.ap->name_lab >= 0 && size_t(m.ap->name_lab) < text_.flat_ok.size() && text_.flat_ok[size_t(m.ap->name_lab)];
  }
  std::vector<double> ceil_obs_;      // 천장 추정: 얇은 수평 관측(중앙 높이 > 1.8 m)의 높이(최근 400)
  double ceil_est_ = 0;               // 0 = 아직 모름
  std::vector<double> wsegs_;         // 마지막 keyframe 의 벽 선분(창 자리 이음 포함) — 물체 기하 판정
  double cam_w_ = 0;
  ObjParams p_;
  std::vector<MapObject> objs_;
  std::vector<ObjEvent> ev_;
  std::vector<DetAssoc> assoc_;
  std::vector<ObsPoints> points_;
  std::vector<uint8_t> kinds_, floor_cls_;
  uint32_t next_id_ = 1;
  bool closed_[2] = {false, false};
  // grasp_check: 그리퍼 값 기준·그때 시각(멈춤 판정), 이번 닫힘에서 이미 골랐는지
  double gref_[2] = {0, 0}, gref_t_[2] = {-1e300, -1e300};
  bool tried_[2] = {false, false};
  double gripGap(double g) const;
  bool holdable(const MapObject& o, double gap) const;
  void release(double t, MapObject& o);
  // update 작업 버퍼(keyframe 마다 재사용)
  std::vector<double> wx_, wy_, wz_, wzc_, wzs_;
  std::vector<int32_t> wpu_, wpv_, wcol_;
  VoxelIndex wseen_;
  double last_cam_t_ = -1e300, last_cam_fwd_[3] = {0, 0, 0};
  static constexpr int kViewBands = 5;                 // 거리 띠 1..5 m
  std::unordered_map<int64_t, std::array<double, kViewBands>> view_first_;   // 칸 → 띠마다 처음 본 시각(-1e300 = 아직)
};

}  // namespace scenemap
