// scenemap ② objmap — 검출(마스크 + 이름 번호) + 깊이 + 자세 → 물체 3D 위치 → 같은 물체 판단 → 바뀐 부분만 갱신.
//
// docs/scenemap_설계.md 3.2. 규칙 요약
//   종류   : 이름 번호마다 물체 / 구조물(벽·바닥·천장·문·창·기둥·칸막이·계단·난간·걸레받이 — 물체가 안 되고 격자만) /
//            고정(가구·가전·붙박이 — 물체 노드지만 movable=false, 사라짐은 큰 것 기준으로 판정). 표는 capi(sm_set_kind_names)가 정함
//   위치   : 마스크를 1 칸 깎은 안쪽 깊이 점(map)의 축별 중앙값, 크기 = 10~90 백분위 폭.
//            그 전에 카메라 깊이가 마스크 안 중앙값에서 max(mad_k·1.4826·MAD, mad_floor) 넘게 떨어진 점은 버림(뒤 벽이 비친 것)
//   거르기 : 점 min_points 미만 버림. 점의 hand_frac 이상이 팔 끝 hand_r 안이면(손에 든 것) 버림. 점의 90 백분위 높이 < floor_h 면
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
//            열리는 순간 그 자리에 놓는다(moved_d 넘게 옮겼으면 옮겨짐). 놓은 점 아래에 xy 가 겹치는(0.1 m 여유) 다른 물체
//            상자가 있으면 그중 윗면이 가장 높은 것(떨어져 닿을 받침)에 붙인다 — 들고 있는 쓰레기통에 넣은 캔이 통을 따라가게
#pragma once
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "scenemap.h"
#include "scenemap/cloud.hpp"

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
};

// 이름 번호의 종류
enum ClassKind : uint8_t { kKindObject = 0, kKindStructure = 1, kKindStatic = 2 };

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
  ObjParams p_;
  std::vector<MapObject> objs_;
  std::vector<ObjEvent> ev_;
  std::vector<DetAssoc> assoc_;
  std::vector<ObsPoints> points_;
  std::vector<uint8_t> kinds_, floor_cls_;
  uint32_t next_id_ = 1;
  bool closed_[2] = {false, false};
  // update 작업 버퍼(keyframe 마다 재사용)
  std::vector<double> wx_, wy_, wz_, wzc_, wzs_;
  std::vector<int32_t> wpu_, wpv_, wcol_;
  VoxelIndex wseen_;
  double last_cam_t_ = -1e300, last_cam_fwd_[3] = {0, 0, 0};
  static constexpr int kViewBands = 5;                 // 거리 띠 1..5 m
  std::unordered_map<int64_t, std::array<double, kViewBands>> view_first_;   // 칸 → 띠마다 처음 본 시각(-1e300 = 아직)
};

}  // namespace scenemap
