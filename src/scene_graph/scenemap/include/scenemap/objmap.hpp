// scenemap ② objmap — 검출(마스크 + 이름 번호) + 깊이 + 자세 → 물체 3D 위치 → 같은 물체 판단 → 바뀐 부분만 갱신.
//
// docs/scenemap_설계.md 3.2. 규칙 요약
//   종류   : 이름 번호마다 물체 / 구조물(벽·바닥·천장·문·창·기둥·칸막이·계단·난간·걸레받이 — 물체가 안 되고 격자만) /
//            고정(가구·가전·붙박이 — 물체 노드지만 movable=false, 사라짐 판정 안 함). 표는 capi(sm_set_kind_names)가 정함
//   위치   : 마스크를 1 칸 깎은 안쪽 깊이 점(map)의 축별 중앙값, 크기 = 10~90 백분위 폭.
//            그 전에 카메라 깊이가 마스크 안 중앙값에서 max(mad_k·1.4826·MAD, mad_floor) 넘게 떨어진 점은 버림(뒤 벽이 비친 것)
//   거르기 : 점 min_points 미만 버림. 점의 hand_frac 이상이 팔 끝 hand_r 안이면(손에 든 것) 버림
//   같은 것: 같은 이름 번호끼리만. 중심 거리 < max(da_min, da_k·큰 쪽 크기) 이거나 map 축 상자 사이 틈 < da_gap.
//            틈(겹치면 0)·중심 거리 순으로 1:1(탐욕). 큰 가구(한 변 > big 또는 고정 종류)는 상자를 합집합으로 키우고
//            위치 = 상자 중심 — 단 keyframe 마다 면마다 grow_max 까지만, 한 변 max_ext 넘게는 안 키움
//   확정   : 서로 다른 keyframe 에서 confirm 번 보이면 물체. 후보가 prune_s 동안 다시 안 보이면 버림
//   옮겨짐 : 확정 물체가 moved_d 넘게 떨어진 자리에서 보이면 그 자리로 옮기고 이력을 남김
//   사라짐 : (작은 물체만, 한 변 ≤ big 이고 고정 종류 아님) 시야 안·가림 없음(깊이가 물체보다 occl 이상 가깝지 않음)·
//            팔 끝 hand_r+0.1 밖인데 안 보이면 +1, gone_misses 번 연속이고 첫 놓침에서 시뮬 gone_min_s 넘게 지나야 사라짐.
//            통 안에 붙은 것은 안 봄
//   들기   : 그리퍼가 닫히는 순간 팔 끝 grasp_r 안 가장 가까운 확정 물체를 든 것으로 — 드는 동안 팔 끝을 따라가고,
//            열리는 순간 그 자리에 놓는다(moved_d 넘게 옮겼으면 옮겨짐). 놓은 점 아래에 xy 가 겹치는(0.1 m 여유) 다른 물체
//            상자가 있으면 그중 윗면이 가장 높은 것(떨어져 닿을 받침)에 붙인다 — 들고 있는 쓰레기통에 넣은 캔이 통을 따라가게
#pragma once
#include <cstdint>
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
  int held_by = -1;               // 0 왼손, 1 오른손
  double held_rel[3] = {0, 0, 0}, grasp_pos[3] = {0, 0, 0};   // held_rel: 팔 끝 → 물체, 베이스 축(로봇이 돌면 같이 돔)
  uint32_t parent = 0;            // 놓을 때 다른 물체(들고 있는 통·탁자) 위·안이면 그 물체에 붙어 같이 움직임
  double parent_rel[3] = {0, 0, 0};
  float score = 0;
  double last_kf = -1;
  ObjCloud cloud;                 // 모양(점 구름): 옮겨짐 잇기(사라짐 → 다른 자리)면 비우고 새로, 들기·받침은 org 를 옮김
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
  explicit ObjectMap(const ObjParams& p = {}) : p_(p) {}
  void update(const ObjFrame& f);
  // 팔 끝·그리퍼만 바뀐 스텝(영상 없음)에도 들고 있는 물체를 따라가게
  void updateHands(double stamp, const double eef[2][3], const float grip[2], double base_yaw);
  const std::vector<MapObject>& objects() const { return objs_; }
  const std::vector<ObjEvent>& events() const { return ev_; }
  // 이름 번호 → 종류(ClassKind). 표 밖 번호는 물체
  void setClassKinds(std::vector<uint8_t> k) { kinds_ = std::move(k); }
  int kindOf(int cls) const { return cls >= 0 && size_t(cls) < kinds_.size() ? kinds_[cls] : int(kKindObject); }
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
  ObjParams p_;
  std::vector<MapObject> objs_;
  std::vector<ObjEvent> ev_;
  std::vector<DetAssoc> assoc_;
  std::vector<ObsPoints> points_;
  std::vector<uint8_t> kinds_;
  uint32_t next_id_ = 1;
  bool closed_[2] = {false, false};
  // update 작업 버퍼(keyframe 마다 재사용)
  std::vector<double> wx_, wy_, wz_, wzc_, wzs_;
  std::vector<int32_t> wpu_, wpv_, wcol_;
  VoxelIndex wseen_;
};

}  // namespace scenemap
