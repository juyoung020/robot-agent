// scenemap — 살아 있는 장면 그래프(Hydra 식 층 구조, 2D 지도에 맞춤). docs/scenemap_설계.md 3.5.
//
// 층(Spark-DSG 번호 그대로 — 기본 spark_dsg 파이썬이 읽음):
//   2 OBJECTS  (partition 0)   확정 물체 'O'<id>  — objmap 이 keyframe 마다 고침(자리·상자·상태·movable)
//   2 AGENTS   (partition 'a') 로봇 keyframe 자세 'a'<k> — 0.5 m·30° 움직이거나 10 s 마다 하나, 앞 노드와 변
//   3 PLACES                   빈 공간 뼈대 'p'<k> — 2D 격자 빈칸 거리 변환(장애물·모름까지)의 능선(GVD 흉내)을 성글게 고른 것.
//                              속성 = 여유(장애물까지 m), frontier(가장 가까운 막힘이 모름). 변 = 직선 시야가 빈칸인 이웃,
//                              무게 = 변을 따라 가장 작은 여유(Hydra 병목 무게)
//   4 ROOMS                    방 'R'<id>(rooms.hpp), 방–방 변 = 문(위치·폭)
//   (Map_Vla: 건물 층은 없다. 뷰어는 물체·방 2층만 그리고 place 는 그리지 않는다 — place·frontier 는 이동·탐색(move_robot, explore)이 쓰는 백엔드 계산.)
//   층 사이: 물체 → 가장 가까운 place, place → 방, agent → 가장 가까운 place, 방 → 물체.
//   물체끼리 관계(on/in/near)는 만들지 않는다(Map_Vla) — 위치·상자 메타데이터로 소비자가 추론. 물체의 부모: 방, place.
// 갱신(Hydra 앞단처럼 활성 창만): 격자 보이는 값이 바뀐 상자(+2 m)만 다시 계산 — place 를 지우고 다시 고르고(가까운 옛 id 는
//   다시 씀), 그 둘레 변만 다시 잇는다. 물체·agent 는 keyframe 마다 바뀐 것만. 방은 방 나눔이 바뀔 때만.
// 읽기: publish() 가 바뀌었을 때만 바뀌지 않는 사본(GraphView)을 만들고 스냅숏이 포인터만 나눠 씀.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "scenemap.h"
#include "scenemap/geom.hpp"
#include "scenemap/rooms.hpp"

namespace scenemap {

constexpr uint64_t nodeSym(char c, uint64_t idx) { return (uint64_t(uint8_t(c)) << 56) | (idx & 0x00FFFFFFFFFFFFFFull); }
inline char symChar(uint64_t id) { return char(id >> 56); }
inline uint64_t symIdx(uint64_t id) { return id & 0x00FFFFFFFFFFFFFFull; }

enum GRel : int32_t {
  kRelGeneric = 0,   // 층 사이(부모 → 자식)
  kRelPlace = 1,     // place–place(무게 = 병목 여유)
  kRelDoor = 2,      // 방–방(무게 = 문 폭)
  kRelAgent = 6,     // agent 앞뒤
};

struct GNode {
  uint64_t id = 0;
  int32_t layer = 0, partition = 0;
  double pos[3] = {0, 0, 0};
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // 상자(없으면 lo = hi = pos)
  std::string name;
  int32_t state = 0;            // 물체: SM_SEEN..; place: 1 = frontier
  float clearance = 0;          // place: 여유 m, 방: 가장 큰 여유
  double yaw = 0, stamp = 0;    // agent: 자세, 시각
  int32_t movable = 0;          // 물체
  uint32_t ver = 0;             // 바뀔 때마다 +1(저장 캐시)
};

struct GEdge {
  uint64_t a = 0, b = 0;        // 물체 관계는 a = 주어("a on b")
  float weight = 1.f;
  int32_t rel = kRelGeneric;
  float pos[2] = {0, 0};        // 문: 자리
};

// 바뀌지 않는 사본(스냅숏이 나눠 씀)
struct GraphView {
  std::vector<GNode> nodes;               // 층 순(OBJECTS, AGENTS, PLACES, ROOMS, BUILDINGS) 안에서 id 순
  std::vector<GEdge> edges;
  std::vector<uint32_t> adj_off, adj;     // nodes 자리 기준 CSR 이웃(edges 번호)
  std::unordered_map<uint64_t, uint32_t> index;   // id → nodes 자리
  std::vector<sm_gnode> cn;               // C ABI 사본(nodes 와 같은 순서, name 은 nodes[i].name 을 가리킴)
  int32_t layer_off[6] = {0};             // [OBJECTS, AGENTS, PLACES, ROOMS, BUILDINGS, 끝]
  uint64_t version = 0;
  const GNode* find(uint64_t id) const {
    auto it = index.find(id);
    return it == index.end() ? nullptr : &nodes[it->second];
  }
};

struct GraphParams {
  double win_margin = 2.0;      // 바뀐 상자 둘레 다시 계산 폭 m
  double min_clear = 0.20;      // place 가 되는 여유 하한(로봇 반폭 근처)
  double max_clear = 1.0;       // 거리 변환 상한(이보다 먼 장애물은 같음 — 창 여백도 이만큼)
  double spacing_min = 0.5, spacing_max = 1.5;   // place 사이 간격 = clamp(1.5·여유, …)
  double edge_r = 2.0;          // place 변 최대 길이
  int edge_k = 6;               // place 마다 가까운 이웃 최대
  double edge_min_clear = 0.15; // 변을 따라 여유 하한
  double frontier_tol = 0.15;   // 여유 + 이것 반지름 원 위에 모름이 있으면 frontier
  double period_s = 0.5;        // place 다시 계산 최소 간격(시뮬 s)
  double agent_xy = 0.5, agent_yaw = 30 * M_PI / 180, agent_s = 10.0;
};

struct ObjIn {                  // 물체 층 입력(확정 물체)
  uint32_t id;
  std::string name;
  double pos[3], lo[3], hi[3];
  int32_t state;
  int32_t movable;
};

class SceneGraph {
 public:
  explicit SceneGraph(const GraphParams& p = {}) : p_(p) {}
  void reset();
  // 물체(확정만) — 바뀐 것만 버전 올림, 없어진 것 지움, 물체 관계·물체 → place 다시
  void updateObjects(const std::vector<ObjIn>& objs);
  // 로봇 자세(keyframe) — 움직였으면 agent 노드 하나
  void updateAgent(double stamp, const Pose2& pose);
  // 격자(보이는 값 −1/0..100) — dirty 상자(전역 칸, 끝 포함)가 있으면 그 둘레 place 다시. force = 전부
  void updatePlaces(const int8_t* cells, int w, int h, int gx0, int gy0, double res, double stamp, const int* dirty_box, bool force);
  // 이번에 updatePlaces 가 실제로 계산할까(격자 사본을 미리 만들지 판단)
  bool placesDue(double stamp, bool dirty, int w, int h, int gx0, int gy0) const {
    const bool reshaped = w != cw_ || h != ch_ || gx0 != cgx0_ || gy0 != cgy0_;
    return reshaped || ((dirty || pending_dirty_) && stamp - last_places_ >= p_.period_s);
  }
  // 방 나눔(바뀌었을 때) + 물체 → 방
  void updateRooms(const std::shared_ptr<const RoomSeg>& rs, const std::vector<std::pair<uint32_t, uint32_t>>& obj_room,
                   const std::vector<std::string>& room_names);
  // 바뀌었으면 새 사본, 아니면 그대로
  std::shared_ptr<const GraphView> publish();
  const GraphParams& params() const { return p_; }
  // 시간(µs, 마지막 호출)
  double us_places = 0, us_objects = 0, us_publish = 0;
  int n_place_updates = 0;

 private:
  GNode& upsert(uint64_t id, int layer, int partition);
  void erase(uint64_t id);
  void eraseEdgesOf(uint64_t id, int rel_mask);
  void addEdge(uint64_t a, uint64_t b, float w, int rel, const float* pos = nullptr);
  uint64_t nearestPlace(double x, double y, double rmax) const;
  void linkObjectsToPlaces();
  void linkAgentsToPlaces(double x0, double y0, double x1, double y1);
  void linkPlacesToRooms();

  GraphParams p_;
  std::unordered_map<uint64_t, GNode> nodes_;
  std::unordered_map<uint64_t, std::vector<GEdge>> adj_;   // id → 그 노드가 가진 변(양쪽에 한 벌씩)
  bool changed_ = true;
  uint64_t version_ = 0;
  std::shared_ptr<const GraphView> view_;
  // places
  uint64_t next_place_ = 1, next_agent_ = 0;
  double last_places_ = -1e9;
  bool pending_dirty_ = false;
  int pd_[4] = {0, 0, 0, 0};
  std::vector<double> edt_, z_;
  std::vector<float> clr_;
  int win_[4] = {0, 0, 0, 0};
  std::vector<int> v_;
  // 지금 격자(마지막 updatePlaces) — place → 방, 변 시야 검사
  std::vector<int8_t> cells_;
  int cw_ = 0, ch_ = 0, cgx0_ = 0, cgy0_ = 0;
  double cres_ = 0.05;
  std::shared_ptr<const RoomSeg> rooms_;
  Pose2 last_agent_;
  double last_agent_t_ = -1e9;
  bool have_agent_ = false;
};

}  // namespace scenemap
