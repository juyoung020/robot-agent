// scenemap — 방 나누기(Hydra room finder 의 2D 판). docs/scenemap_설계.md 3.4.
//
// Hydra(RSS 2022) 는 places(GVD) 그래프 노드의 장애물 거리로 문턱을 키워 가며(dilation) 그래프를 끊고, 전역 문턱
// 하나(PLATEAU: 성분 수가 오래 같은 구간 중 수가 가장 많은 것)의 연결 성분을 방 씨앗으로 삼은 뒤 나머지 place 를
// 변 무게 큰 순으로 붙인다(NEIGHBORS). 여기서는 places 대신 2D 점유 격자의 빈칸을 그대로 쓰고(칸마다 방 id 가 필요),
// 문턱은 성분마다 수명으로 고른다(ToMATo 꼴 지속성 나누기 — 크기 다른 방이 섞여도 됨). docs/scenemap_설계.md 3.4:
//   1) 빈칸(0..free_max) / 점유(≥ occ_min) / 모름. 안쪽의 작은 모름 구멍은 빈칸으로, 빈칸에 둘러싸인 작은 점유 점
//      (의자 다리)은 지운다(나누기에만).
//   2) 빈칸의 유클리드 거리 변환(벽 면까지 여유 c = 칸 중심 거리 − res/2). 줄마다 빈칸 토막별 봉투.
//   3) 한 번 쓸기 거름 + 붙이기: c > dil_max 칸은 한 층이라 줄 토막 연결 성분으로 바로 묶고, 나머지 빈칸은 dil_step
//      통으로 한 번 세기 정렬해 큰 여유부터 넣는다. 칸은 들어온 이웃 중 여유가 가장 큰 칸의 씨앗을 물려받고(가파른
//      오르막 = 넘치기), 두 성분이 만나면 어린 쪽 수명(태어난 문턱 − 만난 문턱)이 min_life 이상이고 둘 다 min_seed
//      이상일 때 둘 다 방 씨앗으로 얼린다(Hydra barcode). 아니면 짧게 산 쪽 씨앗을 만난 자리 건너편 씨앗에 합친다.
//      dil_min 아래에서는 새 씨앗이 없고, 남은 큰 홑성분은 씨앗, 작은 것은 만나는 대로 건너편에 붙는다. 씨앗 없는
//      빈칸 덩이는 min_room 이상이면 방 하나. → 문 반폭 ≈ dil_max − min_life 보다 좁은 통로로만 이어진 곳이 다른 방.
//   4) 방마다 넓이·무게중심·상자·최대 여유와 이음매(두 방이 맞닿은 칸 쌍)를 한 번 쓸기로 모음.
//   5) 합치기: 이음매가 max_door 보다 길면 문이 아님 → 합침. min_room 보다 작은 방은 이음매가
//      가장 긴 이웃에 합침(이웃 없으면 버림).
//   6) 방–방 변(문): 남은 이음매마다 위치(가운데), 폭(2·최대 여유 + res).
//   7) id 유지: 이전 나눔과 칸 겹침(세계 칸 좌표, 격자가 넓어져도)으로 욕심쟁이 짝짓기 — 짝 없는 방만 새 id.
// 물체 배정: 바닥 자리(xy 상자 + 여유)의 방 칸 다수결, 없으면 obj_search 안 가장 가까운 방 칸.
// 이름: 들어 있는 물체 이름 규칙(냉장고·싱크·오븐·가스레인지·전자레인지 → kitchen, 침대 → bedroom, 소파·TV·커피
// 테이블 → living room, 변기·욕조 → bathroom, 책상·모니터 → office), 점수 → 확률·확신도·근거. 외부 이름(LLM 등)은
// RoomTracker::setName 으로 덮어씀(sm_set_room_name).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace scenemap {

struct RoomParams {
  bool enabled = true;
  int free_max = 49;            // export8 값 0..free_max = 빈칸(한 번 지나간 광선 = 40 %)
  int occ_min = 65;             // 이상 = 점유. 그 사이·모름(−1) = 막힘
  double dil_min = 0.30, dil_max = 0.60, dil_step = 0.025;   // 거름 문턱(벽 면까지 여유, m)
  double min_life = 0.10;       // 씨앗이 되려면 이만큼(문턱 m) 따로 살아야
  double min_seed_m2 = 0.5;     // 씨앗 최소 넓이(만날 때)
  double min_room_m2 = 2.0;     // 방 최소 넓이(다시 키운 뒤)
  double max_door_m = 1.6;      // 이음매가 이보다 길면 문이 아님 → 합침(0 = 안 함)
  double hole_m2 = 0.25;        // 이보다 작은 안쪽 모름 구멍 → 빈칸
  double speck_m2 = 0.01;       // 빈칸에 둘러싸인 이 이하 점유 덩이 → 빈칸(나누기에만)
  double obj_search_m = 1.0;    // 물체 자리에 방 칸이 없을 때 찾는 반경
  double footprint_margin = 0.15;
  double period_s = 3.0;        // 다시 나누기 주기(시뮬 시각)
  double min_change_m2 = 0.25;  // 빈칸 분류가 이만큼 바뀌었을 때만
  double match_min = 0.3;       // id 잇기: 겹침 ≥ match_min × 작은 쪽 넓이
};

// 격자 보기: cells[y·w + x] = 전역 칸 (gx0 + x, gy0 + y), 칸 (i, j) 왼쪽 아래 = (i, j)·res
struct GridView {
  const int8_t* cells = nullptr;
  int w = 0, h = 0;
  double res = 0.05;
  int gx0 = 0, gy0 = 0;
};

struct RoomGeom {
  uint32_t id = 0;
  int n_cells = 0;
  double area_m2 = 0;
  double centroid[2] = {0, 0};
  double bmin[2] = {0, 0}, bmax[2] = {0, 0};
  double max_clear = 0;         // 가장 넓은 곳의 벽까지 여유(m)
};

struct RoomDoor {
  uint32_t a = 0, b = 0;        // a < b(방 id)
  double pos[2] = {0, 0};       // 이음매 가운데
  double width = 0;             // 2·최대 여유 + res
  int seam = 0;                 // 맞닿은 칸 쌍 수
};

struct RoomSeg {
  int w = 0, h = 0, gx0 = 0, gy0 = 0;
  double res = 0.05;
  std::vector<uint32_t> ids;          // 칸마다 방 id(0 = 없음)
  std::vector<RoomGeom> rooms;        // id 오름차순
  std::vector<RoomDoor> doors;
  std::vector<uint8_t> rawfree;       // 나눌 때의 빈칸 분류(다시 나눌지 판정)
  std::vector<std::pair<double, int>> filtration;   // (문턱 m, 씨앗 크기 이상 성분 수) — 높은 문턱부터
  int n_seeds = 0;
  double stamp = 0;
  double ms = 0;                      // 나누기 시간
  uint32_t at(double x, double y) const;   // map xy → 방 id(0 = 없음/밖)
  int index(uint32_t id) const;            // rooms 안 자리(−1 = 없음)
};

// 한 번 나누기. 방 id = 1..n(크기 순 아님, 이전 기억 없음)
std::shared_ptr<RoomSeg> segmentRooms(const GridView& g, const RoomParams& p);
// prev 와 겹침으로 cur 의 id 를 바꿈(짝 없으면 *next_id++). ids·rooms·doors 를 고침
void matchRoomIds(RoomSeg& cur, const RoomSeg* prev, uint32_t* next_id, double match_min);

struct RoomObj {
  uint32_t id = 0;
  std::string name;
  double pos[3] = {0, 0, 0}, ext[3] = {0, 0, 0};
  bool movable = true;
};
struct RoomEvidence {
  uint32_t obj = 0;
  std::string name, type;       // 물체 이름, 가리킨 방 종류
  double w = 0;
};
struct RoomLabel {
  uint32_t id = 0;
  std::string name;             // "kitchen", "kitchen 2", "room 7", 외부 이름
  std::string type;             // "kitchen" ..., "" = 모름
  float conf = 0;
  std::map<std::string, double> probs;   // 종류별 확률 + "unknown"
  std::vector<RoomEvidence> evidence;
  std::vector<uint32_t> objects;
  bool external = false;        // 외부 이름(sm_set_room_name)
};
struct RoomNaming {
  std::vector<RoomLabel> rooms;     // seg.rooms 와 같은 순서
  std::vector<uint32_t> obj_room;   // objs 와 같은 순서(0 = 방 없음)
};
struct NameOverride {
  std::string name;
  float conf = 1;
};

uint32_t assignObject(const RoomSeg& s, const RoomObj& o, const RoomParams& p);
RoomNaming nameRooms(const RoomSeg& s, const std::vector<RoomObj>& objs, const RoomParams& p,
                     const std::unordered_map<uint32_t, NameOverride>* ov = nullptr);

// 판마다 하나. update 는 아무 스레드에서(나누기는 잠금 밖, 한 번에 하나 — 다른 쪽은 지난 결과를 받음)
class RoomTracker {
 public:
  void setParams(const RoomParams& p);
  RoomParams params() const;
  // 주기(period_s)가 지났고 빈칸이 min_change 이상 바뀌었으면(또는 force·처음) 다시 나눈다. 지금 결과(null 가능)
  std::shared_ptr<const RoomSeg> update(const GridView& g, double stamp, bool force = false);
  std::shared_ptr<const RoomSeg> current() const;
  void setName(uint32_t id, const char* name, float conf);   // name == NULL: 지움
  std::unordered_map<uint32_t, NameOverride> overrides() const;
  void reset();                 // 새 판(id 1 부터, 외부 이름 지움)
  int n_runs() const;           // 다시 나눈 횟수

 private:
  mutable std::mutex mu_;
  RoomParams p_;
  std::shared_ptr<const RoomSeg> cur_;
  uint32_t next_id_ = 1;
  bool busy_ = false;
  uint64_t epoch_ = 0;
  double last_check_ = -1e18;
  int runs_ = 0;
  std::unordered_map<uint32_t, NameOverride> ov_;
};

}  // namespace scenemap
