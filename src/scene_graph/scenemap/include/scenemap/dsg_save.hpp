// scenemap 저장 — 로봇 기억을 파일로(sm_save_dsg). docs/scenemap_설계.md 3절 "save".
//   scene.json : Spark-DSG DynamicSceneGraph. OBJECTS 층에 확정 물체 하나 = 노드 하나(NodeSymbol 'O', id).
//                위치 = map xyz, 상자 = 위치 ± 크기/2, 이름 = 프롬프트 이름, 메타데이터 = 상태·관측 수·처음 위치·마지막 시각.
//   view.json  : 계획기·뷰어용 요약(자세, 물체, 최근 사건). 의존 없는 손 JSON.
//   map.pgm/.yaml : 2D 점유 격자(ROS map_server 형식: 254 빈칸, 0 점유, 205 모름).
//   objects/O<id>_rgb.png · O<id>_depth.png : 물체별 best view(8 비트 RGB · 16 비트 회색 mm). 모습이 바뀐 것(png_dirty)
//                이나 파일이 없는 것만 다시 쓴다. scene.json 노드 metadata.rgbd · view.json objects[].rgbd 가 이 경로를 가리킨다.
//   objects/O<id>_mask.png : best view 마스크(8 비트 회색, 255 = 마스크 안, rgb·depth 와 같은 상자·크기).
//   objects/O<id>_points.ply : 물체 점 구름(binary_little_endian, float x,y,z map m + uchar red,green,blue). 구름이 바뀐 것
//                (ply_dirty)이나 파일이 없는 것만. 노드 metadata.points 가 가리킨다.
//   rooms.pgm : 방 칸 그림(map.pgm 과 같은 크기·방향, 8 비트, 0 = 방 없음, k = view.json rooms[] 의 value). 방 나눔이 있을 때만.
//                scene.json ROOMS 층(NodeSymbol 'R', 방 id)·방→물체 변·방–방 변(문) — rooms.hpp
//   순서: PNG·PLY → scene.json → view.json(뷰어는 view.json 이 바뀌면 다시 읽으니 그때는 가리키는 파일이 다 있다).
// 모든 파일은 임시 이름으로 쓰고 rename 으로 바꾼다(읽는 쪽이 반쯤 쓴 파일을 보지 않게).
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/cloud.hpp"
#include "scenemap/objmap.hpp"
#include "scenemap/rooms.hpp"
#include "scenemap/sgraph.hpp"

namespace scenemap {

// 노드 JSON 조각 캐시: 노드 id → (그래프 노드 ver, JSON). 바뀐 노드만 다시 씀
struct JsonCache {
  std::unordered_map<uint64_t, std::pair<uint32_t, std::string>> nodes;
};

struct SaveInput {
  double stamp = 0;
  double pose[3] = {0, 0, 0};      // x, y, yaw (map)
  const sm_object* objs = nullptr;
  int n_objs = 0;
  std::vector<ObjEvent> events;    // 최근 사건(오래된 것부터)
  double grid_res = 0.05, grid_ox = 0, grid_oy = 0;
  int grid_w = 0, grid_h = 0;
  const int8_t* cells = nullptr;   // −1 모름, 0..100 점유 %
  // 물체별(objs 와 같은 순서, 비어 있어도 됨)
  std::vector<BestViewPtr> views;  // best view(없으면 null)
  std::vector<uint8_t> png_dirty;  // 1: 지난 저장 뒤 모습이 바뀜
  std::vector<uint8_t> movable;    // 1: 옮길 수 있는 물체, 0: 가구·가전·붙박이(비어 있으면 모두 1)
  std::vector<ObjCloud> clouds;    // 점 구름(objs 와 같은 순서, 비어 있어도 됨)
  std::vector<uint8_t> ply_dirty;  // 1: 지난 저장 뒤 구름이 바뀜
  double voxel = 0.02;
  bool clean_objects = false;      // objects/ 에서 지금 물체가 아닌 O<id>_*.png 지우기(새 판·새 디렉터리)
  std::shared_ptr<const RoomSeg> rooms;   // 방 나눔(null = 방 없음 — 방 파일·키 안 씀)
  RoomNaming room_names;           // rooms->rooms 와 같은 순서, obj_room 은 objs 순서
  std::shared_ptr<const GraphView> graph;   // 살아 있는 장면 그래프(AGENTS·PLACES·BUILDINGS 층과 모든 변) — null 이면 물체·방만
  std::vector<std::string> obj_meta;        // objs[i] 노드 metadata 에 덧붙일 JSON 멤버("\"emb\":{...}" 꼴, 비어 있어도 됨)
  bool stream_lite = false;                 // 실시간 스트림용 요약: PLACES 노드·변을 뺀다(뷰어가 안 그리고 용량 대부분)
  struct JsonCache* json_cache = nullptr;   // place·agent·방·건물 노드 JSON 캐시(저장 사이 재사용, 호출자 소유)
};

struct SaveOut {
  int n_png = 0;                   // 이번에 쓴 PNG 파일 수
  double png_ms = 0;               // PNG 만들기·쓰기 시간
  std::vector<uint8_t> png_ok;     // objs[i]: objects/ 에 지금 모습 파일이 있음(이번에 썼거나 그대로)
  std::vector<uint8_t> ply_ok;     // objs[i]: objects/ 에 지금 구름 PLY 가 있음
  int n_ply = 0;                   // 이번에 쓴 PLY 수
  double ply_ms = 0;
  double json_ms = 0;              // scene.json 만들기·쓰기
  size_t json_bytes = 0;
};

// view.json 내용만 문자열로(파일을 쓰지 않음) — 실시간 스트림용. out.png_ok·ply_ok 는 파일이 이미 있는 물체(지난 저장 기준)
std::string sceneViewJson(const SaveInput& in, const SaveOut& out);

// 0 = 성공. Spark-DSG 없이 빌드하면 scene.json 만 빠진다.
int saveScene(const SaveInput& in, const std::string& dir, SaveOut* out = nullptr);

}  // namespace scenemap
