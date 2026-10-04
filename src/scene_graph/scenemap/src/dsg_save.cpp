// scenemap 저장 구현(include/scenemap/dsg_save.hpp).
#include "scenemap/dsg_save.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "scenemap/png.hpp"

#ifdef SM_HAVE_SPARK_DSG
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#endif

namespace scenemap {
namespace {

namespace fs = std::filesystem;

const char* stateName(int s) {
  switch (s) {
    case SM_SEEN: return "seen";
    case SM_GONE: return "gone";
    case SM_MOVED: return "moved";
    case SM_HELD: return "held";
  }
  return "?";
}

const char* eventName(int k) {
  static const char* n[] = {"candidate", "confirmed", "moved", "gone", "picked", "placed", "seen_again", "merged"};
  return k >= 0 && k < 8 ? n[k] : "?";
}

// JSON 문자열 이스케이프(이름은 프롬프트 표 — 따옴표·역슬래시만 막으면 된다)
std::string esc(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if (static_cast<unsigned char>(ch) >= 0x20) o += ch;
  }
  return o;
}

// 임시 파일에 쓰고 rename(같은 디렉터리라 원자적)
bool writeAtomic(const fs::path& path, const std::string& data) {
  const fs::path tmp = path.string() + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), std::streamsize(data.size()));
    if (!f) return false;
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  return !ec;
}

std::string objPath(uint32_t id, const char* what) { return "objects/O" + std::to_string(id) + "_" + what + ".png"; }

std::string plyPath(uint32_t id) { return "objects/O" + std::to_string(id) + "_points.ply"; }

bool hasView(const SaveInput& in, const SaveOut& out, int i) { return i < int(out.png_ok.size()) && out.png_ok[i] && in.views[i]; }
bool hasPly(const SaveOut& out, int i) { return i < int(out.ply_ok.size()) && out.ply_ok[i]; }

// 점 구름 → binary_little_endian PLY(머리 다음 점마다 float x,y,z(map) + uchar r,g,b = 15 바이트)
std::string plyBytes(const ObjCloud& c) {
  const size_t n = c.size();
  std::string o = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(n) +
                  "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                  "property uchar blue\nend_header\n";
  const size_t h = o.size();
  o.resize(h + n * 15);
  char* p = &o[h];
  for (size_t i = 0; i < n; ++i, p += 15) {
    const CloudPt& q = c.data->pts[i];
    const float xyz[3] = {float(c.org[0] + q.x), float(c.org[1] + q.y), float(c.org[2] + q.z)};
    std::memcpy(p, xyz, 12);   // x86 은 리틀 엔디언
    p[12] = char(q.r);
    p[13] = char(q.g);
    p[14] = char(q.b);
  }
  return o;
}

// ---- 방(rooms.hpp) ----
uint32_t objRoom(const SaveInput& in, int i) { return i < int(in.room_names.obj_room.size()) ? in.room_names.obj_room[i] : 0; }
const RoomLabel* roomLabel(const SaveInput& in, size_t k) { return k < in.room_names.rooms.size() ? &in.room_names.rooms[k] : nullptr; }
// 뷰어 색(방 id 마다 고정: 황금비 색상환, 채도 0.55·명도 0.95)
void roomColor(uint32_t id, int rgb[3]) {
  const double h = std::fmod(id * 0.6180339887, 1.0) * 6, s = 0.55, v = 0.95;
  const int i = int(h);
  const double f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  const double r[6] = {v, q, p, p, t, v}, g[6] = {t, v, v, q, p, p}, b[6] = {p, p, t, v, v, q};
  rgb[0] = int(r[i % 6] * 255);
  rgb[1] = int(g[i % 6] * 255);
  rgb[2] = int(b[i % 6] * 255);
}

// view.json 끝에 붙는 방 키: rooms(value = rooms.pgm 값), room_doors, rooms_grid
std::string roomsJson(const SaveInput& in) {
  if (!in.rooms) return "";
  const RoomSeg& R = *in.rooms;
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(3);
  o << ",\"rooms_grid\":\"rooms.pgm\",\"rooms\":[";
  for (size_t k = 0; k < R.rooms.size(); ++k) {
    const RoomGeom& r = R.rooms[k];
    const RoomLabel* L = roomLabel(in, k);
    int c[3];
    roomColor(r.id, c);
    o << (k ? "," : "") << "{\"id\":" << r.id << ",\"value\":" << std::min<size_t>(k + 1, 255) << ",\"name\":\"" << esc(L ? L->name : "")
      << "\",\"type\":\"" << esc(L ? L->type : "") << "\",\"conf\":" << (L ? L->conf : 0.f) << ",\"centroid\":[" << r.centroid[0] << ","
      << r.centroid[1] << "],\"area\":" << r.area_m2 << ",\"bbox\":[" << r.bmin[0] << "," << r.bmin[1] << "," << r.bmax[0] << "," << r.bmax[1]
      << "],\"color\":[" << c[0] << "," << c[1] << "," << c[2] << "],\"objects\":[";
    if (L)
      for (size_t j = 0; j < L->objects.size(); ++j) o << (j ? "," : "") << L->objects[j];
    o << "]}";
  }
  o << "],\"room_doors\":[";
  for (size_t k = 0; k < R.doors.size(); ++k) {
    const RoomDoor& d = R.doors[k];
    o << (k ? "," : "") << "{\"a\":" << d.a << ",\"b\":" << d.b << ",\"pos\":[" << d.pos[0] << "," << d.pos[1] << "],\"width\":" << d.width << "}";
  }
  o << "]";
  return o.str();
}

// map.pgm 과 같은 크기·방향(지금 격자 칸마다 나눔 칸을 찾음 — 나눔이 조금 옛 격자여도)
std::string roomsPgm(const SaveInput& in) {
  const RoomSeg& R = *in.rooms;
  std::string o = "P5\n" + std::to_string(in.grid_w) + " " + std::to_string(in.grid_h) + "\n255\n";
  const size_t h0 = o.size();
  o.resize(h0 + size_t(in.grid_w) * in.grid_h, char(0));
  std::unordered_map<uint32_t, int> val;
  for (size_t k = 0; k < R.rooms.size(); ++k) val[R.rooms[k].id] = int(std::min<size_t>(k + 1, 255));
  const int gx0 = int(std::lround(in.grid_ox / in.grid_res)), gy0 = int(std::lround(in.grid_oy / in.grid_res));
  const int dx = gx0 - R.gx0, dy = gy0 - R.gy0;
  for (int y = 0; y < in.grid_h; ++y) {
    const int ry = y + dy;
    if (ry < 0 || ry >= R.h) continue;
    char* row = &o[h0 + size_t(in.grid_h - 1 - y) * in.grid_w];
    for (int x = 0; x < in.grid_w; ++x) {
      const int rx = x + dx;
      if (rx < 0 || rx >= R.w) continue;
      const uint32_t id = R.ids[size_t(ry) * R.w + rx];
      if (id) row[x] = char(val[id]);
    }
  }
  return o;
}

std::string graphJson(const SaveInput& in);

std::string viewJson(const SaveInput& in, const SaveOut& ok) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(3);
  o << "{\"stamp\":" << in.stamp << ",\"pose\":[" << in.pose[0] << "," << in.pose[1] << "," << in.pose[2] << "],";
  o << "\"grid\":{\"resolution\":" << in.grid_res << ",\"origin\":[" << in.grid_ox << "," << in.grid_oy << "],\"width\":" << in.grid_w
    << ",\"height\":" << in.grid_h << "},\"objects\":[";
  for (int i = 0; i < in.n_objs; ++i) {
    const sm_object& b = in.objs[i];
    o << (i ? "," : "") << "{\"id\":" << b.id << ",\"name\":\"" << esc(b.name ? b.name : "") << "\",\"state\":\"" << stateName(b.state)
      << "\",\"pos\":[" << b.pos[0] << "," << b.pos[1] << "," << b.pos[2] << "],\"extent\":[" << b.extent[0] << "," << b.extent[1] << ","
      << b.extent[2] << "],\"first_pos\":[" << b.first_pos[0] << "," << b.first_pos[1] << "," << b.first_pos[2] << "],\"n_obs\":" << b.n_obs
      << ",\"last_seen\":" << b.last_seen << ",\"score\":" << b.score << ",\"structural\":" << (b.structural ? "true" : "false")
      << ",\"movable\":" << (i >= int(in.movable.size()) || in.movable[i] ? "true" : "false");
    if (hasView(in, ok, i))
      o << ",\"rgbd\":{\"rgb\":\"" << objPath(b.id, "rgb") << "\",\"depth\":\"" << objPath(b.id, "depth") << "\",\"mask\":\""
        << objPath(b.id, "mask") << "\"}";
    if (hasPly(ok, i))
      o << ",\"points\":{\"path\":\"" << plyPath(b.id) << "\",\"n\":" << in.clouds[i].size() << ",\"voxel\":" << in.voxel
        << ",\"stamp\":" << in.clouds[i].stamp << "}";
    if (in.rooms) o << ",\"room\":" << objRoom(in, i);
    if (i < int(in.obj_meta.size()) && !in.obj_meta[i].empty()) o << "," << in.obj_meta[i];   // objprob 불확실성·벡터 경로 등
    o << "}";
  }
  o << "],\"events\":[";
  for (size_t i = 0; i < in.events.size(); ++i) {
    const ObjEvent& e = in.events[i];
    o << (i ? "," : "") << "{\"t\":" << e.t << ",\"id\":" << e.id << ",\"kind\":\"" << eventName(e.kind) << "\",\"pos\":[" << e.pos[0] << ","
      << e.pos[1] << "," << e.pos[2] << "]}";
  }
  o << "]" << roomsJson(in) << graphJson(in) << "}\n";
  return o.str();
}

// ROS map_server 형식(위가 +y — 격자 행을 뒤집어 쓴다)
std::string pgm(const SaveInput& in) {
  std::string o = "P5\n" + std::to_string(in.grid_w) + " " + std::to_string(in.grid_h) + "\n255\n";
  o.reserve(o.size() + size_t(in.grid_w) * in.grid_h);
  for (int y = in.grid_h - 1; y >= 0; --y)
    for (int x = 0; x < in.grid_w; ++x) {
      const int v = in.cells[size_t(y) * in.grid_w + x];
      o += char(v < 0 ? 205 : (v >= 65 ? 0 : (v <= 25 ? 254 : 254 - (v * 254) / 100)));
    }
  return o;
}

std::string yaml(const SaveInput& in) {
  std::ostringstream o;
  o << "image: map.pgm\nresolution: " << in.grid_res << "\norigin: [" << in.grid_ox << ", " << in.grid_oy
    << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n";
  return o.str();
}

#ifdef SM_HAVE_SPARK_DSG
// ROOMS 층: 방 노드('R', id) — 이름·위치(무게중심, z 0)·바닥 상자·종류 확률, metadata{area_m2, name_confidence, evidence, ...},
// 방→물체 변, 방–방 변(metadata relation door·pos·width)
void sceneRooms(const SaveInput& in, spark_dsg::DynamicSceneGraph& g) {
  using namespace spark_dsg;
  if (!in.rooms) return;
  const RoomSeg& R = *in.rooms;
  for (size_t k = 0; k < R.rooms.size(); ++k) {
    const RoomGeom& r = R.rooms[k];
    const RoomLabel* L = roomLabel(in, k);
    auto a = std::make_unique<RoomNodeAttributes>();
    a->position = Eigen::Vector3d(r.centroid[0], r.centroid[1], 0);
    a->name = L ? L->name : "room " + std::to_string(r.id);
    const Eigen::Vector3f dim(float(r.bmax[0] - r.bmin[0]), float(r.bmax[1] - r.bmin[1]), 0.f);
    const Eigen::Vector3f ctr(float(r.bmax[0] + r.bmin[0]) / 2, float(r.bmax[1] + r.bmin[1]) / 2, 0.f);
    a->bounding_box = BoundingBox(dim, ctr);
    a->last_update_time_ns = uint64_t(std::max(0.0, R.stamp) * 1e9);
    nlohmann::json ev = nlohmann::json::array();
    if (L) {
      a->semantic_class_probabilities = L->probs;
      for (const RoomEvidence& e : L->evidence) ev.push_back({{"object", e.obj}, {"name", e.name}, {"type", e.type}, {"weight", e.w}});
    }
    a->metadata.add({{"area_m2", r.area_m2},
                     {"name_confidence", L ? L->conf : 0.f},
                     {"evidence", ev},
                     {"type", L ? L->type : ""},
                     {"external_name", L && L->external},
                     {"max_clear_m", r.max_clear},
                     {"grid_value", std::min<size_t>(k + 1, 255)}});
    g.emplaceNode(DsgLayers::ROOMS, NodeSymbol('R', r.id), std::move(a));
  }
  for (int i = 0; i < in.n_objs; ++i)
    if (const uint32_t rid = objRoom(in, i)) g.insertEdge(NodeSymbol('R', rid), NodeSymbol('O', in.objs[i].id));
  for (const RoomDoor& d : R.doors) {
    auto e = std::make_unique<EdgeAttributes>(d.width);
    e->metadata.add({{"relation", "door"}, {"pos", {d.pos[0], d.pos[1]}}, {"width", d.width}});
    g.insertEdge(NodeSymbol('R', d.a), NodeSymbol('R', d.b), std::move(e));
  }
}

bool sceneDsg(const SaveInput& in, const SaveOut& ok, const fs::path& path) {
  using namespace spark_dsg;
  DynamicSceneGraph g;
  for (int i = 0; i < in.n_objs; ++i) {
    const sm_object& b = in.objs[i];
    auto a = std::make_unique<ObjectNodeAttributes>();
    a->position = Eigen::Vector3d(b.pos[0], b.pos[1], b.pos[2]);
    a->name = b.name ? b.name : "";
    a->bounding_box = BoundingBox(Eigen::Vector3f(float(b.extent[0]), float(b.extent[1]), float(b.extent[2])), a->position.cast<float>());
    a->last_update_time_ns = uint64_t(std::max(0.0, b.last_seen) * 1e9);
    a->is_active = b.state != SM_GONE;
    a->metadata.add({{"state", stateName(b.state)},
                     {"n_obs", b.n_obs},
                     {"score", b.score},
                     {"first_pos", {b.first_pos[0], b.first_pos[1], b.first_pos[2]}},
                     {"structural", bool(b.structural)},
                     {"handled", bool(b.handled)},
                     {"movable", i >= int(in.movable.size()) || in.movable[i] != 0}});
    if (hasView(in, ok, i)) {
      const BestView& v = *in.views[i];
      std::vector<double> T(v.cam_T, v.cam_T + 12);
      a->metadata.add({{"rgbd",
                        {{"rgb", objPath(b.id, "rgb")},
                         {"depth", objPath(b.id, "depth")},
                         {"mask", objPath(b.id, "mask")},
                         {"stamp", v.stamp},
                         {"box_px", {v.box[0], v.box[1], v.box[2], v.box[3]}},
                         {"det_box_px", {v.det_box[0], v.det_box[1], v.det_box[2], v.det_box[3]}},
                         {"mask_area", v.mask_area},
                         {"depth_m", v.depth_m},
                         {"score", v.score},
                         {"cam_T", T}}}});
    }
    if (hasPly(ok, i))
      a->metadata.add({{"points",
                        {{"path", plyPath(b.id)}, {"n", in.clouds[i].size()}, {"voxel", in.voxel}, {"stamp", in.clouds[i].stamp}}}});
    g.emplaceNode(DsgLayers::OBJECTS, NodeSymbol('O', b.id), std::move(a));
  }
  sceneRooms(in, g);
  g.metadata.add({{"stamp", in.stamp}, {"robot_pose", {in.pose[0], in.pose[1], in.pose[2]}}, {"grid", "map.pgm"}});
  const fs::path tmp = path.string() + ".tmp.json";
  g.save(tmp);
  std::error_code ec;
  fs::rename(tmp, path, ec);
  return !ec;
}
#endif


// ---- scene.json 빠른 쓰기(Spark-DSG 1.1.3 JSON 형식 그대로 — 기본 spark_dsg 가 읽음, 라이브러리 없이 문자열로) ----
struct J {
  std::string& o;
  void raw(const char* s) { o += s; }
  void raw(const std::string& s) { o += s; }
  void num(double v) {
    if (!std::isfinite(v)) { o += "0.0"; return; }
    char b[32];
    auto r = std::to_chars(b, b + sizeof b, v);
    o.append(b, r.ptr);
    // 정수처럼 보이면 ".0"(spark_dsg 는 float 칸을 정수로 읽어도 되지만 보기 좋게)
    bool dot = false;
    for (char* p = b; p < r.ptr; ++p) dot |= (*p == '.' || *p == 'e' || *p == 'n' || *p == 'i');
    if (!dot) o += ".0";
  }
  void inum(int64_t v) {
    char b[24];
    auto r = std::to_chars(b, b + sizeof b, v);
    o.append(b, r.ptr);
  }
  void unum(uint64_t v) {
    char b[24];
    auto r = std::to_chars(b, b + sizeof b, v);
    o.append(b, r.ptr);
  }
  void str(const std::string& s) { o += '"'; o += esc(s); o += '"'; }
  void vec3(const double* v) { o += '['; num(v[0]); o += ','; num(v[1]); o += ','; num(v[2]); o += ']'; }
  void key(const char* k) { o += '"'; o += k; o += "\":"; }
};

const char* kHeader = "{\"SPARK_DSG_header\":{\"project_name\":\"main\",\"version\":{\"major\":1,\"minor\":1,\"patch\":3}},\"directed\":false,";
const char* kColor = "\"color\":{\"a\":255,\"b\":0,\"g\":0,\"r\":0},";
const char* kNoFeat = "\"semantic_feature\":{\"cols\":0,\"data\":null,\"rows\":0},\"semantic_label\":4294967295,";
const char* kIdentQ = "{\"w\":1.0,\"x\":0.0,\"y\":0.0,\"z\":0.0}";

void bbox(J& j, const double lo[3], const double hi[3], bool valid) {
  const double c[3] = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
  const double d[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
  j.raw("\"bounding_box\":{\"dimensions\":");
  j.vec3(d);
  j.raw(valid ? ",\"type\":\"AABB\",\"world_P_center\":" : ",\"type\":\"INVALID\",\"world_P_center\":");
  j.vec3(c);
  j.raw(",\"world_R_center\":");
  j.raw(kIdentQ);
  j.raw("},");
}

void nodeTail(J& j, uint64_t id, int layer, int partition) {
  j.raw("},\"id\":");
  j.unum(id);
  j.raw(",\"layer\":");
  j.inum(layer);
  j.raw(",\"partition\":");
  j.inum(partition);
  j.raw("}");
}

void objectNode(J& j, const SaveInput& in, const SaveOut& ok, int i) {
  const sm_object& b = in.objs[i];
  double lo[3], hi[3];
  for (int k = 0; k < 3; ++k) { lo[k] = b.pos[k] - b.extent[k] / 2; hi[k] = b.pos[k] + b.extent[k] / 2; }
  j.raw("{\"attributes\":{");
  bbox(j, lo, hi, true);
  j.raw(kColor);
  j.raw(b.state != SM_GONE ? "\"is_active\":true," : "\"is_active\":false,");
  j.raw("\"is_predicted\":false,\"last_update_time_ns\":");
  j.unum(uint64_t(std::max(0.0, b.last_seen) * 1e9));
  j.raw(",\"metadata\":{\"first_pos\":");
  j.vec3(b.first_pos);
  j.raw(",\"handled\":");
  j.raw(b.handled ? "true" : "false");
  j.raw(",\"movable\":");
  j.raw(i >= int(in.movable.size()) || in.movable[i] != 0 ? "true" : "false");
  j.raw(",\"n_obs\":");
  j.unum(b.n_obs);
  if (hasPly(ok, i)) {
    j.raw(",\"points\":{\"n\":");
    j.unum(in.clouds[i].size());
    j.raw(",\"path\":");
    j.str(plyPath(b.id));
    j.raw(",\"stamp\":");
    j.num(in.clouds[i].stamp);
    j.raw(",\"voxel\":");
    j.num(in.voxel);
    j.raw("}");
  }
  if (hasView(in, ok, i)) {
    const BestView& v = *in.views[i];
    j.raw(",\"rgbd\":{\"box_px\":[");
    for (int k = 0; k < 4; ++k) { if (k) j.raw(","); j.inum(v.box[k]); }
    j.raw("],\"cam_T\":[");
    for (int k = 0; k < 12; ++k) { if (k) j.raw(","); j.num(v.cam_T[k]); }
    j.raw("],\"depth\":");
    j.str(objPath(b.id, "depth"));
    j.raw(",\"depth_m\":");
    j.num(v.depth_m);
    j.raw(",\"det_box_px\":[");
    for (int k = 0; k < 4; ++k) { if (k) j.raw(","); j.inum(v.det_box[k]); }
    j.raw("],\"mask\":");
    j.str(objPath(b.id, "mask"));
    j.raw(",\"mask_area\":");
    j.num(v.mask_area);
    j.raw(",\"rgb\":");
    j.str(objPath(b.id, "rgb"));
    j.raw(",\"score\":");
    j.num(v.score);
    j.raw(",\"stamp\":");
    j.num(v.stamp);
    j.raw("}");
  }
  j.raw(",\"score\":");
  j.num(b.score);
  j.raw(",\"state\":");
  j.str(stateName(b.state));
  j.raw(",\"structural\":");
  j.raw(b.structural ? "true" : "false");
  if (i < int(in.obj_meta.size()) && !in.obj_meta[i].empty()) {
    j.raw(",");
    j.raw(in.obj_meta[i]);
  }
  j.raw("},\"name\":");
  j.str(b.name ? b.name : "");
  j.raw(",\"position\":");
  j.vec3(b.pos);
  j.raw(",\"registered\":false,");
  j.raw(kNoFeat);
  j.raw("\"type\":\"ObjectNodeAttributes\",\"world_R_object\":");
  j.raw(kIdentQ);
  nodeTail(j, nodeSym('O', b.id), 2, 0);
}

void roomNode(J& j, const SaveInput& in, size_t k) {
  const RoomGeom& r = in.rooms->rooms[k];
  const RoomLabel* L = roomLabel(in, k);
  const double lo[3] = {r.bmin[0], r.bmin[1], 0}, hi[3] = {r.bmax[0], r.bmax[1], 0};
  j.raw("{\"attributes\":{");
  bbox(j, lo, hi, true);
  j.raw(kColor);
  j.raw("\"is_active\":false,\"is_predicted\":false,\"last_update_time_ns\":");
  j.unum(uint64_t(std::max(0.0, in.rooms->stamp) * 1e9));
  j.raw(",\"metadata\":{\"area_m2\":");
  j.num(r.area_m2);
  j.raw(",\"evidence\":[");
  if (L)
    for (size_t e = 0; e < L->evidence.size(); ++e) {
      const RoomEvidence& v = L->evidence[e];
      j.raw(e ? ",{\"name\":" : "{\"name\":");
      j.str(v.name);
      j.raw(",\"object\":");
      j.unum(v.obj);
      j.raw(",\"type\":");
      j.str(v.type);
      j.raw(",\"weight\":");
      j.num(v.w);
      j.raw("}");
    }
  j.raw("],\"external_name\":");
  j.raw(L && L->external ? "true" : "false");
  j.raw(",\"grid_value\":");
  j.unum(std::min<size_t>(k + 1, 255));
  j.raw(",\"max_clear_m\":");
  j.num(r.max_clear);
  j.raw(",\"name_confidence\":");
  j.num(L ? L->conf : 0.0);
  j.raw(",\"type\":");
  j.str(L ? L->type : "");
  j.raw("},\"name\":");
  j.str(L ? L->name : "room " + std::to_string(r.id));
  const double c[3] = {r.centroid[0], r.centroid[1], 0};
  j.raw(",\"position\":");
  j.vec3(c);
  j.raw(",\"semantic_class_probabilities\":{");
  if (L && !L->probs.empty()) {
    bool first = true;
    for (const auto& [n, pr] : L->probs) {
      if (!first) j.raw(",");
      first = false;
      j.str(n);
      j.raw(":");
      j.num(pr);
    }
  }
  j.raw("},");
  j.raw(kNoFeat);
  j.raw("\"type\":\"RoomNodeAttributes\"");
  nodeTail(j, nodeSym('R', r.id), 4, 0);
}

void graphNode(J& j, const GNode& n) {
  j.raw("{\"attributes\":{");
  if (n.partition == 'a') {   // AgentNodeAttributes
    j.raw("\"dbow_ids\":null,\"dbow_values\":null,\"external_key\":");
    j.unum(n.id);
    j.raw(",\"is_active\":false,\"is_predicted\":false,\"last_update_time_ns\":");
    j.unum(uint64_t(std::max(0.0, n.stamp) * 1e9));
    j.raw(",\"metadata\":{\"yaw\":");
    j.num(n.yaw);
    j.raw("},\"position\":");
    const double p[3] = {n.pos[0], n.pos[1], 0};
    j.vec3(p);
    j.raw(",\"timestamp\":");
    j.unum(uint64_t(std::max(0.0, n.stamp) * 1e9));
    j.raw(",\"type\":\"AgentNodeAttributes\",\"world_R_body\":{\"w\":");
    j.num(std::cos(n.yaw / 2));
    j.raw(",\"x\":0.0,\"y\":0.0,\"z\":");
    j.num(std::sin(n.yaw / 2));
    j.raw("}");
    nodeTail(j, n.id, 2, 'a');
    return;
  }
  if (n.layer == 3) {   // PlaceNodeAttributes(2D: z 0, distance = 여유)
    // Map_Vla: trimmed place schema — only the semantic base + distance (the frontier flag and the GVD/mesh fields are gone)
    const double z[3] = {0, 0, 0};
    bbox(j, z, z, false);
    j.raw(kColor);
    j.raw("\"distance\":");
    j.num(n.clearance);
    j.raw(",\"is_active\":true,\"is_predicted\":false,\"last_update_time_ns\":");
    j.unum(uint64_t(std::max(0.0, n.stamp) * 1e9));
    j.raw(",\"metadata\":{},\"name\":\"\",\"position\":");
    const double p[3] = {n.pos[0], n.pos[1], 0};
    j.vec3(p);
    j.raw(",");
    j.raw(kNoFeat);
    j.raw("\"type\":\"PlaceNodeAttributes\"");
    nodeTail(j, n.id, 3, 0);
  }
}

const char* relName(int r) {
  static const char* n[] = {"parent", "place", "door", "?", "?", "?", "agent"};
  return r >= 0 && r < 7 ? n[r] : "?";
}

bool sceneJsonFast(const SaveInput& in, const SaveOut& ok, const fs::path& path, SaveOut* out) {
  const auto t0 = std::chrono::steady_clock::now();
  std::string o;
  o.reserve(64 * 1024);
  J j{o};
  j.raw(kHeader);
  // 변
  j.raw("\"edges\":[");
  bool first = true;
  auto edge = [&](uint64_t a, uint64_t b, double w, bool weighted, const std::string& meta) {
    if (!first) j.raw(",");
    first = false;
    j.raw("{\"info\":{\"metadata\":{");
    j.raw(meta);
    j.raw("},\"type\":\"EdgeAttributes\",\"weight\":");
    j.num(w);
    j.raw(weighted ? ",\"weighted\":true},\"source\":" : ",\"weighted\":false},\"source\":");
    j.unum(a);
    j.raw(",\"target\":");
    j.unum(b);
    j.raw("}");
  };
  // 저장할 노드 = 물체(in.objs) + 방(in.rooms) + 그래프의 agent·place·건물. 변은 양 끝이 저장되는 것만
  std::unordered_map<uint64_t, char> have;
  for (int i = 0; i < in.n_objs; ++i) have[nodeSym('O', in.objs[i].id)] = 1;
  if (in.rooms)
    for (const RoomGeom& r : in.rooms->rooms) have[nodeSym('R', r.id)] = 1;
  if (in.graph)
    for (const GNode& n : in.graph->nodes)
      if (n.partition == 'a' || n.layer == 3) have[n.id] = 1;
  if (in.graph) {
    std::unordered_map<uint64_t, char> pairs;   // Spark-DSG 는 한 쌍에 변 하나
    for (const GEdge& e : in.graph->edges) {
      if (!have.count(e.a) || !have.count(e.b)) continue;
      const uint64_t lo = std::min(e.a, e.b), hi = std::max(e.a, e.b);
      if (!pairs.emplace(lo * 1000003ull ^ hi, 1).second) continue;
      std::string meta;
      if (e.rel != kRelGeneric) {
        meta = "\"relation\":\"" + std::string(relName(e.rel)) + "\"";
        if (e.rel == kRelDoor) {
          meta += ",\"pos\":[";
          J m{meta};
          m.num(e.pos[0]);
          meta += ",";
          m.num(e.pos[1]);
          meta += "],\"width\":";
          m.num(e.weight);
        } else if (e.rel == kRelPlace) {
          meta += ",\"min_clear_m\":";
          J m{meta};
          m.num(e.weight);
        }
      }
      edge(e.a, e.b, e.rel == kRelGeneric || e.rel == kRelAgent ? 1.0 : e.weight,
           e.rel == kRelPlace || e.rel == kRelDoor, meta);
    }
  } else {
    for (int i = 0; i < in.n_objs; ++i)
      if (const uint32_t rid = objRoom(in, i)) edge(nodeSym('R', rid), nodeSym('O', in.objs[i].id), 1.0, false, "");
    if (in.rooms)
      for (const RoomDoor& d : in.rooms->doors) {
        std::string meta = "\"pos\":[";
        J m{meta};
        m.num(d.pos[0]); meta += ","; m.num(d.pos[1]); meta += "],\"relation\":\"door\",\"width\":"; m.num(d.width);
        edge(nodeSym('R', d.a), nodeSym('R', d.b), d.width, true, meta);
      }
  }
  // 층
  bool has_agents = false;
  if (in.graph)
    for (const GNode& n : in.graph->nodes) {
      has_agents |= n.partition == 'a';
    }
  j.raw("],\"layer_keys\":[{\"layer\":2,\"partition\":0},");
  if (has_agents) j.raw("{\"layer\":2,\"partition\":97},");
  j.raw("{\"layer\":3,\"partition\":0},{\"layer\":4,\"partition\":0}],");
  j.raw("\"layer_names\":{\"AGENTS\":{\"layer\":2,\"partition\":97},"   /* agents live in partition 'a' (97); the original wrote 0 = the OBJECTS layer */
        "\"OBJECTS\":{\"layer\":2,\"partition\":0},\"PLACES\":{\"layer\":3,\"partition\":0},\"ROOMS\":{\"layer\":4,\"partition\":0}},");
  j.raw("\"metadata\":{\"grid\":\"map.pgm\",\"robot_pose\":");
  j.vec3(in.pose);
  j.raw(",\"stamp\":");
  j.num(in.stamp);
  j.raw("},\"multigraph\":false,\"nodes\":[");
  first = true;
  auto sep = [&]() { if (!first) j.raw(","); first = false; };
  for (int i = 0; i < in.n_objs; ++i) { sep(); objectNode(j, in, ok, i); }
  if (in.rooms)
    for (size_t k = 0; k < in.rooms->rooms.size(); ++k) { sep(); roomNode(j, in, k); }
  if (in.graph)
    for (const GNode& n : in.graph->nodes) {
      if (!(n.partition == 'a' || n.layer == 3)) continue;
      sep();
      if (in.json_cache) {   // 바뀌지 않은 노드는 지난 조각 그대로
        auto& slot = in.json_cache->nodes[n.id];
        if (slot.second.empty() || slot.first != n.ver) {
          slot.second.clear();
          J c{slot.second};
          graphNode(c, n);
          slot.first = n.ver;
        }
        o += slot.second;
      } else {
        graphNode(j, n);
      }
    }
  j.raw("]}\n");
  if (in.json_cache && in.graph && in.json_cache->nodes.size() > 2 * in.graph->nodes.size() + 64) {   // 지워진 노드 정리
    for (auto it = in.json_cache->nodes.begin(); it != in.json_cache->nodes.end();)
      it = in.graph->find(it->first) ? std::next(it) : in.json_cache->nodes.erase(it);
  }
  const bool good = writeAtomic(path, o);
  if (out) {
    out->json_bytes = o.size();
    out->json_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return good;
}

// view.json "graph": 뷰어용 짧은 꼴(agent·place·건물 노드, 모든 변)
std::string graphJson(const SaveInput& in) {
  if (!in.graph) return "";
  std::string o = ",\"graph\":{\"nodes\":[";
  J j{o};
  bool first = true;
  for (const GNode& n : in.graph->nodes) {
    const char* kind = n.partition == 'a' ? "agent" : n.layer == 3 ? "place" : nullptr;
    if (!kind || (in.stream_lite && n.layer == 3)) continue;
    if (!first) o += ",";
    first = false;
    o += "{\"id\":\"";
    o += char(symChar(n.id));
    j.unum(symIdx(n.id));
    o += "\",\"kind\":\"";
    o += kind;
    o += "\",\"pos\":[";
    j.num(n.pos[0]); o += ","; j.num(n.pos[1]);
    o += "]";
    if (n.layer == 3) { o += ",\"clear\":"; j.num(n.clearance); }
    if (n.partition == 'a') { o += ",\"yaw\":"; j.num(n.yaw); o += ",\"t\":"; j.num(n.stamp); }
    o += "}";
  }
  o += "],\"edges\":[";
  first = true;
  for (const GEdge& e : in.graph->edges) {
    if (in.stream_lite && (symChar(e.a) == 'p' || symChar(e.b) == 'p')) continue;
    if (!first) o += ",";
    first = false;
    o += "[\"";
    o += char(symChar(e.a));
    j.unum(symIdx(e.a));
    o += "\",\"";
    o += char(symChar(e.b));
    j.unum(symIdx(e.b));
    o += "\",\"";
    o += relName(e.rel);
    o += "\",";
    j.num(e.weight);
    o += "]";
  }
  o += "]}";
  return o;
}

bool hasCloud(const SaveInput& in, int i) { return i < int(in.clouds.size()) && in.clouds[i].size() > 0; }

// objects/ 의 best view PNG·구름 PLY: 바뀐 것·없는 것만 쓰고, png_ok·ply_ok[i] = 파일 있음
bool savePngs(const SaveInput& in, const fs::path& od, SaveOut& out) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<uint8_t>& ok = out.png_ok;
  ok.assign(in.n_objs, 0);
  out.ply_ok.assign(in.n_objs, 0);
  std::error_code ec;
  bool any = false;
  for (int i = 0; i < in.n_objs; ++i) any = any || (i < int(in.views.size()) && in.views[i]) || hasCloud(in, i);
  if (!any && !in.clean_objects) return true;
  fs::create_directories(od, ec);
  if (in.clean_objects) {   // 지금 물체가 아닌 O<id>_{rgb,depth,mask}.png · O<id>_points.ply 지우기
    std::vector<std::string> keep;
    for (int i = 0; i < in.n_objs; ++i) {
      const std::string b = "O" + std::to_string(in.objs[i].id);
      if (i < int(in.views.size()) && in.views[i])
        for (const char* w : {"_rgb.png", "_depth.png", "_mask.png"}) keep.push_back(b + w);
      if (hasCloud(in, i)) keep.push_back(b + "_points.ply");
    }
    auto endsWith = [](const std::string& n, const char* e) {
      const size_t m = std::strlen(e);
      return n.size() > m && n.compare(n.size() - m, m, e) == 0;
    };
    for (const auto& e : fs::directory_iterator(od, ec)) {
      const std::string n = e.path().filename().string();
      const bool ours = n.size() > 1 && n[0] == 'O' && std::isdigit(static_cast<unsigned char>(n[1])) &&
                        (endsWith(n, "_rgb.png") || endsWith(n, "_depth.png") || endsWith(n, "_mask.png") || endsWith(n, "_points.ply"));
      if (ours && std::find(keep.begin(), keep.end(), n) == keep.end()) fs::remove(e.path(), ec);
    }
  }
  bool good = true;
  for (int i = 0; i < in.n_objs && i < int(in.views.size()); ++i) {
    const BestView* v = in.views[i].get();
    if (!v || v->rgb.empty() || v->depth.empty()) continue;
    const fs::path pr = od.parent_path() / objPath(in.objs[i].id, "rgb"), pd = od.parent_path() / objPath(in.objs[i].id, "depth");
    const bool dirty = i < int(in.png_dirty.size()) && in.png_dirty[i];
    bool ok_i = true;
    if (dirty || !fs::exists(pr, ec)) {
      const std::string s = pngRgb8(v->rgb.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pr, s);
      out.n_png += ok_i;
    }
    if (ok_i && (dirty || !fs::exists(pd, ec))) {
      const std::string s = pngGray16(v->depth.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pd, s);
      out.n_png += ok_i;
    }
    const fs::path pm = od.parent_path() / objPath(in.objs[i].id, "mask");
    if (ok_i && !v->mask.empty() && (dirty || !fs::exists(pm, ec))) {
      const std::string s = pngGray8(v->mask.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pm, s);
      out.n_png += ok_i;
    }
    ok[i] = ok_i;
    good = good && ok_i;
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.png_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (int i = 0; i < in.n_objs; ++i) {
    if (!hasCloud(in, i)) continue;
    const fs::path pp = od.parent_path() / plyPath(in.objs[i].id);
    const bool dirty = i < int(in.ply_dirty.size()) && in.ply_dirty[i];
    bool ok_i = true;
    if (dirty || !fs::exists(pp, ec)) {
      ok_i = writeAtomic(pp, plyBytes(in.clouds[i]));
      out.n_ply += ok_i;
    }
    out.ply_ok[i] = ok_i;
    good = good && ok_i;
  }
  out.ply_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
  return good;
}

}  // namespace

int saveScene(const SaveInput& in, const std::string& dir, SaveOut* out_) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path d(dir);
  SaveOut local;
  SaveOut& out = out_ ? *out_ : local;
  out = SaveOut{};
  bool ok = savePngs(in, d / "objects", out);
  if (in.grid_w > 0 && in.cells) {
    ok &= writeAtomic(d / "map.pgm", pgm(in));
    ok &= writeAtomic(d / "map.yaml", yaml(in));
    if (in.rooms) ok &= writeAtomic(d / "rooms.pgm", roomsPgm(in));
  }
  // scene.json: 기본은 빠른 직접 쓰기(같은 Spark-DSG JSON 형식). SM_DSG_SAVE=spark 면 Spark-DSG 라이브러리로(비교용)
  static const bool use_lib = [] { const char* e = std::getenv("SM_DSG_SAVE"); return e && std::string(e) == "spark"; }();
#ifdef SM_HAVE_SPARK_DSG
  if (use_lib) {
    const auto tj = std::chrono::steady_clock::now();
    ok &= sceneDsg(in, out, d / "scene.json");
    out.json_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tj).count();
  } else
#endif
  ok &= sceneJsonFast(in, out, d / "scene.json", &out);
  (void)use_lib;
  ok &= writeAtomic(d / "view.json", viewJson(in, out));   // 마지막에: 뷰어는 view.json 이 바뀌면 다시 읽는다
  return ok ? 0 : -1;
}

std::string sceneViewJson(const SaveInput& in, const SaveOut& out) { return viewJson(in, out); }

}  // namespace scenemap
