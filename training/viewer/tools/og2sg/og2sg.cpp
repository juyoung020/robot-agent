// og2sg — OmniGibson LIMO 탐사 한 판(sgrt 기록 rec.bin + 실행 폴더) → 학습 뷰어 재생용 "sgview 판"(<이름>.sg/ 폴더).
//
// 기록(관절·바퀴 오도메트리·RGB-D·라이다)을 실제 파이프라인 libsgrt(ObjectSAM + SigLIP 2 + objprob + Cartographer, src/scene_graph)로
// 다시 돌리면서 그 안 scenemap 의 sgview 스트림(sm_stream_start: POSE · MAP_RECT · VIEW · JOINTS 프레임)을 켜고, 그 스트림을 이 프로세스가 소켓으로 받아
// 시뮬 시각을 붙여 파일에 적는다. 재생(trainview /stream)은 이 프레임을 시각대로 다시 보낸다 — 그리기는 sgview 의 index.html 그대로.
//
//   og2sg --rec r4live.bin --run <OG 실행 폴더> --out <…/replays/ep_000000_explore.sg> [--robot limo_omx] [--view-hz 10] [--cam-hz 2] [--frames N]
//
// 쓰는 것(<out>/):
//   stream.sgs    "SGS1" 다음 [f64 sim_t][u32 len][u8 type][payload] 이어짐 (type = scenemap stream.hpp: 1 POSE, 2 MAP_RECT, 3 VIEW, 4 JOINTS)
//   memory/       끝에 sm_save_dsg — scene.json · view.json · map.pgm · objects/O<id>_{rgb,mask,depth}.png · O<id>_points.ply (sgview /file/ 이 읽음)
//   cam/          몸통 카메라 RGB JPEG (cam_hz) + cam.json [{t, file}]
//   meta.json     판 줄(meta), 실행·과제·장면, map_from_world, GT 궤적(map 좌표, 시뮬 시각), GT 물체(map 좌표), 라벨 이름 짝(정답 실행의 view.json 과 위치로)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rasc.h"       // BEHAVIOR 장면 RASC v3 로더(training/RL/tools/b1kconv/cpp, 읽기만)
#include "rec_util.h"   // jpeg::encode, Obj, jnum, mkdirs (training/viewer/tools/record_replay)
#include "scenemap.h"
#include "ra_paths.h"
#include "sg_capture.h"   // Capture(스트림 받기)·sg_drain — record_replay 와 같이 씀
#include "sgrec.hpp"
#include "sgrt.h"

using namespace rec;

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

static std::string b64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  o.reserve((d.size() + 2) / 3 * 4);
  for (size_t i = 0; i < d.size(); i += 3) {
    const uint32_t v = (uint32_t(d[i]) << 16) | (i + 1 < d.size() ? uint32_t(d[i + 1]) << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(v >> 18) & 63]; o += T[(v >> 12) & 63];
    o += i + 1 < d.size() ? T[(v >> 6) & 63] : '=';
    o += i + 2 < d.size() ? T[v & 63] : '=';
  }
  return o;
}
static float qyaw(const float q[4]) { return std::atan2(2.f * (q[3] * q[2] + q[0] * q[1]), 1.f - 2.f * (q[1] * q[1] + q[2] * q[2])); }

// BEHAVIOR 집 배치(세계 좌표) — 재생 화면의 바탕 층: 방 칸(방 종류 색), 벽 칸(벽만 막힌 다닐 곳 층 → 세워 그림, 문 자리는 비어 있음),
// 가구·문·창 상자, 놓을 곳(받침, 장면 수준)과 집을 것(과제 인스턴스), 이 과제의 물체.
static std::string underlay_json(const std::string& path, const std::string& task_name, std::string* scene_name) {
  rasc::Scene S;
  std::string err;
  if (!rasc::load(S, path.c_str(), &err)) { std::fprintf(stderr, "rasc %s: %s\n", path.c_str(), err.c_str()); return "null"; }
  *scene_name = S.name();
  const RascFileHeader& H = *S.h;
  const uint32_t W = H.grid_w, Hh = H.grid_h;
  std::vector<uint8_t> rooms(size_t(W) * Hh), walls(size_t(W) * Hh, 0);
  for (size_t i = 0; i < rooms.size(); ++i) rooms[i] = S.room_grid[i];
  for (uint32_t r = 0; r < Hh; ++r)
    for (uint32_t c = 0; c < W; ++c) {
      if (S.free_cell(1, int(r), int(c))) continue;   // TRAV_NO_OBJ: 벽만 막힘
      bool near = false;
      for (int dr = -1; dr <= 1 && !near; ++dr)
        for (int dc = -1; dc <= 1 && !near; ++dc) {
          const int rr = int(r) + dr, cc = int(c) + dc;
          if (rr >= 0 && cc >= 0 && rr < int(Hh) && cc < int(W) && S.room_grid[size_t(rr) * W + cc]) near = true;
        }
      walls[size_t(r) * W + c] = near ? 1 : 0;
    }
  auto cat = [&](uint16_t k) { return k < S.cats.size() ? std::string(S.str(S.cats[k].name)) : std::string("?"); };
  std::string boxes = "[";
  int nb = 0;
  for (const RascBoxRec& b : S.boxes) {
    if (b.flags & (RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING)) continue;   // 벽은 칸으로 세워 그림
    const char* kind = (b.flags & RASC_F_DOOR) ? "door" : (b.flags & RASC_F_WINDOW) ? "window" : (b.flags & RASC_F_CARPET) ? "carpet" : "furniture";
    boxes += std::string(nb++ ? "," : "") + Obj().str("kind", kind).str("cat", cat(b.cat)).str("name", b.obj < S.objs.size() ? S.str(S.objs[b.obj].name) : "")
                                                  .raw("c", arr3(b.center)).raw("h", arr3(b.half)).num("yaw", b.yaw).done();
  }
  std::string roomsj = "[";
  for (size_t k = 0; k < S.rooms.size(); ++k) {
    const RascRoomRec& r = S.rooms[k];
    std::string nm = S.str(r.name), ty = nm.substr(0, nm.rfind('_'));
    roomsj += std::string(k ? "," : "") + Obj().num("id", double(k + 1)).str("name", nm).str("type", ty).raw("centroid", "[" + jnum(r.centroid[0]) + "," + jnum(r.centroid[1]) + "]")
                                             .raw("bmin", "[" + jnum(r.bmin[0]) + "," + jnum(r.bmin[1]) + "]").raw("bmax", "[" + jnum(r.bmax[0]) + "," + jnum(r.bmax[1]) + "]").done();
  }
  // 집기·놓기 표(v3): 장면 수준 = 마지막 PNP_RANGES. 과제 인스턴스 수준은 이 과제의 첫 인스턴스
  std::string places = "[", picks = "[", tobjs = "[";
  int npl = 0, npk = 0, nto = 0;
  auto add_ranges = [&](const RascPnpRange& R) {
    for (uint32_t k = 0; k < R.n_place; ++k) {
      const RascPlaceRec& p = S.places[R.place_off + k];
      if (p.kind == 3) continue;   // 방 바닥은 뺌
      const std::string ct = p.obj < S.objs.size() ? cat(S.objs[p.obj].cat) : "";
      places += std::string(npl++ ? "," : "") + Obj().num("kind", p.kind).str("cat", ct).raw("c", arr3(p.center)).raw("half", "[" + jnum(p.half[0]) + "," + jnum(p.half[1]) + "]")
                                                   .num("yaw", p.yaw).num("top", p.top).b("inner", p.flags & 1).done();
    }
    for (uint32_t k = 0; k < R.n_pick; ++k) {
      const RascPickRec& p = S.picks[R.pick_off + k];
      picks += std::string(npk++ ? "," : "") + Obj().raw("c", arr3(p.center)).num("z0", p.z0).num("top", p.top).num("w", p.min_w).num("yaw", p.yaw)
                                                  .b("inner", p.flags & 1).done();
    }
  };
  if (S.pnp_ranges.size()) add_ranges(S.pnp_ranges[S.pnp_ranges.size() - 1]);
  for (const RascTaskRec& T : S.tasks) {
    if (task_name.empty() || S.str(T.name) != task_name) continue;
    auto ins = S.insts_of(T);
    if (!ins.size()) break;
    const RascInstRec& I = ins[0];
    const size_t ii = size_t(&I - S.insts.p);
    if (ii < S.pnp_ranges.size() - 1) add_ranges(S.pnp_ranges[ii]);
    auto to = S.objs_of(T);
    auto po = S.poses_of(I);
    for (size_t k = 0; k < to.size() && k < po.size(); ++k) {
      const RascTaskObjRec& t = to[k];
      if (t.flags & (RASC_F_AGENT | RASC_F_SYSTEM) || !(t.half[0] > 0) || po[k].src == 0) continue;
      const float y = qyaw(po[k].quat), cy = std::cos(y), sy = std::sin(y);
      const float c3[3] = {po[k].pos[0] + cy * t.offset[0] - sy * t.offset[1], po[k].pos[1] + sy * t.offset[0] + cy * t.offset[1], po[k].pos[2] + t.offset[2]};
      tobjs += std::string(nto++ ? "," : "") + Obj().str("name", S.str(t.inst)).str("cat", t.cat < S.cats.size() ? cat(t.cat) : "?").raw("c", arr3(c3)).raw("h", arr3(t.half)).num("yaw", y).done();
    }
    break;
  }
  return Obj().str("scene", S.name()).str("file", path).raw("grid", Obj().num("w", W).num("h", Hh).num("res", H.cell).num("ox", H.origin[0]).num("oy", H.origin[1]).done())
      .str("room_grid_b64", b64(rooms)).str("wall_grid_b64", b64(walls)).raw("rooms", roomsj + "]").raw("boxes", boxes + "]").raw("places", places + "]").raw("picks", picks + "]")
      .raw("task_objects", tobjs + "]").str("task", task_name).done();
}

// 기록은 한 줄씩 읽는다(sgrec::Reader — 판마다 수 GB, scenemap/tools/sgrec.hpp 하나)
using sgrec::Rec;


// 배치만(--layout): 시뮬 기록 없이 BEHAVIOR 장면 하나를 sgview 판으로 — 다닐 곳 격자를 점유 지도로, 방을 방 노드로, 로봇은 과제 인스턴스 시작 자세.
// 좌표는 세계 그대로(map_from_world = 항등). 바탕 층(underlay.json)은 같은 집 배치.
static int layout_only(const std::string& rasc_path, const std::string& out, const std::string& run_out, long ep) {
  rasc::Scene S;
  std::string err;
  if (!rasc::load(S, rasc_path.c_str(), &err)) { std::fprintf(stderr, "rasc %s: %s\n", rasc_path.c_str(), err.c_str()); return 1; }
  const RascFileHeader& H = *S.h;
  mkdirs(out);
  FILE* f = std::fopen((out + "/stream.sgs").c_str(), "wb");
  std::fwrite("SGS1", 1, 4, f);
  auto frame = [&](uint8_t ty, const std::vector<uint8_t>& pl) { const double t = 0; const uint32_t n = uint32_t(pl.size()); std::fwrite(&t, 8, 1, f); std::fwrite(&n, 4, 1, f); std::fwrite(&ty, 1, 1, f); std::fwrite(pl.data(), 1, pl.size(), f); };
  auto put = [](std::vector<uint8_t>& v, const void* p, size_t n) { const uint8_t* q = (const uint8_t*)p; v.insert(v.end(), q, q + n); };
  double gox = 0, goy = 0;
  int gw = 0, gh = 0;   // 잘라 낸 격자(요약 view.grid 도 같게 — sgview 가 이것으로 화면을 맞춤)
  {   // MAP_RECT: 다님 0 %, 벽만 막힌 칸(방 둘레) 100 %, 나머지 모름
    // 집만(방 칸 + 1 m 둘레)으로 잘라 새 격자로 — 화면이 집에 맞춰지게
    std::vector<uint8_t> pl;
    const int GW = H.grid_w, GH = H.grid_h, M = int(std::lround(1.0 / H.cell));
    int r0 = GH, r1 = -1, c0 = GW, c1 = -1;
    for (int r = 0; r < GH; ++r) for (int c = 0; c < GW; ++c) if (S.room_grid[size_t(r) * GW + c]) { r0 = std::min(r0, r); r1 = std::max(r1, r); c0 = std::min(c0, c); c1 = std::max(c1, c); }
    if (r1 < 0) { r0 = 0; r1 = GH - 1; c0 = 0; c1 = GW - 1; }
    r0 = std::max(0, r0 - M); c0 = std::max(0, c0 - M); r1 = std::min(GH - 1, r1 + M); c1 = std::min(GW - 1, c1 + M);
    const int32_t w = c1 - c0 + 1, h = r1 - r0 + 1, x0 = 0, y0 = 0, x1 = w - 1, y1 = h - 1;
    const double res = H.cell, ox = H.origin[0] + c0 * H.cell, oy = H.origin[1] + r0 * H.cell;
    put(pl, &w, 4); put(pl, &h, 4); put(pl, &res, 8); put(pl, &ox, 8); put(pl, &oy, 8); put(pl, &x0, 4); put(pl, &y0, 4); put(pl, &x1, 4); put(pl, &y1, 4);
    gox = ox; goy = oy; gw = w; gh = h;
    for (int r = r0; r <= r1; ++r)
      for (int c = c0; c <= c1; ++c) {
        int8_t v = S.free_cell(1, r, c) ? 0 : -1;
        if (v < 0) for (int dr = -1; dr <= 1 && v < 0; ++dr) for (int dc = -1; dc <= 1 && v < 0; ++dc) { const int rr = r + dr, cc = c + dc; if (rr >= 0 && cc >= 0 && rr < GH && cc < GW && S.room_grid[size_t(rr) * GW + cc]) v = 100; }
        pl.push_back(uint8_t(v));
      }
    frame(2, pl);
  }
  const RascInstRec* I = S.insts.size() ? &S.insts[0] : nullptr;
  const double px = I ? I->robot_pos[0] : 0, py = I ? I->robot_pos[1] : 0, pyaw = I ? I->robot_yaw : 0;
  std::string rooms = "[";
  for (size_t k = 0; k < S.rooms.size(); ++k) {
    const RascRoomRec& r = S.rooms[k];
    std::string nm = S.str(r.name), ty = nm.substr(0, nm.rfind('_'));
    for (char& ch : ty) if (ch == '_') ch = ' ';
    const uint32_t hsh = uint32_t(std::hash<std::string>{}(ty));
    rooms += std::string(k ? "," : "") + Obj().num("id", double(k + 1)).num("value", double(k + 1)).str("name", nm).str("type", ty).num("conf", 1)
                                           .raw("centroid", "[" + jnum(r.centroid[0]) + "," + jnum(r.centroid[1]) + "]").num("area", r.area_m2)
                                           .raw("bbox", "[" + jnum(r.bmin[0]) + "," + jnum(r.bmin[1]) + "," + jnum(r.bmax[0]) + "," + jnum(r.bmax[1]) + "]")
                                           .raw("color", "[" + std::to_string(80 + hsh % 160) + "," + std::to_string(80 + (hsh / 160) % 160) + "," + std::to_string(120 + (hsh / 25600) % 130) + "]")
                                           .raw("objects", "[]").done();
  }
  const std::string view = Obj().num("stamp", 0).raw("pose", "[" + jnum(px) + "," + jnum(py) + "," + jnum(pyaw) + "]")
                               .raw("grid", Obj().num("resolution", H.cell).raw("origin", "[" + jnum(gox) + "," + jnum(goy) + "]").num("width", gw).num("height", gh).done())
                               .raw("objects", "[]").raw("events", "[]").raw("rooms", rooms + "]").raw("room_doors", "[]")
                               .raw("graph", "{\"nodes\":[{\"id\":\"a0\",\"kind\":\"agent\",\"pos\":[" + jnum(px) + "," + jnum(py) + "],\"yaw\":" + jnum(pyaw) + ",\"t\":0}],\"edges\":[]}").done();
  frame(3, std::vector<uint8_t>(view.begin(), view.end()));
  { std::vector<uint8_t> pl; const double t = 0; put(pl, &t, 8); put(pl, &px, 8); put(pl, &py, 8); put(pl, &pyaw, 8); frame(1, pl); }
  { std::vector<uint8_t> pl; const double t = 0; const int32_t n = 12; const float q[12] = {0, 0, 0, 0, 0, 0, 0, -1.6f, 1.45f, 0.15f, 0, 0}; put(pl, &t, 8); put(pl, &n, 4); put(pl, q, sizeof q); frame(4, pl); }
  std::fclose(f);
  std::string scene;
  const std::string task = I ? S.str(S.tasks[I->task].name) : "";
  std::ofstream(out + "/underlay.json") << underlay_json(rasc_path, task, &scene);
  const std::string meta = Obj().num("ep", ep).str("skill", "layout").str("home", scene).str("task", task).str("stage", "B1").str("outcome", "layout").str("driver", "none")
                               .num("t", 0).num("steps", 1).num("n_rooms", S.rooms.size()).raw("success", "null").done();
  std::ofstream(out + "/meta.json") << Obj().str("format", "SGS1").raw("meta", meta).str("robot", "limo_omx").raw("map_from_world", "[1,0,0,0]").raw("gt_path", "[]").raw("labels", "{}")
                                           .raw("cams", "[]").num("duration", 0).str("rasc", rasc_path)
                                           .raw("joint_order", "[\"\",\"\",\"\",\"\",\"\",\"\",\"omx_joint1\",\"omx_joint2\",\"omx_joint3\",\"omx_joint4\",\"omx_joint5\",\"omx_gripper_joint_1\"]").done();
  if (!run_out.empty()) {
    std::ofstream(run_out + "/episodes.jsonl", std::ios::app) << Obj().num("ep", ep).str("skill", "layout").str("home", scene).str("task", task).str("outcome", "layout")
                                                                    .str("replay", out.substr(out.rfind('/') + 1)).done() << "\n";
  }
  std::printf("og2sg --layout: %s -> %s (%zu rooms, grid %ux%u)\n", scene.c_str(), out.c_str(), S.rooms.size(), H.grid_w, H.grid_h);
  return 0;
}

int main(int argc, char** argv) {
  std::string recp, run, out, robot = "limo_omx", rasc_path, run_out, underlay_out, task_arg;
  bool layout = false;
  long ep_no = 0;
  double view_hz = 10, cam_hz = 2;
  long max_frames = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--rec") recp = nx(); else if (a == "--run") run = nx(); else if (a == "--out") out = nx(); else if (a == "--robot") robot = nx();
    else if (a == "--view-hz") view_hz = std::stod(nx()); else if (a == "--cam-hz") cam_hz = std::stod(nx()); else if (a == "--frames") max_frames = std::stol(nx());
    else if (a == "--rasc") rasc_path = nx(); else if (a == "--run-out") run_out = nx();
    else if (a == "--layout") layout = true; else if (a == "--ep") ep_no = std::stol(nx());
    else if (a == "--underlay") underlay_out = nx(); else if (a == "--task") task_arg = nx();
  }
  if (!underlay_out.empty()) {   // 바탕 층만(세계 좌표 BEHAVIOR 집 배치) — og_replay.py 가 OG 판 폴더에 씀
    std::string sc;
    const std::string u = underlay_json(rasc_path, task_arg, &sc);
    std::ofstream(underlay_out) << u;
    return u == "null" ? 1 : 0;
  }
  if (layout) {
    std::string sc = rasc_path.substr(rasc_path.rfind('/') + 1);
    sc = sc.substr(0, sc.find('.'));
    char nm[160];
    std::snprintf(nm, sizeof nm, "/replays/ep_%06ld_layout_%s.sg", ep_no, sc.c_str());
    if (out.empty()) out = run_out + nm;
    return layout_only(rasc_path, out, run_out, ep_no);
  }
  if (out.empty() && !run_out.empty()) out = run_out + "/replays/ep_000000_explore.sg";
  if (recp.empty() || run.empty() || out.empty()) { std::fprintf(stderr, "usage: og2sg --rec rec.bin --run OG_RUN_DIR --out EP.sg [--robot limo_omx]\n"); return 2; }
  mkdirs(out + "/cam");
  sgrec::Reader reader;
  if (!reader.open(recp.c_str())) { std::fprintf(stderr, "not an SGRC v1 record: %s\n", recp.c_str()); return 1; }

  Capture cap;
  if (!cap.start(out + "/stream.sgs")) { std::perror("capture"); return 1; }
  // 실제 파이프라인 그대로: libsgrt(엔진·자세 원천·objprob 기본값은 거기서). 이 프로세스가 스트림을 받으므로 sgrt 자신의 스트림·기록은 끔
  ::unsetenv("SGRT_STREAM");
  ::unsetenv("SGRT_RECORD");
  ::setenv("SGRT_ROBOT", robot.c_str(), 1);
  const std::string memdir = out + "/memory";
  sgrt_config scfg;
  sgrt_default_config(&scfg);
  scfg.out_dir = memdir.c_str();
  scfg.save_s = 1e9;   // 저장은 끝에 한 번
  const char* ee = std::getenv("SGRT_ENGINE");   // 분할 엔진: SGRT_ENGINE, 없으면 ObjectSAM(실제 파이프라인 기본)
  const std::string engine = ee && *ee ? ee : ra::models() + "/x86_sm120/yolo26n-seg-obj-416.plan", names = engine + ".names.txt";
  scfg.engine = engine.c_str();
  scfg.names = names.c_str();
  char err[512] = {0};
  sgrt* sg = sgrt_create(&scfg, err, sizeof err);
  if (!sg) { std::fprintf(stderr, "sgrt_create: %s\n", err); cap.stop(); return 1; }
  if (sgrt_begin(sg, nullptr, 0, err, sizeof err) != 0) { std::fprintf(stderr, "sgrt_begin: %s\n", err); return 1; }
  sm_ctx* c = sgrt_scenemap(sg);
  const std::string hp = "127.0.0.1:" + std::to_string(cap.port);
  if (sm_stream_start(c, hp.c_str())) { std::fprintf(stderr, "sm_stream_start failed\n"); return 1; }
  // 받는 쪽이 붙고 따라잡을 때까지 기다림(스트림은 링이 차면 프레임을 버린다 — 기록이므로 한 프레임도 버리지 않게 걸음을 맞춘다)
  // frames_sent 는 묶음 수라 프레임 수와 견줄 수 없다: 보낸 바이트가 두 번 연달아(송신 스레드 한 바퀴 ≥ 100 µs) 그대로이고 받은 바이트가 따라잡으면 비었다고 본다
  auto drain = [&]() { sg_drain(c, cap); };

  drain();

  std::vector<double> hist;
  double cur = 0, last_view = -1e9, last_cam = -1e9, t_end = 0;
  long steps = 0, keyframes = 0, cams = 0;
  std::string camjson = "[";
  std::vector<std::array<double, 4>> gt;   // t, x, y, yaw (world)
  Rec r;
  const auto w0 = std::chrono::steady_clock::now();
  // sgrt 기록 순서: 스텝마다 'P'(관절·오도메트리), 영상이 있으면 같은 시각 'I' 가 바로 뒤. 'L'(라이다)·'G'(정답 자세)는 그 사이.
  // 'P' 하나를 들고 있다가 다음 레코드가 같은 시각 'I' 면 영상과 함께, 아니면 관절만으로 sgrt_step.
  std::vector<float> pend;
  double pend_t = 0;
  bool have_pend = false, warned_rgb = false;
  auto step = [&](const Rec* im) {
    cur = pend_t;
    t_end = cur;
    cap.now = cur;
    if (im && !im->rgba.empty()) {
      sgrt_step(sg, pend_t, pend.data(), int(pend.size()), im->rgba.data(), 0, int64_t(im->w) * 4, 4, im->w, im->h, im->f.data(), im->K[0], im->K[1], im->K[2], im->K[3]);
      ++keyframes;
    } else {
      if (im && !warned_rgb) { warned_rgb = true; std::fprintf(stderr, "og2sg: record has no RGB — perception cannot re-run (record with RGB)\n"); }
      sgrt_step(sg, pend_t, pend.data(), int(pend.size()), nullptr, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0);
    }
    if (cur - last_view >= 1.0 / view_hz) { sm_stream_view(c); last_view = cur; }
    drain();
    ++steps;
    have_pend = false;
  };
  auto flush = [&]() { if (have_pend) step(nullptr); };
  while (reader.next(&r)) {
    if (r.tag == 'G') {
      flush();
      sgrt_push_pose(sg, r.stamp, r.g[0], r.g[1], r.g[2]);   // 정답 자세: 떠밀림 진단용(SGRT_POSE=gt 일 때만 지도 자세)
      gt.push_back({r.stamp, r.g[0], r.g[1], r.g[2]});
    } else if (r.tag == 'L') {
      flush();
      sgrt_push_scan(sg, r.stamp, int(r.f.size()), r.f.data(), r.scan[0], r.scan[1], r.scan[2], r.scan[3], r.scan[4]);
    } else if (r.tag == 'P') {
      flush();
      if (max_frames > 0 && steps >= max_frames) break;
      pend = r.f;
      pend_t = r.stamp;
      have_pend = true;
    } else if (r.tag == 'I') {
      if (!have_pend || r.stamp != pend_t) continue;   // 짝 없는 영상(기록 앞뒤 잘림)은 버림
      step(&r);
      if (!r.rgba.empty() && cur - last_cam >= 1.0 / cam_hz) {   // 카메라 그림: 256 폭으로 줄여 JPEG
        const int W = 256, H = std::max(1, r.h * W / r.w);
        std::vector<uint8_t> rgb(size_t(W) * H * 3);
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const uint8_t* s = r.rgba.data() + (size_t(y * r.h / H) * r.w + size_t(x * r.w / W)) * 4;
            uint8_t* o = rgb.data() + (size_t(y) * W + x) * 3;
            o[0] = s[0]; o[1] = s[1]; o[2] = s[2];
          }
        const std::vector<uint8_t> j = jpeg::encode(rgb.data(), W, H, 80);
        char fn[64];
        std::snprintf(fn, sizeof fn, "cam/%06ld.jpg", cams);
        FILE* jf = std::fopen((out + "/" + fn).c_str(), "wb");
        if (jf) { std::fwrite(j.data(), 1, j.size(), jf); std::fclose(jf); }
        camjson += std::string(cams ? "," : "") + "{\"t\":" + jnum(cur) + ",\"file\":\"" + fn + "\"}";
        ++cams;
        last_cam = cur;
      }
    }
  }
  flush();
  sm_stream_view(c);
  drain();
  // 메모리 폴더(sgview 가 /file/ 로 읽는 물체 조각·점구름)
  sm_save_stats ss{};
  sm_save_dsg_ex(c, (out + "/memory").c_str(), &ss);
  sm_stream_stats st{};
  sm_stream_get_stats(c, &st);
  sm_stream_stop(c);
  sm_snapshot_t* snap = nullptr;
  sm_snapshot(c, &snap);
  const sm_object* objs = nullptr;
  const int no = sm_snap_objects(snap, &objs);
  // 이름은 실제 파이프라인(SigLIP 2 + objprob)이 붙인 것 그대로 — 따로 짝짓지 않음
  const std::string labels = "{}";
  const int nl = 0;
  sm_snapshot_release(snap);
  sgrt_destroy(sg);
  cap.stop();
  // GT 궤적 → map 좌표(frame.json map_from_world = [cos, sin, tx, ty]: map = R·world + t)
  const std::string fj = slurp(run + "/frame.json");
  double mfw[4] = {1, 0, 0, 0};
  {
    const size_t p = fj.find("map_from_world");
    if (p != std::string::npos) std::sscanf(fj.c_str() + fj.find('[', p), "[%lf, %lf, %lf, %lf", &mfw[0], &mfw[1], &mfw[2], &mfw[3]);
  }
  std::string gtj = "[";
  double lt = -1;
  int ng = 0;
  for (auto& g : gt) {
    if (g[0] - lt < 0.1) continue;   // 10 Hz 로 솎음
    lt = g[0];
    const double mx = mfw[0] * g[1] - mfw[1] * g[2] + mfw[2], my = mfw[1] * g[1] + mfw[0] * g[2] + mfw[3], myaw = g[3] + std::atan2(mfw[1], mfw[0]);
    gtj += std::string(ng++ ? "," : "") + "[" + jnum(g[0]) + "," + jnum(mx) + "," + jnum(my) + "," + jnum(myaw) + "]";
  }
  gtj += "]";
  const std::string summ = slurp(run + "/summary.json");
  auto sget = [&](const char* k) { const size_t p = summ.rfind(std::string("\"") + k + "\"");   /* 맨 위 칸(앞의 coverage_marks 안 같은 이름이 아니라) */ if (p == std::string::npos) return std::string("null"); size_t a = summ.find(':', p) + 1; while (summ[a] == ' ') ++a; size_t b = a; while (b < summ.size() && summ[b] != ',' && summ[b] != '\n' && summ[b] != '}') ++b; return summ.substr(a, b - a); };
  std::string runname = run;
  while (!runname.empty() && runname.back() == '/') runname.pop_back();
  runname = runname.substr(runname.rfind('/') + 1);
  const std::string meta = Obj().str("run", runname).str("skill", "explore").raw("task", sget("task")).raw("end", sget("end")).raw("gt_cov", sget("gt_cov"))
                               .raw("path_m", sget("path_m")).raw("sim_s", sget("sim_s")).num("t", t_end).num("steps", steps).num("keyframes", keyframes)
                               .str("outcome", "explore").str("driver", "frontier").b("success", false).str("map_mode", "slam").done();
  const std::string mj = Obj().str("format", "SGS1").raw("meta", meta).str("og_run", run).str("rec", recp).str("robot", robot)
                             .raw("map_from_world", "[" + jnum(mfw[0]) + "," + jnum(mfw[1]) + "," + jnum(mfw[2]) + "," + jnum(mfw[3]) + "]")
                             .raw("gt_path", gtj).raw("labels", labels).raw("cams", camjson + "]").num("duration", t_end)
                             .raw("stream", Obj().num("frames", (double)cap.frames).num("pose", (double)cap.by_type[1]).num("map", (double)cap.by_type[2])
                                                .num("view", (double)cap.by_type[3]).num("joints", (double)cap.by_type[4]).num("dropped", (double)st.dropped).done())
                             .raw("joint_order", "[\"\",\"\",\"\",\"\",\"\",\"\",\"omx_joint1\",\"omx_joint2\",\"omx_joint3\",\"omx_joint4\",\"omx_joint5\",\"omx_gripper_joint_1\",\"front_left_wheel\",\"front_right_wheel\",\"rear_left_wheel\",\"rear_right_wheel\"]")
                             .num("n_objects", no).done();
  std::ofstream(out + "/meta.json") << mj;
  // 바탕 층(BEHAVIOR 집 배치, 세계 좌표) — 화면은 map_from_world 로 돌려 놓는다
  std::string scene_name;
  if (!rasc_path.empty()) {
    std::string task = sget("task");
    if (task.size() > 1 && task[0] == '"') task = task.substr(1, task.size() - 2);
    std::ofstream(out + "/underlay.json") << underlay_json(rasc_path, task, &scene_name);
  }
  // 실행 폴더(학습 뷰어 규약): run.json + episodes.jsonl 한 줄
  if (!run_out.empty()) {
    const std::string ep = Obj().num("ep", 0).str("skill", "explore").str("home", scene_name).raw("task", sget("task")).str("stage", "B1").str("outcome", "explore")
                               .str("driver", "frontier").raw("success", "null").num("t", t_end).num("steps", steps).raw("gt_cov", sget("gt_cov")).raw("path_len", sget("path_m"))
                               .str("map_mode", "slam").str("replay", out.substr(out.rfind('/') + 1)).done();
    std::ofstream(run_out + "/episodes.jsonl") << ep << "\n";
    std::ofstream(run_out + "/run.json") << Obj().num("schema", 1).str("kind", "behavior").str("name", runname).str("group", "behavior_og_limo").str("trainer", "og2sg")
                                                .str("note", "OmniGibson LIMO 탐사 한 판(진짜 시뮬 기록 rec.bin) — 실제 파이프라인 libsgrt 로 다시 돌린 sgview 스트림").str("og_run", run).str("rec", recp)
                                                .raw("task", sget("task")).str("scene", scene_name).raw("skills", "[\"explore\"]").num("ended", 1)
                                                .raw("logged", "{\"progress\": false, \"episodes\": true, \"replays\": true}").done();
  }
  std::printf("og2sg: %s -> %s: %ld steps (%.1f s sim), %ld keyframes, %ld camera jpeg, stream %llu frames (pose %llu map %llu view %llu joints %llu, dropped %llu), %d objects, labels %d, %.1f s wall\n",
              recp.c_str(), out.c_str(), steps, t_end, keyframes, cams, (unsigned long long)cap.frames, (unsigned long long)cap.by_type[1], (unsigned long long)cap.by_type[2],
              (unsigned long long)cap.by_type[3], (unsigned long long)cap.by_type[4], (unsigned long long)st.dropped, no, nl,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
  return 0;
}
