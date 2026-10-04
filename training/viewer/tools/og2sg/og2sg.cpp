// og2sg — OmniGibson LIMO 탐사 한 판(sgrt 기록 rec.bin + 실행 폴더) → 학습 뷰어 재생용 "sgview 판"(<이름>.sg/ 폴더).
//
// sgview(장면 그래프 실시간 뷰어)가 그리는 것과 똑같게 보이도록, 기록을 scenemap(C ABI, behavior-2026 서브모듈 — 읽기만)으로 다시 돌리면서
// scenemap 자신의 sgview 스트림(sm_stream_start: POSE · MAP_RECT · VIEW · JOINTS 프레임)을 켜고, 그 스트림을 이 프로세스가 소켓으로 받아
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

// ---- 기록 한 줄씩 읽기(sgrec.hpp 와 같은 형식, 전부 메모리에 올리지 않음 — 2 GB 기록) ----
struct Rec {
  char tag = 0;
  double stamp = 0;
  std::vector<float> f;
  double g[3]{};
  int w = 0, h = 0;
  double K[4]{};
  std::vector<uint8_t> rgba;
  int n = 0, img_w = 0, img_h = 0, mask_w = 0, mask_h = 0;
  float msx = 0, msy = 0, mox = 0, moy = 0;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
};
template <class T> static bool rd(FILE* f, T* v) { return std::fread(v, sizeof(T), 1, f) == 1; }
static bool next(FILE* f, Rec& r) {
  const int t = std::fgetc(f);
  if (t == EOF) return false;
  r.tag = char(t);
  if (!rd(f, &r.stamp)) return false;
  if (t == 'P') {
    int32_t n;
    if (!rd(f, &n)) return false;
    r.f.resize(n);
    return std::fread(r.f.data(), 4, n, f) == size_t(n);
  }
  if (t == 'G') return std::fread(r.g, 8, 3, f) == 3;
  if (t != 'I') return false;
  int32_t w, h;
  if (!rd(f, &w) || !rd(f, &h) || std::fread(r.K, 8, 4, f) != 4) return false;
  r.w = w; r.h = h;
  r.f.resize(size_t(w) * h);
  if (std::fread(r.f.data(), 4, r.f.size(), f) != r.f.size()) return false;
  r.rgba.clear();
  if (std::fgetc(f) == 1) {
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    if (std::fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) return false;
    r.rgba.resize(size_t(w) * h * 4);
    for (size_t i = 0; i < size_t(w) * h; ++i) { r.rgba[4 * i] = rgb[3 * i]; r.rgba[4 * i + 1] = rgb[3 * i + 1]; r.rgba[4 * i + 2] = rgb[3 * i + 2]; r.rgba[4 * i + 3] = 255; }
  }
  int32_t n, iw, ih, mw, mh;
  if (!rd(f, &n) || !rd(f, &iw) || !rd(f, &ih) || !rd(f, &mw) || !rd(f, &mh) || !rd(f, &r.msx) || !rd(f, &r.msy) || !rd(f, &r.mox) || !rd(f, &r.moy)) return false;
  r.n = n; r.img_w = iw; r.img_h = ih; r.mask_w = mw; r.mask_h = mh;
  r.cls.resize(n); r.score.resize(n); r.box.resize(4 * size_t(n));
  const size_t words = (size_t(mw) * mh + 31) / 32;
  r.bits.resize(words * n);
  if (n && (std::fread(r.cls.data(), 4, n, f) != size_t(n) || std::fread(r.score.data(), 4, n, f) != size_t(n) || std::fread(r.box.data(), 4, r.box.size(), f) != r.box.size() ||
            std::fread(r.bits.data(), 4, r.bits.size(), f) != r.bits.size()))
    return false;
  return true;
}

// ---- 스트림 받기(scenemap 이 sgview --ingest 에 보내듯 이 프로세스로) ----
struct Capture {
  int lfd = -1, port = 0;
  std::atomic<double> now{0.0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<bool> quit{false};
  FILE* out = nullptr;
  std::thread th;
  uint64_t frames = 0, by_type[8] = {};
  bool start(const std::string& path) {
    out = std::fopen(path.c_str(), "wb");
    if (!out) return false;
    std::fwrite("SGS1", 1, 4, out);
    lfd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (bind(lfd, (sockaddr*)&a, sizeof a) || listen(lfd, 1)) return false;
    socklen_t al = sizeof a;
    getsockname(lfd, (sockaddr*)&a, &al);
    port = ntohs(a.sin_port);
    th = std::thread([this] { run(); });
    return true;
  }
  void run() {
    const int c = accept(lfd, nullptr, nullptr);
    if (c < 0) return;
    std::vector<uint8_t> buf;
    auto rd_all = [&](void* p, size_t n) {
      uint8_t* q = (uint8_t*)p;
      while (n) { const ssize_t k = read(c, q, n); if (k <= 0) return false; q += k; n -= size_t(k); bytes += uint64_t(k); }
      return true;
    };
    for (;;) {
      uint8_t hd[5];
      if (!rd_all(hd, 5)) break;
      uint32_t len;
      std::memcpy(&len, hd, 4);
      buf.resize(len);
      if (!rd_all(buf.data(), len)) break;
      const double t = now.load();
      std::fwrite(&t, 8, 1, out);
      std::fwrite(hd, 1, 5, out);
      std::fwrite(buf.data(), 1, len, out);
      ++frames;
      if (hd[4] < 8) ++by_type[hd[4]];
    }
    close(c);
  }
  void stop() {
    if (th.joinable()) th.join();
    if (lfd >= 0) close(lfd);
    if (out) std::fclose(out);
  }
};

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
  std::string recp, run, out, robot = "limo_omx", rasc_path, run_out;
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
  FILE* rf = std::fopen(recp.c_str(), "rb");
  char magic[4];
  uint32_t ver = 0;
  if (!rf || std::fread(magic, 1, 4, rf) != 4 || std::memcmp(magic, "SGRC", 4) || !rd(rf, &ver) || ver != 1) { std::fprintf(stderr, "not an SGRC v1 record: %s\n", recp.c_str()); return 1; }

  Capture cap;
  if (!cap.start(out + "/stream.sgs")) { std::perror("capture"); return 1; }
  const std::string cfg = "{\"robot\": \"" + robot + "\"}";
  sm_ctx* c = sm_create(cfg.c_str());
  if (!c) { std::fprintf(stderr, "sm_create rejected %s\n", cfg.c_str()); return 1; }
  // 라벨: 기록에는 검출 번호(cls)만 있다 — 우선 cls<k>, 끝에 정답 실행의 view.json 물체와 위치로 짝지어 이름을 바꾼다
  std::vector<std::string> names;
  for (int k = 0; k < 512; ++k) names.push_back("cls" + std::to_string(k));
  std::vector<const char*> np;
  for (auto& n : names) np.push_back(n.c_str());
  sm_set_labels(c, np.data(), int(np.size()));
  sm_set_pose_mode(c, SM_POSE_SLAM);
  const std::string hp = "127.0.0.1:" + std::to_string(cap.port);
  if (sm_stream_start(c, hp.c_str())) { std::fprintf(stderr, "sm_stream_start failed\n"); return 1; }
  // 받는 쪽이 붙고 따라잡을 때까지 기다림(스트림은 링이 차면 프레임을 버린다 — 기록이므로 한 프레임도 버리지 않게 걸음을 맞춘다)
  // frames_sent 는 묶음 수라 프레임 수와 견줄 수 없다: 보낸 바이트가 두 번 연달아(송신 스레드 한 바퀴 ≥ 100 µs) 그대로이고 받은 바이트가 따라잡으면 비었다고 본다
  auto drain = [&]() {
    uint64_t prev = ~0ull;
    int same = 0;
    for (int k = 0; k < 20000; ++k) {
      sm_stream_stats ss{};
      sm_stream_get_stats(c, &ss);
      if (ss.connected && ss.bytes_sent == prev && cap.bytes.load() >= ss.bytes_sent) { if (++same >= 2) return; }
      else same = 0;
      prev = ss.bytes_sent;
      std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
  };
  drain();

  std::vector<double> hist;
  double cur = 0, last_view = -1e9, last_cam = -1e9, t_end = 0;
  long steps = 0, keyframes = 0, cams = 0;
  std::string camjson = "[";
  std::vector<std::array<double, 4>> gt;   // t, x, y, yaw (world)
  Rec r;
  const auto w0 = std::chrono::steady_clock::now();
  while (next(rf, r)) {
    if (r.tag == 'G') {
      const sm_pose2 p{r.stamp, r.g[0], r.g[1], r.g[2]};
      sm_push_pose(c, &p);
      gt.push_back({r.stamp, r.g[0], r.g[1], r.g[2]});
    } else if (r.tag == 'P') {
      if (max_frames > 0 && steps >= max_frames) break;
      cur = r.stamp;
      t_end = cur;
      cap.now = cur;
      hist.push_back(cur);
      if (hist.size() > 8) hist.erase(hist.begin());
      sm_proprio p{r.stamp, r.f.data(), int(r.f.size())};
      sm_push_proprio(c, &p);   // 자세·관절 프레임은 이 안에서 스트림으로
      if (cur - last_view >= 1.0 / view_hz) { sm_stream_view(c); last_view = cur; }
      drain();
      ++steps;
    } else if (r.tag == 'I') {
      const double st = hist.size() >= 2 ? hist[hist.size() - 2] : cur;   // 영상 stamp = 한 스텝 앞(sm_bench --lag 1 과 같음)
      sm_image im{st, 0, r.w, r.h, r.rgba.empty() ? nullptr : r.rgba.data(), r.f.data(), r.K[0], r.K[1], r.K[2], r.K[3]};
      sm_detections d{};
      d.stamp = st; d.img_w = r.img_w; d.img_h = r.img_h; d.n = r.n;
      d.cls = r.cls.data(); d.score = r.score.data(); d.box = r.box.data();
      d.mask_w = r.mask_w; d.mask_h = r.mask_h; d.mask_sx = r.msx; d.mask_sy = r.msy; d.mask_ox = r.mox; d.mask_oy = r.moy; d.mask_bits = r.bits.data();
      sm_push_image_rgb(c, &im, r.img_w > 0 ? &d : nullptr, nullptr);
      drain();
      ++keyframes;
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
  std::fclose(rf);
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
  // 라벨 이름 짝: 정답 실행(같은 기록으로 돈 원래 판)의 memory/view.json 물체와 map 좌표 거리로 — cls<k> 마다 가장 많이 맞은 이름
  std::map<std::string, std::map<std::string, int>> votes;
  {
    const std::string vj = slurp(run + "/memory/view.json");
    // 아주 작은 읽기: "name": "...", "state": ..., "pos": [x, y, z] 를 차례로
    struct O { std::string name; double x, y, z; };
    std::vector<O> ref;
    size_t p = vj.find("\"objects\"");
    const size_t pend = vj.find("\"events\"");
    while (p != std::string::npos && p < pend) {
      p = vj.find("\"name\"", p);
      if (p == std::string::npos || p > pend) break;
      const size_t a = vj.find('"', vj.find(':', p) + 1), b = vj.find('"', a + 1);
      const std::string nm = vj.substr(a + 1, b - a - 1);
      const size_t q = vj.find("\"pos\"", b);
      const size_t lb = vj.find('[', q);
      double x = 0, y = 0, z = 0;
      std::sscanf(vj.c_str() + lb, "[%lf, %lf, %lf", &x, &y, &z);
      ref.push_back({nm, x, y, z});
      p = lb;
    }
    for (int k = 0; k < no; ++k) {
      double best = 0.75, bd = 1e9;
      std::string bn;
      for (const O& o : ref) {
        const double d = std::hypot(o.x - objs[k].pos[0], o.y - objs[k].pos[1], o.z - objs[k].pos[2]);
        if (d < best && d < bd) { bd = d; bn = o.name; }
      }
      if (!bn.empty()) votes[objs[k].name][bn]++;
    }
  }
  std::string labels = "{";
  int nl = 0;
  for (auto& [k, m] : votes) {
    std::string bn;
    int bv = 0;
    for (auto& [n, v] : m) if (v > bv) { bv = v; bn = n; }
    labels += std::string(nl++ ? "," : "") + jstr(k) + ":" + jstr(bn);
  }
  labels += "}";
  sm_snapshot_release(snap);
  sm_destroy(c);
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
                                                .str("note", "OmniGibson LIMO 탐사 한 판(진짜 시뮬 기록 rec.bin) — scenemap 으로 다시 돌린 sgview 스트림").str("og_run", run).str("rec", recp)
                                                .raw("task", sget("task")).str("scene", scene_name).raw("skills", "[\"explore\"]").num("ended", 1)
                                                .raw("logged", "{\"progress\": false, \"episodes\": true, \"replays\": true}").done();
  }
  std::printf("og2sg: %s -> %s: %ld steps (%.1f s sim), %ld keyframes, %ld camera jpeg, stream %llu frames (pose %llu map %llu view %llu joints %llu, dropped %llu), %d objects, labels %d, %.1f s wall\n",
              recp.c_str(), out.c_str(), steps, t_end, keyframes, cams, (unsigned long long)cap.frames, (unsigned long long)cap.by_type[1], (unsigned long long)cap.by_type[2],
              (unsigned long long)cap.by_type[3], (unsigned long long)cap.by_type[4], (unsigned long long)st.dropped, no, nl,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
  return 0;
}
