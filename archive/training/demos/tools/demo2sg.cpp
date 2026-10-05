// demo2sg — 사람 시연 한 판(demo2rec.py 가 쓴 SGRC rec.bin + labels.txt + demo.json) → scenemap 재생 → 학습 뷰어 재생 판(<이름>.sg/)
//            + (선택) 지도 토큰(MapTok, sm_tok.h)을 keyframe 마다.
//
// training/viewer/tools/og2sg 와 같은 방식(scenemap 자신의 sgview 스트림을 받아 시뮬 시각을 붙여 stream.sgs 로, 끝에 sm_save_dsg)이다. 다른 점:
//   - 검출 이름 = labels.txt(정답 범주 이름) — og2sg 는 cls<k> 를 나중에 정답 실행과 짝지어 바꿈
//   - G(정답 베이스 자세)는 이미 map 프레임(= t0 베이스) → gt_path 그대로, 바탕 층은 demo.json map_from_world 로 돌림
//   - 로봇 고르기(r1pro 기본), 자세 원천(slam|gt), 실행 폴더 group = human_demos, 선택 --ingest 로 sgview 에 실시간으로도 보냄
//
//   demo2sg --demo <demo2rec 출력 폴더> --run-out <trainview 뿌리>/human_demos/<이름> [--rasc house.rasc] [--pose slam|gt] [--robot r1pro]
//           [--view-hz 10] [--cam-hz 5] [--ingest 127.0.0.1:9097] [--realtime 1]
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rasc.h"
#include "rec_util.h"
#include "scenemap.h"

using namespace rec;

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::string b64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  for (size_t i = 0; i < d.size(); i += 3) {
    const uint32_t v = (uint32_t(d[i]) << 16) | (i + 1 < d.size() ? uint32_t(d[i + 1]) << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(v >> 18) & 63]; o += T[(v >> 12) & 63];
    o += i + 1 < d.size() ? T[(v >> 6) & 63] : '=';
    o += i + 2 < d.size() ? T[v & 63] : '=';
  }
  return o;
}
static float qyaw(const float q[4]) { return std::atan2(2.f * (q[3] * q[2] + q[0] * q[1]), 1.f - 2.f * (q[1] * q[1] + q[2] * q[2])); }

// 바탕 층(BEHAVIOR 집 배치, 세계 좌표) — og2sg underlay_json 과 같은 내용
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
      if (S.free_cell(1, int(r), int(c))) continue;
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
    if (b.flags & (RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING)) continue;
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
  std::string tobjs = "[";
  int nto = 0;
  for (const RascTaskRec& T : S.tasks) {
    if (task_name.empty() || S.str(T.name) != task_name) continue;
    auto ins = S.insts_of(T);
    if (!ins.size()) break;
    auto to = S.objs_of(T);
    auto po = S.poses_of(ins[0]);
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
      .str("room_grid_b64", b64(rooms)).str("wall_grid_b64", b64(walls)).raw("rooms", roomsj + "]").raw("boxes", boxes + "]").raw("places", "[]").raw("picks", "[]")
      .raw("task_objects", tobjs + "]").str("task", task_name).done();
}

// ---- SGRC 한 레코드씩 ----
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

// ---- 스트림 받기(+ 선택: sgview --ingest 로 그대로 넘김) ----
struct Capture {
  int lfd = -1, port = 0, fwd = -1;
  std::atomic<double> now{0.0};
  std::atomic<uint64_t> bytes{0};
  FILE* out = nullptr;
  std::thread th;
  uint64_t frames = 0, by_type[8] = {};
  bool start(const std::string& path, const std::string& ingest) {
    out = std::fopen(path.c_str(), "wb");
    if (!out) return false;
    std::fwrite("SGS1", 1, 4, out);
    if (!ingest.empty()) {   // sgview --ingest 주소(host:port) — 못 붙으면 파일만
      const size_t c = ingest.rfind(':');
      sockaddr_in a{};
      a.sin_family = AF_INET; a.sin_port = htons(uint16_t(std::atoi(ingest.c_str() + c + 1)));
      inet_pton(AF_INET, ingest.substr(0, c).c_str(), &a.sin_addr);
      fwd = socket(AF_INET, SOCK_STREAM, 0);
      if (connect(fwd, (sockaddr*)&a, sizeof a)) { std::fprintf(stderr, "ingest %s: connect failed (file only)\n", ingest.c_str()); close(fwd); fwd = -1; }
    }
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
    auto wr_all = [&](const void* p, size_t n) {
      const uint8_t* q = (const uint8_t*)p;
      while (fwd >= 0 && n) { const ssize_t k = write(fwd, q, n); if (k <= 0) { close(fwd); fwd = -1; return; } q += k; n -= size_t(k); }
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
      wr_all(hd, 5);
      wr_all(buf.data(), len);
      ++frames;
      if (hd[4] < 8) ++by_type[hd[4]];
    }
    close(c);
  }
  void stop() {
    if (th.joinable()) th.join();
    if (lfd >= 0) close(lfd);
    if (fwd >= 0) close(fwd);
    if (out) std::fclose(out);
  }
};

// demo.json 에서 숫자·문자열 하나(작은 읽기 — 맨 위 키만 씀)
static std::string jget(const std::string& j, const char* k) {
  const size_t p = j.find(std::string("\"") + k + "\"");
  if (p == std::string::npos) return "null";
  size_t a = j.find(':', p) + 1;
  while (j[a] == ' ') ++a;
  size_t b = a;
  int depth = 0;
  bool str = false;
  for (; b < j.size(); ++b) {
    const char ch = j[b];
    if (str) { if (ch == '\\') ++b; else if (ch == '"') str = false; continue; }
    if (ch == '"') str = true;
    else if (ch == '[' || ch == '{') ++depth;
    else if (ch == ']' || ch == '}') { if (depth == 0) break; --depth; }
    else if (ch == ',' && depth == 0) break;
  }
  return j.substr(a, b - a);
}

int main(int argc, char** argv) {
  std::string demo, out, run_out, rasc_path, robot = "r1pro", pose = "slam", ingest, name;
  double view_hz = 10, cam_hz = 5, realtime = 0;
  long max_frames = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--demo") demo = nx(); else if (a == "--out") out = nx(); else if (a == "--run-out") run_out = nx(); else if (a == "--rasc") rasc_path = nx();
    else if (a == "--robot") robot = nx(); else if (a == "--pose") pose = nx(); else if (a == "--ingest") ingest = nx(); else if (a == "--name") name = nx();
    else if (a == "--view-hz") view_hz = std::stod(nx()); else if (a == "--cam-hz") cam_hz = std::stod(nx()); else if (a == "--frames") max_frames = std::stol(nx());
    else if (a == "--realtime") realtime = std::stod(nx());
  }
  if (demo.empty() || (out.empty() && run_out.empty())) {
    std::fprintf(stderr, "usage: demo2sg --demo DIR --run-out RUN_DIR [--out EP.sg] [--rasc R] [--pose slam|gt] [--robot r1pro] [--ingest host:port] [--realtime x]\n");
    return 2;
  }
  const std::string dj = slurp(demo + "/demo.json");
  if (dj.empty()) { std::fprintf(stderr, "no %s/demo.json\n", demo.c_str()); return 1; }
  std::string task = jget(dj, "task"), scene = jget(dj, "scene"), ep = jget(dj, "episode"), mfw = jget(dj, "map_from_world"), src = jget(dj, "map_src");
  if (task.size() > 2 && task[0] == '[') task = task.substr(1, task.size() - 2);   // ["turning_on_radio"]
  auto unq = [](std::string s) { return s.size() >= 2 && s[0] == '"' ? s.substr(1, s.size() - 2) : s; };
  const std::string task_s = unq(task), scene_s = unq(scene);
  if (out.empty()) out = run_out + "/replays/ep_" + std::string(6 - std::min<size_t>(6, ep.size()), '0') + ep + "_demo.sg";
  mkdirs(out + "/cam");
  std::vector<std::string> labels;
  {
    std::ifstream lf(demo + "/labels.txt");
    for (std::string l; std::getline(lf, l);) if (!l.empty()) labels.push_back(l);
  }
  FILE* rf = std::fopen((demo + "/rec.bin").c_str(), "rb");
  char magic[4];
  uint32_t ver = 0;
  if (!rf || std::fread(magic, 1, 4, rf) != 4 || std::memcmp(magic, "SGRC", 4) || !rd(rf, &ver) || ver != 1) { std::fprintf(stderr, "not an SGRC v1 record: %s/rec.bin\n", demo.c_str()); return 1; }

  Capture cap;
  if (!cap.start(out + "/stream.sgs", ingest)) { std::perror("capture"); return 1; }
  const std::string cfg = "{\"robot\": \"" + robot + "\"}";
  sm_ctx* c = sm_create(cfg.c_str());
  if (!c) { std::fprintf(stderr, "sm_create rejected %s\n", cfg.c_str()); return 1; }
  std::vector<const char*> np;
  for (auto& n : labels) np.push_back(n.c_str());
  sm_set_labels(c, np.data(), int(np.size()));
  sm_set_pose_mode(c, pose == "gt" ? SM_POSE_GT : SM_POSE_SLAM);
  const std::string hp = "127.0.0.1:" + std::to_string(cap.port);
  if (sm_stream_start(c, hp.c_str())) { std::fprintf(stderr, "sm_stream_start failed\n"); return 1; }
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
  std::string camjson = "[", gtj = "[";
  double lt = -1;
  int ng = 0;
  Rec r;
  const auto w0 = std::chrono::steady_clock::now();
  double sm_us = 0;
  while (next(rf, r)) {
    if (r.tag == 'G') {
      const sm_pose2 p{r.stamp, r.g[0], r.g[1], r.g[2]};
      sm_push_pose(c, &p);
      if (r.stamp - lt >= 0.1) { lt = r.stamp; gtj += std::string(ng++ ? "," : "") + "[" + jnum(r.stamp) + "," + jnum(r.g[0]) + "," + jnum(r.g[1]) + "," + jnum(r.g[2]) + "]"; }
    } else if (r.tag == 'P') {
      if (max_frames > 0 && steps >= max_frames) break;
      cur = r.stamp;
      t_end = cur;
      cap.now = cur;
      hist.push_back(cur);
      if (hist.size() > 8) hist.erase(hist.begin());
      const auto a0 = std::chrono::steady_clock::now();
      sm_proprio p{r.stamp, r.f.data(), int(r.f.size())};
      sm_push_proprio(c, &p);
      sm_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count();
      if (cur - last_view >= 1.0 / view_hz) { sm_stream_view(c); last_view = cur; }
      drain();
      ++steps;
      if (realtime > 0) {   // 실시간 보기(sgview --ingest): 시뮬 시각 / realtime 배속에 맞춰 걸음
        const double want = cur / realtime, have = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        if (want > have) std::this_thread::sleep_for(std::chrono::duration<double>(want - have));
      }
    } else if (r.tag == 'I') {
      const double st = hist.size() >= 2 ? hist[hist.size() - 2] : cur;
      sm_image im{st, 0, r.w, r.h, r.rgba.empty() ? nullptr : r.rgba.data(), r.f.data(), r.K[0], r.K[1], r.K[2], r.K[3]};
      sm_detections d{};
      d.stamp = st; d.img_w = r.img_w; d.img_h = r.img_h; d.n = r.n;
      d.cls = r.cls.data(); d.score = r.score.data(); d.box = r.box.data();
      d.mask_w = r.mask_w; d.mask_h = r.mask_h; d.mask_sx = r.msx; d.mask_sy = r.msy; d.mask_ox = r.mox; d.mask_oy = r.moy; d.mask_bits = r.bits.data();
      const auto a0 = std::chrono::steady_clock::now();
      sm_push_image_rgb(c, &im, r.img_w > 0 ? &d : nullptr, nullptr);
      sm_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count();
      drain();
      ++keyframes;
      if (std::getenv("DEMO_TRACE")) {   // 진단: keyframe 마다 든(HELD) 물체·구름 점 수·상자
        sm_snapshot_t* sn = nullptr;
        sm_snapshot(c, &sn);
        const sm_object* ob = nullptr;
        const int k = sm_snap_objects(sn, &ob);
        for (int i = 0; i < k; ++i) {
          if (ob[i].state != SM_HELD) continue;
          sm_cloud cl{};
          sm_snap_points(sn, ob[i].id, &cl);
          std::printf("TRACE t %.2f held id %u %s ext %.3f %.3f %.3f pos %.2f %.2f %.2f n_obs %u pts %d grip %.3f %.3f\n", cur, ob[i].id, ob[i].name,
                      ob[i].extent[0], ob[i].extent[1], ob[i].extent[2], ob[i].pos[0], ob[i].pos[1], ob[i].pos[2], ob[i].n_obs, cl.n, r.f.size() ? 0.f : 0.f, 0.f);
        }
        sm_snapshot_release(sn);
      }
      if (!r.rgba.empty() && cur - last_cam >= 1.0 / cam_hz) {
        const int W = r.w, H = r.h;
        std::vector<uint8_t> rgb(size_t(W) * H * 3);
        for (size_t i = 0; i < size_t(W) * H; ++i) { rgb[3 * i] = r.rgba[4 * i]; rgb[3 * i + 1] = r.rgba[4 * i + 1]; rgb[3 * i + 2] = r.rgba[4 * i + 2]; }
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
  sm_save_stats ss{};
  sm_save_dsg_ex(c, (out + "/memory").c_str(), &ss);
  sm_stream_stats st{};
  sm_stream_get_stats(c, &st);
  sm_stream_stop(c);
  sm_snapshot_t* snap = nullptr;
  sm_snapshot(c, &snap);
  const sm_object* objs = nullptr;
  const int no = sm_snap_objects(snap, &objs);
  sm_snapshot_release(snap);
  sm_destroy(c);
  cap.stop();
  gtj += "]";
  if (name.empty()) name = "b1k_ep" + ep + "_" + task_s;
  std::string labj = "{";
  for (size_t k = 0; k < labels.size(); ++k) labj += std::string(k ? "," : "") + jstr(labels[k]) + ":" + jstr(labels[k]);
  labj += "}";
  const std::string meta = Obj().str("run", name).str("skill", "human_demo").str("task", task_s).str("home", scene_s).num("t", t_end).num("steps", steps).num("keyframes", keyframes)
                               .str("outcome", "demo").str("driver", "human_teleop").raw("success", "null").str("map_mode", pose).str("map_src", unq(src)).raw("episode", ep).done();
  const std::string mj = Obj().str("format", "SGS1").raw("meta", meta).str("demo", demo).str("robot", robot).raw("map_from_world", mfw).raw("gt_path", gtj).raw("labels", labj)
                             .raw("cams", camjson + "]").num("duration", t_end)
                             .raw("stream", Obj().num("frames", (double)cap.frames).num("pose", (double)cap.by_type[1]).num("map", (double)cap.by_type[2])
                                                .num("view", (double)cap.by_type[3]).num("joints", (double)cap.by_type[4]).num("dropped", (double)st.dropped).done())
                             .raw("joint_order", "[]").num("n_objects", no).done();
  std::ofstream(out + "/meta.json") << mj;
  std::string scene_name = scene_s;
  if (!rasc_path.empty()) std::ofstream(out + "/underlay.json") << underlay_json(rasc_path, task_s, &scene_name);
  if (!run_out.empty()) {
    const std::string epl = Obj().num("ep", std::atof(ep.c_str())).str("skill", "human_demo").str("home", scene_name).str("task", task_s).str("stage", "demo").str("outcome", "demo")
                                .str("driver", "human_teleop").raw("success", "null").num("t", t_end).num("steps", steps).str("map_mode", pose)
                                .str("replay", out.substr(out.rfind('/') + 1)).done();
    std::ofstream(run_out + "/episodes.jsonl", std::ios::app) << epl << "\n";
    std::ofstream(run_out + "/run.json") << Obj().num("schema", 1).str("kind", "behavior").str("name", name).str("group", "human_demos").str("trainer", "demo2sg")
                                                .str("note", "BEHAVIOR 2026 사람 원격조종 시연(R1 Pro) — 기록을 scenemap 으로 다시 돌린 sgview 스트림. 검출 = 정답 물체 상자(map_src)")
                                                .str("demo", demo).str("task", task_s).str("scene", scene_name).raw("skills", "[\"human_demo\"]").num("ended", 1)
                                                .raw("logged", "{\"progress\": false, \"episodes\": true, \"replays\": true}").done();
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  std::printf("demo2sg: %s -> %s: %ld steps (%.1f s sim), %ld keyframes, %ld jpeg, stream %llu frames (dropped %llu), %d objects, scenemap %.2f s, %.1f s wall\n",
              demo.c_str(), out.c_str(), steps, t_end, keyframes, cams, (unsigned long long)cap.frames, (unsigned long long)st.dropped, no, sm_us * 1e-6, wall);
  return 0;
}
