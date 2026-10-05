// dynamic-object-mapping-benchmark 시퀀스 → scenemap C ABI → map_timeline.csv (dom_bench·dom_bench_det 가 같이 씀).
//
// 시퀀스 폴더(toolkit 배치, toolkit/DATA_FORMAT.md): intrinsics.json · frames.csv · poses.txt(TUM, world ← 카메라 광학) ·
// rgb/*.png(8 비트 RGB) · depth/*.png(uint16 mm, z 깊이). 방법이 읽어도 되는 것은 이것뿐이다(instance/·objects.csv 는 정답 —
// '완벽한 검출'(dom_bench)만 읽는다).
//
// 카메라만 있는 기록이라 로봇 몸이 없다. scenemap 에는 이렇게 넣는다:
//   베이스 자세 = 카메라의 바닥 투영(x, y, 광축의 수평 방향 yaw) → sm_push_pose(SM_POSE_GT)
//   베이스 ← 카메라 = Rz(−yaw)·R_wc, 평행 이동 (0, 0, z) → sm_set_cam_extrinsic(순기구학 대신)
//   proprio = R1 61 개 0, 단 팔 끝 둘을 베이스 아래 100 m 에(손에 든 것 거르기·잡기 규칙이 안 걸리게), 손가락 열림(0.05 + 0.05)
// 그러면 map = 벤치마크 world(Z 위, 바닥 z = 0)이고, 물체 위치를 그대로 쓴다.
#pragma once
#include <png.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "scenemap.h"

namespace dom {

// ---------------- PNG 읽기(libpng): 8 비트 RGB(A) → RGB, 16 비트 회색 → uint16 ----------------
inline bool readPng(const std::string& path, int* w, int* h, int* ch, int* bits, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(p);
  if (setjmp(png_jmpbuf(p))) {
    png_destroy_read_struct(&p, &info, nullptr);
    std::fclose(f);
    return false;
  }
  png_init_io(p, f);
  png_read_info(p, info);
  *w = int(png_get_image_width(p, info));
  *h = int(png_get_image_height(p, info));
  *bits = png_get_bit_depth(p, info);
  if (png_get_color_type(p, info) == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(p);
  if (*bits == 16) png_set_swap(p);   // 리틀 엔디언 uint16
  png_read_update_info(p, info);
  *ch = png_get_channels(p, info);
  *bits = png_get_bit_depth(p, info);
  const size_t row = png_get_rowbytes(p, info);
  out->resize(row * size_t(*h));
  std::vector<png_bytep> rows(static_cast<size_t>(*h));
  for (int y = 0; y < *h; ++y) rows[size_t(y)] = out->data() + row * size_t(y);
  png_read_image(p, rows.data());
  png_destroy_read_struct(&p, &info, nullptr);
  std::fclose(f);
  return true;
}

inline bool readRgb(const std::string& path, int w, int h, std::vector<uint8_t>* rgb) {
  int W, H, C, B;
  std::vector<uint8_t> raw;
  if (!readPng(path, &W, &H, &C, &B, &raw) || W != w || H != h || B != 8 || C < 3) return false;
  rgb->resize(size_t(w) * h * 3);
  for (size_t i = 0; i < size_t(w) * h; ++i)
    for (int k = 0; k < 3; ++k) (*rgb)[3 * i + k] = raw[size_t(C) * i + k];
  return true;
}

inline bool readU16(const std::string& path, int w, int h, std::vector<uint16_t>* v) {
  int W, H, C, B;
  std::vector<uint8_t> raw;
  if (!readPng(path, &W, &H, &C, &B, &raw) || W != w || H != h || B != 16 || C != 1) return false;
  v->resize(size_t(w) * h);
  std::memcpy(v->data(), raw.data(), v->size() * 2);
  return true;
}

// ---------------- 시퀀스 ----------------
struct Frame {
  int idx;
  int64_t stamp_ns;
  std::string rgb, depth, inst;
  double t[3], q[4];   // world ← 카메라 광학, q = x y z w
};

struct Seq {
  std::string dir, name;
  int w = 0, h = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::vector<Frame> frames;
};

inline double jsonNum(const std::string& s, const std::string& key) {
  const size_t p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return NAN;
  const size_t c = s.find(':', p);
  return std::strtod(s.c_str() + c + 1, nullptr);
}

inline std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> v;
  std::string cur;
  std::stringstream ss(line);
  while (std::getline(ss, cur, ',')) v.push_back(cur);
  if (!line.empty() && line.back() == ',') v.emplace_back();
  return v;
}

inline bool loadSeq(const std::string& dir, Seq* s) {
  namespace fs = std::filesystem;
  s->dir = dir;
  s->name = fs::path(dir).filename().string();
  if (s->name.empty()) s->name = fs::path(dir).parent_path().filename().string();
  std::ifstream ji(dir + "/intrinsics.json");
  if (!ji) return false;
  const std::string js((std::istreambuf_iterator<char>(ji)), {});
  s->w = int(jsonNum(js, "width"));
  s->h = int(jsonNum(js, "height"));
  s->fx = jsonNum(js, "fx"); s->fy = jsonNum(js, "fy"); s->cx = jsonNum(js, "cx"); s->cy = jsonNum(js, "cy");
  std::ifstream fi(dir + "/frames.csv");
  std::string line;
  if (!fi || !std::getline(fi, line)) return false;
  const auto hdr = splitCsv(line);
  auto col = [&](const char* n) { return int(std::find(hdr.begin(), hdr.end(), n) - hdr.begin()); };
  const int cf = col("frame"), cs = col("stamp_ns"), cr = col("rgb"), cd = col("depth"), ci = col("instance");
  while (std::getline(fi, line)) {
    if (line.empty()) continue;
    const auto v = splitCsv(line);
    Frame f{};
    f.idx = std::atoi(v[size_t(cf)].c_str());
    f.stamp_ns = std::atoll(v[size_t(cs)].c_str());
    f.rgb = dir + "/" + v[size_t(cr)];
    f.depth = dir + "/" + v[size_t(cd)];
    if (ci < int(v.size())) f.inst = dir + "/" + v[size_t(ci)];
    s->frames.push_back(f);
  }
  std::ifstream pi(dir + "/poses.txt");
  size_t k = 0;
  while (std::getline(pi, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (k >= s->frames.size()) return false;
    double ts;
    Frame& f = s->frames[k++];
    if (std::sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", &ts, &f.t[0], &f.t[1], &f.t[2], &f.q[0], &f.q[1], &f.q[2], &f.q[3]) != 8)
      return false;
  }
  return k == s->frames.size() && s->w > 0;
}

// 쿼터니언(x y z w) → 행 우선 3×3
inline void quatToR(const double q[4], double R[9]) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w);     R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w);     R[7] = 2 * (y * z + x * w);     R[8] = 1 - 2 * (x * x + y * y);
}

// world ← 카메라 = (베이스 x, y, yaw) ∘ T_bc. yaw = 광축(카메라 z)의 수평 방향. 돌려주는 값: 다시 합친 것과 원래 것의 최대 차
inline double splitPose(const Frame& f, sm_pose2* base, double T_bc[12]) {
  double R[9];
  quatToR(f.q, R);
  const double yaw = std::atan2(R[5], R[2]);   // 광축 = R 의 셋째 열
  base->x = f.t[0];
  base->y = f.t[1];
  base->yaw = yaw;
  const double c = std::cos(yaw), s = std::sin(yaw);
  for (int k = 0; k < 3; ++k) {   // Rz(−yaw)·R
    T_bc[0 * 4 + k] = c * R[0 * 3 + k] + s * R[1 * 3 + k];
    T_bc[1 * 4 + k] = -s * R[0 * 3 + k] + c * R[1 * 3 + k];
    T_bc[2 * 4 + k] = R[2 * 3 + k];
  }
  T_bc[3] = 0; T_bc[7] = 0; T_bc[11] = f.t[2];
  double err = 0;   // 확인: Rz(yaw)·T_bc + (x, y, 0) == world ← 카메라
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 4; ++k) {
      const double a = r == 0 ? c * T_bc[k] - s * T_bc[4 + k] : r == 1 ? s * T_bc[k] + c * T_bc[4 + k] : T_bc[8 + k];
      const double b = k < 3 ? R[r * 3 + k] : f.t[r] - (r == 0 ? base->x : r == 1 ? base->y : 0);
      err = std::max(err, std::fabs(a - b));
    }
  return err;
}

// ---------------- numpy .npz(zip 저장 방식 + NPY 1.0) — map_timeline.cpp 와 같은 형식 ----------------
class Npz {
 public:
  explicit Npz(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {}
  ~Npz() { close(); }
  void add(const std::string& key, const std::vector<float>& xyz) {
    if (!f_ || n_ >= 65535) return;
    const size_t n = xyz.size() / 3;
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(n) + ", 3), }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    std::string npy("\x93NUMPY\x01\x00", 8);
    npy += char(h.size() & 0xff);
    npy += char(h.size() >> 8);
    npy += h;
    npy.append(reinterpret_cast<const char*>(xyz.data()), xyz.size() * 4);
    const std::string name = key + ".npy";
    const uint32_t crc = uint32_t(crc32(0, reinterpret_cast<const Bytef*>(npy.data()), uInt(npy.size())));
    const uint32_t off = uint32_t(pos_);
    std::string lh;
    put32(lh, 0x04034b50); put16(lh, 20); put16(lh, 0); put16(lh, 0); put16(lh, 0); put16(lh, 0x21);
    put32(lh, crc); put32(lh, uint32_t(npy.size())); put32(lh, uint32_t(npy.size())); put16(lh, uint16_t(name.size())); put16(lh, 0);
    lh += name;
    write(lh);
    write(npy);
    std::string ch;
    put32(ch, 0x02014b50); put16(ch, 20); put16(ch, 20); put16(ch, 0); put16(ch, 0); put16(ch, 0); put16(ch, 0x21);
    put32(ch, crc); put32(ch, uint32_t(npy.size())); put32(ch, uint32_t(npy.size())); put16(ch, uint16_t(name.size()));
    put16(ch, 0); put16(ch, 0); put16(ch, 0); put16(ch, 0); put32(ch, 0); put32(ch, off);
    ch += name;
    cd_ += ch;
    ++n_;
  }
  int count() const { return n_; }
  void close() {
    if (!f_) return;
    const uint32_t cd_off = uint32_t(pos_);
    write(cd_);
    std::string e;
    put32(e, 0x06054b50); put16(e, 0); put16(e, 0); put16(e, uint16_t(n_)); put16(e, uint16_t(n_));
    put32(e, uint32_t(cd_.size())); put32(e, cd_off); put16(e, 0);
    write(e);
    std::fclose(f_);
    f_ = nullptr;
  }

 private:
  static void put16(std::string& o, uint16_t v) { o += char(v & 0xff); o += char(v >> 8); }
  static void put32(std::string& o, uint32_t v) { put16(o, uint16_t(v & 0xffff)); put16(o, uint16_t(v >> 16)); }
  void write(const std::string& b) {
    std::fwrite(b.data(), 1, b.size(), f_);
    pos_ += b.size();
  }
  FILE* f_;
  size_t pos_ = 0;
  std::string cd_;
  int n_ = 0;
};

// ---------------- 실행기: 프레임마다 자세·proprio·영상(+검출) → 지도가 바뀌면 map_timeline 줄 ----------------
class Runner {
 public:
  // labels: 프롬프트 표(검출 cls 번호). static_names: 고정 종류 표를 바꿀 때(빈 것 = scenemap 기본 표)
  Runner(const Seq& s, const std::string& out_dir, const std::vector<std::string>& labels, const std::vector<std::string>* static_names)
      : seq_(s) {
    c_ = sm_create("{\"robot\": \"r1pro\"}");   // R1 매개변수(스캔 높이 띠 0.10–1.80 m — 사람 키 카메라에 맞음)
    sm_set_pose_mode(c_, SM_POSE_GT);
    std::vector<const char*> lab;
    for (auto& x : labels) lab.push_back(x.c_str());
    sm_set_labels(c_, lab.data(), int(lab.size()));
    if (static_names) {
      std::vector<const char*> v;
      for (auto& x : *static_names) v.push_back(x.c_str());
      sm_set_kind_names(c_, SM_KIND_STATIC, v.data(), int(v.size()));
    }
    std::filesystem::create_directories(out_dir);
    o_ = std::fopen((std::filesystem::path(out_dir) / "map_timeline.csv").string().c_str(), "w");
    std::fprintf(o_, "frame,obj_id,x,y,z,label,moving\n");
    npz_ = std::make_unique<Npz>((std::filesystem::path(out_dir) / "map_points.npz").string());
    prop_.assign(SM_R1PRO_PROPRIO_DIM, 0.f);
    for (int k : {19, 44}) prop_[size_t(k)] = -100.f;   // 팔 끝 z(베이스 기준): 멀리
    for (int k : {24, 25, 49, 50}) prop_[size_t(k)] = 0.05f;   // 손가락 합 0.1 > 0.09: 열림
  }
  ~Runner() {
    if (o_) std::fclose(o_);
    if (npz_) npz_->close();
    sm_destroy(c_);
  }
  sm_ctx* ctx() { return c_; }

  // 프레임 하나: dets == NULL 이면 지도(격자)만
  void step(const Frame& f, const std::vector<float>& depth_m, const uint8_t* rgba, const sm_detections* dets) {
    const double t = f.stamp_ns * 1e-9;
    sm_pose2 base{};
    double T_bc[12];
    max_split_err_ = std::max(max_split_err_, splitPose(f, &base, T_bc));
    base.stamp = t;
    sm_push_pose(c_, &base);
    sm_proprio p{t, prop_.data(), int(prop_.size())};
    sm_push_proprio(c_, &p);
    sm_set_cam_extrinsic(c_, 0, T_bc);
    sm_image im{t, 0, seq_.w, seq_.h, rgba, depth_m.data(), seq_.fx, seq_.fy, seq_.cx, seq_.cy};
    sm_push_image(c_, &im, dets);
    writeMap(f.idx);
  }
  double maxSplitErr() const { return max_split_err_; }
  int snapshots() const { return n_snap_; }
  int npzEntries() const { return npz_->count(); }
  // 지금 지도 물체 수(사라짐 뺌)
  int liveObjects() const { return live_; }

 private:
  void writeMap(int frame) {
    sm_snapshot_t* s = nullptr;
    sm_snapshot(c_, &s);
    const sm_object* ob = nullptr;
    const int m = sm_snap_objects(s, &ob);
    std::string cur, body;
    char line[512];
    live_ = 0;
    std::map<uint32_t, std::array<double, 3>> cur_pos;
    for (int k = 0; k < m; ++k) {
      if (ob[k].state == SM_GONE) continue;
      ++live_;
      std::string lab = ob[k].name ? ob[k].name : "";
      std::replace(lab.begin(), lab.end(), ',', ' ');
      // moving: 든 것, 또는 옮겨짐 상태이고 지난 프레임보다 중심이 kMoveStep 넘게 바뀐 것(scenemap 은 움직이는 동안 관측 자리로
      // 바로 따라가고 쉬는 물체는 평균이라 프레임마다 거의 안 바뀜). 지도 출력만으로 정함
      auto pit = prev_pos_.find(ob[k].id);
      const bool stepped = pit != prev_pos_.end() && std::hypot(pit->second[0] - ob[k].pos[0], pit->second[1] - ob[k].pos[1]) > kMoveStep;
      // 두 프레임 잇달아 옮겨 가야(옮겨짐 잇기·병합의 한 번 뜀은 움직임이 아님)
      const bool stepped2 = stepped && pit->second[2] > 0;
      const int moving = ob[k].state == SM_HELD || (ob[k].state == SM_MOVED && stepped2);
      cur_pos[ob[k].id] = {ob[k].pos[0], ob[k].pos[1], stepped ? 1.0 : 0.0};
      const int n = std::snprintf(line, sizeof(line), "%u,%.3f,%.3f,%.3f,%s,%d\n", ob[k].id, ob[k].pos[0], ob[k].pos[1], ob[k].pos[2],
                                  lab.c_str(), moving);
      body.append(line, size_t(n));
      cur += std::to_string(frame) + "," + std::string(line, size_t(n));
    }
    prev_pos_.swap(cur_pos);
    if (cur.empty()) {
      cur = std::to_string(frame) + ",,,,,,\n";
      body = "empty";
    }
    const bool write_rows = body != prev_;   // 0차 유지: 바뀐 때만
    if (write_rows) {
      std::fputs(cur.c_str(), o_);
      prev_ = body;
      ++n_snap_;
      // 모양이 바뀐 물체의 점(그 프레임 중심 기준)
      for (int k = 0; k < m; ++k) {
        if (ob[k].state == SM_GONE) continue;
        sm_cloud cl{};
        if (sm_snap_points(s, ob[k].id, &cl) != 1 || cl.n <= 0) continue;
        Shape& sh = shape_[ob[k].id];
        const bool moved = sh.any && (std::fabs(sh.pos[0] - ob[k].pos[0]) + std::fabs(sh.pos[1] - ob[k].pos[1]) +
                                      std::fabs(sh.pos[2] - ob[k].pos[2])) > 0.05;   // 중심이 5 cm 넘게 바뀌면(옮겨짐 포함) 새 키
        const bool changed = !sh.any || moved ||
                             (cl.version != sh.version && (std::abs(cl.n - sh.n) * 10 > sh.n || frame - sh.frame >= 100));
        if (!changed) continue;
        std::vector<float> rel;
        rel.reserve(3 * size_t(cl.n));
        for (int q = 0; q < cl.n; ++q) {
          const float r[3] = {float(cl.origin[0] + cl.pts[q].x - ob[k].pos[0]), float(cl.origin[1] + cl.pts[q].y - ob[k].pos[1]),
                              float(cl.origin[2] + cl.pts[q].z - ob[k].pos[2])};
          if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] > 4.9f * 4.9f) continue;   // 채점기는 중심에서 5 m 넘는 점이 있으면 그 키를 버림
          rel.insert(rel.end(), r, r + 3);
        }
        if (rel.empty()) continue;
        npz_->add(std::to_string(ob[k].id) + "@" + std::to_string(frame), rel);
        sh = Shape{cl.version, cl.n, frame, true, {ob[k].pos[0], ob[k].pos[1], ob[k].pos[2]}};
      }
    }
    sm_snapshot_release(s);
  }

  struct Shape {
    uint32_t version = 0;
    int n = 0;
    int frame = 0;
    bool any = false;
    double pos[3] = {0, 0, 0};
  };
  const Seq& seq_;
  sm_ctx* c_ = nullptr;
  FILE* o_ = nullptr;
  std::unique_ptr<Npz> npz_;
  std::vector<float> prop_;
  std::string prev_ = "\x01";
  std::map<uint32_t, Shape> shape_;
  std::map<uint32_t, std::array<double, 3>> prev_pos_;   // 지난 프레임 중심(x, y), 그때 옮겨 갔나(1/0)
  static constexpr double kMoveStep = 0.03;              // m/프레임(10 Hz 에서 0.3 m/s)
  int n_snap_ = 0, live_ = 0;
  double max_split_err_ = 0;
};

inline void depthToM(const std::vector<uint16_t>& mm, std::vector<float>* m) {
  m->resize(mm.size());
  for (size_t i = 0; i < mm.size(); ++i) (*m)[i] = mm[i] * 1e-3f;
}

}  // namespace dom
