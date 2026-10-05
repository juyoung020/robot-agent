// carto_run — 스트림 폴더(tools/realbag bag2stream 형식 + --scan-only 의 scans.bin) → slam_carto → 궤적·ATE·지도.
//   carto_run <stream dir> <out dir> [--config openloris_hokuyo.lua] [--no-odom] [--every 1] [--res 0.05] [--max-t S]
// 바퀴 오도메트리(odom.csv)와 스캔(scans.bin)을 시각 순으로 넣고, 영상 프레임 시각(frames.csv)마다 sc_pose_at 으로 자세를 묻는다
// (= 실시간에 keyframe 이 받는 자세). 끝에 sc_finish(마지막 전역 최적화) 뒤 노드 궤적도 따로 잰다.
// 나오는 것: <out>/traj.csv(frame, t, 추정 x y yaw, 오도메트리 x y yaw, 정답 x y yaw), <out>/nodes.csv(최적화 노드),
//            <out>/carto_map.pgm·yaml(Cartographer 확률 격자), <out>/metrics.json(ATE: 첫 프레임 맞춤 = 실시간 떠밀림, SE(2) 맞춤).
// 정답·ATE 는 realbag_run 과 같은 식(베이스 자세 + 카메라 xy = 베이스 ∘ T_bc 평행 이동).
#include <slam_carto.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
bool jsonArr(const std::string& js, const char* key, double* v, int n) {
  const size_t k = js.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) return false;
  const char* q = js.c_str() + js.find('[', k) + 1;
  for (int i = 0; i < n; ++i) {
    char* e;
    v[i] = std::strtod(q, &e);
    if (e == q) return false;
    q = e;
    while (*q == ',' || *q == ' ' || *q == '\n') ++q;
  }
  return true;
}
double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

struct Scan { double t, a0, da, dt, rmin, rmax; std::vector<float> r; };
struct Odo { double t, x, y, yaw; };
struct Frame { int idx; double t; int gt_ok; double gx, gy, gyaw; };

struct Se2 { double c = 1, s = 0, tx = 0, ty = 0; };
Se2 alignUmeyama(const std::vector<std::array<double, 2>>& a, const std::vector<std::array<double, 2>>& b) {
  Se2 T;
  const size_t n = a.size();
  if (n < 2) return T;
  double ma[2] = {0, 0}, mb[2] = {0, 0};
  for (size_t i = 0; i < n; ++i) { ma[0] += a[i][0]; ma[1] += a[i][1]; mb[0] += b[i][0]; mb[1] += b[i][1]; }
  for (double& v : ma) v /= double(n);
  for (double& v : mb) v /= double(n);
  double sxx = 0, sxy = 0;
  for (size_t i = 0; i < n; ++i) {
    const double ax = a[i][0] - ma[0], ay = a[i][1] - ma[1], bx = b[i][0] - mb[0], by = b[i][1] - mb[1];
    sxx += ax * bx + ay * by;
    sxy += ax * by - ay * bx;
  }
  const double th = std::atan2(sxy, sxx);
  T.c = std::cos(th); T.s = std::sin(th);
  T.tx = mb[0] - (T.c * ma[0] - T.s * ma[1]);
  T.ty = mb[1] - (T.s * ma[0] + T.c * ma[1]);
  return T;
}
Se2 alignFirst(double ax, double ay, double ayaw, double bx, double by, double byaw) {
  Se2 T;
  const double th = wrap(byaw - ayaw);
  T.c = std::cos(th); T.s = std::sin(th);
  T.tx = bx - (T.c * ax - T.s * ay);
  T.ty = by - (T.s * ax + T.c * ay);
  return T;
}
struct Ate { double rms = 0, max = 0, final = 0, yaw_rms = 0, yaw_max = 0; int n = 0; };
Ate ate(const std::vector<std::array<double, 3>>& a, const std::vector<std::array<double, 3>>& b, const Se2& T) {
  Ate r;
  double s2 = 0, y2 = 0;
  const double th = std::atan2(T.s, T.c);
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = T.c * a[i][0] - T.s * a[i][1] + T.tx, y = T.s * a[i][0] + T.c * a[i][1] + T.ty;
    const double e = std::hypot(x - b[i][0], y - b[i][1]);
    const double ey = std::fabs(wrap(a[i][2] + th - b[i][2]));
    s2 += e * e; y2 += ey * ey;
    r.max = std::max(r.max, e); r.yaw_max = std::max(r.yaw_max, ey);
    r.final = e;
  }
  r.n = int(a.size());
  if (r.n) { r.rms = std::sqrt(s2 / r.n); r.yaw_rms = std::sqrt(y2 / r.n); }
  return r;
}
std::string ateJ(const Ate& a) {
  char b[256];
  std::snprintf(b, sizeof b, "{\"n\": %d, \"rms\": %.4f, \"max\": %.4f, \"final\": %.4f, \"yaw_rms_deg\": %.3f, \"yaw_max_deg\": %.3f}", a.n, a.rms, a.max,
                a.final, a.yaw_rms * 180 / M_PI, a.yaw_max * 180 / M_PI);
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: carto_run <stream dir> <out dir> [--config openloris_hokuyo.lua] [--no-odom] [--every 1] [--res 0.05] [--max-t S] [--scan-dt S]\n");
    return 2;
  }
  const std::string dir = argv[1], out = argv[2];
  std::string config;
  bool use_odom = true;
  int every = 1;
  double res = 0.05, max_t = 1e18, scan_dt = std::nan("");   // 스캔 시각 보정(초). 없으면 meta.json scan_dt(bag2stream 이 정답 없이 잼)
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--config") config = nx(); else if (a == "--no-odom") use_odom = false; else if (a == "--every") every = std::max(1, std::stoi(nx()));
    else if (a == "--res") res = std::stod(nx()); else if (a == "--scan-dt") scan_dt = std::stod(nx()); else if (a == "--max-t") max_t = std::stod(nx());
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  std::filesystem::create_directories(out);
  const std::string meta = slurp(dir + "/meta.json");
  double T_bl[12], T_bc[12];
  if (!jsonArr(meta, "T_bl", T_bl, 12)) { std::fprintf(stderr, "%s/meta.json: no T_bl (bag2stream.py --scan-only)\n", dir.c_str()); return 1; }
  if (!jsonArr(meta, "T_bc", T_bc, 12)) std::fill(T_bc, T_bc + 12, 0.0);
  if (std::isnan(scan_dt)) {
    const size_t k = meta.find("\"scan_dt\"");
    scan_dt = k == std::string::npos ? 0.0 : std::atof(meta.c_str() + meta.find(':', k) + 1);
  }
  if (config.empty()) config = meta.find("\"openloris\"") != std::string::npos ? "openloris_hokuyo.lua" : "limo_x2l.lua";

  // 스캔
  std::vector<Scan> scans;
  {
    std::ifstream f(dir + "/scans.bin", std::ios::binary);
    char mg[4];
    if (!f.read(mg, 4) || std::memcmp(mg, "SCN1", 4)) { std::fprintf(stderr, "%s/scans.bin: bad file\n", dir.c_str()); return 1; }
    while (true) {
      Scan s;
      int32_t n;
      if (!f.read(reinterpret_cast<char*>(&s.t), 8) || !f.read(reinterpret_cast<char*>(&n), 4)) break;
      double h[5];
      f.read(reinterpret_cast<char*>(h), 40);
      s.a0 = h[0]; s.da = h[1]; s.dt = h[2]; s.rmin = h[3]; s.rmax = h[4];
      s.r.resize(size_t(n));
      f.read(reinterpret_cast<char*>(s.r.data()), std::streamsize(4 * n));
      if (!f) break;
      s.t += scan_dt;
      scans.push_back(std::move(s));
    }
  }
  std::vector<Odo> odom;
  {
    std::ifstream f(dir + "/odom.csv");
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      Odo o;
      if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &o.t, &o.x, &o.y, &o.yaw) == 4) odom.push_back(o);
    }
  }
  std::vector<Frame> frames;
  {
    std::ifstream f(dir + "/frames.csv");
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      Frame fr{};
      char rgb[256], dep[256];
      // frame,stamp,rgb,depth,gt_ok,gt_x,gt_y,gt_yaw,...
      std::replace(line.begin(), line.end(), ',', ' ');
      if (std::sscanf(line.c_str(), "%d %lf %255s %255s %d %lf %lf %lf", &fr.idx, &fr.t, rgb, dep, &fr.gt_ok, &fr.gx, &fr.gy, &fr.gyaw) == 8) frames.push_back(fr);
    }
  }
  if (scans.empty() || frames.empty()) { std::fprintf(stderr, "no scans or frames\n"); return 1; }
  const double t0 = frames.front().t;
  std::fprintf(stderr, "[carto_run] %s: scans %zu odom %zu frames %zu config %s odom %s scan_dt %+.2f\n", dir.c_str(), scans.size(), odom.size(), frames.size(),
               config.c_str(), use_odom ? "on" : "off", scan_dt);

  sc_config cfg = sc_default_config();
  cfg.config_name = config.c_str();
  cfg.laser_xyz[0] = T_bl[3]; cfg.laser_xyz[1] = T_bl[7]; cfg.laser_xyz[2] = T_bl[11];
  cfg.laser_yaw = std::atan2(T_bl[4], T_bl[0]);
  cfg.use_odom = use_odom ? 1 : 0;
  char err[256] = {0};
  sc_ctx* c = sc_create(&cfg, err, sizeof err);
  if (!c) { std::fprintf(stderr, "%s\n", err); return 1; }

  FILE* ft = std::fopen((out + "/traj.csv").c_str(), "w");
  std::fprintf(ft, "frame,t,est_x,est_y,est_yaw,odo_x,odo_y,odo_yaw,gt_ok,gt_x,gt_y,gt_yaw\n");
  size_t si = 0, oi = 0;
  // 오도메트리는 첫 프레임 0.5 s 앞부터
  while (oi < odom.size() && odom[oi].t < t0 - 0.5) ++oi;
  while (si < scans.size() && scans[si].t < t0 - 0.5) ++si;
  auto feedUntil = [&](double t) {
    while (true) {
      const double ts = si < scans.size() ? scans[si].t : 1e300, to = oi < odom.size() ? odom[oi].t : 1e300;
      if (si >= scans.size() && oi >= odom.size()) break;
      if (std::min(ts, to) > t) break;
      if (to <= ts) { const Odo& o = odom[oi++]; sc_push_odom(c, o.t - t0, o.x, o.y, o.yaw); }
      else {
        const Scan& s = scans[si++];
        sc_push_scan(c, s.t - t0, int32_t(s.r.size()), s.r.data(), s.a0, s.da, s.dt, s.rmin, s.rmax);
      }
    }
  };
  std::vector<std::array<double, 3>> est, gt, odo;
  std::vector<double> est_t;
  std::vector<std::array<double, 2>> est_c, gt_c, odo_c;
  const auto w0 = std::chrono::steady_clock::now();
  auto camxy = [&](const std::array<double, 3>& p) {
    const double cc = std::cos(p[2]), s = std::sin(p[2]);
    return std::array<double, 2>{p[0] + cc * T_bc[3] - s * T_bc[7], p[1] + s * T_bc[3] + cc * T_bc[7]};
  };
  auto odoAt = [&](double t, std::array<double, 3>* o) {
    if (odom.empty()) return false;
    auto it = std::lower_bound(odom.begin(), odom.end(), t, [](const Odo& a, double tt) { return a.t < tt; });
    if (it == odom.begin()) { *o = {it->x, it->y, it->yaw}; return true; }
    if (it == odom.end()) { *o = {odom.back().x, odom.back().y, odom.back().yaw}; return true; }
    const Odo& b = *it;
    const Odo& a = *(it - 1);
    const double w = (t - a.t) / std::max(1e-9, b.t - a.t);
    *o = {a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.yaw + wrap(b.yaw - a.yaw) * w};
    return true;
  };
  int n_nopose = 0;
  for (size_t fi = 0; fi < frames.size(); fi += size_t(every)) {
    const Frame& f = frames[fi];
    if (f.t - t0 > max_t) break;
    feedUntil(f.t);
    double p[3];
    if (sc_pose_at(c, f.t - t0, p)) { ++n_nopose; continue; }
    std::array<double, 3> o{0, 0, 0};
    odoAt(f.t, &o);
    std::fprintf(ft, "%d,%.4f,%.5f,%.5f,%.6f,%.5f,%.5f,%.6f,%d,%.5f,%.5f,%.6f\n", f.idx, f.t - t0, p[0], p[1], p[2], o[0], o[1], o[2], f.gt_ok, f.gx, f.gy, f.gyaw);
    if (f.gt_ok) {
      est.push_back({p[0], p[1], p[2]}); gt.push_back({f.gx, f.gy, f.gyaw}); odo.push_back(o);
      est_t.push_back(f.t - t0);
      est_c.push_back(camxy(est.back())); gt_c.push_back(camxy(gt.back())); odo_c.push_back(camxy(o));
    }
  }
  std::fclose(ft);
  feedUntil(1e300);
  const double wall_online = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  sc_stats st{};
  sc_get_stats(c, &st);
  sc_finish(c);
  sc_stats st2{};
  sc_get_stats(c, &st2);
  // 최적화 노드
  const int nn = sc_nodes(c, nullptr, nullptr, 0);
  std::vector<double> nt(size_t(std::max(nn, 0))), nxy(size_t(std::max(nn, 0)) * 3);
  sc_nodes(c, nt.data(), nxy.data(), nn);
  {
    FILE* fn = std::fopen((out + "/nodes.csv").c_str(), "w");
    std::fprintf(fn, "t,x,y,yaw\n");
    for (int i = 0; i < nn; ++i) std::fprintf(fn, "%.4f,%.5f,%.5f,%.6f\n", nt[size_t(i)], nxy[3 * size_t(i)], nxy[3 * size_t(i) + 1], nxy[3 * size_t(i) + 2]);
    std::fclose(fn);
  }
  // 최적화 궤적의 프레임 자세: 최적화 노드 사이 보간(노드 밖은 끝 노드 + 오도메트리 차)
  std::vector<std::array<double, 3>> opt;
  for (size_t k = 0; k < est_t.size(); ++k) {
    const double t = est_t[k];
    size_t j = size_t(std::lower_bound(nt.begin(), nt.end(), t) - nt.begin());
    std::array<double, 3> p{est[k]};
    if (nn > 0) {
      if (j == 0 || j >= size_t(nn)) {
        const size_t q = j == 0 ? 0 : size_t(nn) - 1;
        std::array<double, 3> oq, ot;
        odoAt(nt[q] + t0, &oq); odoAt(t + t0, &ot);
        const double c0 = std::cos(oq[2]), s0 = std::sin(oq[2]);
        const double dx = c0 * (ot[0] - oq[0]) + s0 * (ot[1] - oq[1]), dy = -s0 * (ot[0] - oq[0]) + c0 * (ot[1] - oq[1]);
        const double* b = &nxy[3 * q];
        p = {b[0] + std::cos(b[2]) * dx - std::sin(b[2]) * dy, b[1] + std::sin(b[2]) * dx + std::cos(b[2]) * dy, b[2] + wrap(ot[2] - oq[2])};
      } else {
        const double* a = &nxy[3 * (j - 1)];
        const double* b = &nxy[3 * j];
        const double w = (t - nt[j - 1]) / std::max(1e-9, nt[j] - nt[j - 1]);
        p = {a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + wrap(b[2] - a[2]) * w};
      }
    }
    opt.push_back(p);
  }
  sc_write_map(c, (out + "/carto_map.pgm").c_str(), res);
  sc_destroy(c);

  // ATE: 베이스 자세(첫 프레임 맞춤 / SE(2)), 카메라 xy(SE(2) — realbag_run 의 ate_se2_cam 과 같은 식)
  auto first = [&](const std::vector<std::array<double, 3>>& a) {
    return a.empty() ? Se2{} : alignFirst(a[0][0], a[0][1], a[0][2], gt[0][0], gt[0][1], gt[0][2]);
  };
  auto xy = [](const std::vector<std::array<double, 3>>& a) {
    std::vector<std::array<double, 2>> r;
    for (auto& p : a) r.push_back({p[0], p[1]});
    return r;
  };
  std::vector<std::array<double, 3>> ec3, gc3, oc3;
  for (size_t i = 0; i < est_c.size(); ++i) { ec3.push_back({est_c[i][0], est_c[i][1], est[i][2]}); gc3.push_back({gt_c[i][0], gt_c[i][1], gt[i][2]}); oc3.push_back({odo_c[i][0], odo_c[i][1], odo[i][2]}); }
  const Ate a_first = ate(est, gt, first(est)), a_se2 = ate(est, gt, alignUmeyama(xy(est), xy(gt)));
  const Ate o_first = ate(odo, gt, first(odo)), o_se2 = ate(odo, gt, alignUmeyama(xy(odo), xy(gt)));
  const Ate p_first = ate(opt, gt, first(opt)), p_se2 = ate(opt, gt, alignUmeyama(xy(opt), xy(gt)));
  const Ate c_se2 = ate(ec3, gc3, alignUmeyama(est_c, gt_c)), oc_se2 = ate(oc3, gc3, alignUmeyama(odo_c, gt_c));
  double len = 0, turn = 0;
  for (size_t i = 1; i < gt.size(); ++i) { len += std::hypot(gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1]); turn += std::fabs(wrap(gt[i][2] - gt[i - 1][2])); }
  FILE* fm = std::fopen((out + "/metrics.json").c_str(), "w");
  std::fprintf(fm,
               "{\n \"stream\": \"%s\", \"config\": \"%s\", \"odom\": %s, \"frames\": %zu, \"no_pose\": %d, \"gt_len_m\": %.3f, \"gt_turn_deg\": %.1f,\n"
               " \"ate_first_base\": %s,\n \"ate_se2_base\": %s,\n \"ate_se2_cam\": %s,\n"
               " \"opt_ate_first_base\": %s,\n \"opt_ate_se2_base\": %s,\n"
               " \"odom_ate_first_base\": %s,\n \"odom_ate_se2_base\": %s,\n \"odom_ate_se2_cam\": %s,\n"
               " \"carto\": {\"scans\": %lld, \"local\": %lld, \"nodes\": %lld, \"submaps\": %lld, \"loop_constraints_online\": %lld, \"loop_constraints_final\": %lld,"
               " \"us_scan_mean\": %.1f, \"us_scan_max\": %.1f, \"wall_s\": %.2f}\n}\n",
               dir.c_str(), config.c_str(), use_odom ? "true" : "false", frames.size(), n_nopose, len, turn * 180 / M_PI, ateJ(a_first).c_str(),
               ateJ(a_se2).c_str(), ateJ(c_se2).c_str(), ateJ(p_first).c_str(), ateJ(p_se2).c_str(), ateJ(o_first).c_str(), ateJ(o_se2).c_str(),
               ateJ(oc_se2).c_str(), (long long)st.n_scans, (long long)st.n_local, (long long)st.n_nodes, (long long)st2.n_submaps,
               (long long)st.n_loop_constraints, (long long)st2.n_loop_constraints, st.us_scan_mean, st.us_scan_max, wall_online);
  std::fclose(fm);
  std::fprintf(stderr, "[carto_run] path %.2f m turn %.0f deg | online ATE first %.3f m (max %.3f, yaw %.2f deg) se2 %.3f | cam se2 %.3f | opt se2 %.3f | odom se2 %.3f "
                       "| nodes %lld submaps %lld loops %lld | %.0f us/scan | wall %.1f s\n",
               len, turn * 180 / M_PI, a_first.rms, a_first.max, a_first.yaw_rms * 180 / M_PI, a_se2.rms, c_se2.rms, p_se2.rms, o_se2.rms,
               (long long)st.n_nodes, (long long)st2.n_submaps, (long long)st2.n_loop_constraints, st.us_scan_mean, wall_online);
  return 0;
}
