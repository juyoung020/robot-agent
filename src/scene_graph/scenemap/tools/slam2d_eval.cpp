// slam2d 채점 입력 재생: eval/export_episode.py 가 만든 ep_*.bin 을 읽어 slam2d 를 돌리고,
// 프레임마다 추정 자세(f64 x, y, yaw)와 keyframe 통계를 쓴다. 채점은 eval/score_slam.py.
//
//   slam2d_eval <ep.bin> <out prefix> [--method A|B] [--carto-prior] [--pgm]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "scenemap/fk.hpp"
#include "scenemap/slam2d.hpp"

using namespace scenemap;

#pragma pack(push, 1)
struct Row {   // numpy 구조 dtype(채움 없음)과 같은 배치
  float qvel[3], eefL[3], eefR[3], gl, gr;
  double gt[3];
  float prop[61];
};
#pragma pack(pop)
static_assert(sizeof(Row) == 68 + 61 * 4, "row layout");

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: slam2d_eval <ep.bin> <out prefix> [--method A|B] [--carto-prior] [--pgm]\n");
    return 2;
  }
  SlamParams p;
  bool pgm = false, gt_pose = false, fk_body = true, fk_cam = true;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--method") && i + 1 < argc) p.method = argv[++i][0];
    else if (!std::strcmp(argv[i], "--carto-prior")) p.a_carto_prior = true;
    else if (!std::strcmp(argv[i], "--pgm")) pgm = true;
    else if (!std::strcmp(argv[i], "--no-fk-body")) fk_body = false;   // 몸 가리기: 어깨 대략값–팔 끝(옛 규칙)
    else if (!std::strcmp(argv[i], "--no-fk-cam")) fk_cam = false;     // 카메라 자세: 데이터의 robot2cam
    else if (!std::strcmp(argv[i], "--no-attach")) p.attach.enabled = false;
    else if (!std::strcmp(argv[i], "--no-mf")) p.motion_filter = false;
    else if (!std::strcmp(argv[i], "--no-still")) p.stationary_rule = false;
    else if (!std::strcmp(argv[i], "--pca-normals")) p.cell_normals = false;
    else if (!std::strcmp(argv[i], "--no-deadband")) p.deadband = false;
    else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) p.grid.res = float(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "--match-cell") && i + 1 < argc) p.scan.match_cell = float(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "--gate-yaw") && i + 1 < argc) p.gate_yaw = std::atof(argv[++i]) * M_PI / 180.0;
    else if (!std::strcmp(argv[i], "--gate-xy") && i + 1 < argc) p.gate_xy = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--prior-yaw-k") && i + 1 < argc) p.prior_yaw_k = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--prior-xy-k") && i + 1 < argc) p.prior_xy_k = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--cell-normals-only")) p.cell_normals_only = true;
    else if (!std::strcmp(argv[i], "--zmax") && i + 1 < argc) p.scan.zmax = float(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "--laser")) p.scan.dense = false;   // 레이저 한 줄로만 맞춤
    else if (!std::strcmp(argv[i], "--gt-pose")) gt_pose = true;   // 진단: 정답 자세로 지도, jump = 정답에서 끌려간 양
    else if (!std::strcmp(argv[i], "--min-inliers") && i + 1 < argc) p.min_inliers = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--sigma-occ") && i + 1 < argc) p.a_sigma_occ = std::atof(argv[++i]);
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror(argv[1]); return 1; }
  char magic[4];
  uint32_t hdr[5];
  float K[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "SMEP", 4) || std::fread(hdr, 4, 5, f) != 5 ||
      std::fread(K, 4, 4, f) != 4 || hdr[0] != 2) {
    std::fprintf(stderr, "bad header (ver 2 필요: export_episode.py 다시)\n");
    return 1;
  }
  const uint32_t n = hdr[1], w = hdr[3], h = hdr[4];
  std::vector<Row> rows(n);
  if (std::fread(rows.data(), sizeof(Row), n, f) != n) { std::fprintf(stderr, "short rows\n"); return 1; }
  uint32_t m;
  if (std::fread(&m, 4, 1, f) != 1) return 1;
  std::vector<uint16_t> depth(size_t(w) * h);
  DepthView dv;
  dv.w = int(w); dv.h = int(h); dv.step = 1; dv.mm = depth.data();
  dv.fx = K[0]; dv.fy = K[1]; dv.cx = K[2]; dv.cy = K[3];

  Slam2D slam(p);
  std::vector<double> est(size_t(n) * 3, 0.0);
  FILE* ks = std::fopen((std::string(argv[2]) + "_kf.csv").c_str(), "w");
  std::fprintf(ks, "frame,matched,accepted,hits,inliers,attached,still,inserted,jump_xy,jump_yaw,us_scan,us_match,us_insert\n");
  uint32_t next_frame = 0, done = 0;
  double max_fk_err = 0;   // 순기구학 T_bc 와 데이터 robot2cam 의 최대 성분 차
  auto readKf = [&]() -> bool {
    if (done >= m) return false;
    if (std::fread(&next_frame, 4, 1, f) != 1 || std::fread(dv.T_bc, 4, 12, f) != 12 ||
        std::fread(depth.data(), 2, depth.size(), f) != depth.size()) return false;
    ++done;
    return true;
  };
  bool have = readKf();
  for (uint32_t i = 0; i < n; ++i) {
    if (i > 0) slam.pushVelocity(rows[i].qvel[0], rows[i].qvel[1], rows[i].qvel[2], 1.0 / 30.0);
    if (have && next_frame == i) {
      float eef[2][3];
      std::memcpy(eef[0], rows[i].eefL, 12);
      std::memcpy(eef[1], rows[i].eefR, 12);
      BodyFk fk;
      computeBodyFk(rows[i].prop, &fk);
      BodyState b;
      if (fk_body) {
        b = bodyFromFk(fk, eef);
      } else {
        std::memcpy(b.eef, eef, sizeof(eef));
      }
      if (fk_cam) {
        double e = 0;
        for (int k = 0; k < 12; ++k) e = std::max(e, double(std::fabs(fk.T_head[k] - dv.T_bc[k])));
        max_fk_err = std::max(max_fk_err, e);
        std::memcpy(dv.T_bc, fk.T_head, sizeof(fk.T_head));
      }
      const Pose2 g{rows[i].gt[0], rows[i].gt[1], rows[i].gt[2]};
      const KeyframeStats st = slam.keyframe(dv, b, gt_pose ? &g : nullptr);
      std::fprintf(ks, "%u,%d,%d,%d,%d,%d,%d,%d,%.4f,%.5f,%.0f,%.0f,%.0f\n", i, st.matched, st.accepted, st.n_hits, st.inliers,
                   st.n_attached, st.still, st.inserted, st.jump_xy, st.jump_yaw, st.us_scan, st.us_match, st.us_insert);
      have = readKf();
    }
    const Pose2 q = slam.pose();
    est[i * 3] = q.x; est[i * 3 + 1] = q.y; est[i * 3 + 2] = q.th;
  }
  std::fclose(ks);
  if (fk_cam) std::printf("fk T_bc vs data robot2cam: max |diff| %.2e\n", max_fk_err);
  std::fclose(f);
  FILE* o = std::fopen((std::string(argv[2]) + "_est.bin").c_str(), "wb");
  std::fwrite(est.data(), 8, est.size(), o);
  std::fclose(o);
  if (pgm) {
    const OccGrid& g = slam.grid();
    const auto v = g.export8();
    FILE* pf = std::fopen((std::string(argv[2]) + "_map.pgm").c_str(), "wb");
    std::fprintf(pf, "P5\n%d %d\n255\n", g.width(), g.height());
    for (int y = g.height() - 1; y >= 0; --y)
      for (int x = 0; x < g.width(); ++x) {
        const int8_t c = v[size_t(y) * g.width() + x];
        const unsigned char px = c < 0 ? 205 : (unsigned char)(254 - c * 254 / 100);
        std::fputc(px, pf);
      }
    std::fclose(pf);
    std::printf("map %dx%d origin cell (%d,%d) res %.2f\n", g.width(), g.height(), g.x0(), g.y0(), g.res());
  }
  return 0;
}
