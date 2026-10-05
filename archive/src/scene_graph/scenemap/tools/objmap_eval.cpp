// objmap 채점 재생: ep_<ep>.bin(proprio·slam2d keyframe) + ep_<ep>_det.bin(정답 라벨 = '완벽한 검출')으로
// slam2d 자세 위에 물체 지도를 만들고 결과(물체 표·사건)를 쓴다. 채점은 eval/score_objmap.py.
//
//   objmap_eval <ep.bin> <det.bin> <out prefix> [--gt-pose] [--min-cells N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "scenemap/fk.hpp"
#include "scenemap/objmap.hpp"
#include "scenemap/slam2d.hpp"

using namespace scenemap;

#pragma pack(push, 1)
struct Row {
  float qvel[3], eefL[3], eefR[3], gl, gr;
  double gt[3];
  float prop[61];
};
#pragma pack(pop)

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: objmap_eval <ep.bin> <det.bin> <out prefix> [--gt-pose] [--min-cells N]\n");
    return 2;
  }
  bool gt_pose = false;
  int min_cells = 8;
  ObjParams op;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--gt-pose")) gt_pose = true;
    else if (!std::strcmp(argv[i], "--min-cells") && i + 1 < argc) min_cells = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--min-points") && i + 1 < argc) op.min_points = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gone") && i + 1 < argc) op.gone_misses = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  // ep.bin
  FILE* f = std::fopen(argv[1], "rb");
  char magic[4];
  uint32_t hdr[5];
  float K[4];
  if (!f || std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5 || std::fread(K, 4, 4, f) != 4 || hdr[0] != 2) {
    std::fprintf(stderr, "ep.bin\n");
    return 1;
  }
  const uint32_t n = hdr[1], w = hdr[3], h = hdr[4];
  std::vector<Row> rows(n);
  if (std::fread(rows.data(), sizeof(Row), n, f) != n) return 1;
  uint32_t m;
  if (std::fread(&m, 4, 1, f) != 1) return 1;
  std::vector<uint16_t> depth(size_t(w) * h);
  float Tbc[12];
  uint32_t next = 0, done = 0;
  auto readKf = [&]() -> bool {
    if (done >= m) return false;
    if (std::fread(&next, 4, 1, f) != 1 || std::fread(Tbc, 4, 12, f) != 12 || std::fread(depth.data(), 2, depth.size(), f) != depth.size())
      return false;
    ++done;
    return true;
  };
  // det.bin
  FILE* g = std::fopen(argv[2], "rb");
  uint32_t dh[3];
  float DK[4];
  if (!g || std::fread(magic, 1, 4, g) != 4 || std::memcmp(magic, "SMDT", 4) || std::fread(dh, 4, 3, g) != 3 ||
      std::fread(DK, 4, 4, g) != 4) {
    std::fprintf(stderr, "det.bin\n");
    return 1;
  }
  const int DW = int(dh[1]), DH = int(dh[2]);
  uint32_t n_obj;
  if (std::fread(&n_obj, 4, 1, g) != 1) return 1;
  std::vector<std::string> oname(n_obj), ocat(n_obj);
  std::vector<uint8_t> ostruct(n_obj);
  auto rstr = [&](std::string* s) {
    uint16_t L;
    if (std::fread(&L, 2, 1, g) != 1) return false;
    s->resize(L);
    return L == 0 || std::fread(&(*s)[0], 1, L, g) == L;
  };
  for (uint32_t k = 0; k < n_obj; ++k)
    if (!rstr(&oname[k]) || !rstr(&ocat[k]) || std::fread(&ostruct[k], 1, 1, g) != 1) return 1;
  std::map<std::string, int> cat_id;
  for (uint32_t k = 0; k < n_obj; ++k)
    if (!ostruct[k]) cat_id.emplace(ocat[k], 0);
  std::vector<std::string> cats;
  for (auto& kv : cat_id) { kv.second = int(cats.size()); cats.push_back(kv.first); }
  uint32_t dm;
  if (std::fread(&dm, 4, 1, g) != 1) return 1;
  std::vector<uint16_t> ddepth(size_t(DW) * DH), dlab(size_t(DW) * DH);
  float dT[12];
  uint32_t dnext = 0, ddone = 0;
  auto readDet = [&]() -> bool {
    if (ddone >= dm) return false;
    if (std::fread(&dnext, 4, 1, g) != 1 || std::fread(dT, 4, 12, g) != 12 || std::fread(ddepth.data(), 2, ddepth.size(), g) != ddepth.size() ||
        std::fread(dlab.data(), 2, dlab.size(), g) != dlab.size())
      return false;
    ++ddone;
    return true;
  };

  SlamParams sp;
  Slam2D slam(sp);
  ObjectMap om(op);
  bool have = readKf(), dhave = readDet();
  const size_t words = (size_t(DW) * DH + 31) / 32;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
  std::vector<int> gt_of;   // 검출 → 정답 물체 번호(채점용)
  FILE* dl = std::fopen((std::string(argv[3]) + "_dets.csv").c_str(), "w");
  std::fprintf(dl, "t,gt,cls,cells\n");
  for (uint32_t i = 0; i < n; ++i) {
    if (i > 0) slam.pushVelocity(rows[i].qvel[0], rows[i].qvel[1], rows[i].qvel[2], 1.0 / 30.0);
    BodyFk fk;
    const float eef[2][3] = {{rows[i].eefL[0], rows[i].eefL[1], rows[i].eefL[2]}, {rows[i].eefR[0], rows[i].eefR[1], rows[i].eefR[2]}};
    if (have && next == i) {
      computeBodyFk(rows[i].prop, &fk);
      DepthView dv;
      dv.w = int(w); dv.h = int(h); dv.mm = depth.data(); dv.fx = K[0]; dv.fy = K[1]; dv.cx = K[2]; dv.cy = K[3];
      std::memcpy(dv.T_bc, fk.T_head, sizeof(dv.T_bc));
      const Pose2 gp{rows[i].gt[0], rows[i].gt[1], rows[i].gt[2]};
      slam.keyframe(dv, bodyFromFk(fk, eef), nullptr);
      (void)gp;
      have = readKf();
    }
    const Pose2 pose = gt_pose ? Pose2{rows[i].gt[0], rows[i].gt[1], rows[i].gt[2]} : slam.pose();
    const double c = std::cos(pose.th), s = std::sin(pose.th);
    double eefm[2][3];
    for (int k = 0; k < 2; ++k) {
      eefm[k][0] = pose.x + c * eef[k][0] - s * eef[k][1];
      eefm[k][1] = pose.y + s * eef[k][0] + c * eef[k][1];
      eefm[k][2] = eef[k][2];
    }
    const float grip[2] = {rows[i].gl, rows[i].gr};
    const double t = i / 30.0;
    if (dhave && dnext == i) {
      // 라벨 → 검출(물체마다 마스크 한 장)
      std::map<int, std::vector<int>> cells;
      for (int k = 0; k < DW * DH; ++k)
        if (dlab[k] != 0xFFFF && dlab[k] < n_obj && !ostruct[dlab[k]]) cells[dlab[k]].push_back(k);
      cls.clear(); score.clear(); box.clear(); bits.clear(); gt_of.clear();
      for (auto& [o, v] : cells) {
        if (int(v.size()) < min_cells) continue;
        const size_t base = bits.size();
        bits.resize(base + words, 0);
        int x0 = DW, y0 = DH, x1 = 0, y1 = 0;
        for (int k : v) {
          bits[base + (k >> 5)] |= 1u << (k & 31);
          const int x = k % DW, y = k / DW;
          x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
        }
        cls.push_back(cat_id[ocat[o]]);
        score.push_back(1.f);
        box.insert(box.end(), {float(x0), float(y0), float(x1 + 1), float(y1 + 1)});
        gt_of.push_back(o);
        std::fprintf(dl, "%.3f,%d,%d,%zu\n", t, o, cat_id[ocat[o]], v.size());
      }
      sm_detections D{};
      D.stamp = t; D.cam = 0; D.img_w = DW; D.img_h = DH; D.n = int(cls.size());
      D.cls = cls.data(); D.score = score.data(); D.box = box.data();
      D.mask_w = DW; D.mask_h = DH; D.mask_sx = 1; D.mask_sy = 1; D.mask_ox = 0; D.mask_oy = 0; D.mask_bits = bits.data();
      ObjFrame F;
      F.stamp = t; F.w = DW; F.h = DH; F.depth_mm = ddepth.data();
      F.fx = DK[0]; F.fy = DK[1]; F.cx = DK[2]; F.cy = DK[3];
      // map ← 카메라 = 베이스 자세 ∘ T_bc(이 프레임 순기구학 — det.bin 의 T 는 데이터 robot2cam, 같은 값)
      computeBodyFk(rows[i].prop, &fk);
      const float* B = fk.T_head;
      for (int r = 0; r < 3; ++r) {
        const double R0 = r == 0 ? c : (r == 1 ? s : 0), R1 = r == 0 ? -s : (r == 1 ? c : 0), R2 = r == 2 ? 1 : 0;
        for (int k = 0; k < 4; ++k) F.T_mc[r * 4 + k] = R0 * B[k] + R1 * B[4 + k] + R2 * B[8 + k];
      }
      F.T_mc[3] += pose.x; F.T_mc[7] += pose.y;
      F.dets = &D;
      std::memcpy(F.eef, eefm, sizeof(eefm));
      F.grip[0] = grip[0]; F.grip[1] = grip[1];
      F.base_yaw = pose.th;
      om.update(F);
      dhave = readDet();
    } else {
      om.updateHands(t, eefm, grip, pose.th);
    }
  }
  std::fclose(dl);
  FILE* o = std::fopen((std::string(argv[3]) + "_objs.csv").c_str(), "w");
  std::fprintf(o, "id,cls,category,x,y,z,ex,ey,ez,fx,fy,fz,n_obs,first_seen,last_seen,state,confirmed,moved\n");
  for (const auto& q : om.objects())
    std::fprintf(o, "%u,%d,%s,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%u,%.2f,%.2f,%d,%d,%d\n", q.id, q.cls, cats[q.cls].c_str(),
                 q.pos[0], q.pos[1], q.pos[2], q.ext[0], q.ext[1], q.ext[2], q.first_pos[0], q.first_pos[1], q.first_pos[2], q.n_obs,
                 q.first_seen, q.last_seen, q.state, int(q.confirmed), int(q.moved));
  std::fclose(o);
  FILE* e = std::fopen((std::string(argv[3]) + "_events.csv").c_str(), "w");
  std::fprintf(e, "t,id,kind,x,y,z\n");
  for (const auto& v : om.events()) std::fprintf(e, "%.3f,%u,%d,%.4f,%.4f,%.4f\n", v.t, v.id, v.kind, v.pos[0], v.pos[1], v.pos[2]);
  std::fclose(e);
  int conf = 0;
  for (const auto& q : om.objects()) conf += q.confirmed;
  std::printf("objects %zu (confirmed %d), events %zu, categories %zu\n", om.objects().size(), conf, om.events().size(), cats.size());
  return 0;
}
