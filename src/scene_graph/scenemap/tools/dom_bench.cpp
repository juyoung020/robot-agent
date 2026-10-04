// dynamic-object-mapping-benchmark 시퀀스를 scenemap 에 '완벽한 검출'(정답 인스턴스 마스크 + objects.csv 범주)로 넣어
// <pred_root>/<seq>/map_timeline.csv · map_points.npz 를 쓴다. 업데이터(objmap) 논리만 따로 채점하는 상한 기준.
// 실제 검출(FastSAM + SigLIP 2)판은 ../runtime/tools/dom_bench_det.cpp. 넣는 방법은 dom_seq.hpp 머리말.
//
//   dom_bench <seq dir> <pred root> [--min-px 100] [--every 1] [--frames N] [--bench-static]
//     --min-px      인스턴스 화소가 이보다 적은 물체는 검출에서 뺌
//     --every       k 프레임마다 검출(사이 프레임은 깊이로 격자만)
//     --bench-static 고정 종류 표를 scenemap 기본 대신 벤치마크 범주표의 movable = 0 범주로(scenarios/office_classes.json 에서
//                   가져온 이름 목록 — 아래 kBenchStatic)
#include <chrono>

#include "dom_seq.hpp"

namespace {
// scenarios/office_classes.json 의 movable = 0 범주(10-04 기준)
const std::vector<std::string> kBenchStatic = {"appliance", "counter",  "desk",       "fixture", "flat_item",
                                               "planter",   "sanitary", "stationery", "storage", "wall_item"};
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: dom_bench <seq dir> <pred root> [--min-px N] [--every K] [--frames N] [--bench-static]\n");
    return 2;
  }
  int min_px = 100, every = 1, max_frames = 1 << 30;
  bool bench_static = false;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--min-px") && i + 1 < argc) min_px = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--every") && i + 1 < argc) every = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bench-static")) bench_static = true;
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  dom::Seq seq;
  if (!dom::loadSeq(argv[1], &seq)) { std::fprintf(stderr, "cannot read sequence %s\n", argv[1]); return 1; }
  // 정답 범주(완벽한 검출만 읽음): instance id → 범주 번호
  std::ifstream oi(seq.dir + "/objects.csv");
  std::string line;
  std::getline(oi, line);
  std::map<int, std::string> id_cls;
  std::map<std::string, int> cat;
  while (std::getline(oi, line)) {
    const auto v = dom::splitCsv(line);
    if (v.size() < 2) continue;
    id_cls[std::atoi(v[0].c_str())] = v[1];
    cat.emplace(v[1], 0);
  }
  std::vector<std::string> labels;
  for (auto& kv : cat) { kv.second = int(labels.size()); labels.push_back(kv.first); }
  std::vector<int> cls_of(65536, -1);
  for (auto& [id, c] : id_cls) cls_of[size_t(id)] = cat[c];

  dom::Runner run(seq, (std::filesystem::path(argv[2]) / seq.name).string(), labels, bench_static ? &kBenchStatic : nullptr);
  const int W = seq.w, H = seq.h;
  const size_t words = (size_t(W) * H + 31) / 32;
  std::vector<uint16_t> dmm, inst;
  std::vector<float> dm;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
  const auto t0 = std::chrono::steady_clock::now();
  int n_det = 0, n_kf = 0;
  const int nf = std::min<int>(int(seq.frames.size()), max_frames);
  for (int fi = 0; fi < nf; ++fi) {
    const dom::Frame& f = seq.frames[size_t(fi)];
    if (!dom::readU16(f.depth, W, H, &dmm)) { std::fprintf(stderr, "depth %s\n", f.depth.c_str()); return 1; }
    dom::depthToM(dmm, &dm);
    if (fi % every) { run.step(f, dm, nullptr, nullptr); continue; }
    if (!dom::readU16(f.inst, W, H, &inst)) { std::fprintf(stderr, "instance %s\n", f.inst.c_str()); return 1; }
    // 인스턴스마다 마스크 한 장(화소 수 min_px 이상)
    std::map<int, std::vector<int>> px;
    for (int k = 0; k < W * H; ++k)
      if (inst[size_t(k)] && cls_of[inst[size_t(k)]] >= 0) px[inst[size_t(k)]].push_back(k);
    cls.clear(); score.clear(); box.clear(); bits.clear();
    for (auto& [id, v] : px) {
      if (int(v.size()) < min_px) continue;
      const size_t b0 = bits.size();
      bits.resize(b0 + words, 0);
      int x0 = W, y0 = H, x1 = 0, y1 = 0;
      for (int k : v) {
        bits[b0 + size_t(k >> 5)] |= 1u << (k & 31);
        const int x = k % W, y = k / W;
        x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
      }
      cls.push_back(cls_of[size_t(id)]);
      score.push_back(1.f);
      box.insert(box.end(), {float(x0), float(y0), float(x1 + 1), float(y1 + 1)});
    }
    sm_detections D{};
    D.stamp = f.stamp_ns * 1e-9; D.cam = 0; D.img_w = W; D.img_h = H; D.n = int(cls.size());
    D.cls = cls.data(); D.score = score.data(); D.box = box.data();
    D.mask_w = W; D.mask_h = H; D.mask_sx = 1; D.mask_sy = 1; D.mask_ox = 0; D.mask_oy = 0; D.mask_bits = bits.data();
    run.step(f, dm, nullptr, &D);
    n_det += D.n;
    ++n_kf;
    if (fi % 500 == 0)
      std::fprintf(stderr, "[%s] frame %d/%d dets %d objects %d\n", seq.name.c_str(), fi, nf, D.n, run.liveObjects());
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: frames %d, keyframes %d, detections %d, snapshots %d, npz %d, final objects %d, pose split err %.2e, %.1f s\n",
              seq.name.c_str(), nf, n_kf, n_det, run.snapshots(), run.npzEntries(), run.liveObjects(), run.maxSplitErr(), sec);
  return 0;
}
