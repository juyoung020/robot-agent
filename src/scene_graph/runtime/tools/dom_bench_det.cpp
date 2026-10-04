// dynamic-object-mapping-benchmark 시퀀스를 실제 검출로 scenemap 에 넣어 <pred_root>/<seq>/map_timeline.csv · map_points.npz 를 쓴다.
// 읽는 것은 방법이 읽어도 되는 것만(rgb·depth·intrinsics·poses·frames). 넣는 방법은 ../scenemap/tools/dom_seq.hpp,
// 완벽한 검출판은 ../scenemap/tools/dom_bench.cpp.
//
// 검출: ovdet(FastSAM-s 416 TensorRT, 이름 없음 'object', sgrt 와 같은 conf 0.25). 이름(--classify):
//   검출마다 sgclip(SigLIP 2 B/32 영상 탑, 마스크 풀링) 768-d → 아래 kPrompts 글 임베딩(라벨 표 objects-v1 의 글 벡터)과 코사인
//   최대인 줄의 이름을 cls 로. 이름은 벤치마크 범주 이름이거나 구조물(wall·floor …) — scenemap 이 구조물을 물체로 만들지 않고,
//   같은 이름끼리만 같은 물체로 잇는다. --classify 없이는 sgrt 와 같다: 모든 검출 cls 0 'object'(위치만으로 잇기).
//
//   dom_bench_det <seq dir> <pred root> [--engine plan] [--conf 0.25] [--every 1] [--frames N]
//                 [--classify] [--clip plan] [--labels dir] [--dump dets.gz | --load dets.gz]
//     --dump  검출·이름(이름 표 포함)을 프레임마다 gzip 파일로 남김. --load 는 그것을 읽어 GPU 없이 scenemap 만 다시 돌림
//             (objmap 규칙 고치기·매개변수 비교용 — 검출은 같은 것)
#include <zlib.h>

#include <chrono>

#include "dom_seq.hpp"
#include "ovdet.h"
#include "sgclip.h"

namespace {
// 글 프롬프트 → 이름(벤치마크 범주 또는 구조물). 프롬프트는 라벨 표에 있는 영어 이름(sgc_labels_find 로 확인, 없으면 빼고 알림)
struct Prompt { const char* text; const char* label; };
const Prompt kPrompts[] = {
    {"wall", "wall"}, {"floor", "floor"}, {"ceiling", "ceiling"}, {"door", "door"}, {"window", "window"}, {"pillar", "pillar"},
    {"column", "pillar"}, {"baseboard", "baseboard"}, {"partition", "partition"}, {"glass wall", "wall"},
    {"armchair", "armchair"}, {"binder", "binder"}, {"ring binder", "binder"}, {"book", "book"}, {"bottle", "bottle"}, {"box", "box"},
    {"cardboard box", "box"}, {"briefcase", "briefcase"}, {"chair", "chair"}, {"office chair", "chair"}, {"computer", "computer"},
    {"desktop computer", "computer"}, {"cup", "cup"}, {"mug", "cup"}, {"fire extinguisher", "extinguisher"}, {"food", "food"},
    {"keyboard", "keyboard"}, {"lamp", "lamp"}, {"desk lamp", "lamp"}, {"laptop", "laptop"}, {"microphone", "microphone"},
    {"monitor", "monitor"}, {"computer monitor", "monitor"}, {"mouse", "mouse"}, {"computer mouse", "mouse"},
    {"pencil case", "pencil_case"}, {"phone", "phone"}, {"telephone", "phone"}, {"plant", "plant"}, {"potted plant", "plant"},
    {"printer", "printer"}, {"sculpture", "sculpture"}, {"statue", "sculpture"}, {"sofa", "sofa"}, {"couch", "sofa"},
    {"table", "table"}, {"trash can", "trash_can"}, {"wastebasket", "trash_can"}, {"vase", "vase"}, {"water cooler", "water_cooler"},
    {"appliance", "appliance"}, {"microwave", "appliance"}, {"refrigerator", "appliance"}, {"counter", "counter"},
    {"desk", "desk"}, {"fixture", "fixture"}, {"light fixture", "fixture"}, {"paper", "flat_item"}, {"folder", "flat_item"},
    {"planter", "planter"}, {"flower pot", "planter"}, {"sink", "sanitary"}, {"toilet", "sanitary"}, {"pen", "stationery"},
    {"stapler", "stationery"}, {"cabinet", "storage"}, {"shelf", "storage"}, {"bookcase", "storage"}, {"drawer", "storage"},
    {"clock", "wall_item"}, {"picture frame", "wall_item"}, {"poster", "wall_item"}, {"whiteboard", "wall_item"},
};
}  // namespace

static constexpr int32_t kDumpMagic = 0x44424431;   // 'DBD1'
static void putI(gzFile z, int32_t v) { gzwrite(z, &v, 4); }
static bool getI(gzFile z, int32_t* v) { return gzread(z, v, 4) == 4; }
static size_t maskWords(const sm_detections& D) { return (size_t(D.mask_w) * D.mask_h + 31) / 32; }

// 프레임 하나: fi n img_w img_h | cls[n] score[n] box[4n] mask_w mask_h sx sy ox oy bits[n·words]
static void writeDets(gzFile z, int fi, const sm_detections& D) {
  putI(z, fi); putI(z, D.n); putI(z, D.img_w); putI(z, D.img_h);
  if (D.n <= 0) return;
  gzwrite(z, D.cls, unsigned(4 * D.n));
  std::vector<float> sc(size_t(D.n), 1.f);
  if (D.score) std::copy(D.score, D.score + D.n, sc.begin());
  gzwrite(z, sc.data(), unsigned(4 * D.n));
  gzwrite(z, D.box, unsigned(16 * D.n));
  putI(z, D.mask_w); putI(z, D.mask_h);
  const float ms[4] = {D.mask_sx, D.mask_sy, D.mask_ox, D.mask_oy};
  gzwrite(z, ms, 16);
  gzwrite(z, D.mask_bits, unsigned(4 * maskWords(D) * size_t(D.n)));
}

static int replayDump(const dom::Seq& seq, const char* pred_root, const std::string& path, int max_frames) {
  gzFile z = gzopen(path.c_str(), "rb");
  int32_t magic = 0, nl = 0;
  if (!z || !getI(z, &magic) || magic != kDumpMagic || !getI(z, &nl)) { std::fprintf(stderr, "bad dump %s\n", path.c_str()); return 1; }
  std::vector<std::string> labels(static_cast<size_t>(nl));
  for (auto& l : labels) {
    int32_t k = 0;
    getI(z, &k);
    l.resize(size_t(k));
    gzread(z, l.data(), unsigned(k));
  }
  dom::Runner run(seq, (std::filesystem::path(pred_root) / seq.name).string(), labels, nullptr);
  const int W = seq.w, H = seq.h;
  std::vector<uint16_t> dmm;
  std::vector<float> dm, score, box;
  std::vector<int32_t> cls;
  std::vector<uint32_t> bits;
  const auto t0 = std::chrono::steady_clock::now();
  int fi = 0, n_det = 0, n_kf = 0;
  const int nf = std::min<int>(int(seq.frames.size()), max_frames);
  int32_t rec = -1, n = 0, iw = 0, ih = 0;
  bool have = getI(z, &rec);
  for (; fi < nf; ++fi) {
    const dom::Frame& f = seq.frames[size_t(fi)];
    if (!dom::readU16(f.depth, W, H, &dmm)) { std::fprintf(stderr, "depth %s\n", f.depth.c_str()); return 1; }
    dom::depthToM(dmm, &dm);
    if (!have || rec != fi) { run.step(f, dm, nullptr, nullptr); continue; }
    getI(z, &n); getI(z, &iw); getI(z, &ih);
    sm_detections D{};
    D.stamp = f.stamp_ns * 1e-9; D.cam = 0; D.img_w = iw; D.img_h = ih; D.n = n;
    if (n > 0) {
      cls.resize(size_t(n)); score.resize(size_t(n)); box.resize(size_t(4 * n));
      gzread(z, cls.data(), unsigned(4 * n)); gzread(z, score.data(), unsigned(4 * n)); gzread(z, box.data(), unsigned(16 * n));
      getI(z, &D.mask_w); getI(z, &D.mask_h);
      float ms[4];
      gzread(z, ms, 16);
      D.mask_sx = ms[0]; D.mask_sy = ms[1]; D.mask_ox = ms[2]; D.mask_oy = ms[3];
      bits.resize(maskWords(D) * size_t(n));
      gzread(z, bits.data(), unsigned(4 * bits.size()));
      D.cls = cls.data(); D.score = score.data(); D.box = box.data(); D.mask_bits = bits.data();
    }
    run.step(f, dm, nullptr, &D);
    n_det += n;
    ++n_kf;
    have = getI(z, &rec);
  }
  gzclose(z);
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s (replay %s): frames %d, keyframes %d, detections %d, snapshots %d, npz %d, final objects %d, %.1f s\n", seq.name.c_str(),
              path.c_str(), nf, n_kf, n_det, run.snapshots(), run.npzEntries(), run.liveObjects(), sec);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: dom_bench_det <seq dir> <pred root> [--engine plan] [--conf 0.25] [--every K] [--frames N] [--classify] "
                 "[--clip plan] [--labels dir] [--dump dets.gz | --load dets.gz]\n");
    return 2;
  }
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : ".";
  std::string engine = home + "/ovdet_models/x86_sm120/FastSAM-s-416.plan";
  std::string clip_plan = home + "/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan";
  std::string labels_dir = home + "/embed_work/labels/objects-v1";
  float conf = 0.25f;
  int every = 1, max_frames = 1 << 30;
  bool classify = false;
  std::string dump_path, load_path;
  for (int i = 3; i < argc; ++i) {
    auto next = [&]() { return std::string(argv[++i]); };
    if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) engine = next();
    else if (!std::strcmp(argv[i], "--conf") && i + 1 < argc) conf = std::stof(next());
    else if (!std::strcmp(argv[i], "--every") && i + 1 < argc) every = std::max(1, std::stoi(next()));
    else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = std::stoi(next());
    else if (!std::strcmp(argv[i], "--classify")) classify = true;
    else if (!std::strcmp(argv[i], "--clip") && i + 1 < argc) clip_plan = next();
    else if (!std::strcmp(argv[i], "--labels") && i + 1 < argc) labels_dir = next();
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = next();
    else if (!std::strcmp(argv[i], "--load") && i + 1 < argc) load_path = next();
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  dom::Seq seq;
  if (!dom::loadSeq(argv[1], &seq)) { std::fprintf(stderr, "cannot read sequence %s\n", argv[1]); return 1; }
  if (!load_path.empty()) return replayDump(seq, argv[2], load_path, max_frames);
  char err[1024] = {0};
  OvdConfig oc;
  ovd_default_config(&oc);
  const std::string names = engine + ".names.txt";
  oc.seg_engine = engine.c_str();
  oc.names = names.c_str();
  oc.conf_th = conf;
  OvdHandle* det = ovd_create(&oc, err, sizeof err);
  if (!det) { std::fprintf(stderr, "ovd_create: %s\n", err); return 1; }
  ovd_set_prompt(det, nullptr, 0, nullptr, 0);   // 엔진 어휘 전부(FastSAM: 'object' 하나)

  // 이름 표
  std::vector<std::string> labels;      // scenemap 프롬프트 표
  std::vector<int> prompt_cls;          // 글 프롬프트 줄 → labels 번호
  std::vector<float> text;              // 글 임베딩(행마다 SGC_DIM, L2 정규화)
  sgc_encoder* enc = nullptr;
  sgc_labels* lt = nullptr;
  if (classify) {
    sgc_config cc;
    sgc_default_config(&cc);
    cc.engine = clip_plan.c_str();
    enc = sgc_create(&cc, err, sizeof err);
    if (!enc) { std::fprintf(stderr, "sgc_create: %s\n", err); return 1; }
    lt = sgc_labels_open(labels_dir.c_str(), nullptr, err, sizeof err);
    if (!lt) { std::fprintf(stderr, "sgc_labels_open: %s\n", err); return 1; }
    std::map<std::string, int> lab_id;
    std::vector<float> e(SGC_DIM);
    for (const Prompt& p : kPrompts) {
      const int row = sgc_labels_find(lt, p.text);
      if (row < 0 || sgc_labels_text_emb(lt, row, e.data()) != 0) { std::fprintf(stderr, "prompt '%s' not in label table, skipped\n", p.text); continue; }
      double n = 0;
      for (float x : e) n += double(x) * x;
      n = std::sqrt(std::max(n, 1e-20));
      for (float& x : e) x = float(x / n);
      auto it = lab_id.find(p.label);
      if (it == lab_id.end()) { it = lab_id.emplace(p.label, int(labels.size())).first; labels.push_back(p.label); }
      prompt_cls.push_back(it->second);
      text.insert(text.end(), e.begin(), e.end());
    }
    std::fprintf(stderr, "classify: %zu prompts -> %zu labels (table %s)\n", prompt_cls.size(), labels.size(), sgc_labels_sha(lt));
  } else {
    labels.push_back("object");
  }

  dom::Runner run(seq, (std::filesystem::path(argv[2]) / seq.name).string(), labels, nullptr);
  gzFile dz = nullptr;
  if (!dump_path.empty()) {
    dz = gzopen(dump_path.c_str(), "wb1");
    if (!dz) { std::fprintf(stderr, "cannot write %s\n", dump_path.c_str()); return 1; }
    putI(dz, kDumpMagic);
    putI(dz, int32_t(labels.size()));
    for (auto& l : labels) { putI(dz, int32_t(l.size())); gzwrite(dz, l.data(), unsigned(l.size())); }
  }
  const int W = seq.w, H = seq.h;
  std::vector<uint16_t> dmm;
  std::vector<float> dm;
  std::vector<uint8_t> rgb;
  std::vector<int32_t> cls;
  std::vector<sgc_item> items;
  std::vector<sgc_result> res(64);
  std::map<int, int> label_hist;
  const auto t0 = std::chrono::steady_clock::now();
  double det_ms = 0, clip_ms = 0;
  int n_det = 0, n_kf = 0;
  const int nf = std::min<int>(int(seq.frames.size()), max_frames);
  for (int fi = 0; fi < nf; ++fi) {
    const dom::Frame& f = seq.frames[size_t(fi)];
    if (!dom::readU16(f.depth, W, H, &dmm)) { std::fprintf(stderr, "depth %s\n", f.depth.c_str()); return 1; }
    dom::depthToM(dmm, &dm);
    if (fi % every) { run.step(f, dm, nullptr, nullptr); continue; }
    if (!dom::readRgb(f.rgb, W, H, &rgb)) { std::fprintf(stderr, "rgb %s\n", f.rgb.c_str()); return 1; }
    const auto ta = std::chrono::steady_clock::now();
    OvdImage im{};
    im.stamp = f.stamp_ns * 1e-9; im.cam = 0; im.data = rgb.data(); im.h = H; im.w = W; im.row_stride = int64_t(W) * 3; im.pix_stride = 3;
    const sm_detections* d = ovd_detect(det, &im, nullptr);
    if (!d) { std::fprintf(stderr, "ovd_detect: %s\n", ovd_last_error(det)); return 1; }
    sm_detections D = *d;
    const auto tb = std::chrono::steady_clock::now();
    det_ms += std::chrono::duration<double, std::milli>(tb - ta).count();
    cls.assign(size_t(D.n), 0);
    if (classify && D.n > 0) {
      sgc_frame fr{};
      fr.rgb = rgb.data(); fr.on_device = 0; fr.row_stride = int64_t(W) * 3; fr.pix_stride = 3; fr.w = W; fr.h = H;
      fr.mask_w = D.mask_w; fr.mask_h = D.mask_h; fr.mask_sx = D.mask_sx; fr.mask_sy = D.mask_sy; fr.mask_ox = D.mask_ox;
      fr.mask_oy = D.mask_oy; fr.mask_bits = D.mask_bits;
      items.resize(size_t(D.n));
      for (int k = 0; k < D.n; ++k) {
        items[size_t(k)] = sgc_item{uint32_t(k), k, {D.box[4 * k], D.box[4 * k + 1], D.box[4 * k + 2], D.box[4 * k + 3]}, 1.f};
      }
      int sent = 0, got = 0;
      while (got < D.n) {
        if (sent < D.n) {
          const int s = sgc_submit(enc, im.stamp, &fr, items.data() + sent, D.n - sent);
          if (s < 0) { std::fprintf(stderr, "sgc_submit failed\n"); return 1; }
          sent += s;
        }
        const int r = sgc_poll(enc, res.data(), int(res.size()), 1);
        for (int q = 0; q < r; ++q) {
          const float* e = res[size_t(q)].emb;
          int best = 0;
          float bs = -2.f;
          for (size_t p = 0; p < prompt_cls.size(); ++p) {
            float sdot = 0.f;
            const float* tp = text.data() + p * SGC_DIM;
            for (int j = 0; j < SGC_DIM; ++j) sdot += e[j] * tp[j];
            if (sdot > bs) { bs = sdot; best = int(p); }
          }
          cls[res[size_t(q)].id] = prompt_cls[size_t(best)];
          ++label_hist[prompt_cls[size_t(best)]];
        }
        got += r;
      }
      clip_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
    }
    D.cls = cls.data();
    if (dz) writeDets(dz, fi, D);
    run.step(f, dm, nullptr, &D);
    n_det += D.n;
    ++n_kf;
    if (fi % 500 == 0)
      std::fprintf(stderr, "[%s] frame %d/%d dets %d objects %d\n", seq.name.c_str(), fi, nf, D.n, run.liveObjects());
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: frames %d, keyframes %d, detections %d (%.1f/kf), snapshots %d, npz %d, final objects %d, det %.1f ms/kf, clip %.1f ms/kf, %.1f s\n",
              seq.name.c_str(), nf, n_kf, n_det, n_kf ? double(n_det) / n_kf : 0.0, run.snapshots(), run.npzEntries(), run.liveObjects(),
              n_kf ? det_ms / n_kf : 0.0, n_kf ? clip_ms / n_kf : 0.0, sec);
  if (classify) {
    std::printf("  labels:");
    for (auto& [c, n] : label_hist) std::printf(" %s %d", labels[size_t(c)].c_str(), n);
    std::printf("\n");
  }
  if (dz) gzclose(dz);
  if (enc) sgc_destroy(enc);
  if (lt) sgc_labels_close(lt);
  ovd_destroy(det);
  return 0;
}
