// realbag_run — 공개 실제 로봇 ROS bag(bag2stream.py 가 만든 스트림 폴더)을 우리 지각 파이프라인(ovdet 검출 + scenemap SLAM·물체 기억·
// 장면 그래프)으로 돌린다. scenemap·ovdet·sgclip 은 이 저장소(src/scene_graph)의 것을 같이 빌드해 C ABI 로만 쓴다.
//
//   realbag_run <stream dir>[,<stream dir>…] <out dir> [옵션]
//     --robot limo_omx|r1pro   scenemap 로봇 매개변수(기본 limo_omx — 우리 로봇의 SLAM·몸 크기 설정)
//     --pose carto|odom|gt     자세 원천(기본 carto = Cartographer(../../slam_carto): 2D 라이다 scans.bin + 바퀴 오도메트리(bag2stream.py
//                              --scan-only) → 영상마다 sm_push_ext_pose(SM_POSE_EXT). slam = carto(옛 이름 — 깊이 맞추기 slam2d 는 10-06 archive).
//                              스트림에 scans.bin 이 없으면 경고하고 odom. --carto-config <lua>(기본 openloris → openloris_hokuyo.lua),
//                              끝에 <out>/carto_map.pgm·metrics.json "carto". gt = 정답 베이스 자세(여러 판을
//                              한 지도에 이을 때 — 판 사이 재위치 추정이 없으므로). slam·odom 에서도 정답은 진단(sm_get_pose_diag)에만 넣는다
//     --det fastsam|yolo|none  검출(기본 fastsam = 이름 없는 분할 엔진 + SigLIP 2 이름(dom_bench_det --classify 와 같은 길). 분할 엔진 기본은
//                              ObjectSAM(YOLO26n 학생, yolo26n-seg-obj-416.plan — objprob_front.hpp kDefaultEngine), 원래 FastSAM-s 는
//                              --engine models/ovdet/x86_sm120/FastSAM-s-416.plan(버린 FastSAM-s 재학습은 보관 models/ovdet/archive/x86_sm120/FastSAM-s-416-obj.plan).
//     --det yolo               닫힌 어휘 YOLO 분할(기본 yolo26s-seg-416, COCO 80) — 엔진 어휘 전부
//     --namer siglip|engine    이름 붙이기(기본 fastsam = siglip, yolo = engine 클래스). siglip = 검출기 마스크마다 SigLIP 2 조각 임베딩을
//                              아래 낱말 글 임베딩과 맞춤(세 검출기를 같은 이름 표로 비교할 때)
//     --det-every K            K 프레임마다 검출(기본 3 — 스트림 15 Hz 면 5 Hz, sgrt kf_every 6 @ 30 Hz 와 같음). 나머지 프레임은 깊이로 지도만
//     --dump dets.gz | --load a.gz[,b.gz…]   검출·이름 캐시(--load 면 GPU 없이 scenemap 만, 여러 판이면 판마다 하나)
//     --live host:port [--rate R]       sgview(--ingest)로 실시간 스트림, R 배속으로 걸음 맞춤(기본 1)
//     --sg <run 폴더>          학습 뷰어(trainview) 재생 판: <run>/replays/ep_<n>_<이름>.sg(stream.sgs·memory·cam·meta.json)·run.json(group real_bags)
//     --ref-map <memory dir>   지도 비교 기준(같은 스트림을 --pose gt 로 돈 저장 폴더): 점유 칸 ±5/10 cm 정밀도·재현율
//     --snap-at t1,t2,…        그 시각(스트림 초)에 기억 저장(<out>/snap_<t>/ — 손 확인용)
//     --max-depth M            이보다 먼 깊이는 버림(기본 4 m — RealSense D435·Kinect 잡음, 시뮬은 8 m)
//     --frames N · --conf 0.25 · --engine plan · --clip plan · --labels dir · --gap S(판 사이 시각 틈, 기본 5)
//     --objprob | --no-objprob  확률 물체 모델(objprob)(scenemap README "scenemap 확률 모드") — siglip 이름 검출(또는 RBD2 캐시)의 조각 임베딩 + 통째 다시 담기.
//                              기본 켬(siglip 이름 길·RBD2 캐시일 때 — 엔진 클래스 이름·RBD1 캐시는 끔), --no-objprob = 옛 이름 규칙
//     --label-prior f.json     objprob 라벨 log 사전(objprob_fit.py label_prior.json). 진단: RB_REENC_DUMP=<dir> 이면 다시 담기 마스크 그림(앞 400 개)
//     --objprob-params f.json  objprob 엔진별 매개변수(obj_params = sm_set_obj_params 문자열, label_prior = 옆 파일). 없으면 objprob_params/<엔진>.json,
//                              "none" = 안 씀(내장 기본값). --label-prior 가 파일의 사전보다 이김
//     --inspect                살펴본 정도(scenemap README "살펴본 정도") — memory/view.json·scene.json 물체에 "inspect", metrics.json objmap_us
//
// 쓰는 것(<out>/): memory/(sm_save_dsg — sgview 가 읽음, 1 s 마다), traj.csv(프레임마다 추정·오도메트리·정답 카메라 xy), objects.csv(노드 전부, structural 열), walls.csv(scenemap 벽 선분),
//   events.csv(새 물체·사라짐·옮겨짐·지움, 판 번호), metrics.json(ATE·지도·물체 수), stdout 요약.
#include "ra_paths.h"
#include <zlib.h>

#include <chrono>
#include <set>
#include <thread>

#include "dom_seq.hpp"   // PNG 읽기(scenemap/tools)
#include "ovdet.h"
#include "rb_util.hpp"   // JSON·폴더·JPEG·스트림 받기
#include "sgclip.h"
#include "objprob_front.hpp"   // 낱말 표·기본 엔진·확률 모드 설정(libsgrt 와 같이 씀, runtime/src)

using rb::jnum;
using rb::jstr;
using rb::Obj;

namespace {

using objprob_front::ApLabel;
using objprob_front::kApLabels;
using objprob_front::kDefaultEngine;   // 기본 분할 엔진(ObjectSAM YOLO26n 학생) — objprob_front.hpp
using objprob_front::kVocab;
using objprob_front::kVocabAp;
using objprob_front::Word;
#ifndef RB_PARAMS_DIR
#define RB_PARAMS_DIR "objprob_params"
#endif
constexpr int32_t kDumpMagic = 0x52424431;   // 'RBD1'
constexpr int32_t kDumpMagic2 = 0x52424432;  // 'RBD2': RBD1 + 프레임마다 임베딩 dim(0 = 없음)·n × dim FP16
void putI(gzFile z, int32_t v) { gzwrite(z, &v, 4); }
bool getI(gzFile z, int32_t* v) { return gzread(z, v, 4) == 4; }
size_t maskWords(const sm_detections& D) { return (size_t(D.mask_w) * D.mask_h + 31) / 32; }

#ifdef RB_HAVE_CARTO
#include <slam_carto.h>
#endif

// ---------------- 스트림 폴더 ----------------
struct FrameRow {
  int idx;
  double stamp;
  std::string rgb, depth;
  int gt_ok;
  double gx, gy, gyaw, gcx, gcy;
};
struct Odo { double t, x, y, yaw, vx, vy, wz; };
struct Stream {
  std::string dir, name, kind;
  int w = 0, h = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double T_bc[12]{};
  std::vector<FrameRow> frames;
  std::vector<Odo> odom;
  double T_bl[12]{};            // base ← 2D 라이다(bag2stream --scan-only), 없으면 0
  bool have_bl = false;
  double scan_dt = 0;           // 스캔 시각 + scan_dt = 오도메트리·영상 시계
};
struct ScanRec { double t, a0, da, dt, rmin, rmax; std::vector<float> r; };
bool loadScans(const std::string& dir, double dt, std::vector<ScanRec>* out) {
  std::ifstream f(dir + "/scans.bin", std::ios::binary);
  char mg[4];
  if (!f.read(mg, 4) || std::memcmp(mg, "SCN1", 4)) return false;
  while (true) {
    ScanRec s;
    int32_t n;
    double h[5];
    if (!f.read(reinterpret_cast<char*>(&s.t), 8) || !f.read(reinterpret_cast<char*>(&n), 4) || !f.read(reinterpret_cast<char*>(h), 40)) break;
    s.a0 = h[0]; s.da = h[1]; s.dt = h[2]; s.rmin = h[3]; s.rmax = h[4];
    s.r.resize(size_t(n));
    if (!f.read(reinterpret_cast<char*>(s.r.data()), std::streamsize(4 * n))) break;
    s.t += dt;
    out->push_back(std::move(s));
  }
  return !out->empty();
}

std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

bool loadStream(const std::string& dir, Stream* s) {
  namespace fs = std::filesystem;
  s->dir = dir;
  s->name = fs::path(dir).filename().string();
  if (s->name.empty()) s->name = fs::path(dir).parent_path().filename().string();
  const std::string js = slurp(dir + "/meta.json");
  if (js.empty()) return false;
  s->w = int(dom::jsonNum(js, "width")); s->h = int(dom::jsonNum(js, "height"));
  s->fx = dom::jsonNum(js, "fx"); s->fy = dom::jsonNum(js, "fy"); s->cx = dom::jsonNum(js, "cx"); s->cy = dom::jsonNum(js, "cy");
  {
    const size_t k = js.find("\"kind\"");
    const size_t a = js.find('"', js.find(':', k) + 1), b = js.find('"', a + 1);
    s->kind = js.substr(a + 1, b - a - 1);
    const size_t p = js.find('[', js.find("\"T_bc\""));
    const char* q = js.c_str() + p + 1;
    for (double& v : s->T_bc) { char* e; v = std::strtod(q, &e); q = e; while (*q == ',' || *q == ' ' || *q == '\n') ++q; }
    const size_t kb = js.find("\"T_bl\"");
    if (kb != std::string::npos) {
      const char* r = js.c_str() + js.find('[', kb) + 1;
      for (double& v : s->T_bl) { char* e; v = std::strtod(r, &e); r = e; while (*r == ',' || *r == ' ' || *r == '\n') ++r; }
      s->have_bl = true;
    }
    const size_t kd = js.find("\"scan_dt\"");
    if (kd != std::string::npos) s->scan_dt = std::atof(js.c_str() + js.find(':', kd) + 1);
  }
  std::ifstream fi(dir + "/frames.csv");
  std::string line;
  std::getline(fi, line);
  while (std::getline(fi, line)) {
    const auto v = dom::splitCsv(line);
    if (v.size() < 10) continue;
    FrameRow f{};
    f.idx = std::atoi(v[0].c_str()); f.stamp = std::atof(v[1].c_str());
    f.rgb = dir + "/" + v[2]; f.depth = dir + "/" + v[3];
    f.gt_ok = std::atoi(v[4].c_str());
    f.gx = std::atof(v[5].c_str()); f.gy = std::atof(v[6].c_str()); f.gyaw = std::atof(v[7].c_str());
    f.gcx = std::atof(v[8].c_str()); f.gcy = std::atof(v[9].c_str());
    s->frames.push_back(f);
  }
  std::ifstream fo(dir + "/odom.csv");
  std::getline(fo, line);
  while (std::getline(fo, line)) {
    Odo o{};
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &o.t, &o.x, &o.y, &o.yaw, &o.vx, &o.vy, &o.wz) == 7) s->odom.push_back(o);
  }
  return s->w > 0 && !s->frames.empty();
}

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

// 정답 베이스 자세(map = 정답 world). OpenLORIS: 정답이 base_link. TUM: 정답이 카메라 → 광축 yaw − 카메라 yaw(베이스 기준), xy − R·t
sm_pose2 gtBase(const Stream& s, const FrameRow& f) {
  if (s.kind != "tum_pioneer") return sm_pose2{0, f.gx, f.gy, f.gyaw};
  const double yaw_c = std::atan2(s.T_bc[6], s.T_bc[2]);   // 광축(셋째 열) 수평 방향
  const double yaw = wrap(f.gyaw - yaw_c);
  const double c = std::cos(yaw), sn = std::sin(yaw);
  return sm_pose2{0, f.gcx - (c * s.T_bc[3] - sn * s.T_bc[7]), f.gcy - (sn * s.T_bc[3] + c * s.T_bc[7]), yaw};
}

// 카메라 xy(map) = 베이스 ∘ T_bc 평행 이동
void camXY(const Stream& s, double x, double y, double yaw, double* cx, double* cy) {
  const double c = std::cos(yaw), sn = std::sin(yaw);
  *cx = x + c * s.T_bc[3] - sn * s.T_bc[7];
  *cy = y + sn * s.T_bc[3] + c * s.T_bc[7];
}

// SE(2) 맞춤(Umeyama, 크기 없음): b ≈ R a + t
struct Se2 { double c = 1, s = 0, tx = 0, ty = 0; };
Se2 align(const std::vector<std::array<double, 2>>& a, const std::vector<std::array<double, 2>>& b) {
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
struct Ate { double rms = 0, max = 0, final = 0, mean = 0; int n = 0; };
Ate ate(const std::vector<std::array<double, 2>>& a, const std::vector<std::array<double, 2>>& b, const Se2& T) {
  Ate r;
  double s2 = 0, s1 = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = T.c * a[i][0] - T.s * a[i][1] + T.tx, y = T.s * a[i][0] + T.c * a[i][1] + T.ty;
    const double e = std::hypot(x - b[i][0], y - b[i][1]);
    s2 += e * e; s1 += e;
    r.max = std::max(r.max, e);
    r.final = e;
  }
  r.n = int(a.size());
  if (r.n) { r.rms = std::sqrt(s2 / r.n); r.mean = s1 / r.n; }
  return r;
}
// 첫 프레임 맞춤(scenemap pose_diag 와 같은 방식): a 의 첫 자세를 b 의 첫 자세에 겹침(yaw 포함)
Se2 alignFirst(double ax, double ay, double ayaw, double bx, double by, double byaw) {
  Se2 T;
  const double th = wrap(byaw - ayaw);
  T.c = std::cos(th); T.s = std::sin(th);
  T.tx = bx - (T.c * ax - T.s * ay);
  T.ty = by - (T.s * ax + T.c * ay);
  return T;
}

// PGM 지도(map_server 형식)
struct Pgm { int w = 0, h = 0; double res = 0.05, ox = 0, oy = 0; std::vector<uint8_t> v; };
bool loadPgm(const std::string& dir, Pgm* m) {
  std::ifstream f(dir + "/map.pgm", std::ios::binary);
  std::string magic;
  int maxv;
  if (!(f >> magic >> m->w >> m->h >> maxv) || magic != "P5") return false;
  f.get();
  m->v.resize(size_t(m->w) * m->h);
  f.read(reinterpret_cast<char*>(m->v.data()), std::streamsize(m->v.size()));
  const std::string y = slurp(dir + "/map.yaml");
  m->res = dom::jsonNum(y, "resolution");
  if (!(m->res > 0)) {   // yaml: "resolution: 0.05"
    const size_t p = y.find("resolution:");
    m->res = std::atof(y.c_str() + p + 11);
  }
  const size_t p = y.find('[', y.find("origin"));
  std::sscanf(y.c_str() + p, "[%lf, %lf", &m->ox, &m->oy);
  return true;
}

// ---------------- 검출 ----------------
struct Detector {
  std::string mode;   // fastsam | yolo | none
  OvdHandle* det = nullptr;
  sgc_encoder* enc = nullptr;
  sgc_labels* lt = nullptr;
  std::vector<std::string> labels;   // scenemap 프롬프트 표
  std::string namer, engine_path;     // siglip | engine
  std::vector<int> prompt_cls;       // engine 이름: 검출 cls → labels 번호
  std::vector<int> text_cls;         // siglip 이름: 글 줄 → labels 번호
  std::vector<float> text;           // siglip: 글 임베딩
  std::vector<int32_t> cls;
  bool objprob = false;               // objprob: 배경 낱말(kVocabAp)도 글 표에
  std::vector<float> emb;            // siglip: 검출마다 SigLIP 2 마스크 임베딩(n × SGC_DIM, L2 정규화) — 확률 물체 모델(objprob)이 씀
  double det_ms = 0, clip_ms = 0;
  int n_calls = 0;

  int labelId(const std::string& l) {
    for (size_t k = 0; k < labels.size(); ++k) if (labels[k] == l) return int(k);
    labels.push_back(l);
    return int(labels.size()) - 1;
  }
  bool init(const std::string& m, const std::string& engine_in, const std::string& clip_plan, const std::string& labels_dir, float conf,
            const std::string& namer_in) {
    mode = m;
    if (mode == "none") return true;
    namer = namer_in.empty() ? (mode == "fastsam" ? "siglip" : "engine") : namer_in;
    if (mode == "fastsam" && namer != "siglip") { std::fprintf(stderr, "fastsam needs --namer siglip\n"); return false; }
    std::string engine = engine_in;
    if (engine.empty())
      engine = ra::models() + (mode == "yolo" ? "/archive/x86_sm120/yolo26s-seg-416.plan"   // yolo: 비교용, 보관 엔진(2026-10-05)
                                      : std::string("/x86_sm120/") + kDefaultEngine);
    engine_path = engine;
    char err[2048] = {0};
    OvdConfig oc;
    ovd_default_config(&oc);
    const std::string names = engine + ".names.txt";
    oc.seg_engine = engine.c_str();
    oc.names = names.c_str();
    oc.conf_th = conf;
    det = ovd_create(&oc, err, sizeof err);
    if (!det) { std::fprintf(stderr, "ovd_create: %s\n", err); return false; }
    if (mode == "yolo") {   // 닫힌 어휘(COCO 등): 엔진 어휘 전부. engine 이름 = kVocab 에 있으면 그 지도 이름, 없으면 엔진 이름 그대로
      ovd_set_prompt(det, nullptr, 0, nullptr, 0);
      if (namer == "engine")
        for (int i = 0; i < ovd_vocab_size(det); ++i) {
          std::string lab = ovd_vocab_name(det, i);
          for (const Word& w : kVocab) if (lab == w.text) { lab = w.label; break; }
          prompt_cls.push_back(labelId(lab));
        }
      std::fprintf(stderr, "yolo: %d engine classes\n", ovd_vocab_size(det));
    } else {
      ovd_set_prompt(det, nullptr, 0, nullptr, 0);   // FastSAM: 'object' 하나
    }
    if (namer != "siglip") { std::fprintf(stderr, "naming: engine classes -> %zu labels\n", labels.size()); return true; }
    return initSiglip(clip_plan, labels_dir);
  }
  // SigLIP 2 이름(세 검출기 모두 같은 글 표·같은 라벨 표): 마스크 조각 임베딩 · 글 임베딩 최댓값. objprob 는 검출 캐시(--load)여도 부름(통째 다시 담기·글 모델)
  bool initSiglip(const std::string& clip_plan, const std::string& labels_dir) {
    char err[2048] = {0};
    sgc_config cc;
    sgc_default_config(&cc);
    cc.engine = clip_plan.c_str();
    enc = sgc_create(&cc, err, sizeof err);
    if (!enc) { std::fprintf(stderr, "sgc_create: %s\n", err); return false; }
    lt = sgc_labels_open(labels_dir.c_str(), nullptr, err, sizeof err);
    if (!lt) { std::fprintf(stderr, "sgc_labels_open: %s\n", err); return false; }
    std::vector<float> e(SGC_DIM);
    std::string missing;
    std::vector<Word> words(std::begin(kVocab), std::end(kVocab));
    if (objprob) words.insert(words.end(), std::begin(kVocabAp), std::end(kVocabAp));
    for (const Word& w : words) {
      const int row = sgc_labels_find(lt, w.text);
      if (row < 0 || sgc_labels_text_emb(lt, row, e.data()) != 0) { missing += std::string(" '") + w.text + "'"; continue; }
      double n = 0;
      for (float x : e) n += double(x) * x;
      n = std::sqrt(std::max(n, 1e-20));
      for (float& x : e) x = float(x / n);
      text_cls.push_back(labelId(w.label));
      text.insert(text.end(), e.begin(), e.end());
    }
    std::fprintf(stderr, "%s+siglip2: %zu prompts -> %zu labels (table %s; not in table:%s)\n", mode.c_str(), text_cls.size(), labels.size(),
                 sgc_labels_sha(lt), missing.c_str());
    return true;
  }
  // objprob 통째 다시 담기: 요청 마스크(검출 마스크 격자 배치)로 SigLIP 2. 결과 emb(n × SGC_DIM, 요청 순서)
  double reenc_ms = 0;
  long n_reenc = 0, n_enc_dets = 0;
  bool encodeMasks(const std::vector<uint8_t>& rgb, int W, int H, double stamp, const sm_detections& Dm, const sm_reenc_req* rq,
                   const uint32_t* bits, int n, std::vector<float>* out) {
    const auto t0 = std::chrono::steady_clock::now();
    sgc_frame fr{};
    fr.rgb = rgb.data(); fr.on_device = 0; fr.row_stride = int64_t(W) * 3; fr.pix_stride = 3; fr.w = W; fr.h = H;
    fr.mask_w = Dm.mask_w; fr.mask_h = Dm.mask_h; fr.mask_sx = Dm.mask_sx; fr.mask_sy = Dm.mask_sy; fr.mask_ox = Dm.mask_ox; fr.mask_oy = Dm.mask_oy;
    fr.mask_bits = bits;
    std::vector<sgc_item> items(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) items[size_t(k)] = sgc_item{uint32_t(k), k, {rq[k].box[0], rq[k].box[1], rq[k].box[2], rq[k].box[3]}, rq[k].kappa};
    out->assign(size_t(n) * SGC_DIM, 0.f);
    std::vector<sgc_result> res(64);
    int sent = 0, got = 0;
    while (got < n) {
      if (sent < n) {
        const int s2 = sgc_submit(enc, stamp, &fr, items.data() + sent, n - sent);
        if (s2 < 0) return false;
        sent += s2;
      }
      const int r = sgc_poll(enc, res.data(), int(res.size()), 1);
      for (int q = 0; q < r; ++q) std::copy(res[size_t(q)].emb, res[size_t(q)].emb + SGC_DIM, out->begin() + size_t(res[size_t(q)].id) * SGC_DIM);
      got += r;
    }
    reenc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    n_reenc += n;
    return true;
  }
  int64_t deviceBytes() const {
    int64_t b = det ? ovd_device_bytes(det) : 0;
    if (enc) { sgc_timing tm{}; sgc_get_timing(enc, &tm); b += tm.device_bytes; }
    return b;
  }
  // RGB(호스트, 3 채널) → 검출. 결과 D 는 다음 detect 까지 유효
  bool detect(const std::vector<uint8_t>& rgb, int W, int H, double stamp, sm_detections* D) {
    const auto ta = std::chrono::steady_clock::now();
    OvdImage im{};
    im.stamp = stamp; im.cam = 0; im.data = rgb.data(); im.h = H; im.w = W; im.row_stride = int64_t(W) * 3; im.pix_stride = 3;
    const sm_detections* d = ovd_detect(det, &im, nullptr);
    if (!d) { std::fprintf(stderr, "ovd_detect: %s\n", ovd_last_error(det)); return false; }
    *D = *d;
    const auto tb = std::chrono::steady_clock::now();
    det_ms += std::chrono::duration<double, std::milli>(tb - ta).count();
    ++n_calls;
    cls.assign(size_t(D->n), 0);
    emb.clear();
    if (namer == "engine") {
      for (int k = 0; k < D->n; ++k) cls[size_t(k)] = prompt_cls[size_t(D->cls[k])];
    } else if (D->n > 0) {
      sgc_frame fr{};
      fr.rgb = rgb.data(); fr.on_device = 0; fr.row_stride = int64_t(W) * 3; fr.pix_stride = 3; fr.w = W; fr.h = H;
      fr.mask_w = D->mask_w; fr.mask_h = D->mask_h; fr.mask_sx = D->mask_sx; fr.mask_sy = D->mask_sy; fr.mask_ox = D->mask_ox;
      fr.mask_oy = D->mask_oy; fr.mask_bits = D->mask_bits;
      std::vector<sgc_item> items(size_t(D->n));
      for (int k = 0; k < D->n; ++k) items[size_t(k)] = sgc_item{uint32_t(k), k, {D->box[4 * k], D->box[4 * k + 1], D->box[4 * k + 2], D->box[4 * k + 3]}, 1.f};
      n_enc_dets += D->n;
      std::vector<sgc_result> res(64);
      emb.assign(size_t(D->n) * SGC_DIM, 0.f);
      int sent = 0, got = 0;
      while (got < D->n) {
        if (sent < D->n) {
          const int s = sgc_submit(enc, stamp, &fr, items.data() + sent, D->n - sent);
          if (s < 0) { std::fprintf(stderr, "sgc_submit failed\n"); return false; }
          sent += s;
        }
        const int r = sgc_poll(enc, res.data(), int(res.size()), 1);
        for (int q = 0; q < r; ++q) {
          const float* e = res[size_t(q)].emb;
          std::copy(e, e + SGC_DIM, emb.begin() + size_t(res[size_t(q)].id) * SGC_DIM);
          int best = 0;
          float bs = -2.f;
          for (size_t p = 0; p < text_cls.size(); ++p) {
            float sd = 0.f;
            const float* tp = text.data() + p * SGC_DIM;
            for (int j = 0; j < SGC_DIM; ++j) sd += e[j] * tp[j];
            if (sd > bs) { bs = sd; best = int(p); }
          }
          cls[res[size_t(q)].id] = text_cls[size_t(best)];
        }
        got += r;
      }
      clip_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
    }
    D->cls = cls.data();
    return true;
  }
  ~Detector() {
    if (enc) sgc_destroy(enc);
    if (lt) sgc_labels_close(lt);
    if (det) ovd_destroy(det);
  }
};

// 검출 캐시: 머리 magic nl [len name]… 다음 프레임마다 key(프레임 번호) n img_w img_h | cls score box mask_w mask_h ms[4] bits
//   (RBD2: 그 뒤 dim, n × dim FP16 임베딩 — 검출이 0 개여도 dim 은 씀)
void writeDets(gzFile z, int key, const sm_detections& D, const std::vector<float>& emb) {
  putI(z, key); putI(z, D.n); putI(z, D.img_w); putI(z, D.img_h);
  const int dim = D.n > 0 && emb.size() == size_t(D.n) * SGC_DIM ? SGC_DIM : 0;
  if (D.n <= 0) { putI(z, 0); return; }
  gzwrite(z, D.cls, unsigned(4 * D.n));
  std::vector<float> sc(size_t(D.n), 1.f);
  if (D.score) std::copy(D.score, D.score + D.n, sc.begin());
  gzwrite(z, sc.data(), unsigned(4 * D.n));
  gzwrite(z, D.box, unsigned(16 * D.n));
  putI(z, D.mask_w); putI(z, D.mask_h);
  const float ms[4] = {D.mask_sx, D.mask_sy, D.mask_ox, D.mask_oy};
  gzwrite(z, ms, 16);
  gzwrite(z, D.mask_bits, unsigned(4 * maskWords(D) * size_t(D.n)));
  putI(z, dim);
  if (dim) {
    std::vector<uint16_t> h(emb.size());
    sgc_f32_to_f16(emb.data(), h.data(), int32_t(h.size()));
    gzwrite(z, h.data(), unsigned(2 * h.size()));
  }
}
struct DumpReader {
  gzFile z = nullptr;
  int32_t key = -1;
  bool have = false;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
  std::vector<float> emb;             // RBD2: 이 프레임 임베딩(n × SGC_DIM, 없으면 빔)
  bool v2 = false;
  bool open(const std::string& p, std::vector<std::string>* labels) {
    z = gzopen(p.c_str(), "rb");
    int32_t magic = 0, nl = 0;
    if (!z || !getI(z, &magic) || (magic != kDumpMagic && magic != kDumpMagic2) || !getI(z, &nl)) return false;
    v2 = magic == kDumpMagic2;
    labels->resize(size_t(nl));
    for (auto& l : *labels) { int32_t k = 0; getI(z, &k); l.resize(size_t(k)); gzread(z, l.data(), unsigned(k)); }
    have = getI(z, &key);
    return true;
  }
  // key 가 맞으면 D 를 채움
  bool get(int want, double stamp, sm_detections* D) {
    if (!have || key != want) return false;
    int32_t n, iw, ih;
    getI(z, &n); getI(z, &iw); getI(z, &ih);
    *D = sm_detections{};
    D->stamp = stamp; D->img_w = iw; D->img_h = ih; D->n = n;
    if (n > 0) {
      cls.resize(size_t(n)); score.resize(size_t(n)); box.resize(size_t(4 * n));
      gzread(z, cls.data(), unsigned(4 * n)); gzread(z, score.data(), unsigned(4 * n)); gzread(z, box.data(), unsigned(16 * n));
      getI(z, &D->mask_w); getI(z, &D->mask_h);
      float ms[4];
      gzread(z, ms, 16);
      D->mask_sx = ms[0]; D->mask_sy = ms[1]; D->mask_ox = ms[2]; D->mask_oy = ms[3];
      bits.resize(maskWords(*D) * size_t(n));
      gzread(z, bits.data(), unsigned(4 * bits.size()));
      D->cls = cls.data(); D->score = score.data(); D->box = box.data(); D->mask_bits = bits.data();
    }
    emb.clear();
    if (v2) {
      int32_t dim = 0;
      getI(z, &dim);
      if (dim > 0) {
        std::vector<uint16_t> h(size_t(n) * dim);
        gzread(z, h.data(), unsigned(2 * h.size()));
        emb.resize(h.size());
        sgc_f16_to_f32(h.data(), emb.data(), int32_t(h.size()));
      }
    }
    have = getI(z, &key);
    return true;
  }
};

const char* stateName(int s) { return s == SM_SEEN ? "seen" : s == SM_GONE ? "gone" : s == SM_MOVED ? "moved" : s == SM_HELD ? "held" : "?"; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: realbag_run <stream dir>[,<stream dir>...] <out dir> [--robot limo_omx|r1pro] [--pose carto|odom|gt] "
                         "[--det fastsam|yolo|none] [--namer siglip|engine] [--det-every 3] [--dump f.gz|--load f.gz] [--live host:port] [--rate 1] [--sg run_dir] "
                         "[--ref-map memdir] [--snap-at t,..] [--frames N] [--objprob [--label-prior f.json]] [--inspect]\n");
    return 2;
  }
  std::string carto_config;
  std::string robot = "limo_omx", pose = "carto", det_mode = "fastsam", namer, engine, dump_path, load_path, live, sg_run, ref_map, snap_at;
  std::string clip_plan = ra::models() + "/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan";
  std::string labels_dir = ra::labels();
  int det_every = 3;
  long max_frames = 1L << 40;
  double rate = 1.0, gap = 5.0, save_s = 1.0;
  float max_depth = 4.0f;
  float conf = 0.25f;
  std::string label_prior;   // objprob 라벨 사전(objprob_fit.py label_prior.json: 이름 → log 사전, 정답 없이 조각 임베딩 EM 으로 잰 것)
  std::string obj_params_file;   // objprob 엔진별 매개변수(objprob_params/<엔진>.json). 비면 엔진 이름으로 고름, "none" = 안 씀
  bool inspect = false;   // 살펴본 정도(scenemap README "살펴본 정도", sm_set_inspect) — view.json·scene.json objects[].inspect
  int objprob_flag = -1;   // 확률 물체 모델(objprob)(분할 조각 + SigLIP 2 임베딩 — 이름 없는 같은 것·기하 구조물·vMF 벡터·이름 사후·통째 다시 담기).
                           // -1(기본) = SigLIP 2 이름 길(--det fastsam, --namer siglip, RBD2 캐시)이면 켬, --objprob = 켬, --no-objprob = 옛 규칙
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--robot") robot = nx(); else if (a == "--pose") pose = nx(); else if (a == "--det") det_mode = nx();
    else if (a == "--det-every") det_every = std::max(1, std::stoi(nx())); else if (a == "--dump") dump_path = nx();
    else if (a == "--load") load_path = nx(); else if (a == "--live") live = nx(); else if (a == "--rate") rate = std::stod(nx());
    else if (a == "--sg") sg_run = nx(); else if (a == "--ref-map") ref_map = nx(); else if (a == "--snap-at") snap_at = nx();
    else if (a == "--frames") max_frames = std::stol(nx()); else if (a == "--conf") conf = std::stof(nx());
    else if (a == "--engine") engine = nx(); else if (a == "--namer") namer = nx(); else if (a == "--clip") clip_plan = nx(); else if (a == "--labels") labels_dir = nx();
    else if (a == "--objprob") objprob_flag = 1; else if (a == "--no-objprob") objprob_flag = 0; else if (a == "--label-prior") label_prior = nx(); else if (a == "--inspect") inspect = true; else if (a == "--objprob-params") obj_params_file = nx();
    else if (a == "--carto-config") carto_config = nx();
    else if (a == "--gap") gap = std::stod(nx()); else if (a == "--max-depth") max_depth = std::stof(nx()); else if (a == "--save-s") save_s = std::stod(nx());
    else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
  }
  std::vector<Stream> streams;
  {
    std::stringstream ss(argv[1]);
    std::string d;
    while (std::getline(ss, d, ',')) {
      Stream s;
      if (!loadStream(d, &s)) { std::fprintf(stderr, "cannot read stream %s\n", d.c_str()); return 1; }
      streams.push_back(std::move(s));
    }
  }
  if (streams.size() > 1 && pose != "gt") { std::fprintf(stderr, "several streams need --pose gt (no relocalisation between sessions)\n"); return 2; }
  const std::string out = argv[2];
  rb::mkdirs(out + "/memory");
  std::vector<double> snap_times;
  { std::stringstream ss(snap_at); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) snap_times.push_back(std::stod(t)); }

  // 검출
  // objprob 기본: 마스크마다 SigLIP 2 임베딩이 있는 길(fastsam·--namer siglip·RBD2 캐시)이면 켬. 엔진 클래스 이름(yolo)·RBD1 캐시·none 은 옛 규칙
  bool objprob = objprob_flag == 1;
  if (objprob_flag < 0) {
    if (!load_path.empty()) {
      objprob = true;
      std::stringstream ss(load_path);
      for (std::string p; std::getline(ss, p, ',');) {
        gzFile z = gzopen(p.c_str(), "rb");
        int32_t mg = 0;
        if (!z || !getI(z, &mg) || mg != kDumpMagic2) objprob = false;
        if (z) gzclose(z);
      }
    } else {
      objprob = det_mode == "fastsam" || (det_mode != "none" && namer == "siglip");
    }
  }
  std::fprintf(stderr, "objprob %s\n", objprob ? "on" : "off (old rules)");
  Detector D;
  D.objprob = objprob;
  std::vector<DumpReader> dr;   // 판마다 하나(--load a.gz,b.gz,…). 이름 표는 모두 같아야 함
  if (!load_path.empty()) {
    std::stringstream ss(load_path);
    std::string p;
    while (std::getline(ss, p, ',')) {
      std::vector<std::string> lab;
      dr.emplace_back();
      if (!dr.back().open(p, &lab)) { std::fprintf(stderr, "bad dump %s\n", p.c_str()); return 1; }
      if (D.labels.empty()) D.labels = lab;
      else if (lab != D.labels) { std::fprintf(stderr, "dump %s has a different label table\n", p.c_str()); return 1; }
    }
    if (dr.size() != streams.size()) { std::fprintf(stderr, "--load needs one dump per stream (%zu vs %zu)\n", dr.size(), streams.size()); return 1; }
    D.mode = "load";
  } else if (!D.init(det_mode, engine, clip_plan, labels_dir, conf, namer)) {
    return 1;
  }
  if (objprob) {
    if (!D.enc && !D.initSiglip(clip_plan, labels_dir)) return 1;
    for (const ApLabel& a : kApLabels) D.labelId(a.label);   // 상위어 라벨(furniture …)·object 를 표에 더함
  }
  if (D.labels.empty()) D.labels.push_back("object");
  gzFile dz = nullptr;
  if (!dump_path.empty()) {
    dz = gzopen(dump_path.c_str(), "wb1");
    putI(dz, kDumpMagic2);
    putI(dz, int32_t(D.labels.size()));
    for (auto& l : D.labels) { putI(dz, int32_t(l.size())); gzwrite(dz, l.data(), unsigned(l.size())); }
  }

  // scenemap
  const std::string cfg = "{\"robot\": \"" + robot + "\"}";
  sm_ctx* c = sm_create(cfg.c_str());
  if (!c) { std::fprintf(stderr, "sm_create rejected %s\n", cfg.c_str()); return 1; }
  {
    std::vector<const char*> lp;
    for (auto& l : D.labels) lp.push_back(l.c_str());
    sm_set_labels(c, lp.data(), int(lp.size()));
  }
  std::string obj_kv;   // 엔진별 매개변수 파일의 obj_params(sm_set_obj_params)
  if (objprob && obj_params_file != "none") {
    // 엔진별 매개변수: --objprob-params 가 없으면 objprob_params/<엔진 파일 이름에서 .plan 뺀 것>.json(엔진을 안 주면 기본 엔진 —
    // --load 캐시는 그 엔진을 --engine 으로 알려 줘야 맞는 파일을 고름). 파일: {"obj_params": "key=val,…", "label_prior": "옆 파일.json"}
    std::string pf = obj_params_file;
    if (pf.empty()) {
      pf = std::string(RB_PARAMS_DIR) + "/" + objprob_front::engineStem(engine) + ".json";
      if (!std::filesystem::exists(pf)) { std::fprintf(stderr, "objprob: no params file %s (built-in defaults)\n", pf.c_str()); pf.clear(); }
    }
    if (!pf.empty()) {
      if (!objprob_front::readParamsFile(pf, &obj_kv, &label_prior)) { std::fprintf(stderr, "cannot read %s\n", pf.c_str()); return 1; }
      obj_params_file = pf;
      std::fprintf(stderr, "objprob params %s (label prior %s)\n", pf.c_str(), label_prior.c_str());
    }
  }
  if (objprob) {   // objprob: 글 모델(낱말 줄 → 라벨)·라벨 크기 사전·상위어·엔진별 매개변수(objprob_front.hpp)
    std::string er;
    if (!objprob_front::enable(c, &D.labels, D.text, D.text_cls, label_prior, obj_kv, &er)) {
      std::fprintf(stderr, "%s%s\n", er.c_str(), er.find("unknown key") != std::string::npos ? (" in " + obj_params_file).c_str() : "");
      return 1;
    }
  }
  if (inspect) sm_set_inspect(c, 1);
  if (pose == "slam") pose = "carto";
  if (pose == "carto" && streams.size() == 1 && !(streams[0].have_bl && std::filesystem::exists(streams[0].dir + "/scans.bin"))) {
    std::fprintf(stderr, "[carto] %s has no 2D lidar (scans.bin/T_bl, bag2stream.py --scan-only) -> --pose odom\n", streams[0].dir.c_str());
    pose = "odom";
  }
  if (pose != "carto" && pose != "odom" && pose != "gt") { std::fprintf(stderr, "--pose carto|odom|gt (slam2d is archived)\n"); return 2; }
  sm_set_pose_mode(c, pose == "gt" ? SM_POSE_GT : pose == "odom" ? SM_POSE_ODOM : SM_POSE_EXT);
  // Cartographer(--pose carto): 스캔 + 바퀴 오도메트리 → 영상마다 자세(sm_push_ext_pose)
  std::vector<ScanRec> scans;
#ifdef RB_HAVE_CARTO
  sc_ctx* carto = nullptr;
#endif
  if (pose == "carto") {
#ifdef RB_HAVE_CARTO
    const Stream& S0 = streams[0];
    if (!S0.have_bl || !loadScans(S0.dir, S0.scan_dt, &scans)) {
      std::fprintf(stderr, "--pose carto: %s has no scans.bin/T_bl (bag2stream.py --scan-only)\n", S0.dir.c_str());
      return 1;
    }
    sc_config cc = sc_default_config();
    if (carto_config.empty()) carto_config = S0.kind == "openloris" ? "openloris_hokuyo.lua" : "limo_x2l.lua";
    cc.config_name = carto_config.c_str();
    cc.laser_xyz[0] = S0.T_bl[3]; cc.laser_xyz[1] = S0.T_bl[7]; cc.laser_xyz[2] = S0.T_bl[11];
    cc.laser_yaw = std::atan2(S0.T_bl[4], S0.T_bl[0]);
    char er[256] = {0};
    carto = sc_create(&cc, er, sizeof er);
    if (!carto) { std::fprintf(stderr, "%s\n", er); return 1; }
    std::fprintf(stderr, "[carto] %zu scans, config %s, scan_dt %+.2f s\n", scans.size(), carto_config.c_str(), S0.scan_dt);
#else
    std::fprintf(stderr, "--pose carto: built without slam_carto (tools/build_all.sh cartographer realbag)\n");
    return 1;
#endif
  }
  sm_set_cam_extrinsic(c, 0, streams[0].T_bc);
  const bool limo = robot == "limo_omx";
  const int n_prop = limo ? SM_LIMO_PROPRIO_DIM : SM_R1PRO_PROPRIO_DIM;

  // 스트림·기록
  rb::Capture cap;
  std::string sg_out, ep_name;
  if (!sg_run.empty()) {
    std::string nm;
    for (auto& s : streams) nm += (nm.empty() ? "" : "+") + s.name;
    if (nm.size() > 60) nm = streams.front().name + "+" + std::to_string(streams.size() - 1) + "more";
    ep_name = nm;
    sg_out = sg_run + "/replays/ep_000000_" + nm + ".sg";
    rb::mkdirs(sg_out + "/cam");
    if (!cap.start(sg_out + "/stream.sgs")) { std::perror("capture"); return 1; }
    if (sm_stream_start(c, ("127.0.0.1:" + std::to_string(cap.port)).c_str())) { std::fprintf(stderr, "sm_stream_start failed\n"); return 1; }
    rb::drain(c, cap);
  } else if (!live.empty()) {
    if (sm_stream_start(c, live.c_str())) { std::fprintf(stderr, "sm_stream_start %s failed\n", live.c_str()); return 1; }
  }

  FILE* ft = std::fopen((out + "/traj.csv").c_str(), "w");
  std::fprintf(ft, "session,frame,t,est_x,est_y,est_yaw,est_cx,est_cy,odo_cx,odo_cy,gt_ok,gt_cx,gt_cy,gt_yaw,n_det,n_obj\n");
  FILE* fe = std::fopen((out + "/events.csv").c_str(), "w");
  std::fprintf(fe, "t,session,frame,event,id,name,x,y,z\n");

  std::vector<std::array<double, 2>> est_c, odo_c, gt_c;   // 정답 있는 프레임만
  std::vector<double> est_yaw, gt_yaw;
  std::map<uint32_t, int> prev_state;                       // 지난 스냅숏 물체 상태
  std::map<uint32_t, std::string> prev_name;
  std::map<uint32_t, std::array<double, 3>> prev_pos;
  std::map<std::string, int> ev_count;
  std::vector<std::map<std::string, int>> ev_sess(streams.size());
  std::string camjson = "[", gtjson = "[";
  long n_frames = 0, n_detf = 0, n_dets = 0, cams = 0;
  double last_save = -1e9, last_view = -1e9, last_cam = -1e9, t_off = 0, t_end = 0;
  size_t snap_i = 0;
  std::vector<float> prop(size_t(n_prop), 0.f), dm;
  std::vector<uint16_t> dmm;
  std::vector<uint8_t> rgb, rgba;
  const auto w0 = std::chrono::steady_clock::now();
  bool have_first = false;
  double f_ex = 0, f_ey = 0, f_eyaw = 0, f_gx = 0, f_gy = 0, f_gyaw = 0, f_px = 0, f_py = 0, f_pyaw = 0;   // 첫 정답 프레임(첫 프레임 맞춤)
  double prev_ox = 0, prev_oy = 0, prev_oyaw = 0, prev_ot = -1;             // R1: 오도메트리 차 → base_qvel

  auto make_prop = [&](const Odo& o) {
    std::fill(prop.begin(), prop.end(), 0.f);
    if (limo) {
      prop[SM_LIMO_ODOM_X] = float(o.x); prop[SM_LIMO_ODOM_Y] = float(o.y); prop[SM_LIMO_ODOM_YAW] = float(o.yaw);
      prop[SM_LIMO_VX] = float(o.vx); prop[SM_LIMO_VY] = float(o.vy); prop[SM_LIMO_WZ] = float(o.wz);
      const float home_q[5] = {0.f, -1.6f, 1.45f, 0.15f, 0.f};   // OMX 홈(팔 접음), 그리퍼 0(닫힌 채 그대로 — 잡기 규칙 안 걸림)
      for (int k = 0; k < 5; ++k) prop[size_t(SM_LIMO_ARM_Q + k)] = home_q[k];
      prop[SM_LIMO_GRIPPER] = 0.f;
    } else {   // R1 61: base_qvel(0..2) = 이 구간 속도(베이스 기준) — 오도메트리 자세 차에서
      if (prev_ot >= 0 && o.t > prev_ot + 1e-6) {
        const double dt = o.t - prev_ot, c0 = std::cos(prev_oyaw), s0 = std::sin(prev_oyaw);
        const double dx = o.x - prev_ox, dy = o.y - prev_oy;
        prop[0] = float((c0 * dx + s0 * dy) / dt); prop[1] = float((-s0 * dx + c0 * dy) / dt); prop[2] = float(wrap(o.yaw - prev_oyaw) / dt);
      }
      prev_ox = o.x; prev_oy = o.y; prev_oyaw = o.yaw; prev_ot = o.t;
      for (int k : {19, 44}) prop[size_t(k)] = -100.f;
      for (int k : {24, 25, 49, 50}) prop[size_t(k)] = 0.05f;
    }
  };
  auto interp = [](const std::vector<Odo>& v, double t) {
    auto it = std::lower_bound(v.begin(), v.end(), t, [](const Odo& o, double tt) { return o.t < tt; });
    if (it == v.begin()) return v.front();
    if (it == v.end()) return v.back();
    const Odo& b = *it;
    const Odo& a = *(it - 1);
    const double w = (t - a.t) / std::max(1e-9, b.t - a.t);
    Odo o = a;
    o.t = t;
    o.x = a.x + (b.x - a.x) * w; o.y = a.y + (b.y - a.y) * w; o.yaw = a.yaw + wrap(b.yaw - a.yaw) * w;
    o.vx = a.vx + (b.vx - a.vx) * w; o.vy = a.vy + (b.vy - a.vy) * w; o.wz = a.wz + (b.wz - a.wz) * w;
    return o;
  };

  for (size_t si = 0; si < streams.size() && n_frames < max_frames; ++si) {
    const Stream& S = streams[si];
    const double t0 = S.frames.front().stamp;
    size_t oi = 0;
    while (oi < S.odom.size() && S.odom[oi].t < t0 - 0.5) ++oi;
    size_t csi = 0, coi = oi;   // Cartographer 넣기 위치(스캔·오도메트리, 시각 순)
    while (csi < scans.size() && scans[csi].t < t0 - 0.5) ++csi;
    const double sess_start = t_off;
    for (size_t fi = 0; fi < S.frames.size() && n_frames < max_frames; ++fi) {
      const FrameRow& f = S.frames[fi];
      const double t = f.stamp - t0 + t_off;
      // 걸음 맞춤(실시간 스트림)
      if (!live.empty() && rate > 0) {
        const double due = t / rate;
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        if (due > now) std::this_thread::sleep_for(std::chrono::duration<double>(due - now));
      }
      cap.now = t;
      // 정답 자세(gt 모드 = 지도 자세, 아니면 떠밀림 진단)
      if (f.gt_ok) {
        sm_pose2 g = gtBase(S, f);
        g.stamp = t;
        sm_push_pose(c, &g);
      }
      // 오도메트리: 영상 시각까지 전부 + 영상 시각에 보간 하나
      while (oi < S.odom.size() && S.odom[oi].t - t0 + t_off < t - 1e-6) {
        Odo o = S.odom[oi++];
        o.t = o.t - t0 + t_off;
        make_prop(o);
        const sm_proprio p{o.t, prop.data(), n_prop};
        sm_push_proprio(c, &p);
      }
      Odo oc = interp(S.odom, f.stamp);
      oc.t = t;
      make_prop(oc);
      { const sm_proprio p{t, prop.data(), n_prop}; sm_push_proprio(c, &p); }
      // 깊이(+ 검출 프레임이면 RGB·검출)
      if (!dom::readU16(f.depth, S.w, S.h, &dmm)) { std::fprintf(stderr, "depth %s\n", f.depth.c_str()); return 1; }
      dom::depthToM(dmm, &dm);
      for (float& v : dm) if (v > max_depth) v = 0.f;   // 실제 깊이 센서: 먼 값은 잡음이 커서(구조광·스테레오 오차 ∝ z²) 무효로
      const bool kf = (fi % size_t(det_every)) == 0;
      sm_detections Dt{};
      bool have_det = false;
      if (kf) {
        if (!dom::readRgb(f.rgb, S.w, S.h, &rgb)) { std::fprintf(stderr, "rgb %s\n", f.rgb.c_str()); return 1; }
        rgba.resize(size_t(S.w) * S.h * 4);
        for (size_t i = 0; i < size_t(S.w) * S.h; ++i) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2]; rgba[4 * i + 3] = 255; }
        if (D.mode == "load") have_det = dr[si].get(int(fi), t, &Dt);
        else if (D.mode != "none") {
          if (!D.detect(rgb, S.w, S.h, t, &Dt)) return 1;
          have_det = true;
          if (dz) writeDets(dz, int(fi), Dt, D.emb);   // 판 하나 기준 번호(여러 판이면 --dump 는 판마다 따로 돌릴 것)
        }
        if (have_det) { Dt.stamp = t; Dt.cam = 0; ++n_detf; n_dets += Dt.n; }
      }
#ifdef RB_HAVE_CARTO
      if (carto) {   // 영상 시각까지 스캔·오도메트리를 시각 순으로 넣고 그 시각 자세를 scenemap 에
        while (csi < scans.size() || coi < S.odom.size()) {
          const double ts = csi < scans.size() ? scans[csi].t : 1e300, to = coi < S.odom.size() ? S.odom[coi].t : 1e300;
          if (std::min(ts, to) > f.stamp) break;
          if (to <= ts) { const Odo& o = S.odom[coi++]; sc_push_odom(carto, o.t - t0 + t_off, o.x, o.y, o.yaw); }
          else {
            const ScanRec& r = scans[csi++];
            sc_push_scan(carto, r.t - t0 + t_off, int32_t(r.r.size()), r.r.data(), r.a0, r.da, r.dt, r.rmin, r.rmax);
          }
        }
        double cp[3];
        if (!sc_pose_at(carto, t, cp)) { const sm_pose2 ep{t, cp[0], cp[1], cp[2]}; sm_push_ext_pose(c, &ep); }
      }
#endif
      sm_image im{t, 0, S.w, S.h, kf ? rgba.data() : nullptr, dm.data(), S.fx, S.fy, S.cx, S.cy};
      if (objprob && have_det) {
        const std::vector<float>& E = D.mode == "load" ? dr[si].emb : D.emb;
        if (E.size() == size_t(Dt.n) * SGC_DIM) sm_set_det_embeddings(c, E.data(), Dt.n, SGC_DIM);
        else if (Dt.n > 0) { std::fprintf(stderr, "objprob: no embeddings in this dump (needs RBD2 from a siglip run)\n"); return 1; }
      }
      sm_push_image_rgb(c, &im, have_det ? &Dt : nullptr, nullptr);
      if (objprob && have_det) {   // 통째 다시 담기: 합친 물체·더 좋은 모습을 구름 투영 마스크로 SigLIP 2
        const sm_reenc_req* rq = nullptr;
        const uint32_t* rb = nullptr;
        const int nr = sm_reencode_requests(c, &rq, &rb);
        if (nr > 0) {
          std::vector<uint32_t> bits(rb, rb + size_t(nr) * maskWords(Dt));
          std::vector<sm_reenc_req> reqs(rq, rq + nr);
          std::vector<float> ze;
          if (!D.encodeMasks(rgb, S.w, S.h, t, Dt, reqs.data(), bits.data(), nr, &ze)) { std::fprintf(stderr, "reencode failed\n"); return 1; }
          std::vector<uint32_t> ids(static_cast<size_t>(nr));
          for (int k = 0; k < nr; ++k) ids[size_t(k)] = reqs[size_t(k)].id;
          static const char* rdump = std::getenv("RB_REENC_DUMP");   // 진단: 다시 담기 상자 그림(마스크 밖 어둡게) <dir>/t_<시각>_O<id>.jpg
          static int n_rd = 0;
          if (rdump && n_rd < 400)
            for (int k = 0; k < nr; ++k, ++n_rd) {
              const sm_reenc_req& q = reqs[size_t(k)];
              const int x0 = std::max(0, int(q.box[0])), y0 = std::max(0, int(q.box[1])), x1 = std::min(S.w, int(q.box[2])), y1 = std::min(S.h, int(q.box[3]));
              if (x1 - x0 < 4 || y1 - y0 < 4) continue;
              std::vector<uint8_t> im2(size_t(x1 - x0) * (y1 - y0) * 3);
              const uint32_t* bb = bits.data() + size_t(k) * maskWords(Dt);
              for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                  const int i = int(std::floor((x + 0.5f - Dt.mask_ox) / Dt.mask_sx)), j = int(std::floor((y + 0.5f - Dt.mask_oy) / Dt.mask_sy));
                  const size_t cc = size_t(std::clamp(j, 0, Dt.mask_h - 1)) * Dt.mask_w + size_t(std::clamp(i, 0, Dt.mask_w - 1));
                  const bool in = (bb[cc >> 5] >> (cc & 31)) & 1u;
                  for (int ch = 0; ch < 3; ++ch) im2[(size_t(y - y0) * (x1 - x0) + (x - x0)) * 3 + ch] = in ? rgb[(size_t(y) * S.w + x) * 3 + ch] : rgb[(size_t(y) * S.w + x) * 3 + ch] / 4;
                }
              char fn[256];
              std::snprintf(fn, sizeof fn, "%s/t%07.2f_O%u.jpg", rdump, t, q.id);
              rb::writeJpeg(fn, im2.data(), x1 - x0, y1 - y0, 85);
            }
          sm_set_object_embeddings(c, ids.data(), ze.data(), nr, SGC_DIM);
        }
      }
      if (!sg_out.empty()) rb::drain(c, cap);
      ++n_frames;
      t_end = t;
      // 궤적·물체 사건
      sm_snapshot_t* snap = nullptr;
      sm_snapshot(c, &snap);
      const sm_pose2 P = sm_snap_pose(snap);
      double ecx, ecy, ocx, ocy;
      camXY(S, P.x, P.y, P.yaw, &ecx, &ecy);
      camXY(S, oc.x, oc.y, oc.yaw, &ocx, &ocy);
      const sm_object* ob = nullptr;
      const int no = sm_snap_objects(snap, &ob);
      int live_n = 0;
      if (kf) {
        std::map<uint32_t, int> cur;
        for (int k = 0; k < no; ++k) {
          const sm_object& o = ob[k];
          if (o.structural) continue;
          cur[o.id] = o.state;
          if (o.state != SM_GONE) ++live_n;
          auto it = prev_state.find(o.id);
          const char* ev = nullptr;
          if (it == prev_state.end()) ev = "new";
          else if (it->second != o.state) ev = o.state == SM_GONE ? "gone" : o.state == SM_MOVED ? "moved" : o.state == SM_HELD ? "held" : "seen_again";
          if (ev) {
            std::fprintf(fe, "%.3f,%zu,%zu,%s,%u,%s,%.3f,%.3f,%.3f\n", t, si, fi, ev, o.id, o.name ? o.name : "", o.pos[0], o.pos[1], o.pos[2]);
            ++ev_count[ev];
            ++ev_sess[si][ev];
          }
          prev_name[o.id] = o.name ? o.name : "";
          prev_pos[o.id] = {o.pos[0], o.pos[1], o.pos[2]};
        }
        for (auto& [id, st] : prev_state)
          if (!cur.count(id)) {
            const auto& p = prev_pos[id];
            std::fprintf(fe, "%.3f,%zu,%zu,removed,%u,%s,%.3f,%.3f,%.3f\n", t, si, fi, id, prev_name[id].c_str(), p[0], p[1], p[2]);
            ++ev_count["removed"];
            ++ev_sess[si]["removed"];
          }
        prev_state.swap(cur);
      }
      std::fprintf(ft, "%zu,%zu,%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%d,%d\n", si, fi, t, P.x, P.y, P.yaw, ecx, ecy, ocx, ocy,
                   f.gt_ok, f.gcx, f.gcy, f.gyaw, have_det ? Dt.n : -1, kf ? live_n : -1);
      if (f.gt_ok) {
        const sm_pose2 g = gtBase(S, f);
        if (!have_first) { have_first = true; f_ex = oc.x; f_ey = oc.y; f_eyaw = oc.yaw; f_gx = g.x; f_gy = g.y; f_gyaw = g.yaw; f_px = P.x; f_py = P.y; f_pyaw = P.yaw; }
        est_c.push_back({ecx, ecy}); odo_c.push_back({ocx, ocy}); gt_c.push_back({f.gcx, f.gcy});
        est_yaw.push_back(P.yaw); gt_yaw.push_back(g.yaw);
      }
      sm_snapshot_release(snap);
      // 뷰어 요약·저장·카메라 그림
      if (!sg_out.empty() || !live.empty()) {
        if (t - last_view >= 0.1) { sm_stream_view(c); last_view = t; if (!sg_out.empty()) rb::drain(c, cap); }
      }
      // 주기 저장. 재생 판(--sg)이면 그 판의 memory/ 에 — 나중에 지워지거나 합쳐진 물체의 조각·점구름도 남아 재생 화면이 그 시각에 그린다
      if (t - last_save >= save_s) { sm_save_dsg(c, ((sg_out.empty() ? out : sg_out) + "/memory").c_str()); last_save = t; }
      while (snap_i < snap_times.size() && t >= snap_times[snap_i]) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "/snap_%07.2f", snap_times[snap_i]);
        sm_save_dsg(c, (out + nm).c_str());
        ++snap_i;
      }
      if (!sg_out.empty() && kf && t - last_cam >= 0.5) {   // 카메라 그림 2 Hz, 256 폭 JPEG
        const int W = 256, H = std::max(1, S.h * W / S.w);
        std::vector<uint8_t> small(size_t(W) * H * 3);
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const uint8_t* sp = rgb.data() + (size_t(y * S.h / H) * S.w + size_t(x * S.w / W)) * 3;
            uint8_t* o = small.data() + (size_t(y) * W + x) * 3;
            o[0] = sp[0]; o[1] = sp[1]; o[2] = sp[2];
          }
        char fn[64];
        std::snprintf(fn, sizeof fn, "cam/%06ld.jpg", cams);
        rb::writeJpeg(sg_out + "/" + fn, small.data(), W, H, 80);
        camjson += std::string(cams ? "," : "") + "{\"t\":" + jnum(t) + ",\"file\":\"" + fn + "\"}";
        ++cams;
        last_cam = t;
      }
      if (n_frames % 150 == 0)
        std::fprintf(stderr, "[%s] t %.1f frame %zu/%zu dets %d objects %d\n", S.name.c_str(), t, fi, S.frames.size(), have_det ? Dt.n : -1, no);
    }
    t_off = t_end + gap;
    (void)sess_start;
  }
  if (dz) gzclose(dz);
  if (!sg_out.empty() || !live.empty()) { sm_stream_view(c); if (!sg_out.empty()) rb::drain(c, cap); }
  sm_save_stats ss{};
  sm_save_dsg_ex(c, (out + "/memory").c_str(), &ss);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();

  // ---- 잰 값 ----
  const Se2 Tu = align(est_c, gt_c), To = align(odo_c, gt_c);
  const Ate a_est = ate(est_c, gt_c, Tu), a_odo = ate(odo_c, gt_c, To);
  // 첫 프레임 맞춤(정답 없이 출발 자세만 맞춘 것 — 실제로 쓸 때와 같은 떠밀림). map 원점 = 첫 오도메트리 자세(scenemap)라 첫 프레임 추정 = 오도메트리 0
  Ate a_est1, a_odo1;
  if (!est_c.empty()) {
    const Se2 Tf = alignFirst(f_px, f_py, f_pyaw, f_gx, f_gy, f_gyaw);
    a_est1 = ate(est_c, gt_c, Tf);
    const Se2 Tof = alignFirst(f_ex, f_ey, f_eyaw, f_gx, f_gy, f_gyaw);
    a_odo1 = ate(odo_c, gt_c, Tof);
  }
  double yaw_rms = 0, gt_len = 0;
  {
    const double th = std::atan2(Tu.s, Tu.c);
    for (size_t i = 0; i < est_yaw.size(); ++i) { const double e = wrap(est_yaw[i] + th - gt_yaw[i]); yaw_rms += e * e; }
    if (!est_yaw.empty()) yaw_rms = std::sqrt(yaw_rms / double(est_yaw.size()));
    for (size_t i = 1; i < gt_c.size(); ++i) gt_len += std::hypot(gt_c[i][0] - gt_c[i - 1][0], gt_c[i][1] - gt_c[i - 1][1]);
  }
  sm_pose_diag pd{};
  sm_get_pose_diag(c, &pd);
  std::string cartoj = "null";
#ifdef RB_HAVE_CARTO
  if (carto) {
    sc_stats cs{};
    sc_get_stats(carto, &cs);
    sc_write_map(carto, (out + "/carto_map.pgm").c_str(), 0.05);
    cartoj = Obj().str("config", carto_config).num("scan_dt", streams[0].scan_dt).num("scans", double(cs.n_scans)).num("nodes", double(cs.n_nodes))
                 .num("submaps", double(cs.n_submaps)).num("loop_constraints", double(cs.n_loop_constraints)).num("us_scan_mean", cs.us_scan_mean)
                 .num("us_scan_max", cs.us_scan_max).done();
    sc_destroy(carto);
    carto = nullptr;
  }
#endif
  // 정답 궤적(map 좌표, 재생 화면 초록 선): map ← world = Tu⁻¹
  {
    const double ci = Tu.c, si2 = -Tu.s;
    int ng = 0;
    double lt = -1;
    std::ifstream tr(out + "/traj.csv");
    std::string line;
    std::getline(tr, line);
    while (std::getline(tr, line)) {
      const auto v = dom::splitCsv(line);
      if (v.size() < 14 || v[10] != "1") continue;
      const double t = std::atof(v[2].c_str());
      if (t - lt < 0.1) continue;
      lt = t;
      const double gx = std::atof(v[11].c_str()) - Tu.tx, gy = std::atof(v[12].c_str()) - Tu.ty;
      const double mx = ci * gx - si2 * gy, my = si2 * gx + ci * gy;
      gtjson += std::string(ng++ ? "," : "") + "[" + jnum(t) + "," + jnum(mx) + "," + jnum(my) + "," + jnum(wrap(std::atof(v[13].c_str()) - std::atan2(Tu.s, Tu.c))) + "]";
    }
  }
  gtjson += "]";
  std::fclose(ft);
  std::fclose(fe);
  // 물체 표·중복
  sm_snapshot_t* snap = nullptr;
  sm_snapshot(c, &snap);
  const sm_object* ob = nullptr;
  const int no = sm_snap_objects(snap, &ob);
  // 물체 표: 전부(큰 것·고정 종류 = structural 1 포함 — sgview 가 그리는 것과 같음). 옛 지표 live·dup_pairs 는 그대로 작은·옮길 수 있는 것만,
  // live_all·dup_pairs_all·by_label_all 은 전부
  std::map<std::string, int> by_label, by_state, by_label_all;
  int n_live = 0, n_dup = 0, n_live_all = 0, n_dup_all = 0;
  {
    FILE* fo = std::fopen((out + "/objects.csv").c_str(), "w");
    std::fprintf(fo, "id,name,state,x,y,z,ex,ey,ez,n_obs,last_seen,structural\n");
    for (int k = 0; k < no; ++k) {
      const sm_object& o = ob[k];
      std::fprintf(fo, "%u,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%.2f,%d\n", o.id, o.name ? o.name : "", stateName(o.state), o.pos[0], o.pos[1], o.pos[2],
                   o.extent[0], o.extent[1], o.extent[2], o.n_obs, o.last_seen, o.structural ? 1 : 0);
      if (o.state != SM_GONE) {
        ++n_live_all;
        ++by_label_all[o.name ? o.name : ""];
        for (int j = 0; j < k; ++j) {
          const sm_object& q = ob[j];
          if (q.state == SM_GONE || !o.name || !q.name || std::strcmp(o.name, q.name)) continue;
          if (std::hypot(o.pos[0] - q.pos[0], o.pos[1] - q.pos[1], o.pos[2] - q.pos[2]) < 0.5) ++n_dup_all;
        }
      }
      if (o.structural) continue;
      ++by_state[stateName(o.state)];
      if (o.state == SM_GONE) continue;
      ++n_live;
      ++by_label[o.name ? o.name : ""];
      for (int j = 0; j < k; ++j) {   // 중복 후보: 같은 이름, 살아 있음, 중심 0.5 m 안
        const sm_object& q = ob[j];
        if (q.structural || q.state == SM_GONE || !o.name || !q.name || std::strcmp(o.name, q.name)) continue;
        if (std::hypot(o.pos[0] - q.pos[0], o.pos[1] - q.pos[1], o.pos[2] - q.pos[2]) < 0.5) ++n_dup;
      }
    }
    std::fclose(fo);
  }
  // 지도
  sm_grid g{};
  sm_snap_map(snap, &g);
  long n_free = 0, n_occ = 0, n_known = 0;
  for (int i = 0; i < g.width * g.height; ++i) {
    const int v = g.cells[i];
    if (v < 0) continue;
    ++n_known;
    if (v >= 65) ++n_occ; else if (v <= 25) ++n_free;
  }
  std::vector<double> segs(4 * 512);
  const int n_seg = sm_snap_wall_segments(snap, segs.data(), 512);
  {   // 벽 선분(map, m) — 정답 벽과 비교용
    FILE* fw = std::fopen((out + "/walls.csv").c_str(), "w");
    std::fprintf(fw, "ax,ay,bx,by\n");
    for (int k = 0; k < std::min(n_seg, 512); ++k) std::fprintf(fw, "%.4f,%.4f,%.4f,%.4f\n", segs[4 * k], segs[4 * k + 1], segs[4 * k + 2], segs[4 * k + 3]);
    std::fclose(fw);
  }
  // 기준 지도(정답 자세로 만든 지도)와 비교: 이 지도 점유 칸을 Tu 로 정답 world 에 옮겨 기준 점유 칸까지 거리
  std::string mapcmp = "null";
  if (!ref_map.empty()) {
    Pgm R;
    if (loadPgm(ref_map, &R)) {
      auto ref_occ = [&](double x, double y, double r) {
        const int cx = int(std::floor((x - R.ox) / R.res)), cy = int(std::floor((y - R.oy) / R.res)), k = int(std::ceil(r / R.res));
        for (int dy = -k; dy <= k; ++dy)
          for (int dx = -k; dx <= k; ++dx) {
            const int xx = cx + dx, yy = cy + dy;
            if (xx < 0 || yy < 0 || xx >= R.w || yy >= R.h) continue;
            if (std::hypot(dx * R.res, dy * R.res) > r + 1e-9) continue;
            if (R.v[size_t(R.h - 1 - yy) * R.w + xx] == 0) return true;   // PGM 첫 줄 = 위(y 최대)
          }
        return false;
      };
      long n = 0, p5 = 0, p10 = 0;
      for (int y = 0; y < g.height; ++y)
        for (int x = 0; x < g.width; ++x) {
          if (g.cells[size_t(y) * g.width + x] < 65) continue;
          const double mx = g.origin[0] + (x + 0.5) * g.resolution, my = g.origin[1] + (y + 0.5) * g.resolution;
          const double wx = Tu.c * mx - Tu.s * my + Tu.tx, wy = Tu.s * mx + Tu.c * my + Tu.ty;
          ++n;
          if (ref_occ(wx, wy, 0.05)) ++p5;
          if (ref_occ(wx, wy, 0.10)) ++p10;
        }
      // 재현율: 기준 점유 칸 중 이 지도(점유)가 ±10 cm 안에 있는 것
      long rn = 0, r10 = 0;
      const double ci = Tu.c, sn = -Tu.s;
      for (int y = 0; y < R.h; ++y)
        for (int x = 0; x < R.w; ++x) {
          if (R.v[size_t(R.h - 1 - y) * R.w + x] != 0) continue;
          const double wx = R.ox + (x + 0.5) * R.res - Tu.tx, wy = R.oy + (y + 0.5) * R.res - Tu.ty;
          const double mx = ci * wx - sn * wy, my = sn * wx + ci * wy;
          const int gx = int(std::floor((mx - g.origin[0]) / g.resolution)), gy = int(std::floor((my - g.origin[1]) / g.resolution));
          ++rn;
          const int k = int(std::ceil(0.10 / g.resolution));
          bool hit = false;
          for (int dy = -k; dy <= k && !hit; ++dy)
            for (int dx = -k; dx <= k && !hit; ++dx) {
              const int xx = gx + dx, yy = gy + dy;
              if (xx < 0 || yy < 0 || xx >= g.width || yy >= g.height) continue;
              if (std::hypot(dx * g.resolution, dy * g.resolution) > 0.10 + 1e-9) continue;
              if (g.cells[size_t(yy) * g.width + xx] >= 65) hit = true;
            }
          if (hit) ++r10;
        }
      mapcmp = Obj().num("occ_cells", double(n)).num("prec_5cm", n ? double(p5) / n : 0).num("prec_10cm", n ? double(p10) / n : 0)
                   .num("ref_occ_cells", double(rn)).num("recall_10cm", rn ? double(r10) / rn : 0).str("ref", ref_map).done();
    }
  }
  sm_snapshot_release(snap);
  std::string apj = "null";
  if (objprob) {   // objprob 진단 셈·다시 담기 시간(SigLIP 호출 = 검출 조각 + 통째)
    int64_t a[16] = {0};
    sm_get_objprob_stats(c, a);
    const char* nm[16] = {"obs", "struct_wall_big", "struct_wall_name", "struct_ceiling", "struct_floor", "struct_det_name", "assoc", "new", "merge",
                          "struct_object", "reenc_req", "reenc_done", "through_window", "struct_obj_too_big", "struct_wall_tall", "obs_blocked"};
    Obj o;
    for (int k = 0; k < 16; ++k) o.num(nm[k], double(a[k]));
    o.num("reenc_ms_per_det_frame", n_detf ? D.reenc_ms / n_detf : 0).num("siglip_crops_per_det_frame", n_detf ? double(D.n_enc_dets + D.n_reenc) / n_detf : 0)
     .num("reenc_per_det_frame", n_detf ? double(D.n_reenc) / n_detf : 0);
    apj = o.done();
  }
  std::string labj = "{", evj = "{", stj = "{", sessj = "[", laj = "{";
  for (auto& [k, v] : by_label_all) laj += std::string(laj.size() > 1 ? "," : "") + jstr(k) + ":" + std::to_string(v);
  laj += "}";
  for (auto& [k, v] : by_label) labj += std::string(labj.size() > 1 ? "," : "") + jstr(k) + ":" + std::to_string(v);
  for (auto& [k, v] : ev_count) evj += std::string(evj.size() > 1 ? "," : "") + jstr(k) + ":" + std::to_string(v);
  for (auto& [k, v] : by_state) stj += std::string(stj.size() > 1 ? "," : "") + jstr(k) + ":" + std::to_string(v);
  for (size_t i = 0; i < ev_sess.size(); ++i) {
    std::string e = "{";
    for (auto& [k, v] : ev_sess[i]) e += std::string(e.size() > 1 ? "," : "") + jstr(k) + ":" + std::to_string(v);
    sessj += std::string(i ? "," : "") + Obj().str("stream", streams[i].name).raw("events", e + "}").done();
  }
  labj += "}"; evj += "}"; stj += "}"; sessj += "]";
  auto ateJ = [](const Ate& a) { return Obj().num("rms", a.rms).num("mean", a.mean).num("max", a.max).num("final", a.final).num("n", a.n).done(); };
  std::string snames;
  for (auto& s : streams) snames += (snames.empty() ? "" : ",") + s.name;
  double objmap_us = 0;   // scenemap objmap 단계 평균 µs/검출 keyframe
  {
    sm_stage_timing tm[32];
    const int nt = sm_get_timing(c, tm, 32);
    for (int i = 0; i < std::min(nt, 32); ++i)
      if (std::string(tm[i].name) == "objmap") objmap_us = tm[i].mean_us;
  }
  const std::string metrics =
      Obj().str("streams", snames).str("robot", robot).str("pose", pose).str("det", D.mode == "load" ? "load:" + load_path : D.mode)
          .str("namer", D.namer).str("engine", D.engine_path).num("det_gpu_mb", double(D.deviceBytes()) / 1048576.0)
          .raw("se2_map_to_gt", "[" + jnum(Tu.c) + "," + jnum(Tu.s) + "," + jnum(Tu.tx) + "," + jnum(Tu.ty) + "]")
          .num("det_every", det_every).num("max_depth", max_depth).num("frames", double(n_frames)).num("det_frames", double(n_detf))
          .num("dets_per_frame", n_detf ? double(n_dets) / n_detf : 0).num("det_ms", D.n_calls ? D.det_ms / D.n_calls : 0)
          .num("clip_ms", D.n_calls ? D.clip_ms / D.n_calls : 0).raw("objprob", apj).num("objmap_us", objmap_us).num("inspect", inspect ? 1 : 0).str("objprob_params", objprob ? obj_params_file : std::string()).num("duration_s", t_end).num("wall_s", wall).num("gt_path_m", gt_len)
          .raw("ate_se2_cam", ateJ(a_est)).raw("ate_se2_odom", ateJ(a_odo)).raw("ate_first_cam", ateJ(a_est1)).raw("ate_first_odom", ateJ(a_odo1))
          .num("yaw_rms_deg", yaw_rms * 180 / M_PI).raw("carto", cartoj)
          .raw("pose_diag", Obj().num("n", pd.n).num("rms_xy", pd.rms_xy).num("max_xy", pd.max_xy).num("last_xy", pd.last_xy)
                                .num("rms_yaw_deg", pd.rms_yaw * 180 / M_PI).num("max_yaw_deg", pd.max_yaw * 180 / M_PI).done())
          .raw("map", Obj().num("res", g.resolution).num("known_m2", n_known * g.resolution * g.resolution).num("free_m2", n_free * g.resolution * g.resolution)
                          .num("occ_m2", n_occ * g.resolution * g.resolution).num("wall_segments", n_seg).raw("vs_ref", mapcmp).done())
          .raw("objects", Obj().num("live", n_live).num("dup_pairs_same_name_0p5m", n_dup).raw("by_state", stj).raw("by_label", labj)
                              .str("note", "live/dup/by_label/by_state = small movable only (structural 0); *_all = every live node incl. furniture/big")
                              .num("live_all", n_live_all).num("dup_pairs_same_name_0p5m_all", n_dup_all).raw("by_label_all", laj).done())
          .raw("events", evj).raw("sessions", sessj).done();
  std::ofstream(out + "/metrics.json") << metrics << "\n";
  std::printf("%s\n", metrics.c_str());

  // 학습 뷰어 재생 판
  if (!sg_out.empty()) {
    sm_stream_stats st{};
    sm_stream_get_stats(c, &st);
    sm_save_dsg_ex(c, (sg_out + "/memory").c_str(), nullptr);
    sm_stream_stop(c);
    cap.stop();
    const std::string meta = Obj().str("run", ep_name).str("skill", "realbag").str("dataset", streams[0].kind).str("streams", snames)
                                 .num("t", t_end).num("steps", double(n_frames)).num("keyframes", double(n_detf)).str("outcome", "realbag")
                                 .str("driver", "human").raw("success", "null").str("map_mode", pose).num("ate_rms", a_est.rms).done();
    std::ofstream(sg_out + "/meta.json") << Obj().str("format", "SGS1").raw("meta", meta).str("robot", robot)
                                                .raw("map_from_world", "[" + jnum(Tu.c) + "," + jnum(-Tu.s) + "," + jnum(-(Tu.c * Tu.tx + Tu.s * Tu.ty)) + "," + jnum(Tu.s * Tu.tx - Tu.c * Tu.ty) + "]")
                                                .raw("gt_path", gtjson).raw("labels", "{}").raw("cams", camjson + "]").num("duration", t_end)
                                                .raw("stream", Obj().num("frames", double(cap.frames)).num("pose", double(cap.by_type[1])).num("map", double(cap.by_type[2]))
                                                                   .num("view", double(cap.by_type[3])).num("joints", double(cap.by_type[4])).num("dropped", double(st.dropped)).done())
                                                .raw("joint_order", limo ? "[\"\",\"\",\"\",\"\",\"\",\"\",\"omx_joint1\",\"omx_joint2\",\"omx_joint3\",\"omx_joint4\",\"omx_joint5\",\"omx_gripper_joint_1\"]" : "[]")
                                                .num("n_objects", n_live).raw("metrics", metrics).done();
    const std::string ep = Obj().num("ep", 0).str("skill", "realbag").str("home", snames).str("task", streams[0].kind).str("stage", "real")
                               .str("outcome", "realbag").str("driver", "human").raw("success", "null").num("t", t_end).num("steps", double(n_frames))
                               .num("path_len", gt_len).str("map_mode", pose).str("replay", sg_out.substr(sg_out.rfind('/') + 1)).done();
    std::ofstream(sg_run + "/episodes.jsonl") << ep << "\n";
    std::ofstream(sg_run + "/run.json") << Obj().num("schema", 1).str("kind", "realbag").str("name", ep_name).str("group", "real_bags").str("trainer", "realbag_run")
                                               .str("note", "공개 실제 로봇 ROS bag → 우리 ovdet + scenemap(" + robot + ", 자세 " + pose + ") 다시 돌린 sgview 스트림")
                                               .str("streams", snames).str("det", D.mode).raw("skills", "[\"realbag\"]").num("ended", 1)
                                               .raw("logged", "{\"progress\": false, \"episodes\": true, \"replays\": true}").done();
    std::fprintf(stderr, "sg: %s (stream %llu frames, dropped %llu, cams %ld)\n", sg_out.c_str(), (unsigned long long)cap.frames,
                 (unsigned long long)st.dropped, cams);
  } else if (!live.empty()) {
    sm_stream_stop(c);
  }
  sm_destroy(c);
  return 0;
}
