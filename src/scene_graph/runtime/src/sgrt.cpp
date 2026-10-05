// sgrt 구현(include/sgrt.h): ovdet + scenemap + 주기 저장을 한 C ABI 로.
#include "ra_paths.h"
#include "sgrt.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "crop.hpp"
#include "objprob_front.hpp"
#include "ovdet.h"
#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/timing.hpp"
#include "sgrt_clip.hpp"
#ifdef SGRT_HAVE_CARTO
#include "slam_carto.h"
#endif

// objprob 앞단(확률 물체 모델, scenemap README "scenemap 확률 모드") — 물체 지도의 유일한 규칙, 늘 켬. 검출 엔진은 이름 없는 분할(ObjectSAM·FastSAM-s).
// 검출마다 SigLIP 2 마스크 임베딩 → 낱말 표 최댓값 이름 + sm_set_det_embeddings, 갱신 뒤 sm_reencode_requests(통째 다시 담기)도 같은 스텝에.
// 낱말 표·라벨 사전·엔진별 매개변수 = realbag_run 과 같은 것(objprob_front.hpp, tools/realbag/objprob_params/<엔진>.json).
struct ObjprobFront {
  sgc_encoder* enc = nullptr;
  sgc_labels* lt = nullptr;
  std::vector<float> text;          // 낱말 줄마다 글 임베딩(L2)
  std::vector<int> text_cls;        // 낱말 줄 → 라벨 번호
  std::string obj_kv, label_prior, params_file;
  std::vector<float> emb, remb;     // 이 keyframe 검출 임베딩, 다시 담기 결과
  std::vector<int32_t> cls;
  std::vector<sgc_item> items;
  std::vector<sgc_result> res = std::vector<sgc_result>(64);
  std::vector<uint32_t> rbits, rids;
  std::vector<sm_reenc_req> rreq;
  double enc_ms = 0, reenc_ms = 0;
  int64_t n_kf = 0, n_enc = 0, n_reenc = 0;
  // items 를 모두 넣고 결과를 out[id] 에(id = 요청 번호). on_res(id, emb) 는 결과마다
  template <class F>
  bool run(double stamp, const sgc_frame& fr, int n, std::vector<float>* out, F on_res) {
    out->assign(size_t(n) * SGC_DIM, 0.f);
    int sent = 0, got = 0;
    while (got < n) {
      if (sent < n) {
        const int k = sgc_submit(enc, stamp, &fr, items.data() + sent, n - sent);
        if (k < 0) return false;
        sent += k;
      }
      const int r = sgc_poll(enc, res.data(), int(res.size()), 1);
      for (int q = 0; q < r; ++q) {
        const sgc_result& x = res[size_t(q)];
        std::copy(x.emb, x.emb + SGC_DIM, out->begin() + size_t(x.id) * SGC_DIM);
        on_res(x.id, x.emb);
      }
      got += r;
    }
    return true;
  }
  ~ObjprobFront() {
    if (enc) sgc_destroy(enc);
    if (lt) sgc_labels_close(lt);
  }
};   // 이 이하 어휘 = 닫힌 어휘 엔진(COCO-80), 기본으로 어휘 전부

struct sgrt {
  sgrt_config cfg{};
#ifdef SGRT_HAVE_CARTO
  sc_ctx* carto = nullptr;          // Cartographer(SGRT_POSE 기본 = slam·carto) — 스캔 + 오도메트리 → 스텝마다 sm_push_ext_pose
#endif
  int64_t n_scans = 0;
  bool warned_noscan = false;
  std::string out_dir;
  std::vector<std::string> labels;   // 이번 판 이름 표(scenemap labels)
  OvdHandle* det = nullptr;
  sm_ctx* sm = nullptr;
  int64_t step = 0;
  double last_save = -1e9;
  int32_t n_kf = 0, n_det = 0;
  float det_ms = 0, save_ms = 0;
  float kf_ms = 0, crop_ms = 0;   // 마지막 keyframe: scenemap 갱신 전체(자르기 포함), best view 자르기(장치 → 호스트)
  int32_t n_crops = 0;
  int32_t n_png = 0;              // 마지막 저장에서 쓴 PNG 수
  int32_t n_ply = 0;
  float gather_ms = 0;            // 마지막 keyframe: 구름 점 색 모으기(장치 → 호스트)
  int32_t n_points = 0;
  sgrt_crop::Gpu* crop = nullptr; // 처음 장치 영상이 올 때 만듦
  sm_snapshot_t* map_snap = nullptr; // sgrt_map 이 넘긴 포인터의 주인
  std::vector<float> movable;         // sgrt_map: 옮길 수 있는 물체 x, y, r
  // 영상 stamp = 직전 스텝 stamp(SGRT_IMAGE_LAG)
  int image_lag = 1;             // 0..7 스텝
  double prev_stamp = -1;
  double stamps[8] = {0};         // 최근 스텝 시각(고리)
  int64_t n_stamps = 0;
  // 기록(SGRT_RECORD)
  FILE* rec = nullptr;
  std::vector<uint8_t> rec_rgb;
  // sgrt 단계 시간: det, step, map, record
  scenemap::Timings tm;
  // 주기 저장은 저장 스레드에서(PNG·JSON·파일 쓰기가 스텝을 막지 않게). SGRT_SAVE_SYNC=1 이면 예전처럼 스텝 안에서
  bool save_async = true;
  std::thread saver;
  std::mutex smu;
  std::condition_variable scv;
  bool save_req = false, save_quit = false, save_busy = false;
  int32_t n_save_skipped = 0;
  // 실시간 스트림(SGRT_STREAM=host:port): 자세·지도 변화분은 sm_push_* 안에서 링에 바로 들어가고, 물체·방·그래프 요약은 이 스레드가
  // SGRT_STREAM_HZ(기본 60, 0.5–240 으로 자름)로 만든다 — 스텝 스레드는 아무것도 기다리지 않는다. 파일 저장(위 saver)과 별개.
  std::thread viewer;
  std::atomic<bool> view_quit{false};
  sgrt_clip::ClipMem clip;          // 물체 영상 임베딩 캐시(SGRT_CLIP — objprob 판에서는 init 하지 않아 꺼져 있음; 물체 벡터는 scenemap 이 μ 로 저장)
  ObjprobFront op;                  // objprob 앞단
  std::vector<std::string> name_buf;  // sgrt_object_names 문자열
};

namespace {
void put(char* err, size_t n, const char* msg) {
  if (err && n) std::snprintf(err, n, "%s", msg);
}
double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
enum { kSgDet = 0, kSgStep, kSgMap, kSgRecord, kSgCount };
const char* const kSgNames[kSgCount] = {"det", "step", "map", "record"};

// 기록 파일: 머리 "SGRC" u32 판 1, 그 뒤 레코드(꼬리표 1 바이트):
//   'P' f64 stamp, i32 n, f32[n] proprio
//   'L' f64 stamp(마지막 광선), i32 n, f64 angle_min, angle_inc, time_inc, range_min, range_max, f32[n] 거리(2D 라이다, 10-06)
//   'G' f64 stamp, f64 x, y, yaw                                   (외부 자세)
//   'I' f64 stamp(스텝 시각, 늦춤 전), i32 w, h, f64 fx, fy, cx, cy, f32[w·h] 깊이 m, u8 has_rgb, [u8[w·h·3] RGB],
//       i32 n, img_w, img_h, mask_w, mask_h, f32 sx, sy, ox, oy, i32[n] cls, f32[n] score, f32[4n] box, u32[n·words] mask
template <class T>
void wr(FILE* f, const T& v) { std::fwrite(&v, sizeof(T), 1, f); }
void recImage(sgrt* s, double stamp, const uint8_t* rgb, int on_dev, int64_t rs, int ps, int w, int h, const float* depth,
              double fx, double fy, double cx, double cy, const sm_detections* d) {
  FILE* f = s->rec;
  std::fputc('I', f);
  wr(f, stamp); wr(f, int32_t(w)); wr(f, int32_t(h)); wr(f, fx); wr(f, fy); wr(f, cx); wr(f, cy);
  std::fwrite(depth, sizeof(float), size_t(w) * h, f);
  s->rec_rgb.resize(size_t(w) * h * 3);
  bool ok = false;
  if (rgb && ps >= 3) {
    if (on_dev) {
      ok = cudaMemcpy2D(s->rec_rgb.data(), size_t(w) * 3, rgb, size_t(rs), size_t(w) * 3, size_t(h), cudaMemcpyDeviceToHost) == cudaSuccess;
      if (ok && ps != 3) ok = false;   // 장치 RGBA 는 아래에서 한 번 더
      if (!ok && ps == 4) {
        std::vector<uint8_t> tmp(size_t(w) * h * 4);
        ok = cudaMemcpy2D(tmp.data(), size_t(w) * 4, rgb, size_t(rs), size_t(w) * 4, size_t(h), cudaMemcpyDeviceToHost) == cudaSuccess;
        for (size_t i = 0; ok && i < size_t(w) * h; ++i)
          for (int k = 0; k < 3; ++k) s->rec_rgb[3 * i + k] = tmp[4 * i + k];
      }
    } else {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          for (int k = 0; k < 3; ++k) s->rec_rgb[(size_t(y) * w + x) * 3 + k] = rgb[y * rs + int64_t(x) * ps + k];
      ok = true;
    }
  }
  std::fputc(ok ? 1 : 0, f);
  if (ok) std::fwrite(s->rec_rgb.data(), 1, s->rec_rgb.size(), f);
  const int n = d ? d->n : 0;
  wr(f, int32_t(n));
  wr(f, int32_t(d ? d->img_w : 0)); wr(f, int32_t(d ? d->img_h : 0));
  wr(f, int32_t(d ? d->mask_w : 0)); wr(f, int32_t(d ? d->mask_h : 0));
  wr(f, d ? d->mask_sx : 0.f); wr(f, d ? d->mask_sy : 0.f); wr(f, d ? d->mask_ox : 0.f); wr(f, d ? d->mask_oy : 0.f);
  if (n) {
    const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
    std::fwrite(d->cls, 4, n, f);
    std::vector<float> sc(n, 1.f);
    if (d->score) std::copy(d->score, d->score + n, sc.begin());
    std::fwrite(sc.data(), 4, n, f);
    std::fwrite(d->box, 4, size_t(4) * n, f);
    std::fwrite(d->mask_bits, 4, words * n, f);
  }
}
// sm_crop_fn: 이 keyframe 의 머리 RGB(장치 또는 호스트)에서 best view 상자만 자름
struct CropSrc {
  sgrt* s;
  const uint8_t* rgb;
  int64_t rs;
  int ps;
  int on_device;
  int w, h;
};
int cropCb(void* user, const sm_crop_req* reqs, int32_t n) {
  auto* c = static_cast<CropSrc*>(user);
  const auto t0 = std::chrono::steady_clock::now();
  int rc = 0;
  if (c->on_device) {
    if (!c->s->crop) c->s->crop = sgrt_crop::create();
    rc = c->s->crop ? sgrt_crop::run(c->s->crop, c->rgb, c->rs, c->ps, reqs, n) : -1;
  } else {
    for (int k = 0; k < n; ++k) scenemap::cropRgbHost(c->rgb, c->rs, c->ps, reqs[k]);
  }
  c->s->crop_ms += float(msSince(t0));
  c->s->n_crops += n;
  return rc;
}
// sm_gather_fn: 구름 점 색(남긴 화소만)
int gatherCb(void* user, const int32_t* xy, int32_t n, uint8_t* rgb) {
  auto* c = static_cast<CropSrc*>(user);
  const auto t0 = std::chrono::steady_clock::now();
  int rc = 0;
  if (c->on_device) {
    if (!c->s->crop) c->s->crop = sgrt_crop::create();
    rc = c->s->crop ? sgrt_crop::gather(c->s->crop, c->rgb, c->rs, c->ps, c->w, c->h, xy, n, rgb) : -1;
  } else {
    scenemap::gatherRgbHost(c->rgb, c->rs, c->ps, c->w, c->h, xy, n, rgb);
  }
  c->s->gather_ms += float(msSince(t0));
  c->s->n_points += n;
  return rc;
}
// SigLIP 2·낱말 표·매개변수 파일
bool objprobInit(sgrt* s, char* err, size_t err_len) {
  const char* cp = std::getenv("SGRT_OBJPROB_CLIP");
  const std::string clip_plan = cp && *cp ? cp : ra::models() + "/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan";
  const char* lp = std::getenv("SGRT_LABELS");
  const std::string ldir = lp && *lp ? lp : ra::labels();
  sgc_config cc;
  sgc_default_config(&cc);
  cc.engine = clip_plan.c_str();
  s->op.enc = sgc_create(&cc, err, err_len);
  if (!s->op.enc) return false;
  s->op.lt = sgc_labels_open(ldir.c_str(), nullptr, err, err_len);
  if (!s->op.lt) return false;
  // 엔진별 매개변수: SGRT_OBJPROB_PARAMS = 파일 | none, 없으면 <매개변수 폴더>/<엔진 줄기>.json
  const char* pe = std::getenv("SGRT_OBJPROB_PARAMS");
  std::string pf = pe ? pe : "";
  if (pf.empty()) {
    const std::string stem = objprob_front::engineStem(s->cfg.engine);
    for (const char* d : {SGRT_PARAMS_DIR, SGRT_PARAMS_DIR2}) {
      const std::string c = std::string(d) + "/" + stem + ".json";
      if (std::FILE* f = std::fopen(c.c_str(), "r")) { std::fclose(f); pf = c; break; }
    }
    if (pf.empty()) std::fprintf(stderr, "[sgrt] objprob: no params file for %s (built-in defaults)\n", stem.c_str());
  }
  if (!pf.empty() && pf != "none") {
    if (!objprob_front::readParamsFile(pf, &s->op.obj_kv, &s->op.label_prior)) {
      put(err, err_len, ("sgrt objprob: cannot read params " + pf).c_str());
      return false;
    }
    s->op.params_file = pf;
  }
  std::fprintf(stderr, "[sgrt] objprob on: SigLIP 2 %s, labels %s, params %s (label prior %s)\n", clip_plan.c_str(), ldir.c_str(),
               pf.empty() ? "built-in" : pf.c_str(), s->op.label_prior.empty() ? "uniform" : s->op.label_prior.c_str());
  return true;
}
// objprob 판 시작: 낱말 표(+ 과제 이름 중 낱말 표에 없는 것) → 라벨 → scenemap 확률 모드
int objprobBegin(sgrt* s, const char* const* prompt, int32_t n, char* err, size_t err_len) {
  ObjprobFront& o = s->op;
  ovd_set_prompt(s->det, nullptr, 0, nullptr, 0);   // 분할 엔진: 'object' 하나
  s->labels.clear();
  o.text.clear();
  o.text_cls.clear();
  std::vector<std::pair<std::string, std::string>> extra;
  for (int i = 0; i < n && prompt; ++i) {
    if (!prompt[i] || !*prompt[i]) continue;
    bool have = false;
    for (const auto& w : objprob_front::kVocab) have = have || std::strcmp(w.text, prompt[i]) == 0 || std::strcmp(w.label, prompt[i]) == 0;
    for (const auto& w : objprob_front::kVocabAp) have = have || std::strcmp(w.text, prompt[i]) == 0;
    if (!have) extra.emplace_back(prompt[i], prompt[i]);
  }
  std::string missing;
  objprob_front::buildTextTable(o.lt, true, extra, &s->labels, &o.text, &o.text_cls, &missing);
  objprob_front::addApLabels(&s->labels);
  std::vector<const char*> lp;
  for (const auto& l : s->labels) lp.push_back(l.c_str());
  sm_set_labels(s->sm, lp.data(), int(lp.size()));
  std::string er;
  if (!objprob_front::enable(s->sm, &s->labels, o.text, o.text_cls, o.label_prior, o.obj_kv, &er)) {
    put(err, err_len, er.c_str());
    return -1;
  }
  std::fprintf(stderr, "[sgrt] objprob: %zu words -> %zu labels (task words added %zu; not in SigLIP table:%s)\n", o.text_cls.size(),
               s->labels.size(), extra.size(), missing.c_str());
  return 0;
}
// keyframe: 검출마다 SigLIP 2 마스크 임베딩 → 이름(낱말 최댓값) → d2(cls 바꾼 사본) + sm_set_det_embeddings
bool objprobName(sgrt* s, double stamp, const uint8_t* rgb, int on_dev, int64_t rs, int ps, int w, int h, const sm_detections* d,
                 sm_detections* d2) {
  ObjprobFront& o = s->op;
  const auto t0 = std::chrono::steady_clock::now();
  *d2 = *d;
  o.cls.assign(size_t(std::max(0, d->n)), 0);
  if (d->n > 0) {
    const sgc_frame fr{rgb, on_dev, rs, ps, w, h, d->mask_w, d->mask_h, d->mask_sx, d->mask_sy, d->mask_ox, d->mask_oy, d->mask_bits};
    o.items.resize(size_t(d->n));
    for (int k = 0; k < d->n; ++k) o.items[size_t(k)] = sgc_item{uint32_t(k), k, {d->box[4 * k], d->box[4 * k + 1], d->box[4 * k + 2], d->box[4 * k + 3]}, 1.f};
    const bool ok = o.run(stamp, fr, d->n, &o.emb, [&](uint32_t id, const float* e) {
      int best = 0;
      float bs = -2.f;
      for (size_t p = 0; p < o.text_cls.size(); ++p) {
        const float* tp = o.text.data() + p * SGC_DIM;
        float sd = 0.f;
        for (int j = 0; j < SGC_DIM; ++j) sd += e[j] * tp[j];
        if (sd > bs) { bs = sd; best = int(p); }
      }
      o.cls[id] = o.text_cls[size_t(best)];
    });
    if (!ok) return false;
    sm_set_det_embeddings(s->sm, o.emb.data(), d->n, SGC_DIM);
    o.n_enc += d->n;
  }
  d2->cls = o.cls.data();
  o.enc_ms += msSince(t0);
  ++o.n_kf;
  return true;
}
// 갱신 뒤: 통째 다시 담기(합친 물체·더 좋은 모습을 구름 투영 마스크로)
void objprobReenc(sgrt* s, double stamp, const uint8_t* rgb, int on_dev, int64_t rs, int ps, int w, int h, const sm_detections* d) {
  ObjprobFront& o = s->op;
  const sm_reenc_req* rq = nullptr;
  const uint32_t* rb = nullptr;
  const int nr = sm_reencode_requests(s->sm, &rq, &rb);
  if (nr <= 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
  o.rbits.assign(rb, rb + size_t(nr) * words);
  o.rreq.assign(rq, rq + nr);
  const sgc_frame fr{rgb, on_dev, rs, ps, w, h, d->mask_w, d->mask_h, d->mask_sx, d->mask_sy, d->mask_ox, d->mask_oy, o.rbits.data()};
  o.items.resize(size_t(nr));
  o.rids.resize(size_t(nr));
  for (int k = 0; k < nr; ++k) {
    const sm_reenc_req& q = o.rreq[size_t(k)];
    o.items[size_t(k)] = sgc_item{uint32_t(k), k, {q.box[0], q.box[1], q.box[2], q.box[3]}, q.kappa};
    o.rids[size_t(k)] = q.id;
  }
  if (o.run(stamp, fr, nr, &o.remb, [](uint32_t, const float*) {}))
    sm_set_object_embeddings(s->sm, o.rids.data(), o.remb.data(), nr, SGC_DIM);
  o.reenc_ms += msSince(t0);
  o.n_reenc += nr;
}
}  // namespace

extern "C" {

void sgrt_default_config(sgrt_config* c) {
  if (!c) return;
  *c = sgrt_config{};
  c->kf_every = 6;
  c->save_s = 1.0;
  c->conf_th = 0.25f;
}

sgrt* sgrt_create(const sgrt_config* c, char* err, size_t err_len) {
  if (!c || !c->engine || !c->names || !c->out_dir) {
    put(err, err_len, "sgrt_create: engine, names and out_dir are required");
    return nullptr;
  }
  auto* s = new sgrt();
  s->cfg = *c;
  s->out_dir = c->out_dir;
  if (s->cfg.kf_every < 1) s->cfg.kf_every = 1;
  OvdConfig oc;
  ovd_default_config(&oc);
  oc.seg_engine = c->engine;
  oc.names = c->names;
  if (c->conf_th > 0) oc.conf_th = c->conf_th;
  s->det = ovd_create(&oc, err, err_len);
  if (!s->det) {
    delete s;
    return nullptr;
  }
  // 로봇 고르기: SGRT_SM_CONFIG(sm_create config_json 그대로) > SGRT_ROBOT(limo_omx) > 없음(LIMO + OMX-F, sm_create(NULL))
  std::string smj;
  if (const char* sj = std::getenv("SGRT_SM_CONFIG"); sj && *sj) smj = sj;
  else if (const char* rb = std::getenv("SGRT_ROBOT"); rb && *rb) smj = std::string("{\"robot\": \"") + rb + "\"}";
  s->sm = sm_create(smj.empty() ? nullptr : smj.c_str());
  if (!s->sm) {
    put(err, err_len, ("sgrt_create: sm_create rejected config " + smj + " (SGRT_ROBOT = limo_omx)").c_str());
    ovd_destroy(s->det);
    delete s;
    return nullptr;
  }
  if (!objprobInit(s, err, err_len)) {
    sm_destroy(s->sm);
    ovd_destroy(s->det);
    delete s;
    return nullptr;
  }
  {
    const char* pm = std::getenv("SGRT_POSE");
    const std::string m = pm && *pm ? pm : "carto";
    const int mode = m == "gt" ? SM_POSE_GT : m == "odom" ? SM_POSE_ODOM : SM_POSE_EXT;
    if (mode == SM_POSE_EXT && m != "carto")
      std::fprintf(stderr, "[sgrt] SGRT_POSE=%s unknown -> carto (carto|odom|gt)\n", m.c_str());
    sgrt_set_pose_mode(s, mode);
  }
  if (const char* pl = std::getenv("SGRT_MAP_POLICY")) sm_set_map_update(s->sm, std::atoi(pl) ? 1 : 0, 0);
  if (const char* ip = std::getenv("SGRT_INSPECT"); ip && std::atoi(ip)) sm_set_inspect(s->sm, 1);   // 살펴본 정도(view.json·scene.json "inspect")
  if (const char* lg = std::getenv("SGRT_IMAGE_LAG")) s->image_lag = std::clamp(std::atoi(lg), 0, 7);
  if (const char* ss = std::getenv("SGRT_SAVE_SYNC")) s->save_async = std::atoi(ss) == 0;
  if (s->save_async)
    s->saver = std::thread([s] {
      std::unique_lock<std::mutex> lk(s->smu);
      for (;;) {
        s->scv.wait(lk, [s] { return s->save_req || s->save_quit; });
        if (s->save_quit && !s->save_req) return;
        s->save_req = false;
        s->save_busy = true;
        lk.unlock();
        sgrt_save(s);
        lk.lock();
        s->save_busy = false;
      }
    });
  if (const char* st = std::getenv("SGRT_STREAM")) {
    if (sm_stream_start(s->sm, st) == 0) {
      double hz = 60.0;
      if (const char* h = std::getenv("SGRT_STREAM_HZ")) hz = std::clamp(std::atof(h), 0.5, 240.0);
      const auto period = std::chrono::microseconds(int64_t(1e6 / hz));
      s->viewer = std::thread([s, period] {
        auto t_log = std::chrono::steady_clock::now();
        sm_stream_stats prev{};
        while (!s->view_quit.load(std::memory_order_relaxed)) {
          sm_stream_view(s->sm);
          std::this_thread::sleep_for(period);
          const auto now = std::chrono::steady_clock::now();
          if (now - t_log >= std::chrono::seconds(5)) {   // 5 s 마다 스트림 통계(로그): 보낸 양·버려진 것·요약 만드는 시간
            sm_stream_stats ss{};
            if (sm_stream_get_stats(s->sm, &ss) == 0) {
              const double dt = std::chrono::duration<double>(now - t_log).count();
              std::fprintf(stderr, "[sgrt] stream %.0fs: %.0f frames/s in, %llu dropped, %.2f MB/s sent, summaries built %llu skipped %llu (build %.0f us)\n", dt,
                           double(ss.frames_in - prev.frames_in) / dt, (unsigned long long)ss.dropped, double(ss.bytes_sent - prev.bytes_sent) / dt / 1e6,
                           (unsigned long long)(ss.views_built - prev.views_built), (unsigned long long)(ss.views_skipped - prev.views_skipped), ss.view_build_us);
            }
            prev = ss;
            t_log = now;
          }
        }
      });
      std::fprintf(stderr, "[sgrt] streaming to %s (view %.1f Hz)\n", st, 1e6 / double(period.count()));
    }
  }
  if (const char* rp = std::getenv("SGRT_RECORD")) {
    s->rec = std::fopen(rp, "wb");
    if (s->rec) {
      std::fwrite("SGRC", 1, 4, s->rec);
      wr(s->rec, uint32_t(1));
      std::fprintf(stderr, "[sgrt] recording inputs to %s\n", rp);
    }
  }
  std::fprintf(stderr, "[sgrt] pose mode %d (1 odom, 2 gt, 3 Cartographer), image lag %d\n", sm_get_pose_mode(s->sm), s->image_lag);
  return s;
}

void sgrt_destroy(sgrt* s) {
  if (!s) return;
  if (s->viewer.joinable()) { s->view_quit = true; s->viewer.join(); }
  if (s->sm) {
    sm_stream_stats ss{};
    if (sm_stream_get_stats(s->sm, &ss) == 0 && (ss.frames_in || ss.reconnects))
      std::fprintf(stderr, "[sgrt] stream: %llu frames in, %llu dropped, %llu bytes sent, %llu connects, summary build %.0f us\n", (unsigned long long)ss.frames_in,
                   (unsigned long long)ss.dropped, (unsigned long long)ss.bytes_sent, (unsigned long long)ss.reconnects, ss.view_build_us);
    sm_stream_stop(s->sm);
  }
  if (s->saver.joinable()) {
    {
      std::lock_guard<std::mutex> lk(s->smu);
      s->save_quit = true;
    }
    s->scv.notify_all();
    s->saver.join();
  }
  if (s->op.n_kf)
    std::fprintf(stderr, "[sgrt] objprob: %lld keyframes, SigLIP 2 %.2f ms/kf (%.1f masks/kf), whole re-encode %.2f ms/kf (%lld)\n",
                 (long long)s->op.n_kf, s->op.enc_ms / double(s->op.n_kf), double(s->op.n_enc) / double(s->op.n_kf),
                 s->op.reenc_ms / double(s->op.n_kf), (long long)s->op.n_reenc);
  if (s->det) ovd_destroy(s->det);
#ifdef SGRT_HAVE_CARTO
  if (s->carto) {
    sc_stats cs{};
    sc_get_stats(s->carto, &cs);
    std::fprintf(stderr, "[sgrt] Cartographer: %lld scans, %lld nodes, %lld submaps, %lld loop constraints, push %.0f us/scan\n", (long long)cs.n_scans,
                 (long long)cs.n_nodes, (long long)cs.n_submaps, (long long)cs.n_loop_constraints, cs.us_scan_mean);
    sc_destroy(s->carto);
  }
#endif
  if (s->sm) sm_destroy(s->sm);
  sgrt_crop::destroy(s->crop);
  sm_snapshot_release(s->map_snap);
  if (s->rec) std::fclose(s->rec);
  delete s;
}

namespace {
// 새 판: Cartographer 도 새 궤적(지도·자세 원점을 scenemap 과 같이 비움)
void cartoReset(sgrt* s) {
#ifdef SGRT_HAVE_CARTO
  if (!s->carto) return;
  sc_destroy(s->carto);
  s->carto = nullptr;
  s->n_scans = 0;
  s->warned_noscan = false;
  sgrt_set_pose_mode(s, SM_POSE_EXT);
#else
  (void)s;
#endif
}
}  // namespace

int sgrt_begin(sgrt* s, const char* const* prompt, int32_t n, char* err, size_t err_len) {
  if (!s) return -1;
  // 이름은 SigLIP 2 낱말 표(과제 이름을 더함), 검출 엔진은 'object' 하나
  sm_reset(s->sm);
  cartoReset(s);
  s->clip.reset();
  if (objprobBegin(s, prompt, n, err, err_len) != 0) return -1;
  s->step = 0;
  s->prev_stamp = -1;
  s->n_stamps = 0;
  s->last_save = -1e9;
  s->n_kf = s->n_det = 0;
  return 0;
}

int sgrt_set_kind_names(sgrt* s, int32_t kind, const char* const* names, int32_t n) {
  return s ? sm_set_kind_names(s->sm, kind, names, n) : -1;
}

int sgrt_want_image(const sgrt* s) { return s && (s->step % s->cfg.kf_every) == 0; }

int sgrt_step(sgrt* s, double stamp, const float* proprio, int32_t n_proprio, const uint8_t* rgb, int32_t rgb_on_device,
              int64_t row_stride, int32_t pix_stride, int32_t w, int32_t h, const float* depth_m, double fx, double fy, double cx,
              double cy) {
  if (!s || !proprio) return -1;
  const auto t_step = std::chrono::steady_clock::now();
  if (s->rec) {
    std::fputc('P', s->rec);
    wr(s->rec, stamp);
    wr(s->rec, int32_t(n_proprio));
    std::fwrite(proprio, 4, size_t(std::max(0, n_proprio)), s->rec);
  }
#ifdef SGRT_HAVE_CARTO
  if (s->carto && n_proprio >= SM_LIMO_PROPRIO_DIM) {   // 바퀴 오도메트리 → Cartographer, 이 스텝 자세 → scenemap(EXT)
    sc_push_odom(s->carto, stamp, proprio[SM_LIMO_ODOM_X], proprio[SM_LIMO_ODOM_Y], proprio[SM_LIMO_ODOM_YAW]);
    double cp[3];
    if (sc_pose_at(s->carto, stamp, cp) == 0) {
      const sm_pose2 ep{stamp, cp[0], cp[1], cp[2]};
      sm_push_ext_pose(s->sm, &ep);
    } else if (!s->n_scans && !s->warned_noscan && s->step > 90) {
      s->warned_noscan = true;
      std::fprintf(stderr, "[sgrt] Cartographer: no lidar scans (sgrt_push_scan) after %lld steps -> pose = wheel odometry only "
                           "(no SLAM: feed the 2D lidar)\n", (long long)s->step);
    }
  }
#endif
  sm_proprio p{stamp, proprio, n_proprio};
  int rc = sm_push_proprio(s->sm, &p);
  // 영상 k = 장면 k-1: 직전 스텝 시각(첫 스텝은 자기 시각)
  s->stamps[s->n_stamps % 8] = stamp;
  ++s->n_stamps;
  const int64_t back = std::min<int64_t>(s->image_lag, s->n_stamps - 1);
  const double im_stamp = s->stamps[(s->n_stamps - 1 - back) % 8];
  s->prev_stamp = stamp;
  if (rc == 0 && rgb && depth_m && w > 0 && h > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    OvdImage im{};
    im.stamp = stamp;
    im.cam = 0;
    im.data = rgb;
    im.w = w;
    im.h = h;
    im.row_stride = row_stride;
    im.pix_stride = pix_stride;
    im.on_device = rgb_on_device;
    const sm_detections* d = ovd_detect(s->det, &im, nullptr);
    sm_detections dn{};
    if (d) {   // objprob: 마스크마다 SigLIP 2 → 이름·임베딩(검출 시간에 넣음)
      if (!objprobName(s, stamp, rgb, rgb_on_device, row_stride, pix_stride, w, h, d, &dn)) {
        std::fprintf(stderr, "[sgrt] objprob: SigLIP 2 submit failed\n");
        return -1;
      }
      d = &dn;
    }
    s->det_ms = float(msSince(t0));
    s->tm.h[kSgDet].add(s->det_ms * 1e3);
    if (s->rec) {
      const auto tr = std::chrono::steady_clock::now();
      recImage(s, stamp, rgb, rgb_on_device, row_stride, pix_stride, w, h, depth_m, fx, fy, cx, cy, d);
      s->tm.h[kSgRecord].add(msSince(tr) * 1e3);
    }
    sm_image si{};
    si.stamp = im_stamp;
    si.cam = 0;
    si.w = w;
    si.h = h;
    si.depth_m = depth_m;
    si.fx = fx; si.fy = fy; si.cx = cx; si.cy = cy;
    const auto t1 = std::chrono::steady_clock::now();
    s->crop_ms = s->gather_ms = 0;
    s->n_crops = s->n_points = 0;
    CropSrc cs{s, rgb, row_stride, pix_stride, rgb_on_device, w, h};
    const sm_rgb_source src{&cropCb, &gatherCb, &cs};
    rc = d ? sm_push_image_rgb(s->sm, &si, d, &src) : sm_push_image(s->sm, &si, nullptr);
    if (d) objprobReenc(s, im_stamp, rgb, rgb_on_device, row_stride, pix_stride, w, h, d);
    s->kf_ms = float(msSince(t1));
    if (d) s->clip.keyframe(im_stamp, rgb, rgb_on_device, row_stride, pix_stride, w, h, d, s->sm);   // 새·좋아진 물체만, 비동기
    s->n_kf++;
    s->n_det = d ? d->n : 0;
  }
  else if (rc == 0 && !rgb && depth_m && w > 0 && h > 0) {
    // 지도 전용 스텝(SGRT_MAP_EVERY): 깊이만 있고 검출 키프레임이 아님 — 격자만 갱신(스캔 ≈ 0.35 ms + 격자 ≈ 0.06 ms).
    // 물체 지도·검출·임베딩은 그대로 키프레임(kf_every)에서만. 지도(와 스트림 지도 영역)가 키프레임 주기가 아니라 이 주기로 갱신된다.
    sm_image si{};
    si.stamp = im_stamp;
    si.cam = 0;
    si.w = w;
    si.h = h;
    si.depth_m = depth_m;
    si.fx = fx; si.fy = fy; si.cx = cx; si.cy = cy;
    rc = sm_push_image(s->sm, &si, nullptr);
  }
  s->clip.poll();
  s->step++;
  if (stamp - s->last_save >= s->cfg.save_s) {
    s->last_save = stamp;
    if (s->save_async) {   // 저장 스레드에 맡김(앞 저장이 아직이면 이번 것은 건너뜀 — 다음 주기에 최신으로)
      std::lock_guard<std::mutex> lk(s->smu);
      if (s->save_busy || s->save_req) ++s->n_save_skipped;
      else s->save_req = true;
      s->scv.notify_one();
    } else {
      sgrt_save(s);
    }
  }
  s->tm.h[kSgStep].add(msSince(t_step) * 1e3);
  return rc;
}

int sgrt_save(sgrt* s) {
  if (!s) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  sm_save_stats st{};
  s->clip.save(s->out_dir, s->sm);   // emb·이름 캐시 → sm_set_object_meta(아래 scene.json 에 들어감)
  const int rc = sm_save_dsg_ex(s->sm, s->out_dir.c_str(), &st);   // 바뀐 best view 만 PNG 로
  s->save_ms = float(msSince(t0));
  s->n_png = st.n_png;
  s->n_ply = st.n_ply;
  return rc;
}

void sgrt_stats(const sgrt* s, int32_t* n_kf, int32_t* n_det, int32_t* n_obj, float* det_ms, float* save_ms) {
  if (!s) return;
  if (n_kf) *n_kf = s->n_kf;
  if (n_det) *n_det = s->n_det;
  if (n_obj) {
    sm_snapshot_t* snap = nullptr;
    *n_obj = sm_snapshot(s->sm, &snap) == 0 ? sm_snap_status(snap).n_objects : 0;
    sm_snapshot_release(snap);
  }
  if (det_ms) *det_ms = s->det_ms;
  if (save_ms) *save_ms = s->save_ms;
}

void sgrt_get_timing(const sgrt* s, sgrt_timing* t) {
  if (!s || !t) return;
  *t = sgrt_timing{};
  t->det_ms = s->det_ms;
  t->kf_ms = s->kf_ms;
  t->crop_ms = s->crop_ms;
  t->save_ms = s->save_ms;
  t->n_crops = s->n_crops;
  t->n_png = s->n_png;
  t->n_ply = s->n_ply;
  t->gather_ms = s->gather_ms;
  t->n_points = s->n_points;
}

int sgrt_map(sgrt* s, sgrt_map_view* out) {
  if (!s || !out) return -1;
  scenemap::ScopedStage tmap(&s->tm, kSgMap);
  int32_t dbox[4] = {0, 0, 0, 0};
  uint64_t ver = 0;
  const int dirty = sm_take_dirty(s->sm, dbox, &ver);   // 스냅숏 앞: 그 사이 insert 는 다음 부름의 상자에 들어감
  sm_snapshot_t* snap = nullptr;
  if (sm_snapshot(s->sm, &snap) != 0 || !snap) return -2;
  sm_snapshot_release(s->map_snap);
  s->map_snap = snap;
  *out = sgrt_map_view{};
  const sm_pose2 p = sm_snap_pose(snap);
  out->stamp = p.stamp;
  out->pose[0] = p.x; out->pose[1] = p.y; out->pose[2] = p.yaw;
  sm_grid g{};
  sm_snap_map(snap, &g);
  out->res = g.resolution;
  out->origin[0] = g.origin[0]; out->origin[1] = g.origin[1];
  out->w = g.width; out->h = g.height;
  out->cells = g.cells;
  sm_room_grid rg{};
  if (sm_snap_room_grid(snap, &rg) == 0 && rg.ids) {
    out->room_res = rg.resolution;
    out->room_origin[0] = rg.origin[0]; out->room_origin[1] = rg.origin[1];
    out->room_w = rg.width; out->room_h = rg.height;
    out->room_ids = rg.ids;
  }
  const sm_room* rooms = nullptr;
  out->n_rooms = sm_snap_rooms(snap, &rooms);
  if (out->n_rooms < 0) out->n_rooms = 0;
  sm_scan2 sc{};
  if (sm_snap_scan(snap, &sc) == 0) {
    out->scan_pose[0] = sc.pose.x; out->scan_pose[1] = sc.pose.y; out->scan_pose[2] = sc.pose.yaw;
    out->scan_origin[0] = sc.ox; out->scan_origin[1] = sc.oy;
    out->n_hit = sc.n_hit; out->hit_x = sc.hx; out->hit_y = sc.hy;
    out->n_free = sc.n_free; out->free_x = sc.fx; out->free_y = sc.fy;
  }
  out->dirty = dirty > 0;
  for (int k = 0; k < 4; ++k) out->dirty_box[k] = dbox[k];
  out->map_version = ver;
  s->movable.clear();
  const sm_object* objs = nullptr;
  const int no = sm_snap_objects(snap, &objs);
  for (int k = 0; k < no; ++k) {
    if (objs[k].structural || sm_snap_movable(snap, objs[k].id) != 1 || objs[k].state == SM_HELD) continue;
    s->movable.push_back(float(objs[k].pos[0]));
    s->movable.push_back(float(objs[k].pos[1]));
    s->movable.push_back(float(0.5 * std::hypot(objs[k].extent[0], objs[k].extent[1])));
  }
  out->n_movable = int32_t(s->movable.size() / 3);
  out->movable_xyr = s->movable.empty() ? nullptr : s->movable.data();
  return 0;
}

sm_snapshot_t* sgrt_map_snapshot(sgrt* s) { return s ? s->map_snap : nullptr; }

sm_ctx* sgrt_scenemap(sgrt* s) { return s ? s->sm : nullptr; }

int sgrt_set_pose_mode(sgrt* s, int32_t mode) {
  if (!s) return -1;
#ifdef SGRT_HAVE_CARTO
  if (mode == SM_POSE_EXT && !s->carto) {
    sc_config cc = sc_default_config();
    const char* cn = std::getenv("SGRT_CARTO_CONFIG");
    if (cn && *cn) cc.config_name = cn;
    if (const char* ls = std::getenv("SGRT_LASER"))
      std::sscanf(ls, "%lf,%lf,%lf,%lf", &cc.laser_xyz[0], &cc.laser_xyz[1], &cc.laser_xyz[2], &cc.laser_yaw);
    char er[256] = {0};
    s->carto = sc_create(&cc, er, sizeof er);
    if (!s->carto) {
      std::fprintf(stderr, "[sgrt] Cartographer: %s -> odom\n", er);
      mode = SM_POSE_ODOM;
    } else {
      std::fprintf(stderr, "[sgrt] Cartographer pose source (%s, laser %.3f %.3f %.3f yaw %.3f)\n", cc.config_name ? cc.config_name : "limo_x2l.lua",
                   cc.laser_xyz[0], cc.laser_xyz[1], cc.laser_xyz[2], cc.laser_yaw);
    }
  }
  if (mode != SM_POSE_EXT && s->carto) { sc_destroy(s->carto); s->carto = nullptr; }
#else
  if (mode == SM_POSE_EXT) {
    std::fprintf(stderr, "[sgrt] built without slam_carto (tools/build_all.sh cartographer sgrt) -> odom\n");
    mode = SM_POSE_ODOM;
  }
#endif
  return sm_set_pose_mode(s->sm, mode);
}

int sgrt_push_scan(sgrt* s, double stamp, int32_t n, const float* r, double a0, double da, double dt, double rmin, double rmax) {
  if (!s || !r || n <= 0) return -1;
  if (s->rec) {
    std::fputc('L', s->rec);
    wr(s->rec, stamp); wr(s->rec, n); wr(s->rec, a0); wr(s->rec, da); wr(s->rec, dt); wr(s->rec, rmin); wr(s->rec, rmax);
    std::fwrite(r, 4, size_t(n), s->rec);
  }
  ++s->n_scans;
#ifdef SGRT_HAVE_CARTO
  if (s->carto) return sc_push_scan(s->carto, stamp, n, r, a0, da, dt, rmin, rmax);
#endif
  return 0;
}

int sgrt_set_robot(sgrt* s, int32_t robot) { return s ? sm_set_robot(s->sm, robot) : -1; }

int sgrt_get_robot(const sgrt* s) { return s ? sm_get_robot(s->sm) : -1; }

int sgrt_proprio_dim(const sgrt* s) { return s ? sm_proprio_dim(sm_get_robot(s->sm)) : -1; }

int sgrt_push_pose(sgrt* s, double stamp, double x, double y, double yaw) {
  if (!s) return -1;
  if (s->rec) {
    std::fputc('G', s->rec);
    wr(s->rec, stamp); wr(s->rec, x); wr(s->rec, y); wr(s->rec, yaw);
  }
  const sm_pose2 p{stamp, x, y, yaw};
  return sm_push_pose(s->sm, &p);
}

int sgrt_get_pose_diag(const sgrt* s, sgrt_pose_diag* out) {
  static_assert(sizeof(sgrt_pose_diag) == sizeof(sm_pose_diag), "same layout");
  if (!s || !out) return -1;
  return sm_get_pose_diag(s->sm, reinterpret_cast<sm_pose_diag*>(out));
}

int sgrt_get_stage_timing(const sgrt* s, sgrt_stage_timing* out, int32_t cap) {
  static_assert(sizeof(sgrt_stage_timing) == sizeof(sm_stage_timing), "same layout");
  if (!s) return -1;
  const int n = sm_get_timing(s->sm, reinterpret_cast<sm_stage_timing*>(out), cap);
  if (n < 0) return n;
  for (int k = 0; k < kSgCount; ++k) {
    if (out && n + k < cap) {
      const scenemap::StageHist& h = s->tm.h[k];
      sgrt_stage_timing& o = out[n + k];
      o.name = kSgNames[k];
      o.n = int64_t(h.n);
      o.total_us = h.sum;
      o.mean_us = h.n ? h.sum / double(h.n) : 0;
      o.p50_us = h.quantile(0.5);
      o.p99_us = h.quantile(0.99);
      o.max_us = h.max;
      o.last_us = h.last;
    }
  }
  return n + kSgCount;
}

int sgrt_reset_stage_timing(sgrt* s) {
  if (!s) return -1;
  s->tm.clear();
  return sm_reset_timing(s->sm);
}

}  // extern "C"

extern "C" {

int sgrt_clip_enabled(const sgrt* s) { return s && s->clip.on() ? 1 : 0; }

int sgrt_objprob_enabled(const sgrt* s) { return s ? 1 : 0; }

int sgrt_object_embedding(sgrt* s, uint32_t id, float* out) { return s ? s->clip.embedding(id, out) : -1; }

int sgrt_query_embedding(sgrt* s, const float* q, int32_t k, uint32_t* ids, float* scores) {
  return s ? s->clip.query(q, k, ids, scores, s->sm) : -1;
}

int sgrt_query_label(sgrt* s, const char* text, int32_t k, uint32_t* ids, float* scores) {
  return s ? s->clip.queryText(text, k, ids, scores, s->sm) : -1;
}

int sgrt_object_names(sgrt* s, uint32_t id, sgrt_name* out, int32_t cap, const char** level_en, const char** level_ko, int32_t* structural) {
  if (!s) return -1;
  std::vector<std::pair<std::string, std::string>> nk;
  std::vector<float> sc;
  std::string lv, lk;
  int st = 0;
  if (!s->clip.names(id, &nk, &sc, &lv, &lk, &st)) return 0;
  s->name_buf.clear();
  for (auto& [e, k] : nk) s->name_buf.push_back(e), s->name_buf.push_back(k);
  s->name_buf.push_back(lv);
  s->name_buf.push_back(lk);
  const int n = std::min<int>(cap, int(nk.size()));
  for (int i = 0; i < n && out; ++i) out[i] = sgrt_name{s->name_buf[2 * i].c_str(), s->name_buf[2 * i + 1].c_str(), sc[i]};
  if (level_en) *level_en = s->name_buf[s->name_buf.size() - 2].c_str();
  if (level_ko) *level_ko = s->name_buf.back().c_str();
  if (structural) *structural = st;
  return n;
}

int sgrt_get_clip_stats(const sgrt* s, sgrt_clip_stats* o) {
  if (!s || !o) return -1;
  const sgrt_clip::Stats t = s->clip.stats();
  *o = sgrt_clip_stats{s->clip.on() ? 1 : 0, t.n_objects, t.n_named, t.n_submitted, t.n_done, t.n_dropped, t.last_batch,
                       t.crop_ms, t.net_ms, t.submit_us, t.names_us, t.save_ms};
  return 0;
}

}  // extern "C"
