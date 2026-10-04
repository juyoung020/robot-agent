// sgrt 구현(include/sgrt.h): ovdet + scenemap + 주기 저장을 한 C ABI 로.
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
#include "ovdet.h"
#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/timing.hpp"
#include "sgrt_clip.hpp"

constexpr int kClosedVocabMax = 200;   // 이 이하 어휘 = 닫힌 어휘 엔진(COCO-80), 기본으로 어휘 전부

struct sgrt {
  sgrt_config cfg{};
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
  // SGRT_STREAM_HZ(기본 5)로 만든다 — 스텝 스레드는 아무것도 기다리지 않는다. 파일 저장(위 saver)과 별개.
  std::thread viewer;
  std::atomic<bool> view_quit{false};
  sgrt_clip::ClipMem clip;          // 물체 영상 임베딩(SGRT_CLIP 이면 켜짐)
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
  // 로봇 고르기: SGRT_SM_CONFIG(sm_create config_json 그대로) > SGRT_ROBOT(r1pro | limo_omx) > 없음(R1, sm_create(NULL) — 옛 동작 그대로)
  std::string smj;
  if (const char* sj = std::getenv("SGRT_SM_CONFIG"); sj && *sj) smj = sj;
  else if (const char* rb = std::getenv("SGRT_ROBOT"); rb && *rb) smj = std::string("{\"robot\": \"") + rb + "\"}";
  s->sm = sm_create(smj.empty() ? nullptr : smj.c_str());
  if (!s->sm) {
    put(err, err_len, ("sgrt_create: sm_create rejected config " + smj + " (SGRT_ROBOT = r1pro | limo_omx)").c_str());
    ovd_destroy(s->det);
    delete s;
    return nullptr;
  }
  if (sm_get_robot(s->sm) != SM_ROBOT_R1PRO)
    std::fprintf(stderr, "[sgrt] robot %d (0 r1pro, 1 limo_omx), proprio >= %d, config %s\n", sm_get_robot(s->sm),
                 sm_proprio_dim(sm_get_robot(s->sm)), smj.c_str());
  if (const char* pm = std::getenv("SGRT_POSE")) {
    const std::string m = pm;
    sm_set_pose_mode(s->sm, m == "gt" ? SM_POSE_GT : m == "odom" ? SM_POSE_ODOM : SM_POSE_SLAM);
  }
  if (const char* pl = std::getenv("SGRT_MAP_POLICY")) sm_set_map_update(s->sm, std::atoi(pl) ? 1 : 0, 0);
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
  s->clip.init(s->out_dir);
  if (const char* rp = std::getenv("SGRT_RECORD")) {
    s->rec = std::fopen(rp, "wb");
    if (s->rec) {
      std::fwrite("SGRC", 1, 4, s->rec);
      wr(s->rec, uint32_t(1));
      std::fprintf(stderr, "[sgrt] recording inputs to %s\n", rp);
    }
  }
  std::fprintf(stderr, "[sgrt] pose mode %d (0 slam, 1 odom, 2 gt), image lag %d\n", sm_get_pose_mode(s->sm), s->image_lag);
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
  if (s->det) ovd_destroy(s->det);
  if (s->sm) sm_destroy(s->sm);
  sgrt_crop::destroy(s->crop);
  sm_snapshot_release(s->map_snap);
  if (s->rec) std::fclose(s->rec);
  delete s;
}

int sgrt_begin(sgrt* s, const char* const* prompt, int32_t n, char* err, size_t err_len) {
  if (!s) return -1;
  // 프롬프트 방식: SGRT_PROMPT=task(과제 이름만) | all(엔진 어휘 전부) | auto(기본: 어휘가 kClosedVocabMax 이하인 닫힌 어휘
  // 엔진 — COCO-80 YOLO-seg — 이면 all, YOLOE 큰 어휘면 task)
  const char* mode = std::getenv("SGRT_PROMPT");
  const std::string m = mode ? mode : "auto";
  const int V = ovd_vocab_size(s->det);
  const bool all = m == "all" || (m != "task" && V <= kClosedVocabMax) || !prompt || n <= 0;
  const int found = ovd_set_prompt(s->det, prompt, n, err, err_len);   // 어휘 밖 이름은 err 에 적히고 번호는 유지(검출 안 됨)
  sm_reset(s->sm);
  s->clip.reset();
  s->labels.clear();
  if (all) {   // 엔진 어휘 전부(순서 = 엔진 번호). 과제 이름 중 어휘 밖의 것은 위 err 에 남음
    ovd_set_prompt(s->det, nullptr, 0, nullptr, 0);
    for (int i = 0; i < V; ++i) s->labels.push_back(ovd_vocab_name(s->det, i));
  } else {
    for (int i = 0; i < n; ++i) s->labels.push_back(prompt[i] ? prompt[i] : "");
  }
  std::vector<const char*> lp;
  for (const auto& l : s->labels) lp.push_back(l.c_str());
  sm_set_labels(s->sm, lp.data(), int(lp.size()));   // 같은 순서 = 검출 cls 가 그대로 이름 번호
  std::fprintf(stderr, "[sgrt] prompt %s: %d labels (task names in vocabulary %d/%d, engine vocabulary %d)\n", all ? "all" : "task",
               int(lp.size()), found, n, V);
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
    s->kf_ms = float(msSince(t1));
    if (d) s->clip.keyframe(im_stamp, rgb, rgb_on_device, row_stride, pix_stride, w, h, d, s->sm);   // 새·좋아진 물체만, 비동기
    s->n_kf++;
    s->n_det = d ? d->n : 0;
  }
  else if (rc == 0 && !rgb && depth_m && w > 0 && h > 0) {
    // 지도 전용 스텝(SGRT_MAP_EVERY): 깊이만 있고 검출 키프레임이 아님 — slam2d 지도만 갱신(스캔 ≈ 0.35 ms + 격자 ≈ 0.06 ms).
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

int sgrt_set_pose_mode(sgrt* s, int32_t mode) { return s ? sm_set_pose_mode(s->sm, mode) : -1; }

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
