// sgrt_clip.hpp 구현.
#include "ra_paths.h"
#include "sgrt_clip.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "memstore.hpp"

namespace fs = std::filesystem;

namespace sgrt_clip {
namespace {
double usSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}
constexpr float kImprove = 1.2f;   // best view 품질이 임베딩 때의 1.2 배 이상이면 다시
constexpr int kMaxPerKf = 8;
}  // namespace

ClipMem::~ClipMem() {
  if (loader_.joinable()) loader_.join();
  if (enc_) {
    sgc_poll(enc_, nullptr, 0, 1);
    sgc_destroy(enc_);
  }
  sgc_labels_close(labels_);
  delete static_cast<sgclip::NameCache*>(cache_);
}

bool ClipMem::init(const std::string& out_dir) {
  const char* e = std::getenv("SGRT_CLIP");
  if (!e || !*e || std::string(e) == "0") return false;
  std::string engine = e;
  if (engine == "1") engine = ra::models() + "/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan";
  sgc_config c;
  sgc_default_config(&c);
  c.engine = engine.c_str();
  if (const char* g = std::getenv("SGRT_CLIP_GRAPH")) c.use_graph = std::atoi(g);
  char err[512] = {0};
  enc_ = sgc_create(&c, err, sizeof(err));
  if (!enc_) {
    std::fprintf(stderr, "[sgrt] clip off: %s\n", err);
    return false;
  }
  cache_ = new sgclip::NameCache();
  std::string ldir = std::getenv("SGRT_LABELS") ? std::getenv("SGRT_LABELS") : ra::labels();
  std::string sample = std::getenv("SGC_IMG_SAMPLE") ? std::getenv("SGC_IMG_SAMPLE")
                                                     : ra::models() + "/x86_sm120/siglip2_b32/img_sample_lvis10k.f16";
  if (!fs::exists(sample)) sample.clear();
  const std::string idx = out_dir + "/cache/index";
  loader_ = std::thread([this, ldir, idx, sample] {   // 색인을 처음 만들면 1 s 안팎 — 스텝을 막지 않게
    char er[512] = {0};
    sgc_labels* L = sgc_labels_open_ex(ldir.c_str(), idx.c_str(), sample.empty() ? nullptr : sample.c_str(), er, sizeof(er));
    if (!L) {
      std::fprintf(stderr, "[sgrt] clip labels off (%s): %s\n", ldir.c_str(), er);
      return;
    }
    labels_ = L;
    labels_ready_ = true;
  });
  sgc_timing t;
  sgc_get_timing(enc_, &t);
  std::fprintf(stderr, "[sgrt] clip on: %s (S %d, grid %d, %.0f MB), labels %s\n", engine.c_str(), t.input_size, t.grid,
               t.device_bytes / 1048576.0, ldir.c_str());
  res_.resize(32);
  items_.reserve(kMaxPerKf * 4);
  return true;
}

void ClipMem::reset() {
  if (!enc_) return;
  sgc_poll(enc_, nullptr, 0, 1);   // 진행 중인 것 버림
  std::lock_guard<std::mutex> g(mu_);
  obj_.clear();
  saved_dir_.clear();
}

void ClipMem::take(const sgc_result& r) {   // mu_ 잡고
  Obj& o = obj_[r.id];
  o.pending = false;
  o.emb.assign(r.emb, r.emb + SGC_DIM);
  o.f16.resize(SGC_DIM);
  sgc_f32_to_f16(r.emb, o.f16.data(), SGC_DIM);
  o.sha = sgclip::embSha(o.f16.data());
  o.q = r.quality;
  o.stamp = r.stamp;
  ++o.ver;
  ++st_.n_done;
}

void ClipMem::poll() {
  if (!enc_ || sgc_pending(enc_) == 0) return;
  const int n = sgc_poll(enc_, res_.data(), int(res_.size()), 0);
  if (n <= 0) return;
  std::lock_guard<std::mutex> g(mu_);
  for (int i = 0; i < n; ++i) take(res_[i]);
}

void ClipMem::keyframe(double stamp, const uint8_t* rgb, int on_device, int64_t rs, int ps, int w, int h, const sm_detections* d,
                       sm_ctx* sm) {
  if (!enc_ || !d || d->n <= 0 || !rgb) return;
  poll();
  const int n = d->n;
  assoc_.resize(n);
  upd_.resize(n);
  qual_.resize(n);
  if (sm_last_assoc(sm, assoc_.data(), n) != n || sm_last_views(sm, upd_.data(), qual_.data(), n) != n) return;
  items_.clear();
  {
    std::lock_guard<std::mutex> g(mu_);
    for (int k = 0; k < n; ++k) {
      const uint32_t id = assoc_[k];
      if (!id) continue;
      auto it = obj_.find(id);
      const bool have = it != obj_.end() && !it->second.emb.empty();
      if (it != obj_.end() && it->second.pending) continue;
      // 새 물체, 또는 best view 가 바뀌고 품질이 임베딩 때의 kImprove 배 이상
      if (have && !(upd_[k] && qual_[k] >= kImprove * it->second.q)) continue;
      bool dup = false;   // 같은 물체에 검출 둘 — 품질 큰 것만
      for (sgc_item& x : items_)
        if (x.id == id) {
          dup = true;
          if (qual_[k] > x.quality) x = sgc_item{id, k, {d->box[4 * k], d->box[4 * k + 1], d->box[4 * k + 2], d->box[4 * k + 3]}, qual_[k]};
        }
      if (dup) continue;
      items_.push_back(sgc_item{id, k, {d->box[4 * k], d->box[4 * k + 1], d->box[4 * k + 2], d->box[4 * k + 3]}, qual_[k]});
    }
    // 순서: 새 물체 → 좋아진 비율
    std::stable_sort(items_.begin(), items_.end(), [&](const sgc_item& a, const sgc_item& b) {
      auto pa = obj_.find(a.id), pb = obj_.find(b.id);
      const bool na = pa == obj_.end() || pa->second.emb.empty(), nb = pb == obj_.end() || pb->second.emb.empty();
      if (na != nb) return na;
      if (na) return a.quality > b.quality;
      return a.quality / std::max(pa->second.q, 1e-3f) > b.quality / std::max(pb->second.q, 1e-3f);
    });
  }
  if (items_.empty()) return;
  const int m = std::min<int>(kMaxPerKf, int(items_.size()));
  sgc_frame fr{rgb, on_device, rs, ps, w, h, d->mask_w, d->mask_h, d->mask_sx, d->mask_sy, d->mask_ox, d->mask_oy, d->mask_bits};
  const int got = sgc_submit(enc_, stamp, &fr, items_.data(), m);
  std::lock_guard<std::mutex> g(mu_);
  if (got > 0) {
    for (int i = 0; i < got; ++i) obj_[items_[i].id].pending = true;
    st_.n_submitted += got;
  } else {
    st_.n_dropped += m;
  }
  sgc_timing t;
  sgc_get_timing(enc_, &t);
  st_.crop_ms = t.crop_ms;
  st_.net_ms = t.net_ms;
  st_.submit_us = t.submit_us;
  st_.last_batch = t.last_batch;
}

void ClipMem::save(const std::string& dir, sm_ctx* sm) {
  if (!enc_) return;
  const auto t0 = std::chrono::steady_clock::now();
  auto* cache = static_cast<sgclip::NameCache*>(cache_);
  // 바뀐 것 모으기(잠금 짧게)
  struct W {
    uint32_t id;
    std::vector<uint16_t> f16;
    std::vector<float> emb;
    std::string sha;
    double stamp;
    uint32_t ver;
  };
  std::vector<W> todo;
  std::vector<sgclip::ObjRef> refs;
  std::vector<W> all;
  bool new_dir;
  {
    std::lock_guard<std::mutex> g(mu_);
    new_dir = saved_dir_ != dir;
    for (auto& [id, o] : obj_) {
      if (o.emb.empty()) continue;
      W w{id, o.f16, o.emb, o.sha, o.stamp, o.ver};
      if (new_dir || o.saved_ver != o.ver) todo.push_back(w);
      all.push_back(std::move(w));
    }
  }
  std::error_code ec;
  if (new_dir) {   // 이 폴더 처음: 지금 물체가 아닌 O<id>_emb.f16 지우기, 이름 캐시 읽기(표·emb_sha 같으면 그대로 씀)
    fs::create_directories(dir + "/objects", ec);
    for (const auto& e : fs::directory_iterator(dir + "/objects", ec)) {
      const std::string n = e.path().filename().string();
      if (n.size() > 9 && n[0] == 'O' && n.compare(n.size() - 8, 8, "_emb.f16") == 0) {
        const uint32_t id = uint32_t(std::strtoul(n.c_str() + 1, nullptr, 10));
        if (std::none_of(all.begin(), all.end(), [&](const W& w) { return w.id == id; })) fs::remove(e.path(), ec);
      }
    }
    std::lock_guard<std::mutex> g(mu_);
    cache->load(dir);
  }
  for (const W& w : todo) sgclip::writeEmb(dir, w.id, w.f16.data());
  int named = 0;
  if (const sgc_labels* L = labels()) {
    const auto tn = std::chrono::steady_clock::now();
    for (const W& w : all) refs.push_back({w.id, w.sha, w.emb.data()});
    {
      std::lock_guard<std::mutex> g(mu_);   // names() 와 같이 읽음
      named = cache->refresh(L, refs);
    }
    if (named || new_dir || cache_sha_ != cache->table_sha) cache->save(dir);
    cache_sha_ = cache->table_sha;
    if (named) st_.names_us = float(usSince(tn) / named);
  }
  // 노드 메타: emb 가 바뀌었거나 이름이 새로 생긴 물체
  {
    std::lock_guard<std::mutex> g(mu_);
    for (const W& w : all) {
      auto it = obj_.find(w.id);
      if (it == obj_.end()) continue;
      const bool has_name = cache->get(w.id) != nullptr;
      const uint32_t want = w.ver * 2 + (has_name ? 1 : 0);
      if (it->second.meta_ver != want || new_dir) {
        sm_set_object_meta(sm, w.id, cache->nodeMeta(w.id, w.sha, w.stamp).c_str());
        it->second.meta_ver = want;
      }
      if (std::any_of(todo.begin(), todo.end(), [&](const W& t) { return t.id == w.id && t.ver == w.ver; })) it->second.saved_ver = w.ver;
    }
    saved_dir_ = dir;
    st_.n_named = int(cache->obj.size());
    st_.n_objects = int(all.size());
    st_.save_ms = float(usSince(t0) / 1000);
  }
}

int ClipMem::embedding(uint32_t id, float* out) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = obj_.find(id);
  if (it == obj_.end() || it->second.emb.empty()) return 0;
  if (out) std::copy(it->second.emb.begin(), it->second.emb.end(), out);
  return 1;
}

int ClipMem::query(const float* q, int k, uint32_t* ids, float* scores, sm_ctx* sm) {
  if (!q || k <= 0) return -1;
  // 살아 있는(스냅숏에 있는) 물체만. scenemap 의 structural 은 "큰·고정 가구"라 소파·냉장고도 들어가므로 거르지 않는다
  sm_snapshot_t* snap = nullptr;
  std::vector<uint32_t> live;
  if (sm && sm_snapshot(sm, &snap) == 0) {
    const sm_object* o = nullptr;
    const int n = sm_snap_objects(snap, &o);
    for (int i = 0; i < n; ++i)
      live.push_back(o[i].id);
    sm_snapshot_release(snap);
  }
  std::vector<std::pair<float, uint32_t>> s;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (const auto& [id, ob] : obj_) {
      if (ob.emb.empty() || (sm && std::find(live.begin(), live.end(), id) == live.end())) continue;
      double d = 0;
      for (int i = 0; i < SGC_DIM; ++i) d += double(ob.emb[i]) * q[i];
      s.emplace_back(float(d), id);
    }
  }
  const int n = std::min<int>(k, int(s.size()));
  std::partial_sort(s.begin(), s.begin() + n, s.end(), [](auto& a, auto& b) { return a.first > b.first; });
  for (int i = 0; i < n; ++i) {
    if (ids) ids[i] = s[i].second;
    if (scores) scores[i] = s[i].first;
  }
  return n;
}

int ClipMem::queryText(const char* text, int k, uint32_t* ids, float* scores, sm_ctx* sm) {
  const sgc_labels* L = labels();
  if (!L) return -3;
  const int row = sgc_labels_find(L, text);
  if (row < 0) return -2;
  std::vector<float> q(SGC_DIM);
  sgc_labels_text_emb(L, row, q.data());
  return query(q.data(), k, ids, scores, sm);
}

int ClipMem::names(uint32_t id, std::vector<std::pair<std::string, std::string>>* en_ko, std::vector<float>* sc, std::string* level,
                   std::string* level_ko, int* structural) {
  const auto* cache = static_cast<const sgclip::NameCache*>(cache_);
  std::lock_guard<std::mutex> g(mu_);
  const sgclip::NameEntry* e = cache ? cache->get(id) : nullptr;
  if (!e) return 0;
  en_ko->clear();
  sc->clear();
  for (size_t i = 0; i < e->en.size(); ++i) {
    std::string ko;
    for (const auto& [k, s] : e->ko)
      if (std::abs(s - e->en[i].second) < 1e-6f) { ko = k; break; }
    en_ko->emplace_back(e->en[i].first, ko);
    sc->push_back(e->en[i].second);
  }
  *level = e->level;
  *level_ko = e->level_ko;
  *structural = e->structural;
  return 1;
}

Stats ClipMem::stats() const {
  std::lock_guard<std::mutex> g(mu_);
  Stats s = st_;
  return s;
}

}  // namespace sgrt_clip
