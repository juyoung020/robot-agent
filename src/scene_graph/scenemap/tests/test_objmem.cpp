// 물체 기억 시험(C ABI 만 씀): 합성 머리 RGB-D + 검출로
//   구조물   — 벽·바닥·문 검출은 물체가 안 됨(격자만), 그림 액자·러그·가구는 물체(movable 표)
//   상자     — 마스크 안 깊이 이상값(뒤 벽이 비침)에도 크기가 참값 근처, 큰 가구 상자는 keyframe 마다 자람 한도
//   사라짐   — 작은 물체는 안 보인 지 gone_min_s 넘어야, 가구는 끝까지 안 사라짐
//   best view — 품질(넓이 × 점수) 최대·같으면 최근, 자른 그림 내용·상자
//   저장     — PNG(서명·IHDR·zlib 풀어 화소 비교, 자체 디코더), scene.json 을 spark_dsg 로 다시 읽어 rgbd 메타데이터 확인,
//              바뀐 모습만 다시 씀
//   모양     — 점 구름이 map 의 참 자리(< 1 cm)·색, 한도, 들기(손 따라감)·사라짐(유지)·옮겨짐 잇기(새로), PLY·마스크 PNG
//   시간     — keyframe 갱신, 저장(모습 바뀜 / 안 바뀜)
// 사용: test_objmem [출력 디렉터리]
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "scenemap.h"

#ifdef SM_TEST_SPARK_DSG
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#endif

namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(c, ...)                                         \
  do {                                                        \
    if (!(c)) {                                               \
      ++g_fail;                                               \
      std::printf("  FAIL %s:%d %s — ", __FILE__, __LINE__, #c); \
      std::printf(__VA_ARGS__);                               \
      std::printf("\n");                                      \
    }                                                         \
  } while (0)

static double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- 합성 장면 ----
constexpr int W = 720, H = 720, MS = 4, MW = W / MS, MH = H / MS;
constexpr double FX = 306, CX = 360;

struct Rect {
  int cls;
  int x0, y0, x1, y1;           // 화소, [x0, x1)
  float depth;                  // m
  uint8_t rgb[3];
  float score = 0.9f;
  float out_frac = 0;           // 마스크 안 이 비율의 화소는 out_depth(뒤 벽이 비침)
  float out_depth = 0;
  bool detect = true;           // false: 깊이·색만 그리고 검출은 안 냄
};

struct Frame {
  std::vector<float> depth;
  std::vector<uint8_t> rgba;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
  sm_detections d{};
};

static void render(Frame& f, const std::vector<Rect>& rs, float bg, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(0, 1);
  f.depth.assign(size_t(W) * H, bg);
  f.rgba.assign(size_t(W) * H * 4, 0);
  for (size_t i = 0; i < size_t(W) * H; ++i) { f.rgba[4 * i] = 40; f.rgba[4 * i + 1] = 40; f.rgba[4 * i + 2] = 40; f.rgba[4 * i + 3] = 255; }
  const size_t words = (size_t(MW) * MH + 31) / 32;
  f.cls.clear(); f.score.clear(); f.box.clear(); f.bits.clear();
  for (const Rect& r : rs) {
    for (int y = r.y0; y < r.y1; ++y)
      for (int x = r.x0; x < r.x1; ++x) {
        const size_t i = size_t(y) * W + x;
        f.depth[i] = (r.out_frac > 0 && U(rng) < r.out_frac) ? r.out_depth : r.depth;
        std::memcpy(&f.rgba[4 * i], r.rgb, 3);
      }
    if (!r.detect) continue;
    f.cls.push_back(r.cls);
    f.score.push_back(r.score);
    f.box.insert(f.box.end(), {float(r.x0), float(r.y0), float(r.x1), float(r.y1)});
    const size_t base = f.bits.size();
    f.bits.resize(base + words, 0);
    for (int j = r.y0 / MS; j < r.y1 / MS; ++j)
      for (int i = r.x0 / MS; i < r.x1 / MS; ++i) {
        const size_t c = size_t(j) * MW + i;
        f.bits[base + (c >> 5)] |= 1u << (c & 31);
      }
  }
  sm_detections& d = f.d;
  d = sm_detections{};
  d.img_w = W; d.img_h = H; d.n = int(f.cls.size());
  d.cls = f.cls.data(); d.score = f.score.data(); d.box = f.box.data();
  d.mask_w = MW; d.mask_h = MH; d.mask_sx = MS; d.mask_sy = MS; d.mask_ox = 0; d.mask_oy = 0;
  d.mask_bits = f.bits.data();
}

// 실제 시연 한 장면의 관절(test_fk 와 같은 표본) — 머리가 앞 아래를 봄. 팔 끝은 로봇 뒤로 치움(손 거르기에 안 걸리게)
static std::vector<float> proprio() {
  static const float S[61] = {
      0.f, 0.f, 0.f, -0.5135943f, 0.1073003f, -0.0477751f, -1.0079254f, 0.3504786f, 0.6210044f,
      0.5144721f, -0.0518736f, -0.0281572f, -0.0266854f, -0.2630615f, 0.6464736f, -0.2789042f, 0.010686f, 0.6501044f,
      0.3716516f, 0.5129846f, -0.1484585f, 0.941596f, 0.1572233f, 0.2581432f, 0.0075572f, 0.0003005f, 0.0209931f,
      -0.0171822f, -0.4822987f, 0.1745f, 0.7338722f, -1.5772073f, -0.0225522f, 1.0406232f, 0.0773677f, -0.0002653f,
      0.0003976f, 0.0011009f, 0.0045551f, 0.0169314f, 0.0107928f, 0.0021622f, 0.5823845f, 0.0940711f, 0.6938694f,
      -0.3479472f, 0.8557385f, 0.1888288f, 0.3331488f, 0.05f, 0.0244954f, 0.008631f, 0.0002444f, 1.2696129f, -1.896482f,
      -0.9405322f, -0.0004273f, -0.0026646f, -0.0017128f, 0.0187404f, -0.0033622f};
  std::vector<float> q(S, S + 61);
  q[17] = -2.0f; q[18] = 1.0f; q[19] = 0.2f;    // 왼 팔 끝(베이스 기준)
  q[42] = -2.0f; q[43] = -1.0f; q[44] = 0.2f;   // 오른 팔 끝
  q[24] = q[25] = q[49] = q[50] = 0.05f;        // 그리퍼 열림(합 0.1)
  return q;
}

struct Rig {
  sm_ctx* c;
  std::vector<float> q = proprio();
  Frame f;
  explicit Rig(const std::vector<const char*>& labels) {
    c = sm_create(nullptr);
    sm_set_labels(c, labels.data(), int(labels.size()));
  }
  ~Rig() { sm_destroy(c); }
  // 시각 t 에 proprio + 영상(장면 rs) 하나. 돌려줌: sm_push_image_ex 시간 ms
  double kf(double t, const std::vector<Rect>& rs, float bg = 3.0f, bool with_rgb = true, bool with_dets = true) {
    render(f, rs, bg, uint32_t(t * 1000) + 7);
    sm_proprio p{t, q.data(), 61};
    sm_push_proprio(c, &p);
    sm_image im{};
    im.stamp = t; im.cam = 0; im.w = W; im.h = H;
    im.rgba = with_rgb ? f.rgba.data() : nullptr;
    im.depth_m = f.depth.data();
    im.fx = FX; im.fy = FX; im.cx = CX; im.cy = CX;
    f.d.stamp = t;
    const double t0 = nowMs();
    sm_push_image_ex(c, &im, with_dets ? &f.d : nullptr, nullptr, nullptr);
    return nowMs() - t0;
  }
  std::vector<uint32_t> assoc() {
    std::vector<uint32_t> ids(64, 0);
    const int n = sm_last_assoc(c, ids.data(), 64);
    ids.resize(std::max(0, n));
    return ids;
  }
};

struct Snap {
  sm_snapshot_t* s = nullptr;
  explicit Snap(sm_ctx* c) { sm_snapshot(c, &s); }
  ~Snap() { sm_snapshot_release(s); }
  std::vector<sm_object> objs() const {
    const sm_object* o = nullptr;
    const int n = sm_snap_objects(s, &o);
    return std::vector<sm_object>(o, o + n);
  }
  const sm_object* byName(const char* n) const {
    const sm_object* o = nullptr;
    const int k = sm_snap_objects(s, &o);
    for (int i = 0; i < k; ++i)
      if (std::strcmp(o[i].name, n) == 0) return &o[i];
    return nullptr;
  }
};

enum { C_CUP, C_WALL, C_FLOOR, C_DOOR, C_SOFA, C_FRAME, C_RUG, C_BOOK };
static const std::vector<const char*> kLabels = {"cup", "wall", "floor", "glass door", "sofa", "picture frame", "rug", "book"};

static Rect R(int cls, int x0, int y0, int x1, int y1, float d, uint8_t r, uint8_t g, uint8_t b, float sc = 0.9f) {
  Rect q{};
  q.cls = cls; q.x0 = x0; q.y0 = y0; q.x1 = x1; q.y1 = y1; q.depth = d;
  q.rgb[0] = r; q.rgb[1] = g; q.rgb[2] = b; q.score = sc;
  return q;
}

// ---- 1. 구조물·movable ----
static void testStructures() {
  std::printf("[structures]\n");
  Rig g(kLabels);
  std::vector<Rect> rs = {R(C_WALL, 0, 0, 720, 200, 3.0f, 200, 200, 200), R(C_FLOOR, 0, 600, 720, 720, 2.0f, 90, 60, 30),
                          R(C_DOOR, 40, 220, 160, 560, 2.8f, 120, 80, 40), R(C_FRAME, 500, 220, 600, 300, 2.9f, 10, 200, 10),
                          R(C_RUG, 200, 520, 520, 590, 2.2f, 200, 10, 10), R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0)};
  for (int k = 0; k < 4; ++k) g.kf(0.2 * k, rs);
  const auto as = g.assoc();
  CHECK(as.size() == rs.size(), "assoc %zu", as.size());
  CHECK(as[0] == 0 && as[1] == 0 && as[2] == 0, "wall/floor/door assoc %u %u %u", as[0], as[1], as[2]);
  CHECK(as[3] != 0 && as[4] != 0 && as[5] != 0, "frame/rug/cup assoc %u %u %u", as[3], as[4], as[5]);
  Snap s(g.c);
  CHECK(!s.byName("wall") && !s.byName("floor") && !s.byName("glass door"), "structure became a node");
  const sm_object* fr = s.byName("picture frame");
  const sm_object* cup = s.byName("cup");
  CHECK(fr && s.byName("rug") && cup, "objects missing");
  if (fr && cup) {
    CHECK(sm_snap_movable(s.s, fr->id) == 0, "picture frame movable=%d", sm_snap_movable(s.s, fr->id));
    CHECK(sm_snap_movable(s.s, cup->id) == 1, "cup movable=%d", sm_snap_movable(s.s, cup->id));
  }
  // 목록 바꾸기(C ABI): 문을 물체로, 컵을 구조물로
  const char* st[] = {"wall", "floor", "cup"};
  sm_set_kind_names(g.c, SM_KIND_STRUCTURE, st, 3);
  sm_reset(g.c);
  sm_set_labels(g.c, kLabels.data(), int(kLabels.size()));
  for (int k = 0; k < 3; ++k) g.kf(0.2 * k, rs);
  const auto as2 = g.assoc();
  CHECK(as2[2] != 0 && as2[5] == 0, "custom list: door %u cup %u", as2[2], as2[5]);
  sm_set_kind_names(g.c, SM_KIND_STRUCTURE, nullptr, -1);   // 기본값으로
  std::printf("  nodes %zu (wall/floor/door 없음)\n", s.objs().size());
  // COCO-80 이름(닫힌 어휘 YOLO-seg): person 은 노드 아님, dining table·couch·tv 는 고정, cup·chair 는 옮길 수 있음.
  // (평면 사각형이라 영상 아래쪽에 두면 바닥 높이가 되어 바닥 조각으로 걸러짐 — 소파·탁자는 가운데 높이에)
  const std::vector<const char*> coco = {"person", "cup", "chair", "couch", "dining table", "tv", "potted plant"};
  Rig c(coco);
  std::vector<Rect> cr = {R(0, 40, 220, 120, 560, 2.5f, 200, 160, 120), R(1, 300, 300, 340, 340, 1.5f, 250, 0, 0),
                          R(2, 160, 300, 260, 460, 2.0f, 90, 60, 30),   R(3, 400, 310, 700, 400, 1.7f, 30, 30, 160),
                          R(4, 280, 360, 380, 460, 1.8f, 120, 80, 40),  R(5, 500, 220, 600, 300, 2.9f, 10, 10, 10),
                          R(6, 620, 220, 700, 300, 2.9f, 10, 200, 10)};
  for (int k = 0; k < 3; ++k) c.kf(0.2 * k, cr);
  Snap cs(c.c);
  CHECK(!cs.byName("person"), "person became a node");
  int bad = 0;
  for (const char* n : {"dining table", "couch", "tv", "potted plant"}) {
    const sm_object* o = cs.byName(n);
    bad += !o || sm_snap_movable(cs.s, o->id) != 0;
  }
  for (const char* n : {"cup", "chair"}) {
    const sm_object* o = cs.byName(n);
    bad += !o || sm_snap_movable(cs.s, o->id) != 1;
  }
  if (bad)
    for (const sm_object& o : cs.objs()) std::printf("    node %s pos %.2f %.2f %.2f movable %d\n", o.name, o.pos[0], o.pos[1], o.pos[2], sm_snap_movable(cs.s, o.id));
  CHECK(bad == 0, "coco kinds wrong %d", bad);
  std::printf("  COCO: person 노드 없음, 가구·tv·화분 고정, cup·chair 옮길 수 있음 (노드 %zu)\n", cs.objs().size());
}

// ---- 2. 상자: 깊이 이상값·큰 가구 자람 한도 ----
static void testBoxes() {
  std::printf("[boxes]\n");
  Rig g(kLabels);
  Rect cup = R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0);
  cup.out_frac = 0.25f;   // 마스크의 25 % 가 뒤 벽(2.6 m)
  cup.out_depth = 2.6f;
  for (int k = 0; k < 6; ++k) g.kf(0.2 * k, {cup});
  Snap s(g.c);
  const sm_object* o = s.byName("cup");
  CHECK(o, "cup missing");
  const double truth = 40 / FX * 1.5;   // 화면 40 px @ 1.5 m
  if (o) {
    const double e = std::max({o->extent[0], o->extent[1], o->extent[2]});
    std::printf("  cup extent %.3f %.3f %.3f (참 한 변 %.3f, 이상값 25 %%)\n", o->extent[0], o->extent[1], o->extent[2], truth);
    CHECK(e < 1.6 * truth, "cup extent %.3f > %.3f", e, 1.6 * truth);
  }
  // 소파: 처음 두 번은 가운데만(0.6 m), 그 뒤 온 폭(2.9 m)이 갑자기 보이고 마스크의 20 % 는 뒤 벽(4.5 m) — 상자는
  // keyframe 마다 면마다 grow_max(0.25 m)까지만 자라고, 이상값으로 참 크기를 넘지 않음
  Rig h(kLabels);
  double prev = 0, max_step = 0;
  for (int k = 0; k < 20; ++k) {
    Rect so = k < 2 ? R(C_SOFA, 300, 350, 420, 470, 1.5f, 30, 30, 160) : R(C_SOFA, 60, 350, 660, 470, 1.5f, 30, 30, 160);
    so.out_frac = k < 2 ? 0.f : 0.2f;
    so.out_depth = 4.5f;
    h.kf(0.2 * k, {so});
    Snap t(h.c);
    const sm_object* q = t.byName("sofa");
    if (!q) continue;
    const double e = std::max({q->extent[0], q->extent[1], q->extent[2]});
    if (prev > 0) max_step = std::max(max_step, e - prev);
    prev = e;
  }
  const double sofa_truth = 600 / FX * 1.5;
  std::printf("  sofa 가장 긴 변 %.3f (참 %.3f), keyframe 당 최대 자람 %.3f (한도 면마다 0.25)\n", prev, sofa_truth, max_step);
  CHECK(prev > 0.7 * sofa_truth && prev < 1.15 * sofa_truth, "sofa extent %.3f", prev);
  CHECK(max_step <= 0.5 + 1e-6 && max_step > 0.3, "sofa grew %.3f in one keyframe", max_step);
}

// ---- 3. 사라짐 히스테리시스 ----
static void testGone() {
  std::printf("[gone]\n");
  Rig g(kLabels);
  const Rect cup = R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0);
  const Rect sofa = R(C_SOFA, 100, 350, 620, 440, 1.7f, 30, 30, 160);
  double t = 0;
  for (int k = 0; k < 6; ++k, t += 0.2) g.kf(t, {cup, sofa});   // 관측 6 번(spurious_obs 5 이상: 헛검출로 지우지 않음)
  // 이제 둘 다 없음(뒤는 3 m 벽 — 가림 없음). 0.2 s 마다
  const double t_miss = t;
  double gone_at = -1, sofa_gone_at = -1;
  int state_1s = -1;
  for (; t < t_miss + 6.0; t += 0.2) {
    g.kf(t, {});
    Snap s(g.c);
    const sm_object* c = s.byName("cup");
    if (c && t - t_miss < 1.0 + 1e-9) state_1s = c->state;
    if (c && c->state == SM_GONE && gone_at < 0) gone_at = t - t_miss;
    const sm_object* so = s.byName("sofa");
    CHECK(so, "sofa missing at %.1f", t - t_miss);
    if (so && so->state == SM_GONE && sofa_gone_at < 0) sofa_gone_at = t - t_miss;
  }
  // 큰 가구도 사라짐 판정을 하지만 근거를 더 모은다(gone_misses_big 6 번, 서 있는 카메라면 gone_min_s_big 4 s)
  std::printf("  cup: 1 s 안 상태 %d(0=seen), 사라짐 판정 %.1f s 뒤; sofa(큰 것) %.1f s 뒤\n", state_1s, gone_at, sofa_gone_at);
  CHECK(state_1s == SM_SEEN, "cup state within 1 s = %d", state_1s);
  CHECK(gone_at >= 2.0 - 1e-6 && gone_at < 3.0, "cup gone after %.2f s", gone_at);
  CHECK(sofa_gone_at >= 4.0 - 1e-6 && sofa_gone_at < 5.0, "sofa gone after %.2f s", sofa_gone_at);
}

// ---- 4. best view ----
static void testBestView() {
  std::printf("[best view]\n");
  Rig g(kLabels);
  struct F { int half; float sc; uint8_t r; };
  // 넓이 × 점수: (40²·0.5) (60²·0.9 최고) (40²·0.9 작음) (60²·0.9 같음 → 최근) (80²·0.1 작음)
  const F fs[] = {{20, 0.5f, 10}, {30, 0.9f, 20}, {20, 0.9f, 30}, {30, 0.9f, 40}, {40, 0.1f, 50}};
  const double ts[] = {0.0, 0.2, 0.4, 0.6, 0.8};
  uint32_t ver[5] = {0};
  double stamp[5] = {0};
  for (int k = 0; k < 5; ++k) {
    g.kf(ts[k], {R(C_BOOK, 320 - fs[k].half, 320 - fs[k].half, 320 + fs[k].half, 320 + fs[k].half, 1.5f, fs[k].r, 100, 200, fs[k].sc)});
    Snap s(g.c);
    const sm_object* o = s.byName("book");
    sm_view v{};
    if (o && sm_snap_view(s.s, o->id, &v) == 1) { ver[k] = v.version; stamp[k] = v.stamp; }
  }
  std::printf("  version %u %u %u %u %u, stamp %.1f %.1f %.1f %.1f %.1f\n", ver[0], ver[1], ver[2], ver[3], ver[4], stamp[0], stamp[1],
              stamp[2], stamp[3], stamp[4]);
  CHECK(stamp[1] == 0.2 && stamp[2] == 0.2 && stamp[3] == 0.6 && stamp[4] == 0.6, "best-view stamps");
  CHECK(ver[2] == ver[1] && ver[3] > ver[2] && ver[4] == ver[3], "versions");
  Snap s(g.c);
  const sm_object* o = s.byName("book");
  sm_view v{};
  CHECK(o && sm_snap_view(s.s, o->id, &v) == 1, "no view");
  if (o) {
    // 상자 60 px + 변마다 10 % = 72 px, 줄임 없음
    CHECK(v.box_px[0] == 284 && v.box_px[1] == 284 && v.box_px[2] == 356 && v.box_px[3] == 356, "box %d %d %d %d", v.box_px[0],
          v.box_px[1], v.box_px[2], v.box_px[3]);
    CHECK(v.w == 72 && v.h == 72 && v.rgb && v.depth_mm, "crop %dx%d", v.w, v.h);
    if (v.rgb) {
      const uint8_t* c = v.rgb + (36 * 72 + 36) * 3;
      CHECK(c[0] == 40 && c[1] == 100 && c[2] == 200, "centre rgb %u %u %u", c[0], c[1], c[2]);
      CHECK(v.rgb[0] == 40 && v.rgb[1] == 40, "margin rgb %u", v.rgb[0]);
    }
    if (v.depth_mm) CHECK(v.depth_mm[36 * 72 + 36] == 1500 && v.depth_mm[0] == 3000, "depth %u %u", v.depth_mm[36 * 72 + 36], v.depth_mm[0]);
    CHECK(std::fabs(v.depth_m - 1.5f) < 1e-3, "depth_m %.3f", v.depth_m);
    CHECK(v.mask_area > 2000 && v.mask_area <= 3600, "mask area %.0f", v.mask_area);
  }
  // 큰 상자는 긴 변 256 으로 줄임
  Rig h(kLabels);
  for (int k = 0; k < 2; ++k) h.kf(0.2 * k, {R(C_SOFA, 60, 200, 660, 500, 2.0f, 30, 30, 160)});
  Snap t(h.c);
  const sm_object* so = t.byName("sofa");
  sm_view u{};
  CHECK(so && sm_snap_view(t.s, so->id, &u) == 1 && std::max(u.w, u.h) == 256, "sofa crop %dx%d", u.w, u.h);
  std::printf("  sofa crop %dx%d (상자 %d..%d)\n", u.w, u.h, u.box_px[0], u.box_px[2]);
}

// ---- 5. 저장: PNG·scene.json·더러움 ----
static uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

// 최소 PNG 디코더(시험용): 서명·IHDR·CRC 확인, IDAT 이어 붙여 inflate, 필터 0..4 되돌림. 8 비트 RGB / 16 비트 회색만
static bool decodePng(const std::string& path, int* w, int* h, int* depth, int* ctype, std::vector<uint8_t>* raw) {
  std::ifstream f(path, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (s.size() < 33 || std::memcmp(s.data(), sig, 8) != 0) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
  size_t o = 8;
  std::string idat;
  bool ihdr = false, iend = false;
  while (o + 12 <= s.size()) {
    const uint32_t len = be32(p + o);
    if (o + 12 + len > s.size()) return false;
    const std::string type(s.data() + o + 4, 4);
    if (uint32_t(crc32(0, p + o + 4, len + 4)) != be32(p + o + 8 + len)) return false;
    const uint8_t* d = p + o + 8;
    if (type == "IHDR") {
      if (o != 8 || len != 13) return false;
      *w = int(be32(d)); *h = int(be32(d + 4)); *depth = d[8]; *ctype = d[9];
      if (d[10] || d[11] || d[12]) return false;
      ihdr = true;
    } else if (type == "IDAT") {
      idat.append(reinterpret_cast<const char*>(d), len);
    } else if (type == "IEND") {
      iend = true;
      break;
    }
    o += 12 + len;
  }
  if (!ihdr || !iend) return false;
  const int bpp = (*ctype == 2 ? 3 : 1) * (*depth / 8);
  const size_t stride = size_t(*w) * bpp;
  std::vector<uint8_t> z(size_t(*h) * (stride + 1));
  uLongf zl = uLongf(z.size());
  if (uncompress(z.data(), &zl, reinterpret_cast<const Bytef*>(idat.data()), uLong(idat.size())) != Z_OK || zl != z.size()) return false;
  raw->assign(size_t(*h) * stride, 0);
  for (int y = 0; y < *h; ++y) {
    const uint8_t ft = z[y * (stride + 1)];
    const uint8_t* in = &z[y * (stride + 1) + 1];
    uint8_t* out = &(*raw)[y * stride];
    const uint8_t* up = y ? &(*raw)[(y - 1) * stride] : nullptr;
    for (size_t i = 0; i < stride; ++i) {
      const int a = i >= size_t(bpp) ? out[i - bpp] : 0, b = up ? up[i] : 0, c = (up && i >= size_t(bpp)) ? up[i - bpp] : 0;
      int v = in[i];
      if (ft == 1) v += a;
      else if (ft == 2) v += b;
      else if (ft == 3) v += (a + b) / 2;
      else if (ft == 4) {
        const int pp = a + b - c, pa = std::abs(pp - a), pb = std::abs(pp - b), pc = std::abs(pp - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      } else if (ft != 0) return false;
      out[i] = uint8_t(v);
    }
  }
  return true;
}

// binary_little_endian PLY(float x,y,z + uchar red,green,blue)만 읽음 — 머리 줄도 정확히 확인
static bool readPly(const std::string& path, std::vector<float>* xyz, std::vector<uint8_t>* rgb) {
  std::ifstream f(path, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const size_t e = s.find("end_header\n");
  if (e == std::string::npos) return false;
  const std::string head = s.substr(0, e);
  size_t n = 0;
  if (std::sscanf(head.c_str(), "ply\nformat binary_little_endian 1.0\nelement vertex %zu", &n) != 1) return false;
  const std::string want = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(n) +
                           "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                           "property uchar blue\n";
  if (head != want) return false;
  const char* p = s.data() + e + 11;
  if (s.size() != e + 11 + n * 15) return false;
  xyz->resize(3 * n);
  rgb->resize(3 * n);
  for (size_t i = 0; i < n; ++i, p += 15) {
    std::memcpy(&(*xyz)[3 * i], p, 12);
    std::memcpy(&(*rgb)[3 * i], p + 12, 3);
  }
  return true;
}

static std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- 살펴본 정도(sm_set_inspect): 끄면 저장 파일에 없음·sm_snap_inspect -2, 켜면 물체마다 값, 그 멤버를 빼면 view.json 이 바이트 같음 ----
static std::string stripInspect(std::string t) {
  for (size_t k; (k = t.find(",\"inspect\":{")) != std::string::npos;) t.erase(k, t.find('}', k) + 1 - k);
  return t;
}

static void testInspect(const std::string& dir) {
  std::printf("[inspect]\n");
  const std::vector<Rect> rs = {R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0), R(C_SOFA, 60, 350, 660, 440, 1.7f, 30, 30, 160)};
  std::string vj[2];
  for (int on = 0; on < 2; ++on) {
    Rig g(kLabels);
    if (on) CHECK(sm_set_inspect(g.c, 1) == 0, "set inspect");
    for (int k = 0; k < 5; ++k) g.kf(0.2 * k, rs);
    Snap s(g.c);
    const sm_inspect* q = nullptr;
    const int n = sm_snap_inspect(s.s, &q);
    const auto objs = s.objs();
    if (!on) CHECK(n == -2, "off: sm_snap_inspect %d", n);
    if (on) {
      CHECK(n == int(objs.size()) && n >= 2, "on: count %d objs %zu", n, objs.size());
      for (int i = 0; i < n && q; ++i) {
        CHECK(q[i].id == objs[size_t(i)].id, "order");
        std::printf("  O%u %s closest %.2f views %d top %.3f\n", q[i].id, objs[size_t(i)].name, q[i].closest_view_m, q[i].n_views, q[i].top_seen);
        CHECK(q[i].n_views == 1, "standing camera -> 1 view (%d)", q[i].n_views);
        CHECK(q[i].closest_view_m > 1.0f && q[i].closest_view_m < 2.5f, "closest %.2f", q[i].closest_view_m);
        if (std::strcmp(objs[size_t(i)].name, "cup") == 0) CHECK(q[i].top_seen == -1.f, "cup has no top %.2f", q[i].top_seen);
      }
    }
    const std::string d = dir + (on ? "_insp_on" : "_insp_off");
    fs::remove_all(d);
    CHECK(sm_save_dsg(g.c, d.c_str()) == 0, "save");
    vj[on] = slurp(fs::path(d) / "view.json");
    const std::string sj = slurp(fs::path(d) / "scene.json");
    CHECK((sj.find("\"inspect\":{") != std::string::npos) == bool(on), "scene.json inspect member on=%d", on);
    fs::remove_all(d);
  }
  CHECK(vj[0].find("\"inspect\"") == std::string::npos, "off view.json has inspect");
  CHECK(vj[1].find("\"inspect\":{\"closest_view_m\":") != std::string::npos, "on view.json lacks inspect");
  CHECK(stripInspect(vj[1]) == vj[0], "view.json differs beyond inspect members");
  // 설정 키 "inspect": 1 은 sm_set_robot 뒤에도 남음
  sm_ctx* c = sm_create("{\"inspect\": 1, \"robot\": \"limo_omx\"}");
  CHECK(c != nullptr, "create");
  if (c) {
    sm_set_robot(c, SM_ROBOT_R1PRO);
    sm_snapshot_t* s = nullptr;
    sm_snapshot(c, &s);
    const sm_inspect* q = nullptr;
    CHECK(sm_snap_inspect(s, &q) == 0, "config inspect kept after set_robot");
    sm_snapshot_release(s);
    sm_destroy(c);
  }
}

static void testSave(const std::string& dir) {
  std::printf("[save] %s\n", dir.c_str());
  fs::remove_all(dir);
  Rig g(kLabels);
  std::vector<Rect> rs = {R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0), R(C_BOOK, 420, 270, 480, 345, 1.8f, 0, 0, 250),
                          R(C_SOFA, 60, 350, 660, 440, 1.7f, 30, 30, 160), R(C_WALL, 0, 0, 720, 200, 3.0f, 200, 200, 200)};
  double kf_ms = 0;
  int nkf = 0;
  for (int k = 0; k < 4; ++k, ++nkf) kf_ms += g.kf(0.2 * k, rs);
  sm_save_stats st{};
  double t0 = nowMs();
  CHECK(sm_save_dsg_ex(g.c, dir.c_str(), &st) == 0, "save failed");
  const double save1 = nowMs() - t0;
  CHECK(st.n_png == 9 && st.n_ply == 3, "first save png %d ply %d", st.n_png, st.n_ply);
  // 곧바로 다시 저장: 바뀐 것 없음 → PNG·PLY 안 씀
  sm_save_stats st0{};
  sm_save_dsg_ex(g.c, dir.c_str(), &st0);
  CHECK(st0.n_png == 0 && st0.n_ply == 0, "unchanged save png %d ply %d", st0.n_png, st0.n_ply);
  // 같은 장면(품질 같음 → 최근으로 바뀜)이 아니라, 더 작은 점수로 다시 봄 → 모습 안 바뀜 → PNG 안 씀
  for (auto& r : rs) r.score = 0.3f;
  for (int k = 4; k < 8; ++k, ++nkf) kf_ms += g.kf(0.2 * k, rs);
  sm_save_stats st2{};
  t0 = nowMs();
  CHECK(sm_save_dsg_ex(g.c, dir.c_str(), &st2) == 0, "save2 failed");
  const double save2 = nowMs() - t0;
  CHECK(st2.n_png == 0 && st2.n_ply == 3, "clean save wrote %d png %d ply", st2.n_png, st2.n_ply);   // 구름은 새 관측으로 바뀜
  // 책만 더 잘 보임 → 책 두 장만
  rs[1].score = 0.99f;
  ++nkf;
  kf_ms += g.kf(1.8, rs);
  sm_save_stats st3{};
  sm_save_dsg_ex(g.c, dir.c_str(), &st3);
  CHECK(st3.n_png == 3, "dirty save wrote %d png", st3.n_png);
  std::printf("  keyframe 평균 %.3f ms (720², 검출 4), 저장: 다 바뀜(PNG 9·PLY 3) %.3f ms, 안 바뀜 %.3f ms, 구름만(PLY 3) %.3f ms, "
              "책 모습+구름(PNG 3·PLY %d) %.3f ms\n",
              kf_ms / nkf, save1, st0.total_ms, save2, st3.n_ply, st3.total_ms);
  std::printf("  (sm_save_stats 첫 저장: png %.3f ms, ply %.3f ms, 전체 %.3f ms)\n", st.png_ms, st.ply_ms, st.total_ms);

  // PNG 확인: 스냅숏 모습과 화소 비교
  Snap s(g.c);
  int checked = 0;
  for (const sm_object& o : s.objs()) {
    sm_view v{};
    if (sm_snap_view(s.s, o.id, &v) != 1) continue;
    const fs::path rp = fs::path(dir) / "objects" / ("O" + std::to_string(o.id) + "_rgb.png");
    const fs::path dp = fs::path(dir) / "objects" / ("O" + std::to_string(o.id) + "_depth.png");
    int w, h, bd, ct;
    std::vector<uint8_t> raw;
    CHECK(decodePng(rp.string(), &w, &h, &bd, &ct, &raw), "decode %s", rp.c_str());
    CHECK(w == v.w && h == v.h && bd == 8 && ct == 2, "%s IHDR %dx%d %d %d", rp.c_str(), w, h, bd, ct);
    CHECK(raw.size() == size_t(v.w) * v.h * 3 && std::memcmp(raw.data(), v.rgb, raw.size()) == 0, "%s pixels", rp.c_str());
    CHECK(decodePng(dp.string(), &w, &h, &bd, &ct, &raw), "decode %s", dp.c_str());
    CHECK(w == v.w && h == v.h && bd == 16 && ct == 0, "%s IHDR %dx%d %d %d", dp.c_str(), w, h, bd, ct);
    bool same = raw.size() == size_t(v.w) * v.h * 2;
    for (int i = 0; same && i < v.w * v.h; ++i) same = (uint16_t(raw[2 * i]) << 8 | raw[2 * i + 1]) == v.depth_mm[i];
    CHECK(same, "%s depth pixels", dp.c_str());
    const fs::path mp = fs::path(dir) / "objects" / ("O" + std::to_string(o.id) + "_mask.png");
    CHECK(decodePng(mp.string(), &w, &h, &bd, &ct, &raw), "decode %s", mp.c_str());
    CHECK(w == v.w && h == v.h && bd == 8 && ct == 0 && v.mask, "%s IHDR %dx%d %d %d", mp.c_str(), w, h, bd, ct);
    CHECK(v.mask && raw.size() == size_t(v.w) * v.h && std::memcmp(raw.data(), v.mask, raw.size()) == 0, "%s pixels", mp.c_str());
    // PLY: 머리·점 = 스냅숏 구름(map)
    sm_cloud cl{};
    CHECK(sm_snap_points(s.s, o.id, &cl) == 1 && cl.n > 0, "cloud");
    std::vector<float> xyz;
    std::vector<uint8_t> col;
    const fs::path pp = fs::path(dir) / "objects" / ("O" + std::to_string(o.id) + "_points.ply");
    CHECK(readPly(pp.string(), &xyz, &col), "read %s", pp.c_str());
    bool pts_same = int(xyz.size()) == 3 * cl.n;
    for (int i = 0; pts_same && i < cl.n; ++i) {
      const sm_cloud_pt& q = cl.pts[i];
      pts_same = std::fabs(xyz[3 * i] - float(cl.origin[0] + q.x)) < 1e-6f && std::fabs(xyz[3 * i + 2] - float(cl.origin[2] + q.z)) < 1e-6f &&
                 col[3 * i] == q.r && col[3 * i + 2] == q.b;
    }
    CHECK(pts_same, "%s points (%zu vs %d)", pp.c_str(), xyz.size() / 3, cl.n);
    ++checked;
  }
  CHECK(checked == 3, "views checked %d", checked);
  std::printf("  PNG(rgb·depth·mask) %d 벌 디코드·화소 일치, PLY 점·색 일치\n", checked);

  // view.json 에 경로
  const std::string vj = slurp(fs::path(dir) / "view.json");
  CHECK(vj.find("\"rgbd\":{\"rgb\":\"objects/O") != std::string::npos, "view.json rgbd");
  CHECK(vj.find("\"movable\":") != std::string::npos, "view.json movable");
  CHECK(vj.find("_mask.png\"}") != std::string::npos && vj.find("\"points\":{\"path\":\"objects/O") != std::string::npos, "view.json mask/points");

#ifdef SM_TEST_SPARK_DSG
  using namespace spark_dsg;
  auto G = DynamicSceneGraph::load((fs::path(dir) / "scene.json").string());
  CHECK(G != nullptr, "load scene.json");
  if (G) {
    CHECK(G->metadata().contains("stamp") && G->metadata().contains("robot_pose") && G->metadata().contains("grid"), "graph metadata");
    int n = 0, with = 0;
    for (const auto& [id, node] : G->getLayer(DsgLayers::OBJECTS).nodes()) {
      ++n;
      const auto& a = node->attributes<ObjectNodeAttributes>();
      const auto& m = a.metadata();
      for (const char* k : {"state", "n_obs", "score", "first_pos", "structural", "handled", "movable"})
        CHECK(m.contains(k), "node %s missing %s", NodeSymbol(id).str().c_str(), k);
      CHECK(!a.name.empty() && a.bounding_box.isValid(), "name/bbox");
      if (!m.contains("rgbd")) continue;
      ++with;
      const auto& r = m["rgbd"];
      const std::string rgb = r["rgb"].get<std::string>(), dep = r["depth"].get<std::string>();
      const uint32_t oid = uint32_t(NodeSymbol(id).categoryId());
      CHECK(rgb == "objects/O" + std::to_string(oid) + "_rgb.png" && dep == "objects/O" + std::to_string(oid) + "_depth.png", "paths %s",
            rgb.c_str());
      CHECK(fs::exists(fs::path(dir) / rgb) && fs::exists(fs::path(dir) / dep), "png files exist");
      CHECK(r["box_px"].size() == 4 && r["cam_T"].size() == 12 && r.contains("stamp") && r.contains("mask_area") && r.contains("depth_m"),
            "rgbd fields");
      sm_view v{};
      if (sm_snap_view(s.s, oid, &v) == 1) {
        CHECK(std::fabs(r["depth_m"].get<double>() - v.depth_m) < 1e-3 && r["box_px"][2].get<int>() == v.box_px[2], "rgbd values");
        double dT = 0;
        for (int i = 0; i < 12; ++i) dT = std::max(dT, std::fabs(r["cam_T"][i].get<double>() - v.cam_T[i]));
        CHECK(dT < 1e-5, "cam_T diff %.2e", dT);
      }
      CHECK(r.contains("mask") && r["mask"].get<std::string>() == "objects/O" + std::to_string(oid) + "_mask.png", "rgbd.mask");
      CHECK(m.contains("points"), "node points");
      if (m.contains("points")) {
        const auto& P = m["points"];
        sm_cloud cl{};
        sm_snap_points(s.s, oid, &cl);
        CHECK(P["path"].get<std::string>() == "objects/O" + std::to_string(oid) + "_points.ply" && P["n"].get<int>() == cl.n &&
                  std::fabs(P["voxel"].get<double>() - 0.02) < 1e-9 && std::fabs(P["stamp"].get<double>() - cl.stamp) < 1e-9,
              "points meta");
      }
      if (a.name == "sofa" && m.contains("movable")) CHECK(m["movable"].get<bool>() == false, "sofa movable");
      if (a.name == "cup" && m.contains("movable")) CHECK(m["movable"].get<bool>() == true, "cup movable");
    }
    std::printf("  scene.json: OBJECTS %d 노드, rgbd %d (spark_dsg 로 다시 읽음)\n", n, with);
    CHECK(n == 3 && with == 3, "nodes %d with rgbd %d", n, with);
  }
#endif
  // 새 판: 옛 PNG 지움
  sm_reset(g.c);
  sm_set_labels(g.c, kLabels.data(), int(kLabels.size()));
  sm_save_dsg(g.c, dir.c_str());
  int left = 0;
  for (const auto& e : fs::directory_iterator(fs::path(dir) / "objects")) left += e.path().extension() == ".png" || e.path().extension() == ".ply";
  CHECK(left == 0, "stale png after reset %d", left);
}

// ---- 6. 모양(점 구름) ----
// 점 p(map) → 카메라 화소·깊이(cam_T = map ← 카메라 광학)
static void project(const double T[12], const double p[3], double* u, double* v, double* z) {
  const double d[3] = {p[0] - T[3], p[1] - T[7], p[2] - T[11]};
  const double xc = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
  const double yc = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
  const double zc = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
  *u = FX * xc / zc + CX;
  *v = FX * yc / zc + CX;
  *z = zc;
}

struct CloudCheck {
  int n = 0;
  double max_dz = 0;           // 깊이 오차 m
  int outside = 0;             // 사각형 밖으로 투영된 점
  int wrong_rgb = 0;
  double mean[3] = {0, 0, 0};
};
static CloudCheck checkCloud(const sm_snapshot_t* s, uint32_t id, const double T[12], const Rect& r) {
  CloudCheck c;
  sm_cloud cl{};
  if (sm_snap_points(s, id, &cl) != 1) return c;
  c.n = cl.n;
  for (int i = 0; i < cl.n; ++i) {
    const sm_cloud_pt& q = cl.pts[i];
    const double p[3] = {cl.origin[0] + q.x, cl.origin[1] + q.y, cl.origin[2] + q.z};
    double u, v, z;
    project(T, p, &u, &v, &z);
    c.max_dz = std::max(c.max_dz, std::fabs(z - r.depth));
    c.outside += u < r.x0 - 0.5 || u > r.x1 + 0.5 || v < r.y0 - 0.5 || v > r.y1 + 0.5;
    c.wrong_rgb += q.r != r.rgb[0] || q.g != r.rgb[1] || q.b != r.rgb[2];
    for (int k = 0; k < 3; ++k) c.mean[k] += p[k] / std::max(1, cl.n);
  }
  return c;
}

static void testCloud() {
  std::printf("[cloud]\n");
  // (a) 자리·색: 컵 사각형 + 뒤 벽 이상값 20 % — 남은 점은 모두 1.5 m 면 위, 사각형 안, 컵 색
  {
    Rig g(kLabels);
    Rect cup = R(C_CUP, 300, 300, 360, 360, 1.5f, 250, 10, 20);
    cup.out_frac = 0.2f;
    cup.out_depth = 2.6f;
    for (int k = 0; k < 5; ++k) g.kf(0.2 * k, {cup});
    Snap s(g.c);
    const sm_object* o = s.byName("cup");
    sm_view v{};
    CHECK(o && sm_snap_view(s.s, o->id, &v) == 1, "cup view");
    if (o) {
      const CloudCheck c = checkCloud(s.s, o->id, v.cam_T, cup);
      sm_cloud cl{};
      sm_snap_points(s.s, o->id, &cl);
      std::printf("  cup 점 %d (복셀 %.2f), 깊이 오차 최대 %.4f m, 사각형 밖 %d, 색 틀림 %d\n", c.n, cl.voxel, c.max_dz, c.outside, c.wrong_rgb);
      CHECK(c.n > 20 && c.max_dz < 0.01 && c.outside == 0 && c.wrong_rgb == 0, "cup cloud");
      // 마스크(best view): 상자 72→ 가운데는 255, 가장자리(여유)는 0
      CHECK(v.mask && v.mask[(v.h / 2) * v.w + v.w / 2] == 255 && v.mask[0] == 0, "best-view mask");
    }
  }
  // (b) 한도: 복셀 0.005·한도 500 으로 큰 소파를 거듭 봄 → 늘 500 이하
  {
    Rig g(kLabels);
    sm_set_cloud_params(g.c, 0.005, 500);
    int maxn = 0;
    for (int k = 0; k < 8; ++k) {
      g.kf(0.2 * k, {R(C_SOFA, 60 + 5 * k, 350, 660, 520, 1.8f, 30, 30, 160)});
      Snap s(g.c);
      const sm_object* o = s.byName("sofa");
      sm_cloud cl{};
      if (o && sm_snap_points(s.s, o->id, &cl) == 1) maxn = std::max(maxn, cl.n);
    }
    std::printf("  한도 500(복셀 0.005): 최대 %d 점\n", maxn);
    CHECK(maxn > 300 && maxn <= 500, "cap %d", maxn);
  }
  // (c) 들기: 팔 끝을 컵에 대고 그리퍼 닫음 → 팔을 +0.3 m(x) 옮기면 구름도 같이 0.3 m
  {
    Rig g(kLabels);
    const Rect cup = R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0);
    for (int k = 0; k < 4; ++k) g.kf(0.2 * k, {cup});
    double before[3] = {0, 0, 0}, opos[3] = {0, 0, 0};
    uint32_t id = 0;
    {
      Snap s(g.c);
      const sm_object* o = s.byName("cup");
      sm_view v{};
      if (o && sm_snap_view(s.s, o->id, &v) == 1) {
        id = o->id;
        const CloudCheck c = checkCloud(s.s, o->id, v.cam_T, cup);
        for (int k = 0; k < 3; ++k) { before[k] = c.mean[k]; opos[k] = o->pos[k]; }
      }
    }
    // 베이스는 원점(속도 0) → map = 베이스. 왼 팔 끝을 물체 자리로, 닫음
    g.q[17] = float(opos[0]); g.q[18] = float(opos[1]); g.q[19] = float(opos[2]);
    g.q[24] = g.q[25] = 0.02f;
    g.kf(1.0, {}, 3.0f, true, false);
    g.q[17] += 0.3f;
    g.kf(1.2, {}, 3.0f, true, false);
    Snap s(g.c);
    const sm_object* o = s.byName("cup");
    sm_cloud cl{};
    double mean[3] = {0, 0, 0};
    if (o && sm_snap_points(s.s, id, &cl) == 1)
      for (int i = 0; i < cl.n; ++i) {
        mean[0] += (cl.origin[0] + cl.pts[i].x) / cl.n;
        mean[1] += (cl.origin[1] + cl.pts[i].y) / cl.n;
        mean[2] += (cl.origin[2] + cl.pts[i].z) / cl.n;
      }
    std::printf("  들기: 상태 %d(3=held), 구름 중심 이동 (%.3f, %.3f, %.3f)\n", o ? o->state : -1, mean[0] - before[0], mean[1] - before[1],
                mean[2] - before[2]);
    CHECK(o && o->state == SM_HELD, "held state");
    CHECK(std::fabs(mean[0] - before[0] - 0.3) < 1e-3 && std::fabs(mean[1] - before[1]) < 1e-3, "held cloud follows hand");
  }
  // (d) 사라짐 → 구름 유지, 다른 자리(전에 본 곳)에 나타난 같은 이름 물체를 다시 이음(옮겨짐) → 새 자리 점만.
  //     잇기는 새 물체가 link_min_obs 번 보이고, 그 자리를 link_view_gap_s(5 s) 넘게 전에 본 적 있어야
  {
    Rig g(kLabels);
    const Rect a = R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0);
    const Rect b = R(C_CUP, 520, 300, 560, 340, 1.5f, 250, 0, 0);
    double t = 0;
    for (int k = 0; k < 6; ++k, t += 0.2) g.kf(t, {a});
    int n_gone = -1;
    for (; t < 6.0; t += 0.2) g.kf(t, {});
    {
      Snap s(g.c);
      const sm_object* o = s.byName("cup");
      sm_cloud cl{};
      if (o && o->state == SM_GONE && sm_snap_points(s.s, o->id, &cl) == 1) n_gone = cl.n;
    }
    uint32_t gone_id = 0;
    {
      Snap s0(g.c);
      if (const sm_object* o0 = s0.byName("cup")) gone_id = o0->id;
    }
    for (int k = 0; k < 3; ++k, t += 0.2) g.kf(t, {b});
    Snap s(g.c);
    const sm_object* o = s.byName("cup");
    CHECK(o && o->id == gone_id && s.objs().size() == 1, "relinked id %u (gone %u), objects %zu", o ? o->id : 0, gone_id, s.objs().size());
    sm_view v{};
    CloudCheck c;
    if (o && sm_snap_view(s.s, o->id, &v) == 1) c = checkCloud(s.s, o->id, v.cam_T, b);
    std::printf("  사라짐: 구름 %d 점 유지 → 다른 자리(상태 %d, 2=moved): 점 %d, 새 사각형 밖 %d\n", n_gone, o ? o->state : -1, c.n, c.outside);
    CHECK(n_gone > 20, "gone keeps cloud %d", n_gone);
    CHECK(o && o->state == SM_MOVED && c.n > 20 && c.outside == 0, "moved cloud rebuilt (outside %d)", c.outside);
    sm_reset(g.c);
    Snap s2(g.c);
    CHECK(s2.objs().empty(), "reset clears");
  }
}

// keyframe 시간: slam2d 만(검출 없음) / slam2d + objmap + best view(검출 6, 호스트 RGBA 자르기)
static void testTiming() {
  std::printf("[timing]\n");
  std::vector<Rect> rs = {R(C_CUP, 300, 300, 340, 340, 1.5f, 250, 0, 0), R(C_BOOK, 420, 270, 480, 345, 1.8f, 0, 0, 250),
                          R(C_SOFA, 60, 350, 660, 440, 1.7f, 30, 30, 160), R(C_WALL, 0, 0, 720, 200, 3.0f, 200, 200, 200),
                          R(C_FRAME, 500, 220, 600, 300, 2.9f, 10, 200, 10), R(C_RUG, 200, 610, 520, 700, 2.0f, 200, 10, 10)};
  double a = 0, b = 0;
  const int N = 30;
  {
    Rig g(kLabels);
    for (int k = 0; k < N; ++k) a += g.kf(0.2 * k, rs, 3.0f, true, false);
  }
  {
    Rig g(kLabels);
    for (int k = 0; k < N; ++k) {
      for (auto& r : rs) r.score = 0.5f + 0.4f * float(k) / N;   // 점수가 오르며 매번 best view 바뀜(최악)
      b += g.kf(0.2 * k, rs);
    }
  }
  std::printf("  keyframe 720²: slam2d 만 %.3f ms, + objmap·best view(검출 6, 매번 자르기) %.3f ms\n", a / N, b / N);
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string dir = argc > 1 ? argv[1] : (fs::temp_directory_path() / "sm_test_objmem").string();
  testStructures();
  testBoxes();
  testGone();
  testBestView();
  testCloud();
  testSave(dir);
  testInspect(dir);
  testTiming();
  std::printf(g_fail ? "FAILED %d\n" : "ok\n", g_fail);
  return g_fail ? 1 : 0;
}
