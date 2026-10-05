// LIMO 잡기 규칙(held)과 팔 가림 — C ABI 로, 합성 깊이(바닥·탁자·컵 + 순기구학 팔 캡슐을 광선 추적) + 검출 마스크.
//   1) 탁자 앞을 팔이 가림(검출기가 가린 팔 화소를 탁자 마스크에 넣음) → 그리퍼가 빈손으로 끝까지 닫힘:
//      탁자는 held 가 아니고 옮겨짐·사라짐도 아님, 상자가 팔 쪽으로 자라지 않음.
//   2) 컵(4 × 4 × 8 cm)이 잡는 점에: 그리퍼 열린 채 → held 아님, 끝까지 닫힘(0 rad, 빈손) → held 아님,
//      다시 열고 4 cm 를 쥔 각도(0.41 rad, E0)로 닫혀 멈춤 → held
//   3) 큰 컵(8 × 8 × 8 cm, 그리퍼 한도 넘음)은 0.41 rad 로 닫혀도 held 아님
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <algorithm>
#include <vector>

#include "scenemap.h"
#include "fake_siglip.h"
#include "scenemap/fk.hpp"

using namespace scenemap;

namespace {
int fail = 0;
#define CHECK(c, ...)                                 \
  do {                                                \
    const bool ok_ = (c);                             \
    std::printf("%s: ", ok_ ? "ok  " : "FAIL");       \
    std::printf(__VA_ARGS__);                         \
    std::printf("\n");                                \
    fail += !ok_;                                     \
  } while (0)

constexpr int W = 160, H = 120;
const double FX = 80.0 / std::tan(71.0 / 2 * M_PI / 180), FY = FX, CX = 79.5, CY = 59.5;

struct Box { double lo[3], hi[3]; int cls; };   // cls: 0 cup, 1 table
// 팔: 몸통 카메라 앞을 지나 잡는 점이 (0.30, 0, 0.16)(베이스 = map)에, 손가락 축 수평 앞
const double kHome[5] = {0, 1.3, -1.9, 0.7, 0};
const double kFront[5] = {0, 1.0685, -0.3593, -0.7093, 0};

std::vector<float> proprio(const double arm[5], double grip) {
  std::vector<float> q(12, 0.f);
  for (int k = 0; k < 5; ++k) q[SM_LIMO_ARM_Q + k] = float(arm[k]);
  q[SM_LIMO_GRIPPER] = float(grip);
  return q;
}

bool slab(const double o[3], const double r[3], const double lo[3], const double hi[3], double* t) {
  double t0 = 0, t1 = 1e9;
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(r[i]) < 1e-12) { if (o[i] < lo[i] || o[i] > hi[i]) return false; continue; }
    double a = (lo[i] - o[i]) / r[i], b = (hi[i] - o[i]) / r[i];
    if (a > b) std::swap(a, b);
    t0 = std::fmax(t0, a); t1 = std::fmin(t1, b);
    if (t0 > t1) return false;
  }
  *t = t0;
  return t0 > 0;
}

// 팔 = 뼈대 선분(omx_link0 .. 팔 끝)을 따라 1 cm 마다 반지름 0.025 m 구
void armSpheres(const float* q, std::vector<std::array<double, 3>>* c) {
  LimoFk f;
  computeLimoFk(q, &f);
  c->clear();
  for (int k = 0; k + 1 < LimoFk::kPts; ++k) {
    const double* a = f.pts[k];
    const double* b = f.pts[k + 1];
    const double L = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
    const int n = std::max(1, int(L / 0.01));
    for (int i = 0; i <= n; ++i) {
      const double u = double(i) / n;
      c->push_back({a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2])});
    }
  }
}

struct Rig {
  sm_ctx* c = nullptr;
  std::vector<Box> boxes;
  bool lump_arm = false;     // 검출기가 탁자 상자 안 팔 화소를 탁자 마스크에 넣음
  double t = 0;
  std::vector<float> depth;
  std::vector<int> hit;      // −1 바닥·없음, −2 팔, 그 밖 = boxes 번호
  explicit Rig(const char* env) {
    if (env) setenv("SM_OBJ_PARAMS", env, 1);
    c = sm_create("{\"robot\": \"limo_omx\"}");
    if (env) unsetenv("SM_OBJ_PARAMS");
    const char* labels[] = {"cup", "table"};
    sm_set_labels(c, labels, 2);
    fake_siglip::textModel(c, 2);
  }
  ~Rig() { sm_destroy(c); }
  void step(const double arm[5], double grip) {
    t += 0.1;
    std::vector<float> q = proprio(arm, grip);
    sm_proprio pp{t, q.data(), int(q.size())};
    sm_push_proprio(c, &pp);
    sm_body_fk fk;
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk);
    const double* T = fk.T_cam[0];   // 베이스 = map(로봇은 원점에 서 있음)
    std::vector<std::array<double, 3>> sph;
    armSpheres(q.data(), &sph);
    depth.assign(size_t(W) * H, 0.f);
    hit.assign(size_t(W) * H, -1);
    for (int v = 0; v < H; ++v)
      for (int u = 0; u < W; ++u) {
        const double d[3] = {(u - CX) / FX, (v - CY) / FY, 1.0};
        double o[3], r[3];
        for (int i = 0; i < 3; ++i) {
          o[i] = T[i * 4 + 3];
          r[i] = T[i * 4] * d[0] + T[i * 4 + 1] * d[1] + T[i * 4 + 2] * d[2];
        }
        double best = 1e9;
        int who = -1;
        if (r[2] < -1e-9) best = -o[2] / r[2];   // 바닥
        if (r[0] > 1e-9) best = std::fmin(best, (3.0 - o[0]) / r[0]);   // 뒤 벽 x = 3
        for (size_t b = 0; b < boxes.size(); ++b) {
          double tb;
          if (slab(o, r, boxes[b].lo, boxes[b].hi, &tb) && tb < best) { best = tb; who = int(b); }
        }
        const double rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for (const auto& s : sph) {   // 광선–구
          const double oc[3] = {o[0] - s[0], o[1] - s[1], o[2] - s[2]};
          const double bq = oc[0] * r[0] + oc[1] * r[1] + oc[2] * r[2];
          const double cq = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - 0.025 * 0.025;
          const double disc = bq * bq - rr * cq;
          if (disc < 0) continue;
          const double ts = (-bq - std::sqrt(disc)) / rr;
          if (ts > 0 && ts < best) { best = ts; who = -2; }
        }
        if (best < 8.0) depth[size_t(v) * W + u] = float(best);
        hit[size_t(v) * W + u] = who;
      }
    // 검출: 상자마다 마스크(보이는 화소). lump_arm 이면 탁자 상자(팔 없이 본 탁자 화소의 상자) 안 팔 화소도 탁자
    const size_t words = (size_t(W) * H + 31) / 32;
    std::vector<int32_t> cls;
    std::vector<float> score, box;
    std::vector<uint32_t> bits;
    for (size_t b = 0; b < boxes.size(); ++b) {
      int x0 = W, y0 = H, x1 = -1, y1 = -1;
      for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u)
          if (hit[size_t(v) * W + u] == int(b)) { x0 = std::min(x0, u); y0 = std::min(y0, v); x1 = std::max(x1, u); y1 = std::max(y1, v); }
      if (x1 < 0) continue;
      const bool lump = lump_arm && boxes[b].cls == 1;
      const size_t base = bits.size();
      bits.resize(base + words, 0);
      for (int v = y0; v <= y1; ++v)
        for (int u = x0; u <= x1; ++u) {
          const int h = hit[size_t(v) * W + u];
          if (h == int(b) || (lump && h == -2)) {
            const size_t k = size_t(v) * W + u;
            bits[base + (k >> 5)] |= 1u << (k & 31);
          }
        }
      cls.push_back(boxes[b].cls);
      score.push_back(0.9f);
      box.insert(box.end(), {float(x0), float(y0), float(x1 + 1), float(y1 + 1)});
    }
    sm_detections d{};
    d.stamp = t; d.cam = 0; d.img_w = W; d.img_h = H; d.n = int(cls.size());
    d.cls = cls.data(); d.score = score.data(); d.box = box.data();
    d.mask_w = W; d.mask_h = H; d.mask_sx = 1; d.mask_sy = 1; d.mask_ox = 0; d.mask_oy = 0; d.mask_bits = bits.data();
    sm_image im{t, 0, W, H, nullptr, depth.data(), FX, FY, CX, CY};
    fake_siglip::detEmb(c, &d, 2);
    sm_push_image(c, &im, &d);
  }
  // 이름 name 인 물체(없으면 NULL). 스냅숏은 호출자가 놓음
  const sm_object* find(sm_snapshot_t* s, const char* name) {
    const sm_object* o = nullptr;
    const int n = sm_snap_objects(s, &o);
    for (int i = 0; i < n; ++i)
      if (!std::strcmp(o[i].name, name)) return &o[i];
    return nullptr;
  }
  int stateOf(const char* name, double pos[3] = nullptr) {
    sm_snapshot_t* s = nullptr;
    sm_snapshot(c, &s);
    const sm_object* o = find(s, name);
    const int st = o ? o->state : -1;
    if (o && pos) for (int i = 0; i < 3; ++i) pos[i] = o->pos[i];
    sm_snapshot_release(s);
    return st;
  }
};

// 1) 탁자 + 팔 가림. 돌려줌: 끝 상태
int tableRun(const char* env, double* moved_m, double* min_lo_x, int* ever_bad) {
  Rig r(env);
  r.boxes.push_back({{0.34, -0.35, 0.0}, {0.94, 0.35, 0.30}, 1});
  r.lump_arm = true;
  for (int k = 0; k < 5; ++k) r.step(kHome, 1.0);   // 팔 접고 탁자를 봄(확정)
  double p0[3] = {0, 0, 0};
  const int s0 = r.stateOf("table", p0);
  CHECK(s0 == SM_SEEN, "%s: table confirmed with the arm folded (state %d)", env ? "old rule" : "new rule", s0);
  // 팔을 몸통 카메라 앞으로(탁자 앞면 3 cm 앞까지) 뻗고 그리퍼를 연 채 1 s, 빈손으로 끝까지 닫아 2 s
  *ever_bad = 0;
  *min_lo_x = 1e9;
  for (int k = 0; k < 30; ++k) {
    r.step(kFront, k < 10 ? 1.0 : 0.0);
    sm_snapshot_t* s = nullptr;
    sm_snapshot(r.c, &s);
    if (const sm_object* o = r.find(s, "table")) {
      if (o->state == SM_HELD || o->state == SM_MOVED || o->state == SM_GONE) *ever_bad = 1;
      *min_lo_x = std::fmin(*min_lo_x, o->pos[0] - 0.5 * o->extent[0]);
    }
    sm_snapshot_release(s);
  }
  double p1[3] = {0, 0, 0};
  const int st = r.stateOf("table", p1);
  *moved_m = std::sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[2] - p0[2]) * (p1[2] - p0[2]));
  return st;
}
}  // namespace

int main() {
  // 1) 탁자 아래 가리는 팔
  {
    double mv = 0, lo = 0;
    int bad = 0;
    const int st = tableRun(nullptr, &mv, &lo, &bad);
    CHECK(st == SM_SEEN && !bad, "table under the occluding arm, gripper closed on nothing: not held / moved / gone (state %d, ever bad %d)", st, bad);
    CHECK(mv < 0.03, "table position unchanged by the arm pixels (%.3f m)", mv);
    CHECK(lo > 0.30, "table box did not grow toward the arm (front face x %.3f, true 0.34)", lo);
  }
  // 2) 작은 컵이 잡는 점에(공중 받침 — 몸통 카메라에 보이는 높이)
  {
    Rig r(nullptr);
    r.boxes.push_back({{0.28, -0.0216, 0.12}, {0.32, 0.0184, 0.20}, 0});   // 4 × 4 × 8 cm, 중심 = 잡는 점(0.30, −0.0016, 0.16)
    for (int k = 0; k < 5; ++k) r.step(kHome, 1.0);
    CHECK(r.stateOf("cup") == SM_SEEN, "cup confirmed (state %d)", r.stateOf("cup"));
    for (int k = 0; k < 10; ++k) r.step(kFront, 1.0);   // 팔을 뻗어 컵을 손가락 사이에(열림)
    CHECK(r.stateOf("cup") == SM_SEEN, "open gripper around the cup: not held (state %d)", r.stateOf("cup"));
    for (int k = 0; k < 6; ++k) r.step(kFront, 0.0);    // 끝까지 닫힘 = 손가락 사이에 아무것도 없음
    CHECK(r.stateOf("cup") == SM_SEEN, "gripper closed to 0 rad (empty): not held (state %d)", r.stateOf("cup"));
    for (int k = 0; k < 3; ++k) r.step(kFront, 1.0);
    for (int k = 0; k < 2; ++k) r.step(kFront, 0.41);   // 닫히는 중(아직 안 멈춤)
    for (int k = 0; k < 4; ++k) r.step(kFront, 0.41);   // 4 cm 를 쥐고 멈춤(E0: 0.408–0.417 rad)
    double p[3];
    const int st = r.stateOf("cup", p);
    CHECK(st == SM_HELD, "small cup between closed fingers (0.41 rad ≈ 4 cm gap): held (state %d)", st);
    // 열면 놓음
    r.step(kFront, 1.0);
    CHECK(r.stateOf("cup") != SM_HELD, "gripper opens: released (state %d)", r.stateOf("cup"));
  }
  // 3) 큰 컵(8 cm, 그리퍼 한도 0.06 m 넘음)
  {
    Rig r(nullptr);
    r.boxes.push_back({{0.28, -0.0416, 0.12}, {0.36, 0.0384, 0.20}, 0});
    for (int k = 0; k < 5; ++k) r.step(kHome, 1.0);
    for (int k = 0; k < 10; ++k) r.step(kFront, 1.0);
    for (int k = 0; k < 6; ++k) r.step(kFront, 0.41);
    CHECK(r.stateOf("cup") == SM_SEEN, "8 cm cup (over the 0.06 m gripper limit): not held (state %d)", r.stateOf("cup"));
  }
  std::printf(fail ? "FAILED %d\n" : "all ok\n", fail);
  return fail ? 1 : 0;
}
