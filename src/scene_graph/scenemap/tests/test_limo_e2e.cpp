// LIMO + OMX-F 끝에서 끝까지(C ABI 만): config "robot": "limo_omx" 로 만들고, LIMO proprio(오도메트리 자세는 map 과 다른 원점)
// + 합성 깊이(순기구학 깊이 카메라 자세에서 광선 추적한 벽 둘·바닥·컵) + 컵 검출 마스크를 넣는다.
//   1) 첫 자리 A 에서: 벽 칸이 점유, 벽 앞 칸이 빈칸, 벽 뒤가 모름 — 격자가 map 의 맞는 자리에
//   2) 컵이 확정 물체 하나로 정답 자리(3 cm 안)에
//   3) 오도메트리로 B 까지 0.5 m 가고 0.25 rad 돌면 sm_snap_pose 가 그만큼, 컵은 여전히 하나·같은 자리
//   4) 팔을 뻗어 잡는 점을 컵에 대고 그리퍼를 열었다 컵 폭(4 cm)에서 닫혀 멈추면 SM_HELD, 베이스가 움직이면 컵이 따라감
//   5) 모르는 로봇 이름(r1pro 포함)은 sm_create NULL
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "scenemap.h"

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
const double FX = 80.0 / std::tan(71.0 / 2 * M_PI / 180), FY = FX, CX = 79.5, CY = 59.5;   // Dabai 컬러 H-FOV 71°
// 장면(map 프레임 = 첫 proprio 베이스): 벽 x = 2.5, 벽 y = 1.0, 바닥 z = 0, 컵 상자
constexpr double kWallX = 2.5, kWallY = 1.0;
double cup_lo[3], cup_hi[3];
// 오도메트리 프레임 원점(map 과 다름): odom = O ∘ map
constexpr double OX = 5.0, OY = -2.0, OTH = 0.3;

struct Pose { double x, y, th; };

// LIMO proprio: 오도메트리 자세(map 자세를 O 로 옮김) + twist + 팔 + 그리퍼
std::vector<float> proprio(const Pose& p, const double arm[5], double grip, const double tw[3]) {
  std::vector<float> q(12, 0.f);
  const double c = std::cos(OTH), s = std::sin(OTH);
  q[SM_LIMO_ODOM_X] = float(OX + c * p.x - s * p.y);
  q[SM_LIMO_ODOM_Y] = float(OY + s * p.x + c * p.y);
  q[SM_LIMO_ODOM_YAW] = float(OTH + p.th);
  q[SM_LIMO_VX] = float(tw[0]); q[SM_LIMO_VY] = float(tw[1]); q[SM_LIMO_WZ] = float(tw[2]);
  for (int k = 0; k < 5; ++k) q[SM_LIMO_ARM_Q + k] = float(arm[k]);
  q[SM_LIMO_GRIPPER] = float(grip);
  return q;
}

// 광선 추적: map ← 카메라 광학 T(3×4). 깊이(광학 z) + 컵 마스크
void render(const double T[12], std::vector<float>* depth, std::vector<uint8_t>* cup) {
  depth->assign(size_t(W) * H, 0.f);
  cup->assign(size_t(W) * H, 0);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const double d[3] = {(u - CX) / FX, (v - CY) / FY, 1.0};
      double o[3], r[3];
      for (int i = 0; i < 3; ++i) {
        o[i] = T[i * 4 + 3];
        r[i] = T[i * 4] * d[0] + T[i * 4 + 1] * d[1] + T[i * 4 + 2] * d[2];
      }
      double best = 1e9;
      bool is_cup = false;
      if (r[0] > 1e-9) best = std::fmin(best, (kWallX - o[0]) / r[0]);
      if (r[1] > 1e-9) best = std::fmin(best, (kWallY - o[1]) / r[1]);
      if (r[2] < -1e-9) best = std::fmin(best, -o[2] / r[2]);
      double t0 = 0, t1 = 1e9;   // 상자 slab
      bool hit = true;
      for (int i = 0; i < 3 && hit; ++i) {
        if (std::fabs(r[i]) < 1e-12) { hit = o[i] >= cup_lo[i] && o[i] <= cup_hi[i]; continue; }
        double a = (cup_lo[i] - o[i]) / r[i], b = (cup_hi[i] - o[i]) / r[i];
        if (a > b) std::swap(a, b);
        t0 = std::fmax(t0, a); t1 = std::fmin(t1, b);
        hit = t0 <= t1;
      }
      if (hit && t0 > 0 && t0 < best) { best = t0; is_cup = true; }
      if (best < 8.0) (*depth)[size_t(v) * W + u] = float(best);   // 광학 z = t (d 의 z = 1)
      (*cup)[size_t(v) * W + u] = is_cup;
    }
}

// map ← 광학 = 베이스 자세 ∘ 순기구학(베이스 ← 광학)
void camInMap(const Pose& p, const double B[12], double T[12]) {
  const double c = std::cos(p.th), s = std::sin(p.th);
  for (int k = 0; k < 4; ++k) {
    T[k] = c * B[k] - s * B[4 + k];
    T[4 + k] = s * B[k] + c * B[4 + k];
    T[8 + k] = B[8 + k];
  }
  T[3] += p.x;
  T[7] += p.y;
}

int8_t cellAt(const sm_grid& g, double x, double y) {
  const int ix = int(std::floor((x - g.origin[0]) / g.resolution)), iy = int(std::floor((y - g.origin[1]) / g.resolution));
  if (ix < 0 || iy < 0 || ix >= g.width || iy >= g.height) return -1;
  return g.cells[size_t(iy) * g.width + ix];
}

const sm_object* findCup(sm_snapshot_t* s, int* n) {
  const sm_object* o = nullptr;
  *n = sm_snap_objects(s, &o);
  for (int i = 0; i < *n; ++i)
    if (!std::strcmp(o[i].name, "cup")) return &o[i];
  return nullptr;
}
}  // namespace

int main() {
  CHECK(sm_create("{\"robot\": \"spot\"}") == nullptr, "unknown robot name rejected");
  sm_ctx* dflt = sm_create(nullptr);
  CHECK(dflt && sm_get_robot(dflt) == SM_ROBOT_LIMO_OMX, "default robot is LIMO + OMX-F");
  sm_destroy(dflt);
  CHECK(sm_create("{\"robot\": \"r1pro\"}") == nullptr, "r1pro no longer exists");
  sm_ctx* c = sm_create("{\"robot\": \"limo_omx\"}");
  CHECK(c && sm_get_robot(c) == SM_ROBOT_LIMO_OMX, "config robot limo_omx");
  if (!c) return 1;
  const char* labels[] = {"cup", "wall"};
  sm_set_labels(c, labels, 2);

  const double home[5] = {0, 1.3, -1.9, 0.7, 0};
  const double reach[5] = {0, 0.9, 0.2, 0.4, 0};   // 앞으로 뻗어 손끝을 낮게
  const double zero3[3] = {0, 0, 0};
  {
    std::vector<float> q = proprio({0, 0, 0}, home, 0, zero3);
    sm_proprio p{0.0, q.data(), 11};
    CHECK(sm_push_proprio(c, &p) < 0, "11-value proprio rejected (needs 12)");
  }
  // 컵: B 에서 팔을 뻗은 손끝 자리(map). B = A + 0.5 m 앞, 0.25 rad 왼쪽 돌기
  const Pose A{0, 0, 0}, B{0.5, 0, 0.25};
  sm_body_fk fk_home, fk_reach;
  {
    std::vector<float> q = proprio(A, home, 0, zero3);
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk_home);
    q = proprio(A, reach, 0, zero3);
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk_reach);
  }
  double cup_c[3];
  {
    const double* e = fk_reach.T_eef[0];
    const double cs = std::cos(B.th), sn = std::sin(B.th);
    cup_c[0] = B.x + cs * e[3] - sn * e[7];
    cup_c[1] = B.y + sn * e[3] + cs * e[7];
    // 컵: 바닥 위 4 × 4 × 10 cm(그리퍼 한도 0.06 m 안), xy 중심 = 잡는 점(높이 e[11] 은 컵 높이 안)
    for (int i = 0; i < 2; ++i) { cup_lo[i] = cup_c[i] - 0.02; cup_hi[i] = cup_c[i] + 0.02; }
    cup_lo[2] = 0.0; cup_hi[2] = 0.10;
    cup_c[2] = 0.05;
    std::printf("cup centre (map) %.3f %.3f %.3f (reach eef in base: %.3f %.3f %.3f)\n", cup_c[0], cup_c[1], cup_c[2], e[3], e[7], e[11]);
  }

  double t = 0;
  const double dt = 0.1;
  std::vector<float> depth;
  std::vector<uint8_t> mask;
  auto step = [&](const Pose& p, const double arm[5], double grip, const double tw[3], bool image) {
    t += dt;
    std::vector<float> q = proprio(p, arm, grip, tw);
    sm_proprio pp{t, q.data(), int(q.size())};
    sm_push_proprio(c, &pp);
    if (!image) return;
    sm_body_fk fk;
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk);
    double T[12];
    camInMap(p, fk.T_cam[0], T);
    render(T, &depth, &mask);
    // 검출: 컵 마스크(화소 격자 그대로)
    std::vector<uint32_t> bits((size_t(W) * H + 31) / 32, 0);
    int x0 = W, y0 = H, x1 = -1, y1 = -1, npx = 0;
    for (int v = 0; v < H; ++v)
      for (int u = 0; u < W; ++u)
        if (mask[size_t(v) * W + u]) {
          const size_t k = size_t(v) * W + u;
          bits[k >> 5] |= 1u << (k & 31);
          x0 = std::min(x0, u); y0 = std::min(y0, v); x1 = std::max(x1, u); y1 = std::max(y1, v);
          ++npx;
        }
    const int32_t cls = 0;
    const float score = 0.9f;
    const float box[4] = {float(x0), float(y0), float(x1 + 1), float(y1 + 1)};
    sm_detections d{};
    d.stamp = t; d.cam = 0; d.img_w = W; d.img_h = H; d.n = npx > 0 ? 1 : 0;
    d.cls = &cls; d.score = &score; d.box = box;
    d.mask_w = W; d.mask_h = H; d.mask_sx = 1; d.mask_sy = 1; d.mask_ox = 0; d.mask_oy = 0; d.mask_bits = bits.data();
    sm_image im{t, 0, W, H, nullptr, depth.data(), FX, FY, CX, CY};
    sm_push_image(c, &im, &d);
    // 손목 카메라(cam 1)는 지도에 안 씀: 넣어도 상태가 그대로여야 함
    sm_image wr{t, 1, W, H, nullptr, depth.data(), FX, FY, CX, CY};
    sm_push_image(c, &wr, nullptr);
  };

  // 1) A 에서 keyframe 3 번(서 있음)
  for (int k = 0; k < 3; ++k) step(A, home, 0, zero3, true);
  sm_snapshot_t* s = nullptr;
  sm_snapshot(c, &s);
  sm_grid g{};
  sm_snap_map(s, &g);
  {
    int8_t wall = -1;
    for (double dx = -0.05; dx <= 0.05; dx += 0.025) wall = std::max<int8_t>(wall, cellAt(g, kWallX + dx, -0.3));
    int8_t side = -1;
    for (double dy = -0.05; dy <= 0.05; dy += 0.025) side = std::max<int8_t>(side, cellAt(g, 2.0, kWallY + dy));
    const int8_t freec = cellAt(g, 1.5, -0.4), behind = cellAt(g, kWallX + 0.5, -0.3);
    CHECK(wall > 50, "front wall x = %.1f occupied in map (%d)", kWallX, wall);
    CHECK(side > 50, "side wall y = %.1f occupied in map (%d)", kWallY, side);
    CHECK(freec >= 0 && freec < 50, "floor in front of the wall is free (%d)", freec);
    CHECK(behind == -1, "behind the wall unknown (%d)", behind);
    // 로봇 바로 옆(몸 원 self_r 0.22 안)에 점유가 생기지 않음
    int8_t near = -1;
    for (double a = 0; a < 2 * M_PI; a += 0.3) near = std::max<int8_t>(near, cellAt(g, 0.15 * std::cos(a), 0.15 * std::sin(a)));
    CHECK(near < 50, "no occupied cell on the robot body (%d)", near);
  }
  int n = 0;
  const sm_object* cup = findCup(s, &n);
  CHECK(cup != nullptr && n == 1, "one object, the cup (n = %d)", n);
  // objmap 위치 = 보이는 면 점의 중앙값 → 컵 상자(8 cm)에서 2 cm 안이면 맞는 자리
  auto boxDist = [&](const double* q) {
    double d2 = 0;
    for (int i = 0; i < 3; ++i) { const double e = std::fmax(0.0, std::fmax(cup_lo[i] - q[i], q[i] - cup_hi[i])); d2 += e * e; }
    return std::sqrt(d2);
  };
  if (cup) {
    const double e = std::hypot(cup->pos[0] - cup_c[0], cup->pos[1] - cup_c[1]);
    CHECK(boxDist(cup->pos) < 0.02 && e < 0.06, "cup at its true place: %.3f m outside the cup box, centre xy %.3f m (pos %.3f %.3f %.3f)",
          boxDist(cup->pos), e, cup->pos[0], cup->pos[1], cup->pos[2]);
  }
  sm_snapshot_release(s);

  // 2) A → B: 2 s 동안 오도메트리로 앞 0.5 m + 0.25 rad(호를 따라), 매 스텝 영상
  const int N = 20;
  for (int k = 1; k <= N; ++k) {
    const double a = double(k) / N;
    const Pose p{A.x + (B.x - A.x) * a, 0, B.th * a};
    const double tw[3] = {0.40, 0, 0.0};   // twist 는 일부러 틀림(0.4 m/s, 회전 없음): 적분은 오도메트리 자세 차로 해야 B 에 닿음
    step(p, home, 0, tw, true);
  }
  sm_snapshot(c, &s);
  {
    const sm_pose2 P = sm_snap_pose(s);
    CHECK(std::hypot(P.x - B.x, P.y - B.y) < 0.03 && std::fabs(P.yaw - B.th) < 2 * M_PI / 180, "map pose at B (%.3f, %.3f, %.1f°) vs (%.3f, %.3f, %.1f°)",
          P.x, P.y, P.yaw * 180 / M_PI, B.x, B.y, B.th * 180 / M_PI);
    cup = findCup(s, &n);
    CHECK(cup != nullptr && n == 1, "still one object after driving (n = %d)", n);
    if (cup) CHECK(boxDist(cup->pos) < 0.02, "cup stays at its place after driving: %.3f m outside the cup box", boxDist(cup->pos));
  }
  sm_snapshot_release(s);

  // 3) 팔 뻗기(그리퍼 열림 1.0) → 닫기(0.41 rad = 4 cm 를 쥔 각도, E0) → 멈추면(0.2 s) 컵을 듦. 그 뒤 베이스 0.3 m 앞으로: 컵이 따라감(그리는 컵도 손과 함께 옮김).
  //    proprio 는 cam 0 영상 stamp 까지 적분되므로(잡기 규칙도 그때) 매 스텝 영상을 넣는다
  for (int k = 0; k < 3; ++k) step(B, reach, 1.0, zero3, true);
  for (int k = 0; k < 4; ++k) step(B, reach, 0.41, zero3, true);
  double held0[3] = {0, 0, 0};
  sm_snapshot(c, &s);
  cup = findCup(s, &n);
  CHECK(cup && cup->state == SM_HELD, "gripper closes at the cup → SM_HELD (state %d)", cup ? cup->state : -1);
  if (cup) for (int i = 0; i < 3; ++i) held0[i] = cup->pos[i];
  sm_snapshot_release(s);
  const Pose C{B.x + 0.3 * std::cos(B.th), B.y + 0.3 * std::sin(B.th), B.th};
  for (int k = 1; k <= 10; ++k) {
    const Pose p{B.x + (C.x - B.x) * k / 10.0, B.y + (C.y - B.y) * k / 10.0, B.th};
    const double d[2] = {(C.x - B.x) / 10.0, (C.y - B.y) / 10.0};
    for (int i = 0; i < 2; ++i) { cup_lo[i] += d[i]; cup_hi[i] += d[i]; }
    const double tw[3] = {0.3, 0, 0};
    step(p, reach, 0.41, tw, true);
  }
  sm_snapshot(c, &s);
  cup = findCup(s, &n);
  if (cup) {
    const double mv = std::hypot(cup->pos[0] - held0[0], cup->pos[1] - held0[1]);
    CHECK(cup->state == SM_HELD && std::fabs(mv - 0.3) < 0.03, "held cup follows the hand: moved %.3f m (base 0.300), state %d", mv,
          cup->state);
  } else {
    CHECK(false, "cup missing after carrying");
  }
  {
    const sm_pose2 P = sm_snap_pose(s);
    CHECK(std::hypot(P.x - C.x, P.y - C.y) < 0.03, "map pose at C (%.3f, %.3f) vs (%.3f, %.3f)", P.x, P.y, C.x, C.y);
  }
  sm_snapshot_release(s);
  // 4) 놓기: 그리퍼 열면 SM_MOVED(0.3 m 옮겼으니)
  step(C, reach, 1.0, zero3, true);
  sm_snapshot(c, &s);
  cup = findCup(s, &n);
  CHECK(cup && cup->state == SM_MOVED, "gripper opens → released, SM_MOVED (state %d)", cup ? cup->state : -1);
  sm_snapshot_release(s);

  sm_destroy(c);
  std::printf(fail ? "FAILED %d\n" : "all ok\n", fail);
  return fail ? 1 : 0;
}
