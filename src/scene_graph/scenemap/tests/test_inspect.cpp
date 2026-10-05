// 살펴본 정도(inspect.hpp) 시험: 시점 세기(0.3 m·15° 안은 같은 시점, 한도), 가장 가까운 거리, 윗면 칸(광선 추적 합성 깊이 —
// 위에서 본 탁자 = 거의 다, 낮은 카메라 = 앞 줄만·가까우면 앞 모서리에 가림, 가리는 벽 = 0, 2 m 밖 = 0), 상자 자람·합치기의 칸 옮김,
// ObjectMap 통합(켜도 물체·사건이 꺼짐과 같음, 다가가며 거리·시점·윗면), CPU 시간. 덤: 물체 지도 매개변수 문자열(applyParams)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "scenemap/inspect.hpp"
#include "scenemap/objmap.hpp"

using namespace scenemap;

static int bad = 0;
#define CHECK(c, ...)                         \
  do {                                        \
    if (!(c)) {                               \
      std::printf("FAIL %s: ", #c);           \
      std::printf(__VA_ARGS__);               \
      std::printf("\n");                      \
      ++bad;                                  \
    }                                         \
  } while (0)

// ---- 광선 추적 합성 장면: 축 맞춤 상자들 + 바닥(z = 0) ----
struct Box { double lo[3], hi[3]; int id; };
constexpr int IW = 160, IH = 120;
constexpr double FX = 120, CXP = 80, CYP = 60;

struct Cam {
  double T[12];
  // 위치, yaw(map z 둘레), pitch(+ 위). 광학: x 오른쪽, y 아래, z 앞
  Cam(double x, double y, double z, double yaw, double pitch) {
    const double f[3] = {std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch)};
    const double r[3] = {std::sin(yaw), -std::cos(yaw), 0};
    const double d[3] = {f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]};
    const double p[3] = {x, y, z};
    for (int k = 0; k < 3; ++k) { T[4 * k] = r[k]; T[4 * k + 1] = d[k]; T[4 * k + 2] = f[k]; T[4 * k + 3] = p[k]; }
  }
  double cam6(int k) const { return k < 3 ? T[4 * k + 3] : T[4 * (k - 3) + 2]; }
};

// 화소마다 깊이(광축 z)와 맞은 상자 id(-1 바닥·없음)
static void render(const Cam& c, const std::vector<Box>& bs, std::vector<float>* depth, std::vector<int>* hit) {
  depth->assign(size_t(IW) * IH, 0.f);
  hit->assign(size_t(IW) * IH, -1);
  const double* T = c.T;
  for (int v = 0; v < IH; ++v)
    for (int u = 0; u < IW; ++u) {
      const double dc[3] = {(u - CXP) / FX, (v - CYP) / FX, 1.0};
      const double dm[3] = {T[0] * dc[0] + T[1] * dc[1] + T[2], T[4] * dc[0] + T[5] * dc[1] + T[6], T[8] * dc[0] + T[9] * dc[1] + T[10]};
      const double o[3] = {T[3], T[7], T[11]};
      double best = 1e9;
      int bid = -1;
      if (dm[2] < -1e-9) best = -o[2] / dm[2];   // 바닥
      for (const Box& b : bs) {
        double t0 = 0, t1 = 1e9;
        bool ok = true;
        for (int k = 0; k < 3 && ok; ++k) {
          if (std::fabs(dm[k]) < 1e-12) { ok = o[k] >= b.lo[k] && o[k] <= b.hi[k]; continue; }
          double a = (b.lo[k] - o[k]) / dm[k], e = (b.hi[k] - o[k]) / dm[k];
          if (a > e) std::swap(a, e);
          t0 = std::max(t0, a);
          t1 = std::min(t1, e);
          ok = t0 <= t1;
        }
        if (ok && t0 > 1e-6 && t0 < best) { best = t0; bid = b.id; }
      }
      if (best < 6.0) { (*depth)[size_t(v) * IW + u] = float(best); (*hit)[size_t(v) * IW + u] = bid; }
    }
}

static InspectFrame frameOf(const Cam& c, const std::vector<float>& depth) {
  InspectFrame f;
  f.w = IW; f.h = IH; f.depth_m = depth.data();
  f.fx = float(FX); f.fy = float(FX); f.cx = float(CXP); f.cy = float(CYP);
  f.T_mc = c.T;
  return f;
}

static const Box kTable{{0, 0, 0}, {1.0, 0.8, 0.75}, 1};

static double ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
  InspectParams p;
  p.on = true;
  // 1. 시점·거리
  {
    InspectState s;
    const double a[6] = {0, 0, 1, 1, 0, 0};
    inspObserve(s, a, 2.0, p);
    inspObserve(s, a, 2.5, p);
    const double b[6] = {0.2, 0, 1, 1, 0, 0};   // 0.2 m: 같은 시점
    inspObserve(s, b, 1.9, p);
    CHECK(s.nViews() == 1 && std::fabs(s.closest - 1.9f) < 1e-6, "views %d closest %.2f", s.nViews(), s.closest);
    const double c[6] = {0.4, 0, 1, 1, 0, 0};   // 0.4 m: 새 시점
    inspObserve(s, c, 3.0, p);
    const double d[6] = {0, 0, 1, std::cos(0.35), std::sin(0.35), 0};   // 20°: 새 시점
    inspObserve(s, d, 3.0, p);
    const double e[6] = {0, 0, 1, std::cos(0.17), std::sin(0.17), 0};   // 9.7°: a 와 같음
    inspObserve(s, e, 3.0, p);
    CHECK(s.nViews() == 3 && std::fabs(s.closest - 1.9f) < 1e-6, "views %d closest %.2f", s.nViews(), s.closest);
    for (int k = 0; k < 100; ++k) {
      const double g[6] = {1.0 * k, 5, 1, 1, 0, 0};
      inspObserve(s, g, 4.0, p);
    }
    CHECK(s.nViews() == p.view_cap, "cap %d", s.nViews());
  }
  std::vector<float> dep;
  std::vector<int> hit;
  // 2. 위에서 본 탁자(카메라 높이 1.5 m, 45° 아래): 거의 다 봄
  {
    InspectState s;
    const Cam c(-0.5, 0.4, 1.5, 0, -0.75);
    render(c, {kTable}, &dep, &hit);
    inspTop(s, kTable.lo, kTable.hi, frameOf(c, dep), p);
    const float ts = inspTopSeen(s, kTable.lo, kTable.hi, p);
    std::printf("top from above: %.3f\n", ts);
    CHECK(ts >= 0.75f, "above %.3f", ts);
  }
  // 3. 낮은 카메라(LIMO 0.18 m, 수평): 1.7 m 에서는 앞 줄만(뒤 줄은 앞 모서리에 가리고 2 m 밖), 가까이서는 아무것도 안 보임
  {
    InspectState s;
    const Cam far(-1.7, 0.4, 0.18, 0, 0);
    const Box back{{4.0, -3, 0}, {4.1, 3, 2.5}, 4};   // 뒤 벽(탁자 위로 지나간 광선이 닿음)
    InspectState s0;
    render(far, {kTable}, &dep, &hit);   // 뒤 벽 없음: 탁자 위 광선은 깊이 0(모름) → 안 셈
    inspTop(s0, kTable.lo, kTable.hi, frameOf(far, dep), p);
    CHECK(inspTopSeen(s0, kTable.lo, kTable.hi, p) == 0.f, "no return counted %.3f", inspTopSeen(s0, kTable.lo, kTable.hi, p));
    render(far, {kTable, back}, &dep, &hit);
    inspTop(s, kTable.lo, kTable.hi, frameOf(far, dep), p);
    const float t1 = inspTopSeen(s, kTable.lo, kTable.hi, p);
    InspectState s2;
    const Cam near(-0.5, 0.4, 0.18, 0, 0);
    render(near, {kTable, back}, &dep, &hit);
    inspTop(s2, kTable.lo, kTable.hi, frameOf(near, dep), p);
    const float t2 = inspTopSeen(s2, kTable.lo, kTable.hi, p);
    std::printf("top low camera: far %.3f near %.3f\n", t1, t2);
    CHECK(t1 > 0.f && t1 <= 0.25f + 1e-6f, "low far %.3f", t1);
    CHECK(t2 == 0.f, "low near %.3f", t2);
    // 탁자 위 작은 컵(점 자리에 있음)은 가리지 않음: 앞 줄 하나 칸 위 컵
    InspectState s3;
    const Box cup{{0.08, 0.26, 0.75}, {0.16, 0.34, 0.85}, 2};   // 칸 (0, 1) 점(0.125, 0.3)을 덮음
    render(far, {kTable, cup, back}, &dep, &hit);
    inspTop(s3, kTable.lo, kTable.hi, frameOf(far, dep), p);
    const float t3 = inspTopSeen(s3, kTable.lo, kTable.hi, p);
    CHECK(std::fabs(t3 - t1) < 1e-6f, "cup on top %.3f vs %.3f", t3, t1);
  }
  // 4. 가리는 벽, 2 m 밖
  {
    InspectState s;
    const Cam c(-0.5, 0.4, 1.5, 0, -0.75);
    const Box wall{{-0.3, -1, 0}, {-0.25, 2, 2.5}, 3};
    render(c, {kTable, wall}, &dep, &hit);
    inspTop(s, kTable.lo, kTable.hi, frameOf(c, dep), p);
    CHECK(inspTopSeen(s, kTable.lo, kTable.hi, p) == 0.f, "occluded %.3f", inspTopSeen(s, kTable.lo, kTable.hi, p));
    InspectState s2;
    const Cam f(-2.5, 0.4, 1.5, 0, -0.4);
    render(f, {kTable}, &dep, &hit);
    inspTop(s2, kTable.lo, kTable.hi, frameOf(f, dep), p);
    CHECK(inspTopSeen(s2, kTable.lo, kTable.hi, p) == 0.f, "beyond 2 m %.3f", inspTopSeen(s2, kTable.lo, kTable.hi, p));
    // 윗면 없음: 작은 것, 너무 높은 것
    const double clo[3] = {0, 0, 0.75}, chi[3] = {0.08, 0.08, 0.85};
    CHECK(inspTopSeen(s, clo, chi, p) == -1.f, "cup has no top");
    const double wlo[3] = {0, 0, 0}, whi[3] = {1, 0.5, 2.0};
    CHECK(inspTopSeen(s, wlo, whi, p) == -1.f, "wardrobe top above 1.5 m");
  }
  // 5. 상자 자람·합치기
  {
    InspectState s;
    inspAlign(s, kTable.lo, kTable.hi);
    s.top_bits = 0x000F;   // 앞(y 낮은) 줄
    const double hi2[3] = {1.25, 0.8, 0.75};
    // x 로 1.25 배: 새 칸 가운데 x 0.16·0.47·0.78 은 옛 상자 안(앞 줄), 1.09 는 밖 → 3 칸
    CHECK(std::fabs(inspTopSeen(s, kTable.lo, hi2, p) - 3.f / 16) < 1e-6f, "grown x %.3f", inspTopSeen(s, kTable.lo, hi2, p));
    InspectState s1;
    inspAlign(s1, kTable.lo, kTable.hi);
    s1.top_bits = 0x00F0;   // 둘째 줄(y 0.2..0.4)
    const double hi3[3] = {1.0, 1.6, 0.75};   // y 로 두 배: 새 첫 줄 가운데 y 0.2 가 옛 둘째 줄
    CHECK(std::fabs(inspTopSeen(s1, kTable.lo, hi3, p) - 0.25f) < 1e-6f, "grown y %.3f", inspTopSeen(s1, kTable.lo, hi3, p));
    InspectState t;
    inspAlign(t, kTable.lo, kTable.hi);
    t.top_bits = 0xF000;   // 뒤 줄
    t.closest = 0.7f;
    const double v[6] = {9, 9, 1, 1, 0, 0};
    inspObserve(t, v, 2.0, p);
    s.closest = 1.2f;
    inspMerge(s, t, kTable.lo, kTable.hi, p);
    CHECK(std::fabs(inspTopSeen(s, kTable.lo, kTable.hi, p) - 0.5f) < 1e-6f && std::fabs(s.closest - 0.7f) < 1e-6f && s.nViews() == 1,
          "merge %.3f %.2f %d", inspTopSeen(s, kTable.lo, kTable.hi, p), s.closest, s.nViews());
  }
  // 6. ObjectMap 통합: 탁자에 다가가는 1.5 m 높이 카메라 — 켜도 물체·사건은 꺼짐과 같음
  {
    ObjParams on, off;
    on.insp.on = true;
    ObjectMap A(off), B(on);
    // 물체 지도는 objprob 하나라 검출마다 임베딩이 있어야 함: 라벨 1 개(table), 글 벡터 = 임베딩 = e0
    const float emb[2] = {1.f, 0.f};
    for (ObjectMap* m : {&A, &B}) {
      ApText t;
      t.dim = 2; t.text = {1.f, 0.f}; t.row_label = {0}; t.scale = 117.3f; t.bias = -12.7f; t.n_labels = 1;
      m->setTextModel(std::move(t));
    }
    const std::vector<Box> scene{kTable};
    std::vector<double> dists;
    double t_on = 0, t_off = 0;
    int nk = 0;
    const double xs[] = {-2.2, -1.8, -1.4, -1.0, -0.6, -0.2, -0.2};   // 0.3 m 넘게 떨어진 시점 6 개, 마지막은 같은 자리. 깊이 한계 3 m 안
    for (double x : xs) {
      const Cam c(x, 0.4, 1.5, 0, -std::atan2(1.5 - 0.4, 0.5 - x));
      render(c, scene, &dep, &hit);
      std::vector<uint32_t> bits((size_t(IW) * IH + 31) / 32, 0);
      int u0 = IW, v0 = IH, u1 = 0, v1 = 0;
      for (int v = 0; v < IH; ++v)
        for (int u = 0; u < IW; ++u)
          if (hit[size_t(v) * IW + u] == 1) {
            const size_t k = size_t(v) * IW + u;
            bits[k >> 5] |= 1u << (k & 31);
            u0 = std::min(u0, u); v0 = std::min(v0, v); u1 = std::max(u1, u + 1); v1 = std::max(v1, v + 1);
          }
      const int32_t cls = 0;
      const float score = 0.9f, box[4] = {float(u0), float(v0), float(u1), float(v1)};
      sm_detections d{};
      d.img_w = IW; d.img_h = IH; d.n = 1; d.cls = &cls; d.score = &score; d.box = box;
      d.mask_w = IW; d.mask_h = IH; d.mask_sx = 1; d.mask_sy = 1; d.mask_bits = bits.data();
      ObjFrame F;
      F.stamp = 0.5 * nk++;
      F.w = IW; F.h = IH; F.depth_m = dep.data();
      F.fx = float(FX); F.fy = float(FX); F.cx = float(CXP); F.cy = float(CYP);
      std::memcpy(F.T_mc, c.T, sizeof c.T);
      F.dets = &d;
      F.emb = emb; F.emb_dim = 2;
      F.eef[0][0] = F.eef[1][0] = -50;   // 손은 멀리
      F.base_xy[0] = x; F.base_xy[1] = 0.4;
      double t0 = ms();
      A.update(F);
      for (const ObsPoints& q : A.lastPoints()) A.addPoints(q.obj_id, q.xyz.data(), nullptr, int(q.xyz.size() / 3), F.stamp);   // 구름(접촉 판정에 씀)
      t_off += ms() - t0;
      t0 = ms();
      B.update(F);
      for (const ObsPoints& q : B.lastPoints()) B.addPoints(q.obj_id, q.xyz.data(), nullptr, int(q.xyz.size() / 3), F.stamp);
      t_on += ms() - t0;
      // 관측 중심 거리(참값 근사): 카메라 ↔ 이 관측의 마스크 점 중앙값 — 아래에서 단조 감소만 봄
      if (!B.objects().empty()) dists.push_back(B.objects()[0].insp.closest);
    }
    CHECK(A.objects().size() == 1 && B.objects().size() == 1, "objects %zu %zu", A.objects().size(), B.objects().size());
    if (A.objects().size() == 1 && B.objects().size() == 1) {
      const MapObject &a = A.objects()[0], &b = B.objects()[0];
      bool same = a.n_obs == b.n_obs && a.confirmed == b.confirmed && A.events().size() == B.events().size();
      for (int k = 0; k < 3; ++k) same = same && a.pos[k] == b.pos[k] && a.lo[k] == b.lo[k] && a.hi[k] == b.hi[k];
      CHECK(same, "on/off objects differ");
      CHECK(a.insp.closest < 0 && a.insp.nViews() == 0 && !a.insp.top_set, "off fills nothing");
      const float ts = inspTopSeen(b.insp, b.lo, b.hi, on.insp);
      std::printf("approach: closest %.2f views %d top %.3f box %.2f..%.2f %.2f..%.2f top z %.2f; objmap ms/kf off %.3f on %.3f\n", b.insp.closest,
                  b.insp.nViews(), ts, b.lo[0], b.hi[0], b.lo[1], b.hi[1], b.hi[2], t_off / nk, t_on / nk);
      CHECK(b.insp.nViews() == 6, "views %d", b.insp.nViews());
      bool mono = true;
      for (size_t k = 1; k < dists.size(); ++k) mono = mono && dists[k] <= dists[k - 1] + 1e-6;
      CHECK(mono && b.insp.closest > 0.6 && b.insp.closest < 1.6, "closest %.2f mono %d", b.insp.closest, int(mono));
      CHECK(ts >= 0.5f, "top seen %.3f", ts);
    }
  }
  // 7. 시간: 윗면 있는 물체 100 개 × 깊이 한 장
  {
    std::vector<InspectState> ss(100);
    const Cam c(-0.5, 0.4, 1.5, 0, -0.75);
    render(c, {kTable}, &dep, &hit);
    const InspectFrame f = frameOf(c, dep);
    const double t0 = ms();
    for (int r = 0; r < 100; ++r)
      for (size_t i = 0; i < ss.size(); ++i) {
        ss[i].top_bits = 0;
        const double lo[3] = {0.01 * double(i % 7), 0, 0}, hi[3] = {1.0, 0.8, 0.75};
        inspTop(ss[i], lo, hi, f, p);
      }
    const double us = (ms() - t0) * 1000 / 100;
    std::printf("inspTop 100 objects: %.1f us/keyframe\n", us);
    CHECK(us < 2000, "slow %.1f us", us);
  }
  // 8. 매개변수 문자열(sm_set_obj_params·엔진별 매개변수 파일이 쓰는 길): 로지스틱·κ·문턱, 모르는 이름 셈
  {
    ObjParams q;
    const int nb = ObjectMap::applyParams(&q, "ap_w0=-1.5,ap_wm7=2.25,kap_k0=1000,kap_d0=8,ap_same_p=0.55,inspect=1,bogus=3");
    CHECK(nb == 1 && q.ap.w[0] == -1.5 && q.ap.wm[7] == 2.25 && q.kap.k0 == 1000 && q.kap.d0 == 8 && q.ap.same_p == 0.55 && q.insp.on,
          "applyParams bad %d", nb);
  }
  std::printf(bad ? "FAILED %d\n" : "ok\n", bad);
  return bad ? 1 : 0;
}
