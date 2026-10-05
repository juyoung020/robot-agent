// 확률 물체 모델(objprob)(objprob.hpp) 시험: vMF r 합·μ, 같은 시각 조각은 서로 덜 세지 않음·비슷한 시점은 덜 셈, 이름 사후(상위어로 올림·
// 엔트로피), 바깥 이름 관측이 영상 모습에 덮이지 않음, 받침이면 같은 것이 아님, 평면 맞춤(세운 얇은 평면·수평면), 접촉
#include <cmath>
#include <cstdio>
#include <vector>

#include "scenemap/objprob.hpp"

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

// 차원 4 의 작은 글 모델: 라벨 0 cup, 1 mug(부모 2), 2 container(상위어, 낱말 없음), 3 sofa
static ApText makeText() {
  ApText t;
  t.dim = 4;
  t.n_labels = 4;
  const float rows[3][4] = {{1, 0, 0, 0}, {0.8f, 0.6f, 0, 0}, {0, 0, 1, 0}};
  for (auto& r : rows) t.text.insert(t.text.end(), r, r + 4);
  t.row_label = {0, 1, 3};
  t.parent = {2, 2, -1, -1};
  t.scale = 10.f;
  t.bias = -5.f;
  return t;
}

int main() {
  ApParams p;
  const ApText T = makeText();
  // 1. vMF: r = Σκz, μ = r/‖r‖, 합치면 r1 + r2
  {
    ApState a, b;
    const float z1[4] = {1, 0, 0, 0}, z2[4] = {0, 1, 0, 0};
    const double cam1[6] = {0, 0, 0, 1, 0, 0}, cam2[6] = {5, 0, 0, 0, 1, 0};
    apAddView(a, &T, p, z1, 4, 300, 1.0, cam1, false);
    apAddView(b, &T, p, z2, 4, 100, 2.0, cam2, false);
    apMerge(a, b, p);
    std::vector<float> mu;
    double conf = 0;
    CHECK(apMu(a, &mu, &conf), "mu");
    CHECK(std::fabs(conf - std::hypot(300.0, 100.0)) < 1e-3, "conf %.3f", conf);
    CHECK(std::fabs(mu[0] - 300 / std::hypot(300.0, 100.0)) < 1e-5, "mu0 %.4f", mu[0]);
    CHECK(a.need_whole && a.k_whole == 0, "merge -> need whole");
  }
  // 2. 덜 세기: 같은 시각(한 영상의 다른 조각)은 그대로, 다른 시각의 비슷한 시점은 temper 배
  {
    ApState s;
    const float z[4] = {1, 0, 0, 0};
    const double cam[6] = {0, 0, 0, 1, 0, 0};
    const double k1 = apAddView(s, &T, p, z, 4, 100, 1.0, cam, false);
    const double k2 = apAddView(s, &T, p, z, 4, 100, 1.0, cam, false);
    const double k3 = apAddView(s, &T, p, z, 4, 100, 2.0, cam, false);
    CHECK(k1 == 100 && k2 == 100, "same stamp not tempered %.1f %.1f", k1, k2);
    CHECK(std::fabs(k3 - 100 * p.temper) < 1e-9, "near view tempered %.1f", k3);
  }
  // 3. 이름: cup 과 mug 사이 애매하면 상위어 container 로, 확실하면 cup
  {
    ApParams q = p;
    q.kappa_ref = 100;
    q.name_tau = 0.8;
    ApState s;
    const float zc[4] = {0.9f, 0.3f, 0, 0};   // cup 0.9·mug 0.9 → 둘이 비슷
    const double cam[6] = {0, 0, 0, 1, 0, 0};
    apAddView(s, &T, q, zc, 4, 100, 1.0, cam, false);
    apName(s, T, q, 0.1);
    CHECK(s.rolled && s.name_lab == 2, "rolled to container: lab %d p %.2f top %d %.2f", s.name_lab, s.name_p, s.top_lab, s.top_p);
    CHECK(s.name_H > 0.5, "entropy %.2f", s.name_H);
    // 바깥 관측(confirm_object): mug 라는 강한 증거 → mug, 영상 모습이 더 와도 남음
    apObserveName(s, T.n_labels, 1, 8.0);
    apName(s, T, q, 0.1);
    CHECK(s.name_lab == 1 && !s.rolled, "external mug: lab %d", s.name_lab);
    for (int i = 0; i < 5; ++i) {
      const double c2[6] = {double(i), 0, 0, 1, 0, 0};
      apAddView(s, &T, q, zc, 4, 100, 3.0 + i, c2, false);
    }
    apName(s, T, q, 0.1);
    CHECK(s.name_lab == 1, "external evidence kept: lab %d", s.name_lab);
  }
  // 4. 받침(작은 것이 큰 것 윗면 위)이면 로짓이 막힘
  {
    ApPair q;
    q.f[0] = 1.0;   // 다 닿음
    q.f[3] = 0.2;
    q.f[5] = 1.0;
    CHECK(apLogit(q, p) < -10, "support veto %.1f", apLogit(q, p));
    q.f[5] = 0;
    CHECK(apLogit(q, p) > 0, "touching same %.2f", apLogit(q, p));
  }
  // 5. 평면 맞춤(RANSAC): x = 1 의 세운 얇은 평면(2 m × 2 m), z = 2.4 의 수평면
  {
    std::vector<float> P;
    for (int i = 0; i < 40; ++i)
      for (int j = 0; j < 40; ++j) P.insert(P.end(), {1.0f + 0.001f * float((i * 7 + j) % 3), 0.05f * i, 0.05f * j});
    const ApPlane a = apPlaneFit(P.data(), int(P.size() / 3), 0.02, 1);
    CHECK(a.ok && a.thick < 0.005 && std::fabs(a.n[2]) < 0.1 && a.hspan > 1.5 && a.inl > 0.99, "vertical plane thick %.3f nz %.2f hspan %.2f inl %.2f",
          a.thick, a.n[2], a.hspan, a.inl);
    P.clear();
    for (int i = 0; i < 40; ++i)
      for (int j = 0; j < 40; ++j) P.insert(P.end(), {0.05f * i, 0.05f * j, 2.4f});
    const ApPlane b = apPlaneFit(P.data(), int(P.size() / 3), 0.02, 2);
    CHECK(b.ok && std::fabs(b.n[2]) > 0.99 && std::fabs(b.zmed - 2.4) < 1e-3, "horizontal nz %.2f z %.2f", b.n[2], b.zmed);
  }
  // 5b. 잡음 + 바깥 점: 천장(z = 2.5, σ 1 cm) 1200 점 + 벽 모서리 쪽 세운 띠 300 점 + 흩어진 300 점(합 1800, 평면 67 %)
  //     — 안쪽 점 법선·높이는 천장 그대로(PCA 는 기울고 중앙 높이가 내려감). 같은 시드는 같은 답
  {
    uint64_t r = 12345;
    auto u01 = [&]() { r = r * 6364136223846793005ull + 1442695040888963407ull; return double(r >> 11) * (1.0 / 9007199254740992.0); };
    auto gau = [&]() { return std::sqrt(-2 * std::log(std::max(1e-12, u01()))) * std::cos(6.283185307 * u01()); };
    std::vector<float> P;
    for (int i = 0; i < 1200; ++i) P.insert(P.end(), {float(3 * u01()), float(2 * u01()), float(2.5 + 0.01 * gau())});
    for (int i = 0; i < 300; ++i) P.insert(P.end(), {float(3 * u01()), float(2.0 + 0.01 * gau()), float(1.5 + u01())});
    for (int i = 0; i < 300; ++i) P.insert(P.end(), {float(3 * u01()), float(2 * u01()), float(3 * u01())});
    const int n = int(P.size() / 3);
    const ApPlane a = apPlaneFit(P.data(), n, 0.03, apSeed(7, 1));
    CHECK(a.ok && std::fabs(a.n[2]) > 0.99 && std::fabs(a.zmed - 2.5) < 0.01 && a.inl > 0.6 && a.inl < 0.8 && a.thick < 0.015,
          "noisy ceiling nz %.3f z %.3f inl %.2f thick %.3f", a.n[2], a.zmed, a.inl, a.thick);
    const ApPlane b = apPlaneFit(P.data(), n, 0.03, apSeed(7, 1));
    CHECK(b.inl == a.inl && b.zmed == a.zmed && b.n[2] == a.n[2], "deterministic %.4f %.4f", a.inl, b.inl);
    // 세운 벽(y = 0, σ 1.5 cm) 80 % + 앞 가구 20 %: 벽 법선
    P.clear();
    for (int i = 0; i < 800; ++i) P.insert(P.end(), {float(2 * u01()), float(0.015 * gau()), float(2.4 * u01())});
    for (int i = 0; i < 200; ++i) P.insert(P.end(), {float(0.5 + 0.5 * u01()), float(0.1 + 0.5 * u01()), float(0.8 * u01())});
    const ApPlane c = apPlaneFit(P.data(), int(P.size() / 3), 0.04, apSeed(8, 2));
    CHECK(c.ok && std::fabs(c.n[1]) > 0.99 && c.inl > 0.75 && c.hspan > 1.5 && c.zhi - c.zlo > 1.5, "wall ny %.3f inl %.2f hspan %.2f",
          c.n[1], c.inl, c.hspan);
  }
  // 5c. 휜 면(반지름 0.4 m 반원통 — 소파 등받이·화분)은 지배 평면이 아님: 안쪽 비율 < 0.7(ApParams::plane_inl 0.8 아래)
  {
    std::vector<float> P;
    for (int i = 0; i < 60; ++i)
      for (int j = 0; j < 20; ++j) {
        const double t = 3.14159265 * i / 59;
        P.insert(P.end(), {float(0.4 * std::cos(t)), float(0.4 * std::sin(t)), float(0.04 * j)});
      }
    const ApPlane a = apPlaneFit(P.data(), int(P.size() / 3), 0.02, apSeed(9, 3));
    CHECK(a.ok && a.inl < 0.7 && a.inl < ApParams().plane_inl, "curved inl %.2f", a.inl);
    // 구(반지름 0.3 m)도
    P.clear();
    for (int i = 0; i < 30; ++i)
      for (int j = 0; j < 30; ++j) {
        const double th = 3.14159265 * (i + 0.5) / 30, ph = 6.2831853 * j / 30;
        P.insert(P.end(), {float(0.3 * std::sin(th) * std::cos(ph)), float(0.3 * std::sin(th) * std::sin(ph)), float(1 + 0.3 * std::cos(th))});
      }
    const ApPlane b = apPlaneFit(P.data(), int(P.size() / 3), 0.02, apSeed(9, 4));
    CHECK(b.inl < 0.7 && b.inl < ApParams().plane_inl, "sphere inl %.2f", b.inl);
  }
  // 6. 접촉: 4 cm 칸 이웃 안
  {
    const float A[6] = {0, 0, 0, 0.1f, 0, 0};
    ApContactIdx keys;
    apBuildContact(A, 2, 0.04, &keys);
    const float B[6] = {0.03f, 0, 0, 0.5f, 0, 0};
    CHECK(std::fabs(apContact(B, 2, keys, 0.04) - 0.5) < 1e-9, "contact %.2f", apContact(B, 2, keys, 0.04));
  }
  std::printf("%s (%d failures)\n", bad ? "FAIL" : "ok", bad);
  return bad ? 1 : 0;
}
