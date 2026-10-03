// scenemap — 평면 자세와 작은 선형대수(외부 의존 없음).
#pragma once
#include <cmath>

namespace scenemap {

struct Pose2 {
  double x = 0, y = 0, th = 0;
};

inline double wrapAngle(double a) {
  a = std::fmod(a + M_PI, 2 * M_PI);
  if (a < 0) a += 2 * M_PI;
  return a - M_PI;
}

// a ⊕ d : a 기준으로 표현된 이동 d 를 합성
inline Pose2 compose(const Pose2& a, const Pose2& d) {
  const double c = std::cos(a.th), s = std::sin(a.th);
  return {a.x + c * d.x - s * d.y, a.y + s * d.x + c * d.y, a.th + d.th};
}

// 3×3 대칭 선형계 H x = b (촐레스키). 실패하면 false.
inline bool solve3(const double H[9], const double b[3], double x[3]) {
  double L[9] = {0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = H[i * 3 + j];
      for (int k = 0; k < j; ++k) s -= L[i * 3 + k] * L[j * 3 + k];
      if (i == j) {
        if (s <= 1e-12) return false;
        L[i * 3 + i] = std::sqrt(s);
      } else {
        L[i * 3 + j] = s / L[j * 3 + j];
      }
    }
  double y[3];
  for (int i = 0; i < 3; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) s -= L[i * 3 + k] * y[k];
    y[i] = s / L[i * 3 + i];
  }
  for (int i = 2; i >= 0; --i) {
    double s = y[i];
    for (int k = i + 1; k < 3; ++k) s -= L[k * 3 + i] * x[k];
    x[i] = s / L[i * 3 + i];
  }
  return true;
}

}  // namespace scenemap
