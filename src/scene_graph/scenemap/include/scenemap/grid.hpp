// scenemap — 2D 점유 격자(로그 오즈). 필요할 때 넓어진다.
//
// 칸마다: 로그 오즈 L, 본 적 있는지, 맞은 점 합(점-선 맞추기용 칸 평균). L 이 0 이하로 떨어지면 점 합을 비운다
// (움직인 물체의 흔적이 다시 점유될 때 옛 평균이 섞이지 않게).
//
// 배치(10-03): 광선이 지나며 만지는 '뜨거운' 칸 8 바이트(L int16 고정소수점 1/256, 보이는 값 i8, 표시 비트, 스캔 번호)와
// 맞은 점 합 '차가운' 칸 20 바이트를 따로 둔다 — 빈칸 광선 한 칸 = 캐시 줄 하나. 보이는 값(−1 모름, 0..100 %)은 L 이 바뀔 때
// 표(LUT)로 바로 고쳐 두어 export 가 복사 하나다. 바뀐 영역(dirty)은 보이는 값이 실제로 바뀐 칸만 감싼다.
#pragma once
#include <cstdint>
#include <vector>

#include "scenemap/geom.hpp"
#include "scenemap/scan.hpp"

namespace scenemap {

struct GridParams {
  float res = 0.05f;
  float l_hit = 0.85f, l_miss = -0.4f, l_min = -4.f, l_max = 4.f;
};

class OccGrid {
 public:
  static constexpr int kQ = 256;   // L 고정소수점 눈금(1/256)
  explicit OccGrid(const GridParams& p = {});

  // 스캔을 자세 pose 로 넣는다(맞은 칸 +, 광선 위 칸 −, 한 스캔에서 칸마다 한 번, 맞음 우선). 로그 오즈가 바뀐 칸 수를 돌려줌
  int insert(const Scan2& s, const Pose2& pose);

  // 칸 번호(전역 정수 좌표) ↔ 값
  bool inside(int ix, int iy) const { return ix >= x0_ && iy >= y0_ && ix < x0_ + w_ && iy < y0_ + h_; }
  float logOdds(int ix, int iy) const { return inside(ix, iy) ? float(hot_[idx(ix, iy)].L) / kQ : 0.f; }
  bool seen(int ix, int iy) const { return inside(ix, iy) && (hot_[idx(ix, iy)].flags & kSeen); }
  // 칸 평균 점(맞은 점이 없으면 false)
  bool mean(int ix, int iy, float* mx, float* my, int* n = nullptr) const;
  // 칸에 쌓인 수평 법선(맞추기 점에서) — 방향이 서로 맞지 않으면(|합| < 0.5·개수) false
  bool normal(int ix, int iy, float* nx, float* ny) const;
  // 점유 확률 [pmin, pmax], 본 적 없는 칸 = pmin(Cartographer 와 같이 '맞을 곳 아님')
  float prob(int ix, int iy) const;

  int cellOf(double v) const { return int(std::floor(v / p_.res)); }
  float res() const { return p_.res; }
  int x0() const { return x0_; }
  int y0() const { return y0_; }
  int width() const { return w_; }
  int height() const { return h_; }
  // 계획기용: −1 모름, 0..100 점유 확률(%)
  std::vector<int8_t> export8() const;
  void export8(int8_t* out) const;   // width·height 칸
  // 바뀐 영역: 지난 takeDirty 뒤 보이는 값(export8)이 바뀐 칸의 경계 상자(전역 칸 좌표, 끝 포함). 없으면 false. 부를 때마다 비움.
  bool takeDirty(int* ix0, int* iy0, int* ix1, int* iy1, int consumer = 0);   // consumer 0 = C ABI(sm_take_dirty), 1 = 장면 그래프, 2 = 벽 추출(WallExtractor)
  uint64_t version() const { return version_; }
  // 보이는 값이 바뀔 때마다 +1(격자 모양이 바뀌어도 +1) — 스냅숏 격자 사본을 다시 만들지 판단
  uint64_t cellsVersion() const { return cells_ver_; }

  float pmin = 0.1f, pmax = 0.9f;

 private:
  enum : uint8_t { kSeen = 1, kSum = 2 };
  struct Hot {
    int16_t L;
    int8_t c8;
    uint8_t flags;
    uint32_t stamp;
  };
  struct Cold {
    float sx, sy, nx, ny;
    uint16_t cnt, cntn;
  };
  size_t idx(int ix, int iy) const { return size_t(iy - y0_) * w_ + (ix - x0_); }
  void ensure(int ix0, int iy0, int ix1, int iy1);
  inline void hit(size_t i, float hx, float hy, float nx, float ny, int ix, int iy);
  inline void miss(size_t i, int ix, int iy);
  inline void mark(int ix, int iy, int8_t old_c8, int8_t new_c8);

  GridParams p_;
  float inv_res_;
  int16_t q_hit_, q_miss_, q_min_, q_max_;
  std::vector<int8_t> lut8v_;          // L + off → 보이는 값
  std::vector<float> lutp_;            // L + off → 확률(pmin..pmax 자르기 전)
  int lut_off_ = 0;
  int x0_ = 0, y0_ = 0, w_ = 0, h_ = 0;
  std::vector<Hot> hot_;
  std::vector<Cold> cold_;
  uint32_t scan_id_ = 0;
  int changed_ = 0;
  uint64_t version_ = 0;          // insert 마다 +1
  uint64_t cells_ver_ = 0;
  bool cells_changed_ = false;
  bool dirty_[3] = {false, false, false};
  int db_[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  // insert 작업 버퍼(재사용): 점의 칸·map 좌표
  std::vector<int32_t> wc_;
  std::vector<float> wf_;
};

}  // namespace scenemap
