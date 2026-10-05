// scenemap — 깊이 영상 → 가상 레이저 스캔(베이스 기준 2D).
//
// 높이 띠 [band_lo, band_hi] 안의 점 = 장애물. 방위 칸마다 가장 가까운 것 하나가 레이저 한 줄.
// 띠 아래(바닥) 점은 그 방위가 거기까지 비어 있다는 증거(빈칸 광선 끝). 로봇 자신(몸체·팔 끝·팔 선분)은 뺀다.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "scenemap/geom.hpp"

namespace scenemap {

struct DepthView {
  int w = 0, h = 0;
  const uint16_t* mm = nullptr;   // 깊이 mm (0 = 없음)
  const float* m = nullptr;       // 또는 깊이 m(평가기 원 텐서, 0·NaN = 없음). m 이 있으면 m 을 쓴다
  int step = 1;                   // 화소 간격(원 해상도 입력이면 4)
  float fx = 0, fy = 0, cx = 0, cy = 0;
  float T_bc[12] = {0};           // 베이스 ← 카메라 광학 프레임, 행 우선 3×4
};

struct Capsule {
  float a[3], b[3], r;
};

// 로봇 몸 가리기. caps 가 있으면(순기구학 팔 뼈대) 그것을, 없으면 어깨 대략값–팔 끝 선분을 쓴다.
struct BodyState {
  float eef[2][3] = {{0}};        // 팔 끝(베이스 기준). OMX-F 는 팔 하나라 두 칸에 같은 값
  std::vector<Capsule> caps;
};

struct BodyFk;
// 순기구학 뼈대 → 캡슐(팔 링크 arm_r, 그리퍼–손끝 hand_r). 몸은 스캔의 self_r 원으로 뺀다
BodyState bodyFromFk(const BodyFk& fk, const float eef[2][3], float arm_r = 0.09f, float hand_r = 0.10f);

struct ScanParams {
  float zmin = 0.3f, zmax = 8.0f;       // 광학 z 범위(8 m: 12비트 로그 양자화 간격 약 4 mm, 긴 판 넓은 방에서 필요)
  float band_lo = 0.10f, band_hi = 1.80f;
  int bins = 720;                       // 0.5°
  float self_r = 0.55f, eef_r = 0.35f, arm_r = 0.20f;
  float shoulder[2][3] = {{0.f, 0.22f, 1.25f}, {0.f, -0.22f, 1.25f}};
  // 맞추기 점: 수직면(|n_z| < vert_nz) 쪽 점을 [band_lo, match_hi] 에서 모아 2D 칸(match_cell)마다 하나로.
  // 가장 가까운 것만 쓰는 레이저 한 줄과 달리, 탁자 뒤 벽·띠 위 벽도 쓴다(머리 카메라가 탁자를 볼 때가 많다).
  float match_hi = 3.0f, vert_nz = 0.7f, match_cell = 0.025f;
  bool dense = true;   // false: 레이저 한 줄(hx, hy)로만 맞춘다
};

struct Scan2 {
  float ox = 0, oy = 0;                 // 광선 시작(카메라의 베이스 기준 수평 위치)
  std::vector<float> hx, hy;            // 장애물 점(베이스 기준)
  std::vector<float> fx, fy;            // 빈칸만 있는 광선의 끝(장애물 없음)
  std::vector<float> mx, my;            // 맞추기 점(지도에도 맞음으로 넣음, 광선은 안 쏨)
  std::vector<float> mnx, mny;          // 맞추기 점의 수평 법선(깊이 영상 이웃으로, 카메라 쪽을 향하게)
  // 방위 칸 서명(넣기 정책의 '스캔이 바뀌었나'): 칸마다 장애물 거리(칸 단위, 양수) 또는 −빈 광선 길이, 0 = 없음
  std::vector<int16_t> sig;
};

// makeScan 작업 버퍼(keyframe 마다 다시 쓰고 할당하지 않음)
struct ScanWork {
  std::vector<float> P, Zo, hit_r, hx, hy, floor_r;
  // 맞추기 칸(2.5 cm) 열린 주소 해시: 열쇠·세대·모음 번호. 모음은 처음 본 순서(래스터)로 쌓는다
  std::vector<int64_t> hkey;
  std::vector<uint32_t> hgen, hidx;
  uint32_t gen = 0;
  struct Acc { float x, y, nx, ny; int n; };
  std::vector<Acc> acc;
  // 화소마다 방위 칸·수평 거리 배율²(카메라 회전·내부 파라미터에만 달림 — 머리가 그대로면 keyframe 사이에 다시 씀)
  std::vector<int16_t> pbin;
  std::vector<float> phs2;
  float pkey[16] = {0};
  bool pvalid = false;
  int n_cache_hit = 0, n_cache_miss = 0;
};

// 로봇에 붙어 같이 움직이는 것(들고 있는 물체·팔 등) 걸러내기.
// 베이스 기준 3D 칸(5 cm)이 로봇이 충분히 움직인 뒤에도 같은 자리(±1 칸)에 계속 있으면 '붙은 것'이다.
// 세상에 고정된 것은 로봇이 움직이면 베이스 기준 자리가 바뀐다. 멀리 있는 벽이 우연히 겹치는 것(벽을 따라 직진)을
// 피하려고 로봇 가까이(radius) 점만 본다.
struct AttachParams {
  bool enabled = true;
  float vox = 0.05f, radius = 1.3f;
  double move_xy = 0.10, move_yaw = 6.0 * M_PI / 180.0;   // 이만큼 움직일 때마다 한 번 비교
  int min_score = 2;                                      // 연속 몇 번 같은 자리면 붙은 것
};

class AttachFilter {
 public:
  explicit AttachFilter(const AttachParams& p = {}) : p_(p) {}
  bool attached(float x, float y, float z) const {
    if (!p_.enabled || score_.empty()) return false;
    auto it = score_.find(key(x, y, z));
    return it != score_.end() && it->second >= p_.min_score;
  }
  bool near(float x, float y) const { return x * x + y * y < p_.radius * p_.radius; }
  int64_t key(float x, float y, float z) const {
    return key3(int(std::floor(x / p_.vox)), int(std::floor(y / p_.vox)), int(std::floor(z / p_.vox)));
  }
  static int64_t key3(int i, int j, int k) {
    return (int64_t(i & 0xfffff) << 40) | (int64_t(j & 0xfffff) << 20) | int64_t(k & 0xfffff);
  }
  // 이번 keyframe 의 베이스 기준 칸들(가까운 점만)과 적분 자세
  void update(const std::vector<int64_t>& cur, const Pose2& odom_pose);
  size_t nAttached() const;

 private:
  AttachParams p_;
  std::unordered_map<int64_t, int> score_;   // 지난 비교 때 칸 → 연속 횟수
  bool have_ = false;
  Pose2 check_;
};

// att != nullptr 이면 붙은 것으로 판정된 점은 뺀다. vox_out != nullptr 이면 가까운 점의 베이스 칸 목록(중복 없음)을 낸다.
void makeScan(const DepthView& d, const BodyState& b, const ScanParams& p, Scan2* out,
              const AttachFilter* att = nullptr, std::vector<int64_t>* vox_out = nullptr);
// 같은 계산, 작업 버퍼 재사용. sig_cell > 0 이면 out->sig(방위 칸 서명, 칸 크기 sig_cell m)도 채운다
void makeScan(const DepthView& d, const BodyState& b, const ScanParams& p, Scan2* out, const AttachFilter* att,
              std::vector<int64_t>* vox_out, ScanWork* work, float sig_cell = 0.f);

}  // namespace scenemap
