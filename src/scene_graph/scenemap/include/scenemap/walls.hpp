// scenemap — 벽을 2D 로: 점유 격자 → 벽 선분 → 로봇 좌표 수치(벽 상태). viewer/walls2d.py 의 C++ 포팅(같은 값).
//
// 격자 배치는 sm_grid 와 같다: cells[y·w + x] 가 칸 (x, y), 칸 왼쪽 아래 모서리 = origin + (x, y)·res. −1 모름, 0..100 점유 %.
// 점유 = cells ≥ kOccMin(65 %). (파이썬은 PGM 회색값 ≤ 90 = 같은 문턱, 행 순서만 위아래가 뒤집혀 있다 — 여기서 맞춰 준다.)
//
// 벽 상태 벡터 (float32, 길이 kStateLen = kSectors + kSegments·5), 로봇 좌표(x 앞, y 왼쪽):
//   [0 : kSectors]            섹터 i 방향 첫 점유 칸까지 거리 / kMaxRange. 섹터 0 = 정면, 반시계, 간격 2π/kSectors. 1.0 = 범위 안에 없음.
//   [kSectors + 5j : +5]      가까운 순 j번째 벽 선분: ax, ay, bx, by (m / kMaxRange, [−1, 1] 로 자름), valid(0/1). 안 쓴 칸은 0.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace scenemap {

constexpr int kOccMin = 65;       // 점유 % 문턱
constexpr double kMinLen = 0.5;   // 가장 짧은 벽 선분 [m]
constexpr double kMaxThick = 0.5; // 이보다 두꺼운 덩어리는 가구·잡동사니지 벽이 아님 [m]
constexpr int kSectors = 16;
constexpr int kSegments = 8;
constexpr double kMaxRange = 4.0; // [m]
constexpr int kStateLen = kSectors + kSegments * 5;

struct WallSeg { double ax, ay, bx, by; };   // 지도 좌표 [m]
// 벽으로 치지 않을 영역(바닥에 놓인 가구·물체의 바닥 면적, 지도 좌표 [m]). 이 안의 점유 칸은 벽 선분 추출에서 빈 칸으로 본다.
// 소파·탁자처럼 길고 얇은 덩어리가 벽으로 잡히는 것을 막는다. 광선 거리(rayDistances)는 장애물 거리라서 그대로 쓴다.
struct WallRect { double x0, y0, x1, y1; bool operator==(const WallRect& o) const { return x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1; } };

struct WallGrid {
  const int8_t* cells;
  int w, h;
  double res, ox, oy;
};

// 점유 칸에서 축에 맞는(가로·세로) 벽 중심선. 기울어진 벽은 못 찾는다(실내 직교 배치 전제).
std::vector<WallSeg> wallSegments(const WallGrid& g, double min_len = kMinLen, double max_thick = kMaxThick,
                                  double overlap = 0.6, const std::vector<WallRect>* ignore = nullptr);

// 지도 축과 벽 방향이 어긋난 지도(slam: 지도 좌표 = 출발 자세라 벽이 기울어짐)용. 점유 칸의 주된 직교 방향 θ(−45°..45°, rad)를
// 찾아(투영 히스토그램이 가장 뾰족한 각, 0° 보다 kAlignGain 배 이상 뾰족할 때만), |θ| ≤ kAlignTol 이면 wallSegments 그대로(축에 맞는 지도 = 같은 결과), 아니면 격자를 −θ 돌린
// 격자로 다시 뽑아(칸 = 원래 칸 4 개 중 하나라도 점유면 점유) 축에 맞는 벽을 찾고, 각 선분을 원래 격자의 점유 띠에 다시 맞춘다
// (나란히 ±3 칸 안에서 가장 점유가 많은 띠 가운데로 옮김, 점유가 4 칸 넘게 끊기면 나누고 양끝을 점유까지 자름, 점유 < 70 % 조각은 버림,
// 2 칸 안의 겹치는 나란한 선은 하나로) 뒤 θ 돌려 원래 지도 좌표로 돌려준다.
// ignore 영역은 원래 지도 좌표에서 지운 뒤 돌린다. angle_out 이 있으면 쓴 θ 를 넣는다.
constexpr double kAlignTol = 1.0 * 3.14159265358979323846 / 180.0;
constexpr double kAlignGain = 1.08;   // wallAngle: θ 의 투영 점수가 0° 점수의 이 배수보다 작으면 0 을 돌려준다(축에 맞는 지도)
double wallAngle(const WallGrid& g);
double wallAngle(const WallGrid& g, const std::vector<WallRect>* ignore);   // ignore 영역을 지운 격자로
std::vector<WallSeg> wallSegmentsAligned(const WallGrid& g, double min_len = kMinLen, double max_thick = kMaxThick,
                                         double overlap = 0.6, const std::vector<WallRect>* ignore = nullptr,
                                         double* angle_out = nullptr);
// wallSegmentsAligned 와 같되 방향 θ 를 호출자가 준다(wallAngle 을 다시 재지 않음 — capi 가 θ 를 가끔만 재서 들고 있음). |θ| ≤ kAlignTol 이면 wallSegments
std::vector<WallSeg> wallSegmentsAtAngle(const WallGrid& g, double th, double min_len = kMinLen, double max_thick = kMaxThick,
                                         double overlap = 0.6, const std::vector<WallRect>* ignore = nullptr);

// 실시간 갱신용: 비트 격자와 작업 버퍼를 들고 있다가, 격자에서 바뀐 행만 다시 비트로 만든다.
// update(g, y_lo, y_hi): 격자 행 y(아래→위, sm_grid 의 y) y_lo..y_hi(끝 포함)가 바뀌었다. y_lo > y_hi 면 전부(처음·격자 크기가 바뀜).
// 돌려주는 참조는 다음 update/reset 까지 유효. 결과는 wallSegments 와 같다.
class WallExtractor {
 public:
  WallExtractor();
  ~WallExtractor();
  WallExtractor(const WallExtractor&) = delete;
  WallExtractor& operator=(const WallExtractor&) = delete;
  // ignore: 벽으로 치지 않을 영역. 이전 호출과 다르면 전부 다시 만든다(몇 µs).
  const std::vector<WallSeg>& update(const WallGrid& g, int y_lo, int y_hi, const std::vector<WallRect>* ignore = nullptr,
                                     double min_len = kMinLen, double max_thick = kMaxThick, double overlap = 0.6);
  void reset();
  const std::vector<WallSeg>& segments() const { return segs_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::vector<WallSeg> segs_;
  std::vector<WallRect> last_ignore_;
};

// 로봇 자세 (x, y, yaw) 에서 n 개 광선의 첫 점유 칸까지 거리 [m], 없으면 max_range
void rayDistances(const WallGrid& g, const double pose[3], float* out, int n = kSectors, double max_range = kMaxRange);

// 가까운 k 개 선분을 로봇 좌표로. out: k·4 (ax, ay, bx, by) [m], dist: k, 안 쓴 칸은 0 / +inf. 돌려주는 값 = 채운 개수
int segmentsRobotFrame(const std::vector<WallSeg>& segs, const double pose[3], int k, float* out, float* dist);

// 길이 kStateLen 벡터를 out 에 쓴다
void wallStateVector(const WallGrid& g, const std::vector<WallSeg>& segs, const double pose[3], float* out);

}  // namespace scenemap
