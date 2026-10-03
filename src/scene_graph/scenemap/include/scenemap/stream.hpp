// scenemap 실시간 스트림 — 로봇 자세·지도 변화분·물체/그래프 요약을 소켓으로 뷰어(sgview)에 바로 보낸다. 파일을 거치지 않는다.
//
// 설계 (비동기 · µs급)
//   스텝 스레드(생산자)는 락·시스템 호출·할당 없이 메모리 링 버퍼에 프레임을 복사해 넣기만 한다(pushPose ≈ 수백 ns, pushMapRect ≈ 영역 크기에 비례한 수 µs).
//   별도 송신 스레드(소비자)가 링을 비우고 비차단 소켓으로 보낸다. 느린 뷰어·끊긴 연결은 스텝 스레드를 막지 못한다 —
//   링이 가득 차면 그 프레임만 버리고(dropped 센다), 연결이 끊겼다 다시 붙으면 송신 스레드가 들고 있는 그림자 지도·마지막 요약·자세로
//   전체 상태를 다시 보낸다(생산자가 다시 보낼 필요 없음).
//   링은 둘: hot(자세·지도, 스텝 스레드) · view(요약 JSON, 비동기 저장/요약 스레드) — 각각 단일 생산자.
//
// 선(wire) 형식(little-endian): [u32 payload_len][u8 type][payload]
//   1 POSE      f64 stamp, x, y, yaw
//   2 MAP_RECT  i32 w, h; f64 res, ox, oy; i32 x0, y0, x1, y1; i8 cells[(x1-x0+1)*(y1-y0+1)] (sm_grid 배치: 행 y 는 아래→위, −1 모름·0..100 %)
//   3 VIEW      UTF-8 JSON (view.json 과 같은 내용)
//   4 JOINTS    f64 stamp, i32 n, f32 q[n] — 로봇 관절·상태 벡터(순서는 로봇 URDF 가 정함). 뷰어는 URDF 를 올릴 때 이 값으로 로봇을 움직인다
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace scenemap {

enum StreamType : uint8_t { kStreamPose = 1, kStreamMapRect = 2, kStreamView = 3, kStreamJoints = 4 };

// 단일 생산자·단일 소비자 바이트 링(2의 거듭제곱 크기). 프레임 = [u32 len][u8 type][payload]
class SpscRing {
 public:
  explicit SpscRing(size_t cap_pow2 = 1u << 22);
  // 생산자. false = 자리 없음(버림)
  bool push(uint8_t type, const void* payload, size_t n);
  // 두 조각 payload(헤더 + 데이터)를 한 프레임으로 — 복사 한 번에
  bool push2(uint8_t type, const void* a, size_t na, const void* b, size_t nb);
  // 소비자. 한 프레임을 out 에 꺼냄(payload 만), false = 비었음
  bool pop(uint8_t* type, std::vector<uint8_t>* out);

 private:
  void write(uint64_t pos, const void* p, size_t n);
  void read(uint64_t pos, void* p, size_t n) const;
  std::vector<uint8_t> buf_;
  size_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};   // 생산자가 씀
  alignas(64) std::atomic<uint64_t> tail_{0};   // 소비자가 씀
};

struct StreamStats {
  uint64_t frames_in = 0, dropped = 0, frames_sent = 0, bytes_sent = 0, reconnects = 0;
  bool connected = false;
};

class Streamer {
 public:
  Streamer();
  ~Streamer();
  // "host:port"(IPv4)로 연결을 시도하는 송신 스레드를 시작. 연결은 비동기로(실패해도 계속 재시도). 이미 켜져 있으면 false
  bool start(const std::string& host_port);
  void stop();
  bool running() const { return running_.load(std::memory_order_relaxed); }

  // ---- 스텝 스레드(락·시스템 호출·할당 없음) ----
  bool pushPose(double stamp, double x, double y, double yaw);
  bool pushJoints(double stamp, const float* q, int n);   // 로봇 관절·상태 벡터(수십 개 f32 복사)
  // cells: (x1-x0+1)*(y1-y0+1) 칸, 행 우선(y0 행부터). 격자 전체 크기·원점은 매번 함께 보낸다(그림자 지도가 모양 바뀜을 앎)
  bool pushMapRect(int w, int h, double res, double ox, double oy, int x0, int y0, int x1, int y1, const int8_t* cells);
  // ---- 요약 스레드 ----
  bool pushView(const std::string& json);

  StreamStats stats() const;

 private:
  void run();
  SpscRing hot_, view_;
  std::thread th_;
  std::atomic<bool> running_{false}, stop_{false};
  std::string host_port_;
  mutable std::atomic<uint64_t> in_{0}, dropped_{0}, sent_{0}, bytes_{0}, reconn_{0};
  std::atomic<bool> connected_{false};
};

}  // namespace scenemap
