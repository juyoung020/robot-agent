// 실시간 스트림 시험: 루프백 TCP 로 프레임을 받아 내용 확인, 재연결 시 전체 상태 재전송, 스텝 스레드 비용(µs) 측정.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "scenemap/stream.hpp"
using namespace scenemap;
using Clk = std::chrono::steady_clock;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Frame { uint8_t type; std::vector<uint8_t> pl; };

// n 바이트를 timeout 안에 정확히 읽는다
static bool readN(int fd, void* p, size_t n, int ms) {
  auto t0 = Clk::now();
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd, static_cast<uint8_t*>(p) + got, n - got, MSG_DONTWAIT);
    if (r > 0) got += size_t(r);
    else if (r == 0) return false;
    else std::this_thread::sleep_for(std::chrono::microseconds(100));
    if (std::chrono::duration<double, std::milli>(Clk::now() - t0).count() > ms) return false;
  }
  return true;
}
static bool readFrame(int fd, Frame* f, int ms = 1000) {
  uint32_t len;
  if (!readN(fd, &len, 4, ms) || !readN(fd, &f->type, 1, ms)) return false;
  f->pl.resize(len);
  return len == 0 || readN(fd, f->pl.data(), len, ms);
}

int main() {
  std::printf("test_stream\n");
  // 루프백 수신기
  const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
  ::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof(a));
  ::listen(ls, 4);
  socklen_t al = sizeof(a);
  ::getsockname(ls, reinterpret_cast<sockaddr*>(&a), &al);
  const std::string hp = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));

  Streamer s;
  CHECK(s.start(hp), "start");
  CHECK(!s.start(hp), "second start must fail");
  int c = ::accept(ls, nullptr, nullptr);
  CHECK(c >= 0, "accept");

  // 지도(4x3 격자, 전체) + 일부 영역 갱신 + 자세 + 요약
  std::vector<int8_t> full(12, -1);
  CHECK(s.pushMapRect(4, 3, 0.05, -1.0, 2.0, 0, 0, 3, 2, full.data()), "push full map");
  const int8_t patch[2] = {100, 50};
  CHECK(s.pushMapRect(4, 3, 0.05, -1.0, 2.0, 1, 1, 2, 1, patch), "push patch");
  CHECK(s.pushPose(1.5, 0.25, -0.5, 1.0), "push pose");
  CHECK(s.pushView("{\"stamp\":1.5}"), "push view");
  const float jq[3] = {0.5f, -1.25f, 2.0f};
  CHECK(s.pushJoints(1.5, jq, 3), "push joints");
  Frame f;
  bool got_map = false, got_patch = false, got_pose = false, got_view = false, got_joints = false;
  for (int i = 0; i < 5 && readFrame(c, &f); ++i) {
    if (f.type == kStreamMapRect && f.pl.size() == 48 + 12) got_map = true;
    else if (f.type == kStreamMapRect && f.pl.size() == 48 + 2) { got_patch = f.pl[48] == 100 && f.pl[49] == 50; }
    else if (f.type == kStreamPose && f.pl.size() == 32) { double v[4]; std::memcpy(v, f.pl.data(), 32); got_pose = v[1] == 0.25 && v[2] == -0.5; }
    else if (f.type == kStreamView) got_view = std::string(f.pl.begin(), f.pl.end()) == "{\"stamp\":1.5}";
    else if (f.type == kStreamJoints && f.pl.size() == 12 + 12) { float q[3]; std::memcpy(q, f.pl.data() + 12, 12); got_joints = q[0] == 0.5f && q[1] == -1.25f && q[2] == 2.0f; }
  }
  CHECK(got_map && got_patch && got_pose && got_view && got_joints, "frames: map %d patch %d pose %d view %d joints %d", got_map, got_patch, got_pose, got_view, got_joints);

  // 재연결: 끊으면 송신 스레드가 그림자(지도 + 패치 적용된 값·요약·자세)로 전체 상태를 다시 보낸다
  ::close(c);
  s.pushPose(2.0, 1, 1, 0);   // 끊긴 것을 알아채게 한 번 보냄
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  s.pushPose(2.1, 1, 1, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const int c2 = ::accept(ls, nullptr, nullptr);
  CHECK(c2 >= 0, "re-accept after drop");
  bool full_ok = false, view2 = false, pose2 = false, joints2 = false;
  for (int i = 0; i < 5 && readFrame(c2, &f, 2000); ++i) {
    if (f.type == kStreamMapRect && f.pl.size() == 48 + 12) {
      full_ok = f.pl[48 + 1 * 4 + 1] == 100 && f.pl[48 + 1 * 4 + 2] == 50 && int8_t(f.pl[48]) == -1;   // 패치가 그림자에 반영돼 있음
    } else if (f.type == kStreamView) view2 = true;
    else if (f.type == kStreamPose) pose2 = true;
    else if (f.type == kStreamJoints) joints2 = true;
  }
  CHECK(full_ok && view2 && pose2 && joints2, "resync after reconnect: full %d view %d pose %d joints %d", full_ok, view2, pose2, joints2);
  CHECK(s.stats().reconnects >= 2, "reconnects %llu", (unsigned long long)s.stats().reconnects);

  // 스텝 스레드 비용: (1) 수신기가 안 읽는 느린 뷰어여도 막히지 않는다 (2) 수신기가 읽는 정상 상태에서 호출당 µs 급
  {
    const int N = 200000;
    auto t0 = Clk::now();
    for (int i = 0; i < N; ++i) s.pushPose(i * 0.033, i * 0.01, 0, 0.1);
    const double pose_ns = std::chrono::duration<double, std::nano>(Clk::now() - t0).count() / N;
    std::printf("  pushPose (slow viewer, ring may be full) %.0f ns/call, dropped %llu\n", pose_ns, (unsigned long long)s.stats().dropped);
    CHECK(pose_ns < 1000.0, "pushPose too slow: %.0f ns", pose_ns);
  }
  {
    // 수신기: 계속 읽어 비운다
    std::atomic<bool> stop{false};
    std::thread drain([&] { std::vector<uint8_t> b(1 << 20); while (!stop) { if (::recv(c2, b.data(), b.size(), MSG_DONTWAIT) <= 0) std::this_thread::sleep_for(std::chrono::microseconds(50)); } });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // 앞에서 쌓인 것을 비움
    std::vector<int8_t> rect(100 * 100, 50);
    double tot_ns = 0, worst_ns = 0;
    int ok = 0;
    for (int i = 0; i < 2000; ++i) {   // 30 Hz 지도 갱신보다 훨씬 빠른 1 ms 간격 — 송신 스레드가 따라가는지도 같이 본다
      auto t0 = Clk::now();
      const bool r = s.pushMapRect(500, 400, 0.05, 0, 0, 10, 10, 109, 109, rect.data());
      const double ns = std::chrono::duration<double, std::nano>(Clk::now() - t0).count();
      if (r) { tot_ns += ns; worst_ns = std::max(worst_ns, ns); ++ok; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop = true;
    drain.join();
    std::printf("  pushMapRect(100x100 = 10 KB, viewer reading) mean %.2f us, worst %.2f us, accepted %d/2000\n", tot_ns / ok / 1000.0, worst_ns / 1000.0, ok);
    CHECK(ok > 1900, "too many dropped with a reading viewer: %d/2000", ok);
    CHECK(tot_ns / ok / 1000.0 < 20.0, "pushMapRect mean too slow: %.2f us", tot_ns / ok / 1000.0);
  }
  ::close(c2);
  s.stop();
  ::close(ls);
  if (g_fail) { std::printf("test_stream: %d failed\n", g_fail); return 1; }
  std::printf("  ok\n");
  return 0;
}
