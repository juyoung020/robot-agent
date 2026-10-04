// scenemap 의 sgview 스트림(sm_stream_start: POSE · MAP_RECT · VIEW · JOINTS 프레임)을 이 프로세스가 소켓으로 받아 시뮬 시각을 붙여 파일(stream.sgs)에 적는다.
// og2sg(OmniGibson 기록)와 record_replay(GPU 환경 판)가 같이 쓴다. 형식: "SGS1" 다음 [f64 sim_t][u32 len][u8 type][payload].
#pragma once
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "scenemap.h"

struct Capture {
  int lfd = -1, port = 0;
  std::atomic<double> now{0.0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<bool> quit{false};
  FILE* out = nullptr;
  std::thread th;
  uint64_t frames = 0, by_type[8] = {};
  bool start(const std::string& path) {
    out = std::fopen(path.c_str(), "wb");
    if (!out) return false;
    std::fwrite("SGS1", 1, 4, out);
    lfd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (bind(lfd, (sockaddr*)&a, sizeof a) || listen(lfd, 1)) return false;
    socklen_t al = sizeof a;
    getsockname(lfd, (sockaddr*)&a, &al);
    port = ntohs(a.sin_port);
    th = std::thread([this] { run(); });
    return true;
  }
  void run() {
    const int c = accept(lfd, nullptr, nullptr);
    if (c < 0) return;
    std::vector<uint8_t> buf;
    auto rd_all = [&](void* p, size_t n) {
      uint8_t* q = (uint8_t*)p;
      while (n) { const ssize_t k = read(c, q, n); if (k <= 0) return false; q += k; n -= size_t(k); bytes += uint64_t(k); }
      return true;
    };
    for (;;) {
      uint8_t hd[5];
      if (!rd_all(hd, 5)) break;
      uint32_t len;
      std::memcpy(&len, hd, 4);
      buf.resize(len);
      if (!rd_all(buf.data(), len)) break;
      const double t = now.load();
      std::fwrite(&t, 8, 1, out);
      std::fwrite(hd, 1, 5, out);
      std::fwrite(buf.data(), 1, len, out);
      ++frames;
      if (hd[4] < 8) ++by_type[hd[4]];
    }
    close(c);
  }
  void stop() {
    if (th.joinable()) th.join();
    if (lfd >= 0) close(lfd);
    if (out) std::fclose(out);
  }
};

// 송신 스레드가 다 보낼 때까지(보낸 바이트가 두 번 연달아 그대로이고 받은 바이트가 따라잡음) — 기록이므로 한 프레임도 버리지 않게 걸음을 맞춘다
inline void sg_drain(sm_ctx* c, Capture& cap) {
  uint64_t prev = ~0ull;
  int same = 0;
  for (int k = 0; k < 20000; ++k) {
    sm_stream_stats ss{};
    sm_stream_get_stats(c, &ss);
    if (ss.connected && ss.bytes_sent == prev && cap.bytes.load() >= ss.bytes_sent) { if (++same >= 2) return; }
    else same = 0;
    prev = ss.bytes_sent;
    std::this_thread::sleep_for(std::chrono::microseconds(150));
  }
}
