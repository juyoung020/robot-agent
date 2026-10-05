// realbag 도구 공통: 작은 JSON 조립, 폴더 만들기, JPEG 쓰기(libjpeg), sgview 스트림 받기(Capture — 학습 뷰어 og2sg 의 sg_capture.h 와 같은
// 파일 형식 "SGS1" [f64 t][u32 len][u8 type][payload]).
#pragma once
#include <arpa/inet.h>
#include <jpeglib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "scenemap.h"

namespace rb {

inline std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[48];
  std::snprintf(b, sizeof b, "%.7g", v);
  return b;
}
inline std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o + "\"";
}
struct Obj {   // {"k": v, ...} 를 차례대로
  std::string s = "{";
  Obj& raw(const char* k, const std::string& v) { if (s.size() > 1) s += ','; s += jstr(k) + ':' + v; return *this; }
  Obj& num(const char* k, double v) { return raw(k, jnum(v)); }
  Obj& str(const char* k, const std::string& v) { return raw(k, jstr(v)); }
  std::string done() const { return s + "}"; }
};
inline void mkdirs(const std::string& p) {
  std::string cur;
  for (size_t i = 0; i <= p.size(); ++i) {
    if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); }
    if (i < p.size()) cur += p[i];
  }
}
// RGB 8 비트 → JPEG 파일
inline bool writeJpeg(const std::string& path, const uint8_t* rgb, int w, int h, int quality = 80) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  jpeg_compress_struct ci;
  jpeg_error_mgr je;
  ci.err = jpeg_std_error(&je);
  jpeg_create_compress(&ci);
  jpeg_stdio_dest(&ci, f);
  ci.image_width = JDIMENSION(w);
  ci.image_height = JDIMENSION(h);
  ci.input_components = 3;
  ci.in_color_space = JCS_RGB;
  jpeg_set_defaults(&ci);
  jpeg_set_quality(&ci, quality, TRUE);
  jpeg_start_compress(&ci, TRUE);
  while (ci.next_scanline < ci.image_height) {
    JSAMPROW row = const_cast<uint8_t*>(rgb + size_t(ci.next_scanline) * w * 3);
    jpeg_write_scanlines(&ci, &row, 1);
  }
  jpeg_finish_compress(&ci);
  jpeg_destroy_compress(&ci);
  std::fclose(f);
  return true;
}

// scenemap 의 sgview 스트림(sm_stream_start)을 이 프로세스가 루프백 소켓으로 받아 기록 시각(now)을 붙여 파일에 적는다
struct Capture {
  int lfd = -1, port = 0;
  std::atomic<double> now{0.0};
  std::atomic<uint64_t> bytes{0};
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
// 송신 스레드가 다 보낼 때까지 기다림(기록이므로 한 프레임도 버리지 않게 걸음을 맞춘다)
inline void drain(sm_ctx* c, Capture& cap) {
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

}  // namespace rb
