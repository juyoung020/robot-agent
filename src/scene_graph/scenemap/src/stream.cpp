#include "scenemap/stream.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace scenemap {

// ---------------------------------------------------------------------------------------------------------------------
// SpscRing
// ---------------------------------------------------------------------------------------------------------------------
SpscRing::SpscRing(size_t cap) : buf_(cap), mask_(cap - 1) {}

void SpscRing::write(uint64_t pos, const void* p, size_t n) {
  const size_t idx = size_t(pos) & mask_, first = std::min(n, buf_.size() - idx);
  std::memcpy(&buf_[idx], p, first);
  if (first < n) std::memcpy(&buf_[0], static_cast<const uint8_t*>(p) + first, n - first);
}

void SpscRing::read(uint64_t pos, void* p, size_t n) const {
  const size_t idx = size_t(pos) & mask_, first = std::min(n, buf_.size() - idx);
  std::memcpy(p, &buf_[idx], first);
  if (first < n) std::memcpy(static_cast<uint8_t*>(p) + first, &buf_[0], n - first);
}

bool SpscRing::push2(uint8_t type, const void* a, size_t na, const void* b, size_t nb) {
  const size_t total = 5 + na + nb;
  const uint64_t head = head_.load(std::memory_order_relaxed), tail = tail_.load(std::memory_order_acquire);
  if (buf_.size() - size_t(head - tail) < total) return false;
  const uint32_t len = uint32_t(na + nb);
  write(head, &len, 4);
  write(head + 4, &type, 1);
  if (na) write(head + 5, a, na);
  if (nb) write(head + 5 + na, b, nb);
  head_.store(head + total, std::memory_order_release);
  return true;
}

bool SpscRing::push(uint8_t type, const void* payload, size_t n) { return push2(type, payload, n, nullptr, 0); }

bool SpscRing::pop(uint8_t* type, std::vector<uint8_t>* out) {
  const uint64_t tail = tail_.load(std::memory_order_relaxed), head = head_.load(std::memory_order_acquire);
  if (head == tail) return false;
  uint32_t len = 0;
  read(tail, &len, 4);
  read(tail + 4, type, 1);
  out->resize(len);
  if (len) read(tail + 5, out->data(), len);
  tail_.store(tail + 5 + len, std::memory_order_release);
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Streamer
// ---------------------------------------------------------------------------------------------------------------------
namespace {

using Clk = std::chrono::steady_clock;

void appendFrame(std::vector<uint8_t>& out, uint8_t type, const uint8_t* p, size_t n) {
  const uint32_t len = uint32_t(n);
  const size_t o = out.size();
  out.resize(o + 5 + n);
  std::memcpy(&out[o], &len, 4);
  out[o + 4] = type;
  if (n) std::memcpy(&out[o + 5], p, n);
}

// 송신 스레드가 들고 있는 그림자: 다시 연결될 때 전체 상태를 보내려고
struct Shadow {
  int w = 0, h = 0;
  double res = 0, ox = 0, oy = 0;
  std::vector<int8_t> cells;
  std::vector<uint8_t> view;
  std::vector<uint8_t> joints;   // 마지막 JOINTS 페이로드
  double pose[4] = {0, 0, 0, 0};
  bool have_pose = false;

  void applyRect(const std::vector<uint8_t>& pl) {
    constexpr size_t kHead = 48;
    if (pl.size() < kHead) return;
    int32_t w_, h_, x0, y0, x1, y1;
    double res_, ox_, oy_;
    std::memcpy(&w_, &pl[0], 4); std::memcpy(&h_, &pl[4], 4);
    std::memcpy(&res_, &pl[8], 8); std::memcpy(&ox_, &pl[16], 8); std::memcpy(&oy_, &pl[24], 8);
    std::memcpy(&x0, &pl[32], 4); std::memcpy(&y0, &pl[36], 4); std::memcpy(&x1, &pl[40], 4); std::memcpy(&y1, &pl[44], 4);
    if (w_ <= 0 || h_ <= 0 || x0 < 0 || y0 < 0 || x1 >= w_ || y1 >= h_ || x1 < x0 || y1 < y0) return;
    if (size_t(x1 - x0 + 1) * size_t(y1 - y0 + 1) + kHead != pl.size()) return;
    if (w_ != w || h_ != h || res_ != res || ox_ != ox || oy_ != oy) {   // 모양·원점이 바뀜: 새로(모름으로 채우고, 보낸 쪽이 전체를 보냄)
      w = w_; h = h_; res = res_; ox = ox_; oy = oy_;
      cells.assign(size_t(w) * h, int8_t(-1));
    }
    const int8_t* src = reinterpret_cast<const int8_t*>(&pl[kHead]);
    const int rw = x1 - x0 + 1;
    for (int y = y0; y <= y1; ++y) std::memcpy(&cells[size_t(y) * w + x0], src + size_t(y - y0) * rw, size_t(rw));
  }
  void fullFrame(std::vector<uint8_t>& out) const {
    if (w <= 0) return;
    std::vector<uint8_t> pl(48 + cells.size());
    const int32_t z = 0, ww = w, hh = h, xe = w - 1, ye = h - 1;
    std::memcpy(&pl[0], &ww, 4); std::memcpy(&pl[4], &hh, 4);
    std::memcpy(&pl[8], &res, 8); std::memcpy(&pl[16], &ox, 8); std::memcpy(&pl[24], &oy, 8);
    std::memcpy(&pl[32], &z, 4); std::memcpy(&pl[36], &z, 4); std::memcpy(&pl[40], &xe, 4); std::memcpy(&pl[44], &ye, 4);
    std::memcpy(&pl[48], cells.data(), cells.size());
    appendFrame(out, kStreamMapRect, pl.data(), pl.size());
  }
};

int connectTo(const std::string& host_port) {
  const size_t c = host_port.rfind(':');
  if (c == std::string::npos) return -1;
  const std::string host = host_port.substr(0, c);
  const int port = std::atoi(host_port.c_str() + c + 1);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(uint16_t(port));
  if (inet_pton(AF_INET, host.empty() ? "127.0.0.1" : host.c_str(), &a.sin_addr) != 1) return -1;
  const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0 && errno != EINPROGRESS) { ::close(fd); return -1; }
  pollfd p{fd, POLLOUT, 0};
  if (::poll(&p, 1, 50) <= 0) { ::close(fd); return -1; }
  int err = 0;
  socklen_t el = sizeof(err);
  getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el);
  if (err) { ::close(fd); return -1; }
  return fd;
}

}  // namespace

Streamer::Streamer() : hot_(1u << 22), view_(1u << 21) {}
Streamer::~Streamer() { stop(); }

bool Streamer::start(const std::string& host_port) {
  if (running_.exchange(true)) return false;
  host_port_ = host_port;
  stop_ = false;
  th_ = std::thread([this] { run(); });
  return true;
}

void Streamer::stop() {
  if (!running_.load()) return;
  stop_ = true;
  if (th_.joinable()) th_.join();
  running_ = false;
  connected_ = false;
}

bool Streamer::pushPose(double stamp, double x, double y, double yaw) {
  const double v[4] = {stamp, x, y, yaw};
  in_.fetch_add(1, std::memory_order_relaxed);
  if (hot_.push(kStreamPose, v, sizeof(v))) return true;
  dropped_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool Streamer::pushJoints(double stamp, const float* q, int n) {
  if (n <= 0 || n > 4096) return false;
  uint8_t head[12];
  const int32_t n_ = n;
  std::memcpy(head, &stamp, 8);
  std::memcpy(head + 8, &n_, 4);
  in_.fetch_add(1, std::memory_order_relaxed);
  if (hot_.push2(kStreamJoints, head, sizeof(head), q, size_t(n) * 4)) return true;
  dropped_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool Streamer::pushMapRect(int w, int h, double res, double ox, double oy, int x0, int y0, int x1, int y1, const int8_t* cells) {
  uint8_t head[48];
  const int32_t w_ = w, h_ = h, a = x0, b = y0, c = x1, d = y1;
  std::memcpy(head, &w_, 4); std::memcpy(head + 4, &h_, 4);
  std::memcpy(head + 8, &res, 8); std::memcpy(head + 16, &ox, 8); std::memcpy(head + 24, &oy, 8);
  std::memcpy(head + 32, &a, 4); std::memcpy(head + 36, &b, 4); std::memcpy(head + 40, &c, 4); std::memcpy(head + 44, &d, 4);
  in_.fetch_add(1, std::memory_order_relaxed);
  if (hot_.push2(kStreamMapRect, head, sizeof(head), cells, size_t(x1 - x0 + 1) * size_t(y1 - y0 + 1))) return true;
  dropped_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool Streamer::pushView(const std::string& json) {
  in_.fetch_add(1, std::memory_order_relaxed);
  if (view_.push(kStreamView, json.data(), json.size())) return true;
  dropped_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

StreamStats Streamer::stats() const {
  StreamStats s;
  s.frames_in = in_.load(); s.dropped = dropped_.load(); s.frames_sent = sent_.load(); s.bytes_sent = bytes_.load();
  s.reconnects = reconn_.load(); s.connected = connected_.load();
  return s;
}

void Streamer::run() {
  int fd = -1;
  std::vector<uint8_t> out;
  size_t off = 0;
  Shadow sh;
  auto last_try = Clk::now() - std::chrono::seconds(1);
  std::vector<uint8_t> pl;
  uint8_t type = 0;
  while (!stop_.load(std::memory_order_relaxed)) {
    bool work = false;
    bool pose_dirty = false;
    auto handle = [&](uint8_t t, const std::vector<uint8_t>& p) {
      work = true;
      if (t == kStreamPose && p.size() == 32) {
        std::memcpy(sh.pose, p.data(), 32);
        sh.have_pose = true;
        pose_dirty = true;   // 한 번 비울 때 자세는 마지막 것만 보낸다(대역폭)
      } else if (t == kStreamMapRect) {
        sh.applyRect(p);
        if (fd >= 0) appendFrame(out, t, p.data(), p.size());
      } else if (t == kStreamView) {
        sh.view = p;
        if (fd >= 0) appendFrame(out, t, p.data(), p.size());
      } else if (t == kStreamJoints) {
        sh.joints = p;
        if (fd >= 0) appendFrame(out, t, p.data(), p.size());
      }
    };
    while (hot_.pop(&type, &pl)) handle(type, pl);
    while (view_.pop(&type, &pl)) handle(type, pl);
    if (pose_dirty && fd >= 0) appendFrame(out, kStreamPose, reinterpret_cast<const uint8_t*>(sh.pose), 32);
    // 연결(끊겼으면 0.5 s 마다 다시). 붙으면 전체 상태를 보낸다
    if (fd < 0 && Clk::now() - last_try > std::chrono::milliseconds(500)) {
      last_try = Clk::now();
      fd = connectTo(host_port_);
      if (fd >= 0) {
        connected_ = true;
        reconn_.fetch_add(1);
        out.clear(); off = 0;
        sh.fullFrame(out);
        if (!sh.view.empty()) appendFrame(out, kStreamView, sh.view.data(), sh.view.size());
        if (sh.have_pose) appendFrame(out, kStreamPose, reinterpret_cast<const uint8_t*>(sh.pose), 32);
        if (!sh.joints.empty()) appendFrame(out, kStreamJoints, sh.joints.data(), sh.joints.size());
      }
    }
    if (fd >= 0 && off < out.size()) {
      const ssize_t n = ::send(fd, out.data() + off, out.size() - off, MSG_NOSIGNAL | MSG_DONTWAIT);
      if (n > 0) { off += size_t(n); bytes_.fetch_add(uint64_t(n)); work = true; }
      else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { ::close(fd); fd = -1; connected_ = false; out.clear(); off = 0; }
      if (off == out.size()) { sent_.fetch_add(1); out.clear(); off = 0; }
      else if (out.size() - off > (32u << 20)) { ::close(fd); fd = -1; connected_ = false; out.clear(); off = 0; }   // 뷰어가 너무 느림: 끊고 다시 붙을 때 전체 상태
    }
    if (!work) {
      if (fd >= 0 && off < out.size()) { pollfd p{fd, POLLOUT, 0}; ::poll(&p, 1, 1); }
      else std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  if (fd >= 0) ::close(fd);
}

}  // namespace scenemap
