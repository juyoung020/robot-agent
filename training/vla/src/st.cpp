#include "st.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace rvla {

namespace {
struct P {
  const char* p;
  const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
  bool ch(char c) { ws(); if (p < e && *p == c) { ++p; return true; } return false; }
  bool str(std::string& o) {
    ws();
    if (p >= e || *p != '"') return false;
    ++p;
    o.clear();
    while (p < e && *p != '"') { if (*p == '\\' && p + 1 < e) { ++p; } o.push_back(*p++); }
    if (p >= e) return false;
    ++p;
    return true;
  }
  bool num(int64_t& v) { ws(); char* q; v = strtoll(p, &q, 10); if (q == p) return false; p = q; return true; }
  bool skip() {   // 값 하나 건너뛰기
    ws();
    if (p >= e) return false;
    if (*p == '"') { std::string s; return str(s); }
    if (*p == '{' || *p == '[') {
      int d = 0;
      bool in = false;
      for (; p < e; ++p) {
        if (in) { if (*p == '\\') ++p; else if (*p == '"') in = false; continue; }
        if (*p == '"') in = true;
        else if (*p == '{' || *p == '[') ++d;
        else if (*p == '}' || *p == ']') { if (--d == 0) { ++p; return true; } }
      }
      return false;
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']') ++p;
    return true;
  }
};
}  // namespace

bool StFile::open(const std::string& path, std::string* err) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { if (err) *err = "open " + path; return false; }
  struct stat sb;
  fstat(fd, &sb);
  size = (size_t)sb.st_size;
  map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) { map = nullptr; if (err) *err = "mmap"; return false; }
  const uint8_t* b = (const uint8_t*)map;
  uint64_t hl;
  memcpy(&hl, b, 8);
  const uint8_t* base = b + 8 + hl;
  P ps{(const char*)b + 8, (const char*)b + 8 + hl};
  if (!ps.ch('{')) { if (err) *err = "header"; return false; }
  while (true) {
    std::string name;
    if (!ps.str(name)) break;
    if (!ps.ch(':')) return false;
    if (name == "__metadata__") { ps.skip(); ps.ch(','); continue; }
    StTensor tt;
    int64_t o0 = -1, o1 = -1;
    if (!ps.ch('{')) return false;
    while (true) {
      std::string k;
      if (!ps.str(k)) break;
      ps.ch(':');
      if (k == "dtype") ps.str(tt.dtype);
      else if (k == "shape") {
        ps.ch('[');
        int64_t v;
        while (ps.num(v)) { tt.shape.push_back(v); ps.ch(','); }
        ps.ch(']');
      } else if (k == "data_offsets") {
        ps.ch('['); ps.num(o0); ps.ch(','); ps.num(o1); ps.ch(']');
      } else ps.skip();
      ps.ch(',');
    }
    ps.ch('}');
    tt.data = base + o0;
    tt.bytes = (size_t)(o1 - o0);
    t[name] = tt;
    if (!ps.ch(',')) break;
  }
  return !t.empty();
}

void StFile::close() {
  if (map) munmap(map, size);
  map = nullptr;
}

std::vector<float> StFile::f32(const std::string& name) const {
  const StTensor* x = get(name);
  if (!x) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::abort(); }
  std::vector<float> o((size_t)x->numel());
  if (x->dtype == "F32") memcpy(o.data(), x->data, o.size() * 4);
  else if (x->dtype == "BF16") {
    const uint16_t* h = (const uint16_t*)x->data;
    for (size_t i = 0; i < o.size(); ++i) { uint32_t u = (uint32_t)h[i] << 16; memcpy(&o[i], &u, 4); }
  } else { std::fprintf(stderr, "dtype %s\n", x->dtype.c_str()); std::abort(); }
  return o;
}

bool JsonLite::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  s = ss.str();
  return true;
}
std::string JsonLite::section(const std::string& key) const {
  const size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return s;
  const size_t a = s.find('{', k);
  P ps{s.c_str() + a, s.c_str() + s.size()};
  const char* st = ps.p;
  ps.skip();
  return std::string(st, ps.p);
}
double JsonLite::num(const std::string& sec, const std::string& key, double d) {
  const size_t k = sec.find("\"" + key + "\"");
  if (k == std::string::npos) return d;
  const size_t c = sec.find(':', k);
  return strtod(sec.c_str() + c + 1, nullptr);
}
std::vector<std::string> JsonLite::strs(const std::string& sec, const std::string& key) {
  std::vector<std::string> o;
  const size_t k = sec.find("\"" + key + "\"");
  if (k == std::string::npos) return o;
  P ps{sec.c_str() + sec.find('[', k) + 1, sec.c_str() + sec.size()};
  std::string v;
  while (ps.str(v)) { o.push_back(v); ps.ch(','); }
  return o;
}

}  // namespace rvla
