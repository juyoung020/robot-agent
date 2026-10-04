// safetensors 읽기(mmap) + Qwen3.5 config.json 의 text_config 몇 값 읽기. 작은 손 파서(이 두 형식만).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rvla {

struct StTensor {
  std::string dtype;            // "BF16" | "F32" | ...
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t bytes = 0;
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

struct StFile {
  std::map<std::string, StTensor> t;
  void* map = nullptr;
  size_t size = 0;
  bool open(const std::string& path, std::string* err);
  void close();
  ~StFile() { close(); }
  const StTensor* get(const std::string& name) const { auto it = t.find(name); return it == t.end() ? nullptr : &it->second; }
  // 텐서 하나를 FP32 로(BF16·F32)
  std::vector<float> f32(const std::string& name) const;
};

// config.json 에서 "key": 숫자 / 문자열 배열 찾기(text_config 구역 안에서)
struct JsonLite {
  std::string s;
  bool load(const std::string& path);
  std::string section(const std::string& key) const;   // {"key": {...}} 의 {...}
  static double num(const std::string& sec, const std::string& key, double dflt);
  static std::vector<std::string> strs(const std::string& sec, const std::string& key);
};

}  // namespace rvla
