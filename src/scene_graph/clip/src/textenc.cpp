// sgclip 글 인코더(include/sgclip.h "글 인코더"): Gemma BPE 토크나이저 + 토큰 임베딩 모으기(CPU, mmap) + TensorRT 글 탑.
//
//   토큰화(open_clip HFTokenizer, clean="canonicalize" 와 같게 — 시험 test_textenc 가 HF 토큰 번호와 비교):
//     '_' → ' ', ASCII 문장 부호(string.punctuation) 지움, 소문자(ASCII·라틴-1·그리스·키릴), 공백 하나로·앞뒤 지움,
//     ' ' → '▁', 글 전체가 BPE 낱말 하나(Gemma 는 미리 자르지 않음): 문자마다 어휘 번호(없으면 바이트 <0xAB>) →
//     인접 쌍 중 merge 순위가 가장 낮은 것(같으면 왼쪽)을 되풀이해 합침 → 63 개로 자르고 <eos>(1) → 64 까지 <pad>(0).
//     ftfy·html 풀기는 안 함(질의는 평범한 글).
//   모으기: 토큰 번호 → siglip2_b32_tokemb.f16 의 줄(FP16 → FP32). 파일은 mmap(질의 하나가 ≤ 64 줄만 건드림).
//   엔진: tok_emb N×64×768 FP32 → emb N×768(엔진 안에서 L2, FP16 출력이면 여기서 다시 정규화).
#include "ra_paths.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "sgclip.h"

namespace {

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kERROR) std::fprintf(stderr, "[sgclip text TRT] %s\n", msg);
  }
};
Logger gLog;

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct TrtDel {
  void operator()(T* p) const { delete p; }
};
template <class T>
using TrtPtr = std::unique_ptr<T, TrtDel<T>>;

// ---- UTF-8 ----
int u8len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

uint32_t decode(const std::string& s, size_t i, int n) {
  const auto* p = reinterpret_cast<const unsigned char*>(s.data() + i);
  switch (n) {
    case 2: return (uint32_t(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return (uint32_t(p[0] & 0x0f) << 12) | (uint32_t(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    case 4: return (uint32_t(p[0] & 0x07) << 18) | (uint32_t(p[1] & 0x3f) << 12) | (uint32_t(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    default: return p[0];
  }
}

void encode(uint32_t c, std::string* o) {
  if (c < 0x80) o->push_back(char(c));
  else if (c < 0x800) { o->push_back(char(0xc0 | (c >> 6))); o->push_back(char(0x80 | (c & 0x3f))); }
  else if (c < 0x10000) { o->push_back(char(0xe0 | (c >> 12))); o->push_back(char(0x80 | ((c >> 6) & 0x3f))); o->push_back(char(0x80 | (c & 0x3f))); }
  else {
    o->push_back(char(0xf0 | (c >> 18))); o->push_back(char(0x80 | ((c >> 12) & 0x3f)));
    o->push_back(char(0x80 | ((c >> 6) & 0x3f))); o->push_back(char(0x80 | (c & 0x3f)));
  }
}

uint32_t lowerCp(uint32_t c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if ((c >= 0xC0 && c <= 0xDE && c != 0xD7)) return c + 32;            // 라틴-1
  if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 32;           // 그리스
  if (c >= 0x410 && c <= 0x42F) return c + 32;                         // 키릴
  if (c >= 0x400 && c <= 0x40F) return c + 80;
  return c;
}

bool isSpace(uint32_t c) {   // 파이썬 str.split() 의 공백 중 흔한 것
  return c == ' ' || (c >= 0x09 && c <= 0x0d) || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x1f || c == 0x85 || c == 0xa0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

bool isPunct(uint32_t c) { return c < 0x80 && std::strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", int(c)) && c != 0; }

// canonicalize: 결과는 '▁' 로 이은 글(공백 없음)
std::string canonical(const char* text) {
  const std::string s = text ? text : "";
  std::string out;
  bool pend_space = false;
  for (size_t i = 0; i < s.size();) {
    const int n = std::min<int>(u8len(static_cast<unsigned char>(s[i])), int(s.size() - i));
    uint32_t c = decode(s, i, n);
    i += size_t(n);
    if (c == '_') c = ' ';
    if (isPunct(c)) continue;
    if (isSpace(c)) { pend_space = !out.empty(); continue; }
    if (pend_space) { out += "\xe2\x96\x81"; pend_space = false; }   // ▁
    encode(lowerCp(c), &out);
  }
  return out;
}

struct Mmap {
  void* p = MAP_FAILED;
  size_t n = 0;
  ~Mmap() {
    if (p != MAP_FAILED) munmap(p, n);
  }
};

const char* defaultDir() {
  static std::string d;
  if (const char* e = std::getenv("SGC_TEXT_DIR")) d = e;
  else d = ra::models() + "/x86_sm120/siglip2_b32";
  return d.c_str();
}

}  // namespace

struct sgc_text {
  sgc_text_config cfg{};
  // 토크나이저
  std::vector<std::string> vocab;
  std::unordered_map<std::string, int32_t> id;
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merge;   // (왼, 오른) → (순위, 합친 번호)
  int32_t ctx = SGC_TEXT_CTX, eos = 1, pad = 0, unk = 3;
  int32_t byte_id[256];
  // 토큰 임베딩
  Mmap emb;
  const uint16_t* E = nullptr;
  // 엔진
  TrtPtr<nvinfer1::IRuntime> rt;
  TrtPtr<nvinfer1::ICudaEngine> eng;
  TrtPtr<nvinfer1::IExecutionContext> ctxx;
  cudaStream_t stream = nullptr;
  float* d_in = nullptr;
  void* d_out = nullptr;
  std::vector<float> h_in;
  std::vector<uint16_t> h_out16;
  std::vector<float> h_out;
  bool out_half = false;
  int maxb = 8;
  float last_ms = 0;
  int64_t dev_bytes = 0;
#if NV_TENSORRT_MAJOR < 10
  int idx_in = 0, idx_out = 1;
#endif
  ~sgc_text() {
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (stream) cudaStreamDestroy(stream);
  }

  int tokenize(const char* text, int32_t* out) const {
    const std::string s = canonical(text);
    std::vector<int32_t> sym;
    for (size_t i = 0; i < s.size();) {
      const int n = std::min<int>(u8len(static_cast<unsigned char>(s[i])), int(s.size() - i));
      auto it = id.find(s.substr(i, size_t(n)));
      if (it != id.end()) sym.push_back(it->second);
      else
        for (int b = 0; b < n; ++b) {
          const int32_t t = byte_id[static_cast<unsigned char>(s[i + size_t(b)])];
          sym.push_back(t >= 0 ? t : unk);
        }
      i += size_t(n);
    }
    for (;;) {   // 순위가 가장 낮은 인접 쌍(같으면 왼쪽)부터
      int best = -1, br = 1 << 30, bm = 0;
      for (size_t k = 0; k + 1 < sym.size(); ++k) {
        auto it = merge.find((uint64_t(uint32_t(sym[k])) << 32) | uint32_t(sym[k + 1]));
        if (it != merge.end() && it->second.first < br) br = it->second.first, best = int(k), bm = it->second.second;
      }
      if (best < 0) break;
      sym[size_t(best)] = bm;
      sym.erase(sym.begin() + best + 1);
    }
    if (int(sym.size()) > ctx - 1) sym.resize(size_t(ctx - 1));
    sym.push_back(eos);
    for (int k = 0; k < ctx; ++k) out[k] = k < int(sym.size()) ? sym[size_t(k)] : pad;
    return int(sym.size());
  }
};

namespace {

void loadTokenizer(sgc_text* T, const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot open tokenizer ") + path);
  int32_t h[7];
  f.read(reinterpret_cast<char*>(h), sizeof(h));
  if (!f || h[0] != 0x31544753) throw std::runtime_error("bad tokenizer file (magic)");
  const int nv = h[1], nm = h[2];
  T->ctx = h[3], T->eos = h[4], T->pad = h[5], T->unk = h[6];
  if (T->ctx != SGC_TEXT_CTX) throw std::runtime_error("tokenizer context != 64");
  T->vocab.resize(size_t(nv));
  T->id.reserve(size_t(nv) * 2);
  for (int i = 0; i < nv; ++i) {
    uint16_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 2);
    T->vocab[size_t(i)].resize(n);
    f.read(T->vocab[size_t(i)].data(), n);
    T->id.emplace(T->vocab[size_t(i)], i);
  }
  std::vector<int32_t> m(size_t(nm) * 3);
  f.read(reinterpret_cast<char*>(m.data()), std::streamsize(m.size() * 4));
  if (!f) throw std::runtime_error("truncated tokenizer file");
  T->merge.reserve(size_t(nm) * 2);
  for (int r = 0; r < nm; ++r)
    T->merge.emplace((uint64_t(uint32_t(m[size_t(r) * 3])) << 32) | uint32_t(m[size_t(r) * 3 + 1]), std::make_pair(r, m[size_t(r) * 3 + 2]));
  for (int b = 0; b < 256; ++b) {
    char k[8];
    std::snprintf(k, sizeof(k), "<0x%02X>", b);
    auto it = T->id.find(k);
    T->byte_id[b] = it == T->id.end() ? -1 : it->second;
  }
}

float h2f(uint16_t h) {
  float o;
  sgc_f16_to_f32(&h, &o, 1);
  return o;
}

}  // namespace

extern "C" {

void sgc_text_default_config(sgc_text_config* c, const char* dir) {
  if (!c) return;
  static std::string e, t, m;
  const std::string d = dir && *dir ? dir : defaultDir();
  e = d + "/siglip2_b32_text_fp16.plan";
  t = d + "/siglip2_b32_tok.bin";
  m = d + "/siglip2_b32_tokemb.f16";
  *c = sgc_text_config{e.c_str(), t.c_str(), m.c_str(), 0, 8};
}

sgc_text* sgc_text_create(const sgc_text_config* c, char* err, size_t err_len) {
  auto T = std::make_unique<sgc_text>();
  try {
    if (!c || !c->tokenizer) throw std::runtime_error("sgc_text_create: tokenizer path required");
    T->cfg = *c;
    loadTokenizer(T.get(), c->tokenizer);
    if (!c->engine) return T.release();   // 토크나이저만
    if (!c->tok_emb) throw std::runtime_error("token embedding path required with an engine");
    const int fd = open(c->tok_emb, O_RDONLY);
    if (fd < 0) throw std::runtime_error(std::string("cannot open ") + c->tok_emb);
    struct stat st{};
    fstat(fd, &st);
    T->emb.n = size_t(st.st_size);
    T->emb.p = mmap(nullptr, T->emb.n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (T->emb.p == MAP_FAILED) throw std::runtime_error("mmap token embedding failed");
    if (T->emb.n != T->vocab.size() * SGC_DIM * 2) throw std::runtime_error("token embedding size != vocab x 768 x 2");
    T->E = static_cast<const uint16_t*>(T->emb.p);

    ck(cudaSetDevice(c->device), "set device");
    std::ifstream f(c->engine, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + c->engine);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    T->rt.reset(nvinfer1::createInferRuntime(gLog));
#if NV_TENSORRT_MAJOR >= 10
    T->eng.reset(T->rt->deserializeCudaEngine(blob.data(), blob.size()));
#else
    T->eng.reset(T->rt->deserializeCudaEngine(blob.data(), blob.size(), nullptr));
#endif
    if (!T->eng) throw std::runtime_error(std::string("cannot deserialize ") + c->engine);
    T->ctxx.reset(T->eng->createExecutionContext());
    if (!T->ctxx) throw std::runtime_error("cannot create execution context");
#if NV_TENSORRT_MAJOR >= 10
    const nvinfer1::Dims mx = T->eng->getProfileShape("tok_emb", 0, nvinfer1::OptProfileSelector::kMAX);
    if (T->eng->getTensorDataType("tok_emb") != nvinfer1::DataType::kFLOAT) throw std::runtime_error("tok_emb must be FP32");
    T->out_half = T->eng->getTensorDataType("emb") == nvinfer1::DataType::kHALF;
#else
    T->idx_in = T->eng->getBindingIndex("tok_emb");
    T->idx_out = T->eng->getBindingIndex("emb");
    if (T->idx_in < 0 || T->idx_out < 0) throw std::runtime_error("engine tensors tok_emb/emb not found");
    const nvinfer1::Dims mx = T->eng->getProfileDimensions(T->idx_in, 0, nvinfer1::OptProfileSelector::kMAX);
    T->out_half = T->eng->getBindingDataType(T->idx_out) == nvinfer1::DataType::kHALF;
#endif
    if (mx.d[1] != SGC_TEXT_CTX || mx.d[2] != SGC_DIM) throw std::runtime_error("unexpected text engine shape");
    T->maxb = std::max(1, std::min<int>(c->max_batch > 0 ? c->max_batch : 8, int(mx.d[0])));
    ck(cudaStreamCreateWithFlags(&T->stream, cudaStreamNonBlocking), "stream");
    const size_t in_b = size_t(T->maxb) * SGC_TEXT_CTX * SGC_DIM * 4, out_b = size_t(T->maxb) * SGC_DIM * (T->out_half ? 2 : 4);
    ck(cudaMalloc(reinterpret_cast<void**>(&T->d_in), in_b), "malloc in");
    ck(cudaMalloc(&T->d_out, out_b), "malloc out");
    T->h_in.resize(size_t(T->maxb) * SGC_TEXT_CTX * SGC_DIM);
    T->h_out.resize(size_t(T->maxb) * SGC_DIM);
    T->h_out16.resize(size_t(T->maxb) * SGC_DIM);
#if NV_TENSORRT_MAJOR >= 10
    T->dev_bytes = int64_t(T->eng->getDeviceMemorySizeV2()) + int64_t(in_b + out_b) + int64_t(blob.size());
#else
    T->dev_bytes = int64_t(T->eng->getDeviceMemorySize()) + int64_t(in_b + out_b) + int64_t(blob.size());
#endif
    return T.release();
  } catch (const std::exception& x) {
    if (err && err_len) std::snprintf(err, err_len, "%s", x.what());
    return nullptr;
  }
}

void sgc_text_destroy(sgc_text* T) { delete T; }

int32_t sgc_text_tokenize(const sgc_text* T, const char* text, int32_t* ids) {
  if (!T || !ids) return -1;
  return T->tokenize(text, ids);
}

int32_t sgc_text_encode_ids(sgc_text* T, const int32_t* ids, int32_t n, float* out) {
  if (!T || !T->eng || !ids || !out || n < 0) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    for (int off = 0; off < n; off += T->maxb) {
      const int b = std::min(T->maxb, n - off);
      for (int i = 0; i < b; ++i)
        for (int k = 0; k < SGC_TEXT_CTX; ++k) {
          int32_t t = ids[size_t(off + i) * SGC_TEXT_CTX + size_t(k)];
          if (t < 0 || t >= int32_t(T->vocab.size())) t = T->unk;
          sgc_f16_to_f32(T->E + size_t(t) * SGC_DIM, &T->h_in[(size_t(i) * SGC_TEXT_CTX + size_t(k)) * SGC_DIM], SGC_DIM);
        }
      ck(cudaMemcpyAsync(T->d_in, T->h_in.data(), size_t(b) * SGC_TEXT_CTX * SGC_DIM * 4, cudaMemcpyHostToDevice, T->stream), "h2d");
#if NV_TENSORRT_MAJOR >= 10
      T->ctxx->setInputShape("tok_emb", nvinfer1::Dims3{b, SGC_TEXT_CTX, SGC_DIM});
      T->ctxx->setTensorAddress("tok_emb", T->d_in);
      T->ctxx->setTensorAddress("emb", T->d_out);
      if (!T->ctxx->enqueueV3(T->stream)) throw std::runtime_error("enqueue failed");
#else
      T->ctxx->setBindingDimensions(T->idx_in, nvinfer1::Dims3{b, SGC_TEXT_CTX, SGC_DIM});
      void* bind[2];
      bind[T->idx_in] = T->d_in;
      bind[T->idx_out] = T->d_out;
      if (!T->ctxx->enqueueV2(bind, T->stream, nullptr)) throw std::runtime_error("enqueue failed");
#endif
      void* h = T->out_half ? static_cast<void*>(T->h_out16.data()) : static_cast<void*>(T->h_out.data());
      ck(cudaMemcpyAsync(h, T->d_out, size_t(b) * SGC_DIM * (T->out_half ? 2 : 4), cudaMemcpyDeviceToHost, T->stream), "d2h");
      ck(cudaStreamSynchronize(T->stream), "sync");
      for (int i = 0; i < b; ++i) {
        float* o = out + size_t(off + i) * SGC_DIM;
        double nn = 0;
        for (int d = 0; d < SGC_DIM; ++d) {
          o[d] = T->out_half ? h2f(T->h_out16[size_t(i) * SGC_DIM + size_t(d)]) : T->h_out[size_t(i) * SGC_DIM + size_t(d)];
          nn += double(o[d]) * o[d];
        }
        const float inv = nn > 0 ? float(1.0 / std::sqrt(nn)) : 0.f;
        for (int d = 0; d < SGC_DIM; ++d) o[d] *= inv;
      }
    }
  } catch (const std::exception& x) {
    std::fprintf(stderr, "[sgclip text] %s\n", x.what());
    return -1;
  }
  T->last_ms = float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  return 0;
}

int32_t sgc_text_encode(sgc_text* T, const char* const* texts, int32_t n, float* out) {
  if (!T || !texts || n < 0) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<int32_t> ids(size_t(n) * SGC_TEXT_CTX);
  for (int i = 0; i < n; ++i) T->tokenize(texts[i], &ids[size_t(i) * SGC_TEXT_CTX]);
  const int r = sgc_text_encode_ids(T, ids.data(), n, out);
  T->last_ms = float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  return r;
}

float sgc_text_last_ms(const sgc_text* T) { return T ? T->last_ms : 0.f; }
int64_t sgc_text_device_bytes(const sgc_text* T) { return T ? T->dev_bytes : 0; }

}  // extern "C"
