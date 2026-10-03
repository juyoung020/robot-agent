// sgclip 영상 인코더(include/sgclip.h): TensorRT 실행 + 자르기 커널 + 칸 2개 고리.
//
// TensorRT 8.2(JetPack 4.6) / 10 차이는 이 파일 안에서만(NV_TENSORRT_MAJOR):
//   이름 → 텐서: 10 = setTensorAddress / setInputShape / enqueueV3,  8 = getBindingIndex / setBindingDimensions / enqueueV2
//   프로필: 10 = 이름 그대로 + setOptimizationProfileAsync,  8 = 바인딩 번호 + 프로필 k × (바인딩 수 / 프로필 수)
//   자료형: 10 = getTensorDataType,  8 = getBindingDataType
//   CUDA graph 는 10 에서만 쓴다(8.2 + CUDA 10.2 에서 enqueueV2 잡기는 확인 안 함).
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "crop.hpp"
#include "sgclip.h"

using namespace sgclip;

namespace {

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kERROR) std::fprintf(stderr, "[sgclip TRT] %s\n", msg);
  }
};
Logger gLogger;

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct TrtDel {
  void operator()(T* p) const {
#if NV_TENSORRT_MAJOR >= 8
    delete p;
#else
    p->destroy();
#endif
  }
};
template <class T>
using TrtPtr = std::unique_ptr<T, TrtDel<T>>;

uint16_t f2h(float f) {   // 반올림(가장 가까운 짝수), 넘침은 inf
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0));
  if (e >= 31) return uint16_t(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return uint16_t(sign);
    m |= 0x800000u;
    const int sh = 14 - e;
    uint32_t r = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return uint16_t(sign | r);
  }
  uint32_t r = (uint32_t(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return uint16_t(sign | r);
}
float h2f(uint16_t h) {
  const uint32_t s = uint32_t(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, f;
  if (e == 0) {
    if (m == 0) f = s;
    else {
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3ffu;
      f = s | (e << 23) | (m << 13);
    }
  } else if (e == 31) f = s | 0x7f800000u | (m << 13);
  else f = s | ((e + 127 - 15) << 23) | (m << 13);
  float r;
  std::memcpy(&r, &f, 4);
  return r;
}

struct Slot {
  void* img = nullptr;          // 장치 max_batch×3×S×S
  float* wp = nullptr;          // 장치 max_batch×G²
  void* emb = nullptr;          // 장치 max_batch×768 (FP32 또는 FP16)
  void* emb_host = nullptr;     // 고정 메모리
  CropJob* jobs_dev = nullptr;
  uint32_t* bits_dev = nullptr;
  size_t bits_cap = 0;          // 단어
  CropJob* jobs_host = nullptr; // 고정
  uint32_t* bits_host = nullptr;// 고정
  cudaEvent_t e0{}, e_crop{}, e_done{};
  int n = 0;                    // 이 칸에 든 물체 수(0 = 빔)
  int last_n = 0;               // 마지막으로 넣은 수(sgc_debug_inputs)
  bool busy = false;            // 결과 아직 안 꺼냄
  double stamp = 0;
  std::vector<uint32_t> ids;
  std::vector<float> q;
  std::vector<cudaGraphExec_t> graphs;   // 배치 크기별(TRT 10)
};

}  // namespace

struct sgc_encoder {
  sgc_config cfg{};
  TrtPtr<nvinfer1::IRuntime> rt;
  TrtPtr<nvinfer1::ICudaEngine> eng;
  TrtPtr<nvinfer1::IExecutionContext> ctx;
  cudaStream_t stream = nullptr;
  int S = 256, G = 8, D = SGC_DIM;
  bool in_half = false, out_half = false;
  int nprof = 1;
  std::vector<std::pair<int, int>> prof_range;   // 프로필별 배치 (min, max)
  int cur_prof = -1;
  Slot slot[2];
  int next = 0;                 // 다음에 쓸 칸
  int64_t dev_bytes = 0;
  // 호스트 RGB 를 올릴 장치 버퍼
  uint8_t* frame_dev = nullptr;
  size_t frame_cap = 0;
  sgc_timing tm{};
  std::vector<float> pool;      // 결과 벡터 자리(4×max_batch 개 × 768) — poll 이 넘긴 포인터는 다음 poll 까지 유지
  std::vector<int> free_rows, lent;
  struct Pend { uint32_t id; float q; double stamp; int row; };
  std::vector<Pend> pend;       // 아직 못 꺼낸 결과(먼저 끝난 것부터)
  int last_slot = -1;
#if NV_TENSORRT_MAJOR < 10
  int nb_per_prof = 3;
  int idx_images = 0, idx_wpatch = 1, idx_emb = 2;
#endif
};

namespace {

int bucketFor(sgc_encoder* e, int n, int* prof) {
  int best = -1, bn = 1 << 30;
  for (int k = 0; k < e->nprof; ++k) {
    const auto [lo, hi] = e->prof_range[k];
    if (hi < n) continue;
    const int b = std::max(lo, n);
    if (b < bn) bn = b, best = k;
  }
  *prof = best;
  return best < 0 ? -1 : bn;
}

void setShapes(sgc_encoder* e, Slot& s, int prof, int nb) {
  nvinfer1::IExecutionContext* c = e->ctx.get();
  if (prof != e->cur_prof) {
    c->setOptimizationProfileAsync(prof, e->stream);
    e->cur_prof = prof;
  }
#if NV_TENSORRT_MAJOR >= 10
  c->setInputShape("images", nvinfer1::Dims4{nb, 3, e->S, e->S});
  c->setInputShape("wpatch", nvinfer1::Dims2{nb, e->G * e->G});
  c->setTensorAddress("images", s.img);
  c->setTensorAddress("wpatch", s.wp);
  c->setTensorAddress("emb", s.emb);
#else
  const int o = prof * e->nb_per_prof;
  c->setBindingDimensions(o + e->idx_images, nvinfer1::Dims4{nb, 3, e->S, e->S});
  c->setBindingDimensions(o + e->idx_wpatch, nvinfer1::Dims2{nb, e->G * e->G});
  (void)s;
#endif
}

bool enqueue(sgc_encoder* e, [[maybe_unused]] Slot& s, int prof) {
#if NV_TENSORRT_MAJOR >= 10
  (void)prof;
  return e->ctx->enqueueV3(e->stream);
#else
  std::vector<void*> b(size_t(e->nprof * e->nb_per_prof), nullptr);
  const int o = prof * e->nb_per_prof;
  b[o + e->idx_images] = s.img;
  b[o + e->idx_wpatch] = s.wp;
  b[o + e->idx_emb] = s.emb;
  return e->ctx->enqueueV2(b.data(), e->stream, nullptr);
#endif
}

void runNet(sgc_encoder* e, int si, int n) {
  Slot& s = e->slot[si];
  int prof = 0;
  const int nb = bucketFor(e, n, &prof);
  if (nb < 0) throw std::runtime_error("batch outside the engine profiles");
#if NV_TENSORRT_MAJOR >= 10
  // 칸·배치 크기별 graph(잡을 때의 프로필·주소·모양이 그대로 박힘 — 띄울 때 문맥 상태를 안 봄. 시험: 배치 1·3·8 섞어도 같은 답)
  const bool graph = e->cfg.use_graph != 0;
  if (graph && s.graphs[nb]) {
    ck(cudaGraphLaunch(s.graphs[nb], e->stream), "graph launch");
    return;
  }
  setShapes(e, s, prof, nb);
  if (graph) {   // 처음: 한 번 그냥 돌리고(TRT 준비), 다음부터 쓸 graph 를 잡음
    if (!enqueue(e, s, prof)) throw std::runtime_error("enqueue failed");
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal), "capture");
    const bool ok = enqueue(e, s, prof);
    ck(cudaStreamEndCapture(e->stream, &g), "end capture");
    if (ok && g) {
      cudaGraphExec_t x = nullptr;
      if (cudaGraphInstantiate(&x, g, 0) == cudaSuccess) s.graphs[nb] = x;
      cudaGraphDestroy(g);
    }
    return;   // 잡은 graph 는 이번엔 안 돌림(위에서 이미 한 번 돌았음)
  }
#else
  setShapes(e, s, prof, nb);
#endif
  if (!enqueue(e, s, prof)) throw std::runtime_error("enqueue failed");
}

void freeSlot(Slot& s) {
  cudaFree(s.img); cudaFree(s.wp); cudaFree(s.emb); cudaFree(s.jobs_dev); cudaFree(s.bits_dev);
  cudaFreeHost(s.emb_host); cudaFreeHost(s.jobs_host); cudaFreeHost(s.bits_host);
  for (cudaGraphExec_t g : s.graphs)
    if (g) cudaGraphExecDestroy(g);
  if (s.e0) cudaEventDestroy(s.e0);
  if (s.e_crop) cudaEventDestroy(s.e_crop);
  if (s.e_done) cudaEventDestroy(s.e_done);
  s = Slot{};
}

bool collect(sgc_encoder* e, int si) {   // 끝난 칸 → pend(자리가 모자라면 false — 칸은 그대로 둠)
  Slot& s = e->slot[si];
  if (int(e->free_rows.size()) < s.n) return false;
  float a = 0, b = 0;
  cudaEventElapsedTime(&a, s.e0, s.e_crop);
  cudaEventElapsedTime(&b, s.e_crop, s.e_done);
  e->tm.crop_ms = a;
  e->tm.net_ms = b;
  e->tm.last_batch = s.n;
  for (int i = 0; i < s.n; ++i) {
    const int row = e->free_rows.back();
    e->free_rows.pop_back();
    float* o = e->pool.data() + size_t(row) * e->D;
    double nn = 0;
    for (int d = 0; d < e->D; ++d) {
      o[d] = e->out_half ? h2f(static_cast<const uint16_t*>(s.emb_host)[size_t(i) * e->D + d])
                         : static_cast<const float*>(s.emb_host)[size_t(i) * e->D + d];
      nn += double(o[d]) * o[d];
    }
    const float inv = nn > 0 ? float(1.0 / std::sqrt(nn)) : 0.f;   // 엔진이 이미 정규화 — FP16 출력이면 다시
    for (int d = 0; d < e->D; ++d) o[d] *= inv;
    e->pend.push_back(sgc_encoder::Pend{s.ids[i], s.q[i], s.stamp, row});
  }
  e->tm.n_done += s.n;
  s.n = 0;
  s.busy = false;
  return true;
}

}  // namespace

extern "C" {

void sgc_default_config(sgc_config* c) {
  if (!c) return;
  *c = sgc_config{};
  c->max_batch = 8;
  c->use_graph = 1;
  c->margin = 0.1f;
}

sgc_encoder* sgc_create(const sgc_config* c, char* err, size_t err_len) {
  auto* e = new sgc_encoder();
  try {
    if (!c || !c->engine) throw std::runtime_error("sgc_create: engine path required");
    e->cfg = *c;
    if (e->cfg.max_batch <= 0) e->cfg.max_batch = 8;
    if (e->cfg.margin < 0) e->cfg.margin = 0.1f;
    ck(cudaSetDevice(c->device), "set device");
    std::ifstream f(c->engine, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + c->engine);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    e->rt.reset(nvinfer1::createInferRuntime(gLogger));
#if NV_TENSORRT_MAJOR >= 10
    e->eng.reset(e->rt->deserializeCudaEngine(blob.data(), blob.size()));
#else
    e->eng.reset(e->rt->deserializeCudaEngine(blob.data(), blob.size(), nullptr));
#endif
    if (!e->eng) throw std::runtime_error(std::string("cannot deserialize ") + c->engine);
    e->ctx.reset(e->eng->createExecutionContext());
    if (!e->ctx) throw std::runtime_error("cannot create execution context");
    e->nprof = e->eng->getNbOptimizationProfiles();
#if NV_TENSORRT_MAJOR >= 10
    const nvinfer1::Dims di = e->eng->getTensorShape("images"), dw = e->eng->getTensorShape("wpatch"), de = e->eng->getTensorShape("emb");
    e->in_half = e->eng->getTensorDataType("images") == nvinfer1::DataType::kHALF;
    e->out_half = e->eng->getTensorDataType("emb") == nvinfer1::DataType::kHALF;
    if (e->eng->getTensorDataType("wpatch") != nvinfer1::DataType::kFLOAT) throw std::runtime_error("wpatch must be FP32");
    for (int k = 0; k < e->nprof; ++k) {
      const nvinfer1::Dims mn = e->eng->getProfileShape("images", k, nvinfer1::OptProfileSelector::kMIN);
      const nvinfer1::Dims mx = e->eng->getProfileShape("images", k, nvinfer1::OptProfileSelector::kMAX);
      e->prof_range.emplace_back(int(mn.d[0]), int(mx.d[0]));
    }
#else
    e->nb_per_prof = e->eng->getNbBindings() / e->nprof;
    e->idx_images = e->eng->getBindingIndex("images");
    e->idx_wpatch = e->eng->getBindingIndex("wpatch");
    e->idx_emb = e->eng->getBindingIndex("emb");
    if (e->idx_images < 0 || e->idx_wpatch < 0 || e->idx_emb < 0) throw std::runtime_error("engine tensors images/wpatch/emb not found");
    const nvinfer1::Dims di = e->eng->getBindingDimensions(e->idx_images), dw = e->eng->getBindingDimensions(e->idx_wpatch),
                         de = e->eng->getBindingDimensions(e->idx_emb);
    e->in_half = e->eng->getBindingDataType(e->idx_images) == nvinfer1::DataType::kHALF;
    e->out_half = e->eng->getBindingDataType(e->idx_emb) == nvinfer1::DataType::kHALF;
    for (int k = 0; k < e->nprof; ++k) {
      const int b = k * e->nb_per_prof + e->idx_images;
      const nvinfer1::Dims mn = e->eng->getProfileDimensions(b, k, nvinfer1::OptProfileSelector::kMIN);
      const nvinfer1::Dims mx = e->eng->getProfileDimensions(b, k, nvinfer1::OptProfileSelector::kMAX);
      e->prof_range.emplace_back(int(mn.d[0]), int(mx.d[0]));
    }
#endif
    e->S = int(di.d[3]);
    e->G = int(std::lround(std::sqrt(double(dw.d[1]))));
    e->D = int(de.d[1]);
    if (e->D != SGC_DIM || e->G * e->G != dw.d[1] || e->S % e->G) throw std::runtime_error("unexpected engine shapes");
    int maxb = 0;
    for (auto& r : e->prof_range) maxb = std::max(maxb, r.second);
    e->cfg.max_batch = std::min(e->cfg.max_batch, maxb);
#if NV_TENSORRT_MAJOR < 10
    e->cfg.use_graph = 0;
#endif
    ck(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking), "stream");
    const int B = e->cfg.max_batch, S = e->S, G = e->G;
    const size_t img_b = size_t(B) * 3 * S * S * (e->in_half ? 2 : 4), emb_b = size_t(B) * e->D * (e->out_half ? 2 : 4);
    for (Slot& s : e->slot) {
      ck(cudaMalloc(&s.img, img_b), "img");
      ck(cudaMemset(s.img, 0, img_b), "img0");
      ck(cudaMalloc(reinterpret_cast<void**>(&s.wp), size_t(B) * G * G * 4), "wpatch");
      ck(cudaMemset(s.wp, 0, size_t(B) * G * G * 4), "wp0");
      ck(cudaMalloc(&s.emb, emb_b), "emb");
      ck(cudaHostAlloc(&s.emb_host, emb_b, cudaHostAllocDefault), "emb host");
      ck(cudaMalloc(reinterpret_cast<void**>(&s.jobs_dev), sizeof(CropJob) * B), "jobs");
      ck(cudaHostAlloc(reinterpret_cast<void**>(&s.jobs_host), sizeof(CropJob) * B, cudaHostAllocDefault), "jobs host");
      ck(cudaEventCreate(&s.e0), "ev");
      ck(cudaEventCreate(&s.e_crop), "ev");
      ck(cudaEventCreate(&s.e_done), "ev");
      s.ids.resize(B);
      s.q.resize(B);
      s.graphs.assign(size_t(B) + 1, nullptr);
      e->dev_bytes += int64_t(img_b + size_t(B) * G * G * 4 + emb_b + sizeof(CropJob) * B);
    }
    e->pool.resize(size_t(4) * B * e->D);
    for (int r = 4 * B - 1; r >= 0; --r) e->free_rows.push_back(r);
    e->lent.reserve(size_t(4) * B);
    e->pend.reserve(size_t(4) * B);
#if NV_TENSORRT_MAJOR >= 10
    e->dev_bytes += int64_t(e->eng->getDeviceMemorySizeV2()) + int64_t(blob.size());
#else
    e->dev_bytes += int64_t(e->eng->getDeviceMemorySize()) + int64_t(blob.size());
#endif
    e->tm.input_size = S;
    e->tm.grid = G;
    e->tm.device_bytes = e->dev_bytes;
    return e;
  } catch (const std::exception& x) {
    if (err && err_len) std::snprintf(err, err_len, "%s", x.what());
    sgc_destroy(e);
    return nullptr;
  }
}

void sgc_destroy(sgc_encoder* e) {
  if (!e) return;
  if (e->stream) cudaStreamSynchronize(e->stream);
  for (Slot& s : e->slot) freeSlot(s);
  cudaFree(e->frame_dev);
  e->ctx.reset();
  e->eng.reset();
  e->rt.reset();
  if (e->stream) cudaStreamDestroy(e->stream);
  delete e;
}

int32_t sgc_submit(sgc_encoder* e, double stamp, const sgc_frame* f, const sgc_item* items, int32_t n) {
  if (!e || !f || !f->rgb || (n > 0 && !items)) return -1;
  if (n <= 0) return 0;
  const auto t0 = std::chrono::steady_clock::now();
  // 빈 칸 찾기(끝났지만 안 꺼낸 칸은 결과를 먼저 옮김)
  int si = -1;
  for (int k = 0; k < 2; ++k) {
    const int c = (e->next + k) % 2;
    Slot& s = e->slot[c];
    if (s.busy && s.n > 0 && cudaEventQuery(s.e_done) == cudaSuccess) collect(e, c);
    if (!s.busy) { si = c; break; }
  }
  if (si < 0) {
    e->tm.n_dropped += n;
    return 0;
  }
  Slot& s = e->slot[si];
  const int m = std::min<int>(n, e->cfg.max_batch);
  try {
    // 원본: 장치면 그대로, 호스트면 장치 버퍼로(처음·커질 때만 잡음)
    const uint8_t* src = f->rgb;
    int64_t rs = f->row_stride;
    if (!f->on_device) {
      const size_t need = size_t(f->h) * size_t(f->w) * size_t(f->pix_stride);
      if (need > e->frame_cap) {
        cudaFree(e->frame_dev);
        ck(cudaMalloc(reinterpret_cast<void**>(&e->frame_dev), need), "frame");
        e->frame_cap = need;
      }
      ck(cudaMemcpy2DAsync(e->frame_dev, size_t(f->w) * f->pix_stride, f->rgb, size_t(f->row_stride), size_t(f->w) * f->pix_stride,
                           size_t(f->h), cudaMemcpyHostToDevice, e->stream),
         "frame upload");
      src = e->frame_dev;
      rs = int64_t(f->w) * f->pix_stride;
    }
    MaskGeom mg{f->mask_w, f->mask_h, (f->mask_w * f->mask_h + 31) / 32, f->mask_sx, f->mask_sy, f->mask_ox, f->mask_oy};
    const bool have_mask = f->mask_bits && f->mask_w > 0 && f->mask_h > 0;
    const size_t words = have_mask ? size_t(mg.words) * m : 1;
    if (words > s.bits_cap) {
      cudaFree(s.bits_dev);
      cudaFreeHost(s.bits_host);
      ck(cudaMalloc(reinterpret_cast<void**>(&s.bits_dev), words * 4), "bits");
      ck(cudaHostAlloc(reinterpret_cast<void**>(&s.bits_host), words * 4, cudaHostAllocDefault), "bits host");
      s.bits_cap = words;
    }
    for (int i = 0; i < m; ++i) {
      CropJob& j = s.jobs_host[i];
      squareBox(items[i].box, e->cfg.margin, &j.bx, &j.by, &j.side);
      if (have_mask && items[i].det >= 0) {
        j.mask_word0 = i * mg.words;
        std::memcpy(s.bits_host + size_t(i) * mg.words, f->mask_bits + size_t(items[i].det) * mg.words, size_t(mg.words) * 4);
      } else {
        j.mask_word0 = -1;
      }
      s.ids[i] = items[i].id;
      s.q[i] = items[i].quality;
    }
    ck(cudaEventRecord(s.e0, e->stream), "ev0");
    ck(cudaMemcpyAsync(s.jobs_dev, s.jobs_host, sizeof(CropJob) * m, cudaMemcpyHostToDevice, e->stream), "jobs up");
    if (have_mask) ck(cudaMemcpyAsync(s.bits_dev, s.bits_host, words * 4, cudaMemcpyHostToDevice, e->stream), "bits up");
    launchCrop(src, rs, f->pix_stride, f->w, f->h, s.jobs_dev, m, s.bits_dev, mg, e->S, e->G, e->in_half, s.img, s.wp, e->stream);
    ck(cudaGetLastError(), "crop kernel");
    ck(cudaEventRecord(s.e_crop, e->stream), "ev crop");
    runNet(e, si, m);
    ck(cudaMemcpyAsync(s.emb_host, s.emb, size_t(m) * e->D * (e->out_half ? 2 : 4), cudaMemcpyDeviceToHost, e->stream), "emb down");
    ck(cudaEventRecord(s.e_done, e->stream), "ev done");
    // 원본은 호출자 것 — 자르기 커널이 다 읽을 때까지만 기다림(엔진은 계속 돎)
    ck(cudaEventSynchronize(s.e_crop), "crop wait");
  } catch (const std::exception& x) {
    std::fprintf(stderr, "[sgclip] submit: %s\n", x.what());
    return -1;
  }
  s.n = m;
  s.last_n = m;
  s.busy = true;
  s.stamp = stamp;
  e->last_slot = si;
  e->next = (si + 1) % 2;
  e->tm.n_submitted += m;
  e->tm.submit_us = float(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return m;
}

int32_t sgc_poll(sgc_encoder* e, sgc_result* out, int32_t cap, int32_t wait) {
  if (!e) return -1;
  // 앞 poll 이 넘긴 자리 돌려받기(그 포인터는 여기까지 유효)
  for (int r : e->lent) e->free_rows.push_back(r);
  e->lent.clear();
  for (int k = 0; k < 2; ++k) {
    const int c = (e->next + k) % 2;   // 먼저 넣은 칸부터
    Slot& s = e->slot[c];
    if (!s.busy || s.n == 0) continue;
    if (wait) cudaEventSynchronize(s.e_done);
    if (cudaEventQuery(s.e_done) == cudaSuccess) collect(e, c);
  }
  const int n = out ? std::min<int>(cap, int(e->pend.size())) : 0;
  for (int i = 0; i < n; ++i) {
    const auto& p = e->pend[i];
    out[i] = sgc_result{p.id, p.q, p.stamp, e->pool.data() + size_t(p.row) * e->D};
    e->lent.push_back(p.row);
  }
  e->pend.erase(e->pend.begin(), e->pend.begin() + n);
  return n;
}

int32_t sgc_pending(const sgc_encoder* e) {
  if (!e) return 0;
  int n = int(e->pend.size());
  for (const Slot& s : e->slot) n += s.busy ? s.n : 0;
  return n;
}

void sgc_get_timing(const sgc_encoder* e, sgc_timing* t) {
  if (e && t) *t = e->tm;
}

int32_t sgc_debug_inputs(sgc_encoder* e, float* images, float* wpatch, int32_t cap) {
  if (!e || e->last_slot < 0) return 0;
  Slot& s = e->slot[e->last_slot];
  cudaStreamSynchronize(e->stream);
  const int n = std::min<int>(cap, s.last_n);
  const size_t px = size_t(3) * e->S * e->S;
  if (images) {
    if (e->in_half) {
      std::vector<uint16_t> h(px * n);
      cudaMemcpy(h.data(), s.img, h.size() * 2, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < h.size(); ++i) images[i] = h2f(h[i]);
    } else {
      cudaMemcpy(images, s.img, px * n * 4, cudaMemcpyDeviceToHost);
    }
  }
  if (wpatch) cudaMemcpy(wpatch, s.wp, size_t(n) * e->G * e->G * 4, cudaMemcpyDeviceToHost);
  return n;
}

void sgc_f32_to_f16(const float* in, uint16_t* out, int32_t n) {
  for (int i = 0; i < n; ++i) out[i] = f2h(in[i]);
}
void sgc_f16_to_f32(const uint16_t* in, float* out, int32_t n) {
  for (int i = 0; i < n; ++i) out[i] = h2f(in[i]);
}

}  // extern "C"
