// ovdet: open-vocabulary detector (TensorRT YOLOE segmentation head + hand-written CUDA around it). API: include/ovdet.h.
#include "ovdet.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kMaxLanes = 128, kK1 = 1024, kSortCap = 4096, kProtoC = 32;
constexpr unsigned kFull = 0xFFFFFFFFu;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// ------------------------------------------------------------------------------------------------------ kernels
__global__ void k_fill(float* p, float v, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

struct Letter {
  int Sh, Sw, top, left, nw, nh;  // canvas size, pad, resized size
  float r;                    // resize ratio
};

/// cv2.resize INTER_LINEAR source index (half-pixel centres) of the Ultralytics LetterBox.
__device__ inline void lin(int d, float scale, int in, int& i0, int& i1, float& f) {
  float s = (d + 0.5f) * scale - 0.5f;
  if (s < 0.f) s = 0.f;
  i0 = min((int)floorf(s), in - 1);
  i1 = min(i0 + 1, in - 1);
  f = s - (float)i0;
  if (f < 0.f) f = 0.f;
}

__global__ void k_letterbox(const uint8_t* img, int h, int w, long long rs, int ps, int bgr, Letter L, float* canvas) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= L.nw || y >= L.nh) return;
  int x0, x1, y0, y1;
  float fx, fy;
  lin(x, (float)w / L.nw, w, x0, x1, fx);
  lin(y, (float)h / L.nh, h, y0, y1, fy);
  for (int c = 0; c < 3; ++c) {
    const int ch = bgr ? 2 - c : c;
    auto px = [&](int yy, int xx) { return (float)img[yy * rs + (long long)xx * ps + ch]; };
    const float v = (px(y0, x0) * (1.f - fx) + px(y0, x1) * fx) * (1.f - fy) + (px(y1, x0) * (1.f - fx) + px(y1, x1) * fx) * fy;
    canvas[(size_t)c * L.Sh * L.Sw + (size_t)(y + L.top) * L.Sw + x + L.left] = v / 255.f;
  }
}

__global__ void k_class_max(const float* o0, int A, int nc, const uint8_t* active, float* conf, int* cls) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= A) return;
  float best = 0.f;
  int bi = -1;
  for (int c = 0; c < nc; ++c) {
    if (active && !active[c]) continue;
    const float v = o0[(size_t)(4 + c) * A + a];
    if (bi < 0 || v > best) {
      best = v;
      bi = c;
    }
  }
  conf[a] = best;
  cls[a] = bi;
}

__device__ inline unsigned long long selKey(float c, int i) {
  return ((unsigned long long)(0xFFFFFFFFu - __float_as_uint(c)) << 32) | (unsigned)i;
}

__device__ void bitonic(unsigned long long* a, int P) {
  for (int k = 2; k <= P; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const unsigned long long x = a[i], y = a[ixj];
          if ((x > y) == ((i & k) == 0)) {
            a[i] = y;
            a[ixj] = x;
          }
        }
      }
      __syncthreads();
    }
}

struct Lanes {
  int anchor[kMaxLanes], cls[kMaxLanes], valid[kMaxLanes], area[kMaxLanes], stats[4];
  float conf[kMaxLanes], box[kMaxLanes * 4];
};

/// Candidates above th -> score order -> greedy NMS (iterated to its fixed point) -> first max_det kept.
__global__ void __launch_bounds__(1024) k_select(const float* o0, int A, const float* conf, const int* cls, float th,
                                                 float nms_iou, int agnostic, int max_det,
                                                 unsigned long long* gkeys, uint32_t* supT, Lanes* L) {
  __shared__ unsigned long long sm[kSortCap];
  __shared__ int s_cnt, s_pos, s_changed;
  __shared__ uint32_t keepw[32];
  const int tid = threadIdx.x;
  if (tid == 0) s_cnt = s_pos = 0;
  __syncthreads();
  int local = 0;
  for (int i = tid; i < A; i += blockDim.x) local += conf[i] > th;
  atomicAdd(&s_cnt, local);
  __syncthreads();
  const int n = s_cnt;
  unsigned long long* keys = n <= kSortCap ? sm : gkeys;
  int P = 1;
  while (P < n) P <<= 1;
  for (int i = tid; i < A; i += blockDim.x)
    if (conf[i] > th) keys[atomicAdd(&s_pos, 1)] = selKey(conf[i], i);
  for (int i = n + tid; i < P; i += blockDim.x) keys[i] = ~0ull;
  __syncthreads();
  if (n > 1) bitonic(keys, P);
  const int n1 = min(n, kK1);
  int anchor = -1, c_id = 0;
  float cf = 0.f, x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  if (tid < n1) {
    anchor = (int)(keys[tid] & 0xFFFFFFFFull);
    cf = conf[anchor];
    c_id = cls ? cls[anchor] : 0;
  }
  __syncthreads();
  float* bx = reinterpret_cast<float*>(sm);  // x1 y1 x2 y2 area, 5 x 1024
  int* bc = reinterpret_cast<int*>(bx + 5 * kK1);
  if (tid < n1) {
    const float cx = o0[anchor], cy = o0[A + anchor], w = o0[2 * A + anchor], h = o0[3 * A + anchor];
    x1 = cx - w / 2.f;
    y1 = cy - h / 2.f;
    x2 = cx + w / 2.f;
    y2 = cy + h / 2.f;
    bx[tid] = x1;
    bx[kK1 + tid] = y1;
    bx[2 * kK1 + tid] = x2;
    bx[3 * kK1 + tid] = y2;
    bx[4 * kK1 + tid] = fmaxf(x2 - x1, 0.f) * fmaxf(y2 - y1, 0.f);
    bc[tid] = c_id;
  }
  __syncthreads();
  if (tid < n1) {
    for (int wi = 0; wi <= (tid >> 5); ++wi) {
      uint32_t bits = 0;
      for (int b = 0; b < 32; ++b) {
        const int i = wi * 32 + b;
        if (i >= tid) break;
        if (!agnostic && bc[i] != c_id) continue;
        const float inter = fmaxf(fminf(bx[2 * kK1 + i], x2) - fmaxf(bx[i], x1), 0.f) *
                            fmaxf(fminf(bx[3 * kK1 + i], y2) - fmaxf(bx[kK1 + i], y1), 0.f);
        if (inter / ((bx[4 * kK1 + i] + bx[4 * kK1 + tid]) - inter + 1e-9f) > nms_iou) bits |= 1u << b;
      }
      supT[tid * 32 + wi] = bits;
    }
  }
  {
    const unsigned b = __ballot_sync(kFull, tid < n1);
    if ((tid & 31) == 0) keepw[tid >> 5] = b;
  }
  __syncthreads();
  for (int it = 0; it < n1 + 1; ++it) {  // Jacobi rounds until nothing changes = greedy NMS
    bool nk = false;
    if (tid < n1) {
      nk = true;
      for (int wi = 0; wi <= (tid >> 5); ++wi)
        if (supT[tid * 32 + wi] & keepw[wi]) {
          nk = false;
          break;
        }
    }
    if (tid == 0) s_changed = 0;
    __syncthreads();
    const unsigned b = __ballot_sync(kFull, nk);
    if ((tid & 31) == 0) {
      if (keepw[tid >> 5] != b) atomicExch(&s_changed, 1);
      keepw[tid >> 5] = b;
    }
    __syncthreads();
    if (!s_changed) break;
  }
  const bool kept = (keepw[tid >> 5] >> (tid & 31)) & 1u;
  int rank = __popc(keepw[tid >> 5] & ((1u << (tid & 31)) - 1u)), total = 0;
  for (int w = 0; w < 32; ++w) {
    const int c = __popc(keepw[w]);
    if (w < (tid >> 5)) rank += c;
    total += c;
  }
  const int nl = min(total, max_det);
  if (kept && rank < nl) {
    L->anchor[rank] = anchor;
    L->conf[rank] = cf;
    L->cls[rank] = c_id;
    L->box[4 * rank] = x1;
    L->box[4 * rank + 1] = y1;
    L->box[4 * rank + 2] = x2;
    L->box[4 * rank + 3] = y2;
    L->valid[rank] = 1;
  }
  if (tid >= nl && tid < kMaxLanes) {
    L->anchor[tid] = -1;
    L->valid[tid] = 0;
    L->conf[tid] = 0.f;
  }
  if (tid == 0) {
    L->stats[0] = n;
    L->stats[1] = total;
  }
}

/// Mask bits of every valid lane on the prototype grid (Ph x Pw): logit > 0, inside the box (Ultralytics crop_mask:
/// integer cell index in [x1, x2) with the box scaled to the grid) and inside the image region (letterbox pad cut).
__global__ void __launch_bounds__(256) k_bits(const float* o0, int A, int coef_row, const float* o1, int Ph, int Pw, float g,
                                              Lanes* L, int rt, int rb, int rl, int rr, uint32_t* bits) {
  __shared__ float co[kMaxLanes][kProtoC];
  __shared__ int sv[kMaxLanes];
  __shared__ float sb[kMaxLanes][4];
  for (int i = threadIdx.x; i < kMaxLanes * kProtoC; i += blockDim.x) {
    const int l = i / kProtoC, k = i % kProtoC;
    co[l][k] = (L->valid[l] && L->anchor[l] >= 0) ? o0[(size_t)(coef_row + k) * A + L->anchor[l]] : 0.f;
  }
  for (int i = threadIdx.x; i < kMaxLanes * 4; i += blockDim.x) sb[i / 4][i % 4] = L->box[i] * g;
  for (int i = threadIdx.x; i < kMaxLanes; i += blockDim.x) sv[i] = L->valid[i];
  __syncthreads();
  // 격자 칸 수가 256 의 배수가 아니어도 됨(416 입력: 104×104): 끝 블록의 남는 스레드는 0 비트로 ballot 에 참여
  const int P = Ph * Pw, p = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = p < P;
  const int y = live ? p / Pw : 0, x = live ? p % Pw : 0;
  float pr[kProtoC];
  for (int k = 0; k < kProtoC; ++k) pr[k] = live ? o1[(size_t)k * P + p] : 0.f;
  const bool inimg = live && y >= rt && y < rb && x >= rl && x < rr;
  const float fx = (float)x, fy = (float)y;
  for (int l = 0; l < kMaxLanes; ++l) {
    if (!sv[l]) continue;
    float v = 0.f;
    for (int k = 0; k < kProtoC; ++k) v = fmaf(co[l][k], pr[k], v);
    const bool b = v > 0.f && inimg && fx >= sb[l][0] && fx < sb[l][2] && fy >= sb[l][1] && fy < sb[l][3];
    const unsigned w = __ballot_sync(kFull, b);
    if ((threadIdx.x & 31) == 0 && (p >> 5) < (P + 31) / 32) bits[(size_t)l * ((P + 31) / 32) + (p >> 5)] = w;
  }
}

template <class T>
__device__ T warpSum(T v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFull, v, o);
  return v;
}

__global__ void k_area(const uint32_t* bits, int words, Lanes* L, int area_min, int small_area, float small_conf) {
  __shared__ int sm[32];
  const int l = blockIdx.x;
  if (!L->valid[l]) return;
  int s = 0;
  for (int w = threadIdx.x; w < words; w += blockDim.x) s += __popc(bits[(size_t)l * words + w]);
  s = warpSum(s);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    int t = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += sm[i];
    L->area[l] = t;
    L->valid[l] = t >= area_min && (t >= small_area || L->conf[l] >= small_conf);
  }
}

__global__ void k_pair(const uint32_t* bits, int words, const Lanes* L, int* inter) {
  const int i = blockIdx.x;
  if (!L->valid[i]) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int j = i + 1 + warp; j < kMaxLanes; j += nw) {
    if (!L->valid[j]) continue;
    int s = 0;
    for (int w = lane; w < words; w += 32) s += __popc(bits[(size_t)i * words + w] & bits[(size_t)j * words + w]);
    s = warpSum(s);
    if (lane == 0) inter[i * kMaxLanes + j] = inter[j * kMaxLanes + i] = s;
  }
}

/// Mask duplicates: IoU > mask_iou with a higher-scoring kept mask (iterated to its fixed point).
__global__ void k_dedup(Lanes* L, const int* inter, float mask_iou) {
  __shared__ uint32_t vw[kMaxLanes / 32];
  __shared__ int changed;
  const int j = threadIdx.x;
  bool v = L->valid[j] != 0;
  uint32_t col[kMaxLanes / 32] = {0, 0, 0, 0};
  for (int i = 0; i < kMaxLanes; ++i) {
    if (i == j || !L->valid[i] || !v) continue;
    const float in = (float)inter[i * kMaxLanes + j];
    const float iou = in / fmaxf((float)L->area[i] + (float)L->area[j] - in, 1.f);
    if (iou > mask_iou && L->conf[i] > L->conf[j]) col[i >> 5] |= 1u << (i & 31);
  }
  {
    const unsigned b = __ballot_sync(kFull, v);
    if ((j & 31) == 0) vw[j >> 5] = b;
  }
  __syncthreads();
  for (int it = 0; it < kMaxLanes; ++it) {
    const bool nv = v && !((col[0] & vw[0]) | (col[1] & vw[1]) | (col[2] & vw[2]) | (col[3] & vw[3]));
    if (j == 0) changed = 0;
    __syncthreads();
    const unsigned b = __ballot_sync(kFull, nv);
    if ((j & 31) == 0) {
      if (vw[j >> 5] != b) atomicExch(&changed, 1);
      vw[j >> 5] = b;
    }
    __syncthreads();
    v = nv;
    if (!changed) break;
  }
  L->valid[j] = v;
}

/// Output lanes on the device (no host round trip): valid, with a class that is in the prompt (active), in lane order.
/// Their count goes to L->stats[2]; k_gather copies their mask bits to the front of out.
__global__ void k_compact(Lanes* L, const uint8_t* active, int* lanes) {
  __shared__ int wc[kMaxLanes / 32];
  const int l = threadIdx.x;
  const bool k = L->valid[l] && L->cls[l] >= 0 && active[L->cls[l]];
  const unsigned b = __ballot_sync(kFull, k);
  if ((l & 31) == 0) wc[l >> 5] = __popc(b);
  __syncthreads();
  int r = __popc(b & ((1u << (l & 31)) - 1u));
  for (int w = 0; w < (l >> 5); ++w) r += wc[w];
  if (k) lanes[r] = l;
  if (l == 0) L->stats[2] = wc[0] + wc[1] + wc[2] + wc[3];
}

__global__ void k_gather(const uint32_t* bits, int words, const int* lanes, const Lanes* L, uint32_t* out) {
  const int o = blockIdx.y, w = blockIdx.x * blockDim.x + threadIdx.x;
  if (o < L->stats[2] && w < words) out[(size_t)o * words + w] = bits[(size_t)lanes[o] * words + w];
}

// ------------------------------------------------------------------------------------------------------ host side
struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kERROR) std::fprintf(stderr, "[ovdet trt] %s\n", m);
  }
};
Logger g_logger;

struct Engine {
  std::unique_ptr<nvinfer1::IRuntime> rt;
  std::unique_ptr<nvinfer1::ICudaEngine> eng;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  explicit Engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("engine not found: " + path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    rt.reset(nvinfer1::createInferRuntime(g_logger));
    eng.reset(rt->deserializeCudaEngine(blob.data(), blob.size()));
    if (!eng) throw std::runtime_error("cannot deserialize " + path);
    ctx.reset(eng->createExecutionContext());
  }
  std::vector<long> shape(const char* n) const {
    const auto d = eng->getTensorShape(n);
    return std::vector<long>(d.d, d.d + d.nbDims);
  }
};

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  void alloc(size_t k) {
    if (p) cudaFree(p);
    n = k;
    p = nullptr;
    if (k) check(cudaMalloc(reinterpret_cast<void**>(&p), k * sizeof(T)), "cudaMalloc");
  }
  ~Dev() { if (p) cudaFree(p); }
};

std::vector<std::string> readLines(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<std::string> v;
  for (std::string l; std::getline(f, l);) {
    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
    if (!l.empty()) v.push_back(l);
  }
  return v;
}

/// Vocabulary key of a name: ".n.NN" dropped, '_' -> ' ', lower case, single spaces.
std::string normName(const std::string& in) {
  std::string t = in;
  const auto k = t.find(".n.");
  if (k != std::string::npos) t = t.substr(0, k);
  std::string o;
  for (char c : t) {
    c = (c == '_') ? ' ' : (char)std::tolower((unsigned char)c);
    if (c == ' ' && (o.empty() || o.back() == ' ')) continue;
    o += c;
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  return o;
}

template <class T>
struct Pinned {
  T* p = nullptr;
  void alloc(size_t k) {
    if (p) cudaFreeHost(p);
    p = nullptr;
    if (k) check(cudaMallocHost(reinterpret_cast<void**>(&p), k * sizeof(T)), "cudaMallocHost");
  }
  ~Pinned() { if (p) cudaFreeHost(p); }
};

}  // namespace

struct OvdHandle {
  OvdConfig cfg{};
  std::string err;
  cudaStream_t s = nullptr;
  std::unique_ptr<Engine> seg;
  std::vector<std::string> vocab, vkey;  // vocabulary names and their keys
  int Sh = 0, Sw = 0, A = 0, Ph = 0, Pw = 0, nc = 1, rows = 0, words = 0;
  Dev<float> canvas, o0, o1, conf;
  Dev<int> cls, inter, dlanes;
  Dev<unsigned long long> gkeys;
  Dev<uint32_t> supT, bits, obits;
  Dev<uint8_t> active, img;
  Dev<Lanes> lanes;
  std::vector<int> prompt_vocab;  // prompt index -> vocabulary index (-1: unknown)
  std::vector<int> vocab_prompt;  // vocabulary index -> first prompt index (-1: not in the prompt)
  // output (owned; valid until the next call)
  sm_detections det{};
  std::vector<int32_t> o_cls, o_area;
  std::vector<float> o_score, o_box;
  Pinned<uint32_t> h_bits;
  Pinned<Lanes> h_lanes;
  Pinned<uint8_t> h_img;  // pinned staging of a host image: a pageable H2D copy is staged by the driver (slow)
  size_t img_cap = 0, h_img_cap = 0;
  int last_h = -1, last_w = -1;
  int64_t bytes = 0;
  cudaEvent_t ev[5];
  // CUDA graph of one detection (letterbox → TensorRT → post-processing → output copies), keyed by the input layout.
  // OVDET_NO_GRAPH=1 runs the same work stream-ordered (no capture).
  bool use_graph = true;
  cudaGraphExec_t gexec = nullptr;
  int g_h = -1, g_w = -1, g_ps = -1, g_bgr = -1;
  const uint8_t* g_img = nullptr;
  long long g_rs = -1;
  void dropGraph() {
    if (gexec) cudaGraphExecDestroy(gexec);
    gexec = nullptr;
  }
};

extern "C" {

void ovd_default_config(OvdConfig* c) {
  std::memset(c, 0, sizeof(*c));
  c->conf_th = 0.25f;
  c->nms_iou = 0.7f;
  c->mask_iou = 0.7f;
  c->area_min = 24;
  c->small_area = 256;
  c->small_conf = 0.f;
  c->max_det = 100;
  c->class_agnostic = 1;
}

OvdHandle* ovd_create(const OvdConfig* cfg, char* err, size_t err_len) {
  auto h = std::make_unique<OvdHandle>();
  try {
    h->cfg = *cfg;
    if (!cfg->seg_engine) throw std::runtime_error("seg_engine is required");
    if (h->cfg.max_det <= 0 || h->cfg.max_det > kMaxLanes) h->cfg.max_det = kMaxLanes;
    check(cudaSetDevice(cfg->device), "cudaSetDevice");
    size_t free0 = 0, free1 = 0, total = 0;
    check(cudaFree(nullptr), "context");
    cudaMemGetInfo(&free0, &total);
    check(cudaStreamCreateWithFlags(&h->s, cudaStreamNonBlocking), "stream");
    for (auto& e : h->ev) check(cudaEventCreate(&e), "event");
    h->seg = std::make_unique<Engine>(cfg->seg_engine);
    const auto si = h->seg->shape("images"), s0 = h->seg->shape("output0"), s1 = h->seg->shape("output1");
    if (si.size() != 4 || s0.size() != 3 || s1.size() != 4 || s1[1] != kProtoC || si[2] * s1[3] != si[3] * s1[2])
      throw std::runtime_error("unexpected engine I/O (want images 1x3xHxW, output0 1xRxA, output1 1x32xhxw, same aspect)");
    h->Sh = (int)si[2];
    h->Sw = (int)si[3];
    h->rows = (int)s0[1];
    h->A = (int)s0[2];
    h->Ph = (int)s1[2];
    h->Pw = (int)s1[3];
    h->nc = h->rows - 4 - kProtoC;
    h->words = (h->Ph * h->Pw + 31) / 32;   // sm_detections 약속: ceil(mask_w·mask_h / 32) 워드
    if (h->nc < 1) throw std::runtime_error("unsupported head shape");
    if (!cfg->names) throw std::runtime_error("the names file (<engine>.names.txt) is required");
    h->vocab = readLines(cfg->names);
    if ((int)h->vocab.size() != h->nc)
      throw std::runtime_error("names file has " + std::to_string(h->vocab.size()) + " lines, the head " +
                               std::to_string(h->nc) + " classes");
    h->active.alloc(h->nc);
    for (const auto& v : h->vocab) h->vkey.push_back(normName(v));
    h->canvas.alloc((size_t)3 * h->Sh * h->Sw);
    h->o0.alloc((size_t)h->rows * h->A);
    h->o1.alloc((size_t)kProtoC * h->Ph * h->Pw);
    h->conf.alloc(h->A);
    h->cls.alloc(h->A);
    h->gkeys.alloc((size_t)1 << 17);
    h->supT.alloc((size_t)kK1 * 32);
    h->lanes.alloc(1);
    h->bits.alloc((size_t)kMaxLanes * h->words);
    h->obits.alloc((size_t)kMaxLanes * h->words);
    h->inter.alloc((size_t)kMaxLanes * kMaxLanes);
    h->dlanes.alloc(kMaxLanes);
    h->h_bits.alloc((size_t)kMaxLanes * h->words);
    h->h_lanes.alloc(1);
    auto* ctx = h->seg->ctx.get();
    ctx->setTensorAddress("images", h->canvas.p);
    ctx->setTensorAddress("output0", h->o0.p);
    ctx->setTensorAddress("output1", h->o1.p);
    ovd_set_prompt(h.get(), nullptr, 0, nullptr, 0);
    k_fill<<<(unsigned)((h->canvas.n + 255) / 256), 256, 0, h->s>>>(h->canvas.p, 114.f / 255.f, h->canvas.n);
    if (!ctx->enqueueV3(h->s)) throw std::runtime_error("warm-up enqueue failed");
    check(cudaStreamSynchronize(h->s), "warm-up");
    cudaMemGetInfo(&free1, &total);
    h->bytes = (int64_t)free0 - (int64_t)free1;
    if (const char* e = std::getenv("OVDET_NO_GRAPH")) h->use_graph = !(e[0] && e[0] != '0');
    return h.release();
  } catch (const std::exception& e) {
    if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
    return nullptr;
  }
}

void ovd_destroy(OvdHandle* h) {
  if (!h) return;
  cudaStreamSynchronize(h->s);
  h->dropGraph();
  h->seg.reset();
  for (auto& e : h->ev) cudaEventDestroy(e);
  cudaStreamDestroy(h->s);
  delete h;
}

int32_t ovd_vocab_size(const OvdHandle* h) { return (int32_t)h->vocab.size(); }
const char* ovd_vocab_name(const OvdHandle* h, int32_t i) {
  return i >= 0 && i < (int32_t)h->vocab.size() ? h->vocab[i].c_str() : "";
}
int64_t ovd_device_bytes(const OvdHandle* h) { return h->bytes; }
const char* ovd_last_error(const OvdHandle* h) { return h->err.c_str(); }
const int32_t* ovd_last_areas(const OvdHandle* h) { return h->o_area.data(); }

int32_t ovd_set_prompt(OvdHandle* h, const char* const* names, int32_t n, char* err, size_t err_len) {
  const int V = (int)h->vocab.size();
  std::vector<std::string> want;
  if (names && n > 0)
    for (int i = 0; i < n; ++i) want.push_back(names[i] ? names[i] : "");
  else
    want = h->vocab;  // no prompt: the whole vocabulary, in its order
  h->prompt_vocab.assign(want.size(), -1);
  h->vocab_prompt.assign(V, -1);
  std::string miss;
  int found = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const std::string k = normName(want[i]);
    const auto it = std::find(h->vkey.begin(), h->vkey.end(), k);
    if (it == h->vkey.end()) {
      miss += (miss.empty() ? "" : ", ") + want[i];
      continue;
    }
    const int v = (int)(it - h->vkey.begin());
    h->prompt_vocab[i] = v;
    if (h->vocab_prompt[v] < 0) h->vocab_prompt[v] = (int)i;
    ++found;
  }
  {
    std::vector<uint8_t> m(V);
    for (int v = 0; v < V; ++v) m[v] = h->vocab_prompt[v] >= 0;
    check(cudaMemcpy(h->active.p, m.data(), m.size(), cudaMemcpyHostToDevice), "active");
  }
  if (err && err_len) std::snprintf(err, err_len, "%s", miss.empty() ? "" : ("not in the vocabulary: " + miss).c_str());
  return found;
}

const sm_detections* ovd_detect(OvdHandle* h, const OvdImage* im, OvdTiming* timing) {
  try {
    const OvdConfig& c = h->cfg;
    cudaStream_t s = h->s;
    const int H = im->h, W = im->w;
    if (H <= 0 || W <= 0 || (im->pix_stride != 3 && im->pix_stride != 4)) throw std::runtime_error("bad image");
    cudaEventRecord(h->ev[0], s);
    // ---- upload (packed on device). With the graph a device image is copied too, so the graph always reads h->img.
    const uint8_t* dimg = im->data;
    long long rs = im->row_stride;
    if (!im->on_device || h->use_graph) {
      const size_t need = (size_t)H * W * im->pix_stride;
      if (need > h->img_cap) {
        h->img.alloc(need);
        h->img_cap = need;
      }
      const size_t row = (size_t)W * im->pix_stride;
      if (im->on_device) {
        check(cudaMemcpy2DAsync(h->img.p, row, im->data, im->row_stride, row, H, cudaMemcpyDeviceToDevice, s), "upload");
      } else {
        if (need > h->h_img_cap) {
          h->h_img.alloc(need);
          h->h_img_cap = need;
        }
        if ((size_t)im->row_stride == row)
          std::memcpy(h->h_img.p, im->data, need);
        else
          for (int y = 0; y < H; ++y) std::memcpy(h->h_img.p + y * row, im->data + (size_t)y * im->row_stride, row);
        check(cudaMemcpyAsync(h->img.p, h->h_img.p, need, cudaMemcpyHostToDevice, s), "upload");
      }
      dimg = h->img.p;
      rs = (long long)W * im->pix_stride;
    }
    Letter L;
    L.Sh = h->Sh;
    L.Sw = h->Sw;
    L.r = std::min((float)h->Sh / H, (float)h->Sw / W);
    L.nw = (int)std::lround(W * L.r);
    L.nh = (int)std::lround(H * L.r);
    const float dw = (h->Sw - L.nw) / 2.f, dh = (h->Sh - L.nh) / 2.f;
    L.top = (int)std::lround(dh - 0.1f);
    L.left = (int)std::lround(dw - 0.1f);
    if (H != h->last_h || W != h->last_w) {
      k_fill<<<(unsigned)((h->canvas.n + 255) / 256), 256, 0, s>>>(h->canvas.p, 114.f / 255.f, h->canvas.n);
      h->last_h = H;
      h->last_w = W;
    }
    const float g = (float)h->Pw / h->Sw;  // canvas pixel -> grid cell
    const int rt = (int)std::floor(L.top * g), rb = (int)std::ceil((L.top + L.nh) * g);
    const int rl = (int)std::floor(L.left * g), rr = (int)std::ceil((L.left + L.nw) * g);
    const int ncopy = std::min(c.max_det, kMaxLanes);  // at most max_det lanes are ever valid
    // ---- letterbox → network → candidates, NMS, mask bits, area gate, mask duplicates → output lanes → copies
    // inside a capture the event needs cudaEventRecordExternal to become a record node (timing); outside, a plain record
    auto rec = [&](int k, cudaStream_t q) {
      if (h->use_graph) cudaEventRecordWithFlags(h->ev[k], q, cudaEventRecordExternal);
      else cudaEventRecord(h->ev[k], q);
    };
    auto body = [&](cudaStream_t q) {
      k_letterbox<<<dim3((L.nw + 255) / 256, L.nh), 256, 0, q>>>(dimg, H, W, rs, im->pix_stride, im->bgr, L, h->canvas.p);
      rec(1, q);
      if (!h->seg->ctx->enqueueV3(q)) throw std::runtime_error("enqueue failed");
      rec(2, q);
      k_class_max<<<(h->A + 255) / 256, 256, 0, q>>>(h->o0.p, h->A, h->nc, h->active.p, h->conf.p, h->cls.p);
      cudaMemsetAsync(h->lanes.p, 0, sizeof(Lanes), q);
      k_select<<<1, 1024, 0, q>>>(h->o0.p, h->A, h->conf.p, h->cls.p, c.conf_th, c.nms_iou, c.class_agnostic, c.max_det,
                                  h->gkeys.p, h->supT.p, h->lanes.p);
      k_bits<<<(h->Ph * h->Pw + 255) / 256, 256, 0, q>>>(h->o0.p, h->A, 4 + h->nc, h->o1.p, h->Ph, h->Pw, g, h->lanes.p, rt,
                                                         rb, rl, rr, h->bits.p);
      k_area<<<kMaxLanes, 256, 0, q>>>(h->bits.p, h->words, h->lanes.p, c.area_min, c.small_area, c.small_conf);
      if (c.mask_iou > 0) {
        cudaMemsetAsync(h->inter.p, 0, sizeof(int) * kMaxLanes * kMaxLanes, q);
        k_pair<<<kMaxLanes, 256, 0, q>>>(h->bits.p, h->words, h->lanes.p, h->inter.p);
        k_dedup<<<1, kMaxLanes, 0, q>>>(h->lanes.p, h->inter.p, c.mask_iou);
      }
      k_compact<<<1, kMaxLanes, 0, q>>>(h->lanes.p, h->active.p, h->dlanes.p);
      k_gather<<<dim3((h->words + 255) / 256, ncopy), 256, 0, q>>>(h->bits.p, h->words, h->dlanes.p, h->lanes.p, h->obits.p);
      rec(3, q);
      check(cudaMemcpyAsync(h->h_lanes.p, h->lanes.p, sizeof(Lanes), cudaMemcpyDeviceToHost, q), "lanes");
      check(cudaMemcpyAsync(h->h_bits.p, h->obits.p, (size_t)ncopy * h->words * 4, cudaMemcpyDeviceToHost, q), "bits");
    };
    if (h->use_graph) {
      if (h->gexec && (H != h->g_h || W != h->g_w || im->pix_stride != h->g_ps || im->bgr != h->g_bgr || dimg != h->g_img ||
                       rs != h->g_rs))
        h->dropGraph();
      if (!h->gexec) {
        cudaGraph_t gr = nullptr;
        check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture begin");
        try {
          body(s);
        } catch (...) {
          cudaStreamEndCapture(s, &gr);
          if (gr) cudaGraphDestroy(gr);
          throw;
        }
        check(cudaStreamEndCapture(s, &gr), "capture end");
        const cudaError_t e = cudaGraphInstantiate(&h->gexec, gr, 0);
        cudaGraphDestroy(gr);
        check(e, "graph instantiate");
        h->g_h = H;
        h->g_w = W;
        h->g_ps = im->pix_stride;
        h->g_bgr = im->bgr;
        h->g_img = dimg;
        h->g_rs = rs;
      }
      check(cudaGraphLaunch(h->gexec, s), "graph launch");
    } else {
      body(s);
    }
    cudaEventRecord(h->ev[4], s);
    check(cudaEventSynchronize(h->ev[4]), "done");
    const Lanes& hl = *h->h_lanes.p;
    std::vector<int> keep;
    for (int l = 0; l < kMaxLanes; ++l)
      if (hl.valid[l] && hl.cls[l] >= 0 && h->vocab_prompt[hl.cls[l]] >= 0) keep.push_back(l);
    const int n = (int)keep.size();
    if (n != hl.stats[2]) throw std::runtime_error("output lane count mismatch (prompt changed during detect?)");
    // boxes in input pixels; grid cell -> input pixel: x = cell * sx + ox
    const float sx = 1.f / (g * L.r), ox = -L.left / L.r, oy = -L.top / L.r;
    std::vector<float> box((size_t)n * 4);
    h->o_cls.assign(n, -1);
    h->o_score.assign(n, 0.f);
    h->o_area.assign(n, 0);
    for (int i = 0; i < n; ++i) {
      const int l = keep[i];
      const float* b = hl.box + 4 * l;
      box[4 * i] = std::clamp((b[0] - L.left) / L.r, 0.f, (float)W);
      box[4 * i + 1] = std::clamp((b[1] - L.top) / L.r, 0.f, (float)H);
      box[4 * i + 2] = std::clamp((b[2] - L.left) / L.r, 0.f, (float)W);
      box[4 * i + 3] = std::clamp((b[3] - L.top) / L.r, 0.f, (float)H);
      h->o_score[i] = hl.conf[l];
      h->o_area[i] = hl.area[l];
      h->o_cls[i] = h->vocab_prompt[hl.cls[l]];
    }
    h->o_box.swap(box);
    sm_detections& d = h->det;
    d.stamp = im->stamp;
    d.cam = im->cam;
    d.img_w = W;
    d.img_h = H;
    d.n = n;
    d.cls = h->o_cls.data();
    d.score = h->o_score.data();
    d.box = h->o_box.data();
    d.mask_w = h->Pw;
    d.mask_h = h->Ph;
    d.mask_sx = d.mask_sy = sx;
    d.mask_ox = ox;
    d.mask_oy = oy;
    d.mask_bits = h->h_bits.p;
    if (timing) {
      auto ms = [&](int a, int b) {
        float v = 0;
        cudaEventElapsedTime(&v, h->ev[a], h->ev[b]);
        return v;
      };
      timing->upload_ms = ms(0, 1);
      timing->net_ms = ms(1, 2);
      timing->post_ms = ms(2, 3);
      timing->out_ms = ms(3, 4);
      timing->total_ms = ms(0, 4);
      timing->candidates = hl.stats[0];
    }
    return &h->det;
  } catch (const std::exception& e) {
    h->err = e.what();
    return nullptr;
  }
}

}  // extern "C"
