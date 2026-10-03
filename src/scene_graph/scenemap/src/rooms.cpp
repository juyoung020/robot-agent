// 방 나누기 구현(include/scenemap/rooms.hpp).
#include "scenemap/rooms.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <numeric>

namespace scenemap {
namespace {

constexpr float kInf = 1e20f;

// Felzenszwalb–Huttenlocher 1D 제곱 거리 변환(float — 나눗셈 없는 정수 판보다 이 기계에서 빠름, 잼)
void dt1d(const float* f, int n, float* d, int* v, float* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  auto isect = [&](int q, int r) { return ((f[q] + float(q) * q) - (f[r] + float(r) * r)) / (2.f * (q - r)); };
  for (int q = 1; q < n; ++q) {
    float s = isect(q, v[k]);
    while (s <= z[k] && k > 0) {
      --k;
      s = isect(q, v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const float dq = float(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

// 빠른 올림(std::ceil 은 SSE4.1 없는 기본 x86-64 에서 함수 부름)
inline int iceil(float x) {
  const int t = int(x);
  return t + (float(t) < x);
}

// 칸마다 쓰는 일감 버퍼 — 스레드마다 하나를 다시 써서 부를 때마다 수 MB 를 새로 잡지(쪽 잘못) 않는다.
// 방 추적기는 한 번에 하나만 나누므로 보통 스레드 하나 몫.
struct Work {
  std::vector<uint8_t> free, pc;
  std::vector<float> g, clear, d, z;
  std::vector<int> v, run, key, cnt, order, b, stack, comp;
};
Work& work() {
  thread_local Work w;
  return w;
}

// 작은 모름 구멍 → 빈칸, 빈칸에 둘러싸인 작은 점유 점 → 빈칸. 덧댄 격자(밖 = 3)로 범위 검사 없이
void cleanFree(Work& wk, int W, int H, int hole_cells, int speck_cells) {
  const int PW = W + 2;
  std::vector<uint8_t>& pc = wk.pc;   // 덧댄 분류(segmentCore 가 채움): 0 모름, 1 빈칸, 2 점유, 3 밖, |0x80 = 본 칸
  const int o4[4] = {-1, 1, -PW, PW}, od[4] = {-PW - 1, -PW + 1, PW - 1, PW + 1};
  std::vector<int>& stack = wk.stack;
  std::vector<int>& comp = wk.comp;
  for (int y = 1; y <= H; ++y)
    for (int x = 1; x <= W; ++x) {
      const int s0 = y * PW + x;
      const uint8_t want = pc[s0];
      if (want == 1 || want & 0x80) continue;
      const bool eight = want == 2;
      const int limit = want == 0 ? hole_cells : speck_cells;
      stack.assign(1, s0);
      pc[s0] |= 0x80;
      comp.clear();
      bool border = false, keep = limit > 0;
      int nfree = 0, nother = 0;
      while (!stack.empty()) {
        const int c = stack.back();
        stack.pop_back();
        if (keep) {
          comp.push_back(c);
          if (int(comp.size()) > limit) keep = false;
        }
        for (int k = 0; k < 4; ++k) {
          const int j = c + o4[k];
          const uint8_t v = pc[j] & 0x7f;
          if (v == want) {
            if (!(pc[j] & 0x80)) { pc[j] |= 0x80; stack.push_back(j); }
          } else if (v == 3) {
            border = true;
          } else if (v == 1) {
            ++nfree;
          } else {
            ++nother;
          }
        }
        if (eight)
          for (int k = 0; k < 4; ++k) {
            const int j = c + od[k];
            if (pc[j] == want) { pc[j] |= 0x80; stack.push_back(j); }
            else if ((pc[j] & 0x7f) == 3) border = true;
          }
      }
      if (!keep || border || comp.empty()) continue;
      const bool ok = want == 0 ? (nfree > 0 && 2 * nfree >= nfree + nother) : nother == 0;
      if (ok)
        for (int c : comp) wk.free[size_t(c / PW - 1) * W + (c % PW - 1)] = 1;
    }
}

// 빈칸의 벽 면까지 여유(m) clear 와 거름 층 통 key(층 L − Lmin, 막힌 칸 −1), 통마다 칸 수 cnt.
// 정확한 유클리드 거리: 세로(이진이라 위·아래 두 번 쓸기) + 가로 Felzenszwalb–Huttenlocher 아래 봉투, 격자 밖 = 막힘.
void clearance(Work& wk, int W, int H, double res, double dil_min, double step, int Lmin, int nbins) {
  const int PW = W + 2;
  const size_t N = size_t(W) * H;
  const uint8_t* free = wk.free.data();
  wk.g.resize(size_t(H) * PW);
  wk.run.assign(W, 0);
  int* run = wk.run.data();
  for (int y = 0; y < H; ++y) {   // 아래(−y) 쪽 거리
    const uint8_t* fr = free + size_t(y) * W;
    float* row = &wk.g[size_t(y) * PW];
    row[0] = row[PW - 1] = 0.f;   // 덧댄 칸 = 막힘
    for (int x = 0; x < W; ++x) {
      run[x] = fr[x] ? run[x] + 1 : 0;
      row[x + 1] = float(run[x]);
    }
  }
  wk.d.resize(PW);
  wk.z.resize(PW + 1);
  wk.v.resize(PW);
  wk.clear.resize(N);
  wk.key.resize(N);
  wk.cnt.assign(nbins, 0);
  int* cnt = wk.cnt.data();
  const int top = nbins - 1;
  const float fres = float(res), half = float(res / 2), fmin = float(dil_min), inv = float(1.0 / step);
  std::fill(wk.run.begin(), wk.run.end(), 0);
  for (int y = H - 1; y >= 0; --y) {   // 위(+y) 쪽과 작은 것, 제곱 → 이 줄은 끝났으니 바로 가로 봉투
    const uint8_t* fr = free + size_t(y) * W;
    float* row = &wk.g[size_t(y) * PW];
    bool any = false;
    for (int x = 0; x < W; ++x) {
      run[x] = fr[x] ? run[x] + 1 : 0;
      const float m = std::min(row[x + 1], float(run[x]));
      row[x + 1] = m * m;
      any |= fr[x] != 0;
    }
    float* cl = &wk.clear[size_t(y) * W];
    int* ky = &wk.key[size_t(y) * W];
    std::memset(cl, 0, sizeof(float) * W);
    std::fill(ky, ky + W, -1);
    if (!any) continue;   // 빈칸 없는 줄(실제 지도는 대부분 모름)
    // 빈칸 토막마다 따로 봉투: 양 끝 막힌 칸(F = 0)이 닻이라 토막 밖 포물선은 토막 안에서 이길 수 없다
    for (int x = 0; x < W;) {
      if (!fr[x]) { ++x; continue; }
      const int x0 = x;
      while (x < W && fr[x]) ++x;
      const int n = x - x0 + 2;   // 덧댄 좌표 [x0, x + 1) = 닻 + 토막 + 닻
      dt1d(row + x0, n, wk.d.data(), wk.v.data(), wk.z.data());
      const float* d = wk.d.data() + 1;
      for (int q = 0; q < n - 2; ++q) {
        const float c = std::sqrt(d[q]) * fres - half;
        cl[x0 + q] = c;
        const int L = iceil((c - fmin) * inv - 1e-5f) - 1;
        const int k = std::clamp(L - Lmin, 0, top);
        ky[x0 + q] = k;
        if (k < top) cnt[k]++;   // 맨 위 통은 세지 않음(정렬 안 함). 같은 통 잇단 ++ 는 저장–읽기 지연이라 대부분인 맨 위를 뺌
      }
    }
  }
}

}  // namespace

uint32_t RoomSeg::at(double x, double y) const {
  if (ids.empty()) return 0;
  const int ix = int(std::floor(x / res)) - gx0, iy = int(std::floor(y / res)) - gy0;
  if (ix < 0 || iy < 0 || ix >= w || iy >= h) return 0;
  return ids[size_t(iy) * w + ix];
}

int RoomSeg::index(uint32_t id) const {
  for (size_t i = 0; i < rooms.size(); ++i)
    if (rooms[i].id == id) return int(i);
  return -1;
}

namespace {

struct DSU {
  std::vector<int> p;
  int find(int a) {
    while (p[a] != a) a = p[a] = p[p[a]];
    return a;
  }
};

// 한 번 쓸기 거름 + 붙이기(ToMATo 꼴, 문서 3.4):
//   빈칸을 여유 큰 것부터(층 통, 한 번 세기 정렬) 넣는다. 칸은 이미 들어온 이웃 중 여유가 가장 큰 칸의 '태어남'
//   번호를 물려받는다(가파른 오르막 = 넘치기). 태어남 번호 하나가 성분(union-find cpar)이자 씨앗(union-find spar).
//   성분이 만날 때(층 Lc): Hydra barcode 처럼 어린 쪽 수명 = 태어난 층 − 만난 층. 둘 다 크고(min_seed) 오래 살았으면
//   둘 다 얼리고(섞인 성분), 아니면 짧게 산 쪽 씨앗을 만난 자리 건너편 씨앗에 합친다.
//   dil_min 아래로 내려가면(늦은 단계) 새 씨앗은 없고, 남은 홀로 성분 중 큰 것은 씨앗(strong), 나머지는 약함 —
//   약한 쪽은 만나는 대로 건너편에 붙고, 끝까지 약한 덩이는 min_room 이상이면 방 하나.
std::shared_ptr<RoomSeg> segmentCore(const GridView& g, const RoomParams& P) {
  const auto t0 = std::chrono::steady_clock::now();
  auto S = std::make_shared<RoomSeg>();
  S->w = g.w; S->h = g.h; S->gx0 = g.gx0; S->gy0 = g.gy0; S->res = g.res;
  const int W = g.w, H = g.h;
  const size_t N = size_t(W) * H;
  if (!N || !g.cells) return S;
  Work& wk = work();
  const double res = g.res, a1 = res * res;
  // 1) 분류·정리
  S->rawfree.resize(N);
  {   // 덧댄 분류 격자(cleanFree 가 바로 씀): 0 모름, 1 빈칸, 2 점유, 3 밖
    const int PW = W + 2;
    wk.pc.resize(size_t(PW) * (H + 2));
    std::memset(wk.pc.data(), 3, size_t(PW));
    std::memset(wk.pc.data() + size_t(H + 1) * PW, 3, size_t(PW));
    for (int y = 0; y < H; ++y) {
      uint8_t* pr = &wk.pc[size_t(y + 1) * PW];
      uint8_t* rf = &S->rawfree[size_t(y) * W];
      const int8_t* cr = g.cells + size_t(y) * W;
      pr[0] = pr[W + 1] = 3;
      for (int x = 0; x < W; ++x) {
        const int v = cr[x];
        const uint8_t c = (v >= 0 && v <= P.free_max) ? 1 : (v >= P.occ_min ? 2 : 0);
        pr[x + 1] = c;
        rf[x] = c == 1;
      }
    }
  }
  wk.free.assign(S->rawfree.begin(), S->rawfree.end());
  cleanFree(wk, W, H, int(P.hole_m2 / a1), int(P.speck_m2 / a1 + 1e-9));
  // 테두리 빈칸은 잠시 막음(이웃 범위 검사 없이) — 끝에서 안쪽 이웃의 방을 받음
  std::vector<int> rim;
  auto block = [&](int x, int y) {
    const size_t i = size_t(y) * W + x;
    if (wk.free[i]) { rim.push_back(int(i)); wk.free[i] = 0; }
  };
  for (int x = 0; x < W; ++x) { block(x, 0); if (H > 1) block(x, H - 1); }
  for (int y = 1; y + 1 < H; ++y) { block(0, y); if (W > 1) block(W - 1, y); }
  // 2) 거리 변환 → 여유, 층 통
  const double step = std::max(P.dil_step, 1e-3);
  const int K = std::clamp(int(std::floor((P.dil_max - P.dil_min) / step + 1e-9)) + 1, 1, 120);
  const int life = std::max(1, int(std::lround(P.min_life / step)));
  const int min_seed = std::max(1, int(P.min_seed_m2 / a1));
  const int min_room = int(P.min_room_m2 / a1);
  const int Lmin = std::min(-1, int(std::ceil((res / 2 - P.dil_min) / step - 1e-5)) - 2);   // 여유 ≥ res/2, 층 0 아래 통 하나는 늘 있음
  const int nbins = K - Lmin;   // 맨 위 통(nbins − 1) = 층 K − 1 이상(c > dil_max − step) 전부
  const int top = nbins - 1;
  clearance(wk, W, H, res, P.dil_min, step, Lmin, nbins);
  const float* clear = wk.clear.data();
  const int* ky = wk.key.data();
  // 세기 정렬(내림차순, 맨 위 통 빼고): order = [통 top−1]...[통 0], 끝나면 cnt[k] = 통 k 의 끝
  int* st = wk.cnt.data();
  {
    int acc = 0;
    for (int k = top - 1; k >= 0; --k) {
      const int n = st[k];
      st[k] = acc;
      acc += n;
    }
    wk.order.resize(size_t(acc));
    int* od = wk.order.data();
    for (size_t i = 0; i < N; ++i)
      if (ky[i] >= 0 && ky[i] < top) od[st[ky[i]]++] = int(i);
  }
  // 3) 한 번 쓸기 거름 + 붙이기
  wk.b.assign(N, -1);
  int* b = wk.b.data();
  std::vector<int> cpar, csz, cbirth, ccs, spar;   // 태어남 번호마다: 성분 부모·크기·태어난 층·홑씨앗(−1 = 섞임), 씨앗 부모
  std::vector<uint8_t> strong;
  auto cf = [&](int a) {
    while (cpar[a] != a) a = cpar[a] = cpar[cpar[a]];
    return a;
  };
  auto sf = [&](int a) {
    while (spar[a] != a) a = spar[a] = spar[spar[a]];
    return a;
  };
  auto born = [&](int c, int L) {
    const int n = int(cpar.size());
    b[c] = n;
    cpar.push_back(n); spar.push_back(n); csz.push_back(1); cbirth.push_back(L); ccs.push_back(n); strong.push_back(0);
  };
  auto sig = [&](int r, int L) { return csz[r] >= min_seed && cbirth[r] - L >= life; };
  // 맨 위 층: 한 층이라 만남은 언제나 수명 0(흡수) → 그냥 연결 성분. 줄마다 토막(run)으로 아랫줄 토막과 이음(캐시 친화)
  {
    struct Run { int x0, x1, id; };
    std::vector<Run> prev, cur;
    for (int y = 1; y + 1 < H; ++y) {
      cur.clear();
      const int* kr = ky + size_t(y) * W;
      size_t p = 0;
      for (int x = 1; x + 1 < W;) {
        if (kr[x] != top) { ++x; continue; }
        const int x0 = x;
        while (x + 1 < W && kr[x] == top) ++x;
        const int x1 = x - 1;   // [x0, x1]
        int id = -1;
        while (p < prev.size() && prev[p].x1 < x0) ++p;   // 4 이웃: 겹치는 아랫줄 토막
        for (size_t q = p; q < prev.size() && prev[q].x0 <= x1; ++q) {
          if (id < 0) { id = prev[q].id; continue; }
          int ra = cf(id), rb = cf(prev[q].id);
          if (ra == rb) continue;
          if (csz[rb] > csz[ra]) std::swap(ra, rb);
          cpar[rb] = ra;
          spar[ccs[rb]] = ccs[ra];
          csz[ra] += csz[rb];
        }
        const int c0 = y * W + x0;
        if (id < 0) {
          born(c0, K - 1);
          id = b[c0];
          csz[id] = 0;
        }
        std::fill(b + c0, b + y * W + x1 + 1, id);
        csz[cf(id)] += x1 - x0 + 1;
        cur.push_back({x0, x1, id});
      }
      prev.swap(cur);
    }
  }
  int nbig = 0;
  for (int i = 0; i < int(cpar.size()); ++i) nbig += cpar[i] == i && csz[i] >= min_seed;
  S->filtration.push_back({P.dil_min + (K - 1) * step, nbig});
  bool late = false;
  auto goLate = [&]() {   // dil_min 아래로: 섞인 성분의 씨앗·큰 홑성분 씨앗 = strong
    late = true;
    for (int i = 0; i < int(spar.size()); ++i)
      if (spar[i] == i) {
        const int r = cf(i);
        strong[i] = ccs[r] < 0 || csz[r] >= min_seed;
      }
  };
  const int* order = wk.order.data();
  for (int k = top - 1; k >= 0; --k) {
    const int L = k + Lmin, Lc = L;
    if (L < 0 && !late) goLate();
    const int* it = order + (k == top - 1 ? 0 : st[k + 1]);
    const int* end = order + st[k];
    for (; it != end; ++it) {
      const int c = *it;
      const int nb[4] = {c - 1, c + 1, c - W, c + W};   // 빈칸은 테두리에 없음(rim)
      int best = -1;
      float bc = -1.f;
      for (int j : nb)
        if (b[j] >= 0 && clear[j] > bc) { bc = clear[j]; best = j; }
      if (best < 0) {   // 새 성분 = 새 씨앗(늦은 단계면 약함)
        born(c, Lc);
        nbig += min_seed <= 1;
        continue;
      }
      b[c] = b[best];
      int rc = cf(b[c]);
      nbig += ++csz[rc] == min_seed;
      for (int j : nb) {
        if (j == best || b[j] < 0) continue;
        int rb = cf(b[j]);
        if (rb == rc) continue;
        int ra = rc, sideA = sf(b[c]), sideB = sf(b[j]);   // 만난 자리 양쪽 씨앗
        // ra = 나이 많은 쪽(태어난 층 높음, 같으면 큰 쪽)
        if (cbirth[rb] > cbirth[ra] || (cbirth[rb] == cbirth[ra] && csz[rb] > csz[ra])) { std::swap(ra, rb); std::swap(sideA, sideB); }
        const bool bigA = csz[ra] >= min_seed, bigB = csz[rb] >= min_seed;
        const int sa = ccs[ra], sb = ccs[rb];
        int out = -1;
        if (!late) {
          if (sa >= 0 && sb >= 0) {
            if (!(sig(ra, Lc) && sig(rb, Lc))) {   // 짧게 산 쪽을 흡수(둘 다 짧으면 어린 쪽을 나이 많은 쪽에)
              const bool keepA = sig(ra, Lc) || !sig(rb, Lc);
              const int keep = keepA ? sa : sb, lose = keepA ? sb : sa;
              spar[lose] = keep;
              out = keep;
            }
          } else if (sa >= 0 || sb >= 0) {   // 홑성분이 섞인 성분을 만남: 짧게 살았으면 건너편 씨앗에
            const bool singleA = sa >= 0;
            if (!sig(singleA ? ra : rb, Lc)) spar[singleA ? sa : sb] = singleA ? sideB : sideA;
          }
        } else {
          const bool stA = sa < 0 || strong[sa], stB = sb < 0 || strong[sb];
          if (stA && !stB) { spar[sb] = sideA; out = sa; }
          else if (!stA && stB) { spar[sa] = sideB; out = sb; }
          else if (!stA && !stB) { spar[sb] = sa; out = sa; }
        }
        cpar[rb] = ra;
        csz[ra] += csz[rb];
        ccs[ra] = out;
        nbig += (csz[ra] >= min_seed) - bigA - bigB;
        rc = ra;
      }
    }
    if (L >= 0) S->filtration.push_back({P.dil_min + L * step, nbig});
  }
  if (!late) goLate();
  // 씨앗 → 방 번호: strong 씨앗(태어남 순), 그다음 끝까지 약한 덩이 중 min_room 이상
  const int nbirth = int(cpar.size());
  std::vector<int> reg(nbirth, 0);
  int nreg = 0;
  for (int i = 0; i < nbirth; ++i)
    if (spar[i] == i && strong[i]) reg[i] = ++nreg;
  S->n_seeds = nreg;
  for (int i = 0; i < nbirth; ++i)
    if (spar[i] == i && !strong[i] && csz[cf(i)] >= min_room) reg[i] = ++nreg;
  for (int i = 0; i < nbirth; ++i) reg[i] = reg[sf(i)];
  S->ids.assign(N, 0);
  uint32_t* ids = S->ids.data();
  for (size_t i = 0; i < N; ++i)
    if (b[i] >= 0) ids[i] = uint32_t(reg[b[i]]);
  if (W >= 3 && H >= 3)
    for (int c : rim) {
      const int x = std::clamp(c % W, 1, W - 2), y = std::clamp(c / W, 1, H - 2);
      ids[c] = ids[size_t(y) * W + x];
    }
  // 4) 방마다 모음(넓이·무게중심·상자·최대 여유)과 이음매(맞닿은 칸 쌍) — 한 번 쓸기, 합친 뒤엔 모음끼리 더함
  struct Acc {
    int n = 0;
    double sx = 0, sy = 0;
    int x0 = INT_MAX, y0 = INT_MAX, x1 = -1, y1 = -1;
    float cmax = 0;
  };
  struct Seam { int n = 0; double sx = 0, sy = 0; float cmax = 0; };
  std::vector<Acc> acc(nreg + 1);
  std::unordered_map<uint64_t, Seam> sm;
  auto addSeam = [&](uint32_t a, uint32_t bb, double px, double py, float cm) {
    Seam& s = sm[(uint64_t(std::min(a, bb)) << 32) | std::max(a, bb)];
    ++s.n;
    s.sx += px;
    s.sy += py;
    s.cmax = std::max(s.cmax, cm);
  };
  for (int y = 0; y < H; ++y) {
    const uint32_t* row = ids + size_t(y) * W;
    const uint32_t* up = y + 1 < H ? row + W : nullptr;
    const float* cl = clear + size_t(y) * W;
    for (int x = 0; x < W;) {   // 같은 방 토막마다 모음
      const uint32_t a = row[x];
      if (!a) { ++x; continue; }
      const int x0 = x;
      while (x < W && row[x] == a) ++x;
      const int x1 = x - 1;
      float m4[4] = {0.f, 0.f, 0.f, 0.f};   // 최대 여유: 네 갈래로 의존 사슬을 끊음
      int q = x0;
      for (; q + 3 <= x1; q += 4)
        for (int j = 0; j < 4; ++j) m4[j] = std::max(m4[j], cl[q + j]);
      for (; q <= x1; ++q) m4[0] = std::max(m4[0], cl[q]);
      const float cm = std::max(std::max(m4[0], m4[1]), std::max(m4[2], m4[3]));
      if (up)
        for (int q2 = x0; q2 <= x1; ++q2)
          if (up[q2] != a && up[q2]) addSeam(a, up[q2], q2, y + 0.5, std::min(cl[q2], cl[q2 + W]));
      Acc& r = acc[a];
      const int len = x1 - x0 + 1;
      r.n += len;
      r.sx += 0.5 * double(x0 + x1) * len;
      r.sy += double(y) * len;
      r.x0 = std::min(r.x0, x0); r.x1 = std::max(r.x1, x1);
      r.y0 = std::min(r.y0, y); r.y1 = std::max(r.y1, y);
      r.cmax = std::max(r.cmax, cm);
      if (x < W && row[x]) addSeam(a, row[x], x1 + 0.5, y, std::min(cl[x1], cl[x]));
    }
  }
  // 5) 합치기(긴 이음매 · 작은 방)
  DSU rg;
  rg.p.resize(nreg + 1);
  std::iota(rg.p.begin(), rg.p.end(), 0);
  std::vector<int> area(nreg + 1, 0);
  for (int r = 1; r <= nreg; ++r) area[r] = acc[r].n;
  const int max_door = P.max_door_m > 0 ? int(std::lround(P.max_door_m / res)) : 1 << 30;
  for (auto& [k, s] : sm)
    if (s.n >= max_door) {
      const int a = rg.find(int(k >> 32)), bb = rg.find(int(k & 0xffffffffu));
      if (a != bb) { rg.p[bb] = a; area[a] += area[bb]; }
    }
  std::vector<uint8_t> dropped(nreg + 1, 0);
  while (true) {   // 가장 작은 방부터 이음매가 가장 긴 이웃에
    int small = -1;
    for (int r = 1; r <= nreg; ++r)
      if (rg.find(r) == r && !dropped[r] && area[r] < min_room && (small < 0 || area[r] < area[small])) small = r;
    if (small < 0) break;
    std::unordered_map<int, int> nbn;
    for (auto& [k, s] : sm) {
      const int a = rg.find(int(k >> 32)), bb = rg.find(int(k & 0xffffffffu));
      if (a == bb) continue;
      if (a == small) nbn[bb] += s.n;
      else if (bb == small) nbn[a] += s.n;
    }
    int best = -1, bn = 0;
    for (auto& [r, n] : nbn)
      if (n > bn || (n == bn && r < best)) { bn = n; best = r; }
    if (best < 0) { dropped[small] = 1; continue; }
    rg.p[small] = best;
    area[best] += area[small];
  }
  // 다시 번호(1..n)
  std::vector<int> remap(nreg + 1, 0);
  int n = 0;
  bool same = true;
  for (int r = 1; r <= nreg; ++r) {
    const int root = rg.find(r);
    if (dropped[root]) { same = false; continue; }
    if (!remap[root]) remap[root] = ++n;
    remap[r] = remap[root];
    same = same && remap[r] == r;
  }
  if (!same)
    for (size_t i = 0; i < N; ++i) ids[i] = ids[i] ? uint32_t(remap[ids[i]]) : 0;
  // 6) 방 모양·문
  S->rooms.resize(n);
  std::vector<Acc> ra(n);
  for (int r = 1; r <= nreg; ++r) {
    const int v = remap[r];
    if (!v) continue;
    Acc& t = ra[v - 1];
    const Acc& q = acc[r];
    t.n += q.n; t.sx += q.sx; t.sy += q.sy;
    t.x0 = std::min(t.x0, q.x0); t.x1 = std::max(t.x1, q.x1);
    t.y0 = std::min(t.y0, q.y0); t.y1 = std::max(t.y1, q.y1);
    t.cmax = std::max(t.cmax, q.cmax);
  }
  for (int i = 0; i < n; ++i) {
    RoomGeom& r = S->rooms[i];
    const Acc& q = ra[i];
    r.id = uint32_t(i + 1);
    r.n_cells = q.n;
    r.area_m2 = q.n * a1;
    r.max_clear = q.cmax;
    r.centroid[0] = (g.gx0 + q.sx / std::max(1, q.n) + 0.5) * res;
    r.centroid[1] = (g.gy0 + q.sy / std::max(1, q.n) + 0.5) * res;
    r.bmin[0] = (g.gx0 + q.x0) * res; r.bmax[0] = (g.gx0 + q.x1 + 1) * res;
    r.bmin[1] = (g.gy0 + q.y0) * res; r.bmax[1] = (g.gy0 + q.y1 + 1) * res;
  }
  {   // 이음매를 새 번호로 모음(합친 방 안쪽 이음매는 사라짐)
    std::unordered_map<uint64_t, Seam> m2;
    for (auto& [k, q] : sm) {
      const int a = remap[int(k >> 32)], bb = remap[int(k & 0xffffffffu)];
      if (!a || !bb || a == bb) continue;
      Seam& t = m2[(uint64_t(std::min(a, bb)) << 32) | uint32_t(std::max(a, bb))];
      t.n += q.n; t.sx += q.sx; t.sy += q.sy; t.cmax = std::max(t.cmax, q.cmax);
    }
    sm.swap(m2);
  }
  const int min_seam = 2;
  for (auto& [k, s] : sm) {
    if (s.n < min_seam) continue;
    RoomDoor d;
    d.a = uint32_t(k >> 32);
    d.b = uint32_t(k & 0xffffffffu);
    d.pos[0] = (g.gx0 + s.sx / s.n + 0.5) * res;
    d.pos[1] = (g.gy0 + s.sy / s.n + 0.5) * res;
    d.width = 2 * s.cmax + res;
    d.seam = s.n;
    S->doors.push_back(d);
  }
  std::sort(S->doors.begin(), S->doors.end(), [](const RoomDoor& x, const RoomDoor& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
  S->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return S;
}

}  // namespace

// 빈칸 상자(+ 여유 2 칸)만 잘라 나누고 온 격자 크기로 되돌림(실제 지도는 대부분 모름)
std::shared_ptr<RoomSeg> segmentRooms(const GridView& g, const RoomParams& P) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!g.cells || g.w <= 0 || g.h <= 0) return segmentCore(g, P);
  int x0 = g.w, y0 = g.h, x1 = -1, y1 = -1;
  for (int y = 0; y < g.h; ++y) {
    const int8_t* row = g.cells + size_t(y) * g.w;
    int a = -1, b = -1;
    for (int x = 0; x < g.w; ++x)
      if (row[x] >= 0 && row[x] <= P.free_max) { a = x; break; }
    if (a < 0) continue;
    for (int x = g.w - 1; x >= a; --x)
      if (row[x] >= 0 && row[x] <= P.free_max) { b = x; break; }
    x0 = std::min(x0, a); x1 = std::max(x1, b);
    y0 = std::min(y0, y); y1 = y;
  }
  if (x1 < 0) return segmentCore(g, P);
  x0 = std::max(0, x0 - 2); y0 = std::max(0, y0 - 2); x1 = std::min(g.w - 1, x1 + 2); y1 = std::min(g.h - 1, y1 + 2);
  const int w = x1 - x0 + 1, h = y1 - y0 + 1;
  if (size_t(w) * h * 10 > size_t(g.w) * g.h * 9) return segmentCore(g, P);   // 거의 다면 그대로
  std::vector<int8_t> sub(size_t(w) * h);
  for (int y = 0; y < h; ++y) std::memcpy(&sub[size_t(y) * w], &g.cells[size_t(y + y0) * g.w + x0], size_t(w));
  auto S = segmentCore(GridView{sub.data(), w, h, g.res, g.gx0 + x0, g.gy0 + y0}, P);
  std::vector<uint32_t> ids(size_t(g.w) * g.h, 0);
  std::vector<uint8_t> rf(size_t(g.w) * g.h, 0);
  for (int y = 0; y < h; ++y) {
    std::memcpy(&ids[size_t(y + y0) * g.w + x0], &S->ids[size_t(y) * w], sizeof(uint32_t) * w);
    std::memcpy(&rf[size_t(y + y0) * g.w + x0], &S->rawfree[size_t(y) * w], size_t(w));
  }
  S->ids.swap(ids);
  S->rawfree.swap(rf);
  S->w = g.w; S->h = g.h; S->gx0 = g.gx0; S->gy0 = g.gy0;
  S->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return S;
}

void matchRoomIds(RoomSeg& cur, const RoomSeg* prev, uint32_t* next_id, double match_min) {
  const int n = int(cur.rooms.size());
  std::vector<uint32_t> nid(n + 1, 0);   // 지금 id(1..n) → 새 id
  if (prev && !prev->rooms.empty() && !cur.ids.empty()) {
    const int m = int(prev->rooms.size());
    std::unordered_map<uint32_t, int> pidx;
    for (int j = 0; j < m; ++j) pidx[prev->rooms[j].id] = j;
    std::vector<int> ov(size_t(n) * m, 0);
    const int dx = cur.gx0 - prev->gx0, dy = cur.gy0 - prev->gy0;
    for (int y = 0; y < cur.h; ++y) {
      const int py = y + dy;
      if (py < 0 || py >= prev->h) continue;
      for (int x = 0; x < cur.w; ++x) {
        const uint32_t a = cur.ids[size_t(y) * cur.w + x];
        if (!a) continue;
        const int px = x + dx;
        if (px < 0 || px >= prev->w) continue;
        const uint32_t b = prev->ids[size_t(py) * prev->w + px];
        if (!b) continue;
        auto it = pidx.find(b);
        if (it != pidx.end()) ov[size_t(a - 1) * m + it->second]++;
      }
    }
    std::vector<std::tuple<int, int, int>> pairs;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j) {
        const int o = ov[size_t(i) * m + j];
        if (o > 0 && o >= match_min * std::min(cur.rooms[i].n_cells, prev->rooms[j].n_cells)) pairs.push_back({o, i, j});
      }
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<uint8_t> usedc(n, 0), usedp(m, 0);
    for (auto& [o, i, j] : pairs) {
      if (usedc[i] || usedp[j]) continue;
      usedc[i] = usedp[j] = 1;
      nid[i + 1] = prev->rooms[j].id;
    }
  }
  for (int i = 1; i <= n; ++i)
    if (!nid[i]) nid[i] = (*next_id)++;
  for (uint32_t& v : cur.ids) v = v ? nid[v] : 0;
  for (RoomGeom& r : cur.rooms) r.id = nid[r.id];
  for (RoomDoor& d : cur.doors) {
    d.a = nid[d.a];
    d.b = nid[d.b];
    if (d.a > d.b) std::swap(d.a, d.b);
  }
  std::sort(cur.rooms.begin(), cur.rooms.end(), [](const RoomGeom& a, const RoomGeom& b) { return a.id < b.id; });
  std::sort(cur.doors.begin(), cur.doors.end(), [](const RoomDoor& x, const RoomDoor& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
}

// ---- 물체 배정·이름 ----

uint32_t assignObject(const RoomSeg& s, const RoomObj& o, const RoomParams& p) {
  if (s.ids.empty()) return 0;
  const double res = s.res;
  const double hx = std::max(0.1, o.ext[0] / 2) + p.footprint_margin, hy = std::max(0.1, o.ext[1] / 2) + p.footprint_margin;
  const int x0 = std::max(0, int(std::floor((o.pos[0] - hx) / res)) - s.gx0), x1 = std::min(s.w - 1, int(std::floor((o.pos[0] + hx) / res)) - s.gx0);
  const int y0 = std::max(0, int(std::floor((o.pos[1] - hy) / res)) - s.gy0), y1 = std::min(s.h - 1, int(std::floor((o.pos[1] + hy) / res)) - s.gy0);
  std::unordered_map<uint32_t, int> votes;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
      if (const uint32_t v = s.ids[size_t(y) * s.w + x]) votes[v]++;
  uint32_t best = 0;
  int bn = 0;
  for (auto& [id, n] : votes)
    if (n > bn || (n == bn && id < best)) { bn = n; best = id; }
  if (best) return best;
  // 가까운 방 칸(고리 넓히기)
  const int cx = int(std::floor(o.pos[0] / res)) - s.gx0, cy = int(std::floor(o.pos[1] / res)) - s.gy0;
  const int R = int(std::ceil(p.obj_search_m / res));
  double bd = 1e18;
  for (int r = 1; r <= R; ++r) {
    for (int y = cy - r; y <= cy + r; ++y)
      for (int x = cx - r; x <= cx + r; ++x) {
        if (std::max(std::abs(x - cx), std::abs(y - cy)) != r || x < 0 || y < 0 || x >= s.w || y >= s.h) continue;
        const uint32_t v = s.ids[size_t(y) * s.w + x];
        const double d = double(x - cx) * (x - cx) + double(y - cy) * (y - cy);
        if (v && d < bd && d <= double(R) * R) { bd = d; best = v; }
      }
    if (best && bd <= double(r) * r) break;   // 다음 고리는 더 멀다
  }
  return best;
}

namespace {
struct Rule { const char* key; const char* type; double w; };
// 머리 명사 규칙(이름 == key 이거나 " key" 로 끝남, 끝 's' 무시). 한 물체가 여러 규칙에 맞으면 가장 긴 key
const Rule kRules[] = {
    {"refrigerator", "kitchen", 3}, {"fridge", "kitchen", 3}, {"oven", "kitchen", 3}, {"stove", "kitchen", 3},
    {"cooktop", "kitchen", 3},      {"range hood", "kitchen", 2}, {"microwave", "kitchen", 2}, {"dishwasher", "kitchen", 3},
    {"toaster", "kitchen", 2},      {"kettle", "kitchen", 1},  {"coffee maker", "kitchen", 1}, {"kitchen sink", "kitchen", 3},
    {"sink", "kitchen", 1},         {"sink", "bathroom", 1},
    {"toilet", "bathroom", 3},      {"bathtub", "bathroom", 3}, {"bath", "bathroom", 2}, {"shower", "bathroom", 3},
    {"bathroom sink", "bathroom", 3},
    {"bed", "bedroom", 3},          {"nightstand", "bedroom", 2}, {"wardrobe", "bedroom", 1}, {"dresser", "bedroom", 1},
    {"pillow", "bedroom", 1},
    {"sofa", "living room", 3},     {"couch", "living room", 3}, {"tv", "living room", 2}, {"television", "living room", 2},
    {"coffee table", "living room", 2}, {"armchair", "living room", 1}, {"fireplace", "living room", 1},
    {"desk", "office", 2},          {"monitor", "office", 2},  {"office chair", "office", 2}, {"computer", "office", 1},
    {"keyboard", "office", 1},      {"printer", "office", 1},
};
constexpr double kPrior = 1.5;     // "unknown" 몫(점수 단위)
constexpr double kNameMin = 2.0;   // 이 점수 이상이어야 종류 이름

std::string norm(std::string t) {
  const size_t p = t.find(".n.");
  if (p != std::string::npos) t.resize(p);
  for (char& ch : t) ch = ch == '_' ? ' ' : char(std::tolower(static_cast<unsigned char>(ch)));
  while (!t.empty() && t.back() == ' ') t.pop_back();
  while (!t.empty() && t.front() == ' ') t.erase(t.begin());
  return t;
}
bool head(const std::string& n, const std::string& k) {
  auto ends = [](const std::string& a, const std::string& b) {
    return a == b || (a.size() > b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0 && a[a.size() - b.size() - 1] == ' ');
  };
  return ends(n, k) || (n.size() > 1 && n.back() == 's' && ends(n.substr(0, n.size() - 1), k));
}
}  // namespace

RoomNaming nameRooms(const RoomSeg& s, const std::vector<RoomObj>& objs, const RoomParams& p,
                     const std::unordered_map<uint32_t, NameOverride>* ov) {
  RoomNaming out;
  out.rooms.resize(s.rooms.size());
  out.obj_room.resize(objs.size(), 0);
  std::vector<std::map<std::string, double>> score(s.rooms.size());
  for (size_t i = 0; i < objs.size(); ++i) {
    const uint32_t rid = assignObject(s, objs[i], p);
    out.obj_room[i] = rid;
    const int ri = rid ? s.index(rid) : -1;
    if (ri < 0) continue;
    RoomLabel& L = out.rooms[ri];
    L.objects.push_back(objs[i].id);
    const std::string n = norm(objs[i].name);
    size_t best_len = 0;
    for (const Rule& r : kRules)
      if (head(n, r.key)) best_len = std::max(best_len, std::strlen(r.key));
    if (!best_len) continue;
    for (const Rule& r : kRules)
      if (std::strlen(r.key) == best_len && head(n, r.key)) {
        score[ri][r.type] += r.w;
        L.evidence.push_back({objs[i].id, objs[i].name, r.type, r.w});
      }
  }
  std::map<std::string, int> used;
  for (size_t i = 0; i < s.rooms.size(); ++i) {
    RoomLabel& L = out.rooms[i];
    L.id = s.rooms[i].id;
    double tot = kPrior, best = 0;
    std::string bt;
    for (auto& [t, v] : score[i]) {
      tot += v;
      if (v > best) { best = v; bt = t; }
    }
    for (auto& [t, v] : score[i]) L.probs[t] = v / tot;
    L.probs["unknown"] = kPrior / tot;
    if (ov) {
      auto it = ov->find(L.id);
      if (it != ov->end()) {
        L.name = it->second.name;
        L.type = it->second.name;
        L.conf = it->second.conf;
        L.external = true;
        continue;
      }
    }
    if (best >= kNameMin) {
      L.type = bt;
      const int k = ++used[bt];
      L.name = k == 1 ? bt : bt + " " + std::to_string(k);
      L.conf = float(best / tot);
    } else {
      L.name = "room " + std::to_string(L.id);
      L.conf = float(best / tot);   // 이름 없는 방: 가장 그럴듯한 종류의 확률(근거 없으면 0)
    }
  }
  return out;
}

// ---- 추적 ----

void RoomTracker::setParams(const RoomParams& p) {
  std::lock_guard<std::mutex> g(mu_);
  p_ = p;
  last_check_ = -1e18;   // 다음 update 에서 다시 봄
}
RoomParams RoomTracker::params() const {
  std::lock_guard<std::mutex> g(mu_);
  return p_;
}
std::shared_ptr<const RoomSeg> RoomTracker::current() const {
  std::lock_guard<std::mutex> g(mu_);
  return cur_;
}
int RoomTracker::n_runs() const {
  std::lock_guard<std::mutex> g(mu_);
  return runs_;
}
void RoomTracker::setName(uint32_t id, const char* name, float conf) {
  std::lock_guard<std::mutex> g(mu_);
  if (!name) ov_.erase(id);
  else ov_[id] = NameOverride{name, conf};
}
std::unordered_map<uint32_t, NameOverride> RoomTracker::overrides() const {
  std::lock_guard<std::mutex> g(mu_);
  return ov_;
}
void RoomTracker::reset() {
  std::lock_guard<std::mutex> g(mu_);
  cur_.reset();
  next_id_ = 1;
  epoch_++;
  last_check_ = -1e18;
  ov_.clear();
}

std::shared_ptr<const RoomSeg> RoomTracker::update(const GridView& g, double stamp, bool force) {
  std::shared_ptr<const RoomSeg> prev;
  RoomParams P;
  uint32_t next;
  uint64_t ep;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!p_.enabled || busy_ || !g.cells || g.w <= 0 || g.h <= 0) return cur_;
    const bool back = cur_ && stamp < last_check_ - 1e-9;   // 시각이 뒤로(새 판 등)
    if (!force && cur_ && !back && stamp - last_check_ < p_.period_s) return cur_;
    last_check_ = stamp;
    busy_ = true;
    prev = cur_;
    P = p_;
    next = next_id_;
    ep = epoch_;
  }
  bool run = force || !prev;
  if (!run) {   // 빈칸 분류가 얼마나 바뀌었나(넓어진 곳 포함)
    const size_t lim = size_t(std::max(1.0, P.min_change_m2 / (g.res * g.res)));
    size_t diff = 0;
    const int dx = g.gx0 - prev->gx0, dy = g.gy0 - prev->gy0;
    for (int y = 0; y < g.h && diff < lim; ++y)
      for (int x = 0; x < g.w; ++x) {
        const int v = g.cells[size_t(y) * g.w + x];
        const uint8_t f = v >= 0 && v <= P.free_max;
        const int px = x + dx, py = y + dy;
        const uint8_t pf = (px >= 0 && py >= 0 && px < prev->w && py < prev->h) ? prev->rawfree[size_t(py) * prev->w + px] : 0;
        diff += f != pf;
      }
    run = diff >= lim || std::abs(g.res - prev->res) > 1e-9;
  }
  std::shared_ptr<RoomSeg> s;
  if (run) {
    s = segmentRooms(g, P);
    s->stamp = stamp;
    const auto t0 = std::chrono::steady_clock::now();
    matchRoomIds(*s, prev.get(), &next, P.match_min);
    s->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::lock_guard<std::mutex> lk(mu_);
  busy_ = false;
  if (s && ep == epoch_) {
    cur_ = std::move(s);
    next_id_ = next;
    runs_++;
  }
  return cur_;
}

}  // namespace scenemap
