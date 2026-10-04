// BEHAVIOR 장면 묶음 만들기(호스트). 규칙은 bscene_host.h 머리말. RASC 로더는 training/RL/tools/b1kconv/cpp/rasc.h(읽기만).
#include "bscene_host.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <queue>
#include <thread>

#include <cuda_runtime_api.h>

#include "rasc.h"
#include "env_soa.h"   // env_beh.h: grasp_reach_box(잡는 점 작업 공간) — 환경 성공 판정과 같은 함수

namespace bsc {

namespace {

constexpr float R_FREE = 0.13f;      // 로봇 중심 칸: 0.13 m 안에 충돌 상자 없음 (가정: 몸통 반 폭 0.11 + 0.02 — 곧게 지나갈 틈. 외접원 0.194 면 좁은 통로가 다 막힘)
constexpr float R_APP = 0.38f;       // 물체 다가감: 칸 가운데 → 물체 바닥 자국 ≤ 0.38 m (E0: joint1 축에서 잡는 점 0.38 m, 축 ≈ 몸통 가운데)
constexpr float B1_D0 = 0.30f, B1_D1 = 0.60f;   // B1 가구 앞 점: 가구 바닥 자국에서 (가정)
constexpr float WIN_MARGIN = 0.8f;   // 시작·목표가 창 가장자리에서 떨어질 거리
constexpr int MAX_TASK_PRIM = 5;     // 지도 물체 중 과제 물체(목표 빼고) 상한

uint64_t hmix(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return x;
}

float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) x = sign;
    else { float f = (float)m * (1.f / 16777216.f); std::memcpy(&x, &f, 4); x |= sign; }
  } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
  else x = sign | ((e + 112) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

bool read_file(const std::string& p, std::vector<uint8_t>& out) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out.resize(n > 0 ? (size_t)n : 0);
  const size_t got = n > 0 ? std::fread(out.data(), 1, (size_t)n, f) : 0;
  std::fclose(f);
  return (long)got == n;
}

std::string json_str(const std::string& line, const char* key) {   // "key": "value" (names.jsonl 은 한 줄 한 행, 값에 따옴표 없음)
  const std::string k = std::string("\"") + key + "\": \"";
  const size_t a = line.find(k);
  if (a == std::string::npos) return "";
  const size_t b = line.find('"', a + k.size());
  return b == std::string::npos ? "" : line.substr(a + k.size(), b - a - k.size());
}

// ---- 이름 표 ----
struct Names {
  int n = 0;
  std::vector<std::string> en, syn;
  std::vector<int32_t> aux;   // [n][8]: 상위어, 묶음 시작, 묶음 길이, 비슷한 3, heldout, 종류(0 sim, 1 behavior, 2 syn, 3 hyper)
  std::map<std::string, int> by_syn, by_en;
  int furniture = -1;
};
bool load_names(const std::string& dir, Names& nm, SceneBuild& out, std::string* err) {
  std::vector<uint8_t> js, aux, v16, a16;
  if (!read_file(dir + "/names.jsonl", js) || !read_file(dir + "/name_aux.i32", aux) || !read_file(dir + "/name128.f16", v16) ||
      !read_file(dir + "/app128.f16", a16)) {
    *err = "cannot read vla_v1 tables in " + dir;
    return false;
  }
  std::string all(js.begin(), js.end());
  size_t p = 0;
  while (p < all.size()) {
    size_t q = all.find('\n', p);
    if (q == std::string::npos) q = all.size();
    const std::string line = all.substr(p, q - p);
    p = q + 1;
    if (line.empty()) continue;
    nm.en.push_back(json_str(line, "en"));
    nm.syn.push_back(json_str(line, "synset"));
  }
  nm.n = (int)nm.en.size();
  if (aux.size() != (size_t)nm.n * 8 * 4 || v16.size() != (size_t)nm.n * 128 * 2 || a16.size() != 7 * 128 * 2) {
    *err = "vla_v1 table sizes do not match names.jsonl";
    return false;
  }
  nm.aux.resize((size_t)nm.n * 8);
  std::memcpy(nm.aux.data(), aux.data(), aux.size());
  for (int i = 0; i < nm.n; ++i) {
    const int kind = nm.aux[i * 8 + 7];
    if (!nm.syn[i].empty() && (kind == 0 || kind == 1) && !nm.by_syn.count(nm.syn[i])) nm.by_syn[nm.syn[i]] = i;
    if (!nm.en[i].empty() && !nm.by_en.count(nm.en[i])) nm.by_en[nm.en[i]] = i;
  }
  for (int i = 0; i < nm.n; ++i)   // 동의어 행은 앞의 정식 행이 없을 때만
    if (!nm.syn[i].empty() && !nm.by_syn.count(nm.syn[i])) nm.by_syn[nm.syn[i]] = i;
  nm.furniture = nm.by_en.count("furniture") ? nm.by_en["furniture"] : -1;
  // 확신도: vla_vocab_gen.py 와 같은 정의(코사인 = 내적, 어휘 = 상위어·heldout 뺀 행, 2위 = 같은 동의어 묶음 밖 최대)
  std::vector<float> V((size_t)nm.n * 128), A(7 * 128);
  const uint16_t* vh = reinterpret_cast<const uint16_t*>(v16.data());
  const uint16_t* ah = reinterpret_cast<const uint16_t*>(a16.data());
  for (size_t k = 0; k < V.size(); ++k) V[k] = h2f(vh[k]);
  for (size_t k = 0; k < A.size(); ++k) A[k] = h2f(ah[k]);
  out.nname = nm.n;
  out.conf1.assign((size_t)7 * nm.n, 0.f);
  out.conf2.assign((size_t)7 * nm.n, 0.f);
  for (int a = 0; a < 7; ++a) {
    std::vector<float> dot(nm.n);
    for (int i = 0; i < nm.n; ++i) {
      double s = 0;
      for (int k = 0; k < 128; ++k) s += (double)A[a * 128 + k] * (double)V[(size_t)i * 128 + k];
      dot[i] = (float)s;
    }
    for (int i = 0; i < nm.n; ++i) {
      const int g = nm.aux[i * 8 + 1];
      float mx = -1e30f;
      for (int j = 0; j < nm.n; ++j) {
        const bool voc = nm.aux[j * 8 + 7] != 3 && nm.aux[j * 8 + 6] == 0;
        if (voc && nm.aux[j * 8 + 1] != g) mx = std::max(mx, dot[j]);
      }
      out.conf1[(size_t)a * nm.n + i] = dot[i];
      out.conf2[(size_t)a * nm.n + i] = dot[i] - mx;
    }
  }
  out.sim3.resize((size_t)nm.n * 3);
  out.hyper.resize(nm.n);
  out.name_en = nm.en;
  for (int i = 0; i < nm.n; ++i) {
    for (int k = 0; k < 3; ++k) out.sim3[(size_t)i * 3 + k] = (int16_t)nm.aux[i * 8 + 3 + k];
    out.hyper[i] = (int16_t)nm.aux[i * 8 + 0];
  }
  return true;
}
int name_of_synset(const Names& nm, const std::string& syn, bool* hit) {
  auto it = nm.by_syn.find(syn);
  if (it != nm.by_syn.end()) { *hit = true; return it->second; }
  // synset 의 낱말(점 앞, _ → 빈칸)
  std::string w = syn.substr(0, syn.find('.'));
  for (auto& ch : w) if (ch == '_') ch = ' ';
  auto e = nm.by_en.find(w);
  if (e != nm.by_en.end()) { *hit = true; return e->second; }
  *hit = false;
  return nm.furniture;
}
int name_of_category(const Names& nm, const std::string& cat) {
  std::string w = cat;
  for (auto& ch : w) if (ch == '_') ch = ' ';
  auto e = nm.by_en.find(w);
  if (e != nm.by_en.end()) return e->second;
  auto s = nm.by_syn.find(cat + ".n.01");
  if (s != nm.by_syn.end()) return s->second;
  // 마지막 낱말(예: breakfast_table → table, bottom_cabinet → cabinet)
  const size_t u = cat.rfind('_');
  if (u != std::string::npos) {
    auto e2 = nm.by_en.find(cat.substr(u + 1));
    if (e2 != nm.by_en.end()) return e2->second;
  }
  return nm.furniture;
}

int room_type_of(const std::string& name) {   // VLA_INPUT 4절 방 종류 6 (가정: 이름 앞부분)
  auto has = [&](const char* p) { return name.rfind(p, 0) == 0; };
  if (has("kitchen")) return 0;
  if (has("bathroom")) return 1;
  if (has("bedroom") || has("childs_room")) return 2;
  if (has("living_room")) return 3;
  if (has("private_office") || has("shared_office") || has("office")) return 4;
  return 5;
}

// ---- 기하(호스트) ----
float dist_pt_obb(const SBox& b, float x, float y) {   // 점 → 회전 상자 바닥 자국(안이면 0)
  const float dx = x - b.cx, dy = y - b.cy;
  const float lx = b.c * dx + b.s * dy, ly = -b.s * dx + b.c * dy;
  const float ex = std::max(std::fabs(lx) - b.hx, 0.f), ey = std::max(std::fabs(ly) - b.hy, 0.f);
  return std::sqrt(ex * ex + ey * ey);
}
float dist_pt_rect(const float lo[3], const float hi[3], float x, float y) {
  const float ex = std::max(std::max(lo[0] - x, x - hi[0]), 0.f), ey = std::max(std::max(lo[1] - y, y - hi[1]), 0.f);
  return std::sqrt(ex * ex + ey * ey);
}
void obb_aabb(const SBox& b, float lo[2], float hi[2]) {
  const float ex = std::fabs(b.c) * b.hx + std::fabs(b.s) * b.hy, ey = std::fabs(b.s) * b.hx + std::fabs(b.c) * b.hy;
  lo[0] = b.cx - ex; hi[0] = b.cx + ex; lo[1] = b.cy - ey; hi[1] = b.cy + ey;
}

struct TObj { int k; float lo[3], hi[3]; int16_t name; };   // 인스턴스의 과제 물체(세계 AABB)

bool trav_bit(const rasc::Scene& s, int layer, int r, int c) { return s.free_cell(layer, r, c); }

// 창 안 다익스트라(공용): 장면 칸 기준 창 (c0, r0) 부터 WIN × WIN
void dijkstra_win(const SceneBuild::Sc& s, int c0, int r0, const std::vector<std::pair<int, float>>& seeds, std::vector<float>& dist) {
  const int W = s.d.W, H = s.d.H;
  dist.assign((size_t)WIN * WIN, -1.f);
  auto freec = [&](int i, int j) -> bool {
    const int c = c0 + i, r = r0 + j;
    return i >= 0 && j >= 0 && i < WIN && j < WIN && c >= 0 && r >= 0 && c < W && r < H && s.freeg[(size_t)r * W + c];
  };
  auto fz = [&](int i, int j) -> int { const int16_t v = s.floor_mm[(size_t)(r0 + j) * W + (c0 + i)]; return v == INT16_MIN ? 0 : v; };
  const int thr = (int)std::lround(s.threshold * 1000.f);
  typedef std::pair<float, int> QE;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
  std::vector<float> best((size_t)WIN * WIN, 1e30f);
  for (auto& sd : seeds) {
    if (sd.first < 0) continue;
    if (sd.second < best[sd.first]) { best[sd.first] = sd.second; pq.push({sd.second, sd.first}); }
  }
  const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (!pq.empty()) {
    const QE q = pq.top();
    pq.pop();
    const int u = q.second;
    if (q.first > best[u]) continue;
    dist[u] = q.first;
    const int i = u % WIN, j = u / WIN;
    for (int k = 0; k < 8; ++k) {
      const int ni = i + dx[k], nj = j + dy[k];
      if (!freec(ni, nj)) continue;
      if (k >= 4 && (!freec(i + dx[k], j) || !freec(i, j + dy[k]))) continue;
      if (std::abs(fz(ni, nj) - fz(i, j)) > thr) continue;
      const float nd = q.first + (k >= 4 ? 0.14142136f : 0.1f);
      const int v = nj * WIN + ni;
      if (nd < best[v]) { best[v] = nd; pq.push({nd, v}); }
    }
  }
}

}  // namespace

std::string default_vla_dir() {
  std::string f = __FILE__;   // .../training/RL/env/src/bscene_host.cpp
  const size_t p = f.rfind("/training/RL/env/");
  return p == std::string::npos ? "training/data/vla_v1" : f.substr(0, p) + "/training/data/vla_v1";
}
std::string default_rasc_dir() {
  const char* h = std::getenv("HOME");
  return std::string(h ? h : ".") + "/ra_b1k";
}

void window_dijkstra(const SceneBuild::Sc& s, float wx, float wy, float sx, float sy, std::vector<float>& dist) {
  const int c0 = (int)std::lround((wx - WIN_HALF - s.d.ox) / CELL), r0 = (int)std::lround((wy - WIN_HALF - s.d.oy) / CELL);
  std::vector<std::pair<int, float>> seeds;
  const int si = (int)std::floor((sx + WIN_HALF) / CELL), sj = (int)std::floor((sy + WIN_HALF) / CELL);
  for (int dj = -3; dj <= 3; ++dj)
    for (int di = -3; di <= 3; ++di) {
      const int i = si + di, j = sj + dj, c = c0 + i, r = r0 + j;
      if (i < 0 || j < 0 || i >= WIN || j >= WIN || c < 0 || r < 0 || c >= s.d.W || r >= s.d.H || !s.freeg[(size_t)r * s.d.W + c]) continue;
      const float cx = ((float)i + 0.5f) * CELL - WIN_HALF, cy = ((float)j + 0.5f) * CELL - WIN_HALF;
      const float d = std::sqrt((cx - sx) * (cx - sx) + (cy - sy) * (cy - sy));
      if (d <= 0.3f) seeds.push_back({j * WIN + i, d});
    }
  dijkstra_win(s, c0, r0, seeds, dist);
}

// B3 서는 자리: 창 칸 가운데에 로봇 가운데를 두고 목표 가운데를 바라볼 때(카메라 에임 ≈ 0) 몸통이 안 닿고 잡는 점 작업 공간에 물체 상자가 걸림
bool stance_ok(const SceneBuild::Sc& s, const Entry& e, float cx, float cy) {
  const float yaw = std::atan2(e.gy - cy, e.gx - cx);
  if (!body_free_host(s, e, cx, cy, yaw)) return false;
  float sn, cs;
  dm::sincosf_d(yaw, &sn, &cs);
  const float ctr[3] = {e.gx, e.gy, e.gz};
  return env::grasp_reach_box(ctr, e.ext, cx, cy, cs, sn);
}

void goal_field(const SceneBuild::Sc& s, const Entry& e, std::vector<float>& dist) {
  const int c0 = (int)std::lround((e.wx - WIN_HALF - s.d.ox) / CELL), r0 = (int)std::lround((e.wy - WIN_HALF - s.d.oy) / CELL);
  std::vector<std::pair<int, float>> seeds;
  for (int c = 0; c < WIN * WIN; ++c) {
    const int i = c % WIN, j = c / WIN, sc = c0 + i, sr = r0 + j;
    if (sc < 0 || sr < 0 || sc >= s.d.W || sr >= s.d.H || !s.freeg[(size_t)sr * s.d.W + sc]) continue;
    const float cx = ((float)i + 0.5f) * CELL - WIN_HALF, cy = ((float)j + 0.5f) * CELL - WIN_HALF;
    if (e.list == L_ROOM) { if (std::fabs(cx - e.gx) < 1e-4f && std::fabs(cy - e.gy) < 1e-4f) seeds.push_back({c, 0.f}); }
    else if (dist_pt_rect(e.prim[0].lo, e.prim[0].hi, cx, cy) <= 0.6f && stance_ok(s, e, cx, cy)) seeds.push_back({c, 0.f});
  }
  dijkstra_win(s, c0, r0, seeds, dist);
}

bool body_free_host(const SceneBuild::Sc& s, const Entry& e, float x, float y, float yaw) {
  float sn, co;
  dm::sincosf_d(yaw, &sn, &co);   // 환경 커널과 같은 sin·cos
  if (body_hits_scene(s.d, x + e.wx, y + e.wy, sn, co, 0.16f, 0.11f)) return false;
  for (int p = 0; p < e.nprim; ++p) {
    const BPrim& P = e.prim[p];
    if (P.sbox >= 0 || !(P.lo[2] < H_COLL)) continue;
    if (rect_hits_aabb(x, y, sn, co, 0.16f, 0.11f, P.lo, P.hi)) return false;
  }
  return true;
}

namespace {

bool build_one(const std::string& path, const Names& nm, const BuildOpt& opt, SceneBuild::Sc& S, std::vector<Entry>& ents, SceneStats& st,
               int scene_idx, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  rasc::Scene s;
  if (!rasc::load(s, path.c_str(), err)) return false;
  S.name = s.name();
  st.name = S.name;
  const int W = (int)s.h->grid_w, H = (int)s.h->grid_h;
  st.W = W; st.H = H;
  SceneDev& d = S.d;
  d.ox = s.h->origin[0]; d.oy = s.h->origin[1]; d.W = W; d.H = H;
  S.threshold = s.limits[0].threshold;
  // 1. 정적 상자. 벽은 상자(메시 아님)라 문 구멍이 막혀 있다 → 벽 상자에서 문 열린 자리를 잘라 내고 문 위(인방)만 남긴다.
  //    바닥 덮개(잔디·포장·차도처럼 윗면 ≤ 0.2 m 이고 넓이 > 1 m²)는 바닥으로 보고 뺀다(다닐 곳 = RASC 바닥 높이 칸)
  auto push = [&](const SBox& b, uint8_t base, int16_t name) {
    uint8_t kind = base;
    if (b.z0 < H_COLL) kind |= BK_COLL;
    if (b.z1 > 0.05f && b.z0 < 0.50f) kind |= BK_BAND;
    S.box.push_back(b);
    S.bkind.push_back(kind);
    S.bname.push_back(name);
  };
  for (size_t k = 0; k < s.boxes.n; ++k) {
    const RascBoxRec& r = s.boxes[k];
    ++st.nbox_in;
    if (r.flags & (RASC_F_DOOR | RASC_F_CARPET | RASC_F_VISUAL_ONLY)) continue;
    if (r.zmax < 0.03f || r.half[0] <= 0.f || r.half[1] <= 0.f) continue;
    if (r.zmax <= 0.2f && 4.f * r.half[0] * r.half[1] > 1.f) { ++st.ground; continue; }
    SBox b;
    b.cx = r.center[0]; b.cy = r.center[1]; b.hx = r.half[0]; b.hy = r.half[1];
    b.c = std::cos(r.yaw); b.s = std::sin(r.yaw); b.z0 = r.zmin; b.z1 = r.zmax;
    const uint8_t base = (r.flags & RASC_F_WALL) ? BK_WALL : (r.flags & RASC_F_WINDOW) ? BK_WINDOW : BK_FURN;
    const int16_t name = (int16_t)name_of_category(nm, s.str(s.cats[r.cat].name));
    if (!(r.flags & RASC_F_WALL)) { push(b, base, name); continue; }
    // 벽: 긴 축 u(상자 x 또는 y), 문마다 그 축 위 구간 [a, b] 를 빼고 문 위 인방 상자
    const bool ux = b.hx >= b.hy;
    const float ax = ux ? b.c : -b.s, ay = ux ? b.s : b.c;   // 긴 축 방향(세계)
    const float L = ux ? b.hx : b.hy, T = ux ? b.hy : b.hx;  // 반 길이·반 두께
    std::vector<std::pair<float, float>> cut;
    std::vector<float> top;
    for (auto& dr : s.doors) {
      const float dx = dr.center[0] - b.cx, dy = dr.center[1] - b.cy;
      const float u = ax * dx + ay * dy, v = -ay * dx + ax * dy;
      if (std::fabs(v) > T + 0.25f || std::fabs(u) > L + 0.5f) continue;   // 문이 이 벽 안에 있지 않음
      const float dc = std::cos(dr.yaw), ds = std::sin(dr.yaw);
      const float wdir = std::fabs(ax * dc + ay * ds) * dr.half[0] + std::fabs(-ax * ds + ay * dc) * dr.half[1];   // 벽 축으로 본 문 반 폭
      if (wdir < 0.2f) continue;
      cut.push_back({u - wdir, u + wdir});
      top.push_back(dr.center[2] + dr.half[2]);
    }
    if (cut.empty()) { push(b, base, name); continue; }
    std::sort(cut.begin(), cut.end());
    float u0 = -L;
    auto piece = [&](float a, float bb, float z0, float z1) {
      if (bb - a < 0.02f || z1 - z0 < 0.02f) return;
      SBox p = b;
      const float m = 0.5f * (a + bb);
      p.cx = b.cx + ax * m; p.cy = b.cy + ay * m;
      if (ux) p.hx = 0.5f * (bb - a); else p.hy = 0.5f * (bb - a);
      p.z0 = z0; p.z1 = z1;
      push(p, base, name);
    };
    for (size_t q = 0; q < cut.size(); ++q) {
      const float a = std::max(cut[q].first, -L), e = std::min(cut[q].second, L);
      if (e <= u0) continue;
      piece(u0, a, b.z0, b.z1);
      piece(std::max(a, u0), e, std::min(top[q], b.z1), b.z1);   // 인방
      u0 = std::max(u0, e);
      ++st.door_cuts;
    }
    piece(u0, L, b.z0, b.z1);
  }
  st.nbox = d.nbox = (int)S.box.size();
  if (S.box.size() >= 65535) { *err = "too many boxes"; return false; }
  d.BW = (int)std::ceil(W * CELL / BIN) + 1; d.BH = (int)std::ceil(H * CELL / BIN) + 1;
  std::vector<std::vector<uint16_t>> bins((size_t)d.BW * d.BH);
  for (size_t k = 0; k < S.box.size(); ++k) {
    float lo[2], hi[2];
    obb_aabb(S.box[k], lo, hi);
    int bx0, by0, bx1, by1;
    bin_of(d, lo[0] - 1e-3f, lo[1] - 1e-3f, bx0, by0);
    bin_of(d, hi[0] + 1e-3f, hi[1] + 1e-3f, bx1, by1);
    bx0 = std::max(bx0, 0); by0 = std::max(by0, 0); bx1 = std::min(bx1, d.BW - 1); by1 = std::min(by1, d.BH - 1);
    for (int by = by0; by <= by1; ++by) for (int bx = bx0; bx <= bx1; ++bx) bins[(size_t)by * d.BW + bx].push_back((uint16_t)k);
  }
  S.bstart.push_back(0);
  for (auto& v : bins) { for (uint16_t x : v) S.bitem.push_back(x); S.bstart.push_back((uint32_t)S.bitem.size()); }
  st.nbin_items = (int)S.bitem.size();
  // 방·방 종류·문
  S.room.assign(s.room_grid.p, s.room_grid.p + s.room_grid.n);
  for (auto& r : s.rooms) { S.room_names.push_back(s.str(r.name)); S.rtype.push_back((int8_t)room_type_of(s.str(r.name))); }
  d.nroom = st.nroom = (int)S.rtype.size();
  for (auto& dr : s.doors) {
    SDoor o;
    o.x = dr.center[0]; o.y = dr.center[1];
    // 양쪽 방: 얇은 축으로 ±0.5 m 의 방 칸(RASC room_a/b 는 한쪽만 있는 문이 많음)
    const float c = std::cos(dr.yaw), sn = std::sin(dr.yaw);
    const bool thin_x = dr.half[0] < dr.half[1];
    const float ux = thin_x ? c : -sn, uy = thin_x ? sn : c;
    const int ra = s.room_at(o.x + 0.5f * ux, o.y + 0.5f * uy), rb = s.room_at(o.x - 0.5f * ux, o.y - 0.5f * uy);
    o.ra = (int16_t)(ra >= 0 ? ra : (dr.room_a != RASC_NONE16 ? dr.room_a : -1));
    o.rb = (int16_t)(rb >= 0 ? rb : (dr.room_b != RASC_NONE16 ? dr.room_b : -1));
    if (o.ra == o.rb) o.rb = -1;
    S.door.push_back(o);
  }
  d.ndoor = st.ndoor = (int)S.door.size();
  // 띠 점유 래스터
  S.occ.assign(((size_t)W * H + 31) / 32, 0u);
  for (size_t k = 0; k < S.box.size(); ++k) {
    if (!(S.bkind[k] & BK_BAND)) continue;
    float lo[2], hi[2];
    obb_aabb(S.box[k], lo, hi);
    const int c0 = std::max(0, (int)std::floor((lo[0] - d.ox) / CELL)), c1 = std::min(W - 1, (int)std::floor((hi[0] - d.ox) / CELL));
    const int r0 = std::max(0, (int)std::floor((lo[1] - d.oy) / CELL)), r1 = std::min(H - 1, (int)std::floor((hi[1] - d.oy) / CELL));
    for (int r = r0; r <= r1; ++r)
      for (int c = c0; c <= c1; ++c) {
        const float cx = d.ox + ((float)c + 0.5f) * CELL, cy = d.oy + ((float)r + 0.5f) * CELL;
        if (rect_hits_obb(cx, cy, 0.f, 1.f, 0.5f * CELL, 0.5f * CELL, S.box[k])) {
          const size_t i = (size_t)r * W + c;
          S.occ[i >> 5] |= 1u << (i & 31);
        }
      }
  }
  // 호스트 포인터(이 뒤로 벡터 크기 안 바뀜) — 아래 계산이 장치와 같은 함수(body_hits_scene·room_at)를 씀
  d.bstart = S.bstart.data(); d.bitem = S.bitem.data(); d.box = S.box.data(); d.bkind = S.bkind.data(); d.bname = S.bname.data();
  d.room = S.room.data(); d.occ = S.occ.data(); d.rtype = S.rtype.data(); d.door = S.door.data();
  // 로봇 중심 칸 + 래스터 대 TRAV_OPEN_DOOR 비교(방 칸 안)
  S.floor_mm.assign(s.floor_z.p, s.floor_z.p + s.floor_z.n);
  S.freeg.assign((size_t)W * H, 0);
  long inter = 0, uni = 0, agree = 0, nroomc = 0;
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      const size_t i = (size_t)r * W + c;
      const float cx = d.ox + ((float)c + 0.5f) * CELL, cy = d.oy + ((float)r + 0.5f) * CELL;
      // 바닥 있음 = 0.2 m 안에 TRAV_NO_OBJ(벽만 막은 바닥) 빈 칸이 있음 — 0.1 m 칸 축소(100 화소 모두 빈 칸)가 문간을 막아서 문간 칸도 살림
      bool fr = false;
      for (int dr = -2; dr <= 2 && !fr; ++dr)
        for (int dc = -2; dc <= 2 && !fr; ++dc) fr = trav_bit(s, 1, r + dr, c + dc);
      if (fr) {
        int bx0, by0, bx1, by1;
        bin_of(d, cx - R_FREE, cy - R_FREE, bx0, by0);
        bin_of(d, cx + R_FREE, cy + R_FREE, bx1, by1);
        bx0 = std::max(bx0, 0); by0 = std::max(by0, 0); bx1 = std::min(bx1, d.BW - 1); by1 = std::min(by1, d.BH - 1);
        for (int by = by0; by <= by1 && fr; ++by)
          for (int bx = bx0; bx <= bx1 && fr; ++bx) {
            const int b = by * d.BW + bx;
            for (uint32_t q = S.bstart[b]; q < S.bstart[b + 1] && fr; ++q)
              if ((S.bkind[S.bitem[q]] & BK_COLL) && dist_pt_obb(S.box[S.bitem[q]], cx, cy) < R_FREE) fr = false;
          }
      }
      S.freeg[i] = fr ? 1 : 0;
      st.free_cells += fr;
      if (S.room[i]) {
        ++nroomc;
        const bool a = (S.occ[i >> 5] >> (i & 31)) & 1u, bb = !trav_bit(s, 3, r, c);
        inter += a && bb; uni += a || bb; agree += a == bb;
      }
    }
  st.cells = (long)W * H;
  st.occ_iou = uni ? (double)inter / uni : 0;
  st.occ_agree = nroomc ? (double)agree / nroomc : 0;
  // 2. 인스턴스 → 시작 조건 표
  st.inst = (int)s.insts.n;
  std::vector<int> order(s.insts.n);
  for (size_t k = 0; k < order.size(); ++k) order[k] = (int)k;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const uint64_t ha = hmix((uint64_t)a * 2654435761ull + 7), hb = hmix((uint64_t)b * 2654435761ull + 7);
    return ha != hb ? ha < hb : a < b;
  });
  std::vector<Entry> lst[2][2];   // [표][split]
  std::vector<uint8_t> inner_flag[2];
  auto snap = [&](float mx, float my, float& wx, float& wy) {
    const int c0 = (int)std::lround((mx - d.ox) / CELL) - WIN / 2, r0 = (int)std::lround((my - d.oy) / CELL) - WIN / 2;
    wx = d.ox + (float)(c0 + WIN / 2) * CELL;
    wy = d.oy + (float)(r0 + WIN / 2) * CELL;
  };
  auto fits = [&](float ax, float ay, float bx, float by) {
    return std::fabs(ax - bx) <= 2.f * (WIN_HALF - WIN_MARGIN) && std::fabs(ay - by) <= 2.f * (WIN_HALF - WIN_MARGIN);
  };
  auto in_win = [&](float x, float y, float m) { return std::fabs(x) <= WIN_HALF - m && std::fabs(y) <= WIN_HALF - m; };
  auto add_prims = [&](Entry& e, const std::vector<TObj>& tobjs, int tgt_k, int tgt_box, float fx, float fy) {
    // 목표 다음: 과제 물체(창 안, 목표에서 가까운 순, ≤ MAX_TASK_PRIM), 그다음 가구(창 안, 가까운 순)
    std::vector<std::pair<float, int>> tl;
    for (size_t q = 0; q < tobjs.size(); ++q) {
      if (tobjs[q].k == tgt_k) continue;
      const float cx = 0.5f * (tobjs[q].lo[0] + tobjs[q].hi[0]) - e.wx, cy = 0.5f * (tobjs[q].lo[1] + tobjs[q].hi[1]) - e.wy;
      if (!in_win(cx, cy, 0.2f)) continue;
      tl.push_back({std::hypot(cx - fx, cy - fy), (int)q});
    }
    std::sort(tl.begin(), tl.end());
    for (size_t q = 0; q < tl.size() && q < (size_t)MAX_TASK_PRIM && e.nprim < NPRIM; ++q) {
      const TObj& o = tobjs[tl[q].second];
      BPrim& P = e.prim[e.nprim++];
      for (int a = 0; a < 3; ++a) { P.lo[a] = o.lo[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); P.hi[a] = o.hi[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); }
      P.name = o.name; P.sbox = -1;
    }
    std::vector<std::pair<float, int>> fl;
    for (size_t k = 0; k < S.box.size(); ++k) {
      if (!(S.bkind[k] & BK_FURN) || (int)k == tgt_box) continue;
      const SBox& b = S.box[k];
      if (2.f * std::max(b.hx, b.hy) > 4.f || b.z0 >= 0.5f) continue;
      const float cx = b.cx - e.wx, cy = b.cy - e.wy;
      if (!in_win(cx, cy, 0.2f)) continue;
      fl.push_back({std::hypot(cx - fx, cy - fy), (int)k});
    }
    std::sort(fl.begin(), fl.end());
    for (size_t q = 0; q < fl.size() && e.nprim < NPRIM; ++q) {
      const SBox& b = S.box[fl[q].second];
      float lo[2], hi[2];
      obb_aabb(b, lo, hi);
      BPrim& P = e.prim[e.nprim++];
      P.lo[0] = lo[0] - e.wx; P.lo[1] = lo[1] - e.wy; P.lo[2] = b.z0;
      P.hi[0] = hi[0] - e.wx; P.hi[1] = hi[1] - e.wy; P.hi[2] = b.z1;
      P.name = S.bname[fl[q].second]; P.sbox = (int16_t)fl[q].second;
    }
  };
  std::vector<float> dist;
  for (int ii : order) {
    const RascInstRec& in = s.insts[ii];
    const RascTaskRec& tk = s.tasks[in.task];
    const int split = in.split ? 1 : 0;
    // 과제 물체(세계 AABB)
    std::vector<TObj> tobjs;
    auto objs = s.objs_of(tk);
    auto poses = s.poses_of(in);
    for (size_t k = 0; k < objs.n && k < poses.n; ++k) {
      const RascTaskObjRec& o = objs[k];
      const RascPoseRec& ps = poses[k];
      if (o.flags & (RASC_F_AGENT | RASC_F_SYSTEM | RASC_F_FUTURE | RASC_F_UNMAPPED)) continue;
      if (o.scene_obj >= 0 || ps.src == 0 || (o.half[0] <= 0.f && o.half[1] <= 0.f && o.half[2] <= 0.f)) continue;
      const float x = ps.quat[0], y = ps.quat[1], z = ps.quat[2], w = ps.quat[3];
      const float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                          2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
      TObj t;
      t.k = (int)k;
      for (int a = 0; a < 3; ++a) {
        const float cc = ps.pos[a] + R[3 * a] * o.offset[0] + R[3 * a + 1] * o.offset[1] + R[3 * a + 2] * o.offset[2];
        const float e = std::fabs(R[3 * a]) * o.half[0] + std::fabs(R[3 * a + 1]) * o.half[1] + std::fabs(R[3 * a + 2]) * o.half[2];
        t.lo[a] = cc - e; t.hi[a] = cc + e;
      }
      bool hit = false;
      t.name = (int16_t)name_of_synset(nm, s.str(o.syn), &hit);
      if (hit) ++st.name_hit; else ++st.name_miss;
      tobjs.push_back(t);
    }
    const float sxw = in.robot_pos[0], syw = in.robot_pos[1], syaw = in.robot_yaw;
    // ---- 물체 표 ----
    const RascPnpRange& pr = s.pnp_ranges[ii];
    bool any_pick = false, any_reach = false;
    for (uint32_t q = 0; q < pr.n_pick; ++q) {
      const RascPickRec& pk = s.picks[pr.pick_off + q];
      if (!(pk.obj & RASC_PNP_TASKOBJ)) continue;
      any_pick = true;
      if (pk.comp == RASC_NONE16 || pk.comp != pr.robot_comp) continue;
      any_reach = true;
      const int k = (int)(pk.obj & ~RASC_PNP_TASKOBJ) - (int)tk.obj_off;   // TaskObjRec 전역 번호 → 과제 안 번호
      int ti = -1;
      for (size_t t = 0; t < tobjs.size(); ++t) if (tobjs[t].k == k) ti = (int)t;
      if (ti < 0) continue;
      const bool inner = (pk.flags & RASC_PK_INNER) && pk.comp_inner != RASC_NONE16 && pk.comp_inner == pr.robot_comp_inner;
      ++st.obj_try;
      if ((int)lst[L_OBJ][split].size() >= opt.cap_obj) { ++st.rej_cap; continue; }
      const TObj& T = tobjs[ti];
      const float tcx = 0.5f * (T.lo[0] + T.hi[0]), tcy = 0.5f * (T.lo[1] + T.hi[1]);
      if (!fits(sxw, syw, tcx, tcy)) { ++st.rej_window; continue; }
      Entry e{};
      e.scene = (int16_t)scene_idx; e.list = L_OBJ; e.split = (int16_t)split; e.inst = ii; e.task = (int16_t)in.task;
      snap(0.5f * (sxw + tcx), 0.5f * (syw + tcy), e.wx, e.wy);
      e.sx = sxw - e.wx; e.sy = syw - e.wy; e.syaw = syaw;
      e.gx = tcx - e.wx; e.gy = tcy - e.wy; e.gz = 0.5f * (T.lo[2] + T.hi[2]);
      for (int a = 0; a < 3; ++a) e.ext[a] = T.hi[a] - T.lo[a];
      e.groom = (int16_t)room_at(d, tcx, tcy);
      BPrim& P0 = e.prim[0];
      for (int a = 0; a < 3; ++a) { P0.lo[a] = T.lo[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); P0.hi[a] = T.hi[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); }
      P0.name = T.name; P0.sbox = -1;
      e.nprim = 1;
      add_prims(e, tobjs, k, -1, e.gx, e.gy);
      if (!body_free_host(S, e, e.sx, e.sy, e.syaw)) { ++st.rej_start; continue; }
      window_dijkstra(S, e.wx, e.wy, e.sx, e.sy, dist);
      float best = -1.f, bst = -1.f;
      for (int c = 0; c < WIN * WIN; ++c) {
        if (dist[c] < 0.f) continue;
        const float cx = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF, cy = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF;
        const float df = dist_pt_rect(P0.lo, P0.hi, cx, cy);
        if (df <= R_APP && (best < 0.f || dist[c] < best)) best = dist[c];
        if (df <= 0.6f && (bst < 0.f || dist[c] < bst) && stance_ok(S, e, cx, cy)) bst = dist[c];
      }
      if (best < 0.f) { ++st.rej_unreach; continue; }
      if (bst < 0.f) { ++st.rej_stance; continue; }   // B3 성공 자세(잡는 점 작업 공간 + 몸통 안 닿음)가 닿는 칸에 없음
      e.path = best;
      lst[L_OBJ][split].push_back(e);
      inner_flag[split].push_back(inner ? 1 : 0);
      ++st.obj_ok;
      st.obj_inner += inner;
      st.path_sum[L_OBJ] += best;
      st.ratio_sum[L_OBJ] += best / std::max(0.1f, std::hypot(e.gx - e.sx, e.gy - e.sy));
    }
    st.inst_pick += any_pick;
    st.inst_pick_reach += any_reach;
    // ---- 방 표(B1) ----
    if ((int)lst[L_ROOM][split].size() < opt.cap_room) {
      const int rs = room_at(d, sxw, syw);
      std::vector<int> rooms;
      for (int R = 0; R < d.nroom; ++R) if (R != rs) rooms.push_back(R);
      std::sort(rooms.begin(), rooms.end(), [&](int a, int b) {
        const uint64_t ha = hmix(((uint64_t)ii << 16) ^ (uint64_t)a), hb = hmix(((uint64_t)ii << 16) ^ (uint64_t)b);
        return ha != hb ? ha < hb : a < b;
      });
      int made = 0;
      for (int R : rooms) {
        if (made >= opt.rooms_per_inst || (int)lst[L_ROOM][split].size() >= opt.cap_room) break;
        ++st.room_try;
        // 그 방의 바닥 가구, 시작에서 창에 들어가는 것
        std::vector<int> cands;
        for (size_t k = 0; k < S.box.size(); ++k) {
          const SBox& b = S.box[k];
          if (!(S.bkind[k] & BK_FURN) || !(S.bkind[k] & BK_COLL) || 2.f * std::max(b.hx, b.hy) > 3.f) continue;
          if (room_at(d, b.cx, b.cy) != R || !fits(sxw, syw, b.cx, b.cy)) continue;
          cands.push_back((int)k);
        }
        if (cands.empty()) { ++st.room_nocand; continue; }
        const int fk = cands[hmix(((uint64_t)ii << 20) ^ (uint64_t)R) % cands.size()];
        const SBox& F = S.box[fk];
        Entry e{};
        e.scene = (int16_t)scene_idx; e.list = L_ROOM; e.split = (int16_t)split; e.inst = ii; e.task = (int16_t)in.task;
        snap(0.5f * (sxw + F.cx), 0.5f * (syw + F.cy), e.wx, e.wy);
        e.sx = sxw - e.wx; e.sy = syw - e.wy; e.syaw = syaw;
        e.groom = (int16_t)R;
        float lo[2], hi[2];
        obb_aabb(F, lo, hi);
        BPrim& P0 = e.prim[0];
        P0.lo[0] = lo[0] - e.wx; P0.lo[1] = lo[1] - e.wy; P0.lo[2] = F.z0; P0.hi[0] = hi[0] - e.wx; P0.hi[1] = hi[1] - e.wy; P0.hi[2] = F.z1;
        P0.name = S.bname[fk]; P0.sbox = (int16_t)fk;
        e.nprim = 1;
        e.ext[0] = hi[0] - lo[0]; e.ext[1] = hi[1] - lo[1]; e.ext[2] = F.z1 - F.z0;
        add_prims(e, tobjs, -1, fk, F.cx - e.wx, F.cy - e.wy);
        if (!body_free_host(S, e, e.sx, e.sy, e.syaw)) { ++st.rej_start; continue; }
        window_dijkstra(S, e.wx, e.wy, e.sx, e.sy, dist);
        int bc = -1;
        for (int c = 0; c < WIN * WIN; ++c) {
          if (dist[c] < 0.f) continue;
          const float cx = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF, cy = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF;
          if (!in_win(cx, cy, WIN_MARGIN)) continue;
          const float df = dist_pt_obb(F, cx + e.wx, cy + e.wy);
          if (df < B1_D0 || df > B1_D1 || room_at(d, cx + e.wx, cy + e.wy) != R) continue;
          if (bc < 0 || dist[c] < dist[bc]) bc = c;
        }
        if (bc < 0) { ++st.room_unreach; continue; }
        e.gx = ((float)(bc % WIN) + 0.5f) * CELL - WIN_HALF; e.gy = ((float)(bc / WIN) + 0.5f) * CELL - WIN_HALF; e.gz = 0.f;
        e.path = dist[bc];
        // 목표 점에 몸통이 설 수 있는지(가구를 향한 yaw 로)
        if (!body_free_host(S, e, e.gx, e.gy, std::atan2(F.cy - e.wy - e.gy, F.cx - e.wx - e.gx))) { ++st.room_unreach; continue; }
        lst[L_ROOM][split].push_back(e);
        ++made;
        ++st.room_ok;
        st.path_sum[L_ROOM] += e.path;
        st.ratio_sum[L_ROOM] += e.path / std::max(0.1f, std::hypot(e.gx - e.sx, e.gy - e.sy));
      }
    }
  }
  // 엄격 판을 앞으로(물체 표), 그 안은 처음 순서 그대로
  for (int sp = 0; sp < 2; ++sp) {
    std::vector<Entry> a, b;
    for (size_t k = 0; k < lst[L_OBJ][sp].size(); ++k) (inner_flag[sp][k] ? a : b).push_back(lst[L_OBJ][sp][k]);
    st.ent_in[sp] = (int)a.size();
    a.insert(a.end(), b.begin(), b.end());
    lst[L_OBJ][sp] = a;
  }
  for (int l = 0; l < 2; ++l)
    for (int sp = 0; sp < 2; ++sp) {
      st.ent[l][sp] = (int)lst[l][sp].size();
      for (auto& e : lst[l][sp]) ents.push_back(e);
    }
  st.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return true;
}

}  // namespace

bool build_scenes(const BuildOpt& opt0, SceneBuild& out, std::string* err) {
  BuildOpt opt = opt0;
  if (opt.dir.empty()) opt.dir = default_rasc_dir();
  if (opt.vla.empty()) opt.vla = default_vla_dir();
  Names nm;
  if (!load_names(opt.vla, nm, out, err)) return false;
  // 장면 파일: 폴더의 *.rasc, RASC scene_index 순
  std::vector<std::pair<int, std::string>> files;
  DIR* dp = opendir(opt.dir.c_str());
  if (!dp) { *err = "cannot open " + opt.dir; return false; }
  while (dirent* de = readdir(dp)) {
    const std::string n = de->d_name;
    if (n.size() < 6 || n.substr(n.size() - 5) != ".rasc") continue;
    const std::string base = n.substr(0, n.size() - 5);
    if (!opt.only.empty() && std::find(opt.only.begin(), opt.only.end(), base) == opt.only.end()) continue;
    rasc::Scene s;
    std::string e2;
    if (!rasc::load(s, (opt.dir + "/" + n).c_str(), &e2)) { *err = n + ": " + e2; closedir(dp); return false; }
    files.push_back({(int)s.h->scene_index, opt.dir + "/" + n});
  }
  closedir(dp);
  std::sort(files.begin(), files.end());
  if (files.empty() || files.size() > (size_t)MAXSC) { *err = "no RASC scenes (or too many) in " + opt.dir; return false; }
  const int ns = (int)files.size();
  out.sc.assign(ns, SceneBuild::Sc{});
  out.stats.assign(ns, SceneStats{});
  std::vector<std::vector<Entry>> ents(ns);
  std::vector<std::string> errs(ns);
  std::vector<int> ok(ns, 0);
  std::vector<std::thread> th;
  for (int i = 0; i < ns; ++i)
    th.emplace_back([&, i] { ok[i] = build_one(files[i].second, nm, opt, out.sc[i], ents[i], out.stats[i], i, &errs[i]); });
  for (auto& t : th) t.join();
  for (int i = 0; i < ns; ++i) if (!ok[i]) { *err = files[i].second + ": " + errs[i]; return false; }
  // 표 이어 붙이기·범위
  SceneSet& H = out.host;
  H = SceneSet{};
  H.nsc = ns;
  out.ent.clear();
  for (int i = 0; i < ns; ++i) {
    size_t k = 0;
    for (int l = 0; l < 2; ++l)
      for (int sp = 0; sp < 2; ++sp) {
        H.loff[i][l][sp] = (int)(out.ent.size() + k);
        H.lcnt[i][l][sp] = out.stats[i].ent[l][sp];
        H.lcnt_in[i][l][sp] = l == L_OBJ ? out.stats[i].ent_in[sp] : out.stats[i].ent[l][sp];
        k += out.stats[i].ent[l][sp];
      }
    out.ent.insert(out.ent.end(), ents[i].begin(), ents[i].end());
  }
  for (int i = 0; i < ns; ++i) {
    SceneBuild::Sc& S = out.sc[i];
    S.d.bstart = S.bstart.data(); S.d.bitem = S.bitem.data(); S.d.box = S.box.data(); S.d.bkind = S.bkind.data(); S.d.bname = S.bname.data();
    S.d.room = S.room.data(); S.d.occ = S.occ.data(); S.d.rtype = S.rtype.data(); S.d.door = S.door.data();
    H.sc[i] = S.d;
  }
  H.ent = out.ent.data(); H.nent = (int)out.ent.size();
  H.conf1 = out.conf1.data(); H.conf2 = out.conf2.data(); H.sim3 = out.sim3.data(); H.hyper = out.hyper.data(); H.nname = out.nname;
  if (H.nent == 0) { *err = "no entries built"; return false; }
  return true;
}

bool upload(SceneBuild& b, std::string* err) {
  free_dev(b);
  auto put = [&](const void* src, size_t n) -> void* {
    void* p = nullptr;
    if (n == 0) n = 4;
    if (cudaMalloc(&p, n) != cudaSuccess) return nullptr;
    if (src) cudaMemcpy(p, src, n, cudaMemcpyHostToDevice);
    b.dbuf.push_back(p);
    b.dev_bytes += n;
    return p;
  };
  SceneSet D = b.host;
  for (int i = 0; i < D.nsc; ++i) {
    SceneBuild::Sc& S = b.sc[i];
    SceneDev& d = D.sc[i];
    d.bstart = (const uint32_t*)put(S.bstart.data(), S.bstart.size() * 4);
    d.bitem = (const uint16_t*)put(S.bitem.data(), S.bitem.size() * 2);
    d.box = (const SBox*)put(S.box.data(), S.box.size() * sizeof(SBox));
    d.bkind = (const uint8_t*)put(S.bkind.data(), S.bkind.size());
    d.bname = (const int16_t*)put(S.bname.data(), S.bname.size() * 2);
    d.room = (const uint8_t*)put(S.room.data(), S.room.size());
    d.occ = (const uint32_t*)put(S.occ.data(), S.occ.size() * 4);
    d.rtype = (const int8_t*)put(S.rtype.data(), S.rtype.size());
    d.door = (const SDoor*)put(S.door.data(), S.door.size() * sizeof(SDoor));
  }
  D.ent = (const Entry*)put(b.ent.data(), b.ent.size() * sizeof(Entry));
  D.conf1 = (const float*)put(b.conf1.data(), b.conf1.size() * 4);
  D.conf2 = (const float*)put(b.conf2.data(), b.conf2.size() * 4);
  D.sim3 = (const int16_t*)put(b.sim3.data(), b.sim3.size() * 2);
  D.hyper = (const int16_t*)put(b.hyper.data(), b.hyper.size() * 2);
  b.dev = (SceneSet*)put(&D, sizeof D);
  for (void* p : b.dbuf) if (!p) { *err = "cudaMalloc failed"; return false; }
  return cudaDeviceSynchronize() == cudaSuccess;
}
void free_dev(SceneBuild& b) {
  for (void* p : b.dbuf) if (p) cudaFree(p);
  b.dbuf.clear();
  b.dev = nullptr;
  b.dev_bytes = 0;
}

std::string stats_text(const SceneBuild& b) {
  std::string o;
  char buf[512];
  std::snprintf(buf, sizeof buf, "%-26s %5s %4s %4s %4s %7s %6s %6s | cut ground | %5s %5s %5s | obj ok/try (strict) rej start/win/unreach/stance/cap | room ok/try nocand unreach | entries obj tr/ev (strict) room tr/ev | path obj/room m (ratio) | s\n",
                "scene", "grid", "box", "room", "door", "free%", "iou", "agree", "inst", "pick", "reach");
  o += buf;
  for (auto& st : b.stats) {
    const int no = st.ent[L_OBJ][0] + st.ent[L_OBJ][1], nr = st.ent[L_ROOM][0] + st.ent[L_ROOM][1];
    std::snprintf(buf, sizeof buf,
                  "%-26s %5d %4d %4d %4d %6.1f%% %6.3f %6.3f | %3d %4d | %5d %5d %5d | %d/%d (%d) %d/%d/%d/%d/%d | %d/%d %d %d | %d/%d (%d/%d) %d/%d | %.2f (%.2f) / %.2f (%.2f) | %.1f\n",
                  st.name.c_str(), st.W, st.nbox, st.nroom, st.ndoor, 100.0 * st.free_cells / std::max(1L, st.cells), st.occ_iou, st.occ_agree, st.door_cuts, st.ground, st.inst,
                  st.inst_pick, st.inst_pick_reach, st.obj_ok, st.obj_try, st.obj_inner, st.rej_start, st.rej_window, st.rej_unreach, st.rej_stance, st.rej_cap,
                  st.room_ok, st.room_try, st.room_nocand, st.room_unreach, st.ent[L_OBJ][0], st.ent[L_OBJ][1], st.ent_in[0], st.ent_in[1], st.ent[L_ROOM][0],
                  st.ent[L_ROOM][1], no ? st.path_sum[L_OBJ] / no : 0.0, no ? st.ratio_sum[L_OBJ] / no : 0.0, nr ? st.path_sum[L_ROOM] / nr : 0.0,
                  nr ? st.ratio_sum[L_ROOM] / nr : 0.0, st.secs);
    o += buf;
  }
  std::snprintf(buf, sizeof buf, "entries %zu (%.1f MB), names %d\n", b.ent.size(), b.ent.size() * sizeof(Entry) / 1e6, b.nname);
  o += buf;
  return o;
}

}  // namespace bsc
