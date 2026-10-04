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
#include <tuple>
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

struct TObj { int k; float lo[3], hi[3]; int16_t name; uint32_t flags; float yaw, dim[3]; };   // dim·yaw: 바닥 자국을 가장 작게 덮는 회전 상자(E6 물체 모형)
struct PE { Entry e; bool in; std::vector<uint32_t> rb, rb_in; };   // 집기·놓기 엔트리 후보(엄격 지남, 창 닿는 칸 비트)   // 인스턴스의 과제 물체(세계 AABB, 과제 안 번호 k)

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
std::string default_pnp_dir() {
  std::string f = __FILE__;
  const size_t p = f.rfind("/training/RL/env/");
  return p == std::string::npos ? "training/data/pnp_v1" : f.substr(0, p) + "/training/data/pnp_v1";
}
PnpFilter pnp_filter_default() {   // 문서 CURRICULUM_BEHAVIOR2026.md B3–B5 거르개 표(E0 5.3절) — b1kconv limits_default 와 같은 값
  PnpFilter f{};
  f.pick_z[0] = 0.50f; f.pick_z[1] = 0.45f;
  f.place_top[0] = 0.52f; f.place_top[1] = 0.48f;
  f.max_mass[0] = 0.40f; f.max_mass[1] = 0.25f;
  f.max_w[0] = 0.06f; f.max_w[1] = 0.04f;
  f.threshold[0] = 0.025f; f.threshold[1] = 0.02f;
  f.topdown_z = 0.25f; f.edge_dist = 0.10f;
  f.reach_low = std::max(0.11f + 0.27f, 0.21f + 0.17f);   // max(옆 가장자리 0.11 + 옆 0.27, 앞 0.21 + 앞 0.17) = 0.38
  f.reach_high = 0.11f + 0.10f;                           // 옆 가장자리 0.11 + edge_dist 0.10 = 0.21
  f.inside_margin = 0.05f; f.min_side = 0.15f; f.min_top = 0.05f;
  f.free_margin = 0.02f; f.stance_r = 0.6f; f.floor_spot_r0 = 0.6f; f.floor_spot_r1 = 4.0f;
  return f;
}
bool dump_combos(const SceneBuild& b, const std::string& path, std::string* err) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) { *err = "cannot write " + path; return false; }
  std::fprintf(f, "# combos for the pick-and-place instruction table: idx obj_row src_row dst_row rel (-2 = floor, <= -1000 = floor of room category -1000-row; rel 2 ontop, 3 inside). en names after '|'\n");
  for (size_t k = 0; k < b.combos.size(); ++k) {
    const auto& c = b.combos[k];
    auto nm = [&](int r) {
      if (r <= -1000) { for (auto& sc : b.sc) { auto it = sc.sem_name.find(-1000 - r); if (it != sc.sem_name.end()) return it->second + " floor"; } return std::string("floor"); }
      return r == -2 ? std::string("floor") : (r >= 0 && r < (int)b.name_en.size() ? b.name_en[r] : std::string("?"));
    };
    std::fprintf(f, "%zu %d %d %d %d | %s | %s | %s\n", k, c.obj, c.src, c.dst, c.rel, nm(c.obj).c_str(), nm(c.src).c_str(), nm(c.dst).c_str());
  }
  std::fclose(f);
  return true;
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

void disk_field(const SceneBuild::Sc& s, const Entry& e, float tx, float ty, float rad, std::vector<float>& dist) {
  const int c0 = (int)std::lround((e.wx - WIN_HALF - s.d.ox) / CELL), r0 = (int)std::lround((e.wy - WIN_HALF - s.d.oy) / CELL);
  std::vector<std::pair<int, float>> seeds;
  for (int c = 0; c < WIN * WIN; ++c) {
    const float cx = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF, cy = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF;
    if ((cx - tx) * (cx - tx) + (cy - ty) * (cy - ty) <= rad * rad) seeds.push_back({c, 0.f});
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

void goal_field(const SceneBuild::Sc& s, const Entry& e, std::vector<float>& dist, const float* pt) {
  const int c0 = (int)std::lround((e.wx - WIN_HALF - s.d.ox) / CELL), r0 = (int)std::lround((e.wy - WIN_HALF - s.d.oy) / CELL);
  std::vector<std::pair<int, float>> seeds;
  for (int c = 0; c < WIN * WIN; ++c) {
    const int i = c % WIN, j = c / WIN, sc = c0 + i, sr = r0 + j;
    if (sc < 0 || sr < 0 || sc >= s.d.W || sr >= s.d.H || !s.freeg[(size_t)sr * s.d.W + sc]) continue;
    const float cx = ((float)i + 0.5f) * CELL - WIN_HALF, cy = ((float)j + 0.5f) * CELL - WIN_HALF;
    if (pt) {   // 점(창 좌표): 점에서 0.20–0.35 m 칸(몸통 옆에 점을 두고 팔이 닿는 거리 — 대본 정책용, 가정)
      const float d = std::hypot(cx - pt[0], cy - pt[1]);
      if (d >= 0.20f && d <= 0.35f) seeds.push_back({c, 0.f});
    } else if (e.list == L_ROOM) { if (std::fabs(cx - e.gx) < 1e-4f && std::fabs(cy - e.gy) < 1e-4f) seeds.push_back({c, 0.f}); }
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

bool build_one(const std::string& path, const Names& nm, const BuildOpt& opt, const PnpFilter& filt, SceneBuild::Sc& S, std::vector<Entry>& ents, SceneStats& st,
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
  int cur_obj = -1;
  auto push = [&](const SBox& b, uint8_t base, int16_t name) {
    S.bobj.push_back(cur_obj);
    uint8_t kind = base;
    if (b.z0 < H_COLL) kind |= BK_COLL;
    if (b.z1 > 0.05f && b.z0 < 0.50f) kind |= BK_BAND;
    S.box.push_back(b);
    S.bkind.push_back(kind);
    S.bname.push_back(name);
  };
  for (size_t k = 0; k < s.boxes.n; ++k) {
    const RascBoxRec& r = s.boxes[k];
    cur_obj = (int)r.obj;
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
  // 로봇 중심 칸 성분(8 이웃, 대각은 양 옆이 빈 칸일 때, 이웃 바닥 높이 차 ≤ 문턱) — 느슨·엄격 문턱(RASC LIMITS)
  auto label = [&](float thr, std::vector<uint16_t>& comp) {
    comp.assign((size_t)W * H, 0);
    const int tmm = (int)std::lround(thr * 1000.f);
    auto fz = [&](int i) { const int16_t v = S.floor_mm[i]; return v == INT16_MIN ? 0 : (int)v; };
    int n = 0;
    std::vector<int> q;
    const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    for (int i0 = 0; i0 < W * H; ++i0) {
      if (!S.freeg[i0] || comp[i0]) continue;
      if (n == 65535) { comp[i0] = 0; continue; }
      ++n;
      comp[i0] = (uint16_t)n;
      q.assign(1, i0);
      while (!q.empty()) {
        const int u = q.back();
        q.pop_back();
        const int uc = u % W, ur = u / W;
        for (int k = 0; k < 8; ++k) {
          const int nc = uc + dx[k], nr = ur + dy[k];
          if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;
          const int v = nr * W + nc;
          if (!S.freeg[v] || comp[v]) continue;
          if (k >= 4 && (!S.freeg[ur * W + nc] || !S.freeg[nr * W + uc])) continue;
          if (std::abs(fz(v) - fz(u)) > tmm) continue;
          comp[v] = (uint16_t)n;
          q.push_back(v);
        }
      }
    }
  };
  label(s.limits[0].threshold, S.comp);
  label(s.limits[1].threshold, S.comp_in);
  d.comp = S.comp.data();
  d.comp_in = S.comp_in.data();
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
  std::vector<std::vector<std::vector<PE>>> blocks[2];   // [split] 인스턴스 → 집을 물체 → 짝
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
      {   // 잡기 물리(E6): 물체 상자(회전)의 바닥 자국을 가장 작게 덮는 yaw 직사각형 — 후보 = 물체 축의 수평 투영 방향. 높이 = 세계 AABB
        t.yaw = 0.f; t.dim[0] = t.hi[0] - t.lo[0]; t.dim[1] = t.hi[1] - t.lo[1]; t.dim[2] = t.hi[2] - t.lo[2];
        float best = t.dim[0] * t.dim[1];
        for (int j = 0; j < 3; ++j) {
          const float hx = R[j], hy = R[3 + j];
          if (hx * hx + hy * hy < 0.09f) continue;
          const float yw = std::atan2(hy, hx), cu = std::cos(yw), su = std::sin(yw);
          float eu = 0.f, ev = 0.f;
          for (int q = 0; q < 3; ++q) {
            eu += 2.f * o.half[q] * std::fabs(R[q] * cu + R[3 + q] * su);
            ev += 2.f * o.half[q] * std::fabs(-R[q] * su + R[3 + q] * cu);
          }
          if (eu * ev < best - 1e-9f) { best = eu * ev; t.yaw = yw; t.dim[0] = eu; t.dim[1] = ev; }
        }
      }
      bool hit = false;
      t.name = (int16_t)name_of_synset(nm, s.str(o.syn), &hit);
      t.flags = o.flags;
      if (hit) ++st.name_hit; else ++st.name_miss;
      tobjs.push_back(t);
    }
    const float sxw = in.robot_pos[0], syw = in.robot_pos[1], syaw = in.robot_yaw;
    // ---- 집기·놓기 표(B2–B5, 문서 B3–B5 거르개) ----
    // RASC PICKS(바깥 = 느슨 한도를 지난 집을 물체, PK_INNER = 엄격)·PAIRS(같은 성분)에 환경 거르개를 더한다: 관절체·닫힌 곳 안 빼기,
    // 창(12.8 m)에 집을 물체·놓을 곳이 들어감, 잡는 자세 칸(잡는 점 작업 공간 + 몸통 안 닿음), 놓을 곳 가장자리에 닿는 칸(같은 성분),
    // 놓을 면 빈 넓이, 대신 쓸 시작 칸. 엄격 = 엄격 한도(RASC INNER) + 엄격 문턱 성분까지 같음
    {
      const RascPnpRange& pr = s.pnp_ranges[ii];
      bool any_pick = false, any_reach = false;
      std::vector<std::vector<PE>> per_pick;
      SceneSet tmp{};
      tmp.sc[scene_idx] = d;
      for (uint32_t q = 0; q < pr.n_pick; ++q) {
        const uint32_t pidx = pr.pick_off + q;
        const RascPickRec& pk = s.picks[pidx];
        if (!(pk.obj & RASC_PNP_TASKOBJ)) { ++st.pk_scene; continue; }   // 장면 물체 집을 것(v3 에 없음)
        any_pick = true;
        const int k = (int)(pk.obj & ~RASC_PNP_TASKOBJ) - (int)tk.obj_off;
        int ti = -1;
        for (size_t t = 0; t < tobjs.size(); ++t) if (tobjs[t].k == k) ti = (int)t;
        if (ti < 0) continue;
        ++st.pk_cand;
        const TObj& T = tobjs[ti];
        if (!(opt.nofilter & NF_ARTIC) && (T.flags & (RASC_F_ARTICULATED | RASC_F_FIXED_BASE))) { ++st.rj_artic; continue; }
        if (!(opt.nofilter & NF_IN_CLOSED) && (pk.flags & RASC_PK_IN_CLOSED)) { ++st.rj_closed; continue; }
        const bool pick_in = (pk.flags & RASC_PK_INNER) != 0;
        const float tcx = 0.5f * (T.lo[0] + T.hi[0]), tcy = 0.5f * (T.lo[1] + T.hi[1]);
        std::vector<PE> cand;
        // 잡는 자세 칸(장면 칸 번호, 물체 바닥 자국에서 거리 순): 물체 둘레 창(물체 가운데)에서 한 번 — 짝마다 같음
        struct StC { float df; int si; };
        std::vector<StC> stance;
        {
          Entry pe{};
          pe.scene = (int16_t)scene_idx;
          snap(tcx, tcy, pe.wx, pe.wy);
          pe.gx = tcx - pe.wx; pe.gy = tcy - pe.wy; pe.gz = 0.5f * (T.lo[2] + T.hi[2]);
          for (int a = 0; a < 3; ++a) pe.ext[a] = T.hi[a] - T.lo[a];
          BPrim& Q0 = pe.prim[0];
          for (int a = 0; a < 3; ++a) { Q0.lo[a] = T.lo[a] - (a == 0 ? pe.wx : a == 1 ? pe.wy : 0.f); Q0.hi[a] = T.hi[a] - (a == 0 ? pe.wx : a == 1 ? pe.wy : 0.f); }
          Q0.name = T.name; Q0.sbox = -1;
          pe.nprim = 1;
          add_prims(pe, tobjs, k, -1, pe.gx, pe.gy);
          const int pc0 = (int)std::lround((pe.wx - WIN_HALF - d.ox) / CELL), pr0 = (int)std::lround((pe.wy - WIN_HALF - d.oy) / CELL);
          const int rr = (int)std::ceil((filt.stance_r + 0.5f * std::max(pe.ext[0], pe.ext[1])) / CELL) + 1;
          for (int j = WIN / 2 - rr; j <= WIN / 2 + rr; ++j)
            for (int i2 = WIN / 2 - rr; i2 <= WIN / 2 + rr; ++i2) {
              const int sc = pc0 + i2, sr = pr0 + j;
              if (sc < 0 || sr < 0 || sc >= W || sr >= H || !S.comp[(size_t)sr * W + sc]) continue;
              const float x = ((float)i2 + 0.5f) * CELL - WIN_HALF, y = ((float)j + 0.5f) * CELL - WIN_HALF;
              const float df = dist_pt_rect(Q0.lo, Q0.hi, x, y);
              if (df > filt.stance_r) continue;
              if (!(opt.nofilter & NF_STANCE) && !stance_ok(S, pe, x, y)) continue;
              stance.push_back({df, sr * W + sc});
            }
          std::sort(stance.begin(), stance.end(), [](const StC& a, const StC& b2) { return a.df != b2.df ? a.df < b2.df : a.si < b2.si; });
        }
        if (stance.empty()) { ++st.rj_stance; continue; }
        for (uint32_t r2 = 0; r2 < pr.n_pair; ++r2) {
          const RascPairRec& pa = s.pairs[pr.pair_off + r2];
          if (pa.pick != pidx) continue;
          ++st.pr_cand;
          if (!(pa.reachable & 1)) { ++st.rj_reach; continue; }   // 집을 것·놓을 곳이 같은 TRAV 성분(느슨 문턱)
          const RascPlaceRec& D = s.places[pa.dst];
          Entry e{};
          e.scene = (int16_t)scene_idx; e.list = L_OBJ; e.split = (int16_t)split; e.inst = ii; e.task = (int16_t)in.task;
          e.dkind = (int16_t)D.kind; e.rel = pa.rel; e.pick_rec = (int32_t)pidx; e.dst_rec = (int32_t)pa.dst; e.src_rec = (int32_t)pk.src_place;
          e.dst_room = (int16_t)(D.room == RASC_NONE16 ? -1 : D.room);
          // 놓을 곳 상자(세계): 장면 물체면 그 정적 상자, 과제 물체면 그 상자
          int dbox = -1, dti = -1;
          float dlo[3] = {0, 0, 0}, dhi[3] = {0, 0, 0};
          if (D.kind != DK_FLOOR) {
            if (D.obj & RASC_PNP_TASKOBJ) {
              const int dk = (int)(D.obj & ~RASC_PNP_TASKOBJ) - (int)tk.obj_off;
              for (size_t t = 0; t < tobjs.size(); ++t) if (tobjs[t].k == dk) dti = (int)t;
              if (dti < 0) continue;
              for (int a = 0; a < 3; ++a) { dlo[a] = tobjs[dti].lo[a]; dhi[a] = tobjs[dti].hi[a]; }
              e.dst_name = tobjs[dti].name;
            } else {
              for (size_t bk = 0; bk < S.bobj.size() && dbox < 0; ++bk) if (S.bobj[bk] == (int)D.obj && (S.bkind[bk] & BK_FURN)) dbox = (int)bk;
              const RascObjRec& O = s.objs[D.obj];
              if (std::strstr(s.str(s.cats[O.cat].name), "baseboard")) { ++st.rj_struct; continue; }   // 구조물(걸레받이) 받침 빼기
              for (int a = 0; a < 3; ++a) { dlo[a] = O.aabb_min[a]; dhi[a] = O.aabb_max[a]; }
              e.dst_name = (int16_t)name_of_category(nm, s.str(s.cats[O.cat].name));
            }
          } else {   // 바닥: 그 방 종류(room_categories 줄 번호 sem)로 −1000 − sem (지시문 "the kitchen floor")
            e.dst_name = (int16_t)(D.room != RASC_NONE16 && D.room < s.rooms.n ? -1000 - (int)s.rooms[D.room].sem : -2);
            if (D.room != RASC_NONE16 && D.room < s.rooms.n) {
              std::string rn = s.str(s.rooms[D.room].name);
              const size_t u = rn.rfind('_');
              if (u != std::string::npos) rn = rn.substr(0, u);
              for (auto& ch : rn) if (ch == '_') ch = ' ';
              S.sem_name[s.rooms[D.room].sem] = rn;
            }
          }
          // 출발 받침 이름
          if (pk.src_place == RASC_NONE32) e.src_name = -2;
          else {
            const RascPlaceRec& Sp = s.places[pk.src_place];
            if (Sp.kind == DK_FLOOR) e.src_name = -2;
            else if (Sp.obj & RASC_PNP_TASKOBJ) {
              const int sk = (int)(Sp.obj & ~RASC_PNP_TASKOBJ) - (int)tk.obj_off;
              e.src_name = -2;
              for (size_t t = 0; t < tobjs.size(); ++t) if (tobjs[t].k == sk) e.src_name = tobjs[t].name;
            } else e.src_name = (int16_t)name_of_category(nm, s.str(s.cats[s.objs[Sp.obj].cat].name));
          }
          // 창: 면·용기 = 집을 것과 놓을 곳 가운데, 바닥 = 집을 것 가운데
          const float dcx = D.kind == DK_FLOOR ? tcx : D.center[0], dcy = D.kind == DK_FLOOR ? tcy : D.center[1];
          if (!fits(tcx, tcy, dcx, dcy)) { ++st.rj_win; continue; }
          snap(0.5f * (tcx + dcx), 0.5f * (tcy + dcy), e.wx, e.wy);
          e.gx = tcx - e.wx; e.gy = tcy - e.wy; e.gz = 0.5f * (T.lo[2] + T.hi[2]);
          for (int a = 0; a < 3; ++a) e.ext[a] = T.hi[a] - T.lo[a];
          e.groom = (int16_t)room_at(d, tcx, tcy);
          BPrim& P0 = e.prim[0];
          for (int a = 0; a < 3; ++a) { P0.lo[a] = T.lo[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); P0.hi[a] = T.hi[a] - (a == 0 ? e.wx : a == 1 ? e.wy : 0.f); }
          P0.name = T.name; P0.sbox = -1;
          e.nprim = 1;
          if (D.kind != DK_FLOOR) {
            BPrim& P1 = e.prim[e.nprim++];
            if (dbox >= 0) {
              float lo2[2], hi2[2];
              obb_aabb(S.box[dbox], lo2, hi2);
              P1.lo[0] = lo2[0] - e.wx; P1.lo[1] = lo2[1] - e.wy; P1.lo[2] = S.box[dbox].z0;
              P1.hi[0] = hi2[0] - e.wx; P1.hi[1] = hi2[1] - e.wy; P1.hi[2] = S.box[dbox].z1;
              P1.sbox = (int16_t)dbox;
            } else {
              P1.lo[0] = dlo[0] - e.wx; P1.lo[1] = dlo[1] - e.wy; P1.lo[2] = dlo[2];
              P1.hi[0] = dhi[0] - e.wx; P1.hi[1] = dhi[1] - e.wy; P1.hi[2] = dhi[2];
              P1.sbox = -1;
            }
            P1.name = e.dst_name;
            for (int a = 0; a < 3; ++a) { e.dlo[a] = P1.lo[a]; e.dhi[a] = P1.hi[a]; }
          }
          add_prims(e, tobjs, k, dbox, e.gx, e.gy);
          if (dti >= 0) {   // 과제 물체 놓을 곳이 add_prims 에 또 들어갔으면 빼기
            for (int p2 = 2; p2 < e.nprim; ++p2)
              if (e.prim[p2].sbox < 0 && e.prim[p2].lo[0] == e.prim[1].lo[0] && e.prim[p2].lo[1] == e.prim[1].lo[1] && e.prim[p2].hi[2] == e.prim[1].hi[2]) {
                for (int p3 = p2; p3 + 1 < e.nprim; ++p3) e.prim[p3] = e.prim[p3 + 1];
                --e.nprim;
                break;
              }
          }
          // 놓을 면 빈 넓이(면·용기): 받침 사각형 − 그 위(안)에 있는 물체 바닥 자국 ≥ (물체 가로 + 2m)(세로 + 2m)
          if (D.kind != DK_FLOOR && !(opt.nofilter & NF_FREE_AREA)) {
            const float cs = std::cos(D.yaw), sn = std::sin(D.yaw);
            const float area = 4.f * D.half[0] * D.half[1];
            float occ_a = 0.f;
            auto occupant = [&](const float lo[3], const float hi[3]) {
              const float ox = 0.5f * (lo[0] + hi[0]) - D.center[0], oy = 0.5f * (lo[1] + hi[1]) - D.center[1];
              const float lx = cs * ox + sn * oy, ly = -sn * ox + cs * oy;
              if (std::fabs(lx) > D.half[0] || std::fabs(ly) > D.half[1]) return;
              const bool on = D.kind == DK_ONTOP ? (lo[2] >= D.top - 0.05f && lo[2] <= D.top + 0.05f) : (lo[2] >= D.top - 0.6f && lo[2] < D.top);
              if (on) occ_a += std::min((hi[0] - lo[0]) * (hi[1] - lo[1]), area);
            };
            for (size_t t = 0; t < tobjs.size(); ++t) if ((int)t != ti && (int)t != dti) occupant(tobjs[t].lo, tobjs[t].hi);
            for (size_t ob = 0; ob < s.objs.n; ++ob) {
              const RascObjRec& O = s.objs[ob];
              if ((O.flags & (RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING | RASC_F_DOOR | RASC_F_WINDOW | RASC_F_CARPET)) || (int)ob == (int)D.obj) continue;
              occupant(O.aabb_min, O.aabb_max);
            }
            const float need = (e.ext[0] + 2.f * filt.free_margin) * (e.ext[1] + 2.f * filt.free_margin);
            if (area - occ_a < need) { ++st.rj_area; continue; }
          }
          // 놓을 곳 가장자리에 닿는 칸(같은 성분) — 잡는 자세 칸은 위에서(물체마다 한 번)
          const int c0 = (int)std::lround((e.wx - WIN_HALF - d.ox) / CELL), r0 = (int)std::lround((e.wy - WIN_HALF - d.oy) / CELL);
          auto cell_of = [&](int c) { const int sc = c0 + c % WIN, sr = r0 + c / WIN; return (sc < 0 || sr < 0 || sc >= W || sr >= H) ? -1 : sr * W + sc; };
          auto cxy = [&](int c, float& x, float& y) { x = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF; y = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF; };
          auto win_cell = [&](int si) { const int c = si % W - c0, r = si / W - r0; return (c < 0 || r < 0 || c >= WIN || r >= WIN) ? -1 : r * WIN + c; };
          const bool dst_high = D.kind != DK_FLOOR && D.top > filt.topdown_z;
          const float dreach = dst_high ? filt.reach_high : filt.reach_low;
          // 찾을 창 칸 범위(놓을 곳 상자 ± 닿는 거리, 바닥이면 집을 것 ± floor_spot_r1)
          int i0, i1, j0, j1;
          {
            float lo0, lo1, hi0, hi1;
            if (D.kind == DK_FLOOR) { lo0 = e.gx - filt.floor_spot_r1; hi0 = e.gx + filt.floor_spot_r1; lo1 = e.gy - filt.floor_spot_r1; hi1 = e.gy + filt.floor_spot_r1; }
            else { lo0 = e.prim[1].lo[0] - dreach; hi0 = e.prim[1].hi[0] + dreach; lo1 = e.prim[1].lo[1] - dreach; hi1 = e.prim[1].hi[1] + dreach; }
            i0 = std::max(0, (int)std::floor((lo0 + WIN_HALF) / CELL)); i1 = std::min(WIN - 1, (int)std::floor((hi0 + WIN_HALF) / CELL));
            j0 = std::max(0, (int)std::floor((lo1 + WIN_HALF) / CELL)); j1 = std::min(WIN - 1, (int)std::floor((hi1 + WIN_HALF) / CELL));
          }
          // 바닥 놓기: 그 방, 집을 것에서 floor_spot_r0–r1, 3 × 3 칸이 같은 성분인 칸, 해시 순서
          std::vector<int> spot_cells;
          if (D.kind == DK_FLOOR) {
            for (int j = j0; j <= j1; ++j)
              for (int i2 = i0; i2 <= i1; ++i2) {
                const int c = j * WIN + i2, si = cell_of(c);
                if (si < 0 || !S.comp[si] || (int)S.room[si] - 1 != (int)D.room) continue;
                float x, y;
                cxy(c, x, y);
                const float dd = std::hypot(x - e.gx, y - e.gy);
                if (dd < filt.floor_spot_r0 || dd > filt.floor_spot_r1 || !in_win(x, y, WIN_MARGIN)) continue;
                bool all = true;
                for (int a = -1; a <= 1 && all; ++a) for (int b2 = -1; b2 <= 1 && all; ++b2) { const int sj = si + a * W + b2; all = sj >= 0 && sj < W * H && S.comp[sj] == S.comp[si]; }
                if (all) spot_cells.push_back(c);
              }
            std::sort(spot_cells.begin(), spot_cells.end(), [&](int a, int b2) {
              const uint64_t ha = hmix(((uint64_t)pidx << 24) ^ (uint64_t)a), hb = hmix(((uint64_t)pidx << 24) ^ (uint64_t)b2);
              return ha != hb ? ha < hb : a < b2;
            });
          }
          // 창 안 BFS(로봇 중심 칸, 8 이웃·대각은 양 옆, 이웃 바닥 높이 차 ≤ 문턱) → 창 칸 비트
          auto wfree = [&](int i2, int j2) { const int c = j2 * WIN + i2; const int si = (i2 < 0 || j2 < 0 || i2 >= WIN || j2 >= WIN) ? -1 : cell_of(c); return si >= 0 && S.freeg[si]; };
          auto wfz = [&](int i2, int j2) { const int16_t v = S.floor_mm[cell_of(j2 * WIN + i2)]; return v == INT16_MIN ? 0 : (int)v; };
          auto bfs = [&](int start, float thr, std::vector<uint32_t>& bits) {
            bits.assign(WIN * WIN / 32, 0u);
            const int tmm = (int)std::lround(thr * 1000.f);
            std::vector<int> q(1, start);
            bits[start >> 5] |= 1u << (start & 31);
            const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
            for (size_t h = 0; h < q.size(); ++h) {
              const int u = q[h], ui = u % WIN, uj = u / WIN;
              for (int k2 = 0; k2 < 8; ++k2) {
                const int ni = ui + dx[k2], nj = uj + dy[k2], v = nj * WIN + ni;
                if (!wfree(ni, nj) || ((bits[v >> 5] >> (v & 31)) & 1u)) continue;
                if (k2 >= 4 && (!wfree(ni, uj) || !wfree(ui, nj))) continue;
                if (std::abs(wfz(ni, nj) - wfz(ui, uj)) > tmm) continue;
                bits[v >> 5] |= 1u << (v & 31);
                q.push_back(v);
              }
            }
          };
          auto has = [](const std::vector<uint32_t>& bits, int c) { return (bits[c >> 5] >> (c & 31)) & 1u; };
          auto dst_in = [&](const std::vector<uint32_t>& bits) -> int {   // 놓을 곳에 닿는 칸 하나(없으면 −1)
            if (D.kind == DK_FLOOR) {
              for (int c : spot_cells) if (has(bits, c)) return c;
              return -1;
            }
            for (int j = j0; j <= j1; ++j)
              for (int i2 = i0; i2 <= i1; ++i2) {
                const int c = j * WIN + i2;
                if (!has(bits, c)) continue;
                float x, y;
                cxy(c, x, y);
                const float df = dbox >= 0 ? dist_pt_obb(S.box[dbox], x + e.wx, y + e.wy) : dist_pt_rect(e.prim[1].lo, e.prim[1].hi, x, y);
                if (df > 0.f && df <= dreach) return c;
              }
            return -1;
          };
          int stc = -1, dc = -1, stc_in = -1, dc_in = -1;
          const bool want_in = pick_in && (D.flags & 1) && (pa.reachable & 2);
          std::vector<uint32_t> rb, rb_in, cur, tried(WIN * WIN / 32, 0u), tried_in(WIN * WIN / 32, 0u);
          for (auto& sc2 : stance) {
            const int wc = win_cell(sc2.si);
            if (wc < 0) continue;
            {   // 이 짝의 물체(놓을 곳이 과제 물체면 그것도)로 다시: 잡는 자세가 그대로 되는가
              float x, y;
              cxy(wc, x, y);
              if (!(opt.nofilter & NF_STANCE) && !stance_ok(S, e, x, y)) continue;
            }
            if (stc < 0 && !has(tried, wc)) {
              bfs(wc, filt.threshold[0], cur);
              const int d2 = (opt.nofilter & NF_DST_REACH) ? wc : dst_in(cur);
              if (d2 >= 0) { stc = wc; dc = d2; rb = cur; }
              else for (int w = 0; w < WIN * WIN / 32; ++w) tried[w] |= cur[w];
            }
            if (want_in && stc_in < 0 && !has(tried_in, wc)) {
              bfs(wc, filt.threshold[1], cur);
              const int d2 = dst_in(cur);
              if (d2 >= 0) { stc_in = wc; dc_in = d2; rb_in = cur; }
              else for (int w = 0; w < WIN * WIN / 32; ++w) tried_in[w] |= cur[w];
            }
            if (stc >= 0 && (!want_in || stc_in >= 0)) break;
          }
          if (stc < 0) { ++st.rj_stance; continue; }
          if (D.kind == DK_FLOOR && dc < 0) { ++st.rj_floor; continue; }
          const bool inner = want_in && stc_in >= 0;
          const int use_st = inner ? stc_in : stc, use_dc = inner ? dc_in : dc;
          e.comp = S.comp[cell_of(use_st)];
          e.comp_in = S.comp_in[cell_of(use_st)];
          e.fset = inner ? 1 : 0;
          // 잡기 물리(E6): 무게, 잡는 자세 칸·놓을 곳에 닿는 칸 가운데(창 좌표), 처음 받침 윗면
          e.mass = pk.mass;
          e.oyaw = T.yaw;
          for (int a2 = 0; a2 < 3; ++a2) e.odim[a2] = T.dim[a2];
          cxy(use_st, e.st[0], e.st[1]);
          if (use_dc >= 0) cxy(use_dc, e.dst_st[0], e.dst_st[1]); else { e.dst_st[0] = e.st[0]; e.dst_st[1] = e.st[1]; }
          e.src_top = pk.src_place != RASC_NONE32 ? s.places[pk.src_place].top : T.lo[2];
          if (D.kind == DK_FLOOR) {
            float x, y;
            cxy(use_dc, x, y);
            e.dlo[0] = x - 0.2f; e.dlo[1] = y - 0.2f; e.dlo[2] = 0.f;
            e.dhi[0] = x + 0.2f; e.dhi[1] = y + 0.2f; e.dhi[2] = 0.02f;
          }
          // 대신 쓸 시작: 창 칸을 해시 순열로, 환경과 같은 spawn_ok(엄격 판이면 엄격 비트로), yaw 4 개
          bool sp_ok = false;
          {
            std::vector<uint32_t> pool2 = rb;   // 임시 비트 묶음: 0 = 느슨, 512 = 엄격
            pool2.resize(2 * WIN * WIN / 32, 0u);
            if (inner) std::copy(rb_in.begin(), rb_in.end(), pool2.begin() + WIN * WIN / 32);
            tmp.rbits = pool2.data();
            e.rb = 0;
            e.rb_in = inner ? WIN * WIN / 32 : -1;
            const uint32_t a = (uint32_t)(hmix(((uint64_t)pidx << 20) ^ (uint64_t)pa.dst) | 1u) & (WIN * WIN - 1), bb = (uint32_t)hmix((uint64_t)pidx * 31u + pa.dst) & (WIN * WIN - 1);
            const std::vector<uint32_t>& mine = inner ? rb_in : rb;
            for (uint32_t q2 = 0; q2 < (uint32_t)(WIN * WIN) && !sp_ok; ++q2) {
              const int c = (int)((a * q2 + bb) & (WIN * WIN - 1));   // 2^14 의 전주기 순열(a 홀수)
              if (!has(mine, c)) continue;
              float x, y;
              cxy(c, x, y);
              for (int yk = 0; yk < 4 && !sp_ok; ++yk) {
                const float yaw = -kPi + kHalfPi * (float)yk + 0.3f;
                if (env::spawn_ok(tmp, e, x, y, yaw, e.fset, 0)) { e.sx = x; e.sy = y; e.syaw = yaw; sp_ok = true; }
              }
            }
            tmp.rbits = nullptr;
          }
          if (!sp_ok) { ++st.rj_spawn; continue; }
          // 놓을 점(목표 점, 2026-10-05 — VLA_INPUT 2.1, CURRICULUM_BEHAVIOR2026 3.2): 바닥 = 고른 자리 가운데(z 0), 면(ontop) = 윗면 위 점.
          // 면 점: 받침 사각형(물체 축) 안 0.05 m 격자를 해시 차례로, 물체 바닥 자국 반지름(가로·세로 큰 쪽 반 + free_margin)만큼 안쪽,
          // 그 면 위(바닥 높이 윗면 ± 0.05 m)에 있는 다른 물체 상자와 바닥 자국 정사각형이 안 겹치고, 닿는 칸(느슨 BFS 비트, 엄격 판이면 엄격 비트도)
          // 하나에서 점까지 ≤ 팔 닿는 거리(낮은 면 reach_low 0.38, 높은 면 reach_high 0.21 + edge_dist 0.10 = 0.31 — 가장자리 거르개(칸 → 가장자리 0.21)에
          // 옆 잡기 가장자리 안 거리(0.10)를 더함, 실행기 move_robot goal.rs REACH_HIGH 와 같은 값). 용기(inside) = 없음
          e.ppt_ok = 0;
          e.ppt[0] = e.ppt[1] = e.ppt[2] = 0.f;
          if (D.kind == DK_FLOOR) {
            cxy(use_dc, e.ppt[0], e.ppt[1]);
            e.ppt_ok = 1;
          } else if (D.kind == DK_ONTOP) {
            const float ro = 0.5f * std::max(e.ext[0], e.ext[1]) + filt.free_margin;
            const float pcs = std::cos(D.yaw), psn = std::sin(D.yaw);
            const float ax = D.half[0] - ro, ay = D.half[1] - ro;
            if (ax >= 0.f && ay >= 0.f) {
              const float G = 0.05f;
              const int nx = (int)std::floor(ax / G), ny = (int)std::floor(ay / G);
              std::vector<std::pair<uint64_t, int>> pc;
              for (int j = -ny; j <= ny; ++j)
                for (int i2 = -nx; i2 <= nx; ++i2) {
                  const int key = ((j + 512) << 10) | (i2 + 512);
                  pc.push_back({hmix(((uint64_t)pidx << 32) ^ ((uint64_t)pa.dst << 20) ^ (uint64_t)key), key});
                }
              std::sort(pc.begin(), pc.end());
              auto clear_at = [&](float wx2, float wy2) {   // 세계 점의 물체 바닥 자국 정사각형(반 ro)이 그 면 위 다른 물체와 안 겹침
                auto hit = [&](const float lo[3], const float hi[3]) {
                  if (!(lo[2] >= D.top - 0.05f && lo[2] <= D.top + 0.05f)) return false;
                  return lo[0] < wx2 + ro && hi[0] > wx2 - ro && lo[1] < wy2 + ro && hi[1] > wy2 - ro;
                };
                for (size_t t = 0; t < tobjs.size(); ++t) if ((int)t != ti && (int)t != dti && hit(tobjs[t].lo, tobjs[t].hi)) return false;
                for (size_t ob = 0; ob < s.objs.n; ++ob) {
                  const RascObjRec& O = s.objs[ob];
                  if ((O.flags & (RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING | RASC_F_DOOR | RASC_F_WINDOW | RASC_F_CARPET)) || (int)ob == (int)D.obj) continue;
                  if (hit(O.aabb_min, O.aabb_max)) return false;
                }
                return true;
              };
              const float preach = dst_high ? filt.reach_high + filt.edge_dist : filt.reach_low;
              auto reach_at = [&](const std::vector<uint32_t>& bits, float px, float py) {   // 창 좌표 점에 팔 닿는 거리 안 닿는 칸이 있나
                const int i0p = std::max(0, (int)std::floor((px - preach + WIN_HALF) / CELL)), i1p = std::min(WIN - 1, (int)std::floor((px + preach + WIN_HALF) / CELL));
                const int j0p = std::max(0, (int)std::floor((py - preach + WIN_HALF) / CELL)), j1p = std::min(WIN - 1, (int)std::floor((py + preach + WIN_HALF) / CELL));
                for (int j = j0p; j <= j1p; ++j)
                  for (int i2 = i0p; i2 <= i1p; ++i2) {
                    const int c = j * WIN + i2;
                    if (!has(bits, c)) continue;
                    float x, y;
                    cxy(c, x, y);
                    if (std::hypot(x - px, y - py) <= preach) return true;
                  }
                return false;
              };
              for (auto& q : pc) {
                const float lx = (float)((q.second & 1023) - 512) * G, ly = (float)((q.second >> 10) - 512) * G;
                const float wx2 = D.center[0] + pcs * lx - psn * ly, wy2 = D.center[1] + psn * lx + pcs * ly;
                if (!clear_at(wx2, wy2)) continue;
                const float px = wx2 - e.wx, py = wy2 - e.wy;
                if (!reach_at(rb, px, py) || (inner && !reach_at(rb_in, px, py))) continue;
                e.ppt[0] = px; e.ppt[1] = py; e.ppt[2] = D.top;
                e.ppt_ok = 1;
                break;
              }
            }
          }
          st.ppt_ok += e.ppt_ok;
          st.ppt_onto += (D.kind == DK_ONTOP);
          st.ppt_onto_ok += (D.kind == DK_ONTOP) && e.ppt_ok;
          cand.push_back({e, inner, rb, rb_in});
        }
        if (cand.empty()) continue;
        any_reach = true;
        // 엄격 먼저, 그 안은 해시 순서, 상한
        std::stable_sort(cand.begin(), cand.end(), [&](const PE& a, const PE& b2) {
          if (a.in != b2.in) return a.in;
          return hmix((uint64_t)a.e.dst_rec * 977u + pidx) < hmix((uint64_t)b2.e.dst_rec * 977u + pidx);
        });
        if ((int)cand.size() > opt.cap_pairs) { st.rj_cap += (int)cand.size() - opt.cap_pairs; cand.resize(opt.cap_pairs); }
        per_pick.push_back(cand);
      }
      st.inst_pick += any_pick;
      st.inst_pick_reach += any_reach;
      if (!per_pick.empty()) {
        // 엄격 짝이 있는 집을 물체를 앞에
        std::stable_sort(per_pick.begin(), per_pick.end(), [](const std::vector<PE>& a, const std::vector<PE>& b2) { return a[0].in && !b2[0].in; });
        blocks[split].push_back(std::move(per_pick));
      }
    }
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
  // 집기·놓기 표 이어 붙이기: split 마다 엄격 가능한 인스턴스 먼저(그 안은 해시 순서)
  for (int sp = 0; sp < 2; ++sp) {
    std::stable_sort(blocks[sp].begin(), blocks[sp].end(), [](const std::vector<std::vector<PE>>& a, const std::vector<std::vector<PE>>& b2) {
      return a[0][0].in && !b2[0][0].in;
    });
    for (auto& blk : blocks[sp]) {
      PnpInst I{};
      I.pick_off = (int)S.ppick.size(); I.scene = scene_idx; I.split = sp;
      for (auto& v : blk) {
        PnpPick P{};
        P.ent_off = (int)lst[L_OBJ][sp].size();   // 장면·split 안 번호(build_scenes 가 고침)
        for (auto& pe : v) {
          Entry e2 = pe.e;
          e2.rb = (int)S.rbits.size();                          // 장면 안 자리(build_scenes 가 고침)
          S.rbits.insert(S.rbits.end(), pe.rb.begin(), pe.rb.end());
          e2.rb_in = -1;
          if (pe.in) { e2.rb_in = (int)S.rbits.size(); S.rbits.insert(S.rbits.end(), pe.rb_in.begin(), pe.rb_in.end()); }
          lst[L_OBJ][sp].push_back(e2);
          P.n_in += pe.in; ++st.pr_ok; st.pr_ok_in += pe.in;
        }
        P.n = (int)v.size();
        S.ppick.push_back(P);
        ++I.npick;
        I.npick_in += P.n_in > 0;
        ++st.pk_ok;
        st.pk_ok_in += P.n_in > 0;
      }
      S.pinst.push_back(I);
      S.pinst_split.push_back((uint8_t)sp);
    }
  }
  // 집기·놓기: 받침 수(서로 다른 RASC 받침), 엄격
  {
    std::vector<int> seen(s.places.n, 0);
    for (int sp = 0; sp < 2; ++sp)
      for (auto& e : lst[L_OBJ][sp]) { seen[e.dst_rec] |= 1 | (e.fset ? 2 : 0); }
    for (int v : seen) { st.sup_ok += (v & 1) != 0; st.sup_ok_in += (v & 2) != 0; }
    for (auto& I : S.pinst) { ++st.inst_ok; st.inst_ok_in += I.npick_in > 0; }
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
  // 거르개 표: RASC LIMITS(b1kconv 가 쓴 값)를 읽고, 모든 장면이 같은지·문서 표(기본값)와 같은지 본다
  out.filt = pnp_filter_default();
  {
    bool first = true;
    for (auto& f : files) {
      rasc::Scene s2;
      std::string e2;
      rasc::load(s2, f.second.c_str(), &e2);
      PnpFilter g = out.filt;
      for (int k = 0; k < 2; ++k) {
        const RascLimitsRec& L = s2.limits[k];
        g.pick_z[k] = L.pick_z; g.place_top[k] = L.place_top; g.max_mass[k] = L.max_mass; g.max_w[k] = L.max_w; g.threshold[k] = L.threshold;
      }
      const RascLimitsRec& L0 = s2.limits[0];
      g.topdown_z = L0.topdown_z; g.edge_dist = L0.edge_dist; g.inside_margin = L0.inside_margin; g.min_side = L0.min_side; g.min_top = L0.min_top;
      g.reach_low = std::max(L0.edge_side + L0.reach_side, L0.edge_front + L0.reach_front);
      g.reach_high = L0.edge_side + L0.edge_dist;
      if (first) {
        const PnpFilter dflt = pnp_filter_default();
        if (std::memcmp(&g, &dflt, sizeof g) && !opt.quiet) std::fprintf(stderr, "bscene: RASC LIMITS differ from the documented filter table defaults — using the RASC values\n");
        out.filt = g;
        first = false;
      } else if (std::memcmp(&g, &out.filt, sizeof g)) {
        *err = f.second + ": LIMITS differ between scenes";
        return false;
      }
    }
  }
  if (files.empty() || files.size() > (size_t)MAXSC) { *err = "no RASC scenes (or too many) in " + opt.dir; return false; }
  const int ns = (int)files.size();
  out.sc.assign(ns, SceneBuild::Sc{});
  out.stats.assign(ns, SceneStats{});
  std::vector<std::vector<Entry>> ents(ns);
  std::vector<std::string> errs(ns);
  std::vector<int> ok(ns, 0);
  std::vector<std::thread> th;
  for (int i = 0; i < ns; ++i)
    th.emplace_back([&, i] { ok[i] = build_one(files[i].second, nm, opt, out.filt, out.sc[i], ents[i], out.stats[i], i, &errs[i]); });
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
    S.d.comp = S.comp.data(); S.d.comp_in = S.comp_in.data();
    H.sc[i] = S.d;
  }
  // 집기·놓기 표 이어 붙이기(장면 안 번호 → 전체 번호), [장면][split] 인스턴스 범위
  out.pinst.clear();
  out.ppick.clear();
  for (int i = 0; i < ns; ++i) {
    SceneBuild::Sc& S = out.sc[i];
    for (int sp = 0; sp < 2; ++sp) { H.ioff[i][sp] = (int)out.pinst.size(); H.icnt[i][sp] = 0; H.icnt_in[i][sp] = 0; }
    for (int sp = 0; sp < 2; ++sp) {
      H.ioff[i][sp] = (int)out.pinst.size();
      for (size_t k = 0; k < S.pinst.size(); ++k) {
        if (S.pinst_split[k] != sp) continue;
        PnpInst I = S.pinst[k];
        const int po = (int)out.ppick.size();
        for (int q = 0; q < I.npick; ++q) {
          PnpPick P = S.ppick[I.pick_off + q];
          P.ent_off += H.loff[i][L_OBJ][sp];
          out.ppick.push_back(P);
        }
        I.pick_off = po;
        out.pinst.push_back(I);
        ++H.icnt[i][sp];
        H.icnt_in[i][sp] += I.npick_in > 0;
      }
    }
  }
  H.pinst = out.pinst.data();
  H.ppick = out.ppick.data();
  out.rbits.clear();
  {
    size_t k = 0;
    for (int i = 0; i < ns; ++i) {
      const int base = (int)out.rbits.size();
      out.rbits.insert(out.rbits.end(), out.sc[i].rbits.begin(), out.sc[i].rbits.end());
      const size_t n_i = ents[i].size();
      for (size_t q = 0; q < n_i; ++q) {
        Entry& e = out.ent[k + q];
        if (e.list != L_OBJ) { e.rb = -1; e.rb_in = -1; continue; }
        e.rb += base;
        if (e.rb_in >= 0) e.rb_in += base;
      }
      k += n_i;
    }
  }
  H.rbits = out.rbits.data();
  // 지시문 조합: (집을 것, 출발, 놓을 곳, 술어) 이름 행. pnp_v1/combos.tsv 가 있으면 그 번호(지시문 표 행 = 조합 × ntpl + 문장)
  {
    std::map<std::tuple<int, int, int, int>, int> key;
    for (auto& e : out.ent)
      if (e.list == L_OBJ) key[std::make_tuple((int)e.prim[0].name, (int)e.src_name, (int)e.dst_name, (int)e.rel)] = 0;
    out.combos.clear();
    for (auto& kv : key) out.combos.push_back(SceneBuild::Combo{(int16_t)std::get<0>(kv.first), (int16_t)std::get<1>(kv.first), (int16_t)std::get<2>(kv.first), (int16_t)std::get<3>(kv.first)});
    const std::string pd = opt.pnp_dir.empty() ? default_pnp_dir() : opt.pnp_dir;
    std::map<std::tuple<int, int, int, int>, int> tab;
    FILE* f = std::fopen((pd + "/combos.tsv").c_str(), "r");
    out.ntpl = 0; out.ntpl_train = 0; out.ncombo = 0; out.iblocks = 1;
    if (f) {
      char line[512];
      while (std::fgets(line, sizeof line, f)) {
        int idx, a, b2, c, r;
        if (line[0] == '#') {
          if (std::sscanf(line, "# ntpl %d ntpl_train %d", &out.ntpl, &out.ntpl_train) == 2) out.iblocks = std::strstr(line, "blocks combo,point,goto") ? 3 : 1;
          continue;
        }
        if (std::sscanf(line, "%d %d %d %d %d", &idx, &a, &b2, &c, &r) == 5) { tab[std::make_tuple(a, b2, c, r)] = idx; out.ncombo = std::max(out.ncombo, idx + 1); }
      }
      std::fclose(f);
    }
    out.combo_missing = 0;
    for (auto& e : out.ent) {
      if (e.list != L_OBJ) { e.combo = -1; continue; }
      auto it = tab.find(std::make_tuple((int)e.prim[0].name, (int)e.src_name, (int)e.dst_name, (int)e.rel));
      e.combo = (it == tab.end() || out.ntpl <= 0) ? -1 : it->second;
      out.combo_missing += e.combo < 0;
    }
    H.ntpl = out.ntpl; H.ntpl_train = out.ntpl_train; H.ncombo = out.ncombo; H.iblocks = out.iblocks;
  }
  for (int i = 0; i < ns; ++i)
    for (int sp = 0; sp < 2; ++sp) {   // 물체 표의 엄격 판 수(통계)
      int n = 0;
      for (int k = 0; k < H.lcnt[i][L_OBJ][sp]; ++k) n += out.ent[H.loff[i][L_OBJ][sp] + k].fset;
      H.lcnt_in[i][L_OBJ][sp] = n;
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
    d.comp = (const uint16_t*)put(S.comp.data(), S.comp.size() * 2);
    d.comp_in = (const uint16_t*)put(S.comp_in.data(), S.comp_in.size() * 2);
  }
  D.pinst = (const PnpInst*)put(b.pinst.data(), b.pinst.size() * sizeof(PnpInst));
  D.rbits = (const uint32_t*)put(b.rbits.data(), b.rbits.size() * 4);
  D.ppick = (const PnpPick*)put(b.ppick.data(), b.ppick.size() * sizeof(PnpPick));
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
  o += "pick-and-place filter (doc B3-B5 table; loose / strict): instances passing, objects (instance x object), supports (distinct), pairs | candidates and rejections\n";
  for (auto& st : b.stats) {
    std::snprintf(buf, sizeof buf,
                  "  %-26s inst %5d / %5d (of %5d) | objects %5d / %5d (cand %5d, scene-level %d) | supports %5d / %5d | pairs %6d / %6d (cand %6d) | rej artic %d closed %d struct %d comp %d window %d stance %d floor %d area %d spawn %d cap %d\n",
                  st.name.c_str(), st.inst_ok, st.inst_ok_in, st.inst, st.pk_ok, st.pk_ok_in, st.pk_cand, st.pk_scene, st.sup_ok, st.sup_ok_in, st.pr_ok, st.pr_ok_in,
                  st.pr_cand, st.rj_artic, st.rj_closed, st.rj_struct, st.rj_reach, st.rj_win, st.rj_stance, st.rj_floor, st.rj_area, st.rj_spawn, st.rj_cap);
    o += buf;
  }
  for (auto& st : b.stats) {
    std::snprintf(buf, sizeof buf, "  %-26s place points (before pair cap): pairs with a point %d, ontop pairs %d, ontop with a point %d (floor pairs always have one)\n",
                  st.name.c_str(), st.ppt_ok, st.ppt_onto, st.ppt_onto_ok);
    o += buf;
  }
  std::snprintf(buf, sizeof buf, "  instruction combos %zu (table %d), entries without an instruction row %d (ntpl %d, train %d, blocks %d)\n", b.combos.size(), b.ncombo, b.combo_missing, b.ntpl, b.ntpl_train, b.iblocks);
  o += buf;
  return o;
}

}  // namespace bsc
