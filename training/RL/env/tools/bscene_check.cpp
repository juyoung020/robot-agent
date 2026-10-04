// BEHAVIOR 장면 묶음 확인(CPU, E2 "장면 바름"): bscene_host 로 묶음을 만들고 표를 다른 길로 다시 잰다.
//   bscene_check [RASC 폴더] [--only 장면,...]
// 1. 시작 자세: 몸통(환경과 같은 SAT·sin/cos)이 정적 상자·과제 물체 상자에 안 닿음 — 모든 판
// 2. 닿음: 창 안 다익스트라(같은 칸 규칙)로 목표 쪽 칸까지 길이 Entry::path 와 같음 + 독립 확인 — RASC TRAV_OPEN_DOOR 칸만으로 4 이웃 BFS(상자와
//    무관한 BEHAVIOR 의 다닐 곳)에서 시작 칸 → 목표 쪽 칸 이어짐 비율
// 3. 방: B1 목표 점의 방 = Entry::groom 을 RASC 로더(rasc.h room_at)로 다시 봄, 지도 물체 0 의 방
// 4. 창: 시작·목표·지도 물체 가운데가 창 안
// 5. 이름 확신도 표: 시뮬 6 행의 conf1·conf2 가 vla_vocab.h(파이썬 생성)와 같은지(오차)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>

#include "bscene_host.h"
#include "env_api.h"
#include "rasc.h"
#include "../../map/include/vla_vocab.h"

using namespace bsc;

int main(int argc, char** argv) {
  BuildOpt opt;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--only") && a + 1 < argc) {
      std::string v = argv[++a];
      size_t p = 0;
      while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); opt.only.push_back(v.substr(p, q - p)); p = q + 1; }
    } else opt.dir = argv[a];
  }
  if (opt.dir.empty()) opt.dir = default_rasc_dir();
  SceneBuild b;
  std::string err;
  if (!build_scenes(opt, b, &err)) { std::printf("build failed: %s\n", err.c_str()); return 1; }
  std::printf("%s", stats_text(b).c_str());
  long bad = 0;
  // 5. 확신도 표
  {
    double m1 = 0, m2 = 0;
    for (int a = 0; a < 7; ++a)
      for (int c = 0; c < 6; ++c) {
        const int n = vlav::kSimName[c];
        m1 = std::max(m1, (double)std::fabs(b.conf1[(size_t)a * b.nname + n] - vlav::kConf1[a][c]));
        m2 = std::max(m2, (double)std::fabs(b.conf2[(size_t)a * b.nname + n] - vlav::kConf2[a][c]));
      }
    std::printf("name confidence table (C++ from name128/app128) vs vla_vocab.h sim rows: max |d conf1| %.2e, |d conf2| %.2e\n", m1, m2);
    if (m1 > 1e-4 || m2 > 1e-4) ++bad;
  }
  for (int si = 0; si < b.host.nsc; ++si) {
    const SceneBuild::Sc& S = b.sc[si];
    rasc::Scene rs;
    if (!rasc::load(rs, (opt.dir + "/" + S.name + ".rasc").c_str(), &err)) { std::printf("%s\n", err.c_str()); return 1; }
    long n = 0, start_bad = 0, path_bad = 0, room_bad = 0, win_bad = 0, trav_ok = 0, trav_n = 0, goal_free_bad = 0;
    double path_err = 0;
    std::vector<float> dist;
    for (int l = 0; l < 2; ++l)
      for (int sp = 0; sp < 2; ++sp)
        for (int k = 0; k < b.host.lcnt[si][l][sp]; ++k) {
          const Entry& e = b.ent[b.host.loff[si][l][sp] + k];
          ++n;
          if (!body_free_host(S, e, e.sx, e.sy, e.syaw)) ++start_bad;
          window_dijkstra(S, e.wx, e.wy, e.sx, e.sy, dist);
          float best = -1.f;
          int bc = -1;
          for (int c = 0; c < WIN * WIN; ++c) {
            if (dist[c] < 0.f) continue;
            const float cx = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF, cy = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF;
            bool goal;
            if (l == L_ROOM) goal = std::fabs(cx - e.gx) < 1e-4f && std::fabs(cy - e.gy) < 1e-4f;
            else {
              const BPrim& P = e.prim[0];
              const float ex = std::max(std::max(P.lo[0] - cx, cx - P.hi[0]), 0.f), ey = std::max(std::max(P.lo[1] - cy, cy - P.hi[1]), 0.f);
              goal = std::sqrt(ex * ex + ey * ey) <= 0.38f;
            }
            if (goal && (best < 0.f || dist[c] < best)) { best = dist[c]; bc = c; }
          }
          if (best < 0.f || std::fabs(best - e.path) > 1e-4f) ++path_bad;
          path_err = std::max(path_err, (double)std::fabs(best - e.path));
          if (l == L_ROOM) {
            if (rs.room_at(e.gx + e.wx, e.gy + e.wy) != e.groom) ++room_bad;
            float yaw = std::atan2(0.5f * (e.prim[0].lo[1] + e.prim[0].hi[1]) - e.gy, 0.5f * (e.prim[0].lo[0] + e.prim[0].hi[0]) - e.gx);
            if (!body_free_host(S, e, e.gx, e.gy, yaw)) ++goal_free_bad;
          } else if (rs.room_at(0.5f * (e.prim[0].lo[0] + e.prim[0].hi[0]) + e.wx, 0.5f * (e.prim[0].lo[1] + e.prim[0].hi[1]) + e.wy) != e.groom) ++room_bad;
          auto inw = [](float x, float y) { return std::fabs(x) < WIN_HALF && std::fabs(y) < WIN_HALF; };
          bool w = inw(e.sx, e.sy) && inw(e.gx, e.gy);
          for (int p = 0; p < e.nprim; ++p) w = w && inw(0.5f * (e.prim[p].lo[0] + e.prim[p].hi[0]), 0.5f * (e.prim[p].lo[1] + e.prim[p].hi[1]));
          if (!w) ++win_bad;
          // 독립: RASC TRAV_OPEN_DOOR(BEHAVIOR 다닐 곳)에서 4 이웃 BFS, 장면 전체, 시작 칸 → 목표 쪽 칸(bc) 이어지나
          if (bc >= 0) {
            ++trav_n;
            int sr, sc0, gr, gc;
            const float gx = ((float)(bc % WIN) + 0.5f) * CELL - WIN_HALF + e.wx, gy = ((float)(bc / WIN) + 0.5f) * CELL - WIN_HALF + e.wy;
            if (rs.cell(e.sx + e.wx, e.sy + e.wy, &sr, &sc0) && rs.cell(gx, gy, &gr, &gc)) {
              const int Wd = (int)rs.h->grid_w, Hd = (int)rs.h->grid_h;
              // 시작·목표 칸이 막혀 있으면 3 칸 안 가장 가까운 빈 칸
              auto nearfree = [&](int& r, int& c) {
                for (int rad = 0; rad <= 3; ++rad)
                  for (int dr = -rad; dr <= rad; ++dr)
                    for (int dc = -rad; dc <= rad; ++dc)
                      if (rs.free_cell(3, r + dr, c + dc)) { r += dr; c += dc; return true; }
                return false;
              };
              if (nearfree(sr, sc0) && nearfree(gr, gc)) {
                std::vector<uint8_t> vis((size_t)Wd * Hd, 0);
                std::queue<int> q;
                q.push(sr * Wd + sc0);
                vis[(size_t)sr * Wd + sc0] = 1;
                bool found = false;
                while (!q.empty() && !found) {
                  const int u = q.front(); q.pop();
                  if (u == gr * Wd + gc) { found = true; break; }
                  const int r = u / Wd, c = u % Wd;
                  const int dr[4] = {1, -1, 0, 0}, dc[4] = {0, 0, 1, -1};
                  for (int k2 = 0; k2 < 4; ++k2) {
                    const int nr = r + dr[k2], nc = c + dc[k2];
                    if (!rs.free_cell(3, nr, nc) || vis[(size_t)nr * Wd + nc]) continue;
                    vis[(size_t)nr * Wd + nc] = 1;
                    q.push(nr * Wd + nc);
                  }
                }
                trav_ok += found;
              }
            }
          }
        }
    std::printf("%-26s entries %6ld | start pose collides %ld | path != Entry.path %ld (max err %.1e) | room != RASC %ld | outside window %ld | "
                "B1 goal pose collides %ld | connected on RASC TRAV_OPEN_DOOR (independent 4-nbr BFS) %ld / %ld\n",
                S.name.c_str(), n, start_bad, path_bad, path_err, room_bad, win_bad, goal_free_bad, trav_ok, trav_n);
    bad += start_bad + path_bad + room_bad + win_bad + goal_free_bad;
  }
  std::printf(bad ? "FAIL: %ld bad\n" : "OK: scene tables pass (%ld bad)\n", bad);
  return bad ? 1 : 0;
}
