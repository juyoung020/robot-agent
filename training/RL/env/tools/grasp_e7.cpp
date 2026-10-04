// E7(잡기 물리 E6 대조): E0 OmniGibson 잡기 시험(src/robot/og/e0, 그리퍼 kp 1e6) 78 경우를 GPU 잡기 모형(grasp.h — 장치 커널과 같은 함수)에 그대로 넣어
// "잡혀 들렸나" 예측이 OmniGibson 결과와 맞는 비율을 낸다. 닿음(역기구학)도 따로: OmniGibson 쪽 kin.ik(관절 한계 전체 + 몸통 충돌)이 자세를 찾았나 대 우리 ik_grasp(행동 범위).
//   python3 src/robot/og/e7/e0_cases.py > cases.csv && grasp_e7 cases.csv [max_w grip_top]
//   max_w·grip_top: 모형 값 바꿔 보기(기본 = KG::max_w 0.06, 위에서 잡기 그리퍼 한도 = KG::grip_mass_top — 더 작게만 걸림). 닿음·들기 닿음은 관절 한계 전체(E0 kin.ik 와 같게 — 빌드 TEACH_FULLRANGE)
// 경우마다 상자 가운데는 link5 x = g_place(y 0, z 0)에 놓였다(E0 trial). 우리 잡는 점은 link5 (0.08003, −0.0016, 0). 도구 기울기 phi: E0 DOWN = +90°(아래) → 우리 −π/2.
// 그리퍼 벌림 = 우리 행동 상한 0.6 rad(틈 0.066 m — E0 는 다 엶). 들기 = 무게 ≤ payload_max(s)(s = joint1 축 → 상자 가운데 수평 거리).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <vector>

#include "env_api.h"

using namespace env;

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: grasp_e7 cases.csv [max_w grip_top]\n"); return 2; }
  const float max_w = argc > 2 ? (float)std::atof(argv[2]) : KG::max_w, grip_top = argc > 3 ? (float)std::atof(argv[3]) : KG::grip_mass_top;
  FILE* f = std::fopen(argv[1], "r");
  if (!f) { std::perror(argv[1]); return 2; }
  char line[512];
  std::fgets(line, sizeof line, f);   // 머리
  std::map<std::string, std::pair<int, int>> by;   // 묶음 → (같음, 시도)
  int n = 0, att = 0, agree = 0, tp = 0, tn = 0, fp = 0, fn = 0, reach_agree = 0;
  std::printf("%-28s %-7s %5s %5s %6s %5s | OG ok | model: width w   grasp  lift  pred | reach OG/ours\n", "case", "group", "s", "z", "phi", "mass");
  while (std::fgets(line, sizeof line, f)) {
    char grp[32], name[96];
    float s, zc, psi, phi, sx, sy, sz, mass, g;
    int attempted, in_hand, ok;
    if (std::sscanf(line, "%31[^,],%95[^,],%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d", grp, name, &s, &zc, &psi, &phi, &sx, &sy, &sz, &mass, &g, &attempted, &in_hand, &ok) != 14) continue;
    ++n;
    const bool down = phi > 45.f;
    // 닿음: 우리 역기구학(행동 범위) — base_link 목표, 기울기 −phi
    const float ps = psi * (kPi / 180.f), pc = cosf(ps), pn = sinf(ps);
    const float tgt[3] = {KIK::j1x + s * pc, s * pn, zc - 0.15f};
    float q[5];
    const float phr = -phi * (kPi / 180.f);
    const bool reach = ik_grasp(tgt, phr, 0.f, 0, q) || ik_grasp(tgt, phr, 0.f, 1, q);
    reach_agree += reach == (attempted != 0);
    if (!attempted) {
      std::printf("%-28s %-7s %5.2f %5.2f %6.0f %5.2f | (no IK in OG) | %-34s | %d/%d\n", name, grp, s, zc, phi, mass, "-", attempted, reach ? 1 : 0);
      continue;
    }
    ++att;
    // 손 축(a 다가감, n 닫음, b 나머지)으로 본 상자 크기
    // 팔 방향(psi): 옆(90°) 이면 팔 = 세계 y, 닫는 축 = 세계 x / 앞(0°) 이면 팔 = x, 닫는 축 = y
    const bool side = psi > 45.f;
    const float lat = side ? sx : sy, along = side ? sy : sx;
    const float e[3] = {down ? sz : along, lat, down ? along : sz};
    Hand h;
    for (int a = 0; a < 3; ++a) { h.p[a] = 0.f; h.a[a] = a == 0; h.n[a] = a == 1; h.b[a] = a == 2; }
    const float o[3] = {g - 0.08003f, 0.0016f, 0.f};
    float dn;
    const float w = grasp_width(h, o, e, 1.f, 0.f, grip_gap_of(0.6f), dn);
    const bool grasp = w > 0.f && w <= max_w;
    // 들기: 무게 ≤ 가반 하중(위에서 잡기면 그리퍼 한도 grip_top 도) + E0 처럼 0.08 m 위가 닿음
    const float tl[3] = {tgt[0], tgt[1], tgt[2] + 0.08f};
    float ql[5];
    const bool lreach = ik_grasp(tl, phr, 0.f, 0, ql) || ik_grasp(tl, phr, 0.f, 1, ql);
    const bool lift = mass_of(mass) <= payload_max(s, down ? -1.f : 0.f) && (!down || mass_of(mass) <= grip_top) && lreach;
    const bool pred = grasp && lift;
    agree += pred == (ok != 0);
    by[grp].first += pred == (ok != 0); by[grp].second += 1;
    tp += pred && ok; tn += !pred && !ok; fp += pred && !ok; fn += !pred && ok;
    std::printf("%-28s %-7s %5.2f %5.2f %6.0f %5.2f | %5d | %5.3f %5d %5d %5d | %d/%d\n", name, grp, s, zc, phi, mass, ok, w, grasp ? 1 : 0, lift ? 1 : 0, pred ? 1 : 0, attempted, reach ? 1 : 0);
  }
  std::fclose(f);
  for (auto& kv : by) std::printf("  group %-8s agreement %d / %d\n", kv.first.c_str(), kv.second.first, kv.second.second);
  std::printf("model max_w %.3f, top-down grip limit %.2f kg\n", max_w, grip_top);
  std::printf("E7 grasp model vs OmniGibson (E0 cases, kp 1e6): %d cases, %d attempted in OG; outcome agreement %d / %d = %.3f (pred ok & OG ok %d, both fail %d, model ok but OG fail %d, model fail but OG ok %d); "
              "reach agreement (OG kin.ik found a pose vs our ik_grasp, both full joint range) %d / %d = %.3f\n",
              n, att, agree, att, att ? (double)agree / att : 0.0, tp, tn, fp, fn, reach_agree, n, n ? (double)reach_agree / n : 0.0);
  return 0;
}
