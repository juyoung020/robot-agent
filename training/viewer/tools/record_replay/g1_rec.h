// G1 환경(training/RL/env) + G2 지도(training/RL/map) 상태 → .trp 판 하나(TRAIN_VIEWER.md 4.4). 학습기 코드는 바꾸지 않고 공개 헤더의 구조만 읽는다:
//   env::Soa / Core / load / step_core / fk (env.h·env_soa.h), gmap::MapHost / MapCore / Slot / Prim (map_api.h·map.h).
// 호출하는 쪽(record_ppo·record_bc)은 스텝마다 [환경·지도 내려받기 → 정책 한 스텝 → 행동·보상·끝 내려받기] 뒤 step() 을 부른다.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "env.h"
#include "env_soa.h"
#include "map_api.h"
#include "rec_util.h"
#include "gpu_sg.h"

namespace rec {

static const char* kClsName[gmap::NCLS] = {"cup", "item", "chair", "table", "cabinet", "bin"};
static const char* kFrameCols =
    "t,x,y,yaw,sx,sy,syaw,vx,wz,wl,wr,q1,q2,q3,q4,q5,qg,ee_x,ee_y,ee_z,a_vx,a_wz,a_q1,a_q2,a_q3,a_q4,a_q5,a_g,r_total,ret_cum,value,ev,tgt_slot,tgt_x,tgt_y,tgt_z,"
    "completion,map_seen,goal_known";
static const char* kSlotCols = "id,bx,by,bz,px,py,pz,ex,ey,ez,src,unc,state,is_tgt,age,confirmed";
constexpr int kNC = 39, kNSC = 16;
enum { EV_CONTACT = 1, EV_GRASP = 2, EV_DROP = 4, EV_SUCCESS = 8, EV_FILTER = 16, EV_RESET = 32 };

inline std::string joint_map_json() {
  std::string s = "{";
  for (int k = 1; k <= 5; ++k) s += "\"q" + std::to_string(k) + "\":[[\"omx_joint" + std::to_string(k) + "\",1,0]],";
  // 그리퍼: env q[5] (rad, 0 열림 … 1.745) → URDF 손가락 두 관절(부호 반대, 가정 — robot.json 한계 [0, 1.745] / [−1.745, 0])
  return s + "\"qg\":[[\"omx_gripper_joint_1\",1,0],[\"omx_gripper_joint_2\",-1,0]]}";
}

struct G1Rec {
  struct Ep {
    TrpWriter* w = nullptr;
    long ep = 0;
    int frames = 0;
    double ret = 0, path = 0, d0 = 0;
    float lx = 0, ly = 0;
    int contacts = 0, init_stage = 0, init_conf = 0, stage = 0;
    std::vector<int8_t> prev;
    std::vector<float> segs_prev;
    float last_m[3] = {0, 0, 0};   // 끝 프레임용 slam 자세
    GpuSg gsg;                     // sgview 판: GPU 지도(정책이 본 것)를 그대로(gpu_sg.h)
    std::vector<float> gt4;        // 참 궤적 [시각, x, y, yaw]·스텝(meta gt_path)
    std::string scene;             // scene_fn 이 판 시작에 만든 장면(BEHAVIOR 판 — 끝에서는 장치가 이미 새 판)
  };
  int N = 0;
  std::vector<int> tracked;
  std::map<int, Ep> act;
  Out out;
  std::string driver, source_json, skill = "approach", home_prefix = "G1", tag;   // tag: 체크포인트 이름(it000200 / final) — 파일 이름·판 줄에
  double ckpt_iter = NAN;
  int max_success = 1 << 30;
  // BEHAVIOR 판(stage 3, 2026-10-06): 장면 JSON 을 부르는 쪽이 만듦(env i → 창 좌표 정적 상자·집을 물체). 있으면 끝 프레임을 호스트에서 다시 돌리지 않음
  // (env.h step_core 는 상자 방 식) — 스텝 전 마지막 프레임에 끝 표시만. sgview 판(진짜 scenemap)도 안 만듦(장면 광선 추적이 상자 방 식)
  std::function<std::string(int)> scene_fn;
  // 덧붙임(BEHAVIOR 판, beh_rec.h): 스텝마다 그 판 기록에 섹션을 더함(판 i, 판 번호, 프레임) / 판을 쓰기 전(머리 더하기) / 쓴 뒤(경로 — OG 다시 돌리기 줄 세우기)
  std::function<void(int, long, uint32_t, TrpWriter*)> frame_hook;
  std::function<void(int, long, TrpWriter*)> pre_finish;
  std::function<void(int, long, const std::string&, const std::string&)> post_finish;
  bool sg = true;   // 판마다 진짜 scenemap 을 돌려 sgview 판(.sg)으로(--no-sg 면 .trp 만)   // 성공 판은 이만큼만 쓰고(그 뒤 성공은 버림) 실패를 더 기다린다
  long next_ep = 0;
  int finished = 0, img_every = 5;
  int n_success = 0, n_coll = 0, n_tout = 0, dropped_success = 0;
  // 끝: 판 K 개를 썼거나, 성공 몫(K−F)이 찼는데 실패가 안 나와 성공 판을 4K 개 버렸으면(실패가 드묾 — 있는 만큼만)
  bool done_enough(int episodes, int keep_fail) const { (void)keep_fail; return finished >= episodes || (n_success >= max_success && dropped_success >= 4 * episodes); }

  G1Rec(int N_, int n_track, const Out& o, std::string driver_, std::string source) : N(N_), out(o), driver(std::move(driver_)), source_json(std::move(source)) {
    for (int i = 0; i < n_track && i < N; ++i) tracked.push_back(i);
    next_ep = out.next_ep();
  }

  static std::string scene_json(const env::Core& c, const gmap::MapCore& m, int stage) {
    // 참 장면: 방 벽 4(반치수 rhx·rhy, 두께 0.1, 보기용 높이 0.5) + 지도 prim(가구·작은 물건·컵, 축 정렬 상자)
    std::string b = "[";
    auto box = [&](const char* kind, const char* name, float cx, float cy, float cz, float hx, float hy, float hz) {
      if (b.size() > 1) b += ',';
      b += Obj().str("kind", kind).str("name", name).raw("c", "[" + jnum(cx) + "," + jnum(cy) + "," + jnum(cz) + "]").raw("h", "[" + jnum(hx) + "," + jnum(hy) + "," + jnum(hz) + "]").num("yaw", 0).done();
    };
    const float t = 0.05f, wh = 0.25f;
    box("wall", "wall", 0, c.rhy + t, wh, c.rhx + 2 * t, t, wh);
    box("wall", "wall", 0, -c.rhy - t, wh, c.rhx + 2 * t, t, wh);
    box("wall", "wall", c.rhx + t, 0, wh, t, c.rhy, wh);
    box("wall", "wall", -c.rhx - t, 0, wh, t, c.rhy, wh);
    for (int k = 0; k < gmap::N_PRIM; ++k) {
      const gmap::Prim& p = m.prim[k];
      if (!(p.hi[0] > p.lo[0] && p.hi[1] > p.lo[1])) continue;
      const int cl = p.cls >= 0 && p.cls < gmap::NCLS ? p.cls : 1;
      box(k == 0 ? "target" : cl == gmap::C_ITEM ? "object" : "furniture", kClsName[cl], 0.5f * (p.lo[0] + p.hi[0]), 0.5f * (p.lo[1] + p.hi[1]), 0.5f * (p.lo[2] + p.hi[2]),
          0.5f * (p.hi[0] - p.lo[0]), 0.5f * (p.hi[1] - p.lo[1]), 0.5f * (p.hi[2] - p.lo[2]));
    }
    b += "]";
    return Obj().str("kind", "g1_room").str("name", "A" + std::to_string(stage) + " room " + jnum(2 * c.rhx) + "×" + jnum(2 * c.rhy) + " m").raw("boxes", b)
        .raw("rooms", "[{\"name\":\"room\",\"bmin\":[" + jnum(-c.rhx) + "," + jnum(-c.rhy) + "],\"bmax\":[" + jnum(c.rhx) + "," + jnum(c.rhy) + "]}]").done();
  }

  void start(int i, const env::Core& c, const gmap::MapCore& m, int stage) {
    Ep e;
    e.ep = next_ep++;
    e.stage = stage;
    e.init_stage = m.init_stage;
    e.init_conf = m.init_conf;
    e.d0 = c.prev_dist;
    e.lx = c.x; e.ly = c.y;
    e.prev.assign(gmap::NCELL, (int8_t)-2);
    std::string objs = "[";
    for (int k = 0; k < gmap::NCLS; ++k) objs += std::string(k ? "," : "") + "{\"id\":" + std::to_string(k) + ",\"name\":" + jstr(kClsName[k]) + ",\"cat\":" + jstr(kClsName[k]) + "}";
    objs += "]";
    const float ox = (float)gmap::GX0 * gmap::RES;
    std::string head = Obj().raw("meta", "{}").num("dt", 0.1).num("stride", 1).raw("objects", objs).raw("joint_map", joint_map_json())
        .raw("grid", Obj().num("res", gmap::RES).num("ox", ox).num("oy", ox).num("w", gmap::GW).num("h", gmap::GW).done())
        .raw("scene", scene_json(c, m, stage)).str("slot_z", "center").raw("source", source_json).b("synthetic", false)
        .raw("ev_bits", "{\"1\":\"collision\",\"8\":\"success\",\"32\":\"reset\"}").done();
    e.w = trp_new(kFrameCols, kSlotCols, gmap::KSLOT, head.c_str());
    if (scene_fn) { e.scene = scene_fn(i); trp_set_head(e.w, "scene", e.scene.c_str()); }
    act[i] = std::move(e);
  }

  void frame_row(const env::Core& c, const gmap::MapCore& m, const gmap::Slot* ob, const float* met, int i, const float* a, float r, float value, int ev, Ep& e, float* row, float* sl) {
    env::Fk f;
    env::fk(c.q, c.qd, f);
    const float sn = std::sin(c.yaw), cs = std::cos(c.yaw);
    const float ex = c.x + cs * f.ee_p[0] - sn * f.ee_p[1], ey = c.y + sn * f.ee_p[0] + cs * f.ee_p[1], ez = gmap::MP::base_z + f.ee_p[2];
    // 그릴 물체 KSLOT 개: 저장소(ob[NOBJ], 쓰는 칸 = m.objv)에서 확정 목표(컵) 먼저, 그다음 확정, 그다음 믿는 자세에서 가까운 순(토큰 고르기와 같은 뜻)
    int pick[gmap::KSLOT], np = 0;
    {
      std::vector<std::pair<float, int>> cand;
      for (int g = 0; g < gmap::NOBJ; ++g) {
        if (!((m.objv[g >> 5] >> (g & 31)) & 1u)) continue;
        const gmap::Slot& s = ob[g];
        const float d = std::hypot(s.pos[0] - m.ex, s.pos[1] - m.ey);
        const float pri = (s.confirmed && s.cls == gmap::C_CUP) ? 0.f : s.confirmed ? 1000.f : 2000.f;
        cand.push_back({pri + d, g});
      }
      std::sort(cand.begin(), cand.end());
      for (const auto& q : cand) { if (np == gmap::KSLOT) break; pick[np++] = q.second; }
    }
    int tslot = -1;
    for (int k = 0; k < np; ++k) if (ob[pick[k]].confirmed && ob[pick[k]].cls == gmap::C_CUP) tslot = k;
    e.ret += r;
    const float v[kNC] = {c.step * 0.1f, c.x, c.y, c.yaw, m.ex, m.ey, m.eyaw, c.v, c.w, c.wl, c.wr, c.q[0], c.q[1], c.q[2], c.q[3], c.q[4], c.q[5], ex, ey, ez,
                          a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], r, (float)e.ret, value, (float)ev, (float)tslot, c.tx, c.ty, env::K::tgt_z,
                          met[gmap::M_OBJ * N + i], met[gmap::M_SEEN * N + i], met[gmap::M_TASK * N + i]};
    for (int k = 0; k < kNC; ++k) row[k] = v[k];
    for (int k = 0; k < gmap::KSLOT; ++k) {
      float* o = sl + k * kNSC;
      for (int q = 0; q < kNSC; ++q) o[q] = 0.f;
      if (k >= np) continue;
      const gmap::Slot& s = ob[pick[k]];
      float px = s.pos[0], py = s.pos[1], pz = s.pos[2];
      if (s.src >= 0 && s.src < gmap::N_PRIM) {
        const gmap::Prim& p = m.prim[s.src];
        px = 0.5f * (p.lo[0] + p.hi[0]); py = 0.5f * (p.lo[1] + p.hi[1]); pz = 0.5f * (p.lo[2] + p.hi[2]);
      }
      const float unc = 0.02f + 0.03f * std::fmax(0.f, m.plen - s.seen_len);   // 마지막으로 본 뒤 믿는 이동 거리에 비례(가정, 토큰의 불확실도와 같은 재료)
      const float vals[kNSC] = {(float)s.cls, s.pos[0], s.pos[1], s.pos[2], px, py, pz, s.ext[0], s.ext[1], s.ext[2], s.last_seen == m.t ? 1.f : 2.f, unc,
                                (float)(s.held ? gmap::S_HELD : s.state), s.cls == gmap::C_CUP ? 1.f : 0.f, (m.t - s.last_seen) * 0.1f, s.confirmed ? 1.f : 0.f};
      for (int q = 0; q < kNSC; ++q) o[q] = vals[q];
    }
  }

  void map_rec(int i, const gmap::MapHost& mh, Ep& e, uint32_t fr) {
    const int16_t* L = mh.L.data() + (size_t)i * gmap::NCELL;
    const uint32_t* sn = mh.seen.data() + (size_t)i * gmap::NWORD;
    std::vector<int8_t> cur(gmap::NCELL);
    int x0 = gmap::GW, y0 = gmap::GW, x1 = -1, y1 = -1;
    for (int idx = 0; idx < gmap::NCELL; ++idx) {
      int8_t v = -1;   // MAP_RECT: −1 모름, 0–100 % (점유 확률 = 로그 오즈 L/256 의 시그모이드)
      if ((sn[idx >> 5] >> (idx & 31)) & 1u) v = (int8_t)std::lround(100.0 / (1.0 + std::exp(-L[idx] / 256.0)));
      cur[idx] = v;
      if (v != e.prev[idx]) { const int x = idx % gmap::GW, y = idx / gmap::GW; x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
    }
    if (x1 >= 0) {
      std::vector<int8_t> cells;
      cells.reserve((size_t)(x1 - x0 + 1) * (y1 - y0 + 1));
      for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) cells.push_back(cur[y * gmap::GW + x]);
      const double ox = gmap::GX0 * (double)gmap::RES;
      trp_map_rect(e.w, fr, gmap::GW, gmap::GW, gmap::RES, ox, ox, x0, y0, x1, y1, cells.data());
    }
    e.prev.swap(cur);
    // 벽 선분(지도가 뽑은 것): 반 칸 정수(창 원점 기준) → m. 바뀔 때만 "segs" 섹션에 기록
    const gmap::MapCore& m = mh.core[i];
    const int16_t* sg = mh.segs.data() + (size_t)i * gmap::SEGW;
    std::vector<float> segs;
    for (int P = 0; P < 2; ++P) {
      const int n = std::min(P ? m.nseg_v : m.nseg_h, gmap::MAXSEG);
      for (int k = 0; k < n; ++k)
        for (int q = 0; q < 4; ++q) segs.push_back((sg[(P * gmap::MAXSEG + k) * 4 + q] * 0.5f + gmap::GX0) * gmap::RES);
    }
    if (segs != e.segs_prev || fr == 0) {
      trp_record(e.w, "segs", fr, reinterpret_cast<const uint8_t*>(segs.data()), segs.size() * sizeof(float));
      e.segs_prev = segs;
    }
  }

  // 한 스텝. fs·iv·rg = 스텝 전 환경, mh = 스텝 전 지도, a = 행동 [k*N+i], rew·done·val [N]. img(i, cam) → 그 판의 256² RGB(없으면 nullptr)
  template <class ImgFn>
  void step(std::vector<float>& fs, std::vector<int>& iv, std::vector<uint64_t>& rg, const gmap::MapHost& mh, const std::vector<float>& a, const std::vector<float>& rew,
            const std::vector<int>& done, const std::vector<float>& val, int stage, ImgFn img, int img_res) {
    env::Soa hs{fs.data(), iv.data(), rg.data(), N};
    for (int i : tracked) {
      env::Core c;
      env::load(hs, i, c);
      if (!act.count(i)) {
        if (c.step != 0) continue;   // 판 중간부터는 안 씀
        start(i, c, mh.core[i], stage);
      }
      Ep& e = act[i];
      float ai[env::N_ACT];
      for (int k = 0; k < env::N_ACT; ++k) ai[k] = a[(size_t)k * N + i];
      float row[kNC], sl[gmap::KSLOT * kNSC];
      const uint32_t fr = (uint32_t)e.frames;
      frame_row(c, mh.core[i], mh.objs.data() + (size_t)i * gmap::NOBJ, mh.met.data(), i, ai, rew[i], val.empty() ? NAN : val[i], fr == 0 ? EV_RESET : 0, e, row, sl);
      trp_frame(e.w, row, sl);
      map_rec(i, mh, e, fr);
      if (frame_hook) frame_hook(i, e.ep, fr, e.w);
      if (img_res > 0 && e.frames % img_every == 0)
        for (int cam = 0; cam < 2; ++cam) {
          const uint8_t* p = img(i, cam);
          if (!p) continue;
          const std::vector<uint8_t> j = jpeg::encode(p, img_res, img_res, 80);
          trp_image(e.w, fr, (uint8_t)cam, j.data(), j.size());
        }
      if (sg && !scene_fn) {   // sgview 판: 이 스텝 전 GPU 지도 그대로(자세·관절 스텝마다, 격자·물체는 keyframe 마다)
        const gmap::MapCore& A = mh.core[i];
        float q12[12] = {A.ex, A.ey, A.eyaw, c.v, 0.f, c.w, c.q[0], c.q[1], c.q[2], c.q[3], c.q[4], c.q[5]};
        const double t = e.frames * 0.1;
        e.gsg.step(t, A, mh.objs.data() + (size_t)i * gmap::NOBJ, mh.L.data() + (size_t)i * gmap::NCELL, mh.seen.data() + (size_t)i * gmap::NWORD, q12,
                   A.kf_flag || e.frames == 0, [](int cl) { return cl >= 0 && cl < gmap::NCLS ? kClsName[cl] : "object"; });
        e.gt4.insert(e.gt4.end(), {(float)t, c.x, c.y, c.yaw});
      }
      e.path += std::hypot(c.x - e.lx, c.y - e.ly);
      e.lx = c.x; e.ly = c.y;
      e.frames++;
      for (int k = 0; k < 3; ++k) e.last_m[k] = (&mh.core[i].ex)[k];
      if (done[i] == env::kRunning) continue;
      if (scene_fn) {   // BEHAVIOR: 끝 표시만(스텝 전 프레임 그대로)
        const int ev = done[i] == env::kSuccess ? EV_SUCCESS : done[i] == env::kCollision ? EV_CONTACT : 0;
        frame_row(c, mh.core[i], mh.objs.data() + (size_t)i * gmap::NOBJ, mh.met.data(), i, ai, 0.f, NAN, ev, e, row, sl);
        trp_frame(e.w, row, sl);
        env::StepOut so{};
        finish(i, e, c, done[i], so, mh.core[i]);
        act.erase(i);
        continue;
      }
      // 끝 프레임: 같은 스텝을 호스트에서 다시(env.h 같은 소스) — 장치 상태는 이미 새 판으로 리셋됨
      env::Core cc;
      env::load(hs, i, cc);
      env::StepOut so;
      env::step_core(cc, ai, so, false, env::NoHook{});
      const int ev = done[i] == env::kSuccess ? EV_SUCCESS : done[i] == env::kCollision ? EV_CONTACT : 0;
      const float zero[env::N_ACT] = {0, 0, 0, 0, 0, 0, 0, 0};
      frame_row(cc, mh.core[i], mh.objs.data() + (size_t)i * gmap::NOBJ, mh.met.data(), i, zero, 0.f, NAN, ev, e, row, sl);
      row[4] = e.last_m[0]; row[5] = e.last_m[1]; row[6] = e.last_m[2];
      trp_frame(e.w, row, sl);
      e.path += std::hypot(cc.x - e.lx, cc.y - e.ly);
      finish(i, e, cc, done[i], so, mh.core[i]);
      act.erase(i);
    }
  }

  void finish(int i, Ep& e, const env::Core& cc, int d, const env::StepOut& so, const gmap::MapCore& m) {
    if (d == env::kSuccess && n_success >= max_success) { trp_free(e.w); e.w = nullptr; ++dropped_success; return; }
    // 장면·처음 지도 단계는 판 끝의 지도 상태에서(판의 첫 프레임에는 지도가 아직 리셋되기 전일 수 있다 — 각 판 첫 판)
    e.init_stage = m.init_stage;
    e.init_conf = m.init_conf;
    trp_set_head(e.w, "scene", scene_fn ? e.scene.c_str() : scene_json(cc, m, e.stage).c_str());
    const char* oc = d == env::kSuccess ? "success" : d == env::kCollision ? "collision" : "timeout";
    (d == env::kSuccess ? n_success : d == env::kCollision ? n_coll : n_tout)++;
    const char* stg[3] = {"C0", "C1", "C2"};
    char file[96];
    if (tag.empty()) std::snprintf(file, sizeof file, "ep_%06ld_%s_%s.trp", e.ep, skill.c_str(), oc);
    else std::snprintf(file, sizeof file, "ep_%06ld_%s_%s_%s.trp", e.ep, tag.c_str(), skill.c_str(), oc);
    // 성공 판정 거리(컵 겉면 0.4–0.8 m)를 뺀 경로 길이를 최단 L* 로(가정)
    const double lopt = std::max(0.0, e.d0 - 0.6);
    const float surf = std::hypot(cc.tx - cc.x, cc.ty - cc.y) - env::K::tgt_r;
    const float aim = std::fabs(std::remainder(std::atan2(cc.ty - cc.y, cc.tx - cc.x) - cc.yaw, 2 * M_PI)) * 57.29578f;
    std::string line = Obj().num("ts", (double)time(nullptr)).num("env", i).num("ep", e.ep).str("skill", skill).str("home", home_prefix)
        .str("stage", e.init_stage >= 0 && e.init_stage < 3 ? stg[e.init_stage] : "?").str("map_mode", "slam").num("completion0", e.init_conf / (double)gmap::N_PRIM)
        .str("driver", driver).b("success", d == env::kSuccess).str("outcome", oc).b("collided", d == env::kCollision).b("timeout", d == env::kTimeout)
        .num("t", e.frames * 0.1).num("steps", e.frames).num("ret", e.ret).raw("r", "{\"total\":" + jnum(e.ret) + "}").num("contacts", d == env::kCollision ? 1 : 0)
        .num("path_len", e.path).num("path_len_opt", lopt).num("final_dist", surf).num("final_aim_deg", aim).str("replay", file).str("ckpt", tag).num("ckpt_iter", ckpt_iter).done();
    (void)so;
    std::string path = out.rep + "/" + file;
    std::string sgdir;
    if (sg && !scene_fn && e.gsg.frames > 0) {   // sgview 판: <판>.sg/ 안에 스트림·메모리·바탕·정책 지도(episode.trp)
      std::string f2 = file;
      f2.replace(f2.size() - 4, 4, ".sg");
      sgdir = out.rep + "/" + f2;
      mkdirs(sgdir);
      path = sgdir + "/episode.trp";
      line.replace(line.find(std::string("\"replay\":\"") + file + "\""), std::string("\"replay\":\"").size() + std::strlen(file) + 1, std::string("\"replay\":\"") + f2 + "\"");
    }
    if (pre_finish) pre_finish(i, e.ep, e.w);
    trp_set_head(e.w, "meta", line.c_str());
    const long long nb = trp_finish(e.w, path.c_str());
    trp_free(e.w);
    e.w = nullptr;
    if (!sgdir.empty()) {
      const long nfr = e.gsg.write(sgdir);
      const int nobj = e.gsg.n_obj;
      std::string gt = "[";   // 지도 틀 = GPU 지도 창 틀(세계와 같음)
      for (size_t k = 0; k + 3 < e.gt4.size(); k += 4)
        gt += std::string(k ? "," : "") + "[" + jnum(e.gt4[k]) + "," + jnum(e.gt4[k + 1]) + "," + jnum(e.gt4[k + 2]) + "," + jnum(e.gt4[k + 3]) + "]";
      gt += "]";
      const double cth = 1.0, sth = 0.0, tx = 0.0, ty = 0.0;
      // 바탕 층: 같은 장면 상자(세계) — 화면이 map_from_world 로 돌림
      const std::string scene = scene_json(cc, m, e.stage);
      std::ofstream(sgdir + "/underlay.json") << "{\"scene\":\"A" << e.stage << " room\",\"boxes\":" << scene.substr(scene.find("\"boxes\":") + 8, scene.find(",\"rooms\"") - scene.find("\"boxes\":") - 8)
                                             << ",\"rooms\":[{\"name\":\"room\",\"type\":\"room\",\"centroid\":[0,0],\"bmin\":[" << jnum(-cc.rhx) << "," << jnum(-cc.rhy) << "],\"bmax\":[" << jnum(cc.rhx) << "," << jnum(cc.rhy) << "]}]}";
      std::ofstream(sgdir + "/meta.json") << Obj().str("format", "SGS1").raw("meta", line).str("robot", "limo_omx").raw("map_from_world", "[" + jnum(cth) + "," + jnum(sth) + "," + jnum(tx) + "," + jnum(ty) + "]")
                                                .raw("gt_path", gt).raw("labels", "{}").raw("cams", "[]").num("duration", e.gt4.empty() ? 0.0 : (double)e.gt4[e.gt4.size() - 4]).str("policy_trp", "episode.trp")
                                                .raw("joint_order", "[\"\",\"\",\"\",\"\",\"\",\"\",\"omx_joint1\",\"omx_joint2\",\"omx_joint3\",\"omx_joint4\",\"omx_joint5\",\"omx_gripper_joint_1\"]")
                                                .num("n_objects", nobj).num("stream_frames", (double)nfr).str("source", "GPU env episode, GPU map as trained (training/RL/map objprob store) — gpu_sg.h").done();
    }
    out.append_episode(line);
    if (post_finish) post_finish(i, e.ep, path, line);
    ++finished;
    std::printf("  ep %ld env %d: %s, %d frames, return %.2f -> %s (%.0f KB)\n", e.ep, i, oc, e.frames + 1, e.ret, sgdir.empty() ? path.c_str() : sgdir.c_str(), nb / 1024.0);
  }
};

}  // namespace rec
