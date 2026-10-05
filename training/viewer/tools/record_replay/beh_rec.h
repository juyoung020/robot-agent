// BEHAVIOR 집기·놓기 판(stage 3)의 재생 기록 — record_bc 가 판 시작·스텝마다 부른다(학습기 코드는 읽기만).
//
//   1. 장면 머리(scene): 창 좌표 상자 = 벽·창·가구(진짜 종류 이름 — 장치 bname → vla_v1 이름 표, RASC 물체 이름·종류) + 문(환경은 열린 채로 빼는 것)
//      + 집을 물체(target)·놓을 곳(place). "world" = OmniGibson 다시 돌리기에 필요한 것(장면 이름, 창 가운데 wx·wy, 과제·인스턴스, 집을 물체 OG 이름…),
//      "pnp" = 이 짝이 왜 되는지(잡기 모형 grasp.h: 손 벌림 0.06 m 대 물체 폭, 질량 대 가반 하중, 받침·놓을 면 높이 대 팔 닿는 띠, 잡기 가능 표 비트·까닭,
//      서는 자리 후보 pnp_stance_cands, 표 자리 gst4·gst·pst5·pst6), "picks" = 창 안 집을 후보마다 잡기 모형 까닭(색칠용).
//   2. 스텝마다 "inputs" 섹션 한 기록(정책이 받은 입력, 아래 InLayout) + 판 끝에 이름 표(이 판에 나온 행만)·지시문 글.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bscene_host.h"
#include "env.h"
#include "env_beh.h"
#include "env_soa.h"
#include "map_api.h"
#include "rasc.h"
#include "rec_util.h"
#include "teacher_sl.h"

namespace rec {

// ---- 스텝 입력 기록 한 줄(바이트, little-endian). 머리 "inputs" 에 같은 배치를 적는다(뷰어가 이름으로 읽음) ----
//   tok   1360 B  지도 토큰 MapTok 앞부분 그대로(v3 배치: slot[16][33] f16, name_id[16] i16, app_id[16] i16, wall[56] f16, room[10] f16, comp[4] f16,
//                 n_slot i16, flags i16, front[8] f16, way[4] f16, instr1 u16, bkind u16, pad2[2], goal[2][16] f16) — m·rad 그대로(정규화 전)
//   obs     80 f32  환경 관측(G1 obs 80, 학생은 53–55·72–79 를 0 으로 가림 — x0 에 그대로 보임)
//   x0     208 bf16 신경망 입력 X0 의 [128, 304) + [432, 464) (정규화 뒤 = 학생/교사 MLP 가 받은 값 그대로; 지시 벡터 128 · 칸 풀링 128 · 위 그림 512 는 뺌)
//   act     8 f32  정책 출력(학생 μ, 교사 판은 교사 행동)
//   label   8 f32  대본 교사 라벨(이 상태에서 교사가 할 행동 = DAgger 라벨)
//   exec    8 f32  실제로 낸 행동(가림 뒤)
//   priv   24 f32  특권 상태(교사만 앎): 물체 x y z yaw 상태, 교사 단계, 서는 자리 계획 sx sy syaw dp 유효 됨, 놓을 점 x y z, 목표 꼴, 단계, 든 상대 a n b, 막힌 수, 0
constexpr int IN_TOK = 1360, IN_OBS = 80, IN_X0A = 128, IN_X0B = 304, IN_X0C = 432, IN_X0D = 464, IN_X0 = (IN_X0B - IN_X0A) + (IN_X0D - IN_X0C), IN_PRIV = 24;
constexpr int IN_BYTES = IN_TOK + 4 * IN_OBS + 2 * IN_X0 + 4 * 8 * 3 + 4 * IN_PRIV;
static_assert(IN_X0 == 208, "x0 parts");

inline std::string in_layout_json() {
  return Obj().num("v", 1).num("bytes", IN_BYTES).num("tok", 0).num("obs", IN_TOK).num("x0", IN_TOK + 4 * IN_OBS).num("act", IN_TOK + 4 * IN_OBS + 2 * IN_X0)
      .num("label", IN_TOK + 4 * IN_OBS + 2 * IN_X0 + 32).num("exec", IN_TOK + 4 * IN_OBS + 2 * IN_X0 + 64).num("priv", IN_TOK + 4 * IN_OBS + 2 * IN_X0 + 96)
      .raw("x0_cols", "[[128,304],[432,464]]").num("n_x0", IN_X0).num("n_priv", IN_PRIV).done();
}

static const char* kSlModeName[] = {"none", "fail", "nav", "approach0", "rotate", "arm_ready", "drive", "fine", "wait", "arm_grasp", "close", "reopen",
                                    "lift", "fold", "back", "arm_place", "open", "retreat", "done", "explore", "fail_arm", "backup"};
static const char* kFeasReason[] = {"ok", "too wide", "too heavy", "too thin", "no stance", "no place", "no path", "arm", "drops", "stuck", "miss", "pre-grasp", "not findable"};

inline std::string jarr(const float* v, int n) {
  std::string s = "[";
  for (int k = 0; k < n; ++k) s += std::string(k ? "," : "") + jnum(v[k]);
  return s + "]";
}

struct BehRec {
  const bsc::SceneBuild* B = nullptr;
  std::map<int, std::unique_ptr<rasc::Scene>> rs;
  std::vector<std::string> instr;   // pnp_v1 instr.jsonl 글
  explicit BehRec(const bsc::SceneBuild* b) : B(b) {
    std::ifstream f(bsc::default_pnp_dir() + "/instr.jsonl");
    std::string line;
    while (std::getline(f, line)) {
      const size_t p = line.find("\"text\": \"");
      instr.push_back(p == std::string::npos ? "" : line.substr(p + 9, line.find('"', p + 9) - p - 9));
    }
  }
  const rasc::Scene* rasc_of(int si) {
    auto it = rs.find(si);
    if (it != rs.end()) return it->second.get();
    auto S = std::make_unique<rasc::Scene>();
    std::string err;
    const std::string path = bsc::default_rasc_dir() + "/" + B->sc[si].name + ".rasc";
    if (!rasc::load(*S, path.c_str(), &err)) { std::fprintf(stderr, "beh_rec: rasc %s: %s\n", path.c_str(), err.c_str()); S.reset(); }
    return (rs[si] = std::move(S)).get();
  }
  std::string nm(int row) const { return row == -2 ? std::string("floor") : row <= -1000 ? B->sc.empty() ? "floor" : "room floor" : row >= 0 && row < (int)B->name_en.size() ? B->name_en[row] : std::string(); }
  std::string instr_text(int row) const { return row >= 0 && row < (int)instr.size() ? instr[row] : std::string(); }

  // RASC 집기 기록 → (OG 이름, 종류, BDDL 인스턴스, 과제 물체 번호)
  struct ObjId { std::string og, cat, inst, model; int taskobj = -1, sceneobj = -1; };
  ObjId pick_id(const rasc::Scene& S, uint32_t obj) const {
    ObjId o;
    auto cat = [&](uint16_t k) { return k < S.cats.size() ? std::string(S.str(S.cats[k].name)) : std::string(); };
    if (obj == 0xffffffffu) return o;
    if (obj & RASC_PNP_TASKOBJ) {
      const uint32_t k = obj & ~RASC_PNP_TASKOBJ;
      if (k < S.task_objs.size()) {
        const RascTaskObjRec& t = S.task_objs[k];
        o.og = S.str(t.og_name); o.cat = cat(t.cat); o.inst = S.str(t.inst); o.taskobj = (int)k; o.sceneobj = t.scene_obj;
        if (t.scene_obj >= 0 && (size_t)t.scene_obj < S.objs.size()) o.model = S.str(S.objs[t.scene_obj].model);
      }
    } else if (obj < S.objs.size()) {
      o.og = S.str(S.objs[obj].name); o.cat = cat(S.objs[obj].cat); o.model = S.str(S.objs[obj].model); o.sceneobj = (int)obj;
    }
    return o;
  }

  // 판 시작의 장면 머리(창 좌표). b·p = 판 시작 상태
  std::string scene_json(const env::BState& b, const env::PState& p) {
    const bsc::SceneBuild::Sc& sc = B->sc[b.scene];
    const bsc::SceneDev& d = sc.d;
    const bsc::Entry& E = B->ent[b.ent];
    const rasc::Scene* S = rasc_of(b.scene);
    auto cat = [&](uint16_t k) { return S && k < S->cats.size() ? std::string(S->str(S->cats[k].name)) : std::string(); };
    std::string bx = "[";
    auto box = [&](const char* kind, const std::string& name, const std::string& catn, const std::string& obj, float cx, float cy, float cz, float hx, float hy, float hz, float yaw,
                   const char* extra = nullptr) {
      if (bx.size() > 1) bx += ',';
      Obj o;
      o.str("kind", kind).str("name", name).str("cat", catn).str("obj", obj).raw("c", "[" + jnum(cx) + "," + jnum(cy) + "," + jnum(cz) + "]")
          .raw("h", "[" + jnum(hx) + "," + jnum(hy) + "," + jnum(hz) + "]").num("yaw", yaw);
      if (extra) o.raw("env", extra);
      bx += o.done();
    };
    const float L = bsc::WIN_HALF + 0.5f;
    int nwall = 0, nfurn = 0, nwin = 0, ndoor = 0;
    for (int k = 0; k < d.nbox; ++k) {
      const bsc::SBox& q = d.box[k];
      const float x = q.cx - b.wx, y = q.cy - b.wy;
      if (std::fabs(x) > L + std::max(q.hx, q.hy) || std::fabs(y) > L + std::max(q.hx, q.hy)) continue;
      const uint8_t bk = d.bkind[k];
      const char* kind = (bk & bsc::BK_WALL) ? "wall" : (bk & bsc::BK_WINDOW) ? "window" : "furniture";
      if (!(bk & bsc::BK_WALL) && q.z0 > 1.8f) continue;   // 천장 등
      std::string name = nm(d.bname[k]), catn, obj;
      const int ro = k < (int)sc.bobj.size() ? sc.bobj[k] : -1;
      if (S && ro >= 0 && (size_t)ro < S->objs.size()) { catn = cat(S->objs[ro].cat); obj = S->str(S->objs[ro].name); }
      if (name.empty() || name == "?" || name == "furniture") name = catn.empty() ? name : catn;   // 장치 이름 표가 넓은 말(furniture)이면 RASC 종류
      if (bk & bsc::BK_WALL) name = "wall";
      for (char& ch : name) if (ch == '_') ch = ' ';
      (bk & bsc::BK_WALL ? nwall : bk & bsc::BK_WINDOW ? nwin : nfurn)++;
      box(kind, name, catn, obj, x, y, 0.5f * (q.z0 + q.z1), q.hx, q.hy, 0.5f * (q.z1 - q.z0), std::atan2(q.s, q.c), bk & bsc::BK_COLL ? "\"collide\"" : "\"box\"");
    }
    if (S)   // 문: 환경은 열린 채로 둔다(상자에서 뺌) — 보기용
      for (const RascBoxRec& r : S->boxes) {
        if (!(r.flags & RASC_F_DOOR)) continue;
        const float x = r.center[0] - b.wx, y = r.center[1] - b.wy;
        if (std::fabs(x) > L || std::fabs(y) > L) continue;
        ++ndoor;
        box("door", "door", cat(r.cat), r.obj < S->objs.size() ? S->str(S->objs[r.obj].name) : "", x, y, r.center[2], r.half[0], r.half[1], r.half[2], r.yaw, "\"removed (open)\"");
      }
    ObjId pk, dst, src;
    if (S && E.pick_rec >= 0 && (size_t)E.pick_rec < S->picks.size()) pk = pick_id(*S, S->picks[E.pick_rec].obj);
    if (S && E.dst_rec >= 0 && (size_t)E.dst_rec < S->places.size()) dst = pick_id(*S, S->places[E.dst_rec].obj);
    if (S && E.src_rec >= 0 && (size_t)E.src_rec < S->places.size()) src = pick_id(*S, S->places[E.src_rec].obj);
    std::string pname = nm(E.prim[0].name);
    if (pname.empty() || pname == "furniture") pname = pk.cat;
    std::string dname = nm(E.dst_name);
    if ((dname.empty() || dname == "furniture") && !dst.cat.empty()) dname = dst.cat;
    for (char& ch : dname) if (ch == '_') ch = ' ';
    box("target", pname, pk.cat, pk.og, p.o[0], p.o[1], p.o[2], 0.5f * E.odim[0], 0.5f * E.odim[1], 0.5f * E.odim[2], p.yaw, "\"task object\"");
    if (E.dhi[0] > E.dlo[0])
      box("place", "place: " + dname, dst.cat, dst.og, 0.5f * (E.dlo[0] + E.dhi[0]), 0.5f * (E.dlo[1] + E.dhi[1]), 0.5f * (E.dlo[2] + E.dhi[2]), 0.5f * (E.dhi[0] - E.dlo[0]),
          0.5f * (E.dhi[1] - E.dlo[1]), std::max(0.005f, 0.5f * (E.dhi[2] - E.dlo[2])), 0.f, "\"place region\"");
    bx += "]";
    // 방(창 안 보기용)
    std::string rooms = "[";
    if (S)
      for (size_t k = 0; k < S->rooms.size(); ++k) {
        const RascRoomRec& r = S->rooms[k];
        if (r.bmax[0] - b.wx < -L || r.bmin[0] - b.wx > L || r.bmax[1] - b.wy < -L || r.bmin[1] - b.wy > L) continue;
        rooms += std::string(rooms.size() > 1 ? "," : "") + Obj().str("name", S->str(r.name)).raw("bmin", "[" + jnum(r.bmin[0] - b.wx) + "," + jnum(r.bmin[1] - b.wy) + "]")
                                                         .raw("bmax", "[" + jnum(r.bmax[0] - b.wx) + "," + jnum(r.bmax[1] - b.wy) + "]").done();
      }
    rooms += "]";
    std::string task, inst_id = "null", split = "null";
    if (S && E.task >= 0 && (size_t)E.task < S->tasks.size()) task = S->str(S->tasks[E.task].name);
    if (S && E.inst >= 0 && (size_t)E.inst < S->insts.size()) { inst_id = std::to_string(S->insts[E.inst].inst_id); split = std::to_string(S->insts[E.inst].split); }
    auto oid = [&](const ObjId& o, const std::string& name) {
      return Obj().str("og_name", o.og).str("cat", o.cat).str("inst", o.inst).str("model", o.model).num("taskobj", o.taskobj).num("sceneobj", o.sceneobj).str("name", name).done();
    };
    const std::string world = Obj().str("scene", sc.name).num("scene_index", b.scene).num("wx", b.wx).num("wy", b.wy).str("task", task).raw("inst_id", inst_id).raw("split", split)
                                  .num("rasc_inst", E.inst).num("entry", b.ent).num("kind", b.kind).str("rasc", bsc::default_rasc_dir() + "/" + sc.name + ".rasc")
                                  .raw("pick", oid(pk, pname)).raw("place", oid(dst, dname)).raw("source", oid(src, nm(E.src_name)))
                                  .num("dkind", E.dkind).num("rel", E.rel).num("instr", b.instr).str("instr_text", instr_text(b.instr))
                                  .raw("start", "[" + jnum(E.sx) + "," + jnum(E.sy) + "," + jnum(E.syaw) + "]").done();
    char nmb[160];
    std::snprintf(nmb, sizeof nmb, "%s · %s · B%d", sc.name.c_str(), task.c_str(), b.kind);
    return Obj().str("kind", "behavior_window").str("name", nmb).raw("boxes", bx).raw("rooms", rooms).raw("world", world).raw("pnp", pnp_json(b, p, E))
        .raw("picks", picks_json(b, E)).raw("counts", Obj().num("wall", nwall).num("furniture", nfurn).num("window", nwin).num("door", ndoor).done()).done();
  }

  // 이 짝이 되는 까닭(잡기 모형 grasp.h · 거르개 표 · 잡기 가능 표 · 서는 자리)
  std::string pnp_json(const env::BState& b, const env::PState& p, const bsc::Entry& E) {
    (void)p;
    const bsc::PnpFilter& F = B->filt;
    const float w = std::min(E.odim[0], E.odim[1]), mass = env::mass_of(E.mass);
    const int sf = env::obj_static_feas(E.odim, E.mass);
    auto rsn = [](int r) { return std::string(r >= 0 && r < 13 ? kFeasReason[r] : "?"); };
    const int fe = E.feas;
    const float dtop = E.ppt_ok ? E.ppt[2] : E.dhi[2] > E.dlo[2] ? E.dhi[2] : 0.f;   // 놓을 면 높이(점 → 그 z, 상자 → 윗면, 바닥 0)
    std::string cands = "[";
    int ncand = 0;
    if (!B->gcand.empty() && !B->toccix.empty() && b.ent < (int)B->toccix.size() && B->toccix[b.ent] >= 0) {
      const int ix = B->toccix[b.ent];
      ncand = B->gcn[ix];
      for (int k = 0; k < ncand && k < env::GC_K; ++k) {
        const float* g = B->gcand.data() + ((size_t)ix * env::GC_K + k) * 4;
        cands += std::string(k ? "," : "") + jarr(g, 4);
      }
    }
    cands += "]";
    return Obj()
        .raw("grasp", Obj().num("aperture_max", env::KG::max_w).num("obj_width", w).num("obj_depth", std::max(E.odim[0], E.odim[1])).num("obj_height", E.odim[2])
                          .num("min_height", env::kMinGraspH).b("width_ok", w <= env::KG::max_w).b("height_ok", E.odim[2] >= env::kMinGraspH).done())
        .raw("mass", Obj().num("kg", mass).b("known", E.mass == E.mass && E.mass > 0).num("payload_max", env::payload_max(0.f, 0.f)).num("grip_side", env::KG::grip_mass_max)
                         .num("grip_top", env::KG::grip_mass_top).num("arm_near", env::KG::arm_mass_near).num("arm_far", env::KG::arm_mass_far)
                         .num("arm_r_near", env::KG::arm_r_near).num("arm_r_far", env::KG::arm_r_far).b("ok", mass <= env::payload_max(0.f, 0.f)).done())
        .str("static", rsn(sf))
        .raw("surface", Obj().num("src_top", E.src_top).num("dst_top", dtop).num("obj_z0", p.o[2] - 0.5f * E.odim[2])
                            .raw("pick_z", "[" + jnum(F.pick_z[0]) + "," + jnum(F.pick_z[1]) + "]").raw("place_top", "[" + jnum(F.place_top[0]) + "," + jnum(F.place_top[1]) + "]")
                            .num("reach_low", F.reach_low).num("reach_high", F.reach_high).num("edge_dist", F.edge_dist).num("topdown_z", F.topdown_z)
                            .raw("max_w", "[" + jnum(F.max_w[0]) + "," + jnum(F.max_w[1]) + "]").raw("max_mass", "[" + jnum(F.max_mass[0]) + "," + jnum(F.max_mass[1]) + "]")
                            .b("src_ok", E.src_top <= F.pick_z[0] + 1e-3f).b("dst_ok", dtop <= F.place_top[0] + 1e-3f).done())
        .raw("feas", Obj().num("bits", fe & 0xff).b("B4", fe & bsc::FE_GRASP).b("B5", fe & bsc::FE_PLACE5).b("B6", fe & bsc::FE_PLACE6).str("grasp_why", rsn((fe >> 8) & 0xff))
                         .str("place5_why", rsn((fe >> 16) & 0xff)).str("place6_why", rsn((fe >> 24) & 0xff)).num("fset", E.fset)
                         .b("findable", fe & bsc::FE_FIND).b("findable_place", fe & bsc::FE_FINDP).b("find_table", B->host.has_find).raw("find_pose", jarr(E.fpose, 4)).done())
        .raw("stance", Obj().raw("grasp_cell", jarr(E.st, 2)).raw("gst4", jarr(E.gst4, 4)).raw("gst", jarr(E.gst, 4)).raw("pst5", jarr(E.pst5, 4)).raw("pst6", jarr(E.pst6, 4))
                           .raw("dst_cell", jarr(E.dst_st, 2)).raw("cands", cands).num("n_cands", ncand).done())
        .raw("place_pt", E.ppt_ok ? jarr(E.ppt, 3) : std::string("null")).raw("place_box", "[" + jarr(E.dlo, 3) + "," + jarr(E.dhi, 3) + "]").done();
  }

  // 창 안 집을 후보(장면 수준 + 이 인스턴스)마다 잡기 모형 까닭 — 색칠용
  std::string picks_json(const env::BState& b, const bsc::Entry& E) {
    const rasc::Scene* S = rasc_of(b.scene);
    std::string o = "[";
    if (!S || !S->pnp_ranges.size()) return o + "]";
    const float L = bsc::WIN_HALF;
    int n = 0;
    auto add = [&](const RascPnpRange& R) {
      for (uint32_t k = 0; k < R.n_pick && n < 400; ++k) {
        const RascPickRec& pr = S->picks[R.pick_off + k];
        const float x = pr.center[0] - b.wx, y = pr.center[1] - b.wy;
        if (std::fabs(x) > L || std::fabs(y) > L) continue;
        const float e[3] = {pr.min_w, pr.min_w, pr.top - pr.z0};
        const int r = env::obj_static_feas(e, pr.mass);
        const ObjId id = pick_id(*S, pr.obj);
        std::string why = r >= 0 && r < 12 ? kFeasReason[r] : "?";
        o += std::string(n++ ? "," : "") + Obj().str("name", id.cat).str("obj", id.og).raw("c", "[" + jnum(x) + "," + jnum(y) + "," + jnum(pr.center[2]) + "]")
                                                .num("w", pr.min_w).num("h", pr.top - pr.z0).num("kg", env::mass_of(pr.mass)).str("why", why).b("ok", r == 0)
                                                .b("target", (int)(R.pick_off + k) == E.pick_rec).done();
      }
    };
    add(S->pnp_ranges[S->pnp_ranges.size() - 1]);
    if (E.inst >= 0 && (size_t)E.inst + 1 < S->pnp_ranges.size()) add(S->pnp_ranges[E.inst]);
    return o + "]";
  }
};

// 판 하나 동안 모으는 것(이름 행·서는 자리 선택)
struct InEp {
  std::set<int> names;
  int last_mode = -1;
  float chosen[4] = {NAN, NAN, NAN, NAN};
  int instr = -1;
};

// 스텝 입력 한 줄을 만든다. 호스트 배열: tok(판 i 의 MapTok), obs 열 우선 [80][N], x0 bf16 행 [976], act·label·exec 8, sl(판 i 의 SlRec 또는 nullptr), hs = 스텝 전 환경
inline void in_pack(std::vector<uint8_t>& out, const gmap::MapTok& tok, const float* obs_col, int N, int i, const uint16_t* x0row, const float* act, const float* label,
                    const float* exec, const env::SlRec* sl, const env::Soa& hs, InEp& ep) {
  out.assign(IN_BYTES, 0);
  uint8_t* q = out.data();
  std::memcpy(q, &tok, IN_TOK);
  q += IN_TOK;
  float* o = reinterpret_cast<float*>(q);
  for (int k = 0; k < IN_OBS; ++k) o[k] = obs_col[(size_t)k * N + i];
  q += 4 * IN_OBS;
  uint16_t* x = reinterpret_cast<uint16_t*>(q);
  if (x0row) {
    std::memcpy(x, x0row + IN_X0A, 2 * (IN_X0B - IN_X0A));
    std::memcpy(x + (IN_X0B - IN_X0A), x0row + IN_X0C, 2 * (IN_X0D - IN_X0C));
  }
  q += 2 * IN_X0;
  float* a = reinterpret_cast<float*>(q);
  for (int k = 0; k < 8; ++k) { a[k] = act ? act[k] : NAN; a[8 + k] = label ? label[k] : NAN; a[16 + k] = exec ? exec[k] : NAN; }
  q += 4 * 24;
  float* pv = reinterpret_cast<float*>(q);
  env::BState b;
  env::load_b(hs, i, b);
  env::PState p;
  env::load_p(hs, i, p);
  const float v[IN_PRIV] = {p.o[0], p.o[1], p.o[2], p.yaw, (float)p.st, sl ? (float)sl->mode : -1.f, sl ? sl->st.sx : NAN, sl ? sl->st.sy : NAN, sl ? sl->st.syaw : NAN,
                            sl ? sl->st.dp : NAN, sl ? (float)sl->st.valid : 0.f, sl ? (float)sl->st.ok : 0.f, b.gp[0], b.gp[1], b.gp[2], (float)b.gmode, (float)b.kind,
                            p.rel[0], p.rel[1], p.rel[2], (float)p.ndrop, (float)b.instr, 0.f, 0.f};
  std::memcpy(pv, v, sizeof v);
  for (int k = 0; k < gmap::KSLOT; ++k) if (tok.name_id[k] >= 0) ep.names.insert(tok.name_id[k]);
  if (sl && sl->st.valid && sl->st.ok) { ep.chosen[0] = sl->st.sx; ep.chosen[1] = sl->st.sy; ep.chosen[2] = sl->st.syaw; ep.chosen[3] = sl->st.dp; }
  ep.instr = b.instr;
}

}  // namespace rec
