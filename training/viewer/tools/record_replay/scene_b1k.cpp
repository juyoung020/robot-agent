// scene_b1k — BEHAVIOR 장면(b1kconv 의 RASC 파일, ~/ra_b1k/*.rasc)을 재생 탭에서 볼 수 있게 프레임 하나짜리 .trp 로 쓴다(B1–B5 준비).
// 로더는 training/RL/tools/b1kconv/cpp/rasc.h 를 읽기만 한다. 담는 것: 벽·문·가구 상자(yaw 상자), 방(이름·범위), 다닐 곳 격자(TRAV_NO_OBJ: 벽만 막힘),
// 과제 인스턴스의 로봇 시작 자세와 과제 물체(그 인스턴스 자세, 머리 objects + 슬롯).
//
//   scene_b1k --out RUN_DIR [--split scenes] [--inst K] ~/ra_b1k/Rs_int.rasc …
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "rasc.h"
#include "rec_util.h"

using namespace rec;

static float yaw_of(const float q[4]) { return std::atan2(2.f * (q[3] * q[2] + q[0] * q[1]), 1.f - 2.f * (q[1] * q[1] + q[2] * q[2])); }

int main(int argc, char** argv) {
  std::string out, split = "scenes";
  int inst_k = 0;
  std::vector<std::string> files;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    if (s == "--out" && a + 1 < argc) out = argv[++a];
    else if (s == "--split" && a + 1 < argc) split = argv[++a];
    else if (s == "--inst" && a + 1 < argc) inst_k = std::stoi(argv[++a]);
    else files.push_back(s);
  }
  if (out.empty() || files.empty()) { std::fprintf(stderr, "usage: scene_b1k --out RUN_DIR [--split scenes] [--inst K] SCENE.rasc ...\n"); return 2; }
  Out o(out, split);
  long ep = o.next_ep();
  for (const std::string& fp : files) {
    rasc::Scene S;
    std::string err;
    if (!rasc::load(S, fp.c_str(), &err)) { std::fprintf(stderr, "%s: %s\n", fp.c_str(), err.c_str()); continue; }
    const RascFileHeader& H = *S.h;
    // 상자: 벽·문·가구(바닥·천장은 BOXES 에 없음). 창은 벽과 같이 그림
    std::string boxes = "[";
    int nb = 0;
    for (const RascBoxRec& b : S.boxes) {
      const char* kind = (b.flags & RASC_F_WALL) ? "wall" : (b.flags & RASC_F_DOOR) ? "door" : (b.flags & RASC_F_WINDOW) ? "window" : (b.flags & RASC_F_CARPET) ? "carpet" : "furniture";
      if (nb++) boxes += ',';
      const std::string cat = b.cat < S.cats.size() ? S.str(S.cats[b.cat].name) : "?";
      boxes += Obj().str("kind", kind).str("name", cat).raw("c", arr3(b.center)).raw("h", arr3(b.half)).num("yaw", b.yaw).done();
    }
    // 과제 인스턴스(학습 나눔의 k 번째) → 로봇 시작 자세 + 과제 물체 자세
    const RascInstRec* I = S.insts.size() ? &S.insts[std::min<size_t>(inst_k, S.insts.size() - 1)] : nullptr;
    std::string task = "", objs = "[";
    std::vector<float> slots(16 * 15, 0.f);
    int ns = 0;
    if (I) {
      const RascTaskRec& T = S.tasks[I->task];
      task = S.str(T.name);
      auto po = S.poses_of(*I);
      auto to = S.objs_of(T);
      for (size_t k = 0; k < to.size() && k < po.size(); ++k) {
        const RascTaskObjRec& t = to[k];
        if (t.flags & (RASC_F_AGENT | RASC_F_SYSTEM) || po[k].src == 0 || !(t.half[0] > 0)) continue;
        if (t.scene_obj >= 0) continue;   // 장면 물체는 이미 상자에 있음 — 옮길 수 있는 과제 물체만 슬롯으로
        if (ns >= 16) break;
        const float y = yaw_of(po[k].quat), cy = std::cos(y), sy = std::sin(y);
        const float cx = po[k].pos[0] + cy * t.offset[0] - sy * t.offset[1], cyy = po[k].pos[1] + sy * t.offset[0] + cy * t.offset[1], cz = po[k].pos[2] + t.offset[2];
        const float v[15] = {(float)ns, cx, cyy, cz, cx, cyy, cz, 2 * t.half[0], 2 * t.half[1], 2 * t.half[2], 1.f, 0.f, 0.f, 0.f, 0.f};
        for (int q = 0; q < 15; ++q) slots[ns * 15 + q] = v[q];
        if (ns) objs += ',';
        objs += Obj().num("id", ns).num("slot", ns).str("name", S.str(t.inst)).str("cat", t.cat < S.cats.size() ? S.str(S.cats[t.cat].name) : "?").done();
        ++ns;
      }
    }
    objs += "]";
    std::string rooms = "[";
    for (size_t k = 0; k < S.rooms.size(); ++k) {
      const RascRoomRec& r = S.rooms[k];
      if (k) rooms += ',';
      rooms += Obj().str("name", S.str(r.name)).raw("bmin", "[" + jnum(r.bmin[0]) + "," + jnum(r.bmin[1]) + "]").raw("bmax", "[" + jnum(r.bmax[0]) + "," + jnum(r.bmax[1]) + "]")
                   .raw("centroid", "[" + jnum(r.centroid[0]) + "," + jnum(r.centroid[1]) + "]").num("area_m2", r.area_m2).done();
    }
    rooms += "]";
    const std::string scene = Obj().str("kind", "behavior").str("name", S.name()).raw("boxes", boxes + "]").raw("rooms", rooms).str("task", task)
                                  .num("inst_id", I ? I->inst_id : -1).str("file", fp).done();
    const float rx = I ? I->robot_pos[0] : 0.f, ry = I ? I->robot_pos[1] : 0.f, ryaw = I ? I->robot_yaw : 0.f;
    char file[160];
    std::snprintf(file, sizeof file, "ep_%06ld_scene_%s.trp", ep, S.name().c_str());
    const std::string line = Obj().num("ep", ep).str("skill", "scene").str("home", S.name()).str("stage", "B1").str("outcome", "layout").str("driver", "none")
                                 .num("t", 0).num("steps", 1).str("task", task).num("n_rooms", S.rooms.size()).num("n_boxes", nb).str("replay", file).done();
    const std::string head = Obj().raw("meta", line).num("dt", 0.1).num("stride", 1).raw("objects", objs)
        .raw("joint_map", "{\"q1\":[[\"omx_joint1\",1,0]],\"q2\":[[\"omx_joint2\",1,0]],\"q3\":[[\"omx_joint3\",1,0]],\"q4\":[[\"omx_joint4\",1,0]],\"q5\":[[\"omx_joint5\",1,0]]}")
        .raw("grid", Obj().num("res", H.cell).num("ox", H.origin[0]).num("oy", H.origin[1]).num("w", H.grid_w).num("h", H.grid_h).done()).raw("scene", scene)
        .str("slot_z", "center").b("synthetic", false).raw("source", Obj().str("kind", "behavior_rasc").str("file", fp).done()).done();
    TrpWriter* w = trp_new("t,x,y,yaw,sx,sy,syaw,vx,wz,q1,q2,q3,q4,q5,qg", "id,bx,by,bz,px,py,pz,ex,ey,ez,src,unc,state,is_tgt,age", 16, head.c_str());
    const float row[15] = {0, rx, ry, ryaw, rx, ry, ryaw, 0, 0, 0.f, 1.3f, -1.9f, 0.7f, 0.f, 0.f};
    trp_frame(w, row, slots.data());
    // 다닐 곳 격자: 벽만 막힌 층(TRAV_NO_OBJ). 다님 = 0 %(흰), 막힘 = 100 %(검), 방 밖 막힘 = 모름
    std::vector<int8_t> cells((size_t)H.grid_w * H.grid_h);
    for (uint32_t r = 0; r < H.grid_h; ++r)
      for (uint32_t c = 0; c < H.grid_w; ++c) {
        const size_t i = (size_t)r * H.grid_w + c;
        const bool fr = S.free_cell(1, (int)r, (int)c);
        cells[i] = fr ? 0 : (S.room_grid[i] ? 100 : -1);
      }
    trp_map_rect(w, 0, H.grid_w, H.grid_h, H.cell, H.origin[0], H.origin[1], 0, 0, H.grid_w - 1, H.grid_h - 1, cells.data());
    const std::string path = o.rep + "/" + file;
    const long long n = trp_finish(w, path.c_str());
    trp_free(w);
    o.append_episode(line);
    std::printf("scene_b1k: %s -> %s (%d boxes, %zu rooms, %d task objects, grid %ux%u, %.0f KB)\n", S.name().c_str(), path.c_str(), nb, S.rooms.size(), ns, H.grid_w, H.grid_h, n / 1024.0);
    ++ep;
  }
  return 0;
}
