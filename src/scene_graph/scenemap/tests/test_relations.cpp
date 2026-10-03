// Object graph edges (sgraph.cpp). Map_Vla design: NO object-object prepositions (on / in / near). Every object keeps its position and box
// as metadata, so a consumer (the LLM) infers "on top of / inside" itself. The only object links are parents: room -> object, place -> object.
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "scenemap/rooms.hpp"
#include "scenemap/sgraph.hpp"

using namespace scenemap;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static ObjIn box(uint32_t id, double cx, double cy, double cz, double sx, double sy, double sz) {
  ObjIn o{};
  o.id = id;
  o.name = "obj" + std::to_string(id);
  o.pos[0] = cx; o.pos[1] = cy; o.pos[2] = cz;
  o.lo[0] = cx - sx / 2; o.lo[1] = cy - sy / 2; o.lo[2] = cz - sz / 2;
  o.hi[0] = cx + sx / 2; o.hi[1] = cy + sy / 2; o.hi[2] = cz + sz / 2;
  o.state = SM_SEEN;
  o.movable = 1;
  return o;
}

static int objectObjectEdges(const GraphView& v) {
  int n = 0;
  for (const GEdge& e : v.edges) n += symChar(e.a) == 'O' && symChar(e.b) == 'O';
  return n;
}

int main() {
  std::printf("test_relations\n");
  // objects that the old rules would have linked: a cup in a cabinet, a cup on a table, a blob that "contains" everything, objects 0.3 m apart
  std::vector<ObjIn> objs = {
      box(1, 0, 0, 0.45, 0.6, 0.6, 0.9),     // cabinet
      box(2, 0, 0, 0.45, 0.1, 0.1, 0.1),     // cup inside it   (old: in)
      box(3, 3, 0, 0.375, 1.2, 0.8, 0.75),   // table
      box(4, 3, 0, 0.80, 0.1, 0.1, 0.1),     // cup on it       (old: on)
      box(5, 3.3, 0.3, 0.5, 0.2, 0.2, 0.2),  // near the table  (old: near)
      box(6, 6, 0, 1.2, 3.9, 3.2, 2.4),      // merged blob     (old: absorbed everything under it)
      box(7, 6, 0, 0.5, 0.1, 0.1, 0.1),
  };
  SceneGraph g;
  g.updateObjects(objs);
  // a room that holds objects 1 and 2
  auto rs = std::make_shared<RoomSeg>();
  RoomGeom r;
  r.id = 1;
  r.area_m2 = 12.0;
  r.centroid[0] = 0; r.centroid[1] = 0;
  r.bmin[0] = -3; r.bmin[1] = -3; r.bmax[0] = 3; r.bmax[1] = 3;
  rs->rooms.push_back(r);
  g.updateRooms(rs, {{1, 1}, {2, 1}, {3, 1}}, {"kitchen"});
  auto v = g.publish();

  CHECK(objectObjectEdges(*v) == 0, "object-object edges: %d (on/in/near must not be produced)", objectObjectEdges(*v));

  // object boxes stay available as metadata for the consumer
  for (const ObjIn& o : objs) {
    const GNode* n = v->find(nodeSym('O', o.id));
    CHECK(n != nullptr, "object %u missing", o.id);
    if (!n) continue;
    bool same = true;
    for (int k = 0; k < 3; ++k) same = same && std::fabs(n->pos[k] - o.pos[k]) < 1e-9 && std::fabs(n->lo[k] - o.lo[k]) < 1e-9 && std::fabs(n->hi[k] - o.hi[k]) < 1e-9;
    CHECK(same, "object %u position/box changed", o.id);
  }

  // room -> object parent edges exist for the assigned objects
  int ro = 0;
  bool has1 = false, has3 = false;
  for (const GEdge& e : v->edges)
    if (e.rel == kRelGeneric && symChar(e.a) == 'R' && symChar(e.b) == 'O') {
      ++ro;
      has1 = has1 || e.b == nodeSym('O', 1);
      has3 = has3 || e.b == nodeSym('O', 3);
    }
  CHECK(ro == 3 && has1 && has3, "room->object edges %d (want 3)", ro);
  const GNode* room = v->find(nodeSym('R', 1));
  CHECK(room && room->name == "kitchen", "room name");
  // three layers only: no BUILDINGS node, no building -> room edge
  CHECK(v->layer_off[SM_GL_BUILDINGS] == v->layer_off[SM_GL_BUILDINGS + 1], "building nodes: %d", v->layer_off[SM_GL_BUILDINGS + 1] - v->layer_off[SM_GL_BUILDINGS]);
  int b = 0;
  for (const GEdge& e : v->edges) b += symChar(e.a) == 'B' || symChar(e.b) == 'B';
  CHECK(b == 0, "building edges: %d", b);

  if (g_fail) { std::printf("test_relations: %d failed\n", g_fail); return 1; }
  std::printf("  ok (object-object edges 0, room->object edges %d)\n", ro);
  return 0;
}
