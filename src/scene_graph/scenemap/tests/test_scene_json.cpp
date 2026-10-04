// scene.json in the trimmed Spark-DSG schema (Map_Vla): three layers OBJECTS(+AGENTS), PLACES, ROOMS. No BUILDINGS layer, no frontier flag,
// no mesh / GVD fields on places. The written file must load with the trimmed Spark-DSG and carry the right values.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "scenemap/dsg_save.hpp"
#include "scenemap/sgraph.hpp"

#ifdef SM_TEST_SPARK_DSG
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#endif

using namespace scenemap;
namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string slurp(const fs::path& p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

int main() {
  std::printf("test_scene_json\n");
  // an 8 x 6 m free room (res 0.1), walls on the border; unknown outside is not needed
  const int w = 80, h = 60;
  std::vector<int8_t> cells(size_t(w) * h, 0);
  for (int x = 0; x < w; ++x) cells[x] = cells[size_t(h - 1) * w + x] = 100;
  for (int y = 0; y < h; ++y) cells[size_t(y) * w] = cells[size_t(y) * w + w - 1] = 100;
  SceneGraph g;
  g.updateAgent(0.0, Pose2{1.0, 1.0, 0.0});
  g.updateAgent(5.0, Pose2{3.0, 1.0, 0.5});
  g.updatePlaces(cells.data(), w, h, 0, 0, 0.1, 1.0, nullptr, true);
  sm_object o{};
  o.id = 7; o.name = "cup"; o.score = 0.9f; o.n_obs = 5; o.state = SM_SEEN;
  o.pos[0] = 2; o.pos[1] = 2; o.pos[2] = 0.8;
  o.extent[0] = o.extent[1] = o.extent[2] = 0.1;
  std::vector<ObjIn> oi(1);
  oi[0].id = 7; oi[0].name = "cup"; oi[0].state = SM_SEEN; oi[0].movable = 1;
  for (int k = 0; k < 3; ++k) { oi[0].pos[k] = o.pos[k]; oi[0].lo[k] = o.pos[k] - 0.05; oi[0].hi[k] = o.pos[k] + 0.05; }
  g.updateObjects(oi);
  auto view = g.publish();
  int n_places = 0;
  for (const GNode& n : view->nodes) n_places += n.layer == 3;
  CHECK(n_places >= 6, "places %d", n_places);

  SaveInput in;
  in.stamp = 5.0;
  in.pose[0] = 3; in.pose[1] = 1; in.pose[2] = 0.5;
  in.objs = &o;
  in.n_objs = 1;
  in.grid_res = 0.1; in.grid_w = w; in.grid_h = h;
  in.cells = cells.data();
  in.graph = view;
  in.events = {ObjEvent{4.0, 7, 6, {2, 2, 0.8}}, ObjEvent{5.0, 7, 7, {2, 2, 0.8}}};   // 다시 보임, 병합(objmap 이 7 로 기록)
  const fs::path dir = fs::temp_directory_path() / "sm_test_scene_json";
  fs::remove_all(dir);
  CHECK(saveScene(in, dir.string()) == 0, "saveScene");

  const std::string sj = slurp(dir / "scene.json"), vj = slurp(dir / "view.json");
  CHECK(!sj.empty() && !vj.empty(), "files written");
  CHECK(sj.find("frontier") == std::string::npos && vj.find("frontier") == std::string::npos, "no frontier flag anywhere");
  CHECK(sj.find("mesh") == std::string::npos, "no mesh fields");
  CHECK(sj.find("BUILDINGS") == std::string::npos && vj.find("\"building\"") == std::string::npos, "no building layer");
  CHECK(sj.find("real_place") == std::string::npos && sj.find("num_basis_points") == std::string::npos, "no GVD place fields");
  CHECK(vj.find("\"kind\":\"seen_again\"") != std::string::npos && vj.find("\"kind\":\"merged\"") != std::string::npos &&
            vj.find("\"kind\":\"?\"") == std::string::npos, "event names (7 = merged)");

#ifdef SM_TEST_SPARK_DSG
  using namespace spark_dsg;
  auto G = DynamicSceneGraph::load(dir / "scene.json");
  CHECK(G != nullptr, "load with the trimmed Spark-DSG");
  if (G) {
    CHECK(G->hasLayer(DsgLayers::OBJECTS) && G->hasLayer(DsgLayers::PLACES) && G->hasLayer(DsgLayers::ROOMS), "layers");
    CHECK(G->numLayers() == 3, "layer ids %zu (2 = OBJECTS + AGENTS partition, 3 = PLACES, 4 = ROOMS)", G->numLayers());
    const auto& P = G->getLayer(DsgLayers::PLACES);
    CHECK(int(P.numNodes()) == n_places, "place nodes %zu vs %d", P.numNodes(), n_places);
    for (const auto& [nid, node] : P.nodes()) {
      const auto& a = node->attributes<PlaceNodeAttributes>();
      CHECK(a.distance > 0 && a.distance <= 1.0 + 1e-9, "place distance %f", a.distance);
    }
    CHECK(G->hasLayer(2, 'a') && G->getLayer(DsgLayers::AGENTS).id == LayerKey(2, 97), "AGENTS name must map to partition a, not to OBJECTS");
    const auto& A = G->getLayer(DsgLayers::AGENTS);
    CHECK(A.numNodes() == 2, "agent nodes %zu", A.numNodes());
    const auto& O = G->getLayer(DsgLayers::OBJECTS);
    CHECK(O.numNodes() == 1, "object nodes %zu", O.numNodes());
    for (const auto& [nid, node] : O.nodes()) CHECK(node->attributes().position.isApprox(Eigen::Vector3d(2, 2, 0.8)), "object position");
    std::printf("  scene.json reload: %zu places, %zu agents, %zu objects\n", P.numNodes(), A.numNodes(), O.numNodes());
  }
#endif
  if (!g_fail && !std::getenv("SM_KEEP")) fs::remove_all(dir);  // SM_KEEP=1 keeps /tmp/sm_test_scene_json for manual checks
  if (g_fail) { std::printf("test_scene_json: %d failed\n", g_fail); return 1; }
  std::printf("  ok\n");
  return 0;
}
