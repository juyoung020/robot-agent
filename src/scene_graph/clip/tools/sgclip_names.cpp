// 기억 폴더 이름 캐시를 오프라인으로 새로 고침(docs/clip_candidates.md 3.5 다시 만드는 조건):
//   sgclip_names MEMORY_DIR LABEL_DIR
// objects/O<id>_emb.f16 을 모두 읽어 emb_sha 를 셈 → cache/names.json 과 비교해 표 sha 가 바뀌었거나 emb_sha 가 바뀐 물체만 다시 뽑고
// 원자적으로 씀. 라벨 색인은 MEMORY_DIR/cache/index/. 다시 뽑은 수와 이름을 찍는다.
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "memstore.hpp"

namespace fs = std::filesystem;
using namespace sgclip;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: sgclip_names MEMORY_DIR LABEL_DIR\n");
    return 2;
  }
  const std::string mem = argv[1];
  char err[256] = {0};
  const std::string idx = mem + "/cache/index";
  fs::create_directories(idx);
  sgc_labels* L = sgc_labels_open(argv[2], idx.c_str(), err, sizeof(err));
  if (!L) { std::fprintf(stderr, "labels: %s\n", err); return 1; }
  std::vector<uint32_t> ids;
  std::vector<std::string> shas;
  std::vector<std::vector<float>> embs;
  for (const auto& e : fs::directory_iterator(mem + "/objects")) {
    const std::string n = e.path().filename().string();
    if (n.size() < 10 || n[0] != 'O' || n.substr(n.size() - 8) != "_emb.f16") continue;
    std::vector<uint16_t> h;
    if (!readEmb(e.path().string(), &h)) continue;
    ids.push_back(uint32_t(std::stoul(n.substr(1, n.size() - 9))));
    shas.push_back(embSha(h.data()));
    embs.emplace_back(SGC_DIM);
    sgc_f16_to_f32(h.data(), embs.back().data(), SGC_DIM);
  }
  std::vector<ObjRef> objs;
  for (size_t i = 0; i < ids.size(); ++i) objs.push_back({ids[i], shas[i], embs[i].data()});
  NameCache c;
  c.load(mem);
  const std::string old = c.table_sha;
  const int n = c.refresh(L, objs);
  c.save(mem);
  std::printf("%zu objects, table %s (cache had %s): re-named %d\n", objs.size(), sgc_labels_sha(L), old.empty() ? "-" : old.c_str(), n);
  for (const auto& [id, e] : c.obj)
    std::printf("  O%u  %-22s %-12s %.3f%s%s\n", id, e.level.c_str(), e.level_ko.c_str(), e.score, e.rolled ? " (roll-up)" : "",
                e.structural ? " [structural]" : "");
  sgc_labels_close(L);
  return 0;
}
