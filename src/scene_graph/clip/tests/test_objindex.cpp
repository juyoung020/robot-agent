// 물체 찾기 색인(include/sgsearch.h) 논리 시험 — GPU·글 인코더 없이 라벨 표만.
//   test_sgclip_objindex LABEL_DIR OUT_DIR
// 가짜 기억: 물체 벡터 = 라벨 표 글 벡터 섞기(A′ 형식 objects/O<id>_views.f16 · _emb.f16), view.json.
//   O1 "fire extinguisher" 로 등록, 생김새는 라디오(+ 소화기 조금)   — 사용자 시나리오
//   O2 "cup" / O3 "straight chair" / O4 "sofa"(벡터 없음) / O5 "radio" 아닌 다른 시점 둘(A′ views 2 개)
// 확인: ① 이름(아래말 "chair" → straight chair, 한국어 "소파"), 없는 물체 0 개, ② "radio"/"라디오" → O1 생김새 후보(자동),
//       확인(user) 뒤 ① 이 바로 찾음, confirmations.jsonl 한 줄, 다시 열어도 그대로(기록 다시 적용), names.json,
//       sgs_search_vec(RecallVLA), 잘못된 인자 → error JSON.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "sgsearch.h"

namespace fs = std::filesystem;
using nlohmann::json;

static int fails = 0;
#define CHECK(c, ...)                         \
  do {                                        \
    if (!(c)) {                               \
      ++fails;                                \
      std::printf("FAIL %s: ", #c);           \
      std::printf(__VA_ARGS__);               \
      std::printf("\n");                      \
    }                                         \
  } while (0)

static std::vector<float> tv(const sgc_labels* L, const char* name) {
  std::vector<float> v(SGC_DIM, 0.f);
  const int r = sgc_labels_find_name(L, name);
  if (r >= 0) sgc_labels_text_emb(L, r, v.data());
  return v;
}

static void writeViews(const std::string& path, const std::vector<std::vector<float>>& vs) {
  std::vector<uint16_t> h;
  for (auto v : vs) {
    double s = 0;
    for (float x : v) s += double(x) * x;
    for (float& x : v) x = float(x / std::sqrt(s));
    std::vector<uint16_t> t(SGC_DIM);
    sgc_f32_to_f16(v.data(), t.data(), SGC_DIM);
    h.insert(h.end(), t.begin(), t.end());
  }
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(h.data()), std::streamsize(h.size() * 2));
}

static std::vector<float> mix(const std::vector<float>& a, float wa, const std::vector<float>& b, float wb) {
  std::vector<float> o(SGC_DIM);
  for (int d = 0; d < SGC_DIM; ++d) o[size_t(d)] = wa * a[size_t(d)] + wb * b[size_t(d)];
  return o;
}

static json search(sgs_index* X, const char* q) {
  std::vector<char> b(1 << 20);
  const int n = sgs_search_json(X, q, 5, 0, b.data(), int(b.size()));
  return n >= 0 ? json::parse(b.data()) : json{};
}

int main(int argc, char** argv) {
  if (argc < 3 || !fs::exists(std::string(argv[1]) + "/manifest.json")) {
    std::printf("skip: label table missing\n");
    return 77;
  }
  char err[256] = {0};
  sgc_labels* L = sgc_labels_open_ex(argv[1], (std::string(argv[2]) + "_idx").c_str(), nullptr, err, sizeof(err));
  if (!L) { std::printf("labels: %s\n", err); return 1; }
  const std::string mem = argv[2];
  fs::remove_all(mem);
  fs::create_directories(mem + "/objects");
  const auto radio = tv(L, "radio"), ext = tv(L, "fire extinguisher"), cup = tv(L, "cup"), chair = tv(L, "straight chair"),
             mug = tv(L, "mug");
  writeViews(mem + "/objects/O1_views.f16", {mix(radio, 1.f, ext, 0.35f)});
  writeViews(mem + "/objects/O2_emb.f16", {cup});
  writeViews(mem + "/objects/O3_views.f16", {chair, mix(chair, 1.f, cup, 0.2f)});
  writeViews(mem + "/objects/O5_views.f16", {mix(mug, 1.f, cup, 0.3f), mug});
  json v;
  v["objects"] = json::array({{{"id", 1}, {"name", "fire extinguisher"}}, {{"id", 2}, {"name", "cup"}}, {{"id", 3}, {"name", "straight chair"}},
                              {{"id", 4}, {"name", "sofa"}}, {{"id", 5}, {"name", "bottle"}}});
  std::ofstream(mem + "/view.json") << v.dump();

  sgs_config c;
  sgs_default_config(&c);
  c.mem_dir = mem.c_str();
  c.labels = L;
  sgs_index* X = sgs_open(&c, err, sizeof(err));
  if (!X) { std::printf("open: %s\n", err); return 1; }
  std::vector<char> b(1 << 16);
  sgs_stats_json(X, b.data(), int(b.size()));
  std::printf("stats %s\n", b.data());
  CHECK(sgs_count(X) == 5, "count %d", sgs_count(X));

  // ① 이름
  json r = search(X, "chair");
  CHECK(!r["hits"].empty() && r["hits"][0]["id"] == 3 && r["hits"][0]["match_type"] == "name", "chair -> O3 by name (hyponym): %s", r.dump().c_str());
  r = search(X, "소파");
  CHECK(!r["hits"].empty() && r["hits"][0]["id"] == 4 && r["hits"][0]["match_type"] == "name", "소파 -> O4: %s", r.dump().c_str());
  r = search(X, "bicycle");
  CHECK(r["hits"].empty(), "absent bicycle -> no hits: %s", r.dump().c_str());
  // ② 생김새(자동)
  for (const char* q : {"radio", "라디오"}) {
    r = search(X, q);
    std::printf("%s -> %s\n", q, r.dump().c_str());
    CHECK(r["step2"] == true && r["n_name_hits"] == 0, "%s: step 2 should run", q);
    CHECK(!r["hits"].empty() && r["hits"][0]["id"] == 1 && r["hits"][0]["match_type"] == "appearance", "%s -> O1 by appearance", q);
    if (!r["hits"].empty()) CHECK(r["hits"][0]["registered"] == "fire extinguisher", "registered name kept");
  }
  // ③ 확인 → 이름
  sgs_confirm(X, 1, "radio", "bogus", "", b.data(), int(b.size()));
  CHECK(json::parse(b.data())["status"] == "error", "bad source -> error");
  sgs_confirm(X, 99, "radio", "user", "", b.data(), int(b.size()));
  CHECK(json::parse(b.data())["status"] == "error", "unknown id -> error");
  sgs_confirm(X, 1, "라디오", "user", "라디오 가져와", b.data(), int(b.size()));
  json cf = json::parse(b.data());
  std::printf("confirm -> %s\n", cf.dump().c_str());
  CHECK(cf["status"] == "ok" && cf["p_after"].get<double>() > 0.9 && cf["p_after"].get<double>() > cf["p_before"].get<double>(), "posterior up");
  r = search(X, "radio");
  CHECK(!r["hits"].empty() && r["hits"][0]["id"] == 1 && r["hits"][0]["match_type"] == "name" && r["step2"] == false, "after confirm: name hit: %s",
        r.dump().c_str());
  int lines = 0;
  {
    std::ifstream f(mem + "/confirmations.jsonl");
    for (std::string l; std::getline(f, l);) lines += !l.empty();
  }
  CHECK(lines == 1, "confirmations.jsonl lines %d", lines);
  CHECK(fs::exists(mem + "/cache/objsearch/names.json"), "names.json written");
  sgs_close(X);
  X = sgs_open(&c, err, sizeof(err));   // 다시 열기: 기록이 다시 적용됨
  r = search(X, "라디오");
  CHECK(!r["hits"].empty() && r["hits"][0]["id"] == 1 && r["hits"][0]["match_type"] == "name", "reopen keeps confirmation: %s", r.dump().c_str());
  // 확인한 물체가 본보기 → 생김새 닮은 것(영상↔영상)은 force 로
  // RecallVLA
  sgs_slot s[3];
  const int n = sgs_search_vec(X, cup.data(), 3, s);
  CHECK(n == 3 && (s[0].id == 2 || s[0].id == 5), "vec(cup) top O2/O5: %u", n > 0 ? s[0].id : 0);
  std::printf("vec(cup): O%u %.3f (cos %.3f), O%u %.3f, O%u %.3f\n", s[0].id, s[0].score, s[0].cos, s[1].id, s[1].score, s[2].id, s[2].score);
  sgs_object_json(X, 1, b.data(), int(b.size()));
  std::printf("O1 %s\n", b.data());
  CHECK(json::parse(b.data())["name"] == "radio", "O1 name radio after confirm");
  sgs_close(X);
  sgc_labels_close(L);
  std::printf(fails ? "FAIL (%d)\n" : "OK\n", fails);
  return fails ? 1 : 0;
}
