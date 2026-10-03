// memstore.hpp 구현.
#include "memstore.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace sgclip {

std::string embSha(const uint16_t* f16, int n) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const uint8_t*>(f16);
  for (size_t i = 0; i < size_t(n) * 2; ++i) h = (h ^ p[i]) * 1099511628211ull;
  char b[17];
  std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(h));
  return b;
}

bool writeAtomic(const std::string& path, const void* data, size_t n) {
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;
    f.write(static_cast<const char*>(data), std::streamsize(n));
    if (!f) return false;
  }
  fs::rename(tmp, path, ec);
  return !ec;
}

std::string embRel(uint32_t id) { return "objects/O" + std::to_string(id) + "_emb.f16"; }

bool writeEmb(const std::string& dir, uint32_t id, const uint16_t* f16) {
  return writeAtomic(dir + "/" + embRel(id), f16, size_t(SGC_DIM) * 2);
}

bool readEmb(const std::string& path, std::vector<uint16_t>* f16) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f16->resize(SGC_DIM);
  f.read(reinterpret_cast<char*>(f16->data()), SGC_DIM * 2);
  return f.gcount() == SGC_DIM * 2;
}

bool NameCache::load(const std::string& dir) {
  obj.clear();
  table_sha.clear();
  std::ifstream f(dir + "/cache/names.json");
  if (!f) return false;
  try {
    const nlohmann::json j = nlohmann::json::parse(f);
    table_name = j["table"].value("name", "");
    table_version = j["table"].value("version", "");
    table_sha = j["table"].value("sha", "");
    for (auto it = j["objects"].begin(); it != j["objects"].end(); ++it) {
      const std::string k = it.key();
      if (k.size() < 2 || k[0] != 'O') continue;
      NameEntry e;
      const auto& v = it.value();
      e.emb_sha = v.value("emb_sha", "");
      for (const auto& x : v["en"]) e.en.emplace_back(x[0].get<std::string>(), x[1].get<float>());
      for (const auto& x : v["ko"]) e.ko.emplace_back(x[0].get<std::string>(), x[1].get<float>());
      e.level = v.value("level", "");
      e.level_ko = v.value("level_ko", "");
      e.general = v.value("general", "");
      e.general_ko = v.value("general_ko", "");
      e.score = v.value("score", 0.f);
      e.prob = v.value("prob", 0.f);
      e.margin = v.value("margin", 0.f);
      e.rolled = v.value("rolled", false);
      e.structural = v.value("structural", false);
      obj[uint32_t(std::stoul(k.substr(1)))] = std::move(e);
    }
    return true;
  } catch (...) {
    obj.clear();
    table_sha.clear();
    return false;
  }
}

int NameCache::refresh(const sgc_labels* L, const std::vector<ObjRef>& objs, const sgc_lookup_params* p) {
  if (!L) return 0;
  const std::string sha = sgc_labels_sha(L);
  if (sha != table_sha) obj.clear();
  table_sha = sha;
  const std::string nm = sgc_labels_name(L);
  const size_t dash = nm.rfind('-');
  table_name = nm.substr(0, dash);
  table_version = dash == std::string::npos ? "" : nm.substr(dash + 1);
  std::map<uint32_t, NameEntry> keep;
  int n = 0;
  for (const ObjRef& o : objs) {
    auto it = obj.find(o.id);
    if (it != obj.end() && it->second.emb_sha == o.emb_sha) {
      keep[o.id] = std::move(it->second);
      continue;
    }
    sgc_names r{};
    sgc_lookup_params pm;   // 기본: 집 물건 주 표(tier main)만 — 평가 crop 이름 정답 0.29/0.36(시연/깨끗) 대 표 전체 0.20/0.24
    sgc_default_lookup(&pm);
    pm.main_only = 1;
    if (sgc_labels_names(L, o.emb, &r, p ? p : &pm) != 0) continue;
    NameEntry e;
    e.emb_sha = o.emb_sha;
    for (int i = 0; i < r.n; ++i) {
      e.en.emplace_back(r.top[i].en, r.top[i].score);
      if (r.top[i].ko && *r.top[i].ko) e.ko.emplace_back(r.top[i].ko, r.top[i].score);
    }
    // 보여 줄 이름 = 1위(상위어로 올리면 정답률이 조금 떨어짐 — 상위어는 general 로 따로)
    e.level = r.top[0].en ? r.top[0].en : "";
    e.level_ko = r.top[0].ko ? r.top[0].ko : "";
    if (r.rolled) {
      e.general = r.level_en ? r.level_en : "";
      e.general_ko = r.level_ko ? r.level_ko : "";
    }
    e.score = r.level_score;
    e.prob = r.prob;
    e.margin = r.margin;
    e.rolled = r.rolled;
    e.structural = r.structural;
    keep[o.id] = std::move(e);
    ++n;
  }
  obj.swap(keep);
  return n;
}

namespace {
nlohmann::json pairs(const std::vector<std::pair<std::string, float>>& v) {
  nlohmann::json a = nlohmann::json::array();
  for (const auto& [s, f] : v) a.push_back({s, std::round(double(f) * 1e4) / 1e4});
  return a;
}
double r4(float f) { return std::round(double(f) * 1e4) / 1e4; }
}  // namespace

bool NameCache::save(const std::string& dir) const {
  nlohmann::json j;
  j["table"] = {{"name", table_name}, {"version", table_version}, {"sha", table_sha}};
  j["objects"] = nlohmann::json::object();
  for (const auto& [id, e] : obj)
    j["objects"]["O" + std::to_string(id)] = {{"emb_sha", e.emb_sha}, {"en", pairs(e.en)},     {"ko", pairs(e.ko)},
                                              {"level", e.level},     {"level_ko", e.level_ko}, {"general", e.general}, {"general_ko", e.general_ko}, {"score", r4(e.score)},
                                              {"prob", r4(e.prob)},   {"margin", r4(e.margin)}, {"rolled", e.rolled},
                                              {"structural", e.structural}};
  const std::string s = j.dump(1);
  return writeAtomic(dir + "/cache/names.json", s.data(), s.size());
}

const NameEntry* NameCache::get(uint32_t id) const {
  auto it = obj.find(id);
  return it == obj.end() ? nullptr : &it->second;
}

std::string NameCache::nodeMeta(uint32_t id, const std::string& emb_sha, double stamp) const {
  nlohmann::json m;
  m["emb"] = {{"path", embRel(id)}, {"sha", emb_sha}, {"dim", SGC_DIM}, {"dtype", "f16"}, {"model", "siglip2_b32_256_maskmap"},
              {"stamp", stamp}};
  std::string out = "\"emb\":" + m["emb"].dump();
  if (const NameEntry* e = get(id)) {
    nlohmann::json n = {{"en", e->level},           {"ko", e->level_ko},     {"score", r4(e->score)},
                        {"prob", r4(e->prob)},      {"general", e->general},
                        {"general_ko", e->general_ko},    {"rolled", e->rolled},
                        {"structural", e->structural}, {"table", table_sha}, {"top", pairs(e->en)}};
    out += ",\"names\":" + n.dump();
  }
  return out;
}

}  // namespace sgclip
