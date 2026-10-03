// 이름 캐시 다시 만들기 규칙(memstore.hpp): 작은 라벨 표 두 판(sha 다름)을 만들어
//   (1) 처음: 전부 뽑음  (2) 같은 표·같은 emb_sha: 0 개  (3) 물체 하나 emb 바뀜: 1 개  (4) 표 sha 바뀜: 전부
//   (5) 없어진 물체는 지움  (6) names.json 다시 읽어도 같음(원자적 쓰기 뒤 .tmp 없음)  (7) emb 파일 쓰기·읽기·sha
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "memstore.hpp"

namespace fs = std::filesystem;
using namespace sgclip;

static void makeTable(const std::string& dir, const char* sha, int K, std::mt19937& rng, std::vector<float>* rows) {
  fs::create_directories(dir);
  std::ofstream(dir + "/manifest.json") << "{\"name\":\"test\",\"version\":\"v1\",\"sha\":\"" << sha
                                        << "\",\"count\":" << K << ",\"model\":\"siglip2_b32\",\"dim\":768}";
  std::ofstream t(dir + "/table.jsonl");
  const char* en[] = {"chair", "stool", "sofa", "radio", "wall", "cup"};
  const char* ko[] = {"의자", "스툴", "소파", "라디오", "벽", "컵"};
  for (int i = 0; i < K; ++i)
    t << "{\"i\":" << i << ",\"en\":\"" << en[i % 6] << (i >= 6 ? std::to_string(i) : "") << "\",\"ko\":[\"" << ko[i % 6]
      << "\"],\"synset\":\"s" << i << ".n.01\",\"hypernyms\":[\"seat.n.03\",\"furniture.n.01\",\"a.n.01\",\"b.n.01\",\"c.n.01\"],"
      << "\"tier\":\"main\",\"structural\":" << (i % 6 == 4 ? "true" : "false") << "}\n";
  std::normal_distribution<float> nd;
  rows->assign(size_t(K) * SGC_DIM, 0);
  std::vector<uint16_t> h(size_t(K) * SGC_DIM);
  for (int i = 0; i < K; ++i) {
    double s = 0;
    for (int d = 0; d < SGC_DIM; ++d) s += std::pow((*rows)[size_t(i) * SGC_DIM + d] = nd(rng), 2);
    for (int d = 0; d < SGC_DIM; ++d) (*rows)[size_t(i) * SGC_DIM + d] /= float(std::sqrt(s));
  }
  sgc_f32_to_f16(rows->data(), h.data(), K * SGC_DIM);
  std::ofstream(dir + "/text_siglip2_b32.f16", std::ios::binary).write(reinterpret_cast<const char*>(h.data()), std::streamsize(h.size() * 2));
}

int main(int argc, char** argv) {
  const std::string out = argc > 1 ? argv[1] : "test_cache_out";
  fs::remove_all(out);
  std::mt19937 rng(5);
  std::vector<float> ra, rb;
  makeTable(out + "/tA", "aaaaaaaaaaaaaaaa", 40, rng, &ra);
  makeTable(out + "/tB", "bbbbbbbbbbbbbbbb", 40, rng, &rb);
  char err[256];
  sgc_labels* A = sgc_labels_open((out + "/tA").c_str(), (out + "/mem/cache/index").c_str(), err, sizeof(err));
  sgc_labels* B = sgc_labels_open((out + "/tB").c_str(), nullptr, err, sizeof(err));
  if (!A || !B) { std::printf("open: %s\n", err); return 1; }
  int fails = 0;
  auto expect = [&](bool c, const char* what) { std::printf("%-46s %s\n", what, c ? "ok" : "FAIL"); fails += !c; };
  // 물체 3 개: 표 A 의 줄 0, 3, 4 에 가까운 벡터
  const std::string mem = out + "/mem";
  std::vector<std::vector<float>> e(3, std::vector<float>(SGC_DIM));
  std::vector<std::vector<uint16_t>> h(3, std::vector<uint16_t>(SGC_DIM));
  const int src[3] = {0, 3, 4};
  std::vector<ObjRef> objs;
  for (int k = 0; k < 3; ++k) {
    for (int d = 0; d < SGC_DIM; ++d) e[k][d] = ra[size_t(src[k]) * SGC_DIM + d];
    sgc_f32_to_f16(e[k].data(), h[k].data(), SGC_DIM);
    expect(writeEmb(mem, 10 + k, h[k].data()), "write emb");
  }
  for (int k = 0; k < 3; ++k) objs.push_back({uint32_t(10 + k), embSha(h[k].data()), e[k].data()});
  NameCache c;
  expect(!c.load(mem), "no names.json yet");
  expect(c.refresh(A, objs) == 3, "(1) first refresh: all 3");
  expect(c.get(10) && c.get(10)->level == "chair" && c.get(10)->ko[0].first == "의자", "names top-1 chair / 의자");
  expect(c.get(12) && c.get(12)->structural, "wall is structural");
  expect(c.save(mem) && !fs::exists(mem + "/cache/names.json.tmp"), "save names.json (atomic)");
  NameCache c2;
  expect(c2.load(mem) && c2.table_sha == "aaaaaaaaaaaaaaaa" && c2.obj.size() == 3, "(6) reload");
  expect(c2.refresh(A, objs) == 0, "(2) same table + same emb_sha: 0");
  for (int d = 0; d < SGC_DIM; ++d) e[1][d] = ra[size_t(2) * SGC_DIM + d];   // 물체 11 의 best view 바뀜 → sofa
  sgc_f32_to_f16(e[1].data(), h[1].data(), SGC_DIM);
  objs[1].emb_sha = embSha(h[1].data());
  expect(c2.refresh(A, objs) == 1 && c2.get(11)->level == "sofa", "(3) one emb changed: 1 (sofa)");
  expect(c2.refresh(B, objs) == 3 && c2.table_sha == "bbbbbbbbbbbbbbbb", "(4) table sha changed: all");
  objs.pop_back();
  c2.refresh(B, objs);
  expect(c2.obj.size() == 2 && !c2.get(12), "(5) gone object dropped");
  std::vector<uint16_t> back;
  expect(readEmb(mem + "/" + embRel(10), &back) && embSha(back.data()) == embSha(h[0].data()), "(7) emb file round trip + sha");
  expect(fs::exists(mem + "/cache/index/labels_aaaaaaaaaaaaaaaa_t.idx"), "label index cached in cache/index");
  const std::string m = c2.nodeMeta(10, objs[0].emb_sha, 1.5);
  expect(m.find("\"emb\":{") == 0 && m.find("\"names\":{") != std::string::npos, "node meta has emb + names");
  std::printf("node meta: %s\n", m.c_str());
  sgc_labels_close(A);
  sgc_labels_close(B);
  return fails ? 1 : 0;
}
