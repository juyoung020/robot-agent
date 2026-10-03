// 라벨 찾기 = 전부 훑기(기준)와 맞는가 + µs.
//   test_sgclip_lookup LABEL_DIR INDEX_DIR [QUERIES_F32]   (QUERIES: Q × 768 FP32, 없으면 라벨 글 + 잡음)
// 확인: (1) 돌려준 점수 = 그 줄의 정확한 768-d 내적, 점수 순  (2) nprobe = 전부·rerank = 전부면 정확히 같은 상위 5
//       (3) 색인 캐시를 다시 읽어도 같은 답. 기본 설정의 1위 일치·정확 1위가 상위 5 안 비율과 시간은 찍기만 한다.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "sgclip.h"

int main(int argc, char** argv) {
  if (argc < 3 || !std::filesystem::exists(std::string(argv[1]) + "/manifest.json")) {
    std::printf("skip: no label table\n");
    return 77;
  }
  std::filesystem::create_directories(argv[2]);
  char err[256] = {0};
  auto t0 = std::chrono::steady_clock::now();
  sgc_labels* L = sgc_labels_open(argv[1], argv[2], err, sizeof(err));
  if (!L) { std::printf("open: %s\n", err); return 1; }
  std::printf("open %.0f ms, %d rows, sha %s, simd %s\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), sgc_labels_count(L), sgc_labels_sha(L),
              sgc_simd());
  const int D = SGC_DIM;
  std::vector<float> Q;
  if (argc > 3 && std::filesystem::exists(argv[3])) {
    std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
    Q.resize(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(Q.data()), std::streamsize(Q.size() * 4));
  } else {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 0.03f);
    std::vector<float> t(D);
    for (int i = 0; i < 300; ++i) {
      sgc_labels_text_emb(L, int(rng() % sgc_labels_count(L)), t.data());
      for (float& v : t) v += nd(rng);
      Q.insert(Q.end(), t.begin(), t.end());
    }
  }
  const int nq = int(Q.size() / D);
  for (int i = 0; i < nq; ++i) {   // L2
    double s = 0;
    for (int d = 0; d < D; ++d) s += double(Q[size_t(i) * D + d]) * Q[size_t(i) * D + d];
    for (int d = 0; d < D; ++d) Q[size_t(i) * D + d] /= float(std::sqrt(s));
  }
  sgc_lookup_params pe, pd, pall;
  sgc_default_lookup(&pd);
  pe = pd; pe.exact = 1;
  pall = pd; pall.nprobe = 256; pall.rerank = sgc_labels_count(L);
  int fails = 0, agree = 0, in5 = 0;
  double us_d = 0, us_e = 0, us_n = 0;
  std::vector<float> row(D);
  for (int i = 0; i < nq; ++i) {
    const float* q = &Q[size_t(i) * D];
    sgc_hit he[5], hd[5], ha[5];
    auto a = std::chrono::steady_clock::now();
    sgc_labels_lookup(L, q, 5, he, &pe);
    auto b = std::chrono::steady_clock::now();
    const int n = sgc_labels_lookup(L, q, 5, hd, &pd);
    auto c = std::chrono::steady_clock::now();
    sgc_names nm;
    sgc_labels_names(L, q, &nm, &pd);
    auto e = std::chrono::steady_clock::now();
    us_e += std::chrono::duration<double, std::micro>(b - a).count();
    us_d += std::chrono::duration<double, std::micro>(c - b).count();
    us_n += std::chrono::duration<double, std::micro>(e - c).count();
    for (int k = 0; k < n; ++k) {
      sgc_labels_text_emb(L, hd[k].row, row.data());
      double s = 0;
      for (int d = 0; d < D; ++d) s += double(row[d]) * q[d];
      if (std::abs(s - hd[k].score) > 1e-4 || (k && hd[k].score > hd[k - 1].score + 1e-7)) ++fails;
    }
    sgc_labels_lookup(L, q, 5, ha, &pall);
    for (int k = 0; k < 5; ++k)
      if (ha[k].row != he[k].row && std::abs(ha[k].score - he[k].score) > 1e-6) ++fails;
    agree += hd[0].row == he[0].row;
    for (int k = 0; k < 5; ++k) in5 += hd[k].row == he[0].row;
  }
  // 캐시 다시 읽기
  sgc_labels* L2 = sgc_labels_open(argv[1], argv[2], err, sizeof(err));
  for (int i = 0; i < std::min(nq, 50); ++i) {
    sgc_hit a[3], b[3];
    sgc_labels_lookup(L, &Q[size_t(i) * D], 3, a, nullptr);
    sgc_labels_lookup(L2, &Q[size_t(i) * D], 3, b, nullptr);
    for (int k = 0; k < 3; ++k) fails += a[k].row != b[k].row;
  }
  std::printf("%d queries: default (nprobe %d, rerank %d) top-1 = exact %.3f, exact top-1 in top-5 %.3f | us/query: ivf+bits+rerank %.1f, "
              "names(16 + roll-up) %.1f, exact %.0f | errors %d\n",
              nq, pd.nprobe, pd.rerank, double(agree) / nq, double(in5) / nq, us_d / nq, us_n / nq, us_e / nq, fails);
  const int r = sgc_labels_find(L, "radio");
  const int rk = sgc_labels_find(L, "라디오");
  std::printf("find radio -> %d, 라디오 -> %d\n", r, rk);
  sgc_labels_close(L);
  sgc_labels_close(L2);
  return fails ? 1 : 0;
}
