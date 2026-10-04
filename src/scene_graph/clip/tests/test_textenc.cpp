// 글 인코더 = PyTorch(open_clip) 인가(tools/export_siglip2_text.py 의 text_parity.bin: 한국어·일본어·중국어·문장 부호·긴 글 섞은 질의).
//   test_sgclip_textenc DIR      (DIR 에 siglip2_b32_tok.bin · siglip2_b32_tokemb.f16 · siglip2_b32_text_fp16.plan · text_parity.bin)
// 확인: 토큰 번호가 HF 토크나이저와 전부 같음, 임베딩 코사인 최소 ≥ 0.998·평균 ≥ 0.9995, 배치 1 과 배치 n 이 같은 답, ms.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sgclip.h"

int main(int argc, char** argv) {
  sgc_text_config c;
  sgc_text_default_config(&c, argc > 1 ? argv[1] : nullptr);
  const std::string par = std::string(argc > 1 ? argv[1] : std::filesystem::path(c.tokenizer).parent_path().string()) + "/text_parity.bin";
  if (!std::filesystem::exists(c.tokenizer) || !std::filesystem::exists(par)) {
    std::printf("skip: tokenizer or text_parity.bin missing\n");
    return 77;
  }
  std::ifstream f(par, std::ios::binary);
  int32_t h[2];
  f.read(reinterpret_cast<char*>(h), 8);
  const int n = h[1];
  std::vector<std::string> q(static_cast<size_t>(n));
  std::vector<int32_t> ids(size_t(n) * SGC_TEXT_CTX);
  std::vector<float> ref(size_t(n) * SGC_DIM);
  for (int i = 0; i < n; ++i) {
    uint16_t L = 0;
    f.read(reinterpret_cast<char*>(&L), 2);
    q[size_t(i)].resize(L);
    f.read(q[size_t(i)].data(), L);
    f.read(reinterpret_cast<char*>(&ids[size_t(i) * SGC_TEXT_CTX]), SGC_TEXT_CTX * 4);
    f.read(reinterpret_cast<char*>(&ref[size_t(i) * SGC_DIM]), SGC_DIM * 4);
  }
  if (!f) { std::printf("bad parity file\n"); return 1; }
  const bool engine = std::filesystem::exists(c.engine) && std::filesystem::exists(c.tok_emb);
  if (!engine) c.engine = nullptr;
  char err[256] = {0};
  const auto t0 = std::chrono::steady_clock::now();
  sgc_text* T = sgc_text_create(&c, err, sizeof(err));
  if (!T) { std::printf("create: %s\n", err); return 1; }
  std::printf("open %.0f ms (tokenizer%s), device %.0f MB\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), engine ? " + engine" : " only",
              double(sgc_text_device_bytes(T)) / (1 << 20));
  int fails = 0;
  for (int i = 0; i < n; ++i) {
    int32_t mine[SGC_TEXT_CTX];
    sgc_text_tokenize(T, q[size_t(i)].c_str(), mine);
    if (!std::equal(mine, mine + SGC_TEXT_CTX, &ids[size_t(i) * SGC_TEXT_CTX])) {
      ++fails;
      std::printf("TOKENS DIFFER [%s]\n  hf  :", q[size_t(i)].c_str());
      for (int k = 0; k < 12; ++k) std::printf(" %d", ids[size_t(i) * SGC_TEXT_CTX + size_t(k)]);
      std::printf("\n  mine:");
      for (int k = 0; k < 12; ++k) std::printf(" %d", mine[k]);
      std::printf("\n");
    }
  }
  std::printf("tokens: %d / %d queries identical to HF\n", n - fails, n);
  if (engine) {
    std::vector<const char*> p;
    for (auto& s : q) p.push_back(s.c_str());
    std::vector<float> out(size_t(n) * SGC_DIM), one(SGC_DIM);
    if (sgc_text_encode(T, p.data(), n, out.data()) != 0) { std::printf("encode failed\n"); return 1; }
    double mn = 1, mean = 0, maxd = 0;
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int d = 0; d < SGC_DIM; ++d) s += double(out[size_t(i) * SGC_DIM + size_t(d)]) * ref[size_t(i) * SGC_DIM + size_t(d)];
      mn = std::min(mn, s), mean += s / n;
      sgc_text_encode(T, &p[size_t(i)], 1, one.data());
      for (int d = 0; d < SGC_DIM; ++d) maxd = std::max(maxd, double(std::fabs(one[size_t(d)] - out[size_t(i) * SGC_DIM + size_t(d)])));
      if (s < 0.998) std::printf("  low cosine %.5f [%s]\n", s, q[size_t(i)].c_str());
    }
    std::vector<double> ms;
    for (int r = 0; r < 30; ++r) {
      sgc_text_encode(T, &p[0], 1, one.data());
      ms.push_back(sgc_text_last_ms(T));
    }
    std::sort(ms.begin(), ms.end());
    std::printf("engine vs PyTorch FP32: cosine min %.5f mean %.6f; batch-1 vs batch-%d max |diff| %.2e; one query %.2f ms (p50) %.2f (max)\n", mn,
                mean, n, maxd, ms[ms.size() / 2], ms.back());
    if (mn < 0.998 || mean < 0.9995) ++fails;
    if (maxd > 2e-3) ++fails;
  } else {
    std::printf("engine not built: token check only\n");
  }
  sgc_text_destroy(T);
  std::printf(fails ? "FAIL\n" : "OK\n");
  return fails ? 1 : 0;
}
