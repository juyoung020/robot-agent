// 물체 찾기 명령(include/sgsearch.h): 기억 폴더 하나에 찾기·확인·통계. 평가(tools/eval_objsearch.py 가 ctypes 대신 이것을 써도 됨)·시연용.
//
//   sgsearch MEM [--labels DIR] [--no-text] [--encode] [--cache DIR] [--k N] [--force] CMD …
//     search Q [Q …]              질의마다 결과 JSON 한 줄
//     bench N Q [Q …]             질의마다 N 번 돌려 µs p50 / p95(글 인코더 포함)
//     confirm ID NAME SOURCE [Q]   이름 고치기(SOURCE = user | close_look) → 결과 JSON
//     object ID                   물체 이름·속성 JSON
//     stats                       열기 통계 JSON
//   --encode: objprob 벡터가 없는 물체를 best view 사진에서 뽑음(영상 엔진 SGC_ENGINE, 기본 models/ovdet/x86_sm120/siglip2_b32/
//             siglip2_b32_mask_fp16.plan). --no-text: 글 인코더 없이(라벨 표 이름 질의만). 라벨 표 기본 SGRT_LABELS 또는
//             data/embed_work/labels/objects-v1, 색인 캐시 ~/.cache/sgclip.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sgsearch.h"

static std::string home(const char* rel) { return std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + rel; }

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: sgsearch MEM [--labels DIR] [--no-text] [--encode] [--cache DIR] [--k N] [--force] search|bench|confirm|object|stats …\n");
    return 2;
  }
  std::string mem = argv[1], labels = std::getenv("SGRT_LABELS") ? std::getenv("SGRT_LABELS") : home("/embed_work/labels/objects-v1"), cache;
  bool text = true, enc = false, force = false;
  int k = 0, i = 2;
  for (; i < argc && argv[i][0] == '-'; ++i) {
    const std::string a = argv[i];
    if (a == "--labels" && i + 1 < argc) labels = argv[++i];
    else if (a == "--cache" && i + 1 < argc) cache = argv[++i];
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--no-text") text = false;
    else if (a == "--encode") enc = true;
    else if (a == "--force") force = true;
  }
  if (i >= argc) return 2;
  const std::string cmd = argv[i++];
  char err[512] = {0};
  const std::string idx = home("/.cache/sgclip");
  const char* sample = std::getenv("SGC_IMG_SAMPLE");
  const std::string def_sample = home("/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16");
  sgc_labels* L = sgc_labels_open_ex(labels.c_str(), idx.c_str(), sample ? sample : def_sample.c_str(), err, sizeof(err));
  if (!L) { std::fprintf(stderr, "labels: %s\n", err); return 1; }
  sgc_text* T = nullptr;
  if (text) {
    sgc_text_config tc;
    sgc_text_default_config(&tc, nullptr);
    T = sgc_text_create(&tc, err, sizeof(err));
    if (!T) std::fprintf(stderr, "text encoder off: %s\n", err);
  }
  sgc_encoder* E = nullptr;
  if (enc) {
    sgc_config ec;
    sgc_default_config(&ec);
    const std::string eng = std::getenv("SGC_ENGINE") ? std::getenv("SGC_ENGINE") : home("/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan");
    ec.engine = eng.c_str();
    ec.margin = 0.f;   // best view 사진은 이미 잘라 둔 것
    E = sgc_create(&ec, err, sizeof(err));
    if (!E) { std::fprintf(stderr, "image encoder: %s\n", err); return 1; }
  }
  sgs_config c;
  sgs_default_config(&c);
  c.mem_dir = mem.c_str();
  c.cache_dir = cache.empty() ? nullptr : cache.c_str();
  c.labels = L;
  c.text = T;
  c.encoder = E;
  sgs_index* X = sgs_open(&c, err, sizeof(err));
  if (!X) { std::fprintf(stderr, "open: %s\n", err); return 1; }
  std::vector<char> buf(1 << 22);
  int rc = 0;
  if (cmd == "search") {
    for (; i < argc; ++i) {
      const int n = sgs_search_json(X, argv[i], k, force, buf.data(), int(buf.size()));
      std::printf("%s\n", n >= 0 ? buf.data() : "{\"status\":\"error\"}");
    }
  } else if (cmd == "bench" && i < argc) {
    const int reps = std::atoi(argv[i++]);
    for (; i < argc; ++i) {
      std::vector<double> us;
      for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        sgs_search_json(X, argv[i], k, force, buf.data(), int(buf.size()));
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(us.begin(), us.end());
      std::printf("{\"query\":\"%s\",\"objects\":%d,\"reps\":%d,\"us_p50\":%.1f,\"us_p95\":%.1f}\n", argv[i], sgs_count(X), reps, us[us.size() / 2],
                  us[size_t(double(us.size()) * 0.95)]);
    }
  } else if (cmd == "confirm" && i + 2 < argc) {
    sgs_confirm(X, uint32_t(std::strtoul(argv[i][0] == 'O' ? argv[i] + 1 : argv[i], nullptr, 10)), argv[i + 1], argv[i + 2], i + 3 < argc ? argv[i + 3] : "",
                buf.data(), int(buf.size()));
    std::printf("%s\n", buf.data());
  } else if (cmd == "object" && i < argc) {
    sgs_object_json(X, uint32_t(std::strtoul(argv[i][0] == 'O' ? argv[i] + 1 : argv[i], nullptr, 10)), buf.data(), int(buf.size()));
    std::printf("%s\n", buf.data());
  } else if (cmd == "stats") {
    sgs_stats_json(X, buf.data(), int(buf.size()));
    std::printf("%s\n", buf.data());
  } else {
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    rc = 2;
  }
  sgs_close(X);
  if (E) sgc_destroy(E);
  if (T) sgc_text_destroy(T);
  sgc_labels_close(L);
  return rc;
}
