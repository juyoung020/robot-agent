// 라벨 표 안쪽 자료(labels.cpp 와 objindex.cpp 가 같이 씀 — 공개 C ABI 는 include/sgclip.h).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "sgclip.h"

namespace sgclip_detail {

struct Row {
  std::string en, ko, synset;
  std::vector<std::string> ko_all, en_syn;   // 한국어 이름 전부, 영어 동의어
  int syn = -1;                 // synset 번호(없으면 -1: 줄마다 고유)
  std::vector<int> chain;       // [자기 synset, 상위어 …] 번호
  bool main = false, structural = false;
};

}  // namespace sgclip_detail

struct sgc_labels {
  std::string dir, name, sha;
  int K = 0;
  std::vector<uint16_t> text;   // K × D FP16
  std::vector<sgclip_detail::Row> rows;
  std::vector<std::string> syn_names;
  std::unordered_map<std::string, int> syn_id;
  std::vector<int> syn_row;     // synset → 표시 줄(main 먼저)
  std::unordered_map<std::string, int> by_text;
  std::unordered_map<std::string, int> by_name; // 이름 검색: main 줄(영어·한국어 전부·동의어) 먼저
  std::unordered_map<std::string, std::vector<int>> names_main, names_tail;   // 글 → 그 글을 이름으로 가진 줄 전부
  std::vector<uint8_t> generic; // synset → 이름으로 쓰기엔 너무 넓은 상위어(artifact, instrumentality …)
  // 색인
  std::vector<float> mu, P;     // P: PD × D(행 = 출력 차원)
  std::vector<float> cent;      // NC × PD
  std::vector<int32_t> off, ids;
  std::vector<float> qc;        // 질의에서 뺄 평균(영상 표본 평균, 표본이 없으면 0)
  std::vector<uint16_t> lp;     // 목록 순서, 줄마다 128-d 투영(FP16) — 1단계 점수
  std::string proj = "text";    // 투영을 맞춘 것: "text"(라벨 글 PCA) 또는 "img:<표본 sha>"(영상 표본 PCA)
  std::vector<uint64_t> codes;  // 목록 순서, 줄마다 2 단어
  std::vector<uint8_t> is_main; // 목록 순서
  double build_ms = 0;
  bool from_cache = false;
};
