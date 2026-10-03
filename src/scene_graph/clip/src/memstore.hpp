// 기억 폴더의 임베딩 원본과 이름 캐시(docs/clip_candidates.md 3.5).
//
//   memory/objects/O<id>_emb.f16   원본: 768 × FP16(리틀 엔디언), L2 정규화. emb_sha = 그 1536 바이트의 FNV-1a 64 비트(16 진 16 자)
//   memory/cache/names.json        파생: {"table": {name, version, sha}, "objects": {"O12": {emb_sha, en: [[이름, 점수] ×5],
//                                  ko: [[이름, 점수] ×5], level, level_ko(1위), general, general_ko(확신 낮을 때 상위어), score, prob, margin, rolled,
//                                  structural}}}
//   memory/cache/index/            파생: 라벨 표 색인(labels_<sha>.idx)
//   다시 뽑는 조건: 표 sha 가 바뀌었거나 물체의 emb_sha 가 바뀌었을 때만. 쓰기는 모두 임시 파일 → rename.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sgclip.h"

namespace sgclip {

std::string embSha(const uint16_t* f16, int n = SGC_DIM);
bool writeAtomic(const std::string& path, const void* data, size_t n);
bool writeEmb(const std::string& mem_dir, uint32_t id, const uint16_t* f16);       // objects/O<id>_emb.f16
bool readEmb(const std::string& path, std::vector<uint16_t>* f16);
std::string embRel(uint32_t id);                                                    // "objects/O<id>_emb.f16"

struct NameEntry {
  std::string emb_sha;
  std::vector<std::pair<std::string, float>> en, ko;
  std::string level, level_ko;      // 보여 줄 이름(1위, 주 표)
  std::string general, general_ko;  // 확신이 낮을 때 공통 상위어(없으면 "")
  float score = 0, prob = 0, margin = 0;
  bool rolled = false, structural = false;
};

struct ObjRef {
  uint32_t id;
  std::string emb_sha;
  const float* emb;     // SGC_DIM, L2
};

class NameCache {
 public:
  bool load(const std::string& mem_dir);                   // 없으면 빈 캐시(false)
  // 표 sha 가 다르면 전부, 아니면 emb_sha 가 바뀐/없는 물체만 다시 뽑음. objs 에 없는 항목은 지움. 다시 뽑은 수
  int refresh(const sgc_labels* L, const std::vector<ObjRef>& objs, const sgc_lookup_params* p = nullptr);
  bool save(const std::string& mem_dir) const;             // cache/names.json(원자적)
  const NameEntry* get(uint32_t id) const;
  // scene.json 노드 metadata 에 덧붙일 JSON 멤버("emb":{..},"names":{..}). 이름이 없으면 emb 만
  std::string nodeMeta(uint32_t id, const std::string& emb_sha, double stamp) const;
  std::string table_name, table_version, table_sha;
  std::map<uint32_t, NameEntry> obj;
};

}  // namespace sgclip
