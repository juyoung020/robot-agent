// sgrt 의 물체 영상 임베딩(SigLIP 2, src/scene_graph/clip) 연결. docs/clip_candidates.md 3.5·6.
//
//   keyframe: ovdet 검출 → scenemap 갱신 뒤, sm_last_assoc(검출 → 물체) · sm_last_views(best view 가 바뀐 검출) 를 보고
//             (1) 임베딩이 아직 없는 물체(새 물체) (2) best view 가 바뀌고 품질이 임베딩 때보다 1.2 배 이상인 물체 순으로
//             최대 8 개를 sgclip 에 넣는다(원본 RGB 에서 자름, 비동기). 결과는 다음 스텝들에서 poll 로 받는다.
//   저장    : sgrt_save(저장 스레드) 가 sm_save_dsg_ex 앞에서 부름 — 바뀐 물체만 objects/O<id>_emb.f16, 라벨 표 이름 캐시
//             (cache/names.json, 표 sha·emb_sha 가 바뀐 것만 다시), scene.json 노드 metadata 는 sm_set_object_meta 로(emb·names).
//   켜기    : 환경 변수 SGRT_CLIP = 엔진 plan 경로(또는 1 = 기본 경로). 없으면 꺼짐(다른 실행에 영향 없음).
//             SGRT_LABELS = 라벨 표 폴더(기본 ~/embed_work/labels/objects-v1), SGC_IMG_SAMPLE = 투영 맞출 영상 표본(기본 있으면).
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "scenemap.h"
#include "sgclip.h"

namespace sgrt_clip {

struct Obj {
  std::vector<float> emb;       // 768, L2
  std::vector<uint16_t> f16;
  std::string sha;
  float q = 0;                  // 그 임베딩을 만든 모습 품질
  double stamp = 0;
  uint32_t ver = 0;             // 바뀔 때마다 +1
  uint32_t saved_ver = 0;       // 파일로 쓴 판
  uint32_t meta_ver = 0;        // 노드 메타로 넘긴 판(+ 이름 캐시 판)
  bool pending = false;
};

struct Stats {
  int32_t n_submitted = 0, n_done = 0, n_dropped = 0, n_objects = 0, last_batch = 0, n_named = 0;
  float crop_ms = 0, net_ms = 0, submit_us = 0, names_us = 0, save_ms = 0;
};

class ClipMem {
 public:
  ~ClipMem();
  bool init(const std::string& out_dir);   // false = 꺼짐(환경 변수 없음·엔진 실패)
  bool on() const { return enc_ != nullptr; }
  void reset();
  void poll();                             // 매 스텝(가벼움: 이벤트 질의)
  void keyframe(double stamp, const uint8_t* rgb, int on_device, int64_t row_stride, int pix_stride, int w, int h,
                const sm_detections* d, sm_ctx* sm);
  void save(const std::string& dir, sm_ctx* sm);   // 저장 스레드에서
  // ABI
  int embedding(uint32_t id, float* out);
  int query(const float* q, int k, uint32_t* ids, float* scores, sm_ctx* sm);
  int queryText(const char* text, int k, uint32_t* ids, float* scores, sm_ctx* sm);
  int names(uint32_t id, std::vector<std::pair<std::string, std::string>>* en_ko, std::vector<float>* sc, std::string* level,
            std::string* level_ko, int* structural);
  Stats stats() const;
  const sgc_labels* labels() const { return labels_ready_ ? labels_ : nullptr; }

 private:
  sgc_encoder* enc_ = nullptr;
  sgc_labels* labels_ = nullptr;
  std::atomic<bool> labels_ready_{false};
  std::thread loader_;
  mutable std::mutex mu_;
  std::unordered_map<uint32_t, Obj> obj_;
  std::vector<sgc_result> res_;
  std::vector<sgc_item> items_;
  std::vector<uint32_t> assoc_;
  std::vector<uint8_t> upd_;
  std::vector<float> qual_;
  std::string saved_dir_;
  void* cache_ = nullptr;                  // sgclip::NameCache
  std::string cache_sha_;                  // 이름 캐시를 만든 표 sha
  Stats st_;
  void take(const sgc_result& r);
};

}  // namespace sgrt_clip
