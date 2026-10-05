// objprob 앞단(검출 → 이름 낱말 → scenemap 확률 모드 설정) — tools/realbag/realbag_run.cpp 와 libsgrt(src/sgrt.cpp)가 같이 쓴다.
//   낱말 표(kVocab·kVocabAp): SigLIP 2 글 → 지도 이름, 라벨 크기·상위어(kApLabels), 기본 분할 엔진 이름,
//   엔진별 매개변수 파일(objprob_params/<엔진>.json) 읽기, 글 표 만들기(sgc_labels), sm_set_text_model·sm_set_label_stats·sm_set_obj_params.
// 헤더만(scenemap.h·sgclip.h C ABI 만 씀).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "scenemap.h"
#include "sgclip.h"

namespace objprob_front {

// 기본 분할 엔진(models/ovdet/x86_sm120/ 아래): ObjectSAM = YOLO26n 학생(things 만, 이름 없는 'object').
// objprob 매개변수는 objprob_params/<이 이름에서 .plan 뺀 것>.json. 다른 엔진은 --engine / SGRT_ENGINE 으로
// (선생 FastSAM-s-416.plan 도 매개변수 파일 있음)
constexpr const char* kDefaultEngine = "yolo26n-seg-obj-416.plan";

// 검출 낱말: 글 → 지도 이름. 구조물(wall·floor …, person)은 scenemap 기본 표대로 노드가 안 된다. SigLIP 2 라벨 표에 있는 글만 쓴다.
struct Word { const char* text; const char* label; };
inline const Word kVocab[] = {
    // 구조물·사람(노드 안 됨)
    {"wall", "wall"}, {"floor", "floor"}, {"ceiling", "ceiling"}, {"door", "door"}, {"window", "window"}, {"pillar", "pillar"},
    {"column", "pillar"}, {"partition", "partition"}, {"glass wall", "wall"}, {"baseboard", "baseboard"}, {"railing", "railing"},
    {"staircase", "staircase"}, {"person", "person"}, {"man", "person"}, {"woman", "person"},
    // 고정 가구·가전(movable 0)
    {"desk", "desk"}, {"table", "table"}, {"coffee table", "table"}, {"carrel", "desk"}, {"cubicle", "partition"},
    {"cabinet", "cabinet"}, {"filing cabinet", "cabinet"}, {"cabinet base", "cabinet"}, {"shelf", "shelf"}, {"bookcase", "bookcase"},
    {"sofa", "sofa"}, {"couch", "sofa"}, {"bed", "bed"}, {"refrigerator", "refrigerator"}, {"electric refrigerator", "refrigerator"},
    {"microwave", "microwave"}, {"sink", "sink"}, {"toilet", "toilet"}, {"television", "tv"}, {"television receiver", "tv"},
    {"lamp", "lamp"}, {"table lamp", "lamp"}, {"floor lamp", "lamp"}, {"light fixture", "fixture"}, {"ceiling light", "fixture"},
    {"radiator", "radiator"}, {"curtain", "curtain"}, {"window blind", "curtain"}, {"air conditioner", "appliance"},
    {"water dispenser", "appliance"}, {"water cooler", "appliance"}, {"printer", "appliance"}, {"facsimile", "appliance"},
    {"scanner", "appliance"}, {"whiteboard", "whiteboard"}, {"picture frame", "picture frame"}, {"poster", "picture frame"},
    {"plant", "plant"}, {"potted plant", "plant"}, {"pot plant", "plant"}, {"rug", "rug"},
    // 옮길 수 있는 것
    {"chair", "chair"}, {"office chair", "chair"}, {"swivel chair", "chair"}, {"armchair", "chair"}, {"stool", "stool"},
    {"footstool", "stool"}, {"monitor", "monitor"}, {"computer monitor", "monitor"}, {"computer", "computer"},
    {"desktop computer", "computer"}, {"computer case", "computer"}, {"laptop", "laptop"}, {"keyboard", "keyboard"},
    {"mouse", "mouse"}, {"computer mouse", "mouse"}, {"box", "box"}, {"cardboard box", "box"}, {"packing box", "box"},
    {"storage box", "box"}, {"container", "box"}, {"storage container", "box"}, {"bottle", "bottle"}, {"water bottle", "bottle"},
    {"water jug", "bottle"}, {"cup", "cup"}, {"mug", "cup"}, {"coffee cup", "cup"}, {"trash can", "trash can"},
    {"wastebasket", "trash can"}, {"ashcan", "trash can"}, {"recycling bin", "trash can"}, {"bucket", "bucket"},
    {"book", "book"}, {"notebook", "book"}, {"folder", "book"}, {"paper", "paper"}, {"backpack", "bag"}, {"bag", "bag"},
    {"handbag", "bag"}, {"briefcase", "bag"}, {"satchel", "bag"}, {"sack", "bag"}, {"basket", "basket"}, {"phone", "phone"},
    {"tripod", "tripod"}, {"camera tripod", "tripod"}, {"stand", "tripod"}, {"fan", "fan"}, {"clock", "clock"},
    {"jacket", "clothing"}, {"coat", "clothing"}, {"umbrella", "umbrella"}, {"speaker", "speaker"}, {"electric kettle", "kettle"},
    {"kettle", "kettle"}, {"coffee maker", "appliance"}, {"pillow", "pillow"}, {"shoe", "shoe"}, {"vase", "vase"},
    {"toolbox", "box"}, {"cable", "cable"}, {"router", "modem"}, {"modem", "modem"}, {"pole", "pole"},
};

// objprob 라벨 통계: 지도 이름 → 가장 긴 변(m)의 대표값·log 표준편차(크기 우도, 흔한 가구·물건 치수에서 손으로 — 이 장면 정답에서 맞추지 않음),
// 상위어(이름 사후가 낮을 때 올라갈 곳). 상위어 라벨(furniture …)과 "object" 는 낱말 줄이 없어 직접 이름이 되지 않는다
struct ApLabel { const char* label; float size, sd; const char* parent; };
inline const ApLabel kApLabels[] = {
    {"structure", 0, 0, nullptr}, {"outdoors", 0, 0, "structure"}, {"furniture", 0, 0, nullptr}, {"electronics", 0, 0, nullptr}, {"container", 0, 0, nullptr},
    {"decoration", 0, 0, nullptr}, {"object", 0, 0, nullptr},
    {"wall", 3.0f, 0.8f, "structure"}, {"floor", 3.0f, 0.8f, "structure"}, {"ceiling", 3.0f, 0.8f, "structure"}, {"door", 2.0f, 0.3f, "structure"},
    {"window", 1.2f, 0.5f, "structure"}, {"pillar", 2.4f, 0.4f, "structure"}, {"partition", 1.5f, 0.5f, "structure"},
    {"staircase", 2.5f, 0.5f, "structure"}, {"railing", 1.5f, 0.6f, "structure"}, {"baseboard", 1.5f, 0.8f, "structure"}, {"person", 1.7f, 0.25f, nullptr},
    {"desk", 1.4f, 0.3f, "furniture"}, {"table", 1.2f, 0.4f, "furniture"}, {"cabinet", 1.0f, 0.5f, "furniture"}, {"shelf", 1.0f, 0.5f, "furniture"},
    {"bookcase", 1.6f, 0.4f, "furniture"}, {"sofa", 2.0f, 0.3f, "furniture"}, {"bed", 2.0f, 0.25f, "furniture"}, {"chair", 0.9f, 0.25f, "furniture"},
    {"stool", 0.5f, 0.3f, "furniture"}, {"refrigerator", 1.8f, 0.2f, "appliance"}, {"microwave", 0.5f, 0.3f, "appliance"}, {"sink", 0.6f, 0.4f, nullptr},
    {"toilet", 0.7f, 0.2f, nullptr}, {"tv", 1.0f, 0.4f, "electronics"}, {"lamp", 1.0f, 0.6f, "decoration"}, {"fixture", 0.4f, 0.7f, nullptr},
    {"radiator", 0.9f, 0.4f, nullptr}, {"curtain", 1.8f, 0.5f, "decoration"}, {"appliance", 0.8f, 0.7f, nullptr}, {"whiteboard", 1.5f, 0.4f, "decoration"},
    {"picture frame", 0.6f, 0.6f, "decoration"}, {"plant", 0.8f, 0.6f, "decoration"}, {"rug", 2.0f, 0.4f, "decoration"},
    {"monitor", 0.55f, 0.3f, "electronics"}, {"computer", 0.45f, 0.4f, "electronics"}, {"laptop", 0.35f, 0.2f, "electronics"},
    {"keyboard", 0.45f, 0.2f, "electronics"}, {"mouse", 0.11f, 0.3f, "electronics"}, {"phone", 0.15f, 0.3f, "electronics"},
    {"speaker", 0.3f, 0.5f, "electronics"}, {"modem", 0.25f, 0.3f, "electronics"}, {"box", 0.4f, 0.6f, "container"}, {"bottle", 0.25f, 0.3f, "container"},
    {"cup", 0.1f, 0.3f, "container"}, {"trash can", 0.4f, 0.3f, "container"}, {"bucket", 0.35f, 0.3f, "container"}, {"bag", 0.45f, 0.4f, "container"},
    {"basket", 0.4f, 0.4f, "container"}, {"book", 0.25f, 0.3f, nullptr}, {"paper", 0.3f, 0.4f, nullptr}, {"tripod", 1.2f, 0.4f, nullptr},
    {"fan", 0.6f, 0.5f, "appliance"}, {"clock", 0.3f, 0.4f, "decoration"}, {"clothing", 0.8f, 0.4f, nullptr}, {"umbrella", 0.9f, 0.4f, nullptr},
    {"kettle", 0.25f, 0.2f, "appliance"}, {"pillow", 0.5f, 0.3f, "decoration"}, {"shoe", 0.28f, 0.2f, nullptr}, {"vase", 0.3f, 0.4f, "decoration"},
    {"cable", 1.0f, 1.0f, nullptr}, {"pole", 1.8f, 0.5f, nullptr},
};

// 배경 낱말(구조물 쪽): 분할 조각에 계단 디딤판·문틀·창 밖 덤불이 많아 그 말이 없으면 book·curtain·plant 로 불린다.
// "outdoors" = 창·유리문 너머 바깥(구조물 종류)
inline const Word kVocabAp[] = {
    {"stair", "staircase"}, {"stairway", "staircase"}, {"steps", "staircase"}, {"handrail", "railing"}, {"doorcase", "door"},
    {"window frame", "window"}, {"windowpane", "window"}, {"skirting board", "baseboard"}, {"bush", "outdoors"}, {"shrub", "outdoors"},
    {"hedgerow", "outdoors"}, {"lawn", "outdoors"},
};

// SigLIP 2 B/32 로짓 척도·치우침(글 모델)
constexpr float kLogitScale = 111.83257f, kLogitBias = -16.766876f;

inline std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

// 라벨 표: 이름 → 번호(없으면 끝에 더함)
inline int labelId(std::vector<std::string>* labels, const std::string& l) {
  for (size_t k = 0; k < labels->size(); ++k) if ((*labels)[k] == l) return int(k);
  labels->push_back(l);
  return int(labels->size()) - 1;
}

// 엔진 경로(또는 이름) → 매개변수 파일 이름 줄기("…/yolo26n-seg-obj-416.plan" → "yolo26n-seg-obj-416")
inline std::string engineStem(const std::string& engine) {
  std::string st = engine.empty() ? std::string(kDefaultEngine) : engine;
  st = st.substr(st.find_last_of('/') + 1);
  if (st.size() > 5 && st.compare(st.size() - 5, 5, ".plan") == 0) st.resize(st.size() - 5);
  return st;
}

// 엔진별 매개변수 파일 {"obj_params": "key=val,…", "label_prior": "옆 파일.json"} 읽기. label_prior 는 비어 있을 때만 채움(상대 경로 = 파일 옆)
inline bool readParamsFile(const std::string& pf, std::string* obj_kv, std::string* label_prior) {
  const std::string js = slurp(pf);
  if (js.empty()) return false;
  auto str = [&](const char* key) {
    const size_t k = js.find(std::string("\"") + key + "\":");
    if (k == std::string::npos) return std::string();
    const size_t q0 = js.find('"', k + std::strlen(key) + 3), q1 = q0 == std::string::npos ? q0 : js.find('"', q0 + 1);
    return q1 == std::string::npos ? std::string() : js.substr(q0 + 1, q1 - q0 - 1);
  };
  *obj_kv = str("obj_params");
  const std::string lp = str("label_prior");
  if (label_prior->empty() && !lp.empty()) *label_prior = lp[0] == '/' ? lp : pf.substr(0, pf.find_last_of('/') + 1) + lp;
  return true;
}

// 글 표: 낱말(kVocab [+ kVocabAp] + extra)마다 SigLIP 2 글 임베딩(L2) 한 줄 → 라벨 번호. extra = {글, 라벨}(과제 물체 이름 등)
inline void buildTextTable(const sgc_labels* lt, bool with_ap, const std::vector<std::pair<std::string, std::string>>& extra,
                           std::vector<std::string>* labels, std::vector<float>* text, std::vector<int>* text_cls, std::string* missing) {
  std::vector<std::pair<std::string, std::string>> words;
  for (const Word& w : kVocab) words.emplace_back(w.text, w.label);
  if (with_ap) for (const Word& w : kVocabAp) words.emplace_back(w.text, w.label);
  words.insert(words.end(), extra.begin(), extra.end());
  std::vector<float> e(SGC_DIM);
  for (const auto& [txt, lab] : words) {
    const int row = sgc_labels_find(lt, txt.c_str());
    if (row < 0 || sgc_labels_text_emb(lt, row, e.data()) != 0) { if (missing) *missing += " '" + txt + "'"; continue; }
    double n = 0;
    for (float x : e) n += double(x) * x;
    n = std::sqrt(std::max(n, 1e-20));
    for (float& x : e) x = float(x / n);
    text_cls->push_back(labelId(labels, lab));
    text->insert(text->end(), e.begin(), e.end());
  }
}

// 확률 모드 켜기: 상위어 라벨은 미리 표에 더해 둘 것(addApLabels). sm_set_labels 를 이미 부른 뒤.
// 글 모델·라벨 크기 사전·상위어·(있으면) 라벨 log 사전 → sm_set_object_model(1) → obj_kv. 실패면 err 에 까닭, false
inline void addApLabels(std::vector<std::string>* labels) { for (const ApLabel& a : kApLabels) labelId(labels, a.label); }
inline bool enable(sm_ctx* c, std::vector<std::string>* labels, const std::vector<float>& text, const std::vector<int>& text_cls,
                   const std::string& label_prior, const std::string& obj_kv, std::string* err) {
  sm_set_text_model(c, text.data(), text_cls.data(), int32_t(text_cls.size()), SGC_DIM, kLogitScale, kLogitBias);
  const int nl = int(labels->size());
  std::vector<float> mu(size_t(nl), 0.f), sd(size_t(nl), 0.f);
  std::vector<int32_t> par(size_t(nl), -1);
  for (const ApLabel& a : kApLabels) {
    const int l = labelId(labels, a.label);
    if (a.size > 0) { mu[size_t(l)] = std::log(a.size); sd[size_t(l)] = a.sd; }
    if (a.parent) par[size_t(l)] = labelId(labels, a.parent);
  }
  if (int(labels->size()) != nl) { *err = "objprob: label table grew"; return false; }
  std::vector<float> lpri(size_t(nl), -std::log(float(nl)));   // 표에 없는 라벨은 고른 사전 1/n
  if (!label_prior.empty()) {
    const std::string js = slurp(label_prior);
    if (js.empty()) { *err = "cannot read " + label_prior; return false; }
    for (int l = 0; l < nl; ++l) {   // 상위어 라벨은 낱말 줄이 없어 직접 이름이 안 됨(사전 값은 상관없음)
      const std::string& nm = (*labels)[size_t(l)];
      const size_t k = js.find("\"" + nm + "\":");
      if (k != std::string::npos) lpri[size_t(l)] = float(std::atof(js.c_str() + k + nm.size() + 3));
    }
  }
  sm_set_label_stats(c, label_prior.empty() ? nullptr : lpri.data(), mu.data(), sd.data(), par.data(), nl, labelId(labels, "object"));
  sm_set_object_model(c, 1);
  if (!obj_kv.empty() && sm_set_obj_params(c, obj_kv.c_str()) != 0) { *err = "objprob: unknown key in obj_params"; return false; }
  return true;
}

}  // namespace objprob_front
