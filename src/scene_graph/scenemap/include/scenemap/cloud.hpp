// 물체별 모양(점 구름): objmap 이 물체에 붙인 관측의 마스크 안 깊이 점(MAD 띠·손·몸 거른 뒤)을 map 에 올려 복셀로 줄여 쌓는다.
//
//   복셀   : voxel(기본 0.02 m) 칸마다 점 하나 — 새 관측이 같은 칸에 오면 그 점(자리·색)을 새것으로 바꾼다.
//   한도   : cap(기본 4000)을 넘으면 오래 안 고쳐진 점부터 버려 cap 의 90 % 로(매 keyframe 다시 만들지 않게 여유).
//   좌표   : 점은 원점 org 기준(float)으로 두고, 물체가 움직이면(들기·받침 따라가기) org 만 옮긴다(O(1)) — 회전은 안 함.
//            map 좌표 = org + 점.
//   공유   : 점 배열은 바뀌지 않는 객체(shared_ptr<const CloudData>)를 바꿀 때 새로 만든다(쓸 때 복사) — 스냅숏·저장은
//            포인터만 복사한다. version 은 점이나 org 가 바뀔 때마다 +1(저장 더러움).
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace scenemap {

// uint64 → uint32 열린 주소 해시(복셀 번호 → 점 번호). 지우기 없음(다시 만듦), 비우기는 세대 번호로 O(1)
class VoxelIndex {
 public:
  void reserve(size_t n);
  void clear();                   // 비움(자리는 그대로 — 관측마다 다시 쓰는 것이 싸게)
  // 없으면 넣고 true, 있으면 *val 에 그 값을 주고 false
  bool insert(uint64_t key, uint32_t val, uint32_t* old);
  size_t size() const { return n_; }

 private:
  void grow();
  std::vector<uint64_t> keys_;
  std::vector<uint32_t> vals_;
  std::vector<uint32_t> gen_;     // 칸이 이 세대(cur_)면 차 있음 — clear 는 세대만 올림(O(1))
  uint32_t cur_ = 1;
  size_t n_ = 0;
};

uint64_t voxelKey(float x, float y, float z, float inv_voxel);

struct CloudPt {
  float x, y, z;                  // org 기준
  uint8_t r, g, b, a;             // a 는 채움(0)
  uint32_t seq;                   // 마지막으로 고친 순번(한도 넘으면 작은 것부터 버림)
};

struct CloudData {
  std::vector<CloudPt> pts;
  VoxelIndex index;
  uint32_t seq = 0;
};

struct ObjCloud {
  std::shared_ptr<const CloudData> data;   // 없으면 빈 것
  double org[3] = {0, 0, 0};
  bool has_org = false;
  uint32_t version = 0;
  double stamp = 0;                        // 마지막으로 바뀐 시뮬 시각

  size_t size() const { return data ? data->pts.size() : 0; }
  void clear(double t);
  void translate(const double d[3], double t);
  // map 좌표 점 n 개(xyz 3n, rgb 3n — NULL 이면 회색)를 넣음
  void add(const float* xyz, const uint8_t* rgb, int n, double voxel, int cap, double t);
};

}  // namespace scenemap
