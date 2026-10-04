// da — 데이터 연관(data association): 프레임마다 나온 물체 세그를 하나의 물체로 합친다.
//
// objmap 의 keyframe 연결(같은 이름이고 가까우면 1:1)이 놓친 중복을 확정 물체끼리 사후에 합친다. 중복이 생기는 이유:
//   · 한 프레임에 일부만 보인 물체가 새 후보가 되었다가 확정됨(소파 한쪽 끝, 탁자 다리만 …)
//   · 한 물체가 한 프레임에서 마스크 둘로 쪼개짐(1:1 이라 하나만 붙고 나머지는 새 후보)
//   · 옮겨진 뒤 옛 자리 물체와 새 자리 물체가 겹쳐 남음
//
// 규칙(mergeDuplicates)
//   대상   : 둘 다 확정, 같은 이름 번호(종류 kinds 는 이름 번호마다 하나라 따로 안 봄), 들고 있지 않음, 사라짐 아님
//   같은 것: 두 상자가 축마다 겹치는 비율(겹친 길이 / 둘 중 짧은 변, 변이 얇으면 min_ext 로 올림)의 곱 ≥ overlap_min.
//            러그·액자처럼 한 변이 거의 0 인 물체도 되도록 부피가 아니라 축별 비율로 본다
//   합침   : 관측이 많은 쪽 id 를 남김(같으면 먼저 본 쪽). n_obs 합, first_* 는 먼저 본 쪽, last_seen 최댓값,
//            작은 물체는 n_obs 가중 평균, 큰 가구(한 변 > big 또는 고정 종류)는 상자 합집합(합집합 한 변이 ObjParams::max_ext 넘으면 안 합침)
//            점 구름은 합친다(복셀마다 하나 — 새 점이 이김)
//   반복   : 가장 많이 겹치는 쌍부터 하나씩 합치고 상자가 바뀌므로 다시 센다 — 합칠 때마다 물체가 줄어 반드시 끝난다
#pragma once
#include <cstdint>
#include <vector>

#include "scenemap/objmap.hpp"

namespace scenemap::da {

struct MergeParams {
  bool enable = true;
  double overlap_min = 0.35;   // 축별 겹침 비율의 곱 하한
  double min_ext = 0.05;       // 변이 이보다 얇으면 이 값으로 올려서 비율을 계산(납작한 물체)
};

struct MergeResult {
  uint32_t keep = 0, drop = 0;
  double overlap = 0;
};

// objs 안에서 중복을 합친다(drop 쪽을 지움). 합친 쌍을 돌려준다. stamp = 점 구름·이벤트 시각, op = 점 구름 복셀·한도
std::vector<MergeResult> mergeDuplicates(std::vector<MapObject>& objs, const MergeParams& mp, const ObjParams& op, double stamp,
                                         const std::vector<uint8_t>* kinds = nullptr);

// 두 상자의 겹침 비율(0..1): 축마다 겹친 길이 / 둘 중 짧은 변 의 곱
double boxOverlap(const double alo[3], const double ahi[3], const double blo[3], const double bhi[3], double min_ext);

}  // namespace scenemap::da
