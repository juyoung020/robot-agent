// da — 물체 합치기 도구: objprob 의 apMergePass(scenemap objmap.cpp)가 같은 것으로 판정한 두 물체를 하나로 합친다.
//
// absorbObject: drop 을 keep 에 합친다. keep 은 호출자가 정함(관측이 많은, 같으면 먼저 본 쪽 id).
//   위치·상자: 작은 물체는 n_obs 가중 평균, 큰 가구(한 변 > big 또는 고정 종류)는 상자 합집합
//   n_obs 합, first_* 는 먼저 본 쪽, last_seen 최댓값, 이름 표 합, 점 구름 합침(복셀마다 하나 — 새 점이 이김)
// boxOverlap: 두 상자의 축별 겹침 비율의 곱(얇은 변은 min_ext 까지 부풀림) — objmap 움직임 판정이 씀
#pragma once
#include <cstdint>
#include <vector>

#include "scenemap/objmap.hpp"

namespace scenemap::da {

// drop 을 keep 에 합친다(mergeDuplicates 와 같은 규칙: 위치·상자·관측 수·이름 표·점 구름). drop 은 호출자가 지운다
void absorbObject(MapObject& keep, MapObject& drop, const ObjParams& op, double stamp, const std::vector<uint8_t>* kinds = nullptr);

// 두 상자의 겹침 비율(0..1): 축마다 겹친 길이 / 둘 중 짧은 변 의 곱
double boxOverlap(const double alo[3], const double ahi[3], const double blo[3], const double bhi[3], double min_ext);

}  // namespace scenemap::da
