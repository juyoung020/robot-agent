# da — 물체 합치기 도구

objprob 의 `apMergePass`(scenemap `src/objmap.cpp`, 다음 keyframe 앞)가 **같은 물체**(이름 없이 SigLIP 2 임베딩 vMF + 위치·접촉 로지스틱)로 판정한 두 물체를
하나로 합치는 `absorbObject` 와 상자 겹침 비율 `boxOverlap`. 같은 것 판정은 여기 없다(objprob 하나). 옛 이름 기반 `mergeDuplicates`·`MergeParams` 는 `archive/` 로 옮김.

따로 빌드하지 않는다. `scenemap` 정적 라이브러리에 같이 들어간다(`../scenemap/CMakeLists.txt`).

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/da/merge.hpp` | `absorbObject`(한 쌍 합치기 — `apMergePass` 가 씀), `boxOverlap`(축별 겹침 비율의 곱 — 움직임 판정이 씀) |
| `src/merge.cpp` | 합치기(`absorb`)·겹침 계산 |
| `tests/test_merge.cpp` | 시험(ctest `da_merge`) |

## 합치기

- **남는 쪽**은 호출자가 정한다(관측이 많은 쪽, 같으면 먼저 본 id).
- 큰 가구(고정 종류이거나 xy 한 변이 `ObjParams::big` 초과): 상자 합집합, 위치 = 상자 중심. 작은 물체: 위치·크기·상자를 `n_obs` 가중 평균.
- 이름 표(`votes`)를 합치고 이름 = 표의 최댓값. 보이는데 놓친 수(`n_vis_miss`)는 합, `max_det_z` 는 큰 값.
- `n_obs` 합, `first_seen`·`first_pos` 는 먼저 본 쪽, `last_seen`·`last_kf`·`score` 는 큰 값, `misses` 는 작은 값, `moved` 는 OR.
  한쪽이 `SM_SEEN` 이면 `SM_SEEN`. 남는 쪽에 받침(parent)이 없으면 지운 쪽 것을 받는다.
- 점 구름: 지운 쪽 점을 남는 쪽 구름에 넣는다(같은 복셀이면 새 점이 이김, 복셀·한도는 `ObjParams`).

## 시험 (ctest `da_merge`)

```bash
ctest --test-dir build/scenemap -R da_merge
```

작은 물체 가중 평균·이름 표, 큰 가구 상자 합집합, 점 구름 합침, `boxOverlap`(같은 상자·맞닿기만·납작한 상자).
