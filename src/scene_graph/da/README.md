# da — 데이터 연관: 중복 물체 합치기

objmap 이 프레임마다 만든 물체 중 **같은 물체가 둘 이상으로 확정된 것**을 사후에 하나로 합친다.
objmap 의 keyframe 연결(같은 이름·가까우면 1:1)이 놓친 중복을 잡는 용도다. 중복이 생기는 경우:

- 한 프레임에 일부만 보인 물체가 새 후보가 되었다가 확정됨(소파 한쪽 끝, 탁자 다리만).
- 한 물체의 마스크가 한 프레임에서 둘로 쪼개짐(1:1 이라 하나만 붙고 나머지는 새 후보).
- 옮겨진 뒤 옛 자리 물체와 새 자리 물체가 겹쳐 남음.

따로 빌드하지 않는다. `scenemap` 정적 라이브러리에 같이 들어가고(`../scenemap/CMakeLists.txt`), objmap 갱신 끝(`ObjectMap::update` 6단계)에서 부른다.
설계는 [docs/scenemap_설계.md](../../../docs/scenemap_설계.md) 3.2절(objmap).

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/da/merge.hpp` | `MergeParams`(켜기, `overlap_min` 0.35, `min_ext` 0.05 m), `mergeDuplicates`, `absorbObject`(한 쌍 합치기 — objmap 조각 합치기가 씀), `boxOverlap` |
| `src/merge.cpp` | 겹침 계산, 합칠 수 있는지 판단, 합치기(`absorb`), 반복 |
| `tests/test_merge.cpp` | 시험(아래) |

## 규칙

- **대상**: 둘 다 확정, 같은 이름 번호, 들고 있지 않음(`held_by < 0`), 사라짐(`SM_GONE`) 아님. 후보(미확정)는 건드리지 않는다.
  이름 모으기(`ObjParams::name_vote`, 기본 켬)면 이름이 달라도 두 상자의 3D IoU 가 `name_merge_iou`(0.5) 이상이면 대상
  (한 물체가 프레임마다 다른 이름으로 불려 따로 생긴 것 — 탁자 위 컵처럼 한 상자가 다른 상자 안이면 IoU 가 작아 안 합침).
- **같은 것**: 두 상자의 축별 겹침 비율(겹친 길이 / 둘 중 짧은 변)을 세 축 곱한 값이 `overlap_min` 이상.
  변이 `min_ext` 보다 얇으면 중심 둘레로 `min_ext` 까지 부풀려 계산한다. 러그·액자처럼 한 변이 거의 0 인 물체도 합칠 수 있다.
- **남는 쪽**: 관측 수(`n_obs`)가 많은 쪽. 같으면 id 가 작은 쪽.
- **합치기**:
  - 큰 가구(고정 종류이거나 xy 한 변이 `ObjParams::big` 초과): 상자 합집합, 위치 = 상자 중심.
  - 작은 물체: 위치·크기·상자를 `n_obs` 가중 평균.
  - 이름 표(`votes`)를 합치고 이름 = 표의 최댓값(이름 모으기를 켰을 때). 보이는데 놓친 수(`n_vis_miss`)는 합, `max_det_z` 는 큰 값.
  - `n_obs` 합, `first_seen`·`first_pos` 는 먼저 본 쪽, `last_seen`·`last_kf`·`score` 는 큰 값, `misses` 는 작은 값, `moved` 는 OR.
    한쪽이 `SM_SEEN` 이면 `SM_SEEN`. 남는 쪽에 받침(parent)이 없으면 지운 쪽 것을 받는다.
  - 점 구름: 지운 쪽 점을 남는 쪽 구름에 넣는다(같은 복셀이면 새 점이 이김, 복셀·한도는 `ObjParams`).
- **반복**: 가장 많이 겹치는 쌍 하나를 합치고 다시 센다(합치면 상자가 바뀜). 합칠 때마다 물체가 하나 줄어 반드시 끝난다.

objmap 쪽 연결(`src/objmap.cpp`): 합친 뒤 이번 프레임의 검출 연결·점 기록에서 지운 id 를 남는 id 로 바꾸고 사건(7)을 남긴다.
설정은 `ObjParams::merge`·`merge_overlap`·`merge_min_ext`. 환경 변수 `SM_NO_MERGE`(있으면 끔, A/B 비교용), `SM_MERGE_LOG`(있으면 합칠 때마다 stderr 에 한 줄).

## 시험 (ctest `da_merge`)

scenemap 빌드에 같이 들어간다.

```bash
cmake -S src/scene_graph/scenemap -B build/scenemap && cmake --build build/scenemap -j
ctest --test-dir build/scenemap -R da_merge
```

| 경우 | 확인 |
|---|---|
| 크게 겹친 같은 탁자 둘 | 하나로, 관측 많은 id 가 남음, `n_obs` 합, `first_seen` 은 먼저 본 쪽 |
| 맞닿기만 한 의자 둘 | 그대로 |
| 상자가 같고 이름이 다름 | 이름 모으기 끄면 그대로, 켜면(기본) 합침 + 이름 표 합침 |
| 이름이 다르고 한 상자가 다른 상자 안(탁자 위 컵) | 그대로 |
| 납작한 러그 둘(두께 ≈ 0) | 합침 |
| 들고 있음 / 사라짐 / 후보 | 합치지 않음 |
| 셋이 이어서 겹침 | 연쇄로 합쳐지고 큰 가구 상자는 합집합 |
| `enable = false` | 아무것도 안 합침 |
| 점 구름 | 합친 뒤 점 = 두 구름의 합집합(같은 복셀은 하나) |
