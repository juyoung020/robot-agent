# 잔재 체크리스트

`python3 tools/audit.py` 가 만든다(폴더마다 다섯 검사). 0 이면 ✅. 허용 예외는 tools/audit.py ALLOW.

| 폴더 | 결정 키워드(코드) | 결정 키워드(문서) | 저장소 밖 경로 | 깨진 링크 | 안 부르는 파일 | 상태 |
|---|---|---|---|---|---|---|
| `(뿌리)` | · | · | · | · | · | ✅ |
| `config` | · | · | · | · | · | ✅ |
| `docs` | · | · | · | · | · | ✅ |
| `scripts` | · | · | · | · | · | ✅ |
| `src` | · | · | · | · | · | ✅ |
| `src/agent` | · | · | · | · | · | ✅ |
| `src/agent/devtools` | · | · | · | · | · | ✅ |
| `src/agent/planner` | · | · | · | · | · | ✅ |
| `src/agent/prompts` | · | · | · | · | · | ✅ |
| `src/agent/runtime` | · | · | · | · | · | ✅ |
| `src/agent/skills` | · | · | · | · | · | ✅ |
| `src/agent/tools` | · | · | · | · | · | ✅ |
| `src/app` | · | · | · | · | · | ✅ |
| `src/robot` | · | · | · | · | · | ✅ |
| `src/scene_graph` | · | · | · | · | · | ✅ |
| `src/scene_graph/clip` | · | · | · | · | · | ✅ |
| `src/scene_graph/common` | · | · | · | · | · | ✅ |
| `src/scene_graph/da` | · | · | · | · | · | ✅ |
| `src/scene_graph/ovdet` | · | · | · | · | · | ✅ |
| `src/scene_graph/runtime` | · | · | · | · | · | ✅ |
| `src/scene_graph/scenemap` | · | · | · | · | · | ✅ |
| `src/scene_graph/sgview` | · | · | · | · | · | ✅ |
| `src/scene_graph/slam_carto` | · | · | · | · | · | ✅ |
| `src/scene_graph/tools` | · | · | · | · | · | ✅ |
| `src/sim` | · | · | · | · | · | ✅ |
| `src/vla` | · | · | · | · | · | ✅ |
| `tests` | · | · | · | · | · | ✅ |
| `tools` | · | · | · | · | · | ✅ |
| `training/BC` | · | · | · | · | · | ✅ |
| `training/README.md` | · | · | · | · | · | ✅ |
| `training/RL` | · | · | · | · | · | ✅ |
| `training/RL/config` | · | · | · | · | · | ✅ |
| `training/RL/env` | · | · | · | · | · | ✅ |
| `training/RL/map` | 3 | 6 | · | · | · | ⬜ |
| `training/RL/map_calib` | · | · | · | · | · | ✅ |
| `training/RL/map_cmp` | · | 1 | · | · | · | ⬜ |
| `training/RL/network` | · | · | · | · | · | ✅ |
| `training/RL/observation` | · | · | · | · | · | ✅ |
| `training/RL/ppo` | · | · | · | · | · | ✅ |
| `training/RL/reward` | · | · | · | · | · | ✅ |
| `training/RL/tools` | · | · | · | · | · | ✅ |
| `training/data` | · | · | · | · | · | ✅ |
| `training/embed` | · | · | · | · | · | ✅ |
| `training/fastsam` | · | · | · | · | · | ✅ |
| `training/model` | · | · | · | · | · | ✅ |
| `training/render_engine` | · | · | · | · | · | ✅ |
| `training/viewer` | · | · | · | · | · | ✅ |
| `training/vla` | · | · | · | · | · | ✅ |

## 남은 항목

### `training/RL/map`
- [ ] 결정 키워드(코드): training/RL/map/include/map.h:22: // 자세 오차는 Cartographer 흉내(drift_params.h, GPU_MAP_PORT 0.3). 옛 slam2d 맞춤 값(odo_t 0.015·kf_corr_xy 등, map_drift
- [ ] 결정 키워드(코드): training/RL/map/include/percept.h:77: // keyframe 카메라 방향·회전 빠르기(움직임 근거). 자세 고침은 Cartographer 흉내(map.h phase_begin, 스텝마다) — slam2d 의 깊이 열 맞추기 되돌림(kf_
- [ ] 결정 키워드(코드): training/RL/map/tools/map_drift.cpp:8: // kind 5: 옛 LIMO SLAM(slam2d, 보관) 기록 넷(data/datasets/limo_rec r1–r3·r4live: 25.0–37.9 m, 1,788–2,663°, 103–14
- [ ] 결정 키워드(문서): training/RL/map/README.md:64: 1. **시작(판마다 한 스레드)**: 환경 판 번호가 바뀌면 지도 리셋(격자·물체 비움, 장면 다시 만듦). 아니면 참 증분에 Cartographer 걸음 오차(`drift_params.h`)를 
- [ ] 결정 키워드(문서): training/RL/map/README.md:67: - (slam2d 의 keyframe 맞추기 되돌림 `kf_corr`·`min_hits` 는 없앰 — Cartographer 흉내는 시작 단계에서 끝남, GPU_MAP_PORT 0.3)
- [ ] 결정 키워드(문서): training/RL/map/README.md:91: ## scenemap 바뀜 규칙·LIMO 잡기 확인·새 SLAM 옮김 (behavior-2026 3ed710f·d58c978·84b373c, 2026-10-04, 잰 값)
- [ ] 결정 키워드(문서): training/RL/map/README.md:102: | 중복 병합(da mergeDuplicates, 근사판은 예전에 병합이 없었음): 확정·안 든·사라짐 아닌 쌍, 같은 이름 또는 3D IoU ≥ 0.5, 축별 겹침 곱 ≥ 0.35, 큰 쌍은 합집
- [ ] 결정 키워드(문서): training/RL/map/README.md:485: scenemap 기본값에서 가져온 값: 격자 Q·l_hit·l_miss·l_min/max, scan 의 높이 띠, slam2d 의 mf_xy·mf_yaw·mf_kf·still_v/w, objmap 
- [ ] 결정 키워드(문서): training/RL/map/README.md:499: - ~~**5.2 진짜 scenemap 과 맞추기(V2, `map_verify --check` 의 진짜 판)**: 막혀 있다.~~ **(했다)**: scenemap 에 LIMO + OMX 순기구학이

### `training/RL/map_cmp`
- [ ] 결정 키워드(문서): training/RL/map_cmp/README.md:3: GPU 지도 근사판(`../map`)과 진짜 scenemap(behavior-2026 서브모듈 `src/scene_graph/scenemap`, LIMO + OMX, CPU)에 **같은 입력**을 

