# 사람 시연 → scenemap 재생 → RecallVLA 자료

다른 로봇의 사람 원격조종 시연을 우리 실시간 지도기(scenemap, C++)로 다시 돌려 지도 토큰을 만들고, 행동을 공통 행동 표현으로 바꿔 RecallVLA 학습 자료로 쓴다. 데이터 방침은 [MAPVLA_SPEC.md](../../docs/map_vla/MAPVLA_SPEC.md) 4.3c(사람 시연이 주 자료, RL 은 선택 단계).
Python 은 데이터셋이 Python 도구만 주는 오프라인 형식 바꾸기에만 쓴다(LeRobot parquet·HEVC 영상·HDF5·OmniGibson). 재생·토큰·읽기는 C++.

## 파일

| 파일 | 하는 것 |
|---|---|
| `b1k/screen.py` | BEHAVIOR 2026 시연 20,000 판을 **메타데이터만으로** 거름: 주석 스킬 구간마다 물체·받침을 과제 인스턴스 장면 JSON 에서 찾아 E0 한도(CURRICULUM_BEHAVIOR2026 3.1·5.3)로 `action_full / action_base_only / highlevel_only / drop` 태그 + 이유. `segments.csv`·`episodes.csv`·`tasks.csv`·`summary.json` |
| `b1k/fetch_sizes.py` | HF 파일 크기 목록만(원본 HDF5 판별·LeRobot 영상) → `sizes.json`(받기 크기 추정) |
| `b1k/demo2rec.py` | 시연 한 판(LeRobot 머리 RGB·깊이·proprio 61 + 원본 HDF5 정답 자세) → SGRC `rec.bin`(sgrt 기록 형식) + `labels.txt` + `demo.json`. 검출 = 정답 물체 상자 라벨(`map_src: gt_box`) |
| `b1k/og_probe.py` | OmniGibson 재생 시험(`flock /tmp/claude-1000/og.lock`, `gm.HEADLESS`) — 모달리티를 인자로 |
| `tools/demo2sg.cpp` | `rec.bin` → scenemap(r1pro, slam|gt 자세) → 학습 뷰어 재생 판(`.sg`: sgview 스트림·memory·카메라 JPEG·바탕 층), 실행 폴더 group `human_demos`. `--ingest host:port` 로 sgview 에 실시간으로도. `DEMO_TRACE=1` 은 keyframe 마다 든 물체·구름 점 수 |

## 쓰는 법

```bash
cmake -S training/demos -B ~/ra_demos_build -DSCENEMAP_DIR=$HOME/robot-agent/src/behavior-2026/src/scene_graph/scenemap && cmake --build ~/ra_demos_build -j4
P=~/miniconda3/envs/behavior/bin/python
# 거르기(주석: git sparse clone — 파일별 HF 받기는 20,000 요청이라 429)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:limit=200k --no-checkout https://huggingface.co/datasets/behavior-1k/2026-challenge-demos ~/datasets/b1k_ann_git
(cd ~/datasets/b1k_ann_git && git sparse-checkout set annotations meta/tasks.jsonl && GIT_LFS_SKIP_SMUDGE=1 git checkout)
$P training/demos/b1k/fetch_sizes.py; $P training/demos/b1k/screen.py
# 한 판 재생 → 학습 뷰어(Replay 탭, group human_demos)
$P training/demos/b1k/demo2rec.py --episode 0 --raw ~/datasets/b1k_raw/task-0000/episode_00000010.hdf5 --out ~/datasets/human_demos/b1k/ep000000
~/ra_demos_build/demo2sg --demo ~/datasets/human_demos/b1k/ep000000 --run-out ~/trainview_work/behavior_og/human_demos/b1k_ep0_turning_on_radio --rasc ~/ra_b1k/house_double_floor_lower.rasc
```

## 잰 값 (2026-10-04)

- ep 0(turning_on_radio, 65.2 s, 1,956 프레임): `demo2rec` 101 s(상자 라벨 97 s — Python), keyframe 652(10 Hz, 320×240), 검출 5.5/keyframe, `rec.bin` 360 MB(중간 파일), `demo2sg` 2.7 s(scenemap 0.83 s), 판 12 MB, 물체 11.
- 거르기(20,000 판, 1,953 h, 66 s): action_full 28.5 h(1.5 %), action_base_only 749 h, highlevel_only 1,087 h, drop 91 h. action_full 이 있는 판 2,116(18 과제). 표는 MAPVLA_SPEC 4.3c.
- OmniGibson 재생: `gm.HEADLESS=True` 로 RGB·깊이는 됨. `seg_instance`·`seg_instance_id` 는 이 PC 에서 SyntheticData 후처리 segfault(모달리티 하나씩 빼서 확인) → 인스턴스 마스크는 아직 상자 라벨.
