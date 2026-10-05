# sim

OmniGibson(BEHAVIOR-1K 장면) 안에서 리모 + OMX-F 를 굴리는 실행기. 인지는 실제 파이프라인 libsgrt(ObjectSAM + SigLIP 2 + objprob + Cartographer)를 그대로 쓴다.
경로는 `config/paths.env`(`B1K_ROOT`, `CONDA_SH`, `OG_CONDA_ENV`). OmniGibson 은 한 번에 하나(`OG_LOCK` flock).

| 폴더·파일 | 하는 일 |
|---|---|
| `explore/run_explore.sh <frontier\|llm> <과제> <tag>` | 탐사 한 판(에이전트 `run-skill` + libsgrt) → `data/outputs/explore_*` |
| `explore/viewer_8080.sh <run dir>` | 그 판을 sgview(8080)로 보기. 실시간은 `tools/run_explore_live.sh` |
| `explore/render_run.py <run dir> [out.png]` | 판 그림: 지도 위에 정답 바닥·지나온 길·접촉 자리 |
| `explore/gt_trav.py`, `explore/gt/` | 과제 장면의 정답 바닥·방·닿는 곳 지도(채점용) |
| `limo/run_limo_map.sh [과제] [스텝] [tag]` | 리모를 짧게 움직이며 지도·자세만 확인(Cartographer 오차 재기) |
| `lidar/limo_lidar.py` | 시뮬 2D 라이다(EAI X2L 흉내: 6 Hz, 500 광선, 0.12–8 m) |
| `move_robot/move_robot_limo.py` | `move_robot` 도구(libmove_robot)의 행동 8 을 시뮬 행동 칸에 놓기 |
| `move_robot/move_robot_sim.py`, `test_move_robot_sim.py` | 시뮬 쪽 접착부와 그 시험 |
