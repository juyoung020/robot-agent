# src

코드를 둔다. ROS 2 패키지는 이 폴더 안에 만든다 (레포 루트에서 `colcon build`).

| 폴더 | 역할 |
|---|---|
| `robot/` | 리모(AgileX LIMO) + 매니퓰레이터(ROBOTIS OMX-F) 로봇 설명: 통합 URDF·RViz(`map_vla_description`), 업스트림 패키지 받기(`fetch_upstream.sh`). [robot/README.md](robot/README.md) |
| `scene_graph/` | 동적 scene graph: 로봇이 본 물체를 2D 지도에 등록·갱신 (로봇의 기억). [scene_graph/README.md](scene_graph/README.md) |
| `agent/` | LLM: 사람과 대화, 물체 기억을 읽어 π0.5 에게 상황 풀어 주기, 물체를 놓치면 다시 계획 |
| `agent/skills/` | 스킬: LLM 에이전트가 한 가지 일을 끝까지 해내는 단위(시스템 프롬프트·도구·과제 문장). 지금은 `explore/` (Rust). [agent/skills/README.md](agent/skills/README.md) |
| `agent/tools/` | LLM 에게 보이는 도구. 지금은 `move_robot/` (Rust, 닫힌 고리 실행기). [agent/tools/README.md](agent/tools/README.md) |
| `agent/prompts/` | 스킬이 같이 쓰는 공통 프롬프트. [agent/prompts/README.md](agent/prompts/README.md) |
| `vla/` | VLA(π0.5): 지시 + 카메라 영상 → 작은 계획·행동, 눈앞의 실패 복구 |
| `app/` | 휴대폰 앱 (iOS·Android 네이티브). 카카오톡식 채팅으로 로봇에게 명령, 지금은 로봇1 만 |
| `behavior-2026/` | 서브모듈 [juyoung020/behavior-2026](https://github.com/juyoung020/behavior-2026): BEHAVIOR Challenge 2026 작업(시뮬레이터 평가·π0.5 네이티브 엔진·물체 기억). 받기: `git submodule update --init src/behavior-2026` |

## 물체 기억 (`scene_graph/`)

실제 로봇 쪽 코드가 이 저장소 `scene_graph/` 에 있다. 원본은 서브모듈의 `behavior-2026/src/scene_graph/` 이고, 거기서 고친 뒤 `tools/sync_scene_graph.sh` 로 여기에 맞춘다(`--check` 로 어긋남 확인).

| 폴더 | 역할 |
|---|---|
| `scenemap/` | 2D SLAM + 물체 지도 + 계획기 질의, Spark-DSG 저장 (C++/CUDA). 설계: [scenemap_설계.md](https://github.com/juyoung020/behavior-2026/blob/main/docs/scenemap_설계.md) |
| `da/` | 데이터 연관: 프레임마다 나온 같은 물체 세그를 하나로 병합 |
| `ovdet/` | 열린 어휘 검출기 (YOLOE, TensorRT, C API) |
| `clip/` | sgclip: 물체 조각 → SigLIP 2 B/32 영상 임베딩(TensorRT) + 라벨 표 찾기 (C++/CUDA, 작업 중). 후보 조사: [docs/clip_candidates.md](../docs/clip_candidates.md), 라벨 표·학습: [training/README.md](../training/README.md) |
| `runtime/` | sgrt: 평가기(또는 로봇) 프로세스 안에서 물체 기억을 굴리는 C ABI 하나. 자세는 `SGRT_POSE` |
| `sgview/` | **장면 그래프 뷰어 = sgview** (Spark-DSG 장면 그래프 보기, Rust 서버 + three.js, 실시간). 실행: `tools/run_sgview.sh <memory_dir> [--live]`, 탐사 + 뷰어 `tools/run_explore_live.sh`. [sgview/README.md](scene_graph/sgview/README.md) |
| `spark_dsg/` | Spark-DSG 우리 사본(BSD-3, 바꾼 것은 `OUR_CHANGES.md`) |

옛 파이썬 뷰어 sgviz(Spark-DSG + viser, 서브모듈 `behavior-2026/src/scene_graph/viewer/`)는 파일 폴링이라 실시간이 아니고 **쓰지 않는다**(기록용). 뷰어는 sgview 하나다.
