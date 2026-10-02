# src

코드를 둔다. ROS 2 패키지는 이 폴더 안에 만든다 (레포 루트에서 `colcon build`).

| 폴더 | 역할 |
|---|---|
| `scene_graph/` | 동적 scene graph: 로봇이 본 물체를 2D 지도에 등록·갱신 (로봇의 기억) |
| `agent/` | LLM: 사람과 대화, 물체 기억을 읽어 π0.5 에게 상황 풀어 주기, 물체를 놓치면 다시 계획 |
| `agent/skills/` | 스킬: LLM 에이전트가 한 가지 일을 끝까지 해내는 단위(시스템 프롬프트·도구·과제 문장). 지금은 `explore/` (Rust). [agent/skills/README.md](agent/skills/README.md) |
| `agent/tools/` | LLM 에게 보이는 도구. 지금은 `move_robot/` (Rust, 닫힌 고리 실행기). [agent/tools/README.md](agent/tools/README.md) |
| `agent/prompts/` | 스킬이 같이 쓰는 공통 프롬프트. [agent/prompts/README.md](agent/prompts/README.md) |
| `vla/` | VLA(π0.5): 지시 + 카메라 영상 → 작은 계획·행동, 눈앞의 실패 복구 |
| `app/` | 휴대폰 앱 (iOS·Android 네이티브). 카카오톡식 채팅으로 로봇에게 명령, 지금은 로봇1 만 |
| `behavior-2026/` | 서브모듈 [juyoung020/behavior-2026](https://github.com/juyoung020/behavior-2026): BEHAVIOR Challenge 2026 작업(시뮬레이터 평가·π0.5 네이티브 엔진·물체 기억). 받기: `git submodule update --init src/behavior-2026` |

## behavior-2026 안의 물체 기억 (`behavior-2026/src/scene_graph/`)

이 저장소 `scene_graph/` 의 실제 코드는 아직 서브모듈 쪽에 있다.

| 폴더 | 역할 |
|---|---|
| `scenemap/` | 2D SLAM + 물체 지도 + 계획기 질의, Spark-DSG 저장 (C++/CUDA). 설계: [scenemap_설계.md](https://github.com/juyoung020/behavior-2026/blob/main/docs/scenemap_설계.md) |
| `ovdet/` | 열린 어휘 검출기 (YOLOE, TensorRT, C API) |
| `clip/` | sgclip: 물체 조각 → SigLIP 2 B/32 영상 임베딩(TensorRT) + 라벨 표 찾기 (C++/CUDA, 작업 중). 후보 조사: [docs/clip_candidates.md](../docs/clip_candidates.md), 라벨 표·학습: [training/README.md](../training/README.md) |
| `runtime/` | sgrt: 평가기(또는 로봇) 프로세스 안에서 물체 기억을 굴리는 C ABI 하나. 자세는 `SGRT_POSE` |
| `viewer/` | sgviz: 물체 기억 실시간 뷰어 (Spark-DSG + viser) |

Spark-DSG 는 서브모듈의 `third_party/spark_dsg/` 에 우리 사본으로 둔다(BSD-3, 바꾼 것은 `OUR_CHANGES.md`).
