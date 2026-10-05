# sgrt — 장면 그래프 런타임(libsgrt.so): 검출 + 물체 기억을 한 C ABI 로

평가기(또는 로봇) 프로세스 안에서 물체 기억을 실시간으로 굴린다. 공유 라이브러리 하나에 세 가지를 묶는다.

- 검출: ovdet(TensorRT). 기본은 이름 없는 분할 엔진 ObjectSAM(FastSAM-s 에서 증류한 YOLO26n 학생, things 만, `yolo26n-seg-obj-416.plan`, https://github.com/juyoung020/ObjectSAM) + SigLIP 2 이름·임베딩 + scenemap 확률 모드(아래 "objprob 앞단"). 보관한 YOLOE·닫힌 어휘 YOLO-seg 도 고를 수 있다(옛 이름 규칙). 머리 RGB 를 장치 메모리에서 바로 읽는다.
- 지도·물체 기억: scenemap(slam2d·objmap·방·장면 그래프·저장). 깊이는 호스트 f32 미터.
- (선택) 물체 영상 임베딩: sgclip(SigLIP 2, `../clip`). `SGRT_CLIP` 이면 켜짐.

파이썬은 포인터만 넘긴다(`glue/sgrt_glue.py`). 한 스레드에서 부른다. 설계는 [docs/scenemap_설계.md](../../../docs/scenemap_설계.md) 3.2.2·3.5·3.6·3.7절.

```
매 스텝     sgrt_step(proprio)                    → scenemap 자세 적분, 든 물체 따라가기
keyframe    sgrt_step(proprio + 머리 RGB + 깊이)  → ovdet 검출 → scenemap 물체 지도 갱신(+ best view 자르기·점 구름 색)
지도 스텝   sgrt_step(proprio + 깊이만)           → slam2d 격자만 갱신(SGRT_MAP_EVERY, 글루가 정함)
주기 저장   시뮬 save_s 마다                       → 저장 스레드가 out_dir 에 scene.json · view.json · map.pgm …
```

keyframe 인지는 `sgrt_want_image()` 가 알려 준다(`kf_every` 스텝마다, 기본 6). 호출자는 그 스텝에만 영상 포인터를 넘긴다.

## 배치

| 파일 | 하는 일 |
|---|---|
| `include/sgrt.h` | C ABI: `sgrt_create`·`sgrt_begin`·`sgrt_step`·`sgrt_save`·`sgrt_destroy`, 통계·시간(`sgrt_stats`·`sgrt_get_timing`·`sgrt_get_stage_timing`), 지도 보기(`sgrt_map`·`sgrt_map_snapshot`, scenemap 문맥 `sgrt_scenemap` — 다른 스레드의 읽기 도구가 자기 스냅숏·`sm_observe_object_name` 을 쓰려고, 10-05), 자세 원천(`sgrt_set_pose_mode`·`sgrt_push_pose`·`sgrt_get_pose_diag`), 이름 종류(`sgrt_set_kind_names`), 임베딩·이름 찾기(`sgrt_object_embedding`·`sgrt_query_embedding`·`sgrt_query_label`·`sgrt_object_names`·`sgrt_get_clip_stats`) |
| `src/sgrt.cpp` | 구현: ovdet + scenemap 연결, 프롬프트 표(`SGRT_PROMPT`), 영상 시각 늦춤(`SGRT_IMAGE_LAG`), 저장 스레드, 스트림 요약 스레드(`SGRT_STREAM`), 입력 기록(`SGRT_RECORD`), 단계 시간(det·step·map·record) |
| `src/crop.cu` · `crop.hpp` | CUDA 커널 둘: best view 상자 자르기(넓이 평균, 요청 32개씩 한 번에, 자른 것만 고정 메모리로 내려받기)와 점 구름 화소 색 모으기. 식은 `scenemap/src/bestview.cpp` 호스트 판과 같다 |
| `src/sgrt_clip.cpp` · `.hpp` | sgclip 연결: keyframe 마다 새 물체·best view 품질이 임베딩 때의 1.2배 이상인 물체를 최대 8개 비동기로 임베딩. 저장 때 `objects/O<id>_emb.f16`, `cache/names.json`, scene.json 노드 metadata(`sm_set_object_meta`). 라벨 표는 따로 스레드에서 읽음 |
| `glue/sgrt_glue.py` | 평가기 쪽 접착부 `SceneMemory(task, out_dir)` · `step(obs)` · `close()`. 과제 프롬프트(`../ovdet/config/task_prompts.txt` 의 과제 줄 + `_scene` 줄), GT 자세 넣기, GT 기록, 900 스텝마다 진단·시간 출력 |
| `tests/test_crop.cpp` | 장치 자르기·색 모으기 = 호스트 식, 시간 |
| `tools/sgrt_replay.cpp` | sgrt 기록(`SGRT_RECORD`)을 libsgrt 로 다시 굴림(검출 → 지도 → 저장). 라이브러리는 dlopen — 옛 빌드와 새 빌드를 같은 입력으로 바이트 비교 |
| `tools/dom_bench_det.cpp` | dynamic-object-mapping-benchmark 시퀀스를 실제 검출(ovdet FastSAM-s 416, conf 0.25)로 scenemap 에 넣어 `map_timeline.csv`·`map_points.npz`. `--classify`: 검출마다 SigLIP 2 임베딩 → 글 프롬프트(벤치마크 범주·구조물) 코사인 최대를 cls 로. 없으면 sgrt 와 같이 모두 'object' |
| `tools/sgrt_frames.cpp` | 실제 엔진으로 끝까지 확인: raw RGB 프레임을 장치에 올려 `sgrt_step` → 저장. 깊이는 평평한 2 m(가짜) — 검출 → 자르기·색 모으기 → PNG/PLY 경로 확인용 |

## 만들기

CUDA 12.8(`/usr/local/cuda-12.8`, `OVDET_CUDA_ROOT`), sm_120(`CMAKE_CUDA_ARCHITECTURES`). `../scenemap`·`../ovdet`·`../clip` 을 `add_subdirectory` 로 같이 빌드한다.

```bash
cmake -S src/scene_graph/runtime -B ~/sgrt_build && cmake --build ~/sgrt_build -j
ctest --test-dir ~/sgrt_build -R crop             # 장치 자르기 시험
~/sgrt_build/sgrt_frames <engine.plan> <out_dir> <w> <h> <frames.rgb ...>   # <engine.plan>.names.txt 가 옆에 있어야 함
SGRT_SAVE_SYNC=1 ~/sgrt_build/sgrt_replay ~/sgrt_build/libsgrt.so <engine.plan> rec.bin <out_dir> [--traj t.csv] [--frames N]
```

글루는 `~/sgrt_build/libsgrt.so` 를 기본으로 읽는다(`SGRT_LIB`).

```python
from sgrt_glue import SceneMemory
mem = SceneMemory(task_name, out_dir)   # 프로세스에 한 번
mem.step(obs)                           # 평가기 스텝마다, 정책 앞에서
mem.close()                             # 이름 보고·진단 출력, 마지막 저장
```

- 머리 RGB 는 GPU 텐서 그대로 넘긴다. 깊이는 keyframe(과 지도 스텝)에만 고정 메모리 버퍼로 내려받는다.
- 관측에 머리 RGB-D 가 없으면 한 번 경고한다(`RGBDFullResWrapper` 를 쓸 것).
- 쓰는 곳: `src/sim/move_robot/run_eval_move.py`, `src/sim/explore/run_explore.py`.

## 로봇 고르기(R1 Pro · LIMO + OMX-F)

기본은 R1 Pro(옛 동작 그대로 — `sm_create(NULL)`). `sgrt_create` 때 환경 변수로 고른다.

| 방법 | 뜻 |
|---|---|
| `SGRT_ROBOT=r1pro \| limo_omx` | `sm_create("{\"robot\": \"<값>\"}")` |
| `SGRT_SM_CONFIG='<json>'` | `sm_create` 의 config_json 그대로(`robot`·`odom`·`grip_closed`, `../scenemap/README.md` LIMO 절). `SGRT_ROBOT` 보다 먼저 |
| `sgrt_set_robot(s, 1)` | 만든 뒤 바꾸기(지도·물체 비움). `sgrt_begin` 앞에서. `sgrt_get_robot`·`sgrt_proprio_dim` 으로 확인 |

모르는 로봇·틀린 json 이면 `sgrt_create` 가 NULL(err 에 까닭). LIMO 면 `sgrt_step` 의 proprio 는 12 f32(`SM_LIMO_*`), 영상은 몸통 앞 깊이 카메라(scenemap cam 0 = `depth_camera_lens_optical_frame`, 렌즈)와 그 내부 파라미터다. `sgrt_step` 은 영상 하나만 받으므로 손목 카메라(cam 1, 깊이 없음)는 넘기지 않는다(지도에도 안 씀).

**글루(`SceneMemory`)** 가 로봇을 정하는 순서: 인자 `robot_model=` → `SGRT_ROBOT` → 시뮬 로봇의 `robot.model`(`limo_omx`) → 첫 스텝 관측에 `:eyes:Camera:0` 이 있으면 LIMO 로 바꿈(`sgrt_set_robot`). R1 이면 아무것도 안 바꾼다.

| | R1 Pro | LIMO + OMX-F (OmniGibson `limo_omx`, robot-agent `src/robot/og/limo_omx_eval.yaml`) |
|---|---|---|
| 지도 카메라 | `<r>:zed_link:Camera:0` (720×720) | `robot_limo:eyes:Camera:0` (`RGBDFullResWrapper` 면 720×720) → cam 0 |
| 내부 파라미터 | 고정 `HEAD_K`(306, 306, 360, 360) | 시뮬 센서 `intrinsic_matrix` 에서 읽음(해상도가 바뀌면 다시) |
| proprio | 평가기 61 그대로 | 평가기 proprio(`base_qvel, arm_0_qpos, arm_0_qvel, eef_0_pos, eef_0_quat, gripper_0_qpos, gripper_0_qvel`, 24) → 12: 0–2 오도메트리 = `base_qvel` 을 30 Hz 로 적분(정답 자세 안 씀), 3–5 `base_qvel`(vx, vy, wz 베이스 기준), 6–10 `arm_0_qpos`(omx_joint1..5), 11 `gripper_0_qpos[0]`(omx_gripper_joint_1). 자리는 `robot._proprio_obs` 에서 구하고(없으면 yaml 순서) 크기가 다르면 멈춤 |
| 손목 | — | `robot_limo:wrist_eye:Camera:0` = cam 1(RGB 만), 넘기지 않음 |

시뮬 확인: `src/sim/limo/run_limo_map.sh [task] [steps]` — 평가기를 LIMO 로 띄우고(robot-agent `eval_with_limo.py`) 제자리 한 바퀴 + 앞이 비면 직진·막히면 왼쪽으로 꺾기, 끝에 지도 ↔ 정답 바닥 지도(`src/sim/explore/gt/`)·자세 오차·몸통 카메라 외부 파라미터 ↔ scenemap 순기구학을 `summary.json`·`overlay.png` 로. 탐색은 아래 "LIMO 탐색".

**LIMO 시뮬 확인**(10-04, turning_on_radio 인스턴스 0, headless, `SGRT_POSE=slam`, 엔진 yolo26s-seg, 1200 스텝 = 40 s): 정답 경로 5.0 m·회전 662°, slam ↔ 정답 자세 keyframe 200 개 rms 1.3 cm / 0.21°, 최대 2.7 cm / 0.37°. 지도 50 m² 알려짐, 점유 칸의 95 %(±5 cm)가 정답 바닥 밖(벽·가구), 빈칸의 90 % 가 정답 바닥, 아는 영역 안 정답 바닥 경계(벽)의 95 % 가 ±10 cm 안에서 점유. 물체 19 개. 몸통 카메라 외부 파라미터: 시뮬 센서 ↔ scenemap 순기구학 cam 0 위치 차 2e-7 m, 회전 0.02°. 내부 파라미터 720×720 fx = fy = 306, cx = cy = 360. 기록(`SGRT_RECORD`)을 `sm_bench --robot limo_omx` 로 재생하면 실시간과 같은 자세·물체 수.

**로봇 자산(카메라)**: 예전 `limo_omx` 는 `eyes` 카메라가 `depth_camera_link` 자리, 차체 껍데기 앞면보다 약 1 cm 안쪽이라 자기 몸만 봤다(깊이 ≈ 0.0096 m, RGB 검정). robot-agent 391c04b 부터 카메라를 렌즈 자리(`depth_camera_link` +x 0.010 m, base_link (0.094, 0, 0.03))로 옮기고 `eval_with_limo.py` 가 가까운 자르기 0.05 m 를 준다. `run_limo_map.py`·`run_explore.py` 는 옛 자산 대비로 가까운 자르기 면을 `LIMO_NEAR_CLIP`(기본 0.05 m, 0 = 안 건드림)까지 **올리기만** 한다(`src/sim/move_robot/move_robot_limo.apply_near_clip`). scenemap 순기구학의 cam 0 은 아직 `depth_camera_link`(렌즈 +0.010 m 안 넣음) — 1 cm 차이, 표 맞추기는 따로.

### LIMO 탐색 (스킬 explore)

`SGRT_ROBOT=limo_omx src/sim/explore/run_explore.sh frontier turning_on_radio [tag]` — R1 과 같은 스크립트·에이전트(robot-agent `explore`)·move_robot(libmove_robot)·지도 넘기기. R1 과 다른 점만:

| R1 가정 | LIMO 에서 |
|---|---|
| 평가기 `omnigibson.eval.eval` + 기본 R1 Pro 설정 | `$ROBOT_AGENT/src/robot/og/eval_with_limo.py`(미리 뽑은 시작 자세 별칭, agent_metric 고침) + `--robot-config limo_omx_eval.yaml` (`run_explore.py --robot limo_omx --limo-shim`, 기본은 `SGRT_ROBOT`) |
| move_robot proprio 61 / 행동 23 (베이스·몸통·두 팔·두 그리퍼) | `src/sim/move_robot/move_robot_limo.py`: libmove_robot 은 그대로 R1 으로 돌리고 바이트만 바꿈. proprio 24 의 `base_qvel` → R1 61 의 0–2(나머지 0), 행동 23 의 베이스(BASE_OUT 0.75, 0.75, 1.0 으로 m/s·rad/s) → LIMO 9: 베이스 0–2 = 홀로노믹 vx, vy, wz / (0.5, 0.5, 0.8727) 자르기, 팔 3–7 = `reset_joint_pos`(홈 [0, -1.6, 1.45, 0.15, 0]) 고정, 그리퍼 8 = -1(닫힘, 0 은 반 열림이라 안 씀). 칸은 시뮬 로봇의 `controller_action_idx`·`_proprio_obs` 에서 읽음 |
| 몸통·팔 집어넣기(tuck), 머리 카메라 기울기 = 몸통 관절 유지 | 없음(카메라 몸통 고정). 베이스 아닌 `part` 호출은 move_robot 에 안 가고 오류로 답함 |
| 머리 카메라 `zed_link`, 고정 HEAD_K | 글루가 `eyes` 카메라·센서 내부 파라미터(위 표) |
| 정답 기록 끔 | `SGRT_GT_LOG=<out>/gt_poses.csv` 기본(+ `.objects.json` 정답 물체), `<out>/poses.csv`(keyframe 마다 정답 world·map 틀 자세 ↔ slam 자세), `<out>/pose_diag.json`(sgrt_get_pose_diag, 약 1 초마다 — 평가기가 close 전에 끝나므로) |
| 몸통(libmove_robot `nav.rs` Footprint, R1 원 0.37 m · 계획 부풀림 0.40) | **LIMO 사각형 0.36 × 0.22 m**(시뮬 충돌 모양: 몸통 0.322, 바퀴 폭 0.217, 홈 자세 팔이 뒤로 0.18 m 까지 → 대칭), 부풀림 = 외접원 0.211 + 0.03 = 0.241 m. `MOVE_ROBOT_FOOTPRINT=limo_omx`(run_explore.sh 가 LIMO 일 때 기본으로 넣음, `rect:LxW`·`circle:R` 도 됨). 없으면 R1 그대로 |

그 밖: `MOVE_ROBOT_LIB` 기본 = `$ROBOT_AGENT/src/agent/tools/move_robot/target/release/libmove_robot.so`, 정답 바닥 지도는 `src/sim/explore/gt` 가 없으면 `~/behavior-2026/src/sim/explore/gt`(서브모듈 안에서는 둘 다 같은 경로). libsgrt 는 `sgrt_set_robot` 이 있는 빌드여야 함(없으면 멈춤). R1(`SGRT_ROBOT` 없음·r1pro)은 바뀐 것 없음.

**LIMO 탐색 결과**(10-04, turning_on_radio 인스턴스 0 = house_double_floor_lower, headless, frontier, `SGRT_POSE=slam`, robot-agent 391c04b 자산 — `LIMO_NEAR_CLIP` 우회 안 씀(이미 0.05), 엔진 기본 yoloe-11l): `no_frontier` 로 끝, go_to 5 번, 시뮬 58.4 s(벽 62 s). 경로 정답 13.1 m(move_robot `path_m` 11.0 — base_qvel 적분, 약 16 % 짧음), 빈칸 53.3 m², 닿을 수 있는 정답 바닥의 88.7 %. slam ↔ 정답 keyframe 291 개 rms 3.1 cm / 0.26°, 최대 5.3 cm / 0.66°, 끝 4.1 cm / 0.49°. 막힘·멈춤·접촉 0, 최소 여유 0.10 m. 지도: 빈칸의 90 % 가 정답 바닥, 점유 칸의 96 % 가 정답 비바닥 ±10 cm 안. 물체 16 개 모두 정답 물체 AABB 10 cm 안, 범주로 보면 13 개 맞음(radio·sofa·shelf·coffee/breakfast table·조명), 3 개 틀림(lamp→stairs, picture frame→hall_tree, radio receiver→downlight). 같은 조건 R1 확인 판: 16.6 m·56.5 m²·90.3 %·67 s.

**LIMO 탐색 결과 2**(10-04, 같은 조건 + 렌즈 프레임 cam 0(scenemap FK = OmniGibson eyes, 차 2.6e-7 m), eyes 수평 화각 67.9°(Dabai 깊이, fx 534.7 @ 720), move_robot 몸 0.36 × 0.22 m(`MOVE_ROBOT_FOOTPRINT=limo_omx`), path_m 고침): `no_frontier`, go_to 15 번, 시뮬 157.4 s(벽 183 s). move_robot `path_m` 38.35 m 대 정답 38.15 m(+0.5 %, 전 −16 %). 빈칸 59.2 m², 정답 바닥의 95.0 %. slam ↔ 정답 keyframe 786 개 rms 9.9 cm / 0.30°, 최대 12.6 cm / 1.75°, 끝 11.2 cm / 0.43°(경로가 3 배 길고 화각이 좁아짐). 막힘·멈춤·접촉 0, 최소 여유 0.05 m. 판: `outputs/explore_20261004_100358_turning_on_radio_frontier_limo`.

**R1 회귀**(10-04): 바꾸기 전(7219187)과 뒤의 libsgrt 를 `sgrt_replay` 로 같은 기록 3 개(`mem_pose_slam_*`, `mem_pose_gt_move*`, 엔진 yolo26s-seg) × 자세 모드 3 개(slam·gt·odom)에 굴려(`SGRT_SAVE_SYNC=1`) 저장 디렉터리 전부(map.pgm·scene.json·view.json·물체 PNG/PLY)와 keyframe 자세 CSV 가 바이트까지 같음 — `SGRT_ROBOT` 없음, `SGRT_ROBOT=r1pro` 둘 다. `sm_bench --save --traj` 9 개도 같음.

## 환경 변수

`src/sgrt.cpp`·`src/sgrt_clip.cpp`(라이브러리)와 `glue/sgrt_glue.py`(글루)가 읽는 것 전부.

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `SGRT_ROBOT` | 없음(R1) | 로봇: `r1pro` · `limo_omx`(위 "로봇 고르기"). 글루도 읽는다 |
| `SGRT_SM_CONFIG` | 없음 | scenemap `sm_create` config_json 그대로(예: `{"robot": "limo_omx", "odom": "twist"}`). 있으면 `SGRT_ROBOT` 무시 |
| `SGRT_POSE` | `slam` | 자세 원천. `slam`(적분 + 스캔 맞추기) · `odom`(적분만) · `gt`(외부 정답 베이스 자세, map = 시뮬 world — 진단·시각화용, 대회 제출 금지). 그 밖의 값은 `slam`. 시뮬은 실제 로봇과 같게 `slam` 으로 재고, `gt` 는 확인용으로만 |
| `SGRT_MAP_POLICY` | `1` | 격자 넣기 정책. 1 = 사건 기반(서 있어도 바뀐 장애물을 넣고 지움), 0 = 옛 움직임 거르기 |
| `SGRT_IMAGE_LAG` | `1` | 영상 stamp 를 몇 스텝 앞 시각으로 둘지(0..7). 1 = 직전 스텝(평가기 영상 k = 장면 k-1). 글루도 GT 자세를 읽을 스텝을 고를 때 같은 값을 쓴다 |
| `SGRT_SAVE_SYNC` | `0` | 0 이 아니면 주기 저장을 스텝 안에서 한다. 0 이면 저장 스레드(앞 저장이 안 끝났으면 그 주기는 건너뜀) |
| `SGRT_STREAM` | 없음(끔) | `host:port`. sgview(`--ingest`)로 실시간 스트림. 자세·지도 변화분은 스텝 안에서 링 버퍼로, 요약은 따로 스레드가 만든다. 5 s 마다 stderr 에 스트림 통계 |
| `SGRT_STREAM_HZ` | `60` | 요약(view) 만드는 주기 Hz, 0.5–240 으로 자름 |
| `SGRT_RECORD` | 없음(끔) | 파일 경로. 받은 입력(proprio·외부 자세·keyframe 깊이·RGB·검출)을 이진(`SGRC` 판 1)으로 기록. `scenemap/tools/sm_bench`·`stage_bench` 가 재생 |
| `SGRT_PROMPT` | `auto` | 프롬프트 표. `task`(과제 이름만) · `all`(엔진 어휘 전부) · `auto`(어휘 200 이하 닫힌 어휘 엔진이면 `all`, 아니면 `task`). 프롬프트가 비면 늘 `all` |
| `SGRT_CLIP` | 없음(끔) | SigLIP 2 엔진 plan 경로. `1` = `~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan`, `0`·빈 값 = 끔 |
| `SGRT_CLIP_GRAPH` | sgclip 기본(1) | 배치 크기별 CUDA graph 쓰기(`sgc_config.use_graph`) |
| `SGRT_LABELS` | `~/embed_work/labels/objects-v1` | 라벨 표 폴더(이름 찾기). 색인 캐시는 `<out_dir>/cache/index` |
| `SGC_IMG_SAMPLE` | `~/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16`(있으면) | 라벨 표 투영을 맞출 영상 임베딩 표본(sgclip 변수, sgrt_clip 이 읽어 넘김) |
| `SGRT_LIB` | `~/sgrt_build/libsgrt.so` | 글루: 읽을 라이브러리 |
| `SGRT_ENGINE` | `~/ovdet_models/x86_sm120/yolo26n-seg-obj-416.plan` (ObjectSAM — YOLO26n 학생, 이름 없는 분할) | 글루: 검출 엔진. 이름 표는 `<엔진>.names.txt`. 원래 FastSAM-s = `FastSAM-s-416.plan`, 버린 FastSAM-s 재학습 = `~/ovdet_models/archive/x86_sm120/FastSAM-s-416-obj.plan`(보관), YOLOE·YOLO26s = `~/ovdet_models/archive/x86_sm120/`(옛 이름 규칙) |
| `SGRT_OBJPROB` | 없음(자동) | objprob 앞단(아래 "objprob 앞단"). `1` 켬 · `0` 끔 · 없음 = 엔진 어휘가 `object` 하나(분할 엔진)면 켬 |
| `SGRT_OBJPROB_PARAMS` | `tools/realbag/objprob_params/<엔진 줄기>.json` | objprob 엔진별 매개변수 파일, `none` = 내장 기본값 |
| `SGRT_OBJPROB_CLIP` | `~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan` | objprob 마스크 임베딩 SigLIP 2 엔진 |
| `SGRT_INSPECT` | `0` | `1` = 살펴본 정도(scenemap README "살펴본 정도", view.json·scene.json 물체 `inspect`) |
| `SGRT_GT_POSE` | `1` | 글루: 시뮬 로봇 베이스 정답 자세를 `sgrt_push_pose` 로 넣기(`0` = 끔). gt 모드면 map 자세, 아니면 떠밀림 진단에만 쓰임 |
| `SGRT_GT_EVERY` | `0` | 글루: `1` 이면 정답 자세를 매 스텝 읽음. 아니면 영상 시각·keyframe·지도 스텝에 필요한 스텝만 |
| `SGRT_MAP_EVERY` | `0` | 글루: n 스텝마다 깊이만으로 격자 갱신(검출·물체 지도·임베딩은 keyframe 그대로). 0 = keyframe 에서만 |
| `SGRT_GT_LOG` | 없음(끔) | 글루: CSV 경로. keyframe 마다 정답 베이스·머리 카메라 자세를 쓰고, 정답 물체(이름·범주·위치·AABB)를 `<csv>.objects.json` 에 씀(`scenemap/eval/score_map_gt.py` 가 읽음) |

`sgrt_config` 기본값(`sgrt_default_config`): `kf_every` 6, `save_s` 1.0(시뮬 초), `conf_th` 0.25. 글루도 같은 값을 넘긴다.

## objprob 앞단 (10-05, 기본 켬)

분할 엔진(ObjectSAM `yolo26n-seg-obj-416.plan`, 어휘 `object` 하나)이면 libsgrt 가 `tools/realbag/realbag_run` 과 같은 길로 돈다(낱말 표·라벨 사전·엔진별 매개변수는 `src/objprob_front.hpp` 하나를 같이 씀).

1. keyframe: ovdet 분할 → 마스크마다 SigLIP 2 임베딩(장치 RGB 에서 바로, 같은 스텝 안에서 기다림) → 낱말 표 글 임베딩 최댓값 = 검출 이름, `sm_set_det_embeddings`.
2. `sm_push_image_rgb` — scenemap 확률 모드: vMF 벡터 합치기, 베이즈 이름 합치기, 이름 없는 같은 것 판정(로지스틱 = 로그 우도비), 칼만 위치, 랜색 평면으로 벽·천장·바닥 조각 거르기, 문·창 크기 검사, 벽 높이 규칙.
3. `sm_reencode_requests` → 구름 투영 마스크로 SigLIP 2 → `sm_set_object_embeddings`(통째 다시 담기).

판 시작(`sgrt_begin`)의 과제 이름은 엔진 프롬프트가 아니라 낱말 표에 더한다(이미 있는 낱말·라벨은 빼고, SigLIP 2 라벨 표에 있는 것만). 켜면 `SGRT_CLIP`(ClipMem)은 안 켠다 — 물체 벡터 `objects/O<id>_emb.f16`(μ)·`_views.f16` 은 scenemap 이 저장하고 `sgsearch`·`search_objects` 가 읽는다. 끝날 때 stderr 에 keyframe 당 SigLIP 2·다시 담기 시간.

확인: radio r3 기록(`~/datasets/limo_rec/r3.bin`)을 `sgrt_replay` 로 — 730 keyframe, 마스크 17.8 개/kf, SigLIP 2 3.7 ms/kf, 다시 담기 0.33 ms/kf, 살아 있는 노드 104(같은 기록을 realbag 스트림으로 바꾼 `realbag_run` 판 96 — 깊이 4 m 자름·영상 늦춤이 다름).

## 시험 (ctest)

| 시험 | 확인 |
|---|---|
| `crop` | 720 × 720 RGBA 난수 영상에서 상자 6개(작은 것·줄이는 것·가장자리) 장치 자르기 = 호스트 `cropRgbHost`, 화소 3000개(영상 밖 좌표 포함) 색 모으기 = 호스트 `gatherRgbHost`. 장치 자르기·온 영상 내려받기 시간을 찍는다. GPU 가 없으면 실패 |

scenemap 시험은 [../scenemap/README.md](../scenemap/README.md), sgclip 시험은 [../clip/README.md](../clip/README.md).
