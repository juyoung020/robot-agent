# LIMO + OMX-F GPU 환경 (계획서 G1)

GPU_TRAINING.md 4.3 설계: 환경 하나 = 스레드 하나, 상태는 레지스터(`Core`), 메모리는 SoA. CPU 참조(`env_ref.cpp`)와 GPU 커널이 **같은 소스**(`env.h`)를 쓰고, `detmath.h`(자체 sin/cos/atan2/RNG)와 `--fmad=false`/`-ffp-contract=off` 로 **비트 단위 동일**을 보장한다.

## 구성
- `include/limo_omx_model.h` — URDF 에서 Rust `training/RL/tools/urdf2hdr` 가 생성 (관절 범위는 `src/robot/real_limits.json` 이 덮어씀; 뷰어 에셋과 같은 파일)
- `include/env.h` — 상수 `K`, 리셋, 스텝(10 서브스텝/제어 1틱), FK, 충돌, 관측 80, 보상(CURRICULUM A0/A1/A2 — A2 는 아래 절)
- `tools/env_verify.cu` 비트 동일 검증 + `--negative` 음성 대조, `env_bench.cu` 처리량, `env_view.cpp` 한 판을 sgview 로 스트리밍

## 빌드/실행
```
cmake -S . -B ~/ra_envbuild && cmake --build ~/ra_envbuild -j
~/ra_envbuild/env_verify 2048 600          # GPU == CPU 비트 동일이어야 통과
~/ra_envbuild/env_verify 2048 600 --negative   # 실패해야 정상
~/ra_envbuild/env_verify 2048 600 --stage 2   # A2(가구). --stage 0 = A0, 기본 1 = A1
~/ra_envbuild/env_verify 2048 600 --arm        # 팔을 풀고 행동 8 모두 무작위(VLA_INPUT 5절)
~/ra_envbuild/env_verify 2048 600 --arm-zero   # GPU 팔 풂 + 팔 행동 0 대 CPU 팔 묶음 — 비트가 같아야(학습기가 늘 팔을 풀어 두는 근거)
~/ra_envbuild/env_bench
~/ra_envbuild/env_view 127.0.0.1:9001 20   # 뷰어(--ingest)에 실시간 전송
```

## 결과 (RTX, sm_120)
- 2048 env × 600 스텝 비트 동일(상태·관측·보상·done·rng), 에피소드 종료: 성공 8,907 / 충돌 3,420 / 시간초과 1,543(카메라를 렌즈 프레임 x 0.094 m·FOV 67.9° 로 바꾼 뒤. 전: 0.084 m·71° 에서 성공 8,879). 음성 대조는 49,587,736 불일치로 실패(정상).
- 처리량: N=262,144 → 7.3e8 env-step/s, N=1,048,576 → 7.8e8 env-step/s (1 env-step = 0.1 s 시뮬 = 10 서브스텝+FK+관측+보상).

- 행동 8(2026-10-04): `env_verify 2048 600` 기본·`--arm`·`--arm-zero`·`--stage 2 --arm`·`--stage 2 --arm-zero` 모두 비트 동일, 끝 판수 A1 8,907 / 3,420 / 1,543, A2 21 / 22,170 / 586(팔 묶음과 같음). `--negative` 49,456,558 불일치로 실패(정상). 기본 판은 이제 팔 행동을 0 으로 준다(묶인 팔은 어차피 안 씀).

## 사양 출처
| 값 | 출처 |
|---|---|
| LIMO 322×220×251 mm, tread 175 mm, 최대 1 m/s | Trossen LIMO specs |
| 카메라 렌즈 프레임 base_footprint (0.094, 0, 0.18) m | URDF depth_camera_lens_link(커밋 72a9f7a), scenemap LIMO cam 0 과 같음 |
| 카메라 H-FOV 67.9°(`K::cam_hfov`, 보임 판정·성공 조건) | Orbbec Dabai 깊이 H-FOV. 시뮬은 RGB·깊이 모두 67.9° 로 렌더(72a9f7a). 컬러 사양 71° 는 쓰지 않음 |
| OMX-F 관절 범위, XL430(61 rpm@12 V)/XL330, 도달 400 mm | ROBOTIS OMX specifications |

## 아직 가정인 값 (`env.h` 의 K, `(가정)` 표시)
선/각가속 한도(a_v 1.0, a_w 3.0), 서보 추종 이득 8, XL330 속도(5 V 기준), 성공 조건 임계값. 회사 제공값/실측으로 교체할 것.

## A2 — 가구가 몸통을 막고 카메라를 가림 (CURRICULUM 4절 A2)
`reset_core(c, 2)` = `reset_a2`. 문서 A2 = "가구 몇 개(목표와 로봇 사이 장애물), 시작 2–4 m, 지도 다 앎, 넘어가기 ≥ 85 %·충돌률 ≤ 5 %". 다가가기 보상은 2절대로 **경로 거리**다.

| 무엇 | 구현 (`env.h`) |
|---|---|
| 장면 | 방 반치수 2.5–3.5 m, 목표 0.6 m 여유, 시작 2–4 m(방향 무작위). 상자 8 개 = 지도 장면 상자와 같은 수·종류(`N_FURN`, `FurnCls` = map `Cls`): 막는 가구 1–2 개(로봇–목표 선분 35–65 % 자리, 옆으로 ±0.2 m, 선분을 가로질러 긴 변 0.7–1.4 m, 로봇 0.45 m·목표 0.5 m 여유), 나머지 가구는 벽에 붙임(지도 `make_scene` 과 같은 크기 규칙), 작은 물건 3 개는 방 안 아무 데나 **(가정: 크기·개수·여유)**. 못 놓은 상자는 방 모서리의 작은 물건. 길이 없으면 막는 가구부터 모서리로 옮김 |
| 충돌 | 몸통 직사각형(회전) 대 축 정렬 상자, 분리축 4 개(`body_hits_box`). 서브스텝마다, 벽 다음·컵 앞 |
| 벽 광선 16(G1 관측) | 방 벽 + 가구 상자(`ray_box2`). "장애물 가까움" 벌도 이 광선으로(2절: 몸통 둘레 가장 가까운 장애물) |
| 보임 | 화각 + 카메라(높이 0.18 m) → 컵 중심(0.05 m) 선분이 상자(3D)에 가리지 않음(1절 "깊이로 가리지 않음"). 성공 조건·보임 보상이 이것을 씀 |
| 경로 거리 | 상자를 `path_r` 0.2 m(몸통 외접원 0.194 m, 가정) 부풀리고, 부풀린 모서리 32 개 + 목표의 보임 그래프 최단 경로. 리셋 때 꼭짓점마다 목표까지 다익스트라(`path_prepare`, 상태 `pd[32]`), 스텝마다 로봇에서 보이는 꼭짓점 중 최소(`path_dist`). 다가가기 보상·에임 보상 문턱(1.5 m)·관측 "거리"(72+2)·시간 예산(경로 / 0.3 m/s + 10 s)이 이것을 씀. 직선으로 보이면 직선 거리 |

- **문서와 다른 점**: 2절은 경로 거리를 "정책이 아는 지도(그 순간 자란 지도)" 로 재라고 한다. 여기서는 참 장면(상자)으로 잰다. A2 표의 지도 "다 앎"(= C0)이면 같고, C1·C2(학습기 커리큘럼)에서는 보상만 참 장면을 쓰는 비대칭이다(교사 보상, 관측 아님 — 목표 참값 보상과 같은 성격).
- 지도(`../map`)는 판 리셋 때 환경 SoA 의 상자를 그대로 장면으로 쓴다(광선·가림·검출이 동역학과 같은 장면). A0/A1 은 예전처럼 지도가 스스로 장면을 만든다(몸통과 부딪히지 않음).
- 상태: `Core` 에 `nf, fc[8], fb[8][5], pd[32]`, SoA 에 `F_FB0..`, `F_PD0..`, `I_NF`, `I_FC0..`. `nf = 0`(A0/A1)이면 이 값들을 읽지도 쓰지도 않고 모든 가구 고리가 0 번 돈다 → **A0/A1 은 예전과 비트가 같다**. 장치 상태는 처음에 0 으로 채운다(CPU 참조판과 같게).
- 리셋 통계(CPU, 20,000 판): 경로가 직선보다 긴 판 0.999, 경로/직선 평균 1.147(최대 2.04), 경로 평균 3.23 m, 시작에서 컵이 가려진 판 0.999, 모서리로 옮긴 상자 0.27 개/판, 시작 충돌 0.

검증(`env_verify 2048 600 [--stage s]`, 행동은 앞과 같음 — 무작위 + 짝수 판 비례 제어):

| 단계 | 결과 | 끝: 성공 / 충돌 / 시간초과 |
|---|---|---|
| A1(기본) | 비트 동일 | **8,907 / 3,420 / 1,543** (바꾸기 전과 같음) |
| A0 | 비트 동일 | 13,295 / 2,598 / 1,979 (바꾸기 전 소스로 같은 도구를 A0 로 돌린 값과 같음) |
| A2 | 비트 동일(가구·경로 꼭짓점 상태 포함) | 21 / 22,170 / 586 (비례 제어는 막는 가구로 곧장 감) |
| `--negative` A1 / A2 | 실패(정상) | 49,587,736 / 107,816,782 불일치 |

처리량(`env_bench 300 [stage]`, GPU 다른 일 없음, 같은 때 잼):

| | N 4,096 | N 1,048,576 |
|---|---|---|
| A1 바꾸기 전 소스 | 0.024 ms/스텝 | 1.261 ms/스텝 (8.3e8 env-step/s) |
| A1 지금(가구 코드 뺀 커널 `step_kernel<false>`) | 0.026 | 1.554 (6.7e8) |
| A2 처음(스레드 하나가 리셋 다익스트라까지) | 0.460 | 11.45 |
| A2 + 경로 나눗셈 한 번 + 가구를 Core 밖(상태 자리)으로 | 0.335 | 9.35 |
| A2 지금(리셋 경로 꼭짓점을 워프가 나눔 `step_kernel_a2`) | **0.172** | 4.82 |

- 가구 배열을 `Core` 에 넣었을 때 A1 커널이 레지스터 208 → 168·넘침 356 B 로 1.8 배 느려졌다. 가구 값은 `Core::Furn`(상태 자리를 가리킴)으로 빼고, A0/A1 은 가구 수를 컴파일 때 0 으로 둔 커널(`FURN = false`)을 띄운다. 그래도 N 1M 에서 −19 %(N 4,096 에서는 +2 µs/스텝 = 바퀴당 0.13 ms)가 남았다 — 원인은 찾지 않았다.
- A2 는 판 하나가 스레드 하나라 리셋하는 판(다익스트라: 꼭짓점 32 × 32 × 상자 8 보임 검사)이 커널 전체를 붙잡는다(N 4,096 이면 지연에 묶임). `step_kernel_a2` 는 리셋하는 판마다 워프 32 레인이 꼭짓점 하나씩 맡는다(같으면 앞 번호인 최솟값을 워프 셔플로, 식·순서는 `path_prepare` 그대로 → CPU 참조판과 비트 동일). 남은 0.17 ms 중 리셋 아닌 스텝의 `path_dist`(모든 판 0.05 ms)와 장면 만들기가 대부분이다(가정 — 나눠 재지 않음).

## E2 — BEHAVIOR 집 장면(stage 3, 커리큘럼 B1–B5) (2026-10-04)
계획서 [CURRICULUM_BEHAVIOR2026](../../../docs/map_vla/CURRICULUM_BEHAVIOR2026.md) 3.1절(집기·놓기 거르개 표)·5.4절(상태·잰 값 전부). 상자 방 A0–A2(= B0)는 바이트 그대로다.

```
cmake --build ~/ra_envbuild -j4
~/ra_envbuild/bscene_check                      # 장면 묶음 만들기 + 표 확인(시작 자세·경로·방·창), 장면별 거르개 지남 수
~/ra_envbuild/bscene_check --dump-combos F      # 지시문 조합 → training/embed/pnp_instr.py → training/data/pnp_v1
~/ra_envbuild/pnp_check [10000] [--strict] [--negative free_area|in_closed|artic|spawn_reach|spawn_free|dst_reach|stance]
~/ra_envbuild/env_verify 2048 600 --stage 3 --follow [--strict] [--split 1] [--mix p1,p2] [--negative | --negative-scene]
~/ra_envbuild/env_bench 300 3 32768
```

| 파일 | 내용 |
|---|---|
| `include/bscene.h` | 장치 장면(`SceneSet`: 장면 ≤ 8 × 회전 상자·1 m 묶음·방 격자·문·띠 점유·칸 성분, 시작 조건 `Entry`, 집기·놓기 고르기 표, 이름 확신도 표), 커리큘럼 `BCurr`, 지도 되먹임 `NavFb`, 기하(OBB SAT·광선·묶음 걷기), B4·B5 판정 정의 |
| `include/bscene_host.h`, `src/bscene_host.cpp` | 호스트: RASC v3(`tools/b1kconv/cpp/rasc.h`) + vla_v1 이름 표 → 묶음, 벽 상자 문 자르기, 시작 조건 표(B1 방 표, 집기·놓기 표 + 창 닿는 칸 비트), 거르개 표 `PnpFilter`(RASC LIMITS 를 읽음), `upload`. **장치 커널은 이 파일에 기대지 않는다**(학습기·BC·뷰어 빌드 그대로) |
| `include/env_beh.h` | 판 리셋(B1 방 표 / B2–B5 인스턴스 → 물체 → 놓을 곳, 무작위 시작 `spawn_ok`), 스텝(공용 `act_prepare`·`substeps`·`obs_body` + 장면 충돌·광선·보임·거리·성공) |
| `include/omx_workspace_grasp.h` | `tools/omx_ws --grasp` 생성: E0 잡는 점(ee 링크 −0.0119 m) 작업 공간 — B3 성공 판정 |

API(뷰어·학습기용, 뒤로 맞음 — 예전 호출은 그대로):
- `DeviceEnv(N, stage, seed, arm_free, const bsc::SceneSet* ss_dev = nullptr, const bsc::BCurr& cu0 = kBCurrDefault)`, `kStageBeh = 3`. `bcurr_dev()`·`set_bcurr_source(ptr)`(장치 커리큘럼 값), `set_nav(map.nav_fb())`(지도 거리장·목표 확정 되먹임). `CpuEnv(..., &scenes.host, cu)` 와 `CpuEnv::nav`.
- SoA 끝에 더함(앞 자리 번호 그대로): `F_B_WX..F_B_DIST`(창 가운데 세계 좌표 — **판의 x·y·목표는 창 좌표**, 세계 = 창 + (F_B_WX, F_B_WY)), `I_B_KIND`(0 상자 방, 1 B1, 2 B2, 3 B3), `I_B_SCENE`, `I_B_ENT`(→ `SceneBuild::ent`: 물체 9·놓을 곳·이름·RASC 번호), `I_B_ROOM`, `I_B_FSET`(쓴 거르개), `I_B_INSTR`(지시문 행). `I_NF` = 0 이라 상자 가구는 없다 — BEHAVIOR 판 장면은 `SceneBuild::sc[I_B_SCENE]`(상자·방)·`ent[I_B_ENT].prim` 으로 그린다.
- `step_core` 는 같은 연산을 공용 조각으로 나눈 것(결과 바이트 같음, 뷰어 훅 그대로).

잰 값(5.4절): `env_verify 2048 600 --stage 3 --follow` 비트 동일(대본 판 B1 성공 407·B2 312·B3 241), 음성 대조 30,059,004·11,539,830 불일치로 실패(정상). 상자 방 env 스트림 해시 6 설정이 f5b4ec7 빌드와 같음. 처리량(ms/스텝) A2 0.176 / 0.235, BEHAVIOR 0.336 / 0.444(N 4,096 / 32,768). `pnp_check` 10,000 판 위반 0(느슨·엄격), GPU == CPU 고르기 비트 동일. 장면 묶음 장치 61.2 MB, 만들기 약 9 s(CPU, 장면마다 스레드).

## 성능 메모(호스트 의존 점검, 2026-10-04 — `../ppo/README.md`·`../../BC/README.md` 같은 절)
- step·판 리셋은 모두 장치 안이다(호스트 일 없음). PPO·BC 정상 구간의 바퀴·그래프당 호스트 동기 0(nsys).
- 남은 호스트 의존 = **단계·씨앗 바꾸기마다 `DeviceEnv` 를 새로 만듦**(cudaMalloc·동기 cudaMemcpy·init 커널) → 학습기가 띄운 바퀴를 비우고 동기 + 그래프 다시 잡기. 잰 GPU 빈틈: PPO 환경 단계 바꾸기 번마다 약 7 ms, BC `bc_reset_env`(평가·기록 씨앗) 번마다 약 16 ms. 환경 단계·씨앗도 `BCurr` 처럼 장치 값(`set_stage`/`reseed`: 같은 버퍼에 init 커널만)이면 다시 만들기·다시 잡기가 없어진다(제안, 안 고침).
- BEHAVIOR 리셋(`pick_pnp` 장면 ≤ 32 훑기 + `spawn_ok` 최대 64 번)은 step 커널 안에서 판마다 돈다 → 리셋하는 판이 든 워프는 그 판을 기다린다. 잰 0.336 ms/스텝(N 4,096)에서 리셋 몫은 따로 재지 않았다 — 판 수·리셋 비율이 커지면 clock64 구간으로 볼 것.
