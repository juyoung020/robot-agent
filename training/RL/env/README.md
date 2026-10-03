# LIMO + OMX-F GPU 환경 (계획서 G1)

GPU_TRAINING.md 4.3 설계: 환경 하나 = 스레드 하나, 상태는 레지스터(`Core`), 메모리는 SoA. CPU 참조(`env_ref.cpp`)와 GPU 커널이 **같은 소스**(`env.h`)를 쓰고, `detmath.h`(자체 sin/cos/atan2/RNG)와 `--fmad=false`/`-ffp-contract=off` 로 **비트 단위 동일**을 보장한다.

## 구성
- `include/limo_omx_model.h` — URDF 에서 Rust `training/RL/tools/urdf2hdr` 가 생성 (관절 범위는 `src/robot/real_limits.json` 이 덮어씀; 뷰어 에셋과 같은 파일)
- `include/env.h` — 상수 `K`, 리셋, 스텝(10 서브스텝/제어 1틱), FK, 충돌, 관측 80, 보상(CURRICULUM A0/A1)
- `tools/env_verify.cu` 비트 동일 검증 + `--negative` 음성 대조, `env_bench.cu` 처리량, `env_view.cpp` 한 판을 sgview 로 스트리밍

## 빌드/실행
```
cmake -S . -B ~/ra_envbuild && cmake --build ~/ra_envbuild -j
~/ra_envbuild/env_verify 2048 600          # GPU == CPU 비트 동일이어야 통과
~/ra_envbuild/env_verify 2048 600 --negative   # 실패해야 정상
~/ra_envbuild/env_bench
~/ra_envbuild/env_view 127.0.0.1:9001 20   # 뷰어(--ingest)에 실시간 전송
```

## 결과 (RTX, sm_120)
- 2048 env × 600 스텝 비트 동일(상태·관측·보상·done·rng), 에피소드 종료: 성공 8,879 / 충돌 3,420 / 시간초과 1,543. 음성 대조는 약 5천만 불일치로 실패(정상).
- 처리량: N=262,144 → 7.3e8 env-step/s, N=1,048,576 → 7.8e8 env-step/s (1 env-step = 0.1 s 시뮬 = 10 서브스텝+FK+관측+보상).

## 사양 출처
| 값 | 출처 |
|---|---|
| LIMO 322×220×251 mm, tread 175 mm, 최대 1 m/s | Trossen LIMO specs |
| 카메라 컬러 H-FOV 71° | Orbbec Dabai |
| OMX-F 관절 범위, XL430(61 rpm@12 V)/XL330, 도달 400 mm | ROBOTIS OMX specifications |

## 아직 가정인 값 (`env.h` 의 K, `(가정)` 표시)
선/각가속 한도(a_v 1.0, a_w 3.0), 서보 추종 이득 8, XL330 속도(5 V 기준), 성공 조건 임계값. 회사 제공값/실측으로 교체할 것.
