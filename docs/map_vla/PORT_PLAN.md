# MuJoCo Warp 전체 포팅 계획

> **대체됨 (10-04)**: 이 포팅은 하지 않는다. [BEHAVIOR_ENGINE_REVIEW.md](BEHAVIOR_ENGINE_REVIEW.md) 6절 제안대로 물리는 **우리 환경 커널**(G1, `training/RL/env`, `0eddb72` — LIMO + OMX, CPU/GPU 비트 동일)로, 영상은 **팀 `RenderBatch`**(서브모듈 `src/behavior-2026/src/sim/engine`, include 로 읽기만 — `training/BC/src/bc_render.cu`, `b27aafd`)로 간다. 저장소에 MuJoCo 코드는 없다. 지금 계획은 [GPU_TRAINING.md](GPU_TRAINING.md) 11절, 상태는 [TODO_TRACKER.md](TODO_TRACKER.md). 아래는 기록으로 남긴다.

> **주의 (10-03)**: `~/behavior-2026/src/sim/engine`에 CUDA 물리와 배치 렌더러가 이미 있다. 이 계획과 겹치는지 먼저 `BEHAVIOR_ENGINE_REVIEW.md`를 볼 것. 빌드 기준 CUDA는 12.8이다(13.2 아님).

결정: `refs/code/mujoco_warp` **전체**를 C++/CUDA 네이티브로 포팅한다 (`CLAUDE.md`: PyTorch/Python 금지).
이 문서는 조사한 규모와 의존 관계, 포팅 순서다. 코드는 아직 한 줄도 쓰지 않았다.

## 1. 규모 (`mujoco_warp/_src`, 테스트 제외)
- Python 약 **60,000줄** (테스트 별도 약 33,000줄), `@wp.kernel` 354개, `@wp.func` 392개.
- 큰 모듈: `constraint` 6.1k, `smooth` 4.6k, `solver` 4.4k, `derivative` 4.3k, `collision_flex` 3.9k, `io` 3.3k, `sensor` 3.1k, `types` 2.8k, `collision_gjk` 2.7k, `forward` 2.2k, `passive` 2.1k.
- 줄 수는 Python 기준이다. C++/CUDA로 옮기면 템플릿 때문에 더 늘 수 있다 (추정, 측정 안 함).

## 2. Warp 내장 기능 중 직접 구현해야 하는 것
소스에서 센 사용 횟수:
| Warp 기능 | 횟수 | C++/CUDA 대체 |
|---|---|---|
| `wp.vec/mat/spatial` 타입 | 3101 | 자체 작은 선형대수 헤더 (`__host__ __device__`) |
| `wp.static` (컴파일 타임 특수화) | 403 | C++ 템플릿 / `constexpr` |
| `wp.tile` (블록 협력 연산) | 263 | 공유 메모리 + cooperative groups, 필요 시 cuBLASDx 등 |
| `wp.atomic_*` | 212 | CUDA atomics |
| `wp.Bvh` / `wp.Mesh` / `mesh_query_ray` | 약 25 | 자체 LBVH (Morton 코드 + CUB radix sort) + 광선 순회 |
| tile Cholesky (`block_cholesky.py`) | 33 | 블록 Cholesky 커널 직접 구현 |
| `wp.capture_*` (CUDA graph) | 5 | CUDA Graph API |

## 3. 모델 로딩은 어떻게 하나 (결정 필요)
`io.py`는 `mujoco`(CPU 라이브러리)로 컴파일한 `MjModel`을 GPU 구조체로 옮기는 코드다. MJCF/URDF 파서와 모델 컴파일러는 mujoco_warp 안에 **없다**.
- **제안**: 공식 MuJoCo C 라이브러리(3.14.0, `libmujoco.so`)를 링크해서 모델 컴파일과 CPU 기준 구현(오라클)으로 쓴다. 네이티브 C이고 Python이 필요 없다. 포팅 범위가 `io.py`의 변환부로 줄고, 모듈마다 `mj_step` 결과와 비교해 검증할 수 있다.
- 릴리스: `mujoco-3.14.0-linux-x86_64.tar.gz` (GitHub google-deepmind/mujoco). mujoco_warp는 `mujoco>=3.12.0`을 요구한다.
- 현재 이 PC에는 libmujoco가 없다. 받는 것은 아직 안 했다.

## 4. 의존 레벨 (import 분석)
```
L0 types(2806) block_cholesky(295) warp_util util_pkg
L1 math(434) history(1184) island(942) sleep(999)
L2 support(1287) util_misc(892) bvh(1273) ray(1332) collision_core(528) collision_primitive_core(1910)
L3 smooth(4625) constraint(6088) passive(2094) collision_primitive(1611) collision_gjk(2729) collision_sdf(1107) render_util(796)
L4 sensor(3145) derivative(4333) set_const(953) collision_convex(1507) collision_flex(3876) render(1604)
L5 solver(4404) collision_driver(984)
L6 forward(2152) io(3290)
L7 inverse(234) cli(297)
```

## 5. 포팅 순서 (마일스톤)
앞 단계가 뒤 단계를 막지 않게, 로봇 팔과 depth가 가장 빨리 쓸 수 있는 순서다.
| 단계 | 모듈 | 얻는 것 | 검증 (libmujoco 대비) |
|---|---|---|---|
| A. 기반 | types, math, 선형대수 헤더, io(변환부) | 모델/데이터 GPU 구조체 | 모델 필드 왕복 일치 |
| B. 운동학·동역학 | support, util_misc, smooth | FK, CRB, RNE, 질량행렬 | `mj_kinematics`, `mj_crb`, `mj_rne` 값 비교. 우리 URDF zero pose의 TF 값 |
| C. 렌더 | bvh, ray, render_util, render | **배치 depth/RGB** | 같은 장면의 `mjr`/ray 결과 비교 |
| D. 접촉 없는 스텝 | forward(접촉 제외), passive, sensor | 관절 PD로 팔 구동 | `mj_step` 궤적 비교 |
| E. 제약·솔버 | constraint, block_cholesky, solver, derivative | 정확한 제약 동역학 | 솔버 반복 수렴·`efc_*` 비교 |
| F. 충돌 | collision_core, primitive(_core), convex, gjk, driver | 물체 접촉 | 접촉점·법선 비교 |
| G. 나머지 | collision_sdf, collision_flex, island, sleep, history, set_const, inverse, cli | 전체 완성 | 모듈별 테스트 |

## 6. 위험
- Warp의 JIT 특수화(`wp.static`)에 많이 의존한다. C++ 템플릿으로 옮길 때 인스턴스 폭발과 컴파일 시간이 문제가 될 수 있다.
- 부동소수점 결과가 CPU와 비트 단위로 같지 않다. 허용 오차 기준을 모듈마다 정해야 한다.
- `collision_flex`(3.9k줄)는 변형체용이다. 우리 로봇에 필요한지 불확실하다.
- 5070 Ti는 sm_120이다. nvcc 13.2는 확인됐지만 `-arch=sm_120` 컴파일과 실행은 아직 시험하지 않았다.
