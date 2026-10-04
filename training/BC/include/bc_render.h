// BC 영상 학생의 렌더(계획서 4.5·4.6 (a), 6절 K11 앞): 기록된 RenderState(144 B) → 카메라 2 대 RGB u8 [E][256][256][3].
// 팀 RenderBatch(src/behavior-2026/src/sim/engine/cuda/render, 서브모듈 — include 로 **읽기만**, 고친 것 없음)를 src/bc_render.cu 한 곳에서만 쓴다
// (팀 렌더 빌드 규칙 -ftz=true 등을 그 파일에만 주려고 이 머리에는 팀 헤더가 없다).
//
// 장면(가정 — 색·조명은 render_bench 와 같음): 바닥 1 + 벽 4 + 가구 칸 8 × 종류 5(종류마다 색이 달라 칸마다 맞는 종류 하나만 보임 비트로 켬) + 컵 1 = 인스턴스 46.
// 인스턴스 하나 = 단위 상자 하나 = 기준 prim(anchor) 하나 → 판마다 축척·자리를 anchor 행렬로. 카메라:
//   머리(RGB-D 렌즈): 몸 기준 (0.094, 0, 0.18) m, 수평, H-FOV 67.9° (env K::cam_x·cam_z·cam_hfov 와 같음)
//   손목(OMX wrist_cam_link): URDF 순기구학(limo_omx_model.h)으로 팔 6 각에서 자세를 판마다 계산, H-FOV 87° (가정)
// 둘 다 256 × 256(SigLIP 2 B/32-256 의 입력 그대로 — 크기 조정 없음). 화소 난수 씨앗은 (판 칸 e, 카메라, 화소, frame = 0) → 같은 상태라도
// 묶음 안 자리 e 가 다르면 그림자 표본 잡음이 다르다(같은 자리면 비트 같음).
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace bc { struct RenderState; }

namespace bcr {

constexpr int RES = 256;
enum Profile { TEAM_DEFAULT = 0, CHEAP = 1 };   // 팀 기본(spp 1, 튕김 1, 반사 1, 잡음 제거 4) / 싼 설정(튕김 0, 반사 0, 잡음 제거 0)

struct Renderer;
// E = 한 번에 그리는 판 수(작업 공간이 E × 256² × 72 B 라 큰 묶음은 나눠 그린다). 만들 때 한 번 그려 둔다(작업 공간 cudaMalloc 을 그래프 잡기 전에).
// 렌더 흔들기(VLA_INPUT 6절·POLICY 3.4 "학생만: 조명·질감·카메라 노출"): on 0 이면 예전과 같은 장면(인스턴스 46). 세기 0..1:
//   color = 무리(바닥·벽·컵·가구 칸)마다 기본 색 대신 흔든 색 변형(3 개 중)을 쓸 확률, light = 조명 방향·자리 흔들기, expo = 노출·대비·채널 이득·감마
// 판마다 값은 RenderState 씨앗(pad[0..3] = 에피소드 번호)에서 → 같은 표본은 같은 그림(기록 = 본 것), 판 안에서는 같은 색·조명
struct RenderAug { int on; float color, light, expo; };
Renderer* create(int E, int profile, const RenderAug* aug = nullptr);
void destroy(Renderer* r);
// 장치 rs[0..n) (n ≤ E) → 카메라 c 의 rgb(c) 칸 [0, n). 비동기, 그래프로 잡힘(잡기 전에 create 가 한 번 그렸으므로 cudaMalloc 없음)
void render(Renderer* r, const bc::RenderState* rs, int n, cudaStream_t st);
// 시험용: 작업 공간을 잡지 않은 새 RenderBatch 를 바로 그래프로 잡아 본다(잡히면 0, 안 되면 CUDA 오류 번호)
int capture_cold_probe(int E, int profile);
const uint8_t* rgb(Renderer* r, int cam);
int batch(Renderer* r);
size_t bytes(Renderer* r);

}  // namespace bcr
