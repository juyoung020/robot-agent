# docs

회의 자료, 설계 문서, 발표 자료를 둔다.

| 문서 | 내용 |
|---|---|
| [plan.md](plan.md) | 계획: 무엇을, 어디서, 어떤 순서로 |
| [model_selection.md](model_selection.md) | 모델 선택: 부품마다 무엇을 골랐고 왜 골랐나 |
| [perception_model_candidates.md](perception_model_candidates.md) | 물체 인식 모델 후보 (① 물체 기억, 추천만) |
| [clip_candidates.md](clip_candidates.md) | CLIP 류 임베딩 모델 후보·측정 (분할 마스크 → 이름·임베딩, 측정 때 분할은 FastSAM-s — 지금은 ObjectSAM) |
| [terms.md](terms.md) | 용어 정리: stuff·things, 확률론적 물체 수준 매핑, DA·과분할 병합·라벨 융합, PCA 와 랜색(RANSAC) — 우리 코드 위치와 함께 |
| [known_bugs.md](known_bugs.md) | 알려진 버그(코드): 옛 경로 import, `/mnt/c/behavior-2026` 박아 둔 스크립트 등 |
| [map_vla/](map_vla/README.md) | Map_Vla(리모 + 매니퓰레이터 VLA) 조사·설계 문서 |
| [assets/](assets/) | README 아이콘·뷰어·시뮬·로봇 그림 |

## 다른 곳의 문서

| 문서 | 내용 |
|---|---|
| [training/README.md](../training/README.md) | 로봇에 올릴 작은 모델 학습 (embed/: 영상–글 임베딩 증류, RL/·BC/: VLA 학습, model/: 베이스 모델) |
| [src/scene_graph/README.md](../src/scene_graph/README.md) | 물체 기억 코드(실제 로봇 쪽)·동기화·빌드 |
| [src/robot/README.md](../src/robot/README.md) | 리모 + 매니퓰레이터 로봇 설명(URDF·RViz) |
| [scenemap_설계.md](https://github.com/juyoung020/behavior-2026/blob/main/docs/scenemap_설계.md) | 서브모듈: 물체 기억 scenemap 설계 (2D SLAM·물체 지도·계획기 질의·자세 원천) |
| [archive/README.md](https://github.com/juyoung020/behavior-2026/blob/main/archive/README.md) | 서브모듈: 지금 안 쓰는 모듈, 왜 옮겼고 어떻게 되살리나 |
| [tools/README.md](https://github.com/juyoung020/behavior-2026/blob/main/tools/README.md) | 서브모듈: 실행·측정·검증 스크립트 |
| [behavior-2026 docs/](https://github.com/juyoung020/behavior-2026/blob/main/docs/README.md) | 서브모듈: BEHAVIOR Challenge 2026 문서 전체 목록 |

## 정한 것

- 이미지 임베딩: SigLIP 2 B/32.
- 분할(10-05 결정): ObjectSAM(FastSAM-s 에서 증류한 YOLO26n 학생, things 만 — 엔진 `yolo26n-seg-obj-416`, [github.com/juyoung020/ObjectSAM](https://github.com/juyoung020/ObjectSAM) v1.0) + SigLIP 2 + objprob(scenemap 확률 모드, 기본 켬, 매개변수 `objprob_params/yolo26n-seg-obj-416.json`), 입력 416. 까닭: FastSAM-s 계산의 약 1/10 이라 LIMO 의 Jetson(특히 Nano)에 맞다 — 기기 위 시간은 아직 안 잼.
- 물체 벡터는 원본 임베딩 그대로 두고, 이름은 기억 폴더의 `cache/` 에 둔다.
- CUDA 12.8.
- 지도 자세: `SGRT_POSE`(`slam`·`odom`·`gt`, 실제 로봇 기본 `slam`, 시뮬 시험은 `gt`).
