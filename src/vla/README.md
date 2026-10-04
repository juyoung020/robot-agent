# vla

작은 계획·행동 (우리 VLA). 지시 + 카메라 영상을 받아 할 일을 잘게 나눠 로봇 행동을 내고, 눈앞의 실패는 스스로 복구한다.

- π0.5 는 쓰지 않는다(10-04 버림). 우리 VLA 는 얼린 SigLIP 2 영상 탑 + 물체 기억 지도 토큰 + 지시 문장 → flow matching 행동 전문가인 작은 VLA 다. 시뮬 RL 교사(`training/RL`)에게서 BC·DAgger 로 배운다.
- 학습 코드와 결과는 [`training/BC`](../../training/BC/README.md)(G5). 얼린 Qwen 을 쓰는 큰 판은 G7 계획([GPU_TRAINING.md](../../docs/map_vla/GPU_TRAINING.md) 11절).
- 이 폴더는 아직 README 만 있다(로봇에서 돌리는 실행 코드 없음).
