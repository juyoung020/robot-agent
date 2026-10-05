# vla

작은 계획·행동 (우리 VLA). 지시 + 카메라 영상을 받아 할 일을 잘게 나눠 로봇 행동을 내고, 눈앞의 실패는 스스로 복구한다.

- 우리 VLA 는 **RecallVLA**(Qwen3.5-0.8B 전부 학습 + SigLIP 2 + 물체 기억 → 단계 문장 + 행동, [MAPVLA_SPEC.md](../../docs/map_vla/MAPVLA_SPEC.md))다. 학습기는 [`training/vla`](../../training/vla/README.md)(M1–M5 끝, 큰 학습은 아직).
- 지금 도는 작은 학생(얼린 SigLIP 2 영상 탑 + 물체 기억 지도 토큰 + 지시 문장 → flow matching 행동)은 [`training/BC`](../../training/BC/README.md) — 시뮬 RL·대본 교사(`training/RL`)에게서 BC·DAgger 로 배운다.
- 이 폴더는 아직 README 만 있다(로봇에서 돌리는 실행 코드 없음).
