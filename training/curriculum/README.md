# 강화학습 커리큘럼 — 시연 판 (2026-10-06 결정)

한 장면(OmniGibson `house_single_floor`, 지금 시연 판의 침실 쪽)에서 네 단계를 차례로 배운다. 지도는 정답(GT)이 아니라 빈 지도에서 SLAM 으로 자라는 지도다.

| 단계 | 배우는 것 | 지금 학습 환경 단계 | 끝 판정 |
|---|---|---|---|
| 1 | 탐색 | B2 찾기(빈 지도에서 지시된 물체 종류를 지도에 확정하고 카메라에 넣기) | 목표 물체 확정 + 보임 |
| 2 | 목표를 주면 가기(goal) | B1 점·방으로 가기(`p_goto`) | 목표 점 0.3 m 안에서 멈춤 |
| 3 | 목표 집기 | B4 집기(잡는 자세에서) | 들어 올림 0.05 m · 0.5 s |
| 4 | 목표를 주면 가서 집기 | B3 다가가기 + B4 집기(놓기 없음) | 3 과 같음 |

## 누가 배우나
- **교사 = RL 정책**(`training/RL`, PPO): 시뮬 상태를 아는 정책이 단계마다 과제를 먼저 익힌다. 아래 시연 GIF 는 이 교사의 판이다.
- **학생 = RecallVLA**(`training/BC`·`training/vla`): 교사의 시연을 모방학습(BC → DAgger)해, 카메라·물체 기억(기억 인코더)만으로 같은 과제를 한다.

## 어디서 배우나
- 맞는 곳: OmniGibson 에서 실제 인지(ObjectSAM + SigLIP 2 + scenemap, Cartographer SLAM)의 점구름 지도로 직접 학습. **나중에 이쪽으로 옮긴다.**
- 지금(시연용): GPU 환경(`training/RL`, 상자 근사 지도, 빈 지도 C2)에서 학습 → 판 기록(`record_ppo`) → 같은 판을 OmniGibson + libsgrt 로 다시 돌림(`og_replay`, `_og.sg`: 점구름 지도·몸통 카메라·손목 카메라) → 학습 뷰어 리플레이 탭을 1배속으로 찍어 GIF.

## 시연 GIF 만들기
1. 학습: `build/cargo/ppo_driver/release/ppo_run <설정> --out training/runs/ppo/<이름>`
2. 판 기록: `build/record_replay/record_ppo --ckpt <ckpt_final.bin> --out <실행 폴더> --config <설정> --stage 3 --pnp … --scenes house_single_floor`
3. 실제 인지로 다시 돌리기: `training/viewer/tools/og_replay/og_replay.sh <판.trp>` (og.lock)
4. 학습 뷰어 리플레이 탭을 1배속으로 재생하며 화면을 연속으로 찍고(찍은 시각 = 파일 시각), `tools/frames2gif.py <프레임 폴더> <out.gif>` 로 찍은 시각 그대로 GIF.

## 상태
| 단계 | 학습 | 시연 GIF |
|---|---|---|
| 1 탐색 | 아직(README 의 탐색 GIF 는 학습한 정책이 아니라 frontier 탐사기) | — |
| 2 가기 | 20 분, 성공률 0.75(`config/c2_goto.json`) | `docs/assets/curriculum_2_goto.gif` |
| 3 집기 | PPO 처음부터 10 분: 성공 0 — 대본 교사 → BC → PPO 가 필요 | — |
| 4 가서 집기 | 아직(놓기 없는 판을 환경에 더해야 함) | — |
