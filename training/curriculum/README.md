# 강화학습 커리큘럼 — 교사 시연 → RecallVLA 모방학습 (2026-10-06 결정)

큰 그림: **다양한 RL 교사**를 만들어 **다양한 스폰·맵·시나리오**에서 모범 답안(시연)을 얻고, RecallVLA(학생)가 그 시연을 **모방학습**(BC → DAgger)한다. 지도는 정답(GT)이 아니라 빈 지도에서 SLAM 으로 자라는 지도다.

| 단계 | 배우는 것 | 교사가 받는 목표 | 지금 학습 환경 단계 | 끝 판정 |
|---|---|---|---|---|
| 1 | 지도 최대한 많이 쌓기 | 없음(탐사) | **새로 만들어야 함** — 새로 본 칸 = 보상(GPU 지도가 방마다 이미 셈) | 정해진 시간 안 덮은 비율 |
| 2 | 목표 지점 주면 가기 | 지도 위 점 | B1 점으로 가기(`p_goto`) | 점 0.3 m 안에서 멈춤 |
| 3 | 물체 찾기 | 물체 종류 | B2 찾기 | 목표 물체 지도 확정 + 카메라에 보임 |
| 4 | 물체 잡기 | 물체 | B4 집기(잡는 자세에서) | 들어 올림 0.05 m · 0.5 s |
| 5 | 물체 찾아서 잡기 | 물체 종류 | B2 찾기 + B3 다가가기 + B4 집기(놓기 없음, 환경에 더해야 함) | 4 와 같음 |

## 누가 배우나
- **교사 = RL 정책**(`training/RL`, PPO): 시뮬 상태를 아는 정책이 단계마다 과제를 먼저 익힌다. 교사는 글을 읽지 않는다 — 목표는 목표 칸(지점 좌표 또는 물체)으로만 받는다. 판에 붙은 지시문은 학생용이다.
- **학생 = RecallVLA**(`training/BC`·`training/vla`): 교사들의 시연을 모방학습해, 카메라·물체 기억(기억 인코더)·지시문만으로 같은 과제를 한다.

## 어디서 배우나
- 맞는 곳: OmniGibson 에서 실제 인지(ObjectSAM + SigLIP 2 + scenemap, Cartographer SLAM)의 점구름 지도로 직접 학습. **나중에 이쪽으로 옮긴다.**
- 지금(시연용): GPU 환경(`training/RL`, 상자 근사 지도, 빈 지도 C2)에서 학습 → 판 기록(`record_ppo`) → 같은 판을 OmniGibson + libsgrt 로 다시 돌림(`og_replay`, `_og.sg`: 점구름 지도·몸통 카메라·손목 카메라) → 학습 뷰어 리플레이 탭을 1배속으로 찍어 GIF.

## 시연 GIF 만들기
1. 학습: `build/cargo/ppo_driver/release/ppo_run <설정> --out training/runs/ppo/<이름>`
2. 판 기록: `build/record_replay/record_ppo --ckpt <ckpt_final.bin> --out <실행 폴더> --config <설정> --stage 3 --pnp … --scenes house_single_floor`
3. 실제 인지로 다시 돌리기: `training/viewer/tools/og_replay/og_replay.sh <판.trp>` (og.lock)
4. 학습 뷰어 리플레이 탭을 1배속으로 재생하며 화면을 연속으로 찍고(찍은 시각 = 파일 시각), `tools/frames2gif.py <프레임 폴더> <out.gif>` 로 찍은 시각 그대로 GIF.

## 상태
| 단계 | 설정 | 학습 | 시연 GIF |
|---|---|---|---|
| 1 지도 쌓기 | — | 환경에 과제를 더해야 함 | — |
| 2 지점 가기 | `config/c2_goto_point.json` | 20 분, 성공률 0.75 | `docs/assets/curriculum_2_goto.gif` |
| 3 물체 찾기 | `config/c3_find.json` | 20 분, 성공률 낮음(약 0.08) — 다시 맞출 것 | `docs/assets/curriculum_3_find.gif`(펜 찾기 성공 판) |
| 4 잡기 | — | PPO 처음부터 10 분: 성공 0 — 대본 교사 → BC → PPO 가 필요 | — |
| 5 찾아서 잡기 | — | 3·4 뒤 | — |
