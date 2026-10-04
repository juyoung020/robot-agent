# 설정 (G3)

Rust 실행기(`../ppo/driver`)가 읽는 JSON. 빠진 키는 실행기 기본값(`main.rs` `make_config`)을 쓴다.

| 파일 | 내용 |
|---|---|
| `ppo_a0a1.json` | G1 CURRICULUM A0 → A1, G2 자라는 지도 토큰. 아래 값 |

| 키 | 값 | 근거 |
|---|---|---|
| `n_env` × `horizon` | 4,096 × 64 | 계획서 8.1 |
| `epochs`, `minibatches` | 5, 4 (미니배치 65,536) | (가정) 흔한 PPO 값 |
| `gamma`, `lam`, `clip`, `vclip`, `vf_coef`, `ent_coef` | 0.99, 0.95, 0.2, 0.2, 0.5, 0 | (가정) |
| `lr`, `adaptive_lr`, `kl_target`, `lr_min/max` | 3e-4, 켬(미니배치 KL > 2·목표면 ÷1.5, < ½·목표면 ×1.5), 0.01, 1e-5/1e-3 | (가정) 장치 값으로 조절 |
| `max_grad_norm` | 1.0 | (가정) |
| `init_logstd` | −0.5 | (가정) |
| `act_dims` | 2 | CURRICULUM 3절: approach 는 vx, wz 만 학습. G1 행동 8 중 나머지 6 은 0 고정 |
| `shaping` | coef 1, near 8, aim 1, zone 1, v 4, w 2 | `../reward/README.md` (가정) |
| `curriculum` | 단계 [A0, A1], 넘어가기 = 최근 20 바퀴 확률적 성공률 평균 ≥ 0.9, 멈춤 = 마지막 단계 ≥ 0.98 | CURRICULUM 4절(≥ 90 %). "최근 1,000 판" 대신 바퀴 평균 (가정) |
| `budget_minutes`, `ckpt_every`, `inflight` | 30, 200 바퀴, 3 | 띄워 둔 바퀴 수(기록 링 16 칸 안) |
