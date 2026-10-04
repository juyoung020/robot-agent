# 설정 (G3)

Rust 실행기(`../ppo/driver`)가 읽는 JSON. 빠진 키는 실행기 기본값(`main.rs` `make_config`)을 쓴다.

| 파일 | 내용 |
|---|---|
| `ppo_a0a1.json` | G3: G1 CURRICULUM A0 → A1, G2 자라는 지도 토큰, 목표 참값(특권). 아래 값 |
| `ppo_a0a1_bound.json` | G3 와 같고 `bound_coef` 0.5 만 더함(벽 충돌 고침 확인용, 10 분) |
| `ppo_a0a1_coll.json` | 위 + 충돌 추가 벌 `shaping.coll` −20(벽 충돌 원인 확인 — 결정적 충돌 0.0298 → 0.0006) |
| `ppo_g4.json` | **G4**: 목표는 지도에서(`goal_from_map` 1), 처음 지도 커리큘럼 A0C0 → A1C0 → A1C1 → A1C2, `bound_coef` 0.5 |
| `ppo_g4_notok.json` | G4 와 같고 지도 토큰만 끔(`use_map` 0) — 토큰 켬/끔 비교 |

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

G4 에서 더한 키:

| 키 | 값 | 근거 |
|---|---|---|
| `goal_from_map` | 1 | 계획서 5.4: 교사도 지도에 확정된 목표만 앎(`../observation/README.md`). 0 = G3 특권 |
| `shaping.coll` | 0(G4 판), −20(`ppo_a0a1_coll`) | (가정) 충돌 스텝에 더하는 보상. `../reward/README.md` |
| `bound_coef` | 0.5 | (가정) 정책 평균이 ±1 밖이면 coef·(|μ|−1)². `../network/README.md` |
| `map_kmin`, `map_kmax`, `map_reveal_r` | 3, 6, 1.5 m | C1: 참 물체 9 개 중 3–6 개(33–67 %, 계획서 30–70 %) 미리 확정, 그 둘레 1.5 m 격자 공개 (반경은 가정) |
| `curriculum.stages[]` | `{name, env, map: [C0, C1], promote, metric}` | `env` = G1 단계(A0/A1), `map` = 처음 지도 비율(나머지 C2), `metric` = 넘어가기를 재는 처음 지도(0 C0, 1 C1, 2 C2, −1 전체). 정수만 쓰면 예전 꼴(G3) |
| 단계 | A0C0 [1,0] ≥ 0.9 → A1C0 [1,0] ≥ 0.8 → A1C1 [0.2, 0.8] ≥ 0.7 → A1C2 [0.1, 0.1] | 계획서 5.5: C0 ≥ 0.8, C1 ≥ 0.7. 앞 단계 약 20 % 유지(가정) — C1 단계는 C0 20 %, C2 단계는 C0·C1 각 10 %. A0 는 G3 처럼 ≥ 0.9 |
| `window` | 20 바퀴, 에피소드 가중 | 지도 비율을 바꾸면 이미 띄운 바퀴(예전 비율)의 기록은 넘어가기 판단에서 뺌 |
| `budget_minutes` | 35 | 토큰 켬/끔 두 판 + 확인 판 10 분 = GPU 약 80 분 |
