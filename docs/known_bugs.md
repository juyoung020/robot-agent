# 알려진 버그 (코드)

문서를 코드와 맞추다가 찾은, **코드 쪽** 문제를 모은다. 고치면 이 표에서 지우거나 "고침" 으로 적는다.
처음 정리 2026-10-03. 경로는 이 저장소 기준(서브모듈은 `src/behavior-2026/...`).

| # | 어디 | 무엇 | 영향 | 고치는 법 |
|---|---|---|---|---|
| 1 | `src/behavior-2026/src/scene_graph/ovdet/tools/ovdet_eval.py:33` (동기화 사본 `src/scene_graph/ovdet/tools/ovdet_eval.py` 도 같음) | `sys.path` 에 옛 경로 `src/scenemap/eval` 을 넣는다. 폴더 개편 뒤 경로는 `src/scene_graph/scenemap/eval` | ovdet 평가 스크립트가 `gt_scene` 을 못 불러와 실패할 것으로 보임 | `os.path.join(ROOT, 'src', 'scene_graph', 'scenemap', 'eval')` 로. 서브모듈에서 고친 뒤 `tools/sync_scene_graph.sh` |
| 2 | `src/agent/skills/explore/src/bin/explore.rs:91` | 주석은 몸통 원 0.28 m 인데 코드(같은 파일 115 줄)는 0.37 m | 동작은 0.37 m. 주석만 틀림 | 주석을 0.37 m 로 |
| 3 | 서브모듈 스크립트 약 80 개 (아래 목록) | 저장소 경로를 `/mnt/c/behavior-2026` 으로 박아 둠(Windows·WSL 시절 경로) | 작업 PC `jy-desktop` 에서는 심볼릭 링크 `/mnt/c/behavior-2026` → 서브모듈 이 있어서 돈다(서브모듈 `docs/Linux_설치.md` 3.1). 링크가 없는 PC(학교 4090 등)에서는 실패 | 스크립트 위치에서 저장소 뿌리를 찾거나(`$(cd "$(dirname "$0")/../.." && pwd)`) 환경 변수로 받는다. `tools/README.md` 의 리눅스판 스크립트가 이미 이렇게 한다 |
| 4 | `src/behavior-2026/src/sim/integ/build_simlink.sh:4,8` | 사용 예가 `wsl.exe -d Ubuntu-22.04 ...`, 본문이 `cd /mnt/c/behavior-2026/...` | 3 과 같음 | 3 과 같이 |
| 5 | `src/behavior-2026/src/agent/planner/src/main.rs:214` | `build-assets --root` 기본값이 `/mnt/c/behavior-2026` | `--root` 를 안 주면 3 과 같음 | 기본값을 실행 파일 위치나 현재 폴더 기준으로 |
| 6 | `src/behavior-2026/tools/check_agent_hook.py:16` | `sys.path` 에 `/mnt/c/behavior-2026/tools` | 3 과 같음 | 파일 위치 기준(`Path(__file__).parent`) |

## 3 번 목록 (`/mnt/c/behavior-2026` 을 쓰는 실행 줄이 있는 파일, 주석만 있는 것은 뺌)

서브모듈 `src/behavior-2026/` 기준.

| 폴더 | 파일 |
|---|---|
| `src/sim/engine/capture/` | `export_s3.py`, `extract_batch.sh`, `extract_instance.sh`, `run_capture_linux.sh` |
| `src/sim/engine/eval/` | `run_ported_engine.sh` |
| `src/sim/engine/scripts/` | `build_replay.sh`, `gen_aos.py`, `gen_aos_test.py`, `gpu_lock.sh`, `linux_env.sh`, `setup_linux_official.sh` |
| `src/sim/engine/tests/` | `articulation/` (`build.sh`, `l1_all.sh`, `ptxinfo.sh`, `run_gpu.sh`, `run_gpu_all.sh`, `run_gpu_regs.sh`, `snap.sh`, `snapneg.sh`, `sweep.sh`), `common/` (`run_sleef_trigf.sh`, `test_sleef_trigf.py`), `joints/` (`build.sh`, `run_cpu.sh`, `run_gpu.sh`, `run_gpu_sq.sh`), `omni/` (`build_omni.sh`, `gen_bddl_ref.py`, `run_gfmat.sh`, `run_obs.sh`, `test_gfmat.py`, `test_obs.py`), `particles/` (`build_particles.sh`, `capture_*.py`, `harvest_spawn.py`, `probe_load_order.py`, `run_*.sh`), `render/` (`build_render.sh`, `compare/build.sh`, `gpu_tests.sh`) |
| `src/sim/fasteval/` | `pi05_chunk_server.py`, `verify_chunk_equivalence.py`, `replaysrv/verify.sh`, `replaysrv/verify_vs_python.py`, `tracecmp/verify_vs_python.sh` |
| `src/sim/integ/` | `build_simlink.sh`, `fk/fit_cam_fk.py` |
| `src/vla/pi05_native/tools/` | `dump_reference.py`, `dump_reference_pb.py`, `export_pb_all.sh`, `export_weights.py`, `gemm_run.sh`, `make_inputs.py`, `make_pb_host_cases.py`, `pb_host_check.sh`, `run_dumps.sh`, `server_check.sh`, `server_client_test.py`, `server_protocol_test.py`, `session_verify.sh`, `tok_check.sh` |
| `src/vla/pi05_train/tools/` | `aug_ref.py`, `make_state.py`, `session.sh`, `train_ref.py` |
| `tools/` | `check_agent_hook.py`, `ft_gpu_suite.sh`, `ft_nvdec_util.sh`, `ft_run.sh`, `run_pi05_chunk_server.sh`, `run_pi05_server.sh`, `run_pi05_server_agent.sh`, `run_verify_chunk.sh`, `serve_b1k_agent.py` |
| `src/agent/planner/src/` | `main.rs` (5 번) |

다시 뽑기(서브모듈 안에서):

```bash
git grep -n -E "/mnt/c/behavior-2026|wsl\.exe" -- 'src/*' 'tools/*' ':!*.md' ':!archive' \
  | grep -v -E ":[0-9]+:\s*(#|//|\"\"\"|\*)" | cut -d: -f1 | sort -u
```
