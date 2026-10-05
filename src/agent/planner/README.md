# agent — high-level planner agent + evaluator↔VLA relay (Rust)

The "judgement" layer of `plan.md` §1–2. It keeps the long plan, memory and step tracking, and gives VLA only the
instruction for the current step. Design, decisions, measurements and how to run: **[docs/에이전트_설계.md](../../../docs/에이전트_설계.md)** (Korean).

## Two paths

| Path | What | Files |
|---|---|---|
| Every step (relay) | Evaluator ↔ relay ↔ VLA server. Observation and action bytes pass through unchanged; the relay appends `__agent_prompt__` (and `__agent_flush__`) to the observation map. Masked WebSocket payloads are never unmasked (mask-key rotation), frames are cut-through streamed, sockets use `TCP_QUICKACK`. Odometry and boundary detection are plain arithmetic. **No LLM here.** | `relay.rs` `ws.rs` `msgpack.rs` `wire.rs` `session.rs` `odom.rs` `monitor.rs` |
| Step boundaries (agent) | Called only at episode start, budget exhaustion, periodic checks, base settling, gripper changes. OpenAI-compatible Chat Completions with tool calling (`tools` / `tool_calls` / `role:"tool"`) decides the next step. | `planner.rs` `tools.rs` `context.rs` `memory.rs` `plan.rs` `graph.rs` `llm.rs` |

The VLA server side must read `__agent_prompt__` as the prompt and remove the injected keys (the old π0.5 server hook was
removed 10-06; VLA = RecallVLA, robot-agent `training/vla`).

## Decision interface

`Agent = Core (state) + Decider`. `Core` owns the stage log, plan checklist, object memory (`back` / `the other`),
reference resolution, step budgets and instruction rendering. Deciders: `LlmDecider` (tool-calling loop) and
`PriorDecider` (no LLM, follows the demonstration prior). An RL decider plugs into the same trait;
`Core::decision_input()` is the structured state and is written to every boundary record.

## Files

| File | Role |
|---|---|
| `main.rs` | CLI `bagent`: `relay`, `sim`, `replay`, `build-assets`, `render`, `schedule`, `mock-llm`, `fake-pi`, `bench`, `bench-local`, `bench-image`, `llm-check`, `link`, `link-bench` |
| `relay.rs` | Relay: cut-through vs. hold, planner threads, ping handling while planning, latency stats |
| `ws.rs` | Hand-written RFC 6455: handshake (`/healthz`), frames, mask rotation, vectored writes, `poll(2)`, `TCP_QUICKACK` |
| `msgpack.rs` / `wire.rs` | Zero-copy scan of masked msgpack; observation keys, `base_qvel`, grippers, images, injected suffix |
| `session.rs` / `odom.rs` / `monitor.rs` | Per-environment odometry (same integration as `camera_poses()` in `src/scene_graph/scenemap/eval/demo_data.py`), five boundary triggers |
| `planner.rs` | `Core`, `Decider`, `LlmDecider`, `PriorDecider`, automatic evidence, deterministic fallback |
| `tools.rs` | Tool schemas and execution: `issue_command`, `continue_current`, `finish`, `graph_query`, `resolve_reference`, `look`, `robot_state`, `goal_status`, `set_plan`, `remember` |
| `context.rs` / `memory.rs` / `plan.rs` | Single system message context, recent-turn window + summaries after the decision, checklist |
| `graph.rs` | `SceneQuery` / `ScenemapGraph` (in-process scenemap, `--graph scenemap`), HTTP and static-file (`{"objects": [...]}`) graphs, object memory |
| `catalog.rs` / `bddl.rs` / `vocab.rs` / `instruction.rs` | Task cards (`assets/tasks.json`: prompts, limits, BDDL, top demo step orders, step budgets), 35-skill vocabulary, 4 instruction formats, VLA token budget |
| `llm.rs` / `http.rs` / `codec.rs` | Chat Completions (explicit deterministic sampling), hand-written HTTP, `curl` for HTTPS, replay/fake LLMs, local-server up/down hooks; base64, SHA-1 |
| `link.rs` | Evaluator link (`bagent link`): TCP transport to the in-evaluator glue (`src/sim/integ/glue/simlink_policy.py`), keyframe requests, stage tracking, `ObsSink` |
| `pose.rs` / `fk.rs` | Replaceable pose estimator (`PoseEstimator`, base_qvel integration + external correction), R1Pro camera forward kinematics from proprio (`src/sim/integ/fk/r1pro_cam_fk.json`) |
| `setup.rs` | Command-line → planner parts (catalog, LLM, graph, decider, agent factory); shared with `src/sim/integ/simlink` |
| `image.rs` / `util.rs` | Downscale + JPEG for LLM images; time, deterministic RNG, name cleanup, token estimate, CLI args |
| `trace.rs` / `replay.rs` | JSONL execution records, timeline, single-file HTML player, replay verification |
| `mockworld.rs` / `fakes.rs` | Fake world (fake executor), rule-based fake LLM (in-process and HTTP), fake VLA server, fake evaluator |
| `prompts/system.md` | System prompt (English) |
| `tests/e2e.rs` | End-to-end tests (relay byte identity, fake-world episodes, fallback, replay, HTTP LLM) |
| `tests/link.rs` | Evaluator-link end-to-end tests |

## Quick start (Linux)

`--llm kau` is Qwen3.5-9B through the KAU API of the AI agent class (`https://agent.kau.ac.kr/v1`); a local llama.cpp server is optional.

```bash
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/cargo-target/agent
cd src/agent/planner
cargo test --release                                  # 51 tests
cargo run --release -- sim --scenario trash --llm oracle
set -a; . ~/.config/behavior-2026/kau.env; set +a     # API key via environment only
cargo run --release -- sim --scenario radio --llm kau --no-images
cargo run --release -- relay --listen 0.0.0.0:8000 --upstream 127.0.0.1:8100 --mode agent --llm kau --graph scenemap   # scenemap only inside simlink; elsewhere falls back to no graph
```

Runs are written to `runs/` (git-ignored).
