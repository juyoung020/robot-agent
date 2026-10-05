"""Launcher: official evaluator (omnigibson.eval.eval, unmodified) with --policy local, whose LocalPolicy gets
MoveRobotPolicy (move_robot tool calls -> R1Pro actions); the evaluator is patched at
runtime (only LocalPolicy.__init__ is replaced, BEHAVIOR-1K unmodified).

    # agent connects (robot-agent: move-robot call '<args>' / move-robot llm "..." --addr 127.0.0.1:8771)
    python run_eval_move.py --listen 127.0.0.1:8771 -- --task-name turning_on_radio --max-steps 6000
    # headless check from a script of tool calls (JSONL), then hold until --max-steps
    python run_eval_move.py --script calls.jsonl --log out.jsonl -- --task-name turning_on_radio --max-steps 900

Build the executor first: cd robot-agent/src/agent/tools/move_robot && cargo build --release (or --lib / MOVE_ROBOT_LIB).
"""
import argparse
import json
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--listen", help="host:port for tool calls (line-delimited JSON)")
    g.add_argument("--script", help="JSONL file of tool-call arguments, run in order")
    ap.add_argument("--lib", default=None, help="libmove_robot.so (default: MOVE_ROBOT_LIB or the superproject build)")
    ap.add_argument("--log", default=None, help="append calls/results here (JSONL)")
    ap.add_argument("--scene-out", default=None,
                    help="object memory on (src/scene_graph/runtime, libsgrt): YOLO -> scenemap map/objects/graph -> files in this "
                         "dir; needs --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper. SGRT_POSE=gt|slam|odom picks the pose")
    args = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--policy" in eval_args:
        i = eval_args.index("--policy")
        del eval_args[i:i + 2]

    from move_robot_sim import MoveRobotPolicy, ScriptSource, TcpSource
    import omnigibson.eval.policies as P

    if args.listen:
        host, port = args.listen.rsplit(":", 1)
        src = TcpSource(host, int(port))
    else:
        src = ScriptSource(args.script)

    orig_init = P.LocalPolicy.__init__
    holder = {}

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.policy = MoveRobotPolicy(src, lib_path=args.lib, log_path=args.log)
        holder["p"] = self.policy
        if args.scene_out:
            sys.path.insert(0, str(HERE.parents[1] / "scene_graph" / "runtime" / "glue"))
            from sgrt_glue import SceneMemory

            task = eval_args[eval_args.index("--task-name") + 1]
            mem = SceneMemory(task, args.scene_out)
            holder["mem"] = mem
            act = self.policy.act

            def act_with_memory(obs, _act=act, _mem=mem):
                _mem.step(obs)
                return _act(obs)

            self.policy.act = act_with_memory

    P.LocalPolicy.__init__ = init
    sys.argv = ["omnigibson.eval.eval", *eval_args, "--policy", "local"]
    try:
        runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)
    finally:
        if isinstance(src, ScriptSource):
            print(f"[move_robot] results: {json.dumps(src.results)}", flush=True)
            if src.calls:
                print(f"[move_robot] {len(src.calls)} call(s) not run (episode ended first)", flush=True)
        if "p" in holder:
            holder["p"].close()
        if "mem" in holder:
            print(f"[sgrt] {holder['mem'].stats()}", flush=True)
            holder["mem"].close()


if __name__ == "__main__":
    main()
