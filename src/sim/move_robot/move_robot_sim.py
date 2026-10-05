"""Simulator side of the `move_robot` LLM tool: a policy that turns tool calls into R1Pro actions inside the evaluator.

The closed-loop executor (schema check, joint limits, safe speeds, min-jerk interpolation, base odometry control,
reached/blocked/timeout) is the agent's Rust crate `robot-agent/src/agent/tools/move_robot`, built as
`libmove_robot.so` and called here through ctypes (stdlib only). This file only moves bytes:

    every step : obs[*::proprio] (61 f32) -> mr_tick -> action (23 f32) -> evaluator
    tool call  : one JSON line from a client (TCP --listen) or from a script -> mr_command
    result     : JSON line back to the client (and to the log)

Action layout = omnigibson/eval/utils/eval_utils.py ACTION_QPOS_INDICES["R1Pro"]
(base vx,vy,wz [-1,1] | torso 4 | left arm 7 | left gripper | right arm 7 | right gripper), controllers from
omnigibson/eval/r1pro.yaml. Idle = hold the last targets (base 0 velocity).

No numpy/torch needed (torch is used for the return value only when importable).
"""
import ctypes
import json
import os
import pathlib
import socket
import time

HERE = pathlib.Path(__file__).resolve().parent
ACTION_DIM = 23
PROPRIO_DIM = 61


def default_lib_path() -> pathlib.Path:
    env = os.environ.get("MOVE_ROBOT_LIB")
    if env:
        return pathlib.Path(env)
    # robot-agent/src/sim/move_robot -> robot-agent/build/bin (tools/build_all.sh move_robot)/tools/move_robot
    return HERE.parents[2] / "build/bin/libmove_robot.so"


class MoveRobotLib:
    """ctypes binding of include/move_robot.h"""

    def __init__(self, path=None, hz: float = 30.0):
        path = pathlib.Path(path) if path else default_lib_path()
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: cargo build --release in robot-agent/src/agent/tools/move_robot "
                                    f"(or set MOVE_ROBOT_LIB)")
        L = ctypes.CDLL(str(path))
        fp = ctypes.POINTER(ctypes.c_float)
        L.mr_new.restype = ctypes.c_void_p
        L.mr_new.argtypes = [ctypes.c_double]
        L.mr_free.argtypes = [ctypes.c_void_p]
        L.mr_tick.restype = ctypes.c_int
        L.mr_tick.argtypes = [ctypes.c_void_p, fp, ctypes.c_size_t, fp]
        L.mr_command.restype = ctypes.c_int
        L.mr_command.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.mr_busy.restype = ctypes.c_int
        L.mr_busy.argtypes = [ctypes.c_void_p]
        L.mr_reset.argtypes = [ctypes.c_void_p]
        # GT pose input (Map_Vla): older libmove_robot.so builds do not export it
        self.has_gt_pose = hasattr(L, "mr_set_gt_pose")
        if self.has_gt_pose:
            L.mr_set_gt_pose.restype = ctypes.c_int
            L.mr_set_gt_pose.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        # VLA executor (LIMO + OMX-F, robot-agent docs/map_vla/POLICY.md 1.3): older builds do not export it
        self.has_vla = hasattr(L, "mr_vla_start")
        if self.has_vla:
            L.mr_vla_start.restype = ctypes.c_int
            L.mr_vla_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            L.mr_vla_tick.restype = ctypes.c_int
            L.mr_vla_tick.argtypes = [ctypes.c_void_p, fp, ctypes.c_size_t, fp]
            L.mr_vla_set_objects.restype = ctypes.c_int
            L.mr_vla_set_objects.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
            L.mr_vla_set_objects_json.restype = ctypes.c_int
            L.mr_vla_set_objects_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
            L.mr_vla_contacts.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
            L.mr_vla_busy.restype = ctypes.c_int
            L.mr_vla_busy.argtypes = [ctypes.c_void_p]
            L.mr_vla_stop.restype = ctypes.c_int
            L.mr_vla_stop.argtypes = [ctypes.c_void_p]
            self.vprop = (ctypes.c_float * 64)()
            self.vout = (ctypes.c_float * 8)()
        L.mr_take_result.restype = ctypes.c_ssize_t
        L.mr_take_result.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        L.mr_tool_definition.restype = ctypes.c_char_p
        self.L = L
        self.h = L.mr_new(hz)
        self.prop = (ctypes.c_float * PROPRIO_DIM)()
        self.act = (ctypes.c_float * ACTION_DIM)()
        self.buf = ctypes.create_string_buffer(4096)

    def tick(self, proprio) -> int:
        """0 idle, 1 moving, 2 finished this step, -1 bad proprio"""
        n = min(len(proprio), PROPRIO_DIM)
        if hasattr(proprio, "tobytes") and getattr(proprio, "dtype", None) is not None and str(proprio.dtype) == "float32" and n == PROPRIO_DIM:
            ctypes.memmove(self.prop, proprio.tobytes(), 4 * PROPRIO_DIM)
        else:
            for i in range(n):
                self.prop[i] = float(proprio[i])
        return self.L.mr_tick(self.h, self.prop, n, self.act)

    def vla_start(self, call) -> int:
        """0 started, 1 finished at once (error / handback: take_result)"""
        s = call if isinstance(call, str) else json.dumps(call)
        return self.L.mr_vla_start(self.h, s.encode("utf-8"))

    def vla_tick(self, proprio):
        """LIMO proprio (24) -> (rc, filtered action 8 [vx, wz, j1..j5, gripper 0..1]); rc 0 idle, 1 running, 2 finished"""
        n = min(len(proprio), 64)
        for i in range(n):
            self.vprop[i] = float(proprio[i])
        rc = self.L.mr_vla_tick(self.h, self.vprop, n, self.vout)
        return rc, list(self.vout)

    def vla_busy(self) -> bool:
        return bool(self.L.mr_vla_busy(self.h))

    def set_gt_pose(self, x: float, y: float, yaw: float) -> int:
        """GT base pose in the map frame for the NEXT tick only (replaces base_qvel integration there). -1 if the lib is old."""
        return self.L.mr_set_gt_pose(self.h, x, y, yaw) if self.has_gt_pose else -1

    def command(self, args) -> int:
        """0 started, 1 finished at once (read or error: take_result)"""
        s = args if isinstance(args, str) else json.dumps(args)
        return self.L.mr_command(self.h, s.encode("utf-8"))

    def busy(self) -> bool:
        return bool(self.L.mr_busy(self.h))

    def reset(self):
        self.L.mr_reset(self.h)

    def take_result(self):
        n = self.L.mr_take_result(self.h, self.buf, len(self.buf))
        if n < 0:
            self.buf = ctypes.create_string_buffer(-n)
            n = self.L.mr_take_result(self.h, self.buf, len(self.buf))
        return json.loads(self.buf.value.decode("utf-8")) if n > 0 else None

    def definition(self) -> dict:
        return json.loads(self.L.mr_tool_definition().decode("utf-8"))

    def action(self) -> list:
        return list(self.act)

    def close(self):
        if self.h:
            self.L.mr_free(self.h)
            self.h = None


class ScriptSource:
    """Tool calls from a list / JSONL file (one args object per line, '#' comments). For headless checks."""

    def __init__(self, calls):
        if isinstance(calls, (str, pathlib.Path)):
            lines = pathlib.Path(calls).read_text(encoding="utf-8").splitlines()
            calls = [json.loads(l) for l in lines if l.strip() and not l.lstrip().startswith("#")]
        self.calls = list(calls)
        self.results = []

    def poll(self):
        return self.calls.pop(0) if self.calls else None

    def reply(self, result):
        self.results.append(result)

    @property
    def done(self):
        return not self.calls

    def close(self):
        pass


class TcpSource:
    """Line-delimited JSON over TCP: client sends tool args, gets the result line, connection closes.
    Non-blocking: polled once per sim step, never stalls the simulator."""

    def __init__(self, host="127.0.0.1", port=8771):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(4)
        self.srv.setblocking(False)
        self.conn = None
        self.rx = b""
        self.waiting = False  # a call from self.conn is executing
        print(f"[move_robot] listening on {host}:{port}", flush=True)

    def poll(self):
        if self.conn is None:
            try:
                self.conn, _ = self.srv.accept()
                self.conn.setblocking(False)
                self.rx = b""
            except (BlockingIOError, InterruptedError):
                return None
        if self.waiting:
            return None
        try:
            chunk = self.conn.recv(65536)
            if not chunk:
                self._drop()
                return None
            self.rx += chunk
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self._drop()
            return None
        if b"\n" not in self.rx:
            return None
        line, self.rx = self.rx.split(b"\n", 1)
        self.waiting = True
        try:
            return json.loads(line.decode("utf-8"))
        except ValueError:
            return line.decode("utf-8", "replace")  # the executor turns it into an error observation

    def reply(self, result):
        if self.conn is not None:
            try:
                self.conn.setblocking(True)
                self.conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
            except OSError:
                pass
            self._drop()
        self.waiting = False

    def _drop(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = None
        self.waiting = False

    done = False

    def close(self):
        if self.conn is not None:
            self._drop()
        self.srv.close()


class MoveRobotPolicy:
    """Evaluator policy (act(obs) -> (23,) action) driven by move_robot tool calls."""

    def __init__(self, source, lib_path=None, log_path=None, hz=30.0):
        self.lib = MoveRobotLib(lib_path, hz)
        self.src = source
        self.prop_key = None
        self.step = 0
        self.t_call = None
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None

    def _proprio(self, obs):
        if self.prop_key is None:
            found = [k for k in obs if str(k).endswith("::proprio")]
            if len(found) != 1:
                raise KeyError(f"no unique '*::proprio' key in obs: {sorted(map(str, obs))}")
            self.prop_key = found[0]
        p = obs[self.prop_key]
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        batched = hasattr(p, "ndim") and p.ndim == 2 or (isinstance(p, (list, tuple)) and p and isinstance(p[0], (list, tuple)))
        row = p[0] if batched else p
        if hasattr(row, "astype"):
            row = row.astype("float32", copy=False)
        return row, batched

    def _finish(self, result):
        if result is None:
            return
        result = dict(result)
        if self.t_call is not None:
            result["wall_s"] = round(time.perf_counter() - self.t_call, 2)
        self.src.reply(result)
        print(f"[move_robot] step {self.step}: {json.dumps(result)}", flush=True)
        if self.log:
            self.log.write(json.dumps({"step": self.step, "result": result}) + "\n")
            self.log.flush()
        self.t_call = None

    def act(self, obs):
        row, batched = self._proprio(obs)
        rc = self.lib.tick(row)
        if rc == 2:
            self._finish(self.lib.take_result())
        if not self.lib.busy():
            call = self.src.poll()
            if call is not None:
                self.t_call = time.perf_counter()
                if self.log:
                    self.log.write(json.dumps({"step": self.step, "call": call}) + "\n")
                if self.lib.command(call) == 1:
                    self._finish(self.lib.take_result())
        self.step += 1
        a = self.lib.action()
        try:
            import torch

            t = torch.tensor(a, dtype=torch.float32)
            return t[None] if batched else t
        except ImportError:
            return [a] if batched else a

    def reset(self):
        self.lib.reset()
        self._finish(self.lib.take_result())

    def close(self):
        self.src.close()
        self.lib.close()
        if self.log:
            self.log.close()
