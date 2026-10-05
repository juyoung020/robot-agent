"""Tests for the simulator adapter without Isaac Sim: ctypes binding, policy loop over a tiny fake plant,
script source and the TCP source (agent <-> sim line protocol). stdlib only.

    python3 test_move_robot_sim.py        (needs libmove_robot.so, see move_robot_sim.default_lib_path)
"""
import json
import math
import socket
import threading
import unittest

from move_robot_sim import ACTION_DIM, PROPRIO_DIM, MoveRobotLib, MoveRobotPolicy, ScriptSource, TcpSource, default_lib_path

KEY = "robot_r1::proprio"


class FakePlant:
    """First-order joints, smooth gripper, velocity base — enough to close the loop."""

    def __init__(self):
        self.p = [0.0] * PROPRIO_DIM
        self.p[53:57] = [1.025, -1.45, -0.47, 0.0]
        self.p[24:26] = [0.05, 0.05]
        self.p[49:51] = [0.05, 0.05]
        self.x = 0.0

    def obs(self):
        return {KEY: list(self.p)}

    def step(self, a):
        dt, k = 1 / 30, 1 - math.exp(-25 / 30)
        for qs, vs, acts in ((53, 57, 3), (3, 10, 7), (28, 35, 15)):
            n = 4 if qs == 53 else 7
            for j in range(n):
                old = self.p[qs + j]
                new = old + (a[acts + j] - old) * k
                self.p[qs + j], self.p[vs + j] = new, (new - old) / dt
        for qs, vs, ai in ((24, 26, 14), (49, 51, 22)):
            tgt = (a[ai] + 1) / 2 * 0.05
            old = self.p[qs]
            new = old + max(-0.25 * dt, min(0.25 * dt, tgt - old))
            self.p[qs:qs + 2] = [new, new]
            self.p[vs:vs + 2] = [(new - old) / dt] * 2
        self.p[0] = a[0] * 0.75
        self.x += self.p[0] * dt


def run(policy, plant, steps):
    for _ in range(steps):
        a = policy.act(plant.obs())
        a = a.tolist() if hasattr(a, "tolist") else a
        assert len(a) == ACTION_DIM
        plant.step(a)


@unittest.skipUnless(default_lib_path().exists(), "libmove_robot.so not built")
class T(unittest.TestCase):
    def test_definition(self):
        lib = MoveRobotLib()
        d = lib.definition()
        self.assertEqual(d["function"]["name"], "move_robot")
        lib.close()

    def test_hold_layout(self):
        lib = MoveRobotLib()
        plant = FakePlant()
        self.assertEqual(lib.tick(plant.p), 0)
        a = lib.action()
        self.assertEqual(a[0:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(a[3], 1.025, places=5)
        self.assertEqual(a[14], 1.0)
        self.assertEqual(a[22], 1.0)
        self.assertEqual(lib.tick([0.0] * 5), -1)
        lib.close()

    def test_script(self):
        src = ScriptSource([
            {"part": "torso", "mode": "delta", "values": [0, 0, 0, 0]},
            {"part": "right_arm", "mode": "delta", "values": [-20, 0, 0, 0, 0, 0, 0]},
            {"part": "left_gripper", "mode": "absolute", "values": [0.2]},
            {"part": "base", "mode": "delta", "values": [0.4, 0, 0]},
            {"part": "head", "mode": "delta", "values": [1]},
        ])
        pol, plant = MoveRobotPolicy(src), FakePlant()
        run(pol, plant, 600)
        st = [r["status"] for r in src.results]
        self.assertEqual(st, ["reached"] * 4 + ["error"], src.results)
        self.assertAlmostEqual(src.results[1]["state"][0], -20.0, delta=1.6)
        self.assertAlmostEqual(plant.x, 0.4, delta=0.05)
        pol.close()

    def test_tcp(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        src = TcpSource("127.0.0.1", port)
        pol, plant = MoveRobotPolicy(src), FakePlant()
        out = {}

        def client():
            c = socket.create_connection(("127.0.0.1", port), timeout=10)
            c.sendall(b'{"part":"left_arm","mode":"absolute","values":[0,20,0,-30,0,0,0]}\n')
            out["r"] = json.loads(c.makefile().readline())
            c.close()

        th = threading.Thread(target=client)
        th.start()
        for _ in range(400):
            run(pol, plant, 1)
            if "r" in out:
                break
        th.join(5)
        self.assertEqual(out["r"]["status"], "reached", out)
        self.assertAlmostEqual(out["r"]["state"][3], -30.0, delta=1.6)
        pol.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
