"""Walls as 2D: occupancy map -> wall segments -> robot-frame numbers (the "wall state").

Pure numpy (no viser / spark_dsg), so the viewer, tests and a dump script share it.
This is a Python development tool. The training/inference path must use the native
version (C++ on `sm_grid`, see docs/MAP_STATE_PLAN.md); keep the layout below identical.

Map convention (ROS map_server, `map.pgm` + `map.yaml`): PGM row 0 = max y, cell (ix, iy) covers
x in [ox + ix*res, ox + (ix+1)*res), y in [oy + (H-1-iy)*res, oy + (H-iy)*res).
Occupied = grey level <= OCC_MAX (0 = occupied; 205 = unknown; 254 = free).

Wall state vector (float32, length N_SECTORS + K_SEGMENTS * 5), robot frame (x forward, y left):
  [0 : N_SECTORS]              distance to the first occupied cell along sector i, divided by MAX_RANGE.
                               Sector 0 = straight ahead, counter-clockwise, 2*pi/N_SECTORS apart. 1.0 = nothing within range.
  [N_SECTORS + 5*j : +5]       j-th nearest wall segment: ax, ay, bx, by (metres / MAX_RANGE, clipped to [-1, 1]), valid (0/1).
                               Sorted by distance from the robot; unused slots are all zero.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

OCC_MAX = 90            # trinary map_saver: occupied_thresh 0.65 -> grey <= 255 * (1 - 0.65) ~ 89
MIN_LEN = 0.5           # shortest wall segment [m]
MAX_THICK = 0.5         # thicker blobs are furniture/clutter, not walls [m]
N_SECTORS = 16
K_SEGMENTS = 8
MAX_RANGE = 4.0         # [m]


def occupied_mask(pgm: np.ndarray, occ_max: int = OCC_MAX) -> np.ndarray:
    return np.asarray(pgm) <= occ_max


def read_map_yaml(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = (x.strip() for x in line.split(":", 1))
            if v.startswith("["):
                out[k] = [float(x) for x in v.strip("[]").split(",") if x.strip()]
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _runs(line: np.ndarray):
    d = np.diff(np.concatenate(([0], line.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _axis_groups(grid: np.ndarray, min_run: int, overlap: float):
    """Group horizontal runs of consecutive rows (thick walls span several rows)."""
    done, active = [], []
    for r in range(grid.shape[0]):
        runs = [(a, b) for a, b in _runs(grid[r]) if b - a + 1 >= min_run]
        nxt, used = [], set()
        for a, b in runs:
            hit = None
            for gi, g in enumerate(active):
                if gi in used or g["r1"] != r - 1:
                    continue
                ov = min(b, g["x1"]) - max(a, g["x0"]) + 1
                if ov >= overlap * min(b - a + 1, g["x1"] - g["x0"] + 1):
                    hit = gi
                    break
            if hit is None:
                nxt.append({"r0": r, "r1": r, "x0": a, "x1": b})
            else:
                g = active[hit]
                used.add(hit)
                g["r1"] = r
                g["x0"] = min(g["x0"], a)
                g["x1"] = max(g["x1"], b)
                nxt.append(g)
        done.extend(g for gi, g in enumerate(active) if gi not in used)
        active = nxt
    return done + active


def wall_segments(occ: np.ndarray, res: float, origin, min_len: float = MIN_LEN,
                  max_thick: float = MAX_THICK, overlap: float = 0.6) -> np.ndarray:
    """Axis-aligned wall centre lines from the occupied cells. Returns (N, 4) = ax, ay, bx, by in the map frame [m].

    Tilted walls are not detected (indoor Manhattan layouts only)."""
    occ = np.asarray(occ, bool)
    H, W = occ.shape
    ox, oy = float(origin[0]), float(origin[1])
    min_run = max(1, int(round(min_len / res)))
    out = []
    for transpose in (False, True):
        grid = occ.T if transpose else occ
        for g in _axis_groups(grid, min_run, overlap):
            if (g["r1"] - g["r0"] + 1) * res > max_thick:
                continue
            rc = (g["r0"] + g["r1"]) / 2.0
            pts = []
            for c in (g["x0"], g["x1"]):
                # grid (row=rc, col=c) -> original (iy, ix)
                iy, ix = (c, rc) if transpose else (rc, c)
                pts.append((ox + (ix + 0.5) * res, oy + (H - 1 - iy + 0.5) * res))
            out.append([pts[0][0], pts[0][1], pts[1][0], pts[1][1]])
    return np.asarray(out, np.float64).reshape(-1, 4)


def ray_distances(occ: np.ndarray, res: float, origin, pose, n: int = N_SECTORS,
                  max_range: float = MAX_RANGE) -> np.ndarray:
    """Distance [m] to the first occupied cell along n rays (sector 0 = ahead, counter-clockwise)."""
    occ = np.asarray(occ, bool)
    H, W = occ.shape
    ox, oy = float(origin[0]), float(origin[1])
    x, y, yaw = (float(v) for v in pose[:3])
    ts = np.arange(res * 0.5, max_range + res * 0.5, res * 0.5)
    out = np.full(n, max_range, np.float32)
    for i in range(n):
        th = yaw + 2.0 * math.pi * i / n
        ix = np.floor((x + ts * math.cos(th) - ox) / res).astype(int)
        iy = (H - 1 - np.floor((y + ts * math.sin(th) - oy) / res)).astype(int)
        inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        hit = np.zeros(len(ts), bool)
        hit[inb] = occ[iy[inb], ix[inb]]
        if hit.any():
            out[i] = ts[int(np.argmax(hit))]
    return out


def segments_robot_frame(segs: np.ndarray, pose, k: int = K_SEGMENTS) -> Tuple[np.ndarray, np.ndarray]:
    """k nearest segments in the robot frame. Returns ((k, 4) metres, distances (k,))."""
    out = np.zeros((k, 4), np.float32)
    dist = np.full(k, np.inf, np.float32)
    if len(segs) == 0:
        return out, dist
    x, y, yaw = (float(v) for v in pose[:3])
    c, s = math.cos(-yaw), math.sin(-yaw)

    def tf(p):
        d = p - np.array([x, y])
        return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)

    a, b = tf(segs[:, 0:2]), tf(segs[:, 2:4])
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(a + ab * t[:, None], axis=1)
    order = np.argsort(d)[:k]
    m = len(order)
    out[:m, 0:2], out[:m, 2:4], dist[:m] = a[order], b[order], d[order]
    return out, dist


def wall_state_vector(occ: np.ndarray, res: float, origin, segs: np.ndarray, pose,
                      n: int = N_SECTORS, k: int = K_SEGMENTS, max_range: float = MAX_RANGE) -> np.ndarray:
    rays = ray_distances(occ, res, origin, pose, n, max_range) / max_range
    seg, dist = segments_robot_frame(segs, pose, k)
    valid = np.isfinite(dist) & (dist <= max_range * 1.5)
    seg = np.clip(seg / max_range, -1.0, 1.0) * valid[:, None]
    block = np.concatenate([seg, valid[:, None].astype(np.float32)], axis=1)
    return np.concatenate([rays.astype(np.float32), block.reshape(-1).astype(np.float32)])


def load_memory_map(memory_dir: str, grid_name: Optional[str] = None):
    """(occupied mask, resolution, origin, robot_pose or None) of a memory folder."""
    from PIL import Image

    pgm_path = os.path.join(memory_dir, grid_name or "map.pgm")
    meta = read_map_yaml(os.path.splitext(pgm_path)[0] + ".yaml")
    occ = occupied_mask(np.array(Image.open(pgm_path).convert("L")))
    pose = None
    try:
        with open(os.path.join(memory_dir, "scene.json")) as f:
            rp = json.load(f).get("metadata", {}).get("robot_pose")
        if isinstance(rp, list) and len(rp) >= 3:
            pose = tuple(float(v) for v in rp[:3])
    except (OSError, ValueError):
        pass
    return occ, float(meta.get("resolution", 0.05)), (meta.get("origin") or [0.0, 0.0])[:2], pose


def state_from_memory(memory_dir: str, pose=None) -> Dict[str, Any]:
    occ, res, origin, rpose = load_memory_map(memory_dir)
    pose = tuple(pose) if pose is not None else rpose
    if pose is None:
        raise ValueError("no robot pose: pass pose=(x, y, yaw)")
    segs = wall_segments(occ, res, origin)
    vec = wall_state_vector(occ, res, origin, segs, pose)
    return {"memory_dir": memory_dir, "pose": list(pose), "resolution": res, "origin": list(origin),
            "n_sectors": N_SECTORS, "k_segments": K_SEGMENTS, "max_range": MAX_RANGE,
            "segments_map_frame": segs.tolist(), "state_vector": vec.tolist()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the wall state of a memory folder.")
    ap.add_argument("memory_dir")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    ap.add_argument("--out", default=None, help="write JSON here")
    a = ap.parse_args()
    st = state_from_memory(a.memory_dir, a.pose)
    print(f"pose {st['pose']}  segments {len(st['segments_map_frame'])}  vector length {len(st['state_vector'])}")
    print("rays/max_range:", np.round(st["state_vector"][:N_SECTORS], 3).tolist())
    if a.out:
        with open(a.out, "w") as f:
            json.dump(st, f)
