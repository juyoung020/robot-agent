"""Score a scenemap memory dir against the simulator ground truth (world frame).

    python score_map_gt.py <memory_dir> --gt-map <gt_trav .pgm> [--gt-objects gt.csv.objects.json] [--slam-start x y yaw]

Map (needs src/sim/explore/gt_trav.py output: 255 = traversable floor in a loaded room, world frame, rows = +y):
  free precision  = our free cells (0..49) that are GT traversable
  occupied near   = our occupied cells (>= 65) within 10 cm of a GT non-traversable cell (wall / furniture)
  wall offset     = median distance from our occupied cells to the nearest GT non-traversable cell
Objects (needs the glue's SGRT_GT_LOG objects dump): distance from each object centre to the nearest GT object AABB
  (any category; COCO names differ from BEHAVIOR categories), median / 90th percentile.
A SLAM-mode map lives in the episode-start frame: pass the GT start base pose (--slam-start) to put it in world.
"""
import argparse
import json
import math
import pathlib

import numpy as np


def read_pgm(p):
    b = pathlib.Path(p).read_bytes()
    parts, i = [], 0
    while len(parts) < 4:
        while b[i:i + 1].isspace():
            i += 1
        if b[i:i + 1] == b"#":
            while b[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        parts.append(b[i:j])
        i = j
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)


def edt_bool(mask, res):
    """distance [m] from every cell to the nearest True cell (brute-force by dilation rings, capped at 1 m)"""
    from collections import deque
    h, w = mask.shape
    d = np.full((h, w), np.inf)
    q = deque()
    ys, xs = np.nonzero(mask)
    d[ys, xs] = 0
    q.extend(zip(ys, xs))
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while q:
        y, x = q.popleft()
        for dy, dx in nb:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                nd = d[y, x] + (1.0 if dy == 0 or dx == 0 else 1.4142)
                if nd < d[yy, xx] and nd * res <= 1.0:
                    d[yy, xx] = nd
                    q.append((yy, xx))
    return d * res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mem")
    ap.add_argument("--gt-map", required=True)
    ap.add_argument("--gt-objects")
    ap.add_argument("--slam-start", nargs=3, type=float, default=None)
    a = ap.parse_args()
    mem = pathlib.Path(a.mem)
    view = json.loads((mem / "view.json").read_text())
    g = view["grid"]
    pgm = read_pgm(mem / "map.pgm")[::-1]  # rows -> increasing y
    # map.pgm values: 0 occupied, 254 free-ish, 205 unknown (scenemap pgm()) -> classes
    occ = pgm == 0
    free = (pgm == 254) | ((pgm >= 130) & (pgm <= 188))  # v <= 49
    gj = json.loads(pathlib.Path(a.gt_map).with_suffix(".json").read_text())
    gt = read_pgm(a.gt_map)  # rows increasing y already (gt_trav writes row 0 = smallest y)
    gres, gox, goy = gj["res"], gj["origin"][0], gj["origin"][1]
    nontrav = gt != 255
    dist_nt = edt_bool(nontrav, gres)

    def to_world(x, y):
        if a.slam_start is None:
            return x, y
        sx, sy, th = a.slam_start
        c, s = math.cos(th), math.sin(th)
        return sx + c * x - s * y, sy + s * x + c * y

    def gt_cell(wx, wy):
        return int(math.floor((wx - gox) / gres)), int(math.floor((wy - goy) / gres))

    res, ox, oy = g["resolution"], g["origin"][0], g["origin"][1]
    n_free = n_free_ok = 0
    occ_d = []
    for (cls, arr) in (("free", free), ("occ", occ)):
        ys, xs = np.nonzero(arr)
        for y, x in zip(ys, xs):
            wx, wy = to_world(ox + (x + 0.5) * res, oy + (y + 0.5) * res)
            gx, gy = gt_cell(wx, wy)
            inside = 0 <= gx < gt.shape[1] and 0 <= gy < gt.shape[0]
            if cls == "free":
                n_free += 1
                n_free_ok += inside and gt[gy, gx] == 255
            else:
                occ_d.append(dist_nt[gy, gx] if inside else 0.0)
    occ_d = np.array(occ_d)
    out = {"free_cells": n_free, "free_precision": n_free_ok / max(1, n_free), "occ_cells": int(occ.sum()),
           "occ_within_10cm": float((occ_d <= 0.10).mean()) if len(occ_d) else None,
           "occ_dist_median_m": float(np.median(occ_d)) if len(occ_d) else None}
    if a.gt_objects:
        gobj = json.loads(pathlib.Path(a.gt_objects).read_text())
        gobj = [o for o in gobj if o.get("category") not in ("floors", "walls", "ceilings", "agent", "robot")]
        ds = []
        for o in view["objects"]:
            if o.get("state") == "gone":
                continue
            px, py = to_world(o["pos"][0], o["pos"][1])
            p = np.array([px, py, o["pos"][2]])
            best = 1e9
            for q in gobj:
                lo, hi = np.array(q["lo"]), np.array(q["hi"])
                d = np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi)))
                best = min(best, d)
            ds.append(best)
        ds = np.array(ds)
        out.update({"objects": len(ds), "obj_to_gt_aabb_median_m": float(np.median(ds)) if len(ds) else None,
                    "obj_to_gt_aabb_p90_m": float(np.percentile(ds, 90)) if len(ds) else None,
                    "obj_within_10cm": float((ds <= 0.10).mean()) if len(ds) else None})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
