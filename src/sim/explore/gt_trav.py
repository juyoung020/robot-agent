"""Ground-truth floor map of a BEHAVIOR scene for the explore skill (metrics + mock world), from the dataset layout images
(no simulator): floor_trav_0.png (traversable floor with objects, 0.01 m/px) restricted to the task's loaded rooms
(B100_task_misc.csv "Rooms to inlcude", same room-instance naming as omnigibson.maps.segmentation_map).

    python gt_trav.py --task bringing_water --out gt/            # -> gt/<scene>__<task>.pgm + .json

Output grid, WORLD frame, resolution 0.05 m: cell (x, y) covers world [ox + x*res, ox + (x+1)*res) x [oy + y*res, ...),
rows stored in increasing y (row 0 = smallest y). Values: 255 traversable floor in a loaded room, 0 otherwise.
Image pixel (row r, col c) is world (x = (c - S/2)*0.01, y = (r - S/2)*0.01) (omnigibson BaseMap.map_to_world).
"""
import argparse
import csv
import json
import pathlib

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(__import__("os").environ.get("B1K_ROOT", str(HERE.parents[2] / "third_party/BEHAVIOR-1K"))) / "datasets"


def scene_of(task: str) -> str:
    for p in (DATA / "2026-challenge-task-instances/scenes").glob(f"*/json/*_task_{task}_0_0_template.json"):
        return p.parts[-3]
    raise SystemExit(f"no scene for {task}")


def rooms_of(task: str) -> list[str]:
    with open(DATA / "2026-challenge-task-instances/metadata/B100_task_misc.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Task"] == task:
                return [r.strip() for r in row["Rooms to inlcude"].splitlines() if r.strip()]
    return []


def room_ids(scene_dir: pathlib.Path, ins: np.ndarray, sem: np.ndarray) -> dict[str, int]:
    """same naming as SegmentationMap._load_map (ins ids in increasing order per semantic class)"""
    cats = (DATA / "behavior-1k-assets/metadata/room_categories.txt").read_text().splitlines()
    out, per_sem = {}, {}
    for iid in np.unique(ins):
        if iid == 0:
            continue
        ys, xs = np.nonzero(ins == iid)
        sid = int(sem[ys[0], xs[0]])
        per_sem.setdefault(sid, []).append(int(iid))
    for sid, iids in per_sem.items():
        for i, iid in enumerate(iids):
            out[f"{cats[sid - 1].rstrip()}_{i}"] = iid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(HERE / "gt"))
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--exclude", default="garden_0", help="comma list of loaded rooms left out of the house reference")
    ap.add_argument("--robot-r", type=float, default=0.37, help="body radius for the reachable reference")
    ap.add_argument("--instance", type=int, default=301, help="public_test instance id for the start pose (robot_poses)")
    a = ap.parse_args()
    scene = scene_of(a.task)
    sd = DATA / "behavior-1k-assets/scenes" / scene / "layout"
    trav = np.array(Image.open(sd / "floor_trav_0.png")) > 0
    ins = np.array(Image.open(sd / "floor_insseg_0.png"))
    sem = np.array(Image.open(sd / "floor_semseg_0.png"))
    names = room_ids(sd.parent, ins, sem)
    ex = {r.strip() for r in a.exclude.split(",") if r.strip()}
    want = [r for r in rooms_of(a.task) if r not in ex]
    keep = [names[r] for r in want if r in names]
    inroom = np.isin(ins, keep)
    ok = trav & inroom
    S = ok.shape[0]
    k = int(round(a.res / 0.01))
    n = S // k
    g = ok[: n * k, : n * k].reshape(n, k, n, k).mean(axis=(1, 3)) >= 0.5
    # crop to the loaded rooms (+1 m)
    ys, xs = np.nonzero(g)
    pad = int(1.0 / a.res)
    y0, y1 = max(0, ys.min() - pad), min(n, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(n, xs.max() + pad + 1)
    g = g[y0:y1, x0:x1]
    ox = (x0 * k - S / 2) * 0.01
    oy = (y0 * k - S / 2) * 0.01
    # room id grid (same crop) for per-room coverage
    rid = ins[: n * k : k, : n * k : k][y0:y1, x0:x1]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{scene}__{a.task}"
    h, w = g.shape
    with open(f"{stem}.pgm", "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write((g.astype(np.uint8) * 255).tobytes())
    rooms = {r: int(names[r]) for r in want if r in names}
    meta = {"scene": scene, "task": a.task, "res": a.res, "origin": [ox, oy], "width": w, "height": h,
            "rows": "increasing y", "rooms": rooms, "missing_rooms": [r for r in want if r not in names],
            "floor_m2": float(g.sum() * a.res * a.res),
            "room_m2": {r: float(((rid == i) & g).sum() * a.res * a.res) for r, i in rooms.items()}}
    import math
    start = None
    for p in (DATA / "2026-challenge-task-instances/scene_test/public").glob(f"{scene}/json/*_task_{a.task}_instances/*_{a.instance}_template-tro_state.json"):
        rp = json.loads(p.read_text())["robot_poses"]["robot"][0]
        qx, qy, qz, qw = rp["orientation"]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        start = [rp["position"][0], rp["position"][1], yaw]
    meta["start_pose_world"] = start
    # 닿을 수 있는 기준(.reach.pgm): 닫힌 문을 막힌 것으로 본 바닥(floor_trav_no_door_0) 중, 시작 자리에서 몸통 반지름
    # a.robot_r 이상 떨어진 칸으로 이어진 곳 + 그 둘레 0.6 m(서서 볼 수 있는 바닥). 덮음 지표의 분모.
    if start is not None:
        from scipy import ndimage
        nd = np.array(Image.open(sd / "floor_trav_no_door_0.png")) > 0
        okd = nd & inroom
        gd = okd[: n * k, : n * k].reshape(n, k, n, k).mean(axis=(1, 3)) >= 0.5
        gd = gd[y0:y1, x0:x1] & g
        dist = ndimage.distance_transform_edt(gd) * a.res
        sx = int((start[0] - ox) / a.res); sy = int((start[1] - oy) / a.res)
        core = dist >= a.robot_r
        yy, xx = np.ogrid[: core.shape[0], : core.shape[1]]
        core |= ((yy - sy) ** 2 + (xx - sx) ** 2 <= (0.4 / a.res) ** 2) & gd
        lab, _ = ndimage.label(core)
        reach = lab == lab[sy, sx] if lab[sy, sx] > 0 else np.zeros_like(core)
        band = ndimage.binary_dilation(reach, iterations=int(0.6 / a.res)) & gd
        with open(f"{stem}.reach.pgm", "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write((band.astype(np.uint8) * 255).tobytes())
        meta["reach_m2"] = float(band.sum() * a.res * a.res)
        meta["reach_robot_r"] = a.robot_r
    meta["instance"] = a.instance
    meta["excluded"] = sorted(ex)
    pathlib.Path(f"{stem}.json").write_text(json.dumps(meta, indent=1))
    with open(f"{stem}.rooms.pgm", "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(np.where(g, rid, 0).astype(np.uint8).tobytes())
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
