#!/usr/bin/env python3
"""LEGACY Spark-DSG viewer (Python + viser) for the robot's object memory. NOT real-time.

    *** Do not use. The real viewer is sgview (Rust server + three.js):
    ***   src/scene_graph/sgview  — robot-agent: tools/run_sgview.sh <memory_dir> [--live]
    ***   live run + viewer       — robot-agent: tools/run_explore_live.sh (LIMO: SGRT_ROBOT=limo_omx)
    *** This file only polls scene.json; sgview gets sgrt's socket stream at 60 Hz+.

    ~/sdsg_venv/bin/python src/scene_graph/viewer/sgviz.py <memory_dir> [--port 8080]

<memory_dir> is the directory the memory runtime rewrites every ~1 s:
scene.json (Spark-DSG DynamicSceneGraph), map.pgm + map.yaml, objects/*.png.

Built on spark_dsg.viser.ViserRenderer (server ownership / clearing) and the
spark_dsg python API (graph loading, bounding-box corners and edge indices).
The stock GraphHandle redraws the whole graph per call (and uses viser APIs
that are gone in viser 1.x), so the per-node incremental drawing, map, robot
and metadata panel are done here on top of the renderer's server.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LEGACY_WARNING = """
################################################################################
#  sgviz.py is the OLD Python viewer (Spark-DSG + viser). LEGACY, NOT REAL-TIME #
#  (it only polls scene.json files). Kept for history only.                    #
#                                                                              #
#  Use sgview (Rust + three.js) instead:                                       #
#    robot-agent/tools/run_sgview.sh <memory_dir> [--live]                     #
#    robot-agent/tools/run_explore_live.sh   (LIMO: SGRT_ROBOT=limo_omx)       #
#    docs: src/scene_graph/sgview/README.md                                    #
#  옛 Python 뷰어(실시간 아님). 지금 뷰어는 sgview(Rust).                      #
################################################################################
"""
if __name__ == "__main__":  # loud, before the heavy imports below can fail
    print(LEGACY_WARNING, file=sys.stderr, flush=True)

import numpy as np

# `python sgviz.py` puts this folder on sys.path; nothing here may shadow
# `viser` or `spark_dsg` (Spark-DSG's own python dir contains a viser.py).
import spark_dsg as dsg
import walls2d  # occupancy map -> 2D wall segments -> robot-frame numbers (this folder)
from spark_dsg.viser import BOUNDING_BOX_EDGE_INDICES, ViserRenderer

STATE_COLORS = {
    "seen": (46, 160, 67),
    "moved": (245, 140, 20),
    "held": (40, 110, 230),
    "gone": (140, 140, 140),
}
DEFAULT_COLOR = (200, 60, 200)
WALL_LINE_COLOR = (0, 200, 255)  # cyan: orange is already the "moved" colour (trails)
# Ceiling / wall detection by head noun (names are open-vocabulary and noisy): "wall mounted tv" is NOT a wall.
CEILING_RE = re.compile(r"(?:^|\s)(?:ceilings?|roofs?)$", re.I)
WALL_RE = re.compile(r"(?:^|\s)(?:walls?|baseboards?|wainscoting|drywall)$", re.I)


# --------------------------------------------------------------------------
# Parsing (headless, testable)
# --------------------------------------------------------------------------


@dataclass
class ObjInfo:
    id: int
    sym: str
    name: str
    pos: Tuple[float, float, float]
    state: str = "seen"
    n_obs: int = 0
    score: float = 0.0
    first_pos: Optional[Tuple[float, float, float]] = None
    last_seen: float = 0.0  # seconds (from last_update_time_ns)
    is_active: bool = True
    structural: bool = False
    handled: bool = False
    bbox_dims: Optional[Tuple[float, float, float]] = None
    bbox_center: Optional[Tuple[float, float, float]] = None
    bbox_corners: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    rgbd: Optional[Dict[str, Any]] = None
    room: str = ""  # parent room (ROOMS layer -> OBJECTS layer edge), "" = none
    points: Optional[Dict[str, Any]] = None  # {"path", "n", "voxel", "stamp"}
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def names(self) -> Dict[str, Any]:
        """SigLIP 2 names cached by the runtime (scene.json metadata.names: en, ko, score, general, top, table)."""
        n = self.metadata.get("names")
        return n if isinstance(n, dict) else {}

    @property
    def display_name(self) -> str:
        """Detector name, or the embedding name when the segmenter is class-agnostic (FastSAM: "object")."""
        en = self.names.get("en")
        if en and self.name in ("", "object"):
            return en
        return self.name

    @property
    def label(self) -> str:
        return f"{self.display_name}#{self.id}"

    def draw_sig(self):
        """Fields that affect the 3D drawing."""
        return (self.display_name, self.pos, self.state, self.structural, self.is_active,
                self.first_pos, self.bbox_dims, self.bbox_center, repr(self.points))

    def full_sig(self):
        return (self.draw_sig(), self.n_obs, self.score, self.last_seen,
                self.handled, repr(self.rgbd), repr(sorted(self.metadata.items())))


@dataclass
class Scene:
    stamp: float = 0.0
    robot_pose: Optional[Tuple[float, float, float]] = None
    grid: Optional[str] = None
    objects: Dict[int, ObjInfo] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _r(v, nd=3):
    return tuple(round(float(x), nd) for x in v)


def _plain(v):
    """spark_dsg metadata comes back as (nested) mappingproxy; make it plain."""
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _meta(obj) -> Dict[str, Any]:
    try:
        m = _plain(obj.metadata.get())
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def scene_from_graph(G: dsg.DynamicSceneGraph) -> Scene:
    gm = _meta(G)
    sc = Scene(metadata=gm)
    sc.stamp = float(gm.get("stamp", 0.0) or 0.0)
    rp = gm.get("robot_pose")
    if isinstance(rp, (list, tuple)) and len(rp) >= 3:
        sc.robot_pose = _r(rp[:3], 4)
    sc.grid = gm.get("grid")

    if not G.has_layer(dsg.DsgLayers.OBJECTS):
        return sc
    layer = G.get_layer(dsg.DsgLayers.OBJECTS)
    for node in layer.nodes:
        a = node.attributes
        md = _meta(a)
        sym = node.id
        oid = int(sym.category_id)
        fp = md.get("first_pos")
        info = ObjInfo(
            id=oid,
            sym=sym.str(),
            name=str(getattr(a, "name", "") or sym.str()),
            pos=_r(a.position),
            state=str(md.get("state", "seen")),
            n_obs=int(md.get("n_obs", 0) or 0),
            score=float(md.get("score", 0.0) or 0.0),
            first_pos=_r(fp) if isinstance(fp, (list, tuple)) and len(fp) == 3 else None,
            last_seen=float(getattr(a, "last_update_time_ns", 0)) * 1e-9,
            is_active=bool(getattr(a, "is_active", True)),
            structural=bool(md.get("structural", False)),
            handled=bool(md.get("handled", False)),
            rgbd=md.get("rgbd") if isinstance(md.get("rgbd"), dict) else None,
            points=md.get("points") if isinstance(md.get("points"), dict) else None,
            metadata=md,
        )
        bb = getattr(a, "bounding_box", None)
        try:
            if bb is not None and bb.is_valid():
                info.bbox_dims = _r(bb.dimensions)
                info.bbox_center = _r(bb.world_P_center)
                info.bbox_corners = np.asarray(bb.corners(), dtype=np.float64)
        except Exception:
            pass
        sc.objects[oid] = info

    # room -> object parent edges (the only object link the graph keeps; no on/in/near between objects)
    if G.has_layer(dsg.DsgLayers.ROOMS):
        room_name = {int(n.id.category_id): str(getattr(n.attributes, "name", "") or n.id.str())
                     for n in G.get_layer(dsg.DsgLayers.ROOMS).nodes}
        for ie in G.interlayer_edges:
            ps, ch = dsg.NodeSymbol(ie.source), dsg.NodeSymbol(ie.target)
            if ps.category == "R" and ch.category == "O" and int(ch.category_id) in sc.objects:
                sc.objects[int(ch.category_id)].room = room_name.get(int(ps.category_id), ps.str())
    return sc


def load_scene(dirpath: str) -> Scene:
    G = dsg.DynamicSceneGraph.load(os.path.join(dirpath, "scene.json"))
    return scene_from_graph(G)


@dataclass
class Diff:
    added: List[int] = field(default_factory=list)
    redraw: List[int] = field(default_factory=list)   # geometry changed
    changed: List[int] = field(default_factory=list)  # any field changed (superset of redraw)
    removed: List[int] = field(default_factory=list)

    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed)


def diff_scenes(old: Optional[Scene], new: Scene) -> Diff:
    d = Diff()
    oo = old.objects if old else {}
    for oid, o in new.objects.items():
        p = oo.get(oid)
        if p is None:
            d.added.append(oid)
        else:
            if p.draw_sig() != o.draw_sig():
                d.redraw.append(oid)
            if p.full_sig() != o.full_sig():
                d.changed.append(oid)
    d.removed = [oid for oid in oo if oid not in new.objects]
    return d


def read_map_yaml(path: str) -> Dict[str, Any]:
    """Tiny parser for ROS map_server yaml (no pyyaml dependency)."""
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


def map_texture(pgm: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Occupancy PGM -> RGB texture.

    Cells keep their grey level (0 = occupied ... 254 = free; the runtime writes
    graded values), except the map_saver "unknown" value 205, drawn blue-grey.
    """
    g = pgm.astype(np.uint8)
    if int(meta.get("negate", 0) or 0):
        g = 255 - g
    rgb = np.repeat(g[..., None], 3, axis=2)
    rgb[pgm == 205] = (150, 150, 165)
    return rgb


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Minimal PLY vertex reader (binary little/big endian or ascii; numpy only).

    Returns (xyz float32 [N,3], rgb uint8 [N,3] or None). Elements after
    "vertex" (faces etc.) are ignored.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt, n, props, cur, before = None, 0, [], None, 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header without end_header")
            tok = line.decode("ascii", "replace").split()
            if not tok or tok[0] in ("comment", "obj_info"):
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                cur = tok[1]
                if cur == "vertex":
                    n = int(tok[2])
                elif n == 0:
                    before += 1  # an element before vertex: unsupported layout
            elif tok[0] == "property" and cur == "vertex":
                if tok[1] == "list":
                    raise ValueError("list property in vertex element")
                props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if before:
            raise ValueError("elements before vertex are not supported")
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            col = {nm: data[:, i] for i, nm in enumerate(names)}
        else:
            end = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(nm, end + t) for nm, t in props])
            buf = f.read(dt.itemsize * n)
            if len(buf) < dt.itemsize * n:
                raise ValueError("truncated PLY")
            arr = np.frombuffer(buf, dtype=dt, count=n)
            col = {nm: arr[nm] for nm in names}
    xyz = np.stack([col["x"], col["y"], col["z"]], axis=1).astype(np.float32)
    rgb = None
    for keys in (("red", "green", "blue"), ("r", "g", "b")):
        if all(k in col for k in keys):
            rgb = np.stack([col[k] for k in keys], axis=1)
            if rgb.dtype.kind == "f":
                rgb = rgb * (255.0 if rgb.max() <= 1.0 else 1.0)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            break
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok], (rgb[ok] if rgb is not None else None)


class PointCache:
    """(path, mtime_ns, size)-keyed cache of object point clouds."""

    def __init__(self, maxlen: int = 512):
        self._d: Dict[str, Tuple[Tuple[int, int], Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
        self._maxlen = maxlen

    @staticmethod
    def file_key(path: str) -> Optional[Tuple[int, int]]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def get(self, path: str):
        """-> (key, (xyz, rgb)) or (None, None) if missing/unreadable."""
        key = self.file_key(path)
        if key is None:
            return None, None
        hit = self._d.get(path)
        if hit is not None and hit[0] == key:
            return key, hit[1]
        try:
            pts = read_ply(path)
        except Exception as ex:  # half-written -> try again on a later poll
            print(f"[sgviz] ply read failed {path}: {ex}", file=sys.stderr)
            return None, None
        if len(self._d) >= self._maxlen and path not in self._d:
            self._d.pop(next(iter(self._d)))
        self._d[path] = (key, pts)
        return key, pts


def _erode(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def rgb_overlay(rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """RGB crop with the segment shown: outside dimmed, inside kept, yellow outline."""
    if mask is None:
        return rgb
    out = rgb.astype(np.float32)
    out[~mask] = out[~mask] * 0.45 + 30.0
    edge = mask & ~_erode(mask)
    edge |= np.roll(edge, 1, axis=1)  # 2 px wide
    out[edge] = (255, 210, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_gray(depth: np.ndarray, mask: Optional[np.ndarray] = None):
    """Depth crop -> grayscale RGB (near bright, far dark) + stats in metres.

    Range is a robust 2-98 percentile of valid depth inside the mask (whole crop
    when no mask). Pixels outside the mask are dimmed; invalid (0) is black.
    Input: uint16 mm, or float metres. Returns (rgb uint8, (min, median, max) m or None).
    """
    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if depth.dtype.kind in "ui":
        d = d * 1e-3
    valid = np.isfinite(d) & (d > 0)
    sel = valid & mask if mask is not None else valid
    if not sel.any():
        sel = valid
    rgb = np.zeros(d.shape + (3,), np.uint8)
    if not sel.any():
        return rgb, None
    vals = d[sel]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo < 1e-3:
        lo, hi = lo - 0.05, hi + 0.05
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    g = 255.0 - 200.0 * t  # near 255, far 55
    if mask is not None:
        g = np.where(mask, g, g * 0.35)
    g = np.where(valid, g, 0.0)
    rgb[:] = g.astype(np.uint8)[..., None]
    stats = (float(vals.min()), float(np.median(vals)), float(vals.max()))
    return rgb, stats


def maybe_crop(img: np.ndarray, box_px, margin: int = 8) -> np.ndarray:
    """Crop to box_px if the stored image is a full frame (box fits well inside)."""
    if not box_px or len(box_px) != 4:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box_px)
    if (x1 - x0, y1 - y0) == (w, h):
        return img  # stored image is exactly the box crop
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return img  # already a crop (box is in full-frame coords)
    if (x1 - x0) * (y1 - y0) > 0.8 * w * h:
        return img
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
    return img[y0:y1, x0:x1]


def _read_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.array(im, dtype=np.uint16 if im.mode.startswith("I;16") else np.int32)
        return np.array(im.convert("RGB"))


def _fit(img: np.ndarray, max_side: int = 320, min_side: int = 240) -> np.ndarray:
    """Downsample big crops; upscale small ones (nearest, integer) so the panel shows them."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s < min_side:
        k = max(1, min(8, max_side // s))
        return np.repeat(np.repeat(img, k, axis=0), k, axis=1) if k > 1 else img
    if s <= max_side:
        return img
    step = int(math.ceil(s / max_side))
    return img[::step, ::step]


class CropCache:
    """Lazy cache of the panel images for one rgbd record (keyed by file mtimes)."""

    def __init__(self, maxlen: int = 64):
        self._d: Dict[Any, Dict[str, Any]] = {}
        self._maxlen = maxlen

    def get(self, root: str, rgbd: Dict[str, Any]) -> Dict[str, Any]:
        """-> {"rgb": img|None, "depth": img|None, "stats": (min,med,max)|None, "mask": bool}"""
        paths = {k: os.path.join(root, rgbd[k]) for k in ("rgb", "depth", "mask")
                 if isinstance(rgbd.get(k), str)}
        key = [repr(rgbd.get("box_px"))]
        for k in sorted(paths):
            fk = PointCache.file_key(paths[k])
            key.append((k, fk))
            if fk is None:
                paths.pop(k)
        key = tuple(key)
        if key in self._d:
            return self._d[key]
        box = rgbd.get("box_px")
        out: Dict[str, Any] = {"rgb": None, "depth": None, "stats": None, "mask": False}
        rgb = dep = None
        try:
            if "rgb" in paths:
                rgb = maybe_crop(_read_image(paths["rgb"]), box)
                if rgb.ndim == 2:
                    rgb = np.repeat(rgb[..., None], 3, axis=2).astype(np.uint8)
            if "depth" in paths:
                dep = maybe_crop(_read_image(paths["depth"]), box)
        except Exception as ex:
            print(f"[sgviz] crop read failed: {ex}", file=sys.stderr)
            return out  # not cached: retry on next selection/update
        mask_rgb = mask_dep = None
        if "mask" in paths:
            ref = rgb if rgb is not None else dep
            try:
                if ref is not None:
                    full = _read_image(paths["mask"])
                    full = full if full.ndim == 2 else full[..., 0]
                    full = maybe_crop(full, box)
                    from PIL import Image as _I

                    def fit_mask(shape):
                        im = _I.fromarray(full.astype(np.uint8))
                        if im.size != (shape[1], shape[0]):
                            im = im.resize((shape[1], shape[0]), _I.NEAREST)
                        return np.array(im) >= 128

                    mask_rgb = fit_mask(rgb.shape) if rgb is not None else None
                    mask_dep = fit_mask(dep.shape) if dep is not None else None
                    out["mask"] = True
            except Exception as ex:
                print(f"[sgviz] mask read failed: {ex}", file=sys.stderr)
        if rgb is not None:
            # scale first so the outline stays thin on upscaled small crops
            m = _fit(mask_rgb) if mask_rgb is not None else None
            out["rgb"] = np.ascontiguousarray(rgb_overlay(_fit(rgb), m))
        if dep is not None:
            img, stats = depth_gray(dep, mask_dep)
            out["depth"], out["stats"] = _fit(np.ascontiguousarray(img)), stats
        if len(self._d) >= self._maxlen:
            self._d.pop(next(iter(self._d)))
        self._d[key] = out
        return out


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------


class PickIndex:
    """Ray picking over object point clouds (and fallback spheres), numpy only.

    Each entry keeps its points (float32 [N,3]), a hit radius and an AABB
    grown by that radius. pick(origin, direction) returns the id of the object
    whose nearest hit point (perpendicular distance <= radius, in front of the
    origin) is closest along the ray, or None.
    """

    def __init__(self):
        self._e: Dict[int, Tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self._e)

    def __contains__(self, oid):
        return oid in self._e

    def set_points(self, oid: int, xyz: np.ndarray, radius: float):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        self._e[oid] = (xyz, float(radius), xyz.min(axis=0) - radius, xyz.max(axis=0) + radius)

    def set_sphere(self, oid: int, center, radius: float):
        c = np.asarray(center, np.float32).reshape(1, 3)
        self._e[oid] = (c, float(radius), c[0] - radius, c[0] + radius)

    def remove(self, oid: int):
        self._e.pop(oid, None)

    def pick(self, origin, direction) -> Optional[int]:
        o = np.asarray(origin, np.float64)
        d = np.asarray(direction, np.float64)
        n = np.linalg.norm(d)
        if not np.isfinite(n) or n == 0:
            return None
        d = d / n
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
        o32, d32 = o.astype(np.float32), d.astype(np.float32)
        best, best_t = None, np.inf
        for oid, (pts, r, lo, hi) in self._e.items():
            # slab test against the radius-grown AABB (cheap reject)
            with np.errstate(invalid="ignore"):
                t1, t2 = (lo - o) * inv, (hi - o) * inv
            t1 = np.where(np.isnan(t1), -np.inf, t1)
            t2 = np.where(np.isnan(t2), np.inf, t2)
            tmin = float(np.max(np.minimum(t1, t2)))
            tmax = float(np.min(np.maximum(t1, t2)))
            if tmax < max(tmin, 0.0) or tmin > best_t:
                continue
            v = pts - o32
            t = v @ d32
            perp2 = np.einsum("ij,ij->i", v, v) - t * t
            ok = (t > 0) & (perp2 <= r * r)
            if not ok.any():
                continue
            th = float(t[ok].min())
            if th < best_t:
                best, best_t = oid, th
        return best


def _yaw_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _box_segments(o: ObjInfo, max_dim: float) -> Optional[np.ndarray]:
    if o.bbox_dims is None or o.bbox_center is None:
        return None
    dims = np.asarray(o.bbox_dims, float)
    if not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        return None
    if np.all(dims <= max_dim) and o.bbox_corners is not None:
        corners = o.bbox_corners
    else:
        # clamp absurd extents (merged floor blobs etc.) around the center
        dims = np.minimum(dims, max_dim)
        c = np.asarray(o.bbox_center, float)
        bb = dsg.BoundingBox(dims.astype(np.float32), c.astype(np.float32))
        corners = np.asarray(bb.corners(), float)
    return corners[BOUNDING_BOX_EDGE_INDICES]


class SgViewer(ViserRenderer):
    def __init__(self, dirpath: str, ip: str = "0.0.0.0", port: int = 8080, poll: float = 0.25):
        super().__init__(ip, port=port, clear_at_exit=False)
        if self._server is None:
            raise RuntimeError("viser is not installed in this environment")
        self.server = self._server
        self.dir = os.path.abspath(dirpath)
        self.poll = poll
        self.lock = threading.RLock()
        self.scene: Optional[Scene] = None
        self.selected: Optional[int] = None
        self.crops = CropCache()
        self.points = PointCache()
        self._points_key: Dict[int, Any] = {}  # oid -> file key of the drawn cloud
        self._scene_mtime = None
        self._map_hash = None
        self._map_handle = None
        self._robot = None
        self._node_handles: Dict[int, List[Any]] = {}
        self._sel_handle = None
        self._camera_set = False
        self.picker = PickIndex()
        self._suppress_dropdown = False
        self._occ = None            # occupied mask of the current map (PGM row order)
        self._res = 0.05
        self._origin = (0.0, 0.0)
        self._wall_segs = None      # (N, 4) map-frame wall segments from the occupancy map
        self._wall_handles: List[Any] = []
        self._wall_vec = None       # last wall state vector
        self._wall_pose = None
        self._build_gui()
        from sglayers import LayerView  # Hydra-style stacked graph layers (view.json "graph")
        self.layers = LayerView(self.server, self.dir)

    # ---------------- GUI ----------------
    def _build_gui(self):
        gui = self.server.gui
        self.g_status = gui.add_markdown("waiting for scene.json ...")
        with gui.add_folder("Display", expand_by_default=False):
            self.g_labels = gui.add_checkbox("Labels", True)
            self.g_points = gui.add_checkbox("Segment points", True)
            self.g_pcolor = gui.add_dropdown("Point colour", ["true colour", "state colour"],
                                             initial_value="true colour")
            self.g_psize = gui.add_number("Point size x voxel", 1.0, min=0.2, max=5.0, step=0.1)
            self.g_markers = gui.add_checkbox("Centre markers", False)
            self.g_boxes = gui.add_checkbox("Boxes (objects w/o points)", False)
            self.g_roomlbl = gui.add_checkbox("Room in label", False)
            self.g_minobs = gui.add_number("Min observations", 3, min=1, max=200, step=1)
            self.g_trails = gui.add_checkbox("Moved trails", True)
            self.g_struct = gui.add_checkbox("Structural objects", True)
            self.g_noceil = gui.add_checkbox("Hide ceiling", True)
            self.g_ceilz = gui.add_number("Ceiling cut height [m]", 2.1, min=0.5, max=6.0, step=0.05)
            self.g_walls2d = gui.add_checkbox("Walls as 2D lines", True)
            self.g_gone = gui.add_checkbox("Gone objects", True)
            self.g_map = gui.add_checkbox("Occupancy map", True)
            self.g_size = gui.add_number("Node radius (no points)", 0.06, min=0.01, max=0.5, step=0.01)
            self.g_maxbox = gui.add_number("Max box side [m]", 2.0, min=0.1, max=20.0, step=0.1)
        for h in (self.g_labels, self.g_points, self.g_pcolor, self.g_psize, self.g_markers,
                  self.g_boxes, self.g_trails,
                  self.g_struct, self.g_gone, self.g_size, self.g_maxbox,
                  self.g_noceil, self.g_ceilz, self.g_walls2d, self.g_roomlbl, self.g_minobs):
            h.on_update(lambda _: self._redraw_all())
        self.g_map.on_update(lambda _: self._map_visibility())

        with gui.add_folder("Wall state (2D walls -> numbers)", expand_by_default=True):
            self.g_wstate = gui.add_markdown("waiting for the map ...")
            self.g_wsave = gui.add_button("Save wall state (.json)")
        self.g_wsave.on_click(lambda _: self._save_wall_state())

        with gui.add_folder("Object", expand_by_default=True):
            self.g_select = gui.add_dropdown("Select", ["(none)"], initial_value="(none)")
            self.g_info = gui.add_markdown("Click a node or pick one above.")
            blank = np.zeros((2, 2, 3), np.uint8)
            self.g_rgb = gui.add_image(blank, label="RGB crop", visible=False)
            self.g_depth = gui.add_image(blank, label="Depth crop (near bright)", visible=False)
            self.g_dstats = gui.add_markdown("", visible=False)
        self.g_select.on_update(lambda _: self._on_dropdown())
        # text search over object embeddings: tools/text_query.py --serve (SigLIP 2 text tower, off-board)
        self.query_url = os.environ.get("SGVIZ_QUERY_URL", "http://127.0.0.1:8091")
        with gui.add_folder("Search (text -> objects)", expand_by_default=True):
            self.g_query = gui.add_text("Query", initial_value="")
            self.g_qbtn = gui.add_button("Search")
            self.g_qres = gui.add_markdown(f"e.g. radio, 라디오, 흰 의자 — server `{self.query_url}`")
        self.g_qbtn.on_click(lambda _: self._search(self.g_query.value))

        # click anywhere in the 3D view -> ray pick over the object clouds/spheres
        @self.server.scene.on_click()
        def _(ev):
            self.on_scene_click(ev.ray_origin, ev.ray_direction)

    def on_scene_click(self, origin, direction) -> Optional[int]:
        """Select the object hit by the click ray (a miss keeps the current selection)."""
        if origin is None or direction is None:
            return None
        with self.lock:
            oid = self.picker.pick(origin, direction)
        if oid is not None and oid != self.selected:
            self.select(oid)
        return oid

    def _options(self) -> List[str]:
        objs = self.scene.objects if self.scene else {}
        return ["(none)"] + [objs[k].label for k in sorted(objs)]

    def _on_dropdown(self):
        if self._suppress_dropdown:
            return
        v = self.g_select.value
        oid = None
        if v and "#" in v:
            try:
                oid = int(v.rsplit("#", 1)[1])
            except ValueError:
                oid = None
        self.select(oid, from_dropdown=True)

    def select(self, oid: Optional[int], from_dropdown: bool = False):
        with self.lock:
            prev, self.selected = self.selected, oid
            if not from_dropdown:
                self._set_dropdown()
            if self.scene is not None:  # re-tint old/new selection
                with self.server.atomic():
                    for k in {prev, oid}:
                        if k is not None and k in self.scene.objects and k in self._node_handles:
                            self._draw_node(self.scene.objects[k])
            self._draw_selection()
            self._render_panel()

    def _set_dropdown(self):
        opts = self._options()
        cur = "(none)"
        if self.scene and self.selected in self.scene.objects:
            cur = self.scene.objects[self.selected].label
        self._suppress_dropdown = True
        try:
            if tuple(self.g_select.options) != tuple(opts):
                self.g_select.options = opts
            if self.g_select.value != cur:
                self.g_select.value = cur
        finally:
            self._suppress_dropdown = False

    def _search(self, q: str):
        q = (q or "").strip()
        if not q:
            return
        import urllib.parse
        import urllib.request
        url = f"{self.query_url}/search?" + urllib.parse.urlencode({"mem": os.path.abspath(self.dir), "q": q, "k": 5})
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                res = json.loads(r.read().decode("utf-8"))
        except Exception as ex:
            self.g_qres.content = f"search failed: {ex} — start `tools/text_query.py --serve 8091` (src/scene_graph/clip)"
            return
        hits = res.get("hits") or []
        lines = [f"**{q}**", "", "| # | object | name | cos |", "|---|---|---|---|"]
        for i, h in enumerate(hits):
            nm = h.get("name", "") + (f" / {h['name_ko']}" if h.get("name_ko") else "")
            lines.append(f"| {i + 1} | #{h['id']} | {nm} | {float(h['score']):.3f} |")
        self.g_qres.content = "\n".join(lines) if hits else f"**{q}**: no objects with embeddings yet"
        if hits and self.scene is not None and hits[0]["id"] in self.scene.objects:
            self.select(hits[0]["id"])

    def _render_panel(self):
        sc = self.scene
        o = sc.objects.get(self.selected) if (sc and self.selected is not None) else None
        if o is None:
            self.g_info.content = "Click a node or pick one above."
            self.g_rgb.visible = False
            self.g_depth.visible = False
            self.g_dstats.visible = False
            return
        f3 = lambda v: "-" if v is None else "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
        lines = [
            f"**{o.display_name}** #{o.id} (`{o.sym}`)",
            "",
            "| | |", "|---|---|",
            f"| state | **{o.state}**{' (structural)' if o.structural else ''}{' handled' if o.handled else ''} |",
            f"| pos | {f3(o.pos)} |",
            f"| first_pos | {f3(o.first_pos)} |",
            f"| n_obs | {o.n_obs} |",
            f"| score | {o.score:.3f} |",
            f"| last seen | {o.last_seen:.2f} s (now {sc.stamp:.2f}) |",
            f"| bbox | {f3(o.bbox_dims)} |",
            f"| room | {o.room or '-'} |",
        ]
        nm = o.names
        if nm:
            ko = f" / {nm['ko']}" if nm.get("ko") else ""
            lines.append(f"| name (SigLIP 2) | **{nm.get('en', '')}{ko}** {float(nm.get('score', 0)):.3f} |")
            if nm.get("general"):
                lines.append(f"| general | {nm['general']}{' / ' + nm['general_ko'] if nm.get('general_ko') else ''} |")
            top = nm.get("top") or []
            if top:
                lines.append("| top | " + ", ".join(f"{t[0]} {float(t[1]):.3f}" for t in top[:5]) + " |")
            lines.append(f"| labels | `{nm.get('table', '')}`{' structural' if nm.get('structural') else ''} |")
        emb = o.metadata.get("emb")
        if isinstance(emb, dict):
            lines.append(f"| embedding | `{emb.get('path', '')}` {emb.get('dim', '')}-d {emb.get('dtype', '')} sha `{emb.get('sha', '')}` |")
        if o.points:
            n = o.points.get("n", "?")
            vx = o.points.get("voxel")
            lines.append(f"| points | {n} pts, voxel {vx} m, stamp {o.points.get('stamp', '-')} |")
        rgbd = o.rgbd or {}
        if rgbd:
            lines.append(f"| rgbd stamp | {rgbd.get('stamp', '-')} |")
            lines.append(f"| box_px | {rgbd.get('box_px', '-')} |")
            dm = rgbd.get("depth_m")
            lines.append(f"| depth_m | {dm:.3f} |" if isinstance(dm, (int, float)) else f"| depth_m | {dm} |")
            if "mask_area" in rgbd:
                lines.append(f"| mask_area | {rgbd.get('mask_area')} |")
        self.g_info.content = "\n".join(lines)

        crop = self.crops.get(self.dir, rgbd) if rgbd else {}
        for kind, handle in (("rgb", self.g_rgb), ("depth", self.g_depth)):
            img = crop.get(kind)
            if img is None:
                handle.visible = False
            else:
                handle.image = img
                handle.visible = True
        st = crop.get("stats")
        if st is not None and self.g_depth.visible:
            where = "inside mask" if crop.get("mask") else "whole crop"
            self.g_dstats.content = (f"depth min / median / max = **{st[0]:.3f} / {st[1]:.3f} / "
                                     f"{st[2]:.3f} m** ({where})")
            self.g_dstats.visible = True
        else:
            self.g_dstats.visible = False

    # ---------------- drawing ----------------
    def _visible(self, o: ObjInfo) -> bool:
        if o.structural and not self.g_struct.value:
            return False
        if self.g_noceil.value and (CEILING_RE.search(o.display_name) or o.pos[2] > float(self.g_ceilz.value)):
            return False
        if self.g_walls2d.value and WALL_RE.search(o.display_name):
            return False  # drawn as 2D lines from the occupancy map instead
        if o.n_obs < int(self.g_minobs.value):
            return False  # seen only once or twice: usually a stray segment
        if o.state == "gone" and not self.g_gone.value:
            return False
        return True

    def _remove_node(self, oid: int):
        self.picker.remove(oid)
        for h in self._node_handles.pop(oid, []):
            try:
                h.remove()
            except Exception:
                pass

    def _points_path(self, o: ObjInfo) -> Optional[str]:
        p = (o.points or {}).get("path")
        if not isinstance(p, str) or not p:
            return None
        return p if os.path.isabs(p) else os.path.join(self.dir, p)

    def _draw_node(self, o: ObjInfo):
        self._remove_node(o.id)
        self._points_key.pop(o.id, None)
        if not self._visible(o):
            return
        sc = self.server.scene
        base = f"/objects/O{o.id}"
        col = STATE_COLORS.get(o.state, DEFAULT_COLOR)
        opacity = 0.35 if o.state == "gone" else None
        r = float(self.g_size.value)
        hs: List[Any] = []

        # segment shape (true-colour points) is the object's representation
        top_z = o.pos[2]
        has_pts = False
        path = self._points_path(o)
        if path is not None:
            key, pts = self.points.get(path)
            self._points_key[o.id] = key  # None = missing/unreadable -> retried in poll
            if pts is not None and len(pts[0]) > 0 and self.g_points.value:
                xyz, rgb = pts
                if self.g_noceil.value:
                    keep = xyz[:, 2] <= float(self.g_ceilz.value)
                    if not keep.any():
                        return  # the whole segment is above the ceiling cut
                    if not keep.all():
                        xyz = xyz[keep]
                        rgb = rgb[keep] if rgb is not None else None
                c = xyz.mean(axis=0)
                if rgb is None or self.g_pcolor.value == "state colour":
                    colors = np.tile(np.asarray(col, np.uint8), (len(xyz), 1))
                else:
                    colors = rgb
                if o.state == "gone":
                    colors = (colors.astype(np.float32) * 0.4 + 140 * 0.6).astype(np.uint8)
                vox = float((o.points or {}).get("voxel") or 0.02)
                psize = max(1e-3, vox * float(self.g_psize.value))
                if o.id == self.selected:  # highlight: tint towards yellow, slightly bigger
                    colors = (colors.astype(np.float32) * 0.45
                              + np.array([255, 225, 40], np.float32) * 0.55).astype(np.uint8)
                    psize *= 1.25
                hs.append(sc.add_point_cloud(
                    f"{base}/points", (xyz - c).astype(np.float32), colors,
                    point_size=psize,
                    point_shape="square", position=tuple(float(v) for v in c)))
                self.picker.set_points(o.id, xyz, max(2.0 * vox, 0.02))
                top_z = float(xyz[:, 2].max())
                has_pts = True
            elif pts is not None and len(pts[0]) > 0:
                has_pts = True  # points hidden by toggle: still no box

        # fallback sphere (objects without points) / optional centre marker.
        # Clicks are handled by the scene-level ray pick, not per-mesh on_click.
        if not has_pts or self.g_markers.value:
            rad = 0.02 if has_pts else r
            hs.append(sc.add_icosphere(f"{base}/node", radius=rad, color=col, position=o.pos,
                                       subdivisions=2, opacity=opacity))
            if not has_pts:
                self.picker.set_sphere(o.id, o.pos, 1.5 * r)
        if self.g_labels.value:
            z = max(top_z, o.pos[2]) + (0.04 if has_pts else 1.8 * r)
            hs.append(sc.add_label(f"{base}/label", o.label + (f" [{o.room}]" if (self.g_roomlbl.value and o.room) else ""), position=(o.pos[0], o.pos[1], z),
                                   anchor="bottom-center", font_screen_scale=0.8))
        if self.g_boxes.value and not has_pts:
            seg = _box_segments(o, float(self.g_maxbox.value))
            if seg is not None:
                hs.append(sc.add_line_segments(f"{base}/bbox", seg, col, thickness=1.5,
                                               thickness_units="screen"))
        if self.g_trails.value and o.first_pos is not None and o.state == "moved":
            fp = np.asarray(o.first_pos, float)
            if np.linalg.norm(fp - np.asarray(o.pos)) > 0.02:
                hs.append(sc.add_line_segments(
                    f"{base}/trail", np.array([[fp, o.pos]], float), STATE_COLORS["moved"],
                    thickness=3.0, thickness_units="screen"))
                hs.append(sc.add_icosphere(f"{base}/first", radius=0.03, color=(180, 180, 180),
                                           position=tuple(fp), subdivisions=1, opacity=0.6))
        self._node_handles[o.id] = hs

    def _refresh_points(self) -> List[int]:
        """Redraw objects whose PLY file appeared/changed after the node was drawn."""
        if self.scene is None:
            return []
        todo = []
        for oid, o in self.scene.objects.items():
            path = self._points_path(o)
            if path is None or oid not in self._node_handles:
                continue
            if PointCache.file_key(path) != self._points_key.get(oid):
                todo.append(oid)
        if todo:
            with self.server.atomic():
                for oid in todo:
                    self._draw_node(self.scene.objects[oid])
        return todo

    def _draw_selection(self):
        if self._sel_handle is not None:
            self._sel_handle.remove()
            self._sel_handle = None
        sc = self.scene
        if sc is None or self.selected not in sc.objects:
            return
        if any(h.name.endswith("/points") for h in self._node_handles.get(self.selected, [])):
            return  # point-cloud objects are highlighted by tinting their points
        o = sc.objects[self.selected]
        self._sel_handle = self.server.scene.add_icosphere(
            "/selection", radius=2.2 * float(self.g_size.value), color=(255, 220, 0),
            position=o.pos, subdivisions=2, wireframe=True)

    def _redraw_all(self):
        with self.lock:
            if self.scene is None:
                return
            with self.server.atomic():
                for o in self.scene.objects.values():
                    self._draw_node(o)
                self._draw_selection()
            self._draw_walls2d()

    # ---------------- walls as 2D + wall state ----------------
    def _draw_walls2d(self):
        for h in self._wall_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._wall_handles = []
        segs = self._wall_segs
        if not self.g_walls2d.value or segs is None or len(segs) == 0:
            return
        seg3 = np.zeros((len(segs), 2, 3), float)
        seg3[:, 0, :2] = segs[:, 0:2]
        seg3[:, 1, :2] = segs[:, 2:4]
        seg3[:, :, 2] = 0.03
        self._wall_handles.append(self.server.scene.add_line_segments(
            "/walls2d", seg3, WALL_LINE_COLOR, thickness=4.0, thickness_units="screen"))

    def _update_wall_state(self, pose):
        if self._occ is None or self._wall_segs is None or pose is None:
            return
        self._wall_pose = tuple(float(v) for v in pose[:3])
        vec = walls2d.wall_state_vector(self._occ, self._res, self._origin, self._wall_segs, self._wall_pose)
        self._wall_vec = vec
        n, k, R = walls2d.N_SECTORS, walls2d.K_SEGMENTS, walls2d.MAX_RANGE
        rays = (vec[:n] * R)
        segs = vec[n:].reshape(k, 5)
        rows = [f"| {j} | {s[0]*R:+.2f} | {s[1]*R:+.2f} | {s[2]*R:+.2f} | {s[3]*R:+.2f} |"
                for j, s in enumerate(segs) if s[4] > 0][:5]
        self.g_wstate.content = (
            f"pose (map) x {pose[0]:.2f}  y {pose[1]:.2f}  yaw {pose[2]:.2f}  \n"
            f"wall segments in map: **{len(self._wall_segs)}**  |  state vector length **{len(vec)}**  \n"
            f"ray distance [m] ({n} sectors from straight ahead, CCW, max {R:.0f}):  \n"
            + " ".join(f"{d:.1f}" for d in rays)
            + "  \n\nnearest wall segments, robot frame [m]:  \n| # | ax | ay | bx | by |\n|---|---|---|---|---|\n"
            + "\n".join(rows))

    def _save_wall_state(self):
        if self._wall_vec is None:
            return
        import json
        out_dir = os.environ.get("SGVIZ_STATE_DIR") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "wall_state")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        tag = os.path.basename(os.path.dirname(self.dir.rstrip("/"))) or "memory"
        stamp = self.scene.stamp if self.scene is not None else 0.0
        base = os.path.join(out_dir, f"{tag}_t{stamp:.1f}")
        with open(base + ".json", "w") as f:
            json.dump({"memory_dir": self.dir, "stamp": stamp, "pose": list(self._wall_pose),
                       "resolution": self._res, "origin": list(self._origin),
                       "n_sectors": walls2d.N_SECTORS, "k_segments": walls2d.K_SEGMENTS,
                       "max_range": walls2d.MAX_RANGE,
                       "segments_map_frame": self._wall_segs.tolist(),
                       "state_vector": self._wall_vec.tolist()}, f)
        np.save(base + ".npy", self._wall_vec)
        self.g_wstate.content += f"  \n\nsaved: `{base}.json` / `.npy`"

    def _draw_robot(self, pose):
        if pose is None:
            return
        x, y, yaw = pose
        if self._robot is None:
            sc = self.server.scene
            root = sc.add_frame("/robot", show_axes=False, position=(x, y, 0.0), wxyz=_yaw_wxyz(yaw))
            body = sc.add_icosphere("/robot/body", radius=0.25, color=(30, 30, 30),
                                    scale=(1.0, 1.0, 0.15), position=(0, 0, 0.05), subdivisions=2)
            arrow = sc.add_arrows("/robot/heading", np.array([[[0, 0, 0.1], [0.6, 0, 0.1]]], float),
                                  (220, 30, 30), shaft_radius=0.03, head_radius=0.08, head_length=0.15)
            self._robot = (root, body, arrow)
        else:
            self._robot[0].position = (x, y, 0.0)
            self._robot[0].wxyz = _yaw_wxyz(yaw)

    def _map_visibility(self):
        if self._map_handle is not None:
            self._map_handle.visible = bool(self.g_map.value)

    def _update_map(self, grid_name: Optional[str]):
        pgm_path = os.path.join(self.dir, grid_name or "map.pgm")
        yaml_path = os.path.splitext(pgm_path)[0] + ".yaml"
        try:
            with open(pgm_path, "rb") as f:
                raw = f.read()
            with open(yaml_path, "rb") as f:
                ytxt = f.read()
        except OSError:
            return
        h = hashlib.md5(raw + ytxt).hexdigest()
        if h == self._map_hash:
            return
        try:
            import io
            from PIL import Image

            meta = read_map_yaml(yaml_path)
            pgm = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        except Exception as ex:  # partially written file -> retry next poll
            print(f"[sgviz] map read failed: {ex}", file=sys.stderr)
            return
        self._map_hash = h
        res = float(meta.get("resolution", 0.05))
        ox, oy = (meta.get("origin") or [0.0, 0.0])[:2]
        H, W = pgm.shape
        self._occ = walls2d.occupied_mask(pgm)
        self._res, self._origin = res, (float(ox), float(oy))
        self._wall_segs = walls2d.wall_segments(self._occ, res, self._origin)
        self._draw_walls2d()
        tex = map_texture(pgm, meta)
        # viser image planes have row 0 at local -Y; PGM row 0 is max-y.
        tex = np.ascontiguousarray(np.flipud(tex))
        if self._map_handle is not None:
            self._map_handle.remove()
        self._map_handle = self.server.scene.add_image(
            "/map", tex, render_width=W * res, render_height=H * res, format="png",
            position=(ox + W * res / 2, oy + H * res / 2, -0.01),
            cast_shadow=False, receive_shadow=False, visible=bool(self.g_map.value))

    def apply(self, new: Scene) -> Diff:
        with self.lock:
            d = diff_scenes(self.scene, new)
            self.scene = new
            if not d.empty():
                with self.server.atomic():
                    for oid in d.removed:
                        self._remove_node(oid)
                    for oid in d.added + d.redraw:
                        self._draw_node(new.objects[oid])
                    moved = set(d.redraw) | set(d.removed) | set(d.added)
                    if self.selected is not None and self.selected in moved:
                        self._draw_selection()
            if d.added or d.removed:
                self._set_dropdown()
            if self.selected is not None and (self.selected in d.changed or d.removed):
                if self.selected not in new.objects:
                    self.selected = None
                    self._set_dropdown()
                    self._draw_selection()
                self._render_panel()
            self._draw_robot(new.robot_pose)
            self._update_map(new.grid)
            self._update_wall_state(new.robot_pose)
            n_moved = sum(o.state == "moved" for o in new.objects.values())
            self.g_status.content = (
                f"`{self.dir}`  \nstamp **{new.stamp:.1f} s** | objects {len(new.objects)} "
                f"(moved {n_moved})" + (f"  \n{self._explore_line}" if getattr(self, "_explore_line", None) else ""))
            if not self._camera_set and new.robot_pose is not None:
                x, y, _ = new.robot_pose
                # Map_Vla: the stacked Hydra layers float at z = 4 .. 10 m, so start further back and higher to see house + graph
                self.server.initial_camera.position = (x - 9.0, y - 9.0, 9.0)
                self.server.initial_camera.look_at = (x + 1.0, y + 1.0, 4.0)
                self._camera_set = True
            return d

    def poll_once(self) -> Optional[Diff]:
        path = os.path.join(self.dir, "scene.json")
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key == self._scene_mtime:
            return None
        try:
            new = load_scene(self.dir)
        except Exception as ex:  # mid-write / malformed: retry next poll
            print(f"[sgviz] scene.json load failed: {ex}", file=sys.stderr)
            return None
        self._scene_mtime = key
        return self.apply(new)

    def _update_explore(self):
        """explore.json (skill explore, run_explore.py): trail (grey), planned path (blue), goal (magenta), frontier ids"""
        path = os.path.join(self.dir, "explore.json")
        try:
            st = os.stat(path)
        except OSError:
            return
        key = (st.st_mtime_ns, st.st_size)
        if key == getattr(self, "_explore_key", None):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                ex = json.load(f)
        except Exception:
            return
        self._explore_key = key
        sc = self.server.scene
        for h in getattr(self, "_explore_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        hs = []

        def polyline(name, pts, color, z):
            if len(pts) < 2:
                return
            a = np.array(pts, float)
            seg = np.stack([np.c_[a[:-1], np.full(len(a) - 1, z)], np.c_[a[1:], np.full(len(a) - 1, z)]], axis=1)
            hs.append(sc.add_line_segments(name, seg, colors=color, line_width=3.0))

        polyline("/explore/trail", ex.get("trail") or [], (120, 120, 120), 0.03)
        polyline("/explore/plan", ex.get("plan") or [], (30, 90, 230), 0.05)
        g = ex.get("goal")
        if g and g.get("xy"):
            hs.append(sc.add_icosphere("/explore/goal", radius=0.12, color=(220, 40, 200), position=(g["xy"][0], g["xy"][1], 0.15)))
        for t in ex.get("targets") or []:
            x, y = t["xy"]
            hs.append(sc.add_label(f"/explore/t_{t['id']}", f"{t['id']} {t.get('path_m', '')}m", position=(x, y, 0.3)))
        self._explore_handles = hs
        cov = ex.get("gt_cov")
        self._explore_line = f"explore: odo {ex.get('odo_m')} m, contacts {ex.get('contacts')}" + (f", GT cover {cov:.1%}" if cov else "")

    def run(self):
        print(f"[sgviz] watching {self.dir}; open http://localhost:{self.server.get_port()}", flush=True)
        while True:
            try:
                with self.lock:
                    self._update_explore()
                self.poll_once()
                with self.lock:
                    self._refresh_points()
                    if self.layers.poll() and self.scene is not None:
                        self.g_status.content = self.g_status.content.split("  \ngraph:")[0] + "  \n" + self.layers.status()
            except Exception as ex:
                print(f"[sgviz] update error: {ex!r}", file=sys.stderr)
            time.sleep(self.poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="memory output dir (contains scene.json)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--poll", type=float, default=0.25, help="scene.json poll period [s]")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.dir):
        ap.error(f"not a directory: {args.dir}")
    viewer = SgViewer(args.dir, ip=args.host, port=args.port, poll=min(args.poll, 0.5))
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
