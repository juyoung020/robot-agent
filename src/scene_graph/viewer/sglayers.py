"""Hydra-style stacked scene-graph layers for sgviz (viser).

Reads the compact graph in view.json ("graph": agent/place nodes and every edge, "rooms", "objects")
written by scenemap (docs/scenemap_설계.md 3.5) and draws it the way Hydra's figures do: the map and object
clouds stay at their true height, the graph layers float above, stacked by height:

    z = 0            occupancy map, object point clouds (drawn by sgviz itself)
    z = obj_z        OBJECTS layer nodes (+ best-view thumbnails) and the AGENTS trajectory
    z = base         PLACES (coloured by clearance) + place-place edges
    z = base + h     ROOMS (room colour) + door edges

Map_Vla: three layers (objects, places, rooms); the BUILDING layer was dropped (old files that still have one are not drawn).

Inter-layer edges are thin lines (object->place, place->room, room->object, place->agent).
Every layer is one point cloud and every edge type one line-segment set; each is re-sent only when its
content hash changes (no full re-upload per refresh). Thumbnails are re-sent only when the PNG changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LAYER_COLORS = {"object": (46, 160, 67), "agent": (40, 110, 230), "building": (60, 60, 60), "frontier": (230, 40, 200)}
EDGE_COLORS = {
    "place": (90, 90, 90),
    "door": (230, 120, 20),
    "agent": (40, 110, 230),
    "inter": (150, 150, 150),
}


def clearance_color(c: np.ndarray) -> np.ndarray:
    """Clearance [m] -> colour (0 red .. 1 m+ blue), like Hydra's distance colouring of places."""
    t = np.clip(np.asarray(c, float) / 1.0, 0, 1)[:, None]
    lo, mid, hi = np.array([220, 50, 40.0]), np.array([240, 200, 40.0]), np.array([40, 120, 230.0])
    out = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    return out.astype(np.uint8)


def _hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()


def parse_layers(view: Dict[str, Any]) -> Dict[str, Any]:
    """view.json -> node positions per layer + edges with end points (headless, testable)."""
    g = view.get("graph") or {}
    nodes: Dict[str, Dict[str, Any]] = {}
    for n in g.get("nodes", []):
        nodes[n["id"]] = n
    for o in view.get("objects", []):
        nodes[f"O{o['id']}"] = {"id": f"O{o['id']}", "kind": "object", "pos": o["pos"], "name": o.get("name", ""),
                                "state": o.get("state"), "rgbd": o.get("rgbd")}
    for r in view.get("rooms", []):
        nodes[f"R{r['id']}"] = {"id": f"R{r['id']}", "kind": "room", "pos": r["centroid"], "name": r.get("name", ""),
                                "color": r.get("color", [200, 200, 200])}
    edges = []
    for e in g.get("edges", []):
        a, b, rel, w = e[0], e[1], e[2], e[3]
        if a in nodes and b in nodes:
            edges.append((a, b, rel, w))
    return {"nodes": nodes, "edges": edges}


class LayerView:
    def __init__(self, server, dirpath: str):
        self.server = server
        self.dir = dirpath
        self._key = None
        self._sent: Dict[str, str] = {}      # handle name -> content hash
        self._handles: Dict[str, Any] = {}
        self._thumb_key: Dict[str, Any] = {}
        self.layers: Optional[Dict[str, Any]] = None
        gui = server.gui
        with gui.add_folder("Scene graph layers (Hydra)", expand_by_default=True):
            self.g_on = gui.add_checkbox("Show graph layers", True)
            self.g_objs = gui.add_checkbox("Objects layer", True)
            self.g_thumbs = gui.add_checkbox("Object thumbnails", True)
            self.g_agents = gui.add_checkbox("Agent trajectory", True)
            self.g_places = gui.add_checkbox("Places", True)
            self.g_rooms = gui.add_checkbox("Rooms", True)
            self.g_intra = gui.add_checkbox("Intra-layer edges", True)
            self.g_inter = gui.add_checkbox("Inter-layer edges", True)
            self.g_objz = gui.add_number("Objects layer z [m]", 4.0, min=0.0, max=10.0, step=0.1)
            self.g_base = gui.add_number("Places layer z [m]", 6.0, min=0.0, max=20.0, step=0.1)
            self.g_h = gui.add_slider("Layer spacing [m]", min=0.3, max=5.0, step=0.1, initial_value=2.0)
            self.g_thumb = gui.add_number("Thumbnail size [m]", 0.35, min=0.05, max=2.0, step=0.05)
            # Map_Vla: legend
            gui.add_markdown(
                "**Legend** (z: objects 4 m, places 6 m, rooms 8 m)  \n"
                "Objects (have image / points): 🟩 seen · 🟧 moved · 🟦 held · ⬜ gone  \n"
                "Places = free-space skeleton, **no image**: red→yellow→blue = clearance 0→1 m  \n"
                "Rooms: one colour each, big spheres  \n"
                "Lines: dark = place–place, grey = parent–child (object→place→room)")
        for h in (self.g_on, self.g_objs, self.g_thumbs, self.g_agents, self.g_places, self.g_rooms, self.g_intra,
                  self.g_inter, self.g_objz, self.g_base, self.g_h, self.g_thumb):
            h.on_update(lambda _: self.redraw())

    # ---------------- geometry ----------------
    def z_of(self, n: Dict[str, Any]) -> float:
        k = n["kind"]
        base, h = float(self.g_base.value), float(self.g_h.value)
        if k in ("object", "agent"):
            return float(self.g_objz.value)
        if k == "place":
            return base
        if k == "room":
            return base + h
        return base + 2 * h

    def xyz(self, n) -> Tuple[float, float, float]:
        return (float(n["pos"][0]), float(n["pos"][1]), self.z_of(n))

    def _shown(self, kind: str) -> bool:
        if not self.g_on.value:
            return False
        return {"object": self.g_objs.value, "agent": self.g_agents.value, "place": self.g_places.value,
                "room": self.g_rooms.value}.get(kind, False)

    # ---------------- drawing (each handle re-sent only on change) ----------------
    def _put(self, name: str, sig: str, make):
        if self._sent.get(name) == sig:
            return
        old = self._handles.pop(name, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        h = make()
        if h is not None:
            self._handles[name] = h
        self._sent[name] = sig

    def _drop(self, name: str):
        self._put(name, "none", lambda: None)

    def redraw(self):
        L = self.layers
        if L is None:
            return
        sc = self.server.scene
        nodes, edges = L["nodes"], L["edges"]
        with self.server.atomic():
            # node clouds per layer
            for kind, size in (("object", 0.10), ("agent", 0.08), ("place", 0.12), ("room", 0.30)):
                ns = [n for n in nodes.values() if n["kind"] == kind]
                name = f"/graph/{kind}_nodes"
                if not ns or not self._shown(kind):
                    self._drop(name)
                    continue
                pts = np.array([self.xyz(n) for n in ns], np.float32)
                if kind == "place":
                    # Map_Vla: frontier is an internal planning flag (explore), not something to draw. Hydra's own python/Spark-DSG
                    # viewers do not draw it either; places are coloured by clearance only.
                    col = clearance_color([n.get("clear", 0) for n in ns])
                elif kind == "room":
                    col = np.array([n.get("color", (200, 200, 200)) for n in ns], np.uint8)
                elif kind == "object":
                    st = {"seen": (46, 160, 67), "moved": (245, 140, 20), "held": (40, 110, 230), "gone": (140, 140, 140)}
                    col = np.array([st.get(n.get("state"), (200, 60, 200)) for n in ns], np.uint8)
                else:
                    col = np.tile(np.array(LAYER_COLORS[kind], np.uint8), (len(ns), 1))
                sig = _hash(pts.round(3).tobytes(), col.tobytes(), size)
                self._put(name, sig, lambda pts=pts, col=col, size=size: sc.add_point_cloud(
                    name, pts, col, point_size=size, point_shape="circle"))
            # labels for rooms (few)
            for n in nodes.values():
                if n["kind"] != "room":
                    continue
                name = f"/graph/label_{n['id']}"
                if not self._shown(n["kind"]):
                    self._drop(name)
                    continue
                p = self.xyz(n)
                text = n.get("name") or n["id"]
                self._put(name, _hash(p, text), lambda name=name, text=text, p=p: sc.add_label(
                    name, text, position=(p[0], p[1], p[2] + 0.25), anchor="bottom-center"))
            # edges per type
            groups: Dict[str, List[Tuple[Tuple[float, ...], Tuple[float, ...]]]] = {}
            for a, b, rel, _w in edges:
                na, nb = nodes[a], nodes[b]
                if not (self._shown(na["kind"]) and self._shown(nb["kind"])):
                    continue
                same = na["kind"] == nb["kind"] or {na["kind"], nb["kind"]} == {"object", "agent"}
                if rel in ("on", "in", "near"):
                    continue  # object-object prepositions are not used (old files may still contain them)
                if same:
                    if not self.g_intra.value:
                        continue
                    key = rel if rel in EDGE_COLORS else "place"
                else:
                    if not self.g_inter.value:
                        continue
                    key = "inter"
                groups.setdefault(key, []).append((self.xyz(na), self.xyz(nb)))
            for key in EDGE_COLORS:
                name = f"/graph/edges_{key}"
                seg = groups.get(key)
                if not seg:
                    self._drop(name)
                    continue
                arr = np.array(seg, np.float32)
                thick = 0.02 if key in ("place", "door", "agent") else 0.008
                self._put(name, _hash(arr.round(3).tobytes(), thick), lambda name=name, arr=arr, key=key, thick=thick:
                          sc.add_line_segments(name, arr, colors=EDGE_COLORS[key], thickness=thick))
            self._draw_thumbs(nodes)

    def _draw_thumbs(self, nodes):
        sc = self.server.scene
        want = set()
        if self.g_on.value and self.g_objs.value and self.g_thumbs.value:
            for n in nodes.values():
                if n["kind"] != "object" or not n.get("rgbd"):
                    continue
                path = os.path.join(self.dir, n["rgbd"]["rgb"])
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                name = f"/graph/thumb_{n['id']}"
                want.add(name)
                x, y, z = self.xyz(n)
                size = float(self.g_thumb.value)
                sig = _hash(st.st_mtime_ns, st.st_size, round(x, 3), round(y, 3), round(z, 3), size)
                if self._sent.get(name) == sig:
                    continue
                try:
                    from PIL import Image
                    img = np.asarray(Image.open(path).convert("RGB"))
                except Exception:
                    continue
                h, w = img.shape[:2]
                rw, rh = (size, size * h / w) if w >= h else (size * w / h, size)
                # image plane (xy) stood up (Rx 90°) and turned to face the default camera side (-x, -y)
                q = _quat_mul(_quat_axis((0, 0, 1), -np.pi / 4 + np.pi), _quat_axis((1, 0, 0), np.pi / 2))
                self._put(name, sig, lambda name=name, img=img, rw=rw, rh=rh, x=x, y=y, z=z, q=q: sc.add_image(
                    name, img, rw, rh, wxyz=q, position=(x, y, z + 0.12 + rh / 2), cast_shadow=False, receive_shadow=False))
        for name in [k for k in list(self._sent) if k.startswith("/graph/thumb_") and k not in want]:
            self._drop(name)
            self._sent.pop(name, None)

    # ---------------- polling ----------------
    def poll(self) -> bool:
        path = os.path.join(self.dir, "view.json")
        try:
            st = os.stat(path)
        except OSError:
            return False
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key == self._key:
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                view = json.load(f)
        except Exception:
            return False  # mid-write: next poll
        self._key = key
        self.layers = parse_layers(view)
        self.redraw()
        return True

    def status(self) -> str:
        if not self.layers:
            return ""
        from collections import Counter
        c = Counter(n["kind"] for n in self.layers["nodes"].values())
        return (f"graph: objects {c['object']}, agents {c['agent']}, places {c['place']}, rooms {c['room']}, "
                f"edges {len(self.layers['edges'])}")


def _quat_axis(axis, ang) -> Tuple[float, float, float, float]:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    s = np.sin(ang / 2)
    return (float(np.cos(ang / 2)), float(a[0] * s), float(a[1] * s), float(a[2] * s))


def _quat_mul(p, q) -> Tuple[float, float, float, float]:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
