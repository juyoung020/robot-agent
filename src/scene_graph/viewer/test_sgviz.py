#!/usr/bin/env python3
"""Headless test of sgviz loading/diff code (no browser, no viser server).

    ~/sdsg_venv/bin/python src/scene_graph/viewer/test_sgviz.py
"""

import os
import sys
import tempfile

import numpy as np
import spark_dsg as dsg
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgviz  # noqa: E402


def _obj(name, pos, dims, md):
    a = dsg.ObjectNodeAttributes()
    a.name = name
    a.position = np.array(pos, float)
    a.bounding_box = dsg.BoundingBox(np.array(dims, np.float32), np.array(pos, np.float32))
    a.last_update_time_ns = int(12.5e9)
    a.metadata.set(md)
    return a


def write_ply(path, xyz, rgb):
    """binary_little_endian PLY, float x,y,z + uchar red,green,blue (runtime format)."""
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment sgviz test\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = xyz.T
    a["red"], a["green"], a["blue"] = rgb.T
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(a.tobytes())
    os.replace(tmp, path)


def cup_points(center, n=200, seed=0):
    rng = np.random.default_rng(seed)
    xyz = (np.asarray(center) + rng.uniform(-0.04, 0.04, (n, 3))).astype(np.float32)
    rgb = np.tile(np.array([[200, 30, 30]], np.uint8), (n, 1))
    return xyz, rgb


def write_dir(d, cup_pos=(1.0, 0.5, 0.8), with_edge=True):
    os.makedirs(os.path.join(d, "objects"), exist_ok=True)
    G = dsg.DynamicSceneGraph()
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 1),
               _obj("table", (1.0, 0.5, 0.4), (1.2, 0.8, 0.8),
                    {"state": "seen", "n_obs": 40, "score": 0.8, "first_pos": [1.0, 0.5, 0.4],
                     "structural": True, "handled": False}))
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 2),
               _obj("cup", cup_pos, (0.08, 0.08, 0.1),
                    {"state": "moved", "n_obs": 7, "score": 0.6, "first_pos": [0.2, 0.1, 0.8],
                     "structural": False, "handled": False,
                     "points": {"path": "objects/O2_points.ply", "n": 200, "voxel": 0.01,
                                "stamp": 12.4},
                     "rgbd": {"rgb": "objects/O2_rgb.png", "depth": "objects/O2_depth.png",
                              "mask": "objects/O2_mask.png",
                              "stamp": 12.4, "box_px": [110, 112, 130, 140], "mask_area": 410,
                              "depth_m": 1.234, "cam_T": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}}))
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 3),
               _obj("floor", (0.0, 0.0, 0.0), (30.0, 20.0, 0.5),
                    {"state": "gone", "n_obs": 3, "score": 0.3, "first_pos": [0, 0, 0],
                     "structural": True}))
    if with_edge:
        e = dsg.EdgeAttributes()
        e.metadata.set({"relation": "on"})
        G.insert_edge(dsg.NodeSymbol("O", 2), dsg.NodeSymbol("O", 1), e)
    G.metadata.set({"stamp": 12.5, "robot_pose": [0.1, -0.2, 0.5], "grid": "map.pgm"})
    G.save(os.path.join(d, "scene.json"))

    # crops are exactly box_px sized (28 rows x 20 cols), like the runtime writes
    rgb = np.full((28, 20, 3), 200, np.uint8)
    Image.fromarray(rgb).save(os.path.join(d, "objects", "O2_rgb.png"))
    depth = np.full((28, 20), 3000, np.uint16)          # background 3 m
    depth[6:22, 4:16] = 1200 + np.arange(12, dtype=np.uint16) * 10  # cup 1.20-1.31 m
    depth[0, :] = 0                                     # invalid row
    Image.fromarray(depth).save(os.path.join(d, "objects", "O2_depth.png"))
    mask = np.zeros((28, 20), np.uint8)
    mask[6:22, 4:16] = 255
    Image.fromarray(mask).save(os.path.join(d, "objects", "O2_mask.png"))
    write_ply(os.path.join(d, "objects", "O2_points.ply"), *cup_points(cup_pos))

    pgm = np.full((20, 30), 205, np.uint8)
    pgm[5:15, 5:25] = 254
    pgm[10, :] = 0
    Image.fromarray(pgm).save(os.path.join(d, "map.pgm"))
    with open(os.path.join(d, "map.yaml"), "w") as f:
        f.write("image: map.pgm\nresolution: 0.05\norigin: [-0.5, -0.5, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.65\nfree_thresh: 0.25\n")


def main():
    with tempfile.TemporaryDirectory() as d:
        write_dir(d)
        sc = sgviz.load_scene(d)
        assert set(sc.objects) == {1, 2, 3}, sc.objects.keys()
        assert abs(sc.stamp - 12.5) < 1e-9 and sc.robot_pose == (0.1, -0.2, 0.5)
        assert sc.grid == "map.pgm"
        cup = sc.objects[2]
        assert cup.name == "cup" and cup.label == "cup#2" and cup.sym == "O2"
        assert cup.state == "moved" and cup.n_obs == 7 and abs(cup.score - 0.6) < 1e-9
        assert cup.first_pos == (0.2, 0.1, 0.8) and abs(cup.last_seen - 12.5) < 1e-6
        assert cup.rgbd["box_px"] == [110, 112, 130, 140] and cup.rgbd["depth_m"] == 1.234
        assert sc.objects[1].rgbd is None and sc.objects[1].structural
        # the fixture still writes an old-style object-object "on" edge: it is ignored (no object relations are used any more)
        assert not hasattr(sc, "edges") and not hasattr(sgviz, "relations_of")

        # boxes: real one kept, absurd floor clamped
        seg = sgviz._box_segments(sc.objects[2], 2.0)
        assert seg.shape == (12, 2, 3)
        fl = sgviz._box_segments(sc.objects[3], 2.0)
        assert np.ptp(fl[..., 0]) <= 2.0 + 1e-5 and np.ptp(fl[..., 1]) <= 2.0 + 1e-5

        # PLY reader (binary from the runtime format; ascii too)
        xyz, prgb = sgviz.read_ply(os.path.join(d, "objects", "O2_points.ply"))
        exyz, ergb = cup_points((1.0, 0.5, 0.8))
        assert xyz.shape == (200, 3) and np.allclose(xyz, exyz) and (prgb == ergb).all()
        with open(os.path.join(d, "a.ply"), "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
                    "property float y\nproperty float z\nend_header\n0 0 0\n1 2 3\n")
        axyz, argb = sgviz.read_ply(os.path.join(d, "a.ply"))
        assert axyz.shape == (2, 3) and argb is None and axyz[1, 2] == 3
        assert cup.points["voxel"] == 0.01 and sc.objects[1].points is None

        # crop panel: RGB with mask outline, grayscale depth (near bright) + stats
        crop = sgviz.CropCache().get(d, cup.rgbd)
        # panel upscales small crops by an integer factor (28 px -> x8 = 224 rows)
        assert crop["rgb"].shape == (224, 160, 3), crop["rgb"].shape
        r, dep = crop["rgb"][::8, ::8], crop["depth"][::8, ::8]
        assert crop["mask"] and r.shape == (28, 20, 3) and dep.shape == (28, 20, 3)
        assert tuple(r[14, 10]) == (200, 200, 200)       # inside mask untouched
        assert r[2, 2, 0] < 150                           # outside dimmed
        assert tuple(r[6, 8]) == (255, 210, 0)            # outline
        assert (dep[..., 0] == dep[..., 1]).all() and (dep[..., 1] == dep[..., 2]).all()  # gray
        assert dep[14, 4, 0] > dep[14, 15, 0]             # nearer column brighter
        assert dep[10, 1, 0] < 100 and dep[0, 5, 0] == 0  # outside mask dark, invalid black
        mn, md, mx = crop["stats"]
        assert abs(mn - 1.20) < 1e-6 and abs(mx - 1.31) < 1e-6 and 1.2 < md < 1.31
        no = sgviz.CropCache().get(d, {"rgb": "objects/missing.png"})
        assert no["rgb"] is None and no["depth"] is None
        # full-frame image + box -> cropped (8 px margin); exact-size crop -> untouched
        assert sgviz.maybe_crop(np.zeros((100, 100)), [10, 12, 30, 40]).shape == (44, 36)
        assert sgviz.maybe_crop(np.zeros((28, 20)), [110, 112, 130, 140]).shape == (28, 20)

        # map
        meta = sgviz.read_map_yaml(os.path.join(d, "map.yaml"))
        assert meta["resolution"] == 0.05 and meta["origin"] == [-0.5, -0.5, 0.0]
        tex = sgviz.map_texture(np.array(Image.open(os.path.join(d, "map.pgm"))), meta)
        assert tuple(tex[10, 0]) == (0, 0, 0) and tuple(tex[6, 6]) == (254, 254, 254)
        assert tuple(tex[0, 0]) == (150, 150, 165)

        # diff: identical -> empty; move cup -> redraw cup
        assert sgviz.diff_scenes(sc, sgviz.load_scene(d)).empty()
        d0 = sgviz.diff_scenes(None, sc)
        assert sorted(d0.added) == [1, 2, 3]
        write_dir(d, cup_pos=(1.5, 0.5, 0.8), with_edge=False)
        sc2 = sgviz.load_scene(d)
        df = sgviz.diff_scenes(sc, sc2)
        assert df.redraw == [2] and df.changed == [2] and not df.added and not df.removed, df

        # no rgbd / no metadata at all must not crash
        G = dsg.DynamicSceneGraph()
        a = dsg.ObjectNodeAttributes()
        a.position = np.zeros(3)
        G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 9), a)
        s3 = sgviz.scene_from_graph(G)
        assert s3.objects[9].state == "seen" and s3.objects[9].bbox_dims is None
        assert s3.robot_pose is None

        server_check(d)

    pick_check()
    print("test_sgviz: OK")


def pick_check():
    """Ray picking: hits, nearer-along-ray wins, misses, timing."""
    import time

    rng = np.random.default_rng(3)
    P = sgviz.PickIndex()
    a = (np.array([2.0, 0.0, 0.5]) + rng.uniform(-0.1, 0.1, (1000, 3))).astype(np.float32)
    b = (np.array([4.0, 0.0, 0.5]) + rng.uniform(-0.1, 0.1, (1000, 3))).astype(np.float32)
    P.set_points(10, a, 0.02)
    P.set_points(11, b, 0.02)
    P.set_sphere(12, (0.0, 3.0, 0.5), 0.09)
    o = (0.0, 0.0, 0.5)
    assert P.pick(o, (1, 0, 0)) == 10            # both on the ray: nearer one along it
    assert P.pick((6.0, 0.0, 0.5), (-1, 0, 0)) == 11  # from the other side
    assert P.pick((2.0, -3.0, 0.5), (0, 1, 0)) == 10  # sideways through a
    assert P.pick((4.0, 0.0, 3.0), (0, 0, -2)) == 11  # unnormalised direction ok
    assert P.pick(o, (0, 1, 0)) == 12            # fallback sphere
    assert P.pick(o, (0, -1, 0)) is None         # miss
    assert P.pick(o, (1, 0, 1)) is None          # passes over both
    assert P.pick((3.0, 0.0, 0.5), (1, 0, 0)) == 11  # a is behind the origin
    assert P.pick(o, (0, 0, 0)) is None
    # grazing: a point 1.5 cm off the ray is hit with radius 2 cm, 3 cm is not
    Q = sgviz.PickIndex()
    Q.set_points(1, np.array([[1.0, 0.015, 0.0]], np.float32), 0.02)
    Q.set_points(2, np.array([[0.5, 0.03, 0.0]], np.float32), 0.02)
    assert Q.pick((0, 0, 0), (1, 0, 0)) == 1
    P.remove(10)
    assert P.pick(o, (1, 0, 0)) == 11

    # timing: 50 objects x 4000 points, rays that pass through many AABBs
    T = sgviz.PickIndex()
    for i in range(50):
        c = np.array([i * 0.3, 0.0, 0.5])
        T.set_points(i, (c + rng.uniform(-0.15, 0.15, (4000, 3))).astype(np.float32), 0.02)
    rays = [((-2.0, 0.0, 0.5), (1.0, 0.0, 0.0)),        # through all 50 boxes
            ((7.5, -5.0, 3.0), (0.0, 1.0, -0.5)),      # one object
            ((-2.0, 5.0, 5.0), (1.0, 0.1, 0.0)),       # misses everything
            ((17.0, 0.0, 0.5), (-1.0, 0.0, 0.0))]      # worst case: every object fully tested
    for r in rays:
        T.pick(*r)  # warm up
    dt = {}
    for k, r in enumerate(rays):
        t0 = time.perf_counter()
        for _ in range(20):
            T.pick(*r)
        dt[k] = (time.perf_counter() - t0) / 20 * 1e3
    assert T.pick(*rays[0]) == 0 and T.pick(*rays[2]) is None and T.pick(*rays[3]) == 49
    print("pick ms (through-all / one / miss / worst 50x4000): "
          + ", ".join(f"{v:.2f}" for v in dt.values()))
    assert max(dt.values()) < 5.0, dt


def server_check(d):
    """Run the real viewer (viser server, no browser) on the synthetic dir."""
    import socket
    import urllib.request

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    v = sgviz.SgViewer(d, ip="127.0.0.1", port=port)
    try:
        write_dir(d)  # back to the version with the (ignored) old edge
        df = v.poll_once()
        assert sorted(df.added) == [1, 2, 3]
        assert v.poll_once() is None  # unchanged mtime -> no work
        assert set(v._node_handles) == {1, 2, 3} and not hasattr(v, "_edge_handles")
        names = lambda oid: sorted(h.name.rsplit("/", 1)[1] for h in v._node_handles[oid])
        assert "points" in names(2) and "bbox" not in names(2), names(2)
        assert "points" not in names(1) and "bbox" not in names(1)  # boxes default off
        v.g_boxes.value = True
        v._redraw_all()
        assert "bbox" in names(1) and "bbox" not in names(2)  # box only without points
        v.g_boxes.value = False
        v._redraw_all()
        assert v._refresh_points() == []
        write_ply(os.path.join(d, "objects", "O2_points.ply"), *cup_points((1.0, 0.5, 0.8), n=50, seed=1))
        assert v._refresh_points() == [2]
        assert v._refresh_points() == []
        assert "cup#2" in v.g_select.options
        v.select(2)
        assert v.g_select.value == "cup#2"
        txt = v.g_info.content
        for frag in ("cup", "moved", "n_obs | 7", "first_pos", "box_px", "1.234"):
            assert frag in txt, (frag, txt)
        assert v.g_rgb.visible and v.g_depth.visible and v.g_dstats.visible
        assert "1.200 / " in v.g_dstats.content and "inside mask" in v.g_dstats.content
        assert "200 pts" in txt
        v.select(1)
        assert not v.g_rgb.visible and not v.g_dstats.visible  # table has no rgbd

        # click picking through the viewer: no hidden centre sphere on the cup
        assert "node" not in names(2) and "node" in names(1)
        cup_c = (1.0, 0.5, 0.8)
        assert v.on_scene_click((cup_c[0], cup_c[1] - 3, cup_c[2]), (0, 1, 0)) == 2
        assert v.selected == 2 and v.g_select.value == "cup#2" and v.g_rgb.visible
        assert v._sel_handle is None  # cloud highlighted by tint, no wireframe sphere
        # straight down: cup (z 0.8) is above table sphere (z 0.4) -> cup first
        assert v.on_scene_click((1.0, 0.5, 5.0), (0, 0, -1)) == 2
        # ray only through the table's fallback sphere
        assert v.on_scene_click((1.0, -3.0, 0.4), (0, 1, 0)) == 1 and v.selected == 1
        assert v._sel_handle is not None  # sphere objects keep the wireframe highlight
        # a miss keeps the current selection
        assert v.on_scene_click((9.0, 9.0, 9.0), (0, 0, 1)) is None and v.selected == 1
        # gone objects hidden -> no longer pickable
        v.g_gone.value = False
        v._redraw_all()
        assert 3 not in v.picker
        v.g_gone.value = True
        v._redraw_all()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
    finally:
        v.server.stop()


if __name__ == "__main__":
    main()
