"""Headless check of sglayers (Hydra-style stacked layers): parse view.json graph, draw into a real viser server,
re-poll without change sends nothing, a changed place layer re-sends only that layer.

    ~/sdsg_venv/bin/python src/scene_graph/viewer/test_sglayers.py
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viser  # noqa: E402

from sglayers import LayerView, parse_layers  # noqa: E402


def view(n_places=5, clear=0.5):
    nodes = [{"id": f"p{i}", "kind": "place", "pos": [i * 1.0, 0.0], "clear": clear, "frontier": i == 0} for i in range(n_places)]
    nodes += [{"id": "a0", "kind": "agent", "pos": [0, 0], "yaw": 0, "t": 0}]
    edges = [[f"p{i}", f"p{i + 1}", "place", clear] for i in range(n_places - 1)]
    edges += [["R1", "p0", "parent", 1], ["p1", "O3", "parent", 1], ["O3", "O4", "on", 1], ["p0", "a0", "parent", 1]]
    return {"objects": [{"id": 3, "pos": [1, 0, 0.8], "name": "cup", "state": "seen"}, {"id": 4, "pos": [1, 0, 0.4], "name": "table", "state": "seen"}],
            "rooms": [{"id": 1, "centroid": [2, 0], "name": "kitchen", "color": [200, 100, 50]}],
            "graph": {"nodes": nodes, "edges": edges}}


def main():
    L = parse_layers(view())
    assert len(L["nodes"]) == 9 and len(L["edges"]) == 8, (len(L["nodes"]), len(L["edges"]))
    d = tempfile.mkdtemp()
    srv = viser.ViserServer(port=8093, verbose=False)
    lv = LayerView(srv, d)
    p = os.path.join(d, "view.json")
    json.dump(view(), open(p, "w"))
    assert lv.poll()
    names = set(lv._handles)
    for want in ("/graph/place_nodes", "/graph/room_nodes", "/graph/object_nodes", "/graph/agent_nodes",
                 "/graph/edges_place", "/graph/edges_inter"):
        assert want in names, (want, names)
    assert "/graph/edges_on" not in names, "object-object prepositions in old files must not be drawn"
    assert lv.z_of(L["nodes"]["p0"]) < lv.z_of(L["nodes"]["R1"])
    before = dict(lv._handles)
    assert not lv.poll()  # unchanged file: nothing
    time.sleep(0.01)
    json.dump(view(n_places=6), open(p, "w"))
    assert lv.poll()
    same = [k for k in before if lv._handles.get(k) is before[k]]
    assert "/graph/room_nodes" in same and "/graph/place_nodes" not in same, same
    print("sglayers OK:", lv.status())
    srv.stop()


if __name__ == "__main__":
    main()
