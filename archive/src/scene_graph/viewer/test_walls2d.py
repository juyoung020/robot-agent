"""walls2d: wall segments and the robot-frame wall state (pure numpy, no viser). Run: python test_walls2d.py"""
import math

import numpy as np

import walls2d as W

RES = 0.05
ORIGIN = (-1.0, -1.0)


def room(w_m=6.0, h_m=4.0, t=0.1):
    """Rectangular room: outer wall ring of thickness t. PGM convention: row 0 = max y, 0 = occupied, 254 = free."""
    nx, ny = int(w_m / RES), int(h_m / RES)
    pgm = np.full((ny, nx), 254, np.uint8)
    k = int(round(t / RES))
    pgm[:k, :] = 0
    pgm[-k:, :] = 0
    pgm[:, :k] = 0
    pgm[:, -k:] = 0
    return pgm


def main():
    pgm = room()
    occ = W.occupied_mask(pgm)
    segs = W.wall_segments(occ, RES, ORIGIN)
    # a closed rectangle: 2 horizontal (6 m) + 2 vertical (4 m) centre lines
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    horiz = L[np.abs(segs[:, 3] - segs[:, 1]) < 1e-9]
    vert = L[np.abs(segs[:, 2] - segs[:, 0]) < 1e-9]
    assert len(horiz) == 2 and len(vert) == 2, (len(horiz), len(vert))
    assert all(abs(x - 5.9) < 0.2 for x in horiz) and all(abs(x - 3.9) < 0.2 for x in vert), (horiz, vert)
    # a free-standing 0.2 m x 0.2 m block is not a wall (shorter than MIN_LEN)
    pgm2 = pgm.copy()
    pgm2[40:44, 60:64] = 0
    assert len(W.wall_segments(W.occupied_mask(pgm2), RES, ORIGIN)) == 4
    # state vector layout
    pose = (2.0, 1.0, 0.3)  # inside the room (x -1..5, y -1..3)
    v = W.wall_state_vector(occ, RES, ORIGIN, segs, pose)
    assert v.shape == (W.N_SECTORS + W.K_SEGMENTS * 5,) and v.dtype == np.float32
    rays = v[: W.N_SECTORS]
    assert rays.min() > 0 and rays.max() <= 1.0
    # distance straight ahead (yaw 0, facing +x) = distance to the right wall: x_wall_inner = 5 - 0.1 -> 2.9 m from x = 2
    ahead = W.ray_distances(occ, RES, ORIGIN, (2.0, 1.0, 0.0))[0]
    assert abs(ahead - 2.9) < 0.1, ahead
    # rotating the robot by one sector shifts the ray vector by one slot (rotation consistency)
    v1 = W.ray_distances(occ, RES, ORIGIN, pose)
    v2 = W.ray_distances(occ, RES, ORIGIN, (pose[0], pose[1], pose[2] + 2 * math.pi / W.N_SECTORS))
    assert np.allclose(np.roll(v1, -1), v2, atol=0.06), (v1, v2)
    # nearest segment slots are valid, sorted by distance, in the robot frame; unused slots are zero
    blk = v[W.N_SECTORS:].reshape(W.K_SEGMENTS, 5)
    assert blk[:4, 4].tolist() == [1.0] * 4 and blk[4:].sum() == 0, blk
    seg_r, dist = W.segments_robot_frame(segs, pose)
    assert np.all(np.diff(dist[:4]) >= -1e-6)
    # moving the robot (translation) changes the robot-frame segments but not the map-frame segments
    seg_r2, _ = W.segments_robot_frame(segs, (pose[0] + 1.0, pose[1], pose[2]))
    assert not np.allclose(seg_r, seg_r2)
    # empty map -> all rays at max range, no segments
    e = W.wall_state_vector(np.zeros((40, 40), bool), RES, ORIGIN, np.zeros((0, 4)), (0.0, 0.0, 0.0))
    assert np.allclose(e[: W.N_SECTORS], 1.0) and e[W.N_SECTORS:].sum() == 0
    print("walls2d OK:", len(segs), "segments, vector length", len(v))


if __name__ == "__main__":
    main()
