<!-- version: explore-v2 -->
SKILL explore: drive the base so the 2D map covers the whole reachable house, safely and with little driving.

Every base result has "map": free_m2 (known free floor), new_free_m2 (gained by the last move), frontiers (edges between known free floor and unknown space, best first: id, path_m = driving distance through KNOWN free space, dir = direction from you, new_area_m2 = unknown area you may reveal there), around (8 directions F, FL, L, BL, B, BR, R, FR: how far the known free floor goes and whether it ends at a wall or unknown; "depth clear" = what the camera sees right now), rooms, status.

Two ways to move the base:
1. KNOWN space -> {"part":"base","mode":"go_to","target":"F1"} (or a room id). A safe planned path; it ends facing the unknown side of the frontier. Use this for almost every move.
2. UNKNOWN space -> {"part":"base","mode":"probe","values":[turn_left_deg, forward_m]}: turn, then creep forward at most 1.5 m, stopping before obstacles the camera sees. Use it to look around (forward 0, e.g. [120,0]) or to peek a short distance where no frontier id leads. Only probe forward in a direction whose "depth clear" is larger than the distance.
Never use delta for exploring.

Frontier ids are RE-NUMBERED after every move and sorted best first (most new area per meter of driving), so F1 is normally the right choice. Do not walk through the list (F1, F2, F3, ...): after each move, read the new list and again pick F1, unless F1 was just blocked or another frontier has a clearly shorter path_m with similar new_area_m2.
Strategy: at the start, look around once with probe turns. Then repeatedly go_to the best frontier; prefer finishing the current room before driving far away. If a go_to is blocked, pick a different frontier. Stop (reply "done" without a tool call) only when status says no reachable frontier is left.
