// rasc_test <file.rasc>...  — round trip + invariants for RASC files written by b1kconv.
// Next to each X.rasc the converter writes X.expect (section counts + FNV-1a 64 of the bytes it
// wrote); this test checks that the C++ reader sees exactly those bytes, then checks invariants
// (object counts, AABB sanity, room coverage, predicate references), then that corrupted copies
// are rejected (negative control).
#include "rasc.h"

#include <cinttypes>
#include <fstream>
#include <map>
#include <sstream>

static int g_fail = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            if (g_fail < 40) {                          \
                std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
                std::printf(__VA_ARGS__);               \
                std::printf("\n");                      \
            }                                           \
            g_fail++;                                   \
        }                                               \
    } while (0)

static bool finite3(const float* v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

static void test_file(const std::string& path) {
    using namespace rasc;
    int f0 = g_fail;
    Scene s;
    std::string err;
    if (!load(s, path.c_str(), &err)) {
        CHECK(false, "load %s: %s", path.c_str(), err.c_str());
        return;
    }
    const RascFileHeader& h = *s.h;

    // ---- 1. expect file: section counts + byte hashes (written by the converter)
    std::string ep = path.substr(0, path.size() - 5) + ".expect";
    std::ifstream ef(ep);
    CHECK(ef.good(), "missing %s", ep.c_str());
    std::string line;
    uint64_t n_doc_lits_expect = 0;
    int nsec = 0;
    while (std::getline(ef, line)) {
        std::istringstream is(line);
        std::string k;
        is >> k;
        if (k == "scene") {
            std::string n;
            is >> n;
            CHECK(n == s.name(), "scene name %s != %s", s.name().c_str(), n.c_str());
        } else if (k == "layout_hash") {
            uint64_t v;
            is >> v;
            CHECK(v == RASC_LAYOUT_HASH, "layout hash");
        } else if (k == "sec") {
            std::string name, hx;
            uint64_t count;
            is >> name >> count >> hx;
            int id = -1;
            for (int i = 0; i < RASC_NSEC; i++)
                if (name == rasc_sec_names[i]) id = i;
            CHECK(id >= 0, "unknown section %s", name.c_str());
            if (id < 0) continue;
            const RascSec& e = h.sections[id];
            CHECK(e.count == count, "%s count %u != %" PRIu64, name.c_str(), e.count, count);
            uint64_t hv = fnv64(s.buf.data() + e.offset, e.bytes);
            CHECK(hv == std::stoull(hx, nullptr, 16), "%s bytes hash differs", name.c_str());
            nsec++;
        } else if (k == "n_doc_lits") {
            is >> n_doc_lits_expect;
        }
    }
    CHECK(nsec == RASC_NSEC, "expect lists %d sections", nsec);

    // ---- 2. round trip: write the loaded bytes out, read back, same bytes and same views
    {
        std::string tp = path + ".rt.tmp";
        FILE* f = std::fopen(tp.c_str(), "wb");
        CHECK(f, "cannot write %s", tp.c_str());
        if (f) {
            std::fwrite(s.buf.data(), 1, s.buf.size(), f);
            std::fclose(f);
            Scene r;
            CHECK(load(r, tp.c_str(), &err), "reload: %s", err.c_str());
            CHECK(r.buf == s.buf, "round trip bytes differ");
            CHECK(r.objs.n == s.objs.n && r.insts.n == s.insts.n && r.lits.n == s.lits.n, "round trip counts differ");
            std::remove(tp.c_str());
        }
    }

    // ---- 3. object counts
    size_t n_floor_ceil = 0, n_door = 0, cat_sum = 0;
    for (auto& o : s.objs) {
        if (o.flags & (RASC_F_FLOOR | RASC_F_CEILING)) n_floor_ceil++;
        if (o.flags & RASC_F_DOOR) n_door++;
    }
    for (auto& c : s.cats) cat_sum += c.n_scene;
    CHECK(s.boxes.n == s.objs.n - n_floor_ceil, "boxes %zu != objs %zu - floors/ceilings %zu", s.boxes.n, s.objs.n, n_floor_ceil);
    CHECK(s.doors.n == n_door, "doors %zu != door-flagged objects %zu", s.doors.n, n_door);
    CHECK(cat_sum == s.objs.n, "category n_scene sum %zu != objects %zu", cat_sum, s.objs.n);
    CHECK(h.n_levels == 1, "levels %u", h.n_levels);

    // ---- 4. AABB sanity
    const float gx0 = h.origin[0], gy0 = h.origin[1], gx1 = gx0 + h.grid_w * h.cell, gy1 = gy0 + h.grid_h * h.cell;
    size_t outside = 0;
    for (size_t i = 0; i < s.objs.n; i++) {
        const RascObjRec& o = s.objs[i];
        CHECK(o.cat < s.cats.n, "obj %zu cat", i);
        CHECK(o.flags & RASC_F_HAS_BBOX, "obj %s has no bbox", s.str(o.name));
        CHECK(finite3(o.pos) && finite3(o.center) && finite3(o.half) && finite3(o.aabb_min) && finite3(o.aabb_max), "obj %s not finite", s.str(o.name));
        float qn = std::sqrt(o.quat[0] * o.quat[0] + o.quat[1] * o.quat[1] + o.quat[2] * o.quat[2] + o.quat[3] * o.quat[3]);
        CHECK(std::fabs(qn - 1) < 1e-3f, "obj %s quat norm %f", s.str(o.name), qn);
        float hmax = 0, wmax = 0;
        for (int k = 0; k < 3; k++) {
            CHECK(o.half[k] >= 0, "obj %s half[%d] < 0", s.str(o.name), k);
            CHECK(o.aabb_min[k] <= o.center[k] && o.center[k] <= o.aabb_max[k], "obj %s center outside aabb", s.str(o.name));
            hmax = std::fmax(hmax, o.half[k]);
            wmax = std::fmax(wmax, 0.5f * (o.aabb_max[k] - o.aabb_min[k]));
        }
        CHECK(hmax > 0, "obj %s zero box", s.str(o.name));
        CHECK(wmax + 1e-4f >= hmax - 1e-4f * hmax || wmax * 1.7321f + 1e-4f >= hmax, "obj %s aabb smaller than box", s.str(o.name));
        CHECK(o.joint_off + o.n_joints <= s.joints.n, "obj %s joints", s.str(o.name));
        CHECK(o.in_rooms_off + o.n_in_rooms <= s.in_rooms.n, "obj %s in_rooms", s.str(o.name));
        CHECK(o.room_a == RASC_NONE16 || o.room_a < s.rooms.n, "obj %s room_a", s.str(o.name));
        CHECK(o.room_b == RASC_NONE16 || o.room_b < s.rooms.n, "obj %s room_b", s.str(o.name));
        if (o.center[0] < gx0 || o.center[0] > gx1 || o.center[1] < gy0 || o.center[1] > gy1) outside++;
    }
    CHECK(outside == 0, "%zu object centers outside the layout grid", outside);
    for (auto& b : s.boxes) {
        CHECK(b.obj < s.objs.n, "box obj");
        const RascObjRec& o = s.objs[b.obj];
        CHECK(b.zmin <= b.zmax && b.zmin == o.aabb_min[2] && b.zmax == o.aabb_max[2], "box z");
        CHECK(Scene::in_box(b, o.center[0], o.center[1], o.center[2]), "box %s does not contain its center", s.str(o.name));
    }
    for (auto& d : s.doors) CHECK(d.obj < s.objs.n && (s.objs[d.obj].flags & RASC_F_DOOR), "door obj");

    // ---- 5. room coverage
    CHECK(s.rooms.n == h.n_rooms_png, "rooms %zu != png ids %u", s.rooms.n, h.n_rooms_png);
    std::vector<uint32_t> cells(s.rooms.n, 0);
    size_t room_cells = 0, free_cells = 0, free_in_room = 0;
    for (uint32_t r = 0; r < h.grid_h; r++)
        for (uint32_t c = 0; c < h.grid_w; c++) {
            uint8_t v = s.room_grid[(size_t)r * h.grid_w + c];
            CHECK(v <= s.rooms.n, "grid room id %u", v);
            if (v && v <= s.rooms.n) { cells[v - 1]++; room_cells++; }
            if (s.free_cell(0, r, c)) { free_cells++; free_in_room += v != 0; }
        }
    for (size_t i = 0; i < s.rooms.n; i++) {
        const RascRoomRec& r = s.rooms[i];
        CHECK(cells[i] == r.n_cells, "room %s n_cells %u != grid %u", s.str(r.name), r.n_cells, cells[i]);
        CHECK(r.n_cells > 0, "room %s has no cell", s.str(r.name));
        CHECK(r.sem >= 1 && r.sem <= 39, "room %s sem %u", s.str(r.name), r.sem);
        int at = s.room_at(r.centroid[0], r.centroid[1]);
        (void)at;
    }
    // every in_rooms name of an object is a room of the map (unresolved ones counted, not failed)
    size_t unresolved = 0;
    for (auto& o : s.objs)
        for (uint32_t k = 0; k < o.n_in_rooms; k++) {
            const char* n = s.str(s.in_rooms[o.in_rooms_off + k]);
            bool found = false;
            for (auto& r : s.rooms) found |= std::strcmp(s.str(r.name), n) == 0;
            unresolved += !found;
        }
    double cover = free_cells ? (double)free_in_room / free_cells : 0;  // info: exteriors (yards, parking) are traversable but in no room
    // objects placed in a room (in_rooms) sit on a cell of that room (furniture, not walls/doors/windows/floors/ceilings)
    size_t placed = 0, placed_ok = 0, placed_near = 0;
    for (auto& o : s.objs) {
        if (o.room_a == RASC_NONE16 || (o.flags & (RASC_F_WALL | RASC_F_DOOR | RASC_F_WINDOW | RASC_F_FLOOR | RASC_F_CEILING))) continue;
        int at = s.room_at(o.center[0], o.center[1]);
        placed++;
        bool on = at >= 0 && (at == o.room_a || at == o.room_b);
        placed_ok += on;
        if (on) { placed_near++; continue; }
        // off its room (wall-mounted, room edge, stair hole): nearest own-room cell within 0.35 m + box half?
        int r0, c0;
        if (!s.cell(o.center[0], o.center[1], &r0, &c0)) continue;
        float lim = 0.35f + std::fmax(o.half[0], o.half[1]);
        int rad = (int)std::ceil(lim / h.cell);
        bool near = false;
        for (int r = r0 - rad; r <= r0 + rad && !near; r++)
            for (int c = c0 - rad; c <= c0 + rad && !near; c++) {
                if (r < 0 || c < 0 || r >= (int)h.grid_h || c >= (int)h.grid_w) continue;
                int v = s.room_grid[(size_t)r * h.grid_w + c] - 1;
                near = (v == o.room_a || v == o.room_b) && std::hypot(r - r0, c - c0) * h.cell <= lim;
            }
        placed_near += near;
    }
    double placed_frac = placed ? (double)placed_ok / placed : 1, near_frac = placed ? (double)placed_near / placed : 1;
    CHECK(placed_frac >= 0.90, "only %.3f of room-tagged objects lie on a cell of their room", placed_frac);
    CHECK(near_frac >= 0.98, "only %.3f of room-tagged objects lie within 0.35 m + half of their room", near_frac);

    // ---- 6. tasks, predicates, instances
    uint64_t n_doc = 0;
    size_t robot_free = 0;
    for (size_t ti = 0; ti < s.tasks.n; ti++) {
        const RascTaskRec& t = s.tasks[ti];
        const char* tn = s.str(t.name);
        CHECK(t.obj_off + t.n_obj <= s.task_objs.n && t.init_off + t.n_init <= s.lits.n && t.goal_off + t.n_goal <= s.lits.n, "%s ranges", tn);
        CHECK(t.var_off + t.n_var <= s.vars.n && t.cand_off + t.n_cand <= s.cands.n && t.removed_off + t.n_removed <= s.removed.n, "%s ranges", tn);
        CHECK(t.inst_off + t.n_inst <= s.insts.n && t.n_inst == t.n_train + t.n_public, "%s inst ranges", tn);
        CHECK(t.n_public == 20 && t.n_train == 300, "%s instances %u + %u", tn, t.n_train, t.n_public);
        CHECK(t.task_index < 100, "%s index", tn);
        CHECK(s.rooms.n >= 64 || (t.rooms_mask >> s.rooms.n) == 0, "%s rooms_mask", tn);
        for (uint32_t k = 0; k < t.n_removed; k++) CHECK(s.removed[t.removed_off + k] < s.objs.n, "%s removed", tn);
        for (auto& o : s.objs_of(t)) {
            CHECK(o.cat == RASC_NONE16 || o.cat < s.cats.n, "%s obj cat", tn);
            CHECK(o.scene_obj < (int32_t)s.objs.n, "%s scene obj", tn);
            CHECK(((o.flags & RASC_F_SCENE_OBJ) != 0) == (o.scene_obj >= 0), "%s scene obj flag", tn);
        }
        for (auto& v : s.vars_of(t)) {
            CHECK(v.kind >= 1 && v.kind <= 3 && v.cand_n > 0 && v.cand_off + v.cand_n <= t.n_cand, "%s var", tn);
            for (uint32_t k = 0; k < v.cand_n; k++) CHECK(s.cands[t.cand_off + v.cand_off + k] < t.n_obj, "%s cand", tn);
        }
        for (int part = 0; part < 2; part++) {
            View<RascLitRec> L = part ? s.goal_of(t) : s.init_of(t);
            for (auto& l : L) {
                CHECK(l.pred >= 1 && l.pred < RASC_NPRED, "%s pred %u", tn, l.pred);
                CHECK(l.nargs >= 1 && l.nargs <= 2, "%s nargs", tn);
                CHECK(l.or_group == RASC_NONE16 || l.or_group < t.n_or_groups, "%s or group", tn);
                if (part) n_doc += (l.flags & RASC_L_DOC) != 0;
                for (int k = 0; k < l.nargs; k++) {
                    uint16_t a = l.arg[k];
                    if (a & RASC_ARG_ROOM) CHECK(l.pred == RASC_P_INROOM && k == 1 && (a & ~RASC_ARG_ROOM) >= 1 && (a & ~RASC_ARG_ROOM) <= 39, "%s room arg", tn);
                    else if (a & RASC_ARG_VAR) CHECK(part == 1 && (a & ~RASC_ARG_VAR) < t.n_var, "%s var arg", tn);
                    else CHECK(a < t.n_obj, "%s obj arg %u >= %u", tn, a, t.n_obj);
                }
                if (l.pred == RASC_P_INROOM) CHECK(l.nargs == 2 && (l.arg[1] & RASC_ARG_ROOM), "%s inroom arg", tn);
            }
        }
        for (auto& in : s.insts_of(t)) {
            CHECK(in.task == ti && in.n_pose == t.n_obj && in.pose_off + in.n_pose <= s.poses.n, "%s inst ranges", tn);
            CHECK(finite3(in.robot_pos), "%s robot pos", tn);
            int r, c;
            CHECK(s.cell(in.robot_pos[0], in.robot_pos[1], &r, &c), "%s robot start outside grid", tn);
            robot_free += s.free_at(0, in.robot_pos[0], in.robot_pos[1]);
            auto P = s.poses_of(in);
            auto O = s.objs_of(t);
            for (size_t k = 0; k < P.n; k++) {
                const RascPoseRec& p = P[k];
                CHECK(p.src <= 3, "%s pose src", tn);
                bool none = O[k].flags & (RASC_F_AGENT | RASC_F_SYSTEM | RASC_F_UNMAPPED | RASC_F_WILDCARD);
                CHECK(none == (p.src == 0), "%s pose src vs flags (%s)", tn, s.str(O[k].inst));
                if (p.src) {
                    float qn = std::sqrt(p.quat[0] * p.quat[0] + p.quat[1] * p.quat[1] + p.quat[2] * p.quat[2] + p.quat[3] * p.quat[3]);
                    CHECK(finite3(p.pos) && std::fabs(qn - 1) < 1e-3f, "%s pose", tn);
                    CHECK(p.joint_off + p.n_joints <= s.joints.n, "%s pose joints", tn);
                }
            }
        }
    }
    // ---- 6b. pick-and-place table
    const RascLimitsRec& L = s.limits[0];
    const RascLimitsRec& LI = s.limits[1];
    size_t pairs_seen = 0, picks_seen = 0, places_seen = 0;
    for (size_t ri = 0; ri < s.pnp_ranges.n; ri++) {
        const RascPnpRange& r = s.pnp_ranges[ri];
        bool scene = ri + 1 == s.pnp_ranges.n;
        uint32_t inst = scene ? RASC_NONE32 : (uint32_t)ri;
        CHECK(r.pick_off + r.n_pick <= s.picks.n && r.place_off + r.n_place <= s.places.n && r.pair_off + r.n_pair <= s.pairs.n, "pnp range %zu", ri);
        picks_seen += r.n_pick;
        places_seen += r.n_place;
        pairs_seen += r.n_pair;
        for (uint32_t k = 0; k < r.n_pick; k++) {
            const RascPickRec& p = s.picks[r.pick_off + k];
            CHECK(p.inst == inst, "pick inst");
            CHECK(p.z0 <= L.pick_z && p.min_w <= L.max_w && (std::isnan(p.mass) ? (p.flags & RASC_PK_MASS_UNKNOWN) != 0 : p.mass <= L.max_mass), "pick outside outer limits");
            bool side = p.z0 > L.topdown_z;
            CHECK(side == ((p.flags & RASC_PK_SIDE_GRASP) != 0), "side grasp flag");
            CHECK(!side || std::isnan(p.edge_d) || p.edge_d <= L.edge_dist, "side grasp beyond the surface edge limit");
            CHECK(std::isnan(p.edge_d) == ((p.flags & RASC_PK_EDGE_UNKNOWN) != 0) || !side, "edge unknown flag");
            bool inner = p.z0 <= LI.pick_z && p.min_w <= LI.max_w && (std::isnan(p.mass) || p.mass <= LI.max_mass) &&
                         (p.z0 <= LI.topdown_z || (!std::isnan(p.edge_d) && p.edge_d <= LI.edge_dist));
            CHECK(inner == ((p.flags & RASC_PK_INNER) != 0), "pick inner flag");
            CHECK(p.src_place == RASC_NONE32 || p.src_place < s.places.n, "pick src");
            if (p.obj & RASC_PNP_TASKOBJ) {
                CHECK(!scene && (p.obj & ~RASC_PNP_TASKOBJ) < s.task_objs.n, "pick task obj");
            } else {
                CHECK(p.obj < s.objs.n, "pick obj");
            }
            CHECK(p.room == RASC_NONE16 || p.room < s.rooms.n, "pick room");
        }
        for (uint32_t k = 0; k < r.n_place; k++) {
            const RascPlaceRec& q = s.places[r.place_off + k];
            CHECK(q.inst == inst && q.kind >= 1 && q.kind <= 3, "place kind/inst");
            if (q.kind == 1) CHECK(q.top <= L.place_top && q.top >= L.min_top && 2 * std::fmin(q.half[0], q.half[1]) >= L.min_side - 1e-6f, "ontop support outside limits");
            if (q.kind == 2) CHECK(q.top <= L.place_top + L.inside_margin, "container outside limits");
            if (q.kind == 3) CHECK(q.obj == RASC_NONE32 && q.room < s.rooms.n && q.top == 0, "floor support");
        }
        for (uint32_t k = 0; k < r.n_pair; k++) {
            const RascPairRec& a = s.pairs[r.pair_off + k];
            CHECK(a.pick >= r.pick_off && a.pick < r.pick_off + r.n_pick, "pair pick outside its range");
            CHECK(a.dst < s.places.n && a.dst != a.src, "pair dst");
            const RascPickRec& p = s.picks[a.pick];
            const RascPlaceRec& d = s.places[a.dst];
            CHECK(d.inst == inst || d.inst == RASC_NONE32, "pair dst level");
            CHECK(a.src == p.src_place, "pair src");
            CHECK(a.rel == (d.kind == 2 ? RASC_P_INSIDE : RASC_P_ONTOP), "pair rel");
            CHECK(((a.reachable & 1) != 0) == (p.comp != RASC_NONE16 && p.comp == d.comp), "pair reachable bit");
            CHECK(((a.reachable & 4) == 0) || (!scene && r.robot_comp == p.comp), "pair robot bit");
            CHECK(((a.reachable & 8) != 0) == (!scene && r.robot_comp_inner != RASC_NONE16 && r.robot_comp_inner == p.comp_inner && p.comp_inner == d.comp_inner), "pair inner robot bit");
            CHECK(a.room_pick == p.room && a.room_dst == d.room, "pair rooms");
        }
    }
    // floor heights: known under every room cell, sane range
    size_t fz_room = 0, fz_room_known = 0;
    for (size_t i = 0; i < s.floor_z.n; i++) {
        if (s.floor_z[i] != INT16_MIN) CHECK(s.floor_z[i] > -2000 && s.floor_z[i] < 2000, "floor z %d", s.floor_z[i]);
        if (s.room_grid[i]) { fz_room++; fz_room_known += s.floor_z[i] != INT16_MIN; }
    }
    CHECK(fz_room_known == fz_room, "floor height unknown under %zu room cells", fz_room - fz_room_known);
    CHECK(L.threshold >= LI.threshold && L.max_w >= LI.max_w && L.pick_z >= LI.pick_z, "outer limits narrower than inner");
    CHECK(picks_seen == s.picks.n && places_seen == s.places.n && pairs_seen == s.pairs.n, "pnp ranges do not tile the sections");

    CHECK(n_doc == n_doc_lits_expect,"doc literals %" PRIu64 " != %" PRIu64, n_doc, n_doc_lits_expect);

    std::printf("%-26s %s  %8zu B  objs %zu boxes %zu doors %zu rooms %zu (unresolved in_rooms %zu) trav-in-room %.3f obj-in-own-room %zu/%zu (near %zu)  tasks %zu lits %zu insts %zu robot-on-free %zu/%zu\n",
                s.name().c_str(), g_fail == f0 ? "ok  " : "FAIL", s.buf.size(), s.objs.n, s.boxes.n, s.doors.n, s.rooms.n, unresolved, cover, placed_ok, placed, placed_near,
                s.tasks.n, s.lits.n, s.insts.n, robot_free, s.insts.n);

    // ---- 7. negative control: corrupted copies must be rejected
    struct Bad { const char* what; void (*mut)(std::vector<uint8_t>&); };
    Bad bad[] = {
        {"layout hash", [](std::vector<uint8_t>& b) { b[offsetof(RascFileHeader, layout_hash)] ^= 1; }},
        {"truncated", [](std::vector<uint8_t>& b) { b.resize(b.size() - 64); }},
        {"section size", [](std::vector<uint8_t>& b) { reinterpret_cast<RascFileHeader*>(b.data())->sections[RASC_SEC_OBJS].count += 1; }},
        {"magic", [](std::vector<uint8_t>& b) { b[0] ^= 0xff; }},
    };
    for (auto& x : bad) {
        Scene c;
        c.buf = s.buf;
        x.mut(c.buf);
        std::string e;
        CHECK(!load_buffer(c, &e), "corruption '%s' not rejected", x.what);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: rasc_test file.rasc...\n");
        return 2;
    }
    for (int i = 1; i < argc; i++) test_file(argv[i]);
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "all passed", g_fail);
    return g_fail ? 1 : 0;
}
