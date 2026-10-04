// RASC v1 loader + CPU reference reader (header only, C++17). Layout: rasc_format.h (generated
// by b1kconv — never edit). The device side (E2) uploads the same bytes; these helpers are the
// CPU reference for anything a kernel reads from the file.
#pragma once
#include "rasc_format.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rasc {

template <class T>
struct View {
    const T* p = nullptr;
    size_t n = 0;
    const T& operator[](size_t i) const { return p[i]; }
    const T* begin() const { return p; }
    const T* end() const { return p + n; }
    size_t size() const { return n; }
};

struct Scene {
    std::vector<uint8_t> buf;  // whole file (8-byte aligned storage)
    const RascFileHeader* h = nullptr;

    // records
    View<RascCatRec> cats;
    View<RascRoomRec> rooms;
    View<RascObjRec> objs;
    View<RascBoxRec> boxes;
    View<RascDoorRec> doors;
    View<float> joints;
    View<uint8_t> room_grid;
    View<uint8_t> trav[4];  // TRAV, TRAV_NO_OBJ, TRAV_NO_DOOR, TRAV_OPEN_DOOR
    View<RascTaskRec> tasks;
    View<RascTaskObjRec> task_objs;
    View<RascLitRec> lits;
    View<RascVarRec> vars;
    View<uint16_t> cands;
    View<uint16_t> removed;
    View<RascInstRec> insts;
    View<RascPoseRec> poses;
    View<uint32_t> in_rooms;
    View<RascLimitsRec> limits;  // [0] outer (inclusion), [1] inner (flag bits)
    View<RascPickRec> picks;
    View<RascPlaceRec> places;
    View<RascPairRec> pairs;
    View<RascPnpRange> pnp_ranges;  // one per instance + last = scene level
    View<int16_t> floor_z;          // [grid_h][grid_w] floor top (mm), INT16_MIN = none
    View<char> strings;

    const char* str(uint32_t off) const { return off < strings.n ? strings.p + off : ""; }
    std::string name() const { const char* n = (const char*)h->scene_name; return std::string(n, strnlen(n, sizeof h->scene_name)); }

    // ---- grid (CPU reference) ----
    // world xy -> cell; false if outside the grid
    bool cell(float x, float y, int* r, int* c) const {
        float fc = std::floor((x - h->origin[0]) / h->cell), fr = std::floor((y - h->origin[1]) / h->cell);
        if (fc < 0 || fr < 0 || fc >= (float)h->grid_w || fr >= (float)h->grid_h) return false;
        *r = (int)fr;
        *c = (int)fc;
        return true;
    }
    // RoomRec index at a world point, -1 = none / outside
    int room_at(float x, float y) const {
        int r, c;
        if (!cell(x, y, &r, &c)) return -1;
        return (int)room_grid[(size_t)r * h->grid_w + c] - 1;
    }
    // layer 0..3 (RASC_SEC_TRAV + layer); outside = blocked
    bool free_cell(int layer, int r, int c) const {
        if (r < 0 || c < 0 || r >= (int)h->grid_h || c >= (int)h->grid_w) return false;
        size_t i = (size_t)r * h->grid_w + c;
        return (trav[layer][i >> 3] >> (i & 7)) & 1;
    }
    bool free_at(int layer, float x, float y) const {
        int r, c;
        return cell(x, y, &r, &c) && free_cell(layer, r, c);
    }
    // point inside a static box (yaw-only OBB)
    static bool in_box(const RascBoxRec& b, float x, float y, float z) {
        float dx = x - b.center[0], dy = y - b.center[1], cs = std::cos(b.yaw), sn = std::sin(b.yaw);
        float lx = cs * dx + sn * dy, ly = -sn * dx + cs * dy;
        return std::fabs(lx) <= b.half[0] && std::fabs(ly) <= b.half[1] && z >= b.zmin && z <= b.zmax;
    }

    // ---- tasks ----
    View<RascTaskObjRec> objs_of(const RascTaskRec& t) const { return {task_objs.p + t.obj_off, t.n_obj}; }
    View<RascLitRec> init_of(const RascTaskRec& t) const { return {lits.p + t.init_off, t.n_init}; }
    View<RascLitRec> goal_of(const RascTaskRec& t) const { return {lits.p + t.goal_off, t.n_goal}; }
    View<RascVarRec> vars_of(const RascTaskRec& t) const { return {vars.p + t.var_off, t.n_var}; }
    View<RascInstRec> insts_of(const RascTaskRec& t) const { return {insts.p + t.inst_off, t.n_inst}; }
    View<RascPoseRec> poses_of(const RascInstRec& i) const { return {poses.p + i.pose_off, i.n_pose}; }
    // candidates (task-local object indices) of a literal argument
    View<uint16_t> arg_cands(const RascTaskRec& t, uint16_t a, uint16_t* tmp) const {
        if (a & RASC_ARG_VAR) {
            const RascVarRec& v = vars[t.var_off + (a & ~RASC_ARG_VAR)];
            return {cands.p + t.cand_off + v.cand_off, v.cand_n};
        }
        *tmp = a;
        return {tmp, 1};
    }
    std::string lit_str(const RascTaskRec& t, const RascLitRec& l) const {
        std::string s = std::string("(") + rasc_pred_names[l.pred];
        for (int k = 0; k < l.nargs; k++) {
            uint16_t a = l.arg[k];
            s += ' ';
            if (a & RASC_ARG_ROOM) s += "room:" + std::to_string(a & ~RASC_ARG_ROOM);
            else if (a & RASC_ARG_VAR) s += "?v" + std::to_string(a & ~RASC_ARG_VAR);
            else s += str(task_objs[t.obj_off + a].inst);
        }
        s += ')';
        return l.neg ? "(not " + s + ")" : s;
    }
};

template <class T>
static bool sec_view(const Scene& s, int id, View<T>* v, size_t elem, std::string* err) {
    const RascSec& e = s.h->sections[id];
    size_t want = elem ? (size_t)e.count * elem : (e.elem == 0 ? ((size_t)e.count + 7) / 8 : (size_t)e.count);
    if (e.bytes && (e.offset % 64 || e.offset + e.bytes > s.buf.size())) {
        *err = std::string("section ") + rasc_sec_names[id] + " out of range";
        return false;
    }
    if (elem && e.elem != elem) {
        *err = std::string("section ") + rasc_sec_names[id] + " element size " + std::to_string(e.elem) + " != " + std::to_string(elem);
        return false;
    }
    if (e.bytes != want) {
        *err = std::string("section ") + rasc_sec_names[id] + " size mismatch";
        return false;
    }
    v->p = reinterpret_cast<const T*>(s.buf.data() + e.offset);
    v->n = elem ? e.count : (size_t)e.bytes / sizeof(T);
    return true;
}

// Validate + map a buffer that already holds a whole file.
inline bool load_buffer(Scene& s, std::string* err) {
    if (s.buf.size() < sizeof(RascFileHeader)) { *err = "file shorter than header"; return false; }
    s.h = reinterpret_cast<const RascFileHeader*>(s.buf.data());
    const RascFileHeader& h = *s.h;
    if (h.magic != RASC_MAGIC) { *err = "bad magic"; return false; }
    if (h.version != RASC_VERSION) { *err = "version " + std::to_string(h.version); return false; }
    if (h.header_bytes != sizeof(RascFileHeader)) { *err = "header size mismatch"; return false; }
    if (h.layout_hash != RASC_LAYOUT_HASH) { *err = "layout hash mismatch (file written by a different b1kconv layout)"; return false; }
    bool ok = sec_view(s, RASC_SEC_STRINGS, &s.strings, 1, err) && sec_view(s, RASC_SEC_CATS, &s.cats, sizeof(RascCatRec), err) &&
              sec_view(s, RASC_SEC_ROOMS, &s.rooms, sizeof(RascRoomRec), err) && sec_view(s, RASC_SEC_OBJS, &s.objs, sizeof(RascObjRec), err) &&
              sec_view(s, RASC_SEC_BOXES, &s.boxes, sizeof(RascBoxRec), err) && sec_view(s, RASC_SEC_DOORS, &s.doors, sizeof(RascDoorRec), err) &&
              sec_view(s, RASC_SEC_JOINTS, &s.joints, 4, err) && sec_view(s, RASC_SEC_ROOM_GRID, &s.room_grid, 1, err) &&
              sec_view(s, RASC_SEC_TRAV, &s.trav[0], 0, err) && sec_view(s, RASC_SEC_TRAV_NO_OBJ, &s.trav[1], 0, err) &&
              sec_view(s, RASC_SEC_TRAV_NO_DOOR, &s.trav[2], 0, err) && sec_view(s, RASC_SEC_TRAV_OPEN_DOOR, &s.trav[3], 0, err) &&
              sec_view(s, RASC_SEC_TASKS, &s.tasks, sizeof(RascTaskRec), err) &&
              sec_view(s, RASC_SEC_TASK_OBJS, &s.task_objs, sizeof(RascTaskObjRec), err) &&
              sec_view(s, RASC_SEC_LITS, &s.lits, sizeof(RascLitRec), err) && sec_view(s, RASC_SEC_VARS, &s.vars, sizeof(RascVarRec), err) &&
              sec_view(s, RASC_SEC_CANDS, &s.cands, 2, err) && sec_view(s, RASC_SEC_REMOVED, &s.removed, 2, err) &&
              sec_view(s, RASC_SEC_INSTS, &s.insts, sizeof(RascInstRec), err) && sec_view(s, RASC_SEC_POSES, &s.poses, sizeof(RascPoseRec), err) &&
              sec_view(s, RASC_SEC_IN_ROOMS, &s.in_rooms, 4, err) && sec_view(s, RASC_SEC_LIMITS, &s.limits, sizeof(RascLimitsRec), err) &&
              sec_view(s, RASC_SEC_PICKS, &s.picks, sizeof(RascPickRec), err) && sec_view(s, RASC_SEC_PLACES, &s.places, sizeof(RascPlaceRec), err) &&
              sec_view(s, RASC_SEC_PAIRS, &s.pairs, sizeof(RascPairRec), err) && sec_view(s, RASC_SEC_PNP_RANGES, &s.pnp_ranges, sizeof(RascPnpRange), err) &&
              sec_view(s, RASC_SEC_FLOOR_Z, &s.floor_z, 2, err);
    if (ok && (s.limits.n != 2 || s.pnp_ranges.n != s.insts.n + 1 || s.floor_z.n != (size_t)h.grid_w * h.grid_h)) { *err = "pick-and-place tables malformed"; return false; }
    if (!ok) return false;
    if (s.room_grid.n != (size_t)h.grid_w * h.grid_h) { *err = "room grid size"; return false; }
    if (s.strings.n == 0 || s.strings.p[s.strings.n - 1] != 0) { *err = "string pool not NUL terminated"; return false; }
    return true;
}

inline bool load(Scene& s, const char* path, std::string* err) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { *err = std::string(path) + ": cannot open"; return false; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    s.buf.resize(n > 0 ? (size_t)n : 0);
    size_t got = n > 0 ? std::fread(s.buf.data(), 1, (size_t)n, f) : 0;
    std::fclose(f);
    if ((long)got != n) { *err = std::string(path) + ": short read"; return false; }
    return load_buffer(s, err);
}

inline uint64_t fnv64(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

}  // namespace rasc
