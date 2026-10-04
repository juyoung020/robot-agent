"""Object search evaluation (include/sgsearch.h through ctypes, libsgclip_c.so).

Ground truth: a realbag / detcmp run folder with objects_eval.csv (map object -> matched GT category) next to memory/.
Queries are what a user would say for each GT category present in the map (e.g. straight_chair -> "chair", fridge ->
"refrigerator"); relevant objects = map objects matched to those categories. Absent queries name household objects that
are NOT in the scene; any hit there is a false positive.

    ~/clip_venv/bin/python eval_objsearch.py RUN_DIR [--mem COPY_OF_MEMORY] [--lib ~/sgclip_search_build/libsgclip_c.so]
            [--sweep] [--json OUT]

Reports, for name-only search (step 2 disabled) and name + appearance re-search (default config):
    recall@1 / @5 over category queries (any relevant object in the top k), split into
        all categories / "mislabeled" categories (some relevant object's registered name does not name-match the query),
    recall of the mislabeled OBJECTS themselves (in top 5), absent-query false-positive rate (>= 1 hit; and hits the agent
    would not ask about), Korean query variants, latency per query (p50 / p95, text encoder included).
The memory folder gets cache/objsearch/ (vectors, names.json) — pass --mem with a scratch copy to keep the run folder clean.
"""
import argparse
import csv
import ctypes as C
import json
import os
import statistics
import time

# what a user would say -> GT categories it should find
CATEGORY_QUERIES = {
    "radio": ("radio", "라디오", ["radio"]),
    "chair": ("chair", "의자", ["straight_chair", "garden_chair"]),
    "sofa": ("sofa", "소파", ["sofa"]),
    "refrigerator": ("refrigerator", "냉장고", ["fridge"]),
    "picture": ("picture", "그림", ["picture"]),
    "coffee table": ("coffee table", "커피 테이블", ["coffee_table"]),
    "table": ("table", "탁자", ["coffee_table", "breakfast_table"]),
    "fireplace": ("fireplace", "벽난로", ["wood_fireplace"]),
    "dishwasher": ("dishwasher", "식기세척기", ["dishwasher"]),
    "microwave": ("microwave", "전자레인지", ["microwave"]),
    "lamp": ("lamp", "전등", ["room_light"]),
    "shelf": ("shelf", "선반", ["shelf"]),
    "television": ("television", "텔레비전", ["wall_mounted_tv"]),
    "oven": ("oven", "오븐", ["oven"]),
    "sink": ("sink", "싱크대", ["drop_in_sink"]),
    "cabinet": ("cabinet", "수납장", ["bottom_cabinet", "top_cabinet"]),
    "mirror": ("mirror", "거울", ["standing_mirror"]),
    "hall tree": ("hall tree", "옷걸이", ["hall_tree"]),
}
ABSENT = [("cup", "컵"), ("mug", "머그잔"), ("bottle", "병"), ("bowl", "그릇"), ("book", "책"), ("laptop", "노트북"), ("keyboard", "키보드"),
          ("cell phone", "휴대폰"), ("remote control", "리모컨"), ("toothbrush", "칫솔"), ("umbrella", "우산"), ("backpack", "배낭"),
          ("shoe", "신발"), ("teddy bear", "곰 인형"), ("scissors", "가위"), ("banana", "바나나"), ("apple", "사과"), ("guitar", "기타"),
          ("bicycle", "자전거"), ("vase", "꽃병"), ("bed", "침대"), ("toilet", "변기"), ("bathtub", "욕조"), ("washing machine", "세탁기"),
          ("printer", "프린터"), ("trash can", "쓰레기통"), ("pillow", "베개"), ("towel", "수건"), ("hat", "모자"), ("toaster", "토스터"),
          ("blender", "믹서기"), ("electric fan", "선풍기"), ("candle", "양초"), ("basket", "바구니"), ("cardboard box", "상자"),
          ("bucket", "양동이"), ("broom", "빗자루"), ("fire extinguisher", "소화기"), ("kettle", "주전자"), ("handbag", "핸드백")]
NODE_BASE = 5692549928996306944   # scene.json object node id = 'O' layer key + object id


class Cfg(C.Structure):
    _fields_ = [("mem_dir", C.c_char_p), ("cache_dir", C.c_char_p), ("labels", C.c_void_p), ("text", C.c_void_p), ("encoder", C.c_void_p),
                ("threads", C.c_int32), ("reg_lr", C.c_float), ("user_lr", C.c_float), ("look_lr", C.c_float), ("weak_name", C.c_float),
                ("name_min", C.c_float), ("app_min", C.c_float), ("app_rank", C.c_int32), ("exemplar_min", C.c_float), ("img_a", C.c_float),
                ("img_c0", C.c_float), ("img_min", C.c_float), ("attr_min", C.c_float)]


class TextCfg(C.Structure):
    _fields_ = [("engine", C.c_char_p), ("tokenizer", C.c_char_p), ("tok_emb", C.c_char_p), ("device", C.c_int32), ("max_batch", C.c_int32)]


class EncCfg(C.Structure):
    _fields_ = [("engine", C.c_char_p), ("device", C.c_int32), ("max_batch", C.c_int32), ("use_graph", C.c_int32), ("margin", C.c_float)]


class Search:
    def __init__(self, lib, labels_dir, encode=True):
        L = self.L = C.CDLL(os.path.expanduser(lib))
        for f in ("sgc_labels_open_ex", "sgc_text_create", "sgc_create", "sgs_open"):
            getattr(L, f).restype = C.c_void_p
        err = C.create_string_buffer(512)
        home = os.path.expanduser("~")
        self.lab = L.sgc_labels_open_ex(labels_dir.encode(), f"{home}/.cache/sgclip".encode(),
                                        f"{home}/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16".encode(), err, 512)
        assert self.lab, err.value
        tc = TextCfg()
        L.sgc_text_default_config(C.byref(tc), None)
        self.text = L.sgc_text_create(C.byref(tc), err, 512)
        assert self.text, err.value
        self.enc = None
        if encode:
            ec = EncCfg()
            L.sgc_default_config(C.byref(ec))
            ec.engine = f"{home}/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan".encode()
            ec.margin = 0.0
            self.enc = L.sgc_create(C.byref(ec), err, 512)
            assert self.enc, err.value
        self.idx = None
        self.buf = C.create_string_buffer(1 << 22)

    def open(self, mem, **kw):
        if self.idx:
            self.L.sgs_close(C.c_void_p(self.idx))
        c = Cfg()
        self.L.sgs_default_config(C.byref(c))
        for k, v in kw.items():
            setattr(c, k, v)
        c.mem_dir, c.labels, c.text, c.encoder = mem.encode(), self.lab, self.text, self.enc
        err = C.create_string_buffer(512)
        self.idx = self.L.sgs_open(C.byref(c), err, 512)
        assert self.idx, err.value
        return self

    def stats(self):
        self.L.sgs_stats_json(C.c_void_p(self.idx), self.buf, len(self.buf))
        return json.loads(self.buf.value)

    def search(self, q, k=5, force=0):
        n = self.L.sgs_search_json(C.c_void_p(self.idx), q.encode(), k, force, self.buf, len(self.buf))
        assert n >= 0, n
        return json.loads(self.buf.value)


def load_gt(run, mem, gt_objects=None, pad=0.05):
    """object id -> set of GT categories: the evaluator's one-to-one match (objects_eval.csv gt_category) plus every GT object
    whose box (padded) contains the map object's centre — duplicates / fragments of the same thing (e.g. a second radio node)
    that the one-to-one match leaves unmatched. Map -> world: 2-D rigid fit on objects_eval.csv (identity for gt-pose runs)."""
    import numpy as np
    boxes = []
    if gt_objects is None:
        import glob
        c = glob.glob(os.path.join(run, "..", "streams", "*", "gt_objects.json")) + glob.glob(os.path.join(run, "..", "..", "streams", "*", "gt_objects.json"))
        gt_objects = c[0] if c else None
    if gt_objects:
        for g in json.load(open(gt_objects)):
            boxes.append((g["category"], [x - pad for x in g["lo"]], [x + pad for x in g["hi"]]))
    gt, A, B = {}, [], []
    for r in csv.DictReader(open(os.path.join(run, "objects_eval.csv"))):
        if r["gt_category"]:
            gt[int(r["node_id"]) - NODE_BASE] = {r["gt_category"]}
        A.append([float(r["map_x"]), float(r["map_y"])])
        B.append([float(r["world_x"]), float(r["world_y"])])
    A, B = np.array(A), np.array(B)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    t = cb - R @ ca
    for o in json.load(open(os.path.join(mem, "view.json")))["objects"]:
        xy = R @ np.array(o["pos"][:2]) + t
        p = [xy[0], xy[1], o["pos"][2]]
        for c, lo, hi in boxes:
            if all(lo[k] <= p[k] <= hi[k] for k in range(3)):
                gt.setdefault(o["id"], set()).add(c)
    return gt


def evaluate(S, mem, gt, cfg, korean=False):
    S.open(mem, **cfg)
    rows, lat = [], []
    for key, (en, ko, cats) in CATEGORY_QUERIES.items():
        rel = {i for i, c in gt.items() if c & set(cats)}
        if not rel:
            continue
        q = ko if korean else en
        t = time.perf_counter()
        r = S.search(q, 5)
        lat.append((time.perf_counter() - t) * 1e6)
        ids = [h["id"] for h in r["hits"]]
        rows.append({"q": q, "rel": sorted(rel), "ids": ids, "mis": False, "types": [h["match_type"] for h in r["hits"]], "r1": bool(ids[:1] and ids[0] in rel),
                     "r5": any(i in rel for i in ids[:5]), "step2": r["step2"]})
    absent = []
    for en, ko in ABSENT:
        q = ko if korean else en
        t = time.perf_counter()
        r = S.search(q, 5)
        lat.append((time.perf_counter() - t) * 1e6)
        absent.append({"q": q, "hits": [(h["id"], h["name"], h["registered"], h["match_type"], h["match"], ",".join(sorted(gt.get(h["id"], ())))) for h in r["hits"]]})
    return rows, absent, lat


def name_only(cfg):
    c = dict(cfg)
    c.update(app_min=2.0, img_min=2.0)   # step 2 can run but never admits a candidate
    return c


def summarize(tag, rows, absent, lat, misl):
    """misl: query -> relevant objects the NAME step cannot find (registered name does not match); mis_r* = one of THOSE in top k"""
    n = len(rows)
    m = [r for r in rows if misl.get(r["q"])]
    m1 = [bool(r["ids"][:1] and r["ids"][0] in misl[r["q"]]) for r in m]
    m5 = [any(i in misl[r["q"]] for i in r["ids"][:5]) for r in m]
    fp = sum(1 for a in absent if a["hits"])
    # the agent tool acts without asking only when the top hit is a name match with match >= 0.5 (search_objects advice())
    fp_app = sum(1 for a in absent if a["hits"] and a["hits"][0][3] == "name" and a["hits"][0][4] >= 0.5)
    s = {"tag": tag, "queries": n, "r1": sum(r["r1"] for r in rows) / n, "r5": sum(r["r5"] for r in rows) / n,
         "misnamed_queries": len(m), "mis_r1": (sum(m1) / len(m)) if m else None,
         "mis_r5": (sum(m5) / len(m)) if m else None, "absent": len(absent), "absent_fp": fp / len(absent),
         "absent_fp_name": fp_app / len(absent), "us_p50": statistics.median(lat), "us_p95": sorted(lat)[int(0.95 * len(lat)) - 1]}
    print(f"{tag:34s} R@1 {s['r1']:.2f}  R@5 {s['r5']:.2f} | misnamed({len(m)}) R@1 {s['mis_r1'] or 0:.2f} R@5 {s['mis_r5'] or 0:.2f} | "
          f"absent FP {s['absent_fp']:.2f} (no-ask {s['absent_fp_name']:.2f}) | {s['us_p50']:.0f}/{s['us_p95']:.0f} us")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mem")
    ap.add_argument("--lib", default="~/sgclip_search_build/libsgclip_c.so")
    ap.add_argument("--labels", default=os.path.expanduser("~/embed_work/labels/objects-v1"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    mem = a.mem or os.path.join(a.run, "memory")
    gt = load_gt(a.run, mem)
    S = Search(a.lib, a.labels)
    S.open(mem)
    print("index:", json.dumps(S.stats()))
    # misnamed objects per query: relevant objects that name-only search (all name hits, k = 0) does not return
    misl, misl_ko = {}, {}
    for korean, dst in ((False, misl), (True, misl_ko)):
        S.open(mem, **name_only({}))
        for key, (en, ko, cats) in CATEGORY_QUERIES.items():
            rel = {i for i, c in gt.items() if c & set(cats)}
            q = ko if korean else en
            got = {h["id"] for h in S.search(q, 0)["hits"]}
            if rel - got:
                dst[q] = rel - got
    out = {"gt_objects": len(gt), "misnamed": {k: sorted(v) for k, v in misl.items()}}
    res = []
    for korean in (False, True):
        ms = misl_ko if korean else misl
        lang = "ko" if korean else "en"
        r, ab, lat = evaluate(S, mem, gt, name_only({}), korean)
        res.append(summarize(f"name only [{lang}]", r, ab, lat, ms))
        r, ab, lat = evaluate(S, mem, gt, {}, korean)
        res.append(summarize(f"name + appearance [{lang}]", r, ab, lat, ms))
        if a.verbose:
            for x in r:
                print("   ", x["q"], "R@1" if x["r1"] else "   ", "R@5" if x["r5"] else "   ", x["ids"], x["types"], "rel", x["rel"][:6])
            for x in ab:
                if x["hits"]:
                    print("    FP", x["q"], x["hits"])
    if a.sweep:
        for app_min in (0.02, 0.05, 0.1, 0.15, 0.25):
            for app_rank in (1, 2, 3):
                r, ab, lat = evaluate(S, mem, gt, {"app_min": app_min, "app_rank": app_rank})
                res.append(summarize(f"sweep app_min {app_min} rank<={app_rank}", r, ab, lat, misl))
    out["results"] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
