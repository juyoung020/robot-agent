"""Text side for object search (runs off-board / agent side): SigLIP 2 B/32-256 text tower -> 768-d L2 vector in the
same space as the robot's object embeddings (memory/objects/O<id>_emb.f16).

    # one shot: top-k objects of a memory folder for each query (JSON on stdout)
    python text_query.py --mem MEMORY_DIR "radio" "라디오" "흰 의자"
    # encode only: raw 768 x FP32 little-endian per query to OUT
    python text_query.py --encode OUT.f32 "white chair"
    # server: model stays loaded. GET /encode?q=...  -> {"q", "vec": [768]}
    #                             GET /search?mem=DIR&q=...&k=5 -> {"q", "hits": [{"id", "score", "name", "name_ko"}]}
    python text_query.py --serve 8091

Query text is used as is (SigLIP 2's Gemma tokenizer is multilingual: Korean works without translation); English single
nouns also get the label-table prompt ("a photo of a {}."), like the label table. Names come from memory/cache/names.json.
The on-device Korean student (training/embed) will replace this with the same interface: text -> 768-d vector.
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TIMM_FUSED_ATTN", "0")
import numpy as np
import torch

_M = None


def model():
    global _M
    if _M is None:
        import open_clip
        torch.set_num_threads(min(16, os.cpu_count() or 4))
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-SigLIP2-256", pretrained="webli")
        _M = (m.eval(), open_clip.get_tokenizer("ViT-B-32-SigLIP2-256"))
    return _M


def encode(texts):
    m, tok = model()
    t = [("a photo of a %s." % q) if re.fullmatch(r"[A-Za-z][A-Za-z \-]*", q) and len(q.split()) <= 3 else q for q in texts]
    with torch.no_grad():
        e = m.encode_text(tok(t)).float().numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def load_mem(mem):
    ids, E = [], []
    for p in sorted(glob.glob(os.path.join(mem, "objects", "O*_emb.f16"))):
        v = np.fromfile(p, np.float16).astype(np.float32)
        if v.size != 768:
            continue
        ids.append(int(os.path.basename(p)[1:].split("_")[0]))
        E.append(v / np.linalg.norm(v))
    names = {}
    try:
        names = json.load(open(os.path.join(mem, "cache", "names.json")))["objects"]
    except Exception:
        pass
    live = None
    try:   # only objects present in the latest view.json (scenemap "structural" = big / fixed furniture — keep those)
        vj = json.load(open(os.path.join(mem, "view.json")))
        live = {o["id"] for o in vj.get("objects", [])}
    except Exception:
        pass
    keep = [i for i, x in enumerate(ids) if live is None or x in live]
    return [ids[i] for i in keep], (np.stack([E[i] for i in keep]) if keep else np.zeros((0, 768), np.float32)), names


def search(mem, queries, k=5):
    ids, E, names = load_mem(mem)
    Q = encode(queries)
    out = []
    for q, v in zip(queries, Q):
        s = E @ v if len(ids) else np.zeros(0)
        o = np.argsort(-s)[:k]
        out.append({"q": q, "hits": [{"id": ids[i], "score": round(float(s[i]), 4), "name": names.get(f"O{ids[i]}", {}).get("level", ""),
                                      "name_ko": names.get(f"O{ids[i]}", {}).get("level_ko", "")} for i in o]})
    return out


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        a = urllib.parse.parse_qs(u.query)
        q = a.get("q", [""])[0]
        try:
            if u.path == "/encode":
                r = {"q": q, "vec": [round(float(x), 6) for x in encode([q])[0]]}
            elif u.path == "/search":
                r = search(a["mem"][0], [q], int(a.get("k", ["5"])[0]))[0]
            else:
                self.send_error(404)
                return
            b = json.dumps(r, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--mem")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encode")
    ap.add_argument("--serve", type=int)
    a = ap.parse_args()
    if a.serve:
        model()
        print(f"text_query serving on 127.0.0.1:{a.serve}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", a.serve), H).serve_forever()
    elif a.encode:
        encode(a.queries).astype("<f4").tofile(a.encode)
    elif a.mem:
        print(json.dumps(search(a.mem, a.queries, a.k), ensure_ascii=False, indent=1))
    else:
        sys.exit("give --mem, --encode or --serve")


if __name__ == "__main__":
    main()
