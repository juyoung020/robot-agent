"""SigLIP 2 B/32-256 TEXT tower -> ONNX + tokenizer / token-embedding files for the C++ text encoder (src/textenc.cpp).

Free-text object search (src/objindex.cpp, include/sgsearch.h) needs query vectors at runtime in the same 768-d space as
the label table and the object image embeddings. This is the text twin of export_siglip2.py.

Split: the 256k x 768 token-embedding table (197 M of the tower's ~282 M parameters) stays OUT of the engine. The C++ side
gathers rows from an mmap'd FP16 file on the CPU (a query touches <= 64 rows), so the GPU engine is only the 12-layer
transformer (~85 M parameters, ~170 MB FP16) — this matters on the Jetson Nano (4 GB shared).

    outputs (all in --out, default models/ovdet/x86_sm120/siglip2_b32)
      siglip2_b32_text.onnx     input  tok_emb  N x 64 x 768 float32  (token embedding rows, pad id 0 included)
                                output emb      N x 768       float32  L2-normalised (positional emb, 12 blocks, ln_final,
                                                                        'last' pool = position 63, projection with bias)
      siglip2_b32_tokemb.f16    256000 x 768 FP16 little-endian (row = token id)
      siglip2_b32_tok.bin       Gemma BPE tokenizer for src/textenc.cpp (format below)
      text_parity.bin           C++ test reference: queries, token ids (HF), PyTorch FP32 embeddings

    tokenizer file 'SGT1' (little-endian):
      int32 magic 0x31544753, int32 n_vocab, int32 n_merges, int32 context (64), int32 eos (1), int32 pad (0), int32 unk (3)
      n_vocab x (uint16 len, bytes)            token strings (UTF-8, '▁' for space, byte-fallback tokens '<0xAB>')
      n_merges x (int32 left, int32 right, int32 merged)   in rank order (rank = index)
    text_parity.bin 'TPR1': int32 magic, int32 n | per query: uint16 len, utf8 | int32 ids[64] | float32 emb[768]

The tokenizer is open_clip's HFTokenizer(timm/ViT-B-32-SigLIP2-256, clean="canonicalize"): text -> '_' to space ->
ASCII punctuation removed -> lower case -> whitespace collapsed -> ' ' to '▁' -> one BPE word (merges by rank, byte
fallback) -> + <eos> -> truncate to 64 (eos kept last) -> pad 0. ftfy / html unescape are not reproduced (queries are plain).

    python export_siglip2_text.py
    python build_engine.py models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_text.onnx \
        models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_text_fp16.plan --pin norm
"""
import argparse
import json
import os
import struct

os.environ.setdefault("TIMM_FUSED_ATTN", "0")
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
RA_ROOT = os.environ.get("RA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
RA_MODELS = os.environ.get("OVDET_MODELS", os.path.join(RA_ROOT, "models/ovdet"))
RA_EMBED = os.environ.get("RA_EMBED_WORK", os.path.join(RA_ROOT, "training/data/embed"))
RA_BENCH = os.path.join(RA_ROOT, "data/clip_bench")
RA_BUILD = os.environ.get("RA_BUILD", os.path.join(RA_ROOT, "build"))


def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """scaled_dot_product_attention does not export at opset 13: plain matmul/softmax (as export_siglip2.py)."""
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    a = torch.matmul(q * s, k.transpose(-2, -1))
    if attn_mask is not None:
        a = a + attn_mask
    return torch.matmul(a.softmax(-1), v)


F.scaled_dot_product_attention = _sdpa
torch.backends.mha.set_fastpath_enabled(False)   # nn.MultiheadAttention fast path (_native_multi_head_attention) has no ONNX op

MODEL = ("ViT-B-32-SigLIP2-256", "webli")

# parity / sanity queries: English label prompts, free text, Korean, mixed, punctuation, long
QUERIES = [
    "a photo of a radio.", "radio", "fire extinguisher", "a photo of a fire extinguisher.", "라디오", "소화기",
    "흰 의자", "빨간 컵", "the red mug on the kitchen table", "Fire Extinguisher!", "TV remote_control", "냉장고 옆 쓰레기통",
    "a wooden chair", "a metal object", "a blue object", "a photo of a coffee mug.", "책상 위에 있는 노트북",
    "an object made of glass", "small", "ラジオ", "收音机", "café au lait", "a photo of a hall tree.", "",
    " ".join(["very long query"] * 30),
]


class TextTower(nn.Module):
    """open_clip TextTransformer.forward without the embedding gather (input = gathered rows)."""

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, tok_emb):
        t = self.t
        x = tok_emb + t.positional_embedding[: tok_emb.shape[1]]
        x = t.transformer(x, attn_mask=None)
        x = t.ln_final(x)
        x = x[:, -1]                       # pool_type 'last' (no pad mask in SigLIP 2: always position 63)
        x = t.text_projection(x)
        return x / x.norm(dim=-1, keepdim=True)


def write_tokenizer(tok, path):
    j = json.loads(tok.backend_tokenizer.to_str())
    m = j["model"]
    assert m["type"] == "BPE" and m["byte_fallback"], m["type"]
    vocab = m["vocab"]
    n = max(vocab.values()) + 1
    inv = [""] * n
    for s, i in vocab.items():
        inv[i] = s
    merges = []
    for mg in m["merges"]:
        a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
        if a in vocab and b in vocab and (a + b) in vocab:
            merges.append((vocab[a], vocab[b], vocab[a + b]))
    with open(path, "wb") as f:
        f.write(struct.pack("<7i", 0x31544753, n, len(merges), 64, 1, 0, vocab.get("<unk>", 3)))
        for s in inv:
            b = s.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        f.write(np.asarray(merges, np.int32).tobytes())
    return n, len(merges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RA_MODELS, "x86_sm120/siglip2_b32"))
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(*MODEL)
    m = m.eval()
    tok = open_clip.get_tokenizer(MODEL[0])
    os.makedirs(a.out, exist_ok=True)

    nv, nm = write_tokenizer(tok.tokenizer, f"{a.out}/siglip2_b32_tok.bin")
    W = m.text.token_embedding.weight.detach().float().numpy()
    W.astype("<f2").tofile(f"{a.out}/siglip2_b32_tokemb.f16")
    print(f"tokenizer: vocab {nv}, merges {nm}; token embedding {W.shape} FP16 {W.nbytes // 2 >> 20} MB")

    mod = TextTower(m.text).eval()
    with torch.no_grad():
        ids = tok(QUERIES)
        ref = torch.nn.functional.normalize(m.encode_text(ids), dim=-1)
        x = m.text.token_embedding(ids)
        mine = mod(x)
        print("split tower vs encode_text cosine min %.7f" % torch.nn.functional.cosine_similarity(mine, ref).min().item())
        torch.onnx.export(mod, (x[: a.batch],), f"{a.out}/siglip2_b32_text.onnx", input_names=["tok_emb"], output_names=["emb"],
                          opset_version=13, dynamo=False, dynamic_axes={"tok_emb": {0: "n"}, "emb": {0: "n"}})
    with open(f"{a.out}/text_parity.bin", "wb") as f:
        f.write(struct.pack("<ii", 0x31525054, len(QUERIES)))
        for q, i, e in zip(QUERIES, ids.numpy().astype(np.int32), ref.numpy().astype(np.float32)):
            b = q.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
            f.write(i.tobytes())
            f.write(e.tobytes())
    print("wrote", f"{a.out}/siglip2_b32_text.onnx", "and text_parity.bin", len(QUERIES), "queries")


if __name__ == "__main__":
    main()
