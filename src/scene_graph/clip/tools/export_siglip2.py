"""SigLIP 2 B/32-256 image tower -> static opset-13 ONNX with mask-aware MAP pooling (docs/clip_candidates.md 2.2, 3.5).

    inputs   images  N x 3 x S x S  float32, (rgb/255 - 0.5) / 0.5   (S = 256 by default)
             wpatch  N x g*g        float32, mask fraction per 32 px patch cell (g = S / 32), row-major
    output   emb     N x 768        float32, L2-normalised; mask enters as log(max(w, 0.01)) on the MAP-head attention logits
                                    (w = 1 everywhere == the stock image embedding)

Variants (efficiency study, docs/clip_candidates.md 8):
    --res R          input size (224 -> 49 tokens, 192 -> 36), position embeddings resampled (timm set_input_size)
    --layers L       keep only the first L of 12 transformer blocks (early exit; final norm + MAP head unchanged)
    --keep K@J       mask-guided token dropping: before block J keep the K tokens with the largest w (static K)
    --conv-patch     keep the 32x32 patch Conv (default: reshape + one GEMM, faster in TensorRT)
    --tome R@J       ToMe bipartite token merging, R tokens per block from block J on (proportional attention,
                     mask weight and size carried through, size enters the MAP head as log(size))

Python only for this offline export (clip_venv: torch CPU, open_clip 3.3, timm 1.0). Run:
    python export_siglip2.py --out models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask.onnx
then build the engine with build_engine.py (TensorRT Python from the TensorRT Python env).
"""
import argparse
import math
import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = ("ViT-B-32-SigLIP2-256", "webli")


def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """scaled_dot_product_attention does not export at opset 13: plain matmul/softmax instead."""
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    a = torch.matmul(q * s, k.transpose(-2, -1))
    if attn_mask is not None:
        a = a + attn_mask
    return torch.matmul(a.softmax(-1), v)


F.scaled_dot_product_attention = _sdpa


def load_model():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(MODEL[0], pretrained=MODEL[1])
    return m.eval()


def attn_fwd(at, x, bias=None):
    """timm Attention with an additive key bias (B,1,1,N) — used for ToMe proportional attention."""
    B, N, C = x.shape
    qkv = at.qkv(x).reshape(B, N, 3, at.num_heads, at.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    a = (q * at.scale) @ k.transpose(-2, -1)
    if bias is not None:
        a = a + bias
    x = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
    return at.proj(at.norm(x)), k.mean(1)   # keys averaged over heads = ToMe metric


def tome_merge(x, w, s, metric, r):
    """Bipartite soft matching (Bolya et al. 2023): merge r tokens of set A (even) into their best match in B (odd).
    x (B,N,C) features, w (B,N) mask fraction, s (B,N) token size. Size-weighted averages."""
    B, N, C = x.shape
    m = metric / metric.norm(dim=-1, keepdim=True)
    a, b = m[:, ::2], m[:, 1::2]
    sc = a @ b.transpose(-1, -2)
    node_max, node_idx = sc.max(-1)
    order = node_max.argsort(-1, descending=True)[..., None]
    src_idx, unm_idx = order[:, :r], order[:, r:]
    dst_idx = node_idx[..., None].gather(1, src_idx)

    nb = b.shape[1]
    onehot = (dst_idx == torch.arange(nb, device=x.device).view(1, 1, nb)).to(x.dtype)   # (B,r,nb): opset 13 has no scatter-add

    def merge(t, d):   # t (B,N,d): size-weighted sums
        ta, tb = t[:, ::2], t[:, 1::2]
        unm = ta.gather(1, unm_idx.expand(B, -1, d))
        src = ta.gather(1, src_idx.expand(B, -1, d))
        tb = tb + onehot.transpose(1, 2) @ src
        return torch.cat([unm, tb], 1)

    sw = s[..., None]
    xs = merge(x * sw, C)
    ws = merge((w * s)[..., None], 1)
    ss = merge(sw, 1)
    return xs / ss, (ws / ss)[..., 0], ss[..., 0]


class MaskEmbed(nn.Module):
    def __init__(self, m, layers=12, keep=None, tome=None, pe_gemm=True):
        super().__init__()
        self.tr, self.hd = m.visual.trunk, m.visual.head
        self.layers, self.keep, self.tome, self.pe_gemm = layers, keep, tome, pe_gemm

    def patches(self, images):
        """32 x 32 stride-32 patch conv as reshape + one GEMM (TensorRT picks a slow implicit-GEMM conv for this shape:
        ~13 % of the engine time; as a MatMul it is a plain tensor-core GEMM). Same numbers as the conv."""
        pe = self.tr.patch_embed
        if not self.pe_gemm:
            return pe(images)
        n, c, h, w = images.shape
        p = pe.proj.kernel_size[0]
        g = h // p
        x = images.reshape(n, c, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, c * p * p)
        wt = pe.proj.weight.reshape(pe.proj.weight.shape[0], -1)
        return x @ wt.t() + pe.proj.bias

    def forward(self, images, wpatch):
        tr = self.tr
        x = self.patches(images)
        x = tr._pos_embed(x)
        x = tr.norm_pre(x)
        w = wpatch
        s = torch.ones_like(w)
        for j, blk in enumerate(tr.blocks[: self.layers]):
            if self.keep is not None and j == self.keep[1]:
                K = self.keep[0]
                idx = w.topk(K, dim=1).indices                       # static K, mask-weighted tokens first
                x = x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
                w = w.gather(1, idx)
                s = s.gather(1, idx)
            if self.tome is not None and j >= self.tome[1]:
                y, metric = attn_fwd(blk.attn, blk.norm1(x), torch.log(s)[:, None, None, :])
                x = x + blk.ls1(y)
                x, w, s = tome_merge(x, w, s, metric, self.tome[0])
                x = x + blk.ls2(blk.mlp(blk.norm2(x)))
            else:
                x = blk(x)
        x = tr.norm(x)
        bias = (torch.log(w.clamp(min=1e-2)) + torch.log(s))[:, None, None, :]
        p = tr.attn_pool(x, attn_mask=bias)
        e = self.hd(tr.head(tr.fc_norm(p)))
        return e / e.norm(dim=-1, keepdim=True)


def build(res=256, layers=12, keep=None, tome=None, model=None, pe_gemm=True):
    m = model if model is not None else load_model()
    if res != 256:
        m.visual.trunk.set_input_size(img_size=(res, res))
    return MaskEmbed(m, layers, keep, tome, pe_gemm).eval()


def _pair(v):
    if not v:
        return None
    a, b = v.split("@")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--keep", default="", help="K@J: keep K tokens (largest mask weight) before block J")
    ap.add_argument("--tome", default="", help="R@J: merge R tokens per block from block J on")
    ap.add_argument("--conv-patch", action="store_true", help="keep the patch-embed Conv (default: reshape + GEMM)")
    ap.add_argument("--batch", type=int, default=8, help="dummy batch for tracing (dynamic batch axis)")
    a = ap.parse_args()
    mod = build(a.res, a.layers, _pair(a.keep), _pair(a.tome), pe_gemm=not a.conv_patch)
    g = a.res // 32
    x, w = torch.randn(a.batch, 3, a.res, a.res), torch.rand(a.batch, g * g)
    with torch.no_grad():
        # w = 1 must equal the stock embedding (only for the unmodified network)
        if a.res == 256 and a.layers == 12 and not a.keep and not a.tome:
            stock = F.normalize(load_model().visual(x), dim=-1)
            ones = mod(x, torch.ones(a.batch, g * g))
            print("w=1 vs stock cosine min %.6f" % F.cosine_similarity(ones, stock).min().item())
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.onnx.export(mod, (x, w), a.out, input_names=["images", "wpatch"], output_names=["emb"], opset_version=13,
                          dynamo=False, dynamic_axes={"images": {0: "n"}, "wpatch": {0: "n"}, "emb": {0: "n"}})
    print("exported", a.out, "res", a.res, "grid", g, "layers", a.layers, "keep", a.keep, "tome", a.tome)


if __name__ == "__main__":
    main()
