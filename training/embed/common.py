"""Shared paths, model wrappers and crop views for the embedding-head training (training/embed).

Views follow the CLIP study (box / stretch / masked) so our numbers line up with it.
Everything heavy (data, embeddings, weights) lives under $RA_EMBED_WORK (default training/data/embed, gitignored).
Paths follow config/paths.env: RA_EMBED_WORK, OVDET_MODELS, RA_BUILD.
"""
import os
os.environ.setdefault("TIMM_FUSED_ATTN", "0")
import numpy as np, torch, torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
WORK = os.environ.get('RA_EMBED_WORK') or os.environ.get('EMBED_WORK') or f'{REPO}/training/data/embed'
OVDET = os.environ.get('OVDET_MODELS') or f'{REPO}/models/ovdet'
BUILD = os.environ.get('RA_BUILD') or f'{REPO}/build'
BENCH = os.environ.get('CLIP_BENCH') or f'{WORK}/clip_bench'   # the CLIP study's eval set (read-only; archived — set CLIP_BENCH to use)
VIEWS = ('box', 'stretch', 'masked')

# key: (open_clip name, pretrained tag, license)
MODELS = {
    'b32_openai': ('ViT-B-32-quickgelu', 'openai', 'MIT'),
    'mc2_s0': ('MobileCLIP2-S0', 'dfndr2b', 'Apple ML research (research only)'),
    'siglip2_so400m': ('ViT-SO400M-16-SigLIP2-384', 'webli', 'Apache-2.0'),
    'pe_l14': ('PE-Core-L-14-336', 'meta', 'Apache-2.0'),
    'eva02_l14': ('EVA02-L-14-336', 'merged2b_s6b_b61k', 'MIT'),
    'siglip2_b16': ('ViT-B-16-SigLIP2', 'webli', 'Apache-2.0'),
    'siglip2_b32': ('ViT-B-32-SigLIP2-256', 'webli', 'Apache-2.0'),
}
POOL = {'siglip2_b32', 'siglip2_b16'}      # models with a mask-pooled output (clip_candidates.md 3.5: emb_mask)
TPL = ['a photo of a {}.', 'a photo of the {}.', 'a cropped photo of a {}.', 'a {} in a room.']


def l2(x, axis=-1):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


class Enc:
    """open_clip model: image (N,3,S,S in [0,1]) -> embedding, text list -> embedding."""

    def __init__(self, key, device='cuda', half=True, visual_only=False):
        import open_clip
        name, pt, _ = MODELS[key]
        m, _, pre = open_clip.create_model_and_transforms(name, pretrained=pt)
        if key.startswith('mc'):
            from timm.utils import reparameterize_model
            m = reparameterize_model(m)
        self.key, self.dev = key, device
        self.dtype = torch.float16 if (half and device == 'cuda') else torch.float32
        if visual_only:                       # image tower on the GPU, the rest stays on the CPU (shared GPU)
            import types                      # drop the text tower: nothing big left in RAM to copy into workers
            v = m.visual.eval().to(device, self.dtype); del m
            self.m = types.SimpleNamespace(visual=v, encode_image=v); m = self.m
        else:
            self.m = m.eval().to(device, self.dtype)
        self.tok = open_clip.get_tokenizer(name)
        t = [x for x in pre.transforms if type(x).__name__ == 'Normalize']
        self.mean = np.array(t[0].mean if t else (0.5,) * 3, np.float32)
        self.std = np.array(t[0].std if t else (0.5,) * 3, np.float32)
        sz = getattr(m.visual, 'image_size', 224)
        self.size = sz[0] if isinstance(sz, (tuple, list)) else int(sz)
        self.dim = None

    @torch.no_grad()
    def image(self, x):
        """x: float tensor (N,3,S,S) in [0,1] at self.size."""
        mu = torch.tensor(self.mean, device=self.dev)[:, None, None]; sd = torch.tensor(self.std, device=self.dev)[:, None, None]
        x = ((x.to(self.dev) - mu) / sd).to(self.dtype)
        return self.m.encode_image(x).float()

    @torch.no_grad()
    def image_pool(self, x, w):
        """SigLIP MAP head with the mask as additive log-weight on the attention logits (= clip_bench patchpool).
        x (N,3,S,S) box view in [0,1], w (N,S,S) mask in the same frame."""
        mu = torch.tensor(self.mean, device=self.dev)[:, None, None]; sd = torch.tensor(self.std, device=self.dev)[:, None, None]
        x = ((x.to(self.dev) - mu) / sd).to(self.dtype)
        tr = self.m.visual.trunk
        h = tr.forward_features(x)
        npre = getattr(tr, 'num_prefix_tokens', 0); g = int(round((h.shape[1] - npre) ** 0.5))
        ww = F.adaptive_avg_pool2d(w.to(self.dev).float()[:, None], g).flatten(1)
        if npre: ww = torch.cat([torch.ones(ww.shape[0], npre, device=self.dev), ww], 1)
        bias = torch.log(ww.clamp(min=1e-2))[:, None, None, :].to(self.dtype)
        p = tr.attn_pool(h, attn_mask=bias)
        p = tr.head(tr.head_drop(tr.fc_norm(p)))
        return self.m.visual.head(p).float()

    @torch.no_grad()
    def text(self, texts, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            out.append(self.m.encode_text(self.tok(texts[i:i + bs]).to(self.dev)).float().cpu())
        return torch.cat(out).numpy()

    def label_bank(self, names, tpl=TPL):
        """template-averaged, L2-normalised text embedding per name."""
        flat = [t.format(n) for n in names for t in tpl]
        e = l2(self.text(flat)).reshape(len(names), len(tpl), -1).mean(1)
        return l2(e)


def view_mask(m, box, S):
    """mask in the box-view frame (for pooled embeddings)."""
    t = view(np.repeat(m[..., None], 3, -1), m, box, S, 'box', np.zeros(3, np.float32))
    return t[0]


def view(im, m, box, S, mode, mean):
    """im HxWx3 float [0,1], m HxW float mask, box x0,y0,x1,y1 -> (3,S,S) tensor. Same as clip_bench/embed.py prep."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = im.shape[:2]
    if mode == 'stretch':
        c = im[y0:y1, x0:x1]; cm = m[y0:y1, x0:x1]
    else:
        cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.1
        a0, b0 = int(round(cx - s / 2)), int(round(cy - s / 2)); n = max(int(round(s)), 1)
        c = np.ones((n, n, 3), np.float32) * mean; cm = np.zeros((n, n), np.float32)
        sx0, sy0, sx1, sy1 = max(a0, 0), max(b0, 0), min(a0 + n, W), min(b0 + n, H)
        if sx1 > sx0 and sy1 > sy0:
            c[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = im[sy0:sy1, sx0:sx1]
            cm[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = m[sy0:sy1, sx0:sx1]
        if mode == 'masked':
            k = cm[..., None]; c = c * k + mean * (1 - k)
    if c.size == 0:
        c = np.ones((S, S, 3), np.float32) * mean
    t = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1)[None]
    return F.interpolate(t, (S, S), mode='bicubic', align_corners=False).clamp(0, 1)[0]
