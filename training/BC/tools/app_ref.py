#!/usr/bin/env python3
"""PyTorch FP32 reference for the appearance tool (OFFLINE check, HF cache only — not on any training/inference path).

  app_table --dump DIR [--negative B];  ~/embed_venv/bin/python training/BC/tools/app_ref.py DIR [--min 0.999]
(1) same tokens: open_clip ViT-B-32-SigLIP2-256 trunk.attn_pool + embed head h (train_head.Head) on the C++ tool's encoder tokens
    vs the C++ MAP pool + head (apph::encode)  → isolates the pool+head implementation.
(2) end to end: open_clip image tower on the same rendered RGB (no mask = plain MAP) + head h vs the C++ path
    (our C++ SigLIP 2 tower FP16 → bf16 tokens → C++ pool+head).
Exit 1 if (1) min cosine of the 128-d output < --min (negative-control dumps must fail).
"""
import os, sys, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'embed'))
import numpy as np, torch, torch.nn.functional as F
import open_clip
from train_head import Head

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--min', type=float, default=0.999)
ap.add_argument('--head', default=os.path.expanduser('~/embed_work/runs/sb32_pe_300k/head.pt')); a = ap.parse_args()
NI, E, V, bug = map(int, open(f'{a.dir}/meta.txt').read().split())
tok = torch.from_numpy(np.fromfile(f'{a.dir}/tokens.f32', np.float32).reshape(NI, 64, 768))
pc = np.fromfile(f'{a.dir}/pooled_cpp.f32', np.float32).reshape(NI, 768)
oc = np.fromfile(f'{a.dir}/out_cpp.f32', np.float32).reshape(NI, 128)
m, _, _ = open_clip.create_model_and_transforms('ViT-B-32-SigLIP2-256', pretrained='webli'); m.eval()
ck = torch.load(a.head, map_location='cpu', weights_only=False)
h = Head(768, 128, ck['args']['hid'], ck['args']['depth']); h.load_state_dict(ck['h']); h.eval()
cos = lambda x, y: (F.normalize(torch.as_tensor(x).double(), dim=-1) * F.normalize(torch.as_tensor(y).double(), dim=-1)).sum(-1)
with torch.no_grad():
    tr = m.visual.trunk
    pr = tr.attn_pool(tok)
    orf = F.normalize(h(pr), dim=-1)
    c1p, c1 = cos(pr, pc), cos(orf, oc)
    print(f'(1) same tokens, C++ vs PyTorch: pooled cos mean {c1p.mean():.7f} min {c1p.min():.7f}; 128-d cos mean {c1.mean():.7f} min {c1.min():.7f} '
          f'(rel L2 pooled {((torch.as_tensor(pc) - pr).norm(dim=-1) / pr.norm(dim=-1)).max():.2e})')
    rgb = np.stack([np.fromfile(f'{a.dir}/rgb{c}.u8', np.uint8).reshape(E, 256, 256, 3) for c in range(2)], 1).reshape(NI, 256, 256, 3)
    x = (torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255. - 0.5) / 0.5
    full = []
    for i in range(0, NI, 32):
        full.append(F.normalize(h(m.encode_image(x[i:i + 32])), dim=-1))
    full = torch.cat(full)
    c2 = cos(full, oc)
    print(f'(2) end to end, C++ tower+pool+head vs PyTorch FP32 from the same RGB: 128-d cos mean {c2.mean():.5f} min {c2.min():.5f}')
    # class means (the table rows) from the PyTorch path vs C++
    rows = NI // (2 * V)
    mr = F.normalize(full.reshape(rows, 2 * V, 128).mean(1), dim=-1); mc = F.normalize(torch.as_tensor(oc).reshape(rows, 2 * V, 128).mean(1), dim=-1)
    print('    table rows (class means) cos:', ' '.join(f'{v:.5f}' for v in cos(mr, mc).tolist()))
ok = c1.min().item() >= a.min
print('PASS' if ok else 'FAIL', f'(negative control {bug})' if bug else '')
sys.exit(0 if ok else 1)
