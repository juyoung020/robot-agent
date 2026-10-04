"""Embed head h (train_head.py Head, depth 1) → raw FP32 file for the C++ appearance tool (training/BC/tools/app_head.cpp). OFFLINE.

  ~/embed_venv/bin/python training/embed/export_head_f32.py [run=sb32_pe_300k]   # -> ~/embed_work/runs/<run>/head_h.f32
Order: ln.weight, ln.bias, blocks.0.0.weight [1024][768], blocks.0.0.bias, blocks.0.3.weight [768][1024], blocks.0.3.bias, out.weight [128][768], out.bias.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

run = sys.argv[1] if len(sys.argv) > 1 else 'sb32_pe_300k'
h = torch.load(f'{WORK}/runs/{run}/head.pt', map_location='cpu', weights_only=False)['h']
keys = ['ln.weight', 'ln.bias', 'blocks.0.0.weight', 'blocks.0.0.bias', 'blocks.0.3.weight', 'blocks.0.3.bias', 'out.weight', 'out.bias']
assert set(keys) == set(h), sorted(h)
out = f'{WORK}/runs/{run}/head_h.f32'
np.concatenate([h[k].float().numpy().ravel() for k in keys]).astype(np.float32).tofile(out)
print(out, sum(h[k].numel() for k in keys), 'floats')
