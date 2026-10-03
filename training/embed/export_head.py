"""Export a trained head to ONNX (runtime: 768-d SigLIP 2 vector -> L2-normalised 128-d; first 64 = 64-d) + 128-d label table.

  python export_head.py <run>      # -> ~/embed_work/runs/<run>/head128.onnx (opset 13, dynamic batch)

Graph = LayerNorm -> residual MLP (GELU erf) -> Linear -> L2 norm. Only MatMul/Add/Erf/ReduceMean/Div/Sqrt ops (TRT 8.2 ok).
The input is the stored per-object vector itself (L2-normalised emb), so the head can run on the CPU too (~1.6 MFLOP).
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from train_head import Head, DIMS


class Exp(nn.Module):
    def __init__(self, h): super().__init__(); self.h = h
    def forward(self, x): return F.normalize(self.h(F.normalize(x, dim=-1)), dim=-1)


def main(run):
    R = f'{WORK}/runs/{run}'; ck = torch.load(f'{R}/head.pt', map_location='cpu'); a = ck['args']
    h = Head(ck['Db'], DIMS[0], a['hid'], a['depth']); h.load_state_dict(ck['h']); m = Exp(h).eval()
    x = torch.randn(8, ck['Db'])
    torch.onnx.export(m, x, f'{R}/head128.onnx', input_names=['emb'], output_names=['emb128'], opset_version=13,
                      dynamic_axes={'emb': {0: 'n'}, 'emb128': {0: 'n'}}, dynamo=False)
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(f'{R}/head128.onnx', providers=['CPUExecutionProvider'])
    d = np.abs(s.run(None, {'emb': x.numpy()})[0] - m(x).detach().numpy()).max()
    print(run, 'onnx ok, max diff', d, 'params', sum(p.numel() for p in h.parameters()))


if __name__ == '__main__':
    for r in sys.argv[1:]: main(r)
