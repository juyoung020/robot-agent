# 오프라인 한 번 쓰는 RoPE 기준값 덤프(qwen_verify rope 가 읽음). HF transformers FP32 CPU.
#   python qwen_rope_ref.py MODEL_DIR REF_DIR      (REF_DIR = qwen_ref.py 출력 — p{i}_ids.i32 를 다시 쓰고, 긴 프롬프트 하나를 더함)
# 풀 어텐션 층 f(0..5)마다: r{i}_pq{f}.f32 = q_proj 출력 [n][nq·2·hd](머리마다 q|gate), r{i}_pk{f}.f32 = k_proj 출력 [n][nkv·hd],
#                          r{i}_rq{f}.f32 = q_norm + RoPE 뒤 [n][nq][hd], r{i}_rk{f}.f32 = k_norm + RoPE 뒤 [n][nkv][hd]
# r{i}_ids.i32 = 토큰. 프롬프트 = p0..p4 + 긴 글(약 600 토큰, 위치가 큰 쪽).
import os, sys

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

mdir, ref = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(mdir)
model = Qwen3_5ForConditionalGeneration.from_pretrained(mdir, torch_dtype=torch.float32)
model.eval()
lm = model.model.language_model
full = [i for i, l in enumerate(lm.layers) if hasattr(l, "self_attn")]

seqs = []
i = 0
while os.path.exists(f"{ref}/p{i}_ids.i32"):
    seqs.append(np.fromfile(f"{ref}/p{i}_ids.i32", dtype=np.int32))
    i += 1
long_text = " ".join(
    f"Step {k}: the robot checks room {k % 7} for the {['cup', 'book', 'phone', 'towel', 'plate'][k % 5]}, updates its map with {k * 3 % 11} objects, "
    f"and reports the result to the agent before moving on." for k in range(40))
seqs.append(tok(long_text, return_tensors="np").input_ids[0][:600].astype(np.int32))

cur = {}
orig = mq.apply_rotary_pos_emb
def rec(q, k, cos, sin, unsqueeze_dim=1):
    qo, ko = orig(q, k, cos, sin, unsqueeze_dim)
    cur.setdefault("rq", []).append(qo.detach().clone())
    cur.setdefault("rk", []).append(ko.detach().clone())
    return qo, ko
mq.apply_rotary_pos_emb = rec

for si, ids in enumerate(seqs):
    cur.clear()
    hooks = []
    for f, li in enumerate(full):
        a = lm.layers[li].self_attn
        hooks.append(a.q_proj.register_forward_hook(lambda m, i_, o, f=f: cur.__setitem__(("pq", f), o.detach().clone())))
        hooks.append(a.k_proj.register_forward_hook(lambda m, i_, o, f=f: cur.__setitem__(("pk", f), o.detach().clone())))
    with torch.no_grad():
        model(input_ids=torch.from_numpy(ids[None].astype(np.int64)), use_cache=False)
    for h in hooks:
        h.remove()
    ids.tofile(f"{ref}/r{si}_ids.i32")
    for f in range(len(full)):
        cur[("pq", f)][0].float().numpy().tofile(f"{ref}/r{si}_pq{f}.f32")
        cur[("pk", f)][0].float().numpy().tofile(f"{ref}/r{si}_pk{f}.f32")
        cur["rq"][f][0].transpose(0, 1).contiguous().float().numpy().tofile(f"{ref}/r{si}_rq{f}.f32")   # [n][nq][hd]
        cur["rk"][f][0].transpose(0, 1).contiguous().float().numpy().tofile(f"{ref}/r{si}_rk{f}.f32")
    print(si, len(ids), flush=True)
print("done")
