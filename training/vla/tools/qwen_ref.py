# 오프라인 한 번 쓰는 기준값 덤프(GPU_TRAINING 9절 "Python 기준값" — 학습·추론 경로에 없음).
# HF transformers(FP32, CPU)로 Qwen3.5 글 몸통의 층별 은닉 상태와 탐욕 디코딩 토큰을 뽑는다.
#   python qwen_ref.py MODEL_DIR OUT_DIR
# 출력: OUT_DIR/manifest.json, p{i}_ids.i32, p{i}_h{l}.f32([L][hidden], l = 0 임베딩, 1..24 층 출력, 25 끝 norm 뒤),
#       p{i}_gen.i32(탐욕 새 토큰), vocab.bin(토큰 번호 → 바이트, 디코딩 표), tok_*.i32(지시·단계 문장 토큰 표).
import json, os, struct, sys

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

mdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(mdir)
model = Qwen3_5ForConditionalGeneration.from_pretrained(mdir, torch_dtype=torch.float32)
model.eval()
lm = model.model.language_model

def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

prompts = [
    chat("You are a mobile robot. Task: bring the cup to the kitchen table. What is the next step?"),
    chat("로봇아, 식탁 위 컵을 집어서 선반에 올려줘. 지금 할 단계는?"),
    "The robot sees a red cup on the table and a shelf next to the door. To put the cup on the shelf, it should first",
    chat("List three objects usually found in a kitchen."),
    chat("The house has a kitchen, a living room, a bathroom and two bedrooms. The map currently contains: a red mug on the "
         "kitchen counter (seen 3 times, confidence 0.91), a wooden chair next to the dining table, a closed fridge, a sofa in the "
         "living room, a bookshelf near the window, a laundry basket in the second bedroom and an unknown box in the corridor. "
         "The robot is in the corridor, facing north, its gripper is empty and the battery is at 64 percent. The user asked: "
         "please take the mug from the kitchen and put it on the bookshelf in the living room, then come back to the charger. "
         "Describe the sequence of subtasks the robot should execute, one per line, using short imperative sentences."),
]
man = {"model": mdir, "prompts": []}
for i, p in enumerate(prompts):
    ids = tok(p, return_tensors="pt").input_ids
    hs = {}
    hooks = []
    for li, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, li=li: hs.__setitem__(li + 1, (o[0] if isinstance(o, tuple) else o).detach())))
    hooks.append(lm.norm.register_forward_hook(lambda m, a, o: hs.__setitem__(len(lm.layers) + 1, o.detach())))
    with torch.no_grad():
        emb = lm.embed_tokens(ids)
        hs[0] = emb
        model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    ids.numpy().astype("int32").tofile(f"{out}/p{i}_ids.i32")
    for l, v in hs.items():
        v[0].float().numpy().tofile(f"{out}/p{i}_h{l}.f32")
    with torch.no_grad():
        g = model.generate(input_ids=ids, max_new_tokens=24, do_sample=False, use_cache=True)
    gen = g[0, ids.shape[1]:]
    gen.numpy().astype("int32").tofile(f"{out}/p{i}_gen.i32")
    man["prompts"].append({"text": p, "n": int(ids.shape[1]), "gen": tok.decode(gen), "n_gen": int(gen.shape[0])})
    print(i, ids.shape[1], repr(tok.decode(gen)), flush=True)

# 디코딩 표: 토큰마다 바이트열(특수 토큰은 그 글자 그대로). [u32 n] ([u32 len][bytes])*
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
k = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + k); k += 1
byte_dec = {chr(c): b for b, c in zip(bs, cs)}   # GPT-2 바이트 수준 글자 → 바이트
special = {int(k): v["content"] for k, v in json.load(open(f"{mdir}/tokenizer_config.json"))["added_tokens_decoder"].items()}
n = model.config.text_config.vocab_size
with open(f"{out}/vocab.bin", "wb") as f:
    f.write(struct.pack("<I", n))
    for t in range(n):
        if t in special:
            b = special[t].encode("utf-8")
        else:
            s = tok.convert_ids_to_tokens(t) if t < len(tok) else None
            b = bytes(byte_dec[ch] for ch in s) if s is not None and all(ch in byte_dec for ch in s) else b""
        f.write(struct.pack("<I", len(b)))
        f.write(b)
json.dump(man, open(f"{out}/manifest.json", "w"), ensure_ascii=False, indent=1)
print("done")
