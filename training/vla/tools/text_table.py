# 오프라인 한 번: Qwen3.5 토크나이저로 RecallVLA 글 표(틀·지시·단계 문장 토큰 번호)를 만든다. 학습·추론 경로에는 표만 쓴다(MAPVLA_SPEC 4.1).
#   python text_table.py MODEL_DIR OUT.bin
# 형식: [u32 n] ([u32 kind][u32 len][i32 × len])*, kind 0 = 열 앞 틀, 1 = vision_start, 2 = vision_end, 3 = 지시 뒤 틀(im_end \n im_start assistant \n),
#       4 = 단계 문장 끝(im_end), 10 = 지시 문장, 11 = 단계 문장. 같은 이름 OUT.txt 에 사람이 읽는 목록.
import struct, sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = sys.argv[2]
ents = []
def add(kind, text, special=False):
    ids = tok(text, add_special_tokens=False).input_ids
    ents.append((kind, text, ids))
add(0, "<|im_start|>user\n")
add(1, "<|vision_start|>")
add(2, "<|vision_end|>")
add(3, "<|im_end|>\n<|im_start|>assistant\n")
add(4, "<|im_end|>")
objs = ["cup", "mug", "bowl", "can of soda", "toy figure", "book"]
srcs = ["table", "floor", "coffee table", "shelf"]
dsts = ["trash can", "basket", "toy box", "coffee table"]
rooms = ["kitchen", "living room", "bedroom", "bathroom"]
instr = [f"bring the {o} to the {d}" for o in objs[:3] for d in dsts[:2]] + [f"put the {o} in the {d}" for o in objs[3:] for d in dsts[2:]] + \
        ["컵을 쓰레기통에 넣어줘", "식탁 위 컵을 집어서 바구니에 넣어"]
for s in instr: add(10, s)
sub = [f"go to the {r}" for r in rooms] + [f"find the {o}" for o in objs] + \
      [f"the {o} is not in the map; explore the {r}" for o in objs[:2] for r in rooms[:2]] + \
      [f"pick up the {o} from the {s}" for o in objs[:3] for s in srcs[:2]] + \
      [f"put the {o} on the {d}" for o in objs[:2] for d in dsts[3:]] + [f"put the {o} in the {d}" for o in objs[:2] for d in dsts[:2]]
for s in sub: add(11, s)
with open(out, "wb") as f:
    f.write(struct.pack("<I", len(ents)))
    for k, _, ids in ents:
        f.write(struct.pack("<II", k, len(ids)))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
with open(out.rsplit(".", 1)[0] + ".txt", "w") as f:
    for k, t, ids in ents:
        f.write(f"{k}\t{len(ids)}\t{t!r}\t{ids}\n")
print(len(ents), "entries; max subtask len", max(len(i) for k, _, i in ents if k == 11), "max instr len", max(len(i) for k, _, i in ents if k == 10))
