#!/usr/bin/env python3
"""지시 문장 글 벡터 표(가정 — A2 과제 "컵으로 가" 의 바꿔 말하기 8 개)를 오프라인에서 한 번 만든다.

VLA_INPUT 1·6절: 지시 문장은 얼린 SigLIP 2 글 인코더 벡터로 넣고 같은 뜻 다른 말로 흔든다. 글 탑을 C++ 로 옮기지 않고
(학습 중 글 인코더를 돌리지 않음 — 계획서 4.4 의 이름 벡터와 같은 "미리 계산한 표" 방식) open_clip `ViT-B-32-SigLIP2-256`(webli, HF 캐시,
내려받지 않음)의 글 탑으로 L2 정규화 768-d 를 뽑아 √768 배(원소 크기 약 1) 한 f32 [k][768] 을 쓴다. 학습·추론은 이 표만 읽는다(C++).
SigLIP 2 글 탑은 다국어(Gemma 토크나이저)라 한국어 문장도 같은 공간에 들어간다.

  ~/clip_venv/bin/python training/BC/tools/text_table.py training/BC/data/instr_a2.f32
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402

SENTENCES = [
    "go to the cup",
    "move to the cup",
    "drive over to the mug",
    "approach the red cup",
    "find the cup and go to it",
    "컵으로 가",
    "컵 쪽으로 이동해",
    "머그잔 앞으로 가줘",
]


def main():
    out = sys.argv[1]
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-SigLIP2-256", pretrained="webli")
    tok = open_clip.get_tokenizer("ViT-B-32-SigLIP2-256")
    model.eval()
    with torch.no_grad():
        e = model.encode_text(tok(SENTENCES), normalize=True).float().numpy()
    e = (e * np.sqrt(768.0)).astype(np.float32)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    e.tofile(out)
    with open(out + ".txt", "w") as f:
        f.write("\n".join(SENTENCES) + "\n")
    sim = e @ e.T / 768.0
    print(f"{len(SENTENCES)} sentences -> {out} {e.shape}; cos between paraphrases min {sim.min():.3f} mean {sim[np.triu_indices(len(e), 1)].mean():.3f}")


if __name__ == "__main__":
    main()
