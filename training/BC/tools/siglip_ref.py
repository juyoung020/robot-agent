#!/usr/bin/env python3
"""오프라인 한 번 쓰는 기준값 덤프(검증 전용 — 학습·추론 경로에 들어가지 않음, README "Python 기준값" 규칙).

open_clip `ViT-B-32-SigLIP2-256`(webli, HF 캐시에 이미 있는 가중치, 내려받지 않음)의 영상 탑으로
vit_verify dump 가 쓴 DIR/imgs.u8 (n × [256][256][3] u8) 의 끝 LN 뒤 패치 토큰을 FP32 로 DIR/ref_tok.f32 ([n × 64][768]) 에 쓴다.
전처리는 GPU K11 과 같은 식: x = u8/255, (x − 0.5)/0.5 (크기 조정 없음 — 이미 256²).

  ~/clip_venv/bin/python training/BC/tools/siglip_ref.py DIR
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402


def main():
    d = sys.argv[1]
    raw = np.fromfile(os.path.join(d, "imgs.u8"), dtype=np.uint8)
    n = raw.size // (256 * 256 * 3)
    x = torch.from_numpy(raw.reshape(n, 256, 256, 3)).permute(0, 3, 1, 2).float().div(255.0)
    x = (x - 0.5) / 0.5
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-SigLIP2-256", pretrained="webli")
    model.eval()
    trunk = model.visual.trunk
    print("gelu:", trunk.blocks[0].mlp.act, "ln eps:", trunk.blocks[0].norm1.eps, "img:", trunk.patch_embed.img_size, "has cls:", trunk.cls_token is not None)
    with torch.no_grad():
        out = []
        for i in range(0, n, 16):
            out.append(trunk.forward_features(x[i:i + 16]).float())
        tok = torch.cat(out).reshape(n * 64, 768).numpy().astype(np.float32)
    tok.tofile(os.path.join(d, "ref_tok.f32"))
    print(f"ref: {n} images -> {d}/ref_tok.f32 {tok.shape}")


if __name__ == "__main__":
    main()
