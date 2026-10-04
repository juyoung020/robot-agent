# training/model — 베이스 모델 (저장소에 없음, 직접 받는다)

| 폴더 | 출처 | 라이선스 | 크기 | 용도 |
|---|---|---|---|---|
| `Qwen3.5-2B/` | Hugging Face `Qwen/Qwen3.5-2B` | Apache-2.0 | 4.3 GB | VLA 몸통. 하이브리드 24층(선형 어텐션 18 + 풀 어텐션 6)과 **자체 비전 타워**(24층)를 가짐 |
| `smolvla_base/` | Hugging Face `lerobot/smolvla_base` | Apache-2.0 | 0.9 GB | 액션 전문가 구조 참고용 |
| `siglip2_b32/` | 팀 `src/scene_graph/clip/tools` 결과(ONNX + TensorRT 엔진) | — | 0.7 GB | **물체 임베딩 검색용**. 출력이 풀링된 768차원 벡터 하나라서 VLA 비전 인코더로는 못 쓴다. BC 영상 학생은 같은 모델의 영상 탑을 HF 캐시 safetensors(`models--timm--ViT-B-32-SigLIP2-256`)에서 직접 읽는 C++/CUDA 판(`BC/src/vit.cu`, 패치 토큰)을 쓴다 |

받기: `huggingface-cli download Qwen/Qwen3.5-2B --local-dir training/model/Qwen3.5-2B`, `huggingface-cli download lerobot/smolvla_base --local-dir training/model/smolvla_base`. `siglip2_b32` 는 `src/scene_graph/clip` 의 `tools/export_siglip2.py` → `tools/build_engine.py` 로 만든다([clip/README.md](../../src/scene_graph/clip/README.md)).
