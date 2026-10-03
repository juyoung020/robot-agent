# training/model — 베이스 모델 (저장소에 없음, 직접 받는다)

| 폴더 | 출처 | 라이선스 | 크기 | 용도 |
|---|---|---|---|---|
| `Qwen3.5-2B/` | Hugging Face `Qwen/Qwen3.5-2B` | Apache-2.0 | 4.3 GB | VLA 몸통. 하이브리드 24층(선형 어텐션 18 + 풀 어텐션 6)과 **자체 비전 타워**(24층)를 가짐 |
| `smolvla_base/` | Hugging Face `lerobot/smolvla_base` | Apache-2.0 | 0.9 GB | 액션 전문가 구조 참고용 |
| `siglip2_b32/` | 팀 `training/embed` 결과(ONNX + TensorRT 엔진) | — | 0.7 GB | **물체 임베딩 검색용**. 출력이 풀링된 768차원 벡터 하나라서 VLA 비전 인코더로는 못 쓴다 |

받기: `huggingface-cli download Qwen/Qwen3.5-2B --local-dir training/model/Qwen3.5-2B`, `huggingface-cli download lerobot/smolvla_base --local-dir training/model/smolvla_base`. `siglip2_b32` 는 `training/embed` 로 만든다.
