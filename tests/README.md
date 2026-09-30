# tests

코드가 제대로 동작하는지 확인하는 곳. 사람도, AI(Claude 등)도 코드를 고친 뒤 여기서 실행해 확인한다.

```bash
python3 -m pytest tests/
```

| 위치 | 용도 | 깃에 올라가나 |
|---|---|---|
| `tests/test_*.py` | 자동 테스트 (pytest). 파트별로 `test_scene_graph.py`, `test_agent.py`, `test_vla.py` 처럼 추가 | O |
| `tests/sandbox/` | AI·사람이 마음대로 돌려보는 실험 공간 (스크립트, 출력, 로그 등) | O |

sandbox 에서 해본 것 중 계속 쓸 만한 건 `test_*.py` 로 옮기거나 `src/`·`scripts/` 로 옮긴다.
