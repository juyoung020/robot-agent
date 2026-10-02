# 작업 규칙

## 작업 단위마다 pull → 작업 → 커밋·푸시
- 작업을 시작하기 전에: `git pull --rebase --autostash` 와 `git submodule update --init src/behavior-2026`.
- 작업 하나가 끝나면 바로: 바뀐 경로만 지정해 커밋(`git add <경로>`, `git add -A` 금지) → `git push`.
- 서브모듈(`src/behavior-2026`)을 고쳤으면 서브모듈 안에서 먼저 커밋·푸시하고(그 저장소의 `CLAUDE.md`), 상위 저장소에서 서브모듈 포인터를 커밋·푸시한다.
- 참고 논문·코드(`refs/papers/`, `refs/code/`)는 커밋하지 않는다.
