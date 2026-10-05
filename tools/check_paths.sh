#!/bin/bash
# 추적 중인 코드·설정·스크립트가 저장소 밖 홈 경로·behavior-2026·개인 빌드 폴더를 박아 두지 않았는지 본다.
#   tools/check_paths.sh            # 저장소 전체(보고, 걸리면 종료코드 1)
#   tools/check_paths.sh --staged   # 이번 커밋에서 새로 더한 줄만(pre-commit 훅이 씀)
# 걸리는 것: ~/ · $HOME/ · /home/<누구>/ 경로, behavior-2026, ra_*·realbag_build*·sgrt_build* 폴더 이름(뒤에 / 나 끝이 오는 것만 — ra_paths.h 같은 파일 이름은 통과).
# 경로는 config/paths.env 변수(RA_ROOT RA_BUILD RA_MODELS RA_DATASETS B1K_ROOT …)로 읽을 것.
# 안 보는 것: *.md·docs/·refs/(역사·설명), config/ 자신, 이 도구와 훅, conda 환경($HOME/miniconda3 · ~/.config 설정).
# 한 줄만 예외로 하려면 그 줄에 `paths-ok` 를 적는다.
cd "$(git rev-parse --show-toplevel)" || exit 2
PAT='(~|\$HOME|\$\{HOME\}|\$ENV\{HOME\}|\{h\}|\{HOME\}|/home/[a-z][a-z0-9_-]*)/|Path\.home\(\)|behavior-2026|(^|[^A-Za-z0-9_])(ra_[a-z0-9_]+|realbag_build[a-z0-9_]*|sgrt_build[a-z0-9_]*)(/|$|[[:space:]"'\''])'
OKPAT='paths-ok|(~|\$HOME|\$\{HOME\})/(miniconda3|\.config/|\.cache/|\.cargo|\.local/bin)'
SKIP='^(docs/|refs/|config/|training/fastsam/publish_objectsam\.sh|CLAUDE\.md|tools/check_paths\.sh|tools/audit\.py|tools/git-hooks/)|\.(md|pdf|png|jpg|glb|lock|npy|f16|f32|bin|jsonl)$'
if [ "${1:-}" = "--staged" ]; then
  hits=$(git diff --cached -U0 --no-color --diff-filter=ACMR | awk '/^\+\+\+ b\//{f=substr($0,7);next} /^\+[^+]/ || /^\+$/{print f":"substr($0,2)}' \
    | grep -Ev "^($SKIP)" | grep -E "$PAT" | grep -Ev "$OKPAT")
else
  hits=$(git ls-files | grep -Ev "$SKIP" | xargs -d '\n' grep -InE "$PAT" 2>/dev/null | grep -Ev "$OKPAT")
fi
if [ -n "$hits" ]; then
  n=$(printf '%s\n' "$hits" | wc -l)
  echo "check_paths: 저장소 밖 경로·behavior-2026·개인 빌드 폴더 $n 곳 — config/paths.env 변수로 바꿀 것(예외: 줄에 paths-ok)" >&2
  printf '%s\n' "$hits" | cut -c1-200 | head -${CHECK_PATHS_MAX:-40} >&2
  exit 1
fi
echo "check_paths: 깨끗함"
