#!/usr/bin/env bash
# refs/README.md 의 논문 PDF는 refs/papers/, 참고 코드는 refs/code/ 에 받는다 (둘 다 깃에는 올라가지 않음).
set -e
cd "$(dirname "$0")" && mkdir -p papers
while read -r id name; do
  [ -f "papers/$name.pdf" ] && continue
  echo "받는 중: $name ($id)"
  curl -sSL --fail -A "Mozilla/5.0" -o "papers/$name.pdf" "https://arxiv.org/pdf/$id"
  sleep 1
done <<'LIST'
2504.16054 pi05
2410.24164 pi0
2410.11989 DovSG
2201.13360 Hydra
2402.13817 Khronos
2309.16650 ConceptGraphs
2403.17846 HOV-SG
2307.06135 SayPlan
2403.08605 MoMa-LLM
2512.06951 BEHAVIOR25-1st-RLC
2512.10071 BEHAVIOR25-2nd-Comet
2607.06256 BEHAVIOR25-3rd-SimpleAI
LIST
echo "논문 완료: $(ls papers/*.pdf | wc -l)개"

mkdir -p code
for repo in \
  Physical-Intelligence/openpi \
  BJHYZJ/DovSG \
  MIT-SPARK/Hydra \
  MIT-SPARK/Khronos \
  concept-graphs/concept-graphs \
  hovsg/HOV-SG \
  robot-learning-freiburg/MoMa-LLM \
  IliaLarchenko/behavior-1k-solution \
  mli0603/openpi-comet; do
  dir="code/${repo#*/}"
  [ -d "$dir" ] && continue
  echo "클론 중: $repo"
  git clone -q --depth 1 "https://github.com/$repo.git" "$dir"
done
echo "코드 완료: $(ls -d code/*/ | wc -l)개"
