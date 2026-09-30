#!/usr/bin/env bash
# refs/README.md 의 arXiv 논문 PDF를 refs/papers/ 에 받는다 (깃에는 올라가지 않음).
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
echo "완료: $(ls papers/*.pdf | wc -l)개"
