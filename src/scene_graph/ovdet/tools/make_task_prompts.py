#!/usr/bin/env python3
"""Task prompt table of the open-vocabulary segmenter: every challenge task -> its BDDL object categories, plus the
scene structures ('_scene'). Names are the category vocabulary's (config/vocab_all.txt), so each is a class of the
YOLOE 'all' engine.

  python make_task_prompts.py   # -> src/scene_graph/ovdet/config/task_prompts.txt
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # repository root
OUT = f'{ROOT}/src/scene_graph/ovdet/config/task_prompts.txt'
SCENE = ['wall', 'floor', 'ceiling', 'door', 'window', 'rug', 'curtain', 'picture frame', 'lamp', 'plant',
         'staircase', 'railing', 'baseboard', 'light switch', 'electric outlet', 'radiator', 'sofa', 'shelf']


def main():
    tasks = [json.loads(l)['task_name'] for l in
             open(f'{ROOT}/data/2026-challenge-demos/meta/tasks.jsonl', encoding='utf-8')]
    lines = ['# task: BDDL object categories + "_scene" structures (ovdet prompt table, tools/make_task_prompts.py)',
             '_scene: ' + ', '.join(SCENE)]
    for t in tasks:
        txt = open(f'{ROOT}/BEHAVIOR-1K/bddl3/bddl/activity_definitions/{t}/problem0.bddl', encoding='utf-8').read()
        m = re.search(r'\(:objects(.*?)\)\s*\(:init', txt, re.S)
        names = []
        for s in re.findall(r'-\s+([a-z0-9_]+\.n\.\d+)', m.group(1) if m else ''):
            if s in ('agent.n.01', 'floor.n.01'):
                continue
            n = re.sub(r'\s+', ' ', s.split('.n.')[0].replace('_', ' ')).strip()
            if n not in names:
                names.append(n)
        lines.append(f'{t}: ' + ', '.join(names))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{len(tasks)} tasks -> {OUT}')


if __name__ == '__main__':
    main()
