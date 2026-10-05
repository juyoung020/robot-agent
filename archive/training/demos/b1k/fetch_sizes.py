#!/usr/bin/env python3
"""HF 파일 크기만 목록으로(받지 않음): 원본 HDF5(과제별 판 파일)·LeRobot 영상·data → ~/datasets/b1k_meta/sizes.json (screen.py 가 씀)."""
import json
import os
import time

from huggingface_hub import HfApi

api = HfApi()
R, RAW = "behavior-1k/2026-challenge-demos", "behavior-1k/2026-challenge-rawdata"
sizes = {"raw": {}, "lerobot": {}}
for t in range(100):
    for k in range(5):
        try:
            sizes["raw"][t] = {e.path.split('/')[-1]: e.size for e in api.list_repo_tree(RAW, path_in_repo=f"task-{t:04d}", repo_type="dataset")}
            break
        except Exception as ex:
            print("retry", t, ex, flush=True)
            time.sleep(20 * (k + 1))
for sub in ("videos", "data"):
    for e in api.list_repo_tree(R, path_in_repo=sub, repo_type="dataset", recursive=True):
        if getattr(e, "size", None):
            sizes["lerobot"][e.path] = e.size
os.makedirs(os.path.expanduser("~/datasets/b1k_meta"), exist_ok=True)
json.dump(sizes, open(os.path.expanduser("~/datasets/b1k_meta/sizes.json"), "w"))
print("done", len(sizes["raw"]), len(sizes["lerobot"]))
