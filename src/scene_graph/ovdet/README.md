# ovdet — open-vocabulary object detector (YOLOE, TensorRT FP16, C API)

`ovdet` is the detector of the scenemap perception stack (`docs/scenemap_설계.md`).

> ovdet (YOLOE) is what the code uses today, but object recognition is moving to FastSAM-s 416 + SigLIP 2 B/32 (`src/scene_graph/clip/`, in progress).

- **In:** one camera image and a prompt, the task's BDDL object names.
- **Out:** a list of objects. Each object has a prompt index, a score, a box and a mask.

The network is a YOLOE text-prompt segmentation head. It runs in TensorRT (FP16). Everything around the network is hand-written CUDA/C++:

- letterbox
- class selection under the prompt
- score sort
- greedy NMS
- mask assembly on the prototype grid
- area gate
- mask de-duplication

There is no Python and no ROS at run time, so the evaluator process calls it directly.

## API (`include/ovdet.h`)

```c
OvdConfig cfg; ovd_default_config(&cfg);
cfg.seg_engine = "yoloe-11l-all.plan"; cfg.names = "yoloe-11l-all.plan.names.txt";
OvdHandle* h = ovd_create(&cfg, err, sizeof err);
ovd_set_prompt(h, names, n, err, sizeof err);            // once per episode: the prompt table
const sm_detections* d = ovd_detect(h, &img, &timing);   // per image; valid until the next call
```

- **Output struct.** The output is `sm_detections`, the detector contract of `docs/scenemap_설계.md` 4.2, field for field. `scenemap.h` uses the same `SM_DETECTIONS_DEFINED` guard, so the two headers can be included together.
  - `cls`: index into the prompt table, i.e. the order of the names given to `ovd_set_prompt`.
  - `score`: the class score.
  - `box`: x0, y0, x1, y1 in input pixels.
  - `stamp` and `cam`: passed through from the input image.
- **Masks.** Masks stay on the detector's grid, which is the prototype grid: 256 x 256 for a 1024 input.
  - Each mask is a row-major bit array. Cell `k = j * mask_w + i` is bit `k & 31` (LSB first) of word `k >> 5`.
  - Input pixel = cell × `mask_s` + `mask_o`, which undoes the letterbox. Cell (i, j) covers x in [i·sx + ox, (i+1)·sx + ox).
  - Nothing is copied: the arrays belong to the handle.
- **Prompt names.** Names are matched to the engine's vocabulary after normalisation: `.n.NN` dropped, `_` → space, lower case. So `radio_receiver.n.01` matches `radio receiver`.
  - A name outside the vocabulary keeps its index, but is never detected.
  - Such names are reported in `err`.
- **Input image.** RGB, BGR or RGBA u8, in host or device memory, with any row stride.
- **Thread safety.** One handle per thread.

## Engines

The engine is built once with a whole vocabulary: every task's BDDL objects plus 18 scene structures, 272 names in `config/vocab_all.txt`.

A prompt switches classes on and off. YOLOE's class scores are independent sigmoids per class, so the result equals an engine exported with only the prompt's names. `config/task_prompts.txt` lists each task's names, plus the `_scene` line.

```
~/ovdet_export_venv/bin/python tools/export_yoloe.py --model yoloe-11l-seg --vocab all --out ~/ovdet_models/onnx/yoloe-11l-all.onnx   # CPU, Ultralytics
~/ovdet_venv/bin/python tools/build_engines.py ~/ovdet_models/onnx/yoloe-11l-all.onnx                                              # GPU lock
```

### Closed-vocabulary YOLO-seg engines (YOLO11 / YOLO26, COCO-80)

The same code runs Ultralytics YOLO11-seg and YOLO26-seg engines. Their head has the same layout as YOLOE: `output0` 1 x (4 + 80 + 32) x A and `output1` 1 x 32 x h x w.

- **Export:** use `end2end=False`. That is the default ONNX export for these models (metadata `end2end: False`), so ovdet runs its own CUDA NMS. The NMS-free YOLO26 end2end output is not used.
- **Input size:** taken from the engine, for example 640 or 416. The mask grid can be any size, such as 104 x 104 for a 416 input.
- **Names:** `config/coco80.txt`, copied next to the plan as `<plan>.names.txt` (`build_engines.py --names`).
- **Prompt:** prompt names are matched to the 80 classes; names that are not found are reported in `err`. A NULL prompt means all classes. sgrt uses all classes for closed-vocabulary engines (`SGRT_PROMPT`, see `runtime/include/sgrt.h`).

```
cd ~/ovdet_models/onnx416 && ~/ovdet_export_venv/bin/python -c "from ultralytics import YOLO; YOLO('../pt/yolo26s-seg.pt').export(format='onnx', imgsz=416, opset=13, end2end=False)"
~/ovdet_venv/bin/python tools/build_engines.py ~/ovdet_models/onnx416/yolo26s-seg-416.onnx --names config/coco80.txt --workspace-gb 1
```

## Build (Linux; CUDA 12.8, TensorRT 10)

```
bash scripts/build_linux.sh        # -> ~/ovdet_build/libovdet.so, ovdet_smoke
```

The library is built and run on Linux, which is also the submission Docker's OS.

## Per-call latency (CUDA graph)

At 416 the network is small enough that one call was launch-bound. `nsys` on the YOLO26n student showed about 235 TensorRT kernel launches per frame at about 2.7 µs each, plus a host round trip in the middle of post-processing. Our own CUDA post-processing was only about 70 µs per frame: select/NMS 31, mask de-duplication 14, mask bits 13, the rest under 5.

`ovd_detect` now does the following:

- **One graph per frame.** Letterbox, TensorRT `enqueueV3`, post-processing and the output copies are captured once into a CUDA graph and replayed every frame.
  - The graph is rebuilt when the input size or layout changes.
  - A device input is first copied into the handle's buffer, so the graph always reads the same address.
- **One host sync.** The output lanes (valid, class in the prompt) are compacted on the device (`k_compact`), so there is a single sync at the end.
  - The host copies `max_det` lanes of mask bits and keeps the first n.
- **Pinned staging.** A host image goes through a pinned staging buffer instead of a pageable `cudaMemcpy2DAsync`.
- **Fallback.** `OVDET_NO_GRAPH=1` runs the same work stream-ordered, without capture.

Outputs are bit-identical to the previous code: scores, boxes, classes and mask bits on 400 frames (radio r3 and COCO/ADE val), for both 416 engines, with and without the graph.

Median `ovd_detect` latency on an idle RTX 5070 Ti, 300 radio r3 frames 640×480, with `OvdTiming` fields:

| engine | before: total / upload+letterbox / net / post+out ms | after |
|---|---|---|
| yolo26n-seg-obj-416 | 0.934 / 0.222 / 0.612 / 0.103 | 0.624 / 0.158 / 0.392 / 0.081 |
| FastSAM-s-416-obj | 0.892 / 0.217 / 0.545 / 0.112 | 0.617 / 0.115 / 0.420 / 0.071 |

The network share of the student, from `trtexec` per-layer profiling, is backbone 36 %, neck 31 %, box/cls head 17 %, mask prototypes 7 % and mask coefficients 6 %. TensorRT builder optimisation level 5 gave no gain.

## Detector comparison

`scripts/eval_linux.sh` runs `tools/ovdet_eval.py` on the same frames for every head:

- ep0, 0–40 s, every 5th frame
- ep200, whole episode, every 15th frame

Ground truth works as follows:

1. Depth pixels on a 2-px grid are placed in the map with the ground-truth camera pose.
2. Each point is labelled by `gt_scene` (`src/scene_graph/scenemap/eval/`) with the GT object whose box contains it.

The earlier FastSAM + CLIP comparison row has been retired; its numbers remain in `docs/ovdet_검출기.md`.

The results and the reasoning behind the choice are in `docs/ovdet_검출기.md`.

## Licence (AGPL-3.0)

YOLOE's code and weights are AGPL-3.0:

- Ultralytics (`ultralytics` 8.4, `yoloe-11*-seg.pt`)
- THU-MIG (`THU-MIG/yoloe`)

The TensorRT engines are derived from those weights. ovdet itself contains no Ultralytics code.

The competition submission (Docker image given to the organizers) therefore ships AGPL-covered weights. On 2026-09-30 the user decided that **the submission's source is published under AGPL-3.0**. `docs/제출지침.md` lists this as a submission checklist item: a source link and the LICENSE go into the README.

## Status (2026-09-30) and what is left

Done:

- The library and its C API, output in the `sm_detections` format.
- The Linux build.
- The FastSAM + CLIP path, compared and retired.
- The detector comparison, including a confidence sweep. The recommendation is YOLOE-11m, the 272-name engine, `conf_th` 0.10. Details are in `docs/ovdet_검출기.md`.
- The AGPL notes.

Left:

1. **The ep0 radio is never found** with the task prompt "radio receiver". This holds even at conf 0.05. With the whole 272-name vocabulary it is found but named "satchel". Next step: add synonyms such as "radio" to the vocabulary, re-export, and re-run the comparison.
2. `tools/ref_check.py` compares ovdet with Ultralytics' own FP32 prediction. It is written but has not been run.
3. Coffee tables are often named "floor" or "rug". Check whether the cause is the GT labels (floor points inside the table's GT box).
4. Whether to make the recommended setting the default of `ovd_default_config` is not decided yet. It depends on scenemap's object-map scores.
