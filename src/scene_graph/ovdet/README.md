# ovdet — object detector (ObjectSAM segmentation, TensorRT FP16, C API)

`ovdet` is the detector of the scenemap perception stack (`docs/scenemap_설계.md`).

> **Decided (2026-10-05): ObjectSAM (YOLO26n student, engine `models/ovdet/x86_sm120/yolo26n-seg-obj-416.plan`) + SigLIP 2 + objprob.**
> The default engine everywhere (libsgrt glue `SGRT_ENGINE`, `realbag_run`, `dom_bench_det`, the sim/LIMO launchers) is the class-agnostic
> ObjectSAM segmenter: a YOLO26n student distilled from FastSAM-s, things-only (https://github.com/juyoung020/ObjectSAM, release v1.0). Its vocabulary is one name, `object`. Why: about 1/10 of FastSAM-s's compute, so it suits LIMO's Jetson (Nano especially); on-device timing is still to be measured. Names and embeddings come from
> SigLIP 2 B/32 per mask (`src/scene_graph/clip/`), and the scenemap probabilistic object model (`objprob`) fuses them with the per-engine
> parameters `src/scene_graph/tools/realbag/objprob_params/yolo26n-seg-obj-416.json`. The teacher FastSAM-s (`FastSAM-s-416.plan`) stays selectable with `SGRT_ENGINE`.

- **In:** one camera image (ObjectSAM needs no prompt).
- **Out:** a list of objects. Each object has a prompt index, a score, a box and a mask.

The network is a YOLO-seg head (default ObjectSAM, YOLO26n student). It runs in TensorRT (FP16). Everything around the network is hand-written CUDA/C++:

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
cfg.seg_engine = "yolo26n-seg-obj-416.plan"; cfg.names = "yolo26n-seg-obj-416.plan.names.txt";   // ObjectSAM
OvdHandle* h = ovd_create(&cfg, err, sizeof err);
ovd_set_prompt(h, names, n, err, sizeof err);            // once per episode: the prompt table (NULL, 0 = whole vocabulary, e.g. ObjectSAM's "object")
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

ObjectSAM (default, `yolo26n-seg-obj-416.plan`) has one name, `object`, and needs no prompt (`ovd_set_prompt(h, NULL, 0, …)`). It is built by `training/fastsam/build_engine.py`. SigLIP 2 (`../clip`) names each mask afterwards.

### Closed-vocabulary YOLO-seg engines (YOLO11 / YOLO26, COCO-80)

The same code runs Ultralytics YOLO11-seg and YOLO26-seg engines. Their head layout: `output0` 1 x (4 + 80 + 32) x A and `output1` 1 x 32 x h x w.

- **Export:** use `end2end=False`. That is the default ONNX export for these models (metadata `end2end: False`), so ovdet runs its own CUDA NMS. The NMS-free YOLO26 end2end output is not used.
- **Input size:** taken from the engine, for example 640 or 416. The mask grid can be any size, such as 104 x 104 for a 416 input.
- **Names:** `config/coco80.txt`, copied next to the plan as `<plan>.names.txt` (`build_engines.py --names`).
- **Prompt:** prompt names are matched to the 80 classes; names that are not found are reported in `err`. A NULL prompt means all classes. sgrt uses all classes for closed-vocabulary engines (`SGRT_PROMPT`, see `runtime/include/sgrt.h`).

```
cd models/ovdet/onnx416 && python -c "from ultralytics import YOLO; YOLO('../pt/yolo26s-seg.pt').export(format='onnx', imgsz=416, opset=13, end2end=False)"
$TRT_PY tools/build_engines.py models/ovdet/onnx416/yolo26s-seg-416.onnx --names config/coco80.txt --workspace-gb 1
```

## Build (Linux; CUDA 12.8, TensorRT 10)

```
bash scripts/build_linux.sh        # -> build/sgrt/ovdet/libovdet.so, ovdet_smoke
```

The library is built and run on Linux (x86 RTX 50 series; Jetson later).

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

The network share of the student, from `trtexec` per-layer profiling, is backbone 36 %, neck 31 %, box/cls head 17 %, mask prototypes 7 % and mask coefficients 6 %. TensorRT builder optimisation level 5 gave no gain.

## Evaluation

ObjectSAM is evaluated per frame by `training/fastsam/eval_det.py` and end to end (map nodes, found objects, duplicates) by `src/scene_graph/tools/realbag/objprob_eval.py`. Results and the choice are in `docs/model_selection.md`.

## Licence (AGPL-3.0)

ObjectSAM is trained with Ultralytics YOLO26 and its weights are AGPL-3.0 (https://github.com/juyoung020/ObjectSAM). The TensorRT engines are derived from those weights. ovdet itself contains no Ultralytics code.
