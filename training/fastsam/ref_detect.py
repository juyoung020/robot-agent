"""Reference detector: ONNX Runtime + numpy post-processing identical to the ovdet TensorRT path
(letterbox 114, conf > th, class-agnostic greedy NMS 0.7, max 100, mask = proto logit > 0 inside the box and the
non-padded image region on the 1/4 grid, area >= 24 cells, greedy mask-IoU de-duplication 0.7).

Same interface as common.Detector: detect(rgb) -> dict(score, box, grid, masks(full-res bool)).
Used when libovdet / TensorRT is not available (e.g. the standalone ObjectSAM repo).
"""
import numpy as np


class OrtDetector:
    def __init__(self, onnx_path, conf_th=0.25, nms_iou=0.7, mask_iou=0.7, area_min=24, max_det=100, threads=8):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        shp = self.s.get_inputs()[0].shape
        self.S = int(shp[2])
        self.conf_th, self.nms_iou, self.mask_iou, self.area_min, self.max_det = conf_th, nms_iou, mask_iou, area_min, max_det
        self.ms, self.mem_mb = [], 0.0

    def detect(self, rgb, full=True):
        import time
        import cv2
        t0 = time.perf_counter()
        H, W = rgb.shape[:2]
        S = self.S
        r = min(S / H, S / W)
        nw, nh = int(round(W * r)), int(round(H * r))
        left, top = (S - nw) // 2, (S - nh) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb[..., :3], (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        o0, o1 = self.s.run(None, {self.s.get_inputs()[0].name: x})
        o0, o1 = o0[0], o1[0]                       # (4+nc+32, A), (32, Ph, Pw)
        nc = o0.shape[0] - 4 - o1.shape[0]
        conf = o0[4:4 + nc].max(0)
        idx = np.nonzero(conf > self.conf_th)[0]
        idx = idx[np.argsort(-conf[idx], kind='stable')]
        xywh = o0[:4, idx].T
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        keep = []
        for i in range(len(idx)):                   # greedy NMS in score order
            if len(keep) >= self.max_det:
                break
            b = boxes[i]
            ok = True
            for k in keep:
                c = boxes[k]
                iw = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
                ih = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
                inter = iw * ih
                u = (b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter
                if u > 0 and inter / u > self.nms_iou:
                    ok = False
                    break
            if ok:
                keep.append(i)
        Ph, Pw = o1.shape[1:]
        g = Pw / S
        coef = o0[4 + nc:, idx[keep]].T              # (n, 32)
        logits = coef @ o1.reshape(o1.shape[0], -1)   # (n, Ph*Pw)
        yy, xx = np.mgrid[0:Ph, 0:Pw]
        rt, rb = int(np.floor(top * g)), int(np.ceil((top + nh) * g))
        rl, rr = int(np.floor(left * g)), int(np.ceil((left + nw) * g))
        inimg = (yy >= rt) & (yy < rb) & (xx >= rl) & (xx < rr)
        grids, scores, bx = [], [], []
        for j, i in enumerate(keep):
            b = boxes[i] * g
            m = (logits[j].reshape(Ph, Pw) > 0) & inimg & (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
            if m.sum() < self.area_min:
                continue
            grids.append(m)
            scores.append(conf[idx[i]])
            bx.append(boxes[i])
        sel = []                                    # greedy mask-IoU de-duplication (scores already descending)
        for j, m in enumerate(grids):
            a = m.sum()
            if all((m & grids[k]).sum() / max(a + grids[k].sum() - (m & grids[k]).sum(), 1) <= self.mask_iou for k in sel):
                sel.append(j)
        grid = np.stack([grids[j] for j in sel]) if sel else np.zeros((0, Ph, Pw), bool)
        out = dict(score=np.array([scores[j] for j in sel], np.float32), box=np.array([bx[j] for j in sel], np.float32).reshape(-1, 4),
                   grid=grid)
        if full:
            sx = 1 / (r * g)
            fi = np.clip(((np.arange(W) + 0.5) * r * g + left * g).astype(int), 0, Pw - 1)
            fj = np.clip(((np.arange(H) + 0.5) * r * g + top * g).astype(int), 0, Ph - 1)
            out['masks'] = grid[:, fj][:, :, fi] if len(sel) else np.zeros((0, H, W), bool)
            del sx
        self.ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def close(self):
        pass
