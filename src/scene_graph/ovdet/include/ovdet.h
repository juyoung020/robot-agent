/* ovdet — open-vocabulary object detector (masks + names) as a C library.
 *
 * One image and a prompt (the task's BDDL object names) in, a list of objects out: per object the prompt index, a
 * score, a box and a mask. The segmentation network (YOLOE, text-prompt head) runs in TensorRT (FP16); everything
 * around it — letterbox, class selection, NMS, mask assembly — is hand-written CUDA. No Python, no ROS; the evaluator
 * process calls it directly.
 *
 * Output = sm_detections of docs/scenemap_설계.md section 4.2 (the scenemap stack's detector contract): masks stay at
 * the detector's own grid (the network's prototype grid, e.g. 256 x 256 for a 1024 input) as bits, with the scale and
 * offset back to the input image. Nothing is copied to the caller; the arrays live in the handle and stay valid until
 * the next ovd_detect / ovd_set_prompt / ovd_destroy on that handle.
 *
 * Engine: images 1 x 3 x H x W, output0 1 x (4 + nc + 32) x A, output1 1 x 32 x h x w (tools/export_yoloe.py,
 * tools/build_engines.py). It is built once with a whole vocabulary (every task's BDDL objects + scene structures,
 * <engine>.names.txt); the prompt switches classes on and off. YOLOE's class scores are independent per class, so this
 * equals an engine built with only the prompt's names.
 *
 * Licence: YOLOE (Ultralytics / THU-MIG) is AGPL-3.0, and so are the weights in the engine; see README.md.
 *
 * Thread safety: one handle per thread. Calls are synchronous on the handle's own CUDA stream.
 */
#ifndef OVDET_H
#define OVDET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- detector output: scenemap design 4.2, field for field (same guard, so either header may define it) ---- */
#ifndef SM_DETECTIONS_DEFINED
#define SM_DETECTIONS_DEFINED
typedef struct {
  double stamp;               /* the input image's stamp, unchanged */
  int cam;                    /* 0 head, 1 left wrist, 2 right wrist (passed through) */
  int img_w, img_h;           /* input image size */
  int n;                      /* number of detections (score order) */
  const int32_t* cls;         /* n: index into the prompt table (order of the names given to ovd_set_prompt) */
  const float* score;         /* n: confidence */
  const float* box;           /* n x 4: x0, y0, x1, y1 in input pixels */
  int mask_w, mask_h;         /* mask grid */
  float mask_sx, mask_sy, mask_ox, mask_oy; /* input pixel = mask cell x s + o: cell (i, j) covers
                                               x in [i sx + ox, (i+1) sx + ox), y in [j sy + oy, (j+1) sy + oy) */
  const uint32_t* mask_bits;  /* n x ceil(mask_w x mask_h / 32) words: row-major bits, cell k = j mask_w + i is bit
                                 (k & 31) (LSB first) of word k >> 5 */
} sm_detections;
#endif

typedef struct OvdHandle OvdHandle;

typedef struct {
  const char* seg_engine;   /* TensorRT plan of the segmentation head (required) */
  const char* names;        /* class names of the engine, one per line (<engine>.names.txt) */
  int32_t device;
  float conf_th;            /* candidate score (0.25, Ultralytics' default) */
  float nms_iou;            /* box NMS IoU (0.7) */
  float mask_iou;           /* mask duplicate IoU (0.7), <= 0 = off */
  int32_t area_min;         /* minimum mask area in mask cells (24) */
  int32_t small_area;       /* masks below this area (cells) ... (256) */
  float small_conf;         /* ... need at least this score (0 = off) */
  int32_t max_det;          /* at most this many objects per image (<= 128) */
  int32_t class_agnostic;   /* 1: one NMS over all classes (one object, one name), 0: per class */
} OvdConfig;

void ovd_default_config(OvdConfig* cfg);

typedef struct {
  double stamp;             /* copied to the output */
  int32_t cam;              /* copied to the output */
  const uint8_t* data;      /* first pixel, u8 */
  int32_t h, w;
  int64_t row_stride;       /* bytes between rows */
  int32_t pix_stride;       /* bytes between pixels: 3 (RGB/BGR) or 4 (RGBA, alpha ignored) */
  int32_t bgr;              /* 1: channels are B, G, R */
  int32_t on_device;        /* 1: CUDA device memory on the handle's device */
} OvdImage;

typedef struct {
  float upload_ms, net_ms, post_ms, out_ms, total_ms;
  int32_t candidates;       /* anchors above conf_th */
} OvdTiming;

/* NULL on failure; the reason is in err (if given). */
OvdHandle* ovd_create(const OvdConfig* cfg, char* err, size_t err_len);
void ovd_destroy(OvdHandle* h);

/* Vocabulary of the head (what a prompt may name). */
int32_t ovd_vocab_size(const OvdHandle* h);
const char* ovd_vocab_name(const OvdHandle* h, int32_t i);
/* GPU memory the handle holds (engines + buffers), bytes. */
int64_t ovd_device_bytes(const OvdHandle* h);

/* Prompt table: the names in order; detections carry the index into this list. Names are matched to the vocabulary
 * after normalisation (".n.NN" dropped, '_' -> ' ', lower case), so "radio_receiver.n.01" = "radio receiver". A name
 * outside the vocabulary keeps its index but is never detected. Returns the number of names found; the unknown ones
 * are listed in err (if given). */
int32_t ovd_set_prompt(OvdHandle* h, const char* const* names, int32_t n, char* err, size_t err_len);

/* Detect. The returned pointer (and all arrays in it) belong to the handle and stay valid until the next call on it.
 * NULL on error (ovd_last_error). */
const sm_detections* ovd_detect(OvdHandle* h, const OvdImage* img, OvdTiming* timing);
/* Mask area in cells of each detection of the last ovd_detect (n entries, same lifetime). */
const int32_t* ovd_last_areas(const OvdHandle* h);
const char* ovd_last_error(const OvdHandle* h);

#ifdef __cplusplus
}
#endif
#endif
