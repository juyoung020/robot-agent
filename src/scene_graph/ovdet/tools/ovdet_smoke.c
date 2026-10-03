/* ovdet smoke test: ovdet_smoke <engine> <image.rgb> <h> <w> [name ...]
 * image.rgb = raw h x w x 3 RGB u8; names = the prompt (default: the whole vocabulary). Prints the detections and the
 * mean timing of 40 runs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ovdet.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s engine image.rgb h w [name ...]\n", argv[0]);
    return 2;
  }
  char names[4096];
  snprintf(names, sizeof names, "%s.names.txt", argv[1]);
  OvdConfig cfg;
  ovd_default_config(&cfg);
  cfg.seg_engine = argv[1];
  cfg.names = names;
  const int h = atoi(argv[3]), w = atoi(argv[4]);
  unsigned char* rgb = (unsigned char*)malloc((size_t)h * w * 3);
  FILE* f = fopen(argv[2], "rb");
  if (!f || fread(rgb, 1, (size_t)h * w * 3, f) != (size_t)h * w * 3) {
    fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  fclose(f);
  char err[512];
  OvdHandle* d = ovd_create(&cfg, err, sizeof err);
  if (!d) {
    fprintf(stderr, "create: %s\n", err);
    return 1;
  }
  printf("vocabulary %d, device %.1f MB\n", ovd_vocab_size(d), ovd_device_bytes(d) / 1048576.0);
  if (argc > 5) {
    const int found = ovd_set_prompt(d, (const char* const*)(argv + 5), argc - 5, err, sizeof err);
    printf("prompt %d/%d %s\n", found, argc - 5, err);
  }
  OvdImage im = {0.0, 0, rgb, h, w, (long long)w * 3, 3, 0, 0};
  OvdTiming t;
  const sm_detections* r = NULL;
  float tot = 0, net = 0;
  for (int i = 0; i < 50; ++i) {
    r = ovd_detect(d, &im, &t);
    if (!r) {
      fprintf(stderr, "detect: %s\n", ovd_last_error(d));
      return 1;
    }
    if (i >= 10) {
      tot += t.total_ms;
      net += t.net_ms;
    }
  }
  const int* ar = ovd_last_areas(d);
  printf("%d objects (candidates %d), grid %dx%d s=%.3f o=(%.1f,%.1f), mean total %.2f ms net %.2f ms\n", r->n,
         t.candidates, r->mask_w, r->mask_h, r->mask_sx, r->mask_ox, r->mask_oy, tot / 40, net / 40);
  for (int i = 0; i < r->n; ++i)
    printf("  %2d cls %3d  score %.3f  box %.0f %.0f %.0f %.0f  area %d\n", i, r->cls[i], r->score[i], r->box[4 * i],
           r->box[4 * i + 1], r->box[4 * i + 2], r->box[4 * i + 3], ar[i]);
  ovd_destroy(d);
  free(rgb);
  return 0;
}
