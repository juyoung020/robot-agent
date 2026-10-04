/* sgclip — 물체 영상 임베딩(SigLIP 2 B/32-256, TensorRT) + 라벨 표 찾기(CPU) C ABI. docs/clip_candidates.md 3.5·6·8.
 *
 *   영상 쪽(GPU): 원본 RGB(장치 또는 호스트) + 검출 상자·마스크 비트 → CUDA 커널 하나(정사각 상자 + 10 % 둘레, 양선형 256 × 256,
 *                 정규화, 8 × 8 마스크 비율 격자) → TensorRT(FP16, LayerNorm·GELU FP32) → 물체마다 768-d L2 정규화 벡터 1개.
 *                 자기 CUDA 스트림(같은 프로세스·같은 CUDA 문맥 — ovdet 과 나눠 씀), 배치 ≤ 8, 칸 2개 고리(비동기).
 *                 sgc_submit 은 자르기 커널이 원본을 다 읽을 때까지만 기다리고(수십 µs) 돌아온다. 결과는 sgc_poll.
 *                 처음 몇 번(배치 크기별 CUDA graph 잡기) 뒤에는 프레임마다 메모리를 잡지 않는다.
 *   라벨 쪽(CPU): 라벨 표 폴더(training/embed/README.md "라벨 표 형식": manifest.json · table.jsonl · text_siglip2_b32.f16) →
 *                 IVF(128-d PCA 공간, 256 묶음) → (선택 128-bit 부호 해밍 거름) → 128-d FP16 점수 상위 32 → 768-d FP16 다시 매김
 *                 (AVX2 / NEON / 일반).
 *                 색인은 표 sha 별 파일로 캐시(index_dir/labels_<sha>.idx). 확신이 낮으면 WordNet 상위어로 올림.
 *
 * TensorRT 8.2(JetPack 4.6, Nano) 와 10(PC) 둘 다 빌드된다(src/encoder.cpp 의 NV_TENSORRT_MAJOR 분기).
 * 스레드: sgc_encoder 는 한 스레드(제출·poll 같은 스레드). sgc_labels 의 찾기 함수는 여러 스레드에서 동시에 불러도 된다.
 */
#ifndef SGCLIP_H
#define SGCLIP_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SGC_DIM 768

/* ---------------- 영상 인코더 ---------------- */
typedef struct sgc_encoder sgc_encoder;

typedef struct {
  const char* engine;       /* TensorRT plan(tools/build_engine.py), 입력 images N×3×S×S(FP16 또는 FP32) + wpatch N×G², 출력 emb N×768 */
  int32_t device;           /* CUDA 장치(0) */
  int32_t max_batch;        /* 한 번에 넣는 물체 수 상한(8, 엔진 프로필 안) */
  int32_t use_graph;        /* 1: 배치 크기별 CUDA graph(TRT 10 만, 기본 1) */
  float margin;             /* 정사각 상자 둘레 비율(0.1) */
} sgc_config;
void sgc_default_config(sgc_config* c);

/* 한 프레임: 원본 RGB 와 검출기 마스크 격자(sm_detections 와 같은 뜻) */
typedef struct {
  const uint8_t* rgb;       /* 첫 화소 */
  int32_t on_device;        /* 1: 장치 메모리, 0: 호스트(장치로 한 번 올림) */
  int64_t row_stride;       /* 바이트 */
  int32_t pix_stride;       /* 3 또는 4 */
  int32_t w, h;
  int32_t mask_w, mask_h;   /* 마스크 격자(0 이면 마스크 없음 → wpatch = 1) */
  float mask_sx, mask_sy, mask_ox, mask_oy;   /* 입력 화소 = 칸 × s + o */
  const uint32_t* mask_bits;/* 호스트, 검출마다 ceil(mask_w·mask_h / 32) 단어(행 우선, LSB 먼저) */
} sgc_frame;

typedef struct {
  uint32_t id;              /* 호출자 물체 id(결과에 그대로) */
  int32_t det;              /* mask_bits 의 검출 번호(< 0 = 마스크 없음) */
  float box[4];             /* x0, y0, x1, y1 입력 화소 */
  float quality;            /* 호출자 값(결과에 그대로, 예: 마스크 넓이 × 점수) */
} sgc_item;

typedef struct {
  uint32_t id;
  float quality;
  double stamp;
  const float* emb;         /* SGC_DIM 개, L2 정규화. 다음 sgc_poll 까지 유효 */
} sgc_result;

typedef struct {
  float crop_ms, net_ms;    /* 마지막으로 끝난 묶음의 GPU 시간(자르기 커널, 엔진) */
  float submit_us;          /* 마지막 sgc_submit 의 CPU 시간(자르기 끝 기다림 포함) */
  int32_t last_batch;
  int64_t n_submitted, n_done, n_dropped;   /* 물체 수 누계(dropped = 두 칸 다 바빠서 못 넣음) */
  int32_t input_size, grid; /* 엔진 S, G */
  int64_t device_bytes;     /* 엔진 + 버퍼 */
} sgc_timing;

sgc_encoder* sgc_create(const sgc_config* c, char* err, size_t err_len);
void sgc_destroy(sgc_encoder*);
/* items 중 앞 min(n, max_batch) 개를 넣는다. 넣은 수(0 = 두 칸 다 바쁨 → 다음에), < 0 = 오류. stamp 는 결과에 그대로. */
int32_t sgc_submit(sgc_encoder*, double stamp, const sgc_frame* f, const sgc_item* items, int32_t n);
/* 끝난 묶음의 결과를 out 에 min(끝난 수, cap) 개(못 담은 것은 다음 poll). wait = 1 이면 진행 중인 것이 끝날 때까지 기다림. */
int32_t sgc_poll(sgc_encoder*, sgc_result* out, int32_t cap, int32_t wait);
int32_t sgc_pending(const sgc_encoder*);   /* 아직 결과를 안 꺼낸 물체 수 */
void sgc_get_timing(const sgc_encoder*, sgc_timing* out);
/* 시험용: 마지막 제출 칸의 엔진 입력(images FP32 로 바꿔서 n×3×S×S, wpatch n×G²)을 호스트로. 반환 n */
int32_t sgc_debug_inputs(sgc_encoder*, float* images, float* wpatch, int32_t cap);

/* ---------------- 라벨 표 ---------------- */
typedef struct sgc_labels sgc_labels;

typedef struct {
  int32_t nprobe;           /* IVF 묶음 몇 개를 볼지(8) */
  int32_t rerank;           /* 해밍 상위 몇 개를 768-d 로 다시 매길지(32) */
  int32_t exact;            /* 1: 표 전체 768-d(기준, 느림) */
  int32_t main_only;        /* 1: tier "main"(집 물건 주 표)만 */
  int32_t prefilter;        /* > 0: 128-bit 해밍으로 이 수까지 먼저 거름(0 = 묶음 후보 전부 128-d 점수) */
} sgc_lookup_params;
void sgc_default_lookup(sgc_lookup_params* p);

typedef struct {
  int32_t row;              /* table.jsonl 줄 번호 */
  float score;              /* 코사인 */
  const char* en;           /* 표 수명 동안 유효 */
  const char* ko;           /* 첫 한국어 이름("" = 없음) */
  int32_t structural;
} sgc_hit;

typedef struct {
  sgc_hit top[5];
  int32_t n;
  const char* level_en;     /* 확신에 맞춘 이름(상위어로 올렸을 수 있음) */
  const char* level_ko;
  float level_score;        /* 1위 코사인 */
  float margin;             /* 1위 − 다른 synset 의 2위 */
  float prob;               /* SigLIP 시그모이드 확률 sigmoid(cos·t + b) */
  int32_t rolled;           /* 1: 상위어로 올림 */
  int32_t structural;       /* 1위가 구조물(벽·바닥·문 …) */
} sgc_names;

/* dir = 라벨 표 폴더. index_dir 에 labels_<sha>.idx 가 있으면 읽고, 없으면 만들어 씀(NULL = 캐시 없이 메모리에만). */
sgc_labels* sgc_labels_open(const char* dir, const char* index_dir, char* err, size_t err_len);
/* img_sample: N × 768 FP16 영상 임베딩 표본(질의 분포, 예: LVIS crop 1만 개). 있으면 128-d 투영을 영상 분포 PCA 로 맞추고
 * 질의에서 표본 평균을 뺀다(1단계 순위가 훨씬 좋아짐). sgc_labels_open 은 환경 변수 SGC_IMG_SAMPLE 을 쓴다. 색인 파일 이름에 표본 sha */
sgc_labels* sgc_labels_open_ex(const char* dir, const char* index_dir, const char* img_sample, char* err, size_t err_len);
void sgc_labels_close(sgc_labels*);
const char* sgc_labels_sha(const sgc_labels*);       /* manifest sha(16 자) */
const char* sgc_labels_name(const sgc_labels*);      /* "objects-v1" */
int32_t sgc_labels_count(const sgc_labels*);
/* q: SGC_DIM, L2 정규화된 영상(또는 글) 벡터 → 코사인 상위 k(점수 순). 개수 */
int32_t sgc_labels_lookup(const sgc_labels*, const float* q, int32_t k, sgc_hit* out, const sgc_lookup_params* p /* NULL = 기본 */);
/* 이름 붙이기: 상위 5 + 확신 맞춘 이름. 0 = 성공 */
int32_t sgc_labels_names(const sgc_labels*, const float* q, sgc_names* out, const sgc_lookup_params* p);
/* 글 → 표 줄(영어 이름·한국어 이름 정확히 같음, 소문자·앞뒤 공백 무시). 없으면 -1 */
int32_t sgc_labels_find(const sgc_labels*, const char* text);
/* 이름 검색용: 영어 이름·한국어 이름 전부·영어 동의어(en_syn), main(집 물건) 줄 먼저. 없으면 -1 */
int32_t sgc_labels_find_name(const sgc_labels*, const char* text);
/* 그 글을 이름(영어·한국어 전부·동의어)으로 가진 줄 전부(뜻이 여럿인 말: "의자" → chair·armchair). main 줄이 있으면 main 만, 없으면 tail.
 * rows 에 최대 cap 개, 반환 = 개수(0 = 없음) */
int32_t sgc_labels_find_names(const sgc_labels*, const char* text, int32_t* rows, int32_t cap);
/* 줄의 글 임베딩(SGC_DIM, FP32 로 풀어 out 에). 0 = 성공 */
int32_t sgc_labels_text_emb(const sgc_labels*, int32_t row, float* out);

/* ---------------- 글 인코더(SigLIP 2 B/32 글 탑, TensorRT) ----------------
 * 글 → 768-d L2 벡터(라벨 표·물체 영상 벡터와 같은 공간). 물체 찾기(sgsearch.h)의 자유 글 질의용. tools/export_siglip2_text.py 가
 * 만든 파일 셋: 토크나이저(siglip2_b32_tok.bin, Gemma BPE), 토큰 임베딩 표(siglip2_b32_tokemb.f16, 256000 × 768, CPU mmap —
 * 엔진 밖에 두어 GPU 는 변환기 12 층만, 약 170 MB), 엔진(siglip2_b32_text_fp16.plan, 입력 tok_emb N×64×768 FP32, 출력 emb N×768).
 * 토큰화 = open_clip HFTokenizer(clean="canonicalize"): '_'→공백, ASCII 문장 부호 지움, 소문자, 공백 하나로, BPE, + eos, 64 로 자르고 0 채움.
 * 동기 호출(한 번에 ≤ max_batch, 넘으면 나눠 돎). 한 스레드에서만. */
typedef struct sgc_text sgc_text;

typedef struct {
  const char* engine;       /* NULL = 토크나이저만(sgc_text_tokenize) */
  const char* tokenizer;    /* siglip2_b32_tok.bin */
  const char* tok_emb;      /* siglip2_b32_tokemb.f16 */
  int32_t device;
  int32_t max_batch;        /* 8(엔진 프로필 안) */
} sgc_text_config;
/* dir(NULL = 환경 변수 SGC_TEXT_DIR, 없으면 ~/ovdet_models/x86_sm120/siglip2_b32) 아래 기본 파일 이름. 문자열은 정적 버퍼 */
void sgc_text_default_config(sgc_text_config* c, const char* dir);
sgc_text* sgc_text_create(const sgc_text_config* c, char* err, size_t err_len);
void sgc_text_destroy(sgc_text*);
#define SGC_TEXT_CTX 64
/* ids: SGC_TEXT_CTX 칸(0 채움). 반환 = eos 포함 토큰 수, < 0 = 오류 */
int32_t sgc_text_tokenize(const sgc_text*, const char* text, int32_t* ids);
/* texts n 개 → out n × SGC_DIM(L2). 0 = 성공 */
int32_t sgc_text_encode(sgc_text*, const char* const* texts, int32_t n, float* out);
int32_t sgc_text_encode_ids(sgc_text*, const int32_t* ids /* n × SGC_TEXT_CTX */, int32_t n, float* out);
float sgc_text_last_ms(const sgc_text*);   /* 마지막 encode 의 벽시계 ms(토큰화·모으기·엔진·복사) */
int64_t sgc_text_device_bytes(const sgc_text*);

/* ---------------- 작은 도구 ---------------- */
void sgc_f32_to_f16(const float* in, uint16_t* out, int32_t n);
void sgc_f16_to_f32(const uint16_t* in, float* out, int32_t n);
float sgc_dot_f16(const uint16_t* a, const float* b, int32_t n);   /* SIMD(AVX2 F16C / NEON / 일반) */
const char* sgc_simd(void);                                          /* "avx2" / "neon" / "scalar" */

#ifdef __cplusplus
}
#endif
#endif /* SGCLIP_H */
