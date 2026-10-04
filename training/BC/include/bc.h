// BC 학생 학습기(C++). 장치 메모리 배치, 롤아웃·갱신 몸통, 그래프 잡기·실행, 기록 링. 계획서 4.6·5.3·5.4.
// 모든 CUDA 호출은 이 스레드의 기본 스트림(--default-stream per-thread → cudaStreamPerThread). G1·G2 의 step 도 같은 스트림이라 같은 그래프로 잡힌다.
//
// 학생 신경망(StudentNet): student-lite 와 같은 칸 MLP(S1·S2) + 집합 + A1–A3 몸통에
//   영상(vision): 카메라 2 장 → 렌더(bc_render) → 얼린 SigLIP 2 패치 토큰 128 개 × 768(vit) → P1(784 → 16, ELU, 토큰마다 같은 가중치) → 펼친 2,048
//   글(text): 지시 문장의 얼린 128-d 벡터(training/data/vla_v1 instr128 — 이름·생김새와 같은 공간, 문장 번호는 에피소드마다 해시로 같은 과제의 바꿔 말하기 중 하나)
//   A1 입력 = [X0 304 | 영상 2,048 | 글 128]
//   머리: MSE(A4 → 8) 또는 flow matching 행동 전문가(E1: [몸통 128 | 1 | x_τ 128 | 시간 sin·cos 32 | 0] 304 → 256 → 256 → 128 = 청크 16 × 행동 8)
// arch 1(토큰마다 학생, tf.h): 영상 128 + 글 1 + 몸 3 + 물체 16 + 벽·방 2 토큰 → 트랜스포머 → flow 행동 전문가(행동 8 × 청크 H)
#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "bc_capi.h"
#include "bc_data.h"
#include "env_api.h"
#include "map_api.h"
#include "net.h"
#include "net_ops.h"
#include "tf.h"
#include "bscene_host.h"
#include "vec_tab.h"
#include "vit.h"

namespace bcr { struct Renderer; }

namespace bc {

// 장치 값(그래프가 읽음): 누가 움직이나·기록하나
struct Mode { int actor, record, pad0, pad1; };
// 장치 값: 자료 버퍼 쓰기 자리·표본 수·롤아웃 번호
struct Data { long long cursor, count, rollouts, pad; };

// 교사 신경망(정책 사슬 A 와 칸 MLP)의 작업 버퍼(행 M)
struct NetBufs {
  int M = 0;
  uint16_t *x0 = nullptr, *sin = nullptr, *s1o = nullptr, *s2o = nullptr;
  uint32_t* mask = nullptr;
  uint8_t* amax = nullptr;
  uint16_t* ho[net::N_LAYER] = {};
  float* mean = nullptr;
};

// ---- 학생 신경망 모양 ----
enum SLayer { SL_S1, SL_S2, SL_P1, SL_A1, SL_A2, SL_A3, SL_A4, SL_E1, SL_E2, SL_E3, SL_N };
constexpr int IMG_TOK = 2 * vit::NTOK;   // 표본 하나의 영상 토큰 128
constexpr int IMG_D = 16;                // P1 출력(토큰마다)
constexpr int IMG_W = IMG_TOK * IMG_D;   // 2,048
constexpr int TXT_W = vlav::DIM;         // 128 (얼린 128-d 지시 벡터)
constexpr int MAX_H = 16;                // 청크 최대 길이
constexpr int FLOW_W = MAX_H * N_LAB;    // 128
constexpr int TEMB = 32;                 // 시간 sin 16 + cos 16
constexpr int E_X = 129, E_T = E_X + FLOW_W, E_IN = 304;   // E1 입력 칸: 몸통 0..127, 1 = 128, x_τ 129..256, 시간 257..288, 0 289..303
static_assert(E_T + TEMB <= E_IN && E_IN % 16 == 0, "E1 input");
// 지시 표 txt: [0, 64) = vla_v1 지시(과제 바꿔 말하기, 상자 방) 또는 bc_load_text_table 표, [64, 64 + 1024) = 집기·놓기 지시(training/data/pnp_v1 —
// BEHAVIOR 판의 행 = 지도 토큰 instr1 − 1 = 환경 I_B_INSTR, 판 시작 때 환경이 고른 문장)
constexpr int TXT_PNP0 = 64;
constexpr int MAX_TXT = TXT_PNP0 + 1024;
// 지시 문장 고르기(장치 값): 과제의 학습용 바꿔 말하기 / 처음 보는 바꿔 말하기(heldout, VLA_INPUT 7절 평가) 표 행
struct TxtSel { int n_train, n_held, pad0, pad1; int train[16], held[16]; };
struct StudentNet {
  net::LayerDesc L[SL_N];
  bool on[SL_N];
  long long off[SL_N], total;
  int k1 = net::X0_W;          // A1 입력 폭
  int x_img = 0, x_txt = 0;    // A1 입력 안 영상·글 시작 칸
  int vision = 0, text = 0, head = 0, H = 1, arch = 0;
  bool ext() const { return vision || text; }
};
StudentNet student_net(const BcConfig& c);

// ---- 장치·호스트 공용 작은 함수(검증이 같은 식을 CPU 에서 부른다) ----
// 지시 문장 번호 = 에피소드 번호 해시(롤아웃·모으기가 같은 식)
NDEV int text_id(uint32_t epi, int n_txt) { return n_txt > 1 ? (int)(net::mix64((uint64_t)epi * 0x9E3779B97F4A7C15ull + 0x7478u) % (uint64_t)n_txt) : 0; }
// 표 행: 학습은 학습용 바꿔 말하기 중 하나, eval_unseen 이면 처음 보는 바꿔 말하기 중 하나
NDEV int text_row(uint32_t epi, const TxtSel& s, int eval_unseen) {
  if (eval_unseen && s.n_held > 0) return s.held[text_id(epi, s.n_held)];
  return s.n_train > 0 ? s.train[text_id(epi, s.n_train)] : 0;
}
// 표본(지도 토큰)의 지시 행: 집기·놓기 판이면 환경이 고른 pnp 행(heldout 은 환경 BCurr::eval_instr 가 고름), 아니면 과제 바꿔 말하기 해시
NDEV int text_row_tok(const gmap::MapTok& tok, uint32_t epi, const TxtSel& s, int eval_unseen) {
  return tok.instr1 ? TXT_PNP0 + (int)tok.instr1 - 1 : text_row(epi, s, eval_unseen);
}

// 시간 τ 의 sin/cos 16 주기(0.004 … 4.0, 로그 간격 — π0 방식, 가정)
NDEV void temb_write(float tau, uint16_t* dst) {
  for (int i = 0; i < TEMB / 2; ++i) {
    const float period = 0.004f * powf(1000.f, (float)i / (float)(TEMB / 2 - 1));
    const float a = tau * (6.283185307179586f / period);
    dst[i] = net::f2bf(sinf(a));
    dst[TEMB / 2 + i] = net::f2bf(cosf(a));
  }
}
NDEV float u01(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.f / 16777216.f); }   // (0, 1)
NDEV float gauss(uint64_t h) {   // Box–Muller(한 쌍 중 하나)
  const float a = u01(h), b = u01(net::mix64(h ^ 0x5bd1e995ull));
  return sqrtf(-2.f * logf(a)) * cosf(6.283185307179586f * b);
}


// 학생 작업 버퍼(행 M = max(N, MB))
struct SBufs {
  uint16_t *x0 = nullptr, *sin = nullptr, *s1o = nullptr, *s2o = nullptr, *x0e = nullptr;
  uint32_t* mask = nullptr;
  uint8_t* amax = nullptr;
  int* tid = nullptr;            // [M] 지시 문장 번호
  uint16_t* tok = nullptr;       // [M × 128][784] 얼린 인코더 출력
  uint16_t* imgf = nullptr;      // [M][2048] P1 출력
  uint16_t* h[SL_N] = {};        // 층 출력(bf16)
  float* out = nullptr;          // [M][32] A4 평균(앞 8) 또는 E3 속도
  float* act = nullptr;          // [M][8] 행동(학생 μ 자리: MSE = A4 평균, flow = 청크 첫 행동)
  float* xf = nullptr;           // [M][32] flow 상태(추론 오일러) / 학습 목표 u = ε − a
  float* fm = nullptr;           // [M][16] 청크 칸 가림(학습)
  uint16_t* dz[SL_N] = {};
  float* dpool = nullptr;
};

struct Bc {
  BcConfig cfg;
  net::ParamLayout lay;   // 교사(RL 배치)
  StudentNet sn;          // 학생
  int N, T, MB;

  // v2 입력: 얼린 표, 흔들기(학생만, 교사 라벨은 늘 끔), 행동 가림, 지시 문장 고르기 — 모두 장치 값
  obsv::VecTables vt;
  obsv::ObsAug* aug_d = nullptr;
  uint32_t* amask_d = nullptr;
  TxtSel* tsel_d = nullptr;
  // arch 1: 토큰마다 학생
  tfm::Tf tf;
  uint16_t* tg[tfm::N_GRP] = {};   // 묶음 입력 줄(영상은 sb.tok)
  uint32_t *tobj = nullptr, *toff = nullptr;
  float *tact = nullptr, *txb = nullptr;   // [N][H][8] 추론 청크, 오일러 작업

  std::unique_ptr<bsc::SceneBuild> scenes;   // E2 BEHAVIOR(stage 3): 장면 묶음(beh 1 일 때만)
  bsc::BCurr* bcurr_d = nullptr;             // 커리큘럼 장치 값(환경이 판 리셋 때 읽음)
  std::unique_ptr<env::DeviceEnv> env;
  std::unique_ptr<gmap::DeviceMap> map;
  std::unique_ptr<gmap::TokenRecorder> tok;   // 줄 2 개 고리: 스텝 t 가 읽는 줄 t%2, 지도가 쓰는 줄 (t+1)%2 (T 짝수 → 다음 롤아웃 0 줄 = 지난 끝 줄)
  float* obs_col = nullptr;   // [2][80][N] 같은 고리(G1 은 [k*N + i] 로 씀)
  float* act_env = nullptr;   // [8][N]
  float* rew = nullptr;       // [N]
  int* done = nullptr;        // [N]
  int* cur_len = nullptr;     // [N] 진행 중 에피소드 스텝
  uint32_t* ep_uid = nullptr; // [N] 진행 중 에피소드 번호(0 아님, 판·에피소드마다 다름)
  int* it_stat = nullptr;     // [3][3] 이 롤아웃 처음 지도별 (끝난 수, 성공, 충돌)
  int* it_out = nullptr;      // [8] 이 롤아웃 (성공, 충돌, 시간초과) 합, [3] 접지 판, [4] 접지 맞음, [5] B2·B3 끝난 판, [6] 그 성공
  int* gnd = nullptr;         // [N] 지금 스텝에 로봇에서 가장 가까운 과제 물체가 목표인가(1/0), 판정 안 함 −1 (BEHAVIOR B2·B3, 과제 물체 ≥ 2)
  unsigned long long* tab = nullptr;   // [3][2][10][6] 누적(ppo 와 같은 배치)
  float* dis = nullptr;       // [T][N] (학생 μ − 교사 라벨)² 합 — 롤아웃 끝에 고정 순서로 더함
  gmap::MapCurr* curr_d = nullptr;
  Mode* mode_d = nullptr;
  Data* data_d = nullptr;
  int host_actor = 0;         // 호스트가 마지막으로 정한 actor(어느 롤아웃 그래프를 띄울지)
  int cur_t = 0;              // 롤아웃 스텝(잡을 때 고정되는 호스트 값 — arch 1 추론 잡음 열쇠)
  // 비동기 장치 값 바꾸기용 고정 호스트 링(칸마다 이벤트)
  uint8_t* stage_h = nullptr;
  cudaEvent_t stage_ev[16] = {};
  int stage_slot = 0;

  // 자료 버퍼(장치, 표본 cap 개)
  long long cap = 0;
  uint16_t* d_obs = nullptr;        // [cap][80] bf16
  gmap::MapTok* d_tok = nullptr;    // [cap]
  float* d_lab = nullptr;           // [cap][8]
  uint32_t* d_meta = nullptr;       // [cap]
  uint32_t* d_epi = nullptr;        // [cap] 에피소드 번호(청크 라벨·지시 번호)
  RenderState* d_rs = nullptr;      // [cap] (store_render)

  // 신경망: 교사(앞만, bf16 사본) · 학생(FP32 원본·기울기·Adam·bf16 사본)
  uint16_t* PbT = nullptr;
  float *P = nullptr, *G = nullptr, *Am = nullptr, *Av = nullptr;
  uint16_t* Pb = nullptr;
  NetBufs nt;     // 교사(행 N)
  SBufs sb;       // 학생(행 max(N, MB))
  int SM = 0;     // 학생 버퍼 행 수
  float* lab_mb = nullptr;          // [MB][8]
  float* chunk_mb = nullptr;        // [MB][16][8] 청크 라벨
  float* ws[SL_N] = {};
  float *gn_part = nullptr, *loss_part = nullptr;
  net::TrainState* ts = nullptr;    // iter = 갱신 스텝 수(미니배치 난수 열쇠), s_pg = 이 그래프 손실 합, n_mb = 스텝 수, lr
  net::AdamHyper ah;

  // 영상: 렌더(묶음 render_batch 판) + 얼린 인코더 + 렌더 상태(롤아웃 지금 / 미니배치)
  bcr::Renderer* rnd = nullptr;
  bcr::Renderer* rnd_team = nullptr;   // render_team_mix > 0: 팀 기본 설정 렌더
  vit::Encoder enc;
  RenderState* rs_roll = nullptr;   // [N]
  RenderState* rs_mb = nullptr;     // [MB]
  uint16_t* txt = nullptr;          // [MAX_TXT][768] bf16 지시 문장 글 벡터
  int n_txt = 1;

  // 기록 링(매핑된 고정 호스트 메모리에 커널이 직접 씀)
  BcLog* ring_h = nullptr;
  BcLog* ring_d = nullptr;
  std::vector<cudaEvent_t> ev_a, ev_b;
  long long issued = 0, polled = 0;

  cudaGraphExec_t g_roll = nullptr, g_roll_s = nullptr, g_upd = nullptr;   // 교사가 움직이는 롤아웃 / 학생이 움직이는 롤아웃 / 갱신
  size_t dev_bytes = 0;
  int bug = 0;                 // 음성 대조(검증): 1 MSE 기울기 배율 2 빠뜨림, 2 A3 → A2 dX 의 ELU' 빠뜨림, 3 flow 손실 기울기 배율 2 빠뜨림,
                               //                  4 P1 dX(영상 칸)의 ELU' 빠뜨림, 5 E1 → 몸통 dX 의 ELU' 빠뜨림
  bool keep_slot_bufs = false; // 검증: 칸 MLP 뒤 묶음이 dZ S2·S1 도 전역에
  bool vis_skip = false;       // 측정용(bench 겹침 시험): vis_encode 를 건너뜀

  explicit Bc(const BcConfig& c);
  ~Bc();
  void make_env(uint64_t env_seed);   // 환경·지도를 새로 만듦(처음 한 번, 그리고 검증 bc_verify reseed 의 예전 판 흉내)
  int reseed(uint64_t env_seed);      // 장치 씨앗 바꾸기 요청(다시 만들기·동기·그래프 다시 잡기 없음) — 다음 롤아웃 앞에서 적용
  void apply_body();                  // 롤아웃 그래프 맨 앞: 요청이 있으면 환경·지도·고리 버퍼를 새로 만든 것과 같게(없으면 바로 끝남)
  int* bc_pend = nullptr;             // 장치: 고리 버퍼 지우기 요청
  void capture();
  void set_dev(void* dst, const void* src, size_t n);   // 고정 호스트 링 + 비동기 복사

  // 몸통(즉시 실행 또는 그래프 잡기 중)
  void rollout_body_t() { rollout_body(false); }
  void rollout_body_s() { rollout_body(true); }
  void rollout_body(bool student);
  void rollout_step(int t, bool student);
  void update_body();
  void update_step();
  // 부분(검증용)
  void forward_teacher(NetBufs& b, int M);
  void vis_encode(const RenderState* rs, int M);   // 렌더 → 패치 → 인코더 → sb.tok
  void student_trunk(int M, const RenderState* rs); // 칸 MLP·집합 → (영상·글) → A1–A3 (arch 1: 묶음 줄 → 영상 → tf prefix)
  void tf_rows(const float* obs, int stride, const gmap::MapTok* tok, const uint32_t* epi, int M, uint64_t k0sel, int role);   // arch 1 묶음 줄
  void student_act(int M);                           // 머리 추론 → sb.act (MSE 평균 / flow 오일러)
  void gather();
  void flow_inputs(int M);                           // flow 학습 입력 x_τ·시간·목표 u·가림
  void head_forward(int M);                          // 학습: A4 또는 E1–E3
  void loss(int M);
  void backward(int M);
  void optimizer();

  int launch(int kind);
  int poll(BcLog* out);

  template <class T_>
  T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace bc
