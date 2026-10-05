// 호스트 쪽 틀 (C++). 상태·관측은 장치에 있고 호스트는 실행만 한다.
#pragma once
#include <cstdint>
#include <vector>

#include "env.h"
#include "env_soa.h"
#include "teacher.h"
#include "teacher_sl.h"

namespace bsc { struct SceneBuild; }

namespace env {

// stage: 0/1/2 = 상자 방 A0/A1/A2(= 커리큘럼 B0, 예전 그대로), kStageBeh(3) = BEHAVIOR 집 장면(B1–B3, env_beh.h) — 장치 SceneSet 이 있어야 함
constexpr int kStageBeh = 3;

// 장치 단계 값(set_dynamic 판): 커널이 stage 를 읽고(맞지 않는 커널 무리는 바로 끝남), pend >= 0 이면 apply() 가 모든 판을 그 단계·씨앗으로 새로 시작한다.
// 다시 만들기(cudaMalloc·동기·init)와 그래프 다시 잡기가 없다 — 학습기 장치 커리큘럼·BC 씨앗 바꾸기가 쓴다(README 성능 메모)
struct EnvCtl {
  int stage;                // 지금 단계
  int pend;                 // 바꿀 단계(−1 없음)
  unsigned long long seed;  // 바꿀 때 쓸 씨앗(생성자 seed 와 같은 뜻)
};
// 커널 무리 비트(단계 → 무리): 0·1 상자 방(가구 없는 커널), 2 A2(가구·경로 워프 커널), 3 BEHAVIOR
constexpr uint32_t kFamBox = 1u, kFamA2 = 2u, kFamBeh = 4u;
inline uint32_t stage_family(int stage) { return stage >= kStageBeh ? kFamBeh : stage >= 2 ? kFamA2 : kFamBox; }

class DeviceEnv {
 public:
  // ss_dev: BEHAVIOR 판의 장치 장면 묶음(bscene_host.h SceneUpload::dev). cu0: 처음 커리큘럼 값(안쪽 장치 버퍼에 복사)
  DeviceEnv(int N, int stage, uint64_t seed, bool arm_free = false, const bsc::SceneSet* ss_dev = nullptr, const bsc::BCurr& cu0 = bsc::kBCurrDefault);
  ~DeviceEnv();
  DeviceEnv(const DeviceEnv&) = delete;
  DeviceEnv& operator=(const DeviceEnv&) = delete;
  int N() const { return N_; }
  // act[k*N+i] (장치), obs[k*N+i], rew[i], done[i] (장치). 비동기 — 기다리지 않는다. bug != 0 은 검증의 음성 대조용
  void step(const float* act, float* obs, float* rew, int* done, int bug = 0);
  void download(std::vector<float>& f, std::vector<int>& iv, std::vector<uint64_t>& rng) const;
  // 장치 상태 보기(읽기 전용으로 쓸 것) — 지도 단계(training/RL/map)가 스텝 뒤에 이어서 읽는다
  Soa soa() const { return Soa{f_, iv_, rng_, N_}; }
  int stage() const { return stage_; }
  // BEHAVIOR 판: 커리큘럼 장치 값(판 리셋 때 커널이 읽음 — 바꿔도 다시 잡기 없음). src 를 주면 그 장치 자리를 읽는다(학습기 버퍼)
  bsc::BCurr* bcurr_dev() { return bcurr_; }
  void set_bcurr_source(const bsc::BCurr* src) { bcurr_src_ = src; }
  // 지도 → 환경 되먹임(정책이 아는 지도의 거리장·목표 확정). 기본 없음 = 직선 거리, B2 는 보임만. gmap::DeviceMap::nav_fb() 를 넣는다
  void set_nav(const bsc::NavFb& fb) { nav_ = fb; }
  const bsc::SceneSet* scenes() const { return ss_; }
  // 대본 특권 교사(teacher.h, E6): 모든 판의 교사 행동 act[k*N+i](장치)를 씀 — 잡기 물리 판(B4–B6) 아니면 0. 지도 되먹임(set_nav)이 있으면
  // B6 는 목표가 지도에 확정된 뒤에만 물체로(그 전엔 탐사). 커널 셋: 앞(판마다) → 계획(요청한 판만, 장치 목록) → 행동(판마다).
  // 교사 기억은 환경 상태(I_T_*) + 교사 버퍼(tbuf, 생성자가 장면 묶음이 있으면 잡음). 비동기, 호스트 동기 없음(그래프에 넣을 수 있음)
  void teacher(float* act) const;
  const TBuf& tbuf() const { return tb_; }
  // 재기용(env_bench): 교사 커널 셋을 따로 띄움(teacher() 와 같은 일)
  void teacher_pre(const bsc::NavFb& fb) const;
  void teacher_plan() const;
  void teacher_act(float* act) const;
  // 상태 없는 교사(teacher_sl.h, DAgger 라벨): 행동 = 지금 상태의 함수(캐시만 — 열쇠가 같으면 같은 값). enable_teacher_sl() 로 버퍼를 먼저 잡음(그래프 잡기 전).
  // 상태 있는 교사(teacher)와 버퍼·상태를 나누지 않음 — 같은 스텝에 둘 다 불러도 서로 안 바꿈
  void enable_teacher_sl();
  void teacher_sl(float* act) const;
  const SlBuf& slbuf() const { return sl_; }

  // ---- 장치 단계·씨앗(다시 만들기 없이) ----
  // set_dynamic(무리 비트 = stage_family 의 합): 그 뒤 step 은 단계를 장치 값에서 읽고 무리마다 커널을 띄운다(맞지 않는 무리는 바로 끝남 — 결과 같음).
  // 처음 단계의 무리는 늘 넣는다. BEHAVIOR 무리는 장면 묶음이 있어야 한다
  void set_dynamic(uint32_t families);
  bool dynamic() const { return dyn_; }
  uint32_t families() const { return fam_; }
  // 단계·씨앗 바꾸기 요청(비동기 복사 하나, 동기 없음): 다음 apply() 에서 모든 판을 처음부터(생성자와 같은 상태 — 0 으로 채우고 init).
  // 무리가 set_dynamic 에 없으면 −1. stage 는 호스트 거울(stage())도 바꿈
  int request_stage(int stage, uint64_t seed);
  // 요청이 있으면 적용(커널 둘, 요청이 없으면 바로 끝남 — 그래프에 넣어 둬도 됨). 학습기 장치 커리큘럼은 ctl() 에 직접 pend 를 쓴다
  void apply();
  EnvCtl* ctl() { return ctl_; }

 private:
  int N_, stage_;
  bool arm_free_;
  const bsc::SceneSet* ss_ = nullptr;
  bsc::BCurr* bcurr_ = nullptr;
  const bsc::BCurr* bcurr_src_ = nullptr;
  bsc::NavFb nav_{nullptr, nullptr, nullptr, nullptr};
  float* f_ = nullptr;
  int* iv_ = nullptr;
  uint64_t* rng_ = nullptr;
  EnvCtl* ctl_ = nullptr;       // 장치
  EnvCtl* ctl_h_ = nullptr;     // 고정 호스트 칸 8 개(요청 복사용)
  void* ctl_ev_[8] = {};        // cudaEvent_t
  int ctl_slot_ = 0;
  bool dyn_ = false;
  uint32_t fam_ = 0;
  TBuf tb_{nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr};
  SlBuf sl_{nullptr, nullptr, nullptr, nullptr, 0, 0};
};

// CPU 참조판: 같은 step_env 를 순서대로 돌린다(비교의 정답)
struct CpuEnv {
  int N, stage;
  bool arm_free;
  const bsc::SceneSet* ss = nullptr;   // BEHAVIOR 판: 호스트 장면 묶음(bscene_host.h SceneUpload::host)
  bsc::BCurr curr;
  bsc::NavFb nav{nullptr, nullptr, nullptr, nullptr};   // 호스트 배열(CpuMap 의 nav_fb())
  std::vector<float> f;
  std::vector<int> iv;
  std::vector<uint64_t> rng;
  CpuEnv(int N_, int stage_, uint64_t seed, bool arm_free_ = false, const bsc::SceneSet* ss_host = nullptr, const bsc::BCurr& cu0 = bsc::kBCurrDefault);
  void step(const std::vector<float>& act, std::vector<float>& obs, std::vector<float>& rew, std::vector<int>& done);
  void teacher(std::vector<float>& act);   // DeviceEnv::teacher 와 같은 것(CPU, 판마다 앞 → 계획 → 행동)
  std::vector<float> tf;                    // 교사 버퍼(TBuf, 계획 작업 메모리는 하나)
  std::vector<int> tiv, tlist;
  std::vector<uint8_t> tscr;
  long n_plan = 0;                          // 교사 계획 수(잰 값)
  std::vector<uint32_t> trb;                // 서는 자리 찾기 닿는 칸 비트(TBuf::rb)
  TBuf tbuf() { return TBuf{tf.data(), tiv.data(), tscr.data(), tlist.data(), N, 1, trb.data()}; }
  // 상태 없는 교사(CPU 참조판, DeviceEnv::teacher_sl 과 같은 것)
  std::vector<SlRec> slrec;
  std::vector<uint8_t> slfld;
  long n_slplan = 0;
  void teacher_sl(std::vector<float>& act);
  SlBuf slbuf() { return SlBuf{slrec.data(), slfld.data(), tscr.data(), tlist.data(), N, 1}; }
};

// 잡기 가능 표(Entry::feas·gst·pst5·pst6·grel, SceneSet::has_feas)를 장치에서 계산해 호스트·장치 표에 씀(teacher.h feas_entry — 잡기 모형·교사 계획과 같은 코드).
// upload 뒤에 한 번. 반환 = 잰 초. PF_FEAS 고르기·교사가 판 시작에 서는 자리를 바로 쓰는 데 필요
double pnp_feasibility(bsc::SceneBuild& b, bool quiet = false);
// CPU 로 짝 하나(확인용 — 장치 값과 비트 비교)
void pnp_feasibility_cpu(const bsc::SceneSet& host, int ent, FeasOut& o);

}  // namespace env
