// record_ppo — PPO 교사 체크포인트를 돌려 판 몇 개를 .trp 로 남긴다(학습 뷰어 재생 탭, TRAIN_VIEWER.md 4.4).
// 학습기(libppo)의 공개 헤더(trainer.h, ppo_capi.h)만 쓰고 고치지 않는다: Trainer 를 즉시 실행(그래프 없음)으로 만들고 rollout_step 을 한 스텝씩 부른 뒤
// 스텝마다 환경(DeviceEnv::download)·지도(DeviceMap::download)를 내려받는다 — ppo_verify eval 의 충돌 다시 보기와 같은 방식. 갱신은 하지 않는다.
//
//   record_ppo --ckpt CKPT --out RUN_DIR [--config config.json] [--split eval] [--episodes 8] [--n-env 64] [--track 8]
//              [--stage 2] [--use-map 2] [--goal-from-map 1] [--map 0 0] [--pnp B4 B5 B6] [--scenes a,b] [--seed 7] [--stochastic] [--max-steps 3000]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <memory>

#include "g1_rec.h"
#include "trainer.h"
#include "ppo_capi.h"
#include "env_beh.h"
#include "beh_rec.h"   // 위 env·g1_rec 헤더 뒤에

#define RCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

template <class T> static std::vector<T> dl(const T* d, size_t n) { std::vector<T> h(n); RCK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

// 설정 JSON 에서 숫자 하나(첫 "key": 뒤). 없으면 d
static double jget(const std::string& t, const char* key, double d) {
  const std::string k = std::string("\"") + key + "\"";
  size_t p = t.find(k);
  if (p == std::string::npos) return d;
  p = t.find(':', p + k.size());
  if (p == std::string::npos) return d;
  return std::strtod(t.c_str() + p + 1, nullptr);
}

int main(int argc, char** argv) {
  std::string ckpt, out, cfgp, split = "eval";
  int episodes = 8, N = 64, track = 8, stage = 2, use_map = -1, goal = -1, max_steps = 3000;
  float p0 = 0.f, p1 = 0.f;   // 처음 지도 C0·C1 비율 — 기본 빈 지도 C2(학습과 같음, 10-06 결정)
  float pnp[3] = {0.f, 0.f, 0.f};
  std::string scenes;   // BEHAVIOR 장면 이름(쉼표) — 기본 = 설정 beh.scenes(학습과 같은 장면)   // 잡기 물리 판 B4·B5·B6 비율(stage 3)
  uint64_t seed = 7;
  bool stochastic = false;
  std::string tag;
  int keep_fail = 0;
  bool no_sg = false;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto nx = [&]() { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
    if (s == "--ckpt") ckpt = nx(); else if (s == "--out") out = nx(); else if (s == "--config") cfgp = nx(); else if (s == "--split") split = nx();
    else if (s == "--episodes") episodes = std::stoi(nx()); else if (s == "--n-env") N = std::stoi(nx()); else if (s == "--track") track = std::stoi(nx());
    else if (s == "--stage") stage = std::stoi(nx()); else if (s == "--use-map") use_map = std::stoi(nx()); else if (s == "--goal-from-map") goal = std::stoi(nx());
    else if (s == "--map") { p0 = std::stof(nx()); p1 = std::stof(nx()); }
    else if (s == "--pnp") { for (float& x : pnp) x = std::stof(nx()); }
    else if (s == "--scenes") scenes = nx(); else if (s == "--seed") seed = std::stoull(nx());
    else if (s == "--stochastic") stochastic = true; else if (s == "--max-steps") max_steps = std::stoi(nx());
    else if (s == "--no-sg") no_sg = true; else if (s == "--tag") tag = nx(); else if (s == "--keep-fail") keep_fail = std::stoi(nx());
  }
  if (ckpt.empty() || out.empty()) { std::fprintf(stderr, "usage: record_ppo --ckpt CKPT --out RUN_DIR [--config config.json] [--split eval] [--episodes 8] ...\n"); return 2; }
  if (cfgp.empty()) { const size_t sl = ckpt.rfind('/'); cfgp = (sl == std::string::npos ? std::string(".") : ckpt.substr(0, sl)) + "/config.json"; }
  std::string cfg;
  { std::ifstream f(cfgp); std::stringstream ss; ss << f.rdbuf(); cfg = ss.str(); }
  if (use_map < 0) use_map = (int)jget(cfg, "use_map", 1);
  if (goal < 0) goal = (int)jget(cfg, "goal_from_map", 0);

  PpoConfig c{};
  c.n_env = N; c.horizon = 64; c.epochs = 1; c.minibatches = 4; c.stage = stage; c.use_map = use_map; c.adaptive_lr = 0; c.use_graphs = 0;
  c.log_ring = 8; c.dw_chunk = 1024; c.seed = seed;
  c.gamma = 0.99f; c.lam = 0.95f; c.clip = 0.2f; c.vclip = 0.2f; c.vf_coef = 0.5f; c.lr = 0.f; c.lr_min = 0.f; c.lr_max = 0.f; c.kl_target = 0.01f; c.max_grad_norm = 1.f;
  c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.init_logstd = -0.5f; c.reward_scale = 1.f;
  c.act_dims = (int)jget(cfg, "act_dims", 2); c.goal_from_map = goal;
  c.map_p0 = p0; c.map_p1 = p1; c.map_kmin = 1; c.map_kmax = 8; c.map_reveal_r = 1.5f;
  c.fp8 = (int)jget(cfg, "fp8", 0);
  c.bcurr.p4 = pnp[0]; c.bcurr.p5 = pnp[1]; c.bcurr.p6 = pnp[2];
  if (pnp[0] + pnp[1] + pnp[2] > 0.f) c.bcurr.phys = 8;   // PF_FEAS: 잡기 가능 짝에서만(ppo_pnp 의 feas 1 과 같음)
  ppo::Trainer tr(c);
  if (scenes.empty()) {   // 설정의 "scenes": [ "a", "b" ] 를 그대로
    const size_t k = cfg.find("\"scenes\""), a = cfg.find('[', k), b = cfg.find(']', a);
    if (k != std::string::npos && a != std::string::npos && b != std::string::npos)
      for (size_t q = cfg.find('"', a); q < b; q = cfg.find('"', q + 1)) {
        const size_t e = cfg.find('"', q + 1);
        scenes += (scenes.empty() ? "" : ",") + cfg.substr(q + 1, e - q - 1);
        q = e;
      }
  }
  if (stage >= 3 && !scenes.empty()) {
    PpoBCurr bc = c.bcurr;
    bc.scene_mask = ppo_scene_mask(&tr, scenes.c_str());
    if (!bc.scene_mask) { std::fprintf(stderr, "none of the scenes %s is in the scene set\n", scenes.c_str()); return 2; }
    ppo_set_bcurr(&tr, &bc);
    ppo_set_stage(&tr, stage);   // 환경을 새 장면 값으로 다시 시작(첫 판부터 이 장면)
    tr.apply_body();
  }
  // 체크포인트(ppo_verify eval 과 같은 배치: 머리 64 B + 변수·Adam m·v + 학습 상태). 결정적 정책 = log σ → −12
  std::vector<uint8_t> b(64 + sizeof(float) * 3 * tr.lay.total + sizeof(net::TrainState));
  FILE* f = std::fopen(ckpt.c_str(), "rb");
  if (!f) { std::perror(ckpt.c_str()); return 2; }
  const size_t got = std::fread(b.data(), 1, b.size(), f);
  std::fclose(f);
  float* P = reinterpret_cast<float*>(b.data() + 64);
  if (!stochastic) for (int k = 0; k < env::N_ACT; ++k) P[tr.lay.logstd + k] = -12.f;
  if (got != b.size() || ppo_load(&tr, b.data(), (int64_t)b.size())) { std::fprintf(stderr, "bad checkpoint %s (%zu of %zu bytes)\n", ckpt.c_str(), got, b.size()); return 2; }
  RCK(cudaDeviceSynchronize());

  rec::Out o(out, split);
  const std::string src = rec::Obj().str("kind", "ppo_teacher").str("ckpt", ckpt).str("config", cfgp).num("stage", stage).num("use_map", use_map)
                              .num("goal_from_map", goal).raw("map_p", "[" + rec::jnum(p0) + "," + rec::jnum(p1) + "]").b("deterministic", !stochastic).num("seed", (double)seed).done();
  rec::G1Rec R(N, track, o, "teacher", src);
  R.tag = tag;
  R.sg = !no_sg;
  { size_t p = tag.find_first_of("0123456789"); if (p != std::string::npos) R.ckpt_iter = std::atof(tag.c_str() + p); }
  if (keep_fail > 0) R.max_success = std::max(1, episodes - keep_fail);   // 성공 K−F 개 + 실패 F 개(있으면) — 실패가 없으면 성공으로 채움
  R.home_prefix = "A" + std::to_string(stage) + "_room";
  std::printf("record_ppo: %s  A%d use_map %d goal_from_map %d  first map C0 %.2f C1 %.2f  N %d track %d -> %s (%s)\n", ckpt.c_str(), stage, use_map, goal, p0, p1, N,
              track, o.rep.c_str(), stochastic ? "stochastic" : "deterministic");
  std::vector<float> fs;
  std::vector<int> iv;
  std::vector<uint64_t> rg;
  gmap::MapHost mh;
  std::unique_ptr<rec::BehRec> beh;   // BEHAVIOR 판(stage 3): 장면 머리 = 벽·창·가구·문 + 집을 물체·놓을 곳(beh_rec.h, record_bc 와 같음)
  if (stage >= 3 && tr.scenes) {   // .trp 만 — 점구름 지도는 og_replay(_og.sg)가 만든다
    R.sg = false;
    R.skill = pnp[0] + pnp[1] + pnp[2] > 0.f ? "pick" : "approach";
    R.home_prefix = "B";
    beh = std::make_unique<rec::BehRec>(tr.scenes.get());
    R.scene_fn = [&](int i) {
      env::Soa hs{fs.data(), iv.data(), rg.data(), N};
      env::BState b;
      env::load_b(hs, i, b);
      env::PState p;
      env::load_p(hs, i, p);
      return beh->scene_json(b, p);
    };
  }
  const int T = tr.T;
  for (int k = 0; k < max_steps && !R.done_enough(episodes, keep_fail); ++k) {
    const int t = k % T;
    if (t == 0) {   // 지난 끝 줄(관측·지도 토큰) → 0 줄 (rollout_body 와 같음)
      RCK(cudaMemcpy(tr.obs_buf, tr.obs_buf + (size_t)T * net::N_OBS_G1 * N, sizeof(float) * net::N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
      RCK(cudaMemcpy(tr.obs_rows, tr.obs_rows + (size_t)T * net::N_OBS_G1 * N, sizeof(float) * net::N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
      RCK(cudaMemcpy(tr.tok->at(0), tr.tok->at(T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice));
    }
    RCK(cudaDeviceSynchronize());
    tr.env->download(fs, iv, rg);
    tr.map->download(mh);
    tr.rollout_step(t);
    RCK(cudaDeviceSynchronize());
    const std::vector<float> act = dl(tr.act_env, (size_t)env::N_ACT * N);
    const std::vector<float> rew = dl(tr.rew_buf + (size_t)t * N, N);
    const std::vector<int> done = dl(tr.done_buf + (size_t)t * N, N);
    const std::vector<float> val = dl(tr.val_buf + (size_t)t * N, N);
    R.step(fs, iv, rg, mh, act, rew, done, val, stage, [](int, int) -> const uint8_t* { return nullptr; }, 0);
    if (t == T - 1) tr.rollout_step(T);   // 마지막 가치(부트스트랩) — 다음 바퀴 0 줄 복사와 같은 차례
  }
  std::printf("record_ppo: %d episodes written (success %d, collision %d, timeout %d) -> %s\n", R.finished, R.n_success, R.n_coll, R.n_tout, o.eps.c_str());
  return R.finished ? 0 : 1;
}
