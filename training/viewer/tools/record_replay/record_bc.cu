// record_bc — BC 학생(student-lite 또는 G5 영상 학생) 체크포인트로 판 몇 개를 .trp 로 남긴다(학습 뷰어 재생 탭, TRAIN_VIEWER.md 4.4).
// 학습기(libbc)의 공개 헤더(bc.h, bc_capi.h, bc_render.h)만 쓰고 고치지 않는다: 그래프 없이 Bc::rollout_step(t, 학생) 을 한 스텝씩 부르고
// 스텝마다 환경·지도를 내려받는다. 학생이 움직이고(교사 앞 계산은 라벨용으로 그대로 돎) 기록 버퍼에는 쓰지 않는다(record 0).
// 영상 학생이면 학생이 본 카메라 2 장(같은 bcr 렌더, 256²)을 0.5 s 마다 JPEG 으로 img 섹션에 싣는다.
//
//   record_bc --student STUDENT.bin --out RUN_DIR [--config config.json] [--teacher CKPT] [--split eval] [--episodes 8] [--n-env 64] [--track 8]
//             [--map 0.2 0.6] [--seed 7] [--no-images] [--max-steps 3000]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bc.h"
#include "bc_capi.h"
#include "bc_render.h"
#include "bscene_host.h"
#include "env_beh.h"
#include "g1_rec.h"
#include "beh_rec.h"
#include <map>
#include <memory>
#include <unistd.h>

#define RCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

template <class T> static std::vector<T> dl(const T* d, size_t n) { std::vector<T> h(n); RCK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
static double jget(const std::string& t, const char* key, double d) {
  const std::string k = std::string("\"") + key + "\"";
  size_t p = t.find(k);
  if (p == std::string::npos) return d;
  p = t.find(':', p + k.size());
  return p == std::string::npos ? d : std::strtod(t.c_str() + p + 1, nullptr);
}
static std::string jgets(const std::string& t, const char* key) {
  const std::string k = std::string("\"") + key + "\"";
  size_t p = t.find(k);
  if (p == std::string::npos) return "";
  p = t.find('"', t.find(':', p + k.size()) + 1);
  const size_t e = t.find('"', p + 1);
  return p == std::string::npos || e == std::string::npos ? "" : t.substr(p + 1, e - p - 1);
}
// 설정의 배열 "key": [a, b, c] 의 k 번째 수(없으면 d)
static double jarr(const std::string& t, const char* key, int k, double d) {
  const std::string kk = std::string("\"") + key + "\"";
  size_t p = t.find(kk);
  if (p == std::string::npos) return d;
  p = t.find('[', p + kk.size());
  if (p == std::string::npos) return d;
  const char* q = t.c_str() + p + 1;
  for (int i = 0; i < k; ++i) { q = std::strchr(q, ','); if (!q) return d; ++q; }
  return std::strtod(q, nullptr);
}
static std::string tilde(std::string s) { if (!s.empty() && s[0] == '~') s = std::string(std::getenv("HOME")) + s.substr(1); return s; }

int main(int argc, char** argv) {
  std::string student, out, cfgp, teacher, split = "eval", text_table;
  int episodes = 8, N = 64, track = 8, max_steps = 3000;
  float p0 = 0.2f, p1 = 0.6f;
  uint64_t seed = 7;
  bool images = true;
  std::string tag, actor = "student";
  int keep_fail = 0;
  bool no_sg = false, map_given = false;
  int og_per = 2;                                   // BEHAVIOR 판: 체크포인트마다 학생·교사 각 몇 판을 OG 다시 돌리기 줄에(0 = 끔)
  std::string og_queue = getenv("RA_TRAINVIEW_WORK") && *getenv("RA_TRAINVIEW_WORK") ? std::string(getenv("RA_TRAINVIEW_WORK")) + "/og_queue" : std::string(REPO_DIR) + "/data/trainview_work/og_queue";
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto nx = [&]() { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
    if (s == "--student") student = nx(); else if (s == "--out") out = nx(); else if (s == "--config") cfgp = nx(); else if (s == "--teacher") teacher = nx();
    else if (s == "--split") split = nx(); else if (s == "--episodes") episodes = std::stoi(nx()); else if (s == "--n-env") N = std::stoi(nx());
    else if (s == "--track") track = std::stoi(nx()); else if (s == "--map") { p0 = std::stof(nx()); p1 = std::stof(nx()); map_given = true; } else if (s == "--seed") seed = std::stoull(nx());
    else if (s == "--no-images") images = false; else if (s == "--max-steps") max_steps = std::stoi(nx());
    else if (s == "--no-sg") no_sg = true; else if (s == "--tag") tag = nx(); else if (s == "--keep-fail") keep_fail = std::stoi(nx()); else if (s == "--text-table") text_table = nx();
    else if (s == "--actor") actor = nx();
    else if (s == "--og-per") og_per = std::stoi(nx()); else if (s == "--og-queue") og_queue = nx();   // student(기본) | teacher(대본 교사가 몲 — BEHAVIOR 집기·놓기 판) | both
  }
  if (student.empty() || out.empty()) { std::fprintf(stderr, "usage: record_bc --student STUDENT.bin --out RUN_DIR [--config config.json] ...\n"); return 2; }
  if (cfgp.empty()) { const size_t sl = student.rfind('/'); cfgp = (sl == std::string::npos ? std::string(".") : student.substr(0, sl)) + "/config.json"; }
  std::string cfg;
  { std::ifstream f(cfgp); std::stringstream ss; ss << f.rdbuf(); cfg = ss.str(); }
  if (teacher.empty()) teacher = tilde(jgets(cfg, "teacher"));
  if (!map_given) { p0 = (float)jget(cfg, "map_p0", p0); p1 = (float)jget(cfg, "map_p1", p1); }   // 학습 설정의 처음 지도 비율
  BcConfig c{};
  c.n_env = N; c.horizon = 64; c.stage = (int)jget(cfg, "stage", 2); c.use_map = (int)jget(cfg, "use_map", 2); c.teacher_use_map = (int)jget(cfg, "teacher_use_map", 2);
  c.student_goal = (int)jget(cfg, "student_goal", 0); c.use_graphs = 0; c.log_ring = 16; c.mb = 64; c.upd_steps = 1; c.dw_chunk = 1024; c.store_render = 1;   // 영상 학생은 store_render 1 이 필요(기록은 안 하므로 버퍼만)
  c.cap = 4096; c.seed = 1; c.env_seed = seed; c.lr = 0.f; c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.max_grad_norm = 1.f;
  c.map_p0 = p0; c.map_p1 = p1; c.map_kmin = 1; c.map_kmax = 8; c.map_reveal_r = 1.5f; c.fp8 = (int)jget(cfg, "fp8", 0);
  c.vision = (int)jget(cfg, "vision", 0); c.head = (int)jget(cfg, "head", 0); c.chunk = (int)jget(cfg, "chunk", 16); c.flow_steps = (int)jget(cfg, "flow_steps", 10);
  c.text = (int)jget(cfg, "text", 0); c.render_profile = (int)jget(cfg, "render_profile", 1); c.render_batch = N; c.img_dim = 16; c.sample_render = 1;
  c.vit_prec = (int)jget(cfg, "vit_prec", 0);
  c.mlp_w = (int)jget(cfg, "mlp_w", 0);   // MLP 학생 몸통 폭   // G5 영상 학생은 vit_prec 0(FP32 누산)으로 학습했다(BC README G6) — 설정에 없으면 0
  // BEHAVIOR 집기·놓기(stage 3, E6): 장면 묶음·단계 비율·대본 교사·잡기 가능 표·서는 자리 후보·단계 문턱(bc_run 설정 키 그대로)
  if (c.stage >= 3) {
    c.beh = 1; c.map_nav_k = (int)jget(cfg, "nav_k", 0);
    c.b_p1 = (float)jarr(cfg, "mix", 0, 0); c.b_p2 = (float)jarr(cfg, "mix", 1, 0);
    c.b_p4 = (float)jarr(cfg, "pnp", 0, 0); c.b_p5 = (float)jarr(cfg, "pnp", 1, 0); c.b_p6 = (float)jarr(cfg, "pnp", 2, 0);
    c.b_p_slip = (float)jarr(cfg, "fail", 0, 0); c.b_p_occ = (float)jarr(cfg, "fail", 1, 0);
    c.teacher_script = (int)jget(cfg, "teacher_script", 0); c.b_feas = (int)jget(cfg, "feas", 0);
    c.b_gcand = (int)jget(cfg, "gcand", 0); c.b_sltol = (int)jget(cfg, "sltol", 0);
    c.act_mask = (uint32_t)jget(cfg, "act_mask", 0); c.goal_drop = 0.f;
    c.arch = (int)jget(cfg, "arch", 0);
  }
  if (actor != "student" && !c.teacher_script) { std::fprintf(stderr, "record_bc: --actor teacher needs teacher_script (BEHAVIOR) in the config\n"); return 2; }
  void* h = bc_create(&c);
  bc::Bc& B = *static_cast<bc::Bc*>(h);
  if (!(c.teacher_script && teacher.empty()) && bc_load_teacher(h, teacher.c_str())) { std::fprintf(stderr, "teacher %s failed\n", teacher.c_str()); return 2; }
  if (c.text) {
    if (text_table.empty()) text_table = tilde(jgets(cfg, "text_table"));
    if (text_table.empty()) text_table = std::string(REPO_DIR) + "/training/BC/data/instr_a2.f32";
    if (bc_load_text_table(h, text_table.c_str()) <= 0) { std::fprintf(stderr, "text table %s failed\n", text_table.c_str()); return 2; }
  }
  if (bc_load_student(h, student.c_str())) { std::fprintf(stderr, "student %s failed\n", student.c_str()); return 2; }
  // 한 번 돌기(actor: 학생 / 대본 교사). --actor both 면 학생 K 판 뒤 같은 씨앗으로 교사 K 판(태그 뒤 "-teacher")
  int total = 0;
  std::unique_ptr<rec::BehRec> beh;   // BEHAVIOR 판 기록(beh_rec.h)
  std::map<long, rec::InEp> in_ep;
  std::vector<gmap::MapTok> in_tok;
  std::vector<float> in_obs, in_act, in_sact, in_exec;
  std::vector<uint16_t> in_x0;
  std::vector<env::SlRec> in_sl;
  std::vector<uint8_t> in_buf;
  // OG 다시 돌리기 줄(og_queue.py): 체크포인트마다 학생·교사 앞 판 og_per 개씩 — 실제 렌더 RGB-D + 우리 인지(ObjectSAM + SigLIP 2 + objprob)로 <판>_og.sg
  int og_n[2] = {0, 0};
  bool og_spawned = false;
  auto og_queue_add = [&](const std::string& path, const std::string& line, bool stud) {
    if (og_per <= 0 || og_n[stud ? 0 : 1] >= og_per) return;
    ++og_n[stud ? 0 : 1];
    const std::string q = tilde(og_queue);
    rec::mkdirs(q);
    char nm[96];
    std::snprintf(nm, sizeof nm, "/%ld_%d_%c%d.job", (long)time(nullptr), (int)getpid(), stud ? 's' : 't', og_n[stud ? 0 : 1]);
    const std::string jp = q + nm;
    std::ofstream(jp + ".tmp") << rec::Obj().str("trp", path).raw("meta", line).str("ckpt", tag).str("actor", stud ? "student" : "teacher").done() << "\n";
    std::rename((jp + ".tmp").c_str(), jp.c_str());
    if (!og_spawned) {   // 일꾼(하나만 — flock -n): 새 세션으로 떼어 띄움(학습기·이 도구가 끝나도 줄을 마저 돌림, og.lock 으로 OG 하나)
      og_spawned = true;
      const std::string cmd = "setsid -f nice -n 19 " + std::string(REPO_DIR) + "/training/viewer/tools/og_replay/og_queue.sh '" + q + "' >> '" + q + "/worker.log' 2>&1 < /dev/null";
      if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "record_bc: cannot start og_queue worker\n");
    }
  };
  auto run_pass = [&](bool stud, const std::string& tag) {
  bc_reset_env(h, seed);
  bc_set_mode(h, stud ? 1 : 0, 0);
  RCK(cudaDeviceSynchronize());
  RCK(cudaDeviceSynchronize());

  rec::Out o(out, split);
  const std::string src = rec::Obj().str("kind", "bc_student").str("student", student).str("teacher", teacher).str("config", cfgp).num("vision", c.vision).num("head", c.head)
                              .num("text", c.text).num("use_map", c.use_map).raw("map_p", "[" + rec::jnum(p0) + "," + rec::jnum(p1) + "]").num("seed", (double)seed).done();
  rec::G1Rec R(N, track, o, stud ? "student" : "teacher", src);
  R.tag = tag;
  R.sg = !no_sg;
  { size_t p = tag.find_first_of("0123456789"); if (p != std::string::npos) R.ckpt_iter = std::atof(tag.c_str() + p); }
  if (keep_fail > 0) R.max_success = std::max(1, episodes - keep_fail);   // 성공 K−F 개 + 실패 F 개(있으면) — 실패가 없으면 성공으로 채움
  R.home_prefix = "A" + std::to_string(c.stage) + "_room";
  std::vector<float> fs;
  std::vector<int> iv;
  std::vector<uint64_t> rg;
  if (c.stage >= 3 && B.scenes) {   // BEHAVIOR 판: 창 안 정적 상자(창 좌표) + 집을 물체 회전 상자 — 판 시작에
    R.sg = false;
    R.skill = "pick";
    R.home_prefix = "B";
    if (!beh) beh = std::make_unique<rec::BehRec>(B.scenes.get());
    // 장면 머리: 벽·창·가구(진짜 이름)·문 + 집을 물체·놓을 곳 + OG 다시 돌리기 정보(world) + 짝이 되는 까닭(pnp) — beh_rec.h
    R.scene_fn = [&](int i) {
      env::Soa hs{fs.data(), iv.data(), rg.data(), N};
      env::BState b;
      env::load_b(hs, i, b);
      env::PState p;
      env::load_p(hs, i, p);
      return beh->scene_json(b, p);
    };
    // 스텝마다 정책 입력(beh_rec.h InLayout): 지도 토큰·관측·X0·행동·교사 라벨·특권 상태
    R.frame_hook = [&](int i, long ep, uint32_t fr, TrpWriter* w) {
      if (i >= (int)in_tok.size()) return;
      env::Soa hs{fs.data(), iv.data(), rg.data(), N};
      float lab[8], ex[8], ac[8];
      for (int k = 0; k < 8; ++k) { lab[k] = in_sact.empty() ? NAN : in_sact[(size_t)k * N + i]; ex[k] = in_exec[(size_t)k * N + i]; ac[k] = stud ? in_act[(size_t)i * 8 + k] : lab[k]; }
      rec::in_pack(in_buf, in_tok[i], in_obs.data(), N, i, in_x0.empty() ? nullptr : in_x0.data() + (size_t)i * net::X0_W, ac, lab, ex, in_sl.empty() ? nullptr : &in_sl[i], hs, in_ep[ep]);
      trp_record(w, "inputs", fr, in_buf.data(), in_buf.size());
    };
    R.pre_finish = [&](int i, long ep, TrpWriter* w) {
      (void)i;
      rec::InEp& E = in_ep[ep];
      std::string nm = "{";
      for (int r : E.names) nm += std::string(nm.size() > 1 ? "," : "") + "\"" + std::to_string(r) + "\":" + rec::jstr(beh->nm(r));
      nm += "}";
      const std::string j = rec::Obj().raw("layout", rec::in_layout_json()).str("actor", stud ? "student" : "teacher").b("student_goal_hidden", c.student_goal == 0)
                                .num("use_map", stud ? c.use_map : c.teacher_use_map).num("mlp_w", c.mlp_w).raw("names", nm).num("instr", E.instr).str("instr_text", beh->instr_text(E.instr))
                                .raw("chosen_stance", "[" + rec::jnum(E.chosen[0]) + "," + rec::jnum(E.chosen[1]) + "," + rec::jnum(E.chosen[2]) + "," + rec::jnum(E.chosen[3]) + "]").done();
      trp_set_head(w, "inputs", j.c_str());
      in_ep.erase(ep);
    };
    R.post_finish = [&](int i, long ep, const std::string& path, const std::string& line) { (void)i; (void)ep; og_queue_add(path, line, stud); };
  }
  const bool cams = images && c.vision && B.rnd;
  std::printf("record_bc [%s]: %s (vision %d head %d text %d) teacher %s  A%d  N %d track %d -> %s%s\n", actor.c_str(), student.c_str(), c.vision, c.head, c.text, teacher.c_str(), c.stage, N,
              track, o.rep.c_str(), cams ? "  + camera JPEG 2 Hz" : "");
  gmap::MapHost mh;
  std::vector<uint8_t> rgb[2];
  const int T = B.T, K = std::min(track, N), RES = bcr::RES;
  for (int k = 0; k < max_steps && !R.done_enough(episodes, keep_fail); ++k) {
    const int t = k % T;
    RCK(cudaDeviceSynchronize());
    B.env->download(fs, iv, rg);
    B.map->download(mh);
    B.rollout_step(t, stud);
    RCK(cudaDeviceSynchronize());
    if (cams) {   // 이 스텝에 학생이 본 그림(rs_roll = 스텝 전 상태): 추적하는 판 0..K-1 을 한 번 더 그림(다음 스텝 인코더가 다시 그리므로 정책에 영향 없음)
      bcr::render(B.rnd, B.rs_roll, K, 0);
      RCK(cudaDeviceSynchronize());
      for (int cam = 0; cam < 2; ++cam) rgb[cam] = dl(bcr::rgb(B.rnd, cam), (size_t)K * RES * RES * 3);
    }
    const std::vector<float> act = dl(B.act_env, (size_t)env::N_ACT * N);
    if (R.frame_hook) {   // 이 스텝 정책 입력(rollout_step 이 쓴 그대로 — 다음 스텝 전까지 유효)
      in_tok = dl(B.tok->at(t), (size_t)K);
      in_obs = dl(B.obs_col + (size_t)(t % 2) * env::N_OBS * N, (size_t)env::N_OBS * N);
      in_x0 = dl(stud ? B.sb.x0 : B.nt.x0, (size_t)K * net::X0_W);
      if (stud) in_act = dl(B.sb.act, (size_t)K * 8);
      if (c.teacher_script) in_sact = dl(B.sact, (size_t)8 * N);
      if (B.env->slbuf().rec) in_sl = dl(B.env->slbuf().rec, (size_t)K);
      in_exec = act;
    }
    const std::vector<float> rew = dl(B.rew, N);
    const std::vector<int> done = dl(B.done, N);
    R.step(fs, iv, rg, mh, act, rew, done, {}, c.stage,
           [&](int i, int cam) -> const uint8_t* { return cams && i < K ? rgb[cam].data() + (size_t)i * RES * RES * 3 : nullptr; }, cams ? RES : 0);
  }
  std::printf("record_bc: %d episodes written (success %d, collision %d, timeout %d) -> %s\n", R.finished, R.n_success, R.n_coll, R.n_tout, o.eps.c_str());
  total += R.finished;
  };
  if (actor == "both" || actor == "student") run_pass(true, tag);
  if (actor == "both" || actor == "teacher") run_pass(false, actor == "both" ? (tag.empty() ? std::string("teacher") : tag + "-teacher") : tag);
  bc_destroy(h);
  return total ? 0 : 1;
}
