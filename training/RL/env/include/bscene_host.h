// BEHAVIOR 장면 묶음 만들기(호스트, C++17): RASC v3 파일(training/RL/tools/b1kconv) + vla_v1 이름 표 → bsc::SceneSet(호스트 판·장치 판).
// 장치 커널(env_kernel.cu, map_kernel.cu)은 이 파일에 기대지 않는다 — 학습기·뷰어 빌드는 rasc.h 없이 그대로 컴파일된다.
//
// 하는 일(장면마다, 결정적):
//   1. 정적 상자: RASC BOXES 에서 문(열린 채로 둠)·카펫·보이기만·납작한(z1 < 0.03 m)·바닥 덮개(윗면 ≤ 0.2 m, 넓이 > 1 m²: 잔디·포장) 뺌.
//      벽 상자는 문 자리를 잘라 냄(벽 상자는 메시 구멍이 없음) + 문 위 인방. 종류 비트(벽·창·가구, 몸통 충돌 z0 < 0.35, 지도 띠 겹침).
//      1 m 묶음 목록(CSR). 지도 띠 [0.05, 0.50] 점유 래스터(0.1 m 칸, C0·C1 처음 지도의 참 점유).
//   2. 로봇 중심이 설 수 있는 칸(호스트 계획용): 중심 0.13 m 안에 충돌 상자가 없고(몸통 반 폭 0.11 + 0.02) 0.2 m 안에 TRAV_NO_OBJ(벽만 막은 바닥) 빈 칸, 이웃 칸 바닥 높이 차
//      ≤ 문턱(RASC LIMITS 바깥 threshold)만 건넘. 8 이웃 다익스트라(대각은 양 옆이 빈 칸일 때만).
//   3. 시작 조건 표(Entry):
//      - 물체 표(B2·B3): 인스턴스의 집을 과제 물체(RASC PICKS 중 과제 물체, 로봇 시작과 같은 TRAV 성분 = 바깥 한도). 엄격(안 한도: PK_INNER +
//        엄격 문턱 성분이 같음) 판을 앞에. 시작 자세가 몸통 충돌 없음, 시작과 물체가 창(12.8 m) 안에 0.8 m 여유로 들어감, 창 안 다익스트라로
//        물체 둘레 팔 닿는 원(0.38 m) 칸에 닿음 — 아니면 뺀다(뺀 까닭을 센다).
//      - 방 표(B1): 시작 방이 아닌 방의 바닥 가구(z0 < 0.35, 한 변 ≤ 3 m) 앞 칸(가구에서 0.30–0.60 m, 그 방 안)까지 닿는 것 중 가장 가까운 점.
//      - 지도 물체 9: 목표 + 다른 과제 물체(≤ 5, 창 안) + 창 안 가구(가까운 순). 창 좌표 축 정렬 상자, 이름 표 행.
//   4. 이름 표: 과제 물체 synset → vla_v1 names 행(종류 sim/behavior 먼저), 가구 종류 → 같은 글자 이름·synset·낱말 규칙·"furniture".
//      확신도 conf1·conf2[생김새 7][이름] = name128·app128 코사인(vla_vocab_gen.py 와 같은 정의, 어휘 = 상위어·평가용 뺀 행).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "bscene.h"

namespace bsc {

struct BuildOpt {
  std::string dir;                 // RASC 폴더(기본 ~/ra_b1k)
  std::string vla;                 // vla_v1 폴더(기본: 저장소 training/data/vla_v1)
  std::vector<std::string> only;   // 장면 이름만(비면 폴더의 *.rasc 전부, RASC scene_index 순)
  int cap_room = 4096;             // (장면, 방 표, split) 판 수 상한
  int cap_obj = 4096;              // (장면, 물체 표, split)
  int rooms_per_inst = 2;          // 인스턴스 하나에서 B1 목표 방 수
  bool quiet = false;
};

struct SceneStats {               // 확인·보고용(잰 값)
  std::string name;
  int W = 0, H = 0, nbox_in = 0, nbox = 0, ground = 0, door_cuts = 0, nbin_items = 0, nroom = 0, ndoor = 0;
  long cells = 0, free_cells = 0;
  double occ_iou = 0;              // 띠 점유 래스터 대 !TRAV_OPEN_DOOR(방 칸 안) IoU
  double occ_agree = 0;            // 같은 칸 비율(방 칸 안)
  int inst = 0, inst_pick = 0, inst_pick_reach = 0;          // 인스턴스, 집을 과제 물체가 있는, 그중 시작 성분과 같은 짝이 있는(RASC)
  int obj_try = 0, obj_ok = 0, obj_inner = 0, rej_stance = 0;                 // 물체 표: 해 본 (인스턴스, 물체), 넣은 것, 그중 엄격
  int rej_start = 0, rej_window = 0, rej_unreach = 0, rej_cap = 0;   // 뺀 까닭: 시작 충돌, 창에 안 들어감, 창 안 길 없음, 상한
  int room_try = 0, room_ok = 0, room_nocand = 0, room_unreach = 0;
  int ent[2][2] = {{0, 0}, {0, 0}}, ent_in[2] = {0, 0};   // [표][split], 물체 표 엄격 [split]
  double path_sum[2] = {0, 0};     // 표마다 경로 합(평균용)
  double ratio_sum[2] = {0, 0};    // 경로 / 직선
  int name_hit = 0, name_miss = 0; // 과제 물체 이름 찾음 / 못 찾음(상위어로)
  double secs = 0;
};

// 호스트 소유 묶음. host 는 호스트 포인터 SceneSet(CPU 참조판), dev 는 장치에 올린 SceneSet 포인터(커널 인자)
struct SceneBuild {
  struct Sc {
    std::string name;
    SceneDev d{};                            // 호스트 포인터(아래 벡터)
    std::vector<SBox> box;
    std::vector<uint8_t> bkind;
    std::vector<int16_t> bname;
    std::vector<uint32_t> bstart;
    std::vector<uint16_t> bitem;
    std::vector<uint8_t> room;
    std::vector<uint32_t> occ;
    std::vector<int8_t> rtype;
    std::vector<SDoor> door;
    std::vector<uint8_t> freeg;              // 로봇 중심 설 수 있음(호스트 계획·확인용, 장치에 안 올림)
    std::vector<int16_t> floor_mm;           // RASC FLOOR_Z
    std::vector<std::string> room_names;
    float threshold = 0.025f;
  };
  std::vector<Sc> sc;
  std::vector<Entry> ent;
  std::vector<float> conf1, conf2;           // [7][nname]
  std::vector<int16_t> sim3, hyper;
  std::vector<std::string> name_en;
  int nname = 0;
  SceneSet host{};
  SceneSet* dev = nullptr;                   // 장치 SceneSet(upload 뒤)
  std::vector<void*> dbuf;                   // 장치 버퍼(free_dev 가 풂)
  std::vector<SceneStats> stats;
  size_t dev_bytes = 0;
};

bool build_scenes(const BuildOpt& opt, SceneBuild& out, std::string* err);   // RASC·이름 표 읽고 묶음·표 만듦(호스트)
bool upload(SceneBuild& b, std::string* err);                                // 장치에 올림(b.dev)
void free_dev(SceneBuild& b);
std::string stats_text(const SceneBuild& b);                                 // 사람이 읽는 표
std::string default_vla_dir();                                               // 이 파일 기준 저장소 training/data/vla_v1
std::string default_rasc_dir();                                              // $HOME/ra_b1k

// 확인 도구가 쓰는 호스트 계산(같은 규칙)
bool body_free_host(const SceneBuild::Sc& s, const Entry& e, float x, float y, float yaw);   // 창 좌표 자세에서 몸통이 정적·과제 물체 상자에 안 닿음
// 창 안 8 이웃 다익스트라(로봇 중심 칸): 창 좌표 시작에서, 창 칸 [WIN²] 거리(m, 못 가면 −1)
void window_dijkstra(const SceneBuild::Sc& s, float wx, float wy, float sx, float sy, std::vector<float>& dist);
// 목표에서 거꾸로(참 장면): B1 = 목표 점 칸, 물체 표 = 물체 바닥 자국에서 0.38 m 안 칸이 씨앗(0). 대본 정책·SPL 용
void goal_field(const SceneBuild::Sc& s, const Entry& e, std::vector<float>& dist);
// 참 장면 다익스트라, 씨앗 = (tx, ty) 에서 rad 안 칸(빈 칸이 아니어도) — 지도 거리장(map.h 8 절)과 같은 씨앗으로 견주기용
void disk_field(const SceneBuild::Sc& s, const Entry& e, float tx, float ty, float rad, std::vector<float>& dist);
bool stance_ok(const SceneBuild::Sc& s, const Entry& e, float cx, float cy);   // B3 서는 자리(목표를 봄, 몸통 안 닿음, 잡는 점 작업 공간에 물체)

}  // namespace bsc
