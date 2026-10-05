// scenemap C ABI(include/scenemap.h) 구현 — mapper2d(외부 자세로 격자) + objmap(검출 → 물체 지도) + 저장(sm_save_dsg, dsg_save.cpp).
//
// 입력 스레드 하나가 push 를 부르고, 계획기 쪽이 아무 때나 스냅숏을 만든다. 상태는 뮤텍스 하나 아래.
// 스냅숏은 그 순간의 자세·상태·격자(i8)를 통째로 복사한 것(참조 카운트) — 격자 600×600 에서 약 1 ms.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scenemap.h"
#include "scenemap/stream.hpp"
#include "scenemap/walls.hpp"
#include "scenemap/bestview.hpp"
#include "scenemap/dsg_save.hpp"
#include "scenemap/fk.hpp"
#include "scenemap/objmap.hpp"
#include "scenemap/rooms.hpp"
#include "scenemap/sgraph.hpp"
#include "scenemap/mapper2d.hpp"
#include "scenemap/timing.hpp"

using namespace scenemap;

namespace {
struct Prop {
  double stamp;
  float q[kMaxProprioDim];
  double v[3];                  // 적분에 쓰는 베이스 속도(로봇 기준 vx, vy, wz). 오도메트리 자세 차 / dt
};
struct ViewSlot {
  BestViewPtr v;
  double q = 0;                 // 비교에 쓰는 품질(옮겨짐·놓기 때 0 으로)
};
// 이름 종류 기본 표(팀 벤치마크 dynamic-object-mapping-benchmark 의 정답 정의: 배경 구조물은 instance 0 — 물체 아님,
// 가구·가전·붙박이는 정답 물체지만 movable 0). 비교는 머리 명사: 정규화한 이름 == 항목 이거나 " 항목" 으로 끝남
// ("glass door" → door, "floor lamp" → lamp, "coffee table" → table). 끝 's' 하나는 무시.
const char* const kStructureNames[] = {"wall", "floor", "ceiling", "door", "doorway", "door frame", "window", "pillar", "column",
                                       "partition", "staircase", "stairs", "stair", "railing", "baseboard",
                                       // 사람(COCO person): 시뮬에는 없고 로봇 팔·몸이 이것으로 잘못 잡힘 — 노드로 만들지 않음
                                       "person",
                                       // objprob 상위어 라벨(realbag_run kApLabels): 구조물 묶음
                                       "structure", "outdoors"};
const char* const kStaticNames[] = {
    "table", "desk", "counter", "countertop", "sofa", "couch", "shelf", "shelving unit", "bookshelf", "bookcase", "cabinet",
    "wardrobe", "dresser", "chest of drawers", "nightstand", "sideboard", "bed", "bench", "island", "refrigerator", "fridge",
    "freezer", "oven", "stove", "range", "cooktop", "microwave", "dishwasher", "washer", "dryer", "washing machine", "sink",
    "toilet", "bathtub", "bath", "shower", "fireplace", "piano", "television", "tv", "lamp", "chandelier", "plant",
    "picture frame", "picture", "painting", "mirror", "rug", "carpet", "curtain", "blind", "radiator", "heater",
    "light switch", "electric outlet", "outlet", "socket", "vent", "fixture", "appliance",
    "furniture"};   // objprob 상위어 라벨(가구 묶음)

// objprob 구조 물체(지우지 않고 structural 노드로): 지도·RecallVLA 가 문 토큰·방 나누기·계단 위험으로 씀
const char* const kStructObjNames[] = {"door", "doorway", "door frame", "window", "staircase", "stairs", "stair", "railing", "pillar", "column"};

// 바닥에 깔리는 물체: objmap 바닥 조각 거르기(점이 거의 다 바닥 높이 floor_h 아래면 물체 아님)에서 뺀다
const char* const kFloorLevelNames[] = {"rug", "carpet", "mat", "doormat", "floor mat", "bath mat"};

std::string normName(std::string t) {
  // ".n.01" 같은 WordNet 꼬리 버림, '_' → ' ', 소문자, 앞뒤 공백 정리
  const size_t p = t.find(".n.");
  if (p != std::string::npos) t.resize(p);
  for (char& ch : t) ch = ch == '_' ? ' ' : char(std::tolower(static_cast<unsigned char>(ch)));
  while (!t.empty() && t.back() == ' ') t.pop_back();
  while (!t.empty() && t.front() == ' ') t.erase(t.begin());
  return t;
}

bool headMatch(const std::string& name, const std::string& e) {
  auto ends = [](const std::string& a, const std::string& b) {
    return a == b || (a.size() > b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0 && a[a.size() - b.size() - 1] == ' ');
  };
  if (e.empty()) return false;
  return ends(name, e) || (name.size() > 1 && name.back() == 's' && ends(name.substr(0, name.size() - 1), e));
}

constexpr float kCropMargin = 0.10f;
constexpr double kWallAngleEvery = 2.0;          // 벽 방향 θ 다시 재는 간격(영상 시각 s)
constexpr double kWallAngleHyst = M_PI / 180.0;  // θ 가 이만큼 넘게 바뀔 때만 새 값
constexpr double kWallAlignedEvery = 0.5;        // 기울어진 지도: 돌린 격자 벽 다시 뽑는 간격(s, 격자가 바뀌었을 때)
constexpr int kCropMaxSide = 256;
}  // namespace

struct sm_ctx {
  std::mutex mu;
  MapperParams params;
  Mapper2D mapper;                    // 2D 격자 쌓기(자세는 밖에서 — Cartographer·정답·오도메트리)
  ObjParams oparams;
  ObjectMap om;
  std::vector<std::string> labels;
  std::vector<std::string> kind_names[3];   // [1] 구조물, [2] 고정(정규화한 이름)
  std::vector<uint8_t> kinds;               // labels[i] 의 종류
  std::vector<uint32_t> handled;
  // 로봇(sm_set_robot·config "robot"; limo_omx 하나): proprio 형식·순기구학·몸 크기 매개변수
  int robot = kRobotLimoOmx;
  bool odom_pose = true;        // LIMO: 오도메트리 자세 차로 적분(false: proprio 의 twist 3..6 을 그대로)
  bool have_odom = false;       // LIMO: 지난 proprio 의 오도메트리 자세
  double odom_prev[3] = {0, 0, 0}, odom_prev_stamp = 0;
  std::deque<Prop> pending;     // 아직 적분하지 않은 proprio(영상 stamp 를 기다림)
  Prop last_used{};             // 가장 최근에 적분한 proprio
  bool have_used = false;
  sm_status st{};
  // best view
  std::unordered_map<uint32_t, ViewSlot> views;
  std::vector<uint32_t> last_assoc;   // 마지막 영상의 검출 → 물체 id
  std::vector<uint8_t> last_view_upd; // 마지막 영상의 검출 k 가 그 물체의 best view 가 됨(sm_last_views — CLIP 갱신 신호)
  std::vector<float> last_view_q;     // 검출 k 의 모습 품질(유효 마스크 넓이 × 점수, 안 붙으면 0)
  size_t ev_seen = 0;                 // 처리한 objmap 사건 수(옮겨짐·놓기 → 품질 0)
  uint32_t view_ver = 0;
  uint64_t epoch = 0;                 // sm_reset 마다 +1(잠금 밖 자르기 중 reset 이면 버림)
  // 저장(PNG 더러움): 물체 id → 마지막으로 쓴 모습 version. 디렉터리가 바뀌거나 새 판이면 비우고 objects/ 정리
  std::unordered_map<uint32_t, uint32_t> saved_ver;
  std::unordered_map<uint32_t, uint32_t> saved_cloud_ver;   // 물체 id → 마지막으로 쓴 구름 version
  std::string saved_dir;
  bool clean_objects = true;
  RoomTracker rooms;                  // 방 나누기(자기 잠금, sm_snapshot 이 잠금 밖에서 부름)
  std::atomic<bool> rooms_force{false};
  // 자세 원천(sm_set_pose_mode): EXT(기본, sm_push_ext_pose — Cartographer, 없으면 적분) / ODOM(적분만) / GT(sm_push_pose 정답 자세)
  int pose_mode = SM_POSE_EXT;
  // 외부 카메라 외부 자세(sm_set_cam_extrinsic): cam 0 의 베이스 ← 광학 프레임을 순기구학 대신 이 값으로(카메라만 있는 기록 — 벤치마크)
  bool ext_cam = false;
  float ext_T[12] = {0};
  std::deque<sm_pose2> gtq;           // 외부 자세(stamp 순). GT 가 아닌 모드에서는 진단(떠밀림)에만 씀
  std::deque<sm_pose2> extq;          // EXT 모드 자세(sm_push_ext_pose — Cartographer 등 다른 SLAM). 정답(gtq)과 따로라 진단은 그대로
  bool diag_align = false;            // 진단: map ← 외부 프레임 맞춤(첫 keyframe 에서 오차 0)
  Pose2 align;
  sm_pose_diag diag{};
  double diag_s2xy = 0, diag_s2yaw = 0;
  int map_policy = 1, still_every = 50;
  FILE* slog = nullptr;               // 진단: SM_SLAM_LOG=<파일> 이면 keyframe 마다 한 줄(예측·맞추기·정답)
  KeyframeStats last_kf;
  // 스냅숏 격자 사본(보이는 값이 바뀌었을 때만 새로 — 그 사이 스냅숏은 같은 배열을 나눠 씀)
  std::shared_ptr<const std::vector<int8_t>> grid8;
  uint64_t grid8_ver = ~0ull;
  int grid8_w = 0, grid8_h = 0;
  // 벽 선분(walls.hpp): 격자 사본이 새로 만들어질 때 바뀐 행만 다시 계산해 둔다. 스냅숏은 이 결과를 나눠 쓴다
  scenemap::WallExtractor walls;
  std::shared_ptr<const std::vector<scenemap::WallSeg>> wallsp;
  int walls_x0 = 0, walls_y0 = 0;
  std::vector<scenemap::WallRect> walls_rects;
  // 실시간 스트림(stream.hpp): 마지막으로 보낸 격자 모양(바뀌면 전체), 영역 복사 버퍼
  scenemap::Streamer stream;
  int st_w = 0, st_h = 0, st_x0 = 0, st_y0 = 0;
  std::vector<int8_t> st_scratch;
  std::string st_last_view;                 // 마지막으로 보낸 요약(맨 앞 stamp·pose 제외) — 같으면 안 보냄
  TClock::time_point st_last_view_t{};
  uint64_t st_view_key = ~0ull;             // 마지막으로 요약을 만든 때의 상태 키(영상 수 + 그래프 버전) — 같으면 만들지 않는다
  std::atomic<uint64_t> st_views_built{0}, st_views_skipped{0};
  std::atomic<double> st_view_us{0};         // 요약 한 번 만드는 시간(µs, 비동기 스레드)
  uint64_t walls_grid_ver = ~0ull;
  // 벽 방향(10-05): slam 지도는 출발 자세 기준이라 벽이 지도 축에서 기울 수 있다(radio r3: 41°). 주된 방향 θ 를 가끔(kWallAngleEvery 초)
  // 재어 들고, |θ| > kAlignTol 이면 돌린 격자에서 뽑는다(wallSegmentsAtAngle — 전부 다시, kWallAlignedEvery 초에 한 번까지).
  // 축에 맞는 지도(gt 자세·축 맞은 출발)는 예전 증분 추출 그대로. 진단: SM_WALLS_AXIS=1 이면 옛 동작(축만)
  double wall_th = 0, wall_th_t = -1e300, walls_full_t = -1e300;
  // 단계별 시간(timing.hpp). 쓰기는 입력·스냅숏 스레드, 읽기는 sm_get_timing — 작은 잠금 하나
  std::mutex tmu;
  Timings tm;
  // 살아 있는 장면 그래프(sgraph.hpp) — mu 아래에서 고침
  SceneGraph graph;
  JsonCache jcache;                   // scene.json 노드 조각(저장 스레드 하나 — save_mu)
  std::mutex save_mu;
  std::unordered_map<uint32_t, std::string> obj_meta;
  std::string obj_kv;                 // sm_set_obj_params 로 준 매개변수(sm_set_robot 뒤에 다시 씀)
  // objprob(sm_set_object_model): 다음 영상의 검출 임베딩, 마지막 keyframe 의 통째 다시 담기 요청, 저장한 벡터 version
  std::vector<float> det_emb;
  int det_emb_n = -1, det_emb_dim = 0;
  std::vector<ReencReq> reenc;
  std::vector<sm_reenc_req> reenc_pub;
  std::vector<uint32_t> reenc_bits;
  std::unordered_map<uint32_t, uint32_t> saved_ap_ver;
  void addT(int st, double us) {
    std::lock_guard<std::mutex> g(tmu);
    tm.h[st].add(us);
  }
  explicit sm_ctx(const MapperParams& p) : params(p), mapper(p), om(oparams) {
    for (const char* n : kStructureNames) kind_names[SM_KIND_STRUCTURE].push_back(n);
    for (const char* n : kStaticNames) kind_names[SM_KIND_STATIC].push_back(n);
  }
  // labels × 표 → 종류(구조물이 먼저: "floor lamp" 는 lamp 로 끝나 고정, "floor" 는 구조물)
  void applyKinds() {
    kinds.assign(labels.size(), SM_KIND_OBJECT);
    for (size_t i = 0; i < labels.size(); ++i) {
      const std::string n = normName(labels[i]);
      for (int k : {SM_KIND_STRUCTURE, SM_KIND_STATIC})
        for (const std::string& e : kind_names[k])
          if (kinds[i] == SM_KIND_OBJECT && headMatch(n, e)) kinds[i] = uint8_t(k);
    }
    if (oparams.objprob)   // objprob: 문·창·계단·난간·기둥은 구조 물체(내보냄) — 벽·바닥·천장·걸레받이·칸막이·바깥은 그대로 구조물(지움)
      for (size_t i = 0; i < labels.size(); ++i) {
        if (kinds[i] != SM_KIND_STRUCTURE) continue;
        const std::string n = normName(labels[i]);
        for (const char* e : kStructObjNames) if (headMatch(n, e)) kinds[i] = SM_KIND_STRUCT_OBJ;
      }
    om.setClassKinds(kinds);
    std::vector<uint8_t> fl(labels.size(), 0);   // 바닥에 깔리는 것(바닥 조각 거르기에서 뺌)
    for (size_t i = 0; i < labels.size(); ++i) {
      const std::string n = normName(labels[i]);
      for (const char* e : kFloorLevelNames) fl[i] = fl[i] || headMatch(n, e);
    }
    om.setFloorClasses(std::move(fl));
  }
};

struct sm_snapshot_t {
  std::atomic<int> refs{1};
  sm_pose2 pose{};
  sm_status st{};
  double res = 0.05, ox = 0, oy = 0;
  int w = 0, h = 0;
  std::vector<sm_object> objs;
  std::shared_ptr<const std::vector<scenemap::WallSeg>> wallsp;   // 벽 선분(격자와 같은 시점)
  std::shared_ptr<const std::vector<int8_t>> cellsp;   // 격자(i8) — 바뀌지 않았으면 이전 스냅숏과 같은 배열
  const int8_t* cells() const { return cellsp ? cellsp->data() : nullptr; }
  std::vector<std::string> names;   // objs[i].name 이 가리키는 문자열(스냅숏 수명 동안)
  std::vector<BestViewPtr> views;   // objs[i] 의 best view(없으면 null)
  std::vector<uint8_t> movable;     // objs[i]: 1 = 옮길 수 있는 물체, 0 = 가구·가전·붙박이
  std::vector<ObjCloud> clouds;     // objs[i] 의 점 구름(점 배열은 공유)
  std::vector<sm_inspect> insp;     // objs[i] 의 살펴본 정도(sm_set_inspect 켰을 때만, 아니면 비어 있음)
  double voxel = 0.02;
  // reachable 용 부풀린 장애물(처음 부를 때 만듦)
  mutable std::once_flag inflate_once;
  mutable std::vector<uint8_t> blocked;
  // 방(rooms.hpp): 나눔(공유)·이 스냅숏 물체 배정·이름, ABI 배열
  std::shared_ptr<const RoomSeg> rseg;
  RoomNaming rnames;
  std::vector<sm_room> rooms;
  std::vector<sm_room_door> rdoors;
  std::shared_ptr<const GraphView> graph;   // 장면 그래프 사본
  // 마지막 가상 스캔(베이스 기준) + 그때 map 자세
  sm_pose2 scan_pose{};
  float scan_ox = 0, scan_oy = 0;
  std::vector<float> scan_hx, scan_hy, scan_fx, scan_fy;
};

namespace {

// 로봇별 팔 끝(베이스 기준)·그리퍼 값. 순기구학 grasp_point(E0 잡는 점) + omx_gripper_joint_1(두 칸 같은 값)
void robotHands(int robot, const float* q, float eef[2][3], float grip[2]) {
  (void)robot;
  LimoFk f;
  computeLimoFk(q, &f);
  for (int k = 0; k < 3; ++k) eef[0][k] = eef[1][k] = float(f.T_eef[k * 4 + 3]);
  grip[0] = grip[1] = q[11];
}

// 로봇별 순기구학 몸(스캔 가리기 캡슐) + T_head + 팔 끝
BodyState robotBody(int robot, const float* q, BodyFk* fk) {
  (void)robot;
  LimoFk f;
  computeLimoFk(q, &f);
  float eef[2][3];
  limoBodyFk(f, fk, eef);
  return bodyFromFk(*fk, eef, 0.05f, 0.06f, 0.f);   // OMX 링크 폭 ≈ 3–4 cm
}

// 로봇별 몸 크기 매개변수LIMO 0.32 × 0.22 × 0.25 m, 깊이 카메라 높이 0.18 m, OMX 팔 닿는 거리 ≈ 0.4 m
void robotParams(int robot, MapperParams* sp, ObjParams* op) {
  (void)robot;
  ScanParams& s = sp->scan;
  s.self_r = 0.22f;            // 몸통 반대각선 0.19 m + 여유(R1 0.55)
  s.eef_r = 0.08f;             // 팔 끝 둘레(R1 0.35)
  s.arm_r = 0.06f;             // 캡슐이 없을 때만 쓰는 어깨–팔 끝 선분
  for (int k = 0; k < 2; ++k) { s.shoulder[k][0] = -0.05f; s.shoulder[k][1] = 0.f; s.shoulder[k][2] = 0.25f; }   // omx_joint2 근처
  s.band_lo = 0.05f;           // 5 cm 넘는 턱이면 못 넘음(R1 0.10)
  s.band_hi = 0.50f;           // 팔 접은 키 ≈ 0.35 m: 탁자 상판(≈ 0.7 m) 밑으로는 지나감(R1 1.80)
  sp->attach.radius = 0.6f;    // 로봇에 붙어 같이 움직이는 것 판정 반경(R1 1.3)
  op->hand_r = 0.10f;          // 손에 든 것 거르기(R1 0.40)
  op->grasp_r = 0.12f;         // 그리퍼가 닫힐 때 이 안 물체를 듦(R1 0.25)
  op->cloud_hand_r = 0.05;
  op->body_r = 0.22;
  op->n_hands = 1;
  // 잡기 확인(10-04): 시뮬 한 판에서 팔이 탁자 앞을 가린 채 그리퍼가 빈손으로 끝까지 닫히자 탁자(1.2 m, 고정 종류)를 'held' 로 들고
  // 0.49 m 옮겼다 — 옛 규칙은 '닫히는 순간 팔 끝 grasp_r 안 가장 가까운 확정 물체' 뿐이었다. 이제 닫힌 채 멈춘 뒤(0.2 s) 한 번 고르고,
  // 큰 것·고정 종류·가장 좁은 변 > 0.06 m 는 못 들고, 손끝 틈이 비지 않았고(≥ 5 mm) 물체 폭과 맞아야(± 2.5 cm) 든다.
  // 틈 표(omx_gripper_joint_1 rad → 잡는 점 손끝 틈 m): E0(robot-agent src/robot/og/e0/results) 쥔 각도 width_height_kp1e6·verify_eef_kp1e6
  // (폭 1·2·3·4 cm 를 쥐면 0.095·0.231·0.347·0.408 rad), 그 위는 finger_gap_hull 의 link5 x 0.08 틈(30° 55 mm, 45° 93 mm).
  // 닫힘 문턱 0.6 rad(틈 ≈ 6.6 cm): 6 cm 물체를 쥐어도 '닫힘'(옛 0.35 rad 는 4 cm 를 쥔 0.41 rad 를 닫힘으로 못 봄)
  op->grip_closed = 0.6f;      // 0 = 완전히 닫힘, 1.745 = 다 열림
  op->grasp_check = true;
  op->grip_gap = {{0.0, 0.0}, {0.095, 0.01}, {0.231, 0.02}, {0.347, 0.03}, {0.408, 0.04}, {0.5236, 0.0551}, {0.7854, 0.0933}};
}

// 베이스 기준 점 → map (slam 자세)
void toMap(const Pose2& P, const float b[2][3], double m[2][3]) {
  const double c = std::cos(P.th), s = std::sin(P.th);
  for (int k = 0; k < 2; ++k) {
    m[k][0] = P.x + c * b[k][0] - s * b[k][1];
    m[k][1] = P.y + s * b[k][0] + c * b[k][1];
    m[k][2] = b[k][2];
  }
}

// 적분 한 표본(데이터: proprio i 의 base_qvel 이 i-1 → i 구간 속도). 제자리 잡음은 Mapper2D 와 같은 규칙.
// 외부 자세 중 stamp 이하 가장 최근 것(max_age 안). 없으면 false
bool poseAt(const std::deque<sm_pose2>& q, double stamp, Pose2* out, double max_age) {
  for (auto it = q.rbegin(); it != q.rend(); ++it) {
    if (it->stamp > stamp + 1e-6) continue;
    if (stamp - it->stamp > max_age) return false;
    *out = Pose2{it->x, it->y, it->yaw};
    return true;
  }
  return false;
}
bool gtAt(const sm_ctx* c, double stamp, Pose2* out, double max_age = 0.1) { return poseAt(c->gtq, stamp, out, max_age); }
bool extAt(const sm_ctx* c, double stamp, Pose2* out, double max_age = 0.1) { return poseAt(c->extq, stamp, out, max_age); }

// 격자에서 바뀐 영역만 스트림으로(잠금 안, 스텝 스레드). 모양·원점이 바뀌면 전체. 보내지 못하면(링이 가득 참) 다음에 전체를 다시 보낸다
void streamMap(sm_ctx* c) {
  if (!c->stream.running()) return;
  OccGrid& gr = c->mapper.gridMut();
  const int w = gr.width(), h = gr.height();
  if (w <= 0 || h <= 0) return;
  int d0, d1, d2, d3;
  const bool dirty = gr.takeDirty(&d0, &d1, &d2, &d3, 3);
  const bool reshaped = w != c->st_w || h != c->st_h || gr.x0() != c->st_x0 || gr.y0() != c->st_y0;
  int lx0, ly0, lx1, ly1;
  if (reshaped) { lx0 = 0; ly0 = 0; lx1 = w - 1; ly1 = h - 1; }
  else if (dirty) {
    lx0 = std::max(0, d0 - gr.x0()); ly0 = std::max(0, d1 - gr.y0());
    lx1 = std::min(w - 1, d2 - gr.x0()); ly1 = std::min(h - 1, d3 - gr.y0());
    if (lx0 > lx1 || ly0 > ly1) return;
  } else return;
  c->st_scratch.resize(size_t(lx1 - lx0 + 1) * size_t(ly1 - ly0 + 1));
  gr.exportRect(lx0, ly0, lx1, ly1, c->st_scratch.data());
  const bool ok = c->stream.pushMapRect(w, h, double(gr.res()), gr.x0() * double(gr.res()), gr.y0() * double(gr.res()), lx0, ly0, lx1, ly1,
                                        c->st_scratch.data());
  if (ok) { c->st_w = w; c->st_h = h; c->st_x0 = gr.x0(); c->st_y0 = gr.y0(); }
  else c->st_w = 0;   // 놓쳤다: 다음에 전체
}

void integrate(sm_ctx* c, const Prop& p) {
  if (c->have_used) {
    const double dt = p.stamp - c->last_used.stamp;
    if (dt > 0 && dt < 1.0) c->mapper.pushVelocity(p.v[0], p.v[1], p.v[2], dt);
  }
  const auto t0 = TClock::now();
  c->last_used = p;
  c->have_used = true;
  Pose2 g;
  // 같은 stamp 의 외부 자세가 있을 때만(없는 스텝은 적분으로 이어감 — 접착부는 영상 짝 스텝에만 넣어도 됨)
  if (c->pose_mode == SM_POSE_GT && gtAt(c, p.stamp, &g, 1e-4)) c->mapper.setPose(g);
  if (c->pose_mode == SM_POSE_EXT && extAt(c, p.stamp, &g, 1e-4)) c->mapper.setPose(g);
  // 영상이 없는 스텝에도 든 물체가 손을 따라가게
  float eef[2][3], grip[2];
  robotHands(c->robot, p.q, eef, grip);
  double eefm[2][3];
  const Pose2 P = c->mapper.pose();
  toMap(P, eef, eefm);
  c->om.updateHands(p.stamp, eefm, grip, P.th);
  if (c->pose_mode != SM_POSE_GT && c->stream.running()) c->stream.pushPose(p.stamp, P.x, P.y, P.th);
  c->addT(kStIntegrate, usBetween(t0, TClock::now()));
}

Pose2 preview(const sm_ctx* c, double* stamp) {
  if (c->pose_mode == SM_POSE_GT && !c->gtq.empty()) {   // 정답 자세: 가장 최근 것 그대로
    const sm_pose2& g = c->gtq.back();
    *stamp = g.stamp;
    return Pose2{g.x, g.y, g.yaw};
  }
  Pose2 q = c->mapper.pose();
  double t = c->have_used ? c->last_used.stamp : 0;
  for (const Prop& p : c->pending) {
    const double dt = p.stamp - t;
    t = p.stamp;
    if (!(dt > 0 && dt < 1.0)) continue;
    double vx = p.v[0], vy = p.v[1], wz = p.v[2];
    if (c->params.deadband && std::hypot(vx, vy) < c->params.still_v && std::fabs(wz) < c->params.still_w) continue;
    const double cs = std::cos(q.th), sn = std::sin(q.th);
    q.x += (cs * vx - sn * vy) * dt;
    q.y += (sn * vx + cs * vy) * dt;
    q.th += wz * dt;
  }
  *stamp = t;
  return q;
}

// 방: 주기·변화가 되면 다시 나누고(잠금 밖), 이 스냅숏 물체를 방에 배정·이름
void snapRooms(sm_ctx* c, sm_snapshot_t* s) {
  if (!s->w || !s->h) return;
  GridView g{s->cells(), s->w, s->h, s->res, int(std::lround(s->ox / s->res)), int(std::lround(s->oy / s->res))};
  s->rseg = c->rooms.update(g, s->pose.stamp, c->rooms_force.exchange(false));
  if (!s->rseg) return;
  std::vector<RoomObj> ro(s->objs.size());
  for (size_t i = 0; i < s->objs.size(); ++i) {
    ro[i].id = s->objs[i].id;
    ro[i].name = s->names[i];
    for (int k = 0; k < 3; ++k) { ro[i].pos[k] = s->objs[i].pos[k]; ro[i].ext[k] = s->objs[i].extent[k]; }
    ro[i].movable = s->movable[i];
  }
  const auto ov = c->rooms.overrides();
  s->rnames = nameRooms(*s->rseg, ro, c->rooms.params(), &ov);
  const RoomSeg& R = *s->rseg;
  s->rooms.resize(R.rooms.size());
  for (size_t k = 0; k < R.rooms.size(); ++k) {
    const RoomGeom& r = R.rooms[k];
    const RoomLabel& L = s->rnames.rooms[k];
    sm_room& o = s->rooms[k];
    o = sm_room{};
    o.id = r.id;
    o.name = L.name.c_str();
    o.type = L.type.c_str();
    o.name_conf = L.conf;
    for (int j = 0; j < 2; ++j) { o.centroid[j] = r.centroid[j]; o.bbox_min[j] = r.bmin[j]; o.bbox_max[j] = r.bmax[j]; }
    o.area_m2 = r.area_m2;
    o.n_objects = int32_t(L.objects.size());
    o.objects = L.objects.empty() ? nullptr : L.objects.data();
  }
  for (const RoomDoor& d : R.doors) s->rdoors.push_back(sm_room_door{d.a, d.b, {d.pos[0], d.pos[1]}, d.width});
}

}  // namespace

extern "C" {

namespace {
// config_json 의 작은 읽기: "key": "문자열" / "key": 숫자. 없으면 false(JSON 전체 파서는 두지 않는다 — 키가 몇 개뿐)
bool cfgString(const std::string& j, const char* key, std::string* out) {
  const size_t k = j.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) return false;
  size_t p = j.find(':', k);
  if (p == std::string::npos) return false;
  p = j.find_first_not_of(" \t\r\n", p + 1);
  if (p == std::string::npos || j[p] != '"') return false;
  const size_t e = j.find('"', p + 1);
  if (e == std::string::npos) return false;
  *out = j.substr(p + 1, e - p - 1);
  return true;
}
bool cfgNumber(const std::string& j, const char* key, double* out) {
  const size_t k = j.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) return false;
  const size_t p = j.find(':', k);
  if (p == std::string::npos) return false;
  char* end = nullptr;
  const double v = std::strtod(j.c_str() + p + 1, &end);
  if (end == j.c_str() + p + 1) return false;
  *out = v;
  return true;
}
}  // namespace

sm_ctx* sm_create(const char* config_json) {
  sm_ctx* c = new sm_ctx(MapperParams{});
  sm_set_robot(c, SM_ROBOT_LIMO_OMX);            // 로봇은 LIMO + OMX-F 하나
  if (!config_json || !*config_json) return c;
  const std::string j(config_json);
  std::string r;
  if (cfgString(j, "robot", &r)) {
    const int k = r == "limo_omx" || r == "limo" ? SM_ROBOT_LIMO_OMX : -1;
    if (k < 0 || sm_set_robot(c, k) != 0) { delete c; return nullptr; }
  }
  std::string od;
  if (cfgString(j, "odom", &od)) {
    if (od != "pose" && od != "twist") { delete c; return nullptr; }
    c->odom_pose = od == "pose";
  }
  double gc;
  if (cfgNumber(j, "grip_closed", &gc)) {
    c->oparams.grip_closed = float(gc);
    sm_reset(c);
  }
  double ins;
  if (cfgNumber(j, "inspect", &ins)) sm_set_inspect(c, ins != 0);
  return c;
}

int sm_set_robot(sm_ctx* c, int32_t robot) {
  if (!c || robot != SM_ROBOT_LIMO_OMX) return -1;
  {
    std::lock_guard<std::mutex> g(c->mu);
    c->robot = robot;
    MapperParams sp{};
    ObjParams op{};
    op.voxel = c->oparams.voxel;          // sm_set_cloud_params 로 바꾼 값은 둔다
    op.cloud_cap = c->oparams.cloud_cap;
    op.objprob = c->oparams.objprob;        // sm_set_object_model 도 둔다
    op.insp = c->oparams.insp;              // sm_set_inspect 도 둔다
    robotParams(robot, &sp, &op);
    ObjectMap::applyParams(&op, c->obj_kv.c_str());   // sm_set_obj_params 도 둔다
    c->params = sp;
    c->oparams = op;
  }
  return sm_reset(c);
}

int sm_get_robot(sm_ctx* c) { return c ? c->robot : -1; }

int sm_proprio_dim(int32_t robot) {
  return robot == SM_ROBOT_LIMO_OMX ? SM_LIMO_PROPRIO_DIM : -1;
}

int sm_robot_fk(int32_t robot, const float* q, int32_t n, sm_body_fk* o) {
  if (!q || !o || sm_proprio_dim(robot) < 0 || n < sm_proprio_dim(robot)) return -1;
  std::memset(o, 0, sizeof(*o));
  float eef[2][3], grip[2];
  robotHands(robot, q, eef, grip);
  LimoFk f;
  computeLimoFk(q, &f);
  o->n_cams = 2;
  o->n_hands = 1;
  o->cam_valid[0] = o->cam_valid[1] = 1;
  std::memcpy(o->T_cam[0], f.T_depth, sizeof(f.T_depth));
  std::memcpy(o->T_cam[1], f.T_wrist, sizeof(f.T_wrist));
  std::memcpy(o->T_eef[0], f.T_eef, sizeof(f.T_eef));
  o->eef_valid[0] = 1;
  o->grip[0] = grip[0];
  return 0;
}

void sm_destroy(sm_ctx* c) {
  if (c && c->slog) std::fclose(c->slog);
  delete c;
}

int sm_set_labels(sm_ctx* c, const char* const* names, int n) {
  if (!c || n < 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->labels.clear();
  for (int i = 0; i < n; ++i) c->labels.emplace_back(names && names[i] ? names[i] : "");
  c->applyKinds();
  return 0;
}

int sm_set_kind_names(sm_ctx* c, int32_t kind, const char* const* names, int32_t n) {
  if (!c || (kind != SM_KIND_STRUCTURE && kind != SM_KIND_STATIC)) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  std::vector<std::string>& v = c->kind_names[kind];
  v.clear();
  if (!names) {   // 기본 표로
    if (kind == SM_KIND_STRUCTURE)
      for (const char* e : kStructureNames) v.push_back(e);
    else
      for (const char* e : kStaticNames) v.push_back(e);
  } else {
    for (int i = 0; i < n; ++i)
      if (names[i]) v.push_back(normName(names[i]));
  }
  c->applyKinds();
  return 0;
}

int sm_reset(sm_ctx* c) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->mapper = Mapper2D(c->params);
  {
    ApText tm = c->om.textModel();     // objprob 글 모델은 판이 바뀌어도 그대로
    c->om = ObjectMap(c->oparams);
    c->om.setTextModel(std::move(tm));
  }
  c->applyKinds();   // 종류·바닥에 깔리는 이름 표
  c->det_emb_n = -1;
  c->reenc.clear(); c->reenc_pub.clear(); c->reenc_bits.clear();
  c->saved_ap_ver.clear();
  c->handled.clear();
  c->pending.clear();
  c->have_used = false;
  c->have_odom = false;
  c->st = sm_status{};
  c->views.clear();
  c->last_assoc.clear();
  c->last_view_upd.clear();
  c->last_view_q.clear();
  c->ev_seen = 0;
  c->epoch++;
  c->saved_ver.clear();
  c->saved_cloud_ver.clear();
  c->clean_objects = true;
  c->rooms.reset();
  c->gtq.clear();
  c->extq.clear();
  c->diag_align = false;
  c->diag = sm_pose_diag{};
  c->diag_s2xy = c->diag_s2yaw = 0;
  if (c->slog) std::fclose(c->slog);
  c->slog = nullptr;
  if (const char* lp = std::getenv("SM_SLAM_LOG")) {
    c->slog = std::fopen(lp, "w");
    if (c->slog) std::fprintf(c->slog, "stamp,x,y,th,ref_x,ref_y,ref_th,n_hits,inserted,odom_xy,odom_yaw\n");
  }
  c->grid8.reset();
  c->grid8_ver = ~0ull;
  c->walls.reset();
  c->wallsp.reset();
  c->walls_rects.clear();
  c->walls_grid_ver = ~0ull;
  c->wall_th = 0; c->wall_th_t = -1e300; c->walls_full_t = -1e300;
  c->graph.reset();
  c->obj_meta.clear();
  c->mapper.setUpdatePolicy(c->map_policy, c->still_every);
  return 0;
}

int sm_push_proprio(sm_ctx* c, const sm_proprio* p) {
  if (!c || !p || !p->proprio || p->n_proprio < proprioDim(c->robot)) return -1;
  const auto t0 = TClock::now();
  std::lock_guard<std::mutex> g(c->mu);
  Prop e;
  e.stamp = p->stamp;
  {
    std::memset(e.q, 0, sizeof(e.q));
    std::memcpy(e.q, p->proprio, sizeof(float) * size_t(std::min(p->n_proprio, kMaxProprioDim)));
    const float* q = p->proprio;
    // 적분 속도: 오도메트리 자세 차(지난 proprio 의 베이스 기준) / dt — 같은 Euler 적분으로 오도메트리 이동을 그대로 되살림.
    // 첫 표본·간격 이상(dt ≤ 0, ≥ 1 s)·odom "twist" 이면 proprio 의 twist
    const double dt = p->stamp - c->odom_prev_stamp;
    if (c->odom_pose && c->have_odom && dt > 0 && dt < 1.0) {
      const double dx = q[SM_LIMO_ODOM_X] - c->odom_prev[0], dy = q[SM_LIMO_ODOM_Y] - c->odom_prev[1];
      const double cy = std::cos(c->odom_prev[2]), sy = std::sin(c->odom_prev[2]);
      e.v[0] = (cy * dx + sy * dy) / dt;
      e.v[1] = (-sy * dx + cy * dy) / dt;
      e.v[2] = wrapAngle(double(q[SM_LIMO_ODOM_YAW]) - c->odom_prev[2]) / dt;
    } else {
      e.v[0] = q[SM_LIMO_VX]; e.v[1] = q[SM_LIMO_VY]; e.v[2] = q[SM_LIMO_WZ];
    }
    c->odom_prev[0] = q[SM_LIMO_ODOM_X]; c->odom_prev[1] = q[SM_LIMO_ODOM_Y]; c->odom_prev[2] = q[SM_LIMO_ODOM_YAW];
    c->odom_prev_stamp = p->stamp;
    c->have_odom = true;
  }
  c->pending.push_back(e);
  if (c->stream.running()) {
    {   // 뷰어 robot.json joint_order: omx_joint1..5, gripper_1, gripper_2(= −1), (바퀴 fl, fr, rl, rr)
      const float* q = p->proprio;
      float j[11] = {q[6], q[7], q[8], q[9], q[10], q[11], -q[11], 0, 0, 0, 0};
      int n = 7;
      if (p->n_proprio >= SM_LIMO_PROPRIO_DIM + 4) { for (int k = 0; k < 4; ++k) j[7 + k] = q[SM_LIMO_WHEEL_FL + k]; n = 11; }
      c->stream.pushJoints(p->stamp, j, n);
    }
  }
  // 영상이 오지 않아도 쌓이지 않게: 최신보다 0.5 s 넘게 오래된 것은 적분해 둔다(영상 stamp 는 최신 − 1 스텝)
  while (c->pending.size() > 1 && c->pending.front().stamp < p->stamp - 0.5) {
    integrate(c, c->pending.front());
    c->pending.pop_front();
  }
  c->st.last_proprio_stamp = p->stamp;
  c->st.n_proprio++;
  c->addT(kStPushProprio, usBetween(t0, TClock::now()));
  return 0;
}

int sm_push_image(sm_ctx* c, const sm_image* im, const sm_detections* dets) { return sm_push_image_ex(c, im, dets, nullptr, nullptr); }

namespace {
// best view 를 바꿀 검출 하나(잠금 안에서 정하고, 잠금 밖에서 자름)
struct ViewCand {
  uint32_t id;
  int det;                     // 검출 번호(sm_last_views)
  double q;
  sm_crop_req req;
  std::shared_ptr<BestView> v;
};
}  // namespace

int sm_push_image_ex(sm_ctx* c, const sm_image* im, const sm_detections* dets, sm_crop_fn crop, void* user) {
  const sm_rgb_source src{crop, nullptr, user};
  return sm_push_image_rgb(c, im, dets, &src);
}

namespace {
// 스냅숏·그래프가 쓰는 격자 사본(보이는 값이 바뀌었을 때만 새로). mu 아래
bool refreshGrid8(sm_ctx* c) {
  const OccGrid& gr = c->mapper.grid();
  if (c->grid8 && c->grid8_ver == gr.cellsVersion() && c->grid8_w == gr.width() && c->grid8_h == gr.height()) return false;
  const auto tg = TClock::now();
  auto v = std::make_shared<std::vector<int8_t>>(size_t(gr.width()) * gr.height());
  gr.export8(v->data());
  c->grid8 = std::move(v);
  c->grid8_ver = gr.cellsVersion();
  c->grid8_w = gr.width();
  c->grid8_h = gr.height();
  c->addT(kStSnapGrid, usBetween(tg, TClock::now()));
  return true;
}

// 벽 선분(walls.hpp): 격자가 바뀌었거나 바닥 가구 영역이 바뀌었을 때만, 바뀐 행만 다시 계산. 스냅숏은 결과를 나눠 쓴다. mu 아래
// 벽 추출에서 빼는 영역 = 바닥에 놓인 확정 물체(소파·탁자 …)의 바닥 면적 + 0.1 m. 벽에 걸린 것(액자·조명, 바닥이 0.4 m 위)은 안 뺀다.
void refreshWalls(sm_ctx* c) {
  const OccGrid& gr = c->mapper.grid();
  const bool grid_changed = c->walls_grid_ver != c->grid8_ver;   // 다른 곳(그래프 갱신)이 격자 사본을 먼저 새로 만들었어도 놓치지 않게 버전으로 본다
  std::vector<scenemap::WallRect> rects;
  for (const MapObject& o : c->om.objects()) {
    if (!c->om.exportable(o) || o.held_by >= 0 || o.state == SM_GONE) continue;
    if (o.lo[2] > 0.4) continue;
    if (o.ap && o.ap->wall_like) continue;   // objprob: 벽 선 위 세운 얇은 평면(벽 조각) — 벽을 지우지 않음
    if (c->om.kindOf(o.cls) == SM_KIND_STRUCT_OBJ) continue;   // 문·창·계단은 벽 선 자리(지우면 벽이 끊김)
    const double dx = o.hi[0] - o.lo[0], dy = o.hi[1] - o.lo[1];
    if (dx > 5.0 || dy > 5.0) continue;   // 방 크기 덩어리는 물체가 아니라 벽·바닥 오인식
    rects.push_back({o.lo[0] - 0.1, o.lo[1] - 0.1, o.hi[0] + 0.1, o.hi[1] + 0.1});
  }
  const bool rects_changed = !(rects == c->walls_rects);
  const double now = c->st.last_image_stamp;
  static const bool axis_only = std::getenv("SM_WALLS_AXIS") != nullptr;
  bool th_changed = false;
  if (!axis_only && c->grid8 && (grid_changed || rects_changed) && now - c->wall_th_t >= kWallAngleEvery) {
    const scenemap::WallGrid wg{c->grid8->data(), gr.width(), gr.height(), double(gr.res()), gr.x0() * double(gr.res()), gr.y0() * double(gr.res())};
    const double th = scenemap::wallAngle(wg, &rects);
    c->wall_th_t = now;
    // 흔들림 막기: 1° 넘게 바뀔 때만 새 θ(축 ↔ 기울어짐이 바뀌면 증분 추출기를 비움)
    if (std::abs(th - c->wall_th) > kWallAngleHyst) {
      const bool was_axis = std::abs(c->wall_th) <= scenemap::kAlignTol, is_axis = std::abs(th) <= scenemap::kAlignTol;
      if (was_axis != is_axis) { c->walls.reset(); c->wallsp.reset(); }
      c->wall_th = th;
      th_changed = true;
    }
  }
  if (std::abs(c->wall_th) > scenemap::kAlignTol) {
    if (c->wallsp && !th_changed && !rects_changed && (!grid_changed || now - c->walls_full_t < kWallAlignedEvery)) return;
    const scenemap::WallGrid wg{c->grid8->data(), gr.width(), gr.height(), double(gr.res()), gr.x0() * double(gr.res()), gr.y0() * double(gr.res())};
    c->wallsp = std::make_shared<const std::vector<scenemap::WallSeg>>(
        scenemap::wallSegmentsAtAngle(wg, c->wall_th, scenemap::kMinLen, scenemap::kMaxThick, 0.6, &rects));
    c->walls_rects = std::move(rects);
    c->walls_x0 = gr.x0(); c->walls_y0 = gr.y0();
    c->walls_grid_ver = c->grid8_ver;
    c->walls_full_t = now;
    return;
  }
  const bool same = c->wallsp && c->walls_x0 == gr.x0() && c->walls_y0 == gr.y0();
  if (same && !grid_changed && !rects_changed) return;
  int dx0, dy0, dx1, dy1;
  const bool d = c->mapper.gridMut().takeDirty(&dx0, &dy0, &dx1, &dy1, 2);
  scenemap::WallGrid wg{c->grid8->data(), gr.width(), gr.height(), double(gr.res()), gr.x0() * double(gr.res()), gr.y0() * double(gr.res())};
  int ylo = 0, yhi = -1;   // 전부
  if (same && d) { ylo = dy0 - gr.y0(); yhi = dy1 - gr.y0(); }
  else if (same) { ylo = 1; yhi = 0; }   // 바뀐 칸 없음 — 영역만 바뀌었으면 update 가 전부 다시
  c->wallsp = std::make_shared<const std::vector<scenemap::WallSeg>>(c->walls.update(wg, ylo, yhi, &rects));
  c->walls_rects = std::move(rects);
  c->walls_x0 = gr.x0(); c->walls_y0 = gr.y0();
  c->walls_grid_ver = c->grid8_ver;
  static const bool check = std::getenv("SM_WALLS_CHECK") != nullptr;   // 진단: 증분 결과를 처음부터 계산한 것과 비교
  if (check) {
    const auto full = scenemap::wallSegments(wg, scenemap::kMinLen, scenemap::kMaxThick, 0.6, &c->walls_rects);
    const auto& inc = *c->wallsp;
    bool eq = full.size() == inc.size();
    for (size_t i = 0; eq && i < full.size(); ++i)
      eq = full[i].ax == inc[i].ax && full[i].ay == inc[i].ay && full[i].bx == inc[i].bx && full[i].by == inc[i].by;
    static int n = 0, bad = 0;
    ++n; if (!eq) ++bad;
    if (!eq || n % 50 == 0) std::fprintf(stderr, "[walls-check] updates %d mismatches %d (segments %zu vs %zu, ignore %zu)\n", n, bad, inc.size(), full.size(), c->walls_rects.size());
  }
}

// keyframe 뒤 장면 그래프: agent, (검출이 있었으면) 물체, 주기마다 바뀐 격자 둘레 place. mu 아래
void updateGraph(sm_ctx* c, double stamp, bool objects) {
  const auto t0 = TClock::now();
  // GT mode: the trajectory comes straight from the pushed GT poses (sm_push_pose), not from the SLAM/odometry pose sampled at keyframes.
  if (c->pose_mode != SM_POSE_GT) c->graph.updateAgent(stamp, c->mapper.pose());
  if (objects) {
    std::vector<ObjIn> v;
    for (const MapObject& o : c->om.objects()) {
      if (!c->om.exportable(o)) continue;
      ObjIn q;
      q.id = o.id;
      q.name = o.cls >= 0 && size_t(o.cls) < c->labels.size() ? c->labels[o.cls] : std::string("?");
      for (int k = 0; k < 3; ++k) { q.pos[k] = o.pos[k]; q.lo[k] = o.lo[k]; q.hi[k] = o.hi[k]; }
      q.state = o.held_by >= 0 ? SM_HELD : o.state;
      q.movable = c->om.kindOf(o.cls) != SM_KIND_STATIC && c->om.kindOf(o.cls) != SM_KIND_STRUCT_OBJ;
      v.push_back(std::move(q));
    }
    c->graph.updateObjects(v);
  }
  const auto t1 = TClock::now();
  c->addT(kStGraphObj, usBetween(t0, t1));
  OccGrid& gr = c->mapper.gridMut();
  int b[4];
  const bool d = gr.takeDirty(&b[0], &b[1], &b[2], &b[3], 1);
  if (gr.width() > 0 && c->graph.placesDue(stamp, d, gr.width(), gr.height(), gr.x0(), gr.y0())) {
    refreshGrid8(c);
    c->graph.updatePlaces(c->grid8->data(), gr.width(), gr.height(), gr.x0(), gr.y0(), gr.res(), stamp, d ? b : nullptr, false);
    c->addT(kStGraphPlaces, usBetween(t1, TClock::now()));
  } else if (d) {
    c->graph.updatePlaces(nullptr, 0, 0, 0, 0, gr.res(), stamp, b, false);   // 상자만 쌓아 둠
  }
}

// 진단: 외부 자세(정답)가 있으면 지금 자세와 비교(첫 keyframe 에서 맞춤 — 그 뒤 떠밀림)
void poseDiag(sm_ctx* c, double stamp) {
  Pose2 ref;
  if (c->pose_mode == SM_POSE_GT || !gtAt(c, stamp, &ref)) return;
  const Pose2 est = c->mapper.pose();
  if (!c->diag_align) {   // align = est ∘ ref⁻¹
    const double cs = std::cos(ref.th), sn = std::sin(ref.th);
    const Pose2 inv{-(cs * ref.x + sn * ref.y), -(-sn * ref.x + cs * ref.y), -ref.th};
    c->align = compose(est, inv);
    c->diag_align = true;
  }
  const Pose2 r = compose(c->align, ref);
  const double exy = std::hypot(est.x - r.x, est.y - r.y), eyaw = std::fabs(wrapAngle(est.th - r.th));
  sm_pose_diag& d = c->diag;
  d.n++;
  d.last_xy = exy;
  d.last_yaw = eyaw;
  d.max_xy = std::max(d.max_xy, exy);
  d.max_yaw = std::max(d.max_yaw, eyaw);
  c->diag_s2xy += exy * exy;
  c->diag_s2yaw += eyaw * eyaw;
  d.rms_xy = std::sqrt(c->diag_s2xy / d.n);
  d.rms_yaw = std::sqrt(c->diag_s2yaw / d.n);
  d.est[0] = est.x; d.est[1] = est.y; d.est[2] = est.th;
  d.ref[0] = r.x; d.ref[1] = r.y; d.ref[2] = r.th;
  d.stamp = stamp;
}
// SM_SLAM_LOG 한 줄: stamp, 넣은 자세 x y th, 정답(맞춤) x y th, 점 수, 넣음, 적분 이동
void slamLog(sm_ctx* c, double stamp) {
  const KeyframeStats& k = c->last_kf;
  Pose2 ref{NAN, NAN, NAN}, g;
  if (c->diag_align && gtAt(c, stamp, &g)) ref = compose(c->align, g);
  std::fprintf(c->slog, "%.4f,%.5f,%.5f,%.6f,%.5f,%.5f,%.6f,%d,%d,%.5f,%.6f\n", stamp, k.pose.x, k.pose.y, k.pose.th, ref.x, ref.y, ref.th,
               k.n_hits, int(k.inserted), k.odom_xy, k.odom_yaw);
}
}  // namespace

int sm_push_image_rgb(sm_ctx* c, const sm_image* im, const sm_detections* dets, const sm_rgb_source* src) {
  if (!c || !im || im->w <= 0 || im->h <= 0) return -1;
  const auto t_img = TClock::now();
  struct Tot {
    sm_ctx* c; TClock::time_point t; bool on = false;
    ~Tot() { if (on) c->addT(kStImage, usBetween(t, TClock::now())); }
  } tot{c, t_img};
  const sm_crop_fn crop = src ? src->crop : nullptr;
  const sm_gather_fn gather = src ? src->gather : nullptr;
  void* user = src ? src->user : nullptr;
  std::vector<ViewCand> cand;
  std::vector<ObsPoints> pts;
  bool host_rgb = false;
  uint64_t epoch = 0;
  ObjFrame F;
  bool ap_frame = false;              // objprob keyframe(검출 있음): 끝에서 통째 다시 담기 요청을 만듦
  std::vector<double> wsegs;
  {
  std::lock_guard<std::mutex> g(c->mu);
  c->st.last_image_stamp = im->stamp;
  c->st.n_images++;
  if (im->cam != 0 || !im->depth_m) return 0;   // 격자는 cam 0 깊이만(R1 머리, LIMO 몸통 앞 깊이 카메라)
  tot.on = true;
  // 영상 stamp 까지 적분(같은 stamp 의 proprio 가 이 영상의 짝)
  const auto tp = TClock::now();
  const double eps = 1e-6;
  while (!c->pending.empty() && c->pending.front().stamp <= im->stamp + eps) {
    integrate(c, c->pending.front());
    c->pending.pop_front();
  }
  if (!c->have_used) return 0;                  // 짝지을 proprio 가 아직 없음
  Pose2 known;
  // 자세: GT = 정답, EXT = 외부 SLAM(Cartographer) — 둘 다 영상 시각의 것이 없으면 지난 외부 자세 + 적분. ODOM = 적분만
  const bool have_known = (c->pose_mode == SM_POSE_GT && gtAt(c, im->stamp, &known)) || (c->pose_mode == SM_POSE_EXT && extAt(c, im->stamp, &known));
  if (!have_known) known = c->mapper.pose();
  const auto tf = TClock::now();
  c->addT(kStPair, usBetween(tp, tf));
  BodyFk fk;
  const BodyState body = robotBody(c->robot, c->last_used.q, &fk);
  if (c->ext_cam) std::memcpy(fk.T_head, c->ext_T, sizeof(fk.T_head));
  c->addT(kStFk, usBetween(tf, TClock::now()));
  DepthView dv;
  dv.w = im->w;
  dv.h = im->h;
  dv.m = im->depth_m;
  dv.step = std::max(1, int(std::lround(im->w / 160.0)));   // 채점과 같은 표본 밀도(가로 160 점 안팎)
  dv.fx = float(im->fx); dv.fy = float(im->fy); dv.cx = float(im->cx); dv.cy = float(im->cy);
  std::memcpy(dv.T_bc, fk.T_head, sizeof(dv.T_bc));
  {
    std::lock_guard<std::mutex> tg(c->tmu);   // 단계 시간은 mapper2d 가 직접 더함
    c->last_kf = c->mapper.keyframe(dv, body, known, &c->tm);
  }
  poseDiag(c, im->stamp);
  if (c->slog) slamLog(c, im->stamp);
  streamMap(c);
  c->reenc.clear();
  c->reenc_pub.clear();
  c->reenc_bits.clear();
  if (!dets) { c->last_assoc.clear(); c->last_view_upd.clear(); c->last_view_q.clear(); updateGraph(c, im->stamp, false); c->det_emb_n = -1; return 0; }   // 검출 없음: 격자만
  // 물체 지도: map ← 카메라 = 지도 자세 ∘ 순기구학 머리 카메라
  const Pose2 P = c->mapper.pose();
  const double cs = std::cos(P.th), sn = std::sin(P.th);
  F.stamp = im->stamp;
  F.w = im->w;
  F.h = im->h;
  F.depth_m = im->depth_m;
  F.fx = float(im->fx); F.fy = float(im->fy); F.cx = float(im->cx); F.cy = float(im->cy);
  const float* B = fk.T_head;
  for (int r = 0; r < 3; ++r) {
    const double R0 = r == 0 ? cs : (r == 1 ? sn : 0), R1 = r == 0 ? -sn : (r == 1 ? cs : 0), R2 = r == 2 ? 1 : 0;
    for (int k = 0; k < 4; ++k) F.T_mc[r * 4 + k] = R0 * B[k] + R1 * B[4 + k] + R2 * B[8 + k];
  }
  F.T_mc[3] += P.x;
  F.T_mc[7] += P.y;
  F.dets = dets;
  float eefb[2][3], grip[2];
  robotHands(c->robot, c->last_used.q, eefb, grip);
  toMap(P, eefb, F.eef);
  F.grip[0] = grip[0];
  F.grip[1] = grip[1];
  F.base_yaw = P.th;
  F.base_xy[0] = P.x;
  F.base_xy[1] = P.y;
  // LIMO: 팔 캡슐(순기구학, 스캔 몸 가리기와 같은 것)을 map 으로 — 팔이 몸통 카메라 앞을 가릴 때 그 화소가 물체 점이 안 되게. R1 은 안 씀
  std::vector<Capsule> self_caps;
  if (c->robot == kRobotLimoOmx) {
    self_caps.reserve(body.caps.size());
    for (const Capsule& k : body.caps) {
      Capsule m = k;
      m.a[0] = float(P.x + cs * k.a[0] - sn * k.a[1]); m.a[1] = float(P.y + sn * k.a[0] + cs * k.a[1]);
      m.b[0] = float(P.x + cs * k.b[0] - sn * k.b[1]); m.b[1] = float(P.y + sn * k.b[0] + cs * k.b[1]);
      self_caps.push_back(m);
    }
    F.self_caps = self_caps.data();
    F.n_self_caps = int(self_caps.size());
  }
  if (c->oparams.objprob) {   // objprob: 검출 임베딩·벽 선분(기하 구조물 거르기)
    if (c->det_emb_n == dets->n && c->det_emb_dim > 0) { F.emb = c->det_emb.data(); F.emb_dim = c->det_emb_dim; }
    refreshGrid8(c);
    refreshWalls(c);
    if (c->wallsp) {
      const auto& W = *c->wallsp;
      for (const scenemap::WallSeg& w : W) wsegs.insert(wsegs.end(), {w.ax, w.ay, w.bx, w.by});
      // 창·유리문 자리 잇기: 같은 직선(나란하고 수직 거리 < 0.1 m) 위 두 벽 선분 사이 틈이 ap.bridge_max 안이면 그 틈도 벽으로(창은 깊이가
      // 지나가 점유가 없어 벽 선분이 끊김) — 기하 구조물 거르기의 '벽 너머' 판정에만 씀
      for (size_t i = 0; i < W.size(); ++i)
        for (size_t j = i + 1; j < W.size(); ++j) {
          const double ux = W[i].bx - W[i].ax, uy = W[i].by - W[i].ay, li = std::hypot(ux, uy);
          const double vx = W[j].bx - W[j].ax, vy = W[j].by - W[j].ay, lj = std::hypot(vx, vy);
          if (li < 1e-6 || lj < 1e-6 || std::fabs(ux * vy - uy * vx) / (li * lj) > 0.03) continue;
          const double nx = -uy / li, ny = ux / li;
          if (std::fabs((W[j].ax - W[i].ax) * nx + (W[j].ay - W[i].ay) * ny) > 0.1) continue;
          // i 직선 위 좌표로 두 구간, 사이 틈
          auto at = [&](double x, double y) { return ((x - W[i].ax) * ux + (y - W[i].ay) * uy) / li; };
          double a0 = 0, a1 = li, b0 = at(W[j].ax, W[j].ay), b1 = at(W[j].bx, W[j].by);
          if (b0 > b1) std::swap(b0, b1);
          const double lo = a1 <= b0 ? a1 : b1, hi = a1 <= b0 ? b0 : a0;   // a 가 앞이면 틈 [a1, b0], 뒤면 [b1, a0]
          if (hi - lo <= 0 || hi - lo > c->om.params().ap.bridge_max) continue;
          wsegs.insert(wsegs.end(), {W[i].ax + ux / li * lo, W[i].ay + uy / li * lo, W[i].ax + ux / li * hi, W[i].ay + uy / li * hi});
        }
    }
    F.wall_segs = wsegs.data();
    F.n_wall_segs = int(wsegs.size() / 4);
    ap_frame = F.emb != nullptr;
  }
  c->det_emb_n = -1;
  const auto to = TClock::now();
  c->om.update(F);
  const auto tv = TClock::now();
  c->addT(kStObjmap, usBetween(to, tv));
  pts.swap(c->om.lastPoints());   // 구름 후보(색은 잠금 밖에서)
  // 검출 → 물체 id
  const std::vector<DetAssoc>& as = c->om.lastAssoc();
  c->last_assoc.resize(as.size());
  for (size_t k = 0; k < as.size(); ++k) c->last_assoc[k] = as[k].obj_id;
  c->last_view_upd.assign(as.size(), 0);
  c->last_view_q.assign(as.size(), 0.f);
  for (size_t k = 0; k < as.size(); ++k)
    if (as[k].obj_id) c->last_view_q[k] = float(double(as[k].area_px) * (dets->score ? dets->score[k] : 1.f));
  // 옮겨짐·놓기: 지금 모습은 옛 자리 — 품질을 내려 다음 관측이 바꾸게
  const auto& ev = c->om.events();
  for (; c->ev_seen < ev.size(); ++c->ev_seen)
    if (ev[c->ev_seen].kind == 2 || ev[c->ev_seen].kind == 5) {
      auto it = c->views.find(ev[c->ev_seen].id);
      if (it != c->views.end()) it->second.q = 0;
    }
  // 없어진 물체(버린 후보)의 모습 버리기
  if (!c->views.empty()) {
    std::unordered_set<uint32_t> live;
    for (const MapObject& o : c->om.objects()) live.insert(o.id);
    for (auto it = c->views.begin(); it != c->views.end();) it = live.count(it->first) ? std::next(it) : c->views.erase(it);
  }
  // 새 모습 후보: 품질 = 유효 마스크 넓이 × 점수, 지금 것 이상(같으면 최근)
  host_rgb = im->rgba && dets->img_w == im->w && dets->img_h == im->h;
  for (size_t k = 0; k < as.size() && (crop || host_rgb); ++k) {
    if (!as[k].obj_id) continue;
    const float sc = dets->score ? dets->score[k] : 1.f;
    const double q = double(as[k].area_px) * sc;
    auto it = c->views.find(as[k].obj_id);
    if (it != c->views.end() && q < it->second.q) continue;
    ViewCand vc{};
    vc.id = as[k].obj_id;
    vc.det = int(k);
    vc.q = q;
    auto v = std::make_shared<BestView>();
    const float* b = dets->box + 4 * k;
    if (!cropGeometry(b, dets->img_w, dets->img_h, kCropMargin, kCropMaxSide, v->box, &v->w, &v->h)) continue;
    v->id = vc.id;
    v->q = q;
    v->stamp = im->stamp;
    for (int i = 0; i < 4; ++i) v->det_box[i] = int32_t(std::lround(b[i]));
    v->mask_area = as[k].area_px;
    v->depth_m = as[k].depth_med;
    v->score = sc;
    std::memcpy(v->cam_T, F.T_mc, sizeof(v->cam_T));
    v->rgb.resize(size_t(v->w) * v->h * 3);
    v->depth.resize(size_t(v->w) * v->h);
    cropMask(dets, int(k), v->box, v->w, v->h, &v->mask);
    vc.req = sm_crop_req{v->box[0], v->box[1], v->box[2], v->box[3], v->w, v->h, v->rgb.data()};
    vc.v = std::move(v);
    cand.push_back(std::move(vc));
  }
  epoch = c->epoch;
  c->addT(kStViewPrep, usBetween(tv, TClock::now()));
  updateGraph(c, im->stamp, true);
  }  // 잠금 끝: 자르기(장치 → 호스트 복사)는 잠금 밖에서
  if (cand.empty() && pts.empty() && !ap_frame) return 0;
  // 구름 점 색: 남긴 화소에서만(sgrt 는 장치에서 모아 그 색만 내려받음). 영상이 없으면 회색
  size_t npt = 0;
  for (const ObsPoints& q : pts) npt += q.px.size() / 2;
  std::vector<int32_t> xy;
  std::vector<uint8_t> rgb;
  bool have_rgb = false;
  const auto tg0 = TClock::now();
  if (npt) {
    xy.reserve(2 * npt);
    for (const ObsPoints& q : pts) xy.insert(xy.end(), q.px.begin(), q.px.end());
    rgb.assign(3 * npt, 128);
    if (gather) {
      have_rgb = gather(user, xy.data(), int32_t(npt), rgb.data()) == 0;
    } else if (host_rgb) {
      gatherRgbHost(im->rgba, int64_t(im->w) * 4, 4, im->w, im->h, xy.data(), int(npt), rgb.data());
      have_rgb = true;
    }
  }
  if (npt) c->addT(kStGather, usBetween(tg0, TClock::now()));
  const auto tc0 = TClock::now();
  bool crop_ok = !cand.empty();
  if (!cand.empty()) {
    std::vector<sm_crop_req> reqs(cand.size());
    for (size_t i = 0; i < cand.size(); ++i) reqs[i] = cand[i].req;
    if (crop) {
      crop_ok = crop(user, reqs.data(), int32_t(reqs.size())) == 0;   // 자르기 실패: 이번 모습은 버림
    } else {
      for (const sm_crop_req& r : reqs) cropRgbHost(im->rgba, int64_t(im->w) * 4, 4, r);
    }
    if (crop_ok)
      for (ViewCand& vc : cand)
        cropDepthMm(im->depth_m, im->w, im->h, dets->img_w, dets->img_h, vc.v->box, vc.v->w, vc.v->h, vc.v->depth.data());
    c->addT(kStCrop, usBetween(tc0, TClock::now()));
  }
  std::lock_guard<std::mutex> g(c->mu);
  if (c->epoch != epoch) return 0;
  const auto ta = TClock::now();
  size_t off = 0;
  for (const ObsPoints& q : pts) {
    const int n = int(q.px.size() / 2);
    c->om.addPoints(q.obj_id, q.xyz.data(), have_rgb ? rgb.data() + 3 * off : nullptr, n, im->stamp);
    off += size_t(n);
  }
  for (ViewCand& vc : cand) {
    if (!crop_ok) break;
    ViewSlot& sl = c->views[vc.id];
    if (sl.v && vc.q < sl.q) continue;
    vc.v->version = ++c->view_ver;
    sl.v = std::move(vc.v);
    sl.q = vc.q;
    if (vc.det >= 0 && vc.det < int(c->last_view_upd.size())) c->last_view_upd[vc.det] = 1;
  }
  c->addT(kStCloud, usBetween(ta, TClock::now()));
  if (ap_frame) {   // objprob: 이번 구름이 쌓인 뒤 통째 다시 담기 요청(투영 마스크)
    F.dets = nullptr;
    c->om.buildReencode(F, dets->img_w, dets->img_h, dets->mask_w, dets->mask_h, dets->mask_sx, dets->mask_sy, dets->mask_ox, dets->mask_oy,
                        &c->reenc);
    for (const ReencReq& r : c->reenc) {
      c->reenc_pub.push_back(sm_reenc_req{r.id, {r.box[0], r.box[1], r.box[2], r.box[3]}, r.kappa});
      c->reenc_bits.insert(c->reenc_bits.end(), r.bits.begin(), r.bits.end());
    }
  }
  return 0;
}

int sm_last_assoc(sm_ctx* c, uint32_t* ids, int cap) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  const int n = int(c->last_assoc.size());
  for (int k = 0; k < std::min(n, cap) && ids; ++k) ids[k] = c->last_assoc[k];
  return n;
}

int sm_last_views(sm_ctx* c, uint8_t* updated, float* quality, int cap) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  const int n = int(c->last_view_upd.size());
  for (int k = 0; k < std::min(n, cap); ++k) {
    if (updated) updated[k] = c->last_view_upd[k];
    if (quality) quality[k] = c->last_view_q[k];
  }
  return n;
}

int sm_mark_handled(sm_ctx* c, uint32_t id) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (std::find(c->handled.begin(), c->handled.end(), id) == c->handled.end()) c->handled.push_back(id);
  return 0;
}

int sm_snapshot(sm_ctx* c, sm_snapshot_t** out) {
  if (!c || !out) return -1;
  const auto t0 = TClock::now();
  auto* s = new sm_snapshot_t();
  {
    std::lock_guard<std::mutex> g(c->mu);
    double t;
    const Pose2 q = preview(c, &t);
    s->pose = sm_pose2{t, q.x, q.y, q.th};
    s->st = c->st;
    const OccGrid& gr = c->mapper.grid();
    s->res = gr.res();
    s->ox = gr.x0() * double(gr.res());
    s->oy = gr.y0() * double(gr.res());
    s->w = gr.width();
    s->h = gr.height();
    refreshGrid8(c);
    refreshWalls(c);
    s->cellsp = c->grid8;
    s->wallsp = c->wallsp;
    {
      const Scan2& sc = c->mapper.lastScan();
      const Pose2 sp = c->mapper.lastScanPose();
      s->scan_pose = sm_pose2{c->st.last_image_stamp, sp.x, sp.y, sp.th};
      s->scan_ox = sc.ox; s->scan_oy = sc.oy;
      s->scan_hx = sc.hx; s->scan_hy = sc.hy; s->scan_fx = sc.fx; s->scan_fy = sc.fy;
    }
    s->voxel = c->om.params().voxel;
    for (const MapObject& o : c->om.objects()) {
      if (!c->om.exportable(o)) continue;
      s->names.push_back(o.cls >= 0 && size_t(o.cls) < c->labels.size() ? c->labels[o.cls] : std::string("?"));
      sm_object e{};
      e.id = o.id;
      e.score = o.score;
      for (int k = 0; k < 3; ++k) { e.pos[k] = o.pos[k]; e.extent[k] = o.ext[k]; e.first_pos[k] = o.first_pos[k]; }
      e.n_obs = o.n_obs;
      e.last_seen = o.last_seen;
      e.state = o.held_by >= 0 ? SM_HELD : o.state;
      e.handled = std::find(c->handled.begin(), c->handled.end(), o.id) != c->handled.end();
      const int kind = c->om.kindOf(o.cls);
      e.structural = kind == SM_KIND_STATIC || kind == SM_KIND_STRUCT_OBJ || std::max({o.ext[0], o.ext[1], o.ext[2]}) > c->oparams.big;
      s->objs.push_back(e);
      s->movable.push_back(kind != SM_KIND_STATIC && kind != SM_KIND_STRUCT_OBJ);
      s->clouds.push_back(o.cloud);
      if (c->oparams.insp.on) {
        const InspectParams& ip = c->om.params().insp;
        s->insp.push_back(sm_inspect{o.id, o.insp.closest, o.insp.nViews(), inspTopSeen(o.insp, o.lo, o.hi, ip)});
      }
      auto it = c->views.find(o.id);
      s->views.push_back(it != c->views.end() ? it->second.v : nullptr);
    }
  }
  for (size_t i = 0; i < s->objs.size(); ++i) s->objs[i].name = s->names[i].c_str();
  s->st.n_objects = int32_t(s->objs.size());
  const auto tr = TClock::now();
  snapRooms(c, s);
  const auto t1 = TClock::now();
  {   // 방 → 그래프(바뀌었을 때), 그래프 사본
    std::lock_guard<std::mutex> g(c->mu);
    if (s->rseg) {
      std::vector<std::pair<uint32_t, uint32_t>> orl;
      for (size_t i = 0; i < s->objs.size() && i < s->rnames.obj_room.size(); ++i) orl.push_back({s->objs[i].id, s->rnames.obj_room[i]});
      std::vector<std::string> rn;
      for (const RoomLabel& L : s->rnames.rooms) rn.push_back(L.name);
      c->graph.updateRooms(s->rseg, orl, rn);
    }
    const auto tp = TClock::now();
    s->graph = c->graph.publish();
    c->addT(kStGraphPublish, usBetween(tp, TClock::now()));
  }
  c->addT(kStRooms, usBetween(tr, t1));
  c->addT(kStSnapshot, usBetween(t0, t1));
  *out = s;
  return 0;
}

void sm_snapshot_release(sm_snapshot_t* s) {
  if (s && s->refs.fetch_sub(1) == 1) delete s;
}

sm_pose2 sm_snap_pose(const sm_snapshot_t* s) { return s ? s->pose : sm_pose2{}; }
sm_status sm_snap_status(const sm_snapshot_t* s) { return s ? s->st : sm_status{}; }

int sm_snap_objects(const sm_snapshot_t* s, const sm_object** out) {
  if (!s || !out) return -1;
  *out = s->objs.data();
  return int(s->objs.size());
}

int sm_snap_inspect(const sm_snapshot_t* s, const sm_inspect** out) {
  if (!s || !out) return -1;
  if (s->insp.size() != s->objs.size()) { *out = nullptr; return -2; }   // 꺼짐
  *out = s->insp.data();
  return int(s->insp.size());
}

namespace {
// 살펴본 정도 → 노드 metadata 멤버(scene.json·view.json objects[]·스트림 view)
void inspMeta(const sm_snapshot_t* s, std::vector<std::string>* meta) {
  if (s->insp.size() != s->objs.size()) return;
  meta->resize(s->objs.size());
  for (size_t i = 0; i < s->insp.size(); ++i) {
    const sm_inspect& q = s->insp[i];
    char b[160];
    char cv[24], ts[24];
    if (q.closest_view_m >= 0) std::snprintf(cv, sizeof cv, "%.3f", double(q.closest_view_m)); else std::snprintf(cv, sizeof cv, "null");
    if (q.top_seen >= 0) std::snprintf(ts, sizeof ts, "%.4g", double(q.top_seen)); else std::snprintf(ts, sizeof ts, "null");
    std::snprintf(b, sizeof b, "\"inspect\":{\"closest_view_m\":%s,\"n_views\":%d,\"top_seen\":%s}", cv, int(q.n_views), ts);
    std::string& d = (*meta)[i];
    d = d.empty() ? std::string(b) : d + "," + b;
  }
}
}  // namespace

// 이름 부분 일치(대소문자·'_'↔' ' 무시), 점수 = 일치 길이 비율 × 관측 신뢰도. 점수 순.
int sm_snap_find(const sm_snapshot_t* s, const char* name, uint32_t* ids, float* scores, int cap) {
  if (!s || !name) return -1;
  auto norm = [](std::string t) {
    for (char& ch : t) ch = ch == '_' ? ' ' : char(std::tolower(static_cast<unsigned char>(ch)));
    return t;
  };
  const std::string q = norm(name);
  std::vector<std::pair<float, uint32_t>> hit;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    const std::string n = norm(s->names[i]);
    if (q.empty() || n.find(q) == std::string::npos) continue;
    hit.push_back({float(q.size()) / float(std::max<size_t>(1, n.size())) * std::max(0.05f, s->objs[i].score), s->objs[i].id});
  }
  std::sort(hit.begin(), hit.end(), [](auto& a, auto& b) { return a.first > b.first; });
  const int m = std::min<int>(cap, int(hit.size()));
  for (int i = 0; i < m; ++i) {
    if (ids) ids[i] = hit[i].second;
    if (scores) scores[i] = hit[i].first;
  }
  return m;
}

// p 에서 r 안의 물체(중심 거리), 가까운 순.
int sm_snap_near(const sm_snapshot_t* s, const double p[3], double r, uint32_t* ids, int cap) {
  if (!s || !p) return -1;
  std::vector<std::pair<double, uint32_t>> hit;
  for (const sm_object& o : s->objs) {
    const double d = std::sqrt((o.pos[0] - p[0]) * (o.pos[0] - p[0]) + (o.pos[1] - p[1]) * (o.pos[1] - p[1]) + (o.pos[2] - p[2]) * (o.pos[2] - p[2]));
    if (d <= r) hit.push_back({d, o.id});
  }
  std::sort(hit.begin(), hit.end());
  const int m = std::min<int>(cap, int(hit.size()));
  for (int i = 0; i < m && ids; ++i) ids[i] = hit[i].second;
  return m;
}

int sm_snap_view(const sm_snapshot_t* s, uint32_t id, sm_view* out) {
  if (!s || !out) return -1;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    if (s->objs[i].id != id) continue;
    const BestView* v = s->views[i].get();
    if (!v) return 0;
    sm_view& o = *out;
    o = sm_view{};
    o.id = v->id;
    o.version = v->version;
    o.stamp = v->stamp;
    std::memcpy(o.box_px, v->box, sizeof(o.box_px));
    std::memcpy(o.det_box_px, v->det_box, sizeof(o.det_box_px));
    o.mask_area = v->mask_area;
    o.depth_m = v->depth_m;
    o.score = v->score;
    std::memcpy(o.cam_T, v->cam_T, sizeof(o.cam_T));
    o.w = v->w;
    o.h = v->h;
    o.rgb = v->rgb.empty() ? nullptr : v->rgb.data();
    o.depth_mm = v->depth.empty() ? nullptr : v->depth.data();
    o.mask = v->mask.empty() ? nullptr : v->mask.data();
    return 1;
  }
  return 0;
}

int sm_snap_points(const sm_snapshot_t* s, uint32_t id, sm_cloud* out) {
  if (!s || !out) return -1;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    if (s->objs[i].id != id) continue;
    const ObjCloud& cl = s->clouds[i];
    *out = sm_cloud{};
    out->id = id;
    out->version = cl.version;
    out->n = int32_t(cl.size());
    for (int k = 0; k < 3; ++k) out->origin[k] = cl.org[k];
    out->voxel = s->voxel;
    out->stamp = cl.stamp;
    static_assert(sizeof(sm_cloud_pt) == sizeof(CloudPt), "sm_cloud_pt == CloudPt");
    out->pts = cl.data ? reinterpret_cast<const sm_cloud_pt*>(cl.data->pts.data()) : nullptr;
    return 1;
  }
  return 0;
}

int sm_set_cloud_params(sm_ctx* c, double voxel, int32_t cap) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (voxel > 0) c->oparams.voxel = voxel;
  if (cap > 0) c->oparams.cloud_cap = cap;
  c->om.setCloudParams(voxel, cap);
  return 0;
}

int sm_snap_movable(const sm_snapshot_t* s, uint32_t id) {
  if (!s) return -1;
  for (size_t i = 0; i < s->objs.size(); ++i)
    if (s->objs[i].id == id) return s->movable[i];
  return -1;
}

int sm_take_dirty(sm_ctx* c, int32_t out[4], uint64_t* version) {
  if (!c || !out) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  OccGrid& gr = c->mapper.gridMut();
  if (version) *version = gr.version();
  int x0, y0, x1, y1;
  if (!gr.takeDirty(&x0, &y0, &x1, &y1)) return 0;
  out[0] = x0 - gr.x0(); out[1] = y0 - gr.y0(); out[2] = x1 - gr.x0(); out[3] = y1 - gr.y0();
  return 1;
}

int sm_snap_scan(const sm_snapshot_t* s, sm_scan2* out) {
  if (!s || !out) return -1;
  out->pose = s->scan_pose;
  out->ox = s->scan_ox; out->oy = s->scan_oy;
  out->n_hit = int32_t(s->scan_hx.size());
  out->hx = s->scan_hx.data(); out->hy = s->scan_hy.data();
  out->n_free = int32_t(s->scan_fx.size());
  out->fx = s->scan_fx.data(); out->fy = s->scan_fy.data();
  return out->n_hit + out->n_free > 0 ? 0 : 1;
}

int sm_snap_map(const sm_snapshot_t* s, sm_grid* out) {
  if (!s || !out) return -1;
  out->resolution = s->res;
  out->origin[0] = s->ox;
  out->origin[1] = s->oy;
  out->width = s->w;
  out->height = s->h;
  out->cells = s->cells();
  return 0;
}

// 벽 선분은 스냅숏을 만들 때(refreshGrid8) 격자와 함께 갱신돼 있다 — 여기서는 읽기만
namespace {
scenemap::WallGrid wallView(const sm_snapshot_t* s) { return {s->cells(), s->w, s->h, s->res, s->ox, s->oy}; }
}  // namespace

int sm_snap_wall_state(const sm_snapshot_t* s, const double pose[3], float out[SM_WALL_STATE_LEN]) {
  if (!s || !out || !s->cells() || s->w <= 0 || s->h <= 0) return -1;
  const double p[3] = {pose ? pose[0] : s->pose.x, pose ? pose[1] : s->pose.y, pose ? pose[2] : s->pose.yaw};
  if (!s->wallsp) return -1;
  scenemap::wallStateVector(wallView(s), *s->wallsp, p, out);
  return 0;
}

int sm_snap_wall_segments(const sm_snapshot_t* s, double* out, int cap) {
  if (!s || !s->cells() || s->w <= 0 || s->h <= 0) return -1;
  if (!s->wallsp) return -1;
  const auto& segs = s->wallsp;
  for (int i = 0; out && i < cap && i < int(segs->size()); ++i) {
    const auto& w = (*segs)[i];
    out[4 * i] = w.ax; out[4 * i + 1] = w.ay; out[4 * i + 2] = w.bx; out[4 * i + 3] = w.by;
  }
  return int(segs->size());
}

// 격자 위 8방향 A*. 점유(≥ 65 %) 칸을 로봇 반경 0.30 m 만큼 부풀려 막는다. 모르는 칸은 지나갈 수 있되 1.5 배 비용.
// 목표: `to` 에서 0.6 m 안의 막히지 않은 칸(물체는 가구 위에 있으니 그 앞까지). 출발 칸이 부풀림 안이면 0.4 m 안은 풀어 준다.
// 두 점 중 하나라도 지도 밖이면 직선 거리(모름).
double sm_snap_reachable(const sm_snapshot_t* s, const double from[2], const double to[2]) {
  if (!s || !from || !to) return -1;
  const double straight = std::hypot(to[0] - from[0], to[1] - from[1]);
  const int W = s->w, H = s->h;
  auto cell = [&](const double* p, int* x, int* y) {
    *x = int(std::floor((p[0] - s->ox) / s->res));
    *y = int(std::floor((p[1] - s->oy) / s->res));
    return *x >= 0 && *y >= 0 && *x < W && *y < H;
  };
  int sx, sy, gx, gy;
  if (!W || !cell(from, &sx, &sy) || !cell(to, &gx, &gy)) return straight;
  std::call_once(s->inflate_once, [&] {
    s->blocked.assign(size_t(W) * H, 0);
    const int r = int(std::ceil(0.30 / s->res));
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (s->cells()[size_t(y) * W + x] < 65) continue;
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            const int X = x + dx, Y = y + dy;
            if (X < 0 || Y < 0 || X >= W || Y >= H || dx * dx + dy * dy > r * r) continue;
            s->blocked[size_t(Y) * W + X] = 1;
          }
      }
  });
  const int rs = int(std::ceil(0.40 / s->res)), rg = int(std::ceil(0.60 / s->res));
  auto isBlocked = [&](int x, int y) {
    if ((x - sx) * (x - sx) + (y - sy) * (y - sy) <= rs * rs) return false;
    return s->blocked[size_t(y) * W + x] != 0;
  };
  auto isGoal = [&](int x, int y) { return (x - gx) * (x - gx) + (y - gy) * (y - gy) <= rg * rg; };
  std::vector<float> dist(size_t(W) * H, INFINITY);
  using QE = std::pair<float, int>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
  auto hcost = [&](int x, int y) { return float(std::hypot(x - gx, y - gy)); };
  dist[size_t(sy) * W + sx] = 0;
  pq.push({hcost(sx, sy), sy * W + sx});
  const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (!pq.empty()) {
    const auto [f, id] = pq.top();
    pq.pop();
    const int x = id % W, y = id / W;
    const float d = dist[id];
    if (f > d + hcost(x, y) + 1e-3f) continue;
    if (isGoal(x, y)) {
      const double rest = std::max(0.0, std::hypot(x - gx, y - gy) * s->res);   // 목표 둘레에서 멈춘 만큼(가구 위 물체까지)
      return d * s->res + rest;
    }
    for (int k = 0; k < 8; ++k) {
      const int X = x + DX[k], Y = y + DY[k];
      if (X < 0 || Y < 0 || X >= W || Y >= H || isBlocked(X, Y)) continue;
      const float step = (k < 4 ? 1.f : 1.41421356f) * (s->cells()[size_t(Y) * W + X] < 0 ? 1.5f : 1.f);
      const size_t j = size_t(Y) * W + X;
      if (d + step < dist[j]) {
        dist[j] = d + step;
        pq.push({dist[j] + hcost(X, Y), int(j)});
      }
    }
  }
  return -1;
}

namespace {
// objprob 저장: objects/O<id>_emb.f16 = μ(768 × FP16, L2 — 예전 형식 그대로), objects/O<id>_views.f16 = 상위 K 모습(K × 768 FP16, κ 큰 순,
// 통째 먼저). 노드 metadata(scene.json·view.json objects[]) 에 "emb"·"name_post"·"pos_sd" — README "확률 모드" 저장 형식
struct ApFile { uint32_t id; std::vector<uint16_t> mu, views; };

std::string jf(double v) { char b[32]; std::snprintf(b, sizeof b, "%.4g", v); return b; }
std::string jesc(const std::string& t) {
  std::string o;
  for (char ch : t) { if (ch == '"' || ch == '\\') o += '\\'; o += ch; }
  return o;
}

void apSaveMeta(sm_ctx* c, const sm_snapshot_t* s, std::vector<std::string>* meta, std::vector<ApFile>* files) {
  std::unordered_map<uint32_t, const MapObject*> by;
  for (const MapObject& o : c->om.objects()) by[o.id] = &o;
  for (size_t i = 0; i < s->objs.size(); ++i) {
    auto it = by.find(s->objs[i].id);
    if (it == by.end() || !it->second->ap) continue;
    const MapObject& o = *it->second;
    const ApState& a = *o.ap;
    std::vector<float> mu;
    double conf = 0;
    if (!apMu(a, &mu, &conf)) continue;
    const std::string b = "objects/O" + std::to_string(o.id);
    std::string m = "\"emb\":{\"path\":\"" + b + "_emb.f16\",\"views\":\"" + b + "_views.f16\",\"dim\":" + std::to_string(mu.size()) +
                    ",\"n_views\":" + std::to_string(a.views.size()) + ",\"conf\":" + jf(conf) + ",\"kappa_sum\":" + jf(a.k_whole > 0 ? a.k_whole : a.k_frag) +
                    ",\"source\":\"" + (a.k_whole > 0 ? "whole" : "frag") + "\",\"n_whole\":" + std::to_string(a.n_whole) + ",\"kappa\":[";
    for (size_t k = 0; k < a.views.size(); ++k) m += (k ? "," : "") + jf(a.views[k].kappa);
    m += "],\"whole\":[";
    for (size_t k = 0; k < a.views.size(); ++k) m += std::string(k ? "," : "") + (a.views[k].whole ? "1" : "0");
    m += "]}";
    if (!a.post.empty()) {   // 이름 사후 상위 5·이름 확률·엔트로피(nat)·상위어로 올렸나·바깥 관측 있나
      std::vector<int> ord(a.post.size());
      for (size_t k = 0; k < ord.size(); ++k) ord[k] = int(k);
      std::partial_sort(ord.begin(), ord.begin() + std::min<size_t>(5, ord.size()), ord.end(),
                        [&](int x, int y) { return a.post[size_t(x)] > a.post[size_t(y)]; });
      m += ",\"name_post\":{\"top\":[";
      for (size_t k = 0; k < std::min<size_t>(5, ord.size()); ++k) {
        const int l = ord[k];
        m += std::string(k ? "," : "") + "[\"" + jesc(size_t(l) < c->labels.size() ? c->labels[size_t(l)] : "?") + "\"," + jf(a.post[size_t(l)]) + "]";
      }
      m += "],\"p\":" + jf(a.name_p) + ",\"entropy\":" + jf(a.name_H) + ",\"rolled\":" + (a.rolled ? "true" : "false") +
           ",\"external\":" + (a.L_ext.empty() ? "false" : "true") + "}";
    }
    m += ",\"pos_sd\":[" + jf(std::sqrt(a.P[0])) + "," + jf(std::sqrt(a.P[1])) + "," + jf(std::sqrt(a.P[2])) + "]";
    std::string& dst = (*meta)[i];
    dst = dst.empty() ? m : dst + "," + m;
    auto sv = c->saved_ap_ver.find(o.id);
    if (sv != c->saved_ap_ver.end() && sv->second == a.ver) continue;
    c->saved_ap_ver[o.id] = a.ver;
    ApFile f;
    f.id = o.id;
    f.mu.resize(mu.size());
    apToF16(mu.data(), f.mu.data(), int(mu.size()));
    for (const ApView& v : a.views) f.views.insert(f.views.end(), v.z.begin(), v.z.end());
    files->push_back(std::move(f));
  }
}

void apWriteFiles(const char* dir, const std::vector<ApFile>& files) {
  if (files.empty()) return;
  const std::string od = std::string(dir) + "/objects";
  std::error_code ec;
  std::filesystem::create_directories(od, ec);
  auto put = [&](const std::string& path, const std::vector<uint16_t>& v) {
    const std::string tmp = path + ".tmp";
    FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) return;
    std::fwrite(v.data(), 2, v.size(), fp);
    std::fclose(fp);
    std::rename(tmp.c_str(), path.c_str());
  };
  for (const ApFile& f : files) {
    const std::string b = od + "/O" + std::to_string(f.id);
    put(b + "_emb.f16", f.mu);
    put(b + "_views.f16", f.views);
  }
}
}  // namespace

int sm_save_dsg(sm_ctx* c, const char* dir) { return sm_save_dsg_ex(c, dir, nullptr); }

int sm_save_dsg_ex(sm_ctx* c, const char* dir, sm_save_stats* stats) {
  if (!c || !dir) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  sm_snapshot_t* s = nullptr;
  if (sm_snapshot(c, &s) != 0) return -1;
  SaveInput in;
  std::vector<ApFile> apw;
  in.stamp = s->pose.stamp;
  in.pose[0] = s->pose.x; in.pose[1] = s->pose.y; in.pose[2] = s->pose.yaw;
  in.objs = s->objs.data();
  in.n_objs = int(s->objs.size());
  in.grid_res = s->res; in.grid_ox = s->ox; in.grid_oy = s->oy; in.grid_w = s->w; in.grid_h = s->h;
  in.cells = s->cells();
  in.views = s->views;
  in.movable = s->movable;
  in.png_dirty.assign(s->objs.size(), 0);
  in.clouds = s->clouds;
  in.ply_dirty.assign(s->objs.size(), 0);
  in.voxel = s->voxel;
  in.rooms = s->rseg;
  in.room_names = s->rnames;
  in.graph = s->graph;
  {
    std::lock_guard<std::mutex> g(c->mu);
    const auto& ev = c->om.events();
    in.events.assign(ev.end() - std::min<size_t>(ev.size(), 50), ev.end());   // 최근 50 개
    if (c->saved_dir != dir) {
      c->saved_dir = dir;
      c->saved_ver.clear();
      c->saved_cloud_ver.clear();
      c->saved_ap_ver.clear();
      c->clean_objects = true;
    }
    for (size_t i = 0; i < s->objs.size(); ++i) {
      auto it = c->saved_cloud_ver.find(s->objs[i].id);
      in.ply_dirty[i] = it == c->saved_cloud_ver.end() || it->second != s->clouds[i].version;
    }
    for (size_t i = 0; i < s->objs.size(); ++i) {
      if (!s->views[i]) continue;
      auto it = c->saved_ver.find(s->objs[i].id);
      in.png_dirty[i] = it == c->saved_ver.end() || it->second != s->views[i]->version;
    }
    in.clean_objects = c->clean_objects;
    in.obj_meta.assign(s->objs.size(), std::string());
    for (size_t i = 0; i < s->objs.size(); ++i) {
      auto it = c->obj_meta.find(s->objs[i].id);
      if (it != c->obj_meta.end()) in.obj_meta[i] = it->second;
    }
    if (c->oparams.objprob) apSaveMeta(c, s, &in.obj_meta, &apw);
    inspMeta(s, &in.obj_meta);
  }
  SaveOut out;
  std::unique_lock<std::mutex> sl(c->save_mu);   // 저장 하나씩(노드 조각 캐시)
  in.json_cache = &c->jcache;
  apWriteFiles(dir, apw);   // scene.json·view.json 이 가리키기 전에
  const int rc = saveScene(in, dir, &out);
  sl.unlock();
  {
    std::lock_guard<std::mutex> g(c->mu);
    if (c->saved_dir == dir) {
      for (size_t i = 0; i < s->objs.size() && i < out.png_ok.size(); ++i)
        if (out.png_ok[i]) c->saved_ver[s->objs[i].id] = s->views[i]->version;
      for (size_t i = 0; i < s->objs.size() && i < out.ply_ok.size(); ++i)
        if (out.ply_ok[i]) c->saved_cloud_ver[s->objs[i].id] = s->clouds[i].version;
      if (rc == 0) c->clean_objects = false;
    }
  }
  c->addT(kStSave, usBetween(t0, std::chrono::steady_clock::now()));
  if (stats) {
    stats->n_objects = int32_t(s->objs.size());
    stats->n_png = out.n_png;
    stats->png_ms = float(out.png_ms);
    stats->n_ply = out.n_ply;
    stats->ply_ms = float(out.ply_ms);
    stats->total_ms = float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  sm_snapshot_release(s);
  return rc;
}

// ---- 방(rooms.hpp) ----

int sm_get_room_params(sm_ctx* c, sm_room_params* o) {
  if (!c || !o) return -1;
  const RoomParams p = c->rooms.params();
  *o = sm_room_params{p.enabled ? 1 : 0, p.free_max, p.occ_min, p.dil_min, p.dil_max, p.dil_step, p.min_life, p.min_seed_m2,
                      p.min_room_m2, p.max_door_m, p.hole_m2, p.speck_m2, p.obj_search_m, p.footprint_margin, p.period_s,
                      p.min_change_m2, p.match_min};
  return 0;
}

int sm_set_room_params(sm_ctx* c, const sm_room_params* q) {
  if (!c || !q || q->dil_step_m <= 0 || q->dil_max_m < q->dil_min_m) return -1;
  RoomParams p;
  p.enabled = q->enabled != 0;
  p.free_max = q->free_max; p.occ_min = q->occ_min;
  p.dil_min = q->dil_min_m; p.dil_max = q->dil_max_m; p.dil_step = q->dil_step_m; p.min_life = q->min_life_m;
  p.min_seed_m2 = q->min_seed_m2; p.min_room_m2 = q->min_room_m2; p.max_door_m = q->max_door_m;
  p.hole_m2 = q->hole_m2; p.speck_m2 = q->speck_m2; p.obj_search_m = q->obj_search_m; p.footprint_margin = q->footprint_margin_m;
  p.period_s = q->period_s; p.min_change_m2 = q->min_change_m2; p.match_min = q->match_min;
  c->rooms.setParams(p);
  return 0;
}

int sm_update_rooms(sm_ctx* c, int32_t force) {
  if (!c) return -1;
  c->rooms_force = force != 0;
  sm_snapshot_t* s = nullptr;
  if (sm_snapshot(c, &s) != 0) return -1;
  const int n = int(s->rooms.size());
  sm_snapshot_release(s);
  return n;
}

int sm_set_room_name(sm_ctx* c, uint32_t id, const char* name, float conf) {
  if (!c) return -1;
  c->rooms.setName(id, name, conf);
  return 0;
}

int sm_snap_rooms(const sm_snapshot_t* s, const sm_room** out) {
  if (!s || !out) return -1;
  *out = s->rooms.empty() ? nullptr : s->rooms.data();
  return int(s->rooms.size());
}

int sm_snap_room_doors(const sm_snapshot_t* s, const sm_room_door** out) {
  if (!s || !out) return -1;
  *out = s->rdoors.empty() ? nullptr : s->rdoors.data();
  return int(s->rdoors.size());
}

int sm_snap_room_grid(const sm_snapshot_t* s, sm_room_grid* out) {
  if (!s || !out) return -1;
  if (!s->rseg) return 1;
  const RoomSeg& R = *s->rseg;
  *out = sm_room_grid{R.res, {R.gx0 * R.res, R.gy0 * R.res}, R.w, R.h, R.ids.empty() ? nullptr : R.ids.data()};
  return 0;
}

uint32_t sm_snap_room_at(const sm_snapshot_t* s, const double p[2]) {
  if (!s || !p || !s->rseg) return 0;
  return s->rseg->at(p[0], p[1]);
}

uint32_t sm_snap_object_room(const sm_snapshot_t* s, uint32_t id) {
  if (!s) return 0;
  for (size_t i = 0; i < s->objs.size() && i < s->rnames.obj_room.size(); ++i)
    if (s->objs[i].id == id) return s->rnames.obj_room[i];
  return 0;
}

// ---- 자세 원천·넣기 정책·단계 시간(추가 ABI) ----

int sm_set_pose_mode(sm_ctx* c, int32_t mode) {
  if (!c || mode < SM_POSE_SLAM || mode > SM_POSE_EXT) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->pose_mode = mode == SM_POSE_SLAM ? SM_POSE_EXT : mode;   // SM_POSE_SLAM(옛 slam2d, archive) = EXT
  return 0;
}

int sm_set_cam_extrinsic(sm_ctx* c, int32_t cam, const double* T_bc) {
  if (!c || cam != 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->ext_cam = T_bc != nullptr;
  if (T_bc)
    for (int k = 0; k < 12; ++k) c->ext_T[k] = float(T_bc[k]);
  return 0;
}

int sm_get_pose_mode(sm_ctx* c) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  return c->pose_mode;
}

int sm_push_pose(sm_ctx* c, const sm_pose2* p) {
  if (!c || !p || !std::isfinite(p->x) || !std::isfinite(p->y) || !std::isfinite(p->yaw)) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (!c->gtq.empty() && p->stamp < c->gtq.back().stamp - 1e-9) c->gtq.clear();   // 시각이 되돌아감(새 판)
  c->gtq.push_back(*p);
  // The robot trajectory (AGENTS layer) follows the GT pose directly: every pushed pose goes through the same keyframe thresholds
  // (GraphParams agent_xy / agent_yaw / agent_s), with no image or proprio needed in between.
  if (c->pose_mode == SM_POSE_GT) {
    c->graph.updateAgent(p->stamp, Pose2{p->x, p->y, p->yaw});
    if (c->stream.running()) c->stream.pushPose(p->stamp, p->x, p->y, p->yaw);
  }
  // 적분·짝짓기에 쓸 만큼만(가장 오래 기다리는 proprio·영상보다 2 s 앞까지)
  const double keep = (c->have_used ? c->last_used.stamp : p->stamp) - 2.0;
  while (c->gtq.size() > 2 && c->gtq[1].stamp < keep) c->gtq.pop_front();
  return 0;
}

int sm_push_ext_pose(sm_ctx* c, const sm_pose2* p) {
  if (!c || !p || !std::isfinite(p->x) || !std::isfinite(p->y) || !std::isfinite(p->yaw)) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (!c->extq.empty() && p->stamp < c->extq.back().stamp - 1e-9) c->extq.clear();   // 시각이 되돌아감(새 판)
  c->extq.push_back(*p);
  const double keep = (c->have_used ? c->last_used.stamp : p->stamp) - 2.0;
  while (c->extq.size() > 2 && c->extq[1].stamp < keep) c->extq.pop_front();
  return 0;
}

int sm_get_pose_diag(sm_ctx* c, sm_pose_diag* out) {
  if (!c || !out) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  *out = c->diag;
  return 0;
}

int sm_set_map_update(sm_ctx* c, int32_t policy, int32_t still_every) {
  if (!c || policy < 0 || policy > 1) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->map_policy = policy;
  if (still_every > 0) c->still_every = still_every;
  c->mapper.setUpdatePolicy(policy, still_every);
  return 0;
}

int sm_get_timing(sm_ctx* c, sm_stage_timing* out, int32_t cap) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->tmu);
  for (int k = 0; k < kStCount && k < cap && out; ++k) {
    const StageHist& h = c->tm.h[k];
    sm_stage_timing& o = out[k];
    o.name = stageName(k);
    o.n = int64_t(h.n);
    o.total_us = h.sum;
    o.mean_us = h.n ? h.sum / double(h.n) : 0;
    o.p50_us = h.quantile(0.5);
    o.p99_us = h.quantile(0.99);
    o.max_us = h.max;
    o.last_us = h.last;
  }
  return kStCount;
}

int sm_reset_timing(sm_ctx* c) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->tmu);
  c->tm.clear();
  return 0;
}

// ---- 장면 그래프(추가 ABI) ----

int sm_snap_graph_nodes(const sm_snapshot_t* s, int32_t group, const sm_gnode** out) {
  if (!s || !out) return -1;
  *out = nullptr;
  if (!s->graph) return 0;
  const GraphView& g = *s->graph;
  if (group < 0) { *out = g.cn.empty() ? nullptr : g.cn.data(); return int(g.cn.size()); }
  if (group > 4) return -1;
  const int a = g.layer_off[group], b = g.layer_off[group + 1];
  *out = b > a ? g.cn.data() + a : nullptr;
  return b - a;
}

int sm_snap_graph_edges(const sm_snapshot_t* s, const sm_gedge** out) {
  if (!s || !out) return -1;
  *out = nullptr;
  if (!s->graph || s->graph->edges.empty()) return 0;
  *out = reinterpret_cast<const sm_gedge*>(s->graph->edges.data());
  return int(s->graph->edges.size());
}

const sm_gnode* sm_snap_graph_node(const sm_snapshot_t* s, uint64_t id) {
  if (!s || !s->graph) return nullptr;
  auto it = s->graph->index.find(id);
  return it == s->graph->index.end() ? nullptr : &s->graph->cn[it->second];
}

int sm_snap_graph_neighbors(const sm_snapshot_t* s, uint64_t id, int32_t* edge_idx, int32_t cap) {
  if (!s || !s->graph) return -1;
  const GraphView& g = *s->graph;
  auto it = g.index.find(id);
  if (it == g.index.end()) return -1;
  const uint32_t a = g.adj_off[it->second], b = g.adj_off[it->second + 1];
  for (uint32_t k = a; k < b && edge_idx && int32_t(k - a) < cap; ++k) edge_idx[k - a] = int32_t(g.adj[k]);
  return int(b - a);
}

int sm_snap_place_path(const sm_snapshot_t* s, const double from[2], const double to[2], double min_clear, uint64_t* ids, int32_t cap,
                       double* length) {
  if (!s || !from || !to) return -1;
  if (length) *length = 0;
  if (!s->graph) return 0;
  const GraphView& g = *s->graph;
  const int a = g.layer_off[SM_GL_PLACES], b = g.layer_off[SM_GL_PLACES + 1];
  if (b <= a) return 0;
  auto nearest = [&](const double* p) {
    int best = -1;
    double bd = 2.0 * 2.0;
    for (int i = a; i < b; ++i) {
      const double d = (g.nodes[i].pos[0] - p[0]) * (g.nodes[i].pos[0] - p[0]) + (g.nodes[i].pos[1] - p[1]) * (g.nodes[i].pos[1] - p[1]);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  const int s0 = nearest(from), g0 = nearest(to);
  if (s0 < 0 || g0 < 0) return 0;
  const int n = b - a;
  std::vector<double> dist(n, INFINITY);
  std::vector<int> prev(n, -1);
  using QE = std::pair<double, int>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
  dist[s0 - a] = 0;
  pq.push({0, s0});
  while (!pq.empty()) {
    const auto [d, u] = pq.top();
    pq.pop();
    if (d > dist[u - a]) continue;
    if (u == g0) break;
    for (uint32_t k = g.adj_off[u]; k < g.adj_off[u + 1]; ++k) {
      const GEdge& e = g.edges[g.adj[k]];
      if (e.rel != kRelPlace || e.weight < min_clear) continue;
      auto jt = g.index.find(e.a == g.nodes[u].id ? e.b : e.a);
      if (jt == g.index.end()) continue;
      const int v = int(jt->second);
      if (v < a || v >= b) continue;
      const double w = std::hypot(g.nodes[v].pos[0] - g.nodes[u].pos[0], g.nodes[v].pos[1] - g.nodes[u].pos[1]);
      if (d + w < dist[v - a]) {
        dist[v - a] = d + w;
        prev[v - a] = u;
        pq.push({d + w, v});
      }
    }
  }
  if (!std::isfinite(dist[g0 - a])) return 0;
  std::vector<uint64_t> path;
  for (int u = g0; u >= 0; u = prev[u - a]) path.push_back(g.nodes[u].id);
  std::reverse(path.begin(), path.end());
  for (int i = 0; i < int(path.size()) && i < cap && ids; ++i) ids[i] = path[i];
  if (length) *length = dist[g0 - a];
  return int(path.size());
}

int sm_set_obj_params(sm_ctx* c, const char* kv) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (kv && *kv) c->obj_kv += (c->obj_kv.empty() ? "" : ",") + std::string(kv);
  const int bad = ObjectMap::applyParams(&c->oparams, kv);
  ObjectMap::applyParams(&c->om.paramsMut(), kv);
  ObjectMap::envOverrides(&c->om.paramsMut());   // 진단 변수가 이김(ObjectMap 을 새로 만들 때와 같은 순서)
  return bad ? -2 : 0;
}

int sm_set_inspect(sm_ctx* c, int32_t on) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->oparams.insp.on = on != 0;
  c->om.paramsMut().insp.on = on != 0;
  return 0;
}

int sm_set_object_meta(sm_ctx* c, uint32_t id, const char* json) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (!json || !*json) c->obj_meta.erase(id);
  else c->obj_meta[id] = json;
  return 0;
}

// ---- objprob ----
int sm_set_object_model(sm_ctx* c, int32_t objprob) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->oparams.objprob = objprob != 0;
  c->om.paramsMut().objprob = objprob != 0;
  c->applyKinds();
  if (objprob) {   // 큰 가구(소파 ≈ 4 m²)의 구름이 접촉 판정에 쓰이므로 한도를 넉넉히
    c->oparams.cloud_cap = std::max(c->oparams.cloud_cap, 8000);
    c->om.setCloudParams(0, c->oparams.cloud_cap);
  }
  ObjectMap::envOverrides(&c->om.paramsMut());
  return 0;
}

int sm_set_text_model(sm_ctx* c, const float* text, const int32_t* row_label, int32_t rows, int32_t dim, float scale, float bias) {
  if (!c || !text || !row_label || rows <= 0 || dim <= 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  ApText t = c->om.textModel();
  // 벽에 붙는 납작한 물체(벽 선 위 얇은 평면이어도 남김)
  static const char* const kFlat[] = {"picture frame", "picture", "painting", "poster", "tv", "television", "monitor", "whiteboard", "clock", "sign", "mirror"};
  t.flat_ok.assign(c->labels.size(), 0);
  for (size_t i = 0; i < c->labels.size(); ++i)
    for (const char* e : kFlat) t.flat_ok[i] = t.flat_ok[i] || headMatch(normName(c->labels[i]), e);
  t.so_shape.assign(c->labels.size(), 0);   // 구조 물체 모양 묶음(크기 확인)
  for (size_t i = 0; i < c->labels.size(); ++i) {
    const std::string n = normName(c->labels[i]);
    for (const char* e : {"door", "doorway", "door frame", "window"}) if (headMatch(n, e)) t.so_shape[i] = 1;
    for (const char* e : {"staircase", "stairs", "stair", "railing"}) if (headMatch(n, e)) t.so_shape[i] = 2;
    for (const char* e : {"pillar", "column"}) if (headMatch(n, e)) t.so_shape[i] = 3;
  }
  t.dim = dim;
  t.text.assign(text, text + size_t(rows) * dim);
  t.row_label.assign(row_label, row_label + rows);
  t.scale = scale;
  t.bias = bias;
  t.n_labels = int(c->labels.size());
  c->om.setTextModel(std::move(t));
  return 0;
}

int sm_set_label_stats(sm_ctx* c, const float* log_prior, const float* mu, const float* sd, const int32_t* parent, int32_t n, int32_t object_label) {
  if (!c || n < 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  ApText t = c->om.textModel();
  t.n_labels = int(c->labels.size());
  if (log_prior) t.log_prior.assign(log_prior, log_prior + n); else t.log_prior.clear();
  if (mu && sd) { t.ls_mu.assign(mu, mu + n); t.ls_sd.assign(sd, sd + n); } else { t.ls_mu.clear(); t.ls_sd.clear(); }
  if (parent) t.parent.assign(parent, parent + n); else t.parent.clear();
  t.object_label = object_label;
  c->om.setTextModel(std::move(t));
  return 0;
}

int sm_set_det_embeddings(sm_ctx* c, const float* emb, int32_t n, int32_t dim) {
  if (!c || n < 0 || dim <= 0 || (n > 0 && !emb)) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  c->det_emb.assign(emb, emb + size_t(n) * dim);
  c->det_emb_n = n;
  c->det_emb_dim = dim;
  return 0;
}

int sm_reencode_requests(sm_ctx* c, const sm_reenc_req** reqs, const uint32_t** bits) {
  if (!c) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  if (reqs) *reqs = c->reenc_pub.data();
  if (bits) *bits = c->reenc_bits.data();
  return int(c->reenc_pub.size());
}

int sm_set_object_embeddings(sm_ctx* c, const uint32_t* ids, const float* emb, int32_t n, int32_t dim) {
  if (!c || n < 0 || (n > 0 && (!ids || !emb)) || dim <= 0) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  for (int i = 0; i < n; ++i) {
    const ReencReq* r = nullptr;
    for (const ReencReq& q : c->reenc) if (q.id == ids[i]) { r = &q; break; }
    if (!r) continue;   // 요청 뒤 합쳐져 없어진 물체 등
    c->om.addWholeView(ids[i], emb + size_t(i) * dim, dim, r->kappa, c->st.last_image_stamp, r->cam);
  }
  return 0;
}

int sm_observe_object_name(sm_ctx* c, uint32_t id, const char* name, float log_lr) {
  if (!c || !name) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  int lab = -1;
  for (size_t i = 0; i < c->labels.size(); ++i) if (c->labels[i] == name) { lab = int(i); break; }
  if (lab < 0) return -2;
  bool found = false;
  for (const MapObject& o : c->om.objects()) found = found || (o.id == id && o.ap);
  if (!found) return -3;
  c->om.observeName(id, lab, log_lr);
  return 0;
}

int sm_get_objprob_stats(sm_ctx* c, int64_t out[16]) {
  if (!c || !out) return -1;
  std::lock_guard<std::mutex> g(c->mu);
  const ApStats& a = c->om.apStats();
  const long v[16] = {a.n_obs, a.n_wall, a.n_wall_name, a.n_ceil, a.n_floor, a.n_name_struct, a.n_assoc, a.n_new, a.n_merge, a.n_obj_struct,
                      a.n_reenc_req, a.n_reenc_done, a.n_through, a.n_so_big, a.n_wall_tall, a.n_blocked};
  for (int i = 0; i < 16; ++i) out[i] = v[i];
  return 0;
}

// ---- 실시간 스트림 ----
int sm_stream_joints(sm_ctx* c, double stamp, const float* q, int n) { return c && q && c->stream.running() && c->stream.pushJoints(stamp, q, n) ? 0 : -1; }
int sm_stream_start(sm_ctx* c, const char* host_port) { return c && host_port && c->stream.start(host_port) ? 0 : -1; }
void sm_stream_stop(sm_ctx* c) { if (c) c->stream.stop(); }

int sm_stream_view(sm_ctx* c) {
  if (!c || !c->stream.running()) return -1;
  const auto t_start = TClock::now();
  {
    // 바뀐 게 없으면 스냅숏도 안 만든다: 요약이 바뀌는 때는 영상(지도·물체)이 들어왔거나 그래프(로봇 궤적·방)가 새 사본이 되었을 때뿐
    std::lock_guard<std::mutex> g(c->mu);
    const auto gv = c->graph.publish();
    const uint64_t key = uint64_t(c->st.n_images) * 1000003ull + (gv ? gv->version : 0) * 7919ull + c->om.events().size();
    if (key == c->st_view_key && usBetween(c->st_last_view_t, TClock::now()) < 1000000) { c->st_views_skipped.fetch_add(1, std::memory_order_relaxed); return 0; }
    c->st_view_key = key;
  }
  sm_snapshot_t* s = nullptr;
  if (sm_snapshot(c, &s) != 0) return -1;
  SaveInput in;
  in.stamp = s->pose.stamp;
  in.pose[0] = s->pose.x; in.pose[1] = s->pose.y; in.pose[2] = s->pose.yaw;
  in.objs = s->objs.data();
  in.n_objs = int(s->objs.size());
  in.grid_res = s->res; in.grid_ox = s->ox; in.grid_oy = s->oy; in.grid_w = s->w; in.grid_h = s->h;
  in.views = s->views;
  in.movable = s->movable;
  in.clouds = s->clouds;
  in.voxel = s->voxel;
  in.rooms = s->rseg;
  in.room_names = s->rnames;
  in.graph = s->graph;
  in.stream_lite = true;
  inspMeta(s, &in.obj_meta);
  SaveOut out;
  out.png_ok.assign(s->objs.size(), 0);
  out.ply_ok.assign(s->objs.size(), 0);
  {
    std::lock_guard<std::mutex> g(c->mu);
    const auto& ev = c->om.events();
    in.events.assign(ev.end() - std::min<size_t>(ev.size(), 50), ev.end());
    for (size_t i = 0; i < s->objs.size(); ++i) {   // 파일이 이미 있는 것(지난 저장 기준)만 이미지·점 경로를 가리킨다
      out.png_ok[i] = c->saved_ver.count(s->objs[i].id) && s->views[i] ? 1 : 0;
      out.ply_ok[i] = c->saved_cloud_ver.count(s->objs[i].id) ? 1 : 0;
    }
  }
  const std::string json = sceneViewJson(in, out);
  sm_snapshot_release(s);
  // 바뀐 게 없으면 보내지 않는다(stamp·pose 는 맨 앞이라 제외하고 비교 — 자세는 pose 프레임이 따로). 그래도 1 s 에 한 번은 보낸다(stamp 갱신·새로 붙은 뷰어)
  const size_t k = json.find("\"grid\":");
  const std::string tail = k == std::string::npos ? json : json.substr(k);
  const auto now = TClock::now();
  if (tail == c->st_last_view && usBetween(c->st_last_view_t, now) < 1000000) return 0;
  c->st_last_view = tail;
  c->st_last_view_t = now;
  c->st_views_built.fetch_add(1, std::memory_order_relaxed);
  c->st_view_us.store(usBetween(t_start, TClock::now()), std::memory_order_relaxed);
  return c->stream.pushView(json) ? 0 : -1;
}

int sm_stream_get_stats(sm_ctx* c, sm_stream_stats* o) {
  if (!c || !o) return -1;
  const StreamStats t = c->stream.stats();
  *o = sm_stream_stats{t.frames_in, t.dropped, t.frames_sent, t.bytes_sent, t.reconnects, t.connected ? 1 : 0, float(c->st_view_us.load()), c->st_views_built.load(), c->st_views_skipped.load()};
  return 0;
}

}  // extern "C"
