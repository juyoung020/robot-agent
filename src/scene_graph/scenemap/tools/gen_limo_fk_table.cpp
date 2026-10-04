// LIMO + OMX-F URDF(map_vla.urdf) → C++ 순기구학 표 include/scenemap/limo_omx_fk_table.hpp.
//
//   gen_limo_fk_table <map_vla.urdf> <out.hpp>
//
// URDF 의 <joint> 를 읽어(작은 정규식 파서 — xacro 결과물만 받음) base_footprint 에서 각 목표 링크까지 사슬을 찾고, 관절마다
// 원 숫자(origin xyz·rpy, axis — URDF 문자열 그대로)와 proprio 번호를 쓴다. 계산은 fk.cpp 의 walk(R1 표와 같은 규칙):
//   t += R·xyz,  R = R·rpy(r, p, y),  회전 관절이면 R = R·회전(axis, q[번호]).
// 목표: 몸통 카메라 렌즈 광학(depth_camera_lens_optical_frame = depth_camera_link +x 0.010 m, OmniGibson eyes 와 같은 자리),
//       손목 카메라 광학(wrist_cam_optical_frame), 팔 끝(omx_end_effector_link).
// 사슬 안의 움직이는 관절은 아래 kQIndex 에만 있어야 한다(없으면 실패). 손으로 고치지 말고 이 도구로 다시 만든다.
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct J {
  std::string name, type, parent, child, xyz = "0 0 0", rpy = "0 0 0", axis = "1 0 0";
};

// proprio(LIMO 형식, scenemap.h SM_LIMO_*) 번호
const std::map<std::string, int> kQIndex = {{"omx_joint1", 6}, {"omx_joint2", 7}, {"omx_joint3", 8},
                                            {"omx_joint4", 9}, {"omx_joint5", 10}, {"omx_gripper_joint_1", 11}};

std::string attr(const std::string& tag, const char* key) {
  const std::regex re(std::string("\\b") + key + "\\s*=\\s*\"([^\"]*)\"");
  std::smatch m;
  return std::regex_search(tag, m, re) ? m[1].str() : std::string();
}

std::vector<double> nums(const std::string& s) {
  std::istringstream in(s);
  std::vector<double> v;
  for (double x; in >> x;) v.push_back(x);
  return v;
}

std::string lit(const std::string& s) {   // URDF 숫자 세 개 → "{a, b, c}"(원 문자열을 double 로 읽은 값, %.17g)
  const auto v = nums(s);
  if (v.size() != 3) throw std::runtime_error("need 3 numbers: '" + s + "'");
  char b[160];
  std::snprintf(b, sizeof b, "{%.17g, %.17g, %.17g}", v[0], v[1], v[2]);
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: gen_limo_fk_table <map_vla.urdf> <out.hpp>\n");
    return 2;
  }
  std::ifstream f(argv[1]);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string x = ss.str();
  x = std::regex_replace(x, std::regex("<!--[\\s\\S]*?-->"), "");   // 주석 버림
  std::map<std::string, J> by_child;
  const std::regex jre("<joint\\b([^>]*)>([\\s\\S]*?)</joint>");
  for (auto it = std::sregex_iterator(x.begin(), x.end(), jre); it != std::sregex_iterator(); ++it) {
    const std::string head = (*it)[1].str(), body = (*it)[2].str();
    J j;
    j.name = attr(head, "name");
    j.type = attr(head, "type");
    std::smatch m;
    if (std::regex_search(body, m, std::regex("<parent\\b[^>]*>"))) j.parent = attr(m[0].str(), "link");
    if (std::regex_search(body, m, std::regex("<child\\b[^>]*>"))) j.child = attr(m[0].str(), "link");
    if (std::regex_search(body, m, std::regex("<origin\\b[^>]*>"))) {
      const std::string o = m[0].str();
      if (!attr(o, "xyz").empty()) j.xyz = attr(o, "xyz");
      if (!attr(o, "rpy").empty()) j.rpy = attr(o, "rpy");
    }
    if (std::regex_search(body, m, std::regex("<axis\\b[^>]*>"))) j.axis = attr(m[0].str(), "xyz");
    if (j.child.empty() || j.parent.empty()) { std::fprintf(stderr, "joint %s: no parent/child\n", j.name.c_str()); return 1; }
    by_child[j.child] = j;
  }
  const char* root = "base_footprint";
  struct Target { const char* var; const char* link; const char* what; };
  const Target targets[] = {{"depth_cam", "depth_camera_lens_optical_frame", "몸통 카메라 렌즈 광학(cam 0, depth_camera_link +x 0.010 m, z 앞·x 오른쪽·y 아래)"},
                            {"wrist_cam", "wrist_cam_optical_frame", "손목 카메라 광학(cam 1, 깊이 없음)"},
                            {"eef", "omx_end_effector_link", "팔 끝(잡기 점) — 관절 원점이 팔 뼈대"}};
  std::ostringstream o;
  o << "// 자동 생성: tools/gen_limo_fk_table.cpp ← map_vla.urdf(LIMO + OMX-F). 손으로 고치지 말 것.\n"
    << "// 다시 만들기: gen_limo_fk_table ~/ra_ws/map_vla.urdf include/scenemap/limo_omx_fk_table.hpp\n"
    << "// 사슬 시작 = base_footprint(바닥, scenemap 의 '베이스' 프레임). q = LIMO proprio 번호(scenemap.h SM_LIMO_*), -1 = 고정.\n"
    << "#pragma once\n\nnamespace scenemap::limo {\n\n"
    << "struct JointDef { int kind; double xyz[3], rpy[3], axis[3]; int q; const char* name; };\n"
    << "struct ChainDef { int n; const JointDef* j; double cam_xyz[3], cam_xyzw[4]; };\n\n";
  for (const Target& t : targets) {
    std::vector<J> chain;
    for (std::string l = t.link; l != root;) {
      auto it = by_child.find(l);
      if (it == by_child.end()) { std::fprintf(stderr, "no path %s → %s (stuck at %s)\n", root, t.link, l.c_str()); return 1; }
      chain.insert(chain.begin(), it->second);
      l = it->second.parent;
    }
    o << "// " << root << " → " << t.link << ": " << t.what << "\n";
    o << "inline constexpr JointDef k_" << t.var << "[] = {\n";
    for (const J& j : chain) {
      int kind = 0, q = -1;
      if (j.type == "revolute" || j.type == "continuous") kind = 1;
      else if (j.type == "prismatic") kind = 2;
      else if (j.type != "fixed") { std::fprintf(stderr, "joint %s: type %s\n", j.name.c_str(), j.type.c_str()); return 1; }
      if (kind) {
        auto qi = kQIndex.find(j.name);
        if (qi == kQIndex.end()) { std::fprintf(stderr, "movable joint %s has no proprio index\n", j.name.c_str()); return 1; }
        q = qi->second;
      }
      o << "  {" << kind << ", " << lit(j.xyz) << ", " << lit(j.rpy) << ", " << lit(kind ? j.axis : "0 0 0") << ", " << q << ", \""
        << j.name << "\"},\n";
    }
    o << "};\n";
    o << "inline constexpr ChainDef k_" << t.var << "_chain = {" << chain.size() << ", k_" << t.var << ", {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};\n\n";
  }
  o << "}  // namespace scenemap::limo\n";
  std::ofstream out(argv[2], std::ios::binary);
  out << o.str();
  if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
  std::printf("wrote %s\n", argv[2]);
  return 0;
}
