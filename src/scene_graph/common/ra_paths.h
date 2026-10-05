// 저장소 안 기본 경로 — config/paths.env 와 같은 규칙.
//   모델·엔진: $OVDET_MODELS, 없으면 <robot-agent>/models/ovdet
//   라벨 표:   $RA_LABELS,   없으면 <robot-agent>/models/labels/objects-v1
// <robot-agent> 는 빌드 때 CMake 가 RA_ROOT_DIR 로 넣는다(없으면 $RA_ROOT, 그것도 없으면 ".").
#pragma once
#include <cstdlib>
#include <string>

namespace ra {
inline std::string root() {
  if (const char* e = std::getenv("RA_ROOT"); e && *e) return e;
#ifdef RA_ROOT_DIR
  return RA_ROOT_DIR;
#else
  return ".";
#endif
}
inline std::string models() {   // TensorRT 엔진·ONNX (예: models() + "/x86_sm120/...")
  if (const char* e = std::getenv("OVDET_MODELS"); e && *e) return e;
  return root() + "/models/ovdet";
}
inline std::string labels() {   // 라벨 표 폴더
  if (const char* e = std::getenv("RA_LABELS"); e && *e) return e;
  return root() + "/models/labels/objects-v1";
}
}  // namespace ra
