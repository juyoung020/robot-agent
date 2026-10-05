// 생성 파일 — training/RL/map_calib/percept/percept_header.py (손으로 고치지 말 것). GPU_MAP_PORT.md 2.2
// 원천: training/RL/map_calib/percept/percept_calib.json (sha256 02a6dbcb589f); 진짜 기록 40 개(sha256 0ff38caadda8bed1)
// 파이프라인: git 75ecd76aae, 엔진 yolo26n-seg-obj-416.plan, objprob yolo26n-seg-obj-416.json (b04b5ab273e4)
#pragma once
#ifndef PE_B0_V
#define PE_B0_V -1.106153f   // 검출 확률 로짓 절편: p = σ(b0 + b1·ln(px / px0) + 앞 상태 항), px = 보이는 화소(640 × 400)
#endif
#ifndef PE_B1_V
#define PE_B1_V 1.633399f   // 기울기(ln 화소)
#endif
#ifndef PE_PX0_V
#define PE_PX0_V 600.0f   // 화소 기준
#endif
#ifndef PE_LG_HIT_V
#define PE_LG_HIT_V 0.6134832f   // 지난 keyframe 에 검출됨 → 로짓 더함(진짜: 연이은 검출이 몰림)
#endif
#ifndef PE_LG_MISS_V
#define PE_LG_MISS_V -0.2153415f   // 지난 keyframe 에 놓침 → 로짓 더함
#endif
#ifndef PE_FRAG_MIN_V
#define PE_FRAG_MIN_V 0.8f   // 조각나는 가구: 수평 긴 변 ≥ 이 값(m)
#endif
#ifndef PE_P_FRAG2_V
#define PE_P_FRAG2_V 0.6f   // 그런 가구가 판에서 2 조각 이상으로 늘 나뉨(판·상자마다 고정)
#endif
#ifndef PE_P_FRAG3_V
#define PE_P_FRAG3_V 0.4f   // 2 조각 이상 중 3 조각(긴 변 ≥ 2·frag_min 일 때)
#endif
#ifndef PE_P_WHOLE_V
#define PE_P_WHOLE_V 0.3f   // 조각나는 가구가 이 keyframe 에 통째 한 마스크로 나옴
#endif
#ifndef PE_COS_PART_V
#define PE_COS_PART_V 0.7f   // 같은 가구의 다른 조각끼리 원형 생김새 cos(SigLIP 2 조각 잘라낸 모습이 다름)
#endif
#ifndef PE_WALL_SITE_V
#define PE_WALL_SITE_V 0.25f   // 벽 상자 1 m 토막이 판에서 유령 자리(창·문틀·액자·난간 모습)일 확률
#endif
#ifndef PE_WALL_DET_V
#define PE_WALL_DET_V 0.5f   // 유령 자리가 보이는 keyframe 마다 검출
#endif
#ifndef PE_PHANTOM_V
#define PE_PHANTOM_V 0.3f   // 가구가 판에서 어긋난 헛조각(정답 상자 밖 — 유령으로 셈)을 가짐
#endif
#ifndef PE_PHANTOM_DET_V
#define PE_PHANTOM_DET_V 0.4f   // 그 가구를 검출한 keyframe 마다 헛조각도 나옴
#endif
#ifndef PE_PHANTOM_D_V
#define PE_PHANTOM_D_V 0.6f   // 헛조각 어긋남(m, 방향은 판·상자마다 고정)
#endif
#ifndef PE_FURN_PX_V
#define PE_FURN_PX_V 3.0f   // 가구 보이는 화소 배율(맞은 광선 수 × 광선 넓이 × 이 값) — 작은 물체로 맞춘 검출 곡선이 큰 가구를 너무 낮게 봄(진짜 yolo26n 은 큰 가구를 거의 늘 찾음)
#endif
