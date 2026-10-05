/* slam_carto — Cartographer(2D, ROS 없이) 를 자세 원천으로 쓰는 작은 C ABI.
 *
 * 넣는 것: 2D 레이저 스캔(라이다 기준 거리 배열) + 바퀴 오도메트리 자세(+ 있으면 IMU).
 * 내는 것: 아무 시각 t 의 map ← base 자세(x, y, yaw). keyframe(영상) 시각마다 sc_pose_at 으로 묻는다.
 *   자세(t) = 전역 ← 지역(최근 최적화) ∘ 지역 스캔 맞춤 자세(t_s) ∘ 오도메트리(t_s)⁻¹ ∘ 오도메트리(t)
 *   — t_s 는 t 이하 가장 최근 스캔 맞춤. 오도메트리가 없으면 맞춘 자세 그대로.
 * map 원점 = 첫 sc_push_odom 때 베이스(오도메트리가 없으면 첫 스캔 때 베이스). 시각은 초(double, 단조 증가). 한 스레드에서 부른다
 * (Cartographer 안의 전역 최적화는 자기 스레드에서 돈다).
 * 설정은 lua(config 폴더의 .lua). 찾는 순서: sc_create 의 config_dir → SLAM_CARTO_CONFIG_DIR → 빌드 때 박은 소스 config/ →
 * Cartographer 설치 configuration_files.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_ctx sc_ctx;

typedef struct {
  const char* config_dir;       /* NULL = 기본 */
  const char* config_name;      /* lua 파일 이름, NULL = "limo_x2l.lua" */
  double laser_xyz[3];          /* base ← laser 평행 이동(m) */
  double laser_yaw;             /* base ← laser 회전(rad, 수평 장착) */
  int32_t use_odom;             /* 1 = 바퀴 오도메트리 넣기(기본) */
  int32_t use_imu;              /* 1 = IMU 넣기(lua use_imu_data 를 켬) */
} sc_config;

sc_config sc_default_config(void);    /* LIMO X2L: laser (0.103, 0, -0.034), yaw 0 */
sc_ctx* sc_create(const sc_config* cfg, char* err, int errlen);
void sc_destroy(sc_ctx*);

/* 바퀴 오도메트리: odom 프레임의 베이스 자세(누적). 0 성공 */
int sc_push_odom(sc_ctx*, double t, double x, double y, double yaw);
/* IMU(베이스 기준): 가속도 m/s², 각속도 rad/s */
int sc_push_imu(sc_ctx*, double t, const double acc[3], const double gyr[3]);
/* 스캔 하나: t = 마지막 광선 시각. ranges[i] 방향 = angle_min + i·angle_inc(라이다 기준), 시각 = t − (n−1−i)·time_inc.
 * range_min..range_max 밖·NaN·inf 는 버림. 0 성공 */
int sc_push_scan(sc_ctx*, double t, int32_t n, const float* ranges, double angle_min, double angle_inc, double time_inc,
                 double range_min, double range_max);

/* 시각 t 의 map ← base 자세. 아직 스캔 맞춤이 없으면 -1 */
int sc_pose_at(sc_ctx*, double t, double out_xyyaw[3]);
/* 가장 최근 스캔 맞춤의 시각·자세(전역). 없으면 -1 */
int sc_last_pose(sc_ctx*, double* t, double out_xyyaw[3]);

typedef struct {
  int64_t n_scans, n_odom, n_imu;
  int64_t n_local;              /* 지역 맞춤 결과 수 */
  int64_t n_nodes;              /* 지도에 들어간 노드 수 */
  int64_t n_submaps;
  int64_t n_loop_constraints;   /* 서로 다른 submap 사이 제약(되돌아옴) */
  double us_scan_mean;          /* sc_push_scan 평균 µs(호출 스레드) */
  double us_scan_max;
} sc_stats;
int sc_get_stats(sc_ctx*, sc_stats* out);

/* 끝: 궤적을 끝내고 전역 최적화를 마지막으로 돌림(이후 sc_pose_at 은 최적화된 값) */
int sc_finish(sc_ctx*);
/* 최적화된 노드 궤적: 시각, x, y, yaw 를 cap 개까지. 개수 반환 */
int sc_nodes(sc_ctx*, double* t, double* xyyaw, int32_t cap);
/* 확률 격자를 PGM(map_server 형식, 0 점유 · 254 빈칸 · 205 모름) + yaml 로. res = 칸 크기(m) */
int sc_write_map(sc_ctx*, const char* pgm_path, double res);

#ifdef __cplusplus
}
#endif
