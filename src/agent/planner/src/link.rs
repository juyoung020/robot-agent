//! 평가기 연결(link): 평가기 프로세스 안 파이썬 접착부 ↔ [link] — 계획기 호스트 + 관측 내보내기.
//!
//! VLA 가 평가기 프로세스 안 정책으로 들어가면 웹소켓 중계기([`crate::relay`]) 자리가 없다.
//! 대신 평가기 안 접착부(`src/sim/integ/glue/simlink_policy.py`)가 이 연결에 스텝마다 요약을 보내고, 경계에서 결정
//! (단계 문장 또는 단계 번호)을 받아 VLA 에 넣는다. 계획기의 Core·Decider·경계 감시는 그대로 쓰고 전송층만 새로 짰다.
//!
//! 한 연결 = TCP 하나(Windows 평가기 → WSL). 메시지 = 머리 12 B(`magic u32, type u16, flags u16, len u32`, 리틀 엔디언) + 몸통.
//!
//! | 방향 | type | 몸통 |
//! |---|---|---|
//! | → | HELLO(1) | JSON [`Hello`]: 과제·환경 수·카메라(크기·K)·cam_rel_poses 순서·단계 수 |
//! | ← | HELLO_ACK(0x81) | JSON: `hold`, `want`(첫 스텝) |
//! | → | RESET(2) | 없음(새 판) |
//! | → | STEP(3) | [`StepHead`] 32 B + proprio f32 × n + cam_rel_poses f32 × n + 영상 머리 16 B × k + 영상 바이트 |
//! | ← | ACK(0x83) | step u64, hold_next u8, has_decision u8, want_next u16, json_len u32 + 결정 JSON |
//! | → | BYE(4) | 없음 |
//!
//! 매 스텝 경로(병목 제로): 접착부는 STEP 을 보내고 **답을 기다리지 않는다**. 다음 스텝 첫머리에 전 스텝의 ACK(이미
//! 도착해 있음)를 읽어 `hold_next`·`want_next` 를 안다. 파이썬 쪽 비용 = 작은 머리 쓰기 + 버퍼에서 ACK 읽기.
//! - `hold_next` = 다음 스텝이 경계일 수 있다(판 시작·예산·정기 확인이 올 스텝, 또는 이번 스텝에 사건 경계가 났다).
//!   접착부는 그 스텝을 WAIT 로 보내고 결정이 올 때까지 기다린다 → 시뮬레이터 시간 정지(점수 영향 없음).
//!   사건 경계(이동 멈춤·그리퍼)가 1스텝 늦게 반영되는 것은 중계기와 같다.
//! - `want_next` = 다음 스텝에 보낼 카메라(비트: 카메라 c 의 RGB = 1<<2c, 깊이 = 1<<2c+1; c = 0 머리, 1 왼손목,
//!   2 오른손목). keyframe 정책은 여기(Rust)에 있다: 머리는 카메라가 움직였거나(5 cm·3°) 몇 스텝 지났을 때, 경계 스텝은
//!   전부(LLM 영상 + 그래프 갱신). 영상은 평가기 텐서 그대로(RGBA u8, 깊이 f32 m) 오고 변환은 받는 쪽이 한다.
//!
//! 위치: [`crate::pose::PoseEstimator`] 하나가 계획기(`BoundaryEvent.pose`)와 관측 내보내기(`ObsPacket.base`)에 같은
//! 값을 준다. 바깥 추정기(scenemap slam2d)의 자세는 [`ObsSink::external_fix`] 로 받아 보정으로 바꾼다.
//!
//! 관측 내보내기는 [`ObsSink`] 뒤에 숨긴다. 이 크레이트는 C/C++ 를 링크하지 않는다: WSL 통합 노드(`src/sim/integ/simlink`)가
//! scenemap C ABI(같은 프로세스)에 넣는 구현을 넣는다. 시험·단독 실행은 [`NullSink`].
//!
//! 시각: 모든 stamp 는 **시뮬 시각**(초, 판 시작 = 0, 스텝 k = k / hz)이다. 경계에서 시뮬이 멈추면 시간도 멈춘다
//! (docs/scenemap_설계.md 2절 "시간 규칙 = stamp"). 영상 k 의 stamp 는 k-1(렌더가 한 스텝 늦음).

use crate::catalog::{Catalog, TaskCard};
use crate::image::Rgb;
use crate::monitor::{MonitorCfg, Trigger};
use crate::odom::Pose;
use crate::planner::{Agent, Core, Decision};
use crate::pose::{self, Mat3, PoseEstimator};
use crate::session::EnvSession;
use crate::setup::AgentFactory;
use crate::trace::Tracer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MAGIC: u32 = u32::from_le_bytes(*b"BLNK");
pub const T_HELLO: u16 = 1;
pub const T_RESET: u16 = 2;
pub const T_STEP: u16 = 3;
pub const T_BYE: u16 = 4;
pub const T_HELLO_ACK: u16 = 0x81;
pub const T_ACK: u16 = 0x83;
/// STEP flags: 접착부가 이 스텝의 ACK(결정)를 기다린다
pub const F_WAIT: u32 = 1;

pub const KIND_RGBA8: u8 = 0;
pub const KIND_RGB8: u8 = 1;
pub const KIND_DEPTH_F32: u8 = 2;
pub const KIND_DEPTH_U16MM: u8 = 3;

pub const CAM_HEAD: u8 = 0;
pub const CAM_LEFT: u8 = 1;
pub const CAM_RIGHT: u8 = 2;
pub const CAM_NAMES: [&str; 3] = ["head", "left_wrist", "right_wrist"];

pub const fn want_rgb(cam: u8) -> u16 {
    1 << (2 * cam)
}
pub const fn want_depth(cam: u8) -> u16 {
    1 << (2 * cam + 1)
}
pub const WANT_HEAD: u16 = want_rgb(CAM_HEAD) | want_depth(CAM_HEAD);
pub const WANT_ALL: u16 = 0x3F;
pub const WANT_ALL_RGB: u16 = want_rgb(0) | want_rgb(1) | want_rgb(2);

/// proprio 안 위치 (OmniGibson `eval/utils/eval_utils.py` PROPRIOCEPTION_INDICES["R1Pro"], [`crate::wire`] 와 같음)
const BASE_QVEL: usize = crate::wire::BASE_QVEL;
const GRIP_LEFT: usize = crate::wire::GRIP_LEFT;
const GRIP_RIGHT: usize = crate::wire::GRIP_RIGHT;
/// 팔 끝 자세 시작 번호(PROPRIOCEPTION_INDICES["R1Pro"]: eef_left_pos 17:20 + eef_left_quat 20:24, eef_right_pos 42:45 + eef_right_quat 45:49)
pub const EEF_LEFT: usize = 17;
pub const EEF_RIGHT: usize = 42;

/// proprio → (팔 끝 자세 [왼, 오른] xyz + xyzw, 그리퍼 [왼, 오른] 두 손가락 합). 평가기 `get_relative_eef_pose`(베이스 기준) 값 그대로.
pub fn eef_grip(proprio: &[f32]) -> ([[f64; 7]; 2], [f64; 2]) {
    let p = |i: usize| proprio.get(i).copied().unwrap_or(0.0) as f64;
    let e = |o: usize| -> [f64; 7] { std::array::from_fn(|k| p(o + k)) };
    ([e(EEF_LEFT), e(EEF_RIGHT)], [p(GRIP_LEFT) + p(GRIP_LEFT + 1), p(GRIP_RIGHT) + p(GRIP_RIGHT + 1)])
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CamSpec {
    /// head | left_wrist | right_wrist
    pub name: String,
    pub w: u32,
    pub h: u32,
    /// fx, fy, cx, cy (평가기 `eval_utils.CAMERA_INTRINSICS`, 그 해상도 기준)
    pub k: [f64; 4],
}

fn one() -> usize {
    1
}
fn thirty() -> f64 {
    30.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Hello {
    pub task: String,
    #[serde(default)]
    pub task_id: Option<i64>,
    #[serde(default)]
    pub instance: Option<i64>,
    #[serde(default = "one")]
    pub num_envs: usize,
    #[serde(default = "thirty")]
    pub hz: f64,
    #[serde(default)]
    pub max_steps: Option<u64>,
    /// 카메라(이름으로 머리·손목을 가림)
    #[serde(default)]
    pub cams: Vec<CamSpec>,
    /// `cam_rel_poses` 7개 묶음의 카메라 순서(평가기 robot_camera_names 순서, 보통 left_wrist, right_wrist, head)
    #[serde(default)]
    pub crp_order: Vec<String>,
    /// VLA 단계 입력의 이 과제 단계 수(2025 1위 모델 표). 0 = 단계 입력 없음
    #[serde(default)]
    pub stage_count: u32,
    /// VLA 가 지금 쓰는 과제 문장
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub client: String,
}

/// STEP 몸통 머리(32 B, 리틀 엔디언).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepHead {
    pub step: u64,
    /// 접착부 시계(초, 기록용 — 두 시계는 다르다)
    pub t_client: f64,
    pub flags: u32,
    pub env: u16,
    pub n_proprio: u16,
    pub n_crp: u16,
    pub n_frames: u16,
    pub _pad: u32,
}

impl StepHead {
    pub const LEN: usize = 32;
    pub fn encode(&self) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&self.step.to_le_bytes());
        b[8..16].copy_from_slice(&self.t_client.to_le_bytes());
        b[16..20].copy_from_slice(&self.flags.to_le_bytes());
        b[20..22].copy_from_slice(&self.env.to_le_bytes());
        b[22..24].copy_from_slice(&self.n_proprio.to_le_bytes());
        b[24..26].copy_from_slice(&self.n_crp.to_le_bytes());
        b[26..28].copy_from_slice(&self.n_frames.to_le_bytes());
        b
    }
    pub fn decode(b: &[u8]) -> StepHead {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        StepHead {
            step: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            t_client: f64::from_le_bytes(b[8..16].try_into().unwrap()),
            flags: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            env: u16_at(20),
            n_proprio: u16_at(22),
            n_crp: u16_at(24),
            n_frames: u16_at(26),
            _pad: 0,
        }
    }
}

/// 영상 머리(16 B): cam u8, kind u8, 0 u16, h u32, w u32, nbytes u32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameHead {
    pub cam: u8,
    pub kind: u8,
    pub h: u32,
    pub w: u32,
    pub nbytes: u32,
}

impl FrameHead {
    pub const LEN: usize = 16;
    pub fn encode(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0] = self.cam;
        b[1] = self.kind;
        b[4..8].copy_from_slice(&self.h.to_le_bytes());
        b[8..12].copy_from_slice(&self.w.to_le_bytes());
        b[12..16].copy_from_slice(&self.nbytes.to_le_bytes());
        b
    }
    pub fn decode(b: &[u8]) -> FrameHead {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        FrameHead { cam: b[0], kind: b[1], h: u32_at(4), w: u32_at(8), nbytes: u32_at(12) }
    }
    pub fn bytes_per_px(kind: u8) -> u32 {
        match kind {
            KIND_RGBA8 | KIND_DEPTH_F32 => 4,
            KIND_RGB8 => 3,
            KIND_DEPTH_U16MM => 2,
            _ => 0,
        }
    }
}

/// 받은 영상(소유).
#[derive(Debug, Clone)]
pub struct Frame {
    pub cam: u8,
    pub kind: u8,
    pub h: u32,
    pub w: u32,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn is_rgb(&self) -> bool {
        self.kind == KIND_RGBA8 || self.kind == KIND_RGB8
    }
    /// RGB(3채널) 복사. 단계 경계(LLM 영상)에서만.
    pub fn to_rgb(&self) -> Option<Rgb> {
        let (w, h) = (self.w as usize, self.h as usize);
        match self.kind {
            KIND_RGB8 => Some(Rgb { w, h, px: self.data.clone() }),
            KIND_RGBA8 => Some(Rgb { w, h, px: self.data.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect() }),
            _ => None,
        }
    }
    /// RGB 가 전부 0 인가(검은 프레임, 평가기_가속설계 (A)(B))
    pub fn is_black(&self) -> bool {
        match self.kind {
            KIND_RGBA8 => self.data.chunks_exact(4).all(|c| c[0] == 0 && c[1] == 0 && c[2] == 0),
            KIND_RGB8 => self.data.iter().all(|&b| b == 0),
            _ => false,
        }
    }
}

/// 관측 내보내기로 넘기는 한 스텝(영상이 없어도 자세는 매 스텝).
#[derive(Debug, Clone)]
pub struct ObsPacket {
    pub env: usize,
    pub episode: u32,
    pub step: u64,
    /// link 가 이 스텝을 다 받은 순간
    pub recv: Instant,
    /// 이 스텝 **상태**(proprio·자세)의 시뮬 시각 [s] = step / hz
    pub t: f64,
    /// 이 스텝 **영상**의 stamp = 직전 스텝의 stamp. 평가기 관측 영상(스텝 k)은 스텝 k-1 끝의 장면으로 그려지고(렌더가 한 스텝
    /// 늦음, render 에이전트 확인: 깊이 1~2 ulp 일치), proprio 는 스텝 k 뒤 값이다. 그래서 영상은 k-1 의 자세·robot2cam 과
    /// 짝지어야 한다 — 같은 stamp 로 이미 k-1 에 낸 자세와 맞물린다. 판 첫 스텝은 직전이 없어 자기 stamp.
    pub img_t: f64,
    /// 영상 시각(k-1)의 베이스 자세·카메라 외부 자세
    pub img_base: Pose,
    pub img_cam_rel: [Option<[f64; 7]>; 3],
    /// 추정 베이스 자세(map)
    pub base: Pose,
    /// 이 스텝 proprio 원값(61) — scenemap `sm_push_proprio`(순기구학·팔 끝·그리퍼는 scenemap 이 여기서 계산)
    pub proprio: Vec<f32>,
    /// 이 스텝의 base_qvel 원값(로봇 기준 vx, vy, wz)
    pub qvel: [f64; 3],
    /// 이 스텝 팔 끝 자세 [왼, 오른](베이스 기준 xyz + xyzw, proprio 17:24·42:49)
    pub eef: [[f64; 7]; 2],
    /// 이 스텝 그리퍼 벌어짐 [왼, 오른](두 손가락 합, proprio 24:26·49:51)
    pub grip: [f64; 2],
    /// 카메라별(0 머리, 1 왼손목, 2 오른손목) 베이스 기준 자세 xyz + xyzw(= robot2cam, proprio 순기구학). 모르면 None
    pub cam_rel: [Option<[f64; 7]>; 3],
    pub frames: Vec<Frame>,
    /// 이 스텝에서 계획기가 결정한다(그래프가 이 프레임을 반영하길 기다릴 수 있다)
    pub boundary: bool,
}

/// 관측 내보내기(scenemap 등). `push` 는 막지 않아야 한다(매 스텝 경로).
pub trait ObsSink: Send {
    fn hello(&mut self, _h: &Hello) {}
    fn reset(&mut self, _episode: u32) {}
    fn push(&mut self, pkt: ObsPacket);
    /// 경계에서 결정 전에: 그래프가 `step` 의 프레임을 반영할 때까지(또는 timeout) 기다린다. 기다린 ms.
    fn settle(&mut self, _env: usize, _step: u64, _timeout: Duration) -> Option<f64> {
        None
    }
    /// 바깥 위치 추정기(scenemap slam2d)의 새 절대 자세 (그 시뮬 시각, 자세). 새 값이 있을 때만 Some.
    /// 시각은 패킷의 `t`·`img_t` 와 같은 시계다.
    fn external_fix(&mut self, _env: usize) -> Option<(f64, Pose)> {
        None
    }
    fn stats(&mut self) -> Value {
        Value::Null
    }
}

/// 아무 데도 안 내보내고 세기만 한다(시험·단독 실행).
#[derive(Default)]
pub struct NullSink {
    pub packets: u64,
    pub frames: u64,
    pub bytes: u64,
    pub black: u64,
    pub last: Option<ObsPacket>,
    pub keep_last: bool,
}

impl ObsSink for NullSink {
    fn push(&mut self, pkt: ObsPacket) {
        self.packets += 1;
        self.frames += pkt.frames.len() as u64;
        self.bytes += pkt.frames.iter().map(|f| f.data.len() as u64).sum::<u64>();
        self.black += pkt.frames.iter().filter(|f| f.is_rgb() && f.is_black()).count() as u64;
        if self.keep_last {
            self.last = Some(pkt);
        }
    }
    fn stats(&mut self) -> Value {
        json!({"packets": self.packets, "frames": self.frames, "bytes": self.bytes, "black_rgb_frames": self.black})
    }
}

/// VLA 에 무엇을 넣나(plan.md 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptMode {
    /// (다) 과제 단위: VLA 는 학습된 과제 문장 그대로. 계획기 결정은 단계 번호·기억·복구로만
    Task,
    /// 계획기의 단계 문장을 VLA 문장으로(중계기 agent 모드와 같음)
    Subtask,
}

/// 단계 번호 통로(plan.md 4.1 (라)): 안쪽 정책의 `set_stage(env, stage, fixed)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageMode {
    /// 안 보냄
    Off,
    /// 모델 자체 투표(mode 0). 계획기 추정은 기록에만(비교용)
    Vote,
    /// 계획기 추정을 고정 입력으로(mode 1, 투표 끔)
    External,
}

impl StageMode {
    pub fn parse(s: &str) -> Option<StageMode> {
        match s {
            "off" => Some(StageMode::Off),
            "vote" => Some(StageMode::Vote),
            "external" | "ext" => Some(StageMode::External),
            _ => None,
        }
    }
}

/// keyframe 정책(무엇을 언제 보내 달라고 할지).
#[derive(Debug, Clone, Serialize)]
pub struct KeyCfg {
    /// 머리 RGB-D 를 보낼 최대 간격(스텝). 0 = 매 스텝
    pub head_max_gap: u64,
    /// 카메라가 이만큼 움직이면 간격 전이라도 보냄
    pub move_m: f64,
    pub turn_deg: f64,
    /// 손목 RGB-D 간격(0 = 경계 스텝에서만)
    pub wrist_every: u64,
    /// 머리 깊이를 보낼까(scenemap 입력)
    pub head_depth: bool,
    /// 경계 스텝에 모든 카메라 RGB-D
    pub boundary_all: bool,
}

impl Default for KeyCfg {
    fn default() -> Self {
        KeyCfg { head_max_gap: 6, move_m: 0.05, turn_deg: 3.0, wrist_every: 0, head_depth: true, boundary_all: true }
    }
}

#[derive(Debug, Clone, Default)]
struct KeyState {
    last_head: Option<u64>,
    last_head_pose: Option<(Mat3, [f64; 3])>,
    last_wrist: Option<u64>,
}

impl KeyState {
    fn got(&mut self, step: u64, frames: &[Frame], head_pose: Option<(Mat3, [f64; 3])>) {
        if frames.iter().any(|f| f.cam == CAM_HEAD && f.is_rgb()) {
            self.last_head = Some(step);
            self.last_head_pose = head_pose;
        }
        if frames.iter().any(|f| f.cam != CAM_HEAD && f.is_rgb()) {
            self.last_wrist = Some(step);
        }
    }

    /// 다음 스텝(next)에 요청할 영상.
    fn want(&self, c: &KeyCfg, next: u64, head_now: Option<&(Mat3, [f64; 3])>, hold_next: bool) -> u16 {
        if hold_next && c.boundary_all {
            return WANT_ALL;
        }
        let head_bits = if c.head_depth { WANT_HEAD } else { want_rgb(CAM_HEAD) };
        let mut w = 0u16;
        let due = match self.last_head {
            None => true,
            Some(l) => c.head_max_gap == 0 || next.saturating_sub(l) >= c.head_max_gap,
        };
        let moved = match (head_now, &self.last_head_pose) {
            (Some((r, t)), Some((r0, t0))) => {
                let d = ((t[0] - t0[0]).powi(2) + (t[1] - t0[1]).powi(2) + (t[2] - t0[2]).powi(2)).sqrt();
                d >= c.move_m || pose::rot_angle(r, r0).to_degrees() >= c.turn_deg
            }
            _ => false,
        };
        if due || moved {
            w |= head_bits;
        }
        if c.wrist_every > 0 && self.last_wrist.map(|l| next.saturating_sub(l) >= c.wrist_every).unwrap_or(true) {
            w |= want_rgb(CAM_LEFT) | want_depth(CAM_LEFT) | want_rgb(CAM_RIGHT) | want_depth(CAM_RIGHT);
        }
        w
    }
}

/// 단계 번호 추정: 계획 진행률(시연 단계 p50 길이 가중) × 단계 수.
/// 2025 1위 모델의 단계는 시연 시간을 과제마다 5~15 등분한 것이다(`refs/behavior-1k-solution` README "solely based on
/// the timestamps"). 그래서 진행률도 "시연에서 그 단계까지 걸린 시간 비율"로 잰다.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default)]
pub struct StageTrack {
    /// 지금 단계 앞 단계들의 무게 합(스텝)
    pub before: f64,
    /// 지금 단계 무게(p50 스텝)
    pub cur: f64,
    pub total: f64,
    /// 지금 단계를 시작한 스텝
    pub start: u64,
    pub finished: bool,
    /// 지금 계획 단계 id(같은 단계를 계속·재시도하면 걸린 스텝을 이어 센다)
    pub cur_id: Option<u32>,
}

impl StageTrack {
    pub fn from_core(core: &Core, catalog: &Catalog, step: u64) -> StageTrack {
        let w = |skill: &str| -> f64 {
            catalog
                .skills
                .iter()
                .find(|s| s.name == skill)
                .map(|s| s.p50_steps as f64)
                .unwrap_or_else(|| core.budget_for(skill) as f64 / 1.5)
                .max(30.0)
        };
        let steps = &core.plan.steps;
        let total: f64 = steps.iter().map(|s| w(&s.skill)).sum();
        let i = steps
            .iter()
            .position(|s| s.status == crate::plan::Status::Doing)
            .or_else(|| steps.iter().position(|s| s.status == crate::plan::Status::Todo))
            .unwrap_or(steps.len());
        let before: f64 = steps[..i].iter().map(|s| w(&s.skill)).sum();
        let cur = steps.get(i).map(|s| w(&s.skill)).unwrap_or(0.0);
        StageTrack { before, cur, total, start: step, finished: core.finished || i >= steps.len(), cur_id: steps.get(i).map(|s| s.id) }
    }

    /// 새 결정 뒤 추적. 같은 계획 단계가 이어지면(계속·재시도) 그 단계 시작 스텝을 유지한다.
    pub fn continued_from(mut self, old: &StageTrack) -> StageTrack {
        if self.cur_id.is_some() && self.cur_id == old.cur_id && !self.finished {
            self.start = old.start;
        }
        self
    }

    /// 진행률 0..=1 (지금 단계 안에서는 걸린 스텝만큼, 그 단계 p50 까지)
    pub fn frac(&self, step: u64) -> f64 {
        if self.finished || self.total <= 0.0 {
            return if self.total <= 0.0 { 0.0 } else { 1.0 };
        }
        let inside = (step.saturating_sub(self.start) as f64).min(self.cur);
        ((self.before + inside) / self.total).clamp(0.0, 1.0)
    }

    pub fn stage(&self, step: u64, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        ((self.frac(step) * n as f64).floor() as u32).min(n - 1)
    }
}

#[derive(Clone)]
pub struct LinkCfg {
    pub listen: String,
    pub catalog: Arc<Catalog>,
    /// None = 계획기 없이 관측 전달만
    pub factory: Option<AgentFactory>,
    pub monitor: MonitorCfg,
    /// 위치 추정기 이름([`pose::make`])
    pub pose: String,
    /// 카메라 외부 자세 = proprio 순기구학(기본). None 이면 접착부가 보낸 cam_rel_poses(시험·비교용)
    pub fk: Option<Arc<crate::fk::CamFk>>,
    pub prompt_mode: PromptMode,
    pub stage: StageMode,
    pub key: KeyCfg,
    /// 경계에서 LLM 에 영상
    pub images: bool,
    pub image_side: usize,
    pub jpeg_quality: u8,
    /// 경계에서 그래프가 그 프레임을 반영하길 기다리는 최대 시간
    pub settle_ms: u64,
    pub trace_dir: Option<PathBuf>,
    pub task_override: Option<String>,
    pub max_steps_override: Option<u64>,
    /// 연결 하나만 받고 끝(시험·한 판)
    pub once: bool,
    /// 이 스텝마다 지연 통계 기록
    pub stats_every: u64,
}

impl LinkCfg {
    pub fn new(listen: &str, catalog: Arc<Catalog>) -> LinkCfg {
        LinkCfg {
            listen: listen.into(),
            catalog,
            factory: None,
            monitor: MonitorCfg::default(),
            pose: "integrate".into(),
            fk: crate::fk::CamFk::load(&crate::fk::CamFk::default_path()).ok().map(Arc::new),
            prompt_mode: PromptMode::Task,
            stage: StageMode::External,
            key: KeyCfg::default(),
            images: true,
            image_side: 448,
            jpeg_quality: 80,
            settle_ms: 400,
            trace_dir: None,
            task_override: None,
            max_steps_override: None,
            once: false,
            stats_every: 1000,
        }
    }
}

/// 명령줄 → 설정(`bagent link` 와 WSL 통합 노드 simlink 가 같이 쓴다).
/// 계획기 인자(--llm --graph --decider --format …)는 relay 와 같다. `--no-planner` 면 관측 전달만.
pub fn cfg_from_args(a: &crate::util::Args, catalog: Arc<Catalog>) -> Result<LinkCfg, String> {
    let mut c = LinkCfg::new(&a.str_or("listen", "0.0.0.0:7801"), catalog.clone());
    if !a.flag("no-planner") {
        let pcfg = crate::setup::planner_cfg(a)?;
        if pcfg.format == crate::instruction::Format::Metric && !a.flag("allow-metric-experiment") {
            return Err("link 에서 --format metric 은 쓰지 않는다(숫자 명령 금지). 실험이면 --allow-metric-experiment".into());
        }
        c.image_side = pcfg.image_side;
        c.jpeg_quality = pcfg.jpeg_quality;
        c.images = pcfg.send_images;
        c.factory = Some(crate::setup::agent_factory(a, catalog)?);
    }
    c.pose = a.str_or("pose", "integrate");
    pose::make(&c.pose, 30.0)?;
    c.fk = match a.str_or("cam-pose", "fk").as_str() {
        "fk" => Some(Arc::new(crate::fk::CamFk::load(&a.get("fk").map(PathBuf::from).unwrap_or_else(crate::fk::CamFk::default_path))?)),
        "obs" => None,
        o => return Err(format!("--cam-pose 는 fk|obs: {o}")),
    };
    c.prompt_mode = match a.str_or("prompt-mode", "task").as_str() {
        "task" => PromptMode::Task,
        "subtask" => PromptMode::Subtask,
        o => return Err(format!("--prompt-mode 는 task|subtask: {o}")),
    };
    c.stage = StageMode::parse(&a.str_or("stage", "external")).ok_or("--stage 는 external|vote|off")?;
    c.key.head_max_gap = a.num("head-gap", c.key.head_max_gap);
    c.key.move_m = a.num("key-move-m", c.key.move_m);
    c.key.turn_deg = a.num("key-turn-deg", c.key.turn_deg);
    c.key.wrist_every = a.num("wrist-every", c.key.wrist_every);
    c.key.head_depth = !a.flag("no-head-depth");
    c.settle_ms = a.num("settle-ms", c.settle_ms);
    c.task_override = a.get("task").map(|s| s.to_string());
    c.max_steps_override = a.get("max-steps").and_then(|s| s.parse().ok());
    c.once = a.flag("once");
    c.stats_every = a.num("stats-every", c.stats_every);
    c.trace_dir = if a.flag("no-trace") {
        None
    } else {
        Some(a.get("trace-dir").map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runs").join(format!("link_{}", crate::util::unix_ms()))
        }))
    };
    Ok(c)
}

// ---------------- 소켓 ----------------

/// TCP 읽기(버퍼 + 읽을 때마다 TCP_QUICKACK) — Windows ↔ WSL 지연 ACK 40 ms 대기 제거(docs/에이전트_설계.md 1.10).
pub struct Rd {
    s: TcpStream,
    buf: Vec<u8>,
    rp: usize,
    re: usize,
    quick: bool,
}

impl Rd {
    pub fn new(s: TcpStream) -> Rd {
        Rd { s, buf: vec![0; 64 * 1024], rp: 0, re: 0, quick: crate::ws::quickack_enabled() }
    }
    fn fill(&mut self) -> io::Result<()> {
        if self.rp == self.re {
            self.rp = 0;
            self.re = 0;
        }
        let n = self.s.read(&mut self.buf[self.re..])?;
        if self.quick {
            crate::ws::quickack(&self.s);
        }
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 닫힘"));
        }
        self.re += n;
        Ok(())
    }
    pub fn read_exact(&mut self, dst: &mut [u8]) -> io::Result<()> {
        let mut off = 0;
        // 버퍼에 남은 것 먼저
        let have = (self.re - self.rp).min(dst.len());
        dst[..have].copy_from_slice(&self.buf[self.rp..self.rp + have]);
        self.rp += have;
        off += have;
        if off == dst.len() {
            return Ok(());
        }
        // 큰 것은 바로 받기(버퍼 거치지 않음)
        if dst.len() - off >= self.buf.len() / 2 {
            while off < dst.len() {
                let n = self.s.read(&mut dst[off..])?;
                if self.quick {
                    crate::ws::quickack(&self.s);
                }
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 닫힘"));
                }
                off += n;
            }
            return Ok(());
        }
        while off < dst.len() {
            if self.rp == self.re {
                self.fill()?;
            }
            let k = (self.re - self.rp).min(dst.len() - off);
            dst[off..off + k].copy_from_slice(&self.buf[self.rp..self.rp + k]);
            self.rp += k;
            off += k;
        }
        Ok(())
    }
    pub fn stream(&self) -> &TcpStream {
        &self.s
    }
}

pub fn read_msg_head(r: &mut Rd) -> io::Result<(u16, u16, usize)> {
    let mut h = [0u8; 12];
    r.read_exact(&mut h)?;
    let magic = u32::from_le_bytes(h[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("magic {magic:#x}")));
    }
    Ok((u16::from_le_bytes([h[4], h[5]]), u16::from_le_bytes([h[6], h[7]]), u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize))
}

pub fn msg_head(ty: u16, flags: u16, len: usize) -> [u8; 12] {
    let mut h = [0u8; 12];
    h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    h[4..6].copy_from_slice(&ty.to_le_bytes());
    h[6..8].copy_from_slice(&flags.to_le_bytes());
    h[8..12].copy_from_slice(&(len as u32).to_le_bytes());
    h
}

/// ACK 한 개.
#[derive(Debug, Clone, PartialEq)]
pub struct Ack {
    pub step: u64,
    pub hold_next: bool,
    pub want_next: u16,
    pub decision: Option<Value>,
}

impl Ack {
    pub fn encode(&self) -> Vec<u8> {
        let js = self.decision.as_ref().map(|d| d.to_string().into_bytes()).unwrap_or_default();
        let body = 16 + js.len();
        let mut v = Vec::with_capacity(12 + body);
        v.extend_from_slice(&msg_head(T_ACK, 0, body));
        v.extend_from_slice(&self.step.to_le_bytes());
        v.push(self.hold_next as u8);
        v.push(self.decision.is_some() as u8);
        v.extend_from_slice(&self.want_next.to_le_bytes());
        v.extend_from_slice(&(js.len() as u32).to_le_bytes());
        v.extend_from_slice(&js);
        v
    }
    pub fn decode(body: &[u8]) -> Result<Ack, String> {
        if body.len() < 16 {
            return Err("ACK 짧음".into());
        }
        let n = u32::from_le_bytes(body[12..16].try_into().unwrap()) as usize;
        let decision = if body[9] != 0 {
            Some(serde_json::from_slice(&body[16..16 + n]).map_err(|e| e.to_string())?)
        } else {
            None
        };
        Ok(Ack { step: u64::from_le_bytes(body[0..8].try_into().unwrap()), hold_next: body[8] != 0, want_next: u16::from_le_bytes([body[10], body[11]]), decision })
    }
}

// ---------------- 서버 ----------------

enum PMsg {
    Decision(usize, Decision, StageTrack, u64),
    Returned(usize, Box<Agent>),
}

struct EnvState {
    sess: EnvSession,
    est: Box<dyn PoseEstimator>,
    agent: Option<Box<Agent>>,
    owns: bool,
    pending: Option<(Trigger, [f64; 2])>,
    key: KeyState,
    track: StageTrack,
    stage_sent: Option<u32>,
    stage_logged: Option<u32>,
    head_now: Option<(Mat3, [f64; 3])>,
    steps_seen: u64,
    /// 최근 스텝의 (stamp, 보정 전 자세) — 바깥 추정기 고정점을 그 시점 자세와 맞춘다
    hist: std::collections::VecDeque<(f64, Pose)>,
    fixes: u64,
    /// 직전 스텝 (stamp, 자세, 카메라 외부 자세) — 영상 시각 맞추기
    prev: Option<(f64, Pose, [Option<[f64; 7]>; 3])>,
}

/// 스텝마다 잰 시간(µs)
#[derive(Default, Debug)]
pub struct LinkLat {
    /// STEP 머리를 읽고 몸통(영상 포함)을 다 받을 때까지
    pub recv: Vec<u32>,
    /// 받은 뒤 ACK 를 보낼 때까지(경계가 아닌 스텝)
    pub proc_: Vec<u32>,
    /// 경계 스텝: 받은 뒤 결정 ACK 까지(그래프 기다림 + 계획)
    pub boundary: Vec<u32>,
    pub settle: Vec<u32>,
    pub decide: Vec<u32>,
    pub frames: u64,
    pub frame_bytes: u64,
    pub black_frames: u64,
    pub steps: u64,
    pub holds: u64,
    pub decisions: u64,
}

fn pct(v: &[u32]) -> Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let p = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    json!({"n": s.len(), "p50": p(0.5), "p90": p(0.9), "p99": p(0.99), "max": s[s.len() - 1]})
}

impl LinkLat {
    pub fn summary(&self) -> Value {
        json!({"unit": "us", "steps": self.steps, "holds": self.holds, "decisions": self.decisions, "frames": self.frames,
               "frame_mb": self.frame_bytes as f64 / 1e6, "black_rgb_frames": self.black_frames,
               "recv": pct(&self.recv), "proc": pct(&self.proc_), "boundary": pct(&self.boundary),
               "settle": pct(&self.settle), "decide": pct(&self.decide)})
    }
}

pub struct Server<'a> {
    cfg: &'a LinkCfg,
    tracer: Option<Tracer>,
    sink: Box<dyn ObsSink + 'a>,
    hello: Hello,
    task: Option<TaskCard>,
    envs: Vec<EnvState>,
    episode: u32,
    tx: mpsc::Sender<PMsg>,
    rx: mpsc::Receiver<PMsg>,
    /// cam_rel_poses 묶음 i → 카메라 id
    crp_map: Vec<Option<u8>>,
    pub lat: LinkLat,
}

fn cam_id(name: &str) -> Option<u8> {
    match name {
        "head" => Some(CAM_HEAD),
        "left_wrist" | "left" => Some(CAM_LEFT),
        "right_wrist" | "right" => Some(CAM_RIGHT),
        _ => None,
    }
}

#[inline]
fn us(d: Duration) -> u32 {
    d.as_micros().min(u32::MAX as u128) as u32
}

/// 스텝 k 의 시뮬 시각 [s]
pub fn st_step_time(step: u64, hz: f64) -> f64 {
    step as f64 / if hz > 0.0 { hz } else { 30.0 }
}


impl<'a> Server<'a> {
    pub fn new(cfg: &'a LinkCfg, sink: Box<dyn ObsSink + 'a>, tracer: Option<Tracer>) -> Server<'a> {
        let (tx, rx) = mpsc::channel();
        Server { cfg, tracer, sink, hello: Hello::default(), task: None, envs: vec![], episode: 0, tx, rx, crp_map: vec![], lat: LinkLat::default() }
    }

    fn trace(&self, env: usize, kind: &str, v: Value) {
        if let Some(t) = &self.tracer {
            t.event(env, kind, v);
        }
    }

    /// HELLO → 과제·환경 준비. 돌려주는 JSON 이 HELLO_ACK.
    pub fn on_hello(&mut self, h: Hello) -> Value {
        let cat = &self.cfg.catalog;
        let name = self.cfg.task_override.clone().unwrap_or_else(|| h.task.clone());
        self.task = cat.task_by_name(&name).cloned().or_else(|| h.task_id.and_then(|i| cat.task_by_index(i)).cloned());
        self.crp_map = h.crp_order.iter().map(|n| cam_id(n)).collect();
        let (tp, ms) = self.task.as_ref().map(|t| (t.prompt.clone(), t.max_steps)).unwrap_or_default();
        let ms = self.cfg.max_steps_override.or(h.max_steps).unwrap_or(if ms == 0 { 100_000 } else { ms });
        let n = h.num_envs.max(1);
        self.envs = (0..n)
            .map(|e| {
                let mut agent = match (&self.cfg.factory, &self.task) {
                    (Some(f), Some(t)) => Some(Box::new(f(e, t))),
                    _ => None,
                };
                if let Some(a) = agent.as_mut() {
                    a.set_tracer(self.tracer.clone());
                    a.reset_episode(0);
                }
                let track = agent.as_ref().map(|a| StageTrack::from_core(&a.core, cat, 0)).unwrap_or_default();
                EnvState {
                    sess: EnvSession::new(e, self.cfg.monitor.clone(), h.hz, ms, &tp),
                    est: pose::make(&self.cfg.pose, h.hz).unwrap_or_else(|_| Box::new(pose::QvelIntegrator::new(h.hz))),
                    owns: agent.is_some(),
                    agent,
                    pending: None,
                    key: KeyState::default(),
                    track,
                    stage_sent: None,
                    stage_logged: None,
                    head_now: None,
                    steps_seen: 0,
                    hist: std::collections::VecDeque::with_capacity(1024),
                    fixes: 0,
                    prev: None,
                }
            })
            .collect();
        self.sink.hello(&h);
        let est = self.envs.first().map(|e| e.est.name()).unwrap_or_default();
        self.trace(
            0,
            "link_hello",
            json!({"hello": h, "task": self.task.as_ref().map(|t| t.name.clone()), "max_steps": ms, "prompt_mode": self.cfg.prompt_mode,
                   "stage_mode": self.cfg.stage, "pose": est, "cam_pose": if self.cfg.fk.is_some() { "proprio_fk" } else { "obs_cam_rel_poses" },
                   "key": self.cfg.key, "planner": self.cfg.factory.is_some()}),
        );
        self.hello = h;
        json!({"ok": true, "task": self.task.as_ref().map(|t| t.name.clone()), "planner": self.envs.iter().any(|e| e.owns),
               "hold": true, "want": WANT_ALL, "pose": est})
    }

    /// 계획기 스레드에서 돌아올 것을 기다린다(결정 순서 고정).
    fn handle(&mut self, m: PMsg) -> Option<(usize, Decision, StageTrack, u64)> {
        match m {
            PMsg::Decision(e, d, t, ms) => Some((e, d, t, ms)),
            PMsg::Returned(e, a) => {
                if let Some(s) = self.envs.get_mut(e) {
                    s.agent = Some(a);
                }
                None
            }
        }
    }

    fn wait_agent(&mut self, e: usize) {
        while self.envs[e].agent.is_none() {
            match self.rx.recv() {
                Ok(m) => {
                    let _ = self.handle(m);
                }
                Err(_) => return,
            }
        }
    }

    fn wait_all_agents(&mut self) {
        for e in 0..self.envs.len() {
            if self.envs[e].owns {
                self.wait_agent(e);
            }
        }
    }

    pub fn on_reset(&mut self) {
        self.wait_all_agents();
        if self.envs.iter().any(|e| e.steps_seen > 0) {
            self.episode += 1;
        }
        let ep = self.episode;
        let cat = self.cfg.catalog.clone();
        for s in self.envs.iter_mut() {
            s.sess.reset(ep);
            s.est.reset();
            s.pending = None;
            s.key = KeyState::default();
            s.stage_sent = None;
            s.stage_logged = None;
            s.head_now = None;
            s.steps_seen = 0;
            s.hist.clear();
            s.prev = None;
            if let Some(a) = s.agent.as_mut() {
                a.reset_episode(ep);
                s.track = StageTrack::from_core(&a.core, &cat, 0);
            }
        }
        self.sink.reset(ep);
        self.trace(0, "link_reset", json!({"episode": ep}));
    }

    /// 결정 → 접착부 JSON.
    fn to_glue(&self, e: usize, d: &Decision, step: u64, trig: Trigger, t: &StageTrack, decide_ms: u64, settle_ms: Option<f64>) -> Value {
        let s = &self.envs[e].sess;
        let (prompt, flush) = match self.cfg.prompt_mode {
            PromptMode::Task => (Value::Null, false),
            PromptMode::Subtask => (json!(s.prompt), s.pending_flush),
        };
        let n = self.hello.stage_count;
        let stage_est = if n > 0 { Some(t.stage(step, n)) } else { None };
        let (stage, mode) = match (self.cfg.stage, stage_est) {
            (StageMode::External, Some(k)) => (json!(k), 1),
            _ => (Value::Null, 0),
        };
        json!({"kind": d_kind(d), "prompt": prompt, "flush": flush, "stage": stage, "stage_mode": mode, "stage_est": stage_est,
               "text": d.prompt(), "source": d.source(), "step": step, "trigger": trig, "decide_ms": decide_ms, "settle_ms": settle_ms})
    }

    /// STEP 하나. 반환: ACK.
    pub fn on_step(&mut self, h: &StepHead, proprio: &[f32], crp: &[f32], frames: Vec<Frame>, recv: Instant) -> Ack {
        let e = h.env as usize;
        if e >= self.envs.len() {
            return Ack { step: h.step, hold_next: false, want_next: 0, decision: None };
        }
        let wait = h.flags & F_WAIT != 0;
        let p = |i: usize| proprio.get(i).copied().unwrap_or(0.0) as f64;
        let qvel = [p(BASE_QVEL), p(BASE_QVEL + 1), p(BASE_QVEL + 2)];
        let grips = [p(GRIP_LEFT) + p(GRIP_LEFT + 1), p(GRIP_RIGHT) + p(GRIP_RIGHT + 1)];
        let t_sim = st_step_time(self.envs[e].sess.mon.step, self.hello.hz);
        let fix = self.sink.external_fix(e);
        // 위치(추정기 하나 → 계획기·scenemap 같은 값)
        let st = &mut self.envs[e];
        st.est.step(qvel);
        st.steps_seen += 1;
        st.hist.push_back((t_sim, st.est.raw_pose()));
        if st.hist.len() > 1024 {
            st.hist.pop_front();
        }
        if let Some((ts, p)) = fix {
            // 그 stamp 의 보정 전 자세를 찾아 보정으로(없으면 가장 가까운 이전 스텝)
            if let Some((_, raw_t)) = st.hist.iter().rev().find(|(t, _)| *t <= ts).or(st.hist.front()) {
                st.est.set_correction(pose::correction_from_fix(&p, raw_t));
                st.fixes += 1;
            }
        }
        let base = st.est.pose();
        let mut cam_rel: [Option<[f64; 7]>; 3] = [None, None, None];
        match &self.cfg.fk {
            // 카메라 외부 자세 = proprio 관절값 + 순기구학(평가기 cam_rel_poses 는 안 씀)
            Some(fk) => {
                for (c, slot) in cam_rel.iter_mut().enumerate() {
                    *slot = fk.cam_rel(c, proprio);
                }
            }
            None => {
                for (i, cid) in self.crp_map.iter().enumerate() {
                    if let (Some(c), Some(v)) = (cid, crp.get(i * 7..i * 7 + 7)) {
                        let mut a = [0.0; 7];
                        for (k, x) in v.iter().enumerate() {
                            a[k] = *x as f64;
                        }
                        cam_rel[*c as usize] = Some(a);
                    }
                }
            }
        }
        st.head_now = cam_rel[0].as_ref().map(|r| pose::cam_optical(&base, r));
        // 감시(산술만)
        let trig_now = st.sess.on_obs(qvel, grips);
        let trig = if wait {
            st.pending.take().or(trig_now.map(|t| (t, grips)))
        } else {
            if let Some(t) = trig_now {
                st.pending = Some((t, grips));
            }
            None
        };
        let step = st.sess.mon.step;
        let decide_here = wait && trig.is_some() && st.owns;
        // 경계 영상(LLM) — 보내기 전에 만든다
        let images: Vec<(String, Vec<u8>)> = if decide_here && self.cfg.images {
            frames
                .iter()
                .filter(|f| f.is_rgb())
                .filter_map(|f| {
                    let side = if f.cam == CAM_HEAD { self.cfg.image_side } else { self.cfg.image_side / 2 };
                    f.to_rgb().map(|r| (CAM_NAMES[f.cam.min(2) as usize].to_string(), r.downscale(side).jpeg(self.cfg.jpeg_quality)))
                })
                .collect()
        } else {
            vec![]
        };
        let head_now = st.head_now;
        st.key.got(step, &frames, head_now);
        self.lat.frames += frames.len() as u64;
        self.lat.frame_bytes += frames.iter().map(|f| f.data.len() as u64).sum::<u64>();
        let black: Vec<&str> = frames.iter().filter(|f| f.is_rgb() && f.is_black()).map(|f| CAM_NAMES[f.cam.min(2) as usize]).collect();
        if !black.is_empty() {
            self.lat.black_frames += black.len() as u64;
            self.trace(e, "link_black_frame", json!({"step": step, "cams": black}));
        }
        let (img_t, img_base, img_cam_rel) = self.envs[e].prev.unwrap_or((t_sim, base, cam_rel));
        self.envs[e].prev = Some((t_sim, base, cam_rel));
        self.sink.push(ObsPacket {
            env: e,
            episode: self.episode,
            step,
            recv,
            t: t_sim,
            img_t,
            img_base,
            img_cam_rel,
            base,
            proprio: proprio.to_vec(),
            qvel,
            eef: eef_grip(proprio).0,
            grip: grips,
            cam_rel,
            frames,
            boundary: decide_here,
        });

        let mut decision: Option<Value> = None;
        if decide_here {
            let (t, g) = trig.unwrap();
            self.wait_agent(e);
            let settle = self.sink.settle(e, step, Duration::from_millis(self.cfg.settle_ms));
            if let Some(ms) = settle {
                self.lat.settle.push((ms * 1000.0) as u32);
            }
            let st = &mut self.envs[e];
            let mut bev = st.sess.event(t, g, images);
            bev.pose = base;
            if let Some(tr) = &self.tracer {
                bev.image_files = bev.images.iter().map(|(c, j)| tr.save_image(&format!("ep{}_env{}_s{}_{}.jpg", bev.episode, e, bev.step, c), j)).collect();
            }
            let mut agent = st.agent.take().unwrap();
            st.sess.mon.busy = true;
            let txc = self.tx.clone();
            let cat = self.cfg.catalog.clone();
            let t0 = Instant::now();
            std::thread::spawn(move || {
                let d = agent.decide(&bev);
                let track = StageTrack::from_core(&agent.core, &cat, bev.step);
                let ms = t0.elapsed().as_millis() as u64;
                let _ = txc.send(PMsg::Decision(e, d, track, ms));
                agent.maintain();
                agent.release();
                let _ = txc.send(PMsg::Returned(e, agent));
            });
            // 결정이 올 때까지(시뮬레이터 정지). 다른 환경의 에이전트가 돌아오는 것도 받는다.
            let got = loop {
                match self.rx.recv() {
                    Ok(m) => {
                        if let Some(x) = self.handle(m) {
                            break Some(x);
                        }
                    }
                    Err(_) => break None,
                }
            };
            if let Some((de, d, track, ms)) = got {
                self.lat.decide.push((ms * 1000).min(u32::MAX as u64) as u32);
                self.lat.decisions += 1;
                let s = &mut self.envs[de];
                s.sess.apply(&d);
                s.sess.mon.busy = false;
                let track = track.continued_from(&s.track);
                s.track = track;
                let g = self.to_glue(de, &d, step, t, &track, ms, settle);
                let s = &mut self.envs[de];
                s.sess.pending_flush = false;
                if let Some(k) = g.get("stage").and_then(|v| v.as_u64()) {
                    s.stage_sent = Some(k as u32);
                }
                s.stage_logged = g.get("stage_est").and_then(|v| v.as_u64()).map(|k| k as u32);
                self.trace(de, "link_decision", json!({"step": step, "glue": g, "pose": base}));
                decision = Some(g);
            }
        } else if self.cfg.stage != StageMode::Off && self.hello.stage_count > 0 && self.envs[e].owns {
            // 단계 번호를 스텝마다(진행률이 단계 경계를 넘을 때만 보냄)
            let n = self.hello.stage_count;
            let s = &mut self.envs[e];
            let k = s.track.stage(step, n);
            if s.stage_logged != Some(k) {
                s.stage_logged = Some(k);
                let external = self.cfg.stage == StageMode::External;
                if external {
                    s.stage_sent = Some(k);
                    decision = Some(json!({"kind": "stage", "prompt": null, "flush": false, "stage": k, "stage_mode": 1, "stage_est": k, "step": step}));
                }
                self.trace(e, "link_stage", json!({"step": step, "stage_est": k, "sent": external}));
            }
        }

        let s = &mut self.envs[e];
        s.sess.mon.advance();
        let hold_next = s.owns && (s.pending.is_some() || s.sess.mon.scheduled_due());
        let want_next = s.key.want(&self.cfg.key, step + 1, s.head_now.as_ref(), hold_next);
        self.lat.steps += 1;
        if wait {
            self.lat.holds += 1;
        }
        Ack { step: h.step, hold_next, want_next, decision }
    }

    /// HELLO 를 받았나
    pub fn greeted(&self) -> bool {
        !self.envs.is_empty()
    }

    pub fn finish(&mut self) -> Value {
        self.wait_all_agents();
        let fixes: Vec<u64> = self.envs.iter().map(|s| s.fixes).collect();
        let stats = json!({"link": self.lat.summary(), "sink": self.sink.stats(), "pose_fixes": fixes});
        self.trace(0, "link_end", stats.clone());
        for s in &self.envs {
            if let Some(a) = &s.agent {
                self.trace(s.sess.env, "agent_stats", json!(a.core.stats));
            }
        }
        if let Some(t) = &self.tracer {
            t.flush();
        }
        stats
    }
}

fn d_kind(d: &Decision) -> &'static str {
    match d {
        Decision::Issue { .. } => "issue",
        Decision::Continue { .. } => "continue",
        Decision::Finish { .. } => "finish",
    }
}

fn bad(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// STEP 몸통을 읽는다(머리 12 B 뒤). 영상은 영상마다 제 Vec 으로 바로 받는다(복사 없음).
pub fn read_step(r: &mut Rd, len: usize, scratch: &mut Vec<u8>) -> io::Result<(StepHead, Vec<f32>, Vec<f32>, Vec<Frame>)> {
    let mut hb = [0u8; StepHead::LEN];
    r.read_exact(&mut hb)?;
    let h = StepHead::decode(&hb);
    let nf = (h.n_proprio as usize + h.n_crp as usize) * 4;
    let nh = h.n_frames as usize * FrameHead::LEN;
    scratch.resize(nf + nh, 0);
    r.read_exact(scratch)?;
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let np = h.n_proprio as usize * 4;
    let proprio = f32s(&scratch[..np]);
    let crp = f32s(&scratch[np..nf]);
    let heads: Vec<FrameHead> = (0..h.n_frames as usize).map(|i| FrameHead::decode(&scratch[nf + i * 16..nf + i * 16 + 16])).collect();
    let mut total = StepHead::LEN + nf + nh;
    let mut frames = Vec::with_capacity(heads.len());
    for fh in heads {
        let want = fh.h as u64 * fh.w as u64 * FrameHead::bytes_per_px(fh.kind) as u64;
        if want != fh.nbytes as u64 || fh.cam > 2 {
            return Err(bad(format!("영상 머리 이상: {fh:?}")));
        }
        let mut data = vec![0u8; fh.nbytes as usize];
        r.read_exact(&mut data)?;
        total += data.len();
        frames.push(Frame { cam: fh.cam, kind: fh.kind, h: fh.h, w: fh.w, data });
    }
    if total != len {
        return Err(bad(format!("STEP 길이 {len} ≠ 읽은 {total}")));
    }
    Ok((h, proprio, crp, frames))
}

/// 연결 하나.
pub fn serve(cfg: &LinkCfg, stream: TcpStream, sink: Box<dyn ObsSink + '_>, tracer: Option<Tracer>) -> io::Result<Value> {
    let _ = stream.set_nodelay(true);
    let mut w = stream.try_clone()?;
    let mut r = Rd::new(stream);
    let mut srv = Server::new(cfg, sink, tracer);
    let mut scratch = Vec::with_capacity(4096);
    let mut body = Vec::with_capacity(4096);
    loop {
        let (ty, _flags, len) = match read_msg_head(&mut r) {
            Ok(x) => x,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                srv.finish();
                return Err(e);
            }
        };
        let t_in = Instant::now();
        match ty {
            T_HELLO => {
                body.resize(len, 0);
                r.read_exact(&mut body)?;
                let h: Hello = serde_json::from_slice(&body).map_err(bad)?;
                eprintln!("[link] HELLO task={} envs={} cams={} stage_count={} client={}", h.task, h.num_envs, h.cams.len(), h.stage_count, h.client);
                let ack = srv.on_hello(h).to_string().into_bytes();
                let mut v = msg_head(T_HELLO_ACK, 0, ack.len()).to_vec();
                v.extend_from_slice(&ack);
                w.write_all(&v)?;
            }
            T_RESET => {
                body.resize(len, 0);
                r.read_exact(&mut body)?;
                srv.on_reset();
            }
            T_STEP => {
                let (h, proprio, crp, frames) = read_step(&mut r, len, &mut scratch)?;
                let t_rx = Instant::now();
                srv.lat.recv.push(us(t_rx - t_in));
                let ack = srv.on_step(&h, &proprio, &crp, frames, t_rx);
                let dec = ack.decision.is_some() && ack.decision.as_ref().and_then(|d| d.get("kind")).and_then(|k| k.as_str()) != Some("stage");
                w.write_all(&ack.encode())?;
                if dec {
                    srv.lat.boundary.push(us(t_rx.elapsed()));
                } else {
                    srv.lat.proc_.push(us(t_rx.elapsed()));
                }
                if cfg.stats_every > 0 && srv.lat.steps % cfg.stats_every == 0 {
                    let s = json!({"steps": srv.lat.steps, "stats": srv.lat.summary(), "sink": srv.sink.stats()});
                    srv.trace(0, "link_stats", s);
                }
            }
            T_BYE => break,
            o => {
                body.resize(len, 0);
                r.read_exact(&mut body)?;
                eprintln!("[link] 모르는 메시지 {o}");
            }
        }
    }
    if !srv.greeted() {
        // HELLO 없이 닫힌 연결(포트 확인 등)은 한 판으로 치지 않는다
        return Ok(Value::Null);
    }
    Ok(srv.finish())
}

/// 듣기. `make_sink` 는 연결마다 새 내보내기를 만든다.
pub fn run(cfg: LinkCfg, make_sink: &dyn Fn() -> Box<dyn ObsSink>) -> io::Result<()> {
    let l = TcpListener::bind(&cfg.listen)?;
    run_listener(l, cfg, make_sink)
}

pub fn run_listener(l: TcpListener, cfg: LinkCfg, make_sink: &dyn Fn() -> Box<dyn ObsSink>) -> io::Result<()> {
    eprintln!(
        "[link] {} 에서 평가기 접착부를 기다림 (계획기 {}, VLA 입력 {:?}, 단계 {:?}, 위치 {})",
        l.local_addr().map(|a| a.to_string()).unwrap_or_default(),
        cfg.factory.is_some(),
        cfg.prompt_mode,
        cfg.stage,
        cfg.pose
    );
    for s in l.incoming() {
        let s = match s {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[link] accept 오류: {e}");
                continue;
            }
        };
        eprintln!("[link] 연결: {:?}", s.peer_addr());
        let tracer = match &cfg.trace_dir {
            Some(d) => {
                let sub = if cfg.once { d.clone() } else { d.join(format!("conn_{}", crate::util::unix_ms())) };
                Some(Tracer::create(&sub)?)
            }
            None => None,
        };
        // 연결은 한 번에 하나(평가기 하나). 끝나면 다음 연결.
        let played = match serve(&cfg, s, make_sink(), tracer) {
            Ok(Value::Null) => false,
            Ok(v) => {
                eprintln!("[link] 연결 끝: {v}");
                true
            }
            Err(e) => {
                eprintln!("[link] 연결 오류: {e}");
                true
            }
        };
        if cfg.once && played {
            break;
        }
    }
    Ok(())
}

// ---------------- 접착부 쪽(시험·Rust 가짜 평가기) ----------------

/// Rust 로 짠 접착부 판(시험·벤치). 파이썬 접착부와 같은 순서로 보낸다.
pub struct Client {
    pub w: TcpStream,
    pub r: Rd,
    /// 전 스텝 ACK 를 아직 안 읽었나
    outstanding: bool,
    pub hold: bool,
    pub want: u16,
}

impl Client {
    pub fn connect(addr: &str, hello: &Hello) -> io::Result<(Client, Value)> {
        let s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        let mut w = s.try_clone()?;
        let mut r = Rd::new(s);
        let js = serde_json::to_vec(hello).map_err(bad)?;
        let mut v = msg_head(T_HELLO, 0, js.len()).to_vec();
        v.extend_from_slice(&js);
        w.write_all(&v)?;
        let (ty, _, len) = read_msg_head(&mut r)?;
        if ty != T_HELLO_ACK {
            return Err(bad(format!("HELLO_ACK 대신 {ty}")));
        }
        let mut b = vec![0u8; len];
        r.read_exact(&mut b)?;
        let ack: Value = serde_json::from_slice(&b).map_err(bad)?;
        let want = ack.get("want").and_then(|x| x.as_u64()).unwrap_or(0) as u16;
        Ok((Client { w, r, outstanding: false, hold: true, want }, ack))
    }

    pub fn reset(&mut self) -> io::Result<()> {
        self.drain()?;
        self.w.write_all(&msg_head(T_RESET, 0, 0))?;
        self.hold = true;
        self.want = WANT_ALL;
        Ok(())
    }

    fn read_ack(&mut self) -> io::Result<Ack> {
        let (ty, _, len) = read_msg_head(&mut self.r)?;
        let mut b = vec![0u8; len];
        self.r.read_exact(&mut b)?;
        if ty != T_ACK {
            return Err(bad(format!("ACK 대신 {ty}")));
        }
        Ack::decode(&b).map_err(bad)
    }

    /// 전 스텝 ACK 를 읽어 hold/want 갱신. 결정(비경계 스텝의 단계 갱신)이 있으면 돌려준다.
    pub fn drain(&mut self) -> io::Result<Option<Value>> {
        if !self.outstanding {
            return Ok(None);
        }
        self.outstanding = false;
        let a = self.read_ack()?;
        self.hold = a.hold_next;
        self.want = a.want_next;
        Ok(a.decision)
    }

    /// 한 스텝: 전 ACK 읽기 → 요청된 영상과 함께 보내기 → hold 면 결정 기다리기. 반환: (보낸 영상 비트, 결정들)
    pub fn step(&mut self, env: u16, step: u64, proprio: &[f32], crp: &[f32], frames: &dyn Fn(u16) -> Vec<Frame>) -> io::Result<(u16, bool, Vec<Value>)> {
        let mut decs = Vec::new();
        if let Some(d) = self.drain()? {
            decs.push(d);
        }
        let wait = self.hold;
        let fr = frames(self.want);
        let mut bits = 0u16;
        for f in &fr {
            bits |= if f.is_rgb() { want_rgb(f.cam) } else { want_depth(f.cam) };
        }
        let h = StepHead { step, t_client: 0.0, flags: if wait { F_WAIT } else { 0 }, env, n_proprio: proprio.len() as u16, n_crp: crp.len() as u16, n_frames: fr.len() as u16, _pad: 0 };
        let len = StepHead::LEN + (proprio.len() + crp.len()) * 4 + fr.len() * 16 + fr.iter().map(|f| f.data.len()).sum::<usize>();
        let mut small = Vec::with_capacity(12 + 32 + (proprio.len() + crp.len()) * 4 + fr.len() * 16);
        small.extend_from_slice(&msg_head(T_STEP, 0, len));
        small.extend_from_slice(&h.encode());
        for x in proprio.iter().chain(crp.iter()) {
            small.extend_from_slice(&x.to_le_bytes());
        }
        for f in &fr {
            small.extend_from_slice(&FrameHead { cam: f.cam, kind: f.kind, h: f.h, w: f.w, nbytes: f.data.len() as u32 }.encode());
        }
        self.w.write_all(&small)?;
        for f in &fr {
            self.w.write_all(&f.data)?;
        }
        self.outstanding = true;
        if wait {
            if let Some(d) = self.drain()? {
                decs.push(d);
            }
        }
        Ok((bits, wait, decs))
    }

    pub fn bye(mut self) -> io::Result<()> {
        self.drain()?;
        self.w.write_all(&msg_head(T_BYE, 0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_roundtrip() {
        let h = StepHead { step: 12345, t_client: 1.5, flags: F_WAIT, env: 2, n_proprio: 61, n_crp: 21, n_frames: 3, _pad: 0 };
        assert_eq!(StepHead::decode(&h.encode()), h);
        let f = FrameHead { cam: 2, kind: KIND_DEPTH_F32, h: 480, w: 480, nbytes: 480 * 480 * 4 };
        assert_eq!(FrameHead::decode(&f.encode()), f);
        let a = Ack { step: 7, hold_next: true, want_next: WANT_HEAD, decision: Some(json!({"kind": "issue", "stage": 3})) };
        let enc = a.encode();
        assert_eq!(Ack::decode(&enc[12..]).unwrap(), a);
        let a2 = Ack { step: 8, hold_next: false, want_next: 0, decision: None };
        assert_eq!(Ack::decode(&a2.encode()[12..]).unwrap(), a2);
    }

    #[test]
    fn eef_grip_indices() {
        // 평가기 proprio 표(eval_utils PROPRIOCEPTION_INDICES["R1Pro"])와 대조: 값 = 번호
        let p: Vec<f32> = (0..61).map(|i| i as f32).collect();
        let (e, g) = eef_grip(&p);
        assert_eq!(e[0], [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
        assert_eq!(e[1], [42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0]);
        assert_eq!(g, [24.0 + 25.0, 49.0 + 50.0]);
    }

    #[test]
    fn keyframe_policy() {
        let c = KeyCfg::default();
        let mut k = KeyState::default();
        let pose0 = pose::cam_optical(&Pose::default(), &[0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        // 처음은 머리 요청
        assert_eq!(k.want(&c, 0, Some(&pose0), false), WANT_HEAD);
        // 경계면 전부
        assert_eq!(k.want(&c, 0, Some(&pose0), true), WANT_ALL);
        let fr = vec![Frame { cam: 0, kind: KIND_RGBA8, h: 1, w: 1, data: vec![1, 2, 3, 255] }];
        k.got(10, &fr, Some(pose0));
        // 안 움직이면 간격(6) 전엔 안 요청
        assert_eq!(k.want(&c, 11, Some(&pose0), false), 0);
        assert_eq!(k.want(&c, 16, Some(&pose0), false), WANT_HEAD);
        // 4 cm 는 아직, 6 cm 는 요청
        let p4 = pose::cam_optical(&Pose { x: 0.04, ..Default::default() }, &[0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        let p6 = pose::cam_optical(&Pose { x: 0.06, ..Default::default() }, &[0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.want(&c, 11, Some(&p4), false), 0);
        assert_eq!(k.want(&c, 11, Some(&p6), false), WANT_HEAD);
        // 4° 회전은 요청
        let r4 = pose::cam_optical(&Pose { yaw: 4f64.to_radians(), ..Default::default() }, &[0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.want(&c, 11, Some(&r4), false), WANT_HEAD);
        // 손목 주기
        let c2 = KeyCfg { wrist_every: 10, ..KeyCfg::default() };
        assert_eq!(k.want(&c2, 11, Some(&pose0), false), WANT_ALL & !WANT_HEAD);
    }

    #[test]
    fn stage_track_math() {
        let t = StageTrack { before: 300.0, cur: 100.0, total: 1000.0, start: 50, finished: false, cur_id: Some(2) };
        let again = StageTrack { start: 120, ..t }.continued_from(&t);
        assert_eq!(again.start, 50);
        let next = StageTrack { start: 120, cur_id: Some(3), ..t }.continued_from(&t);
        assert_eq!(next.start, 120);
        assert!((t.frac(50) - 0.3).abs() < 1e-12);
        assert!((t.frac(100) - 0.35).abs() < 1e-12);
        assert!((t.frac(10_000) - 0.4).abs() < 1e-12); // 지금 단계 p50 까지만
        assert_eq!(t.stage(50, 10), 3);
        assert_eq!(t.stage(149, 10), 3);
        assert_eq!(t.stage(150, 10), 4);
        let f = StageTrack { finished: true, ..t };
        assert_eq!(f.stage(0, 10), 9);
        assert_eq!(StageTrack::default().stage(5, 7), 0);
    }
}
