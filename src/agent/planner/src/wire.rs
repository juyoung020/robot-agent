//! 평가기 관측 메시지 해석(복사 없이)과 지시 주입.
//!
//! 관측 키(평가기 `evaluator.py` `_preprocess_obs`, r1pro.yaml): `<robot>::proprio`(61, 앞 3개가 base_qvel),
//! `<robot>::<robot>:zed_link:Camera:0::rgb` 등 카메라, `<robot>::cam_rel_poses`, `task_id`. 로봇 이름은 설정마다
//! 달라(robot / robot_r1) 키는 뒷부분으로 찾는다. 배치 평가(`--num-envs`)면 배열 맨 앞 차원이 환경 수.
//!
//! 주입: 원래 항목 바이트는 그대로 두고 맵 끝에 `__agent_prompt__`(문자열 또는 환경별 문자열 배열),
//! 필요하면 `__agent_flush__`(불 또는 배열)를 붙인다. 맵 머리의 항목 수만 바뀐다.

use crate::msgpack::{self, NdRef, Src, TopMap};

pub const PROMPT_KEY: &str = "__agent_prompt__";
pub const FLUSH_KEY: &str = "__agent_flush__";

/// proprio 안 위치 (OmniGibson `eval/utils/eval_utils.py` PROPRIOCEPTION_INDICES["R1Pro"])
pub const PROPRIO_DIM: usize = 61;
pub const BASE_QVEL: usize = 0; // 0..3: 로봇 기준 vx, vy, wz
pub const GRIP_LEFT: usize = 24; // 24..26
pub const GRIP_RIGHT: usize = 49; // 49..51

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cam {
    Head,
    LeftWrist,
    RightWrist,
}

impl Cam {
    pub fn key_part(self) -> &'static str {
        match self {
            Cam::Head => "zed_link:Camera:0::rgb",
            Cam::LeftWrist => "left_realsense_link:Camera:0::rgb",
            Cam::RightWrist => "right_realsense_link:Camera:0::rgb",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Cam::Head => "head",
            Cam::LeftWrist => "left_wrist",
            Cam::RightWrist => "right_wrist",
        }
    }
    pub fn parse(s: &str) -> Option<Cam> {
        match s {
            "head" => Some(Cam::Head),
            "left_wrist" | "left" => Some(Cam::LeftWrist),
            "right_wrist" | "right" => Some(Cam::RightWrist),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObsView {
    pub top: TopMap,
    pub is_reset: bool,
    pub batch: usize,
    pub batched: bool,
    pub proprio: Option<NdRef>,
    pub task_id: Option<i64>,
    pub cams: [Option<NdRef>; 3],
}

pub fn view<S: Src>(s: &S) -> msgpack::R<ObsView> {
    let top = msgpack::scan_top(s)?;
    let is_reset = top.find(|k| k == "reset").is_some();
    let mut v = ObsView { top, is_reset, batch: 1, batched: false, proprio: None, task_id: None, cams: [None, None, None] };
    if is_reset {
        return Ok(v);
    }
    if let Some(e) = v.top.find(|k| k.ends_with("::proprio")) {
        if let Some(nd) = msgpack::parse_nd(s, e.val_off)? {
            if nd.shape.len() == 2 {
                v.batch = nd.shape[0];
                v.batched = true;
            }
            v.proprio = Some(nd);
        }
    }
    if let Some(e) = v.top.find(|k| k == "task_id") {
        if let Some(nd) = msgpack::parse_nd(s, e.val_off)? {
            v.task_id = nd.get_f64(s, 0).map(|x| x as i64);
        } else {
            v.task_id = msgpack::Rd::new(s, e.val_off).int().ok();
        }
    }
    for (i, c) in [Cam::Head, Cam::LeftWrist, Cam::RightWrist].iter().enumerate() {
        if let Some(e) = v.top.find(|k| k.ends_with(c.key_part())) {
            v.cams[i] = msgpack::parse_nd(s, e.val_off)?;
        }
    }
    Ok(v)
}

impl ObsView {
    fn proprio_at<S: Src>(&self, s: &S, env: usize, idx: usize) -> f64 {
        self.proprio
            .as_ref()
            .and_then(|nd| {
                let dim = *nd.shape.last().unwrap_or(&PROPRIO_DIM);
                nd.get_f64(s, env * dim + idx)
            })
            .unwrap_or(0.0)
    }
    pub fn base_qvel<S: Src>(&self, s: &S, env: usize) -> [f64; 3] {
        [self.proprio_at(s, env, BASE_QVEL), self.proprio_at(s, env, BASE_QVEL + 1), self.proprio_at(s, env, BASE_QVEL + 2)]
    }
    /// 그리퍼 벌어짐(두 손가락 합, m). R1Pro 손가락 하나 0~0.05(시연 stats.json).
    pub fn grippers<S: Src>(&self, s: &S, env: usize) -> [f64; 2] {
        [
            self.proprio_at(s, env, GRIP_LEFT) + self.proprio_at(s, env, GRIP_LEFT + 1),
            self.proprio_at(s, env, GRIP_RIGHT) + self.proprio_at(s, env, GRIP_RIGHT + 1),
        ]
    }
    /// 카메라 영상(RGB 로, 환경 env 하나)을 복사해 온다. 단계 경계에서만 부른다.
    pub fn image<S: Src>(&self, s: &S, cam: Cam, env: usize) -> Option<crate::image::Rgb> {
        let nd = self.cams[cam as usize].as_ref()?;
        let sh = &nd.shape;
        let (h, w, c) = match sh.len() {
            3 => (sh[0], sh[1], sh[2]),
            4 => (sh[1], sh[2], sh[3]),
            _ => return None,
        };
        if nd.elem_size() != 1 || c < 3 {
            return None;
        }
        let per = h * w * c;
        let base = nd.off + env.min(self.batch.saturating_sub(1)) * per;
        if base + per > nd.off + nd.len {
            return None;
        }
        let mut raw = vec![0u8; per];
        s.copy(base, &mut raw);
        let mut rgb = Vec::with_capacity(h * w * 3);
        for px in raw.chunks_exact(c) {
            rgb.extend_from_slice(&px[..3]);
        }
        Some(crate::image::Rgb { w, h, px: rgb })
    }
}

/// 주입할 꼬리(맵 항목들)와 항목 수.
pub fn build_suffix(prompts: &[String], flush: Option<&[bool]>, batched: bool, out: &mut Vec<u8>) -> usize {
    out.clear();
    let mut n = 0;
    msgpack::w_str(out, PROMPT_KEY);
    if batched {
        msgpack::w_arr(out, prompts.len());
        for p in prompts {
            msgpack::w_str(out, p);
        }
    } else {
        msgpack::w_str(out, prompts.first().map(|s| s.as_str()).unwrap_or(""));
    }
    n += 1;
    if let Some(f) = flush {
        msgpack::w_str(out, FLUSH_KEY);
        if batched {
            msgpack::w_arr(out, f.len());
            for &b in f {
                msgpack::w_bool(out, b);
            }
        } else {
            msgpack::w_bool(out, f.first().copied().unwrap_or(false));
        }
        n += 1;
    }
    n
}

/// 평문 관측에 꼬리를 붙인 새 메시지(시험·비교용; 중계기 본 경로는 복사 없이 조각으로 보낸다).
pub fn inject_plain(msg: &[u8], top: &TopMap, suffix: &[u8], extra: usize) -> Vec<u8> {
    let (h, hl) = msgpack::map_hdr(top.count + extra);
    let mut o = Vec::with_capacity(msg.len() + suffix.len() + 4);
    o.extend_from_slice(&h[..hl]);
    o.extend_from_slice(&msg[top.hdr_len..]);
    o.extend_from_slice(suffix);
    o
}

/// 주입된 메시지에서 주입 항목을 빼면 원래 항목 바이트와 같은지(시험·검증용).
pub fn strip_injected(msg: &[u8]) -> msgpack::R<(Vec<u8>, Option<serde_json::Value>)> {
    let s = msgpack::Plain(msg);
    let top = msgpack::scan_top(&s)?;
    let keep: Vec<_> = top.entries.iter().filter(|e| e.key != PROMPT_KEY && e.key != FLUSH_KEY).collect();
    let mut o = Vec::new();
    msgpack::w_map(&mut o, keep.len());
    for e in &keep {
        o.extend_from_slice(&msg[e.key_off..e.end]);
    }
    let prompt = top.entries.iter().find(|e| e.key == PROMPT_KEY).and_then(|e| {
        let mut r = msgpack::Rd::new(&s, e.val_off);
        msgpack::to_json(&mut r, 0).ok()
    });
    Ok((o, prompt))
}
