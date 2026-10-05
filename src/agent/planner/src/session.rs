//! 환경 하나(배치 평가면 환경마다)의 매 스텝 상태: 오도메트리 + 단계 감시 + 지금 지시 문장.
//! 중계기와 가짜 세계가 같은 코드를 쓴다.

use crate::monitor::{Monitor, MonitorCfg, Trigger};
use crate::odom::Odom;
use crate::planner::{BoundaryEvent, Decision};

pub struct EnvSession {
    pub env: usize,
    pub episode: u32,
    pub odom: Odom,
    pub mon: Monitor,
    pub prompt: String,
    pub pending_flush: bool,
    pub max_steps: u64,
    pub task_prompt: String,
}

impl EnvSession {
    pub fn new(env: usize, mcfg: MonitorCfg, hz: f64, max_steps: u64, task_prompt: &str) -> EnvSession {
        EnvSession {
            env,
            episode: 0,
            odom: Odom::new(hz),
            mon: Monitor::new(mcfg),
            prompt: task_prompt.to_string(),
            pending_flush: false,
            max_steps,
            task_prompt: task_prompt.to_string(),
        }
    }

    pub fn reset(&mut self, episode: u32) {
        self.episode = episode;
        self.odom.reset();
        self.mon.reset();
        self.prompt = self.task_prompt.clone();
        self.pending_flush = false;
    }

    /// 관측 한 스텝(매 스텝 경로: 산술만).
    #[inline]
    pub fn on_obs(&mut self, qvel: [f64; 3], grips: [f64; 2]) -> Option<Trigger> {
        self.odom.step(qvel);
        self.mon.on_step(self.odom.dist, self.odom.turned, self.odom.speed(), grips)
    }

    pub fn event(&self, trigger: Trigger, grips: [f64; 2], images: Vec<(String, Vec<u8>)>) -> BoundaryEvent {
        BoundaryEvent {
            env: self.env,
            episode: self.episode,
            step: self.mon.step,
            trigger,
            max_steps: self.max_steps,
            pose: self.odom.pose,
            stage_steps: self.mon.steps_in_stage(),
            stage_budget: if self.mon.budget == u64::MAX { 0 } else { self.mon.budget },
            moved_in_stage: self.mon.moved_in_stage(self.odom.dist),
            base_speed: self.odom.speed().0,
            still_steps: self.mon.still_steps(),
            grippers: grips,
            images,
            image_files: vec![],
        }
    }

    /// 결정을 반영: 문장·예산·확인 시점.
    pub fn apply(&mut self, d: &Decision) {
        match d {
            Decision::Issue { instruction, prompt, budget, check_every, flush, .. } => {
                if *flush && *prompt != self.prompt {
                    self.pending_flush = true;
                }
                self.prompt = prompt.clone();
                let nav = crate::vocab::lookup(&instruction.skill).map(|k| k.kind == crate::vocab::Kind::Navigation).unwrap_or(false);
                self.mon.begin_stage(*budget, *check_every, nav, self.odom.dist, self.odom.turned);
            }
            Decision::Continue { extra, check_every, .. } => self.mon.extend(*extra, *check_every),
            Decision::Finish { prompt, .. } => {
                if *prompt != self.prompt {
                    self.pending_flush = true;
                }
                self.prompt = prompt.clone();
                let left = self.max_steps.saturating_sub(self.mon.step).max(1);
                self.mon.begin_stage(left, 300, false, self.odom.dist, self.odom.turned);
            }
        }
    }
}
