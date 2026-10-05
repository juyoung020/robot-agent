//! 매 스텝 경로의 단계 감시. 산술 몇 개뿐이고 LLM·할당이 없다.
//! 언제 에이전트(LLM)를 부를지 정한다: 판 시작, 스텝 예산 소진, 정기 확인, 이동이 멈춤(이동 단계), 그리퍼 열림/닫힘 변화.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorCfg {
    /// 정기 확인 간격(스텝). 0 이면 단계 예산의 1/3 (최소 60).
    pub check_every: u64,
    /// 두 경계 사이 최소 간격(스텝)
    pub min_gap: u64,
    /// 이동 단계: 이만큼 움직인 뒤 멈추면 확인
    pub settle_min_move: f64,
    pub settle_speed: f64,
    pub settle_yaw_rate: f64,
    pub settle_steps: u32,
    /// 그리퍼 벌어짐(두 손가락 합, m): 이보다 작으면 닫힘, 크면 열림(사이값은 유지)
    pub grip_closed_below: f64,
    pub grip_open_above: f64,
    pub grip_hold_steps: u32,
}

impl Default for MonitorCfg {
    fn default() -> Self {
        MonitorCfg {
            check_every: 0,
            min_gap: 15,
            settle_min_move: 0.3,
            settle_speed: 0.03,
            settle_yaw_rate: 0.05,
            settle_steps: 20,
            grip_closed_below: 0.07,
            grip_open_above: 0.09,
            grip_hold_steps: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    EpisodeStart,
    BudgetExhausted,
    CheckDue,
    Settled,
    GripperChange,
}

impl Trigger {
    pub fn describe(self) -> &'static str {
        match self {
            Trigger::EpisodeStart => "episode start",
            Trigger::BudgetExhausted => "step budget for the current step is used up",
            Trigger::CheckDue => "periodic progress check",
            Trigger::Settled => "the base stopped moving after travelling (navigation may be finished)",
            Trigger::GripperChange => "a gripper opened or closed (grasp/release may have happened)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Monitor {
    pub cfg: MonitorCfg,
    /// 이번 판에서 받은 관측 수(첫 관측 = 0)
    pub step: u64,
    pub stage_start: u64,
    pub budget: u64,
    pub next_check: u64,
    pub last_trigger: Option<u64>,
    pub nav: bool,
    dist_at_start: f64,
    turn_at_start: f64,
    still: u32,
    settled_fired: bool,
    grip_state: [Option<bool>; 2],
    grip_cand: [u32; 2],
    grip_fired: bool,
    /// 에이전트가 생각 중이면 새 경계를 만들지 않는다.
    pub busy: bool,
    started: bool,
}

impl Monitor {
    pub fn new(cfg: MonitorCfg) -> Monitor {
        Monitor {
            cfg,
            step: 0,
            stage_start: 0,
            budget: u64::MAX,
            next_check: u64::MAX,
            last_trigger: None,
            nav: false,
            dist_at_start: 0.0,
            turn_at_start: 0.0,
            still: 0,
            settled_fired: false,
            grip_state: [None, None],
            grip_cand: [0, 0],
            grip_fired: false,
            busy: false,
            started: false,
        }
    }

    pub fn reset(&mut self) {
        let cfg = self.cfg.clone();
        *self = Monitor::new(cfg);
    }

    /// 다음 관측에서 스텝 수로 정해지는 경계(판 시작·예산 소진·정기 확인)가 날 수 있는가.
    /// 중계기는 이때만 관측을 통째로 붙잡고, 나머지 스텝은 흘려보낸다.
    pub fn scheduled_due(&self) -> bool {
        if !self.started {
            return true;
        }
        if self.busy || self.last_trigger.map(|lt| self.step < lt + self.cfg.min_gap).unwrap_or(false) {
            return false;
        }
        self.steps_in_stage() >= self.budget || self.step >= self.next_check
    }

    /// base 가 연달아 멈춰 있던 스텝 수
    pub fn still_steps(&self) -> u32 {
        self.still
    }

    pub fn moved_in_stage(&self, dist: f64) -> f64 {
        (dist - self.dist_at_start).max(0.0)
    }

    pub fn steps_in_stage(&self) -> u64 {
        self.step.saturating_sub(self.stage_start)
    }

    /// 새 단계 시작(지시가 바뀜).
    pub fn begin_stage(&mut self, budget: u64, check_every: u64, nav: bool, dist: f64, turned: f64) {
        self.stage_start = self.step;
        self.budget = budget.max(1);
        let ce = if check_every > 0 {
            check_every
        } else if self.cfg.check_every > 0 {
            self.cfg.check_every
        } else {
            (budget / 3).max(60)
        };
        self.next_check = self.step + ce;
        self.nav = nav;
        self.dist_at_start = dist;
        self.turn_at_start = turned;
        self.still = 0;
        self.settled_fired = false;
        self.grip_fired = false;
    }

    /// 지금 지시를 계속(예산 연장·다음 확인 시점만 바꿈).
    pub fn extend(&mut self, extra: u64, check_every: u64) {
        self.budget = self.budget.saturating_add(extra);
        let ce = if check_every > 0 { check_every } else { (self.budget / 3).max(60) };
        self.next_check = self.step + ce;
    }

    /// 관측 한 스텝. `speed` = (선속도 m/s, 각속도 rad/s), `grips` = 좌·우 벌어짐.
    pub fn on_step(&mut self, dist: f64, turned: f64, speed: (f64, f64), grips: [f64; 2]) -> Option<Trigger> {
        let t = self.eval(dist, turned, speed, grips);
        if t.is_some() {
            self.last_trigger = Some(self.step);
        }
        t
    }

    /// 다음 관측으로 넘어간다(관측을 다 처리한 뒤 한 번).
    pub fn advance(&mut self) {
        self.step += 1;
    }

    fn eval(&mut self, dist: f64, turned: f64, speed: (f64, f64), grips: [f64; 2]) -> Option<Trigger> {
        // 그리퍼 상태는 바쁠 때도 추적한다(변화 감지가 어긋나지 않게).
        let mut grip_changed = false;
        for (i, &g) in grips.iter().enumerate() {
            let now = if g < self.cfg.grip_closed_below {
                Some(false)
            } else if g > self.cfg.grip_open_above {
                Some(true)
            } else {
                None
            };
            if let Some(now) = now {
                match self.grip_state[i] {
                    None => self.grip_state[i] = Some(now),
                    Some(prev) if prev != now => {
                        self.grip_cand[i] += 1;
                        if self.grip_cand[i] >= self.cfg.grip_hold_steps {
                            self.grip_state[i] = Some(now);
                            self.grip_cand[i] = 0;
                            grip_changed = true;
                        }
                    }
                    _ => self.grip_cand[i] = 0,
                }
            }
        }
        if grip_changed {
            self.grip_fired = false;
        }
        let (v, w) = speed;
        if v < self.cfg.settle_speed && w < self.cfg.settle_yaw_rate {
            self.still = self.still.saturating_add(1);
        } else {
            self.still = 0;
        }

        if !self.started {
            self.started = true;
            return Some(Trigger::EpisodeStart);
        }
        if self.busy {
            return None;
        }
        if let Some(lt) = self.last_trigger {
            if self.step < lt + self.cfg.min_gap {
                return None;
            }
        }
        if self.steps_in_stage() >= self.budget {
            return Some(Trigger::BudgetExhausted);
        }
        if self.nav && !self.settled_fired {
            let moved = (dist - self.dist_at_start) + 0.5 * (turned - self.turn_at_start);
            if moved >= self.cfg.settle_min_move && self.still >= self.cfg.settle_steps {
                self.settled_fired = true;
                return Some(Trigger::Settled);
            }
        }
        if grip_changed && !self.nav && !self.grip_fired && self.steps_in_stage() >= self.cfg.min_gap {
            self.grip_fired = true;
            return Some(Trigger::GripperChange);
        }
        if self.step >= self.next_check {
            return Some(Trigger::CheckDue);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_and_check() {
        let mut m = Monitor::new(MonitorCfg::default());
        assert_eq!(m.on_step(0.0, 0.0, (0.0, 0.0), [0.1, 0.1]), Some(Trigger::EpisodeStart));
        m.begin_stage(200, 100, false, 0.0, 0.0);
        let mut got = vec![];
        for _ in 0..250 {
            m.advance();
            if let Some(t) = m.on_step(0.0, 0.0, (0.0, 0.0), [0.1, 0.1]) {
                got.push((m.step, t));
                if t == Trigger::CheckDue {
                    m.extend(0, 100);
                }
            }
        }
        assert_eq!(got[0], (100, Trigger::CheckDue));
        assert!(got.iter().any(|&(s, t)| t == Trigger::BudgetExhausted && s == 200));
    }

    #[test]
    fn settle_after_moving() {
        let mut m = Monitor::new(MonitorCfg::default());
        m.on_step(0.0, 0.0, (0.0, 0.0), [0.1, 0.1]);
        m.begin_stage(10_000, 10_000, true, 0.0, 0.0);
        let mut d = 0.0;
        let mut fired = None;
        for i in 0..200 {
            m.advance();
            let v = if i < 60 { 0.5 } else { 0.0 };
            d += v / 30.0;
            if let Some(t) = m.on_step(d, 0.0, (v, 0.0), [0.1, 0.1]) {
                fired = Some((m.step, t));
                break;
            }
        }
        let (s, t) = fired.unwrap();
        assert_eq!(t, Trigger::Settled);
        assert!((80..=82).contains(&s), "{s}");
    }

    #[test]
    fn gripper_close_fires_once() {
        let mut m = Monitor::new(MonitorCfg::default());
        m.on_step(0.0, 0.0, (0.0, 0.0), [0.1, 0.1]);
        m.begin_stage(10_000, 10_000, false, 0.0, 0.0);
        let mut n = 0;
        for i in 0..200 {
            m.advance();
            let g = if i < 50 { 0.1 } else { 0.03 };
            if m.on_step(0.0, 0.0, (0.0, 0.0), [g, 0.1]) == Some(Trigger::GripperChange) {
                n += 1;
            }
        }
        assert_eq!(n, 1);
    }
}
