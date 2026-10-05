//! 중계기(다리): 평가기 ↔ [중계기] ↔ VLA 서버.
//!
//! 매 스텝 경로(병목 제로 목표):
//! - 평가기 프레임(클라이언트 → 서버, XOR 마스크됨)의 **마스크를 풀지 않는다**. 맵 머리 길이가 바뀐 만큼 마스크 키를
//!   회전시켜([`ws::rotate_key`]) 원래 바이트를 그대로 다시 보낸다(재마스크·재직렬화 0회).
//! - **흘려보내기(cut-through)**: 경계가 아닌 스텝은 프레임 머리를 받자마자 새 머리(항목 수 + 주입 수)를 먼저 보내고,
//!   몸통은 받는 조각마다 바로 VLA 쪽으로 쓴다. 지시 문장은 이전 스텝에 이미 정해져 있어 꼬리(주입 항목)를 미리 안다.
//!   다 받은 뒤에 마스크된 채로 훑어 base_qvel·그리퍼만 읽어(수십 바이트) 오도메트리·감시를 갱신한다.
//! - **붙잡기(store)**: 스텝 수로 정해지는 경계(판 시작·예산 소진·정기 확인)가 올 스텝, 또는 직전 스텝에서 사건 경계
//!   (이동 멈춤·그리퍼 변화)가 난 다음 스텝은 프레임을 통째로 받아 두고 계획이 끝난 뒤 보낸다(`pause`).
//!   사건 경계는 그래서 한 스텝(1/30 s) 늦게 반영된다. 작은 프레임(reset 등)도 붙잡는다.
//! - VLA 응답(서버 → 클라이언트, 마스크 없음)은 그대로 평가기에 넘긴다.
//! - LLM 은 이 경로에 없다. 계획은 별도 스레드. `pause` 면 결정이 올 때까지 기다리며 두 소켓의 ping 에 답한다
//!   (평가기 ping_timeout 300 s, VLA 서버 기본 20 s). `pause` 가 아니면 붙잡지 않고, 결정이 도착한 스텝부터 문장을 바꾼다.
//!
//! 관측·행동 값은 한 바이트도 바꾸지 않는다: 원래 항목 바이트는 그대로 가고, 주입 키(`__agent_prompt__`,
//! `__agent_flush__`)는 VLA 서버 쪽 훅이 꺼내 쓰고 지운다(옛 훅 tools/serve_b1k_agent.py 는 10-06 지움).

use crate::catalog::{Catalog, TaskCard};
use crate::monitor::{MonitorCfg, Trigger};
use crate::msgpack::{self, AnySrc, Masked, Plain};
use crate::planner::{Agent, Decision};
use crate::session::EnvSession;
use crate::trace::Tracer;
use crate::wire::{self, Cam};
use crate::ws::{self, Accepted, Buf, Conn, Head, OP_BIN, OP_CLOSE, OP_CONT, OP_PING, OP_PONG};
use serde_json::json;
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum Mode {
    /// 아무것도 주입하지 않고 그대로(오버헤드 기준선)
    Passthrough,
    /// 고정 문장 주입
    Fixed(String),
    /// 스텝별 문장 표(시연 구간대로 바꾸기 등): (시작 스텝, 문장)
    Schedule(Vec<(u64, String)>),
    /// 에이전트가 경계마다 정한다
    Agent,
}

pub type AgentFactory = Arc<dyn Fn(usize, &TaskCard) -> Agent + Send + Sync>;

#[derive(Clone)]
pub struct RelayCfg {
    pub listen: String,
    pub upstream: String,
    pub mode: Mode,
    pub catalog: Arc<Catalog>,
    pub task_override: Option<String>,
    pub max_steps_override: Option<u64>,
    pub monitor: MonitorCfg,
    pub pause: bool,
    /// 흘려보내기(cut-through). 끄면 모든 프레임을 다 받은 뒤 보낸다(비교용).
    pub stream: bool,
    /// 이보다 작은 프레임은 붙잡는다(reset 등)
    pub stream_min_bytes: usize,
    pub images: bool,
    pub image_side: usize,
    pub jpeg_quality: u8,
    pub trace_dir: Option<PathBuf>,
    pub factory: Option<AgentFactory>,
    /// 연결 하나만 처리하고 끝(시험용)
    pub once: bool,
    /// 이 스텝마다 지연 통계를 기록
    pub latency_every: u64,
    /// pause 에서 결정을 기다리는 최대 시간(넘으면 지금 문장으로 계속, 결정은 도착 시 반영)
    pub hard_wait_s: f64,
    pub hz: f64,
}

impl RelayCfg {
    pub fn new(listen: &str, upstream: &str, mode: Mode, catalog: Arc<Catalog>) -> RelayCfg {
        RelayCfg {
            listen: listen.into(),
            upstream: upstream.into(),
            mode,
            catalog,
            task_override: None,
            max_steps_override: None,
            monitor: MonitorCfg::default(),
            pause: true,
            stream: true,
            stream_min_bytes: 64 * 1024,
            images: true,
            image_side: 448,
            jpeg_quality: 80,
            trace_dir: None,
            factory: None,
            once: false,
            latency_every: 1000,
            hard_wait_s: 240.0,
            hz: 30.0,
        }
    }
}

enum PMsg {
    Decision(usize, Decision),
    Returned(usize, Box<Agent>),
}

struct Slot {
    sess: EnvSession,
    agent: Option<Box<Agent>>,
    /// 에이전트를 가진 환경인가(과제를 모르면 없다)
    owns: bool,
    /// 결정을 기다리는 중
    waiting: bool,
}

fn idle(s: &Slot) -> bool {
    !s.owns || (s.agent.is_some() && !s.waiting)
}

/// 스텝마다 잰 시간(µs)
#[derive(Default)]
pub struct Lat {
    pub parse: Vec<u32>,
    pub hold_fwd: Vec<u32>,
    pub upstream: Vec<u32>,
    pub back: Vec<u32>,
    pub plan_wait: Vec<u32>,
    pub streamed: u64,
    pub held: u64,
}

fn pct(v: &[u32]) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let p = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    json!({"n": s.len(), "p50": p(0.5), "p90": p(0.9), "p99": p(0.99), "max": s[s.len() - 1]})
}

impl Lat {
    pub fn summary(&self) -> serde_json::Value {
        json!({"unit": "us", "streamed_frames": self.streamed, "held_frames": self.held, "parse": pct(&self.parse),
               "hold_forward": pct(&self.hold_fwd), "upstream_rtt": pct(&self.upstream), "back": pct(&self.back), "plan_wait": pct(&self.plan_wait)})
    }
}

pub fn run(cfg: RelayCfg) -> io::Result<()> {
    let l = TcpListener::bind(&cfg.listen)?;
    run_listener(l, cfg)
}

/// 이미 연 소켓으로(시험: 포트 0 으로 열고 주소를 먼저 알아 둔다).
pub fn run_listener(l: TcpListener, cfg: RelayCfg) -> io::Result<()> {
    eprintln!(
        "[relay] {} ← 평가기, → VLA {} (모드 {}, pause {}, stream {})",
        l.local_addr().map(|a| a.to_string()).unwrap_or_default(),
        cfg.upstream,
        mode_name(&cfg.mode),
        cfg.pause,
        cfg.stream
    );
    let tracer = match &cfg.trace_dir {
        Some(d) => Some(Tracer::create(d)?),
        None => None,
    };
    for s in l.incoming() {
        let s = match s {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[relay] accept 오류: {e}");
                continue;
            }
        };
        let cfg2 = cfg.clone();
        let tr = tracer.clone();
        let up = cfg.upstream.clone();
        let health = move || crate::http::get(&format!("http://{up}/healthz"), Duration::from_secs(2)).map(|r| r.status == 200).unwrap_or(false);
        let conn = match Conn::accept_with(s, health) {
            Ok(Accepted::Ws(c)) => c,
            Ok(Accepted::Http(_)) => continue,
            Err(e) => {
                eprintln!("[relay] 핸드셰이크 오류: {e}");
                continue;
            }
        };
        if cfg.once {
            let r = serve(cfg2, conn, tr.clone());
            if let Some(t) = &tr {
                t.flush();
            }
            return r.map(|_| ());
        }
        std::thread::spawn(move || {
            if let Err(e) = serve(cfg2, conn, tr.clone()) {
                eprintln!("[relay] 연결 끝: {e}");
            }
            if let Some(t) = &tr {
                t.flush();
            }
        });
    }
    Ok(())
}

fn mode_name(m: &Mode) -> &'static str {
    match m {
        Mode::Passthrough => "passthrough",
        Mode::Fixed(_) => "fixed",
        Mode::Schedule(_) => "schedule",
        Mode::Agent => "agent",
    }
}

fn io_bad(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

struct Ctx<'a> {
    cfg: &'a RelayCfg,
    tracer: &'a Option<Tracer>,
    slots: Vec<Slot>,
    task: Option<TaskCard>,
    episode: u32,
    batched: bool,
    /// 직전 스텝에서 난(흘려보낸 프레임이라 아직 처리 못 한) 사건 경계
    pending: Vec<(usize, Trigger, [f64; 2])>,
    tx: mpsc::Sender<PMsg>,
    rx: mpsc::Receiver<PMsg>,
    suffix: Vec<u8>,
    scratch: Vec<u8>,
    cbuf: Buf,
}

impl Ctx<'_> {
    fn trace(&self, env: usize, kind: &str, v: serde_json::Value) {
        if let Some(t) = self.tracer {
            t.event(env, kind, v);
        }
    }

    /// 이번 프레임을 붙잡아야 하나(흘려보낼 수 없나)
    fn must_hold(&self, h: &Head) -> bool {
        if !self.cfg.stream || h.mask.is_none() || h.len < self.cfg.stream_min_bytes || !h.fin {
            return true;
        }
        match &self.cfg.mode {
            Mode::Passthrough => false,
            Mode::Fixed(_) | Mode::Schedule(_) => self.slots.is_empty(),
            Mode::Agent => {
                if self.slots.is_empty() {
                    return true;
                }
                if !self.cfg.pause {
                    return false;
                }
                // 사건 경계가 미뤄져 있거나, 스텝 수로 정해진 경계가 올 스텝만 붙잡는다
                !self.pending.is_empty() || self.slots.iter().any(|s| s.owns && s.sess.mon.scheduled_due())
            }
        }
    }

    /// 과제·환경 수가 정해지면 환경마다 세션·에이전트를 만든다.
    fn ensure_slots(&mut self, view: &wire::ObsView) {
        if self.task.is_none() {
            self.task = view.task_id.and_then(|i| self.cfg.catalog.task_by_index(i)).cloned();
            self.trace(0, "task", json!({"task_id": view.task_id, "task": self.task.as_ref().map(|t| t.name.clone())}));
        }
        self.batched = view.batched;
        if self.slots.len() == view.batch {
            return;
        }
        let (tp, ms) = self.task.as_ref().map(|t| (t.prompt.clone(), t.max_steps)).unwrap_or_default();
        let ms = self.cfg.max_steps_override.unwrap_or(if ms == 0 { 100_000 } else { ms });
        let cfg = self.cfg;
        let task = self.task.clone();
        let tracer = self.tracer.clone();
        let episode = self.episode;
        self.slots = (0..view.batch)
            .map(|e| {
                let mut agent = match (&cfg.mode, &cfg.factory, &task) {
                    (Mode::Agent, Some(f), Some(t)) => Some(Box::new(f(e, t))),
                    _ => None,
                };
                if let Some(a) = agent.as_mut() {
                    a.set_tracer(tracer.clone());
                    a.reset_episode(episode);
                }
                Slot { sess: EnvSession::new(e, cfg.monitor.clone(), cfg.hz, ms, &tp), owns: agent.is_some(), agent, waiting: false }
            })
            .collect();
    }

    /// 고정·표 모드의 이번 스텝 문장
    fn update_static_prompts(&mut self) {
        match &self.cfg.mode {
            Mode::Fixed(p) => {
                for s in self.slots.iter_mut() {
                    s.sess.prompt = p.clone();
                }
            }
            Mode::Schedule(tab) => {
                for s in self.slots.iter_mut() {
                    if let Some((_, p)) = tab.iter().rev().find(|(st, _)| *st <= s.sess.mon.step) {
                        if *p != s.sess.prompt {
                            s.sess.prompt = p.clone();
                            s.sess.pending_flush = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// 주입 꼬리(평문)를 만들고 항목 수를 돌려준다.
    fn build_suffix(&mut self) -> usize {
        let inject = match self.cfg.mode {
            Mode::Passthrough => false,
            Mode::Agent => self.task.is_some(),
            _ => true,
        };
        if !inject || self.slots.is_empty() {
            self.suffix.clear();
            return 0;
        }
        let prompts: Vec<String> = self.slots.iter().map(|s| s.sess.prompt.clone()).collect();
        let any_flush = self.slots.iter().any(|s| s.sess.pending_flush);
        let flush: Vec<bool> = self.slots.iter().map(|s| s.sess.pending_flush).collect();
        for s in self.slots.iter() {
            if s.sess.pending_flush {
                self.trace(s.sess.env, "prompt_applied", json!({"step": s.sess.mon.step, "prompt": s.sess.prompt}));
            }
        }
        for s in self.slots.iter_mut() {
            s.sess.pending_flush = false;
        }
        wire::build_suffix(&prompts, if any_flush { Some(&flush) } else { None }, self.batched, &mut self.suffix)
    }

    /// 계획기 스레드로 넘긴다.
    fn dispatch(&mut self, e: usize, t: Trigger, g: [f64; 2], view: &wire::ObsView, src: &AnySrc) {
        let Some(mut agent) = self.slots[e].agent.take() else { return };
        let images = if self.cfg.images { snapshot(view, src, e, self.cfg.image_side, self.cfg.jpeg_quality) } else { vec![] };
        let mut bev = self.slots[e].sess.event(t, g, images);
        if let Some(tr) = self.tracer {
            bev.image_files = bev.images.iter().map(|(c, j)| tr.save_image(&format!("ep{}_env{}_s{}_{}.jpg", bev.episode, e, bev.step, c), j)).collect();
        }
        self.slots[e].waiting = true;
        self.slots[e].sess.mon.busy = true;
        let txc = self.tx.clone();
        std::thread::spawn(move || {
            let d = agent.decide(&bev);
            let _ = txc.send(PMsg::Decision(e, d));
            agent.maintain();
            agent.release();
            let _ = txc.send(PMsg::Returned(e, agent));
        });
    }

    fn drain(&mut self) {
        while let Ok(m) = self.rx.try_recv() {
            self.handle(m);
        }
    }

    fn handle(&mut self, m: PMsg) {
        match m {
            PMsg::Decision(e, d) => {
                if let Some(s) = self.slots.get_mut(e) {
                    s.sess.apply(&d);
                    s.waiting = false;
                    s.sess.mon.busy = false;
                    let step = s.sess.mon.step;
                    self.trace(e, "decision_applied", json!({"step": step, "kind": d.key()}));
                }
            }
            PMsg::Returned(e, a) => {
                if let Some(s) = self.slots.get_mut(e) {
                    s.agent = Some(a);
                }
            }
        }
    }

    /// 조건이 맞을 때까지 두 소켓의 제어 프레임(ping)에 답하며 기다린다.
    fn pump_until(&mut self, done: &dyn Fn(&[Slot]) -> bool, ev: &mut Conn, up: &mut Conn) -> io::Result<bool> {
        let t0 = Instant::now();
        loop {
            self.drain();
            if done(&self.slots) {
                return Ok(true);
            }
            if t0.elapsed().as_secs_f64() > self.cfg.hard_wait_s {
                return Ok(false);
            }
            let ready = ws::poll_readable(&[&*ev, &*up], 20)?;
            if ready[0] && !ev.service_control(&mut self.cbuf)? {
                return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "평가기가 계획 중에 연결을 닫음"));
            }
            if ready[1] && !up.service_control(&mut self.cbuf)? {
                return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "VLA 서버가 계획 중에 연결을 닫음"));
            }
        }
    }

    fn wait_all_idle(&mut self, ev: &mut Conn, up: &mut Conn) -> io::Result<()> {
        if self.slots.iter().all(idle) {
            return Ok(());
        }
        self.pump_until(&|s: &[Slot]| s.iter().all(idle), ev, up).map(|_| ())
    }
}

/// 평가기 연결 하나를 처리한다. 반환: 지연 통계.
pub fn serve(cfg: RelayCfg, mut ev: Conn, tracer: Option<Tracer>) -> io::Result<Lat> {
    let mut up = Conn::connect(&cfg.upstream)?;
    let mut ebuf = Buf::default();
    let mut ubuf = Buf::default();
    let mut lat = Lat::default();

    // 서버가 먼저 보내는 metadata 를 그대로 넘긴다
    let (op, _) = up.read_message(&mut ubuf)?;
    ev.write_frame_raw(true, op, None, &[ubuf.data()])?;

    let (tx, rx) = mpsc::channel::<PMsg>();
    let task = cfg.task_override.as_ref().and_then(|n| cfg.catalog.task_by_name(n).cloned());
    let mut c = Ctx {
        cfg: &cfg,
        tracer: &tracer,
        slots: Vec::new(),
        task,
        episode: 0,
        batched: true,
        pending: Vec::new(),
        tx,
        rx,
        suffix: Vec::with_capacity(1024),
        scratch: Vec::with_capacity(1024),
        cbuf: Buf::default(),
    };
    let mut steps: u64 = 0;
    c.trace(0, "relay_start", json!({"mode": mode_name(&cfg.mode), "pause": cfg.pause, "stream": cfg.stream, "upstream": cfg.upstream}));

    loop {
        let h = ev.read_head()?;
        // 제어 프레임
        if matches!(h.op, OP_PING | OP_PONG | OP_CLOSE) {
            let dst = c.cbuf.ensure(h.len);
            ev.read_exact(dst)?;
            if let Some(k) = h.mask {
                ws::mask_at(c.cbuf.data_mut(), k, 0);
            }
            match h.op {
                OP_PING => {
                    let p = c.cbuf.data().to_vec();
                    ev.send(ws::OP_PONG, &[&p])?;
                }
                OP_CLOSE => {
                    let _ = up.send_close(1000);
                    break;
                }
                _ => {}
            }
            continue;
        }
        c.drain();
        let hold = c.must_hold(&h);
        let t_in = Instant::now();
        let mut plan_us = 0u32;

        if !hold {
            // ---------- 흘려보내기 ----------
            c.update_static_prompts();
            let extra = c.build_suffix();
            stream_frame(&mut ev, &mut up, &h, &mut ebuf, &c.suffix, extra, &mut c.scratch)?;
            let t_f = Instant::now();
            lat.streamed += 1;
            // 다 받은 뒤: 마스크된 채로 훑어 오도메트리·감시 갱신
            if !matches!(cfg.mode, Mode::Passthrough) {
                let src = AnySrc::M(Masked { buf: ebuf.data(), key: h.mask.unwrap() });
                let view = wire::view(&src).map_err(io_bad)?;
                let mut trig = Vec::new();
                for (e, s) in c.slots.iter_mut().enumerate() {
                    let g = view.grippers(&src, e);
                    if let Some(t) = s.sess.on_obs(view.base_qvel(&src, e), g) {
                        trig.push((e, t, g));
                    }
                }
                if matches!(cfg.mode, Mode::Agent) {
                    if cfg.pause {
                        c.pending.extend(trig);
                    } else {
                        for (e, t, g) in trig {
                            if c.slots[e].owns && c.slots[e].agent.is_some() && !c.slots[e].waiting {
                                c.dispatch(e, t, g, &view, &src);
                            }
                        }
                    }
                }
                lat.parse.push(us(t_f.elapsed()));
            }
            let t_p = Instant::now();
            let (op2, _) = up.read_message(&mut ubuf)?;
            let t_r = Instant::now();
            ev.write_frame_raw(true, op2, None, &[ubuf.data()])?;
            lat.upstream.push(us(t_r - t_p));
            lat.back.push(us(t_r.elapsed()));
        } else {
            // ---------- 붙잡기 ----------
            let (op, mask) = read_rest(&mut ev, &h, &mut ebuf)?;
            if op != OP_BIN {
                forward_raw(&mut up, op, mask, ebuf.data())?;
                let (op2, _) = up.read_message(&mut ubuf)?;
                ev.write_frame_raw(true, op2, None, &[ubuf.data()])?;
                continue;
            }
            let src = match mask {
                Some(k) => AnySrc::M(Masked { buf: ebuf.data(), key: k }),
                None => AnySrc::P(Plain(ebuf.data())),
            };
            let view = wire::view(&src).map_err(io_bad)?;
            if view.is_reset {
                // 새 판: 계획기가 돌아올 때까지 기다린 뒤 모두 초기화. reset 은 응답이 없다.
                c.wait_all_idle(&mut ev, &mut up)?;
                c.episode += 1;
                c.pending.clear();
                let ep = c.episode;
                for s in c.slots.iter_mut() {
                    s.sess.reset(ep);
                    if let Some(a) = s.agent.as_mut() {
                        a.reset_episode(ep);
                    }
                }
                c.trace(0, "episode_reset", json!({"episode": ep, "steps_total": steps}));
                forward_raw(&mut up, op, mask, ebuf.data())?;
                continue;
            }
            if matches!(cfg.mode, Mode::Passthrough) {
                forward_raw(&mut up, op, mask, ebuf.data())?;
            } else {
                c.ensure_slots(&view);
                // 매 스텝: 오도메트리·감시(산술만) + 직전 스텝에서 미뤄 둔 사건 경계
                let mut trig: Vec<(usize, Trigger, [f64; 2])> = std::mem::take(&mut c.pending);
                for (e, s) in c.slots.iter_mut().enumerate() {
                    let g = view.grippers(&src, e);
                    if let Some(t) = s.sess.on_obs(view.base_qvel(&src, e), g) {
                        if !trig.iter().any(|x| x.0 == e) {
                            trig.push((e, t, g));
                        }
                    }
                }
                if matches!(cfg.mode, Mode::Agent) && !trig.is_empty() {
                    let tp = Instant::now();
                    for (e, t, g) in trig {
                        if !c.slots[e].owns || c.slots[e].waiting {
                            continue;
                        }
                        if c.slots[e].agent.is_none() {
                            if !cfg.pause {
                                continue;
                            }
                            // 앞 결정의 기억 정리가 아직이면 기다린다(결정 순서 고정)
                            c.pump_until(&|s: &[Slot]| s[e].agent.is_some(), &mut ev, &mut up)?;
                        }
                        c.dispatch(e, t, g, &view, &src);
                    }
                    if cfg.pause && !c.pump_until(&|s: &[Slot]| s.iter().all(|x| !x.waiting), &mut ev, &mut up)? {
                        eprintln!("[relay] 결정이 {} s 안에 안 옴 — 지금 문장으로 계속, 결정은 도착하면 반영", cfg.hard_wait_s);
                    }
                    plan_us = us(tp.elapsed());
                }
                c.update_static_prompts();
                let extra = c.build_suffix();
                forward_obs(&mut up, ebuf.data(), mask, &view.top, &c.suffix, extra, &mut c.scratch)?;
            }
            let t_f = Instant::now();
            lat.held += 1;
            let (op2, _) = up.read_message(&mut ubuf)?;
            let t_r = Instant::now();
            ev.write_frame_raw(true, op2, None, &[ubuf.data()])?;
            lat.hold_fwd.push(us((t_f - t_in).saturating_sub(Duration::from_micros(plan_us as u64))));
            lat.upstream.push(us(t_r - t_f));
            lat.back.push(us(t_r.elapsed()));
            if plan_us > 0 {
                lat.plan_wait.push(plan_us);
            }
        }
        for s in c.slots.iter_mut() {
            s.sess.mon.advance();
        }
        steps += 1;
        if cfg.latency_every > 0 && steps % cfg.latency_every == 0 {
            c.trace(0, "latency", json!({"steps": steps, "stats": lat.summary()}));
        }
    }
    let _ = c.wait_all_idle(&mut ev, &mut up);
    c.trace(0, "relay_end", json!({"steps": steps, "stats": lat.summary()}));
    for s in &c.slots {
        if let Some(a) = &s.agent {
            c.trace(s.sess.env, "agent_stats", json!(a.core.stats));
        }
    }
    Ok(lat)
}

#[inline]
fn us(d: Duration) -> u32 {
    d.as_micros().min(u32::MAX as u128) as u32
}

/// 머리를 읽은 프레임의 나머지를 받는다. 조각난 메시지는 이어 붙여 평문(마스크 None)으로.
fn read_rest(ev: &mut Conn, h: &Head, buf: &mut Buf) -> io::Result<(u8, Option<[u8; 4]>)> {
    let dst = buf.ensure(h.len);
    ev.read_exact(dst)?;
    if h.fin {
        return Ok((h.op, h.mask));
    }
    let mut all = buf.data().to_vec();
    if let Some(k) = h.mask {
        ws::mask_at(&mut all, k, 0);
    }
    let mut tmp = Buf::default();
    loop {
        let h2 = ev.read_frame(&mut tmp)?;
        match h2.op {
            OP_PING => {
                let mut p = tmp.data().to_vec();
                if let Some(k) = h2.mask {
                    ws::mask_at(&mut p, k, 0);
                }
                ev.send(OP_PONG, &[&p])?;
            }
            OP_PONG => {}
            OP_CONT => {
                let s = all.len();
                all.extend_from_slice(tmp.data());
                if let Some(k) = h2.mask {
                    ws::mask_at(&mut all[s..], k, 0);
                }
                if h2.fin {
                    break;
                }
            }
            o => return Err(io_bad(format!("조각 사이에 op {o}"))),
        }
    }
    buf.ensure(all.len()).copy_from_slice(&all);
    Ok((h.op, None))
}

/// 흘려보내기: 새 머리를 먼저 보내고, 몸통은 받는 대로 쓰고, 마지막에 꼬리.
fn stream_frame(ev: &mut Conn, up: &mut Conn, h: &Head, ebuf: &mut Buf, suffix: &[u8], extra: usize, scratch: &mut Vec<u8>) -> io::Result<()> {
    let k = h.mask.expect("흘려보내기는 마스크된 프레임만");
    let len = h.len;
    let dst = ebuf.ensure(len);
    ev.read_exact(&mut dst[..1])?;
    let b0 = dst[0] ^ k[0];
    let hl = match b0 {
        0x80..=0x8f => 1,
        0xde => 3,
        0xdf => 5,
        _ => 0,
    };
    let (new_len, k2, head_part): (usize, [u8; 4], Vec<u8>);
    let body_start;
    if hl == 0 || extra == 0 {
        // 맵이 아니거나 주입 없음: 머리·키 그대로
        new_len = len;
        k2 = k;
        head_part = vec![dst[0]];
        body_start = 1;
    } else {
        ev.read_exact(&mut dst[1..hl])?;
        let mut hb = [0u8; 5];
        for i in 0..hl {
            hb[i] = dst[i] ^ k[i & 3];
        }
        let count = match hl {
            1 => (hb[0] & 0x0f) as usize,
            3 => u16::from_be_bytes([hb[1], hb[2]]) as usize,
            _ => u32::from_be_bytes([hb[1], hb[2], hb[3], hb[4]]) as usize,
        };
        let (nh, nhl) = msgpack::map_hdr(count + extra);
        k2 = ws::rotate_key(k, nhl as isize - hl as isize);
        let mut hp = nh[..nhl].to_vec();
        ws::mask_at(&mut hp, k2, 0);
        head_part = hp;
        new_len = len - hl + nhl + suffix.len();
        body_start = hl;
    }
    up.write_head(true, h.op, Some(k2), new_len)?;
    up.write_raw(&[&head_part])?;
    let mut off = body_start;
    while off < len {
        let n = ev.read_some(&mut dst[off..])?;
        up.write_raw(&[&dst[off..off + n]])?;
        off += n;
    }
    if extra > 0 && hl > 0 {
        scratch.clear();
        scratch.extend_from_slice(suffix);
        ws::mask_at(scratch, k2, new_len - suffix.len());
        up.write_raw(&[scratch])?;
    }
    Ok(())
}

/// 프레임을 바이트 그대로 넘긴다(같은 마스크 키).
fn forward_raw(up: &mut Conn, op: u8, mask: Option<[u8; 4]>, data: &[u8]) -> io::Result<()> {
    match mask {
        Some(k) => up.write_frame_raw(true, op, Some(k), &[data]),
        None => up.send(op, &[data]),
    }
}

/// 다 받은 관측 + 주입 꼬리를 복사 없이 보낸다.
pub fn forward_obs(up: &mut Conn, data: &[u8], mask: Option<[u8; 4]>, top: &msgpack::TopMap, suffix: &[u8], extra: usize, scratch: &mut Vec<u8>) -> io::Result<()> {
    let (nh, nhl) = msgpack::map_hdr(top.count + extra);
    let old = top.hdr_len;
    let body = &data[old..];
    match mask {
        Some(k) if extra == 0 => up.write_frame_raw(true, OP_BIN, Some(k), &[data]),
        Some(k) => {
            let k2 = ws::rotate_key(k, nhl as isize - old as isize);
            let mut h = [0u8; 5];
            h[..nhl].copy_from_slice(&nh[..nhl]);
            ws::mask_at(&mut h[..nhl], k2, 0);
            scratch.clear();
            scratch.extend_from_slice(suffix);
            ws::mask_at(scratch, k2, nhl + body.len());
            up.write_frame_raw(true, OP_BIN, Some(k2), &[&h[..nhl], body, scratch])
        }
        None => {
            // 조각 모음 등 평문: 새 키로 전부 마스크(드문 경로)
            let k2 = up.new_mask_key();
            scratch.clear();
            scratch.extend_from_slice(&nh[..nhl]);
            scratch.extend_from_slice(body);
            scratch.extend_from_slice(suffix);
            ws::mask_at(scratch, k2, 0);
            up.write_frame_raw(true, OP_BIN, Some(k2), &[scratch])
        }
    }
}

/// 경계 순간의 카메라 영상 → 축소 → JPEG.
fn snapshot(view: &wire::ObsView, src: &AnySrc, env: usize, side: usize, q: u8) -> Vec<(String, Vec<u8>)> {
    [Cam::Head, Cam::LeftWrist, Cam::RightWrist]
        .iter()
        .filter_map(|&c| view.image(src, c, env).map(|img| (c.name().to_string(), img.downscale(if c == Cam::Head { side } else { side / 2 }).jpeg(q))))
        .collect()
}
