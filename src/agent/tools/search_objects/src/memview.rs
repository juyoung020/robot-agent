//! 물체 기억 보기 — [`Memory`] trait(plan.md 3.2 `memview.rs`)와 오프라인 구현 [`ViewJson`](기억 폴더의 `view.json`).
//!
//! 도구가 기억에서 읽는 것은 이것뿐이다: 물체(id·등록 이름·자리·크기·상태·마지막으로 본 시각·방·고정 여부), 방 이름,
//! 기억 시각(`stamp`), 로봇 자세(`pose`). 이름 사후·벡터는 공용 색인(sys.rs)이 따로 읽는다.
//! 실시간 판은 [`crate::live::LiveMem`](scenemap `sm_snapshot`) — 같은 trait 라 도구 쪽 코드는 그대로.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjInfo {
    pub id: u32,
    /// 등록 이름(view.json `name` — 지도 쪽이 붙인 것)
    pub name: String,
    pub pos: [f64; 3],
    pub extent: [f64; 3],
    /// seen | moved | held | gone
    pub state: String,
    pub last_seen: f64,
    pub room: Option<i64>,
    pub movable: bool,
    pub structural: bool,
    pub n_obs: u32,
    /// A′ 위치 불확실도 [σx, σy, σz] m(view.json `pos_sd`, 없으면 None)
    pub pos_sd: Option<[f64; 3]>,
}

impl ObjInfo {
    pub fn key(&self) -> String {
        format!("O{}", self.id)
    }
    pub fn top_z(&self) -> f64 {
        self.pos[2] + 0.5 * self.extent[2]
    }
    /// 수평 거리: 점에서 이 물체 상자(xy)까지(안이면 0)
    pub fn xy_dist_to_box(&self, x: f64, y: f64) -> f64 {
        let dx = ((x - self.pos[0]).abs() - 0.5 * self.extent[0]).max(0.0);
        let dy = ((y - self.pos[1]).abs() - 0.5 * self.extent[1]).max(0.0);
        dx.hypot(dy)
    }
}

pub trait Memory {
    fn objects(&self) -> &[ObjInfo];
    /// 기억 시각(s, 시뮬·bag 시계) — `last_seen` 과 같은 시계
    fn now(&self) -> f64;
    /// 로봇 자세 [x, y, yaw](map), 모르면 None
    fn pose(&self) -> Option<[f64; 3]>;
    fn room_name(&self, room: i64) -> Option<String>;
    /// "R2" · "2" · 방 이름(대소문자 무시) → 방 번호들
    fn find_rooms(&self, s: &str) -> Vec<i64>;

    fn get(&self, id: u32) -> Option<&ObjInfo> {
        self.objects().iter().find(|o| o.id == id)
    }
    /// 물체 id 둘레 r m 안의 물체(자기 빼고, 가까운 순): (id, 수평 거리)
    fn near(&self, id: u32, r: f64) -> Vec<(u32, f64)> {
        let Some(c) = self.get(id) else { return vec![] };
        let mut v: Vec<(u32, f64)> = self
            .objects()
            .iter()
            .filter(|o| o.id != id)
            .map(|o| (o.id, o.xy_dist_to_box(c.pos[0], c.pos[1])))
            .filter(|(_, d)| *d <= r)
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v
    }
}

/// 도구가 쓰는 기억: 보기([`Memory`]) + 새로 읽기 + (실시간이면) 지도에 이름 관측 넣기.
pub trait MemSource: Send {
    fn mem(&self) -> &dyn Memory;
    /// 새로 읽음(바뀌었으면 true). 실시간은 늘 새 스냅숏
    fn refresh(&mut self) -> Result<bool, String>;
    /// 지도(scenemap A′)에 바깥 이름 관측: `sm_observe_object_name` 반환값(0 성공, -2 라벨 없음, -3 물체 없음·A′ 아님). 오프라인은 None
    fn observe_name(&mut self, _id: u32, _label: &str, _log_lr: f32) -> Option<i32> {
        None
    }
    /// "view_json" | "live"
    fn kind(&self) -> &'static str;
}

impl MemSource for ViewJson {
    fn mem(&self) -> &dyn Memory {
        self
    }
    fn refresh(&mut self) -> Result<bool, String> {
        ViewJson::refresh(self)
    }
    fn kind(&self) -> &'static str {
        "view_json"
    }
}

/// `view.json`(scenemap 저장 / sgrt 기억 폴더) 읽기. 파일이 바뀌면 [`ViewJson::refresh`] 가 다시 읽는다.
#[derive(Clone, Debug, Default)]
pub struct ViewJson {
    pub path: PathBuf,
    mtime: Option<SystemTime>,
    objs: Vec<ObjInfo>,
    stamp: f64,
    pose: Option<[f64; 3]>,
    rooms: Vec<(i64, String)>,
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

pub(crate) fn arr3(v: &Value) -> [f64; 3] {
    let a = v.as_array().cloned().unwrap_or_default();
    [a.first().map(f).unwrap_or(0.0), a.get(1).map(f).unwrap_or(0.0), a.get(2).map(f).unwrap_or(0.0)]
}

impl ViewJson {
    pub fn load(mem_dir: &Path) -> Result<ViewJson, String> {
        let mut v = ViewJson { path: mem_dir.join("view.json"), ..Default::default() };
        v.read()?;
        Ok(v)
    }

    fn read(&mut self) -> Result<(), String> {
        let s = std::fs::read_to_string(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let j: Value = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.mtime = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        self.stamp = f(&j["stamp"]);
        self.pose = j["pose"].as_array().filter(|a| a.len() >= 3).map(|_| arr3(&j["pose"]));
        self.rooms = j["rooms"]
            .as_array()
            .map(|a| a.iter().map(|r| (r["id"].as_i64().unwrap_or(-1), r["name"].as_str().unwrap_or("").to_string())).collect())
            .unwrap_or_default();
        self.objs = j["objects"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|o| ObjInfo {
                        id: o["id"].as_u64().unwrap_or(0) as u32,
                        name: o["name"].as_str().unwrap_or("").to_string(),
                        pos: arr3(&o["pos"]),
                        extent: arr3(&o["extent"]),
                        state: o["state"].as_str().unwrap_or("seen").to_string(),
                        last_seen: f(&o["last_seen"]),
                        room: o["room"].as_i64(),
                        movable: o["movable"].as_bool().unwrap_or(true),
                        structural: o["structural"].as_bool().unwrap_or(false),
                        n_obs: o["n_obs"].as_u64().unwrap_or(0) as u32,
                        pos_sd: o["pos_sd"].as_array().filter(|a| a.len() == 3).map(|_| arr3(&o["pos_sd"])),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(())
    }

    /// 파일이 바뀌었으면 다시 읽고 true
    pub fn refresh(&mut self) -> Result<bool, String> {
        let m = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if m.is_some() && m != self.mtime {
            self.read()?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl Memory for ViewJson {
    fn objects(&self) -> &[ObjInfo] {
        &self.objs
    }
    fn now(&self) -> f64 {
        self.stamp
    }
    fn pose(&self) -> Option<[f64; 3]> {
        self.pose
    }
    fn room_name(&self, room: i64) -> Option<String> {
        self.rooms.iter().find(|r| r.0 == room).map(|r| r.1.clone()).filter(|s| !s.is_empty())
    }
    fn find_rooms(&self, s: &str) -> Vec<i64> {
        let t = s.trim().to_lowercase();
        let num = t.trim_start_matches('r').parse::<i64>().ok();
        self.rooms.iter().filter(|r| Some(r.0) == num || r.1.to_lowercase() == t).map(|r| r.0).collect()
    }
}
