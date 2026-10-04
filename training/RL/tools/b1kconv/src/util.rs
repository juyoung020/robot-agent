use serde_json::Value;
use std::collections::HashMap;

pub fn read_json(path: &str) -> Result<Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("{path}: {e}"))
}

pub fn get<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut c = v;
    for k in path {
        c = c.get(*k)?;
    }
    Some(c)
}

pub fn f64s(v: Option<&Value>) -> Option<Vec<f64>> {
    v?.as_array()?.iter().map(|x| x.as_f64()).collect()
}

pub fn arr3(v: Option<&Value>) -> Option<[f64; 3]> {
    let a = f64s(v)?;
    if a.len() == 3 {
        Some([a[0], a[1], a[2]])
    } else {
        None
    }
}

pub fn arr4(v: Option<&Value>) -> Option<[f64; 4]> {
    let a = f64s(v)?;
    if a.len() == 4 {
        Some([a[0], a[1], a[2], a[3]])
    } else {
        None
    }
}

pub fn f3(a: [f64; 3]) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}
pub fn f4(a: [f64; 4]) -> [f32; 4] {
    [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]
}

/// rotation matrix of a quaternion (x, y, z, w)
pub fn rot(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (x, y, z, w) = if n > 0.0 { (q[0] / n, q[1] / n, q[2] / n, q[3] / n) } else { (0.0, 0.0, 0.0, 1.0) };
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

pub fn yaw(q: [f64; 4]) -> f64 {
    let r = rot(q);
    r[1][0].atan2(r[0][0])
}

/// world bbox of a box (center offset `off` and half extents `half` in the object frame)
/// -> (world center, world AABB half extents, tilt from upright in rad)
pub fn world_box(pos: [f64; 3], q: [f64; 4], off: [f64; 3], half: [f64; 3]) -> ([f64; 3], [f64; 3], f64) {
    let r = rot(q);
    let mut c = pos;
    let mut h = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i] += r[i][j] * off[j];
            h[i] += r[i][j].abs() * half[j];
        }
    }
    (c, h, r[2][2].clamp(-1.0, 1.0).acos())
}

#[derive(Default)]
pub struct StrPool {
    pub bytes: Vec<u8>,
    map: HashMap<String, u32>,
}
impl StrPool {
    pub fn new() -> Self {
        let mut p = StrPool::default();
        p.add(""); // offset 0 = empty string
        p
    }
    pub fn add(&mut self, s: &str) -> u32 {
        if let Some(&o) = self.map.get(s) {
            return o;
        }
        let o = self.bytes.len() as u32;
        self.bytes.extend_from_slice(s.as_bytes());
        self.bytes.push(0);
        self.map.insert(s.to_string(), o);
        o
    }
}

pub fn cstr(pool: &[u8], off: u32) -> String {
    let s = &pool[off as usize..];
    let e = s.iter().position(|&b| b == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..e]).into_owned()
}

/// Python round(x, 2) (round half to even on the exact binary value) — matters for 1/8, 5/8.
pub fn py_round2(x: f64) -> f64 {
    let y = x * 100.0;
    let f = y.floor();
    let d = y - f;
    let r = if d > 0.5 { f + 1.0 } else if d < 0.5 { f } else if f % 2.0 == 0.0 { f } else { f + 1.0 };
    r / 100.0
}

/// write to `<path>.tmp.<pid>`, fsync, rename over `path` (readers never see a half file)
pub fn write_atomic(path: &str, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let tmp = format!("{path}.tmp.{}", std::process::id());
    let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{tmp}: {e}"))?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| format!("{tmp}: {e}"))?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {tmp} -> {path}: {e}"))
}
