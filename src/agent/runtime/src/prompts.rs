//! 프롬프트 읽기: 공통 `src/agent/prompts/common.md` + 스킬 `system.md`·`task.md` + 스킬이 쓰는 도구마다 설명 파일(기본 `tool.md`).
//! 파일 첫 줄 `<!-- version: … -->` 를 판 이름으로 모으고, 읽은 바이트 전체의 FNV-64 지문을 기록마다 남긴다.

use std::path::{Path, PathBuf};

pub struct Prompts {
    pub system: String,
    pub task: String,
    /// 도구마다 LLM 에게 보일 설명(스킬 설정 `tools` 순서)
    pub tool_descs: Vec<String>,
    pub version: String,
    pub sha: String,
    pub files: Vec<String>,
}

pub fn fnv64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn strip_header(s: &str) -> (String, Option<String>) {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<!--") {
        if let Some(end) = rest.find("-->") {
            let head = &rest[..end];
            let ver = head.split("version:").nth(1).map(|v| v.trim().to_string());
            return (rest[end + 3..].trim().to_string(), ver);
        }
    }
    (t.trim().to_string(), None)
}

impl Prompts {
    /// skill_dir = src/agent/skills/<skill>, common = src/agent/prompts/common.md, tool_files = 스킬 폴더 안 도구 설명 파일 이름들
    pub fn load(skill_dir: &Path, common: &Path, tool_files: &[String]) -> Result<Prompts, String> {
        let rd = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
        let mut names: Vec<PathBuf> = vec![common.to_path_buf(), skill_dir.join("system.md"), skill_dir.join("task.md")];
        names.extend(tool_files.iter().map(|f| skill_dir.join(f)));
        let raw: Vec<String> = names.iter().map(|p| rd(p)).collect::<Result<_, _>>()?;
        let mut all = vec![];
        for r in &raw {
            all.extend_from_slice(r.as_bytes());
        }
        let parts: Vec<(String, Option<String>)> = raw.iter().map(|r| strip_header(r)).collect();
        let version = parts.iter().filter_map(|p| p.1.clone()).collect::<Vec<_>>().join("+");
        Ok(Prompts {
            system: format!("{}\n\n{}", parts[0].0, parts[1].0),
            task: parts[2].0.clone(),
            tool_descs: parts[3..].iter().map(|p| p.0.replace('\n', " ")).collect(),
            version,
            sha: format!("{:016x}", fnv64(&all)),
            files: names.iter().map(|p| p.display().to_string()).collect(),
        })
    }
}
