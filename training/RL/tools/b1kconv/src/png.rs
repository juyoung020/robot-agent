//! Minimal PNG reader for the BEHAVIOR layout maps: 8-bit grayscale, non-interlaced only.
//! Anything else is an error (the layout PNGs are all this kind; a different kind means the
//! dataset changed and the converter must be looked at again).

pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>, // row-major, row 0 = first row in the file
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

pub fn read_gray(path: &str) -> Result<Gray, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(format!("{path}: not a PNG"));
    }
    let mut p = 8;
    let (mut w, mut h) = (0usize, 0usize);
    let mut idat = Vec::new();
    while p + 8 <= data.len() {
        let len = be32(&data[p..]) as usize;
        let ty = &data[p + 4..p + 8];
        let body = data.get(p + 8..p + 8 + len).ok_or(format!("{path}: truncated chunk"))?;
        match ty {
            b"IHDR" => {
                w = be32(&body[0..]) as usize;
                h = be32(&body[4..]) as usize;
                let (depth, color, interlace) = (body[8], body[9], body[12]);
                if depth != 8 || color != 0 || interlace != 0 {
                    return Err(format!(
                        "{path}: unsupported PNG (depth {depth}, color type {color}, interlace {interlace}); need 8-bit gray"
                    ));
                }
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        p += 12 + len;
    }
    if w == 0 || h == 0 {
        return Err(format!("{path}: no IHDR"));
    }
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).map_err(|e| format!("{path}: inflate {e:?}"))?;
    if raw.len() != h * (w + 1) {
        return Err(format!("{path}: data size {} != {}", raw.len(), h * (w + 1)));
    }
    let mut px = vec![0u8; w * h];
    let mut prev = vec![0u8; w];
    for r in 0..h {
        let f = raw[r * (w + 1)];
        let src = &raw[r * (w + 1) + 1..(r + 1) * (w + 1)];
        let mut cur = vec![0u8; w];
        for c in 0..w {
            let a = if c > 0 { cur[c - 1] as i16 } else { 0 };
            let b = prev[c] as i16;
            let cc = if c > 0 { prev[c - 1] as i16 } else { 0 };
            let pred = match f {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let pp = a + b - cc;
                    let (pa, pb, pc) = ((pp - a).abs(), (pp - b).abs(), (pp - cc).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        cc
                    }
                }
                _ => return Err(format!("{path}: bad filter {f}")),
            };
            cur[c] = src[c].wrapping_add(pred as u8);
        }
        px[r * w..(r + 1) * w].copy_from_slice(&cur);
        prev = cur;
    }
    Ok(Gray { w, h, px })
}
