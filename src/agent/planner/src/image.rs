//! 카메라 영상 → LLM 입력(JPEG, data URL). 단계 경계에서만 쓰므로 CPU 로 충분하다
//! (720² → 448² 면적 평균 축소 + JPEG 약 2~4 ms, 과제당 수십 번 — 실측은 docs/에이전트_설계.md).

#[derive(Clone, Debug)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Rgb {
    /// 긴 변이 max 이하가 되도록 면적 평균으로 줄인다(정수 배율이 아니어도 됨).
    pub fn downscale(&self, max: usize) -> Rgb {
        let long = self.w.max(self.h);
        if long <= max || max == 0 {
            return self.clone();
        }
        let s = max as f64 / long as f64;
        let (nw, nh) = (((self.w as f64) * s).round().max(1.0) as usize, ((self.h as f64) * s).round().max(1.0) as usize);
        let mut out = vec![0u8; nw * nh * 3];
        for y in 0..nh {
            let y0 = y * self.h / nh;
            let y1 = ((y + 1) * self.h / nh).max(y0 + 1);
            for x in 0..nw {
                let x0 = x * self.w / nw;
                let x1 = ((x + 1) * self.w / nw).max(x0 + 1);
                let mut acc = [0u32; 3];
                for yy in y0..y1 {
                    let row = &self.px[(yy * self.w + x0) * 3..(yy * self.w + x1) * 3];
                    for p in row.chunks_exact(3) {
                        acc[0] += p[0] as u32;
                        acc[1] += p[1] as u32;
                        acc[2] += p[2] as u32;
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as u32;
                let o = (y * nw + x) * 3;
                out[o] = (acc[0] / n) as u8;
                out[o + 1] = (acc[1] / n) as u8;
                out[o + 2] = (acc[2] / n) as u8;
            }
        }
        Rgb { w: nw, h: nh, px: out }
    }

    pub fn jpeg(&self, quality: u8) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h / 4);
        let enc = jpeg_encoder::Encoder::new(&mut out, quality);
        if enc.encode(&self.px, self.w as u16, self.h as u16, jpeg_encoder::ColorType::Rgb).is_err() {
            return Vec::new();
        }
        out
    }

    /// 시험용 단색/무늬 영상.
    pub fn pattern(w: usize, h: usize, seed: u8) -> Rgb {
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                px[o] = (x as u8).wrapping_add(seed);
                px[o + 1] = (y as u8).wrapping_mul(2);
                px[o + 2] = seed;
            }
        }
        Rgb { w, h, px }
    }
}

pub fn data_url_jpeg(jpeg: &[u8]) -> String {
    format!("data:image/jpeg;base64,{}", crate::codec::b64_encode(jpeg))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn downscale_and_jpeg() {
        let img = Rgb::pattern(720, 720, 3);
        let s = img.downscale(448);
        assert_eq!((s.w, s.h), (448, 448));
        let j = s.jpeg(85);
        assert!(j.len() > 100 && j[0] == 0xFF && j[1] == 0xD8);
    }
}
