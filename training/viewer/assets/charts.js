// 캔버스 곡선(차트 라이브러리 없음 — TRAIN_VIEWER.md 7절). DPR, 축 하나, 눈금, 기준선, 시점 커서, 띠(최소·최대 또는 ±σ), 십자선 값 읽기.
// 축은 하나다: 단위가 다른 두 값은 두 그림으로 나눈다(이중 축은 읽는 사람을 속인다).

export const COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export function color(i) { return css(COLORS[i % COLORS.length]); }
export function cssv(n) { return css(n); }

export function fmt(v, pct) {
  if (v == null || !isFinite(v)) return "—";
  if (pct) return (v * 100).toFixed(Math.abs(v) < 0.1 ? 2 : 1) + " %";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 1 : 2) + "G";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  return v.toPrecision(3);
}

function ticks(lo, hi, n) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

/** EMA over finite values (null 은 건너뛰고 그대로 null) */
export function ema(y, a) {
  if (!a || !y) return y;
  let n = 0; for (const v of y) if (v != null && isFinite(v)) n++;
  if (n < 60) return y;   // 드문 값(평가 줄 등)은 평활하지 않는다 — 몇 점을 섞으면 값이 틀려진다
  const out = new Array(y.length); let m = null;
  for (let i = 0; i < y.length; i++) {
    const v = y[i];
    if (v == null || !isFinite(v)) { out[i] = null; continue; }
    m = m == null ? v : a * m + (1 - a) * v; out[i] = m;
  }
  return out;
}

const tip = () => document.getElementById("tip");

export class Plot {
  constructor(canvas) {
    this.cv = canvas; this.spec = null; this.hover = null; this.sig = "";
    canvas.addEventListener("mousemove", e => { const r = canvas.getBoundingClientRect(); this.hover = { x: e.clientX - r.left, y: e.clientY - r.top, cx: e.clientX, cy: e.clientY }; this.draw(this.spec, true); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; tip().hidden = true; this.draw(this.spec, true); });
    canvas.addEventListener("click", () => { if (this.spec && this.spec.onPick && this.hx != null) this.spec.onPick(this.hx); });
  }
  /** spec: {series:[{name,x,y,lo,hi,color,dash,width}], refs:[{y,label}], xfmt, pct, cursor, xlabel, ymin, ymax, onPick} */
  draw(spec, force) {
    if (!spec) return;
    this.spec = spec;
    const cv = this.cv, dpr = Math.min(devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink2 = css("--ink2"), muted = css("--muted"), grid = css("--grid");
    // 범위
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of spec.series) {
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i]; if (x == null || !isFinite(x)) continue;
        const ys = [s.y[i], s.lo ? s.lo[i] : null, s.hi ? s.hi[i] : null];
        let any = false;
        for (const v of ys) if (v != null && isFinite(v)) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); any = true; }
        if (any) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
      }
    }
    for (const r of spec.refs || []) if (isFinite(r.y)) { y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y); }
    g.font = "11px system-ui, sans-serif";
    if (!isFinite(x0)) { g.fillStyle = muted; g.fillText(spec.empty || "값 없음", 10, 20); return; }
    if (spec.ymin != null) y0 = Math.min(y0, spec.ymin);
    if (spec.ymax != null) y1 = Math.max(y1, spec.ymax);
    if (y1 - y0 < 1e-12) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
    const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
    if (spec.pct) { y0 = Math.max(y0, Math.min(0, y0)); }
    if (x1 === x0) { x1 = x0 + 1; }
    const yt = ticks(y0, y1, Math.max(2, Math.floor(H / 40)));
    const lw = Math.max(...yt.map(v => g.measureText(fmt(v, spec.pct)).width)) + 8;
    const L = lw, R = W - 8, T = 6, B = H - 18;
    const X = x => L + (x - x0) / (x1 - x0) * (R - L), Y = y => B - (y - y0) / (y1 - y0) * (B - T);
    this.geo = { L, R, T, B, x0, x1, y0, y1, X };
    // 눈금(희미하게)
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted; g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of yt) { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke(); g.fillText(fmt(v, spec.pct), L - 4, y); }
    const xt = ticks(x0, x1, Math.max(2, Math.floor((R - L) / 90)));
    g.textAlign = "center"; g.textBaseline = "top";
    const xf = spec.xfmt || fmt;
    for (const v of xt) { const x = X(v); if (x < L - 1 || x > R + 1) continue; g.fillText(xf(v), x, B + 4); }
    // 기준선
    for (const r of spec.refs || []) {
      if (!isFinite(r.y)) continue;
      const y = Math.round(Y(r.y)) + 0.5;
      g.save(); g.setLineDash([5, 4]); g.strokeStyle = ink2; g.lineWidth = 1; g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke(); g.restore();
      g.fillStyle = ink2; g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText(r.label, R - 2, y - 2);
    }
    // 띠 → 선
    g.save(); g.beginPath(); g.rect(L, T, R - L, B - T); g.clip();
    for (const s of spec.series) {
      if (!s.lo || !s.hi) continue;
      g.fillStyle = s.color; g.globalAlpha = 0.16; g.beginPath();
      let open = false; const top = [];
      const flush = () => { if (!open) return; for (let k = top.length - 1; k >= 0; k--) g.lineTo(top[k][0], top[k][1]); g.closePath(); g.fill(); g.beginPath(); open = false; top.length = 0; };
      for (let i = 0; i < s.x.length; i++) {
        const lo = s.lo[i], hi = s.hi[i];
        if (lo == null || hi == null || !isFinite(lo) || !isFinite(hi)) { flush(); continue; }
        const x = X(s.x[i]);
        if (!open) { g.moveTo(x, Y(lo)); open = true; } else g.lineTo(x, Y(lo));
        top.push([x, Y(hi)]);
      }
      flush(); g.globalAlpha = 1;
    }
    for (const s of spec.series) {
      g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash ? [6, 4] : []); g.lineJoin = "round";
      // 키가 없는 줄(그 줄에 안 잰 값)은 건너뛰고 잇는다 — 학생 실행처럼 단계마다 다른 키를 쓰는 기록도 선이 된다
      g.beginPath(); let pen = false; const pts = [];
      for (let i = 0; i < s.x.length; i++) {
        const v = s.y[i];
        if (v == null || !isFinite(v) || s.x[i] == null) continue;
        const x = X(s.x[i]), y = Y(v);
        if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
        pts.push([x, y]);
      }
      g.stroke();
      if (pts.length < 60) { g.fillStyle = s.color; for (const [x, y] of pts) { g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); } }
    }
    g.restore(); g.setLineDash([]);
    // 커서
    if (spec.cursor != null && isFinite(spec.cursor) && spec.cursor < x1) {
      const x = Math.round(X(spec.cursor)) + 0.5;
      g.strokeStyle = css("--accent"); g.lineWidth = 1; g.beginPath(); g.moveTo(x, T); g.lineTo(x, B); g.stroke();
    }
    // 십자선 + 값 읽기
    const h = this.hover; this.hx = null;
    if (h && h.x >= L && h.x <= R) {
      const xv = x0 + (h.x - L) / (R - L) * (x1 - x0); this.hx = xv;
      g.strokeStyle = muted; g.lineWidth = 1; g.beginPath(); g.moveTo(Math.round(h.x) + 0.5, T); g.lineTo(Math.round(h.x) + 0.5, B); g.stroke();
      const rows = [];
      for (const s of spec.series) {
        let bi = -1, bd = Infinity;
        for (let i = 0; i < s.x.length; i++) { const v = s.y[i]; if (v == null || !isFinite(v)) continue; const d = Math.abs(s.x[i] - xv); if (d < bd) { bd = d; bi = i; } }
        if (bi < 0) continue;
        g.fillStyle = s.color; g.strokeStyle = css("--surface"); g.lineWidth = 2;
        g.beginPath(); g.arc(X(s.x[bi]), Y(s.y[bi]), 4, 0, 7); g.fill(); g.stroke();
        rows.push(`<div class="r"><i style="background:${s.color}"></i>${esc(s.name)} <b>${fmt(s.y[bi], spec.pct)}</b>${s.lo && s.lo[bi] != null ? ` <span class="muted">[${fmt(s.lo[bi], spec.pct)}, ${fmt(s.hi[bi], spec.pct)}]</span>` : ""}</div>`);
      }
      const t = tip();
      t.innerHTML = `<div class="muted">${esc(spec.xlabel || "x")} ${xf(xv)}</div>` + rows.join("");
      t.hidden = false;
      const tw = t.offsetWidth;
      t.style.left = (h.cx + 14 + tw > innerWidth ? h.cx - tw - 14 : h.cx + 14) + "px"; t.style.top = (h.cy + 12) + "px";
    }
  }
}

/** 막대 히스토그램(표본 판 기록용) */
export function hist(canvas, vals, bins, label, col) {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "11px system-ui, sans-serif";
  const v = vals.filter(x => x != null && isFinite(x));
  if (!v.length) { g.fillStyle = css("--muted"); g.fillText("표본 없음", 10, 20); return; }
  let lo = Math.min(...v), hi = Math.max(...v); if (hi === lo) hi = lo + 1;
  const c = new Array(bins).fill(0);
  for (const x of v) c[Math.min(bins - 1, Math.floor((x - lo) / (hi - lo) * bins))]++;
  const mx = Math.max(...c), L = 34, R = W - 8, T = 6, B = H - 18, bw = (R - L) / bins;
  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted"); g.textAlign = "right"; g.textBaseline = "middle";
  for (const t of ticks(0, mx, 3)) { const y = B - t / mx * (B - T); g.beginPath(); g.moveTo(L, y + 0.5); g.lineTo(R, y + 0.5); g.stroke(); g.fillText(fmt(t), L - 4, y); }
  g.fillStyle = col;
  for (let i = 0; i < bins; i++) {
    const h = c[i] / mx * (B - T); if (h <= 0) continue;
    const x = L + i * bw + 1, w = Math.max(1, bw - 2), r = Math.min(4, w / 2, h);
    g.beginPath(); g.moveTo(x, B); g.lineTo(x, B - h + r); g.quadraticCurveTo(x, B - h, x + r, B - h); g.lineTo(x + w - r, B - h); g.quadraticCurveTo(x + w, B - h, x + w, B - h + r); g.lineTo(x + w, B); g.fill();
  }
  g.fillStyle = css("--muted"); g.textAlign = "center"; g.textBaseline = "top";
  for (const t of ticks(lo, hi, 5)) { const x = L + (t - lo) / (hi - lo) * (R - L); if (x >= L && x <= R) g.fillText(fmt(t), x, B + 4); }
  g.textAlign = "right"; g.fillText(`${label} (n ${v.length})`, R, T);
}

export function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
