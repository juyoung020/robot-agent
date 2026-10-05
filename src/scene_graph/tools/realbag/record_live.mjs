// sgview 화면 녹화: 머리 없는 Chrome(CDP)으로 sgview 페이지를 일정 간격으로 찍으면서, 실제 bag 재생(realbag_run --live)을 띄운다.
//   node src/scene_graph/tools/realbag/record_live.mjs <out_dir> <interval_s> <warm_s> <tail_s> -- <재생 명령 …(sgs_play 또는 realbag_run --live)>
// 쓰는 것: <out_dir>/f_000000.png …, <out_dir>/frames.json [{k, wall}] 와 재생 시작 벽시각(start_wall). 영상 합치기는 make_video.sh.
// 환경: SGV_URL(기본 http://127.0.0.1:8080/), SGV_JS(페이지 연 뒤 실행할 JS), CHROME(기본 google-chrome), CDP_PORT(기본 9351), W·H(기본 1520×900)
import { spawn } from "child_process";
import { mkdirSync, writeFileSync } from "fs";

const args = process.argv.slice(2);
const sep = args.indexOf("--");
const [outDir, interval = "0.4", warm = "6", tail = "4"] = args.slice(0, sep);
const cmd = args.slice(sep + 1);
const url = process.env.SGV_URL || "http://127.0.0.1:8080/";
const port = +(process.env.CDP_PORT || 9351);
const W = +(process.env.W || 1520), H = +(process.env.H || 900);
mkdirSync(outDir, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(process.env.CHROME || "google-chrome", [
  "--headless=new", `--remote-debugging-port=${port}`, `--window-size=${W},${H}`, "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
  "--no-first-run", "--no-default-browser-check", `--user-data-dir=${outDir}/.chrome`, "about:blank",
], { stdio: "ignore" });
let tabs;
for (let i = 0; i < 100; ++i) {
  try { tabs = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json(); break; } catch { await sleep(200); }
}
const ws = new WebSocket(tabs.webSocketDebuggerUrl);
let id = 0;
const pend = {};
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend[m.id]) { pend[m.id](m); delete pend[m.id]; } };
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
await sleep(3000);
if (process.env.SGV_JS) await send("Runtime.evaluate", { expression: process.env.SGV_JS });   // 화면 설정(예: 이름표 끄기)

const t0 = Date.now();
const frames = [];
let child = null, startWall = null, endWall = null;
const iv = +interval * 1000;
for (let k = 0; ; ++k) {
  const due = t0 + k * iv;
  const now = Date.now();
  if (due > now) await sleep(due - now);
  const wall = (Date.now() - t0) / 1000;
  if (!child && wall >= +warm) {
    child = spawn(cmd[0], cmd.slice(1), { stdio: ["ignore", "inherit", "inherit"] });
    startWall = (Date.now() - t0) / 1000;
    child.on("exit", () => { endWall = (Date.now() - t0) / 1000; });
  }
  const s = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${outDir}/f_${String(k).padStart(6, "0")}.png`, Buffer.from(s.result.data, "base64"));
  frames.push({ k, wall });
  if (endWall !== null && wall > endWall + +tail) break;
}
writeFileSync(`${outDir}/frames.json`, JSON.stringify({ interval: +interval, start_wall: startWall, end_wall: endWall, frames }));
console.log(`recorded ${frames.length} frames, replay ${startWall}..${endWall} s`);
ws.close();
chrome.kill();
process.exit(0);
