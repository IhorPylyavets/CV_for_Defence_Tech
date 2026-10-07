import { SplatRenderer } from "./gl_render.js";
import { STRIDE, DEFAULT_CUTOFF } from "./engine.js";
import { SAMPLES } from "./images.js";

const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js", { type: "module" });
const renderer = new SplatRenderer($("recon-gl"));

// ------------------------------------------------------------------------------------------
// Options (sent to the worker)
// ------------------------------------------------------------------------------------------
const LR_DEFAULT = [-2.3, -1.7, -1.7, -1.7, -1.3]; // log10 learning rates: position, scale, rotation, colour, opacity
const opts = {
  n: 500, init: "random", initScale: 0.03, initOpacity: 0.5, randomRotation: false,
  blend: "additive", background: [0, 0, 0],
  loss: "l1", l2Weight: 0.5, lrScale: 1,
  lr: LR_DEFAULT.map((v) => 10 ** v), learn: [true, true, true, true, true],
  decay: false, decayTo: 0.1, decayIters: 3000, maxIters: 0,
  densify: false, densifyEvery: 100, growFrac: 0.15, maxN: 3000, splitScale: 0.01, pruneThreshold: 0.005, densifyUntil: 15000,
};
const display = { view: "render", showOutlines: true, showCenters: false, outlineK: 1.75, errGain: 4 };

const state = {
  params: new Float32Array(0), n: 0, W: 0, H: 0, image: null, target: null,
  playing: false, history: [], hover: -1, dirty: true, source: null, webcam: null,
};
const sendOpts = () => worker.postMessage({ cmd: "opts", opts });

// ------------------------------------------------------------------------------------------
// Target image handling
// ------------------------------------------------------------------------------------------
const srcCanvas = document.createElement("canvas");  // full-resolution source
let resMax = 192;

function setSourceFromSample(key) {
  const s = SAMPLES[key];
  srcCanvas.width = 768; srcCanvas.height = Math.round(768 * s.aspect);
  s.draw(srcCanvas.getContext("2d"), srcCanvas.width, srcCanvas.height);
  state.source = key;
  pushTarget(true);
}

function setSourceFromImage(img) {
  const k = Math.min(1, 1024 / Math.max(img.width, img.height));
  srcCanvas.width = Math.round(img.width * k); srcCanvas.height = Math.round(img.height * k);
  srcCanvas.getContext("2d").drawImage(img, 0, 0, srcCanvas.width, srcCanvas.height);
  pushTarget(true);
}

// downsample the source to training resolution and send it to the worker
function pushTarget(reset) {
  const a = srcCanvas.height / srcCanvas.width;
  const W = a <= 1 ? resMax : Math.round(resMax / a), H = a <= 1 ? Math.round(resMax * a) : resMax;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(srcCanvas, 0, 0, W, H);
  const d = ctx.getImageData(0, 0, W, H).data;
  const target = new Float32Array(3 * W * H);
  for (let p = 0; p < W * H; p++) { target[3 * p] = d[4 * p] / 255; target[3 * p + 1] = d[4 * p + 1] / 255; target[3 * p + 2] = d[4 * p + 2] / 255; }
  state.target = target;
  const tc = $("target");
  tc.width = srcCanvas.width; tc.height = srcCanvas.height;
  tc.getContext("2d").drawImage(srcCanvas, 0, 0);
  $("target-res").textContent = `· training at ${W}×${H}`;
  worker.postMessage({ cmd: "target", target: target.slice(), W, H, reset });
  layout();
}

const sel = $("source");
for (const [k, s] of Object.entries(SAMPLES)) sel.add(new Option(s.name, k));
sel.add(new Option("Uploaded image", "upload"));
sel.options[sel.options.length - 1].disabled = true;
sel.addEventListener("change", () => { stopWebcam(); if (SAMPLES[sel.value]) setSourceFromSample(sel.value); });

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = () => { stopWebcam(); sel.value = "upload"; setSourceFromImage(img); URL.revokeObjectURL(img.src); };
  img.src = URL.createObjectURL(file);
}
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const ts = $("target-stage");
ts.addEventListener("dragover", (e) => { e.preventDefault(); ts.classList.add("drag"); });
ts.addEventListener("dragleave", () => ts.classList.remove("drag"));
ts.addEventListener("drop", (e) => { e.preventDefault(); ts.classList.remove("drag"); loadFile(e.dataTransfer.files[0]); });
window.addEventListener("paste", (e) => { const f = [...e.clipboardData.files][0]; if (f) loadFile(f); });

// Webcam: the target changes every frame and the Gaussians keep tracking it
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const v = $("video");
    v.srcObject = stream; await v.play();
    srcCanvas.width = v.videoWidth; srcCanvas.height = v.videoHeight;
    let first = true;
    state.webcam = setInterval(() => {
      const ctx = srcCanvas.getContext("2d");
      ctx.save(); ctx.scale(-1, 1); ctx.drawImage(v, -srcCanvas.width, 0); ctx.restore();
      pushTarget(first); first = false;
    }, 100);
    state.webcamStream = stream;
    $("webcam").textContent = "Stop webcam";
    if (!state.playing) togglePlay();
  } catch (err) {
    log(`webcam unavailable: ${err.message}`);
  }
}
function stopWebcam() {
  if (!state.webcam) return;
  clearInterval(state.webcam); state.webcam = null;
  state.webcamStream.getTracks().forEach((t) => t.stop());
  $("webcam").textContent = "Webcam (live)";
}
$("webcam").addEventListener("click", () => (state.webcam ? stopWebcam() : startWebcam()));

// ------------------------------------------------------------------------------------------
// Controls
// ------------------------------------------------------------------------------------------
function slider(id, fmt, apply, live = true) {
  const el = $(id), out = el.nextElementSibling;
  const upd = (send) => { const v = +el.value; out.textContent = fmt(v); apply(v); if (send && live) sendOpts(); };
  el.addEventListener("input", () => upd(true));
  upd(false);
  return el;
}
function seg(id, apply) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    apply(b.dataset.v);
  });
}
function check(id, apply, live = true) {
  const el = $(id);
  el.addEventListener("change", () => { apply(el.checked); if (live) sendOpts(); });
  apply(el.checked);
}
const pct = (v) => v.toFixed(1) + "%";
const sci = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1));

slider("res", (v) => `${v}px`, (v) => { resMax = v; });
$("res").addEventListener("change", () => pushTarget(false));
slider("n", (v) => v.toFixed(0), (v) => (opts.n = v), false);
slider("initScale", pct, (v) => (opts.initScale = v / 100), false);
slider("initOpacity", (v) => v.toFixed(2), (v) => (opts.initOpacity = v), false);
$("init").addEventListener("change", (e) => (opts.init = e.target.value));
check("randomRotation", (v) => (opts.randomRotation = v), false);
slider("l2Weight", (v) => v.toFixed(2), (v) => (opts.l2Weight = v));
slider("lrScale", (v) => (10 ** v).toFixed(2) + "×", (v) => (opts.lrScale = 10 ** v));
for (let k = 0; k < 5; k++) {
  $(`lr${k}`).value = LR_DEFAULT[k];
  slider(`lr${k}`, (v) => sci(10 ** v), (v) => (opts.lr[k] = 10 ** v));
  check(`learn${k}`, (v) => (opts.learn[k] = v));
}
check("decay", (v) => (opts.decay = v));
slider("decayIters", (v) => v.toFixed(0), (v) => (opts.decayIters = v));
slider("maxIters", (v) => (v ? v.toFixed(0) : "∞"), (v) => (opts.maxIters = v));
check("densify", (v) => (opts.densify = v));
slider("densifyEvery", (v) => v.toFixed(0), (v) => (opts.densifyEvery = v));
slider("growFrac", (v) => Math.round(v * 100) + "%", (v) => (opts.growFrac = v));
slider("maxN", (v) => v.toFixed(0), (v) => (opts.maxN = v));
slider("splitScale", pct, (v) => (opts.splitScale = v / 100));
slider("pruneThreshold", (v) => v.toFixed(3), (v) => (opts.pruneThreshold = v));
slider("densifyUntil", (v) => v.toFixed(0), (v) => (opts.densifyUntil = v));
slider("outlineK", (v) => v.toFixed(2) + "σ", (v) => { display.outlineK = v; state.dirty = true; });
slider("errGain", (v) => v.toFixed(1) + "×", (v) => { display.errGain = v; state.dirty = true; });
check("showOutlines", (v) => { display.showOutlines = v; state.dirty = true; }, false);
check("showCenters", (v) => { display.showCenters = v; state.dirty = true; }, false);

const BLEND_HELP = {
  additive: "Reference model: every Gaussian adds its colour, I(p) = Σ cᵢ Gᵢ(p). There's no occlusion or opacity, and colours may go negative.",
  alpha: "3DGS model: Gaussians are composited front to back (index = depth), I = Σ cᵢ αᵢ Tᵢ + T·bg with αᵢ = oᵢ Gᵢ(p). Each Gaussian learns an opacity oᵢ.",
};
function setBlend(v) {
  opts.blend = v;
  $("blend-help").textContent = BLEND_HELP[v];
  $("bg-field").classList.toggle("hidden", v !== "alpha");
  $("op-lr-row").classList.toggle("hidden", v !== "alpha");
  $("initOpacity").closest(".slider").classList.toggle("hidden", v !== "alpha");
  sendOpts();
  state.dirty = true;
}
seg("blend", setBlend);
seg("bg", (v) => { opts.background = [+v, +v, +v]; sendOpts(); state.dirty = true; });
seg("loss", (v) => { opts.loss = v; $("l2w-row").classList.toggle("hidden", v !== "mix"); sendOpts(); });
$("l2w-row").classList.add("hidden");
seg("view", (v) => {
  display.view = v;
  worker.postMessage({ cmd: "wantImage", value: true });
  state.dirty = true;
});

function togglePlay() {
  state.playing = !state.playing;
  worker.postMessage({ cmd: state.playing ? "play" : "pause" });
  $("play").textContent = state.playing ? "❚❚ Pause" : "▶ Train";
}
$("play").addEventListener("click", togglePlay);
$("step").addEventListener("click", () => { if (state.playing) togglePlay(); worker.postMessage({ cmd: "step", count: 1 }); });
$("reset").addEventListener("click", () => { worker.postMessage({ cmd: "reset", opts }); $("log").textContent = ""; log("reset"); });
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.key === "s" || e.key === "S") $("step").click();
  else if (e.key === "r" || e.key === "R") $("reset").click();
});

function log(msg) {
  const el = $("log");
  el.textContent = msg + "\n" + el.textContent.slice(0, 4000);
}

// ------------------------------------------------------------------------------------------
// Worker messages
// ------------------------------------------------------------------------------------------
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type !== "state") return;
  Object.assign(state, { params: m.params, n: m.n, W: m.W, H: m.H, history: m.history });
  if (m.image) state.image = m.image;
  if (!m.playing && state.playing && opts.maxIters && m.iter >= opts.maxIters) togglePlay();
  $("s-iter").textContent = m.iter;
  $("s-n").textContent = m.n;
  $("s-psnr").textContent = m.psnr ? m.psnr.toFixed(2) + " dB" : "–";
  $("s-loss").textContent = m.loss ? m.loss.toFixed(4) : "–";
  $("s-itps").textContent = m.itps ? m.itps.toFixed(0) : "–";
  $("s-lr").textContent = m.lr.toFixed(3);
  for (const ev of m.events || []) log(`it ${ev.iter}: split ${ev.split}, cloned ${ev.cloned}, pruned ${ev.pruned} → N=${m.n}`);
  state.dirty = true;
};

// ------------------------------------------------------------------------------------------
// Drawing
// ------------------------------------------------------------------------------------------
let rect = { x: 0, y: 0, w: 1, h: 1 };
function layout() {
  const a = srcCanvas.height / srcCanvas.width || 1;
  for (const stageId of ["target-stage", "recon-stage"]) {
    const st = $(stageId), r = st.getBoundingClientRect();
    let w = r.width, h = w * a;
    if (h > r.height) { h = r.height; w = h / a; }
    const x = (r.width - w) / 2, y = (r.height - h) / 2;
    for (const c of st.querySelectorAll("canvas")) Object.assign(c.style, { left: x + "px", top: y + "px", width: w + "px", height: h + "px" });
    if (stageId === "recon-stage") rect = { x, y, w, h };
  }
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const c of [$("recon-gl"), $("overlay")]) { c.width = Math.round(rect.w * dpr); c.height = Math.round(rect.h * dpr); }
  state.dirty = true;
}
window.addEventListener("resize", layout);

function turbo(t) {
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b];
}

function drawExact() {
  const c = $("recon-2d"), { W, H, image, target } = state;
  if (!image || !target || image.length !== W * H * 4) return;
  c.width = W; c.height = H;
  const out = new ImageData(W, H);
  if (display.view === "exact") out.data.set(image);
  else {
    for (let p = 0; p < W * H; p++) {
      const e = (Math.abs(image[4 * p] / 255 - target[3 * p]) + Math.abs(image[4 * p + 1] / 255 - target[3 * p + 1]) + Math.abs(image[4 * p + 2] / 255 - target[3 * p + 2])) / 3;
      const [r, g, b] = turbo(Math.min(1, e * display.errGain));
      out.data[4 * p] = r * 255; out.data[4 * p + 1] = g * 255; out.data[4 * p + 2] = b * 255; out.data[4 * p + 3] = 255;
    }
  }
  c.getContext("2d").putImageData(out, 0, 0);
}

function drawOverlay() {
  const c = $("overlay"), ctx = c.getContext("2d"), P = state.params, n = state.n;
  ctx.clearRect(0, 0, c.width, c.height);
  const s = c.width; // normalised units are in widths
  const k = display.outlineK;
  if (display.showOutlines && n) {
    ctx.strokeStyle = "rgba(255,60,60,.55)";
    ctx.lineWidth = Math.max(0.6, c.width / 900);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const o = i * STRIDE;
      const x = P[o] * s, y = P[o + 1] * s, rx = Math.exp(P[o + 2]) * s * k, ry = Math.exp(P[o + 3]) * s * k;
      if (rx < 0.3 && ry < 0.3) continue;
      ctx.moveTo(x + rx * Math.cos(P[o + 4]), y + rx * Math.sin(P[o + 4]));
      ctx.ellipse(x, y, rx, ry, P[o + 4], 0, Math.PI * 2);
    }
    ctx.stroke();
  }
  if (display.showCenters && n) {
    ctx.fillStyle = "#ff3c3c";
    const r = Math.max(1, c.width / 500);
    for (let i = 0; i < n; i++) ctx.fillRect(P[i * STRIDE] * s - r / 2, P[i * STRIDE + 1] * s - r / 2, r, r);
  }
  const h = state.hover;
  if (h >= 0 && h < n) {
    const o = h * STRIDE;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(1.5, c.width / 400);
    for (const kk of [1, 2]) {
      ctx.beginPath();
      ctx.ellipse(P[o] * s, P[o + 1] * s, Math.exp(P[o + 2]) * s * kk, Math.exp(P[o + 3]) * s * kk, P[o + 4], 0, Math.PI * 2);
      ctx.setLineDash(kk === 1 ? [] : [4, 4]); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
}

function drawChart() {
  const c = $("chart"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  c.width = r.width * dpr; c.height = r.height * dpr;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const H = state.history, W = r.width, Ht = r.height;
  const L = 40, R = 46, T = 22, B = 20, w = W - L - R, h = Ht - T - B;
  ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#98a1b0"; ctx.strokeStyle = "#2a303b";
  if (H.length < 2) { ctx.fillText("press ▶ Train", L, T + h / 2); return; }
  const maxIt = H[H.length - 1][0] || 1;
  let pMin = Infinity, pMax = -Infinity, nMax = 0;
  for (const [, p, n] of H) { pMin = Math.min(pMin, p); pMax = Math.max(pMax, p); nMax = Math.max(nMax, n); }
  pMin = Math.floor(pMin / 2) * 2; pMax = Math.ceil(pMax / 2) * 2 + 1;
  const X = (it) => L + (it / maxIt) * w, Yp = (p) => T + (1 - (p - pMin) / (pMax - pMin)) * h, Yn = (n) => T + (1 - n / (nMax * 1.1)) * h;
  for (let i = 0; i <= 4; i++) {
    const p = pMin + ((pMax - pMin) * i) / 4, y = Yp(p);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + w, y); ctx.stroke();
    ctx.fillStyle = "#7aa8ff"; ctx.fillText(p.toFixed(1), 4, y + 3);
    ctx.fillStyle = "#e0a458"; ctx.fillText(Math.round((nMax * 1.1 * i) / 4).toString(), L + w + 6, Yn((nMax * 1.1 * i) / 4) + 3);
  }
  ctx.fillStyle = "#98a1b0"; ctx.fillText("0", L, Ht - 6);
  const lbl = `${maxIt} it`; ctx.fillText(lbl, L + w - ctx.measureText(lbl).width, Ht - 6);
  const line = (yf, col, idx) => {
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.beginPath();
    H.forEach((d, i) => (i ? ctx.lineTo(X(d[0]), yf(d[idx])) : ctx.moveTo(X(d[0]), yf(d[idx]))));
    ctx.stroke();
  };
  line(Yn, "#e0a458", 2);
  line(Yp, "#7aa8ff", 1);
}

function frame() {
  if (state.dirty) {
    state.dirty = false;
    const aspect = srcCanvas.height / srcCanvas.width || 1;
    const exact = display.view === "exact" || display.view === "error";
    $("recon-2d").classList.toggle("hidden", !exact);
    $("recon-gl").classList.toggle("hidden", exact);
    if (exact) drawExact();
    else {
      renderer.setParams(state.params, state.n, opts.blend === "alpha" || display.view === "splats");
      renderer.render(display.view === "splats" ? "splats" : opts.blend, aspect, display.view === "splats" ? [0.07, 0.08, 0.1] : opts.background);
    }
    drawOverlay();
    drawChart();
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------------------------------
// Hover inspection: which Gaussian contributes most to this pixel?
// ------------------------------------------------------------------------------------------
$("recon-stage").addEventListener("mousemove", (e) => {
  const r = $("overlay").getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = ((e.clientY - r.top) / r.width);
  if (x < 0 || x > 1 || y < 0 || y > r.height / r.width) { hideTip(); return; }
  const P = state.params, n = state.n, alpha = opts.blend === "alpha";
  let best = -1, bestW = 0, T = 1;
  for (let i = 0; i < n; i++) {
    const o = i * STRIDE, dx = x - P[o], dy = y - P[o + 1];
    const c = Math.cos(P[o + 4]), s = Math.sin(P[o + 4]);
    const u1 = (c * dx + s * dy) / Math.exp(P[o + 2]), u2 = (-s * dx + c * dy) / Math.exp(P[o + 3]);
    const q = u1 * u1 + u2 * u2;
    if (q > DEFAULT_CUTOFF ** 2) continue;
    const G = Math.exp(-0.5 * q);
    let w;
    if (alpha) {
      const a = Math.min(0.99, G / (1 + Math.exp(-P[o + 8])));
      if (a < 1 / 255) continue;
      w = a * T; T *= 1 - a;
    } else w = G * Math.max(Math.abs(P[o + 5]), Math.abs(P[o + 6]), Math.abs(P[o + 7]));
    if (w > bestW) { bestW = w; best = i; }
  }
  state.hover = best; state.dirty = true;
  const tip = $("tooltip");
  if (best < 0) { tip.classList.add("hidden"); return; }
  const o = best * STRIDE, W = state.W;
  const col = [P[o + 5], P[o + 6], P[o + 7]];
  const css = `rgb(${col.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255)).join(",")})`;
  tip.innerHTML =
    `<b>Gaussian #${best}</b>${alpha ? `  (depth order ${best})` : ""}\n` +
    `μ  = (${(P[o] * W).toFixed(1)}, ${(P[o + 1] * W).toFixed(1)}) px\n` +
    `σ  = (${(Math.exp(P[o + 2]) * W).toFixed(2)}, ${(Math.exp(P[o + 3]) * W).toFixed(2)}) px\n` +
    `θ  = ${((P[o + 4] * 180) / Math.PI % 360).toFixed(1)}°\n` +
    `c  = <span class="sw" style="background:${css}"></span>(${col.map((v) => v.toFixed(2)).join(", ")})` +
    (alpha ? `\no  = ${(1 / (1 + Math.exp(-P[o + 8]))).toFixed(3)}` : "") +
    `\n${alpha ? "weight αT" : "c·G"} here = ${bestW.toFixed(3)}`;
  tip.classList.remove("hidden");
  const sr = $("recon-stage").getBoundingClientRect();
  let tx = e.clientX - sr.left + 14, ty = e.clientY - sr.top + 14;
  if (tx + 230 > sr.width) tx -= 250;
  if (ty + 140 > sr.height) ty -= 150;
  tip.style.left = tx + "px"; tip.style.top = ty + "px";
});
function hideTip() { $("tooltip").classList.add("hidden"); state.hover = -1; state.dirty = true; }
$("recon-stage").addEventListener("mouseleave", hideTip);

// ------------------------------------------------------------------------------------------
// Export
// ------------------------------------------------------------------------------------------
function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("exportJson").addEventListener("click", () => {
  const P = state.params, g = [];
  for (let i = 0; i < state.n; i++) {
    const o = i * STRIDE;
    g.push({ center: [P[o], P[o + 1]], sigma: [Math.exp(P[o + 2]), Math.exp(P[o + 3])], theta: P[o + 4],
      color: [P[o + 5], P[o + 6], P[o + 7]], opacity: 1 / (1 + Math.exp(-P[o + 8])) });
  }
  const json = { units: "image widths (y down)", aspect: state.H / state.W, blend: opts.blend, background: opts.background, gaussians: g };
  download("gaussians.json", new Blob([JSON.stringify(json)], { type: "application/json" }));
});
$("exportPng").addEventListener("click", () => {
  const c = display.view === "exact" || display.view === "error" ? $("recon-2d") : $("recon-gl");
  c.toBlob((b) => download("reconstruction.png", b));
});

// ------------------------------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------------------------------
setBlend("additive");
worker.postMessage({ cmd: "opts", opts });
sel.value = "sunset";
setSourceFromSample("sunset");
requestAnimationFrame(frame);
window.__app = { state, opts, display, worker, togglePlay };
