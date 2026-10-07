// COLMAP Sparse Explorer: run pycolmap SfM through server.py and inspect the result in WebGL2.
// Points, camera frustums and image planes are drawn here; the server only reads and writes COLMAP models.

const $ = (id) => document.getElementById(id);
const glCanvas = $("gl");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const gl = glCanvas.getContext("webgl2", { antialias: true });
if (!gl) {
  document.body.innerHTML = "<p style='padding:20px'>WebGL2 is required.</p>";
  throw new Error("no webgl2");
}

// ------------------------------------------------------------------------------------------
// Small vector / matrix helpers (column-major 4x4, like WebGL)
// ------------------------------------------------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => mul(a, 1 / (len(a) || 1));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const DEG = Math.PI / 180;

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function viewMatrix(eye, r, u, f) {
  return new Float32Array([r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0,
    -dot(r, eye), -dot(u, eye), dot(f, eye), 1]);
}
function matMul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function quantile(arr, q) {
  if (!arr.length) return 0;
  const s = Float64Array.from(arr).sort();
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
}

// Perceptually ordered sequential ramp (matplotlib "inferno"), started above black so that
// low values stay visible on the dark background.
const INFERNO = ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"];
const RAMP_LO = 0.18;
const hex3 = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
document.querySelector(".legend-ramp").style.background =
  `linear-gradient(90deg, ${INFERNO.map((c, i) => [c, i / 10]).filter(([, t]) => t >= RAMP_LO - 0.1)
    .map(([c, t]) => `${c} ${Math.max(0, (t - RAMP_LO) / (1 - RAMP_LO)) * 100}%`).join(",")})`;

// ------------------------------------------------------------------------------------------
// Shaders
// ------------------------------------------------------------------------------------------
const POINT_VS = `#version 300 es
in vec3 aPos; in vec3 aCol; in vec2 aAttr; in float aFlag;
uniform mat4 uVP;
uniform float uSize, uFocal;
uniform bool uAtten, uDim;
uniform int uMode;
uniform float uMaxErr, uMinTrack, uErrRef, uTrackRef;
uniform vec2 uYRange;
uniform vec3 uRamp[11];
uniform float uRampLo;
out vec3 vCol;
vec3 ramp(float t) {
  t = uRampLo + (1.0 - uRampLo) * clamp(t, 0.0, 1.0);
  float x = t * 10.0; int i = int(min(floor(x), 9.0));
  return mix(uRamp[i], uRamp[i + 1], x - float(i));
}
void main() {
  if (aAttr.x > uMaxErr || aAttr.y < uMinTrack) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec4 p = uVP * vec4(aPos, 1.0);
  gl_Position = p;
  float s = uAtten ? uSize * 0.0025 * uFocal / max(p.w, 1e-3) : uSize;
  if (uMode == 0) vCol = aCol;
  else if (uMode == 1) vCol = ramp(aAttr.x / uErrRef);
  else if (uMode == 2) vCol = ramp((aAttr.y - 2.0) / (uTrackRef - 2.0));
  else vCol = ramp((aPos.y - uYRange.x) / (uYRange.y - uYRange.x));
  if (uDim) {
    if (aFlag < 0.5) { vCol = mix(vCol, vec3(0.16), 0.75); s *= 0.8; }
    else s = max(s * 1.4, 3.0);
  }
  gl_PointSize = clamp(s, 1.0, 20.0);
}`;
const POINT_FS = `#version 300 es
precision mediump float;
in vec3 vCol; out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  o = vec4(vCol, 1.0);
}`;
const LINE_VS = `#version 300 es
in vec3 aPos; in vec4 aCol;
uniform mat4 uVP;
out vec4 vCol;
void main() { vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); }`;
const LINE_FS = `#version 300 es
precision mediump float;
in vec4 vCol; out vec4 o;
void main() { o = vCol; }`;
const QUAD_VS = `#version 300 es
in vec3 aPos; in vec2 aUv;
uniform mat4 uVP;
out vec2 vUv;
void main() { vUv = aUv; gl_Position = uVP * vec4(aPos, 1.0); }`;
const QUAD_FS = `#version 300 es
precision mediump float;
in vec2 vUv; out vec4 o;
uniform sampler2D uTex; uniform float uAlpha;
void main() { o = vec4(texture(uTex, vUv).rgb, uAlpha); }`;

function program(vs, fs) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace("[0]", "");
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u, a: (name) => gl.getAttribLocation(p, name) };
}
const P_POINT = program(POINT_VS, POINT_FS);
const P_LINE = program(LINE_VS, LINE_FS);
const P_QUAD = program(QUAD_VS, QUAD_FS);

// ------------------------------------------------------------------------------------------
// State
// ------------------------------------------------------------------------------------------
const ui = {
  colorMode: 0, pointSize: 2, maxErr: 4, minTrack: 2, camSize: 0.05, photoAlpha: 0.5,
  showPoints: true, showCams: true, showPlanes: false, showPath: false, showGrid: true,
  align: true, sizeAtten: true, autoRotate: false, showUntri: false,
};
let datasets = [];
let model = null;         // the loaded model (server JSON + derived arrays)
let align = null;         // { c, A, s, key, aligned } maps COLMAP world -> viewer world
let selImage = null;      // image object of model.images
let selImageData = null;  // /api/image response
let selPoint = null;      // /api/point response
let hoverCam = null;
let loadingPath = null, pendingPath = null;
let jobRunning = false, logSince = 0, pollTimer = 0, liveSnap = null;
let shownCount = 0;
const cam = { target: [0, 0, 0], dist: 2.8, yaw: 90 * DEG, pitch: -25 * DEG, roll: 0, fov: 50 * DEG, fromImage: null };
let anim = null;

// GPU buffers
const pointVao = gl.createVertexArray();
const pointBufs = { pos: gl.createBuffer(), col: gl.createBuffer(), attr: gl.createBuffer(), flag: gl.createBuffer() };
let numPoints = 0;
const lineVao = gl.createVertexArray();
const lineBuf = gl.createBuffer();
let lineSegs = { cams: [0, 0], path: [0, 0], grid: [0, 0], rays: [0, 0] };
let lineCount = 0;
const quadVao = gl.createVertexArray();
const quadBuf = gl.createBuffer();
const planeTex = new Map(); // image id -> texture

function setupPointVao() {
  gl.bindVertexArray(pointVao);
  const bind = (buf, name, size, type, normalized) => {
    const loc = P_POINT.a(name);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
  };
  bind(pointBufs.pos, "aPos", 3, gl.FLOAT, false);
  bind(pointBufs.col, "aCol", 3, gl.UNSIGNED_BYTE, true);
  bind(pointBufs.attr, "aAttr", 2, gl.FLOAT, false);
  bind(pointBufs.flag, "aFlag", 1, gl.UNSIGNED_BYTE, false);
  gl.bindVertexArray(lineVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.enableVertexAttribArray(P_LINE.a("aPos"));
  gl.vertexAttribPointer(P_LINE.a("aPos"), 3, gl.FLOAT, false, 28, 0);
  gl.enableVertexAttribArray(P_LINE.a("aCol"));
  gl.vertexAttribPointer(P_LINE.a("aCol"), 4, gl.FLOAT, false, 28, 12);
  gl.bindVertexArray(quadVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(P_QUAD.a("aPos"));
  gl.vertexAttribPointer(P_QUAD.a("aPos"), 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(P_QUAD.a("aUv"));
  gl.vertexAttribPointer(P_QUAD.a("aUv"), 2, gl.FLOAT, false, 20, 12);
  gl.bindVertexArray(null);
}
setupPointVao();

// ------------------------------------------------------------------------------------------
// Server API
// ------------------------------------------------------------------------------------------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
const thumbUrl = (ds, name, size) => `/api/thumb?dataset=${encodeURIComponent(ds)}&name=${encodeURIComponent(name)}&size=${size}`;
const imgCache = new Map();
function loadImg(url) {
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => { imgCache.delete(url); rej(new Error("image load failed")); };
      im.src = url;
    }));
  }
  return imgCache.get(url);
}

let toastTimer = 0;
function toast(msg, isErr = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), isErr ? 7000 : 3500);
}

// ------------------------------------------------------------------------------------------
// Data panel: datasets, models, snapshots
// ------------------------------------------------------------------------------------------
const curDataset = () => datasets.find((d) => d.name === $("dataset").value);
const curModelEntry = () => curDataset()?.models.find((m) => (m.path || m.run) === $("model").value);

async function refreshDatasets(selectPath) {
  let info;
  try {
    info = await api("/api/datasets");
  } catch (e) {
    toast(`Server not reachable: ${e.message}`, true);
    return;
  }
  datasets = info.datasets;
  if (!info.pycolmap) toast("pycolmap is not installed on the server: pip install -r requirements.txt", true);
  const dsSel = $("dataset");
  const prevDs = selectPath ? selectPath.split("/")[0] : dsSel.value;
  dsSel.innerHTML = datasets.map((d) => `<option>${d.name}</option>`).join("");
  if (datasets.some((d) => d.name === prevDs)) dsSel.value = prevDs;
  $("empty").classList.toggle("hidden", datasets.length > 0);
  $("download").classList.toggle("hidden", datasets.length > 0);
  $("recon").classList.toggle("hidden", !datasets.length);
  fillModels(selectPath);
  updateReconSummary();
}

function fillModels(selectPath) {
  const ds = curDataset();
  const sel = $("model");
  const prev = selectPath || sel.value;
  if (!ds) {
    sel.innerHTML = "";
    $("dataset-info").innerHTML = "";
    return;
  }
  $("dataset-info").innerHTML = `<b>${ds.num_images}</b> images · database.db ${ds.has_database ? "<b>present</b>" : "absent"}`;
  const opt = (m) => `<option value="${m.path || m.run}">${m.label}</option>`;
  const refs = ds.models.filter((m) => m.kind === "reference"), runs = ds.models.filter((m) => m.kind === "run");
  sel.innerHTML = (refs.length ? `<optgroup label="Reference">${refs.map(opt).join("")}</optgroup>` : "") +
    (runs.length ? `<optgroup label="Your runs">${runs.map(opt).join("")}</optgroup>` : "");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  onModelSelect();
}

function onModelSelect() {
  const m = curModelEntry();
  $("deleteRun").classList.toggle("hidden", !m || m.kind !== "run");
  $("loadModel").disabled = !m || !m.path;
  updateModelInfo();
  updateReplay();
}

// Which model is selected, and whether it is the one on screen (with its key numbers).
function updateModelInfo() {
  const m = curModelEntry();
  if (!m) {
    $("model-info").innerHTML = "";
    $("replay").classList.add("hidden");
    return;
  }
  let info = m.kind === "reference" ? "Made by COLMAP and shipped with the dataset." : m.params || "";
  if (m.snapshots?.length) info += ` · ${m.snapshots.length} snapshots`;
  if (m.incomplete) info += " · <b>did not finish</b>";
  const shown = model && (model.path === m.path || replaySteps(m).includes(model.path));
  if (loadingPath) info += "<br>Loading…";
  else if (shown) {
    const st = model.stats;
    info += `<br>On screen: <b>${st.num_reg_images}</b> images · <b>${st.num_points3D.toLocaleString()}</b> points · ` +
      `<b>${st.mean_reprojection_error.toFixed(3)} px</b> mean error`;
  } else if (model) info += `<br>On screen: <span class="mono">${model.path}</span>`;
  $("model-info").innerHTML = info;
}

// Replay steps: every snapshot, then the final model.
function replaySteps(entry) {
  if (!entry || !entry.snapshots?.length) return [];
  const steps = [...entry.snapshots];
  if (!entry.incomplete && entry.path) steps.push(entry.path);
  return steps;
}
function updateReplay() {
  const m = curModelEntry();
  const steps = replaySteps(m);
  const show = steps.length > 1 && model && steps.includes(model.path);
  $("replay").classList.toggle("hidden", !show);
  if (!show) return;
  const s = $("snapIdx");
  s.max = steps.length - 1;
  s.value = steps.indexOf(model.path);
  s.nextElementSibling.textContent = `${+s.value + 1}/${steps.length}`;
}

const loadSelected = () => { const m = curModelEntry(); if (m?.path) loadModel(m.path); };
$("dataset").onchange = () => { fillModels(); updateReconSummary(); loadSelected(); };
$("model").onchange = () => { onModelSelect(); loadSelected(); };
$("loadModel").onclick = loadSelected;
$("snapIdx").oninput = (e) => {
  const steps = replaySteps(curModelEntry());
  const p = steps[+e.target.value];
  e.target.nextElementSibling.textContent = `${+e.target.value + 1}/${steps.length}`;
  if (p) loadModel(p, { quiet: true });
};
$("deleteRun").onclick = async () => {
  const m = curModelEntry();
  if (!m || m.kind !== "run" || !confirm(`Delete ${m.run}? This removes the folder from disk.`)) return;
  try {
    await api("/api/delete", { run: m.run });
    if (model && model.path.startsWith(m.run + "/")) clearModel();
    toast(`Deleted ${m.run}`);
    refreshDatasets();
  } catch (e) { toast(e.message, true); }
};
const startDownload = async () => {
  try { await api("/api/download", {}); pollJob(true); } catch (e) { toast(e.message, true); }
};
$("download").onclick = startDownload;
$("emptyDownload").onclick = startDownload;

// ------------------------------------------------------------------------------------------
// Model loading and alignment
// ------------------------------------------------------------------------------------------
const seriesKey = (path) => { const m = path.match(/^([^/]+\/runs\/[^/]+)\//); return m ? m[1] : path; };

async function loadModel(path, { quiet = false, live = false, fresh = false } = {}) {
  if (loadingPath) { pendingPath = { path, quiet, live, fresh }; return; }
  loadingPath = path;
  updateModelInfo();
  if (!quiet && !live) {
    $("loading-text").textContent = `Loading ${path}…`;
    $("loading").classList.remove("hidden");
  }
  try {
    const m = await api(`/api/model?path=${encodeURIComponent(path)}`);
    // snapshots and the final model of one run share the view and the alignment
    const sameSeries = !fresh && model && seriesKey(model.path) === seriesKey(path);
    setModel(m, sameSeries);
  } catch (e) {
    toast(`Could not load ${path}: ${e.message}`, true);
  } finally {
    loadingPath = null;
    $("loading").classList.add("hidden");
    updateModelInfo();
    if (pendingPath) { const p = pendingPath; pendingPath = null; loadModel(p.path, p); }
  }
}

function clearModel() {
  model = null; selImage = selImageData = selPoint = null;
  numPoints = 0; lineCount = 0;
  $("stats-sec").classList.add("hidden");
  $("insp-img").classList.add("hidden");
  $("insp-pt").classList.add("hidden");
  $("hud-model").textContent = ""; $("hud-count").textContent = "";
}

function computeAlignment(m) {
  const imgs = m.images;
  const xyz = m.points.xyz, n = m.points.id.length;
  const centers = imgs.map((im) => im.center);
  // centre: per-axis median of the points (robust to far outliers), else of the cameras
  const c = [0, 1, 2].map((k) => n ? quantile(Array.from({ length: n }, (_, i) => xyz[3 * i + k]), 0.5)
    : quantile(centers.map((p) => p[k]), 0.5));
  let A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  if (ui.align && imgs.length) {
    // COLMAP cameras have +y pointing down in the image, so "up" is minus the mean camera y axis.
    let up = [0, 0, 0], fwd = [0, 0, 0];
    for (const im of imgs) { up = sub(up, im.axes[1]); fwd = add(fwd, im.axes[2]); }
    up = norm(up);
    let x = sub(fwd, mul(up, dot(fwd, up)));
    if (len(x) < 1e-3 * imgs.length) x = Math.abs(up[0]) < 0.9 ? cross(up, [1, 0, 0]) : cross(up, [0, 0, 1]);
    x = norm(x);
    A = [x, up, cross(x, up)];
  }
  const rot = (p) => [dot(A[0], p), dot(A[1], p), dot(A[2], p)];
  const radii = centers.map((p) => len(sub(p, c)));
  const step = Math.max(1, Math.floor(n / 20000));
  for (let i = 0; i < n; i += step) radii.push(Math.hypot(xyz[3 * i] - c[0], xyz[3 * i + 1] - c[1], xyz[3 * i + 2] - c[2]));
  const s = 1 / (quantile(radii, 0.9) || 1);
  return { c, A, s, aligned: ui.align, key: seriesKey(m.path), apply: (p) => mul(rot(sub(p, c)), s), rot };
}

function setModel(m, keepView) {
  // derived per-image quantities in COLMAP world coordinates
  for (const im of m.images) {
    const R = im.R;
    im.axes = [[R[0], R[1], R[2]], [R[3], R[4], R[5]], [R[6], R[7], R[8]]]; // camera x, y, z in world
    im.center = mul(add(add(mul(im.axes[0], im.t[0]), mul(im.axes[1], im.t[1])), mul(im.axes[2], im.t[2])), -1);
    im.camera = m.cameras.find((c) => c.id === im.camera_id);
  }
  m.imagesByName = [...m.images].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  m.imageById = new Map(m.images.map((im) => [im.id, im]));
  m.pointIndex = new Map(m.points.id.map((id, i) => [id, i]));

  const reuseAlign = keepView && align && align.key === seriesKey(m.path) && align.aligned === ui.align;
  model = m;
  if (!reuseAlign) align = computeAlignment(m);
  applyAlignment();

  // keep selections that still exist in the new model
  if (selImage) {
    selImage = m.imageById.get(selImage.id) || null;
    if (selImage) selectImage(selImage, { keepPoint: true });
    else $("insp-img").classList.add("hidden");
  }
  if (selPoint && !m.pointIndex.has(selPoint.id)) { selPoint = null; $("insp-pt").classList.add("hidden"); }
  else if (selPoint) selectPoint(selPoint.id);
  updateFlags();
  if (!keepView) resetView(false);
  updateStats();
  updateReplay();
  updateLegend();
  countShown();
  $("hud-model").textContent = m.path;
}

// (Re)compute viewer-space geometry from COLMAP coordinates and upload it.
function applyAlignment() {
  const m = model;
  const n = m.points.id.length;
  const pos = new Float32Array(3 * n), attr = new Float32Array(2 * n);
  const xyz = m.points.xyz;
  for (let i = 0; i < n; i++) {
    const p = align.apply([xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]]);
    pos[3 * i] = p[0]; pos[3 * i + 1] = p[1]; pos[3 * i + 2] = p[2];
    attr[2 * i] = m.points.err[i]; attr[2 * i + 1] = m.points.track[i];
  }
  m.pos = pos;
  for (const im of m.images) {
    im.vc = align.apply(im.center);
    im.vaxes = im.axes.map(align.rot);
  }
  const ys = [];
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 20000))) ys.push(pos[3 * i + 1]);
  m.yRange = n ? [quantile(ys, 0.02), quantile(ys, 0.98)] : [-1, 1];
  m.groundY = Math.min(m.yRange[0], ...m.images.map((im) => im.vc[1]));

  gl.bindBuffer(gl.ARRAY_BUFFER, pointBufs.pos); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBufs.col); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(m.points.rgb), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBufs.attr); gl.bufferData(gl.ARRAY_BUFFER, attr, gl.STATIC_DRAW);
  m.flags = new Uint8Array(n);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBufs.flag); gl.bufferData(gl.ARRAY_BUFFER, m.flags, gl.DYNAMIC_DRAW);
  numPoints = n;
  buildLines();
  buildQuads();
}

// ------------------------------------------------------------------------------------------
// Line geometry: frustums, camera path, ground grid, track rays
// ------------------------------------------------------------------------------------------
const COL = {
  cam: [0.95, 0.65, 0.25, 0.9], camSel: [0.43, 0.66, 1.0, 1], camObs: [0.3, 0.82, 0.54, 1],
  camHover: [1, 1, 1, 1], path: [0.6, 0.64, 0.7, 0.55], grid: [1, 1, 1, 0.07], gridAxis: [1, 1, 1, 0.16],
  ray: [1, 0.31, 0.85, 0.85],
};

function frustumCorners(im, d) {
  const c = im.camera, [rx, ry, rz] = im.vaxes;
  const at = (u, v) => add(im.vc, mul(add(add(mul(rx, (u - c.cx) / c.fx), mul(ry, (v - c.cy) / c.fy)), rz), d));
  return { at, corners: [at(0, 0), at(c.width, 0), at(c.width, c.height), at(0, c.height)] };
}

function buildLines() {
  const v = [];
  const seg = (a, b, col) => v.push(...a, ...col, ...b, ...col);
  const m = model;
  if (!m) return;
  const obs = new Set(selPoint ? selPoint.track.map((o) => o.image_id) : []);
  const d = ui.camSize;

  let start = 0;
  for (const im of m.images) {
    if (im.id === cam.fromImage) continue;
    const col = im === selImage ? COL.camSel : obs.has(im.id) ? COL.camObs : im === hoverCam ? COL.camHover : COL.cam;
    const { at, corners: k } = frustumCorners(im, d);
    for (let i = 0; i < 4; i++) { seg(im.vc, k[i], col); seg(k[i], k[(i + 1) % 4], col); }
    const w = im.camera.width, h = im.camera.height;
    const tip = at(w / 2, -0.22 * h); // "up" marker above the top edge of the image
    seg(at(0.3 * w, 0), tip, col); seg(tip, at(0.7 * w, 0), col);
  }
  lineSegs.cams = [start, v.length / 7 - start];

  start = v.length / 7;
  const seq = m.imagesByName;
  for (let i = 1; i < seq.length; i++) seg(seq[i - 1].vc, seq[i].vc, COL.path);
  lineSegs.path = [start, v.length / 7 - start];

  start = v.length / 7;
  const y = m.groundY - 0.05, ext = 2.5, stp = 0.25;
  for (let t = -ext; t <= ext + 1e-6; t += stp) {
    const c = Math.abs(t) < 1e-6 ? COL.gridAxis : COL.grid;
    seg([t, y, -ext], [t, y, ext], c); seg([-ext, y, t], [ext, y, t], c);
  }
  lineSegs.grid = [start, v.length / 7 - start];

  start = v.length / 7;
  if (selPoint) {
    const p = align.apply(selPoint.xyz);
    for (const o of selPoint.track) {
      const im = m.imageById.get(o.image_id);
      if (im) seg(p, im.vc, COL.ray);
    }
  }
  lineSegs.rays = [start, v.length / 7 - start];

  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.DYNAMIC_DRAW);
  lineCount = v.length / 7;
}

function buildQuads() {
  if (!model) return;
  const v = [];
  const uv = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const im of model.images) {
    const { corners } = frustumCorners(im, ui.camSize * 0.999);
    corners.forEach((p, i) => v.push(...p, ...uv[i]));
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.DYNAMIC_DRAW);
}

function ensurePlaneTextures() {
  if (!model || !ui.showPlanes) return;
  for (const im of model.images) {
    const key = `${model.dataset}/${im.name}`;
    if (planeTex.has(key)) continue;
    planeTex.set(key, null);
    loadImg(thumbUrl(model.dataset, im.name, 256)).then((img) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      planeTex.set(key, t);
    }).catch(() => {});
  }
}

function updateFlags() {
  if (!model) return;
  const f = model.flags;
  f.fill(0);
  if (selImageData && selImage) {
    for (const id of selImageData.point3D) {
      if (id < 0) continue;
      const i = model.pointIndex.get(id);
      if (i !== undefined) f[i] = 1;
    }
  }
  if (selPoint) {
    const i = model.pointIndex.get(selPoint.id);
    if (i !== undefined) f[i] = 1;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBufs.flag);
  gl.bufferData(gl.ARRAY_BUFFER, f, gl.DYNAMIC_DRAW);
}

// ------------------------------------------------------------------------------------------
// Camera
// ------------------------------------------------------------------------------------------
function camBasis(c = cam) {
  const cp = Math.cos(c.pitch);
  const f = [cp * Math.sin(c.yaw), Math.sin(c.pitch), cp * Math.cos(c.yaw)];
  const r0 = norm(cross(f, [0, 1, 0])), u0 = cross(r0, f);
  const cr = Math.cos(c.roll), sr = Math.sin(c.roll);
  const r = add(mul(r0, cr), mul(u0, sr)), u = sub(mul(u0, cr), mul(r0, sr));
  return { f, r, u, eye: sub(c.target, mul(f, c.dist)) };
}

function viewport() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return { w: glCanvas.width, h: glCanvas.height, dpr };
}

function matrices() {
  const { w, h } = viewport();
  const b = camBasis();
  const proj = perspective(cam.fov, w / h, 0.003, 200);
  const view = viewMatrix(b.eye, b.r, b.u, b.f);
  return { vp: matMul(proj, view), b, focal: h / 2 / Math.tan(cam.fov / 2) };
}

function resetView(animate = true) {
  const to = { target: [0, model ? (model.yRange[0] + model.yRange[1]) / 2 : 0, 0], dist: 2.8, yaw: 90 * DEG, pitch: -28 * DEG, roll: 0, fov: 50 * DEG };
  flyTo(to, animate ? 500 : 0);
}

// Orbit parameters that reproduce the pose of a registered image exactly (incl. roll and FOV).
function cameraFromImage(im) {
  const [rx, ry, rz] = im.vaxes;
  const f = rz;
  const pitch = Math.asin(clamp(f[1], -0.999, 0.999)), yaw = Math.atan2(f[0], f[2]);
  const r0 = norm(cross(f, [0, 1, 0])), u0 = cross(r0, f);
  const roll = Math.atan2(dot(rx, u0), dot(rx, r0));
  const dist = Math.max(0.3, dot(sub([0, 0, 0], im.vc), f));
  return { target: add(im.vc, mul(f, dist)), dist, yaw, pitch, roll, fov: 2 * Math.atan(im.camera.height / 2 / im.camera.fy) };
}

function flyTo(to, ms = 600, onDone) {
  cam.fromImage = null;
  if (!ms) { Object.assign(cam, to); onDone?.(); return; }
  const from = { ...cam, target: [...cam.target] };
  let dy = to.yaw - from.yaw;
  dy = ((dy + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  anim = { from, to: { ...to, yaw: from.yaw + dy }, t0: performance.now(), ms, onDone };
}

function stepAnim(now) {
  if (!anim) return;
  const t = clamp((now - anim.t0) / anim.ms, 0, 1), e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  const { from, to } = anim;
  const L = (a, b) => a + (b - a) * e;
  cam.target = [0, 1, 2].map((k) => L(from.target[k], to.target[k]));
  // interpolate the eye distance in log space so zooming feels uniform
  cam.dist = Math.exp(L(Math.log(from.dist), Math.log(to.dist)));
  cam.yaw = L(from.yaw, to.yaw); cam.pitch = L(from.pitch, to.pitch);
  cam.roll = L(from.roll, to.roll); cam.fov = L(from.fov, to.fov);
  if (t >= 1) { const done = anim.onDone; anim = null; done?.(); }
}

function viewFromImage(im) {
  if (!im) return;
  flyTo(cameraFromImage(im), 700, () => { cam.fromImage = im.id; buildLines(); });
}

// ------------------------------------------------------------------------------------------
// Input
// ------------------------------------------------------------------------------------------
let drag = null;
glCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
glCanvas.addEventListener("pointerdown", (e) => {
  glCanvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false };
  glCanvas.classList.add("dragging");
});
glCanvas.addEventListener("pointermove", (e) => {
  if (!drag) { hover(e); return; }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 3 && !drag.moved) {
    drag.moved = true;
    anim = null;
    if (cam.fromImage !== null) { cam.fromImage = null; buildLines(); }
  }
  if (!drag.moved) return;
  if (drag.pan) {
    const b = camBasis();
    const s = 2 * cam.dist * Math.tan(cam.fov / 2) / glCanvas.clientHeight;
    cam.target = add(cam.target, add(mul(b.r, -dx * s), mul(b.u, dy * s)));
  } else {
    cam.roll *= 0.9; // orbiting drifts back to a level horizon after "view from camera"
    cam.yaw -= dx * 0.005;
    cam.pitch = clamp(cam.pitch - dy * 0.005, -89 * DEG, 89 * DEG);
  }
});
glCanvas.addEventListener("pointerup", (e) => {
  const d = drag;
  drag = null;
  glCanvas.classList.remove("dragging");
  if (d && !d.moved) pick(e);
});
glCanvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  anim = null;
  if (cam.fromImage !== null) { cam.fromImage = null; buildLines(); }
  cam.dist = clamp(cam.dist * Math.exp(e.deltaY * 0.0012), 0.02, 50);
}, { passive: false });
glCanvas.addEventListener("dblclick", (e) => {
  const hit = pickPoint(e);
  if (hit < 0) return;
  const p = [model.pos[3 * hit], model.pos[3 * hit + 1], model.pos[3 * hit + 2]];
  const b = camBasis();
  const dist = Math.max(0.05, len(sub(p, b.eye)));
  // keep the eye where it is and turn to look at the point
  const f = norm(sub(p, b.eye));
  flyTo({ target: p, dist, yaw: Math.atan2(f[0], f[2]), pitch: Math.asin(clamp(f[1], -0.99, 0.99)), roll: 0, fov: cam.fov }, 450);
});

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input[type=text], input[type=number], select, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === "r") resetView();
  else if (k === "a") { ui.autoRotate = !ui.autoRotate; $("autoRotate").checked = ui.autoRotate; }
  else if (k === "c") { ui.showCams = !ui.showCams; $("showCams").checked = ui.showCams; }
  else if (k === "v") viewFromImage(selImage);
  else if (k === "escape") { clearSelection(); }
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") { stepImage(e.key === "ArrowRight" ? 1 : -1); e.preventDefault(); }
});

function project(vp, p, w, h) {
  const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
  const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
  const ww = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (ww <= 1e-4) return null;
  return [(x / ww * 0.5 + 0.5) * w, (1 - (y / ww * 0.5 + 0.5)) * h, ww];
}

function mouseCss(e) {
  const r = glCanvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top, r.width, r.height];
}

function pickCamera(e, radius = 14) {
  if (!model || !ui.showCams) return null;
  const [mx, my, w, h] = mouseCss(e);
  const { vp } = matrices();
  let best = null, bd = radius * radius;
  for (const im of model.images) {
    if (im.id === cam.fromImage) continue;
    const { at } = frustumCorners(im, ui.camSize);
    for (const p of [im.vc, at(im.camera.width / 2, im.camera.height / 2)]) {
      const s = project(vp, p, w, h);
      if (!s) continue;
      const d = (s[0] - mx) ** 2 + (s[1] - my) ** 2;
      if (d < bd) { bd = d; best = im; }
    }
  }
  return best;
}

function pickPoint(e, radius = 8) {
  if (!model || !ui.showPoints) return -1;
  const [mx, my, w, h] = mouseCss(e);
  const { vp } = matrices();
  const pos = model.pos, err = model.points.err, tr = model.points.track;
  const maxErr = ui.maxErr >= 4 ? Infinity : ui.maxErr;
  let best = -1, bs = Infinity;
  for (let i = 0; i < numPoints; i++) {
    if (err[i] > maxErr || tr[i] < ui.minTrack) continue;
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const ww = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (ww <= 1e-3) continue;
    const sx = ((vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / ww * 0.5 + 0.5) * w;
    const sy = (1 - ((vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / ww * 0.5 + 0.5)) * h;
    const d = (sx - mx) ** 2 + (sy - my) ** 2;
    if (d > radius * radius) continue;
    const score = d + ww * 4; // prefer nearer points among those under the cursor
    if (score < bs) { bs = score; best = i; }
  }
  return best;
}

function pick(e) {
  const c = pickCamera(e);
  if (c) { selectImage(c); return; }
  const i = pickPoint(e);
  if (i >= 0) selectPoint(model.points.id[i]);
}

function hover(e) {
  const c = pickCamera(e);
  if (c !== hoverCam) { hoverCam = c; buildLines(); }
  glCanvas.style.cursor = c ? "pointer" : "";
}

// ------------------------------------------------------------------------------------------
// Selection: image and point inspectors
// ------------------------------------------------------------------------------------------
function clearSelection() {
  selImage = selImageData = selPoint = null;
  $("insp-img").classList.add("hidden");
  $("insp-pt").classList.add("hidden");
  updateFlags(); buildLines();
}

function stepImage(dir) {
  if (!model) return;
  const seq = model.imagesByName;
  const i = selImage ? seq.indexOf(selImage) : -1;
  const next = seq[(i + dir + seq.length) % seq.length];
  const following = cam.fromImage !== null;
  selectImage(next, { keepPoint: false });
  if (following) viewFromImage(next);
}

async function selectImage(im, { keepPoint = false } = {}) {
  if (selImageData?.id !== im.id) selImageData = null;
  selImage = im;
  if (!keepPoint) { selPoint = null; $("insp-pt").classList.add("hidden"); }
  $("insp-img").classList.remove("hidden");
  $("img-title").textContent = im.name;
  const c = im.camera;
  const pc = (v) => v.map((x) => x.toFixed(3)).join(", ");
  $("img-info").innerHTML = [
    ["image id / camera id", `${im.id} / ${im.camera_id}`],
    ["camera model", c.model],
    ["size", `${c.width} × ${c.height}`],
    ["params", c.params.map((p) => +p.toFixed(4)).join(", ")],
    ["keypoints", im.num_points2D.toLocaleString()],
    ["triangulated", `${im.num_points3D.toLocaleString()} (${(100 * im.num_points3D / Math.max(1, im.num_points2D)).toFixed(1)} %)`],
    ["centre (COLMAP)", pc(im.center)],
    ["viewing dir", pc(im.axes[2])],
  ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  buildLines();
  const path = model.path;
  try {
    const d = await api(`/api/image?path=${encodeURIComponent(path)}&id=${im.id}`);
    if (selImage !== im || model.path !== path) return;
    selImageData = d;
    updateFlags();
    drawImageInspector();
  } catch (e) { toast(e.message, true); }
}

async function drawImageInspector() {
  const im = selImage;
  if (!im) return;
  const cv = $("imgCanvas");
  const W = cv.clientWidth || 358, dpr = Math.min(devicePixelRatio || 1, 2);
  const H = W * im.camera.height / im.camera.width;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = `${H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  try {
    const img = await loadImg(thumbUrl(model.dataset, im.name, 1024));
    if (selImage !== im) return;
    g.drawImage(img, 0, 0, W, H);
  } catch { g.fillStyle = "#222"; g.fillRect(0, 0, W, H); }
  g.fillStyle = "rgba(0,0,0,0.25)"; g.fillRect(0, 0, W, H);
  const d = selImageData;
  if (!d || d.id !== im.id) return;
  const s = W / im.camera.width;
  const css = getComputedStyle(document.documentElement);
  if (ui.showUntri) {
    g.fillStyle = css.getPropertyValue("--c-untri");
    for (let i = 0; i < d.point3D.length; i++) if (d.point3D[i] < 0) g.fillRect(d.xy[2 * i] * s - 0.5, d.xy[2 * i + 1] * s - 0.5, 1, 1);
  }
  g.fillStyle = css.getPropertyValue("--c-tri");
  for (let i = 0; i < d.point3D.length; i++) if (d.point3D[i] >= 0) g.fillRect(d.xy[2 * i] * s - 1, d.xy[2 * i + 1] * s - 1, 2, 2);
  if (selPoint) {
    const o = selPoint.track.find((t) => t.image_id === im.id);
    if (o) {
      g.strokeStyle = css.getPropertyValue("--c-sel"); g.lineWidth = 2;
      g.beginPath(); g.arc(o.xy[0] * s, o.xy[1] * s, 7, 0, 2 * Math.PI); g.stroke();
    }
  }
}

$("imgCanvas").addEventListener("click", (e) => {
  const d = selImageData, im = selImage;
  if (!d || !im) return;
  const r = e.target.getBoundingClientRect();
  const s = r.width / im.camera.width;
  const mx = (e.clientX - r.left) / s, my = (e.clientY - r.top) / s;
  let best = -1, bd = (8 / s) ** 2;
  for (let i = 0; i < d.point3D.length; i++) {
    if (d.point3D[i] < 0) continue;
    const dd = (d.xy[2 * i] - mx) ** 2 + (d.xy[2 * i + 1] - my) ** 2;
    if (dd < bd) { bd = dd; best = i; }
  }
  if (best >= 0) selectPoint(d.point3D[best], { keepImage: true });
});

async function selectPoint(id, { keepImage = true } = {}) {
  if (!keepImage) { selImage = selImageData = null; $("insp-img").classList.add("hidden"); }
  const path = model.path;
  let d;
  try { d = await api(`/api/point?path=${encodeURIComponent(path)}&id=${id}`); } catch (e) { toast(e.message, true); return; }
  if (model.path !== path) return;
  selPoint = d;
  $("insp-pt").classList.remove("hidden");
  $("pt-title").textContent = `#${id}`;
  $("pt-swatch").style.background = `rgb(${d.rgb.join(",")})`;
  $("pt-info").innerHTML = [
    ["xyz (COLMAP)", d.xyz.map((v) => v.toFixed(3)).join(", ")],
    ["reproj. error", `${d.reproj_error.toFixed(3)} px`],
    ["track length", `${d.track.length} images`],
  ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  updateFlags(); buildLines();
  if (selImage) drawImageInspector();
  drawPatches(d);
}

function drawPatches(d) {
  const box = $("patches");
  box.innerHTML = "";
  const sorted = [...d.track].sort((a, b) => {
    const na = model.imageById.get(a.image_id)?.name || "", nb = model.imageById.get(b.image_id)?.name || "";
    return na.localeCompare(nb, undefined, { numeric: true });
  });
  for (const o of sorted) {
    const im = model.imageById.get(o.image_id);
    if (!im) continue;
    const fig = document.createElement("figure");
    const cv = document.createElement("canvas");
    cv.width = cv.height = 96;
    fig.append(cv);
    const cap = document.createElement("figcaption");
    cap.textContent = im.name;
    fig.append(cap);
    fig.title = `${im.name}  (${o.xy[0].toFixed(1)}, ${o.xy[1].toFixed(1)})`;
    fig.onclick = () => selectImage(im, { keepPoint: true });
    box.append(fig);
    loadImg(thumbUrl(model.dataset, im.name, 1024)).then((img) => {
      if (selPoint !== d) return;
      const s = img.naturalWidth / im.camera.width, half = 28; // 56x56 thumbnail pixels around the keypoint
      const g = cv.getContext("2d");
      g.fillStyle = "#111"; g.fillRect(0, 0, 96, 96);
      g.drawImage(img, o.xy[0] * s - half, o.xy[1] * s - half, 2 * half, 2 * half, 0, 0, 96, 96);
      g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--c-sel");
      g.lineWidth = 1.5;
      g.beginPath(); g.arc(48, 48, 6, 0, 2 * Math.PI); g.stroke();
    }).catch(() => {});
  }
}

$("prevImg").onclick = () => stepImage(-1);
$("nextImg").onclick = () => stepImage(1);
$("viewFrom").onclick = () => viewFromImage(selImage);
$("clearImg").onclick = () => { selImage = selImageData = null; $("insp-img").classList.add("hidden"); updateFlags(); buildLines(); };
$("clearPt").onclick = () => { selPoint = null; $("insp-pt").classList.add("hidden"); updateFlags(); buildLines(); drawImageInspector(); };

// ------------------------------------------------------------------------------------------
// Statistics and histograms
// ------------------------------------------------------------------------------------------
function updateStats() {
  const m = model;
  if (!m) return;
  const st = m.stats;
  $("stats-sec").classList.remove("hidden");
  $("stats").innerHTML = [
    ["registered images", `${st.num_reg_images} / ${curDataset()?.num_images ?? st.num_images}`],
    ["cameras (intrinsics)", m.cameras.length],
    ["3D points", st.num_points3D.toLocaleString()],
    ["observations", st.num_observations.toLocaleString()],
    ["mean track length", st.mean_track_length.toFixed(2)],
    ["mean observations / image", st.mean_observations_per_image.toFixed(0)],
    ["mean reprojection error", `${st.mean_reprojection_error.toFixed(3)} px`],
  ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  drawHistograms();
}

function histBins() {
  const m = model;
  const tb = new Array(19).fill(0); // track 2..19, last bin = 20+
  for (const t of m.points.track) tb[Math.min(18, Math.max(0, t - 2))]++;
  const eMax = 3, nb = 30;
  const eb = new Array(nb).fill(0);
  for (const e of m.points.err) eb[Math.min(nb - 1, Math.floor(e / eMax * nb))]++;
  return {
    track: tb.map((c, i) => ({ label: i === 18 ? "≥ 20" : `${i + 2}`, count: c, active: i + 2 >= ui.minTrack, tick: (i + 2) % 3 === 2 })),
    err: eb.map((c, i) => {
      const lo = i * eMax / nb, hi = (i + 1) * eMax / nb;
      return { label: i === nb - 1 ? `≥ ${lo.toFixed(1)} px` : `${lo.toFixed(1)}–${hi.toFixed(1)} px`, count: c, active: ui.maxErr >= 4 || lo < ui.maxErr, tick: i % 5 === 0, tickLabel: lo.toFixed(1) };
    }),
  };
}

function drawHist(cv, bins, mean) {
  const W = cv.clientWidth, H = 110, dpr = Math.min(devicePixelRatio || 1, 2);
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const accent = css.getPropertyValue("--accent"), muted = css.getPropertyValue("--muted"), line = css.getPropertyValue("--line");
  const padB = 16, padT = 6, plotH = H - padB - padT;
  const max = Math.max(1, ...bins.map((b) => b.count));
  const bw = W / bins.length, gap = 2;
  g.strokeStyle = line; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, H - padB + 0.5); g.lineTo(W, H - padB + 0.5); g.stroke();
  bins.forEach((b, i) => {
    const h = b.count / max * plotH;
    const x = i * bw + gap / 2, w = Math.max(1, bw - gap), y = H - padB - h;
    g.fillStyle = accent;
    g.globalAlpha = b.active ? (cv._hover === i ? 1 : 0.85) : 0.22;
    if (h > 0) {
      const r = Math.min(2, w / 2, h);
      g.beginPath();
      g.moveTo(x, H - padB); g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y);
      g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r); g.lineTo(x + w, H - padB); g.fill();
    }
    g.globalAlpha = 1;
    if (b.tick) {
      const text = b.tickLabel ?? b.label, tx = i * bw + (b.tickLabel ? 0 : bw / 2);
      g.fillStyle = muted; g.font = "10px ui-monospace, Menlo, monospace";
      const tw = g.measureText(text).width;
      g.textAlign = "left";
      g.fillText(text, clamp(tx - tw / 2, 0, W - tw), H - 4);
    }
  });
  if (mean !== undefined) {
    const x = mean * W;
    g.strokeStyle = css.getPropertyValue("--text"); g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, padT); g.lineTo(x, H - padB); g.stroke(); g.setLineDash([]);
    g.fillStyle = css.getPropertyValue("--text"); g.font = "10px system-ui"; g.textAlign = x > W - 40 ? "right" : "left";
    g.fillText(" mean ", x, padT + 8);
  }
  cv._bins = bins;
}

function drawHistograms() {
  if (!model) return;
  const b = histBins();
  drawHist($("histTrack"), b.track);
  drawHist($("histErr"), b.err, model.stats.mean_reprojection_error / 3);
}

for (const id of ["histTrack", "histErr"]) {
  const cv = $(id), tip = cv.nextElementSibling;
  cv.addEventListener("pointermove", (e) => {
    if (!cv._bins) return;
    const r = cv.getBoundingClientRect();
    const i = clamp(Math.floor((e.clientX - r.left) / r.width * cv._bins.length), 0, cv._bins.length - 1);
    const b = cv._bins[i];
    const total = model ? model.points.id.length : 1;
    tip.textContent = `${b.label}: ${b.count.toLocaleString()} points (${(100 * b.count / total).toFixed(1)} %)`;
    tip.classList.remove("hidden");
    tip.style.left = `${clamp(e.clientX - r.left - tip.offsetWidth / 2, 0, r.width - tip.offsetWidth)}px`;
    if (cv._hover !== i) { cv._hover = i; drawHistograms(); }
  });
  cv.addEventListener("pointerleave", () => { tip.classList.add("hidden"); cv._hover = -1; drawHistograms(); });
}

// ------------------------------------------------------------------------------------------
// Display controls
// ------------------------------------------------------------------------------------------
const COLOR_HELP = [
  "Colour of each point, averaged from the pixels that observe it.",
  "Mean reprojection error of each point's observations (0 – 2 px). High values mark weak or wrong matches.",
  "Track length: number of images that observe the point (2 – 12+). Longer tracks are better constrained.",
  "Height along the up axis, estimated from the average camera orientation.",
];
function setSeg(id, attr, val) {
  for (const b of $(id).querySelectorAll("button")) b.classList.toggle("on", b.dataset[attr] === String(val));
}
$("colorMode").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  ui.colorMode = +b.dataset.mode;
  setSeg("colorMode", "mode", ui.colorMode);
  updateLegend();
});
function updateLegend() {
  $("color-help").textContent = COLOR_HELP[ui.colorMode];
  const lg = $("legend");
  lg.classList.toggle("hidden", ui.colorMode === 0 || !model);
  if (ui.colorMode === 1) { $("legend-title").textContent = "reprojection error"; $("legend-lo").textContent = "0 px"; $("legend-hi").textContent = "≥ 2 px"; }
  if (ui.colorMode === 2) { $("legend-title").textContent = "track length"; $("legend-lo").textContent = "2"; $("legend-hi").textContent = "≥ 12"; }
  if (ui.colorMode === 3) { $("legend-title").textContent = "height"; $("legend-lo").textContent = "low"; $("legend-hi").textContent = "high"; }
}

const fmt = {
  pointSize: (v) => `${v} px`, maxErr: (v) => (+v >= 4 ? "off" : `${(+v).toFixed(2)} px`), minTrack: (v) => `${v}`,
  camSize: (v) => (+v).toFixed(3), photoAlpha: (v) => `${Math.round(v * 100)} %`,
};
for (const id of Object.keys(fmt)) {
  const el = $(id), out = el.nextElementSibling;
  const set = () => {
    ui[id] = +el.value;
    out.textContent = fmt[id](el.value);
    if (id === "camSize") { buildLines(); buildQuads(); }
    if (id === "maxErr" || id === "minTrack") { drawHistograms(); countShown(); }
  };
  el.addEventListener("input", set);
  set();
}
for (const id of ["showPoints", "showCams", "showPlanes", "showPath", "showGrid", "align", "sizeAtten", "autoRotate", "showUntri"]) {
  const el = $(id);
  el.checked = ui[id];
  el.addEventListener("change", () => {
    ui[id] = el.checked;
    if (id === "align" && model) { align = computeAlignment(model); applyAlignment(); updateFlags(); resetView(); }
    if (id === "showPlanes") ensurePlaneTextures();
    if (id === "showUntri") drawImageInspector();
  });
}

// ------------------------------------------------------------------------------------------
// Render loop
// ------------------------------------------------------------------------------------------
let lastT = performance.now(), fpsAcc = 0, fpsN = 0;
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(glCanvas.clientWidth * dpr), h = Math.round(glCanvas.clientHeight * dpr);
  if (glCanvas.width !== w || glCanvas.height !== h) { glCanvas.width = w; glCanvas.height = h; }
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
}

const RAMP = new Float32Array(INFERNO.flatMap(hex3));

function frame(now) {
  const dt = now - lastT;
  lastT = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 500) { $("hud-fps").textContent = `${(1000 * fpsN / fpsAcc).toFixed(0)} fps`; fpsAcc = fpsN = 0; }
  resize();
  stepAnim(now);
  if (ui.autoRotate && !drag && !anim) { cam.yaw += dt * 0.00025; if (cam.fromImage !== null) { cam.fromImage = null; buildLines(); } }

  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.clearColor(0.035, 0.04, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!model) { requestAnimationFrame(frame); return; }

  const { vp, focal } = matrices();
  gl.enable(gl.DEPTH_TEST);

  if (ui.showPoints && numPoints) {
    gl.useProgram(P_POINT.p);
    const u = P_POINT.u;
    gl.uniformMatrix4fv(u.uVP, false, vp);
    gl.uniform1f(u.uSize, ui.pointSize * (ui.sizeAtten ? 1 : Math.min(devicePixelRatio || 1, 2)));
    gl.uniform1f(u.uFocal, focal);
    gl.uniform1i(u.uAtten, ui.sizeAtten);
    gl.uniform1i(u.uDim, !!(selImageData && selImage) || !!selPoint);
    gl.uniform1i(u.uMode, ui.colorMode);
    gl.uniform1f(u.uMaxErr, ui.maxErr >= 4 ? 1e9 : ui.maxErr);
    gl.uniform1f(u.uMinTrack, ui.minTrack);
    gl.uniform1f(u.uErrRef, 2);
    gl.uniform1f(u.uTrackRef, 12);
    gl.uniform2f(u.uYRange, model.yRange[0], model.yRange[1]);
    gl.uniform3fv(u.uRamp, RAMP);
    gl.uniform1f(u.uRampLo, RAMP_LO);
    gl.bindVertexArray(pointVao);
    gl.drawArrays(gl.POINTS, 0, numPoints);
  }

  if (ui.showPlanes && ui.showCams) {
    ensurePlaneTextures();
    gl.useProgram(P_QUAD.p);
    gl.uniformMatrix4fv(P_QUAD.u.uVP, false, vp);
    gl.uniform1f(P_QUAD.u.uAlpha, 0.9);
    gl.uniform1i(P_QUAD.u.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(quadVao);
    model.images.forEach((im, i) => {
      const t = planeTex.get(`${model.dataset}/${im.name}`);
      if (!t || im.id === cam.fromImage) return;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.drawArrays(gl.TRIANGLE_FAN, 4 * i, 4);
    });
  }

  if (lineCount) {
    gl.useProgram(P_LINE.p);
    gl.uniformMatrix4fv(P_LINE.u.uVP, false, vp);
    gl.bindVertexArray(lineVao);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const draw = ([s, n]) => n && gl.drawArrays(gl.LINES, s, n);
    if (ui.showGrid) { gl.depthMask(false); draw(lineSegs.grid); gl.depthMask(true); }
    if (ui.showCams) draw(lineSegs.cams);
    if (ui.showPath) draw(lineSegs.path);
    gl.disable(gl.DEPTH_TEST);
    draw(lineSegs.rays);
    gl.disable(gl.BLEND);
  }
  gl.bindVertexArray(null);
  drawOverlay(vp);
  requestAnimationFrame(frame);
}

function drawOverlay(vp) {
  const w = overlay.width, h = overlay.height, dpr = w / Math.max(1, overlay.clientWidth);
  const css = getComputedStyle(document.documentElement);
  // photo of the image we are looking through, for comparing the reprojection by eye
  if (cam.fromImage !== null && ui.photoAlpha > 0) {
    const im = model.imageById.get(cam.fromImage);
    const url = im && thumbUrl(model.dataset, im.name, 1024);
    imgCache.has(url) || loadImg(url).catch(() => {});
    imgCache.get(url)?.then((img) => { im._photo = img; });
    if (im?._photo) {
      const s = h / im.camera.height, iw = im.camera.width * s;
      const ox = w / 2 - im.camera.cx * s, oy = h / 2 - im.camera.cy * s;
      octx.globalAlpha = ui.photoAlpha;
      octx.drawImage(im._photo, ox, oy, iw, h);
      octx.globalAlpha = 1;
    }
  }
  if (selPoint) {
    const p = project(vp, align.apply(selPoint.xyz), w, h);
    if (p) {
      octx.strokeStyle = css.getPropertyValue("--c-sel"); octx.lineWidth = 2 * dpr;
      octx.beginPath(); octx.arc(p[0], p[1], 9 * dpr, 0, 2 * Math.PI); octx.stroke();
    }
  }
  const label = (im, color) => {
    const p = project(vp, im.vc, w, h);
    if (!p) return;
    octx.font = `${11 * dpr}px ui-monospace, Menlo, monospace`;
    const tw = octx.measureText(im.name).width + 10 * dpr;
    octx.fillStyle = "rgba(10,12,16,.8)";
    octx.fillRect(p[0] + 8 * dpr, p[1] - 9 * dpr, tw, 17 * dpr);
    octx.fillStyle = color;
    octx.fillText(im.name, p[0] + 13 * dpr, p[1] + 3.5 * dpr);
  };
  if (ui.showCams) {
    if (hoverCam && hoverCam !== selImage) label(hoverCam, "#fff");
    if (selImage && selImage.id !== cam.fromImage) label(selImage, css.getPropertyValue("--accent"));
  }
  $("hud-count").textContent = `${shownCount.toLocaleString()} pts · ${model.images.length} cams`;
}

function countShown() {
  if (!model) return;
  const maxErr = ui.maxErr >= 4 ? Infinity : ui.maxErr, err = model.points.err, tr = model.points.track;
  let n = 0;
  for (let i = 0; i < err.length; i++) if (err[i] <= maxErr && tr[i] >= ui.minTrack) n++;
  shownCount = n;
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------------------------------
// Reconstruction form and job progress
// ------------------------------------------------------------------------------------------
// Presets instead of raw options. Everything not listed uses COLMAP's defaults
// (reconstruct.py --help shows the full set for the CLI).
const PRESETS = {
  best: {
    count: 0, step: 1, max_image_size: 3200, max_features: 8192, matcher: "exhaustive",
    // seconds per image / per pair, measured on a 16-core CPU with the CPU-only pycolmap wheel
    eta: (n) => n * 2.5 + n * (n - 1) / 2 * 0.05 + n * 2.0,
  },
  quick: {
    count: 0, step: 2, max_image_size: 1600, max_features: 8192, matcher: "exhaustive",
    eta: (n) => { const m = Math.ceil(n / 2); return m * 0.4 + m * (m - 1) / 2 * 0.034 + m * 1.6; },
  },
  reuse: { count: 0, step: 1, reuse_db: true, eta: (n) => n * 1.3 },
};
const COMMON = {
  camera_model: "SIMPLE_RADIAL", camera_mode: "single", mapper: "incremental",
  min_num_matches: 15, snapshot_every: 5,
};
const curPreset = () => document.querySelector("input[name=preset]:checked").value;
const fmtEta = (s) => (s < 90 ? `~${Math.max(10, Math.round(s / 10) * 10)} s` : `~${Math.round(s / 60)} min`);
for (const r of document.querySelectorAll("input[name=preset]")) r.addEventListener("change", updateReconSummary);

function updateReconSummary() {
  const ds = curDataset();
  const n = ds ? ds.num_images : 0;
  $("preset-reuse").classList.toggle("disabled", !ds?.has_database);
  $("preset-reuse").querySelector("input").disabled = !ds?.has_database;
  for (const el of document.querySelectorAll(".eta")) el.textContent = n ? fmtEta(PRESETS[el.dataset.eta].eta(n)) : "";
  const p = { ...COMMON, ...PRESETS[curPreset()] };
  const m = Math.ceil(n / p.step);
  const rows = [["images", p.step > 1 ? `${m} of ${n} (every ${p.step}nd)` : `all ${n}`]];
  if (p.reuse_db) rows.push(["features + matches", "from database.db"]);
  else {
    rows.push(["SIFT", `≤ ${p.max_features} features, ` + (p.max_image_size >= 3072 ? "full resolution" : `≤ ${p.max_image_size} px`)]);
    rows.push(["matching", `exhaustive: ${(m * (m - 1) / 2).toLocaleString()} pairs + RANSAC`]);
  }
  rows.push(["camera", `${p.camera_model}, one shared for all images`]);
  rows.push(["mapper", "incremental, refines focal + distortion"]);
  $("r-settings").innerHTML = rows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  $("runBtn").disabled = n < 2 || jobRunning;
}

$("runBtn").onclick = async () => {
  const ds = curDataset();
  if (!ds) return;
  const preset = curPreset();
  const { eta, ...p } = { ...COMMON, ...PRESETS[preset] };
  const t = new Date(), z = (v) => String(v).padStart(2, "0");
  const run = `${preset}-${t.getFullYear()}${z(t.getMonth() + 1)}${z(t.getDate())}-${z(t.getHours())}${z(t.getMinutes())}${z(t.getSeconds())}`;
  try {
    await api("/api/run", { dataset: ds.name, run, ...p });
    liveSnap = null;
    pollJob(true);
  } catch (e) { toast(e.message, true); }
};
$("cancelBtn").onclick = async () => { try { await api("/api/cancel", {}); } catch (e) { toast(e.message, true); } };

async function pollJob(fresh = false) {
  clearTimeout(pollTimer);
  if (fresh) { logSince = 0; $("log").textContent = ""; }
  let s;
  try { s = await api(`/api/status?since=${logSince}`); } catch { pollTimer = setTimeout(pollJob, 3000); return; }
  if (s.kind === "idle") { jobRunning = false; updateReconSummary(); return; }

  $("job").classList.remove("hidden");
  const wasRunning = jobRunning;
  jobRunning = s.state === "running";
  $("cancelBtn").classList.toggle("hidden", !jobRunning);
  $("runBtn").disabled = jobRunning;
  $("download").disabled = $("emptyDownload").disabled = jobRunning;

  // progress bars
  for (const el of document.querySelectorAll(".stage")) {
    const st = el.dataset.stage, p = s.progress[st];
    const relevant = s.kind === "download" ? st === "download" : st !== "download";
    el.classList.toggle("hidden", !relevant);
    el.classList.remove("done", "failed", "skipped");
    const bar = el.querySelector(".bar > div"), out = el.querySelector("output");
    if (!p) { bar.style.width = "0"; out.textContent = ""; continue; }
    if (p.state === "skipped") { el.classList.add("skipped"); bar.style.width = "0"; out.textContent = "skipped"; continue; }
    const frac = p.total ? p.done / p.total : 0;
    bar.style.width = `${(p.state === "done" ? 1 : frac) * 100}%`;
    if (p.state === "done") el.classList.add("done");
    if (p.state === "failed" || (s.state === "failed" && s.stage === st)) el.classList.add("failed");
    out.textContent = st === "download" ? `${p.done.toFixed(0)} %`
      : p.state === "done" && p.seconds !== undefined ? `${st === "mapping" ? `${p.done}/${p.total} · ` : ""}${p.seconds.toFixed(1)} s`
        : `${p.done}/${p.total}${p.unit ? " blk" : ""}`;
  }
  const db = s.progress.database;
  let info = `<b>${s.state}</b> · ${s.elapsed.toFixed(0)} s` + (s.run_dir ? ` · <span class="mono">${s.run_dir}</span>` : "");
  if (db) info += `<br>${db.keypoints.toLocaleString()} keypoints · ${db.verified_pairs.toLocaleString()} verified pairs · ${db.inlier_matches.toLocaleString()} inlier matches`;
  if (s.error) info += `<br><span style="color:var(--danger)">${s.error.replace(/</g, "&lt;")}</span>`;
  $("job-info").innerHTML = info;

  if (s.log.length) {
    const pre = $("log");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent += s.log.join("\n") + "\n";
    if (pre.textContent.length > 400000) pre.textContent = pre.textContent.slice(-300000);
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
  logSince = s.log_end;

  // live preview of the incremental mapper
  if (jobRunning && $("live").checked && s.latest_snapshot && s.latest_snapshot !== liveSnap) {
    liveSnap = s.latest_snapshot;
    loadModel(liveSnap, { live: true });
  }

  if (wasRunning && !jobRunning) {
    if (s.state === "done" && s.kind === "reconstruct" && s.models.length) {
      toast(`Reconstruction finished: ${s.models[0].images} images, ${s.models[0].points.toLocaleString()} points`);
      await refreshDatasets(s.models[0].path);
      loadModel(s.models[0].path, { quiet: true, fresh: true });
    } else if (s.state === "done" && s.kind === "download") {
      toast("Dataset downloaded");
      await refreshDatasets();
      const ref = curDataset()?.models.find((m) => m.kind === "reference");
      if (ref) loadModel(ref.path);
    } else if (s.state === "failed") {
      toast(`Job failed: ${s.error || "see log"}`, true);
      refreshDatasets();
    } else {
      refreshDatasets();
    }
  }
  updateReconSummary();
  if (jobRunning) pollTimer = setTimeout(pollJob, 700);
}

// ------------------------------------------------------------------------------------------
// Start
// ------------------------------------------------------------------------------------------
window.addEventListener("resize", () => { if (selImage) drawImageInspector(); drawHistograms(); });
updateLegend();
(async () => {
  await refreshDatasets();
  await pollJob(true);
  if (!jobRunning) {
    const ref = curModelEntry();
    if (ref?.path) loadModel(ref.path);
  }
})();
