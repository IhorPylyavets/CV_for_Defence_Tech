// Spherical Harmonics Explorer: plain WebGL2, no dependencies.
// SH constants, order and signs match 3DGS (utils/sh_utils.py, forward.cu) and 02-nerf-visualize/bake_voxels.py.
"use strict";

// --------------------------------------------------------------------------- //
// Spherical harmonics (JS copy of the GLSL below, used for the swatches and UI)
// --------------------------------------------------------------------------- //
const C0 = 0.28209479177387814, C1 = 0.4886025119029199;
const C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396];
const C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
            -0.4570457994644658, 1.445305721320277, -0.5900435899266435];

function shBasis([x, y, z]) {
  const xx = x * x, yy = y * y, zz = z * z;
  return [
    C0,
    -C1 * y, C1 * z, -C1 * x,
    C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
    C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
    C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
    C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy),
  ];
}

const FORMULAS = [
  "0.282",
  "−0.489 · y", "0.489 · z", "−0.489 · x",
  "1.093 · xy", "−1.093 · yz", "0.315 · (3z² − 1)", "−1.093 · xz", "0.546 · (x² − y²)",
  "−0.590 · y(3x² − y²)", "2.891 · xyz", "−0.457 · y(4z² − x² − y²)", "0.373 · z(2z² − 3x² − 3y²)",
  "−0.457 · x(4z² − x² − y²)", "1.445 · z(x² − y²)", "−0.590 · x(x² − 3y²)",
];
const LM = [];  // k -> [l, m], k = l² + l + m
for (let l = 0; l <= 3; l++) for (let m = -l; m <= l; m++) LM.push([l, m]);

// colour as in 3DGS: c(d) = max(0, 0.5 + Σ_k f_k Y_k(d)), only the first (deg+1)² terms
function shColor(coefs, deg, d) {
  const Y = shBasis(d), n = (deg + 1) ** 2, c = [0.5, 0.5, 0.5];
  for (let k = 0; k < n; k++) for (let i = 0; i < 3; i++) c[i] += coefs[k][i] * Y[k];
  return c.map(v => Math.max(0, v));
}

function fibonacciSphere(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / n), th = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
    out.push([Math.cos(th) * Math.sin(phi), Math.sin(th) * Math.sin(phi), Math.cos(phi)]);
  }
  return out;
}

// max |Y_k| over the sphere, so every basis function uses the full colour range
const MAXABS = new Array(16).fill(0);
for (const d of fibonacciSphere(4000)) shBasis(d).forEach((v, k) => { MAXABS[k] = Math.max(MAXABS[k], Math.abs(v)); });

// --------------------------------------------------------------------------- //
// State
// --------------------------------------------------------------------------- //
const state = {
  coefs: Array.from({ length: 16 }, () => [0, 0, 0]),
  deg: 3,
  sel: 0,
  mode: 1,          // 1 colour sphere, 2 colour lobe
  az: 0.6, el: 0.35, dist: 4.2,
  spin: true, axes: true,
};

const PRESETS = {
  gray(c) {},
  diffuse(c) { c[0] = [0.8, 0.35, 0.2].map(v => (v - 0.5) / C0); },
  top(c) { c[0] = [0.1, 0.1, 0.1].map(v => v / C0); c[2] = [0.7, 0.65, 0.55]; c[6] = [0.25, 0.25, 0.3]; },
  rgb(c) { c[3][0] = -1; c[1][1] = -1; c[2][2] = 1; },  // Y_1,1 ∝ −x, Y_1,−1 ∝ −y, Y_1,0 ∝ z
  spec(c) {
    // band-limited "highlight" toward w: f_k ∝ g_l Y_k(w), with g_l a smooth low-pass window
    const w = normalize([0.8, 0.4, 0.6]), Y = shBasis(w);
    const g = l => Math.exp(-0.25 * l * l);
    let peak = 0;
    for (let k = 0; k < 16; k++) peak += g(LM[k][0]) * Y[k] * Y[k];
    for (let k = 0; k < 16; k++) { const v = 0.6 * g(LM[k][0]) * Y[k] / peak; c[k] = [v, v, v]; }
    c[0] = c[0].map((v, i) => v + ([0.12, 0.16, 0.3][i] - 0.5) / C0);  // dark blue base colour
  },
  random(c) {
    c[0] = [0, 1, 2].map(() => (0.2 + 0.6 * Math.random() - 0.5) / C0);
    for (let k = 1; k < 16; k++) c[k] = [0, 1, 2].map(() => (Math.random() * 2 - 1) * 0.9 / (LM[k][0] + 0.5));
  },
};
function applyPreset(name) {
  state.coefs = Array.from({ length: 16 }, () => [0, 0, 0]);
  PRESETS[name](state.coefs);
  state.coefs = state.coefs.map(c => c.map(v => Math.max(-2, Math.min(2, v))));
  syncUI();
}

// --------------------------------------------------------------------------- //
// Math helpers (column-major 4×4)
// --------------------------------------------------------------------------- //
function normalize(v) { const n = Math.hypot(...v); return v.map(x => x / n); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function lookAt(eye, up) {  // looking at the origin
  const z = normalize(eye), x = normalize(cross(up, z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
function eyePos(dist) {
  const ce = Math.cos(state.el);
  return [dist * ce * Math.cos(state.az), dist * ce * Math.sin(state.az), dist * Math.sin(state.el)];
}

// --------------------------------------------------------------------------- //
// WebGL
// --------------------------------------------------------------------------- //
const SH_GLSL = `
const float C0 = 0.28209479177387814, C1 = 0.4886025119029199;
void shEval(vec3 d, out float Y[16]) {
  float x = d.x, y = d.y, z = d.z, xx = x*x, yy = y*y, zz = z*z;
  Y[0] = C0;
  Y[1] = -C1*y; Y[2] = C1*z; Y[3] = -C1*x;
  Y[4] = 1.0925484305920792*x*y; Y[5] = -1.0925484305920792*y*z; Y[6] = 0.31539156525252005*(2.0*zz-xx-yy);
  Y[7] = -1.0925484305920792*x*z; Y[8] = 0.5462742152960396*(xx-yy);
  Y[9] = -0.5900435899266435*y*(3.0*xx-yy); Y[10] = 2.890611442640554*x*y*z;
  Y[11] = -0.4570457994644658*y*(4.0*zz-xx-yy); Y[12] = 0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy);
  Y[13] = -0.4570457994644658*x*(4.0*zz-xx-yy); Y[14] = 1.445305721320277*z*(xx-yy);
  Y[15] = -0.5900435899266435*x*(xx-3.0*yy);
}
uniform vec3 uCoef[16];
uniform int uNum;
vec3 shColor(vec3 d) {
  float Y[16]; shEval(d, Y);
  vec3 c = vec3(0.5);
  for (int k = 0; k < 16; k++) if (k < uNum) c += uCoef[k] * Y[k];
  return max(c, 0.0);
}`;

const VS = `#version 300 es
precision highp float; precision highp int;
in vec3 aDir;
uniform mat4 uMVP;
uniform int uMode;   // 0: basis Y_k on the sphere, 1: colour sphere, 2: colour lobe
out vec3 vDir, vPos;
${SH_GLSL}
void main() {
  vec3 d = normalize(aDir);
  float r = 1.0;
  if (uMode == 2) r = max(dot(shColor(d), vec3(1.0 / 3.0)), 0.02);
  vDir = d; vPos = d * r;
  gl_Position = uMVP * vec4(vPos, 1.0);
}`;

const FS = `#version 300 es
precision highp float; precision highp int;
in vec3 vDir, vPos;
uniform int uMode, uK;
uniform vec3 uEye;
uniform float uDim, uScale;
out vec4 outColor;
${SH_GLSL}
void main() {
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  float shade = 0.3 + 0.7 * abs(dot(n, normalize(uEye - vPos)));
  vec3 c;
  if (uMode == 0) {
    float Y[16]; shEval(normalize(vDir), Y);
    float t = clamp(Y[uK] * uScale, -1.0, 1.0);              // Y / max|Y|, in [-1, 1]
    vec3 zero = vec3(0.86, 0.87, 0.89);                        // diverging map: blue < 0 < orange
    c = t >= 0.0 ? mix(zero, vec3(0.85, 0.42, 0.05), t) : mix(zero, vec3(0.12, 0.35, 0.9), -t);
    if (abs(t) < 1.2 * fwidth(t)) c = vec3(0.2);                // nodal lines, where Y = 0
    c *= (0.6 + 0.4 * shade) * uDim;
  } else if (uMode == 1) {
    c = min(shColor(normalize(vDir)), 1.0);                    // unlit: this IS the colour
  } else {
    c = min(shColor(normalize(vDir)), 1.0) * (0.55 + 0.45 * shade);
  }
  outColor = vec4(c, 1.0);
}`;

const LINE_VS = `#version 300 es
in vec3 aPos; in vec3 aCol; uniform mat4 uMVP; out vec3 vCol;
void main() { vCol = aCol; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const LINE_FS = `#version 300 es
precision highp float; in vec3 vCol; out vec4 o; void main() { o = vec4(vCol, 1.0); }`;

function makeProgram(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(p, i).name.replace("[0]", "");
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

// One renderer per canvas (a WebGL context can't share buffers with another canvas).
function makeRenderer(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) throw new Error("WebGL2 is not available");
  const prog = makeProgram(gl, VS, FS), lines = makeProgram(gl, LINE_VS, LINE_FS);

  // sphere of directions (the vertex shader displaces it)
  const NLAT = 72, NLON = 144, dirs = [], idx = [];
  for (let i = 0; i <= NLAT; i++) {
    const th = Math.PI * i / NLAT;
    for (let j = 0; j <= NLON; j++) {
      const ph = 2 * Math.PI * j / NLON;
      dirs.push(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th));
    }
  }
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const a = i * (NLON + 1) + j, b = a + NLON + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  const sphere = gl.createVertexArray();
  gl.bindVertexArray(sphere);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(dirs), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog.p, "aDir");
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);

  // x, y, z axes
  const L = 1.6, axisData = new Float32Array([
    0, 0, 0, 1, .35, .35, L, 0, 0, 1, .35, .35,
    0, 0, 0, .4, .85, .45, 0, L, 0, .4, .85, .45,
    0, 0, 0, .45, .6, 1, 0, 0, L, .45, .6, 1,
  ]);
  const axes = gl.createVertexArray();
  gl.bindVertexArray(axes);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, axisData, gl.STATIC_DRAW);
  for (const [name, off] of [["aPos", 0], ["aCol", 12]]) {
    const l = gl.getAttribLocation(lines.p, name);
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 3, gl.FLOAT, false, 24, off);
  }
  gl.bindVertexArray(null);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  // draw into the viewport [x, y, w, h] (GL pixels, origin bottom-left)
  function draw(vp, opts) {
    gl.viewport(...vp); gl.scissor(...vp);
    const eye = eyePos(opts.dist);
    const mvp = mul(perspective(0.7, vp[2] / vp[3], 0.1, 50), lookAt(eye, [0, 0, 1]));
    gl.useProgram(prog.p);
    gl.uniformMatrix4fv(prog.u.uMVP, false, mvp);
    gl.uniform1i(prog.u.uMode, opts.mode);
    gl.uniform1i(prog.u.uK, opts.k || 0);
    gl.uniform1f(prog.u.uScale, opts.k != null ? 1 / MAXABS[opts.k] : 1);
    gl.uniform1f(prog.u.uDim, opts.dim ?? 1);
    gl.uniform3fv(prog.u.uEye, eye);
    gl.uniform3fv(prog.u.uCoef, state.coefs.flat());
    gl.uniform1i(prog.u.uNum, (state.deg + 1) ** 2);
    gl.bindVertexArray(sphere);
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_INT, 0);
    if (opts.axes) {
      gl.useProgram(lines.p);
      gl.uniformMatrix4fv(lines.u.uMVP, false, mvp);
      gl.bindVertexArray(axes);
      gl.drawArrays(gl.LINES, 0, 6);
    }
  }

  function begin() {
    resize();
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, canvas.width, canvas.height); gl.scissor(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.082, 0.098, 0.125, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }
  return { canvas, begin, draw };
}

// --------------------------------------------------------------------------- //
// Views
// --------------------------------------------------------------------------- //
const $ = id => document.getElementById(id);
const mainR = makeRenderer($("view"));
const basisR = makeRenderer($("basis"));
const basisStage = $("basis-stage");

// basis pyramid: row l has 2l+1 cells, centred in a 7-wide grid
function basisCells() {
  const W = basisStage.clientWidth, H = basisStage.clientHeight;
  const s = Math.min(W / 7, H / 4), ox = (W - 7 * s) / 2, oy = (H - 4 * s) / 2;
  return LM.map(([l, m], k) => ({ k, l, m, x: ox + (3 + m) * s, y: oy + l * s, s }));
}

const labels = LM.map(([l, m]) => {
  const el = document.createElement("div");
  el.className = "cell-label";
  el.innerHTML = `Y<sub>${l}</sub><sup>${m}</sup>`;
  basisStage.appendChild(el);
  return el;
});
const selBox = document.createElement("div");
selBox.className = "cell-box";
basisStage.appendChild(selBox);

function layoutLabels() {
  for (const c of basisCells()) {
    labels[c.k].style.left = `${c.x + c.s / 2}px`;
    labels[c.k].style.top = `${c.y + c.s - 15}px`;
    labels[c.k].style.opacity = c.l <= state.deg ? 1 : 0.4;
    if (c.k === state.sel) Object.assign(selBox.style, { left: `${c.x + 2}px`, top: `${c.y + 2}px`, width: `${c.s - 4}px`, height: `${c.s - 4}px` });
  }
}

function render() {
  mainR.begin();
  mainR.draw([0, 0, mainR.canvas.width, mainR.canvas.height], { mode: state.mode, dist: state.dist, axes: state.axes });

  basisR.begin();
  const dpr = basisR.canvas.width / basisStage.clientWidth, H = basisR.canvas.height;
  for (const c of basisCells()) {
    const vp = [c.x, H / dpr - c.y - c.s, c.s, c.s].map(v => Math.round(v * dpr));
    basisR.draw(vp, { mode: 0, k: c.k, dist: 4.0, dim: c.l <= state.deg ? 1 : 0.35 });
  }
  layoutLabels();
  updateSwatches();
}

// --------------------------------------------------------------------------- //
// UI
// --------------------------------------------------------------------------- //
const css = c => `rgb(${c.map(v => Math.round(255 * Math.min(1, v))).join(",")})`;
const splatBg = c => `radial-gradient(closest-side, ${css(c)} 0%, ${css(c)} 35%, transparent 100%)`;

function updateSwatches() {
  const d = normalize(eyePos(1).map(v => -v));  // camera -> Gaussian centre (at the origin)
  $("sw-view").style.background = splatBg(shColor(state.coefs, state.deg, d));
  $("sw-dc").style.background = splatBg(shColor(state.coefs, 0, d));
  $("sw-deg").textContent = state.deg;
  $("sw-dir").textContent = `d = (${d.map(v => v.toFixed(2)).join(", ")})`;
}

const sliders = ["cr", "cg", "cb"].map(id => $(id));
sliders.forEach((s, i) => s.addEventListener("input", () => {
  state.coefs[state.sel][i] = +s.value;
  syncUI();
}));

function syncUI() {
  const [l, m] = LM[state.sel];
  $("sel-name").innerHTML = `f<sub>${state.sel}</sub> · Y<sub>${l}</sub><sup>${m}</sup>`;
  $("sel-formula").textContent = `Y = ${FORMULAS[state.sel]}` + (l > state.deg ? "   (not used at this degree)" : "");
  sliders.forEach((s, i) => { s.value = state.coefs[state.sel][i]; s.nextElementSibling.textContent = (+s.value).toFixed(2); });

  const n = (state.deg + 1) ** 2;
  $("count").textContent = `${n} × 3 = ${3 * n} floats`;
  $("deg-help").textContent = state.deg === 0
    ? "Degree 0: one constant per channel, so the colour is the same from every direction (diffuse)."
    : `Degree ${state.deg}: (${state.deg}+1)² = ${n} coefficients per colour channel. ` +
      (state.deg === 3 ? "This is what 3DGS uses: f_dc (3) + f_rest (45) = 48 floats per Gaussian."
                       : "Higher degree means sharper view-dependent effects.");

  $("coefs").innerHTML = "<tr><td>k  (l,m)</td><td>R</td><td>G</td><td>B</td></tr>" + state.coefs.map((c, k) =>
    `<tr data-k="${k}" class="${k === state.sel ? "sel" : ""} ${k >= n ? "off" : ""}">` +
    `<td>${String(k).padStart(2)}  (${LM[k][0]},${LM[k][1]})</td>${c.map(v => `<td>${v.toFixed(2)}</td>`).join("")}</tr>`).join("");
  document.querySelectorAll("#degree button").forEach(b => b.classList.toggle("on", +b.dataset.v === state.deg));
  document.querySelectorAll("#mode button").forEach(b => b.classList.toggle("on", +b.dataset.v === state.mode));
}

function select(k) { state.sel = k; syncUI(); }
$("coefs").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (tr) select(+tr.dataset.k); });
$("zero-sel").addEventListener("click", () => { state.coefs[state.sel] = [0, 0, 0]; syncUI(); });
document.querySelectorAll("#degree button").forEach(b => b.addEventListener("click", () => { state.deg = +b.dataset.v; syncUI(); }));
document.querySelectorAll("#mode button").forEach(b => b.addEventListener("click", () => { state.mode = +b.dataset.v; syncUI(); }));
document.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", () => applyPreset(b.dataset.preset)));
$("spin").addEventListener("change", e => { state.spin = e.target.checked; });
$("axes").addEventListener("change", e => { state.axes = e.target.checked; });

// orbit (both canvases share one camera); a click without dragging on the pyramid selects a basis
for (const canvas of [mainR.canvas, basisR.canvas]) {
  let drag = null;
  canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy); drag.x = e.clientX; drag.y = e.clientY;
    state.az -= dx * 0.01;
    state.el = Math.max(-1.5, Math.min(1.5, state.el + dy * 0.01));
  });
  canvas.addEventListener("pointerup", e => {
    if (drag && drag.moved < 4 && canvas === basisR.canvas) {
      const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const hit = basisCells().find(c => px >= c.x && px < c.x + c.s && py >= c.y && py < c.y + c.s);
      if (hit) select(hit.k);
    }
    drag = null;
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    if (canvas === mainR.canvas) state.dist = Math.max(2, Math.min(10, state.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });
}

let last = performance.now();
function frame(t) {
  if (state.spin) state.az += (t - last) * 0.0003;
  last = t;
  render();
  requestAnimationFrame(frame);
}

applyPreset("random");
requestAnimationFrame(frame);
