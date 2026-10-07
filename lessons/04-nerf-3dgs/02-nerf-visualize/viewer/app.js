// NeRF Voxel Explorer: WebGL2 alpha compositing through a baked NeRF voxel grid.
// No MLP is evaluated here; σ and SH colour coefficients come from 3D textures.

const $ = (id) => document.getElementById(id);
const glCanvas = $("gl");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const gl = glCanvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
if (!gl) {
  document.body.innerHTML = "<p style='padding:20px'>WebGL2 is required.</p>";
  throw new Error("no webgl2");
}

// ------------------------------------------------------------------------------------------
// Small vector helpers
// ------------------------------------------------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => mul(a, 1 / Math.hypot(a[0], a[1], a[2]));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// float16 -> float32 lookup table (for the CPU-side ray inspector)
const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  HALF[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
}

// ------------------------------------------------------------------------------------------
// Shaders
// ------------------------------------------------------------------------------------------
const VS = `#version 300 es
in vec2 aPos; out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
out vec4 outColor;

uniform sampler3D uT0;                       // rgb = SH band 0, a = sigma
uniform sampler3D uT1, uT2, uT3;             // SH degree 1
uniform sampler3D uT4, uT5, uT6, uT7, uT8;   // SH degree 2
uniform int uShDeg;
uniform bool uViewDep;

uniform vec3 uPos, uRight, uUp, uFwd;
uniform float uTanH, uAspect, uOrthoH;
uniform bool uOrtho;
uniform vec3 uBMin, uBMax;
uniform vec3 uDirSign;                        // NDC scenes: view dirs were world-space
uniform float uDt, uSigmaScale, uSigmaThresh;
uniform bool uEarly, uJitter;
uniform float uFrame;
uniform vec3 uBg;
uniform int uMode;
uniform vec2 uDepthRange;
uniform float uSigmaRef;
uniform int uSliceAxis;
uniform float uSlicePos, uSliceAlpha;
uniform vec3 uRes;

vec3 turbo(float x) {
  const vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kR2 = vec2(-152.94239396, 59.28637943);
  const vec2 kG2 = vec2(4.27729857, 2.82956604);
  const vec2 kB2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kR) + dot(v2, kR2), dot(v4, kG) + dot(v2, kG2), dot(v4, kB) + dot(v2, kB2));
}

float hash(vec2 p) {
  p = fract(p * vec2(443.897, 441.423) + uFrame * 0.618);
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

vec3 shColor(vec3 uvw, vec3 c0, vec3 d) {
  vec3 c = 0.28209479 * c0;
  if (uViewDep && uShDeg >= 1) {
    float x = d.x, y = d.y, z = d.z;
    c += -0.48860251 * y * texture(uT1, uvw).rgb
         + 0.48860251 * z * texture(uT2, uvw).rgb
         - 0.48860251 * x * texture(uT3, uvw).rgb;
    if (uShDeg >= 2) {
      c += 1.09254843 * x * y * texture(uT4, uvw).rgb
         - 1.09254843 * y * z * texture(uT5, uvw).rgb
         + 0.31539157 * (2.0 * z * z - x * x - y * y) * texture(uT6, uvw).rgb
         - 1.09254843 * x * z * texture(uT7, uvw).rgb
         + 0.54627422 * (x * x - y * y) * texture(uT8, uvw).rgb;
    }
  }
  return clamp(c, 0.0, 1.0);
}

vec4 sliceColor(vec3 uvw) {
  float s = texture(uT0, uvw).a * uSigmaScale;
  vec3 col = turbo(log(1.0 + s) / log(1.0 + uSigmaRef));
  // grid lines every 8 voxels
  vec3 g = uvw * uRes / 8.0;
  vec3 fw = fwidth(g);
  vec3 l = smoothstep(vec3(0.0), fw * 1.2, abs(fract(g - 0.5) - 0.5));
  float line = 1.0;
  if (uSliceAxis != 0) line = min(line, l.x);
  if (uSliceAxis != 1) line = min(line, l.y);
  if (uSliceAxis != 2) line = min(line, l.z);
  col = mix(vec3(1.0), col, 0.65 + 0.35 * line);
  return vec4(col, uSliceAlpha);
}

void main() {
  vec3 ro, rd;
  if (uOrtho) {
    ro = uPos + vNdc.x * uOrthoH * uAspect * uRight + vNdc.y * uOrthoH * uUp;
    rd = uFwd;
  } else {
    ro = uPos;
    rd = normalize(uFwd + vNdc.x * uTanH * uAspect * uRight + vNdc.y * uTanH * uUp);
  }
  vec3 bg = uMode == 0 ? uBg : vec3(0.06);

  // ray / box intersection
  vec3 inv = 1.0 / rd;
  vec3 ta = (uBMin - ro) * inv, tb = (uBMax - ro) * inv;
  vec3 tmn = min(ta, tb), tmx = max(ta, tb);
  float t0 = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
  float t1 = min(min(tmx.x, tmx.y), tmx.z);
  if (t0 >= t1) { outColor = vec4(bg, 1.0); return; }

  vec3 bsize = uBMax - uBMin;
  vec3 dSH = rd * uDirSign;

  // slice plane hit
  float tSlice = 1e30;
  if (uSliceAxis >= 0) {
    float planeW = uBMin[uSliceAxis] + uSlicePos * bsize[uSliceAxis];
    float ts = (planeW - ro[uSliceAxis]) / rd[uSliceAxis];
    if (ts >= t0 && ts <= t1) tSlice = ts;
  }

  vec3 C = vec3(0.0);
  vec4 sliceOver = vec4(0.0);
  float T = 1.0, depth = 0.0, sigMax = 0.0;
  float t = t0 + uDt * (uJitter ? hash(gl_FragCoord.xy) : 0.5);

  for (int i = 0; i < 4096; i++) {
    if (t > t1) break;
    if (tSlice < t) {
      vec4 sc = sliceColor((ro + tSlice * rd - uBMin) / bsize);
      if (uMode == 0) { C += T * sc.a * sc.rgb; T *= 1.0 - sc.a; }
      else sliceOver = vec4(sc.rgb, sc.a * step(1e-3, T));
      tSlice = 1e30;
    }
    vec3 uvw = (ro + t * rd - uBMin) / bsize;
    vec4 s0 = texture(uT0, uvw);
    float sigma = s0.a * uSigmaScale;
    if (sigma < uSigmaThresh) sigma = 0.0;
    sigMax = max(sigMax, sigma);
    if (sigma > 0.0) {
      float alpha = 1.0 - exp(-sigma * uDt);
      float w = T * alpha;
      if (uMode == 0 && w > 1e-5) C += w * shColor(uvw, s0.rgb, dSH);
      depth += w * t;
      T *= 1.0 - alpha;
      if (uEarly && T < 1e-3) break;
    }
    t += uDt;
  }
  if (tSlice < 1e29 && T >= 1e-3) {   // slice lies behind the last sample taken
    vec4 sc = sliceColor((ro + tSlice * rd - uBMin) / bsize);
    if (uMode == 0) { C += T * sc.a * sc.rgb; T *= 1.0 - sc.a; }
    else sliceOver = sc;
  }

  float acc = 1.0 - T;
  vec3 col;
  if (uMode == 0) col = C + T * bg;
  else if (uMode == 1) {
    float dn = (depth / max(acc, 1e-6) - uDepthRange.x) / (uDepthRange.y - uDepthRange.x);
    col = mix(bg, turbo(1.0 - dn), acc);
  }
  else if (uMode == 2) col = vec3(acc);
  else col = sigMax > 0.0 ? turbo(log(1.0 + sigMax) / log(1.0 + uSigmaRef)) : bg;
  if (uMode != 0) col = mix(col, sliceOver.rgb, sliceOver.a);
  outColor = vec4(col, 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);
const U = {};
for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++) {
  const name = gl.getActiveUniform(prog, i).name;
  U[name] = gl.getUniformLocation(prog, name);
}
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, "aPos");
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

const timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");

// ------------------------------------------------------------------------------------------
// State
// ------------------------------------------------------------------------------------------
const S = {
  meta: null, sigma: null, sh: null, textures: [],
  mode: 0, steps: 320, sigmaScale: 1, sigmaThresh: 0, renderScale: 1, fov: 40,
  viewDep: true, early: true, jitter: false, ortho: false, whiteBg: true, showBox: true, autoRotate: false,
  sliceAxis: -1, slicePos: 0.5, sliceAlpha: 0.85,
  cam: { target: [0, 0, 0], radius: 4, yaw: 0.8, pitch: 0.5 },
  ray: null, hoverSample: -1, frame: 0, dirty: true, lastGpuMs: null,
};

const MODE_HELP = [
  "Alpha-composited colour: C = Σ Tᵢ αᵢ cᵢ + T·background.",
  "Expected ray termination depth Σ wᵢ tᵢ / Σ wᵢ (near = warm).",
  "Accumulated opacity 1 − T. Grey halos are semi-transparent 'fog' the MLP learned.",
  "Maximum σ along each ray (log scale), similar to an X-ray / MIP.",
];

// ------------------------------------------------------------------------------------------
// Camera
// ------------------------------------------------------------------------------------------
function upAxis() { return S.meta ? S.meta.up : [0, 0, 1]; }
function camBasis() {
  const up = upAxis();
  // two axes perpendicular to "up" for the yaw circle
  const e1 = up[2] ? [1, 0, 0] : [0, 0, 1];
  const e2 = cross(up, e1);
  const { target, radius, yaw, pitch } = S.cam;
  const dir = add(mul(add(mul(e1, Math.cos(yaw)), mul(e2, Math.sin(yaw))), Math.cos(pitch)), mul(up, Math.sin(pitch)));
  const pos = add(target, mul(dir, radius));
  const fwd = norm(sub(target, pos));
  let right = norm(cross(fwd, up));
  const camUp = cross(right, fwd);
  if (S.meta && S.meta.ndc) right = mul(right, -1);  // NDC space is mirrored in x
  const tanH = Math.tan((S.fov * Math.PI) / 360);
  return { pos, fwd, right, up: camUp, tanH, orthoH: radius * tanH };
}

function resetCamera() {
  const m = S.meta;
  const c = mul(add(m.bbox_min, m.bbox_max), 0.5);
  if (m.ndc) {
    S.cam = { target: c, radius: 3.0, yaw: Math.PI, pitch: 0.0 };
    S.ortho = true; S.fov = 40;
  } else {
    // blender synthetic scenes: cameras on a sphere of radius ~4, camera_angle_x = 0.691 rad
    S.cam = { target: c, radius: 4.0, yaw: -Math.PI / 4 - Math.PI / 2, pitch: 0.5 };
    S.ortho = false; S.fov = 40;
  }
  $("ortho").checked = S.ortho;
  setSlider("fov", S.fov);
  S.dirty = true;
}

function rayFor(px, py, w, h, cb) {
  const nx = (px / w) * 2 - 1, ny = 1 - (py / h) * 2, aspect = w / h;
  if (S.ortho) {
    const o = add(cb.pos, add(mul(cb.right, nx * cb.orthoH * aspect), mul(cb.up, ny * cb.orthoH)));
    return { o, d: cb.fwd };
  }
  const d = norm(add(cb.fwd, add(mul(cb.right, nx * cb.tanH * aspect), mul(cb.up, ny * cb.tanH))));
  return { o: cb.pos, d };
}

function project(p, cb, w, h) {
  const v = sub(p, cb.pos), z = dot(v, cb.fwd), aspect = w / h;
  if (!S.ortho && z < 1e-3) return null;
  const sx = S.ortho ? dot(v, cb.right) / (cb.orthoH * aspect) : dot(v, cb.right) / (z * cb.tanH * aspect);
  const sy = S.ortho ? dot(v, cb.up) / cb.orthoH : dot(v, cb.up) / (z * cb.tanH);
  return [(sx * 0.5 + 0.5) * w, (0.5 - sy * 0.5) * h, z];
}

// ------------------------------------------------------------------------------------------
// Loading
// ------------------------------------------------------------------------------------------
async function fetchBinary(url, label, base, span) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) setLoading(`${label} ${(got / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MB`, base + span * (got / total));
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return new Uint16Array(out.buffer);
}

function setLoading(text, frac) {
  $("loading").classList.toggle("hidden", text == null);
  if (text != null) { $("loading-text").textContent = text; $("loading-bar").style.width = `${(frac * 100).toFixed(1)}%`; }
}

function makeTex3D(internal, format, data, rx, ry, rz) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, internal, rx, ry, rz, 0, format, gl.HALF_FLOAT, data);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
  return t;
}

async function loadScene(name) {
  try {
    setLoading(`Loading ${name}…`, 0);
    const meta = await (await fetch(`scenes/${name}/meta.json`)).json();
    const sigma = await fetchBinary(`scenes/${name}/${meta.files.density}`, "density", 0, 0.1);
    const sh = await fetchBinary(`scenes/${name}/${meta.files.sh}`, "colour (SH)", 0.1, 0.85);
    setLoading("Uploading 3D textures to GPU…", 0.97);
    await new Promise((r) => setTimeout(r, 10));

    for (const t of S.textures) gl.deleteTexture(t);
    const [rx, ry, rz] = meta.resolution, n = rx * ry * rz;
    const nCoef = (meta.sh_degree + 1) ** 2;
    if (sh.length !== nCoef * 3 * n || sigma.length !== n) throw new Error("file sizes do not match meta.json");

    // texture 0: SH band 0 (rgb) + sigma (a)
    const t0 = new Uint16Array(n * 4);
    for (let i = 0; i < n; i++) {
      t0[4 * i] = sh[i]; t0[4 * i + 1] = sh[n + i]; t0[4 * i + 2] = sh[2 * n + i]; t0[4 * i + 3] = sigma[i];
    }
    const textures = [makeTex3D(gl.RGBA16F, gl.RGBA, t0, rx, ry, rz)];
    const tk = new Uint16Array(n * 3);
    for (let k = 1; k < nCoef; k++) {
      const b = k * 3 * n;
      for (let i = 0; i < n; i++) { tk[3 * i] = sh[b + i]; tk[3 * i + 1] = sh[b + n + i]; tk[3 * i + 2] = sh[b + 2 * n + i]; }
      textures.push(makeTex3D(gl.RGB16F, gl.RGB, tk, rx, ry, rz));
    }
    S.textures = textures;
    S.meta = meta; S.sigma = sigma; S.sh = sh; S.ray = null;
    S.whiteBg = meta.white_background; $("whiteBg").checked = S.whiteBg;
    $("viewdep").disabled = meta.sh_degree === 0;
    resetCamera();
    updateSceneInfo();
    updateInspector();
    setLoading(null);
  } catch (e) {
    console.error(e);
    setLoading(`Failed to load ${name}: ${e.message}`, 0);
  }
}

function updateSceneInfo() {
  const m = S.meta;
  const [rx, ry, rz] = m.resolution;
  const mb = ((rx * ry * rz * (1 + 3 * (m.sh_degree + 1) ** 2) * 2) / 2 ** 20).toFixed(1);
  const vs = m.voxel_size.map((v) => v.toFixed(4)).join(" × ");
  $("scene-info").innerHTML =
    `<b>${rx}×${ry}×${rz}</b> voxels · SH degree <b>${m.sh_degree}</b> · ${mb} MB<br>` +
    `voxel size ${vs}${m.ndc ? " (NDC units)" : ""}<br>` +
    `source MLP: ${m.model}${m.psnr ? `, train PSNR ${m.psnr.toFixed(1)} dB` : ""}<br>` +
    `<span class="mono">${m.checkpoint}</span>`;
}

// ------------------------------------------------------------------------------------------
// Render
// ------------------------------------------------------------------------------------------
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = glCanvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr * S.renderScale));
  const h = Math.max(1, Math.round(r.height * dpr * S.renderScale));
  if (glCanvas.width !== w || glCanvas.height !== h) { glCanvas.width = w; glCanvas.height = h; S.dirty = true; }
  const ow = Math.round(r.width * dpr), oh = Math.round(r.height * dpr);
  if (overlay.width !== ow || overlay.height !== oh) { overlay.width = ow; overlay.height = oh; S.dirty = true; }
}

function stepSize() {
  const m = S.meta;
  return Math.hypot(...sub(m.bbox_max, m.bbox_min)) / S.steps;
}

let pendingQuery = null;
function render() {
  const m = S.meta;
  if (!m) return;
  const w = glCanvas.width, h = glCanvas.height;
  gl.viewport(0, 0, w, h);
  const cb = camBasis();
  gl.uniform3fv(U.uPos, cb.pos); gl.uniform3fv(U.uRight, cb.right); gl.uniform3fv(U.uUp, cb.up); gl.uniform3fv(U.uFwd, cb.fwd);
  gl.uniform1f(U.uTanH, cb.tanH); gl.uniform1f(U.uAspect, w / h); gl.uniform1f(U.uOrthoH, cb.orthoH);
  gl.uniform1i(U.uOrtho, S.ortho);
  gl.uniform3fv(U.uBMin, m.bbox_min); gl.uniform3fv(U.uBMax, m.bbox_max);
  gl.uniform3fv(U.uRes, m.resolution);
  gl.uniform3fv(U.uDirSign, m.ndc ? [1, 1, -1] : [1, 1, 1]);
  gl.uniform1f(U.uDt, stepSize());
  gl.uniform1f(U.uSigmaScale, S.sigmaScale); gl.uniform1f(U.uSigmaThresh, S.sigmaThresh);
  gl.uniform1i(U.uEarly, S.early); gl.uniform1i(U.uJitter, S.jitter);
  gl.uniform1f(U.uFrame, S.frame % 1000);
  gl.uniform3fv(U.uBg, S.whiteBg ? [1, 1, 1] : [0, 0, 0]);
  gl.uniform1i(U.uMode, S.mode);
  const dc = Math.hypot(...sub(cb.pos, S.cam.target)), half = Math.hypot(...sub(m.bbox_max, m.bbox_min)) / 2;
  gl.uniform2f(U.uDepthRange, Math.max(0, dc - half), dc + half);
  gl.uniform1f(U.uSigmaRef, Math.max(1, m.sigma_p99 * S.sigmaScale));
  gl.uniform1i(U.uSliceAxis, S.sliceAxis); gl.uniform1f(U.uSlicePos, S.slicePos); gl.uniform1f(U.uSliceAlpha, S.sliceAlpha);
  gl.uniform1i(U.uShDeg, m.sh_degree); gl.uniform1i(U.uViewDep, S.viewDep);
  for (let i = 0; i < 9; i++) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_3D, S.textures[Math.min(i, S.textures.length - 1)]);
    gl.uniform1i(U[`uT${i}`], i);
  }

  let q = null;
  if (timerExt && !pendingQuery) { q = gl.createQuery(); gl.beginQuery(timerExt.TIME_ELAPSED_EXT, q); }
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  if (q) { gl.endQuery(timerExt.TIME_ELAPSED_EXT); pendingQuery = q; }
  drawOverlay(cb);
  S.frame++;
}

function pollTimer() {
  if (!pendingQuery) return;
  const avail = gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT_AVAILABLE);
  const disjoint = gl.getParameter(timerExt.GPU_DISJOINT_EXT);
  if (avail || disjoint) {
    if (avail && !disjoint) S.lastGpuMs = gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT) / 1e6;
    gl.deleteQuery(pendingQuery);
    pendingQuery = null;
  }
}

function drawOverlay(cb) {
  const w = overlay.width, h = overlay.height, m = S.meta;
  octx.clearRect(0, 0, w, h);
  const dpr = w / glCanvas.getBoundingClientRect().width;
  if (S.showBox) {
    const a = m.bbox_min, b = m.bbox_max;
    const corners = [];
    for (let i = 0; i < 8; i++) corners.push([i & 1 ? b[0] : a[0], i & 2 ? b[1] : a[1], i & 4 ? b[2] : a[2]]);
    const P = corners.map((c) => project(c, cb, w, h));
    octx.lineWidth = 1 * dpr;
    octx.strokeStyle = S.whiteBg && S.mode === 0 ? "rgba(40,60,100,.45)" : "rgba(160,190,255,.45)";
    octx.beginPath();
    for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) {
      const j = i | bit;
      if (j !== i && P[i] && P[j]) { octx.moveTo(P[i][0], P[i][1]); octx.lineTo(P[j][0], P[j][1]); }
    }
    octx.stroke();
    // axis gizmo at the min corner
    const cols = ["#ff5d5d", "#5dd35d", "#5d8bff"];
    const L = Math.min(...sub(b, a)) * 0.25;
    for (let k = 0; k < 3; k++) {
      const e = [...a]; e[k] += L;
      const p0 = P[0], p1 = project(e, cb, w, h);
      if (!p0 || !p1) continue;
      octx.strokeStyle = cols[k]; octx.lineWidth = 2 * dpr;
      octx.beginPath(); octx.moveTo(p0[0], p0[1]); octx.lineTo(p1[0], p1[1]); octx.stroke();
      octx.fillStyle = cols[k]; octx.font = `${11 * dpr}px system-ui`;
      octx.fillText("xyz"[k], p1[0] + 3 * dpr, p1[1] - 3 * dpr);
    }
  }
  if (S.ray) drawRayOverlay(cb, w, h, dpr);
}

function drawRayOverlay(cb, w, h, dpr) {
  const R = S.ray;
  const pa = project(add(R.o, mul(R.d, R.t0)), cb, w, h), pb = project(add(R.o, mul(R.d, R.tEnd)), cb, w, h);
  if (pa && pb) {
    octx.beginPath(); octx.moveTo(pa[0], pa[1]); octx.lineTo(pb[0], pb[1]);
    octx.strokeStyle = "rgba(0,0,0,.55)"; octx.lineWidth = 4 * dpr; octx.stroke();
    octx.strokeStyle = "rgba(255,220,80,.95)"; octx.lineWidth = 1.5 * dpr; octx.setLineDash([5 * dpr, 4 * dpr]);
    octx.stroke(); octx.setLineDash([]);
  }
  const wmax = Math.max(1e-6, ...R.samples.map((s) => s.w));
  for (let i = 0; i < R.samples.length; i++) {
    const s = R.samples[i];
    const p = project(add(R.o, mul(R.d, s.t)), cb, w, h);
    if (!p) continue;
    const rr = (1.2 + 5 * Math.sqrt(s.w / wmax)) * dpr;
    octx.fillStyle = `rgb(${s.c.map((v) => Math.round(v * 255)).join(",")})`;
    octx.strokeStyle = i === S.hoverSample ? "#fff" : "rgba(0,0,0,.6)";
    octx.lineWidth = (i === S.hoverSample ? 2 : 0.8) * dpr;
    octx.beginPath(); octx.arc(p[0], p[1], i === S.hoverSample ? rr + 3 * dpr : rr, 0, Math.PI * 2); octx.fill(); octx.stroke();
  }
  // the pixel we clicked, in the view it was clicked from
  const po = project(add(R.o, mul(R.d, R.t0)), cb, w, h);
  if (po) {
    octx.strokeStyle = "#ffdc50"; octx.lineWidth = 1.5 * dpr;
    octx.beginPath(); octx.arc(po[0], po[1], 6 * dpr, 0, Math.PI * 2); octx.stroke();
  }
}

// ------------------------------------------------------------------------------------------
// CPU ray inspector: same maths as the shader, but keeps every sample
// ------------------------------------------------------------------------------------------
function sampleGrid(p) {
  // replicates GL_LINEAR + CLAMP_TO_EDGE lookup at texture coord uvw
  const m = S.meta, [rx, ry, rz] = m.resolution, n = rx * ry * rz;
  const res = [rx, ry, rz];
  const f = [0, 0, 0], i0 = [0, 0, 0], i1 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const u = ((p[k] - m.bbox_min[k]) / (m.bbox_max[k] - m.bbox_min[k])) * res[k] - 0.5;
    const fl = Math.floor(u);
    f[k] = u - fl;
    i0[k] = clamp(fl, 0, res[k] - 1); i1[k] = clamp(fl + 1, 0, res[k] - 1);
  }
  const nC = (m.sh_degree + 1) ** 2;
  const out = new Float32Array(1 + nC * 3);
  for (let c = 0; c < 8; c++) {
    const ix = c & 1 ? i1[0] : i0[0], iy = c & 2 ? i1[1] : i0[1], iz = c & 4 ? i1[2] : i0[2];
    const wgt = (c & 1 ? f[0] : 1 - f[0]) * (c & 2 ? f[1] : 1 - f[1]) * (c & 4 ? f[2] : 1 - f[2]);
    if (wgt === 0) continue;
    const idx = (iz * ry + iy) * rx + ix;
    out[0] += wgt * HALF[S.sigma[idx]];
    for (let j = 0; j < nC * 3; j++) out[1 + j] += wgt * HALF[S.sh[j * n + idx]];
  }
  return out;
}

function evalSH(v, d) {
  const deg = S.viewDep ? S.meta.sh_degree : 0;
  const [x, y, z] = d;
  const Y = [0.28209479];
  if (deg >= 1) Y.push(-0.48860251 * y, 0.48860251 * z, -0.48860251 * x);
  if (deg >= 2) Y.push(1.09254843 * x * y, -1.09254843 * y * z, 0.31539157 * (2 * z * z - x * x - y * y), -1.09254843 * x * z, 0.54627422 * (x * x - y * y));
  const c = [0, 0, 0];
  for (let k = 0; k < Y.length; k++) for (let ch = 0; ch < 3; ch++) c[ch] += Y[k] * v[1 + k * 3 + ch];
  return c.map((x) => clamp(x, 0, 1));
}

function traceRay(o, d) {
  const m = S.meta, dt = stepSize();
  let t0 = 0, t1 = Infinity;
  for (let k = 0; k < 3; k++) {
    const inv = 1 / d[k];
    let ta = (m.bbox_min[k] - o[k]) * inv, tb = (m.bbox_max[k] - o[k]) * inv;
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
  }
  if (!(t0 < t1)) return null;
  const dSH = m.ndc ? [d[0], d[1], -d[2]] : d;
  const samples = [];
  let T = 1, C = [0, 0, 0], depth = 0;
  for (let t = t0 + 0.5 * dt; t <= t1; t += dt) {
    const v = sampleGrid(add(o, mul(d, t)));
    let sigma = v[0] * S.sigmaScale;
    if (sigma < S.sigmaThresh) sigma = 0;
    const alpha = 1 - Math.exp(-sigma * dt);
    const w = T * alpha;
    const c = evalSH(v, dSH);
    samples.push({ t, sigma, alpha, T, w, c });
    for (let ch = 0; ch < 3; ch++) C[ch] += w * c[ch];
    depth += w * t;
    T *= 1 - alpha;
    if (S.early && T < 1e-3) break;
  }
  const bg = S.whiteBg ? 1 : 0;
  return {
    o, d, t0, t1, dt, samples, T, depth: depth / Math.max(1e-6, 1 - T),
    tEnd: samples.length ? samples[samples.length - 1].t : t1,
    color: C.map((x) => x + T * bg),
  };
}

function inspectAt(clientX, clientY) {
  if (!S.meta) return;
  const r = glCanvas.getBoundingClientRect();
  const cb = camBasis();
  const { o, d } = rayFor(clientX - r.left, clientY - r.top, r.width, r.height, cb);
  S.ray = traceRay(o, d);
  S.rayPixel = [clientX - r.left, clientY - r.top];
  S.hoverSample = -1;
  updateInspector();
  S.dirty = true;
}

function retrace() {
  if (S.ray) { S.ray = traceRay(S.ray.o, S.ray.d); updateInspector(); }
}

const plot = $("plot"), pctx = plot.getContext("2d");
function plotRange() {
  const R = S.ray, smp = R.samples;
  if (!$("zoomSurface").checked) return [0, smp.length - 1];
  let a = smp.findIndex((s) => s.alpha > 1e-3);
  if (a < 0) return [0, smp.length - 1];
  let b = a;
  for (let i = a; i < smp.length; i++) { if (smp[i].alpha > 1e-3) b = i; if (smp[i].T < 2e-3) break; }
  const pad = Math.max(4, Math.round((b - a) * 0.15));
  return [Math.max(0, a - pad), Math.min(smp.length - 1, b + pad)];
}

function updateInspector() {
  const R = S.ray;
  $("insp-empty").classList.toggle("hidden", !!R);
  $("insp-body").classList.toggle("hidden", !R);
  if (!R) return;
  const css = (c) => `rgb(${c.map((v) => Math.round(clamp(v, 0, 1) * 255)).join(",")})`;
  $("insp-swatch").style.background = css(R.color);
  const nz = R.samples.filter((s) => s.alpha > 1e-4).length;
  $("insp-stats").innerHTML =
    `C = (<b>${R.color.map((v) => v.toFixed(3)).join(", ")}</b>)<br>` +
    `samples <b>${R.samples.length}</b> (${nz} with σ&gt;0), δ = <b>${R.dt.toFixed(4)}</b><br>` +
    `opacity Σw = <b>${(1 - R.T).toFixed(3)}</b>, background T<sub>N</sub> = <b>${R.T.toFixed(3)}</b><br>` +
    `depth E[t] = <b>${(1 - R.T) > 0.01 ? R.depth.toFixed(3) : "–"}</b>`;
  drawPlot();
}

function drawPlot() {
  const R = S.ray;
  if (!R) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = plot.clientWidth, ch = plot.clientHeight;
  plot.width = cw * dpr; plot.height = ch * dpr;
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, cw, ch);
  const [a, b] = plotRange();
  const smp = R.samples.slice(a, b + 1);
  const L = 30, Rr = 8, T = 8, B = 20, W = cw - L - Rr, H = ch - T - B;
  const X = (i) => L + (smp.length > 1 ? (i / (smp.length - 1)) * W : W / 2);
  const Y = (v) => T + (1 - v) * H;
  // axes
  pctx.strokeStyle = "#2f3642"; pctx.lineWidth = 1;
  pctx.fillStyle = "#8a93a3"; pctx.font = "10px ui-monospace, monospace";
  for (const v of [0, 0.5, 1]) {
    pctx.beginPath(); pctx.moveTo(L, Y(v)); pctx.lineTo(L + W, Y(v)); pctx.stroke();
    pctx.fillText(v.toFixed(1), 4, Y(v) + 3);
  }
  if (smp.length) {
    pctx.fillText(`t=${smp[0].t.toFixed(2)}`, L, ch - 5);
    const tl = `t=${smp[smp.length - 1].t.toFixed(2)}`;
    pctx.fillText(tl, L + W - pctx.measureText(tl).width, ch - 5);
  }
  const css = getComputedStyle(document.documentElement);
  const smax = Math.max(1e-6, ...smp.map((s) => s.sigma));
  const wmax = Math.max(1e-6, ...smp.map((s) => s.w));
  // sigma bars
  const bw = Math.max(1, W / Math.max(1, smp.length) - 0.5);
  pctx.fillStyle = css.getPropertyValue("--c-sigma") + "66";
  smp.forEach((s, i) => { const y = Y(s.sigma / smax); pctx.fillRect(X(i) - bw / 2, y, bw, T + H - y); });
  const line = (key, color, scale = 1) => {
    pctx.strokeStyle = color; pctx.lineWidth = 1.6; pctx.beginPath();
    smp.forEach((s, i) => (i ? pctx.lineTo : pctx.moveTo).call(pctx, X(i), Y(s[key] / scale)));
    pctx.stroke();
  };
  line("alpha", css.getPropertyValue("--c-alpha"));
  line("T", css.getPropertyValue("--c-trans"));
  line("w", css.getPropertyValue("--c-weight"), wmax);
  // hover marker
  const hi = S.hoverSample - a;
  if (hi >= 0 && hi < smp.length) {
    pctx.strokeStyle = "#fff"; pctx.setLineDash([3, 3]);
    pctx.beginPath(); pctx.moveTo(X(hi), T); pctx.lineTo(X(hi), T + H); pctx.stroke(); pctx.setLineDash([]);
  }
  // colour strips
  for (const [id, fn] of [["strip-c", (s) => s.c], ["strip-w", (s) => s.c.map((v) => v * s.w / wmax)]]) {
    const cv = $(id), cx = cv.getContext("2d");
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.fillStyle = "#000"; cx.fillRect(0, 0, cv.clientWidth, cv.clientHeight);
    smp.forEach((s, i) => {
      const x0 = i ? (X(i - 1) + X(i)) / 2 : L, x1 = i < smp.length - 1 ? (X(i) + X(i + 1)) / 2 : L + W;
      cx.fillStyle = `rgb(${fn(s).map((v) => Math.round(clamp(v, 0, 1) * 255)).join(",")})`;
      cx.fillRect(x0, 0, x1 - x0 + 0.5, cv.clientHeight);
    });
  }
  plot._map = { a, n: smp.length, L, W };
}

plot.addEventListener("mousemove", (e) => {
  if (!S.ray || !plot._map) return;
  const { a, n, L, W } = plot._map;
  const x = e.offsetX;
  const i = clamp(Math.round(((x - L) / W) * (n - 1)), 0, n - 1) + a;
  if (i === S.hoverSample) return;
  S.hoverSample = i;
  const s = S.ray.samples[i];
  $("insp-hover").textContent =
    `sample ${i}: t=${s.t.toFixed(3)}  σ=${s.sigma.toFixed(2)}\n` +
    `α = 1−exp(−σδ) = ${s.alpha.toFixed(4)}   T = ${s.T.toFixed(4)}\n` +
    `w = T·α = ${s.w.toFixed(4)}   c = (${s.c.map((v) => v.toFixed(2)).join(", ")})`;
  drawPlot();
  S.dirty = true;
});
plot.addEventListener("mouseleave", () => { S.hoverSample = -1; drawPlot(); S.dirty = true; });

// ------------------------------------------------------------------------------------------
// UI wiring
// ------------------------------------------------------------------------------------------
const SLIDERS = {
  steps: { fmt: (v) => v.toFixed(0), set: (v) => (S.steps = v), retrace: true },
  sigmaScale: { fmt: (v) => (10 ** v).toFixed(2) + "×", set: (v) => (S.sigmaScale = 10 ** v), retrace: true },
  sigmaThresh: { fmt: (v) => v.toFixed(0), set: (v) => (S.sigmaThresh = v), retrace: true },
  renderScale: { fmt: (v) => Math.round(v * 100) + "%", set: (v) => { S.renderScale = v; resize(); } },
  fov: { fmt: (v) => v.toFixed(0) + "°", set: (v) => (S.fov = v) },
  slicePos: { fmt: (v) => v.toFixed(3), set: (v) => (S.slicePos = v) },
  sliceAlpha: { fmt: (v) => v.toFixed(2), set: (v) => (S.sliceAlpha = v) },
};
function setSlider(id, v) { const el = $(id); el.value = v; el.nextElementSibling.textContent = SLIDERS[id].fmt(+v); }
for (const [id, cfg] of Object.entries(SLIDERS)) {
  const el = $(id);
  const on = () => { const v = +el.value; cfg.set(v); el.nextElementSibling.textContent = cfg.fmt(v); if (cfg.retrace) retrace(); S.dirty = true; };
  el.addEventListener("input", on);
  on();
}
const CHECKS = { viewdep: "viewDep", early: "early", jitter: "jitter", ortho: "ortho", whiteBg: "whiteBg", showBox: "showBox", autoRotate: "autoRotate" };
for (const [id, key] of Object.entries(CHECKS)) {
  $(id).addEventListener("change", (e) => { S[key] = e.target.checked; if (["viewdep", "early", "whiteBg"].includes(id)) retrace(); S.dirty = true; });
}
$("zoomSurface").addEventListener("change", drawPlot);
$("clearRay").addEventListener("click", () => { S.ray = null; updateInspector(); S.dirty = true; });

function segmented(id, attr, onPick) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onPick(+b.dataset[attr]);
    S.dirty = true;
  });
}
const LEGENDS = [null, ["Depth", "far", "near"], ["Opacity 1−T", "0", "1", true], ["max σ (log)", "0", "p99"]];
function setMode(m) {
  S.mode = m;
  $("mode-help").textContent = MODE_HELP[m];
  $("hud-mode").textContent = ["RGB", "Depth", "Opacity", "Density"][m];
  const L = LEGENDS[m];
  $("legend").classList.toggle("hidden", !L);
  if (L) {
    $("legend-title").textContent = L[0]; $("legend-lo").textContent = L[1]; $("legend-hi").textContent = L[2];
    document.querySelector(".legend-ramp").classList.toggle("gray", !!L[3]);
    if (m === 3 && S.meta) $("legend-hi").textContent = `σ ${(S.meta.sigma_p99 * S.sigmaScale).toFixed(0)}`;
  }
}
segmented("mode", "mode", setMode);
segmented("sliceAxis", "axis", (a) => (S.sliceAxis = a));
setMode(0);

// mouse / touch camera controls
let drag = null;
glCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
glCanvas.addEventListener("pointerdown", (e) => {
  glCanvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false };
  glCanvas.classList.add("dragging");
});
glCanvas.addEventListener("pointermove", (e) => {
  if (!drag || !S.meta) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 4) drag.moved = true;
  if (drag.pan) {
    const cb = camBasis(), h = glCanvas.getBoundingClientRect().height;
    const scale = (2 * (S.ortho ? cb.orthoH : S.cam.radius * cb.tanH)) / h;
    S.cam.target = add(S.cam.target, add(mul(cb.right, -dx * scale), mul(cb.up, dy * scale)));
  } else {
    const flip = S.meta.ndc ? -1 : 1;
    S.cam.yaw -= flip * dx * 0.008;
    S.cam.pitch = clamp(S.cam.pitch + dy * 0.008, -1.55, 1.55);
  }
  S.dirty = true;
});
glCanvas.addEventListener("pointerup", (e) => {
  glCanvas.classList.remove("dragging");
  if (drag && !drag.moved && e.button === 0) inspectAt(e.clientX, e.clientY);
  drag = null;
});
glCanvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  S.cam.radius = clamp(S.cam.radius * Math.exp(e.deltaY * 0.001), 0.2, 30);
  S.dirty = true;
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "r" || e.key === "R") resetCamera();
  if (e.key === "a" || e.key === "A") { S.autoRotate = !S.autoRotate; $("autoRotate").checked = S.autoRotate; }
  for (let k = 1; k <= 4; k++) if (e.key === String(k)) $("mode").querySelectorAll("button")[k - 1].click();
});
window.addEventListener("resize", () => { resize(); if (S.ray) drawPlot(); });

// main loop: only re-render when something changed (or while animating)
let lastT = performance.now(), fpsEma = 0;
function loop(now) {
  const dt = now - lastT; lastT = now;
  resize();
  if (timerExt) pollTimer();
  if (S.autoRotate && S.meta) {
    // forward-facing (NDC) scenes only make sense from the front, so wobble instead of orbiting
    if (S.meta.ndc) S.cam.yaw = Math.PI + 0.35 * Math.sin(now * 0.0008);
    else S.cam.yaw += dt * 0.0004;
    S.dirty = true;
  }
  if (S.jitter) S.dirty = true;
  if (S.dirty && S.meta) {
    render();
    S.dirty = false;
    fpsEma = fpsEma ? fpsEma * 0.9 + (1000 / Math.max(dt, 1)) * 0.1 : 1000 / Math.max(dt, 1);
    $("hud-fps").textContent = S.lastGpuMs != null ? `${S.lastGpuMs.toFixed(1)} ms GPU` : `${fpsEma.toFixed(0)} fps`;
    $("hud-res").textContent = `${glCanvas.width}×${glCanvas.height} · ${S.steps} spp`;
  }
  requestAnimationFrame(loop);
}

// boot
(async () => {
  let scenes = [];
  try { scenes = (await (await fetch("scenes/index.json")).json()).scenes; } catch (e) { /* none baked yet */ }
  const sel = $("scene");
  if (!scenes.length) {
    setLoading("No baked scenes found. Run:  python bake_voxels.py", 0);
    return;
  }
  for (const s of scenes) sel.add(new Option(s, s));
  const initial = new URLSearchParams(location.search).get("scene");
  sel.value = scenes.includes(initial) ? initial : scenes.includes("lego") ? "lego" : scenes[0];
  sel.addEventListener("change", () => loadScene(sel.value));
  await loadScene(sel.value);
  requestAnimationFrame(loop);
})();
