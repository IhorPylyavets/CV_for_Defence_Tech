// WebGL2 splat renderer for display. Gaussians are continuous, so the display can be
// rendered at screen resolution regardless of the (small) training resolution.
// Additive mode: sum of c*G into a float framebuffer.  Alpha mode: premultiplied
// "over" blending, drawing back-to-front (last Gaussian first), which equals the
// front-to-back compositing used in training.
import { STRIDE, DEFAULT_CUTOFF } from "./engine.js";

const VS = `#version 300 es
in vec2 aCorner;
in vec4 aA;      // mean.xy, log scale.xy
in vec4 aB;      // theta, rgb
in float aOp;    // opacity logit
uniform float uAspect, uCut;
uniform int uHighlight;
out vec2 vLocal; out vec3 vColor; out float vOp; flat out int vId;
void main() {
  vec2 s = exp(aA.zw);
  float c = cos(aB.x), sn = sin(aB.x);
  vec2 l = aCorner * uCut;
  vec2 off = vec2(c * l.x * s.x - sn * l.y * s.y, sn * l.x * s.x + c * l.y * s.y);
  vec2 p = aA.xy + off;
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y / uAspect * 2.0, 0.0, 1.0);
  vLocal = l; vColor = aB.yzw; vOp = 1.0 / (1.0 + exp(-aOp)); vId = gl_InstanceID;
}`;

const FS = `#version 300 es
precision highp float;
in vec2 vLocal; in vec3 vColor; in float vOp; flat in int vId;
uniform int uMode;   // 0 additive, 1 alpha, 2 flat ellipses
uniform float uCut;
out vec4 o;
void main() {
  float q = dot(vLocal, vLocal);
  if (q > uCut * uCut) discard;
  float G = exp(-0.5 * q);
  if (uMode == 0) o = vec4(vColor * G, 0.0);
  else if (uMode == 1) {
    float a = min(0.99, vOp * G);
    if (a < 1.0 / 255.0) discard;
    o = vec4(vColor * a, a);
  } else {
    if (q > 1.0) discard;
    float a = 0.75;
    o = vec4(clamp(vColor, 0.0, 1.0) * a, a);
  }
}`;

const VS_Q = `#version 300 es
in vec2 aCorner; out vec2 vUv;
void main() { vUv = aCorner * 0.5 + 0.5; gl_Position = vec4(aCorner, 0.0, 1.0); }`;
const FS_Q = `#version 300 es
precision highp float;
in vec2 vUv; uniform sampler2D uTex; uniform vec3 uBg; uniform int uUseBg; out vec4 o;
void main() {
  vec4 a = texture(uTex, vUv);
  vec3 c = a.rgb + (uUseBg == 1 ? (1.0 - a.a) * uBg : vec3(0.0));
  o = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

export class SplatRenderer {
  constructor(canvas) {
    const gl = (this.gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true }));
    if (!gl) throw new Error("WebGL2 not available");
    this.canvas = canvas;
    this.floatOK = !!gl.getExtension("EXT_color_buffer_float");
    this.prog = program(gl, VS, FS);
    this.progQ = program(gl, VS_Q, FS_Q);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = (n) => gl.getAttribLocation(this.prog, n);
    gl.enableVertexAttribArray(loc("aCorner"));
    gl.vertexAttribPointer(loc("aCorner"), 2, gl.FLOAT, false, 0, 0);
    this.inst = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    const B = STRIDE * 4;
    for (const [name, size, off] of [["aA", 4, 0], ["aB", 4, 16], ["aOp", 1, 32]]) {
      const l = loc(name);
      gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, size, gl.FLOAT, false, B, off);
      gl.vertexAttribDivisor(l, 1);
    }
    this.vaoQ = gl.createVertexArray();
    gl.bindVertexArray(this.vaoQ);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const lq = gl.getAttribLocation(this.progQ, "aCorner");
    gl.enableVertexAttribArray(lq);
    gl.vertexAttribPointer(lq, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.n = 0; this.fbo = null; this.fw = 0; this.fh = 0;
  }

  setParams(params, n, reverse) {
    const gl = this.gl;
    let data = params;
    if (reverse) { // back-to-front for "over" blending
      data = new Float32Array(n * STRIDE);
      for (let i = 0; i < n; i++) data.set(params.subarray((n - 1 - i) * STRIDE, (n - i) * STRIDE), i * STRIDE);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, n * STRIDE), gl.DYNAMIC_DRAW);
    this.n = n;
  }

  ensureFbo(w, h) {
    const gl = this.gl;
    if (this.fbo && this.fw === w && this.fh === h) return;
    if (this.fbo) { gl.deleteFramebuffer(this.fbo); gl.deleteTexture(this.tex); }
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.floatOK ? gl.RGBA16F : gl.RGBA8, w, h, 0, gl.RGBA, this.floatOK ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    this.fw = w; this.fh = h;
  }

  // mode: "additive" | "alpha" | "splats"
  render(mode, aspect, bg) {
    const gl = this.gl, w = this.canvas.width, h = this.canvas.height;
    this.ensureFbo(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.enable(gl.BLEND);
    if (mode === "additive") gl.blendFunc(gl.ONE, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform1f(gl.getUniformLocation(this.prog, "uAspect"), aspect);
    gl.uniform1f(gl.getUniformLocation(this.prog, "uCut"), DEFAULT_CUTOFF);
    gl.uniform1i(gl.getUniformLocation(this.prog, "uMode"), mode === "additive" ? 0 : mode === "alpha" ? 1 : 2);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.n);
    gl.disable(gl.BLEND);
    // composite to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.progQ);
    gl.bindVertexArray(this.vaoQ);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(gl.getUniformLocation(this.progQ, "uTex"), 0);
    gl.uniform3fv(gl.getUniformLocation(this.progQ, "uBg"), bg);
    gl.uniform1i(gl.getUniformLocation(this.progQ, "uUseBg"), mode === "additive" ? 0 : 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }
}
