// Tiny 2D Gaussian splatting engine: forward render, analytic backward pass, Adam,
// and 3DGS-style adaptive density control. Plain JS, no dependencies, runs in a worker.
//
// Each Gaussian has STRIDE = 9 parameters:
//   0,1  mean (x, y)       normalised: x in [0,1], y in [0, H/W]  (image coords, y down)
//   2,3  log scale (sx,sy) log of std-dev, in units of image width
//   4    rotation theta    radians
//   5-7  colour (r, g, b)  unbounded, as in the reference notebook
//   8    opacity logit     only used for alpha compositing: o = sigmoid(logit)
//
// Two image formation models:
//   additive (reference notebook):   I(p) = sum_i c_i G_i(p)
//   alpha    (3D Gaussian Splatting): I(p) = sum_i c_i a_i T_i + T_N bg,
//                                      a_i = min(0.99, o_i G_i(p)),  T_i = prod_{j<i} (1 - a_j)
// with G_i(p) = exp(-1/2 (p - mu_i)^T Sigma_i^-1 (p - mu_i)),  Sigma_i = R diag(s^2) R^T.

export const STRIDE = 9;
export const GROUPS = ["position", "scale", "rotation", "color", "opacity"];
const GROUP_OF = [0, 0, 1, 1, 2, 3, 3, 3, 4];
export const DEFAULT_CUTOFF = 3.5; // Gaussians are evaluated within 3.5 sigma
const TILE = 16;
const DEFAULT_ALPHA_MIN = 1 / 255, ALPHA_MAX = 0.99, DEFAULT_T_MIN = 1e-4;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => Math.log(p / (1 - p));

function gauss() { // standard normal sample
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

export class Model {
  constructor(capacity = 1024) {
    this.n = 0;
    this.cap = 0;
    this.grow(capacity);
  }

  grow(cap) {
    if (cap <= this.cap) return;
    const re = (Old, Ctor, size) => { const a = new Ctor(size); if (Old) a.set(Old.subarray(0, Math.min(Old.length, size))); return a; };
    this.p = re(this.p, Float32Array, cap * STRIDE);   // parameters
    this.m = re(this.m, Float32Array, cap * STRIDE);   // Adam 1st moment
    this.v = re(this.v, Float32Array, cap * STRIDE);   // Adam 2nd moment
    this.t = re(this.t, Int32Array, cap);              // Adam step per Gaussian (for bias correction)
    this.gacc = re(this.gacc, Float32Array, cap);      // accumulated |dL/dmean| (densification statistic)
    this.gcnt = re(this.gcnt, Int32Array, cap);
    this.g = new Float64Array(cap * STRIDE);           // gradient scratch
    this.cap = cap;
  }

  // append one Gaussian, returns its index (fresh optimiser state)
  push(vals) {
    if (this.n >= this.cap) this.grow(Math.max(16, this.cap * 2));
    const i = this.n++, o = i * STRIDE;
    for (let k = 0; k < STRIDE; k++) { this.p[o + k] = vals[k]; this.m[o + k] = 0; this.v[o + k] = 0; }
    this.t[i] = 0; this.gacc[i] = 0; this.gcnt[i] = 0;
    return i;
  }

  get(i) { return Array.from(this.p.subarray(i * STRIDE, (i + 1) * STRIDE)); }

  // keep only Gaussians with keep[i] = true, preserving order
  compact(keep) {
    let w = 0;
    for (let i = 0; i < this.n; i++) {
      if (!keep[i]) continue;
      if (w !== i) {
        this.p.copyWithin(w * STRIDE, i * STRIDE, (i + 1) * STRIDE);
        this.m.copyWithin(w * STRIDE, i * STRIDE, (i + 1) * STRIDE);
        this.v.copyWithin(w * STRIDE, i * STRIDE, (i + 1) * STRIDE);
        this.t[w] = this.t[i]; this.gacc[w] = this.gacc[i]; this.gcnt[w] = this.gcnt[i];
      }
      w++;
    }
    this.n = w;
  }
}

// --------------------------------------------------------------------------------------
// Initialisation
// --------------------------------------------------------------------------------------
export function initModel(model, n, target, W, H, opts) {
  model.n = 0;
  model.grow(n);
  const aspect = H / W;
  const ls = Math.log(opts.initScale);
  const side = Math.ceil(Math.sqrt(n / aspect));
  for (let i = 0; i < n; i++) {
    let x, y;
    if (opts.init === "grid") {
      const cols = side, rows = Math.ceil(n / cols);
      x = ((i % cols) + 0.5) / cols;
      y = ((Math.floor(i / cols) + 0.5) / rows) * aspect;
    } else {
      x = Math.random();
      y = Math.random() * aspect;
    }
    let r, g, b;
    if (opts.init === "random") {
      r = Math.random(); g = Math.random(); b = Math.random();
    } else {
      const px = Math.min(W - 1, Math.floor(x * W)), py = Math.min(H - 1, Math.floor((y / aspect) * H));
      const o = 3 * (py * W + px);
      // additive model: overlapping Gaussians sum up, so start with a fraction of the colour
      const f = opts.blend === "additive" ? 0.5 : 1;
      r = target[o] * f; g = target[o + 1] * f; b = target[o + 2] * f;
    }
    const theta = opts.randomRotation ? Math.random() * Math.PI : 0;
    model.push([x, y, ls, ls, theta, r, g, b, logit(opts.initOpacity)]);
  }
}

// exp(-q/2) for q in [0, qmax] as a lookup table with linear interpolation:
// several times faster than Math.exp in the inner loops, max abs. error ~1e-7.
const LUT_CACHE = new Map();
function expTable(qmax) {
  let t = LUT_CACHE.get(qmax);
  if (!t) {
    const N = Math.ceil(qmax * 4096);
    t = new Float64Array(N + 2);
    for (let j = 0; j < N + 2; j++) t[j] = Math.exp((-0.5 * j) / 4096);
    t.scale = 4096;
    LUT_CACHE.set(qmax, t);
  }
  return t;
}

// --------------------------------------------------------------------------------------
// Scratch buffers (reused between steps)
// --------------------------------------------------------------------------------------
const S = { cap: 0, pix: 0, list: new Int32Array(1 << 16) };
function scratch(n, W, H) {
  if (n > S.cap) {
    const c = Math.max(n, S.cap * 2, 256);
    for (const k of ["px", "py", "cs", "sn", "isx", "isy", "cA", "cB", "cC", "op"]) S[k] = new Float64Array(c);
    for (const k of ["bx0", "bx1", "by0", "by1"]) S[k] = new Int32Array(c);
    S.cap = c;
  }
  if (W * H !== S.pix) {
    S.img = new Float32Array(3 * W * H);
    S.dimg = new Float32Array(3 * W * H);
    S.Tfin = new Float32Array(W * H);   // transmittance after the last contributing Gaussian
    S.Tb = new Float32Array(W * H);     // running transmittance in the backward pass
    S.acc = new Float32Array(3 * W * H); // colour accumulated behind, backward pass
    S.last = new Int32Array(W * H);     // list position after the last contributing Gaussian
    S.done = new Uint8Array(W * H);     // early ray termination flag
    S.pix = W * H;
  }
}

// --------------------------------------------------------------------------------------
// Forward (+ optional backward) pass
// --------------------------------------------------------------------------------------
// Loop order is tile -> Gaussian (in index order) -> pixels of the tile inside the Gaussian's
// bounding box. Every pixel still sees its Gaussians in index (= depth) order, so this is the
// same as the per-pixel front-to-back loop of 3DGS, but each Gaussian's parameters stay in
// registers and pixels far outside it are never visited.
//
// Returns { img (Float32Array 3WH, the rendered image), loss, mse }.
// If target is given, gradients are written to model.g.
export function renderAndBackprop(model, W, H, opts, target) {
  const n = model.n, P = model.p;
  scratch(n, W, H);
  const { px, py, cs, sn, isx, isy, cA, cB, cC, op, bx0, bx1, by0, by1 } = S;
  const alphaMode = opts.blend === "alpha";
  const bg = opts.background, bgR = bg[0], bgG = bg[1], bgB = bg[2];
  // thresholds can be overridden (the gradient check disables them to get a smooth function)
  const CUTOFF = opts.cutoff ?? DEFAULT_CUTOFF, CUT2 = CUTOFF * CUTOFF;
  const ALPHA_MIN = opts.alphaMin ?? DEFAULT_ALPHA_MIN, T_MIN = opts.tMin ?? DEFAULT_T_MIN;
  const TX = Math.ceil(W / TILE), TY = Math.ceil(H / TILE), NT = TX * TY;
  const LUT = expTable(CUT2), LUT_SCALE = LUT.scale;

  // 1. per-Gaussian setup (pixel units), pixel bounding boxes and tile counts
  const tileCount = new Int32Array(NT + 1);
  for (let i = 0; i < n; i++) {
    const o = i * STRIDE;
    const x = P[o] * W - 0.5, y = P[o + 1] * W - 0.5;          // pixel centres sit at integer coords
    const sx = Math.exp(P[o + 2]) * W, sy = Math.exp(P[o + 3]) * W;
    const c = Math.cos(P[o + 4]), s = Math.sin(P[o + 4]);
    const ix = 1 / (sx * sx), iy = 1 / (sy * sy);
    px[i] = x; py[i] = y; cs[i] = c; sn[i] = s; isx[i] = ix; isy[i] = iy;
    // conic: q = A dx^2 + 2 B dx dy + C dy^2
    cA[i] = c * c * ix + s * s * iy; cB[i] = c * s * (ix - iy); cC[i] = s * s * ix + c * c * iy;
    op[i] = sigmoid(P[o + 8]);
    // axis-aligned extent of the cutoff ellipse
    const ex = CUTOFF * Math.sqrt(c * c * sx * sx + s * s * sy * sy);
    const ey = CUTOFF * Math.sqrt(s * s * sx * sx + c * c * sy * sy);
    const x0 = Math.max(0, Math.ceil(x - ex)), x1 = Math.min(W - 1, Math.floor(x + ex));
    const y0 = Math.max(0, Math.ceil(y - ey)), y1 = Math.min(H - 1, Math.floor(y + ey));
    if (!(x0 <= x1 && y0 <= y1)) { bx0[i] = 1; bx1[i] = 0; continue; }   // off-screen / degenerate
    bx0[i] = x0; bx1[i] = x1; by0[i] = y0; by1[i] = y1;
    for (let ty = (y0 / TILE) | 0; ty <= ((y1 / TILE) | 0); ty++)
      for (let tx = (x0 / TILE) | 0; tx <= ((x1 / TILE) | 0); tx++) tileCount[ty * TX + tx + 1]++;
  }
  // 2. bin Gaussians into tiles (keeps index order = depth order for alpha compositing)
  for (let t = 0; t < NT; t++) tileCount[t + 1] += tileCount[t];
  const total = tileCount[NT];
  if (S.list.length < total) S.list = new Int32Array(total * 2);
  const list = S.list, cursor = tileCount.slice(0, NT);
  for (let i = 0; i < n; i++) {
    if (bx0[i] > bx1[i]) continue;
    for (let ty = (by0[i] / TILE) | 0; ty <= ((by1[i] / TILE) | 0); ty++)
      for (let tx = (bx0[i] / TILE) | 0; tx <= ((bx1[i] / TILE) | 0); tx++) list[cursor[ty * TX + tx]++] = i;
  }

  // 3. forward
  const img = S.img, Tfin = S.Tfin, last = S.last, done = S.done;
  img.fill(0);
  if (alphaMode) { Tfin.fill(1); done.fill(0); }
  for (let ty = 0; ty < TY; ty++) for (let tx = 0; tx < TX; tx++) {
    const t = ty * TX + tx, l0 = tileCount[t], l1 = tileCount[t + 1];
    const X0 = tx * TILE, Y0 = ty * TILE, X1 = Math.min(W, X0 + TILE) - 1, Y1 = Math.min(H, Y0 + TILE) - 1;
    if (alphaMode) for (let y = Y0; y <= Y1; y++) for (let x = X0; x <= X1; x++) last[y * W + x] = l0;
    for (let k = l0; k < l1; k++) {
      const i = list[k], o = i * STRIDE;
      const gx = px[i], gy = py[i], A = cA[i], B2 = 2 * cB[i], C = cC[i], cr = P[o + 5], cg = P[o + 6], cb = P[o + 7], oi = op[i];
      const xa = Math.max(X0, bx0[i]), xb = Math.min(X1, bx1[i]), ya = Math.max(Y0, by0[i]), yb = Math.min(Y1, by1[i]);
      for (let y = ya; y <= yb; y++) {
        const dy = y - gy, Cdy2 = C * dy * dy, Bdy = B2 * dy;
        for (let x = xa, p = y * W + xa; x <= xb; x++, p++) {
          const dx = x - gx, q = A * dx * dx + Bdy * dx + Cdy2;
          if (q > CUT2) continue;
          const f = q * LUT_SCALE, j = f | 0, g = LUT[j] + (f - j) * (LUT[j + 1] - LUT[j]);
          if (alphaMode) {
            if (done[p]) continue;
            const a = Math.min(ALPHA_MAX, oi * g);
            if (a < ALPHA_MIN) continue;
            const T = Tfin[p], Tn = T * (1 - a);
            if (Tn < T_MIN) { done[p] = 1; continue; }
            const w = a * T;
            img[3 * p] += w * cr; img[3 * p + 1] += w * cg; img[3 * p + 2] += w * cb;
            Tfin[p] = Tn; last[p] = k + 1;
          } else {
            img[3 * p] += g * cr; img[3 * p + 1] += g * cg; img[3 * p + 2] += g * cb;
          }
        }
      }
    }
  }
  if (alphaMode) for (let p = 0; p < W * H; p++) {
    const T = Tfin[p];
    img[3 * p] += T * bgR; img[3 * p + 1] += T * bgG; img[3 * p + 2] += T * bgB;
  }
  if (!target) return { img, loss: 0, mse: 0 };

  // 4. loss and dL/dI
  const M = 3 * W * H, dimg = S.dimg, lam = opts.loss === "l1" ? 0 : opts.loss === "l2" ? 1 : opts.l2Weight;
  let loss = 0, mse = 0;
  for (let j = 0; j < M; j++) {
    const r = img[j] - target[j];
    const rc = Math.min(1, Math.max(0, img[j])) - target[j];
    mse += rc * rc;
    loss += (1 - lam) * Math.abs(r) + lam * r * r;
    dimg[j] = ((1 - lam) * Math.sign(r) + lam * 2 * r) / M;
  }
  loss /= M; mse /= M;

  // 5. backward (alpha mode walks each tile's list back to front)
  const gr = model.g, Tb = S.Tb, acc = S.acc;
  gr.fill(0, 0, n * STRIDE);
  if (alphaMode) for (let p = 0; p < W * H; p++) {
    const T = Tfin[p];
    Tb[p] = T; acc[3 * p] = T * bgR; acc[3 * p + 1] = T * bgG; acc[3 * p + 2] = T * bgB;
  }
  for (let ty = 0; ty < TY; ty++) for (let tx = 0; tx < TX; tx++) {
    const t = ty * TX + tx, l0 = tileCount[t], l1 = tileCount[t + 1];
    const X0 = tx * TILE, Y0 = ty * TILE, X1 = Math.min(W, X0 + TILE) - 1, Y1 = Math.min(H, Y0 + TILE) - 1;
    for (let kk = 0; kk < l1 - l0; kk++) {
      const k = alphaMode ? l1 - 1 - kk : l0 + kk;
      const i = list[k], o = i * STRIDE;
      const gx = px[i], gy = py[i], A = cA[i], B2 = 2 * cB[i], C = cC[i], cr = P[o + 5], cg = P[o + 6], cb = P[o + 7], oi = op[i];
      const c = cs[i], s = sn[i], ix = isx[i], iy = isy[i];
      const xa = Math.max(X0, bx0[i]), xb = Math.min(X1, bx1[i]), ya = Math.max(Y0, by0[i]), yb = Math.min(Y1, by1[i]);
      // gradient accumulators for this Gaussian within this tile
      let gmx = 0, gmy = 0, glx = 0, gly = 0, gth = 0, gR = 0, gG = 0, gB = 0, gop = 0;
      for (let y = ya; y <= yb; y++) {
        const dy = y - gy, Cdy2 = C * dy * dy, Bdy = B2 * dy;
        for (let x = xa, p = y * W + xa; x <= xb; x++, p++) {
          const dx = x - gx, q = A * dx * dx + Bdy * dx + Cdy2;
          if (q > CUT2) continue;
          const f = q * LUT_SCALE, j = f | 0, g = LUT[j] + (f - j) * (LUT[j + 1] - LUT[j]);
          const dR = dimg[3 * p], dG = dimg[3 * p + 1], dB = dimg[3 * p + 2];
          let dLdG; // dL/dG
          if (alphaMode) {
            if (k >= last[p]) continue;                 // behind the termination point
            const oa = oi * g, a = Math.min(ALPHA_MAX, oa);
            if (a < ALPHA_MIN) continue;
            const inv = 1 / (1 - a), T = Tb[p] * inv;   // transmittance in front of Gaussian i
            Tb[p] = T;
            const w = a * T;
            gR += dR * w; gG += dG * w; gB += dB * w;
            // dC/da = T_i c_i - (colour behind) / (1 - a_i)
            const dLda = dR * (T * cr - acc[3 * p] * inv) + dG * (T * cg - acc[3 * p + 1] * inv) + dB * (T * cb - acc[3 * p + 2] * inv);
            acc[3 * p] += w * cr; acc[3 * p + 1] += w * cg; acc[3 * p + 2] += w * cb;
            if (oa >= ALPHA_MAX) continue;              // clamped: no gradient through G or o
            gop += dLda * g;
            dLdG = dLda * oi;
          } else {
            gR += dR * g; gG += dG * g; gB += dB * g;
            dLdG = dR * cr + dG * cg + dB * cb;
          }
          // chain rule: q = (u1/sx)^2 + (u2/sy)^2,  u1 = c dx + s dy,  u2 = -s dx + c dy,  dG/dq = -G/2
          const dq = -0.5 * dLdG * g;
          const u1 = c * dx + s * dy, u2 = -s * dx + c * dy;
          const q1 = 2 * u1 * ix, q2 = 2 * u2 * iy;     // dq/du1, dq/du2
          gmx += dq * (-c * q1 + s * q2);
          gmy += dq * (-s * q1 - c * q2);
          glx += dq * (-2 * u1 * u1 * ix);
          gly += dq * (-2 * u2 * u2 * iy);
          gth += dq * (2 * u1 * u2 * (ix - iy));
        }
      }
      gr[o] += gmx * W; gr[o + 1] += gmy * W;          // pixel units -> normalised mean
      gr[o + 2] += glx; gr[o + 3] += gly; gr[o + 4] += gth;
      gr[o + 5] += gR; gr[o + 6] += gG; gr[o + 7] += gB;
      gr[o + 8] += gop * oi * (1 - oi);
    }
  }
  return { img, loss, mse };
}

// --------------------------------------------------------------------------------------
// Adam
// --------------------------------------------------------------------------------------
export function adamStep(model, opts, W) {
  const { p, m, v, g, t } = model;
  const lr = GROUP_OF.map((gi) => (opts.learn[gi] ? opts.lr[gi] * opts.lrScale : 0));
  if (opts.blend !== "alpha") lr[8] = 0;
  const lsMin = Math.log(0.3 / W), lsMax = Math.log(0.5);
  for (let i = 0; i < model.n; i++) {
    const ti = ++t[i];
    const bc1 = 1 - Math.pow(B1, ti), bc2 = 1 - Math.pow(B2, ti);
    const o = i * STRIDE;
    model.gacc[i] += Math.hypot(g[o], g[o + 1]);
    model.gcnt[i]++;
    for (let k = 0; k < STRIDE; k++) {
      if (lr[k] === 0) continue;
      const j = o + k, gj = g[j];
      m[j] = B1 * m[j] + (1 - B1) * gj;
      v[j] = B2 * v[j] + (1 - B2) * gj * gj;
      p[j] -= (lr[k] * (m[j] / bc1)) / (Math.sqrt(v[j] / bc2) + EPS);
    }
    p[o + 2] = Math.min(lsMax, Math.max(lsMin, p[o + 2]));
    p[o + 3] = Math.min(lsMax, Math.max(lsMin, p[o + 3]));
    p[o + 8] = Math.min(8, Math.max(-8, p[o + 8]));
  }
}

// --------------------------------------------------------------------------------------
// Adaptive density control (as in 3DGS): clone / split Gaussians with large positional
// gradients, prune transparent / dark ones.
// --------------------------------------------------------------------------------------
export function densify(model, opts, aspect) {
  const n = model.n, P = model.p;
  const alphaMode = opts.blend === "alpha";
  let cloned = 0, split = 0, pruned = 0;
  const budget = Math.max(0, Math.min(opts.maxN - n, Math.ceil(n * opts.growFrac)));
  if (budget > 0) {
    const idx = [];
    for (let i = 0; i < n; i++) if (model.gcnt[i] > 0) idx.push(i);
    const avg = (i) => model.gacc[i] / model.gcnt[i];
    idx.sort((a, b) => avg(b) - avg(a));
    for (const i of idx.slice(0, budget)) {
      const o = i * STRIDE, vals = model.get(i);
      const sx = Math.exp(vals[2]), sy = Math.exp(vals[3]);
      if (Math.max(sx, sy) > opts.splitScale) {
        // split: two children sampled from the parent's distribution, 1.6x smaller
        const c = Math.cos(vals[4]), s = Math.sin(vals[4]);
        const child = () => {
          const zx = gauss() * sx, zy = gauss() * sy, ch = vals.slice();
          ch[0] += c * zx - s * zy; ch[1] += s * zx + c * zy;
          ch[2] -= Math.log(1.6); ch[3] -= Math.log(1.6);
          return ch;
        };
        const a = child(), b = child();
        for (let k = 0; k < STRIDE; k++) { P[o + k] = a[k]; model.m[o + k] = 0; model.v[o + k] = 0; }
        model.t[i] = 0;
        model.push(b);
        split++;
      } else {
        // clone: an identical copy. In the additive model two copies would double the
        // brightness, so the colour is shared between them.
        if (!alphaMode) { for (const k of [5, 6, 7]) { vals[k] *= 0.5; P[o + k] *= 0.5; } }
        model.push(vals);
        cloned++;
      }
    }
  }
  // prune
  const keep = new Uint8Array(model.n);
  for (let i = 0; i < model.n; i++) {
    const o = i * STRIDE;
    let ok = P[o] > -0.1 && P[o] < 1.1 && P[o + 1] > -0.1 && P[o + 1] < aspect + 0.1;
    if (alphaMode) ok = ok && sigmoid(P[o + 8]) > opts.pruneThreshold;
    else ok = ok && Math.max(Math.abs(P[o + 5]), Math.abs(P[o + 6]), Math.abs(P[o + 7])) > opts.pruneThreshold;
    keep[i] = ok && model.n - pruned > 1 ? 1 : (pruned++, 0);
  }
  if (pruned) model.compact(keep);
  model.gacc.fill(0, 0, model.n);
  model.gcnt.fill(0, 0, model.n);
  return { cloned, split, pruned };
}

// --------------------------------------------------------------------------------------
// Finite-difference gradient check (used by gradcheck.html)
// --------------------------------------------------------------------------------------
export function gradCheck(blend, loss = "l2") {
  const W = 24, H = 20, model = new Model(8);
  const opts = { blend, loss, l2Weight: 0.5, background: [0.2, 0.3, 0.4], cutoff: 12, alphaMin: 0, tMin: 0 };
  const target = new Float32Array(3 * W * H).map(() => Math.random());
  for (let i = 0; i < 5; i++)
    model.push([0.2 + 0.6 * Math.random(), (0.2 + 0.6 * Math.random()) * (H / W), Math.log(0.08 + 0.1 * Math.random()),
      Math.log(0.05 + 0.1 * Math.random()), Math.random() * 3, Math.random(), Math.random(), Math.random(), Math.random() * 2 - 1]);
  renderAndBackprop(model, W, H, opts, target);
  const analytic = Array.from(model.g.subarray(0, model.n * STRIDE));
  const rows = [];
  for (let i = 0; i < model.n; i++) for (let k = 0; k < STRIDE; k++) {
    if (blend !== "alpha" && k === 8) continue;
    const j = i * STRIDE + k, h = 1e-3, v0 = model.p[j];
    model.p[j] = v0 + h; const lp = renderAndBackprop(model, W, H, opts, target).loss;
    model.p[j] = v0 - h; const lm = renderAndBackprop(model, W, H, opts, target).loss;
    model.p[j] = v0;
    rows.push({ i, k, analytic: analytic[j], numeric: (lp - lm) / (2 * h) });
  }
  return rows;
}
