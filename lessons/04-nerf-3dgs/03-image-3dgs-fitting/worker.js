// Training loop. Runs in a Web Worker so the UI stays responsive.
import { Model, initModel, renderAndBackprop, adamStep, densify, STRIDE } from "./engine.js";

let model = new Model(1024);
let target = null, W = 0, H = 0;
let opts = null;
let playing = false, iter = 0, last = null;
let wantImage = true;
const history = []; // [iter, psnr, n]

function lrFactor() {
  // exponential decay to opts.decayTo over opts.decayIters iterations (then constant)
  if (!opts.decay) return 1;
  return Math.pow(opts.decayTo, Math.min(1, iter / opts.decayIters));
}

function step() {
  const o = { ...opts, lrScale: opts.lrScale * lrFactor() };
  last = renderAndBackprop(model, W, H, o, target);
  adamStep(model, o, W);
  iter++;
  let event = null;
  if (opts.densify && iter % opts.densifyEvery === 0 && iter <= opts.densifyUntil) {
    event = densify(model, opts, H / W);
    event.iter = iter;
  }
  if (iter % 5 === 0 || iter < 20) history.push([iter, last.mse > 0 ? -10 * Math.log10(last.mse) : 99, model.n]);
  if (history.length > 4000) { // keep every other point
    for (let i = 0; i < history.length / 2; i++) history[i] = history[2 * i];
    history.length = Math.floor(history.length / 2);
  }
  return event;
}

function post(itps, events) {
  const msg = {
    type: "state", iter, n: model.n, W, H, itps, events, playing,
    loss: last ? last.loss : 0,
    psnr: last && last.mse > 0 ? -10 * Math.log10(last.mse) : 0,
    lr: opts.lrScale * lrFactor(),
    params: model.p.slice(0, model.n * STRIDE),
    history: history.slice(),
  };
  const transfer = [msg.params.buffer];
  if (wantImage) {
    // re-render current parameters (last.img is from before the Adam step)
    const { img } = renderAndBackprop(model, W, H, opts, null);
    const rgba = new Uint8ClampedArray(W * H * 4);
    for (let p = 0; p < W * H; p++) {
      rgba[4 * p] = img[3 * p] * 255; rgba[4 * p + 1] = img[3 * p + 1] * 255; rgba[4 * p + 2] = img[3 * p + 2] * 255; rgba[4 * p + 3] = 255;
    }
    msg.image = rgba;
    transfer.push(rgba.buffer);
  }
  postMessage(msg, transfer);
}

function loop() {
  if (!playing) return;
  const t0 = performance.now();
  let k = 0;
  const events = [];
  // run as many steps as fit in ~30 ms, then report
  do {
    const ev = step();
    if (ev) events.push(ev);
    k++;
    if (opts.maxIters && iter >= opts.maxIters) { playing = false; break; }
  } while (performance.now() - t0 < 30);
  const dt = (performance.now() - t0) / 1000;
  post(k / dt, events);
  setTimeout(loop, 0);
}

function reset() {
  initModel(model, opts.n, target, W, H, opts);
  iter = 0;
  history.length = 0;
  last = renderAndBackprop(model, W, H, opts, target);
  history.push([0, -10 * Math.log10(last.mse), model.n]);
}

onmessage = (e) => {
  const m = e.data;
  switch (m.cmd) {
    case "target": // new target image (also used for live webcam frames and resolution changes)
      target = m.target; W = m.W; H = m.H;
      if (m.reset || model.n === 0) reset();
      if (!playing) { last = renderAndBackprop(model, W, H, opts, target); post(0, []); }
      break;
    case "opts":
      opts = { ...opts, ...m.opts };
      if (!playing && target) post(0, []);
      break;
    case "reset":
      if (m.opts) opts = { ...opts, ...m.opts };
      reset();
      post(0, []);
      break;
    case "play":
      if (!playing) { playing = true; loop(); }
      break;
    case "pause":
      playing = false;
      post(0, []);
      break;
    case "step": {
      playing = false;
      const events = [];
      for (let i = 0; i < (m.count || 1); i++) { const ev = step(); if (ev) events.push(ev); }
      post(0, events);
      break;
    }
    case "wantImage":
      wantImage = m.value;
      break;
    case "load": // import Gaussians from JSON
      model.n = 0;
      model.grow(m.params.length / STRIDE);
      for (let i = 0; i < m.params.length / STRIDE; i++) model.push(Array.from(m.params.slice(i * STRIDE, (i + 1) * STRIDE)));
      iter = 0; history.length = 0;
      last = renderAndBackprop(model, W, H, opts, target);
      post(0, []);
      break;
  }
};
