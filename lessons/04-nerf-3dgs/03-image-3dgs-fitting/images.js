// Procedural sample targets (no image files needed). Each draws onto a 2D canvas.

function synthetic(ctx, w, h) {
  // like Part 1 of the reference notebook: the target itself is a sum of a few Gaussians
  const img = ctx.createImageData(w, h);
  const gs = [
    { c: [0.7, 0.1, 0.2], x: 0.25, y: 0.35, sx: 0.18, sy: 0.12, t: 0.2 },
    { c: [0.3, 0.8, 0.1], x: 0.62, y: 0.28, sx: 0.14, sy: 0.05, t: 0.8 },
    { c: [0.1, 0.3, 0.9], x: 0.55, y: 0.68, sx: 0.2, sy: 0.08, t: -0.5 },
    { c: [0.9, 0.7, 0.1], x: 0.3, y: 0.75, sx: 0.07, sy: 0.07, t: 0 },
    { c: [0.6, 0.2, 0.8], x: 0.82, y: 0.6, sx: 0.04, sy: 0.15, t: 0.3 },
  ];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const rgb = [0, 0, 0];
    for (const g of gs) {
      const dx = x / w - g.x, dy = y / h - g.y, c = Math.cos(g.t), s = Math.sin(g.t);
      const u1 = c * dx + s * dy, u2 = -s * dx + c * dy;
      const G = Math.exp(-0.5 * ((u1 / g.sx) ** 2 + (u2 / g.sy) ** 2));
      for (let k = 0; k < 3; k++) rgb[k] += g.c[k] * G;
    }
    const o = 4 * (y * w + x);
    img.data[o] = rgb[0] * 255; img.data[o + 1] = rgb[1] * 255; img.data[o + 2] = rgb[2] * 255; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function sunset(ctx, w, h) {
  const sky = ctx.createLinearGradient(0, 0, 0, h * 0.65);
  sky.addColorStop(0, "#1b1f4b"); sky.addColorStop(0.45, "#a23c6b"); sky.addColorStop(0.8, "#f39a4a"); sky.addColorStop(1, "#ffd27a");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
  const sun = ctx.createRadialGradient(w * 0.62, h * 0.6, 0, w * 0.62, h * 0.6, w * 0.16);
  sun.addColorStop(0, "#fff6d0"); sun.addColorStop(0.45, "#ffe08a"); sun.addColorStop(1, "rgba(255,200,100,0)");
  ctx.fillStyle = sun; ctx.beginPath(); ctx.arc(w * 0.62, h * 0.6, w * 0.16, 0, Math.PI * 2); ctx.fill();
  const hill = (base, amp, freq, phase, color) => {
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 2) ctx.lineTo(x, h * base - Math.sin(x / w * freq + phase) * h * amp - Math.sin(x / w * freq * 2.7 + phase * 2) * h * amp * 0.4);
    ctx.lineTo(w, h); ctx.fill();
  };
  hill(0.68, 0.05, 5, 1, "#5b2a52"); hill(0.78, 0.06, 3.4, 2.2, "#341a3d"); hill(0.9, 0.04, 6, 0.4, "#1a0f24");
  ctx.fillStyle = "#120a18";
  for (let i = 0; i < 7; i++) { // pine trees
    const x = w * (0.08 + i * 0.13 + (i % 2) * 0.03), base = h * 0.93, th = h * (0.16 + (i % 3) * 0.05);
    ctx.beginPath(); ctx.moveTo(x, base - th); ctx.lineTo(x - th * 0.22, base); ctx.lineTo(x + th * 0.22, base); ctx.fill();
  }
}

function shapes(ctx, w, h) {
  ctx.fillStyle = "#f4efe6"; ctx.fillRect(0, 0, w, h);
  const s = w / 256;
  ctx.fillStyle = "#e63946"; ctx.beginPath(); ctx.arc(70 * s, 75 * s, 45 * s, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#1d3557"; ctx.fillRect(130 * s, 30 * s, 95 * s, 95 * s);
  ctx.fillStyle = "#2a9d8f"; ctx.beginPath(); ctx.moveTo(40 * s, 230 * s); ctx.lineTo(110 * s, 140 * s); ctx.lineTo(180 * s, 230 * s); ctx.fill();
  ctx.fillStyle = "#f4a261";
  for (let i = 0; i < 6; i++) ctx.fillRect((150 + i * 16) * s, 145 * s, 8 * s, 90 * s);
  ctx.strokeStyle = "#264653"; ctx.lineWidth = 3 * s;
  ctx.beginPath(); ctx.moveTo(10 * s, 128 * s); ctx.bezierCurveTo(80 * s, 100 * s, 120 * s, 160 * s, 246 * s, 132 * s); ctx.stroke();
  ctx.fillStyle = "#111"; ctx.font = `bold ${30 * s}px system-ui, sans-serif`; ctx.fillText("2DGS", 148 * s, 92 * s);
}

function rings(ctx, w, h) {
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = x / w - 0.5, v = y / h - 0.5, r = Math.hypot(u, v), a = Math.atan2(v, u);
    const o = 4 * (y * w + x);
    img.data[o] = 128 + 127 * Math.sin(r * 40);
    img.data[o + 1] = 128 + 127 * Math.sin(a * 3 + r * 10);
    img.data[o + 2] = 128 + 127 * Math.cos(r * 18 - a);
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export const SAMPLES = {
  sunset: { name: "Sunset (procedural)", draw: sunset, aspect: 0.75 },
  shapes: { name: "Shapes & text", draw: shapes, aspect: 1 },
  synthetic: { name: "Synthetic Gaussians (notebook part 1)", draw: synthetic, aspect: 1 },
  rings: { name: "Rings (high frequency)", draw: rings, aspect: 1 },
};
