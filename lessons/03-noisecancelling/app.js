import { TAU, TONE_KHZ, TONE_PERIOD, TONE_AMP, NOISE, wantedWave, noiseUnit, forecastUnit, toneRatio, breakEvenDegrees, mainSample } from './signal-model.js';

(() => {
  /* =====================================================================
     Constants
     Time is measured in milliseconds, frequency in kHz (so cycles = kHz * ms).
     ===================================================================== */
  const WINDOW_MS = 25;            // time span visible across every waveform
  const RANGE = 2;                 // vertical half-range, shared so volumes compare honestly
  const SCROLL_MS_PER_S = 9;       // how fast the waves drift to the right
  const STEP = 0.05;               // ms between samples in the analysis tables
  const MAX_DELAY = 8.4;           // ms, top of the delay slider (about one hum cycle)
  const MARK_X = [0.1, 0.26, 0.42, 0.58, 0.74, 0.9];  // x positions of the peak/dip markers

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* =====================================================================
     State
     ===================================================================== */
  const DEFAULTS = { anc: true, type: 'hum', noiseVol: 0.7, wantedVol: 0.6, strength: 1, delay: 0.1 };
  const state = {
    ...DEFAULTS,
    tonePhase: 0,                  // degrees of lateness in the timing experiment
    paused: reducedMotion,
    stage: 0,
    tour: false
  };

  /* Signals as functions of time t (ms). Everything on screen is drawn from these. */
  const sig = {
    wanted: t => state.wantedVol * wantedWave(t),
    noise: t => state.noiseVol * noiseUnit(state.type, t),
    antiIdeal: t => -state.noiseVol * noiseUnit(state.type, t),
    anti: t => state.anc ? -state.strength * state.noiseVol * forecastUnit(state.type, t - state.delay) : 0,
    mic: t => sig.wanted(t) + sig.noise(t),
    out: t => mainSample(state, t),
    residual: t => sig.noise(t) + sig.anti(t)
  };

  const toneDelay = () => state.tonePhase / 360 * TONE_PERIOD;
  const tone = {
    noise: t => TONE_AMP * Math.sin(TAU * TONE_KHZ * t),
    ideal: t => -state.strength * TONE_AMP * Math.sin(TAU * TONE_KHZ * t),
    anti: t => -state.strength * TONE_AMP * Math.sin(TAU * TONE_KHZ * (t - toneDelay())),
    result: t => tone.noise(t) + tone.anti(t)
  };
  // Leftover amplitude when a sine is cancelled by a copy that is `deg` late:  |1 - s*e^(-j*phi)|

  /* =====================================================================
     Analysis: how much noise is left? (computed once per settings change)
     Each noise type is sampled into a table so the delay curve is cheap.
     ===================================================================== */
  const tables = {};
  function table(type) {
    if (tables[type]) return tables[type];
    const pad = Math.ceil(MAX_DELAY / STEP) + 2, len = 2400;
    const N = new Float32Array(pad + len), Q = new Float32Array(pad + len);
    let energy = 0;
    for (let i = 0; i < pad + len; i++) {
      const t = (i - pad) * STEP;
      N[i] = noiseUnit(type, t);
      Q[i] = forecastUnit(type, t);
      if (i >= pad) energy += N[i] * N[i];
    }
    return (tables[type] = { N, Q, pad, len, energy });
  }

  // RMS of (noise - strength * delayed forecast) relative to the RMS of the noise
  function residualRatio(tb, strength, k) {
    let sum = 0;
    for (let i = tb.pad; i < tb.pad + tb.len; i++) {
      const r = tb.N[i] - strength * tb.Q[i - k];
      sum += r * r;
    }
    return Math.sqrt(sum / tb.energy);
  }

  let A = {};                       // latest analysis
  let curveKey = '', curve = [];
  function analyse() {
    const tb = table(state.type);
    const key = state.type + state.strength;
    if (key !== curveKey) {         // delay curve only depends on type and strength
      curveKey = key;
      curve = [];
      for (let k = 0; k <= Math.round(MAX_DELAY / STEP); k += 2) curve.push(residualRatio(tb, state.strength, k));
    }
    const ratio = residualRatio(tb, state.strength, Math.round(state.delay / STEP));
    const best = residualRatio(tb, 1, 0);
    A = {
      ratio,                                              // 1 = no change, 0 = silence, >1 = worse
      cut: 1 - ratio,
      db: 20 * Math.log10(Math.max(ratio, 0.001)),
      pred: 1 - best * best,                              // share of the noise power that is predictable
      curve,
      spread: Math.max(...curve) - Math.min(...curve)
    };
  }

  const pct = v => Math.round(v * 100);
  function grade(cut) {
    if (cut >= 0.9) return { word: 'Excellent', level: 'good' };
    if (cut >= 0.6) return { word: 'Strong', level: 'good' };
    if (cut >= 0.3) return { word: 'Partial', level: 'warn' };
    if (cut >= 0.05) return { word: 'Weak', level: 'warn' };
    if (cut >= -0.05) return { word: 'No effect', level: 'neutral' };
    return { word: 'Worse', level: 'bad' };
  }

  /* =====================================================================
     Canvas plumbing
     ===================================================================== */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  let COL = {};
  let dirty = true;
  let scroll = 0;

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = n => cs.getPropertyValue(n).trim();
    COL = {
      wanted: g('--wanted'), noise: g('--noise'), anti: g('--anti'), mic: g('--mic'),
      out: g('--out'), proc: g('--proc'), grid: g('--grid'), ghost: g('--ghost'), muted: g('--muted'), ink: g('--ink'),
      good: g('--good'), bad: g('--bad')
    };
    dirty = true;
  }

  const canvases = $$('canvas[data-cv]').map(el => {
    const c = { name: el.dataset.cv, el, ctx: el.getContext('2d'), w: 0, h: 0, visible: true };
    new ResizeObserver(() => fit(c)).observe(el);
    new IntersectionObserver(([e]) => { c.visible = e.isIntersecting; if (c.visible) dirty = true; }).observe(el);
    return c;
  });

  function fit(c) {
    const dpr = window.devicePixelRatio || 1;
    c.w = c.el.clientWidth;
    c.h = c.el.clientHeight;
    c.el.width = Math.round(c.w * dpr);
    c.el.height = Math.round(c.h * dpr);
    c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    dirty = true;
  }

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function backdrop(c, range, guides) {
    const { ctx, w, h } = c;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = COL.grid;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    if (guides) {
      ctx.setLineDash([2, 5]);
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      for (const s of [-1, 1]) { const y = h / 2 - s * (h / 2 / range); ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
  }

  function stroke(c, fn, layer, range) {
    const { ctx, w, h } = c;
    const mid = h / 2, k = mid / range;
    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const y = mid - clamp(fn(x / w * WINDOW_MS - scroll), -range, range) * k;
      x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = layer.width || 2.2;
    ctx.globalAlpha = layer.alpha == null ? 1 : layer.alpha;
    ctx.setLineDash(layer.dash || []);
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }

  // Dotted stems from the zero line to the wave: they make peaks vs dips easy to compare
  function stems(c, fn, color) {
    const { ctx, w, h } = c;
    const mid = h / 2, k = mid / RANGE;
    for (const f of MARK_X) {
      const x = f * w;
      const y = mid - clamp(fn(f * WINDOW_MS - scroll), -RANGE, RANGE) * k;
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x, y); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, TAU); ctx.fill();
    }
  }

  function caption(c, text) {
    const { ctx, w, h } = c;
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const tw = ctx.measureText(text).width + 20;
    ctx.fillStyle = COL.grid; ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.roundRect(w / 2 - tw / 2, h / 2 - 13, tw, 26, 13); ctx.fill();
    ctx.globalAlpha = 1; ctx.fillStyle = COL.ink;
    ctx.fillText(text, w / 2, h / 2 + 1);
  }

  function lane(c, layers, opt = {}) {
    const range = opt.range || RANGE;
    backdrop(c, range, !opt.thumb);
    for (const l of layers) stroke(c, l.fn, l, range);
    if (opt.stems) for (const s of opt.stems) stems(c, s.fn, s.color);
    if (opt.caption) caption(c, opt.caption);
  }

  /* ---- Line chart used by the timing experiment ---- */
  function chart(c, { ys, xmax, marker, ticks }) {
    const { ctx, w, h } = c;
    const L = 34, R = 20, T = 12, B = 24, yMax = 2.6;
    const pw = w - L - R, ph = h - T - B;
    const X = v => L + v / xmax * pw;
    const Y = v => T + ph - Math.min(v, yMax) / yMax * ph;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '12px system-ui, sans-serif';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'right';

    for (const v of [0, 1, 2]) {
      ctx.strokeStyle = COL.grid; ctx.lineWidth = v === 1 ? 1.5 : 1;
      ctx.setLineDash(v === 1 ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + pw, Y(v)); ctx.stroke();
      ctx.fillStyle = COL.muted; ctx.fillText(v + '×', L - 6, Y(v));
    }
    ctx.setLineDash([]);

    const pts = ys.map((v, i) => [X(i / (ys.length - 1) * xmax), Y(v)]);
    const fill = (clipTop, clipBottom, color) => {   // shade between the curve and the 1× baseline
      ctx.save();
      ctx.beginPath(); ctx.rect(L, clipTop, pw, clipBottom - clipTop); ctx.clip();
      ctx.beginPath(); ctx.moveTo(pts[0][0], Y(1));
      for (const p of pts) ctx.lineTo(p[0], p[1]);
      ctx.lineTo(pts[pts.length - 1][0], Y(1)); ctx.closePath();
      ctx.fillStyle = color; ctx.globalAlpha = 0.28; ctx.fill();
      ctx.restore();
    };
    fill(Y(1), T + ph, COL.good);
    fill(T, Y(1), COL.bad);

    ctx.beginPath();
    pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.strokeStyle = COL.out; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.stroke();

    ctx.textAlign = 'center'; ctx.fillStyle = COL.muted;
    for (const [v, label] of ticks) ctx.fillText(label, X(v), h - 8);

    // current setting
    const mx = X(marker);
    const idx = clamp(Math.round(marker / xmax * (ys.length - 1)), 0, ys.length - 1);
    ctx.strokeStyle = COL.ink; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.6; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, T); ctx.lineTo(mx, T + ph); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = COL.out; ctx.strokeStyle = COL.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(mx, Y(ys[idx]), 5.5, 0, TAU); ctx.fill(); ctx.stroke();
  }

  /* =====================================================================
     What each canvas draws
     ===================================================================== */
  const paint = {};
  const W = () => ({ color: COL.wanted });
  const N = () => ({ color: COL.noise });

  paint['scene-wanted'] = c => lane(c, [{ fn: sig.wanted, ...W() }]);
  paint['scene-noise'] = c => lane(c, [{ fn: sig.noise, ...N() }]);
  paint['scene-mic'] = c => lane(c, [
    { fn: sig.wanted, color: COL.wanted, alpha: 0.3, width: 1.2 },
    { fn: sig.noise, color: COL.noise, alpha: 0.3, width: 1.2 },
    { fn: sig.mic, color: COL.mic, width: 2.6 }
  ]);

  paint.out = c => lane(c, [
    ...(state.anc ? [{ fn: sig.mic, color: COL.ghost, width: 1.6, alpha: 0.75 }] : []),
    { fn: sig.wanted, color: COL.wanted, width: 1.4, dash: [6, 4], alpha: 0.8 },
    { fn: sig.out, color: COL.out, width: 2.6 }
  ]);

  paint['demo-noise'] = c => lane(c, [{ fn: sig.noise, ...N() }], { stems: [{ fn: sig.noise, color: COL.noise }] });
  paint['demo-anti'] = c => lane(c, [
    { fn: sig.antiIdeal, color: COL.anti, width: 1.6, dash: [6, 4], alpha: 0.55 },
    { fn: sig.anti, color: COL.anti }
  ], { stems: [{ fn: sig.anti, color: COL.anti }], caption: state.anc ? null : 'Noise cancelling is OFF, so no anti-noise is made' });
  paint['demo-result'] = c => lane(c, [
    { fn: sig.noise, color: COL.noise, width: 1.2, alpha: 0.3 },
    { fn: sig.residual, color: COL.out, width: 2.6 }
  ], { stems: [{ fn: sig.residual, color: COL.out }] });

  paint['tone-noise'] = c => lane(c, [{ fn: tone.noise, ...N() }]);
  paint['tone-anti'] = c => lane(c, [
    { fn: tone.ideal, color: COL.anti, width: 1.6, dash: [6, 4], alpha: 0.55 },
    { fn: tone.anti, color: COL.anti }
  ]);
  paint['tone-result'] = c => lane(c, [
    { fn: tone.noise, color: COL.noise, width: 1.2, alpha: 0.3 },
    { fn: tone.result, color: COL.out, width: 2.6 }
  ]);

  paint['chart-tone'] = c => {
    const ys = [];
    for (let d = 0; d <= 360; d += 2) ys.push(toneRatio(state.strength, d));
    chart(c, { ys, xmax: 360, marker: state.tonePhase, ticks: [[0, '0°'], [90, '90°'], [180, '180°'], [270, '270°'], [360, '360°']] });
  };
  paint['chart-noise'] = c =>
    chart(c, { ys: A.curve, xmax: MAX_DELAY, marker: state.delay, ticks: [[0, '0 ms'], [2, '2'], [4, '4'], [6, '6'], [8, '8 ms']] });

  // Pipeline thumbnails: the wave as it looks at each stage
  const thumb = (c, layers, extra) => lane(c, layers, { range: 1.7, thumb: true, ...extra });
  const faint = (fn, color) => ({ fn, color, alpha: 0.3, width: 1.2 });
  const T_NOISE = () => ({ fn: sig.noise, color: COL.noise, width: 1.8 });
  paint['thumb-0'] = c => thumb(c, [T_NOISE()]);
  paint['thumb-1'] = c => {
    thumb(c, [T_NOISE()]);
    // vertical ticks = the mic measuring ("sampling") the wave
    const { ctx, w, h } = c, mid = h / 2, k = mid / 1.7;
    ctx.strokeStyle = COL.mic; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
    for (let x = 6; x < w; x += 10) {
      const y = mid - clamp(sig.noise(x / w * WINDOW_MS - scroll), -1.7, 1.7) * k;
      ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x, y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  };
  paint['thumb-2'] = c => thumb(c, [
    faint(sig.noise, COL.noise),
    { fn: t => state.noiseVol * forecastUnit(state.type, t), color: COL.proc, width: 1.8, dash: [5, 3] }
  ]);
  paint['thumb-3'] = c => thumb(c, [{ fn: sig.anti, color: COL.anti, width: 1.8 }]);
  paint['thumb-4'] = c => thumb(c, [T_NOISE(), { fn: sig.anti, color: COL.anti, width: 1.8 }]);
  paint['thumb-5'] = c => thumb(c, [faint(sig.noise, COL.noise), { fn: sig.residual, color: COL.out, width: 2 }]);

  /* =====================================================================
     Animation loop
     ===================================================================== */
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!state.paused) { scroll += dt * SCROLL_MS_PER_S; dirty = true; }
    if (dirty) {
      dirty = false;
      for (const c of canvases) if (c.visible && c.w) paint[c.name](c);
    }
    requestAnimationFrame(frame);
  }

  /* =====================================================================
     Text and UI
     ===================================================================== */
  const ST = () => NOISE[state.type];
  const msText = ms => ms.toFixed(1) + ' ms';
  const humDeg = ms => Math.round(ms / TONE_PERIOD * 360);

  // Is the current delay hurting? Judged from the delay curve for this noise type.
  function timing() {
    const d = msText(state.delay);
    if (!state.anc) return { level: 'neutral', short: 'Cancelling is off', text: 'Cancelling is off, so timing does not matter yet.' };
    if (A.spread < 0.06) return { level: 'neutral', short: 'Timing barely matters here', text: 'Timing barely matters for this noise. There is no steady pattern to line up with.' };
    const lost = A.ratio - A.curve[0];
    if (lost < 0.08) return { level: 'good', short: 'On time', text: `A delay of ${d} is short enough that the anti-noise still lines up well.` };
    if (A.ratio < 1) return { level: 'warn', short: 'Late: weaker', text: `The anti-noise arrives ${d} late, so peaks and dips only partly line up and cancellation is weaker.` };
    return { level: 'bad', short: 'Too late: backfires', text: `The anti-noise arrives ${d} late and lines up with the noise instead of opposing it, so it makes the noise worse.` };
  }
  function strengthNote() {
    const s = state.strength;
    if (s < 0.9) return { level: 'warn', text: `Anti-noise is only ${pct(s)}% as strong as the noise, so some noise is left over.` };
    if (s > 1.1) return { level: 'warn', text: `Anti-noise is ${pct(s)}% as strong as the noise. It over-shoots and leaves a flipped copy behind.` };
    return { level: 'good', text: 'Anti-noise strength matches the noise, which is what you want.' };
  }
  function predNote() {
    const p = pct(A.pred);
    if (A.pred >= 0.85) return { level: 'good', text: `${ST().label} is steady and repeats (${p}% predictable), so the processor can forecast it.` };
    if (A.pred >= 0.45) return { level: 'warn', text: `${ST().label} is only partly predictable (${p}%). Part of it is too random to flip in time.` };
    return { level: 'bad', text: `${ST().label} is mostly random (${p}% predictable). There is very little the processor can forecast.` };
  }

  function renderControls() {
    $('#ctl-noise').value = pct(state.noiseVol);
    $('#ctl-wanted').value = pct(state.wantedVol);
    $('#ctl-strength').value = pct(state.strength);
    $('#ctl-delay').value = Math.round(state.delay * 10);
    $('#o-noise').textContent = pct(state.noiseVol) + '%';
    $('#o-wanted').textContent = pct(state.wantedVol) + '%';
    $('#o-strength').textContent = pct(state.strength) + '%';
    $('#o-delay').textContent = msText(state.delay);
    $('#v-wanted').textContent = 'volume ' + pct(state.wantedVol) + '%';
    $('#v-noise').textContent = 'volume ' + pct(state.noiseVol) + '%';
    $('#n-noise').textContent = ST().label + ': ' + ST().blurb.split('.')[0];

    for (const b of $$('[data-type]')) b.setAttribute('aria-checked', String(b.dataset.type === state.type));
    for (const b of $$('[data-anc]')) b.setAttribute('aria-pressed', String((b.dataset.anc === 'on') === state.anc));
    $('#v-pred').textContent = pct(A.pred) + '% predictable';
    $('#h-type').textContent = ST().blurb;

    $('#h-noise').textContent = state.noiseVol === 0
      ? 'No noise at all, nothing to cancel.'
      : 'Louder noise makes a bigger red wave. The anti-noise grows with it.';
    $('#h-wanted').textContent = 'The music or voice you want to keep. Cancelling should leave this untouched.';
    $('#h-strength').textContent = state.strength < 0.9 ? 'Too weak: some noise is left over.'
      : state.strength > 1.1 ? 'Too strong: it over-shoots and flips the noise.'
      : 'Matched to the noise (100% = exact match).';
    const tm = timing();
    const deg = humDeg(state.delay);
    $('#h-delay').textContent = `${tm.short}. For a 120 Hz hum, ${msText(state.delay)} is ${deg}° of one wave cycle.`;
  }

  function renderOutput() {
    const g = grade(A.cut);
    const pill = $('#pill');
    if (state.anc) {
      pill.textContent = A.cut >= 0 ? `Noise −${pct(A.cut)}% · ${g.word}` : `Noise +${pct(-A.cut)}% · Worse`;
      pill.dataset.level = g.level;
    } else {
      pill.textContent = 'ANC off · full noise';
      pill.dataset.level = 'bad';
    }

    const off = pct(state.noiseVol), on = pct(state.noiseVol * A.ratio);
    $('#bar-off').style.width = off + '%';
    $('#bar-on').style.width = Math.min(100, on) + '%';
    $('#m-off').textContent = off + '%';
    $('#m-on').textContent = on + '%';
    $('.meter.c-out').style.opacity = state.anc ? 1 : 0.45;
    $('.meter.c-noise').style.opacity = state.anc ? 0.6 : 1;

    $('#n-out').textContent = state.anc
      ? 'Wanted sound plus whatever noise is left'
      : 'Wanted sound buried in full-strength noise';
    $('#m-caption').textContent = !state.anc
      ? 'Noise cancelling is OFF: all of the noise reaches your ear, so the output is messy.'
      : A.cut >= 0
        ? `Noise cancelling is ON: ${pct(A.cut)}% of the noise is removed (${A.db.toFixed(0)} dB). ${g.word} cancellation.`
        : `Noise cancelling is ON, but the noise is ${pct(-A.cut)}% louder. The anti-noise is hurting.`;
    $('#n-demo-result').textContent = state.anc
      ? (A.cut >= 0 ? `Noise + anti-noise leaves ${pct(A.ratio)}% of the original` : `Out of step: the noise got ${pct(-A.cut)}% louder`)
      : 'No anti-noise, so the result is just the noise';
  }

  /* ---- Pipeline ---- */
  const STAGES = [
    {
      title: '1 · Outside noise',
      body: 'Fans, engines and air conditioners push the air back and forth. That moving air is a sound wave: air pressure going up (peak) and down (dip), over and over.',
      live: () => `${ST().label} at ${pct(state.noiseVol)}% volume. It is the red wave.`
    },
    {
      title: '2 · Microphone detects noise',
      body: 'Tiny microphones on the outside of the headphones listen to the air. They turn the up-and-down pressure into a fluctuating electrical signal, a copy of the wave.',
      live: () => 'The mic cannot tell wanted sound from noise. It hears them added together (the amber wave in the sound scene).'
    },
    {
      title: '3 · Processor analyzes wave',
      body: 'A small chip studies the signal: how tall, how fast, how regular. If the pattern repeats, it can predict what the wave will do in the next few milliseconds, and it has to do this faster than the sound reaches your ear.',
      live: () => `${ST().label} is ${pct(A.pred)}% predictable. ${A.pred >= 0.85 ? 'Easy forecast!' : A.pred >= 0.45 ? 'Only part of it can be forecast.' : 'Hard to forecast.'} The dashed teal line is the prediction.`
    },
    {
      title: '4 · Anti-noise generated',
      body: 'The chip builds a mirror image of the predicted wave. Every peak becomes a dip and every dip becomes a peak. It also sets the loudness to match the noise.',
      live: () => state.anc
        ? `Strength ${pct(state.strength)}%, arriving ${msText(state.delay)} after the noise (${timing().short.toLowerCase()}).`
        : 'Noise cancelling is OFF, so no anti-noise is being made.'
    },
    {
      title: '5 · Waves combine',
      body: 'The speaker plays the anti-noise right next to your ear. There the noise and anti-noise meet and their pressures add up: a peak (+1) plus a dip (−1) is 0. This is called destructive interference.',
      live: () => state.anc
        ? `Noise (red) + anti-noise (blue) leaves the purple wave, ${pct(A.ratio)}% as big as the original noise.`
        : 'With cancelling OFF there is nothing to add, so the red noise passes straight through.'
    },
    {
      title: '6 · Reduced noise output',
      body: 'What reaches your ear is your music or voice plus a much quieter noise. Cancelling is never perfect, but steady sounds can drop a lot.',
      live: () => state.anc
        ? `Noise at your ear: ${pct(state.noiseVol)}% → ${pct(state.noiseVol * A.ratio)}% (${grade(A.cut).word.toLowerCase()}).`
        : `Noise at your ear: ${pct(state.noiseVol)}%, unchanged.`
    }
  ];

  function renderStage() {
    const s = STAGES[state.stage];
    $('#sd-title').textContent = s.title;
    $('#sd-body').textContent = s.body;
    $('#sd-live').textContent = 'In the simulator now: ' + s.live();
    for (const b of $$('.stage-btn')) b.setAttribute('aria-pressed', String(+b.dataset.stage === state.stage));
    $('#st-tour').setAttribute('aria-pressed', String(state.tour));
    $('#st-tour').textContent = state.tour ? '❚❚ Stop tour' : '▶ Auto tour';
  }

  /* ---- Timing experiment ---- */
  function renderTone() {
    const r = toneRatio(state.strength, state.tonePhase);
    const cut = 1 - r;
    const ms = toneDelay();
    $('#ctl-tone').value = state.tonePhase;
    $('#o-tone').textContent = `${state.tonePhase}° · ${ms.toFixed(1)} ms`;
    for (const b of $$('[data-phase]')) b.setAttribute('aria-pressed', String(+b.dataset.phase === state.tonePhase));

    let title, body, level;
    if (cut >= 0.9) {
      level = 'good'; title = `Good timing: hum ${pct(cut)}% quieter`;
      body = 'The anti-noise peaks land exactly on the noise dips (and the other way round), so they cancel almost completely.';
    } else if (cut >= 0.6) {
      level = 'good'; title = `Slight delay: hum ${pct(cut)}% quieter`;
      body = `Even ${ms.toFixed(1)} ms late is still helpful, but a small leftover wobble appears. Cancellation is strong, not perfect.`;
    } else if (cut >= 0.2) {
      level = 'warn'; title = `Noticeable delay: hum only ${pct(cut)}% quieter`;
      body = 'The anti-noise is now far enough out of step that peaks and dips only partly line up. Most of the noise survives.';
    } else if (cut >= -0.05) {
      level = 'neutral'; title = 'Too late to help: no real change';
      body = 'At roughly 60° late the anti-noise adds about as much as it takes away. This is the break-even point.';
    } else {
      level = 'bad'; title = `Bad timing: hum ${pct(-cut)}% louder`;
      body = state.tonePhase > 150 && state.tonePhase < 210
        ? 'The anti-noise is half a cycle late, so its peaks line up with the noise peaks. They add together and the noise doubles.'
        : 'The anti-noise is far enough out of step that it adds to the noise. Cancelling now makes things worse.';
    }
    const box = $('#tone-verdict');
    box.dataset.level = level;
    $('#tv-title').textContent = title;
    $('#tv-body').textContent = body;

    $('#chart-noise-note').textContent = A.spread < 0.06
      ? `${ST().label} has no steady pattern, so the curve is almost flat. Timing hardly matters, and neither does cancelling.`
      : `Move the Processing delay control to slide the dot. ${ST().label} is built from several tones, so its curve is bumpier than the pure hum.`;
  }

  /* ---- Logic panel ---- */
  function renderLogic() {
    const tm = timing(), sn = strengthNote(), pn = predNote();
    const list = $('#why-list');
    list.textContent = '';
    const items = state.anc ? [pn, tm, sn] : [{ level: 'neutral', text: 'Noise cancelling is OFF. Turn it on to see how well it works.' }, pn];
    for (const it of items) {
      const li = document.createElement('li');
      li.dataset.level = it.level;
      li.textContent = it.text;
      list.appendChild(li);
    }

    const g = grade(A.cut);
    const box = $('#why');
    if (state.anc) {
      box.dataset.level = g.level;
      $('#why-title').textContent = A.cut >= 0
        ? `Right now: ${pct(A.cut)}% of the noise is removed (${g.word.toLowerCase()})`
        : `Right now: the noise is ${pct(-A.cut)}% louder (cancelling is backfiring)`;
    } else {
      box.dataset.level = 'neutral';
      $('#why-title').textContent = 'Right now: no cancelling, you hear all of the noise';
    }

    const set = (n, text, here) => {
      const el = $('#lg-' + n);
      $('p', el).textContent = text;
      el.dataset.here = String(!!here);
    };
    set(1, `The outside microphone listens for background noise, like ${ST().label.toLowerCase()} here at ${pct(state.noiseVol)}% volume. It can only measure air pressure, so it hears your music or voice mixed in too.`);
    set(2, `The processor looks for a pattern and estimates the noise wave a few milliseconds ahead. ${ST().label} is ${pct(A.pred)}% predictable, so it ${A.pred >= 0.85 ? 'can forecast it very accurately.' : A.pred >= 0.45 ? 'can only forecast part of it.' : 'has almost nothing to forecast.'}`);
    set(3, state.anc
      ? `The speaker plays an opposite wave: peaks become dips. It is set to ${pct(state.strength)}% strength and arrives ${msText(state.delay)} after the noise.`
      : 'The speaker is silent because cancelling is off. No opposite wave is played.');
    set(4, state.anc
      ? `A peak (+1) meeting a dip (−1) adds to 0. Because the match is not perfect, ${pct(A.ratio)}% of the noise remains (${A.db > 0 ? '+' : ''}${A.db.toFixed(0)} dB). Cancelling is partial, not magic.`
      : 'With no opposite wave, nothing cancels. The full noise reaches your ear.');
    set(5, 'Fans, engines and air conditioners make the same wave again and again. Once the processor learns the pattern, the next moment is easy to guess, so the anti-noise lines up well.', state.type === 'hum' || state.type === 'cabin');
    set(6, 'A clap is over before the processor can react. Speech and crowd chatter change every few milliseconds. The anti-noise is built for a sound that has already changed, so it cancels little.', state.type === 'crowd' || state.type === 'static');
  }

  function render() {
    analyse();
    renderControls();
    renderOutput();
    renderStage();
    renderTone();
    renderLogic();
    dirty = true;
  }

  /* =====================================================================
     Events
     ===================================================================== */
  const on = (sel, ev, fn) => $$(sel).forEach(el => el.addEventListener(ev, fn));

  on('#ctl-noise', 'input', e => { state.noiseVol = e.target.value / 100; render(); });
  on('#ctl-wanted', 'input', e => { state.wantedVol = e.target.value / 100; render(); });
  on('#ctl-strength', 'input', e => { state.strength = e.target.value / 100; render(); });
  on('#ctl-delay', 'input', e => { state.delay = e.target.value / 10; render(); });
  on('[data-type]', 'click', e => { state.type = e.currentTarget.dataset.type; render(); });
  on('[data-anc]', 'click', e => { state.anc = e.currentTarget.dataset.anc === 'on'; render(); });
  on('#btn-reset', 'click', () => { Object.assign(state, DEFAULTS, { tonePhase: 0 }); render(); });

  on('#btn-pause', 'click', () => setPaused(!state.paused));
  function setPaused(p) {
    state.paused = p;
    $('#btn-pause').textContent = p ? 'Play' : 'Pause';
    dirty = true;
  }

  on('#btn-theme', 'click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    readColors();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);

  // Pipeline: click or hover (mouse only) selects a stage
  let tourTimer = null;
  function setTour(on) {
    state.tour = on;
    clearInterval(tourTimer);
    if (on) tourTimer = setInterval(() => selectStage((state.stage + 1) % STAGES.length, true), 3800);
    renderStage();
  }
  function selectStage(i, fromTour) {
    if (!fromTour && state.tour) setTour(false);
    state.stage = (i + STAGES.length) % STAGES.length;
    renderStage();
  }
  on('.stage-btn', 'click', e => selectStage(+e.currentTarget.dataset.stage));
  on('.stage-btn', 'pointerenter', e => { if (e.pointerType === 'mouse') selectStage(+e.currentTarget.dataset.stage); });
  on('#st-prev', 'click', () => selectStage(state.stage - 1));
  on('#st-next', 'click', () => selectStage(state.stage + 1));
  on('#st-tour', 'click', () => setTour(!state.tour));

  // Timing experiment
  on('#ctl-tone', 'input', e => { state.tonePhase = +e.target.value; renderTone(); dirty = true; });
  on('[data-phase]', 'click', e => { state.tonePhase = +e.currentTarget.dataset.phase; renderTone(); dirty = true; });
  on('#btn-apply', 'click', () => {
    state.type = 'hum';
    state.anc = true;
    state.delay = clamp(Math.round(toneDelay() * 10) / 10, 0, MAX_DELAY);
    render();
    $('#scene').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
  });

  /* =====================================================================
     Start
     ===================================================================== */
  readColors();
  setPaused(state.paused);
  render();
  requestAnimationFrame(frame);
})();
