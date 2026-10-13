// Shared by the waveform renderer and the audio thread.
// t is milliseconds; frequency is kHz. Noise presets are illustrative syntheses.
export const TAU = Math.PI * 2;
const SKILL = 0.6;
export const TONE_KHZ = 0.12;
export const TONE_PERIOD = 1 / TONE_KHZ;
export const TONE_AMP = 0.9;
const hash = n => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
};
const vnoise = x => {                       // smooth random value noise in [-1, 1]
  const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
  return hash(i) * (1 - u) + hash(i + 1) * u;
};
const slowPart = t => 0.7 * vnoise(t * 0.22 + 10) + 0.5 * vnoise(t * 0.57 + 50);
const fastPart = t => 0.7 * vnoise(t * 2.3 + 90) + 0.45 * vnoise(t * 4.9 + 130);

export const NOISE = {
  hum: {
    label: 'Fan hum', gp: 1, gs: 0.05, gf: 0.03,
    tone: t => 0.8 * Math.sin(TAU * 0.12 * t) + 0.3 * Math.sin(TAU * 0.24 * t + 0.6) + 0.12 * Math.sin(TAU * 0.36 * t + 1.9),
    blurb: 'A steady, repeating low tone. The same wave over and over, so it is easy to predict.',
    steady: true
  },
  cabin: {
    label: 'Airplane cabin', gp: 0.8, gs: 0.3, gf: 0.12,
    tone: t => 0.6 * Math.sin(TAU * 0.08 * t + 0.3) + 0.45 * Math.sin(TAU * 0.13 * t + 1.1) + 0.3 * Math.sin(TAU * 0.2 * t + 2.2) + 0.2 * Math.sin(TAU * 0.31 * t),
    blurb: 'Constant engine rumble with a little turbulence. Mostly steady, a bit of randomness.',
    steady: true
  },
  crowd: {
    label: 'Crowd noise', gp: 0.15, gs: 0.5, gf: 0.7,
    tone: t => Math.sin(TAU * 0.1 * t),
    blurb: 'Voices that start, stop and overlap. Constantly changing, so hard to predict.',
    steady: false
  },
  static: {
    label: 'Random static', gp: 0, gs: 0.1, gf: 0.9,
    tone: () => 0,
    blurb: 'Pure hiss with no pattern at all. Nothing to predict, so almost nothing to cancel.',
    steady: false
  }
};

export const wantedWave = t =>
  0.85 * (0.7 * Math.sin(TAU * 0.28 * t) + 0.3 * Math.sin(TAU * 0.56 * t + 0.8)) * (1 + 0.25 * Math.sin(TAU * 0.03 * t));

export const noiseUnit = (type, t) => {
  const n = NOISE[type];
  return n.gp * n.tone(t) + n.gs * slowPart(t) + n.gf * fastPart(t);
};
export const forecastUnit = (type, t) => {         // what the processor can predict
  const n = NOISE[type];
  return n.gp * n.tone(t) + SKILL * n.gs * slowPart(t);
};

export const toneRatio = (strength, degrees) =>
  Math.sqrt(Math.max(0, 1 + strength * strength - 2 * strength * Math.cos(degrees * Math.PI / 180)));

export function mainSample(settings, t) {
  const noise = settings.noiseVol * noiseUnit(settings.type, t);
  const anti = settings.anc ? -settings.strength * settings.noiseVol * forecastUnit(settings.type, t - settings.delay) : 0;
  return settings.wantedVol * wantedWave(t) + noise + anti;
}

export function timingSample(settings, t) {
  const delay = settings.tonePhase / 360 * TONE_PERIOD;
  return TONE_AMP * (Math.sin(TAU * TONE_KHZ * t)
    - settings.strength * Math.sin(TAU * TONE_KHZ * (t - delay)));
}

export function breakEvenDegrees(strength) {
  return strength > 0 && strength <= 2 ? Math.acos(strength / 2) * 180 / Math.PI : null;
}
