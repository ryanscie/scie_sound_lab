import test from 'node:test';
import assert from 'node:assert/strict';
import { breakEvenDegrees, toneRatio, wantedWave } from '../signal-model.js';

let Processor;
globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class { constructor() { this.port = {}; } };
globalThis.registerProcessor = (name, implementation) => { Processor = implementation; };
await import('../audio-worklet.js');

const defaults = { anc: true, type: 'hum', noiseVol: 0.7, wantedVol: 0, strength: 1, delay: 0.1, tonePhase: 0, source: 'main' };
function render(settings, count = 48000) {
  const processor = new Processor();
  processor.port.onmessage({ data: { ...defaults, ...settings } });
  // Discard the initial transition before measuring stable playback.
  processor.process([], [[new Float32Array(2400)]]);
  const output = new Float32Array(count);
  processor.process([], [[output]]);
  return output;
}
const rms = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
const near = (actual, expected, tolerance = 1e-5) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('actual audio output suppresses the fan hum; zero strength matches ANC off', () => {
  const off = rms(render({ anc: false }));
  const on = rms(render({}));
  assert.ok(on / off < 0.12);
  near(rms(render({ strength: 0 })), off);
});

test('silence when both sources are zero; ANC preserves the wanted tone', () => {
  assert.ok(render({ noiseVol: 0 }).every(value => value === 0));
  const wanted = render({ noiseVol: 0, wantedVol: 0.6 });
  for (let i = 0; i < wanted.length; i += 137) {
    near(wanted[i], 0.6 * wantedWave((i + 2400) / 48));
  }
});

test('timing audio cancels at 0° and 360°, reinforces by 2.5× at 150% strength', () => {
  near(rms(render({ source: 'timing', tonePhase: 0 })), 0);
  near(rms(render({ source: 'timing', tonePhase: 360 })), 0);
  const off = rms(render({ source: 'timing', strength: 0 }));
  near(rms(render({ source: 'timing', strength: 1.5, tonePhase: 180 })) / off, 2.5);
  near(rms(render({ source: 'timing', strength: 1, tonePhase: 90 })) / off, Math.sqrt(2));
});

test('break-even follows strength, rather than always being 60°', () => {
  near(breakEvenDegrees(1), 60);
  near(breakEvenDegrees(1.5), 41.40962210927086);
  assert.equal(breakEvenDegrees(0), null);
  for (const strength of [0.1, 0.5, 1, 1.5]) {
    near(toneRatio(strength, breakEvenDegrees(strength)), 1);
    near(toneRatio(strength, 360 - breakEvenDegrees(strength)), 1);
  }
});

test('all presets and extreme controls produce finite audio within fixed playback headroom', () => {
  for (const type of ['hum', 'cabin', 'crowd', 'static']) {
    for (const delay of [0, 4.2, 8.4]) {
      const samples = render({ type, delay, noiseVol: 1, wantedVol: 1, strength: 1.5 });
      assert.ok(samples.every(value => Number.isFinite(value) && Math.abs(value * 0.15) < 1));
    }
  }
});
