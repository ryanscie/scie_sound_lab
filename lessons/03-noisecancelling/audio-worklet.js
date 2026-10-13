import { mainSample, timingSample } from './signal-model.js';

class NoiseLabProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.settings = { anc: true, type: 'hum', noiseVol: 0.7, wantedVol: 0.6, strength: 1, delay: 0.1, tonePhase: 0, source: 'main' };
    this.previous = this.settings;
    this.fade = 1;
    this.samples = 0;
    this.port.onmessage = ({ data }) => {
      this.previous = this.settings;
      this.settings = data;
      this.fade = 0;
    };
  }

  process(inputs, outputs) {
    const channel = outputs[0]?.[0];
    if (!channel) return true;
    const sample = (settings, t) => settings.source === 'timing' ? timingSample(settings, t) : mainSample(settings, t);
    for (let i = 0; i < channel.length; i++) {
      const t = this.samples++ / sampleRate * 1000;
      // Crossfade settings over 25 ms to soften switches and slider changes.
      this.fade = Math.min(1, this.fade + 1 / (sampleRate * 0.025));
      const current = sample(this.settings, t);
      channel[i] = this.fade === 1 ? current
        : sample(this.previous, t) * (1 - this.fade) + current * this.fade;
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(channel);
    return true;
  }
}

registerProcessor('noise-lab', NoiseLabProcessor);
