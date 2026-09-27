#!/usr/bin/env python3
"""ECA fair demo -- Tier 2b: live single-note harmonic-series viewer.

Shows that ONE musical note is not one frequency -- it is a fundamental
plus a series of weaker overtones (harmonics) sitting at integer
multiples of that fundamental.

Hardened DSP pipeline (see HarmonicsAnalyzer.process_block):
    mic -> windowed FFT -> linear power spectrum -> per-bin adaptive
    noise floor / gate -> local-maxima peak picking (never "top-N bins")
    -> fundamental-candidate scoring by counting how many of its own
    integer multiples have a real, gated peak nearby -> a block is only
    allowed to nominate a fundamental if it clears both a per-peak SNR
    gate and a whole-block energy-above-floor gate.
A block-level nomination is then debounced over several consecutive
ticks (FundamentalTracker) before it is ever shown, and cleared after a
few consecutive misses -- so brief blips/noise show "Listening..."
instead of a guess, and a sustained note appears within a fraction of a
second.

Audio capture, FFT, and the whole peak/harmonic pipeline run on the
PortAudio callback thread; the Qt GUI thread only drains a queue and
redraws, so the UI never blocks on audio work.

Standalone from live_audio_demo.py (Tier 1) and live_chroma_demo.py
(Tier 2) -- neither file is touched by this one.

Run the app:
    python live_harmonics_demo.py

Run the internal self-tests (no GUI, no microphone, no audio playback):
    python live_harmonics_demo.py --selftest
"""

import queue
import sys

import numpy as np
import sounddevice as sd
from scipy.signal.windows import hann
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024              # samples per audio callback / analysis hop
FFT_SIZE = 8192                # analysis window (~5.4 Hz/bin -- resolves low notes)
HISTORY_SECONDS = 4.0          # waveform panel history window
DISPLAY_MIN_FREQ_HZ = 0.0
DISPLAY_MAX_FREQ_HZ = 2500.0   # spectrum panel x-axis range
UI_REFRESH_MS = 33             # ~30 fps

# --- fundamental search range: covers cello (~65 Hz) up to violin/soprano ---
MIN_FUND_HZ = 60.0
MAX_FUND_HZ = 1000.0
MAX_HARMONIC = 8                # highest harmonic index ever labeled (n in "nf")
MIN_HARMONICS_FOR_FUNDAMENTAL = 4  # fundamental + at least 3 overtones must line up, contiguously

# --- per-bin adaptive noise floor (linear power), leaky minimum follower:
# rises slowly (a held note isn't mistaken for the new ambient floor) but
# falls quickly (the floor keeps up once the room actually goes quiet). ---
BIN_FLOOR_RISE = 0.0008
BIN_FLOOR_FALL = 0.2
GATE_RATIO = 6.0                 # a bin must exceed GATE_RATIO x its own floor to be a peak

# whole-block energy gate: total gated power above the floor, across the
# search range, must reach this multiple of the floor's own total energy
# before we even try to nominate a fundamental -- rejects short/weak blips.
MIN_BLOCK_ENERGY_RATIO = 2.5

HARMONIC_TOL_MIN_BINS = 2.0      # minimum matching tolerance, in FFT bins
HARMONIC_TOL_FRACTION = 0.02     # + tolerance proportional to the target frequency
HARMONIC_TOL_MAX_HZ = 30.0       # absolute cap -- keeps high harmonics from getting a
                                  # implausibly wide window that noise can wander into

# --- temporal debouncing so the displayed fundamental doesn't flicker ---
# Consecutive blocks are NOT independent samples: with an 8192-sample FFT
# window advanced by a 1024-sample hop, adjacent blocks share 7/8 of their
# analysis window, so a spurious peak configuration can look "stable" for
# several ticks in a row even in random noise. CONFIRM_TICKS is kept at or
# above FFT_SIZE / BLOCK_SIZE so a confirmation always spans at least one
# full window's worth of genuinely new audio.
CONFIRM_TICKS = 10                 # consecutive matching ticks needed before display
RELEASE_TICKS = 14                 # consecutive misses before falling back to "Listening..."
NOTE_MATCH_SEMITONES = 0.5         # same note = within a half semitone of the last candidate

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note_name(freq_hz: float) -> str:
    midi = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    midi_round = int(round(midi))
    name = PITCH_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    return f"{name}{octave}"


def freq_to_midi(freq_hz: float) -> float:
    return 69.0 + 12.0 * np.log2(freq_hz / 440.0)


def find_local_peaks(power, floor, gate_ratio=GATE_RATIO):
    """Indices of local maxima in `power` that clear `gate_ratio` x their own
    adaptive noise floor. Never just "top-N bins" -- a real peak must be a
    strict local maximum AND stand out above the noise, DC excluded."""
    n = len(power)
    is_peak = np.zeros(n, dtype=bool)
    if n > 2:
        is_peak[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    gated = power > (floor * gate_ratio)
    is_peak &= gated
    is_peak[0] = False  # DC always excluded
    return np.nonzero(is_peak)[0]


def _nearest_peak_within(target_hz, peak_freqs, peak_power, tol_hz):
    if peak_freqs.size == 0:
        return None
    diffs = np.abs(peak_freqs - target_hz)
    within = diffs <= tol_hz
    if not np.any(within):
        return None
    candidates = np.nonzero(within)[0]
    best = candidates[np.argmax(peak_power[candidates])]
    return int(best)


def _harmonic_tolerance_hz(target_hz, bin_width_hz):
    return min(HARMONIC_TOL_MAX_HZ,
               max(HARMONIC_TOL_MIN_BINS * bin_width_hz, HARMONIC_TOL_FRACTION * target_hz))


def score_fundamental_candidate(f0, peak_freqs, peak_power, bin_width_hz, nyquist_hz,
                                 max_harmonic=MAX_HARMONIC):
    """Count a CONTIGUOUS run of f0's integer multiples (starting at f0
    itself) that each have a real gated peak nearby, stopping at the first
    gap. A contiguous run is far harder for scattered noise peaks to fake
    than "any N of M hits", which is what actually rejects broadband noise.
    Returns (contiguous_matched_count, matched_power_sum)."""
    matched = 0
    matched_power_sum = 0.0
    for k in range(1, max_harmonic + 1):
        target = k * f0
        if target > nyquist_hz:
            break
        tol = _harmonic_tolerance_hz(target, bin_width_hz)
        hit = _nearest_peak_within(target, peak_freqs, peak_power, tol)
        if hit is None:
            break
        matched += 1
        matched_power_sum += float(peak_power[hit])
    return matched, matched_power_sum


class HarmonicsAnalyzer:
    """Turns a stream of audio blocks into a display spectrum plus, when
    confident, a detected fundamental and its harmonic peaks.

    Pure numpy/scipy, no Qt or sounddevice dependency, so it can be driven
    both by the live microphone callback and by offline self-tests.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, fft_size=FFT_SIZE):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._window = hann(fft_size, sym=False).astype(np.float32)
        self._buffer = np.zeros(fft_size, dtype=np.float32)

        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.bin_width_hz = sample_rate / fft_size
        self.nyquist_hz = sample_rate / 2.0

        self._display_mask = (self.freqs >= DISPLAY_MIN_FREQ_HZ) & (self.freqs <= DISPLAY_MAX_FREQ_HZ)
        self._search_mask = self.freqs <= (MAX_FUND_HZ * MAX_HARMONIC)
        self._search_mask[0] = False  # DC excluded from every energy/peak computation

        self._bin_floor = None  # lazily initialized on first block

        # diagnostics from the most recent block (read-only, for --selftest use)
        self.last_energy_ratio = 0.0
        self.last_matched_harmonics = 0

    def process_block(self, samples):
        n = len(samples)
        self._buffer = np.roll(self._buffer, -n)
        self._buffer[-n:] = samples

        spectrum = np.fft.rfft(self._buffer * self._window)
        power = np.abs(spectrum) ** 2  # linear power; DC/invalid handled via masks below
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        if self._bin_floor is None:
            self._bin_floor = power.copy()
        else:
            rising = power >= self._bin_floor
            self._bin_floor = np.where(
                rising,
                self._bin_floor * (1.0 - BIN_FLOOR_RISE) + power * BIN_FLOOR_RISE,
                self._bin_floor * (1.0 - BIN_FLOOR_FALL) + power * BIN_FLOOR_FALL,
            )

        peak_idx = find_local_peaks(power, self._bin_floor)
        peak_freqs = self.freqs[peak_idx]
        peak_power = power[peak_idx]

        # whole-block energy gate: reject silence, room noise, and brief blips
        # before ever trying to nominate a fundamental.
        search_power = power[self._search_mask]
        search_floor = self._bin_floor[self._search_mask]
        floor_energy = float(np.sum(search_floor)) + 1e-12
        energy_ratio = float(np.sum(search_power)) / floor_energy
        self.last_energy_ratio = energy_ratio

        fundamental_hz = None
        fundamental_confidence = 0.0
        best_matched = 0

        if energy_ratio >= MIN_BLOCK_ENERGY_RATIO and peak_idx.size > 0:
            cand_mask = (peak_freqs >= MIN_FUND_HZ) & (peak_freqs <= MAX_FUND_HZ)
            cand_freqs = peak_freqs[cand_mask]
            cand_power = peak_power[cand_mask]

            best_score = (-1, -1.0)
            best_f0 = None
            for f0, p0 in zip(cand_freqs, cand_power):
                matched, matched_power = score_fundamental_candidate(
                    float(f0), peak_freqs, peak_power, self.bin_width_hz, self.nyquist_hz
                )
                score = (matched, matched_power)
                if score > best_score:
                    best_score = score
                    best_f0 = float(f0)

            if best_f0 is not None and best_score[0] >= MIN_HARMONICS_FOR_FUNDAMENTAL:
                fundamental_hz = best_f0
                best_matched = best_score[0]
                fundamental_confidence = min(1.0, best_matched / float(MAX_HARMONIC))

        self.last_matched_harmonics = best_matched

        harmonics = []
        if fundamental_hz is not None:
            for k in range(1, MAX_HARMONIC + 1):
                target = k * fundamental_hz
                if target > DISPLAY_MAX_FREQ_HZ:
                    break
                tol = _harmonic_tolerance_hz(target, self.bin_width_hz)
                hit = _nearest_peak_within(target, peak_freqs, peak_power, tol)
                if hit is not None:
                    harmonics.append((k, float(peak_freqs[hit]), float(peak_power[hit])))

        display_freqs = self.freqs[self._display_mask]
        display_magnitude = np.sqrt(np.maximum(power[self._display_mask], 0.0))
        display_magnitude = np.nan_to_num(display_magnitude, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "freqs": display_freqs,
            "magnitude": display_magnitude,
            "fundamental_hz": fundamental_hz,
            "fundamental_confidence": fundamental_confidence,
            "harmonics": harmonics,  # list of (n, freq_hz, power) for n>=1, n=1 is the fundamental itself
        }


class FundamentalTracker:
    """Debounces per-block fundamental nominations over time.

    A raw per-block candidate must repeat (same note, within tolerance) for
    CONFIRM_TICKS consecutive blocks before it is shown; RELEASE_TICKS
    consecutive misses are required before falling back to "Listening...".
    This is what turns a brief noise blip or a very short note into a
    graceful "Listening..." instead of a flickering guess.
    """

    def __init__(self, confirm_ticks=CONFIRM_TICKS, release_ticks=RELEASE_TICKS):
        self.confirm_ticks = confirm_ticks
        self.release_ticks = release_ticks
        self._candidate_midi = None
        self._candidate_streak = 0
        self._miss_streak = 0
        self.confirmed_hz = None
        self.confirmed_confidence = 0.0

    def update(self, raw_hz, raw_confidence):
        if raw_hz is not None:
            midi = freq_to_midi(raw_hz)
            if self._candidate_midi is not None and abs(midi - self._candidate_midi) <= NOTE_MATCH_SEMITONES:
                self._candidate_streak += 1
            else:
                self._candidate_midi = midi
                self._candidate_streak = 1
            self._miss_streak = 0

            if self._candidate_streak >= self.confirm_ticks:
                self.confirmed_hz = raw_hz
                self.confirmed_confidence = raw_confidence
        else:
            self._candidate_midi = None
            self._candidate_streak = 0
            self._miss_streak += 1
            if self._miss_streak >= self.release_ticks:
                self.confirmed_hz = None
                self.confirmed_confidence = 0.0

        return self.confirmed_hz, self.confirmed_confidence


class AudioEngine(QtCore.QObject):
    """Owns the microphone stream. FFT + harmonic pipeline run on the audio thread."""

    error = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, fft_size=FFT_SIZE):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.analyzer = HarmonicsAnalyzer(sample_rate=sample_rate, fft_size=fft_size)
        self.tracker = FundamentalTracker()
        self.out_queue: "queue.Queue[dict]" = queue.Queue()
        self.stream = None

        wave_len = int(HISTORY_SECONDS * sample_rate)
        self._wave_buf = np.zeros(wave_len, dtype=np.float32)

    def start(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            self.error.emit(
                "Could not access the microphone.\n\n"
                f"{exc}\n\n"
                "Check System Settings -> Privacy & Security -> Microphone "
                "and make sure your terminal/IDE is allowed."
            )

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's own thread -- never touches Qt widgets directly.
        samples = indata[:, 0].copy()

        n = len(samples)
        self._wave_buf = np.roll(self._wave_buf, -n)
        self._wave_buf[-n:] = samples

        result = self.analyzer.process_block(samples)
        confirmed_hz, confirmed_conf = self.tracker.update(
            result["fundamental_hz"], result["fundamental_confidence"]
        )
        result["confirmed_hz"] = confirmed_hz
        result["confirmed_confidence"] = confirmed_conf
        result["waveform"] = self._wave_buf.copy()

        try:
            self.out_queue.put_nowait(result)
        except queue.Full:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECA -- One Note Is Many Frequencies")
        self.resize(1500, 950)
        self.setStyleSheet("background-color: #0a0a0d;")

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#0a0a0d")
        pg.setConfigOption("foreground", "#d8d8dc")

        self.engine = AudioEngine()
        self.engine.error.connect(self.show_error)

        # slow ceiling follower for the spectrum's y-axis, so the scale stays
        # stable/comparable across frames instead of auto-stretching to fit
        # whatever noise is present this instant.
        self._y_ceiling = 0.05

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        self.setCentralWidget(central)

        title = QtWidgets.QLabel("ONE NOTE IS MANY FREQUENCIES")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            "color: #e6e6ea; font-size: 30px; font-weight: 700; letter-spacing: 3px;"
        )
        layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.fundamental_label = QtWidgets.QLabel("Listening...")
        self.fundamental_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fundamental_label.setStyleSheet(
            "color: #ffc43c; font-size: 40px; font-weight: 800; letter-spacing: 2px;"
        )
        layout.addWidget(self.fundamental_label)

        # ---- TIME DOMAIN panel ----
        wave_caption = QtWidgets.QLabel("TIME DOMAIN")
        wave_caption.setStyleSheet(
            "color: #7a7a82; font-size: 13px; font-weight: 600; letter-spacing: 3px;"
        )
        layout.addWidget(wave_caption)

        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.hideButtons()
        self.wave_plot.setMenuEnabled(False)
        self.wave_plot.showGrid(x=True, y=True, alpha=0.15)
        self.wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.wave_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Time (s)")
        self._wave_x = np.linspace(-HISTORY_SECONDS, 0, int(HISTORY_SECONDS * SAMPLE_RATE))
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.5))
        layout.addWidget(self.wave_plot, stretch=1)

        # ---- FREQUENCY DOMAIN panel ----
        spec_caption = QtWidgets.QLabel("FREQUENCY DOMAIN")
        spec_caption.setStyleSheet(
            "color: #7a7a82; font-size: 13px; font-weight: 600; letter-spacing: 3px;"
        )
        layout.addWidget(spec_caption)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setMouseEnabled(x=False, y=False)
        self.spectrum_plot.hideButtons()
        self.spectrum_plot.setMenuEnabled(False)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectrum_plot.setXRange(DISPLAY_MIN_FREQ_HZ, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.spectrum_plot.setYRange(0.0, self._y_ceiling, padding=0.05)
        self.spectrum_plot.setLabel("left", "Amplitude")
        self.spectrum_plot.setLabel("bottom", "Frequency (Hz)")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.5))
        layout.addWidget(self.spectrum_plot, stretch=1)

        # scatter markers + a fixed pool of text labels for the fundamental
        # and up to MAX_HARMONIC overtones -- reused every frame so we never
        # leak graphics items.
        self.fundamental_scatter = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush("#ffc43c"), pen=pg.mkPen("#0a0a0d", width=1.5)
        )
        self.harmonic_scatter = pg.ScatterPlotItem(
            size=9, brush=pg.mkBrush("#4fd1ff"), pen=pg.mkPen("#0a0a0d", width=1)
        )
        self.spectrum_plot.addItem(self.harmonic_scatter)
        self.spectrum_plot.addItem(self.fundamental_scatter)

        self._peak_labels = []
        for _ in range(MAX_HARMONIC + 1):
            label = pg.TextItem(anchor=(0.5, 1.0))
            label.setVisible(False)
            self.spectrum_plot.addItem(label)
            self._peak_labels.append(label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UI_REFRESH_MS)

        self.engine.start()

    def show_error(self, message: str):
        self.status_label.setText(message)
        self.status_label.show()
        self.wave_plot.hide()
        self.spectrum_plot.hide()
        self.fundamental_label.hide()

    def update_display(self):
        drained = 0
        latest = None
        try:
            while True:
                latest = self.engine.out_queue.get_nowait()
                drained += 1
                if drained >= 64:  # avoid unbounded work if the GUI ever falls behind
                    break
        except queue.Empty:
            pass

        if drained == 0 or latest is None:
            return

        waveform = np.nan_to_num(latest["waveform"], nan=0.0, posinf=0.0, neginf=0.0)
        self.wave_curve.setData(self._wave_x, waveform)

        freqs = latest["freqs"]
        magnitude = np.nan_to_num(latest["magnitude"], nan=0.0, posinf=0.0, neginf=0.0)
        self.spectrum_curve.setData(freqs, magnitude)

        current_peak = float(np.max(magnitude)) if magnitude.size else 0.0
        if current_peak > self._y_ceiling:
            self._y_ceiling = self._y_ceiling * 0.9 + current_peak * 1.15 * 0.1
        else:
            self._y_ceiling = self._y_ceiling * 0.999 + current_peak * 0.001
        self._y_ceiling = max(self._y_ceiling, 0.01)
        self.spectrum_plot.setYRange(0.0, self._y_ceiling, padding=0.05)

        confirmed_hz = latest["confirmed_hz"]
        for label in self._peak_labels:
            label.setVisible(False)
        self.fundamental_scatter.setData([], [])
        self.harmonic_scatter.setData([], [])

        if confirmed_hz is None:
            self.fundamental_label.setText("Listening...")
        else:
            note_name = freq_to_note_name(confirmed_hz)
            self.fundamental_label.setText(f"{note_name} — {confirmed_hz:.0f} Hz")

            harmonics = latest["harmonics"]
            harm_x, harm_y = [], []
            label_idx = 0
            for n, hz, power in harmonics:
                mag = float(np.sqrt(max(power, 0.0)))
                if n == 1:
                    self.fundamental_scatter.setData([hz], [mag])
                    text = f"1f\n{note_name}"
                    color = "#ffc43c"
                else:
                    harm_x.append(hz)
                    harm_y.append(mag)
                    text = f"{n}f"
                    color = "#4fd1ff"

                if label_idx < len(self._peak_labels):
                    label = self._peak_labels[label_idx]
                    label.setText(text, color=color)
                    label.setPos(hz, mag + self._y_ceiling * 0.04)
                    label.setVisible(True)
                    label_idx += 1

            if harm_x:
                self.harmonic_scatter.setData(harm_x, harm_y)

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(event)


def _make_harmonic_tone(f0, n_harmonics=6, duration=2.0, sample_rate=SAMPLE_RATE, amplitude=0.2):
    """Synthesize a note with a real harmonic series -- fundamental strongest,
    each successive overtone progressively weaker (like a bowed string)."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        signal += (1.0 / k) * np.sin(2.0 * np.pi * f0 * k * t)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= amplitude / peak
    return signal.astype(np.float32)


def _make_test_noise(duration=2.0, sample_rate=SAMPLE_RATE, amplitude=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(duration * sample_rate))).astype(np.float32)


def _run_pipeline(signal, analyzer, tracker, block_size=BLOCK_SIZE):
    """Feed a signal through analyzer -> tracker, return the final block's
    (confirmed_hz, confirmed_confidence, last_result)."""
    confirmed_hz, confirmed_conf, result = None, 0.0, None
    for start in range(0, len(signal) - block_size + 1, block_size):
        result = analyzer.process_block(signal[start:start + block_size])
        confirmed_hz, confirmed_conf = tracker.update(
            result["fundamental_hz"], result["fundamental_confidence"]
        )
    return confirmed_hz, confirmed_conf, result


def run_self_tests():
    """Offline validation of the analysis pipeline -- no GUI, no mic, no playback."""
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        if not cond:
            failures.append(name)

    def diag(label, hz, conf, result):
        harm_str = "none" if result is None else [(n, round(f, 1)) for n, f, _ in result["harmonics"]]
        print(f"    -- diag [{label}] confirmed_hz={hz} confidence={conf:.2f} harmonics={harm_str}")

    def warmed_up(noise_amp=0.02, warmup_s=15.0, seed=1):
        analyzer = HarmonicsAnalyzer()
        tracker = FundamentalTracker()
        _run_pipeline(_make_test_noise(warmup_s, amplitude=noise_amp, seed=seed), analyzer, tracker)
        # fresh tracker so ambient warmup doesn't leave a stale streak
        return analyzer, FundamentalTracker()

    # 1. Peak-picking sanity: a pure tone produces a local-maxima peak near its bin.
    analyzer = HarmonicsAnalyzer()
    tone = _make_harmonic_tone(220.0, n_harmonics=1, duration=1.0)
    result = None
    for start in range(0, len(tone) - BLOCK_SIZE + 1, BLOCK_SIZE):
        result = analyzer.process_block(tone[start:start + BLOCK_SIZE])
    mags = result["magnitude"]
    freqs = result["freqs"]
    check("spectrum has exactly 12 or more finite values", np.all(np.isfinite(mags)))
    near_220 = np.any(np.abs(freqs[np.argsort(mags)[-5:]] - 220.0) < 10.0)
    check("strongest displayed bins sit near 220 Hz for a pure 220 Hz tone", near_220)

    # 2. Silence -> no fundamental, no harmonics, "Listening..." condition.
    analyzer, tracker = warmed_up()
    hz, conf, result = _run_pipeline(np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32), analyzer, tracker)
    diag("silence", hz, conf, result)
    check("silence produces no confirmed fundamental", hz is None)
    check("silence spectrum stays finite and non-negative",
          np.all(np.isfinite(result["magnitude"])) and np.all(result["magnitude"] >= 0.0))

    # 3. Broadband noise (room noise / speech stand-in) -> no false harmonic series.
    for noise_amp in (0.02, 0.08, 0.2):
        analyzer, tracker = warmed_up(noise_amp=noise_amp)
        hz, conf, result = _run_pipeline(
            _make_test_noise(3.0, amplitude=noise_amp, seed=42), analyzer, tracker
        )
        diag(f"broadband noise amp={noise_amp}", hz, conf, result)
        check(f"broadband noise (amp={noise_amp}) does not produce a confirmed fundamental", hz is None)

    # 4. Sustained cello-like note (~110 Hz, A2) -> clear fundamental + several harmonics.
    analyzer, tracker = warmed_up()
    tone = _make_harmonic_tone(110.0, n_harmonics=6, duration=3.0)
    hz, conf, result = _run_pipeline(tone, analyzer, tracker)
    diag("sustained A2 (110 Hz) with harmonics", hz, conf, result)
    check("sustained harmonic-rich note yields a confirmed fundamental", hz is not None)
    if hz is not None:
        check("confirmed fundamental is close to 110 Hz", abs(hz - 110.0) < 5.0)
        check("at least 3 harmonics (incl. fundamental) are detected",
              len(result["harmonics"]) >= 3)
        check("harmonic frequencies sit near integer multiples of the fundamental",
              all(abs(f - n * hz) < max(5.0, 0.05 * n * hz) for n, f, _ in result["harmonics"]))
        check("note name resolves near A2", freq_to_note_name(hz) in ("A2", "G#2", "A#2"))

    # 5. Changing note: 110 Hz -> 220 Hz updates the detected fundamental.
    analyzer, tracker = warmed_up()
    tone_low = _make_harmonic_tone(110.0, n_harmonics=6, duration=2.0)
    _run_pipeline(tone_low, analyzer, tracker)
    tone_high = _make_harmonic_tone(220.0, n_harmonics=6, duration=2.0)
    hz2, conf2, result2 = _run_pipeline(tone_high, analyzer, tracker)
    diag("after switching 110 Hz -> 220 Hz", hz2, conf2, result2)
    check("switching note updates the confirmed fundamental to ~220 Hz",
          hz2 is not None and abs(hz2 - 220.0) < 8.0)

    # 6. Short/weak note -> gracefully shows "Listening..." rather than guessing.
    analyzer, tracker = warmed_up()
    short_tone = _make_harmonic_tone(330.0, n_harmonics=6, duration=0.05, amplitude=0.06)
    hz3, conf3, result3 = _run_pipeline(short_tone, analyzer, tracker)
    diag("very short/weak note (50ms)", hz3, conf3, result3)
    check("a very short/weak note does not get confirmed as a fundamental", hz3 is None)

    # 7. All displayed data stays finite across a pathological all-zero sequence.
    analyzer = HarmonicsAnalyzer()
    result = None
    zero_signal = np.zeros(FFT_SIZE * 2, dtype=np.float32)
    for start in range(0, len(zero_signal) - BLOCK_SIZE + 1, BLOCK_SIZE):
        result = analyzer.process_block(zero_signal[start:start + BLOCK_SIZE])
    check("degenerate all-zero input yields a finite, non-negative spectrum",
          np.all(np.isfinite(result["magnitude"])) and np.all(result["magnitude"] >= 0.0))

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All self-tests passed.")
    return 0


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_self_tests())
    main()
