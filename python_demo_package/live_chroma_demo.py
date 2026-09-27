#!/usr/bin/env python3
"""ECA fair demo -- Tier 2: live 12-note chroma (pitch class) detector.

Captures microphone audio continuously, takes its FFT, and folds every
frequency bin into one of the 12 pitch classes (C, C#, D, ... B) by
summing LINEAR power across octaves. This does NOT identify chords --
it only shows which notes are present.

Hardened DSP pipeline (see ChromaAnalyzer.process_block):
    mic -> FFT magnitude -> linear power -> per-bin adaptive noise gate
    -> aggregate LINEAR power into 12 pitch classes -> adaptive
    noise-floor subtraction (per pitch class) -> dB ratio relative to
    that floor -> contrast compression -> fixed 0-100 display range.
Temporal smoothing (EMA) is then applied once per UI tick, on the GUI
thread, as the final step before the bars are drawn.

Audio capture, FFT, and pitch-class aggregation run on the PortAudio
callback thread; the Qt GUI thread only drains a queue, smooths the
result over time, and redraws, so the UI never blocks on audio work.

Standalone from live_audio_demo.py (Tier 1) -- that file is untouched.

Run the app:
    python live_chroma_demo.py

Run the internal self-tests (no GUI, no microphone, no audio playback):
    python live_chroma_demo.py --selftest
"""

import queue
import sys

import numpy as np
import sounddevice as sd
from scipy.signal.windows import hann
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024          # samples per audio callback / chroma update hop
FFT_SIZE = 8192            # analysis window (long enough to resolve low notes)
MIN_FREQ_HZ = 50.0         # low end of the musical range we fold into chroma
MAX_FREQ_HZ = 5000.0       # high end of the musical range we fold into chroma
UI_REFRESH_MS = 33         # ~30 fps
SMOOTHING_ALPHA = 0.30     # exponential smoothing applied once per UI tick

# --- per-bin adaptive noise floor (linear power), tracked continuously ---
# Asymmetric "leaky minimum follower": rises slowly (so a held note isn't
# mistaken for the new ambient floor) but falls quickly (so the floor keeps
# up when the room actually goes quiet).
BIN_FLOOR_RISE = 0.0008
BIN_FLOOR_FALL = 0.2
# A bin must exceed GATE_RATIO x its own floor to contribute any energy at
# all -- this is the per-bin noise gate. Bins below it are fully silenced
# before aggregation, so broadband room noise contributes little/nothing.
GATE_RATIO = 6.0

# --- per-pitch-class adaptive baseline, tracked on the aggregated (already
# gated) linear power. This is what "adaptive noise-floor subtraction" and
# "express each value relative to the current noise floor" operate against.
# Same asymmetric idea, one octave slower so a several-second note doesn't
# get absorbed into its own baseline.
CHROMA_FLOOR_RISE = 0.0015
CHROMA_FLOOR_FALL = 0.1
CHROMA_FLOOR_ADD = 50.0    # additive stabilizer, avoids blow-up when floor ~ 0

# --- contrast compression into a fixed, non-adaptive 0-100 display range ---
SQUELCH_RATIO = 0.35       # dead zone: ratios below this display as exactly 0
DISPLAY_DB_RANGE = 15.0    # dB span (above the noise floor) mapped to 0-100
DISPLAY_GAMMA = 0.8        # <1 lifts weak-but-real notes without lifting noise

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --- Tier 3: chord recognition (major/minor triads only, no machine learning) ---
MAJOR_INTERVALS = (0, 4, 7)   # root, major 3rd, perfect 5th
MINOR_INTERVALS = (0, 3, 7)   # root, minor 3rd, perfect 5th

CHORD_SCORE_THRESH = 0.82       # cosine similarity required to even consider a chord
CHORD_CONCENTRATION_THRESH = 0.65  # fraction of total chroma energy that must sit in the triad
CHORD_MIN_ENERGY = 40.0         # total chroma energy (0-100 scale, summed) required to look for a chord
CHORD_CONFIRM_TICKS = 6         # consecutive matching ticks needed before a chord is shown
CHORD_RELEASE_TICKS = 10        # consecutive non-matching ticks before falling back to "Listening..."


def build_pitch_class_map(fft_size=FFT_SIZE, sample_rate=SAMPLE_RATE,
                           min_freq=MIN_FREQ_HZ, max_freq=MAX_FREQ_HZ):
    """Map each rFFT bin index to a pitch class 0-11, or -1 if out of range/DC."""
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    pitch_class = np.full(freqs.shape, -1, dtype=np.int64)
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    f = freqs[valid]
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    pitch_class[valid] = np.round(midi).astype(np.int64) % 12
    return pitch_class


class ChromaAnalyzer:
    """Turns a stream of audio blocks into 12 bounded display values (0-100).

    Pure numpy/scipy, no Qt or sounddevice dependency, so it can be driven
    both by the live microphone callback and by offline self-tests.

    DSP pipeline per block (see module docstring for the full chain):
      1. FFT -> linear power spectrum (never dB at this stage).
      2. Per-bin adaptive noise floor + hard gate: a bin only contributes
         (power - floor) once power exceeds GATE_RATIO x floor.
      3. Aggregate: sum LINEAR power across octaves into 12 pitch classes
         (never sum dB across bins -- dB is only used at the very end, on
         the already-aggregated scalar, purely for display compression).
      4. A second, slower adaptive floor tracks each pitch class's own
         gated-aggregate baseline; the display value is expressed relative
         to that baseline, so it self-calibrates to whatever the ambient
         noise level actually is instead of assuming a fixed loudness.
      5. Squelch + dB compression + gamma curve into a fixed 0-100 range
         (no auto-scaling by the current frame's peak).
    """

    def __init__(self, sample_rate=SAMPLE_RATE, fft_size=FFT_SIZE):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._window = hann(fft_size, sym=False).astype(np.float32)
        self._buffer = np.zeros(fft_size, dtype=np.float32)

        pitch_class_map = build_pitch_class_map(fft_size, sample_rate)
        self._valid_mask = pitch_class_map >= 0
        self._valid_pc = pitch_class_map[self._valid_mask]

        self._bin_floor = None          # lazily initialized on first block
        self._chroma_floor = np.zeros(12, dtype=np.float64)

        # diagnostics from the most recent block (read-only, for --diag use)
        self.last_noise_floor_mean = 0.0
        self.last_chroma_raw = np.zeros(12, dtype=np.float64)

    def process_block(self, samples):
        n = len(samples)
        self._buffer = np.roll(self._buffer, -n)
        self._buffer[-n:] = samples

        spectrum = np.fft.rfft(self._buffer * self._window)
        power = np.abs(spectrum) ** 2  # linear power, DC/invalid bins excluded below

        if self._bin_floor is None:
            self._bin_floor = power.copy()
        else:
            rising = power >= self._bin_floor
            self._bin_floor = np.where(
                rising,
                self._bin_floor * (1.0 - BIN_FLOOR_RISE) + power * BIN_FLOOR_RISE,
                self._bin_floor * (1.0 - BIN_FLOOR_FALL) + power * BIN_FLOOR_FALL,
            )

        bin_excess = power - self._bin_floor
        gated = power > (self._bin_floor * GATE_RATIO)
        bin_excess = np.where(gated, np.maximum(bin_excess, 0.0), 0.0)

        # Aggregate LINEAR power across octaves -- never dB.
        chroma_excess = np.bincount(
            self._valid_pc, weights=bin_excess[self._valid_mask], minlength=12
        )[:12].astype(np.float64)

        rising_c = chroma_excess >= self._chroma_floor
        self._chroma_floor = np.where(
            rising_c,
            self._chroma_floor * (1.0 - CHROMA_FLOOR_RISE) + chroma_excess * CHROMA_FLOOR_RISE,
            self._chroma_floor * (1.0 - CHROMA_FLOOR_FALL) + chroma_excess * CHROMA_FLOOR_FALL,
        )

        above_floor = np.maximum(chroma_excess - self._chroma_floor, 0.0)
        ratio = above_floor / (self._chroma_floor + CHROMA_FLOOR_ADD)

        squelched = np.where(ratio < SQUELCH_RATIO, 0.0, ratio - SQUELCH_RATIO)
        db = 10.0 * np.log10(1.0 + np.maximum(squelched, 0.0))
        db = np.clip(db, 0.0, DISPLAY_DB_RANGE)
        display = 100.0 * (db / DISPLAY_DB_RANGE) ** DISPLAY_GAMMA

        display = np.nan_to_num(display, nan=0.0, posinf=0.0, neginf=0.0)
        display = np.clip(display, 0.0, 100.0)

        self.last_noise_floor_mean = float(np.mean(self._bin_floor[self._valid_mask]))
        self.last_chroma_raw = chroma_excess
        return display


def build_chord_templates():
    """24 unit-norm major/minor triad templates over the 12 pitch classes.

    Row layout: index = root*2 + (0 for major, 1 for minor).
    """
    templates = np.zeros((24, 12), dtype=np.float64)
    labels = []
    for root in range(12):
        for quality_idx, intervals in enumerate((MAJOR_INTERVALS, MINOR_INTERVALS)):
            row = root * 2 + quality_idx
            for interval in intervals:
                templates[row, (root + interval) % 12] = 1.0
            labels.append((root, "major" if quality_idx == 0 else "minor"))
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates_unit = templates / norms
    return templates_unit, labels


class ChordDetector:
    """Matches a 12-bin chroma vector against 24 major/minor triad templates.

    Pure template matching (cosine similarity) -- no machine learning. A
    match is only reported once it has: (a) enough total chroma energy to
    be a real sound rather than noise floor jitter, (b) a cosine similarity
    above CHORD_SCORE_THRESH against its best-matching triad, and (c) most
    of that energy actually concentrated in the triad's three notes (so a
    single strong note plus scattered weak harmonics/noise can't "win" by
    being the least-bad match among 24 imperfect templates). The result is
    then debounced in time so the displayed chord doesn't flicker.
    """

    def __init__(self,
                 score_thresh=CHORD_SCORE_THRESH,
                 concentration_thresh=CHORD_CONCENTRATION_THRESH,
                 min_energy=CHORD_MIN_ENERGY,
                 confirm_ticks=CHORD_CONFIRM_TICKS,
                 release_ticks=CHORD_RELEASE_TICKS):
        self.templates, self.labels = build_chord_templates()
        self.score_thresh = score_thresh
        self.concentration_thresh = concentration_thresh
        self.min_energy = min_energy
        self.confirm_ticks = confirm_ticks
        self.release_ticks = release_ticks

        self._candidate = None
        self._candidate_streak = 0
        self._miss_streak = 0
        self.confirmed = None            # (root_idx, "major"/"minor") or None
        self.confirmed_confidence = 0.0  # 0-1

    def update(self, chroma_vec):
        """Feed one frame's chroma vector; returns (confirmed_label_or_None, confidence)."""
        chroma_vec = np.nan_to_num(np.asarray(chroma_vec, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        chroma_vec = np.clip(chroma_vec, 0.0, None)

        best_label = None
        best_score = 0.0
        total_energy = float(np.sum(chroma_vec))

        if total_energy >= self.min_energy:
            norm = float(np.linalg.norm(chroma_vec))
            if norm > 1e-9:
                unit = chroma_vec / norm
                scores = self.templates @ unit
                idx = int(np.argmax(scores))
                score = float(np.clip(scores[idx], 0.0, 1.0))
                template_mask = self.templates[idx] > 0
                concentration = float(np.sum(chroma_vec[template_mask]) / total_energy)
                if score >= self.score_thresh and concentration >= self.concentration_thresh:
                    best_label = self.labels[idx]
                    best_score = score

        if best_label is not None and best_label == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = best_label
            self._candidate_streak = 1 if best_label is not None else 0

        if best_label is not None:
            self._miss_streak = 0
            if self._candidate_streak >= self.confirm_ticks:
                self.confirmed = best_label
                self.confirmed_confidence = best_score
        else:
            self._miss_streak += 1
            if self._miss_streak >= self.release_ticks:
                self.confirmed = None
                self.confirmed_confidence = 0.0

        return self.confirmed, self.confirmed_confidence


class AudioEngine(QtCore.QObject):
    """Owns the microphone stream. FFT + chroma pipeline run on the audio thread."""

    error = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, fft_size=FFT_SIZE):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.analyzer = ChromaAnalyzer(sample_rate=sample_rate, fft_size=fft_size)
        self.out_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None

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
        display = self.analyzer.process_block(samples)
        try:
            self.out_queue.put_nowait(display)
        except queue.Full:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECA -- What Did It Hear?")
        self.resize(1400, 850)
        self.setStyleSheet("background-color: #0a0a0d;")

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#0a0a0d")
        pg.setConfigOption("foreground", "#d8d8dc")

        self.chroma_smooth = np.zeros(12, dtype=np.float64)
        self.chord_detector = ChordDetector()

        self.engine = AudioEngine()
        self.engine.error.connect(self.show_error)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        title = QtWidgets.QLabel("WHAT DID IT HEAR?")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            "color: #e6e6ea; font-size: 34px; font-weight: 700; letter-spacing: 4px;"
        )
        layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.bar_plot = pg.PlotWidget()
        self.bar_plot.setMouseEnabled(x=False, y=False)
        self.bar_plot.hideButtons()
        self.bar_plot.setMenuEnabled(False)
        self.bar_plot.showGrid(x=False, y=True, alpha=0.15)
        # Fixed 0-100 display range -- not auto-scaled by the current frame's
        # peak, so background noise can never stretch to fill the display.
        self.bar_plot.setYRange(0.0, 100.0, padding=0.03)
        self.bar_plot.setXRange(-0.7, 11.7, padding=0)
        self.bar_plot.getAxis("bottom").setTicks([[(i, name) for i, name in enumerate(PITCH_NAMES)]])
        self.bar_plot.getAxis("bottom").setStyle(tickFont=QtGui.QFont("", 16, QtGui.QFont.Bold))
        self.bar_plot.getAxis("left").hide()

        self.bar_item = pg.BarGraphItem(
            x=np.arange(12), height=np.zeros(12), width=0.7,
            brushes=["#2a4a55"] * 12, pen=pg.mkPen(None),
        )
        self.bar_plot.addItem(self.bar_item)
        layout.addWidget(self.bar_plot, stretch=1)

        chord_box = QtWidgets.QVBoxLayout()
        chord_box.setSpacing(4)

        chord_caption = QtWidgets.QLabel("DETECTED CHORD")
        chord_caption.setAlignment(QtCore.Qt.AlignCenter)
        chord_caption.setStyleSheet(
            "color: #7a7a82; font-size: 16px; font-weight: 600; letter-spacing: 3px;"
        )
        chord_box.addWidget(chord_caption)

        self.chord_label = QtWidgets.QLabel("Listening...")
        self.chord_label.setAlignment(QtCore.Qt.AlignCenter)
        self.chord_label.setStyleSheet(
            "color: #ffc43c; font-size: 56px; font-weight: 800; letter-spacing: 3px;"
        )
        chord_box.addWidget(self.chord_label)

        self.confidence_label = QtWidgets.QLabel("")
        self.confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #9a9aa2; font-size: 16px;")
        chord_box.addWidget(self.confidence_label)

        layout.addLayout(chord_box)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UI_REFRESH_MS)

        self.engine.start()

    def show_error(self, message: str):
        self.status_label.setText(message)
        self.status_label.show()
        self.bar_plot.hide()

    @staticmethod
    def _bar_color(level_0_100: float) -> str:
        # Cool, dim blue-grey for weak pitch classes; bright warm gold for strong ones.
        level = min(max(level_0_100, 0.0), 100.0) / 100.0
        lo = np.array([42, 74, 85])
        hi = np.array([255, 196, 60])
        rgb = (lo + (hi - lo) * level).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

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

        latest = np.nan_to_num(np.asarray(latest, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if latest.shape != (12,):
            return  # defensive: never let a malformed vector reach the display
        latest = np.clip(latest, 0.0, 100.0)

        # Final pipeline step: moderate temporal smoothing on the already
        # fixed-range (0-100) values -- real notes appear within a couple of
        # ticks, single-frame noise blips get averaged down before they're seen.
        self.chroma_smooth = (
            SMOOTHING_ALPHA * latest + (1.0 - SMOOTHING_ALPHA) * self.chroma_smooth
        )
        self.chroma_smooth = np.clip(self.chroma_smooth, 0.0, 100.0)

        colors = [self._bar_color(v) for v in self.chroma_smooth]
        self.bar_item.setOpts(height=self.chroma_smooth, brushes=colors)

        chord, confidence = self.chord_detector.update(self.chroma_smooth)
        if chord is None:
            self.chord_label.setText("Listening...")
            self.confidence_label.setText("")
        else:
            root_idx, quality = chord
            self.chord_label.setText(f"{PITCH_NAMES[root_idx]} {quality.upper()}")
            self.confidence_label.setText(f"Confidence: {int(round(confidence * 100))}%")

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(event)


def _make_test_tone(freqs_hz, duration=2.0, sample_rate=SAMPLE_RATE, amplitude=0.15):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t, dtype=np.float64)
    for f in freqs_hz:
        signal += np.sin(2.0 * np.pi * f * t)
    signal *= amplitude / max(len(freqs_hz), 1)
    return signal.astype(np.float32)


def _make_test_noise(duration=2.0, sample_rate=SAMPLE_RATE, amplitude=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(duration * sample_rate))).astype(np.float32)


def _run_analyzer(signal, analyzer, block_size=BLOCK_SIZE):
    """Feed a signal through an (already-warmed) analyzer, return the final block's display."""
    display = np.zeros(12, dtype=np.float64)
    for start in range(0, len(signal) - block_size + 1, block_size):
        display = analyzer.process_block(signal[start:start + block_size])
    return display


def run_self_tests():
    """Offline validation of the analysis pipeline -- no GUI, no mic, no playback."""
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        if not cond:
            failures.append(name)

    def diag(label, analyzer, display):
        print(f"    -- diag [{label}] noise_floor_mean={analyzer.last_noise_floor_mean:.4g} "
              f"strongest_pc={PITCH_NAMES[int(np.argmax(display))]}({display.max():.1f}) "
              f"values={np.round(display, 1).tolist()}")

    # 1. Pitch-class map sanity: A4 (440 Hz) -> pitch class 9 (A).
    pc_map = build_pitch_class_map()
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    a4_bin = int(np.argmin(np.abs(freqs - 440.0)))
    check("440 Hz bin maps to pitch class A (9)", pc_map[a4_bin] == 9)

    # 2. DC and out-of-range bins are excluded.
    check("DC bin (0 Hz) is excluded", pc_map[0] == -1)
    high_bin = int(np.searchsorted(freqs, MAX_FREQ_HZ + 500))
    if high_bin < len(pc_map):
        check("bin above MAX_FREQ_HZ is excluded", pc_map[high_bin] == -1)

    # 3. Raw samples change / FFT magnitudes vary across bins for a real tone.
    tone_a = _make_test_tone([440.0], duration=1.0)
    buf = np.zeros(FFT_SIZE, dtype=np.float32)
    buf[:] = tone_a[-FFT_SIZE:]
    spectrum = np.fft.rfft(buf * hann(FFT_SIZE, sym=False))
    power = np.abs(spectrum) ** 2
    check("FFT power varies across bins", np.ptp(power) > 1e-6)

    # Shared ambient-noise warmup so the adaptive floors are established
    # before every subsequent test, like a real fair environment.
    def warmed_up_analyzer(noise_amp=0.03, warmup_s=15.0, seed=1):
        a = ChromaAnalyzer()
        _run_analyzer(_make_test_noise(warmup_s, amplitude=noise_amp, seed=seed), a)
        return a

    # 4. Single A (440 Hz) -> chroma peak at pitch class 9 (A).
    a = warmed_up_analyzer()
    disp_a = _run_analyzer(_make_test_tone([440.0]), a)
    diag("single A440", a, disp_a)
    check("chroma has exactly 12 finite values (A test)",
          disp_a.shape == (12,) and np.all(np.isfinite(disp_a)))
    check("single A440 tone peaks at pitch class A (9)", int(np.argmax(disp_a)) == 9)

    # 5. Single C (261.63 Hz, C4) -> chroma peak at pitch class 0 (C).
    a = warmed_up_analyzer()
    disp_c = _run_analyzer(_make_test_tone([261.63]), a)
    diag("single C4", a, disp_c)
    check("single C4 tone peaks at pitch class C (0)", int(np.argmax(disp_c)) == 0)
    check("chroma values are not all identical (C test)", np.ptp(disp_c) > 1e-6)

    # 6. C + E -> both C (0) and E (4) are the two strongest pitch classes.
    a = warmed_up_analyzer()
    disp_ce = _run_analyzer(_make_test_tone([261.63, 329.63]), a)
    diag("C+E", a, disp_ce)
    top2 = set(np.argsort(disp_ce)[-2:])
    check("C+E tone has C and E as the two strongest pitch classes", top2 == {0, 4})

    # 7. C + E + G -> C, E, G are the three strongest pitch classes.
    a = warmed_up_analyzer()
    disp_ceg = _run_analyzer(_make_test_tone([261.63, 329.63, 392.00]), a)
    diag("C+E+G", a, disp_ceg)
    top3 = set(np.argsort(disp_ceg)[-3:])
    check("C+E+G tone has C, E, G as the three strongest pitch classes", top3 == {0, 4, 7})

    # 8. Silence -> near-zero display everywhere.
    a = warmed_up_analyzer()
    disp_silence = _run_analyzer(np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32), a)
    diag("silence", a, disp_silence)
    check("silence produces near-zero display", np.max(disp_silence) < 5.0)
    check("silence display is exactly 12 finite, bounded values",
          disp_silence.shape == (12,) and np.all(np.isfinite(disp_silence))
          and np.all(disp_silence >= 0.0) and np.all(disp_silence <= 100.0))

    # 9. Broadband noise (speech/room noise stand-in) must NOT produce a wall
    #    of strong bars, across a range of noise loudnesses.
    for noise_amp in (0.01, 0.05, 0.1, 0.3):
        a = warmed_up_analyzer(noise_amp=noise_amp)
        disp_noise = _run_analyzer(
            _make_test_noise(3.0, amplitude=noise_amp, seed=99), a
        )
        strong_bars = int(np.sum(disp_noise > 30.0))
        diag(f"broadband noise amp={noise_amp}", a, disp_noise)
        check(f"broadband noise (amp={noise_amp}) does not light up many bars (<=3 above 30)",
              strong_bars <= 3)
        check(f"broadband noise (amp={noise_amp}) stays well below a real note (max < 60)",
              disp_noise.max() < 60.0)

    # 10. Note change: playing C then G updates the correct bars.
    a = warmed_up_analyzer()
    _run_analyzer(_make_test_tone([261.63], duration=2.0), a)
    disp_after_c = a.process_block(_make_test_tone([261.63], duration=2.0)[-BLOCK_SIZE:])
    disp_after_g = _run_analyzer(_make_test_tone([392.00], duration=2.0), a)
    diag("after switching C->G", a, disp_after_g)
    check("switching from C to G makes G the strongest pitch class", int(np.argmax(disp_after_g)) == 7)

    # 11. No NaN/Inf can survive a pathological (all-zero-signal) block sequence.
    a = ChromaAnalyzer()
    disp_zero = _run_analyzer(np.zeros(FFT_SIZE * 2, dtype=np.float32), a)
    check("degenerate all-zero input yields finite, bounded display",
          np.all(np.isfinite(disp_zero)) and np.all(disp_zero >= 0.0) and np.all(disp_zero <= 100.0))

    # --- Tier 3: chord recognition ---

    def confirmed_chord_for(signal, noise_amp=0.03, warmup_s=15.0, seed=1):
        """Run the same analyzer -> UI-smoothing -> ChordDetector chain the app uses."""
        analyzer = ChromaAnalyzer()
        _run_analyzer(_make_test_noise(warmup_s, amplitude=noise_amp, seed=seed), analyzer)
        detector = ChordDetector()
        smooth = np.zeros(12, dtype=np.float64)
        chord, confidence = None, 0.0
        for start in range(0, len(signal) - BLOCK_SIZE + 1, BLOCK_SIZE):
            disp = analyzer.process_block(signal[start:start + BLOCK_SIZE])
            smooth = SMOOTHING_ALPHA * disp + (1.0 - SMOOTHING_ALPHA) * smooth
            chord, confidence = detector.update(smooth)
        return chord, confidence, smooth

    def chord_diag(label, chord, confidence, smooth):
        name = "None" if chord is None else f"{PITCH_NAMES[chord[0]]} {chord[1]}"
        print(f"    -- diag [{label}] chord={name} confidence={confidence:.2f} "
              f"chroma={np.round(smooth, 1).tolist()}")

    # 12. C + E + G -> C major.
    chord, conf, smooth = confirmed_chord_for(_make_test_tone([261.63, 329.63, 392.00], duration=3.0))
    chord_diag("C+E+G", chord, conf, smooth)
    check("C+E+G is recognized as C major", chord == (0, "major"))
    check("chord confidence is finite and bounded", np.isfinite(conf) and 0.0 <= conf <= 1.0)

    # 13. A + C + E -> A minor.
    chord, conf, smooth = confirmed_chord_for(_make_test_tone([220.00, 261.63, 329.63], duration=3.0))
    chord_diag("A+C+E", chord, conf, smooth)
    check("A+C+E is recognized as A minor", chord == (9, "minor"))

    # 14. G + B + D -> G major.
    chord, conf, smooth = confirmed_chord_for(_make_test_tone([196.00, 246.94, 293.66], duration=3.0))
    chord_diag("G+B+D", chord, conf, smooth)
    check("G+B+D is recognized as G major", chord == (7, "major"))

    # 15. A single note should usually show no chord (not enough notes for a triad).
    chord, conf, smooth = confirmed_chord_for(_make_test_tone([261.63], duration=3.0))
    chord_diag("single C4", chord, conf, smooth)
    check("a single note does not get called a chord", chord is None)

    # 16. Room noise / speech-like broadband noise should show no chord.
    chord, conf, smooth = confirmed_chord_for(_make_test_noise(3.0, amplitude=0.1, seed=42), noise_amp=0.1)
    chord_diag("broadband noise", chord, conf, smooth)
    check("broadband noise does not get called a chord", chord is None)

    # 17. An ambiguous / dissonant cluster (no clean triad shape) should show no chord.
    chord, conf, smooth = confirmed_chord_for(
        _make_test_tone([261.63, 277.18, 293.66], duration=3.0)  # C, C#, D -- a tone cluster
    )
    chord_diag("ambiguous cluster (C, C#, D)", chord, conf, smooth)
    check("an ambiguous tone cluster does not get called a chord", chord is None)

    # 18. A bare fifth (root + 5th, no 3rd) is ambiguous between major/minor -> no chord.
    chord, conf, smooth = confirmed_chord_for(_make_test_tone([261.63, 392.00], duration=3.0))
    chord_diag("bare fifth (C, G)", chord, conf, smooth)
    check("a bare fifth (no 3rd) does not get called a chord", chord is None)

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
