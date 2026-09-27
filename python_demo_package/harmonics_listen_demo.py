#!/usr/bin/env python3
"""ECA fair demo -- "Harmonics Demo" (interactive, multi-stage).

Record a short, sustained note (e.g. a cello note), then walk through four
stages that build up the idea "one note is actually many frequencies":

  Stage 1 -- SEE THE HARMONICS
      Waveform + frequency spectrum of the recording, with the detected
      fundamental and each harmonic peak labeled (1f, 2f, 3f, ...) together
      with its frequency and approximate note name.

  Stage 2 -- LISTEN TO INDIVIDUAL HARMONICS
      One row per detected harmonic ("3rd harmonic -- 1320 Hz -- E6  [Play]").
      Pressing Play reconstructs and plays ONLY that single harmonic as a
      pure sinusoid (its own detected frequency/amplitude/phase) -- never the
      original recording with other components merely turned down.

  Stage 3 -- BUILD THE SOUND
      Buttons to reconstruct from a growing subset of harmonics
      (fundamental only -> 1f+2f -> ... -> all detected harmonics) plus the
      original recording, so the timbre audibly grows richer. The selected
      harmonics are highlighted on the spectrum.

  Stage 4 -- EXPLAIN THE CONNECTION
      A short, visual "ONE NOTE -> FUNDAMENTAL + HARMONICS -> TIMBRE" summary
      plus the actual detected f0, 2f0, 3f0, ... frequencies.

Pipeline (unchanged from the original harmonics_listen_demo.py):
    RECORD (fixed duration)
        -> windowed FFT of the recording
        -> robust fundamental detection (harmonic product spectrum)
        -> peak-picking near n x f0 for harmonics 2..N, with a noise-relative
           amplitude threshold to reject weak/spurious peaks
        -> reconstruction sums sinusoids using each selected harmonic's own
           detected frequency, amplitude, and phase (never zeroed FFT bins)
        -> the recording's amplitude envelope is applied so playback is
           time-limited to the note, not a sustained synth drone
        -> safe peak normalization; nothing plays automatically

This file is completely standalone: it does not import or modify
live_audio_demo.py, live_chroma_demo.py, live_harmonics_demo.py, or any
other existing file in this project.

Run the app:
    python harmonics_listen_demo.py

Run the internal self-tests (no GUI, no microphone, no audio playback):
    python harmonics_listen_demo.py --selftest
"""

import sys

import numpy as np
import sounddevice as sd
from scipy.signal.windows import hann
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
RECORD_SECONDS = 3.0
FFT_SIZE = 1 << 16          # long window -> fine frequency resolution for HPS

# --- fundamental detection (harmonic product spectrum) ---
HPS_DOWNSAMPLE = 5          # number of harmonics multiplied together
F0_MIN_HZ = 60.0            # low cello ~C2 = 65 Hz
F0_MAX_HZ = 1000.0

# --- harmonic peak picking ---
MAX_HARMONIC = 8            # highest harmonic number searched for (1 = fundamental)
HARMONIC_SEARCH_TOL = 0.06  # fractional window around n*f0 to search for a local peak
HARMONIC_MIN_REL_DB = -40.0 # a harmonic must be within this many dB of the fundamental
NOISE_FLOOR_MARGIN_DB = 10.0  # ...and this many dB above the spectrum's noise floor

# --- silence / noise guard ---
MIN_RMS_FOR_SIGNAL = 0.003  # recordings quieter than this are treated as silence

# --- playback safety ---
PLAYBACK_PEAK = 0.85        # normalize every playback buffer to this peak amplitude

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Stage 3 "build the sound" harmonic sets. "ORIGINAL" is a sentinel meaning
# "play the recording itself", not a reconstruction.
BUILD_SETS = [
    ("Fundamental only", [1]),
    ("1f + 2f", [1, 2]),
    ("1f + 2f + 3f", [1, 2, 3]),
    ("1f–4f", [1, 2, 3, 4]),
    ("1f–5f", [1, 2, 3, 4, 5]),
    ("All detected harmonics", None),
    ("Original recording", "ORIGINAL"),
]

ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def ordinal(n):
    return ORDINALS.get(n, f"{n}th")


def freq_to_note_name(freq_hz):
    """Nearest equal-tempered note name (A440 reference), e.g. 440.0 -> 'A4'."""
    if freq_hz is None or freq_hz <= 0 or not np.isfinite(freq_hz):
        return "?"
    midi = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    midi_round = int(round(midi))
    name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    return f"{name}{octave}"


# ------------------------------------------------------------------------
# Pure DSP (no Qt / sounddevice dependency) -- unit-testable in isolation.
# ------------------------------------------------------------------------

def compute_envelope(samples, sample_rate, smooth_ms=15.0):
    """Amplitude envelope via rectify + moving-average smoothing."""
    win = max(1, int(sample_rate * smooth_ms / 1000.0))
    rectified = np.abs(samples)
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.convolve(rectified, kernel, mode="same")
    return env


def detect_fundamental(samples, sample_rate, fft_size=FFT_SIZE):
    """Harmonic product spectrum: robust to a strong 2nd harmonic overpowering
    a weak fundamental (common on stringed instruments)."""
    n = min(fft_size, len(samples))
    if n < 1024:
        return None, None, None

    window = hann(n, sym=False)
    spectrum = np.fft.rfft(samples[:n] * window)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    hps = mag.copy()
    for h in range(2, HPS_DOWNSAMPLE + 1):
        decim = mag[::h]
        hps[:len(decim)] *= decim
        hps[len(decim):] = 0.0

    valid = (freqs >= F0_MIN_HZ) & (freqs <= F0_MAX_HZ)
    if not np.any(valid):
        return None, mag, freqs

    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(hps[valid_idx])]

    f0 = _parabolic_peak_freq(mag, freqs, best)
    return f0, mag, freqs


def _parabolic_peak_freq(mag, freqs, idx):
    if idx <= 0 or idx >= len(mag) - 1:
        return freqs[idx]
    y0, y1, y2 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return freqs[idx]
    delta = 0.5 * (y0 - y2) / denom
    delta = np.clip(delta, -1.0, 1.0)
    bin_hz = freqs[1] - freqs[0]
    return freqs[idx] + delta * bin_hz


def _spectrum_noise_floor(mag):
    """Median magnitude as a robust noise-floor estimate, ignoring DC."""
    if len(mag) < 2:
        return 0.0
    return float(np.median(mag[1:]))


def detect_harmonics(samples, sample_rate, f0, fft_size=FFT_SIZE,
                      max_harmonic=MAX_HARMONIC):
    """Find amplitude + phase for harmonics 1..max_harmonic near n*f0.

    Returns a dict {harmonic_number: (freq_hz, amplitude, phase_rad)}.
    Weak or absent harmonics (below the noise floor or too far below the
    fundamental) are simply omitted, never invented.
    """
    n = min(fft_size, len(samples))
    if f0 is None or n < 1024:
        return {}

    window = hann(n, sym=False)
    spectrum = np.fft.rfft(samples[:n] * window)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    bin_hz = freqs[1] - freqs[0]

    noise_floor = _spectrum_noise_floor(mag)
    nyquist = freqs[-1]

    harmonics = {}
    fundamental_mag = None

    for h in range(1, max_harmonic + 1):
        target = f0 * h
        if target >= nyquist * 0.98:
            break

        window_hz = max(target * HARMONIC_SEARCH_TOL, bin_hz * 2)
        lo = target - window_hz
        hi = target + window_hz
        idx_lo = max(0, int(np.searchsorted(freqs, lo)))
        idx_hi = min(len(freqs) - 1, int(np.searchsorted(freqs, hi)))
        if idx_hi <= idx_lo:
            continue

        local = mag[idx_lo:idx_hi + 1]
        peak_local = int(np.argmax(local))
        peak_idx = idx_lo + peak_local
        peak_mag = mag[peak_idx]

        if peak_mag <= 0 or peak_mag < noise_floor * (10 ** (NOISE_FLOOR_MARGIN_DB / 20.0)):
            continue  # too weak relative to the noise floor -- likely not a real partial

        if h == 1:
            fundamental_mag = peak_mag
        elif fundamental_mag is not None:
            rel_db = 20 * np.log10(peak_mag / fundamental_mag + 1e-12)
            if rel_db < HARMONIC_MIN_REL_DB:
                continue  # too weak relative to the fundamental

        freq_hz = _parabolic_peak_freq(mag, freqs, peak_idx)
        harmonics[h] = (float(freq_hz), float(peak_mag), float(phase[peak_idx]))

    return harmonics


def reconstruct(harmonics, harmonic_numbers, num_samples, sample_rate,
                 envelope=None):
    """Sum sinusoids for the requested harmonic numbers, each using its own
    detected frequency/amplitude/phase, then apply the recording's envelope
    so the result is time-limited to the note rather than a sustained tone.
    """
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    out = np.zeros(num_samples, dtype=np.float64)

    used = [h for h in harmonic_numbers if h in harmonics]
    if not used:
        return out, used

    max_amp = max(harmonics[h][1] for h in used)
    if max_amp <= 0:
        return out, used

    for h in used:
        freq_hz, amp, phase = harmonics[h]
        out += (amp / max_amp) * np.sin(2 * np.pi * freq_hz * t + phase)

    out /= len(used)

    if envelope is not None:
        env = envelope[:num_samples]
        if len(env) < num_samples:
            env = np.pad(env, (0, num_samples - len(env)))
        env_max = env.max()
        if env_max > 0:
            out *= env / env_max

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, used


def normalize_safely(samples, peak=PLAYBACK_PEAK):
    """Peak-normalize to a comfortable level; silence stays silent; never clips."""
    if samples.size == 0:
        return samples
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.max(np.abs(samples))
    if m < 1e-9:
        return np.zeros_like(samples)
    return (samples / m) * peak


def is_silence_or_noise(samples, harmonics):
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    if rms < MIN_RMS_FOR_SIGNAL:
        return True
    if 1 not in harmonics:
        return True
    return False


def decimate_for_plot(samples, max_points=4000):
    """Downsample a waveform for fast, lightweight plotting."""
    n = len(samples)
    if n <= max_points:
        return np.arange(n), samples
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return idx, samples[idx]


DARK_STYLESHEET = """
QWidget { background-color: #1e1e24; color: #f0f0f0; font-family: Helvetica, Arial, sans-serif; }
QPushButton {
    background-color: #33333d; color: #f0f0f0; border: 1px solid #4a4a58;
    border-radius: 6px; padding: 6px 10px; font-size: 15px;
}
QPushButton:hover { background-color: #3f3f4c; }
QPushButton:disabled { color: #777; background-color: #26262c; }
QPushButton:checked { background-color: #e67e22; color: #1e1e24; font-weight: bold; }
QTabWidget::pane { border: 1px solid #4a4a58; }
QTabBar::tab {
    background: #26262c; color: #ccc; padding: 10px 16px; font-size: 15px;
}
QTabBar::tab:selected { background: #33333d; color: #fff; font-weight: bold; }
QScrollArea { border: none; }
QLabel { color: #f0f0f0; }
"""


# ------------------------------------------------------------------------
# Qt application
# ------------------------------------------------------------------------

class HarmonicsListenDemo(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmonics Demo")
        self.resize(980, 780)
        self.setStyleSheet(DARK_STYLESHEET)

        self.recording = None          # np.ndarray, float64, mono
        self.envelope = None
        self.f0 = None
        self.harmonics = {}            # {harmonic_number: (freq, amp, phase)}
        self._is_recording = False

        pg.setConfigOption("background", "#1e1e24")
        pg.setConfigOption("foreground", "#f0f0f0")

        self._build_ui()

    # -- UI construction --------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("HARMONICS DEMO")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 6px;")
        layout.addWidget(title)

        record_row = QtWidgets.QHBoxLayout()
        self.record_btn = QtWidgets.QPushButton(f"RECORD {RECORD_SECONDS:.0f}s")
        self.record_btn.setMinimumHeight(64)
        self.record_btn.setStyleSheet(
            "font-size: 20px; font-weight: bold; background-color: #c0392b; color: white;"
        )
        self.record_btn.clicked.connect(self.on_record_clicked)
        record_row.addWidget(self.record_btn)
        layout.addLayout(record_row)

        self.status_label = QtWidgets.QLabel("Press RECORD, then play a sustained note.")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #bbb; padding: 4px;")
        layout.addWidget(self.status_label)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #e67e22; padding: 2px;")
        layout.addWidget(self.info_label)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self._build_stage1()
        self._build_stage2()
        self._build_stage3()
        self._build_stage4()

        self.tabs.addTab(self.stage1_widget, "1. See the Harmonics")
        self.tabs.addTab(self.stage2_widget, "2. Listen to Individual Harmonics")
        self.tabs.addTab(self.stage3_widget, "3. Build the Sound")
        self.tabs.addTab(self.stage4_widget, "4. The Connection")

        self._set_stages_enabled(False)

    def _set_stages_enabled(self, enabled):
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, enabled)

    # -- Stage 1: SEE THE HARMONICS ---------------------------------------

    def _build_stage1(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        heading = QtWidgets.QLabel("ONE NOTE IS MANY FREQUENCIES")
        heading.setAlignment(QtCore.Qt.AlignCenter)
        heading.setStyleSheet("font-size: 19px; font-weight: bold; padding: 6px;")
        layout.addWidget(heading)

        self.fundamental_label = QtWidgets.QLabel("Fundamental: --")
        self.fundamental_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fundamental_label.setStyleSheet("font-size: 16px; padding: 4px;")
        layout.addWidget(self.fundamental_label)

        self.waveform_plot = pg.PlotWidget(title="Recorded waveform")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setMinimumHeight(160)
        self.waveform_curve = self.waveform_plot.plot([], [], pen=pg.mkPen("#3498db", width=1))
        layout.addWidget(self.waveform_plot)

        self.spectrum_plot = pg.PlotWidget(title="Frequency spectrum (harmonics labeled)")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Amplitude")
        self.spectrum_plot.setMinimumHeight(300)
        self.spectrum_curve = self.spectrum_plot.plot([], [], pen=pg.mkPen("#3498db", width=1))
        self.spectrum_markers = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#e67e22"), pen=pg.mkPen(None))
        self.spectrum_plot.addItem(self.spectrum_markers)
        self._spectrum_labels = []
        layout.addWidget(self.spectrum_plot)

        self.stage1_widget = w

    # -- Stage 2: LISTEN TO INDIVIDUAL HARMONICS --------------------------

    def _build_stage2(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)

        heading = QtWidgets.QLabel("LISTEN TO INDIVIDUAL HARMONICS")
        heading.setAlignment(QtCore.Qt.AlignCenter)
        heading.setStyleSheet("font-size: 19px; font-weight: bold; padding: 6px;")
        outer.addWidget(heading)

        sub = QtWidgets.QLabel(
            "Each button plays ONLY that one frequency component, isolated from the rest of the note."
        )
        sub.setAlignment(QtCore.Qt.AlignCenter)
        sub.setStyleSheet("font-size: 13px; color: #bbb; padding: 2px 8px 10px 8px;")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.stage2_list_widget = QtWidgets.QWidget()
        self.stage2_list_layout = QtWidgets.QVBoxLayout(self.stage2_list_widget)
        self.stage2_list_layout.addStretch(1)
        scroll.setWidget(self.stage2_list_widget)
        outer.addWidget(scroll, stretch=1)

        self.stage2_widget = w

    def _rebuild_stage2_rows(self):
        layout = self.stage2_list_layout
        # leave the trailing stretch (last item) in place; only remove row widgets
        while layout.count() > 1:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for h in sorted(self.harmonics):
            freq_hz, _amp, _phase = self.harmonics[h]
            note = freq_to_note_name(freq_hz)
            row = QtWidgets.QHBoxLayout()
            text = f"{ordinal(h)} harmonic — {freq_hz:.0f} Hz — {note}"
            if h == 1:
                text = f"Fundamental (1st harmonic) — {freq_hz:.0f} Hz — {note}"
            label = QtWidgets.QLabel(text)
            label.setStyleSheet("font-size: 16px;")
            btn = QtWidgets.QPushButton("▶ Play")
            btn.setFixedWidth(90)
            btn.clicked.connect(lambda _checked, hn=h: self.on_play_single_harmonic(hn))
            row.addWidget(label, stretch=1)
            row.addWidget(btn)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            layout.insertWidget(layout.count() - 1, container)

    # -- Stage 3: BUILD THE SOUND -----------------------------------------

    def _build_stage3(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)

        heading = QtWidgets.QLabel("BUILD THE SOUND")
        heading.setAlignment(QtCore.Qt.AlignCenter)
        heading.setStyleSheet("font-size: 19px; font-weight: bold; padding: 6px;")
        outer.addWidget(heading)

        sub = QtWidgets.QLabel("Add harmonics one at a time and listen to the timbre grow richer.")
        sub.setAlignment(QtCore.Qt.AlignCenter)
        sub.setStyleSheet("font-size: 13px; color: #bbb; padding: 2px 8px 10px 8px;")
        outer.addWidget(sub)

        self.build_button_group = QtWidgets.QButtonGroup(self)
        self.build_button_group.setExclusive(True)
        self.build_buttons = {}
        btn_grid = QtWidgets.QGridLayout()
        for i, (label_text, spec) in enumerate(BUILD_SETS):
            btn = QtWidgets.QPushButton(label_text)
            btn.setCheckable(True)
            btn.setMinimumHeight(48)
            btn.clicked.connect(lambda _checked, s=spec, lt=label_text: self.on_build_selected(s, lt))
            self.build_button_group.addButton(btn)
            self.build_buttons[label_text] = btn
            btn_grid.addWidget(btn, i // 2, i % 2)
        outer.addLayout(btn_grid)

        self.build_spectrum_plot = pg.PlotWidget(title="Selected harmonics (highlighted)")
        self.build_spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.build_spectrum_plot.setLabel("left", "Amplitude")
        self.build_spectrum_plot.setMinimumHeight(260)
        self.build_spectrum_curve = self.build_spectrum_plot.plot([], [], pen=pg.mkPen("#3498db", width=1))
        self.build_markers_inactive = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#555566"), pen=pg.mkPen(None))
        self.build_markers_active = pg.ScatterPlotItem(size=13, brush=pg.mkBrush("#2ecc71"), pen=pg.mkPen(None))
        self.build_spectrum_plot.addItem(self.build_markers_inactive)
        self.build_spectrum_plot.addItem(self.build_markers_active)
        outer.addWidget(self.build_spectrum_plot, stretch=1)

        self.stage3_widget = w

    # -- Stage 4: EXPLAIN THE CONNECTION -----------------------------------

    def _build_stage4(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.addStretch(1)

        def flow_label(text, big=False, color="#f0f0f0"):
            lab = QtWidgets.QLabel(text)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            size = 26 if big else 20
            lab.setStyleSheet(f"font-size: {size}px; font-weight: bold; color: {color}; padding: 8px;")
            return lab

        def arrow():
            lab = QtWidgets.QLabel("↓")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet("font-size: 26px; color: #e67e22;")
            return lab

        outer.addWidget(flow_label("ONE NOTE", big=True))
        outer.addWidget(arrow())
        outer.addWidget(flow_label("FUNDAMENTAL + HARMONICS", big=True, color="#3498db"))
        outer.addWidget(arrow())
        outer.addWidget(flow_label("TIMBRE", big=True, color="#2ecc71"))

        outer.addSpacing(20)

        self.stage4_freqs_label = QtWidgets.QLabel(
            "f₀, 2f₀, 3f₀, 4f₀, 5f₀ ... — record a note to see the actual values"
        )
        self.stage4_freqs_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stage4_freqs_label.setWordWrap(True)
        self.stage4_freqs_label.setStyleSheet("font-size: 17px; padding: 12px; color: #f0f0f0;")
        outer.addWidget(self.stage4_freqs_label)

        outer.addStretch(2)
        self.stage4_widget = w

    # -- Recording ----------------------------------------------------------

    def on_record_clicked(self):
        if self._is_recording:
            return
        self._is_recording = True
        self.record_btn.setEnabled(False)
        self.record_btn.setText("RECORDING...")
        self.status_label.setText("Recording...")
        self._set_stages_enabled(False)

        num_frames = int(RECORD_SECONDS * SAMPLE_RATE)
        try:
            audio = sd.rec(num_frames, samplerate=SAMPLE_RATE, channels=1, dtype="float64")
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._finish_recording(audio))
            timer.start(int(RECORD_SECONDS * 1000) + 150)
        except Exception as exc:
            self._is_recording = False
            self.record_btn.setEnabled(True)
            self.record_btn.setText(f"RECORD {RECORD_SECONDS:.0f}s")
            self.status_label.setText(f"Recording failed: {exc}")

    def _finish_recording(self, audio_buffer):
        sd.wait()
        self._is_recording = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText(f"RECORD {RECORD_SECONDS:.0f}s")

        samples = np.asarray(audio_buffer, dtype=np.float64).reshape(-1)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        self.recording = samples
        self._analyze_and_update_ui(samples)

    # -- Analysis + UI update -------------------------------------------

    def _analyze_and_update_ui(self, samples):
        self.envelope = compute_envelope(samples, SAMPLE_RATE)
        f0, mag, freqs = detect_fundamental(samples, SAMPLE_RATE)
        harmonics = detect_harmonics(samples, SAMPLE_RATE, f0) if f0 is not None else {}

        self.f0 = f0
        self.harmonics = harmonics
        self._last_mag = mag
        self._last_freqs = freqs

        for btn in self.build_buttons.values():
            btn.setChecked(False)

        if is_silence_or_noise(samples, harmonics):
            self.status_label.setText(
                "Recording was too quiet or had no clear pitch — try again with a "
                "sustained, clearly bowed note."
            )
            self.info_label.setText("")
            self.fundamental_label.setText("Fundamental: not detected")
            self._update_waveform_plot(samples)
            self._update_stage1_spectrum(mag, freqs, {})
            self._update_stage3_spectrum(mag, freqs, {}, set())
            self._rebuild_stage2_rows()
            self.stage4_freqs_label.setText(
                "No clear pitch detected — record a sustained note to see f₀, 2f₀, 3f₀ ..."
            )
            self._set_stages_enabled(True)
            self.build_buttons["Original recording"].setEnabled(True)
            for label_text, spec in BUILD_SETS:
                if spec != "ORIGINAL":
                    self.build_buttons[label_text].setEnabled(False)
            return

        for label_text, spec in BUILD_SETS:
            self.build_buttons[label_text].setEnabled(True)

        harmonic_desc = ", ".join(
            f"{h}f={harmonics[h][0]:.1f}Hz" for h in sorted(harmonics)
        )
        self.status_label.setText("Analysis complete.")
        self.info_label.setText(
            f"Fundamental: {f0:.1f} Hz ({freq_to_note_name(f0)})   |   Harmonics: {harmonic_desc}"
        )
        self.fundamental_label.setText(f"Fundamental: {f0:.1f} Hz ≈ {freq_to_note_name(f0)}")

        self._update_waveform_plot(samples)
        self._update_stage1_spectrum(mag, freqs, harmonics)
        self._update_stage3_spectrum(mag, freqs, harmonics, set())
        self._rebuild_stage2_rows()

        freq_terms = "   ".join(
            f"{h}f₀ = {harmonics[h][0]:.1f} Hz" for h in sorted(harmonics)
        )
        self.stage4_freqs_label.setText(freq_terms if freq_terms else "No harmonics detected.")

        self._set_stages_enabled(True)

    def _update_waveform_plot(self, samples):
        t = np.arange(len(samples)) / SAMPLE_RATE
        idx, ys = decimate_for_plot(samples)
        self.waveform_curve.setData(t[idx], ys)

    def _clear_spectrum_labels(self):
        for item in self._spectrum_labels:
            self.spectrum_plot.removeItem(item)
        self._spectrum_labels = []

    def _update_stage1_spectrum(self, mag, freqs, harmonics):
        self._clear_spectrum_labels()
        if mag is None or freqs is None:
            self.spectrum_curve.setData([], [])
            self.spectrum_markers.setData([], [])
            return

        display_max = min(len(freqs), int(np.searchsorted(freqs, F0_MAX_HZ * (MAX_HARMONIC + 1))))
        display_max = max(display_max, 10)
        self.spectrum_curve.setData(freqs[:display_max], mag[:display_max])

        if harmonics:
            xs = [harmonics[h][0] for h in sorted(harmonics)]
            ys = [harmonics[h][1] for h in sorted(harmonics)]
            self.spectrum_markers.setData(xs, ys)

            y_range = max(ys) if ys else 1.0
            for h in sorted(harmonics):
                freq_hz, amp, _phase = harmonics[h]
                note = freq_to_note_name(freq_hz)
                text = pg.TextItem(
                    text=f"{h}f\n{freq_hz:.0f} Hz\n{note}",
                    color="#f0f0f0", anchor=(0.5, 1.0),
                )
                text.setPos(freq_hz, amp + y_range * 0.05)
                self.spectrum_plot.addItem(text)
                self._spectrum_labels.append(text)
        else:
            self.spectrum_markers.setData([], [])

    def _update_stage3_spectrum(self, mag, freqs, harmonics, active_set):
        if mag is None or freqs is None:
            self.build_spectrum_curve.setData([], [])
            self.build_markers_inactive.setData([], [])
            self.build_markers_active.setData([], [])
            return

        display_max = min(len(freqs), int(np.searchsorted(freqs, F0_MAX_HZ * (MAX_HARMONIC + 1))))
        display_max = max(display_max, 10)
        self.build_spectrum_curve.setData(freqs[:display_max], mag[:display_max])

        inactive_x, inactive_y, active_x, active_y = [], [], [], []
        for h in sorted(harmonics):
            freq_hz, amp, _phase = harmonics[h]
            if h in active_set:
                active_x.append(freq_hz)
                active_y.append(amp)
            else:
                inactive_x.append(freq_hz)
                inactive_y.append(amp)
        self.build_markers_inactive.setData(inactive_x, inactive_y)
        self.build_markers_active.setData(active_x, active_y)

    # -- Playback ---------------------------------------------------------

    def _play(self, audio):
        audio = normalize_safely(audio)
        sd.play(audio, samplerate=SAMPLE_RATE)

    def on_play_single_harmonic(self, harmonic_number):
        if self.recording is None or harmonic_number not in self.harmonics:
            return
        audio, used = reconstruct(
            self.harmonics, [harmonic_number], len(self.recording), SAMPLE_RATE, self.envelope
        )
        if not used:
            return
        self._play(audio)

    def on_build_selected(self, spec, label_text):
        if self.recording is None:
            return

        if spec == "ORIGINAL":
            self._update_stage3_spectrum(self._last_mag, self._last_freqs, self.harmonics, set())
            self._play(self.recording)
            self.status_label.setText(f"Playing: {label_text}")
            return

        if not self.harmonics:
            self.status_label.setText("No harmonics detected in this recording.")
            return

        wanted = spec if spec is not None else sorted(self.harmonics)
        audio, used = reconstruct(
            self.harmonics, wanted, len(self.recording), SAMPLE_RATE, self.envelope
        )
        self._update_stage3_spectrum(self._last_mag, self._last_freqs, self.harmonics, set(used))
        if not used:
            self.status_label.setText("None of those harmonics were detected in this recording.")
            return
        self._play(audio)
        self.status_label.setText(f"Playing: {label_text}")


# ------------------------------------------------------------------------
# Self-tests: pure DSP, no GUI/microphone/audio playback.
# ------------------------------------------------------------------------

def _make_test_tone(f0, harmonics_amp, sample_rate=SAMPLE_RATE, duration=1.0, noise=0.0):
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    for h, amp in harmonics_amp.items():
        out += amp * np.sin(2 * np.pi * f0 * h * t)
    if noise > 0:
        out += np.random.default_rng(0).normal(0, noise, n)
    return out


def _selftest():
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        if not cond:
            failures.append(name)

    # 1. Fundamental detection on an A4-ish tone with a strong 2nd harmonic
    #    (the classic "missing fundamental" trap for naive peak-picking).
    tone = _make_test_tone(440.0, {1: 0.3, 2: 1.0, 3: 0.6, 4: 0.3, 5: 0.15})
    f0, mag, freqs = detect_fundamental(tone, SAMPLE_RATE)
    check(f"fundamental detected near 440Hz (got {f0:.2f})", f0 is not None and abs(f0 - 440.0) < 2.0)

    # 2. Harmonic detection finds the expected partials
    harmonics = detect_harmonics(tone, SAMPLE_RATE, f0)
    check("harmonics 1-5 detected", all(h in harmonics for h in (1, 2, 3, 4, 5)))
    check("harmonic 6 (not present) is not spuriously detected", 6 not in harmonics)

    # 3. Note-name labeling matches the spec's worked example (A4 = 440 Hz)
    check("freq_to_note_name(440) == A4", freq_to_note_name(440.0) == "A4")
    check("freq_to_note_name(880) == A5", freq_to_note_name(880.0) == "A5")
    check("freq_to_note_name(1320) == E6", freq_to_note_name(1320.0) == "E6")
    check("freq_to_note_name(1760) == A6", freq_to_note_name(1760.0) == "A6")
    check("freq_to_note_name(2200) == C#7", freq_to_note_name(2200.0) == "C#7")

    # 4. Labels are computed from the DETECTED fundamental, not assumed 440 Hz
    detuned = _make_test_tone(432.0, {1: 1.0, 2: 0.5, 3: 0.3})
    d_f0, _, _ = detect_fundamental(detuned, SAMPLE_RATE)
    d_harmonics = detect_harmonics(detuned, SAMPLE_RATE, d_f0)
    expected_h3_note = freq_to_note_name(432.0 * 3)
    check(
        f"3rd harmonic of a 432Hz note is labeled from detected freq (expected {expected_h3_note})",
        3 in d_harmonics and freq_to_note_name(d_harmonics[3][0]) == expected_h3_note,
    )

    # 5. Reconstruction: a single harmonic contains ONLY that harmonic
    env = compute_envelope(tone, SAMPLE_RATE)
    recon_h1, used_h1 = reconstruct(harmonics, [1], len(tone), SAMPLE_RATE, env)
    recon_h3, used_h3 = reconstruct(harmonics, [3], len(tone), SAMPLE_RATE, env)
    check("single-harmonic reconstruction (h=1) uses exactly [1]", used_h1 == [1])
    check("single-harmonic reconstruction (h=3) uses exactly [3]", used_h3 == [3])
    check("h=1 and h=3 single-harmonic reconstructions differ", not np.allclose(recon_h1, recon_h3))

    # 6. Build-the-sound progression changes as harmonics are added
    recon_full, used_full = reconstruct(harmonics, [1, 2, 3, 4, 5], len(tone), SAMPLE_RATE, env)
    check("full reconstruction differs from fundamental-only", not np.allclose(recon_h1, recon_full))
    orig_norm = normalize_safely(tone)
    fund_norm = normalize_safely(recon_h1)
    full_norm = normalize_safely(recon_full)
    dist_full = np.sqrt(np.mean((orig_norm - full_norm) ** 2))
    dist_fund = np.sqrt(np.mean((orig_norm - fund_norm) ** 2))
    check("all-harmonics reconstruction is closer to original than fundamental-only", dist_full < dist_fund)

    # 7. Normalization avoids clipping, preserves silence, and stays finite
    loud = normalize_safely(recon_full * 100)
    check("normalized playback peak <= safety level", np.max(np.abs(loud)) <= PLAYBACK_PEAK + 1e-6)
    silent = normalize_safely(np.zeros(1000))
    check("normalizing silence stays silent", np.allclose(silent, 0.0))
    check("reconstruction output is finite", np.all(np.isfinite(recon_full)))

    # 8. Pure noise / silence is flagged, not misreported as a pitched note
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.001, int(SAMPLE_RATE * 0.5))
    n_f0, n_mag, n_freqs = detect_fundamental(noise, SAMPLE_RATE)
    n_harmonics = detect_harmonics(noise, SAMPLE_RATE, n_f0) if n_f0 is not None else {}
    check("near-silent noise flagged as silence/noise", is_silence_or_noise(noise, n_harmonics))

    true_silence = np.zeros(int(SAMPLE_RATE * 0.5))
    check("true silence flagged as silence/noise", is_silence_or_noise(true_silence, {}))
    check("clean tone is not flagged as silence/noise", not is_silence_or_noise(tone, harmonics))

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All self-tests passed.")
    return 0


def main():
    if "--selftest" in sys.argv:
        sys.exit(_selftest())

    app = QtWidgets.QApplication(sys.argv)
    window = HarmonicsListenDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
