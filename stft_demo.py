#!/usr/bin/env python3
"""ECA fair demo -- STFT (Short-Time Fourier Transform) intuition builder.

The whole point of this demo is one sentence:

    STFT = repeated FFTs of short overlapping time windows

Pipeline shown end-to-end, exactly as it is computed:

    audio -> overlapping window -> window function -> FFT ->
    magnitude/power -> dB -> one spectrogram column

Three tabs build the intuition in layers:

    Tab 1  LIVE STFT             -- waveform on top, scrolling spectrogram
                                     below it, built one column at a time.
    Tab 2  WINDOW EXPLAINER       -- drag a highlighted time window and watch
                                     its waveform slice, its FFT, and its
                                     spectrogram column update together.
    Tab 3  WINDOW SIZE COMPARE    -- the SAME audio through two window sizes
                                     side by side, at a fixed physical size,
                                     so the time/frequency resolution
                                     trade-off is visible rather than implied.

All four window sizes (512/1024/2048/4096) are analyzed continuously and
concurrently on the audio thread; switching which one is displayed is just a
GUI-thread choice of which precomputed channel to draw, so it updates
instantly with no audio glitch.

Audio capture and every FFT/dB computation run on the PortAudio callback
thread behind a lock; the Qt GUI thread only reads snapshots and redraws, so
the UI never blocks on audio work.

Standalone from live_audio_demo.py (Tier 1), live_chroma_demo.py (Tier 2),
and live_harmonics_demo.py (Tier 2b) -- none of those files are touched by
this one.

Run the app:
    python stft_demo.py

Run the internal self-tests (no GUI, no microphone, no audio playback):
    python stft_demo.py --selftest
"""

import sys
import threading

import numpy as np
import sounddevice as sd
from scipy.signal.windows import hann
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
BLOCK_SIZE = 256                 # samples per audio callback (~5.8 ms)
HISTORY_SECONDS = 5.0            # waveform / spectrogram time span shown on screen
HIST_SAMPLES = int(HISTORY_SECONDS * SAMPLE_RATE)

WINDOW_SIZES = [512, 1024, 2048, 4096]
DEFAULT_WINDOW_SIZE = 1024
HOP_DIVISOR = 4                  # 75% overlap, the standard STFT choice

DISPLAY_MAX_FREQ_HZ = 5000.0     # covers fundamentals + overtones of voice/strings
DB_FLOOR = -80.0                 # fixed color/axis floor -- never autoscaled
DB_CEIL = 0.0                    # 0 dB == a full-scale sinusoid (see ref_power)

UI_REFRESH_MS = 33               # ~30 fps


# --------------------------------------------------------------------------
# Color map (self-contained -- no matplotlib dependency)
# --------------------------------------------------------------------------

def build_spectrogram_colormap():
    """Black -> purple -> magenta -> orange -> pale yellow, like a classic
    'inferno'-style spectrogram palette, built from a handful of stops so no
    extra dependency is required."""
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [
        (5, 5, 10, 255),
        (55, 15, 90, 255),
        (170, 30, 90, 255),
        (240, 120, 20, 255),
        (255, 250, 190, 255),
    ]
    return pg.ColorMap(pos, colors)


# --------------------------------------------------------------------------
# One STFT "channel" = one window size, continuously maintained
# --------------------------------------------------------------------------

class STFTChannel:
    """Everything needed to turn audio into dB-scaled STFT columns at one
    fixed window size: the window function, the FFT bookkeeping, a fixed
    dB reference so 0 dB always means "a full-scale sinusoid" (this is what
    keeps color scaling stable across window sizes and across time), and a
    scrolling spectrogram buffer covering HISTORY_SECONDS.
    """

    def __init__(self, window_size, sample_rate=SAMPLE_RATE):
        self.window_size = window_size
        self.hop = max(1, window_size // HOP_DIVISOR)
        self.window_func = hann(window_size, sym=False).astype(np.float32)

        freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
        self.display_mask = freqs <= DISPLAY_MAX_FREQ_HZ
        self.freqs_disp = freqs[self.display_mask]
        self.bin_width_hz = sample_rate / window_size

        # a full-scale (amplitude 1.0) sinusoid, Hann-windowed, produces an
        # rfft bin magnitude of ~coherent_gain/2; squaring gives the power
        # reference so that signal reaches ~0 dB regardless of window size.
        coherent_gain = float(np.sum(self.window_func))
        self.ref_power = max((coherent_gain / 2.0) ** 2, 1e-12)

        self.n_cols = max(2, HIST_SAMPLES // self.hop)
        self.spec = np.full((self.n_cols, self.freqs_disp.size), DB_FLOOR, dtype=np.float32)
        self.since_hop = 0

    def _segment_to_db(self, seg):
        windowed = seg * self.window_func
        spectrum = np.fft.rfft(windowed)
        power = np.abs(spectrum) ** 2
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        db = 10.0 * np.log10(power / self.ref_power + 1e-12)
        db = np.nan_to_num(db, nan=DB_FLOOR, posinf=DB_CEIL, neginf=DB_FLOOR)
        return np.clip(db, DB_FLOOR, DB_CEIL)

    def push_column(self, seg):
        """seg must be exactly window_size samples, most-recent-last."""
        db_disp = self._segment_to_db(seg)[self.display_mask]
        self.spec = np.roll(self.spec, -1, axis=0)
        self.spec[-1, :] = db_disp

    def analyze_segment(self, seg):
        """One-off FFT of an arbitrary segment (padded/trimmed to
        window_size), used by the window-explainer view. Does not touch the
        scrolling spectrogram buffer."""
        if len(seg) < self.window_size:
            seg = np.pad(seg, (self.window_size - len(seg), 0))
        else:
            seg = seg[-self.window_size:]
        return self._segment_to_db(seg)[self.display_mask]


# --------------------------------------------------------------------------
# Audio engine -- owns the microphone stream and every STFTChannel
# --------------------------------------------------------------------------

class AudioEngine(QtCore.QObject):
    """Runs the whole DSP pipeline on the PortAudio callback thread. The Qt
    GUI thread never touches audio directly -- it only calls the getter
    methods below, which take a short lock and return copies."""

    error = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._lock = threading.Lock()
        self.wave_buf = np.zeros(HIST_SAMPLES, dtype=np.float32)
        self.channels = {size: STFTChannel(size, sample_rate) for size in WINDOW_SIZES}
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
        # Runs on PortAudio's own thread -- never touches Qt widgets.
        samples = indata[:, 0].astype(np.float32, copy=True)
        n = len(samples)
        with self._lock:
            self.wave_buf = np.roll(self.wave_buf, -n)
            self.wave_buf[-n:] = samples
            for ch in self.channels.values():
                ch.since_hop += n
                while ch.since_hop >= ch.hop:
                    seg = self.wave_buf[-ch.window_size:]
                    ch.push_column(seg)
                    ch.since_hop -= ch.hop

    # ---- GUI-thread getters (each takes the lock briefly) ----

    def get_wave(self):
        with self._lock:
            return self.wave_buf.copy()

    def get_spectrogram(self, window_size):
        with self._lock:
            ch = self.channels[window_size]
            return ch.spec.copy(), ch.freqs_disp

    def get_window_segment(self, window_size, t0):
        """t0 is a time in seconds, relative to 'now' == 0, oldest == -HISTORY_SECONDS.
        Returns (segment, spectrum_db, freqs_disp, clamped_t0)."""
        with self._lock:
            ch = self.channels[window_size]
            win_dur = window_size / self.sample_rate
            t0 = max(-HISTORY_SECONDS, min(t0, -win_dur))
            idx0 = int(round((t0 + HISTORY_SECONDS) * self.sample_rate))
            idx0 = max(0, min(idx0, HIST_SAMPLES - window_size))
            seg = self.wave_buf[idx0:idx0 + window_size].copy()
            spectrum_db = ch.analyze_segment(seg)
            freqs_disp = ch.freqs_disp
            clamped_t0 = idx0 / self.sample_rate - HISTORY_SECONDS
        return seg, spectrum_db, freqs_disp, clamped_t0


# --------------------------------------------------------------------------
# Small reusable widget builders
# --------------------------------------------------------------------------

CAPTION_STYLE = "color: #7a7a82; font-size: 12px; font-weight: 600; letter-spacing: 2.5px;"


def make_caption(text):
    label = QtWidgets.QLabel(text)
    label.setStyleSheet(CAPTION_STYLE)
    return label


def make_plot(x_label="", y_label=""):
    plot = pg.PlotWidget()
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    plot.setMenuEnabled(False)
    plot.showGrid(x=True, y=True, alpha=0.15)
    if x_label:
        plot.setLabel("bottom", x_label)
    if y_label:
        plot.setLabel("left", y_label)
    return plot


def make_spectrogram_item(colormap):
    img = pg.ImageItem()
    lut = colormap.getLookupTable(0.0, 1.0, 256)
    img.setLookupTable(lut)
    img.setLevels([DB_FLOOR, DB_CEIL])  # fixed -- never autoscaled
    return img


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECA -- STFT: Repeated FFTs of Short Time Windows")
        self.resize(1500, 1000)
        self.setStyleSheet("background-color: #0a0a0d;")

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#0a0a0d")
        pg.setConfigOption("foreground", "#d8d8dc")

        self.colormap = build_spectrogram_colormap()
        self.engine = AudioEngine()
        self.engine.error.connect(self.show_error)

        self.window_size = DEFAULT_WINDOW_SIZE
        self.window_t0 = -1.0  # window-explainer left edge, seconds relative to "now"

        self._wave_x = np.linspace(-HISTORY_SECONDS, 0, HIST_SAMPLES)

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)
        self.setCentralWidget(central)

        title = QtWidgets.QLabel("SHORT-TIME FOURIER TRANSFORM")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #e6e6ea; font-size: 26px; font-weight: 800; letter-spacing: 3px;")
        outer.addWidget(title)

        subtitle = QtWidgets.QLabel("STFT = repeated FFTs of short overlapping time windows")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet("color: #4fd1ff; font-size: 15px; font-weight: 600;")
        outer.addWidget(subtitle)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        outer.addWidget(self.status_label)

        outer.addWidget(self._build_window_size_row())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { background: #16161c; color: #b8b8c0; padding: 8px 18px;"
            " font-size: 13px; font-weight: 600; letter-spacing: 1px; }"
            "QTabBar::tab:selected { background: #24242e; color: #ffffff; }"
            "QTabWidget::pane { border: 1px solid #24242e; }"
        )
        self.tabs.addTab(self._build_live_tab(), "MODE 1 -- LIVE STFT")
        self.tabs.addTab(self._build_explainer_tab(), "MODE 2 -- WINDOW EXPLAINER")
        self.tabs.addTab(self._build_compare_tab(), "MODE 3 -- WINDOW SIZE EXPERIMENT")
        outer.addWidget(self.tabs, stretch=1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UI_REFRESH_MS)

        self._update_explainer_bounds(recenter=True)
        self.engine.start()

    # ---- window-size control (shared by Tab 1 and Tab 2) ----

    def _build_window_size_row(self):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        label = QtWidgets.QLabel("WINDOW SIZE")
        label.setStyleSheet(CAPTION_STYLE)
        layout.addWidget(label)

        self.size_group = QtWidgets.QButtonGroup(self)
        self.size_buttons = {}
        for size in WINDOW_SIZES:
            btn = QtWidgets.QPushButton(str(size))
            btn.setCheckable(True)
            btn.setChecked(size == DEFAULT_WINDOW_SIZE)
            btn.setStyleSheet(self._size_button_style())
            btn.clicked.connect(lambda _checked, s=size: self.set_window_size(s))
            self.size_group.addButton(btn)
            self.size_buttons[size] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        return row

    @staticmethod
    def _size_button_style():
        return (
            "QPushButton { background: #16161c; color: #b8b8c0; border: 1px solid #2c2c36;"
            " border-radius: 4px; padding: 6px 16px; font-size: 13px; font-weight: 700; }"
            "QPushButton:checked { background: #4fd1ff; color: #0a0a0d; border-color: #4fd1ff; }"
        )

    def set_window_size(self, size):
        self.window_size = size
        for s, btn in self.size_buttons.items():
            btn.setChecked(s == size)
        self._update_explainer_bounds(recenter=False)
        self.update_display()

    # ---- Tab 1: LIVE STFT ----

    def _build_live_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(6)

        layout.addWidget(make_caption("TIME DOMAIN -- live waveform"))
        self.live_wave_plot = make_plot("Time (s)", "Amplitude")
        self.live_wave_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.live_wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.live_wave_curve = self.live_wave_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.2))
        layout.addWidget(self.live_wave_plot, stretch=1)

        layout.addWidget(make_caption("STFT -- audio -> window -> FFT -> dB -> one column, repeated"))
        self.live_spec_plot = make_plot("Time (s)", "Frequency (Hz)")
        self.live_spec_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.live_spec_plot.setYRange(0, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.live_spec_img = make_spectrogram_item(self.colormap)
        self.live_spec_plot.addItem(self.live_spec_img)
        layout.addWidget(self.live_spec_plot, stretch=2)

        return tab

    # ---- Tab 2: WINDOW EXPLAINER ----

    def _build_explainer_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(6)

        hint = QtWidgets.QLabel("Drag the highlighted window below -- waveform slice, FFT, and spectrogram column update together.")
        hint.setStyleSheet("color: #9a9aa2; font-size: 12px;")
        layout.addWidget(hint)

        layout.addWidget(make_caption("TIME DOMAIN"))
        self.exp_wave_plot = make_plot("Time (s)", "Amplitude")
        self.exp_wave_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.exp_wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.exp_wave_curve = self.exp_wave_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.2))

        self.window_region_wave = pg.LinearRegionItem(
            values=[-1.0, -0.9], movable=False,
            brush=pg.mkBrush(255, 196, 60, 60), pen=pg.mkPen("#ffc43c", width=1.5),
        )
        self.exp_wave_plot.addItem(self.window_region_wave)

        self.window_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#ffc43c", width=2),
            hoverPen=pg.mkPen("#ffe08a", width=2),
        )
        self.window_line.sigPositionChanged.connect(self._on_window_line_moved)
        self.exp_wave_plot.addItem(self.window_line)
        layout.addWidget(self.exp_wave_plot, stretch=1)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)

        col1 = QtWidgets.QVBoxLayout()
        col1.addWidget(make_caption("WAVEFORM SEGMENT INSIDE WINDOW"))
        self.seg_plot = make_plot("Time (ms)", "Amplitude")
        self.seg_plot.setYRange(-1.0, 1.0, padding=0)
        self.seg_curve = self.seg_plot.plot(pen=pg.mkPen("#ffc43c", width=1.5))
        col1.addWidget(self.seg_plot)
        row.addLayout(col1, stretch=1)

        col2 = QtWidgets.QVBoxLayout()
        col2.addWidget(make_caption("FFT OF THIS WINDOW"))
        self.fft_plot = make_plot("Frequency (Hz)", "dB")
        self.fft_plot.setXRange(0, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.fft_plot.setYRange(DB_FLOOR, DB_CEIL, padding=0.02)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.5), fillLevel=DB_FLOOR,
                                             brush=pg.mkBrush(79, 209, 255, 40))
        col2.addWidget(self.fft_plot)
        row.addLayout(col2, stretch=1)

        layout.addLayout(row, stretch=1)

        layout.addWidget(make_caption("STFT -- highlighted column is this exact window"))
        self.exp_spec_plot = make_plot("Time (s)", "Frequency (Hz)")
        self.exp_spec_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.exp_spec_plot.setYRange(0, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.exp_spec_img = make_spectrogram_item(self.colormap)
        self.exp_spec_plot.addItem(self.exp_spec_img)
        self.window_region_spec = pg.LinearRegionItem(
            values=[-1.0, -0.9], movable=False,
            brush=pg.mkBrush(255, 196, 60, 50), pen=pg.mkPen("#ffc43c", width=1.5),
        )
        self.exp_spec_plot.addItem(self.window_region_spec)
        layout.addWidget(self.exp_spec_plot, stretch=2)

        return tab

    def _on_window_line_moved(self):
        win_dur = self.window_size / SAMPLE_RATE
        t0 = self.window_line.value() - win_dur / 2.0
        self.window_t0 = t0
        self.window_region_wave.setRegion([t0, t0 + win_dur])
        self.window_region_spec.setRegion([t0, t0 + win_dur])
        self.update_window_explainer()

    def _update_explainer_bounds(self, recenter):
        win_dur = self.window_size / SAMPLE_RATE
        lo = -HISTORY_SECONDS + win_dur / 2.0
        hi = -win_dur / 2.0
        self.window_line.setBounds([lo, hi])
        if recenter:
            center = max(lo, min(-1.0 + win_dur / 2.0, hi))
        else:
            center = max(lo, min(self.window_line.value(), hi))
        self.window_line.blockSignals(True)
        self.window_line.setValue(center)
        self.window_line.blockSignals(False)
        self.window_t0 = center - win_dur / 2.0
        self.window_region_wave.setRegion([self.window_t0, self.window_t0 + win_dur])
        self.window_region_spec.setRegion([self.window_t0, self.window_t0 + win_dur])
        self.update_window_explainer()

    def update_window_explainer(self):
        seg, spectrum_db, freqs_disp, t0 = self.engine.get_window_segment(self.window_size, self.window_t0)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        seg_ms = np.linspace(0.0, 1000.0 * self.window_size / SAMPLE_RATE, len(seg))
        self.seg_curve.setData(seg_ms, seg)
        self.seg_plot.setXRange(0, seg_ms[-1] if len(seg_ms) else 1.0, padding=0)

        spectrum_db = np.nan_to_num(spectrum_db, nan=DB_FLOOR, posinf=DB_CEIL, neginf=DB_FLOOR)
        self.fft_curve.setData(freqs_disp, spectrum_db)

    # ---- Tab 3: WINDOW SIZE EXPERIMENT ----

    def _build_compare_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(6)

        hint = QtWidgets.QLabel(
            "Same microphone audio, two window sizes, drawn at the identical physical size: "
            "short window = sharp in time / blurry in frequency; long window = the opposite."
        )
        hint.setStyleSheet("color: #9a9aa2; font-size: 12px;")
        layout.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        self.compare_combo_a = QtWidgets.QComboBox()
        self.compare_combo_b = QtWidgets.QComboBox()
        for combo in (self.compare_combo_a, self.compare_combo_b):
            for size in WINDOW_SIZES:
                combo.addItem(str(size), size)
            combo.setStyleSheet("color: #d8d8dc; background: #16161c; padding: 4px 10px;")
        self.compare_combo_a.setCurrentIndex(WINDOW_SIZES.index(512))
        self.compare_combo_b.setCurrentIndex(WINDOW_SIZES.index(4096))
        self.compare_combo_a.currentIndexChanged.connect(self.update_display)
        self.compare_combo_b.currentIndexChanged.connect(self.update_display)

        row.addWidget(QtWidgets.QLabel("A:"))
        row.addWidget(self.compare_combo_a)
        row.addSpacing(20)
        row.addWidget(QtWidgets.QLabel("B:"))
        row.addWidget(self.compare_combo_b)
        row.addStretch(1)
        layout.addLayout(row)

        self.compare_caption_a = make_caption("WINDOW SIZE 512")
        layout.addWidget(self.compare_caption_a)
        self.compare_plot_a = make_plot("Time (s)", "Frequency (Hz)")
        self.compare_plot_a.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.compare_plot_a.setYRange(0, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.compare_img_a = make_spectrogram_item(self.colormap)
        self.compare_plot_a.addItem(self.compare_img_a)
        layout.addWidget(self.compare_plot_a, stretch=1)

        self.compare_caption_b = make_caption("WINDOW SIZE 4096")
        layout.addWidget(self.compare_caption_b)
        self.compare_plot_b = make_plot("Time (s)", "Frequency (Hz)")
        self.compare_plot_b.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.compare_plot_b.setYRange(0, DISPLAY_MAX_FREQ_HZ, padding=0)
        self.compare_img_b = make_spectrogram_item(self.colormap)
        self.compare_plot_b.addItem(self.compare_img_b)
        layout.addWidget(self.compare_plot_b, stretch=1)

        return tab

    # ---- shared update loop ----

    def show_error(self, message):
        self.status_label.setText(message)
        self.status_label.show()
        self.tabs.hide()

    def _set_spec_image(self, img_item, spec_matrix):
        spec_matrix = np.nan_to_num(spec_matrix, nan=DB_FLOOR, posinf=DB_CEIL, neginf=DB_FLOOR)
        img_item.setImage(spec_matrix, autoLevels=False)
        img_item.setRect(QtCore.QRectF(-HISTORY_SECONDS, 0, HISTORY_SECONDS, DISPLAY_MAX_FREQ_HZ))
        img_item.setLevels([DB_FLOOR, DB_CEIL])

    def update_display(self):
        if self.engine.stream is None and not self.status_label.isVisible():
            pass  # stream may still be starting; nothing to do yet either way

        wave = np.nan_to_num(self.engine.get_wave(), nan=0.0, posinf=0.0, neginf=0.0)
        tab_idx = self.tabs.currentIndex()

        if tab_idx == 0:
            self.live_wave_curve.setData(self._wave_x, wave)
            spec, _ = self.engine.get_spectrogram(self.window_size)
            self._set_spec_image(self.live_spec_img, spec)
        elif tab_idx == 1:
            self.exp_wave_curve.setData(self._wave_x, wave)
            spec, _ = self.engine.get_spectrogram(self.window_size)
            self._set_spec_image(self.exp_spec_img, spec)
            self.update_window_explainer()
        else:
            size_a = self.compare_combo_a.currentData()
            size_b = self.compare_combo_b.currentData()
            self.compare_caption_a.setText(f"WINDOW SIZE {size_a}")
            self.compare_caption_b.setText(f"WINDOW SIZE {size_b}")
            spec_a, _ = self.engine.get_spectrogram(size_a)
            spec_b, _ = self.engine.get_spectrogram(size_b)
            self._set_spec_image(self.compare_img_a, spec_a)
            self._set_spec_image(self.compare_img_b, spec_b)

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(event)


# --------------------------------------------------------------------------
# Offline self-tests -- pure DSP, no GUI / mic / playback
# --------------------------------------------------------------------------

def _make_tone(freq_hz, n_samples, sample_rate=SAMPLE_RATE, amplitude=0.8, phase=0.0):
    t = np.arange(n_samples) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


def run_self_tests():
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}")
        if not cond:
            failures.append(name)

    # 1. Pure tone -> spectral peak near the right frequency, for every window size.
    for size in WINDOW_SIZES:
        ch = STFTChannel(size)
        seg = _make_tone(440.0, size)
        db = ch.analyze_segment(seg)
        check(f"window={size}: spectrum is finite", np.all(np.isfinite(db)))
        peak_freq = float(ch.freqs_disp[np.argmax(db)])
        check(f"window={size}: peak near 440 Hz (bin width {ch.bin_width_hz:.1f} Hz)",
              abs(peak_freq - 440.0) <= max(2 * ch.bin_width_hz, 5.0))
        check(f"window={size}: full-scale-ish tone peaks near 0 dB, not clipped away",
              db.max() > DB_FLOOR + 10.0)

    # 2. Silence -> flat floor, finite, not NaN/Inf.
    ch = STFTChannel(DEFAULT_WINDOW_SIZE)
    silence = np.zeros(DEFAULT_WINDOW_SIZE, dtype=np.float32)
    db = ch.analyze_segment(silence)
    check("silence: spectrum is finite", np.all(np.isfinite(db)))
    check("silence: spectrum sits at the dB floor (no phantom energy)", np.allclose(db, DB_FLOOR, atol=1e-3))
    check("silence: display is NOT a solid bright/clipped block",
          bool(db.max() < DB_CEIL - 1.0))

    # 3. Extreme / pathological input never produces NaN/Inf.
    ch = STFTChannel(DEFAULT_WINDOW_SIZE)
    weird = np.array([np.nan, np.inf, -np.inf] + [0.0] * (DEFAULT_WINDOW_SIZE - 3), dtype=np.float32)
    weird_clean = np.nan_to_num(weird, nan=0.0, posinf=0.0, neginf=0.0)
    db = ch.analyze_segment(weird_clean)
    check("pathological input (nan/inf sanitized upstream) stays finite", np.all(np.isfinite(db)))

    # 4. Frequency resolution: two close tones (30 Hz apart) -- a small window
    #    (wide bins) blurs them into one peak, a large window (narrow bins)
    #    resolves two distinct peaks. This is the frequency-resolution half
    #    of the time/frequency trade-off.
    f1, f2 = 1000.0, 1030.0
    small = STFTChannel(512)     # bin width ~86 Hz -> cannot separate 30 Hz apart
    large = STFTChannel(4096)    # bin width ~10.8 Hz -> can

    def two_tone(n):
        return _make_tone(f1, n, amplitude=0.5) + _make_tone(f2, n, amplitude=0.5, phase=0.3)

    def count_peaks_near(ch_, lo=900.0, hi=1150.0, prominence_db=6.0):
        seg = two_tone(ch_.window_size)
        db = ch_.analyze_segment(seg)
        mask = (ch_.freqs_disp >= lo) & (ch_.freqs_disp <= hi)
        f = ch_.freqs_disp[mask]
        d = db[mask]
        if len(d) < 3:
            return 0
        is_peak = (d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > d.max() - prominence_db)
        return int(np.sum(is_peak))

    small_peaks = count_peaks_near(small)
    large_peaks = count_peaks_near(large)
    check("frequency resolution: small window (512) blurs two close tones into ~1 peak",
          small_peaks <= 1)
    check("frequency resolution: large window (4096) resolves two close tones into 2 peaks",
          large_peaks >= 2)

    # 5. Time resolution: a short click localized in time. A small window
    #    should confine its energy to fewer spectrogram columns than a large
    #    window, which smears it across more columns.
    def click_signal(total_samples, click_at, click_len=40, sample_rate=SAMPLE_RATE):
        sig = np.zeros(total_samples, dtype=np.float32)
        end = min(total_samples, click_at + click_len)
        sig[click_at:end] = _make_tone(2000.0, end - click_at, sample_rate, amplitude=0.9)
        return sig

    def columns_lit(ch_, sig, threshold_db=DB_FLOOR + 20.0):
        ch_local = STFTChannel(ch_.window_size)
        lit = 0
        for start in range(0, len(sig) - ch_local.window_size + 1, ch_local.hop):
            seg = sig[start:start + ch_local.window_size]
            db = ch_local._segment_to_db(seg)
            if db.max() > threshold_db:
                lit += 1
        return lit

    total_len = SAMPLE_RATE  # 1 second buffer, click roughly in the middle
    sig = click_signal(total_len, total_len // 2)
    small_lit = columns_lit(STFTChannel(512), sig)
    large_lit = columns_lit(STFTChannel(4096), sig)
    check("time resolution: short window (512) localizes the click to fewer columns",
          small_lit > 0)
    check("time resolution: long window (4096) smears the click across more (or equal) columns",
          large_lit >= small_lit)

    # 6. Continuous push_column pipeline (mirrors the live audio path) stays
    #    finite across many hops, including a silence -> tone -> silence sequence.
    ch = STFTChannel(1024)
    seq = np.concatenate([
        np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
        _make_tone(220.0, SAMPLE_RATE // 2, amplitude=0.6),
        np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
    ])
    for start in range(0, len(seq) - ch.window_size + 1, ch.hop):
        ch.push_column(seq[start:start + ch.window_size])
    check("continuous STFT pipeline stays finite over silence->tone->silence", np.all(np.isfinite(ch.spec)))
    check("continuous STFT pipeline shows more energy during the tone than pure silence",
          ch.spec.max() > DB_FLOOR + 15.0)

    # 7. Changing window size changes both hop and bin width as expected
    #    (the mechanism behind the resolution trade-off).
    ch_small, ch_large = STFTChannel(512), STFTChannel(4096)
    check("larger window has narrower frequency bins (better frequency resolution)",
          ch_large.bin_width_hz < ch_small.bin_width_hz)
    check("larger window has a larger hop in samples (coarser time resolution)",
          ch_large.hop > ch_small.hop)

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
