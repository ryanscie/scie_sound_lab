#!/usr/bin/env python3
"""ECA fair demo -- Tier 1: live microphone waveform + spectrogram.

Captures microphone audio continuously and renders:
  - top:    scrolling waveform
  - bottom: scrolling spectrogram (time -> horizontal, frequency -> vertical)

Audio capture and FFT run on the PortAudio callback thread; the Qt GUI
thread only drains a queue and updates plots, so the UI never blocks on
audio processing.
"""

import queue
import sys

import numpy as np
import sounddevice as sd
from scipy.signal.windows import hann
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
BLOCK_SIZE = 512            # samples per audio callback / spectrogram hop
FFT_SIZE = 2048             # spectrogram analysis window
HISTORY_SECONDS = 6.0       # shared history window for both waveform and spectrogram
MAX_FREQ_HZ = 5000          # spectrogram vertical range (covers voice, clap, cello)
UI_REFRESH_MS = 30          # ~33 fps
SPECTROGRAM_DYNAMIC_RANGE_DB = 70.0  # dB span mapped across the colormap


class AudioEngine(QtCore.QObject):
    """Owns the microphone stream. FFT runs on the audio callback thread."""

    error = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, fft_size=FFT_SIZE):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.fft_size = fft_size
        self._window = hann(fft_size, sym=False).astype(np.float32)
        self._fft_buffer = np.zeros(fft_size, dtype=np.float32)
        self.out_queue: "queue.Queue[tuple[np.ndarray, np.ndarray]]" = queue.Queue()
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

        self._fft_buffer = np.roll(self._fft_buffer, -len(samples))
        self._fft_buffer[-len(samples):] = samples

        spectrum = np.fft.rfft(self._fft_buffer * self._window)
        mag_db = 20.0 * np.log10(np.abs(spectrum) + 1e-6)

        try:
            self.out_queue.put_nowait((samples, mag_db))
        except queue.Full:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECA -- Live Audio")
        self.resize(1500, 950)
        self.setStyleSheet("background-color: #0a0a0d;")

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#0a0a0d")
        pg.setConfigOption("foreground", "#d8d8dc")

        # ---- buffers ----
        self.wave_len = int(HISTORY_SECONDS * SAMPLE_RATE)
        self.wave_buf = np.zeros(self.wave_len, dtype=np.float32)
        self.wave_x = np.linspace(-HISTORY_SECONDS, 0, self.wave_len)

        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        self.freq_cutoff_idx = int(np.searchsorted(freqs, MAX_FREQ_HZ))
        self.n_freq_bins = self.freq_cutoff_idx

        self.n_cols = max(1, int(HISTORY_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
        self.spec_buf = np.full((self.n_freq_bins, self.n_cols), -100.0, dtype=np.float32)

        # ---- engine ----
        self.engine = AudioEngine()
        self.engine.error.connect(self.show_error)

        # ---- UI ----
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.setCentralWidget(central)

        title = QtWidgets.QLabel("LIVE MICROPHONE INPUT")
        title.setStyleSheet(
            "color: #e6e6ea; font-size: 20px; font-weight: 600; letter-spacing: 2px;"
        )
        layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # waveform panel
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.hideButtons()
        self.wave_plot.setMenuEnabled(False)
        self.wave_plot.showGrid(x=True, y=True, alpha=0.15)
        self.wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.wave_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Time (s)")
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen("#4fd1ff", width=1.5))
        layout.addWidget(self.wave_plot, stretch=1)

        # spectrogram panel
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMouseEnabled(x=False, y=False)
        self.spec_plot.hideButtons()
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.setLabel("left", "Frequency (Hz)")
        self.spec_plot.setLabel("bottom", "Time (s)")
        self.spec_plot.setYRange(0, MAX_FREQ_HZ, padding=0)
        self.spec_plot.setXRange(-HISTORY_SECONDS, 0, padding=0)

        self.spec_image = pg.ImageItem()
        # setImage() must be called before setRect(): ImageItem.setRect() scales
        # its transform using the image's current pixel dimensions, which are
        # unset (silently treated as 1x1) until an image has been assigned at
        # least once. Calling setRect() first (as this used to) produced a
        # transform stretched ~n_cols x n_freq_bins beyond the viewport, so only
        # a sliver of real data ever fell inside the visible time/frequency
        # window -- which is what made the plot look like a solid color.
        self.spec_image.setImage(self.spec_buf.T, autoLevels=False)
        self.spec_image.setRect(QtCore.QRectF(-HISTORY_SECONDS, 0, HISTORY_SECONDS, MAX_FREQ_HZ))
        colormap = pg.colormap.get("inferno")
        self.spec_image.setColorMap(colormap)
        self._level_lo = -100.0
        self._level_hi = self._level_lo + SPECTROGRAM_DYNAMIC_RANGE_DB
        self.spec_image.setLevels((self._level_lo, self._level_hi))
        self.spec_plot.addItem(self.spec_image)

        self.colorbar = pg.ColorBarItem(
            values=(self._level_lo, self._level_hi),
            colorMap=colormap,
            label="Intensity (dB)",
            interactive=False,
        )
        self.colorbar.setImageItem(self.spec_image, insert_in=self.spec_plot.getPlotItem())

        layout.addWidget(self.spec_plot, stretch=2)

        # ---- timer: drains audio queue and repaints on the GUI thread ----
        self._level_tick = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(UI_REFRESH_MS)

        self.engine.start()

    def show_error(self, message: str):
        self.status_label.setText(message)
        self.status_label.show()
        self.wave_plot.hide()
        self.spec_plot.hide()

    def update_plots(self):
        drained = 0
        new_cols = []
        try:
            while True:
                samples, mag_db = self.engine.out_queue.get_nowait()
                n = len(samples)
                self.wave_buf = np.roll(self.wave_buf, -n)
                self.wave_buf[-n:] = samples
                new_cols.append(mag_db[: self.n_freq_bins])
                drained += 1
                if drained >= 64:  # avoid unbounded work if the GUI ever falls behind
                    break
        except queue.Empty:
            pass

        if drained == 0:
            return

        self.wave_curve.setData(self.wave_x, self.wave_buf)

        k = len(new_cols)
        if k >= self.n_cols:
            self.spec_buf = np.stack(new_cols[-self.n_cols :], axis=1)
        else:
            self.spec_buf = np.roll(self.spec_buf, -k, axis=1)
            self.spec_buf[:, -k:] = np.stack(new_cols, axis=1)

        self._level_tick += 1
        if self._level_tick % 15 == 0:
            # Anchor the color range to the current noise floor and stretch a
            # fixed dB span above it, so quiet bins stay dark and anything
            # louder than the floor spreads visibly across the colormap.
            finite = self.spec_buf[np.isfinite(self.spec_buf)]
            if finite.size:
                self._level_lo = float(np.percentile(finite, 10))
                self._level_hi = self._level_lo + SPECTROGRAM_DYNAMIC_RANGE_DB
                self.colorbar.setLevels(low=self._level_lo, high=self._level_hi)

        self.spec_image.setImage(self.spec_buf.T, autoLevels=False)

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
