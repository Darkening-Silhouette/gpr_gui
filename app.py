#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressDialog, QPushButton, QDoubleSpinBox, QSpinBox, QScrollBar,
    QTabWidget, QVBoxLayout, QWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, Normalize
from matplotlib import colormaps
import pyvista as pv
from pyvistaqt import QtInteractor

try:
    from scipy.signal import butter, filtfilt, sosfiltfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


@dataclass
class GPRLine:
    number: int
    name: str
    direction: str
    folder: Path
    rad: Path
    rd7: Path
    cor: Optional[Path]
    mrk: Optional[Path]
    mrkj: Optional[Path]
    proj: Optional[Path]
    samples: int = 0
    traces: int = 0
    time_window_ns: float = 0.0
    dt_ns: float = 0.0
    fs_mhz: float = 0.0
    antenna: str = ""
    raw: Optional[np.ndarray] = None
    processed: Optional[np.ndarray] = None
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    elev: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None


def natural_line_key(path: Path) -> int:
    m = re.search(r"line(\d+)_", path.name)
    return int(m.group(1)) if m else 999999


def parse_rad(rad_path: Path) -> dict:
    text = rad_path.read_text(errors="ignore")
    out = {"samples": None, "traces": None, "time_window_ns": None, "antenna": ""}

    for line in text.splitlines():
        low = line.lower()

        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)

        if "samples" in low and nums and out["samples"] is None:
            out["samples"] = int(float(nums[-1]))

        if ("traces" in low or "last trace" in low) and nums and out["traces"] is None:
            out["traces"] = int(float(nums[-1]))

        if ("timewindow" in low or "time window" in low or "range" in low) and nums and out["time_window_ns"] is None:
            val = float(nums[-1])
            if 1 <= val <= 5000:
                out["time_window_ns"] = val

        if "antenna" in low and ":" in line:
            out["antenna"] = line.split(":", 1)[1].strip()

    if out["samples"] is None:
        m = re.search(r"SAMPLES\s*[:=]\s*(\d+)", text, re.I)
        if m:
            out["samples"] = int(m.group(1))

    if out["time_window_ns"] is None:
        m = re.search(r"TIME(?:\s*WINDOW|WINDOW)?\s*[:=]\s*([0-9.]+)", text, re.I)
        if m:
            out["time_window_ns"] = float(m.group(1))

    return out


def read_rd7(rd7_path: Path, samples_hint: int, traces_hint: Optional[int] = None) -> np.ndarray:
    raw_bytes = rd7_path.read_bytes()

    candidates = []
    for dtype in (np.int32, np.int16, np.float32):
        arr = np.frombuffer(raw_bytes, dtype=dtype)
        if samples_hint > 0 and arr.size % samples_hint == 0:
            traces = arr.size // samples_hint
            score = 0
            if traces_hint and abs(traces - traces_hint) < 5:
                score -= 10
            if dtype == np.int32:
                score -= 5
            candidates.append((score, dtype, arr, traces))

    if not candidates:
        raise ValueError(f"Cannot reshape {rd7_path.name}; samples_hint={samples_hint}, bytes={len(raw_bytes)}")

    _, dtype, arr, traces = sorted(candidates, key=lambda x: x[0])[0]
    data = arr.reshape(traces, samples_hint).astype(np.float64)
    return data


def parse_cor(cor_path: Optional[Path]):
    if cor_path is None or not cor_path.exists():
        return None, None, None

    lat, lon, elev = [], [], []

    for line in cor_path.read_text(errors="ignore").splitlines():
        vals = []
        for token in re.split(r"[,;\s]+", line.strip()):
            try:
                vals.append(float(token))
            except Exception:
                pass

        if len(vals) < 3:
            continue

        found = False
        for i in range(len(vals) - 1):
            a, b = vals[i], vals[i + 1]
            if 40.0 <= a <= 50.0 and 5.0 <= b <= 12.0:
                z = np.nan
                for j in range(i + 2, len(vals)):
                    if 250.0 <= vals[j] <= 1000.0:
                        z = vals[j]
                        break
                lat.append(a)
                lon.append(b)
                elev.append(z)
                found = True
                break

        if not found and len(vals) >= 4:
            a, b = vals[-3], vals[-2]
            if 40.0 <= a <= 50.0 and 5.0 <= b <= 12.0:
                lat.append(a)
                lon.append(b)
                elev.append(vals[-1])

    if len(lat) < 2:
        return None, None, None

    return np.asarray(lat), np.asarray(lon), np.asarray(elev)


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    r = 6371000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = r * (lon_rad - lon0_rad) * math.cos(lat0_rad)
    y = r * (lat_rad - lat0_rad)
    return x, y


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.r_[0.0, np.cumsum(ds)]


def robust_symmetric_limits(data: np.ndarray, pct: float):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    amp = np.nanpercentile(np.abs(finite), pct)
    amp = max(float(amp), 1e-12)
    return -amp, amp


def display_matrix(data: np.ndarray) -> np.ndarray:
    return data.T


def dewow(data: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        return data.copy()

    ntr, ns = data.shape

    if ns < 3:
        return data.copy()

    window = int(window)

    if window < 3:
        return data.copy()

    # Important: np.convolve(mode="same") returns max(len(trace), len(kernel)).
    # Therefore the kernel/window must not be longer than the trace/sample axis.
    window = min(window, ns)

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return data.copy()

    kernel = np.ones(window, dtype=np.float64) / float(window)
    trend = np.apply_along_axis(lambda tr: np.convolve(tr, kernel, mode="same"), 1, data)

    if trend.shape != data.shape:
        trend = trend[:, :ns]

    return data - trend



def local_background_remove(data: np.ndarray, window_traces: int) -> np.ndarray:
    """
    Local median background removal along profile direction.
    Scientifically safer for hyperbolas than full-line subtraction because
    it suppresses horizontal ringing while preserving local diffraction limbs.
    """
    ntr, ns = data.shape
    w = max(3, int(window_traces))
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty_like(data, dtype=np.float64)
    for i in range(ntr):
        a = max(0, i - half)
        b = min(ntr, i + half + 1)
        out[i, :] = data[i, :] - np.nanmedian(data[a:b, :], axis=0)
    return out


def agc_gain(data: np.ndarray, dt_ns: float, window_ns: float) -> np.ndarray:
    """
    Conservative AGC for hyperbola visibility.
    Uses an RMS floor so late-time random noise is not boosted excessively.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        return data

    ntr, ns = data.shape

    if dt_ns <= 0 or window_ns <= 0 or ns < 3:
        return data

    w = max(3, int(round(window_ns / dt_ns)))

    # Same fix as dewow: never allow the kernel to exceed the sample axis.
    w = min(w, ns)

    if w % 2 == 0:
        w -= 1

    if w < 3:
        return data

    kernel = np.ones(w, dtype=np.float64) / float(w)
    rms = np.sqrt(np.apply_along_axis(lambda tr: np.convolve(tr * tr, kernel, mode="same"), 1, data))

    if rms.shape != data.shape:
        rms = rms[:, :ns]

    finite = rms[np.isfinite(rms) & (rms > 0)]
    if finite.size == 0:
        return data

    scale = np.nanmedian(finite)
    floor = np.nanpercentile(finite, 35)

    return data * scale / np.maximum(rms, floor)


def process_gpr(
    data: np.ndarray,
    dt_ns: float,
    dewow_ns: float,
    remove_background: bool,
    use_bandpass: bool,
    low_mhz: float,
    high_mhz: float,
    sec_power: float,
    background_window_traces: int = 151,
    use_agc: bool = True,
    agc_window_ns: float = 40.0,
) -> np.ndarray:
    out = data.astype(np.float64).copy()

    # Trace-wise DC removal
    out -= np.nanmean(out, axis=1, keepdims=True)

    # Dewow: remove very low-frequency instrument drift
    if dt_ns > 0 and dewow_ns > 0:
        win = max(3, int(round(dewow_ns / dt_ns)))
        out = dewow(out, win)

    # Local median background removal: better for hyperbolas than full-line median
    if remove_background:
        out = local_background_remove(out, background_window_traces)

    # GX160 practical hyperbola band: preserve 160 MHz centre energy, reject slow drift/high noise.
    # Robust version: use SOS filtering and safely skip unstable/invalid filters.
    if use_bandpass and SCIPY_OK and dt_ns > 0 and out.ndim == 2 and out.shape[1] >= 12:
        fs_hz = 1.0 / (dt_ns * 1e-9)
        nyq = 0.5 * fs_hz

        low = low_mhz * 1e6 / nyq
        high = high_mhz * 1e6 / nyq

        low = max(low, 1e-5)
        high = min(high, 0.999)

        if 0 < low < high < 1:
            try:
                sos = butter(2, [low, high], btype="band", output="sos")
                padlen = min(30, max(0, out.shape[1] - 1))
                if padlen >= 3:
                    out = sosfiltfilt(sos, out, axis=1, padlen=padlen)
            except Exception:
                # If one short/odd line makes the filter unstable, keep the unfiltered result
                # rather than crashing the full 3D analysis.
                pass

    # SEC gain: compensate geometrical/attenuation decay with time
    if sec_power > 0 and dt_ns > 0:
        t = np.arange(out.shape[1], dtype=float) * dt_ns
        gain = (1.0 + t / max(t[-1], 1.0)) ** sec_power
        out *= gain[None, :]

    # AGC: maximise weak hyperbola limb visibility
    if use_agc:
        out = agc_gain(out, dt_ns, agc_window_ns)

    return out


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(self.fig)
        self._home_limits = {}
        self._scroll_callbacks = []
        self.mpl_connect("scroll_event", self._on_scroll)

    def remember_home(self):
        self._home_limits = {}
        for ax in self.fig.axes:
            self._home_limits[ax] = (ax.get_xlim(), ax.get_ylim())

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        scale = 1.0 / 1.25 if event.button == "up" else 1.25

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x = event.xdata
        y = event.ydata

        ax.set_xlim(x + (x0 - x) * scale, x + (x1 - x) * scale)
        ax.set_ylim(y + (y0 - y) * scale, y + (y1 - y) * scale)
        self.draw_idle()

        for cb in list(self._scroll_callbacks):
            try:
                cb()
            except Exception:
                pass


class LineTab(QWidget):
    def __init__(self, line: GPRLine, main_window):
        super().__init__()
        self.line = line
        self.main = main_window

        self.raw_canvas = MplCanvas(width=10, height=4)
        self.proc_canvas = MplCanvas(width=10, height=4)

        self.raw_hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.raw_vbar = QScrollBar(Qt.Orientation.Vertical)
        self.proc_hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.proc_vbar = QScrollBar(Qt.Orientation.Vertical)

        for bar in [self.raw_hbar, self.raw_vbar, self.proc_hbar, self.proc_vbar]:
            bar.setRange(0, 10000)

        self._syncing = False
        self.raw_full_xlim = None
        self.raw_full_ylim = None
        self.proc_full_xlim = None
        self.proc_full_ylim = None
        self.raw_ax = None
        self.proc_ax = None

        raw_box = QWidget()
        raw_grid = QGridLayout(raw_box)
        raw_grid.setContentsMargins(0, 0, 0, 0)
        raw_grid.addWidget(self.raw_canvas, 0, 0)
        raw_grid.addWidget(self.raw_vbar, 0, 1)
        raw_grid.addWidget(self.raw_hbar, 1, 0)

        proc_box = QWidget()
        proc_grid = QGridLayout(proc_box)
        proc_grid.setContentsMargins(0, 0, 0, 0)
        proc_grid.addWidget(self.proc_canvas, 0, 0)
        proc_grid.addWidget(self.proc_vbar, 0, 1)
        proc_grid.addWidget(self.proc_hbar, 1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(raw_box)
        layout.addWidget(proc_box)

        self.raw_hbar.valueChanged.connect(lambda: self._scrollbar_changed("raw"))
        self.raw_vbar.valueChanged.connect(lambda: self._scrollbar_changed("raw"))
        self.proc_hbar.valueChanged.connect(lambda: self._scrollbar_changed("proc"))
        self.proc_vbar.valueChanged.connect(lambda: self._scrollbar_changed("proc"))

        self.raw_canvas._scroll_callbacks.append(lambda: self._sync_scrollbars_from_axis("raw"))
        self.proc_canvas._scroll_callbacks.append(lambda: self._sync_scrollbars_from_axis("proc"))

    def oriented(self, data: np.ndarray, dist: np.ndarray):
        """
        Spatially truthful display convention:
        A is always on the left, B is always on the right.

        AB files are already A -> B.
        BA files were acquired B -> A, so they are reversed for display only.
        Raw data on disk are not modified.
        """
        if self.line.direction in ("ba", "dc"):
            return data[::-1, :], dist.max() - dist[::-1]
        return data, dist

    def imshow_kwargs(self, data: np.ndarray):
        mode = self.main.scale_mode.currentText()
        cmap = self.main.cmap.currentText()
        clip = float(self.main.clip.value())
        data = np.asarray(data, dtype=float)

        if mode == "linear":
            vmin, vmax = robust_symmetric_limits(data, clip)
            return data, {"cmap": cmap, "vmin": vmin, "vmax": vmax}

        if mode == "symlog":
            vmax = float(np.nanpercentile(np.abs(data), clip))
            vmax = max(vmax, 1e-12)
            linthresh = max(vmax * 0.02, 1e-12)
            return data, {"cmap": cmap, "norm": SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)}

        absdata = np.abs(data)
        finite = absdata[np.isfinite(absdata) & (absdata > 0)]

        if finite.size:
            vmin = float(np.nanpercentile(finite, max(0.1, 100.0 - clip)))
            vmax = float(np.nanpercentile(finite, clip))
        else:
            vmin, vmax = 1e-12, 1e-9

        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 10.0)

        return absdata, {"cmap": cmap, "norm": LogNorm(vmin=vmin, vmax=vmax)}

    def _get_parts(self, which):
        if which == "raw":
            return self.raw_ax, self.raw_hbar, self.raw_vbar, self.raw_full_xlim, self.raw_full_ylim, self.raw_canvas
        return self.proc_ax, self.proc_hbar, self.proc_vbar, self.proc_full_xlim, self.proc_full_ylim, self.proc_canvas

    def _sync_scrollbars_from_axis(self, which):
        ax, hbar, vbar, full_xlim, full_ylim, canvas = self._get_parts(which)

        if ax is None or full_xlim is None or full_ylim is None:
            return

        self._syncing = True
        try:
            fx0, fx1 = full_xlim
            fy0, fy1 = full_ylim

            full_x_min, full_x_max = min(fx0, fx1), max(fx0, fx1)
            full_y_min, full_y_max = min(fy0, fy1), max(fy0, fy1)

            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            view_x_min, view_x_max = min(x0, x1), max(x0, x1)
            view_y_min, view_y_max = min(y0, y1), max(y0, y1)

            full_x_span = full_x_max - full_x_min
            full_y_span = full_y_max - full_y_min
            view_x_span = view_x_max - view_x_min
            view_y_span = view_y_max - view_y_min

            if full_x_span <= view_x_span:
                hbar.setValue(0)
            else:
                val = int(10000 * (view_x_min - full_x_min) / max(full_x_span - view_x_span, 1e-12))
                hbar.setValue(max(0, min(10000, val)))

            if full_y_span <= view_y_span:
                vbar.setValue(0)
            else:
                val = int(10000 * (view_y_min - full_y_min) / max(full_y_span - view_y_span, 1e-12))
                vbar.setValue(max(0, min(10000, val)))
        finally:
            self._syncing = False

    def _scrollbar_changed(self, which):
        if self._syncing:
            return

        ax, hbar, vbar, full_xlim, full_ylim, canvas = self._get_parts(which)

        if ax is None or full_xlim is None or full_ylim is None:
            return

        fx0, fx1 = full_xlim
        fy0, fy1 = full_ylim

        full_x_min, full_x_max = min(fx0, fx1), max(fx0, fx1)
        full_y_min, full_y_max = min(fy0, fy1), max(fy0, fy1)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        view_x_span = min(abs(x1 - x0), full_x_max - full_x_min)
        view_y_span = min(abs(y1 - y0), full_y_max - full_y_min)

        new_x_min = full_x_min + (hbar.value() / 10000.0) * max((full_x_max - full_x_min) - view_x_span, 0.0)
        new_y_min = full_y_min + (vbar.value() / 10000.0) * max((full_y_max - full_y_min) - view_y_span, 0.0)

        ax.set_xlim(new_x_min, new_x_min + view_x_span)

        # Keep GPR time axis downward: 0 ns at top, larger time downward.
        ax.set_ylim(new_y_min + view_y_span, new_y_min)

        canvas.draw_idle()

    def _distance_and_extent(self, data):
        dist = self.line.dist
        if dist is None or len(dist) != data.shape[0]:
            dist = np.linspace(0.0, data.shape[0] - 1, data.shape[0])

        data, dist = self.oriented(data, dist)

        tw_full = self.line.time_window_ns if self.line.time_window_ns > 0 else data.shape[1]
        tw_display = min(float(self.main.display_tmax.value()), float(tw_full))

        if self.line.dt_ns > 0:
            nsamp = max(2, min(data.shape[1], int(round(tw_display / self.line.dt_ns))))
            data = data[:, :nsamp]
            tw_display = nsamp * self.line.dt_ns

        extent = [float(dist.min()), float(dist.max()), float(tw_display), 0.0]
        full_xlim = (float(dist.min()), float(dist.max()))
        full_ylim = (0.0, float(tw_display))
        return data, dist, extent, full_xlim, full_ylim

    def _apply_vertical_exaggeration(self, ax, full_xlim, full_ylim):
        """
        Display-only vertical exaggeration.

        Keeps x-axis as true distance [m] and y-axis as true two-way time [ns].
        Vertical exaggeration is implemented as initial vertical zoom, not by
        distorting metre/ns aspect ratio.
        """
        x0, x1 = full_xlim
        t0, t1 = full_ylim

        ve = max(1.0, float(self.main.vertical_exag.value()))
        full_t = max(t1 - t0, 1e-9)
        visible_t = full_t / ve

        ax.set_xlim(x0, x1)
        ax.set_ylim(t0 + visible_t, t0)

    def plot_raw(self):
        self.raw_canvas.fig.clear()

        if self.line.raw is None:
            ax = self.raw_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Click 'Load raw' or 'Process current line'", ha="center", va="center")
            ax.set_axis_off()
            self.raw_ax = None
            self.raw_canvas.draw_idle()
            return

        raw, dist, extent, full_xlim, full_ylim = self._distance_and_extent(self.line.raw)

        ax = self.raw_canvas.fig.add_subplot(111)
        img, kw = self.imshow_kwargs(raw)
        ax.imshow(display_matrix(img), aspect="auto", extent=extent, interpolation="bilinear", **kw)
        ax.set_title(f"{self.line.name}: raw radargram")
        ax.set_ylabel("Two-way time [ns]")
        ax.set_xlabel("Distance along displayed line [m]")
        self.raw_ax = ax
        self.raw_full_xlim = full_xlim
        self.raw_full_ylim = full_ylim
        self._apply_vertical_exaggeration(ax, full_xlim, full_ylim)

        self.raw_canvas.remember_home()
        self.raw_canvas.draw_idle()
        self._sync_scrollbars_from_axis("raw")

    def plot_processed(self):
        self.proc_canvas.fig.clear()

        if self.line.processed is None:
            ax = self.proc_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Processed radargram will appear here", ha="center", va="center")
            ax.set_axis_off()
            self.proc_ax = None
            self.proc_canvas.draw_idle()
            return

        proc, dist, extent, full_xlim, full_ylim = self._distance_and_extent(self.line.processed)

        ax = self.proc_canvas.fig.add_subplot(111)
        img, kw = self.imshow_kwargs(proc)
        ax.imshow(display_matrix(img), aspect="auto", extent=extent, interpolation="bilinear", **kw)
        ax.set_title("Processed: DC removal + dewow + background removal + optional bandpass + SEC gain")
        ax.set_ylabel("Two-way time [ns]")
        ax.set_xlabel("Distance along displayed line [m]")
        self.proc_ax = ax
        self.proc_full_xlim = full_xlim
        self.proc_full_ylim = full_ylim
        self._apply_vertical_exaggeration(ax, full_xlim, full_ylim)

        self.proc_canvas.remember_home()
        self.proc_canvas.draw_idle()
        self._sync_scrollbars_from_axis("proc")

    def plot(self):
        self.plot_raw()
        self.plot_processed()



class GPR3DAnalysisTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.volume_grid = None
        self.volume_name = "GPR amplitude volume"
        self._progress_dialog = None
        self._progress_logs = []
        self._progress_total = 1
        self._progress_count = 0

        self.mode = QComboBox()
        self.mode.addItems(["processed", "raw"])

        self.line_group = QComboBox()
        self.line_group.addItems(["both", "inline only", "crossline only"])

        self.cross_step = QSpinBox()
        self.cross_step.setRange(1, 50)
        self.cross_step.setValue(1)

        self.inline_step = QSpinBox()
        self.inline_step.setRange(1, 17)
        self.inline_step.setValue(1)

        self.max_lines = QSpinBox()
        self.max_lines.setRange(1, 250)
        self.max_lines.setValue(45)

        self.trace_step = QSpinBox()
        self.trace_step.setRange(1, 100)
        self.trace_step.setValue(8)

        self.sample_step = QSpinBox()
        self.sample_step.setRange(1, 100)
        self.sample_step.setValue(6)

        self.slice_time = QDoubleSpinBox()
        self.slice_time.setRange(0.0, 1000.0)
        self.slice_time.setValue(80.0)
        self.slice_time.setSuffix(" ns")

        self.depth = QDoubleSpinBox()
        self.depth.setRange(0.0, 20.0)
        self.depth.setValue(2.0)
        self.depth.setSuffix(" m")

        self.velocity = QDoubleSpinBox()
        self.velocity.setRange(0.01, 0.30)
        self.velocity.setValue(0.10)
        self.velocity.setSingleStep(0.005)
        self.velocity.setSuffix(" m/ns")

        self.proj_tmin = QDoubleSpinBox()
        self.proj_tmin.setRange(0.0, 1000.0)
        self.proj_tmin.setValue(35.0)
        self.proj_tmin.setSuffix(" ns")

        self.proj_tmax = QDoubleSpinBox()
        self.proj_tmax.setRange(0.0, 1000.0)
        self.proj_tmax.setValue(180.0)
        self.proj_tmax.setSuffix(" ns")

        self.nx = QSpinBox()
        self.nx.setRange(20, 350)
        self.nx.setValue(110)

        self.ny = QSpinBox()
        self.ny.setRange(20, 350)
        self.ny.setValue(80)

        self.nz = QSpinBox()
        self.nz.setRange(20, 250)
        self.nz.setValue(90)

        self.iso_percentile = QDoubleSpinBox()
        self.iso_percentile.setRange(50.0, 99.9)
        self.iso_percentile.setValue(96.0)
        self.iso_percentile.setSuffix(" %")

        self.fence_style = QComboBox()
        self.fence_style.addItems([
            "Dark high contrast",
            "Dark seismic",
            "Dark turbo",
            "Light seismic",
            "Light grayscale",
            "Transparent pale"
        ])
        self.fence_style.setCurrentText("Dark high contrast")

        self.update_btn = QPushButton("Update selected view")
        self.build_volume_btn = QPushButton("Build / refresh PyVista volume")
        self.export_vtk_btn = QPushButton("Export volume .vti")

        self.tabs = QTabWidget()

        self.fence_plotter = QtInteractor(self)
        self.volume_plotter = QtInteractor(self)
        self.slice3d_plotter = QtInteractor(self)
        self.iso_plotter = QtInteractor(self)

        self.time_canvas = MplCanvas(width=10, height=7)
        self.depth_canvas = MplCanvas(width=10, height=7)
        self.proj_canvas = MplCanvas(width=10, height=7)

        self.tabs.addTab(self.fence_plotter.interactor, "PyVista 3D Fence")
        self.tabs.addTab(self.volume_plotter.interactor, "PyVista Volume")
        self.tabs.addTab(self.slice3d_plotter.interactor, "PyVista 3D Slices")
        self.tabs.addTab(self.iso_plotter.interactor, "PyVista Isosurface")
        self.tabs.addTab(self.time_canvas, "Time Slice Map")
        self.tabs.addTab(self.depth_canvas, "Depth Slice Map")
        self.tabs.addTab(self.proj_canvas, "Amplitude Projection")

        controls = QWidget()
        grid = QGridLayout(controls)

        grid.addWidget(QLabel("Data"), 0, 0)
        grid.addWidget(self.mode, 0, 1)
        grid.addWidget(QLabel("Lines"), 0, 2)
        grid.addWidget(self.line_group, 0, 3)
        grid.addWidget(QLabel("Cross step"), 0, 4)
        grid.addWidget(self.cross_step, 0, 5)
        grid.addWidget(QLabel("Inline step"), 0, 6)
        grid.addWidget(self.inline_step, 0, 7)
        grid.addWidget(QLabel("Max lines"), 0, 8)
        grid.addWidget(self.max_lines, 0, 9)

        grid.addWidget(QLabel("Trace step"), 1, 0)
        grid.addWidget(self.trace_step, 1, 1)
        grid.addWidget(QLabel("Sample step"), 1, 2)
        grid.addWidget(self.sample_step, 1, 3)
        grid.addWidget(QLabel("Time slice"), 1, 4)
        grid.addWidget(self.slice_time, 1, 5)
        grid.addWidget(QLabel("Depth"), 1, 6)
        grid.addWidget(self.depth, 1, 7)
        grid.addWidget(QLabel("Velocity"), 1, 8)
        grid.addWidget(self.velocity, 1, 9)

        grid.addWidget(QLabel("Projection tmin"), 2, 0)
        grid.addWidget(self.proj_tmin, 2, 1)
        grid.addWidget(QLabel("Projection tmax"), 2, 2)
        grid.addWidget(self.proj_tmax, 2, 3)
        grid.addWidget(QLabel("Volume nx/ny/nz"), 2, 4)
        grid.addWidget(self.nx, 2, 5)
        grid.addWidget(self.ny, 2, 6)
        grid.addWidget(self.nz, 2, 7)
        grid.addWidget(QLabel("Iso"), 2, 8)
        grid.addWidget(self.iso_percentile, 2, 9)

        grid.addWidget(QLabel("Fence style"), 3, 0)
        grid.addWidget(self.fence_style, 3, 1, 1, 3)

        grid.addWidget(self.update_btn, 3, 6, 1, 2)
        grid.addWidget(self.build_volume_btn, 3, 8, 1, 1)
        grid.addWidget(self.export_vtk_btn, 3, 9, 1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(self.tabs)

        self.update_btn.clicked.connect(self.update_current_view)
        self.build_volume_btn.clicked.connect(self.build_and_show_volume)
        self.export_vtk_btn.clicked.connect(self.export_volume)


    def progress_start(self, title, total):
        self._progress_logs = []
        self._progress_total = max(1, int(total))
        self._progress_count = 0
        self._progress_dialog = QProgressDialog(title, "Cancel", 0, self._progress_total, self)
        self._progress_dialog.setWindowTitle("3D GPR processing progress")
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.setValue(0)
        QApplication.processEvents()

    def progress_log(self, msg):
        self._progress_logs.append(str(msg))
        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText(str(msg))
            QApplication.processEvents()

    def progress_step(self, msg=None):
        if msg:
            self.progress_log(msg)
        self._progress_count += 1
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(min(self._progress_count, self._progress_total))
            QApplication.processEvents()
            if self._progress_dialog.wasCanceled():
                raise RuntimeError("Processing cancelled by user")

    def progress_finish(self, title="3D GPR processing finished"):
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(self._progress_total)
            QApplication.processEvents()
            self._progress_dialog.close()
            self._progress_dialog = None

        log_text = "\n".join(self._progress_logs) if self._progress_logs else "No processing log."
        log_path = Path("/home/luqman/gpr_gui/data/3d_gpr_last_run.log")
        try:
            log_path.write_text(log_text + "\n")
            self.main.status.setText(f"{title}. Log saved to {log_path}")
        except Exception:
            self.main.status.setText(title)

    def selected_lines(self):
        group = self.line_group.currentText()
        cross_step = max(1, int(self.cross_step.value()))
        inline_step = max(1, int(self.inline_step.value()))
        out = []

        for line in self.main.lines:
            parent = line.folder.parent.name.lower()

            if group == "inline only" and parent != "inline":
                continue
            if group == "crossline only" and parent != "crossline":
                continue

            if parent == "inline" and ((line.number - 1) % inline_step != 0):
                continue
            if parent == "crossline" and ((line.number - 1) % cross_step != 0) and line.number not in {1, 201}:
                continue

            out.append(line)

        out = sorted(out, key=lambda ln: (0 if ln.folder.parent.name.lower() == "inline" else 1, ln.number))
        return out[: int(self.max_lines.value())]


    def normalize_radargram_orientation(self, line, data):
        if data is None or data.ndim != 2:
            return data

        expected_traces = int(line.traces or 0)
        expected_samples = int(line.samples or 0)

        # Desired convention for all 3D tools: data[trace, sample]
        if expected_traces > 0 and expected_samples > 0:
            if abs(data.shape[0] - expected_traces) <= abs(data.shape[1] - expected_traces):
                return data
            if abs(data.shape[1] - expected_traces) < abs(data.shape[0] - expected_traces):
                return data.T.copy()

        if expected_samples > 0:
            if abs(data.shape[1] - expected_samples) <= abs(data.shape[0] - expected_samples):
                return data
            if abs(data.shape[0] - expected_samples) < abs(data.shape[1] - expected_samples):
                return data.T.copy()

        # Fallback: usually traces are fewer than samples.
        if data.shape[0] > data.shape[1]:
            return data.T.copy()

        return data


    def ensure_data(self, line):
        mode = self.mode.currentText()
        label = f"{line.folder.parent.name}/line {line.number} {line.direction.upper()}"

        if mode == "raw":
            if line.raw is None:
                self.progress_log(f"Loading raw: {label}")
            else:
                self.progress_log(f"Using cached raw: {label}")
            raw = self.normalize_radargram_orientation(line, self.main.load_raw_line(line))
            self.progress_step(f"Loaded raw: {label}, shape={raw.shape}")
            return raw

        if line.processed is not None:
            line.processed = self.normalize_radargram_orientation(line, line.processed)
            self.progress_step(f"Using cached processed: {label}, shape={line.processed.shape}")
            return line.processed

        self.progress_log(f"Loading raw for processing: {label}")
        raw = self.normalize_radargram_orientation(line, self.main.load_raw_line(line))

        self.progress_log(
            f"Processing: {label} | "
            f"dewow={float(self.main.dewow_ns.value()):.1f} ns, "
            f"background={self.main.bg.isChecked()}, "
            f"bandpass={self.main.bp.isChecked()} "
            f"{float(self.main.low_mhz.value()):.1f}-{float(self.main.high_mhz.value()):.1f} MHz, "
            f"SEC={float(self.main.sec_power.value()):.2f}, "
            f"AGC={self.main.agc.isChecked()}"
        )

        line.processed = process_gpr(
            raw,
            dt_ns=line.dt_ns,
            dewow_ns=float(self.main.dewow_ns.value()),
            remove_background=self.main.bg.isChecked(),
            use_bandpass=self.main.bp.isChecked(),
            low_mhz=float(self.main.low_mhz.value()),
            high_mhz=float(self.main.high_mhz.value()),
            sec_power=float(self.main.sec_power.value()),
            background_window_traces=int(self.main.bg_window.value()),
            use_agc=self.main.agc.isChecked(),
            agc_window_ns=float(self.main.agc_window.value()),
        )

        line.processed = self.normalize_radargram_orientation(line, line.processed)
        self.progress_step(f"Processed: {label}, raw={raw.shape}, processed={line.processed.shape}")
        return line.processed

    def trace_xyz(self, line, ntraces):
        if line.x is None or line.y is None or len(line.x) < 2:
            x = np.arange(ntraces, dtype=float)
            y = np.zeros(ntraces, dtype=float)
            z = np.zeros(ntraces, dtype=float)
            return x, y, z

        gps_i = np.linspace(0.0, 1.0, len(line.x))
        tr_i = np.linspace(0.0, 1.0, ntraces)

        x = np.interp(tr_i, gps_i, line.x)
        y = np.interp(tr_i, gps_i, line.y)

        if line.elev is not None and len(line.elev) == len(line.x):
            z = np.interp(tr_i, gps_i, line.elev)
        else:
            z = np.zeros(ntraces, dtype=float)

        return x, y, z

    def time_vector(self, line, nsamp):
        if line.dt_ns > 0:
            return np.arange(nsamp, dtype=float) * line.dt_ns
        if line.time_window_ns > 0:
            return np.linspace(0.0, line.time_window_ns, nsamp)
        return np.arange(nsamp, dtype=float)

    def robust_clip(self, arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return -1.0, 1.0
        pct = float(self.main.clip.value()) if hasattr(self.main, "clip") else 98.0
        amp = np.nanpercentile(np.abs(finite), pct)
        amp = max(float(amp), 1e-12)
        return -amp, amp

    def update_current_view(self):
        idx = self.tabs.currentIndex()
        lines = self.selected_lines()
        view_name = self.tabs.tabText(idx)

        try:
            self.progress_start(f"Preparing {view_name} for {len(lines)} selected lines...", max(1, len(lines)))

            self.progress_log(f"Selected view: {view_name}")
            self.progress_log(f"Selected lines: {len(lines)}")
            self.progress_log(f"Data mode: {self.mode.currentText()}")
            self.progress_log(f"Line group: {self.line_group.currentText()}")
            self.progress_log(f"Cross step: {int(self.cross_step.value())}, inline step: {int(self.inline_step.value())}")
            self.progress_log(f"Trace step: {int(self.trace_step.value())}, sample step: {int(self.sample_step.value())}")

            if idx == 0:
                self.plot_pyvista_fence()
            elif idx == 1:
                self.show_volume()
            elif idx == 2:
                self.show_3d_slices()
            elif idx == 3:
                self.show_isosurface()
            elif idx == 4:
                self.plot_time_slice()
            elif idx == 5:
                self.plot_depth_slice()
            elif idx == 6:
                self.plot_projection()

            self.progress_log("Finished rendering selected view.")
            self.progress_finish("3D GPR update finished")
            self.main.status.setText("Updated PyVista 3D GPR Analysis")

        except Exception as e:
            if self._progress_dialog is not None:
                self._progress_dialog.close()
                self._progress_dialog = None
            tb = traceback.format_exc()
            print("\n===== 3D ANALYSIS TRACEBACK =====")
            print(tb)
            print("===== END TRACEBACK =====\n")
            QMessageBox.critical(self, "3D analysis error", tb)

    def plot_pyvista_fence(self):
        p = self.fence_plotter
        p.clear()

        lines = self.selected_lines()
        if not lines:
            p.add_text("No selected lines", position="upper_left")
            p.reset_camera()
            return

        trace_step = max(1, int(self.trace_step.value()))
        sample_step = max(1, int(self.sample_step.value()))
        cmap = self.main.cmap.currentText() if hasattr(self.main, "cmap") else "gray"

        cache = []
        all_amp = []
        debug = []

        for line in lines:
            data = self.ensure_data(line)

            if data is None or data.ndim != 2:
                continue

            data = np.asarray(data)

            gps_n = len(line.x) if line.x is not None else 0

            # Critical rule:
            # The trace axis must correspond to the GPS coordinate axis.
            # Therefore choose/transposed orientation by matching data dimension to GPS length.
            if gps_n > 1:
                d0 = abs(data.shape[0] - gps_n)
                d1 = abs(data.shape[1] - gps_n)

                if d1 < d0:
                    data = data.T.copy()

            # Fallback: radargram convention should be traces x samples.
            if data.shape[0] > data.shape[1] and gps_n <= 1:
                data = data.T.copy()

            ntrace, nsamp = data.shape

            if ntrace < 2 or nsamp < 2:
                continue

            x_full, y_full, _ = self.trace_xyz(line, ntrace)
            t_full = self.time_vector(line, nsamp)

            if len(x_full) != ntrace or len(y_full) != ntrace:
                raise ValueError(
                    f"GPS interpolation mismatch for {line.folder.parent.name}/line {line.number}: "
                    f"data={data.shape}, len(x)={len(x_full)}, len(y)={len(y_full)}, gps_n={gps_n}"
                )

            if len(t_full) != nsamp:
                raise ValueError(
                    f"Time vector mismatch for {line.folder.parent.name}/line {line.number}: "
                    f"data={data.shape}, len(t)={len(t_full)}"
                )

            tr_idx = np.arange(0, ntrace, trace_step)
            sm_idx = np.arange(0, nsamp, sample_step)

            if len(tr_idx) < 2 or len(sm_idx) < 2:
                continue

            x = x_full[tr_idx]
            y = y_full[tr_idx]
            z = -t_full[sm_idx]

            amp = data[np.ix_(tr_idx, sm_idx)].T

            expected = (len(z), len(x))
            if amp.shape != expected:
                raise ValueError(
                    f"Fence shape mismatch at {line.folder.parent.name}/line {line.number}: "
                    f"final data={data.shape}, amp={amp.shape}, expected={expected}, "
                    f"gps_n={gps_n}, trace_step={trace_step}, sample_step={sample_step}"
                )

            cache.append((line, x, y, z, amp))
            all_amp.append(amp.ravel())
            debug.append(f"{line.folder.parent.name}/line {line.number}: data={data.shape}, gps={gps_n}, fence={amp.shape}")

        if not cache:
            p.add_text("No radargram data loaded", position="upper_left")
            p.reset_camera()
            return

        vmin, vmax = self.robust_clip(np.concatenate(all_amp))

        for line, x, y, z, amp in cache:
            X = np.repeat(x[None, :], len(z), axis=0)
            Y = np.repeat(y[None, :], len(z), axis=0)
            Z = np.repeat(z[:, None], len(x), axis=1)

            if X.shape != Y.shape or X.shape != Z.shape or X.shape != amp.shape:
                raise ValueError(
                    f"Final mesh mismatch at {line.folder.parent.name}/line {line.number}: "
                    f"X={X.shape}, Y={Y.shape}, Z={Z.shape}, amp={amp.shape}"
                )

            grid = pv.StructuredGrid(X, Y, Z)
            grid["amplitude"] = amp.ravel(order="F")

            parent = line.folder.parent.name.lower()
            opacity = 0.95 if parent == "inline" else 0.55

            p.add_mesh(
                grid,
                scalars="amplitude",
                cmap=cmap,
                clim=(vmin, vmax),
                opacity=opacity,
                show_scalar_bar=False,
            )

            if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
                label = ("I" if parent == "inline" else "C") + str(line.number)
                j = len(x) // 2
                p.add_point_labels(
                    np.array([[x[j], y[j], 0.0]]),
                    [label],
                    font_size=10,
                    point_size=0,
                    shape_opacity=0.0,
                )

        p.add_axes()
        p.add_text(
            f"PyVista 3D fence: {len(cache)} lines | trace step={trace_step}, sample step={sample_step}",
            position="upper_left",
            font_size=10,
        )

        self.progress_log("Fence debug summary:")
        for row in debug[:20]:
            self.progress_log("  " + row)
        if len(debug) > 20:
            self.progress_log(f"  ... {len(debug)-20} more lines")

        p.view_isometric()
        p.reset_camera()
        p.render()

    def collect_volume_points(self):
        xs, ys, zs, vals = [], [], [], []

        tmin = float(self.proj_tmin.value())
        tmax = float(self.proj_tmax.value())
        t_step = max(1, int(self.trace_step.value()))
        s_step = max(1, int(self.sample_step.value()))

        for line in self.selected_lines():
            data = self.ensure_data(line)
            if data is None or data.ndim != 2:
                continue

            t = self.time_vector(line, data.shape[1])
            good_s = np.where((t >= tmin) & (t <= tmax))[0]
            good_s = good_s[::s_step]
            if good_s.size < 2:
                continue

            tr_idx = np.arange(0, data.shape[0], t_step)
            x, y, _ = self.trace_xyz(line, data.shape[0])

            for j in good_s:
                xs.extend(x[tr_idx])
                ys.extend(y[tr_idx])
                zs.extend(-t[j] * np.ones_like(tr_idx, dtype=float))
                vals.extend(data[tr_idx, j])

        return np.asarray(xs), np.asarray(ys), np.asarray(zs), np.asarray(vals)

    def build_volume(self):
        from scipy.interpolate import griddata

        x, y, z, val = self.collect_volume_points()

        if len(val) < 20:
            raise ValueError("Not enough points to build volume. Lower steps or include more lines.")

        nx = int(self.nx.value())
        ny = int(self.ny.value())
        nz = int(self.nz.value())

        xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
        yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
        zi = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), nz)

        X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")

        V = griddata((x, y, z), val, (X, Y, Z), method="linear")
        nearest = griddata((x, y, z), val, (X, Y, Z), method="nearest")
        V = np.where(np.isfinite(V), V, nearest)
        V = np.nan_to_num(V, nan=0.0)

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.origin = (float(xi[0]), float(yi[0]), float(zi[0]))
        grid.spacing = (
            float((xi[-1] - xi[0]) / max(nx - 1, 1)),
            float((yi[-1] - yi[0]) / max(ny - 1, 1)),
            float((zi[-1] - zi[0]) / max(nz - 1, 1)),
        )
        grid["amplitude"] = V.ravel(order="F")
        self.volume_grid = grid
        return grid

    def build_and_show_volume(self):
        lines = self.selected_lines()
        try:
            self.progress_start(f"Building PyVista 3D volume from {len(lines)} selected lines...", max(1, len(lines)))
            self.progress_log("Building interpolated 3D amplitude volume.")
            self.progress_log(f"Volume grid: nx={int(self.nx.value())}, ny={int(self.ny.value())}, nz={int(self.nz.value())}")
            self.progress_log(f"Projection time window: {float(self.proj_tmin.value()):.1f}–{float(self.proj_tmax.value()):.1f} ns")
            self.progress_log(f"Trace step={int(self.trace_step.value())}, sample step={int(self.sample_step.value())}")

            self.volume_grid = self.build_volume()

            self.progress_log("Volume interpolation finished. Rendering volume, slices, and isosurface.")
            self.show_volume()
            self.show_3d_slices()
            self.show_isosurface()

            self.progress_log("Finished building and rendering PyVista volume.")
            self.progress_finish("3D GPR volume build finished")
            self.main.status.setText("Built PyVista GPR volume")

        except Exception as e:
            if self._progress_dialog is not None:
                self._progress_dialog.close()
                self._progress_dialog = None
            tb = traceback.format_exc()
            print("\n===== VOLUME BUILD TRACEBACK =====")
            print(tb)
            print("===== END TRACEBACK =====\n")
            QMessageBox.critical(self, "Volume build error", tb)

    def show_volume(self):
        p = self.volume_plotter
        p.clear()

        if self.volume_grid is None:
            p.add_text("Click 'Build / refresh PyVista volume' first", position="upper_left")
            p.reset_camera()
            return

        arr = self.volume_grid["amplitude"]
        vmin, vmax = self.robust_clip(arr)
        p.add_volume(self.volume_grid, scalars="amplitude", cmap="seismic", clim=(vmin, vmax), opacity="sigmoid", shade=False)
        p.add_axes()
        p.add_text("PyVista interpolated 3D GPR amplitude volume", position="upper_left", font_size=10)
        p.view_isometric()
        p.reset_camera()
        p.render()

    def show_3d_slices(self):
        p = self.slice3d_plotter
        p.clear()

        if self.volume_grid is None:
            p.add_text("Click 'Build / refresh PyVista volume' first", position="upper_left")
            p.reset_camera()
            return

        arr = self.volume_grid["amplitude"]
        vmin, vmax = self.robust_clip(arr)

        slices = self.volume_grid.slice_orthogonal()
        p.add_mesh(slices, scalars="amplitude", cmap="seismic", clim=(vmin, vmax), opacity=0.95)
        p.add_outline(self.volume_grid)
        p.add_axes()
        p.add_text("PyVista orthogonal slices through interpolated GPR volume", position="upper_left", font_size=10)
        p.view_isometric()
        p.reset_camera()
        p.render()

    def show_isosurface(self):
        p = self.iso_plotter
        p.clear()

        if self.volume_grid is None:
            p.add_text("Click 'Build / refresh PyVista volume' first", position="upper_left")
            p.reset_camera()
            return

        arr = np.abs(self.volume_grid["amplitude"])
        level = float(np.nanpercentile(arr[np.isfinite(arr)], float(self.iso_percentile.value())))
        grid = self.volume_grid.copy()
        grid["abs_amplitude"] = arr

        try:
            surf = grid.contour([level], scalars="abs_amplitude")
            p.add_mesh(surf, scalars="abs_amplitude", cmap="inferno", opacity=0.75)
            p.add_outline(grid)
            p.add_text(f"PyVista isosurface |amplitude| >= p{float(self.iso_percentile.value()):.1f}", position="upper_left", font_size=10)
        except Exception as e:
            p.add_text(f"Could not build isosurface: {e}", position="upper_left")

        p.add_axes()
        p.view_isometric()
        p.reset_camera()
        p.render()

    def export_volume(self):
        if self.volume_grid is None:
            QMessageBox.warning(self, "No volume", "Build the PyVista volume first.")
            return

        out = Path("/home/luqman/gpr_gui/data/gpr_3d_volume.vti")
        self.volume_grid.save(out)
        QMessageBox.information(self, "Exported", f"Saved:\n{out}")
        self.main.status.setText(f"Exported volume: {out}")

    def collect_slice_points(self, time_ns):
        xs, ys, vals = [], [], []

        for line in self.selected_lines():
            data = self.ensure_data(line)
            if data is None or data.ndim != 2:
                continue

            t = self.time_vector(line, data.shape[1])
            if time_ns < t.min() or time_ns > t.max():
                continue

            sample = int(np.argmin(np.abs(t - time_ns)))
            x, y, _ = self.trace_xyz(line, data.shape[0])

            step = max(1, int(self.trace_step.value()))
            idx = np.arange(0, data.shape[0], step)

            xs.extend(x[idx])
            ys.extend(y[idx])
            vals.extend(data[idx, sample])

        return np.asarray(xs), np.asarray(ys), np.asarray(vals)

    def plot_slice_on_canvas(self, canvas, time_ns, title):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        x, y, val = self.collect_slice_points(time_ns)

        if len(val) < 4:
            ax.text(0.5, 0.5, "Not enough slice points", transform=ax.transAxes, ha="center", va="center")
            canvas.draw_idle()
            return

        vmin, vmax = self.robust_clip(val)
        cmap = self.main.cmap.currentText() if hasattr(self.main, "cmap") else "gray"

        try:
            from scipy.interpolate import griddata
            nx, ny = 220, 160
            xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
            yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), val, (X, Y), method="linear")
            ax.imshow(Z, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.scatter(x, y, s=2, c="k", alpha=0.18)
        except Exception:
            ax.scatter(x, y, c=val, s=8, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        canvas.remember_home()
        canvas.draw_idle()

    def plot_time_slice(self):
        t = float(self.slice_time.value())
        self.plot_slice_on_canvas(self.time_canvas, t, f"Time slice amplitude map at {t:.1f} ns")

    def plot_depth_slice(self):
        depth = float(self.depth.value())
        vel = max(float(self.velocity.value()), 1e-9)
        t = 2.0 * depth / vel
        self.plot_slice_on_canvas(self.depth_canvas, t, f"Depth slice amplitude map at {depth:.2f} m, v={vel:.3f} m/ns, TWT={t:.1f} ns")

    def plot_projection(self):
        self.proj_canvas.fig.clear()
        ax = self.proj_canvas.fig.add_subplot(111)

        tmin = float(self.proj_tmin.value())
        tmax = float(self.proj_tmax.value())
        if tmax <= tmin:
            tmax = tmin + 1.0

        xs, ys, vals = [], [], []

        for line in self.selected_lines():
            data = self.ensure_data(line)
            if data is None or data.ndim != 2:
                continue

            t = self.time_vector(line, data.shape[1])
            good_t = np.where((t >= tmin) & (t <= tmax))[0]
            if good_t.size < 2:
                continue

            x, y, _ = self.trace_xyz(line, data.shape[0])
            step = max(1, int(self.trace_step.value()))
            idx = np.arange(0, data.shape[0], step)

            amp = np.nanmax(np.abs(data[np.ix_(idx, good_t)]), axis=1)
            xs.extend(x[idx])
            ys.extend(y[idx])
            vals.extend(amp)

        x = np.asarray(xs)
        y = np.asarray(ys)
        val = np.asarray(vals)

        if len(val) < 4:
            ax.text(0.5, 0.5, "Not enough projection points", transform=ax.transAxes, ha="center", va="center")
            self.proj_canvas.draw_idle()
            return

        finite = val[np.isfinite(val)]
        vmax = np.nanpercentile(finite, float(self.main.clip.value())) if finite.size and hasattr(self.main, "clip") else 1.0
        vmax = max(float(vmax), 1e-12)

        try:
            from scipy.interpolate import griddata
            nx, ny = 220, 160
            xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
            yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), val, (X, Y), method="linear")
            ax.imshow(Z, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="inferno", vmin=0, vmax=vmax, aspect="equal")
            ax.scatter(x, y, s=2, c="k", alpha=0.18)
        except Exception:
            ax.scatter(x, y, c=val, s=8, cmap="inferno", vmin=0, vmax=vmax)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title(f"Maximum absolute amplitude projection, {tmin:.1f}–{tmax:.1f} ns")
        ax.grid(True, alpha=0.25)
        self.proj_canvas.remember_home()
        self.proj_canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("MALÅ GPR Fieldwork Processing GUI - inline + crossline")
        self.root = root
        self.lines: list[GPRLine] = []
        self.line_tabs: dict[str, LineTab] = {}

        self.tabs = QTabWidget()

        self.root_label = QLabel("")
        self.btn_choose = QPushButton("Choose project folder")
        self.btn_reload = QPushButton("Reload")
        self.btn_load = QPushButton("Load raw")
        self.btn_process = QPushButton("Process current line")

        self.dewow_ns = QDoubleSpinBox()
        self.dewow_ns.setRange(0.0, 500.0)
        self.dewow_ns.setValue(25.0)
        self.dewow_ns.setSuffix(" ns")

        self.low_mhz = QDoubleSpinBox()
        self.low_mhz.setRange(0.1, 5000.0)
        self.low_mhz.setValue(50.0)
        self.low_mhz.setSuffix(" MHz")

        self.high_mhz = QDoubleSpinBox()
        self.high_mhz.setRange(0.1, 5000.0)
        self.high_mhz.setValue(250.0)
        self.high_mhz.setSuffix(" MHz")

        self.sec_power = QDoubleSpinBox()
        self.sec_power.setRange(0.0, 10.0)
        self.sec_power.setValue(0.9)
        self.sec_power.setSingleStep(0.1)

        self.clip = QDoubleSpinBox()
        self.clip.setRange(80.0, 99.99)
        self.clip.setValue(99.7)
        self.clip.setSuffix(" %")

        self.bg = QCheckBox("Local median background removal")
        self.bg.setChecked(True)

        self.bp = QCheckBox("Bandpass")
        self.bp.setChecked(True)

        self.bg_window = QSpinBox()
        self.bg_window.setRange(11, 1001)
        self.bg_window.setSingleStep(10)
        self.bg_window.setValue(151)

        self.agc = QCheckBox("AGC gain")
        self.agc.setChecked(True)

        self.agc_window = QDoubleSpinBox()
        self.agc_window.setRange(5.0, 300.0)
        self.agc_window.setValue(80.0)
        self.agc_window.setSuffix(" ns")

        self.display_tmin = QDoubleSpinBox()
        self.display_tmin.setRange(0.0, 500.0)
        self.display_tmin.setValue(35.0)
        self.display_tmin.setSuffix(" ns")

        self.display_tmax = QDoubleSpinBox()
        self.display_tmax.setRange(20.0, 1000.0)
        self.display_tmax.setValue(250.0)
        self.display_tmax.setSuffix(" ns")

        self.vertical_exag = QDoubleSpinBox()
        self.vertical_exag.setRange(0.5, 5.0)
        self.vertical_exag.setValue(1.5)
        self.vertical_exag.setSingleStep(0.1)
        self.vertical_exag.setSuffix("×")

        self.align_display = QCheckBox("Display convention: A left → B right")
        self.align_display.setChecked(True)
        self.align_display.setEnabled(False)

        self.scale_mode = QComboBox()
        self.scale_mode.addItems(["linear", "symlog", "log(abs)"])
        self.scale_mode.setCurrentText("symlog")

        self.cmap = QComboBox()
        self.cmap.addItems(["gray", "seismic", "RdBu_r", "Greys", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"])
        self.cmap.setCurrentText("seismic")

        self.status = QLabel("")

        top = QWidget()
        main_layout = QVBoxLayout(top)

        row0 = QHBoxLayout()
        row0.addWidget(self.root_label)
        row0.addStretch(1)
        row0.addWidget(self.btn_choose)
        row0.addWidget(self.btn_reload)
        main_layout.addLayout(row0)

        controls = QWidget()
        grid = QGridLayout(controls)
        grid.addWidget(self.btn_load, 0, 0)
        grid.addWidget(self.btn_process, 0, 1)
        grid.addWidget(QLabel("Dewow window"), 0, 2)
        grid.addWidget(self.dewow_ns, 0, 3)
        grid.addWidget(QLabel("Low cut"), 0, 4)
        grid.addWidget(self.low_mhz, 0, 5)
        grid.addWidget(QLabel("High cut"), 0, 6)
        grid.addWidget(self.high_mhz, 0, 7)
        grid.addWidget(QLabel("SEC gain power"), 1, 0)
        grid.addWidget(self.sec_power, 1, 1)
        grid.addWidget(QLabel("Display clip"), 1, 2)
        grid.addWidget(self.clip, 1, 3)
        grid.addWidget(self.bg, 1, 4)
        grid.addWidget(self.bp, 1, 5)
        grid.addWidget(self.align_display, 1, 6, 1, 2)
        grid.addWidget(QLabel("Scale"), 2, 0)
        grid.addWidget(self.scale_mode, 2, 1)
        grid.addWidget(QLabel("Colour map"), 2, 2)
        grid.addWidget(self.cmap, 2, 3)
        grid.addWidget(QLabel("BG window"), 2, 4)
        grid.addWidget(self.bg_window, 2, 5)
        grid.addWidget(self.agc, 2, 6)
        grid.addWidget(QLabel("AGC window"), 2, 7)
        grid.addWidget(self.agc_window, 2, 8)
        grid.addWidget(QLabel("Display min time"), 3, 0)
        grid.addWidget(self.display_tmin, 3, 1)
        grid.addWidget(QLabel("Display max time"), 3, 2)
        grid.addWidget(self.display_tmax, 3, 3)
        grid.addWidget(QLabel("Vertical exaggeration"), 3, 4)
        grid.addWidget(self.vertical_exag, 3, 5)
        self.processing_controls = controls
        self.processing_controls.setVisible(False)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.status)

        self.setCentralWidget(top)

        self.btn_choose.clicked.connect(self.choose_root)
        self.btn_reload.clicked.connect(self.load_project)
        self.btn_load.clicked.connect(self.load_current_raw)
        self.btn_process.clicked.connect(self.process_current_line)

        for w in [self.scale_mode, self.cmap]:
            w.currentTextChanged.connect(self.redraw_all)
        for w in [self.clip, self.sec_power, self.dewow_ns, self.low_mhz, self.high_mhz, self.bg_window, self.agc_window, self.display_tmin, self.display_tmax, self.vertical_exag]:
            w.valueChanged.connect(self.redraw_all)
        for w in [self.align_display, self.agc]:
            w.stateChanged.connect(self.redraw_all)

        self.load_project()


    def move_processing_controls_to_active_line_tab(self, idx=None):
        if not hasattr(self, "processing_controls"):
            return

        if idx is None:
            idx = self.tabs.currentIndex()

        title = self.tabs.tabText(idx).lower() if idx >= 0 else ""

        if title.startswith("inline") and hasattr(self, "inline_layout"):
            self.processing_controls.setVisible(True)
            self.inline_layout.insertWidget(0, self.processing_controls)
            return

        if title.startswith("crossline") and hasattr(self, "crossline_layout"):
            self.processing_controls.setVisible(True)
            self.crossline_layout.insertWidget(0, self.processing_controls)
            return

        self.processing_controls.setVisible(False)


    def choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose MALÅ project root", str(self.root))
        if d:
            self.root = Path(d)
            self.load_project()

    def discover_lines(self):
        folders = []
        for folder in self.root.rglob("*"):
            if not folder.is_dir():
                continue
            name = folder.name.lower()
            if re.fullmatch(r"line\d+_(ab|ba)_t", name) or re.fullmatch(r"line_\d+_(cd|dc)_t", name):
                folders.append(folder)

        def folder_key(folder: Path):
            name = folder.name.lower()
            m = re.fullmatch(r"line(\d+)_(ab|ba)_t", name)
            if m:
                return (0, int(m.group(1)))
            m = re.fullmatch(r"line_(\d+)_(cd|dc)_t", name)
            if m:
                return (1, int(m.group(1)))
            return (9, 999999)

        lines = []

        for folder in sorted(folders, key=folder_key):
            name = folder.name.lower()

            m = re.fullmatch(r"line(\d+)_(ab|ba)_t", name)
            if m:
                number = int(m.group(1))
                direction = m.group(2)
            else:
                m = re.fullmatch(r"line_(\d+)_(cd|dc)_t", name)
                if not m:
                    continue
                number = int(m.group(1))
                direction = m.group(2)

            rad = next(folder.glob("*.rad"), None)
            rd7 = next(folder.glob("*.rd7"), None)
            cor = next(folder.glob("*.cor"), None)
            mrk = next(folder.glob("*.mrk"), None)
            mrkj = next(folder.glob("*.mrkj"), None)
            proj = next(folder.glob("*.proj"), None)

            if rad is None or rd7 is None:
                continue

            meta = parse_rad(rad)
            samples = int(meta["samples"]) if meta["samples"] else 0

            if samples <= 0:
                raise ValueError(f"Cannot read number of samples from {rad}")

            traces_guess = None
            if rd7.stat().st_size % (samples * 4) == 0:
                traces_guess = rd7.stat().st_size // (samples * 4)
            elif rd7.stat().st_size % (samples * 2) == 0:
                traces_guess = rd7.stat().st_size // (samples * 2)

            tw = float(meta["time_window_ns"]) if meta["time_window_ns"] else 0.0
            dt = tw / (samples - 1) if tw > 0 and samples > 1 else 0.0
            fs_mhz = 1000.0 / dt if dt > 0 else 0.0

            line = GPRLine(
                number=number,
                name=f"{folder.parent.name}/{folder.name}",
                direction=direction,
                folder=folder,
                rad=rad,
                rd7=rd7,
                cor=cor,
                mrk=mrk,
                mrkj=mrkj,
                proj=proj,
                samples=samples,
                traces=int(traces_guess or 0),
                time_window_ns=tw,
                dt_ns=dt,
                fs_mhz=fs_mhz,
                antenna=str(meta.get("antenna") or ""),
            )
            lines.append(line)

        return sorted(lines, key=lambda x: (0 if x.folder.parent.name.lower() == "inline" else 1, x.number))

    def load_project(self):
        try:
            self.lines = self.discover_lines()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        if not self.lines:
            QMessageBox.warning(self, "No lines found", f"No inline/crossline MALÅ folders found in:\n{self.root}")
            return

        self.root_label.setText(f"Data root: {self.root}")

        if hasattr(self, "processing_controls"):
            self.processing_controls.setParent(None)
            self.processing_controls.setVisible(False)

        self.tabs.clear()
        self.line_tabs.clear()

        self.load_all_gps()

        gps_canvas = MplCanvas(width=9, height=7)
        self.plot_gps_plan(gps_canvas)
        self.tabs.addTab(gps_canvas, "GPS plan view")

        map3d_canvas = MplCanvas(width=9, height=7)
        self.plot_3d_map(map3d_canvas)
        self.tabs.addTab(map3d_canvas, "3D elevation map")

        self.gpr3d_tab = None

        self.inline_container = QWidget()
        self.inline_layout = QVBoxLayout(self.inline_container)
        self.inline_layout.setContentsMargins(0, 0, 0, 0)
        self.inline_tabs = QTabWidget()
        self.inline_layout.addWidget(self.inline_tabs)

        self.crossline_container = QWidget()
        self.crossline_layout = QVBoxLayout(self.crossline_container)
        self.crossline_layout.setContentsMargins(0, 0, 0, 0)
        self.crossline_tabs = QTabWidget()
        self.crossline_layout.addWidget(self.crossline_tabs)

        inline_count = 0
        crossline_count = 0

        for line in self.lines:
            tab = LineTab(line, self)
            tab.plot()
            self.line_tabs[line.name] = tab

            parent = line.folder.parent.name.lower()
            if parent == "inline":
                self.inline_tabs.addTab(tab, f"I{line.number}: {line.direction.upper()}")
                inline_count += 1
            elif parent == "crossline":
                self.crossline_tabs.addTab(tab, f"C{line.number}: {line.direction.upper()}")
                crossline_count += 1
            else:
                self.inline_tabs.addTab(tab, f"{line.name}: {line.direction.upper()}")

        self.tabs.addTab(self.inline_container, f"Inline ({inline_count})")
        self.tabs.addTab(self.crossline_container, f"Crossline ({crossline_count})")

        self.gpr3d_tab = GPR3DAnalysisTab(self)
        self.tabs.addTab(self.gpr3d_tab, "3D GPR Analysis")

        try:
            self.tabs.currentChanged.disconnect(self.move_processing_controls_to_active_line_tab)
        except Exception:
            pass
        self.tabs.currentChanged.connect(self.move_processing_controls_to_active_line_tab)
        self.move_processing_controls_to_active_line_tab(self.tabs.currentIndex())

        self.status.setText(
            f"Loaded {len(self.lines)} lines from {self.root}: "
            f"{inline_count} inline, {crossline_count} crossline. "
            f"PyVista/VTK 3D enabled. scipy bandpass available={SCIPY_OK}"
        )

    def load_all_gps(self):
        valid_lats, valid_lons = [], []

        for line in self.lines:
            lat, lon, elev = parse_cor(line.cor)
            line.lat, line.lon, line.elev = lat, lon, elev
            if lat is not None:
                valid_lats.append(lat)
                valid_lons.append(lon)

        if not valid_lats:
            return

        lat0 = float(np.nanmean(np.concatenate(valid_lats)))
        lon0 = float(np.nanmean(np.concatenate(valid_lons)))

        for line in self.lines:
            if line.lat is None:
                continue
            x, y = latlon_to_xy(line.lat, line.lon, lat0, lon0)
            line.x, line.y = x, y
            line.dist = cumulative_distance(x, y)

    def plot_gps_plan(self, canvas: MplCanvas):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        inline_lines = [ln for ln in self.lines if ln.folder.parent.name.lower() == "inline"]
        cross_lines = [ln for ln in self.lines if ln.folder.parent.name.lower() == "crossline"]

        for line in sorted(cross_lines, key=lambda ln: ln.number):
            if line.x is None or line.y is None:
                continue
            ax.plot(line.x, line.y, linewidth=0.6, alpha=0.35)
            if line.number in {1, 50, 90, 92, 100, 150, 201}:
                j = len(line.x) // 2
                ax.text(line.x[j], line.y[j], f"C{line.number}", fontsize=8)

        for line in sorted(inline_lines, key=lambda ln: ln.number):
            if line.x is None or line.y is None:
                continue
            ax.plot(line.x, line.y, linewidth=2.0, alpha=0.95)
            j = len(line.x) // 2
            ax.text(line.x[j], line.y[j], f"I{line.number}", fontsize=9, weight="bold")
            ax.plot(line.x[0], line.y[0], marker="o", markersize=4)
            ax.plot(line.x[-1], line.y[-1], marker="x", markersize=4)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title("GPS plan view: inline + crossline layout; circle=start, x=end")
        ax.grid(True, alpha=0.3)
        canvas.remember_home()
        canvas.draw_idle()

    def plot_3d_map(self, canvas: MplCanvas):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111, projection="3d")

        inline_lines = [ln for ln in self.lines if ln.folder.parent.name.lower() == "inline"]
        cross_lines = [ln for ln in self.lines if ln.folder.parent.name.lower() == "crossline"]

        any_data = False

        for line in sorted(cross_lines, key=lambda ln: ln.number):
            if line.x is None or line.y is None or line.elev is None:
                continue
            good = np.isfinite(line.x) & np.isfinite(line.y) & np.isfinite(line.elev)
            if np.count_nonzero(good) < 2:
                continue
            ax.plot(line.x[good], line.y[good], line.elev[good], linewidth=0.6, alpha=0.35)
            if line.number in {1, 50, 90, 92, 100, 150, 201}:
                idxs = np.flatnonzero(good)
                idx = idxs[len(idxs) // 2]
                ax.text(line.x[idx], line.y[idx], line.elev[idx], f"C{line.number}", fontsize=8)
            any_data = True

        for line in sorted(inline_lines, key=lambda ln: ln.number):
            if line.x is None or line.y is None or line.elev is None:
                continue
            good = np.isfinite(line.x) & np.isfinite(line.y) & np.isfinite(line.elev)
            if np.count_nonzero(good) < 2:
                continue
            ax.plot(line.x[good], line.y[good], line.elev[good], linewidth=2.0, alpha=0.95)
            idxs = np.flatnonzero(good)
            idx = idxs[len(idxs) // 2]
            ax.text(line.x[idx], line.y[idx], line.elev[idx], f"I{line.number}", fontsize=8)
            any_data = True

        if not any_data:
            ax.text2D(0.5, 0.5, "No usable 3D GPS/elevation data", transform=ax.transAxes, ha="center", va="center")

        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_zlabel("Elevation [m]")
        ax.set_title("3D GPS elevation map: drag with mouse to rotate")
        try:
            ax.set_box_aspect((1, 1, 0.25))
        except Exception:
            pass
        ax.grid(True, alpha=0.3)
        canvas.remember_home()
        canvas.draw_idle()

    def current_line_tab(self) -> Optional[LineTab]:
        """
        Return the active radargram LineTab.

        Needed because Inline/Crossline tabs are now containers:
        main tabs -> inline_container/crossline_container -> nested QTabWidget -> LineTab
        """
        def find_line_tab(widget):
            if widget is None:
                return None

            if isinstance(widget, LineTab):
                return widget

            if isinstance(widget, QTabWidget):
                return find_line_tab(widget.currentWidget())

            # Search children recursively, preferring the visible nested QTabWidget.
            try:
                tabs = widget.findChildren(QTabWidget)
                for tabw in tabs:
                    if tabw.isVisible():
                        found = find_line_tab(tabw.currentWidget())
                        if found is not None:
                            return found
                for tabw in tabs:
                    found = find_line_tab(tabw.currentWidget())
                    if found is not None:
                        return found
            except Exception:
                pass

            return None

        return find_line_tab(self.tabs.currentWidget())

    def load_raw_line(self, line: GPRLine):
        if line.raw is None:
            data = read_rd7(line.rd7, line.samples, line.traces)
            line.raw = data
            line.traces = data.shape[0]

            if line.dist is None or len(line.dist) != line.traces:
                if line.x is not None and len(line.x) >= 2:
                    d0 = cumulative_distance(line.x, line.y)
                    line.dist = np.interp(
                        np.linspace(0, len(d0) - 1, line.traces),
                        np.arange(len(d0)),
                        d0
                    )
                else:
                    line.dist = np.arange(line.traces, dtype=float)

        return line.raw

    def load_current_raw(self):
        tab = self.current_line_tab()
        if tab is None:
            return
        try:
            self.load_raw_line(tab.line)
            tab.plot()
            ln = tab.line
            length = float(ln.dist[-1]) if ln.dist is not None and len(ln.dist) else np.nan
            self.status.setText(
                f"{ln.name} | traces={ln.traces}, samples={ln.samples}, "
                f"timewindow={ln.time_window_ns:.2f} ns, dt={ln.dt_ns:.4f} ns, "
                f"fs={ln.fs_mhz:.1f} MHz, length={length:.2f} m"
            )
        except Exception as e:
            QMessageBox.critical(self, "Raw load error", str(e))

    def process_current_line(self):
        tab = self.current_line_tab()
        if tab is None:
            return

        try:
            line = tab.line
            raw = self.load_raw_line(line)
            line.processed = process_gpr(
                raw,
                dt_ns=line.dt_ns,
                dewow_ns=float(self.dewow_ns.value()),
                remove_background=self.bg.isChecked(),
                use_bandpass=self.bp.isChecked(),
                low_mhz=float(self.low_mhz.value()),
                high_mhz=float(self.high_mhz.value()),
                sec_power=float(self.sec_power.value()),
                background_window_traces=int(self.bg_window.value()),
                use_agc=self.agc.isChecked(),
                agc_window_ns=float(self.agc_window.value()),
            )
            tab.plot()
            self.status.setText(f"Processed {line.name}")
        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))

    def redraw_all(self):
        w = self.tabs.currentWidget()

        try:
            if isinstance(w, MplCanvas):
                title = self.tabs.tabText(self.tabs.currentIndex()).lower()
                if "gps" in title:
                    self.plot_gps_plan(w)
                elif "3d elevation" in title:
                    self.plot_3d_map(w)
            elif isinstance(w, GPR3DAnalysisTab):
                w.update_current_view()
            elif isinstance(w, QTabWidget):
                inner = w.currentWidget()
                if isinstance(inner, LineTab):
                    inner.plot()
            elif isinstance(w, LineTab):
                w.plot()
        except Exception as e:
            self.status.setText(f"Redraw skipped: {e}")

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/luqman/gpr_gui/data")
    app = QApplication(sys.argv)
    win = MainWindow(root)
    win.resize(1800, 950)
    win.show()
    sys.exit(app.exec())




# ---- HARD OVERRIDE: robust PyVista fence renderer ----

def _gpr3d_fixed_orient(self, line, data):
    data = np.asarray(data)
    if data.ndim != 2:
        return data

    gps_n = len(line.x) if getattr(line, "x", None) is not None else 0
    expected_traces = int(getattr(line, "traces", 0) or 0)
    expected_samples = int(getattr(line, "samples", 0) or 0)

    # Prefer matching the trace axis to GPS count.
    if gps_n > 1:
        d0 = abs(data.shape[0] - gps_n)
        d1 = abs(data.shape[1] - gps_n)
        if d1 < d0:
            return data.T.copy()
        return data

    # Otherwise prefer metadata trace/sample convention.
    if expected_traces > 0:
        d0 = abs(data.shape[0] - expected_traces)
        d1 = abs(data.shape[1] - expected_traces)
        if d1 < d0:
            return data.T.copy()
        return data

    if expected_samples > 0:
        d0 = abs(data.shape[0] - expected_samples)
        d1 = abs(data.shape[1] - expected_samples)
        if d0 < d1:
            return data.T.copy()
        return data

    # Fallback: traces usually fewer than samples.
    return data.T.copy() if data.shape[0] > data.shape[1] else data


def _gpr3d_fixed_ensure_data(self, line):
    mode = self.mode.currentText()
    label = f"{line.folder.parent.name}/line {line.number} {line.direction.upper()}"

    if mode == "raw":
        raw = _gpr3d_fixed_orient(self, line, self.main.load_raw_line(line))
        if hasattr(self, "progress_step"):
            self.progress_step(f"Loaded raw: {label}, shape={raw.shape}")
        return raw

    if line.processed is not None:
        line.processed = _gpr3d_fixed_orient(self, line, line.processed)
        if hasattr(self, "progress_step"):
            self.progress_step(f"Using cached processed: {label}, shape={line.processed.shape}")
        return line.processed

    raw = _gpr3d_fixed_orient(self, line, self.main.load_raw_line(line))

    if hasattr(self, "progress_log"):
        self.progress_log(f"Processing: {label}, raw shape={raw.shape}")

    line.processed = process_gpr(
        raw,
        dt_ns=line.dt_ns,
        dewow_ns=float(self.main.dewow_ns.value()),
        remove_background=self.main.bg.isChecked(),
        use_bandpass=self.main.bp.isChecked(),
        low_mhz=float(self.main.low_mhz.value()),
        high_mhz=float(self.main.high_mhz.value()),
        sec_power=float(self.main.sec_power.value()),
        background_window_traces=int(self.main.bg_window.value()),
        use_agc=self.main.agc.isChecked(),
        agc_window_ns=float(self.main.agc_window.value()),
    )

    line.processed = _gpr3d_fixed_orient(self, line, line.processed)

    if hasattr(self, "progress_step"):
        self.progress_step(f"Processed: {label}, processed shape={line.processed.shape}")

    return line.processed


def _gpr3d_fixed_plot_pyvista_fence(self):
    p = self.fence_plotter
    p.clear()

    lines = self.selected_lines()
    if not lines:
        p.add_text("No selected lines", position="upper_left")
        p.reset_camera()
        return

    trace_step = max(1, int(self.trace_step.value()))
    sample_step = max(1, int(self.sample_step.value()))
    cmap = self.main.cmap.currentText() if hasattr(self.main, "cmap") else "gray"

    cache = []
    all_amp = []
    debug = []

    for line in lines:
        data = _gpr3d_fixed_ensure_data(self, line)

        if data is None or data.ndim != 2:
            continue

        data = _gpr3d_fixed_orient(self, line, data)
        ntrace, nsamp = data.shape

        if ntrace < 2 or nsamp < 2:
            continue

        x_full, y_full, _ = self.trace_xyz(line, ntrace)
        t_full = self.time_vector(line, nsamp)

        tr_idx = np.arange(0, ntrace, trace_step)
        sm_idx = np.arange(0, nsamp, sample_step)

        if len(tr_idx) < 2 or len(sm_idx) < 2:
            continue

        x = np.asarray(x_full)[tr_idx]
        y = np.asarray(y_full)[tr_idx]
        z = -np.asarray(t_full)[sm_idx]

        amp = data[np.ix_(tr_idx, sm_idx)].T

        X = np.repeat(x[None, :], len(z), axis=0)
        Y = np.repeat(y[None, :], len(z), axis=0)
        Z = np.repeat(z[:, None], len(x), axis=1)

        if X.shape != amp.shape:
            raise ValueError(
                f"Fence mismatch at {line.folder.parent.name}/line {line.number}: "
                f"data={data.shape}, X={X.shape}, amp={amp.shape}, "
                f"gps={len(line.x) if line.x is not None else 0}, "
                f"trace_step={trace_step}, sample_step={sample_step}"
            )

        cache.append((line, X, Y, Z, amp))
        all_amp.append(amp.ravel())
        debug.append(f"{line.folder.parent.name}/line {line.number}: data={data.shape}, fence={amp.shape}")

    if not cache:
        p.add_text("No radargram data loaded", position="upper_left")
        p.reset_camera()
        return

    vmin, vmax = self.robust_clip(np.concatenate(all_amp))

    for line, X, Y, Z, amp in cache:
        grid = pv.StructuredGrid(X, Y, Z)
        grid["amplitude"] = amp.ravel(order="F")

        parent = line.folder.parent.name.lower()
        opacity = 0.95 if parent == "inline" else 0.55

        p.add_mesh(
            grid,
            scalars="amplitude",
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=opacity,
            show_scalar_bar=False,
        )

        if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
            label = ("I" if parent == "inline" else "C") + str(line.number)
            j = X.shape[1] // 2
            p.add_point_labels(
                np.array([[X[0, j], Y[0, j], 0.0]]),
                [label],
                font_size=10,
                point_size=0,
                shape_opacity=0.0,
            )

    if hasattr(self, "progress_log"):
        self.progress_log("Fence renderer used: HARD OVERRIDE")
        for row in debug[:30]:
            self.progress_log("  " + row)
        if len(debug) > 30:
            self.progress_log(f"  ... {len(debug)-30} more lines")

    p.add_axes()
    p.add_text(
        f"PyVista 3D fence: {len(cache)} lines | trace step={trace_step}, sample step={sample_step}",
        position="upper_left",
        font_size=10,
    )
    p.view_isometric()
    p.reset_camera()
    p.render()


GPR3DAnalysisTab.ensure_data = _gpr3d_fixed_ensure_data
GPR3DAnalysisTab.plot_pyvista_fence = _gpr3d_fixed_plot_pyvista_fence

# ---- END HARD OVERRIDE ----




# ---- HARD OVERRIDE: PyVista fence visual style ----

def _gpr3d_fence_visual_settings(self):
    style = self.fence_style.currentText() if hasattr(self, "fence_style") else "Dark high contrast"

    if style == "Dark high contrast":
        return {
            "background": "black",
            "cmap": "seismic",
            "inline_opacity": 1.0,
            "cross_opacity": 0.90,
            "label_color": "white",
            "text_color": "white",
            "show_scalar_bar": True,
        }

    if style == "Dark seismic":
        return {
            "background": "black",
            "cmap": "seismic",
            "inline_opacity": 0.95,
            "cross_opacity": 0.70,
            "label_color": "white",
            "text_color": "white",
            "show_scalar_bar": True,
        }

    if style == "Dark turbo":
        return {
            "background": "black",
            "cmap": "turbo",
            "inline_opacity": 0.95,
            "cross_opacity": 0.75,
            "label_color": "white",
            "text_color": "white",
            "show_scalar_bar": True,
        }

    if style == "Light grayscale":
        return {
            "background": "white",
            "cmap": "gray",
            "inline_opacity": 0.95,
            "cross_opacity": 0.70,
            "label_color": "black",
            "text_color": "black",
            "show_scalar_bar": True,
        }

    if style == "Transparent pale":
        return {
            "background": "white",
            "cmap": "seismic",
            "inline_opacity": 0.65,
            "cross_opacity": 0.35,
            "label_color": "black",
            "text_color": "black",
            "show_scalar_bar": False,
        }

    return {
        "background": "white",
        "cmap": "seismic",
        "inline_opacity": 0.95,
        "cross_opacity": 0.70,
        "label_color": "black",
        "text_color": "black",
        "show_scalar_bar": True,
    }


def _gpr3d_styled_plot_pyvista_fence(self):
    p = self.fence_plotter
    p.clear()

    vis = _gpr3d_fence_visual_settings(self)
    p.set_background(vis["background"])

    lines = self.selected_lines()
    if not lines:
        p.add_text("No selected lines", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    trace_step = max(1, int(self.trace_step.value()))
    sample_step = max(1, int(self.sample_step.value()))
    cmap = vis["cmap"]

    cache = []
    all_amp = []

    for line in lines:
        data = _gpr3d_fixed_ensure_data(self, line)

        if data is None or data.ndim != 2:
            continue

        data = _gpr3d_fixed_orient(self, line, data)
        ntrace, nsamp = data.shape

        if ntrace < 2 or nsamp < 2:
            continue

        x_full, y_full, _ = self.trace_xyz(line, ntrace)
        t_full = self.time_vector(line, nsamp)

        tr_idx = np.arange(0, ntrace, trace_step)
        sm_idx = np.arange(0, nsamp, sample_step)

        if len(tr_idx) < 2 or len(sm_idx) < 2:
            continue

        x = np.asarray(x_full)[tr_idx]
        y = np.asarray(y_full)[tr_idx]
        z = -np.asarray(t_full)[sm_idx]

        amp = data[np.ix_(tr_idx, sm_idx)].T

        X = np.repeat(x[None, :], len(z), axis=0)
        Y = np.repeat(y[None, :], len(z), axis=0)
        Z = np.repeat(z[:, None], len(x), axis=1)

        if X.shape != amp.shape:
            raise ValueError(
                f"Fence mismatch at {line.folder.parent.name}/line {line.number}: "
                f"data={data.shape}, X={X.shape}, amp={amp.shape}"
            )

        cache.append((line, X, Y, Z, amp))
        all_amp.append(amp.ravel())

    if not cache:
        p.add_text("No radargram data loaded", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    vmin, vmax = self.robust_clip(np.concatenate(all_amp))

    first = True
    for line, X, Y, Z, amp in cache:
        grid = pv.StructuredGrid(X, Y, Z)
        grid["amplitude"] = amp.ravel(order="F")

        parent = line.folder.parent.name.lower()
        opacity = vis["inline_opacity"] if parent == "inline" else vis["cross_opacity"]

        p.add_mesh(
            grid,
            scalars="amplitude",
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=opacity,
            show_scalar_bar=bool(vis["show_scalar_bar"] and first),
            scalar_bar_args={"title": "Amplitude"} if first else None,
        )
        first = False

        if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
            label = ("I" if parent == "inline" else "C") + str(line.number)
            j = X.shape[1] // 2
            p.add_point_labels(
                np.array([[X[0, j], Y[0, j], 0.0]]),
                [label],
                font_size=10,
                point_size=0,
                text_color=vis["label_color"],
                shape_opacity=0.0,
            )

    p.add_axes()
    p.add_text(
        f"PyVista 3D fence: {len(cache)} lines | style={self.fence_style.currentText()} | trace step={trace_step}, sample step={sample_step}",
        position="upper_left",
        font_size=10,
        color=vis["text_color"],
    )

    if hasattr(self, "progress_log"):
        self.progress_log(f"Fence visual style: {self.fence_style.currentText()}")
        self.progress_log(f"Rendered {len(cache)} fence panels with cmap={cmap}, background={vis['background']}")

    p.view_isometric()
    p.reset_camera()
    p.render()


GPR3DAnalysisTab.plot_pyvista_fence = _gpr3d_styled_plot_pyvista_fence

# ---- END HARD OVERRIDE: PyVista fence visual style ----




# ---- HARD OVERRIDE: ABCD markers and final tab order helpers ----

def _abcd_get_lines(obj):
    lines = getattr(obj, "lines", None)
    if lines is None and hasattr(obj, "main"):
        lines = getattr(obj.main, "lines", [])
    return lines or []


def _abcd_midpoint(line):
    if line is None or line.x is None or line.y is None or len(line.x) == 0:
        return None

    i = len(line.x) // 2
    z = 0.0
    if getattr(line, "elev", None) is not None and len(line.elev) > i:
        try:
            z = float(line.elev[i])
        except Exception:
            z = 0.0

    return float(line.x[i]), float(line.y[i]), z


def _abcd_references(obj):
    lines = _abcd_get_lines(obj)

    inline = [ln for ln in lines if ln.folder.parent.name.lower() == "inline"]
    cross = [ln for ln in lines if ln.folder.parent.name.lower() == "crossline"]

    def find(items, n):
        for ln in items:
            if ln.number == n:
                return ln
        return None

    refs = {
        "A": _abcd_midpoint(find(cross, 1)),       # A-side: crossline 1
        "B": _abcd_midpoint(find(cross, 201)),     # B-side: crossline 201
        "C high": _abcd_midpoint(find(inline, 1)), # C-side: higher elevation edge
        "D low": _abcd_midpoint(find(inline, 17)), # D-side: lower elevation edge
    }

    return {k: v for k, v in refs.items() if v is not None}


def _abcd_add_mpl_2d(ax, refs):
    for label, pt in refs.items():
        x, y, _ = pt
        ax.scatter([x], [y], s=70, marker="o")
        ax.text(x, y, "  " + label, fontsize=11, weight="bold")


def _abcd_add_mpl_3d(ax, refs):
    for label, pt in refs.items():
        x, y, z = pt
        ax.scatter([x], [y], [z], s=70, marker="o")
        ax.text(x, y, z, "  " + label, fontsize=11, weight="bold")


def _abcd_add_pyvista(plotter, refs, z_mode="surface"):
    if not refs:
        return

    pts = []
    labels = []

    for label, pt in refs.items():
        x, y, z = pt
        if z_mode == "time_zero":
            z = 0.0
        pts.append([x, y, z])
        labels.append(label)

    pts = np.asarray(pts, dtype=float)

    plotter.add_points(pts, point_size=14, render_points_as_spheres=True)
    plotter.add_point_labels(
        pts,
        labels,
        font_size=15,
        point_size=0,
        text_color="yellow",
        shape_color="black",
        shape_opacity=0.55,
    )


# Wrap GPS plan view.
_abcd_old_plot_gps_plan = MainWindow.plot_gps_plan

def _abcd_plot_gps_plan(self, canvas):
    _abcd_old_plot_gps_plan(self, canvas)
    if canvas.fig.axes:
        ax = canvas.fig.axes[0]
        _abcd_add_mpl_2d(ax, _abcd_references(self))
        ax.set_title("GPS plan view: inline + crossline layout with A/B/C/D markers")
        canvas.draw_idle()

MainWindow.plot_gps_plan = _abcd_plot_gps_plan


# Wrap 3D elevation map.
_abcd_old_plot_3d_map = MainWindow.plot_3d_map

def _abcd_plot_3d_map(self, canvas):
    _abcd_old_plot_3d_map(self, canvas)
    if canvas.fig.axes:
        ax = canvas.fig.axes[0]
        _abcd_add_mpl_3d(ax, _abcd_references(self))
        ax.set_title("3D GPS elevation map with A/B/C/D markers; C high, D low")
        canvas.draw_idle()

MainWindow.plot_3d_map = _abcd_plot_3d_map


# Wrap PyVista fence.
_abcd_old_fence = GPR3DAnalysisTab.plot_pyvista_fence

def _abcd_plot_pyvista_fence(self):
    _abcd_old_fence(self)
    refs = _abcd_references(self.main)
    _abcd_add_pyvista(self.fence_plotter, refs, z_mode="time_zero")
    self.fence_plotter.add_text(
        "ABCD markers: A=crossline 1 side, B=crossline 201 side, C=high edge, D=low edge",
        position="lower_left",
        font_size=9,
        color="yellow",
    )
    self.fence_plotter.render()

GPR3DAnalysisTab.plot_pyvista_fence = _abcd_plot_pyvista_fence


# Wrap PyVista volume/slice/isurface views.
for _name, _plotter_attr in [
    ("show_volume", "volume_plotter"),
    ("show_3d_slices", "slice3d_plotter"),
    ("show_isosurface", "iso_plotter"),
]:
    if hasattr(GPR3DAnalysisTab, _name):
        _old = getattr(GPR3DAnalysisTab, _name)

        def _make_wrapper(old_func, plotter_attr):
            def _wrapped(self):
                old_func(self)
                plotter = getattr(self, plotter_attr)
                _abcd_add_pyvista(plotter, _abcd_references(self.main), z_mode="time_zero")
                plotter.render()
            return _wrapped

        setattr(GPR3DAnalysisTab, _name, _make_wrapper(_old, _plotter_attr))


# Wrap time/depth slice maps.
if hasattr(GPR3DAnalysisTab, "plot_slice_on_canvas"):
    _abcd_old_slice_canvas = GPR3DAnalysisTab.plot_slice_on_canvas

    def _abcd_plot_slice_on_canvas(self, canvas, time_ns, title):
        _abcd_old_slice_canvas(self, canvas, time_ns, title)
        if canvas.fig.axes:
            ax = canvas.fig.axes[0]
            _abcd_add_mpl_2d(ax, _abcd_references(self.main))
            canvas.draw_idle()

    GPR3DAnalysisTab.plot_slice_on_canvas = _abcd_plot_slice_on_canvas


# Wrap amplitude projection map.
if hasattr(GPR3DAnalysisTab, "plot_projection"):
    _abcd_old_projection = GPR3DAnalysisTab.plot_projection

    def _abcd_plot_projection(self):
        _abcd_old_projection(self)
        if self.proj_canvas.fig.axes:
            ax = self.proj_canvas.fig.axes[0]
            _abcd_add_mpl_2d(ax, _abcd_references(self.main))
            self.proj_canvas.draw_idle()

    GPR3DAnalysisTab.plot_projection = _abcd_plot_projection


# Mark radargram line endpoints in Inline/Crossline tabs.
_abcd_old_line_plot_raw = LineTab.plot_raw
_abcd_old_line_plot_processed = LineTab.plot_processed

def _abcd_label_line_axis(tab, ax):
    parent = tab.line.folder.parent.name.lower()

    if parent == "inline":
        left, right = "A", "B"
    elif parent == "crossline":
        left, right = "C high", "D low"
    else:
        left, right = "Start", "End"

    ax.text(
        0.01, 0.96, left,
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
    )
    ax.text(
        0.99, 0.96, right,
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
    )

def _abcd_plot_raw(self):
    _abcd_old_line_plot_raw(self)
    if self.raw_ax is not None:
        _abcd_label_line_axis(self, self.raw_ax)
        self.raw_canvas.draw_idle()

def _abcd_plot_processed(self):
    _abcd_old_line_plot_processed(self)
    if self.proc_ax is not None:
        _abcd_label_line_axis(self, self.proc_ax)
        self.proc_canvas.draw_idle()

LineTab.plot_raw = _abcd_plot_raw
LineTab.plot_processed = _abcd_plot_processed

# ---- END HARD OVERRIDE: ABCD markers ----




# ---- HARD OVERRIDE: fast full-survey PyVista fence ----

def _gpr3d_fast_full_fence(self):
    p = self.fence_plotter
    p.clear()

    vis = _gpr3d_fence_visual_settings(self) if "_gpr3d_fence_visual_settings" in globals() else {
        "background": "black",
        "cmap": "seismic",
        "inline_opacity": 1.0,
        "cross_opacity": 0.80,
        "label_color": "yellow",
        "text_color": "yellow",
        "show_scalar_bar": True,
    }

    p.set_background(vis["background"])

    lines = self.selected_lines()
    if not lines:
        p.add_text("No selected lines", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    # Fast full-survey rule:
    # - all inlines are textured
    # - crosslines 1, 201, and every 10th crossline are textured
    # - all other crosslines are shown as guide lines only
    textured = []
    guides = []

    for line in lines:
        parent = line.folder.parent.name.lower()
        if parent == "inline":
            textured.append(line)
        elif parent == "crossline" and (line.number in {1, 201} or line.number % 10 == 0):
            textured.append(line)
        else:
            guides.append(line)

    trace_step = max(12, int(self.trace_step.value()))
    sample_step = max(10, int(self.sample_step.value()))
    cmap = vis["cmap"]

    # Draw guide lines first: all survey geometry remains visible.
    for line in guides:
        if line.x is None or line.y is None or len(line.x) < 2:
            continue
        z = np.zeros_like(line.x, dtype=float)
        pts = np.column_stack([line.x, line.y, z])
        try:
            poly = pv.PolyData(pts)
            poly.lines = np.hstack([[len(pts)], np.arange(len(pts))])
            p.add_mesh(poly, color="gray", opacity=0.20, line_width=1)
        except Exception:
            pass

    cache = []
    all_amp = []

    self.progress_log(
        f"Fast full-survey fence mode: {len(textured)} textured panels, "
        f"{len(guides)} guide-only lines. "
        f"Effective trace step={trace_step}, sample step={sample_step}"
    )

    for line in textured:
        data = _gpr3d_fixed_ensure_data(self, line)

        if data is None or data.ndim != 2:
            continue

        data = _gpr3d_fixed_orient(self, line, data)
        ntrace, nsamp = data.shape

        if ntrace < 2 or nsamp < 2:
            continue

        x_full, y_full, _ = self.trace_xyz(line, ntrace)
        t_full = self.time_vector(line, nsamp)

        tr_idx = np.arange(0, ntrace, trace_step)
        sm_idx = np.arange(0, nsamp, sample_step)

        if len(tr_idx) < 2 or len(sm_idx) < 2:
            continue

        x = np.asarray(x_full)[tr_idx]
        y = np.asarray(y_full)[tr_idx]
        z = -np.asarray(t_full)[sm_idx]
        amp = data[np.ix_(tr_idx, sm_idx)].T

        X = np.repeat(x[None, :], len(z), axis=0)
        Y = np.repeat(y[None, :], len(z), axis=0)
        Z = np.repeat(z[:, None], len(x), axis=1)

        if X.shape != amp.shape:
            continue

        cache.append((line, X, Y, Z, amp))
        all_amp.append(amp.ravel())

    if not cache:
        p.add_text("No textured radargrams rendered", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    vmin, vmax = self.robust_clip(np.concatenate(all_amp))

    first = True
    for line, X, Y, Z, amp in cache:
        grid = pv.StructuredGrid(X, Y, Z)
        grid["amplitude"] = amp.ravel(order="F")

        parent = line.folder.parent.name.lower()
        opacity = vis["inline_opacity"] if parent == "inline" else vis["cross_opacity"]

        p.add_mesh(
            grid,
            scalars="amplitude",
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=opacity,
            show_scalar_bar=bool(vis["show_scalar_bar"] and first),
            scalar_bar_args={"title": "Amplitude"} if first else None,
        )
        first = False

        if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
            label = ("I" if parent == "inline" else "C") + str(line.number)
            j = X.shape[1] // 2
            p.add_point_labels(
                np.array([[X[0, j], Y[0, j], 0.0]]),
                [label],
                font_size=10,
                point_size=0,
                text_color=vis["label_color"],
                shape_opacity=0.0,
            )

    _abcd_add_pyvista(p, _abcd_references(self.main), z_mode="time_zero")

    p.add_axes()
    p.add_text(
        f"Fast full-survey 3D fence: all {len(lines)} lines visible; "
        f"{len(cache)} textured panels, {len(guides)} guide lines",
        position="upper_left",
        font_size=10,
        color=vis["text_color"],
    )
    p.add_text(
        "ABCD markers: A=crossline 1 side, B=crossline 201 side, C=high edge, D=low edge",
        position="lower_left",
        font_size=9,
        color=vis["text_color"],
    )

    p.view_isometric()
    p.reset_camera()
    p.render()

GPR3DAnalysisTab.plot_pyvista_fence = _gpr3d_fast_full_fence

# ---- END HARD OVERRIDE: fast full-survey PyVista fence ----




# ---- HARD OVERRIDE: safe PyVista volume workflow ----

def _gpr3d_safe_build_volume_only(self):
    lines = self.selected_lines()
    try:
        self.progress_start(
            f"Building PyVista 3D volume from {len(lines)} selected lines...",
            max(1, len(lines))
        )

        self.progress_log("Building interpolated 3D amplitude volume only.")
        self.progress_log("Rendering is not started automatically anymore.")
        self.progress_log(f"Volume grid: nx={int(self.nx.value())}, ny={int(self.ny.value())}, nz={int(self.nz.value())}")
        self.progress_log(f"Time window: {float(self.proj_tmin.value()):.1f}–{float(self.proj_tmax.value()):.1f} ns")
        self.progress_log(f"Trace step={int(self.trace_step.value())}, sample step={int(self.sample_step.value())}")

        self.volume_grid = self.build_volume()

        dims = getattr(self.volume_grid, "dimensions", None)
        self.progress_log(f"Volume built successfully. Dimensions={dims}")
        self.progress_log("Next: select PyVista Volume / 3D Slices / Isosurface, then click Update selected view.")

        self.progress_finish("3D GPR volume built")
        self.main.status.setText(
            "Volume built. Now select PyVista Volume, PyVista 3D Slices, or PyVista Isosurface and click Update selected view."
        )

    except Exception as e:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        tb = traceback.format_exc() if "traceback" in globals() else str(e)
        print(tb)
        QMessageBox.critical(self, "Volume build error", tb)


def _gpr3d_safe_show_volume(self):
    p = self.volume_plotter
    p.clear()

    if self.volume_grid is None:
        p.add_text("Click 'Build / refresh PyVista volume' first", position="upper_left")
        p.reset_camera()
        return

    arr = self.volume_grid["amplitude"]
    vmin, vmax = self.robust_clip(arr)

    # Safer than full volume rendering:
    # show outline + orthogonal slices instead of dense ray-cast volume.
    try:
        slices = self.volume_grid.slice_orthogonal()
        p.add_mesh(
            slices,
            scalars="amplitude",
            cmap="seismic",
            clim=(vmin, vmax),
            opacity=0.95,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Amplitude"},
        )
        p.add_outline(self.volume_grid, color="white")
        p.add_text(
            "Safe PyVista Volume view: orthogonal slices through built 3D volume",
            position="upper_left",
            font_size=10,
            color="white",
        )
    except Exception as e:
        p.add_text(f"Could not render volume slices: {e}", position="upper_left")

    _abcd_add_pyvista(p, _abcd_references(self.main), z_mode="time_zero")

    p.add_axes()
    p.view_isometric()
    p.reset_camera()
    p.render()


GPR3DAnalysisTab.build_and_show_volume = _gpr3d_safe_build_volume_only
GPR3DAnalysisTab.show_volume = _gpr3d_safe_show_volume

# ---- END HARD OVERRIDE: safe PyVista volume workflow ----




# ---- HARD OVERRIDE: all 217 lightweight textured fence panels ----

def _gpr3d_all217_lightweight_fence(self):
    p = self.fence_plotter
    p.clear()

    vis = _gpr3d_fence_visual_settings(self) if "_gpr3d_fence_visual_settings" in globals() else {
        "background": "black",
        "cmap": "seismic",
        "inline_opacity": 1.0,
        "cross_opacity": 0.70,
        "label_color": "yellow",
        "text_color": "yellow",
        "show_scalar_bar": True,
    }

    p.set_background(vis["background"])

    lines = self.selected_lines()
    if not lines:
        p.add_text("No selected lines", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    # Force all selected lines to be plotted as textured radargram panels.
    # Decimation is deliberately aggressive to keep 217 panels renderable.
    trace_step = max(35, int(self.trace_step.value()))
    sample_step = max(30, int(self.sample_step.value()))
    cmap = vis["cmap"]

    cache = []
    all_amp = []

    self.progress_log(
        f"All-217 lightweight fence mode: plotting {len(lines)} textured panels. "
        f"Effective trace step={trace_step}, sample step={sample_step}."
    )

    for line in lines:
        data = _gpr3d_fixed_ensure_data(self, line)

        if data is None or data.ndim != 2:
            continue

        data = _gpr3d_fixed_orient(self, line, data)
        ntrace, nsamp = data.shape

        if ntrace < 2 or nsamp < 2:
            continue

        x_full, y_full, _ = self.trace_xyz(line, ntrace)
        t_full = self.time_vector(line, nsamp)

        tr_idx = np.arange(0, ntrace, trace_step)
        sm_idx = np.arange(0, nsamp, sample_step)

        if len(tr_idx) < 2 or len(sm_idx) < 2:
            continue

        x = np.asarray(x_full)[tr_idx]
        y = np.asarray(y_full)[tr_idx]
        z = -np.asarray(t_full)[sm_idx]
        amp = data[np.ix_(tr_idx, sm_idx)].T

        X = np.repeat(x[None, :], len(z), axis=0)
        Y = np.repeat(y[None, :], len(z), axis=0)
        Z = np.repeat(z[:, None], len(x), axis=1)

        if X.shape != amp.shape:
            continue

        cache.append((line, X, Y, Z, amp))
        all_amp.append(amp.ravel())

    # Close modal progress popup before PyVista render.
    if self._progress_dialog is not None:
        self._progress_dialog.close()
        self._progress_dialog = None
        QApplication.processEvents()

    if not cache:
        p.add_text("No radargram panels rendered", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    try:
        self.main.status.setText(f"Rendering {len(cache)} lightweight PyVista fence panels. Please wait...")
        QApplication.processEvents()
    except Exception:
        pass

    vmin, vmax = self.robust_clip(np.concatenate(all_amp))

    first = True
    for line, X, Y, Z, amp in cache:
        grid = pv.StructuredGrid(X, Y, Z)
        grid["amplitude"] = amp.ravel(order="F")

        parent = line.folder.parent.name.lower()
        opacity = vis["inline_opacity"] if parent == "inline" else vis["cross_opacity"]

        p.add_mesh(
            grid,
            scalars="amplitude",
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=opacity,
            show_scalar_bar=bool(vis["show_scalar_bar"] and first),
            scalar_bar_args={"title": "Amplitude"} if first else None,
        )
        first = False

        # Label only key lines. Labelling all 217 would clutter and slow the view.
        if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
            label = ("I" if parent == "inline" else "C") + str(line.number)
            j = X.shape[1] // 2
            p.add_point_labels(
                np.array([[X[0, j], Y[0, j], 0.0]]),
                [label],
                font_size=10,
                point_size=0,
                text_color=vis["label_color"],
                shape_opacity=0.0,
            )

    _abcd_add_pyvista(p, _abcd_references(self.main), z_mode="time_zero")

    p.add_axes()
    p.add_text(
        f"All-217 lightweight 3D fence: {len(cache)} textured panels | trace step={trace_step}, sample step={sample_step}",
        position="upper_left",
        font_size=10,
        color=vis["text_color"],
    )
    p.add_text(
        "All lines are plotted. This is decimated preview, not full-resolution rendering.",
        position="lower_left",
        font_size=9,
        color=vis["text_color"],
    )

    p.view_isometric()
    p.reset_camera()
    p.render()

    try:
        self.main.status.setText(f"Rendered all {len(cache)} lightweight 3D fence panels")
    except Exception:
        pass


GPR3DAnalysisTab.plot_pyvista_fence = _gpr3d_all217_lightweight_fence

# ---- END HARD OVERRIDE: all 217 lightweight textured fence panels ----




# ---- HARD OVERRIDE: all-217 single-mesh PyVista fence ----

def _gpr3d_all217_single_mesh_fence(self):
    p = self.fence_plotter
    p.clear()

    vis = _gpr3d_fence_visual_settings(self) if "_gpr3d_fence_visual_settings" in globals() else {
        "background": "black",
        "cmap": "seismic",
        "inline_opacity": 1.0,
        "cross_opacity": 0.75,
        "label_color": "yellow",
        "text_color": "yellow",
        "show_scalar_bar": True,
    }

    p.set_background(vis["background"])

    lines = self.selected_lines()
    if not lines:
        p.add_text("No selected lines", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    # Aggressive preview decimation, but all selected lines are included.
    trace_step = max(45, int(self.trace_step.value()))
    sample_step = max(40, int(self.sample_step.value()))

    points = []
    faces = []
    scalars = []
    labels_pts = []
    labels_txt = []

    point_offset = 0
    rendered = 0
    skipped = 0
    all_vals = []

    self.progress_log(
        f"Single-mesh all-line fence: attempting {len(lines)} lines. "
        f"Effective trace step={trace_step}, sample step={sample_step}."
    )

    for line in lines:
        data = _gpr3d_fixed_ensure_data(self, line)

        if data is None or data.ndim != 2:
            skipped += 1
            continue

        data = _gpr3d_fixed_orient(self, line, data)
        ntrace, nsamp = data.shape

        if ntrace < 2 or nsamp < 2:
            skipped += 1
            continue

        x_full, y_full, _ = self.trace_xyz(line, ntrace)
        t_full = self.time_vector(line, nsamp)

        tr_idx = np.arange(0, ntrace, trace_step)
        sm_idx = np.arange(0, nsamp, sample_step)

        if len(tr_idx) < 2 or len(sm_idx) < 2:
            skipped += 1
            continue

        x = np.asarray(x_full)[tr_idx]
        y = np.asarray(y_full)[tr_idx]
        z = -np.asarray(t_full)[sm_idx]
        amp = data[np.ix_(tr_idx, sm_idx)].T  # samples x traces

        nz, nx = amp.shape

        # Add grid vertices.
        for iz in range(nz):
            for ix in range(nx):
                points.append([x[ix], y[ix], z[iz]])
                scalars.append(float(amp[iz, ix]))

        # Add quad faces.
        for iz in range(nz - 1):
            for ix in range(nx - 1):
                a = point_offset + iz * nx + ix
                b = point_offset + iz * nx + ix + 1
                c = point_offset + (iz + 1) * nx + ix + 1
                d = point_offset + (iz + 1) * nx + ix
                faces.extend([4, a, b, c, d])

        point_offset += nz * nx
        rendered += 1
        all_vals.append(amp.ravel())

        parent = line.folder.parent.name.lower()
        if parent == "inline" or line.number in {1, 50, 100, 150, 201}:
            j = len(x) // 2
            labels_pts.append([x[j], y[j], 0.0])
            labels_txt.append(("I" if parent == "inline" else "C") + str(line.number))

    # Close progress popup before heavy render.
    if self._progress_dialog is not None:
        self._progress_dialog.close()
        self._progress_dialog = None
        QApplication.processEvents()

    if rendered == 0 or not points or not faces:
        p.add_text("No radargram panels rendered", position="upper_left", color=vis["text_color"])
        p.reset_camera()
        return

    pts = np.asarray(points, dtype=float)
    fcs = np.asarray(faces, dtype=np.int64)
    vals = np.asarray(scalars, dtype=float)

    mesh = pv.PolyData(pts, fcs)
    mesh["amplitude"] = vals

    vmin, vmax = self.robust_clip(vals)

    p.add_mesh(
        mesh,
        scalars="amplitude",
        cmap=vis["cmap"],
        clim=(vmin, vmax),
        opacity=0.82,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Amplitude"},
    )

    if labels_pts:
        p.add_point_labels(
            np.asarray(labels_pts, dtype=float),
            labels_txt,
            font_size=10,
            point_size=0,
            text_color=vis["label_color"],
            shape_opacity=0.0,
        )

    _abcd_add_pyvista(p, _abcd_references(self.main), z_mode="time_zero")

    p.add_axes()
    p.add_text(
        f"All-line single-mesh fence: {rendered}/{len(lines)} radargrams plotted, skipped={skipped}",
        position="upper_left",
        font_size=10,
        color=vis["text_color"],
    )
    p.add_text(
        f"Single PyVista mesh, decimated preview: trace step={trace_step}, sample step={sample_step}",
        position="lower_left",
        font_size=9,
        color=vis["text_color"],
    )

    p.view_isometric()
    p.reset_camera()
    p.render()

    try:
        self.main.status.setText(f"Rendered {rendered}/{len(lines)} radargrams as one PyVista mesh")
    except Exception:
        pass


GPR3DAnalysisTab.plot_pyvista_fence = _gpr3d_all217_single_mesh_fence

# ---- END HARD OVERRIDE: all-217 single-mesh PyVista fence ----




# ---- STANDARD INTERPRETABLE 3D GPR ANALYSIS TAB ----
# Replaces heavy PyVista all-radargram rendering with standard GPR products:
# survey overview, time slice, depth slice, amplitude projection, selected fence diagram.

class GPR3DStandardAnalysisTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        self.mode = QComboBox()
        self.mode.addItems(["processed", "raw"])
        self.mode.setCurrentText("processed")

        self.line_group = QComboBox()
        self.line_group.addItems(["both", "inline only", "crossline only"])
        self.line_group.setCurrentText("both")

        self.cross_step = QSpinBox()
        self.cross_step.setRange(1, 50)
        self.cross_step.setValue(1)

        self.inline_step = QSpinBox()
        self.inline_step.setRange(1, 17)
        self.inline_step.setValue(1)

        self.max_lines = QSpinBox()
        self.max_lines.setRange(1, 217)
        self.max_lines.setValue(217)

        self.trace_step = QSpinBox()
        self.trace_step.setRange(1, 100)
        self.trace_step.setValue(5)

        self.slice_time = QDoubleSpinBox()
        self.slice_time.setRange(0.0, 1000.0)
        self.slice_time.setValue(80.0)
        self.slice_time.setSuffix(" ns")

        self.depth = QDoubleSpinBox()
        self.depth.setRange(0.0, 20.0)
        self.depth.setValue(2.0)
        self.depth.setSuffix(" m")

        self.velocity = QDoubleSpinBox()
        self.velocity.setRange(0.01, 0.30)
        self.velocity.setValue(0.10)
        self.velocity.setSingleStep(0.005)
        self.velocity.setSuffix(" m/ns")

        self.proj_tmin = QDoubleSpinBox()
        self.proj_tmin.setRange(0.0, 1000.0)
        self.proj_tmin.setValue(35.0)
        self.proj_tmin.setSuffix(" ns")

        self.proj_tmax = QDoubleSpinBox()
        self.proj_tmax.setRange(0.0, 1000.0)
        self.proj_tmax.setValue(180.0)
        self.proj_tmax.setSuffix(" ns")

        self.fence_cross_step = QSpinBox()
        self.fence_cross_step.setRange(1, 50)
        self.fence_cross_step.setValue(20)

        self.update_btn = QPushButton("Update selected standard view")
        self.export_btn = QPushButton("Export current figure PNG")

        controls = QWidget()
        grid = QGridLayout(controls)

        grid.addWidget(QLabel("Data"), 0, 0)
        grid.addWidget(self.mode, 0, 1)
        grid.addWidget(QLabel("Lines for maps"), 0, 2)
        grid.addWidget(self.line_group, 0, 3)
        grid.addWidget(QLabel("Cross step"), 0, 4)
        grid.addWidget(self.cross_step, 0, 5)
        grid.addWidget(QLabel("Inline step"), 0, 6)
        grid.addWidget(self.inline_step, 0, 7)
        grid.addWidget(QLabel("Max lines"), 0, 8)
        grid.addWidget(self.max_lines, 0, 9)

        grid.addWidget(QLabel("Trace step"), 1, 0)
        grid.addWidget(self.trace_step, 1, 1)
        grid.addWidget(QLabel("Time slice"), 1, 2)
        grid.addWidget(self.slice_time, 1, 3)
        grid.addWidget(QLabel("Depth"), 1, 4)
        grid.addWidget(self.depth, 1, 5)
        grid.addWidget(QLabel("Velocity"), 1, 6)
        grid.addWidget(self.velocity, 1, 7)
        grid.addWidget(QLabel("Fence cross step"), 1, 8)
        grid.addWidget(self.fence_cross_step, 1, 9)

        grid.addWidget(QLabel("Projection tmin"), 2, 0)
        grid.addWidget(self.proj_tmin, 2, 1)
        grid.addWidget(QLabel("Projection tmax"), 2, 2)
        grid.addWidget(self.proj_tmax, 2, 3)
        grid.addWidget(self.update_btn, 2, 8)
        grid.addWidget(self.export_btn, 2, 9)

        self.tabs = QTabWidget()

        self.overview_canvas = MplCanvas(width=10, height=7)
        self.time_canvas = MplCanvas(width=10, height=7)
        self.depth_canvas = MplCanvas(width=10, height=7)
        self.proj_canvas = MplCanvas(width=10, height=7)
        self.fence_canvas = MplCanvas(width=10, height=7)

        self.tabs.addTab(self.overview_canvas, "Survey Overview")
        self.tabs.addTab(self.time_canvas, "Time Slice Map")
        self.tabs.addTab(self.depth_canvas, "Depth Slice Map")
        self.tabs.addTab(self.proj_canvas, "Amplitude Projection")
        self.tabs.addTab(self.fence_canvas, "Selected Fence Diagram")

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(self.tabs)

        self.update_btn.clicked.connect(self.update_current_view)
        self.export_btn.clicked.connect(self.export_current_png)

        self.plot_overview()

    def selected_lines_for_maps(self):
        group = self.line_group.currentText()
        cross_step = max(1, int(self.cross_step.value()))
        inline_step = max(1, int(self.inline_step.value()))

        out = []
        for line in self.main.lines:
            parent = line.folder.parent.name.lower()

            if group == "inline only" and parent != "inline":
                continue
            if group == "crossline only" and parent != "crossline":
                continue

            if parent == "inline" and ((line.number - 1) % inline_step != 0):
                continue
            if parent == "crossline" and ((line.number - 1) % cross_step != 0) and line.number not in {1, 201}:
                continue

            out.append(line)

        out = sorted(out, key=lambda ln: (0 if ln.folder.parent.name.lower() == "inline" else 1, ln.number))
        return out[: int(self.max_lines.value())]

    def selected_lines_for_fence(self):
        step = max(1, int(self.fence_cross_step.value()))
        out = []

        for line in self.main.lines:
            parent = line.folder.parent.name.lower()

            if parent == "inline":
                out.append(line)
            elif parent == "crossline" and (line.number in {1, 100, 201} or line.number % step == 0):
                out.append(line)

        return sorted(out, key=lambda ln: (0 if ln.folder.parent.name.lower() == "inline" else 1, ln.number))

    def ensure_data(self, line):
        raw = self.main.load_raw_line(line)

        if self.mode.currentText() == "raw":
            return raw

        if line.processed is None:
            line.processed = process_gpr(
                raw,
                dt_ns=line.dt_ns,
                dewow_ns=float(self.main.dewow_ns.value()),
                remove_background=self.main.bg.isChecked(),
                use_bandpass=self.main.bp.isChecked(),
                low_mhz=float(self.main.low_mhz.value()),
                high_mhz=float(self.main.high_mhz.value()),
                sec_power=float(self.main.sec_power.value()),
                background_window_traces=int(self.main.bg_window.value()),
                use_agc=self.main.agc.isChecked(),
                agc_window_ns=float(self.main.agc_window.value()),
            )

        return line.processed

    def time_vector(self, line, nsamp):
        if line.dt_ns > 0:
            return np.arange(nsamp, dtype=float) * line.dt_ns
        if line.time_window_ns > 0:
            return np.linspace(0.0, line.time_window_ns, nsamp)
        return np.arange(nsamp, dtype=float)

    def trace_xy(self, line, ntraces):
        if line.x is None or line.y is None or len(line.x) < 2:
            return np.arange(ntraces, dtype=float), np.zeros(ntraces, dtype=float)

        gps_i = np.linspace(0.0, 1.0, len(line.x))
        tr_i = np.linspace(0.0, 1.0, ntraces)

        x = np.interp(tr_i, gps_i, line.x)
        y = np.interp(tr_i, gps_i, line.y)

        return x, y

    def robust_clip(self, arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return -1.0, 1.0
        amp = np.nanpercentile(np.abs(finite), float(self.main.clip.value()))
        amp = max(float(amp), 1e-12)
        return -amp, amp

    def abcd_refs(self):
        inline = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "inline"]
        cross = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "crossline"]

        def find(items, n):
            for ln in items:
                if ln.number == n:
                    return ln
            return None

        def mid(line):
            if line is None or line.x is None or line.y is None or len(line.x) == 0:
                return None
            i = len(line.x) // 2
            return float(line.x[i]), float(line.y[i])

        refs = {
            "A": mid(find(cross, 1)),
            "B": mid(find(cross, 201)),
            "C high": mid(find(inline, 1)),
            "D low": mid(find(inline, 17)),
        }

        return {k: v for k, v in refs.items() if v is not None}

    def add_abcd_2d(self, ax):
        for label, pt in self.abcd_refs().items():
            x, y = pt
            ax.scatter([x], [y], s=80, marker="o")
            ax.text(x, y, "  " + label, fontsize=11, weight="bold")

    def plot_overview(self):
        c = self.overview_canvas
        c.fig.clear()
        ax = c.fig.add_subplot(111)

        inline = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "inline"]
        cross = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "crossline"]

        for line in sorted(cross, key=lambda ln: ln.number):
            if line.x is None or line.y is None:
                continue
            ax.plot(line.x, line.y, linewidth=0.5, alpha=0.30)
            if line.number in {1, 50, 100, 150, 201}:
                j = len(line.x) // 2
                ax.text(line.x[j], line.y[j], f"C{line.number}", fontsize=8)

        for line in sorted(inline, key=lambda ln: ln.number):
            if line.x is None or line.y is None:
                continue
            ax.plot(line.x, line.y, linewidth=2.0, alpha=0.95)
            j = len(line.x) // 2
            ax.text(line.x[j], line.y[j], f"I{line.number}", fontsize=9, weight="bold")

        self.add_abcd_2d(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title("Standard 3D GPR overview: full survey geometry with ABCD markers")
        ax.grid(True, alpha=0.3)
        c.remember_home()
        c.draw_idle()

    def collect_slice_points(self, time_ns):
        xs, ys, vals = [], [], []
        lines = self.selected_lines_for_maps()
        step = max(1, int(self.trace_step.value()))

        for k, line in enumerate(lines, 1):
            try:
                self.main.status.setText(f"Sampling {k}/{len(lines)}: {line.name}")
                QApplication.processEvents()

                data = self.ensure_data(line)
                if data is None or data.ndim != 2:
                    continue

                t = self.time_vector(line, data.shape[1])
                if time_ns < np.nanmin(t) or time_ns > np.nanmax(t):
                    continue

                j = int(np.argmin(np.abs(t - time_ns)))
                x, y = self.trace_xy(line, data.shape[0])
                idx = np.arange(0, data.shape[0], step)

                xs.extend(x[idx])
                ys.extend(y[idx])
                vals.extend(data[idx, j])
            except Exception as e:
                print("Skipping", line.name, e)

        return np.asarray(xs), np.asarray(ys), np.asarray(vals)

    def plot_slice(self, canvas, time_ns, title):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        x, y, val = self.collect_slice_points(time_ns)

        if len(val) < 4:
            ax.text(0.5, 0.5, "Not enough points for slice", transform=ax.transAxes, ha="center", va="center")
            canvas.draw_idle()
            return

        vmin, vmax = self.robust_clip(val)

        try:
            from scipy.interpolate import griddata
            nx, ny = 280, 180
            xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
            yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), val, (X, Y), method="linear")

            im = ax.imshow(
                Z,
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                cmap=self.main.cmap.currentText(),
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            canvas.fig.colorbar(im, ax=ax, shrink=0.8, label="Amplitude")
            ax.scatter(x, y, s=1.5, c="k", alpha=0.15)
        except Exception:
            sc = ax.scatter(x, y, c=val, s=6, cmap=self.main.cmap.currentText(), vmin=vmin, vmax=vmax)
            canvas.fig.colorbar(sc, ax=ax, shrink=0.8, label="Amplitude")

        self.add_abcd_2d(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        canvas.remember_home()
        canvas.draw_idle()
        self.main.status.setText(f"Updated {title}")

    def plot_time_slice(self):
        t = float(self.slice_time.value())
        self.plot_slice(self.time_canvas, t, f"Time slice map at {t:.1f} ns")

    def plot_depth_slice(self):
        depth = float(self.depth.value())
        vel = max(float(self.velocity.value()), 1e-9)
        t = 2.0 * depth / vel
        self.plot_slice(self.depth_canvas, t, f"Depth slice map at {depth:.2f} m, v={vel:.3f} m/ns, TWT={t:.1f} ns")

    def plot_projection(self):
        c = self.proj_canvas
        c.fig.clear()
        ax = c.fig.add_subplot(111)

        tmin = float(self.proj_tmin.value())
        tmax = float(self.proj_tmax.value())

        if tmax <= tmin:
            tmax = tmin + 1.0

        xs, ys, vals = [], [], []
        lines = self.selected_lines_for_maps()
        step = max(1, int(self.trace_step.value()))

        for k, line in enumerate(lines, 1):
            try:
                self.main.status.setText(f"Projection {k}/{len(lines)}: {line.name}")
                QApplication.processEvents()

                data = self.ensure_data(line)
                t = self.time_vector(line, data.shape[1])
                good = np.where((t >= tmin) & (t <= tmax))[0]

                if len(good) < 2:
                    continue

                x, y = self.trace_xy(line, data.shape[0])
                idx = np.arange(0, data.shape[0], step)
                amp = np.nanmax(np.abs(data[np.ix_(idx, good)]), axis=1)

                xs.extend(x[idx])
                ys.extend(y[idx])
                vals.extend(amp)
            except Exception as e:
                print("Skipping projection", line.name, e)

        x = np.asarray(xs)
        y = np.asarray(ys)
        val = np.asarray(vals)

        if len(val) < 4:
            ax.text(0.5, 0.5, "Not enough points for projection", transform=ax.transAxes, ha="center", va="center")
            c.draw_idle()
            return

        vmax = np.nanpercentile(val[np.isfinite(val)], float(self.main.clip.value()))

        try:
            from scipy.interpolate import griddata
            nx, ny = 280, 180
            xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
            yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), val, (X, Y), method="linear")

            im = ax.imshow(
                Z,
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=vmax,
                aspect="equal",
            )
            c.fig.colorbar(im, ax=ax, shrink=0.8, label="Max |amplitude|")
            ax.scatter(x, y, s=1.5, c="k", alpha=0.15)
        except Exception:
            sc = ax.scatter(x, y, c=val, s=6, cmap="inferno", vmin=0, vmax=vmax)
            c.fig.colorbar(sc, ax=ax, shrink=0.8, label="Max |amplitude|")

        self.add_abcd_2d(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_title(f"Amplitude projection map, max |amplitude| from {tmin:.1f}–{tmax:.1f} ns")
        ax.grid(True, alpha=0.25)
        c.remember_home()
        c.draw_idle()
        self.main.status.setText("Updated amplitude projection map")

    def plot_selected_fence(self):
        c = self.fence_canvas
        c.fig.clear()
        ax = c.fig.add_subplot(111, projection="3d")

        lines = self.selected_lines_for_fence()

        all_vals = []
        cache = []

        for k, line in enumerate(lines, 1):
            try:
                self.main.status.setText(f"Fence {k}/{len(lines)}: {line.name}")
                QApplication.processEvents()

                data = self.ensure_data(line)
                ntrace, nsamp = data.shape

                x, y = self.trace_xy(line, ntrace)
                t = self.time_vector(line, nsamp)

                tr_idx = np.linspace(0, ntrace - 1, min(80, ntrace)).astype(int)
                sm_idx = np.linspace(0, nsamp - 1, min(160, nsamp)).astype(int)

                X = np.repeat(x[tr_idx][None, :], len(sm_idx), axis=0)
                Y = np.repeat(y[tr_idx][None, :], len(sm_idx), axis=0)
                Z = -np.repeat(t[sm_idx][:, None], len(tr_idx), axis=1)
                A = data[np.ix_(tr_idx, sm_idx)].T

                cache.append((line, X, Y, Z, A))
                all_vals.append(A.ravel())
            except Exception as e:
                print("Skipping fence", line.name, e)

        if not cache:
            ax.text2D(0.5, 0.5, "No fence lines available", transform=ax.transAxes, ha="center")
            c.draw_idle()
            return

        vmin, vmax = self.robust_clip(np.concatenate(all_vals))
        cmap = colormaps[self.main.cmap.currentText()]
        norm = Normalize(vmin=vmin, vmax=vmax)

        for line, X, Y, Z, A in cache:
            fc = cmap(norm(A))
            parent = line.folder.parent.name.lower()
            fc[..., 3] = 0.92 if parent == "inline" else 0.55

            ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

            if parent == "inline" or line.number in {1, 100, 201}:
                j = X.shape[1] // 2
                ax.text(X[0, j], Y[0, j], 0.0, ("I" if parent == "inline" else "C") + str(line.number), fontsize=8)

        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_zlabel("Two-way time [ns], downward")
        ax.set_title(f"Selected fence diagram: all inlines + every {int(self.fence_cross_step.value())}th crossline")
        try:
            ax.set_box_aspect((1, 1, 0.45))
        except Exception:
            pass

        c.remember_home()
        c.draw_idle()
        self.main.status.setText("Updated selected fence diagram")

    def update_current_view(self):
        idx = self.tabs.currentIndex()

        if idx == 0:
            self.plot_overview()
        elif idx == 1:
            self.plot_time_slice()
        elif idx == 2:
            self.plot_depth_slice()
        elif idx == 3:
            self.plot_projection()
        elif idx == 4:
            self.plot_selected_fence()

    def export_current_png(self):
        idx = self.tabs.currentIndex()
        name = self.tabs.tabText(idx).lower().replace(" ", "_")
        out = Path("/home/luqman/gpr_gui/data") / f"standard_3d_gpr_{name}.png"

        canvas = self.tabs.currentWidget()
        if isinstance(canvas, MplCanvas):
            canvas.fig.savefig(out, dpi=300)
            self.main.status.setText(f"Saved {out}")
            QMessageBox.information(self, "Exported", f"Saved:\n{out}")
        else:
            QMessageBox.warning(self, "Export failed", "Current tab is not a Matplotlib figure.")


GPR3DAnalysisTab = GPR3DStandardAnalysisTab

# ---- END STANDARD INTERPRETABLE 3D GPR ANALYSIS TAB ----




# ---- HARD OVERRIDE: progress/log for Selected Fence Diagram ----

def _standard_progress_plot_selected_fence(self):
    c = self.fence_canvas
    c.fig.clear()
    ax = c.fig.add_subplot(111, projection="3d")

    lines = self.selected_lines_for_fence()
    total = max(1, len(lines) + 2)

    progress = QProgressDialog("Preparing selected fence diagram...", "Cancel", 0, total, self)
    progress.setWindowTitle("Selected fence progress")
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setValue(0)
    QApplication.processEvents()

    log = []
    cache = []
    all_vals = []

    try:
        for k, line in enumerate(lines, 1):
            msg = f"Fence {k}/{len(lines)}: loading/processing {line.name}"
            log.append(msg)
            progress.setLabelText(msg)
            progress.setValue(k)
            self.main.status.setText(msg)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.main.status.setText("Selected fence cancelled.")
                return

            try:
                data = self.ensure_data(line)
                ntrace, nsamp = data.shape

                x, y = self.trace_xy(line, ntrace)
                t = self.time_vector(line, nsamp)

                tr_idx = np.linspace(0, ntrace - 1, min(80, ntrace)).astype(int)
                sm_idx = np.linspace(0, nsamp - 1, min(160, nsamp)).astype(int)

                X = np.repeat(x[tr_idx][None, :], len(sm_idx), axis=0)
                Y = np.repeat(y[tr_idx][None, :], len(sm_idx), axis=0)
                Z = -np.repeat(t[sm_idx][:, None], len(tr_idx), axis=1)
                A = data[np.ix_(tr_idx, sm_idx)].T

                cache.append((line, X, Y, Z, A))
                all_vals.append(A.ravel())
                log.append(f"  OK {line.name}: data={data.shape}, fence={A.shape}")

            except Exception as e:
                log.append(f"  SKIP {line.name}: {e}")

        progress.setLabelText("Rendering selected fence diagram...")
        progress.setValue(len(lines) + 1)
        self.main.status.setText("Rendering selected fence diagram...")
        QApplication.processEvents()

        if not cache:
            ax.text2D(0.5, 0.5, "No fence lines available", transform=ax.transAxes, ha="center")
            c.draw_idle()
            self.main.status.setText("Selected fence failed: no valid lines.")
            return

        vmin, vmax = self.robust_clip(np.concatenate(all_vals))
        cmap = colormaps[self.main.cmap.currentText()]
        norm = Normalize(vmin=vmin, vmax=vmax)

        for i, (line, X, Y, Z, A) in enumerate(cache, 1):
            progress.setLabelText(f"Rendering surface {i}/{len(cache)}: {line.name}")
            QApplication.processEvents()

            if progress.wasCanceled():
                self.main.status.setText("Selected fence rendering cancelled.")
                return

            fc = cmap(norm(A))
            parent = line.folder.parent.name.lower()
            fc[..., 3] = 0.92 if parent == "inline" else 0.55

            ax.plot_surface(
                X, Y, Z,
                facecolors=fc,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
            )

            if parent == "inline" or line.number in {1, 100, 201}:
                j = X.shape[1] // 2
                ax.text(
                    X[0, j], Y[0, j], 0.0,
                    ("I" if parent == "inline" else "C") + str(line.number),
                    fontsize=8,
                )

        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_zlabel("Two-way time [ns], downward")
        ax.set_title(f"Selected fence diagram: all inlines + every {int(self.fence_cross_step.value())}th crossline")

        try:
            ax.set_box_aspect((1, 1, 0.45))
        except Exception:
            pass

        progress.setLabelText("Final drawing...")
        progress.setValue(total)
        QApplication.processEvents()

        c.remember_home()
        c.draw_idle()

        log_path = Path("/home/luqman/gpr_gui/data/selected_fence_last_run.log")
        log_path.write_text("\n".join(log) + "\n")

        self.main.status.setText(f"Updated selected fence diagram. Log: {log_path}")

    finally:
        progress.close()
        QApplication.processEvents()


GPR3DStandardAnalysisTab.plot_selected_fence = _standard_progress_plot_selected_fence
GPR3DAnalysisTab = GPR3DStandardAnalysisTab

# ---- END HARD OVERRIDE: progress/log for Selected Fence Diagram ----




# ---- HARD OVERRIDE: PyVista fast selected fence diagram ----

_old_standard_init_for_pyvista_fence = GPR3DStandardAnalysisTab.__init__

def _standard_init_with_pyvista_fence(self, main_window):
    _old_standard_init_for_pyvista_fence(self, main_window)

    # Replace the old Matplotlib selected-fence tab with a PyVista viewer.
    try:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Selected Fence Diagram":
                self.tabs.removeTab(i)
                break

        self.fence_plotter = QtInteractor(self)
        self.tabs.addTab(self.fence_plotter.interactor, "Selected Fence Diagram")
    except Exception as e:
        print("Could not replace selected fence with PyVista:", e)

GPR3DStandardAnalysisTab.__init__ = _standard_init_with_pyvista_fence


def _standard_pyvista_selected_fence(self):
    if not hasattr(self, "fence_plotter"):
        QMessageBox.warning(self, "Fence error", "PyVista fence viewer not initialized.")
        return

    p = self.fence_plotter
    p.clear()
    p.set_background("black")

    lines = self.selected_lines_for_fence()
    total = max(1, len(lines))

    progress = QProgressDialog("Preparing fast PyVista selected fence...", "Cancel", 0, total, self)
    progress.setWindowTitle("Selected fence progress")
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setValue(0)
    QApplication.processEvents()

    # Performance caps. This is what makes rotation usable.
    max_trace_cols = 80
    max_time_rows = 160

    points = []
    faces = []
    scalars = []
    labels_pts = []
    labels_txt = []

    offset = 0
    rendered = 0
    skipped = 0
    log = []

    try:
        for k, line in enumerate(lines, 1):
            msg = f"Preparing fence {k}/{len(lines)}: {line.name}"
            progress.setLabelText(msg)
            progress.setValue(k)
            self.main.status.setText(msg)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.main.status.setText("Selected fence cancelled.")
                return

            try:
                data = self.ensure_data(line)

                if data is None or data.ndim != 2:
                    skipped += 1
                    continue

                ntrace, nsamp = data.shape
                if ntrace < 2 or nsamp < 2:
                    skipped += 1
                    continue

                x, y = self.trace_xy(line, ntrace)
                t = self.time_vector(line, nsamp)

                tr_idx = np.unique(np.linspace(0, ntrace - 1, min(max_trace_cols, ntrace)).astype(int))
                sm_idx = np.unique(np.linspace(0, nsamp - 1, min(max_time_rows, nsamp)).astype(int))

                if len(tr_idx) < 2 or len(sm_idx) < 2:
                    skipped += 1
                    continue

                xs = np.asarray(x)[tr_idx]
                ys = np.asarray(y)[tr_idx]
                zs = -np.asarray(t)[sm_idx]
                amp = data[np.ix_(tr_idx, sm_idx)].T

                nz, nx = amp.shape

                for iz in range(nz):
                    for ix in range(nx):
                        points.append([xs[ix], ys[ix], zs[iz]])
                        scalars.append(float(amp[iz, ix]))

                for iz in range(nz - 1):
                    for ix in range(nx - 1):
                        a = offset + iz * nx + ix
                        b = offset + iz * nx + ix + 1
                        c = offset + (iz + 1) * nx + ix + 1
                        d = offset + (iz + 1) * nx + ix
                        faces.extend([4, a, b, c, d])

                offset += nz * nx
                rendered += 1
                log.append(f"OK {line.name}: data={data.shape}, preview={amp.shape}")

                parent = line.folder.parent.name.lower()
                if parent == "inline" or line.number in {1, 100, 201}:
                    j = len(xs) // 2
                    labels_pts.append([xs[j], ys[j], 0.0])
                    labels_txt.append(("I" if parent == "inline" else "C") + str(line.number))

            except Exception as e:
                skipped += 1
                log.append(f"SKIP {line.name}: {e}")

        progress.close()
        QApplication.processEvents()

        if rendered == 0:
            p.add_text("No fence panels rendered", position="upper_left", color="white")
            p.reset_camera()
            return

        pts = np.asarray(points, dtype=float)
        fcs = np.asarray(faces, dtype=np.int64)
        vals = np.asarray(scalars, dtype=float)

        mesh = pv.PolyData(pts, fcs)
        mesh["amplitude"] = vals

        vmin, vmax = self.robust_clip(vals)

        p.add_mesh(
            mesh,
            scalars="amplitude",
            cmap=self.main.cmap.currentText(),
            clim=(vmin, vmax),
            opacity=0.92,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Amplitude"},
        )

        if labels_pts:
            p.add_point_labels(
                np.asarray(labels_pts, dtype=float),
                labels_txt,
                font_size=10,
                point_size=0,
                text_color="yellow",
                shape_color="black",
                shape_opacity=0.50,
            )

        try:
            _abcd_add_pyvista(p, _abcd_references(self.main), z_mode="time_zero")
        except Exception:
            pass

        p.add_axes()
        p.add_text(
            f"Fast PyVista selected fence: {rendered}/{len(lines)} panels, skipped={skipped}",
            position="upper_left",
            color="white",
            font_size=10,
        )
        p.add_text(
            f"Single mesh preview: max {max_trace_cols} traces × {max_time_rows} samples per panel",
            position="lower_left",
            color="white",
            font_size=9,
        )

        p.view_isometric()
        p.reset_camera()
        p.render()

        log_path = Path("/home/luqman/gpr_gui/data/selected_fence_last_run.log")
        log_path.write_text("\n".join(log) + "\n")

        self.main.status.setText(f"Updated fast PyVista selected fence. Log: {log_path}")

    finally:
        try:
            progress.close()
        except Exception:
            pass
        QApplication.processEvents()


GPR3DStandardAnalysisTab.plot_selected_fence = _standard_pyvista_selected_fence
GPR3DAnalysisTab = GPR3DStandardAnalysisTab

# ---- END HARD OVERRIDE: PyVista fast selected fence diagram ----




# ---- HARD OVERRIDE: quantitative suspicious-zone detector ----

_old_standard_init_for_suspicious = GPR3DStandardAnalysisTab.__init__

def _standard_init_with_suspicious(self, main_window):
    _old_standard_init_for_suspicious(self, main_window)

    if not hasattr(self, "suspicious_canvas"):
        self.suspicious_canvas = MplCanvas(width=10, height=7)
        self.tabs.addTab(self.suspicious_canvas, "Suspicious Zones")

GPR3DStandardAnalysisTab.__init__ = _standard_init_with_suspicious


def _standard_plot_suspicious_zones(self):
    c = self.suspicious_canvas
    c.fig.clear()
    ax = c.fig.add_subplot(111)

    lines = self.selected_lines_for_maps()
    if not lines:
        ax.text(0.5, 0.5, "No selected lines", transform=ax.transAxes, ha="center", va="center")
        c.draw_idle()
        return

    tmin = float(self.proj_tmin.value())
    tmax = float(self.proj_tmax.value())
    if tmax <= tmin:
        tmax = tmin + 1.0

    vel = max(float(self.velocity.value()), 1e-9)
    trace_step = max(1, int(self.trace_step.value()))

    # Grid/bin resolution for quantitative stacking.
    dx = 0.25     # metres
    dy = 0.25     # metres
    dt = 5.0      # ns

    xs_all = []
    ys_all = []
    for line in lines:
        if line.x is not None and line.y is not None and len(line.x) > 1:
            xs_all.extend(line.x)
            ys_all.extend(line.y)

    if not xs_all:
        ax.text(0.5, 0.5, "No GPS coordinates available", transform=ax.transAxes, ha="center", va="center")
        c.draw_idle()
        return

    xmin, xmax = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    ymin, ymax = float(np.nanmin(ys_all)), float(np.nanmax(ys_all))

    bins = {}

    progress = QProgressDialog("Running suspicious-zone detector...", "Cancel", 0, len(lines), self)
    progress.setWindowTitle("Suspicious-zone analysis")
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setValue(0)
    QApplication.processEvents()

    log = []
    skipped = 0

    try:
        for k, line in enumerate(lines, 1):
            msg = f"Scanning {k}/{len(lines)}: {line.name}"
            progress.setLabelText(msg)
            progress.setValue(k)
            self.main.status.setText(msg)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.main.status.setText("Suspicious-zone analysis cancelled.")
                return

            try:
                data = self.ensure_data(line)
                if data is None or data.ndim != 2:
                    skipped += 1
                    continue

                ntrace, nsamp = data.shape
                t = self.time_vector(line, nsamp)
                good_t = np.where((t >= tmin) & (t <= tmax))[0]

                if len(good_t) < 2:
                    skipped += 1
                    continue

                # Limit time samples so the detector is fast but still volumetric.
                if len(good_t) > 70:
                    good_t = good_t[np.linspace(0, len(good_t) - 1, 70).astype(int)]

                x, y = self.trace_xy(line, ntrace)
                tr_idx = np.arange(0, ntrace, trace_step)

                if len(tr_idx) < 2:
                    skipped += 1
                    continue

                win_abs = np.abs(data[:, good_t])
                med = float(np.nanmedian(win_abs))
                mad = float(np.nanmedian(np.abs(win_abs - med)))
                scale = max(1.4826 * mad, 1e-12)

                vals = np.abs(data[np.ix_(tr_idx, good_t)])
                z = np.maximum((vals - med) / scale, 0.0)

                parent = line.folder.parent.name.lower()
                is_inline = 1 if parent == "inline" else 0
                is_cross = 1 if parent == "crossline" else 0

                xx = np.asarray(x)[tr_idx]
                yy = np.asarray(y)[tr_idx]

                for ii, tr in enumerate(tr_idx):
                    ix = int(np.floor((xx[ii] - xmin) / dx))
                    iy = int(np.floor((yy[ii] - ymin) / dy))

                    for jj, ti in enumerate(good_t):
                        zz = float(z[ii, jj])

                        # Ignore weak background. This prevents noise-only stacking.
                        if zz < 2.5:
                            continue

                        it = int(np.floor((float(t[ti]) - tmin) / dt))
                        key = (ix, iy, it)

                        rec = bins.get(key)
                        if rec is None:
                            # max_z, sum_z, count, inline_count, cross_count
                            bins[key] = [zz, zz, 1, is_inline, is_cross]
                        else:
                            rec[0] = max(rec[0], zz)
                            rec[1] += zz
                            rec[2] += 1
                            rec[3] += is_inline
                            rec[4] += is_cross

                log.append(f"OK {line.name}: data={data.shape}, traces used={len(tr_idx)}, times used={len(good_t)}")

            except Exception as e:
                skipped += 1
                log.append(f"SKIP {line.name}: {e}")

    finally:
        progress.close()
        QApplication.processEvents()

    candidates = []

    for (ix, iy, it), rec in bins.items():
        max_z, sum_z, count, inline_count, cross_count = rec
        mean_z = sum_z / max(count, 1)

        x = xmin + (ix + 0.5) * dx
        y = ymin + (iy + 0.5) * dy
        time_ns = tmin + (it + 0.5) * dt
        depth_m = 0.5 * vel * time_ns

        both_bonus = 1.75 if (inline_count > 0 and cross_count > 0) else 1.0
        support_bonus = 1.0 + 0.20 * np.log1p(count)

        score = float((0.70 * max_z + 0.30 * mean_z) * support_bonus * both_bonus)

        candidates.append({
            "score": score,
            "x": x,
            "y": y,
            "time_ns": time_ns,
            "depth_m": depth_m,
            "support": int(count),
            "inline_support": int(inline_count),
            "crossline_support": int(cross_count),
            "max_robust_z": float(max_z),
            "mean_robust_z": float(mean_z),
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)

    # Non-maximum suppression so the top list is not the same anomaly repeated.
    selected = []
    for cand in candidates:
        keep = True
        for old in selected:
            dxy = ((cand["x"] - old["x"]) ** 2 + (cand["y"] - old["y"]) ** 2) ** 0.5
            dt_ns = abs(cand["time_ns"] - old["time_ns"])
            if dxy < 0.75 and dt_ns < 12.0:
                keep = False
                break

        if keep:
            selected.append(cand)

        if len(selected) >= 15:
            break

    # Plot survey geometry.
    inline = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "inline"]
    cross = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "crossline"]

    for line in sorted(cross, key=lambda ln: ln.number):
        if line.x is not None and line.y is not None:
            ax.plot(line.x, line.y, linewidth=0.45, alpha=0.22)

    for line in sorted(inline, key=lambda ln: ln.number):
        if line.x is not None and line.y is not None:
            ax.plot(line.x, line.y, linewidth=1.5, alpha=0.75)

    self.add_abcd_2d(ax)

    if selected:
        sx = np.array([d["x"] for d in selected])
        sy = np.array([d["y"] for d in selected])
        ss = np.array([d["score"] for d in selected])

        sc = ax.scatter(
            sx, sy,
            c=ss,
            s=90,
            cmap="inferno",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        c.fig.colorbar(sc, ax=ax, shrink=0.82, label="Suspicion score")

        for i, d in enumerate(selected, 1):
            ax.text(d["x"], d["y"], f"  #{i}", fontsize=10, weight="bold", color="white", zorder=6)

        table = ["Top suspicious zones:"]
        for i, d in enumerate(selected[:8], 1):
            table.append(
                f"#{i}: score={d['score']:.1f}, t={d['time_ns']:.1f} ns, "
                f"z≈{d['depth_m']:.2f} m, support={d['support']}, "
                f"I/C={d['inline_support']}/{d['crossline_support']}"
            )

        ax.text(
            1.01, 0.99,
            "\n".join(table),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )
    else:
        ax.text(0.5, 0.5, "No suspicious zones above threshold", transform=ax.transAxes, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.set_title(
        f"Quantitative suspicious-zone detector: {tmin:.1f}–{tmax:.1f} ns, "
        f"bin={dx:.2f}×{dy:.2f} m, dt={dt:.1f} ns"
    )
    ax.grid(True, alpha=0.25)

    out_csv = Path("/home/luqman/gpr_gui/data/suspicious_zones_last_run.csv")
    with out_csv.open("w") as f:
        f.write("rank,score,easting,northing,time_ns,approx_depth_m,support,inline_support,crossline_support,max_robust_z,mean_robust_z\n")
        for i, d in enumerate(selected, 1):
            f.write(
                f"{i},{d['score']:.6f},{d['x']:.6f},{d['y']:.6f},"
                f"{d['time_ns']:.6f},{d['depth_m']:.6f},{d['support']},"
                f"{d['inline_support']},{d['crossline_support']},"
                f"{d['max_robust_z']:.6f},{d['mean_robust_z']:.6f}\n"
            )

    out_log = Path("/home/luqman/gpr_gui/data/suspicious_zones_last_run.log")
    out_log.write_text("\n".join(log) + f"\nSkipped lines: {skipped}\nBins used: {len(bins)}\nCandidates ranked: {len(candidates)}\n")

    c.remember_home()
    c.draw_idle()

    self.main.status.setText(f"Suspicious-zone analysis done. CSV: {out_csv}")


_old_standard_update_for_suspicious = GPR3DStandardAnalysisTab.update_current_view

def _standard_update_with_suspicious(self):
    title = self.tabs.tabText(self.tabs.currentIndex())

    if title == "Suspicious Zones":
        self.plot_suspicious_zones()
    else:
        _old_standard_update_for_suspicious(self)

GPR3DStandardAnalysisTab.plot_suspicious_zones = _standard_plot_suspicious_zones
GPR3DStandardAnalysisTab.update_current_view = _standard_update_with_suspicious
GPR3DAnalysisTab = GPR3DStandardAnalysisTab

# ---- END HARD OVERRIDE: quantitative suspicious-zone detector ----




# ---- HARD OVERRIDE: improved suspicious-zone detector ----

def _improved_plot_suspicious_zones(self):
    c = self.suspicious_canvas
    c.fig.clear()
    ax = c.fig.add_subplot(111)

    lines = self.selected_lines_for_maps()
    if not lines:
        ax.text(0.5, 0.5, "No selected lines", transform=ax.transAxes, ha="center", va="center")
        c.draw_idle()
        return

    global_tmin = float(self.proj_tmin.value())
    global_tmax = float(self.proj_tmax.value())
    if global_tmax <= global_tmin:
        global_tmax = global_tmin + 1.0

    vel = max(float(self.velocity.value()), 1e-9)
    trace_step = max(1, int(self.trace_step.value()))

    # Standard separated time windows.
    windows = [
        ("early", max(global_tmin, 35.0), min(global_tmax, 70.0), 1.15),
        ("middle", max(global_tmin, 70.0), min(global_tmax, 120.0), 1.00),
        ("late", max(global_tmin, 120.0), min(global_tmax, 180.0), 0.72),
    ]
    windows = [(name, a, b, w) for name, a, b, w in windows if b > a]

    if not windows:
        ax.text(0.5, 0.5, "No valid time windows", transform=ax.transAxes, ha="center", va="center")
        c.draw_idle()
        return

    dx = 0.25
    dy = 0.25
    dt = 5.0

    xs_all, ys_all = [], []
    for line in lines:
        if line.x is not None and line.y is not None and len(line.x) > 1:
            xs_all.extend(line.x)
            ys_all.extend(line.y)

    if not xs_all:
        ax.text(0.5, 0.5, "No GPS coordinates available", transform=ax.transAxes, ha="center", va="center")
        c.draw_idle()
        return

    xmin, xmax = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    ymin, ymax = float(np.nanmin(ys_all)), float(np.nanmax(ys_all))
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    # bins[(window_name, ix, iy, it)] = [max_z, sum_z, count, inline_count, cross_count]
    bins = {}

    progress = QProgressDialog("Running improved suspicious-zone detector...", "Cancel", 0, len(lines), self)
    progress.setWindowTitle("Suspicious-zone detector")
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setValue(0)
    QApplication.processEvents()

    log = []
    skipped = 0

    try:
        for k, line in enumerate(lines, 1):
            msg = f"Scanning {k}/{len(lines)}: {line.name}"
            progress.setLabelText(msg)
            progress.setValue(k)
            self.main.status.setText(msg)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.main.status.setText("Suspicious-zone analysis cancelled.")
                return

            try:
                data = self.ensure_data(line)
                if data is None or data.ndim != 2:
                    skipped += 1
                    continue

                ntrace, nsamp = data.shape
                t = self.time_vector(line, nsamp)
                x, y = self.trace_xy(line, ntrace)

                tr_idx = np.arange(0, ntrace, trace_step)
                if len(tr_idx) < 2:
                    skipped += 1
                    continue

                parent = line.folder.parent.name.lower()
                is_inline = 1 if parent == "inline" else 0
                is_cross = 1 if parent == "crossline" else 0

                xx = np.asarray(x)[tr_idx]
                yy = np.asarray(y)[tr_idx]

                for win_name, tmin, tmax, win_weight in windows:
                    good_t = np.where((t >= tmin) & (t <= tmax))[0]
                    if len(good_t) < 2:
                        continue

                    # Cap time samples for speed while preserving window coverage.
                    if len(good_t) > 60:
                        good_t = good_t[np.linspace(0, len(good_t) - 1, 60).astype(int)]

                    win_abs = np.abs(data[:, good_t])
                    med = float(np.nanmedian(win_abs))
                    mad = float(np.nanmedian(np.abs(win_abs - med)))
                    scale = max(1.4826 * mad, 1e-12)

                    vals = np.abs(data[np.ix_(tr_idx, good_t)])
                    zrob = np.maximum((vals - med) / scale, 0.0)

                    for ii, tr in enumerate(tr_idx):
                        ix = int(np.floor((xx[ii] - xmin) / dx))
                        iy = int(np.floor((yy[ii] - ymin) / dy))

                        for jj, ti in enumerate(good_t):
                            zz = float(zrob[ii, jj])

                            # Higher threshold than before to reduce background speckle.
                            if zz < 3.0:
                                continue

                            it = int(np.floor((float(t[ti]) - tmin) / dt))
                            key = (win_name, ix, iy, it)

                            rec = bins.get(key)
                            if rec is None:
                                bins[key] = [zz, zz, 1, is_inline, is_cross, win_weight, tmin]
                            else:
                                rec[0] = max(rec[0], zz)
                                rec[1] += zz
                                rec[2] += 1
                                rec[3] += is_inline
                                rec[4] += is_cross

                log.append(f"OK {line.name}: data={data.shape}, traces={len(tr_idx)}")

            except Exception as e:
                skipped += 1
                log.append(f"SKIP {line.name}: {e}")

    finally:
        progress.close()
        QApplication.processEvents()

    candidates_by_window = {name: [] for name, _, _, _ in windows}
    all_candidates = []

    for (win_name, ix, iy, it), rec in bins.items():
        max_z, sum_z, count, inline_count, cross_count, win_weight, tmin = rec
        mean_z = sum_z / max(count, 1)

        x = xmin + (ix + 0.5) * dx
        y = ymin + (iy + 0.5) * dy
        time_ns = tmin + (it + 0.5) * dt
        depth_m = 0.5 * vel * time_ns

        # Edge penalty: strong reflections at exact survey edges are often geometry/edge effects.
        nx = (x - xmin) / xspan
        ny = (y - ymin) / yspan
        edge_dist = min(nx, 1.0 - nx, ny, 1.0 - ny)
        edge_factor = np.clip(edge_dist / 0.08, 0.45, 1.0)

        both_factor = 1.85 if (inline_count > 0 and cross_count > 0) else 0.82
        support_factor = 1.0 + 0.18 * np.log1p(count)

        # Penalize huge support without cross-direction confirmation less aggressively.
        if count > 20 and not (inline_count > 0 and cross_count > 0):
            support_factor *= 0.75

        score = float((0.68 * max_z + 0.32 * mean_z) * support_factor * both_factor * edge_factor * win_weight)

        cand = {
            "window": win_name,
            "score": score,
            "x": x,
            "y": y,
            "time_ns": time_ns,
            "depth_m": depth_m,
            "support": int(count),
            "inline_support": int(inline_count),
            "crossline_support": int(cross_count),
            "max_robust_z": float(max_z),
            "mean_robust_z": float(mean_z),
            "edge_factor": float(edge_factor),
        }

        candidates_by_window[win_name].append(cand)
        all_candidates.append(cand)

    def nms(candidates, limit=6):
        candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)
        selected = []

        for cand in candidates:
            keep = True
            for old in selected:
                dxy = ((cand["x"] - old["x"]) ** 2 + (cand["y"] - old["y"]) ** 2) ** 0.5
                dt_ns = abs(cand["time_ns"] - old["time_ns"])

                # Merge nearby duplicate detections.
                if dxy < 1.0 and dt_ns < 15.0:
                    keep = False
                    break

            if keep:
                selected.append(cand)

            if len(selected) >= limit:
                break

        return selected

    selected_by_window = {name: nms(candidates_by_window.get(name, []), limit=5) for name, _, _, _ in windows}
    selected_all = nms(all_candidates, limit=12)

    # Plot survey geometry.
    inline = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "inline"]
    cross = [ln for ln in self.main.lines if ln.folder.parent.name.lower() == "crossline"]

    for line in sorted(cross, key=lambda ln: ln.number):
        if line.x is not None and line.y is not None:
            ax.plot(line.x, line.y, linewidth=0.45, alpha=0.20)

    for line in sorted(inline, key=lambda ln: ln.number):
        if line.x is not None and line.y is not None:
            ax.plot(line.x, line.y, linewidth=1.5, alpha=0.70)

    self.add_abcd_2d(ax)

    if selected_all:
        sx = np.array([d["x"] for d in selected_all])
        sy = np.array([d["y"] for d in selected_all])
        ss = np.array([d["score"] for d in selected_all])

        sc = ax.scatter(
            sx, sy,
            c=ss,
            s=95,
            cmap="inferno",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        c.fig.colorbar(sc, ax=ax, shrink=0.82, label="Suspicion score")

        for i, d in enumerate(selected_all, 1):
            ax.text(d["x"], d["y"], f"  #{i}", fontsize=10, weight="bold", color="white", zorder=6)

        table = ["Top suspicious zones, penalized:"]
        for i, d in enumerate(selected_all[:8], 1):
            table.append(
                f"#{i} {d['window']:<6} score={d['score']:.1f}, "
                f"t={d['time_ns']:.1f} ns, z≈{d['depth_m']:.2f} m, "
                f"I/C={d['inline_support']}/{d['crossline_support']}"
            )

        table.append("")
        table.append("Window leaders:")
        for name, _, _, _ in windows:
            top = selected_by_window.get(name, [])
            if top:
                d = top[0]
                table.append(
                    f"{name:<6}: score={d['score']:.1f}, "
                    f"t={d['time_ns']:.1f} ns, I/C={d['inline_support']}/{d['crossline_support']}"
                )
            else:
                table.append(f"{name:<6}: none")

        ax.text(
            1.01, 0.99,
            "\n".join(table),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none"),
        )
    else:
        ax.text(0.5, 0.5, "No suspicious zones above threshold", transform=ax.transAxes, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.set_title(
        f"Improved suspicious-zone detector: early/middle/late ranking, "
        f"edge + late-time penalties"
    )
    ax.grid(True, alpha=0.25)

    out_csv = Path("/home/luqman/gpr_gui/data/suspicious_zones_last_run.csv")
    with out_csv.open("w") as f:
        f.write("rank,window,score,easting,northing,time_ns,approx_depth_m,support,inline_support,crossline_support,max_robust_z,mean_robust_z,edge_factor\n")
        for i, d in enumerate(selected_all, 1):
            f.write(
                f"{i},{d['window']},{d['score']:.6f},{d['x']:.6f},{d['y']:.6f},"
                f"{d['time_ns']:.6f},{d['depth_m']:.6f},{d['support']},"
                f"{d['inline_support']},{d['crossline_support']},"
                f"{d['max_robust_z']:.6f},{d['mean_robust_z']:.6f},{d['edge_factor']:.6f}\n"
            )

    out_log = Path("/home/luqman/gpr_gui/data/suspicious_zones_last_run.log")
    out_log.write_text(
        "\n".join(log)
        + f"\nSkipped lines: {skipped}\n"
        + f"Bins used: {len(bins)}\n"
        + f"Candidates ranked: {len(all_candidates)}\n"
        + "Windows: " + ", ".join([f"{n}:{a}-{b}ns" for n, a, b, _ in windows]) + "\n"
    )

    c.remember_home()
    c.draw_idle()
    self.main.status.setText(f"Improved suspicious-zone analysis done. CSV: {out_csv}")


GPR3DStandardAnalysisTab.plot_suspicious_zones = _improved_plot_suspicious_zones
GPR3DAnalysisTab = GPR3DStandardAnalysisTab

# ---- END HARD OVERRIDE: improved suspicious-zone detector ----


if __name__ == "__main__":
    main()

