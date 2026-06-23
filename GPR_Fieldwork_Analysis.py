#!/usr/bin/env python3
from pathlib import Path
from dataclasses import dataclass
import sys, os, re, csv, math, datetime
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QScrollArea, QMessageBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

try:
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import uniform_filter1d
    from scipy.interpolate import griddata
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import app as gpr_app

BASE = Path(__file__).resolve().parent
PROJECTS = [
    ("Schleitheim (MALA)", BASE / "data" / "MALA"),
    ("Bulach (PulseEKKO)", BASE / "data" / "PulseEkko"),
]


class Canvas(FigureCanvas):
    def __init__(self, w=8, h=5):
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


@dataclass
class PELine:
    idx: int
    name: str
    hd: Path
    dt1: Path
    ntraces: int
    nsamples: int
    time_window_ns: float
    distance_m: float
    x: np.ndarray
    y: np.ndarray
    dist: np.ndarray
    raw: np.ndarray | None = None
    proc: np.ndarray | None = None

    @property
    def time_ns(self):
        return np.linspace(0.0, float(self.time_window_ns), int(self.nsamples))


def _find_number(txt, patterns, default=None, cast=float):
    for pat in patterns:
        m = re.search(pat, txt, re.I)
        if m:
            try:
                return cast(m.group(1))
            except Exception:
                pass
    return default


def parse_hd(hd: Path):
    txt = hd.read_text(errors="ignore")

    ntr = _find_number(txt, [
        r"NUMBER\s+OF\s+TRACES\s*=\s*(\d+)",
        r"NTRACES\s*=\s*(\d+)",
        r"TRACES\s*=\s*(\d+)",
    ], None, int)

    ns = _find_number(txt, [
        r"NUMBER\s+OF\s+PTS/TRC\s*=\s*(\d+)",
        r"NUMBER\s+OF\s+POINTS\s+PER\s+TRACE\s*=\s*(\d+)",
        r"SAMPLES\s+PER\s+TRACE\s*=\s*(\d+)",
        r"PTS/TRC\s*=\s*(\d+)",
    ], None, int)

    tw = _find_number(txt, [
        r"TOTAL\s+TIME\s+WINDOW\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"TIME\s+WINDOW\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"RANGE\s*=\s*([-+]?\d+(?:\.\d+)?)",
    ], 100.0, float)

    s0 = _find_number(txt, [r"STARTING\s+POSITION\s*=\s*([-+]?\d+(?:\.\d+)?)"], 0.0, float)
    s1 = _find_number(txt, [r"FINAL\s+POSITION\s*=\s*([-+]?\d+(?:\.\d+)?)"], None, float)
    dist = abs(float(s1) - float(s0)) if s1 is not None else 0.0

    if ntr is None and dist > 0:
        ntr = int(round(dist / 0.02)) + 1

    return ntr, ns, float(tw), float(dist)


def infer_dt1_shape(dt1: Path, ntr, ns):
    size = dt1.stat().st_size
    if ntr and ns:
        return int(ntr), int(ns)

    if ns and not ntr:
        for bps in (4, 2):
            rec = 128 + int(ns) * bps
            if rec > 0 and size % rec == 0:
                return int(size // rec), int(ns)

    if ntr and not ns:
        for bps in (4, 2):
            if size % int(ntr) == 0:
                rec = size // int(ntr)
                if rec > 128 and (rec - 128) % bps == 0:
                    return int(ntr), int((rec - 128) // bps)

    raise RuntimeError(f"Cannot infer DT1 shape for {dt1.name}; check .HD metadata.")


def read_dt1(dt1: Path, ntr, ns):
    ntr, ns = infer_dt1_shape(dt1, ntr, ns)
    size = dt1.stat().st_size

    formats = [
        ("header_float32", 128 + ns * 4, np.dtype("<f4")),
        ("header_int16",   128 + ns * 2, np.dtype("<i2")),
        ("plain_float32",  ns * 4,       np.dtype("<f4")),
        ("plain_int16",    ns * 2,       np.dtype("<i2")),
    ]

    chosen = None
    for name, rec, dtype in formats:
        if rec * ntr == size:
            chosen = (name, rec, dtype)
            break

    if chosen is None:
        for name, rec, dtype in formats:
            if rec > 0 and size % rec == 0:
                ntr = int(size // rec)
                chosen = (name, rec, dtype)
                break

    if chosen is None:
        raise RuntimeError(f"Unsupported DT1 size/layout: {dt1.name}, {size} bytes")

    name, rec, dtype = chosen
    data = np.empty((ntr, ns), dtype=np.float64)

    with dt1.open("rb") as f:
        if name.startswith("header"):
            for i in range(ntr):
                f.seek(128, 1)
                a = np.fromfile(f, dtype=dtype, count=ns)
                if a.size != ns:
                    data = data[:i]
                    break
                data[i, :] = a.astype(np.float64)
        else:
            a = np.fromfile(f, dtype=dtype, count=ntr * ns).astype(np.float64)
            data = a.reshape(ntr, ns)

    data -= np.nanmedian(data, axis=1, keepdims=True)
    return data


def process_gpr(raw, time_ns, dewow_window_ns=25.0, bg_window=151, do_bg=True, do_bp=True,
                low_mhz=50.0, high_mhz=250.0, sec_power=0.90, do_agc=True, agc_window_ns=80.0):
    out = raw.astype(float).copy()
    out -= np.nanmedian(out, axis=1, keepdims=True)

    dt_ns = float(np.nanmedian(np.diff(time_ns))) if len(time_ns) > 2 else 1.0

    if SCIPY_OK:
        dewow_samp = max(3, int(round(dewow_window_ns / max(dt_ns, 1e-9))))
        dewow_samp = min(dewow_samp, max(3, out.shape[1] // 2))
        out -= uniform_filter1d(out, size=dewow_samp, axis=1, mode="nearest")
    else:
        win = max(3, min(80, out.shape[1] // 4))
        ker = np.ones(win) / win
        out -= np.apply_along_axis(lambda x: np.convolve(x, ker, mode="same"), 1, out)

    if do_bg:
        if bg_window > 3 and bg_window < out.shape[0]:
            if SCIPY_OK:
                bg = uniform_filter1d(out, size=int(bg_window), axis=0, mode="nearest")
                out -= bg
            else:
                out -= np.nanmedian(out, axis=0, keepdims=True)
        else:
            out -= np.nanmedian(out, axis=0, keepdims=True)

    if do_bp and SCIPY_OK and len(time_ns) > 4:
        dt = dt_ns * 1e-9
        fs = 1.0 / dt if dt > 0 else 0
        nyq = 0.5 * fs
        lo = max(low_mhz * 1e6, 1.0)
        hi = min(high_mhz * 1e6, nyq * 0.98)
        if fs > 0 and lo < hi:
            sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
            out = sosfiltfilt(sos, out, axis=1)

    # Professor-style capped t-power gain.
    # Capped at 30 ns: close to the reference processing default and safer
    # than full-window SEC for interpretation.
    if sec_power > 0:
        t = np.asarray(time_ns, float)
        tmax_gain = 30.0
        tt = np.minimum(np.maximum(t, 0.0), tmax_gain)
        gain = (1.0 + tt / max(tmax_gain, 1e-9)) ** float(sec_power)
        gain /= max(float(gain[0]), 1e-12)
        out *= gain[None, :]

    if do_agc:
        if SCIPY_OK:
            agc_samp = max(5, int(round(float(agc_window_ns) / max(dt_ns, 1e-9))))
            agc_samp = min(agc_samp, max(5, out.shape[1] // 2))
            env = uniform_filter1d(np.abs(out), size=agc_samp, axis=1, mode="nearest")
            scale = np.nanmedian(env) + 1e-9
            out = out / (env + 0.1 * scale + 1e-9)

    return out


def clip_limits(a, pct=99.70):
    v = np.nanpercentile(np.abs(a), pct)
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    return -v, v


class RadargramPair(QWidget):
    def __init__(self, line: PELine, owner):
        super().__init__()
        self.line = line
        self.owner = owner
        lay = QVBoxLayout(self)

        self.raw_canvas = Canvas(12, 3.7)
        self.proc_canvas = Canvas(12, 3.7)

        self.raw_scroll = QScrollArea()
        self.raw_scroll.setWidgetResizable(True)
        self.raw_scroll.setWidget(self.raw_canvas)

        self.proc_scroll = QScrollArea()
        self.proc_scroll.setWidgetResizable(True)
        self.proc_scroll.setWidget(self.proc_canvas)

        lay.addWidget(self.raw_scroll)
        lay.addWidget(self.proc_scroll)
        self.placeholder()

    def placeholder(self):
        self.raw_canvas.ax.clear()
        self.raw_canvas.ax.text(0.5, 0.5, "Click 'Load raw' or 'Process current line'", ha="center", va="center")
        self.raw_canvas.ax.set_axis_off()
        self.raw_canvas.draw()

        self.proc_canvas.ax.clear()
        self.proc_canvas.ax.text(0.5, 0.5, "Processed radargram will appear here", ha="center", va="center")
        self.proc_canvas.ax.set_axis_off()
        self.proc_canvas.draw()

    def plot(self):
        line = self.line
        if line.raw is None:
            return

        t = self.owner.corrected_time_ns(line)
        tmin = self.owner.display_min.value()
        tmax = self.owner.display_max.value()
        mask = (t >= tmin) & (t <= tmax)
        if not np.any(mask):
            mask = np.ones_like(t, dtype=bool)

        cmap = self.owner.cmap.currentText()
        pct = self.owner.display_clip.value()

        for canvas, arr, title in [
            (self.raw_canvas, line.raw, f"Raw radargram — {line.name}"),
            (self.proc_canvas, line.proc if line.proc is not None else line.raw, f"Processed radargram — {line.name}")
        ]:
            data = arr[:, mask]
            tt = t[mask]
            ax = canvas.ax
            ax.clear()
            vmin, vmax = clip_limits(data, pct)
            ax.imshow(
                data.T,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[line.dist[0], line.dist[-1], tt[-1], tt[0]],
            )
            ax.set_title(title)
            ax.set_xlabel("Distance along line [m]")
            ax.set_ylabel("Two-way time [ns]")
            canvas.draw()


class PulseEkko3DAnalysis(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

        root = QVBoxLayout(self)

        grid = QGridLayout()
        self.data_choice = QComboBox(); self.data_choice.addItems(["processed", "raw"])
        self.lines_choice = QComboBox(); self.lines_choice.addItems(["inline"])
        self.trace_step = QSpinBox(); self.trace_step.setRange(1, 1000); self.trace_step.setValue(5)
        self.time_slice = QDoubleSpinBox(); self.time_slice.setRange(0, 10000); self.time_slice.setValue(5.0); self.time_slice.setSuffix(" ns")
        self.depth = QDoubleSpinBox(); self.depth.setRange(0, 1000); self.depth.setValue(0.30); self.depth.setSuffix(" m")
        self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.01, 0.30); self.velocity.setSingleStep(0.01); self.velocity.setValue(0.10); self.velocity.setSuffix(" m/ns")
        self.proj_tmin = QDoubleSpinBox(); self.proj_tmin.setRange(0, 10000); self.proj_tmin.setValue(0.0); self.proj_tmin.setSuffix(" ns")
        self.proj_tmax = QDoubleSpinBox(); self.proj_tmax.setRange(0, 10000); self.proj_tmax.setValue(50.0); self.proj_tmax.setSuffix(" ns")
        self.line_step = QSpinBox(); self.line_step.setRange(1, 1000); self.line_step.setValue(1)
        self.max_lines = QSpinBox(); self.max_lines.setRange(1, 10000); self.max_lines.setValue(161)
        self.fence_step = QSpinBox(); self.fence_step.setRange(1, 1000); self.fence_step.setValue(10)

        btn_update = QPushButton("Update selected standard view")
        btn_export = QPushButton("Export current figure PNG")
        btn_update.clicked.connect(self.update_selected)
        btn_export.clicked.connect(self.export_png)

        items = [
            ("Data", self.data_choice), ("Lines for maps", self.lines_choice), ("Trace step", self.trace_step),
            ("Time slice", self.time_slice), ("Depth", self.depth), ("Velocity", self.velocity),
            ("Projection tmin", self.proj_tmin), ("Projection tmax", self.proj_tmax),
            ("Inline step", self.line_step), ("Max lines", self.max_lines), ("Fence line step", self.fence_step)
        ]

        positions = [(0,0),(0,2),(1,0),(1,2),(1,4),(1,6),(2,0),(2,2),(0,4),(0,6),(2,4)]
        for (label, widget), (r, c) in zip(items, positions):
            grid.addWidget(QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)

        grid.addWidget(btn_update, 2, 6)
        grid.addWidget(btn_export, 2, 7)
        root.addLayout(grid)

        self.tabs = QTabWidget()
        self.canvases = {}
        for name in ["Survey Overview", "Time Slice Map", "Depth Slice Map", "Amplitude Projection", "Selected Fence Diagram"]:
            c = Canvas(12, 6.5)
            self.canvases[name] = c
            self.tabs.addTab(c, name)
        root.addWidget(self.tabs)

        self.tabs.currentChanged.connect(lambda *_: self.update_selected())

    def selected_canvas(self):
        name = self.tabs.tabText(self.tabs.currentIndex())
        canvas = self.canvases.get(name)
        if canvas is None and name == "Time Lapse Map":
            canvas = self.canvases.get("Depth Slice Map") or self.canvases.get("Time Slice Map") or self.canvases.get("Amplitude Projection")
            if canvas is not None:
                self.canvases[name] = canvas
        if canvas is None:
            raise KeyError(name)
        return name, canvas

    def get_array(self, line):
        if self.data_choice.currentText() == "raw":
            if line.raw is None:
                line.raw = read_dt1(line.dt1, line.ntraces, line.nsamples)
            return line.raw

        if line.proc is None:
            self.owner.ensure_processed(line)
        return line.proc

    def selected_lines(self):
        step = max(1, self.line_step.value())
        lines = self.owner.lines[::step]
        return lines[:self.max_lines.value()]

    def collect_values(self, mode):
        xs, ys, vals = [], [], []
        step = max(1, self.trace_step.value())

        for line in self.selected_lines():
            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line)

            if mode == "time":
                idx = int(np.argmin(np.abs(t - self.time_slice.value())))
                v = arr[::step, idx]
            elif mode == "depth":
                twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
                idx = int(np.argmin(np.abs(t - twt)))
                v = arr[::step, idx]
            else:
                lo, hi = sorted([self.proj_tmin.value(), self.proj_tmax.value()])
                m = (t >= lo) & (t <= hi)
                if not np.any(m):
                    continue
                v = np.nanmax(np.abs(arr[::step][:, m]), axis=1)

            xs.extend(line.x[::step])
            ys.extend(line.y[::step])
            vals.extend(v)

        return np.asarray(xs), np.asarray(ys), np.asarray(vals)

    def plot_overview(self, canvas):
        ax = canvas.ax
        ax.clear()

        for line in self.owner.lines:
            ax.plot(line.x, line.y, linewidth=0.65)
            if line.idx % 10 == 0 or line.idx in (0, self.owner.lines[-1].idx):
                ax.text(line.x[0], line.y[0], line.name, fontsize=7)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Standard 3D GPR overview: Bulach PulseEKKO GPS geometry")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.grid(True, alpha=0.3)
        canvas.draw()

    def plot_map(self, canvas, mode, title):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax

        x, y, v = self.collect_values(mode)
        if len(v) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw()
            return

        if SCIPY_OK and len(v) > 100:
            gx = np.linspace(np.nanmin(x), np.nanmax(x), 280)
            gy = np.linspace(np.nanmin(y), np.nanmax(y), 280)
            X, Y = np.meshgrid(gx, gy)
            Z = griddata((x, y), v, (X, Y), method="linear")
            im = ax.imshow(Z, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin="lower", aspect="equal", cmap="viridis")
        else:
            im = ax.scatter(x, y, c=v, s=4, cmap="viridis")
            ax.set_aspect("equal", adjustable="box")

        ax.set_title(title)
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.grid(True, alpha=0.2)
        canvas.fig.colorbar(im, ax=ax, label="Amplitude")
        canvas.draw()

    def plot_fence(self, canvas):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111, projection="3d")
        canvas.ax = ax

        lines = self.owner.lines[::max(1, self.fence_step.value())]
        lines = lines[:self.max_lines.value()]

        for line in lines:
            arr = self.get_array(line)
            t = line.time_ns

            ridx = np.linspace(0, len(line.x) - 1, min(130, len(line.x))).astype(int)
            tidx = np.linspace(0, len(t) - 1, min(150, len(t))).astype(int)

            X = np.tile(line.x[ridx], (len(tidx), 1))
            Y = np.tile(line.y[ridx], (len(tidx), 1))
            Z = -np.tile(t[tidx][:, None], (1, len(ridx)))
            A = arr[np.ix_(ridx, tidx)].T

            vmax = np.nanpercentile(np.abs(A), 99.0)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0

            colors = cm.seismic(np.clip((A / vmax + 1.0) / 2.0, 0, 1))
            ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=False, shade=False)

        ax.set_title("Selected fence diagram — Bulach PulseEKKO")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_zlabel("-Two-way time [ns]")
        canvas.draw()

    def update_selected(self):
        name, canvas = self.selected_canvas()

        if name == "Survey Overview":
            self.plot_overview(canvas)
        elif name == "Time Slice Map":
            self.plot_map(canvas, "time", f"Time slice map at {self.time_slice.value():.1f} ns")
        elif name == "Depth Slice Map":
            twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
            self.plot_map(canvas, "depth", f"Depth slice map at {self.depth.value():.2f} m ≈ {twt:.1f} ns")
        elif name == "Amplitude Projection":
            self.plot_map(canvas, "projection", f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns")
        elif name == "Selected Fence Diagram":
            self.plot_fence(canvas)

    def export_png(self):
        name, canvas = self.selected_canvas()
        out = self.owner.root / f"bulach_pulseekko_{name.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        canvas.fig.savefig(out, dpi=250)
        self.owner.status.setText(f"Exported: {out}")


class PulseEkkoProjectTab(QWidget):
    def __init__(self, root: Path):
        super().__init__()
        self.root = Path(root)
        self.lines = []
        self.line_widgets = {}
        self.build_ui()
        self.reload_project()

    def build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.root_label = QLabel()
        choose = QPushButton("Choose project folder")
        reload_btn = QPushButton("Reload")
        choose.clicked.connect(self.choose_project)
        reload_btn.clicked.connect(self.reload_project)
        top.addWidget(self.root_label)
        top.addStretch(1)
        top.addWidget(choose)
        top.addWidget(reload_btn)
        root.addLayout(top)

        self.tabs = QTabWidget()

        self.gps_canvas = Canvas(12, 6.5)
        self.inline_widget = QWidget()
        self.analysis_widget = PulseEkko3DAnalysis(self)

        self.build_inline_widget()

        self.tabs.addTab(self.gps_canvas, "GPS plan view")
        self.tabs.addTab(self.inline_widget, "Inline")
        self.tabs.addTab(self.analysis_widget, "3D GPR Analysis")

        root.addWidget(self.tabs)

        self.status = QLabel()
        root.addWidget(self.status)

    def build_inline_widget(self):
        lay = QVBoxLayout(self.inline_widget)

        grid = QGridLayout()

        self.btn_load = QPushButton("Load raw")
        self.btn_process = QPushButton("Process current line")
        self.btn_load.clicked.connect(self.load_current_raw)
        self.btn_process.clicked.connect(self.process_current_line)

        self.dewow_window = QDoubleSpinBox(); self.dewow_window.setRange(1, 1000); self.dewow_window.setValue(5.0); self.dewow_window.setSuffix(" ns")
        self.low_cut = QDoubleSpinBox(); self.low_cut.setRange(1, 5000); self.low_cut.setValue(50.0); self.low_cut.setSuffix(" MHz")
        self.high_cut = QDoubleSpinBox(); self.high_cut.setRange(1, 5000); self.high_cut.setValue(250.0); self.high_cut.setSuffix(" MHz")

        self.sec_gain = QCheckBox("T-power gain"); self.sec_gain.setChecked(False)
        self.sec_power = QDoubleSpinBox(); self.sec_power.setRange(0, 10); self.sec_power.setSingleStep(0.05); self.sec_power.setValue(2.25)
        self.display_clip = QDoubleSpinBox(); self.display_clip.setRange(80, 100); self.display_clip.setDecimals(2); self.display_clip.setValue(99.50); self.display_clip.setSuffix(" %")
        self.bg_remove = QCheckBox("Local median background removal"); self.bg_remove.setChecked(False)
        self.bandpass = QCheckBox("Bandpass"); self.bandpass.setChecked(False)
        self.convention = QCheckBox("Display convention: line start → line end"); self.convention.setChecked(False); self.convention.setEnabled(False)
        self.t0_auto = QCheckBox("Auto time-zero")
        self.t0_auto.setChecked(False)
        self.t0_search_min = QDoubleSpinBox(); self.t0_search_min.setRange(0, 1000); self.t0_search_min.setValue(5.0); self.t0_search_min.setSuffix(" ns")
        self.t0_search_max = QDoubleSpinBox(); self.t0_search_max.setRange(0, 1000); self.t0_search_max.setValue(15.0); self.t0_search_max.setSuffix(" ns")
        self.t0_target = QDoubleSpinBox(); self.t0_target.setRange(-100, 100); self.t0_target.setValue(0.0); self.t0_target.setSuffix(" ns")

        self.scale = QComboBox(); self.scale.addItems(["symlog", "linear"])
        self.cmap = QComboBox(); self.cmap.addItems(["seismic", "gray", "RdYlBu_r", "viridis"])
        self.bg_window = QSpinBox(); self.bg_window.setRange(3, 10001); self.bg_window.setSingleStep(2); self.bg_window.setValue(151)
        self.agc_gain = QCheckBox("AGC gain"); self.agc_gain.setChecked(False)
        self.agc_window = QDoubleSpinBox(); self.agc_window.setRange(1, 1000); self.agc_window.setValue(80.0); self.agc_window.setSuffix(" ns")

        self.display_min = QDoubleSpinBox(); self.display_min.setRange(0, 10000); self.display_min.setValue(0.0); self.display_min.setSuffix(" ns")
        self.display_max = QDoubleSpinBox(); self.display_max.setRange(1, 10000); self.display_max.setValue(50.0); self.display_max.setSuffix(" ns")
        self.vertical_exag = QDoubleSpinBox(); self.vertical_exag.setRange(0.1, 20); self.vertical_exag.setValue(1.50); self.vertical_exag.setSuffix("×")

        grid.addWidget(self.btn_load, 0, 0)
        grid.addWidget(self.btn_process, 0, 1)
        grid.addWidget(QLabel("Dewow window"), 0, 2)
        grid.addWidget(self.dewow_window, 0, 3)
        grid.addWidget(QLabel("Low cut"), 0, 4)
        grid.addWidget(self.low_cut, 0, 5)
        grid.addWidget(QLabel("High cut"), 0, 6)
        grid.addWidget(self.high_cut, 0, 7)

        grid.addWidget(self.sec_gain, 1, 0)
        grid.addWidget(self.sec_power, 1, 1)
        grid.addWidget(QLabel("Display clip"), 1, 2)
        grid.addWidget(self.display_clip, 1, 3)
        grid.addWidget(self.bg_remove, 1, 4)
        grid.addWidget(self.bandpass, 1, 5)
        grid.addWidget(self.convention, 1, 6)

        grid.addWidget(QLabel("Scale"), 2, 0)
        grid.addWidget(self.scale, 2, 1)
        grid.addWidget(QLabel("Colour map"), 2, 2)
        grid.addWidget(self.cmap, 2, 3)
        grid.addWidget(QLabel("BG window"), 2, 4)
        grid.addWidget(self.bg_window, 2, 5)
        grid.addWidget(self.agc_gain, 2, 6)
        grid.addWidget(QLabel("AGC window"), 2, 7)
        grid.addWidget(self.agc_window, 2, 8)

        grid.addWidget(QLabel("Display min time"), 3, 0)
        grid.addWidget(self.display_min, 3, 1)
        grid.addWidget(QLabel("Display max time"), 3, 2)
        grid.addWidget(self.display_max, 3, 3)
        grid.addWidget(QLabel("Vertical exaggeration"), 3, 4)
        grid.addWidget(self.vertical_exag, 3, 5)

        grid.addWidget(self.t0_auto, 4, 0)
        grid.addWidget(QLabel("T0 search min"), 4, 1)
        grid.addWidget(self.t0_search_min, 4, 2)
        grid.addWidget(QLabel("T0 search max"), 4, 3)
        grid.addWidget(self.t0_search_max, 4, 4)
        grid.addWidget(QLabel("T0 target"), 4, 5)
        grid.addWidget(self.t0_target, 4, 6)

        lay.addLayout(grid)

        self.inline_tabs = QTabWidget()
        lay.addWidget(self.inline_tabs)

    def choose_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose PulseEKKO project folder", str(self.root))
        if d:
            self.root = Path(d)
            self.reload_project()

    def reload_project(self):
        self.root_label.setText(f"<b>Data root:</b> {self.root}")
        self.lines = self.load_lines()
        self.rebuild_inline_tabs()
        self.plot_gps()
        self.status.setText(f"Loaded {len(self.lines)} PulseEKKO lines from {self.root}. scipy bandpass available={SCIPY_OK}")

    def load_lines(self):
        gpsdir = self.root / "GPS"
        files = []

        for hd in list(self.root.glob("*.HD")) + list(self.root.glob("*.hd")):
            m = re.search(r"LINE(\d+)", hd.stem, re.I)
            if not m:
                continue

            idx = int(m.group(1))
            dt1 = None
            for cand in [self.root / (hd.stem + ".DT1"), self.root / (hd.stem + ".dt1")]:
                if cand.exists():
                    dt1 = cand
                    break

            if dt1 is not None:
                files.append((idx, hd, dt1))

        lines = []

        for idx, hd, dt1 in sorted(files, key=lambda x: x[0]):
            ntr, ns, tw, dist_m = parse_hd(hd)
            ntr, ns = infer_dt1_shape(dt1, ntr, ns)

            gps = gpsdir / f"LINE{idx:03d}_GPS.csv"

            if gps.exists():
                xs, ys, ds = [], [], []
                with gps.open() as f:
                    for r in csv.DictReader(f):
                        xs.append(float(r["easting_m"]))
                        ys.append(float(r["northing_m"]))
                        ds.append(float(r["distance_m"]))

                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
                ds = np.asarray(ds, float)

                if len(xs) != ntr and len(xs) > 1:
                    old = np.linspace(0, 1, len(xs))
                    new = np.linspace(0, 1, ntr)
                    xs = np.interp(new, old, xs)
                    ys = np.interp(new, old, ys)
                    ds = np.interp(new, old, ds)
            else:
                ds = np.linspace(0, dist_m if dist_m > 0 else ntr - 1, ntr)
                xs = ds.copy()
                ys = np.ones(ntr) * idx * 0.25

            lines.append(PELine(idx, f"LINE{idx:03d}", hd, dt1, ntr, ns, tw, dist_m, xs, ys, ds))

        return lines

    def rebuild_inline_tabs(self):
        self.inline_tabs.clear()
        self.line_widgets = {}

        for line in self.lines:
            pair = RadargramPair(line, self)
            self.line_widgets[line.idx] = pair
            self.inline_tabs.addTab(pair, line.name)

        try:
            self.tabs.setTabText(1, f"Inline ({len(self.lines)})")
        except Exception:
            pass

    def current_line(self):
        if not self.lines:
            return None
        i = self.inline_tabs.currentIndex()
        if i < 0 or i >= len(self.lines):
            return None
        return self.lines[i]

    def ensure_raw(self, line):
        if line.raw is None:
            line.raw = read_dt1(line.dt1, line.ntraces, line.nsamples)
        return line.raw

    def ensure_processed(self, line):
        self.ensure_raw(line)
        gain_on = getattr(self, "sec_gain", None) is not None and self.sec_gain.isChecked()
        gain_power = float(self.sec_power.value()) if gain_on else 0.0
        parts = ["DC removal", "dewow"]
        if self.bg_remove.isChecked():
            parts.append("background removal")
        if self.bandpass.isChecked():
            parts.append("bandpass")
        if gain_on and gain_power > 0:
            parts.append(f"T-power gain ({gain_power:.2f})")
        if self.agc_gain.isChecked():
            parts.append("AGC gain")
        line.proc_label = "Processed radargram — " + line.name + " | " + " + ".join(parts)

        line.proc = process_gpr(
            line.raw,
            line.time_ns,
            dewow_window_ns=self.dewow_window.value(),
            bg_window=self.bg_window.value(),
            do_bg=self.bg_remove.isChecked(),
            do_bp=self.bandpass.isChecked(),
            low_mhz=self.low_cut.value(),
            high_mhz=self.high_cut.value(),
            sec_power=gain_power,
            do_agc=self.agc_gain.isChecked(),
            agc_window_ns=self.agc_window.value(),
        )
        return line.proc


    def estimate_t0_ns(self, line):
        """Estimate first strong arrival / time-zero from raw PulseEKKO trace stack."""
        if not getattr(self, "t0_auto", None) or not self.t0_auto.isChecked():
            return 0.0

        lo, hi = sorted([self.t0_search_min.value(), self.t0_search_max.value()])
        cache_key = f"_t0_cache_{lo:.3f}_{hi:.3f}"
        if hasattr(line, cache_key):
            return float(getattr(line, cache_key))

        raw = self.ensure_raw(line)
        t = line.time_ns
        mask = (t >= lo) & (t <= hi)
        if not np.any(mask):
            return 0.0

        amp = np.nanmedian(np.abs(raw), axis=0)
        amp = amp - np.nanpercentile(amp, 10)

        try:
            dt = float(np.nanmedian(np.diff(t)))
            smooth_n = max(3, int(round(0.8 / max(dt, 1e-9))))
            if SCIPY_OK:
                amp = uniform_filter1d(amp, size=smooth_n, mode="nearest")
        except Exception:
            pass

        sub = amp[mask]
        tt = t[mask]
        base = np.nanpercentile(sub, 40)
        peak = np.nanmax(sub)
        if not np.isfinite(peak) or peak <= base:
            t0 = float(tt[int(np.nanargmax(sub))])
        else:
            thr = base + 0.35 * (peak - base)
            cand = np.where(sub >= thr)[0]
            t0 = float(tt[int(cand[0])]) if len(cand) else float(tt[int(np.nanargmax(sub))])

        setattr(line, cache_key, t0)
        return t0

    def corrected_time_ns(self, line):
        t0 = self.estimate_t0_ns(line) if hasattr(self, "estimate_t0_ns") else 0.0
        target = self.t0_target.value() if hasattr(self, "t0_target") else 0.0
        return line.time_ns - t0 + target


    def load_current_raw(self):
        line = self.current_line()
        if line is None:
            return

        try:
            self.ensure_raw(line)
            self.line_widgets[line.idx].plot()
            self.status.setText(f"Loaded raw {line.name}: {line.raw.shape}; estimated T0={self.estimate_t0_ns(line):.2f} ns")
        except Exception as e:
            QMessageBox.critical(self, "Load raw failed", str(e))

    def process_current_line(self):
        line = self.current_line()
        if line is None:
            return

        try:
            self.ensure_processed(line)
            self.line_widgets[line.idx].plot()
            self.status.setText(f"Processed {line.name}: {line.proc.shape}; estimated T0={self.estimate_t0_ns(line):.2f} ns")
        except Exception as e:
            QMessageBox.critical(self, "Processing failed", str(e))

    def plot_gps(self):
        ax = self.gps_canvas.ax
        ax.clear()

        for line in self.lines:
            ax.plot(line.x, line.y, linewidth=0.65)
            if line.idx % 10 == 0 or line.idx in (0, self.lines[-1].idx):
                ax.text(line.x[0], line.y[0], line.name, fontsize=7)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("GPS plan view: Bulach PulseEKKO line layout")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.grid(True, alpha=0.3)
        self.gps_canvas.draw()



# ---- hard override: Load raw must not fill processed panel ----
def _pulseekko_radargram_pair_plot_fixed(self):
    line = self.line
    if line.raw is None:
        return

    t = self.owner.corrected_time_ns(line)
    tmin = self.owner.display_min.value()
    tmax = self.owner.display_max.value()
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    cmap = self.owner.cmap.currentText()
    pct = self.owner.display_clip.value()
    tt = t[mask]

    # Raw panel always plots raw once loaded.
    ax = self.raw_canvas.ax
    ax.clear()
    data = line.raw[:, mask]
    vmin, vmax = clip_limits(data, pct)
    ax.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[line.dist[0], line.dist[-1], tt[-1], tt[0]],
    )
    ax.set_title(f"Raw radargram — {line.name}")
    ax.set_xlabel("Distance along line [m]")
    ax.set_ylabel("Two-way time [ns]")
    self.raw_canvas.draw()

    # Processed panel stays empty until Process current line is clicked.
    ax = self.proc_canvas.ax
    ax.clear()
    if line.proc is None:
        ax.text(0.5, 0.5, "Click 'Process current line' to generate processed radargram", ha="center", va="center")
        ax.set_axis_off()
        self.proc_canvas.draw()
        return

    data = line.proc[:, mask]
    vmin, vmax = clip_limits(data, pct)
    ax.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[line.dist[0], line.dist[-1], tt[-1], tt[0]],
    )
    ax.set_title(getattr(line, "proc_label", f"Processed radargram — {line.name}"))
    ax.set_xlabel("Distance along line [m]")
    ax.set_ylabel("Two-way time [ns]")
    self.proc_canvas.draw()

RadargramPair.plot = _pulseekko_radargram_pair_plot_fixed
# ---- end hard override ----



# ---- hard override: PulseEKKO 3D progress dialog ----
def _pe_format_eta(elapsed, done, total):
    if done <= 0:
        return "estimating..."
    remaining = max(total - done, 0)
    eta = elapsed / max(done, 1) * remaining
    return f"{eta:0.1f}s"

def _pe_make_progress(owner, title, total):
    from PyQt6.QtWidgets import QProgressDialog, QApplication
    from PyQt6.QtCore import Qt
    import time

    dlg = QProgressDialog("Starting...", "Cancel", 0, max(int(total), 1), owner)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.resize(560, 150)
    dlg.show()
    QApplication.processEvents()
    return dlg, time.time()

def _pe_update_progress(dlg, start, done, total, msg):
    from PyQt6.QtWidgets import QApplication
    import time

    elapsed = time.time() - start
    eta = _pe_format_eta(elapsed, done, total)
    dlg.setMaximum(max(int(total), 1))
    dlg.setValue(min(int(done), int(total)))
    dlg.setLabelText(
        f"{msg}\n\n"
        f"Progress: {done}/{total}\n"
        f"Elapsed: {elapsed:0.1f}s    ETA: {eta}"
    )
    QApplication.processEvents()
    if dlg.wasCanceled():
        raise RuntimeError("Cancelled by user.")

def _pe_collect_values_progress(self, mode, dlg=None, start=None, log=None):
    xs, ys, vals = [], [], []
    step = max(1, self.trace_step.value())
    lines = self.selected_lines()
    total = len(lines)

    for j, line in enumerate(lines, 1):
        if dlg is not None:
            _pe_update_progress(dlg, start, j - 1, total, f"Reading/processing {line.name} for {mode} map...")
        if log is not None:
            log.append(f"[{j}/{total}] {line.name}: reading/processing")

        arr = self.get_array(line)
        t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns

        if mode == "time":
            idx = int(np.argmin(np.abs(t - self.time_slice.value())))
            v = arr[::step, idx]
        elif mode == "depth":
            twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
            idx = int(np.argmin(np.abs(t - twt)))
            v = arr[::step, idx]
        else:
            lo, hi = sorted([self.proj_tmin.value(), self.proj_tmax.value()])
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue
            v = np.nanmax(np.abs(arr[::step][:, m]), axis=1)

        xs.extend(line.x[::step])
        ys.extend(line.y[::step])
        vals.extend(v)

        if dlg is not None:
            _pe_update_progress(dlg, start, j, total, f"Finished {line.name}")

    return np.asarray(xs), np.asarray(ys), np.asarray(vals)

def _pe_plot_map_progress(self, canvas, mode, title):
    import datetime
    log = []
    lines = self.selected_lines()
    total = len(lines) + 2
    dlg, start = _pe_make_progress(self, f"PulseEKKO {title}", total)

    try:
        _pe_update_progress(dlg, start, 0, total, f"Preparing {title}...")
        x, y, v = _pe_collect_values_progress(self, mode, dlg, start, log)

        _pe_update_progress(dlg, start, len(lines), total, "Interpolating and drawing map...")

        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax

        if len(v) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw()
        else:
            if SCIPY_OK and len(v) > 100:
                gx = np.linspace(np.nanmin(x), np.nanmax(x), 280)
                gy = np.linspace(np.nanmin(y), np.nanmax(y), 280)
                X, Y = np.meshgrid(gx, gy)
                Z = griddata((x, y), v, (X, Y), method="linear")
                im = ax.imshow(
                    Z,
                    extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                )
            else:
                im = ax.scatter(x, y, c=v, s=4, cmap="viridis")
                ax.set_aspect("equal", adjustable="box")

            ax.set_title(title)
            ax.set_xlabel("Local easting [m]")
            ax.set_ylabel("Local northing [m]")
            ax.grid(True, alpha=0.2)
            canvas.fig.colorbar(im, ax=ax, label="Amplitude")
            canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        log_path.write_text(
            "PulseEKKO 3D analysis log\n"
            f"Run: {datetime.datetime.now()}\n"
            f"View: {title}\n"
            f"Mode: {mode}\n"
            f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
        )
        self.owner.status.setText(f"Updated {title}. Log: {log_path}")

    except Exception as e:
        dlg.close()
        self.owner.status.setText(f"3D update stopped: {e}")

def _pe_plot_fence_progress(self, canvas):
    import datetime
    log = []
    lines = self.owner.lines[::max(1, self.fence_step.value())]
    lines = lines[:self.max_lines.value()]
    total = len(lines) + 1
    dlg, start = _pe_make_progress(self, "PulseEKKO selected fence diagram", total)

    try:
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111, projection="3d")
        canvas.ax = ax

        for j, line in enumerate(lines, 1):
            _pe_update_progress(dlg, start, j - 1, total, f"Building fence panel {line.name}...")
            log.append(f"[{j}/{len(lines)}] {line.name}: fence panel")

            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns

            ridx = np.linspace(0, len(line.x) - 1, min(130, len(line.x))).astype(int)
            tidx = np.linspace(0, len(t) - 1, min(150, len(t))).astype(int)

            X = np.tile(line.x[ridx], (len(tidx), 1))
            Y = np.tile(line.y[ridx], (len(tidx), 1))
            Z = -np.tile(t[tidx][:, None], (1, len(ridx)))
            A = arr[np.ix_(ridx, tidx)].T

            vmax = np.nanpercentile(np.abs(A), 99.0)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0

            colors = cm.seismic(np.clip((A / vmax + 1.0) / 2.0, 0, 1))
            ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=False, shade=False)

            _pe_update_progress(dlg, start, j, total, f"Finished fence panel {line.name}")

        ax.set_title("Selected fence diagram — Bulach PulseEKKO")
        ax.set_xlabel("Local easting [m]")
        ax.set_ylabel("Local northing [m]")
        ax.set_zlabel("-Two-way time [ns]")
        canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        log_path.write_text(
            "PulseEKKO 3D analysis log\n"
            f"Run: {datetime.datetime.now()}\n"
            "View: Selected Fence Diagram\n"
            f"Fence line step: {self.fence_step.value()}\n"
            f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
        )
        self.owner.status.setText(f"Updated selected fence diagram. Log: {log_path}")

    except Exception as e:
        dlg.close()
        self.owner.status.setText(f"Fence update stopped: {e}")

def _pe_update_selected_progress(self):
    name, canvas = self.selected_canvas()

    if name == "Survey Overview":
        dlg, start = _pe_make_progress(self, "PulseEKKO survey overview", 1)
        try:
            _pe_update_progress(dlg, start, 0, 1, "Drawing survey geometry...")
            self.plot_overview(canvas)
            _pe_update_progress(dlg, start, 1, 1, "Done.")
            dlg.close()
        except Exception as e:
            dlg.close()
            self.owner.status.setText(f"Overview update stopped: {e}")

    elif name == "Time Slice Map":
        _pe_plot_map_progress(self, canvas, "time", f"Time slice map at {self.time_slice.value():.1f} ns")

    elif name == "Depth Slice Map":
        twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
        _pe_plot_map_progress(self, canvas, "depth", f"Depth slice map at {self.depth.value():.2f} m ≈ {twt:.1f} ns")

    elif name == "Amplitude Projection":
        _pe_plot_map_progress(self, canvas, "projection", f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns")

    elif name == "Selected Fence Diagram":
        _pe_plot_fence_progress(self, canvas)

PulseEkko3DAnalysis.collect_values = _pe_collect_values_progress
PulseEkko3DAnalysis.plot_map = _pe_plot_map_progress
PulseEkko3DAnalysis.plot_fence = _pe_plot_fence_progress
PulseEkko3DAnalysis.update_selected = _pe_update_selected_progress
# ---- end hard override: PulseEKKO 3D progress dialog ----



# ---- hard override: clean PulseEKKO maps ----
def _pe_collect_values_clean_maps(self, mode, dlg=None, start=None, log=None):
    xs, ys, vals = [], [], []
    step = max(1, self.trace_step.value())
    lines = self.selected_lines()
    total = len(lines)

    edge_trim_m = 1.0  # remove first/last 1 m to suppress start/stop/coupling edge artefacts

    for j, line in enumerate(lines, 1):
        if dlg is not None:
            _pe_update_progress(dlg, start, j - 1, total, f"Reading/processing {line.name} for {mode} map...")
        if log is not None:
            log.append(f"[{j}/{total}] {line.name}: reading/processing, edge trim={edge_trim_m} m")

        arr = self.get_array(line)
        t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns

        keep = (line.dist >= edge_trim_m) & (line.dist <= (line.dist[-1] - edge_trim_m))
        if not np.any(keep):
            keep = np.ones_like(line.dist, dtype=bool)

        ridx_all = np.where(keep)[0][::step]

        if mode == "time":
            idx = int(np.argmin(np.abs(t - self.time_slice.value())))
            v = np.abs(arr[ridx_all, idx])
        elif mode == "depth":
            twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
            idx = int(np.argmin(np.abs(t - twt)))
            v = np.abs(arr[ridx_all, idx])
        else:
            lo, hi = sorted([self.proj_tmin.value(), self.proj_tmax.value()])
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue
            v = np.nanmax(np.abs(arr[ridx_all][:, m]), axis=1)

        xs.extend(line.x[ridx_all])
        ys.extend(line.y[ridx_all])
        vals.extend(v)

        if dlg is not None:
            _pe_update_progress(dlg, start, j, total, f"Finished {line.name}")

    return np.asarray(xs), np.asarray(ys), np.asarray(vals)

def _pe_plot_map_clean_progress(self, canvas, mode, title):
    import datetime
    log = []
    lines = self.selected_lines()
    total = len(lines) + 2
    dlg, start = _pe_make_progress(self, f"PulseEKKO {title}", total)

    try:
        _pe_update_progress(dlg, start, 0, total, f"Preparing {title}...")
        x, y, v = _pe_collect_values_clean_maps(self, mode, dlg, start, log)

        _pe_update_progress(dlg, start, len(lines), total, "Interpolating and drawing map...")

        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax

        if len(v) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw()
        else:
            vmin = float(np.nanpercentile(v, 2))
            vmax = float(np.nanpercentile(v, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v) + 1e-9)

            if SCIPY_OK and len(v) > 100:
                gx = np.linspace(np.nanmin(x), np.nanmax(x), 280)
                gy = np.linspace(np.nanmin(y), np.nanmax(y), 280)
                X, Y = np.meshgrid(gx, gy)
                Z = griddata((x, y), v, (X, Y), method="linear")
                im = ax.imshow(
                    Z,
                    extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                im = ax.scatter(x, y, c=v, s=4, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_aspect("equal", adjustable="box")

            ax.set_title(title + " | abs amplitude, 1 m edge trim, 2–98% colour clip")
            ax.set_xlabel("Local easting [m]")
            ax.set_ylabel("Local northing [m]")
            ax.grid(True, alpha=0.2)
            canvas.fig.colorbar(im, ax=ax, label="Absolute amplitude")
            canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        log_path.write_text(
            "PulseEKKO 3D analysis log\n"
            f"Run: {datetime.datetime.now()}\n"
            f"View: {title}\n"
            f"Mode: {mode}\n"
            "Map cleaning: abs amplitude, 1 m line-edge trim, 2–98% colour clipping\n"
            f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
        )
        self.owner.status.setText(f"Updated {title}. Log: {log_path}")

    except Exception as e:
        dlg.close()
        self.owner.status.setText(f"3D update stopped: {e}")

def _pe_update_selected_clean_maps(self):
    name, canvas = self.selected_canvas()

    if name == "Survey Overview":
        dlg, start = _pe_make_progress(self, "PulseEKKO survey overview", 1)
        try:
            _pe_update_progress(dlg, start, 0, 1, "Drawing survey geometry...")
            self.plot_overview(canvas)
            _pe_update_progress(dlg, start, 1, 1, "Done.")
            dlg.close()
        except Exception as e:
            dlg.close()
            self.owner.status.setText(f"Overview update stopped: {e}")

    elif name == "Time Slice Map":
        _pe_plot_map_clean_progress(self, canvas, "time", f"Time slice map at {self.time_slice.value():.1f} ns")

    elif name == "Depth Slice Map":
        twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
        _pe_plot_map_clean_progress(self, canvas, "depth", f"Depth slice map at {self.depth.value():.2f} m ≈ {twt:.1f} ns")

    elif name == "Amplitude Projection":
        _pe_plot_map_clean_progress(self, canvas, "projection", f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns")

    elif name == "Selected Fence Diagram":
        _pe_plot_fence_progress(self, canvas)

PulseEkko3DAnalysis.collect_values = _pe_collect_values_clean_maps
PulseEkko3DAnalysis.plot_map = _pe_plot_map_clean_progress
PulseEkko3DAnalysis.update_selected = _pe_update_selected_clean_maps
# ---- end hard override: clean PulseEKKO maps ----



# ---- hard override: standard map PulseEKKO maps ----
def _pe_collect_values_pipe_check(self, mode, dlg=None, start=None, log=None):
    xs, ys, vals = [], [], []
    step = max(1, self.trace_step.value())
    lines = self.selected_lines()
    total = len(lines)

    edge_trim_m = 1.0

    for j, line in enumerate(lines, 1):
        if dlg is not None:
            _pe_update_progress(dlg, start, j - 1, total, f"Standard map map: reading/processing {line.name}...")
        if log is not None:
            log.append(f"[{j}/{total}] {line.name}: standard map line-normalised amplitude")

        arr = self.get_array(line)
        t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns

        keep = (line.dist >= edge_trim_m) & (line.dist <= (line.dist[-1] - edge_trim_m))
        if not np.any(keep):
            keep = np.ones_like(line.dist, dtype=bool)

        ridx_all = np.where(keep)[0][::step]

        if mode == "time":
            idx = int(np.argmin(np.abs(t - self.time_slice.value())))
            v = np.abs(arr[ridx_all, idx])
        elif mode == "depth":
            twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
            idx = int(np.argmin(np.abs(t - twt)))
            v = np.abs(arr[ridx_all, idx])
        else:
            lo, hi = sorted([self.proj_tmin.value(), self.proj_tmax.value()])
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue
            v = np.nanmax(np.abs(arr[ridx_all][:, m]), axis=1)

        # Key fix: line-normalise amplitude so one bad/weak acquisition line does not become a fake pipe.
        med = np.nanmedian(v)
        p90 = np.nanpercentile(v, 90)
        scale = max(float(med), 0.25 * float(p90), 1e-9)
        v = v / scale

        xs.extend(line.x[ridx_all])
        ys.extend(line.y[ridx_all])
        vals.extend(v)

        if dlg is not None:
            _pe_update_progress(dlg, start, j, total, f"Finished {line.name}")

    return np.asarray(xs), np.asarray(ys), np.asarray(vals)

def _pe_plot_map_pipe_check(self, canvas, mode, title):
    import datetime
    log = []
    lines = self.selected_lines()
    total = len(lines) + 2
    dlg, start = _pe_make_progress(self, f"PulseEKKO standard map: {title}", total)

    try:
        _pe_update_progress(dlg, start, 0, total, f"Preparing standard map {title}...")
        x, y, v = _pe_collect_values_pipe_check(self, mode, dlg, start, log)

        _pe_update_progress(dlg, start, len(lines), total, "Interpolating and drawing standard map map...")

        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax

        if len(v) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw()
        else:
            vmin = float(np.nanpercentile(v, 2))
            vmax = float(np.nanpercentile(v, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v) + 1e-9)

            if SCIPY_OK and len(v) > 100:
                gx = np.linspace(np.nanmin(x), np.nanmax(x), 280)
                gy = np.linspace(np.nanmin(y), np.nanmax(y), 280)
                X, Y = np.meshgrid(gx, gy)
                Z = griddata((x, y), v, (X, Y), method="linear")
                im = ax.imshow(
                    Z,
                    extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                im = ax.scatter(x, y, c=v, s=4, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_aspect("equal", adjustable="box")

            ax.set_title(title + " | line-normalised absolute amplitude, 1 m edge trim, 2–98% colour clip")
            ax.set_xlabel("Local easting [m]")
            ax.set_ylabel("Local northing [m]")
            ax.grid(True, alpha=0.2)
            canvas.fig.colorbar(im, ax=ax, label="Relative absolute amplitude")
            canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        log_path.write_text(
            "PulseEKKO standard map log\n"
            f"Run: {datetime.datetime.now()}\n"
            f"View: {title}\n"
            f"Mode: {mode}\n"
            "Settings: line-normalised abs amplitude, 1 m edge trim, 2–98% colour clipping\n"
            f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
        )
        self.owner.status.setText(f"Updated standard map {title}. Log: {log_path}")

    except Exception as e:
        dlg.close()
        self.owner.status.setText(f"Standard map update stopped: {e}")

def _pe_update_selected_pipe_check(self):
    name, canvas = self.selected_canvas()

    if name == "Survey Overview":
        dlg, start = _pe_make_progress(self, "PulseEKKO survey overview", 1)
        try:
            _pe_update_progress(dlg, start, 0, 1, "Drawing survey geometry...")
            self.plot_overview(canvas)
            _pe_update_progress(dlg, start, 1, 1, "Done.")
            dlg.close()
        except Exception as e:
            dlg.close()
            self.owner.status.setText(f"Overview update stopped: {e}")

    elif name == "Time Slice Map":
        _pe_plot_map_pipe_check(self, canvas, "time", f"Time slice map at {self.time_slice.value():.1f} ns")

    elif name == "Depth Slice Map":
        twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
        _pe_plot_map_pipe_check(self, canvas, "depth", f"Depth slice map at {self.depth.value():.2f} m ≈ {twt:.1f} ns")

    elif name == "Amplitude Projection":
        _pe_plot_map_pipe_check(self, canvas, "projection", f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns")

    elif name == "Selected Fence Diagram":
        _pe_plot_fence_progress(self, canvas)

PulseEkko3DAnalysis.collect_values = _pe_collect_values_pipe_check
PulseEkko3DAnalysis.plot_map = _pe_plot_map_pipe_check
PulseEkko3DAnalysis.update_selected = _pe_update_selected_pipe_check
# ---- end hard override: standard map PulseEKKO maps ----



# ---- hard override: destriped PulseEKKO standard map maps ----
def _pe_collect_values_destriped_pipe_check(self, mode, dlg=None, start=None, log=None):
    step = max(1, self.trace_step.value())
    lines = self.selected_lines()
    total = len(lines)
    edge_trim_m = 1.0

    records = []

    for j, line in enumerate(lines, 1):
        if dlg is not None:
            _pe_update_progress(dlg, start, j - 1, total, f"De-striping map: reading/processing {line.name}...")
        if log is not None:
            log.append(f"[{j}/{total}] {line.name}: line-normalised + cross-line median de-striping")

        arr = self.get_array(line)
        t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns

        keep = (line.dist >= edge_trim_m) & (line.dist <= (line.dist[-1] - edge_trim_m))
        if not np.any(keep):
            keep = np.ones_like(line.dist, dtype=bool)

        ridx = np.where(keep)[0][::step]

        if mode == "time":
            idx = int(np.argmin(np.abs(t - self.time_slice.value())))
            v = np.abs(arr[ridx, idx])
        elif mode == "depth":
            twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
            idx = int(np.argmin(np.abs(t - twt)))
            v = np.abs(arr[ridx, idx])
        else:
            lo, hi = sorted([self.proj_tmin.value(), self.proj_tmax.value()])
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue
            v = np.nanmax(np.abs(arr[ridx][:, m]), axis=1)

        s = (line.dist[ridx] - line.dist[0]) / max(line.dist[-1] - line.dist[0], 1e-9)
        records.append((line.name, s, line.x[ridx], line.y[ridx], v))

        if dlg is not None:
            _pe_update_progress(dlg, start, j, total, f"Finished {line.name}")

    if not records:
        return np.array([]), np.array([]), np.array([])

    smin = max(float(np.nanmin(r[1])) for r in records)
    smax = min(float(np.nanmax(r[1])) for r in records)
    ngrid = int(np.nanmedian([len(r[1]) for r in records]))
    ngrid = max(120, min(600, ngrid))
    sgrid = np.linspace(smin, smax, ngrid)

    X, Y, Vraw = [], [], []
    names = []

    for name, s, x, y, v in records:
        good = np.isfinite(s) & np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        if good.sum() < 5:
            continue

        ss = s[good]
        order = np.argsort(ss)
        ss = ss[order]
        xx = x[good][order]
        yy = y[good][order]
        vv = v[good][order]

        X.append(np.interp(sgrid, ss, xx))
        Y.append(np.interp(sgrid, ss, yy))
        Vraw.append(np.interp(sgrid, ss, vv))
        names.append(name)

    X = np.asarray(X)
    Y = np.asarray(Y)
    Vraw = np.asarray(Vraw)

    if Vraw.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Detect full-line acquisition/coupling outliers using raw line median before normalisation.
    line_med = np.nanmedian(Vraw, axis=1)
    n = len(line_med)
    bad = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - 4)
        hi = min(n, i + 5)
        neigh = np.delete(line_med[lo:hi], i - lo)
        neigh = neigh[np.isfinite(neigh)]
        if len(neigh) < 2:
            continue
        ref = np.nanmedian(neigh)
        if ref > 0 and (line_med[i] < 0.45 * ref or line_med[i] > 2.20 * ref):
            bad[i] = True

    # Line normalisation: removes line-to-line gain/coupling differences.
    V = Vraw.copy()
    for i in range(n):
        med = np.nanmedian(V[i])
        p90 = np.nanpercentile(V[i], 90)
        scale = max(float(med), 0.25 * float(p90), 1e-9)
        V[i] = V[i] / scale

    # Replace clearly bad full lines by neighbouring median profile.
    for i in np.where(bad)[0]:
        lo = max(0, i - 3)
        hi = min(n, i + 4)
        neigh = V[lo:hi][~bad[lo:hi]]
        if len(neigh) > 0:
            V[i] = np.nanmedian(neigh, axis=0)

    # Cross-line de-striping: suppress features existing on only one survey line.
    Vdes = V.copy()
    radius = 2  # 5-line median filter across line direction
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        Vdes[i] = np.nanmedian(V[lo:hi], axis=0)

    if log is not None:
        bad_names = [names[i] for i in np.where(bad)[0]]
        log.append(f"Bad full-line outliers replaced: {bad_names}")
        log.append("Applied 5-line cross-line median de-striping.")

    return X.ravel(), Y.ravel(), Vdes.ravel()

def _pe_plot_map_destriped_pipe_check(self, canvas, mode, title):
    import datetime
    log = []
    lines = self.selected_lines()
    total = len(lines) + 2
    dlg, start = _pe_make_progress(self, f"PulseEKKO de-striped standard map: {title}", total)

    try:
        _pe_update_progress(dlg, start, 0, total, f"Preparing de-striped standard map {title}...")
        x, y, v = _pe_collect_values_destriped_pipe_check(self, mode, dlg, start, log)

        _pe_update_progress(dlg, start, len(lines), total, "Interpolating and drawing de-striped map...")

        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax

        if len(v) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw()
        else:
            vmin = float(np.nanpercentile(v, 2))
            vmax = float(np.nanpercentile(v, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v) + 1e-9)

            if SCIPY_OK and len(v) > 100:
                gx = np.linspace(np.nanmin(x), np.nanmax(x), 280)
                gy = np.linspace(np.nanmin(y), np.nanmax(y), 280)
                Xg, Yg = np.meshgrid(gx, gy)
                Z = griddata((x, y), v, (Xg, Yg), method="linear")
                im = ax.imshow(
                    Z,
                    extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                im = ax.scatter(x, y, c=v, s=4, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_aspect("equal", adjustable="box")

            ax.set_title(title + " | de-striped line-normalised amplitude, 5-line median, 1 m edge trim")
            ax.set_xlabel("Local easting [m]")
            ax.set_ylabel("Local northing [m]")
            ax.grid(True, alpha=0.2)
            canvas.fig.colorbar(im, ax=ax, label="Relative absolute amplitude")
            canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        log_path.write_text(
            "PulseEKKO de-striped standard map log\n"
            f"Run: {datetime.datetime.now()}\n"
            f"View: {title}\n"
            f"Mode: {mode}\n"
            "Settings: abs amplitude, line normalisation, bad-line replacement, 5-line cross-line median de-striping, 1 m edge trim, 2–98% colour clip\n"
            f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
        )
        self.owner.status.setText(f"Updated de-striped standard map {title}. Log: {log_path}")

    except Exception as e:
        dlg.close()
        self.owner.status.setText(f"De-striped standard map stopped: {e}")

def _pe_update_selected_destriped_pipe_check(self):
    name, canvas = self.selected_canvas()

    if name == "Survey Overview":
        dlg, start = _pe_make_progress(self, "PulseEKKO survey overview", 1)
        try:
            _pe_update_progress(dlg, start, 0, 1, "Drawing survey geometry...")
            self.plot_overview(canvas)
            _pe_update_progress(dlg, start, 1, 1, "Done.")
            dlg.close()
        except Exception as e:
            dlg.close()
            self.owner.status.setText(f"Overview update stopped: {e}")

    elif name == "Time Slice Map":
        _pe_plot_map_destriped_pipe_check(self, canvas, "time", f"Time slice map at {self.time_slice.value():.1f} ns")

    elif name == "Depth Slice Map":
        twt = 2.0 * self.depth.value() / max(self.velocity.value(), 1e-9)
        _pe_plot_map_destriped_pipe_check(self, canvas, "depth", f"Depth slice map at {self.depth.value():.2f} m ≈ {twt:.1f} ns")

    elif name == "Amplitude Projection":
        _pe_plot_map_destriped_pipe_check(self, canvas, "projection", f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns")

    elif name == "Selected Fence Diagram":
        _pe_plot_fence_progress(self, canvas)

PulseEkko3DAnalysis.collect_values = _pe_collect_values_destriped_pipe_check
PulseEkko3DAnalysis.plot_map = _pe_plot_map_destriped_pipe_check
PulseEkko3DAnalysis.update_selected = _pe_update_selected_destriped_pipe_check
# ---- end hard override: destriped PulseEKKO standard map maps ----


def purge_suspicious_widgets(root):
    try:
        from PyQt6.QtWidgets import QTabWidget, QComboBox
        for tw in root.findChildren(QTabWidget):
            for i in reversed(range(tw.count())):
                if "suspicious" in tw.tabText(i).lower():
                    tw.removeTab(i)
        for cb in root.findChildren(QComboBox):
            for i in reversed(range(cb.count())):
                if "suspicious" in cb.itemText(i).lower():
                    cb.removeItem(i)
    except Exception:
        pass


class FieldworkMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPR_Fieldwork_Analysis")
        self.project_windows = []

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setDocumentMode(True)
        tabs.tabBar().setExpanding(True)

        for label, root in PROJECTS:
            if "PulseEKKO" in label:
                pe_tab = PulseEkkoProjectTab(root)
                purge_suspicious_widgets(pe_tab)
                tabs.addTab(pe_tab, label)
            else:
                try:
                    win = gpr_app.MainWindow(root)
                except TypeError:
                    win = gpr_app.MainWindow()
                    if hasattr(win, "root"):
                        win.root = root
                    if hasattr(win, "load_project"):
                        win.load_project()

                purge_suspicious_widgets(win)
                self.project_windows.append(win)
                tabs.addTab(win, label)

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = FieldworkMainWindow()
    win.resize(1800, 950)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# ---- HARD OVERRIDE: replace Bulach depth slice with time-lapse GIF map ----
def _pe_assets_dir():
    d = Path(__file__).resolve().parent / "Assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pe_hide_depth_controls(self):
    """Hide arbitrary depth/depth-velocity controls after the original UI is built."""
    try:
        for attr in ("depth", "velocity"):
            w = getattr(self, attr, None)
            if w is not None:
                w.hide()
                w.setVisible(False)
        for lab in self.findChildren(QLabel):
            if lab.text().strip().lower() in {"depth", "velocity"}:
                lab.hide()
                lab.setVisible(False)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).strip().lower() == "depth slice map":
                self.tabs.setTabText(i, "Time Lapse Map")
    except Exception:
        pass


def _pe_plot_time_lapse_frame(self, ax, time_ns, title_prefix="Bulach time-lapse map"):
    old_time = None
    try:
        old_time = float(self.time_slice.value())
        self.time_slice.setValue(float(time_ns))
    except Exception:
        pass
    try:
        x, y, v = self.collect_values("time")
    finally:
        try:
            if old_time is not None:
                self.time_slice.setValue(old_time)
        except Exception:
            pass
    ax.clear()
    if len(v) == 0:
        ax.text(0.5, 0.5, f"No data at {time_ns:.1f} ns", transform=ax.transAxes, ha="center", va="center")
        return None
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        ax.text(0.5, 0.5, f"No finite data at {time_ns:.1f} ns", transform=ax.transAxes, ha="center", va="center")
        return None
    vmax = float(np.nanpercentile(np.abs(finite), 98.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = None
    try:
        if SCIPY_OK and len(v) > 100:
            gx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 280)
            gy = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 280)
            X, Y = np.meshgrid(gx, gy)
            Z = griddata((x, y), v, (X, Y), method="linear")
            im = ax.imshow(
                Z,
                extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                origin="lower",
                aspect="equal",
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
            )
        else:
            im = ax.scatter(x, y, c=v, s=4, cmap="inferno", vmin=0.0, vmax=vmax)
    except Exception:
        im = ax.scatter(x, y, c=v, s=4, cmap="inferno", vmin=0.0, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title_prefix}: {time_ns:.1f} ns")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.grid(True, alpha=0.2)
    try:
        if im is not None:
            for _ax in list(ax.figure.axes):
                if getattr(_ax, "_tl_cbar_ax", False):
                    _ax.remove()
            _tl_cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            _tl_cbar.ax._tl_cbar_ax = True
            _tl_cbar.set_label("Relative amplitude", fontsize=8)
            _tl_cbar.ax.tick_params(labelsize=7)
            try:
                ax.figure.set_size_inches(8.0, 6.0, forward=True)
                ax.set_position([0.08, 0.15, 0.73, 0.74])
                _tl_cbar.ax.set_position([0.85, 0.15, 0.025, 0.74])
            except Exception:
                pass
    except Exception:
        pass
    return im


def _pe_plot_time_lapse_map(self):
    from PyQt6.QtWidgets import QProgressDialog, QMessageBox, QApplication
    from PyQt6.QtCore import Qt
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation, PillowWriter

    tmax = float(self.proj_tmax.value())
    if not np.isfinite(tmax) or tmax <= 0:
        tmax = 50.0
    nframes = int(np.clip(round(tmax / 2.5) + 1, 12, 80))
    frames = np.linspace(0.0, tmax, nframes)

    assets = _pe_assets_dir()
    out = assets / f"Bulach_time_lapse_0_to_{tmax:.0f}ns.gif"

    dlg = QProgressDialog("Building Bulach time-lapse GIF...", "Cancel", 0, nframes + 2, self)
    dlg.setWindowTitle("Bulach time-lapse map")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    QApplication.processEvents()

    fig = Figure(figsize=(8, 6), dpi=120, constrained_layout=True)
    ax = fig.add_subplot(111)

    def update(i):
        if dlg.wasCanceled():
            raise RuntimeError("Cancelled by user.")
        tns = float(frames[i])
        dlg.setLabelText(f"Rendering frame {i + 1}/{nframes}: {tns:.1f} ns")
        dlg.setValue(i)
        QApplication.processEvents()
        _pe_plot_time_lapse_frame(self, ax, tns)
        return []

    try:
        anim = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=False, repeat=True)
        anim.save(out, writer=PillowWriter(fps=4))
        dlg.setValue(nframes + 1)
        dlg.setLabelText("Updating GUI preview...")
        QApplication.processEvents()
        name, canvas = self.selected_canvas()
        canvas.fig.clear()
        ax2 = canvas.fig.add_subplot(111)
        canvas.ax = ax2
        _pe_plot_time_lapse_frame(self, ax2, float(frames[-1]), "Bulach time-lapse preview")
        canvas.draw()
        self.owner.status.setText(f"Saved Bulach time-lapse GIF: {out}")
        QMessageBox.information(self, "Time-lapse GIF saved", f"Saved:\n{out}")
    except Exception as e:
        self.owner.status.setText(f"Time-lapse GIF failed: {e}")
        QMessageBox.critical(self, "Time-lapse GIF failed", str(e))
    finally:
        dlg.close()


try:
    _PE_ORIG_INIT_TIME_LAPSE = PulseEkko3DAnalysis.__init__
    def _pe_init_time_lapse(self, *args, **kwargs):
        _PE_ORIG_INIT_TIME_LAPSE(self, *args, **kwargs)
        _pe_hide_depth_controls(self)
    PulseEkko3DAnalysis.__init__ = _pe_init_time_lapse

    PulseEkko3DAnalysis.plot_time_lapse_map = _pe_plot_time_lapse_map

    _PE_ORIG_UPDATE_SELECTED_TIME_LAPSE = PulseEkko3DAnalysis.update_selected
    def _pe_update_selected_time_lapse(self):
        name, canvas = self.selected_canvas()
        if name.strip().lower() == "time lapse map":
            return self.plot_time_lapse_map()
        return _PE_ORIG_UPDATE_SELECTED_TIME_LAPSE(self)
    PulseEkko3DAnalysis.update_selected = _pe_update_selected_time_lapse
except Exception as _e:
    print("Bulach time-lapse override not applied:", _e)
# ---- END HARD OVERRIDE: Bulach time-lapse GIF map ----

# ---- HARD OVERRIDE V2: Bulach animated time-lapse controls ----
def _pe_tl_fmt_num(x):
    try:
        s = ("%.2f" % float(x)).rstrip("0").rstrip(".")
    except Exception:
        s = str(x)
    return s.replace(".", "p")


def _pe_tl_speed_value(self):
    try:
        return float(self.tl_speed.currentText().replace("x", ""))
    except Exception:
        return 1.0


def _pe_tl_step_value(self):
    try:
        v = float(self.tl_step_ns.value())
        return v if v > 0 else 5.0
    except Exception:
        return 5.0


def _pe_tl_frames(self):
    import numpy as _np
    try:
        tmax = float(self.proj_tmax.value())
    except Exception:
        tmax = 50.0
    if not _np.isfinite(tmax) or tmax <= 0:
        tmax = 50.0
    step = _pe_tl_step_value(self)
    frames = list(_np.arange(0.0, tmax + 0.5 * step, step, dtype=float))
    if not frames or frames[-1] < tmax:
        frames.append(float(tmax))
    return frames, float(tmax), float(step)


def _pe_tl_draw_progress_axis(pax, idx, n, time_ns, loop_no=0):
    from matplotlib.patches import Rectangle
    pax.clear()
    pax.set_xlim(0, 1)
    pax.set_ylim(0, 1)
    pax.axis("off")
    frac = 1.0 if n <= 1 else float(idx) / float(n - 1)
    pax.add_patch(Rectangle((0.02, 0.28), 0.96, 0.44, fill=False, linewidth=1.0))
    pax.add_patch(Rectangle((0.02, 0.28), 0.96 * frac, 0.44, alpha=0.65))
    pax.text(0.5, 0.5, f"{time_ns:.1f} ns | frame {idx + 1}/{n} | loop {loop_no + 1}", ha="center", va="center", fontsize=9)


def _pe_tl_current_canvas(self):
    try:
        return self.selected_canvas()[1]
    except Exception:
        return getattr(self, "depth_canvas", None)


def _pe_tl_draw_canvas_frame(self, time_ns, idx, n, loop_no=0):
    c = _pe_tl_current_canvas(self)
    if c is None:
        return
    c.fig.clear()
    gs = c.fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.28)
    ax = c.fig.add_subplot(gs[0, 0])
    pax = c.fig.add_subplot(gs[1, 0])
    _pe_plot_time_lapse_frame(self, ax, float(time_ns), "Bulach time-lapse map")
    _pe_tl_draw_progress_axis(pax, int(idx), int(n), float(time_ns), int(loop_no))
    c.draw_idle()


def _pe_tl_tick(self):
    frames = getattr(self, "_tl_frames", [])
    if not frames:
        return
    n = len(frames)
    idx = int(getattr(self, "_tl_idx", 0))
    loop_no = int(getattr(self, "_tl_loop", 0))
    if idx >= n:
        idx = 0
        loop_no += 1
        self._tl_loop = loop_no
    _pe_tl_draw_canvas_frame(self, frames[idx], idx, n, loop_no)
    try:
        self.tl_progress.setRange(0, max(0, n - 1))
        self.tl_progress.setValue(idx)
        self.tl_progress.setFormat(f"{frames[idx]:.1f} ns  |  frame {idx + 1}/{n}  |  loop {loop_no + 1}")
    except Exception:
        pass
    self._tl_idx = idx + 1


def _pe_tl_start_preview(self, frames):
    from PyQt6.QtCore import QTimer
    try:
        if getattr(self, "_tl_timer", None) is not None:
            self._tl_timer.stop()
    except Exception:
        pass
    self._tl_frames = [float(x) for x in frames]
    self._tl_idx = 0
    self._tl_loop = 0
    timer = QTimer(self)
    timer.timeout.connect(lambda: _pe_tl_tick(self))
    self._tl_timer = timer
    _pe_tl_tick(self)
    interval_ms = max(40, int(round(250.0 / _pe_tl_speed_value(self))))
    timer.start(interval_ms)


def _pe_tl_open_latest(self):
    from pathlib import Path as _Path
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import QMessageBox
    p = _Path(str(getattr(self, "_tl_last_gif", "")))
    if p.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
    else:
        QMessageBox.information(self, "No GIF yet", "Generate the Time Lapse Map first.")


def _pe_tl_save_as(self):
    import shutil
    from pathlib import Path as _Path
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    p = _Path(str(getattr(self, "_tl_last_gif", "")))
    if not p.exists():
        QMessageBox.information(self, "No GIF yet", "Generate the Time Lapse Map first.")
        return
    out, _ = QFileDialog.getSaveFileName(self, "Save time-lapse GIF as", str(p), "GIF files (*.gif)")
    if out:
        if not out.lower().endswith(".gif"):
            out += ".gif"
        shutil.copyfile(p, out)
        QMessageBox.information(self, "GIF copied", f"Saved copy:\n{out}")


def _pe_tl_add_controls(self):
    try:
        if getattr(self, "_tl_controls_added_v2", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QProgressBar
        row = QWidget(self)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)
        box.addWidget(QLabel("Time-lapse step"))
        self.tl_step_ns = QDoubleSpinBox(row)
        self.tl_step_ns.setRange(0.5, 50.0)
        self.tl_step_ns.setDecimals(1)
        self.tl_step_ns.setSingleStep(0.5)
        self.tl_step_ns.setSuffix(" ns")
        self.tl_step_ns.setValue(5.0)
        box.addWidget(self.tl_step_ns)
        box.addWidget(QLabel("Playback"))
        self.tl_speed = QComboBox(row)
        self.tl_speed.addItems(["0.5x", "1.0x", "1.5x"])
        self.tl_speed.setCurrentText("1.0x")
        box.addWidget(self.tl_speed)
        self.tl_open_btn = QPushButton("Open GIF", row)
        self.tl_saveas_btn = QPushButton("Save GIF As", row)
        box.addWidget(self.tl_open_btn)
        box.addWidget(self.tl_saveas_btn)
        box.addStretch(1)
        self.tl_progress = QProgressBar(self)
        self.tl_progress.setTextVisible(True)
        self.tl_progress.setFormat("Time-lapse not generated")
        self.tl_open_btn.clicked.connect(lambda: _pe_tl_open_latest(self))
        self.tl_saveas_btn.clicked.connect(lambda: _pe_tl_save_as(self))
        lay = self.layout()
        if lay is not None:
            lay.insertWidget(1, row)
            lay.addWidget(self.tl_progress)
        self._tl_controls_added_v2 = True
    except Exception as e:
        print("Bulach time-lapse controls not added:", e)


def _pe_plot_time_lapse_map_v2(self):
    from pathlib import Path as _Path
    from PyQt6.QtWidgets import QProgressDialog, QMessageBox, QApplication
    from PyQt6.QtCore import Qt
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames, tmax, step = _pe_tl_frames(self)
    speed = _pe_tl_speed_value(self)
    n = len(frames)
    assets = _Path(__file__).resolve().parent / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / f"Bulach_time_lapse_0_to_{_pe_tl_fmt_num(tmax)}ns_step_{_pe_tl_fmt_num(step)}ns_speed_{_pe_tl_fmt_num(speed)}x.gif"

    dlg = QProgressDialog("Building Bulach time-lapse GIF...", "Cancel", 0, n, self)
    dlg.setWindowTitle("Bulach time-lapse map")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    QApplication.processEvents()

    fig = Figure(figsize=(8, 6), dpi=120, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    pax = fig.add_subplot(gs[1, 0])

    def update(i):
        if dlg.wasCanceled():
            raise RuntimeError("Cancelled by user.")
        tns = float(frames[i])
        dlg.setLabelText(f"Rendering frame {i + 1}/{n}: {tns:.1f} ns")
        dlg.setValue(i)
        QApplication.processEvents()
        _pe_plot_time_lapse_frame(self, ax, tns, "Bulach time-lapse map")
        _pe_tl_draw_progress_axis(pax, i, n, tns, 0)
        return []

    try:
        anim = FuncAnimation(fig, update, frames=n, interval=max(40, int(round(250.0 / speed))), blit=False, repeat=True)
        anim.save(out, writer=PillowWriter(fps=max(1, int(round(4.0 * speed)))))
        dlg.setValue(n)
        self._tl_last_gif = str(out)
        _pe_tl_start_preview(self, frames)
        self.owner.status.setText(f"Saved Bulach time-lapse GIF: {out}")
        QMessageBox.information(self, "Time-lapse GIF saved", f"Saved:\n{out}")
    except Exception as e:
        self.owner.status.setText(f"Time-lapse GIF failed: {e}")
        QMessageBox.critical(self, "Time-lapse GIF failed", str(e))
    finally:
        dlg.close()


try:
    _PE_TL_V2_PREV_INIT = PulseEkko3DAnalysis.__init__
    def _pe_tl_v2_init(self, *args, **kwargs):
        _PE_TL_V2_PREV_INIT(self, *args, **kwargs)
        try:
            _pe_hide_depth_controls(self)
        except Exception:
            pass
        _pe_tl_add_controls(self)
    PulseEkko3DAnalysis.__init__ = _pe_tl_v2_init

    PulseEkko3DAnalysis.plot_time_lapse_map = _pe_plot_time_lapse_map_v2

    _PE_TL_V2_PREV_UPDATE = PulseEkko3DAnalysis.update_selected
    def _pe_tl_v2_update_selected(self):
        name, canvas = self.selected_canvas()
        if name.strip().lower() == "time lapse map":
            return self.plot_time_lapse_map()
        return _PE_TL_V2_PREV_UPDATE(self)
    PulseEkko3DAnalysis.update_selected = _pe_tl_v2_update_selected
except Exception as _e:
    print("Bulach animated time-lapse override not applied:", _e)
# ---- END HARD OVERRIDE V2: Bulach animated time-lapse controls ----

# --- time lapse open/save safe override ---
def _tl_latest_gif(prefix, self=None):
    from pathlib import Path
    root = Path(__file__).resolve().parent
    assets = root / "Assets"
    assets.mkdir(exist_ok=True)
    names = ("tl_last_gif_path", "last_time_lapse_gif", "_time_lapse_gif_path", "time_lapse_gif_path", "last_gif_path")
    for name in names:
        q = getattr(self, name, None) if self is not None else None
        if q and Path(q).is_file() and str(q).lower().endswith(".gif"):
            return Path(q)
    c = list(assets.glob(prefix + "*.gif")) + list(assets.glob("*" + prefix.split("_")[0] + "*time*lapse*.gif"))
    c = [p for p in c if p.is_file() and p.suffix.lower() == ".gif"]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None

def _tl_warn(self, msg):
    try:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Time-lapse GIF", msg)
    except Exception:
        print(msg)

def _tl_open_gif(prefix, self=None):
    p = _tl_latest_gif(prefix, self)
    if not p:
        return _tl_warn(self, "No valid time-lapse GIF found. Generate the Time Lapse Map first.")
    import subprocess
    subprocess.Popen(["xdg-open", str(p)])

def _tl_save_gif_as(prefix, self=None):
    p = _tl_latest_gif(prefix, self)
    if not p:
        return _tl_warn(self, "No valid time-lapse GIF found. Generate the Time Lapse Map first.")
    from pathlib import Path
    import shutil
    try:
        from PyQt6.QtWidgets import QFileDialog
        out, _ = QFileDialog.getSaveFileName(self, "Save time-lapse GIF as", str(Path.home() / p.name), "GIF files (*.gif)")
    except Exception:
        out = str(Path.home() / p.name)
    if not out:
        return
    if not out.lower().endswith(".gif"):
        out += ".gif"
    shutil.copyfile(str(p), out)
    print("Saved GIF as:", out)

def _schl_tl_open_gif(self):
    return _tl_open_gif("Schleitheim_time_lapse", self)

def _schl_tl_save_as(self):
    return _tl_save_gif_as("Schleitheim_time_lapse", self)

def _bulach_tl_open_gif(self):
    return _tl_open_gif("Bulach_time_lapse", self)

def _bulach_tl_save_as(self):
    return _tl_save_gif_as("Bulach_time_lapse", self)

def _pulse_tl_open_gif(self):
    return _tl_open_gif("Bulach_time_lapse", self)

def _pulse_tl_save_as(self):
    return _tl_save_gif_as("Bulach_time_lapse", self)
# --- time lapse return-im colourbar patch ---

# --- selected_canvas time lapse alias patch ---

# --- fixed geometry for time-lapse saved GIF patch ---


# --- stable_time_lapse_patch import: Bulach ---
try:
    import stable_time_lapse_patch as _stable_tl_patch
    _stable_tl_patch.apply_bulach(globals())
except Exception as _e:
    print("Stable Bulach time-lapse patch not applied:", _e)
# --- end stable_time_lapse_patch import: Bulach ---



# --- direction amplitude balancing patch ---
def _pe_dir_amp_add_control(self):
    try:
        if getattr(self, "_direction_amp_added", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
        row = QWidget(self)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)
        box.addWidget(QLabel("Direction amplitude handling"))
        self.direction_amp = QComboBox(row)
        self.direction_amp.addItems(["No direction balancing", "Balance inline/crossline groups", "Per-line normalisation"])
        self.direction_amp.setCurrentText("No direction balancing")
        box.addWidget(self.direction_amp)
        box.addStretch(1)
        lay = self.layout()
        if lay is not None:
            lay.insertWidget(2 if lay.count() >= 2 else lay.count(), row)
        self._direction_amp_added = True
    except Exception as e:
        print("Bulach direction amplitude control not added:", e)
try:
    if not getattr(PulseEkko3DAnalysis, "_direction_amp_patched", False):
        _pe_old_init_dir_amp = PulseEkko3DAnalysis.__init__
        def _pe_new_init_dir_amp(self, *a, **kw):
            _pe_old_init_dir_amp(self, *a, **kw)
            _pe_dir_amp_add_control(self)
        PulseEkko3DAnalysis.__init__ = _pe_new_init_dir_amp
        PulseEkko3DAnalysis._direction_amp_patched = True
except Exception as _e:
    print("Bulach direction amplitude patch failed:", _e)
# --- end direction amplitude balancing patch ---


# ---- 3D STOLT MIGRATION HOOK ----
try:
    import gpr3d_migration as _gpr3d_mig
    _gpr3d_mig.apply_bulach(globals())
except Exception as _e:
    print("3-D Stolt migration hook failed for Bulach/PulseEKKO:", _e)
# ---- END 3D STOLT MIGRATION HOOK ----


# --- Bulach time-lapse signed blue-white-red colour patch ---
def _pe_plot_time_lapse_frame(self, ax, time_ns, title_prefix="Bulach time-lapse map"):
    """Bulach time-lapse frame using signed amplitude: blue negative, white zero, red positive."""
    import numpy as _np

    old_time = None
    try:
        old_time = float(self.time_slice.value())
        self.time_slice.setValue(float(time_ns))
    except Exception:
        pass

    try:
        x, y, v = self.collect_values("time")
    finally:
        try:
            if old_time is not None:
                self.time_slice.setValue(old_time)
        except Exception:
            pass

    ax.clear()

    if len(v) == 0:
        ax.text(0.5, 0.5, f"No data at {time_ns:.1f} ns", transform=ax.transAxes, ha="center", va="center")
        return None

    finite = v[_np.isfinite(v)]
    if finite.size == 0:
        ax.text(0.5, 0.5, f"No finite data at {time_ns:.1f} ns", transform=ax.transAxes, ha="center", va="center")
        return None

    vmax = float(_np.nanpercentile(_np.abs(finite), 98.5))
    if not _np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = None
    try:
        from scipy.interpolate import griddata
        if len(v) > 100:
            gx = _np.linspace(float(_np.nanmin(x)), float(_np.nanmax(x)), 280)
            gy = _np.linspace(float(_np.nanmin(y)), float(_np.nanmax(y)), 280)
            X, Y = _np.meshgrid(gx, gy)
            Z = griddata((x, y), v, (X, Y), method="linear")
            im = ax.imshow(
                Z,
                extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                origin="lower",
                aspect="equal",
                cmap="seismic",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            im = ax.scatter(x, y, c=v, s=4, cmap="seismic", vmin=-vmax, vmax=vmax)
    except Exception:
        im = ax.scatter(x, y, c=v, s=4, cmap="seismic", vmin=-vmax, vmax=vmax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title_prefix}: {time_ns:.1f} ns | signed amplitude, blue-white-red zero-centred scale")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.grid(True, alpha=0.2)

    try:
        if im is not None:
            for _ax in list(ax.figure.axes):
                if getattr(_ax, "_tl_cbar_ax", False):
                    _ax.remove()
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax._tl_cbar_ax = True
            cbar.set_label("Signed amplitude", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            try:
                ax.figure.set_size_inches(8.0, 6.0, forward=True)
                ax.set_position([0.08, 0.15, 0.73, 0.74])
                cbar.ax.set_position([0.85, 0.15, 0.025, 0.74])
            except Exception:
                pass
    except Exception:
        pass

    return im
# --- end Bulach time-lapse signed blue-white-red colour patch ---


# --- restore Bulach clear inline processing defaults patch ---
def _pe_restore_clear_processing_defaults(self):
    """Restore the clearer PulseEKKO inline processing/display defaults."""
    try:
        self.dewow_window.setValue(5.0)
        self.low_cut.setValue(50.0)
        self.high_cut.setValue(250.0)
        self.sec_power.setValue(0.90)
        self.display_clip.setValue(99.50)
        self.bg_remove.setChecked(True)
        self.bandpass.setChecked(True)
        self.agc_gain.setChecked(False)
        self.agc_window.setValue(80.0)
        self.display_min.setValue(0.0)
        self.display_max.setValue(50.0)
        self.vertical_exag.setValue(1.50)
        self.scale.setCurrentText('symlog')
        self.cmap.setCurrentText('seismic')
    except Exception:
        pass


def _pe_clear_processed_cache(self, *args):
    try:
        for line in getattr(self, 'lines', []):
            line.proc = None
        if hasattr(self, 'status'):
            self.status.setText('Processing settings changed; processed cache cleared. Click Process current line.')
    except Exception:
        pass


try:
    if not hasattr(PulseEkkoProjectTab, '_orig_init_restore_clear_processing_defaults'):
        PulseEkkoProjectTab._orig_init_restore_clear_processing_defaults = PulseEkkoProjectTab.__init__

    def _pe_init_restore_clear_processing_defaults(self, *args, **kwargs):
        PulseEkkoProjectTab._orig_init_restore_clear_processing_defaults(self, *args, **kwargs)
        _pe_restore_clear_processing_defaults(self)

        try:
            for w in [self.dewow_window, self.low_cut, self.high_cut, self.sec_power, self.bg_window,
                      self.agc_window, self.display_min, self.display_max, self.display_clip, self.vertical_exag]:
                try:
                    w.valueChanged.connect(lambda *_, obj=self: _pe_clear_processed_cache(obj))
                except Exception:
                    pass
            for cb in [self.bg_remove, self.bandpass, self.agc_gain, self.t0_auto]:
                try:
                    cb.toggled.connect(lambda *_, obj=self: _pe_clear_processed_cache(obj))
                except Exception:
                    pass
            for cb in [self.scale, self.cmap]:
                try:
                    cb.currentTextChanged.connect(lambda *_, obj=self: _pe_clear_processed_cache(obj))
                except Exception:
                    pass
        except Exception:
            pass

    PulseEkkoProjectTab.__init__ = _pe_init_restore_clear_processing_defaults
except Exception:
    pass
# --- end restore Bulach clear inline processing defaults patch ---


# --- multi time-slice figure patch: Bulach/PulseEKKO ---

def _multi_ts_save_hd_pulseekko(self):
    try:
        from pathlib import Path
        import datetime
        canvas = self.canvases.get('Time Slice Map') if hasattr(self, 'canvases') else self.selected_canvas()[1]
        out_dir = Path(getattr(getattr(self, 'owner', None), 'root', Path('.')))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = out_dir / f"bulach_multislice_time_slice_map_{ts}.png"
        pdf = out_dir / f"bulach_multislice_time_slice_map_{ts}.pdf"
        canvas.fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
        canvas.fig.savefig(pdf, dpi=350, bbox_inches="tight", facecolor="white")
        status = getattr(getattr(self, 'owner', None), 'status', None)
        if status is not None:
            try:
                status.setText(f"Saved HD plot: {png} and {pdf}")
            except Exception:
                pass
        print(f"Saved HD plot: {png}")
        print(f"Saved HD plot: {pdf}")
    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save HD plot failed", str(e))
        except Exception:
            print("Save HD plot failed:", e)

def _multi_ts_install_pulseekko(self):
    try:
        if getattr(self, "_multi_ts_controls_added", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox

        row = QWidget(self)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)

        box.addWidget(QLabel("Time slice count"))
        self.time_slice_count = QSpinBox(row)
        self.time_slice_count.setRange(1, 10)
        self.time_slice_count.setValue(4)
        box.addWidget(self.time_slice_count)

        self.time_slice_values = []
        try:
            base = float(self.time_slice.value())
        except Exception:
            base = 5.0

        for i in range(10):
            lab = QLabel(f"T{i+1}", row)
            sp = QDoubleSpinBox(row)
            sp.setRange(0.0, 5000.0)
            sp.setDecimals(2)
            sp.setSingleStep(1.0)
            sp.setSuffix(" ns")
            sp.setValue([5.0, 10.0, 15.0, 20.0, 7.5, 12.5, 17.5, 22.5, 25.0, 30.0][i])
            sp.setMinimumWidth(92)
            box.addWidget(lab)
            box.addWidget(sp)
            self.time_slice_values.append((lab, sp))
        from PyQt6.QtWidgets import QPushButton
        self.multi_ts_save_hd = QPushButton("Save HD Plot", row)
        self.multi_ts_save_hd.setToolTip("Save current multi-time-slice figure as high-resolution PNG and PDF.")
        self.multi_ts_save_hd.clicked.connect(lambda *_: _multi_ts_save_hd_pulseekko(self))
        box.addWidget(self.multi_ts_save_hd)


        box.addStretch(1)

        def refresh(*_):
            n = int(self.time_slice_count.value())
            for j, (lab, sp) in enumerate(self.time_slice_values):
                vis = j < n
                lab.setVisible(vis)
                sp.setVisible(vis)

        self.time_slice_count.valueChanged.connect(refresh)
        try:
            self.time_slice.valueChanged.connect(lambda v: self.time_slice_values[0][1].setValue(float(v)))
        except Exception:
            pass
        refresh()

        lay = self.layout()
        if lay is not None:
            lay.insertWidget(1, row)

        self._multi_ts_controls_added = True
    except Exception as e:
        print("PulseEKKO multi time-slice controls not added:", e)


def _multi_ts_pe_values_at(self, time_ns):
    import numpy as np

    xs, ys, vals = [], [], []
    step = max(1, int(self.trace_step.value()))
    try:
        lines = self.selected_lines()
    except Exception:
        lines = getattr(getattr(self, "owner", None), "lines", [])

    for line in lines:
        try:
            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns
            t = np.asarray(t, float)
            if arr is None or arr.ndim != 2 or len(t) < 2:
                continue
            if float(time_ns) < float(np.nanmin(t)) or float(time_ns) > float(np.nanmax(t)):
                continue
            j = int(np.argmin(np.abs(t - float(time_ns))))

            keep = (line.dist >= 1.0) & (line.dist <= (line.dist[-1] - 1.0))
            if not np.any(keep):
                keep = np.ones_like(line.dist, dtype=bool)
            idx = np.where(keep)[0][::step]

            xs.extend(np.asarray(line.x, float)[idx])
            ys.extend(np.asarray(line.y, float)[idx])
            vals.extend(np.asarray(arr, float)[idx, j])
        except Exception as e:
            print("Skipping PulseEKKO time slice", getattr(line, "name", line), e)

    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(vals, float)


def _multi_ts_plot_pulseekko(self, canvas):
    import math
    import numpy as np

    times = [float(sp.value()) for _, sp in self.time_slice_values[:int(self.time_slice_count.value())]]
    canvas.fig.clear()

    slices = []
    all_vals = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")

    for tv in times:
        x, y, val = _multi_ts_pe_values_at(self, tv)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
        x, y, val = x[good], y[good], val[good]
        slices.append((tv, x, y, val))
        if len(val):
            all_vals.append(val)
            xmin = min(xmin, float(np.nanmin(x)))
            xmax = max(xmax, float(np.nanmax(x)))
            ymin = min(ymin, float(np.nanmin(y)))
            ymax = max(ymax, float(np.nanmax(y)))

    if not all_vals:
        ax = canvas.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Not enough points for selected time slices", transform=ax.transAxes, ha="center", va="center")
        canvas.draw()
        return

    allv = np.concatenate(all_vals)
    amp = float(np.nanpercentile(np.abs(allv[np.isfinite(allv)]), 98.5))
    if not np.isfinite(amp) or amp <= 0:
        amp = 1.0

    try:
        cmap = self.owner.cmap.currentText()
    except Exception:
        cmap = "seismic"

    n = len(slices)
    if n <= 4:
        cols = 2
        rows = int(math.ceil(n / 2))
    else:
        cols = min(4, n)
        rows = int(math.ceil(n / cols))
    pad_x = max((xmax - xmin) * 0.03, 0.5)
    pad_y = max((ymax - ymin) * 0.08, 0.5)
    xmin, xmax, ymin, ymax = xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

    axes = []
    last_im = None
    for i, (tv, x, y, val) in enumerate(slices, 1):
        ax = canvas.fig.add_subplot(rows, cols, i)
        axes.append(ax)

        finite_val = val[np.isfinite(val)]
        if finite_val.size:
            local_amp = float(np.nanpercentile(np.abs(finite_val), pct if "pct" in locals() else 98.5))
        else:
            local_amp = amp
        if not np.isfinite(local_amp) or local_amp <= 0:
            local_amp = amp if np.isfinite(amp) and amp > 0 else 1.0

        if len(val) >= 4:
            try:
                from scipy.interpolate import griddata
                xi = np.linspace(xmin, xmax, 240)
                yi = np.linspace(ymin, ymax, 150)
                X, Y = np.meshgrid(xi, yi)
                Z = griddata((x, y), val, (X, Y), method="linear")
                last_im = ax.imshow(
                    Z, extent=[xmin, xmax, ymin, ymax], origin="lower",
                    aspect="equal", cmap=cmap, vmin=-local_amp, vmax=local_amp
                )
            except Exception:
                last_im = ax.scatter(x, y, c=val, s=3, cmap=cmap, vmin=-local_amp, vmax=local_amp)
                ax.set_aspect("equal", adjustable="box")
        else:
            ax.text(0.5, 0.5, "Not enough points", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(f"{tv:.1f} ns", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Local easting [m]", fontsize=9)
        ax.set_ylabel("Local northing [m]", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)
        if last_im is not None:
            try:
                cb = canvas.fig.colorbar(last_im, ax=ax, fraction=0.032, pad=0.010)
                cb.ax.tick_params(labelsize=7)
                cb.set_label("Amplitude", fontsize=7)
            except Exception:
                pass  # current-slice colourbar for this panel

    canvas.fig.suptitle(f"Time slice maps ({n} slices) — current-slice colour scale", fontsize=12)
    canvas.fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.900, wspace=0.16, hspace=0.34)
    # No shared colourbar: each panel uses its own current-slice colour scale.
    canvas.draw()
    try:
        self.owner.status.setText("Updated multi time-slice map: " + ", ".join(f"{t:.1f} ns" for t in times))
    except Exception:
        pass


try:
    if not hasattr(PulseEkko3DAnalysis, "_multi_ts_old_init"):
        PulseEkko3DAnalysis._multi_ts_old_init = PulseEkko3DAnalysis.__init__
        def _multi_ts_new_init_pe(self, *args, **kwargs):
            PulseEkko3DAnalysis._multi_ts_old_init(self, *args, **kwargs)
            _multi_ts_install_pulseekko(self)
        PulseEkko3DAnalysis.__init__ = _multi_ts_new_init_pe

    if "_multi_ts_old_update_pe" not in globals():
        _multi_ts_old_update_pe = PulseEkko3DAnalysis.update_selected

    def _multi_ts_update_selected_pe(self):
        try:
            name, canvas = self.selected_canvas()
            if name == "Time Slice Map" and hasattr(self, "time_slice_count") and hasattr(self, "time_slice_values"):
                return _multi_ts_plot_pulseekko(self, canvas)
        except Exception:
            pass
        return _multi_ts_old_update_pe(self)

    PulseEkko3DAnalysis.update_selected = _multi_ts_update_selected_pe
except Exception as e:
    print("PulseEKKO multi time-slice patch failed:", e)
# --- end multi time-slice figure patch: Bulach/PulseEKKO ---


# --- Bulach multi-time-slice responsive large layout patch ---
def _bulach_multi_ts_status(self, msg):
    try:
        self.owner.status.setText(str(msg))
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass


def _bulach_multi_ts_save_hd(self):
    try:
        from pathlib import Path
        import datetime
        canvas = self.canvases.get("Time Slice Map") if hasattr(self, "canvases") else self.selected_canvas()[1]
        out_dir = Path(getattr(getattr(self, "owner", None), "root", Path(".")))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = out_dir / f"bulach_multislice_time_slice_map_{ts}.png"
        pdf = out_dir / f"bulach_multislice_time_slice_map_{ts}.pdf"
        canvas.fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
        canvas.fig.savefig(pdf, dpi=350, bbox_inches="tight", facecolor="white")
        _bulach_multi_ts_status(self, f"Saved HD plot: {png} and {pdf}")
        print(f"Saved HD plot: {png}")
        print(f"Saved HD plot: {pdf}")
    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save HD plot failed", str(e))
        except Exception:
            print("Save HD plot failed:", e)


def _bulach_multi_ts_add_save_button(self):
    try:
        if getattr(self, "_bulach_big_save_button_added", False):
            return
        from PyQt6.QtWidgets import QPushButton
        btn = QPushButton("Save HD Plot", self)
        btn.setToolTip("Save the current multi-time-slice figure as high-resolution PNG and PDF.")
        btn.clicked.connect(lambda *_: _bulach_multi_ts_save_hd(self))
        lay = self.layout()
        if lay is not None:
            insert_at = 2 if lay.count() >= 2 else lay.count()
            lay.insertWidget(insert_at, btn)
        self._bulach_big_save_button_added = True
    except Exception as e:
        print("Could not add Bulach HD save button:", e)


def _bulach_multi_ts_values_at_progress(self, time_ns, dlg=None, base=0, total=1):
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    xs, ys, vals = [], [], []
    step = max(1, int(self.trace_step.value()))
    try:
        lines = list(self.selected_lines())
    except Exception:
        lines = list(getattr(getattr(self, "owner", None), "lines", []))

    nlines = max(1, len(lines))
    for j, line in enumerate(lines, 1):
        if dlg is not None:
            dlg.setValue(min(base + j, total))
            dlg.setLabelText(f"Bulach time-slice map\n{float(time_ns):.1f} ns — processing {getattr(line, 'name', line)} ({j}/{nlines})")
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")

        if j % 3 == 0 or j == 1 or j == nlines:
            _bulach_multi_ts_status(self, f"Bulach multi-slice: {float(time_ns):.1f} ns, line {j}/{nlines}")

        try:
            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns
            t = np.asarray(t, float)
            if arr is None or arr.ndim != 2 or len(t) < 2:
                continue
            if float(time_ns) < float(np.nanmin(t)) or float(time_ns) > float(np.nanmax(t)):
                continue
            tidx = int(np.argmin(np.abs(t - float(time_ns))))

            keep = (line.dist >= 1.0) & (line.dist <= (line.dist[-1] - 1.0))
            if not np.any(keep):
                keep = np.ones_like(line.dist, dtype=bool)
            ridx = np.where(keep)[0][::step]

            xs.extend(np.asarray(line.x, float)[ridx])
            ys.extend(np.asarray(line.y, float)[ridx])
            vals.extend(np.asarray(arr, float)[ridx, tidx])
        except Exception as e:
            print("Skipping PulseEKKO time slice", getattr(line, "name", line), e)

    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(vals, float), len(lines)


def _bulach_multi_ts_plot_big_progress(self, canvas):
    import math
    import numpy as np
    from PyQt6.QtWidgets import QProgressDialog, QApplication
    from PyQt6.QtCore import Qt

    times = [float(sp.value()) for _, sp in self.time_slice_values[:int(self.time_slice_count.value())]]
    if not times:
        return

    try:
        nlines = len(list(self.selected_lines()))
    except Exception:
        nlines = len(getattr(getattr(self, "owner", None), "lines", []))
    total = max(1, len(times) * max(1, nlines) + 5)

    dlg = QProgressDialog("Building Bulach multi-time-slice map...", "Cancel", 0, total, self)
    dlg.setWindowTitle("Bulach time-slice maps")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.resize(540, 150)
    dlg.show()
    QApplication.processEvents()

    try:
        canvas.fig.clear()
        _bulach_multi_ts_status(self, "Bulach multi-slice: collecting data...")

        slices = []
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        for i, tv in enumerate(times):
            x, y, val, used_lines = _bulach_multi_ts_values_at_progress(
                self, tv, dlg=dlg, base=i * max(1, nlines), total=total
            )
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
            x, y, val = x[good], y[good], val[good]
            slices.append((tv, x, y, val))
            if len(val):
                xmin = min(xmin, float(np.nanmin(x)))
                xmax = max(xmax, float(np.nanmax(x)))
                ymin = min(ymin, float(np.nanmin(y)))
                ymax = max(ymax, float(np.nanmax(y)))

        dlg.setValue(min(total - 3, total))
        dlg.setLabelText("Drawing 2×2 figure...")
        QApplication.processEvents()

        if not any(len(v) for _, _, _, v in slices):
            ax = canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough points for selected time slices", transform=ax.transAxes, ha="center", va="center")
            canvas.draw()
            dlg.close()
            _bulach_multi_ts_status(self, "Bulach multi-slice: no usable data")
            return

        try:
            cmap = self.owner.cmap.currentText()
        except Exception:
            cmap = "seismic"

        n = len(slices)
        if n <= 4:
            rows, cols = 2, 2
        else:
            cols = min(4, n)
            rows = int(math.ceil(n / cols))

        canvas.fig.subplots_adjust(left=0.040, right=0.985, bottom=0.065, top=0.900, wspace=0.115, hspace=0.285)

        pad_x = max((xmax - xmin) * 0.03, 0.5)
        pad_y = max((ymax - ymin) * 0.03, 0.5)
        xmin2, xmax2 = xmin - pad_x, xmax + pad_x
        ymin2, ymax2 = ymin - pad_y, ymax + pad_y

        for k, (tv, x, y, val) in enumerate(slices, 1):
            ax = canvas.fig.add_subplot(rows, cols, k)

            finite = val[np.isfinite(val)]
            amp = float(np.nanpercentile(np.abs(finite), 98.5)) if finite.size else 1.0
            if not np.isfinite(amp) or amp <= 0:
                amp = 1.0

            im = None
            if len(val) >= 4:
                try:
                    from scipy.interpolate import griddata
                    xi = np.linspace(xmin2, xmax2, 360)
                    yi = np.linspace(ymin2, ymax2, 260)
                    X, Y = np.meshgrid(xi, yi)
                    Z = griddata((x, y), val, (X, Y), method="linear")
                    im = ax.imshow(
                        Z,
                        extent=[xmin2, xmax2, ymin2, ymax2],
                        origin="lower",
                        aspect="auto",
                        cmap=cmap,
                        vmin=-amp,
                        vmax=amp,
                        interpolation="nearest",
                    )
                except Exception:
                    im = ax.scatter(x, y, c=val, s=3, cmap=cmap, vmin=-amp, vmax=amp)
                    ax.set_aspect("auto")
            else:
                ax.text(0.5, 0.5, "Not enough points", transform=ax.transAxes, ha="center", va="center")

            ax.set_title(f"{tv:.1f} ns", fontsize=11)
            ax.set_xlabel("Local easting [m]", fontsize=9)
            ax.set_ylabel("Local northing [m]", fontsize=9)
            ax.set_xlim(xmin2, xmax2)
            ax.set_ylim(ymin2, ymax2)
            ax.grid(True, alpha=0.18)
            ax.tick_params(labelsize=9)

            if im is not None:
                cb = canvas.fig.colorbar(im, ax=ax, fraction=0.032, pad=0.010)
                cb.ax.tick_params(labelsize=7)
                cb.set_label("Amplitude", fontsize=7)

            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")
            dlg.setValue(min(total - 2 + k, total))
            QApplication.processEvents()

        canvas.fig.suptitle(f"Bulach time slice maps ({n} slices) — current-slice colour scale", fontsize=13)
        canvas.draw()
        dlg.setValue(total)
        dlg.close()
        _bulach_multi_ts_status(self, "Bulach multi-slice map updated.")
    except Exception as e:
        try:
            dlg.close()
        except Exception:
            pass
        _bulach_multi_ts_status(self, f"Bulach multi-slice stopped: {e}")
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Bulach multi-slice stopped", str(e))
        except Exception:
            print("Bulach multi-slice stopped:", e)


try:
    if not hasattr(PulseEkko3DAnalysis, "_bulach_big_multi_old_init"):
        PulseEkko3DAnalysis._bulach_big_multi_old_init = PulseEkko3DAnalysis.__init__
        def _bulach_big_multi_new_init(self, *args, **kwargs):
            PulseEkko3DAnalysis._bulach_big_multi_old_init(self, *args, **kwargs)
            _bulach_multi_ts_add_save_button(self)
        PulseEkko3DAnalysis.__init__ = _bulach_big_multi_new_init

    if "_bulach_big_multi_old_update" not in globals():
        _bulach_big_multi_old_update = PulseEkko3DAnalysis.update_selected

    def _bulach_big_multi_update_selected(self):
        try:
            name, canvas = self.selected_canvas()
            if name == "Time Slice Map" and hasattr(self, "time_slice_count") and hasattr(self, "time_slice_values"):
                return _bulach_multi_ts_plot_big_progress(self, canvas)
        except Exception:
            pass
        return _bulach_big_multi_old_update(self)

    PulseEkko3DAnalysis.update_selected = _bulach_big_multi_update_selected
    PulseEkko3DAnalysis.plot_multi_time_slices = _bulach_multi_ts_plot_big_progress
except Exception as e:
    print("Bulach multi-slice big/progress patch failed:", e)
# --- end Bulach multi-time-slice responsive large layout patch ---


# --- Bulach equal-scale scrollable multi-time-slice override ---
def _bulach_equal_multits_status(self, msg):
    try:
        self.owner.status.setText(str(msg))
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass


def _bulach_equal_multits_make_scrollable(self):
    try:
        if getattr(self, "_bulach_equal_scrollable_added", False):
            return
        from PyQt6.QtWidgets import QScrollArea
        from PyQt6.QtCore import Qt

        canvas = self.canvases.get("Time Slice Map")
        if canvas is None:
            return

        try:
            canvas.fig.set_size_inches(17.5, 11.5, forward=True)
            canvas.setMinimumSize(1750, 1150)
            canvas.resize(1750, 1150)
        except Exception:
            pass

        idx = None
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Time Slice Map":
                idx = i
                break
        if idx is None:
            return

        old_widget = self.tabs.widget(idx)
        if isinstance(old_widget, QScrollArea):
            self._bulach_equal_scrollable_added = True
            return

        self.tabs.removeTab(idx)
        scroll = QScrollArea(self.tabs)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(canvas)
        self.tabs.insertTab(idx, scroll, "Time Slice Map")
        self.canvases["Time Slice Map"] = canvas
        self._bulach_equal_scrollable_added = True
    except Exception as e:
        print("Bulach scrollable multi-slice setup failed:", e)


def _bulach_equal_multits_save_hd(self):
    try:
        from pathlib import Path
        import datetime
        canvas = self.canvases.get("Time Slice Map") if hasattr(self, "canvases") else self.selected_canvas()[1]
        out_dir = Path(getattr(getattr(self, "owner", None), "root", Path(".")))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = out_dir / f"bulach_multislice_equal_scale_{ts}.png"
        pdf = out_dir / f"bulach_multislice_equal_scale_{ts}.pdf"

        old_size = canvas.fig.get_size_inches()
        try:
            canvas.fig.set_size_inches(17.5, 11.5, forward=False)
            canvas.fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
            canvas.fig.savefig(pdf, dpi=350, bbox_inches="tight", facecolor="white")
        finally:
            canvas.fig.set_size_inches(old_size, forward=False)

        _bulach_equal_multits_status(self, f"Saved equal-scale HD plot: {png} and {pdf}")
        print(f"Saved equal-scale HD plot: {png}")
        print(f"Saved equal-scale HD plot: {pdf}")
    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save HD plot failed", str(e))
        except Exception:
            print("Save HD plot failed:", e)


def _bulach_equal_multits_add_save_button(self):
    try:
        if getattr(self, "_bulach_equal_save_button_added", False):
            return
        from PyQt6.QtWidgets import QPushButton
        btn = QPushButton("Save HD Equal-Scale Plot", self)
        btn.setToolTip("Save current Bulach multi-time-slice figure as equal-scale HD PNG and PDF.")
        btn.clicked.connect(lambda *_: _bulach_equal_multits_save_hd(self))
        lay = self.layout()
        if lay is not None:
            insert_at = 2 if lay.count() >= 2 else lay.count()
            lay.insertWidget(insert_at, btn)
        self._bulach_equal_save_button_added = True
    except Exception as e:
        print("Could not add Bulach equal-scale save button:", e)


def _bulach_equal_multits_values_at(self, time_ns, dlg=None, base=0, total=1):
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    xs, ys, vals = [], [], []
    step = max(1, int(self.trace_step.value()))
    try:
        lines = list(self.selected_lines())
    except Exception:
        lines = list(getattr(getattr(self, "owner", None), "lines", []))

    nlines = max(1, len(lines))
    for j, line in enumerate(lines, 1):
        if dlg is not None:
            dlg.setValue(min(base + j, total))
            dlg.setLabelText(f"Bulach time-slice map\\n{float(time_ns):.1f} ns — processing {getattr(line, 'name', line)} ({j}/{nlines})")
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")

        if j % 3 == 0 or j == 1 or j == nlines:
            _bulach_equal_multits_status(self, f"Bulach multi-slice: {float(time_ns):.1f} ns, line {j}/{nlines}")

        try:
            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line) if hasattr(self.owner, "corrected_time_ns") else line.time_ns
            t = np.asarray(t, float)
            if arr is None or arr.ndim != 2 or len(t) < 2:
                continue
            if float(time_ns) < float(np.nanmin(t)) or float(time_ns) > float(np.nanmax(t)):
                continue
            tidx = int(np.argmin(np.abs(t - float(time_ns))))

            keep = (line.dist >= 1.0) & (line.dist <= (line.dist[-1] - 1.0))
            if not np.any(keep):
                keep = np.ones_like(line.dist, dtype=bool)
            ridx = np.where(keep)[0][::step]

            xs.extend(np.asarray(line.x, float)[ridx])
            ys.extend(np.asarray(line.y, float)[ridx])
            vals.extend(np.asarray(arr, float)[ridx, tidx])
        except Exception as e:
            print("Skipping PulseEKKO time slice", getattr(line, "name", line), e)

    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(vals, float), len(lines)


def _bulach_equal_multits_plot(self, canvas):
    import math
    import numpy as np
    from PyQt6.QtWidgets import QProgressDialog, QApplication
    from PyQt6.QtCore import Qt

    _bulach_equal_multits_make_scrollable(self)

    canvas = self.canvases.get("Time Slice Map", canvas)
    try:
        canvas.fig.set_size_inches(17.5, 11.5, forward=True)
        canvas.setMinimumSize(1750, 1150)
        canvas.resize(1750, 1150)
    except Exception:
        pass

    times = [float(sp.value()) for _, sp in self.time_slice_values[:int(self.time_slice_count.value())]]
    if not times:
        return

    try:
        nlines = len(list(self.selected_lines()))
    except Exception:
        nlines = len(getattr(getattr(self, "owner", None), "lines", []))
    total = max(1, len(times) * max(1, nlines) + 5)

    dlg = QProgressDialog("Building Bulach equal-scale multi-time-slice map...", "Cancel", 0, total, self)
    dlg.setWindowTitle("Bulach time-slice maps")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.resize(560, 155)
    dlg.show()
    QApplication.processEvents()

    try:
        canvas.fig.clear()
        _bulach_equal_multits_status(self, "Bulach multi-slice: collecting data...")

        slices = []
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        for i, tv in enumerate(times):
            x, y, val, used_lines = _bulach_equal_multits_values_at(
                self, tv, dlg=dlg, base=i * max(1, nlines), total=total
            )
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
            x, y, val = x[good], y[good], val[good]
            slices.append((tv, x, y, val))
            if len(val):
                xmin = min(xmin, float(np.nanmin(x)))
                xmax = max(xmax, float(np.nanmax(x)))
                ymin = min(ymin, float(np.nanmin(y)))
                ymax = max(ymax, float(np.nanmax(y)))

        dlg.setValue(min(total - 3, total))
        dlg.setLabelText("Drawing equal-scale 2×2 figure...")
        QApplication.processEvents()

        if not any(len(v) for _, _, _, v in slices):
            ax = canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough points for selected time slices", transform=ax.transAxes, ha="center", va="center")
            canvas.draw()
            dlg.close()
            _bulach_equal_multits_status(self, "Bulach multi-slice: no usable data")
            return

        try:
            cmap = self.owner.cmap.currentText()
        except Exception:
            cmap = "seismic"

        n = len(slices)
        if n <= 4:
            rows, cols = 2, 2
        else:
            cols = min(4, n)
            rows = int(math.ceil(n / cols))

        canvas.fig.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.905, wspace=0.18, hspace=0.34)

        pad_x = max((xmax - xmin) * 0.03, 0.5)
        pad_y = max((ymax - ymin) * 0.03, 0.5)
        xmin2, xmax2 = xmin - pad_x, xmax + pad_x
        ymin2, ymax2 = ymin - pad_y, ymax + pad_y

        for k, (tv, x, y, val) in enumerate(slices, 1):
            ax = canvas.fig.add_subplot(rows, cols, k)

            finite = val[np.isfinite(val)]
            amp = float(np.nanpercentile(np.abs(finite), 98.5)) if finite.size else 1.0
            if not np.isfinite(amp) or amp <= 0:
                amp = 1.0

            im = None
            if len(val) >= 4:
                try:
                    from scipy.interpolate import griddata
                    xi = np.linspace(xmin2, xmax2, 360)
                    yi = np.linspace(ymin2, ymax2, 260)
                    X, Y = np.meshgrid(xi, yi)
                    Z = griddata((x, y), val, (X, Y), method="linear")
                    im = ax.imshow(
                        Z,
                        extent=[xmin2, xmax2, ymin2, ymax2],
                        origin="lower",
                        aspect="equal",
                        cmap=cmap,
                        vmin=-amp,
                        vmax=amp,
                        interpolation="nearest",
                    )
                except Exception:
                    im = ax.scatter(x, y, c=val, s=3, cmap=cmap, vmin=-amp, vmax=amp)
                    ax.set_aspect("equal", adjustable="box")
            else:
                ax.text(0.5, 0.5, "Not enough points", transform=ax.transAxes, ha="center", va="center")

            ax.set_title(f"{tv:.1f} ns", fontsize=11)
            ax.set_xlabel("Local easting [m]", fontsize=9)
            ax.set_ylabel("Local northing [m]", fontsize=9)
            ax.set_xlim(xmin2, xmax2)
            ax.set_ylim(ymin2, ymax2)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.18)
            ax.tick_params(labelsize=9)

            if im is not None:
                cb = canvas.fig.colorbar(im, ax=ax, fraction=0.032, pad=0.010)
                cb.ax.tick_params(labelsize=7)
                cb.set_label("Amplitude", fontsize=7)

            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")
            dlg.setValue(min(total - 2 + k, total))
            QApplication.processEvents()

        canvas.fig.suptitle(f"Bulach time slice maps ({n} slices) — equal XY scale, current-slice colour scale", fontsize=13)
        canvas.draw()
        try:
            canvas.resize(1750, 1150)
        except Exception:
            pass
        dlg.setValue(total)
        dlg.close()
        _bulach_equal_multits_status(self, "Bulach equal-scale multi-slice map updated.")
    except Exception as e:
        try:
            dlg.close()
        except Exception:
            pass
        _bulach_equal_multits_status(self, f"Bulach multi-slice stopped: {e}")
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Bulach multi-slice stopped", str(e))
        except Exception:
            print("Bulach multi-slice stopped:", e)


try:
    if not hasattr(PulseEkko3DAnalysis, "_bulach_equal_old_init"):
        PulseEkko3DAnalysis._bulach_equal_old_init = PulseEkko3DAnalysis.__init__
        def _bulach_equal_new_init(self, *args, **kwargs):
            PulseEkko3DAnalysis._bulach_equal_old_init(self, *args, **kwargs)
            _bulach_equal_multits_make_scrollable(self)
            _bulach_equal_multits_add_save_button(self)
        PulseEkko3DAnalysis.__init__ = _bulach_equal_new_init

    if "_bulach_equal_old_update" not in globals():
        _bulach_equal_old_update = PulseEkko3DAnalysis.update_selected

    def _bulach_equal_update_selected(self):
        try:
            name, canvas = self.selected_canvas()
            if name == "Time Slice Map" and hasattr(self, "time_slice_count") and hasattr(self, "time_slice_values"):
                return _bulach_equal_multits_plot(self, canvas)
        except Exception:
            pass
        return _bulach_equal_old_update(self)

    PulseEkko3DAnalysis.update_selected = _bulach_equal_update_selected
    PulseEkko3DAnalysis.plot_multi_time_slices = _bulach_equal_multits_plot
except Exception as e:
    print("Bulach equal-scale scrollable multi-slice patch failed:", e)
# --- end Bulach equal-scale scrollable multi-time-slice override ---


# --- Bulach radargram side-by-side fit-to-screen patch ---
def _pe_make_pair_side_by_side(pair):
    try:
        if getattr(pair, "_pe_side_by_side_done", False):
            return

        from PyQt6.QtWidgets import QSplitter, QSizePolicy
        from PyQt6.QtCore import Qt

        lay = pair.layout()
        if lay is None:
            return

        raw_scroll = getattr(pair, "raw_scroll", None)
        proc_scroll = getattr(pair, "proc_scroll", None)
        raw_canvas = getattr(pair, "raw_canvas", None)
        proc_canvas = getattr(pair, "proc_canvas", None)
        if raw_scroll is None or proc_scroll is None or raw_canvas is None or proc_canvas is None:
            return

        # Remove old top/bottom layout.
        try:
            lay.removeWidget(raw_scroll)
            lay.removeWidget(proc_scroll)
        except Exception:
            pass

        splitter = QSplitter(Qt.Orientation.Horizontal, pair)
        splitter.addWidget(raw_scroll)
        splitter.addWidget(proc_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        for scroll, canvas in [(raw_scroll, raw_canvas), (proc_scroll, proc_canvas)]:
            scroll.setWidgetResizable(True)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumWidth(420)
            canvas.setMinimumHeight(470)
            canvas.setMaximumWidth(16777215)
            canvas.setMaximumHeight(16777215)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            try:
                canvas.fig.set_size_inches(8.0, 5.0, forward=True)
                canvas.fig.tight_layout()
            except Exception:
                pass

        lay.addWidget(splitter, 1)
        pair._pe_side_splitter = splitter
        pair._pe_side_by_side_done = True
    except Exception as e:
        print("Bulach side-by-side radargram setup failed:", e)


def _pe_fit_pair_canvases(pair):
    try:
        from PyQt6.QtWidgets import QSizePolicy

        for canvas in [getattr(pair, "raw_canvas", None), getattr(pair, "proc_canvas", None)]:
            if canvas is None:
                continue
            canvas.setMinimumWidth(420)
            canvas.setMinimumHeight(470)
            canvas.setMaximumWidth(16777215)
            canvas.setMaximumHeight(16777215)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            try:
                canvas.fig.set_size_inches(8.0, 5.0, forward=True)
                canvas.fig.tight_layout()
            except Exception:
                pass

        try:
            pair.updateGeometry()
        except Exception:
            pass
    except Exception as e:
        print("Bulach radargram fit failed:", e)


def _pe_apply_side_by_side_to_all(project_tab):
    try:
        widgets = []
        if hasattr(project_tab, "line_widgets"):
            widgets.extend(list(project_tab.line_widgets.values()))
        if hasattr(project_tab, "inline_tabs"):
            for i in range(project_tab.inline_tabs.count()):
                widgets.append(project_tab.inline_tabs.widget(i))

        seen = set()
        for w in widgets:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            if isinstance(w, RadargramPair):
                _pe_make_pair_side_by_side(w)
                _pe_fit_pair_canvases(w)

        try:
            project_tab.status.setText("Bulach radargrams set to side-by-side fit-to-screen view.")
        except Exception:
            pass
    except Exception as e:
        print("Apply Bulach side-by-side to all failed:", e)


try:
    if not hasattr(RadargramPair, "_old_init_for_side_by_side_fit"):
        RadargramPair._old_init_for_side_by_side_fit = RadargramPair.__init__
        def _radargram_pair_init_side_by_side(self, *args, **kwargs):
            RadargramPair._old_init_for_side_by_side_fit(self, *args, **kwargs)
            _pe_make_pair_side_by_side(self)
            _pe_fit_pair_canvases(self)
        RadargramPair.__init__ = _radargram_pair_init_side_by_side

    if not hasattr(RadargramPair, "_old_plot_for_side_by_side_fit"):
        RadargramPair._old_plot_for_side_by_side_fit = RadargramPair.plot
        def _radargram_pair_plot_side_by_side(self, *args, **kwargs):
            out = RadargramPair._old_plot_for_side_by_side_fit(self, *args, **kwargs)
            _pe_make_pair_side_by_side(self)
            _pe_fit_pair_canvases(self)
            return out
        RadargramPair.plot = _radargram_pair_plot_side_by_side

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_inline_tabs_for_side_by_side_fit"):
        PulseEkkoProjectTab._old_rebuild_inline_tabs_for_side_by_side_fit = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_inline_tabs_side_by_side(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_inline_tabs_for_side_by_side_fit(self, *args, **kwargs)
            _pe_apply_side_by_side_to_all(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_inline_tabs_side_by_side

    if not hasattr(PulseEkkoProjectTab, "_old_init_for_side_by_side_fit"):
        PulseEkkoProjectTab._old_init_for_side_by_side_fit = PulseEkkoProjectTab.__init__
        def _pe_project_init_side_by_side(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_side_by_side_fit(self, *args, **kwargs)
            _pe_apply_side_by_side_to_all(self)
        PulseEkkoProjectTab.__init__ = _pe_project_init_side_by_side

except Exception as e:
    print("Bulach radargram side-by-side fit-to-screen patch failed:", e)
# --- end Bulach radargram side-by-side fit-to-screen patch ---


# --- Bulach radargram width option patch ---
def _pe_width_mode_to_px(mode):
    if mode == "Compact":
        return 700
    if mode == "Normal":
        return 950
    if mode == "Wide":
        return 1250
    if mode == "Very wide":
        return 1650
    return None


def _pe_apply_radargram_width_to_pair(pair, mode=None):
    try:
        from PyQt6.QtWidgets import QSizePolicy

        if mode is None:
            try:
                mode = pair.owner.radargram_width_mode.currentText()
            except Exception:
                mode = "Fit to screen"

        fixed_width = _pe_width_mode_to_px(mode)
        canvases = [getattr(pair, "raw_canvas", None), getattr(pair, "proc_canvas", None)]
        scrolls = [getattr(pair, "raw_scroll", None), getattr(pair, "proc_scroll", None)]

        for scroll in scrolls:
            if scroll is None:
                continue
            try:
                scroll.setWidgetResizable(fixed_width is None)
            except Exception:
                pass
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for canvas in canvases:
            if canvas is None:
                continue
            if fixed_width is None:
                canvas.setMinimumWidth(420)
                canvas.setMaximumWidth(16777215)
                canvas.setMinimumHeight(470)
                canvas.setMaximumHeight(16777215)
                canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                try:
                    canvas.fig.set_size_inches(7.8, 4.9, forward=True)
                    canvas.fig.tight_layout()
                except Exception:
                    pass
            else:
                canvas.setMinimumWidth(fixed_width)
                canvas.setMaximumWidth(fixed_width)
                canvas.setMinimumHeight(470)
                canvas.setMaximumHeight(16777215)
                canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
                try:
                    canvas.fig.set_size_inches(max(fixed_width / 110.0, 6.0), 4.9, forward=True)
                    canvas.fig.tight_layout()
                except Exception:
                    pass
        try:
            pair.updateGeometry()
        except Exception:
            pass
    except Exception as e:
        print("Bulach radargram width apply failed:", e)


def _pe_apply_radargram_width_to_all(project_tab):
    try:
        mode = "Fit to screen"
        try:
            mode = project_tab.radargram_width_mode.currentText()
        except Exception:
            pass

        widgets = []
        if hasattr(project_tab, "line_widgets"):
            widgets.extend(list(project_tab.line_widgets.values()))
        if hasattr(project_tab, "inline_tabs"):
            for i in range(project_tab.inline_tabs.count()):
                widgets.append(project_tab.inline_tabs.widget(i))

        seen = set()
        for w in widgets:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            if isinstance(w, RadargramPair):
                try:
                    if "_pe_make_pair_side_by_side" in globals():
                        _pe_make_pair_side_by_side(w)
                except Exception:
                    pass
                _pe_apply_radargram_width_to_pair(w, mode)

        try:
            project_tab.status.setText(f"Bulach radargram width: {mode}")
        except Exception:
            pass
    except Exception as e:
        print("Apply Bulach radargram width to all failed:", e)


def _pe_add_radargram_width_control(project_tab):
    try:
        if getattr(project_tab, "_radargram_width_control_added", False):
            return

        from PyQt6.QtWidgets import QLabel, QComboBox

        project_tab.radargram_width_mode = QComboBox()
        project_tab.radargram_width_mode.addItems([
            "Fit to screen",
            "Compact",
            "Normal",
            "Wide",
            "Very wide",
        ])
        project_tab.radargram_width_mode.setCurrentText("Fit to screen")
        project_tab.radargram_width_mode.setToolTip(
            "Display-only radargram width. Fit to screen uses available panel width; "
            "manual modes use fixed scrollable widths."
        )

        grid = None
        try:
            lay = project_tab.inline_widget.layout()
            if lay is not None and lay.count() > 0:
                item = lay.itemAt(0)
                if item is not None:
                    grid = item.layout()
        except Exception:
            grid = None

        if grid is not None:
            row = 5
            grid.addWidget(QLabel("Radargram width"), row, 0)
            grid.addWidget(project_tab.radargram_width_mode, row, 1, 1, 2)
            project_tab._radargram_width_control_added = True

        try:
            project_tab.radargram_width_mode.currentTextChanged.connect(
                lambda *_: _pe_apply_radargram_width_to_all(project_tab)
            )
            project_tab.radargram_width_mode.currentTextChanged.connect(
                lambda *_: project_tab.process_current_line()
            )
        except Exception:
            pass

        _pe_apply_radargram_width_to_all(project_tab)
    except Exception as e:
        print("Could not add Bulach radargram width option:", e)


try:
    if not hasattr(RadargramPair, "_old_plot_for_bulach_width_option"):
        RadargramPair._old_plot_for_bulach_width_option = RadargramPair.plot
        def _radargram_pair_plot_with_width_option(self, *args, **kwargs):
            out = RadargramPair._old_plot_for_bulach_width_option(self, *args, **kwargs)
            try:
                _pe_apply_radargram_width_to_pair(self)
            except Exception:
                pass
            return out
        RadargramPair.plot = _radargram_pair_plot_with_width_option

    if not hasattr(PulseEkkoProjectTab, "_old_init_for_bulach_width_option"):
        PulseEkkoProjectTab._old_init_for_bulach_width_option = PulseEkkoProjectTab.__init__
        def _pe_project_init_with_width_option(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_bulach_width_option(self, *args, **kwargs)
            _pe_add_radargram_width_control(self)
        PulseEkkoProjectTab.__init__ = _pe_project_init_with_width_option

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_bulach_width_option"):
        PulseEkkoProjectTab._old_rebuild_for_bulach_width_option = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_with_width_option(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_bulach_width_option(self, *args, **kwargs)
            _pe_add_radargram_width_control(self)
            _pe_apply_radargram_width_to_all(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_with_width_option

except Exception as e:
    print("Bulach radargram width option patch failed:", e)
# --- end Bulach radargram width option patch ---


# --- Bulach compact default and white-canvas fix patch ---
def _pe_force_compact_pair(pair):
    try:
        from PyQt6.QtWidgets import QSizePolicy

        # Keep side-by-side, but avoid widget-resizable canvas blow-up that can turn the plot white.
        try:
            if "_pe_make_pair_side_by_side" in globals():
                _pe_make_pair_side_by_side(pair)
        except Exception:
            pass

        fixed_width = 700
        fixed_height = 460

        for scroll in [getattr(pair, "raw_scroll", None), getattr(pair, "proc_scroll", None)]:
            if scroll is None:
                continue
            try:
                scroll.setWidgetResizable(False)
            except Exception:
                pass
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for canvas in [getattr(pair, "raw_canvas", None), getattr(pair, "proc_canvas", None)]:
            if canvas is None:
                continue
            canvas.setMinimumWidth(fixed_width)
            canvas.setMaximumWidth(fixed_width)
            canvas.setMinimumHeight(fixed_height)
            canvas.setMaximumHeight(fixed_height)
            canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            try:
                canvas.fig.set_size_inches(6.8, 4.4, forward=True)
                canvas.fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.90)
                canvas.draw_idle()
            except Exception:
                pass

        try:
            pair.updateGeometry()
        except Exception:
            pass
    except Exception as e:
        print("Bulach compact pair fix failed:", e)


def _pe_force_compact_default(project_tab):
    try:
        combo = getattr(project_tab, "radargram_width_mode", None)
        if combo is not None and combo.findText("Compact") >= 0:
            combo.setCurrentText("Compact")

        widgets = []
        if hasattr(project_tab, "line_widgets"):
            widgets.extend(list(project_tab.line_widgets.values()))
        if hasattr(project_tab, "inline_tabs"):
            for i in range(project_tab.inline_tabs.count()):
                widgets.append(project_tab.inline_tabs.widget(i))

        seen = set()
        for w in widgets:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            if isinstance(w, RadargramPair):
                _pe_force_compact_pair(w)

        try:
            project_tab.status.setText("Bulach radargram width: Compact")
        except Exception:
            pass
    except Exception as e:
        print("Could not force Bulach compact radargram default:", e)


try:
    if not hasattr(RadargramPair, "_old_plot_for_compact_white_fix"):
        RadargramPair._old_plot_for_compact_white_fix = RadargramPair.plot
        def _radargram_pair_plot_compact_white_fix(self, *args, **kwargs):
            out = RadargramPair._old_plot_for_compact_white_fix(self, *args, **kwargs)
            _pe_force_compact_pair(self)
            return out
        RadargramPair.plot = _radargram_pair_plot_compact_white_fix

    if not hasattr(PulseEkkoProjectTab, "_old_init_for_compact_white_fix"):
        PulseEkkoProjectTab._old_init_for_compact_white_fix = PulseEkkoProjectTab.__init__
        def _pe_project_init_compact_white_fix(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_compact_white_fix(self, *args, **kwargs)
            _pe_force_compact_default(self)
        PulseEkkoProjectTab.__init__ = _pe_project_init_compact_white_fix

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_compact_white_fix"):
        PulseEkkoProjectTab._old_rebuild_for_compact_white_fix = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_compact_white_fix(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_compact_white_fix(self, *args, **kwargs)
            _pe_force_compact_default(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_compact_white_fix

    if hasattr(PulseEkkoProjectTab, "process_current_line") and not hasattr(PulseEkkoProjectTab, "_old_process_current_for_compact_white_fix"):
        PulseEkkoProjectTab._old_process_current_for_compact_white_fix = PulseEkkoProjectTab.process_current_line
        def _pe_process_current_compact_white_fix(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_process_current_for_compact_white_fix(self)
            _pe_force_compact_default(self)
            return out
        PulseEkkoProjectTab.process_current_line = _pe_process_current_compact_white_fix

except Exception as e:
    print("Bulach compact default/white-canvas fix patch failed:", e)
# --- end Bulach compact default and white-canvas fix patch ---


# --- Final Bulach radargram plot layout cleanup patch ---
def _pe_final_radar_width_px(project_tab):
    try:
        mode = project_tab.radargram_width_mode.currentText()
    except Exception:
        mode = "Compact"
    if mode == "Compact":
        return 760, 360
    if mode == "Normal":
        return 980, 380
    if mode == "Wide":
        return 1250, 400
    if mode == "Very wide":
        return 1650, 430
    # Fit to screen fallback
    return 760, 360


def _pe_final_make_side_layout(pair):
    try:
        from PyQt6.QtWidgets import QSplitter, QSizePolicy
        from PyQt6.QtCore import Qt

        raw_scroll = getattr(pair, "raw_scroll", None)
        proc_scroll = getattr(pair, "proc_scroll", None)
        if raw_scroll is None or proc_scroll is None:
            return

        lay = pair.layout()
        if lay is None:
            return

        # Reuse existing splitter if one of the older patches made it.
        splitter = getattr(pair, "_pe_final_splitter", None) or getattr(pair, "_pe_side_splitter", None)
        if splitter is None:
            try:
                lay.removeWidget(raw_scroll)
                lay.removeWidget(proc_scroll)
            except Exception:
                pass
            splitter = QSplitter(Qt.Orientation.Horizontal, pair)
            splitter.addWidget(raw_scroll)
            splitter.addWidget(proc_scroll)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setChildrenCollapsible(False)
            lay.addWidget(splitter, 1)

        pair._pe_final_splitter = splitter

        for scroll in (raw_scroll, proc_scroll):
            try:
                scroll.setWidgetResizable(False)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            except Exception:
                pass

        try:
            splitter.setSizes([1, 1])
        except Exception:
            pass
    except Exception as e:
        print("Final Bulach side layout failed:", e)


def _pe_final_apply_canvas_size(pair):
    try:
        from PyQt6.QtWidgets import QSizePolicy
        width, height = _pe_final_radar_width_px(pair.owner)

        for canvas in (getattr(pair, "raw_canvas", None), getattr(pair, "proc_canvas", None)):
            if canvas is None:
                continue
            canvas.setMinimumWidth(width)
            canvas.setMaximumWidth(width)
            canvas.setMinimumHeight(height)
            canvas.setMaximumHeight(height)
            canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            try:
                dpi = float(canvas.fig.dpi)
                canvas.fig.set_size_inches(width / dpi, height / dpi, forward=True)
            except Exception:
                pass

        try:
            pair.updateGeometry()
        except Exception:
            pass
    except Exception as e:
        print("Final Bulach canvas sizing failed:", e)


def _pe_final_plot_one(canvas, line, owner, arr, title, processed=False):
    import numpy as np

    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    canvas.ax = ax

    if arr is None:
        ax.text(
            0.5, 0.5,
            "Click 'Process current line' to generate processed radargram" if processed else "Click 'Load raw'",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.92)
        canvas.draw_idle()
        return

    t = owner.corrected_time_ns(line)
    tmin = owner.display_min.value()
    tmax = owner.display_max.value()
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    data = arr[:, mask]
    tt = t[mask]
    cmap = owner.cmap.currentText()
    pct = owner.display_clip.value()
    vmin, vmax = clip_limits(data, pct)

    ax.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[line.dist[0], line.dist[-1], tt[-1], tt[0]],
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("Distance along line [m]", fontsize=8)
    ax.set_ylabel("Two-way time [ns]", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    # Tight but safe margins. This prevents title clipping and removes large white borders.
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.155, top=0.875)
    canvas.draw_idle()


def _pe_final_radargram_plot(self):
    line = self.line
    if line.raw is None:
        try:
            self.placeholder()
        except Exception:
            pass
        return

    _pe_final_make_side_layout(self)
    _pe_final_apply_canvas_size(self)

    raw_title = f"Raw — {line.name}"
    proc_label = "Processed"
    try:
        if line.proc is not None:
            proc_label = "Processed"
    except Exception:
        pass
    proc_title = f"{proc_label} — {line.name}"

    _pe_final_plot_one(self.raw_canvas, line, self.owner, line.raw, raw_title, processed=False)
    _pe_final_plot_one(self.proc_canvas, line, self.owner, line.proc, proc_title, processed=True)


def _pe_final_apply_all(project_tab):
    try:
        combo = getattr(project_tab, "radargram_width_mode", None)
        if combo is not None and combo.findText("Compact") >= 0:
            combo.setCurrentText("Compact")

        widgets = []
        if hasattr(project_tab, "line_widgets"):
            widgets.extend(list(project_tab.line_widgets.values()))
        if hasattr(project_tab, "inline_tabs"):
            for i in range(project_tab.inline_tabs.count()):
                widgets.append(project_tab.inline_tabs.widget(i))

        seen = set()
        for w in widgets:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            if isinstance(w, RadargramPair):
                _pe_final_make_side_layout(w)
                _pe_final_apply_canvas_size(w)
                try:
                    if getattr(w.line, "raw", None) is not None:
                        _pe_final_radargram_plot(w)
                except Exception:
                    pass

        try:
            project_tab.status.setText("Bulach radargrams cleaned: side-by-side compact view.")
        except Exception:
            pass
    except Exception as e:
        print("Final Bulach apply-all failed:", e)


try:
    # Final override wins over earlier stacked plot wrappers.
    RadargramPair.plot = _pe_final_radargram_plot

    if not hasattr(RadargramPair, "_old_init_for_final_bulach_cleanup"):
        RadargramPair._old_init_for_final_bulach_cleanup = RadargramPair.__init__
        def _radargram_pair_init_final_cleanup(self, *args, **kwargs):
            RadargramPair._old_init_for_final_bulach_cleanup(self, *args, **kwargs)
            _pe_final_make_side_layout(self)
            _pe_final_apply_canvas_size(self)
        RadargramPair.__init__ = _radargram_pair_init_final_cleanup

    if not hasattr(PulseEkkoProjectTab, "_old_init_for_final_bulach_cleanup"):
        PulseEkkoProjectTab._old_init_for_final_bulach_cleanup = PulseEkkoProjectTab.__init__
        def _pe_project_init_final_cleanup(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_final_bulach_cleanup(self, *args, **kwargs)
            _pe_final_apply_all(self)
        PulseEkkoProjectTab.__init__ = _pe_project_init_final_cleanup

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_final_bulach_cleanup"):
        PulseEkkoProjectTab._old_rebuild_for_final_bulach_cleanup = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_final_cleanup(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_final_bulach_cleanup(self, *args, **kwargs)
            _pe_final_apply_all(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_final_cleanup

    if not hasattr(PulseEkkoProjectTab, "_old_process_for_final_bulach_cleanup"):
        PulseEkkoProjectTab._old_process_for_final_bulach_cleanup = PulseEkkoProjectTab.process_current_line
        def _pe_process_final_cleanup(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_process_for_final_bulach_cleanup(self)
            _pe_final_apply_all(self)
            return out
        PulseEkkoProjectTab.process_current_line = _pe_process_final_cleanup

    print("Final Bulach radargram plot layout cleanup patch active.")
except Exception as e:
    print("Final Bulach radargram plot layout cleanup patch failed:", e)
# --- end Final Bulach radargram plot layout cleanup patch ---


# --- Bulach filled fast radargram layout patch ---
def _pe_filled_fast_make_side_layout(pair):
    try:
        from PyQt6.QtWidgets import QSplitter, QSizePolicy
        from PyQt6.QtCore import Qt

        raw_scroll = getattr(pair, "raw_scroll", None)
        proc_scroll = getattr(pair, "proc_scroll", None)
        if raw_scroll is None or proc_scroll is None:
            return

        lay = pair.layout()
        if lay is None:
            return

        splitter = getattr(pair, "_pe_filled_fast_splitter", None)
        if splitter is None:
            try:
                lay.removeWidget(raw_scroll)
                lay.removeWidget(proc_scroll)
            except Exception:
                pass

            splitter = QSplitter(Qt.Orientation.Horizontal, pair)
            splitter.addWidget(raw_scroll)
            splitter.addWidget(proc_scroll)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setChildrenCollapsible(False)
            lay.addWidget(splitter, 1)
            pair._pe_filled_fast_splitter = splitter

        for scroll in (raw_scroll, proc_scroll):
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for canvas in (getattr(pair, "raw_canvas", None), getattr(pair, "proc_canvas", None)):
            if canvas is None:
                continue
            canvas.setMinimumWidth(320)
            canvas.setMinimumHeight(300)
            canvas.setMaximumWidth(16777215)
            canvas.setMaximumHeight(16777215)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            try:
                canvas.fig.set_size_inches(7.6, 4.2, forward=False)
            except Exception:
                pass

        try:
            splitter.setSizes([1, 1])
        except Exception:
            pass
    except Exception as e:
        print("Bulach filled/fast side layout failed:", e)


def _pe_filled_fast_plot_one(canvas, line, owner, arr, title, processed=False):
    import numpy as np

    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    canvas.ax = ax

    if arr is None:
        ax.text(
            0.5, 0.5,
            "Click 'Process current line'" if processed else "Click 'Load raw'",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.90)
        canvas.draw_idle()
        return

    t = owner.corrected_time_ns(line)
    tmin = owner.display_min.value()
    tmax = owner.display_max.value()
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    data = arr[:, mask]
    tt = t[mask]
    cmap = owner.cmap.currentText()
    pct = owner.display_clip.value()
    vmin, vmax = clip_limits(data, pct)

    ax.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[line.dist[0], line.dist[-1], tt[-1], tt[0]],
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("Distance along line [m]", fontsize=8)
    ax.set_ylabel("Two-way time [ns]", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    # Fills the canvas better while leaving enough room for labels.
    fig.subplots_adjust(left=0.085, right=0.990, bottom=0.125, top=0.885)
    canvas.draw_idle()


def _pe_filled_fast_plot(self):
    line = self.line
    _pe_filled_fast_make_side_layout(self)

    if line.raw is None:
        try:
            self.placeholder()
        except Exception:
            pass
        return

    _pe_filled_fast_plot_one(self.raw_canvas, line, self.owner, line.raw, f"Raw — {line.name}", processed=False)
    _pe_filled_fast_plot_one(self.proc_canvas, line, self.owner, line.proc, f"Processed — {line.name}", processed=True)


def _pe_filled_fast_apply_all(project_tab):
    try:
        combo = getattr(project_tab, "radargram_width_mode", None)
        if combo is not None and combo.findText("Compact") >= 0:
            # Keep requested default visible to user, but Compact now expands cleanly into each half-panel.
            combo.setCurrentText("Compact")

        widgets = []
        if hasattr(project_tab, "line_widgets"):
            widgets.extend(list(project_tab.line_widgets.values()))
        if hasattr(project_tab, "inline_tabs"):
            for i in range(project_tab.inline_tabs.count()):
                widgets.append(project_tab.inline_tabs.widget(i))

        seen = set()
        for w in widgets:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            if isinstance(w, RadargramPair):
                _pe_filled_fast_make_side_layout(w)
                try:
                    if getattr(w.line, "raw", None) is not None:
                        _pe_filled_fast_plot(w)
                except Exception:
                    pass

        try:
            project_tab.status.setText("Bulach radargrams: compact side-by-side filled view.")
        except Exception:
            pass
    except Exception as e:
        print("Bulach filled/fast apply-all failed:", e)


try:
    # Final plot override: avoids older stacked wrappers and white/fixed-canvas behaviour.
    RadargramPair.plot = _pe_filled_fast_plot

    if not hasattr(RadargramPair, "_old_init_for_filled_fast"):
        RadargramPair._old_init_for_filled_fast = RadargramPair.__init__
        def _radargram_pair_init_filled_fast(self, *args, **kwargs):
            RadargramPair._old_init_for_filled_fast(self, *args, **kwargs)
            _pe_filled_fast_make_side_layout(self)
        RadargramPair.__init__ = _radargram_pair_init_filled_fast

    if not hasattr(PulseEkkoProjectTab, "_old_init_for_filled_fast"):
        PulseEkkoProjectTab._old_init_for_filled_fast = PulseEkkoProjectTab.__init__
        def _pe_project_init_filled_fast(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_filled_fast(self, *args, **kwargs)
            _pe_filled_fast_apply_all(self)
        PulseEkkoProjectTab.__init__ = _pe_project_init_filled_fast

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_filled_fast"):
        PulseEkkoProjectTab._old_rebuild_for_filled_fast = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_filled_fast(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_filled_fast(self, *args, **kwargs)
            _pe_filled_fast_apply_all(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_filled_fast

    if not hasattr(PulseEkkoProjectTab, "_old_process_for_filled_fast"):
        PulseEkkoProjectTab._old_process_for_filled_fast = PulseEkkoProjectTab.process_current_line
        def _pe_process_filled_fast(self, *args, **kwargs):
            # Width-combo signals send a string argument. Do not reprocess data for that.
            if args and isinstance(args[0], str):
                _pe_filled_fast_apply_all(self)
                return None
            out = PulseEkkoProjectTab._old_process_for_filled_fast(self)
            _pe_filled_fast_apply_all(self)
            return out
        PulseEkkoProjectTab.process_current_line = _pe_process_filled_fast

    print("Bulach filled fast radargram layout patch active.")
except Exception as e:
    print("Bulach filled fast radargram layout patch failed:", e)
# --- end Bulach filled fast radargram layout patch ---


# --- Clean Bulach radargram implementation using Schleitheim-style direct canvases ---
def _pe_clean_radar_pair_init(self, line, owner):
    from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QSizePolicy
    from PyQt6.QtCore import Qt

    QWidget.__init__(self)
    self.line = line
    self.owner = owner

    root = QHBoxLayout(self)
    root.setContentsMargins(4, 4, 4, 4)
    root.setSpacing(6)

    self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
    self.splitter.setChildrenCollapsible(False)

    self.raw_canvas = Canvas(8.0, 4.6)
    self.proc_canvas = Canvas(8.0, 4.6)

    for canvas in (self.raw_canvas, self.proc_canvas):
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setMinimumSize(420, 300)

    raw_box = QWidget(self)
    raw_lay = QVBoxLayout(raw_box)
    raw_lay.setContentsMargins(0, 0, 0, 0)
    raw_lay.addWidget(self.raw_canvas, 1)

    proc_box = QWidget(self)
    proc_lay = QVBoxLayout(proc_box)
    proc_lay.setContentsMargins(0, 0, 0, 0)
    proc_lay.addWidget(self.proc_canvas, 1)

    self.splitter.addWidget(raw_box)
    self.splitter.addWidget(proc_box)
    self.splitter.setStretchFactor(0, 1)
    self.splitter.setStretchFactor(1, 1)

    root.addWidget(self.splitter, 1)
    self.placeholder()


def _pe_clean_radar_placeholder(self):
    for canvas, text in [
        (self.raw_canvas, "Click 'Load raw'"),
        (self.proc_canvas, "Click 'Process current line'"),
    ]:
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        canvas.draw_idle()


def _pe_clean_plot_one(canvas, line, owner, arr, title, empty_text):
    import numpy as np

    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)
    canvas.ax = ax

    if arr is None:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        canvas.fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.90)
        canvas.draw_idle()
        return

    t = owner.corrected_time_ns(line)
    tmin = float(owner.display_min.value())
    tmax = float(owner.display_max.value())
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    data = arr[:, mask]
    tt = t[mask]
    vmin, vmax = clip_limits(data, float(owner.display_clip.value()))

    ax.imshow(
        data.T,
        aspect="auto",
        cmap=owner.cmap.currentText(),
        vmin=vmin,
        vmax=vmax,
        extent=[float(line.dist[0]), float(line.dist[-1]), float(tt[-1]), float(tt[0])],
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Distance along line [m]", fontsize=9)
    ax.set_ylabel("Two-way time [ns]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    canvas.fig.subplots_adjust(left=0.085, right=0.990, bottom=0.135, top=0.885)
    canvas.draw_idle()


def _pe_clean_radar_plot(self):
    line = self.line
    _pe_clean_plot_one(self.raw_canvas, line, self.owner, line.raw, f"Raw — {line.name}", "Click 'Load raw'")
    _pe_clean_plot_one(self.proc_canvas, line, self.owner, line.proc, f"Processed — {line.name}", "Click 'Process current line'")
    try:
        self.splitter.setSizes([1, 1])
    except Exception:
        pass


def _pe_clean_add_width_control(project_tab):
    try:
        if hasattr(project_tab, "radargram_width_mode"):
            if project_tab.radargram_width_mode.findText("Compact") >= 0:
                project_tab.radargram_width_mode.setCurrentText("Compact")
            return

        from PyQt6.QtWidgets import QLabel, QComboBox

        project_tab.radargram_width_mode = QComboBox()
        project_tab.radargram_width_mode.addItems(["Compact", "Fit to screen"])
        project_tab.radargram_width_mode.setCurrentText("Compact")
        project_tab.radargram_width_mode.setToolTip("Compact is the default clean side-by-side view.")

        grid = None
        try:
            lay = project_tab.inline_widget.layout()
            item = lay.itemAt(0) if lay is not None else None
            grid = item.layout() if item is not None else None
        except Exception:
            grid = None

        if grid is not None:
            grid.addWidget(QLabel("Radargram width"), 5, 0)
            grid.addWidget(project_tab.radargram_width_mode, 5, 1, 1, 2)

        try:
            project_tab.radargram_width_mode.currentTextChanged.connect(lambda *_: _pe_clean_redraw_current(project_tab))
        except Exception:
            pass
    except Exception as e:
        print("Could not add clean Bulach width control:", e)


def _pe_clean_redraw_current(project_tab):
    try:
        line = project_tab.current_line()
        if line is not None and line.idx in project_tab.line_widgets:
            project_tab.line_widgets[line.idx].plot()
            project_tab.status.setText("Bulach radargrams: clean compact side-by-side view.")
    except Exception as e:
        print("Clean Bulach redraw failed:", e)


def _pe_clean_rebuild_inline_tabs(self):
    self.inline_tabs.clear()
    self.line_widgets = {}
    for line in self.lines:
        pair = RadargramPair(line, self)
        self.line_widgets[line.idx] = pair
        self.inline_tabs.addTab(pair, line.name)
    try:
        self.tabs.setTabText(1, f"Inline ({len(self.lines)})")
    except Exception:
        pass
    _pe_clean_add_width_control(self)


def _pe_clean_load_current_raw(self, *args, **kwargs):
    line = self.current_line()
    if line is None:
        return
    try:
        self.ensure_raw(line)
        self.line_widgets[line.idx].plot()
        self.status.setText(f"Loaded raw {line.name}: {line.raw.shape}; estimated T0={self.estimate_t0_ns(line):.2f} ns")
    except Exception as e:
        QMessageBox.critical(self, "Load raw failed", str(e))


def _pe_clean_process_current_line(self, *args, **kwargs):
    # Width combo signals pass a text argument; do not use that as a processing argument.
    line = self.current_line()
    if line is None:
        return
    try:
        self.ensure_processed(line)
        self.line_widgets[line.idx].plot()
        self.status.setText(f"Processed {line.name}: {line.proc.shape}; estimated T0={self.estimate_t0_ns(line):.2f} ns")
    except Exception as e:
        QMessageBox.critical(self, "Processing failed", str(e))


def _pe_clean_project_init(self, root):
    QWidget.__init__(self)
    self.root = Path(root)
    self.lines = []
    self.line_widgets = {}
    self.build_ui()
    _pe_clean_add_width_control(self)
    self.reload_project()


try:
    RadargramPair.__init__ = _pe_clean_radar_pair_init
    RadargramPair.placeholder = _pe_clean_radar_placeholder
    RadargramPair.plot = _pe_clean_radar_plot

    PulseEkkoProjectTab.__init__ = _pe_clean_project_init
    PulseEkkoProjectTab.rebuild_inline_tabs = _pe_clean_rebuild_inline_tabs
    PulseEkkoProjectTab.load_current_raw = _pe_clean_load_current_raw
    PulseEkkoProjectTab.process_current_line = _pe_clean_process_current_line

    print("Clean Bulach radargram implementation active: direct side-by-side canvases, no stacked layout wrappers.")
except Exception as e:
    print("Clean Bulach radargram implementation patch failed:", e)
# --- end Clean Bulach radargram implementation ---


# --- Ringing suppression + NMO correction features for Bulach/PulseEKKO ---
def _pe_rs_early_mute_taper(data, time_ns, mute_end_ns, taper_ns):
    import numpy as np
    out = np.asarray(data, dtype=np.float64).copy()
    t = np.asarray(time_ns, dtype=float)
    if out.ndim != 2 or t.size != out.shape[1] or mute_end_ns <= 0:
        return out
    mute_end = float(mute_end_ns)
    taper = max(0.0, float(taper_ns))
    w = np.ones(t.size, dtype=float)
    if taper <= 0:
        w[t < mute_end] = 0.0
    else:
        flat_end = max(0.0, mute_end - taper)
        w[t < flat_end] = 0.0
        m = (t >= flat_end) & (t < mute_end)
        if np.any(m):
            x = (t[m] - flat_end) / max(taper, 1e-9)
            w[m] = 0.5 - 0.5 * np.cos(np.pi * x)
    return out * w[None, :]


def _pe_rs_predictive_decon(data, time_ns, lag_ns, op_ns=10.0, prewhite_pct=0.5):
    import numpy as np
    x = np.asarray(data, dtype=np.float64)
    t = np.asarray(time_ns, dtype=float)
    if x.ndim != 2 or t.size < 3 or t.size != x.shape[1]:
        return x.copy()
    dt = float(np.nanmedian(np.diff(t)))
    if dt <= 0:
        return x.copy()
    lag = max(1, int(round(float(lag_ns) / dt)))
    op = max(lag + 1, int(round(float(op_ns) / dt)))
    ns = x.shape[1]
    if lag >= ns - 2:
        return x.copy()
    out = x.copy()
    end = min(ns, lag + op)
    a = x[:, lag:end]
    b = x[:, :end-lag]
    denom = np.sum(b * b, axis=1) + (float(prewhite_pct) / 100.0) * np.sum(x * x, axis=1) + 1e-12
    alpha = np.sum(a * b, axis=1) / denom
    alpha = np.clip(alpha, -0.95, 0.95)
    out[:, lag:] = x[:, lag:] - alpha[:, None] * x[:, :-lag]
    return out


def _pe_rs_nmo_zero_offset(data, time_ns, offset_m, velocity_m_ns):
    import numpy as np
    x = np.asarray(data, dtype=np.float64)
    t = np.asarray(time_ns, dtype=float)
    if x.ndim != 2 or t.size != x.shape[1] or offset_m <= 0 or velocity_m_ns <= 0:
        return x.copy()
    tau = t.copy()
    t_offset = float(offset_m) / max(float(velocity_m_ns), 1e-9)
    t_meas = np.sqrt(tau * tau + t_offset * t_offset)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i, :] = np.interp(t_meas, t, x[i, :], left=0.0, right=0.0)
    return out


def _pe_rs_tpower_gain(data, time_ns, sec_power):
    import numpy as np
    out = np.asarray(data, dtype=np.float64).copy()
    t = np.asarray(time_ns, dtype=float)
    if out.ndim != 2 or t.size != out.shape[1] or sec_power <= 0:
        return out
    tmax_gain = 30.0
    tt = np.minimum(np.maximum(t, 0.0), tmax_gain)
    gain = (1.0 + tt / max(tmax_gain, 1e-9)) ** float(sec_power)
    gain /= max(float(gain[0]), 1e-12)
    return out * gain[None, :]


def _pe_rs_agc_gain(data, time_ns, window_ns):
    import numpy as np
    x = np.asarray(data, dtype=np.float64)
    t = np.asarray(time_ns, dtype=float)
    if x.ndim != 2 or t.size < 3 or t.size != x.shape[1] or window_ns <= 0:
        return x.copy()
    dt = float(np.nanmedian(np.diff(t)))
    if dt <= 0:
        return x.copy()
    ns = x.shape[1]
    w = max(3, int(round(float(window_ns) / dt)))
    w = min(w, ns)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    rms = np.sqrt(np.apply_along_axis(lambda tr: np.convolve(tr * tr, kernel, mode="same"), 1, x))
    if rms.shape != x.shape:
        rms = rms[:, :ns]
    finite = rms[np.isfinite(rms) & (rms > 0)]
    if finite.size == 0:
        return x.copy()
    scale = np.nanmedian(finite)
    floor = np.nanpercentile(finite, 35)
    return x * scale / np.maximum(rms, floor)


def _pe_clear_proc_cache(project_tab):
    try:
        for ln in getattr(project_tab, "lines", []):
            ln.proc = None
        if hasattr(project_tab, "status"):
            project_tab.status.setText("Processing settings changed; processed cache cleared.")
    except Exception:
        pass


def _pe_add_ringing_nmo_controls(project_tab):
    try:
        if getattr(project_tab, "_ringing_nmo_controls_added", False):
            return

        from PyQt6.QtWidgets import QLabel, QCheckBox, QDoubleSpinBox

        def dspin(value, lo, hi, step, suffix):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setValue(value)
            w.setSingleStep(step)
            w.setSuffix(suffix)
            return w

        project_tab.ring_mute_enabled = QCheckBox("Ringing mute/taper")
        project_tab.ring_mute_enabled.setChecked(True)
        project_tab.ring_mute_end_ns = dspin(5.0, 0.0, 100.0, 0.5, " ns")
        project_tab.ring_mute_taper_ns = dspin(2.0, 0.0, 50.0, 0.5, " ns")

        project_tab.nmo_enabled = QCheckBox("NMO zero-offset")
        project_tab.nmo_enabled.setChecked(True)
        project_tab.nmo_offset_m = dspin(0.35, 0.0, 5.0, 0.01, " m")
        project_tab.nmo_velocity_m_ns = dspin(0.10, 0.02, 0.30, 0.005, " m/ns")

        project_tab.pred_decon_enabled = QCheckBox("Predictive decon")
        project_tab.pred_decon_enabled.setChecked(False)
        project_tab.pred_decon_lag_ns = dspin(2.0, 0.1, 50.0, 0.5, " ns")
        project_tab.pred_decon_op_ns = dspin(10.0, 1.0, 100.0, 1.0, " ns")
        project_tab.pred_decon_white_pct = dspin(0.50, 0.0, 10.0, 0.1, " %")

        grid = None
        try:
            lay = project_tab.inline_widget.layout()
            item = lay.itemAt(0) if lay is not None else None
            grid = item.layout() if item is not None else None
        except Exception:
            grid = None

        if grid is not None:
            row = 6
            grid.addWidget(project_tab.ring_mute_enabled, row, 0)
            grid.addWidget(QLabel("Mute end"), row, 1)
            grid.addWidget(project_tab.ring_mute_end_ns, row, 2)
            grid.addWidget(QLabel("Taper"), row, 3)
            grid.addWidget(project_tab.ring_mute_taper_ns, row, 4)

            row += 1
            grid.addWidget(project_tab.nmo_enabled, row, 0)
            grid.addWidget(QLabel("Offset"), row, 1)
            grid.addWidget(project_tab.nmo_offset_m, row, 2)
            grid.addWidget(QLabel("Velocity"), row, 3)
            grid.addWidget(project_tab.nmo_velocity_m_ns, row, 4)

            row += 1
            grid.addWidget(project_tab.pred_decon_enabled, row, 0)
            grid.addWidget(QLabel("Lag"), row, 1)
            grid.addWidget(project_tab.pred_decon_lag_ns, row, 2)
            grid.addWidget(QLabel("Op"), row, 3)
            grid.addWidget(project_tab.pred_decon_op_ns, row, 4)
            grid.addWidget(QLabel("White"), row, 5)
            grid.addWidget(project_tab.pred_decon_white_pct, row, 6)

        for w in [project_tab.ring_mute_enabled, project_tab.nmo_enabled, project_tab.pred_decon_enabled]:
            try:
                w.toggled.connect(lambda *_, obj=project_tab: _pe_clear_proc_cache(obj))
            except Exception:
                pass
        for w in [
            project_tab.ring_mute_end_ns, project_tab.ring_mute_taper_ns,
            project_tab.nmo_offset_m, project_tab.nmo_velocity_m_ns,
            project_tab.pred_decon_lag_ns, project_tab.pred_decon_op_ns, project_tab.pred_decon_white_pct
        ]:
            try:
                w.valueChanged.connect(lambda *_, obj=project_tab: _pe_clear_proc_cache(obj))
            except Exception:
                pass

        project_tab._ringing_nmo_controls_added = True
    except Exception as e:
        print("Could not add Bulach ringing/NMO controls:", e)


def _pe_process_with_ringing_nmo(project_tab, line):
    project_tab.ensure_raw(line)
    gain_on = getattr(project_tab, "sec_gain", None) is not None and project_tab.sec_gain.isChecked()
    gain_power = float(project_tab.sec_power.value()) if gain_on else 0.0

    parts = ["DC removal", "dewow"]
    if project_tab.bg_remove.isChecked():
        parts.append("background removal")
    if project_tab.bandpass.isChecked():
        parts.append("bandpass")

    out = process_gpr(
        line.raw,
        line.time_ns,
        dewow_window_ns=project_tab.dewow_window.value(),
        bg_window=project_tab.bg_window.value(),
        do_bg=project_tab.bg_remove.isChecked(),
        do_bp=project_tab.bandpass.isChecked(),
        low_mhz=project_tab.low_cut.value(),
        high_mhz=project_tab.high_cut.value(),
        sec_power=0.0,
        do_agc=False,
        agc_window_ns=project_tab.agc_window.value(),
    )

    if getattr(project_tab, "ring_mute_enabled", None) is not None and project_tab.ring_mute_enabled.isChecked():
        out = _pe_rs_early_mute_taper(out, line.time_ns, project_tab.ring_mute_end_ns.value(), project_tab.ring_mute_taper_ns.value())
        parts.append(f"early mute/taper ({project_tab.ring_mute_end_ns.value():.1f} ns)")

    if getattr(project_tab, "pred_decon_enabled", None) is not None and project_tab.pred_decon_enabled.isChecked():
        out = _pe_rs_predictive_decon(
            out, line.time_ns,
            project_tab.pred_decon_lag_ns.value(),
            project_tab.pred_decon_op_ns.value(),
            project_tab.pred_decon_white_pct.value(),
        )
        parts.append("predictive decon")

    if getattr(project_tab, "nmo_enabled", None) is not None and project_tab.nmo_enabled.isChecked():
        out = _pe_rs_nmo_zero_offset(out, line.time_ns, project_tab.nmo_offset_m.value(), project_tab.nmo_velocity_m_ns.value())
        parts.append(f"NMO zero-offset ({project_tab.nmo_offset_m.value():.2f} m)")

    if gain_on and gain_power > 0:
        out = _pe_rs_tpower_gain(out, line.time_ns, gain_power)
        parts.append(f"T-power gain ({gain_power:.2f})")

    if project_tab.agc_gain.isChecked():
        out = _pe_rs_agc_gain(out, line.time_ns, project_tab.agc_window.value())
        parts.append("AGC gain")

    line.proc_label = "Processed radargram — " + line.name + " | " + " + ".join(parts)
    line.proc = out
    return line.proc


try:
    if not hasattr(PulseEkkoProjectTab, "_old_init_for_ringing_nmo"):
        PulseEkkoProjectTab._old_init_for_ringing_nmo = PulseEkkoProjectTab.__init__
        def _pe_init_ringing_nmo(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_ringing_nmo(self, *args, **kwargs)
            _pe_add_ringing_nmo_controls(self)
        PulseEkkoProjectTab.__init__ = _pe_init_ringing_nmo

    def _pe_ensure_processed_ringing_nmo(self, line):
        return _pe_process_with_ringing_nmo(self, line)
    PulseEkkoProjectTab.ensure_processed = _pe_ensure_processed_ringing_nmo

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_ringing_nmo"):
        PulseEkkoProjectTab._old_rebuild_for_ringing_nmo = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_ringing_nmo(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_ringing_nmo(self, *args, **kwargs)
            _pe_add_ringing_nmo_controls(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_ringing_nmo

    if hasattr(PulseEkko3DAnalysis, "get_array"):
        PulseEkko3DAnalysis.get_array = PulseEkko3DAnalysis.get_array

    print("Ringing suppression + NMO correction controls active for Bulach/PulseEKKO.")
except Exception as e:
    print("Bulach ringing/NMO patch failed:", e)
# --- end ringing suppression + NMO correction features for Bulach/PulseEKKO ---


# --- Safe default ringing/NMO settings patch for Bulach/PulseEKKO ---
def _pe_apply_safe_ringing_nmo_defaults(project_tab):
    try:
        # Keep shallow PulseEKKO response visible by default.
        if hasattr(project_tab, "ring_mute_enabled"):
            project_tab.ring_mute_enabled.setChecked(False)
        if hasattr(project_tab, "ring_mute_end_ns"):
            project_tab.ring_mute_end_ns.setValue(5.0)
        if hasattr(project_tab, "ring_mute_taper_ns"):
            project_tab.ring_mute_taper_ns.setValue(2.0)

        if hasattr(project_tab, "nmo_enabled"):
            project_tab.nmo_enabled.setChecked(True)
        if hasattr(project_tab, "nmo_offset_m"):
            project_tab.nmo_offset_m.setValue(0.35)
        if hasattr(project_tab, "nmo_velocity_m_ns"):
            project_tab.nmo_velocity_m_ns.setValue(0.10)

        if hasattr(project_tab, "pred_decon_enabled"):
            project_tab.pred_decon_enabled.setChecked(False)
        if hasattr(project_tab, "pred_decon_lag_ns"):
            project_tab.pred_decon_lag_ns.setValue(2.0)
        if hasattr(project_tab, "pred_decon_op_ns"):
            project_tab.pred_decon_op_ns.setValue(10.0)
        if hasattr(project_tab, "pred_decon_white_pct"):
            project_tab.pred_decon_white_pct.setValue(0.50)

        # Keep the current good shallow visibility defaults.
        if hasattr(project_tab, "agc_gain"):
            project_tab.agc_gain.setChecked(False)
        if hasattr(project_tab, "sec_gain"):
            project_tab.sec_gain.setChecked(False)

        for ln in getattr(project_tab, "lines", []):
            try:
                ln.proc = None
            except Exception:
                pass
        try:
            project_tab.status.setText("Default processing: shallow visible, NMO ON, ringing mute OFF.")
        except Exception:
            pass
    except Exception as e:
        print("Could not apply Bulach safe ringing/NMO defaults:", e)


try:
    if not hasattr(PulseEkkoProjectTab, "_old_init_for_safe_ringing_defaults"):
        PulseEkkoProjectTab._old_init_for_safe_ringing_defaults = PulseEkkoProjectTab.__init__
        def _pe_init_safe_ringing_defaults(self, *args, **kwargs):
            PulseEkkoProjectTab._old_init_for_safe_ringing_defaults(self, *args, **kwargs)
            _pe_apply_safe_ringing_nmo_defaults(self)
        PulseEkkoProjectTab.__init__ = _pe_init_safe_ringing_defaults

    if not hasattr(PulseEkkoProjectTab, "_old_rebuild_for_safe_ringing_defaults"):
        PulseEkkoProjectTab._old_rebuild_for_safe_ringing_defaults = PulseEkkoProjectTab.rebuild_inline_tabs
        def _pe_rebuild_safe_ringing_defaults(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_rebuild_for_safe_ringing_defaults(self, *args, **kwargs)
            _pe_apply_safe_ringing_nmo_defaults(self)
            return out
        PulseEkkoProjectTab.rebuild_inline_tabs = _pe_rebuild_safe_ringing_defaults

    print("Safe default ringing/NMO settings active for Bulach/PulseEKKO.")
except Exception as e:
    print("Safe default ringing/NMO settings patch failed for Bulach:", e)
# --- end safe default ringing/NMO settings patch for Bulach/PulseEKKO ---


# --- Bulach radargram title correction patch ---
def _pe_short_proc_parts(owner):
    parts = ["DC", "dewow"]
    try:
        if owner.bg_remove.isChecked():
            parts.append("bg")
    except Exception:
        pass
    try:
        if owner.bandpass.isChecked():
            parts.append("BP")
    except Exception:
        pass
    try:
        if getattr(owner, "ring_mute_enabled", None) is not None and owner.ring_mute_enabled.isChecked():
            parts.append(f"mute {owner.ring_mute_end_ns.value():.1f} ns")
    except Exception:
        pass
    try:
        if getattr(owner, "nmo_enabled", None) is not None and owner.nmo_enabled.isChecked():
            parts.append("NMO")
    except Exception:
        pass
    try:
        if getattr(owner, "pred_decon_enabled", None) is not None and owner.pred_decon_enabled.isChecked():
            parts.append("pred-decon")
    except Exception:
        pass
    try:
        if getattr(owner, "sec_gain", None) is not None and owner.sec_gain.isChecked():
            parts.append("T-power")
    except Exception:
        pass
    try:
        if owner.agc_gain.isChecked():
            parts.append("AGC")
    except Exception:
        pass
    return " + ".join(parts)


def _pe_title_plot_one(canvas, line, owner, arr, title, empty_text):
    import numpy as np

    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)
    canvas.ax = ax

    if arr is None:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        canvas.fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.90)
        canvas.draw_idle()
        return

    t = owner.corrected_time_ns(line)
    tmin = float(owner.display_min.value())
    tmax = float(owner.display_max.value())
    mask = (t >= tmin) & (t <= tmax)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    data = arr[:, mask]
    tt = t[mask]
    vmin, vmax = clip_limits(data, float(owner.display_clip.value()))

    ax.imshow(
        data.T,
        aspect="auto",
        cmap=owner.cmap.currentText(),
        vmin=vmin,
        vmax=vmax,
        extent=[float(line.dist[0]), float(line.dist[-1]), float(tt[-1]), float(tt[0])],
        interpolation="bilinear",
    )
    if len(title) > 95:
        title = title[:92] + "..."
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel("Distance along line [m]", fontsize=9)
    ax.set_ylabel("Two-way time [ns]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    canvas.fig.subplots_adjust(left=0.085, right=0.990, bottom=0.135, top=0.875)
    canvas.draw_idle()


def _pe_title_radar_plot(self):
    line = self.line
    try:
        if "_pe_clean_plot_one" in globals():
            # Keep the clean direct-canvas layout but replace titles.
            pass
    except Exception:
        pass
    _pe_title_plot_one(self.raw_canvas, line, self.owner, line.raw, f"Raw radargram — {line.name}", "Click 'Load raw'")
    _pe_title_plot_one(self.proc_canvas, line, self.owner, line.proc, f"Processed radargram — {line.name} | {_pe_short_proc_parts(self.owner)}", "Click 'Process current line'")
    try:
        if hasattr(self, "splitter"):
            self.splitter.setSizes([1, 1])
        elif hasattr(self, "_pe_filled_fast_splitter"):
            self._pe_filled_fast_splitter.setSizes([1, 1])
    except Exception:
        pass


try:
    RadargramPair.plot = _pe_title_radar_plot
    print("Bulach radargram title correction active.")
except Exception as e:
    print("Bulach radargram title correction patch failed:", e)
# --- end Bulach radargram title correction patch ---


# --- Bulach save current radargram view patch ---
def _pe_save_safe_name(text):
    import re
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "radargram"


def _pe_current_radargram_pair(owner):
    try:
        tab = owner.inline_tabs.currentWidget()
        if isinstance(tab, RadargramPair):
            return tab
    except Exception:
        pass
    return None


def _pe_save_current_radargram_view(owner):
    try:
        from pathlib import Path
        import datetime
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox

        pair = _pe_current_radargram_pair(owner)
        if pair is None:
            raise RuntimeError("Open a Bulach line tab first.")

        line = getattr(pair, "line", None)
        if line is None:
            raise RuntimeError("No active Bulach line found.")

        out_dir = Path(getattr(owner, "root", Path("."))) / "exports" / "current_views"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        line_name = _pe_save_safe_name(getattr(line, "name", "radargram"))
        base = f"{line_name}_current_view_{ts}"

        raw_png = out_dir / f"{base}_raw.png"
        proc_png = out_dir / f"{base}_processed.png"
        combined_png = out_dir / f"{base}_raw_plus_processed.png"

        # Clean Matplotlib exports. These preserve the current axis limits / time window.
        saved_any = False
        try:
            pair.raw_canvas.fig.savefig(raw_png, dpi=250, bbox_inches="tight", facecolor="white")
            saved_any = True
        except Exception as e:
            print("Could not save Bulach raw current-view figure:", e)

        try:
            pair.proc_canvas.fig.savefig(proc_png, dpi=250, bbox_inches="tight", facecolor="white")
            saved_any = True
        except Exception as e:
            print("Could not save Bulach processed current-view figure:", e)

        # Literal current GUI view, raw left and processed right, matching the side-by-side display.
        raw_pm = pair.raw_canvas.grab()
        proc_pm = pair.proc_canvas.grab()

        width = raw_pm.width() + proc_pm.width()
        height = max(raw_pm.height(), proc_pm.height())
        if width <= 0 or height <= 0:
            raise RuntimeError("Current radargram canvas has invalid size.")

        img = QImage(width, height, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.white)

        painter = QPainter(img)
        painter.drawPixmap(0, 0, raw_pm)
        painter.drawPixmap(raw_pm.width(), 0, proc_pm)
        painter.end()

        if not img.save(str(combined_png)):
            raise RuntimeError(f"Could not save {combined_png}")

        saved_any = True
        try:
            owner.status.setText(f"Saved Bulach current radargram view: {combined_png}")
        except Exception:
            pass

        if saved_any:
            QMessageBox.information(owner, "Bulach current view saved", f"Saved:\n{combined_png}\n\nAlso saved:\n{raw_png}\n{proc_png}")

        print(f"Saved Bulach current radargram view: {combined_png}")
        print(f"Saved Bulach raw: {raw_png}")
        print(f"Saved Bulach processed: {proc_png}")

    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(owner, "Save Bulach current view failed", str(e))
        except Exception:
            print("Save Bulach current view failed:", e)


def _pe_add_save_current_view_button(owner):
    try:
        if getattr(owner, "_pe_save_current_view_button_added", False):
            return

        owner.btn_save_current_view = QPushButton("Save current view")
        owner.btn_save_current_view.setToolTip(
            "Save the currently displayed Bulach raw + processed radargram view. "
            "The saved PNG uses the current line, display width, time window and zoom."
        )
        owner.btn_save_current_view.clicked.connect(lambda *_: _pe_save_current_radargram_view(owner))

        layout = getattr(owner, "inline_widget", None).layout()
        if layout is None:
            raise RuntimeError("Could not find Bulach inline layout.")

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(owner.btn_save_current_view)

        # Insert between the processing controls and the line tabs.
        try:
            layout.insertLayout(1, row)
        except Exception:
            layout.addLayout(row)

        owner._pe_save_current_view_button_added = True
    except Exception as e:
        print("Could not add Bulach Save current view button:", e)


try:
    if not hasattr(PulseEkkoProjectTab, "_old_build_inline_widget_for_save_current_view"):
        PulseEkkoProjectTab._old_build_inline_widget_for_save_current_view = PulseEkkoProjectTab.build_inline_widget

        def _pe_build_inline_widget_with_save_current_view(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_build_inline_widget_for_save_current_view(self, *args, **kwargs)
            try:
                _pe_add_save_current_view_button(self)
            except Exception as e:
                print("Could not add Bulach Save current view button after build:", e)
            return out

        PulseEkkoProjectTab.build_inline_widget = _pe_build_inline_widget_with_save_current_view

    if not hasattr(PulseEkkoProjectTab, "_old_reload_project_for_save_current_view"):
        PulseEkkoProjectTab._old_reload_project_for_save_current_view = PulseEkkoProjectTab.reload_project

        def _pe_reload_project_with_save_current_view(self, *args, **kwargs):
            out = PulseEkkoProjectTab._old_reload_project_for_save_current_view(self, *args, **kwargs)
            try:
                _pe_add_save_current_view_button(self)
            except Exception:
                pass
            return out

        PulseEkkoProjectTab.reload_project = _pe_reload_project_with_save_current_view

    print("Bulach save current radargram view patch active.")
except Exception as e:
    print("Bulach save current radargram view patch failed:", e)
# --- end Bulach save current radargram view patch ---


# --- Bulach amplitude projection centered HD patch ---
def _bulach_amp_status(obj, msg):
    try:
        obj.owner.status.setText(str(msg))
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass


def _bulach_amp_make_scrollable(self):
    """Make the Bulach amplitude-projection canvas large/scrollable, like the HD time-slice view."""
    try:
        if getattr(self, "_bulach_amp_scrollable_added", False):
            return

        from PyQt6.QtWidgets import QScrollArea
        from PyQt6.QtCore import Qt

        canvas = self.canvases.get("Amplitude Projection")
        if canvas is None:
            return

        # Larger canvas gives enough vertical pixels for an equal-scale map instead of a tiny map
        # squeezed into the wide tab area.
        try:
            canvas.fig.set_dpi(120)
            canvas.fig.set_size_inches(15.5, 10.0, forward=True)
            canvas.setMinimumSize(1550, 1000)
            canvas.resize(1550, 1000)
        except Exception:
            pass

        idx = None
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Amplitude Projection":
                idx = i
                break
        if idx is None:
            return

        old_widget = self.tabs.widget(idx)
        if isinstance(old_widget, QScrollArea):
            self._bulach_amp_scrollable_added = True
            return

        self.tabs.removeTab(idx)
        scroll = QScrollArea(self.tabs)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(canvas)
        self.tabs.insertTab(idx, scroll, "Amplitude Projection")
        self.canvases["Amplitude Projection"] = canvas
        self._bulach_amp_scrollable_added = True

    except Exception as e:
        print("Bulach amplitude-projection scrollable setup failed:", e)


def _bulach_amp_axes_rect(fig, x, y):
    """Return centred equal-scale map axes + colourbar axes rectangles."""
    import numpy as _np

    W, H = fig.get_size_inches()
    fig_ratio = max(float(W) / max(float(H), 1e-9), 1e-9)

    xr = float(_np.nanmax(x) - _np.nanmin(x))
    yr = float(_np.nanmax(y) - _np.nanmin(y))
    data_ratio = yr / max(xr, 1e-9)  # physical height / width required

    avail_left, avail_right = 0.06, 0.86
    avail_bottom, avail_top = 0.10, 0.90
    avail_w = avail_right - avail_left
    avail_h = avail_top - avail_bottom

    # For equal aspect: axis_h_fraction = axis_w_fraction * data_ratio * fig_ratio
    required_h = avail_w * data_ratio * fig_ratio

    if required_h <= avail_h:
        ax_w = avail_w
        ax_h = required_h
    else:
        ax_h = avail_h
        ax_w = ax_h / max(data_ratio * fig_ratio, 1e-9)

    ax_left = avail_left + (avail_w - ax_w) / 2.0
    ax_bottom = avail_bottom + (avail_h - ax_h) / 2.0

    cbar_left = min(ax_left + ax_w + 0.018, 0.91)
    cbar = [cbar_left, ax_bottom, 0.018, ax_h]
    return [ax_left, ax_bottom, ax_w, ax_h], cbar


def _bulach_amp_plot_hd(self, canvas):
    """Centred, equal-scale, HD Bulach amplitude-projection plot."""
    import numpy as _np
    import datetime as _datetime

    title = f"Amplitude projection {self.proj_tmin.value():.0f}–{self.proj_tmax.value():.0f} ns"
    lines = self.selected_lines()
    total = len(lines) + 2

    dlg, start = _pe_make_progress(self, f"Bulach HD amplitude projection: {title}", total)
    log = []
    try:
        _pe_update_progress(dlg, start, 0, total, "Collecting line-normalised amplitude values...")
        try:
            x, y, v = self.collect_values("projection", dlg, start, log)
        except TypeError:
            x, y, v = self.collect_values("projection")

        _pe_update_progress(dlg, start, len(lines), total, "Interpolating HD equal-scale map...")

        canvas.fig.clear()
        try:
            canvas.fig.set_dpi(120)
            canvas.fig.set_size_inches(15.5, 10.0, forward=True)
            canvas.setMinimumSize(1550, 1000)
            canvas.resize(1550, 1000)
        except Exception:
            pass

        fig = canvas.fig

        if len(v) == 0:
            ax = fig.add_axes([0.10, 0.12, 0.80, 0.78])
            canvas.ax = ax
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            canvas.draw()
            dlg.close()
            return

        finite = _np.isfinite(x) & _np.isfinite(y) & _np.isfinite(v)
        x, y, v = x[finite], y[finite], v[finite]
        if len(v) == 0:
            ax = fig.add_axes([0.10, 0.12, 0.80, 0.78])
            canvas.ax = ax
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            canvas.draw()
            dlg.close()
            return

        vmin = float(_np.nanpercentile(v, 2))
        vmax = float(_np.nanpercentile(v, 98))
        if not _np.isfinite(vmin) or not _np.isfinite(vmax) or vmax <= vmin:
            vmin = float(_np.nanmin(v))
            vmax = float(_np.nanmax(v) + 1e-9)

        xr = float(_np.nanmax(x) - _np.nanmin(x))
        yr = float(_np.nanmax(y) - _np.nanmin(y))
        nx = 900
        ny = int(max(300, min(900, round(nx * yr / max(xr, 1e-9)))))
        gx = _np.linspace(_np.nanmin(x), _np.nanmax(x), nx)
        gy = _np.linspace(_np.nanmin(y), _np.nanmax(y), ny)

        ax_rect, cbar_rect = _bulach_amp_axes_rect(fig, x, y)
        ax = fig.add_axes(ax_rect)
        cax = fig.add_axes(cbar_rect)
        canvas.ax = ax

        if SCIPY_OK and len(v) > 100:
            Xg, Yg = _np.meshgrid(gx, gy)
            Z = griddata((x, y), v, (Xg, Yg), method="linear")
            im = ax.imshow(
                Z,
                extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                origin="lower",
                aspect="equal",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        else:
            im = ax.scatter(x, y, c=v, s=4, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal", adjustable="box")

        ax.set_anchor("C")
        ax.set_title(
            title + " | de-striped line-normalised amplitude, 5-line median, 1 m edge trim",
            fontsize=12,
            pad=8,
        )
        ax.set_xlabel("Local easting [m]", fontsize=11)
        ax.set_ylabel("Local northing [m]", fontsize=11)
        ax.grid(True, alpha=0.22)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Relative absolute amplitude", fontsize=10)
        cb.ax.tick_params(labelsize=9)

        canvas.draw()

        _pe_update_progress(dlg, start, total, total, "Done.")
        dlg.close()

        log_path = self.owner.root / "pulseekko_3d_analysis_last_run.log"
        try:
            log_path.write_text(
                "PulseEKKO HD amplitude projection log\n"
                f"Run: {_datetime.datetime.now()}\n"
                f"View: {title}\n"
                "Settings: centred equal-scale HD display, de-striped line-normalised amplitude, "
                "5-line median, 1 m edge trim, 2–98% colour clip\n"
                f"Lines processed: {len(lines)}\n\n" + "\n".join(log)
            )
        except Exception:
            pass

        _bulach_amp_status(self, f"Updated centred HD Bulach amplitude projection. Log: {log_path}")

    except Exception as e:
        try:
            dlg.close()
        except Exception:
            pass
        _bulach_amp_status(self, f"Bulach HD amplitude projection stopped: {e}")


def _bulach_save_current_map_hd(self):
    """Save whichever Bulach 3-D analysis map is currently selected as cropped HD PNG+PDF."""
    try:
        from pathlib import Path
        import datetime

        name, canvas = self.selected_canvas()
        out_dir = Path(getattr(getattr(self, "owner", None), "root", Path("."))) / "exports" / "hd_maps"
        out_dir.mkdir(parents=True, exist_ok=True)

        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.lower()).strip("_")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = out_dir / f"bulach_{safe}_{ts}.png"
        pdf = out_dir / f"bulach_{safe}_{ts}.pdf"

        canvas.fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
        canvas.fig.savefig(pdf, dpi=350, bbox_inches="tight", facecolor="white")
        _bulach_amp_status(self, f"Saved HD current map: {png} and {pdf}")
        print(f"Saved HD current map: {png}")
        print(f"Saved HD current map: {pdf}")

    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save current map HD failed", str(e))
        except Exception:
            print("Save current map HD failed:", e)


def _bulach_amp_add_hd_save_button(self):
    try:
        if getattr(self, "_bulach_amp_hd_current_button_added", False):
            return
        from PyQt6.QtWidgets import QPushButton

        btn = QPushButton("Save Current Map HD", self)
        btn.setToolTip("Save the currently selected Bulach 3D-analysis map as cropped HD PNG and PDF.")
        btn.clicked.connect(lambda *_: _bulach_save_current_map_hd(self))

        lay = self.layout()
        if lay is not None:
            # Put it with the other HD buttons, below the controls and above the tab widget.
            insert_at = 3 if lay.count() >= 3 else lay.count()
            lay.insertWidget(insert_at, btn)

        self._bulach_amp_hd_current_button_added = True
    except Exception as e:
        print("Could not add Bulach Save Current Map HD button:", e)


try:
    if not hasattr(PulseEkko3DAnalysis, "_old_init_for_bulach_amp_hd"):
        PulseEkko3DAnalysis._old_init_for_bulach_amp_hd = PulseEkko3DAnalysis.__init__

        def _bulach_amp_hd_init(self, *args, **kwargs):
            PulseEkko3DAnalysis._old_init_for_bulach_amp_hd(self, *args, **kwargs)
            _bulach_amp_make_scrollable(self)
            _bulach_amp_add_hd_save_button(self)

        PulseEkko3DAnalysis.__init__ = _bulach_amp_hd_init

    if not hasattr(PulseEkko3DAnalysis, "_old_update_selected_for_bulach_amp_hd"):
        PulseEkko3DAnalysis._old_update_selected_for_bulach_amp_hd = PulseEkko3DAnalysis.update_selected

        def _bulach_amp_hd_update_selected(self, *args, **kwargs):
            try:
                name, canvas = self.selected_canvas()
                if name == "Amplitude Projection":
                    _bulach_amp_make_scrollable(self)
                    return _bulach_amp_plot_hd(self, canvas)
            except Exception as e:
                print("Bulach HD amplitude projection override failed, falling back:", e)
            return PulseEkko3DAnalysis._old_update_selected_for_bulach_amp_hd(self, *args, **kwargs)

        PulseEkko3DAnalysis.update_selected = _bulach_amp_hd_update_selected

    print("Bulach centred HD amplitude-projection patch active.")
except Exception as e:
    print("Bulach centred HD amplitude-projection patch failed:", e)
# --- end Bulach amplitude projection centered HD patch ---




# --- QGIS GEOTIFF EXPORT PATCH ---
# Adds QGIS-ready GeoTIFF export for GPR GPS-plan raster + amplitude projection raster.
# Schleitheim: uses GNSS/RTK-derived local-to-LV95 affine if pyproj is installed.
# Bülach: uses bulach_georef.json if present; otherwise tries to borrow bounds from nearby EM/magnetic GeoTIFF.

def _qgis_msg(obj, title, text, warn=False):
    try:
        from PyQt6.QtWidgets import QMessageBox
        (QMessageBox.warning if warn else QMessageBox.information)(obj, title, text)
    except Exception:
        print(title + ":", text)

def _qgis_status(obj, msg):
    for target in (getattr(obj, "owner", None), getattr(obj, "main", None), obj):
        st = getattr(target, "status", None)
        if st is not None:
            try:
                st.setText(str(msg)); return
            except Exception:
                pass
    print(msg)

def _qgis_tab_kind(tab):
    return "bulach" if tab.__class__.__name__ == "PulseEkko3DAnalysis" else "schleitheim"

def _qgis_project_root(tab):
    if _qgis_tab_kind(tab) == "bulach":
        return Path(getattr(getattr(tab, "owner", None), "root", Path("/home/luqman/gpr_gui/data/PulseEkko")))
    return Path("/home/luqman/gpr_gui/data/MALA")

def _qgis_lines(tab):
    if _qgis_tab_kind(tab) == "bulach":
        return list(getattr(getattr(tab, "owner", None), "lines", []))
    return list(getattr(getattr(tab, "main", None), "lines", []))

def _qgis_xy(line):
    x = getattr(line, "x", None); y = getattr(line, "y", None)
    if x is None or y is None:
        return None, None
    return np.asarray(x, float), np.asarray(y, float)

def _qgis_affine_fit(src_xy, dst_xy):
    src = np.asarray(src_xy, float); dst = np.asarray(dst_xy, float)
    good = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[good]; dst = dst[good]
    if len(src) < 3:
        raise RuntimeError("Need at least 3 coordinate pairs for georeferencing.")
    A = np.c_[src[:,0], src[:,1], np.ones(len(src))]
    ex = np.linalg.lstsq(A, dst[:,0], rcond=None)[0]
    ny = np.linalg.lstsq(A, dst[:,1], rcond=None)[0]
    return ex, ny

def _qgis_apply_affine(x, y, aff):
    ex, ny = aff
    x = np.asarray(x, float); y = np.asarray(y, float)
    E = ex[0]*x + ex[1]*y + ex[2]
    N = ny[0]*x + ny[1]*y + ny[2]
    return E, N

def _qgis_find_existing_reference_tif(root):
    roots = [root, root.parent, Path("/home/luqman/gpr_gui/data")]
    keys = ("mag", "magnetic", "em", "emi", "gradient", "buelach", "bulach")
    found = []
    for r in roots:
        if not r.exists():
            continue
        for f in list(r.rglob("*.tif")) + list(r.rglob("*.tiff")):
            low = f.name.lower()
            if "gpr_qgis" in low:
                continue
            score = sum(k in low for k in keys)
            found.append((score, f))
    found.sort(reverse=True, key=lambda z: z[0])
    return found[0][1] if found and found[0][0] > 0 else None

def _qgis_affine_for_tab(tab):
    kind = _qgis_tab_kind(tab)
    root = _qgis_project_root(tab)
    lines = _qgis_lines(tab)

    # 1) Preferred for Bülach: manual georef file.
    cfgs = [root / "bulach_georef.json", root.parent / "bulach_georef.json"]
    if kind == "bulach":
        for cfg in cfgs:
            if cfg.exists():
                import json
                c = json.loads(cfg.read_text())
                if "local_points" in c and "lv95_points" in c:
                    return _qgis_affine_fit(c["local_points"], c["lv95_points"])
                if "lv95_bbox" in c:
                    lx, ly = [], []
                    for ln in lines:
                        x, y = _qgis_xy(ln)
                        if x is not None:
                            lx.extend(x); ly.extend(y)
                    xmin,xmax,ymin,ymax = map(float, [np.nanmin(lx),np.nanmax(lx),np.nanmin(ly),np.nanmax(ly)])
                    E0,E1,N0,N1 = map(float, c["lv95_bbox"])
                    local = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
                    lv95  = [[E0,N0],[E1,N0],[E1,N1],[E0,N1]]
                    return _qgis_affine_fit(local, lv95)

    # 2) Schleitheim: fit local x/y to RTK WGS84 converted to LV95.
    src = []; dst = []
    if kind == "schleitheim":
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
            for ln in lines:
                x, y = _qgis_xy(ln)
                lat = getattr(ln, "lat", None); lon = getattr(ln, "lon", None)
                if x is None or lat is None or lon is None or len(lat) < 2:
                    continue
                lat = np.asarray(lat, float); lon = np.asarray(lon, float)
                ii = np.linspace(0, len(x)-1, len(lat)).astype(int)
                E, N = tr.transform(lon, lat)
                src.extend(np.c_[x[ii], y[ii]].tolist())
                dst.extend(np.c_[E, N].tolist())
            if len(src) >= 3:
                return _qgis_affine_fit(src, dst)
        except Exception as e:
            raise RuntimeError("Schleitheim LV95 export needs pyproj. Install with: .venv/bin/python -m pip install pyproj\n" + str(e))

    # 3) Crafty Bülach fallback: borrow LV95 bbox from EM/magnetic GeoTIFF, if available.
    if kind == "bulach":
        ref = _qgis_find_existing_reference_tif(root)
        if ref is not None:
            import rasterio
            with rasterio.open(ref) as ds:
                b = ds.bounds
            lx, ly = [], []
            for ln in lines:
                x, y = _qgis_xy(ln)
                if x is not None:
                    lx.extend(x); ly.extend(y)
            xmin,xmax,ymin,ymax = map(float, [np.nanmin(lx),np.nanmax(lx),np.nanmin(ly),np.nanmax(ly)])
            local = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
            lv95  = [[b.left,b.bottom],[b.right,b.bottom],[b.right,b.top],[b.left,b.top]]
            _qgis_status(tab, f"Using reference GeoTIFF bbox for Bülach georef: {ref}")
            return _qgis_affine_fit(local, lv95)

        template = root / "bulach_georef.json"
        template.write_text('{\n  "local_points": [[0,0], [1,0], [0,1]],\n  "lv95_points": [[2682800,1267900], [2682801,1267900], [2682800,1267901]]\n}\n')
        raise RuntimeError(f"No Bülach georeference found. Created template:\n{template}\nEdit it using 3 matching local GPR points and LV95 points, then export again.")

    raise RuntimeError("Could not define local-to-LV95 transform.")

def _qgis_collect_projection(tab):
    kind = _qgis_tab_kind(tab)
    if kind == "bulach":
        return tab.collect_values("projection")

    xs, ys, vals = [], [], []
    lines = tab.selected_lines_for_maps()
    tmin, tmax = sorted([float(tab.proj_tmin.value()), float(tab.proj_tmax.value())])
    step = max(1, int(tab.trace_step.value()))
    for ln in lines:
        try:
            data = tab.ensure_data(ln)
            ntr, ns = data.shape
            x, y = tab.trace_xy(ln, ntr)
            t = tab.time_vector(ln, ns)
            m = (t >= tmin) & (t <= tmax)
            if not np.any(m):
                continue
            idx = np.arange(0, ntr, step)
            amp = np.nanmax(np.abs(data[idx][:, m]), axis=1)
            xs.extend(np.asarray(x)[idx]); ys.extend(np.asarray(y)[idx]); vals.extend(amp)
        except Exception as e:
            print("Skipping projection export", getattr(ln, "name", ln), e)
    return np.asarray(xs), np.asarray(ys), np.asarray(vals)

def _qgis_export_geotiffs(tab):
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        from scipy.interpolate import griddata
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install needed packages:\n.venv/bin/python -m pip install rasterio pyproj scipy\n\n" + str(e), True)
        return

    root = _qgis_project_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)
    kind = _qgis_tab_kind(tab)
    aff = _qgis_affine_for_tab(tab)
    res = 0.10 if kind == "bulach" else 0.25

    # Amplitude projection GeoTIFF
    x, y, v = _qgis_collect_projection(tab)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[good], y[good], v[good]
    if len(v) < 4:
        raise RuntimeError("Not enough amplitude points to export.")
    E, N = _qgis_apply_affine(x, y, aff)
    xmin,xmax = float(np.nanmin(E)), float(np.nanmax(E))
    ymin,ymax = float(np.nanmin(N)), float(np.nanmax(N))
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((E, N), v, (X, Y), method="linear").astype("float32")
    arr = np.flipud(Z)
    transform = from_origin(xi.min()-res/2, yi.max()+res/2, res, res)
    amp_out = outdir / f"{kind}_gpr_amplitude_projection_epsg2056.tif"
    with rasterio.open(amp_out, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", crs="EPSG:2056", transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(arr, 1)

    # GPS plan GeoTIFF rasterised from line geometry
    shapes = []
    allE, allN = [], []
    for ln in _qgis_lines(tab):
        xx, yy = _qgis_xy(ln)
        if xx is None or len(xx) < 2:
            continue
        EE, NN = _qgis_apply_affine(xx, yy, aff)
        good = np.isfinite(EE) & np.isfinite(NN)
        coords = [(float(a), float(b)) for a, b in zip(EE[good], NN[good])]
        if len(coords) >= 2:
            shapes.append(({"type": "LineString", "coordinates": coords}, 255))
            allE.extend(EE[good]); allN.extend(NN[good])
    if len(shapes) < 1:
        raise RuntimeError("No line geometry available for GPS-plan GeoTIFF.")
    xmin,xmax = float(np.nanmin(allE)), float(np.nanmax(allE))
    ymin,ymax = float(np.nanmin(allN)), float(np.nanmax(allN))
    pad = 2.0
    xi = np.arange(xmin-pad, xmax+pad+res, res)
    yi = np.arange(ymin-pad, ymax+pad+res, res)
    transform = from_origin(xi.min()-res/2, yi.max()+res/2, res, res)
    gps = rasterize(shapes, out_shape=(len(yi), len(xi)), transform=transform, fill=0, dtype="uint8")
    gps_out = outdir / f"{kind}_gpr_gps_plan_epsg2056.tif"
    with rasterio.open(gps_out, "w", driver="GTiff", height=gps.shape[0], width=gps.shape[1], count=1, dtype="uint8", crs="EPSG:2056", transform=transform, nodata=0, compress="deflate") as dst:
        dst.write(gps, 1)

    _qgis_msg(tab, "QGIS GeoTIFF export complete", f"Saved:\n{gps_out}\n{amp_out}")
    _qgis_status(tab, f"Saved QGIS GeoTIFFs: {outdir}")

def _qgis_add_button(tab):
    if getattr(tab, "_qgis_geotiff_button_added", False):
        return
    from PyQt6.QtWidgets import QPushButton
    btn = QPushButton("Export georeferenced TIFFs for QGIS")
    btn.clicked.connect(lambda *_: _qgis_export_geotiffs(tab))
    lay = tab.layout()
    if lay is not None:
        lay.insertWidget(1, btn)
    tab.qgis_geotiff_btn = btn
    tab._qgis_geotiff_button_added = True

def _qgis_patch_class(cls):
    if cls is None or getattr(cls, "_qgis_geotiff_patched", False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        try:
            _qgis_add_button(self)
        except Exception as e:
            print("Could not add QGIS GeoTIFF button:", e)
    cls.__init__ = new_init
    cls._qgis_geotiff_patched = True

try:
    _qgis_patch_class(PulseEkko3DAnalysis)
except Exception as e:
    print("Bülach QGIS patch failed:", e)

try:
    _qgis_patch_class(getattr(gpr_app, "GPR3DStandardAnalysisTab", None))
    _qgis_patch_class(getattr(gpr_app, "GPR3DAnalysisTab", None))
except Exception as e:
    print("Schleitheim QGIS patch failed:", e)

print("QGIS GeoTIFF export patch active for Schleitheim and Bülach.")
# --- END QGIS GEOTIFF EXPORT PATCH ---



# --- QGIS GEOTIFF EXPORT PATCH CSV GPS ---
# Exports QGIS-ready GeoTIFFs for GPS plan view and amplitude projection.
# Bülach: uses /data/PulseEkko/GPS/LINE###_GPS.csv synthetic lat/lon coordinates.
# Schleitheim: uses MALA .cor lat/lon stored on each line.

def _qgis_msg(obj, title, text, warn=False):
    try:
        from PyQt6.QtWidgets import QMessageBox
        (QMessageBox.warning if warn else QMessageBox.information)(obj, title, text)
    except Exception:
        print(title + ":", text)

def _qgis_kind(tab):
    return "bulach" if tab.__class__.__name__ == "PulseEkko3DAnalysis" else "schleitheim"

def _qgis_root(tab):
    if _qgis_kind(tab) == "bulach":
        return Path(getattr(getattr(tab, "owner", None), "root", Path("/home/luqman/gpr_gui/data/PulseEkko")))
    return Path("/home/luqman/gpr_gui/data/MALA")

def _qgis_lines(tab):
    if _qgis_kind(tab) == "bulach":
        return list(getattr(getattr(tab, "owner", None), "lines", []))
    if hasattr(tab, "selected_lines_for_maps"):
        return list(tab.selected_lines_for_maps())
    return list(getattr(getattr(tab, "main", None), "lines", []))

def _read_bulach_gps_csv(root, line):
    import csv
    gpsdir = root / "GPS"
    name = getattr(line, "name", "")
    idx = getattr(line, "idx", None)
    candidates = []
    if name:
        candidates.append(gpsdir / f"{name}_GPS.csv")
    if idx is not None:
        candidates.append(gpsdir / f"LINE{int(idx):03d}_GPS.csv")
    candidates.append(gpsdir / f"{str(name).replace('.HD','')}_GPS.csv")
    f = next((c for c in candidates if c.exists()), None)
    if f is None:
        raise FileNotFoundError(f"No GPS CSV found for {name}. Tried: " + ", ".join(map(str, candidates)))
    lat, lon = [], []
    with f.open() as fh:
        for r in csv.DictReader(fh):
            lat.append(float(r["lat"]))
            lon.append(float(r["lon"]))
    return np.asarray(lat, float), np.asarray(lon, float)

def _line_lv95(tab, line, ntr=None):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    kind = _qgis_kind(tab)

    if kind == "bulach":
        lat, lon = _read_bulach_gps_csv(_qgis_root(tab), line)
    else:
        lat = getattr(line, "lat", None)
        lon = getattr(line, "lon", None)
        if lat is None or lon is None:
            raise RuntimeError(f"No lat/lon stored for Schleitheim line {getattr(line, 'name', line)}")
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)

    E, N = tr.transform(lon, lat)
    E, N = np.asarray(E, float), np.asarray(N, float)

    if ntr is not None and len(E) != int(ntr):
        old = np.linspace(0, 1, len(E))
        new = np.linspace(0, 1, int(ntr))
        E = np.interp(new, old, E)
        N = np.interp(new, old, N)

    return E, N

def _qgis_get_array_and_time(tab, line):
    kind = _qgis_kind(tab)
    if kind == "bulach":
        arr = tab.get_array(line)
        t = tab.owner.corrected_time_ns(line)
    else:
        arr = tab.ensure_data(line)
        t = tab.time_vector(line, arr.shape[1])
    return np.asarray(arr, float), np.asarray(t, float)

def _qgis_collect_amp_points(tab):
    xs, ys, vals = [], [], []
    tmin = float(getattr(tab, "proj_tmin").value())
    tmax = float(getattr(tab, "proj_tmax").value())
    lo, hi = sorted([tmin, tmax])
    step = max(1, int(getattr(tab, "trace_step").value()))

    for line in _qgis_lines(tab):
        try:
            arr, t = _qgis_get_array_and_time(tab, line)
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue
            E, N = _line_lv95(tab, line, arr.shape[0])
            idx = np.arange(0, arr.shape[0], step)
            amp = np.nanmax(np.abs(arr[idx][:, m]), axis=1)
            xs.extend(E[idx])
            ys.extend(N[idx])
            vals.extend(amp)
        except Exception as e:
            print("Skipping amplitude export line", getattr(line, "name", line), ":", e)

    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(vals, float)

def _qgis_export_geotiffs(tab):
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        from scipy.interpolate import griddata
        import pyproj
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install needed packages:\n.venv/bin/python -m pip install rasterio pyproj scipy\n\n" + str(e), True)
        return

    kind = _qgis_kind(tab)
    root = _qgis_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)
    res = 0.10 if kind == "bulach" else 0.25

    # 1) Amplitude projection GeoTIFF
    E, N, V = _qgis_collect_amp_points(tab)
    good = np.isfinite(E) & np.isfinite(N) & np.isfinite(V)
    E, N, V = E[good], N[good], V[good]
    if len(V) < 4:
        _qgis_msg(tab, "Export failed", "Not enough amplitude points to export.", True)
        return

    xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
    ymin, ymax = float(np.nanmin(N)), float(np.nanmax(N))
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((E, N), V, (X, Y), method="linear").astype("float32")
    arr = np.flipud(Z)
    transform = from_origin(xi.min() - res/2, yi.max() + res/2, res, res)

    amp_out = outdir / f"{kind}_gpr_amplitude_projection_epsg2056.tif"
    with rasterio.open(
        amp_out, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs="EPSG:2056", transform=transform,
        nodata=np.nan, compress="deflate"
    ) as dst:
        dst.write(arr, 1)

    # 2) GPS plan view GeoTIFF: rasterised survey lines
    shapes, allE, allN = [], [], []
    for line in _qgis_lines(tab):
        try:
            EE, NN = _line_lv95(tab, line, None)
            good = np.isfinite(EE) & np.isfinite(NN)
            coords = [(float(a), float(b)) for a, b in zip(EE[good], NN[good])]
            if len(coords) >= 2:
                shapes.append(({"type": "LineString", "coordinates": coords}, 255))
                allE.extend(EE[good])
                allN.extend(NN[good])
        except Exception as e:
            print("Skipping GPS line export", getattr(line, "name", line), ":", e)

    if not shapes:
        _qgis_msg(tab, "Export failed", "No GPS line geometry available.", True)
        return

    pad = 2.0
    xmin, xmax = float(np.nanmin(allE)) - pad, float(np.nanmax(allE)) + pad
    ymin, ymax = float(np.nanmin(allN)) - pad, float(np.nanmax(allN)) + pad
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    transform = from_origin(xi.min() - res/2, yi.max() + res/2, res, res)
    gps = rasterize(shapes, out_shape=(len(yi), len(xi)), transform=transform, fill=0, dtype="uint8")

    gps_out = outdir / f"{kind}_gpr_gps_plan_epsg2056.tif"
    with rasterio.open(
        gps_out, "w", driver="GTiff",
        height=gps.shape[0], width=gps.shape[1], count=1,
        dtype="uint8", crs="EPSG:2056", transform=transform,
        nodata=0, compress="deflate"
    ) as dst:
        dst.write(gps, 1)

    _qgis_msg(tab, "QGIS GeoTIFF export complete", f"Saved:\n{gps_out}\n{amp_out}")

def _qgis_add_button(tab):
    if getattr(tab, "_qgis_geotiff_button_added", False):
        return
    from PyQt6.QtWidgets import QPushButton
    btn = QPushButton("Export georeferenced TIFFs for QGIS")
    btn.clicked.connect(lambda *_: _qgis_export_geotiffs(tab))
    lay = tab.layout()
    if lay is not None:
        lay.insertWidget(1, btn)
    tab.qgis_geotiff_btn = btn
    tab._qgis_geotiff_button_added = True

def _qgis_patch_class(cls):
    if cls is None or getattr(cls, "_qgis_csv_geotiff_patched", False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        try:
            _qgis_add_button(self)
        except Exception as e:
            print("Could not add QGIS GeoTIFF button:", e)
    cls.__init__ = new_init
    cls._qgis_csv_geotiff_patched = True

try:
    _qgis_patch_class(PulseEkko3DAnalysis)
except Exception as e:
    print("Bülach QGIS CSV patch failed:", e)

try:
    _qgis_patch_class(getattr(gpr_app, "GPR3DStandardAnalysisTab", None))
    _qgis_patch_class(getattr(gpr_app, "GPR3DAnalysisTab", None))
except Exception as e:
    print("Schleitheim QGIS patch failed:", e)

print("QGIS GeoTIFF CSV-GPS export patch active.")
# --- END QGIS GEOTIFF EXPORT PATCH CSV GPS ---



# --- QGIS VECTOR GPS EXPORT PATCH ---
# Adds proper QGIS vector line export for GPS plan view.
# GeoTIFF is kept for amplitude projection; GPS plan is exported as coloured LineString GeoPackage.

def _qgis_export_gps_lines_gpkg(tab):
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install:\n.venv/bin/python -m pip install geopandas shapely pyproj\n\n" + str(e), True)
        return

    kind = _qgis_kind(tab)
    root = _qgis_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, line in enumerate(_qgis_lines(tab)):
        try:
            E, N = _line_lv95(tab, line, None)
            good = np.isfinite(E) & np.isfinite(N)
            coords = [(float(e), float(n)) for e, n in zip(E[good], N[good])]
            if len(coords) < 2:
                continue
            rows.append({
                "site": kind,
                "line_name": str(getattr(line, "name", f"line_{i}")),
                "line_id": int(getattr(line, "idx", getattr(line, "number", i))),
                "geometry": LineString(coords),
            })
        except Exception as e:
            print("Skipping vector GPS line", getattr(line, "name", line), ":", e)

    if not rows:
        _qgis_msg(tab, "Export failed", "No GPS lines available for vector export.", True)
        return

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:2056")
    out = outdir / f"{kind}_gpr_gps_plan_lines_epsg2056.gpkg"
    gdf.to_file(out, layer="gpr_lines", driver="GPKG")
    _qgis_msg(tab, "QGIS GPS vector export complete", f"Saved:\n{out}")

def _qgis_add_vector_button(tab):
    if getattr(tab, "_qgis_vector_gps_button_added", False):
        return
    from PyQt6.QtWidgets import QPushButton
    btn = QPushButton("Export GPS plan as QGIS vector lines")
    btn.clicked.connect(lambda *_: _qgis_export_gps_lines_gpkg(tab))
    lay = tab.layout()
    if lay is not None:
        lay.insertWidget(2, btn)
    tab.qgis_vector_gps_btn = btn
    tab._qgis_vector_gps_button_added = True

def _qgis_patch_vector_class(cls):
    if cls is None or getattr(cls, "_qgis_vector_gps_patched", False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        try:
            _qgis_add_vector_button(self)
        except Exception as e:
            print("Could not add GPS vector export button:", e)
    cls.__init__ = new_init
    cls._qgis_vector_gps_patched = True

try:
    _qgis_patch_vector_class(PulseEkko3DAnalysis)
except Exception as e:
    print("Bülach vector GPS patch failed:", e)

try:
    _qgis_patch_vector_class(getattr(gpr_app, "GPR3DStandardAnalysisTab", None))
    _qgis_patch_vector_class(getattr(gpr_app, "GPR3DAnalysisTab", None))
except Exception as e:
    print("Schleitheim vector GPS patch failed:", e)

print("QGIS vector GPS export patch active.")
# --- END QGIS VECTOR GPS EXPORT PATCH ---



# --- HIGH RES QGIS AMPLITUDE GEOTIFF PATCH ---
# Overrides the previous GeoTIFF export with finer pixels and smoother interpolation.
# GPS line export should remain vector GPKG; this only improves amplitude raster sharpness.

def _qgis_export_geotiffs(tab):
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
        import pyproj
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install needed packages:\n.venv/bin/python -m pip install rasterio pyproj scipy\n\n" + str(e), True)
        return

    kind = _qgis_kind(tab)
    root = _qgis_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)

    # Finer export pixels for QGIS display.
    # This is visual/interpolation resolution, not true acquisition resolution.
    res = 0.025 if kind == "bulach" else 0.05

    # 1) High-resolution amplitude projection GeoTIFF
    E, N, V = _qgis_collect_amp_points(tab)
    good = np.isfinite(E) & np.isfinite(N) & np.isfinite(V)
    E, N, V = E[good], N[good], V[good]

    if len(V) < 4:
        _qgis_msg(tab, "Export failed", "Not enough amplitude points to export.", True)
        return

    xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
    ymin, ymax = float(np.nanmin(N)), float(np.nanmax(N))

    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    X, Y = np.meshgrid(xi, yi)

    # Cubic gives smoother maps. Nearest is only used to fill edge holes.
    try:
        Z = griddata((E, N), V, (X, Y), method="cubic")
    except Exception:
        Z = griddata((E, N), V, (X, Y), method="linear")

    Z_near = griddata((E, N), V, (X, Y), method="nearest")
    Z = np.where(np.isfinite(Z), Z, Z_near)

    # Light smoothing only for display quality. Keeps broad anomaly pattern.
    Z = gaussian_filter(Z.astype("float32"), sigma=0.8)

    arr = np.flipud(Z.astype("float32"))
    transform = from_origin(xi.min() - res/2, yi.max() + res/2, res, res)

    amp_out = outdir / f"{kind}_gpr_amplitude_projection_HIGHRES_epsg2056.tif"
    with rasterio.open(
        amp_out, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs="EPSG:2056", transform=transform,
        nodata=np.nan, compress="deflate"
    ) as dst:
        dst.write(arr, 1)

    # 2) GPS plan raster, still exported for convenience.
    # For proper survey line display, use the vector GPKG export button.
    shapes, allE, allN = [], [], []
    for line in _qgis_lines(tab):
        try:
            EE, NN = _line_lv95(tab, line, None)
            ok = np.isfinite(EE) & np.isfinite(NN)
            coords = [(float(a), float(b)) for a, b in zip(EE[ok], NN[ok])]
            if len(coords) >= 2:
                shapes.append(({"type": "LineString", "coordinates": coords}, 255))
                allE.extend(EE[ok])
                allN.extend(NN[ok])
        except Exception as e:
            print("Skipping GPS line raster export", getattr(line, "name", line), ":", e)

    if shapes:
        pad = 2.0
        xmin, xmax = float(np.nanmin(allE)) - pad, float(np.nanmax(allE)) + pad
        ymin, ymax = float(np.nanmin(allN)) - pad, float(np.nanmax(allN)) + pad
        xi = np.arange(xmin, xmax + res, res)
        yi = np.arange(ymin, ymax + res, res)
        transform = from_origin(xi.min() - res/2, yi.max() + res/2, res, res)
        gps = rasterize(shapes, out_shape=(len(yi), len(xi)), transform=transform, fill=0, dtype="uint8")

        gps_out = outdir / f"{kind}_gpr_gps_plan_HIGHRES_epsg2056.tif"
        with rasterio.open(
            gps_out, "w", driver="GTiff",
            height=gps.shape[0], width=gps.shape[1], count=1,
            dtype="uint8", crs="EPSG:2056", transform=transform,
            nodata=0, compress="deflate"
        ) as dst:
            dst.write(gps, 1)
    else:
        gps_out = "GPS raster not exported"

    _qgis_msg(tab, "High-res QGIS GeoTIFF export complete", f"Saved:\n{amp_out}\n{gps_out}\n\nUse the GPKG vector layer for survey lines.")
# --- END HIGH RES QGIS AMPLITUDE GEOTIFF PATCH ---



# --- FAST BULACH GPS CSV GEOTIFF PATCH ---
# Fixes Bülach QGIS export freeze/empty output.
# Bülach amplitude is exported as a real line-trace raster using synthetic GPS CSV geometry.
# GPS plan should be exported as vector GPKG, not raster.

def _bulach_read_gps_csv_for_line(tab, line):
    import csv
    root = _qgis_root(tab)
    gpsdir = root / "GPS"
    name = str(getattr(line, "name", ""))
    idx = getattr(line, "idx", None)

    candidates = []
    if name:
        candidates.append(gpsdir / f"{name}_GPS.csv")
    if idx is not None:
        candidates.append(gpsdir / f"LINE{int(idx):03d}_GPS.csv")

    f = next((c for c in candidates if c.exists()), None)
    if f is None:
        raise FileNotFoundError("Missing Bülach GPS CSV for " + name)

    lat, lon, tr = [], [], []
    with f.open() as fh:
        for r in csv.DictReader(fh):
            lat.append(float(r["lat"]))
            lon.append(float(r["lon"]))
            tr.append(int(float(r["trace"])))
    return np.asarray(tr), np.asarray(lat), np.asarray(lon)

def _bulach_export_fast_amplitude_geotiff(tab):
    try:
        import rasterio
        from rasterio.transform import Affine
        from pyproj import Transformer
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install:\n.venv/bin/python -m pip install rasterio pyproj\n\n" + str(e), True)
        return

    root = _qgis_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)

    lines = list(getattr(getattr(tab, "owner", None), "lines", []))
    if not lines:
        _qgis_msg(tab, "Export failed", "No Bülach lines loaded.", True)
        return

    lo, hi = sorted([float(tab.proj_tmin.value()), float(tab.proj_tmax.value())])
    trf = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    rows = []
    fit_pix = []
    fit_lv95 = []
    max_ntr = 0

    for row_i, line in enumerate(lines):
        try:
            arr = tab.get_array(line)
            t = tab.owner.corrected_time_ns(line)
            m = (t >= lo) & (t <= hi)
            if not np.any(m):
                continue

            amp = np.nanmax(np.abs(arr[:, m]), axis=1).astype("float32")
            trace_csv, lat, lon = _bulach_read_gps_csv_for_line(tab, line)
            E, N = trf.transform(lon, lat)
            E = np.asarray(E, float)
            N = np.asarray(N, float)

            if len(E) != len(amp):
                old = np.linspace(0, 1, len(E))
                new = np.linspace(0, 1, len(amp))
                E = np.interp(new, old, E)
                N = np.interp(new, old, N)

            rows.append((row_i, amp))
            max_ntr = max(max_ntr, len(amp))

            # Use sparse points to fit pixel(row/col) -> LV95 affine transform.
            step = max(1, len(amp)//50)
            cols = np.arange(0, len(amp), step)
            fit_pix.extend([[float(c), float(row_i)] for c in cols])
            fit_lv95.extend([[float(E[c]), float(N[c])] for c in cols])
        except Exception as e:
            print("Skipping Bülach line export", getattr(line, "name", line), e)

    if not rows:
        _qgis_msg(tab, "Export failed", "No Bülach amplitude rows could be exported.", True)
        return

    data = np.full((len(lines), max_ntr), np.nan, dtype="float32")
    for row_i, amp in rows:
        data[row_i, :len(amp)] = amp

    pix = np.asarray(fit_pix, float)
    lv = np.asarray(fit_lv95, float)
    A = np.c_[pix[:,0], pix[:,1], np.ones(len(pix))]
    ex = np.linalg.lstsq(A, lv[:,0], rcond=None)[0]
    ny = np.linalg.lstsq(A, lv[:,1], rcond=None)[0]

    # Affine maps pixel corner, so shift fitted pixel-centre transform by half a pixel.
    a, b, c = ex
    d, e, f = ny
    transform = Affine(a, b, c - 0.5*a - 0.5*b, d, e, f - 0.5*d - 0.5*e)

    out = outdir / "bulach_gpr_amplitude_projection_TRACEGRID_epsg2056.tif"
    with rasterio.open(
        out, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs="EPSG:2056", transform=transform,
        nodata=np.nan, compress="deflate"
    ) as dst:
        dst.write(data, 1)

    _qgis_msg(tab, "Bülach amplitude GeoTIFF export complete", f"Saved:\n{out}\n\nThis is the correct high-resolution trace-grid export.")

def _qgis_export_geotiffs(tab):
    # Bülach gets special fast export using synthetic GPS CSV trace geometry.
    if _qgis_kind(tab) == "bulach":
        _bulach_export_fast_amplitude_geotiff(tab)
        return

    # Schleitheim keeps previous high-res scattered interpolation export.
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
        import pyproj
    except Exception as e:
        _qgis_msg(tab, "Missing package", "Install needed packages:\n.venv/bin/python -m pip install rasterio pyproj scipy\n\n" + str(e), True)
        return

    kind = _qgis_kind(tab)
    root = _qgis_root(tab)
    outdir = root / "qgis_exports"
    outdir.mkdir(parents=True, exist_ok=True)
    res = 0.05

    E, N, V = _qgis_collect_amp_points(tab)
    good = np.isfinite(E) & np.isfinite(N) & np.isfinite(V)
    E, N, V = E[good], N[good], V[good]
    if len(V) < 4:
        _qgis_msg(tab, "Export failed", "Not enough amplitude points to export.", True)
        return

    xmin, xmax = float(np.nanmin(E)), float(np.nanmax(E))
    ymin, ymax = float(np.nanmin(N)), float(np.nanmax(N))
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    X, Y = np.meshgrid(xi, yi)

    try:
        Z = griddata((E, N), V, (X, Y), method="cubic")
    except Exception:
        Z = griddata((E, N), V, (X, Y), method="linear")
    Z_near = griddata((E, N), V, (X, Y), method="nearest")
    Z = np.where(np.isfinite(Z), Z, Z_near)
    Z = gaussian_filter(Z.astype("float32"), sigma=0.8)

    arr = np.flipud(Z.astype("float32"))
    transform = from_origin(xi.min() - res/2, yi.max() + res/2, res, res)
    out = outdir / f"{kind}_gpr_amplitude_projection_HIGHRES_epsg2056.tif"

    with rasterio.open(
        out, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs="EPSG:2056", transform=transform,
        nodata=np.nan, compress="deflate"
    ) as dst:
        dst.write(arr, 1)

    _qgis_msg(tab, "Schleitheim amplitude GeoTIFF export complete", f"Saved:\n{out}")

print("Fast Bülach GPS CSV GeoTIFF patch active.")
# --- END FAST BULACH GPS CSV GEOTIFF PATCH ---
