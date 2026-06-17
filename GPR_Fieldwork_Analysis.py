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

    if sec_power > 0:
        t = np.asarray(time_ns, float)
        gain = (1.0 + t / max(float(t[-1]), 1.0)) ** float(sec_power)
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

        self.sec_power = QDoubleSpinBox(); self.sec_power.setRange(0, 10); self.sec_power.setSingleStep(0.05); self.sec_power.setValue(0.0)
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

        grid.addWidget(QLabel("SEC gain power"), 1, 0)
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
        line.proc = process_gpr(
            line.raw,
            line.time_ns,
            dewow_window_ns=self.dewow_window.value(),
            bg_window=self.bg_window.value(),
            do_bg=self.bg_remove.isChecked(),
            do_bp=self.bandpass.isChecked(),
            low_mhz=self.low_cut.value(),
            high_mhz=self.high_cut.value(),
            sec_power=self.sec_power.value(),
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
    ax.set_title(f"Processed radargram — {line.name}")
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
