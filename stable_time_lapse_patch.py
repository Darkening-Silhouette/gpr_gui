# stable_time_lapse_patch.py
# Stable time-lapse GIF renderer for Schleitheim and Bulach.
# Adds controls for amplitude mode and colour-scale source.

from pathlib import Path
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QProgressDialog, QMessageBox, QApplication
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image


# -------------------------
# Generic helpers
# -------------------------

def _fmt_num(x):
    try:
        s = ("%.2f" % float(x)).rstrip("0").rstrip(".")
    except Exception:
        s = str(x)
    return s.replace(".", "p")


def _speed_value(self):
    try:
        return float(self.tl_speed.currentText().replace("x", ""))
    except Exception:
        return 1.0


def _step_value(self, default=5.0):
    try:
        v = float(self.tl_step_ns.value())
        return v if v > 0 else default
    except Exception:
        return default


def _frames(self, default_tmax, default_step):
    try:
        tmax = float(self.proj_tmax.value())
    except Exception:
        tmax = default_tmax
    if not np.isfinite(tmax) or tmax <= 0:
        tmax = default_tmax
    step = _step_value(self, default_step)
    vals = list(np.arange(0.0, tmax + 0.5 * step, step, dtype=float))
    if not vals or vals[-1] < tmax:
        vals.append(float(tmax))
    return vals, float(tmax), float(step)


def _amp_mode(self):
    try:
        txt = self.tl_amp_mode.currentText().strip().lower()
    except Exception:
        txt = "relative amplitude"
    return "normalised" if "normal" in txt else "relative"


def _scale_mode(self):
    try:
        txt = self.tl_scale_mode.currentText().strip().lower()
    except Exception:
        txt = "whole dataset"
    return "current" if "current" in txt or "slice" in txt else "dataset"


def _normalise_values(vals, symmetric=True):
    vals = np.asarray(vals, float)
    good = np.isfinite(vals)
    out = np.full(vals.shape, np.nan, dtype=float)
    if not np.any(good):
        return out, 1.0
    if symmetric:
        scale = float(np.nanpercentile(np.abs(vals[good]), 98.0))
    else:
        scale = float(np.nanpercentile(vals[good], 98.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out[good] = vals[good] / scale
    return out, scale


def _progress_axis(pax, idx, n, time_ns, loop_no=0, color="tab:blue"):
    from matplotlib.patches import Rectangle
    pax.clear()
    pax.set_xlim(0.0, 1.0)
    pax.set_ylim(0.0, 1.0)
    pax.axis("off")
    frac = 1.0 if n <= 1 else float(idx) / float(n - 1)
    pax.add_patch(Rectangle((0.02, 0.27), 0.96, 0.46, fill=False, linewidth=1.0, edgecolor="black"))
    pax.add_patch(Rectangle((0.02, 0.27), 0.96 * frac, 0.46, color=color, alpha=0.70))
    pax.text(
        0.5, 0.5,
        f"{float(time_ns):.1f} ns | frame {int(idx) + 1}/{int(n)} | loop {int(loop_no) + 1}",
        ha="center", va="center", fontsize=9, color="black"
    )


def _fig_axes(fig):
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    fig.clear()
    ax = fig.add_axes([0.075, 0.22, 0.70, 0.63])
    cax = fig.add_axes([0.825, 0.22, 0.026, 0.63])
    pax = fig.add_axes([0.075, 0.08, 0.776, 0.045])
    return ax, cax, pax


def _to_pil(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)


def _save_gif_from_frames(out, pil_frames, speed):
    if not pil_frames:
        raise RuntimeError("No GIF frames were rendered.")
    duration = max(40, int(round(250.0 / max(float(speed), 1e-9))))
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )


def _add_tl_extra_controls(self):
    """Add amplitude/scale dropdowns to either Schleitheim or Bulach time-lapse panel."""
    try:
        if getattr(self, "_tl_amp_controls_added", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
        row = QWidget(self)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)

        box.addWidget(QLabel("Amplitude mode"))
        self.tl_amp_mode = QComboBox(row)
        self.tl_amp_mode.addItems(["Relative amplitude", "Normalised amplitude"])
        self.tl_amp_mode.setCurrentText("Relative amplitude")
        box.addWidget(self.tl_amp_mode)

        box.addWidget(QLabel("Colour scale"))
        self.tl_scale_mode = QComboBox(row)
        self.tl_scale_mode.addItems(["Whole dataset", "Current time slice"])
        self.tl_scale_mode.setCurrentText("Whole dataset")
        box.addWidget(self.tl_scale_mode)
        box.addStretch(1)

        lay = self.layout()
        if lay is not None:
            # Put it below the existing time-lapse step/playback row when possible.
            insert_at = 2 if lay.count() >= 2 else lay.count()
            lay.insertWidget(insert_at, row)
        self._tl_amp_controls_added = True
    except Exception as e:
        print("Time-lapse amplitude controls not added:", e)


# -------------------------
# Schleitheim / MALA
# -------------------------

def _sch_extent(self):
    xs, ys = [], []
    try:
        lines = self.selected_lines()
    except Exception:
        lines = getattr(getattr(self, "main", None), "lines", [])

    for line in lines:
        x = getattr(line, "x", None)
        y = getattr(line, "y", None)
        if x is None or y is None:
            continue
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        xs.extend(x[m].tolist())
        ys.extend(y[m].tolist())

    if not xs or not ys:
        try:
            x, y, _ = self.collect_slice_points(float(self.proj_tmin.value()))
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            m = np.isfinite(x) & np.isfinite(y)
            xs.extend(x[m].tolist())
            ys.extend(y[m].tolist())
        except Exception:
            xs, ys = [-1.0, 1.0], [-1.0, 1.0]

    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    px = max((xmax - xmin) * 0.03, 0.5)
    py = max((ymax - ymin) * 0.08, 0.5)
    return xmin - px, xmax + px, ymin - py, ymax + py


def _sch_values_at_time(self, time_ns):
    try:
        x, y, val = self.collect_slice_points(float(time_ns))
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        val = np.asarray(val, float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
        return x[good], y[good], val[good]
    except Exception:
        return np.asarray([]), np.asarray([]), np.asarray([])


def _sch_global_scale(self, frames):
    vals = []
    use = frames
    if len(frames) > 18:
        use = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 18).astype(int)]
    for t in use:
        _, _, v = _sch_values_at_time(self, float(t))
        if len(v):
            vals.append(v[np.isfinite(v)])
    vals = [v for v in vals if len(v)]
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    try:
        pct = float(self.main.clip.value())
    except Exception:
        pct = 99.0
    amp = float(np.nanpercentile(np.abs(allv), pct))
    if not np.isfinite(amp) or amp <= 0:
        amp = 1.0
    return amp


def _sch_vlim(self, frames):
    if _amp_mode(self) == "normalised":
        return -1.0, 1.0
    amp = _sch_global_scale(self, frames)
    return -amp, amp


def _sch_prepare_values(self, raw_val, frames=None):
    raw_val = np.asarray(raw_val, float)
    if _amp_mode(self) == "normalised":
        if _scale_mode(self) == "dataset":
            scale = getattr(self, "_tl_norm_scale", None)
            if scale is None:
                scale = _sch_global_scale(self, frames or getattr(self, "_tl_frames", [0.0]))
                self._tl_norm_scale = scale
        else:
            scale = float(np.nanpercentile(np.abs(raw_val[np.isfinite(raw_val)]), 98.0)) if np.any(np.isfinite(raw_val)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        return raw_val / scale, (-1.0, 1.0), "Normalised amplitude"

    if _scale_mode(self) == "current":
        amp = float(np.nanpercentile(np.abs(raw_val[np.isfinite(raw_val)]), 98.0)) if np.any(np.isfinite(raw_val)) else 1.0
        if not np.isfinite(amp) or amp <= 0:
            amp = 1.0
        return raw_val, (-amp, amp), "Relative amplitude"

    # Relative amplitude, global whole-dataset scale.
    return raw_val, getattr(self, "_tl_vlim", None) or (-_sch_global_scale(self, frames or [0.0]), _sch_global_scale(self, frames or [0.0])), "Relative amplitude"


def _sch_draw_frame(self, fig, time_ns, idx=0, n=1, loop_no=0, extent=None, vlim=None, title_prefix="Schleitheim time-lapse map"):
    ax, cax, pax = _fig_axes(fig)
    if extent is None:
        extent = _sch_extent(self)
    xmin, xmax, ymin, ymax = extent

    x, y, raw_val = _sch_values_at_time(self, float(time_ns))
    val, local_vlim, label = _sch_prepare_values(self, raw_val, getattr(self, "_tl_frames", [float(time_ns)]))
    vmin, vmax = vlim if (vlim is not None and _scale_mode(self) == "dataset") else local_vlim

    im = None
    if len(val) >= 4:
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(xmin, xmax, 320)
            yi = np.linspace(ymin, ymax, 180)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), val, (X, Y), method="linear")
            im = ax.imshow(
                Z,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap=self.main.cmap.currentText(),
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
        except Exception:
            im = ax.scatter(x, y, c=val, s=5, cmap=self.main.cmap.currentText(), vmin=vmin, vmax=vmax)
    else:
        ax.text(0.5, 0.5, f"Not enough points at {float(time_ns):.1f} ns", transform=ax.transAxes, ha="center", va="center")

    try:
        self.add_abcd_2d(ax)
    except Exception:
        pass

    scale_text = "whole-dataset scale" if _scale_mode(self) == "dataset" else "current-slice scale"
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.set_title(f"{title_prefix}: {float(time_ns):.1f} ns | {label}, {scale_text}")
    ax.grid(True, alpha=0.25)

    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    else:
        cax.axis("off")

    _progress_axis(pax, idx, n, time_ns, loop_no, color="tab:blue")


def _sch_draw_canvas_frame(self, time_ns, idx, n, loop_no=0):
    canvas = self.depth_canvas
    fig = canvas.fig
    extent = getattr(self, "_tl_extent", None) or _sch_extent(self)
    vlim = getattr(self, "_tl_vlim", None) if _scale_mode(self) == "dataset" else None
    _sch_draw_frame(self, fig, float(time_ns), int(idx), int(n), int(loop_no), extent, vlim)
    try:
        canvas.remember_home()
    except Exception:
        pass
    canvas.draw_idle()


def _sch_tick(self):
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
    _sch_draw_canvas_frame(self, frames[idx], idx, n, loop_no)
    try:
        self.tl_progress.setRange(0, max(0, n - 1))
        self.tl_progress.setValue(idx)
        self.tl_progress.setFormat(f"{frames[idx]:.1f} ns  |  frame {idx + 1}/{n}  |  loop {loop_no + 1}")
    except Exception:
        pass
    self._tl_idx = idx + 1


def _sch_start_preview(self, frames):
    try:
        if getattr(self, "_tl_timer", None) is not None:
            self._tl_timer.stop()
    except Exception:
        pass
    self._tl_frames = [float(x) for x in frames]
    self._tl_idx = 0
    self._tl_loop = 0
    self._tl_extent = _sch_extent(self)
    self._tl_norm_scale = None
    self._tl_vlim = _sch_vlim(self, self._tl_frames) if _scale_mode(self) == "dataset" else None
    timer = QTimer(self)
    timer.timeout.connect(lambda: _sch_tick(self))
    self._tl_timer = timer
    _sch_tick(self)
    timer.start(max(40, int(round(250.0 / _speed_value(self)))))


def _sch_plot_time_lapse_map(self):
    frames, tmax, step = _frames(self, default_tmax=180.0, default_step=5.0)
    speed = _speed_value(self)
    n = len(frames)

    assets = Path(__file__).resolve().parent / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / (
        f"Schleitheim_time_lapse_0_to_{_fmt_num(tmax)}ns_step_{_fmt_num(step)}ns_"
        f"speed_{_fmt_num(speed)}x_{_amp_mode(self)}_{_scale_mode(self)}_stable.gif"
    )

    extent = _sch_extent(self)
    self._tl_extent = extent
    self._tl_frames = [float(x) for x in frames]
    self._tl_norm_scale = None
    vlim = _sch_vlim(self, frames) if _scale_mode(self) == "dataset" else None
    self._tl_vlim = vlim

    dlg = QProgressDialog("Building stable Schleitheim time-lapse GIF...", "Cancel", 0, n, self)
    dlg.setWindowTitle("Schleitheim time-lapse map")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    QApplication.processEvents()

    pil_frames = []
    try:
        for i, tns in enumerate(frames):
            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")
            dlg.setLabelText(f"Rendering stable frame {i + 1}/{n}: {float(tns):.1f} ns")
            dlg.setValue(i)
            QApplication.processEvents()
            fig = Figure(figsize=(10, 6), dpi=120, constrained_layout=False)
            _sch_draw_frame(self, fig, float(tns), i, n, 0, extent, vlim)
            pil_frames.append(_to_pil(fig))

        _save_gif_from_frames(out, pil_frames, speed)
        self._tl_last_gif = str(out)
        _sch_start_preview(self, frames)
        self.main.status.setText(f"Saved stable Schleitheim time-lapse GIF: {out}")
        QMessageBox.information(self, "Stable time-lapse GIF saved", f"Saved:\n{out}")
    except Exception as e:
        self.main.status.setText(f"Stable time-lapse GIF failed: {e}")
        QMessageBox.critical(self, "Stable time-lapse GIF failed", str(e))
    finally:
        dlg.close()


def apply_schleitheim(ns):
    cls = ns.get("GPR3DStandardAnalysisTab") or ns.get("GPR3DAnalysisTab")
    if cls is not None:
        old_init = getattr(cls, "__init__", None)
        if old_init is not None and not getattr(cls, "_tl_amp_init_wrapped", False):
            def _init_with_amp_controls(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                _add_tl_extra_controls(self)
            cls.__init__ = _init_with_amp_controls
            cls._tl_amp_init_wrapped = True
        cls.plot_time_lapse_map = _sch_plot_time_lapse_map
    ns["_schl_tl_draw_canvas_frame"] = _sch_draw_canvas_frame
    ns["_schl_tl_start_preview"] = _sch_start_preview
    ns["_schl_tl_tick"] = _sch_tick


# -------------------------
# Bulach / PulseEKKO
# -------------------------

def _pe_extent(self):
    xs, ys = [], []
    try:
        lines = self.selected_lines()
    except Exception:
        lines = getattr(getattr(self, "owner", None), "lines", [])

    for line in lines:
        x = getattr(line, "x", None)
        y = getattr(line, "y", None)
        if x is None or y is None:
            continue
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        xs.extend(x[m].tolist())
        ys.extend(y[m].tolist())

    if not xs or not ys:
        xs, ys = [-1.0, 1.0], [-1.0, 1.0]

    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    px = max((xmax - xmin) * 0.03, 0.5)
    py = max((ymax - ymin) * 0.08, 0.5)
    return xmin - px, xmax + px, ymin - py, ymax + py


def _pe_values_at_time(self, time_ns):
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
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.abs(np.asarray(v, float))
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    return x[good], y[good], v[good]


def _pe_global_scale(self, frames):
    vals = []
    use = frames
    if len(frames) > 18:
        use = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 18).astype(int)]
    for t in use:
        _, _, v = _pe_values_at_time(self, float(t))
        if len(v):
            vals.append(v[np.isfinite(v)])
    vals = [v for v in vals if len(v)]
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    vmax = float(np.nanpercentile(allv, 98.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def _pe_vlim(self, frames):
    if _amp_mode(self) == "normalised":
        return 0.0, 1.0
    return 0.0, _pe_global_scale(self, frames)


def _pe_prepare_values(self, raw_v, frames=None):
    raw_v = np.asarray(raw_v, float)
    if _amp_mode(self) == "normalised":
        if _scale_mode(self) == "dataset":
            scale = getattr(self, "_tl_norm_scale", None)
            if scale is None:
                scale = _pe_global_scale(self, frames or getattr(self, "_tl_frames", [0.0]))
                self._tl_norm_scale = scale
        else:
            scale = float(np.nanpercentile(raw_v[np.isfinite(raw_v)], 98.0)) if np.any(np.isfinite(raw_v)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        return raw_v / scale, (0.0, 1.0), "Normalised absolute amplitude"

    if _scale_mode(self) == "current":
        scale = float(np.nanpercentile(raw_v[np.isfinite(raw_v)], 98.0)) if np.any(np.isfinite(raw_v)) else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return raw_v, (0.0, scale), "Relative absolute amplitude"

    return raw_v, getattr(self, "_tl_vlim", None) or (0.0, _pe_global_scale(self, frames or [0.0])), "Relative absolute amplitude"


def _pe_draw_frame(self, fig, time_ns, idx=0, n=1, loop_no=0, extent=None, vlim=None, title_prefix="Bulach time-lapse map"):
    ax, cax, pax = _fig_axes(fig)
    if extent is None:
        extent = _pe_extent(self)
    xmin, xmax, ymin, ymax = extent

    x, y, raw_v = _pe_values_at_time(self, float(time_ns))
    v, local_vlim, label = _pe_prepare_values(self, raw_v, getattr(self, "_tl_frames", [float(time_ns)]))
    vmin, vmax = vlim if (vlim is not None and _scale_mode(self) == "dataset") else local_vlim

    im = None
    if len(v) >= 4:
        try:
            import GPR_Fieldwork_Analysis as _pe_mod
            scipy_ok = bool(getattr(_pe_mod, "SCIPY_OK", False))
            griddata = getattr(_pe_mod, "griddata", None)
        except Exception:
            scipy_ok = False
            griddata = None

        try:
            if scipy_ok and griddata is not None and len(v) > 100:
                gx = np.linspace(xmin, xmax, 320)
                gy = np.linspace(ymin, ymax, 180)
                X, Y = np.meshgrid(gx, gy)
                Z = griddata((x, y), v, (X, Y), method="linear")
                im = ax.imshow(
                    Z,
                    extent=[xmin, xmax, ymin, ymax],
                    origin="lower",
                    aspect="equal",
                    cmap="inferno",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
            else:
                im = ax.scatter(x, y, c=v, s=5, cmap="inferno", vmin=vmin, vmax=vmax)
        except Exception:
            im = ax.scatter(x, y, c=v, s=5, cmap="inferno", vmin=vmin, vmax=vmax)
    else:
        ax.text(0.5, 0.5, f"No data at {float(time_ns):.1f} ns", transform=ax.transAxes, ha="center", va="center")

    scale_text = "whole-dataset scale" if _scale_mode(self) == "dataset" else "current-slice scale"
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local easting [m]")
    ax.set_ylabel("Local northing [m]")
    ax.set_title(f"{title_prefix}: {float(time_ns):.1f} ns | {label}, {scale_text}")
    ax.grid(True, alpha=0.2)

    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    else:
        cax.axis("off")

    _progress_axis(pax, idx, n, time_ns, loop_no, color="tab:orange")


def _pe_current_canvas(self):
    try:
        return self.selected_canvas()[1]
    except Exception:
        return getattr(self, "depth_canvas", None)


def _pe_draw_canvas_frame(self, time_ns, idx, n, loop_no=0):
    canvas = _pe_current_canvas(self)
    if canvas is None:
        return
    fig = canvas.fig
    extent = getattr(self, "_tl_extent", None) or _pe_extent(self)
    vlim = getattr(self, "_tl_vlim", None) if _scale_mode(self) == "dataset" else None
    _pe_draw_frame(self, fig, float(time_ns), int(idx), int(n), int(loop_no), extent, vlim)
    canvas.draw_idle()


def _pe_tick(self):
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
    _pe_draw_canvas_frame(self, frames[idx], idx, n, loop_no)
    try:
        self.tl_progress.setRange(0, max(0, n - 1))
        self.tl_progress.setValue(idx)
        self.tl_progress.setFormat(f"{frames[idx]:.1f} ns  |  frame {idx + 1}/{n}  |  loop {loop_no + 1}")
    except Exception:
        pass
    self._tl_idx = idx + 1


def _pe_start_preview(self, frames):
    try:
        if getattr(self, "_tl_timer", None) is not None:
            self._tl_timer.stop()
    except Exception:
        pass
    self._tl_frames = [float(x) for x in frames]
    self._tl_idx = 0
    self._tl_loop = 0
    self._tl_extent = _pe_extent(self)
    self._tl_norm_scale = None
    self._tl_vlim = _pe_vlim(self, self._tl_frames) if _scale_mode(self) == "dataset" else None
    timer = QTimer(self)
    timer.timeout.connect(lambda: _pe_tick(self))
    self._tl_timer = timer
    _pe_tick(self)
    timer.start(max(40, int(round(250.0 / _speed_value(self)))))


def _pe_plot_time_lapse_map(self):
    frames, tmax, step = _frames(self, default_tmax=50.0, default_step=2.5)
    speed = _speed_value(self)
    n = len(frames)

    assets = Path(__file__).resolve().parent / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / (
        f"Bulach_time_lapse_0_to_{_fmt_num(tmax)}ns_step_{_fmt_num(step)}ns_"
        f"speed_{_fmt_num(speed)}x_{_amp_mode(self)}_{_scale_mode(self)}_stable.gif"
    )

    extent = _pe_extent(self)
    self._tl_extent = extent
    self._tl_frames = [float(x) for x in frames]
    self._tl_norm_scale = None
    vlim = _pe_vlim(self, frames) if _scale_mode(self) == "dataset" else None
    self._tl_vlim = vlim

    dlg = QProgressDialog("Building stable Bulach time-lapse GIF...", "Cancel", 0, n, self)
    dlg.setWindowTitle("Bulach time-lapse map")
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    QApplication.processEvents()

    pil_frames = []
    try:
        for i, tns in enumerate(frames):
            if dlg.wasCanceled():
                raise RuntimeError("Cancelled by user.")
            dlg.setLabelText(f"Rendering stable frame {i + 1}/{n}: {float(tns):.1f} ns")
            dlg.setValue(i)
            QApplication.processEvents()
            fig = Figure(figsize=(10, 6), dpi=120, constrained_layout=False)
            _pe_draw_frame(self, fig, float(tns), i, n, 0, extent, vlim)
            pil_frames.append(_to_pil(fig))

        _save_gif_from_frames(out, pil_frames, speed)
        self._tl_last_gif = str(out)
        _pe_start_preview(self, frames)
        self.owner.status.setText(f"Saved stable Bulach time-lapse GIF: {out}")
        QMessageBox.information(self, "Stable time-lapse GIF saved", f"Saved:\n{out}")
    except Exception as e:
        self.owner.status.setText(f"Stable time-lapse GIF failed: {e}")
        QMessageBox.critical(self, "Stable time-lapse GIF failed", str(e))
    finally:
        dlg.close()


def apply_bulach(ns):
    cls = ns.get("PulseEkko3DAnalysis")
    if cls is not None:
        old_init = getattr(cls, "__init__", None)
        if old_init is not None and not getattr(cls, "_tl_amp_init_wrapped", False):
            def _init_with_amp_controls(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                _add_tl_extra_controls(self)
            cls.__init__ = _init_with_amp_controls
            cls._tl_amp_init_wrapped = True
        cls.plot_time_lapse_map = _pe_plot_time_lapse_map
    ns["_pe_tl_draw_canvas_frame"] = _pe_draw_canvas_frame
    ns["_pe_tl_start_preview"] = _pe_start_preview
    ns["_pe_tl_tick"] = _pe_tick



# --- app-only time-lapse hover + zoom patch ---
def _tl_status(obj, msg):
    for target in (obj, getattr(obj, "main", None), getattr(obj, "owner", None)):
        try:
            st = getattr(target, "status", None)
            if st is not None:
                st.setText(msg)
                return
        except Exception:
            pass

def _tl_line_label(line):
    try:
        parent = line.folder.parent.name
        return f"{parent}/line_{int(line.number)}_{line.direction}_t"
    except Exception:
        return str(getattr(line, "name", line))

def _tl_get_lines(obj):
    for name in ("selected_lines_for_maps", "selected_lines"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                pass
    try:
        return list(obj.owner.lines)
    except Exception:
        return []

def _tl_line_xy(obj, line):
    import numpy as np
    try:
        x = np.asarray(line.x, float); y = np.asarray(line.y, float)
        if x.size >= 2 and y.size >= 2:
            return x, y
    except Exception:
        pass
    try:
        n = int(getattr(line, "traces", 100) or 100)
        x, y, _ = obj.trace_xyz(line, n)
        return np.asarray(x, float), np.asarray(y, float)
    except Exception:
        return None, None

def _tl_nearest_line(obj, ax, event):
    import numpy as np
    if event.xdata is None or event.ydata is None:
        return None, None
    p = np.array([event.x, event.y], float)
    best = (1e99, None)
    for line in _tl_get_lines(obj):
        x, y = _tl_line_xy(obj, line)
        if x is None or y is None or len(x) < 2:
            continue
        # Use endpoints: survey lines are effectively straight in the map.
        pts = ax.transData.transform(np.array([[x[0], y[0]], [x[-1], y[-1]]], float))
        a, b = pts[0], pts[1]
        ab = b - a
        den = float(np.dot(ab, ab))
        if den <= 0:
            continue
        t = max(0.0, min(1.0, float(np.dot(p - a, ab) / den)))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            best = (d, line)
    if best[1] is not None and best[0] <= 12.0:
        return best[1], best[0]
    return None, None

def _tl_scroll_zoom(obj, canvas, event):
    ax = event.inaxes
    if ax is None or event.xdata is None or event.ydata is None:
        return
    scale = 1.0 / 1.25 if event.button == "up" else 1.25
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = event.xdata, event.ydata
    ax.set_xlim(x + (x0 - x) * scale, x + (x1 - x) * scale)
    ax.set_ylim(y + (y0 - y) * scale, y + (y1 - y) * scale)
    canvas.draw_idle()

def _tl_hover_line(obj, canvas, event):
    ax = event.inaxes
    if ax is None:
        return
    line, dist = _tl_nearest_line(obj, ax, event)
    ann = getattr(canvas, "_tl_hover_annotation", None)
    if ann is None or ann.axes is not ax:
        try:
            ann = ax.annotate("", xy=(0,0), xytext=(12,12), textcoords="offset points", bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85), arrowprops=dict(arrowstyle="->", color="black"))
            ann.set_visible(False)
            canvas._tl_hover_annotation = ann
        except Exception:
            return
    if line is None:
        if ann.get_visible():
            ann.set_visible(False); canvas.draw_idle()
        return
    label = _tl_line_label(line)
    ann.xy = (event.xdata, event.ydata)
    ann.set_text(label)
    ann.set_visible(True)
    _tl_status(obj, f"Hover line: {label}")
    canvas.draw_idle()

def _tl_install_hover_zoom(obj):
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        return
    canvases = []
    try:
        canvases += list(obj.findChildren(FigureCanvas))
    except Exception:
        pass
    for name in ("time_canvas", "depth_canvas", "proj_canvas"):
        c = getattr(obj, name, None)
        if c is not None:
            canvases.append(c)
    d = getattr(obj, "canvases", None)
    if isinstance(d, dict):
        canvases += [c for c in d.values() if c is not None]
    seen = set()
    for c in canvases:
        if id(c) in seen or getattr(c, "_tl_hover_zoom_installed", False):
            continue
        seen.add(id(c))
        c._tl_hover_zoom_installed = True
        c.mpl_connect("scroll_event", lambda ev, obj=obj, canvas=c: _tl_scroll_zoom(obj, canvas, ev))
        c.mpl_connect("motion_notify_event", lambda ev, obj=obj, canvas=c: _tl_hover_line(obj, canvas, ev))

_old_apply_schleitheim_hover = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_hover:
        _old_apply_schleitheim_hover(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab"):
        cls = globs.get(n)
        if cls is not None and not getattr(cls, "_tl_hover_zoom_class_patched", False):
            old = cls.__init__
            def new_init(self, *a, _old=old, **kw):
                _old(self, *a, **kw)
                _tl_install_hover_zoom(self)
            cls.__init__ = new_init
            cls._tl_hover_zoom_class_patched = True

_old_apply_bulach_hover = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_hover:
        _old_apply_bulach_hover(globs)
    cls = globs.get("PulseEkko3DAnalysis")
    if cls is not None and not getattr(cls, "_tl_hover_zoom_class_patched", False):
        old = cls.__init__
        def new_init(self, *a, _old=old, **kw):
            _old(self, *a, **kw)
            _tl_install_hover_zoom(self)
        cls.__init__ = new_init
        cls._tl_hover_zoom_class_patched = True
# --- end app-only time-lapse hover + zoom patch ---




# --- no-redraw hover tooltip patch ---
def _tl_hover_line(obj, canvas, event):
    # Do not draw Matplotlib annotations here. Redrawing the canvas during
    # animation clears the current animated frame and makes the plot disappear.
    ax = event.inaxes
    if ax is None:
        return
    line, dist = _tl_nearest_line(obj, ax, event)
    if line is None:
        return
    label = _tl_line_label(line)
    _tl_status(obj, f"Hover line: {label}")
    try:
        from PyQt6.QtWidgets import QToolTip
        gp = event.guiEvent.globalPosition().toPoint() if hasattr(event.guiEvent, "globalPosition") else event.guiEvent.globalPos()
        QToolTip.showText(gp, label, canvas)
    except Exception:
        pass
# --- end no-redraw hover tooltip patch ---




# --- persistent animation zoom patch ---
def _tl_pick_zoom_ax(canvas):
    ax = getattr(canvas, "_tl_zoom_ax", None)
    if ax is not None and ax in canvas.figure.axes:
        return ax
    best = None
    best_area = -1
    for a in canvas.figure.axes:
        try:
            bb = a.get_position()
            area = bb.width * bb.height
            xlabel = (a.get_xlabel() or "").lower()
            ylabel = (a.get_ylabel() or "").lower()
            if "easting" in xlabel or "northing" in ylabel:
                return a
            if area > best_area:
                best = a; best_area = area
        except Exception:
            pass
    return best

def _tl_apply_saved_zoom(canvas, redraw=False):
    if not getattr(canvas, "_tl_zoom_active", False):
        return
    xlim = getattr(canvas, "_tl_zoom_xlim", None)
    ylim = getattr(canvas, "_tl_zoom_ylim", None)
    if xlim is None or ylim is None:
        return
    ax = _tl_pick_zoom_ax(canvas)
    if ax is None:
        return
    try:
        ax.set_autoscale_on(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if redraw:
            canvas.draw_idle()
    except Exception:
        pass

def _tl_on_draw_keep_zoom(canvas, event=None):
    if getattr(canvas, "_tl_zoom_guard", False):
        return
    if not getattr(canvas, "_tl_zoom_active", False):
        return
    ax = _tl_pick_zoom_ax(canvas)
    if ax is None:
        return
    xlim = getattr(canvas, "_tl_zoom_xlim", None)
    ylim = getattr(canvas, "_tl_zoom_ylim", None)
    if xlim is None or ylim is None:
        return
    try:
        current = ax.get_xlim() + ax.get_ylim()
        target = tuple(xlim) + tuple(ylim)
        if max(abs(current[i]-target[i]) for i in range(4)) < 1e-9:
            return
        canvas._tl_zoom_guard = True
        ax.set_autoscale_on(False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        canvas.draw_idle()
    except Exception:
        pass
    finally:
        canvas._tl_zoom_guard = False

def _tl_scroll_zoom(obj, canvas, event):
    ax = event.inaxes
    if ax is None or event.xdata is None or event.ydata is None:
        return
    # Ignore colourbar/progress axes; only zoom the map axis.
    xlabel = (ax.get_xlabel() or "").lower()
    ylabel = (ax.get_ylabel() or "").lower()
    if not ("easting" in xlabel or "northing" in ylabel):
        main_ax = _tl_pick_zoom_ax(canvas)
        if main_ax is not None:
            ax = main_ax
        else:
            return
    scale = 1.0 / 1.25 if event.button == "up" else 1.25
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = event.xdata, event.ydata
    new_xlim = (x + (x0 - x) * scale, x + (x1 - x) * scale)
    new_ylim = (y + (y0 - y) * scale, y + (y1 - y) * scale)
    canvas._tl_zoom_active = True
    canvas._tl_zoom_ax = ax
    canvas._tl_zoom_xlim = new_xlim
    canvas._tl_zoom_ylim = new_ylim
    _tl_apply_saved_zoom(canvas, redraw=True)

def _tl_install_hover_zoom(obj):
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        return
    canvases = []
    try:
        canvases += list(obj.findChildren(FigureCanvas))
    except Exception:
        pass
    for name in ("time_canvas", "depth_canvas", "proj_canvas"):
        c = getattr(obj, name, None)
        if c is not None:
            canvases.append(c)
    d = getattr(obj, "canvases", None)
    if isinstance(d, dict):
        canvases += [c for c in d.values() if c is not None]
    seen=set()
    for c in canvases:
        if id(c) in seen:
            continue
        seen.add(id(c))
        if not getattr(c, "_tl_hover_zoom_installed", False):
            c._tl_hover_zoom_installed = True
            c.mpl_connect("scroll_event", lambda ev, obj=obj, canvas=c: _tl_scroll_zoom(obj, canvas, ev))
            c.mpl_connect("motion_notify_event", lambda ev, obj=obj, canvas=c: _tl_hover_line(obj, canvas, ev))
        if not getattr(c, "_tl_zoom_draw_hook_installed", False):
            c._tl_zoom_draw_hook_installed = True
            c.mpl_connect("draw_event", lambda ev, canvas=c: _tl_on_draw_keep_zoom(canvas, ev))
# --- end persistent animation zoom patch ---




# --- app zoom fill canvas patch ---
def _tl_scroll_zoom(obj, canvas, event):
    ax = event.inaxes
    if ax is None or event.xdata is None or event.ydata is None:
        return
    xlabel = (ax.get_xlabel() or "").lower()
    ylabel = (ax.get_ylabel() or "").lower()
    if not ("easting" in xlabel or "northing" in ylabel):
        main_ax = _tl_pick_zoom_ax(canvas)
        if main_ax is not None:
            ax = main_ax
        else:
            return
    scale = 1.0 / 1.25 if event.button == "up" else 1.25
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = event.xdata, event.ydata
    new_xlim = (x + (x0 - x) * scale, x + (x1 - x) * scale)
    new_ylim = (y + (y0 - y) * scale, y + (y1 - y) * scale)
    canvas._tl_zoom_active = True
    canvas._tl_zoom_ax = ax
    canvas._tl_zoom_xlim = new_xlim
    canvas._tl_zoom_ylim = new_ylim
    try:
        ax.set_aspect("auto")
        ax.set_position([0.08, 0.18, 0.74, 0.68])
    except Exception:
        pass
    _tl_apply_saved_zoom(canvas, redraw=True)

def _tl_apply_saved_zoom(canvas, redraw=False):
    if not getattr(canvas, "_tl_zoom_active", False):
        return
    xlim = getattr(canvas, "_tl_zoom_xlim", None)
    ylim = getattr(canvas, "_tl_zoom_ylim", None)
    if xlim is None or ylim is None:
        return
    ax = _tl_pick_zoom_ax(canvas)
    if ax is None:
        return
    try:
        ax.set_autoscale_on(False)
        ax.set_aspect("auto")
        ax.set_position([0.08, 0.18, 0.74, 0.68])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if redraw:
            canvas.draw_idle()
    except Exception:
        pass
# --- end app zoom fill canvas patch ---




# --- disable time-lapse zoom keep hover patch ---
def _tl_scroll_zoom(obj, canvas, event):
    # Zoom disabled. Keep time-lapse app view stable during animation.
    return

def _tl_apply_saved_zoom(canvas, redraw=False):
    # Zoom disabled. Do not reapply stored zoom limits.
    return

def _tl_on_draw_keep_zoom(canvas, event=None):
    # Zoom disabled. Do not force axes limits after redraw.
    return

def _tl_install_hover_zoom(obj):
    # Install hover only; no scroll zoom.
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        return
    canvases = []
    try:
        canvases += list(obj.findChildren(FigureCanvas))
    except Exception:
        pass
    for name in ("time_canvas", "depth_canvas", "proj_canvas"):
        c = getattr(obj, name, None)
        if c is not None:
            canvases.append(c)
    d = getattr(obj, "canvases", None)
    if isinstance(d, dict):
        canvases += [c for c in d.values() if c is not None]
    seen = set()
    for c in canvases:
        if id(c) in seen:
            continue
        seen.add(id(c))
        if not getattr(c, "_tl_hover_only_installed", False):
            c._tl_hover_only_installed = True
            c.mpl_connect("motion_notify_event", lambda ev, obj=obj, canvas=c: _tl_hover_line(obj, canvas, ev))
# --- end disable time-lapse zoom keep hover patch ---




# --- hard disable all map scroll callbacks keep hover patch ---
def _tl_disconnect_scroll_callbacks(canvas):
    try:
        reg = canvas.callbacks.callbacks.get("scroll_event", {})
        for cid in list(reg.keys()):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
    except Exception:
        pass
    try:
        canvas._tl_zoom_active = False
        canvas._tl_zoom_xlim = None
        canvas._tl_zoom_ylim = None
        canvas._tl_zoom_ax = None
    except Exception:
        pass

def _tl_scroll_zoom(obj, canvas, event):
    return

def _tl_apply_saved_zoom(canvas, redraw=False):
    return

def _tl_on_draw_keep_zoom(canvas, event=None):
    return

def _tl_install_hover_zoom(obj):
    # Hover only. Remove all scroll zoom callbacks, including MplCanvas default zoom.
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        return
    canvases = []
    try:
        canvases += list(obj.findChildren(FigureCanvas))
    except Exception:
        pass
    for name in ("time_canvas", "depth_canvas", "proj_canvas"):
        c = getattr(obj, name, None)
        if c is not None:
            canvases.append(c)
    d = getattr(obj, "canvases", None)
    if isinstance(d, dict):
        canvases += [c for c in d.values() if c is not None]
    seen = set()
    for c in canvases:
        if id(c) in seen:
            continue
        seen.add(id(c))
        _tl_disconnect_scroll_callbacks(c)
        if not getattr(c, "_tl_hover_only_installed_v2", False):
            c._tl_hover_only_installed_v2 = True
            c.mpl_connect("motion_notify_event", lambda ev, obj=obj, canvas=c: _tl_hover_line(obj, canvas, ev))
# --- end hard disable all map scroll callbacks keep hover patch ---




# --- map smoothing dropdown patch ---
_TL_MAP_SMOOTHING_MODE = globals().get("_TL_MAP_SMOOTHING_MODE", "Direction-aware destripe")

def _tl_set_map_smoothing(txt):
    globals()["_TL_MAP_SMOOTHING_MODE"] = str(txt)

def _tl_get_map_smoothing(obj=None):
    try:
        cb = getattr(obj, "map_smoothing", None)
        if cb is not None:
            return cb.currentText()
    except Exception:
        pass
    return globals().get("_TL_MAP_SMOOTHING_MODE", "Direction-aware destripe")

def _tl_fill_nan_for_filter(z):
    import numpy as np
    z = np.asarray(z, float)
    mask = ~np.isfinite(z)
    if np.all(mask):
        return z.copy(), mask
    fill = float(np.nanmedian(z))
    a = z.copy()
    a[mask] = fill
    return a, mask

def _tl_smooth_map_array(z, mode=None):
    import numpy as np
    z0 = np.asarray(z)
    if z0.ndim != 2 or not np.issubdtype(z0.dtype, np.number):
        return z
    mode = str(mode or globals().get("_TL_MAP_SMOOTHING_MODE", "Direction-aware destripe"))
    if mode.startswith("None"):
        return z
    a, mask = _tl_fill_nan_for_filter(z0)
    try:
        from scipy.ndimage import median_filter, gaussian_filter
    except Exception:
        median_filter = gaussian_filter = None
    out = a.copy()
    if mode.startswith("Light median"):
        if median_filter is not None:
            out = median_filter(out, size=3, mode="nearest")
    elif "Gaussian σ=0.5" in mode or "Gaussian sigma 0.5" in mode:
        if gaussian_filter is not None:
            out = gaussian_filter(out, sigma=0.5, mode="nearest")
    elif "Gaussian σ=1.0" in mode or "Gaussian sigma 1.0" in mode:
        if gaussian_filter is not None:
            out = gaussian_filter(out, sigma=1.0, mode="nearest")
    else:
        # Direction-aware destriping: remove the stronger row/column median bias,
        # then apply only very light smoothing. This reduces acquisition stripes
        # without forcing every line to identical amplitude.
        med = float(np.nanmedian(out))
        col_bias = np.nanmedian(out, axis=0) - med
        row_bias = np.nanmedian(out, axis=1) - med
        if median_filter is not None:
            kcol = min(21, max(3, (len(col_bias)//20)*2+1))
            krow = min(21, max(3, (len(row_bias)//20)*2+1))
            col_bias = median_filter(col_bias, size=kcol, mode="nearest")
            row_bias = median_filter(row_bias, size=krow, mode="nearest")
        col_strength = float(np.nanstd(col_bias)) if col_bias.size else 0.0
        row_strength = float(np.nanstd(row_bias)) if row_bias.size else 0.0
        if col_strength >= row_strength:
            out = out - 0.75 * col_bias[None, :]
        else:
            out = out - 0.75 * row_bias[:, None]
        if gaussian_filter is not None:
            out = gaussian_filter(out, sigma=0.35, mode="nearest")
    out[mask] = np.nan
    return out

def _tl_install_smoothing_imshow_patch():
    try:
        import inspect
        import matplotlib.axes
    except Exception:
        return
    if getattr(matplotlib.axes.Axes.imshow, "_tl_smoothing_patched", False):
        return
    _old_imshow = matplotlib.axes.Axes.imshow
    def _imshow(self, X, *args, **kwargs):
        try:
            # Only smooth map frames rendered through this time-lapse/map patch module.
            stack = inspect.stack()
            if any("stable_time_lapse_patch.py" in str(fr.filename) for fr in stack):
                X = _tl_smooth_map_array(X, globals().get("_TL_MAP_SMOOTHING_MODE", "Direction-aware destripe"))
        except Exception:
            pass
        return _old_imshow(self, X, *args, **kwargs)
    _imshow._tl_smoothing_patched = True
    matplotlib.axes.Axes.imshow = _imshow

def _tl_add_map_smoothing_control(obj):
    try:
        if getattr(obj, "_map_smoothing_added", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
        row = QWidget(obj)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)
        box.addWidget(QLabel("Map smoothing"))
        cb = QComboBox(row)
        cb.addItems([
            "Direction-aware destripe",
            "None / preserve data",
            "Light median 3x3",
            "Gaussian σ=0.5 cell",
            "Gaussian σ=1.0 cell",
        ])
        cb.setCurrentText("Direction-aware destripe")
        cb.currentTextChanged.connect(_tl_set_map_smoothing)
        box.addWidget(cb)
        box.addStretch(1)
        obj.map_smoothing = cb
        _tl_set_map_smoothing(cb.currentText())
        lay = obj.layout()
        if lay is not None:
            lay.insertWidget(4 if lay.count() >= 4 else lay.count(), row)
        obj._map_smoothing_added = True
    except Exception as e:
        print("Map smoothing control not added:", e)

def _tl_patch_smoothing_class(cls):
    if cls is None or getattr(cls, "_tl_smoothing_class_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _tl_add_map_smoothing_control(self)
        _tl_install_smoothing_imshow_patch()
    cls.__init__ = new_init
    cls._tl_smoothing_class_patched = True

_old_apply_schleitheim_smoothing = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    _tl_install_smoothing_imshow_patch()
    if _old_apply_schleitheim_smoothing:
        _old_apply_schleitheim_smoothing(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab"):
        _tl_patch_smoothing_class(globs.get(n))

_old_apply_bulach_smoothing = globals().get("apply_bulach")
def apply_bulach(globs):
    _tl_install_smoothing_imshow_patch()
    if _old_apply_bulach_smoothing:
        _old_apply_bulach_smoothing(globs)
    _tl_patch_smoothing_class(globs.get("PulseEkko3DAnalysis"))
# --- end map smoothing dropdown patch ---




# --- line-wise bias correction patch ---
_TL_LINE_BIAS_MODE = globals().get("_TL_LINE_BIAS_MODE", "Median-center each line")

def _tl_set_line_bias(txt):
    globals()["_TL_LINE_BIAS_MODE"] = str(txt)

def _tl_line_bias_mode(obj=None):
    try:
        cb = getattr(obj, "line_bias_correction", None)
        if cb is not None:
            return cb.currentText()
    except Exception:
        pass
    return globals().get("_TL_LINE_BIAS_MODE", "Median-center each line")

def _tl_correct_line_values(v, obj=None):
    import numpy as np
    v = np.asarray(v, float).copy()
    mode = _tl_line_bias_mode(obj)
    if mode.startswith("Off"):
        return v
    good = np.isfinite(v)
    if not np.any(good):
        return v
    med = float(np.nanmedian(v[good]))
    v[good] = v[good] - med
    if "robust line scale" in mode:
        sc = float(np.nanpercentile(np.abs(v[good]), 95.0))
        if np.isfinite(sc) and sc > 1e-12:
            v[good] = v[good] / sc
    return v

def _tl_add_line_bias_control(obj):
    try:
        if getattr(obj, "_line_bias_added", False):
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
        row = QWidget(obj)
        box = QHBoxLayout(row)
        box.setContentsMargins(4, 2, 4, 2)
        box.addWidget(QLabel("Line-bias correction"))
        cb = QComboBox(row)
        cb.addItems(["Median-center each line", "Off / preserve line amplitudes", "Median-center + robust line scale"])
        cb.setCurrentText("Median-center each line")
        cb.currentTextChanged.connect(_tl_set_line_bias)
        box.addWidget(cb)
        box.addStretch(1)
        obj.line_bias_correction = cb
        _tl_set_line_bias(cb.currentText())
        lay = obj.layout()
        if lay is not None:
            lay.insertWidget(5 if lay.count() >= 5 else lay.count(), row)
        obj._line_bias_added = True
    except Exception as e:
        print("Line-bias correction control not added:", e)

def _tl_selected_lines(obj):
    for name in ("selected_lines_for_maps", "selected_lines"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                pass
    try:
        return list(obj.owner.lines)
    except Exception:
        return []

def _tl_line_xy(obj, line, n):
    import numpy as np
    try:
        if hasattr(obj, "trace_xy"):
            x, y = obj.trace_xy(line, n)
            return np.asarray(x, float), np.asarray(y, float)
    except Exception:
        pass
    try:
        if hasattr(obj, "trace_xyz"):
            x, y, _ = obj.trace_xyz(line, n)
            return np.asarray(x, float), np.asarray(y, float)
    except Exception:
        pass
    try:
        x = np.asarray(line.x, float); y = np.asarray(line.y, float)
        if len(x) != n and len(x) > 1:
            old = np.linspace(0, 1, len(x)); new = np.linspace(0, 1, n)
            x = np.interp(new, old, x); y = np.interp(new, old, y)
        return x, y
    except Exception:
        return None, None

def _sch_values_at_time(self, time_ns):
    import numpy as np
    xs, ys, vals = [], [], []
    step = max(1, int(getattr(self.trace_step, "value", lambda: 5)()))
    for line in _tl_selected_lines(self):
        try:
            data = self.ensure_data(line)
            if data is None or data.ndim != 2:
                continue
            t = self.time_vector(line, data.shape[1])
            j = int(np.argmin(np.abs(np.asarray(t, float) - float(time_ns))))
            x, y = _tl_line_xy(self, line, data.shape[0])
            if x is None or y is None:
                continue
            n = min(data.shape[0], len(x), len(y))
            idx = np.arange(0, n, step)
            v = _tl_correct_line_values(data[idx, j], self)
            xs.extend(np.asarray(x)[idx]); ys.extend(np.asarray(y)[idx]); vals.extend(v)
        except Exception:
            continue
    x = np.asarray(xs, float); y = np.asarray(ys, float); v = np.asarray(vals, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    return x[good], y[good], v[good]

def _pe_values_at_time(self, time_ns):
    import numpy as np
    xs, ys, vals = [], [], []
    step = max(1, int(getattr(self.trace_step, "value", lambda: 5)()))
    for line in _tl_selected_lines(self):
        try:
            arr = self.get_array(line)
            t = self.owner.corrected_time_ns(line)
            j = int(np.argmin(np.abs(np.asarray(t, float) - float(time_ns))))
            n = min(arr.shape[0], len(line.x), len(line.y))
            idx = np.arange(0, n, step)
            v = np.asarray(arr[idx, j], float)
            v = _tl_correct_line_values(v, self)
            xs.extend(np.asarray(line.x)[idx]); ys.extend(np.asarray(line.y)[idx]); vals.extend(v)
        except Exception:
            continue
    x = np.asarray(xs, float); y = np.asarray(ys, float); v = np.asarray(vals, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    return x[good], y[good], v[good]

def _tl_patch_linebias_class(cls):
    if cls is None or getattr(cls, "_tl_linebias_class_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _tl_add_line_bias_control(self)
    cls.__init__ = new_init
    cls._tl_linebias_class_patched = True

_old_apply_schleitheim_linebias = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_linebias:
        _old_apply_schleitheim_linebias(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab"):
        _tl_patch_linebias_class(globs.get(n))

_old_apply_bulach_linebias = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_linebias:
        _old_apply_bulach_linebias(globs)
    _tl_patch_linebias_class(globs.get("PulseEkko3DAnalysis"))
# --- end line-wise bias correction patch ---




# --- layout and requested defaults patch ---
def _tl_text(w):
    try:
        return w.text().strip().lower()
    except Exception:
        return ""

def _tl_combo_items(cb):
    try:
        return [cb.itemText(i) for i in range(cb.count())]
    except Exception:
        return []

def _tl_set_combo_contains(cb, wanted):
    wanted = wanted.lower()
    try:
        for i in range(cb.count()):
            if wanted in cb.itemText(i).lower():
                cb.setCurrentIndex(i)
                return True
    except Exception:
        pass
    return False

def _tl_set_spin_value(spin, value):
    try:
        spin.setValue(value)
        return True
    except Exception:
        return False

def _tl_reposition_extra_controls(obj):
    # Move Map smoothing + Line-bias correction rows above the tab widget/canvas,
    # so Survey Overview and other plots are not blocked at the bottom.
    try:
        from PyQt6.QtWidgets import QTabWidget
        lay = obj.layout()
        if lay is None:
            return
        tab_idx = None
        for i in range(lay.count()):
            item = lay.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, QTabWidget):
                tab_idx = i
                break
        if tab_idx is None:
            return
        rows = []
        for attr in ("map_smoothing", "line_bias_correction"):
            cb = getattr(obj, attr, None)
            if cb is not None:
                row = cb.parentWidget()
                if row is not None and row not in rows:
                    rows.append(row)
        for row in rows:
            lay.removeWidget(row)
        for row in reversed(rows):
            lay.insertWidget(tab_idx, row)
    except Exception as e:
        print("Could not reposition extra controls:", e)

def _tl_apply_requested_defaults(obj):
    try:
        from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox
    except Exception:
        return
    try:
        _tl_reposition_extra_controls(obj)
    except Exception:
        pass
    # Defaults: time-lapse step 2.5 ns, playback 0.5x, colour scale current time slice.
    for cb in obj.findChildren(QComboBox):
        items = [x.lower() for x in _tl_combo_items(cb)]
        joined = " | ".join(items)
        if "whole dataset" in joined and "current time slice" in joined:
            _tl_set_combo_contains(cb, "current time slice")
        if "0.5x" in joined and "1.5x" in joined:
            _tl_set_combo_contains(cb, "0.5x")
    for spin in obj.findChildren(QDoubleSpinBox) + obj.findChildren(QSpinBox):
        try:
            # Time-lapse step control usually has 0.5–20 ns range and current value 2.5 or 5.
            if spin.minimum() <= 2.5 <= spin.maximum() and spin.singleStep() <= 2.5 and abs(float(spin.value()) - 5.0) <= 5.0:
                if spin.suffix().strip().lower() == "ns" and spin.maximum() <= 100:
                    spin.setValue(2.5)
        except Exception:
            pass
    # Default median/local background remover off.
    for chk in obj.findChildren(QCheckBox):
        txt = _tl_text(chk)
        if "background" in txt and ("median" in txt or "local" in txt):
            chk.setChecked(False)

def _tl_patch_defaults_class(cls):
    if cls is None or getattr(cls, "_tl_defaults_layout_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _tl_apply_requested_defaults(self)
    cls.__init__ = new_init
    cls._tl_defaults_layout_patched = True

_old_apply_schleitheim_defaults = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_defaults:
        _old_apply_schleitheim_defaults(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab", "SchleitheimProjectWidget", "MainWindow"):
        _tl_patch_defaults_class(globs.get(n))

_old_apply_bulach_defaults = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_defaults:
        _old_apply_bulach_defaults(globs)
    for n in ("PulseEkko3DAnalysis", "PulseEkkoProjectWidget", "MainWindow"):
        _tl_patch_defaults_class(globs.get(n))
# --- end layout and requested defaults patch ---




# --- real line-bias image-level correction patch ---
def _tl_linebias_mode_global():
    try:
        return globals().get("_TL_LINE_BIAS_MODE", "Median-center each line")
    except Exception:
        return "Median-center each line"

def _tl_apply_linebias_to_grid(z):
    import numpy as np
    z = np.asarray(z, float)
    mode = _tl_linebias_mode_global()
    if z.ndim != 2 or mode.startswith("Off"):
        return z
    out = z.copy()
    good = np.isfinite(out)
    if not np.any(good):
        return z
    # Remove residual survey-line stripe bias in both directions.
    # This is deliberately conservative: subtract median row/column bias, not full data.
    global_med = float(np.nanmedian(out))
    try:
        col_med = np.nanmedian(out, axis=0)
        col_bias = col_med - global_med
        col_bias[~np.isfinite(col_bias)] = 0.0
        out = out - col_bias[None, :]
    except Exception:
        pass
    try:
        row_med = np.nanmedian(out, axis=1)
        row_bias = row_med - float(np.nanmedian(out))
        row_bias[~np.isfinite(row_bias)] = 0.0
        out = out - row_bias[:, None]
    except Exception:
        pass
    if "robust line scale" in mode:
        try:
            # Mild robust equalisation, clipped so it does not destroy real amplitude contrast.
            row_sc = np.nanpercentile(np.abs(out), 95, axis=1)
            ref = float(np.nanmedian(row_sc[np.isfinite(row_sc) & (row_sc > 0)]))
            row_sc[~np.isfinite(row_sc) | (row_sc <= 0)] = ref
            fac = np.clip(ref / row_sc, 0.5, 2.0)
            out = out * fac[:, None]
        except Exception:
            pass
    out[~good] = np.nan
    return out

def _tl_install_real_linebias_imshow_patch():
    try:
        import inspect
        import matplotlib.axes
    except Exception:
        return
    if getattr(matplotlib.axes.Axes.imshow, "_tl_real_linebias_patched", False):
        return
    old = matplotlib.axes.Axes.imshow
    def new_imshow(self, X, *args, **kwargs):
        try:
            if any("stable_time_lapse_patch.py" in str(fr.filename) for fr in inspect.stack()):
                X = _tl_apply_linebias_to_grid(X)
        except Exception:
            pass
        return old(self, X, *args, **kwargs)
    new_imshow._tl_real_linebias_patched = True
    matplotlib.axes.Axes.imshow = new_imshow

_old_apply_schleitheim_real_linebias = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    _tl_install_real_linebias_imshow_patch()
    if _old_apply_schleitheim_real_linebias:
        _old_apply_schleitheim_real_linebias(globs)

_old_apply_bulach_real_linebias = globals().get("apply_bulach")
def apply_bulach(globs):
    _tl_install_real_linebias_imshow_patch()
    if _old_apply_bulach_real_linebias:
        _old_apply_bulach_real_linebias(globs)
# --- end real line-bias image-level correction patch ---




# --- hover line-xy compatibility patch ---
def _tl_line_xy(obj, line, n=None):
    import numpy as np
    try:
        x = np.asarray(getattr(line, "x"), float)
        y = np.asarray(getattr(line, "y"), float)
        if x.size >= 2 and y.size >= 2:
            if n is not None and len(x) != int(n):
                old = np.linspace(0, 1, len(x))
                new = np.linspace(0, 1, int(n))
                x = np.interp(new, old, x)
                y = np.interp(new, old, y)
            return x, y
    except Exception:
        pass
    try:
        nn = int(n) if n is not None else int(getattr(line, "traces", 100) or 100)
        if hasattr(obj, "trace_xy"):
            x, y = obj.trace_xy(line, nn)
            return np.asarray(x, float), np.asarray(y, float)
        if hasattr(obj, "trace_xyz"):
            x, y, _ = obj.trace_xyz(line, nn)
            return np.asarray(x, float), np.asarray(y, float)
    except Exception:
        pass
    return None, None
# --- end hover line-xy compatibility patch ---




# --- compact extra controls into existing rows patch ---
def _tl_find_row_with_label(obj, needle):
    try:
        from PyQt6.QtWidgets import QLabel
        needle = needle.lower()
        for lab in obj.findChildren(QLabel):
            try:
                if needle in lab.text().lower():
                    row = lab.parentWidget()
                    if row is not None and row.layout() is not None:
                        return row, row.layout()
            except Exception:
                pass
    except Exception:
        pass
    return None, None

def _tl_hide_empty_control_row(row):
    try:
        if row is not None:
            row.hide()
            row.setMaximumHeight(0)
    except Exception:
        pass

def _tl_move_combo_to_existing_row(obj, combo_attr, target_label, new_label):
    try:
        from PyQt6.QtWidgets import QLabel
        cb = getattr(obj, combo_attr, None)
        if cb is None:
            return
        target_row, target_lay = _tl_find_row_with_label(obj, target_label)
        if target_lay is None:
            return
        old_row = cb.parentWidget()
        old_lay = old_row.layout() if old_row is not None else None
        if old_lay is not None:
            old_lay.removeWidget(cb)
        cb.setParent(target_row)
        lab = QLabel(new_label, target_row)
        target_lay.addSpacing(18)
        target_lay.addWidget(lab)
        target_lay.addWidget(cb)
        if not hasattr(obj, "_tl_compact_labels"):
            obj._tl_compact_labels = []
        obj._tl_compact_labels.append(lab)
        if old_row is not None and old_row is not target_row:
            _tl_hide_empty_control_row(old_row)
    except Exception as e:
        print("Could not compact", combo_attr, e)

def _tl_compact_extra_controls(obj):
    if getattr(obj, "_tl_extra_controls_compacted", False):
        return
    # Put these into already-existing empty control-row space.
    _tl_move_combo_to_existing_row(obj, "map_smoothing", "Amplitude mode", "Map smoothing")
    _tl_move_combo_to_existing_row(obj, "line_bias_correction", "Bad-line QC", "Line-bias correction")
    obj._tl_extra_controls_compacted = True

def _tl_patch_compact_class(cls):
    if cls is None or getattr(cls, "_tl_compact_controls_class_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _tl_compact_extra_controls(self)
    cls.__init__ = new_init
    cls._tl_compact_controls_class_patched = True

_old_apply_schleitheim_compact = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_compact:
        _old_apply_schleitheim_compact(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab"):
        _tl_patch_compact_class(globs.get(n))

_old_apply_bulach_compact = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_compact:
        _old_apply_bulach_compact(globs)
    _tl_patch_compact_class(globs.get("PulseEkko3DAnalysis"))
# --- end compact extra controls into existing rows patch ---




# --- compact QC controls onto amplitude row patch ---
def _tl_find_label_widget(obj, text_contains):
    try:
        from PyQt6.QtWidgets import QLabel
        key = text_contains.lower()
        for lab in obj.findChildren(QLabel):
            try:
                if key in lab.text().lower():
                    return lab
            except Exception:
                pass
    except Exception:
        pass
    return None

def _tl_row_of_widget(w):
    try:
        row = w.parentWidget()
        lay = row.layout() if row is not None else None
        return row, lay
    except Exception:
        return None, None

def _tl_remove_from_layout(w):
    try:
        row, lay = _tl_row_of_widget(w)
        if lay is not None:
            lay.removeWidget(w)
        return row
    except Exception:
        return None

def _tl_widget_after_label(label):
    try:
        row, lay = _tl_row_of_widget(label)
        if lay is None:
            return None
        idx = lay.indexOf(label)
        if idx >= 0 and idx + 1 < lay.count():
            item = lay.itemAt(idx + 1)
            return item.widget()
    except Exception:
        pass
    return None

def _tl_hide_if_empty_row(row):
    try:
        if row is None:
            return
        lay = row.layout()
        if lay is None:
            return
        visible_widgets = []
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is not None and not isinstance(w, type(None)):
                try:
                    if w.isVisible():
                        visible_widgets.append(w)
                except Exception:
                    pass
        # Hide rows that only contained moved controls.
        texts = []
        from PyQt6.QtWidgets import QLabel
        for w in visible_widgets:
            if isinstance(w, QLabel):
                texts.append(w.text().lower())
        if any(("bad-line" in t or "line-bias" in t) for t in texts):
            row.hide()
            row.setMaximumHeight(0)
    except Exception:
        pass

def _tl_move_pair_to_row(obj, label_text, target_row, target_lay, compact_label=None):
    try:
        from PyQt6.QtWidgets import QLabel
        lab = _tl_find_label_widget(obj, label_text)
        if lab is None:
            return None, None
        widget = _tl_widget_after_label(lab)
        old_row = lab.parentWidget()
        _tl_remove_from_layout(lab)
        if widget is not None:
            _tl_remove_from_layout(widget)
        lab.setParent(target_row)
        if compact_label:
            lab.setText(compact_label)
        target_lay.addSpacing(12)
        target_lay.addWidget(lab)
        if widget is not None:
            widget.setParent(target_row)
            target_lay.addWidget(widget)
        if old_row is not target_row:
            _tl_hide_if_empty_row(old_row)
        return lab, widget
    except Exception as e:
        print("Could not move control", label_text, e)
        return None, None

def _tl_move_known_label_to_row(obj, target_row, target_lay):
    try:
        from PyQt6.QtWidgets import QLabel
        for lab in obj.findChildren(QLabel):
            try:
                txt = lab.text()
                if txt.strip().lower().startswith("known:"):
                    old_row = lab.parentWidget()
                    _tl_remove_from_layout(lab)
                    lab.setParent(target_row)
                    target_lay.addWidget(lab)
                    if old_row is not target_row:
                        _tl_hide_if_empty_row(old_row)
                    return
            except Exception:
                pass
    except Exception:
        pass

def _tl_compact_qc_controls_to_amplitude_row(obj):
    if getattr(obj, "_tl_qc_controls_on_amp_row", False):
        return
    amp_lab = _tl_find_label_widget(obj, "Amplitude mode")
    if amp_lab is None:
        return
    target_row, target_lay = _tl_row_of_widget(amp_lab)
    if target_lay is None:
        return

    _tl_move_pair_to_row(obj, "Bad-line QC", target_row, target_lay, "Bad-line QC")
    _tl_move_pair_to_row(obj, "MAD k", target_row, target_lay, "MAD k")
    _tl_move_known_label_to_row(obj, target_row, target_lay)
    _tl_move_pair_to_row(obj, "Line-bias correction", target_row, target_lay, "Line bias")

    obj._tl_qc_controls_on_amp_row = True

def _tl_patch_qc_compact_class(cls):
    if cls is None or getattr(cls, "_tl_qc_compact_class_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _tl_compact_qc_controls_to_amplitude_row(self)
    cls.__init__ = new_init
    cls._tl_qc_compact_class_patched = True

_old_apply_schleitheim_qc_compact = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_qc_compact:
        _old_apply_schleitheim_qc_compact(globs)
    for n in ("GPR3DStandardAnalysisTab", "GPR3DAnalysisTab"):
        _tl_patch_qc_compact_class(globs.get(n))

_old_apply_bulach_qc_compact = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_qc_compact:
        _old_apply_bulach_qc_compact(globs)
    for n in ("PulseEkko3DAnalysis", "PulseEkkoProjectWidget"):
        _tl_patch_qc_compact_class(globs.get(n))
# --- end compact QC controls onto amplitude row patch ---

