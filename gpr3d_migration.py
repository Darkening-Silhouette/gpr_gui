# gpr3d_migration.py
# Adds scientific 3-D GPR volume processing + constant-velocity 3-D Stolt migration
# to Schleitheim/MALA and Bulach/PulseEKKO GUI tabs.
from __future__ import annotations
import json, math, datetime, traceback
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QTextEdit, QMessageBox,
    QProgressDialog, QApplication, QFileDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from scipy.signal import hilbert, fftconvolve
    from scipy.signal.windows import hann as scipy_hann
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import segyio
    from segyio import TraceField, BinField
    SEGYIO_OK = True
except Exception:
    SEGYIO_OK = False

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False


class _PVCanvas(FigureCanvas):
    def __init__(self, w=12, h=7):
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _cosine_taper_1d(n: int, frac: float) -> np.ndarray:
    frac = float(frac)
    if frac <= 0 or n <= 1:
        return np.ones(int(n), dtype=np.float32)
    frac = min(frac, 0.5)
    m = int(round(frac * n))
    if m < 1:
        return np.ones(int(n), dtype=np.float32)
    w = np.ones(int(n), dtype=np.float32)
    ramp = np.sin(0.5 * np.pi * (np.arange(m, dtype=float) + 1.0) / m) ** 2
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w.astype(np.float32)


def _apply_edge_taper(data, taper_t=0.0, taper_x=0.0, taper_y=0.0):
    nt, nx, ny = data.shape
    wt = _cosine_taper_1d(nt, taper_t)[:, None, None]
    wx = _cosine_taper_1d(nx, taper_x)[None, :, None]
    wy = _cosine_taper_1d(ny, taper_y)[None, None, :]
    return data * wt * wx * wy


def _interp1_complex_regular(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Linear interpolation for complex y(x) on ascending x. Zero outside range."""
    xi = np.asarray(xi, float)
    yi = np.zeros(xi.shape, dtype=y.dtype)
    idx = np.searchsorted(x, xi, side="right") - 1
    valid = (idx >= 0) & (idx < x.size - 1)
    if np.any(valid):
        iv = idx[valid]
        w = (xi[valid] - x[iv]) / (x[iv + 1] - x[iv])
        yi[valid] = (1.0 - w) * y[iv] + w * y[iv + 1]
    return yi


def stolt_migration_3d(data, dt, dx, dy, velocity, dz=None, nz=None,
                       exploding_reflector=True, apply_jacobian=True,
                       pad_t=1.5, pad_x=1.15, pad_y=1.15,
                       taper_t=0.05, taper_x=0.05, taper_y=0.05,
                       pad_to_pow2=True, depth_padding=2.0,
                       progress_cb=None):
    """
    Constant-velocity 3-D Stolt migration, matching the reference notebook logic.

    Input data shape: (nt, nx, ny) = (TWT samples, regular x-grid, regular y-grid).
    For zero-offset GPR, exploding_reflector=True uses vm = velocity/2.
    Output shape: (nz, nx, ny), depth axis dz = vm*dt by default.
    """
    d = np.asarray(data, dtype=np.float32)
    if d.ndim != 3:
        raise ValueError("data must have shape (nt, nx, ny)")
    nt0, nx0, ny0 = map(int, d.shape)
    vm = 0.5 * float(velocity) if exploding_reflector else float(velocity)
    if dz is None:
        dz = vm * float(dt)
    if nz is None:
        nz = nt0
    nz = int(min(max(1, nz), nt0))

    if progress_cb:
        progress_cb(5, "Applying edge taper and one-sided padding...")
    d = _apply_edge_taper(d, taper_t=taper_t, taper_x=taper_x, taper_y=taper_y)

    ntp = max(nt0, int(math.ceil(nt0 * float(pad_t))))
    nxp = max(nx0, int(math.ceil(nx0 * float(pad_x))))
    nyp = max(ny0, int(math.ceil(ny0 * float(pad_y))))
    if pad_to_pow2:
        ntp, nxp, nyp = _next_pow2(ntp), _next_pow2(nxp), _next_pow2(nyp)

    dp = np.zeros((ntp, nxp, nyp), dtype=np.float32)
    dp[:nt0, :nx0, :ny0] = d

    if progress_cb:
        progress_cb(15, f"3-D FFT: padded cube {ntp}×{nxp}×{nyp}...")
    D = np.fft.rfft(dp, axis=0)
    D = np.fft.fft(D, axis=1)
    D = np.fft.fft(D, axis=2)

    f = np.fft.rfftfreq(ntp, d=float(dt))
    fx = np.fft.fftfreq(nxp, d=float(dx))
    fy = np.fft.fftfreq(nyp, d=float(dy))

    nzfft = max(nz, int(math.ceil(nz * float(depth_padding))))
    if pad_to_pow2:
        nzfft = _next_pow2(nzfft)
    fz = np.fft.fftfreq(nzfft, d=float(dz))

    I_k = np.zeros((nzfft, nxp, nyp), dtype=D.dtype)
    total = max(1, nxp)
    if progress_cb:
        progress_cb(25, "Stolt frequency remapping...")
    for ix in range(nxp):
        fx_i = fx[ix]
        for iy in range(nyp):
            fy_i = fy[iy]
            rho = np.sqrt(fx_i * fx_i + fy_i * fy_i + fz * fz)
            f_target = vm * rho
            spec = _interp1_complex_regular(f, D[:, ix, iy], f_target)
            if apply_jacobian:
                scale = np.zeros_like(rho, dtype=float)
                mask = rho > 0
                scale[mask] = np.abs(fz[mask]) / rho[mask]
                spec = spec * scale
            I_k[:, ix, iy] = spec
        if progress_cb and (ix % max(1, nxp // 40) == 0 or ix == nxp - 1):
            pct = 25 + int(45 * (ix + 1) / total)
            progress_cb(pct, f"Stolt remap {ix + 1}/{nxp} spatial frequency rows...")

    if progress_cb:
        progress_cb(75, "Inverse 3-D FFT to depth image...")
    image_full = np.fft.ifftn(I_k, axes=(0, 1, 2)).real.astype(np.float32)
    image = image_full[:nz, :nx0, :ny0]
    if progress_cb:
        progress_cb(85, "Migration complete; post-processing...")
    return image


def _envelope(data: np.ndarray) -> np.ndarray:
    if SCIPY_OK:
        return np.abs(hilbert(data, axis=0)).astype(np.float32)
    return np.abs(data).astype(np.float32)


def _spatial_k_filter(cube: np.ndarray, dx: float, dy: float, mode: str) -> np.ndarray:
    mode = (mode or "Off").lower()
    if mode.startswith("off"):
        return cube
    nt, nx, ny = cube.shape
    fx = np.fft.fftfreq(nx, d=max(float(dx), 1e-9))[:, None]
    fy = np.fft.fftfreq(ny, d=max(float(dy), 1e-9))[None, :]
    kr = np.sqrt(fx * fx + fy * fy)
    kmax = float(np.nanmax(kr)) if kr.size else 1.0
    if "strong" in mode:
        frac = 0.30
    elif "medium" in mode:
        frac = 0.45
    else:
        frac = 0.65
    kc = max(frac * kmax, 1e-9)
    filt = np.exp(-(kr / kc) ** 4).astype(np.float32)
    F = np.fft.fftn(cube, axes=(1, 2))
    return np.fft.ifftn(F * filt[None, :, :], axes=(1, 2)).real.astype(np.float32)


def _hann1d_halfwidth(hw: int) -> np.ndarray:
    hw = int(hw)
    if hw <= 0 or not SCIPY_OK:
        return np.array([1.0], dtype=np.float32)
    w = scipy_hann(2 * hw + 1).astype(np.float32)
    s = float(w.sum())
    return w / s if s > 0 else np.array([1.0], dtype=np.float32)


def _apply_live_taper(cube: np.ndarray, valid2d: np.ndarray, hw_x: int, hw_y: int) -> np.ndarray:
    if not SCIPY_OK or hw_x <= 0 and hw_y <= 0:
        return cube
    kx = _hann1d_halfwidth(hw_x)
    ky = _hann1d_halfwidth(hw_y)
    kern = kx[:, None] * ky[None, :]
    kern /= max(float(kern.sum()), 1e-12)
    taper = fftconvolve(valid2d.astype(np.float32), kern.astype(np.float32), mode="same")
    taper = np.clip(taper, 0.0, 1.0).astype(np.float32)
    return cube * taper[None, :, :]


def _ricker(t, f0):
    a = np.pi * f0 * t
    return (1 - 2*a*a) * np.exp(-a*a)


def _synthetic_validation(parent=None):
    """Small scientific GPR validation: hyperbola should focus near the true depth."""
    nt, nx, ny = 256, 64, 64
    dt, dx, dy, vel = 0.4, 0.05, 0.05, 0.10
    z0 = 0.75
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dy
    tt = np.arange(nt) * dt
    X, Y = np.meshgrid(x, y, indexing="ij")
    twt = 2.0 * np.sqrt(z0*z0 + X*X + Y*Y) / vel
    data = np.zeros((nt, nx, ny), dtype=np.float32)
    f0 = 0.8  # cycles/ns
    for ix in range(nx):
        for iy in range(ny):
            data[:, ix, iy] = _ricker(tt - twt[ix, iy], f0)
    img = stolt_migration_3d(data, dt, dx, dy, vel, pad_t=2.0, pad_x=1.5, pad_y=1.5,
                             taper_t=0.05, taper_x=0.1, taper_y=0.1, nz=nt)
    depth = (vel/2.0) * tt
    peak = np.unravel_index(np.nanargmax(np.abs(img)), img.shape)
    z_peak = float(depth[peak[0]])
    ok = abs(z_peak - z0) <= max(0.10, 3*(vel/2.0)*dt)
    return ok, z0, z_peak


class GPR3DMigrationTab(QWidget):
    def __init__(self, analysis_owner, kind: str):
        super().__init__()
        self.owner = analysis_owner
        self.kind = kind
        self.last = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        grid = QGridLayout()

        self.data_mode = QComboBox(); self.data_mode.addItems(["processed", "raw"])
        self.attribute = QComboBox(); self.attribute.addItems(["Signed amplitude", "Envelope amplitude", "Absolute amplitude"])
        self.attribute.setCurrentText("Signed amplitude")
        self.k_filter = QComboBox(); self.k_filter.addItems(["Off", "Light k-filter", "Medium k-filter", "Strong k-filter"])
        self.k_filter.setCurrentText("Light k-filter")
        self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.02, 0.30); self.velocity.setValue(0.10); self.velocity.setSingleStep(0.005); self.velocity.setSuffix(" m/ns")
        self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(0.0); self.tmin.setSuffix(" ns")
        self.tmax = QDoubleSpinBox(); self.tmax.setRange(0, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(" ns")
        self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20); self.dt_out.setValue(0.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(" ns")
        self.grid_dx = QDoubleSpinBox(); self.grid_dx.setRange(0.02, 2.0); self.grid_dx.setValue(0.25); self.grid_dx.setSingleStep(0.05); self.grid_dx.setSuffix(" m")
        self.grid_dy = QDoubleSpinBox(); self.grid_dy.setRange(0.02, 2.0); self.grid_dy.setValue(0.25); self.grid_dy.setSingleStep(0.05); self.grid_dy.setSuffix(" m")
        self.max_nx = QSpinBox(); self.max_nx.setRange(20, 512); self.max_nx.setValue(180)
        self.max_ny = QSpinBox(); self.max_ny.setRange(20, 512); self.max_ny.setValue(140)
        self.trace_step = QSpinBox(); self.trace_step.setRange(1, 100); self.trace_step.setValue(5)
        self.max_lines = QSpinBox(); self.max_lines.setRange(1, 10000); self.max_lines.setValue(500)
        self.pad_t = QDoubleSpinBox(); self.pad_t.setRange(1.0, 4.0); self.pad_t.setValue(1.5); self.pad_t.setSingleStep(0.25)
        self.pad_xy = QSpinBox(); self.pad_xy.setRange(0, 100); self.pad_xy.setValue(15); self.pad_xy.setSuffix(" cells")
        self.taper_w = QSpinBox(); self.taper_w.setRange(0, 50); self.taper_w.setValue(5); self.taper_w.setSuffix(" cells")
        self.live_taper = QCheckBox("Live/dead mask taper"); self.live_taper.setChecked(True)
        self.blank_dead = QCheckBox("Re-blank dead cells"); self.blank_dead.setChecked(True)
        self.blank_topo = QCheckBox("Topographic blanking"); self.blank_topo.setChecked(True)
        self.jacobian = QCheckBox("Stolt Jacobian"); self.jacobian.setChecked(True)
        self.depth_slice = QDoubleSpinBox(); self.depth_slice.setRange(0.0, 20.0); self.depth_slice.setValue(0.60); self.depth_slice.setSingleStep(0.05); self.depth_slice.setSuffix(" m")
        self.clip_pct = QDoubleSpinBox(); self.clip_pct.setRange(80.0, 100.0); self.clip_pct.setValue(98.5); self.clip_pct.setSingleStep(0.1); self.clip_pct.setSuffix(" %")

        entries = [
            ("Data", self.data_mode), ("Attribute", self.attribute), ("Velocity", self.velocity), ("tmin", self.tmin), ("tmax", self.tmax),
            ("dt", self.dt_out), ("Grid dx", self.grid_dx), ("Grid dy", self.grid_dy), ("Max nx", self.max_nx), ("Max ny", self.max_ny),
            ("Trace step", self.trace_step), ("Max lines", self.max_lines), ("k-filter", self.k_filter), ("Pad t", self.pad_t), ("Pad xy", self.pad_xy),
            ("Taper", self.taper_w), ("Bird's-eye depth slice", self.depth_slice), ("Clip", self.clip_pct)
        ]
        for i, (lab, widget) in enumerate(entries):
            r = i // 6; c = (i % 6) * 2
            grid.addWidget(QLabel(lab), r, c); grid.addWidget(widget, r, c+1)
        base_row = (len(entries)+5)//6
        for j, w in enumerate([self.live_taper, self.blank_dead, self.blank_topo, self.jacobian]):
            grid.addWidget(w, base_row, j*2, 1, 2)
        root.addLayout(grid)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Build volume + run 3-D Stolt migration")
        self.btn_validate = QPushButton("Run synthetic validation")
        self.btn_png = QPushButton("Export PNG")
        self.btn_npz = QPushButton("Export NPZ volume")
        self.btn_segy = QPushButton("Export SEG-Y" + ("" if SEGYIO_OK else " (needs segyio)"))
        self.btn_gis = QPushButton("Export GeoPackage slices" + ("" if RASTERIO_OK else " (needs rasterio)"))
        for b in [self.btn_run, self.btn_validate, self.btn_png, self.btn_npz, self.btn_segy, self.btn_gis]: row.addWidget(b)
        root.addLayout(row)

        self.canvas = _PVCanvas(12, 7)
        root.addWidget(self.canvas, stretch=1)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120)
        root.addWidget(self.log)

        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_validate.clicked.connect(self.run_validation)
        self.btn_png.clicked.connect(self.export_png)
        self.btn_npz.clicked.connect(self.export_npz)
        self.btn_segy.clicked.connect(self.export_segy)
        self.btn_gis.clicked.connect(self.export_gpkg)

    def _status(self, txt):
        self.log.append(str(txt))
        try:
            target = getattr(getattr(self.owner, 'main', None), 'status', None) or getattr(getattr(self.owner, 'owner', None), 'status', None)
            if target is not None: target.setText(str(txt))
        except Exception:
            pass
        QApplication.processEvents()

    def _project_root(self):
        if self.kind == 'schleitheim':
            return Path('/home/luqman/gpr_gui/data/MALA')
        return Path(getattr(getattr(self.owner, 'owner', None), 'root', '/home/luqman/gpr_gui/data/PulseEkko'))

    def _selected_lines(self):
        if self.kind == 'schleitheim':
            try: lines = list(self.owner.selected_lines_for_maps())
            except Exception: lines = list(getattr(self.owner.main, 'lines', []))
        else:
            try: lines = list(self.owner.selected_lines())
            except Exception: lines = list(getattr(self.owner.owner, 'lines', []))
        return lines[:int(self.max_lines.value())]

    def _line_array_time_xyze(self, line):
        if self.kind == 'schleitheim':
            # Temporarily honour local Data mode in this tab, independently of main map tab.
            old = None
            try:
                old = self.owner.mode.currentText()
                self.owner.mode.setCurrentText(self.data_mode.currentText())
            except Exception:
                pass
            try:
                arr = self.owner.ensure_data(line)
            finally:
                if old is not None:
                    try: self.owner.mode.setCurrentText(old)
                    except Exception: pass
            t = self.owner.time_vector(line, arr.shape[1])
            try:
                x, y, z = self.owner.trace_xyz(line, arr.shape[0])
            except Exception:
                x, y = self.owner.trace_xy(line, arr.shape[0])
                elev = getattr(line, 'elev', None)
                if elev is not None and len(elev) >= 2:
                    gps_i = np.linspace(0, 1, len(elev)); tr_i = np.linspace(0, 1, arr.shape[0])
                    z = np.interp(tr_i, gps_i, np.asarray(elev, float))
                else:
                    z = np.full(arr.shape[0], np.nan, dtype=float)
        else:
            old = None
            try:
                old = self.owner.data_choice.currentText()
                self.owner.data_choice.setCurrentText(self.data_mode.currentText())
            except Exception:
                pass
            try:
                arr = self.owner.get_array(line)
            finally:
                if old is not None:
                    try: self.owner.data_choice.setCurrentText(old)
                    except Exception: pass
            try: t = self.owner.owner.corrected_time_ns(line)
            except Exception: t = line.time_ns
            x = np.asarray(line.x, float); y = np.asarray(line.y, float); z = np.full_like(x, np.nan, dtype=float)
        return np.asarray(arr, float), np.asarray(t, float), np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)

    def _collect_traces(self, progress=None):
        lines = self._selected_lines()
        if not lines:
            raise RuntimeError('No selected lines for migration.')
        lo, hi = sorted([float(self.tmin.value()), float(self.tmax.value())])
        if hi <= lo: hi = lo + 1.0
        dt = float(self.dt_out.value())
        t_axis = np.arange(lo, hi + 0.5*dt, dt, dtype=float)
        if t_axis.size < 8:
            raise RuntimeError('Time window too short for migration.')

        xs=[]; ys=[]; zs=[]; data=[]; names=[]
        step = max(1, int(self.trace_step.value()))
        total = max(1, len(lines))
        for k, line in enumerate(lines, 1):
            if progress: progress(2 + int(18*k/total), f'Collecting traces {k}/{total}: {getattr(line,"name",line)}')
            arr, t, x, y, z = self._line_array_time_xyze(line)
            if arr is None or arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 8:
                continue
            idx = np.arange(0, arr.shape[0], step, dtype=int)
            x = np.interp(np.linspace(0,1,arr.shape[0]), np.linspace(0,1,len(x)), x) if len(x) != arr.shape[0] else x
            y = np.interp(np.linspace(0,1,arr.shape[0]), np.linspace(0,1,len(y)), y) if len(y) != arr.shape[0] else y
            if z.size != arr.shape[0] and z.size >= 2:
                z = np.interp(np.linspace(0,1,arr.shape[0]), np.linspace(0,1,len(z)), z)
            elif z.size != arr.shape[0]:
                z = np.full(arr.shape[0], np.nan)
            good_time = (t_axis >= np.nanmin(t)) & (t_axis <= np.nanmax(t))
            if not np.any(good_time):
                continue
            for ii in idx:
                tr = np.interp(t_axis, t, arr[ii, :], left=0.0, right=0.0).astype(np.float32)
                xs.append(float(x[ii])); ys.append(float(y[ii])); zs.append(float(z[ii]) if np.isfinite(z[ii]) else np.nan)
                data.append(tr); names.append(getattr(line, 'name', str(line)))
        if not data:
            raise RuntimeError('No usable traces in selected time window.')
        data = np.vstack(data).astype(np.float32)
        return np.asarray(xs), np.asarray(ys), np.asarray(zs), data, t_axis, names

    def _build_cube(self, progress=None):
        x, y, z, tr, t_axis, names = self._collect_traces(progress)
        dx_req = float(self.grid_dx.value()); dy_req = float(self.grid_dy.value())
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        nx = max(2, int(math.ceil((xmax-xmin)/max(dx_req,1e-9))) + 1)
        ny = max(2, int(math.ceil((ymax-ymin)/max(dy_req,1e-9))) + 1)
        max_nx, max_ny = int(self.max_nx.value()), int(self.max_ny.value())
        if nx > max_nx:
            dx_req = (xmax-xmin)/max(max_nx-1,1); nx=max_nx
        if ny > max_ny:
            dy_req = (ymax-ymin)/max(max_ny-1,1); ny=max_ny
        xi = xmin + np.arange(nx)*dx_req
        yi = ymin + np.arange(ny)*dy_req
        ix = np.rint((x-xmin)/max(dx_req,1e-9)).astype(int)
        iy = np.rint((y-ymin)/max(dy_req,1e-9)).astype(int)
        ok = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&np.all(np.isfinite(tr), axis=1)&np.isfinite(x)&np.isfinite(y)
        ix, iy, tr = ix[ok], iy[ok], tr[ok]
        z_ok = z[ok]
        nt = tr.shape[1]
        cube = np.zeros((nt,nx,ny), dtype=np.float32)
        count = np.zeros((nx,ny), dtype=np.float32)
        elev_sum = np.zeros((nx,ny), dtype=np.float64)
        elev_count = np.zeros((nx,ny), dtype=np.float64)
        total = max(1, tr.shape[0])
        if progress: progress(22, f'Binning {total:,} traces into regular cube {nt}×{nx}×{ny}...')
        for j in range(total):
            cube[:, ix[j], iy[j]] += tr[j]
            count[ix[j], iy[j]] += 1.0
            if np.isfinite(z_ok[j]):
                elev_sum[ix[j], iy[j]] += z_ok[j]
                elev_count[ix[j], iy[j]] += 1.0
            if progress and (j % max(1, total//20) == 0):
                progress(22 + int(16*j/total), f'Binning trace {j+1:,}/{total:,}...')
        valid = count > 0
        cube[:, valid] /= count[valid][None, :]
        elev_grid = np.full((nx,ny), np.nan, dtype=np.float32)
        good_e = elev_count > 0
        elev_grid[good_e] = (elev_sum[good_e]/elev_count[good_e]).astype(np.float32)
        return dict(cube=cube, valid=valid, count=count, elev=elev_grid, x=xi, y=yi, dx=dx_req, dy=dy_req, t=t_axis,
                    n_input=int(tr.shape[0]), names=names)

    def _pipeline(self, progress=None):
        out = self._build_cube(progress)
        cube = out['cube']
        valid = out['valid']
        if progress: progress(40, 'Applying attribute and spatial k-filter...')
        attr = self.attribute.currentText()
        cube_for_display = cube.copy()
        if attr == 'Envelope amplitude':
            cube_for_display = _envelope(cube_for_display)
        elif attr == 'Absolute amplitude':
            cube_for_display = np.abs(cube_for_display).astype(np.float32)
        cube_proc = _spatial_k_filter(cube.astype(np.float32), out['dx'], out['dy'], self.k_filter.currentText())
        if self.live_taper.isChecked():
            hw = int(self.taper_w.value())
            cube_proc = _apply_live_taper(cube_proc, valid, hw, hw)
        vel = float(self.velocity.value())
        dt = float(np.nanmedian(np.diff(out['t']))) if len(out['t']) > 1 else float(self.dt_out.value())
        pad_cells = int(self.pad_xy.value())
        pad_x = (cube_proc.shape[1] + pad_cells) / max(cube_proc.shape[1],1)
        pad_y = (cube_proc.shape[2] + pad_cells) / max(cube_proc.shape[2],1)
        taper_frac_x = min(0.5, int(self.taper_w.value()) / max(cube_proc.shape[1],1))
        taper_frac_y = min(0.5, int(self.taper_w.value()) / max(cube_proc.shape[2],1))
        mig = stolt_migration_3d(cube_proc, dt=dt, dx=out['dx'], dy=out['dy'], velocity=vel,
                                 dz=(vel/2.0)*dt, nz=cube_proc.shape[0], exploding_reflector=True,
                                 apply_jacobian=self.jacobian.isChecked(), pad_t=float(self.pad_t.value()),
                                 pad_x=pad_x, pad_y=pad_y, taper_t=0.05, taper_x=taper_frac_x, taper_y=taper_frac_y,
                                 depth_padding=2.0, progress_cb=progress)
        if self.blank_dead.isChecked():
            mig[:, ~valid] = 0.0
        depth = (vel/2.0) * (out['t'] - out['t'][0])
        # Topographic blanking follows reference convention: depth=0 at final datum, blank samples above local surface.
        if self.blank_topo.isChecked() and np.isfinite(out['elev']).any():
            final_datum = float(np.nanmax(out['elev']))
            dz = float((vel/2.0) * dt)
            depth_surf = final_datum - out['elev']
            n_blank = np.where(np.isfinite(depth_surf), np.floor(depth_surf/max(dz,1e-9)).astype(int), 0)
            n_blank = np.clip(n_blank, 0, mig.shape[0])
            sample_idx = np.arange(mig.shape[0])[:,None,None]
            mig[sample_idx < n_blank[None,:,:]] = 0.0
            out['final_datum'] = final_datum
            out['n_blank'] = n_blank
        out.update(dict(migrated=mig.astype(np.float32), depth=depth.astype(np.float32), cube_display=cube_for_display,
                        velocity=vel, dt=dt, attribute=attr, k_filter=self.k_filter.currentText(),
                        created=datetime.datetime.now().isoformat()))
        return out

    def _progress_dialog(self):
        dlg = QProgressDialog('Starting 3-D Stolt migration...', 'Cancel', 0, 100, self)
        dlg.setWindowTitle('Scientific 3-D migration')
        dlg.setMinimumDuration(0); dlg.setAutoClose(False); dlg.setAutoReset(False); dlg.setValue(0)
        def cb(pct, msg):
            dlg.setValue(int(max(0, min(100, pct))))
            dlg.setLabelText(str(msg))
            self._status(msg)
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise RuntimeError('Migration cancelled by user.')
        return dlg, cb

    def run_pipeline(self):
        dlg, cb = self._progress_dialog()
        try:
            self.log.clear()
            self._status('Scientific pipeline: regular 3-D binning → optional k-filter/live taper → 3-D Stolt migration → dead/topo blanking.')
            self.last = self._pipeline(cb)
            cb(95, 'Drawing migration QC figure...')
            self.plot_result()
            cb(100, '3-D Stolt migration finished.')
        except Exception as e:
            self._status('Migration failed: ' + str(e))
            QMessageBox.critical(self, '3-D migration failed', f'{e}\n\n{traceback.format_exc()}')
        finally:
            dlg.close()

    def plot_result(self):
        if not self.last:
            return
        r = self.last
        fig = self.canvas.fig; fig.clear()
        ax0 = fig.add_subplot(221); ax1 = fig.add_subplot(222); ax2 = fig.add_subplot(223); ax3 = fig.add_subplot(224)
        valid = r['valid']
        ax0.imshow(valid.T, origin='lower', aspect='auto', extent=[r['x'][0], r['x'][-1], r['y'][0], r['y'][-1]], cmap='gray_r')
        ax0.set_title(f'Regular 3-D binning mask: {valid.sum()}/{valid.size} live cells')
        ax0.set_xlabel('X [m]'); ax0.set_ylabel('Y [m]')
        # input attribute time slice
        mid_t = r['cube_display'].shape[0]//2
        s1 = r['cube_display'][mid_t]
        v1 = np.nanpercentile(np.abs(s1[np.isfinite(s1)]), float(self.clip_pct.value())) if np.isfinite(s1).any() else 1.0
        im1=ax1.imshow(s1.T, origin='lower', aspect='auto', extent=[r['x'][0], r['x'][-1], r['y'][0], r['y'][-1]], cmap='seismic', vmin=-v1, vmax=v1)
        ax1.set_title(f'Input binned slice: {r["t"][mid_t]:.1f} ns ({r["attribute"]})')
        fig.colorbar(im1, ax=ax1, shrink=0.8)
        # migrated depth slice
        target = float(self.depth_slice.value())
        iz = int(np.argmin(np.abs(r['depth'] - target)))
        s2 = r['migrated'][iz]
        finite = s2[np.isfinite(s2)]
        v2 = np.nanpercentile(np.abs(finite), float(self.clip_pct.value())) if finite.size else 1.0
        im2=ax2.imshow(s2.T, origin='lower', aspect='auto', extent=[r['x'][0], r['x'][-1], r['y'][0], r['y'][-1]], cmap='seismic', vmin=-v2, vmax=v2)
        ax2.set_title(f'Migrated depth slice: z={r["depth"][iz]:.2f} m')
        ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]'); fig.colorbar(im2, ax=ax2, shrink=0.8)
        # migrated section through middle y
        iy = r['migrated'].shape[2]//2
        sec = r['migrated'][:,:,iy]
        fsec = sec[np.isfinite(sec)]
        v3 = np.nanpercentile(np.abs(fsec), float(self.clip_pct.value())) if fsec.size else 1.0
        im3=ax3.imshow(sec, origin='upper', aspect='auto', extent=[r['x'][0], r['x'][-1], r['depth'][-1], r['depth'][0]], cmap='seismic', vmin=-v3, vmax=v3)
        ax3.set_title(f'Migrated vertical section at Y={r["y"][iy]:.2f} m')
        ax3.set_xlabel('X [m]'); ax3.set_ylabel('Depth [m]'); fig.colorbar(im3, ax=ax3, shrink=0.8)
        fig.suptitle(f'Scientific 3-D Stolt migration | v={r["velocity"]:.3f} m/ns | dx={r["dx"]:.3f}, dy={r["dy"]:.3f} m | k-filter={r["k_filter"]}')
        self.canvas.draw()
        self._status(f'Migration result: cube={r["cube"].shape}, migrated={r["migrated"].shape}, input traces binned={r["n_input"]:,}.')

    def run_validation(self):
        try:
            self._status('Running synthetic migration validation...')
            ok, z0, zp = _synthetic_validation(self)
            msg = f'Synthetic validation: true z={z0:.2f} m, focused peak z={zp:.2f} m -> {"PASS" if ok else "CHECK"}'
            self._status(msg)
            QMessageBox.information(self, 'Synthetic validation', msg)
        except Exception as e:
            QMessageBox.critical(self, 'Validation failed', f'{e}\n\n{traceback.format_exc()}')

    def _default_out(self, suffix):
        root = self._project_root()
        outdir = root / 'migration_exports'
        outdir.mkdir(parents=True, exist_ok=True)
        name = ('schleitheim_mala' if self.kind == 'schleitheim' else 'bulach_pulseekko')
        return outdir / f'{name}_reference_stolt_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{suffix}'

    def export_png(self):
        out = self._default_out('.png')
        self.canvas.fig.savefig(out, dpi=250)
        self._status(f'Saved PNG: {out}')

    def export_npz(self):
        if not self.last:
            QMessageBox.warning(self, 'No migration', 'Run migration first.'); return
        out = self._default_out('.npz')
        r = self.last
        np.savez_compressed(out, migrated=r['migrated'], cube=r['cube'], cube_display=r['cube_display'], valid=r['valid'], count=r['count'], x=r['x'], y=r['y'], time_ns=r['t'], depth_m=r['depth'], elev=r['elev'], params=json.dumps({k: str(v) for k,v in r.items() if k not in {'migrated','cube','cube_display','valid','count','x','y','t','depth','elev','names'}}))
        self._status(f'Saved compressed NPZ volume: {out}')

    def export_segy(self):
        if not self.last:
            QMessageBox.warning(self, 'No migration', 'Run migration first.'); return
        if not SEGYIO_OK:
            QMessageBox.warning(self, 'segyio missing', 'Install segyio first: pip install segyio'); return
        r = self.last
        out = self._default_out('.sgy')
        data = np.nan_to_num(r['migrated'], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        nt, nx, ny = data.shape
        ilines = np.arange(1, nx+1, dtype=np.intc)
        xlines = np.arange(1, ny+1, dtype=np.intc)
        dz = float(np.nanmedian(np.diff(r['depth']))) if len(r['depth']) > 1 else 0.01
        dz_proxy = max(1, min(32767, int(round(dz * 1e6))))
        spec = segyio.spec(); spec.sorting = 2; spec.format = 5; spec.samples = r['depth'].astype(np.float32); spec.ilines = ilines; spec.xlines = xlines
        scalco = -1000; offx = 1000.0; offy = 1000.0
        def ci(m, off): return int(round((float(m)+off)*abs(scalco)))
        with segyio.create(str(out), spec) as f:
            f.bin.update(tsort=2, hdt=dz_proxy, dto=dz_proxy, hns=nt, mfeet=1)
            txt = [
                '3-D GPR POST-STACK MIGRATION Method: STOLT',
                f'EM velocity {r["velocity"]:.4f} m/ns; migration velocity v/2',
                f'Z domain DEPTH metre; z step {dz:.6f} m; samples {nt}',
                f'Grid nx={nx}, ny={ny}; dx={r["dx"]:.4f} m, dy={r["dy"]:.4f} m',
                'Inline byte 189; Crossline byte 193; CDP_X byte 181; CDP_Y byte 185',
                'Coordinate scalar -1000: stored integer /1000 = metres after subtracting 1000 m offset',
                'Override SEG-Y sample interval with true z step in metres in OpenDtect.',
                'SEG Y REV1'
            ]
            while len(txt)<40: txt.append('')
            f.text[0] = ''.join(f'C{i+1:2d} {line}'[:80].ljust(80) for i,line in enumerate(txt[:40])).encode('ascii','replace')
            k=0
            for ix in range(nx):
                for iy in range(ny):
                    f.trace[k] = np.ascontiguousarray(data[:,ix,iy], dtype=np.float32)
                    f.header[k] = {
                        TraceField.TRACE_SEQUENCE_FILE:k+1, TraceField.TRACE_SEQUENCE_LINE:iy+1,
                        TraceField.INLINE_3D:int(ilines[ix]), TraceField.CROSSLINE_3D:int(xlines[iy]),
                        TraceField.CDP_X:ci(r['x'][ix], offx), TraceField.CDP_Y:ci(r['y'][iy], offy),
                        TraceField.SourceX:ci(r['x'][ix], offx), TraceField.SourceY:ci(r['y'][iy], offy),
                        TraceField.GroupX:ci(r['x'][ix], offx), TraceField.GroupY:ci(r['y'][iy], offy),
                        TraceField.SourceGroupScalar:scalco, TraceField.ElevationScalar:scalco,
                        TraceField.CoordinateUnits:1, TraceField.TRACE_SAMPLE_COUNT:nt,
                        TraceField.TRACE_SAMPLE_INTERVAL:dz_proxy, TraceField.DelayRecordingTime:0,
                        TraceField.TraceIdentificationCode:1, TraceField.offset:0, TraceField.NStackedTraces:1}
                    k += 1
        meta = dict(file=str(out), z_step_m=dz, velocity_m_ns=float(r['velocity']), dx_m=float(r['dx']), dy_m=float(r['dy']), inline_byte=189, crossline_byte=193, cdp_x_byte=181, cdp_y_byte=185, coord_scalar=scalco)
        out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
        self._status(f'Saved SEG-Y + JSON companion: {out}')

    def export_gpkg(self):
        if not self.last:
            QMessageBox.warning(self, 'No migration', 'Run migration first.'); return
        if not RASTERIO_OK:
            QMessageBox.warning(self, 'rasterio missing', 'Install rasterio first: pip install rasterio pyproj'); return
        r = self.last
        out = self._default_out('.gpkg')
        if out.exists(): out.unlink()
        data = r['migrated']; x = r['x']; y = r['y']; depth = r['depth']
        # Export 6 representative depth slices as local-metre rasters.
        idxs = np.unique(np.linspace(max(0, int(0.05*len(depth))), max(0, int(0.75*len(depth))), 6).astype(int))
        px = float(np.nanmedian(np.diff(x))) if len(x)>1 else float(r['dx'])
        py = float(np.nanmedian(np.diff(y))) if len(y)>1 else float(r['dy'])
        transform = from_origin(float(x[0]-px/2), float(y[-1]+py/2), abs(px), abs(py))
        for ii, iz in enumerate(idxs):
            layer = f'depth_{int(round(depth[iz]*100)):03d}cm'
            arr = np.flipud(data[iz].T.astype(np.float32))
            with rasterio.open(str(out), 'w', driver='GPKG', dtype='float32', count=1, width=arr.shape[1], height=arr.shape[0], transform=transform, RASTER_TABLE=layer, TILE_FORMAT='TIFF', APPEND_SUBDATASET='YES' if ii else 'NO') as dst:
                dst.write(arr, 1)
        self._status(f'Saved GeoPackage depth slices: {out}')


def _insert_tab_next_to_selected_fence(tabw, widget):
    try:
        idx = tabw.count()
        for i in range(tabw.count()):
            if 'selected fence' in tabw.tabText(i).lower():
                idx = i + 1
                break
        tabw.insertTab(idx, widget, '3D Migration')
        return True
    except Exception:
        return False


def apply_schleitheim(globs):
    cls = globs.get('GPR3DStandardAnalysisTab') or globs.get('GPR3DAnalysisTab')
    if cls is None or getattr(cls, '_gpr3d_stolt_patched', False):
        return
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        try:
            if not hasattr(self, 'gpr3d_migration_tab'):
                self.gpr3d_migration_tab = GPR3DMigrationTab(self, 'schleitheim')
                _insert_tab_next_to_selected_fence(self.tabs, self.gpr3d_migration_tab)
        except Exception as e:
            print('Could not add Schleitheim 3D Migration tab:', e)
    cls.__init__ = new_init
    cls._gpr3d_stolt_patched = True
    print('Scientific 3D Stolt migration patch active for Schleitheim/MALA.')


def apply_bulach(globs):
    cls = globs.get('PulseEkko3DAnalysis')
    if cls is None or getattr(cls, '_gpr3d_stolt_patched', False):
        return
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        try:
            if not hasattr(self, 'gpr3d_migration_tab'):
                self.gpr3d_migration_tab = GPR3DMigrationTab(self, 'bulach')
                _insert_tab_next_to_selected_fence(self.tabs, self.gpr3d_migration_tab)
        except Exception as e:
            print('Could not add Bulach 3D Migration tab:', e)
    cls.__init__ = new_init
    cls._gpr3d_stolt_patched = True
    print('Scientific 3D Stolt migration patch active for Bulach/PulseEKKO.')



# --- compact side-panel layout for 3D migration tab ---
def _pv_compact_sidepanel_build_ui(self):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QVBoxLayout, QGridLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
        QComboBox, QCheckBox, QTextEdit, QWidget, QScrollArea, QSplitter,
        QSizePolicy
    )

    root = QVBoxLayout(self)
    root.setContentsMargins(6, 6, 6, 6)
    root.setSpacing(4)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    root.addWidget(splitter, 1)

    # Left control panel
    left = QWidget()
    left_lay = QVBoxLayout(left)
    left_lay.setContentsMargins(6, 6, 6, 6)
    left_lay.setSpacing(6)

    grid = QGridLayout()
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(5)

    self.data_mode = QComboBox(); self.data_mode.addItems(["processed", "raw"])
    self.attribute = QComboBox(); self.attribute.addItems(["Signed amplitude", "Envelope amplitude", "Absolute amplitude"])
    self.attribute.setCurrentText("Signed amplitude")
    self.k_filter = QComboBox(); self.k_filter.addItems(["Off", "Light k-filter", "Medium k-filter", "Strong k-filter"])
    self.k_filter.setCurrentText("Light k-filter")

    self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.02, 0.30); self.velocity.setValue(0.10); self.velocity.setSingleStep(0.005); self.velocity.setSuffix(" m/ns")
    self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(0.0); self.tmin.setSuffix(" ns")
    self.tmax = QDoubleSpinBox(); self.tmax.setRange(0, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(" ns")
    self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20); self.dt_out.setValue(0.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(" ns")

    self.grid_dx = QDoubleSpinBox(); self.grid_dx.setRange(0.02, 2.0); self.grid_dx.setValue(0.25); self.grid_dx.setSingleStep(0.05); self.grid_dx.setSuffix(" m")
    self.grid_dy = QDoubleSpinBox(); self.grid_dy.setRange(0.02, 2.0); self.grid_dy.setValue(0.25); self.grid_dy.setSingleStep(0.05); self.grid_dy.setSuffix(" m")
    self.max_nx = QSpinBox(); self.max_nx.setRange(20, 512); self.max_nx.setValue(180)
    self.max_ny = QSpinBox(); self.max_ny.setRange(20, 512); self.max_ny.setValue(140)

    self.trace_step = QSpinBox(); self.trace_step.setRange(1, 100); self.trace_step.setValue(5)
    self.max_lines = QSpinBox(); self.max_lines.setRange(1, 10000); self.max_lines.setValue(500)
    self.pad_t = QDoubleSpinBox(); self.pad_t.setRange(1.0, 4.0); self.pad_t.setValue(1.5); self.pad_t.setSingleStep(0.25)
    self.pad_xy = QSpinBox(); self.pad_xy.setRange(0, 100); self.pad_xy.setValue(15); self.pad_xy.setSuffix(" cells")
    self.taper_w = QSpinBox(); self.taper_w.setRange(0, 50); self.taper_w.setValue(5); self.taper_w.setSuffix(" cells")

    self.depth_slice = QDoubleSpinBox(); self.depth_slice.setRange(0.0, 20.0); self.depth_slice.setValue(0.60); self.depth_slice.setSingleStep(0.05); self.depth_slice.setSuffix(" m")
    self.clip_pct = QDoubleSpinBox(); self.clip_pct.setRange(80.0, 100.0); self.clip_pct.setValue(98.5); self.clip_pct.setSingleStep(0.1); self.clip_pct.setSuffix(" %")

    entries = [
        ("Data", self.data_mode),
        ("Attribute", self.attribute),
        ("Velocity", self.velocity),
        ("tmin", self.tmin),
        ("tmax", self.tmax),
        ("dt", self.dt_out),
        ("Grid dx", self.grid_dx),
        ("Grid dy", self.grid_dy),
        ("Max nx", self.max_nx),
        ("Max ny", self.max_ny),
        ("Trace step", self.trace_step),
        ("Max lines", self.max_lines),
        ("k-filter", self.k_filter),
        ("Pad t", self.pad_t),
        ("Pad xy", self.pad_xy),
        ("Taper", self.taper_w),
        ("Bird's-eye depth slice", self.depth_slice),
        ("Clip", self.clip_pct),
    ]

    for r, (lab, widget) in enumerate(entries):
        grid.addWidget(QLabel(lab), r, 0)
        grid.addWidget(widget, r, 1)

    left_lay.addLayout(grid)

    self.live_taper = QCheckBox("Live/dead mask taper"); self.live_taper.setChecked(True)
    self.blank_dead = QCheckBox("Re-blank dead cells"); self.blank_dead.setChecked(True)
    self.blank_topo = QCheckBox("Topographic blanking"); self.blank_topo.setChecked(True)
    self.jacobian = QCheckBox("Stolt Jacobian"); self.jacobian.setChecked(True)

    for w in (self.live_taper, self.blank_dead, self.blank_topo, self.jacobian):
        left_lay.addWidget(w)

    self.btn_run = QPushButton("Build volume + run 3-D Stolt migration")
    self.btn_validate = QPushButton("Run synthetic validation")
    self.btn_png = QPushButton("Export PNG")
    self.btn_npz = QPushButton("Export NPZ volume")
    self.btn_segy = QPushButton("Export SEG-Y" + ("" if SEGYIO_OK else " (needs segyio)"))
    self.btn_gis = QPushButton("Export GeoPackage slices" + ("" if RASTERIO_OK else " (needs rasterio)"))

    for b in (self.btn_run, self.btn_validate, self.btn_png, self.btn_npz, self.btn_segy, self.btn_gis):
        b.setMinimumHeight(28)
        left_lay.addWidget(b)

    left_lay.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(left)
    scroll.setMinimumWidth(330)
    scroll.setMaximumWidth(420)
    splitter.addWidget(scroll)

    # Right plot/log panel
    right = QWidget()
    right_lay = QVBoxLayout(right)
    right_lay.setContentsMargins(4, 0, 4, 4)
    right_lay.setSpacing(4)

    self.canvas = _PVCanvas(14, 8)
    self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    self.canvas.setMinimumHeight(260)
    right_lay.addWidget(self.canvas, 1)

    self.log = QTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumHeight(55)
    self.log.setPlaceholderText("3D migration status/output log")
    right_lay.addWidget(self.log, 0)

    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([360, 1200])

    self.btn_run.clicked.connect(self.run_pipeline)
    self.btn_validate.clicked.connect(self.run_validation)
    self.btn_png.clicked.connect(self.export_png)
    self.btn_npz.clicked.connect(self.export_npz)
    self.btn_segy.clicked.connect(self.export_segy)
    self.btn_gis.clicked.connect(self.export_gpkg)

GPR3DMigrationTab._build_ui = _pv_compact_sidepanel_build_ui
# --- end compact side-panel layout for 3D migration tab ---




# --- force screen-fitting main window patch ---
def _pv_fit_window_to_screen(win):
    try:
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        w = max(900, min(1500, int(g.width() * 0.94)))
        h = max(650, min(900, int(g.height() * 0.88)))
        win.setMinimumSize(900, 600)
        win.resize(w, h)
        win.move(g.x() + max(0, (g.width() - w) // 2), g.y() + max(0, (g.height() - h) // 2))
    except Exception as e:
        print("Screen-fit patch failed:", e)

def _pv_patch_mainwindow_screenfit(cls):
    if cls is None or getattr(cls, "_pv_screenfit_patched", False):
        return
    old = cls.__init__
    def new_init(self, *a, _old=old, **kw):
        _old(self, *a, **kw)
        _pv_fit_window_to_screen(self)
    cls.__init__ = new_init
    cls._pv_screenfit_patched = True

_old_apply_schleitheim_screenfit = globals().get("apply_schleitheim")
def apply_schleitheim(globs):
    if _old_apply_schleitheim_screenfit:
        _old_apply_schleitheim_screenfit(globs)
    _pv_patch_mainwindow_screenfit(globs.get("FieldworkMainWindow") or globs.get("MainWindow"))

_old_apply_bulach_screenfit = globals().get("apply_bulach")
def apply_bulach(globs):
    if _old_apply_bulach_screenfit:
        _old_apply_bulach_screenfit(globs)
    _pv_patch_mainwindow_screenfit(globs.get("FieldworkMainWindow") or globs.get("MainWindow"))
# --- end force screen-fitting main window patch ---




# --- reference workflow upgrade patch ---
# This block tightens the GUI migration pipeline against the reference notebooks:
#   * zero-time preserved: migration cube starts at t=0; tmin is a mute, not a crop
#   * survey-axis/PCA rotation before binning, instead of raw easting/northing axes
#   * scientific mean binning onto a regular cube
#   * scientific inline Butterworth k-filter, not generic image smoothing
#   * optional elevation statics before migration using the cs_2D_3D_ElevationStatics convention
#   * true 3-D live-sample taper, including time direction
#   * scientific before/after section plot, rather than only a QC preview plot

import numpy as _np
import math as _math

try:
    from scipy import signal as _pv_signal
    from scipy.ndimage import uniform_filter as _pv_uniform_filter
    from scipy.signal import fftconvolve as _pv_fftconvolve
    _PV_SCIPY_EXTRA = True
except Exception:
    _PV_SCIPY_EXTRA = False


def _pv_safe_std(a):
    a = _np.asarray(a, float)
    a = a[_np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = 1.5 * _np.sqrt(_np.mean(_np.var(a.reshape(a.shape[0], -1), axis=1))) if a.ndim > 1 else 1.5 * _np.nanstd(a)
    if not _np.isfinite(v) or v <= 0:
        v = float(_np.nanpercentile(_np.abs(a), 98.5)) if a.size else 1.0
    return float(v if v > 0 else 1.0)


def _pv_pca_survey_axes(x, y):
    """Return scientific local survey axes u/v from arbitrary XY points."""
    x = _np.asarray(x, float); y = _np.asarray(y, float)
    ok = _np.isfinite(x) & _np.isfinite(y)
    if ok.sum() < 3:
        return x.copy(), y.copy(), dict(center=(0.0,0.0), e0=(1.0,0.0), e1=(0.0,1.0), rotated=False)
    pts = _np.column_stack([x[ok], y[ok]])
    cen = pts.mean(axis=0)
    q = pts - cen
    cov = _np.cov(q.T)
    vals, vecs = _np.linalg.eigh(cov)
    order = _np.argsort(vals)[::-1]
    e0 = vecs[:, order[0]]
    e1 = vecs[:, order[1]]
    # Stable sign: increasing x mostly to the right.
    if e0[0] < 0:
        e0 = -e0
    # Right-handed across-axis.
    e1 = _np.array([-e0[1], e0[0]])
    allpts = _np.column_stack([x, y]) - cen
    u = allpts @ e0
    v = allpts @ e1
    return u, v, dict(center=(float(cen[0]), float(cen[1])), e0=(float(e0[0]), float(e0[1])), e1=(float(e1[0]), float(e1[1])), rotated=True)


def _pv_nanmean_filter2d(z, size=(15,5)):
    z = _np.asarray(z, float)
    if not _PV_SCIPY_EXTRA:
        return z
    mask = _np.isfinite(z)
    if not mask.any():
        return z
    sz = tuple(int(max(1, s)) for s in size)
    filled = _np.where(mask, z, 0.0)
    win_area = float(sz[0] * sz[1])
    sm_sum = _pv_uniform_filter(filled, size=sz, mode='nearest') * win_area
    sm_cnt = _pv_uniform_filter(mask.astype(float), size=sz, mode='nearest') * win_area
    with _np.errstate(invalid='ignore', divide='ignore'):
        out = sm_sum / sm_cnt
    out[sm_cnt <= 0] = _np.nan
    return out


def _pv_apply_elevation_statics_cube(cube, elev_grid, valid, vel, dt, window=(15,5)):
    """Approximate reference cs_2D_3D_ElevationStatics on a regular cube.
    Positive static inserts zero samples before the trace, matching the notebook convention.
    """
    elev = _np.asarray(elev_grid, float)
    if elev.shape != valid.shape or not _np.isfinite(elev[valid]).any():
        return cube, dict(applied=False)
    z_smooth = _pv_nanmean_filter2d(elev, window)
    datum = float(_np.nanmax(z_smooth[valid]))
    # Reference formula: elevStat = (2*(datum-smoothelev)/velcorr)/dt
    stat = (2.0 * (datum - z_smooth) / max(float(vel), 1e-9)) / max(float(dt), 1e-9)
    stat_i = _np.zeros(valid.shape, dtype=int)
    ok = valid & _np.isfinite(stat)
    stat_i[ok] = _np.fix(stat[ok]).astype(int)
    nt, nx, ny = cube.shape
    maxstat = int(max(0, stat_i[ok].max() if ok.any() else 0))
    new_nt = nt + maxstat
    out = _np.zeros((new_nt, nx, ny), dtype=_np.float32)
    for ix in range(nx):
        for iy in range(ny):
            s = int(stat_i[ix, iy]) if valid[ix, iy] else 0
            if s >= 0:
                out[s:s+nt, ix, iy] = cube[:, ix, iy]
            else:
                src = -s
                if src < nt:
                    out[:nt-src, ix, iy] = cube[src:, ix, iy]
    return out, dict(applied=True, datum=datum, z_smooth=z_smooth.astype(_np.float32), applied_statics_samples=stat_i, max_static=maxstat)


def _pv_reference_k_filter(cube, dx, dy, mode):
    """Scientific low-pass k-filter along the inline/X axis per line.
    The notebooks use a Butterworth low-pass and filtfilt along x for each inline.
    """
    mode = (mode or 'Off').lower()
    if mode.startswith('off') or not _PV_SCIPY_EXTRA:
        return cube.astype(_np.float32, copy=False)
    nx = cube.shape[1]
    if nx < 12:
        return cube.astype(_np.float32, copy=False)
    f_ny = 1.0 / (2.0 * max(float(dx), 1e-9))
    # Reference example was cutoff ≈ 0.4 of Nyquist. Keep named strengths around that.
    if 'strong' in mode:
        frac = 0.22
    elif 'medium' in mode:
        frac = 0.32
    else:
        frac = 0.40
    wn = min(max(frac, 0.03), 0.95)
    b, a = _pv_signal.butter(4, wn, btype='low')
    try:
        padlen = min(3 * max(len(a), len(b)), nx - 2)
        return _pv_signal.filtfilt(b, a, cube.astype(float), axis=1, padlen=padlen).astype(_np.float32)
    except Exception:
        return cube.astype(_np.float32, copy=False)


def _pv_live_sample_taper3d(cube, valid2d, taper_xy=5, taper_t=25):
    """Reference live-sample taper: convolve binary live sample mask with separable Hann kernel in t/x/y."""
    if not _PV_SCIPY_EXTRA:
        return cube.astype(_np.float32, copy=False), None
    nt, nx, ny = cube.shape
    hw_t = int(max(0, min(taper_t, nt // 3)))
    hw_x = int(max(0, min(taper_xy, nx // 3)))
    hw_y = int(max(0, min(taper_xy, ny // 3)))
    if hw_t <= 0 and hw_x <= 0 and hw_y <= 0:
        return cube.astype(_np.float32, copy=False), None
    def hann_hw(hw):
        if hw <= 0:
            return _np.array([1.0], dtype=_np.float32)
        w = _np.hanning(2 * hw + 1).astype(_np.float32)
        s = float(w.sum())
        return w / s if s > 0 else _np.array([1.0], dtype=_np.float32)
    kt, kx, ky = hann_hw(hw_t), hann_hw(hw_x), hann_hw(hw_y)
    kern = kt[:, None, None] * kx[None, :, None] * ky[None, None, :]
    kern = kern / max(float(kern.sum()), 1e-12)
    live = (_np.abs(cube) > 0).astype(_np.float32)
    live[:, ~valid2d] = 0.0
    taper = _pv_fftconvolve(live, kern.astype(_np.float32), mode='same')
    taper = _np.clip(taper, 0.0, 1.0).astype(_np.float32)
    return (cube.astype(_np.float32) * taper).astype(_np.float32), taper


def _pv_collect_traces_t0_mute(self, progress=None):
    """Collect traces on a time axis starting at zero; tmin is used only as a mute.
    This is important because Stolt migration assumes sample 0 corresponds to t=0.
    """
    lines = self._selected_lines()
    if not lines:
        raise RuntimeError('No selected lines for migration.')
    lo, hi = sorted([float(self.tmin.value()), float(self.tmax.value())])
    if hi <= 0:
        raise RuntimeError('tmax must be > 0 for Stolt migration.')
    dt = float(self.dt_out.value())
    t_axis = _np.arange(0.0, hi + 0.5 * dt, dt, dtype=float)
    xs=[]; ys=[]; zs=[]; data=[]; names=[]
    step = max(1, int(self.trace_step.value()))
    for k, line in enumerate(lines, 1):
        if progress:
            progress(2 + int(18 * k / max(1, len(lines))), f'Collecting traces {k}/{len(lines)}: {getattr(line,"name",line)}')
        arr, t, x, y, z = self._line_array_time_xyze(line)
        if arr is None or arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 8:
            continue
        x = _np.asarray(x, float); y = _np.asarray(y, float); z = _np.asarray(z, float); t = _np.asarray(t, float)
        if x.size != arr.shape[0] and x.size >= 2:
            x = _np.interp(_np.linspace(0,1,arr.shape[0]), _np.linspace(0,1,x.size), x)
        if y.size != arr.shape[0] and y.size >= 2:
            y = _np.interp(_np.linspace(0,1,arr.shape[0]), _np.linspace(0,1,y.size), y)
        if z.size != arr.shape[0] and z.size >= 2:
            z = _np.interp(_np.linspace(0,1,arr.shape[0]), _np.linspace(0,1,z.size), z)
        elif z.size != arr.shape[0]:
            z = _np.full(arr.shape[0], _np.nan)
        for ii in range(0, arr.shape[0], step):
            tr = _np.interp(t_axis, t, arr[ii, :], left=0.0, right=0.0).astype(_np.float32)
            tr[t_axis < lo] = 0.0
            if not _np.any(_np.isfinite(tr)):
                continue
            xs.append(float(x[ii])); ys.append(float(y[ii])); zs.append(float(z[ii]) if _np.isfinite(z[ii]) else _np.nan)
            data.append(tr); names.append(getattr(line, 'name', str(line)))
    if not data:
        raise RuntimeError('No usable traces in selected time window.')
    return _np.asarray(xs), _np.asarray(ys), _np.asarray(zs), _np.vstack(data).astype(_np.float32), t_axis, names


def _pv_build_cube_reference(self, progress=None):
    xw, yw, z, tr, t_axis, names = _pv_collect_traces_t0_mute(self, progress)
    # Rotate to local survey coordinates before binning, matching the reference inline/xline cube assumption.
    x, y, rot = _pv_pca_survey_axes(xw, yw)
    dx_req = float(self.grid_dx.value()); dy_req = float(self.grid_dy.value())
    xmin = float(_np.floor(_np.nanmin(x) / dx_req) * dx_req)
    xmax = float(_np.ceil( _np.nanmax(x) / dx_req) * dx_req)
    ymin = float(_np.floor(_np.nanmin(y) / dy_req) * dy_req)
    ymax = float(_np.ceil( _np.nanmax(y) / dy_req) * dy_req)
    nx = int(round(1 + (xmax - xmin) / dx_req)); ny = int(round(1 + (ymax - ymin) / dy_req))
    # Avoid silently coarsening too much; scientific default should preserve 0.25 m if possible.
    max_nx, max_ny = int(self.max_nx.value()), int(self.max_ny.value())
    if nx > max_nx:
        dx_req = (xmax - xmin) / max(max_nx - 1, 1); nx = max_nx
    if ny > max_ny:
        dy_req = (ymax - ymin) / max(max_ny - 1, 1); ny = max_ny
    xi = xmin + _np.arange(nx) * dx_req
    yi = ymin + _np.arange(ny) * dy_req
    ix = _np.rint((x - xmin) / max(dx_req, 1e-9)).astype(int)
    iy = _np.rint((y - ymin) / max(dy_req, 1e-9)).astype(int)
    ok = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&_np.isfinite(x)&_np.isfinite(y)&_np.all(_np.isfinite(tr), axis=1)
    ix, iy, tr, z_ok = ix[ok], iy[ok], tr[ok], z[ok]
    nt = tr.shape[1]
    cube = _np.zeros((nt, nx, ny), dtype=_np.float32)
    count = _np.zeros((nx, ny), dtype=_np.float32)
    elev_sum = _np.zeros((nx, ny), dtype=float); elev_count = _np.zeros((nx, ny), dtype=float)
    total = int(tr.shape[0])
    if progress:
        progress(22, f'Mean-binning {total:,} traces into regular rotated cube {nt}×{nx}×{ny}...')
    for j in range(total):
        cube[:, ix[j], iy[j]] += tr[j]
        count[ix[j], iy[j]] += 1.0
        if _np.isfinite(z_ok[j]):
            elev_sum[ix[j], iy[j]] += z_ok[j]
            elev_count[ix[j], iy[j]] += 1.0
        if progress and (j % max(1, total//20) == 0):
            progress(22 + int(16*j/max(1,total)), f'Binning trace {j+1:,}/{total:,}...')
    valid = count > 0
    cube[:, valid] /= count[valid][None, :]
    elev_grid = _np.full((nx, ny), _np.nan, dtype=_np.float32)
    ge = elev_count > 0
    elev_grid[ge] = (elev_sum[ge] / elev_count[ge]).astype(_np.float32)
    # world coordinates of rotated grid centres for future export/QC
    cen = _np.asarray(rot['center']); e0 = _np.asarray(rot['e0']); e1 = _np.asarray(rot['e1'])
    U, V = _np.meshgrid(xi, yi, indexing='ij')
    world_x = cen[0] + U * e0[0] + V * e1[0]
    world_y = cen[1] + U * e0[1] + V * e1[1]
    return dict(cube=cube, valid=valid, count=count, elev=elev_grid, x=xi.astype(float), y=yi.astype(float),
                world_x=world_x.astype(_np.float32), world_y=world_y.astype(_np.float32), rotation=rot,
                dx=float(dx_req), dy=float(dy_req), t=t_axis.astype(float), n_input=total, names=names,
                tmin_mute_ns=float(self.tmin.value()), zero_time_preserved=True)


def _pv_pipeline_gpr3d_reference(self, progress=None):
    out = _pv_build_cube_reference(self, progress)
    cube0 = out['cube'].astype(_np.float32)
    valid = out['valid']
    if progress:
        progress(39, 'Scientific preprocessing: k-filter → elevation statics → live-sample taper...')
    # Display attribute is separate; migration itself remains signed amplitude, as in the 3-D Stolt notebook.
    attr = self.attribute.currentText()
    cube_display = cube0.copy()
    if attr == 'Envelope amplitude':
        cube_display = _envelope(cube_display)
    elif attr == 'Absolute amplitude':
        cube_display = _np.abs(cube_display).astype(_np.float32)
    cube_proc = _pv_reference_k_filter(cube0, out['dx'], out['dy'], self.k_filter.currentText())
    cube_proc[:, ~valid] = 0.0
    vel = float(self.velocity.value())
    dt = float(_np.nanmedian(_np.diff(out['t']))) if len(out['t']) > 1 else float(self.dt_out.value())
    # Elevation statics before migration, using reference convention. Applied only when elevation exists.
    cube_stat, stat_info = _pv_apply_elevation_statics_cube(cube_proc, out['elev'], valid, vel, dt, window=(15,5))
    if stat_info.get('applied'):
        out['elev'] = stat_info.get('z_smooth', out['elev'])
        out['final_datum'] = stat_info.get('datum')
        out['applied_statics_samples'] = stat_info.get('applied_statics_samples')
        out['elevation_statics'] = True
        if progress:
            progress(43, f'Elevation statics applied; max static={stat_info.get("max_static",0)} samples.')
    else:
        out['elevation_statics'] = False
    if self.live_taper.isChecked():
        cube_pretaper = cube_stat.copy()
        cube_taper, taper_vol = _pv_live_sample_taper3d(cube_stat, valid, int(self.taper_w.value()), 25)
    else:
        cube_pretaper = cube_stat.copy(); cube_taper = cube_stat; taper_vol = None
    pad_cells = int(self.pad_xy.value())
    pad_x = (cube_taper.shape[1] + pad_cells) / max(cube_taper.shape[1], 1)
    pad_y = (cube_taper.shape[2] + pad_cells) / max(cube_taper.shape[2], 1)
    taper_frac_x = min(0.5, int(self.taper_w.value()) / max(cube_taper.shape[1], 1))
    taper_frac_y = min(0.5, int(self.taper_w.value()) / max(cube_taper.shape[2], 1))
    mig = stolt_migration_3d(cube_taper, dt=dt, dx=out['dx'], dy=out['dy'], velocity=vel,
                             dz=(vel/2.0)*dt, nz=cube_taper.shape[0], exploding_reflector=True,
                             apply_jacobian=self.jacobian.isChecked(), pad_t=float(self.pad_t.value()),
                             pad_x=pad_x, pad_y=pad_y, taper_t=0.05, taper_x=taper_frac_x, taper_y=taper_frac_y,
                             depth_padding=2.0, progress_cb=progress)
    if self.blank_dead.isChecked():
        mig[:, ~valid] = 0.0
    depth = (vel / 2.0) * _np.arange(mig.shape[0], dtype=float) * dt
    if self.blank_topo.isChecked() and _np.isfinite(out.get('elev', _np.nan)).any() and out.get('final_datum') is not None:
        dz = max(float((vel/2.0)*dt), 1e-9)
        surf = float(out['final_datum']) - _np.asarray(out['elev'], float)
        n_blank = _np.where(_np.isfinite(surf), _np.floor(surf/dz).astype(int), 0)
        n_blank = _np.clip(n_blank, 0, mig.shape[0])
        mig[_np.arange(mig.shape[0])[:,None,None] < n_blank[None,:,:]] = 0.0
        out['n_blank'] = n_blank
    out.update(dict(cube=cube0.astype(_np.float32), cube_premig=cube_pretaper.astype(_np.float32), cube_migration_input=cube_taper.astype(_np.float32),
                    taper_volume=taper_vol, migrated=mig.astype(_np.float32), depth=depth.astype(_np.float32), cube_display=cube_display,
                    velocity=vel, dt=dt, attribute=attr, k_filter='Butterworth ' + self.k_filter.currentText(),
                    created=datetime.datetime.now().isoformat()))
    return out


def _pv_plot_result_gpr3d_reference(self):
    if not self.last:
        return
    r = self.last
    fig = self.canvas.fig; fig.clear()
    # Reference default display: before/after inline and crossline sections.
    nx, ny = r['migrated'].shape[1], r['migrated'].shape[2]
    il_mid = nx // 2; xl_mid = ny // 2
    x_axis = r['x']; y_axis = r['y']; t_axis = r['t']; depth = r['depth']
    cube_in = r.get('cube_premig', r.get('cube', r['cube_display']))
    mig = r['migrated']
    ax = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    panels = [
        (ax[0], cube_in[:, il_mid, :], y_axis, t_axis, 'Across-survey position [m]', 'TWT [ns]', f'Inline section #{il_mid+1} — input'),
        (ax[1], mig[:, il_mid, :],     y_axis, depth,  'Across-survey position [m]', 'Depth [m]', f'Inline section #{il_mid+1} — migrated'),
        (ax[2], cube_in[:, :, xl_mid], x_axis, t_axis, 'Along-survey position [m]',  'TWT [ns]', f'Crossline section #{xl_mid+1} — input'),
        (ax[3], mig[:, :, xl_mid],     x_axis, depth,  'Along-survey position [m]',  'Depth [m]', f'Crossline section #{xl_mid+1} — migrated'),
    ]
    clim0 = _pv_safe_std(cube_in[:, il_mid, :])
    clim1 = _pv_safe_std(cube_in[:, :, xl_mid])
    clims = [clim0, clim0, clim1, clim1]
    for i, (a, data2d, lateral, vertical, xlabel, ylabel, title) in enumerate(panels):
        im = a.imshow(data2d, aspect='auto', cmap='gray', origin='upper',
                      extent=[float(lateral[0]), float(lateral[-1]), float(vertical[-1]), float(vertical[0])],
                      vmin=-clims[i], vmax=clims[i])
        a.set_title(title, fontsize=10)
        a.set_xlabel(xlabel); a.set_ylabel(ylabel)
        fig.colorbar(im, ax=a, shrink=0.75, label='Amplitude')
    fig.suptitle('3-D Stolt Migration — reference workflow view | '
                 f'v={r["velocity"]:.3f} m/ns | dx={r["dx"]:.3f}, dy={r["dy"]:.3f} m | '
                 f'{r["k_filter"]} | t0 preserved; tmin mute={r.get("tmin_mute_ns",0):.1f} ns', fontsize=12)
    self.canvas.draw()
    live = int(_np.sum(r['valid'])); total = int(r['valid'].size)
    self._status(f'Reference-parity 3-D migration result: cube={r["cube"].shape}, migrated={r["migrated"].shape}, live cells={live}/{total}, input traces binned={r["n_input"]:,}.')


def _pv_synthetic_validation_reference(parent=None):
    # Test 1: point diffractor focuses near the correct depth.
    nt, nx, ny = 256, 64, 64
    dt, dx, dy, vel = 0.4, 0.05, 0.05, 0.10
    z0 = 0.75
    x = (_np.arange(nx) - nx//2) * dx
    y = (_np.arange(ny) - ny//2) * dy
    tt = _np.arange(nt) * dt
    X, Y = _np.meshgrid(x, y, indexing='ij')
    twt = 2.0 * _np.sqrt(z0*z0 + X*X + Y*Y) / vel
    data = _np.zeros((nt, nx, ny), dtype=_np.float32)
    f0 = 0.8
    for ix in range(nx):
        for iy in range(ny):
            data[:, ix, iy] = _ricker(tt - twt[ix, iy], f0)
    img = stolt_migration_3d(data, dt, dx, dy, vel, pad_t=2.0, pad_x=1.5, pad_y=1.5,
                             taper_t=0.05, taper_x=0.1, taper_y=0.1, nz=nt)
    depth = (vel/2.0) * tt
    peak = _np.unravel_index(_np.nanargmax(_np.abs(img)), img.shape)
    z_peak = float(depth[peak[0]])
    ok_focus = abs(z_peak - z0) <= max(0.10, 3*(vel/2.0)*dt)
    # Test 2: single surface-trace impulse should produce a hemispherical impulse response, not a focused point.
    imp = _np.zeros((128, 41, 41), dtype=_np.float32)
    imp[50, 20, 20] = 1.0
    img_imp = stolt_migration_3d(imp, 0.4, 0.05, 0.05, vel, pad_t=2.0, pad_x=1.5, pad_y=1.5,
                                 taper_t=0.05, taper_x=0.1, taper_y=0.1, nz=128)
    nonzero = int(_np.count_nonzero(_np.abs(img_imp) > 0.05 * _np.nanmax(_np.abs(img_imp))))
    ok_impulse = nonzero > 20
    return (ok_focus and ok_impulse), z0, z_peak


def _pv_run_validation_reference(self):
    try:
        self._status('Running scientific synthetic validation: point diffractor + impulse response...')
        ok, z0, zp = _pv_synthetic_validation_reference(self)
        msg = (f'Synthetic validation: point true z={z0:.2f} m, focused peak z={zp:.2f} m; '
               f'impulse response checked -> {"PASS" if ok else "CHECK"}')
        self._status(msg)
        QMessageBox.information(self, 'Synthetic validation', msg)
    except Exception as e:
        QMessageBox.critical(self, 'Validation failed', f'{e}\n\n{traceback.format_exc()}')


def _pv_init_gpr3d_reference_controls(self):
    # Keep screen-fit layout, but make defaults closer to the reference notebook.
    try:
        if hasattr(self, 'dt_out'):
            self.dt_out.setValue(0.5)
        if hasattr(self, 'max_nx') and self.max_nx.value() < 220:
            self.max_nx.setValue(240)
        if hasattr(self, 'max_ny') and self.max_ny.value() < 90:
            self.max_ny.setValue(120)
        if hasattr(self, 'k_filter'):
            self.k_filter.setToolTip('Reference workflow: Butterworth low-pass along the along-survey/X axis, applied per inline before migration.')
        if hasattr(self, 'tmin'):
            self.tmin.setToolTip('Reference workflow: tmin is a mute. The migration cube still starts at t=0 ns.')
        if hasattr(self, 'grid_dx'):
            self.grid_dx.setToolTip('Binning grid spacing along rotated survey axis. 0.25 m is safe; lower values are closer to dense trace spacing but slower.')
    except Exception:
        pass

# Patch class methods after all earlier UI/layout patches.
try:
    _old_pv_init_gpr3d_reference = GPR3DMigrationTab.__init__
    if not getattr(GPR3DMigrationTab, '_pv_gpr3d_reference_init_patched', False):
        def _new_init_gpr3d_reference(self, *a, _old=_old_pv_init_gpr3d_reference, **kw):
            _old(self, *a, **kw)
            _pv_init_gpr3d_reference_controls(self)
        GPR3DMigrationTab.__init__ = _new_init_gpr3d_reference
        GPR3DMigrationTab._pv_gpr3d_reference_init_patched = True
    GPR3DMigrationTab._build_cube = _pv_build_cube_reference
    GPR3DMigrationTab._pipeline = _pv_pipeline_gpr3d_reference
    GPR3DMigrationTab.plot_result = _pv_plot_result_gpr3d_reference
    GPR3DMigrationTab.run_validation = _pv_run_validation_reference
    _synthetic_validation = _pv_synthetic_validation_reference
    print('Reference workflow upgrade active: t0-preserving mute, rotated mean binning, Butterworth k-filter, elevation statics, 3-D live taper, before/after plots.')
except Exception as _e:
    print('Reference workflow upgrade could not be installed:', _e)
# --- end reference workflow upgrade patch ---




# --- professional migration cleanup and added workflow features ---
# Adds missing reference-workflow controls without exposing academic/source wording in the GUI:
#   * view-mode selector for section comparison, depth-slice/mask, input diagnostics, and live-taper QC
#   * optional cross-correlation line-shift alignment, matching the reference notebook concept
#   * MAT export for migrated volume exchange
#   * neutral labels/messages/filenames

import numpy as _gpr3d_np

try:
    from scipy.io import savemat as _gpr3d_savemat
    _GPR3D_SCIPY_IO_OK = True
except Exception:
    _GPR3D_SCIPY_IO_OK = False


def _gpr3d_tab_class():
    return globals().get('GPR3DMigrationTab') or globals().get('Reference3DMigrationTab')


def _gpr3d_clean_text(txt):
    txt = str(txt)
    for a, b in [
        ('Scientific', 'Scientific'), ('scientific', 'scientific'),
        ('Reference workflow', 'Reference workflow'), ('reference workflow', 'reference workflow'),
        ('3-D', '3-D'), ('3-D', '3-D'),
        ('Reference workflow', 'Reference workflow'), ('reference workflow', 'reference workflow'),
        ('Butterworth ', 'Butterworth '),
        ('before/after', 'before/after'),
        ('reference', 'reference'), ('Reference', 'Reference'),
        ('scientific', 'scientific'), ('Scientific', 'Scientific'),
        ('Reference package', 'processing package'), ('reference package', 'processing package'),
    ]:
        txt = txt.replace(a, b)
    return txt


def _gpr3d_patch_status_method(cls):
    old_status = getattr(cls, '_status', None)
    if old_status is None or getattr(old_status, '_gpr3d_cleaned', False):
        return
    def _status_clean(self, txt):
        return old_status(self, _gpr3d_clean_text(txt))
    _status_clean._gpr3d_cleaned = True
    cls._status = _status_clean


def _gpr3d_shift_1d_zero(a, shift):
    a = _gpr3d_np.asarray(a)
    if shift == 0:
        return a.copy()
    out = _gpr3d_np.zeros_like(a)
    if shift > 0:
        out[shift:] = a[:-shift]
    else:
        out[:shift] = a[-shift:]
    return out


def _gpr3d_shift_rows_zero(arr, shift):
    arr = _gpr3d_np.asarray(arr)
    if shift == 0:
        return arr.copy()
    out = _gpr3d_np.zeros_like(arr)
    if shift > 0:
        out[shift:, :] = arr[:-shift, :]
    else:
        out[:shift, :] = arr[-shift:, :]
    return out


def _gpr3d_best_shift_energy(ref, cur, max_shift):
    ref = _gpr3d_np.asarray(ref, float)
    cur = _gpr3d_np.asarray(cur, float)
    n = min(ref.size, cur.size)
    if n < 8:
        return 0
    ref = ref[:n]; cur = cur[:n]
    ref = ref - _gpr3d_np.nanmean(ref)
    cur = cur - _gpr3d_np.nanmean(cur)
    best_s, best_c = 0, -_gpr3d_np.inf
    for s in range(-int(max_shift), int(max_shift)+1):
        cs = _gpr3d_shift_1d_zero(cur, s)
        c = float(_gpr3d_np.nansum(ref * cs))
        if c > best_c:
            best_c, best_s = c, s
    return int(best_s)


def _gpr3d_add_extra_controls(self):
    if getattr(self, '_gpr3d_extra_controls_added', False):
        return
    try:
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QPushButton, QScrollArea
        # Controls
        self.view_mode = QComboBox()
        self.view_mode.addItems(['Section comparison', 'Depth slice + mask', 'Input diagnostics', 'Live taper QC'])
        self.view_mode.setToolTip('Controls how the migrated result is displayed after the volume is built.')
        self.xcorr_align = QCheckBox('XCorr line-shift alignment')
        self.xcorr_align.setChecked(False)
        self.xcorr_align.setToolTip('Optional trace-position alignment by cross-correlating line energy. Use for QC; compare with OFF before interpretation.')
        self.max_shift = QSpinBox(); self.max_shift.setRange(0, 20); self.max_shift.setValue(4); self.max_shift.setSuffix(' traces')
        self.btn_mat = QPushButton('Export MAT' + ('' if _GPR3D_SCIPY_IO_OK else ' (needs scipy)'))
        try:
            self.btn_mat.clicked.connect(self.export_mat)
        except Exception:
            pass
        row = QWidget(self)
        lay = QHBoxLayout(row); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        for w in (QLabel('View'), self.view_mode, self.xcorr_align, QLabel('Max shift'), self.max_shift, self.btn_mat):
            lay.addWidget(w)
        lay.addStretch(1)
        # Prefer inserting into the left scroll/control panel, if the compact layout exists.
        inserted = False
        for scroll in self.findChildren(QScrollArea):
            w = scroll.widget()
            if w is not None and w.layout() is not None:
                w.layout().insertWidget(0, row)
                inserted = True
                break
        if not inserted and self.layout() is not None:
            self.layout().insertWidget(0, row)
        try:
            self.view_mode.currentTextChanged.connect(lambda *_: self.plot_result() if getattr(self, 'last', None) else None)
        except Exception:
            pass
        self._gpr3d_extra_controls_added = True
    except Exception as e:
        print('Could not add 3-D migration extra controls:', e)


def _gpr3d_install_extra_ui(cls):
    old_init = cls.__init__
    if getattr(old_init, '_gpr3d_extra_ui_patched', False):
        return
    def _new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        _gpr3d_add_extra_controls(self)
    _new_init._gpr3d_extra_ui_patched = True
    cls.__init__ = _new_init


def _gpr3d_collect_traces_with_optional_xcorr(self, progress=None):
    # If alignment is disabled, use the previous collector, preserving existing behaviour.
    if not getattr(self, 'xcorr_align', None) or not self.xcorr_align.isChecked():
        return self._gpr3d_old_collect_traces(progress)
    lines = self._selected_lines()
    if not lines:
        raise RuntimeError('No selected lines for migration.')
    lo, hi = sorted([float(self.tmin.value()), float(self.tmax.value())])
    if hi <= 0:
        raise RuntimeError('tmax must be > 0 for Stolt migration.')
    dt = float(self.dt_out.value())
    t_axis = _gpr3d_np.arange(0.0, hi + 0.5 * dt, dt, dtype=float)
    step = max(1, int(self.trace_step.value()))
    max_shift = int(getattr(self, 'max_shift', None).value() if getattr(self, 'max_shift', None) else 4)
    xs=[]; ys=[]; zs=[]; data=[]; names=[]
    shifts=[]
    refs = {}
    total = max(1, len(lines))
    for k, line in enumerate(lines, 1):
        if progress:
            progress(2 + int(18 * k / total), f'Collecting traces {k}/{total}: {getattr(line,"name",line)}')
        arr, t, x, y, z = self._line_array_time_xyze(line)
        if arr is None or arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 8:
            continue
        x = _gpr3d_np.asarray(x, float); y = _gpr3d_np.asarray(y, float); z = _gpr3d_np.asarray(z, float); t = _gpr3d_np.asarray(t, float)
        # Interpolate geometry to trace count before any possible shift.
        ntr = arr.shape[0]
        if x.size != ntr and x.size >= 2:
            x = _gpr3d_np.interp(_gpr3d_np.linspace(0,1,ntr), _gpr3d_np.linspace(0,1,x.size), x)
        if y.size != ntr and y.size >= 2:
            y = _gpr3d_np.interp(_gpr3d_np.linspace(0,1,ntr), _gpr3d_np.linspace(0,1,y.size), y)
        if z.size != ntr and z.size >= 2:
            z = _gpr3d_np.interp(_gpr3d_np.linspace(0,1,ntr), _gpr3d_np.linspace(0,1,z.size), z)
        elif z.size != ntr:
            z = _gpr3d_np.full(ntr, _gpr3d_np.nan)
        # Align only within similar line families to avoid mixing inline/crossline geometry.
        try:
            fam = getattr(line, 'folder', None).parent.name.lower()
        except Exception:
            fam = 'all'
        twin = (t >= lo) & (t <= hi)
        if not _gpr3d_np.any(twin):
            twin = slice(None)
        energy = _gpr3d_np.nanmean(_gpr3d_np.abs(arr[:, twin]), axis=1)
        shift = 0
        if fam in refs and max_shift > 0:
            shift = _gpr3d_best_shift_energy(refs[fam], energy, max_shift)
            arr = _gpr3d_shift_rows_zero(arr, shift)
        else:
            refs[fam] = energy.copy()
        shifts.append((getattr(line, 'name', str(line)), fam, shift))
        for ii in range(0, ntr, step):
            tr = _gpr3d_np.interp(t_axis, t, arr[ii, :], left=0.0, right=0.0).astype(_gpr3d_np.float32)
            tr[t_axis < lo] = 0.0
            xs.append(float(x[ii])); ys.append(float(y[ii])); zs.append(float(z[ii]) if _gpr3d_np.isfinite(z[ii]) else _gpr3d_np.nan)
            data.append(tr); names.append(getattr(line, 'name', str(line)))
    if not data:
        raise RuntimeError('No usable traces in selected time window.')
    self._last_xcorr_shifts = shifts
    if progress:
        nonzero = [(n,s) for n,_,s in shifts if s]
        progress(21, f'XCorr line-shift alignment applied: {len(nonzero)}/{len(shifts)} lines shifted; max shift={max_shift} traces.')
    return _gpr3d_np.asarray(xs), _gpr3d_np.asarray(ys), _gpr3d_np.asarray(zs), _gpr3d_np.vstack(data).astype(_gpr3d_np.float32), t_axis, names


def _gpr3d_patch_collect(cls):
    if not hasattr(cls, '_collect_traces') or getattr(cls, '_gpr3d_xcorr_collect_patched', False):
        return
    cls._gpr3d_old_collect_traces = cls._collect_traces
    cls._collect_traces = _gpr3d_collect_traces_with_optional_xcorr
    cls._gpr3d_xcorr_collect_patched = True


def _gpr3d_safe_clip(a, pct=98.5):
    a = _gpr3d_np.asarray(a, float)
    finite = a[_gpr3d_np.isfinite(a)]
    if finite.size == 0:
        return 1.0
    v = float(_gpr3d_np.nanpercentile(_gpr3d_np.abs(finite), pct))
    return v if _gpr3d_np.isfinite(v) and v > 0 else 1.0


def _gpr3d_plot_depth_slice_mask(self):
    r = self.last; fig = self.canvas.fig; fig.clear()
    depth = r['depth']; iz = int(_gpr3d_np.nanargmin(_gpr3d_np.abs(depth - float(self.depth_slice.value()))))
    cube = r.get('cube_migration_input', r.get('cube_premig', r['cube']))
    t = r['t']; target_t = 2.0 * depth[iz] / max(float(r['velocity']), 1e-9)
    it = int(_gpr3d_np.nanargmin(_gpr3d_np.abs(t - target_t)))
    x = r['x']; y = r['y']; mig = r['migrated']
    axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    im0 = axes[0].imshow(r['valid'].T.astype(float), origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='gray_r')
    axes[0].set_title(f'Binned live-cell mask: {int(r["valid"].sum())}/{r["valid"].size} live')
    axes[0].set_xlabel('Along-survey [m]'); axes[0].set_ylabel('Across-survey [m]')
    v0 = _gpr3d_safe_clip(cube[it])
    im1 = axes[1].imshow(cube[it].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v0, vmax=v0)
    axes[1].set_title(f'Input binned time slice: {t[it]:.1f} ns')
    fig.colorbar(im1, ax=axes[1], shrink=0.75)
    v1 = _gpr3d_safe_clip(mig[iz])
    im2 = axes[2].imshow(mig[iz].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v1, vmax=v1)
    axes[2].set_title(f'Migrated depth slice: z={depth[iz]:.2f} m')
    axes[2].set_xlabel('Along-survey [m]'); axes[2].set_ylabel('Across-survey [m]')
    fig.colorbar(im2, ax=axes[2], shrink=0.75)
    midy = mig.shape[2]//2; v2 = _gpr3d_safe_clip(mig[:,:,midy])
    im3 = axes[3].imshow(mig[:,:,midy], origin='upper', aspect='auto', extent=[x[0], x[-1], depth[-1], depth[0]], cmap='seismic', vmin=-v2, vmax=v2)
    axes[3].set_title(f'Migrated vertical section at across={y[midy]:.2f} m')
    axes[3].set_xlabel('Along-survey [m]'); axes[3].set_ylabel('Depth [m]')
    fig.colorbar(im3, ax=axes[3], shrink=0.75)
    fig.suptitle(f'3-D Stolt Migration QC | v={r["velocity"]:.3f} m/ns | dx={r["dx"]:.3f}, dy={r["dy"]:.3f} m')
    self.canvas.draw()


def _gpr3d_plot_input_diagnostics(self):
    r = self.last; fig = self.canvas.fig; fig.clear()
    t = r['t']; x = r['x']; y = r['y']
    raw = r.get('cube', None)
    premig = r.get('cube_migration_input', r.get('cube_premig', raw))
    it = int(_gpr3d_np.nanargmin(_gpr3d_np.abs(t - max(float(self.tmin.value()), min(float(self.tmax.value()), 0.5*(float(self.tmin.value())+float(self.tmax.value())))))))
    axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    v = _gpr3d_safe_clip(raw[it] if raw is not None else premig[it])
    im0 = axes[0].imshow(raw[it].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v, vmax=v) if raw is not None else None
    axes[0].set_title(f'Input binned cube: {t[it]:.1f} ns')
    if im0: fig.colorbar(im0, ax=axes[0], shrink=0.75)
    im1 = axes[1].imshow(premig[it].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v, vmax=v)
    axes[1].set_title('Pre-migration cube after selected preprocessing')
    fig.colorbar(im1, ax=axes[1], shrink=0.75)
    diff = premig[it] - raw[it] if raw is not None else premig[it]*0
    vd = _gpr3d_safe_clip(diff)
    im2 = axes[2].imshow(diff.T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-vd, vmax=vd)
    axes[2].set_title('Preprocessing difference')
    fig.colorbar(im2, ax=axes[2], shrink=0.75)
    shifts = getattr(self, '_last_xcorr_shifts', [])
    nonzero = [(n, fam, s) for n, fam, s in shifts if s]
    axes[3].axis('off')
    lines = [f'Input traces binned: {r.get("n_input",0):,}', f'Cube shape: {r.get("cube",premig).shape}', f'Valid cells: {int(r["valid"].sum())}/{r["valid"].size}', f'XCorr shifted lines: {len(nonzero)}/{len(shifts)}']
    for row in nonzero[:12]:
        lines.append(f'{row[0]}: {row[2]:+d} traces')
    axes[3].text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', family='monospace')
    fig.suptitle('3-D migration input diagnostics')
    self.canvas.draw()


def _gpr3d_plot_live_taper_qc(self):
    r = self.last; fig = self.canvas.fig; fig.clear()
    taper = r.get('taper_volume', None)
    if taper is None:
        fig.text(0.5, 0.5, 'Live taper is off or no taper volume was stored.', ha='center', va='center')
        self.canvas.draw(); return
    nt, nx, ny = taper.shape
    it, ix, iy = nt//2, nx//2, ny//2
    axes = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233), fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]
    live = (_gpr3d_np.abs(r.get('cube_migration_input', r['cube'])) > 0).astype(float)
    panels = [(live[it].T,'Live mask — time slice'), (live[:,ix,:].T,'Live mask — inline section'), (live[:,:,iy].T,'Live mask — crossline section'), (taper[it].T,'Taper — time slice'), (taper[:,ix,:].T,'Taper — inline section'), (taper[:,:,iy].T,'Taper — crossline section')]
    for ax,(dat,title) in zip(axes,panels):
        im=ax.imshow(dat, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle('Live/dead sample taper QC')
    self.canvas.draw()


def _gpr3d_patch_plot_result(cls):
    old_plot = getattr(cls, 'plot_result', None)
    if old_plot is None or getattr(old_plot, '_gpr3d_viewmode_patched', False):
        return
    def _plot_result_viewmode(self, _old=old_plot):
        if not getattr(self, 'last', None):
            return _old(self)
        mode = self.view_mode.currentText() if getattr(self, 'view_mode', None) else 'Section comparison'
        if mode == 'Depth slice + mask':
            return _gpr3d_plot_depth_slice_mask(self)
        if mode == 'Input diagnostics':
            return _gpr3d_plot_input_diagnostics(self)
        if mode == 'Live taper QC':
            return _gpr3d_plot_live_taper_qc(self)
        return _old(self)
    _plot_result_viewmode._gpr3d_viewmode_patched = True
    cls.plot_result = _plot_result_viewmode


def _gpr3d_patch_default_out(cls):
    old_default = getattr(cls, '_default_out', None)
    if old_default is None or getattr(old_default, '_gpr3d_filename_cleaned', False):
        return
    def _default_out_clean(self, suffix, _old=old_default):
        p = _old(self, suffix)
        try:
            name = p.name.replace('reference_stolt_', 'stolt_').replace('reference_stolt_', 'stolt_').replace('reference', 'migration')
            return p.with_name(name)
        except Exception:
            return p
    _default_out_clean._gpr3d_filename_cleaned = True
    cls._default_out = _default_out_clean


def _gpr3d_export_mat(self):
    if not getattr(self, 'last', None):
        try:
            QMessageBox.warning(self, 'No migration', 'Run migration first.')
        except Exception:
            pass
        return
    if not _GPR3D_SCIPY_IO_OK:
        try:
            QMessageBox.warning(self, 'scipy missing', 'Install scipy first to export MAT files.')
        except Exception:
            pass
        return
    r = self.last
    out = self._default_out('.mat')
    md = {k: v for k, v in r.items() if k in ('cube','cube_premig','cube_migration_input','migrated','taper_volume','valid','count','elev','x','y','world_x','world_y','t','depth','applied_statics_samples') and v is not None}
    md['metadata_json'] = json.dumps({k: str(v) for k, v in r.items() if k not in md}, default=str)
    _gpr3d_savemat(str(out), md, do_compression=True)
    self._status(f'Saved MAT volume: {out}')


def _gpr3d_install_export_mat(cls):
    if not hasattr(cls, 'export_mat'):
        cls.export_mat = _gpr3d_export_mat


def _gpr3d_clean_apply_messages():
    # Clean class flags/prints from old hooks if apply_* below still references old public wording.
    pass

try:
    _cls = _gpr3d_tab_class()
    if _cls is not None:
        _gpr3d_patch_status_method(_cls)
        _gpr3d_install_extra_ui(_cls)
        _gpr3d_patch_collect(_cls)
        _gpr3d_patch_plot_result(_cls)
        _gpr3d_patch_default_out(_cls)
        _gpr3d_install_export_mat(_cls)
        print('3-D migration module loaded: Stolt migration, rotated mean binning, k-filter, elevation statics, live taper, xcorr alignment option, MAT/NPZ/SEG-Y/GIS export.')
except Exception as _e:
    print('3-D migration cleanup/features patch could not be installed:', _e)
# --- end professional migration cleanup and added workflow features ---




# --- compact no-horizontal-scroll 3D migration controls patch ---
def _g3d_compact_build_ui(self):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QDoubleSpinBox,
        QSpinBox, QComboBox, QCheckBox, QTextEdit, QWidget, QScrollArea, QSplitter,
        QSizePolicy
    )

    root = QVBoxLayout(self)
    root.setContentsMargins(6, 6, 6, 6)
    root.setSpacing(4)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    root.addWidget(splitter, 1)

    # ---------- compact left controls ----------
    left = QWidget()
    left_lay = QVBoxLayout(left)
    left_lay.setContentsMargins(6, 6, 6, 6)
    left_lay.setSpacing(5)

    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def _lab(txt):
        l = QLabel(txt)
        l.setMinimumWidth(75)
        l.setMaximumWidth(95)
        return l

    def _add(txt, w):
        w.setMinimumWidth(160)
        w.setMaximumWidth(220)
        form.addRow(_lab(txt), w)

    self.view_mode = QComboBox()
    self.view_mode.addItems(["Section comparison", "Depth slice + mask", "Input diagnostics", "Live taper QC"])
    _add("View", self.view_mode)

    self.xcorr_align = QCheckBox("XCorr line-shift alignment")
    self.xcorr_line_shift = self.xcorr_align
    self.xcorr_line_shift_align = self.xcorr_align
    left_lay.addWidget(self.xcorr_align)

    self.xcorr_max = QSpinBox()
    self.xcorr_max.setRange(0, 50)
    self.xcorr_max.setValue(4)
    self.xcorr_max.setSuffix(" traces")
    self.max_shift = self.xcorr_max
    _add("Max shift", self.xcorr_max)

    self.data_mode = QComboBox(); self.data_mode.addItems(["processed", "raw"])
    self.attribute = QComboBox(); self.attribute.addItems(["Signed amplitude", "Envelope amplitude", "Absolute amplitude"])
    self.attribute.setCurrentText("Signed amplitude")
    self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.02, 0.30); self.velocity.setValue(0.10); self.velocity.setSingleStep(0.005); self.velocity.setSuffix(" m/ns")
    self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(0.0); self.tmin.setSuffix(" ns")
    self.tmax = QDoubleSpinBox(); self.tmax.setRange(0, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(" ns")
    self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20); self.dt_out.setValue(0.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(" ns")
    self.grid_dx = QDoubleSpinBox(); self.grid_dx.setRange(0.02, 2.0); self.grid_dx.setValue(0.25); self.grid_dx.setSingleStep(0.05); self.grid_dx.setSuffix(" m")
    self.grid_dy = QDoubleSpinBox(); self.grid_dy.setRange(0.02, 2.0); self.grid_dy.setValue(0.25); self.grid_dy.setSingleStep(0.05); self.grid_dy.setSuffix(" m")
    self.max_nx = QSpinBox(); self.max_nx.setRange(20, 512); self.max_nx.setValue(240)
    self.max_ny = QSpinBox(); self.max_ny.setRange(20, 512); self.max_ny.setValue(140)
    self.trace_step = QSpinBox(); self.trace_step.setRange(1, 100); self.trace_step.setValue(5)
    self.max_lines = QSpinBox(); self.max_lines.setRange(1, 10000); self.max_lines.setValue(500)
    self.k_filter = QComboBox(); self.k_filter.addItems(["Off", "Light k-filter", "Medium k-filter", "Strong k-filter"])
    self.k_filter.setCurrentText("Light k-filter")
    self.pad_t = QDoubleSpinBox(); self.pad_t.setRange(1.0, 4.0); self.pad_t.setValue(1.5); self.pad_t.setSingleStep(0.25)
    self.pad_xy = QSpinBox(); self.pad_xy.setRange(0, 100); self.pad_xy.setValue(15); self.pad_xy.setSuffix(" cells")
    self.taper_w = QSpinBox(); self.taper_w.setRange(0, 50); self.taper_w.setValue(5); self.taper_w.setSuffix(" cells")
    self.depth_slice = QDoubleSpinBox(); self.depth_slice.setRange(0.0, 20.0); self.depth_slice.setValue(0.60); self.depth_slice.setSingleStep(0.05); self.depth_slice.setSuffix(" m")
    self.clip_pct = QDoubleSpinBox(); self.clip_pct.setRange(80.0, 100.0); self.clip_pct.setValue(98.5); self.clip_pct.setSingleStep(0.1); self.clip_pct.setSuffix(" %")

    for txt, w in [
        ("Data", self.data_mode),
        ("Attribute", self.attribute),
        ("Velocity", self.velocity),
        ("tmin", self.tmin),
        ("tmax", self.tmax),
        ("dt", self.dt_out),
        ("Grid dx", self.grid_dx),
        ("Grid dy", self.grid_dy),
        ("Max nx", self.max_nx),
        ("Max ny", self.max_ny),
        ("Trace step", self.trace_step),
        ("Max lines", self.max_lines),
        ("k-filter", self.k_filter),
        ("Pad t", self.pad_t),
        ("Pad xy", self.pad_xy),
        ("Taper", self.taper_w),
        ("Bird's-eye depth slice", self.depth_slice),
        ("Clip", self.clip_pct),
    ]:
        _add(txt, w)

    left_lay.addLayout(form)

    self.live_taper = QCheckBox("Live/dead mask taper"); self.live_taper.setChecked(True)
    self.blank_dead = QCheckBox("Re-blank dead cells"); self.blank_dead.setChecked(True)
    self.blank_topo = QCheckBox("Topographic blanking"); self.blank_topo.setChecked(True)
    self.jacobian = QCheckBox("Stolt Jacobian"); self.jacobian.setChecked(True)

    for w in (self.live_taper, self.blank_dead, self.blank_topo, self.jacobian):
        left_lay.addWidget(w)

    self.btn_run = QPushButton("Build volume + run 3-D Stolt migration")
    self.btn_validate = QPushButton("Run synthetic validation")
    self.btn_png = QPushButton("Export PNG")
    self.btn_npz = QPushButton("Export NPZ volume")
    self.btn_segy = QPushButton("Export SEG-Y" + ("" if globals().get("SEGYIO_OK", False) else " (needs segyio)"))
    self.btn_gis = QPushButton("Export GeoPackage slices" + ("" if globals().get("RASTERIO_OK", False) else " (needs rasterio)"))

    for b in (self.btn_run, self.btn_validate, self.btn_png, self.btn_npz, self.btn_segy, self.btn_gis):
        b.setMinimumHeight(26)
        left_lay.addWidget(b)

    left_lay.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(left)
    scroll.setMinimumWidth(300)
    scroll.setMaximumWidth(340)
    splitter.addWidget(scroll)

    # ---------- right plot/log ----------
    right = QWidget()
    right_lay = QVBoxLayout(right)
    right_lay.setContentsMargins(4, 0, 4, 4)
    right_lay.setSpacing(4)

    self.canvas = _PVCanvas(13, 7)
    self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    self.canvas.setMinimumHeight(260)
    right_lay.addWidget(self.canvas, 1)

    self.log = QTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumHeight(60)
    self.log.setPlaceholderText("3-D migration status/output log")
    right_lay.addWidget(self.log, 0)

    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([320, 1200])

    self.btn_run.clicked.connect(self.run_pipeline)
    self.btn_validate.clicked.connect(self.run_validation)
    self.btn_png.clicked.connect(self.export_png)
    self.btn_npz.clicked.connect(self.export_npz)
    self.btn_segy.clicked.connect(self.export_segy)
    self.btn_gis.clicked.connect(self.export_gpkg)

for _name in ("GPR3DMigrationTab", "Professor3DMigrationTab"):
    _cls = globals().get(_name)
    if _cls is not None:
        _cls._build_ui = _g3d_compact_build_ui
# --- end compact no-horizontal-scroll 3D migration controls patch ---




# --- fit 3-D migration side-panel buttons patch ---
def _g3d_fit_side_panel_widgets(obj):
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QScrollArea, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLabel

        # Make the left control panel wide enough, but still leave most space for the plot.
        for sa in obj.findChildren(QScrollArea):
            try:
                sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                sa.setMinimumWidth(390)
                sa.setMaximumWidth(430)
                w = sa.widget()
                if w is not None:
                    w.setMinimumWidth(360)
                    w.setMaximumWidth(405)
            except Exception:
                pass

        # Shorter button labels so they do not get clipped.
        replacements = {
            "Build volume + run 3-D Stolt migration": "Run 3-D migration",
            "Build volume + run 3-D migration": "Run 3-D migration",
            "Build volume": "Run 3-D migration",
            "Run synthetic validation": "Validate synthetic",
            "Export NPZ volume": "Export NPZ",
            "Export SEG-Y (needs segyio)": "Export SEG-Y",
            "Export GeoPackage slices (needs rasterio)": "Export GPKG",
        }

        for b in obj.findChildren(QPushButton):
            try:
                txt = b.text()
                if txt in replacements:
                    b.setText(replacements[txt])
                b.setMinimumWidth(0)
                b.setMaximumWidth(360)
                b.setMinimumHeight(26)
            except Exception:
                pass

        # Keep input boxes compact.
        for w in obj.findChildren(QComboBox) + obj.findChildren(QSpinBox) + obj.findChildren(QDoubleSpinBox):
            try:
                w.setMinimumWidth(150)
                w.setMaximumWidth(210)
            except Exception:
                pass

        # Keep labels compact.
        for lab in obj.findChildren(QLabel):
            try:
                if lab.text().strip() in {
                    "Data", "Attribute", "Velocity", "tmin", "tmax", "dt",
                    "Grid dx", "Grid dy", "Max nx", "Max ny", "Trace step",
                    "Max lines", "k-filter", "Pad t", "Pad xy", "Taper",
                    "Depth slice", "Clip", "View", "Max shift"
                }:
                    lab.setMinimumWidth(70)
                    lab.setMaximumWidth(90)
            except Exception:
                pass
    except Exception as e:
        print("3-D side-panel fit failed:", e)

def _g3d_patch_sidepanel_fit_class(cls):
    if cls is None or getattr(cls, "_g3d_sidepanel_fit_patched", False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        _g3d_fit_side_panel_widgets(self)
    cls.__init__ = new_init
    cls._g3d_sidepanel_fit_patched = True

for _name in ("GPR3DMigrationTab", "Professor3DMigrationTab"):
    _cls = globals().get(_name)
    if _cls is not None:
        _g3d_patch_sidepanel_fit_class(_cls)
# --- end fit 3-D migration side-panel buttons patch ---




# --- remove stale overlapping 3-D migration top controls patch ---
def _g3d_hide_stale_top_controls(obj):
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QLabel, QComboBox, QCheckBox, QSpinBox, QPushButton, QScrollArea

        def _hide(w):
            try:
                w.hide()
                w.setMaximumWidth(0)
                w.setMaximumHeight(0)
                w.setVisible(False)
            except Exception:
                pass

        def _keep_last(widgets):
            widgets = [w for w in widgets if w is not None]
            if len(widgets) <= 1:
                return
            for w in widgets[:-1]:
                _hide(w)

        # Duplicate View label/combo from the old horizontal row.
        view_labels = []
        maxshift_labels = []
        for lab in obj.findChildren(QLabel):
            try:
                t = lab.text().strip().lower()
                if t == "view":
                    view_labels.append(lab)
                elif t == "max shift":
                    maxshift_labels.append(lab)
            except Exception:
                pass
        _keep_last(view_labels)
        _keep_last(maxshift_labels)

        view_combos = []
        for cb in obj.findChildren(QComboBox):
            try:
                items = [cb.itemText(i).lower() for i in range(cb.count())]
                if "section comparison" in items:
                    view_combos.append(cb)
            except Exception:
                pass
        _keep_last(view_combos)

        xcorr_boxes = []
        for chk in obj.findChildren(QCheckBox):
            try:
                if "xcorr" in chk.text().lower() or "line-shift" in chk.text().lower():
                    xcorr_boxes.append(chk)
            except Exception:
                pass
        _keep_last(xcorr_boxes)

        shift_spins = []
        for sp in obj.findChildren(QSpinBox):
            try:
                if "trace" in sp.suffix().lower():
                    shift_spins.append(sp)
            except Exception:
                pass
        _keep_last(shift_spins)

        # Hide any old clipped button that sits in the stale top row.
        for b in obj.findChildren(QPushButton):
            try:
                y = b.geometry().y()
                txt = b.text().strip().lower()
                if y < 45 and (txt.startswith("export") or txt.startswith("run") or txt.startswith("build")):
                    _hide(b)
            except Exception:
                pass

        # Force the side panel to avoid horizontal scrolling.
        for sa in obj.findChildren(QScrollArea):
            try:
                sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                sa.setMinimumWidth(390)
                sa.setMaximumWidth(440)
                w = sa.widget()
                if w is not None:
                    w.setMinimumWidth(360)
                    w.setMaximumWidth(410)
            except Exception:
                pass

    except Exception as e:
        print("Could not hide stale 3-D controls:", e)

def _g3d_patch_hide_stale_controls_class(cls):
    if cls is None or getattr(cls, "_g3d_hide_stale_controls_patched", False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: _g3d_hide_stale_top_controls(self))
            QTimer.singleShot(300, lambda: _g3d_hide_stale_top_controls(self))
        except Exception:
            _g3d_hide_stale_top_controls(self)
    cls.__init__ = new_init
    cls._g3d_hide_stale_controls_patched = True

for _name in ("GPR3DMigrationTab", "Professor3DMigrationTab"):
    _cls = globals().get(_name)
    if _cls is not None:
        _g3d_patch_hide_stale_controls_class(_cls)
# --- end remove stale overlapping 3-D migration top controls patch ---




# --- final clean compact 3-D migration side panel patch ---
def _g3d_final_clean_build_ui(self):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QVBoxLayout, QFormLayout, QLabel, QPushButton, QDoubleSpinBox,
        QSpinBox, QComboBox, QCheckBox, QTextEdit, QWidget, QScrollArea,
        QSplitter, QSizePolicy
    )

    root = QVBoxLayout(self)
    root.setContentsMargins(6, 6, 6, 6)
    root.setSpacing(4)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    root.addWidget(splitter, 1)

    # LEFT: one clean compact form only
    left = QWidget()
    left_lay = QVBoxLayout(left)
    left_lay.setContentsMargins(6, 6, 6, 6)
    left_lay.setSpacing(5)

    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

    def lab(txt):
        w = QLabel(txt)
        w.setMinimumWidth(72)
        w.setMaximumWidth(86)
        return w

    def compact(w, width=205):
        try:
            w.setMinimumWidth(175)
            w.setMaximumWidth(width)
        except Exception:
            pass
        return w

    def add(txt, widget):
        form.addRow(lab(txt), compact(widget))

    self.view_mode = QComboBox()
    self.view_mode.addItems(["Section comparison", "Depth slice + mask", "Input diagnostics", "Live taper QC"])
    add("View", self.view_mode)

    self.xcorr_align = QCheckBox("Enable")
    self.xcorr_line_shift = self.xcorr_align
    self.xcorr_line_shift_align = self.xcorr_align
    add("XCorr shift", self.xcorr_align)

    self.xcorr_max = QSpinBox()
    self.xcorr_max.setRange(0, 50)
    self.xcorr_max.setValue(4)
    self.xcorr_max.setSuffix(" traces")
    self.max_shift = self.xcorr_max
    add("Max shift", self.xcorr_max)

    self.data_mode = QComboBox(); self.data_mode.addItems(["processed", "raw"])
    self.attribute = QComboBox(); self.attribute.addItems(["Signed amplitude", "Envelope amplitude", "Absolute amplitude"])
    self.attribute.setCurrentText("Signed amplitude")
    self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.02, 0.30); self.velocity.setValue(0.10); self.velocity.setSingleStep(0.005); self.velocity.setSuffix(" m/ns")
    self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(0.0); self.tmin.setSuffix(" ns")
    self.tmax = QDoubleSpinBox(); self.tmax.setRange(0, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(" ns")
    self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20); self.dt_out.setValue(0.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(" ns")
    self.grid_dx = QDoubleSpinBox(); self.grid_dx.setRange(0.02, 2.0); self.grid_dx.setValue(0.25); self.grid_dx.setSingleStep(0.05); self.grid_dx.setSuffix(" m")
    self.grid_dy = QDoubleSpinBox(); self.grid_dy.setRange(0.02, 2.0); self.grid_dy.setValue(0.25); self.grid_dy.setSingleStep(0.05); self.grid_dy.setSuffix(" m")
    self.max_nx = QSpinBox(); self.max_nx.setRange(20, 512); self.max_nx.setValue(240)
    self.max_ny = QSpinBox(); self.max_ny.setRange(20, 512); self.max_ny.setValue(140)
    self.trace_step = QSpinBox(); self.trace_step.setRange(1, 100); self.trace_step.setValue(5)
    self.max_lines = QSpinBox(); self.max_lines.setRange(1, 10000); self.max_lines.setValue(500)
    self.k_filter = QComboBox(); self.k_filter.addItems(["Off", "Light k-filter", "Medium k-filter", "Strong k-filter"])
    self.k_filter.setCurrentText("Light k-filter")
    self.pad_t = QDoubleSpinBox(); self.pad_t.setRange(1.0, 4.0); self.pad_t.setValue(1.5); self.pad_t.setSingleStep(0.25)
    self.pad_xy = QSpinBox(); self.pad_xy.setRange(0, 100); self.pad_xy.setValue(15); self.pad_xy.setSuffix(" cells")
    self.taper_w = QSpinBox(); self.taper_w.setRange(0, 50); self.taper_w.setValue(5); self.taper_w.setSuffix(" cells")
    self.depth_slice = QDoubleSpinBox(); self.depth_slice.setRange(0.0, 20.0); self.depth_slice.setValue(0.60); self.depth_slice.setSingleStep(0.05); self.depth_slice.setSuffix(" m")
    self.clip_pct = QDoubleSpinBox(); self.clip_pct.setRange(80.0, 100.0); self.clip_pct.setValue(98.5); self.clip_pct.setSingleStep(0.1); self.clip_pct.setSuffix(" %")

    for txt, w in [
        ("Data", self.data_mode), ("Attribute", self.attribute), ("Velocity", self.velocity),
        ("tmin", self.tmin), ("tmax", self.tmax), ("dt", self.dt_out),
        ("Grid dx", self.grid_dx), ("Grid dy", self.grid_dy),
        ("Max nx", self.max_nx), ("Max ny", self.max_ny),
        ("Trace step", self.trace_step), ("Max lines", self.max_lines),
        ("k-filter", self.k_filter), ("Pad t", self.pad_t), ("Pad xy", self.pad_xy),
        ("Taper", self.taper_w), ("Bird's-eye depth slice", self.depth_slice), ("Clip", self.clip_pct),
    ]:
        add(txt, w)

    left_lay.addLayout(form)

    self.live_taper = QCheckBox("Live/dead mask taper"); self.live_taper.setChecked(True)
    self.blank_dead = QCheckBox("Re-blank dead cells"); self.blank_dead.setChecked(True)
    self.blank_topo = QCheckBox("Topographic blanking"); self.blank_topo.setChecked(True)
    self.jacobian = QCheckBox("Stolt Jacobian"); self.jacobian.setChecked(True)

    for w in (self.live_taper, self.blank_dead, self.blank_topo, self.jacobian):
        left_lay.addWidget(w)

    self.btn_run = QPushButton("Run 3-D migration")
    self.btn_validate = QPushButton("Validate synthetic")
    self.btn_png = QPushButton("Export PNG")
    self.btn_npz = QPushButton("Export NPZ")
    self.btn_segy = QPushButton("Export SEG-Y")
    self.btn_gis = QPushButton("Export GPKG")

    for b in (self.btn_run, self.btn_validate, self.btn_png, self.btn_npz, self.btn_segy, self.btn_gis):
        b.setMinimumHeight(26)
        b.setMaximumWidth(285)
        left_lay.addWidget(b)

    left_lay.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(left)
    scroll.setMinimumWidth(315)
    scroll.setMaximumWidth(350)
    splitter.addWidget(scroll)

    # RIGHT: plot + compact log
    right = QWidget()
    right_lay = QVBoxLayout(right)
    right_lay.setContentsMargins(4, 0, 4, 4)
    right_lay.setSpacing(4)

    self.canvas = _PVCanvas(13, 7)
    self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    self.canvas.setMinimumHeight(260)
    right_lay.addWidget(self.canvas, 1)

    self.log = QTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumHeight(60)
    self.log.setPlaceholderText("3-D migration status/output log")
    right_lay.addWidget(self.log, 0)

    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([330, 1200])

    self.btn_run.clicked.connect(self.run_pipeline)
    self.btn_validate.clicked.connect(self.run_validation)
    self.btn_png.clicked.connect(self.export_png)
    self.btn_npz.clicked.connect(self.export_npz)
    self.btn_segy.clicked.connect(self.export_segy)
    self.btn_gis.clicked.connect(self.export_gpkg)

for _name in ("GPR3DMigrationTab", "Professor3DMigrationTab"):
    _cls = globals().get(_name)
    if _cls is not None:
        _cls._build_ui = _g3d_final_clean_build_ui
# --- end final clean compact 3-D migration side panel patch ---


# --- final disable stale 3-D migration extra row patch ---
# The migration tab now has the View/XCorr/Max-shift controls built directly
# into the compact form layout. Older patches still wrapped __init__ and tried
# to add the same controls again as a horizontal row. That stale row caused the
# visible overlap. Keep this override last so the old wrapper becomes harmless.
def _gpr3d_add_extra_controls(self):
    try:
        self._gpr3d_extra_controls_added = True
    except Exception:
        pass
    return

def _gpr3d_install_extra_ui(cls):
    # No-op: extra controls are part of the final compact _build_ui.
    return

def _g3d_add_mat_button_clean(self):
    try:
        if getattr(self, '_g3d_clean_mat_button_added', False):
            return
        if not hasattr(self, 'export_mat'):
            return
        from PyQt6.QtWidgets import QPushButton, QScrollArea
        btn = QPushButton('Export MAT')
        btn.setMinimumHeight(26)
        btn.setMaximumWidth(285)
        btn.clicked.connect(self.export_mat)
        inserted = False
        for sa in self.findChildren(QScrollArea):
            w = sa.widget()
            lay = w.layout() if w is not None else None
            if lay is not None:
                # Insert near the other export buttons, before the final stretch.
                idx = max(0, lay.count() - 1)
                lay.insertWidget(idx, btn)
                inserted = True
                break
        if inserted:
            self.btn_mat = btn
            self._g3d_clean_mat_button_added = True
    except Exception as e:
        print('Could not add clean MAT export button:', e)

def _g3d_patch_clean_mat_class(cls):
    if cls is None or getattr(cls, '_g3d_clean_mat_init_patched', False):
        return
    old_init = cls.__init__
    def new_init(self, *a, _old=old_init, **kw):
        _old(self, *a, **kw)
        _g3d_add_mat_button_clean(self)
    cls.__init__ = new_init
    cls._g3d_clean_mat_init_patched = True

for _name in ('GPR3DMigrationTab', 'Professor3DMigrationTab'):
    _cls = globals().get(_name)
    if _cls is not None:
        _g3d_patch_clean_mat_class(_cls)
# --- end final disable stale 3-D migration extra row patch ---


# --- selectable 3-D migration section controls patch ---
# Adds user-selectable inline/crossline/depth controls for the migrated 3-D cube.
# The controls choose slices through the regular binned/migrated volume, not raw file folders.

def _g3d_safe_clip_v2(arr, pct=98.5):
    import numpy as _np
    a = _np.asarray(arr, dtype=float)
    a = a[_np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = float(_np.nanpercentile(_np.abs(a), pct))
    return v if _np.isfinite(v) and v > 0 else 1.0


def _g3d_add_section_selector_controls(self):
    if getattr(self, '_g3d_section_selector_controls_added', False):
        return
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QGroupBox, QFormLayout, QWidget, QHBoxLayout, QSlider, QSpinBox, QLabel

    box = QGroupBox('Display slice selectors')
    form = QFormLayout(box)
    form.setContentsMargins(8, 8, 8, 8)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)

    def make_slider_spin():
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        slider.setEnabled(False)
        spin = QSpinBox()
        spin.setRange(0, 0)
        spin.setValue(0)
        spin.setFixedWidth(130)
        spin.setSpecialValueText('run migration first')
        spin.setSuffix('')
        spin.setEnabled(False)
        lay.addWidget(slider, 1)
        lay.addWidget(spin, 0)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return row, slider, spin

    self.inline_section_row, self.inline_section_slider, self.inline_section_spin = make_slider_spin()
    self.crossline_section_row, self.crossline_section_slider, self.crossline_section_spin = make_slider_spin()

    self.depth_index_row = QWidget()
    dlay = QHBoxLayout(self.depth_index_row)
    dlay.setContentsMargins(0, 0, 0, 0)
    dlay.setSpacing(5)
    self.depth_index_slider = QSlider(Qt.Orientation.Horizontal)
    self.depth_index_slider.setRange(0, 0)
    self.depth_index_slider.setEnabled(False)
    self.depth_index_label = QLabel('run migration first')
    self.depth_index_label.setMinimumWidth(72)
    dlay.addWidget(self.depth_index_slider, 1)
    dlay.addWidget(self.depth_index_label, 0)

    form.addRow(QLabel('Along-survey section position'), self.inline_section_row)
    form.addRow(QLabel('Across-survey section position'), self.crossline_section_row)
    form.addRow(QLabel("Bird's-eye depth slice"), self.depth_index_row)

    def request_replot(*_):
        if getattr(self, '_g3d_selector_syncing', False):
            return
        if getattr(self, 'last', None):
            self.plot_result()

    self.inline_section_spin.valueChanged.connect(request_replot)
    self.crossline_section_spin.valueChanged.connect(request_replot)
    self.depth_slice.valueChanged.connect(request_replot)

    def depth_slider_changed(val):
        if getattr(self, '_g3d_selector_syncing', False):
            return
        r = getattr(self, 'last', None)
        if not r:
            return
        import numpy as _np
        depth = _np.asarray(r.get('depth', []), dtype=float)
        if depth.size == 0:
            return
        iz = max(0, min(int(val), depth.size - 1))
        self._g3d_selector_syncing = True
        try:
            self.depth_slice.setValue(float(depth[iz]))
            self.depth_index_label.setText(f'z={float(depth[iz]):.2f} m')
        finally:
            self._g3d_selector_syncing = False
        request_replot()

    self.depth_index_slider.valueChanged.connect(depth_slider_changed)

    left = None
    try:
        left = self.btn_run.parentWidget()
    except Exception:
        left = None
    if left is not None and left.layout() is not None:
        lay = left.layout()
        idx = -1
        for attr in ('live_taper', 'btn_run'):
            w = getattr(self, attr, None)
            if w is not None:
                idx = lay.indexOf(w)
                if idx >= 0:
                    break
        if idx < 0:
            idx = max(0, lay.count() - 1)
        lay.insertWidget(idx, box)
    self._g3d_section_selector_controls_added = True


def _g3d_update_section_selector_ranges(self, r):
    import numpy as _np
    if not getattr(self, '_g3d_section_selector_controls_added', False):
        _g3d_add_section_selector_controls(self)
    mig = _np.asarray(r['migrated'])
    nz, nx, ny = mig.shape
    depth = _np.asarray(r.get('depth', []), dtype=float)
    shape_key = (int(nx), int(ny), int(depth.size), id(r.get('migrated')))
    first_for_shape = getattr(self, '_g3d_selector_shape_key', None) != shape_key
    self._g3d_selector_shape_key = shape_key

    def set_pair(spin, slider, n, default):
        n = max(1, int(n))
        old = int(spin.value()) if spin.maximum() >= 1 else default
        val = default if first_for_shape else max(1, min(old, n))
        spin.blockSignals(True); slider.blockSignals(True)
        try:
            spin.setEnabled(True)
            slider.setEnabled(True)
            spin.setSpecialValueText('')
            spin.setRange(1, n)
            spin.setSuffix(f' / {n}')
            spin.setValue(val)
            slider.setRange(1, n)
            slider.setValue(val)
        finally:
            spin.blockSignals(False); slider.blockSignals(False)

    self._g3d_selector_syncing = True
    try:
        set_pair(self.inline_section_spin, self.inline_section_slider, nx, nx // 2 + 1)
        set_pair(self.crossline_section_spin, self.crossline_section_slider, ny, ny // 2 + 1)
        if depth.size:
            old_z = float(self.depth_slice.value())
            iz = int(_np.nanargmin(_np.abs(depth - old_z)))
            if first_for_shape:
                iz = int(_np.nanargmin(_np.abs(depth - old_z)))
            self.depth_index_slider.setEnabled(True)
            self.depth_index_slider.setRange(0, max(0, depth.size - 1))
            self.depth_index_slider.setValue(max(0, min(iz, depth.size - 1)))
            self.depth_index_label.setText(f'z={float(depth[max(0, min(iz, depth.size - 1))]):.2f} m')
            self.depth_slice.setRange(float(_np.nanmin(depth)), float(_np.nanmax(depth)))
    finally:
        self._g3d_selector_syncing = False


def _g3d_selected_cube_indices(self, r):
    import numpy as _np
    _g3d_update_section_selector_ranges(self, r)
    mig = _np.asarray(r['migrated'])
    nz, nx, ny = mig.shape
    ix = int(getattr(self, 'inline_section_spin').value()) - 1
    iy = int(getattr(self, 'crossline_section_spin').value()) - 1
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    depth = _np.asarray(r.get('depth', []), dtype=float)
    if depth.size:
        iz = int(_np.nanargmin(_np.abs(depth - float(self.depth_slice.value()))))
    else:
        iz = 0
    iz = max(0, min(iz, nz - 1))
    return ix, iy, iz


def _g3d_plot_section_comparison_selected(self):
    import numpy as _np
    r = self.last
    fig = self.canvas.fig
    fig.clear()
    ix, iy, _ = _g3d_selected_cube_indices(self, r)
    x = _np.asarray(r['x'], dtype=float)
    y = _np.asarray(r['y'], dtype=float)
    t = _np.asarray(r['t'], dtype=float)
    depth = _np.asarray(r['depth'], dtype=float)
    mig = _np.asarray(r['migrated'])
    cube_in = r.get('cube_migration_input', r.get('cube_premig', r.get('cube_display', r.get('cube'))))
    cube_in = _np.asarray(cube_in)

    axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    panels = [
        (axes[0], cube_in[:, ix, :], y, t, 'Across-survey position [m]', 'TWT [ns]', f'Vertical section at along-survey X={ix+1}/{mig.shape[1]}  x={x[ix]:.2f} m — input'),
        (axes[1], mig[:, ix, :],     y, depth, 'Across-survey position [m]', 'Depth [m]', f'Vertical section at along-survey X={ix+1}/{mig.shape[1]}  x={x[ix]:.2f} m — migrated'),
        (axes[2], cube_in[:, :, iy], x, t, 'Along-survey position [m]', 'TWT [ns]', f'Vertical section at across-survey Y={iy+1}/{mig.shape[2]}  y={y[iy]:.2f} m — input'),
        (axes[3], mig[:, :, iy],     x, depth, 'Along-survey position [m]', 'Depth [m]', f'Vertical section at across-survey Y={iy+1}/{mig.shape[2]}  y={y[iy]:.2f} m — migrated'),
    ]
    for ax, data2d, lateral, vertical, xlabel, ylabel, title in panels:
        v = _g3d_safe_clip_v2(data2d, float(self.clip_pct.value()) if hasattr(self, 'clip_pct') else 98.5)
        im = ax.imshow(data2d, origin='upper', aspect='auto', cmap='seismic',
                       extent=[float(lateral[0]), float(lateral[-1]), float(vertical[-1]), float(vertical[0])],
                       vmin=-v, vmax=v)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, shrink=0.72, label='Amplitude')
    fig.suptitle('3-D Stolt migration — selectable vertical sections | '
                 f'v={float(r.get("velocity", 0)):.3f} m/ns | dx={float(r.get("dx", 0)):.3f}, dy={float(r.get("dy", 0)):.3f} m', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    self.canvas.draw()
    try:
        self._status(f'Showing selected centre/default grid sections: X={ix+1}/{mig.shape[1]} at x={x[ix]:.2f} m; Y={iy+1}/{mig.shape[2]} at y={y[iy]:.2f} m.')
    except Exception:
        pass


def _g3d_plot_depth_slice_mask_selected(self):
    import numpy as _np
    r = self.last
    fig = self.canvas.fig
    fig.clear()
    ix, iy, iz = _g3d_selected_cube_indices(self, r)
    x = _np.asarray(r['x'], dtype=float)
    y = _np.asarray(r['y'], dtype=float)
    t = _np.asarray(r['t'], dtype=float)
    depth = _np.asarray(r['depth'], dtype=float)
    mig = _np.asarray(r['migrated'])
    cube = _np.asarray(r.get('cube_migration_input', r.get('cube_premig', r.get('cube'))))
    target_t = 2.0 * float(depth[iz]) / max(float(r.get('velocity', 0.10)), 1e-9)
    it = int(_np.nanargmin(_np.abs(t - target_t)))

    axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    valid = _np.asarray(r['valid']).T.astype(float)
    im0 = axes[0].imshow(valid, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='gray_r')
    axes[0].axvline(float(x[ix])); axes[0].axhline(float(y[iy]))
    axes[0].set_title(f'Live-cell mask: {int(_np.asarray(r["valid"]).sum())}/{_np.asarray(r["valid"]).size} live')
    axes[0].set_xlabel('Along-survey [m]'); axes[0].set_ylabel('Across-survey [m]')

    v0 = _g3d_safe_clip_v2(cube[it], float(self.clip_pct.value()) if hasattr(self, 'clip_pct') else 98.5)
    im1 = axes[1].imshow(cube[it].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v0, vmax=v0)
    axes[1].axvline(float(x[ix])); axes[1].axhline(float(y[iy]))
    axes[1].set_title(f'Input time slice near depth: {t[it]:.1f} ns')
    axes[1].set_xlabel('Along-survey [m]'); axes[1].set_ylabel('Across-survey [m]')
    fig.colorbar(im1, ax=axes[1], shrink=0.72)

    v1 = _g3d_safe_clip_v2(mig[iz], float(self.clip_pct.value()) if hasattr(self, 'clip_pct') else 98.5)
    im2 = axes[2].imshow(mig[iz].T, origin='lower', aspect='auto', extent=[x[0], x[-1], y[0], y[-1]], cmap='seismic', vmin=-v1, vmax=v1)
    axes[2].axvline(float(x[ix])); axes[2].axhline(float(y[iy]))
    axes[2].set_title(f'Migrated bird\'s-eye depth slice: z={depth[iz]:.2f} m')
    axes[2].set_xlabel('Along-survey [m]'); axes[2].set_ylabel('Across-survey [m]')
    fig.colorbar(im2, ax=axes[2], shrink=0.72)

    sec = mig[:, :, iy]
    v2 = _g3d_safe_clip_v2(sec, float(self.clip_pct.value()) if hasattr(self, 'clip_pct') else 98.5)
    im3 = axes[3].imshow(sec, origin='upper', aspect='auto', extent=[x[0], x[-1], depth[-1], depth[0]], cmap='seismic', vmin=-v2, vmax=v2)
    axes[3].axvline(float(x[ix])); axes[3].axhline(float(depth[iz]))
    axes[3].set_title(f'Selected vertical section at across-survey Y={iy+1}/{mig.shape[2]}  y={y[iy]:.2f} m')
    axes[3].set_xlabel('Along-survey [m]'); axes[3].set_ylabel('Depth [m]')
    fig.colorbar(im3, ax=axes[3], shrink=0.72)

    fig.suptitle('3-D migration depth-slice + selected section | '
                 f'X={ix+1}/{mig.shape[1]}, Y={iy+1}/{mig.shape[2]}, z={depth[iz]:.2f} m', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    self.canvas.draw()


def _g3d_patch_section_selector_class(cls):
    if getattr(cls, '_g3d_section_selector_patch_applied', False):
        return
    old_build = cls._build_ui
    old_plot = cls.plot_result

    def build_with_section_selectors(self, _old=old_build):
        _old(self)
        _g3d_add_section_selector_controls(self)

    def plot_with_section_selectors(self, _old=old_plot):
        if not getattr(self, 'last', None):
            return _old(self)
        mode = self.view_mode.currentText() if getattr(self, 'view_mode', None) else 'Section comparison'
        if mode == 'Section comparison':
            return _g3d_plot_section_comparison_selected(self)
        if mode == 'Depth slice + mask':
            return _g3d_plot_depth_slice_mask_selected(self)
        if mode == 'Input diagnostics' and globals().get('_gpr3d_plot_input_diagnostics'):
            return globals()['_gpr3d_plot_input_diagnostics'](self)
        if mode == 'Live taper QC' and globals().get('_gpr3d_plot_live_taper_qc'):
            return globals()['_gpr3d_plot_live_taper_qc'](self)
        return _old(self)

    cls._build_ui = build_with_section_selectors
    cls.plot_result = plot_with_section_selectors
    cls._g3d_section_selector_patch_applied = True


for _g3d_cls_name in ('GPR3DMigrationTab', 'Professor3DMigrationTab'):
    _g3d_cls = globals().get(_g3d_cls_name)
    if _g3d_cls is not None:
        _g3d_patch_section_selector_class(_g3d_cls)
# --- end selectable 3-D migration section controls patch ---


# --- centre/default migration view patch ---
def _g3d_update_section_selector_ranges(self, r):
    import numpy as _np
    if not getattr(self, '_g3d_section_selector_controls_added', False):
        _g3d_add_section_selector_controls(self)

    mig = _np.asarray(r['migrated'])
    nz, nx, ny = mig.shape
    depth = _np.asarray(r.get('depth', []), dtype=float)

    shape_key = (int(nx), int(ny), int(depth.size), id(r.get('migrated')))
    first_for_shape = getattr(self, '_g3d_selector_shape_key', None) != shape_key
    self._g3d_selector_shape_key = shape_key

    default_ix = max(1, nx // 2 + 1)
    default_iy = max(1, ny // 2 + 1)

    def set_pair(spin, slider, n, default):
        n = max(1, int(n))
        old = int(spin.value()) if spin.maximum() >= 1 else default
        val = default if first_for_shape else max(1, min(old, n))
        spin.blockSignals(True); slider.blockSignals(True)
        try:
            spin.setEnabled(True)
            slider.setEnabled(True)
            try:
                spin.setSpecialValueText('')
            except Exception:
                pass
            spin.setRange(1, n)
            spin.setSuffix(f' / {n}')
            spin.setValue(max(1, min(val, n)))
            slider.setRange(1, n)
            slider.setValue(max(1, min(val, n)))
        finally:
            spin.blockSignals(False); slider.blockSignals(False)

    self._g3d_selector_syncing = True
    try:
        set_pair(self.inline_section_spin, self.inline_section_slider, nx, default_ix)
        set_pair(self.crossline_section_spin, self.crossline_section_slider, ny, default_iy)

        if depth.size:
            z_wanted = float(self.depth_slice.value())
            if first_for_shape:
                z_wanted = 6.0 if float(_np.nanmin(depth)) <= 6.0 <= float(_np.nanmax(depth)) else float(depth[len(depth)//2])
            iz = int(_np.nanargmin(_np.abs(depth - z_wanted)))
            iz = max(0, min(iz, depth.size - 1))
            self.depth_index_slider.setEnabled(True)
            self.depth_index_slider.setRange(0, max(0, depth.size - 1))
            self.depth_index_slider.setValue(iz)
            self.depth_index_label.setText(f'z={float(depth[iz]):.2f} m')
            self.depth_slice.setRange(float(_np.nanmin(depth)), float(_np.nanmax(depth)))
            if first_for_shape:
                self.depth_slice.setValue(float(depth[iz]))
    finally:
        self._g3d_selector_syncing = False
# --- end centre/default migration view patch ---


# --- 2-D Stolt migration tab patch v2 safe image ---
# Adds a 2-D Stolt/f-k migration tab beside the 3-D migration tab.
# This version avoids Qt Matplotlib canvases for the 2-D tab and renders preview PNGs into a QLabel,
# preventing backend_qt draw recursion on some PyQt6/Matplotlib combinations.


def _gpr2d_stolt_migration(data_tx, dt, dx, velocity, apply_jacobian=True, pad_t=1.5, pad_x=1.5, taper_t=0.05, taper_x=0.05, progress_cb=None):
    """2-D constant-velocity Stolt migration using the same 3-D Stolt core with ny=1."""
    data_tx = np.asarray(data_tx, dtype=np.float32)
    if data_tx.ndim != 2:
        raise ValueError('2-D migration input must have shape (time, distance).')
    cube = data_tx[:, :, None]
    mig = stolt_migration_3d(
        cube, dt=float(dt), dx=float(dx), dy=1.0, velocity=float(velocity),
        exploding_reflector=True, apply_jacobian=bool(apply_jacobian),
        pad_t=float(pad_t), pad_x=float(pad_x), pad_y=1.0,
        taper_t=float(taper_t), taper_x=float(taper_x), taper_y=0.0,
        pad_to_pow2=True, depth_padding=2.0,
        progress_cb=progress_cb,
    )
    return np.asarray(mig[:, :, 0], dtype=np.float32)


def _gpr2d_envelope(a):
    a = np.asarray(a, dtype=np.float32)
    try:
        from scipy.signal import hilbert
        return np.abs(hilbert(a, axis=0)).astype(np.float32)
    except Exception:
        # Fallback: keep absolute amplitude rather than failing.
        return np.abs(a).astype(np.float32)


def _gpr2d_k_filter(data_tx, dx, mode):
    mode = (mode or 'Off').lower()
    if mode.startswith('off'):
        return np.asarray(data_tx, dtype=np.float32)
    d = np.asarray(data_tx, dtype=np.float32)
    nt, nx = d.shape
    if nx < 4:
        return d
    fx = np.fft.fftfreq(nx, d=max(float(dx), 1e-9))
    kmax = float(np.nanmax(np.abs(fx))) if fx.size else 1.0
    if 'strong' in mode:
        frac = 0.30
    elif 'medium' in mode:
        frac = 0.45
    else:
        frac = 0.65
    kc = max(frac * kmax, 1e-9)
    filt = np.exp(-(np.abs(fx) / kc) ** 4).astype(np.float32)
    F = np.fft.fft(d, axis=1)
    return np.fft.ifft(F * filt[None, :], axis=1).real.astype(np.float32)


class GPR2DMigrationTab(QWidget):
    def __init__(self, analysis_owner, kind: str):
        super().__init__()
        self.owner = analysis_owner
        self.kind = kind
        self.lines = []
        self.last = None
        self._last_png = None
        self._build_ui()
        self.refresh_lines()

    def _build_ui(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QVBoxLayout, QGridLayout, QLabel, QPushButton,
            QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit,
            QWidget, QScrollArea, QSplitter, QSizePolicy
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(5)

        self.line_combo = QComboBox()
        self.btn_refresh = QPushButton('Refresh lines')
        self.data_mode = QComboBox(); self.data_mode.addItems(['processed', 'raw'])
        self.attribute = QComboBox(); self.attribute.addItems(['Signed amplitude', 'Envelope amplitude', 'Absolute amplitude'])
        self.attribute.setCurrentText('Signed amplitude')
        self.velocity = QDoubleSpinBox(); self.velocity.setRange(0.02, 0.30); self.velocity.setValue(0.10); self.velocity.setSingleStep(0.005); self.velocity.setSuffix(' m/ns')
        self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(0.0); self.tmin.setSuffix(' ns')
        self.tmax = QDoubleSpinBox(); self.tmax.setRange(1, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(' ns')
        self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20.0); self.dt_out.setValue(0.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(' ns')
        self.trace_step = QSpinBox(); self.trace_step.setRange(1, 100); self.trace_step.setValue(1)
        self.k_filter = QComboBox(); self.k_filter.addItems(['Off', 'Light k-filter', 'Medium k-filter', 'Strong k-filter']); self.k_filter.setCurrentText('Off')
        self.pad_t = QDoubleSpinBox(); self.pad_t.setRange(1.0, 4.0); self.pad_t.setValue(1.5); self.pad_t.setSingleStep(0.25)
        self.pad_x = QDoubleSpinBox(); self.pad_x.setRange(1.0, 4.0); self.pad_x.setValue(1.5); self.pad_x.setSingleStep(0.25)
        self.clip_pct = QDoubleSpinBox(); self.clip_pct.setRange(80.0, 100.0); self.clip_pct.setValue(98.5); self.clip_pct.setSingleStep(0.1); self.clip_pct.setSuffix(' %')
        self.jacobian = QCheckBox('Stolt Jacobian'); self.jacobian.setChecked(True)
        self.mute_top = QCheckBox('Mute before tmin, preserve t=0'); self.mute_top.setChecked(True)

        entries = [
            ('Line', self.line_combo), ('Data', self.data_mode), ('Attribute', self.attribute),
            ('Velocity', self.velocity), ('Mute tmin', self.tmin), ('tmax', self.tmax),
            ('Output dt', self.dt_out), ('Trace step', self.trace_step), ('k-filter', self.k_filter),
            ('Pad time', self.pad_t), ('Pad distance', self.pad_x), ('Clip', self.clip_pct),
        ]
        for r, (lab, w) in enumerate(entries):
            lbl = QLabel(lab); lbl.setMinimumWidth(78)
            grid.addWidget(lbl, r, 0)
            grid.addWidget(w, r, 1)
        grid.addWidget(self.btn_refresh, len(entries), 0, 1, 2)
        left_lay.addLayout(grid)
        left_lay.addWidget(self.jacobian)
        left_lay.addWidget(self.mute_top)

        self.btn_run = QPushButton('Run 2-D migration')
        self.btn_png = QPushButton('Export PNG')
        self.btn_npz = QPushButton('Export NPZ')
        for b in [self.btn_run, self.btn_png, self.btn_npz]:
            left_lay.addWidget(b)
        left_lay.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300); scroll.setMaximumWidth(390); scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.preview_scroll = QScrollArea(); self.preview_scroll.setWidgetResizable(True)
        self.image_label = QLabel('Run 2-D migration to display selected-line input and migrated depth section.')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(700, 360)
        self.preview_scroll.setWidget(self.image_label)
        right_lay.addWidget(self.preview_scroll, stretch=1)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(90)
        right_lay.addWidget(self.log)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        try: splitter.setSizes([330, 1100])
        except Exception: pass

        self.btn_refresh.clicked.connect(self.refresh_lines)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_png.clicked.connect(self.export_png)
        self.btn_npz.clicked.connect(self.export_npz)

    def _status(self, txt):
        from PyQt6.QtWidgets import QApplication
        self.log.append(str(txt))
        try:
            target = getattr(getattr(self.owner, 'main', None), 'status', None) or getattr(getattr(self.owner, 'owner', None), 'status', None)
            if target is not None:
                target.setText(str(txt))
        except Exception:
            pass
        QApplication.processEvents()

    def _project_root(self):
        if self.kind == 'schleitheim':
            return Path('/home/luqman/gpr_gui/data/MALA')
        return Path(getattr(getattr(self.owner, 'owner', None), 'root', '/home/luqman/gpr_gui/data/PulseEkko'))

    def _available_lines(self):
        if self.kind == 'schleitheim':
            try: lines = list(self.owner.selected_lines_for_maps())
            except Exception: lines = list(getattr(getattr(self.owner, 'main', None), 'lines', []))
        else:
            try: lines = list(self.owner.selected_lines())
            except Exception: lines = list(getattr(getattr(self.owner, 'owner', None), 'lines', []))
        def key(line):
            try: parent = line.folder.parent.name.lower()
            except Exception: parent = ''
            rank = 0 if parent == 'inline' else 1 if parent == 'crossline' else 2
            try: num = int(getattr(line, 'number', 999999))
            except Exception: num = 999999
            return (rank, num, str(getattr(line, 'name', '')))
        return sorted(lines, key=key)

    def refresh_lines(self):
        old_name = None
        try: old_name = getattr(self._current_line(), 'name', None)
        except Exception: pass
        self.lines = self._available_lines()
        self.line_combo.blockSignals(True)
        try:
            self.line_combo.clear()
            default_idx = 0
            for i, line in enumerate(self.lines):
                try: parent = line.folder.parent.name.lower()
                except Exception: parent = ''
                prefix = 'I' if parent == 'inline' else 'C' if parent == 'crossline' else 'L'
                try: num = int(getattr(line, 'number', i+1))
                except Exception: num = i + 1
                label = f'{prefix}{num}: {getattr(line,"direction","").upper()}  {getattr(line,"name","")}'
                self.line_combo.addItem(label, i)
                if old_name and getattr(line, 'name', None) == old_name:
                    default_idx = i
                elif (not old_name and parent == 'inline' and num == 3):
                    default_idx = i
                elif (not old_name and default_idx == 0 and parent == 'inline' and num == 1):
                    default_idx = i
            if self.lines:
                self.line_combo.setCurrentIndex(default_idx)
        finally:
            self.line_combo.blockSignals(False)
        self._status(f'2-D migration line list: {len(self.lines)} line(s). Default is inline 1 when available.')

    def _current_line(self):
        if not self.lines:
            # avoid recursive status/update loops; just refresh once
            self.lines = self._available_lines()
        if not self.lines:
            raise RuntimeError('No selected/available line for 2-D migration.')
        idx = self.line_combo.currentData()
        try: idx = int(idx)
        except Exception: idx = int(self.line_combo.currentIndex())
        idx = max(0, min(idx, len(self.lines) - 1))
        return self.lines[idx]

    def _line_array_time_xy(self, line):
        if self.kind == 'schleitheim':
            old = None
            try:
                old = self.owner.mode.currentText()
                self.owner.mode.setCurrentText(self.data_mode.currentText())
            except Exception:
                pass
            try: arr = self.owner.ensure_data(line)
            finally:
                if old is not None:
                    try: self.owner.mode.setCurrentText(old)
                    except Exception: pass
            t = self.owner.time_vector(line, arr.shape[1] if np.ndim(arr) == 2 else 0)
            try: x, y, _z = self.owner.trace_xyz(line, arr.shape[0])
            except Exception:
                try: x, y = self.owner.trace_xy(line, arr.shape[0])
                except Exception:
                    x = np.arange(arr.shape[0], dtype=float); y = np.zeros(arr.shape[0], dtype=float)
        else:
            old = None
            try:
                old = self.owner.data_choice.currentText()
                self.owner.data_choice.setCurrentText(self.data_mode.currentText())
            except Exception:
                pass
            try: arr = self.owner.get_array(line)
            finally:
                if old is not None:
                    try: self.owner.data_choice.setCurrentText(old)
                    except Exception: pass
            try: t = self.owner.owner.corrected_time_ns(line)
            except Exception: t = getattr(line, 'time_ns', None)
            x = np.asarray(getattr(line, 'x', np.arange(arr.shape[0])), dtype=float)
            y = np.asarray(getattr(line, 'y', np.zeros(arr.shape[0])), dtype=float)

        arr = np.asarray(arr, dtype=float)
        t = np.asarray(t, dtype=float)
        if arr.ndim != 2:
            raise RuntimeError('Line data are not 2-D.')
        # Desired convention for this code: arr[trace, sample]
        if t.size == arr.shape[0] and t.size != arr.shape[1]:
            arr = arr.T.copy()
        if t.size != arr.shape[1]:
            t = np.linspace(0.0, float(getattr(line, 'time_window_ns', arr.shape[1]-1) or arr.shape[1]-1), arr.shape[1])
        ntr = arr.shape[0]
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if x.size != ntr and x.size >= 2:
            x = np.interp(np.linspace(0, 1, ntr), np.linspace(0, 1, x.size), x)
        elif x.size != ntr:
            x = np.arange(ntr, dtype=float)
        if y.size != ntr and y.size >= 2:
            y = np.interp(np.linspace(0, 1, ntr), np.linspace(0, 1, y.size), y)
        elif y.size != ntr:
            y = np.zeros(ntr, dtype=float)
        return arr, t, x, y

    def _prepare_line_section(self, progress=None):
        line = self._current_line()
        if progress: progress(5, f'Reading selected line: {getattr(line,"name",line)}')
        arr, t, x, y = self._line_array_time_xy(line)
        step = max(1, int(self.trace_step.value()))
        idx = np.arange(0, arr.shape[0], step, dtype=int)
        arr = arr[idx, :]; x = x[idx]; y = y[idx]
        dist = np.zeros(arr.shape[0], dtype=float)
        if arr.shape[0] > 1:
            dist[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        dd = np.diff(dist)
        dx = float(np.nanmedian(dd)) if dd.size and np.any(np.isfinite(dd)) else 0.05
        if not np.isfinite(dx) or dx <= 0: dx = 0.05
        tmax = max(1.0, float(self.tmax.value()))
        dt = max(0.05, float(self.dt_out.value()))
        t_axis = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=float)
        if t_axis.size < 8:
            raise RuntimeError('Time window too short for 2-D migration.')
        if progress: progress(20, f'Resampling {arr.shape[0]} traces onto regular time axis...')
        data = np.zeros((t_axis.size, arr.shape[0]), dtype=np.float32)
        t_good = np.isfinite(t)
        if np.count_nonzero(t_good) < 2: raise RuntimeError('Invalid time vector for selected line.')
        tt = t[t_good]
        order = np.argsort(tt)
        tt = tt[order]
        for j in range(arr.shape[0]):
            tr = np.asarray(arr[j, :], dtype=float)[t_good][order]
            data[:, j] = np.interp(t_axis, tt, tr, left=0.0, right=0.0).astype(np.float32)
        if self.mute_top.isChecked():
            data[t_axis < float(self.tmin.value()), :] = 0.0
        attr = self.attribute.currentText().lower()
        if attr.startswith('envelope'):
            data_display = _gpr2d_envelope(data)
        elif attr.startswith('absolute'):
            data_display = np.abs(data).astype(np.float32)
        else:
            data_display = data.astype(np.float32)
        data_proc = _gpr2d_k_filter(data_display, dx, self.k_filter.currentText())
        return dict(line=line, input=data_display, proc=data_proc, t=t_axis, dist=dist, dx=dx, dt=dt)

    def run_pipeline(self):
        from PyQt6.QtWidgets import QProgressDialog, QMessageBox, QApplication
        dlg = QProgressDialog('Running 2-D Stolt migration...', 'Cancel', 0, 100, self)
        dlg.setWindowTitle('2-D migration'); dlg.setMinimumDuration(0); dlg.setValue(0)
        def progress(v, msg):
            dlg.setValue(int(max(0, min(100, v))))
            dlg.setLabelText(str(msg)); self._status(str(msg)); QApplication.processEvents()
            if dlg.wasCanceled(): raise RuntimeError('2-D migration cancelled by user.')
        try:
            sec = self._prepare_line_section(progress)
            vel = float(self.velocity.value())
            progress(35, 'Running 2-D Stolt/f-k migration using the same remapping core...')
            mig = _gpr2d_stolt_migration(
                sec['proc'], dt=sec['dt'], dx=sec['dx'], velocity=vel,
                apply_jacobian=self.jacobian.isChecked(), pad_t=float(self.pad_t.value()), pad_x=float(self.pad_x.value()),
                taper_t=0.05, taper_x=0.05, progress_cb=lambda v, m: progress(35 + int(55*v/100), m)
            )
            dz = 0.5 * vel * sec['dt']
            depth = np.arange(mig.shape[0], dtype=float) * dz
            self.last = dict(**sec, migrated=mig, depth=depth, velocity=vel)
            progress(95, 'Drawing 2-D migration comparison...')
            self.plot_result()
            progress(100, '2-D migration finished.')
        except Exception as e:
            self._status('2-D migration failed: ' + str(e))
            QMessageBox.critical(self, '2-D migration failed', f'{e}\n\n{traceback.format_exc()}')
        finally:
            dlg.close()

    def _clip(self, arr):
        a = np.asarray(arr, dtype=float)
        finite = np.isfinite(a)
        if not np.any(finite): return 1.0
        pct = float(self.clip_pct.value())
        v = float(np.nanpercentile(np.abs(a[finite]), pct))
        return max(v, 1e-9)

    def plot_result(self):
        if not self.last: return
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import tempfile
        r = self.last
        fig = Figure(figsize=(14.5, 8.2), tight_layout=False)
        FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        dist = r['dist']; t = r['t']; depth = r['depth']
        extent_in = [float(dist[0]), float(dist[-1]), float(t[-1]), float(t[0])]
        extent_mig = [float(dist[0]), float(dist[-1]), float(depth[-1]), float(depth[0])]
        c1 = self._clip(r['input']); c2 = self._clip(r['migrated'])
        im1 = ax1.imshow(r['input'], aspect='auto', cmap='seismic', vmin=-c1, vmax=c1, extent=extent_in)
        im2 = ax2.imshow(r['migrated'], aspect='auto', cmap='seismic', vmin=-c2, vmax=c2, extent=extent_mig)
        line = r['line']
        try: parent = line.folder.parent.name.lower()
        except Exception: parent = ''
        prefix = 'I' if parent == 'inline' else 'C' if parent == 'crossline' else 'L'
        name = f'{prefix}{getattr(line,"number","")}: {getattr(line,"direction","").upper()}  {getattr(line,"name","")}'
        ax1.set_title(name + ' — input, muted before tmin')
        ax1.set_xlabel('Distance along selected line [m]'); ax1.set_ylabel('TWT [ns]')
        fig.colorbar(im1, ax=ax1, shrink=0.70, pad=0.015, label='Amplitude')
        ax2.set_title(name + ' — 2-D Stolt migrated')
        ax2.set_xlabel('Distance along selected line [m]'); ax2.set_ylabel('Depth [m]')
        fig.colorbar(im2, ax=ax2, shrink=0.70, pad=0.015, label='Amplitude')
        fig.suptitle(f'2-D selected-line Stolt/f-k migration | v={r["velocity"]:.3f} m/ns | dx={r["dx"]:.3f} m | not 3-D volume')
        try: fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception: pass
        out = Path(tempfile.gettempdir()) / f'gpr2d_migration_preview_{id(self)}.png'
        fig.savefig(out, dpi=220, bbox_inches='tight')
        self._last_png = out
        pix = QPixmap(str(out))
        vp = self.preview_scroll.viewport().size()
        w = max(1200, vp.width() - 28)
        if pix.width() > w:
            pix = pix.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_label.setPixmap(pix)
        self.image_label.setMinimumSize(pix.size())
        self.image_label.adjustSize()
        self._status(f'2-D migration result: line={name}; input={r["input"].shape}; migrated={r["migrated"].shape}; dx={r["dx"]:.3f} m.')

    def _default_out(self, suffix):
        root = self._project_root(); outdir = root / 'migration_exports'; outdir.mkdir(parents=True, exist_ok=True)
        line = self.last['line'] if self.last else self._current_line()
        line_name = str(getattr(line, 'name', 'selected_line')).replace('/', '_').replace(' ', '_')
        name = 'schleitheim_mala' if self.kind == 'schleitheim' else 'bulach_pulseekko'
        return outdir / f'{name}_2d_stolt_{line_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{suffix}'

    def export_png(self):
        from PyQt6.QtWidgets import QMessageBox
        if not self.last:
            QMessageBox.warning(self, 'No 2-D migration', 'Run 2-D migration first.'); return
        if self._last_png is None or not Path(self._last_png).exists():
            self.plot_result()
        out = self._default_out('.png')
        import shutil
        shutil.copyfile(self._last_png, out)
        self._status(f'Saved 2-D migration PNG: {out}')

    def export_npz(self):
        from PyQt6.QtWidgets import QMessageBox
        if not self.last:
            QMessageBox.warning(self, 'No 2-D migration', 'Run 2-D migration first.'); return
        out = self._default_out('.npz')
        r = self.last
        np.savez_compressed(out, input=r['input'], processed_for_migration=r['proc'], migrated=r['migrated'], time_ns=r['t'], depth_m=r['depth'], distance_m=r['dist'], dx_m=r['dx'], velocity_m_ns=r['velocity'], line_name=str(getattr(r['line'], 'name', '')))
        self._status(f'Saved 2-D migration NPZ: {out}')


def _insert_2d_tab_next_to_3d(tabw, widget):
    try:
        idx = tabw.count()
        for i in range(tabw.count()):
            txt = tabw.tabText(i).lower().replace('-', '')
            if '3d migration' in txt or '3 d migration' in txt:
                idx = i
                break
        else:
            for i in range(tabw.count()):
                if 'selected fence' in tabw.tabText(i).lower():
                    idx = i + 1; break
        tabw.insertTab(idx, widget, '2D Migration')
        return True
    except Exception:
        return False


_GPR2D_SAFE_PREV_APPLY_SCHLEITHEIM = apply_schleitheim
_GPR2D_SAFE_PREV_APPLY_BULACH = apply_bulach


def apply_schleitheim(globs):
    try: _GPR2D_SAFE_PREV_APPLY_SCHLEITHEIM(globs)
    except Exception as e: print('Existing 3-D migration patch failed before 2-D tab insertion:', e)
    cls = globs.get('GPR3DStandardAnalysisTab') or globs.get('GPR3DAnalysisTab')
    if cls is None or getattr(cls, '_gpr2d_safe_stolt_patched', False): return
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        try:
            if not hasattr(self, 'gpr2d_migration_tab'):
                self.gpr2d_migration_tab = GPR2DMigrationTab(self, 'schleitheim')
                _insert_2d_tab_next_to_3d(self.tabs, self.gpr2d_migration_tab)
        except Exception as e:
            print('Could not add Schleitheim 2-D Migration tab:', e)
    cls.__init__ = new_init
    cls._gpr2d_safe_stolt_patched = True
    print('2-D Stolt migration tab active for Schleitheim/MALA.')


def apply_bulach(globs):
    try: _GPR2D_SAFE_PREV_APPLY_BULACH(globs)
    except Exception as e: print('Existing 3-D migration patch failed before 2-D tab insertion:', e)
    cls = globs.get('PulseEkko3DAnalysis')
    if cls is None or getattr(cls, '_gpr2d_safe_stolt_patched', False): return
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        try:
            if not hasattr(self, 'gpr2d_migration_tab'):
                self.gpr2d_migration_tab = GPR2DMigrationTab(self, 'bulach')
                _insert_2d_tab_next_to_3d(self.tabs, self.gpr2d_migration_tab)
        except Exception as e:
            print('Could not add Bulach 2-D Migration tab:', e)
    cls.__init__ = new_init
    cls._gpr2d_safe_stolt_patched = True
    print('2-D Stolt migration tab active for Bulach/PulseEKKO.')
# --- end 2-D Stolt migration tab patch v2 safe image ---


# --- 2-D migration visual SEC/AGC display patch ---
def _gpr2d_visual_sec_agc(a, z_or_t):
    """Display-only SEC + AGC-style gain for seeing hyperbola collapse."""
    import numpy as _np
    a = _np.asarray(a, dtype=float).copy()
    z = _np.asarray(z_or_t, dtype=float)
    if a.ndim != 2 or z.size != a.shape[0]:
        return a

    finite = _np.isfinite(a)
    if not _np.any(finite):
        return a

    # SEC/t-power style gain along time/depth axis.
    zn = z - _np.nanmin(z)
    denom = float(_np.nanmax(zn)) if _np.nanmax(zn) > 0 else 1.0
    gain = (1.0 + 3.0 * zn / denom) ** 1.35
    a *= gain[:, None]

    # AGC-style running RMS equalisation along vertical axis.
    try:
        from scipy.ndimage import uniform_filter1d
        win = max(7, int(round(a.shape[0] / 18)))
        rms = _np.sqrt(uniform_filter1d(a * a, size=win, axis=0, mode='nearest'))
        ref = float(_np.nanmedian(rms[_np.isfinite(rms) & (rms > 0)])) if _np.any(_np.isfinite(rms) & (rms > 0)) else 1.0
        a = a / (rms + 0.15 * ref) * ref
    except Exception:
        pass

    return a.astype(float)


def _gpr2d_plot_result_visual_gain(self):
    if not self.last:
        return
    from pathlib import Path as _Path
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import tempfile
    import numpy as _np

    r = self.last
    fig = Figure(figsize=(14.5, 8.2), tight_layout=False)
    FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    dist = _np.asarray(r['dist'], dtype=float)
    t = _np.asarray(r['t'], dtype=float)
    depth = _np.asarray(r['depth'], dtype=float)

    inp = _gpr2d_visual_sec_agc(r['input'], t)
    mig = _gpr2d_visual_sec_agc(r['migrated'], depth)

    extent_in = [float(dist[0]), float(dist[-1]), float(t[-1]), float(t[0])]
    extent_mig = [float(dist[0]), float(dist[-1]), float(depth[-1]), float(depth[0])]

    c1 = self._clip(inp)
    c2 = self._clip(mig)

    im1 = ax1.imshow(inp, aspect='auto', cmap='seismic', vmin=-c1, vmax=c1, extent=extent_in)
    im2 = ax2.imshow(mig, aspect='auto', cmap='seismic', vmin=-c2, vmax=c2, extent=extent_mig)

    line = r['line']
    try:
        parent = line.folder.parent.name.lower()
    except Exception:
        parent = ''
    prefix = 'I' if parent == 'inline' else 'C' if parent == 'crossline' else 'L'
    name = f'{prefix}{getattr(line,"number","")}: {getattr(line,"direction","").upper()}  {getattr(line,"name","")}'

    ax1.set_title(name + ' — input, display SEC+AGC gain')
    ax1.set_xlabel('Distance along selected line [m]')
    ax1.set_ylabel('TWT [ns]')
    fig.colorbar(im1, ax=ax1, shrink=0.74, pad=0.012, label='Display amplitude')

    ax2.set_title(name + ' — 2-D Stolt migrated, display SEC+AGC gain')
    ax2.set_xlabel('Distance along selected line [m]')
    ax2.set_ylabel('Depth [m]')
    fig.colorbar(im2, ax=ax2, shrink=0.74, pad=0.012, label='Display amplitude')

    fig.suptitle(f'2-D selected-line Stolt/f-k migration | v={r["velocity"]:.3f} m/ns | dx={r["dx"]:.3f} m | display gain only')
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass

    out = _Path(tempfile.gettempdir()) / f'gpr2d_migration_preview_{id(self)}.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    self._last_png = out

    pix = QPixmap(str(out))
    vp = self.preview_scroll.viewport().size()
    w = max(1200, vp.width() - 28)
    if pix.width() > w:
        pix = pix.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)

    self.preview_scroll.setWidgetResizable(False)
    self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.image_label.setPixmap(pix)
    self.image_label.setMinimumSize(pix.size())
    self.image_label.adjustSize()

    self._status(f'2-D migration result: line={name}; display uses SEC+AGC gain only; saved NPZ remains ungained.')

try:
    GPR2DMigrationTab.plot_result = _gpr2d_plot_result_visual_gain
except Exception:
    pass
# --- end 2-D migration visual SEC/AGC display patch ---


# --- 2-D migration view-in-window patch ---
def _gpr2d_open_preview_window(self):
    from pathlib import Path as _Path
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt

    png = getattr(self, '_last_png', None)
    if not png or not _Path(png).exists():
        self._status('Run 2-D migration first; no preview image available yet.')
        return

    dlg = QDialog(self)
    dlg.setWindowTitle('2-D Migration Preview')
    dlg.resize(1300, 850)

    lab = QLabel()
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setPixmap(QPixmap(str(png)))
    lab.adjustSize()

    scroll = QScrollArea()
    scroll.setWidgetResizable(False)
    scroll.setWidget(lab)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    lay = QVBoxLayout(dlg)
    lay.addWidget(scroll)

    if not hasattr(self, '_preview_dialogs'):
        self._preview_dialogs = []
    self._preview_dialogs.append(dlg)
    dlg.show()


try:
    if not hasattr(GPR2DMigrationTab, '_orig_init_view_window_patch'):
        GPR2DMigrationTab._orig_init_view_window_patch = GPR2DMigrationTab.__init__

    def _gpr2d_init_view_window_patch(self, *args, **kwargs):
        GPR2DMigrationTab._orig_init_view_window_patch(self, *args, **kwargs)
        from PyQt6.QtWidgets import QPushButton
        for btn in self.findChildren(QPushButton):
            if btn.text().strip().lower() == 'export png':
                btn.setText('View in another window')
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _=False, w=self: _gpr2d_open_preview_window(w))
                break

    GPR2DMigrationTab.__init__ = _gpr2d_init_view_window_patch
except Exception:
    pass
# --- end 2-D migration view-in-window patch ---


# --- 2-D migration best default settings patch ---
def _gpr2d_apply_best_defaults(self):
    import re
    try:
        self.tmin.setValue(60.0)       # mute shallow/direct wave; focus deeper hyperbola
        self.tmax.setValue(180.0)
        self.dt_out.setValue(0.5)      # finer migration sampling
        self.trace_step.setValue(1)
        self.pad_time.setValue(1.5)
        self.pad_dist.setValue(1.5)
        self.clip.setValue(98.5)
        if hasattr(self, 'velocity'):
            self.velocity.setValue(0.10)
        if hasattr(self, 'k_filter'):
            self.k_filter.setCurrentText('Off')
        if hasattr(self, 'jacobian'):
            self.jacobian.setChecked(True)
        if hasattr(self, 'mute_before_tmin'):
            self.mute_before_tmin.setChecked(True)
    except Exception:
        pass

    # Prefer inline 3 because that is where the clear hyperbola is seen.
    try:
        combos = [c for c in self.findChildren(__import__('PyQt6.QtWidgets').QtWidgets.QComboBox) if c.count() > 5]
        for cb in combos:
            best = None
            fallback = None
            for i in range(cb.count()):
                txt = cb.itemText(i).lower().replace(' ', '')
                if re.search(r'\bi3:', txt) or '/line3_' in txt or '/line_3_' in txt or 'line3_ba' in txt or 'line_3_ba' in txt:
                    best = i
                    break
                if fallback is None and (re.search(r'\bi1:', txt) or '/line1_' in txt or '/line_1_' in txt):
                    fallback = i
            if best is not None:
                cb.setCurrentIndex(best)
                break
            elif fallback is not None:
                cb.setCurrentIndex(fallback)
    except Exception:
        pass

try:
    if not hasattr(GPR2DMigrationTab, '_orig_init_best_defaults_patch'):
        GPR2DMigrationTab._orig_init_best_defaults_patch = GPR2DMigrationTab.__init__

    def _gpr2d_init_best_defaults_patch(self, *args, **kwargs):
        GPR2DMigrationTab._orig_init_best_defaults_patch(self, *args, **kwargs)
        _gpr2d_apply_best_defaults(self)

    GPR2DMigrationTab.__init__ = _gpr2d_init_best_defaults_patch
except Exception:
    pass
# --- end 2-D migration best default settings patch ---


# --- 2-D velocity sweep patch ---
def _gpr2d_robust_clip_for_plot(a, pct=98.5):
    import numpy as _np
    a = _np.asarray(a, dtype=float)
    good = _np.isfinite(a)
    if not _np.any(good):
        return 1.0
    return max(float(_np.nanpercentile(_np.abs(a[good]), float(pct))), 1e-9)


def _gpr2d_display_gain_safe(a, axis):
    try:
        return _gpr2d_visual_sec_agc(a, axis)
    except Exception:
        return a


def _gpr2d_line_label(self, line):
    try:
        parent = line.folder.parent.name.lower()
    except Exception:
        parent = ''
    prefix = 'I' if parent == 'inline' else 'C' if parent == 'crossline' else 'L'
    return f'{prefix}{getattr(line,"number","")}: {getattr(line,"direction","").upper()}  {getattr(line,"name","")}'


def _gpr2d_run_velocity_sweep(self):
    from PyQt6.QtWidgets import QProgressDialog, QMessageBox, QApplication
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from pathlib import Path as _Path
    import numpy as _np
    import tempfile, traceback

    velocities = [0.06, 0.08, 0.10, 0.12, 0.14]

    dlg = QProgressDialog('Running 2-D velocity sweep...', 'Cancel', 0, 100, self)
    dlg.setWindowTitle('2-D velocity sweep')
    dlg.setMinimumDuration(0)
    dlg.setValue(0)

    def progress(v, msg):
        dlg.setValue(int(max(0, min(100, v))))
        dlg.setLabelText(str(msg))
        self._status(str(msg))
        QApplication.processEvents()
        if dlg.wasCanceled():
            raise RuntimeError('2-D velocity sweep cancelled by user.')

    try:
        # Good defaults for hyperbola-collapse check.
        try:
            self.tmin.setValue(60.0)
            self.tmax.setValue(180.0)
            self.dt_out.setValue(0.5)
            self.trace_step.setValue(1)
            self.k_filter.setCurrentText('Off')
            self.clip_pct.setValue(98.5)
            self.jacobian.setChecked(True)
            self.mute_top.setChecked(True)
        except Exception:
            pass

        sec = self._prepare_line_section(lambda v, m: progress(int(v * 0.2), m))
        line = sec['line']
        name = _gpr2d_line_label(self, line)
        dist = _np.asarray(sec['dist'], dtype=float)
        t = _np.asarray(sec['t'], dtype=float)

        sweep = []
        for i, vel in enumerate(velocities):
            progress(20 + i * 14, f'Running 2-D Stolt migration at v={vel:.2f} m/ns...')
            mig = _gpr2d_stolt_migration(
                sec['proc'], dt=sec['dt'], dx=sec['dx'], velocity=vel,
                apply_jacobian=self.jacobian.isChecked(),
                pad_t=float(self.pad_t.value()),
                pad_x=float(self.pad_x.value()),
                taper_t=0.05,
                taper_x=0.05,
                progress_cb=None,
            )
            dz = 0.5 * vel * sec['dt']
            depth = _np.arange(mig.shape[0], dtype=float) * dz
            sweep.append((vel, mig, depth))

        progress(92, 'Drawing velocity sweep comparison...')

        rows = 1 + len(sweep)
        fig = Figure(figsize=(14.5, 2.6 * rows), tight_layout=False)
        FigureCanvasAgg(fig)

        inp = _gpr2d_display_gain_safe(sec['input'], t)
        cin = _gpr2d_robust_clip_for_plot(inp, self.clip_pct.value())
        ax = fig.add_subplot(rows, 1, 1)
        ax.imshow(inp, aspect='auto', cmap='seismic', vmin=-cin, vmax=cin,
                  extent=[float(dist[0]), float(dist[-1]), float(t[-1]), float(t[0])])
        ax.set_title(f'{name} — input, muted before {float(self.tmin.value()):.0f} ns, display gain')
        ax.set_ylabel('TWT [ns]')
        ax.set_xlabel('Distance [m]')

        for r, (vel, mig, depth) in enumerate(sweep, start=2):
            ax = fig.add_subplot(rows, 1, r)
            show = _gpr2d_display_gain_safe(mig, depth)
            c = _gpr2d_robust_clip_for_plot(show, self.clip_pct.value())
            ax.imshow(show, aspect='auto', cmap='seismic', vmin=-c, vmax=c,
                      extent=[float(dist[0]), float(dist[-1]), float(depth[-1]), float(depth[0])])
            ax.set_title(f'2-D Stolt migrated velocity sweep: v={vel:.2f} m/ns')
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Distance [m]')

        fig.suptitle('2-D migration velocity sweep — choose velocity with tightest hyperbola collapse, least smile/frown artefact')
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass

        out = _Path(tempfile.gettempdir()) / f'gpr2d_velocity_sweep_{id(self)}.png'
        fig.savefig(out, dpi=220, bbox_inches='tight')
        self._last_png = out
        self.last_sweep = dict(section=sec, velocities=velocities, results=sweep, png=out)

        pix = QPixmap(str(out))
        vp = self.preview_scroll.viewport().size()
        w = max(1200, vp.width() - 28)
        if pix.width() > w:
            pix = pix.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)

        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_label.setPixmap(pix)
        self.image_label.setMinimumSize(pix.size())
        self.image_label.adjustSize()

        progress(100, '2-D velocity sweep finished.')
        self._status('Velocity sweep finished: compare which velocity best collapses the hyperbola.')
    except Exception as e:
        self._status('2-D velocity sweep failed: ' + str(e))
        QMessageBox.critical(self, '2-D velocity sweep failed', f'{e}\n\n{traceback.format_exc()}')
    finally:
        dlg.close()


try:
    if not hasattr(GPR2DMigrationTab, '_orig_init_velocity_sweep_patch'):
        GPR2DMigrationTab._orig_init_velocity_sweep_patch = GPR2DMigrationTab.__init__

    def _gpr2d_init_velocity_sweep_patch(self, *args, **kwargs):
        GPR2DMigrationTab._orig_init_velocity_sweep_patch(self, *args, **kwargs)

        from PyQt6.QtWidgets import QPushButton
        import re

        # Strong defaults for line-3 hyperbola testing.
        try:
            self.tmin.setValue(60.0)
            self.tmax.setValue(180.0)
            self.dt_out.setValue(0.5)
            self.trace_step.setValue(1)
            self.k_filter.setCurrentText('Off')
            self.velocity.setValue(0.10)
            self.clip_pct.setValue(98.5)
        except Exception:
            pass

        # Prefer inline 3.
        try:
            for i in range(self.line_combo.count()):
                txt = self.line_combo.itemText(i).lower().replace(' ', '')
                if re.search(r'\bi3:', txt) or '/line3_' in txt or '/line_3_' in txt or 'line3_ba' in txt or 'line_3_ba' in txt:
                    self.line_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

        if not hasattr(self, 'btn_velocity_sweep'):
            self.btn_velocity_sweep = QPushButton('Velocity sweep')
            try:
                lay = self.btn_run.parentWidget().layout()
                idx = lay.indexOf(self.btn_run)
                lay.insertWidget(idx + 1, self.btn_velocity_sweep)
            except Exception:
                try:
                    self.layout().addWidget(self.btn_velocity_sweep)
                except Exception:
                    pass
            self.btn_velocity_sweep.clicked.connect(lambda _=False, w=self: _gpr2d_run_velocity_sweep(w))

    GPR2DMigrationTab.__init__ = _gpr2d_init_velocity_sweep_patch
except Exception:
    pass
# --- end 2-D velocity sweep patch ---


# --- 2-D preview popup maximisable scroll fix ---
def _gpr2d_open_preview_window(self):
    from pathlib import Path as _Path
    from PyQt6.QtWidgets import QMainWindow, QLabel, QScrollArea
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt

    png = getattr(self, '_last_png', None)
    if not png or not _Path(png).exists():
        self._status('Run 2-D migration or velocity sweep first; no preview image available yet.')
        return

    win = QMainWindow(self)
    win.setWindowTitle('2-D Migration Preview')
    win.setWindowFlags(
        Qt.WindowType.Window
        | Qt.WindowType.WindowMinimizeButtonHint
        | Qt.WindowType.WindowMaximizeButtonHint
        | Qt.WindowType.WindowCloseButtonHint
    )

    lab = QLabel()
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    pix = QPixmap(str(png))
    lab.setPixmap(pix)
    lab.setMinimumSize(pix.size())
    lab.adjustSize()

    scroll = QScrollArea()
    scroll.setWidgetResizable(False)
    scroll.setWidget(lab)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    win.setCentralWidget(scroll)
    win.resize(1500, 950)
    win.showMaximized()

    if not hasattr(self, '_preview_windows'):
        self._preview_windows = []
    self._preview_windows.append(win)
# --- end 2-D preview popup maximisable scroll fix ---


# --- 2-D zoomable centred preview window patch ---
def _gpr2d_open_preview_window(self):
    from pathlib import Path as _Path
    from PyQt6.QtWidgets import QMainWindow, QLabel, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt, QTimer

    png = getattr(self, '_last_png', None)
    if not png or not _Path(png).exists():
        self._status('Run 2-D migration or velocity sweep first; no preview image available yet.')
        return

    win = QMainWindow(self)
    win.setWindowTitle('2-D Migration Preview')
    win.setWindowFlags(
        Qt.WindowType.Window
        | Qt.WindowType.WindowMinimizeButtonHint
        | Qt.WindowType.WindowMaximizeButtonHint
        | Qt.WindowType.WindowCloseButtonHint
    )

    orig = QPixmap(str(png))
    lab = QLabel()
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setPixmap(orig)
    lab.setMinimumSize(orig.size())
    lab.adjustSize()

    scroll = QScrollArea()
    scroll.setWidgetResizable(False)
    scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
    scroll.setWidget(lab)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    win._orig_pix = orig
    win._scale = 1.0

    def apply_scale():
        w = max(1, int(win._orig_pix.width() * win._scale))
        h = max(1, int(win._orig_pix.height() * win._scale))
        pix = win._orig_pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        lab.setPixmap(pix)
        lab.setMinimumSize(pix.size())
        lab.adjustSize()

    def zoom(factor):
        win._scale = max(0.15, min(6.0, win._scale * factor))
        apply_scale()

    def fit_width():
        vw = max(200, scroll.viewport().width() - 30)
        win._scale = max(0.15, min(6.0, vw / max(1, win._orig_pix.width())))
        apply_scale()

    def fit_window():
        vw = max(200, scroll.viewport().width() - 30)
        vh = max(200, scroll.viewport().height() - 30)
        win._scale = max(0.15, min(6.0, min(vw / max(1, win._orig_pix.width()), vh / max(1, win._orig_pix.height()))))
        apply_scale()

    def actual_size():
        win._scale = 1.0
        apply_scale()

    def toggle_fullscreen():
        if win.isFullScreen():
            win.showMaximized()
        else:
            win.showFullScreen()
        QTimer.singleShot(100, fit_width)

    top = QWidget()
    bar = QHBoxLayout(top)
    bar.setContentsMargins(6, 4, 6, 4)

    for text, fn in [
        ('Fit width', fit_width),
        ('Fit window', fit_window),
        ('100%', actual_size),
        ('Zoom -', lambda: zoom(0.8)),
        ('Zoom +', lambda: zoom(1.25)),
        ('Full screen', toggle_fullscreen),
    ]:
        b = QPushButton(text)
        b.clicked.connect(fn)
        bar.addWidget(b)

    bar.addStretch(1)

    central = QWidget()
    lay = QVBoxLayout(central)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(top, stretch=0)
    lay.addWidget(scroll, stretch=1)
    win.setCentralWidget(central)

    win.resize(1500, 950)
    win.showMaximized()
    QTimer.singleShot(150, fit_width)

    if not hasattr(self, '_preview_windows'):
        self._preview_windows = []
    self._preview_windows.append(win)
# --- end 2-D zoomable centred preview window patch ---


# --- reference-style 3-D migration plot template patch ---
def _g3d_reference_clip(a):
    import numpy as _np
    a = _np.asarray(a, dtype=float)
    good = _np.isfinite(a)
    if not _np.any(good):
        return 1.0
    # Reference-notebook style: robust standard-deviation display.
    v = 3.0 * float(_np.nanstd(a[good]))
    if not _np.isfinite(v) or v <= 0:
        v = float(_np.nanpercentile(_np.abs(a[good]), 98.5))
    return max(v, 1e-9)


def _g3d_selected_indices_reference_style(self, r):
    import numpy as _np
    mig = _np.asarray(r["migrated"])
    nz, nx, ny = mig.shape

    ix = nx // 2
    iy = ny // 2

    try:
        if hasattr(self, "inline_section_spin") and self.inline_section_spin.maximum() > 1:
            ix = int(self.inline_section_spin.value()) - 1
    except Exception:
        pass

    try:
        if hasattr(self, "crossline_section_spin") and self.crossline_section_spin.maximum() > 1:
            iy = int(self.crossline_section_spin.value()) - 1
    except Exception:
        pass

    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    return ix, iy


def _g3d_plot_reference_style_section_comparison(self):
    import numpy as _np

    if not getattr(self, "last", None):
        return

    r = self.last
    fig = self.canvas.fig
    fig.clear()

    mig = _np.asarray(r["migrated"])
    cube_in = r.get("cube_migration_input", r.get("cube_premig", r.get("cube_display", r.get("cube"))))
    cube_in = _np.asarray(cube_in)

    x = _np.asarray(r.get("x", _np.arange(mig.shape[1])), dtype=float)
    y = _np.asarray(r.get("y", _np.arange(mig.shape[2])), dtype=float)
    t = _np.asarray(r.get("t", _np.arange(cube_in.shape[0])), dtype=float)
    depth = _np.asarray(r.get("depth", _np.arange(mig.shape[0])), dtype=float)

    ix, iy = _g3d_selected_indices_reference_style(self, r)

    ax = [
        fig.add_subplot(221),
        fig.add_subplot(222),
        fig.add_subplot(223),
        fig.add_subplot(224),
    ]

    panels = [
        (ax[0], cube_in[:, ix, :], y, t,     "Across-survey position [m]", "TWT [ns]",   f"Across-survey vertical section at X={ix+1} — input"),
        (ax[1], mig[:, ix, :],     y, depth, "Across-survey position [m]", "Depth [m]", f"Across-survey vertical section at X={ix+1} — migrated"),
        (ax[2], cube_in[:, :, iy], x, t,     "Along-survey position [m]",  "TWT [ns]",   f"Along-survey vertical section at Y={iy+1} — input"),
        (ax[3], mig[:, :, iy],     x, depth, "Along-survey position [m]",  "Depth [m]", f"Along-survey vertical section at Y={iy+1} — migrated"),
    ]

    # Keep input/migrated panels paired with the same visual scale style.
    clim_inline = _g3d_reference_clip(cube_in[:, ix, :])
    clim_cross = _g3d_reference_clip(cube_in[:, :, iy])
    clims = [clim_inline, clim_inline, clim_cross, clim_cross]

    for i, (a, data2d, lateral, vertical, xlabel, ylabel, title) in enumerate(panels):
        data2d = _np.asarray(data2d, dtype=float)
        im = a.imshow(
            data2d,
            aspect="auto",
            cmap="gray",
            origin="upper",
            extent=[float(lateral[0]), float(lateral[-1]), float(vertical[-1]), float(vertical[0])],
            vmin=-clims[i],
            vmax=clims[i],
        )
        a.set_title(title, fontsize=10)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        fig.colorbar(im, ax=a, shrink=0.75, label="Amplitude")

    fig.suptitle(
        "3-D Stolt migration — before/after sections | "
        f"v={float(r.get('velocity', 0)):.3f} m/ns | "
        f"dx={float(r.get('dx', 0)):.3f}, dy={float(r.get('dy', 0)):.3f} m",
        fontsize=12,
    )

    try:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    except Exception:
        pass

    self.canvas.draw()

    try:
        self._status(
            f"3-D migration section plot: across-survey slice X={ix+1}/{mig.shape[1]}, "
            f"crossline #{iy+1}/{mig.shape[2]}."
        )
    except Exception:
        pass


try:
    if not hasattr(GPR3DMigrationTab, "_orig_plot_result_reference_style_patch"):
        GPR3DMigrationTab._orig_plot_result_reference_style_patch = GPR3DMigrationTab.plot_result

    def _g3d_plot_result_reference_style_patch(self, *args, **kwargs):
        mode = ""
        try:
            mode = self.view_mode.currentText()
        except Exception:
            pass

        if mode in ("", "Section comparison"):
            return _g3d_plot_reference_style_section_comparison(self)

        return GPR3DMigrationTab._orig_plot_result_reference_style_patch(self, *args, **kwargs)

    GPR3DMigrationTab.plot_result = _g3d_plot_result_reference_style_patch
except Exception:
    pass
# --- end reference-style 3-D migration plot template patch ---


# --- 3-D migration colourmap option patch ---
def _g3d_add_section_colourmap_control(self):
    if hasattr(self, "section_cmap"):
        return

    try:
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton
    except Exception:
        return

    self.section_cmap = QComboBox()
    self.section_cmap.addItems(["Blue-red polarity", "Grey amplitude"])
    self.section_cmap.setCurrentText("Blue-red polarity")
    self.section_cmap.setToolTip("Display colour map only; migration result is unchanged.")

    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lab = QLabel("Colour map")
    lab.setMinimumWidth(95)
    lay.addWidget(lab)
    lay.addWidget(self.section_cmap, stretch=1)

    try:
        run_btn = None
        for b in self.findChildren(QPushButton):
            if "Run 3-D" in b.text():
                run_btn = b
                break
        if run_btn is not None and run_btn.parentWidget() is not None and run_btn.parentWidget().layout() is not None:
            parent_lay = run_btn.parentWidget().layout()
            idx = parent_lay.indexOf(run_btn)
            parent_lay.insertWidget(max(0, idx), row)
        else:
            self.layout().addWidget(row)
    except Exception:
        pass

    try:
        self.section_cmap.currentTextChanged.connect(lambda *_: self.plot_result() if getattr(self, "last", None) else None)
    except Exception:
        pass


def _g3d_section_cmap(self):
    try:
        txt = self.section_cmap.currentText().lower()
        if "grey" in txt or "gray" in txt:
            return "gray"
    except Exception:
        pass
    return "seismic"


def _g3d_plot_reference_style_section_comparison(self):
    import numpy as _np

    if not getattr(self, "last", None):
        return

    _g3d_add_section_colourmap_control(self)

    r = self.last
    fig = self.canvas.fig
    fig.clear()

    mig = _np.asarray(r["migrated"])
    cube_in = r.get("cube_migration_input", r.get("cube_premig", r.get("cube_display", r.get("cube"))))
    cube_in = _np.asarray(cube_in)

    x = _np.asarray(r.get("x", _np.arange(mig.shape[1])), dtype=float)
    y = _np.asarray(r.get("y", _np.arange(mig.shape[2])), dtype=float)
    t = _np.asarray(r.get("t", _np.arange(cube_in.shape[0])), dtype=float)
    depth = _np.asarray(r.get("depth", _np.arange(mig.shape[0])), dtype=float)

    ix, iy = _g3d_selected_indices_reference_style(self, r)

    ax = [
        fig.add_subplot(221),
        fig.add_subplot(222),
        fig.add_subplot(223),
        fig.add_subplot(224),
    ]

    panels = [
        (ax[0], cube_in[:, ix, :], y, t,     "Across-survey position [m]", "TWT [ns]",   f"Across-survey vertical section at X={ix+1} — input"),
        (ax[1], mig[:, ix, :],     y, depth, "Across-survey position [m]", "Depth [m]", f"Across-survey vertical section at X={ix+1} — migrated"),
        (ax[2], cube_in[:, :, iy], x, t,     "Along-survey position [m]",  "TWT [ns]",   f"Along-survey vertical section at Y={iy+1} — input"),
        (ax[3], mig[:, :, iy],     x, depth, "Along-survey position [m]",  "Depth [m]", f"Along-survey vertical section at Y={iy+1} — migrated"),
    ]

    clim_inline = _g3d_reference_clip(cube_in[:, ix, :])
    clim_cross = _g3d_reference_clip(cube_in[:, :, iy])
    clims = [clim_inline, clim_inline, clim_cross, clim_cross]
    cmap = _g3d_section_cmap(self)

    for i, (a, data2d, lateral, vertical, xlabel, ylabel, title) in enumerate(panels):
        data2d = _np.asarray(data2d, dtype=float)
        im = a.imshow(
            data2d,
            aspect="auto",
            cmap=cmap,
            origin="upper",
            extent=[float(lateral[0]), float(lateral[-1]), float(vertical[-1]), float(vertical[0])],
            vmin=-clims[i],
            vmax=clims[i],
        )
        a.set_title(title, fontsize=10)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        fig.colorbar(im, ax=a, shrink=0.75, label="Amplitude")

    fig.suptitle(
        "3-D Stolt migration — before/after sections | "
        f"v={float(r.get('velocity', 0)):.3f} m/ns | "
        f"dx={float(r.get('dx', 0)):.3f}, dy={float(r.get('dy', 0)):.3f} m",
        fontsize=12,
    )

    try:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    except Exception:
        pass

    self.canvas.draw()

    try:
        self._status(
            f"3-D migration section plot: across-survey slice X={ix+1}/{mig.shape[1]}, "
            f"along-survey slice Y={iy+1}/{mig.shape[2]}, colour map={self.section_cmap.currentText()}."
        )
    except Exception:
        pass


try:
    if not hasattr(GPR3DMigrationTab, "_orig_init_cmap_option_patch"):
        GPR3DMigrationTab._orig_init_cmap_option_patch = GPR3DMigrationTab.__init__

    def _g3d_init_cmap_option_patch(self, *args, **kwargs):
        GPR3DMigrationTab._orig_init_cmap_option_patch(self, *args, **kwargs)
        _g3d_add_section_colourmap_control(self)

    GPR3DMigrationTab.__init__ = _g3d_init_cmap_option_patch
except Exception:
    pass
# --- end 3-D migration colourmap option patch ---


# --- force-enable 3-D section selectors after migration patch ---
def _g3d_force_enable_section_selectors(self):
    import numpy as _np
    r = getattr(self, "last", None)
    if not r:
        return

    if not getattr(self, "_g3d_section_selector_controls_added", False):
        try:
            _g3d_add_section_selector_controls(self)
        except Exception:
            return

    mig = _np.asarray(r.get("migrated"))
    if mig.ndim != 3:
        return

    nz, nx, ny = mig.shape
    depth = _np.asarray(r.get("depth", []), dtype=float)

    def enable_pair(spin, slider, n):
        n = max(1, int(n))
        cur = int(spin.value()) if spin.maximum() >= 1 and int(spin.value()) >= 1 else (n // 2 + 1)
        cur = max(1, min(cur, n))

        spin.blockSignals(True)
        slider.blockSignals(True)
        try:
            spin.setEnabled(True)
            slider.setEnabled(True)
            try:
                spin.setSpecialValueText("")
            except Exception:
                pass
            spin.setRange(1, n)
            spin.setSuffix(f" / {n}")
            spin.setValue(cur)
            slider.setRange(1, n)
            slider.setValue(cur)
        finally:
            spin.blockSignals(False)
            slider.blockSignals(False)

    try:
        enable_pair(self.inline_section_spin, self.inline_section_slider, nx)
        enable_pair(self.crossline_section_spin, self.crossline_section_slider, ny)
    except Exception:
        pass

    try:
        if depth.size:
            self.depth_index_slider.setEnabled(True)
            self.depth_index_slider.setRange(0, max(0, depth.size - 1))
            iz = int(self.depth_index_slider.value())
            iz = max(0, min(iz, depth.size - 1))
            self.depth_index_slider.setValue(iz)
            self.depth_index_label.setText(f"z={float(depth[iz]):.2f} m")
            self.depth_slice.setEnabled(True)
            self.depth_slice.setRange(float(_np.nanmin(depth)), float(_np.nanmax(depth)))
    except Exception:
        pass


try:
    if not hasattr(GPR3DMigrationTab, "_orig_plot_result_enable_selectors_patch"):
        GPR3DMigrationTab._orig_plot_result_enable_selectors_patch = GPR3DMigrationTab.plot_result

    def _g3d_plot_result_enable_selectors_patch(self, *args, **kwargs):
        _g3d_force_enable_section_selectors(self)
        out = GPR3DMigrationTab._orig_plot_result_enable_selectors_patch(self, *args, **kwargs)
        _g3d_force_enable_section_selectors(self)
        return out

    GPR3DMigrationTab.plot_result = _g3d_plot_result_enable_selectors_patch
except Exception:
    pass
# --- end force-enable 3-D section selectors after migration patch ---


# --- Bulach parallel-line 3-D migration labelling patch ---
# Bulach/PulseEKKO is a dense parallel-line survey, not a crossed inline+crossline survey.
# Keep the 3-D migration tab, but label it as parallel-line/pseudo-3-D and warn that the
# across-line direction is gridded/interpolated from parallel profiles.

_BULACH_PARALLEL_3D_PREV_APPLY = globals().get('apply_bulach')

def _bulach_parallel3d_rename_tabs(self):
    try:
        tabs = getattr(self, 'tabs', None)
        if tabs is None:
            return
        for i in range(tabs.count()):
            txt = tabs.tabText(i).strip().lower().replace('-', '').replace(' ', '')
            if txt in ('3dmigration', '3dmigrationtab') or tabs.tabText(i).strip() == '3D Migration':
                tabs.setTabText(i, 'Parallel-line 3D Migration')
                try:
                    tabs.setTabToolTip(i, 'Pseudo/parallel-line 3-D migration: across-line direction is reconstructed from parallel profiles, not measured perpendicular crosslines.')
                except Exception:
                    pass
                break
    except Exception:
        pass


def _bulach_parallel3d_add_warning_to_widget(self):
    try:
        w = getattr(self, 'gpr3d_migration_tab', None)
        if w is not None and getattr(w, 'kind', '') == 'bulach':
            w._bulach_parallel_line_warning = True
            try:
                w._status('Bulach parallel-line 3-D migration: across-line direction is gridded/interpolated from parallel inline profiles; use 2-D migration for measured single-line QC.')
            except Exception:
                pass
    except Exception:
        pass


def apply_bulach(globs):
    if _BULACH_PARALLEL_3D_PREV_APPLY:
        _BULACH_PARALLEL_3D_PREV_APPLY(globs)

    cls = globs.get('PulseEkko3DAnalysis')
    if cls is not None and not getattr(cls, '_bulach_parallel3d_label_patch_installed', False):
        old_init = cls.__init__
        def new_init(self, *args, _old=old_init, **kwargs):
            _old(self, *args, **kwargs)
            _bulach_parallel3d_rename_tabs(self)
            _bulach_parallel3d_add_warning_to_widget(self)
        cls.__init__ = new_init
        cls._bulach_parallel3d_label_patch_installed = True


try:
    if not hasattr(GPR3DMigrationTab, '_orig_plot_result_bulach_parallel3d_label_patch'):
        GPR3DMigrationTab._orig_plot_result_bulach_parallel3d_label_patch = GPR3DMigrationTab.plot_result

    def _plot_result_bulach_parallel3d_label_patch(self, *args, **kwargs):
        out = GPR3DMigrationTab._orig_plot_result_bulach_parallel3d_label_patch(self, *args, **kwargs)
        try:
            if getattr(self, 'kind', '') == 'bulach':
                fig = self.canvas.fig
                r = getattr(self, 'last', None) or {}
                vel = float(r.get('velocity', self.velocity.value() if hasattr(self, 'velocity') else 0.0))
                dx = float(r.get('dx', self.grid_dx.value() if hasattr(self, 'grid_dx') else 0.0))
                dy = float(r.get('dy', self.grid_dy.value() if hasattr(self, 'grid_dy') else 0.0))
                if getattr(fig, '_suptitle', None) is not None:
                    fig._suptitle.set_text(
                        f'Parallel-line 3-D Stolt migration — before/after sections | v={vel:.3f} m/ns | dx={dx:.3f}, dy={dy:.3f} m'
                    )
                else:
                    fig.suptitle(
                        f'Parallel-line 3-D Stolt migration — before/after sections | v={vel:.3f} m/ns | dx={dx:.3f}, dy={dy:.3f} m',
                        fontsize=12,
                    )
                pass  # removed Bulach footer text
                self.canvas.draw_idle()
                try:
                    self._status('Bulach parallel-line 3-D migration displayed. Treat across-line sections as interpolated, not independently measured crosslines.')
                except Exception:
                    pass
        except Exception:
            pass
        return out

    GPR3DMigrationTab.plot_result = _plot_result_bulach_parallel3d_label_patch
except Exception:
    pass
# --- end Bulach parallel-line 3-D migration labelling patch ---


# --- Bulach parallel-line 3-D time/depth window fix ---
def _g3d_is_bulach_parallel(self):
    k = str(getattr(self, "kind", "")).lower()
    return not ("schleitheim" in k or "mala" in k)

def _g3d_apply_bulach_parallel_defaults(self):
    if not _g3d_is_bulach_parallel(self):
        return
    try:
        self.tmin.setValue(0.0)
        self.tmax.setValue(50.0)
        self.depth_slice.setValue(1.0)
        self.trace_step.setValue(1)
        self.max_lines.setValue(161)
        self.k_filter.setCurrentText("Off")
        self.blank_topo.setChecked(False)
    except Exception:
        pass

def _g3d_crop_bulach_parallel_display(self):
    if not _g3d_is_bulach_parallel(self):
        return
    r = getattr(self, "last", None)
    if not r:
        return
    try:
        tmax = min(float(self.tmax.value()), 50.0)
        vel = float(r.get("velocity", self.velocity.value()))
        zmax = max(0.5, 0.5 * vel * tmax)
        for ax in list(self.canvas.fig.axes):
            ylabel = ax.get_ylabel().lower()
            if "twt" in ylabel or "time" in ylabel:
                ax.set_ylim(tmax, 0.0)
            elif "depth" in ylabel:
                ax.set_ylim(zmax, 0.0)
        self.canvas.draw_idle()
    except Exception:
        pass

try:
    if not hasattr(GPR3DMigrationTab, "_orig_init_bulach_window_fix"):
        GPR3DMigrationTab._orig_init_bulach_window_fix = GPR3DMigrationTab.__init__

    def _g3d_init_bulach_window_fix(self, *args, **kwargs):
        GPR3DMigrationTab._orig_init_bulach_window_fix(self, *args, **kwargs)
        _g3d_apply_bulach_parallel_defaults(self)

    GPR3DMigrationTab.__init__ = _g3d_init_bulach_window_fix
except Exception:
    pass

try:
    if not hasattr(GPR3DMigrationTab, "_orig_run_pipeline_bulach_window_fix"):
        GPR3DMigrationTab._orig_run_pipeline_bulach_window_fix = GPR3DMigrationTab.run_pipeline

    def _g3d_run_pipeline_bulach_window_fix(self, *args, **kwargs):
        _g3d_apply_bulach_parallel_defaults(self)
        out = GPR3DMigrationTab._orig_run_pipeline_bulach_window_fix(self)
        _g3d_crop_bulach_parallel_display(self)
        return out

    GPR3DMigrationTab.run_pipeline = _g3d_run_pipeline_bulach_window_fix
except Exception:
    pass

try:
    if not hasattr(GPR3DMigrationTab, "_orig_plot_result_bulach_window_fix"):
        GPR3DMigrationTab._orig_plot_result_bulach_window_fix = GPR3DMigrationTab.plot_result

    def _g3d_plot_result_bulach_window_fix(self, *args, **kwargs):
        out = GPR3DMigrationTab._orig_plot_result_bulach_window_fix(self)
        _g3d_crop_bulach_parallel_display(self)
        return out

    GPR3DMigrationTab.plot_result = _g3d_plot_result_bulach_window_fix
except Exception:
    pass
# --- end Bulach parallel-line 3-D time/depth window fix ---

