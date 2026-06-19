# Compatibility wrapper. The 3-D migration implementation is in gpr3d_migration.py.
from gpr3d_migration import *



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
    self.tmin = QDoubleSpinBox(); self.tmin.setRange(0, 5000); self.tmin.setValue(35.0); self.tmin.setSuffix(" ns")
    self.tmax = QDoubleSpinBox(); self.tmax.setRange(0, 5000); self.tmax.setValue(180.0); self.tmax.setSuffix(" ns")
    self.dt_out = QDoubleSpinBox(); self.dt_out.setRange(0.05, 20); self.dt_out.setValue(2.5); self.dt_out.setSingleStep(0.25); self.dt_out.setSuffix(" ns")
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
        ("Taper", self.taper_w), ("Depth slice", self.depth_slice), ("Clip", self.clip_pct),
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

