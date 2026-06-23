# GPR GUI

Interactive desktop application for Ground Penetrating Radar (GPR) quality control, processing, mapping, and 3-D analysis. The app was built for two fieldwork datasets:

- **Schleitheim / MALA**: `.rd7`, `.rad`, `.cor`, `.mrk/.mrkj`, `.tts`, `.proj`
- **Bulach / PulseEKKO**: `.DT1`, `.HD`, GPS/CSV-derived geometry

The main goal is to let a user inspect raw and processed radargrams, compare line directions, detect acquisition artefacts, generate time-slice maps, create amplitude projection maps, export visual products, and run a regular-grid 3-D volume/migration workflow from one GUI.

## What the app does

The GUI provides:

- raw and processed radargram display
- MALA and PulseEKKO project loading
- survey overview / GPS geometry plots
- line-by-line radargram tabs
- selected fence analysis
- time-lapse map animation and GIF export
- amplitude projection maps over a selected time window
- direction amplitude balancing
- bad-line QC and robust high-amplitude line exclusion
- line-bias correction / destriping controls
- map smoothing options
- envelope / absolute / signed amplitude attributes
- regular 3-D volume binning
- spatial k-filtering
- elevation-static correction where elevation is available
- 3-D Stolt migration tab
- synthetic migration validation
- PNG, GIF, NPZ, SEG-Y, MAT, and GeoPackage-style export options where dependencies are available

## Implementation summary

The application is implemented in Python with a PyQt6 graphical interface. Radar data are read from the vendor-specific files, converted into NumPy arrays, processed with signal-processing filters, and visualised using Matplotlib embedded inside Qt widgets.

The 3-D workflow builds a regular survey cube from irregular trace positions, applies optional preprocessing such as k-filtering, elevation statics, live/dead mask tapering, and then runs constant-velocity 3-D Stolt migration. The migration output is displayed as section comparisons, depth-slice views, mask/QC views, and can be exported for later analysis.

## Main files

```text
main.py                         Main entry point
app.py                          Schleitheim / MALA GUI workflow
GPR_Fieldwork_Analysis.py        Bulach / PulseEKKO GUI workflow
stable_time_lapse_patch.py       Time-lapse, amplitude map, QC, smoothing tools
gpr3d_migration.py              3-D volume and Stolt migration workflow
requirements.txt                Python package requirements
Assets/                          Exported figures/GIFs
data/                            GPR datasets
```

## Main packages and libraries

Core packages:

- `PyQt6` for the desktop GUI
- `numpy` for radar arrays and numerical processing
- `scipy` for signal processing, interpolation, filtering, Hilbert/envelope operations, FFT tools
- `matplotlib` for radargram, map, and migration plotting
- `pandas` for metadata/CSV handling
- `Pillow` / `imageio` for image and GIF export

Optional export / geospatial packages:

- `segyio` for SEG-Y export
- `rasterio` for raster/GIS export
- `pyproj` for coordinate handling when needed

## Fresh install and run

### Ubuntu / Debian Linux

Copy and paste this in a terminal:

```bash
sudo apt update && sudo apt install -y git git-lfs python3 python3-venv python3-pip && git lfs install && git clone https://github.com/Darkening-Silhouette/gpr_gui.git && cd gpr_gui && git lfs pull && python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt && python main.py
```

### macOS with Homebrew

Copy and paste this in Terminal:

```bash
brew install git git-lfs python && git lfs install && git clone https://github.com/Darkening-Silhouette/gpr_gui.git && cd gpr_gui && git lfs pull && python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt && python main.py
```

If Homebrew is not installed, install it first from <https://brew.sh>, then rerun the command above.

### Windows 10/11 PowerShell

Open **PowerShell** and run this. It uses `winget` to install Git, Git LFS, and Python if needed:

```powershell
winget install --id Git.Git -e --source winget; winget install --id GitHub.GitLFS -e --source winget; winget install --id Python.Python.3.12 -e --source winget; git lfs install; git clone https://github.com/Darkening-Silhouette/gpr_gui.git; cd gpr_gui; git lfs pull; py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe main.py
```

If `git` or `py` is not recognised immediately after `winget` installs them, close PowerShell, open a new PowerShell window, and rerun the command from `git lfs install` onward:

```powershell
git lfs install; git clone https://github.com/Darkening-Silhouette/gpr_gui.git; cd gpr_gui; git lfs pull; py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe main.py
```

## Run after it is already installed

### Linux / macOS

```bash
cd gpr_gui && git pull && git lfs pull && source .venv/bin/activate && python -m pip install -r requirements.txt && python main.py
```

### Windows PowerShell

```powershell
cd gpr_gui; git pull; git lfs pull; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe main.py
```

## Notes

- Git LFS is required because the radar data files are large binary files.
- The app is intended for desktop use and requires a working graphical environment.
- If optional exporters are unavailable, the main GUI still works; only specific export buttons may be disabled or marked as needing the missing package.
- For scientific interpretation, compare raw, processed, destriped, and migrated views. Do not interpret a feature only from one heavily corrected view.
