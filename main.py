#!/usr/bin/env python3
from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parent
VENV_PY = BASE / ".venv" / "bin" / "python"

# Re-exec into the project venv unless we are already inside it.
if VENV_PY.exists() and Path(sys.prefix).resolve() != (BASE / ".venv").resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

from GPR_Fieldwork_Analysis import main

if __name__ == "__main__":
    main()
