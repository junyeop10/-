"""Double-click GUI entry point for the file classifier."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_tk_runtime() -> None:
    """Help tkinter locate Tcl/Tk assets when launched from a venv."""
    base_prefix = Path(sys.base_prefix)
    tcl_dir = base_prefix / "tcl"
    dll_dir = base_prefix / "DLLs"
    tcl_library = tcl_dir / "tcl8.6"
    tk_library = tcl_dir / "tk8.6"

    if tcl_library.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
    if tk_library.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk_library))
    if dll_dir.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))


configure_tk_runtime()

from src.gui import main


if __name__ == "__main__":
    main()
