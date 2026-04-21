"""Runtime helpers for the local OpenFace-3.0 checkout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent
OPENFACE3_ROOT = PROJECT_ROOT / "OpenFace-3.0"

OPENFACE3_IMPORT_PATHS = (
    OPENFACE3_ROOT / "Pytorch_Retinaface",
    OPENFACE3_ROOT / "STAR",
    OPENFACE3_ROOT,
)


def ensure_openface3_paths() -> None:
    """Expose OpenFace-3.0 and its vendored dependencies to Python imports."""
    for path in reversed(OPENFACE3_IMPORT_PATHS):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_openface3_interface() -> ModuleType:
    """Load OpenFace-3.0's legacy interface.py module from the local checkout."""
    ensure_openface3_paths()
    module_name = "openface3_interface"
    if module_name in sys.modules:
        return sys.modules[module_name]

    interface_path = OPENFACE3_ROOT / "interface.py"
    spec = importlib.util.spec_from_file_location(module_name, interface_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenFace-3.0 interface from {interface_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
