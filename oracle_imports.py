"""Simple local imports for notebook-derived oracle functions.

Use this when working directly inside `activation_oracles_extensions` without
treating it as an installed package.

Example:
    from oracle_imports import run_oracle_single_layer, run_oracle_multi_layer
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_HERE = Path(__file__).resolve().parent
_MATERIALIZED = _HERE / "materialized"


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing materialized module: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_single_mod = _load_module(
    "activation_oracle_demo_lib_local",
    _MATERIALIZED / "activation_oracle_demo_lib.py",
)
_multi_mod = _load_module(
    "multilayer_activation_oracle_demo_lib_local",
    _MATERIALIZED / "multilayer_activation_oracle_demo_lib.py",
)


# Public aliases with clear names.
run_oracle_single_layer = _single_mod.run_oracle
run_oracle_multi_layer = _multi_mod.run_oracle
visualize_token_selection = _multi_mod.visualize_token_selection

