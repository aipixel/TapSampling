from __future__ import annotations

import sys
from functools import lru_cache
from importlib import util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAPSAMPLING_ROOT = PROJECT_ROOT.parent / "tapsampling"


@lru_cache(maxsize=1)
def load_dual_system_calvin_evaluation_sp():
    if TAPSAMPLING_ROOT.as_posix() not in sys.path:
        sys.path.append(TAPSAMPLING_ROOT.as_posix())

    file_path = TAPSAMPLING_ROOT / "vla-scripts" / "vla_evaluation.py"
    module_name = "tapsampling_vla_evaluation"
    spec = util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError("can't create import spec for " + str(file_path))

    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        return getattr(module, "DualSystemCalvinEvaluation_SP")
    except AttributeError as exc:
        raise ImportError("DualSystemCalvinEvaluation_SP not found in " + str(file_path)) from exc


DualSystemCalvinEvaluation_SP = load_dual_system_calvin_evaluation_sp()
