from __future__ import annotations

"""Convenience helpers to run the AIMO-3 solver locally or under Kaggle evaluation."""

from typing import Optional

from .config import AIMO3Config
from .errors import OptionalDependencyError
from .solver import AIMO3Solver


def build_solver(cfg: Optional[AIMO3Config] = None) -> AIMO3Solver:
    """Create a solver using env defaults (model path, served model name)."""
    cfg = cfg or AIMO3Config.from_env()
    return AIMO3Solver(cfg)


def run_kaggle_inference(predict_fn):
    """Run the Kaggle AIMO-3 inference server if available.

    This is intentionally a thin wrapper so importing this module doesn't require Kaggle.
    """

    try:
        import kaggle_evaluation.aimo_3_inference_server as aimo_server  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "Kaggle inference server package 'kaggle_evaluation' not found. This is expected outside Kaggle."
        ) from e

    server = aimo_server.AIMO3InferenceServer(predict_fn)
    return server
