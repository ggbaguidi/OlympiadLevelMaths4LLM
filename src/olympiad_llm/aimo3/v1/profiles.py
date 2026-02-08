from __future__ import annotations

"""Config profiles for AIMO-3.

These helpers are designed for notebook usage (e.g., Kaggle) where setting many
environment variables is tedious. They are intentionally conservative:

- By default, they do NOT override variables that are already set.
- They just populate a recommended bundle for a given profile.

Note: The core solver also supports `AIMO3_PROFILE=lean` directly in
`AIMO3Config.from_env()`. This module is a convenience layer when you want the
profile to set additional tool/sandbox knobs too.
"""

from collections.abc import MutableMapping


def profile_env(profile: str) -> dict[str, str]:
    p = (profile or "").strip().lower()
    if p in {"", "default", "full"}:
        return {}
    if p != "lean":
        raise ValueError(f"Unknown profile: {profile}")

    # Lean: fewer attempts + fewer turns + reduced extra orchestration.
    # Keep timeouts conservative; let the solver retry-on-timeout logic work.
    return {
        "AIMO3_PROFILE": "lean",
        # Tool/sandbox knobs that help in notebook runtimes:
        "AIMO3_SANDBOX_POOL_SIZE": "2",
        "AIMO3_KERNEL_INIT_WORKERS": "2",
        # Encourage faster, smaller python usage:
        "AIMO3_PYTHON_TOOL_TIMEOUT_CAP_S": "120",
        "AIMO3_PYTHON_TOOL_TIMEOUT_RETRY_ENABLED": "1",
        "AIMO3_PYTHON_TOOL_TIMEOUT_RETRY_MULT": "2.0",
        # Avoid long tie-breaks by default (profile already disables unless explicitly set):
        "AIMO3_TIEBREAK_ENABLED": "0",
    }


def apply_profile(
    profile: str,
    *,
    env: MutableMapping[str, str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Apply a profile to an environment mapping.

    Returns the dict of keys that were set.
    """

    if env is None:
        import os

        env = os.environ

    desired = profile_env(profile)
    applied: dict[str, str] = {}
    for k, v in desired.items():
        if force or (k not in env) or (not str(env.get(k, "")).strip()):
            env[k] = str(v)
            applied[k] = str(v)
    return applied
