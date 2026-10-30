"""Reproducible runtime metadata for performance benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import numpy as np
import scipy


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

SLURM_ENVIRONMENT_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND_TYPE",
    "SLURM_DISTRIBUTION",
)


def runtime_environment(*, include_build_configuration: bool = True) -> dict:
    """Return JSON-compatible process, threading, and build metadata."""

    if not isinstance(include_build_configuration, bool):
        raise TypeError("include_build_configuration must be boolean")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "cpu": {
            "model": _cpu_model(),
            "logical_count": os.cpu_count(),
            "affinity": _cpu_affinity(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "thread_environment": _selected_environment(
            THREAD_ENVIRONMENT_VARIABLES
        ),
        "thread_pools": _thread_pool_information(),
        "slurm": _selected_environment(SLURM_ENVIRONMENT_VARIABLES),
        "grid_tdse_commit": _git_revision(),
    }
    if include_build_configuration:
        result["numpy_build_configuration"] = np.show_config(mode="dicts")
        result["scipy_build_configuration"] = scipy.show_config(mode="dicts")
    return result


def write_json_exclusive(path, payload, *, overwrite: bool = False) -> Path:
    """Write indented JSON without silently replacing an existing result."""

    import json

    destination = Path(path)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def default_result_path(directory, *, stem: str) -> Path:
    """Return a Slurm-aware result path below ``directory``."""

    if not isinstance(stem, str) or not stem:
        raise ValueError("stem must be a nonempty string")
    job_name = os.environ.get("SLURM_JOB_NAME", stem)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    safe_name = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in job_name
    )
    return Path(directory) / f"{safe_name}-{job_id}.json"


def _selected_environment(names) -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in names}


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    value = platform.processor().strip()
    return value or None


def _thread_pool_information() -> list[dict] | dict:
    if importlib.util.find_spec("threadpoolctl") is None:
        return {"available": False}
    from threadpoolctl import threadpool_info

    return threadpool_info()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
