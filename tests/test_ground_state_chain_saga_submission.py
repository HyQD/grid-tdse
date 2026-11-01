from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).parents[1]
SUBMIT_SCRIPT = (
    REPOSITORY_ROOT
    / "examples"
    / "Saga"
    / "ground_state_chains"
    / "submit_ground_state_chains.sh"
)
JOB_SCRIPT = (
    REPOSITORY_ROOT
    / "examples"
    / "Saga"
    / "ground_state_chains"
    / "job_ground_state_chain.sh"
)
WORKER_SCRIPT = (
    REPOSITORY_ROOT
    / "examples"
    / "mpi_non_bo_diatomic_ground_state_chain.py"
)


def test_job_script_has_serial_compute_node_preflight():
    subprocess.run(("bash", "-n", str(JOB_SCRIPT)), check=True)
    contents = JOB_SCRIPT.read_text(encoding="utf-8")

    assert "serial compute-node import and font-cache preflight" in contents
    assert "import numpy as np" in contents
    assert "import scipy" in contents
    assert "font_manager.findfont" in contents
    assert "import numba" in contents
    assert "import grid_tdse" in contents
    assert contents.index("Preflight imports complete") < contents.index(
        "time srun"
    )


def test_submit_script_exposes_exponential_euler():
    subprocess.run(("bash", "-n", str(SUBMIT_SCRIPT)), check=True)
    completed = subprocess.run(
        (str(SUBMIT_SCRIPT), "--help"),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--integrator NAME" in completed.stdout
    assert "exponential-euler" in completed.stdout
    assert "--time-step VALUE" in completed.stdout
    assert "--max-iterations N" in completed.stdout
    assert "--max-krylov-dimension N" in completed.stdout
    assert "--max-l-max N" in completed.stdout

    contents = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "GRID_TDSE_CHAIN_TIME_STEP_L${lmax}" in contents
    assert "GRID_TDSE_CHAIN_MAX_ITERATIONS_L${lmax}" in contents
    assert 'if (( candidate <= endpoint ))' in contents


def test_chain_worker_reports_exponential_euler_krylov_diagnostics():
    contents = WORKER_SCRIPT.read_text(encoding="utf-8")

    assert "Integrator:" in contents
    assert "Time step:" in contents
    assert "Maximum Krylov dimension:" in contents
    assert "Krylov operator calls:" in contents
    assert "Krylov max dimension:" in contents
    assert "Krylov max error estimate:" in contents
