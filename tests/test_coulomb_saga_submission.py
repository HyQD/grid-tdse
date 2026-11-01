import csv
import os
from pathlib import Path
import subprocess

REPOSITORY_ROOT = Path(__file__).parents[1]
JOB_SCRIPT = (
    REPOSITORY_ROOT / "examples" / "Saga" / "coulomb_kernel" / "job_coulomb_kernel.sh"
)
SUBMIT_SCRIPT = JOB_SCRIPT.with_name("submit_coulomb_kernel_benchmarks.sh")


def _write_executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _submission_environment(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sbatch-counter"
    counter.write_text("1000\n", encoding="utf-8")
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
set -euo pipefail
value=$(cat "${FAKE_SBATCH_COUNTER}")
value=$((value + 1))
printf '%s\n' "${value}" > "${FAKE_SBATCH_COUNTER}"
printf '%s\n' "${value}"
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/bin/bash
printf '%s\n' 'mpi_library=fake MPI library'
""",
    )
    project_root = tmp_path / "project"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PROJECT_ID": "test-project",
            "PROJECT_ROOT": str(project_root),
            "SRC_ROOT": str(REPOSITORY_ROOT.parent),
            "GRID_TDSE_MPI_MODULE": "OpenMPI/test",
            "FAKE_SBATCH_COUNTER": str(counter),
        }
    )
    return environment, project_root


def _run_submission(tmp_path, run_name, *extra_arguments):
    environment, project_root = _submission_environment(tmp_path)
    subprocess.run(
        (
            "bash",
            str(SUBMIT_SCRIPT),
            "--stage",
            "threads",
            "--repetitions",
            "1",
            "--tile-size",
            "64",
            "--run-name",
            run_name,
            *extra_arguments,
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return project_root / "jobs" / "grid-tdse" / "coulomb-kernel" / run_name


def test_job_script_has_serial_compute_node_preflight():
    subprocess.run(("bash", "-n", str(JOB_SCRIPT)), check=True)
    contents = JOB_SCRIPT.read_text(encoding="utf-8")

    assert "serial compute-node import and font-cache preflight" in contents
    assert "font_manager.findfont" in contents
    assert contents.index("Preflight imports complete") < contents.index("time srun")


def test_submission_is_dependency_serialized_by_default(tmp_path):
    run_root = _run_submission(tmp_path, "sequential")

    with (run_root / "coulomb_jobs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 6
    assert rows[0]["dependency"] == "none"
    for previous, current in zip(rows, rows[1:]):
        assert current["dependency"] == f"afterany:{previous['job_id']}"
    provenance = (run_root / "provenance.txt").read_text(encoding="utf-8")
    assert "submission_mode=sequential" in provenance


def test_concurrent_option_omits_dependencies(tmp_path):
    run_root = _run_submission(tmp_path, "concurrent", "--concurrent")

    with (run_root / "coulomb_jobs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 6
    assert {row["dependency"] for row in rows} == {"none"}
    provenance = (run_root / "provenance.txt").read_text(encoding="utf-8")
    assert "submission_mode=concurrent" in provenance
