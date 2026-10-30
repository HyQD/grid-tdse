import json

import pytest

from grid_tdse.benchmarking import (
    THREAD_ENVIRONMENT_VARIABLES,
    default_result_path,
    runtime_environment,
    write_json_exclusive,
)


def test_runtime_environment_records_thread_and_cpu_metadata():
    metadata = runtime_environment(include_build_configuration=False)

    assert metadata["python"]["executable"]
    assert metadata["hostname"]
    assert metadata["packages"]["numpy"]
    assert metadata["packages"]["scipy"]
    assert set(metadata["thread_environment"]) == set(
        THREAD_ENVIRONMENT_VARIABLES
    )
    assert "model" in metadata["cpu"]
    assert "affinity" in metadata["cpu"]
    assert "numpy_build_configuration" not in metadata


def test_default_result_path_uses_slurm_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_NAME", "phase 6 / hybrid")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    path = default_result_path(tmp_path, stem="fallback")

    assert path == tmp_path / "phase_6___hybrid-12345.json"


def test_write_json_exclusive_preserves_existing_results(tmp_path):
    path = tmp_path / "result.json"

    assert write_json_exclusive(path, {"value": 3}) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"value": 4})

    write_json_exclusive(path, {"value": 5}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 5}


@pytest.mark.parametrize("value", [None, ""])
def test_default_result_path_rejects_empty_stem(value, tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        default_result_path(tmp_path, stem=value)
