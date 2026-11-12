"""Structural checks for Olivia qualification drivers."""

from pathlib import Path
import subprocess

QUALIFICATION_ROOT = (
    Path(__file__).parents[1]
    / "examples"
    / "spherical"
    / "two_radial"
    / "Olivia"
    / "qualification"
)
MULTI_NODE_SCRIPT = QUALIFICATION_ROOT / "qualify_grid_tdse_multi_node.sh"
COUPLED_STEP16_SCRIPT = QUALIFICATION_ROOT / "qualify_grid_tdse_coupled_step16.sh"
COUPLED_STEP16_DRIVER = QUALIFICATION_ROOT / "qualify_coupled_step16.py"


def test_multi_node_mpi_probe_runs_inside_srun():
    source = MULTI_NODE_SCRIPT.read_text(encoding="utf-8")
    prelaunch, separator, placement_launch = source.partition("\n    srun \\")

    assert separator
    assert "from mpi4py import MPI" not in prelaunch
    assert "MPI.Get_library_version()" in placement_launch
    assert "communicator.rank == 0 and communicator.size == 2" in placement_launch


def test_all_olivia_shell_drivers_source_deployment_runtime_environment():
    for path in sorted(QUALIFICATION_ROOT.glob("qualify_grid_tdse*.sh")):
        source = path.read_text(encoding="utf-8")
        module_position = source.index('module load "${GRID_TDSE_MPI_MODULE}"')
        runtime_position = source.index('source "${runtime_environment}"')
        path_position = source.index('export PATH="${deployment_prefix}/bin:${PATH}"')

        assert module_position < runtime_position < path_position, path.name
        assert 'runtime_environment="${manifest_root}/runtime-environment.sh"' in source
        assert 'test -s "${runtime_environment}"' in source


def test_all_olivia_shell_drivers_have_valid_bash_syntax():
    for path in sorted(QUALIFICATION_ROOT.glob("qualify_grid_tdse*.sh")):
        subprocess.run(["bash", "-n", path], check=True)


def test_large_partition_drivers_do_not_request_memory():
    for path in sorted(QUALIFICATION_ROOT.glob("qualify_grid_tdse*.sh")):
        source = path.read_text(encoding="utf-8")
        if "#SBATCH --partition=large" in source:
            assert "#SBATCH --mem" not in source, path.name


def test_coupled_step16_driver_is_launched_only_under_srun():
    source = COUPLED_STEP16_SCRIPT.read_text(encoding="utf-8")
    prelaunch, separator, launch = source.partition("\n    srun \\")

    assert separator
    assert 'python "${driver}"' not in prelaunch
    assert 'python "${driver}"' in launch
    assert "run_case reference-1r 1 1 1 1 1 --write-reference" in source
    assert "run_case one-node-4r 1 4 4 2 2" in source
    assert "run_case two-node-16r" not in source
    assert "Coupled Step 16 preparation: PASS" in source


def test_coupled_step16_support_driver_uses_slurm_submission_directory():
    source = COUPLED_STEP16_SCRIPT.read_text(encoding="utf-8")

    assert (
        'script_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"'
        in source
    )
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in source
    assert 'driver="${script_root}/qualify_coupled_step16.py"' in source


def test_coupled_step16_preparation_uses_small_partition_and_shared_memory():
    source = COUPLED_STEP16_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --partition=small" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --ntasks=4" in source
    assert "#SBATCH --time=00:10:00" in source
    assert "export OMPI_MCA_pml=ob1" in source
    assert "export OMPI_MCA_btl=sm,self" in source
    assert "export OMPI_MCA_pml=cm" not in source
    assert "export FI_PROVIDER=cxi" not in source
    assert "unset FI_PROVIDER" in source
    assert "unset FI_CXI_RX_MATCH_MODE" in source


def test_only_combined_multi_node_driver_uses_large_partition():
    large_drivers = []
    for path in sorted(QUALIFICATION_ROOT.glob("qualify_grid_tdse*.sh")):
        source = path.read_text(encoding="utf-8")
        if "#SBATCH --partition=large" in source:
            large_drivers.append(path.name)

    assert large_drivers == ["qualify_grid_tdse_multi_node.sh"]


def test_multi_node_driver_reuses_preparation_and_runs_representative_layouts():
    source = MULTI_NODE_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --time=00:10:00" in source
    assert "GRID_TDSE_PREPARATION_JOB_ID" in source
    assert 'script_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"' in source
    assert 'coupled_driver="${script_root}/qualify_coupled_step16.py"' in source
    assert 'python "${coupled_driver}"' in source
    assert "--process-grid 4 4" in source
    assert "Coupled Step 16 Olivia qualification: PASS" in source
    assert "ranks=(2 16)" in source
    assert "tasks_per_node=(1 8)" in source
    assert "process_grids=(1x2 4x4)" in source
    assert "--initial-state-policy require" in source
    assert "ranks=(2 4 8 16)" not in source
    assert "export OMPI_MCA_pml=cm" in source
    assert "export OMPI_MCA_mtl=ofi" in source
    assert "export FI_PROVIDER=cxi" in source


def test_coupled_step16_python_driver_has_expected_gates():
    source = COUPLED_STEP16_DRIVER.read_text(encoding="utf-8")

    compile(source, str(COUPLED_STEP16_DRIVER), "exec")
    assert "wf_representation" in source
    assert "replicated_angular_inventory" in source
    assert "TwoRadialAbsorbingBoundaries" in source
    assert '"cap_norm_losses"' in source
    assert '"total_dipoles_z"' in source
    assert '"total_l_squared"' in source
    assert "load_distributed_two_radial_2d_state" in source
    assert "save_distributed_two_radial_2d_state" in source
    assert '"restart_difference_from_uninterrupted"' in source
