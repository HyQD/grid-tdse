"""Ground states for one-radial-coordinate spherical systems."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import eigh

from .hamiltonian import OneElectronDiatomicHamiltonian


@dataclass(frozen=True)
class OneElectronDiatomicEigenstates:
    """Lowest eigenstates from one signed-``m`` Hamiltonian block."""

    hamiltonian: OneElectronDiatomicHamiltonian
    m: int
    electronic_energies: np.ndarray
    orbitals: np.ndarray

    @property
    def total_energies(self):
        """Return electronic energies plus clamped nuclear repulsion."""

        return (
            self.electronic_energies
            + self.hamiltonian.system.nuclear_repulsion_energy
        )

    @property
    def electronic_energy(self):
        """Return the lowest requested electronic energy."""

        return float(self.electronic_energies[0])

    @property
    def total_energy(self):
        """Return the lowest requested clamped-nuclei total energy."""

        return float(self.total_energies[0])

    @property
    def orbital(self):
        """Return the lowest requested nodal orbital."""

        return self.orbitals[0]


def diagonalize_fixed_m(hamiltonian, *, m=0, n_states=1):
    """Diagonalize a Euclidean-Hermitian fixed-``m`` Hamiltonian block."""

    if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
        raise TypeError(
            "hamiltonian must be a OneElectronDiatomicHamiltonian"
        )
    shape = hamiltonian.fixed_m_shape(m)
    dimension = int(np.prod(shape))
    if isinstance(n_states, (bool, np.bool_)) or not isinstance(
        n_states, Integral
    ):
        raise TypeError("n_states must be an integer")
    n_states = int(n_states)
    if not 1 <= n_states <= dimension:
        raise ValueError(f"n_states must satisfy 1 <= n_states <= {dimension}")

    matrix = hamiltonian.fixed_m_matrix(m, representation="symmetric")
    energies, vectors = eigh(
        matrix,
        subset_by_index=(0, n_states - 1),
        check_finite=False,
        driver="evr",
    )
    symmetric_orbitals = vectors.T.reshape((n_states, *shape))
    orbitals = hamiltonian.from_symmetric_fixed_m(symmetric_orbitals, m)
    for index in range(n_states):
        norm = np.sqrt(
            hamiltonian.inner_product(orbitals[index], orbitals[index]).real
        )
        orbitals[index] /= norm

    energies = np.asarray(energies)
    orbitals = np.asarray(orbitals)
    energies.setflags(write=False)
    orbitals.setflags(write=False)
    return OneElectronDiatomicEigenstates(
        hamiltonian=hamiltonian,
        m=int(m),
        electronic_energies=energies,
        orbitals=orbitals,
    )
