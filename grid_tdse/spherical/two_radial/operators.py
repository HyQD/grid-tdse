import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
    number_of_lm_states,
)

class KineticEnergy:

    def __init__(self, mu, r, D2, l_max, m_max, coord_axis=0):
        self.mu = mu
        self.coord_axis = coord_axis

        self.T = np.zeros((number_of_lm_states(l_max, m_max), r.size, r.size))
        for m in range(-m_max, m_max + 1):
            for l in range(abs(m), l_max + 1):
                I_lm = LM_to_I(l, m, l_max, m_max)
                self.T[I_lm] = -1 / (2 * mu) * (D2 - np.diag(l * (l + 1) / r**2))

        # Matches the right-action layout used in Hamiltonian.__call__ (OPT_SWITCH=True).
        self.TT = np.ascontiguousarray(self.T.transpose(0, 2, 1))[None]

    def __call__(self, Psi):

        """
        Apply the kinetic energy operator to the wavefunction Psi.

        Parameters
        ----------
        Psi : ndarray
            The wavefunction array with shape (n_lm, n_lm, n_r1, n_r2).

        Returns
        -------
        ndarray
            The result of applying the kinetic energy operator to Psi along the specified coordinate axis.
        """

        if Psi.ndim != 4:
            raise ValueError(
                "Psi must have shape (n_lm1, n_lm2, n_r1, n_r2)."
            )

        n_lm = self.T.shape[0]
        if self.coord_axis == 0:
            if Psi.shape[0] != n_lm:
                raise ValueError(
                    "For coord_axis=0, Psi.shape[0] must match kinetic "
                    f"operator size {n_lm}. Got {Psi.shape[0]}."
                )
            return np.matmul(self.T[:, None, :, :], Psi)
        elif self.coord_axis == 1:
            if Psi.shape[1] != n_lm:
                raise ValueError(
                    "For coord_axis=1, Psi.shape[1] must match kinetic "
                    f"operator size {n_lm}. Got {Psi.shape[1]}."
                )
            return np.matmul(Psi, self.TT)
        else:
            raise ValueError("Invalid coordinate axis.")
