import numpy as np
from abc import ABC, abstractmethod

from grid_lib.pseudospectral_grids.setup_grids import setup_grid
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
    number_of_lm_states,
)

class RadialAngularBasis(ABC):

    _ALLOWED_GRID_TYPES = {"radial-sinc", "gll", "femdvr"}

    @abstractmethod
    def __init__(self, grid_config):
        """
        Initialize the shared radial-grid part of a radial-angular basis.

        Parameters
        ----------
        grid_config : dict
            Grid configuration shared across both radial coordinates with separate parameters.
            Required keys:
                - 'grid_type': one of {'radial-sinc', 'gll', 'femdvr'}
                - 'grid_params_r1': parameter dict for the first radial coordinate
                - 'grid_params_r2': parameter dict for the second radial coordinate
        """
        if not isinstance(grid_config, dict):
            raise TypeError("grid_config must be a dict")

        missing = {
            "grid_type",
            "grid_params_r1",
            "grid_params_r2",
        }.difference(grid_config)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(
                "grid_config missing required keys: "
                f"{missing_keys}"
            )

        grid_type = str(grid_config["grid_type"]).lower()
        if grid_type not in self._ALLOWED_GRID_TYPES:
            allowed = ", ".join(sorted(self._ALLOWED_GRID_TYPES))
            raise ValueError(
                f"Unsupported grid_type: {grid_type}. "
                f"Supported types are: {allowed}"
            )

        params_r1 = grid_config["grid_params_r1"]
        params_r2 = grid_config["grid_params_r2"]
        if not isinstance(params_r1, dict) or not isinstance(params_r2, dict):
            raise TypeError("grid_params_r1 and grid_params_r2 must be dicts")

        self.grid_type = grid_type
        self.grid_params_r1 = dict(params_r1)
        self.grid_params_r2 = dict(params_r2)

        self.grid_r1 = setup_grid(self.grid_type, self.grid_params_r1)
        self.grid_r2 = setup_grid(self.grid_type, self.grid_params_r2)

    


class RadialUcYlm(RadialAngularBasis):
    """Radial basis with uncoupled spherical harmonics for each coordinate."""

    def __init__(self, grid_config, l1m1_max, l2m2_max):
        super().__init__(grid_config)
        self.l1_max, self.m1_max = l1m1_max
        self.l2_max, self.m2_max = l2m2_max
        self.n_l1 = self.l1_max + 1
        self.n_l1m1 = number_of_lm_states(self.l1_max, self.m1_max)
        self.n_l2 = self.l2_max + 1
        self.n_l2m2 = number_of_lm_states(self.l2_max, self.m2_max)

class RadialCoupledYlm(RadialAngularBasis):
    """Radial basis with coupled spherical-harmonic angular quantum numbers."""

    def __init__(self, grid_config, L_max, M_max, l1_max, l2_max):
        super().__init__(grid_config)
        self.L_max = L_max
        self.M_max = M_max
        self.l1_max = l1_max
        self.l2_max = l2_max
        self.n_l1 = self.l1_max + 1
        self.n_l2 = self.l2_max + 1
