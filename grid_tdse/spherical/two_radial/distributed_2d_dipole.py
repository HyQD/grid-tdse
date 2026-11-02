r"""Tile-local axial dipole actions on Cartesian two-radial layouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .diatomic import DiatomicIonParameters
from .dipole import TwoRadialAxialDipoleInteraction
from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadialDistributed2DLayout,
)


class TwoRadialDistributed2DLengthGaugeInteraction:
    r"""Apply a z-polarized length-gauge interaction on owned radial tiles.

    The action is

    ``-E_z(t) * (xi_r z_r + xi_R z_R)``.

    Radial multiplication and angular coupling are evaluated entirely in the
    permanently owned Cartesian tile.  No halo exchange or collective
    communication is required.
    """

    def __init__(
        self,
        layout: TwoRadialDistributed2DLayout,
        parameters: DiatomicIonParameters,
        *,
        fields: Callable[[float], float] | Iterable[Callable[[float], float]],
    ) -> None:
        if not isinstance(layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        if not isinstance(parameters, DiatomicIonParameters):
            raise TypeError("parameters must be a DiatomicIonParameters")
        self.layout = layout
        self.representation = layout.representation
        self.parameters = parameters
        self.serial_interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
            self.representation,
            parameters,
            gauge="length",
            fields=fields,
            radial_operator_mode="element-local",
        )
        self.fields = self.serial_interaction.fields
        self.coordinate_operators = self.serial_interaction.coordinate_operators

    def field_amplitude(self, time: float) -> float:
        """Return the summed scalar electric-field amplitude."""

        return self.serial_interaction.field_amplitude(time)

    def __call__(
        self,
        state: DistributedTwoRadial2DState,
        time: float,
    ) -> DistributedTwoRadial2DState:
        """Return the local length-gauge action at ``time``."""

        self._validate_state(state)
        amplitude = self.field_amplitude(time)
        dtype = np.result_type(state.coefficients.dtype, float)
        result = np.zeros(state.coefficients.shape, dtype=dtype)
        if amplitude != 0.0:
            for operator in self.coordinate_operators:
                radial = state.owned_r if operator.coordinate == 1 else state.owned_R
                result += operator.apply_length_gauge_radial_tile(
                    state.coefficients,
                    radial,
                )
            result *= amplitude
        return DistributedTwoRadial2DState(
            self.layout,
            state.rank,
            result,
        )

    def _validate_state(self, state: DistributedTwoRadial2DState) -> None:
        if not isinstance(state, DistributedTwoRadial2DState):
            raise TypeError("state must be a DistributedTwoRadial2DState")
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("state belongs to a different distributed layout")
