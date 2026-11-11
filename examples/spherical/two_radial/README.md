# Two-radial spherical examples

This directory contains atomic and non-Born--Oppenheimer diatomic examples
associated with `grid_tdse.spherical.two_radial`, including serial and MPI
ground-state and real-time calculations.

- [`Saga`](Saga/) contains deployment and benchmark submission scripts.
- [`ground_states`](ground_states/) contains reference state archives used by
  selected high-level examples.
- [`davidson_ground_states`](davidson_ground_states/) is the default output
  location for the Davidson reference example.
- [`coupled_gs_nbo_diatomics.py`](coupled_gs_nbo_diatomics.py) validates the
  serial coupled-angular Davidson path for H2+, HD+, and D2+ on fixed moderate
  grids; its energies are not physical convergence results.
