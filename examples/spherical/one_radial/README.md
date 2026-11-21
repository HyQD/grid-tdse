# One-radial spherical examples

This directory contains the one-electron diatomic validation and convergence
calculations associated with `grid_tdse.spherical.one_radial`.

`tdse_h_atom_femdvr.py` is retained as an older standalone hydrogen FEM-DVR
example pending a separate review.

`h2plus_laser_dynamics.py` is the public-interface, full-angular development
example for clamped-nuclei H2+ laser dynamics at an arbitrary molecular
orientation. Its default field is a laboratory-`z`-polarized, three-cycle,
400 nm `SineSquareLaser` with `E0=0.05` and phase zero. These are development
inputs rather than convergence recommendations. For example:

```bash
python examples/spherical/one_radial/h2plus_laser_dynamics.py \
    --gauge velocity \
    --integrator sia \
    --absorber cap \
    --wavelength-nm 400 \
    --field-strength 0.05 \
    --r-max 64 \
    --l-max 12 \
    --time-step 0.03125 \
    --cap-alpha 0.8 \
    --cap-eta 1.0
```

Add `--theta-deg 45 --phi-deg 30` to orient the bond, or supply both Cartesian
nuclear positions with `--nuclear-position-1` and `--nuclear-position-2`.
The program independently computes a full-angular Davidson state for the
chosen geometry, writes a JSON summary and compressed NPZ archive, and
reports the effective geometry, grid, pulse, state source, propagation
settings, norms, and output locations.
