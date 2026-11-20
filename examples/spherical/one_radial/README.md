# One-radial spherical examples

This directory contains the one-electron diatomic validation and convergence
calculations associated with `grid_tdse.spherical.one_radial`.

`tdse_h_atom_femdvr.py` is retained as an older standalone hydrogen FEM-DVR
example pending a separate review.

`h2plus_laser_dynamics.py` is the public-interface development example for
clamped-nuclei H2+ laser dynamics. Its defaults are intentionally small smoke
inputs. For example:

```bash
python examples/spherical/one_radial/h2plus_laser_dynamics.py \
    --gauge velocity \
    --integrator sia \
    --absorber cap \
    --wavelength-nm 800 \
    --field-strength 0.05 \
    --r-max 64 \
    --l-max 12 \
    --time-step 0.03125 \
    --cap-alpha 0.8 \
    --cap-eta 1.0
```

That command is illustrative rather than converged. The program writes a
JSON summary and compressed NPZ archive and reports the effective grid,
pulse, state source, propagation settings, norms, and output locations.
