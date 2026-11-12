# Quick start for midpoint and two-center Becke H2+

Run commands from the repository root with the existing `haakon_2026`
environment:

```bash
conda run -n haakon_2026 python examples/one_radial_becke_quick_comparison.py
```

The default `quick` preset performs one midpoint-centered calculation and one
two-center Becke calculation for H2+ at `R = 2` bohr.  To run the paired
refinement sequence in one process:

```bash
conda run -n haakon_2026 \
  python examples/one_radial_becke_quick_comparison.py --level all
```

The three levels increase the radial basis, angular cutoff, and integration
quadrature together.  This makes them useful for a fast development loop, but
not for attributing an error to one numerical parameter.  Use the existing
one-parameter-at-a-time study for that:

```bash
conda run -n haakon_2026 \
  python examples/one_radial_becke_convergence_study.py
```

The quick comparison uses the public H2+ validation path because it extends
every atom-centered integration cell far enough to cover the union of the two
finite atomic basis supports.  The midpoint sphere has the same maximum
physical reach, but the two methods still have different finite-domain
shapes.  Their energy difference is therefore a development diagnostic, not
a strict same-domain error estimate.

Useful diagnostics in the table are:

- `N(mid/Becke)`: basis dimensions for the midpoint and Becke calculations;
- `rank`: retained overlap rank at a relative cutoff of `1e-10`;
- `cond(S)`: the untruncated Becke overlap condition number;
- `points`: number of multicenter integration points;
- `time/s`: calculation time after Python imports.

For a clean smoke test of the implementation, run:

```bash
conda run -n haakon_2026 python -m pytest -q \
  tests/test_one_radial_becke_partition.py \
  tests/test_one_radial_becke_quadrature.py \
  tests/test_one_radial_becke_basis.py \
  tests/test_one_radial_becke_assembly.py \
  tests/test_one_radial_becke_ground_state.py \
  tests/test_one_radial_becke_h2plus_validation.py
```
