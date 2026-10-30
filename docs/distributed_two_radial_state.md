# Distributed two-radial state layout

`TwoRadialDistributedLayout` defines the distributed-memory state
representation for non-Born--Oppenheimer diatomic calculations. The
field-free action and physical reductions built on this layout are documented
in [Distributed two-radial Hamiltonian and
reductions](distributed_two_radial_hamiltonian.md).

## Decomposition

Either radial coordinate can be partitioned. For a non-BO diatomic
calculation, coordinate 1 is the electronic coordinate `r` and coordinate 2
is the internuclear coordinate `R`. Coordinate 2 remains the default for
backward compatibility. Every rank retains all angular channels, the complete
unpartitioned radial coordinate, and a contiguous element-aligned interval of
the selected coordinate.

The number of ranks cannot exceed the number of finite elements along the
selected coordinate. Element ranges are kept contiguous and are balanced
approximately by their numbers of owned interior degrees of freedom. The
selected grid must be FEM-DVR because the decomposition relies on
finite-element locality.

At a bridge between two elements, the degree of freedom is permanently owned
by the element on its left, following the ownership convention in `grid-lib`.
If the neighboring element belongs to another rank, that rank lists the bridge
as a halo value. A `TwoRadialRankLayout` records:

- the local element interval;
- permanently owned global indices along the selected coordinate;
- owned-plus-halo indices;
- halo indices and their owning ranks;
- globally shared bridge indices and the owner rank of every distributed
  radial degree of freedom;
- mappings from each local element into the owned-plus-halo work array;
- neighboring ranks.

Only owned values occur in `DistributedTwoRadialState.coefficients`. Halo
values are temporary operator inputs and will be populated by the distributed
Hamiltonian layer. Thus the permanent state has no duplicated bridge values.

## Constructing and inspecting a layout

```python
from grid_tdse.spherical.two_radial import TwoRadialDistributedLayout

layout = TwoRadialDistributedLayout(representation, n_ranks=4)
partition = layout.partition(rank)

print(layout.global_shape)
print(layout.owned_shape(rank))
print(partition.element_start, partition.element_stop)
print(partition.owned_indices)
print(partition.halo_indices)
```

Select coordinate 1 explicitly with:

```python
layout = TwoRadialDistributedLayout(
    representation,
    n_ranks=4,
    partitioned_coordinate=1,
)
```

`layout.partitioned_coordinate`, `layout.partitioned_axis`, and
`layout.partitioned_grid` expose the selected layout. A value of `1` selects
`r`; a value of `2` selects `R`.

`layout.local_shape(rank)` gives the future operator-work shape including
halos, whereas `layout.owned_shape(rank)` gives the permanent state shape.

The pure `layout.distribute(global_state)` and `layout.assemble(local_states)`
operations are useful for validation without MPI. They support both full
angular-product storage and compact fixed-`M_tot` storage.

## MPI collectives

The module does not import or require `mpi4py` at import time. Its collective
helpers accept a communicator implementing the standard `rank`, `size`,
`scatter`, `gather`, and `bcast` interface. An `mpi4py.MPI.Comm` provides this
interface.

All ranks construct an equivalent layout. Only the root supplies the global
state to the scatter:

```python
from mpi4py import MPI
from grid_tdse.spherical.two_radial import (
    TwoRadialDistributedLayout,
    gather_two_radial_state,
    scatter_two_radial_state,
)

comm = MPI.COMM_WORLD
layout = TwoRadialDistributedLayout(representation, comm.size)

local_state = scatter_two_radial_state(
    layout,
    comm,
    psi if comm.rank == 0 else None,
    root=0,
)

psi_again = gather_two_radial_state(local_state, comm, root=0)
```

The current object-based collectives prioritize a clear reference
implementation. Buffer-based MPI collectives can replace them later without
changing the ownership model.

## State archives

Distributed states use the existing self-describing two-radial archive format;
there is no second, incompatible file format. On load, the root validates the
archive against the complete `TwoRadialSystem` and scatters the coefficients.
On save, owned arrays are gathered to the root and passed to
`save_two_radial_state`. The archive's calculation metadata receives a
`distributed_layout` entry containing the selected coordinate, rank count,
and layout identifier. This metadata is informational: a gathered archive can
later be redistributed over a different coordinate or rank count.

```python
from grid_tdse.spherical.two_radial import (
    load_distributed_two_radial_state,
    save_distributed_two_radial_state,
)

local_state = load_distributed_two_radial_state(
    "ground_state.npz",
    layout,
    comm,
)

save_distributed_two_radial_state(
    "propagated_state.npz",
    local_state,
    comm,
    physical_state=physical_state,
    calculation={"time": final_time},
)
```

These initial archive operations gather the complete state on the root. This
preserves compatibility and is sufficient for restart development, but it is
not intended as the final scalable checkpoint implementation.

## Present scope

This layer defines ownership, local and halo ranges, element-local maps,
scatter/gather behavior, and serial archive compatibility. The separate
Hamiltonian layer now supplies halo exchange, the field-free action, and
physical reductions, and the distributed ground-state layer supplies SIL
imaginary-time propagation. The combined distributed implementation does not
yet:

- propagate a state in real time with MPI;
- apply time-dependent fields, CAPs, or masks;
- write sharded or parallel-I/O checkpoints.

Those operations build on this layout in the subsequent MPI implementation
steps.
