# Distributed two-radial state layout

`TwoRadialDistributedLayout` defines the first distributed-memory state
representation for non-Born--Oppenheimer diatomic calculations. It is a
storage and communication foundation; distributed Hamiltonian actions are a
separate implementation step.

## Decomposition

The last state axis is the second radial coordinate. For a non-BO diatomic
calculation this is the internuclear coordinate `R`. Every rank retains:

- all angular channels;
- the complete electronic coordinate `r`;
- a contiguous, element-aligned interval of `R`.

The number of ranks cannot exceed the number of `R` finite elements. Element
ranges are kept contiguous and are balanced approximately by their numbers of
owned interior degrees of freedom. Radial sinc-DVR grids are rejected because
this first decomposition relies on FEM-DVR locality.

At a bridge between two elements, the degree of freedom is permanently owned
by the element on its left, following the ownership convention in `grid-lib`.
If the neighboring element belongs to another rank, that rank lists the bridge
as a halo value. A `TwoRadialRankLayout` records:

- the local element interval;
- permanently owned global `R` indices;
- owned-plus-halo indices;
- halo indices and their owning ranks;
- globally shared bridge indices and the owner rank of every `R` degree of
  freedom;
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
`save_two_radial_state`.

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
scatter/gather behavior, and serial archive compatibility. It does not yet:

- exchange halo values;
- apply a distributed Hamiltonian;
- perform distributed inner products or observable reductions;
- propagate a state with MPI;
- write sharded or parallel-I/O checkpoints.

Those operations build on this layout in the subsequent MPI implementation
steps.
