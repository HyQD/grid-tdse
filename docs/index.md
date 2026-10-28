# grid-tdse documentation

`grid-tdse` is a direct grid-based time-dependent Schrödinger equation
package for few-body systems. The project is under active development, and
feature documentation will be added when each solver is implemented and
validated.

The documentation currently covers the project setup and the first numerical
foundation:

- [Runtime dependencies](runtime_dependencies.md)
- [Installation](installation.md)
- [One-electron diatomic ground-state validation](one_electron_diatomic_validation.md)
- [Saga environment setup and updates](saga_setup.md)
- [Two-radial spherical foundation](two_radial_foundation.md)
- [Imaginary-time ground-state iteration](ground_state_iteration.md)
- [Non-Born--Oppenheimer diatomic-ion ground states](non_bo_diatomic_ground_state.md)
- [Non-BO ground-state integrator comparison](non_bo_diatomic_ground_state_integrator_benchmark.md)
- [Two-electron atomic ground states](two_electron_atomic_ground_state.md)
- [Two-radial state archives](two_radial_state_archives.md)
- [Distributed two-radial state layout](distributed_two_radial_state.md)
- [Distributed two-radial Hamiltonian and reductions](distributed_two_radial_hamiltonian.md)
- [Distributed non-BO diatomic ground states](distributed_ground_state.md)
- [Axial two-radial dipole interactions](two_radial_dipole_interaction.md)
- [Real-time propagation](real_time_propagation.md)

Runnable calculations are kept in the [`examples`](../examples) directory.
