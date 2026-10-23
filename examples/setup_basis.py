import numpy as np
from grid_tdse.spherical.two_radial.basis import (
    RadialCoupledYlm,
    RadialUcYlm,
)

grid_config_sinc = {
    "grid_type": "radial-sinc",
    "grid_params_r1": {"r_max": 10.0, "N": 100},
    "grid_params_r2": {"r_max": 5.0, "N": 50},
}

grid_config_gll = {
    "grid_type": "gll",
    "grid_params_r1": {"r_max": 10.0, "N": 100},
    "grid_params_r2": {"r_max": 5.0, "N": 50},
}

nodes_r1 = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
n_points_r1 = np.array([10, 10, 10, 10])
nodes_r2 = np.array([0.0, 2.5, 5.0])
n_points_r2 = np.array([10, 10])
grid_config_femdvr = {
    "grid_type": "femdvr",
    "grid_params_r1": {
        "nodes": nodes_r1,
        "n_points": n_points_r1,
        "symmetrize": False,
    },
    "grid_params_r2": {
        "nodes": nodes_r2,
        "n_points": n_points_r2,
        "symmetrize": False,
    },
}

l1m1_max = (2, 2)
l2m2_max = (2, 2)

sinc_basis = RadialUcYlm(grid_config_sinc, l1m1_max, l2m2_max)
gll_basis = RadialUcYlm(grid_config_gll, l1m1_max, l2m2_max)
femdvr_basis = RadialUcYlm(grid_config_femdvr, l1m1_max, l2m2_max)

r1_sinc = sinc_basis.grid_r1
r2_sinc = sinc_basis.grid_r2

r1_gll = gll_basis.grid_r1
r2_gll = gll_basis.grid_r2

r1_femdvr = femdvr_basis.grid_r1
r2_femdvr = femdvr_basis.grid_r2

print(r1_sinc.r)
print(r1_gll.r)
print(r1_femdvr.r)
print()
print(r2_sinc.r)
print(r2_gll.r)
print(r2_femdvr.r)
print()
