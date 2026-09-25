import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
    number_of_lm_states,
    get_y_value,
    get_ybar_value,
)


class CoulombInteraction:
    r"""
    Class representing the Coulomb interaction
        W = sum_{(alpha,q1,q2) \in alpha_q1q2} q1*q2/|r1-alpha*r2|
    """

    def __init__(
        self,
        r1,
        r2,
        alpha_q1q2,
        l1_max,
        m1_max,
        l2_max,
        m2_max,
        L_max,
        M_max,
        radial_representation="quadrature",
        OPT_SWITCH=True
    ):
        """
        Initialize the Coulomb interaction between two particles.

        Parameters
        ----------
        r1 : ndarray
            Radial grid for the first particle.
        r2 : ndarray
            Radial grid for the second particle.
        alpha_q1q2 : list of tuples
            Each tuple contains (alpha, q1, q2) representing the coefficient and charges for the Coulomb interaction.
        L_max : int
            Maximum angular momentum in the multipole expansion.
        radial_representation : str, optional
            Representation of the radial part of the Coulomb interaction:
                1. quadrature (default)
                3. #TODO: Implement Poisson

        """
        self.r1 = r1
        self.r2 = r2
        self.alpha_q1q2 = alpha_q1q2

        self.l1_max = l1_max
        self.m1_max = m1_max
        self.l2_max = l2_max
        self.m2_max = m2_max

        self.n_l1m1 = number_of_lm_states(l1_max, m1_max)
        self.n_l2m2 = number_of_lm_states(l2_max, m2_max)

        self.L_max = L_max
        self.M_max = M_max
        self.n_LM = number_of_lm_states(L_max, M_max)


        self.radial_representation = radial_representation
        if self.radial_representation == "quadrature":
            self.W = sum(
                W_p1p2(r1, r2, a, L_max, q1, q2) for a, q1, q2 in alpha_q1q2
            )
        else:
            raise NotImplementedError(
                "Poisson radial representation is not yet implemented."
            )

        if OPT_SWITCH:
            self.wep_pairs = wep_pairs_ij_direct(
                l1_max,
                m1_max,
                l2_max,
                m2_max,
                L_max,
                M_max,
            )
            

def W_p1p2(r1, r2, alpha, L_max, q1=-1.0, q2=1.0):
    r"""
    Compute the multipole expansion of the Coulomb interaction between two particles with charge q1 and q2.

        q1*q2/|r1 - alpha * r2| = sum_{L=0}^{L_max} sum_{M=-L}^{L}
                                  4 * pi / (2 * L + 1) * sgn(alpha)^L * min(r1, |alpha| * r2)^L / (max(r1, |alpha| * r2)^(L+1))
                                    * Y^*_{LM}(\Omega_1) * Y_{LM}(\Omega_2)

    Parameters
    ----------
    r1 : ndarray
        Radial grid for the first particle.
    r2 : ndarray
        Radial grid for the second particle.
    alpha : float
        Coefficient in the Coulomb interaction 1/|r - alpha * R|.
    L_max : int
        Maximum angular momentum in the multipole expansion.

    Returns
    -------
    W1_rR : ndarray
        Multipole expansion of the Coulomb interaction.
    """
    L_vec = np.arange(L_max + 1)[:, None, None]
    r1_grid = r1[:, None]
    r2_grid = r2[None, :]

    r_min = np.minimum(r1_grid, np.abs(alpha) * r2_grid)[None, :, :]
    r_max = np.maximum(r1_grid, np.abs(alpha) * r2_grid)[None, :, :]

    W1_rR = (
        q1
        * q2
        * (4 * np.pi / (2 * L_vec + 1))
        * (r_min**L_vec)
        / (r_max ** (L_vec + 1))
    )
    return np.sign(alpha) ** L_vec * W1_rR


def _build_lm_state_table(l_max, m_max):
    state_table = {}
    for m in range(-m_max, m_max + 1):
        l_values = np.arange(abs(m), l_max + 1, dtype=int)
        indices = np.array(
            [LM_to_I(l, m, l_max, m_max) for l in l_values], dtype=int
        )
        state_table[m] = (l_values, indices)
    return state_table


def wep_pairs_ij_direct(
    l1_max,
    m1_max,
    l2_max,
    m2_max,
    L_max,
    M_max,
    tol=1e-12,
    max_pairs=int(2e7),
):
    """Build the same pair structure as wep_pairs_ij without y/y_bar tensors.

    Basis (l1_max, m1_max) is used for (I, K) with y_bar coefficients and
    basis (l2_max, m2_max) is used for (J, L) with y coefficients.
    """

    state_1 = _build_lm_state_table(l1_max, m1_max)
    state_2 = _build_lm_state_table(l2_max, m2_max)

    total_pairs = 0
    Ip, Jp, Kp, Lp, Mp, Cp = [], [], [], [], [], []

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), L_max + 1):
            I_LM = LM_to_I(L, M, L_max, M_max)

            ib, kb, yb = [], [], []
            for m_i in range(-m1_max, m1_max + 1):
                m_k = m_i + M
                if m_k < -m1_max or m_k > m1_max:
                    continue

                l_i_values, i_values = state_1[m_i]
                _, k_values = state_1[m_k]

                for l_i, I_idx in zip(l_i_values, i_values):
                    l_k_min = max(abs(m_k), abs(l_i - L))
                    l_k_max = min(l1_max, l_i + L)
                    if l_k_min > l_k_max:
                        continue

                    start = l_k_min - abs(m_k)
                    stop = l_k_max - abs(m_k) + 1
                    for l_k, K_idx in zip(
                        range(l_k_min, l_k_max + 1), k_values[start:stop]
                    ):
                        val = get_ybar_value(L, M, int(l_i), m_i, int(l_k), m_k)
                        if abs(val) > tol:
                            ib.append(I_idx)
                            kb.append(K_idx)
                            yb.append(val)

            jb, lb, yv = [], [], []
            for m_j in range(-m2_max, m2_max + 1):
                m_l = m_j - M
                if m_l < -m2_max or m_l > m2_max:
                    continue

                l_j_values, j_values = state_2[m_j]
                _, l_values = state_2[m_l]

                for l_j, J_idx in zip(l_j_values, j_values):
                    l_l_min = max(abs(m_l), abs(l_j - L))
                    l_l_max = min(l2_max, l_j + L)
                    if l_l_min > l_l_max:
                        continue

                    start = l_l_min - abs(m_l)
                    stop = l_l_max - abs(m_l) + 1
                    for l_l, L_idx in zip(
                        range(l_l_min, l_l_max + 1), l_values[start:stop]
                    ):
                        val = get_y_value(L, M, int(l_j), m_j, int(l_l), m_l)
                        if abs(val) > tol:
                            jb.append(J_idx)
                            lb.append(L_idx)
                            yv.append(val)

            if not ib or not jb:
                continue

            ib = np.asarray(ib, dtype=np.int64)
            kb = np.asarray(kb, dtype=np.int64)
            yb = np.asarray(yb)

            jb = np.asarray(jb, dtype=np.int64)
            lb = np.asarray(lb, dtype=np.int64)
            yv = np.asarray(yv)

            na = ib.size
            nb = jb.size
            total_pairs += na * nb
            if total_pairs > max_pairs:
                raise ValueError(
                    f"{total_pairs} quadruples -- coupling arrays dense?"
                )

            Ip.append(np.repeat(ib, nb))
            Kp.append(np.repeat(kb, nb))
            Jp.append(np.tile(jb, na))
            Lp.append(np.tile(lb, na))
            Mp.append(np.full(na * nb, I_LM, dtype=np.int64))
            Cp.append(np.repeat(yb, nb) * np.tile(yv, na))

    if not Ip:
        z32 = np.zeros(0, dtype=np.int32)
        return (
            z32,
            z32,
            np.asarray([0], dtype=np.int64),
            z32,
            z32,
            z32,
            np.zeros(0),
        )

    Ip, Jp, Kp, Lp, Mp, Cp = (
        np.concatenate(v) for v in (Ip, Jp, Kp, Lp, Mp, Cp)
    )

    order = np.lexsort((Mp, Jp, Ip))
    Ip, Jp, Kp, Lp, Mp, Cp = (a[order] for a in (Ip, Jp, Kp, Lp, Mp, Cp))

    key = Ip * (Jp.max() + 1) + Jp
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    off = np.r_[starts, key.size].astype(np.int64)

    i32 = lambda a: np.ascontiguousarray(a.astype(np.int32))

    return (
        i32(Ip[starts]),
        i32(Jp[starts]),
        np.ascontiguousarray(off),
        i32(Kp),
        i32(Lp),
        i32(Mp),
        np.ascontiguousarray(Cp),
    )
