import numpy as np
from opt_einsum import contract

from numba import njit, prange, get_num_threads

from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
)

# from dipole_fields import DipoleFieldLength


# OPT_SWITCH = True
class Hamiltonian:

    def __init__(
        self,
        h1,
        h2,
        W,
        y,
        y_bar,
        L_max,
        M_max,
        w_r,
        w_R,
        frame="nuclear_center",
        OPT_SWITCH=True,
    ):
        self.h1 = h1
        self.h2 = h2
        self.OPT_SWITCH = OPT_SWITCH
        if self.OPT_SWITCH:
            self.h2T = np.ascontiguousarray(h2.transpose(0, 2, 1))[None]
        self.W = W
        self.y = y
        self.y_bar = y_bar
        self.w_r = w_r
        self.w_R = w_R
        self.L_max = L_max
        self.M_max = M_max
        self.n_LM = y.shape[0]

        if frame == "nuclear_center":
            self.frame = frame
        else:
            raise ValueError(
                f"Unsupported frame: '{frame}'. Currently only 'nuclear_center' is supported."
            )

        if frame == "nuclear_center":

            n_r = len(w_r)
            n_R = len(w_R)

            if self.OPT_SWITCH:
                self.W_over_wR = np.ascontiguousarray(
                    (self.W / w_R[None, None, :]).reshape(self.n_LM, n_r * n_R)
                )
                self.wep_pairs = wep_pairs_ij(self.y, self.y_bar, sign=-1.0)

    def __call__(self, Psi):

        if self.OPT_SWITCH:
            H_Psi = np.matmul(self.h1[:, None, :, :], Psi)
            H_Psi += np.matmul(Psi, self.h2T)
        else:
            H_Psi = contract("Iac, IJcb->IJab", self.h1, Psi)
            H_Psi += contract("Jbc, IJac->IJab", self.h2, Psi)

        if self.frame == "nuclear_center":

            if self.OPT_SWITCH:
                compute_Wep_Psi_Bodenstein(
                    Psi,
                    self.W,
                    self.y,
                    self.y_bar,
                    self.w_R,
                    pairs=self.wep_pairs,  # note: sign=-1.0
                    Wf=self.W_over_wR,
                    out=H_Psi,  # accumulate in place
                )
            else:
                H_Psi -= compute_Wep_Psi_contract(
                    Psi,
                    self.W,
                    self.y,
                    self.y_bar,
                    self.w_R,
                )

        else:
            raise ValueError(f"Unsupported frame: {self.frame}")

        return H_Psi


def compute_Wep_Psi(Psi, W, y, y_bar, L_max, M_max, w_R):

    Wep_Psi = np.zeros_like(Psi)
    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), L_max + 1):
            I_LM = LM_to_I(L, M, L_max, M_max)
            y_bar_LM = y_bar[I_LM]
            y_LM = y[I_LM]
            Wep_Psi += contract(
                "ab,IK,JL,KLab->IJab", W[L], y_bar_LM, y_LM, Psi
            )
    Wep_Psi = contract("IJab,b->IJab", Wep_Psi, 1 / w_R)
    return Wep_Psi


def compute_Wep_Psi_contract(Psi, W, y, y_bar, w_R):
    return contract("Mab,MIK,MJL,KLab,b->IJab", W, y_bar, y, Psi, 1 / w_R)


def wep_pairs_ij(y, y_bar, tol=1e-12, max_pairs=int(2e7), sign=1.0):
    """(I,J)-blocked version of wep_pairs

    Returns (gI, gJ, off, Kp, Lp, Mp, Cp) where group g covers quadruples
    off[g]:off[g+1], all of which write to out[gI[g], gJ[g], :].
    """
    counts = [
        int((np.abs(y_bar[M]) > tol).sum()) * int((np.abs(y[M]) > tol).sum())
        for M in range(y.shape[0])
    ]
    if sum(counts) > max_pairs:
        raise ValueError(f"{sum(counts)} quadruples -- coupling arrays dense?")

    Ip, Jp, Kp, Lp, Mp, Cp = [], [], [], [], [], []
    for M in range(y.shape[0]):
        ib, kb = np.nonzero(np.abs(y_bar[M]) > tol)
        jb, lb = np.nonzero(np.abs(y[M]) > tol)
        if ib.size == 0 or jb.size == 0:
            continue
        na, nb = ib.size, jb.size
        Ip.append(np.repeat(ib, nb))
        Kp.append(np.repeat(kb, nb))
        Jp.append(np.tile(jb, na))
        Lp.append(np.tile(lb, na))
        Mp.append(np.full(na * nb, M, dtype=np.int64))
        Cp.append(
            sign * np.repeat(y_bar[M][ib, kb], nb) * np.tile(y[M][jb, lb], na)
        )

    Ip, Jp, Kp, Lp, Mp, Cp = (
        np.concatenate(v) for v in (Ip, Jp, Kp, Lp, Mp, Cp)
    )

    # primary key (I,J), then M for Wf-row locality within a group
    order = np.lexsort((Mp, Jp, Ip))
    Ip, Jp, Kp, Lp, Mp, Cp = (a[order] for a in (Ip, Jp, Kp, Lp, Mp, Cp))

    key = Ip * (Jp.max() + 1) + Jp
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    off = np.r_[starts, key.size].astype(np.int64)

    # int32 halves index traffic; values are < nl (121) and < nL (231)
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


@njit(cache=True, fastmath=True, parallel=True)
def _wep_kernel_ij(out, P, Wf, gI, gJ, off, Kp, Lp, Mp, Cp, tile):
    """(I,J) blocked version of _wep_kernel.

    prange is over (I,J) groups, which write to disjoint out rows, so the
    parallelism is safe without touching the tile decomposition.  acc is
    per-thread scratch of length tile.
    """
    NT = P.shape[2]
    ng = gI.size
    for g in prange(ng):
        I = gI[g]
        J = gJ[g]
        acc = np.zeros(tile, dtype=out.dtype)
        for s in range(0, NT, tile):
            e = min(s + tile, NT)
            n = e - s
            for t in range(n):
                acc[t] = 0
            for p in range(off[g], off[g + 1]):
                K = Kp[p]
                L = Lp[p]
                M = Mp[p]
                c = Cp[p]
                for t in range(n):
                    acc[t] += c * Wf[M, s + t] * P[K, L, s + t]
            for t in range(n):
                out[I, J, s + t] += acc[t]


def compute_Wep_Psi_Bodenstein(
    Psi, W, y, y_bar, w_R, tile=2048, pairs=None, Wf=None, out=None
):
    """(I,J)-blocked version of compute_Wep_Psi_6."""
    nle = y_bar.shape[1]
    nlp = y.shape[1]

    P = Psi.reshape(nle, nlp, -1)
    NT = P.shape[2]

    if pairs is None:
        pairs = wep_pairs_ij(y, y_bar)
    if Wf is None:
        Wf = np.ascontiguousarray(
            (W / w_R[None, None, :]).reshape(W.shape[0], -1)
        )
    if Wf.shape[1] != NT:
        raise ValueError(
            f"Wf has {Wf.shape[1]} radial points, Psi has {NT} -- "
            "slice them consistently"
        )

    assert Psi.flags["C_CONTIGUOUS"], "Psi must be C-contiguous"

    if out is None:
        out = np.zeros(
            (nle, nlp, NT), dtype=np.result_type(Psi.dtype, Wf.dtype, y.dtype)
        )
    else:
        if out.shape != Psi.shape:
            raise ValueError(f"out {out.shape} != Psi {Psi.shape}")
        out = out.reshape(nle, nlp, NT)

    _wep_kernel_ij(out, P, Wf, *pairs, tile)
    return out.reshape(Psi.shape)
