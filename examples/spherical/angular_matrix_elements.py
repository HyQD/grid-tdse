import numpy as np
from scipy.integrate import lebedev_rule
from scipy.special import sph_harm_y


def angular_matrix_element(l1, m1, l2, m2, f, order=101):

    """
    Compute a spherical-harmonic matrix element using Lebedev quadrature.

    Evaluates the angular integral

        ∫ Y*_{l1,m1}(Ω) f(θ, φ) Y_{l2,m2}(Ω) dΩ,

    over the unit sphere, where Y_{l,m} are complex spherical harmonics
    using the convention of ``scipy.special.sph_harm_y``.

    The integral is approximated using a Lebedev quadrature rule obtained
    from ``scipy.integrate.lebedev_rule``. The Lebedev Cartesian nodes are
    converted to spherical coordinates according to

        x = sin(θ) cos(φ)
        y = sin(θ) sin(φ)
        z = cos(θ),

    where ``theta`` is the polar (colatitude) angle in [0, π] and ``phi``
    is the azimuthal angle in [0, 2π).

    Parameters
    ----------
    l1 : int
        Orbital angular-momentum quantum number of the bra spherical
        harmonic. Must satisfy ``l1 >= 0``.

    m1 : int
        Magnetic quantum number of the bra spherical harmonic. Must satisfy
        ``-l1 <= m1 <= l1``.

    l2 : int
        Orbital angular-momentum quantum number of the ket spherical
        harmonic. Must satisfy ``l2 >= 0``.

    m2 : int
        Magnetic quantum number of the ket spherical harmonic. Must satisfy
        ``-l2 <= m2 <= l2``.

    f : callable
        Function of the form ``f(theta, phi)`` representing the angular
        function or operator inserted between the two spherical harmonics.
        The function should accept NumPy arrays for ``theta`` and ``phi``
        and return either an array broadcastable to the same shape or a
        scalar. The return value may be real or complex.

    order : int, optional
        Lebedev quadrature order passed to
        ``scipy.integrate.lebedev_rule``. Higher orders use more quadrature
        points and integrate spherical polynomials of higher degree exactly.
        The default is 29.

        If ``f`` has a finite spherical-harmonic expansion containing only
        angular momenta up to ``L_f``, then a sufficient condition for exact
        integration, up to floating-point error, is approximately

            order >= l1 + l2 + L_f.

        The requested order must be one of the orders supported by SciPy.

    Returns
    -------
    value : complex
        Numerical approximation to

            ∫ Y*_{l1,m1}(Ω) f(Ω) Y_{l2,m2}(Ω) dΩ.

        The result may have a negligible imaginary component due to
        floating-point roundoff even when the exact integral is real.

    Notes
    -----
    ``scipy.special.sph_harm_y`` uses ``theta`` for the polar angle and
    ``phi`` for the azimuthal angle. This differs from the convention used
    by the older, deprecated ``scipy.special.sph_harm`` interface.

    The weights returned by ``scipy.integrate.lebedev_rule`` are normalized
    for direct integration over the unit sphere, so no additional factor of
    ``4*pi`` is required.

    Examples
    --------
    For ``f(θ, φ) = 1``, the matrix element reduces to the orthonormality
    relation for spherical harmonics:

    >>> f_unity = lambda theta, phi: np.ones_like(theta)
    >>> angular_matrix_element(2, 1, 2, 1, f_unity)
    (1+0j)

    whereas different angular-momentum states give zero, up to numerical
    roundoff:

    >>> angular_matrix_element(2, 1, 3, 1, f_unity)
    0j

    >>> fx = lambda theta, phi: np.sin(theta) * np.cos(phi)
    >>> fy = lambda theta, phi: np.sin(theta) * np.sin(phi)
    >>> fz = lambda theta, phi: np.cos(theta)
    >>> angular_matrix_element(2, 1, 2, 1, fx)
    >>> angular_matrix_element(2, 1, 2, 1, fy)
    >>> angular_matrix_element(2, 1, 2, 1, fz)

    See Also
    --------
    scipy.integrate.lebedev_rule
        Generate Lebedev quadrature nodes and weights.
    scipy.special.sph_harm_y
        Complex spherical harmonics.
    """

    # Lebedev points and weights
    xyz, w = lebedev_rule(order)
    x, y, z = xyz

    # Cartesian -> spherical
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2*np.pi)

    # Spherical harmonics
    Y1 = sph_harm_y(l1, m1, theta, phi)
    Y2 = sph_harm_y(l2, m2, theta, phi)

    # f should be vectorized: f(theta, phi)
    fv = f(theta, phi)

    return np.sum(w * np.conj(Y1) * fv * Y2)


x_Omega = lambda theta, phi: np.sin(theta) * np.cos(phi)
y_Omega = lambda theta, phi: np.sin(theta) * np.sin(phi)
z_Omega = lambda theta, phi: np.cos(theta)

l_max = 2
for l1 in range(l_max):
    for m1 in range(-l1, l1+1):
        for l2 in range(l_max):
            for m2 in range(-l2, l2+1):
                x_l1l2 = angular_matrix_element(l1, m1, l2, m2, x_Omega)
                y_l1l2 = angular_matrix_element(l1, m1, l2, m2, y_Omega)
                z_l1l2 = angular_matrix_element(l1, m1, l2, m2, z_Omega)
                print(f"<{l1},{m1}|sin(theta)cos(phi)|{l2},{m2}> = {x_l1l2}")
                print(f"<{l1},{m1}|sin(theta)sin(phi)|{l2},{m2}> = {y_l1l2}")
                print(f"<{l1},{m1}|cos(theta)|{l2},{m2}> = {z_l1l2}")
