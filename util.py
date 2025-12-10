import numpy as np
from scipy.stats import ortho_group


def characteristic_poly(A) -> np.polynomial.Chebyshev:
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    def p(x):
        return np.linalg.det(A - np.tensordot(x, np.eye(n), axes=0))

    return np.polynomial.Chebyshev.interpolate(p, n)


def generate_noise_flips(rng, dim, n):
    # Simulate one ancilla bit
    # TODO: Should this be more?
    random_rotations = ortho_group.rvs(dim * 2, n, rng)
    flipped_rotations = np.concatenate(
        (random_rotations[:, dim:, :], random_rotations[:, :dim, :]), axis=1
    )
    return random_rotations.swapaxes(-1, -2) @ flipped_rotations


def sup_norm(poly):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > -1., extrema < 1.)]
    extrema = np.concatenate((extrema, [-1, 1]))

    return np.max(np.abs(poly(extrema)))
