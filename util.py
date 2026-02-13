import numpy as np
from scipy.stats import ortho_group


def generate_noise_flips(rng, dim, n):
    # Simulate one ancilla bit
    # TODO: Should this be more?
    random_rotations = ortho_group.rvs(dim * 2, n, rng)
    flipped_rotations = np.concatenate(
        (random_rotations[:, dim:, :], random_rotations[:, :dim, :]), axis=1
    )
    return random_rotations.swapaxes(-1, -2) @ flipped_rotations


def sup_norm(poly, range=(-1.,1.)):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > range[0], extrema < range[1])]
    extrema = np.concatenate((extrema, [range[0], range[1]]))

    return np.max(np.abs(poly(extrema)))


def maximal_x(poly, a, b):
    """
    Computes the point with the largest absolute value in [a, b]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > a, extrema < b)]
    extrema = np.concatenate((extrema, [a, b]))

    return extrema[np.argmax(np.abs(poly(extrema)))]
