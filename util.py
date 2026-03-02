import numpy as np
from scipy.stats import unitary_group


def generate_noise_flips(rng, dim, n):
    # Simulate one ancilla bit
    # TODO: Should this be more?
    random_rotations = unitary_group.rvs(dim * 2, n, rng).astype(np.complex64)
    flipped_rotations = np.concatenate(
        (random_rotations[:, dim:, :], random_rotations[:, :dim, :]), axis=1
    )
    return np.conj(random_rotations.swapaxes(-1, -2)) @ flipped_rotations


def sup_norm(poly, range=(-1.0, 1.0)):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > range[0], extrema < range[1])]
    extrema = np.concatenate((extrema, [range[0], range[1]]))

    return np.max(np.abs(poly(extrema)))


def measurement_max_error(reference, approximation):
    """
    Find the quantity of interest that maximizes the error
    """

    reference_norm = np.linalg.norm(reference)

    if (
        np.dot(approximation, reference)
        / reference_norm
        / np.linalg.norm(approximation)
        < 1e-6
    ):
        # The error is in the direction of the exact_solution
        return np.outer(reference, reference) / reference_norm**2
    else:
        # Find the two dimensional subspace that contains both exact solution and approximation
        transform_basis, _ = np.linalg.qr(np.array([reference, approximation]).T)
        error_matrix = (
            transform_basis.T
            @ (np.outer(reference, reference) - np.outer(approximation, approximation))
            @ transform_basis
        )
        eigval, eigvec = np.linalg.eig(error_matrix)
        return (
            transform_basis
            @ eigvec
            @ np.diag(np.sign(eigval))
            @ eigvec.T
            @ transform_basis.T
        )
