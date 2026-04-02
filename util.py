import numpy as np
from scipy.stats import unitary_group


def generate_noise_flips(rng, dim, n):
    # This simulate only one ancilla bit
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


def plot_poly(polys, A, square=False):
    import matplotlib.pyplot as plt

    colors = ["#0087c1", "#f6a800"]
    if not isinstance(polys, list):
        polys = [polys]
    xs = np.linspace(-1, 1)
    ys_max = 0
    for i, poly in enumerate(polys):
        if square:
            ys = poly(xs**2)
        else:
            ys = poly(xs)
        ys_max = max(ys_max, np.max(np.abs(ys)))
        plt.plot(xs, ys, color=colors[i])
    plt.scatter(A.D, 1 / A.D, A.b**2 * 100, color="#ad007c")
    plt.ylim([-ys_max - 1, ys_max + 1])
    plt.plot(xs[xs < 0], 1 / xs[xs < 0], "k--", linewidth=0.5)
    plt.plot(xs[xs > 0], 1 / xs[xs > 0], "k--", linewidth=0.5)
    plt.show()
