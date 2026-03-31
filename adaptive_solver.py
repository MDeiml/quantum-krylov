from functools import cache

import numpy as np
import scipy as sp

from block_encoding_model import BlockEncodingModel


# These functions are cached so that they only execute once
# for each combination of parameters
@cache
def lagrange_basis(deg, transform):
    X = np.polynomial.Chebyshev([0, 1])

    domain = [-1, 1]
    if transform == "square":
        domain = [0, 1]

    if transform == "square_outer":
        assert deg % 2 == 1

    N = deg + 1

    # Polynomial with roots at all interpolation points
    full_poly = np.polynomial.Chebyshev((N - 1) * [0] + [1]).deriv() * (X - 1) * (X + 1)

    basis = []
    for k in range(deg + 1):
        # k-th chebyshev node of the second kind
        xk = np.cos(k / (N - 1) * np.pi)
        poly = full_poly // (X - xk)
        poly /= poly(xk)
        if transform == "square_outer":
            poly -= poly(-X)
        basis.append(np.polynomial.Chebyshev(poly.coef, domain=domain))

    return basis


@cache
def moment_to_gram(steps, square):
    max_degree = 2 * steps + 1

    X = np.polynomial.Chebyshev([0, 1])
    if square:
        X = X * X

    mat_E = np.zeros((steps + 1, steps + 1, max_degree + 1))
    mat_G = np.zeros((steps + 1, steps + 1, max_degree + 1))
    for i in range(steps + 1):
        if square:
            poly_i = np.polynomial.Chebyshev([0] * (2 * i) + [1])
        else:
            poly_i = np.polynomial.Chebyshev([0] * i + [1])

        for j in range(steps + 1):
            if square:
                poly_j = np.polynomial.Chebyshev([0] * (2 * j) + [1])
            else:
                poly_j = np.polynomial.Chebyshev([0] * j + [1])
            poly_E = poly_i * poly_j
            poly_G = X * poly_E
            if square:
                # Moments are only even degrees, other coefficients of poly_E
                # and poly_G will be zero anyways
                mat_E[i, j, : poly_E.degree() // 2 + 1] = poly_E.coef[::2]
                mat_G[i, j, : poly_G.degree() // 2 + 1] = poly_G.coef[::2]
            else:
                mat_E[i, j, : poly_E.degree() + 1] = poly_E.coef
                mat_G[i, j, : poly_G.degree() + 1] = poly_G.coef

    return mat_E, mat_G


def compute_moments(A: BlockEncodingModel, steps, samples, square):
    """
    Estimate moments of the form b^T p(A) b (if qoi = False)
    or m^T p(A) b (if qoi = True)
    """

    max_degree = 2 * steps + 1

    moments = np.zeros(max_degree + 1)
    for i in range(max_degree + 1):
        if square:
            poly = np.polynomial.Chebyshev([0] * (2 * i) + [1])
        else:
            poly = np.polynomial.Chebyshev([0] * i + [1])
        moments[i] = A.estimate_poly(
            poly,
            samples,
            root=square,
        )

    return moments


def estimate_eigenvalues(A: BlockEncodingModel, steps, samples, square, use_kappa):
    moments = compute_moments(A, steps, samples, square)

    # Compute lhs matrix and rhs vector
    mat_E, mat_G = moment_to_gram(steps, square)
    E = mat_E @ moments
    G = mat_G @ moments

    # First remove directions corresponding to small eigenvalues of G
    accuracy = 2 / np.sqrt(samples)
    eigenvalues_E, eigenvectors_E = sp.linalg.eigh(E)
    eval_max = eigenvalues_E[-1]
    cutoff_index = np.searchsorted(eigenvalues_E, eval_max * accuracy)
    min_allowed_ev = accuracy
    if use_kappa:
        min_allowed_ev = 1 / A.kappa

    # Then, if there are still negative eigenvalues, continuously remove the smallest eigenvalue
    while True:
        P = (
            np.diag(1 / np.sqrt(eigenvalues_E[cutoff_index:]))
            @ eigenvectors_E[:, cutoff_index:].T
        )
        G_sub = P @ G @ P.T
        eigenvalues, eigenvectors = sp.linalg.eigh(G_sub)
        if (
            np.min(eigenvalues) >= min_allowed_ev
            or cutoff_index == len(eigenvalues_E) - 1
        ):
            break
        cutoff_index += 1

    return eigenvalues


def compute_polynomial(
    A: BlockEncodingModel, steps, samples, transform, sup_norm_constraint, use_kappa
):
    evs = estimate_eigenvalues(A, steps, samples, transform == "square", use_kappa)

    if not sup_norm_constraint:
        if transform == "square_outer":
            if len(evs) > (steps + 1) // 2:
                evs = evs[: (steps + 1) // 2]
            evs = np.concatenate((np.sqrt(evs), -np.sqrt(evs)))
        deg = min(steps, len(evs) - 1)
        poly = np.polynomial.Chebyshev.fit(evs, 1 / evs, deg, domain=[-1, 1])
        return poly

    best_poly = None

    deg = steps
    if transform == "square_outer":
        evs = np.sqrt(evs)
        if deg % 2 == 0:
            deg -= 1

    while deg > 0:
        # For transform == "square_outer" the lagrange basis is already
        # symmetrizised
        lagrange = lagrange_basis(deg, transform)
        lagrange_at_evs = np.zeros((len(evs), deg + 1))
        for i, poly in enumerate(lagrange):
            lagrange_at_evs[:, i] = poly(evs)

        hess = lagrange_at_evs.T @ lagrange_at_evs
        c = lagrange_at_evs.T @ (1 / evs)

        def f(coef):
            return 0.5 * coef.T @ hess @ coef - c.T @ coef

        def Df(coef):
            return hess @ coef - c

        # We do not want to keep the bound exactly
        relaxation_factor = 1.2
        # Account for evs being negative, which should only happen very seldomly
        min_ev = max(np.min(evs), 0.001)
        bound = relaxation_factor / min_ev

        res = sp.optimize.minimize(
            f,
            np.zeros(deg + 1),
            jac=Df,
            bounds=[(-bound, bound)] * (deg + 1),
        )
        poly = sum([res.x[i] * lagrange[i] for i in range(deg + 1)])

        # If the points are interpolated exactly, reduce the polynomial degree
        # until there is some error, and return the smallest polynomial that has
        # zero error.
        if res.fun + 0.5 * np.dot(1 / evs, 1 / evs) > 1e-6:
            if best_poly is None:
                best_poly = poly
            break
        best_poly = poly

        if transform == "square_outer":
            deg -= 2
        else:
            deg -= 1

    return best_poly
