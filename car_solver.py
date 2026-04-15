from functools import cache

import numpy as np
import scipy as sp

from block_encoding_model import BlockEncodingModel


def estimate_moments(A: BlockEncodingModel, steps, samples, square):
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


def gauss_quadrature(kappa):
    # TODO: Variably choose number of quadrature points
    xq, wq = np.polynomial.legendre.leggauss(100)
    # return xq, wq
    return (xq + 1) / 2 * (1 - 1 / kappa) + 1 / kappa, wq / 2 * (1 - 1 / kappa)


def cheb_at(max_degree, xs):
    result = np.zeros((len(xs), max_degree + 1))
    for i in range(max_degree + 1):
        result[:, i] = np.cos(i * np.arccos(xs))

    return result


@cache
def cheb_at_xq(max_degree, kappa, square):
    xq, _wq = gauss_quadrature(kappa)
    if square:
        # If estimate_moments is used with square=True, then
        # the moments correspond to the polynomials
        # T_{2n}(sqrt(x)) = T_n(2x - 1)
        xq = xq * 2 - 1
    return cheb_at(max_degree, xq)


def maximum_entropy_measure(moments, kappa, square):
    p0 = np.zeros_like(moments)

    xq, wq = gauss_quadrature(kappa)
    cheb_q = cheb_at_xq(len(moments) - 1, kappa, False)

    # Moments at xq
    moments_at_xq = cheb_at_xq(len(moments) - 1, kappa, square)

    def objective(p):
        # rho at quadrature points
        rho_q = np.exp(cheb_q @ p)
        moments_p = moments_at_xq.T @ (wq * rho_q)

        residual = moments_p - moments
        fun = 0.5 * np.dot(residual, residual)

        gradient = cheb_q.T @ ((moments_at_xq @ residual) * wq * rho_q)

        return fun, gradient

    res = sp.optimize.minimize(objective, p0, jac=True)
    p = res.x

    return np.polynomial.Chebyshev(p)


def car_solver(
    A: BlockEncodingModel,
    steps,
    samples,
    transform=None,
    adaptive=True,
):
    square = transform == "square"
    if adaptive:
        moments = estimate_moments(A, steps, samples, square)
        rho_poly = maximum_entropy_measure(moments, A.kappa, square)
    else:
        rho_poly = np.polynomial.Chebyshev([1 / (1 - 1 / A.kappa)])
    xq, wq = gauss_quadrature(A.kappa)
    rho_q = np.exp(rho_poly(xq))

    noise_level = 2 / np.sqrt(samples)

    cheb_q = cheb_at_xq(steps, A.kappa, False)
    x_sup = np.cos(np.linspace(0, np.pi, 2 * steps))
    if square:
        # If transform == "square" then the normalization
        # is exactly the sup norm over positive values
        x_sup = (x_sup + 1) / 2
        cheb_at_x_sup = cheb_at(steps, x_sup)
        constraint = np.zeros((4, len(x_sup), cheb_q.shape[1] + 2))
        constraint[0, :, 0] = 1
        constraint[0, :, 2:] = -cheb_at_x_sup
        constraint[1, :, 0] = 1
        constraint[1, :, 2:] = cheb_at_x_sup
        constraint = constraint.reshape((-1, cheb_q.shape[1] + 2))
    elif transform == "square_outer":
        # If transform == "square_outer", we have to look
        # for an odd polynomial, and the measure is transformed
        # by a square.
        # The normalization is the sup norm over [-1, 1].
        xq, wq = gauss_quadrature(np.sqrt(A.kappa))
        rho_q = np.exp(rho_poly(xq ** 2))
        cheb_q = cheb_at_xq(steps, np.sqrt(A.kappa), False)[:, 1::2]
        cheb_at_x_sup = cheb_at(steps, x_sup)[:, 1::2]
        constraint = np.zeros((4, len(x_sup), cheb_q.shape[1] + 2))
        constraint[0, :, 0] = 1
        constraint[0, :, 2:] = -cheb_at_x_sup
        constraint[1, :, 0] = 1
        constraint[1, :, 2:] = cheb_at_x_sup
        constraint = constraint.reshape((-1, cheb_q.shape[1] + 2))
    else:
        # Otherwise the normalization is the sup norm (over
        # all values in [-1, 1]) for the odd part plus the
        # sup norm of the even part
        cheb_at_x_sup = cheb_at(steps, x_sup)
        cheb_at_x_sup_even = cheb_at_x_sup.copy()
        cheb_at_x_sup_even[:, 1::2] = 0
        cheb_at_x_sup_odd = cheb_at_x_sup.copy()
        cheb_at_x_sup_odd[:, ::2] = 0
        constraint = np.zeros((4, len(x_sup), cheb_q.shape[1] + 2))
        constraint[0, :, 0] = 1
        constraint[0, :, 2:] = -cheb_at_x_sup_even
        constraint[1, :, 0] = 1
        constraint[1, :, 2:] = cheb_at_x_sup_even
        constraint[2, :, 1] = 1
        constraint[2, :, 2:] = -cheb_at_x_sup_odd
        constraint[3, :, 1] = 1
        constraint[3, :, 2:] = cheb_at_x_sup_odd
        constraint = constraint.reshape((-1, cheb_q.shape[1] + 2))

    hess = cheb_q.T @ np.diag(rho_q * wq) @ cheb_q
    c = cheb_q.T @ np.diag(rho_q * wq) @ (1 / xq)

    def f(coef):
        return (
            0.5 * coef[2:].T @ hess @ coef[2:]
            - c.T @ coef[2:]
            + 0.5 * (coef[0] + coef[1]) ** 2 * noise_level
        )

    def Df(coef):
        return np.concatenate(
            (
                [
                    noise_level * (coef[0] + coef[1]),
                    noise_level * (coef[0] + coef[1]),
                ],
                hess @ coef[2:] - c,
            )
        )

    if noise_level == 0:
        coef = np.linalg.solve(hess, c)
    else:
        res = sp.optimize.minimize(
            f,
            np.zeros(cheb_q.shape[1] + 2),
            jac=Df,
            constraints=[
                sp.optimize.LinearConstraint(constraint, lb=0, ub=np.inf),
            ],
        )
        coef = res.x[2:]
    if transform == "square_outer":
        temp = coef
        coef = np.zeros(steps + 1)
        coef[1::2] = temp
    poly = np.polynomial.Chebyshev(coef)

    return poly
