from functools import cached_property

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import sys
from util import maximal_x


class Solver:
    def __init__(
        self,
        default_samples=10000,
        transform_method=None,
    ):
        self.transform_method = transform_method
        self.default_samples = default_samples

    def evaluate(self, A, poly):
        return A.estimate_poly(
            poly,
            self.default_samples,
            qoi=True,
            root=self.transform_method == "square",
        )

    def compute_polynomial(self, A):
        raise NotImplementedError

    def plot(self, A):
        import matplotlib.pyplot as plt

        poly = self.compute_polynomial(A)
        xs = np.linspace(-1, 1, 200)
        if self.transform_method == "square":
            plt.plot(xs, poly(np.sqrt(xs)))
        else:
            plt.plot(xs, poly(xs))
        plt.plot(xs, 1 / xs, "--")
        plt.axvline(x=1 / A.kappa)
        plt.ylim([-11, 11])


class KrylovSolver(Solver):
    def __init__(
        self,
        steps=3,
        poly_kind="chebyshev",
        gram_moments=False,
        default_samples=10000,
        inf_constraint=False,
        transform_method=None,
    ):
        super().__init__(default_samples, transform_method)
        self.steps = steps
        if poly_kind == "monomial":
            self.poly_kind = np.polynomial.Polynomial
        elif poly_kind == "chebyshev":
            self.poly_kind = np.polynomial.Chebyshev
        else:
            raise NotImplementedError
        self.inf_constraint = inf_constraint

    @cached_property
    def moment_basis(self):
        max_degree = 2 * self.steps + 1

        if self.transform_method == "square":
            max_degree *= 2
        return [
            self.poly_kind([0] * i + [1]).convert(kind=np.polynomial.Chebyshev)
            for i in range(max_degree + 1)
        ]

    @cached_property
    def chebyshev_to_moment_basis(self):
        return np.linalg.inv(self.moment_basis_to_chebyshev)

    @cached_property
    def moment_basis_to_chebyshev(self):
        return np.array(
            [
                np.concatenate(
                    (poly.coef, np.zeros(len(self.moment_basis) - 1 - poly.degree()))
                )
                for poly in self.moment_basis
            ]
        ).T

    def compute_moments(self, A):
        """
        Estimate moments of the form b^T p(A) b (if qoi = False)
        or m^T p(A) b (if qoi = True)
        """

        moments = np.zeros(len(self.moment_basis))
        for i, poly in enumerate(self.moment_basis):
            # For transfrom_method == 2 and poly_kind == "chebyshev"
            # we do not actually need the even degrees
            if (
                self.transform_method == "square"
                and self.poly_kind is np.polynomial.Chebyshev
                and i % 2 == 1
            ):
                continue
            moments[i] = A.estimate_poly(
                poly,
                self.default_samples,
                root=self.transform_method == "square",
            )

        return moments

    @cached_property
    def moment_to_gram(self):
        X = np.polynomial.Chebyshev([0, 1])

        mat_G = np.zeros((self.steps + 1, self.steps + 1, len(self.moment_basis)))
        mat_M = np.zeros((self.steps + 1, self.steps + 1, len(self.moment_basis)))
        for i in range(self.steps + 1):
            poly_i = np.polynomial.Chebyshev([0] * i + [1])

            for j in range(self.steps + 1):
                poly_j = np.polynomial.Chebyshev([0] * j + [1])
                poly_G = poly_i * poly_j
                poly_M = X * poly_G
                if self.transform_method == "square":
                    poly_G = poly_G(X * X)
                    poly_M = poly_M(X * X)
                mat_G[i, j] = (
                    self.chebyshev_to_moment_basis[:, : poly_G.degree() + 1]
                    @ poly_G.coef
                )
                mat_M[i, j] = (
                    self.chebyshev_to_moment_basis[:, : poly_M.degree() + 1]
                    @ poly_M.coef
                )

        return mat_G, mat_M

        # # Compute coefficients
        # if self.inf_constraint:
        #     lagrange_to_basis = np.zeros((self.steps + 1, self.steps + 1))
        #     X = np.polynomial.Chebyshev([0, 1])
        #     cheb2 = np.polynomial.Chebyshev([0] * self.steps + [1]).deriv() * (
        #         1 - X * X
        #     )
        #     cheb_nodes = np.cos(np.arange(self.steps + 1) / self.steps * np.pi)
        #     for i in range(self.steps + 1):
        #         assert cheb2(cheb_nodes[i]) < 1e-10
        #         lagrange = cheb2 // (cheb_nodes[i] - X)
        #         lagrange /= lagrange(cheb_nodes[i])
        #         lagrange = lagrange.convert(kind=self.poly_kind)
        #         lagrange_to_basis[:, i] = lagrange.coef

        #     self.L = lagrange_to_basis

    def estimate_eigenvalues(self, A):
        moments = self.compute_moments(A)

        # Compute lhs matrix and rhs vector
        mat_G, mat_M = self.moment_to_gram
        G = mat_G @ moments
        M = mat_M @ moments

        # if self.inf_constraint:
        #     raise NotImplementedError
        #     hess = self.L.T @ G @ self.L
        #     # TODO: How to estimate this
        #     bound = A.kappa
        #     res = minimize(
        #         lambda x: 0.5 * np.dot(x, hess @ x) - np.dot(self.L @ x, r),
        #         np.zeros(self.steps + 1),
        #         jac=lambda x: hess @ x - self.L.T @ r,
        #         bounds=[(-bound, bound)] * (self.steps + 1),
        #     )
        #     if not res.success:
        #         print(res.message, res, file=sys.stderr)
        #     self.coefficients = self.L @ res.x

        # print(moments)
        # print(M, np.linalg.cond(M))
        # print(G, np.linalg.cond(G))
        # print(sp.linalg.eig(G))
        # print(sp.linalg.eig(M))
        # print(sp.linalg.eig(M, G))
        # Ghi = np.linalg.inv(np.linalg.cholesky(G))
        # X = Ghi.T @ M @ Ghi
        # print(np.linalg.eigvals(X))
        # eigenvalues, eigenvectors = sp.linalg.eigh(M, G)

        # First remove directions corresponding to small eigenvalues of G
        E_G, V_G = sp.linalg.eigh(G)
        # print(E_G)
        E_G_max = np.max(E_G)
        cutoff_index = np.searchsorted(E_G, E_G_max * 2 / np.sqrt(self.default_samples))

        # Then, if there are still negative eigenvalues, continuously remove the smallest eigenvalue
        while True:
            P = np.diag(1 / np.sqrt(E_G[cutoff_index:])) @ V_G[:, cutoff_index:].T
            MG = P @ M @ P.T
            eigenvalues, eigenvectors = sp.linalg.eigh(MG)
            if np.min(eigenvalues) > 0 or cutoff_index == len(E_G) - 1:
                break
            cutoff_index += 1

        # print(eigenvalues)
        weights = eigenvectors[0, :]
        print(eigenvalues)
        return eigenvalues.real, weights.real

    def compute_polynomial(self, A):
        evs, weights = self.estimate_eigenvalues(A)
        poly = np.polynomial.Chebyshev.fit(evs, 1 / evs, len(evs) - 1, domain=[-1, 1])
        # np.polynomial.Chebyshev.fit(evs, 1 / evs, len(evs) - 1, w=weights)

        X = np.polynomial.Chebyshev([0, 1])

        # while poly.degree() < self.steps:
        #     new_x = maximal_x(poly * X - 1, np.min(evs), np.max(evs))
        #     evs = np.concatenate((evs, [new_x]))
        #     poly = np.polynomial.Chebyshev.fit(evs, 1 / evs, len(evs) - 1, domain=[-1,1])

        print(poly)

        if self.transform_method == "square":
            poly = poly(X * X)
        return poly


class StationarySolver(Solver):
    def __init__(
        self,
        steps=3,
        default_samples=10000,
        transform_method=None,
        poly_kind="qsvt",
    ):
        super().__init__(default_samples, transform_method)

        self.steps = steps
        self.poly_kind = poly_kind

    def compute_polynomial(self, A):
        if self.poly_kind == "qsvt":
            coefficients = np.zeros(2 * self.steps + 2)

            # TODO: Is there a better way to estimate the correct factor instead of 2?
            b = int(np.ceil(A.kappa**2 * 2))
            b = max(b, self.steps)

            for j in range(self.steps + 1):
                coefficients[2 * j + 1] = (
                    4 * (-1) ** j * sp.special.bdtrc(j + b, 2 * b, 0.5)
                )

            poly = np.polynomial.Chebyshev(coefficients)
        elif self.poly_kind in ["chebyshev_positive", "chebyshev_symmetric"]:
            X = np.polynomial.Chebyshev([0, 1])
            poly = np.polynomial.Chebyshev([0] * self.steps + [1])
            poly = poly((X - (1 / A.kappa + 1) / 2) / (1 - 1 / A.kappa) * 2)
            poly = 1 - poly / poly(0)

            if self.poly_kind == "chebyshev_symmetric":
                poly = poly(X * X)

            poly = poly // X
        else:
            raise NotImplementedError

        if self.transform_method == "square":
            X_sq = np.polynomial.Polynomial([0, 0, 1]).convert(
                kind=np.polynomial.Chebyshev
            )
            poly = poly(X_sq)

        return poly
