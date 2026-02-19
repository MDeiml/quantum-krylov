from functools import cached_property

import numpy as np
import scipy as sp


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
            plt.plot(xs, poly(np.abs(xs)))
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
        default_samples=10000,
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
        self.domain = [0, 1] if transform_method == "square" else [-1, 1]

    @cached_property
    def moment_basis(self):
        max_degree = 2 * self.steps + 1

        basis = [
            self.poly_kind([0] * i + [1], domain=self.domain).convert(
                kind=np.polynomial.Chebyshev, domain=self.domain
            )
            for i in range(max_degree + 1)
        ]
        return basis

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

        X = np.polynomial.Chebyshev([0, 1])
        sq = X * X

        moments = np.zeros(len(self.moment_basis))
        for i, poly in enumerate(self.moment_basis):
            poly = poly.convert(domain=[-1, 1])
            if self.transform_method == "square":
                poly = poly(sq)
            moments[i] = A.estimate_poly(
                poly,
                self.default_samples,
                root=self.transform_method == "square",
            )

        return moments

    @cached_property
    def moment_to_gram(self):
        X = np.polynomial.Chebyshev([0, 1]).convert(domain=self.domain)

        mat_G0 = np.zeros((self.steps + 1, self.steps + 1, len(self.moment_basis)))
        mat_G1 = np.zeros((self.steps + 1, self.steps + 1, len(self.moment_basis)))
        for i in range(self.steps + 1):
            poly_i = np.polynomial.Chebyshev([0] * i + [1], domain=self.domain)

            for j in range(self.steps + 1):
                poly_j = np.polynomial.Chebyshev([0] * j + [1], domain=self.domain)
                poly_G0 = poly_i * poly_j
                poly_G1 = X * poly_G0
                mat_G0[i, j] = (
                    self.chebyshev_to_moment_basis[:, : poly_G0.degree() + 1]
                    @ poly_G0.coef
                )
                mat_G1[i, j] = (
                    self.chebyshev_to_moment_basis[:, : poly_G1.degree() + 1]
                    @ poly_G1.coef
                )

        return mat_G0, mat_G1

    def estimate_eigenvalues(self, A):
        moments = self.compute_moments(A)

        # Compute lhs matrix and rhs vector
        mat_G0, mat_G1 = self.moment_to_gram
        G0 = mat_G0 @ moments
        G1 = mat_G1 @ moments

        # First remove directions corresponding to small eigenvalues of G
        accuracy = 2 / np.sqrt(self.default_samples)
        E0, V0 = sp.linalg.eigh(G0)
        E0_max = E0[-1]
        cutoff_index = np.searchsorted(E0, E0_max * accuracy)

        # Then, if there are still negative eigenvalues, continuously remove the smallest eigenvalue
        while True:
            P = np.diag(1 / np.sqrt(E0[cutoff_index:])) @ V0[:, cutoff_index:].T
            G = P @ G1 @ P.T
            eigenvalues, eigenvectors = sp.linalg.eigh(G)
            if np.min(eigenvalues) > accuracy or cutoff_index == len(E0) - 1:
                break
            cutoff_index += 1

        return eigenvalues

    def compute_polynomial(self, A):
        evs = self.estimate_eigenvalues(A)
        poly = np.polynomial.Chebyshev.fit(evs, 1 / evs, len(evs) - 1, domain=[-1, 1])
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
            poly = np.polynomial.Chebyshev([0] * self.steps + [0, 1])
            poly = poly((X - (1 / A.kappa + 1) / 2) / (1 - 1 / A.kappa) * 2)
            poly = 1 - poly / poly(0)

            if self.poly_kind == "chebyshev_symmetric":
                poly = poly(X * X)

            poly = poly // X
        else:
            raise NotImplementedError

        return poly
