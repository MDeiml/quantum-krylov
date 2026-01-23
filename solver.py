import numpy as np
import scipy as sp
from scipy.optimize import minimize
import sys


class Solver:
    def __init__(
        self,
        default_samples=10000,
        transform_method=None,
    ):
        self.transform_method = transform_method
        self.default_samples = default_samples

    def evaluate(self, A, poly):
        X = np.polynomial.Polynomial([0, 1]).convert(kind=poly.__class__)
        if self.transform_method == "square":
            poly = poly(X * X)
        return A.estimate_poly(
            poly,
            self.default_samples,
            qoi=True,
            root=self.transform_method == "square",
        )

    def precompute(self):
        pass

    def estimate_eigenvalues(self, A) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def compute_polynomial(self, A):
        evs, weights = self.estimate_eigenvalues(A)
        return np.polynomial.Chebyshev.fit(evs, 1/evs, weights)

    def plot(self, A):
        import matplotlib.pyplot as plt

        poly = self.compute_polynomial(A)
        xs = np.linspace(-1, 1, 200)
        if self.transform_method == "square":
            plt.plot(xs, poly(np.abs(xs)))
        else:
            plt.plot(xs, poly(xs))
        plt.plot(xs, 1 / xs, "--")
        plt.axvline(x=1/A.kappa)
        plt.ylim([-11, 11])


class KrylovSolver(Solver):
    def __init__(
        self,
        steps=3,
        poly_kind="chebyshev",
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

    def compute_moments(self, A):
        """
        Estimate moments of the form b^T p(A) b (if qoi = False)
        or m^T p(A) b (if qoi = True)
        """

        max_degree = 2 * self.steps + 1
        even = 1
        if self.transform_method == "square":
            even = 2

        moments = np.zeros(max_degree + 1)
        for i in range(0, max_degree + 1, even):
            moments[i] = A.estimate_poly(
                self.poly_kind([0] * i * even + [1]),
                self.default_samples,
                root=self.transform_method == "square",
            )

        return moments

    # def plot_poly(self):
    #     coefficients = self.compute_coefficients()
    #     poly = self.poly_kind(coefficients)

    #     xs = np.linspace(-1, 1, 200)
    #     ys = poly(xs)
    #     plt.plot(xs, ys)
    #     plt.plot(xs, 1 / xs)
    #     max = np.max(np.abs(ys))
    #     plt.ylim((-max, max))
    #     plt.show()

    def precompute(self):
        X = np.polynomial.Polynomial([0, 1]).convert(kind=self.poly_kind)

        max_degree = 2 * self.steps + 1

        self.poly_r = np.zeros((self.steps + 1, max_degree + 1))
        self.poly_G = np.zeros((self.steps + 1, self.steps + 1, max_degree + 1))
        self.poly_M = np.zeros((self.steps + 1, self.steps + 1, max_degree + 1))
        for i in range(self.steps + 1):
            poly_i = self.poly_kind([0] * i + [1])

            for j in range(self.steps + 1):
                poly_j = self.poly_kind([0] * j + [1])
                poly_G = poly_i * poly_j
                poly_M = X * poly_G
                if self.transform_method == "square":
                    poly_G = poly_G(X * X)
                    poly_M = poly_M(X * X)
                    self.poly_G[i, j, : poly_G.degree() + 1] = poly_G.coef[::2]
                    self.poly_M[i, j, : poly_M.degree() + 1] = poly_M.coef[::2]
                else:
                    self.poly_G[i, j, : poly_G.degree() + 1] = poly_G.coef
                    self.poly_M[i, j, : poly_M.degree() + 1] = poly_M.coef

        # Compute coefficients
        if self.inf_constraint:
            lagrange_to_basis = np.zeros((self.steps + 1, self.steps + 1))
            X = np.polynomial.Chebyshev([0, 1])
            cheb2 = np.polynomial.Chebyshev([0] * self.steps + [1]).deriv() * (
                1 - X * X
            )
            cheb_nodes = np.cos(np.arange(self.steps + 1) / self.steps * np.pi)
            for i in range(self.steps + 1):
                assert cheb2(cheb_nodes[i]) < 1e-10
                lagrange = cheb2 // (cheb_nodes[i] - X)
                lagrange /= lagrange(cheb_nodes[i])
                lagrange = lagrange.convert(kind=self.poly_kind)
                lagrange_to_basis[:, i] = lagrange.coef

            self.L = lagrange_to_basis

    def estimate_eigenvalues(self, A):
        moments = self.compute_moments(A)

        # Compute lhs matrix and rhs vector
        G = self.poly_G @ moments
        M = self.poly_M @ moments

        if self.inf_constraint:
            raise NotImplementedError
            hess = self.L.T @ G @ self.L
            # TODO: How to estimate this
            bound = A.kappa
            res = minimize(
                lambda x: 0.5 * np.dot(x, hess @ x) - np.dot(self.L @ x, r),
                np.zeros(self.steps + 1),
                jac=lambda x: hess @ x - self.L.T @ r,
                bounds=[(-bound, bound)] * (self.steps + 1),
            )
            if not res.success:
                print(res.message, res, file=sys.stderr)
            self.coefficients = self.L @ res.x

        print(moments)
        print(M)
        print(G)
        print(self.poly_M)
        print(self.poly_G)
        eigenvalues, eigenvectors = sp.linalg.eigh(M, G)
        weights = eigenvectors[0, :]
        return eigenvalues, weights


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

            return np.polynomial.Chebyshev(coefficients)
        elif self.poly_kind in ["chebyshev_positive", "chebyshev_symmetric"]:
            X = np.polynomial.Chebyshev([0, 1])
            poly = np.polynomial.Chebyshev([0] * self.steps + [1])
            poly = poly((X - (1 / A.kappa + 1) / 2) / (1 - 1 / A.kappa) * 2)
            poly = 1 - poly / poly(0)

            if self.poly_kind == "chebyshev_symmetric":
                poly = poly(X * X)

            poly = poly // X

            return poly
        else:
            raise NotImplementedError
