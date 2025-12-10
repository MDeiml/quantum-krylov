import numpy as np
import scipy as sp
from scipy.optimize import minimize
import sys
from util import characteristic_poly


class Solver:
    def __init__(
        self,
        default_samples=10000,
        eval_method="chebyshev",
        transform_method=None,
        project_to_b=False,
    ):
        self.eval_method = eval_method
        self.transform_method = transform_method
        self.default_samples = default_samples
        self.project_to_b = project_to_b

    def evaluate(self, A, poly):
        X = np.polynomial.Polynomial([0, 1]).convert(kind=poly.__class__)
        if self.transform_method == "square":
            poly = poly(X * X)
        if self.eval_method == "qsp":
            return A.estimate_poly(
                poly,
                self.default_samples,
                qoi=True,
                project_to_b=self.project_to_b,
                root=self.transform_method == "square",
                poly_normalized=False,
            )
        elif self.eval_method in ["chebyshev", "monomial"]:
            if self.eval_method == "chebyshev":
                poly_kind = np.polynomial.Chebyshev
            else:
                poly_kind = np.polynomial.Polynomial

            poly = poly.convert(kind=poly_kind)

            even = 1
            if self.transform_method == "square":
                even = 2
            degree = poly.degree()
            moments = np.zeros(degree + 1)
            for i in range(0, degree + 1, even):
                moments[i] = A.estimate_poly(
                    poly_kind([0] * i + [1]),
                    self.default_samples,
                    qoi=True,
                    project_to_b=self.project_to_b,
                    root=self.transform_method == "square",
                    poly_normalized=True,
                )
            return np.dot(poly.coef, moments)
        else:
            raise NotImplementedError

    def precompute(self):
        pass

    def compute_polynomial(self, A):
        raise NotImplementedError

    def solve(self, A):
        return self.evaluate(A, self.compute_polynomial(A))

    def plot(self, A):
        import matplotlib.pyplot as plt

        poly = self.compute_polynomial(A)
        xs = np.linspace(-1, 1, 200)
        if self.transform_method == "square":
            plt.plot(xs, poly(np.abs(xs)))
        else:
            plt.plot(xs, poly(xs))
        plt.plot(xs, 1 / xs, "--")
        plt.ylim([-11, 11])

        plt.show()


class EigenfilteringSolver(Solver):
    def __init__(
        self,
        steps=3,
        poly_kind="chebyshev",
        default_samples=10000,
        use_qsp_for_qoi=False,
    ):
        eval_method = "qsp" if use_qsp_for_qoi else poly_kind
        super().__init__(
            default_samples, eval_method, transform_method=None, project_to_b=True
        )
        self.steps = steps
        if poly_kind == "monomial":
            self.poly_kind = np.polynomial.Polynomial
        elif poly_kind == "chebyshev":
            self.poly_kind = np.polynomial.Chebyshev
        else:
            raise NotImplementedError

    def compute_moments(self, A):
        """
        Estimate moments of the form b^T p(A) b (if qoi = False)
        or m^T p(A) b (if qoi = True)
        """

        max_degree = None
        max_degree = 2 * self.steps + 1
        even = 1

        moments = np.zeros(max_degree + 1)
        for i in range(0, max_degree + 1, even):
            moments[i] = A.estimate_poly(
                self.poly_kind([0] * i + [1]),
                self.default_samples,
                poly_normalized=True,
            )

        return moments

    def precompute(self):
        X = np.polynomial.Polynomial([0, 1]).convert(kind=self.poly_kind)

        max_degree = None
        max_degree = 2 * self.steps + 1

        self.poly_G = np.zeros((self.steps + 1, self.steps + 1, max_degree + 1))
        self.poly_M = np.zeros(
            (self.steps + 1, self.steps + 1, max_degree)
        )  # The inner product in the Krylov space
        self.poly_r = np.zeros((self.steps + 1, max_degree + 1))
        for i in range(self.steps + 1):
            poly_i = self.poly_kind([0] * i + [1])
            self.poly_r[i, : poly_i.degree() + 1] = poly_i.coef
            for j in range(self.steps + 1):
                poly_j = self.poly_kind([0] * j + [1])
                poly_G = X * poly_i * poly_j
                poly_M = poly_i * poly_j
                self.poly_G[i, j, : poly_G.degree() + 1] = poly_G.coef
                self.poly_M[i, j, : poly_M.degree() + 1] = poly_M.coef

    def compute_polynomial(self, A):
        moments = self.compute_moments(A)

        G = self.poly_G @ moments
        M = self.poly_M @ moments[:-1]

        # C = np.linalg.cholesky(M, upper=True)
        # C_inv = np.linalg.inv(C)
        # M_inv = np.linalg.inv(M)
        U, S, V = np.linalg.svd(M)

        A = U.T @ G @ V.T / S
        P = characteristic_poly(A)
        # print(S)
        P /= P(0)

        # Compute norm of the solution from moments
        r = self.poly_r @ moments
        c = np.linalg.solve(G, r)
        norm_x_sq = c.T @ M @ c

        return P * np.sqrt(norm_x_sq)


class KrylovSolver(Solver):
    def __init__(
        self,
        steps=3,
        poly_kind="chebyshev",
        default_samples=10000,
        loss_type="cg",
        inf_constraint=False,
        transform_method=None,
        use_qsp_for_qoi=False,
    ):
        eval_method = "qsp" if use_qsp_for_qoi else poly_kind
        super().__init__(default_samples, eval_method, transform_method)
        self.steps = steps
        if poly_kind == "monomial":
            self.poly_kind = np.polynomial.Polynomial
        elif poly_kind == "chebyshev":
            self.poly_kind = np.polynomial.Chebyshev
        else:
            raise NotImplementedError
        self.loss_type = loss_type
        self.inf_constraint = inf_constraint

    def compute_moments(self, A):
        """
        Estimate moments of the form b^T p(A) b (if qoi = False)
        or m^T p(A) b (if qoi = True)
        """

        max_degree = None
        if self.loss_type == "cg":
            max_degree = 2 * self.steps + 1
        elif self.loss_type == "minres":
            max_degree = 2 * self.steps + 2
        else:
            raise NotImplementedError
        even = 1
        if self.transform_method == "square":
            max_degree *= 2
            even = 2

        moments = np.zeros(max_degree + 1)
        for i in range(0, max_degree + 1, even):
            moments[i] = A.estimate_poly(
                self.poly_kind([0] * i + [1]),
                self.default_samples,
                root=self.transform_method == "square",
                poly_normalized=True,
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

        max_degree = None
        if self.loss_type == "cg":
            max_degree = 2 * self.steps + 1
        elif self.loss_type == "minres":
            max_degree = 2 * self.steps + 2
        else:
            raise NotImplementedError
        if self.transform_method == "square":
            max_degree *= 2

        self.poly_r = np.zeros((self.steps + 1, max_degree + 1))
        self.poly_G = np.zeros((self.steps + 1, self.steps + 1, max_degree + 1))
        for i in range(self.steps + 1):
            poly_i = self.poly_kind([0] * i + [1])
            if self.loss_type == "cg":
                poly_r = poly_i
            elif self.loss_type == "minres":
                poly_r = X * poly_i
            else:
                raise NotImplementedError
            if self.transform_method == "square":
                poly_r = poly_r(X * X)
            self.poly_r[i, : poly_r.degree() + 1] = poly_r.coef

            for j in range(self.steps + 1):
                poly_j = self.poly_kind([0] * j + [1])
                if self.loss_type == "cg":
                    poly_G = X * poly_i * poly_j
                elif self.loss_type == "minres":
                    poly_G = X * X * poly_i * poly_j
                else:
                    raise NotImplementedError
                if self.transform_method == "square":
                    poly_G = poly_G(X * X)
                self.poly_G[i, j, : poly_G.degree() + 1] = poly_G.coef

        # Compute coefficients
        if self.inf_constraint:
            legendre_to_basis = np.zeros((self.steps + 1, self.steps + 1))
            X = np.polynomial.Chebyshev([0, 1])
            cheb2 = np.polynomial.Chebyshev([0] * self.steps + [1]).deriv() * (
                1 - X * X
            )
            cheb_nodes = np.cos(np.arange(self.steps + 1) / self.steps * np.pi)
            for i in range(self.steps + 1):
                assert cheb2(cheb_nodes[i]) < 1e-10
                legendre = cheb2 // (cheb_nodes[i] - X)
                legendre /= legendre(cheb_nodes[i])
                legendre = legendre.convert(kind=self.poly_kind)
                legendre_to_basis[:, i] = legendre.coef

            self.L = legendre_to_basis

    def compute_polynomial(self, A):
        moments = self.compute_moments(A)

        # Compute lhs matrix and rhs vector
        r = self.poly_r @ moments
        G = self.poly_G @ moments
        # G = np.zeros((self.steps + 1, self.steps + 1))
        # r = np.zeros(self.steps + 1)

        # Compute coefficients
        if self.inf_constraint:
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
        else:
            self.coefficients = np.linalg.solve(G, r)

        return self.poly_kind(self.coefficients)


class StationarySolver(Solver):
    def __init__(
        self,
        steps=3,
        default_samples=10000,
        eval_method="qsp",
        transform_method=None,
        poly_kind="qsvt",
    ):
        super().__init__(default_samples, eval_method, transform_method)

        self.steps = steps
        self.poly_kind = poly_kind

    def compute_polynomial(self, A):
        if self.poly_kind == "qsvt":
            coefficients = np.zeros(2 * self.steps + 2)

            # TODO: What to do with the extra factor (log(kappa/tol))
            b = int(np.ceil(A.kappa**2))
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
