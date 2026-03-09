import numpy as np
import scipy as sp

from solver import Solver
from block_encoding_model import BlockEncodingModel


class NonadaptiveSolver(Solver):
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

    def compute_polynomial(self, A: BlockEncodingModel):
        X = np.polynomial.Chebyshev([0, 1])
        sq = X * X

        if self.poly_kind == "qsvt":
            coefficients = np.zeros(2 * self.steps + 2)

            accuracy = 2 / np.sqrt(self.default_samples)
            b = int(np.ceil(A.kappa**2 * np.log(A.kappa / accuracy)))
            b = max(b, self.steps)

            for j in range(self.steps + 1):
                coefficients[2 * j + 1] = (
                    4 * (-1) ** j * sp.special.bdtrc(j + b, 2 * b, 0.5)
                )

            poly = np.polynomial.Chebyshev(coefficients)
        elif self.poly_kind in ["chebyshev_positive", "chebyshev_symmetric"]:
            poly = np.polynomial.Chebyshev([0] * self.steps + [0, 1])
            kappa = A.kappa
            if self.poly_kind == "chebyshev_symmetric":
                kappa = kappa ** 2
            poly = poly((X - (1 / kappa + 1) / 2) / (1 - 1 / kappa) * 2)
            poly = 1 - poly / poly(0)

            if self.poly_kind == "chebyshev_symmetric":
                poly = poly(sq)

            poly = poly // X
        else:
            raise NotImplementedError

        return poly
