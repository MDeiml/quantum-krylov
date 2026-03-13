import numpy as np
import scipy as sp

from block_encoding_model import BlockEncodingModel


def compute_polynomial(A: BlockEncodingModel, steps, samples, poly_kind, square):
    X = np.polynomial.Chebyshev([0, 1])
    sq = X * X

    if poly_kind == "qsvt":
        coefficients = np.zeros(2 * steps + 2)

        accuracy = 2 / np.sqrt(samples)
        b = int(np.ceil(A.kappa**2 * np.log(A.kappa / accuracy)))
        b = max(b, steps)

        for j in range(steps + 1):
            coefficients[2 * j + 1] = (
                4 * (-1) ** j * sp.special.bdtrc(j + b, 2 * b, 0.5)
            )

        poly = np.polynomial.Chebyshev(coefficients)
    elif poly_kind in ["chebyshev_positive", "chebyshev_symmetric"]:
        poly = np.polynomial.Chebyshev([0] * steps + [0, 1])
        kappa = A.kappa
        if poly_kind == "chebyshev_symmetric":
            kappa = kappa**2
        poly = poly((X - (1 / kappa + 1) / 2) / (1 - 1 / kappa) * 2)
        poly = 1 - poly / poly(0)

        if poly_kind == "chebyshev_symmetric":
            poly = poly(sq)

        poly = poly // X
    else:
        raise NotImplementedError

    return poly
