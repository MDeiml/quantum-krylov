import numpy as np

def sup_norm(poly):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > -1., extrema < 1.)]
    extrema = np.concatenate((extrema, [-1, 1]))

    return np.max(np.abs(poly(extrema)))
