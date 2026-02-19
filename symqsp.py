import numpy as np
import scipy as sp
from util import sup_norm


def interpolation_points(reduced_degree, parity):
    return np.cos(np.pi * np.arange(reduced_degree) / (2 * reduced_degree))


def DF(reduced_phases, parity, xs):
    state = np.zeros((3, len(xs)))
    xs_sqrt = np.sqrt(1 - xs**2)
    xs_main_diagonal = 2 * xs**2 - 1
    xs_off_diagonal = 2 * xs * xs_sqrt
    xs_operator = np.array(
        [[xs_main_diagonal, -xs_off_diagonal], [xs_off_diagonal, xs_main_diagonal]]
    )
    xs_operator = np.moveaxis(xs_operator, -1, 0)
    if parity == 0:
        state[0] = 1
    else:
        state[0] = xs
        state[2] = xs_sqrt
    cos_phase = np.cos(2 * reduced_phases[0])
    sin_phase = np.sin(2 * reduced_phases[0])
    op = np.array([[cos_phase, -sin_phase], [sin_phase, cos_phase]])
    state[[0, 1]] = op @ state[[0, 1]]
    for i in range(1, len(reduced_phases)):
        cos_phase = np.cos(2 * reduced_phases[i])
        sin_phase = np.sin(2 * reduced_phases[i])
        op = np.array([[cos_phase, -sin_phase], [sin_phase, cos_phase]])
        state[[0, 2]] = np.matvec(xs_operator, state[[0, 2]].T).T
        state[[0, 1]] = op @ state[[0, 1]]

    value = state[1].copy()

    dual_state = np.zeros((3, len(xs)))
    dual_state[1] = 2

    derivative = np.zeros((len(xs), len(reduced_phases)))

    xs_operator = np.swapaxes(xs_operator, -1, -2)
    for i in range(len(reduced_phases) - 1, 0, -1):
        derivative[:, i] = dual_state[1] * state[0] - dual_state[0] * state[1]
        phase = reduced_phases[i]
        cos_phase = np.cos(2 * phase)
        sin_phase = np.sin(2 * phase)
        op = np.array([[cos_phase, sin_phase], [-sin_phase, cos_phase]])
        state[[0, 1]] = op @ state[[0, 1]]
        dual_state[[0, 1]] = op @ dual_state[[0, 1]]
        state[[0, 2]] = np.matvec(xs_operator, state[[0, 2]].T).T
        dual_state[[0, 2]] = np.matvec(xs_operator, dual_state[[0, 2]].T).T

    derivative[:, 0] = dual_state[1] * state[0] - dual_state[0] * state[1]

    return value, derivative


def compute_angles_internal(poly):
    degree = poly.degree()
    reduced_degree = int(np.ceil((degree + 1) / 2))
    parity = degree % 2

    xs = interpolation_points(reduced_degree, parity)
    samples = poly(xs)
    reduced_phases = poly.coef[parity::2] / 2
    assert len(reduced_phases) == reduced_degree

    def fun(x):
        value, derivative = DF(x, parity, xs)
        return value - samples, derivative

    res = sp.optimize.root(fun, reduced_phases, jac=True)
    if not res.success:
        return None
    reduced_phases = res.x

    # Turn angles to encode polynomial in real part
    reduced_phases[-1] -= np.pi / 4
    full_phases = np.zeros(poly.degree() + 1)
    full_phases[reduced_degree - 1 :: -1] = reduced_phases
    full_phases[-reduced_degree::] += reduced_phases

    return full_phases


def compute_angles(poly):
    """
    Computes angles for given polynomial

    This takes into account the parity and normalization of the polynomial.
    It returns a list of one (if the polynomial has definite partiy) or two
    tuples. The first element of the tuples is an angle sequence for QSVT
    and the second element is the corresponding weight, such that the sum of
    QSVT polynomials equals the input to this function.
    """
    poly = poly.convert(kind=np.polynomial.Chebyshev)

    parity = poly.degree() % 2
    is_parity = np.allclose(poly.coef[(1 - parity) :: 2], 0, atol=1e-6)

    coefs_parity = poly.coef.copy()
    coefs_parity[1 - parity :: 2] = 0
    poly_parity = np.polynomial.Chebyshev(coefs_parity)
    polys = [poly_parity]
    if not is_parity:
        coefs_non_parity = poly.coef[:-1].copy()
        coefs_non_parity[parity::2] = 0
        poly_non_parity = np.polynomial.Chebyshev(coefs_non_parity)
        polys.append(poly_non_parity)

    result = []
    for subpoly in polys:
        if np.allclose(subpoly.coef, np.array([0] * subpoly.degree() + [1])):
            angles = np.pi * np.ones(subpoly.degree() + 1)
            if subpoly.degree() % 2 == 0:
                angles[0] -= np.pi / 2
                angles[-1] -= np.pi / 2
            normalization = 1
        else:
            normalization = sup_norm(subpoly) * 1.01
            angles = None
            max_tries = 100
            for _ in range(max_tries):
                angles = compute_angles_internal(subpoly / normalization)
                if angles is not None:
                    break
                normalization *= 1.01
            if angles is None:
                raise RuntimeError(f"Could not compute angles for {subpoly}")

        result.append((subpoly / normalization, angles, normalization))

    return result
