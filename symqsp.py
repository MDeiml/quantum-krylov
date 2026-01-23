import numpy as np
import scipy as sp


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


def compute_angles(poly):
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


def simulate_qsp(angles, S, x):
    x = x.astype(np.complex128)
    x_bad = np.zeros_like(x, dtype=np.complex128)
    x *= np.exp(angles[0] * 1j)
    x_bad *= np.exp(-angles[0] * 1j)
    for angle in angles[1:]:
        x, x_bad = (
            S * x + 1j * np.sqrt(1 - S**2) * x_bad,
            1j * np.sqrt(1 - S**2) * x + S * x_bad,
        )
        x *= np.exp(angle * 1j)
        x_bad *= np.exp(-angle * 1j)

    return x.real


def test(N=8):
    from util import sup_norm

    for i in range(1000):
        parity = np.random.randint(0, 2)
        coef = np.zeros(2 * N + parity)
        coef[parity::2] = np.random.randn(N)
        p = np.polynomial.Chebyshev(coef)
        p /= sup_norm(p) * 1.01

        angles = compute_angles(p)
        if angles is None:
            print(p)

        S = np.linspace(-1, 1)
        res = simulate_qsp(angles, S, np.ones_like(S))

        np.testing.assert_allclose(res, p(S), err_msg=p, atol=1e-8)


# for i in range(1, 10):
#     p = np.polynomial.Chebyshev(i * [0] + [1])

#     # angles = compute_angles(p)
#     angles = np.pi * np.ones(i + 1)
#     if i % 2 == 0:
#         angles[0] = -np.pi/2
#         angles[-1] = -np.pi/2
#     print(angles)

#     S = np.linspace(-1, 1)
#     res = simulate_qsp(angles, S, np.ones_like(S))

#     np.testing.assert_allclose(res, p(S), err_msg=p, atol=1e-8)
