import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from util import sup_norm


N = 3


def plot_basis(basis):
    xs = np.linspace(0, 1)
    for P in basis:
        plt.plot(xs, P(xs))

    plt.show()


def inner_positive(P, Q):
    integral = (P * Q).integ()
    return integral(1) - integral(0)


def gram(basis):
    G = np.zeros((len(basis), len(basis)))
    for i, P in enumerate(basis):
        for j, Q in enumerate(basis):
            G[i, j] = inner_positive(P, Q)
    return G


# def normalize(basis):
#     return [P / sup_norm(P, range=(0.,1.)) for P in basis]


def normalize(basis):
    result = []
    for P in basis:
        norm = sup_norm(P)
        if norm > 1:
            P /= norm
        result.append(P)
    return result


def target(basis):
    return np.linalg.eigvalsh(gram(basis))[-(N+1)]


def lagrange_chebyshev(N):
    basis = []

    for i in range(N + 1):
        P = np.polynomial.Chebyshev([1])
        x_i = np.cos(i * np.pi / N)
        for j in range(N + 1):
            if i == j:
                continue
            x_j = np.cos(j * np.pi / N)
            P *= np.polynomial.Chebyshev([-x_j, 1]) / (x_i - x_j)
        basis.append(P)
    return basis


print("chebyshev")
print(np.polynomial.Chebyshev([-1, 2]).convert(kind=np.polynomial.Polynomial))
# basis = [np.polynomial.Chebyshev([0] * i + [1]) for i in range(N + 1)]
basis = [np.polynomial.Chebyshev([0] * i + [1], domain=[0,1]).convert(domain=[-1,1]) for i in range(N + 1)]
X = np.polynomial.Chebyshev([0, 1])
sq = X * X
for P in basis:
    print(P(sq))
print(target(basis))
plot_basis(basis)

quit()
print("monomial")
basis = normalize([np.polynomial.Polynomial([0] * i + [1]) for i in range(N + 1)])
print(target(basis))
print("chebyshev_both")
basis = normalize(
    [np.polynomial.Chebyshev([0] * i + [1]) for i in range(N + 1)]
    + [np.polynomial.Chebyshev([0] * i + [0, 1]).deriv() for i in range(N + 1)]
)
print(target(basis))
print("chebyshev2")
basis = normalize(
    [np.polynomial.Chebyshev([0] * i + [0, 1]).deriv() for i in range(N + 1)]
)
print(target(basis))
print("legendre")
basis = normalize(
    [np.polynomial.Legendre([0] * i + [1], domain=[0, 1]) for i in range(N + 1)]
)
print(target(basis))

print("lagrange")
lagrange_basis = normalize(lagrange_chebyshev(N))
print(target(lagrange_basis))

print("all")
basis = normalize(
    [np.polynomial.Chebyshev([0] * i + [1]) for i in range(N + 1)]
    + [np.polynomial.Chebyshev([0] * i + [0, 1]).deriv() for i in range(N + 1)]
    # + [np.polynomial.Legendre([0] * i + [1], domain=[0, 1]).convert(kind=np.polynomial.Chebyshev, domain=[-1, 1]) for i in range(N + 1)]
    # + lagrange_chebyshev(N)
)
print(target(basis))

def construct_basis(x):
    x = x.reshape((-1, N + 1))
    basis = []
    for i in range(N + 1):
        P = np.polynomial.Chebyshev(0)
        for j in range(x.shape[1]):
            P += x[j, i] * lagrange_basis[i]
        basis.append(P)
    return basis


def f(x):
    basis = construct_basis(x)
    return -target(basis)


steps = 1000
tries = 10
max_deg = N + 2
winner = normalize(
    [np.polynomial.Chebyshev([0] * i + [0, 1]).deriv() for i in range(max_deg + 1)]
)
for _ in range(tries):
    learning_rate = 0.1
    basis = normalize(
        construct_basis(np.random.randn(N+1, max_deg+1))
    )
    result = target(basis)
    for _ in range(steps):
        basis_new = []
        for P in basis:
            coef = np.zeros(max_deg + 1)
            coef[:P.degree()+1] = P.coef
            basis_new.append(np.polynomial.Chebyshev(coef + learning_rate * np.random.randn(max_deg + 1)))
        basis_new = normalize(basis_new)
        result_new = target(basis_new)
        if result_new > result:
            basis = basis_new
            result = result_new
            # print(learning_rate)
        learning_rate *= 0.99
    print(result)
    if result > target(winner):
        winner = basis

print(target(winner))

plot_basis(winner)
