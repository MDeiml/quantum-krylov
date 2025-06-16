import numpy as np
from numpy.polynomial import Chebyshev
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy

N = 10

kappa = 3

rng = np.random.Generator(np.random.PCG64(4))

# A = rng.uniform(1/kappa, 1, size=N)
A = rng.uniform(1/kappa, 0.5, size=N)
# A[::2] = rng.uniform(0.75, 1, size=N//2)
A[0] = 1/kappa
# A[-1] = 1

b = rng.normal(size=N)
b /= np.linalg.norm(b)

print(A)
print(b)

def poly_conj(p: Chebyshev) -> Chebyshev:
    return Chebyshev(np.conj(p.coef))

def qsp(angles) -> Chebyshev:
    poly_A = Chebyshev([1])
    poly_B = Chebyshev([0])

    X = Chebyshev([0, 1])

    for angle in angles[:-1]:
        poly_A *= np.exp(angle * 1j)
        poly_B *= np.exp(angle * 1j)
        poly_A_new, poly_B_new = X * poly_A + (X * X - 1) * poly_conj(poly_B), poly_conj(poly_A) + X * poly_B
        poly_A = poly_A_new
        poly_B = poly_B_new

    poly_A *= np.exp(angles[-1] * 1j)
    poly_B *= np.exp(angles[-1] * 1j)

    # print(poly_A, ";", poly_B)
    return Chebyshev(poly_A.coef.imag)

# p = qsp(np.array([np.pi/4, 0, 0, 0, 0, np.pi/4]))
    
# xs = np.linspace(-1, 1, 200)
# plt.plot(xs, p(xs))
# plt.show()

# quit()


def construct_basis_l(M: int) -> list[Chebyshev]:
    basis = []
    X = Chebyshev([0, 1])
    U = Chebyshev([0] * M + [1]).deriv() * (X ** 2 - 1)
    U_d = U.deriv()
    for i in range(M + 1):
        x_i = np.cos(i * np.pi / M)
        weight = 1 / U_d(x_i)
        basis.append(weight * (U // (X - x_i)))

    return basis

def interpolate_l(M: int, poly: Chebyshev) -> np.ndarray:
    return poly(np.cos(np.arange(M + 1) * np.pi / M))

def construct_basis_c(M: int) -> list[Chebyshev]:
    basis = []
    for i in range(M + 1):
        basis.append(Chebyshev([0] * i + [1]))

    return basis

def interpolate_c(M: int, poly: Chebyshev) -> np.ndarray:
    coef = np.zeros(M + 1)
    coef[:len(poly.coef)] = poly.coef
    return coef

construct_basis, interpolate = construct_basis_c, interpolate_c

class KrylovSolver:

    def __init__(self, A: np.ndarray, b: np.ndarray, M: int):
        self.N = len(b)
        self.M = M
        self.A = A
        self.b = b
        self.basis = construct_basis(M)
        self.measurements = 0
        self.samples_one = np.zeros(M + 1, dtype=np.uint32)
        self.rng = np.random.Generator(np.random.PCG64(10))

    def measure(self, poly: Chebyshev, n: int) -> int:
        # Simulates modified hadamard test
        p = np.dot(b, poly(A) * b)
        p = (1 + p)/2
        return self.rng.binomial(n, p)

    def predict(self, poly: Chebyshev) -> float:
        assert poly.degree() <= self.M
        coef = interpolate(self.M, poly)
        return np.dot(coef, 2 * (self.samples_one / self.measurements) - 1)
        # return np.dot(b, poly(A) * b)
        # return np.dot(coef, self.samples_one / self.measurements)

    def add_measurements(self, n: int):
        self.measurements += n
        for j in range(self.M + 1):
            self.samples_one[j] += self.measure(self.basis[j], n)

    def poly_minres(self) -> Chebyshev:

        deg = (self.M // 2) - 1

        basis = construct_basis_l(deg)

        N = deg + 1

        params = np.zeros(N + 1)
        params[-1] = 1
        def cost(params):
            poly = np.dot(basis, params[:-1])

            X = Chebyshev([0, 1])
            norm = (X * poly / params[-1] - 1) ** 2
            return self.predict(norm)

        res = minimize(cost, params, method="L-BFGS-B", bounds=[(-1, 1)] * N + [(0.5, 1)])
        print(res)

        params = res.x
        poly = np.dot(basis, params[:-1])

        return poly / params[-1], params[-1]

    def poly_minres_no_bound(self) -> Chebyshev:
        deg = (self.M // 2) - 1
        K = np.zeros((deg, deg))
        r = np.zeros(deg)
        X = Chebyshev([0, 1])
        for i in range(deg):
            r[i] = self.predict(self.basis[0] * self.basis[i] * X)
            for j in range(deg):
                K[i, j] = self.predict(self.basis[i] * self.basis[j] * X * X)
        coef = np.linalg.solve(K, r)
        return np.dot(coef, self.basis[:len(coef)])

    def poly_inv(self, deg) -> Chebyshev:

        # deg = (self.M // 2) - 1

        basis = construct_basis_l(deg)

        more_exact = self.poly_minres_no_bound()

        N = deg + 1

        params = np.zeros(N + 1)
        params[-1] = 1
        def cost(params):
            poly = np.dot(basis, params[:-1])

            norm = (poly / params[-1] - more_exact) ** 2
            return self.predict(norm)

        res = minimize(cost, params, method="L-BFGS-B", bounds=[(-1, 1)] * N + [(0.5, 1)])
        print(res)

        params = res.x
        poly = np.dot(basis, params[:-1])

        return poly / params[-1], params[-1]

M = 28
solver = KrylovSolver(A, b, M)

solver.add_measurements(10000000)
# p = np.dot(solver.basis, np.sqrt(solver.samples_one / solver.measurements))

# for i in range(solver.M + 1):
#     plt.plot(xs, solver.basis[i](xs))
#     plt.show()

# plt.plot(xs, p(xs))
# plt.plot(A, abs(b), "*")
# plt.show()

# poly = solver.poly_minres_no_bound()
# scale = 1
poly, scale = solver.poly_inv(7)
# print(poly)
# plt.plot(xs, poly(xs))
# plt.show()

print(poly(A) * b)
print(b / A)
print(np.linalg.norm(poly(A) * b - b / A))

def poly_qsvt():
    epsilon = 0.5
    b = int(kappa**2 * np.log(kappa / epsilon))
    # j0 = int(np.sqrt(b * np.log(4 * b / epsilon)))
    j0 = poly.degree() // 2

    g = np.polynomial.chebyshev.Chebyshev([0])
    for j in range(j0 + 1):
        gcoef = scipy.special.bdtrc(b + j, 2 * b, 0.5)
        deg = 2 * j + 1
        g += (-1)**j * gcoef * \
            np.polynomial.chebyshev.Chebyshev([0] * deg + [1])
    g = 4 * g
    return g

p_qsvt = poly_qsvt()
print(p_qsvt.degree(), poly.degree())
print(np.linalg.norm(p_qsvt(A) * b - b / A))

xs = np.linspace(-1, 1, 200)
plt.plot(xs, poly(xs))
plt.plot(xs, p_qsvt(xs))
plt.plot(A, abs(b), ".")
plt.show()
