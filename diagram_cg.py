import numpy as np
import matplotlib.pyplot as plt

from simulator import NoiseModel
from solver import KrylovSolver, StationarySolver
from util import generate_noise_flips

np.polynomial.set_default_printstyle("ascii")

dim = 10
kappa = 10
steps = 2
noise_flips_per_problem = 20
noise = 0.2
samples = 10000

seed = np.random.SeedSequence(43)
seed_noise = np.random.SeedSequence(3)
rng = np.random.default_rng(seed.spawn(1)[0])
equation = rng.normal(size=dim).reshape((dim))
equation /= np.linalg.norm(equation)
equation2 = rng.uniform(1 / kappa, 1.0, dim)
equation2[0] = 1 / kappa
noise_flips = generate_noise_flips(rng, dim, noise_flips_per_problem)
print(np.array([equation2, 1 / equation2, equation]).T)
print(equation)
poly = np.polynomial.Chebyshev.fit(equation2, 1 / equation2, dim - 1, domain=[-1, 1])
print(poly.convert(kind=np.polynomial.Polynomial, domain=[0, 1]))

poly_best = np.polynomial.Chebyshev.fit(
    equation2, 1 / equation2, steps, w=equation, domain=[-1, 1]
)
print(poly_best.convert(kind=np.polynomial.Polynomial, domain=[0, 1]))

# Exact
A = NoiseModel(seed_noise, equation, noise_flips, kappa, 0)
A.S[:] = equation2
solver = KrylovSolver(steps, default_samples=np.inf, transform_method="square")
evs_exact = solver.estimate_eigenvalues(A)
poly_exact = solver.compute_polynomial(A)
print(np.array([evs_exact, 1/evs_exact]).T)
print(poly_exact.convert(kind=np.polynomial.Polynomial))

# Chebyshev iteration
solver = StationarySolver(steps, poly_kind="chebyshev_positive", default_samples=np.inf)
evs_cheb = 1 / kappa + (
    1 + np.cos((np.arange(0, steps + 1) + 0.5) * np.pi / (steps + 1))
) / 2 * (1 - 1 / kappa)
poly_cheb = solver.compute_polynomial(A)
print(np.array([evs_cheb, 1/evs_cheb]).T)
print(poly_cheb.convert(kind=np.polynomial.Polynomial))


# Inexact
A = NoiseModel(seed_noise, equation, noise_flips, kappa, noise)
A.S[:] = equation2
solver = KrylovSolver(steps, default_samples=samples, transform_method="square")
evs_inexact = solver.estimate_eigenvalues(A)
poly_inexact = np.polynomial.Chebyshev.fit(
    evs_inexact, 1 / evs_inexact, len(evs_inexact) - 1, domain=[-1, 1]
)
print(np.array([evs_inexact, 1/evs_inexact]).T)
print(poly_inexact.convert(kind=np.polynomial.Polynomial))

xs = np.linspace(0, 1, 100)
plt.ylim([0, kappa + 1])
plt.bar(equation2, kappa * np.abs(equation), width=0.005)
# plt.plot(xs, poly(xs), "b-")
plt.plot(xs, poly_best(xs), "k-")
plt.plot(evs_exact, 1 / evs_exact, "rx")
plt.plot(xs, poly_exact(xs), "r-")
plt.plot(evs_cheb, 1 / evs_cheb, "yx")
plt.plot(xs, poly_cheb(xs), "y-")
plt.plot(evs_inexact, 1 / evs_inexact, "gx")
plt.plot(xs, poly_inexact(xs), "g-")
plt.show()
