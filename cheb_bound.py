import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def cut_norm(A):
    for z
def maximum(P: Chebyshev):
    extrema = np.concatenate((P.deriv().roots(), [-1, 1]))
    return np.max(np.abs(P(extrema)))


def poly_linf_div_coeff_l1(coeff):
    coeff = np.concatenate((coeff, [1]))
    poly = Chebyshev(coeff)
    return maximum(poly) / np.linalg.norm(coeff, ord=1)

xs = np.linspace(-1, 1, 100)

val = []

for N in range(100, 101):
    largest_C = 0
    largest_coeff = None
    countdown = 300
    while countdown > 0:
        rand = np.random.uniform(size=N, low=-1)
        res = minimize(poly_linf_div_coeff_l1, rand, method="Nelder-Mead")
        if 1/res.fun > largest_C:
            countdown = max(min(300, int(-20 * np.log2(1/res.fun - largest_C))), countdown)
            largest_C = 1/res.fun
            largest_coeff = res.x
            print(largest_C)
        countdown -= 1
    coeff = np.concatenate((largest_coeff, [1]))
    poly = Chebyshev(coeff)
    val.append(largest_C)
    print(f"N = {N} C = {largest_C}")
    plt.plot(xs, poly(xs), label=f"{N}")

plt.figure()
plt.semilogx(val)

plt.show()
