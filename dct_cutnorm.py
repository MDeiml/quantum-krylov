import numpy as np
import matplotlib.pyplot as plt
import itertools


def cutnorm_brute_force(A):
    m, n = A.shape
    max_cut_sum = 0.0
    maximizer = None

    # Iterate through all 2^m subsets of rows (I)
    for col_signs in itertools.product([-1, 1], repeat=m):
        col_signs = np.array(col_signs)
        signed_row_sums = A @ col_signs
        current_sum = np.sum(np.abs(signed_row_sums))
        if max_cut_sum < current_sum:
            max_cut_sum = current_sum
            maximizer = col_signs

    return max_cut_sum, maximizer


def cutnorm_random(A, tries):
    m, n = A.shape
    max_cut_sum = 0.0
    maximizer = None

    # Iterate through all 2^m subsets of rows (I)
    for i in range(tries):
        col_signs = np.random.choice([-1, 1], size=n)
        signed_row_sums = A @ col_signs
        current_sum = np.sum(np.abs(signed_row_sums))
        if max_cut_sum < current_sum:
            max_cut_sum = current_sum
            maximizer = col_signs

    return max_cut_sum, maximizer

def legendre(n):
    cheb = np.polynomial.Chebyshev([0] * (n - 1) + [1]).deriv() * np.polynomial.Chebyshev([1, 0, -1])
    roots = cheb.roots()
    polys = []
    for i, root in enumerate(roots):
        poly = cheb // np.polynomial.Chebyshev([root, -1])
        poly /= poly(root)
        polys.append(poly)
    return np.array(polys)

xs = np.linspace(-1, 1, 400)

N = 10

results = np.zeros(N)
test = np.zeros(N)

for n in range(2, N + 2):
    # Compute DCT Matrix
    indices = np.arange(n)
    j, k = np.meshgrid(indices, indices)
    DCT = np.cos(j * k / (n - 1) * np.pi)
    DCT[:, 0] /= 2
    DCT[:, -1] /= 2

    # Inverse DCT
    DCT_inv = DCT * 2 / (n - 1)
    np.testing.assert_allclose(DCT @ DCT_inv, np.eye(n), atol=1e-10)
    tries = n * 100
    if n <= 16:
        results[n - 2], signs = cutnorm_brute_force(DCT_inv)
        test[n-2] = results[n-2]
    else:
        results[n - 2], signs = cutnorm_random(DCT_inv, tries)
        test[n-2], _ = cutnorm_random(DCT_inv, tries * 2)
    polys = legendre(n)
    plt.plot(xs, np.dot(polys, signs)(xs))
    print(results[n-2])

plt.show()

xs = np.arange(2, N + 2)

fit_linear = np.polyfit(xs, results, 1)
fit_sqrt = np.polyfit(np.sqrt(xs), results, 1)
fit_log = np.polyfit(np.log(xs), results, 1)
fit_logsq = np.polyfit(np.log(xs) ** 2, results, 1)
plt.plot(xs, results, label="data")
plt.plot(xs, test, label="test")
plt.plot(xs ,fit_linear[1] + fit_linear[0] * xs, label="linear")
plt.plot(xs ,fit_sqrt[1] + fit_sqrt[0] * np.sqrt(xs), label="sqrt")
plt.plot(xs, fit_log[1] + fit_log[0] * np.log(xs), label="log")
plt.plot(xs, fit_logsq[1] + fit_logsq[0] * np.log(xs) ** 2, label="log^2")
plt.legend(loc="upper left")
plt.show()
