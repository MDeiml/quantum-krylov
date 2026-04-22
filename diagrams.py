import numpy as np
from cap_solver import cap_solver, maximum_entropy_measure, estimate_moments
from semi_iterative_solver import semi_iterative_solver
from runner import Runner

np.polynomial.set_default_printstyle("ascii")

problem_params = {
    "kappa": [10],
    "num_clusters": [3],
    "noise": [0],
}
steps = 3
samples = 160000

dim = 20
runner = Runner(problem_params, dim=dim, tries=1, seed=1)
A = runner.problems[0][1][0]

eig_data = np.array([A.D, 1/A.D, A.b])
np.savetxt("eigs.csv", eig_data.T, newline="\\\\\n")

def print_poly(p):
    print(str(p.convert(kind=np.polynomial.Polynomial, domain=[0, 1])).replace("**", "^").replace("(", "*(").replace("x", "*x"))

print("optimal, cg, cheb, q_cheb, cup, cap, rho")

poly_best = np.polynomial.Chebyshev.fit(A.D, 1 / A.D, steps, w=A.b, domain=[-1, 1])
print_poly(poly_best)

# CG
A.reset()
poly_cg = cap_solver(A, steps, np.inf)
print_poly(poly_cg)

# Chebyshev iteration
A.reset()
poly_cheb = semi_iterative_solver(A, steps, np.inf, poly_kind="cheb")
print_poly(poly_cheb)

A.reset()
poly_qcheb = semi_iterative_solver(A, steps // 2, np.inf, poly_kind="q_cheb")
print_poly(poly_qcheb)

A.reset()
poly_cup = cap_solver(A, steps, samples, adaptive=False)
print_poly(poly_cup)

A.reset()
poly_cap = cap_solver(A, steps, samples, adaptive=True)
print_poly(poly_cap)

A.reset()
moments = estimate_moments(A, steps, samples, False)
rho_poly = maximum_entropy_measure(moments, A.kappa, False)
print_poly(rho_poly)
