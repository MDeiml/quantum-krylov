from runner import Runner
import static_solver

params_problem = {
    "noise": [0, 0.01, 0.02],
    "kappa": [3],
}

runner = Runner(params_problem, tries=1, error_percentiles=[0, 50, 100])

runner.test_solver(
    static_solver.compute_polynomial,
    "static",
    {
        # "steps": range(1, 17),
        "steps": [3, 5],
        "samples": [10000, 40000, 160000],
        "square": [False, True],
        # "poly_kind": ["qsvt", "chebyshev_symmetric", "chebyshev_positive"],
        "poly_kind": ["chebyshev_symmetric"],
    },
)
