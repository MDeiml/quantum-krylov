from runner import Runner
import nonadaptive_solver

params_problem = {
    "noise": [0.02],
    "kappa": [3],
}

runner = Runner(params_problem, tries=4, error_percentiles=[0, 50, 100])

runner.test_solver(
    nonadaptive_solver.compute_polynomial,
    "nonadaptive",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "square": [False, True],
        "poly_kind": ["qsvt", "chebyshev_symmetric", "chebyshev_positive"],
    },
)
