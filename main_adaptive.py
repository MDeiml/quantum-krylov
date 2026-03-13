from runner import Runner
import adaptive_solver

params_problem = {
    "noise": [0.0, 0.01, 0.02],
    "kappa": [3],
}

runner = Runner(
    params_problem,
    tries=4,
    error_percentiles=[0, 50, 100],
)

runner.test_solver(
    adaptive_solver.compute_polynomial,
    "adaptive",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "square": [False, True],
        "sup_norm_constraint": [False, True],
    },
)
