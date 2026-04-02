from runner import Runner
from semi_iterative_solver import semi_iterative_solver

params_problem = {
    "noise": [0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    "kappa": [3, 5],
}

runner = Runner(params_problem)

runner.test_solver(
    semi_iterative_solver,
    "semi_iterative",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "transform": [None, "square"],
        "poly_kind": ["qsvt", "cheb", "q_cheb"],
    },
)

runner.test_solver(
    semi_iterative_solver,
    "semi_iterative",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "transform": ["square_outer"],
        "poly_kind": ["qsvt", "q_cheb"],
    },
)
