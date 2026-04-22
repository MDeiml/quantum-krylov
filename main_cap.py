from runner import Runner
from cap_solver import cap_solver

params_problem = {
    "noise": [0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    "kappa": [3, 5],
    "num_clusters": [None, 4],
}

runner = Runner(params_problem)

runner.test_solver(
    cap_solver,
    "cap",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "transform": [None, "square_outer", "square"],
        "adaptive": [True, False],
    },
)
