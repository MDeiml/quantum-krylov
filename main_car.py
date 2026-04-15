from runner import Runner
from car_solver import car_solver

params_problem = {
    "noise": [0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    "kappa": [3, 5],
}

runner = Runner(params_problem)

runner.test_solver(
    car_solver,
    "car",
    {
        "steps": range(1, 17),
        "samples": [10000, 40000, 160000],
        "transform": [None, "square_outer", "square"],
        "adaptive": [True, False],
    },
)
