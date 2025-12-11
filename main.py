import numpy as np
from runner import Runner
from solver import KrylovSolver, StationarySolver

params_problem = {
    "noise": np.array([0.001]),
}

runner = Runner(params_problem)

runner.run_test_for_solver(
    StationarySolver,
    {
        "steps": np.arange(1, 17),
        "eval_method": ["qsp"],
        "default_samples": np.array([10000]),
        "transform_method": [None, "square"],
        # "transform_method": [None],
        "poly_kind": ["qsvt", "chebyshev_positive"],
        # "poly_kind": ["qsvt"],
    },
)
runner.run_test_for_solver(
    KrylovSolver,
    {
        "steps": np.arange(1, 13),
        # "poly_kind": ["monomial", "chebyshev"],
        "poly_kind": ["chebyshev"],
        "default_samples": np.array([10000]),
        "loss_type": ["cg", "minres"],
        "inf_constraint": [False, True],
        "transform_method": [None, "square"],
        # "use_qsp_for_qoi": [False, True],
        "use_qsp_for_qoi": [True],
    },
)
