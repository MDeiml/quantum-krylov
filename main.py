import numpy as np
from sampling import NoiseModel
from util import generate_noise_flips
from solver import KrylovSolver, StationarySolver
from tqdm import tqdm
import itertools


tries = 1000
error_percentile = 95
kappa = 10
dim = 128

params_problem = {
    "noise": np.array([0.001]),
}
# params_problem = {
#     "coherent_noise": np.array([0.0]),
#     "decoherent_noise_per_iter": np.array([0.0]),
# }

params_problem_names = list(params_problem.keys())
params_problem = list(
    dict(zip(params_problem.keys(), x))
    for x in itertools.product(*params_problem.values())
)
global_rng = np.random.default_rng(0)

# Generate bs and ms
equations = global_rng.normal(size=tries * dim * 2).reshape((tries * 2, dim))
for i in range(equations.shape[0]):
    norm = np.linalg.norm(equations[i])
    assert norm > 1e-4
    equations[i] /= norm
equations = equations.reshape((tries, 2, dim))

noise_flips_per_problem = 20
noise_flips = generate_noise_flips(global_rng, dim, tries * noise_flips_per_problem)
noise_flips = noise_flips.reshape((tries, noise_flips_per_problem, 2 * dim, 2 * dim))

rngs = global_rng.spawn(tries)
problems = [
    (
        params,
        [
            NoiseModel(rng, equation[0], equation[1], nf, kappa, **params)
            for (rng, equation, nf) in zip(rngs, equations, noise_flips)
        ],
    )
    for params in params_problem
]


def run_test_for_solver(solver_class, params_solver):
    params_solver_names = list(params_solver.keys())

    iter = (
        dict(zip(params_solver.keys(), x))
        for x in itertools.product(*params_solver.values())
    )

    total = 1
    for param in params_solver.values():
        total *= len(param)

    with open(f"results_{solver_class.__name__}.csv", "w") as f:
        f.write(
            ";".join(params_solver_names + params_problem_names)
            + f";complexity;error {error_percentile} percentile;\n"
        )

        for params in tqdm(iter, total=total, desc=f"Testing {solver_class.__name__}"):
            solver = solver_class(**params)
            solver.precompute()
            for problem_params, subproblems in problems:
                avg_complexity = 0
                errors = []
                for A in subproblems:
                    A.reset_profiling()
                    reference = A.reference_solution()
                    solution = solver.solve(A)
                    error = np.abs(reference - solution)
                    errors.append(error)
                    avg_complexity += A.complexity() / tries
                res = (
                    list(params.values())
                    + list(problem_params.values())
                    + [avg_complexity, np.percentile(errors, error_percentile)]
                )
                f.write(";".join(list(map(str, res))) + ";\n")


# run_test_for_solver(
#     StationarySolver,
#     {
#         "steps": np.arange(1, 17),
#         "eval_method": ["qsp"],
#         "default_samples": np.array([10000]),
#         "transform_method": [None, "square"],
#         # "transform_method": [None],
#         "poly_kind": ["qsvt", "chebyshev_positive"],
#         # "poly_kind": ["qsvt"],
#     },
# )
run_test_for_solver(
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
