import numpy as np
from sampling import NoiseModel
from util import generate_noise_flips
from tqdm import tqdm
import itertools
import git
import os.path


class Runner:
    def __init__(
        self, params_problem, tries=1000, error_percentile=95, kappa=10, dim=128
    ):
        self.tries = tries
        self.error_percentile = error_percentile
        self.dim = dim

        print("Checking git hash...")
        repo = git.Repo(search_parent_directories=True)
        assert not repo.is_dirty(), (
            "Create a git commit before running to ensure proper tagging"
        )
        self.commit_hash = repo.head.object.hexsha

        print("Generating problems...")
        seed_sequence = np.random.SeedSequence(0)
        global_rng = np.random.default_rng(seed_sequence.spawn(1)[0])

        self.params_problem_names = list(params_problem.keys())
        params_problem = list(
            dict(zip(params_problem.keys(), x))
            for x in itertools.product(*params_problem.values())
        )

        # Generate bs and ms
        equations = global_rng.normal(size=tries * dim).reshape((tries, dim))
        for i in range(equations.shape[0]):
            norm = np.linalg.norm(equations[i])
            assert norm > 1e-4
            equations[i] /= norm
        equations = equations.reshape((tries, dim))

        noise_flips_per_problem = 20
        noise_flips = generate_noise_flips(
            global_rng, dim, tries * noise_flips_per_problem
        )
        noise_flips = noise_flips.reshape(
            (tries, noise_flips_per_problem, 2 * dim, 2 * dim)
        )

        seeds = seed_sequence.spawn(tries)
        self.problems = [
            (
                params,
                [
                    NoiseModel(seed, equation, nf, kappa, **params)
                    for (seed, equation, nf) in zip(seeds, equations, noise_flips)
                ],
            )
            for params in params_problem
        ]

    def run_test_for_solver(self, solver_class, params_solver):
        params_solver_names = list(params_solver.keys())

        iter = (
            dict(zip(params_solver.keys(), x))
            for x in itertools.product(*params_solver.values())
        )

        total = 1
        for param in params_solver.values():
            total *= len(param)

        filename = f"results/{solver_class.__name__}_{self.commit_hash}.csv"
        was_created = not os.path.isfile("filename")

        with open(filename, "a") as f:
            if was_created:
                f.write(
                    ";".join(params_solver_names + self.params_problem_names)
                    + f";complexity;error {self.error_percentile} percentile;\n"
                )
                f.flush()

            for params in tqdm(
                iter, total=total, desc=f"Testing {solver_class.__name__}"
            ):
                solver = solver_class(**params)
                for problem_params, subproblems in self.problems:
                    errors = []
                    complexities = []
                    for A in subproblems:
                        A.reset()
                        poly = solver.compute_polynomial(A)
                        if solver.transform_method == "square":
                            X = np.polynomial.Chebyshev([0, 1])
                            poly = poly(X * X)
                        error = A.estimate_error(
                            poly,
                            solver.default_samples,
                            solver.transform_method == "square",
                        )
                        errors.append(error)
                        complexities.append(A.complexity())
                    res = (
                        list(params.values())
                        + list(problem_params.values())
                        + [
                            np.average(complexities),
                            np.percentile(errors, self.error_percentile),
                        ]
                    )
                    f.write(";".join(list(map(str, res))) + ";\n")
                    f.flush()
