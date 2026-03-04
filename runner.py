import numpy as np
from block_encoding_model import BlockEncodingModel
from util import generate_noise_flips
from tqdm import tqdm
import itertools
import git
import os.path


class Runner:
    def __init__(
        self,
        params_problem,
        tries=1000,
        error_percentiles=[0, 5, 50, 95, 100],
        condition=10,
        dim=128,
        enforce_git_commit=True,
    ):
        self.tries = tries
        self.error_percentiles = error_percentiles
        self.dim = dim
        self.condition = condition

        print("Checking git hash...")
        repo = git.Repo(search_parent_directories=True)
        if enforce_git_commit:
            assert not repo.is_dirty(), (
                "Create a git commit before running to ensure proper tagging"
            )
        self.commit_hash = repo.head.object.hexsha
        if repo.is_dirty():
            self.commit_hash += "_modified"

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
                    BlockEncodingModel(seed, equation, nf, condition, **params)
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
                    ";".join(
                        params_solver_names
                        + ["condition"]
                        + self.params_problem_names
                        + ["complexity"]
                        + [f"error {p} percentile" for p in self.error_percentiles]
                    )
                    + ";\n"
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
                        + [self.condition]
                        + list(problem_params.values())
                        + [np.average(complexities)]
                        + [np.percentile(errors, p) for p in self.error_percentiles]
                    )
                    f.write(";".join(list(map(str, res))) + ";\n")
                    f.flush()
