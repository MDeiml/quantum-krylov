from collections.abc import Callable
from typing import Concatenate
import itertools
import numpy as np
from block_encoding_model import BlockEncodingModel
from util import generate_noise_flips
from tqdm import tqdm
import git
import os.path
import multiprocessing

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class Runner:
    """
    Helper for testing solvers with many combinations of parameters.

    Will write a ``.csv`` file tagged with the hash of the current git commit.
    Contains the parameters for the equation, the parameters for the solver, and
    the requested error percentiles.

    :param params_problem:
        Dictionary, with keys corresponding to parameters to
        `BlockEncodingModel`, and values as lists of parameters to try.
    :param tries:
        The number of equations to try each setup with. Will test each solver
        with the same equations, and will simulate each solver for each equation
        once.
    :param error_percentiles:
        What error quantities to output into the ``.csv`` file.
    :param dim:
        The dimension of each equation
    """

    def __init__(
        self,
        params_problem: dict,
        tries: int = 200,
        error_percentiles: list[float] = [0, 5, 50, 95, 100],
        condition_bound_inaccuaracy=0.1,
        dim: int = 128,
        num_processes: int | None = None,
    ):
        self.tries = tries
        self.error_percentiles = error_percentiles
        self.dim = dim
        self.num_processes = min(
            num_processes if num_processes is not None else multiprocessing.cpu_count(),
            tries,
        )

        print("Checking git hash...")
        repo = git.Repo(search_parent_directories=True)
        self.commit_hash = repo.head.object.hexsha
        if repo.is_dirty():
            print(
                "Consider creating a git commit before running to ensure proper tagging"
            )
            self.commit_hash += "_modified"

        print("Generating problems...")
        seed_sequence = np.random.SeedSequence(0)
        global_rng = np.random.default_rng(seed_sequence.spawn(1)[0])

        # Generate bs and Ds
        kappas = params_problem["kappa"]
        del params_problem["kappa"]
        Ds = np.zeros((len(kappas), tries, dim))
        bs = np.zeros((len(kappas), tries, dim))
        for i, kappa in enumerate(kappas):
            smin = global_rng.uniform(
                1 / kappa, 1 / kappa / (1 - condition_bound_inaccuaracy), tries
            )
            Ds[i] = global_rng.uniform(smin[:, np.newaxis], 1, (tries, dim))
            bs[i] = global_rng.normal(size=(tries, dim))
            for j in range(tries):
                norm = np.linalg.norm(bs[i, j])
                assert norm > 1e-4
                bs[i, j] /= norm

        noise_flips_per_problem = 20
        noise_flips = generate_noise_flips(
            global_rng, dim, len(kappas) * tries * noise_flips_per_problem
        )
        noise_flips = noise_flips.reshape(
            (len(kappas), tries, noise_flips_per_problem, 2 * dim, 2 * dim)
        )

        self.params_problem_names = list(params_problem.keys()) + ["kappa"]
        params_problem = list(
            dict(zip(params_problem.keys(), x))
            for x in itertools.product(*params_problem.values())
        )

        seeds = [seed_sequence.spawn(tries) for _ in kappas]
        self.problems = [
            (
                params | {"kappa": kappa},
                [
                    BlockEncodingModel(
                        D, b, kappa=kappa, noise_flips=nf, seed=seed, **params
                    )
                    for seed, D, b, nf in zip(seeds[i], Ds[i], bs[i], noise_flips[i])
                    for i, kappa in enumerate(kappas)
                ],
            )
            for params in params_problem
        ]

    def test_solver(
        self,
        solver: Callable[Concatenate[BlockEncodingModel, ...], np.polynomial.Chebyshev],
        name: str,
        params_solver: dict,
    ):
        """
        Test the given solver.

        The output will be stored in the file ``results/{name}_{commit
        hash}.csv``, or ``results/{name}_{commit hash}_modified.csv`` if there
        are uncommited changes.

        :param solver:
            Function taking an equation $Ax = b$ represented by a
            `BlockEncodingModel` as its first argument, and returning a
            polynomial $p$ of type `np.polynomial.Chebshev` such that $p(A)b
            \\approx A^{-1}b$.
        :param name:
            Name for the ``.csv`` file
        :param params_solver:
            Dictionary, with keys corresponding to the names of all but the
            first parameter to `solver`, and values as lists of parameters to
            try.
        """
        params_solver_names = list(params_solver.keys())

        iter = (
            dict(zip(params_solver.keys(), x))
            for x in itertools.product(*params_solver.values())
        )

        total = 1
        for param in params_solver.values():
            total *= len(param)

        filename = f"results/{name}_{self.commit_hash}.csv"
        was_created = not os.path.isfile(filename)

        with open(filename, "a") as f, multiprocessing.Pool(self.num_processes) as pool:
            if was_created:
                f.write(
                    ";".join(
                        params_solver_names
                        + self.params_problem_names
                        + ["complexity"]
                        + [f"error {p} percentile" for p in self.error_percentiles]
                    )
                    + "\n"
                )
                f.flush()

            tqdm_iter = tqdm(iter, total=total, desc=f"Testing {name}")
            for params in tqdm_iter:
                for problem_params, subproblems in self.problems:
                    params_string = ", ".join(
                        [f"{k}={v}" for k, v in (params | problem_params).items()]
                    )
                    tqdm_iter.set_description(f"Testing {name}({params_string})")
                    errors = []
                    complexities = []
                    results = pool.map(
                        _process_solver,
                        zip(subproblems, itertools.repeat((solver, params))),
                    )
                    errors, subproblems[:] = zip(*results)
                    complexities = [p.complexity() for p in subproblems]
                    res = (
                        list(params.values())
                        + list(problem_params.values())
                        + [np.average(complexities)]
                        + [np.percentile(errors, p) for p in self.error_percentiles]
                    )
                    f.write(";".join(list(map(str, res))) + "\n")
                    f.flush()


def _process_solver(p):
    A, (solver, params) = p
    A.reset()
    poly = solver(A, **params)
    if params["square"]:
        X = np.polynomial.Chebyshev([0, 1])
        poly = poly(X * X)
    error = A.estimate_error(
        poly,
        params["samples"],
        params["square"],
    )
    return error, A
