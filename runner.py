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
        dim: int = 128,
        num_processes: int | None = None,
        seed: int = 0,
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
        seed_sequence = np.random.SeedSequence(seed)
        problem_seed, noise_flip_seed = seed_sequence.spawn(2)

        # Generate bs and Ds
        kappas = params_problem["kappa"]
        num_clusters_list = params_problem["num_clusters"]
        del params_problem["kappa"]
        del params_problem["num_clusters"]
        Ds = np.zeros((len(kappas), len(num_clusters_list), tries, dim))
        bs = np.zeros((len(kappas), len(num_clusters_list), tries, dim))
        for i, kappa in enumerate(kappas):
            for j, num_clusters in enumerate(num_clusters_list):
                problem_rng = np.random.default_rng(problem_seed)
                if num_clusters is not None:
                    cluster_size = 0.1
                    clusters = problem_rng.uniform(1 / kappa, 1, (tries, num_clusters))
                    cluster_index = problem_rng.integers(
                        0, num_clusters, size=(tries, dim)
                    )
                    Ds[i, j] = problem_rng.normal(
                        0,
                        cluster_size * (1 - 1 / kappa) / num_clusters,
                        size=(tries, dim),
                    )
                    Ds[i, j] += np.take_along_axis(clusters, cluster_index, axis=1)
                    Ds[i, j] = np.maximum(1 / kappa, np.minimum(1, Ds[i, j]))
                else:
                    Ds[i, j] = problem_rng.uniform(1 / kappa, 1, (tries, dim))
                bs[i, j] = problem_rng.normal(size=(tries, dim))
                for k in range(tries):
                    norm = np.linalg.norm(bs[i, j, k])
                    assert norm > 1e-4
                    bs[i, j, k] /= norm

        noise_flip_rng = np.random.default_rng(noise_flip_seed)
        noise_flips_per_problem = 20
        noise_flips = generate_noise_flips(
            noise_flip_rng,
            dim,
            len(kappas) * len(num_clusters_list) * tries * noise_flips_per_problem,
        )
        noise_flips = noise_flips.reshape(
            (
                len(kappas),
                len(num_clusters_list),
                tries,
                noise_flips_per_problem,
                2 * dim,
                2 * dim,
            )
        )

        self.params_problem_names = list(params_problem.keys()) + [
            "kappa",
            "num_clusters",
        ]
        params_problem = list(
            dict(zip(params_problem.keys(), x))
            for x in itertools.product(*params_problem.values())
        )

        seeds = seed_sequence.spawn(tries)
        self.problems = [
            (
                params | {"kappa": kappa, "num_clusters": num_clusters},
                [
                    BlockEncodingModel(
                        D, b, kappa=kappa, noise_flips=nf, seed=seed, **params
                    )
                    for seed, D, b, nf in zip(
                        seeds, Ds[i, j], bs[i, j], noise_flips[i, j]
                    )
                ],
            )
            for params in params_problem
            for i, kappa in enumerate(kappas)
            for j, num_clusters in enumerate(num_clusters_list)
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
    if params["transform"] == "square":
        X = np.polynomial.Chebyshev([0, 1])
        poly = poly(X * X)
    if params["transform"] == "square_outer":
        poly = poly**2
    error = A.estimate_error(
        poly,
        params["samples"],
        params["transform"] in ["square", "square_outer"],
    )
    return error, A
