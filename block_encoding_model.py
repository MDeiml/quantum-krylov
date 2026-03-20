import numpy as np
from symqsp import compute_angles
from simulator import Simulator
from util import generate_noise_flips


class BlockEncodingModel:
    """
    Helper to simulate quantum computation involving a linear system $Dx = b$ where $D$ is diagonal.

    :param D:
        The entries of the diagonal matrix $S$
    :param b:
        The right hand side of the linear system
    :param noise:
        Value between 0 and 1. This is the probability that a random flip will
        be applied after an application of a block encoding.
    :param noise_flips:
        Unitary matrices, corresponding to the possible noise that can be
        introduced after each application of $A$. Specifically, if noise is
        applied, the state is multiplied with a random element of `noise_flips`.
        Should have the shape ``(M, N, N)`` where ``N = b.shape[0]``. Use
        `util.generate_noise_flips` to generate these matrices.
    :param seed:
        Seed for the random generators used for noise and sampling.
    """

    def __init__(
        self,
        D: np.ndarray,
        b: np.ndarray,
        kappa: float | None = None,
        noise: float = 0.01,
        noise_flips: np.ndarray | None = None,
        seed: np.random.SeedSequence | None = None,
    ):
        assert b.ndim == 1, "b should be a vector"
        assert np.isclose(np.linalg.norm(b), 1, atol=1e-8), "b should be normalized"
        self.b = b

        assert b.shape == D.shape, "D should be diagonal and of the same size as b"
        smin = np.min(D)
        assert smin > 0, "D must be positive definite"
        self.kappa = kappa if kappa is not None else 1 / smin
        self.D = D

        if seed is None:
            seed = np.random.SeedSequence(0)
        self.simulator = Simulator(seed, noise)

        if noise_flips is None:
            noise_flips = generate_noise_flips(
                np.random.default_rng(np.random.SeedSequence(43)), b.shape[0], 20
            )
        self.noise_flips = noise_flips

    def reset(self):
        """
        Reset the random generators and the complexity counter
        """
        self.simulator.reset()

    def complexity(self):
        """
        The number of times a block encoding was applied

        This is counting since this object was created or the last call to
        `reset`.
        """
        return self.simulator.complexity()

    def estimate_error(
        self, poly: np.polynomial.Chebyshev, samples: int, root: bool = False
    ):
        """
        Estimate the relative error of $p(A)b - A^{-1}b$ as specified in the
        paper.

        :param poly:
            The polynomial $p$ approximating $1 / x$.
        :param samples:
            The number of samples that should be used to measure the observable.
            Can be `np.inf` to turn of noise and sampling error.
        :param root:
            If ``True``, compute the error of $p(B)b - A^{-1}b$ instead where $B
            = A^{1/2}$.
        """
        D = self.D
        if root:
            D = np.sqrt(self.D)

        exact_solution = self.b / self.D
        exact_solution_norm = np.linalg.norm(exact_solution)

        # Find the quantity of interest that maximizes the error
        approximation = poly(D) * self.b

        if np.isinf(samples):
            self.simulator.calls = np.inf
            diff = exact_solution - approximation
            return np.dot(diff, diff) / exact_solution_norm**2

        # If the polynomial is zero, we can conlude that the result must be
        # zero without using the quantum computer
        if np.allclose(poly.coef, 0, atol=1e-6):
            return 1

        angle_sequences = compute_angles(poly)

        if root:
            assert all(
                [subpoly.degree() % 2 == 0 for subpoly, _, _ in angle_sequences]
            ), "Can only evaluate even polynomials for squareroot of matrix"

        total_normalization = sum(
            [normalization for _, _, normalization in angle_sequences]
        )

        weights = [
            np.sqrt(normalization / total_normalization)
            for _, _, normalization in angle_sequences
        ]

        # Measure xx^T and Id - xx^T
        def measure(x, temp):
            x = x[:, :, 0, :]
            for i, weight in enumerate(weights):
                x[:, i, :] *= weight
            np.sum(x[:, :, :], axis=1, out=x[:, 0, :])
            x = x[:, 0, :]
            return np.stack(
                (
                    np.vecdot(x.real, exact_solution / exact_solution_norm) ** 2,
                    np.vecdot(x.real, x.real)
                    - np.vecdot(x.real, exact_solution / exact_solution_norm) ** 2,
                ),
                axis=-1,
            )

        result = self.simulator.simulate_qsvt(
            D,
            self.b,
            [angles for _, angles, _ in angle_sequences],
            samples,
            measure,
            self.noise_flips,
            weights=weights,
            reference=np.array(
                [
                    np.dot(
                        approximation / total_normalization,
                        exact_solution / exact_solution_norm,
                    )
                    ** 2,  # axial norm
                    np.dot(
                        approximation / total_normalization,
                        approximation / total_normalization,
                    )
                    - np.dot(
                        approximation / total_normalization,
                        exact_solution / exact_solution_norm,
                    )
                    ** 2,  # orthogonal norm
                ]
            ),
        )
        result *= total_normalization**2
        result_axial_sq, result_orthogonal_sq = result

        # Assume, that approximation at least points in the correct direction,
        # so np.dot(approximation, exact_solution) is positive
        error_sq = (np.sqrt(result_axial_sq) - exact_solution_norm) ** 2 + np.abs(
            result_orthogonal_sq
        )

        return error_sq / exact_solution_norm**2

    def estimate_poly(
        self, poly: np.polynomial.Chebyshev, samples: int, root: bool = False
    ):
        """
        Simulate estimation of $b^T poly(A) b$

        :param poly:
            The matrix polynomial to be computed
        :param samples:
            The number of simulated samples. Can be `np.inf` to turn of noise
            and sampling error.
        :param root:
            If True, compute poly(sqrt(A)) instead
        """

        D = self.D
        if root:
            D = np.sqrt(self.D)

        exact = np.dot(self.b, poly(D) * self.b)
        # TODO: Also consider applications of b so that samples = inf makes sense
        if np.isinf(samples):
            self.simulator.calls = np.inf
            return exact

        angle_sequences = compute_angles(poly)

        assert len(angle_sequences) == 1, "not implemented"

        result = 0
        subpoly, angles, normalization = angle_sequences[0]
        if root:
            assert subpoly.degree() % 2 == 0, (
                "Can only evaluate even polynomials for squareroot of matrix"
            )

        result = self.simulator.simulate_qsvt_folding(
            D,
            self.b,
            angles,
            samples,
            self.noise_flips,
            reference=exact / normalization,
        )

        return normalization * result
