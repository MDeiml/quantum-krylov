import numpy as np
from symqsp import compute_angles
from simulator import Simulator
from util import measurement_max_error


class BlockEncodingModel:
    def __init__(
        self,
        seed: np.random.SeedSequence,
        b,
        noise_flips,
        kappa,
        noise=0.01,
    ):
        """
        Wrapper to simulate quantum device

        :param A:
            Matrix (2d) or singular values (1d) or condition number (scalar)
            number (scalar)
        """
        self.kappa = kappa

        [self.general_rng_seed, self.noise_rng_seed] = seed.spawn(2)

        n = b.shape[0]
        self.S = np.linspace(1 / kappa, 1, n)

        self.b = b

        self.noise = noise
        self.noise_flips = noise_flips

        self.simulator = Simulator(seed, noise_flips, noise)

    def reset(self):
        self.simulator.reset()

    def complexity(self):
        return self.simulator.complexity()

    def estimate_error(self, poly, samples, root=False):
        S = self.S
        if root:
            S = np.sqrt(self.S)

        exact_solution = self.b / self.S
        exact_solution_norm = np.linalg.norm(exact_solution)

        # Find the quantity of interest that maximizes the error
        approximation = poly(S) * self.b
        measurement = measurement_max_error(exact_solution, approximation)
        qoi_exact = exact_solution.T @ measurement @ exact_solution
        qoi_approximation = approximation.T @ measurement @ approximation

        if np.isinf(samples):
            self.simulator.calls = np.inf
            return abs(qoi_exact - qoi_approximation) / exact_solution_norm**2

        # If the polynomial is zero, we can conlude that the result must be
        # zero without using the quantum computer
        if np.allclose(poly.coef, 0, atol=1e-6):
            return abs(qoi_exact) / exact_solution_norm

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

        def measure(x, temp):
            x = x[:, :, 0, :]
            for i, weight in enumerate(weights):
                x[:, i, :] *= weight
            np.sum(x[:, :, :], axis=1, out=x[:, 0, :])
            x = x[:, 0, :]
            np.matvec(measurement, x, out=temp[:, 0, 0, :])
            return np.vecdot(x.real, temp[:, 0, 0, :].real)

        results = self.simulator.simulate_qsvt(
            S,
            self.b,
            [angles for _, angles, _ in angle_sequences],
            samples,
            measure,
            weights=weights,
            test=qoi_approximation / total_normalization**2,
        )
        qoi_measured = total_normalization**2 * results

        return abs(qoi_exact - qoi_measured) / exact_solution_norm**2

    def estimate_poly(self, poly, samples, root=False):
        """
        Simulate estimation of b^T poly(A) b

        :param poly: The matrix polynomial to be computed
        :param samples: The number of simulated samples. Can be np.inf
        :param root: If True, compute poly(sqrt(A)) instead
        """

        S = self.S
        if root:
            S = np.sqrt(self.S)

        exact = np.dot(self.b, poly(S) * self.b)
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
            S, self.b, angles, samples, test=exact / normalization
        )

        return normalization * result
