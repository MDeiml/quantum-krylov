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
        if np.isinf(samples):
            self.simulator.calls = np.inf
            return (
                abs(
                    exact_solution_norm**2
                    - np.dot(poly(S) * self.b, exact_solution / exact_solution_norm)
                    ** 2
                )
                / exact_solution_norm**2
            )

        angle_sequences = compute_angles(poly)
        results = np.zeros((samples, self.b.shape[0]))
        total_normalization = 0

        # Simulate LCU
        for subpoly, angles, normalization in angle_sequences:
            if root:
                assert subpoly.degree() % 2 == 0, (
                    "Can only evaluate even polynomials for squareroot of matrix"
                )

            states = self.simulator.simulate_qsvt(S, self.b, angles, samples, folding=False)

            results += states[:, : self.b.shape[0]].real * normalization
            total_normalization += normalization

        # Find the quantity of interest that maximizes the error
        approximation = np.average(results, axis=0)
        measurement = measurement_max_error(exact_solution, approximation)
        qoi_exact = exact_solution.T @ measurement @ exact_solution
        qoi_simulated = np.vecdot(results, results @ measurement)

        # Simulate sampling
        noisy_probabilities = (1 - qoi_simulated / total_normalization**2) / 2
        probability_estimate = 1 - 2 * self.simulator.measure(noisy_probabilities)
        qoi_measured = total_normalization**2 * probability_estimate

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

        # TODO: Also consider applications of b so that samples = inf makes sense
        if np.isinf(samples):
            self.simulator.calls = np.inf
            return np.dot(self.b, poly(S) * self.b)

        angle_sequences = compute_angles(poly)

        result = 0
        for subpoly, angles, normalization in angle_sequences:
            if root:
                assert subpoly.degree() % 2 == 0, (
                    "Can only evaluate even polynomials for squareroot of matrix"
                )

            states = self.simulator.simulate_qsvt(S, self.b, angles, samples, folding=True)
            dim = self.b.shape[0]

            # Simulate final hadamard
            states_one = (states[:, : 2 * dim] - states[:, 2 * dim :]) / np.sqrt(2)

            # Simulate sampling
            probabilities_one = np.vecdot(states_one, states_one).real

            # Estimate corresponding to the hadamard test
            probability_estimate = 1 - 2 * self.simulator.measure(probabilities_one)
            result += normalization * probability_estimate

        return result
