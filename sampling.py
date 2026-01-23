import numpy as np
from util import sup_norm
from scipy.stats import poisson
from symqsp import compute_angles


class NoiseModel:
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
        self.solution = self.b / self.S

        self.noise = noise
        self.noise_flips = noise_flips
        self.reset()

    def reset(self):
        self.calls = 0
        # There are two different generators, to make sure that the total amount
        # of noise for two runs is the same if they use the same number of
        # applications of A and the same number of samples
        self.general_rng = np.random.default_rng(self.general_rng_seed)
        self.noise_rng = np.random.default_rng(self.noise_rng_seed)

    def complexity(self):
        return self.calls

    def compute_angles(self, poly):
        """
        Computes angles for given polynomial

        This takes into account the parity and normalization of the polynomial.
        It returns a list of one (if the polynomial has definite partiy) or two
        tuples. The first element of the tuples is an angle sequence for QSVT
        and the second element is the corresponding weight, such that the sum of
        QSVT polynomials equals the input to this function.
        """
        poly = poly.convert(kind=np.polynomial.Chebyshev)

        parity = poly.degree() % 2
        is_parity = np.allclose(poly.coef[(1 - parity) :: 2], 0, atol=1e-6)

        coefs_parity = poly.coef.copy()
        coefs_parity[1 - parity :: 2] = 0
        poly_parity = np.polynomial.Chebyshev(coefs_parity)
        polys = [poly_parity]
        if not is_parity:
            # TODO: Consider this and folding is complexity counting
            coefs_non_parity = poly.coef[:-1].copy()
            coefs_non_parity[parity::2] = 0
            poly_non_parity = np.polynomial.Chebyshev(coefs_non_parity)
            polys.append(poly_non_parity)

        result = []
        for subpoly in polys:
            if np.allclose(subpoly.coef, np.array([0] * subpoly.degree() + [1])):
                angles = np.pi * np.ones(subpoly.degree() + 1)
                if subpoly.degree() % 2 == 0:
                    angles[0] = -np.pi/2
                    angles[-1] = -np.pi/2
                normalization = 1
            else:
                normalization = sup_norm(subpoly) * 1.01
                angles = None
                max_tries = 100
                for _ in range(max_tries):
                    angles = compute_angles(subpoly / normalization)
                    if angles is not None:
                        break
                    normalization *= 1.01
                if angles is None:
                    raise RuntimeError(f"Could not compute angles for {subpoly}")

            result.append((subpoly / normalization, angles, normalization))

        return result

    def estimate_error(self, poly, samples, root=False):
        self.calls += poly.degree() * samples
        S = self.S
        if root:
            S = np.sqrt(self.S)

        if np.isinf(samples):
            return np.linalg.norm(poly(S) * self.b - self.b / self.S)

        angle_sequences = self.compute_angles(poly)

        results = np.zeros((samples, self.b.shape[0]))
        total_normalization = 0
        for subpoly, angles, normalization in angle_sequences:
            states = self.simulate_qsvt(S, angles, samples, folding=False)

            subresult = states[:, :self.b.shape[0]]
            results += normalization * subresult
            total_normalization += normalization

        diff = np.average(results, axis=0) - self.b / self.S

        # TODO: This is not the correct error measure
        return np.linalg.norm(diff) / np.linalg.norm(self.b / self.S) # + total_normalization / np.sqrt(samples)

    def estimate_poly(self, poly, samples, root=False):
        """
        Simulate estimation of b^T poly(A) b

        :param poly: The matrix polynomial to be computed
        :param samples: The number of simulated samples. Can be np.inf
        :param root: If True, compute poly(sqrt(A)) instead
        """

        # TODO: Also consider applications of b so that samples = inf makes sense
        self.calls += poly.degree() * samples
        S = self.S
        if root:
            S = np.sqrt(self.S)

        angle_sequences = self.compute_angles(poly)

        result = 0
        print("---")
        for subpoly, angles, normalization in angle_sequences:
            print(normalization, angles)
            # Resulting state
            exact_result = np.dot(self.b, subpoly(S) * self.b)

            if np.isinf(samples):
                return normalization * exact_result

            states = self.simulate_qsvt(S, angles, samples, folding=True)
            dim = self.b.shape[0]

            # Simulate final hadamard
            one_amplitude = (states[:, : 2 * dim] - states[:, 2 * dim :]) / np.sqrt(2)

            # Simulate sampling
            noisy_probabilities = np.linalg.norm(one_amplitude, axis=1) ** 2
            np.testing.assert_allclose(np.average(1 - 2 * noisy_probabilities), exact_result)
            noisy_probabilities = np.minimum(noisy_probabilities, 1)
            ones = np.sum(
                self.general_rng.binomial(1, noisy_probabilities[1:])
            )

            # Estimate corresponding to the hadamard test
            probability_estimate = 1 - 2 * ones / samples
            result += normalization * probability_estimate

        return result

    def simulate_qsvt(self, S, angles, samples, folding=True):
        """
        Simulates qsvt with noise.

        Returns states corresponding to noisy simulation of qsvt with the given
        angles. Results are given as an array with shape ``(samples, ...)``
        where the second dimension depends on the paramter ``folding``.

        If ``folding == True``, simulates QSVT folding. The number of
        applications of ``S`` is only half the degree of the polynomial. Each
        state has dimension ``dim * 4``, corresponding to the state before the
        last Hadamard gate, which is applied to the most significant bit.

        If ``folding == False``, simulates simple QSVT to prepare a solution
        state. The number of applications of ``S`` is equal to the degree of
        the polynomial. Each state has dimension ``dim * 2``, where the relavant
        part is the first half.
        """
        S_sqrt = 1j * np.sqrt(1 - S**2)

        dim = self.b.shape[0]

        if folding:
            duration = len(angles) // 2
        else:
            duration = len(angles) - 1
        # This is really correct:
        # the polynomial is even, if the number of angles is odd
        uneven = len(angles) % 2 == 0

        # The specific use of noise_rng makes sure that the total sequence of
        # numbers generated using poisson are the same for each separate reset
        noise_events = (
            poisson.rvs(
                self.noise,
                size=samples * duration,
                random_state=self.noise_rng,
            )
            .reshape((samples, -1))
            .astype(np.uint32)
        )
        events_per_sample = np.sum(noise_events, axis=-1)

        # Categorize into noiseless runs and ones with noise
        with_noise = noise_events[events_per_sample > 0]

        # Only need to simulate a single run without noise
        noise_events = np.concatenate((np.zeros((1, duration), dtype=np.uint32), with_noise), axis=0)

        # xs will be the state for the Hadamard test ancilla being 1
        # ys will be the state for the ancilla being 0
        xs = np.zeros((noise_events.shape[0], 2, dim), dtype=np.complex128)
        xs[:, 0, :dim] = np.exp(angles[0] * 1j) * self.b
        ys = None
        for i, angle in enumerate(angles[1 : 1 + duration]):
            if folding and i == duration - 1 and uneven:
                # If the polynomial is uneven and the hadard test ancilla is 0,
                # the final step should only be performed for xs but not ys
                ys = np.conj(xs)

            xs = S * xs + S_sqrt * xs[:, ::-1, :]

            max_flips = np.max(noise_events[:, i])
            for j in range(max_flips):
                should_flip = noise_events[:, i] < j
                flip_xs = xs[should_flip]
                flip_indices = self.general_rng.integers(
                    0, len(self.noise_flips), size=flip_xs.shape[0]
                )
                np.matvec(
                    self.noise_flips[flip_indices],
                    flip_xs.reshape((-1, 2 * dim)),
                    out=flip_xs.reshape((-1, 2 * dim)),
                )

            if folding and i == duration - 1:
                if uneven:
                    pass
                else:
                    xs[:, 0] *= np.exp(angle / 2 * 1j)
                    xs[:, 1] *= np.exp(-angle / 2 * 1j)
            else:
                xs[:, 0] *= np.exp(angle * 1j)
                xs[:, 1] *= np.exp(-angle * 1j)

        if folding:
            if not uneven:
                ys = np.conj(xs)
            xs = np.concatenate((ys, xs), axis=1) / np.sqrt(2)

        xs = xs.reshape((xs.shape[0], -1)).real

        # Undo noiseless sorting
        result = np.zeros((samples, xs.shape[1]))
        result[events_per_sample > 0] = xs[1:]
        result[events_per_sample == 0] = xs[0]

        return result
