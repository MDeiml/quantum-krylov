import numpy as np
from util import sup_norm
from scipy.stats import poisson
from symqsp import compute_angles


class NoiseModel:
    def __init__(
        self,
        rng,
        b,
        m,
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

        self.rng = rng

        n = b.shape[0]
        self.S = np.linspace(1 / kappa, 1, n)

        self.b = b
        self.m = m
        self.solution = np.dot(self.m, self.b / self.S)

        self.noise = noise
        self.noise_flips = noise_flips
        self.reset_profiling()

    def reset_profiling(self):
        self.calls = 0

    def complexity(self):
        return self.calls

    def estimate_poly(
        self, poly, samples, qoi=False, root=False, poly_normalized=False
    ):
        """
        Simulate estimation of b^T poly(A) b (qoi = False) or m^T poly(A) b (qoi = True)

        :param poly: The matrix polynomial to be computed
        :param samples: The number of simulated samples. Can be np.inf
        :param qoi: Whether to compute inner products (m, b) or (b, b)
        :param root: If True, compute poly(sqrt(A)) instead
        :param poly_normalized:
            Should be set to True if |poly(x)| < 1 for x \\in [-1, 1] to avoid
            extra computational work.
        """

        x = self.b
        y = self.m if qoi else self.b

        poly = poly.convert(kind=np.polynomial.Chebyshev)

        parity = poly.degree() % 2
        is_parity = np.allclose(poly.coef[(1 - parity) :: 2], 0, atol=1e-6)

        # TODO: Also consider applications of b so that samples = inf makes sense
        self.calls += poly.degree() * samples
        S = self.S
        if root:
            S = np.sqrt(self.S)

        estimate = 0
        if not is_parity:
            coefs_non_parity = poly.coef[:-1].copy()
            coefs_non_parity[parity::2] = 0
            poly_non_parity = np.polynomial.Chebyshev(coefs_non_parity)
            estimate += self._estimate_poly(S, x, y, poly_non_parity, samples)
        coefs_parity = poly.coef.copy()
        coefs_parity[1 - parity :: 2] = 0
        poly_parity = np.polynomial.Chebyshev(coefs_parity)
        estimate += self._estimate_poly(S, x, y, poly_parity, samples)

        return estimate

    def _estimate_poly(self, S, x, y, poly, samples):

        # Resulting state
        result = poly(S) * x

        probability = np.dot(y, result)
        sign = np.sign(probability)
        probability = np.abs(probability)

        if np.isinf(samples):
            return sign * probability

        # Known angles for Chebyshev polynomials
        if np.allclose(poly.coef, np.array([0] * poly.degree() + [1])):
            angles = np.zeros(poly.degree() + 1)
            normalization = 1
        else:
            normalization = sup_norm(poly) * 1.01
            angles = None
            max_tries = 100
            for _ in range(max_tries):
                angles = compute_angles(poly / normalization)
                if angles is not None:
                    break
                normalization *= 1.01
            if angles is None:
                raise RuntimeError(f"Could not compute angles for {poly}")

        ones = self.simulate_qsp(S, x, y, angles, samples)
        return sign * normalization * np.sqrt(ones / samples)

    def simulate_qsp(self, S, x, y, angles, samples):
        S_sqrt = 1j * np.sqrt(1 - S**2)

        dim = self.b.shape[0]

        # Compute forwards and backwards
        forward_xs = np.zeros((len(angles), 2, dim), dtype=np.complex128)
        forward_xs[0, 0, :] = x * np.exp(angles[0] * 1j)
        backward_xs = np.zeros((len(angles), 2, dim), dtype=np.complex128)
        backward_xs[-1, 0, :] = y
        for i in range(1, len(angles)):
            forward_xs[i] = S * forward_xs[i - 1] + S_sqrt * forward_xs[i - 1, ::-1, :]

            forward_xs[i, 0] *= np.exp(angles[i] * 1j)
            forward_xs[i, 1] *= np.exp(-angles[i] * 1j)

            backward_xs[-(i + 1), 0] *= np.exp(-angles[-(i + 1)] * 1j)
            backward_xs[-(i + 1), 1] *= np.exp(angles[-(i + 1)] * 1j)

            backward_xs[-(i + 1)] = (
                S * backward_xs[-i] + S_sqrt * backward_xs[-i, ::-1, :]
            )

        noise_events = (
            poisson.rvs(
                self.noise, size=samples * (len(angles) - 1), random_state=self.rng
            )
            .reshape((samples, -1))
            .astype(np.uint8)
        )
        events_per_sample = np.sum(noise_events, axis=-1)

        # Categorize noise into a) no noise b) one noise c) more than one noise
        noiseless = events_per_sample == 0
        one_noise = noise_events[events_per_sample == 1]
        more_noise = noise_events[events_per_sample > 1]

        # case a) we already computed
        probability = (
            np.abs(np.dot(forward_xs[-1].reshape(-1), backward_xs[-1].reshape(-1)).real)
            ** 2
        )
        result = self.rng.binomial(np.sum(noiseless), min(1, probability))

        # case b) can be computed from the forward and backward xs
        if one_noise.shape[0] > 0:
            noise_timings = np.nonzero(one_noise)[1] + 1
            flip_indices = self.rng.integers(
                0, len(self.noise_flips), size=one_noise.shape[0]
            )
            flips = self.noise_flips[flip_indices]
            noisy_probabilities = (
                np.abs(
                    np.sum(
                        forward_xs[noise_timings].reshape((-1, 2 * dim))
                        * np.matvec(
                            flips, backward_xs[noise_timings].reshape((-1, 2 * dim))
                        ),
                        axis=-1,
                    )
                )
                ** 2
            )
            noisy_probabilities = np.minimum(noisy_probabilities, 1)
            result += np.sum(self.rng.binomial(1, noisy_probabilities[:]))

        # case c) more than one noise. We actually simulate each run
        if more_noise.shape[0] > 0:
            xs = np.zeros((more_noise.shape[0], 2, dim), dtype=np.complex128)
            xs[:, 0, :dim] = np.exp(angles[0] * 1j) * x
            for i, angle in enumerate(angles[1:]):
                xs = S * xs + S_sqrt * xs[:, ::-1, :]

                max_flips = np.max(more_noise[:, i])
                for j in range(max_flips):
                    should_flip = more_noise[:, i] < j
                    flip_xs = xs[should_flip]
                    flip_indices = self.rng.integers(
                        0, len(self.noise_flips), size=flip_xs.shape[0]
                    )
                    np.matvec(
                        self.noise_flips[flip_indices],
                        flip_xs.reshape((-1, 2 * dim)),
                        out=flip_xs.reshape((-1, 2 * dim)),
                    )

                xs[:, 0] *= np.exp(angle * 1j)
                xs[:, 1] *= np.exp(-angle * 1j)

            noisy_probabilities = np.abs(np.dot(xs[:, 0], y).real) ** 2
            noisy_probabilities = np.minimum(noisy_probabilities, 1)
            result += np.sum(self.rng.binomial(1, noisy_probabilities[:]))

        return result

    def reference_solution(self):
        return self.solution
