import numpy as np


class Simulator:
    def __init__(
        self,
        seed: np.random.SeedSequence,
        noise_flips,
        noise=0.01,
    ):
        """
        Wrapper to simulate quantum device

        :param A:
            Matrix (2d) or singular values (1d) or condition number (scalar)
            number (scalar)
        """

        [self.general_rng_seed, self.noise_rng_seed] = seed.spawn(2)

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

    def simulate_qsvt_folding(self, S, b, angles, samples, test=None):
        def measurement(x, temp):
            # a = (x[:, 0, :, :] - x[:, 1, :, :]) / np.sqrt(2)
            # a.shape = (x.shape[0], -1)
            # return 1 - 2 * np.vecdot(a, a)
            x = x.reshape((x.shape[0], 2, -1))
            return 2 * np.vecdot(x[:, 0], x[:, 1])

        even = len(angles) % 2 == 1
        if even:
            angles_new = angles[: (len(angles) + 1) // 2].copy()
            angles_new[-1] /= 2
            angle_sequences = [angles_new, -angles_new]
        else:
            angle_sequences = [
                angles[: len(angles) // 2],
                -np.concatenate((angles[: len(angles) // 2], [0])),
            ]
        return self.simulate_qsvt(
            S, b, angle_sequences, samples, measurement, test=test
        )

    def simulate_qsvt(
        self, S, b, angle_sequences, samples, measurement, weights=None, test=None
    ):
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
        S_sqrt = np.sqrt(1 - S**2)

        dim = b.shape[0]
        duration = max([len(angles) - 1 for angles in angle_sequences])
        self.calls += duration * samples

        # The specific use of noise_rng makes sure that the total sequence of
        # numbers generated using binomial are the same for each separate reset.
        # poission would be more accurate
        noise_events = (
            self.noise_rng.binomial(
                1,
                self.noise,
                size=samples * duration,
            )
            .reshape((samples, -1))
            .astype(np.uint32)
        )
        events_per_sample = np.sum(noise_events, axis=-1)

        # Categorize into noiseless runs and ones with noise
        with_noise = noise_events[events_per_sample > 0]

        # Only need to simulate a single run without noise
        noise_events = np.concatenate(
            (np.zeros((1, duration), dtype=np.uint8), with_noise), axis=0
        )

        # Precompute phases
        phases = np.zeros((len(angle_sequences), duration + 1))
        for i, angles in enumerate(angle_sequences):
            phases[i, : len(angles)] = angles
        phases = np.exp(phases * 1j)

        batch_size = 16
        xs = np.zeros((batch_size, len(angle_sequences), 2, dim), dtype=np.complex64)
        temp = np.zeros_like(xs)

        result = 0
        for j in range(int(np.ceil(noise_events.shape[0] / batch_size))):
            # Setup initial state corresponding to LCU
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, noise_events.shape[0])
            batch = xs[: batch_end - batch_start]
            temp_batch = temp[: batch_end - batch_start]
            for i, angles in enumerate(angle_sequences):
                factor = (
                    weights[i]
                    if weights is not None
                    else 1 / np.sqrt(len(angle_sequences))
                )
                batch[:, i, 0, :] = factor * phases[i, 0] * b
            batch[:, :, 1, :] = 0
            for i in range(duration):
                # In the final step, the unitary is performed controlled on wether
                # this polynomial has the same partiy as the longest polynomial
                if i == duration - 1:
                    for k, angles in enumerate(angle_sequences):
                        if (len(angles) - 1) % 2 == duration % 2:
                            # For non-folding case, apply the last unitary if the
                            # polynomial has the same partiy as the longest polynomial

                            np.multiply(
                                S_sqrt, batch[:, k, ::-1, :], out=temp_batch[:, 0]
                            )
                            batch[:, k] *= S
                            batch[:, k, 1] *= -1
                            batch[:, k] += temp_batch[:, 0]
                else:
                    np.multiply(S_sqrt, batch[:, :, ::-1, :], out=temp_batch)
                    batch *= S
                    batch[:, :, 1] *= -1
                    batch += temp_batch

                flip_index = self.general_rng.integers(
                    0, len(self.noise_flips), size=batch.shape[0]
                )
                for k in range(batch.shape[0]):
                    if noise_events[batch_start + k, i] != 0:
                        flip = self.noise_flips[flip_index[k]]
                        # For some reason matvec does not work here
                        np.einsum(
                            "...ij,...j",
                            flip,
                            batch[k].reshape(-1, 2 * dim),
                            out=batch[k].reshape(-1, 2 * dim)[:],
                        )

                batch[:, :, 0, :] *= phases[:, i + 1, np.newaxis]
                batch[:, :, 1, :] /= phases[:, i + 1, np.newaxis]

            # Check that squared amplitudes sum up to 1
            shape = (batch.shape[0], -1)
            np.testing.assert_allclose(
                np.vecdot(batch.reshape(shape), batch.reshape(shape)), 1, rtol=1e-5
            )

            # Hadamard test
            measurements = measurement(batch, temp_batch).real
            assert measurements.shape == (batch.shape[0],)
            probabilities = (1 - measurements) / 2
            if j == 0:
                if test is not None:
                    # Check that noiseless simulation is correct
                    np.testing.assert_allclose(measurements[0], test, atol=1e-5)
                # Noiseless run happens multiple times
                result += self.measure(
                    probabilities[0], samples - noise_events.shape[0] - 1
                )
                probabilities = probabilities[1:]
            result += self.measure(probabilities)

        return 1 - 2 * result / samples

    def measure(self, probabilities, samples=1):
        if probabilities.ndim > 0 and probabilities.shape[0] == 0:
            return 0
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        assert min_probability >= 0 and max_probability - 1 < 1e-4, (
            f"probabilities [{min_probability}, {max_probability}] are not between 0 and 1"
        )
        probabilities = np.minimum(samples, probabilities)
        return np.sum(self.general_rng.binomial(samples, probabilities))
