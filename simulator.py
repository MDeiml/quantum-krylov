import numpy as np
import line_profiler


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

    @line_profiler.profile
    def simulate_qsvt(self, S, b, angles, samples, folding=True):
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

        dim = b.shape[0]

        if folding:
            duration = len(angles) // 2
        else:
            duration = len(angles) - 1
        self.calls += duration * samples

        # This is really correct:
        # the polynomial is even, if the number of angles is odd
        uneven = len(angles) % 2 == 0

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
        phases = angles[: duration + 1].copy()
        if folding:
            if uneven:
                phases[duration] = 0
            else:
                phases[duration] /= 2
        phases = np.exp(phases * 1j)

        # xs will be the state for the Hadamard test ancilla being 1
        # ys will be the state for the ancilla being 0
        xs = np.zeros((noise_events.shape[0], 2, dim), dtype=np.complex64)
        ys = np.zeros_like(xs) if folding and uneven else None
        xs[:, 0, :] = np.exp(angles[0] * 1j) * b
        xs[:, 1, :] = 0

        batch_size = 16

        temp = np.zeros((batch_size, 2, dim), dtype=np.complex64)

        for j in range(int(np.ceil(xs.shape[0] / batch_size))):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, xs.shape[0])
            batch = xs[batch_start:batch_end]
            temp_batch = temp[:batch_end - batch_start]
            for i, phase in enumerate(phases[1:]):
                if folding and i == duration - 1 and uneven:
                    # If the polynomial is uneven and the hadard test ancilla is 0,
                    # the final step should only be performed for xs but not ys
                    ys[batch_start:batch_end] = np.conj(xs[batch_start:batch_end])

                # This corresponds to
                # x = S * x + S_sqrt * x[::-1, :]
                np.multiply(S_sqrt, batch[:, ::-1, :], out=temp_batch)
                np.multiply(S, batch, out=batch)
                batch += temp_batch

                flip_index = self.general_rng.integers(
                    0, len(self.noise_flips), size=batch.shape[0]
                )
                for k in range(batch.shape[0]):
                    if noise_events[batch_start + k, i] != 0:
                        temp_batch[k] = 0
                    else:
                        flip = self.noise_flips[flip_index[k]]
                        np.matvec(
                            flip,
                            batch[k].reshape(2 * dim),
                            out=batch[k].reshape(2 * dim),
                        )

        if folding:
            if uneven:
                xs = np.concatenate((ys, xs), axis=1) / np.sqrt(2)
            else:
                xs = np.concatenate((xs, xs), axis=1) / np.sqrt(2)

        xs = xs.reshape((xs.shape[0], -1))

        # Undo noiseless sorting
        result = np.zeros((samples, xs.shape[1]), dtype=np.complex64)
        result[events_per_sample > 0] = xs[1:]
        result[events_per_sample == 0] = xs[0]

        return result

    def measure(self, probabilities):
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        assert min_probability >= 0 and max_probability - 1 < 1e-4, (
            f"probabilities [{min_probability}, {max_probability}] are not between 0 and 1"
        )
        probabilities = np.minimum(1, probabilities)
        return np.average(self.general_rng.binomial(1, probabilities))
