from collections.abc import Callable
import numpy as np


class Simulator:
    """
    Helper to simulate a quantum device

    This is a low level interface. Use `block_encoding_model.BlockEncodingModel`
    for the most common cases.

    :param seed:
        Seed for the random generators used for noise and sampling.
    :param noise:
        Value between 0 and 1. This is the probability that a random flip will
        be applied after an application of a block encoding.
    """

    def __init__(
        self,
        seed: np.random.SeedSequence,
        noise: float = 0.01,
    ):
        # There are two different generators, to make sure that the total amount
        # of noise for two runs is the same if they use the same number of
        # applications of A and the same number of samples
        [self.general_rng_seed, self.noise_rng_seed] = seed.spawn(2)

        self.noise = noise
        self.reset()

    def reset(self):
        """
        Reset the random generators and the complexity counter
        """
        self.calls = 0
        self.general_rng = np.random.default_rng(self.general_rng_seed)
        self.noise_rng = np.random.default_rng(self.noise_rng_seed)

    def complexity(self):
        """
        The number of times a block encoding was applied

        This is counting since this object was created or the last call to
        `reset`.
        """
        return self.calls

    def simulate_qsvt_folding(
        self,
        S: np.ndarray,
        b: np.ndarray,
        angles: np.ndarray,
        samples: int,
        noise_flips: np.ndarray,
        reference: float | None = None,
    ):
        """
        Simulate the measurement of $b^T p(S) b$

        Using the technique of _QSVT bending_ the block encoding of $S$ is only
        applied ``len(angles) // 2`` times.

        This method calls `simulate_qsvt` internally. It can be quite expensive
        if `samples` and `self.noise` are both large.

        :param S:
            Elements of the matrix S. It is assumed that S is a diagonal matrix,
            which can always be achieved by a suitable change of basis. `S`
            should thus be a vector of the same dimension as `b`.
        :param b:
            The vector `b` as a numpy array
        :param angles:
            Angles in R-convention corresponding to the polynomial $p$.
            (The degree of $p$ is ``len(angles) - 1``).
            The angles have to be symmetric in the sense that ``angles[-(i
            + 1)] = angles[i]``. This is true for angles returned by
            `symqsp.compute_angles`.
        :param samples:
            The number of measurements in the Hadamard test for computing the
            inner product. The accuracy of the estimate will roughly be ``2
            / sqrt(samples)``.
        :param noise_flips:
            Unitary matrices, corresponding to the possible noise that can
            be introduced after each application of $A$. Specifically, if
            noise is applied, the state is multiplied with a random element
            of `noise_flips`. Should have the shape ``(M, N, N)`` where ``N
            = b.shape[0]``. Use `util.generate_noise_flips` to generate these
            matrices.
        :param reference:
            Optional reference value, which the simulation should equal if no
            noise is applied. This is only use for testing. Generates an error
            if the simulated result is not `reference`
        """
        np.testing.assert_allclose(
            angles, angles[::-1], err_msg="Angles should be symmetric"
        )

        def observable(x, temp):
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
            S, b, angle_sequences, samples, observable, noise_flips, reference=reference
        )

    def simulate_qsvt(
        self,
        S: np.ndarray,
        b: np.ndarray,
        angle_sequences: list[np.ndarray],
        samples: int,
        observable: Callable[[np.ndarray, np.ndarray], float],
        noise_flips: np.ndarray,
        weights: np.ndarray | None = None,
        reference: float | None = None,
    ):
        """
        Simulate the measurement of the state $b^T p(S)$ with the given observable.

        Specifically, this allows to compute $y^T M y$ where $y = [w_0 p_0(S)
        b, ..., w_{J-1} p_{J-1}(S) b]$ where $M$ is specified by ``observable``,
        $p_0, ..., p_{J-1}$ are polynomials with definite partiy as specified by
        ``angle_sequences`` and $w_0^2 + ... + w_{J-1}^2 = 1$.

        This method  can be quite expensive if `samples` and `self.noise` are
        both large.

        :param S:
            Elements of the matrix S. It is assumed that S is a diagonal matrix,
            which can always be achieved by a suitable change of basis. `S`
            should thus be a vector of the same dimension as `b`.
        :param b:
            The vector `b` as a numpy array.
        :param angle_sequences:
            List of list of angles. The element ``angle_sequences[j]``
            should be an angle sequence in R-convention corresponding to the
            polynomial $p_j$. (The degree of $p_j$ is ``len(angle_sequences[j])
            - 1``). The angles have to be symmetric in the sense that
            ``angle_sequences[j][-(k + 1)] = angles[j][k]``. This is true for
            angles returned by `symqsp.compute_angles`.
        :param samples:
            The number of measurements in the Hadamard test for computing the
            inner product. The accuracy of the estimate will roughly be ``2
            / sqrt(samples)``.
        :param observable:
            Function that computes the observable from a given state. The
            state is given as an array of the shape ``(B, J, 2, N)`` where the
            dimension ``B`` is used to batch calls, i.e. the array contains
            ``B`` unrelated states, the dimension ``J`` corresponds to the
            length of ``angle_sequences``, the dimension ``2`` is ``0`` for the
            relevant part of the state and ``1`` for the "garbarge" part, and
            ``N`` is the dimension of the linear system, i.e. the same dimension
            as `b`.

            (The second argument to `observable` is a vector of the same shape
            as the first argument, which can be used as temporary storage to
            avoid allocations, or can safely be ignored.)
        :param noise_flips:
            Unitary matrices, corresponding to the possible noise that can
            be introduced after each application of $A$. Specifically, if
            noise is applied, the state is multiplied with a random element
            of `noise_flips`. Should have the shape ``(M, N, N)`` where ``N
            = b.shape[0]``. Use `util.generate_noise_flips` to generate these
            matrices.
        :param weights:
            Optional vector containg the weights $w_0, ..., w_{J-1}$. If
            ``None``, the default is $w_j = \\sqrt{1/J}$.
        :param reference:
            Optional reference value, which the simulation should equal if no
            noise is applied. This is only use for testing. Generates an error
            if the simulated result is not `reference`
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
                    0, len(noise_flips), size=batch.shape[0]
                )
                for k in range(batch.shape[0]):
                    if noise_events[batch_start + k, i] != 0:
                        flip = noise_flips[flip_index[k]]
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
            measurements = observable(batch, temp_batch).real
            assert measurements.shape == (batch.shape[0],)
            probabilities = (1 - measurements) / 2
            if j == 0:
                if reference is not None:
                    # Check that noiseless simulation is correct
                    np.testing.assert_allclose(measurements[0], reference, atol=1e-5)
                # Noiseless run happens multiple times
                result += self._measure(
                    probabilities[0], samples - noise_events.shape[0] - 1
                )
                probabilities = probabilities[1:]
            result += self._measure(probabilities)

        return 1 - 2 * result / samples

    def _measure(self, probabilities, samples=1):
        if probabilities.ndim > 0 and probabilities.shape[0] == 0:
            return 0
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        assert min_probability >= 0 and max_probability - 1 < 1e-4, (
            f"probabilities [{min_probability}, {max_probability}] are not between 0 and 1"
        )
        probabilities = np.minimum(samples, probabilities)
        return np.sum(self.general_rng.binomial(samples, probabilities))
