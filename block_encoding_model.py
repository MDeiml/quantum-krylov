from collections.abc import Callable
import numpy as np
from symqsp import compute_angles
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
        [self.general_rng_seed, self.noise_rng_seed] = seed.spawn(2)

        self.noise = noise

        if noise_flips is None:
            noise_flips = generate_noise_flips(
                np.random.default_rng(np.random.SeedSequence(43)), b.shape[0], 20
            )
        self.noise_flips = noise_flips
        # np.polynomial.Chebyshev is not hashable, so we cannot use a dict
        # for hashing polynomials
        self.cache = []
        self.reset()

    def reset(self):
        """
        Reset the random generators and the complexity counter
        """
        self.queries = 0
        self.general_rng = np.random.default_rng(self.general_rng_seed)
        self.noise_rng = np.random.default_rng(self.noise_rng_seed)

    def complexity(self):
        """
        The number of times a block encoding was applied

        This is counting since this object was created or the last call to
        `reset`.
        """
        return self.queries

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

        # B is either D or D^{1/2}
        B = self.D
        if root:
            B = np.sqrt(self.D)

        exact_solution = self.b / self.D
        exact_solution_norm = np.linalg.norm(exact_solution)

        # Find the quantity of interest that maximizes the error
        approximation = poly(B) * self.b

        if np.isinf(samples):
            self.queries = np.inf
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

        result = self._simulate_qsvt(
            B,
            [angles for _, angles, _ in angle_sequences],
            samples,
            measure,
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
        error_sq = (
            np.sqrt(np.maximum(0, result_axial_sq)) - exact_solution_norm
        ) ** 2 + np.abs(result_orthogonal_sq)

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

        # The calls are cached, but this requires some care.
        #
        # * Polynomial coefficients should only be compared approximately using
        #   `np.all_close`.
        # * Calls can only be cached if the state of the random generators
        #   matches.
        # * This class records the number of queries, so the `self.queries`
        #   attribute also should match.
        p = (
            poly.coef,
            samples,
            root,
            self.queries,
            self.general_rng.bit_generator.state,
            self.noise_rng.bit_generator.state,
        )

        for k, v in self.cache:
            if (
                p[0].shape == k[0].shape
                and np.allclose(p[0], k[0], atol=1e-6)
                and p[1:] == k[1:]
            ):
                (
                    result,
                    self.queries,
                    self.general_rng.bit_generator.state,
                    self.noise_rng.bit_generator.state,
                ) = v
                return result

        # B is either D or D^{1/2}
        B = self.D
        if root:
            B = np.sqrt(self.D)

        exact = np.dot(self.b, poly(B) * self.b)
        # TODO: Also consider applications of b so that samples = inf makes sense
        if np.isinf(samples):
            self.queries = np.inf
            return exact

        angle_sequences = compute_angles(poly)

        assert len(angle_sequences) == 1, "not implemented"

        result = 0
        subpoly, angles, normalization = angle_sequences[0]
        if root:
            assert subpoly.degree() % 2 == 0, (
                "Can only evaluate even polynomials for squareroot of matrix"
            )

        result = (
            self._simulate_qsvt_folding(
                B,
                angles,
                samples,
                reference=exact / normalization,
            )
            * normalization
        )

        self.cache.append(
            (
                p,
                (
                    result,
                    self.queries,
                    self.general_rng.bit_generator.state,
                    self.noise_rng.bit_generator.state,
                ),
            )
        )

        return result

    def _simulate_qsvt_folding(
        self,
        B: np.ndarray,
        angles: np.ndarray,
        samples: int,
        reference: float | None = None,
    ):
        """
        Simulate the measurement of $b^T p(B) b$

        Using the technique of _QSVT bending_ the block encoding of $B$ is only
        applied ``len(angles) // 2`` times.

        This method calls `simulate_qsvt` internally. It can be quite expensive
        if `samples` and `self.noise` are both large.

        :param B:
            Elements of the matrix B. It is assumed that B is a diagonal matrix,
            which can always be achieved by a suitable change of basis. `B`
            should thus be a vector of the same dimension as `self.b`.
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

        return self._simulate_qsvt(
            B, angle_sequences, samples, observable, reference=reference
        )

    def _simulate_qsvt(
        self,
        B: np.ndarray,
        angle_sequences: list[np.ndarray],
        samples: int,
        observable: Callable[[np.ndarray, np.ndarray], float],
        weights: np.ndarray | None = None,
        reference: float | None = None,
    ):
        """
        Simulate the measurement of the state $b^T p(B)$ with the given observable.

        Specifically, this allows to compute $y^T M y$ where $y = [w_0 p_0(B)
        b, ..., w_{J-1} p_{J-1}(B) b]$ where $M$ is specified by ``observable``,
        $p_0, ..., p_{J-1}$ are polynomials with definite partiy as specified by
        ``angle_sequences`` and $w_0^2 + ... + w_{J-1}^2 = 1$.

        This method  can be quite expensive if `samples` and `self.noise` are
        both large.

        :param B:
            Elements of the matrix B. It is assumed that B is a diagonal matrix,
            which can always be achieved by a suitable change of basis. `B`
            should thus be a vector of the same dimension as `self.b`.
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
        :param weights:
            Optional vector containg the weights $w_0, ..., w_{J-1}$. If
            ``None``, the default is $w_j = \\sqrt{1/J}$.
        :param reference:
            Optional reference value, which the simulation should equal if no
            noise is applied. This is only use for testing. Generates an error
            if the simulated result is not `reference`
        """
        B_sqrt = np.sqrt(1 - B**2)

        dim = self.b.shape[0]
        duration = max([len(angles) - 1 for angles in angle_sequences])
        self.queries += duration * samples

        # The specific use of noise_rng makes sure that the total sequence of
        # numbers generated using binomial are the same for each separate reset.
        # poission would be more accurate
        noise_events = self.noise_rng.binomial(
            1,
            self.noise,
            size=(samples, duration),
        ).astype(np.uint32)
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

        result = None
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
                batch[:, i, 0, :] = factor * phases[i, 0] * self.b
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
                                B_sqrt, batch[:, k, ::-1, :], out=temp_batch[:, 0]
                            )
                            batch[:, k] *= B
                            batch[:, k, 1] *= -1
                            batch[:, k] += temp_batch[:, 0]
                else:
                    np.multiply(B_sqrt, batch[:, :, ::-1, :], out=temp_batch)
                    batch *= B
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
            measurements = observable(batch, temp_batch).real
            assert measurements.shape[0] == batch.shape[0]
            probabilities = (1 - measurements) / 2
            if j == 0:
                if reference is not None:
                    # Check that noiseless simulation is correct
                    np.testing.assert_allclose(measurements[0], reference, atol=1e-5)
                # Noiseless run happens multiple times
                result = self._measure(
                    probabilities[0:1], samples - noise_events.shape[0] - 1
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
        return np.sum(self.general_rng.binomial(samples, probabilities), axis=0)
