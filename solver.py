import numpy as np

from block_encoding_model import BlockEncodingModel


class Solver:
    def __init__(
        self,
        default_samples=10000,
        transform_method=None,
    ):
        self.transform_method = transform_method
        self.default_samples = default_samples

    def evaluate(self, A: BlockEncodingModel, poly):
        return A.estimate_poly(
            poly,
            self.default_samples,
            qoi=True,
            root=self.transform_method == "square",
        )

    def compute_polynomial(self, A: BlockEncodingModel):
        raise NotImplementedError

    def plot(self, A: BlockEncodingModel):
        import matplotlib.pyplot as plt

        poly = self.compute_polynomial(A)
        xs = np.linspace(-1, 1, 200)
        if self.transform_method == "square":
            plt.plot(xs, poly(np.abs(xs)))
        else:
            plt.plot(xs, poly(xs))
        plt.plot(xs, 1 / xs, "--")
        plt.axvline(x=1 / A.kappa)
        plt.ylim([-11, 11])
