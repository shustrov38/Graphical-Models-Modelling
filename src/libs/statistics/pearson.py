import numpy as np

from .base import Base
from ..math import z_fisher_transform, laplace_transform


class Pearson(Base):
    """Implementation for construction true graphocal model using 
    Pearsons partial correlation
    """

    def _process_samples_impl(self, i: int, j: int) -> float:
        # calculate partial correlation
        r = -self._cov_inv[i][j] / np.sqrt((self._cov_inv[i][i] ** 2) * (self._cov_inv[j][j] ** 2))
        # normalize coefficient
        z = z_fisher_transform(r)
        # calculate pvalue
        p = 1 - laplace_transform(np.sqrt(self._n - self._p - 1) * np.abs(z))
        return p

    def _correction_impl(self, values: np.ndarray) -> np.ndarray:
        # Bonferroni correction
        return np.clip(self._hypothesis_count * 2 * values, 0, 1)
