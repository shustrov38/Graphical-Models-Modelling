import numpy as np

from .base import Base
from ..math import (
    laplace_transform, kendall_tau, 
    bonferroni_method
)


class Kendall(Base):
    """Implementation for construction true graphocal model using 
    Kendalls correlation between linreg residuals
    """

    def __init__(
        self, 
        n: int,
        p: int, 
        true_model: np.ndarray, 
        alpha: float = 0.05,
        correction_method: str = 'holm'
    ) -> None:
        super().__init__(n, p, true_model, alpha)
        self._D_tau = np.sqrt(
            2 * (2 * self._n + 5) / (9 * self._n * (self._n - 1)))
        self._correction_method = correction_method

    def _process_samples_impl(self, i: int, j: int) -> float:
        mask = [it for it in range(self._p) if it != i and it != j]

        Xij = self._X[:, mask]
        Xi = self._X[:, i]
        Xj = self._X[:, j]
        
        beta_i = np.dot(np.dot(np.linalg.inv(np.dot(Xij.T, Xij)), Xij.T), Xi)
        beta_j = np.dot(np.dot(np.linalg.inv(np.dot(Xij.T, Xij)), Xij.T), Xj)

        res_i = Xi - np.dot(Xij, beta_i)
        res_j = Xj - np.dot(Xij, beta_j)

        t = np.abs(kendall_tau(res_i, res_j) / self._D_tau)

        p = 2 * (1 - laplace_transform(t))
        return p
    
    def _correction_impl(self, values: np.ndarray) -> np.ndarray: 
        if self._correction_method == 'holm':
            # Holm correction
            flatten = [(values[i, j], i, j) for i in range(self._p) 
                        for j in range(i + 1, self._p)]
            
            flatten.sort(key=lambda p: p[0])

            done = False
            for index, (pvalue, i, j) in enumerate(flatten, start=1):
                values[i, j] = (self._hypothesis_count - index + 1) * pvalue
                if done or values[i, j] > self._alpha:
                    values[i, j] = 1
                    done = True

            return values
        
        elif self._correction_method == 'bonf':
            return bonferroni_method(
                values, self._hypothesis_count, self._alpha)
        
        else:
            raise NotImplementedError(self._correction_method)
