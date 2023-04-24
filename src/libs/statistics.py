from functools import lru_cache
from typing import final
import numpy as np

from .math import z_fisher_transform, laplace_transform


class Base:

    def __init__(
        self, 
        X: np.ndarray, 
        true_model: np.ndarray,
        alpha: float = 0.05
    ) -> None:
        self.X = X - X.mean(axis=0)
        self.n, self.p = self.X.shape

        self.hypothesis_count = self.p * (self.p - 1) / 2
        self.alpha = alpha

        self.true_model = true_model

        self.cov = np.cov(X.T, ddof=1)
        self.cov_inv = np.linalg.inv(self.cov)
        print(self.cov_inv)

        self.__error_type_I = np.zeros(self.cov.shape)
        self.__error_type_II = np.zeros(self.cov.shape)

        self.__fwer_error_type_I = 0

    @final
    def _update_errors(self, i: int, j: int, value: float) -> bool:
        is_error_type_I_updated = False

        if not self.true_model[i][j] and self.is_edge(value=value):
            self.__error_type_I[i][j] += 1
            is_error_type_I_updated = True

        if self.true_model[i][j] and not self.is_edge(value=value):
            self.__error_type_II[i][j] += 1

        return is_error_type_I_updated

    @final
    @lru_cache
    def value(self) -> np.ndarray:
        """Calculate result statistics and update metrics

        Returns:
            np.ndarray: calculated statistics with corrections
        """
        result = np.zeros(self.cov.shape)
        is_error_updated = False

        for i in range(self.p):
            for j in range(i + 1, self.p):
                result[i][j] = self._value_impl(i, j)
        
        result = self._correction_impl(result)

        for i in range(self.p):
            for j in range(i + 1, self.p):
                is_error_updated |= self._update_errors(i, j, result[i][j])
        
        self.__fwer_error_type_I += is_error_updated
        return result

    @final
    @lru_cache
    def is_edge(self, value: float) -> bool:
        """Check that the calculated statistics signal the presence of an edge

        Args:
            value (float): Input statistic

        Returns:
            bool: Means edge or not
        """
        return self._is_edge_impl(value)
    
    def _value_impl(self, i: int, j: int) -> float:
        return 0.

    def _correction_impl(self, values: np.ndarray) -> None:
        pass

    def _is_edge_impl(self, value: float) -> float:
        return False


class Pearson(Base):

    def _value_impl(self, i: int, j: int) -> float:
        # calculate partial correlation
        r = -self.cov_inv[i][j] / np.sqrt((self.cov_inv[i][i] ** 2) * (self.cov_inv[j][j] ** 2))
        # normalize coefficient
        z = z_fisher_transform(r)
        # calculate pvalue
        p = 1 - laplace_transform(np.sqrt(self.n - self.p - 1) * np.abs(z))
        return p

    def _correction_impl(self, values: np.ndarray) -> np.ndarray:
        # Bonferroni correction
        print(values)
        return np.clip(self.hypothesis_count * 2 * values, 0, 1)

    def _is_edge_impl(self, value: float) -> float:
        return value < self.alpha