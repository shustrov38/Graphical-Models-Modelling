from functools import lru_cache
from typing import final
import numpy as np

from .math import z_fisher_transform, laplace_transform, kendall_tau


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
        return np.clip(self.hypothesis_count * 2 * values, 0, 1)

    def _is_edge_impl(self, value: float) -> float:
        return value < self.alpha
    

class Kendall(Base):

    def __init__(self, X: np.ndarray, true_model: np.ndarray, alpha: float = 0.05) -> None:
        super().__init__(X, true_model, alpha)
        self.D_tau = np.sqrt(2 * (2 * self.n + 5) / (9 * self.n * (self.n - 1)))

    def _value_impl(self, i: int, j: int) -> float:
        mask = [it for it in range(self.p) if it != i and it != j]

        Xij = self.X[:, mask]
        Xi = self.X[:, i]
        Xj = self.X[:, j]
        
        beta_i = np.dot(np.dot(np.linalg.inv(np.dot(Xij.T, Xij)), Xij.T), Xi)
        beta_j = np.dot(np.dot(np.linalg.inv(np.dot(Xij.T, Xij)), Xij.T), Xj)

        res_i = Xi - np.dot(Xij, beta_i)
        res_j = Xj - np.dot(Xij, beta_j)

        t = np.abs(kendall_tau(res_i, res_j) / self.D_tau)

        p = 2 * (1 - laplace_transform(t))

        return p
    
    def _correction_impl(self, values: np.ndarray) -> np.ndarray:
        # Holm correction
        flat = values.ravel()
        indexes_in_sorted_order = np.argsort(flat) + 1
        for i, pvalue in enumerate(flat):
            index = indexes_in_sorted_order[i]
            flat[i] = np.clip((self.hypothesis_count - index + 1) * pvalue, 0, 1)
        return flat.reshape(values.shape)

        # Bonferroni correction
        # return np.clip(self.hypothesis_count * 2 * values, 0, 1)

    def _is_edge_impl(self, value: float) -> float:
        return value < self.alpha