from typing import final
import numpy as np


class Base:

    __PRINT_OPTIONS = np.printoptions(precision=3, suppress=True)

    def __init__(
        self, 
        n: int,
        p: int,
        true_model: np.ndarray,
        alpha: float = 0.05
    ) -> None:
        self._n, self._p = n, p

        self._hypothesis_count = self._p * (self._p - 1) // 2
        self._alpha = alpha

        self._true_model = true_model

        self.__set_errors()
    
    def __eq__(self, __other: object) -> bool:
        if not issubclass(__other.__class__, Base):
            return False

        return \
            isinstance(__other, self.__class__) and \
            self._n == __other._n and \
            self._p == __other._p and \
            (self._true_model == __other._true_model).all() and \
            self._alpha == __other._alpha

    def __neq__(self, __other: object) -> bool:
        return not (self == __other)

    def __iadd__(self, __other: object):
        if self != __other:
            raise ArithmeticError('Bad right operand.')
        
        self._num_tries += __other._num_tries
        self.__error_type_I += __other.__error_type_I
        self.__error_type_II += __other.__error_type_II
        self.__fwer_error_type_I += __other.__fwer_error_type_I
        return self

    @final
    def _update_errors(self, i: int, j: int, value: float) -> bool:
        is_error_type_I_updated = False

        if not self._true_model[i][j] and self.is_edge(value=value):
            self.__error_type_I[i][j] += 1
            is_error_type_I_updated = True

        if self._true_model[i][j] and not self.is_edge(value=value):
            self.__error_type_II[i][j] += 1

        return is_error_type_I_updated

    @final
    def __set_errors(self):
        self._num_tries = 0

        self.__error_type_I = np.zeros(self._true_model.shape)
        self.__error_type_II = np.zeros(self._true_model.shape)

        self.__fwer_error_type_I = 0
    
    @final
    def process_samples(
        self, X: np.ndarray, 
        reset: bool = False
    ) -> np.ndarray:
        """Calculate result statistics and update metrics

        Returns:
            np.ndarray: calculated statistics with corrections
        """
        if reset:
            self.__set_errors()
        self._num_tries += 1

        # center observations 
        self._X = X - X.mean(axis=0)

        # get cov and concentration matrix
        self._cov = np.cov(self._X.T, ddof=1)
        self._cov_inv = np.linalg.inv(self._cov)

        result = np.zeros(self._true_model.shape)
        is_error_updated = False

        for i in range(self._p):
            for j in range(i + 1, self._p):
                result[i][j] = self._process_samples_impl(i, j)
        
        result = self._correction_impl(result)

        for i in range(self._p):
            for j in range(i + 1, self._p):
                is_error_updated |= self._update_errors(i, j, result[i][j])
        
        self.__fwer_error_type_I += is_error_updated
        return result

    @final
    def is_edge(self, value: float) -> bool:
        """Check that the calculated statistics signal the presence of an edge

        Args:
            value (float): Input statistic

        Returns:
            bool: Means edge or not
        """
        return self._is_edge_impl(value)
    
    @final
    @property
    def error_type_I(self) -> np.ndarray:
        return self.__error_type_I / self._num_tries
    
    @final
    @property
    def error_type_II(self) -> np.ndarray:
        return self.__error_type_II / self._num_tries
    
    @final
    @property
    def fwer_error_type_I(self) -> int:
        return self.__fwer_error_type_I / self._num_tries

    def _process_samples_impl(self, i: int, j: int) -> float:
        return 0.

    def _correction_impl(self, values: np.ndarray) -> None:
        pass

    def _is_edge_impl(self, value: float) -> float:
        return value <= self._alpha
    