import numpy as np
from scipy import stats


def z_fisher_transform(x: np.ndarray) -> np.ndarray:
    """Fisher z-transformation of matrix x

    Args:
        x (np.ndarray): input matrix

    Returns:
        np.ndarray: transformed matrix
    """
    return np.log((1 + x) / (1 - x)) / 2


def laplace_transform(x: np.ndarray) -> np.ndarray:
    """Laplace transform of matrix x

    Args:
        x (np.ndarray): input matrix

    Returns:
        np.ndarray: transformed matrix
    """
    return stats.norm.cdf(x)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Kendalls correlation coefficient

    Args:
        x (np.ndarray): lhs
        y (np.ndarray): rhs

    Returns:
        float: kendalls correlation coefficient
    """
    tau, _ = stats.kendalltau(x, y)
    return tau


def bonferroni_method(
    pvalues: np.ndarray,
    hypothesis_count: int, 
    alpha: float
) -> np.ndarray:
    return np.array(hypothesis_count * 2 * pvalues > alpha, dtype=float)
