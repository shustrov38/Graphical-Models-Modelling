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
    tau, pvalue = stats.kendalltau(x, y)
    return tau
