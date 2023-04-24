from typing import List, Union, Sequence
import numpy as np
from scipy import stats


def n_generators(
    n: int, 
    entropy: Union[None, int, Sequence[int]] = None,
    generator: np.random.BitGenerator = np.random.MT19937
) -> List[np.random.Generator]:
    """Create list of jumped numpy random generators

    Args:
        n (int): generators count
        entropy (Union[None, int, Sequence[int]], optional): The entropy for
            creating a SeedSequence. Defaults to None.
        generator (np.random.BitGenerator, optional): Random generator.
            Defaults to np.random.MT19937.

    Returns:
        List[np.random.Generator]: list of jumped generators
    """
    seed = np.random.SeedSequence(entropy)
    bit_generator = generator(seed)
    generators = []
    for _ in range(n):
        generators.append(np.random.Generator(bit_generator))
        # Chain the BitGenerators
        bit_generator = bit_generator.jumped()
    return generators


def mixed_distribution(
    mean: np.ndarray, 
    cov: np.ndarray, 
    n: int = 1, 
    gamma: float = 1., 
    df: int = 3, 
    generator: np.random.Generator = np.random.Generator(np.random.MT19937())
) -> np.ndarray:
    """Generate samples from mixed distribition with parameter gamma.
    F(x) = gamma * N(mean, cov) + (1 - gamma) * T(mean, cov, df)

    Args:
        mean (np.ndarray): Mean of the distribution
        cov (np.ndarray): Covariance of the distribution
        n (int, optional): Samples count. Defaults to 1.
        gamma (float, optional): Fraction of normal distribution. Defaults
            to 1..
        df (int, optional): Degrees of freedom. Defaults to 3.
        generator (np.random.Generator, optional): Random generator.
            Defaults to np.random.Generator(np.random.MT19937()).

    Returns:
        np.ndarray: Array of random generated samples
    """
    norm = generator.multivariate_normal(mean=mean, cov=cov, size=n)
    stud = stats.multivariate_t(loc=mean, shape=cov, 
                                df=df, seed=generator).rvs(size=n)
    data = norm
    for i in range(n):
        if np.random.rand() > gamma:
            data[i] = stud[i]
    return data