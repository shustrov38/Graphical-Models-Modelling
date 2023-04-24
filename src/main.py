from libs.distibutions import mixed_distribution, n_generators
from libs.statistics import Pearson
import numpy as np


def pcorr_from_cov(cov: np.ndarray) -> np.ndarray:
    p = cov.shape[1]
    cov_inv = np.linalg.inv(cov)

    pcorr = np.zeros((p, p))
    for i in range(p):
        for j in range(p):  
            pcorr[i, j] = -cov_inv[i, j] / np.sqrt(cov_inv[i, i] * cov_inv[j, j])
        pcorr[i, i] = 1

    return pcorr


if __name__ == '__main__':
    p = 4
    mean = np.zeros(p)
    cov_inv = np.array([
        [5,2,1,0], 
        [2,2,0,1], 
        [1,0,2,1], 
        [0,1,1,2]
    ])
    cov = np.array([
        [4,-7,-5,6], 
        [-7,13,9,-11], 
        [-5,9,7,-8],
        [6,-11,-8,10]
    ])
    print(pcorr_from_cov(cov))
    with np.printoptions(precision=3, suppress=True):
        gen = n_generators(1, 1234)[0]

        X = mixed_distribution(
            mean=mean,
            cov=cov, 
            n=3000,
            generator=gen,
        )

        pearson = Pearson(X, (cov_inv != 0))

        print(pearson.value())

