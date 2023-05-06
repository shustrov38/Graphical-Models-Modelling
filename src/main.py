from libs.distibutions import mixed_distribution, n_generators
from libs.statistics.pearson import Pearson
from libs.statistics.kendall import Kendall
import numpy as np

from typing import Tuple, Any
import multiprocessing


def process(args: Tuple[Any]) -> None:
    num_tries, gen, n, mean, cov, model = args

    for _ in range(num_tries):
        X = mixed_distribution(
            mean=mean,
            cov=cov, 
            n=n,
            generator=gen,
        )

        model.process_samples(X)

    return model

if __name__ == '__main__':
    n = 30
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
    true_model = (cov_inv != 0)


    with np.printoptions(precision=3, suppress=True):
        num_procs = 10

        gen = n_generators(n=num_procs)
        models = [Kendall(n, p, true_model, correction_method='holm') for _ in range(num_procs)]
        
        num_tries = 10000

        step = num_tries // num_procs
        args = []
        for i in range(num_procs):
            args.append((step, gen[i], n, mean, cov, models[i]))

        with multiprocessing.Pool(processes=num_procs) as pool:
            models = pool.map(process, args)

        result_model = models[0]
        for i in range(1, num_procs):
            result_model += models[i]

        print(result_model.__class__)
        print('Ошибка первого рода:')
        print(result_model.error_type_I)
        print('FWER:', result_model.fwer_error_type_I)
        print()
        print('Ошибка второго рода:')
        print(result_model.error_type_II)