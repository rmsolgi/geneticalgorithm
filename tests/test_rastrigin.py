import sys
sys.path.append('..')

import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga


def f(X):
    dim = len(X)
    OF = 0
    for i in range (0, dim):
        OF+=(X[i]**2)-10*math.cos(2*math.pi*X[i])+10
    return OF


def test_rastrigin():
    parameters={'max_num_iteration': 1000,
                'population_size':200,
                'mutation_probability':0.1,
                'elit_ratio': 0.02,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type':'two_point',
                'max_iteration_without_improv':None,
                'multiprocessing_ncpus': 4,
                'multiprocessing_engine': None,
                }
    varbound = np.array([[-5.12, 5.12]]*2)

    model = ga(function=f, dimension=2, variable_type='real',
            variable_boundaries=varbound, algorithm_parameters=parameters)
    model.run()
    assert model.best_function < 1e-6


if __name__ == '__main__':
    test_rastrigin()
