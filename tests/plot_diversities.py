# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:58:08 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

from geneticalgorithm2 import MiddleCallbacks


dim = 6
rd = np.random.random(size = dim)


def f(X):
    return np.mean(X - rd)


    
varbound = np.array([[0, 1]]*dim)

model = ga(function=f, dimension = dim, variable_type='real', variable_boundaries=varbound,
           
           algorithm_parameters = {'max_num_iteration': 1000,
                   'population_size':50,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'mutation_type': 'uniform_by_center',
                   'selection_type': 'roulette',
                   'max_iteration_without_improv':None}
           )



model.run(
    no_plot = False,
    middle_callbacks = [
        
        MiddleCallbacks.GeneDiversityStats(20)
        
        ]

          )



