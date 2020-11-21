# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:16:57 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    import math
    a = X[0]
    b = X[1]
    c = X[2]
    s = 0
    for i in range(10000):
        s += math.sin(a*i) + math.sin(b*i) + math.cos(c*i)

    return s
 

algorithm_param = {'max_num_iteration': 50,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}   
    
varbound = np.array([[-10,10]]*3)

model = ga(function=f, dimension=3, variable_type='real', variable_boundaries=varbound, algorithm_parameters = algorithm_param)

#%time model.run(no_plot = False)
#%time model.run(no_plot = False, set_function= ga.set_function_multiprocess(f, n_jobs = 6))