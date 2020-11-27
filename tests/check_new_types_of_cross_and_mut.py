# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:02:24 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover, Mutations

def f(X):
    return np.sum(X)
    
varbound = np.array([[0,10]]*3)


mutations = (
    Mutations.gauss_by_center(0.2),
    Mutations.gauss_by_center(0.4),
    Mutations.gauss_by_x(),
    Mutations.gauss_by_x(0.2),
    Mutations.uniform_by_center(),
    Mutations.uniform_by_x(),
    'uniform_by_x',
    'uniform_by_center',
    'gauss_by_x',
    'gauss_by_center'
    )

crossovers = (
    Crossover.one_point(),
    'one_point',
    Crossover.two_point(),
    'two_point',
    Crossover.uniform(),
    'uniform',
    Crossover.shuffle(),
    'shuffle',
    Crossover.segment(),
    'segment',
    # only for real!!!!
    Crossover.mixed(),
    Crossover.arithmetic()
    )


for mutation in mutations:
    for crossover in crossovers:
        print(f"mutation = {mutation}, crossover = {crossover}")

        
        algorithm_param = {
                           'max_num_iteration': 400,
                           'population_size':100,
                           'mutation_probability':0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type':crossover,
                           'mutation_type': mutation,
                           'max_iteration_without_improv':None
                           }
        
        model = ga(function=f, dimension=3, 
                   variable_type='real', 
                   variable_boundaries=varbound,
                   algorithm_parameters = algorithm_param)



        model.run(no_plot = False)