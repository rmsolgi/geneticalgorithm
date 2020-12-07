# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:14:48 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from DiscreteHillClimbing import Hill_Climbing_descent

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer


def f(arr):
    arr2 = arr/25
    return -np.sum(arr2*np.sin(np.sqrt(np.abs(arr2))))**5 + np.sum(np.abs(arr2))**2

iterations = 100    
    
varbound = np.array([[-100, 100]]*15)

available_values = [np.arange(-100, 101)]*15


my_local_optimizer = lambda arr, score: Hill_Climbing_descent(function = f, available_predictors_values=available_values, max_function_evals=50, start_solution=arr )


model = ga(function=f, dimension=varbound.shape[0], 
           variable_type='int', 
           variable_boundaries = varbound,
           algorithm_parameters={
               'max_num_iteration': iterations,
               'population_size': 400
               })


for time in ('before_select', 'after_select', 'never'):
    

    model.run(no_plot = True,
                  population_initializer = Population_initializer(
                      select_best_of = 3,
                      local_optimization_step = time,
                      local_optimizer = my_local_optimizer
                      )
                  )

    
    plt.plot(model.report, label = f"local optimization time = '{time}'")


plt.xlabel('Generation')
plt.ylabel('Minimized function (40 simulations average)')
plt.title('Selection best N object before running GA')
plt.legend()

plt.savefig("init_local_opt.png", dpi = 300)
plt.show()

