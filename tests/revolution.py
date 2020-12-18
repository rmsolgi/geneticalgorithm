# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 20:05:30 2020

@author: qtckp
"""


import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import OppositionOperators
from OptimizationTestFunctions import Eggholder

from geneticalgorithm2 import geneticalgorithm2 as ga



dim = 15
np.random.seed(3)

func = Eggholder(dim = dim)

iterations = 1000    
    
varbound = np.array([[-500,500]]*dim)

model = ga(function=func, dimension=dim, 
           variable_type='real', 
           variable_boundaries=varbound,
           algorithm_parameters={
               'max_num_iteration': iterations,
               'population_size': 400
               })


start_pop = np.random.uniform(low = -500, high = 500, size = (400, dim))

start_gen = {
    'variables': start_pop,
    'scores': None
    }

# default running

model.run(no_plot = True, start_generation=start_gen)
plt.plot(model.report, label = 'without revolution')

# revolutions

model.run(no_plot = True, 
          start_generation=start_gen,
          revolution_after_stagnation_step = 80,
          revolution_part= 0.2,
          revolution_oppositor = OppositionOperators.Continual.quasi(minimums = varbound[:,0], maximums = varbound[:, 1])
          )
plt.plot(model.report, label = 'with revolution (quasi)')


model.run(no_plot = True, 
          start_generation=start_gen,
          revolution_after_stagnation_step = 80,
          revolution_part= 0.2,
          revolution_oppositor= OppositionOperators.Continual.quasi_reflect(minimums = varbound[:,0], maximums = varbound[:, 1])
          )
plt.plot(model.report, label = 'with revolution (quasi_reflect)')



plt.xlabel('Generation')
plt.ylabel('Minimized function')
plt.title('Revolution')
plt.legend()


plt.savefig("revolution.png", dpi = 300)
plt.show()