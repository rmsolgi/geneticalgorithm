# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:15:55 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import OppositionOperators

from geneticalgorithm2 import geneticalgorithm2 as ga



dim = 15
np.random.seed(3)

rands = np.random.uniform(-10, 10, 100)

def func(X):
    return np.sum(rands[X.astype(int)]) + X.sum()

iterations = 900    
    
varbound = np.array([[0,99]]*dim)

model = ga(function=func, dimension=dim, 
           variable_type='int', 
           variable_boundaries=varbound,
           algorithm_parameters={
               'max_num_iteration': iterations
               })


start_pop = np.random.randint(0, 10, size = (100, dim))

start_gen = {
    'variables': start_pop,
    'scores': None
    }


np.random.seed(3)
model.run(no_plot = True, start_generation=start_gen)
plt.plot(model.report, label = 'without dups removing')

np.random.seed(3)
model.run(no_plot = True, 
          start_generation=start_gen,
          remove_duplicates_generation_step = 40,
          )

plt.plot(model.report, label = 'with dups removing + random replace')

np.random.seed(3)
model.run(no_plot = True, 
          start_generation=start_gen,
          remove_duplicates_generation_step = 40,
          duplicates_oppositor=OppositionOperators.Discrete.integers_by_order(minimums = varbound[:,0], maximums = varbound[:, 1])
          )

plt.plot(model.report, label = 'with dups removing + opposion replace')




plt.xlabel('Generation')
plt.ylabel('Minimized function')
plt.title('Duplicates removing')
plt.legend()


plt.savefig("remove_dups.png", dpi = 300)
plt.show()