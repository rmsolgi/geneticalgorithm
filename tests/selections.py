# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:55:59 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Selection

def f(X):
    return np.sum(X)

dim = 50
    
varbound = np.array([[0,10]]*dim)

selections = [
    (Selection.fully_random(),'fully_random'),
    (Selection.roulette(),'roulette'),
    (Selection.stochastic(),'stochastic'),
    (Selection.sigma_scaling(epsilon = 0.05),'sigma_scaling; epsilon = 0.05'),
    (Selection.ranking(),'ranking'),
    (Selection.linear_ranking(selection_pressure = 1.5),'linear_ranking; selection_pressure = 1.5'),
    (Selection.linear_ranking(selection_pressure = 1.9),'linear_ranking; selection_pressure = 1.9'),
    (Selection.tournament(tau = 2),'tournament; size = 2'),
    (Selection.tournament(tau = 4),'tournament; size = 4')
    ]


start_gen = np.random.uniform(0, 10, (100, dim))


for sel, lab in selections:
        
    model = ga(function=f, dimension=dim, 
               variable_type='real', 
               variable_boundaries=varbound,
               algorithm_parameters = {
                   'max_num_iteration': 400,
                   'selection_type': sel
                   })
    
    model.run(no_plot = True, start_generation={'variables':start_gen, 'scores': None})
    
    plt.plot(model.report, label = lab)

    
plt.xlabel('Generation')
plt.ylabel('Minimized function (sum of array)')
plt.title('Several selection types for one task')
plt.legend(fontsize=8)


plt.savefig("selections.png", dpi = 300)
plt.show()