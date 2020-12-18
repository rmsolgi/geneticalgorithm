# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:40:38 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from OppOpPopInit import OppositionOperators
from OptimizationTestFunctions import Ackley

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer


dim = 15

func = Ackley(dim = dim)

iterations = 150    
    
varbound = np.array([[-4,3]]*dim)

model = ga(function=func, dimension=dim, 
           variable_type='real', 
           variable_boundaries=varbound,
           algorithm_parameters={
               'max_num_iteration': iterations,
               'population_size': 400
               })


oppositors = [
    None,
    [OppositionOperators.Continual.quasi(minimums = varbound[:,0], maximums = varbound[:, 1])],
    [
     OppositionOperators.Continual.quasi(minimums = varbound[:,0], maximums = varbound[:, 1]),
     OppositionOperators.Continual.over(minimums = varbound[:,0], maximums = varbound[:, 1])
     ]
    ]

names = [
    'No oppositor, just random',
    'quasi oppositor',
    'quasi + over oppositors'
    ]


for opp, name in zip(oppositors, names):
    
    average_report = np.zeros(iterations+1)
    
    for _ in range(40):
        model.run(no_plot = True,
                  population_initializer=Population_initializer(select_best_of = 3),
                  init_oppositors=opp
                  )
        average_report += np.array(model.report)
   
    average_report /= 40
    
    plt.plot(average_report, label = name)


plt.xlabel('Generation')
plt.ylabel('Minimized function (40 simulations average)')
plt.title('Start gen. using oppositors')
plt.legend()


plt.savefig("init_best_of_opp.png", dpi = 300)
plt.show()