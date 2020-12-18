# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:14:37 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer


def f(X):
    return 10*np.sum(X) - X[0] - X[1] - X[2]*X[3] + X[6]**2 + 1/(0.01 + X[5])

dim = 15
iterations = 150    
    
varbound = np.array([[0,10]]*dim)

model = ga(function=f, dimension=dim, 
           variable_type='real', 
           variable_boundaries=varbound,
           algorithm_parameters={
               'max_num_iteration': iterations,
               'population_size': 400
               })


for best_of in (1, 3, 5):
    
    average_report = np.zeros(iterations+1)
    
    for _ in range(40):
        model.run(no_plot = True,
                  population_initializer=Population_initializer(select_best_of = best_of)
                  )
        average_report += np.array(model.report)
   
    average_report /= 40
    
    plt.plot(average_report, label = f"selected best N from {best_of}N")


plt.xlabel('Generation')
plt.ylabel('Minimized function (40 simulations average)')
plt.title('Selection best N objects before running GA')
plt.legend()


plt.savefig("init_best_of.png", dpi = 300)
plt.show()