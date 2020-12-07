# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:32:18 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
import matplotlib.pyplot as plt

dim = 25

def f(X):
    return np.sum(X)
    
    
varbound = np.array([[0,10]]*dim)

start_gen = np.random.uniform(0, 10, (100, dim))

ratios = [0, 0.02, 0.05, 0.1]

for elit in ratios:

    model = ga(function=f, dimension=dim, variable_type='real', 
               variable_boundaries=varbound,
               algorithm_parameters={
                   'max_num_iteration': 400,
                   'elit_ratio': elit
                   })
    
    model.run(no_plot = True, start_generation={'variables':start_gen, 'scores': None})
    
    plt.plot(model.report, label = f"elit_ratio = {elit}")
    
    
    
plt.xlabel('Generation')
plt.ylabel('Minimized function')
plt.title('Standard GA vs. Elitist GA')
plt.legend()


plt.savefig("standard_vs_elitist.png", dpi = 300)
plt.show()


