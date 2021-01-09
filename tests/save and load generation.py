# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:41:47 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


from OptimizationTestFunctions import Eggholder


dim = 2*15

f =  Eggholder(dim)

xmin, xmax, ymin, ymax = f.bounds
        
varbound = np.array([[xmin, xmax], [ymin, ymax]]*15)
    

model = ga(function=f,
               dimension = dim,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters = {
                       'max_num_iteration': 300,
                       'population_size': 100
                       })

# run and save last generation to file
filename = "eggholder_lastgen.npz"
model.run(save_last_generation_as = filename)


# load start generation from file
model.run(start_generation=filename)


