# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:24:09 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    return np.sum(X)
    
    
varbound = np.array([[0,10]]*20)

model = ga(function=f, dimension=20, variable_type='real', variable_boundaries=varbound)

model.run(no_plot = False,
          time_limit_secs = 3)


from geneticalgorithm2 import time_to_seconds

model.run(no_plot = False,
          time_limit_secs = time_to_seconds(minutes = 0.5, seconds = 2))

