# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:46:10 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    return np.sum(X)
    
    
varbound = np.array([[0,30]]*20)

model = ga(function=f, dimension=20, variable_type='real', variable_boundaries=varbound)

model.run(
    no_plot = True,
    disable_progress_bar=True,
    disable_printing=True
    )