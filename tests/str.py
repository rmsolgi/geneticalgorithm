# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:52:41 2021

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


print(str(model))


