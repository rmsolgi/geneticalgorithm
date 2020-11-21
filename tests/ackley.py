# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:25:17 2020

@author: qtckp
"""
import sys
sys.path.append('..')


import math
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):

    dim=len(X)
        
    t1 = np.sum(X**2)
    t2 = np.sum(np.cos(2*math.pi*X)) 
            
    OF=20+math.e-20*math.exp((t1/dim)*-0.2)-math.exp(t2/dim)
 
    return OF
    
varbound=np.array([[-32.768,32.768]]*2)

model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound)

model.run()