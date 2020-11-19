# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:37:01 2020

@author: qtckp
"""

import numpy as np
import math
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):

    dim=len(X) 
   
    a=0.5
    b=3
    OF=0
    for i in range (0,dim):
        t1=0
        for k in range (0,21):
            t1+=(a**k)*math.cos((2*math.pi*(b**k))*(X[i]+0.5))
        OF+=t1
    t2=0    
    for k in range (0,21):
        t2+=(a**k)*math.cos(math.pi*(b**k))
    OF-=dim*t2
 
    return OF
    
    
varbound=np.array([[-0.5,0.5]]*2)

algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,dimension=2,\
         variable_type='real',\
             variable_boundaries=varbound,
             algorithm_parameters=algorithm_param)

model.run()