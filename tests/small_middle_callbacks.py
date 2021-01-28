# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:01:49 2021

@author: qtckp
"""


import sys
sys.path.append('..')


import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks
from geneticalgorithm2 import Crossover, Mutations


def f(X):
    return np.sum(X)
    
    
varbound = np.array([[0,10]]*20)

model = ga(function=f, 
           dimension=20, 
           variable_type='real', 
           variable_boundaries=varbound)

model.run(
    no_plot = False,
    
    middle_callbacks = [
        #MiddleCallbacks.UniversalCallback(Actions.Stop(), ActionConditions.EachGen(30)),
        #MiddleCallbacks.UniversalCallback(Actions.ReduceMutationProb(reduce_coef = 0.98), ActionConditions.EachGen(30)),
        MiddleCallbacks.UniversalCallback(Actions.ChangeRandomCrossover([
            Crossover.shuffle(),
            Crossover.two_point()
            ]), 
                                          ActionConditions.EachGen(30)),
        MiddleCallbacks.UniversalCallback(Actions.ChangeRandomMutation([
            Mutations.uniform_by_x(),
            Mutations.gauss_by_x()
            ]), 
                                          ActionConditions.EachGen(50))
        ]
          )


