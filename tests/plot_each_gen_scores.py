# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:17:30 2021

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks


def f(X):
    return np.sum(np.abs(X-50))
    
    

dim = 100
    
varbound = np.array([[0,70]]*dim)

model = ga(function=f, 
           dimension=dim, 
           variable_type='int', 
           variable_boundaries=varbound
           )

model.run(           
    middle_callbacks = [
               MiddleCallbacks.UniversalCallback(Actions.PlotPopulationScores(
                   title_pattern= lambda data: f"Gen {data['current_generation']}",
                   save_as_name_pattern=lambda data: f"{data['last_generation']['scores'].min()}.png"
                   ), ActionConditions.EachGen(1))
               ])