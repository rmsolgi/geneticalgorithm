# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:54:11 2020

@author: qtckp
"""
import sys
sys.path.append('..')

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks


dim = 16

def f(X):
    pen=0
    if np.sum(X) < 1:
        pen=500+10*(1-np.sum(X))
    return np.sum(X)+pen
    
varbound=np.array([[0,10]]*dim)

model=ga(function=f,
         dimension=dim,
         variable_type='real',
         variable_boundaries=varbound,
         algorithm_parameters={
             'max_num_iteration': 2000
             })

model.run(
    callbacks=[
        Callbacks.SavePopulation('callback_pop_example', save_gen_step=500, file_prefix='constraints'),
        Callbacks.PlotOptimizationProcess('callback_plot_example', save_gen_step=300, show = False, main_color='red', file_prefix='plot')
        ]
    )