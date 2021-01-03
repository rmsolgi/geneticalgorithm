# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:13:11 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga

from geneticalgorithm2 import plot_pop_scores # for plotting scores without ga object

def f(X):
    return 50*np.sum(X) - np.sum(np.sqrt(X)*np.sin(X))
    
dim = 25
varbound = np.array([[0,10]]*dim)

# create start population
start_pop = np.random.uniform(0, 10, (50, dim))
# eval scores of start population
start_scores = np.array([f(start_pop[i]) for i in range(start_pop.shape[0])])

# plot start scores using plot_pop_scores function
plot_pop_scores(start_scores, title = 'Population scores before beggining of searching', save_as= 'plot_scores_start.png')


model = ga(function=f, dimension=dim, variable_type='real', variable_boundaries=varbound)
# run optimization process
model.run(no_plot = True,
          start_generation={
              'variables': start_pop,
              'scores': start_scores
              })
# plot and save optimization process plot
model.plot_results(save_as = 'plot_scores_process.png')

# plot scores of last population
model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= 'plot_scores_end.png')