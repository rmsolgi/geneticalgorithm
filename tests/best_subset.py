# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:09:48 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga


subset_size = 20 # how many objects we can choose

objects_count = 100 # how many objects are in set

my_set = np.random.random(objects_count)*10 - 5 # set values

# minimized function
def f(X):
    return abs(np.mean(my_set[X==1]) - np.median(my_set[X==1]))

# initialize start generation and params

N = 1000 # size of population
start_generation = np.zeros((N, objects_count))
indexes = np.arange(0, objects_count, dtype = np.int8) # indexes of variables

for i in range(N):
    inds = np.random.choice(indexes, subset_size, replace = False)
    start_generation[i, inds] = 1 


def my_crossover(parent_a, parent_b):
    a_indexes = set(indexes[parent_a == 1])
    b_indexes = set(indexes[parent_b == 1])
    
    intersect = a_indexes.intersection(b_indexes) # elements in both parents
    a_only = a_indexes - intersect # elements only in 'a' parent
    b_only = b_indexes - intersect
    
    child_inds = np.array(list(a_only) + list(b_only), dtype = np.int8)
    np.random.shuffle(child_inds) # mix
    
    childs = np.zeros((2, parent_a.size))
    if intersect:
        childs[:, np.array(list(intersect))] = 1
    childs[0, child_inds[:int(child_inds.size/2)]] = 1
    childs[1, child_inds[int(child_inds.size/2):]] = 1
    
    return childs[0,:], childs[1,:]
    

model = ga(function=f, 
           dimension=objects_count, 
           variable_type='bool',
           algorithm_parameters={
                       'max_num_iteration': 500,
                       'mutation_probability': 0, # no mutation, just crossover
                       'elit_ratio': 0.05,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': my_crossover,
                       'max_iteration_without_improv': 20
               }
           )

model.run(no_plot = False, start_generation={'variables': start_generation, 'scores': None})