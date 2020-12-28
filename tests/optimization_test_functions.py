# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:25:17 2020

@author: qtckp
"""
import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

from OptimizationTestFunctions import Sphere, Ackley, AckleyTest, Rosenbrock, Fletcher, Griewank, Penalty2, Quartic, Rastrigin, SchwefelDouble, SchwefelMax, SchwefelAbs, SchwefelSin, Stairs, Abs, Michalewicz, Scheffer, Eggholder, Weierstrass


dim = 2

functions = [
        Sphere(dim, degree = 2),
        Ackley(dim),
        AckleyTest(dim),
        Rosenbrock(dim),
        Fletcher(dim, seed = 1488),
        Griewank(dim),
        Penalty2(dim),
        Quartic(dim),
        Rastrigin(dim),
        SchwefelDouble(dim),
        SchwefelMax(dim),
        SchwefelAbs(dim),
        SchwefelSin(dim),
        Stairs(dim),
        Abs(dim),
        Michalewicz(),
        Scheffer(dim),
        Eggholder(dim),
        Weierstrass(dim)
    ]




for f in functions:

    xmin, xmax, ymin, ymax = f.bounds
        
    varbound = np.array([[xmin, xmax], [ymin, ymax]])
    
    model = ga(function=f,
               dimension = dim,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters = {
                       'max_num_iteration': 500,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type':'uniform',
                       'mutation_type': 'uniform_by_center',
                       'selection_type': 'roulette',
                       'max_iteration_without_improv':100
                       })
    
    model.run(no_plot = True, stop_when_reached = (f.f_best + 1e-5/(xmax - xmin)) if not (f.f_best is None) else None)
    
    title = f"Optimization process for {type(f).__name__}"
    
    model.plot_results(title = title, save_as = f"{title}.png", main_color = 'green')
