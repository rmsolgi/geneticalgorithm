# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:16:09 2020

@author: qtckp
"""

from OptimizationTestFunctions import Sphere, Ackley, AckleyTest, Rosenbrock, Fletcher, Griewank, Penalty2, Quartic, Rastrigin, SchwefelDouble, SchwefelMax, SchwefelAbs, SchwefelSin, Stairs, Abs, Michalewicz, Scheffer, Eggholder, Weierstrass


lines = []


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
    
    name = type(f).__name__
    
    lines.append(fr"### [{name}](https://github.com/PasaOpasen/OptimizationTestFunctions#{name.lower()})")
    
    lines.append(fr"![](https://github.com/PasaOpasen/OptimizationTestFunctions/blob/main/tests/heatmap%20for%20{name}.png)")
    
    lines.append(fr"![](tests/Optimization%20process%20for%20{name}.png)")
    
    lines.append('')
    

with open('optimization_test_func_code_for_md.txt', 'w') as file:
    file.writelines([line + '\n' for line in lines])


