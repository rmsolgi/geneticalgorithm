# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:40:24 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import Crossover


crossovers = [
    Crossover.one_point(),
    Crossover.two_point(),
    Crossover.uniform(),
    Crossover.uniform_window(window = 3),
    Crossover.shuffle(),
    Crossover.segment(),
    
    Crossover.arithmetic(),
    Crossover.mixed(alpha = 0.4)
    ]



x = np.ones(15)
y = x*0

lines = []
for cr in crossovers:

    new_x, new_y = cr(x, y)
    
    print(cr.__qualname__.split('.')[1])
    print(new_x)
    print(new_y)
    print()
    
    lines += [
        f"* **{cr.__qualname__.split('.')[1]}**:\n",
        '|' + ' | '.join(np.round(new_x, 2).astype(str))+'|',
        '|' + ' | '.join( [':---:']*x.size  )+'|',
        '|' + ' | '.join(np.round(new_y, 2).astype(str))+'|',
        ''
        ]
    

with open('crossovers_example.txt', 'w') as file:
    file.writelines([line.replace('.0','') + '\n' for line in lines])







