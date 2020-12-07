import math
import random
import numpy as np

class Mutations:
    
    def uniform_by_x():
        
        def func(x, left, right):
            alp = min(x - left, right - x)
            return np.random.uniform(x - alp, x + alp)
        return func
    
    
    def uniform_by_center():
        
        def func(x, left, right):
            return np.random.uniform(left, right)
        
        return func
    
    def gauss_by_x(sd = 0.3):
        """
        gauss mutation with x as center and sd*length_of_zone as std
        """
        def func(x, left, right):
            std = sd * (right - left)
            return max(left, min(right, np.random.normal(loc = x, scale = std)))
        
        return func
    
    def gauss_by_center(sd = 0.3):
        """
        gauss mutation with (left+right)/2 as center and sd*length_of_zone as std
        """
        def func(x, left, right):
            std = sd * (right - left)
            return max(left, min(right, np.random.normal(loc = (left+right)*0.5, scale = std)))
        
        return func