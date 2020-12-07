import math
import random
import numpy as np

class Crossover:
    
    def get_copies(x, y):
        return x.copy(), y.copy()
    
    def one_point():
        
        def func(x, y):
            ofs1, ofs2 = Crossover.get_copies(x, y)
        
            ran=np.random.randint(0, x.size)
            
            ofs1[:ran] = y[:ran]
            ofs2[:ran] = x[:ran]
            
            return ofs1, ofs2
        return func
    
    def two_point():
        
        def func(x, y):
            ofs1, ofs2 = Crossover.get_copies(x, y)
        
            ran1=np.random.randint(0, x.size)
            ran2=np.random.randint(ran1, x.size)
            
            ofs1[ran1:ran2] = y[ran1:ran2]
            ofs2[ran1:ran2] = x[ran1:ran2]
              
            
            return ofs1, ofs2
        return func
    
    def uniform():
        
        def func(x, y):
            ofs1, ofs2 = Crossover.get_copies(x, y)
        
            ran = np.random.random(x.size) < 0.5
            ofs1[ran] = y[ran]
            ofs2[ran] = x[ran]
              
            return ofs1, ofs2
        
        return func
    
    def segment(prob = 0.6):
        
        def func(x, y):
            
            ofs1, ofs2 = Crossover.get_copies(x, y)
            
            p = np.random.random(x.size) < prob
            
            for i, val in enumerate(p):
                if val:
                    ofs1[i], ofs2[i] = ofs2[i], ofs1[i]
            
            return ofs1, ofs2
        
        return func
    
    def shuffle():
        
        def func(x, y):
            
            ofs1, ofs2 = Crossover.get_copies(x, y)
            
            index = np.random.choice(np.arange(0, x.size), x.size, replace = False)
            
            ran = np.random.randint(0, x.size)
            
            for i in range(ran):
                ind = index[i]
                ofs1[ind] = y[ind]
                ofs2[ind] = x[ind]
            
            return ofs1, ofs2
            
        return func
    
    #
    #
    # ONLY FOR REAL VARIABLES
    #
    #
    
    def arithmetic():
        
        def func(x, y):
            b = np.random.random()
            a = 1-b
            return a*x + b*y, a*y + b*x
        
        return func
    
    
    def mixed(alpha = 0.5):
        
        def func(x,y):
            
            a = np.empty(x.size)
            b = np.empty(y.size)
            
            x_min = np.minimum(a, b)
            x_max = np.maximum(a, b)
            delta = alpha*(x_max-x_min)
            
            for i in range(x.size):
                a[i] = np.random.uniform(x_min[i] - delta[i], x_max[i] + delta[i])
                b[i] = np.random.uniform(x_min[i] - delta[i], x_max[i] + delta[i])
            
            return a, b
        
        return func
        


