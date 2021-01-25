import math
import random
import numpy as np



def get_copies(x, y):
    return x.copy(), y.copy()


class Crossover:
    
    @staticmethod
    def one_point():
        
        def func(x, y):
            ofs1, ofs2 = get_copies(x, y)
        
            ran=np.random.randint(0, x.size)
            
            ofs1[:ran] = y[:ran]
            ofs2[:ran] = x[:ran]
            
            return ofs1, ofs2
        return func
    
    @staticmethod
    def two_point():
        
        def func(x, y):
            ofs1, ofs2 = get_copies(x, y)
        
            ran1=np.random.randint(0, x.size)
            ran2=np.random.randint(ran1, x.size)
            
            ofs1[ran1:ran2] = y[ran1:ran2]
            ofs2[ran1:ran2] = x[ran1:ran2]
              
            
            return ofs1, ofs2
        return func
    
    @staticmethod
    def uniform():
        
        def func(x, y):
            ofs1, ofs2 = get_copies(x, y)
        
            ran = np.random.random(x.size) < 0.5
            ofs1[ran] = y[ran]
            ofs2[ran] = x[ran]
              
            return ofs1, ofs2
        
        return func
    
    @staticmethod
    def segment(prob = 0.6):
        
        def func(x, y):
            
            ofs1, ofs2 = get_copies(x, y)
            
            p = np.random.random(x.size) < prob
            
            for i, val in enumerate(p):
                if val:
                    ofs1[i], ofs2[i] = ofs2[i], ofs1[i]
            
            return ofs1, ofs2
        
        return func
    
    @staticmethod
    def shuffle():
        
        def func(x, y):
            
            ofs1, ofs2 = get_copies(x, y)
            
            index = np.random.choice(np.arange(0, x.size), x.size, replace = False)
            
            ran = np.random.randint(0, x.size)
            
            for i in range(ran):
                ind = index[i]
                ofs1[ind] = y[ind]
                ofs2[ind] = x[ind]
            
            return ofs1, ofs2
            
        return func
    
    @staticmethod
    def uniform_window(window = 7):

        base_uniform = Crossover.uniform()

        def func(x, y):

            if x.size % window != 0:
                raise Exception(f"dimension {x.size} cannot be divided by window {window}")
            
            items = int(x.size/window)

            zip_x, zip_y = base_uniform(np.zeros(items), np.ones(items))
            
            ofs1 = np.empty(x.size)
            ofs2 = np.empty(x.size)
            for i in range(items):
                sls = slice(i*window, (i+1)*window, 1)
                if zip_x[i] == 0:
                    ofs1[sls] = x[sls]
                    ofs2[sls] = y[sls]
                else:
                    ofs2[sls] = x[sls]
                    ofs1[sls] = y[sls]                    

            return ofs1, ofs2
        

        return func




    #
    #
    # ONLY FOR REAL VARIABLES
    #
    #
    
    @staticmethod
    def arithmetic():
        
        def func(x, y):
            b = random.random()
            a = 1-b
            return a*x + b*y, a*y + b*x
        
        return func
    
    @staticmethod
    def mixed(alpha = 0.5):
        
        def func(x,y):
            
            a = np.empty(x.size)
            b = np.empty(y.size)
            
            x_min = np.minimum(x, y)
            x_max = np.maximum(x, y)
            delta = alpha*(x_max-x_min)
            
            for i in range(x.size):
                a[i] = np.random.uniform(x_min[i] - delta[i], x_max[i] + delta[i])
                b[i] = np.random.uniform(x_min[i] + delta[i], x_max[i] - delta[i])
            
            return a, b
        
        return func
        


