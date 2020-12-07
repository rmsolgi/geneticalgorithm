
import math
import random
import numpy as np



class Selection:
    
    def __inverse_scores(scores):
        """
        inverse scores (min val goes to max)
        """
        minobj = scores[0]
        normobj = scores - minobj if minobj < 0 else scores
                
        return (np.amax(normobj) + 1) - normobj 
    
    
    def fully_random():
        
        def func(scores, parents_count):
            indexes = np.arange(parents_count)
            return np.random.choice(indexes, parents_count, replace = False)
        
        return func
    
    def __roulette(scores, parents_count):
        
        sum_normobj = np.sum(scores)
        prob = scores/sum_normobj
        cumprob = np.cumsum(prob)            
            
        parents_indexes = np.empty(parents_count)
            
        # it can be vectorized
        for k in range(parents_count):
            index = np.searchsorted(cumprob, np.random.random())
            if index < cumprob.size:
                parents_indexes[k] = index
            else:
                parents_indexes[k] = np.random.randint(0, index - 1)
            
        return parents_indexes

    
    def roulette():
        
        def func(scores, parents_count):

            normobj = Selection.__inverse_scores(scores)

            return Selection.__roulette(normobj, parents_count)
        
        return func
    
    def stochastic():
        
        def func(scores, parents_count):
            f = Selection.__inverse_scores(scores)
            
            fN = 1.0/parents_count
            k = 0
            acc = 0.0
            parents = []
            r = random.random()*fN
            
            while len(parents) < parents_count:
                
                acc += f[k]
                
                while acc > r:
                    parents.append(k)
                    if len(parents) == parents_count: break
                    r += fN
                
                k += 1
            
            return np.array(parents[:parents_count])
        
        return func
    
    def sigma_scaling(epsilon = 0.01, is_noisy = False):
        
        def func(scores, parents_count):
            f = Selection.__inverse_scores(scores)
            
            sigma = np.std(f, ddof = 1) if is_noisy else np.std(f)
            average = np.mean(f)
            
            if sigma == 0:
                f = 1
            else:
                f = np.maximum(epsilon, 1 + (f - average)/(2*sigma))
            
            return Selection.__roulette(f, parents_count)
        
        return func
    
    def ranking():
        
        def func(scores, parents_count):
            return Selection.__roulette(1 + np.arange(parents_count)[::-1], parents_count)
        
        return func
    
    def linear_ranking(selection_pressure = 1.5):
        
        assert (selection_pressure > 1 and selection_pressure < 2), f"selection_pressure should be in (1, 2), but got {selection_pressure}"
        
        def func(scores, parents_count):
            tmp = parents_count*(parents_count-1)
            alpha = (2*parents_count - selection_pressure*(parents_count + 1))/tmp
            beta = 2*(selection_pressure - 1)/tmp
            
            
            a = -2*alpha - beta
            b = (2*alpha + beta)**2
            c = 8*beta
            d = 2*beta
            
            indexes = np.arange(parents_count)
            
            return np.array([indexes[-round((a + math.sqrt(b + c*random.random()))/d)] for _ in range(parents_count)])
            
        
        return func
    
    def tournament(tau = 2):
        
        def func(scores, parents_count):
            
            indexes = np.arange(parents_count)
            
            return np.array([np.min(np.random.choice(indexes, tau, replace = False)) for _ in range(parents_count)])
            
        
        return func
    
    
    

















