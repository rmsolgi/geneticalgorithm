
import numpy as np

from functools import lru_cache, wraps
#from fastcache import clru_cache



def np_lru_cache(*args, **kwargs):
    """
    LRU cache implementation for functions whose FIRST parameter is a numpy array
        forked from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
    """

    def decorator(function):
        
        @wraps(function)
        def wrapper(np_array):
            return cached_wrapper( tuple(np_array))

        @lru_cache(*args, **kwargs)
        #@clru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array):
            return function(np.array(hashable_array))

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    
    return decorator





if __name__ == '__main__':

    ar = np.random.randint(0, 100, (1000, 100))

    f = lambda arr: np.std(arr + arr/(1 + arr**2) - arr + np.sin(arr) * np.cos(arr) + 2)

    def no_c(arr):
        return f(arr)
    
    @np_lru_cache(maxsize = 700, typed = True)
    def with_c(arr):
        return f(arr)

    #%time for _ in range(50): [no_c(arr) for arr in ar[np.random.rand(ar.shape[0]).argsort()]]
    #%time for _ in range(50): [with_c(arr) for arr in ar[np.random.rand(ar.shape[0]).argsort()]]





