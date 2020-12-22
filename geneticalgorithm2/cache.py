import numpy as np
from functools import lru_cache, wraps



def npCacheMap(*args, **kwargs):
    ''' LRU cache implementation for functions whose FIRST parameter is a numpy array
        forked from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator






