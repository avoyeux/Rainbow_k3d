"""
Has functions that I regularly use.
"""

import time
import numpy as np

from functools import wraps
from typeguard import typechecked
from typing import Callable, TypeVar, Any 

# General function and decorator types
F = TypeVar('F', bound=Callable[..., Any])
D = Callable[[F], Any]

@typechecked
def Decorator_to_all(decorator: D, functiontype: F | str = 'all'):
    """
    Class decorator that applies a given decorator to class functions with the specified function type
    (i.e. classmethod, staticmethod, 'regular' or 'instance' -- for an instance method, 
    'all' for all the class functions).
    """

    if isinstance(functiontype, str):
        if functiontype not in ['all', 'regular', 'instance']:
            raise ValueError(f"The string value '{functiontype}' for functiontype is not supported. Choose 'regular', 'instance', or 'all'")

    def Class_rebuilder(cls):
        """
        Rebuilds the class adding the new decorators.
        """

        class NewClass(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        for name, obj in cls.__dict__.items():
            if callable(obj):
                if not isinstance(functiontype, str):
                    if isinstance(obj, functiontype):
                        original_func = obj
                        method = functiontype(decorator(original_func))
                        setattr(NewClass, name, method)
                elif functiontype=='all':
                    method = decorator(obj)
                    setattr(NewClass, name, method)
                elif not isinstance(obj, (staticmethod, classmethod)):
                    method = decorator(obj)
                    setattr(NewClass, name, method)
        return NewClass
    return Class_rebuilder
                


class decorators:
    """
    To store decorators that I use.
    """
    
    @staticmethod
    def running_time(func):
        """
        Gives the starting time (in blue) and ending time (in green) of a given function.
        The name of said function is also printed out.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            START_time = time.time()
            print(f"\033[94m{func.__name__} started on {time.ctime(START_time)}. \033[0m")
            result = func(*args, **kwargs)
            END_time = time.time()
            print(f"\033[92m{func.__name__} ended on {time.ctime(END_time)} "
                  f"({round(END_time - START_time, 2)}s).\033[0m")
            return result
        return wrapper

    @staticmethod
    def batch_processor(batch_size):
        """
        For RAM management. If the number of files, given by their path is too large, then you can use this to split the paths in
        batches and adds the output together to use less RAM. STill a draft
        """

        def decorator(func):
            @wraps(func)
            def wrapper(file_paths, *args, **kwargs):
                batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
                results = []

                for batch in batches:
                    batch_result = func(batch, *args, **kwargs)

                    
                    results.extend(batch_result)
                return results
            return wrapper
        return decorator

@Decorator_to_all(staticmethod, 'all')
@Decorator_to_all(typechecked, 'all')
class PlotFunctions:
    """
    To store regularly used plotting functions
    """

    def Contours(mask: np.ndarray) -> list:
        """
        To plot the contours given a mask
        Source: https://stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges
        """

        pad = np.pad(mask, [(1, 1), (1, 1)])  # zero padding
        im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
        im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
        lines = []
        for ii, jj in np.ndindex(im0.shape):
            if im0[ii, jj] == 1:
                lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
            if im1[ii, jj] == 1:
                lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]
        return lines


class ArrayManipulation:
    """
    To store functions related to resampling and resizing arrays.
    """

    @staticmethod
    @typechecked
    def Downsampling(array2D: np.ndarray, downsampling_size: tuple[int, int], return_npndarray: bool = True) -> np.ndarray:
        """
        To Downsample and image using PIL with the high quality Lanczos method.
        """

        # Import
        from PIL import Image

        array2D = Image.fromarray(array2D)
        array2D = array2D.resize(downsampling_size, Image.Resampling.LANCZOS)
        
        if return_npndarray:
            return np.array(array2D)
        else:
            return array2D