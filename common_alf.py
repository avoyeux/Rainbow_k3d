"""
Has functions that I regularly use.
"""

import time
import numpy as np
from functools import wraps


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


class PlotFunctions:
    """
    To store regularly used plotting functions
    """

    @staticmethod
    def Contours(mask):
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
