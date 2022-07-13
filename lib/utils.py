import numpy as np
from numba import njit, f8, i8

@njit("f8(f8[:])")
def multimodal_function(X: list[f8]) -> f8:
    r = 0
    for x in X:
        r += x ** 4 - 16 * x ** 2 + 0.2 * x
    return r / 2

@njit("f8(f8[:])")
def target_function(X: list[f8]) -> f8:
    return - multimodal_function(X)
