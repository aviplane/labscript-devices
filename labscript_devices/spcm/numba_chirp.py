from numba import jit, njit
from numpy import asarray
import numpy as np

pi = np.pi

@njit(fastmath = True)
def numba_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    """
    Calculate the phase used by numba_chirp to generate its output.
    See `chirp` for a description of the arguments.
    """
    t = asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    beta = (f1 - f0) / t1
    phase = 2 * pi * (f0 * t + 0.5 * beta * t ** 2)
    return phase

@njit(fastmath = True)
def numba_chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    phase = numba_phase(t, f0, t1, f1, method, vertex_zero)
    c = np.cos(phase + phi * pi/180)
    return c
