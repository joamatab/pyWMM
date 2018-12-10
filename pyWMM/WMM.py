# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

import numpy as np
import scipy
from scipy import integrate

# ---------------------------------------------------------------------------- #
# Define constants
# ---------------------------------------------------------------------------- #

EPS0 = 8.854e-12 * 1e-6
MU0  = 4 * np.pi* 1e-7
C0   = 299792458

CARTESIAN   = 0
CYLINDRICAL = 1

E   = 0
H   = 1
Eps = 2



# ---------------------------------------------------------------------------- #
# Useful functions
# ---------------------------------------------------------------------------- #

def complex_quadrature(func, xmin, xmax, ymin, ymax,nx=1e2,ny=1e2, **kwargs):
    nx = int(nx); ny = int(ny);
    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    z = np.zeros((nx,ny),dtype=np.complex128)
    z = func(y,x)
    ans = np.trapz(np.trapz(z, x), y)
    return ans
'''
def complex_quadrature(func, xmin, xmax, ymin, ymax, **kwargs):
    def real_func(y,x):
        return scipy.real(func(y,x))
    def imag_func(y,x):
        return scipy.imag(func(y,x))
    real_integral = integrate.dblquad(real_func,xmin,xmax,lambda x: ymin, lambda x: ymax)
    imag_integral = integrate.dblquad(imag_func,xmin,xmax,lambda x: ymin, lambda x: ymax)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
'''
