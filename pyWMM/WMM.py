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

EPS0 = 8.854e-12
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
def complex_quadrature(func, xmin, xmax, ymin, ymax,nx=1e3,ny=1e3, **kwargs):
    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    z = func(x,y)
    ans = np.simps(np.simps(z, y), x)
    print('===================')
    print(ans)
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
