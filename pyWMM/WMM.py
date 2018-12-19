# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

import numpy as np
import scipy
from scipy import integrate
from scipy.linalg import expm, pinv

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

def complex_quadrature(func, xmin, xmax, ymin, ymax,nx=5e2,ny=1e2, **kwargs):
    nx = int(nx); ny = int(ny);
    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    X, Y = np.meshgrid(x,y,sparse=True, indexing='ij')
    z = np.zeros((nx,ny),dtype=np.complex128)
    z = func(Y,X)
    ans = np.trapz(np.trapz(z, y), x)
    return ans
'''
def complex_quadrature(func, xmin, xmax, ymin, ymax, **kwargs):
    def real_func(y,x):
        return scipy.real(func(y,x))
    def imag_func(y,x):
        return scipy.imag(func(y,x))
    real_integral = integrate.dblquad(real_func,xmin,xmax,lambda x: ymin, lambda x: ymax)
    imag_integral = integrate.dblquad(imag_func,xmin,xmax,lambda x: ymin, lambda x: ymax)
    return real_integral[0] + 1j*imag_integral[0]
'''

def TMM(func,A0,zmin,zmax,nz):
    z = np.linspace(zmin,zmax,nz)
    dz = z[1] - z[0]

    a = np.zeros((nz,2,2),dtype=np.complex128)
    f = np.zeros((nz,2),dtype=np.complex128)
    F_bank = np.zeros((nz,2,2),dtype=np.complex128)
    S_bank = np.zeros((nz,2,2),dtype=np.complex128)
    Q_bank = np.zeros((nz,2,2),dtype=np.complex128)

    # evaluate function
    for iter in range(nz):
        a[iter,:,:],Q_bank[iter,:,:] = np.squeeze(func(z[iter]))

    # Initialize routine
    F = np.identity(2)
    F_bank[0,:,:] = np.matmul(expm(-z[0]*a[0]),F)
    f[0,:] = F_bank[0,:,:].dot(A0)
    P = Q_bank[0,:,:]
    Pinv = pinv(P)
    S_bank[0] = np.matmul(np.matmul(Q_bank[0], F_bank[0]),Pinv)
    # multiply by transfer matrix
    for iter in range(1,nz):
        mat = 0.5 * (z[iter]*a[iter]-z[iter-1]*a[iter-1]-z[iter]*a[iter-1]+z[iter-1]*a[iter])
        mat = expm(mat)
        F   = np.matmul(mat,F)
        #F_bank[iter,:,:] = F
        F_bank[iter,:,:] = np.matmul(expm(-z[iter]*a[iter]),F)
        print('++++++++')
        print(F_bank[iter,:,:])
        #np.exp(-1j*beta*(z[iter]-zmin))
        A = np.squeeze(np.array([1,0]))
        f[iter,:] = F_bank[iter,:,:].dot(A)
        S_bank[iter] = np.matmul(Q_bank[iter], np.matmul(F_bank[iter],Pinv))
        print(S_bank[iter,:,:])
        print('++++++++')
    return f, F_bank, S_bank
