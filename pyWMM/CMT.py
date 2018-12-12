# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
import numpy as np
from scipy import integrate
from pyWMM import WMM as wmm
from pyWMM import mode
from scipy import linalg

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''
   Coupled mode theory:

   Input:

   Output:
'''
def CMTsetup(modeList,xmin,xmax,ymin,ymax,z=0):
    n = len(modeList)

    S = np.zeros((n,n),dtype=np.complex128)
    C = np.zeros((n,n),dtype=np.complex128)

    # TODO: Validate input

    omega = modeList[0].omega

    # Calculate full permittivity
    def eps_full(x,y):
        eps_bank = (np.zeros((n,x.size,y.size),dtype=np.complex128))
        for listIter in range(n):
            eps_bank[listIter,:,:] = modeList[listIter].Eps(x,y,z)
        return np.max(eps_bank,axis=0)

    # Iterate through modes
    for rowIter in range(n):
        for colIter in range(n):

            # Calculate left hand side (S matrix)
            m = modeList[rowIter]
            k = modeList[colIter]
            integrand = lambda y,x: (\
            m.Ex(x,y,z) * k.Hy(x,y,z).conj() - \
            m.Ey(x,y,z) * k.Hx(x,y,z).conj() + \
            k.Ex(x,y,z).conj() * m.Hy(x,y,z) - \
            k.Ey(x,y,z).conj() * m.Hx(x,y,z)) \
            / np.sqrt(m.total_power*k.total_power)

            intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
            S[rowIter,colIter] = intresult
            #S[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)

            # Calculate right hand side (C matrix)
            if rowIter == colIter:
                C[rowIter,colIter] = 0.0
            else:
                m = modeList[rowIter]
                k = modeList[colIter]
                integrand = lambda y,x: \
                -1j * 2 * np.pi * omega * wmm.EPS0 / np.sqrt(m.total_power*k.total_power) *\
                (eps_full(x,y) - k.Eps(x,y,z)) * \
                (m.Ex(x,y,z) * k.Ex(x,y,z).conj() + \
                 m.Ey(x,y,z) * k.Ey(x,y,z).conj() + \
                 m.Ez(x,y,z) * k.Ez(x,y,z).conj())

                intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
                C[rowIter,colIter] = intresult
                #C[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)
    result = np.matmul(linalg.pinv(S), C)
    return result

def getCrossSection(modeList,x,y,z=0):
    n = len(modeList)
    eps_bank = (np.zeros((n,x.size,y.size),dtype=np.complex128))
    for listIter in range(n):
        eps_bank[listIter,:,:] = modeList[listIter].Eps(x,y,z)
    return np.max(eps_bank,axis=0).T

def getCrossSection_Ex(modeList,x,y,z=0):
    n = len(modeList)
    ey_bank = (np.zeros((n,x.size,y.size),dtype=np.complex128))
    for listIter in range(n):
        ey_bank[listIter,:,:] = modeList[listIter].Ex(x,y,z)
    return np.sum(ey_bank,axis=0).T

def getCrossSection_Ey(modeList,x,y,z=0):
    n = len(modeList)
    ex_bank = (np.zeros((n,x.size,y.size),dtype=np.complex128))
    for listIter in range(n):
        ex_bank[listIter,:,:] = modeList[listIter].Ez(x,y,z)
    return np.sum(ex_bank,axis=0).T

def getTopView(modeList,x,z,y=0):
    n = len(modeList)
    eps_bank = (np.zeros((n,x.size,z.size),dtype=np.complex128))
    for listIter in range(n):
        eps_bank[listIter,:,:] = modeList[listIter].Eps(x,y,z)
    return np.max(eps_bank,axis=0).T

def getTopView_Ex(modeList,x,z,y=0):
    n = len(modeList)
    ex_bank = (np.zeros((n,x.size,z.size),dtype=np.complex128))
    for listIter in range(n):
        ex_bank[listIter,:,:] = modeList[listIter].Ex(x,y,z)
    return np.sum(ex_bank,axis=0).T

def getTopView_Ey(modeList,x,z,y=0):
    n = len(modeList)
    ey_bank = (np.zeros((n,x.size,z.size),dtype=np.complex128))
    for listIter in range(n):
        ey_bank[listIter,:,:] = modeList[listIter].Ey(x,y,z)
    return np.sum(ey_bank,axis=0).T

def getTopView_Ez(modeList,x,z,y=0):
    n = len(modeList)
    ez_bank = (np.zeros((n,x.size,z.size),dtype=np.complex128))
    for listIter in range(n):
        ez_bank[listIter,:,:] = modeList[listIter].Ez(x,y,z)
    return np.sum(ez_bank,axis=0).T

def makeSupermode(mode1, mode2, x, y):
    #X, Y  = np.meshgrid(x,y)
    numX = x.size; numY = y.size;
    #X = X.flatten(); Y = Y.flatten();
    Eps1 = (mode1.Eps(x,y))
    Eps2 = (mode2.Eps(x,y))

    Eps = Eps1 + Eps2
    return Eps.T
