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
def CMTsetup(modeList,xmin,xmax,ymin,ymax):
    n = len(modeList)

    S = np.zeros((n,n),dtype=np.complex128)
    C = np.zeros((n,n),dtype=np.complex128)

    ez = np.array([0, 0, 1])

    # TODO: Validate input

    omega = modeList[0].omega

    # Calculate full permittivity
    def eps_full(x,y):
        eps_bank = (np.zeros((n,x.size,y.size),dtype=np.complex128))
        for listIter in range(n):
            eps_bank[listIter,:,:] = modeList[listIter].Eps(x,y)
        return np.max(eps_bank,axis=0)

    # Iterate through modes
    for rowIter in range(n):
        for colIter in range(n):

            # Calculate left hand side (S matrix)
            m = modeList[rowIter]
            k = modeList[colIter]
            integrand = lambda y,x: (\
            m.Ex(x,y) * k.Hy(x,y).conj() - \
            m.Ey(x,y) * k.Hx(x,y).conj() + \
            k.Ex(x,y).conj() * m.Hy(x,y) - \
            k.Ey(x,y).conj() * m.Hx(x,y)) \
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
                (eps_full(x,y) - k.Eps(x,y)) * \
                (m.Ex(x,y) * k.Ex(x,y).conj() + \
                 m.Ey(x,y) * k.Ey(x,y).conj() + \
                 m.Ez(x,y) * k.Ez(x,y).conj())

                intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
                C[rowIter,colIter] = intresult
                #C[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)
    print('======')
    print(-1j * 2 * np.pi * omega * wmm.EPS0 / np.sqrt(m.total_power*k.total_power))
    print(S)
    print(C)
    result = np.matmul(linalg.pinv(S), C)
    print(result)
    return result


def makeSupermode(mode1, mode2, x, y):
    #X, Y  = np.meshgrid(x,y)
    numX = x.size; numY = y.size;
    #X = X.flatten(); Y = Y.flatten();
    Eps1 = (mode1.Eps(x,y))
    Eps2 = (mode2.Eps(x,y))

    Eps = Eps1 + Eps2
    return Eps.T
