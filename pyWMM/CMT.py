# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
import numpy as np
from scipy import integrate
from pyWMM import WMM as wmm
from pyWMM import mode
from scipy import linalg
from matplotlib import pyplot as plt

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
    P = np.zeros((n,n),dtype=np.complex128)
    Q = np.zeros((n,n),dtype=np.complex128)

    mask = np.tril(np.ones((n,n)))  # upper diagonal is zeros

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

            if rowIter == colIter:
                #S[rowIter,colIter] = 0.5 * (m.total_power + k.total_power)
            #    / (np.sqrt((m.total_power*k.total_power)))
                Q[rowIter,colIter] = m.getPhasor(z)
            #else:
            integrand = lambda y,x: (\
            m.Ex(x,y,z).conj() * k.Hy(x,y,z) - \
            m.Ey(x,y,z).conj() * k.Hx(x,y,z) + \
            k.Ex(x,y,z) * m.Hy(x,y,z).conj()        - \
            k.Ey(x,y,z) * m.Hx(x,y,z)).conj()         \

            intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
            S[rowIter,colIter] = 0.25*intresult
            '''
            xT = np.linspace(xmin,xmax,100)
            yT = np.linspace(ymin,ymax,100)
            X, Y = np.meshgrid(xT,yT,sparse=True, indexing='ij')
            zT = np.zeros((100,100),dtype=np.complex128)
            zT = integrand(Y,X)
            print('********')
            print(rowIter)
            print(colIter)
            plt.figure()
            plt.imshow(np.rot90(np.real(zT)),extent = (xmin,xmax,ymin,ymax),origin='lower')
            plt.show()
            print('********')
            '''
                #S[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)

            # Calculate right hand side (C matrix)
            integrand = lambda y,x: \
            -1j * 0.25 * omega * wmm.EPS0 *\
            (eps_full(x,y) - k.Eps(x,y,z)) * \
            (m.Ex(x,y,z).conj() * k.Ex(x,y,z) + \
             m.Ey(x,y,z).conj() * k.Ey(x,y,z) + \
             m.Ez(x,y,z).conj() * k.Ez(x,y,z))
            intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
            C[rowIter,colIter] = intresult
                #C[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)
    # Mask the P & Q matrices
    Msb = S[1,0]
    Mss = S[1,1]
    Q[1,0] =  Msb / Mss * Q[1,1]
    result = np.matmul(linalg.pinv(S), C)
    return result, Q

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
