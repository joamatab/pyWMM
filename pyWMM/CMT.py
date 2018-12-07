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
def CMTsetup(modeList,xmin,xmax,ymin,ymax,z,A):
    n = len(modeList)

    S = np.zeros((n,n),dtype=np.complex128)
    C = np.zeros((n,n),dtype=np.complex128)

    ez = np.array([0, 0, 1])

    # TODO: Validate input

    omega = modeList[0].omega

    # Iterate through modes
    for rowIter in range(n):
        for colIter in range(n):

            # Calculate left hand side (S matrix)
            integrand = lambda y,x : np.dot(ez,
            np.cross(modeList[colIter].get_field(wmm.E,x,y,z),np.conjugate(modeList[rowIter].get_field(wmm.H,x,y,z))) +
            np.cross(np.conjugate(modeList[rowIter].get_field(wmm.E,x,y,z)),(modeList[colIter].get_field(wmm.H,x,y,z)))
            )
            intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
            S[rowIter,colIter] = intresult[0]
            #S[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)

            # Calculate right hand side (C matrix)
            if rowIter == colIter:
                C[rowIter,colIter] = 0
            else:
                integrand = lambda y,x: -1j*omega*wmm.EPS0*(modeList[rowIter].get_field(wmm.Eps,x,y,z) - modeList[colIter].get_field(wmm.Eps,x,y,z)) * np.dot(modeList[colIter].get_field(wmm.E,x,y,z),np.conjugate(modeList[rowIter].get_field(wmm.E,x,y,z)))
                intresult = wmm.complex_quadrature(integrand, xmin, xmax, ymin, ymax)
                print('---------------')
                C[rowIter,colIter] = intresult[0]
                #C[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)

    result = np.matmul(linalg.pinv(S), C)
    return (result).dot(A)
