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
    numModes = len(modeList)

    S = np.zeros((n,n),dtype=np.complex128)
    C = np.zeros((n,n),dtype=np.complex128)

    ez = np.array([0, 0, 1])

    # TODO: Validate input

    omega = modeList[0].omega

    # Iterate through modes
    for rowIter in range(n):
        for coliter in range(n):

            # Calculate left hand side (S matrix)
            integrand = lambda y,x : np.dot(ez,
            np.cross(modeList[colIter].getField(wmm.E,x,y,z),np.conjugate(modeList[rowIter].getField(wmm.H),x,y,z)) +
            np.cross(np.conjugate(modeList[rowIter].getField(wmm.E,x,y,z)),(modeList[colIter].getField(wmm.H),x,y,z))
            )
            S[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)

            # Calculate right hand side (C matrix)
            if row Iter == colIter:
                C[rowIter,colIter] = 0
            else:
                integrand = lambda y,x: -1i*omega*wmm.EPS0*(modeList[rowMode].getField(wmm.Eps,x,y,z) - modeList[colMode].getField(wmm.Eps,x,y,z)) *
                np.dot(modeList[colMode].getField(wmm.E,x,y,z),np.conjugate(modeList[colRow].getField(wmm.E,x,y,z)))
                C[rowIter,colIter] = integrate.dblquad(integrand,xmin,xmax,lambda x: ymin, lambda x: ymax)


    return (np.matmul(linalg.pinv(S), O)).dot(A)
