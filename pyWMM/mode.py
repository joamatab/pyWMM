# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''





'''

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
import pyWMM as wmm
import numpy as np
from scipy import interpolate
from scipy import integrate

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

'''
   general mode
'''
class Mode:

    def __init__(self, Eps, Eps_cladding,
                 Ex = None, Ey, Ez = None, Er = None, Ephi = None,
                 Hx = None, Hy, Hz = None, Hr = None, Hphi = None,
                 x = None, y, r = None,
                 kVec, pol = None, wavelength = None, neff = None
                 ):

        # Ensure the correct coordinate system
        if (Ex == None and Er == None and Ez == None and Ephi == none)
        or (Hx == None and Hr == None and Hz == None and Hphi == none)
        or (x == None and r == None):
            raise ValueError('Missing either an x component or r component in fields!')
        elif Ex == None and Hx == None and x == None:
            self.coordinates = wmm.CYLINDRICAL
        elif Er == None and Hr == None and r == None:
            self.coordinates = wmm.CARTESIAN
        else:
            raise ValueError('Invalid coordinate system defined!')

        # Interpolate all of the fields on the given grid
        self.y = y
        if self.coordinates = wmm.CARTESIAN:
            self.Eps          = interpolate.RectBivariateSpline(x, y, Eps)
            self.Eps_cladding = interpolate.RectBivariateSpline(x, y, Eps)
            self.Ex           = interpolate.RectBivariateSpline(x, y, Ex)
            self.Ey           = interpolate.RectBivariateSpline(x, y, Ey)
            self.Ez           = interpolate.RectBivariateSpline(x, y, Ez)
            self.Hx           = interpolate.RectBivariateSpline(x, y, Hx)
            self.Hy           = interpolate.RectBivariateSpline(x, y, Hy)
            self.Hz           = interpolate.RectBivariateSpline(x, y, Hz)
            self.x            = x
        elif self.coordinates = wmm.CYLINDRICAL:
            self.Eps          = interpolate.RectBivariateSpline(r, y, Eps)
            self.Eps_cladding = interpolate.RectBivariateSpline(r, y, Eps_cladding)
            self.Er           = interpolate.RectBivariateSpline(r, y, Er)
            self.Ey           = interpolate.RectBivariateSpline(r, y, Ey)
            self.Ephi         = interpolate.RectBivariateSpline(r, y, Ephi)
            self.Hr           = interpolate.RectBivariateSpline(r, y, Hr)
            self.Hy           = interpolate.RectBivariateSpline(r, y, Hy)
            self.Hphi         = interpolate.RectBivariateSpline(r, y, Hphi)
            self.r            = r
        else:
            raise ValueError('Invalid coordinate system defined!')

        # Get the power of the modes
        self.TEpower     = self.calc_TE_power()
        self.TMpower     = self.calc_TM_power()
        self.total_power = self.TEpower + self.TMpower

        # Get the normalizing factor
        self.nrm    = 1
        self.ampfac = np.sqrt(self.nrm)/np.sqrt(self.total_power)

        #Initialize all the other variables
    	self.pol  = None
    	self.sym  = None
    	self.beta = None
    	self.k0   = None
    	self.neff = None

	# field values at point (x, y)
	# 	Fcomp: EX - HZ, SZ
	def get_field(self, fComp, x, y, z):

        # Convert the coordinate system if necesarry
        if self.coordinates = wmm.CYLINDRICAL:
            x = np.sqrt(x ** 2 + y ** 2)

        if fComp == wmm.E:
            if self.coordinates = wmm.CYLINDRICAL:
                r = np.sqrt(x ** 2 + z ** 2)
                Er   = self.Er(r,y)
                Ey   = self.Ey(r,y)
                Ephi = self.Ephi(r,y)
                Ex   = Er * np.sin(Ephi)
                Ez   = Er * np.cos(Ephi)
            else:
                Ex = self.Ex(x,y)
                Ey = self.Ey(x,y)
                Ez = self.Ez(x,y)
            E = np.array([Ex,Ey,Ez])
        elif fComp == wmm.H:
            if self.coordinates = wmm.CYLINDRICAL:
                r = np.sqrt(x ** 2 + z ** 2)
                Hr   = self.Hr(r,y)
                Hy   = self.Hy(r,y)
                Hphi = self.Hphi(r,y)
                Hx   = Hr * np.sin(Hphi)
                Hz   = Hr * np.cos(Hphi)
            else:
                Hx = self.Hx(x,y)
                Hy = self.Hy(x,y)
                Hz = self.Hz(x,y)
            H = np.array([Hx,Hy,Hz])
        elif fComp == wmm.eps:
            return self.Ez(x,y)
        elif fComp == wmm.Eps_cladding:
            r = np.sqrt(x**2)
            return self.Er(x,y)
        else
            raise ValueError("Invalide component specified!")

	# longitudinal component of the Poyntingvector,
	# integrated over the entire x-y-domain
	# TE part
	def calc_TE_power(self):
        if self.coordinates = wmm.CARTESIAN:
            f = lambda y, x: self.Ey(x,y) * self.Hx(x,y)
            x0 = self.x[0]; x1 = self.x[-1];
        elif self.coordinates = wmm.CYLINDRICAL:
            f = lambda y, r: self.Ey(x,y) * self.Hr(r,y)
            x0 = self.r[0]; x1 = self.r[-1];
        else:
            raise ValueError('Invalid coordinate system defined!')
        y0 = self.y[0]; y1 = self.y[-1];
        -0.5 * integrate.dblquad(f, y0, y1, lambda x: x0, lambda x: x1)

	# longitudinal component of the Poyntingvector,
	# integrated over the entire x-y-domain
	# TM part
	def calc_TM_power():
        if self.coordinates = wmm.CARTESIAN:
            f = lambda y, x: self.Ex(x,y) * self.Hy(x,y)
            x0 = self.x[0]; x1 = self.x[-1];
        elif self.coordinates = wmm.CYLINDRICAL:
            f = lambda y, r: self.Ex(x,y) * self.Hy(r,y)
            x0 = self.r[0]; x1 = self.r[-1];
        else:
            raise ValueError('Invalid coordinate system defined!')
        y0 = self.y[0]; y1 = self.y[-1];
        0.5 * integrate.dblquad(f, y0, y1, lambda x: x0, lambda x: x1)

class Mode_Array():
    return
