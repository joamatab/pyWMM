# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
'''





'''

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
from pyWMM import WMM as wmm
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

    def __init__(self, Eps, kVec, center, omega,
                 Ex = None, Ey = None, Ez = None, Er = None, Ephi = None,
                 Hx = None, Hy = None, Hz = None, Hr = None, Hphi = None,
                 x = None, y = None, r = None, radius = None,
                 pol = None, wavelength = None, neff = None
                 ):

        # Ensure the correct coordinate system
        if (Ex is None and Er is None and Ez is None and Ephi is None) or (Hx is None and Hr is None and Hz is None and Hphi is None) or (x is None and r is None):
            raise ValueError('Missing either an x component or r component in fields!')
        elif Ex is None and Hx is None and x is None:
            self.coordinates = wmm.CYLINDRICAL
        elif Er is None and Hr is None and r is None:
            self.coordinates = wmm.CARTESIAN
        else:
            raise ValueError('Invalid coordinate system defined!')

        if self.coordinates == wmm.CYLINDRICAL and radius == None:
            raise ValueError('Failed to specify the waveguide\'s center radius!')

        # Interpolate all of the fields on the given grid
        self.y = y
        if self.coordinates == wmm.CARTESIAN:
            self.Eps_r          = interpolate.RectBivariateSpline(x, y, np.real(Eps))
            self.Eps_i          = interpolate.RectBivariateSpline(x, y, np.imag(Eps))
            self.Eps = lambda x,y: self.Eps_r(x,y,grid=False) + 1j*self.Eps_i(x,y,grid=True)

            self.Ex_r           = interpolate.RectBivariateSpline(x, y, np.real(Ex))
            self.Ex_i           = interpolate.RectBivariateSpline(x, y, np.imag(Ex))
            self.Ex = lambda x,y: self.Ex_r(x,y,grid=False) + 1j*self.Ex_i(x,y,grid=True)

            self.Ey_r           = interpolate.RectBivariateSpline(x, y, np.real(Ey))
            self.Ey_i           = interpolate.RectBivariateSpline(x, y, np.imag(Ey))
            self.Ey = lambda x,y: self.Ey_r(x,y,grid=False) + 1j*self.Ey_i(x,y,grid=True)

            self.Ez_r           = interpolate.RectBivariateSpline(x, y, np.real(Ez))
            self.Ez_i           = interpolate.RectBivariateSpline(x, y, np.imag(Ez))
            self.Ez = lambda x,y: self.Ez_r(x,y,grid=False) + 1j*self.Ez_i(x,y,grid=True)

            self.Hx_r           = interpolate.RectBivariateSpline(x, y, np.real(Hx))
            self.Hx_i           = interpolate.RectBivariateSpline(x, y, np.imag(Hx))
            self.Hx = lambda x,y: self.Hx_r(x,y,grid=False) + 1j*self.Hx_i(x,y,grid=True)

            self.Hy_r           = interpolate.RectBivariateSpline(x, y, np.real(Hy))
            self.Hy_i           = interpolate.RectBivariateSpline(x, y, np.imag(Hy))
            self.Hy = lambda x,y: self.Hy_r(x,y,grid=False) + 1j*self.Hy_i(x,y,grid=True)

            self.Hz_r           = interpolate.RectBivariateSpline(x, y, np.real(Hz))
            self.Hz_i           = interpolate.RectBivariateSpline(x, y, np.imag(Hz))
            self.Hz = lambda x,y: self.Hz_r(x,y,grid=False) + 1j*self.Hz_i(x,y,grid=True)

            self.x            = x
        elif self.coordinates == wmm.CYLINDRICAL:
            self.Eps          = interpolate.RectBivariateSpline(r, y, Eps)
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
        print('calculating TE power')
        #self.calc_TE_power()
        print('calculating TM power')
        #self.calc_TM_power()
        print('calculating total power')
        #self.calc_total_power()

        # Get the normalizing factor
        #self.nrm    = 1
        #self.ampfac = np.sqrt(self.nrm)/np.sqrt(self.total_power)
        self.ampfac = 1
        #Initialize all the other variables
        self.omega = omega
        self.pol  = None
        self.sym  = None
        self.beta = None
        self.k0   = None
        self.neff = None
        self.center = center

	# field values at point (x, y)
	# 	Fcomp: E, H, Eps
    # returns a #D vector
    def get_field(self, fComp, x, y, z):

        #Extract center
        centerX = self.center[0]
        centerY = self.center[1]
        centerZ = self.center[2]

        if fComp == wmm.E:
            if self.coordinates == wmm.CYLINDRICAL:
                r = np.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2)
                Er   = self.Er(r,(y-centerY))
                Ey   = self.Ey(r,(y-centerY))
                Ephi = self.Ephi(r,(y-centerY))
                Ex   = Er * np.sin(Ephi)
                Ez   = Er * np.cos(Ephi)
            else:
                Ex = self.Ex((x-centerX),(y-centerY))
                Ey = self.Ey((x-centerX),(y-centerY))
                Ez = self.Ez((x-centerX),(y-centerY))
            E = np.array([Ex,Ey,Ez]) * self.ampfac
            return E
        elif fComp == wmm.H:
            if self.coordinates == wmm.CYLINDRICAL:
                r = np.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2)
                Hr   = self.Hr(r,(y-centerY))
                Hy   = self.Hy(r,(y-centerY))
                Hphi = self.Hphi(r,(y-centerY))
                Hx   = Hr * np.sin(Hphi)
                Hz   = Hr * np.cos(Hphi)
            else:
                Hx = self.Hx((x-centerX),(y-centerY))
                Hy = self.Hy((x-centerX),(y-centerY))
                Hz = self.Hz((x-centerX),(y-centerY))
            H = np.array([Hx,Hy,Hz]) * self.ampfac
            return H
        elif fComp == wmm.Eps:
            if self.coordinates == wmm.CYLINDRICAL:
                r = np.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2)
                Eps = self.Eps(r,(y-centerY))
            else:
                Eps = self.Eps((x-centerX),(y-centerY))
            return Eps
        else:
            raise ValueError("Invalid component specified!")

	# longitudinal component of the Poyntingvector,
	# integrated over the entire x-y-domain
	# TE part
    def calc_TE_power(self):
        if self.coordinates == wmm.CARTESIAN:
            f = lambda y, x: self.Ey(x,y) * self.Hx(x,y)
            xmin = self.x[0]; xmax = self.x[-1];
        elif self.coordinates == wmm.CYLINDRICAL:
            f = lambda y, r: self.Ey(x,y) * self.Hr(r,y)
            xmin = self.r[0]; xmax = self.r[-1];
        else:
            raise ValueError('Invalid coordinate system defined!')
        ymin = self.y[0]; ymax = self.y[-1];

        intResults = wmm.complex_quadrature(f, xmin, xmax, ymin, ymax)
        self.TE_power = -0.5 * intResults[0]
        print(self.TE_power)

	# longitudinal component of the Poyntingvector,
	# integrated over the entire x-y-domain
	# TM part
    def calc_TM_power(self):
        if self.coordinates == wmm.CARTESIAN:
            f = lambda y, x: self.Ex(x,y) * self.Hy(x,y)
            xmin = np.min(self.x); xmax = np.min(self.x[-1]);
        elif self.coordinates == wmm.CYLINDRICAL:
            f = lambda y, r: self.Ex(x,y) * self.Hy(r,y)
            xmin = self.r[0]; xmax = self.r[-1];
        else:
            raise ValueError('Invalid coordinate system defined!')
        ymin = np.min(self.y); ymax = np.min(self.y[-1]);

        intResults = wmm.complex_quadrature(f, xmin, xmax, ymin, ymax)
        self.TM_power = 0.5 * intResults[0]
        print(self.TM_power)

    def calc_total_power(self):
        self.total_power = self.TM_power + self.TE_power
        print(self.total_power)
