# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

import numpy as np
from matplotlib import pyplot as plt
from pyWMM import WMM as wmm
from pyWMM import mode
from pyWMM import CMT
from scipy import integrate
from scipy import io as sio
# ---------------------------------------------------------------------------- #
# Load in Mode data
# ---------------------------------------------------------------------------- #
format = 'python'
if format == 'matlab':
    matfile = sio.loadmat('modesweep.mat')
    x = matfile['x'] * 1e6
    y = matfile['y'] * 1e6
    numX = x.size
    numY = y.size
    Ex = matfile['mode1_Ex']
    Ey = matfile['mode1_Ey']
    Ez = matfile['mode1_Ez']

    Hx = matfile['mode1_Hx']
    Hy = matfile['mode1_Hy']
    Hz = matfile['mode1_Hz']
    effective_index = matfile['effective_index']

    Eps = matfile['index_x'] ** 2
    wavelength = 1.55
    omega = wmm.C0 / (wavelength * 1e-6)
    kVec = 2*np.pi*effective_index/wavelength
else:
    filename = 'sweepdata.npz'
    npzfile = np.load(filename)
    x = npzfile['x']
    y = npzfile['y']
    Eps = npzfile['Eps']
    Er = npzfile['Er']
    Ez = npzfile['Ez']
    Ephi = npzfile['Ephi']

    Hr = npzfile['Hr']
    Hz = npzfile['Hz']
    Hphi = npzfile['Hphi']
    waveNumbers = npzfile['waveNumbers']
    lambdaSweep = npzfile['lambdaSweep']

    modeNumber = 0
    wavelengthNumber = 0
    omega = wmm.C0 / (lambdaSweep[wavelengthNumber] * 1e-6)
    kVec = waveNumbers[wavelengthNumber]

    Ex = Er[wavelengthNumber,modeNumber,:,:]
    Ey = Ez[wavelengthNumber,modeNumber,:,:]
    Ez = Ephi[wavelengthNumber,modeNumber,:,:]

    Hx = Hr[wavelengthNumber,modeNumber,:,:]
    Hy = Hz[wavelengthNumber,modeNumber,:,:]
    Hz = Hphi[wavelengthNumber,modeNumber,:,:]



gap = 0.2
waveguideWidths = 0.5

centerLeft = np.array([-gap/2 - waveguideWidths/2,0,0])
wgLeft = mode.Mode(Eps = Eps, kVec = kVec, center=centerLeft, omega = omega,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )
centerRight = np.array([gap/2+waveguideWidths/2,0,0])
wgRight = mode.Mode(Eps = Eps, kVec = kVec, center=centerRight, omega = omega,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )

# ---------------------------------------------------------------------------- #
# Define domain and problem
# ---------------------------------------------------------------------------- #
'''
data = CMT.makeSupermode(wgLeft, wgRight, x, y)
plt.imshow(np.real(CMT.makeSupermode(wgLeft, wgRight, x, y)))
plt.show()

quit()
'''
modeList = [wgLeft,wgRight]
zmin = 0; zmax = 15;
xmin = -5; xmax = 5;
ymin = -5;  ymax = 5;
A0 = np.squeeze(np.array([1,0]))

M = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
func = lambda zFunc,A: M.dot(A)

# ---------------------------------------------------------------------------- #
# Solve
# ---------------------------------------------------------------------------- #
zVec = np.linspace(zmin,zmax,100)
r = integrate.complex_ode(func)
r.set_initial_value(A0,zmin)
r.set_integrator('vode',nsteps=500,method='bdf')
dt = 0.01
y = []
z = []
while r.successful() and r.t < zmax:
    r.integrate(r.t+dt)
    z.append(r.t)
    y.append(r.y)
print(np.abs(y) ** 2)
plt.plot(z,np.abs(y) ** 2)
plt.show()
# ---------------------------------------------------------------------------- #
# Plot results
# ---------------------------------------------------------------------------- #
