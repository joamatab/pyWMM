# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #

import numpy as np
from matplotlib import pyplot as plt
from pyWMM import WMM as wmm
from pyWMM import mode
from pyWMM import CMT
from scipy import integrate
# ---------------------------------------------------------------------------- #
# Load in Mode data
# ---------------------------------------------------------------------------- #

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

#lambdaSweep = npzfile['lambdaSweep']

modeNumber = 0
wavelengthNumber = 0
gap = 0.1
omega = wmm.C0 / (lambdaSweep[wavelengthNumber] * 1e-6)
print(omega)

centerLeft = np.array([-gap/2,0,0])
wgLeft = mode.Mode(Eps = Eps, kVec = waveNumbers[wavelengthNumber], center=centerLeft, omega = omega,
                   Ex = Er[wavelengthNumber,modeNumber,:,:],Ey = Ez[wavelengthNumber,modeNumber,:,:],
                   Ez = Ephi[wavelengthNumber,modeNumber,:,:],
                   Hx = Hr[wavelengthNumber,modeNumber,:,:],Hy = Hz[wavelengthNumber,modeNumber,:,:],
                   Hz = Hphi[wavelengthNumber,modeNumber,:,:],
                   x=x,y=y
                   )
centerRight = np.array([gap/2,0,0])
wgRight = mode.Mode(Eps = Eps, kVec = waveNumbers[wavelengthNumber], center=centerRight, omega = omega,
                   Ex = Er[wavelengthNumber,modeNumber,:,:],Ey = Ez[wavelengthNumber,modeNumber,:,:],
                   Ez = Ephi[wavelengthNumber,modeNumber,:,:],
                   Hx = Hr[wavelengthNumber,modeNumber,:,:],Hy = Hz[wavelengthNumber,modeNumber,:,:],
                   Hz = Hphi[wavelengthNumber,modeNumber,:,:],
                   x=x,y=y
                   )

# ---------------------------------------------------------------------------- #
# Define domain and problem
# ---------------------------------------------------------------------------- #
modeList = [wgLeft,wgRight]
zmin = -5; zmax = 5;
xmin = -5; xmax = 5;
ymin = -5;  ymax = 5;
A0 = np.array([0,0])

func = lambda zFunc,A: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc,A)

# ---------------------------------------------------------------------------- #
# Solve
# ---------------------------------------------------------------------------- #
zVec = np.linspace(zmin,zmax,100)
r = integrate.ode(func)
r.set_initial_value(A0,zmin)
r.set_integrator('zvode', method='bdf')
dt = 0.01
while r.successful() and r.t < zmax:
    print(r.t+dt, r.integrate(r.t+dt))
# ---------------------------------------------------------------------------- #
# Plot results
# ---------------------------------------------------------------------------- #
