import numpy as np
from matplotlib import pyplot as plt
from pyWMM import WMM as wmm
from pyWMM import mode
from pyWMM import CMT
from scipy import integrate
from scipy import io as sio

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
radius = 5
gap = 0.2
waveguideWidth = 0.5
centerLeft = np.array([-radius  - waveguideWidth - gap,0,2*radius])
wgLeft = mode.Mode(Eps = Eps, kVec = kVec, center=centerLeft, omega = omega,
                   Er = Ex,Ey = Ey,
                   Ephi = Ez,
                   Hr = Hx,Hy = Hy,
                   Hphi = Hz,
                   r=x,y=y,
                   radius = radius
                   )
centerRight = np.array([0,0,0])
wgRight = mode.Mode(Eps = Eps, kVec = kVec, center=centerRight, omega = omega,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )

nRange  = 1e3
modeList = [wgLeft,wgRight]
zmin = 0; zmax = 4*radius;
xmin = -4; xmax = 4;
ymin = -5;  ymax = 5;
xRange = np.linspace(xmin,xmax,nRange)
yRange = np.linspace(ymin,ymax,nRange)
zRange = np.linspace(zmin,zmax,nRange)

A0 = np.squeeze(np.array([1,0]))

func = lambda zFunc,A: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc).dot(A)

zVec = np.linspace(zmin,zmax,100)
r = integrate.complex_ode(func)
r.set_initial_value(A0,zmin)
r.set_integrator('vode',nsteps=500,method='bdf')
dt = 0.1
y = []
z = []
while r.successful() and r.t < zmax:
    r.integrate(r.t+dt)
    z.append(r.t)
    y.append(r.y)

y = np.array(y)

plt.figure()
plt.plot(z,y)

plt.figure()
plt.subplot(1,2,1)
crossSection = CMT.getCrossSection(modeList,xRange,yRange,z=2*radius)
plt.imshow(np.real(crossSection),cmap='Greys',extent = (xmin,xmax,ymin,ymax),origin='lower')
plt.title('Cross Section')
plt.xlabel('X (microns)')
plt.ylabel('Y (microns)')

plt.subplot(1,2,2)
topView =  CMT.getTopView(modeList,xRange,zRange)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')

plt.tight_layout()
plt.show()
