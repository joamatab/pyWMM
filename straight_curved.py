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
wavelength = lambdaSweep[wavelengthNumber]
omega = wmm.C0 / (lambdaSweep[wavelengthNumber] * 1e-6)
beta = waveNumbers[wavelengthNumber]

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
centerLeft = np.array([-radius  - waveguideWidth - gap,0,0])
wgLeft = mode.Mode(Eps = Eps, beta = beta, center=centerLeft, wavelength = wavelength,
                   Er = Ex,Ey = Ey,
                   Ephi = Ez,
                   Hr = Hx,Hy = Hy,
                   Hphi = Hz,
                   r=x,y=y,
                   radius = radius
                   )
centerRight = np.array([0,0,0])
wgRight = mode.Mode(Eps = Eps, beta = beta, center=centerRight, wavelength = wavelength,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )

nRange  = 1e3
modeList = [wgLeft,wgRight]
zmin = -6; zmax = 6;
xmin = -3; xmax = 1;
ymin = -1;  ymax = 1;
nz = 500
xRange = np.linspace(xmin,xmax,nRange)
yRange = np.linspace(ymin,ymax,nRange)
zRange = np.linspace(zmin,zmax,nRange)


A0 = np.squeeze(np.array([1,0]))

M = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
func = lambda zFunc: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc)

y, F_bank = wmm.TMM(func,A0,zmin,zmax,beta,nz)
z = np.linspace(zmin,zmax,nz)

'''
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
'''
y = np.array(y)

plt.figure()
plt.plot(z,np.abs(y) ** 2)
#plt.show()


plt.figure()
plt.subplot(1,4,1)
topView =  CMT.getTopView(modeList,xRange,zRange)
topView_ex =  CMT.getTopView_Ex(modeList,xRange,zRange)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.imshow(np.abs(topView_ex),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')
plt.title('Ex')
'''
crossSection = CMT.getCrossSection(modeList,xRange,yRange,z=2*radius)
plt.imshow(np.real(crossSection),cmap='Greys',extent = (xmin,xmax,ymin,ymax),origin='lower')
plt.title('Cross Section')
plt.xlabel('X (microns)')
plt.ylabel('Y (microns)')
'''

plt.subplot(1,4,2)
topView =  CMT.getTopView(modeList,xRange,zRange)
topView_ey =  CMT.getTopView_Ey(modeList,xRange,zRange)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.imshow(np.real(topView_ey),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')
plt.title('Ey')

plt.subplot(1,4,3)
topView =  CMT.getTopView(modeList,xRange,zRange)
topView_ez =  CMT.getTopView_Ez(modeList,xRange,zRange)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.imshow(np.real(topView_ez),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')
plt.title('Ez')

plt.subplot(1,4,4)
topView =  CMT.getTopView(modeList,xRange,zRange)
topView_tot =  np.sqrt(np.abs(topView_ex) ** 2 + np.abs(topView_ey) ** 2 + np.abs(topView_ez) ** 2)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.imshow(np.real(topView_tot),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')
plt.title('Ez')


plt.tight_layout()
plt.savefig('threeD_view.png')
plt.show()
quit()

zSweep = [radius, 1.25*radius,1.5*radius,2*radius, 2.5*radius, 2.75 *radius, 3*radius]
plt.figure()
for k in range(len(zSweep)):
    plt.subplot(len(zSweep),2,2*k+1)
    Ex = CMT.getCrossSection_Ex(modeList,xRange,yRange,zSweep[k])
    plt.imshow(np.real(Ex),extent = (xmin,xmax,ymin,ymax),origin='lower')

    plt.subplot(len(zSweep),2,2*k+2)
    Ey = CMT.getCrossSection_Ey(modeList,xRange,yRange,zSweep[k])
    plt.imshow(np.real(Ey),extent = (xmin,xmax,ymin,ymax),origin='lower')


plt.tight_layout()
plt.show()
