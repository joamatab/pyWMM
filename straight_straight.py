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
# Analytic solution
# ---------------------------------------------------------------------------- #

supermode_data = sio.loadmat('mode1_super.mat')
singlemode_data = sio.loadmat('modesweep.mat')

mode1_effective_index = supermode_data['effective_index'][0,0]
mode2_effective_index = supermode_data['effective_index'][1,0]
singlemode_effective_index = singlemode_data['effective_index'][0,0]

wavelength = 1.55

mode1_beta = 2 * np.pi * mode1_effective_index/ wavelength
mode2_beta = 2 * np.pi * mode2_effective_index/ wavelength
singlemode_beta = 2 * np.pi * singlemode_effective_index / wavelength

dBeta = np.abs(np.pi * (mode1_effective_index - mode2_effective_index) / wavelength)

L = np.linspace(0,15,200)
coupler1_power = np.abs((np.cos(dBeta*L)) ) ** 2
coupler2_power = np.abs((np.sin(dBeta*L)) ) ** 2
# ---------------------------------------------------------------------------- #
# Load in Mode data
# ---------------------------------------------------------------------------- #
format = 'matlab'
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
    effective_index = np.squeeze(matfile['effective_index'])

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
    wavelength = lambdaSweep[wavelengthNumber]
    omega = wmm.C0 / (lambdaSweep[wavelengthNumber] * 1e-6)
    kVec = np.squeeze(waveNumbers[wavelengthNumber])

    Ex = Er[wavelengthNumber,modeNumber,:,:]
    Ey = Ez[wavelengthNumber,modeNumber,:,:]
    Ez = Ephi[wavelengthNumber,modeNumber,:,:]

    Hx = Hr[wavelengthNumber,modeNumber,:,:]
    Hy = Hz[wavelengthNumber,modeNumber,:,:]
    Hz = Hphi[wavelengthNumber,modeNumber,:,:]



gap = 0.2
waveguideWidths = 0.5

centerLeft = np.array([-gap/2 - waveguideWidths/2,0,0])
wgLeft = mode.Mode(Eps = Eps, beta = kVec, center=centerLeft, wavelength = wavelength,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )
centerRight = np.array([gap/2+waveguideWidths/2,0,0])
wgRight = mode.Mode(Eps = Eps, beta = kVec, center=centerRight, wavelength = wavelength,
                   Ex = Ex,Ey = Ey,
                   Ez = Ez,
                   Hx = Hx,Hy = Hy,
                   Hz = Hz,
                   x=x,y=y
                   )

# ---------------------------------------------------------------------------- #
# Define domain and problem
# ---------------------------------------------------------------------------- #
nRange  = 1e3
modeList = [wgLeft,wgRight]
zmin = 0; zmax = 15;
xmin = -5; xmax = 5;
ymin = -5;  ymax = 5;
nz = 100
xRange = np.linspace(xmin,xmax,nRange)
yRange = np.linspace(ymin,ymax,nRange)
zRange = np.linspace(zmin,zmax,nz)
betaq = kVec
A0 = np.squeeze(np.array([np.exp(-1j*betaq*(zmax-zmin)),0]))


M = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
func = lambda zFunc: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc)

y, F_bank = wmm.TMM(func,A0,zmin,zmax,nz)
# ---------------------------------------------------------------------------- #
# Solve
# ---------------------------------------------------------------------------- #

'''
M = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
func = lambda zFunc,A: M.dot(A)
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

y = np.array(y)
'''
# ---------------------------------------------------------------------------- #
# Plot results
# ---------------------------------------------------------------------------- #
plt.figure()
plt.subplot(1,2,1)
crossSection = CMT.getCrossSection(modeList,xRange,yRange)
plt.imshow(np.real(crossSection),cmap='Greys',extent = (xmin,xmax,ymin,ymax))
plt.title('Cross Section')
plt.xlabel('X (microns)')
plt.ylabel('Y (microns)')

plt.subplot(1,2,2)
topView =  CMT.getTopView(modeList,xRange,zRange)
plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax))
plt.title('Top View')
plt.xlabel('X (microns)')
plt.ylabel('Z (microns)')

plt.tight_layout()
plt.savefig('straight_straight_geo.png')

plt.figure()
plt.subplot(2,1,1)
plt.plot(L,coupler1_power,linewidth=2,color='blue',label='Analytic')
plt.plot(zRange,np.abs(y[:,0]) ** 2,'--',color='red',linewidth=2,label='CMT')
plt.title('Waveguide 1')
plt.xlabel('Z position (microns)')
plt.ylabel('Relative Power')
plt.legend()
plt.ylim(-1,2)
plt.grid(True)


plt.subplot(2,1,2)
plt.plot(L,coupler2_power,linewidth=2,color='blue',label='Analytic')
plt.plot(zRange,np.abs(y[:,1]) ** 2,'--',color='red',linewidth=2,label='CMT')
plt.title('Waveguide 2')
plt.xlabel('Z position (microns)')
plt.ylabel('Relative Power')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('straight_straight_results.png')
plt.show()
