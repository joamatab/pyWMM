import numpy as np
from matplotlib import pyplot as plt
from pyWMM import WMM as wmm
from pyWMM import mode
from pyWMM import CMT
from scipy import integrate
from scipy import io as sio

# ---------------------------------------------------------------------------- #
# Parse input parameters
# ---------------------------------------------------------------------------- #

modeNumber       = 0
straightFilename = '../test_batch/test_batchsweepData.npz'
bentFilename     = '../test_batch/test_batchsweepData.npz'
radius           = 10
output_folder    = ''
coreEps          = 9
claddingEps      = 4
gap              = 0.2

width_bent       = 0.5
thickness_bent   = 0.22

nRange  = int(1e3)
nz = int(500)
zmin = -5
zmax = 5
xmin = -3
xmax = 1
ymin = -1
ymax = 1

xRange = np.linspace(xmin,xmax,nRange)
yRange = np.linspace(ymin,ymax,nRange)
zRange = np.linspace(zmin,zmax,nRange)

# ---------------------------------------------------------------------------- #
# Read in data
# ---------------------------------------------------------------------------- #
print('Loading data..')
# -------- Bent data --------------- #
npzfileBent      = np.load(bentFilename)
x_bent           = npzfileBent['x']
y_bent           = npzfileBent['y']
Er_bent          = npzfileBent['Er']
Ey_bent          = npzfileBent['Ez']
Ephi_bent        = npzfileBent['Ephi']
Hr_bent          = npzfileBent['Hr']
Hy_bent          = npzfileBent['Hz']
Hphi_bent        = npzfileBent['Hphi']
lambdaSweep_bent = npzfileBent['lambdaSweep']
waveNumbers_bent = npzfileBent['waveNumbers']

# ---------- Straight data --------- #
npzfileStraight      = np.load(straightFilename)
x_straight           = npzfileStraight['x']
y_straight           = npzfileStraight['y']
Ex_straight          = npzfileStraight['Er']
Ey_straight          = npzfileStraight['Ez']
Ez_straight          = npzfileStraight['Ephi']
Hx_straight          = npzfileStraight['Hr']
Hy_straight          = npzfileStraight['Hz']
Hz_straight          = npzfileStraight['Hphi']
Eps_straight          = npzfileStraight['Eps']
lambdaSweep_straight = npzfileStraight['lambdaSweep']
waveNumbers_straight = npzfileStraight['waveNumbers']
width_straight       = 0.5
thickness_straight   = 0.22

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #
numWavelength  = lambdaSweep_straight.size

centerBent     = np.array([-radius  - width_straight/2 - width_bent/2 - gap,0,0])
centerStraight = np.array([0,0,0])
S = np.zeros((numWavelength,2,2),dtype=np.complex128)
for iter in range(numWavelength):
    print('Wavelength {:d} of {:d}: {:e} microns.'.format(iter+1,numWavelength,lambdaSweep_bent[iter]))
    wgBent = mode.Mode(beta = waveNumbers_bent[iter,modeNumber],
                       center             = centerBent,
                       wavelength         = lambdaSweep_bent[iter],
                       waveguideWidth     = width_bent,
                       waveguideThickness = thickness_bent,
                       coreEps            = coreEps,
                       claddingEps        = claddingEps,
                       Er                 = Er_bent[iter,modeNumber,:,:],
                       Ey                 = Ey_bent[iter,modeNumber,:,:],
                       Ephi               = Ephi_bent[iter,modeNumber,:,:],
                       Hr                 = Hr_bent[iter,modeNumber,:,:],
                       Hy                 = Hy_bent[iter,modeNumber,:,:],
                       Hphi               = Hphi_bent[iter,modeNumber,:,:],
                       r                  = x_bent,
                       y                  = y_bent,
                       radius             = radius
                       )

    wgStraight = mode.Mode(beta               = waveNumbers_straight[iter,modeNumber],
                           center             = centerStraight,
                           wavelength         = lambdaSweep_straight[iter],
                           waveguideWidth     = width_straight,
                           waveguideThickness = thickness_straight,
                           coreEps            = coreEps,
                           claddingEps        = claddingEps,
                           Ex                 = Ex_straight[iter,modeNumber,:,:],
                           Ey                 = Ey_straight[iter,modeNumber,:,:],
                           Ez                 = Ez_straight[iter,modeNumber,:,:],
                           Hx                 = Hx_straight[iter,modeNumber,:,:],
                           Hy                 = Hy_straight[iter,modeNumber,:,:],
                           Hz                 = Hz_straight[iter,modeNumber,:,:],
                           x                  = x_straight,
                           y                  = y_straight
                       )
    modeList = [wgBent,wgStraight]
    # ------------------------------------------------------------------------ #
    # Visualize
    # ------------------------------------------------------------------------ #
    plt.figure()
    xTemp = np.linspace(-1,1,1000)
    yTemp = np.linspace(-1,1,1000)
    Eps_straight
    plt.imshow(np.real(np.rot90(wgStraight.Eps(xTemp,y=yTemp,z=0))),extent = (-1,1,-1,1),origin='lower' )
    #plt.imshow(np.abs(np.rot90(Eps_straight)),extent = (x_straight[0],x_straight[-1],y_straight[0],y_straight[-1]),origin='lower')
    #plt.imshow(np.abs(np.rot90(Ex_straight[iter,modeNumber,:,:])) ** 2,extent = (x_straight[0],x_straight[-1],y_straight[0],y_straight[-1]),origin='lower',vmax=10  )
    #plt.imshow(np.abs(np.rot90(wgStraight.Ez(xTemp,y=yTemp,z=0))) ** 2,extent = (-1,1,-1,1),origin='lower' )
    plt.colorbar()
    plt.savefig('resolution.png')

    quit()
    plt.figure()

    plt.subplot(2,2,1)

    plt.imshow(abs(np.rot90(wgStraight.Ex(xRange,y=0,z=zRange))) ** 2,extent = (xmin,xmax,zmin,zmax),origin='lower')

    plt.subplot(2,2,3)
    plt.imshow(abs(np.rot90(wgBent.Ex(xRange,y=0,z=zRange))) ** 2,extent = (xmin,xmax,zmin,zmax),origin='lower')

    plt.subplot(2,2,4)
    plt.imshow(abs((Ephi_bent[iter,modeNumber,:,:].T)) ** 2)

    plt.savefig('plot_2.png')

    plt.figure()
    plt.subplot(1,4,1)
    topView =  CMT.getTopView(modeList,xRange,zRange)
    topView_ex =  CMT.getTopView_Ex(modeList,xRange,zRange)
    plt.imshow((np.real(topView)),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.imshow(10*np.log10(np.abs(topView_ex)),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.title('Top View')
    plt.xlabel('X (microns)')
    plt.ylabel('Z (microns)')
    plt.title('Ex')

    plt.subplot(1,4,2)
    topView =  CMT.getTopView(modeList,xRange,zRange)
    topView_ey =  CMT.getTopView_Ey(modeList,xRange,zRange)
    plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.imshow(10*np.log10(np.abs(topView_ey)),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.title('Top View')
    plt.xlabel('X (microns)')
    plt.ylabel('Z (microns)')
    plt.title('Ey')

    plt.subplot(1,4,3)
    topView =  CMT.getTopView(modeList,xRange,zRange)
    topView_ez =  CMT.getTopView_Ez(modeList,xRange,zRange)
    plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.imshow(10*np.log10(np.abs(topView_ez)),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.title('Top View')
    plt.xlabel('X (microns)')
    plt.ylabel('Z (microns)')
    plt.title('Ez')

    plt.subplot(1,4,4)
    topView =  CMT.getTopView(modeList,xRange,zRange)
    topView_tot =  np.sqrt(np.abs(topView_ex) ** 2 + np.abs(topView_ey) ** 2 + np.abs(topView_ez) ** 2)
    plt.imshow(np.real(topView),cmap='Greys',extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.imshow(10*np.log10(np.abs(topView_tot)),alpha=0.5,extent = (xmin,xmax,zmin,zmax),origin='lower')
    plt.title('Top View')
    plt.xlabel('X (microns)')
    plt.ylabel('Z (microns)')
    plt.title('|E|$^2$')


    plt.tight_layout()
    plt.savefig('threeD_view.png')
    quit()

    # ------------------------------------------------------------------------ #
    # Run solver
    # ------------------------------------------------------------------------ #

    A0   = np.squeeze(np.array([1,0]))
    M    = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
    func = lambda zFunc: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc)

    y, F_iter, S_iter = wmm.TMM(func,A0,zmin,zmax,nz)
    S[iter,:,:] = S_iter[-1,:,:]


# ---------------------------------------------------------------------------- #
# Save data
# ---------------------------------------------------------------------------- #
np.savez('CMT.npz',
  S=S,gap=gap,wavelength=lambdaSweep_straight)
