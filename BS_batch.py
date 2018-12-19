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
straightFilename = '../straight_data/......'
bentFilename     = '../bend10_data/......'
radius           = 10
output_folder    = ''
coreEps          = 9
claddingEps      = 4
gap              = 0.2

nRange  = 1e3
nz = 500
zmin = -5
zmax = 5
xmin = -4
xmax = 1
ymin = -1
ymax = 1


# ---------------------------------------------------------------------------- #
# Read in data
# ---------------------------------------------------------------------------- #

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
width_bent       = 0.5
thickness_bent   = 0.22

# ---------- Straight data --------- #
npzfileStraight      = np.load(straightFilename)
x_straight           = npzfileBent['x']
y_straight           = npzfileBent['y']
Ex_straight          = npzfileBent['Er']
Ey_straight          = npzfileBent['Ez']
Ez_straight          = npzfileBent['Ephi']
Hx_straight          = npzfileBent['Hr']
Hy_straight          = npzfileBent['Hz']
Hz_straight          = npzfileBent['Hphi']
lambdaSweep_straight = npzfileBent['lambdaSweep']
waveNumbers_straight = npzfileBent['waveNumbers']
width_straight       = 0.5
thickness_straight   = 0.22

# ---------------------------------------------------------------------------- #
# Run sweep
# ---------------------------------------------------------------------------- #
numWavelength  = lambdaSweep_straight.size

centerBent     = np.array([-radius  - waveguideWidth - gap,0,0])
centerStraight = np.array([0,0,0])
S = np.zeros((numWavelength,2,2),dtype=np.complex128)
for iter in range(numWavelength):
    wgBent = mode.Mode(beta = waveNumbers_bent[iter], center=centerBent,
                       wavelength = lambdaSweep_bent[iter],
                       waveguideWidth = width_bent, waveguideThickness = thickness_bent,
                       coreEps = coreEps, claddingEps = claddingEps,
                       Er = Er_bent[iter,modeNumber,:,:],Ey = Ey_bent[iter,modeNumber,:,:],
                       Ephi = Ez_bent[iter,modeNumber,:,:],
                       Hr = Hr_bent[wavelengthNumber,modeNumber,:,:],Hy = Hy_bent[wavelengthNumber,modeNumber,:,:],
                       Hphi = Hphi_bent[wavelengthNumber,modeNumber,:,:],
                       r=x_bend,y=y_bend,
                       radius = radius
                       )
    wgStraight = mode.Mode(beta = waveNumbers_straight[iter], center=centerStraight,
                           wavelength = lambdaSweep_straight[iter],
                           waveguideWidth = width_straight, waveguideThickness = thickness_straight,
                           coreEps = coreEps, claddingEps = claddingEps,
                           Ex = Ex_straight[iter,modeNumber,:,:],Ey = Ey_straight[iter,modeNumber,:,:],
                           Ez = Ez_straight[iter,modeNumber,:,:],
                           Hx = Hx_straight[iter,modeNumber,:,:],Hy = Hy_straight[iter,modeNumber,:,:],
                           Hz = Hz_straight[iter,modeNumber,:,:],
                           x=x_straight,y=y_straight
                       )
    modeList = [wgBent,wgStraight]

    A0 = np.squeeze(np.array([1,0]))
    M = CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax)
    func = lambda zFunc: CMT.CMTsetup(modeList,xmin,xmax,ymin,ymax,zFunc)

    y, F_iter, S_iter = wmm.TMM(func,A0,zmin,zmax,nz)
    S[iter,:,: = ]S_iter[-1,:,:]


# ---------------------------------------------------------------------------- #
# Save data
# ---------------------------------------------------------------------------- #
np.savez(output_folder + '/CMT.npz',
  S=S,gap=gap,wavelength=lambdaSweep_straight)
