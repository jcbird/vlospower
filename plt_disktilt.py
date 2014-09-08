"""
plt_disktilt.py
============

Plotting results of disk tilt check
Loooking for symmetry. 

"""

from __future__ import print_function
import numpy as np
import pandas as pd
import scipy
import pynbody
from scipy import interpolate as interpolate
import matplotlib as mpl
from matplotlib import pyplot as plt
import decomp
import astroML.resample
import seaborn as sns


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(


def gcCoords(obsDataDict, sunpos):
    """
    Function to calculate rectagular galacto-centric coordinates given l, b,
    distance2d, distance3d.

    obsDataDict is a dictionary object with the following keys:

    'l', 'b', 'd2', 'd3'

    Assume you want Sun at (xgc=8,ygc=0, zgc=0)

    sunpos  a tuple or array (3d)
    """
    try:

        xGC = obsDataDict['d2'] * np.cos(np.radians(obsDataDict['l']))
        yGC = obsDataDict['d2'] * -1.0*np.sin(np.radians(obsDataDict['l']))
        zGC = obsDataDict['d3'] * np.sin(np.radians(obsDataDict['b']))

        return (np.vstack([8.0 - xGC, yGC, zGC])).transpose()

    except KeyError:
        print("Data missing from Dictionary.\nMake sure object has l,b,d2,d3 with correct naming.")

originalsim = st.load_faceon(400, config.params)
decomp.decomp(originalsim, config)  # Disk and Thick disk are 1 and 2
#Making the decomposition here assuming that small rotations won't affect the distribution dramatically
originalsubSim = originalsim[pynbody.filt.Disc('20 kpc', '0.5 kpc')]

xangles = np.linspace(-1.8,0.6,num=6)
yangles = np.linspace(-1.8,0.6, num=6)

combos = np.array([thetas for thetas in itertools.product(xangles,yangles)])
##combos = np.array([(2.0,2.0), (-2.0,-2.0)])

extentx = [-15.0,15.0]
extenty = [-15.0,15.0]
#extentx = [5.25,12.25]
#extenty = [-3.0,4.0]
nbinsx,nbinsy = 60,60
#nbinsx,nbinsy = 14,14
binsx = np.arange(extentx[0], extentx[1]+0.001, (extentx[1]-extentx[0])/nbinsx)
binsy = np.arange(extenty[0], extenty[1]+0.001, (extenty[1]-extenty[0])/nbinsy)

med_dVr = []
med_dVt = []
med_dVrrot = []
med_dVtrot = []
meddiff_rotdVr = []
meddiff_rotdVt = []

for j, angles in enumerate(combos):
    print(angles)
    sim = originalsubSim.__deepcopy__()
#2D velocities
    #sim['vcxy']
    #sim['vrxy']

    sim.rotate_x(angles[0])
    sim.rotate_y(angles[1])

    #Zcut in GC coords, also needed for original sim for GC vels
    ###dfzcut = dfObsDataCoords[(np.abs(dfObsDataCoords.zgc) < 0.25) & (sim.s['decomp']<1.5)]
    subSim = sim[np.asarray(np.abs(sim.s['z'])< 0.25) & (sim.s['decomp']<1.5)]
    sim_class = stp.SimPynbody(subSim)
    dfzcut = stp.sim2df(sim_class)

    #Need to decide if Vr and vt should be mass averaged or not; for now, yes, so
    #use pynbody; otherwise can use pandas to do a straight number average
    # We need to make the cuts for disk stars only. Halo stars only have low Vc and therefore make a very non-gaussian tail
    radprofileSim = pynbody.analysis.profile.Profile(subSim, min=0.00, max=16.0, nbins=48)
    vcxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'], radprofileSim['vcxy'])
    vrxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'], radprofileSim['vrxy'])

    rgc = subSim['rxy']
    dfzcut['rgc'] = pd.Series(np.asarray(rgc), index = dfzcut.index)


    deltaVr = subSim['vrxy'] - vrxyRfit(rgc)
    deltaVt = subSim['vcxy'] - vcxyRfit(rgc)

    histres = np.histogram2d(np.asarray(dfzcut.x), np.asarray(dfzcut.y), bins=[binsx,binsy], range=[extentx, extenty])
    hist1d = np.ravel(histres[0])
    emptybinMask = hist1d == 0   # where are there NO stars, can later update for minimum star #

    dfzcut['deltaVr'] = pd.Series(np.asarray(deltaVr), index = dfzcut.index)
    dfzcut['deltaVt'] = pd.Series(np.asarray(deltaVt), index = dfzcut.index)
    #dfzcut['deltaVlos'] = pd.Series(np.asarray(deltaVlos), index = dfzcut.index)
    #dfzcut['vcxy'] = pd.Series(np.asarray(subSim['vcxy']), index = dfzcut.index)
    #dfzcut['vrxy'] = pd.Series(np.asarray(subSim['vrxy']), index = dfzcut.index)

    binned = dfzcut.groupby([pd.cut(dfzcut.x, binsx), pd.cut(dfzcut.y, binsy)])
    medDeltaVr = binned.median().deltaVr
    medDeltaVt = binned.median().deltaVt

    hist1d[emptybinMask] = np.nan # set the emptpy bins not to be plotted

    hist1d[~emptybinMask] = np.asarray(medDeltaVr)
    deltaVrmap = (hist1d.reshape(nbinsx,nbinsy)).copy()
    hist1d[~emptybinMask] = np.asarray(medDeltaVt)
    deltaVtmap = (hist1d.reshape(nbinsx,nbinsy)).copy()

    dVrrot = np.rot90(deltaVrmap)
    dVtrot = np.rot90(deltaVtmap)

    meddiff_rotdVr.append(scipy.stats.nanmedian(np.ravel((dVrrot-deltaVrmap)/dVrrot)))
    meddiff_rotdVt.append(scipy.stats.nanmedian(np.ravel((dVtrot-deltaVtmap)/dVtrot)))
    med_dVr.append(deltaVrmap)
    med_dVt.append(deltaVtmap)
    med_dVrrot.append(dVrrot)
    med_dVtrot.append(dVtrot)
    del sim
    del sim_class
    del subSim
    del dfzcut
    del radprofileSim
    del histres
    del binned
    del dVrrot
    del dVtrot


np.savez('tilt_check2.npz', theta=combos, diffrot90Vr = np.array(meddiff_rotdVr), diffrot90Vt = np.array(meddiff_rotdVt), med_dVr=np.array(med_dVr), med_dVt=np.array(med_dVt), med_dVrrot=np.array(med_dVrrot), med_dVtrot=np.array(med_dVtrot))

