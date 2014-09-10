"""
run1_eris.py
============

Serves as first draft for getting peculiar velocity plots to Jo Bovy for paper.

Note email dated 8/10/2014 from D. Weinberg
"""

from __future__ import print_function
import matplotlib as mpl
#mpl.use('Agg')
import config
import sim_tools as st
import survey_sims as ss
import numpy as np
import simtopandas as stp
import pandas as pd
import scipy
import pynbody
from scipy import interpolate as interpolate
from matplotlib import pyplot as plt
import pyfits
#import decomp
import sys


def rc_spatial(rc_path):
    fits = pyfits.open(rc_path)
    data = fits[1].data
    xgc = data['RC_GALR']*np.cos(data['RC_GALPHI'])
    ygc = data['RC_GALR']*np.sin(data['RC_GALPHI'])
    zgc = data['RC_GALZ']
    fits.close()
    return xgc, ygc, zgc

def gcCoords(obsDataDict, sim, sunpos):
    """
    Function to calculate rectagular galacto-centric coordinates given l, b,
    distance2d, distance3d.

    obsDataDict is a dictionary object with the following keys:

    'l', 'b', 'd2', 'd3'

    Assume you want Sun at (xgc=8,ygc=0, zgc=0)

    sunpos  a tuple or array (3d)
    """
    try:

        xGC = sim['rxy'] * np.cos(np.radians(obsDataDict['az']))
        yGC = sim['rxy'] * np.sin(np.radians(obsDataDict['az']))
        zGC = sim['z'] #* np.sin(np.radians(obsDataDict['b']))

        return (np.vstack([xGC, yGC, zGC])).transpose()

    except KeyError:
        print("Data missing from Dictionary.\nMake sure object has l,b,d2,d3 with correct naming.")

originalsim = st.load_faceon(400, config.params)
## Let's just look at subSim centered on disk
tmin = (originalsim['tform']>9.3) 
tmax = (originalsim['tform']<12.32)
radmax = (originalsim['r']<40.0)
subs = originalsim.__deepcopy__()[tmin & tmax & radmax ]
#subs = originalsim.__deepcopy__()[(originalsim['r']<40.0)]
cpos = pynbody.analysis.halo.hybrid_center(subs, r='12 kpc')
subs['pos']-=cpos
vcen = pynbody.analysis.halo.vel_center(subs, retcen=True, cen_size='29 kpc')
subs['vel']-=vcen
trans = pynbody.analysis.angmom.calc_faceon_matrix(pynbody.analysis.angmom.ang_mom_vec(subs))
pynbody.transformation.transform(subs, trans)
sim = subs[pynbody.filt.Disc('17 kpc', '0.5 kpc')]# & pynbody.filt.BandPasubs('tform', 1.0,4.0)]

#osim = ss.ObsSim(sim, 5.565,5.565)
osim = ss.ObsSim(sim, 0.0,-8.0)
obsdata = ss.lbdrv(sim['pos'], sim['vel'], osim.sunpos)
galCoords = gcCoords(obsdata, sim, osim.sunpos)
gcDict = {i:galCoords[:,j] for (j,i) in enumerate(['xgc', 'ygc', 'zgc'])}

dfObsDataCoords = pd.DataFrame(obsdata)
for i in gcDict.keys():
    dfObsDataCoords[i] = pd.Series(gcDict[i], index=dfObsDataCoords.index)

#MIDPLANE for simulation and observed data
dfzcut = dfObsDataCoords[(np.abs(sim['z']) < 0.25)] #& (np.abs(np.asarray(dfObsDataCoords['b']))<1.5)]
subSim = sim[np.asarray(np.abs(sim['z']) < 0.25)] #& (np.abs(np.asarray(dfObsDataCoords['b']))<1.5)]

#BUILD UP ROTATION CURVE
galcutout_rotcuve = (np.asarray(dfzcut['l'])>30.0) & (np.asarray(dfzcut['l'])<210.0) \
        & (np.asarray(dfzcut['d2'])<5.0) & (np.abs(np.asarray(dfzcut['b']))<1.5)


#Divide galaxy cutouf into bins
sim_rotcurve = subSim[galcutout_rotcuve]

radbins = np.linspace(4.0,16.5,25)
radinds = np.digitize(sim_rotcurve['rxy'], radbins)
rbins = 0.5*(radbins[1:]+radbins[:-1])
uniqueinds = range(1,len(radbins)) #avoids last and first bins

medvcxy = np.array([np.median(sim_rotcurve['vcxy'][(radinds==radialID)]) for radialID in uniqueinds]) 
medvrxy = np.array([np.median(sim_rotcurve['vrxy'][(radinds==radialID)]) for radialID in uniqueinds]) 
radmask = ~np.isnan(medvcxy)

vcxyRfit = interpolate.InterpolatedUnivariateSpline(rbins[radmask], medvcxy[radmask])
vrxyRfit = interpolate.InterpolatedUnivariateSpline(rbins[radmask], medvrxy[radmask])

#Alternatively, if you want MEAN rotational velocity as funciton of radius
#radprofileSim = pynbody.analysis.profile.Profile(sim_rotcurve, min=4.0, max=16.5, nbins=25)
#vrxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'][radmask], radprofileSim['vrxy'][radmask])
#vcxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'][radmask], radprofileSim['vcxy'][radmask])


#USING RC catalog, determine area over which to perform measurement
rc_x, rc_y, rc_z = rc_spatial('rcsample_v402.fits')
zmask = np.abs(rc_z) < 0.25
binsx = np.arange(5.5,12.251,0.75)
binsy = np.arange(-3.0, 3.751, 0.75)
rangex = [5.5,12.25]
rangey = [-3.0, 3.75]
histres = np.histogram2d(rc_x[zmask], rc_y[zmask], bins=[binsx,binsy], range=[rangex, rangey])
#make mask to match BOVY geometry
rcmask = histres[0]<30

#NOW grab residual velocities to model

rgc = subSim['rxy']
dfzcut['rgc'] = pd.Series(np.asarray(rgc), index = dfzcut.index)
deltaVr = subSim['vrxy'] - vrxyRfit(subSim['rxy'])
deltaVt = subSim['vcxy'] - vcxyRfit(subSim['rxy'])
####deltaVlos = dfzcut['rv']/np.cos(np.radians(dfzcut.b)) - (vrxyRfit(rgc)*np.cos(np.pi - np.radians(dfzcut.l + dfzcut.az)) + vcxyRfit(rgc)*np.sin(np.radians(dfzcut.l + dfzcut.az)))
deltaVlos = dfzcut['rv']/np.cos(np.radians(dfzcut.b)) - (vrxyRfit(subSim['rxy'])*np.cos(np.pi - np.radians(dfzcut.l + dfzcut.az)) - vcxyRfit(subSim['rxy'])*np.sin(np.radians(dfzcut.l + dfzcut.az)))

dfzcut['deltaVr'] = pd.Series(np.asarray(deltaVr), index = dfzcut.index)
dfzcut['deltaVt'] = pd.Series(np.asarray(deltaVt), index = dfzcut.index)
dfzcut['deltaVlos'] = pd.Series(np.asarray(deltaVlos), index = dfzcut.index)

#Get median value as a function of position in galaxy
binned = dfzcut.groupby([pd.cut(dfzcut['xgc'], binsx), pd.cut(dfzcut['ygc'], binsy)])
medDeltaVr = binned.median().deltaVr
medDeltaVt = binned.median().deltaVt
medDeltaVlos = binned.median().deltaVlos

def plotlabel(ax,title=''):
    ax.set_xlabel(r'X$_{\mathrm{gc}}$ [kpc]', fontsize='x-large')
    ax.set_ylabel(r'Y$_{\mathrm{gc}}$ [kpc]', fontsize='x-large')
    ax.set_title(title, fontsize='x-large')

#now make plots
histshape = histres[0].shape
cmapp = 'jet'
minmax_cm = 16
#DeltaVc

fig = plt.figure()
ax = fig.add_subplot(111)
deltaV = medDeltaVt.reshape(histshape)
deltaV[rcmask] = np.nan
im = ax.imshow(deltaV.T, aspect='auto',  interpolation='none', \
        extent=[rangex[0], rangex[1], rangey[0], rangey[1]], cmap=cmapp, \
        origin='lower', vmax=minmax_cm, vmin=-minmax_cm)
 
cb = plt.colorbar(im)
cb.set_label(r'median $\Delta\mathrm{V}_{\phi,\phi}$ [km/s]', fontsize = 'large')
plotlabel(ax)
plt.savefig('vt_map.png', format='png')
sys.exit()

hist1d[~emptybinMask] = np.asarray(medDeltaVt)
deltaV2map = hist1d.reshape(nbinsx,nbinsy)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
im2 = ax2.imshow(deltaV2map.T, aspect='auto',  interpolation='none', extent=[extentx[0], extentx[1], extenty[0], extenty[1]], cmap=cmapp, origin='lower', vmax=minmax_cm, vmin=-minmax_cm)

histres = np.histogram2d(np.asarray(subSim['x']), np.asarray(subSim['y']), bins=[binsx,binsy], range=[extentx, extenty])
hist1d = np.ravel(histres[0])
emptybinMask = hist1d == 0   # where are there NO stars, can later update for minimum star #




hist1d[emptybinMask] = np.nan # set the emptpy bins not to be plotted
hist1d[~emptybinMask] = np.asarray(medDeltaVr)


minmax_cm = 16 #N used for vmax, vmin
cmapp = 'Spectral'
deltaVmap = hist1d.reshape(nbinsx,nbinsy)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(deltaVmap.T, aspect='auto',  interpolation='none', extent=[extentx[0], extentx[1], extenty[0], extenty[1]], cmap=cmapp, origin='lower', vmax=minmax_cm, vmin=-minmax_cm)
 
cb = plt.colorbar(im)
cb.set_label(r'median $\Delta\mathrm{V}_{\mathrm{r,r}}$ [km/s]', fontsize = 'large')
plotlabel(ax)
plt.savefig('vrmap.png', format='png')


hist1d[~emptybinMask] = np.asarray(medDeltaVt)
deltaV2map = hist1d.reshape(nbinsx,nbinsy)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
im2 = ax2.imshow(deltaV2map.T, aspect='auto',  interpolation='none', extent=[extentx[0], extentx[1], extenty[0], extenty[1]], cmap=cmapp, origin='lower', vmax=minmax_cm, vmin=-minmax_cm)

cb2 = plt.colorbar(im2)

cb2.set_label(r'median $\Delta\mathrm{V}_{\phi,\phi}$ [km/s]', fontsize = 'large')
plotlabel(ax2)
plt.savefig('vtmap.png', format='png')
#plt.show()

hist1d[~emptybinMask] = np.asarray(medDeltaVlos)
deltaV3map = hist1d.reshape(nbinsx,nbinsy)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
im3 = ax3.imshow(deltaV3map.T, aspect='auto',  interpolation='none', extent=[extentx[0], extentx[1], extenty[0], extenty[1]], cmap=cmapp, origin='lower', vmax=minmax_cm, vmin=-minmax_cm)

cb3 = plt.colorbar(im3)

cb3.set_label(r'median $\Delta\mathrm{V}_{\mathrm{los}, \phi; \mathrm{r}}$ [km/s]', fontsize = 'large')
plotlabel(ax3)
plt.savefig('vlosmap.png', format='png')
#plt.show()


