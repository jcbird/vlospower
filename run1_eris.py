"""
run0_eris.py
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
import decomp



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
        yGC = obsDataDict['d2'] * np.sin(np.radians(obsDataDict['l']))
        zGC = obsDataDict['d3'] * np.sin(np.radians(obsDataDict['b']))

        return (np.vstack([8.0 - xGC, yGC, zGC])).transpose()

    except KeyError:
        print("Data missing from Dictionary.\nMake sure object has l,b,d2,d3 with correct naming.")

originalsim = st.load_faceon(400, config.params)
## Let's just look at subSim centered on disk
##sim = originalsim[pynbody.filt.Disc('17 kpc', '0.5 kpc')]# & pynbody.filt.LowPass('tform', 10.0)]
#subs = originalsim.__deepcopy__()[(originalsim['tform']>9.3) & (originalsim['tform']<12.32) & (originalsim['r']>1.0) & (originalsim['r']<30.0)]
subs = originalsim.__deepcopy__()[(originalsim['tform']>8.3) & (originalsim['tform']<12.32) & (originalsim['r']<30.0)]
cpos = pynbody.analysis.halo.hybrid_center(subs, r='12 kpc')
subs['pos']-=cpos
vcen = pynbody.analysis.halo.vel_center(subs, retcen=True, cen_size='29 kpc')
subs['vel']-=vcen
trans = pynbody.analysis.angmom.calc_faceon_matrix(pynbody.analysis.angmom.ang_mom_vec(subs))
pynbody.transformation.transform(subs, trans)
sim = subs[pynbody.filt.Disc('17 kpc', '0.5 kpc')]# & pynbody.filt.BandPasubs('tform', 1.0,4.0)]
#rint(pynbody.analysis.angmom.ang_mom_vec(sim))
#pynbody.analysis.angmom.faceon(sim)
#print(pynbody.analysis.angmom.ang_mom_vec(sim))

#2D velocities
#ksim['vcxy']
#ksim['vrxy']

#osim = ss.ObsSim(sim, 5.565,5.565)
osim = ss.ObsSim(sim, 8.0, 0.0)

obsdata = ss.lbdrv(sim['pos'], sim['vel'], osim.sunpos)
galCoords = gcCoords(obsdata, osim.sunpos)

gcDict = {i:galCoords[:,j] for (j,i) in enumerate(['xgc', 'ygc', 'zgc'])}
##pynSim = stp.SimPynbody(sim, sid=400)
##dfSim = stp.sim2df(pynSim)

#dfObsdata = pd.DataFrame(obsdata)
#dfGalcoords = pd.DataFrame(galCoords)

#d = [obsdata,gcDict]
dfObsDataCoords = pd.DataFrame(obsdata)
for i in gcDict.keys():
    dfObsDataCoords[i] = pd.Series(gcDict[i], index=dfObsDataCoords.index)

#extentx = [-15.0,15.0]
#extenty = [-15.0,15.0]
extentx = [5.0,12.5]
extenty = [-4.0,3.5]
#nbinsx,nbinsy = 40,40
nbinsx,nbinsy = 10,10
binsx = np.arange(extentx[0], extentx[1]+0.001, (extentx[1]-extentx[0])/nbinsx)
binsy = np.arange(extenty[0], extenty[1]+0.001, (extenty[1]-extenty[0])/nbinsy)

#Zcut in GC coords, also needed for original sim for GC vels
####decomp.decomp(originalsim, config)  # Disk and Thick disk are 1 and 2
#dfzcut = dfObsDataCoords[(np.abs(dfObsDataCoords.zgc) < 0.25)]
#ubSim = sim[np.asarray(np.abs(dfObsDataCoords.zgc) < 0.25)]
dfzcut = dfObsDataCoords[(np.abs(sim['z']) < 0.25)]
subSim = sim[np.asarray(np.abs(sim['z']) < 0.25)]
#dfzcut = dfObsDataCoords[(np.abs(sim['z']) < 0.25) & (sim.s['decomp']<1.5)]
#subSim = sim[(np.abs(sim['z']) < 0.25) & (sim.s['decomp']<1.5)]

#Need to decide if Vr and vt should be mass averaged or not; for now, yes, so
#use pynbody; otherwise can use pandas to do a straight number average

# We need to make the cuts for disk stars only. Halo stars only have low Vc and therefore make a very non-gaussian tail

radbins = np.linspace(0.1,16.5,30)
radinds = np.digitize(subSim['rxy'], radbins)
rbins = 0.5*(radbins[1:]+radbins[:-1])
uniqueinds = range(1,len(radbins)) #avoids last and first bins
galcutout = (np.asarray(dfzcut['l'])>30.0) & (np.asarray(dfzcut['l'])<210.0)
####medvcxy = np.array([np.median(subSim['vcxy'][(radinds==radialID]) for radialID in uniqueinds]) 
####medvrxy = np.array([np.median(subSim['vrxy'][radinds==radialID]) for radialID in uniqueinds]) 
#####galcutout = (subSim['x']>extentx[0]) & (subSim['x']<extentx[1]) & (subSim['y']>extenty[0]) & (subSim['y']<extenty[1])

medvcxy = np.array([np.median(subSim['vcxy'][(radinds==radialID) & (galcutout)]) for radialID in uniqueinds]) 
medvrxy = np.array([np.median(subSim['vrxy'][(radinds==radialID) & (galcutout)]) for radialID in uniqueinds]) 

#radprofileSim = pynbody.analysis.profile.Profile(subSim[galcutout], min=0.00, max=16.5, nbins=45)
radprofileSim = pynbody.analysis.profile.Profile(subSim, min=0.00, max=16.5, nbins=45)
#Cut off empty radii
radmask = ~np.isnan(medvcxy)
###radmask = ~np.isnan(radprofileSim['vcxy'])
####medvcxy = medvcxy[radmask]
####medvrxy = medvrxy[radmask]
####rbins = rbins[radmask]
#vcxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'], radprofileSim['vcxy'])
vcxyRfit = interpolate.InterpolatedUnivariateSpline(rbins[radmask], medvcxy[radmask])
vrxyRfit = interpolate.InterpolatedUnivariateSpline(rbins[radmask], medvrxy[radmask])
####vrxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'][radmask], radprofileSim['vrxy'][radmask])
####vcxyRfit = interpolate.InterpolatedUnivariateSpline(radprofileSim['rbins'][radmask], radprofileSim['vcxy'][radmask])

##rcfile = np.load('stargasdm.rotcurve.00400.npz')
##rad = rcfile['radius']
##vc = rcfile['jz_max']/rcfile['radius']

##radVcFit = interpolate.InterpolatedUnivariateSpline(rad[1:],vc[1:])
rgc = subSim['rxy']
dfzcut['rgc'] = pd.Series(np.asarray(rgc), index = dfzcut.index)


deltaVr = subSim['vrxy'] - vrxyRfit(subSim['rxy'])
deltaVt = subSim['vcxy'] - vcxyRfit(subSim['rxy'])

####deltaVlos = dfzcut['rv']/np.cos(np.radians(dfzcut.b)) - (vrxyRfit(rgc)*np.cos(np.pi - np.radians(dfzcut.l + dfzcut.az)) + vcxyRfit(rgc)*np.sin(np.radians(dfzcut.l + dfzcut.az)))
deltaVlos = dfzcut['rv']/np.cos(np.radians(dfzcut.b)) - (vrxyRfit(subSim['rxy'])*np.cos(np.pi - np.radians(dfzcut.l + dfzcut.az)) - vcxyRfit(subSim['rxy'])*np.sin(np.radians(dfzcut.l + dfzcut.az)))
#deltaVlos = dfzcut['rv']/np.cos(np.radians(dfzcut.b)) + (vcxyRfit(rgc)*np.sin(np.radians(dfzcut.l + dfzcut.az)))

#deltaV = dfzcut.rv/np.cos(np.radians(dfzcut.b)) - vcDerived*np.sin(np.radians(dfzcut.l + dfzcut.az)) 
####histres = np.histogram2d(np.asarray(dfzcut.xgc), np.asarray(dfzcut.ygc), bins=[binsx,binsy], range=[extentx, extenty])
histres = np.histogram2d(np.asarray(subSim['x']), np.asarray(subSim['y']), bins=[binsx,binsy], range=[extentx, extenty])
hist1d = np.ravel(histres[0])
emptybinMask = hist1d == 0   # where are there NO stars, can later update for minimum star #

dfzcut['deltaVr'] = pd.Series(np.asarray(deltaVr), index = dfzcut.index)
dfzcut['deltaVt'] = pd.Series(np.asarray(deltaVt), index = dfzcut.index)
dfzcut['deltaVlos'] = pd.Series(np.asarray(deltaVlos), index = dfzcut.index)

###binned = dfzcut.groupby([pd.cut(dfzcut.xgc, binsx), pd.cut(dfzcut.ygc, binsy)])
binned = dfzcut.groupby([pd.cut(subSim['x'], binsx), pd.cut(subSim['y'], binsy)])
medDeltaVr = binned.median().deltaVr
medDeltaVt = binned.median().deltaVt
medDeltaVlos = binned.median().deltaVlos


hist1d[emptybinMask] = np.nan # set the emptpy bins not to be plotted
hist1d[~emptybinMask] = np.asarray(medDeltaVr)

def plotlabel(ax,title=''):
    ax.set_xlabel(r'X$_{\mathrm{gc}}$ [kpc]', fontsize='x-large')
    ax.set_ylabel(r'Y$_{\mathrm{gc}}$ [kpc]', fontsize='x-large')
    ax.set_title(title, fontsize='x-large')

minmax_cm = 16 #N used for vmax, vmin
cmapp = 'Spectral'
cmapp = 'jet'
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


