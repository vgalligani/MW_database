#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:58:02 2021
@purpose : Climatology analysis during RELAMPAGO SEASON OF RPFs Ku! 
@author  : V. Galligani
@email   : victoria.galligani@cima.fcen.uba.ar
-------------------------------------------------------------------------------
@TODOS(?):

-------------------------------------------------------------------------------
"""
#################################################################
# REMEMBER TO USE conda activate py37 in ernest
#################################################################
# Load libraries
import sys
import numpy as np
import h5py 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import shutil
import pandas as pd
import geopandas as gpd
from shapely.ops import cascaded_union
from numpy import genfromtxt;
from math import pi, cos, sin
from pyhdf import SD
from pyproj import Proj, transform
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

import Climatology_plots as ClimPlot
import Plots as PlottingGMITools
import osgeo


# Some matplotlib figure definitions
plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

#################################################################
prov = genfromtxt("/home/victoria.galligani/Work/Tools/provincias.txt", delimiter='')
samerica = genfromtxt("/home/victoria.galligani/Work/Tools/samerica.txt", delimiter='')

#################################################################
# Data base limits
#
xlim_min = -75; # (actually used -70 in search)
xlim_max = -50; 
ylim_min = -40; 
ylim_max = -19; 

opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
        'ylim_min': ylim_min, 'ylim_max': ylim_max}

#-------------------------------------------------------------------------- KuRPF 
import numpy.ma as ma
import gc
import psutil

#Kurpf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/KURPF/'

Kurpf_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/KURPF/'
Kurpf_data = ClimPlot.merge_KuRPF_dicts_all(Kurpf_path)

# So far this generates e.g. Kurpf_data['LON'][0:37]. To to join ... 
Kurpf = {}
for key in Kurpf_data.keys():
    Kurpf[key] =  np.concatenate(Kurpf_data[key][:])

del Kurpf_data

# for key in Kurpf_data.keys():
#     print(key)
#     Kurpf[key] =  np.ma.append( np.ma.append( np.ma.append( np.ma.append( np.ma.append( np.ma.append( np.ma.append( Kurpf_data[key][0], Kurpf_data[key][1]), 
#                               Kurpf_data[key][2]), Kurpf_data[key][3]), Kurpf_data[key][4]), Kurpf_data[key][5]), 
#                               Kurpf_data[key][6]), Kurpf_data[key][7])

#--------------------------------------------------------------------------    
# 1)  Apply area fileter
selectKurpf = np.logical_and(np.logical_and(Kurpf['LON'] >= opts['xlim_min'], Kurpf['LON'] <= opts['xlim_max']), 
                np.logical_and(Kurpf['LAT'] >= opts['ylim_min'], Kurpf['LAT'] <= opts['ylim_max']))

# check area filter 
fig = plt.figure(figsize=(12,12))     
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
plt.plot(Kurpf['R_LON'][selectKurpf], Kurpf['R_LAT'][selectKurpf], 'x', color='darkblue')
ax1.set_xlim([-80,-45])
ax1.set_ylim([-45,-15])
plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
plt.title('Location of PF centers after domain-location filter')
plt.text(-55, -17, 'Nr. PFs:'+str(Kurpf['LAT'][selectKurpf].shape[0]))

#filename = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/RPFscatter_domain.png'
#fig.savefig(filename, dpi=300,transparent=False)        

#--------------------------------------------------------------------------    
# 2) Data: “noise” filter: To remove these noisy signals, PFs with maximum 20 dBZ echo tops greater 
# than 17 km, but with less than four pixels (<80 km2) in size and minimum 85 GHz PCT warmer than 220 K,
#  are excluded from the samples.
elems = np.where(Kurpf['LON'][selectKurpf]!=0)
noise_elems = [] 
for i in elems[0]:
    if Kurpf['MAXHT20'][selectKurpf][i] > 17:    #  PFs with maximum 20 dBZ echo tops > 17 km
        if Kurpf['MIN85PCT'][selectKurpf][i] > 220:  # and warmer than 220 K
            #print('MAXHT20 and MIN85PCT conditions for:'+ str(i)+', which have NPIXELS='+str(Kurpf['NPIXELS'][selectKurpf][i]))
            if Kurpf['NPIXELS'][selectKurpf][i] < 4:   # less than four pixels (<80 km2)
                noise_elems.append(i)
                print('All conditions for:'+ str(i))

# none found for RELAMPAGO-CACTI study period !
# none found for the full data period neither

#-------------------------------------------------------------------------    
# 3) Data: snow filter: In our case summer season -> remove topography ?

# TODO!?



#--------------------------------------------------------------------------    
# ANALISIS PARA EL PAPER (UPDATED 12-2022) 
#--------------------------------------------------------------------------    









#--------------------------------------------------------------------------    
#--------------------------------------------------------------------------    





#--------------------------------------------------------------------------    
# 4) Make scatter plots and geo-locations of PFs within percentile categories
# PFs are categorized from the top 10%, 1%, 0.1%, and 0.01% extreme values (in terms of intensity)
# Then, the percentiles of the distribution of each parameter, corresponding to the cited percentages,
# For the parameters in which frequency in the distribution decreases as their value decreases (e.g., BTs or PCTs), 
# these percentages correspond to the 0.01th, 0.1th, 1th and 10th percentiles of each distribution, respectively. 
# For the parameters in which frequency decreases as their value increases (e.g., maximum radar echo top, or volumetric precipitation),
# these percentages correspond to the 99.99th, 99.9th, 99th, and 90th percentiles of each distribution, respectively.
#dir_name = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/Plots/Climatology'
dir_name = '/home/victoria.galligani/Work/Studies/Hail_MW/GPM.PF/Output/Climatology'
filename = 'full_GMI_parameters.png'
ClimPlot.plot_PCT_percentiles_GMI(dir_name, filename, Kurpf, selectKurpf)
filename = 'full_Ku_parameters.png'
ClimPlot.plot_PCT_percentiles_Ku(dir_name, filename, Kurpf, selectKurpf)

# PLOT  X=MIN37PCT, Y=MIN85PCT, Z=MIN1838
filename = 'full_min1838PCT_distrib.png'
ClimPlot.plot_MIN1838_distrib(dir_name, filename, Kurpf, selectKurpf)

# PLOT  X=MIN37PCT, Y=MAXNSZ, Z=MAXHT40
filename = 'full_MAXHT40_distrib.png'
ClimPlot.plot_MAXHT40_distrib(dir_name, filename, Kurpf, selectKurpf)

# PLOT  X=MIN85PCT, Y=PFsarea, Z=VOLRAIN_Ku
filename = 'full_VolRain_Ku_distrib.png'
ClimPlot.plot_volrain_Ku_distrib(dir_name, filename, Kurpf, selectKurpf)
#------            OJO QUE ACA       --------------------------------
# plot_volrain_Ku_distrib(dir, filename, Kurpf, MWRPF, selectKurpf, selectMWRPF, PFtype1, PFtype_area):
#NECESITO LEER MWRPF PARA DETERMINAR AREA DESDE AHI? 
#---------------------------------------------------------------------
        
        
#-------------------------------------------------------------------------    
# Get the dates of the top 0.01% 
info_orbit_min37 = ClimPlot.get_orbits_extreme(Kurpf, selectKurpf, 'MIN37PCT')
info_orbit_min85 = ClimPlot.get_orbits_extreme(Kurpf, selectKurpf, 'MIN85PCT')
info_orbit_max40 = ClimPlot.get_orbits_extreme_hi(Kurpf, selectKurpf, 'MAXHT40')

#-------------------------------------------------------------------------    
# extend analysis 
# diurnal cycle: For each proxy for the 4 years of GPM observations, the number of elements
# in each hourly bin for land or sea, has been normalized by the total number 
# of elements.
# VG: normalized by the total number of elements within the percentiles, not the total elements. 

MIN85_normelems, percentiles85 = ClimPlot.get_hourly_normbins(Kurpf, selectKurpf, 'MIN85PCT')
MIN37_normelems, percentiles37 = ClimPlot.get_hourly_normbins(Kurpf, selectKurpf, 'MIN37PCT')

MAXHT20_normelems, percentiles_maxht20 = ClimPlot.get_hourly_normbins_hi(Kurpf, selectKurpf, 'MAXHT20')
MAXHT30_normelems, percentiles_maxht30 = ClimPlot.get_hourly_normbins_hi(Kurpf, selectKurpf, 'MAXHT30')
MAXHT40_normelems, percentiles_maxht40 = ClimPlot.get_hourly_normbins_hi(Kurpf, selectKurpf, 'MAXHT40')
VOLRAIN_normelems, percentiles_volrain = ClimPlot.get_hourly_normbins_hi(Kurpf, selectKurpf, 'VOLRAIN_KU')

# maximum between 16:00 and 22:00 UTC (13:00–18:00 local time), another max, at aprox. 4UTC
# coincides with the time of the typical maximum in near-surface temperature and 
# conditional instability, revealing the strong control of thermodynamic diurnal forcing on CI.
# similar to previous studyes. cancelada, casanovas. 
fig = plt.figure(figsize=(12,12))      
gs1 = gridspec.GridSpec(2, 1)

j=1
ax1 = plt.subplot(gs1[0,0])
plt.plot(MIN37_normelems[:,j]*100, marker='s', linestyle='-', color= 'darkred', label='MIN37PCT<'+str(np.round(percentiles37[j], 2))+' K')
plt.plot(MIN85_normelems[:,j]*100, marker='s', linestyle='-', color= 'darkblue', label='MIN85PCT<'+str(np.round(percentiles85[j], 2))+' K')
plt.xlabel('Time (UTC)')
ax1.set_xticks([0,4,8,12,16,20,24])
ax1.set_ylabel('% PFs occurence')
ax1.set_title('Diurnal frequency distribution of top 1% PFs over land')
ax1.set_xlim([0, 24])
ax1.grid(True)
plt.legend()

j=1
ax1 = plt.subplot(gs1[1,0])
plt.plot(MAXHT20_normelems[:,j]*100, marker='s', linestyle='-', color= 'darkgreen', label='MAXHT20>'+str(np.round(percentiles_maxht20[j], 2))+' km')
plt.plot(MAXHT40_normelems[:,j]*100, marker='s', linestyle='-', color= 'maroon', label='MAXHT40>'+str(np.round(percentiles_maxht40[j], 2))+' km')
plt.xlabel('Time (UTC)')
ax1.set_xticks([0,4,8,12,16,20,24])
ax1.set_ylabel('% PFs occurence')
ax1.set_title('Diurnal frequency distribution of top 1% PFs over land')
ax1.set_xlim([0, 24])
ax1.grid(True)
plt.legend(loc='upper left')

filename = 'diurnalfreq_distrib.png'
fig.savefig(dir_name+filename, dpi=300,transparent=False)        

#-note, valores similares a los obtenidos por Maite para TRMM dek percentil 99%
#-------------------------------------------------------------------------    
#-------------------------------------------------------------------------    
# 2x2 degree box
# https://kbkb-wx-python.blogspot.com/2016/08/find-nearest-latitude-and-longitude.html
# https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/KDTree_nearest_neighbor.ipynb
from scipy.interpolate import griddata
lonbins = np.arange(-80, -40, 2) 
latbins = np.arange(-50, -10, 2)
xi,yi   = np.meshgrid(lonbins,latbins)
    
MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = ClimPlot.get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
MIN37PCT    = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]
zi_MIN37PCT = griddata((LON,LAT), MIN37PCT, (xi,yi), method='nearest')
    
MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = ClimPlot.get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
LON         = lonlon[np.where(MIN85PCT_cat < percentiles[1])]   
LAT         = latlat[np.where(MIN85PCT_cat < percentiles[1])]
MIN85PCT    = MIN85PCT_cat[np.where(MIN85PCT_cat < percentiles[1])]
zi_MIN85PCT = griddata((LON,LAT), MIN85PCT, (xi,yi), method='nearest')
    
MAXHT40_cat, latlat, lonlon, percentiles, _, _ = ClimPlot.get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
LON         = lonlon[np.where(MAXHT40_cat > percentiles[1])]   
LAT         = latlat[np.where(MAXHT40_cat > percentiles[1])]
MAXHT40     = MAXHT40_cat[np.where(MAXHT40_cat > percentiles[1])]
zi_MAXHT40 = griddata((LON,LAT), MAXHT40, (xi,yi), method='nearest')
    
# plot the actual values
filename = 'regridded_map.png'
ClimPlot.plot_regrid_map(lonbins, latbins, zi_MIN37PCT, zi_MIN85PCT, zi_MAXHT40, dir_name+filename,'test',LON,LAT,MIN37PCT)

        
# plot the spatial frequency distrb. note that the number of PFs within each grid box is different when
#  different proxies are considered, since different subsets of elements are selected from the entire dataset. 
# GPM sampling maximizes near 60◦S due to the orbital recurvature at that latitude, 
filename = 'spatial_distrib.png'
ClimPlot.plot_spatial_distrib(Kurpf,selectKurpf, dir_name+filename)

# normalized by dividing each bin by the total pixel number
filename = 'norm_spatial_distrib.png'
ClimPlot.plot_norm_spatial_distrib(Kurpf,selectKurpf, dir_name+filename)    


#---------- SEASONS
# Aumentar base de datos! y extender analisis por seasoon! 
# ie. same as abaove, but PFs in wet season (OCT-MARZO, ONDJFM)   10,11,12,01,02,03
#                                dry season (MAYO-AGOSTO, MJJA)   05,06,07,08

select_DRY = np.logical_and(np.logical_and(np.logical_and(Kurpf['LON'] >= opts['xlim_min'], Kurpf['LON'] <= opts['xlim_max']), 
                np.logical_and(Kurpf['LAT'] >= opts['ylim_min'], Kurpf['LAT'] <= opts['ylim_max'])), 
                        np.logical_and(Kurpf['MONTH']>=5.0, Kurpf['MONTH']<=8.0))

select_WET = np.logical_and(np.logical_and(np.logical_and(Kurpf['LON'] >= opts['xlim_min'], Kurpf['LON'] <= opts['xlim_max']), 
                np.logical_and(Kurpf['LAT'] >= opts['ylim_min'], Kurpf['LAT'] <= opts['ylim_max'])), 
                        np.logical_or(Kurpf['MONTH']>=10.0, Kurpf['MONTH']<=3.0))


ClimPlot.plot_seasonal_distributions(Kurpf, select_DRY, 'dry_season', dir_name)

#---------- 
# TODO: 
# For hail? ver paper sarah bang and bedka
# ver como maite dividio en zonas?!
#
# analyze frequency and areal coverage of database??? <----- this is important!! MAYBE LOOK AT CLASSES LOCATION AND CASE-STUDIES?
# EN RESPECTO A ESTO VER OTROS SATELITES! 





