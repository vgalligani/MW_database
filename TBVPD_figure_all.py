#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:36:54 2023

@author: vito.galligani
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import alphashape
import shapely
from matplotlib.path import Path
import matplotlib
import pyart 
#------------------------------------------------------------------------------

def points_in_polygon(pts, polygon):
    """
    Returns if the points are inside the given polygon,

    Implemented with angle accumulation.
    see: 
https://en.wikipedia.org/wiki/Point_in_polygon#Winding_number_algorithm

    :param np.ndarray pts: 2d points
    :param np.ndarray polygon: 2d polygon
    :return: Returns if the points are inside the given polygon, array[i] == True means pts[i] is inside the polygon.
    """
    polygon = np.vstack((polygon, polygon[0, :]))  # close the polygon (if already closed shouldn't hurt)
    sum_angles = np.zeros([len(pts), ])
    for i in range(len(polygon) - 1):
        v1 = polygon[i, :] - pts
        norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
        norm_v1[norm_v1 == 0.0] = 1.0  # prevent divide-by-zero nans
        v1 = v1 / norm_v1
        v2 = polygon[i + 1, :] - pts
        norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
        norm_v2[norm_v2 == 0.0] = 1.0  # prevent divide-by-zero nans
        v2 = v2 / norm_v2
        dot_prods = np.sum(v1 * v2, axis=1)
        cross_prods = np.cross(v1, v2)
        angs = np.arccos(np.clip(dot_prods, -1, 1))
        angs = np.sign(cross_prods) * angs
        sum_angles += angs

    sum_degrees = np.rad2deg(sum_angles)
    # In most cases abs(sum_degrees) should be close to 360 (inside) or to 0 (outside).
    # However, in end cases, points that are on the polygon can be less than 360, so I allow a generous margin..
    return abs(sum_degrees) > 90.0

#------------------------------------------------------------------------------
def calc_PCTs(TB_s1):
    """
    -------------------------------------------------------------
    PCTs(89 GHz) <= 200 K following Cecil et al., 2018
    PCT(10) = 2.5TBV  - 1.5TBH
    PCT(19) = 2.4TBV  - 1.4TBH
    PCT(37) = 2.15TBV - 1.15TBH
    PCT(89) = 1.7TBV  - 0.7TBH
    -------------------------------------------------------------
    OUT    PCT10   
           PCT19
           PCT37
           PCT89
    IN     TB_s1         
    -------------------------------------------------------------
    """
    PCT10 = 2.5  * TB_s1[:,:,0] - 1.5  * TB_s1[:,:,1] 
    PCT19 = 2.4  * TB_s1[:,:,2] - 1.4  * TB_s1[:,:,3] 
    PCT37 = 2.15 * TB_s1[:,:,5] - 1.15 * TB_s1[:,:,6] 
    PCT89 = 1.7  * TB_s1[:,:,7] - 0.7  * TB_s1[:,:,8] 
    
    return PCT10, PCT19, PCT37, PCT89

#------------------------------------------------------------------------------
def GET_TBVH_250_TBVHplots(options, icois, fname, changui, radar, title_in):

    home_dir = '/Users/vito.galligani/'  	
    gmi_dir  = '/Users/vito.galligani/Work/Studies/Hail_MW/GMI_data/'
	
    # ojo que aca agarro los verdaderos PCTMIN, no los que me pasó Sarah B. que estan 
    # ajustados a TMI footprints. 
    # read file
    
    colores_in = ['k','darkgreen','darkred','darkblue','cyan']     

    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()

    S1_sub_lat  = lat_gmi.copy()
    S1_sub_lon  = lon_gmi.copy()
    S1_sub_tb = tb_s1_gmi.copy()

    idx1       = (lat_gmi>=options['ylim_min']-changui) & (lat_gmi<=options['ylim_max']+changui) & (lon_gmi>=options['xlim_min']-changui) & (lon_gmi<=options['xlim_max']+changui)
    S1_sub_lat = np.where(idx1 != False, S1_sub_lat, np.nan) 
    S1_sub_lon = np.where(idx1 != False, S1_sub_lon, np.nan) 
    
    for i in range(tb_s1_gmi.shape[2]):
        S1_sub_tb[:,:,i]  = np.where(np.isnan(S1_sub_lon) != 1, tb_s1_gmi[:,:,i], np.nan)	

    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(S1_sub_tb)

    # contorno:
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,figsize=[14,12])
    contorno89 = plt.contour(lon_gmi[:,:], lat_gmi[:,:], PCT89[:,:], [250] , colors=(['r']), linewidths=1.5);
    #contorno89_FIX = plt.contour(lon_gmi[1:,:], lat_gmi[1:,:], PCT89[0:-1,:] , [250], colors=(['k']), linewidths=1.5);
    axes.set_xlim([options['xlim_min'], options['xlim_max']]) 
    axes.set_ylim([options['ylim_min'], options['ylim_max']])
        
    # tambien me va a interesar el grillado a diferentes resoluciones 	
    GMI_tbs1_37 = []
    GMI_tbs1_85 = [] 	
    # aca agrego V,H 
    GMI_tbs1_37H = []
    GMI_tbs1_85H = [] 
    GMI_tbs1_19 = []
    GMI_tbs1_19H = [] 
    GMI_latlat = []
    GMI_lonlon = []

    S1_sub_tb_v2 = S1_sub_tb[:,:,:][idx1]		

    # GET X and Y points that make the contorno for each contorno: 
    # using https://stackoverflow.com/questions/70701788/find-points-that-lie-in-a-concave-hull-of-a-point-cloud
    for ii in range(len(icois)): 
        X1 = []; Y1 = []; vertices = []
        for ik in range(len(contorno89.collections[0].get_paths()[int(icois[ii])].vertices)):
            X1.append(contorno89.collections[0].get_paths()[icois[ii]].vertices[ik][0])
            Y1.append(contorno89.collections[0].get_paths()[icois[ii]].vertices[ik][1])            
        points = np.vstack([X1, Y1]).T
        
        # DATAPOINTS that I wish to know if they are inside the contour of interest 
        datapts = np.column_stack((lon_gmi[:,:][idx1], lat_gmi[:,:][idx1] ))
        cond = points_in_polygon(datapts, points)
        plt.plot(lon_gmi[:,:][idx1][cond] , lat_gmi[:,:][idx1][cond], 'o', markersize=10, markerfacecolor=colores_in[ii], 
			   markeredgecolor=colores_in[ii], label=str(icois[ii]))

        GMI_tbs1_37.append( S1_sub_tb_v2[cond,5] ) 
        GMI_tbs1_85.append( S1_sub_tb_v2[cond,7] ) 
        GMI_tbs1_37H.append( S1_sub_tb_v2[cond,6] ) 
        GMI_tbs1_85H.append( S1_sub_tb_v2[cond,8] ) 
        GMI_tbs1_19.append( S1_sub_tb_v2[cond,2] ) 
        GMI_tbs1_19H.append( S1_sub_tb_v2[cond,3] ) 
        GMI_latlat.append( lat_gmi[:,:][idx1][cond] )
        GMI_lonlon.append( lon_gmi[:,:][idx1][cond] )
	
    plt.legend()
    plt.title(title_in)

    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)



    return  GMI_latlat, GMI_lonlon, GMI_tbs1_19, GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_19H, GMI_tbs1_37H, GMI_tbs1_85H

#----------------------------------------------------------------------------------------------     
def get_median(y, x, tbvbin):

    TBV_bin  = np.arange(50,300,tbvbin)
    TBVH_bin = np.arange(-5,25,tbvbin)
    figfig = plt.figure(figsize=(30,10))
    [counts, xedges, yedges, _] = plt.hist2d( x,
        y, bins=[TBV_bin, TBVH_bin], norm=matplotlib.colors.LogNorm())
    xbins = np.digitize( x, xedges[1:-1])
    running_median = [np.median(y[xbins==k]) for k in range(len(xedges))]
    del counts, xedges, yedges
    plt.close()
    
    return [running_median]

#------------------------------------------------------------------------------  
def pyplot_rings(lat_radar,lon_radar,radius):
    """
    Calculate lat-lon of the maximum range ring
    lat_radar : float32
       Radar latitude. Positive is north.
    lon_radar : float32
       Radar longitude. Positive is east.
    radius : float32
       Radar range in kilometers.
    returns : numpy.array
       A 2d array containing the 'radius' range latitudes (lat) and longitudes (lon)
    """
    import numpy as np

    R=12742./2.
    m=2.*np.pi*R/360.
    alfa=np.arange(-np.pi,np.pi,0.0001)

    nazim  = 360.0
    nbins  = 480.0
    binres = 0.5

    #azimuth = np.transpose(np.tile(np.arange(0,nazim,1), (int(nbins),1)))
    #rangos  = np.tile(np.arange(0,nbins,1)*binres, (int(nazim),1))
    #lats    = lat_radar + (rangos/m)*np.cos((azimuth)*np.pi/180.0)
    #lons    = lon_radar + (rangos/m)*np.sin((azimuth)*np.pi/180.0)/np.cos(lats*np.pi/180)

    lat_radius = lat_radar + (radius/m)*np.sin(alfa)
    lon_radius = lon_radar + ((radius/m)*np.cos(alfa)/np.cos(lat_radius*np.pi/180))

    return lat_radius, lon_radius


#------------------------------------------------------------------------------

plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12  
    
tbvbin = 15
TBV_bin  = np.arange(50,300,tbvbin)

gmi_dir  = '/Users/vito.galligani/Work/Studies/Hail_MW/GMI_data/'
r_dir    = '/Users/vito.galligani/Work/Studies/Hail_MW/radar_data/'

#------------------------------------------------------------------------------
# VER DE AGREGAR PARAMETRIZACION?
# 89 GHz
TB89 = np.arange(70,10,250)
PD89 = [np.nan, 1.52, 1.59, 1.85, 2.07, 2.09, 2.41, 2.82, 3.11, 3.57, 4.15, 4.82, 5.77, 7.25, 8.04, 8.13, 7.71, 6.68, 5.13]

# 166 GHz
TB166 = np.arange(70,10,250)
PD166 = [np.nan, 1.91, 2.46, 2.67, 2.86, 3.16m 3.80, 5.06, 7.16, 8.80, 9.61, 10.16, 10.53, 10.89, 10.91, 10.27, 9.00, 7.26, 4.97]
#----------------------------------------------------------------------------------------------     
# primero con contarnos de 250 K solo aquellas con phail > 50%


#----------------------------------------------------------------------------------------------     
# RMA1 - 8/2/2018
# con contornos de 250 K, usamos coi=3 y coi=4, donde solo coi=4 tiene Phail > 50% 
gfile    = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'  #21UTC
rfile    = 'RMA1/cfrad.20180208_205749.0000_to_20180208_210014.0000_RMA1_0201_03.nc' 
radar = pyart.io.read(r_dir+rfile)

opts           = {'xlim_min': -65.5, 'xlim_max': -63.5, 'ylim_min': -33, 'ylim_max': -30.5}

# para 250K
coi_250        = [4]  
coi_250_LABELS = ['4']

# para 200K
#coi_250        = [2,3,4]  

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '08/02/2018')
caso_08022018 = {'case': '08/02/2018', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

fig, axes = plt.subplots(nrows=4, ncols=4, constrained_layout=True,figsize=[10,10])

axes[0,0].plot(caso_08022018['GMI_tbv_85'], caso_08022018['GMI_pd_85'],'x')
running_median = get_median( caso_08022018['GMI_pd_85'], caso_08022018['GMI_tbv_85'], tbvbin)
axes[0,0].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-', color='darkblue', label=r'P$_{hail}$=53.4%')
#axes.xlabel('89-GHz TBV (K)')
#axes.ylabel('89-GHz Polarization Difference (K)')
axes[0,0].set_xlim([50,250])
axes[0,0].set_ylim([0,15])
axes[0,0].legend()
axes[0,0].grid(True)      
axes[0,0].set_title('08/02/2018')

running_median0 = running_median.copy()
#----------------------------------------------------------------------------------------------     
# # CSPR2 - 11/11/2018    
gfile     = '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5'
rfile     = 'CSPR2_data/corcsapr2cmacppiM1.c1.20181111.130003.nc' #'corcsapr2cfrppiM1.a1.20181111.130003.nc'
radar = pyart.io.read(r_dir+rfile)

opts = {'xlim_min': -65, 'xlim_max': -63.6, 'ylim_min': -32.6, 'ylim_max': -31.5}

# para 250K
coi_250 =  [10]	

# para 200K
# coi_250 =  [2,3,4,5]	

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '11/11/2018')
caso_11112018 = {'case': '11/11/2018', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }


del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H
        
#fig = plt.figure(figsize=(5,5))
axes[0,1].plot(caso_11112018['GMI_tbv_85'], caso_11112018['GMI_pd_85'],'x')
running_median = get_median( caso_11112018['GMI_pd_85'], caso_11112018['GMI_tbv_85'], tbvbin)
axes[0,1].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-', color='darkblue', label=r'P$_{hail}$=65.3%')
# plt.xlabel('89-GHz TBV (K)')
# plt.ylabel('89-GHz Polarization Difference (K)')
axes[0,1].set_xlim([50,250])
axes[0,1].set_ylim([0,15])
axes[0,1].legend()
axes[0,1].grid(True)      
axes[0,1].set_title('11/11/2018')

running_median1 = running_median.copy()

#----------------------------------------------------------------------------------------------
# # DOW7 - 14/12/2018 
gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
rfile     = 'DOW7/cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc'
radar = pyart.io.read(r_dir+rfile)

#opts = {'xlim_min': -70, 'xlim_max': -50, 'ylim_min': -40, 'ylim_max': -20}
opts = {'xlim_min': -67.5, 'xlim_max': -57.5, 'ylim_min': -37.5, 'ylim_max': -30}

# para 250K
coi_250 =  [0]	

# para 200K
# coi_250 =  [3]	

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile, 0, radar, '14/12/2018')
caso_14122018 = {'case': '14/12/2018', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                 'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]   }


del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H
       
#fig = plt.figure(figsize=(5,5))
axes[0,2].plot(caso_14122018['GMI_tbv_85'], caso_14122018['GMI_pd_85'],'x')
running_median = get_median( caso_14122018['GMI_pd_85'], caso_14122018['GMI_tbv_85'], tbvbin)
axes[0,2].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-',  color='darkblue', label=r'P$_{hail}$=96.7%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[0,2].set_xlim([50,250])
axes[0,2].set_ylim([0,15])
axes[0,2].legend()
axes[0,2].grid(True)
axes[0,2].set_title('14/12/2018') 

running_median2 = running_median.copy()

#----------------------------------------------------------------------------------------------     
# # RMA1 - 03/08/2019
gfile     = '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5'
rfile = 'RMA1/cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc'
radar = pyart.io.read(r_dir+rfile)


opts = {'xlim_min': -65.2, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30}

# para 250K
coi_250 =  [6]	

# para 200K
# coi_250 =  [4,5,6,7]	


[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '03/08/2019')
caso_03082019 = {'case': '03/08/2019', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))
axes[0,3].plot(caso_03082019['GMI_tbv_85'], caso_03082019['GMI_pd_85'],'x')
running_median = get_median( caso_03082019['GMI_pd_85'], caso_03082019['GMI_tbv_85'], tbvbin)
axes[0,3].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-', color='darkblue', label=r'P$_{hail}$=89.5%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[0,3].set_xlim([50,250])
axes[0,3].set_ylim([0,15])
axes[0,3].legend()
axes[0,3].grid(True)
axes[0,3].set_title('03/08/2019')

running_median3 = running_median.copy()

#----------------------------------------------------------------------------------------------     
# # RMA5 - 15/08/2020
gfile    = '1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
rfile = 'RMA5/cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc' 
radar = pyart.io.read(r_dir+rfile)

opts = {'xlim_min': -55.0, 'xlim_max': -52.0, 'ylim_min': -27.5, 'ylim_max': -25.0}

# para 250K
coi_250 =  [4]	

# para 200K
# coi_250 =  [2,4,5,6]	

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '15/08/2020')
caso_15082020 = {'case': '15/08/2020', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))
axes[1,0].plot(caso_15082020['GMI_tbv_85'], caso_15082020['GMI_pd_85'],'x')
running_median = get_median( caso_15082020['GMI_pd_85'], caso_15082020['GMI_tbv_85'], tbvbin)
axes[1,0].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-',  color='darkblue', label=r'P$_{hail}$=72.7%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[1,0].set_xlim([50,250])
axes[1,0].set_ylim([0,15])
axes[1,0].legend()
axes[1,0].grid(True)
axes[1,0].set_title('15/08/2020 (72.7%)')

running_median4 = running_median.copy()

#----------------------------------------------------------------------------------------------     
# # RMA4 - 09/02/2018
gfile    = '1B.GPM.GMI.TB2016.20180209-S184820-E202054.022451.V05A.HDF5' 
rfile = 'RMA4/cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc' 
radar = pyart.io.read(r_dir+rfile)
  
opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26}

 	
# para 200K
coi_250 =  [0]
 	
# para 250K
# coi_250 =  [0,1,2,3]

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '09/02/2018')
caso_09022018_v1 = {'case': '09/02/2018', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))
axes[1,1].plot(caso_09022018_v1['GMI_tbv_85'], caso_09022018_v1['GMI_pd_85'],'x', color='red')
running_median_v1 = get_median( caso_09022018_v1['GMI_pd_85'], caso_09022018_v1['GMI_tbv_85'], tbvbin)
axes[1,1].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median_v1), lw=2, linestyle='-',  color='darkred', label=r'P$_{hail}$=76.2%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[1,1].set_xlim([50,250])
axes[1,1].set_ylim([0,15])
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_title('09/02/2018')

running_median5_v1 = running_median_v1.copy()


# #----------------------------------------------------------------------------------------------  
# # # RMA4 - 31/10/2018
gfile    = '1B.GPM.GMI.TB2016.20181031-S005717-E022950.026546.V05A.HDF5' 
rfile    = 'RMA4/cfrad.20181031_010936.0000_to_20181031_011525.0000_RMA4_0200_01.nc' 
radar = pyart.io.read(r_dir+rfile)
 

opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -25.5}

# para 250K
coi_250 =  [0,4,11,17]# [0,4,11,17]	

# para 200K
# coi_250 =  [4,0,11,17]

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar,  '31/10/2018')
caso_31102018 = {'case': '31/10/2018', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85_1': GMI_tbs1_85[0], 'GMI_pd_85_1': GMI_tbs1_85[0]-GMI_tbs1_85H[0],
                  'GMI_tbv_85_2': GMI_tbs1_85[1], 'GMI_pd_85_2': GMI_tbs1_85[1]-GMI_tbs1_85H[1],
                  'GMI_tbv_85_3': GMI_tbs1_85[2], 'GMI_pd_85_3': GMI_tbs1_85[2]-GMI_tbs1_85H[2],
                  'GMI_tbv_85_4': GMI_tbs1_85[3], 'GMI_pd_85_4': GMI_tbs1_85[3]-GMI_tbs1_85H[3] }
 
del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))



axes[1,2].plot(caso_31102018['GMI_tbv_85_1'], caso_31102018['GMI_pd_85_1'],'x', color='red')
running_median = get_median( caso_31102018['GMI_pd_85_1'], caso_31102018['GMI_tbv_85_1'], tbvbin)
axes[1,2].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median[0]), lw=2, linestyle='-',color='darkred', label=r'P$_{hail}$=93.1?%')
axes[1,2].set_xlim([50,250])
axes[1,2].set_ylim([0,15])
axes[1,2].grid(True)
axes[1,2].legend()
axes[1,2].set_title('31/10/2018') #' (99.3, 93.1, 73.8, 26.6%)')


axes[1,3].plot(caso_31102018['GMI_tbv_85_2'], caso_31102018['GMI_pd_85_2'],'x', color='red')
running_median = get_median( caso_31102018['GMI_pd_85_2'], caso_31102018['GMI_tbv_85_2'], tbvbin)
axes[1,3].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median[0]), lw=2, linestyle='-',color='darkred',label=r'P$_{hail}$=73.8%')
axes[1,3].set_xlim([50,250])
axes[1,3].set_ylim([0,15])
axes[1,3].grid(True)
axes[1,3].legend()
axes[1,3].set_title('31/10/2018') #' (99.3, 93.1, 73.8, 26.6%)')


# axes[1,3].plot(caso_31102018['GMI_tbv_85_3'], caso_31102018['GMI_pd_85_3'],'x', color='red')
# running_median = get_median( caso_31102018['GMI_pd_85_3'], caso_31102018['GMI_tbv_85_3'], tbvbin)
# axes[1,3].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median[0]), lw=2, linestyle='-',color='darkred',label=r'P$_{hail}$=26.6%')
# axes[1,3].set_xlim([50,250])
# axes[1,3].set_ylim([0,15])
# axes[1,3].grid(True)
# axes[1,3].legend()
# axes[1,3].set_title('31/10/2018') #' (99.3, 93.1, 73.8, 26.6%)')


axes[2,0].plot(caso_31102018['GMI_tbv_85_4'], caso_31102018['GMI_pd_85_4'],'x', color='red')
running_median = get_median( caso_31102018['GMI_pd_85_4'], caso_31102018['GMI_tbv_85_4'], tbvbin)
axes[2,0].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median[0]), lw=2, linestyle='-',color='darkred',label=r'P$_{hail}$=99.3%')
axes[2,0].set_xlim([50,250])
axes[2,0].set_ylim([0,15])
axes[2,0].grid(True)
axes[2,0].legend()
axes[2,0].set_title('31/10/2018') #' (99.3, 93.1, 73.8, 26.6%)')


running_median6 = running_median.copy()


#----------------------------------------------------------------------------------------------     
# # RMA3 - 05/03/2019
gfile     = '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5'
rfile     = 'RMA3/cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'
radar = pyart.io.read(r_dir+rfile)

opts = {'xlim_min': -63, 'xlim_max': -58, 'ylim_min': -30, 'ylim_max': -23}

# para 250K
coi_250 =  [17]	

# para 200K
# coi_250 =  [1,2,3,4]

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar, '05/03/2019')
caso_05032019 = {'case': '05/03/2019', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))
axes[2,1].plot(caso_05032019['GMI_tbv_85'], caso_05032019['GMI_pd_85'],'x', color='red')
running_median = get_median( caso_05032019['GMI_pd_85'], caso_05032019['GMI_tbv_85'], tbvbin)
axes[2,1].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-', color='darkred', label=r'P$_{hail}$=73.7%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[2,1].set_xlim([50,250])
axes[2,1].set_ylim([0,15])
axes[2,1].legend()
axes[2,1].grid(True)
axes[2,1].set_title('05/03/2019') #' (73.7 y 10.3 %)')

running_median7 = running_median.copy()

#----------------------------------------------------------------------------------------------  
# # RMA4 - 09/02/2019
gfile    = '1B.GPM.GMI.TB2016.20190209-S191744-E205018.028129.V05A.HDF5'
rfile    = 'RMA4/cfrad.20190209_192724.0000_to_20190209_193317.0000_RMA4_0200_01.nc' 
radar = pyart.io.read(r_dir+rfile)


opts = {'xlim_min': -61.5, 'xlim_max': -56.5, 'ylim_min': -29.5, 'ylim_max': -26}

# para 250K
coi_250 =  [2,4]	

# para 200K
# coi_250 =  [0,1,3,5]

[_, _, _, GMI_tbs1_37, GMI_tbs1_85, _, GMI_tbs1_37H, GMI_tbs1_85H] = GET_TBVH_250_TBVHplots(opts, coi_250, gmi_dir+gfile,1, radar,  '09/02/2019')
caso_09022019 = {'case': '09/02/2019', 'GMI_tbv_37': GMI_tbs1_37, 'GMI_pd_37': GMI_tbs1_37[0]-GMI_tbs1_37H[0],
                  'GMI_tbv_85': GMI_tbs1_85[0], 'GMI_pd_85': GMI_tbs1_85[0]-GMI_tbs1_85H[0]          }

del GMI_tbs1_37, GMI_tbs1_85, GMI_tbs1_37H, GMI_tbs1_85H

#fig = plt.figure(figsize=(5,5))
axes[2,2].plot(caso_09022019['GMI_tbv_85'], caso_09022019['GMI_pd_85'],'x',color='red')
running_median = get_median( caso_09022019['GMI_pd_85'], caso_09022019['GMI_tbv_85'], tbvbin)
axes[2,2].plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median), lw=2, linestyle='-',  color='darkred', label=r'P$_{hail}$=98.9%')
#plt.xlabel('89-GHz TBV (K)')
#plt.ylabel('89-GHz Polarization Difference (K)')
axes[2,2].set_xlim([50,250])
axes[2,2].set_ylim([0,15])
axes[2,2].legend()
axes[2,2].grid(True)
axes[2,2].set_title('09/02/2019')

running_median8 = running_median.copy()

fig.delaxes(axes[2,3])
fig.delaxes(axes[3,0])
fig.delaxes(axes[3,1])
fig.delaxes(axes[3,2])
fig.delaxes(axes[3,3])

axes[0,0].set_xticks([50,100,150,200,250])
axes[0,1].set_xticks([50,100,150,200,250])
axes[0,2].set_xticks([50,100,150,200,250])
axes[0,3].set_xticks([50,100,150,200,250])
axes[1,0].set_xticks([50,100,150,200,250])
axes[1,1].set_xticks([50,100,150,200,250])
axes[1,2].set_xticks([50,100,150,200,250])
axes[1,3].set_xticks([50,100,150,200,250])
axes[2,0].set_xticks([50,100,150,200,250])
axes[2,1].set_xticks([50,100,150,200,250])
axes[2,2].set_xticks([50,100,150,200,250])

axes[2,0].set_xlabel('89-GHz TBV (K)')
axes[2,0].set_ylabel('89-GHz Polarization Difference (K)')


#----------------------------------------------------------------------------------------------  
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    import matplotlib.pyplot as plt
    import numpy as np

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    
    return base.from_list(cmap_name, color_list, N)

#----------------------------------------------------------------------------------------------  
# # # TBV - PD PLOTS
# base = plt.cm.get_cmap('Paired')
# colores_in = base(np.linspace(0, 1, 9))

# fig = plt.figure(figsize=(5,5))

# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median0), lw=2, linestyle='-', color=colores_in[0], label='08/02/2018')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median1), lw=2, linestyle='-', color=colores_in[1], label='11/11/2018')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median2), lw=2, linestyle='-', color=colores_in[2], label='14/12/2018')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median3), lw=2, linestyle='-', color=colores_in[3], label='03/08/2019')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median4), lw=2, linestyle='-', color=colores_in[4], label='15/08/2020')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median5), lw=2, linestyle='-', color=colores_in[5], label='09/02/2018')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median6), lw=2, linestyle='-', color=colores_in[6], label='31/10/2018')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median7), lw=2, linestyle='-', color=colores_in[7], label='05/03/2019')
# plt.plot(TBV_bin-(TBV_bin[1]-TBV_bin[0])/2, np.ravel(running_median8), lw=2, linestyle='-', color=colores_in[8], label='09/02/2019')  
    
# plt.xlabel('89-GHz TBV (K)')
# plt.ylabel('89-GHz Polarization Difference (K)')
# plt.xlim([50,250])
# plt.ylim([0,15])
# plt.legend()
# plt.grid(True)

