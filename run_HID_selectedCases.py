#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:26:28 2022
@author: victoria.galligani
"""
# Code that plots 

from os import listdir
import pyart
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from shapely.geometry import Polygon
import matplotlib
import h5py
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import platform
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from os.path import isfile, join
import pandas as pd
from pyart.correct import phase_proc
import xarray as xr
from pyart.core.transforms import antenna_to_cartesian
from copy import deepcopy
from csu_radartools import csu_fhc
import matplotlib.colors as colors
#import wradlib as wrl


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def set_plot_settings(var): 
    """
    -------------------------------------------------------------
    
    -------------------------------------------------------------
    OUT    [units, cmap, vmin, vmax, mac, int, under] todas de importancia
                    para colormpas de las variables del radar.      
    IN     var      Zhh, Zdr, Kdp, Ah, phidp, rhohv, Doppler Vel. 
    ------
    notes: mas colormaps en https://github.com/jjhelmus/pyart_colormaps_and_limits 
    """
    if var == 'Zhh':
        units = 'Reflectivity [dBZ]'
        cmap = colormaps('ref')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'white'
    elif var == 'Zhh_2':
        units = 'Zh [dBZ]'
        cmap = colormaps('ref2')
        vmin = 0
        vmax = 60
        max = 60.01
        intt = 5
        under = 'white'
        over = 'black'
    elif var == 'Zdr':
        units = 'ZDR [dBZ]'
        vmin = -2
        vmax = 10
        max = 10.01
        intt = 1
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(int(N), 'jet') # colormaps('zdr')
        under = 'white'
        over = 'white'
    elif var == 'Kdp':
        units = 'KDP [deg/km]'
        vmin = -0.1
        vmax = 0.7
        max = 0.71
        intt = 0.1
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(10, 'jet')
        under = 'white'
        over = 'black'
    elif var == 'Ah':
        units = '[dB/km]'
        vmin = 0
        vmax = 0.5
        max = 0.51
        intt = 0.05
        N = (vmax-vmin)/intt
        cmap = discrete_cmap(N, 'brg_r')
        under = 'black'
        over = 'white'
    elif var == 'phidp':
        units = 'PHIDP [deg]'
        vmin = 0
        vmax = 360
        max = 360.1
        intt = 10
        N = (vmax-vmin)/intt
        print(N)
        cmap = discrete_cmap(int(N), 'jet')
        under = 'white'
        over = 'white'
    elif var == 'rhohv':
        units = r'$\rho_{hv}$'
        vmin = 0.5
        vmax = 1.
        max = 1.01
        intt = 0.1
        N = round((vmax-vmin)/intt)
        cmap = pyart.graph.cm.RefDiff #fun.discrete_cmap(N, 'jet')
        under = 'white'
        over = 'white'
    else: #Doppler por ejemplo
        units = r'V$_r$ [m/s]'
        cmap = pyart.graph.cm.BuDRd18
        vmin = -10
        vmax = 10
        max = 10.01
        intt = 2
        under = 'white'
        over = 'white'    
        
    return units, cmap, vmin, vmax, max, intt, under, over

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
def colormaps(variable):
    """
    Choose colormap for a radar variable
    variable : str
       Radar variable to define which colormap to use. (e.g. ref,
       dv, zdr..) More to come
    returns : matplotlib ColorList
       A Color List for the specified *variable*.
    """
    import matplotlib.colors as colors

    # Definicion de las disntitas paletas de colores:
    nws_ref_colors =([ [ 1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   0.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    #[212.0/255.0,   0.0/255.0,   0.0/255.0, 0.4],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [248.0/255.0,   0.0/255.0, 253.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])
    # Definicion de las disntitas paletas de colores:
    nws_ref_colors_transparent =([ [ 1.0/255.0, 159.0/255.0, 244.0/255.0, 0.3],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0, 0.3],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0, 0.3],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0, 0.3],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0, 0.3],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0, 0.3],
                    [229.0/255.0, 188.0/255.0,   0.0/255.0, 1],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0, 1],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0, 1],
                    #[212.0/255.0,   0.0/255.0,   0.0/255.0, 0.4],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0, 1],
                    [248.0/255.0,   0.0/255.0, 253.0/255.0, 1],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0, 1]
                    ])
    
    nws_zdr_colors = ([ [  1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   2.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])

    nws_dv_colors = ([  [0,  1,  1],
                    [0,  0.966666638851166,  1],
                    [0,  0.933333337306976,  1],
                    [0,  0.899999976158142,  1],
                    [0,  0.866666674613953,  1],
                    [0,  0.833333313465118,  1],
                    [0,  0.800000011920929,  1],
                    [0,  0.766666650772095,  1],
                    [0,  0.733333349227905,  1],
                    [0,  0.699999988079071,  1],
                    [0,  0.666666686534882,  1],
                    [0,  0.633333325386047,  1],
                    [0,  0.600000023841858,  1],
                    [0,  0.566666662693024,  1],
                    [0,  0.533333361148834,  1],
                    [0,  0.5,  1],
                    [0,  0.466666668653488,  1],
                    [0,  0.433333337306976,  1],
                    [0,  0.400000005960464,  1],
                    [0,  0.366666674613953,  1],
                    [0,  0.333333343267441,  1],
                    [0,  0.300000011920929,  1],
                    [0,  0.266666680574417,  1],
                    [0,  0.233333334326744,  1],
                    [0,  0.200000002980232,  1],
                    [0,  0.16666667163372,   1],
                    [0,  0.133333340287209,  1],
                    [0,  0.100000001490116,  1],
                    [0,  0.0666666701436043, 1],
                    [0,  0.0333333350718021, 1],
                    [0,  0,  1],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [1,  0,  0],
                    [1,  0.0322580635547638, 0],
                    [1,  0.0645161271095276, 0],
                    [1,  0.0967741906642914, 0],
                    [1,  0.129032254219055,  0],
                    [1,  0.161290317773819,  0],
                    [1,  0.193548381328583,  0],
                    [1,  0.225806444883347,  0],
                    [1,  0.25806450843811,   0],
                    [1,  0.290322571992874,  0],
                    [1,  0.322580635547638,  0],
                    [1,  0.354838699102402,  0],
                    [1,  0.387096762657166,  0],
                    [1,  0.419354826211929,  0],
                    [1,  0.451612889766693,  0],
                    [1,  0.483870953321457,  0],
                    [1,  0.516129016876221,  0],
                    [1,  0.548387110233307,  0],
                    [1,  0.580645143985748,  0],
                    [1,  0.612903237342834,  0],
                    [1,  0.645161271095276,  0],
                    [1,  0.677419364452362,  0],
                    [1,  0.709677398204803,  0],
                    [1,  0.74193549156189,   0],
                    [1,  0.774193525314331,  0],
                    [1,  0.806451618671417,  0],
                    [1,  0.838709652423859,  0],
                    [1,  0.870967745780945,  0],
                    [1,  0.903225779533386,  0],
                    [1,  0.935483872890472,  0],
                    [1,  0.967741906642914,  0],
                    [1,  1,  0]   ])


    cmap_nws_ref = colors.ListedColormap(nws_ref_colors)
    cmap_nws_zdr = colors.ListedColormap(nws_zdr_colors)
    cmap_nws_dv = colors.ListedColormap(nws_dv_colors)
    cmap_nws_ref_trans = colors.ListedColormap(nws_ref_colors_transparent)
    

    if variable == 'ref':
       return cmap_nws_ref
    if variable == 'ref2':
       return cmap_nws_ref_trans
        

    if variable == 'zdr':
       return cmap_nws_zdr

    if variable == 'dv':
       return cmap_nws_dv

#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------  
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

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
def despeckle_phidp(phi, rho):
    '''
    Elimina pixeles aislados de PhiDP
    '''

    # Unmask data and despeckle
    dphi = np.copy(phi)
    
    # Descartamos pixeles donde RHO es menor que un umbral (e.g., 0.7) o no está definido (e.g., NaN)
    dphi[np.isnan(rho)] = np.nan
    rho_thr = 0.7
    dphi[rho < rho_thr] = np.nan
    
    # Calculamos la textura de RHO (rhot) y descartamos todos los pixeles de PHIDP por encima
    # de un umbral de rhot (e.g., 0.25)
    rhot = wrl.dp.texture(rho)
    rhot_thr = 0.5
    dphi[rhot > rhot_thr] = np.nan
    
    # Eliminamos pixeles aislados rodeados de NaNs
    # https://docs.wradlib.org/en/stable/generated/wradlib.dp.linear_despeckle.html     
    dphi = wrl.dp.linear_despeckle(dphi, ndespeckle=5, copy=False)

    return dphi

#------------------------------------------------------------------------------------
def unfold_phidp(phi, rho, diferencia):
    '''
    Unfolding
    '''
       
    # Dimensión del PPI (elevaciones, azimuth, bins)
    nb = phi.shape[1]
    nr = phi.shape[0]

    phi_cor = np.zeros((nr, nb)) #Asigno cero a la nueva variable phidp corregida
    
    v1 = np.zeros(nb)  #Vector v1
    
    # diferencia = 200 # Valor que toma la diferencia entre uno y otro pixel dentro de un mismo azimuth
    
    for m in range(0, nr):
    
        v1 = phi[m,:]
        v2 = np.zeros(nb)
    
        for l in range(0, nb):
            a = v2[l-1] - v1[l]

            if np.isnan(a):
                v2[l] = v2[l-1]

            elif a > diferencia:
                v2[l] = v1[l] + 360
                if v2[l-1] - v2[l] > 100:  # Esto es por el doble folding cuando es mayor a 700
                    v2[l] = v1[l] + v2[l-1]

            else:
                v2[l] = v1[l]
    
        phi_cor[m,:] = v2
    
    phi_cor[phi_cor <= 0] = np.nan
    phi_cor[rho < .7] = np.nan
    
    return phi_cor

#------------------------------------------------------------------------------------    
def subtract_sys_phase(phi, sys_phase):

    nb = phi.shape[1]
    nr = phi.shape[0]
    phi_final = np.copy(phi) * 0
    phi_err=np.ones((nr, nb)) * np.nan
    
    try:
        phi_final = phi-sys_phase
    except:
        phi_final = phi_err
    
    return phi_final
#------------------------------------------------------------------------------------       
def correct_phidp(phi, rho, sys_phase, diferencia):
    
    dphi = despeckle_phidp(phi, rho)
    uphi = unfold_phidp(dphi, rho, diferencia)
    
    # Reemplazo nan por sys_phase para que cuando reste esos puntos queden en cero
    uphi = np.where(np.isnan(uphi), sys_phase, uphi)
    phi_cor = subtract_sys_phase(uphi, sys_phase)
    # phi_cor[rho<0.7] = np.nan
    
    return phi_cor
#------------------------------------------------------------------------------  
# this function adapted from the Scipy Cookbook:
# http://www.scipy.org/Cookbook/SignalSmooth
def smooth_and_trim(x, window_len=11, window='hanning'):
    """
    Smooth data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    Parameters
    ----------
    x : array
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window; should be an odd integer.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' or 'sg_smooth'. A flat window will produce a moving
        average smoothing.
    Returns
    -------
    y : array
        The smoothed signal with length equal to the input signal.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman',
                     'sg_smooth']
    if window not in valid_windows:
        raise ValueError("Window is on of " + ' '.join(valid_windows))

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(int(window_len), 'd')
    elif window == 'sg_smooth':
        w = np.array([0.1, .25, .3, .25, .1])
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2):len(x) + int(window_len / 2)]

#------------------------------------------------------------------------------  
def plot_test_ppi(radar, phi_corr, lat_pf, lon_pf):
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    phidp = radar.fields['PHIDP']['data'][start_index:end_index]
    rhv   = radar.fields['RHOHV']['data'][start_index:end_index]
    dphi  = phi_corr[start_index:end_index]
    #------ 
    # Test plot figure: 
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True,
                        figsize=[14,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        TH = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        TH = radar.fields['DBZH']['data'][start_index:end_index]
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,0].grid(True)
    axes[0,0].plot(lon_pf, lat_pf, marker='o', markersize=60, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=2) 
    #-- RHOHV
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,1].pcolormesh(lons, lats, rhv, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,1].grid(True)
    #-- PHIDP
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[1,0].pcolormesh(lons, lats, phidp, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[1,0].grid(True)
    #-- DPHIDP
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[1,1].pcolormesh(lons, lats, dphi, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[1,1].grid(True)

    return
#------------------------------------------------------------------------------  
def adjust_fhc_colorbar_for_pyart(cb):
    
    # HID types:           Species #:  
    # -------------------------------
    # Drizzle                  1    
    # Rain                     2    
    # Ice Crystals             3
    # Aggregates               4
    # Wet Snow                 5
    # Vertical Ice             6
    # Low-Density Graupel      7
    # High-Density Graupel     8
    # Hail                     9
    # Big Drops                10
    
    
    cb.set_ticks(np.arange(2.4, 10, 0.9))
    cb.ax.set_yticklabels(['Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

#------------------------------------------------------------------------------  
def plot_corrections_ppi_wHID(radar, phi_corr, ZHCORR, attenuation, ZDRoffset, lat_pf, lon_pf):
    
    hid_colors = ['MediumBlue', 'DarkOrange', 'LightPink',
              'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    phidp = radar.fields['PHIDP']['data'][start_index:end_index]
    rhv   = radar.fields['RHOHV']['data'][start_index:end_index]
    dphi  = phi_corr[start_index:end_index]
    Zh_corr      = ZHCORR[start_index:end_index]
    att          = attenuation[start_index:end_index]
    HID          = radar.fields['HID']['data'][start_index:end_index]
    ZDR_corr2    = radar.fields['ZDR_correc_ZPHI']['data'][start_index:end_index]
    Zh_corr2     = radar.fields['dBZ_correc_ZPHI']['data'][start_index:end_index]
    #Zh_corr_zphi = ZHCORR_ZPHI[start_index:end_index]
    #------ 
    # Test plot figure: 
    fig, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True,
                        figsize=[28,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        TH = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        TH = radar.fields['DBZH']['data'][start_index:end_index]
    if 'ZDR' in radar.fields.keys():  
        zdr = radar.fields['ZDR']['data'][start_index:end_index]    
    else:
        zdr = radar.fields['TH']['data'][start_index:end_index]-radar.fields['TV']['data'][start_index:end_index]
    
    #------ Zh (observed)
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,0].grid(True)
    axes[0,0].plot(lon_pf, lat_pf, marker='o', markersize=60, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=2) 
    #-- RHOHV
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,1].pcolormesh(lons, lats, rhv, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,1].grid(True)
    #-- ZDR
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
    pcm1 = axes[0,2].pcolormesh(lons, lats, zdr-ZDRoffset, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,2], shrink=1, label='ZDR (w/ offset)', ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,2].grid(True)
    #-- DPHIDP
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[0,3].pcolormesh(lons, lats, dphi, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,3], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,3].grid(True)
    #-- Zh - Zh_corr
    pcm1 = axes[1,0].pcolormesh(lons, lats, TH - Zh_corr, vmin=-5, vmax=5)
    cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label='ZH-ZHCORR(calc_att)')
    axes[1,0].grid(True)
    #-- HID
    pcm1 = axes[1,1].pcolormesh(lons, lats, HID, cmap=cmaphid, vmin=1.8, vmax=10.4)
    cbar = plt.colorbar(pcm1, ax=axes[1,1], label='CSU HID')
    cbar = adjust_fhc_colorbar_for_pyart(cbar)
    cbar.cmap.set_under('white')
    
    return

#------------------------------------------------------------------------------  
def plot_corrections_ppi(radar, phi_corr, ZHCORR, attenuation, ZDRoffset, lat_pf, lon_pf):
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    phidp = radar.fields['PHIDP']['data'][start_index:end_index]
    rhv   = radar.fields['RHOHV']['data'][start_index:end_index]
    dphi  = phi_corr[start_index:end_index]
    Zh_corr      = ZHCORR[start_index:end_index]
    att          = attenuation[start_index:end_index]
    ZDR_corr2    = radar.fields['ZDR_correc_ZPHI']['data'][start_index:end_index]
    Zh_corr2     = radar.fields['dBZ_correc_ZPHI']['data'][start_index:end_index]
    #Zh_corr_zphi = ZHCORR_ZPHI[start_index:end_index]
    #------ 
    # Test plot figure: 
    fig, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True,
                        figsize=[28,12])
    #-- Zh: 
    if 'TH' in radar.fields.keys():  
        TH = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        TH = radar.fields['DBZH']['data'][start_index:end_index]
    if 'ZDR' in radar.fields.keys():  
        zdr = radar.fields['ZDR']['data'][start_index:end_index]    
    else:
        zdr = radar.fields['TH']['data'][start_index:end_index]-radar.fields['TV']['data'][start_index:end_index]
    
    #------ Zh (observed)
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,0].grid(True)
    axes[0,0].plot(lon_pf, lat_pf, marker='o', markersize=60, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=2) 
    #-- RHOHV
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,1].pcolormesh(lons, lats, rhv, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,1].grid(True)
    #-- ZDR
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
    pcm1 = axes[0,2].pcolormesh(lons, lats, zdr-ZDRoffset, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,2], shrink=1, label='ZDR (w/ offset)', ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,2].grid(True)
    #-- DPHIDP
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[0,3].pcolormesh(lons, lats, dphi, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,3], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,3].grid(True)
    #-- Zh - Zh_corr
    pcm1 = axes[1,0].pcolormesh(lons, lats, TH - Zh_corr, vmin=-5, vmax=5)
    cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label='ZH-ZHCORR(calc_att)')
    axes[1,0].grid(True)
    #-- Zh - Zh_corr (ZPHI)
    pcm1 = axes[1,1].pcolormesh(lons, lats, TH - Zh_corr2)
    cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label='ZH-ZHCORR(calc_att_zphi)')
    axes[1,1].grid(True)    
     #--Z DR - ZDR_corr (ZPHI)
    pcm1 = axes[1,2].pcolormesh(lons, lats, zdr-ZDR_corr2)
    cbar = plt.colorbar(pcm1, ax=axes[1,2], shrink=1, label='ZDR-ZDR_CORR(calc_att_zphi)')
    axes[1,2].grid(True)         
    return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_rhi_RMA(file, fig_dir, dat_dir, radar_name, xlim_range1, xlim_range2, test_transect, ZDRoffset, freezing_lev): 
    
    radar = pyart.io.read(dat_dir+file) 
    # dict_keys(['PHIDP', 'CM', 'RHOHV', 'TH', 'TV', 'KDP']) for RMA1
    # dict_keys(['DBZH', 'KDP', 'RHOHV', 'PHIDP', 'CM']) for RMA5
    # dict_keys(['DBZV', 'DBZH', 'ZDR', 'KDP', 'RHOHV', 'PHIDP', 'VRAD']) for SOME RMA5
    print(radar.fields.keys())
    
    #- Radar sweep
    nelev       = 0
    start_index = radar.sweep_start_ray_index['data'][nelev]
    end_index   = radar.sweep_end_ray_index['data'][nelev]
    lats0        = radar.gate_latitude['data'][start_index:end_index]
    lons0        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]
            
    Ze_transect     = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); Ze_transect[:]=np.nan
    ZDR_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); ZDR_transect[:]=np.nan
    PHI_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); PHI_transect[:]=np.nan
    lon_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lon_transect[:]=np.nan
    lat_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lat_transect[:]=np.nan
    RHO_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full((  len(radar.sweep_start_ray_index['data']), lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); gate_range[:]=np.nan

    azydims = lats0.shape[1]-1
    
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]       
        if radar_name == 'RMA1':
            ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            TV       = radar.fields['TV']['data'][start_index:end_index]
            ZDRZDR      = ZHZH-TV
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]        
        elif radar_name == 'RMA5':
            ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'DBZV' in radar.fields.keys(): 
                TV     = radar.fields['DBZV']['data'][start_index:end_index]     
                ZDRZDR = ZHZH-TV   
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]        
        elif radar_name == 'RMA4':
            if 'TH' in radar.fields.keys():
                ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            elif 'DBZH' in radar.fields.keys():
                ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'TV' in radar.fields.keys(): 
                TV     = radar.fields['TV']['data'][start_index:end_index]     
                ZDRZDR = (ZHZH-TV)-ZDRoffset   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = (radar.fields['ZDR']['data'][start_index:end_index])-ZDRoffset 
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]  
        elif radar_name == 'RMA3':
            if 'TH' in radar.fields.keys():
                ZHZH       = radar.fields['TH']['data'][start_index:end_index]
            elif 'DBZH' in radar.fields.keys():
                ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
            if 'TV' in radar.fields.keys(): 
                TV     = radar.fields['TV']['data'][start_index:end_index]     
                ZDRZDR = ZHZH-TV   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = radar.fields['ZDR']['data'][start_index:end_index]     
            RHORHO  = radar.fields['RHOHV']['data'][start_index:end_index]   
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        # En verdad buscar azimuth no transecta ... 
        azimuths    = radar.azimuth['data'][start_index:end_index]
        target_azimuth = azimuths[test_transect]  #- target azimuth for nlev=0 test case is 301.5
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect[nlev,:]     = lons[filas,:]
        lat_transect[nlev,:]     = lats[filas,:]
        #
        gateZ    = radar.gate_z['data'][start_index:end_index]
        gateX    = radar.gate_x['data'][start_index:end_index]
        gateY    = radar.gate_y['data'][start_index:end_index]
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        Ze_transect[nlev,:]      = ZHZH[filas,:]
        ZDR_transect[nlev,:]     = ZDRZDR[filas,:]
        RHO_transect[nlev,:]     = RHORHO[filas,:]
        # 
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[filas,:]/1e3, azimuths[filas],radar.get_elevation(nlev)[0]);
        approx_altitude[nlev,:] = zgate/1e3
        gate_range[nlev,:]      = gates_range[filas,:]/1e3;
                
    #---------------------------------------- REFLECTIVITY
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect, cmap=colormaps('ref'), vmin=0, vmax=60)

    #- De esta manera me guardo el color con el que rellenar los polygons (scatter plot para sacar el color de cada pixel)
    print(len(radar.sweep_start_ray_index['data']))
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
         fig = plt.figure(figsize=[30,10])
         fig.add_subplot(221)
         sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                 s=1,c=Ze_transect[nlev,:],
                 cmap=colormaps('ref'), vmin=0, vmax=60)
         color[nlev,:,:] = sc.to_rgba(Ze_transect[nlev,:])
         plt.close()

    #- Try polygons
    fig2, axes = plt.subplots(nrows=3,ncols=1,constrained_layout=True,figsize=[8,6])  # 8,4 muy chiquito
    fig1 = plt.figure(figsize=(15,20))
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
         if nlev > 9: continue
         # Create the cone for each elevation IN TERMS OF RANGE. 
         # ===> ACA HABRIA QUE AGREGAR COMO CAMBIA LA ALTURA CON EL RANGE (?)
         ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
         ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
         P1 = Polygon([( gate_range[nlev,0],    approx_altitude[nlev,0]-ancho_haz_i0      ),
                   ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                   ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                   ( gate_range[nlev,0],    approx_altitude[nlev,0]+ancho_haz_i0      )])
         ancho = 50/1E3
         # Location of gates? Every 100m?  
         LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                   (gate_range[nlev,x]+ancho, 0),
                   (gate_range[nlev,x]+ancho, 50),
                   (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
         # Plot
         for i, l in enumerate(LS):
             # Get the polygon of the intersection between the cone and the space 
             #reserved for a specific point
             inter = l.intersection(P1)
             x,y = inter.exterior.xy    
             # Then plot it, filled by the color we want
             axes[0].fill(x, y, color = color[nlev,i,:], )
             x, y = P1.exterior.xy
         axes[0].set_ylim([0, 20])
         axes[0].set_ylabel('Altitude (km)')
         axes[0].grid()
         axes[0].set_xlim((xlim_range1, xlim_range2))
         norm = matplotlib.colors.Normalize(vmin=0.,vmax=60.)
         cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormaps('ref'))
         cax.set_array(Ze_transect)
         cbar_z = fig2.colorbar(cax, ax=axes[0], shrink=1.1, ticks=np.arange(0,60.01,10), label='Zh (dBZ)')
         axes[0].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)
    del mycolorbar, x, y, inter
    #---------------------------------------- ZDR
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                ZDR_transect,
                cmap= colormaps('zdr'), vmin=-2, vmax=5.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=ZDR_transect[nlev,:],
                cmap= colormaps('zdr'), vmin=-2, vmax=5.)
        color[nlev,:,:] = sc.to_rgba(ZDR_transect[nlev,:])
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        if nlev > 9: continue
        # Create the cone for each elevation IN TERMS OF RANGE. 
        ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
        ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
        P1 = Polygon([( gate_range[nlev,0],   approx_altitude[nlev,0]-ancho_haz_i0      ),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                  ( gate_range[nlev,0],       approx_altitude[nlev,0]+ancho_haz_i0      )])
        ancho = 50/1E3
        # Location of gates? Every 100m?  
        LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                  (gate_range[nlev,x]+ancho, 0),
                  (gate_range[nlev,x]+ancho, 50),
                  (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
        # Plot
        #ax1 = plt.gca()
        for i, l in enumerate(LS):
            inter = l.intersection(P1)
            x,y = inter.exterior.xy
            # Then plot it, filled by the color we want
            axes[1].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[1].set_ylim([0, 20])
        axes[1].set_ylabel('Altitude (km)')
        axes[1].grid()
        axes[1].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=-2.,vmax=5.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormaps('zdr'))
        cax.set_array(ZDR_transect)
        cbar_zdr = fig2.colorbar(cax, ax=axes[1], shrink=1.1, ticks=np.arange(-2.,5.01,1.), label='ZDR')     
        axes[1].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, inter
    #---------------------------------------- RHOHV
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(221)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude,
                RHO_transect,
                cmap = pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
    plt.close()

    #- De esta manera me guardo el color con el que rellenar los polygons
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        # scatter plot para sacar el color de cada pixel 
        fig = plt.figure(figsize=[30,10])
        fig.add_subplot(221)
        sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                s=1,c=RHO_transect[nlev,:],
                cmap= pyart.graph.cm.RefDiff, vmin=0.7, vmax=1.)
        color[nlev,:,:] = sc.to_rgba(RHO_transect[nlev,:])   # pyart.graph.cm.RefDiff
        plt.close()

    #- Try polygons
    #fig1.add_subplot(412)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        if nlev > 9: continue
        # Create the cone for each elevation IN TERMS OF RANGE. 
        ancho_haz_i0    = (np.pi/180*gate_range[nlev,0]/2)
        ancho_haz_i1099 = (np.pi/180*gate_range[nlev,azydims]/2)
        P1 = Polygon([( gate_range[nlev,0],   approx_altitude[nlev,0]-ancho_haz_i0      ),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]-ancho_haz_i1099),
                  ( gate_range[nlev,azydims], approx_altitude[nlev,azydims]+ancho_haz_i1099),
                  ( gate_range[nlev,0],       approx_altitude[nlev,0]+ancho_haz_i0      )])
        ancho = 50/1E3
        # Location of gates? Every 100m?  
        LS = [Polygon([(gate_range[nlev,x]-ancho, 0),
                  (gate_range[nlev,x]+ancho, 0),
                  (gate_range[nlev,x]+ancho, 50),
                  (gate_range[nlev,x]-ancho, 50)]) for x in np.arange(approx_altitude.shape[1])]
        # Plot
        #ax1 = plt.gca()
        for i, l in enumerate(LS):
            inter = l.intersection(P1)
            x,y = inter.exterior.xy
            # Then plot it, filled by the color we want
            axes[2].fill(x, y, color = color[nlev,i,:], )
            x, y = P1.exterior.xy
        axes[2].set_ylim([0, 20])
        axes[2].set_ylabel('Altitude (km)')
        axes[2].grid()
        axes[2].set_xlim((xlim_range1, xlim_range2))
        norm = matplotlib.colors.Normalize(vmin=0.7,vmax=1.)
        cax = matplotlib.cm.ScalarMappable(norm=norm, cmap=pyart.graph.cm.RefDiff)
        cax.set_array(RHO_transect)
        cbar_rho = fig2.colorbar(cax, ax=axes[2], shrink=1.1, ticks=np.arange(0.7,1.01,0.1), label=r'$\rho_{hv}$')     
        axes[2].axhline(y=freezing_lev,color='k',linestyle='--', linewidth=1.2)

    del mycolorbar, x, y, inter
    
    #- savefile
    plt.suptitle(radar_name + ': '+str(file[0:12]) ,fontweight='bold')
    #fig.savefig(fig_dir+'pseudo_RHI'+str(file)+'.png', dpi=300,transparent=False)
    plt.close()    
     
    return 

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def interpolate_sounding_to_radar(snd_T, snd_z, radar):
    """Takes sounding data and interpolates it to every radar gate."""
    radar_z = get_z_from_radar(radar)
    radar_T = None
    shape   = np.shape(radar_z)
    rad_z1d = radar_z.ravel()
    rad_T1d = np.interp(rad_z1d, snd_z, snd_T)
    return np.reshape(rad_T1d, shape), radar_z
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    if 'TH' in radar.fields.keys():  
        dz_field='TH'
    elif 'DBZH' in radar.fields.keys():
        dz_field='DBZH'   
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def get_z_from_radar(radar):
    """Input radar object, return z from radar (km, 2D)"""
    azimuth_1D = radar.azimuth['data']
    elevation_1D = radar.elevation['data']
    srange_1D = radar.range['data']
    sr_2d, az_2d = np.meshgrid(srange_1D, azimuth_1D)
    el_2d = np.meshgrid(srange_1D, elevation_1D)[1]
    xx, yy, zz = antenna_to_cartesian(sr_2d/1000.0, az_2d, el_2d) # Cartesian coordinates in meters from the radar.

    return zz + radar.altitude['data'][0]

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def correct_attenuation_zdr(radar, gatefilter, zdr_name, phidp_name, alpha):
    """
    Correct attenuation on differential reflectivity. KDP_GG has been
    cleaned of noise, that's why we use it.
    V. N. Bringi, T. D. Keenan and V. Chandrasekar, "Correcting C-band radar
    reflectivity and differential reflectivity data for rain attenuation: a
    self-consistent method with constraints," in IEEE Transactions on Geoscience
    and Remote Sensing, vol. 39, no. 9, pp. 1906-1915, Sept. 2001.
    doi: 10.1109/36.951081
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter: GateFilter
        Filter excluding non meteorological echoes.
    zdr_name: str
        Differential reflectivity field name.
    phidp_name: str
        PHIDP field name.
    alpha: float == 0.016
        Z-PHI coefficient.
    Returns:
    ========
    zdr_corr: array
        Attenuation corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]["data"].copy()
    phi = radar.fields[phidp_name]["data"].copy()

    zdr_corr = zdr + alpha * phi
    zdr_corr[gatefilter.gate_excluded] = np.NaN
    zdr_corr = np.ma.masked_invalid(zdr_corr)
    np.ma.set_fill_value(zdr_corr, np.NaN)

    # Z-PHI coefficient from Bringi et al. 2001
    zdr_meta = pyart.config.get_metadata("differential_reflectivity")
    zdr_meta["description"] = "Attenuation corrected differential reflectivity using Bringi et al. 2001."
    zdr_meta["_FillValue"] = np.NaN
    zdr_meta["_Least_significant_digit"] = 2
    zdr_meta["data"] = zdr_corr

    return zdr_meta

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def GMI_colormap(): 
    
    _turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],
                        [0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],
                        [0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],
                        [0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],
                        [0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],
                        [0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],
                        [0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],
                        [0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],
                        [0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],
                        [0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],
                        [0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],
                        [0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],
                        [0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],
                        [0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],
                        [0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],
                        [0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],
                        [0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],
                        [0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],
                        [0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],
                        [0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],
                        [0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],
                        [0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],
                        [0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],
                        [0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],
                        [0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],
                        [0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],
                        [0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],
                        [0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],
                        [0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],
                        [0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],
                        [0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],
                        [0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],
                        [0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],
                        [0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],
                        [0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],
                        [0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],
                        [0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],
                        [0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],
                        [0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],
                        [0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],
                        [0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],
                        [0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],
                        [0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],
                        [0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],
                        [0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],
                        [0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],
                        [0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],
                        [0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],
                        [0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],
                        [0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],
                        [0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],
                        [0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],
                        [0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],
                        [0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],
                        [0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],
                        [0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],
                        [0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],
                        [0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],
                        [0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],
                        [0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],
                        [0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],
                        [0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],
                        [0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],
                        [0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],
                        [0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],
                        [0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],
                        [0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],
                        [0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],
                        [0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],
                        [0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],
                        [0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],
                        [0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],
                        [0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],
                        [0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],
                        [0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],
                        [0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],
                        [0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],
                        [0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],
                        [0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],
                        [0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],
                        [0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],
                        [0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],
                        [0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],
                        [0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],
                        [0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],
                        [0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],
                        [0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],
                        [0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],
                        [0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],
                        [0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],
                        [0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],
                        [0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],
                        [0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],
                        [0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],
                        [0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],
                        [0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],
                        [0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],
                        [0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],
                        [0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],
                        [0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],
                        [0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],
                        [0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],
                        [0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],
                        [0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],
                        [0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],
                        [0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],
                        [0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],
                        [0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],
                        [0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],
                        [0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],
                        [0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],
                        [0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],
                        [0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],
                        [0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],
                        [0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],
                        [0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],
                        [0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],
                        [0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],
                        [0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],
                        [0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],
                        [0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],
                        [0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],
                        [0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],
                        [0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],
                        [0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],
                        [0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],
                        [0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],
                        [0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]

    _turbo_colormap_data.reverse()
    cmaps = {}
    cmaps['turbo_r'] = ListedColormap(_turbo_colormap_data, name='turbo_r')
    
    return cmaps


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a K.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} K" if plt.rcParams["text.usetex"] else f"{s} K"

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_37(fname, options, radardat_dir, radar_file, lon_pfs, lat_pfs):
    
    radar = pyart.io.read(radardat_dir+radar_file) 
    reflectivity_name = 'TH'   
    
    fontsize = 12
    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'

    # Shapefiles for cartopy 
    geo_reg_shp = home_dir+'Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)

    countries = shpreader.Reader(home_dir+'Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black')
    rivers = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none',
        edgecolor='black')

    
    cmaps = GMI_colormap() 
    
    # read file
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    
    # keep domain of interest only by keeping those where the center nadir obs is inside domain
    inside_s1   = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <=  options['xlim_max']), 
                              np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))
    inside_s2   = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <=  options['xlim_max']), 
                                         np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))    

    
    tb_s1_gmi[np.where(lon_gmi[:,110] >=  opts['xlim_max']+5),:,:] = np.nan
    tb_s1_gmi[np.where(lon_gmi[:,110] <=  opts['xlim_min']-5),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,110] >=  opts['ylim_max']+5),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,110] <=  opts['ylim_min']-5),:,:] = np.nan
    
    # CALCULATE PCTs
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(tb_s1_gmi) 
    
    fig = plt.figure(figsize=(12,7)) 
    gs1 = gridspec.GridSpec(1, 3)
    
    # BT(37)       
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi[:], lat_gmi[:], 
           c=tb_s1_gmi[:,:,5], s=20, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, '-k', linewidth=1)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    CONTORNO3 = plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
    
    return fig, CONTORNO3
  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_gmi(fname, options, radardat_dir, radar_file, lon_pfs, lat_pfs):
    
    radar = pyart.io.read(radardat_dir+radar_file) 
    reflectivity_name = 'TH'   
    
    fontsize = 12
    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'

    # Shapefiles for cartopy 
    geo_reg_shp = home_dir+'Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)

    countries = shpreader.Reader(home_dir+'Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black')
    rivers = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none',
        edgecolor='black')

    
    cmaps = GMI_colormap() 
    
    # read file
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    
    # keep domain of interest only by keeping those where the center nadir obs is inside domain
    inside_s1   = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <=  options['xlim_max']), 
                              np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))
    inside_s2   = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <=  options['xlim_max']), 
                                         np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))    

    
    tb_s1_gmi[np.where(lon_gmi[:,110] >=  opts['xlim_max']+1),:,:] = np.nan
    tb_s1_gmi[np.where(lon_gmi[:,110] <=  opts['xlim_min']-1),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,110] >=  opts['ylim_max']+1),:,:] = np.nan
    tb_s1_gmi[np.where(lat_gmi[:,110] <=  opts['ylim_min']-1),:,:] = np.nan
    
    # CALCULATE PCTs
    PCT10, PCT19, PCT37, PCT89 = calc_PCTs(tb_s1_gmi) 
    
    fig = plt.figure(figsize=(12,7)) 
    gs1 = gridspec.GridSpec(1, 3)
    
    # BT(37)       
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi[:], lat_gmi[:], 
           c=tb_s1_gmi[:,:,5], s=20, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,1), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, '-k', linewidth=1)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
    
    # BT(89)              
    ax2 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax2.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax2.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax2.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax2.add_feature(states_provinces,linewidth=0.4)
    ax2.add_feature(rivers)
    ax2.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi[inside_s1], lat_gmi[inside_s1], 
           c=tb_s1_gmi[inside_s1,7], s=20, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz', fontsize=fontsize)
    ax2.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax2.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax2.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,1), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5)   
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
     
    # BT(166)           
    ax3 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax3.set_extent([options['xlim_min'], options['xlim_max'], 
                    options['ylim_min'], options['ylim_max']], crs=crs_latlon)
    ax3.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax3.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax3.add_feature(states_provinces,linewidth=0.4)
    ax3.add_feature(rivers)
    ax3.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_s2_gmi[inside_s2], lat_s2_gmi[inside_s2], 
           c=tb_s2_gmi[inside_s2,0], s=20, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 166 GHz', fontsize=fontsize)
    ax3.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax3.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,1), crs=crs_latlon)
    ax3.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,1), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    for i in range(len(lon_pfs)):
        plt.plot(lon_pfs[i], lat_pfs[i], marker='x', markersize=10, markerfacecolor="none",
            markeredgecolor='magenta', markeredgewidth=1.5) 
    # contorno de 200 K: The features are defined as contiguous areas with 85 GHz (89 for GPM) below 200K
    CONTORNO3 = plt.contour(lon_gmi, lat_gmi, PCT89, [200], colors=('m'), linewidths=1.5);
    #ax3.clabel(CONTORNO, CONTORNO.levels, inline=True, fmt=fmt, fontsize=10)
    plt.plot(np.nan, np.nan, '-m', label='PCT89 200 K ')
    plt.legend(loc=1)
    
    p1 = ax1.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[0], 0.2, p2[2]-p1[0], 0.05])
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=0.6,ticks=np.arange(50,300,10), extend='both', orientation="horizontal", label='TBV (K)')   

    return fig, CONTORNO3

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#################################################################
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
#------------------------------------------------------------------------------
def get_contour_verts(cp):
    contours=[]
    for cc in cp.collections:
        paths=[]
        for pp in cc.get_paths():
            xy=[]
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

  plt.matplotlib.rc('font', family='serif', size = 12)
  plt.rcParams['xtick.labelsize']=12
  plt.rcParams['ytick.labelsize']=12  
  
  #- EDIT THI:  
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/'
  #---------
  era5_dir       = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
  main_radar_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
  # Casos seleccionados
  radar_RMAXs = ['cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc', # RMA5:0.7 dBZ [0]
                'cfrad.20171027_034647.0000_to_20171027_034841.0000_RMA1_0123_02.nc',  # RMA1:0 dBZ   [1]
                'cfrad.20171209_005909.0000_to_20171209_010438.0000_RMA1_0123_01.nc',  # RMA1:4 dBZ   [2]
                'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc',  # RMA1:4 dBZ   [3]
                'cfrad.20180209_063643.0000_to_20180209_063908.0000_RMA1_0201_03.nc',  # RMA1:3 dBZ   [4]
                'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc',  # RMA1:1 dBZ   [5]
                'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc',  # RMA1:1 dBZ   [6]
                'cfrad.20190224_061413.0000_to_20190224_061537.0000_RMA1_0301_02.nc',  # RMA1:0 dBZ   [7]
                'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc']  # RMA1:0.5 dBZ [8]
  radar_dirs  = [main_radar_dir+'RMA5/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',
                 main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',
                main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/']
  diferencia_test = [280, 280, 280, 280, 280, 280, 280, 280, 280]
  ZDR_offset      = [0.7, 0, 4, 4, 3, 1, 1, 0, 0.5]
  RMA_name        = ['RMA5','RMA1','RMA1','RMA1','RMA1','RMA1','RMA1','RMA1','RMA1' ]
  toi             = [333]
  range_max       = [150]
  PFs_lat         = [-25.28]
  PFs_lon         = [-54.11]
  ERA5_files      = ['20200805_02_RMA5.grib']
    
    
  p_field    = np.flip([100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 
                  650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])*1e10 
  
    
    
  ojo con i=1, que hace algo raro pq syspahse=0 ? 
    
    
    
  # Loop over radar ... pero por ahora: i=0 
  i = 0

    
    
    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # Read radar file
    radar = pyart.io.read(radar_dirs[i]+radar_RMAXs[i])     
    #----- correct sysphase 
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.8, ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], 0, diferencia_test[i])
    plot_test_ppi(radar, corr_phidp, PFs_lat[i], PFs_lon[i])
    #----- correct attenuation (check calcualte_attenuation vs. calculate_attenuation_zphi   
    d = pyart.lazydict.LazyLoadDict({'data': corr_phidp})
    radar.add_field('PHIDP_c', d, replace_existing=False)
    # a y b regional?? 
    if 'TH' in radar.fields.keys():  
        spec_at, ZHCORR = pyart.correct.calculate_attenuation(radar, z_offset=0, rhv_min=0.6, ncp_min=0.6, a_coef=0.08, beta=0.64884, refl_field='TH', 
                                                        ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP_c')
    elif 'DBZH' in radar.fields.keys():
        spec_at, ZHCORR = pyart.correct.calculate_attenuation(radar, z_offset=0, rhv_min=0.6, ncp_min=0.6, a_coef=0.08, beta=0.64884, refl_field='DBZH', 
                                                        ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP_c')
    radar.add_field('ZHCORR', ZHCORR, replace_existing=True)
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----
    # ---- PARA USAR calculate_attenuation_zphi me falta algun sondeo ... t_field ... closest to RMA5?
    # Find j,k where meets PF lat/lons
    ERA5_field = xr.load_dataset(era5_dir+ERA5_files[i], engine="cfgrib")
    elemj      = find_nearest(ERA5_field['latitude'], PFs_lat[i])
    elemk      = find_nearest(ERA5_field['longitude'], PFs_lon[i])
    tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
    geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
    # Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
    Re         = 6371*1e3
    alt_ref    = (Re*geoph_ref)/(Re-geoph_ref)
    # Print the freezing level
    print('Freezing level at ', np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3), ' km')
    freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3) 
    radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
    radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')  
    #- Add height field for 4/3 propagation
    radar_height = get_z_from_radar(radar)
    radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')    
    iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
    radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
    radar.fields['height_over_iso0']['data'] -= iso0   
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----
    plot_rhi_RMA(radar_RMAXs[i], fig_dir, radar_dirs[i], RMA_name[i], 0, range_max[0], toi[i], ZDR_offset[i], freezing_lev)
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----
    # Apply zphi correction for ZDR attenuation correction?
    gatefilter_vitto = pyart.correct.GateFilter(radar)
    gatefilter_vitto.exclude_below('RHOHV', 0.8, exclude_masked=True)
    if 'TH' in radar.fields.keys(): 
        refl_field = 'TH'
    elif 'DBZH' in radar.fields.keys(): 
        refl_field = 'DBZH'        
    if 'ZDR' in radar.fields.keys(): 
        zdr_field = 'ZDR'
    # Note that pyart defaults for C-band are param_att_dict.update({'C': (a=0.08, b=0.64884, c=0.3, d=1.0804)}), 
    # where c, d : coeff. and exponent of the power law that relates Ah w/ differential attenuation
    # while following Snyder et al., (2010)                               (     ,        ,4.38,1.224) ???? 
    spec_at, pia_dict, cor_z, spec_diff_at, pida_dict, cor_zdr = pyart.correct.calculate_attenuation_zphi(
        radar, zdr_field=zdr_field, refl_field = refl_field, phidp_field = 'PHIDP_c',   # ojo corrected PHIDP
            c=0.3, d=1.0804, temp_ref='height_over_iso0', a_coef=0.08, beta=0.64884,
            temp_field='sounding_temperature', gatefilter=gatefilter_vitto)
    radar.add_field('dBZ_correc_ZPHI', cor_z, replace_existing=True)
    radar.add_field('ZDR_correc_ZPHI', cor_zdr, replace_existing=True)
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    #plot_corrections_ppi(radar, corr_phidp, ZHCORR['data'], spec_at['data'], ZDR_offset[i], PFs_lat[i], PFs_lon[i])
    # FOR NOW IGNORE ZDR_CORRECTION? ZPHI NEEDS TUNING!!
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ---- 
    # Finally, check out simple ZDR correction algorithm implemented above with alpha ZPHI coeff=0.016 ??? 
    #zdr_cor_simple = correct_attenuation_zdr(radar, gatefilter_vitto, 'ZDR', 'PHIDP_c', 0.02)      # ojo corrected PHIDP
    #radar.add_field('ZDR_correc_ZPHI', zdr_cor_simple, replace_existing=True)
    #plot_corrections_ppi(radar, corr_phidp, ZHCORR['data'], spec_at['data'], ZDR_offset[i], PFs_lat[i], PFs_lon[i])

    
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----    
    # hid adicionalmente   
    scores          = csu_fhc.csu_fhc_summer(dz=radar.fields['ZHCORR']['data'], zdr=radar.fields[zdr_field]['data'], 
                                             rho=radar.fields['RHOHV']['data'], kdp=radar.fields['KDP']['data'], 
                                             use_temp=True, band='C', T=radar_T)
    HID             = np.argmax(scores, axis=0) + 1
    radar = add_field_to_radar_object(HID, radar, field_name = 'HID')    
    plot_corrections_ppi_wHID(radar, corr_phidp, ZHCORR['data'], spec_at['data'], ZDR_offset[i], PFs_lat[i], PFs_lon[i])

    FIX! con rhohv lim gatefilter!
    
    

    
    
    
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----    
    # --- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----- ---- ---- ----    
    # CASO SUPERCELDA:
    lon_pfs = [-64.80]
    lat_pfs = [-31.83]
    phail   = [0.534]
    MIN85PCT = [131.1081]
    MIN37PCT = [207.4052]
    #
    rfile = 'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc'
    gfile = '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5'
    opts = {'xlim_min': -66, 'xlim_max': -62, 'ylim_min': -33, 'ylim_max': -30}
    fig, contorno89 = plot_gmi('/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, opts,
                                  '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'+'RMA1/', rfile, lon_pfs, lat_pfs)
    plt.title('GMI'+gfile[18:26]+' Phail='+str(phail[0])+' MIN85PCT: '+str(MIN85PCT[0])+' MIN37PCT:'+str(MIN37PCT[0]))
    # Inside radar PCTs (que en principio son PFs). look at TB distribution w/ MIN85PCT and MIN37PCT.  
    # i.e., the 200 K lines ... onli 1 collection! PERO OJO PORQUE USE UN LAT/LON MAX/MIN muy laxo. 
    for item in contorno89.collections:
        counter=0
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1]
            # Keep only x,y within ops min/max values of interest 
            if np.min(x) < opts['xlim_min']:
                continue
            elif np.max(x) > opts['xlim_max']:
                continue            
            elif np.min(y) < opts['ylim_min']:
                continue
            elif np.max(y) > opts['ylim_max']:
                continue    
            else:
                plt.plot(x,y, label=str(counter))
            counter=counter+1
    plt.legend()
    # So, interested in paths: 1, 2, 3
    # Get vertices of these polygon type shapes
    X1 = []; Y1 = []; vertices = []
    for i in range(len(contorno89.collections[0].get_paths()[1].vertices)): 
        X1.append(contorno89.collections[0].get_paths()[1].vertices[i][0])
        Y1.append(contorno89.collections[0].get_paths()[1].vertices[i][1])
        vertices.append([contorno89.collections[0].get_paths()[1].vertices[i][0], 
                                        contorno89.collections[0].get_paths()[1].vertices[i][1]])
   # Run ConvexHull
   # read file
   f = h5py.File( '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'+gfile, 'r')
   tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
   lon_gmi = f[u'/S1/Longitude'][:,:] 
   lat_gmi = f[u'/S1/Latitude'][:,:]
   tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
   lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
   lat_s2_gmi = f[u'/S2/Latitude'][:,:]
   f.close()
    
    
    
   from scipy.spatial import ConvexHull, convex_hull_plot_2d
   from matplotlib.path import Path

   convexhull = ConvexHull(vertices)
   array_points = np.array(vertices)

   hull_path   = Path( array_points[convexhull.vertices] )
   inside_s1   = np.logical_and(np.logical_and(lon_gmi >= opts['xlim_min'], lon_gmi <=  opts['xlim_max']), 
                              np.logical_and(lat_gmi >= opts['ylim_min'], lat_gmi <= opts['ylim_max']))
   lon_gmi    = lon_gmi[inside_s1] 
   lat_gmi    = lat_gmi[inside_s1]
   tb_s1_gmi  = tb_s1_gmi[inside_s1,:]
   datapts = np.column_stack((lon_gmi,lat_gmi))
   inds = hull_path.contains_points(datapts)
   lon_gmi    = lon_gmi[inds] 
   lat_gmi    = lat_gmi[inds]
   tb_s1_gmi  = tb_s1_gmi[inds,:]
   

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'

    # Shapefiles for cartopy 
    geo_reg_shp = home_dir+'Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)

    countries = shpreader.Reader(home_dir+'Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black')
    rivers = cfeature.NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        facecolor='none',
        edgecolor='black')

    cmaps = GMI_colormap() 
        
   
    fig = plt.figure(figsize=(12,7)) 
    gs1 = gridspec.GridSpec(1, 3)
    
    # BT(37)       
    ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
    crs_latlon = ccrs.PlateCarree()
    ax1.set_extent([opts['xlim_min'], opts['xlim_max'], 
                    opts['ylim_min'], opts['ylim_max']], crs=crs_latlon)
    ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                edgecolor="black", facecolor='none')
    ax1.add_feature(states_provinces,linewidth=0.4)
    ax1.add_feature(rivers)
    ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                edgecolor="black", facecolor='none')
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=tb_s1_gmi[:,5], s=20, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz', fontsize=14)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(opts['ylim_min'], opts['ylim_max']+1,1), crs=crs_latlon)
    ax1.set_xticks(np.arange(opts['xlim_min'], opts['xlim_max']+1,1), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    for simplex in convexhull.simplices:
        plt.plot(array_points[simplex, 0], array_points[simplex, 1], 'c')
        





