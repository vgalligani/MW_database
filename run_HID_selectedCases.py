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
import wradlib as wrl
import pandas as pd
from pyart.correct import phase_proc
import xarray as xr
from pyart.core.transforms import antenna_to_cartesian
from copy import deepcopy
from csu_radartools import csu_fhc
import matplotlib.colors as colors


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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


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
    
    
    
    
    
    
    
    
    
    
    
    
    
  fig = plt.figure()
  start_index = radar.sweep_start_ray_index['data'][0]
  end_index   = radar.sweep_end_ray_index['data'][0]
  lats  = radar.gate_latitude['data'][start_index:end_index]
  lons  = radar.gate_longitude['data'][start_index:end_index]
  phidp = radar.fields['PHIDP']['data'][start_index:end_index]
  rhohv = radar.fields['RHOHV']['data'][start_index:end_index]
  zdr2  = radar.fields['ZDR_correc_ZPHI']['data'][start_index:end_index]
  zdr  = radar.fields['ZDR']['data'][start_index:end_index]
  zh  = radar.fields['DBZH']['data'][start_index:end_index]
  zh2  = radar.fields['dBZ_correc_ZPHI']['data'][start_index:end_index]
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
  pcm1 = plt.pcolormesh(lons, lats, zdr2)
  plt.colorbar()

    
    






