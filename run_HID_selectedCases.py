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
        vmin = 60
        vmax = 180
        max = 180.1
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
