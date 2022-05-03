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
def plot_test_ppi(radar, phi_corr):
    
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
def plot_corrections_ppi(radar, phi_corr, ZHCORR, attenuation):
    
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats  = radar.gate_latitude['data'][start_index:end_index]
    lons  = radar.gate_longitude['data'][start_index:end_index]
    phidp = radar.fields['PHIDP']['data'][start_index:end_index]
    rhv   = radar.fields['RHOHV']['data'][start_index:end_index]
    dphi  = phi_corr[start_index:end_index]
    Zh_corr      = ZHCORR[start_index:end_index]
    att          = attenuation[start_index:end_index]
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
    #-- RHOHV
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[0,1].pcolormesh(lons, lats, rhv, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    axes[0,1].grid(True)
    #-- ZDR
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
    pcm1 = axes[0,2].pcolormesh(lons, lats, zdr, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,2], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
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
    #-- ATT
    pcm1 = axes[1,1].pcolormesh(lons, lats, att, vmin=0, vmax=3)
    cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label='attenuation')
    axes[1,1].grid(True)    
        
    return

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  

if __name__ == '__main__':

  plt.matplotlib.rc('font', family='serif', size = 12)
  plt.rcParams['xtick.labelsize']=12
  plt.rcParams['ytick.labelsize']=12  

  main_radar_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
  # Casos seleccionados
  radar_RMAXs = ['cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc', # RMA5:0.7 dBZ
                'cfrad.20171027_034647.0000_to_20171027_034841.0000_RMA1_0123_02.nc',  # RMA1:0 dBZ
                'cfrad.20171209_005909.0000_to_20171209_010438.0000_RMA1_0123_01.nc',  # RMA1:4 dBZ
                'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc',  # RMA1:4 dBZ
                'cfrad.20180209_063643.0000_to_20180209_063908.0000_RMA1_0201_03.nc',  # RMA1:3 dBZ
                'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc',  # RMA1:1 dBZ
                'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc',  # RMA1:1 dBZ
                'cfrad.20190224_061413.0000_to_20190224_061537.0000_RMA1_0301_02.nc',  # RMA1:0 dBZ
                'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc']  # RMA1:0.5 dBZ
  radar_dirs  = [main_radar_dir+'RMA5/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',
                 main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',
                main_radar_dir+'RMA1/',main_radar_dir+'RMA1/',main_radar_dir+'RMA1/']
  diferencia_test = [280, 280, 280, 280, 280, 280, 280, 280, 280]
  ZDR_offset      = [0.7, 0, 4, 4, 3, 1, 1, 0, 0.5]
    
    ojo con i=1, que hace algo raro pq syspahse=0 ? 
      
  # Loop over radar ... pero por ahora: i=0 
  i = 2

    # Read radar file
    radar = pyart.io.read(radar_dirs[i]+radar_RMAXs[i])     
    #----- correct sysphase 
    sys_phase  = pyart.correct.phase_proc.det_sys_phase(radar, ncp_lev=0.8, rhohv_lev=0.8, ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP')
    corr_phidp = correct_phidp(radar.fields['PHIDP']['data'], radar.fields['RHOHV']['data'], 0, diferencia_test[i])
    plot_test_ppi(radar, corr_phidp)
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
    plot_corrections_ppi(radar, corr_phidp, ZHCORR['data'], spec_at['data'])
    
    
    
  fig = plt.figure()
  start_index = radar.sweep_start_ray_index['data'][-1]
  end_index   = radar.sweep_end_ray_index['data'][-1]
  lats  = radar.gate_latitude['data'][start_index:end_index]
  lons  = radar.gate_longitude['data'][start_index:end_index]
  phidp = radar.fields['PHIDP']['data'][start_index:end_index]
  rhohv = radar.fields['RHOHV']['data'][start_index:end_index]
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
  pcm1 = plt.pcolormesh(lons, lats, phidp, cmap=cmap, vmin=vmin, vmax=vmax)
  dphi = despeckle_phidp(phidp, rhohv)

    
    
    
  # ---- PARA USAR calculate_attenuation_zphi me falta algun sondeo ... t_field ... for now ignore. 
  # Use defaults in pyart param_att_dict.update({'C': (0.08, 0.64884, 0.3, 1.0804)}) 
  # spec_at, _, ZHCORR_ZPHI, _, _, ZDRcorr = pyart.correct.calculate_attenuation_zphi(radar,a_coef=0.08, beta=0.64884, c=0.3, d=1.0804, refl_field='DBZH', zdr_field='ZDR',phidp_field='PHIDP_c')







