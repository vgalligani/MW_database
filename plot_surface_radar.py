#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:26:28 2022
@author: victoria.galligani
"""
# Code that plots initially Zh for the coincident RMA5, RMA1 and PARANA observations for
# case studies of interest. ver .doc "casos granizo" para más info. 

from os import listdir
import pyart
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from shapely.geometry import Polygon


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
        cmap = discrete_cmap(int(N), 'jet') # colormaps('zdr')
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
        vmin = -15
        vmax = 15
        max = 15.01
        intt = 5
        under = 'white'
        over = 'white'    
        
    return units, cmap, vmin, vmax, max, intt, under, over

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi(file, fig_dir, dat_dir, radar_name): 
    
    radar = pyart.io.read(dat_dir+file) 
    # dict_keys(['PHIDP', 'CM', 'RHOHV', 'TH', 'TV', 'KDP']) for RMA1
    # dict_keys(['DBZH', 'KDP', 'RHOHV', 'PHIDP', 'CM']) for RMA5
    # dict_keys(['DBZV', 'DBZH', 'ZDR', 'KDP', 'RHOHV', 'PHIDP', 'VRAD']) for SOME RMA5
    print(radar.fields.keys())
    
    #- Radar sweep
    nelev       = 0
    start_index = radar.sweep_start_ray_index['data'][nelev]
    end_index   = radar.sweep_end_ray_index['data'][nelev]
    lats        = radar.gate_latitude['data'][start_index:end_index]
    lons        = radar.gate_longitude['data'][start_index:end_index]
    azimuths    = radar.azimuth['data'][start_index:end_index]
    
    PHIDP    = radar.fields['PHIDP']['data'][start_index:end_index]
    #CM       = radar.fields['CM']['data'][start_index:end_index]
    RHOHV    = radar.fields['RHOHV']['data'][start_index:end_index]
    if radar_name == 'RMA1':
        TH       = radar.fields['TH']['data'][start_index:end_index]
        TV       = radar.fields['TV']['data'][start_index:end_index]
        ZDR      = TH-TV
    elif radar_name == 'RMA5':
        TH       = radar.fields['DBZH']['data'][start_index:end_index]
        if 'VRAD' in radar.fields.keys(): 
            VRAD       = radar.fields['VRAD']['data'][start_index:end_index]
        if 'DBZV' in radar.fields.keys(): 
            TV       = radar.fields['DBZV']['data'][start_index:end_index]     
            ZDR      = TH-TV
    elif radar_name == 'PARANA':
        print('?')
    
    # plot figure: 
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True,
                        figsize=[14,12])
    #-- Zh: 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)     
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
    axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    axes[0,0].grid()   
    #-- ZDR: 
    if 'ZDR' in locals(): 
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
        pcm1 = axes[0,1].pcolormesh(lons, lats, ZDR, cmap=cmap, vmin=vmin, vmax=max)
        cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        cbar.cmap.set_over(over)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
        axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)     
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
        axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
        axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
        axes[0,1].grid()         
    #-- PHIDP: 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
    pcm1 = axes[1,0].pcolormesh(lons, lats, PHIDP, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)     
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
    axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    axes[1,0].grid()      
    #-- RHOHV: 
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    pcm1 = axes[1,1].pcolormesh(lons, lats, RHOHV, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
    cbar.cmap.set_under(under)
    cbar.cmap.set_over(over)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)     
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
    axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)    
    axes[1,1].set_xlabel('Longitude', fontsize=10)
    axes[1,1].set_ylabel('Latitude', fontsize=10)
    axes[1,1].grid()     
    #- savefile
    plt.suptitle(radar_name+': ncfile '+str(file[6:17]),fontweight='bold')
    fig.savefig(fig_dir+str(file)+'.png', dpi=300,transparent=False)    
    plt.close() 
    
    
    return 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi_parana_all(file, fig_dir, dat_dir, radar_name):

    radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'dBZ.vol')
    if exists(dat_dir+file+'ZDR.vol'): 
        radarZDR = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'ZDR.vol')
    
    if exists(dat_dir+file+'W.vol'): 
        radarW = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'W.vol')         #['spectrum_width']

    if exists(dat_dir+file+'V.vol'): 
        radarV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'V.vol')         #['velocity']

    if exists(dat_dir+file+'uPhiDP.vol'): 
        radaruPHIDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'uPhiDP.vol')    #['uncorrected_differential_phase']

    if exists(dat_dir+file+'RhoHV.vol'):  
        radarRHOHV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')

    if exists(dat_dir+file+'PhiDP.vol'):  
        radarPHIDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'PhiDP.vol') #['differential_phase'] 
    
    if exists(dat_dir+file+'KDP.vol'):  
        radarKDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'KDP.vol')

    
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        
        # Figure
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True,
                        figsize=[14,12])
        #-- Zh: 
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats = radar.gate_latitude['data'][start_index:end_index]
        lons = radar.gate_longitude['data'][start_index:end_index]
        azimuths = radar.azimuth['data'][start_index:end_index]
        elevation = radar.elevation['data'][start_index]
        TH = radar.fields['reflectivity']['data'][start_index:end_index]
        pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        cbar.cmap.set_over(over)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        axes[0,0].grid()

        #-- ZDR: 
        if exists(dat_dir+file+'ZDR.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
            start_index = radarZDR.sweep_start_ray_index['data'][nlev]
            end_index   = radarZDR.sweep_end_ray_index['data'][nlev]
            lats = radarZDR.gate_latitude['data'][start_index:end_index]
            lons = radarZDR.gate_longitude['data'][start_index:end_index]
            azimuths = radarZDR.azimuth['data'][start_index:end_index]
            elevation = radarZDR.elevation['data'][start_index]
            ZDR = radarZDR.fields['differential_reflectivity']['data'][start_index:end_index]
            pcm1 = axes[0,1].pcolormesh(lons, lats, ZDR, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[0,1].grid()       
        
         #-- PHIDP: 
        if exists(dat_dir+file+'uPhiDP.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
            start_index = radaruPHIDP.sweep_start_ray_index['data'][nlev]
            end_index   = radaruPHIDP.sweep_end_ray_index['data'][nlev]
            lats = radaruPHIDP.gate_latitude['data'][start_index:end_index]
            lons = radaruPHIDP.gate_longitude['data'][start_index:end_index]
            azimuths = radaruPHIDP.azimuth['data'][start_index:end_index]
            elevation = radaruPHIDP.elevation['data'][start_index]
            PHIDP = radaruPHIDP.fields['uncorrected_differential_phase']['data'][start_index:end_index]
            pcm1 = axes[1,0].pcolormesh(lons, lats, PHIDP, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[1,0].grid()
            axes[1,0].set_xlabel('Longitude', fontsize=10)
            axes[1,0].set_ylabel('Latitude', fontsize=10)
            axes[1,0].grid()       
               
         #-- RHOHV: 
        if exists(dat_dir+file+'RhoHV.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
            start_index = radarRHOHV.sweep_start_ray_index['data'][nlev]
            end_index   = radarRHOHV.sweep_end_ray_index['data'][nlev]
            lats = radarRHOHV.gate_latitude['data'][start_index:end_index]
            lons = radarRHOHV.gate_longitude['data'][start_index:end_index]
            azimuths = radarRHOHV.azimuth['data'][start_index:end_index]
            elevation = radarRHOHV.elevation['data'][start_index]
            RHOHV = radarRHOHV.fields['cross_correlation_ratio']['data'][start_index:end_index]
            pcm1 = axes[1,1].pcolormesh(lons, lats, RHOHV, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[1,1].grid()
            axes[1,1].set_xlabel('Longitude', fontsize=10)
            axes[1,1].set_ylabel('Latitude', fontsize=10)
        
        #- savefile
        plt.suptitle(radar_name+' ('+str(elevation)+'): '+str(file[0:12]),fontweight='bold')
        fig.savefig(fig_dir+'FULL_'+str(file)+'sweep_'+str(nlev)+'.png', dpi=300,transparent=False)
        plt.close()

    return
 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi_parana_all_zoom(file, fig_dir, dat_dir, radar_name, xlims, ylims):

    radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'dBZ.vol')
    if exists(dat_dir+file+'ZDR.vol'): 
        radarZDR = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'ZDR.vol')
    
    if exists(dat_dir+file+'W.vol'): 
        radarW = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'W.vol')         #['spectrum_width']

    if exists(dat_dir+file+'V.vol'): 
        radarV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'V.vol')         #['velocity']

    if exists(dat_dir+file+'uPhiDP.vol'): 
        radaruPHIDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'uPhiDP.vol')    #['uncorrected_differential_phase']

    if exists(dat_dir+file+'RhoHV.vol'):  
        radarRHOHV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')

    if exists(dat_dir+file+'PhiDP.vol'):  
        radarPHIDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'PhiDP.vol') #['differential_phase'] 
    
    if exists(dat_dir+file+'KDP.vol'):  
        radarKDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'KDP.vol')

    
    for nlev in range(3): #len(radar.sweep_start_ray_index['data'])):   PLOT ONLY FIRST 3
        
        # Figure
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True,
                        figsize=[14,12])
        #-- Zh: 
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats = radar.gate_latitude['data'][start_index:end_index]
        lons = radar.gate_longitude['data'][start_index:end_index]
        azimuths = radar.azimuth['data'][start_index:end_index]
        elevation = radar.elevation['data'][start_index]
        TH = radar.fields['reflectivity']['data'][start_index:end_index]
        pcm1 = axes[0,0].pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm1, ax=axes[0,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        cbar.cmap.set_over(over)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
        axes[0,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        axes[0,0].grid()
        axes[0,0].set_xlim(xlims)
        axes[0,0].set_ylim(ylims)
        
        #-- ZDR: 
        if exists(dat_dir+file+'ZDR.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
            start_index = radarZDR.sweep_start_ray_index['data'][nlev]
            end_index   = radarZDR.sweep_end_ray_index['data'][nlev]
            lats = radarZDR.gate_latitude['data'][start_index:end_index]
            lons = radarZDR.gate_longitude['data'][start_index:end_index]
            azimuths = radarZDR.azimuth['data'][start_index:end_index]
            elevation = radarZDR.elevation['data'][start_index]
            ZDR = radarZDR.fields['differential_reflectivity']['data'][start_index:end_index]
            pcm1 = axes[0,1].pcolormesh(lons, lats, ZDR, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[0,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[0,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[0,1].grid()  
            axes[0,1].set_xlim(xlims)
            axes[0,1].set_ylim(ylims)
        
         #-- PHIDP: 
        if exists(dat_dir+file+'uPhiDP.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('phidp')
            start_index = radaruPHIDP.sweep_start_ray_index['data'][nlev]
            end_index   = radaruPHIDP.sweep_end_ray_index['data'][nlev]
            lats = radaruPHIDP.gate_latitude['data'][start_index:end_index]
            lons = radaruPHIDP.gate_longitude['data'][start_index:end_index]
            azimuths = radaruPHIDP.azimuth['data'][start_index:end_index]
            elevation = radaruPHIDP.elevation['data'][start_index]
            PHIDP = radaruPHIDP.fields['uncorrected_differential_phase']['data'][start_index:end_index]
            pcm1 = axes[1,0].pcolormesh(lons, lats, PHIDP, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[1,0], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[1,0].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[1,0].grid()
            axes[1,0].set_xlabel('Longitude', fontsize=10)
            axes[1,0].set_ylabel('Latitude', fontsize=10)
            axes[1,0].grid()   
            axes[1,0].set_xlim(xlims)
            axes[1,0].set_ylim(ylims)
                               
         #-- KDP:   [RHOHV empty?] 
        if exists(dat_dir+file+'KDP.vol'): 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')  #Kdp
            start_index = radarRHOHV.sweep_start_ray_index['data'][nlev]   # radarKDP
            end_index   = radarRHOHV.sweep_end_ray_index['data'][nlev]
            lats = radarRHOHV.gate_latitude['data'][start_index:end_index]
            lons = radarRHOHV.gate_longitude['data'][start_index:end_index]
            azimuths = radarRHOHV.azimuth['data'][start_index:end_index]
            elevation = radarRHOHV.elevation['data'][start_index]
            RHOHV = radarRHOHV.fields['cross_correlation_ratio']['data'][start_index:end_index]
            pcm1 = axes[1,1].pcolormesh(lons, lats, RHOHV, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes[1,1], shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes[1,1].plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes[1,1].grid()
            axes[1,1].set_xlabel('Longitude', fontsize=10)
            axes[1,1].set_ylabel('Latitude', fontsize=10)
            axes[1,1].set_xlim(xlims)
            axes[1,1].set_ylim(ylims)
        
        #- savefile
        plt.suptitle(radar_name+' ('+str(elevation)+'): '+str(file[0:12]),fontweight='bold')
        fig.savefig(fig_dir+'FULL_ZOOM'+str(file)+'sweep_'+str(nlev)+'.png', dpi=300,transparent=False)
        #plt.close()

    return
 
    
      
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_pseudo_RHI_parana(file, fig_dir, dat_dir, radar_name, test_transect):

    
    radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'dBZ.vol')
    start_index = radar.sweep_start_ray_index['data'][0]
    end_index   = radar.sweep_end_ray_index['data'][0]
    lats0 = radar.gate_latitude['data'][start_index:end_index]
    lons0 = radar.gate_longitude['data'][start_index:end_index]
    
    if exists(dat_dir+file+'ZDR.vol'): 
        radarZDR = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'ZDR.vol')
    if exists(dat_dir+file+'uPhiDP.vol'): 
        radaruPHIDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'uPhiDP.vol')    #['uncorrected_differential_phase']
    if exists(dat_dir+file+'RhoHV.vol'):  
        radarRHOHV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')
    if exists(dat_dir+file+'KDP.vol'):  
        radarKDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'KDP.vol')
       
    Ze_transect     = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); Ze_transect[:]=np.nan
    ZDR_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); ZDR_transect[:]=np.nan
    PHI_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); PHI_transect[:]=np.nan
    lon_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lon_transect[:]=np.nan
    lat_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); lat_transect[:]=np.nan
    RHO_transect    = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); RHO_transect[:]=np.nan
    approx_altitude = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); approx_altitude[:]=np.nan
    color           = np.full((  len(radar.sweep_start_ray_index['data']),  lats0.shape[1], 4), np.nan)
    gate_range      = np.zeros( [len(radar.sweep_start_ray_index['data']), lats0.shape[1] ]); gate_range[:]=np.nan

    azydims = lats0.shape[1]-1

    for nlev in range(len(radar.sweep_start_ray_index['data'])):
        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]       
        ZHZH     = radar.fields['reflectivity']['data'][start_index:end_index]
        azimuths = radar.azimuth['data'][start_index:end_index]
        lats     = radar.gate_latitude['data'][start_index:end_index]
        lons     = radar.gate_longitude['data'][start_index:end_index]
        gateZ    = radar.gate_z['data'][start_index:end_index]
        gateX    = radar.gate_x['data'][start_index:end_index]
        gateY    = radar.gate_y['data'][start_index:end_index]
        gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        #
        start_index = radarZDR.sweep_start_ray_index['data'][nlev]
        end_index   = radarZDR.sweep_end_ray_index['data'][nlev]       
        ZDRZDR      = radarZDR.fields['differential_reflectivity']['data'][start_index:end_index]
        #
        start_index = radaruPHIDP.sweep_start_ray_index['data'][nlev]
        end_index   = radaruPHIDP.sweep_end_ray_index['data'][nlev]       
        PHIDPPHIDP  = radaruPHIDP.fields['uncorrected_differential_phase']['data'][start_index:end_index]        
        #
        target_azimuth = azimuths[test_transect]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        #
        lon_transect[nlev,:]     = lons[filas,:]
        lat_transect[nlev,:]     = lats[filas,:]
        Ze_transect[nlev,:]      = ZHZH[filas,:]
        ZDR_transect[nlev,:]     = ZDRZDR[filas,:]
        PHI_transect[nlev,:]     = PHIDPPHIDP[filas,:]
        #RHO_transect[nlev,:]     = rh[nlev,filas,:]
        # 
        [xgate, ygate, zgate]   = pyart.core.antenna_to_cartesian(gates_range[filas,:]/1e3, azimuths[filas],radar.get_elevation(0)[0]);
        approx_altitude[nlev,:] = zgate/1e3
        gate_range[nlev,:]      = gates_range[filas,:]/1e3;

                
    #- Simple pcolormesh plot! 
    fig = plt.figure(figsize=[15,11])
    fig.add_subplot(111)
    mycolorbar = plt.pcolormesh(lon_transect, approx_altitude, Ze_transect)
        #         cmap=colormaps('ref'), vmin=0, vmax=60)

    #- De esta manera me guardo el color con el que rellenar los polygons (scatter plot para sacar el color de cada pixel)
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
         fig = plt.figure(figsize=[30,10])
         fig.add_subplot(221)
         sc = plt.scatter(lon_transect[nlev,:], approx_altitude[nlev,:],
                 s=1,c=Ze_transect[nlev,:],
                 cmap=colormaps('ref'), vmin=0, vmax=60)
         color[nlev,:,:] = sc.to_rgba(Ze_transect[nlev,:])
         plt.close()

    #- Try polygons
    fig2, axes = plt.subplots(nrows=4,ncols=3,constrained_layout=True,figsize=[8,6])  # 8,4 muy chiquito
    fig1 = plt.figure(figsize=(15,20))
    for nlev in range(len(radar.sweep_start_ray_index['data'])):
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
             axes[0,0].fill(x, y, color = color[nlev,i,:], )
             x, y = P1.exterior.xy


    return 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi_parana(file, fig_dir, dat_dir, radar_name):

    radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file)

    for nlev in range(len(radar.sweep_start_ray_index['data'])):

        start_index = radar.sweep_start_ray_index['data'][nlev]
        end_index   = radar.sweep_end_ray_index['data'][nlev]
        lats = radar.gate_latitude['data'][start_index:end_index]
        lons = radar.gate_longitude['data'][start_index:end_index]
        azimuths = radar.azimuth['data'][start_index:end_index]
        elevation = radar.elevation['data'][start_index]

        TH = radar.fields['reflectivity']['data'][start_index:end_index]
        print(radar.fields.keys())
        # Figure
        fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[14,12])
        #-- Zh: 
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        pcm1 = axes.pcolormesh(lons, lats, TH, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
        cbar.cmap.set_under(under)
        cbar.cmap.set_over(over)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
        axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        axes.grid()
        axes.set_xlabel('Longitude', fontsize=10)
        axes.set_ylabel('Latitude', fontsize=10)
        axes.grid()
        #- savefile
        plt.suptitle(radar_name+' ('+str(elevation)+'): '+str(file[0:12]),fontweight='bold')
        fig.savefig(fig_dir+str(file)+'sweep_'+str(nlev)+'.png', dpi=300,transparent=False)
        plt.close()

        del TH


    return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_ppi_parana_doppler(file, fig_dir, dat_dir, radar_name, xlims, ylims):

    if exists(dat_dir+file+'V.vol'): 
        radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'V.vol')#['velocity']
        radarTH = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'dBZ.vol')#['velocity']

        
        for nlev in range(3): # len(radar.sweep_start_ray_index['data'])):
            start_index = radar.sweep_start_ray_index['data'][nlev]
            end_index   = radar.sweep_end_ray_index['data'][nlev]
            lats = radar.gate_latitude['data'][start_index:end_index]
            lons = radar.gate_longitude['data'][start_index:end_index]
            azimuths = radar.azimuth['data'][start_index:end_index]
            elevation = radar.elevation['data'][start_index]
            # For Zh:
            THstart_index = radarTH.sweep_start_ray_index['data'][nlev]
            THend_index   = radarTH.sweep_end_ray_index['data'][nlev]
            THlats = radarTH.gate_latitude['data'][THstart_index:THend_index]
            THlons = radarTH.gate_longitude['data'][THstart_index:THend_index]
            # Get countours above 20 dBZ
            TH = radarTH.fields['reflectivity']['data'][start_index:end_index]            
            # Doppler velocity
            VEL = radar.fields['velocity']['data'][start_index:end_index]
            print(radar.fields.keys())
            print(radar.instrument_parameters.keys())
            # Correct doppler velo. 
            #vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='velocity', 
            #                                                        nyq=np.min(radar.instrument_parameters['nyquist_velocity']['data']))
            #radar.add_field('velocity_texture', vel_texture, replace_existing=True)
            #gatefilter  = pyart.filters.GateFilter(radar)
            #gatefilter.exclude_above('velocity_texture', 5)
            #nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
            #velocity_dealiased = pyart.correct.dealias_region_based(radar, vel_field='velocity', nyquist_vel=nyq,
            #                                            centered=True, gatefilter=gatefilter)
            #radar.add_field('corrected_velocity', velocity_dealiased, replace_existing=True)
            #VEL_cor   = radar.fields['corrected_velocity']['data'][start_index:end_index]
            
            # Figure
            fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[20,12])  # 14,12
            #-- Doppler: 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('doppler')
            pcm1 = axes.pcolormesh(lons, lats, VEL, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
            cbar.cmap.set_under(under)
            cbar.cmap.set_over(over)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
            axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
            axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],150)
            axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],200)
            axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],250)
            axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
            axes.grid()
            axes.set_xlim(xlims)
            axes.set_ylim(ylims)
            axes.set_xlabel('Longitude', fontsize=10)
            axes.set_ylabel('Latitude', fontsize=10)
            axes.grid()
            plt.contour(THlons, THlats, TH,  [30], colors=('k'), linewidths=1.5);
            #- savefile
            plt.suptitle(radar_name+' ('+str(elevation)+'): '+str(file[0:12]),fontweight='bold')
            fig.savefig(fig_dir+str(file)+'sweep_'+str(nlev)+'_DOPPLER.png', dpi=300,transparent=False)
            #plt.close()

            del VEL
    return

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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  

if __name__ == '__main__':

  plt.matplotlib.rc('font', family='serif', size = 12)
  plt.rcParams['xtick.labelsize']=12
  plt.rcParams['ytick.labelsize']=12  
    
  files_RMA5 = ['cfrad.20181001_231430.0000_to_20181001_232018.0000_RMA5_0200_01.nc',
                'cfrad.20181001_231430.0000_to_20181001_232018.0000_RMA5_0200_01.nc',
                'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc']

  files_RMA1 = ['cfrad.20171027_034647.0000_to_20171027_034841.0000_RMA1_0123_02.nc',
              'cfrad.20171209_005909.0000_to_20171209_010438.0000_RMA1_0123_01.nc',
              'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc',
              'cfrad.20180209_063643.0000_to_20180209_063908.0000_RMA1_0201_03.nc',
              'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc', 
              'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc',
              'cfrad.20190102_204413.0000_to_20190102_204403.0000_RMA1_0301_02.nc',
              'cfrad.20190224_061413.0000_to_20190224_061537.0000_RMA1_0301_02.nc',
              'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc']

  # Files below organized: per line. tested for differente minutes. each line is a case study
  files_PAR = ['2018032407543300dBZ.vol', '2018032407500500dBZ.vol',
             '2018100907443400dBZ.vol',   '2018100907400500dBZ.vol', '2018100907343300dBZ.vol', '2018100907300500dBZ.vol',
             '2018121403043200dBZ.vol',   '2018121402400200dBZ.vol', 
             '2019021109400500dBZ.vol',
             '2019022315143100dBZ.vol',   '2019022315100200dBZ.vol',
             '2020121903100500dBZ.vol']
    
  # the ones finally selected are:



  # start w/ RMA1
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/RMA1/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA1/'
  for ifiles in range(len(files_RMA1)):
    folder = str(files_RMA1[ifiles][6:14])
    if folder[0:4] == '2021':
      ncfile  = '/relampago/datos/salio/RADAR/RMA1/'+ folder + '/' + files_RMA1[ifiles]
    else:
      yearfolder = folder[0:4]
      ncfile  = '/relampago/datos/salio/RADAR/RMA1/'+ yearfolder + '/' + folder + '/' + files_RMA1[ifiles]
    print('original file in: ' + ncfile + '// reading in:'+dat_dir+files_RMA1[ifiles])
    plot_ppi(files_RMA1[ifiles], fig_dir, dat_dir, 'RMA1')
  #--------------------------------------------------------------------------------------------
  # start w/ RMA5
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/RMA5/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA5/'
  for ifiles in range(len(files_RMA5)):
    folder = str(files_RMA5[ifiles][6:14])
    if folder[0:4] == '2021':
      ncfile  = '/relampago/datos/salio/RADAR/RMA5/'+ folder + '/' + files_RMA5[ifiles]
    else:
      yearfolder = folder[0:4]
      ncfile  = '/relampago/datos/salio/RADAR/RMA5/'+ yearfolder + '/' + folder + '/' + files_RMA5[ifiles]
    print('original file in: ' + ncfile + '// reading in:'+dat_dir+files_RMA5[ifiles])
    plot_ppi(files_RMA5[ifiles], fig_dir, dat_dir, 'RMA5')
  #--------------------------------------------------------------------------------------------
  # start w/ PARANA
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/PAR/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/PAR/'
  for ifiles in range(len(files_PAR)):
    folder = str(files_PAR[ifiles][0:8])
    if folder[0:4] == '2021':
      ncfile  = '/relampago/datos/salio/RADAR/PAR/'+ folder + '/vol/' + files_PAR[ifiles]
    else:
      yearfolder = folder[0:4]
      ncfile  = '/relampago/datos/salio/RADAR/PAR/'+ yearfolder + '/' + folder + '/vol/' + files_PAR[ifiles]
    #print('cp ' + ncfile + ' '+dat_dir+'.')
    plot_ppi_parana(files_PAR[ifiles], fig_dir, dat_dir, 'PAR') 
  #--------------------------------------------------------------------------------------------
  # Plot all variables in PARANA
  #--------------------------------------------------------------------------------------------
  # 20180324_07 no hay polarimetrics 
  #file_PAR_all = '2018032407500500' #('2018032407543300dBZ.vol' tampoco tiene polarimetricos) 
  #folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5,-58], [-32,-29.5]) 
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5,-58], [-32,-29.5]) 
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2018100907400500'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-32,-30.5]) # figsize=[20,12])
  plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-32,-30.5]) 
  test_transect [160, 220]
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2018121402400200'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63.3,-58.5], [-34.4,-31.5]) 
  plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63.3,-58.5], [-34.4,-31.5])  
  test_transect = 320
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2019021109400500'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-34,-30]) 
  plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-34,-30]) 
  test_transect = 320
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2019022315100200'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5, -58], [-34,-31]) 
  plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5, -58], [-34,-31]) 
  test_transect = 150
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2020121903100500'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63,-58], [-34,-29.5]) 
  plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63,-58], [-34,-29.5]) 
  test_transect = 180
  plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect)



 CORREGIR Y TERMINAR RHISs. y ver GMI. ver colorbar de rhis


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def check_transec(dat_dir, file_PAR_all, test_transect):       
  radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file_PAR_all+'dBZ.vol')
  nlev  = 2  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12])
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  pcm1 = axes.pcolormesh(lons, lats, radar.fields['reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
  cbar = plt.colorbar(pcm1, ax=axes, shrink=1, label=units, ticks = np.arange(vmin,max,intt))
  cbar.cmap.set_under(under)
  cbar.cmap.set_over(over)
  [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
  axes.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
  axes.grid()
  azimuths = radar.azimuth['data'][start_index:end_index]
  target_azimuth = azimuths[test_transect]
  filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
  lon_transect     = lons[filas,:]
  lat_transect     = lats[filas,:]
  plt.plot(np.ravel(lon_transect), np.ravel(lat_transect), 'k')
  plt.title('Transecta Nr:'+ str(test_transect), Fontsize=20)
    
  return 
#--------------------------------------------------------------------------------------------    
   

    

    
  

