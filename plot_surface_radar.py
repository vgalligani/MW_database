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
        vmin = -10
        vmax = 10
        max = 10.01
        intt = 2
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
    if 'PHIDP' in radar.fields.keys():  
        PHIDP    = radar.fields['PHIDP']['data'][start_index:end_index]
    #CM       = radar.fields['CM']['data'][start_index:end_index]
    if 'RHOHV' in radar.fields.keys():  
        RHOHV    = radar.fields['RHOHV']['data'][start_index:end_index]
    if 'TH' in radar.fields.keys():  
        TH       = radar.fields['TH']['data'][start_index:end_index]
    if 'TV' in radar.fields.keys():  
        TV       = radar.fields['TV']['data'][start_index:end_index]
        ZDR      = TH-TV
    if 'DBZH' in radar.fields.keys():  
        TH       = radar.fields['DBZH']['data'][start_index:end_index]

    #if radar_name == 'RMA1':
    #    TH       = radar.fields['TH']['data'][start_index:end_index]
    #    TV       = radar.fields['TV']['data'][start_index:end_index]
    #    
    #if radar_name == 'RMA4':
    #    TH       = radar.fields['TH']['data'][start_index:end_index]
    #    TV       = radar.fields['TV']['data'][start_index:end_index]
    #    ZDR      = TH-TV
    if radar_name == 'RMA5':
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
    if 'PHIDP' in locals():
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
    if 'RHOHV' in locals():
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
    #plt.close() 
    
    
    return 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_rhi_RMA(file, fig_dir, dat_dir, radar_name, xlim_range1, xlim_range2, test_transect): 
    
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
                ZDRZDR = ZHZH-TV   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = radar.fields['ZDR']['data'][start_index:end_index]     
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
        
    del mycolorbar, x, y, inter
    
    #- savefile
    plt.suptitle(radar_name + ': '+str(file[0:12]) ,fontweight='bold')
    fig.savefig(fig_dir+'pseudo_RHI'+str(file)+'.png', dpi=300,transparent=False)
    plt.close()    
    
    
    
    return 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_rhi_RMA_wOFFSET(file, fig_dir, dat_dir, radar_name, xlim_range1, xlim_range2, test_transect, ZDR_offset): 
    
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
                ZDRZDR = ZHZH-TV   
            elif  'ZDR' in radar.fields.keys(): 
                ZDRZDR     = radar.fields['ZDR']['data'][start_index:end_index]     
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
        ZDR_transect[nlev,:]     = ZDRZDR[filas,:]-ZDR_offset
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
        
    del mycolorbar, x, y, inter
    
    #- savefile
    plt.suptitle(radar_name + ': '+str(file[0:12]) ,fontweight='bold')
    fig.savefig(fig_dir+'pseudo_RHI'+str(file)+'.png', dpi=300,transparent=False)
    #plt.close()    
    
    
    
    return 





#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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
        radarRHOHV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')    #['cross_correlation_ratio']

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
def plot_pseudo_RHI_parana(file, fig_dir, dat_dir, radar_name, test_transect, xlim_range1, xlim_range2):

    
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
        radarRHO = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')
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
        ZHZH        = radar.fields['reflectivity']['data'][start_index:end_index]
        lats        = radar.gate_latitude['data'][start_index:end_index]
        lons        = radar.gate_longitude['data'][start_index:end_index]
        # En verdad buscar azimuth no transecta ... 
        azimuths    = radar.azimuth['data'][start_index:end_index]
        if nlev == 0: 
            target_azimuth = azimuths[test_transect]  #- target azimuth for nlev=0 test case is 301.5
            plt.pcolormesh(lons, lats, radar.fields['reflectivity']['data'][start_index:end_index])
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()
        lon_transect[nlev,:]     = lons[filas,:]
        lat_transect[nlev,:]     = lats[filas,:]
        plt.plot(np.ravel(lon_transect[nlev,:]), np.ravel(lat_transect[nlev,:]))
        #
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
        start_index = radarRHO.sweep_start_ray_index['data'][nlev]
        end_index   = radarRHO.sweep_end_ray_index['data'][nlev]       
        RHORHO  = radarRHO.fields['cross_correlation_ratio']['data'][start_index:end_index]        
        #
        Ze_transect[nlev,:]      = ZHZH[filas,:]
        ZDR_transect[nlev,:]     = ZDRZDR[filas,:]
        PHI_transect[nlev,:]     = PHIDPPHIDP[filas,:]
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
         if nlev > 11: continue
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
    for nlev in range(len(radarZDR.sweep_start_ray_index['data'])):
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
    for nlev in range(len(radarZDR.sweep_start_ray_index['data'])):
        if nlev > 11: continue
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
    for nlev in range(len(radarRHO.sweep_start_ray_index['data'])):
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
    for nlev in range(len(radarRHO.sweep_start_ray_index['data'])):
        if nlev > 11: continue
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
        
    del mycolorbar, x, y, inter
    
    #- savefile
    plt.suptitle(radar_name + ': '+str(file[0:12]) ,fontweight='bold')
    fig.savefig(fig_dir+'pseudo_RHI'+str(file)+'.png', dpi=300,transparent=False)
    plt.close()    
    
    
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
            vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='velocity', 
                                                                    nyq=39.9)
            radar.add_field('velocity_texture', vel_texture, replace_existing=True)
            VEL_texture = radar.fields['velocity_texture']['data'][start_index:end_index]; 
            #- Plot texture 
            hist, bins = np.histogram(VEL_texture[~np.isnan(VEL_texture)], bins=150)
            bins = (bins[1:]+bins[:-1])/2.0
            plt.plot(bins, hist)
            plt.xlabel('Velocity texture')
            plt.ylabel('Count')
            plt.close()
            
            #gatefilter  = pyart.filters.GateFilter(radar)
            #gatefilter.exclude_above('velocity_texture', 3)
            velocity_dealiased = pyart.correct.dealias_region_based(radar, vel_field='velocity', nyquist_vel=39.9,
                                                        centered=True) #, gatefilter=gatefilter)
            radar.add_field('corrected_velocity', velocity_dealiased, replace_existing=True)
            VEL_cor = radar.fields['corrected_velocity']['data'][start_index:end_index]
            
            # Figure
            fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[20,12])  # 14,12
            #-- Doppler: 
            [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('doppler')
            pcm1 = axes.pcolormesh(lons, lats, VEL_cor, cmap=cmap, vmin=vmin, vmax=vmax)
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
            fig.savefig(fig_dir+str(file)+'sweep_'+str(nlev)+'_DOPPLER_corrected.png', dpi=300,transparent=False)
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
def check_transec(dat_dir, file_PAR_all, test_transect):       
  radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file_PAR_all+'dBZ.vol')
  nlev  = 0  
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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
def check_transec_zdr(dat_dir, file_PAR_all, test_transect):       
  radar = pyart.aux_io.read_rainbow_wrl(dat_dir+file_PAR_all+'ZDR.vol')
  nlev  = 0  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12])
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  pcm1 = axes.pcolormesh(lons, lats, radar.fields['differential_reflectivity']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
def check_transec_rma(dat_dir, file_PAR_all, test_transect):       
    
  radar = pyart.io.read(dat_dir+file_PAR_all) 
  nlev  = 0  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12])
  [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  if 'DBZH' in radar.fields.keys():
    pcm1 = axes.pcolormesh(lons, lats, radar.fields['DBZH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
  else: 
    pcm1 = axes.pcolormesh(lons, lats, radar.fields['TH']['data'][start_index:end_index], cmap=cmap, vmin=vmin, vmax=vmax)
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

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def check_transec_rma_campos(dat_dir, file_PAR_all, test_transect, campo, nlev):       

  radar = pyart.io.read(dat_dir+file_PAR_all) 
  start_index = radar.sweep_start_ray_index['data'][nlev]
  end_index   = radar.sweep_end_ray_index['data'][nlev]
  lats = radar.gate_latitude['data'][start_index:end_index]
  lons = radar.gate_longitude['data'][start_index:end_index]
  
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12]) 
  if 'ZDR' in campo:
    campotoplot = 'ZDR'
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
    if 'ZDR' in radar.fields.keys(): 
        campo_field_plot = radar.fields['ZDR']['data'][start_index:end_index]
    elif 'TH' in radar.fields.keys():
        ZHZH             = radar.fields['TH']['data'][start_index:end_index]
        TV               = radar.fields['TV']['data'][start_index:end_index]     
        campo_field_plot = ZHZH-TV
    elif 'DBZH' in radar.fields.keys():
        ZHZH       = radar.fields['DBZH']['data'][start_index:end_index]
        TV         = radar.fields['DBZV']['data'][start_index:end_index]     
        campo_field_plot = ZHZH-TV
  if 'RHO' in campo:
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
    campo_field_plot  = radar.fields['RHOHV']['data'][start_index:end_index]  
    campotoplot = r'$\rho_{HV}$'
  RHOFIELD  = radar.fields['RHOHV']['data'][start_index:end_index]  
 
  if 'ZH' in campo:
    campotoplot = 'Zh'
    [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
    if 'TH' in radar.fields.keys(): 
        campo_field_plot = radar.fields['TH']['data'][start_index:end_index]
    elif 'DBZH' in radar.fields.keys():
        campo_field_plot       = radar.fields['DBZH']['data'][start_index:end_index]      
        
  pcm1 = axes.pcolormesh(lons, lats, campo_field_plot, cmap=cmap, vmin=vmin, vmax=vmax)
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

  gateZ    = radar.gate_z['data'][start_index:end_index]
  gateX    = radar.gate_x['data'][start_index:end_index]
  gateY    = radar.gate_y['data'][start_index:end_index]
  gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
        
  fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                        figsize=[20,12]) 
  axes.plot(np.ravel(gates_range[filas,:]), np.ravel(campo_field_plot[filas,:]),'-k')
  plt.title('Lowest sweep transect of interest', fontsize=14)
  plt.xlabel('Range (km)', fontsize=14)
  plt.ylabel(str(campotoplot), fontsize=14)
  plt.grid(True)
  plt.ylim([-2,5])
    
  ax2= axes.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.plot(np.ravel(gates_range[filas,:]), np.ravel(RHOFIELD[filas,:]),'-r', label='RHOhv')
  plt.ylabel(r'$RHO_{rv}$')  
  plt.xlabel('Range (km)', fontsize=14)

  return



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
def plot_gmi(fname, options, radardat_dir, radar_file, rma):
    
    if rma == 5:
        radar = pyart.io.read(radardat_dir+radar_file) 
        reflectivity_name = 'DBZH'
    if rma == 1:
        radar = pyart.io.read(radardat_dir+radar_file) 
        reflectivity_name = 'TH'
    if rma == 4:
        radar = pyart.io.read(radardat_dir+radar_file) 
        reflectivity_name = 'TH'
    if rma == 8:
        radar = pyart.io.read(radardat_dir+radar_file) 
        reflectivity_name = 'TH'        
        print('rma=8')
    else:
        radar = pyart.aux_io.read_rainbow_wrl(radardat_dir+radar_file+'dBZ.vol')
     
    
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
    #inside_s1   = np.logical_and(np.logical_and(lon_s1[:,45] >= -70, lon_s1[:,45] <= -50), 
    #                          np.logical_and(lat_s1[:,45] >= -50, lat_s1[:,45] <= -20))
    #inside_s2   = np.logical_and(np.logical_and(lon_s2 >= -70, lon_s2 <= -50), 
    #                                     np.logical_and(lat_s2 >= -50, lat_s2 <= -20))    

    fig = plt.figure(figsize=(12,7)) 
    gs1 = gridspec.GridSpec(2, 3)
    
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
    im = plt.scatter(lon_gmi, lat_gmi, 
           c=tb_s1_gmi[:,:,5], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, '-k', linewidth=1)

    # BT(89)              
    ax1 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,:,7], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        
    # BT(166)           
    ax1 = plt.subplot(gs1[0,2], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon_s2_gmi[:], lat_s2_gmi[:], 
           c=tb_s2_gmi[:,:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('BT 166 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    
    # PD(37)
    ax1 = plt.subplot(gs1[1,0], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,:,5]-tb_s1_gmi[:,:,6], s=10, vmin=0, vmax=16, cmap=discrete_cmap(16,  'rainbow'))  
    plt.title('GMI PD 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    
    # PD(89)   
    ax1 = plt.subplot(gs1[1,1], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,:,7]-tb_s1_gmi[:,:,8], s=10, vmin=0, vmax=16, cmap=discrete_cmap(16,  'rainbow'))  
    plt.title('PD 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    
    # PD(166)       
    ax1 = plt.subplot(gs1[1,2], projection=ccrs.PlateCarree())
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
    im = plt.scatter(lon_s2_gmi[:], lat_s2_gmi[:], 
           c=tb_s2_gmi[:,:,0]-tb_s2_gmi[:,:,1], s=10, vmin=0, vmax=12, cmap=discrete_cmap(16,  'rainbow'))  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PD 166 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()       
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    
    return 
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
def plot_DPR(Ku_folder, DPR_file, fname, radar_file, options):

    fontsize = 12
    user = platform.system()
    if   user == 'Linux':
        home_dir = '/home/victoria.galligani/'  
    elif user == 'Darwin':
        home_dir = '/Users/victoria.galligani'
        
    radar = pyart.io.read(radar_file) 
        
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
        
    # list(f['/FS'].keys())
    #['ScanTime',
    #'scanStatus', 'navigation', 'PRE', 'VER', 'CSF', 'SRT', 'DSD', 'Experimental',
    #  'SLV', 'FLG', 'Latitude', 'Longitude', 'sunLocalTime']
    # The Ku-band SF algorithm consists of the preparation (PRE) module, vertical profile (VER) module, 
    # CSF module, drop size distribution (DSD) module, surface reference technique (SRT) module, 
    # and solver (SLV) module (Iguchi et al. 2020). The PRE module provides the measured radar reflectivity factor, 
    #precipitation/no-precipitation classification, and clutter mitigation (Kubota et al. 2016). 
    #The VER module computes the path-integrated attenuation (PIA) due to nonprecipitation particles (piaNP) and generates 
    #a radar reflectivity factor corrected for piaNP (Kubotaet al. 2020). The CSF module classifies precipitation types 
    # and obtains BB information. The SRT module estimates PIA for precipitation pixels (Meneghini et al. 2015). 
    # Finally, the SLV module numerically solves the radar equations and obtains DSD parameters and precipitation rates at
    # each range bin (Seto et al. 2013, 2021; Seto and Iguchi 2015).
        
    f = h5py.File( Ku_folder+DPR_file, 'r')   
    ku_NS_LON  = f[u'/FS/Longitude'][:,:]          
    ku_NS_LAT  = f[u'/FS/Latitude'][:,:]     
    flagPrecip = f[u'/FS/PRE/flagPrecip'][:,:]     
    zFactorMeasured = f[u'/FS/PRE/zFactorMeasured'][:,:,:]  
    heightStormTop  = f[u'/FS/PRE/heightStormTop'][:,:]  
    flagBB = f[u'/FS/CSF/flagBB'][:,:]  
    typePrecip = f[u'/FS/CSF/typePrecip'][:,:]  
    zFactorFinal = f[u'/FS/SLV/zFactorFinal'][:,:,:]   
    precipRate= f[u'/FS/SLV/precipRate'][:,:,:]  
    precipRateNearSurface = f[u'/FS/SLV/precipRateNearSurface'][:,:]  
    f.close()

    cmaps = GMI_colormap() 
    
    # read GMI
    f = h5py.File( fname, 'r')
    tb_s1_gmi = f[u'/S1/Tb'][:,:,:]           
    lon_gmi = f[u'/S1/Longitude'][:,:] 
    lat_gmi = f[u'/S1/Latitude'][:,:]
    tb_s2_gmi = f[u'/S2/Tb'][:,:,:]           
    lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
    lat_s2_gmi = f[u'/S2/Latitude'][:,:]
    f.close()
    

    inside_ku  = np.logical_and(np.logical_and(ku_NS_LON >= options['xlim_min'], 
                                                   ku_NS_LON <= options['xlim_max']), 
                 np.logical_and(ku_NS_LAT >= options['ylim_min'], 
                                ku_NS_LAT <= options['ylim_max']))

    fig = plt.figure(figsize=(12,7)) 
    gs1 = gridspec.GridSpec(2, 2)
   
    # BT(89)              
    ax1 = plt.subplot(gs1[0,1], projection=ccrs.PlateCarree())
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
           c=tb_s1_gmi[:,:,7], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    plt.plot(ku_NS_LON[:,0], ku_NS_LAT[:,0], '-k', linewidth=1.3)
    plt.plot(ku_NS_LON[:,-1], ku_NS_LAT[:,-1], '-k', linewidth=1.3)    
    plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,25), extend='both')

    # max(Zh_DPR)
    ax1 = plt.subplot(gs1[1,1], projection=ccrs.PlateCarree())
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
    im = plt.scatter(ku_NS_LON[inside_ku], ku_NS_LAT[inside_ku],
               c=np.max(zFactorFinal[inside_ku,:],1), vmin=0, vmax=40, s=10)
    plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(0,45,5), extend='both')
    plt.title('DPR Ku Zcorrected', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],np.max(radar.range['data'])/1e3)
    plt.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
    plt.title('DPR_Zcorr')
    plt.plot( ku_NS_LON[:,0] , ku_NS_LAT[:,0], '--k' )    
    plt.plot( ku_NS_LON[:,-1] , ku_NS_LAT[:,-1], '--k' )    
    
    return

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  


if __name__ == '__main__':

  plt.matplotlib.rc('font', family='serif', size = 12)
  plt.rcParams['xtick.labelsize']=12
  plt.rcParams['ytick.labelsize']=12  
    
  files_RMA5 = ['cfrad.20181001_231430.0000_to_20181001_232018.0000_RMA5_0200_01.nc',
                'cfrad.20200815_021618.0000_to_20200815_021906.0000_RMA5_0200_02.nc']

  # check 20181001 at 22UTC to see if ZDR is present before
  # files_RMA5 = ['cfrad.20181001_220245.0000_to_20181001_220337.0000_RMA5_0200_03.nc',
  #'cfrad.20181001_220354.0000_to_20181001_220942.0000_RMA5_0200_01.nc',
  #'cfrad.20181001_220952.0000_to_20181001_221237.0000_RMA5_0200_02.nc', 
  #'cfrad.20181001_221250.0000_to_20181001_221342.0000_RMA5_0200_03.nc',
  #'cfrad.20181001_221958.0000_to_20181001_222245.0000_RMA5_0200_02.nc',
  #'cfrad.20181001_222259.0000_to_20181001_222351.0000_RMA5_0200_03.nc',
  #'cfrad.20181001_223006.0000_to_20181001_223248.0000_RMA5_0200_02.nc',
  #'cfrad.20181001_223302.0000_to_20181001_223354.0000_RMA5_0200_03.nc',
  #'cfrad.20181001_225013.0000_to_20181001_225255.0000_RMA5_0200_02.nc',
  #'cfrad.20181001_225309.0000_to_20181001_225401.0000_RMA5_0200_03.nc'] 

  files_RMA1 = ['cfrad.20171027_034647.0000_to_20171027_034841.0000_RMA1_0123_02.nc',
              'cfrad.20171209_005909.0000_to_20171209_010438.0000_RMA1_0123_01.nc',
              'cfrad.20180208_205455.0000_to_20180208_205739.0000_RMA1_0201_02.nc',
              'cfrad.20180209_063643.0000_to_20180209_063908.0000_RMA1_0201_03.nc',
              'cfrad.20181111_124509.0000_to_20181111_125150.0000_RMA1_0301_01.nc', 
              'cfrad.20181214_030436.0000_to_20181214_031117.0000_RMA1_0301_01.nc',  # cfrad.20181214_025529.0000_to_20181214_030210.0000_RMA1_0301_01.nc
              'cfrad.20190102_204413.0000_to_20190102_204403.0000_RMA1_0301_02.nc',
              'cfrad.20190224_061413.0000_to_20190224_061537.0000_RMA1_0301_02.nc',  # cfrad.20190224_060559.0000_to_20190224_060723.0000_RMA1_0301_02.nc
              'cfrad.20190308_024050.0000_to_20190308_024731.0000_RMA1_0301_01.nc']

  files_RMA1_GMI = ['1B.GPM.GMI.TB2016.20171027-S021318-E034550.020807.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20171209-S005338-E022611.021475.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20180208-S193936-E211210.022436.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20180209-S062741-E080015.022443.V05A.HDF5',  # <--- 
              '1B.GPM.GMI.TB2016.20181111-S113214-E130446.026724.V05A.HDF5',  # <--- 
              '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20190102-S192616-E205848.027538.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20190224-S045410-E062643.028353.V05A.HDF5',
              '1B.GPM.GMI.TB2016.20190308-S004613-E021846.028537.V05A.HDF5']
    
  files_2018_BB_RMA1 = ['cfrad.20180209_005340.0000_to_20180209_005622.0000_RMA1_0201_02.nc', 
                        'cfrad.20180209_040042.0000_to_20180209_040631.0000_RMA1_0201_01.nc',
                        'cfrad.20180209_065921.0000_to_20180209_070146.0000_RMA1_0201_03.nc', 
                        'cfrad.20181112_065108.0000_to_20181112_065756.0000_RMA1_0301_01.nc',
                        'cfrad.20190126_071808.0000_to_20190126_072449.0000_RMA1_0301_01.nc', 
                        'cfrad.20190304_084858.0000_to_20190304_085539.0000_RMA1_0301_01.nc'] 

  files_Ku_RMA1 = ['2A.GPM.Ku.V9-20211125.20171027-S021318-E034550.020807.V07A.HDF5', 
                   '2A.GPM.Ku.V9-20211125.20171209-S005338-E022611.021475.V07A.HDF5',
                   '2A.GPM.Ku.V9-20211125.20180208-S193936-E211210.022436.V07A.HDF5', 
                   '2A.GPM.Ku.V9-20211125.20180209-S062741-E080015.022443.V07A.HDF5', 
                   '',
                   '2A.GPM.Ku.V9-20211125.20181214-S015009-E032242.027231.V07A.HDF5', 
                   '',
                   '', 
                   '2A.GPM.Ku.V9-20211125.20190308-S004613-E021846.028537.V07A.HDF5']                   

  files_RMA4 = ['cfrad.20180124_090045.0000_to_20180124_090316.0000_RMA4_0201_03.nc', #'cfrad.20180124_110519.0000_to_20180124_110751.0000_RMA4_0201_03.nc',
                'cfrad.20181218_014441.0000_to_20181218_015039.0000_RMA4_0200_01.nc',
                'cfrad.20180209_200449.0000_to_20180209_201043.0000_RMA4_0200_01.nc',
                'cfrad.20190209_192724.0000_to_20190209_193317.0000_RMA4_0200_01.nc',
                'cfrad.20181001_095450.0000_to_20181001_100038.0000_RMA4_0200_01.nc',
                'cfrad.20200119_045758.0000_to_20200119_050347.0000_RMA4_0200_01.nc',
                'cfrad.20201213_051236.0000_to_20201213_051830.0000_RMA4_0200_01.nc',
                'cfrad.20181031_010936.0000_to_20181031_011525.0000_RMA4_0200_01.nc',
                'cfrad.20181215_021522.0000_to_20181215_022113.0000_RMA4_0200_01.nc']

  files_RMA4_GMI = ['1B.GPM.GMI.TB2016.20180124-S105204-E122438.022197.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181217-S235720-E012953.027292.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20180209-S184820-E202054.022451.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20190209-S191744-E205018.028129.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181001-S093732-E111006.026085.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20200119-S033832-E051104.033470.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20201213-S035613-E052844.038588.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181031-S005717-E022950.026546.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181215-S005848-E023122.027246.V05A.HDF5']
    
  files_Ku_RMA4 = ['2A.GPM.Ku.V9-20211125.20180124-S105204-E122438.022197.V07A.HDF5',
                   '2A.GPM.Ku.V9-20211125.20181217-S235720-E012953.027292.V07A.HDF5',
                   '',
                   '',
                   '',
                   '',
                   '',
                   '2A.GPM.Ku.V9-20211125.20181031-S005717-E022950.026546.V07A.HDF5',
                   '2A.GPM.Ku.V9-20211125.20181215-S005848-E023122.027246.V07A.HDF5']
    
  files_RMA3 = ['cfrad.20180925_020737.0000_to_20180925_021025.0000_RMA3_0200_02.nc',
                'cfrad.20201026_051729.0000_to_20201026_052017.0000_RMA3_0200_02.nc',
                'cfrad.20181122_190207.0000_to_20181122_190801.0000_RMA3_0200_01.nc', #'cfrad.20181122_191932.0000_to_20181122_192525.0000_RMA3_0200_01.nc',
                'cfrad.20201216_040814.0000_to_20201216_041408.0000_RMA3_0200_01.nc',
                'cfrad.20190305_124638.0000_to_20190305_125231.0000_RMA3_0200_01.nc'] #'cfrad.20190305_125231.0000_to_20190305_125513.0000_RMA3_0200_02.nc']
  
  files_RMA3_GMI = ['1B.GPM.GMI.TB2016.20180925-S005219-E022451.025986.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20201026-S050040-E063312.037842.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181122-S190544-E203818.026900.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20201216-S025258-E042529.038634.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20190305-S123614-E140847.028498.V05A.HDF5']

  files_Ku_RMA3 = ['',
                   '2A.GPM.Ku.V9-20211125.20201026-S050040-E063312.037842.V07A.HDF5',
                   '',
                   '2A.GPM.Ku.V9-20211125.20201216-S025258-E042529.038634.V07A.HDF5',
                   '2A.GPM.Ku.V9-20211125.20190305-S123614-E140847.028498.V07A.HDF5']

  files_RMA8 = ['cfrad.20181001_094859.0000_to_20181001_095449.0000_RMA8_0200_01.nc',
                'cfrad.20181113_115911.0000_to_20181113_120155.0000_RMA8_0200_02.nc', #'cfrad.20181113_115023.0000_to_20181113_115317.0000_RMA8_0200_01.nc',
                'cfrad.20181031_011415.0000_to_20181031_012008.0000_RMA8_0200_01.nc', #'cfrad.20181031_011126.0000_to_20181031_011415.0000_RMA8_0200_02.nc',
                'cfrad.20181212_031143.0000_to_20181212_031739.0000_RMA8_0200_01.nc',
                'cfrad.20181112_115631.0000_to_20181112_120225.0000_RMA8_0200_01.nc',
                'cfrad.20200630_053936.0000_to_20200630_054527.0000_RMA8_0200_01.nc', #'cfrad.20200630_053643.0000_to_20200630_053936.0000_RMA8_0200_02.nc', 
                'cfrad.20181112_214301.0000_to_20181112_214855.0000_RMA8_0200_01.nc']

  files_RMA8_GMI = ['1B.GPM.GMI.TB2016.20181001-S093732-E111006.026085.V05A.HDF5', 
                    '1B.GPM.GMI.TB2016.20181113-S112121-E125353.026755.V05A.HDF5',   REVISAR
                    '1B.GPM.GMI.TB2016.20181031-S005717-E022950.026546.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20181212-S033249-E050523.027201.V05A.HDF5',
                    '1B.GPM.GMI.TB2016.20200630-S042029-E055302.036006.V05A.HDF5', 
                    '1B.GPM.GMI.TB2016.20181112-S212823-E230055.026746.V05A.HDF5']

  # Files below organized: per line. tested for differente minutes. each line is a case study
  files_PAR = ['2018032407543300dBZ.vol', '2018032407500500dBZ.vol',
             '2018100907443400dBZ.vol',   '2018100907400500dBZ.vol', '2018100907343300dBZ.vol', '2018100907300500dBZ.vol',
             '2018121403043200dBZ.vol',   '2018121402400200dBZ.vol', 
             '2019021109400500dBZ.vol',
             '2019022315143100dBZ.vol',   '2019022315100200dBZ.vol',
             '2020121903100500dBZ.vol']
    
  #------------------------------------------------------------------------------  
  #------------------------------------------------------------------------------  
  # start w/ RMA1
  opts = {'xlim_min': -70, 'xlim_max': -60, 'ylim_min': -35, 'ylim_max': -25}
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
    #plot_ppi(files_RMA1[ifiles], fig_dir, dat_dir, 'RMA1')
    # plot GMIs:  
    plot_gmi(gmi_path+'/'+files_RMA1_GMI[ifiles], opts, dat_dir, files_RMA1[ifiles], 1)   # 1 ---> RMA1
 
    check_transec_rma(dat_dir, files_RMA1[0], 345)     
    plot_rhi_RMA(files_RMA1[1], fig_dir, dat_dir, 'RMA1', 0, 100, 7)
    
    test_transect = [345, 7 y 220, 283 y 50, 215 y 110,  ]   el ultimo [8] es 55
    xlim2_ranges = [220, 60 y 120, 120 y 50, 160 y 100, ]    el ultimo [8] es 200
    
    files_RMA1_alternatve = 'cfrad.20181214_025529.0000_to_20181214_030210.0000_RMA1_0301_01.nc'
    files_RMA1_alternatve = 'cfrad.20181214_024550.0000_to_20181214_024714.0000_RMA1_0301_02.nc'
    check_transec_rma(dat_dir, 'cfrad.20181214_024550.0000_to_20181214_024714.0000_RMA1_0301_02.nc', 250)     
    plot_rhi_RMA('cfrad.20181214_024550.0000_to_20181214_024714.0000_RMA1_0301_02.nc', fig_dir, dat_dir, 'RMA1', 0, 160, 183)
    
    # GMI -S045410-E062643. y el Phal es 0613. 
    files_RMA1_alternatve =  'cfrad.20190224_060559.0000_to_20190224_060723.0000_RMA1_0301_02.nc'
    check_transec_rma(dat_dir, 'cfrad.20190224_060559.0000_to_20190224_060723.0000_RMA1_0301_02.nc', 220)
    plot_rhi_RMA('cfrad.20190224_060559.0000_to_20190224_060723.0000_RMA1_0301_02.nc', fig_dir, dat_dir, 'RMA1', 0, 200, 220)
    files_RMA1_alternatve = 'cfrad.20190224_060723.0000_to_20190224_061403.0000_RMA1_0301_01.nc'
    check_transec_rma(dat_dir, 'cfrad.20190224_060723.0000_to_20190224_061403.0000_RMA1_0301_01.nc', 220)
    plot_rhi_RMA('cfrad.20190224_060723.0000_to_20190224_061403.0000_RMA1_0301_01.nc', fig_dir, dat_dir, 'RMA1', 0, 200, 220)
    
    gmi_path = '/home/victoria.galligani/datosmunin2/DATOS_mw/GMI/'
    plot_DPR('/home/victoria.galligani/Work/Studies/Hail_MW/DPR_data/', 
          '2A.GPM.Ku.V9-20211125.20171027-S021318-E034550.020807.V07A.HDF5', 
          gmi_path+'1B.GPM.GMI.TB2016.20171027-S021318-E034550.020807.V05A.HDF5',
          dat_dir+files_RMA1[0], opts)
  #--------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------
  # start w/ RMA4
  opts = {'xlim_min': -65, 'xlim_max': -55, 'ylim_min': -32, 'ylim_max': -24}
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/RMA4/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA4/'
  #gmi_path = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
  this_trans1 = [163, 240, 244, 270, 190, 295, 50, 65, 238]
  this_trans2 = [310, 147, 228, 256, 149, 275, 43, 230, 180]
  for ifiles in range(len(files_RMA4)):
    #plot_ppi(files_RMA4[ifiles], fig_dir, dat_dir, 'RMA4')
    #check_transec_rma(dat_dir, files_RMA4[ifiles], this_trans1[ifiles])
    #check_transec_rma(dat_dir, files_RMA4[ifiles], this_trans2[ifiles] )
    #plot_rhi_RMA(files_RMA4[ifiles], fig_dir, dat_dir, 'RMA4', 0, 200, this_trans1[ifiles])
    #plot_rhi_RMA(files_RMA4[ifiles], fig_dir, dat_dir, 'RMA4', 0, 200, this_trans2[ifiles])
    plot_gmi(gmi_path+'/'+files_RMA4_GMI[ifiles], opts, dat_dir, files_RMA4[ifiles], 4)   # 1 ---> RMA1

  #--------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------
  # start w/ RMA3
  opts = {'xlim_min': -65, 'xlim_max': -55, 'ylim_min': -28, 'ylim_max': -22}
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/RMA3/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA3/'
  #dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA3bis/'
  #onlyfiles = [f for f in listdir(dat_dir) if isfile(join(dat_dir, f))]
  #for ifiles in range(len(onlyfiles)):
  #  plot_ppi(onlyfiles[ifiles], fig_dir, dat_dir, 'RMA3')
    
  for ifiles in range(len(files_RMA3)):
    plot_ppi(files_RMA3[ifiles], fig_dir, dat_dir, 'RMA3')
    
  check_transec_rma(dat_dir, files_RMA3[0], 304])
  plot_rhi_RMA(files_RMA3[0], fig_dir, dat_dir, 'RMA3', 0, 200, 304)
  check_transec_rma(dat_dir, files_RMA3[2], 94])
  check_transec_rma(dat_dir, files_RMA3[2], 109])
  check_transec_rma(dat_dir, files_RMA3[2], 197])
  plot_rhi_RMA(files_RMA3[2], fig_dir, dat_dir, 'RMA3', 0, 200, 94)
  plot_rhi_RMA(files_RMA3[2], fig_dir, dat_dir, 'RMA3', 0, 200, 109)
  plot_rhi_RMA(files_RMA3[2], fig_dir, dat_dir, 'RMA3', 0, 200, 197)
  check_transec_rma(dat_dir, files_RMA3[3], 200])
  plot_rhi_RMA(files_RMA3[3], fig_dir, dat_dir, 'RMA3', 0, 200, 200)
  check_transec_rma(dat_dir, files_RMA3[4], 176])
  check_transec_rma(dat_dir, files_RMA3[4], 30])
  plot_rhi_RMA(files_RMA3[4], fig_dir, dat_dir, 'RMA3', 0, 200, 176)    
  plot_rhi_RMA(files_RMA3[4], fig_dir, dat_dir, 'RMA3', 0, 200, 30)    
    
  plot_gmi(gmi_path+'/'+files_RMA3_GMI[ifiles], opts, dat_dir, files_RMA3[ifiles], 3)   # 1 ---> RMA1  
  #--------------------------------------------------------------------------------------------
  # start w/ RMA8
  opts = {'xlim_min': -65, 'xlim_max': -55, 'ylim_min': -35, 'ylim_max': -25}
  fig_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/RMA8/'
  dat_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/RMA8/'
  for ifiles in range(len(files_RMA8)):
    #plot_ppi(files_RMA8[ifiles], fig_dir, dat_dir, 'RMA8')
    plot_gmi(gmi_path+'/'+files_RMA8_GMI[ifiles], opts, dat_dir, files_RMA8[ifiles], 8)   # 1 ---> RMA1  
    
  check_transec_rma(dat_dir, files_RMA8[0], 30)
  plot_rhi_RMA(files_RMA8[0], fig_dir, dat_dir, 'RMA3', 0, 200, 30)  
    
  
  #--------------------------------------------------------------------------------------------
  # start w/ RMA5
  opts = {'xlim_min': -60, 'xlim_max': -50, 'ylim_min': -30, 'ylim_max': -20}
  test_transect =  [10,333]
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
    check_transec_rma(dat_dir, files_RMA5[ifiles], test_transect[ifiles])     
    plot_rhi_RMA(files_RMA5[ifiles], fig_dir, dat_dir, radar_name, 0, 160, test_transect[ifiles])
  # plot GMIs:  
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20181001-S215813-E233047.026093.V05A.HDF5'
  plot_gmi(gmi_file, opts, dat_dir, files_RMA5[0], 5)   # 5 ---> RMA5
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20200815-S015947-E033219.036720.V05A.HDF5'
  plot_gmi(gmi_file, opts, dat_dir, files_RMA5[1], 5)   # 5 ---> RMA5

  #--------------------------------------------------------------------------------------------
  # start w/ PARANA
  fig_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_figures/PAR/'
  dat_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/PAR/'
  gmi_path = '/home/victoria.galligani/datosmunin2/DATOS_mw/GMI'
  opts = {'xlim_min': -65, 'xlim_max': -55, 'ylim_min': -36, 'ylim_max': -28}
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
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-32,-30.5]) # figsize=[20,12])
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-32,-30.5]) 
  #test_transect = 235
  #check_transec(dat_dir, file_PAR_all, test_transect)     
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 150)  
  #test_transect = 312
  #check_transec(dat_dir, file_PAR_all, test_transect)     
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 150)   
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20181009-S072527-E085801.026208.V05A.HDF5'
  plot_gmi(gmi_file, opts, dat_dir, file_PAR_all)
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2018121402400200'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63.3,-58.5], [-34.4,-31.5]) 
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63.3,-58.5], [-34.4,-31.5])  
  #test_transect = 45
  #check_transec(dat_dir, file_PAR_all, test_transect)     
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 150)  
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5' 
  plot_gmi(gmi_file, opts, dat_dir, file_PAR_all)
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2019021109400500'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-34,-30]) 
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-62,-58], [-34,-30]) 
  #test_transect = 52
  #check_transec(dat_dir, file_PAR_all, test_transect)     
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 150)  
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20190211-S081945-E095219.028153.V05A.HDF5'
  plot_gmi(gmi_file, opts, dat_dir, file_PAR_all)
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2019022315100200'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5, -58], [-34,-31]) 
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-61.5, -58], [-34,-31]) 
  #test_transect = 240
  #check_transec(dat_dir, file_PAR_all, test_transect)     
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 110)
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20190223-S150103-E163336.028344.V05A.HDF5'
  plot_gmi(gmi_file, opts, dat_dir, file_PAR_all)
  #--------------------------------------------------------------------------------------------
  file_PAR_all = '2020121903100500'
  folder = str(file_PAR_all[0:8]) 
  #plot_ppi_parana_all(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR') 
  #plot_ppi_parana_doppler(file_PAR_all,  fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63,-58], [-34,-29.5]) 
  #plot_ppi_parana_all_zoom(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', [-63,-58], [-34,-29.5]) 
  #test_transect = 250
  #check_transec(dat_dir, file_PAR_all, test_transect)       
  #plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', test_transect, 0, 150)
  gmi_file = gmi_path+'/1B.GPM.GMI.TB2016.20201219-S015114-E032348.038680.V05A.HDF5' 
  plot_gmi(gmi_file, opts, dat_dir, file_PAR_all)
  # check ZDR offset ?
  test_transect = 100
  check_transec(dat_dir, file_PAR_all, test_transect)  
  check_transec_zdr(dat_dir, file_PAR_all, test_transect)  
  plot_pseudo_RHI_parana(file_PAR_all, fig_dir+'full_pol/'+folder+'/', dat_dir, 'PAR', 100, 0, 150)
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------    
   

    

    
  

