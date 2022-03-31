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
    radarZDR = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'ZDR.vol')
    radarW = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'W.vol')         #['spectrum_width']
    radarV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'V.vol')         #['velocity']
    radaruPhiDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'uPhiDP.vol')    #['uncorrected_differential_phase']
    radarRhoHV = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'RhoHV.vol')
    radarPhiDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'PhiDP.vol') #['differential_phase'] 
    radarKDP = pyart.aux_io.read_rainbow_wrl(dat_dir+file+'KDP.vol')

    
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
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zdr')
        pcm1 = axes[0,1].pcolormesh(lons, lats, radarZDR, cmap=cmap, vmin=vmin, vmax=vmax)
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
        pcm1 = axes[1,0].pcolormesh(lons, lats, radarPHIDP, cmap=cmap, vmin=vmin, vmax=vmax)
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
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('rhohv')
        pcm1 = axes[1,0].pcolormesh(lons, lats, radarRHOHV, cmap=cmap, vmin=vmin, vmax=vmax)
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
             '2018100907443400dBZ.vol', '2018100907400500dBZ.vol', '2018100907343300dBZ.vol', '2018100907300500dBZ.vol',
             '2018121403043200dBZ.vol', '2018121402400200dBZ.vol', 
             '2019021109400500dBZ.vol',
             '2019022315143100dBZ.vol', '2019022315100200dBZ.vol',
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
  # Plot all variables:
  file_PAR_all = '2019022315100200'
  folder = str(files_PAR[0:8]) 
  plot_ppi_parana_all(file_PAR_all, fig_dir, dat_dir, 'PAR') 

    
#--------------------------------------------------------------------------------------------    
   

    

    
  

