#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:07:55 2021

@author: victoria.galligani
"""

lonbins = np.arange(-80, -40, 2) 
latbins = np.arange(-50, -10, 2)
xi,yi   = np.meshgrid(lonbins,latbins)

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    zi    = scipy.interpolate.griddata((LON,LAT), MIN37PCT, (xi,yi), method='linear')
    pc = ax1.pcolor(lonbins, latbins, zi, vmin=100, vmax=300)
    plt.colorbar(pc)        
    ax1.set_title('Linear')
      
    ax1 = plt.subplot(gs1[0,1])
    zi    = scipy.interpolate.griddata((LON,LAT), MIN37PCT, (xi,yi), method='nearest')
    pc = ax1.pcolor(lonbins, latbins, zi, vmin=100, vmax=300)
    plt.colorbar(pc)    
    ax1.set_title('nearest')
    
    grid_PCT = np.zeros((len(xi), len(yi))); grid_PCT[:] = np.nan
    # Hernan
    for j in range(len(xi)):  
        for k in range(len(yi)): 
            abslat = np.abs(LAT-yi[j,k])
            abslon = np.abs(LON-xi[j,k])
            c = np.maximum(abslon,abslat)
            latlon_idx = np.argmin(c)       
            grid_PCT[j,k] = MIN37PCT.flat[latlon_idx]   
    
    ax1 = plt.subplot(gs1[1,0])
    pc = ax1.pcolor(lonbins, latbins, grid_PCT, vmin=100, vmax=300)
    plt.colorbar(pc)    
    ax1.set_title('Hernan')

    ax1 = plt.subplot(gs1[1,1])
    pc = ax1.pcolor(lonbins, latbins, zi-grid_PCT, vmin=-10, vmax=10)
    plt.colorbar(pc)    
    ax1.set_title('nearest - Hernan')


    # Frequency distribution
    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    ax1 = plt.subplot(gs1[0,0])
    arr = np.random.random((100, 2))
    lobins = np.arange(-80, -40, 2) 
    labins = np.arange(-50, -10, 2)
    labins = arr[:, 0]
    lobins = arr[:, 1]
    hist, xedges, yedges = np.histogram2d(lobins, labins)
    p = ax1.imshow(hist)
    ax1.set_title('Sin interpolar')
    plt.colorbar(p)

    ax1 = plt.subplot(gs1[0,1])
    plt.hist2d(x, y, bins=30, cmap='Blues')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    ax1.set_title('Interpolado')