#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:15:25 2021
@purpose : Analaysis of temporal and spatial coverage during RELAMPAGO w/ GPM-core
@author  : V. Galligani
@email   : victoria.galligani@cima.fcen.uba.ar
NOTE THAT THIS SCRIPT IS USED IN YAKAIRA! 
-------------------------------------------------------------------------------
@TODOS(?):

-------------------------------------------------------------------------------
"""
import h5py 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt;
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from os import listdir
from os.path import isfile, join
import datetime as dt
from datetime import datetime
import netCDF4 as nc
import xarray  as xr
import matplotlib.colors as colors_mat
from collections import defaultdict
import pandas as pd
import seaborn as sns
import Plots as PLOTS
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def get_ellipse_info(pf_data, select, j):

    u=pf_data['R_LON'][select][j]               #x-position of the center
    v=pf_data['R_LAT'][select][j]               #y-position of the center
    # TESTING THIS PROJ.  math.cos(math.radians(1))
    a=pf_data['R_MINOR'][select][j]/111   # x-axis (1degre == 92 km)
    b=pf_data['R_MAJOR'][select][j]/111# y-axis


    t_rot=pf_data['R_ORIENTATION'][select][j]   #rotation angle
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            
    return u+Ell_rot[0,:], v+Ell_rot[1,:]


    t_rot=pf_data['R_ORIENTATION'][select][j]   #rotation angle
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
           
    return u+Ell_rot[0,:], v+Ell_rot[1,:]


def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def get_mhs(mhs_dir, files, platform):
    
    main_month    = []
    main_day      = []
    main_hour     = []
    main_minute   = []
    main_doy      = []
    main_coverage = []
    hour_central  = []

    for i in files:
        if i.startswith(platform):  

            # Read file
            #fname = mhs_dir+test
            fname = mhs_dir+i
            
            # full orbit
            f = h5py.File( fname, 'r')
            if 'GMI' in platform: 
                tbs = f[u'/S1/Tb'][:,:,:]           
            else:
                tbs = f[u'/S1/Tc'][:,:,:]           
            lon = f[u'/S1/Longitude'][:,:] 
            lat = f[u'/S1/Latitude'][:,:]
            mmm = f[u'/S1/ScanTime/Month'][:]           
            ddd = f[u'/S1/ScanTime/DayOfMonth'][:]           
            hhh = f[u'/S1/ScanTime/Hour'][:]           
            minute = f[u'/S1/ScanTime/Minute'][:]           
            doy = f[u'/S1/ScanTime/DayOfYear'][:]   
            f.close()
            
            # keep domain of interest only by keeping those where the center nadir obs is inside domain
            inside   = np.logical_and(np.logical_and(lon[:,45] >= opts['xlim_min'], lon[:,45] <= opts['xlim_max']), 
                         np.logical_and(lat[:,45] >= opts['ylim_min'], lat[:,45] <= opts['ylim_max']))
            
            
            if inside[inside==True].shape[0] > 4:
                
                hour_central.append(hhh[inside==True].mean())

                inside   = np.logical_and(np.logical_and(lon >= opts['xlim_min'], lon <= opts['xlim_max']), 
                                     np.logical_and(lat >= opts['ylim_min'], lat <= opts['ylim_max']))    
                dlon      = lon[inside]
                dlat      = lat[inside]
                
                # Calculate the percentage of domain area covered
                coordinates = np.column_stack((dlon, dlat)) 
                
                if len(coordinates)>2:  # Only for more than two points
                    hull = ConvexHull(coordinates)
                    swatharea = hull.volume
                    # Calculate fraction of domain covered by radiometer
                    cov_perct = swatharea/darea_do
                    
                    # Plot to test hull working
                    # plt.plot(coordinates[:,0],coordinates[:,1], 'o'); plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);
                    # plt.plot(lon[0:400,45], lat[0:400,45],'xr-')
                    # for simplex in hull.simplices:
                    #     plt.plot(coordinates[simplex, 0], coordinates[simplex, 1], 'k-')
                
                    # Keep central time (middle in terms of latitude)
                    mlat   = (np.max(lat[inside])+np.min(lat[inside]))/2
                    arraylat = lat[:,45]
                    arraylon = lon[:,45]
                    check     = np.argpartition(np.abs(arraylat - mlat), 10)
                    check_lat = arraylat[check[:10]]
                    check_lon = arraylon[check[:10]]
                    # Keep those indices where check_lon is closest to -60
                    value, idx  = find_nearest(check_lon, -60.)
                    value, idx  = find_nearest(lon[:,45], value)
                
                    main_month.append(mmm[idx])
                    main_day.append(ddd[idx])
                    main_hour.append(hhh[idx])
                    main_minute.append(minute[idx])
                    main_doy.append(doy[idx])  
                    main_coverage.append(cov_perct)
                    
                    print('Time of passage: '+ str(hhh[idx])+':'+str(minute[idx])+' UTC')
                    
                    
    #- SAVE THE ABOVE in netcdf
    ds = xr.Dataset( {
                    "month":     (main_month),
                    "day":       (main_day),
                    "hour":      (main_hour),
                    "minute":    (main_minute),
                    "doy":       (main_doy),
                    "coverage":  (main_coverage),
                    "hour_central":     (hour_central),
                    }   )
    ds.to_netcdf('/home/victoria.galligani/datosmunin2/DATOS_mw/S1_domain_'+platform+'.nc', 'w')
                
    return main_month, main_day, main_hour, main_minute, main_doy, main_coverage, hour_central


def plot_all_orbits_inside(onlyfiles):
    
    # Plot all '1C.METOPB' files inside domain of interest. 
    for i in onlyfiles:
        if i.startswith('1C.METOPB'):  
            # read file
            fname = mhs_dir+i
            # full orbit
            f = h5py.File( fname, 'r')
            tbs = f[u'/S1/Tc'][:,:,:]           
            lon = f[u'/S1/Longitude'][:,:] 
            lat = f[u'/S1/Latitude'][:,:]
            mmm = f[u'/S1/ScanTime/Month'][:]           
            ddd = f[u'/S1/ScanTime/DayOfMonth'][:]           
            hhh = f[u'/S1/ScanTime/Hour'][:]           
            minute = f[u'/S1/ScanTime/Minute'][:]           
            doy = f[u'/S1/ScanTime/DayOfYear'][:]   
            f.close()
            # keep domain of interest only by keeping those where the center nadir obs is inside domain
            inside   = np.logical_and(np.logical_and(lon[:,45] >= opts['xlim_min'], lon[:,45] <= opts['xlim_max']), 
                             np.logical_and(lat[:,45] >= opts['ylim_min'], lat[:,45] <= opts['ylim_max']))
            if inside[inside==True].shape[0] > 4:
                inside   = np.logical_and(np.logical_and(lon >= opts['xlim_min'], lon <= opts['xlim_max']), 
                                         np.logical_and(lat >= opts['ylim_min'], lat <= opts['ylim_max']))    
                dlon      = lon[inside]
                dlat      = lat[inside]
                fig = plt.figure(figsize=(12,12))  
                plt.plot(dlon,dlat, 'x', color='gray'); 
                plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);
                plt.title(str(i))
    
    return

def plot_histogram(metopA_hour_central, metopB_hour_central, noaa19_hour_central, 
                   F16_hour_central, F17_hour_central, F18_hour_central,
                   atms_noaa20_hour_central, atms_npp_hour_central, gmi_hour_central, amsr2_hour_central, filename): 

    #- PLOT histograms
    fig = plt.figure(figsize=(12,12))  
    ax = plt.subplot(111)
    # mhs
    plt.hist([metopA_hour_central, metopB_hour_central, noaa19_hour_central, 
              F16_hour_central, F17_hour_central, F18_hour_central, 
              atms_noaa20_hour_central, atms_npp_hour_central, gmi_hour_central, amsr2_hour_central], np.linspace(0, 24, 25),
                  color=[colors_mat.to_rgba('red'), colors_mat.to_rgba('darkgreen'), colors_mat.to_rgba('lightcoral'),
                            colors_mat.to_rgba('darkblue'), colors_mat.to_rgba('blue'), colors_mat.to_rgba('royalblue'), 
                            colors_mat.to_rgba('darkred'), colors_mat.to_rgba('forestgreen'), colors_mat.to_rgba('gray'), 
                            colors_mat.to_rgba('orange')], 
                  label=['metop-A (MHS)', 'metop-B (MHS)', 'noaa-19 (MHS)', 
                        'F-16 (SSMIS)', 'F-17 (SSMIS)', 'F-18 (SSMIS)', 
                        'noaa-20 (ATMS)', 'npp (ATMS)', 'GPM-GMI','gcom-w1 (AMSR2)'], range=(0,24), width=0.4, 
                  alpha=0.7, histtype='bar', cumulative=False )
        
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Counts inside domain 01/10/2018-31/12/2018')
    ax.set_xticks(np.arange(0,24+3,3))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              fancybox=True, shadow=True, ncol=5)
   
    fig.savefig(filename, dpi=300,transparent=False)        

    return

def plot_histogram(metopA_hour_central, metopB_hour_central, noaa19_hour_central, 
                   F16_hour_central, F17_hour_central, F18_hour_central,
                   atms_noaa20_hour_central, atms_npp_hour_central, gmi_hour_central, amsr2_hour_central, filename): 

    #- PLOT histograms
    fig = plt.figure(figsize=(12,12))  
    ax = plt.subplot(111)
    # mhs
    plt.hist([metopA_hour_central, metopB_hour_central, noaa19_hour_central, 
              F16_hour_central, F17_hour_central, F18_hour_central, 
              atms_noaa20_hour_central, atms_npp_hour_central, gmi_hour_central, amsr2_hour_central], np.linspace(0, 24, 25),
                  color=[colors_mat.to_rgba('red'), colors_mat.to_rgba('darkgreen'), colors_mat.to_rgba('lightcoral'),
                            colors_mat.to_rgba('darkblue'), colors_mat.to_rgba('blue'), colors_mat.to_rgba('royalblue'), 
                            colors_mat.to_rgba('darkred'), colors_mat.to_rgba('forestgreen'), colors_mat.to_rgba('gray'), 
                            colors_mat.to_rgba('orange')], 
                  label=['metop-A (MHS)', 'metop-B (MHS)', 'noaa-19 (MHS)', 
                        'F-16 (SSMIS)', 'F-17 (SSMIS)', 'F-18 (SSMIS)', 
                        'noaa-20 (ATMS)', 'npp (ATMS)', 'GPM-GMI','gcom-w1 (AMSR2)'], range=(0,24), width=0.5, 
                  alpha=0.7, histtype='barstacked', cumulative=False )
        
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Counts inside domain 01/10/2018-31/12/2018')
    ax.set_xticks(np.arange(0,24+3,3))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              fancybox=True, shadow=True, ncol=5)
   
    fig.savefig(filename, dpi=300,transparent=False)        

    return


def plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
             lon_s2_gmi, lat_s2_gmi, options):
    
    """Create a 2x3 GMI colormap with BT(37, 89, 166) and PD(37, 89, 166)
        w. BT contours on the PD maps. Also include ACP alertsa del SMN. 
        Include PF ellipse from database"""
        
    geo_reg_shp = '/Users/victoria.galligani/Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)
        
    cmaps = PLOTS.GMI_colormap()
    
    countries = shpreader.Reader('/Users/victoria.galligani/Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')

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

    ifile = options['ifile']
    yoi = int(ifile[22:26])
    moi = int(ifile[26:28]) #"%02d" % int(ifile[26:28])
    doi = int(ifile[28:30]) #"%02d" % int(ifile[28:30])
    print('day of interest: '+str(doi) )
    
    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 

    inside  = np.logical_and(np.logical_and(lon_gmi >= options['xlim_min'], lon_gmi <= options['xlim_max']), 
                             np.logical_and(lat_gmi >= options['ylim_min'], lat_gmi <= options['ylim_max']))

    inside2 = np.logical_and(np.logical_and(lon_s2_gmi >= options['xlim_min'], lon_s2_gmi <= options['xlim_max']), 
                             np.logical_and(lat_s2_gmi >= options['ylim_min'], lat_s2_gmi <= options['ylim_max']))

         
    data_tb37 = PLOTS.apply_geofence_on_data (tb_s1_gmi[:,:,5], lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_tb89 = PLOTS.apply_geofence_on_data (tb_s1_gmi[:,:,7], lat_gmi, lon_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    data_tb166 = PLOTS.apply_geofence_on_data (tb_s2_gmi[:,:,0], lat_s2_gmi, lon_s2_gmi, options['ylim_min'], options['ylim_max'], options['xlim_min'],
                                options['xlim_max'])  
    lon  = lon_gmi.view() 
    lat  = lat_gmi.view()
    lon2 = lon_s2_gmi.view() 
    lat2 = lat_s2_gmi.view() 

    lon_gmi    = lon_gmi[inside] 
    lat_gmi    = lat_gmi[inside]
    tb_s1_gmi  = tb_s1_gmi[inside,:]
    tb_s2_gmi  = tb_s2_gmi[inside2,:]
    lon_s2_gmi = lon_s2_gmi[inside2] 
    lat_s2_gmi = lat_s2_gmi[inside2]      
                              
    plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)

    fontsize   = 12
    linewidths = 1
    
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
           c=tb_s1_gmi[:,5], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p1 = ax1.get_position().get_points().flatten()
    # For each polygon detected plot a polygon
    for i in range(len(pol_n)):
        coord = pol_n[i]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        ys, xs = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys, '-m') 
    plt.text(-75, -42, 'ACPs durante el dia', color='m')       


    
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
           c=tb_s1_gmi[:,7], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
    plt.title('BT 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(b)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p1 = ax1.get_position().get_points().flatten()
        
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
           c=tb_s2_gmi[:,0], s=10, vmin=50, vmax=300, cmap=cmaps['turbo_r'])  
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
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
        
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
        
    ax1.text(0.05,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    #ax1.text(-0.1,1.10,'(c)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')

    p2 = ax1.get_position().get_points().flatten()
    # #ax_cbar = fig.add_axes([p1[0], 0.45, p2[2]-p1[0], 0.05])
    ax_cbar = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im,  cax=ax_cbar, shrink=1,ticks=np.arange(50,325,25), 
                        extend='both', orientation="vertical")
    cbar.set_label('BT (K)', fontsize=fontsize)


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
           c=tb_s1_gmi[:,5]-tb_s1_gmi[:,6], s=10, vmin=0, vmax=16, cmap=PLOTS.discrete_cmap(16,  'rainbow'))  
    plt.title('GMI PD 37 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    CS = plt.contour(lon, lat, 
                        data_tb37,[250], colors=('r'), linewidths=(linewidths));
    labels = ["250K"]
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    #ax1.legend(loc='upper left', fontsize=fontsize)
    ax1.set_xlabel('Latitude', fontsize=fontsize)
    ax1.set_ylabel('Longitude', fontsize=fontsize)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(d)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p0 = ax1.get_position().get_points().flatten()

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
           c=tb_s1_gmi[:,7]-tb_s1_gmi[:,8], s=10, vmin=0, vmax=16, cmap=PLOTS.discrete_cmap(16,  'rainbow'))  
    plt.title('PD 89 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon, lat, data_tb89,[180,250], colors=('k','r'), linewidths=(linewidths));    
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(e)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
    p1 = ax1.get_position().get_points().flatten()


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
           c=tb_s2_gmi[:,0]-tb_s2_gmi[:,1], s=10, vmin=0, vmax=12, cmap=PLOTS.discrete_cmap(16,  'rainbow'))  
    #divider = make_axes_locatable(ax1)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #plt.colorbar(im, ax=ax1, shrink=1, ticks=np.arange(50,300,10), extend='both')
    plt.title('PD 166 GHz', fontsize=fontsize)
    ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon)
    ax1.set_yticks(np.arange(options['ylim_min'], options['ylim_max']+1,5), crs=crs_latlon)
    ax1.set_xticks(np.arange(options['xlim_min'], options['xlim_max']+1,5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    CS = plt.contour(lon2, lat2, 
                        data_tb166,[180,250], colors=('k','r'), linewidths=(linewidths));
    labels = ["180K","250K"]
    if len(CS.collections) == 1:
        for i in range(len(labels)-1):
            CS.collections[i].set_label("250K")        
    else:
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
    ax1.legend(loc='lower right', fontsize=10)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)                 
    ax1.text(0.05,1.10,'(f)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
        
       
    p2 = ax1.get_position().get_points().flatten()
    #ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.05])
    ax_cbar = fig.add_axes([0.92, 0.13, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=ax_cbar, shrink=1, ticks=np.arange(0,17,1), extend='both', 
                        orientation="vertical")
    cbar.set_label('PD (K)', fontsize=fontsize)
    
    fig.suptitle(options['title'] ,fontweight='bold' )
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.899)
    #plt.savefig(options['path']+'/'+options['name']+'.eps')
    plt.savefig(options['path']+'_plots'+'/'+options['name']+'_ALLGMICHANNELS.png')
    plt.close()
    
    return 
 
def apply_geofence_on_data(data, data_lat, data_lon, min_latitude, max_latitude, min_longitude,
                               max_longitude):

    data[data_lat < min_latitude]  = np.nan
    data[data_lat > max_latitude]  = np.nan
    data[data_lon > max_longitude] = np.nan
    data[data_lon < min_longitude] = np.nan

    return data

    
def plot_ku(lon, lat, zFactorMeasured, flagPrecip, options):
    
    """"""
        
    Kurpf_path = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/KURPF/'

    geo_reg_shp = '/Users/victoria.galligani/Work/Tools/Shapefiles/ne_50m_lakes/ne_50m_lakes.shp'
    geo_reg = shpreader.Reader(geo_reg_shp)
        
    cmaps = PLOTS.GMI_colormap()
    
    countries = shpreader.Reader('/Users/victoria.galligani/Work/Tools/Shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    
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

    ifile = options['ifile']
    yoi = int(ifile[22:26])
    moi = int(ifile[26:28]) #"%02d" % int(ifile[26:28])
    doi = int(ifile[28:30]) #"%02d" % int(ifile[28:30])
    hoi_window = [int(ifile[32:34]), int(ifile[40:42])]
    print('day of interest: '+str(doi) )
    
    hdf  = SD.SD(Kurpf_path + 'pf_'+ifile[22:26]+ifile[26:28]+'_level2.HDF')
    # Make dictionary and read the HDF file
    dsets = hdf.datasets()
    dsNames = sorted(dsets.keys())
    Kurpf_data = {}         
    for name in dsNames:
        # Get dataset instance
        ds  = hdf.select(name)
        data = ds.get()
        Kurpf_data[name] = data
    hdf.end()
   
    selectKurpf = np.logical_and( np.logical_and(np.logical_and( np.logical_and(Kurpf_data['LON'] >= options['xlim_min'], 
                                                 Kurpf_data['LON'] <= options['xlim_max']), 
                              np.logical_and(Kurpf_data['LAT'] >= options['ylim_min'], Kurpf_data['LAT'] <= options['ylim_max'])), 
                              Kurpf_data['DAY'] == int(ifile[28:30])), Kurpf_data['ORBIT'] == int(ifile[48:53])  )   
     
    pol_n, area_politica = PLOTS.read_acps_FULL(yoi, moi, doi) 


    #--------------------------------- caclular la transecta
    # keep domain of interest only by keeping those where the center nadir obs is inside domain
    inside   =  np.logical_and(np.logical_and(lon <= options['xlim_max_zoom'], lon >= options['xlim_min_zoom']),
                               np.logical_and(lat <= options['ylim_max_zoom'], lat >= options['ylim_min_zoom']))
    
    dataZ = zFactorMeasured.copy()
    dataZ = apply_geofence_on_data(dataZ, lat, lon, options['ylim_min_zoom'], options['ylim_max_zoom'], options['xlim_min_zoom'],
                               options['xlim_max_zoom'])  
    
    colmax     = np.nanmax(dataZ, axis=2)
    maxElement = np.amax(colmax[inside])
    
    if not maxElement == 0:       
        
        index1, index2      = np.where(colmax == maxElement)    
    
   
        height = 1000 * (np.arange(176,0,-1)*0.125)
        hh     = np.zeros((zFactorMeasured.shape[0], 176))
        lonlon = np.zeros((zFactorMeasured.shape[0], 176))
        for ij in range(zFactorMeasured.shape[0]):
            hh[ij,:] = height        
   
        for ij in range(zFactorMeasured.shape[0]):
            lonlon[ij,:] = lon[ij,index2]
        index1 = index1[0]

        xlim_min_zoom = lon[index1,20]-5
        xlim_max_zoom = lon[index1,20]+5
        ylim_min_zoom = lat[index1,20]-5
        ylim_max_zoom = lat[index1,20]+5
            
        
        options1 = { 'xlim_min_zoom': xlim_min_zoom, 'xlim_max_zoom': xlim_max_zoom, 
                     'ylim_min_zoom': ylim_min_zoom, 'ylim_max_zoom': ylim_max_zoom} 
                
        
        #---------------------------------     
        #- FIGURE
        plt.matplotlib.rc('font', family='DejaVu Sans', size = 12)
        fontsize   = 12
        linewidths = 1
        
        #--------------------------------- 
        fig = plt.figure(figsize=(12,7))     
        # COLMAX
        gs1 = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs1[0,0], projection=ccrs.PlateCarree())
        crs_latlon = ccrs.PlateCarree()
        ax1.set_extent([options1['xlim_min_zoom'], options1['xlim_max_zoom'], 
                        options1['ylim_min_zoom'], options1['ylim_max_zoom']], crs=crs_latlon)
        ax1.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax1.add_geometries( countries.geometries(), ccrs.PlateCarree(), 
                    edgecolor="black", facecolor='none')
        ax1.add_feature(states_provinces,linewidth=0.4)
        ax1.add_feature(rivers)
        ax1.add_geometries( geo_reg.geometries(), ccrs.PlateCarree(), \
                    edgecolor="black", facecolor='none')
        im = plt.scatter(lon[index1-1000:index1+1000,:], lat[index1-1000:index1+1000,:], 
               c=np.nanmax(zFactorMeasured[index1-1000:index1+1000,:,:], axis=2), s=10, vmin=0, vmax=55, cmap=get_miub_cmap())  
        plt.title('Ku colmax '+'('+ifile[22:30]+'.'+ifile[47:53]+')', fontsize=fontsize)
        ax1.gridlines(linewidth=0.2, linestyle='dotted', crs=crs_latlon) 
        ax1.text(0.05,1.10,'(a)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right')
        p1 = ax1.get_position().get_points().flatten()
        # For each polygon detected plot a polygon
        for i in range(len(pol_n)):
            coord = pol_n[i]
            coord.append(coord[0]) #repeat the first point to create a 'closed loop'
            ys, xs = zip(*coord) #create lists of x and y values
            plt.plot(xs,ys, '-m') 
        plt.text(-70, ylim_min_zoom-1, 'ACPs durante el dia', color='m')       
        cbar = fig.colorbar(im, shrink=0.8)
        cbar.set_label('zFactorMeasured (dBZ)', fontsize=fontsize)
        # AGREGAR LOS PFs detected
        #plt.plot(Kurpf_data['R_LON'][selectKurpf], Kurpf_data['R_LAT'][selectKurpf], 'x', color='darkblue')
        elems = np.where(Kurpf_data['R_LON'][selectKurpf]!=0)   # and use pf_data['r_lon'][select][j]    
        for j in elems[0]:
            u, v = get_ellipse_info(Kurpf_data, selectKurpf, j)
            plt.plot( u , v, 'k' )    #rotated ellipse
            plt.grid(color='lightgray',linestyle='--')     
            
        #- zoom over the area of interest? 
        ax1.set_yticks(np.arange(options1['ylim_min_zoom'], options1['ylim_max_zoom']+1,5), crs=crs_latlon)
        ax1.set_xticks(np.arange(options1['xlim_min_zoom'], options1['xlim_max_zoom']+1,5), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)                
        #Poner los limites
        plt.plot(lon[:,0], lat[:,0], '-k')
        plt.plot(lon[:,48], lat[:,48], '-k')
        plt.plot(lon[:,index2], lat[:,index2], '--k')
        
        #--------------------------------------------------------
        # corte para mostrar en funcion de altura
        ax1 = plt.subplot(gs1[0,1])
        im = plt.scatter(lonlon[index1-1000:index1+1000,:], hh[index1-1000:index1+1000,:]/1000, 
               c=np.squeeze(zFactorMeasured[index1-1000:index1+1000,index2,:]), s=10, vmin=0, vmax=55, cmap=get_miub_cmap())  
        plt.title('DPR Ku '+'('+ifile[22:30]+'.orbit: '+ifile[47:53]+')', fontsize=fontsize)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Height')
        ax1.text(0.05,1.10,'(b)',transform=ax1.transAxes,fontsize=fontsize,va='top',ha='right') 
        cbar = fig.colorbar(im)
        cbar.set_label('zFactorMeasured (dBZ)', fontsize=fontsize)
        ax1.set_xlim([options1['xlim_min_zoom'],options1['xlim_max_zoom']])
        ax1.set_ylim([0,20])
        
        #fig.suptitle(options['title'] ,fontweight='bold' )
        #plt.tight_layout()
        #plt.subplots_adjust(top=0.899)
        #plt.savefig(options['path']+'/'+options['name']+'.eps')
        plt.savefig(options['path']+'_plots'+'/'+options['ifile']+'_DPRplots.png')
        #plt.close()
    
    return    
        
##############################################################################

if __name__ == '__main__':

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 12)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12

    # Domain of interest
    xlim_min = -75; 
    xlim_max = -50; 
    ylim_min = -40; 
    ylim_max = -19; 
    darea_do =  (ylim_max-ylim_min)*(xlim_max-xlim_min) # "in degrees"
    
    opts = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
            'ylim_min': ylim_min, 'ylim_max': ylim_max}
    
    ##############################################################################
    prov = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/home/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')
    
    mhs_dir     = '/home/victoria.galligani/datosmunin2/DATOS_mw/MHS/'
    ssmis_dir   = '/home/victoria.galligani/datosmunin2/DATOS_mw/SSMIS/'
    atms_dir    = '/home/victoria.galligani/datosmunin2/DATOS_mw/ATMS/'
    gmi_dir     = '/home/victoria.galligani/datosmunin2/DATOS_mw/GMI/'
    atms_dir    = '/home/victoria.galligani/datosmunin2/DATOS_mw/ATMS/'
    amsr2_dir   = '/datosmunin2/victoria.galligani/DATOS_mw/AMSR2/'
    
    test      = '1C.METOPB.MHS.XCAL2016-V.20181220-S002558-E020718.032453.V05A.HDF5'
    onlyfiles = [f for f in listdir(mhs_dir) if isfile(join(mhs_dir, f))]
    
    run_this = 0
    if run_this == 1: 
        # For each orbit, we keep only the domain of interest, and calculate the fraction 
        # of the domain of interest that is covered by the radiometer. We also keep the 
        # central time and doy (since we are looking at only 2018). By central time is the 
        # time of observation at the middle of the swath inside the domain of interest. 
        # hour_central is the average time (HH only) for the lon(:,45) and lat(:,45) inside the domain
                    
        #- MHS
        mhs_metopB_m, mhs_metopB_d, mhs_metopB_h, mhs_metopB_min, mhs_metopB_doy, mhs_metopB_coverage, metopB_hour_central = get_mhs(mhs_dir, onlyfiles, '1C.METOPB')
        mhs_metopA_m, mhs_metopA_d, mhs_metopA_h, mhs_metopA_min, mhs_metopA_doy, mhs_metopA_coverage, metopA_hour_central = get_mhs(mhs_dir, onlyfiles, '1C.METOPA')
        mhs_noaa19_m, mhs_noaa19_d, mhs_noaa19_h, mhs_noaa19_min, mhs_noaa19_doy, mhs_noaa19_coverage, noaa19_hour_central = get_mhs(mhs_dir, onlyfiles, '1C.NOAA19')
        
        #- SSMIS (using get_mhs S1 only, but there is obviously S2, S3, S4 - OJO)
        onlyfiles = [f for f in listdir(ssmis_dir) if isfile(join(ssmis_dir, f))]
        ssmis_F18_m, ssmis_F18_d, ssmis_F18_h, ssmis_F18_min, ssmis_F18_doy, ssmis_F18_coverage, F18_hour_central = get_mhs(ssmis_dir, onlyfiles, '1C.F18')
        ssmis_F17_m, ssmis_F17_d, ssmis_F17_h, ssmis_F17_min, ssmis_F17_doy, ssmis_F17_coverage, F17_hour_central = get_mhs(ssmis_dir, onlyfiles, '1C.F17')
        ssmis_F16_m, ssmis_F16_d, ssmis_F16_h, ssmis_F16_min, ssmis_F16_doy, ssmis_F16_coverage, F16_hour_central = get_mhs(ssmis_dir, onlyfiles, '1C.F16')
        
        #- ATMS (using get_mhs S1 only, but there is obviously S2, S3, S4 - OJO - same as SSMIS above)
        onlyfiles = [f for f in listdir(atms_dir) if isfile(join(atms_dir, f))]
        atms_noaa20_m, atms_noaa20_d, atms_noaa20_h, atms_noaa20_min, atms_noaa20_doy, atms_noaa20_coverage, atms_noaa20_hour_central = get_mhs(atms_dir, onlyfiles, '1C.NOAA20')
        atms_npp_m, atms_npp_d, atms_npp_h, atms_npp_min, atms_npp_doy, atms_npp_coverage, atms_npp_hour_central = get_mhs(atms_dir, onlyfiles, '1C.NPP')
        
        #- GMI (using get_mhs S1 only, but there is obviously S2)
        onlyfiles = [f for f in listdir(gmi_dir) if isfile(join(gmi_dir, f))]
        gmi_m, gmi_d, gmi_h, gmi_min, gmi_doy, gmi_coverage, gmi_hour_central = get_mhs(gmi_dir, onlyfiles, '1B.GPM.GMI')

        #- AMSR2 
        onlyfiles = [f for f in listdir(amsr2_dir) if isfile(join(amsr2_dir, f))]
        amsr2_m, amsr2_d, amsr2_h, amsr2_min, amsr2_doy, amsr2_coverage, amsr2_hour_central = get_mhs(amsr2_dir, onlyfiles, '1C.GCOMW1.AMSR2')
            

            
    #--------------------------------------------------------------------------
    #                               FIGURES 
    #--------------------------------------------------------------------------    

    #====== PLOT HISTOGRAM OF TEMPORAL COVERAGE
    dir_name = '/home/victoria.galligani/datosmunin2/DATOS_mw/Plots/'
    filename = 'histograms_GPMconstellation.png'

    metopA_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/mhs_metopA.nc')
    metopB_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/mhs_metopB.nc')
    noaa19_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/mhs_noaa19.nc')
    F16_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/ssmis_F16.nc')
    F17_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/ssmis_F17.nc')
    F18_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/ssmis_F18.nc')
    atms_noaa20_hour_central = nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/atms_noaa20.nc')
    atms_npp_hour_central    =  nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/atms_npp.nc')
    gmi_hour_central =  nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/S1_domain_1B.GPM.GMI.nc')
    amsr2_hour_central =  nc.Dataset('/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/S1_domain_1C.GCOMW1.AMSR2.nc')
  
    plot_histogram(metopA_hour_central['hour_central'], 
                   metopB_hour_central['hour_central'], 
                   noaa19_hour_central['hour_central'], 
                   F16_hour_central['hour_central'], F17_hour_central['hour_central'],
                   F18_hour_central['hour_central'], atms_noaa20_hour_central['hour_central'],
                   atms_npp_hour_central['hour_central'], gmi_hour_central['hour_central'],amsr2_hour_central['hour_central'], 
                   '/Users/victoria.galligani/'+'check2.png')

    #====== PLOT TOP 0.01 % PERCENTILES SCENES 
    gmi_folder_files = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/top_percentiles_GMIorbits'
    files = listdir(gmi_folder_files)
    for i in files:
        # full orbit
        f = h5py.File( gmi_folder_files+'/'+i, 'r')      
        lon_gmi = f[u'/S1/Longitude'][:,:] 
        lat_gmi = f[u'/S1/Latitude'][:,:]
        tb_s1_gmi = f[u'/S1/Tc'][:,:,:]           
        tb_s2_gmi = f[u'/S2/Tc'][:,:,:]           
        lon_s2_gmi = f[u'/S2/Longitude'][:,:] 
        lat_s2_gmi = f[u'/S2/Latitude'][:,:]
        f.close()    
        # PLOT
        opts2 = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
                 'ylim_min': ylim_min, 'ylim_max': ylim_max,
                 'title': str(i), 'path': gmi_folder_files,
                 'name':  str(i)+'_PLOT_MASKED', 'ifile': i} 
        plot_GMI(lon_gmi, lat_gmi, tb_s1_gmi, tb_s2_gmi, 
             lon_s2_gmi, lat_s2_gmi, opts2)
    

    ku_folder_files = '/Users/victoria.galligani/Work/Data/RELAMPAGO_GPM/top_percentiles_DPRorbits'
    files = listdir(ku_folder_files)
    for filei in files:
        f = h5py.File( ku_folder_files+'/'+filei, 'r')    
        Z = f[u'/NS/SLV/zFactorCorrected'][:,:,:]           
        flagPrecip      = f[u'/NS/PRE/flagPrecip'][:,:]
        lon             = f[u'/NS/Longitude'][:,:]
        lat             = f[u'/NS/Latitude'][:,:]
        f.close()    
        
        xlim_max_zoom = -50
        xlim_min_zoom = -70
        ylim_max_zoom = -20
        ylim_min_zoom = -40
        
        options = {'xlim_min': xlim_min, 'xlim_max': xlim_max, 
                 'ylim_min': ylim_min, 'ylim_max': ylim_max,
                 'title': str(i), 'path': ku_folder_files,
                 'name':  str(i)+'_PLOT_MASKED', 'ifile': filei, 
                 'xlim_min_zoom': xlim_min_zoom, 'xlim_max_zoom': xlim_max_zoom, 
                 'ylim_min_zoom': ylim_min_zoom, 'ylim_max_zoom': ylim_max_zoom} 
        
        plot_ku(lon, lat, Z, flagPrecip, options)


def get_miub_cmap():
    import matplotlib.colors as col
    startcolor = 'white'  # a dark olive
    color1 = '#8ec7ff'#'cyan'    # a bright yellow
    color2 = 'dodgerblue'
    color3 = 'lime'
    color4 = 'yellow'
    color5 = 'darkorange'
    color6 = 'red'
    color7 = 'purple'
    #color6 = 'grey'
    endcolor = 'darkmagenta'    # medium dark red
    colors = [startcolor, color1, color2, color3, color4, color5, color6, endcolor]
    return col.LinearSegmentedColormap.from_list('miub1',colors)
    
