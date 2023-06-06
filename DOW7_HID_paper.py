#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:24:03 2023

@author: vgalligani
"""
from matplotlib import cm;
from os import listdir
import pyart
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
import matplotlib
import matplotlib.gridspec as gridspec
import platform
from matplotlib.colors import ListedColormap
from os.path import isfile, join
import pandas as pd
from copy import deepcopy
from pyart.correct import phase_proc
import xarray as xr
import matplotlib.colors as colors
import wradlib as wrl
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from cycler import cycler
#import seaborn as sns
import cartopy.io.shapereader as shpreader
import copy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gc
import math
from pyart.core.transforms import antenna_to_cartesian
from collections import Counter
import alphashape
from descartes import PolygonPatch
import cartopy.feature as cfeature
from matplotlib.path import Path

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import h5py
from csu_radartools import csu_fhc

plt.matplotlib.rc('font', family='serif', size = 12)
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

#-----------------------------------------------------------------------------
def calc_freezinglevel(era5_dir, era5_file, lat_pf, lon_pf):

    ERA5_field = xr.load_dataset(era5_dir+era5_file, engine="cfgrib")
    elemj      = find_nearest(ERA5_field['latitude'], lat_pf[0])
    elemk      = find_nearest(ERA5_field['longitude'], lon_pf[0])
    tfield_ref = ERA5_field['t'][:,elemj,elemk] - 273 # convert to C
    geoph_ref  = (ERA5_field['z'][:,elemj,elemk])/9.80665
    # Covert to geop. height (https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
    Re         = 6371*1e3
    alt_ref    = (Re*geoph_ref)/(Re-geoph_ref)
    freezing_lev = np.array(alt_ref[find_nearest(tfield_ref, 0)]/1e3)

    return alt_ref, tfield_ref, freezing_lev

#------------------------------------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
    elif 'DBZHCC' in radar.fields.keys():
        dz_field='DBZHCC'
    elif 'reflectivity' in radar.fields.keys():
        dz_field='reflectivity'
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

def add_43prop_field(radar):

    radar_height = get_z_from_radar(radar)
    radar = add_field_to_radar_object(radar_height, radar, field_name = 'height')
    iso0 = np.ma.mean(radar.fields['height']['data'][np.where(np.abs(radar.fields['sounding_temperature']['data']) < 0)])
    radar.fields['height_over_iso0'] = deepcopy(radar.fields['height'])
    radar.fields['height_over_iso0']['data'] -= iso0

    return radar

#------------------------------------------------------------------------------
def stack_ppis(radar, files_list, options, freezing_lev, radar_T, tfield_ref, alt_ref):

    #- HERE MAKE PPIS SIMILAR TO RMA1S ... ? to achive the gridded field ...
    #- Radar sweep
    lats0        = radar.gate_latitude['data']
    lons0        = radar.gate_longitude['data']
    azimuths     = radar.azimuth['data']
    #
    Ze     = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); Ze[:]=np.nan
    ZDR    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); ZDR[:]=np.nan
    lon    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); lon[:]=np.nan
    lat    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); lat[:]=np.nan
    RHO    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); RHO[:]=np.nan
    PHIDP  = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); PHIDP[:]=np.nan
    HID    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); HID[:]=np.nan
    KDP    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); KDP[:]=np.nan
    approx_altitude = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); approx_altitude[:]=np.nan
    #gate_range      = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); gate_range[:]=np.nan
    alt_43aproox    = np.zeros( [len(files_list), lats0.shape[0], lats0.shape[1] ]); alt_43aproox[:]=np.nan
    #
    gate_range      = np.zeros( [len(files_list), lats0.shape[1] ]); gate_range[:]=np.nan
    azy   = np.zeros( [len(files_list), lats0.shape[0] ]); azy[:]=np.nan
    fixed_angle     = np.zeros( [len(files_list)] ); fixed_angle[:]=np.nan
    #
    nlev = 0
    for file in files_list:
        print(file)
        if 'low_v176' in file:
            radar   = pyart.io.read('/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/DOW7/'+file)
            fixed_angle[nlev] = radar.fixed_angle['data'].data[0]
            azy[nlev,:]  = radar.azimuth['data']
            ZHZH    = radar.fields['DBZHCC']['data']
            TV      = radar.fields['DBZVCC']['data']
            ZDRZDR  = radar.fields['ZDRC']['data']
            RHORHO  = radar.fields['RHOHV']['data']
            KDPKDP  = radar.fields['KDP']['data']
            PHIPHI  = radar.fields['PHIDP']['data']
            #
            radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
            radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')
            dzh_  = radar.fields['DBZHCC']['data'].copy()
            dzv_  = radar.fields['DBZVCC']['data'].copy()
            dZDR  = radar.fields['ZDRC']['data'].copy()
            drho_ = radar.fields['RHOHV']['data'].copy()
            dkdp_ = radar.fields['KDP']['data'].copy()
            dphi_ = radar.fields['PHIDP']['data'].copy()
            # Filters
            ni = dzh_.shape[0]
            nj = dzh_.shape[1]
            for i in range(ni):
                rho_h = drho_[i,:]
                zh_h = dzh_[i,:]
                for j in range(nj):
                    if (rho_h[j]<0.7) or (zh_h[j]<30):
                        dzh_[i,j]  = np.nan
                        dzv_[i,j]  = np.nan
                        drho_[i,j]  = np.nan
                        dkdp_[i,j]  = np.nan
                        dphi_[i,j]  = np.nan
            scores = csu_fhc.csu_fhc_summer(dz=dzh_, zdr=dZDR - options['ZDRoffset'], rho=drho_, kdp=dkdp_, use_temp=True, band='C', T=radar_T)
            HIDHID = np.argmax(scores, axis=0) + 1
        
            gateZ    = radar.gate_z['data']
            gateX    = radar.gate_x['data']
            gateY    = radar.gate_y['data']
            gates_range  = np.sqrt(gateX**2 + gateY**2 + gateZ**2)
            #
            lats        = radar.gate_latitude['data']
            lons        = radar.gate_longitude['data']

            #------- aca hay que stack correctamente por azimuths?
            for TransectNo in range(lats0.shape[0]):
                [xgate, ygate, zgate]    = pyart.core.antenna_to_cartesian(gates_range[TransectNo,:]/1e3, azimuths[TransectNo], np.double(    file[41:45]) );
                Ze[nlev,TransectNo,:]    = dzh_[TransectNo,:]
                ZDR[nlev,TransectNo,:]   = dZDR[TransectNo,:]
                RHO[nlev,TransectNo,:]   = drho_[TransectNo,:]
                PHIDP[nlev,TransectNo,:] = dphi_[TransectNo,:]
                HID[nlev,TransectNo,:]   = HIDHID[TransectNo,:]
                KDP[nlev,TransectNo,:]   = dkdp_[TransectNo,:]
                lon[nlev,TransectNo,:]   = lons[TransectNo,:]
                lat[nlev,TransectNo,:]   = lats[TransectNo,:]

            gc.collect()
            nlev = nlev + 1


    # From https://arm-doe.github.io/pyart/notebooks/basic_ingest_using_test_radar_object.html
    radar_stack = pyart.testing.make_empty_ppi_radar(lat.shape[2], lat.shape[1], lat.shape[0])
    # Start filling the radar attributes with variables in the dataset.
    radar_stack.latitude['data']    = np.array([radar.latitude['data'][0]])
    radar_stack.longitude['data']   = np.array([radar.longitude['data'][0]])
    radar_stack.range['data']       = np.array( radar.range['data'][:] )
    radar_stack.fixed_angle['data'] = np.array( fixed_angle )
    azi_all = []
    rays_per_sweep = []
    raye_per_sweep = []
    rayN = 0
    elev = np.zeros( [azy.shape[0]*azy.shape[1]] ); elev[:]=np.nan
    for i in range(azy.shape[0]):
        rays_per_sweep.append(rayN)
        rayN = rayN + azy.shape[1]
        raye_per_sweep.append(rayN-1)
        for j in range(azy.shape[1]):
            azi_all.append(  azy[i,j] )
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(azy.shape[1]):
            elev[ii] = fixed_angle[i]
            ii=ii+1
    radar_stack.azimuth['data'] = np.array( azi_all )
    radar_stack.sweep_number['data'] = np.array(  np.arange(0,nlev,1) )
    radar_stack.sweep_start_ray_index['data'] = np.array( rays_per_sweep )
    radar_stack.sweep_end_ray_index['data']   = np.array( raye_per_sweep )
    radar_stack.altitude['data']        = [ radar.altitude['data'][0] ]
    # elevation is theta too.
    radar_stack.elevation['data'] = elev
    radar_stack.init_gate_altitude()
    radar_stack.init_gate_longitude_latitude()

    #plt.plot(radar_stack.gate_longitude['data'], radar_stack.gate_latitude['data'], 'ok')
    # Let's work on the field data, we will just do reflectivity for now, but any of the
    # other fields can be done the same way and added as a key pair in the fields dict.
    from pyart.config import get_metadata
    Ze_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); Ze_all[:]=np.nan
    ZDR_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); ZDR_all[:]=np.nan
    RHOHV_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); RHOHV_all[:]=np.nan
    PHIDP_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); PHIDP_all[:]=np.nan
    KDP_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); KDP_all[:]=np.nan
    HID_all = np.zeros( [azy.shape[0]*azy.shape[1], Ze.shape[2]] ); HID_all[:]=np.nan
    ii = 0
    for i in range(azy.shape[0]):
        for j in range(Ze.shape[1]):
            Ze_all[ii,:]    = Ze[i,j,:]
            ZDR_all[ii,:]   = ZDR[i,j,:]
            RHOHV_all[ii,:] = RHO[i,j,:]
            PHIDP_all[ii,:] = PHIDP[i,j,:]
            KDP_all[ii,:] = KDP[i,j,:]
            HID_all[ii,:] = HID[i,j,:]
            ii=ii+1

    #- REFLECTIVITY
    ref_dict_ZH = get_metadata('DBZHCC')
    ref_dict_ZH['data'] = np.array(Ze_all)

    #- ZDR
    ref_dict_ZDR = get_metadata('ZDRC')
    ref_dict_ZDR['data'] = np.array(ZDR_all)

    #- RHOHV
    ref_dict_RHOHV = get_metadata('RHOHV')
    ref_dict_RHOHV['data'] = np.array(RHOHV_all)

    #- PHIDP
    ref_dict_PHIDP = get_metadata('PHIDP')
    ref_dict_PHIDP['data'] = np.array(PHIDP_all)

    #- KDP
    ref_dict_KDP = get_metadata('KDP')
    ref_dict_KDP['data'] = np.array(KDP_all)

    #- HID
    ref_dict_HID = get_metadata('HID')
    ref_dict_HID['data'] = np.array(HID_all)

    radar_stack.fields = {'DBZHCC': ref_dict_ZH,
              'ZDRC':   ref_dict_ZDR,
              'RHOHV':  ref_dict_RHOHV,
              'PHIDP':  ref_dict_PHIDP,
              'KDP':    ref_dict_KDP,
              'HID':    ref_dict_HID}



    return radar_stack

#------------------------------------------------------------------------------
def make_pseudoRHISfromGrid_DOW7(gridded_radar, radar, azi_oi, titlecois, xlims_xlims_mins, xlims_xlims, alt_ref, tfield_ref, options): 

    plt.matplotlib.rc('font', family='serif', size = 20)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['font.serif'] = ['Helvetica']
    
    THname = 'DBZHCC' 
    TVname = 'ZDRC'
    
    start_index = radar.sweep_start_ray_index['data']
    end_index   = radar.sweep_end_ray_index['data']
    lats        = radar.gate_latitude['data']
    lons        = radar.gate_longitude['data']
    azimuths    = radar.azimuth['data']

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=[13,12])

    for iz in range(len(azi_oi)):
        target_azimuth = azimuths[azi_oi[iz]]
        filas = np.asarray(abs(azimuths-target_azimuth)<=0.1).nonzero()	 # <==== ?????
        grid_lon   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lon[:]   = np.nan
        grid_lat   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_lat[:]   = np.nan
        grid_THTH  = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_THTH[:]  = np.nan
        grid_ZDR   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_ZDR[:]  = np.nan
        grid_alt   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_alt[:]   = np.nan
        grid_range = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_range[:] = np.nan
        grid_RHO   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_RHO[:]   = np.nan
        grid_HID   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_HID[:]   = np.nan
        grid_KDP   = np.zeros((gridded_radar.fields[THname]['data'].shape[0], lons[filas,:].shape[2])); grid_KDP[:]  = np.nan

        # need to find x/y pair for each gate at the surface 
        for i in range(lons[filas,:].shape[2]):	
            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = np.abs(gridded_radar.point_latitude['data'][0,:,:]  - lats[filas,i])
            abslon = np.abs(gridded_radar.point_longitude['data'][0,:,:] - lons[filas,i])
            c = np.maximum(abslon, abslat)
            ([xloc], [yloc]) = np.where(c == np.min(c))	
            grid_lon[:,i]   = gridded_radar.point_longitude['data'][:,xloc,yloc]
            grid_lat[:,i]   = gridded_radar.point_latitude['data'][:,xloc,yloc]
            grid_ZDR[:,i]   = gridded_radar.fields[TVname]['data'][:,xloc,yloc]
            grid_THTH[:,i]  = gridded_radar.fields[THname]['data'][:,xloc,yloc]
            grid_RHO[:,i]   = gridded_radar.fields['RHOHV']['data'][:,xloc,yloc]
            grid_alt[:,i]   = gridded_radar.z['data'][:]
            x               = gridded_radar.point_x['data'][:,xloc,yloc]
            y               = gridded_radar.point_y['data'][:,xloc,yloc]
            z               = gridded_radar.point_z['data'][:,xloc,yloc]
            grid_range[:,i] = ( x**2 + y**2 + z**2 ) ** 0.5
            grid_KDP[:,i]   = gridded_radar.fields['KDP']['data'][:,xloc,yloc]
            grid_HID[:,i]   = gridded_radar.fields['HID']['data'][:,xloc,yloc]
	
        ni = grid_HID.shape[0]
        nj = grid_HID.shape[1]
        for i in range(ni):
            rho_h = grid_RHO[i,:]
            zh_h = grid_THTH[i,:]
            for j in range(nj):
                if (rho_h[j]<0.7) or (zh_h[j]<0):
                    grid_THTH[i,j] = np.nan
                    grid_ZDR[i,j]  = np.nan
                    grid_RHO[i,j]  = np.nan			
				
				
        #---- plot hid ppi  
        hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
        cmaphid = colors.ListedColormap(hid_colors)
        #cmaphid.set_bad('white')
        #cmaphid.set_under('white')
        # Figure
        [units, cmap, vmin, vmax, max, intt, under, over] = set_plot_settings('Zhh')
        im_TH  = axes[0,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_THTH, cmap=cmap, vmin=vmin, vmax=vmax)

        im_ZDR = axes[1,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_ZDR, cmap=discrete_cmap(int(5+2), 'jet') , vmin=-2, vmax=5)

        im_RHO = axes[2,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_RHO, cmap=pyart.graph.cm.RefDiff , vmin=0.7, vmax=1.)

        im_HID = axes[3,iz].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)

        axes[0,iz].set_title('coi='+titlecois[iz]) 
        if iz == 1:
            axes[0,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[1,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[2,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
            axes[3,iz].set_xlim([xlims_xlims_mins[1],xlims_xlims[1]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == 2:
            axes[0,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[1,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[2,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
            axes[3,iz].set_xlim([xlims_xlims_mins[2],xlims_xlims[2]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == 3:
            axes[0,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[1,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[2,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
            axes[3,iz].set_xlim([xlims_xlims_mins[3],xlims_xlims[3]])
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
		
        if iz == 0:
            axes[0,0].set_ylabel('Altitude (km)')
            axes[1,0].set_ylabel('Altitude (km)')
            axes[2,0].set_ylabel('Altitude (km)')
            axes[3,0].set_ylabel('Altitude (km)')
            axes[0,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[1,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[2,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,iz].set_xlim([xlims_xlims_mins[0],xlims_xlims[0]])
            axes[3,0].set_xlabel('Range (km)')
       	    axes[0,iz].set_ylim([0,15])
       	    axes[1,iz].set_ylim([0,15])
            axes[2,iz].set_ylim([0,15])
            axes[3,iz].set_ylim([0,15])
	
        if iz == len(azi_oi)-1: 
	    # Add colorbars #ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            pm1    = axes[0,iz-1].get_position().get_points().flatten()
            p_last = axes[0,iz].get_position().get_points().flatten(); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.76, 0.02, 0.2])  
            cbar    = fig.colorbar(im_TH,  cax=ax_cbar, shrink=0.9,  ticks=np.arange(0,60.01,10), label='ZH')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            pm2    = axes[1,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.55, 0.02, 0.2])  
            cbar    = fig.colorbar(im_ZDR, cax=ax_cbar, shrink=0.9,  ticks=np.arange(-2.,5.01,1.), label='ZDR')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            pm3   = axes[2,iz-1].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.28, 0.02, 0.2])  
            cbar    = fig.colorbar(im_RHO, cax=ax_cbar, shrink=0.9, ticks=np.arange(0.7,1.01,0.1), label='RHO')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 

            ax_cbar = fig.add_axes([p_last[0]+(p_last[0]-pm1[0])+0.08, 0.03, 0.02, 0.2])  
            cbar    = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')#, ticks=np.arange(0,np.round(VMAXX,2)+0.02,0.01)); 
            cbar = adjust_fhc_colorbar_for_pyart(cbar)
            #cbar.cmap.set_under('white')

            pm2    = axes[3,iz-1].get_position().get_points().flatten()

		
    #- savefile
    fig.savefig(options['fig_dir']+'PseudoRHIS_GRIDDED'+'.png', dpi=300,transparent=False)   
    #plt.close()
	
    #-------------------------------
    for i in range(20):
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[13,12])
        im_HID = ax.pcolormesh(gridded_radar.point_longitude['data'][i,:,:], 
        gridded_radar.point_latitude['data'][i,:,:], gridded_radar.fields['HID']['data'][i,:,:], cmap=cmaphid, vmin=0.2, vmax=10)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],10)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],50)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)
        [lat_radius, lon_radius] = pyplot_rings(radar.latitude['data'][0],radar.longitude['data'][0],100)
        ax.plot(lon_radius, lat_radius, 'k', linewidth=0.8)  
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('HID '+ str( round(grid_alt[i,0]/1e3,1)) +' km')
        cbar = fig.colorbar(im_HID,  cax=ax_cbar, shrink=0.9, label='HID')
        cbar = adjust_fhc_colorbar_for_pyart(cbar)
        ax.set_xlim([options['xlim_min'], options['xlim_max']])	
        ax.set_ylim([options['ylim_min'], options['ylim_max']])
        fig.savefig(options['fig_dir']+'RHIS_GRIDDED_verticalLevel_'+str(i)+'.png', dpi=300,transparent=False); plt.close()
    return grid_HID, grid_lon, grid_lat, grid_range, grid_alt


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
        vmax = 4
        max = 4.1
        intt = 0.1
        N = (vmax-vmin)/intt
        cmap = pyart.graph.cm.Theodore16 #discrete_cmap(10, 'jet')
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
        cmap = pyart.graph.cm.Wild25  #discrete_cmap(int(N), 'jet')
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
    
    
    cb.set_ticks(np.arange(0.4, 10, 0.9))
    cb.ax.set_yticklabels(['','Drizzle','Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

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
def adjust_meth_colorbar_for_pyart(cb, tropical=False):
    if not tropical:
        cb.set_ticks(np.arange(1.25, 5, 0.833))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z)', 'R(Zrain)'])
    else:
        cb.set_ticks(np.arange(1.3, 6, 0.85))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z_all)', 'R(Z_c)', 'R(Z_s)'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

#------------------------------------------------------------------------------
# SOLO DOW7 HID PLOT FOR PAPER

files_list = ['cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc',
          'cfrad.20181214_022019_DOW7low_v176_s02_el1.98_SUR.nc',
          'cfrad.20181214_022031_DOW7low_v176_s03_el3.97_SUR.nc',
          'cfrad.20181214_022043_DOW7low_v176_s04_el5.98_SUR.nc',
          'cfrad.20181214_022055_DOW7low_v176_s05_el7.98_SUR.nc',
          'cfrad.20181214_022106_DOW7low_v176_s06_el9.98_SUR.nc',
          'cfrad.20181214_022118_DOW7low_v176_s07_el11.98_SUR.nc',
          'cfrad.20181214_022130_DOW7low_v176_s08_el13.99_SUR.nc',
          'cfrad.20181214_022142_DOW7low_v176_s09_el15.97_SUR.nc',
          'cfrad.20181214_022154_DOW7low_v176_s10_el17.97_SUR.nc',
          'cfrad.20181214_022206_DOW7low_v176_s11_el19.98_SUR.nc',
          'cfrad.20181214_022218_DOW7low_v176_s12_el21.98_SUR.nc',
          'cfrad.20181214_022230_DOW7low_v176_s13_el23.99_SUR.nc',
          'cfrad.20181214_022241_DOW7low_v176_s14_el25.98_SUR.nc',
          'cfrad.20181214_022253_DOW7low_v176_s15_el27.98_SUR.nc',
          'cfrad.20181214_022305_DOW7low_v176_s16_el29.98_SUR.nc',
          'cfrad.20181214_022317_DOW7low_v176_s17_el31.98_SUR.nc',
          'cfrad.20181214_022329_DOW7low_v176_s18_el33.98_SUR.nc',
          'cfrad.20181214_022341_DOW7low_v176_s19_el36.98_SUR.nc',
          'cfrad.20181214_022353_DOW7low_v176_s20_el40.97_SUR.nc',
          'cfrad.20181214_022405_DOW7low_v176_s21_el44.98_SUR.nc',
          'cfrad.20181214_022416_DOW7low_v176_s22_el49.98_SUR.nc']


gmi_dir  = '/home/victoria.galligani/Work/Studies/Hail_MW/GMI_data/'
era5_dir = '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/'
lon_pfs  = [-63.11] # [-61.40] [-59.65]
lat_pfs  = [-31.90] # [-32.30] [-33.90]
phail    = [0.967] # [0.998] [0.863]
# USE DOW7 for lowest level
rfile = 'cfrad.20181214_022007_DOW7low_v176_s01_el0.77_SUR.nc'
gfile     = '1B.GPM.GMI.TB2016.20181214-S015009-E032242.027231.V05A.HDF5'
era5_file = '20181214_03_RMA1.grib'
reportes_granizo_twitterAPI_geo = [[-32.19, -64.57],[-32.07, -64.54]]
reportes_granizo_twitterAPI_meta = [['0320UTC'],['0100']]
opts = {'xlim_min': -65.3, 'xlim_max': -63.3, 'ylim_min': -32.4, 'ylim_max': -31, 'ZDRoffset': 0, 'caso':'20181214',
        'rfile': 'DOW7/'+rfile, 'gfile': gfile, 'azimuth_ray': 0,
         'radar_name':'DOW7', 'era5_file': era5_file,'alternate_azi':[30], 'ZDRoffset': 0,
         'fig_dir':'/home/victoria.galligani/Work/Studies/Hail_MW/Figures/Caso_20181214_RMA1/',
         'REPORTES_geo': reportes_granizo_twitterAPI_geo, 'REPORTES_meta': reportes_granizo_twitterAPI_meta, 'gmi_dir':gmi_dir,
         'lat_pfs':lat_pfs, 'lon_pfs':lon_pfs, 'MINPCTs_labels':[],'MINPCTs':[], 'phail': phail,
       'icoi_PHAIL': [15], 'files_list':files_list}
icois_input  = [15]
labels_PHAIL = ['Phail=X%']
xlims_xlims_input  = [80]
xlims_mins_input  = [20]


r_dir    = '/home/victoria.galligani/Work/Studies/Hail_MW/radar_data/'
radar = pyart.io.read(r_dir+opts['rfile'])
gc.collect()

alt_ref, tfield_ref, freezing_lev =  calc_freezinglevel( '/home/victoria.galligani/Work/Studies/Hail_MW/ERA5/',opts['era5_file'],opts['lat_pfs'], opts['lon_pfs'])
radar_T,radar_z =  interpolate_sounding_to_radar(tfield_ref, alt_ref, radar)
radar = add_field_to_radar_object(radar_T, radar, field_name='sounding_temperature')
radar = stack_ppis(radar, opts['files_list'], opts, freezing_lev, radar_T, tfield_ref, alt_ref)
grided  = pyart.map.grid_from_radars(radar, grid_shape=(41, 355, 355), grid_limits=((0.,20000,),   #20,470,470 is for 1km
              (-np.max(radar.range['data']), np.max(radar.range['data'])),(-np.max(radar.range['data']), np.max(radar.range['data']))),
                     roi_func='dist', min_radius=100.0, weighting_function='BARNES2')
gc.collect()
#------------------------------------------------------------------------------
rad_T1 = np.interp(grided.point_z['data'].ravel(), alt_ref, tfield_ref)
rad_T1 = np.reshape(rad_T1, np.shape(grided.point_z['data']))
field_dict = {'data': rad_T1,
              'units': 'k',
              'long_name': 'sounding_temperature',
              'standard_name': 'sounding_temperature',
              '_FillValue': -999}
grided.add_field('sounding_temperature', field_dict, replace_existing=True)
gc.collect()
#------------------------------------------------------------------------------
radar_base = pyart.io.read(r_dir+opts['rfile'])
grid_HID, grid_lon, grid_lat, grid_range, grid_alt = make_pseudoRHISfromGrid_DOW7(grided, radar_base, [273], labels_PHAIL, xlims_mins_input, xlims_xlims_input, alt_ref, tfield_ref, opts)
#------------------------------------------------------------------------------
# NOW SUBPLOT FOR THE FIGURE IN THE PPT:
hid_colors = ['White', 'LightBlue','MediumBlue', 'DarkOrange', 'LightPink',
                'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
cmaphid = colors.ListedColormap(hid_colors)
        
fig, axes = plt.subplots(nrows=4, ncols=1, constrained_layout=True,figsize=[7,10])
im_HID = axes[0].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)
axes[0].set_xlim([xlims_mins_input[0], xlims_xlims_input[0]])
axes[0].set_ylim([0,15])   
cbar_HID = plt.colorbar(im_HID, ax=axes[0], shrink=1.1, label=r'HID')    
cbar_HID = adjust_fhc_colorbar_for_pyart(cbar_HID)	
axes[0].set_title('HID Transect of interest') 


im_HID = axes[1].pcolormesh(grid_range/1e3, grid_alt/1e3, grid_HID, cmap=cmaphid, vmin=0.2, vmax=10)
axes[1].set_xlim([xlims_mins_input[0], xlims_xlims_input[0]])
axes[1].set_ylim([0,15])   
axes[1].set_yticks([0,5,10,15])
    
    
    
    
    


