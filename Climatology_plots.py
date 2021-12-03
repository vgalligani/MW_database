#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:29:19 2021

@author: victoria.galligani
"""


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from numpy import genfromtxt;
import numpy as np
from scipy import stats 
from collections import defaultdict
from os import listdir
from pyhdf import SD
from scipy.interpolate import griddata
import Plots as PlottingGMITools

#################################################################
#                   functions 
#################################################################
def get_categoryPF(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [10, 1, 0.1, 0.01])
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    
    return var, latlat, lonlon, percentiles

#################################################################
def get_categoryPF_hi(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [99.99, 99.9, 99, 90])
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    
    return var, latlat, lonlon, percentiles

#################################################################
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

#################################################################
def read_KuRPF_2018(Kurpf_path, imonth):
    
    # Input imonth is string: e.g. "09" 
    
    Kurpf_data_main = {} 
    # open the hdf file
    hdf  = SD.SD(Kurpf_path + 'pf_2018'+imonth+'_level2.HDF')
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
    
    return Kurpf_data

#################################################################
def read_KuRPF_2019(Kurpf_path, imonth):
    
    # Input imonth is string: e.g. "09" 
    
    Kurpf_data_main = {} 
    # open the hdf file
    hdf  = SD.SD(Kurpf_path + 'pf_2019'+imonth+'_level2.HDF')
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
    
    return Kurpf_data
   
#################################################################
def merge_KuRPF_dicts(Kurpf_path):
    
    # pf_data has groups for each month 
    months   = ['09','10','11','12'] 
    
    Kurpf_9  = read_KuRPF_2018(Kurpf_path, '09')
    Kurpf_10 = read_KuRPF_2018(Kurpf_path, '10')
    Kurpf_11 = read_KuRPF_2018(Kurpf_path, '11')
    Kurpf_12 = read_KuRPF_2018(Kurpf_path, '12')
    Kurpf_1  = read_KuRPF_2019(Kurpf_path, '01')
    Kurpf_2  = read_KuRPF_2019(Kurpf_path, '02')
    Kurpf_3  = read_KuRPF_2019(Kurpf_path, '03')
    Kurpf_4  = read_KuRPF_2019(Kurpf_path, '04')
    
    Kurpf_data = defaultdict(list)
    for d in (Kurpf_9, Kurpf_10, Kurpf_11, Kurpf_12, Kurpf_1, Kurpf_2, Kurpf_3, Kurpf_4): 
        for key, value in d.items():
            Kurpf_data[key].append(value)

    return Kurpf_data

#################################################################
def read_KuRPF(ifile):
        
    Kurpf_data_main = {} 
    # open the hdf file
    hdf  = SD.SD(ifile)
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
    
    return Kurpf_data

#################################################################
def merge_KuRPF_dicts_all(Kurpf_path):

    Kurpf_data = defaultdict(list)

    files = listdir(Kurpf_path)
    for i in files: 
        Kurpf = read_KuRPF(Kurpf_path+i)
        for key, value in Kurpf.items():
            Kurpf_data[key].append(value)

    return Kurpf_data










#################################################################
def get_categoryPF(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [10, 1, 0.1, 0.01])
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    hour   = PF_all['HOUR'][select].copy()

    landocean = PF_all['LANDOCEAN'][select].copy()
    
    return var, latlat, lonlon, percentiles, hour, landocean

#################################################################
def get_categoryPF_hi(PF_all, select, vkey):
    
    var        = PF_all[vkey][select].copy()
    percentiles = np.percentile(var, [90, 99, 99.9, 99.99] )
    
    latlat = PF_all['LAT'][select].copy()
    lonlon = PF_all['LON'][select].copy()
    hour   = PF_all['HOUR'][select].copy()

    landocean = PF_all['LANDOCEAN'][select].copy()
    
    return var, latlat, lonlon, percentiles, hour, landocean

def get_hourly_normbins(PF, select, vkey):

    # e.g. PCT_i_binned[0] is between 00UTC - 01UTC for percentile i 
    # normalize by the total numer of elements. 

    hour_bins = np.linspace(0, 24, 25)
    
    PCT_cat, latlat, lonlon, percentiles, hour, surfFlag = get_categoryPF(PF, select, vkey)
    normelems = np.zeros((len(hour_bins)-1, 4)); normelems[:]=np.nan
    counter = 0

    # NEED TO REMOVE SEA! KEEP ONLY LAND!!
    # ['LANDOCEAN'] --->    0: over ocean, 1: over land
    
    for i in percentiles:
        HOUR    = hour[np.where( (PCT_cat < i) & (surfFlag == 1))]   
        PCT_i   = PCT_cat[np.where( (PCT_cat < i) & (surfFlag == 1))]  
        totalelems_i = len(HOUR)
        PCT_i_binned = [PCT_i[np.where((HOUR > low) & (HOUR <= high))] for low, high in zip(hour_bins[:-1], hour_bins[1:])]
        for ih in range(len(hour_bins)-1):
            normelems[ih, counter] = len(PCT_i_binned[ih].data)/totalelems_i
        counter = counter+1
        del HOUR, PCT_i, totalelems_i, PCT_i_binned  
        
    return normelems, percentiles

def get_hourly_normbins_hi(PF, select, vkey):

    # e.g. PCT_i_binned[0] is between 00UTC - 01UTC for percentile i 
    # normalize by the total numer of elements. 

    hour_bins = np.linspace(0, 24, 25)
    
    PCT_cat, latlat, lonlon, percentiles, hour, surfFlag = get_categoryPF_hi(PF, select, vkey)
    normelems = np.zeros((len(hour_bins)-1, 4)); normelems[:]=np.nan
    counter = 0

    # NEED TO REMOVE SEA! KEEP ONLY LAND!!
    # ['LANDOCEAN'] --->    0: over ocean, 1: over land
    
    for i in percentiles:
        HOUR    = hour[np.where( (PCT_cat < i) & (surfFlag == 1))]   
        PCT_i   = PCT_cat[np.where( (PCT_cat < i) & (surfFlag == 1))]  
        totalelems_i = len(HOUR)
        PCT_i_binned = [PCT_i[np.where((HOUR > low) & (HOUR <= high))] for low, high in zip(hour_bins[:-1], hour_bins[1:])]
        for ih in range(len(hour_bins)-1):
            if totalelems_i > 0:
                normelems[ih, counter] = len(PCT_i_binned[ih].data)/totalelems_i
        counter = counter+1
        del HOUR, PCT_i, totalelems_i, PCT_i_binned  
        
    return normelems, percentiles



def get_orbits_extreme(Kurpf, selectKurpf, vkey):
    
    var         = Kurpf[vkey][selectKurpf].copy()
    percentiles = np.percentile(var, [10, 1, 0.1, 0.01])
    
    latlat = Kurpf['LAT'][selectKurpf].copy()
    lonlon = Kurpf['LON'][selectKurpf].copy()
    
    orbit  = Kurpf['ORBIT'][selectKurpf].copy()
    month  = Kurpf['MONTH'][selectKurpf].copy()
    day    = Kurpf['DAY'][selectKurpf].copy()
    hour   = Kurpf['HOUR'][selectKurpf].copy()
    
    LON = lonlon[np.where(var < percentiles[3])]   
    LAT = latlat[np.where(var < percentiles[3])]

    ORB = orbit[np.where(var < percentiles[3])]
    MM  = month[np.where(var < percentiles[3])]
    DD  = day[np.where(var < percentiles[3])]
    HH  = hour[np.where(var < percentiles[3])]
    
    info = []
    for i in range(len(ORB)):
        info.append('orbit Nr: '+str(ORB[i])+' '+str(DD[i])+'/'+str(MM[i])+'/2018 H'+str(HH[i])+' UTC')    
        # # Plot map? 
        # fig = plt.figure(figsize=(12,12))     
        # gs1 = gridspec.GridSpec(1, 1)
        # ax1 = plt.subplot(gs1[0,0])
        # plt.scatter(LON, LAT, c=tb_s1_gmi[inside,7], s=10, cmap=PlottingGMITools.cmaps['turbo_r']) 
        # ax1.set_xlim([-80,-45])
        # ax1.set_ylim([-45,-15])
        # plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
        # plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
        # plt.title('Location of PF centers after domain-location filter')
        # plt.text(-55, -17, 'Nr. PFs:'+str(Kurpf['LAT'][selectKurpf].shape[0]))
    
    
    return(info)    

def get_orbits_extreme_hi(Kurpf, selectKurpf, vkey):
    
    var         = Kurpf[vkey][selectKurpf].copy()
    percentiles = np.percentile(var, [90, 99, 99.9, 99.99] )
    
    latlat = Kurpf['LAT'][selectKurpf].copy()
    lonlon = Kurpf['LON'][selectKurpf].copy()
    
    orbit  = Kurpf['ORBIT'][selectKurpf].copy()
    month  = Kurpf['MONTH'][selectKurpf].copy()
    day    = Kurpf['DAY'][selectKurpf].copy()
    hour   = Kurpf['HOUR'][selectKurpf].copy()
    
    LON = lonlon[np.where(var > percentiles[3])]   
    LAT = latlat[np.where(var > percentiles[3])]

    ORB = orbit[np.where(var > percentiles[3])]
    MM  = month[np.where(var > percentiles[3])]
    DD  = day[np.where(var > percentiles[3])]
    HH  = hour[np.where(var > percentiles[3])]
    
    info = []
    for i in range(len(ORB)):
        info.append('orbit Nr: '+str(ORB[i])+' '+str(DD[i])+'/'+str(MM[i])+'/2018 H'+str(HH[i])+' UTC')
    
    return(info) 



def plot_PCT_percentiles_GMI(dir, filename, Kurpf, selectKurpf):

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 10)
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(2, 2)
    #------ MIN37PCT
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MIN37PCT intensity category')
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MIN37PCT_cat < i)]   
        LAT = latlat[np.where(MIN37PCT_cat < i)]
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ MIN85PCT
    ax1 = plt.subplot(gs1[0,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MIN85PCT intensity category')
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MIN85PCT_cat < i)]   
        LAT = latlat[np.where(MIN85PCT_cat < i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ MIN165V
    ax1 = plt.subplot(gs1[1,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MIN165V intensity category')
    MIN165V_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN165V')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MIN165V_cat < i)]   
        LAT = latlat[np.where(MIN165V_cat < i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    p1 = ax1.get_position().get_points().flatten()
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ MIN165V
    ax1 = plt.subplot(gs1[1,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MIN1838 intensity category')
    MIN1838_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN1838')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MIN1838_cat < i)]   
        LAT = latlat[np.where(MIN1838_cat < i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #-colorbar
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['10', '1', '0.1', '0.01']
    loc = np.arange(0, 4 + 1, 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)

    fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()
    
    return fig 


def plot_PCT_percentiles_Ku(dir, filename, Kurpf, selectKurpf):

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 10)
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(2, 2)
    #------ MAXHT20
    ax1 = plt.subplot(gs1[0,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MAXHT20 intensity category')
    MAXHT20_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT20')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MAXHT20_cat > i)]   
        LAT = latlat[np.where(MAXHT20_cat > i)]
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ MAXHT30
    ax1 = plt.subplot(gs1[0,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MAXHT40 intensity category')
    MAXHT40_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MAXHT40_cat > i)]   
        LAT = latlat[np.where(MAXHT40_cat > i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ VOLRAIN_KU
    ax1 = plt.subplot(gs1[1,0])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF VOLRAIN_KU intensity category')
    VOLRAIN_KU_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'VOLRAIN_KU')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(VOLRAIN_KU_cat > i)]   
        LAT = latlat[np.where(VOLRAIN_KU_cat > i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))      
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    p1 = ax1.get_position().get_points().flatten()
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #------ MAXNSZ
    ax1 = plt.subplot(gs1[1,1])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.title('PF MAXNSZ intensity category')
    MAXNSZ_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXNSZ')
    counter = 0
    for i in percentiles:
        LON  = lonlon[np.where(MAXNSZ_cat > i)]   
        LAT = latlat[np.where(MAXNSZ_cat > i)]   
        if counter < 1:
            plt.scatter(LON, LAT, s=15, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(LON, LAT, s=30, marker='o', c = cmap_f(counter))
        counter = counter+1
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax1.set_xlim([-80,-45])
    ax1.set_ylim([-45,-15])
    p2 = ax1.get_position().get_points().flatten()
    # 
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(4)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, 4)
    img = plt.imshow(np.array([[0,1]]), vmin=0, vmax=4, cmap=cmap_f)
    img.set_visible(False)
    
    #-colorbar
    ax_cbar = fig.add_axes([p1[0], 0.05, p2[2]-p1[0], 0.02])
    cbar = fig.colorbar(img, cax=ax_cbar, ticks=[0, 1, 2, 3, 4], 
                        orientation="horizontal")
    labels = ['90', '99', '99.9', '99.99']
    loc = np.arange(0, 4 + 1, 1) + .5
    cbar.set_ticks(loc)
    cbar.ax.set_xticklabels(labels)

    fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()
    
    return fig 

def plot_MIN1838_distrib(dir, filename, Kurpf, selectKurpf):

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 10)
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MIN37PCT
    ax1 = plt.subplot(gs1[0,0])
    plt.title('PF MIN1838 intensity distribution')
    MIN37PCT_cat, _, _, _, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    MIN85PCT_cat, _, _, _, _, _= get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    MIN1838_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN1838')
    counter = 0
    for i in percentiles:
        x_min37  = MIN37PCT_cat[np.where(MIN1838_cat < i)]   
        y_min85  = MIN85PCT_cat[np.where(MIN1838_cat < i)]
        if counter < 1:
            plt.scatter(x_min37, y_min85, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min37, y_min85, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min37, y_min85, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel('MIN37PCT (K)')
    plt.ylabel('MIN85PCT (K)')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class < 10%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class < 1%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class < 0.1%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class < 0.11%')        
    plt.legend()    
    plt.grid()
    
    fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()
    
    return


def plot_MAXHT40_distrib(dir, filename, Kurpf, selectKurpf):

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 10)
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ MAXHT40
    ax1 = plt.subplot(gs1[0,0])
    plt.title('PF MAXHT40 intensity distribution')
    MIN37PCT_cat, _, _, _, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    MAXNSZ_cat, _, _, _, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXNSZ')
    MAXHT40_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    counter = 0
    for i in percentiles:
        x_min37   = MIN37PCT_cat[np.where(MAXHT40_cat > i)]   
        y_MAXNSZ  = MAXNSZ_cat[np.where(MAXHT40_cat > i)]
        if counter < 1:
            plt.scatter(x_min37, y_MAXNSZ, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min37, y_MAXNSZ, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min37, y_MAXNSZ, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel('MIN37PCT (K)')
    plt.ylabel('MAXNSZ (dBZ)')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class > 90%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class > 99%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class > 99.9%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class > 99.99%')        
    plt.legend()    
    plt.grid()
    
    fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()
    
    return


def plot_volrain_Ku_distrib(dir, filename, Kurpf, selectKurpf):

    import seaborn as sns

    # Some matplotlib figure definitions
    plt.matplotlib.rc('font', family='serif', size = 10)
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    # replace highest temperatures with gray
    cmap1 =  plt.cm.get_cmap('tab20c')
    cmap    = sns.color_palette("tab10", as_cmap=True)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap1(18)
    cmap_f = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
    
    #------------------------- Figure 
    fig = plt.figure(figsize=(12,12))     
    gs1 = gridspec.GridSpec(1, 1)
    #------ VOLRAIN_KU
    ax1 = plt.subplot(gs1[0,0])
    plt.title('PF VOLRAIN_KU intensity distribution')
    MIN85PCT_cat, _, _, _, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    VOLRAIN_KU_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'VOLRAIN_KU')
    #precipitation area IS estimated by the number of pixels associated with each PF.
    NPIXELS_cat, _, _, _, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'NPIXELS')
    npixels = NPIXELS_cat.copy()
    npixels = npixels.astype(np.float32)
    area    = npixels*5.*5.
    
    counter = 0
    for i in percentiles:
        x_min85   = MIN85PCT_cat[np.where(VOLRAIN_KU_cat > i)]   
        y_area    = area[np.where(VOLRAIN_KU_cat > i)]
        if counter < 1:
            plt.scatter(x_min85, y_area, s=15, marker='o', c = cmap_f(counter))
        elif counter < 3:
            plt.scatter(x_min85, y_area, s=30, marker='o', c = cmap_f(counter))
        else:
            plt.scatter(x_min85, y_area, s=50, marker='o', c = cmap_f(counter))        
        counter = counter+1
    plt.xlabel('MIN85PCT (K)')
    plt.ylabel(r'PFs area (km$^2$)')
    ax1.set_yscale('log')
    plt.scatter(np.nan, np.nan, s=15, marker='o', c = cmap_f(0), label='class > 90%')        
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(1), label='class > 99%')             
    plt.scatter(np.nan, np.nan, s=30, marker='o', c = cmap_f(2), label='class > 99.9%')        
    plt.scatter(np.nan, np.nan, s=50, marker='o', c = cmap_f(3), label='class > 99.99%')        
    plt.legend()    
    plt.grid()
    
    fig.savefig(dir+filename, dpi=300,transparent=False)        
    #plt.close()
    
    return

def plot_regrid_map(lonbins, latbins, zi_37, zi_85, zi_max40ht, filename, main_title):

    
    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    pc = ax1.pcolor(lonbins, latbins, zi_37, vmin=50, vmax=300)
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    pc = ax1.pcolor(lonbins, latbins, zi_85, vmin=50, vmax=300)
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    pc = ax1.pcolor(lonbins, latbins, zi_max40ht, vmin=0, vmax=20)
    plt.colorbar(pc)    
    ax1.set_title('MAXHT40 > 1% percentile')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    plt.suptitle(main_title)
    
    fig.savefig(filename, dpi=300,transparent=False)        

    return

def plot_spatial_distrib(Kurpf, selectKurpf, filename, main_title):

    lonbins = np.arange(-80, -40, 2) 
    latbins = np.arange(-50, -10, 2)
    xbins, ybins = len(lonbins), len(latbins) #number of bins in each dimension
    
    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
    PCT37       = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    LON         = lonlon[np.where(MIN85PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN85PCT_cat < percentiles[1])]
    PCT85       = MIN85PCT_cat[np.where(MIN85PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT85, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)    
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    LON         = lonlon[np.where(max40ht_cat > percentiles[1])]   
    LAT         = latlat[np.where(max40ht_cat > percentiles[1])]
    PCT_max40ht = max40ht_cat[np.where(max40ht_cat > percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T)        
    cbar = plt.colorbar(pc)    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 1% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar.set_label('# of elements in 2x2 grid')

    plt.suptitle(main_title)

    
    fig.savefig(filename, dpi=300,transparent=False)        

    return

def plot_norm_spatial_distrib(Kurpf, selectKurpf, filename, main_title):
    
    # normalized by dividing each bin by the total pixel number
    Ntot =  Kurpf['MIN37PCT'][selectKurpf].shape[0]   
    
    lonbins = np.arange(-80, -40, 2) 
    latbins = np.arange(-50, -10, 2)
    xbins, ybins = len(lonbins), len(latbins) #number of bins in each dimension
    
    prov = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/provincias.txt", delimiter='')
    samerica = genfromtxt("/Users/victoria.galligani/Work/Tools/Maps/samerica.txt", delimiter='')

    fig = plt.figure(figsize=(12,12))  
    gs1 = gridspec.GridSpec(2, 2)   
    
    ax1 = plt.subplot(gs1[0,0])
    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
    PCT37       = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT37, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T/Ntot*100)
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)        
    ax1.set_title('MIN37PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
      
    ax1 = plt.subplot(gs1[0,1])
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    LON         = lonlon[np.where(MIN85PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN85PCT_cat < percentiles[1])]
    PCT85       = MIN85PCT_cat[np.where(MIN85PCT_cat < percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT85, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T/Ntot*100)    
    #plt.plot(LON,LAT, 'xr')
    plt.colorbar(pc)    
    ax1.set_title('MIN85PCT < 1% percentile ('+str(np.round(percentiles[1], 2))+' K)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    
    ax1 = plt.subplot(gs1[1,0])
    max40ht_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    LON         = lonlon[np.where(max40ht_cat > percentiles[1])]   
    LAT         = latlat[np.where(max40ht_cat > percentiles[1])]
    PCT_max40ht = max40ht_cat[np.where(max40ht_cat > percentiles[1])]    
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(LON, LAT, values = PCT_max40ht, statistic='count', bins = [xbins, ybins])  
    H = np.ma.masked_where(H==0, H) #masking where there was no data
    XX, YY = np.meshgrid(xedges, yedges)
    pc = ax1.pcolormesh(XX,YY,H.T/Ntot*100)        
    cbar = plt.colorbar(pc)    
    #plt.plot(LON,LAT, 'xr')
    ax1.set_title('MAXHT40 > 1% percentile ('+str(np.round(percentiles[1], 2))+' km)')
    ax1.set_xlim([-75,-50])
    ax1.set_ylim([-40,-19])   
    plt.plot(prov[:,0],prov[:,1],color='k', linewidth=0.5);   
    plt.plot(samerica[:,0],samerica[:,1],color='k', linewidth=0.5);   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar.set_label('Relative Frequency in 2x2 grid')

    plt.suptitle(main_title)

    
    fig.savefig(filename, dpi=300,transparent=False)        

    return

def plot_seasonal_distributions(Kurpf, selectKurpf, season, dir_name):

    from scipy.interpolate import griddata
    lonbins = np.arange(-80, -40, 2) 
    latbins = np.arange(-50, -10, 2)
    xi,yi   = np.meshgrid(lonbins,latbins)
    

    MIN37PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN37PCT')
    LON         = lonlon[np.where(MIN37PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN37PCT_cat < percentiles[1])]
    MIN37PCT    = MIN37PCT_cat[np.where(MIN37PCT_cat < percentiles[1])]
    zi_MIN37PCT = griddata((LON,LAT), MIN37PCT, (xi,yi), method='nearest')
        
    MIN85PCT_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF(Kurpf, selectKurpf, 'MIN85PCT')
    LON         = lonlon[np.where(MIN85PCT_cat < percentiles[1])]   
    LAT         = latlat[np.where(MIN85PCT_cat < percentiles[1])]
    MIN85PCT    = MIN85PCT_cat[np.where(MIN85PCT_cat < percentiles[1])]
    zi_MIN85PCT = griddata((LON,LAT), MIN85PCT, (xi,yi), method='nearest')
        
    MAXHT40_cat, latlat, lonlon, percentiles, _, _ = get_categoryPF_hi(Kurpf, selectKurpf, 'MAXHT40')
    LON         = lonlon[np.where(MAXHT40_cat > percentiles[1])]   
    LAT         = latlat[np.where(MAXHT40_cat > percentiles[1])]
    MAXHT40     = MAXHT40_cat[np.where(MAXHT40_cat > percentiles[1])]
    zi_MAXHT40 = griddata((LON,LAT), MAXHT40, (xi,yi), method='nearest')
    
    
    if season == 'dry_season':
        main_title = 'dry season (MJJA)'
    elif season == 'wet_season':
        main_title = 'wet season (ONDJFM)'
        
    # plot the actual values
    filename = season+'regridded_map.png'
    plot_regrid_map(lonbins, latbins, zi_MIN37PCT, zi_MIN85PCT, zi_MAXHT40, dir_name+filename, main_title)

    # plot the spatial frequency distrb. note that the number of PFs within each grid box is different when
    #  different proxies are considered, since different subsets of elements are selected from the entire dataset. 
    # GPM sampling maximizes near 60◦S due to the orbital recurvature at that latitude, 
    filename = season+'spatial_distrib.png'
    plot_spatial_distrib(Kurpf,selectKurpf, dir_name+filename, main_title)

    # normalized by dividing each bin by the total pixel number
    filename = season+'norm_spatial_distrib.png'
    plot_norm_spatial_distrib(Kurpf,selectKurpf, dir_name+filename, main_title)    

    return


